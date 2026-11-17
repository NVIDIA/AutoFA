# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handled discovery stops finalize once; hard cancellation remains an exit."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import Counter
from io import StringIO

import pytest
from rich.console import Console

from autofa.core.library import FeatureFunction
from autofa.harness import claude, codex
from autofa.harness.claude import TurnResult
from autofa.mcp import tools_eval
from autofa.metrics import claude_judge
from autofa.reports.failure_cards import build_data
from autofa.session import finalization
from autofa.ui.evaluation_progress import current_evaluation_progress, evaluation_progress
from autofa.util.cancellation import check_cancelled

from support.evaluation import _LLM_SOURCE, _REGEX_SOURCE, _record_calls, _runtime, reply, runtime
from support.task_support import _render_card


@pytest.fixture(autouse=True)
def no_paid_final_reviews(monkeypatch):
    from autofa.metrics.final_review import final_review_functions

    def review(*, progress, names=None):
        progress({"total": 0, "completed": 0, "detector": ""})
        return {"ok": True, "reviewed": [], "unresolved": [], "n": 0}

    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", review)
    return final_review_functions


@pytest.fixture
def empty_finalization_runtime(tmp_path, monkeypatch):
    return _runtime(tmp_path, monkeypatch, sample=40, workers=1, functions=[])


@pytest.fixture(params=[claude, codex], ids=["claude", "codex"])
def interactive(request, monkeypatch, tmp_path):
    module = request.param
    cls = module.ClaudeHarness if module is claude else module.CodexHarness
    harness = object.__new__(cls)
    harness.model, harness.cli_path, harness.effort = "test", None, None
    monkeypatch.setattr("prompt_toolkit.PromptSession", lambda **kwargs: object())
    monkeypatch.setattr("autofa.config.autofa_home", lambda: tmp_path)
    monkeypatch.setattr("autofa.ui.menus.prepare_step", lambda: None)
    monkeypatch.setattr(claude, "describe_auth_mode", lambda *args: "test")
    monkeypatch.setattr(module, "console", Console(file=StringIO()))
    commands = ["/exit"]

    async def prompt(session):
        return commands.pop(0)

    monkeypatch.setattr(module, "_async_prompt", prompt)
    return harness, commands


@pytest.mark.parametrize("outcome", [
    "complete", "stopped", "error", "stopped_error", "cancel", "interrupted_finalization",
])
def test_discovery_boundary_finalizes_once_before_returning_control(
    interactive, monkeypatch, tmp_path, outcome,
):
    harness, commands = interactive
    events = []

    async def turn(prompt, **kwargs):
        events.append("turn")
        if outcome == "cancel":
            raise asyncio.CancelledError
        return TurnResult(session_id="saved",
                          stopped=outcome.startswith(("stopped", "interrupted")),
                          error="Connection lost" if "error" in outcome else None)

    def continuation():
        events.append("continuation")
        return None

    async def finalize():
        assert commands == ["/exit"]  # Scoring precedes the next interactive command.
        events.append("finalize")
        if outcome == "interrupted_finalization":
            raise asyncio.CancelledError

    monkeypatch.setattr(harness, "run_turn", turn)
    session = harness.run_session(
        cwd=str(tmp_path), system_prompt="test", initial_prompt="discover",
        discovery_continuation=continuation, discovery_finalization=finalize,
    )
    if outcome in ("cancel", "interrupted_finalization"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(session)
        assert commands == ["/exit"]  # Cancellation exits without restarting discovery.
    else:
        assert asyncio.run(session) == "saved"
    assert events == {
        "complete": ["turn", "continuation", "finalize"],
        "stopped": ["turn", "finalize"],
        "interrupted_finalization": ["turn", "finalize"],
    }.get(outcome, ["turn"])


def test_final_scoring_cancellation_reaches_worker_and_preserves_original_cancel(monkeypatch):
    started = threading.Event()
    cancelled = threading.Event()

    def score(console):
        from autofa.util.cancellation import _CURRENT

        token = _CURRENT.get()
        assert token is not None
        started.set()
        assert token.event.wait(3)
        cancelled.set()
        check_cancelled()
        pytest.fail("A cancelled final scan must not continue scoring")

    monkeypatch.setattr(finalization, "evaluate_final_library", score)

    async def invoke():
        task = asyncio.create_task(finalization.finalize_discovery(Console(file=StringIO())))
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert cancelled.is_set()

    asyncio.run(invoke())


def test_second_cancellation_does_not_wait_for_or_restart_final_scoring(monkeypatch):
    started, cancelled, release = threading.Event(), threading.Event(), threading.Event()
    calls = []
    original = finalization.Cancellation

    class ObservedCancellation(original):
        def cancel(self):
            super().cancel()
            cancelled.set()

    def score(console):
        calls.append(True)
        started.set()
        assert release.wait(3)
        check_cancelled()

    monkeypatch.setattr(finalization, "Cancellation", ObservedCancellation)
    monkeypatch.setattr(finalization, "evaluate_final_library", score)

    async def invoke():
        task = asyncio.create_task(finalization.finalize_discovery(Console(file=StringIO())))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            assert await asyncio.to_thread(cancelled.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            assert not release.is_set() and calls == [True]
        finally:
            release.set()

    asyncio.run(invoke())


@pytest.mark.parametrize("failure", ["partial", "cancel"])
@pytest.mark.usefixtures("empty_finalization_runtime")
def test_failed_final_scoring_closes_progress_scope_and_live_display(monkeypatch, failure):
    monkeypatch.setattr(finalization, "_prepare_final_selection",
                        lambda console: {"selected_names": ["pending_review"], "unresolved": []})
    output = StringIO()
    console = Console(file=output, width=160)

    def score(*, names):
        reporter = current_evaluation_progress()
        reporter.start(["unfinished"], 10)
        reporter.advance("unfinished")
        if failure == "cancel":
            raise asyncio.CancelledError
        return {"ok": False, "error": "candidate blocked"}

    monkeypatch.setattr("autofa.mcp.tools_eval.final_evaluate_functions", score)
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
        finalization._evaluate_final_library(console)
    assert current_evaluation_progress() is None
    assert "Final detector evaluation complete" not in output.getvalue()
    assert not console._live_stack


def test_final_progress_counts_completed_python_llm_and_cached_predictions_once(
    tmp_path, monkeypatch,
):
    python = FeatureFunction(name="python", source=_REGEX_SOURCE,
                             description="Exact", kind="python")
    llm = FeatureFunction(name="llm", source=_LLM_SOURCE, description="Semantic", kind="llm")
    rt = _runtime(tmp_path, monkeypatch, sample=12, workers=2, functions=[python, llm], n=60)
    calls = _record_calls(monkeypatch, lambda src, _: "llm" if src == _LLM_SOURCE else "python")
    assert tools_eval.evaluate_functions(names=[llm.name], limit=24)["ok"]
    assert Counter(calls) == {"llm": 12}
    events = []
    with evaluation_progress(events.append):
        result = tools_eval.final_evaluate_functions()
    assert result["ok"]
    assert events[0]["total"] == events[-1]["completed"] == 120
    assert Counter(calls) == {"llm": 60, "python": 60}  # Includes the 12 earlier cached calls.
    assert all(fn.scores["n_scored"] == 60 for fn in rt.library.functions)


@pytest.mark.usefixtures("empty_finalization_runtime")
def test_unavailable_final_review_is_reported_without_claiming_complete(monkeypatch):
    monkeypatch.setattr(finalization, "_prepare_final_selection",
                        lambda console: {"selected_names": ["pending_review"], "unresolved": []})
    output = StringIO()

    def review(*, progress, names=None):
        progress({"total": 1, "completed": 1, "detector": "answer_loop"})
        return {"ok": False, "reviewed": ["answer_loop"], "unresolved": [
            {"name": "answer_loop", "reason": "Retained image is unavailable."},
        ]}

    monkeypatch.setattr("autofa.mcp.tools_eval.final_evaluate_functions",
                        lambda **kwargs: {"ok": True, "n": 0})
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", review)
    finalization._evaluate_final_library(Console(file=output, width=180))
    text = output.getvalue()
    assert "answer_loop: Retained image is unavailable." in text
    assert "Final scores saved with unresolved" in text
    assert "Final validation complete for selected detectors" not in text


def test_finalization_refreshes_before_recovered_admission_and_candidate_selection(
    empty_finalization_runtime, monkeypatch,
):
    rt = empty_finalization_runtime
    events = []
    output = StringIO()

    def refresh(runtime, *, progress):
        assert runtime is rt and not events
        assert callable(progress)
        events.append("refresh")
        return {"refreshed_names": []}

    def recovered_admission(runtime):
        assert runtime is rt and events == ["refresh"]
        events.append("recovered_admission")

    def select(runtime, *, progress):
        assert runtime is rt and events == ["refresh", "recovered_admission"]
        events.append("selection")
        return {"selected_names": ["resumed"], "skipped": [], "unresolved": []}

    def score(*, names):
        assert names == ["resumed"] and events[-1] == "selection"
        events.append("scan")
        return {"ok": True, "n": 0}

    def review(*, names, progress):
        assert names == ["resumed"] and events[-1] == "scan"
        assert "Final validation complete for selected detectors" not in output.getvalue()
        events.append("review")
        progress({"total": 1, "completed": 0, "detector": "resumed"})
        progress({"total": 1, "completed": 1, "detector": "resumed"})
        return {"ok": True, "reviewed": names, "unresolved": []}

    monkeypatch.setattr("autofa.session.evaluation_refresh.refresh_evaluation_contexts", refresh)
    monkeypatch.setattr("autofa.session.validation_recovery.prepare_recovered_validation",
                        recovered_admission)
    monkeypatch.setattr("autofa.metrics.final_validation_selection.prepare_final_validation",
                        select)
    monkeypatch.setattr("autofa.mcp.tools_eval.final_evaluate_functions", score)
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", review)
    finalization._evaluate_final_library(Console(file=output, width=180))
    assert events == ["refresh", "recovered_admission", "selection", "scan", "review"]
    assert "Final validation complete for selected detectors" in output.getvalue()


def test_finalization_persists_recovered_scores_and_renders_them_in_html(
    tmp_path, monkeypatch, stub_claim_compiler, no_paid_final_reviews,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    assert fn.scores.get("behavior_purity") is None
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions",
                        no_paid_final_reviews)
    monkeypatch.setattr(claude_judge, "_run_review", lambda **kw: reply(kw["review"]))
    finalization.evaluate_final_library(Console(file=StringIO()))

    saved = json.loads((rt.run.run_dir() / "library.json").read_text())
    scores = saved["functions"][0]["scores"]
    assert scores["evaluation_scope_complete"]
    assert scores["behavior_n_completed"] == 3
    assert scores["behavior_purity"] == 0  # The offline judge supplied actual rejections.
    assert scores["selection_review"]["status"] == "complete"
    assert scores["final_validation_selection"]["status"] == "skipped"
    data = build_data(rt.library)
    card = data["functions"][0]
    assert card["metrics"]["behavior_purity"] == 0
    assert card["verdict"]["label"] == "WEAK"
    html = _render_card(data)
    assert "Review pending" not in html and "NOT EVALUATED" not in html


def test_finalization_freezes_library_across_selection_scoring_and_review(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from autofa.session.operation_admission import admit_operation

    _runtime(tmp_path, monkeypatch, sample=40, workers=1, functions=[])

    def concurrent_mutation():
        with admit_operation("clear_library", {}) as busy:
            return busy

    def finalize(console):
        # Nested review shares ownership; another caller cannot change selection.
        with admit_operation("judge_function", {"name": "selected"}) as busy:
            assert busy is None
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(concurrent_mutation).result()["busy"]

    monkeypatch.setattr(finalization, "_evaluate_final_library", finalize)
    finalization.evaluate_final_library(Console(file=StringIO()))
    assert concurrent_mutation() is None
