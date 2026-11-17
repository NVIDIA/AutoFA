# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash/disconnect/transport recovery preserves completed work and judging boundaries."""

import asyncio
import json
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import claude_agent_sdk as sdk
import pytest

from autofa.config import Config
from autofa.core.library import FeatureFunction
from autofa.harness.claude import ClaudeHarness, TurnResult
from autofa.harness.codex import CodexHarness
from autofa.harness.codex_app_server import CodexTurnResult, _error_details
from autofa.mcp import runtime, server, tools_eval
from autofa.metrics import claude_judge
from autofa.metrics.judge_checkpoints import JudgeInterruptedError
from autofa.session.recovery import (
    RunInterruptedError,
    drain_discovery_tools,
    execute_operation,
    read_json,
    resume_prompt,
    run_lock,
    save_turn,
)
from autofa.session.status_log import latest_turn_status
from autofa.ui.activity import AgentActivity
from autofa.util.cancellation import _CURRENT, cancellation_guard

from support.evaluation import (
    _llm_flag,
    _record_calls,
    _runtime,
    _scoring_runtime,
    record_reviews,
    reply,
)


def judge(tmp_path, **changes):
    args = dict(name="claim", description="The agent guesses.", mechanism="It acts on a guess.",
                source="def detect(item): return True", model="test", max_chars=0,
                items=[{"_item_id": key, "response_text": key} for key in ("one", "two")],
                assigned_labels=[True, True], cwd=tmp_path)
    args.update(changes)
    return claude_judge.judge_behavior_with_claude(**args)


def test_killed_judge_reuses_completed_items_and_compilation(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    calls = record_reviews(monkeypatch, interrupt_at=2)
    # This failpoint deliberately interrupts after the first checkpoint is saved.
    with pytest.raises(KeyboardInterrupt):
        judge(tmp_path, workers=1)
    saved = next((tmp_path / "behavior_judge").glob("*/0/review-verdict.json"))
    original = saved.read_bytes()
    result = judge(tmp_path, workers=1)
    assert calls == ["one", "two", "two"]
    assert len(stub_claim_compiler) == 1
    assert result["n_no"] == 2 and result["reused_stages"] == 1
    assert saved.read_bytes() == original
    assert result["sessions"][0]["reused"] is True


def test_confirmation_interrupted_resumes_independently_without_repeating_primary(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    calls = []

    def run(*, review, prompt, system_prompt=None, **kwargs):
        calls.append(system_prompt)
        if len(calls) == 2:
            raise ConnectionResetError("connection reset")
        if len(calls) > 2:
            assert system_prompt == claude_judge.VERIFICATION_SYSTEM
            assert not review.reads and not review.citations
            assert "Reviewed evidence" not in prompt
        return reply(review, "supported")

    monkeypatch.setattr(claude_judge, "_run_review", run)
    args = dict(items=[{"_item_id": "one", "response_text": "one"}], assigned_labels=[True])
    with pytest.raises(JudgeInterruptedError):
        judge(tmp_path, **args)
    assert not list((tmp_path / "behavior_judge").glob("*/0/verdict.json"))
    result = judge(tmp_path, **args)
    assert len(calls) == 3 and result["n_yes"] == 1
    assert result["reused_stages"] == 1
    assert result["verdicts"][0]["verification_status"] == "confirmed"
    # A third call revalidates both completed independent stages without a model call.
    again = judge(tmp_path, **args)
    assert len(calls) == 3 and again["reused_stages"] == 2 and again["n_yes"] == 1


@pytest.mark.parametrize("changed", [
    {"description": "Another claim"}, {"max_chars": 100},
    {"analysis_guidance": "Different instructions"}, {"max_turns": 32},
    # Explicit thinking with effort unset never reuses legacy reviews.
    {"thinking": "disabled"}, {"thinking": "adaptive"},
    {"items": [{"_item_id": "one", "response_text": "changed evidence"},
               {"_item_id": "two", "response_text": "changed evidence"}]},
    # Labels key each review separately from the rubric and evidence snapshot.
    {"assigned_labels": [False, False]},
])
def test_changed_inputs_invalidate_saved_reviews(
    tmp_path, monkeypatch, stub_claim_compiler, changed,
):
    calls = record_reviews(monkeypatch)
    judge(tmp_path)
    assert judge(tmp_path, **changed)["reused_stages"] == 0
    assert len(calls) == 4


@pytest.mark.parametrize("changed", [
    {"effort": "high"}, {"model": "another-model"}, {"thinking": "default"},
])
def test_explicit_judge_settings_are_saved_and_only_reuse_matching_checkpoints(
    tmp_path, monkeypatch, stub_claim_compiler, changed,
):
    calls = []

    def run(*, review, **kwargs):
        calls.append((kwargs["model"], kwargs["effort"]))
        return reply(review)

    monkeypatch.setattr(claude_judge, "_run_review", run)
    first = judge(tmp_path, effort="medium", workers=2)
    resumed = judge(tmp_path, effort="medium", workers=1)
    assert resumed["reused_stages"] == 2 and len(calls) == 2
    assert resumed["judge_rubric_hash"] == first["judge_rubric_hash"]
    settings = {"judge_model": "test", "judge_effort": "medium", "judge_thinking": "adaptive"}
    assert {key: first[key] for key in settings} == settings
    saved = json.loads((tmp_path / "behavior_judge/claim.verdict.json").read_text())
    assert {key: saved[key] for key in settings} == settings
    assert stub_claim_compiler[0]["effort"] == "medium"

    updated = judge(tmp_path, **{"effort": "medium", **changed})
    assert updated["reused_stages"] == 0 and len(calls) == 4
    assert updated["judge_rubric_hash"] != first["judge_rubric_hash"]
    assert len(stub_claim_compiler) == 2
    # Both configurations remain available, so returning to the first reuses it.
    assert judge(tmp_path, effort="medium")["reused_stages"] == 2
    assert len(calls) == 4


def test_corrupt_archive_is_not_reused(tmp_path, monkeypatch, stub_claim_compiler):
    calls = record_reviews(monkeypatch)
    first = judge(tmp_path)
    (Path(first["verdicts"][0]["review_path"]) / "audit.jsonl").write_text("truncated")
    second = judge(tmp_path)
    # The initial independent reviews may start in either worker order.
    assert sorted(calls[:2]) == ["one", "two"]
    assert calls[2:] == ["one"]
    assert second["reused_stages"] == 1 and second["n_no"] == 2


def test_changed_thinking_budget_invalidates_reviews_and_compilation(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    calls = []

    def run(*, review, **kwargs):
        calls.append(kwargs["thinking_budget_tokens"])
        return reply(review)

    monkeypatch.setattr(claude_judge, "_run_review", run)
    first = judge(tmp_path, thinking="enabled", thinking_budget_tokens=2048)
    repeated = judge(tmp_path, thinking="enabled", thinking_budget_tokens=2048)
    assert repeated["reused_stages"] == 2 and calls == [2048, 2048]
    assert first["judge_thinking_budget_tokens"] == 2048
    changed = judge(tmp_path, thinking="enabled", thinking_budget_tokens=4096)
    assert changed["reused_stages"] == 0 and calls == [2048, 2048, 4096, 4096]
    assert changed["judge_rubric_hash"] != first["judge_rubric_hash"]
    assert [call["thinking_budget_tokens"] for call in stub_claim_compiler] == [2048, 4096]


def test_semantic_unknown_is_cached_but_processing_error_is_retried(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    calls = []

    def run(*, review, **kwargs):
        item_id = review.snapshot["item_id"]
        calls.append(item_id)
        # Fail this item's first review regardless of which worker starts first.
        if item_id == "two" and calls.count(item_id) == 1:
            return "broken"
        return reply(review, "unknown")

    monkeypatch.setattr(claude_judge, "_run_review", run)
    assert judge(tmp_path, workers=4)["n_errors"] == 1
    second = judge(tmp_path, workers=4)
    assert sorted(calls[:2]) == ["one", "two"]
    assert calls[2:] == ["two"]
    assert second["n_uncertain"] == 2 and second["n_errors"] == 0
    assert second["reused_stages"] == 1


def test_transient_sdk_failure_resumes_same_review_with_fresh_citations(tmp_path, monkeypatch):
    from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot

    review = EvidenceReview(make_snapshot({"response_text": "one"}, item_id="one", source="src"),
                            tmp_path / "review")
    calls = []

    async def session(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs["budget"].charge_tool("read_evidence")
            review.read("response_text")
            review.session = {"session_id": "same", "num_turns": 2}
            raise ConnectionResetError("connection reset")
        assert not review.covers("response_text", 0, 3, current=True)
        return "recovered"

    async def sleep(_):
        pass

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", session)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    assert claude_judge._run_review(prompt="original", review=review, max_turns=64) == "recovered"
    assert calls[1]["resume"] == "same" and calls[1]["max_turns"] == 64
    assert calls[1]["budget"] is calls[0]["budget"]
    assert calls[1]["budget"].remaining == 63  # One actual tool call, not two SDK turns.


def test_evaluation_disconnect_preserves_completed_predictions(tmp_path, monkeypatch):
    fn = FeatureFunction(name="expensive", description="claim",
                         source="def detect(item): return llm_client.classify('p', 't')")
    runtime = _scoring_runtime(tmp_path, monkeypatch)
    calls = []

    def detector(source, item, **kwargs):
        calls.append(item["_item_id"])
        if len(calls) == 2:
            raise ConnectionResetError("connection reset")
        kwargs["localisation"]["item_id"] = item["_item_id"]
        return True

    monkeypatch.setattr(tools_eval, "run_detector", detector)
    pairs = [(fn, {"_item_id": key}) for key in ("one", "two")]
    with pytest.raises(RunInterruptedError):
        tools_eval._score_items_concurrently(pairs, workers=1, llm=None)
    localisations = {}
    result = tools_eval._score_items_concurrently(pairs, workers=1, llm=None,
                                                 localisations=localisations)
    assert calls == ["one", "two", "two"]
    assert all(pred is True and error is None for _, _, pred, error in result)
    assert len(localisations[fn.name]) == 2
    runtime.config.feature_model = "changed-model"
    tools_eval._score_items_concurrently(pairs, workers=1, llm=None)
    assert calls[-2:] == ["one", "two"]
    for changes in (
        {"feature_base_url": "http://localhost:8000/v1", "feature_api_key_env": "none"},
        {"feature_request_options": {"max_tokens": 2048}},
        {"feature_api_key_env": "LOCAL_KEY"},
        {"feature_api": "decisions"},
    ):
        before = len(calls)
        for key, value in changes.items():
            setattr(runtime.config, key, value)
        tools_eval._score_items_concurrently(pairs, workers=1, llm=None)
        assert calls[before:] == ["one", "two"]
        tools_eval._score_items_concurrently(pairs, workers=1, llm=None)
        assert len(calls) == before + 2


@pytest.mark.parametrize("tool_name, arguments", [
    ("evaluate_functions", {"names": ["saved-function"]}),
    # Real smoke runs leave the operation records that polishing counts as diagnostics.
    ("smoke_test_feature", {"name": "saved-function", "item_ids": ["one"]}),
])
def test_tool_intent_and_original_request_survive_interruption(
    tmp_path, monkeypatch, tool_name, arguments,
):
    rt = SimpleNamespace(run=SimpleNamespace(run_dir=lambda: tmp_path))
    monkeypatch.setattr(runtime, "get_runtime", lambda: rt)
    save_turn(tmp_path, prompt="Inspect these failures for eight rounds", session_id="sid",
              state="running")

    def interrupted(**args):
        op = read_json(next((tmp_path / "operations").glob("*.json")))
        assert op["state"] == "running" and op["args"] == args
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_operation(interrupted, tool_name, arguments)
    prompt = resume_prompt(tmp_path, [], fallback="New discovery", round_idx=3)
    assert "Inspect these failures for eight rounds" in prompt
    assert "stored round is 3" in prompt and "operations/" in prompt
    save_turn(tmp_path, prompt=prompt, session_id="sid", state="running")
    assert read_json(tmp_path / "turn.json")["prompt"] == "Inspect these failures for eight rounds"
    execute_operation(lambda **args: {"ok": True}, tool_name, arguments)
    resumed = resume_prompt(tmp_path, [], fallback="New discovery", round_idx=3)
    assert "Unfinished tool records (read to recover arguments): []" in resumed


def test_run_lock_blocks_duplicate_session_and_survives_crashed_owner(tmp_path):
    import os
    import subprocess
    import sys

    code = (
        "from pathlib import Path; import os, sys; "
        "from autofa.session.recovery import run_lock; "
        "lock = run_lock(Path(sys.argv[1])); lock.__enter__(); os._exit(0)"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    with run_lock(tmp_path):
        child = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env,
                               capture_output=True, text=True, timeout=10)
        assert child.returncode != 0 and "already active" in child.stderr
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, timeout=10)
    assert child.returncode == 0
    # The file survives os._exit, but the OS lock does not.
    with run_lock(tmp_path):
        pass


def test_changed_judge_policy_invalidates_saved_stages(tmp_path, monkeypatch, stub_claim_compiler):
    calls = record_reviews(monkeypatch)
    judge(tmp_path)
    monkeypatch.setattr(claude_judge, "VERIFICATION_SYSTEM", "Changed policy")
    assert judge(tmp_path)["reused_stages"] == 0
    assert sorted(calls[:2]) == ["one", "two"]
    assert sorted(calls[2:]) == ["one", "two"]


def test_cli_resume_legacy_research_run_uses_native_session(tmp_path, monkeypatch):
    from autofa import cli
    from autofa.config import RunConfig
    from autofa.session.research_status import write_research_status

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig(run_id="research", parquet_paths=[], model_id="model", round=2, max_rounds=99,
                    analysis_outputs=["agent"])
    run.save()
    cfg = Config()
    saved = {"hours": 8, "experiments": 3, "elapsed_s": 1000, "inflight": True,
             "state": "stopped", "reason": "connection_lost"}
    write_research_status(run.run_dir(), saved)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning",
                        lambda *a, **kw: SimpleNamespace(summary=lambda: "test"))
    captured = []
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda run, **kw: captured.append((run, kw)))
    cli._resume(SimpleNamespace(run_id=run.run_id, rounds=4))
    assert "resume_state" not in captured[0][1]
    assert (run.run_dir() / "research_status.json").is_file()
    assert captured[0][1]["continue_mode"] is True
    assert captured[0][1]["max_rounds"] == 4
    # The locked runner owns persistence; CLI routing cannot mutate an active run.
    assert RunConfig.load(run.run_id).max_rounds == 99


REMOTE_COMPACTION_ERROR = (
    "Error running remote compact task: stream disconnected before completion: "
    "Transport error: network error: error decoding response body"
)


@pytest.fixture(params=["claude", "codex"])
def discovery(request, monkeypatch):
    cls = ClaudeHarness if request.param == "claude" else CodexHarness
    harness = cls.__new__(cls)
    harness.model, harness.cli_path, harness.effort = "test-model", None, None
    harness.reasoning_options = {}
    monkeypatch.setattr(AgentActivity, "start", lambda self: None)
    monkeypatch.setattr(AgentActivity, "stop", lambda self, **kwargs: None)
    monkeypatch.setattr(sdk, "get_session_messages", lambda *args, **kwargs: [])
    return harness


@pytest.mark.parametrize("entry", ["automatic", "native_failed"])
def test_failed_compaction_replaces_from_durable_state_once(
    tmp_path, monkeypatch, discovery, entry,
):
    original = "Continue discovery within 8 rounds and the same task scope"
    (tmp_path / "config.json").write_text(json.dumps({
        "round": 3, "max_rounds": 8, "exploration_sequence_index": 17,
    }))
    (tmp_path / "library.json").write_text('{"functions":["already-saved"]}')
    (tmp_path / "operations").mkdir()
    (tmp_path / "operations" / "pending.json").write_text(json.dumps({
        "tool": "advance_round", "args": {}, "state": "cancelled",
    }))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    calls, compactions = [], []

    async def attempt(prompt, **kwargs):
        calls.append((prompt, kwargs["resume"]))
        if len(calls) == 1:
            return TurnResult(session_id="old", error=(
                REMOTE_COMPACTION_ERROR if entry == "automatic" else "Context length exceeded"
            ))
        assert kwargs["resume"] is None
        assert "do not advance the round just for recovery" in prompt
        assert "round_validation.json" in prompt and "operations" in prompt
        assert "prediction" in prompt and original in prompt
        assert {p: p.read_bytes() for p in before} == before
        return TurnResult(session_id="fresh", text="Recovered")

    async def compact(**kwargs):
        compactions.append(kwargs)
        return TurnResult(session_id="old", error="Compaction failed: network error")

    monkeypatch.setattr(discovery, "_run_turn_once", attempt)
    monkeypatch.setattr(discovery, "compact_session", compact)
    answer = asyncio.run(discovery.run_turn(original, cwd=str(tmp_path), system_prompt="test"))
    assert answer.session_id == "fresh" and not answer.error
    assert len(calls) == 2 and len(compactions) == (entry == "native_failed")
    handoffs = list((tmp_path / "context_recovery").glob("*/handoff.json"))
    assert len(handoffs) == 1
    handoff = json.loads(handoffs[0].read_text())
    assert handoff["round"] == 3 and handoff["max_rounds"] == 8
    turn = json.loads((tmp_path / "turn.json").read_text())
    assert turn["prompt"] == original and turn["state"] == "finished"


def test_terminal_errors_override_retry_hints_and_never_replace(tmp_path, monkeypatch, discovery):
    calls = []

    async def attempt(*args, **kwargs):
        calls.append(kwargs)
        error = "HTTP 401 unauthorized: " + REMOTE_COMPACTION_ERROR
        return TurnResult(session_id="old", error=error, retryable=True)

    monkeypatch.setattr(discovery, "_run_turn_once", attempt)
    answer = asyncio.run(discovery.run_turn("Continue", cwd=str(tmp_path), system_prompt="test"))
    assert len(calls) == 1 and not answer.retryable
    assert not (tmp_path / "context_recovery").exists()
    turn = json.loads((tmp_path / "turn.json").read_text())
    assert turn["state"] == "failed" and not turn["resumable"]


def test_other_code_remote_compaction_error_has_safe_retry_classification():
    code, message, retryable = _error_details({
        "codexErrorInfo": "other", "message": REMOTE_COMPACTION_ERROR + " SECRET_TOKEN",
    })
    assert code == "remoteCompactionFailed" and retryable
    assert "SECRET_TOKEN" not in message
    assert _error_details({"codexErrorInfo": "other", "message": "provider error"})[2] is False
    code, _, retryable = _error_details({
        "codexErrorInfo": "other", "message": "Content policy: " + REMOTE_COMPACTION_ERROR,
    })
    assert code == "nonretryableError" and not retryable


def test_repeated_remote_compaction_failure_exhausts_one_replacement(
    tmp_path, monkeypatch, discovery,
):
    calls = []

    async def attempt(prompt, **kwargs):
        calls.append(kwargs["resume"])
        return TurnResult(session_id="old" if len(calls) == 1 else "replacement",
                          error=REMOTE_COMPACTION_ERROR, retryable=True)

    monkeypatch.setattr(discovery, "_run_turn_once", attempt)
    answer = asyncio.run(discovery.run_turn("Continue", cwd=str(tmp_path), system_prompt="test"))
    assert len(calls) == 2 and calls[-1] is None
    assert answer.error and not answer.stopped
    turn = json.loads((tmp_path / "turn.json").read_text())
    assert turn["state"] == "interrupted" and turn["resumable"]
    assert turn["recovery_reason"] == "context_recovery_exhausted"
    assert latest_turn_status(tmp_path)["status"] == "interrupted"


def test_nonretryable_native_compaction_error_does_not_replace(tmp_path, monkeypatch, discovery):
    error = "HTTP 401 unauthorized"
    calls = []

    async def attempt(prompt, **kwargs):
        calls.append(prompt)
        return TurnResult(session_id="old", error="Context length exceeded")

    async def compact(**kwargs):
        return TurnResult(session_id="old", error=error)

    monkeypatch.setattr(discovery, "_run_turn_once", attempt)
    monkeypatch.setattr(discovery, "compact_session", compact)
    answer = asyncio.run(discovery.run_turn("Continue", cwd=str(tmp_path), system_prompt="test"))
    assert len(calls) == 1 and answer.error == error
    assert not (tmp_path / "context_recovery").exists()
    turn = json.loads((tmp_path / "turn.json").read_text())
    assert turn["state"] == "failed" and not turn["resumable"]


@pytest.mark.parametrize("stop_during_drain", [False, True])
def test_abandoned_sdk_tool_drains_before_replacement_and_committed_write_is_not_replayed(
    tmp_path, monkeypatch, discovery, stop_during_drain,
):
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(
        run=SimpleNamespace(run_id="drain-test", run_dir=lambda: tmp_path),
    ))
    started, draining, release = threading.Event(), threading.Event(), threading.Event()
    writes, attempts, detached = [], [], []

    def mutate():
        with cancellation_guard():
            writes.append("committed")
            (tmp_path / "library.json").write_text('{"saved":true}')
        started.set()
        assert _CURRENT.get().event.wait(3)
        draining.set()
        assert release.wait(3)
        return {"ok": True, "saved": True}

    monkeypatch.setitem(server.TOOL_CALLABLES, "library_save", mutate)

    async def simulate_turn(resume):
        attempts.append(resume)
        if len(attempts) == 1:
            # Match the SDK shutdown grace period: its callback has detached
            # while the real synchronous operation is still running.
            detached.append(asyncio.create_task(server.invoke_tool("library_save", {})))
            assert await asyncio.to_thread(started.wait, 3)
            return False
        assert release.is_set() and writes == ["committed"]
        receipt = json.loads(next((tmp_path / "operations").glob("*.json")).read_text())
        assert receipt["state"] == "complete" and receipt["cancelled_after_completion"]
        assert resume is None
        return True

    if discovery.name == "claude":
        async def query(**kwargs):
            recovered = await simulate_turn(kwargs["options"].resume)
            yield sdk.ResultMessage(
                subtype="success" if recovered else "error_during_execution",
                duration_ms=1, duration_api_ms=1, is_error=not recovered, num_turns=1,
                session_id="fresh" if recovered else "old",
                result="Recovered" if recovered else None,
                errors=[] if recovered else [REMOTE_COMPACTION_ERROR],
            )
        monkeypatch.setattr(sdk, "query", query)
    else:
        async def turn(**kwargs):
            recovered = await simulate_turn(kwargs["resume"])
            return CodexTurnResult(
                thread_id="fresh" if recovered else "old",
                status="completed" if recovered else "failed",
                text="Recovered" if recovered else "", error=None if recovered else
                REMOTE_COMPACTION_ERROR, error_code=None if recovered else "remoteCompactionFailed",
            )
        monkeypatch.setattr("autofa.harness.codex_app_server.run_codex_turn", turn)

    async def scenario():
        task = asyncio.create_task(discovery.run_turn(
            "Continue", cwd=str(tmp_path), system_prompt="test", show_result=False,
        ))
        try:
            assert await asyncio.to_thread(draining.wait, 3)
            assert not task.done() and len(attempts) == 1
            if stop_during_drain:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            release.set()
            answer = await asyncio.wait_for(task, 3)
            await asyncio.gather(*detached, return_exceptions=True)
            return answer
        finally:
            release.set()

    answer = asyncio.run(scenario())
    assert writes == ["committed"]
    assert len(attempts) == (1 if stop_during_drain else 2)
    assert answer.stopped is stop_during_drain
    turn = json.loads((tmp_path / "turn.json").read_text())
    if stop_during_drain:
        assert turn["state"] == "interrupted" and turn["recovery_reason"] == "user_stopped"
        assert answer.session_id == turn["session_id"] == "old"
    else:
        assert answer.text == "Recovered" and turn["state"] == "finished"


def test_interrupted_round_validation_resumes_predictions_without_new_round(
    tmp_path, monkeypatch, discovery,
):
    from autofa.session.round_validation import validate_round

    fn = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=1, functions=[fn], n=100)
    detector_calls = _record_calls(monkeypatch)
    rt.run.round, rt.run.max_rounds = 1, 8
    rt.run.save()
    directory = rt.run.run_dir()
    before_run = deepcopy(vars(rt.run))
    reviews, attempts, spent = [], [], []

    def review(**kwargs):
        reviews.append(kwargs)
        if len(reviews) == 1:
            raise asyncio.CancelledError("transport closed during validation")
        return {"ok": True, "unresolved": []}

    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", review)
    monkeypatch.setitem(server.TOOL_CALLABLES, "advance_round", lambda: validate_round(rt))

    @drain_discovery_tools
    async def attempt(prompt, **kwargs):
        attempts.append(kwargs["resume"])
        try:
            response = await server.invoke_tool("advance_round", {})
        except asyncio.CancelledError:
            assert fn.scores["round_validation"]["complete"]
            assert fn.scores["n_scored"] == 30
            spent.append(len(detector_calls))
            return TurnResult(session_id="old", error=REMOTE_COMPACTION_ERROR)
        assert json.loads(response["content"][0]["text"])["ok"]
        assert kwargs["resume"] is None and len(detector_calls) == spent[0]
        return TurnResult(session_id="fresh", text="Validation resumed")

    monkeypatch.setattr(discovery, "_run_turn_once", attempt)
    answer = asyncio.run(discovery.run_turn(
        "Finish this validation", cwd=str(directory), system_prompt="test", show_result=False,
    ))
    assert not answer.error and len(attempts) == len(reviews) == 2
    assert vars(rt.run) == before_run and len(detector_calls) == spent[0]
    states = [json.loads(p.read_text())["state"] for p in (directory / "operations").glob("*.json")]
    assert sorted(states) == ["cancelled", "complete"]
    assert latest_turn_status(directory)["status"] == "finished"
