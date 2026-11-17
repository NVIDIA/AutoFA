# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn-end keeper checks return feedback before continuation or completion."""

import asyncio
import json
from threading import Event

import pytest

from autofa.core.dependencies import build_bundle
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.harness.claude import TurnResult
from autofa.mcp import tools_eval
from autofa.metrics.detector_checkpoints import DETECTOR_EXECUTION_VERSION
from autofa.session import runner, turn_validation
from autofa.util.cancellation import check_cancelled

from support.discovery import _quality_stall_assessment, keeper
from support.discovery import interactive as interactive
from support.discovery import interactive_runner as interactive_runner


@pytest.fixture
def validation_runtime(interactive_runner, monkeypatch):
    _, _, rt = interactive_runner
    fn = keeper(0, evaluation_run_id=rt.run.run_id)
    fn.kind = "llm"
    rt.library.functions = [fn]
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda rt: "current-source")
    events = []

    def score(*, names, fraction):
        events.append(("score", names, fraction))
        for selected in rt.library.functions:
            if selected.name in names:
                selected.scores.update(
                    sampled=True, evaluation_context="current-source",
                    evaluation_claim_hash=turn_validation._claim(selected),
                    evaluation_execution_version=DETECTOR_EXECUTION_VERSION,
                    evaluation_parent_identity=tools_eval._evaluation_parent_identity(rt, selected),
                    round_validation={"version": 1, "fraction": fraction, "complete": True,
                                      "panel_fingerprint": "representative-panel"},
                )
        return {"ok": True, "evaluated_names": names, "reused_names": []}

    def review(*, names, scope, **kwargs):
        events.append(("review", names, scope))
        return {"ok": True, "unresolved": []}

    monkeypatch.setattr(tools_eval, "evaluate_round_functions", score)
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", review)
    return rt, fn, events


def test_turn_validation_scores_only_good_llm_detectors_and_keeps_round_open(validation_runtime):
    rt, fn, events = validation_runtime
    cheap = keeper(1, evaluation_run_id=rt.run.run_id)
    weak = keeper(2, evaluation_run_id=rt.run.run_id, ppv_at_pi0=.5)
    weak.kind = "llm"
    rt.library.functions += [cheap, weak]
    original_round = rt.run.round
    result = turn_validation.validate_turn(rt)
    assert result["names"] == [fn.name]
    assert events == [("score", [fn.name], .8), ("review", [fn.name], "round_checkpoint")]
    assert rt.run.round == original_round
    assert not (rt.run.run_dir() / "round_validation.json").exists()
    assert not json.loads((rt.run.run_dir() / "discovery_target.json").read_text())["goal_met"]
    assert turn_validation.validate_turn(rt)["status"] == "not_required"
    assert len(events) == 2  # Feedback does not recursively buy another scan or review.


@pytest.mark.parametrize("fraction", [.8, 1.])
def test_current_larger_scans_require_no_turn_work(validation_runtime, fraction):
    rt, fn, events = validation_runtime
    fn.scores.update(
        evaluation_context="current-source",
        evaluation_execution_version=DETECTOR_EXECUTION_VERSION,
        sampled=fraction < 1,
        round_validation={"version": 1, "fraction": fraction, "complete": True,
                          "panel_fingerprint": "existing-panel"},
    )
    assert turn_validation.validate_turn(rt)["reused_names"] == [fn.name]
    assert events == []


def test_hybrid_dependency_classification_is_used_for_selection(validation_runtime, monkeypatch):
    rt, fn, events = validation_runtime
    fn.kind = "regex"
    semantic = FeatureFunction(name="semantic", description="", source=(
        "def detect(item): return llm.classify('check', item)"))
    fn.depends_on = [semantic.name]
    fn.meta = {"deps": {semantic.name: semantic.source}, "dependency_bundle": build_bundle(
        FunctionLibrary(functions=[semantic]), depends_on=[semantic.name])}
    # Isolate selection from re-judging this synthetic dependency change; actual
    # dependency classification and the evaluation reservation remain real.
    monkeypatch.setattr(turn_validation, "grade_summary", lambda *args, **kwargs: {
        "functions": [{"name": fn.name, "code": "GOOD"}], "counts": {"GOOD": 1},
    })
    result = turn_validation.validate_turn(rt)
    assert result["names"] == [fn.name] and events[0] == ("score", [fn.name], .8)


def test_interrupted_downgrade_resumes_its_selected_review(validation_runtime, monkeypatch):
    rt, fn, events = validation_runtime
    score = tools_eval.evaluate_round_functions

    def interrupted(**kwargs):
        score(**kwargs)
        fn.scores["ppv_at_pi0"] = .55
        raise asyncio.CancelledError()

    monkeypatch.setattr(tools_eval, "evaluate_round_functions", interrupted)
    with pytest.raises(asyncio.CancelledError):
        turn_validation.validate_turn(rt)
    pending = json.loads((rt.run.run_dir() / "turn_validation.json").read_text())
    assert pending["status"] == "pending" and pending["names"] == [fn.name]
    monkeypatch.setattr(tools_eval, "evaluate_round_functions", lambda **kwargs: {
        "ok": True, "reused_names": kwargs["names"], "evaluated_names": [],
    })
    result = turn_validation.validate_turn(rt, pending_only=True)
    assert result["status"] == "complete" and result["reused_names"] == [fn.name]
    assert events[-1] == ("review", [fn.name], "round_checkpoint")
    assert result["feedback"]["detectors"][0]["previous_grade"] in {
        "GOOD", "EXCELLENT", "OUTSTANDING",
    }


@pytest.mark.parametrize("unfinished", ["processing_error", "missing_source"])
def test_unfinished_review_remains_pending_and_resume_retries_saved_cohort(
    validation_runtime, monkeypatch, unfinished,
):
    rt, fn, events = validation_runtime

    def unavailable(**kwargs):
        fn.scores["ppv_at_pi0"] = .55
        fn.scores["round_review"] = {
            "scope": "round_checkpoint", "status": "unavailable",
            "claim_hash": turn_validation._claim(fn),
            "identity": "saved-cohort" if unfinished == "processing_error" else None,
        }
        fn.scores["behavior_n_errors"] = int(unfinished == "processing_error")
        return {"ok": False, "unresolved": [{"name": fn.name, "reason": unfinished}]}

    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", unavailable)
    blocked = turn_validation.validate_turn(rt)
    assert blocked["status"] == "review_blocked" and not blocked["ok"]
    receipt = json.loads((rt.run.run_dir() / "turn_validation.json").read_text())
    assert receipt["status"] == "review_blocked" and receipt["phase"] == "review"
    assert turn_validation.validate_turn(rt)["status"] == "not_required"

    def recovered(*, names, scope, retry_incomplete, **kwargs):
        assert names == [fn.name] and scope == "round_checkpoint" and retry_incomplete is True
        fn.scores["behavior_n_errors"] = 0
        return {"ok": True, "unresolved": []}

    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", recovered)
    monkeypatch.setattr(tools_eval, "evaluate_round_functions", lambda **kwargs: {
        "ok": True, "reused_names": kwargs["names"], "evaluated_names": [],
    })
    resumed = turn_validation.validate_turn(rt, pending_only=True)
    assert resumed["status"] == "complete" and resumed["ok"]
    assert resumed["reused_names"] == [fn.name]
    assert [event[0] for event in events] == ["score"]


def test_completed_uncertainty_is_feedback_without_endless_review_retries(
    validation_runtime, monkeypatch,
):
    rt, fn, _ = validation_runtime

    def inconclusive(**kwargs):
        fn.scores.update(behavior_n_uncertain=1, behavior_n_review_incomplete=0,
                         behavior_n_errors=0, behavior_evidence_stale=False)
        fn.scores["round_review"] = {
            "scope": "round_checkpoint", "status": "unavailable",
            "claim_hash": turn_validation._claim(fn), "identity": "completed-cohort",
        }
        return {"ok": False, "unresolved": [{"name": fn.name, "reason": "Evidence inconclusive"}]}

    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", inconclusive)
    result = turn_validation.validate_turn(rt)
    assert result["status"] == "complete" and result["ok"]
    assert turn_validation.validate_turn(rt, pending_only=True)["status"] == "not_required"


def test_changed_claim_cannot_resume_an_old_pending_selection(validation_runtime, monkeypatch):
    rt, fn, _ = validation_runtime
    monkeypatch.setattr(tools_eval, "evaluate_round_functions", lambda **kwargs: {
        "ok": False, "error": "Connection interrupted",
    })
    with pytest.raises(RuntimeError, match="selection is saved"):
        turn_validation.validate_turn(rt)
    fn.source += "\n# a new revision"
    assert turn_validation.validate_turn(rt, pending_only=True)["status"] == "not_required"
    assert json.loads((rt.run.run_dir() / "turn_validation.json").read_text())[
        "status"] == "superseded"


def test_successful_call_with_incomplete_cohort_does_not_finish_turn_validation(
    validation_runtime, monkeypatch,
):
    rt, fn, events = validation_runtime
    score = tools_eval.evaluate_round_functions

    def incomplete(**kwargs):
        result = score(**kwargs)
        fn.scores["round_validation"]["complete"] = False
        return result

    monkeypatch.setattr(tools_eval, "evaluate_round_functions", incomplete)
    with pytest.raises(RuntimeError, match="did not finish its current scoring cohort"):
        turn_validation.validate_turn(rt)
    assert [event[0] for event in events] == ["score"]
    assert json.loads((rt.run.run_dir() / "turn_validation.json").read_text())[
        "status"] == "pending"


def test_async_turn_checkpoint_cancellation_drains_worker_without_late_receipt(
    validation_runtime, monkeypatch,
):
    rt, _, _ = validation_runtime
    started, released = Event(), Event()

    def slow_score(**kwargs):
        started.set()
        assert released.wait(5)
        check_cancelled()
        pytest.fail("Cancelled checkpoint must not commit late results")

    monkeypatch.setattr(tools_eval, "evaluate_round_functions", slow_score)

    async def exercise():
        task = asyncio.create_task(turn_validation.turn_validation_prompt(rt))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(.01)
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert json.loads((rt.run.run_dir() / "turn_validation.json").read_text())[
        "status"] == "pending"


@pytest.mark.parametrize("interactive", ["claude", "codex"], indirect=True)
def test_harness_delivers_checkpoint_feedback_before_completion(interactive, monkeypatch):
    harness, commands = interactive
    commands.append("/quit")
    events = []

    async def turn(prompt, **kwargs):
        events.append(("turn", prompt))
        return TurnResult(session_id="same-thread", text="Done")

    async def validate():
        events.append(("validate", None))
        return "80% findings: repair precision" if len(events) == 2 else None

    def continuation():
        events.append(("continuation", None))
        return None

    async def finalize():
        events.append(("finalize", None))

    monkeypatch.setattr(harness, "run_turn", turn)
    asyncio.run(harness.run_session(
        cwd=".", system_prompt="system", initial_prompt="discover",
        turn_validation=validate, discovery_continuation=continuation,
        discovery_finalization=finalize,
    ))
    assert events == [("turn", "discover"), ("validate", None),
                      ("turn", "80% findings: repair precision"), ("validate", None),
                      ("continuation", None), ("finalize", None)]


def test_runner_wires_automatic_turn_feedback(interactive_runner, monkeypatch):
    harness, commands, rt = interactive_runner
    commands.append("/quit")
    rt.config.discovery_wait_for_plateau = False
    rt.run.min_failure_modes = 0
    rt.run.round = 0
    calls, prompts = [], []

    async def validate(runtime, *, pending_only=False, console=None):
        calls.append(pending_only)
        return "80% automatic checkpoint findings" if calls == [True, False] else None

    async def turn(prompt, **kwargs):
        prompts.append(prompt)
        rt.run.round = rt.run.max_rounds
        return TurnResult(session_id="session", text="Done")

    monkeypatch.setattr(turn_validation, "turn_validation_prompt", validate)
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, publish_on_exit=False)
    assert calls == [True, False, False]
    assert len(prompts) == 2 and prompts[-1] == "80% automatic checkpoint findings"


def test_resume_delivers_pending_turn_feedback_before_provisional_completion(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    commands.append("/quit")
    rt.config.discovery_wait_for_plateau = False
    rt.run.min_failure_modes = 0
    rt.run.round = rt.run.max_rounds
    prompts = []

    async def validate(runtime, *, pending_only=False, console=None):
        return "80% resumed checkpoint findings" if pending_only else None

    async def turn(prompt, **kwargs):
        prompts.append(prompt)
        return TurnResult(session_id="same-session", text="Reviewed findings")

    monkeypatch.setattr(turn_validation, "turn_validation_prompt", validate)
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True, publish_on_exit=False)
    assert len(prompts) == 1 and prompts[0].startswith("80% resumed checkpoint findings")


@pytest.mark.parametrize("batch", [False, True], ids=["interactive", "batch"])
def test_resume_capacity_stop_precedes_pending_review_feedback_without_a_paid_turn(
    interactive_runner, monkeypatch, batch,
):
    harness, commands, rt = interactive_runner
    if not batch:
        commands.append("/quit")
    capacity = {"status": "blocked", "reason": "Saved reviewer allowance exhausted."}
    assessment = {**_quality_stall_assessment(),
                  "review_capacity_block": capacity, "stop_reason": "review_capacity_exhausted"}
    finalized = []

    async def unexpected_feedback(*args, **kwargs):
        pytest.fail("A saved capacity stop must precede the pending-review feedback hook")

    async def unexpected_turn(*args, **kwargs):
        pytest.fail("Unchanged Resume must not dispatch another discovery turn")

    async def finalize(console):
        finalized.append(True)

    monkeypatch.setattr("autofa.session.discovery_stall.observe_discovery_stall",
                        lambda *args, **kwargs: assessment)
    monkeypatch.setattr("autofa.session.round_stagnation.round_stagnation_status",
                        lambda rt: assessment)
    # Model the crash boundary where the old closing prompt is still persisted.
    monkeypatch.setattr("autofa.session.round_validation.round_refinement_prompt",
                        lambda rt: "Still awaiting the previous closing response")
    monkeypatch.setattr(turn_validation, "turn_validation_prompt", unexpected_feedback)
    monkeypatch.setattr(harness, "run_turn", unexpected_turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True,
                       exit_after_discovery=batch, publish_on_exit=False)
    assert finalized == [True]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["blocked"] and not saved["minimum_met"]
    assert saved["stop_reason"] == "review_capacity_exhausted"
    assert commands == []


@pytest.fixture
def blocked_runtime(validation_runtime, monkeypatch):
    rt, fn, events = validation_runtime
    calls = []

    def unavailable(*, names, **kwargs):
        calls.append((names, kwargs))
        for name in names:
            if name != fn.name:
                continue
            fn.scores.update(behavior_n_errors=1, behavior_processing_failures=[{
                "item_id": "refused-case", "kind": "backend_refusal",
                "detector_rejected": False,
                "retry_policy": "requires_backend_or_configuration_change",
                "recovery_action": "Review backend refused this request; preserve the claim.",
            }])
            fn.scores["round_review"] = {
                "status": "unavailable", "scope": "round_checkpoint",
                "claim_hash": turn_validation._claim(fn), "identity": "frozen-review",
                "item_ids": ["reviewed-yes", "reviewed-no", "refused-case"],
            }
        return {"ok": fn.name not in names, "unresolved": [
            {"name": fn.name, "reason": "The reviewer refused this request."}
        ] if fn.name in names else []}

    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", unavailable)
    return rt, fn, events, calls


def test_new_keeper_checkpoint_preserves_another_detectors_blocked_review(blocked_runtime):
    rt, blocked, events, calls = blocked_runtime
    first = turn_validation.validate_turn(rt)
    keeper2 = keeper(1, evaluation_run_id=rt.run.run_id)
    keeper2.kind = "llm"
    rt.library.functions.append(keeper2)
    second = turn_validation.validate_turn(rt)
    assert second["names"] == [keeper2.name]
    assert second["status"] == "review_blocked"
    assert second["blocked_reviews"] == first["blocked_reviews"]
    assert [names for names, _ in calls] == [[blocked.name], [keeper2.name]]
    assert turn_validation.validate_turn(rt, pending_only=True)["status"] == "review_blocked"
    assert turn_validation.validate_turn(rt)["status"] == "not_required"
    assert len(events) == len(calls) == 2


def test_backend_context_change_reconsiders_blocked_review_without_losing_baseline(
    blocked_runtime, monkeypatch,
):
    rt, fn, _, calls = blocked_runtime
    context = ["reviewer-before"]
    monkeypatch.setattr(turn_validation, "_review_context", lambda rt: context[0])
    first = turn_validation.validate_turn(rt)
    fn.scores["ppv_at_pi0"] = .55
    context[0] = "reviewer-after"
    monkeypatch.setattr(tools_eval, "evaluate_round_functions", lambda **kwargs: {
        "ok": True, "reused_names": kwargs["names"], "evaluated_names": [],
    })
    result = turn_validation.validate_turn(rt)
    assert result["names"] == [fn.name] and result["reused_names"] == [fn.name]
    assert result["before"] == first["before"]
    assert result["blocked_reviews"][fn.name]["review_context"] == "reviewer-after"
    assert len(calls) == 2
    assert turn_validation.validate_turn(rt)["status"] == "not_required"


def test_capability_resolution_during_scoring_does_not_retry_blocked_review(
    blocked_runtime, monkeypatch,
):
    rt, fn, events, calls = blocked_runtime
    context = ["before-probe"]
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda rt: context[0])
    score = tools_eval.evaluate_round_functions

    def resolved(**kwargs):  # The first scan resolves endpoint capabilities.
        result = score(**kwargs)
        context[0] = "current-source"
        return result

    monkeypatch.setattr(tools_eval, "evaluate_round_functions", resolved)
    first = turn_validation.validate_turn(rt)
    assert first["blocked_reviews"][fn.name]["evaluation_context"] == "current-source"
    assert turn_validation.validate_turn(rt)["status"] == "not_required"
    assert len(events) == len(calls) == 1


def test_changed_review_evidence_reconciles_blocker_once(blocked_runtime, monkeypatch):
    rt, fn, _, calls = blocked_runtime
    turn_validation.validate_turn(rt)
    # An explicit successful review supplies usable results; the checkpoint may
    # reconcile them, preserving final_review's representative-cohort checks.
    fn.scores.update(behavior_n_errors=0, behavior_processing_failures=[])
    fn.scores["round_review"]["status"] = "complete"
    reconciliations = []
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions",
                        lambda **kwargs: reconciliations.append(kwargs) or {
                            "ok": True, "unresolved": [], "skipped": [fn.name]})
    monkeypatch.setattr(tools_eval, "evaluate_round_functions", lambda **kwargs: {
        "ok": True, "reused_names": kwargs["names"], "evaluated_names": [],
    })
    result = turn_validation.validate_turn(rt)
    assert result["ok"] and result["status"] == "complete"
    assert result["blocked_reviews"] == {} and result["reused_names"] == [fn.name]
    assert len(reconciliations) == 1 and len(calls) == 1
    assert turn_validation.validate_turn(rt)["status"] == "not_required"


@pytest.mark.parametrize("mutation", ["removed", "revised"])
def test_stale_blocked_receipt_is_retired_without_reviewing_old_claim(blocked_runtime, mutation):
    rt, fn, _, calls = blocked_runtime
    turn_validation.validate_turn(rt)
    if mutation == "removed":
        rt.library.functions = []
    else:
        fn.source += "\n# changed detector revision"
    assert turn_validation.validate_turn(rt, pending_only=True)["status"] == "not_required"
    saved = json.loads((rt.run.run_dir() / "turn_validation.json").read_text())
    assert saved["status"] == "superseded" and saved["blocked_reviews"] == {}
    assert len(calls) == 1


def test_old_pending_refusal_migrates_to_blocked_feedback_without_rescoring(
    blocked_runtime, monkeypatch,
):
    rt, fn, events, calls = blocked_runtime
    first = turn_validation.validate_turn(rt)
    path = rt.run.run_dir() / "turn_validation.json"
    # Reproduce the schema saved by the fatal version used by the affected run.
    old = {key: value for key, value in first.items() if key != "blocked_reviews"}
    old.update(status="pending", phase="review")
    path.write_text(json.dumps(old))
    monkeypatch.setattr(tools_eval, "evaluate_round_functions", lambda **kwargs: {
        "ok": True, "reused_names": kwargs["names"], "evaluated_names": [],
    })
    resumed = turn_validation.validate_turn(rt, pending_only=True)
    assert resumed["status"] == "review_blocked"
    assert resumed["reused_names"] == [fn.name]
    assert resumed["before"] == first["before"]
    assert calls[-1][1]["retry_incomplete"] is True
    assert len(events) == 1  # Only the original score call; no rescan on Resume.
    before = len(calls)
    turn_validation.validate_turn(rt, pending_only=True)
    assert turn_validation.validate_turn(rt)["status"] == "not_required"
    assert len(calls) == before


def test_mixed_processing_failures_can_retry_recoverable_sibling_only(blocked_runtime, monkeypatch):
    rt, fn, _, calls = blocked_runtime
    turn_validation.validate_turn(rt)
    fn.scores["behavior_processing_failures"].append({
        "item_id": "invalid-case", "kind": "invalid_reply", "detector_rejected": False,
        "retry_policy": "one_explicit_stage_retry",
    })
    # Snapshot the mixed cohort as already blocked, so only explicit Resume can
    # request recovery; a changed state is not what makes this eligible.
    path = rt.run.run_dir() / "turn_validation.json"
    saved = json.loads(path.read_text())
    saved["blocked_reviews"][fn.name]["review_state"] = turn_validation._review_state(fn)
    path.write_text(json.dumps(saved))
    assert turn_validation._can_retry_review(fn)
    retries = []

    def recover(**kwargs):
        retries.append(kwargs)
        assert kwargs["retry_incomplete"] is True
        fn.scores["behavior_processing_failures"] = fn.scores["behavior_processing_failures"][:1]
        fn.scores["round_review"]["retry_generation"] = 1
        return {"ok": False, "unresolved": [{"name": fn.name, "reason": "Backend refusal remains"}]}

    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", recover)
    result = turn_validation.validate_turn(rt, pending_only=True)
    assert result["status"] == "review_blocked" and len(retries) == 1
    assert not turn_validation._can_retry_review(fn)
    turn_validation.validate_turn(rt, pending_only=True)
    assert turn_validation.validate_turn(rt)["status"] == "not_required"
    assert len(retries) == len(calls) == 1
