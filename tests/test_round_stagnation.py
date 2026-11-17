# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rounds close after bounded comparisons and validation; patience counts validated progress."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, RLock
from types import SimpleNamespace

import pytest

from autofa.config import RunConfig
from autofa.mcp import tools_eval, tools_library, tools_repair, tools_repair_versions
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.final_validation_selection import prepare_final_validation
from autofa.metrics.task_support import build_task_support
from autofa.session import repair_workflow, round_stagnation, runner
from autofa.session.campaign_policy import validate_campaign_policy
from autofa.session.discovery_stall import complete_discovery_round, observe_discovery_stall
from autofa.session.discovery_target import failure_mode_target_for_runtime
from autofa.session.exploration import round_exploration_seed
from autofa.session.family_discovery import STATE_FILE, family_discovery
from autofa.session.round_stagnation import round_stagnation_status
from autofa.session.round_validation import active_round_refinement

from support.discovery import add, candidate, close, detector, next_round
from support.discovery import rt as rt
from support.families import (
    FamilyDataset,
    _complete_retained_batch,
    _save_checkpoints,
    _terminal_comparison,
    evaluated_family,
    record,
    survey_rows,
)
from support.families import batch_runtime as batch_runtime
from support.repair_review import _propose, _refinement, _synthesis_pending, plan, submit, workflow
from support.repair_review import repair_runtime as repair_runtime
from support.task_support import attach_task_support

# A round consumes bounded comparison effort and validation before rotating persisted seeds.


def _checkpoint_snapshot(rt):
    return {
        "run": deepcopy(vars(rt.run)),
        "config": deepcopy(vars(rt.config)),
        "library": deepcopy(rt.library.to_dict()),
        "scores_history": deepcopy(rt.scores_history),
        "checkpoints": {
            name: (rt.run.run_dir() / name).read_bytes()
            for name in ("config.json", "library.json", "scores_history.json")
        },
    }


def _assert_blocked(rt, *, completed, phase):
    before = _checkpoint_snapshot(rt)
    result = tools_library.advance_round()
    assert not result["ok"], result
    assert result["advanced"] is False
    progress = result["family_discovery"]
    assert progress["comparison_budget"] == 3
    assert progress["completed_comparisons"] == completed
    assert progress["phase"] == phase
    assert progress["round"] == before["run"]["round"]
    assert _checkpoint_snapshot(rt) == before


def _concurrently(call):
    """Start two identical requests together and return both results."""
    ready = Barrier(2)

    def together():
        ready.wait(timeout=5)
        return call()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(together) for _ in range(2)]
        return [future.result(timeout=5) for future in futures]


def _finish_retained_batch():
    result = tools_library.advance_round()
    if result.get("refinement_pending"):
        return tools_library.advance_round(
            validation_response="Keep supported versions; defer unsupported claims to later work.",
        )
    return result


def test_round_cannot_advance_before_its_comparison_budget_is_spent(batch_runtime):
    rt = batch_runtime
    path = rt.run.run_dir() / STATE_FILE
    assert not path.exists()
    _assert_blocked(rt, completed=0, phase="survey")
    state = json.loads(path.read_text())
    assert state["round"] == 1
    assert state["comparisons"] == state["history"] == []
    # One unsuccessful comparison does not exhaust a balanced round.
    result = _terminal_comparison(rt, 0, "no_coherent_family")
    assert result["completed_comparisons"] == 1
    assert result["exploration_exhausted"] is None
    _save_checkpoints(rt)
    family_before = path.read_bytes()
    _assert_blocked(rt, completed=1, phase="survey")
    assert path.read_bytes() == family_before


def test_failed_assessment_keeps_pending_claim_and_seed_checkpoints(batch_runtime):
    rt = batch_runtime
    _terminal_comparison(rt, 0, "no_coherent_family")
    _terminal_comparison(rt, 1, "no_coherent_family")
    assert record(family_name="pending_family", evidence=survey_rows(*range(4, 10)))["ok"]
    fn = evaluated_family(rt)
    fn.description = "An independently saved claim that differs from the recorded hypothesis."
    fn.scores.update(
        evaluation_run_id=rt.run.run_id,
        evaluation_claim_hash=claim_hash(
            source=fn.source, description=fn.description, mechanism=fn.mechanism,
        ),
    )
    _save_checkpoints(rt)
    before = _checkpoint_snapshot(rt)
    family_before = (rt.run.run_dir() / STATE_FILE).read_bytes()

    assessed = family_discovery(
        action="assess_family", family_name=fn.name,
        assessment="Attempt to count the current evaluation of a different claim.",
    )
    assert not assessed["ok"]
    assert "differs from the recorded family" in assessed["error"]
    _assert_blocked(rt, completed=2, phase="assess_family")

    assert _checkpoint_snapshot(rt) == before
    assert (rt.run.run_dir() / STATE_FILE).read_bytes() == family_before


def test_completed_batch_rotates_seed_and_resets_family_together_without_good_modes(
    batch_runtime,
):
    rt = batch_runtime
    # One batch mixes every terminal outcome; each counts toward the comparison budget.
    for index, outcome in enumerate(("assessed", "deferred", "no_coherent_family")):
        result = _terminal_comparison(rt, index, outcome)
        assert result["completed_comparisons"] == index + 1
    assert result["phase"] == "specialize"
    assert result["exploration_exhausted"] == "comparison_budget_completed"
    _save_checkpoints(rt)
    before_history = (rt.run.run_dir() / "scores_history.json").read_bytes()
    prior_family = json.loads((rt.run.run_dir() / STATE_FILE).read_text())

    advanced = _finish_retained_batch()

    assert advanced["ok"], advanced
    assert advanced["advanced"] is True
    assert advanced["n_qualifying_failure_modes"] == 0
    assert advanced["round"] == rt.run.round == 2
    assert advanced["exploration_base_seed"] == rt.run.exploration_base_seed == 42
    assert advanced["exploration_sequence_index"] == rt.run.exploration_sequence_index == 38
    assert advanced["exploration_seed"] == rt.run.exploration_seed == round_exploration_seed(42, 38)
    assert advanced["family_discovery_phase"] == "survey"
    family = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
    assert family["round"] == rt.run.round
    assert family["phase"] == "survey"
    assert family["comparisons"] == []
    assert family["survey"] is family["assessment"] is family["exploration_exhausted"] is None
    assert family["history"][-1]["comparisons"] == prior_family["comparisons"]
    persisted_run = RunConfig.load(rt.run.run_id)
    assert persisted_run.round == 2
    assert persisted_run.exploration_sequence_index == 38
    assert persisted_run.exploration_seed == rt.run.exploration_seed
    library = json.loads((rt.run.run_dir() / "library.json").read_text())
    assert library["exploration_state"] == {"version": 1, "base_seed": 42, "sequence_index": 38}
    assert (rt.run.run_dir() / "scores_history.json").read_bytes() == before_history

    # A completed previous batch must not satisfy this new round's budget.
    _assert_blocked(rt, completed=0, phase="survey")


def test_inspected_inventory_exhaustion_allows_a_shorter_batch(batch_runtime):
    rt = batch_runtime
    rt.dataset = FamilyDataset()
    assert record()["ok"]
    fn = evaluated_family(rt)
    fn.scores["evaluation_run_id"] = rt.run.run_id
    assessed = family_discovery(
        action="assess_family", family_name=fn.name,
        assessment="Zero recall; every available failed task has now been inspected.",
    )
    assert assessed["ok"], assessed
    assert assessed["completed_comparisons"] == 1
    assert assessed["comparison_budget"] == 3
    assert assessed["exploration_exhausted"] == "available_failed_task_evidence_exhausted"

    result = _finish_retained_batch()

    assert result["ok"], result
    assert result["advanced"] is True
    assert result["round"] == 2
    status = family_discovery()
    assert status["round"] == 2
    assert status["completed_comparisons"] == 0


def test_simultaneous_advances_consume_a_completed_batch_only_once(batch_runtime):
    rt = batch_runtime
    for index in range(3):
        _terminal_comparison(rt, index, "no_coherent_family")
    results = _concurrently(tools_library.advance_round)

    advanced = [result for result in results if result["advanced"]]
    blocked = [result for result in results if not result["advanced"]]
    assert len(advanced) == len(blocked) == 1
    assert advanced[0]["ok"]
    assert not blocked[0]["ok"]
    assert blocked[0]["family_discovery"]["completed_comparisons"] == 0
    assert rt.run.round == 2
    assert rt.run.exploration_sequence_index == 38
    assert rt.run.exploration_seed == round_exploration_seed(42, 38)
    family = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
    assert family["round"] == 2
    assert family["comparisons"] == []
    assert len(family["history"]) == 1
    assert len(family["history"][0]["comparisons"]) == 3


def test_driver_completion_exit_precedes_fresh_batch_guard(batch_runtime, monkeypatch):
    rt = batch_runtime
    rt.run.round = rt.run.max_rounds
    _save_checkpoints(rt)
    path = rt.run.run_dir() / STATE_FILE
    assert not path.exists()
    before = _checkpoint_snapshot(rt)
    monkeypatch.setattr(tools_library, "_pareto_status", lambda: {
        "goal_met": True, "ready_for_final_validation": False,
        "final_validation_candidates": [],
        "final_validation_guidance": "Return for final validation.",
    })

    result = tools_library.advance_round()

    assert result["ok"], result
    assert result["advanced"] is False
    assert result["goal_met"] is True
    assert _checkpoint_snapshot(rt) == before
    assert not path.exists()


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_validation_finishes_before_the_round_cursor_moves(batch_runtime, monkeypatch, outcome):
    rt = batch_runtime
    for index in range(3):
        _terminal_comparison(rt, index, "no_coherent_family")
    before = rt.run.round
    calls = []

    def validate(runtime):
        calls.append(runtime.run.round)
        assert runtime.run.round == before
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return {"ok": False, "error": "Restore evidence access."}

    monkeypatch.setattr("autofa.session.round_validation.validate_round", validate)
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            tools_library.advance_round()
    else:
        assert tools_library.advance_round()["advanced"] is False
    assert calls == [before]
    assert rt.run.round == before


def test_validation_feedback_keeps_round_open_and_repeated_request_reuses_it(
    batch_runtime, monkeypatch,
):
    rt = batch_runtime
    _complete_retained_batch(rt)
    before = _checkpoint_snapshot(rt)
    family_before = (rt.run.run_dir() / STATE_FILE).read_bytes()
    calls = []

    def validate(runtime):
        calls.append(runtime.run.round)
        return {"ok": True, "message": "Precision fell after validation.",
                "detectors": [{"name": runtime.library.functions[0].name, "grade": "ok"}]}

    monkeypatch.setattr("autofa.session.round_validation.validate_round", validate)
    first = tools_library.advance_round(
        validation_response="An early acknowledgement is not enough.",
    )
    assert first["ok"] and not first["advanced"]
    assert first["refinement_pending"]
    assert first["round"] == rt.run.round == 1
    assert _checkpoint_snapshot(rt) == before
    assert (rt.run.run_dir() / STATE_FILE).read_bytes() == family_before
    state = active_round_refinement(rt)
    assert state["status"] == "awaiting_response"
    assert state["run_id"] == rt.run.run_id
    assert state["round"] == 1
    assert set(state["evaluated_claims"]) == {fn.name for fn in rt.library.functions}
    assert "response" not in state

    # The receipt survives Resume; a repeated call is not another paid checkpoint.
    rt.run = RunConfig.load(rt.run.run_id)
    repeated = tools_library.advance_round()
    assert repeated["round_validation"] == first["round_validation"]
    assert not repeated["advanced"]
    assert calls == [1]


@pytest.mark.parametrize("edit", [False, True])
def test_acknowledged_validation_rechecks_current_versions_before_advancing(
    batch_runtime, monkeypatch, edit,
):
    rt = batch_runtime
    _complete_retained_batch(rt)
    calls = []

    def validate(runtime):
        calls.append((runtime.run.round, runtime.library.functions[0].description))
        return {"ok": True, "message": "Checked current detector versions."}

    monkeypatch.setattr("autofa.session.round_validation.validate_round", validate)
    first = tools_library.advance_round()
    assert not first["advanced"]
    fn = rt.library.functions[0]
    if edit:
        fn.description += " Only when the required output remains absent."
    response = "Checked the cited cases and retained the supported claim."
    result = tools_library.advance_round(validation_response=response)

    assert result["ok"] and result["advanced"]
    assert rt.run.round == 2
    assert rt.run.exploration_sequence_index == 38
    assert [row[0] for row in calls] == [1, 1]
    assert (calls[0][1] != calls[1][1]) is edit
    state = json.loads((rt.run.run_dir() / "round_validation.json").read_text())["refinement"]
    assert state["status"] == "complete"
    assert state["round"] == 1
    assert state["response"] == response
    assert state["changed_detectors"] == ([fn.name] if edit else [])
    assert state["retained_detectors"] == ([] if edit else [fn.name])
    assert active_round_refinement(rt) is None
    assert not tools_library.advance_round(validation_response=response)["advanced"]
    assert len(calls) == 2


def test_acknowledged_validation_resumes_after_interruption_without_new_response(
    batch_runtime, monkeypatch,
):
    rt = batch_runtime
    _complete_retained_batch(rt)
    assert tools_library.advance_round()["refinement_pending"]
    calls = []

    def validate(runtime):
        calls.append(runtime.run.round)
        assert active_round_refinement(runtime)["status"] == "validating_response"
        if len(calls) == 1:
            raise asyncio.CancelledError
        return {"ok": True, "message": "Saved operation resumed."}

    monkeypatch.setattr("autofa.session.round_validation.validate_round", validate)
    response = "The supported detector is retained after checking counterexamples."
    with pytest.raises(asyncio.CancelledError):
        tools_library.advance_round(validation_response=response)
    assert rt.run.round == 1
    assert rt.run.exploration_sequence_index == 37
    assert active_round_refinement(rt)["response"] == response

    result = tools_library.advance_round()
    assert result["ok"] and result["advanced"]
    assert calls == [1, 1]
    assert rt.run.round == 2
    assert rt.run.exploration_sequence_index == 38


def test_closing_repairs_do_not_require_a_new_family_or_synthesis_batch(batch_runtime):
    rt = batch_runtime
    _complete_retained_batch(rt)
    assert tools_library.advance_round()["refinement_pending"]
    path = rt.run.run_dir() / STATE_FILE
    state = json.loads(path.read_text())
    state["synthesis_required"] = True
    state["phase"] = "assess_family"
    state["survey"]["approach"] = "bottom_up"
    path.write_text(json.dumps(state))

    result = tools_library.advance_round(validation_response="Retain the checked candidates.")
    assert result["ok"] and result["advanced"]
    assert rt.run.round == 2


def test_extra_round_provisional_completion_still_allows_closing_response(
    batch_runtime, monkeypatch,
):
    rt = batch_runtime
    _complete_retained_batch(rt)
    rt.run.max_rounds = rt.run.round
    status = {"goal_met": False, "ready_for_final_validation": True,
              "final_validation_guidance": "Return for final validation."}
    monkeypatch.setattr(tools_library, "_pareto_status", lambda: status)

    result = tools_library.advance_round()
    assert result["refinement_pending"] and not result["advanced"]
    status.update(goal_met=True, ready_for_final_validation=False)
    repeated = tools_library.advance_round()
    assert repeated["refinement_pending"] and not repeated["advanced"]
    result = tools_library.advance_round(validation_response="Retain validated versions.")
    assert result["ok"] and result["advanced"]
    assert rt.run.round == 2


def test_simultaneous_closing_responses_advance_only_once(batch_runtime):
    rt = batch_runtime
    _complete_retained_batch(rt)
    assert tools_library.advance_round()["refinement_pending"]
    results = _concurrently(
        lambda: tools_library.advance_round(validation_response="Retain supported versions."),
    )

    assert sum(bool(result["advanced"]) for result in results) == 1
    assert rt.run.round == 2
    assert rt.run.exploration_sequence_index == 38


def test_stall_round_is_committed_only_after_validation_response(batch_runtime, monkeypatch):
    rt = batch_runtime
    _complete_retained_batch(rt)
    completed = []

    def checkpoint(runtime, validation):
        completed.append((runtime.run.round, validation["refinement"]["status"]))
        assert validation["refinement"]["response"] == "Retain the checked claim."
        return {"should_stop": False, "rounds_without_improve": 1, "patience": 3}

    monkeypatch.setattr("autofa.session.discovery_stall.complete_discovery_round", checkpoint)
    first = tools_library.advance_round()
    assert first["refinement_pending"] and completed == []
    assert tools_library.advance_round()["refinement_pending"] and completed == []
    result = tools_library.advance_round(validation_response="Retain the checked claim.")
    assert completed == [(1, "complete")]
    assert result["advanced"] and rt.run.round == 2
    assert result["quality_stall"]["rounds_without_improve"] == 1


def test_exhausted_round_patience_hands_off_without_starting_another_batch(
    batch_runtime, monkeypatch,
):
    rt = batch_runtime
    _complete_retained_batch(rt)
    assert tools_library.advance_round()["refinement_pending"]
    reason = "3 completed extra recovery rounds without validated library improvement"
    stopped = {"should_stop": True, "reason": reason,
               "rounds_without_improve": 3, "patience": 3}
    monkeypatch.setattr("autofa.session.discovery_stall.complete_discovery_round",
                        lambda runtime, validation: stopped)
    result = tools_library.advance_round(validation_response="Retain the checked claim.")
    assert result["stop_reason"] == "stalled"
    before = _checkpoint_snapshot(rt)
    monkeypatch.setattr("autofa.session.discovery_stall.observe_discovery_stall",
                        lambda runtime: stopped)
    repeated = tools_library.advance_round()
    assert repeated["ok"] and not repeated["advanced"]
    assert repeated["stop_reason"] == "stalled"
    assert _checkpoint_snapshot(rt) == before

# Round-closing feedback permits bounded repairs without resetting search budgets.


def test_validated_candidate_can_plan_repair_without_resetting_family_budget(
    repair_runtime, monkeypatch,
):
    rt, _, calls = repair_runtime
    family_path = _synthesis_pending(rt, monkeypatch)
    before_family = family_path.read_bytes()
    before = deepcopy(rt.library.get("guess").to_dict())
    blocked = _propose()
    assert not blocked["ok"] and "Inspect supported seeds" in blocked["error"]
    assert workflow(rt) == {}

    _refinement(rt)
    planned = _propose()
    assert planned["ok"], planned
    saved = workflow(rt)
    assert saved["latest"]["before"] == before
    assert saved["baseline"] == before
    assert saved["attempts_used"] == 1
    assert family_path.read_bytes() == before_family
    assert calls == []


def test_unrelated_or_closed_refinement_does_not_bypass_synthesis(repair_runtime, monkeypatch):
    rt, _, calls = repair_runtime
    family_path = _synthesis_pending(rt, monkeypatch)
    before = family_path.read_bytes()
    # Each rejected receipt leaves no state behind, so one evaluated runtime checks them all.
    for overrides in ({"run_id": "other-run"}, {"status": "complete"},
                      {"evaluated_claims": {"guess": "different-version"}}):
        _refinement(rt, **overrides)
        result = _propose()
        assert not result["ok"] and "Inspect supported seeds" in result["error"], overrides
        assert workflow(rt) == {}
        assert family_path.read_bytes() == before
    assert calls == []


def test_refinement_does_not_replenish_repair_attempt_budget(repair_runtime, monkeypatch):
    rt, _, calls = repair_runtime
    _synthesis_pending(rt, monkeypatch)
    _refinement(rt)
    for _ in range(repair_workflow.MAX_ATTEMPTS):
        plan("tighten_detector")
        result = tools_repair.defer_function_repair("guess", "Retain this evaluated baseline.")
        assert result["ok"], result
    before = workflow(rt)
    exhausted = _propose()
    assert not exhausted["ok"] and "attempt limit" in exhausted["error"].lower()
    assert workflow(rt) == before
    assert before["attempts_used"] == repair_workflow.MAX_ATTEMPTS
    assert calls == []


def test_refinement_repair_preserves_frozen_plan_and_rollback(repair_runtime, monkeypatch):
    rt, _, calls = repair_runtime
    _synthesis_pending(rt, monkeypatch)
    _refinement(rt)
    baseline = deepcopy(rt.library.get("guess").to_dict())
    attempt = plan("tighten_detector")
    submit(attempt)
    revised = rt.library.get("guess").to_dict()
    assert not tools_repair._round_refinement_candidate(rt, revised)
    assert workflow(rt)["latest"]["before"] == baseline
    deferred = tools_repair.defer_function_repair("guess", "The evaluated revision loses breadth.")
    assert deferred["ok"], deferred
    restored = tools_repair_versions.restore_function_version(
        "guess", repair_workflow._version(baseline), "Retain the broader evaluated version.",
        repair_workflow._version(revised),
    )
    assert restored["ok"], restored
    assert rt.library.get("guess").source == baseline["source"]
    assert workflow(rt)["latest"]["before"] == baseline
    assert workflow(rt)["attempts_used"] == 1
    assert calls == []

# Extra-round patience observes validated libraries, not individual failed probes.


def capacity_blocked_polishing(rt, *, failure_kind="review_budget_exhausted"):
    rt.run.min_failure_modes = 2
    rt.library.functions = [detector(rt, "complete"), detector(rt, "unfinished")]
    close(rt)
    rt.run.round += 1
    fn = rt.library.functions[-1]
    fn.scores.update(tools_eval._review_settings(rt.config)[2])
    fn.scores["behavior_n_errors"] = 1
    fn.scores["final_review"]["status"] = "unavailable"
    fn.scores["behavior_verdicts"][0].update(
        verdict="error", processing_status="error", completion_status="processing_error",
        processing_failure={"kind": failure_kind, "retry_policy": "one_explicit_stage_retry"},
    )
    return {"ok": True, "fraction": 1.0, "review_unresolved": [{"name": fn.name}],
            "refinement": {"mode": "polishing",
                           "frozen_names": [fn.name for fn in rt.library.functions]}}


def test_exhausted_review_stops_impossible_polishing_without_spending_quality_patience(rt):
    validation = capacity_blocked_polishing(rt)
    before = deepcopy([fn.to_dict() for fn in rt.library.functions])
    # Disabling quality patience does not grant additional reviewer capacity.
    rt.config.discovery_terminate_if_stalled = False
    result = complete_discovery_round(rt, validation)
    assert result["should_stop"] and result["stop_reason"] == "review_capacity_exhausted"
    assert result["status"] == "review_capacity_exhausted" and not result["available"]
    assert not result["counted"]
    assert result["rounds_without_improve"] == result["completed_extra_rounds"] == 0
    assert result["last_completed_round"] == 2  # The incomplete round never became a baseline.
    assert result["review_capacity_block"]["maximum_unblocked_candidates"] == 1
    assert round_stagnation_status(rt)["should_stop"]
    assert observe_discovery_stall(rt)["stop_reason"] == "review_capacity_exhausted"
    assert complete_discovery_round(rt, validation)["rounds_without_improve"] == 0
    assert [fn.to_dict() for fn in rt.library.functions] == before


def test_retryable_outage_keeps_incomplete_validation_without_terminal_capacity_block(rt):
    validation = capacity_blocked_polishing(rt, failure_kind="transport")
    path = rt.run.run_dir() / "discovery_round_stagnation.json"
    before = path.read_bytes()
    result = complete_discovery_round(rt, validation)
    assert result["status"] == "validation_incomplete" and not result["counted"]
    assert not result["should_stop"] and not result["review_capacity_block"]
    assert path.read_bytes() == before  # No outage consumes or erases completed history.


def test_capacity_block_does_not_stop_other_frozen_candidates_that_can_meet_target(rt):
    validation = capacity_blocked_polishing(rt)
    rt.library.functions.append(detector(rt, "another"))
    validation["refinement"]["frozen_names"].append("another")
    result = complete_discovery_round(rt, validation)
    assert result["status"] == "validation_incomplete" and not result["should_stop"]


def test_capacity_stop_survives_unrelated_finalization_and_interrupted_cursor_commit(rt):
    validation = capacity_blocked_polishing(rt)
    rt.run.round = rt.run.max_rounds - 1
    assert complete_discovery_round(rt, validation)["should_stop"]
    # The capacity receipt committed, but the round cursor has not advanced yet.
    assert observe_discovery_stall(rt)["should_stop"]
    other = rt.library.functions[0]
    other.scores["final_review"]["identity"] = "new-finalized-review"
    other.scores.update(n_scored=90, n_insufficient=10, recall=.11)
    assert round_stagnation_status(rt)["stop_reason"] == "review_capacity_exhausted"


def test_exhausted_prerequisite_review_can_block_before_a_final_review_exists(rt):
    field, scope = "round_review", "round_checkpoint"
    validation = capacity_blocked_polishing(rt)
    fn = rt.library.functions[-1]
    receipt = fn.scores.pop("final_review")
    fn.scores[field] = {**receipt, "scope": scope, "operation_id": "frozen-operation"}
    fn.scores["behavior_review_operation"] = {
        "scope": scope, "operation_id": "frozen-operation", "origin_run_id": rt.run.run_id,
    }
    validation["final_validation_selection"] = {"unresolved": [{"name": fn.name}]}
    result = complete_discovery_round(rt, validation)
    assert result["stop_reason"] == "review_capacity_exhausted"
    assert result["review_capacity_block"]["detectors"][0]["review_scope"] == scope
    assert "final_review" not in fn.scores
    # A finalizer may add a different-scope receipt while retaining this blocked
    # prerequisite and its failed item. That does not replenish the original budget.
    fn.scores["final_review"] = {**receipt, "scope": "full_corpus", "identity": "new-scope"}
    assert round_stagnation_status(rt)["should_stop"]


@pytest.mark.parametrize("change", ["claim", "reviewer"])
def test_capacity_block_is_current_evidence_and_reviewer_specific(rt, change):
    validation = capacity_blocked_polishing(rt)
    assert complete_discovery_round(rt, validation)["should_stop"]
    fn = rt.library.functions[-1]
    if change == "claim":
        fn.description += " A changed, explicitly approved scope."
    else:
        rt.config.behavior_judge_model = "different-authorized-reviewer"
    assert not round_stagnation_status(rt)["should_stop"]
    assert not complete_discovery_round(rt, validation)["should_stop"]


def test_batch_reports_capacity_block_as_unmet_target_not_success_or_ordinary_resume(
    rt, monkeypatch,
):
    validation = capacity_blocked_polishing(rt)
    assert complete_discovery_round(rt, validation)["should_stop"]
    target_path = rt.run.run_dir() / "discovery_target.json"
    target_path.write_text(json.dumps({"stop_reason": "review_capacity_exhausted"}))
    monkeypatch.setattr(runner, "load_runtime_from_run", lambda *args: rt)
    monkeypatch.setattr("autofa.session.discovery_target.failure_mode_target_for_runtime",
                        lambda rt: {"minimum_met": False, "n_validated_failure_modes": 1,
                                    "min_failure_modes": 2})
    with pytest.raises(RuntimeError, match="blocked by exhausted review capacity"):
        runner.require_batch_target(rt.run, cfg=rt.config)
    saved = json.loads(target_path.read_text())
    assert saved["blocked"] and not saved["minimum_met"]
    assert saved["stop_reason"] == "review_capacity_exhausted"


def test_three_complete_extra_rounds_after_last_planned_baseline(rt):
    assert close(rt)["status"] == "baseline"
    assert close(rt)["reused"]
    for completed in range(1, 4):
        result = next_round(rt)
        assert result["rounds_without_improve"] == completed
        assert result["should_stop"] == (completed == 3)
        assert result["completed_extra_rounds"] == completed
        assert close(rt)["rounds_without_improve"] == completed
    assert observe_discovery_stall(rt)["should_stop"]


def test_investigations_never_spend_round_patience(rt):
    rt.run.run_id = "run"
    rt.run.round = 10
    for index in range(20):
        add(rt, candidate(f"weak{index}"))
    status = observe_discovery_stall(rt)
    assert status["since_improve"] == 20
    assert not status["should_stop"]
    assert status["round_stagnation"]["rounds_without_improve"] == 0


@pytest.mark.parametrize("receipt", [{"ok": True, "fraction": .3},
                                     {"ok": False, "fraction": 1.0}])
def test_only_full_validation_can_checkpoint_round(rt, receipt):
    result = complete_discovery_round(rt, receipt)
    assert not result["counted"]
    assert not round_stagnation_status(rt)["initialized"]


def test_smaller_detector_improvement_resets_below_the_best_library_score(rt):
    leader = detector(rt, "leader", coverage=.8, precision=1.)
    smaller = detector(rt, "smaller", coverage=.1, precision=.8)
    rt.library.functions = [leader, smaller]
    close(rt)
    assert next_round(rt)["rounds_without_improve"] == 1
    smaller.scores["ppv_at_pi0"] = .85
    result = next_round(rt)
    assert result["improved"]
    assert result["progress_signals"] == ["retained_detector_improvement:smaller"]
    assert result["rounds_without_improve"] == 0


@pytest.mark.parametrize("change", [
    "tiny", "tradeoff", "abstaining_gain", "restore", "description_only",
])
def test_nonprogress_does_not_reset_patience(rt, change):
    fn = detector(rt, "supported", coverage=.2, precision=.9)
    rt.library.functions = [fn]
    close(rt)
    if change == "tiny":
        fn.scores["ppv_at_pi0"] += .000001
    elif change == "tradeoff":
        fn.scores.update(recall=.25, ppv_at_pi0=.89)
    elif change == "abstaining_gain":
        # Abstentions complete the round, but rates on a smaller resolved subset do not compare.
        fn.scores.update(n_items=80, n_scored=80, n_unresolved=20, n_insufficient=20,
                         evaluation_complete=False, assessment_coverage=.8, recall=.25,
                         ppv_at_pi0=.95)
    elif change == "restore":
        saved = deepcopy(fn)
        rt.library.functions.clear()
        assert next_round(rt)["rounds_without_improve"] == 1
        rt.library.functions = [saved]
    else:
        replacement = detector(rt, "new_description")
        # Same predicate with a different public claim is still not a new mode.
        replacement.source = fn.source
        replacement = attach_task_support(replacement)
        replacement.failure_localisations = deepcopy(fn.failure_localisations)
        replacement.scores.update(
            task_recall_corpus_fingerprint=rt._family_failed_task_inventory["corpus_fingerprint"])
        replacement.scores.update(build_task_support(replacement, rt._family_failed_task_inventory))
        replacement.scores["final_review"]["claim_hash"] = (
            replacement.scores["evaluation_claim_hash"])
        rt.library.functions.append(replacement)
    result = next_round(rt)
    assert not result["improved"]
    assert result["rounds_without_improve"] >= 1
    if change == "description_only":  # The duplicate predicate is not counted twice.
        assert result["quality"] == {"good": 1, "good_or_ok": 1}


def test_validated_overlapping_subtype_is_useful_without_additional_coverage(rt):
    rt.library.functions = [detector(rt, "parent")]
    close(rt)
    next_round(rt)
    rt.library.functions.append(detector(rt, "subtype", precision=.6))
    result = next_round(rt)
    assert result["progress_signals"] == ["new_useful_detector:subtype"]
    assert result["quality"] == {"good": 1, "good_or_ok": 2}


@pytest.mark.parametrize("adherence,resets", [(1., True), (.6, False)], ids=["good", "ok-only"])
def test_combined_coverage_resets_when_new_matches_do_not_change_individual_rates(
    rt, adherence, resets,
):
    rt.library.functions = [detector(rt, "one", adherence=adherence),
                            detector(rt, "two", adherence=adherence)]
    close(rt)
    next_round(rt)
    # Same measured per-function scores, less overlap: the library covers more failures.
    rt.library.functions[1] = detector(rt, "two", matches=range(5, 10), adherence=adherence)
    result = next_round(rt)
    # Display coverage counts useful OK detectors; patience compares the Good-or-better union.
    assert result["progress_signals"] == (["good_combined_coverage"] if resets else [])
    assert result["rounds_without_improve"] == (0 if resets else 2)


@pytest.mark.parametrize("progress", ["retained_improvement", "good_promotion", "target_repair"])
def test_dropped_useful_detector_cannot_hide_behind_progress_on_another(rt, progress):
    kept = (_bounded_detector(rt) if progress == "target_repair" else
            detector(rt, "kept", precision=.8,
                     adherence=2 / 3 if progress == "good_promotion" else 1.))
    if progress == "good_promotion":
        _abstain(kept)
    rt.library.functions = [kept, detector(rt, "removed", adherence=.6)]  # OK is still useful.
    close(rt)
    if progress == "retained_improvement":
        kept.scores["ppv_at_pi0"] = .9
    elif progress == "good_promotion":
        kept.scores["behavior_purity"] = 1.
    else:
        kept = _bounded_detector(rt, caught=16, false_matches=3, adherence=2 / 3)
    rt.library.functions = [kept]
    result = next_round(rt)
    assert result["counted"] and not result["improved"]


@pytest.mark.parametrize("change", [
    {"sampled": True}, {"behavior_n_errors": 1}, {"n_scored": 80, "n_items": 80},
], ids=["sampled", "review-errors", "incomplete-census"])
def test_incomplete_validation_preserves_durable_patience_without_consuming_round(rt, change):
    fn = detector(rt, "ready")
    rt.library.functions = [fn]
    close(rt)
    next_round(rt)
    path = rt.run.run_dir() / "discovery_round_stagnation.json"
    saved = path.read_text()
    fn.scores.update(change)
    result = next_round(rt)
    assert result["status"] == "validation_incomplete"
    assert not result["counted"] and not result["should_stop"]
    assert path.read_text() == saved


def test_source_outage_keeps_checkpoint_and_source_change_resets_baseline(rt, monkeypatch):
    close(rt)
    next_round(rt)
    # Replaying the same round on a verified different corpus cannot reuse its receipt.
    rt._family_failed_task_inventory["corpus_fingerprint"] = "verified-replay-corpus"
    replayed = close(rt)
    assert replayed["status"] == "baseline" and not replayed.get("reused")
    path = rt.run.run_dir() / "discovery_round_stagnation.json"
    saved = path.read_text()
    with monkeypatch.context() as patch:
        patch.setattr(round_stagnation, "library_quality_coverage_for_runtime",
                      lambda _, **kwargs: {"corpus_fingerprint": None})
        assert next_round(rt)["status"] == "validation_incomplete"
        assert path.read_text() == saved
    rt._family_failed_task_inventory["corpus_fingerprint"] = "verified-new-corpus"
    result = next_round(rt)
    assert result["status"] == "baseline" and result["rounds_without_improve"] == 0


def test_resume_and_user_off_setting_retain_completed_round_evidence(rt):
    close(rt)
    next_round(rt)
    resumed = SimpleNamespace(**vars(rt))
    assert close(resumed)["reused"]
    assert next_round(resumed)["rounds_without_improve"] == 2
    resumed.config.discovery_terminate_if_stalled = False
    assert not next_round(resumed)["should_stop"]
    resumed.config.discovery_terminate_if_stalled = True
    assert round_stagnation_status(resumed)["should_stop"]


@pytest.mark.parametrize("change", ["missing_source", "running", "busy"])
def test_failed_final_review_operation_never_consumes_round_without_error_count(rt, change):
    fn = detector(rt, "supported")
    rt.library.functions = [fn]
    close(rt)
    next_round(rt)
    rt.run.round += 1
    receipt = fn.scores["final_review"]
    validation = {"ok": True, "fraction": 1.0, "review_unresolved": [{"name": fn.name}]}
    if change == "missing_source":
        receipt.update(status="unavailable", reason="Frozen review evidence is missing.")
        receipt.pop("identity")
    elif change == "running":
        receipt.update(status="running", evidence_unavailable=True)
    result = complete_discovery_round(rt, validation)
    assert result["status"] == "validation_incomplete"
    assert not result["counted"] and result["rounds_without_improve"] == 1


def _review_uncertain(fn):
    """A finished final review whose every verdict needs further evidence."""
    fn.scores["final_review"]["status"] = "unavailable"
    fn.scores.update(behavior_purity=None, behavior_n_uncertain=5,
                     behavior_n_review_incomplete=5)
    for row in fn.scores["behavior_verdicts"]:
        row.update(verdict="uncertain", completion_status="needs_evidence_review")


def test_completed_uncertainty_is_a_finished_round_not_an_endless_pending_operation(rt):
    fn = detector(rt, "unsupported")
    rt.library.functions = [fn]
    close(rt)
    _review_uncertain(fn)
    rt.run.round += 1
    result = complete_discovery_round(rt, {
        "ok": True, "fraction": 1., "review_unresolved": [{"name": fn.name}],
    })
    assert result["counted"] and result["rounds_without_improve"] == 1


def _abstain(fn, n=3):
    fn.scores.update(n_items=100 - n, n_scored=100 - n, n_unresolved=n, n_insufficient=n,
                     evaluation_complete=False, assessment_coverage=(100 - n) / 100)


def _bounded_detector(rt, *, caught=20, false_matches=6, abstained_failed=1,
                      abstained_passed=0, adherence=5 / 9):
    """A complete attempted census with explicit unresolved outcome bounds."""
    fn = detector(rt, "repairable", matches=range(caught), adherence=adherence)
    fn.scores.update(contingency(
        caught, false_matches, 50 - abstained_failed - caught,
        50 - abstained_passed - false_matches,
    ))
    _abstain(fn, abstained_failed + abstained_passed)
    fn.scores["evaluation_complete"] = not (abstained_failed + abstained_passed)
    fn.scores["unresolved_bounds"] = {
        "recall": [caught / 50, (caught + abstained_failed) / 50],
        "fpr": [false_matches / 50, (false_matches + abstained_passed) / 50],
    }
    return fn


def test_validated_repair_toward_good_can_narrow_qualified_coverage_and_support(rt):
    fn = detector(rt, "repairable", matches=range(33), coverage=.059459,
                  precision=.583524, adherence=5 / 9)
    rt.library.functions = [fn]
    assert close(rt)["quality"] == {"good": 0, "good_or_ok": 1}
    assert next_round(rt)["rounds_without_improve"] == 1
    rt.library.functions = [detector(rt, fn.name, matches=range(24), coverage=.043243,
                                     precision=.624286, adherence=2 / 3)]
    result = next_round(rt)
    assert result["progress_signals"] == ["validated_good_target_repair:repairable"]
    assert result["target_repairs"][0]["dimensions"] == ["ppv_at_pi0", "behavior_purity"]
    assert result["target_repairs"][0]["basis"] == "full_corpus"
    assert result["quality"] == {"good": 0, "good_or_ok": 1}
    assert result["rounds_without_improve"] == 0
    assert next_round(rt)["rounds_without_improve"] == 1


def test_full_attempted_abstentions_allow_conservative_target_repair(rt):
    rt.library.functions = [_bounded_detector(rt, adherence=5 / 11)]
    assert close(rt)["quality"]["good_or_ok"] == 0
    rt.library.functions = [_bounded_detector(rt)]
    assert next_round(rt)["progress_signals"] == ["new_useful_detector:repairable"]
    rt.library.functions = [_bounded_detector(rt, caught=16, false_matches=3, adherence=2 / 3)]
    result = next_round(rt)
    assert result["progress_signals"] == ["validated_good_target_repair:repairable"]
    assert result["quality"] == {"good": 0, "good_or_ok": 1}
    saved = result["detectors"][0]
    assert saved["target_metric_basis"] == "worst_case_abstentions"
    assert saved["target_metrics"]["recall"] == pytest.approx(16 / 50)
    assert saved["target_metrics"]["ppv_at_pi0"] < saved["metrics"]["ppv_at_pi0"]
    assert rt.library.functions[0].scores["ppv_at_pi0"] == saved["metrics"]["ppv_at_pi0"]


@pytest.mark.parametrize("change", ["missing_bounds", "resolved_subset"])
def test_abstention_rate_inflation_or_missing_bounds_is_not_target_repair(rt, change):
    if change == "resolved_subset":
        original = _bounded_detector(rt, caught=20, false_matches=10,
                                     abstained_failed=0, adherence=.6)
        revised = _bounded_detector(rt, caught=20, false_matches=5,
                                    abstained_failed=0, abstained_passed=5, adherence=.6)
    else:
        original = _bounded_detector(rt)
        revised = _bounded_detector(rt, caught=16, false_matches=3, adherence=2 / 3)
        revised.scores.pop("unresolved_bounds")
    rt.library.functions = [original]
    close(rt)
    rt.library.functions = [revised]
    result = next_round(rt)
    assert result["counted"] and not result["improved"]


@pytest.mark.parametrize("change", ["coverage", "support", "blocker", "surplus_only"])
def test_target_repair_preserves_good_dimensions_and_other_blockers(rt, change):
    original = dict(matches=range(10), coverage=.06, precision=.8, adherence=.6)
    revised = dict(matches=range(8), coverage=.04, precision=.75, adherence=.7)
    if change == "coverage":
        revised["coverage"] = .01  # Good coverage is strictly greater than 1%.
    elif change == "support":
        revised["matches"] = range(4)
    elif change == "blocker":
        original["precision"] = .6
        revised["precision"] = .59
    else:  # More coverage above the Good floor does not repair the adherence blocker.
        revised = {**original, "coverage": .08}
    rt.library.functions = [detector(rt, "repairable", **original)]
    close(rt)
    rt.library.functions = [detector(rt, "repairable", **revised)]
    result = next_round(rt)
    assert not result["improved"]


@pytest.mark.parametrize("change", ["restore", "oscillate"])
def test_target_repair_cannot_repeat_historical_progress(rt, change):
    original = _bounded_detector(rt)
    revised = _bounded_detector(rt, caught=16, false_matches=3, adherence=2 / 3)
    rt.library.functions = [original]
    close(rt)
    rt.library.functions = [revised]
    assert next_round(rt)["improved"]
    if change == "restore":
        rt.library.functions = [original]
        assert not next_round(rt)["improved"]
        rt.library.functions = [revised]
    else:
        # More support above the Good floor with the same blocking dimensions
        # cannot turn a historical target vector into a new repair.
        rt.library.functions = [_bounded_detector(rt, caught=19, false_matches=4,
                                                  adherence=2 / 3)]
    result = next_round(rt)
    assert result["counted"] and not result["improved"]


def test_first_validated_good_or_better_promotion_resets_despite_abstentions(rt):
    fn = detector(rt, "giant_app_dump_breaks_request", matches=range(30),
                  coverage=.053571, precision=.758711, adherence=2 / 3)
    _abstain(fn)
    rt.library.functions = [fn]
    assert close(rt)["quality"] == {"good": 0, "good_or_ok": 1}
    assert next_round(rt)["rounds_without_improve"] == 1
    assert next_round(rt)["rounds_without_improve"] == 2
    fn.scores.update(recall=.055357, ppv_at_pi0=.814678, behavior_purity=1.)
    result = next_round(rt)
    assert result["quality"] == {"good": 1, "good_or_ok": 1}
    assert result["progress_signals"] == [
        "retained_detector_improvement:giant_app_dump_breaks_request"]
    assert result["rounds_without_improve"] == 0 and not result["should_stop"]
    assert next_round(rt)["rounds_without_improve"] == 1


def test_legacy_raw_rates_do_not_obstruct_a_later_conservative_target_repair(rt):
    rt.library.functions = [_bounded_detector(rt, caught=16, false_matches=3)]
    close(rt)
    path = rt.run.run_dir() / "discovery_round_stagnation.json"
    saved = json.loads(path.read_text())
    for row in saved["known_detectors"]:  # Saved before conservative abstention bounds.
        row.pop("target_metrics")
        row.pop("target_metric_basis")
    path.write_text(json.dumps(saved))
    assert not next_round(rt)["improved"]
    # Higher legacy resolved-subset rates must not veto the supported adherence repair.
    rt.library.functions[0].scores["behavior_purity"] = 2 / 3
    assert next_round(rt)["progress_signals"] == ["validated_good_target_repair:repairable"]


@pytest.mark.parametrize("other_blocker", [False, True], ids=["good-promotion", "target-repair"])
def test_small_gain_that_crosses_shared_good_threshold_is_still_progress(rt, other_blocker):
    if other_blocker:  # Narrow support still blocks Good; adherence crosses with a 0.005 gain.
        before = dict(coverage=.06, precision=.8, adherence=.895, matches=range(4))
        after = dict(coverage=.04, precision=.75, adherence=.9, matches=range(4))
        signal = "validated_good_target_repair:almost_good"
    else:
        before, after = dict(precision=.695), dict(precision=.7)
        signal = "retained_detector_improvement:almost_good"
    rt.library.functions = [detector(rt, "almost_good", **before)]
    assert close(rt)["quality"]["good"] == 0
    rt.library.functions = [detector(rt, "almost_good", **after)]
    result = next_round(rt)
    assert signal in result["progress_signals"]
    assert result["quality"]["good"] == int(not other_blocker)


def test_resolving_abstentions_establishes_a_comparable_baseline_for_later_gains(rt):
    fn = detector(rt, "supported", coverage=.1, precision=.8)
    fn.scores.update(n_items=80, n_scored=80, n_unresolved=20, n_insufficient=20,
                     evaluation_complete=False, assessment_coverage=.8)
    rt.library.functions = [fn]
    assert close(rt)["status"] == "baseline"  # Honest abstentions complete a round.
    fn.scores.update(n_items=100, n_scored=100, n_unresolved=0, n_insufficient=0,
                     evaluation_complete=True, assessment_coverage=1.)
    assert not next_round(rt)["improved"]
    fn.scores["ppv_at_pi0"] = .85
    result = next_round(rt)
    assert result["improved"]
    assert "retained_detector_improvement:supported" in result["progress_signals"]
    assert result["rounds_without_improve"] == 0


@pytest.mark.parametrize("unfinished", ["selected_review", "stale_skip"])
def test_skipped_nonkeepers_do_not_hide_other_unfinished_validation(rt, monkeypatch, unfinished):
    skipped = detector(rt, "weak", precision=.2)
    skipped.scores.update(sampled=True, evaluation_scope_complete=False)
    keeper = detector(rt, "keeper")
    rt.library.functions = [skipped, keeper]
    receipt_is_current = True
    monkeypatch.setattr(
        round_stagnation, "current_final_validation_skip",
        lambda runtime, fn: ({"status": "skipped"}
                             if fn is skipped and receipt_is_current else None),
    )
    assert close(rt)["quality"] == {"good": 1, "good_or_ok": 1}
    if unfinished == "selected_review":
        keeper.scores["final_review"]["status"] = "running"
    else:
        receipt_is_current = False
    result = next_round(rt)
    assert result["status"] == "validation_incomplete"
    assert not result["counted"]
    assert result["rounds_without_improve"] == 0


@pytest.mark.parametrize("changed", ["claim", "source_corpus", "policy"])
def test_real_selective_receipt_finishes_round_until_its_evidence_changes(rt, monkeypatch, changed):
    weak = detector(rt, "sampled_weak", precision=.2)
    weak.scores.update(sampled=True, sample_n=20, sample_frame_n=100,
                       evaluation_context="source-context", n_items=20, n_scored=20)
    weak.scores.pop("final_review")
    rt.library.functions = [weak]
    rt._library_lock = RLock()
    rt.persist_library = lambda: None
    context = "source-context"
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda runtime: context)
    selected = prepare_final_validation(rt)
    assert selected["selected_names"] == []
    assert len(selected["skipped"]) == 1
    result = close(rt)
    assert result["status"] == "baseline"
    assert result["quality"] == {"good": 0, "good_or_ok": 0}
    assert not failure_mode_target_for_runtime(rt, minimum=1)["minimum_met"]
    assert next_round(rt)["rounds_without_improve"] == 1
    # The round consumes the skip disposition, not fabricated full-census evidence.
    assert weak.scores["sampled"] is True and "final_review" not in weak.scores
    if changed == "claim":
        weak.source += "\n# modified detector"
    elif changed == "source_corpus":
        context = "different-source"
    else:  # a new validation standard does not inherit the saved stall count
        rt.config.behavior_strictness = "contextual"
    result = next_round(rt)
    assert not result["counted"]
    assert result["status"] == "validation_incomplete"


def test_unresolved_selection_cannot_reuse_old_substantive_uncertainty(rt, monkeypatch):
    fn = detector(rt, "uncertain")
    fn.scores["evaluation_context"] = "original-source"
    rt.library.functions = [fn]
    assert close(rt)["status"] == "baseline"
    _review_uncertain(fn)
    # Same task IDs, changed retained content. Old inconclusive findings cannot
    # finish the current operation while source verification requests a rescore.
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda runtime: "changed-source")
    selection = prepare_final_validation(rt)
    assert not selection["ok"] and selection["unresolved"][0]["name"] == fn.name
    rt.run.round += 1
    result = complete_discovery_round(rt, {
        "ok": True, "fraction": 1., "final_validation_selection": selection,
        "review_unresolved": selection["unresolved"],
    })
    assert result["status"] == "validation_incomplete"
    assert result["pending_detector_names"] == [fn.name]
    assert not result["counted"]


def test_campaign_retains_stall_history_without_ending_on_round_patience(rt):
    rt.library.functions = [detector(rt, "supported")]
    close(rt)
    rt.run.campaign_policy = validate_campaign_policy({"round_limit": 50})
    for _ in range(rt.config.discovery_stall_rounds + 1):
        result = next_round(rt)
    assert result["rounds_without_improve"] >= rt.config.discovery_stall_rounds
    assert result["termination_deferred_by_campaign"]
    assert not result["should_stop"]
    rt.run.campaign_policy = {}
    assert round_stagnation_status(rt)["should_stop"]
