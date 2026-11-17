# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery patience: stall and plateau accounting, recovery, round limits and MCP removal."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_eval, tools_library
from autofa.mcp.runtime import Runtime, get_runtime, set_runtime
from autofa.metrics.completeness import completed_evaluation
from autofa.session import discovery_stall
from autofa.session.discovery_recovery import update_recovery_assignment
from autofa.session.discovery_stall import (
    discovery_stall_recovery,
    observe_discovery_stall,
    stall_termination_allowed,
)

from support.discovery import add, candidate
from support.discovery import runtime as runtime
from support.judge import runtime as prefilter_runtime  # noqa: F401
from support.judge import submit


def test_weak_additions_and_unknown_reviews_only_trigger_recovery(runtime):
    good_lead = candidate("lead", purity=1.)
    add(runtime, good_lead)
    result = observe_discovery_stall(runtime, initialize=True)
    assert result["best_quality"] == {"good": 0, "good_or_ok": 1}
    for i in range(6):
        fn = candidate(f"weak{i}")
        if i % 2:
            fn.scores.update(behavior_n_no=0, behavior_n_completed=0, behavior_n_uncertain=1,
                             behavior_n_review_incomplete=1, behavior_purity=None)
            fn.scores["behavior_verdicts"][0].update(verdict="uncertain",
                                                   completion_status="needs_evidence_review")
        add(runtime, fn)
        result = observe_discovery_stall(runtime)
        assert not result["should_stop"]
    assert result["since_improve"] == 6
    assert result["completed_attempts"] == 7


def test_stall_never_ends_a_planned_round_even_when_minimum_is_unmet(runtime):
    runtime.run.round = 17  # The last planned round.
    runtime.run.max_rounds = 18
    runtime.run.min_failure_modes = 4
    for index in range(7):
        add(runtime, candidate(f"weak{index}"))
    result = observe_discovery_stall(runtime)
    assert result["since_improve"] == result["completed_attempts"] == 7
    assert result["enabled"] and not result["termination_allowed"]
    assert result["stop_deferred_by_round_budget"] and not result["should_stop"]
    assert result["recovery_assignment"] is not None


def test_first_extra_round_never_inherits_exhausted_investigation_patience(runtime):
    runtime.run.round, runtime.run.max_rounds = 17, 18
    for index in range(7):
        add(runtime, candidate(f"weak{index}"))
    before = observe_discovery_stall(runtime)
    assert not before["should_stop"]
    saved = json.loads((runtime.run.run_dir() / "discovery_stall.json").read_text())
    resumed = SimpleNamespace(**vars(runtime))
    resumed.run.round = 18
    after = observe_discovery_stall(resumed, initialize=True)
    assert not after["should_stop"] and after["termination_allowed"]
    assert after["stop_deferred_by_round_budget"]
    assert after["round_stagnation"]["rounds_without_improve"] == 0
    assert after["since_improve"] == before["since_improve"] == 7
    assert after["completed_now"] == 0
    assert after["completed_attempts"] == before["completed_attempts"]
    final = json.loads((runtime.run.run_dir() / "discovery_stall.json").read_text())
    assert final["processed"] == saved["processed"]
    assert final["quality_generation"] == saved["quality_generation"]


@pytest.mark.parametrize("budget", [
    {}, {"round": 0}, {"round": True, "max_rounds": 18},
], ids=["missing", "missing_max_rounds", "bool_round"])
def test_unknown_ordinary_round_budget_cannot_authorize_stall_termination(budget):
    assert not stall_termination_allowed(SimpleNamespace(run=SimpleNamespace(**budget)))


def test_processing_failure_is_pending_and_recovery_counts_exactly_once(runtime):
    fn = candidate("broken")
    add(runtime, fn)
    fn.scores["behavior_n_errors"] = 1
    fn.scores["behavior_verdicts"][0]["processing_status"] = "error"
    assert observe_discovery_stall(runtime)["completed_attempts"] == 0
    fn.scores["behavior_n_errors"] = 0
    fn.scores["behavior_verdicts"][0]["processing_status"] = "completed"
    assert observe_discovery_stall(runtime)["completed_attempts"] == 1
    assert observe_discovery_stall(runtime)["completed_now"] == 0
    resumed = SimpleNamespace(**vars(runtime))
    assert observe_discovery_stall(resumed, initialize=True)["since_improve"] == 1


@pytest.mark.parametrize("unfinished", ["errors", "pending_review", "replaced_claim"])
def test_errors_pending_reviews_and_replaced_claims_never_count_as_research(runtime, unfinished):
    fn = candidate("broken", reviewed=unfinished != "pending_review")
    if unfinished == "errors":
        fn.scores["errors"] = 1
    add(runtime, fn)
    if unfinished == "replaced_claim":
        # Old history that lacks the current claim is not an invented failure.
        runtime.scores_history[0]["evaluated_claims"][fn.name] = "old-claim"
    assert observe_discovery_stall(runtime)["completed_attempts"] == 0


def test_verified_precision_improvement_resets_without_double_counting(runtime):
    fn = candidate("lead", purity=1., precision=.6)
    add(runtime, fn)
    observe_discovery_stall(runtime)
    for i in range(3):
        add(runtime, candidate(f"weak{i}"))
    assert observe_discovery_stall(runtime)["since_improve"] == 3
    fn.scores["ppv_at_pi0"] = .68
    result = observe_discovery_stall(runtime)
    assert result["improved"]
    assert result["progress_signals"] == ["comparable_search_score"]
    assert result["since_improve"] == 0
    assert result["completed_now"] == 0


def test_breadth_improvement_on_identical_panel_resets_patience(runtime):
    # Match counts are outcomes, not panel identity; keying on them would hide the gain.
    fn = candidate("lead", purity=1.)
    fn.scores["search_panel"] = {
        "version": 1, "available": True, "scope": "fixed_scoring_panel",
        "claim_hash": fn.scores["evaluation_claim_hash"], "run_id": "run",
        "corpus_fingerprint": "corpus", "cohort_fingerprint": "fixed-panel",
        "n_items": 100, "n_failed_tasks": 50, "n_matched_failed_tasks": 5,
        "task_recall": .1, "observed_lower_bound": False,
    }
    add(runtime, fn)
    observe_discovery_stall(runtime)
    add(runtime, candidate("weak"))
    assert observe_discovery_stall(runtime)["since_improve"] == 1
    fn.scores["search_panel"].update(n_matched_failed_tasks=6, task_recall=.12)
    result = observe_discovery_stall(runtime)
    assert result["improved"] and result["since_improve"] == 0
    assert result["progress_signals"] == ["comparable_search_score"]
    assert result["completed_now"] == 0


@pytest.mark.parametrize("cohort", [None, "other-corpus"])
def test_unknown_or_different_cohort_is_not_a_scalar_gain(runtime, cohort):
    fn = candidate("lead", purity=1.)
    add(runtime, fn)
    observe_discovery_stall(runtime)
    add(runtime, candidate("weak"))
    observe_discovery_stall(runtime)
    fn.scores["ppv_at_pi0"] = .68
    fn.scores["task_recall_corpus_fingerprint"] = cohort
    result = observe_discovery_stall(runtime)
    assert not result["improved"]
    assert result["since_improve"] == 1


def test_status_pivots_repeated_scans_and_rename_do_not_create_attempts(runtime):
    fn = candidate("lead", purity=1.)
    add(runtime, fn)
    observe_discovery_stall(runtime)
    runtime.run.round = 99
    runtime.scores_history.append(deepcopy(runtime.scores_history[0]))
    assert observe_discovery_stall(runtime)["completed_attempts"] == 1
    renamed = deepcopy(fn)
    renamed.name = "renamed"
    # Identical detector claim and predicate with a display-only name change.
    add(runtime, renamed)
    assert observe_discovery_stall(runtime)["completed_attempts"] == 1


def test_off_preserves_unlimited_discovery_but_resume_keeps_evidence(runtime):
    runtime.config.discovery_terminate_if_stalled = False
    runtime.config.discovery_plateau_patience = 2
    for i in range(2):
        add(runtime, candidate(f"weak{i}"))
    result = observe_discovery_stall(runtime)
    assert result["since_improve"] == 2 and not result["should_stop"]
    runtime.config.discovery_terminate_if_stalled = True
    result = observe_discovery_stall(runtime, initialize=True)
    assert not result["should_stop"] and result["completed_now"] == 0
    assert result["round_stagnation"]["rounds_without_improve"] == 0


def negative_family(item_id, *, index=-.5, outcome="no_coherent_family"):
    return {
        "stall_history_index": index,
        "stall_corpus_fingerprint": "corpus",
        "survey": {"outcome": outcome, "evidence": [{
            "item_id": item_id, "role": "unrelated", "causal_chain": "Checked distinct operation.",
            "quotes": [{"field": "response_text", "quote": f"source for {item_id}"}],
        }]},
        "assessment": {"outcome": outcome, "notes": "Checked evidence defeats the hypothesis."},
    }


def save_families(runtime, comparisons):
    path = runtime.run.run_dir() / "family_discovery.json"
    runtime._family_failed_task_inventory = {"corpus_fingerprint": "corpus",
                                            "identity_by_item_id": {}}
    path.write_text(json.dumps({"run_id": runtime.run.run_id, "behavior_strictness": "causal",
                               "comparisons": comparisons}))


def test_completed_negative_comparisons_count_once_across_rounds_and_resume(runtime):
    first = negative_family("one")
    first["survey"]["evidence"][0]["quotes"].append(
        {"field": "prompt_text", "quote": "Original requirement."})
    first["survey"]["evidence"].extend(negative_family("two")["survey"]["evidence"])
    save_families(runtime, [first])
    assert observe_discovery_stall(runtime)["since_improve"] == 1
    # Reordered evidence and changed annotations or round are not new work.
    repeated = deepcopy(first)
    repeated["round"] = 99
    repeated["assessment"]["notes"] = "Another summary of the same inspected evidence."
    repeated["survey"]["evidence"].reverse()
    for row in repeated["survey"]["evidence"]:
        row["quotes"].reverse()
        row["role"] = "uncertain"
        row["causal_chain"] = "Reworded explanation, without new inspected evidence."
    save_families(runtime, [first, repeated])
    assert observe_discovery_stall(runtime)["completed_now"] == 0
    resumed = SimpleNamespace(**vars(runtime))
    assert observe_discovery_stall(resumed, initialize=True)["since_improve"] == 1
    save_families(runtime, [first, *[negative_family(str(i)) for i in range(5)]])
    result = observe_discovery_stall(runtime)
    assert result["completed_attempts"] == 6 and not result["should_stop"]
    assert result["best_quality"] == {"good": 0, "good_or_ok": 0}


def test_unfinished_or_unsupported_family_records_do_not_consume_patience(runtime):
    deferred = negative_family("deferred", outcome="deferred")
    incomplete = negative_family("missing")
    incomplete["survey"]["evidence"][0]["quotes"] = []
    save_families(runtime, [deferred, incomplete])
    assert observe_discovery_stall(runtime)["completed_attempts"] == 0


@pytest.mark.parametrize("foreign", ["run", "corpus", "legacy"])
def test_foreign_or_unbound_negative_family_history_is_not_charged(runtime, foreign):
    comparison = negative_family("foreign")
    if foreign == "corpus":
        comparison["stall_corpus_fingerprint"] = "other"
    if foreign == "legacy":
        comparison.pop("stall_corpus_fingerprint")
    save_families(runtime, [comparison])
    if foreign == "run":
        path = runtime.run.run_dir() / "family_discovery.json"
        state = json.loads(path.read_text())
        state["run_id"] = "other-run"
        path.write_text(json.dumps(state))
    assert observe_discovery_stall(runtime)["completed_attempts"] == 0


def test_negative_comparison_then_verified_quality_resets_in_order(runtime):
    save_families(runtime, [negative_family("one", index=-.5)])
    add(runtime, candidate("useful", purity=1.))
    result = observe_discovery_stall(runtime)
    assert result["completed_attempts"] == 2
    assert result["improved"] and result["since_improve"] == 0
    save_families(runtime, [negative_family("one"), negative_family("two", index=.5)])
    assert observe_discovery_stall(runtime)["since_improve"] == 1


def test_recovery_persists_exact_assignment_through_reads_rounds_and_resume(runtime):
    for index in range(3):
        fn = candidate(f"weak{index}")
        fn.scores["behavior_verdicts"][0]["claim_assessments"] = [{
            "requirement_id": "later_recovery", "status": "unknown",
            "unknown_reason": "review_incomplete", "rationale": "Inspect the later reply.",
            "citations": [{"field": "response_text"}],
        }]
        add(runtime, fn)
    first = observe_discovery_stall(runtime)
    assignment = first["recovery_assignment"]
    assert assignment["detector_name"] == "weak2"
    runtime.run.round = 50
    runtime.scores_history.append(deepcopy(runtime.scores_history[-1]))
    (runtime.run.run_dir() / "review_evidence.json").write_text(json.dumps({
        "discovery_item_ids": ["another"], "discovery_task_groups": {},
    }))
    assert observe_discovery_stall(runtime)["recovery_assignment"] == assignment
    resumed = SimpleNamespace(**vars(runtime))
    assert observe_discovery_stall(resumed, initialize=True)["recovery_assignment"] == assignment
    assert discovery_stall_recovery(first)["assignment"] == assignment
    assert first["since_improve"] == 3


def test_completed_linked_attempt_advances_recovery_but_unrelated_work_does_not(runtime):
    for index in range(3):
        add(runtime, candidate(f"weak{index}"))
    first = observe_discovery_stall(runtime)["recovery_assignment"]
    add(runtime, candidate("unrelated"))
    assert observe_discovery_stall(runtime)["recovery_assignment"] == first
    target = first["detector_name"]
    runtime.library.functions = [fn for fn in runtime.library.functions if fn.name != target]
    repaired = candidate("new_predicate")
    repaired.name = target
    add(runtime, repaired)
    result = observe_discovery_stall(runtime)
    next_assignment = result["recovery_assignment"]
    assert next_assignment["id"] != first["id"]
    assert next_assignment["detector_name"] != target
    assert next_assignment["previous_result"]["detector_name"] == target
    assert result["since_improve"] == 5  # An assignment never resets patience.
    assert observe_discovery_stall(runtime)["recovery_assignment"] == next_assignment


def recovery_inventory(runtime):
    runtime._family_failed_task_inventory = {
        "identity_by_item_id": {}, "corpus_fingerprint": "corpus",
        "task_by_failed_item_id": {"a-fail": "a", "b-fail": "b", "c-fail": "c",
                                   "d-fail": "d"},
        "task_by_item_id": {"a-pass": "a", "a-fail": "a", "b-fail": "b",
                            "c-fail": "c", "d-fail": "d"},
    }


def test_prompt_only_negative_does_not_close_check_but_response_evidence_does(runtime):
    recovery_inventory(runtime)
    state = {"records": {}, "processed": []}
    assignment = update_recovery_assignment(runtime, state, active=True)
    # A completed comparison of other evidence does not settle this check.
    save_families(runtime, [negative_family("unrelated-item")])
    assert update_recovery_assignment(runtime, state, active=True) == assignment
    comparison = negative_family(assignment["item_ids"][0])
    comparison["survey"]["evidence"][0]["quotes"][0]["field"] = "prompt_text"
    save_families(runtime, [comparison])
    assert update_recovery_assignment(runtime, state, active=True) == assignment
    comparison["survey"]["evidence"][0]["quotes"][0]["field"] = "response_text"
    save_families(runtime, [comparison])
    next_assignment = update_recovery_assignment(runtime, state, active=True)
    assert next_assignment["id"] != assignment["id"]
    assert next_assignment["previous_result"]["outcome"] == "no_coherent_family"
    assert len(state["recovery_history"]) == 1
    assert update_recovery_assignment(runtime, state, active=True) == next_assignment


def test_unavailable_pending_family_gets_executable_deferral_then_next_check(runtime):
    recovery_inventory(runtime)
    fn = candidate("attachment")
    fn.scores["behavior_verdicts"][0].update(verdict="uncertain", claim_assessments=[{
        "requirement_id": "attachment_delivery", "status": "unknown",
        "unknown_reason": "evidence_unavailable", "rationale": "No upload delivery record.",
    }])
    add(runtime, fn)
    family = {"run_id": "run", "phase": "assess_family", "comparisons": [],
              "survey": {"family_name": "missing_attachment", "detector_name": "attachment",
                         "evidence": [{"item_id": "a-fail"}],
                         "mechanism": "Attachment was lost before the model call."}}
    path = runtime.run.run_dir() / "family_discovery.json"
    path.write_text(json.dumps(family))
    state = {"records": {}, "processed": []}
    assignment = update_recovery_assignment(runtime, state, active=True)
    action = assignment["next_action"]
    assert action["arguments"]["action"] == "defer_family"
    family["phase"] = "survey"
    family["comparisons"] = [{"survey": family["survey"], "assessment": {
        "outcome": "deferred", "blocker": action["arguments"]["blocker"],
        "next_step": action["arguments"]["next_step"],
    }}]
    path.write_text(json.dumps(family))
    next_assignment = update_recovery_assignment(runtime, state, active=True)
    assert next_assignment["previous_result"]["outcome"] == "deferred"
    assert state["processed"] == []  # Deferral cannot invent a counted investigation.


@pytest.mark.parametrize("change", ["claim", "policy", "corpus"])
def test_pending_recovery_is_superseded_when_its_evidence_context_changes(runtime, change):
    recovery_inventory(runtime)
    fn = candidate("old_target")
    add(runtime, fn)
    state = {"records": {}, "processed": []}
    assignment = update_recovery_assignment(runtime, state, active=True)
    assert assignment["detector_name"] == "old_target"
    if change == "claim":
        fn.mechanism = "A different mechanism requiring different evidence."
    elif change == "policy":
        state["recovery_assignment"]["baseline_review_policy"] = "earlier-policy"
        fn.scores["judge_policy_hash"] = "earlier-policy"
    else:
        runtime._family_failed_task_inventory["corpus_fingerprint"] = "different-corpus"
    updated = update_recovery_assignment(runtime, state, active=True)
    assert updated["id"] != assignment["id"]
    assert updated["previous_result"]["outcome"] == "superseded"
    assert state["processed"] == []
    assert len(state["recovery_history"]) == 1
    assert update_recovery_assignment(runtime, state, active=True) == updated


def test_implementation_only_repair_retains_pending_counterexample(runtime):
    fn = candidate("repair_me")
    add(runtime, fn)
    state = {"records": {}, "processed": []}
    assignment = update_recovery_assignment(runtime, state, active=True)
    fn.source = "def repair_me(item): return False"
    assert update_recovery_assignment(runtime, state, active=True) == assignment
    assert not state["recovery_history"]


def assessment(since_improve, *, patience=6, enabled=True, **extra):
    return {"enabled": enabled, "since_improve": since_improve, "patience": patience,
            "should_stop": enabled and since_improve >= patience, **extra}


@pytest.mark.parametrize("patience,threshold", [(1, None), (5, 3), (6, 3)])
def test_recovery_window_preserves_a_strict_pretermination_boundary(patience, threshold):
    active = [n for n in range(patience + 2)
              if discovery_stall_recovery(assessment(n, patience=patience)) is not None]
    assert active == ([] if threshold is None else list(range(threshold, patience)))


@pytest.mark.parametrize("state", [
    {}, assessment(3, should_stop=True), assessment(3, enabled=False),
], ids=["missing", "terminal", "disabled"])
def test_disabled_missing_or_terminal_assessment_does_not_schedule_work(state):
    assert discovery_stall_recovery(state) is None


@pytest.mark.usefixtures("prefilter_runtime")
def test_rejected_evaluation_survives_same_turn_removal():
    assert submit()["ok"]
    path = get_runtime().run.run_dir() / "discovery_stall.json"
    assert json.loads(path.read_text())["completed_attempts"] == 1
    assert tools_library.remove_function("guess", reason="Rejected association")["ok"]
    observed = observe_discovery_stall(get_runtime())
    assert observed["completed_attempts"] == observed["since_improve"] == 1
    assert observed["completed_now"] == 0


@pytest.mark.usefixtures("prefilter_runtime")
def test_committed_evaluation_survives_stall_checkpoint_failure(monkeypatch):
    def disk_error(_runtime, **kwargs):
        raise OSError("checkpoint unavailable")

    monkeypatch.setattr(discovery_stall, "observe_discovery_stall", disk_error)
    assert submit()["ok"]
    assert get_runtime().working_library_path().is_file()
    assert tools_eval.evaluate_functions(names=["guess"], mode="full")["ok"]


# Pareto status plateau patience, soft goal_met and the round limit.

def _rt_with_history(
    history: list[dict], *, round_idx: int = 1, max_rounds: int = 3,
    wait_for_plateau: bool = True,
) -> Runtime:
    cfg = Config(behavior_strictness="causal", discovery_plateau_patience=6,
                 discovery_wait_for_plateau=wait_for_plateau)
    # No failure-mode minimum: these fixtures exercise plateau accounting only.
    run = RunConfig(run_id="test-discovery-plateau", parquet_paths=[], model_id="m",
                    benchmarks=["b"], round=round_idx, max_rounds=max_rounds,
                    min_failure_modes=0)
    lib = FunctionLibrary(model_id="m", benchmark="b", functions=[FeatureFunction(
        name="a", description="d", source="def a(item, llm_client=None):\n    return False\n",
        scores={"ppv_at_pi0": 0.5, "recall": 0.2, "behavior_purity": 0.8},
    )])
    ds = Dataset(parquet_paths=[], model_id="m", benchmarks=["b"])
    rt = Runtime(dataset=ds, library=lib, run=run, config=cfg, scores_history=history)
    set_runtime(rt)
    return rt


@pytest.fixture
def isolated_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    def snapshot(rt, **kwargs):
        # Detector a's selected rows are scored once its latest evaluation completed.
        latest = rt.scores_history[-1] if rt.scores_history else {}
        return {
            "available": True, "corpus_fingerprint": "test-corpus",
            "n_failed_tasks": 10, "n_confirmed_failed_tasks": 0,
            "confirmed_task_coverage": 0.0, "detectors": [{
                "name": "a", "current_claim_hash": "claim-a", "review_complete": True,
                "selected_evaluation_complete": latest.get("evaluation_complete", False),
            }],
        }

    monkeypatch.setattr("autofa.session.task_progress.task_coverage_snapshot", snapshot)


def _evaluation(*, complete=True):
    return {"points": [[0.5, 0.3, 0.8, 0.7]], "frontier_value": 0.575,
            "hypervolume": 0.1, "n_functions": 1, "results": [{"name": "a"}],
            "evaluated_names": ["a"], "evaluated_claims": {"a": "claim-a"},
            "evaluation_complete": complete}


@pytest.mark.usefixtures("isolated_progress")
@pytest.mark.parametrize("round_idx,flat,wait,goal_met", [
    (1, 2, True, False),   # Two flat evaluations are not a plateau.
    (10, 6, True, True),   # Patience exhausted after the planned rounds.
    (1, 8, False, False),  # Regular mode never stops before the round limit.
])
def test_flat_evaluations_meet_goal_only_after_patience_and_round_limit(
    round_idx, flat, wait, goal_met,
):
    # Frontier history saved before the coverage checkpoint is not task progress.
    rt = _rt_with_history([_evaluation() for _ in range(20)], round_idx=round_idx,
                          max_rounds=3, wait_for_plateau=wait)
    baseline = tools_library.pareto_status()  # Establish the task-coverage baseline.
    assert baseline["coverage_baseline_initialized"] is True
    assert baseline["goal_met"] is False
    assert baseline["experiments_since_improve"] == 0
    for _ in range(flat):
        rt.scores_history.append(_evaluation())
        out = tools_library.pareto_status()
    assert out["goal_met"] is goal_met
    assert out["experiments_since_improve"] == flat
    # Pure status reads do not consume more patience.
    assert tools_library.pareto_status()["experiments_since_improve"] == flat


@pytest.mark.usefixtures("isolated_progress")
def test_pareto_status_does_not_call_partial_evidence_a_plateau():
    rt = _rt_with_history([], round_idx=10)
    tools_library.pareto_status()
    rt.scores_history.append(_evaluation(complete=False))
    out = tools_library.pareto_status()
    assert out["goal_met"] is False
    assert out["progress_blocked"] is True
    assert out["experiments_since_improve"] == 0


@pytest.mark.usefixtures("isolated_progress")
@pytest.mark.parametrize("round_idx,expected,blocked", [
    (2, False, False), (3, True, False), (3, True, True),
])
def test_regular_off_stops_at_round_limit_without_plateau(
    round_idx, expected, blocked, monkeypatch,
):
    rt = _rt_with_history([], round_idx=round_idx, max_rounds=3, wait_for_plateau=False)
    tools_library.pareto_status()
    if blocked:
        rt.scores_history.append(_evaluation(complete=False))
        monkeypatch.setattr(tools_library, "_family_search_status",
                            lambda rt: {"synthesis_pending": True})
    out = tools_library.pareto_status()
    assert out["discovery_wait_for_plateau"] is False
    assert out["goal_met"] is expected
    assert out["experiments_since_improve"] == 0
    if blocked:
        assert out["progress_blocked"] is True


@pytest.mark.usefixtures("isolated_progress")
def test_partial_evidence_is_not_a_comparable_frontier_evaluation():
    scores = {"errors": 0, "n_insufficient": 572, "assessment_coverage": 0.8856,
              "evaluation_complete": False}
    assert not completed_evaluation({"results": [{"scores": scores}]})
    assert not completed_evaluation({"results": [{"scores": {"errors": 1}}]})
    assert not completed_evaluation({"results": [{"scores": {"parent_errors": 1}}]})
    assert not completed_evaluation({"eval_mode": "capped"})
    assert not completed_evaluation({"results": [{"sampled": True}]})
    assert completed_evaluation({"eval_mode": "full", "results": [{"scores": {"errors": 0}}]})
