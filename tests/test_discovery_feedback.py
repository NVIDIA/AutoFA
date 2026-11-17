# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery grades, their advice and agent goals follow current evidence through MCP tools."""

from copy import deepcopy

import pytest

from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.mcp import tools_eval, tools_library
from autofa.metrics import claude_judge, discovery_feedback
from autofa.metrics.discovery_feedback import build_discovery_feedback
from autofa.metrics.discovery_grades import (
    grade_for_metrics,
    grade_for_scores,
    grade_gap_for_scores,
    grade_summary,
)
from autofa.metrics.judge_evidence import make_snapshot
from autofa.metrics.review_scheduling import automatic_review_schedule
from autofa.metrics.verdicts import extract_metrics, verdict_for_metrics, verdict_for_scores
from autofa.reports.failure_cards import build_data
from autofa.session.family_discovery import initialize_family_discovery

from support.discovery import _function
from support.families import complete_small_corpus_round
from support.repair_review import _result, _runtime
from support.task_support import _function as _state_function
from support.task_support import (
    _ranked_function,
    attach_sampled_task_support,
    attach_task_support,
)


def _scores(**overrides):
    return {
        "evaluated": True, "evaluation_scope_complete": True,
        "ppv_at_pi0": 0.9, "behavior_purity": 1.0, "task_recall": 0.04,
        "recall": 0.8, "task_recall_scope": "full_selected_corpus",
        "n_matched_failed_tasks": 2, "n_failed_tasks": 50,
        **overrides,
    }


def _with_verified_task_support(scores, task_n):
    fn = FeatureFunction(name="breadth", source="def detect(item): return True",
                         description="Observable unfinished work")
    fn.scores = scores
    return attach_task_support(fn, task_n=task_n).scores


def _gap_dimensions(result):
    return {dimension["key"]: dimension for dimension in result["dimensions"]}


def _feedback(scores=None, **validation):
    return build_discovery_feedback(
        _scores() if scores is None else scores,
        judge_feedback={"status": "confirmed", "uncertain": 0, "processing_errors": 0,
                        **validation},
    )


# Agent grades: strict breadth tiers above the shared public Good floors.


@pytest.mark.parametrize("coverage,code", [
    (.01, "OK"), (.01001, "GOOD"), (.10001, "EXCELLENT"), (.20001, "OUTSTANDING"),
])
def test_agent_breadth_grade_boundaries_leave_public_good_unchanged(coverage, code):
    metrics = {"recall": coverage, "ppv_at_pi0": .7, "behavior_purity": .9,
               "n_matched_failed_tasks": 5}
    original = deepcopy(metrics)
    assert grade_for_metrics(metrics)["code"] == code
    assert verdict_for_metrics(metrics)["code"] == ("OK" if code == "OK" else "GOOD")
    assert metrics == original


@pytest.mark.parametrize("changes,expected", [
    ({"ppv_at_pi0": .6999}, "OK"), ({"ppv_at_pi0": .3999}, "WEAK"),
    ({"n_matched_failed_tasks": 4}, "OK"), ({"behavior_purity": None}, "UNSCORED"),
    ({"n_flagged": 0}, "EMPTY"),
])
def test_high_coverage_does_not_bypass_shared_quality_requirements(changes, expected):
    metrics = {"recall": .9, "ppv_at_pi0": .9, "behavior_purity": 1,
               "n_matched_failed_tasks": 5, **changes}
    assert grade_for_metrics(metrics) == verdict_for_metrics(metrics)
    assert grade_for_metrics(metrics)["code"] == expected


def test_grade_scores_derive_verified_support_instead_of_trusting_saved_count():
    fn = _function("repeated", .9, tasks=2, repeats=5)
    fn.scores["n_matched_failed_tasks"] = 10
    assert grade_for_scores(fn.scores)["code"] == "OK"
    fn = _function("supported", .9)
    assert grade_for_scores(fn.scores)["code"] == "OUTSTANDING"
    assert verdict_for_scores(fn.scores)["code"] == "GOOD"
    # Malformed saved review counts cannot create a grade.
    fn.scores["behavior_n_judged"] = "not a count"
    assert grade_for_scores(fn.scores)["code"] == "UNSCORED"


def test_library_summary_resolves_current_claims_and_omits_review_payloads():
    functions = [_function("good", .04), _function("outstanding", .3),
                 _function("excellent", .15), _function("unsupported", .9, tasks=2),
                 _function("changed_claim", .8)]
    functions[-1].description = "Different claim"
    before = [deepcopy(fn.to_dict()) for fn in functions]
    result = grade_summary(functions, behavior_strictness="causal")
    assert result["counts"] == {
        "OUTSTANDING": 1, "EXCELLENT": 1, "GOOD": 1, "OK": 1,
        "WEAK": 0, "EMPTY": 0, "UNSCORED": 1,
    }
    assert (result["n_good_or_better"], result["n_outstanding"]) == (3, 1)
    assert [row["name"] for row in result["functions"]] == [
        "outstanding", "excellent", "good", "unsupported", "changed_claim",
    ]
    assert all("behavior_verdicts" not in row for row in result["functions"])
    assert [fn.to_dict() for fn in functions] == before
    assert grade_summary(before, behavior_strictness="causal") == result


def test_sampled_agent_grades_compare_projected_support_without_claiming_observed_tasks():
    metrics = {"recall": .3, "ppv_at_pi0": .9, "behavior_purity": 1,
               "n_matched_failed_tasks": 2, "estimated_n_matched_failed_tasks": 7.5,
               "task_support_comparison_basis": "estimated",
               "task_support_estimate_status": "provisional"}
    assert grade_for_metrics(metrics)["code"] == "OUTSTANDING"
    assert verdict_for_metrics(metrics)["code"] == "GOOD"


@pytest.mark.parametrize("changes", [
    {"ppv_at_pi0": .5}, {"estimated_n_matched_failed_tasks": 4.99},
    {"task_support_estimate_status": "unavailable"},
])
def test_projected_task_support_does_not_relax_other_grading_requirements(changes):
    metrics = {"recall": .3, "ppv_at_pi0": .9, "behavior_purity": 1,
               "n_matched_failed_tasks": 2, "estimated_n_matched_failed_tasks": 7.5,
               "task_support_comparison_basis": "estimated",
               "task_support_estimate_status": "provisional", **changes}
    assert grade_for_metrics(metrics)["code"] == "OK"


def test_agent_grade_summary_uses_source_backed_projection_and_preserves_observed_count():
    fn = _function("sampled", .3, tasks=2)
    attach_sampled_task_support(fn, task_n=2, full_task_n=20, panel_task_n=5)
    row = grade_summary([fn])["functions"][0]
    assert row["code"] == "OUTSTANDING"
    assert row["n_confirmed_failed_tasks"] == 2
    assert row["estimated_n_confirmed_failed_tasks"] == pytest.approx(8)
    assert row["task_support_comparison_basis"] == "estimated"


# Grade gaps name the measured bottlenecks; unavailable evidence is never a zero.


@pytest.mark.parametrize("changes,blocked", [
    ({"ppv_at_pi0": .5}, "ppv_at_pi0"),
    ({"behavior_purity": .6}, "behavior_purity"),
    ({"recall": .01}, "recall"),
])
def test_next_good_grade_names_exactly_one_measured_bottleneck(changes, blocked):
    scores = _function("single_gap", .04).scores
    scores.update(changes)
    result = grade_gap_for_scores(scores)
    assert (result["current_grade"], result["next_grade"]) == ("OK", "GOOD")
    assert result["bottlenecks"] == [blocked]
    dimensions = _gap_dimensions(result)
    assert dimensions[blocked]["met"] is False
    assert dimensions[blocked]["current"] == changes[blocked]
    assert all(dimension["met"] for key, dimension in dimensions.items() if key != blocked)


def test_missing_source_breadth_names_support_without_counting_review_confirmations():
    scores = _function("few_tasks", .04, tasks=4).scores
    scores["behavior_verdicts"] = scores["behavior_verdicts"][:1]
    scores["n_confirmed_failed_tasks"] = 100
    result = grade_gap_for_scores(scores)
    assert result["bottlenecks"] == ["matched_failed_tasks"]
    support = _gap_dimensions(result)["matched_failed_tasks"]
    assert support["current"] == support["observed"] == 4
    scores.pop("task_support")
    scores["n_matched_failed_tasks"] = 100
    support = _gap_dimensions(grade_gap_for_scores(scores))["matched_failed_tasks"]
    assert support["current"] is None and support["met"] is None
    assert support["status"] == "unavailable"


def test_weak_to_ok_lists_every_blocking_dimension_with_correct_boundaries():
    scores = _function("all_gaps", .005).scores
    scores.update(ppv_at_pi0=.39, behavior_purity=.49)
    result = grade_gap_for_scores(scores)
    assert (result["current_grade"], result["next_grade"]) == ("WEAK", "OK")
    assert result["bottlenecks"] == ["recall", "ppv_at_pi0", "behavior_purity"]
    assert _gap_dimensions(result)["matched_failed_tasks"]["met"] is True
    scores.update(ppv_at_pi0=.4, behavior_purity=.5, recall=.00501)
    assert grade_gap_for_scores(scores)["current_grade"] == "OK"
    # The Good precision and adherence floors are inclusive too.
    scores.update(ppv_at_pi0=.7, behavior_purity=.9, recall=.01)
    result = grade_gap_for_scores(scores)
    assert result["bottlenecks"] == ["recall"]
    assert _gap_dimensions(result)["ppv_at_pi0"]["met"] is True
    assert _gap_dimensions(result)["behavior_purity"]["met"] is True


@pytest.mark.parametrize("coverage,current,target,required", [
    (.10, "GOOD", "EXCELLENT", .10),
    (.10001, "EXCELLENT", "OUTSTANDING", .20),
    (.20, "EXCELLENT", "OUTSTANDING", .20),
    (.20001, "OUTSTANDING", None, .20),
])
def test_agent_grade_upgrade_uses_strict_coverage_and_retains_quality_floors(
    coverage, current, target, required,
):
    result = grade_gap_for_scores(_function("breadth_gap", coverage).scores)
    recall = _gap_dimensions(result)["recall"]
    assert (result["current_grade"], result["next_grade"]) == (current, target)
    assert result["bottlenecks"] == (["recall"] if target else [])
    assert (recall["required"], recall["operator"]) == (required, ">")


@pytest.mark.parametrize("field,value", [("recall", None), ("ppv_at_pi0", float("nan"))])
def test_unavailable_dimension_stays_none_instead_of_becoming_a_zero(field, value):
    scores = _function("unknown", .04).scores
    scores[field] = value
    result = grade_gap_for_scores(scores)
    dimension = _gap_dimensions(result)[field]
    assert result["current_grade"] == "UNSCORED"
    assert dimension["current"] is None and dimension["met"] is None
    assert dimension["status"] == "unavailable"
    assert field in result["bottlenecks"]


def test_missing_or_stale_evaluation_does_not_reuse_historical_rates():
    scores = _function("stale", .04).scores
    scores["evaluation_stale"] = True
    result = grade_gap_for_scores(scores)
    dimensions = _gap_dimensions(result)
    assert result["current_grade"] == "UNSCORED"
    for key in ("recall", "ppv_at_pi0", "matched_failed_tasks"):
        assert dimensions[key]["current"] is None
    assert grade_gap_for_scores({})["current_grade"] == "UNSCORED"


def test_unresolved_review_is_an_evidence_gap_instead_of_measured_bad_adherence():
    scores = _function("uncertain_review", .04).scores
    scores.update(
        behavior_purity=None, behavior_n_judged=1, behavior_n_completed=0,
        behavior_n_yes=0, behavior_n_no=0, behavior_n_uncertain=1,
    )
    result = grade_gap_for_scores(scores)
    adherence = _gap_dimensions(result)["behavior_purity"]
    assert result["current_grade"] == "UNSCORED"
    assert adherence["current"] is None and adherence["met"] is None


def test_incomplete_review_preserves_its_actual_measured_adherence_aggregate():
    scores = _function("partial_review", .04).scores
    scores.update(
        behavior_purity=.67, behavior_n_judged=3, behavior_n_completed=2,
        behavior_n_yes=2, behavior_n_no=0, behavior_n_uncertain=1,
    )
    result = grade_gap_for_scores(scores)
    adherence = _gap_dimensions(result)["behavior_purity"]
    assert result["current_grade"] == "OK"
    assert result["bottlenecks"] == ["behavior_purity"]
    assert (adherence["current"], adherence["met"]) == (.67, False)


def test_historical_adherence_cannot_clear_the_current_review_requirement_for_good():
    scores = _function("historical_review", .04).scores
    scores.pop("judge_contract_version")
    result = grade_gap_for_scores(scores)
    assert (result["current_grade"], result["next_grade"]) == ("OK", "GOOD")
    assert result["bottlenecks"] == ["behavior_purity"]
    assert _gap_dimensions(result)["behavior_purity"]["current"] is None


def test_validated_sample_projection_meets_support_without_five_reviews():
    fn = _function("sample_gap", .04, tasks=2)
    attach_sampled_task_support(fn, task_n=2)
    fn.scores["behavior_verdicts"] = fn.scores["behavior_verdicts"][:1]
    result = grade_gap_for_scores(fn.scores)
    support = _gap_dimensions(result)["matched_failed_tasks"]
    assert (result["current_grade"], result["next_grade"]) == ("GOOD", "EXCELLENT")
    assert result["bottlenecks"] == ["recall"]
    assert result["provisional"] is True
    assert support["observed"] == 2
    assert support["current"] == support["estimated"] == pytest.approx(8)
    assert support["met"] is True


def test_full_scan_with_unresolved_items_is_provisional_without_claiming_sampling():
    scores = _function("unresolved_full_scan", .04).scores
    scores.update(sampled=False, metrics_scope="resolved_items", n_insufficient=2)
    assert grade_gap_for_scores(scores, target_grade="Outstanding")["provisional"] is True


@pytest.mark.parametrize("changes,tasks,blocked", [
    ({"recall": .15}, 5, ["recall"]),
    ({}, 4, ["matched_failed_tasks"]),
    ({"recall": .1, "ppv_at_pi0": .5, "behavior_purity": .6}, 4,
     ["recall", "ppv_at_pi0", "behavior_purity", "matched_failed_tasks"]),
])
def test_explicit_outstanding_target_identifies_all_and_only_its_bottlenecks(
    changes, tasks, blocked,
):
    scores = _function("outstanding_target", .3, tasks=tasks).scores
    scores.update(changes)
    result = grade_gap_for_scores(scores, target_grade="OUTSTANDING")
    assert result["target_grade"] == "OUTSTANDING"
    assert result["target_met"] is False
    assert result["bottlenecks"] == blocked


def test_already_met_explicit_target_does_not_force_another_upgrade():
    result = grade_gap_for_scores(_function("target_met", .3).scores, target_grade="OUTSTANDING")
    assert result["target_met"] is True
    assert result["bottlenecks"] == result["next_steps"] == []


def test_invalid_grade_target_is_explicit_instead_of_silently_inventing_requirements():
    with pytest.raises(ValueError, match="target_grade"):
        grade_gap_for_scores({}, target_grade="WEAK")


# Feedback status: evaluation and review problems take priority over quality scores.


@pytest.mark.parametrize("precision,purity,status", [
    (0.7, 0.9, "choose_next_investigation"),
    (0.69, 0.95, "inspect_boundaries"),
    (0.99, 0.89, "inspect_causal_claim"),
])
def test_quality_scores_choose_investigation_instead_of_unconditional_broadening(
    precision, purity, status,
):
    assert _feedback(_scores(ppv_at_pi0=precision, behavior_purity=purity))["status"] == status


@pytest.mark.parametrize("overrides,status", [
    ({"evaluation_stale": True}, "evaluate_current"),
    ({"parent_errors": 1}, "recover_evaluation"),
    ({"n_insufficient": 30}, "inspect_evidence"),
    ({"eval_truncated": True, "evaluation_scope_complete": False},
     "inspect_evaluation_scope"),
])
def test_evaluation_problems_take_priority_over_quality_scores(overrides, status):
    result = _feedback(_scores(ppv_at_pi0=0.1, behavior_purity=0.1, **overrides))
    assert result["status"] == status


@pytest.mark.parametrize("validation", [
    {"status": "uncertain"}, {"status": "error"},
    {"status": "mixed", "processing_errors": 1},
])
def test_uncertain_or_failed_review_is_not_low_quality(validation):
    assert _feedback(_scores(behavior_purity=0.1), **validation)["status"] == (
        "inspect_review_evidence")


def test_unread_evidence_is_not_overridden_by_confirmed_summary():
    result = _feedback(_scores(behavior_verdicts=[
        {"verdict": "uncertain", "completion_status": "needs_evidence_review"},
    ]))
    assert result["status"] == "inspect_review_evidence"


def test_two_confirmations_and_one_counterexample_prioritize_repair_not_more_reviews():
    scores = _scores(behavior_purity=2 / 3, behavior_verdicts=[
        {"item_id": "supported-a", "verdict": "yes"},
        {"item_id": "supported-b", "verdict": "yes"},
        {"item_id": "counterexample", "verdict": "no", "claim_assessments": [
            {"requirement_id": "claim", "status": "contradicted",
             "rationale": "The answer is correct; the claimed mistake did not occur."},
        ]},
    ])
    feedback = _feedback(scores, rejected=1, confirmed=2, sampled=3)
    assert feedback["status"] == "inspect_causal_claim"
    assert "counterexample" in feedback["issue_types"]


def test_unknown_third_review_is_not_a_counterexample_or_processing_retry():
    scores = _scores(behavior_purity=2 / 3, behavior_verdicts=[
        {"item_id": "uncertain", "verdict": "uncertain", "claim_assessments": [
            {"requirement_id": "claim", "status": "unknown",
             "unknown_reason": "ambiguous_evidence"},
        ]},
    ])
    feedback = _feedback(scores, uncertain=1, confirmed=2, sampled=3)
    assert feedback["status"] == "inspect_review_evidence"
    assert feedback["issue_types"] == ["ambiguous_evidence"]


@pytest.mark.parametrize("reason,issue", [
    ("evidence_unavailable", "missing_evidence"), (None, "unresolved_review"),
])
def test_structured_unknown_reasons_get_distinct_actions(reason, issue):
    assessment = {"status": "unknown", "unknown_reason": reason,
                  "rationale": "A parsing error, missing contract, and counterexample."}
    scores = _scores(behavior_purity=0, behavior_verdicts=[
        {"verdict": "uncertain", "claim_assessments": [assessment]},
    ])
    result = _feedback(scores, status="uncertain", uncertain=1)
    assert result["issue_types"] == [issue]
    assert result["status"] == "inspect_review_evidence"


def test_detector_execution_and_review_processing_failures_have_different_remedies():
    errors = [{"item_id": "case", "error": "KeyError: arguments"}]
    detector = _feedback(_scores(errors=1, error_examples=errors))
    reviewer = _feedback(status="error", processing_errors=1)
    assert detector["issue_types"] == ["detector_execution_failure"]
    assert detector["error_examples"] == errors
    assert reviewer["issue_types"] == ["review_processing_failure"]


def test_missing_current_review_does_not_use_old_purity_for_generalization():
    assert _feedback(status="stale")["status"] == "review_current"


@pytest.mark.parametrize("overrides,review_status", [
    ({}, "legacy"), ({"behavior_n_judged": "not a count"}, "invalid"),
])
def test_purity_without_current_contract_or_valid_counts_needs_current_review(
    overrides, review_status,
):
    result = build_discovery_feedback(_scores(**overrides))
    assert result["status"] == "review_current"
    assert result["signals"]["review_status"] == review_status


def test_intentional_scoring_sample_is_not_an_execution_failure():
    result = _feedback(_scores(sampled=True, evaluation_scope_complete=False, eval_truncated=True))
    assert result["status"] == "inspect_evaluation_scope"


@pytest.mark.parametrize("overrides", [
    {"task_recall": None}, {"task_recall": 1.1},
    {"task_recall_scope": "unavailable", "task_recall_reason": "No source identity inventory."},
], ids=["absent", "invalid", "unavailable_scope"])
def test_absent_or_invalid_task_metric_does_not_fall_back_to_attempt_recall(overrides):
    result = _feedback(_scores(**overrides))
    assert result["status"] == "inspect_task_identity"
    assert result["signals"]["task_recall"] is None


def test_invalid_quality_metric_is_not_read_as_poor_quality():
    assert _feedback(_scores(behavior_purity=float("nan")))["status"] == "inspect_scores"


def test_feedback_does_not_mutate_scores_or_existing_detailed_judge_feedback():
    scores = _scores()
    judge = {"status": "mixed", "next_steps": ["Inspect exact requirement X on task T."]}
    before = deepcopy((scores, judge))
    build_discovery_feedback(scores, judge_feedback=judge)
    assert (scores, judge) == before


def _partially_resolved_scores(**overrides):
    return _scores(
        n_scored=1341, n_insufficient=21, evaluation_scope_complete=False,
        task_recall_scope="observed_lower_bound",
        evaluation_claim_hash="claim", evaluation_run_id="run",
        task_recall_corpus_fingerprint="corpus",
        search_panel={
            "version": 1, "available": True, "scope": "fixed_scoring_panel",
            "claim_hash": "claim", "run_id": "run", "corpus_fingerprint": "corpus",
            "cohort_fingerprint": "panel", "n_items": 200,
            "observed_lower_bound": True,
            "n_failed_tasks": 50, "n_matched_failed_tasks": 2, "task_recall": 0.04,
        },
    ) | overrides


@pytest.mark.parametrize("overrides,status,issues", [
    ({}, "choose_next_investigation", ["detector_abstention"]),
    ({"n_scored": 0}, "inspect_evidence", None),
    ({"errors": 1}, "recover_evaluation", ["detector_execution_failure"]),
], ids=["some_abstentions", "no_resolved_predictions", "execution_errors"])
def test_abstentions_keep_supported_breadth_without_hiding_evaluation_problems(
    overrides, status, issues,
):
    scores = _partially_resolved_scores(**overrides)
    before = deepcopy(scores)
    result = _feedback(scores)
    assert result["status"] == status
    if issues:
        assert result["issue_types"] == issues
    assert scores == before


def test_extrapolation_cannot_override_rejected_behavior_or_review_errors(monkeypatch):
    # Isolate steering from the estimator's separate sampling-design tests.
    monkeypatch.setattr(discovery_feedback, "task_support_assessment", lambda scores: {
        "n_confirmed_failed_tasks": 2, "n_matched_failed_tasks": 4,
        "estimated_n_confirmed_failed_tasks": 8.0, "estimated_n_matched_failed_tasks": 8.0,
        "task_support_comparison": 8.0, "task_support_comparison_basis": "estimated",
        "task_support_estimate_status": "provisional",
        "n_sampled_failed_tasks": 20, "n_sampled_confirmed_failed_tasks": 2,
    })
    scores = _partially_resolved_scores(sampled=True, behavior_purity=.5)
    assert _feedback(scores)["status"] == "inspect_causal_claim"
    result = _feedback(scores, status="error", processing_errors=1)
    assert result["status"] == "inspect_review_evidence"


@pytest.mark.parametrize("change,status", [
    ({"ppv_at_pi0": .5}, "inspect_boundaries"), ({"errors": 1}, "recover_evaluation"),
])
def test_directional_grade_gap_is_attached_even_on_early_repair_paths(change, status):
    scores = _with_verified_task_support(_scores(recall=.04, **change), task_n=5)
    result = _feedback(scores)
    assert result["status"] == status
    assert result["grade_gap_summary"] == result["grade_gap"]["summary"]
    assert result["grade_gap_summary"] in result["next_steps"][0]


def test_automatic_deferral_keeps_directional_bottlenecks_and_schedule_advice():
    scores = _with_verified_task_support(_scores(recall=.04, ppv_at_pi0=.5), task_n=5)
    scores.pop("behavior_purity")
    schedule = automatic_review_schedule(scores)
    result = _feedback(scores, status="unjudged", review_schedule=schedule)
    assert result["status"] == "review_deferred"
    assert result["review_schedule"] == schedule
    assert result["grade_gap"]["current_grade"] == "UNSCORED"
    assert result["grade_gap"]["bottlenecks"] == ["behavior_purity"]


def test_explainer_direction_uses_the_same_precision_scope_as_its_grade():
    scores = _with_verified_task_support(_scores(
        within_parent=True, within_recall=.04, within_ppv_at_pi0=.9,
        recall=.8, ppv_at_pi0=.3,
    ), task_n=5)
    result = _feedback(scores)
    assert result["status"] == "choose_next_investigation"
    assert result["grade_gap"]["scope"] == "within_parent"
    assert result["grade_gap"]["current_grade"] == "GOOD"
    assert result["grade_gap"]["bottlenecks"] == ["recall"]
    assert result["signals"]["standardized_precision"] == .9


def test_active_review_policy_mismatch_cannot_reuse_adherence_for_grade_upgrade():
    scores = _with_verified_task_support(_scores(
        recall=.04, judge_behavior_strictness="behavioral",
    ), task_n=5)
    before = deepcopy(scores)
    result = build_discovery_feedback(scores, behavior_strictness="causal")
    assert result["signals"]["review_status"] == "stale"
    assert result["discovery_grade"]["code"] == "UNSCORED"
    for gap in (result["grade_gap"], result["qualification_gap"]):
        assert _gap_dimensions(gap)["behavior_purity"]["current"] is None
    assert "behavior_purity" in result["grade_gap"]["bottlenecks"]
    assert not result["qualification_gap"]["target_met"]
    assert scores == before


@pytest.mark.parametrize("change,tasks,blocker,status", [
    ({"ppv_at_pi0": .57}, 5, "ppv_at_pi0", "inspect_boundaries"),
    ({}, 3, "matched_failed_tasks", "choose_next_investigation"),
])
def test_good_qualification_is_separate_from_higher_breadth_aspiration(
    change, tasks, blocker, status,
):
    scores = _with_verified_task_support(_scores(**{"recall": .042, **change}), task_n=tasks)
    before = deepcopy(scores)
    result = build_discovery_feedback(
        scores, judge_feedback={"status": "confirmed"}, target_grade="OUTSTANDING",
    )
    assert result["status"] == status
    assert result["qualification_gap"] == grade_gap_for_scores(scores, target_grade="GOOD")
    assert result["qualification_gap"]["bottlenecks"] == [blocker]
    assert result["grade_gap"]["target_grade"] == "OUTSTANDING"
    assert "recall" in result["grade_gap"]["bottlenecks"]
    assert scores == before


@pytest.mark.parametrize("change,validation,status", [
    ({"behavior_purity": None}, {"status": "unjudged"}, "review_current"),
    ({"errors": 1}, {}, "recover_evaluation"),
])
def test_good_milestone_does_not_override_missing_evidence_or_execution_recovery(
    change, validation, status,
):
    scores = _with_verified_task_support(_scores(recall=.042, **change), task_n=5)
    result = build_discovery_feedback(
        scores, judge_feedback={"status": "confirmed", **validation},
        target_grade="OUTSTANDING",
    )
    assert result["status"] == status


def test_good_or_better_feedback_preserves_qualifying_scores_when_pursuing_breadth():
    scores = _with_verified_task_support(_scores(recall=.04), task_n=5)
    result = build_discovery_feedback(
        scores, judge_feedback={"status": "confirmed"}, target_grade="OUTSTANDING",
    )
    assert result["qualification_gap"]["target_met"]
    assert result["qualification_gap"]["bottlenecks"] == []
    assert result["grade_gap"]["target_grade"] == "OUTSTANDING"


def test_ok_target_does_not_invent_precision_adherence_or_five_task_shortfalls():
    scores = _with_verified_task_support(
        _scores(recall=.004, ppv_at_pi0=.5, behavior_purity=.6), task_n=2,
    )
    result = build_discovery_feedback(
        scores, judge_feedback={"status": "confirmed"}, target_grade="OK",
    )
    assert result["status"] == "choose_next_investigation"
    assert result["grade_gap"]["target_grade"] == "OK"
    assert result["grade_gap"]["bottlenecks"] == ["recall"]


# MCP tools carry current feedback without extra paid reviews or rewriting saved evidence.


@pytest.fixture
def task_runtime(tmp_path, monkeypatch):
    rt, fn = _runtime(tmp_path, monkeypatch)
    # Three positive rollouts of one task, nineteen other failed tasks and a pass.
    rt.dataset.items = [
        {"_item_id": f"case-{index}", "model_id": "m", "benchmark": "b",
         "problem_id": str(max(0, index - 2)), "run_idx": index if index < 3 else 0,
         "is_correct": index == 22,
         "response_text": "I guess the value." if index < 3 else "A derived value."}
        for index in range(23)
    ]
    calls = []

    def review(**kwargs):
        calls.append([item["_item_id"] for item in kwargs["items"]])
        result = _result(rt.library.get(fn.name), ["yes"] * len(kwargs["items"]))
        # The review sampler may select a subset in a different order.
        for key in ("verdicts", "item_calibrations"):
            for index, (row, item) in enumerate(zip(result[key], kwargs["items"], strict=True)):
                row["item_id"] = item["_item_id"]
                if key == "verdicts":
                    row["snapshot_id"] = make_snapshot(
                        item, item_id=item["_item_id"], source=kwargs["source"],
                        failure_localisation=kwargs["failure_localisations"][index],
                        claim_contract=result["claim_contract"],
                    )["snapshot_id"]
        return result

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", review)
    yield rt, fn, calls


def test_narrow_task_guidance_survives_existing_review_hint_on_all_main_tools(task_runtime):
    rt, fn, calls = task_runtime
    evaluated = tools_eval.evaluate_functions(names=[fn.name], mode="full")
    assert evaluated["ok"], evaluated
    result = evaluated["results"][0]
    assert result["scores"]["task_recall"] == pytest.approx(1 / 20)
    assert result["scores"]["recall"] == pytest.approx(3 / 22)
    assert result["scores"]["n_matched_failed_tasks"] == 1
    assert result["discovery_feedback"]["status"] == "review_current"

    judged = tools_eval.judge_function(fn.name, review_phase="development")
    assert judged["ok"], judged
    assert calls
    n_calls = len(calls)
    retrieved = tools_library.get_function_feedback(fn.name)
    submitted = tools_library.submit_function(
        name=fn.name, source=fn.source, description=fn.description, mechanism=fn.mechanism,
    )
    assert submitted["ok"], submitted
    assert submitted["feedback"]["task_recall"] == pytest.approx(1 / 20)
    assert len(calls) == n_calls  # Guidance never triggers an extra paid review.
    inspected = tools_library.get_library_function(fn.name)
    for payload in (judged, retrieved, submitted, inspected):
        feedback = payload["discovery_feedback"]
        assert feedback["status"] == "choose_next_investigation"
        assert feedback["signals"]["n_matched_failed_tasks"] == 1
    listed = tools_library.list_library_functions()["functions"][0]
    assert listed["scores"]["task_recall"] == pytest.approx(1 / 20)
    saved = FunctionLibrary.load(rt.working_library_path()).get(fn.name)
    assert saved.scores["n_matched_failed_tasks"] == 1


def test_judge_evidence_outside_evaluation_does_not_inflate_frozen_task_recall(task_runtime):
    rt, fn, _ = task_runtime
    # A later row is another task with the same detector behavior, beyond the cap.
    rt.dataset.items[3]["response_text"] = "I guess the value."
    evaluated = tools_eval.evaluate_functions(names=[fn.name], limit=3)
    assert evaluated["ok"], evaluated
    assert fn.scores["task_recall_scope"] == "observed_lower_bound"
    frozen = {key: deepcopy(value) for key, value in fn.scores.items()
              if key.startswith("task_recall") or key == "n_matched_failed_tasks"}
    assert frozen["task_recall"] == pytest.approx(1 / 20)
    assert "case-3" not in fn.failure_localisations

    judged = tools_eval.judge_function(fn.name)
    assert judged["ok"], judged
    assert "case-3" in fn.failure_localisations
    for key, value in frozen.items():
        assert fn.scores[key] == value
        assert judged["feedback"][key] == value
    retrieved = tools_library.get_function_feedback(fn.name)
    inspected = tools_library.get_library_function(fn.name)
    listed = tools_library.list_library_functions()["functions"][0]
    for scores in (retrieved["feedback"], inspected["function"]["scores"], listed["scores"],
                   tools_eval._current_scores(rt, fn)):
        assert scores["task_recall"] == 1 / 20
        assert scores["n_matched_failed_tasks"] == 1
        assert scores["task_support_comparison"] == 2
    # Quality breadth still includes the additional source-verified match.
    assert extract_metrics(tools_eval._current_scores(rt, fn))["n_matched_failed_tasks"] == 2


def test_changed_claim_uses_current_review_guidance_instead_of_old_purity(task_runtime):
    _, fn, _ = task_runtime
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert tools_eval.judge_function(fn.name)["ok"]
    fn.description += " The assistant also claims to have verified the value."
    retrieved = tools_library.get_function_feedback(fn.name)
    assert retrieved["judge_feedback"]["status"] == "stale"
    assert retrieved["discovery_feedback"]["status"] == "evaluate_current"
    assert retrieved["discovery_feedback"]["signals"]["task_recall"] is None
    inspected = tools_library.get_library_function(fn.name)["function"]["scores"]
    assert inspected["task_recall"] is None
    assert inspected["feedback"]["task_recall"] is None
    assert fn.scores["feedback"]["task_recall"] == 1 / 20  # Saved evidence stays untouched.


def test_judge_prefilter_skip_exposes_investigation_without_review_call(task_runtime):
    rt, fn, calls = task_runtime
    for item in rt.dataset.items:
        item["response_text"] = "A derived value."
    evaluated = tools_eval.evaluate_functions(names=[fn.name], mode="full")
    assert evaluated["ok"], evaluated
    assert evaluated["results"][0]["discovery_feedback"]["status"] == "inspect_association"
    judged = tools_eval.judge_function(fn.name)
    assert judged["ok"] and judged["skipped"]
    assert judged["discovery_feedback"]["status"] == "inspect_association"
    assert not calls


def test_sampled_library_advice_uses_common_panel_instead_of_claiming_frontier_membership(
    task_runtime,
):
    _, fn, _ = task_runtime
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert tools_eval.judge_function(fn.name, review_phase="development")["ok"]
    # Preserve current evaluation/review identity but mark the finite scoring cohort as sampled.
    fn.scores.update(sampled=True, sample_n=23, sample_frame_n=200)
    status = tools_library.pareto_status()
    assert fn.name in status["sampled_functions"]
    assert fn.name in status["top_x"]
    assert fn.name not in status["validated_top_x"]


def test_status_display_order_matches_report_and_keeps_scalar_eligibility_separate(
    tmp_path, monkeypatch,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.library.functions = [
        _ranked_function("okay_broad", .9, .5, 1),
        _ranked_function("good_precise", .15, 1, 1),
        _ranked_function("good_broader", .3, .8, 1),
        _ranked_function("stale", .99, 1, 1),
    ]
    rt.library.functions[-1].description = "Changed claim awaiting review"
    rt.run.top_x = 3
    before = deepcopy(rt.library.to_dict())
    status = tools_library.pareto_status()
    expected = ["good_broader", "good_precise", "okay_broad"]
    assert status["top_x"] == expected
    assert [row["name"] for row in build_data(rt.library)["functions"]][:3] == expected
    assert status["validated_top_x"] == []
    assert rt.library.to_dict() == before


def test_all_library_views_expose_agent_grade_but_keep_public_good(tmp_path, monkeypatch):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = _state_function("good")
    fn.scores["recall"] = .201
    rt.library.functions = [fn]
    before = deepcopy(rt.library.to_dict())

    listed = tools_library.list_library_functions()
    inspected = tools_library.get_library_function(fn.name)
    feedback = tools_library.get_function_feedback(fn.name)
    for row in (listed["functions"][0], inspected, feedback):
        assert row["verdict"]["code"] == "GOOD"
        assert row["discovery_grade"]["code"] == "OUTSTANDING"
    assert inspected["function"]["scores"]["badge"] == "GOOD"
    assert listed["discovery_grades"]["counts"]["OUTSTANDING"] == 1
    assert sum(listed["discovery_grades"]["counts"].values()) == 1
    assert rt.library.to_dict() == before


@pytest.mark.parametrize("change, expected", [
    ("ranking_objective", "OUTSTANDING"), ("judge_policy", "UNSCORED"),
])
def test_agent_grades_distinguish_ranking_preferences_from_judge_policy(
    tmp_path, monkeypatch, change, expected,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.library.functions = [_state_function("good")]
    if change == "ranking_objective":
        rt.config.objective_weights = [50, 20, 30]
    else:
        rt.config.behavior_strictness = "contextual"
    before = deepcopy(rt.library.to_dict())
    response = tools_library.list_library_functions()
    assert response["functions"][0]["discovery_grade"]["code"] == expected
    assert response["discovery_grades"]["counts"][expected] == 1
    assert rt.library.to_dict() == before


def test_unchanged_submit_returns_agent_goal_without_rewriting_saved_grade(
    tmp_path, monkeypatch,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    fn.scores = _state_function("good").scores
    attach_task_support(fn)
    before = deepcopy(fn.scores)
    response = tools_library.submit_function(
        name=fn.name, source=fn.source, description=fn.description, mechanism=fn.mechanism,
        auto_evaluate=False,
    )
    assert response["ok"]
    assert response["verdict"]["code"] == "GOOD"
    assert response["discovery_grade"]["code"] == "OUTSTANDING"
    assert rt.library.get(fn.name).scores == before


def test_outstanding_aspiration_cannot_hide_the_driver_stop_decision(
    tmp_path, monkeypatch,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = _state_function("good")
    fn.scores["recall"] = .15
    rt.library.functions = [fn]
    internal = {
        "ok": True, "goal_met": True, "minimum_met": True, "min_failure_modes": 1,
        "n_qualifying_failure_modes": 1, "qualifying_failure_modes": [fn.name],
        "n_validated_failure_modes": 1, "validated_failure_modes": [fn.name],
        "remaining_failure_modes": 0, "pending_failure_modes": [],
        "failure_mode_thresholds": {"min_failure_recall": .01},
        "final_validation_target": {"minimum_met": True},
        "failure_mode_eligibility": [{
            "name": fn.name, "eligible": True, "reason": None,
            "measured": {"failure_recall": .15, "standardized_precision": .9,
                         "behavior_purity": 1, "n_confirmed_failed_datapoints": 5},
        }],
    }
    before = deepcopy(internal)
    monkeypatch.setattr(tools_library, "_pareto_status", lambda: internal)
    response = tools_library.pareto_status()
    # The agent sees an Outstanding aspiration; the driver's Good stop decision is kept.
    assert response["goal_met"] is True
    assert response["minimum_met"] is False
    assert response["target_grade"] == "OUTSTANDING"
    assert "final_validation_target" not in response
    assert internal == before
    assert tools_library._pareto_status()["goal_met"] is True


@pytest.mark.parametrize("goal_met, round_idx, wait_for_plateau", [
    (True, 3, True), (False, 8, False),
])
def test_round_transition_enforces_completion_before_family_or_seed_changes(
    tmp_path, monkeypatch, goal_met, round_idx, wait_for_plateau,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.run.round, rt.run.max_rounds = round_idx, 8
    rt.config.discovery_wait_for_plateau = wait_for_plateau
    # An empty retained library needs no feedback exchange. Retained candidate
    # validation boundaries are tested in test_round_stagnation.
    if not goal_met:
        rt.library.functions = []
    before_run = deepcopy(vars(rt.run))
    before_library = deepcopy(rt.library.to_dict())
    directory = rt.run.run_dir()
    directory.mkdir(parents=True, exist_ok=True)
    family = directory / "family_discovery.json"
    # The completed driver decision takes priority over a pending family action.
    family.write_text('{"phase": "assess_family", "synthesis_required": true}')
    before_family = family.read_bytes()
    status = {
        "ok": True, "goal_met": goal_met, "round": round_idx, "max_rounds": 8,
        "minimum_met": True, "n_validated_failure_modes": 1,
        "final_validation_target": {"minimum_met": False},
        "ready_for_final_validation": not goal_met,
        "final_validation_guidance": "Freeze candidates and run the final census.",
    }
    monkeypatch.setattr(tools_library, "_pareto_status", lambda: status)
    monkeypatch.setattr(rt, "persist_library", lambda: pytest.fail("Must not save a new round"))

    response = tools_library.advance_round()

    assert response["ok"]
    assert (response["advanced"], response["goal_met"]) == (False, goal_met)
    assert response["round"] == round_idx
    assert "final_validation_target" not in response
    assert vars(rt.run) == before_run
    assert rt.library.to_dict() == before_library
    assert family.read_bytes() == before_family


def test_round_transition_reports_grade_improvement_without_consuming_driver_baseline(
    tmp_path, monkeypatch,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = _state_function("good")
    fn.scores["recall"] = .08
    rt.library.functions = [fn]
    rt._previous_discovery_grades = tools_library._discovery_grades(rt)
    previous = deepcopy(rt._previous_discovery_grades)
    initialize_family_discovery(rt)
    fn.scores["recall"] = .15
    before_round = rt.run.round
    complete_small_corpus_round(rt)
    monkeypatch.setattr("autofa.session.round_validation.validate_round",
                        lambda rt: {"ok": True, "message": "Grade-only fixture checked."})

    assert tools_library.advance_round()["refinement_pending"]
    response = tools_library.advance_round(validation_response="Retain the supported improvement.")

    assert response["ok"]
    assert response["round"] == before_round + 1
    feedback = response["agent_grade_feedback"]
    # The round bulletin compares completed round boundaries, while the fresh
    # next round has not improved yet. The driver's turn baseline stays intact.
    assert feedback["encouraging"] is False
    completed = feedback["round_bulletin"]["last_completed_round"]
    assert completed["encouraging"] is True
    assert completed["grade_count_changes"]["EXCELLENT"] == 1
    assert completed["grade_count_changes"]["GOOD"] == -1
    assert rt._previous_discovery_grades == previous
