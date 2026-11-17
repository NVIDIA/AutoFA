# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic review follows precision, defers actionably, and retries only within its budget."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.core.library import FeatureFunction
from autofa.mcp import tools_eval, tools_repair
from autofa.metrics import claude_judge
from autofa.metrics.association import StreamingBucketAccumulator
from autofa.metrics.judge_budget import ReviewBudgetError
from autofa.metrics.review_failures import processing_failure, review_failures
from autofa.metrics.review_scheduling import automatic_review_schedule, current_review_schedule
from autofa.metrics.verdicts import behavior_validation
from autofa.session.discovery_stall import observe_discovery_stall
from autofa.session.turn_validation import _can_retry_review

from support.judge import GOOD_SOURCE, judge_result, submit
from support.judge import runtime as runtime
from support.task_support import attach_sampled_task_support, attach_task_support


def _scores(*, precision=0.7, coverage=0.02, task_n=5, sampled=False):
    fn = FeatureFunction(
        name="test", source="def detect(item): return True",
        description="The required result remains unfinished",
        mechanism="Unfinished work prevents the required result",
        scores={"ppv_at_pi0": precision, "recall": coverage, "n_scored": 100},
    )
    if sampled:
        attach_sampled_task_support(fn, task_n=task_n)
    else:
        attach_task_support(fn, task_n=task_n)
    return fn.scores


@pytest.mark.parametrize("precision,status,criterion", [
    (0.399999, "deferred", "precision_below_ok"),
    (0.4, "deferred", "precision_below_good"),
    (0.699999, "deferred", "precision_below_good"),
    (0.7, "standard", "good_precision"),
])
def test_precision_boundaries_use_inclusive_shared_thresholds(precision, status, criterion):
    scores = _scores(precision=precision)
    # Raw precision never controls the schedule; only normalized precision does.
    raw = .1 if status == "standard" else .99
    scores.update(raw_ppv=raw, precision=raw, purity=raw)
    result = automatic_review_schedule(scores)
    assert result["status"] == status
    assert criterion in result["criteria"]
    assert result["measured"]["ppv_at_pi0"] == precision


def test_coverage_does_not_gate_review_of_precise_matches():
    # Below the former 0.5% prefilter threshold.
    result = automatic_review_schedule(_scores(coverage=0.004))
    assert result["status"] == "standard"
    assert result["criteria"] == ["good_precision"]
    assert result["measured"]["recall"] == 0.004


def test_valid_sampling_extrapolation_qualifies_without_reviewed_confirmations():
    scores = _scores(task_n=2, sampled=True)
    scores["behavior_verdicts"] = []
    result = automatic_review_schedule(scores)
    assert result["status"] == "standard"
    assert result["measured"]["n_matched_failed_tasks"] == 2
    assert result["measured"]["estimated_n_matched_failed_tasks"] == pytest.approx(8)
    assert result["measured"]["task_support_comparison_basis"] == "estimated"
    assert result["measured"]["provisional"] is True


@pytest.mark.parametrize("change", [
    {"sample_frame_n": 99}, {"sample_reason": "hand_picked"}, {"sample_requested_n": 999},
])
def test_invalid_sampling_provenance_cannot_use_bare_saved_estimate(change):
    scores = _scores(task_n=2, sampled=True)
    scores.update(change)
    scores.update(estimated_n_matched_failed_tasks=1000, task_support_comparison=1000,
                  estimated_n_confirmed_failed_tasks=1000)
    result = automatic_review_schedule(scores)
    assert result["status"] == "standard"
    assert result["measured"]["task_support_comparison"] == 2
    assert result["criteria"] == ["good_precision"]


def test_bare_task_counts_and_confirmations_cannot_replace_source_receipt():
    scores = _scores()
    scores.pop("task_support")
    scores.update(n_matched_failed_tasks=1000, n_confirmed_failed_tasks=1000)
    result = automatic_review_schedule(scores)
    assert result["status"] == "standard"
    assert result["criteria"] == ["good_precision"]
    assert result["measured"]["task_support_comparison"] is None


@pytest.mark.parametrize("flagged_pass_weight,status,criterion", [
    (8.75, "standard", "good_precision"),
    (35.25, "deferred", "precision_below_good"),
])
@pytest.mark.parametrize("with_observed_count", [True, False])
def test_weighted_sample_metrics_keep_fractional_population_totals(
    flagged_pass_weight, status, criterion, with_observed_count,
):
    accumulator = StreamingBucketAccumulator()
    for index in range(5):
        accumulator.update(True, is_correct=False, item_id=f"item-{index}-0", weight=9.23)
    accumulator.update(False, is_correct=False, weight=7.1)
    accumulator.update(True, is_correct=True, weight=flagged_pass_weight)
    accumulator.update(False, is_correct=True, weight=70.3)
    weighted = next(row for row in accumulator.finalize() if row["bucket"] == "True")
    scores = _scores()
    # Keep the receipt's existing boolean operating point while replacing association
    # measurements with the same weighted accumulator output used by evaluation.
    scores.update({key: value for key, value in weighted.items()
                   if key not in {"bucket", "bucket_mode"}})
    if with_observed_count:
        scores["n_scored"] = accumulator.n_items
    else:
        scores.pop("n_scored")
    assert scores["n_flagged"] != int(scores["n_flagged"])
    assert scores["n_items"] != int(scores["n_items"])
    result = automatic_review_schedule(scores)
    assert result["status"] == status
    assert criterion in result["criteria"]
    assert result["measured"]["provisional"] is True
    assert result["measured"]["n_matched_failed_tasks"] == 5


def test_within_parent_metrics_follow_existing_quality_scope():
    scores = _scores(precision=0.3, coverage=0.001)
    scores.update(within_parent=True, within_ppv_at_pi0=0.7, within_recall=0.02)
    result = automatic_review_schedule(scores)
    assert result["status"] == "standard"
    assert result["measured"]["scope"] == "within_parent"
    scores["within_recall"] = 0
    assert automatic_review_schedule(scores)["status"] == "standard"


@pytest.mark.parametrize("value", [None, float("nan"), 1.1])
def test_invalid_precision_defers_evaluation_repair_instead_of_behavior_verdict(value):
    scores = _scores()
    scores["ppv_at_pi0"] = value
    result = automatic_review_schedule(scores)
    assert result["status"] == "deferred"
    assert "invalid_measurements" in result["criteria"]


@pytest.mark.parametrize("value", [None, float("nan"), 1.1])
def test_missing_or_invalid_coverage_does_not_override_usable_precision(value):
    scores = _scores()
    scores["recall"] = value
    result = automatic_review_schedule(scores)
    assert result["status"] == "standard"
    assert result["measured"]["recall"] is None


@pytest.mark.parametrize("change,criterion", [
    ({"n_flagged": 0}, "no_observed_matches"),
    ({"evaluated": False}, "evaluation_unavailable"),
    ({"evaluation_stale": True}, "evaluation_unavailable"),
    ({"errors": 1}, "evaluation_errors"),
    ({"n_scored": 0}, "invalid_measurements"),
    ({"metrics_scope": "unrepresented_strata"}, "invalid_measurements"),
    ({"n_items": True}, "invalid_measurements"),
    ({"n_flagged": -0.5}, "invalid_measurements"),
    # nan already fails ``>= 0``; only infinity exercises the finiteness guard.
    ({"n_items": float("inf")}, "invalid_measurements"),
])
def test_unusable_evaluations_defer_actionably(change, criterion):
    scores = _scores()
    scores.update(change)
    result = automatic_review_schedule(scores)
    assert result["status"] == "deferred"
    assert criterion in result["criteria"]
    assert result["next_steps"]


def test_missing_scores_defer_evaluation():
    assert automatic_review_schedule({})["status"] == "deferred"
    assert automatic_review_schedule(None)["criteria"] == ["evaluation_unavailable"]


def test_helper_reads_shared_quality_thresholds(monkeypatch):
    from autofa.metrics.verdicts import QUALITY_THRESHOLDS

    monkeypatch.setitem(QUALITY_THRESHOLDS["ppv_at_pi0"], "good", 0.8)
    assert automatic_review_schedule(_scores(precision=0.7))["status"] == "deferred"
    monkeypatch.setitem(QUALITY_THRESHOLDS["recall"], "good", 0.03)
    result = automatic_review_schedule(_scores(precision=0.8, coverage=0.02))
    assert result["status"] == "standard"
    assert result["criteria"] == ["good_precision"]


def _save_deferral(scores):
    scores["review_schedule"] = {
        **automatic_review_schedule(scores),
        "evaluation_claim_hash": scores["evaluation_claim_hash"],
        "evaluation_run_id": scores["evaluation_run_id"],
    }
    return scores


def test_current_saved_deferral_recomputes_instead_of_trusting_stored_reason():
    scores = _save_deferral(_scores(precision=0.5))
    scores["review_schedule"]["reason"] = "Old scheduling text"
    refreshed = current_review_schedule(scores)
    assert refreshed["status"] == "deferred"
    assert refreshed["reason"] != "Old scheduling text"
    scores["ppv_at_pi0"] = 0.7
    assert current_review_schedule(scores) is None


@pytest.mark.parametrize("change", [
    {"evaluation_stale": True}, {"evaluated": False},
    {"evaluation_claim_hash": "other"}, {"evaluation_run_id": "other"},
])
def test_stale_saved_deferral_is_not_current(change):
    scores = _save_deferral(_scores(precision=0.5))
    scores.update(change)
    assert current_review_schedule(scores) is None


def test_saved_deferral_requires_provenance_and_matching_cohort_when_recorded():
    scores = _save_deferral(_scores(precision=0.5))
    saved = scores["review_schedule"]
    saved["cohort_fingerprint"] = "panel"
    assert current_review_schedule(scores) is None
    scores["search_panel"] = {"cohort_fingerprint": "panel"}
    assert current_review_schedule(scores)["status"] == "deferred"
    scores["search_panel"]["cohort_fingerprint"] = "other"
    assert current_review_schedule(scores) is None
    saved.pop("cohort_fingerprint")
    saved.pop("evaluation_claim_hash")
    scores.pop("evaluation_claim_hash")
    assert current_review_schedule(scores) is None


def test_saved_deferral_requires_current_evaluation_item_cohort_when_recorded():
    scores = _save_deferral(_scores(precision=0.5))
    scores["evaluation_item_ids"] = ["second", "first"]
    scores["review_schedule"]["evaluation_cohort_hash"] = hashlib.sha256(json.dumps(
        ["first", "second"], separators=(",", ":"),
    ).encode()).hexdigest()
    assert current_review_schedule(scores)["status"] == "deferred"
    scores["evaluation_item_ids"].reverse()
    assert current_review_schedule(scores)["status"] == "deferred"
    scores["evaluation_item_ids"] = ["other", "first"]
    assert current_review_schedule(scores) is None


def test_scheduling_and_refresh_do_not_mutate_scores_or_saved_advice():
    scores = _save_deferral(_scores(precision=0.5, task_n=2, sampled=True))
    original = deepcopy(scores)
    result = automatic_review_schedule(scores)
    current = current_review_schedule(scores)
    result["next_steps"].clear()
    result["measured"]["ppv_at_pi0"] = 1
    current["criteria"].clear()
    assert scores == original


@pytest.mark.parametrize("reason", [
    ReviewBudgetError("workload cannot fit", capacity=True),
    "review workload exceeds bounded capacity before model dispatch: required_read_calls=4000",
])
def test_saved_and_live_capacity_failures_require_supported_recovery(reason):
    failure = processing_failure(reason)
    assert failure["kind"] == "review_budget_exhausted"
    assert failure["retry_policy"] == "requires_backend_or_configuration_change"
    assert failure["detector_rejected"] is False


def test_legacy_capacity_failure_cannot_buy_a_fresh_retry_or_modify_frozen_verdict():
    failure = {"kind": "review_budget_exhausted", "budget_scope": "capacity",
               "retry_policy": "one_explicit_stage_retry"}
    row = {"item_id": "unfinished", "verdict": "error", "processing_failure": failure}
    original = deepcopy(row)
    [normalized] = review_failures([row])
    assert normalized["retry_policy"] == "requires_backend_or_configuration_change"
    assert row == original
    fn = SimpleNamespace(scores={"behavior_processing_failures": [failure]})
    assert not _can_retry_review(fn)
    fn.scores["behavior_n_review_incomplete"] = 1
    fn.scores["behavior_verdicts"] = [row]
    assert not _can_retry_review(fn)  # An aggregate count cannot override the exhausted row.
    fn.scores["behavior_verdicts"].append({
        "item_id": "unfinished-sibling", "completion_status": "needs_evidence_review",
    })
    assert _can_retry_review(fn)
    fn.scores["behavior_verdicts"] = [row]
    # A separate malformed stage retains its bounded retry in the same cohort.
    fn.scores["behavior_processing_failures"].append({
        "kind": "invalid_reply", "retry_policy": "one_explicit_stage_retry",
    })
    assert _can_retry_review(fn)


@pytest.mark.parametrize("reason", [
    ReviewBudgetError("review elapsed-time budget exhausted after 600 s"),
    "review tool-call budget exhausted; remaining evidence is review_incomplete",
])
def test_attempt_budget_retains_exactly_one_stage_retry(reason):
    failure = processing_failure(reason)
    assert failure["kind"] == "review_budget_exhausted"
    assert failure["retry_policy"] == "one_explicit_stage_retry"
    fn = SimpleNamespace(scores={"behavior_processing_failures": [failure]})
    assert _can_retry_review(fn)
    fn.scores["round_review"] = {"retry_generation": 1}
    assert not _can_retry_review(fn)


def _confirming_review(runtime, items):
    return {**judge_result(runtime.library.get("guess"), items),
            **tools_eval._review_settings(runtime.config)[2]}


@pytest.mark.parametrize("failed_matches,passing_matches,automatic", [
    # .35 normalized precision with 14% coverage: high coverage does not rescue it.
    (14, 11, False),
    (70, 30, False),  # .50 normalized precision: useful OK possible.
    (90, 10, True),   # .794 normalized precision: eligible for standard review.
])
def test_real_submission_schedules_review_by_normalized_precision(
    runtime, monkeypatch, failed_matches, passing_matches, automatic,
):
    runtime.dataset.items = [
        {"_item_id": f"case-{i}", "is_correct": i >= 100, "model_id": "m",
         "benchmark": "b", "problem_id": str(i), "run_idx": 0,
         "response_text": "I derive the value." if (
             i < failed_matches or 100 <= i < 100 + passing_matches) else "I guess."}
        for i in range(200)
    ]
    calls = []

    def judge(**kwargs):
        calls.append(kwargs)
        return _confirming_review(runtime, kwargs["items"])

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    result = submit(GOOD_SOURCE)
    assert result["ok"]
    fn = runtime.library.get("guess")
    assert bool(calls) == automatic
    if automatic:
        assert len(calls[0]["items"]) == 10  # Unchanged max(3, 10% of matches).
        assert result["judge_refresh"]["refreshed"]
        return

    assert result["judge_refresh"]["status"] == "review_deferred"
    assert fn.scores.get("behavior_purity") is None
    assert not fn.scores.get("behavior_verdicts")
    assert behavior_validation(fn.scores)["review_schedule"]["status"] == "deferred"
    assert tools_repair.get_function_repair(fn.name)["next_action"]["tool"] != "judge_function"
    # Deferring paid review completes score triage, not quality validation.
    progress = observe_discovery_stall(runtime)
    assert progress["completed_attempts"] == 1
    assert progress["best_quality"]["good"] == 0
    assert observe_discovery_stall(runtime)["completed_now"] == 0

    if failed_matches == 70:
        # A deliberate useful-OK review stays available, with the standard sample.
        judged = tools_eval.judge_function(fn.name, review_mode="standard")
        assert judged["ok"] and calls
        assert fn.scores["behavior_purity"] == 1
        assert behavior_validation(fn.scores).get("review_schedule") is None
        assert judged["verdict"]["code"] == "OK"


@pytest.mark.parametrize("failed_n,matched_n,repeat_task", [
    (300, 1, False),   # Below the former hard 0.5% prefilter threshold.
    (20, 4, True),     # Broad in attempts, but matches only one distinct task.
])
def test_precise_narrow_submission_gets_automatic_review_without_breadth_gate(
    runtime, monkeypatch, failed_n, matched_n, repeat_task,
):
    runtime.dataset.items = [
        {"_item_id": f"case-{i}", "is_correct": i >= failed_n, "model_id": "m",
         "benchmark": "b", "problem_id": "matched-task" if repeat_task and i < matched_n
         else str(i), "run_idx": i if repeat_task and i < matched_n else 0,
         "response_text": "I derive the value." if i < matched_n else "I guess."}
        for i in range(failed_n + 20)
    ]
    reviewed = []

    def judge(**kwargs):
        reviewed.extend(item["_item_id"] for item in kwargs["items"])
        return _confirming_review(runtime, kwargs["items"])

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    submitted = submit(GOOD_SOURCE)
    assert submitted["ok"]
    assert submitted["judge_refresh"]["refreshed"]
    assert reviewed and set(reviewed) <= {f"case-{i}" for i in range(matched_n)}
    scores = runtime.library.get("guess").scores
    assert scores["ppv_at_pi0"] == 1
    assert scores["recall"] == pytest.approx(matched_n / failed_n)
    assert scores["n_matched_failed_tasks"] == (1 if repeat_task else matched_n)
    assert scores["behavior_purity"] == 1
    # Review eligibility does not relax Good's separate breadth requirements.
    assert submitted["verdict"]["code"] in {"OK", "WEAK"}


def test_zero_match_submission_does_not_call_judge(runtime, monkeypatch):
    for item in runtime.dataset.items:
        item["response_text"] = "I guess."
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude",
                        lambda **kwargs: pytest.fail("No matched case exists to review"))
    submitted = submit(GOOD_SOURCE)
    assert submitted["ok"]
    assert runtime.library.get("guess").scores["n_flagged"] == 0
    assert not submitted["judge_refresh"].get("refreshed")


def test_current_review_beats_retained_schedule_and_stale_schedule_is_not_displayed(
    runtime, monkeypatch,
):
    submitted = submit(GOOD_SOURCE)
    assert submitted["judge_refresh"]["deferred"]
    fn = runtime.library.get("guess")
    assert behavior_validation(fn.scores)["review_schedule"]["status"] == "deferred"
    stale = {**deepcopy(fn.scores), "evaluation_claim_hash": "changed-claim"}
    assert "review_schedule" not in behavior_validation(stale)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude",
                        lambda **kwargs: _confirming_review(runtime, kwargs["items"]))
    assert tools_eval.judge_function(fn.name)["ok"]
    assert fn.scores.get("behavior_purity") == 1
    assert "review_schedule" not in behavior_validation(fn.scores)
