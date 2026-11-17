# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Association prefilter: decision rules, and skipping judge work without losing candidates."""

import json
from copy import deepcopy

import pytest

from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.mcp import tools_eval, tools_library
from autofa.metrics import claude_judge
from autofa.metrics.association import contingency
from autofa.metrics.judge_prefilter import judge_prefilter
from autofa.metrics.verdicts import (
    behavior_validation,
    resolved_behavior_purity,
    verdict_for_scores,
)

from support.judge import BAD_SOURCE, GOOD_SOURCE, judge_result, submit
from support.judge import runtime as runtime


def evaluated(a=50, b=40, c=50, d=60, **overrides):
    return {
        **contingency(a, b, c, d),
        "evaluated": True, "n_scored": a + b + c + d,
        "eval_mode": "full", "sampled": False, "errors": 0, "parent_errors": 0,
        **overrides,
    }


def test_full_census_negative_association_skips():
    scores = evaluated(20, 40, 80, 60)
    # A capped evaluation covering the whole corpus is also a census, without intervals.
    capped = {key: value for key, value in evaluated(
        20, 40, 80, 60, eval_mode="capped", evaluation_scope_complete=True,
    ).items() if key not in {"recall_ci_low", "recall_ci_high", "fpr_ci_low", "fpr_ci_high"}}
    for candidate in (scores, capped):
        result = judge_prefilter(candidate)
        assert result["status"] == "skip"
        assert "negative_association" in result["criteria"]
        assert result["measured"]["coverage"] == "census"
    # Equal rates are not a negative association.
    assert judge_prefilter(evaluated(40, 40, 60, 60))["status"] == "pass"


def test_low_ppv_alone_does_not_skip_judging():
    result = judge_prefilter(evaluated(ppv_at_pi0=0.01))
    assert result["status"] == "pass"
    assert result["criteria"] == []


def test_zero_fires_skip_only_without_processing_errors():
    scores = evaluated(0, 0, 100, 100)
    result = judge_prefilter(scores)
    assert result["status"] == "skip"
    assert result["criteria"] == ["no_fires"]
    assert judge_prefilter({**scores, "errors": 1})["status"] == "inconclusive"


def test_zero_fires_on_small_limited_evaluation_remains_inconclusive():
    scores = evaluated(0, 0, 100, 100, sampled=True)
    decision = judge_prefilter(scores)
    assert decision["status"] == "inconclusive"
    assert decision["criteria"] == []
    assert decision["measured"]["failure_coverage"]["estimated"] is True


def test_weighted_zero_from_thirty_failures_does_not_skip_judging():
    from autofa.metrics.sampling import interval_fields

    # Population-weighted counts must not masquerade as 562 inspected failures.
    scores = evaluated(
        0, 3151 / 170, 562, 3151 * 169 / 170,
        sampled=True, eval_mode="capped", n_scored=200, evaluation_scope_complete=True,
        **interval_fields(recall=0, raw_ppv=0, n_eff_fail=30,
                          n_eff_flagged=1, fpr=1 / 170, n_eff_pass=170),
    )
    decision = judge_prefilter(scores)
    assert decision["status"] == "inconclusive"
    assert decision["criteria"] == []
    assert decision["measured"]["recall_ci_high"] == pytest.approx(.11351339317396862)


def test_sampled_low_coverage_does_not_gate_precise_matches():
    scores = evaluated(1, 0, 9999, 10000, sampled=True, eval_mode="capped",
                       evaluation_scope_complete=True, recall_ci_low=0, recall_ci_high=.01)
    decision = judge_prefilter(scores)
    assert decision["status"] == "pass"
    assert decision["criteria"] == []


@pytest.mark.parametrize("cells, bounds, expected", [
    ((20, 70, 80, 30), (0.10, 0.30, 0.60, 0.80), "skip"),
    ((20, 40, 80, 60), (0.10, 0.30, 0.30, 0.50), "inconclusive"),
    ((2, 4, 8, 6), (0.01, 0.70, 0.10, 0.80), "inconclusive"),
], ids=["separated", "touching", "overlapping"])
def test_limited_negative_association_requires_separated_intervals(cells, bounds, expected):
    low, high, fpr_low, fpr_high = bounds
    scores = evaluated(*cells, sampled=True, recall_ci_low=low, recall_ci_high=high,
                       fpr_ci_low=fpr_low, fpr_ci_high=fpr_high)
    result = judge_prefilter(scores)
    assert result["status"] == expected
    assert result["criteria"] == (["negative_association"] if expected == "skip" else [])


def test_limited_ppv_upper_bound_uses_fixed_reference_prevalence():
    scores = evaluated(
        4000, 5000, 6000, 5000, sampled=True,
        recall_ci_low=0.39, recall_ci_high=0.41, fpr_ci_low=0.49, fpr_ci_high=0.51,
    )
    result = judge_prefilter(scores)
    assert result["status"] == "skip"
    assert result["criteria"] == ["negative_association"]
    assert result["measured"]["ppv_upper_bound"] == pytest.approx(
        0.41 * 0.30 / (0.41 * 0.30 + 0.49 * 0.70)
    )


@pytest.mark.parametrize("changes", [
    {"errors": 1}, {"evaluation_complete": False}, {"assessment_coverage": 0.99},
    {"recall": float("nan")}, {"n_scored": 0}, {"n_flagged": 999},
    {"n_items": float("inf")}, {"n_flagged": -1}, {"fpr": True}, {"recall": 1.1},
    {"assessment_coverage": float("nan")}, {"n_scored": 10**1000},
    {"sampled": True, "recall_ci_high": 0.1},  # Interval excludes its own point estimate.
], ids=["errors", "incomplete", "partial_assessment", "nan_rate",
        "nothing_scored", "more_fires_than_items", "infinite_items", "negative_fires",
        "bool_rate", "rate_above_one", "nan_coverage", "oversized_count",
        "interval_excludes_estimate"])
def test_invalid_or_incomplete_evidence_never_skips(changes):
    result = judge_prefilter(evaluated(20, 40, 80, 60, **changes))
    assert result["status"] == "inconclusive"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("sampled,key", [(False, "fpr"), (True, "recall_ci_low")])
def test_missing_metrics_or_limited_evaluation_bounds_never_skip(sampled, key):
    scores = evaluated(20, 40, 80, 60, sampled=sampled)
    del scores[key]
    assert judge_prefilter(scores)["status"] == "inconclusive"


@pytest.mark.parametrize("cells", [(0, 40, 0, 60), (20, 0, 80, 0)])
def test_single_label_class_cannot_establish_association(cells):
    assert judge_prefilter(evaluated(*cells))["status"] == "inconclusive"


def test_explainer_uses_within_zero_and_does_not_fall_back_to_global_rates():
    scores = evaluated(
        90, 10, 10, 90, within_parent="parent", within_n=100,
        within_pct_caught=0.20, within_recall=0.0, within_fpr=0.40, within_ppv_at_pi0=0.0,
        within_a=0, within_b=20, within_c=50, within_d=30,
    )
    result = judge_prefilter(scores)
    assert result["status"] == "skip"
    assert result["measured"]["recall"] == 0.0
    assert result["measured"]["n_flagged"] == 20
    assert result["measured"]["scope"] == "within_parent"


def test_good_within_parent_association_survives_bad_global_association():
    scores = evaluated(
        20, 40, 80, 60, within_parent="parent", within_n=100,
        within_pct_caught=0.25, within_recall=0.8, within_fpr=0.02, within_ppv_at_pi0=0.95,
        within_a=24, within_b=1, within_c=6, within_d=69,
    )
    assert judge_prefilter(scores)["status"] == "pass"


def test_missing_within_metrics_cannot_fall_back_to_global_metrics():
    scores = evaluated(20, 40, 80, 60, within_parent="parent", within_n=100)
    assert judge_prefilter(scores)["status"] == "inconclusive"


def test_limited_explainer_requires_within_intervals():
    scores = evaluated(
        20, 40, 80, 60, sampled=True, within_parent="parent", within_n=100,
        within_pct_caught=0.20, within_recall=0.1, within_fpr=0.70, within_ppv_at_pi0=0.058,
        within_a=5, within_b=35, within_c=45, within_d=15,
    )
    assert judge_prefilter(scores)["status"] == "inconclusive"
    scores.update(within_recall_ci_low=0.05, within_recall_ci_high=0.2,
                  within_fpr_ci_low=0.5, within_fpr_ci_high=0.9)
    assert judge_prefilter(scores)["status"] == "skip"


def test_explicit_zero_within_fire_count_is_preserved():
    scores = evaluated(within_parent="parent", within_n=100, within_n_flagged=0)
    result = judge_prefilter(scores)
    assert result["status"] == "skip"
    assert result["criteria"] == ["no_fires"]


def test_behavior_scores_do_not_affect_prefilter_and_inputs_are_unchanged():
    scores = evaluated(90, 10, 10, 90, behavior_purity=0.0, behavior_specificity=0.0,
                       behavior_n_errors=100, behavior_judgment_stale=True)
    before = deepcopy(scores)
    result = judge_prefilter(scores)
    assert result["status"] == "pass"
    assert scores == before
    assert judge_prefilter(scores) == result
    json.dumps(result, allow_nan=False)


def test_no_evaluation_is_not_a_quality_rejection():
    result = judge_prefilter({"ppv_at_pi0": 0.0})
    assert result["status"] == "not_evaluated"
    assert result["criteria"] == []


def test_uncertain_low_recall_still_allows_conclusive_negative_association():
    scores = evaluated(0, 90, 100, 10, sampled=True, eval_mode="capped",
                       evaluation_scope_complete=True)
    result = judge_prefilter(scores)
    assert result["status"] == "skip"
    assert result["criteria"] == ["negative_association"]
    assert result["measured"]["recall_ci_high"] > .01


@pytest.mark.parametrize("change", [
    {"evaluation_scope_complete": None}, {"eval_truncated": True}, {"recall_ci_high": None},
])
def test_association_skip_requires_valid_rates_and_sampling_frame(change):
    scores = evaluated(0, 90, 100, 10, sampled=True, eval_mode="capped",
                       evaluation_scope_complete=True)
    scores.update(change)
    result = judge_prefilter(scores)
    assert result["status"] == "inconclusive"
    assert result["criteria"] == []


# Workflow: association rejection prevents judge work without losing repairable candidates.


def guess_items(n, failures, derived, other="I guess."):
    """``n`` rollouts; the first ``failures`` fail and ``derived(i)`` ones show a derivation."""
    return [
        {"_item_id": f"case-{i}", "is_correct": i >= failures,
         "model_id": "m", "benchmark": "b", "problem_id": str(i), "run_idx": 0,
         "response_text": "I derive the value." if derived(i) else other}
        for i in range(n)
    ]


def stub_judge(monkeypatch, runtime, **extra):
    """Confirm every reviewed case and return the recorded judge calls."""
    calls = []

    def judge(**kwargs):
        calls.append(kwargs)
        return {**judge_result(runtime.library.get("guess"), kwargs["items"]), **extra}

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    return calls


def reviewed_ids(calls):
    return [item["_item_id"] for call in calls for item in call["items"]]


@pytest.mark.parametrize("historical_scores", [False, True])
def test_evaluation_and_judge_agree_on_literal_positive_category(
    runtime, monkeypatch, historical_scores,
):
    for i, item in enumerate(runtime.dataset.items):
        item["response_text"] = "positive" if i < 8 else "other" if i < 10 else "negative"
    fn = FeatureFunction(
        name="guess", description="The assistant guesses a value.",
        mechanism="The assistant supplies a value without showing its derivation.",
        source="def guess(item): return item['response_text']",
    )
    runtime.library.functions.append(fn)
    evaluation = tools_eval.evaluate_functions(names=[fn.name], mode="full")
    assert evaluation["ok"]
    assert evaluation["results"][0]["verdict"] == verdict_for_scores(fn.scores)
    assert evaluation["results"][0]["verdict"]["code"] == "UNSCORED"
    assert fn.scores["bucket"] == "positive"
    assert fn.scores["bucket_mode"] == "exact"
    assert fn.scores["n_fail_caught"] == 8
    assert fn.scores["task_recall"] == .8
    if historical_scores:
        fn.scores.pop("bucket_mode")  # The saved per-bucket table remains authoritative.
    calls = stub_judge(monkeypatch, runtime)
    judged = tools_eval.judge_function(fn.name, evidence_n=8)
    assert judged["ok"]
    assert judged["verdict"] == verdict_for_scores(fn.scores)
    assert judged["verdict"]["code"] == "GOOD"
    assert fn.scores["n_confirmed_failed_tasks"] == 8
    fn.scores["verdict"] = {"code": "WEAK", "label": "WEAK"}
    reevaluated = tools_eval.evaluate_functions(names=[fn.name], mode="full")
    assert reevaluated["results"][0]["verdict"]["code"] == "GOOD"
    assert fn.scores["n_confirmed_failed_tasks"] == 8
    assert set(reviewed_ids(calls)) == {f"case-{i}" for i in range(8)}


def test_submit_skips_judge_and_persists_explanation(runtime, monkeypatch):
    monkeypatch.setattr(tools_eval, "judge_function", lambda *args, **kwargs: pytest.fail(
        "Rejected association must not schedule a judge"))
    result = submit()
    assert result["ok"]
    assert result["judge_refresh"]["skipped"] is True
    assert result["judge_feedback"]["status"] == "skipped_pre_filter"
    assert result["feedback"]["behavior_purity"] is None
    assert result["feedback"]["behavior_specificity"] is None
    saved = FunctionLibrary.load(runtime.working_library_path()).get("guess")
    assert saved is not None  # Still available for association repair.
    assert saved.scores["judge_prefilter"]["criteria"] == ["negative_association"]
    assert saved.scores["judge_prefilter"]["measured"]["coverage"] == "census"
    assert "behavior_purity" not in saved.scores


def test_explicit_judge_skips_before_scan_compile_or_holdout(runtime, monkeypatch):
    assert submit()["ok"]
    before = deepcopy(runtime.library.get("guess").scores)

    def forbidden(*args, **kwargs):
        pytest.fail("A pre-filtered detector must not consume judge work or repair evidence")

    monkeypatch.setattr(runtime.dataset, "iter_batches", forbidden)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbidden)
    monkeypatch.setattr("autofa.session.repair_workflow.validation_context", forbidden)
    monkeypatch.setattr("autofa.session.repair_workflow.reserve_validation", forbidden)
    result = tools_eval.judge_function("guess")
    assert result["ok"] and result["skipped"]
    assert result["n_judged"] == 0
    assert result["behavior_purity"] is None
    assert result["verdict"] == verdict_for_scores(runtime.library.get("guess").scores)
    assert result["verdict"]["code"] == "UNSCORED"
    assert runtime.library.get("guess").scores["judge_prefilter"] == before["judge_prefilter"]


def test_rare_precise_candidate_receives_automatic_and_explicit_review(runtime, monkeypatch):
    runtime.dataset.items = guess_items(400, 200, lambda i: i == 0)
    calls = stub_judge(monkeypatch, runtime)
    submitted = submit(GOOD_SOURCE)
    assert submitted["ok"] and submitted["judge_refresh"]["refreshed"]
    fn = runtime.library.get("guess")
    assert fn.scores["raw_ppv"] == 1
    assert fn.scores["recall"] == .005
    assert fn.scores["judge_prefilter"]["status"] == "pass"
    assert reviewed_ids(calls) == ["case-0"]
    assert resolved_behavior_purity(fn.scores) == 1
    assert verdict_for_scores(fn.scores)["code"] == "WEAK"
    explicit = tools_eval.judge_function("guess")
    assert explicit["ok"] and not explicit.get("skipped")
    saved = FunctionLibrary.load(runtime.working_library_path()).get("guess")
    assert resolved_behavior_purity(saved.scores) == 1


def test_three_of_562_failures_gets_automatic_review_and_visible_ok_without_good_minimum(
    runtime, monkeypatch,
):
    from autofa.metrics import final_review
    from autofa.reports.failure_cards import _extract_autofa_data, build_data, render_html
    from autofa.session.discovery_target import failure_mode_target_for_runtime

    runtime.dataset.items = guess_items(600, 562, lambda i: i < 3)
    calls = stub_judge(monkeypatch, runtime, **tools_eval._review_settings(runtime.config)[2])
    submitted = submit(GOOD_SOURCE)
    fn = runtime.library.get("guess")
    assert submitted["ok"]
    assert submitted["judge_refresh"]["refreshed"] is True
    assert sorted(reviewed_ids(calls)) == ["case-0", "case-1", "case-2"]
    assert fn.scores["recall"] == pytest.approx(3 / 562)
    assert fn.scores["n_matched_failed_tasks"] == 3
    assert fn.scores["behavior_purity"] == 1
    assert fn.scores["judge_prefilter"]["status"] == "pass"
    assert verdict_for_scores(fn.scores)["code"] == "OK"
    target = failure_mode_target_for_runtime(runtime, minimum=1)
    assert target["n_validated_failure_modes"] == 0
    assert target["minimum_met"] is False

    # Finalization must retain the current completed review, not pay for it again.
    previous = deepcopy(fn.scores["behavior_verdicts"])
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude",
                        lambda **kwargs: pytest.fail("Complete narrow review should be reused"))
    finalized = final_review.final_review_functions()
    assert finalized["ok"]
    assert finalized["reviewed"] == []
    assert sorted(fn.scores["behavior_verdicts"], key=lambda row: row["item_id"]) == sorted(
        previous, key=lambda row: row["item_id"],
    )
    assert fn.scores["judge_prefilter"]["status"] == "pass"
    assert verdict_for_scores(fn.scores)["code"] == "OK"
    data = build_data(runtime.library)
    html_data = _extract_autofa_data(render_html(data))
    assert [(row["name"], row["verdict"]["code"]) for row in html_data["functions"]] == [
        ("guess", "OK"),
    ]


def test_revised_candidate_passing_prefilter_receives_normal_judgment(runtime, monkeypatch):
    assert submit()["judge_refresh"]["skipped"]
    # Good precision, rather than merely passing the hard exclusion prefilter,
    # is now required for automatic review.
    runtime.dataset.items[10]["response_text"] = "I guess."
    calls = stub_judge(monkeypatch, runtime)
    result = submit(GOOD_SOURCE)
    assert result["ok"] and result["judge_refresh"]["refreshed"]
    assert len(calls) == 1
    scores = runtime.library.get("guess").scores
    assert scores["judge_prefilter"]["status"] == "pass"
    assert behavior_validation(scores)["status"] == "confirmed"
    assert resolved_behavior_purity(scores) == 1.0


@pytest.mark.parametrize("judge_harness", ["auto", "claude"])
def test_judge_function_routes_selected_harness_and_keeps_backend_provenance(
    runtime, monkeypatch, judge_harness,
):
    runtime.config.harness = "codex"
    runtime.config.codex_cli = "/configured/codex"
    runtime.config.claude_cli = "/configured/claude"
    runtime.config.behavior_judge_harness = judge_harness
    runtime.config.behavior_judge_effort = "ultra" if judge_harness == "auto" else "medium"
    calls = []

    def judge(**kwargs):
        calls.append(kwargs)
        harness = kwargs.get("harness", "claude")
        return {
            **judge_result(runtime.library.get("guess"), kwargs["items"]),
            "backend": "codex_app_server" if harness == "codex" else "claude_agent_sdk",
            "judge_harness": harness, "judge_model": kwargs["model"],
            "judge_effort": kwargs["effort"], "judge_thinking": kwargs.get("thinking"),
        }

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    submitted = submit(GOOD_SOURCE)
    assert submitted["ok"] and submitted["judge_refresh"]["deferred"]
    assert tools_eval.judge_function("guess", review_mode="standard")["ok"]
    assert len(calls) == 1
    call = calls[0]
    expected = "codex" if judge_harness == "auto" else "claude"
    assert call.get("harness", "claude") == expected
    assert call["cli_path"] == f"/configured/{expected}"
    assert call["model"] == ("gpt-6.1-sol" if expected == "codex" else "claude-sonnet-5-5")
    if expected == "codex":
        assert call["effort"] == "ultra" and "thinking" not in call
    scores = runtime.library.get("guess").scores
    assert scores["judge_harness"] == expected
    assert behavior_validation(scores)["judge_harness"] == expected


def test_re_evaluation_preserves_archived_judgment_and_clears_skip(runtime, monkeypatch):
    stub_judge(monkeypatch, runtime)
    assert submit(GOOD_SOURCE)["ok"]
    assert tools_eval.judge_function("guess", review_mode="standard")["ok"]
    fn = runtime.library.get("guess")
    archived = deepcopy(fn.scores["behavior_verdicts"])
    for item in runtime.dataset.items:
        item["is_correct"] = not item["is_correct"]
    for _ in range(2):
        assert tools_eval.evaluate_functions(names=["guess"], mode="full")["ok"]
        assert fn.scores["behavior_verdicts"] == archived
        assert fn.scores["behavior_purity"] == 1.0  # Raw history remains inspectable.
        assert resolved_behavior_purity(fn.scores) is None
        assert fn.scores["feedback"]["behavior_purity"] is None
        assert behavior_validation(fn.scores)["judgment_history_only"]
        inspected = tools_library.get_library_function("guess")["function"]["scores"]
        assert inspected["behavior_purity"] is None
        assert inspected["historical_behavior_scores"]["behavior_purity"] == 1.0
        assert inspected["behavior_validation"]["judgment_history_only"]
    for item in runtime.dataset.items:
        item["is_correct"] = not item["is_correct"]
    assert tools_eval.evaluate_functions(names=["guess"], mode="full")["ok"]
    assert fn.scores["judge_prefilter"]["status"] == "pass"
    assert fn.scores["behavior_verdicts"] == archived
    assert resolved_behavior_purity(fn.scores) == 1.0


def test_capped_inconclusive_candidate_defers_automatic_but_allows_explicit_review(
    runtime, monkeypatch,
):
    # Two failures, two passes, only one pass fires: negative point estimate,
    # but a four-row sample cannot confidently establish negative association.
    runtime.dataset.items = [runtime.dataset.items[i] for i in (2, 3, 10, 12, *range(4, 10), 13)]
    runtime.config.default_eval_limit = 4
    runtime.run.eval_limit = 4
    calls = stub_judge(monkeypatch, runtime)
    # Only LLM detectors honor the iterative cap by default. The deterministic
    # stand-in avoids model calls while exercising that detector's evaluation path.
    result = submit(kind="llm")
    assert result["ok"]
    scores = runtime.library.get("guess").scores
    assert scores["n_scored"] == 4
    assert scores["evaluation_scope_complete"] is False
    assert scores["judge_prefilter"]["status"] == "inconclusive"
    assert result["judge_refresh"]["deferred"]
    assert calls == []
    assert tools_eval.judge_function("guess", review_mode="standard")["ok"]
    assert len(calls) == 1


def test_evaluation_error_does_not_reject_behavior(runtime):
    runtime.dataset.items[0].pop("response_text")
    fn = FeatureFunction(name="guess", source=BAD_SOURCE, description="Guessing")
    runtime.library.upsert(fn)
    result = tools_eval.evaluate_functions(names=["guess"], mode="full")
    assert result["ok"] and fn.scores["errors"] == 1
    assert fn.scores["judge_prefilter"]["status"] == "inconclusive"


def test_changed_claim_cannot_reuse_stale_association_skip(runtime, monkeypatch):
    assert submit()["ok"]
    fn = runtime.library.get("guess")
    fn.source = GOOD_SOURCE
    stub_judge(monkeypatch, runtime)
    result = tools_eval.judge_function("guess")
    assert result["ok"] and not result.get("skipped")
    assert result["verdict"]["code"] == "UNSCORED"
    assert fn.scores["judge_prefilter"]["status"] == "not_evaluated"
    assert resolved_behavior_purity(fn.scores) == 1.0


@pytest.mark.parametrize("parent_matches_child", [False, True])
def test_bulk_evaluation_never_uses_corpus_association_for_an_explainer(
    runtime, parent_matches_child,
):
    parent_source = GOOD_SOURCE
    if parent_matches_child:
        # The child has negative global association, but inside this parent it
        # catches both failures and neither passing example.
        for i in (0, 1, 10, 11):
            runtime.dataset.items[i]["response_text"] = (
                "in-parent " + runtime.dataset.items[i]["response_text"]
            )
        parent_source = (
            "def parent(item): return item['response_text'].startswith('in-parent ')"
        )
    parent = FeatureFunction(name="parent", source=parent_source, description="Parent")
    child = FeatureFunction(name="guess", source=BAD_SOURCE, description="Explainer",
                            explains="parent")
    runtime.library.upsert(parent)
    runtime.library.upsert(child)
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    assert child.scores["fpr"] > child.scores["recall"]
    prefilter = child.scores["judge_prefilter"]
    assert prefilter["measured"]["scope"] == "within_parent"
    assert "negative_association" not in prefilter["criteria"]
    if parent_matches_child:
        assert child.scores["within_n_flagged"] == 2
        assert prefilter["status"] == "pass"
    else:
        assert child.scores["within_n_flagged"] == 0
        assert prefilter["status"] == "skip"
        assert prefilter["criteria"] == ["no_fires"]


def test_automatic_family_submission_uses_ten_percent_review_without_three_case_cap(
    runtime, monkeypatch,
):
    flagged, expected = 31, 4
    from autofa.session.family_discovery import (
        family_comparison_pending,
        initialize_family_discovery,
    )

    runtime.dataset.items = guess_items(
        flagged + 39, flagged, lambda i: i < flagged, other="An observed result.")
    initialize_family_discovery(runtime)
    assert family_comparison_pending(runtime.run.run_dir())
    calls = stub_judge(monkeypatch, runtime)
    result = submit(GOOD_SOURCE)
    assert result["ok"] and result["evaluated"], result
    reviewed = reviewed_ids(calls)
    assert len(reviewed) == expected, result.get("judge_refresh")
    assert len(set(reviewed)) == expected
    assert runtime.library.get("guess").scores["behavior_n_judged"] == expected
    assert family_comparison_pending(runtime.run.run_dir())
