# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reports: ranking and verdict badges agree in Python and the browser, and low coverage
never hides a precise detector from judge review or report cards."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.judge_prefilter import failure_coverage_eligibility, judge_prefilter
from autofa.metrics.verdicts import (
    extract_metrics,
    format_coverage_percent,
    verdict_for_metrics,
    verdict_for_scores,
    verdict_js_classify_fn,
)
from autofa.reports.failure_cards import (
    _extract_autofa_data,
    build_data,
    refresh_report_data,
    render_html,
    write_failure_cards,
)
from autofa.reports.ordering import (
    has_report_scores,
    order_library_functions,
    order_report_functions,
    rank_display_functions,
    report_assessment,
    visible_report_functions,
)

from support.task_support import attach_task_support, run_node


def _browser_order(functions):
    html = render_html({"functions": []})
    helpers = html[html.index("const VERDICT_PRESENTATIONS"):html.index("const RAW =")]
    # Use the actual boot-time expression, as applied after a server has changed
    # the embedded array order. The original array and stable link IDs survive.
    boot_order = html[html.index("  const fns ="):html.index("  const root =", html.index(
        "  const fns ="))]
    script = helpers + "\nconst RAW = " + json.dumps({"functions": functions}) + ";\n" + (
        "const before = JSON.stringify(RAW.functions);\n" + boot_order
        + "if (JSON.stringify(RAW.functions) !== before) throw Error('Input mutated');\n"
        + "if (!fns.every(fn => RAW.functions.includes(fn))) throw Error('Rows replaced');\n"
        + "console.log(JSON.stringify(fns));"
    )
    return json.loads(run_node(script))


def _row(name, coverage, precision, adherence, **extra):
    return {"name": name, "metrics": {
        "recall": coverage, "ppv_at_pi0": precision, "behavior_purity": adherence,
        "n_matched_failed_tasks": 5,
    }, **extra}


def test_quality_tiers_then_coverage_precision_adherence_and_name():
    # Optimizer eligibility and saved verdicts never reorder display quality.
    functions = [
        _row("weak_broad", .9, .3, 1),
        _row("ok_broad", .8, .6, .8, ranking={"eligible": True}, verdict={"code": "GOOD"}),
        _row("good_precise", .2, 1, 1),
        _row("good_broad", .7, .7, .9, behavior_validation={"ranking_eligible": False}),
        _row("good_tie_lower_precision", .5, .8, 1),
        _row("good_tie_lower_adherence", .5, .9, .9),
        _row("z_good_tie", .5, .9, 1),
        _row("a_good_tie", .5, .9, 1, ranking={"eligible": False}, verdict={"code": "WEAK"}),
        _row("ok_narrow", .007, 1, 1),
        _row("weak_narrow", .004, 1, 1),
        _row("pending_review", .99, 1, None),
        _row("missing", None, None, None),
        _row("empty", 0, 0, None, evaluation={"evaluated": True}),
    ]
    before = deepcopy(functions)
    expected = [
        "good_broad", "a_good_tie", "z_good_tie", "good_tie_lower_adherence",
        "good_tie_lower_precision", "good_precise", "ok_broad", "ok_narrow",
        "weak_broad", "weak_narrow", "empty", "pending_review", "missing",
    ]
    for source in (functions, list(reversed(functions))):
        assert [fn["name"] for fn in order_report_functions(source)] == expected
    assert functions == before


def test_browser_restores_quality_order_after_host_scrambles_report_data():
    functions = [
        _row("ok_broad", .9, .6, .8, verdict={"code": "GOOD"}),
        _row("weak_broad", .8, .3, 1),
        _row("good_low_precision", .2, .7, 1),
        _row("good_low_adherence", .2, .9, .9),
        _row("z_good", .2, .9, 1),
        _row("a_good", .2, .9, 1, verdict={"code": "WEAK"}),
        _row("empty", 0, 0, 0),
        _row("stale", .99, 1, 1, evaluation={"evaluation_stale": True}),
        _row("missing", None, None, None),
    ]
    for function in functions:
        function["card_id"] = "stable-" + function["name"]
    expected = visible_report_functions(functions)
    assert [fn["name"] for fn in expected] == [
        "a_good", "z_good", "good_low_adherence", "good_low_precision", "ok_broad",
    ]
    assert _browser_order(functions) == expected
    assert _browser_order(list(reversed(functions))) == expected


def test_browser_order_matches_cli_with_broad_matches_and_small_review_sample():
    functions = []
    for name, coverage, precision, tasks in (
        ("ok_broad", .8, .6, 5), ("broad_good", .182, .9, 23),
        ("narrower_good", .04, .9, 5),
    ):
        raw = _raw_function(name, coverage, precision)
        fn = FeatureFunction(name=name, source=raw["source"], description=raw["description"],
                             scores=raw["scores"])
        attach_task_support(fn, task_n=tasks)
        if name == "broad_good":
            fn.scores["behavior_verdicts"] = fn.scores["behavior_verdicts"][:2]
            fn.scores.update(behavior_n_judged=2, behavior_n_completed=2, behavior_n_yes=2)
        functions.append(fn.to_dict())
    ranked = rank_display_functions(functions)
    assert [fn["name"] for fn in ranked] == ["broad_good", "narrower_good", "ok_broad"]
    assert ranked[0]["metrics"]["n_matched_failed_tasks"] == 23
    assert ranked[0]["metrics"]["n_confirmed_failed_tasks"] == 2
    compact = [{"name": fn["name"], "metrics": fn["metrics"],
                "verdict": {"code": "OK" if fn["name"] == "broad_good" else "GOOD"},
                "card_id": "stable-" + fn["name"]} for fn in reversed(ranked)]
    assert [fn["name"] for fn in _browser_order(compact)] == [fn["name"] for fn in ranked]


@pytest.mark.parametrize("evaluation,metrics_status", [
    ({"evaluation_stale": True}, None),
    # A metrics-level unavailable state is not revived by evaluation metadata.
    ({"evaluated": True}, "unscored"),
])
def test_unavailable_evaluation_cannot_rank_using_saved_good_metrics(evaluation, metrics_status):
    pending = _row("pending", .9, 1, 1, evaluation=evaluation, verdict={"code": "GOOD"})
    if metrics_status:
        pending["metrics"]["evaluation_status"] = metrics_status
    weak = _row("weak", .001, .2, .3)
    assert order_report_functions([pending, weak]) == [weak, pending]


def _raw_function(name, coverage=.2, precision=.9, adherence=1):
    return {
        "name": name, "source": "def detect(item): return True",
        "description": "Observable mistake",
        "scores": {
            "evaluated": True, "n_items": 100, "n_scored": 100, "n_flagged": 20,
            "recall": coverage, "precision": precision, "ppv_at_pi0": precision,
            "behavior_purity": adherence, "behavior_specificity": 1,
        },
        "failure_localisations": {"id": {"trace": ["large trace"]}},
    }


def test_raw_rows_share_report_ordering_without_copying_traces_or_mutating_scores():
    functions = [_raw_function("good", .02), _raw_function("ok", .8, .6),
                 _raw_function("broader_good", .2)]
    before = deepcopy(functions)
    rows = rank_display_functions(functions)
    # Historical score-only rows have no distinct-task confirmation evidence;
    # they are at most OK and therefore sort by Coverage within that tier.
    assert [row["name"] for row in rows] == ["ok", "broader_good", "good"]
    assert functions == before
    by_name = {function["name"]: function for function in functions}
    for row in rows:
        original = by_name[row["name"]]
        assert row is not original
        assert row["scores"] is not original["scores"]
        assert row["failure_localisations"] is original["failure_localisations"]
        assert verdict_for_metrics(row["metrics"]) == verdict_for_scores(row["scores"])
        assert not row["ranking"]["eligible"]  # Historical scores still have their numeric badge.
    assert [row["name"] for row in order_report_functions(rows)] == [row["name"] for row in rows]


def test_raw_rows_resolve_current_claim_and_active_policy_before_ranking():
    current = _raw_function("current", .1)
    changed = _raw_function("changed_claim", .9)
    other_policy = _raw_function("other_policy", .8)
    for function in (current, changed, other_policy):
        function["scores"].update({
            "judge_contract_version": JUDGE_CONTRACT_VERSION,
            "judge_behavior_strictness": "causal",
            "judge_policy_hash": judge_policy_hash(behavior_strictness="causal"),
            "claim_hash": claim_hash(source=function["source"], description=function["description"],
                                     mechanism=function["description"]),
            "behavior_n_judged": 4, "behavior_n_completed": 4, "behavior_n_yes": 4,
        })
    changed["description"] = "Changed claim"
    other_policy["scores"].update({
        "judge_behavior_strictness": "contextual",
        "judge_policy_hash": judge_policy_hash(behavior_strictness="contextual"),
    })
    before = deepcopy([current, changed, other_policy])
    rows = rank_display_functions([changed, other_policy, current], behavior_strictness="causal")
    assert [row["name"] for row in rows] == ["current", "changed_claim", "other_policy"]
    assert [verdict_for_scores(row["scores"])["code"] for row in rows] == [
        "OK", "UNSCORED", "UNSCORED",
    ]
    assert [current, changed, other_policy] == before


@pytest.mark.parametrize("axis,value,shown", [
    # Measured association scores stay visible without a current review.
    ("behavior_purity", None, True), ("behavior_specificity", -1.0, True),
    ("recall", None, False), ("precision", 1.1, False), ("recall", float("nan"), False),
    ("precision", True, False),
])
def test_scored_card_filter_requires_measured_association_scores(axis, value, shown):
    row = rank_display_functions([_raw_function("measured")])[0]
    assert has_report_scores(row)
    row["metrics"][axis] = value
    assert has_report_scores(row) is shown


def test_judge_policy_change_does_not_hide_saved_detector_measurements():
    function = _raw_function("historical_review")
    function["scores"].update(
        judge_contract_version=JUDGE_CONTRACT_VERSION,
        judge_policy_hash="historical-policy",
        claim_hash=claim_hash(source=function["source"], description=function["description"],
                              mechanism=function["description"]),
        behavior_n_judged=3, behavior_n_completed=3, behavior_n_yes=3,
    )
    before = deepcopy(function)
    row = rank_display_functions([function])[0]
    assert has_report_scores(row)
    assert row["metrics"]["recall"] == function["scores"]["recall"]
    assert row["metrics"]["behavior_purity"] == -1.0
    assert row["metrics"]["behavior_validation_status"] == "stale"
    assert verdict_for_metrics(row["metrics"])["code"] == "UNSCORED"
    assert not row["ranking"]["eligible"]
    assert function == before


def _reviewed_function(name="historical", *, coverage=.2, purity=1, strictness="contextual"):
    raw = _raw_function(name, coverage=coverage, adherence=purity)
    raw["scores"].update(
        judge_behavior_strictness=strictness, behavior_n_judged=5,
        behavior_n_completed=5, behavior_n_yes=5 if purity else 0,
        behavior_n_no=0 if purity else 5,
    )
    fn = FeatureFunction(name=name, source=raw["source"], description=raw["description"],
                         scores=raw["scores"])
    attach_task_support(fn, task_n=5)
    if not purity:
        for row in fn.scores["behavior_verdicts"]:
            row.update(verdict="no", verification_status="rejected")
            row["initial_review"]["verdict"] = "no"
    fn.scores["judge_policy_hash"] = "0" * 64
    return fn.to_dict()


def test_saved_assessment_preserves_historical_scores_only_for_presentation():
    strictness, label = "causal", "Causal contribution"
    function = _reviewed_function(strictness=strictness)
    before = deepcopy(function)
    assessment = report_assessment(function)
    assert assessment["scores"]["behavior_judgment_stale"] is True
    assert verdict_for_scores(assessment["scores"])["code"] == "UNSCORED"
    assert assessment["metrics"]["behavior_purity"] == 1
    assert assessment["metrics"]["behavior_specificity"] == 1
    assert assessment["metrics"]["n_matched_failed_tasks"] == 5
    assert assessment["metrics"]["assessment_scope"] == "saved_review"
    assert verdict_for_metrics(assessment["metrics"])["code"] == "GOOD"
    validation = assessment["behavior_validation"]
    assert validation["status"] == "historical"
    assert label in validation["label"]
    assert validation["judge_policy_hash"] == "0" * 64
    assert not validation["ranking_eligible"]
    assert not assessment["metrics"]["behavior_ranking_eligible"]
    assert function == before


def test_saved_report_order_is_quality_first_without_changing_live_ranking_or_saved_rows():
    good = _reviewed_function("good", coverage=.2)
    weak = _reviewed_function("weak", coverage=.8, purity=0)
    rows = [weak, good]
    before = deepcopy(rows)
    # Live discovery still needs the current evidence policy for either row.
    live = rank_display_functions(rows)
    assert [row["name"] for row in live] == ["weak", "good"]
    assert all(verdict_for_metrics(row["metrics"])["code"] == "UNSCORED" for row in live)
    historical = rank_display_functions(rows, saved_assessment=True)
    assert [row["name"] for row in historical] == ["good", "weak"]
    assert [verdict_for_metrics(row["metrics"])["code"] for row in historical] == ["GOOD", "WEAK"]
    assert all(not row["ranking"]["eligible"] for row in historical)
    assert _browser_order(historical) == [historical[0]]
    originals = order_library_functions(rows)
    assert originals[0] is good and originals[1] is weak
    assert rows == before


@pytest.mark.parametrize("updates", [
    {"judge_contract_version": "behavior-claim-v5"}, {"judge_policy_hash": "malformed"},
    {"judge_behavior_strictness": "invalid"},
    {"judge_behavior_strictness": "causal"},  # saved contextual contract differs
    {"claim_contract": {}}, {"claim_hash": "1" * 64}, {"evaluation_stale": True},
    {"judge_backend": "heuristic"}, {"judge_prefilter": {"status": "skip"}},
    {"behavior_n_completed": 0},
    {"behavior_n_yes": 0, "behavior_n_no": 0, "behavior_n_errors": 5},
])
def test_saved_assessment_does_not_revive_invalid_or_missing_reviews(updates):
    function = _reviewed_function()
    function["scores"].update(updates)
    assessment = report_assessment(function)
    assert assessment["behavior_validation"]["status"] != "historical"
    assert "assessment_scope" not in assessment["metrics"]


@pytest.mark.parametrize("initial,updates", [
    (True, {"content_validation": {"status": "changed"}}),
    (False, {"verification_status": "historical"}),
    (False, {"claim_contract_id": "another-contract"}),
])
def test_saved_assessment_honors_review_content_and_contract_invalidations(initial, updates):
    function = _reviewed_function()
    row = function["scores"]["behavior_verdicts"][0]
    if initial:
        row = row["initial_review"]
    row.update(updates)
    before = deepcopy(function)
    assessment = report_assessment(function)
    assert assessment["behavior_validation"]["status"] == "stale"
    assert verdict_for_metrics(assessment["metrics"])["code"] == "UNSCORED"
    assert function == before


def test_saved_assessment_keeps_partial_review_uncertainty_and_processing_errors():
    function = _reviewed_function()
    scores = function["scores"]
    scores.update(behavior_n_yes=2, behavior_n_no=1, behavior_n_uncertain=1,
                  behavior_n_errors=1, behavior_n_completed=4, behavior_purity=.5,
                  behavior_sample={"target_n": 5, "limited": True, "reused_n": 3})
    rows = scores["behavior_verdicts"]
    rows[2].update(verdict="no", verification_status="rejected")
    rows[3].update(verdict="unknown", verification_status="uncertain")
    rows[4].update(verdict="unknown", processing_status="error", completion_status="incomplete")
    before = deepcopy(function)
    assessment = report_assessment(function)
    assert assessment["behavior_validation"]["status"] == "historical"
    assert assessment["metrics"]["behavior_purity"] == .5
    assert assessment["behavior_validation"]["uncertain"] == 1
    assert assessment["behavior_validation"]["processing_errors"] == 1
    assert assessment["behavior_validation"]["sample"] == scores["behavior_sample"]
    assert not assessment["behavior_validation"]["ranking_eligible"]
    assert function == before


@pytest.mark.parametrize("field,value", [
    ("source", "def detect(item): return False"),
    ("meta", {"dependency_bundle": {"version": 1, "nodes": {}, "helpers": {},
                                    "detectors": {}}}),
])
def test_saved_assessment_does_not_revive_changed_claim_or_dependency(field, value):
    function = _reviewed_function()
    function[field] = value
    assessment = report_assessment(function)
    assert assessment["behavior_validation"]["status"] == "stale"
    assert verdict_for_metrics(assessment["metrics"])["code"] == "UNSCORED"


def test_current_and_legacy_assessments_keep_existing_behavior():
    current = _reviewed_function()
    current["scores"]["judge_policy_hash"] = judge_policy_hash(behavior_strictness="contextual")
    assert report_assessment(current)["behavior_validation"]["status"] == "confirmed"
    legacy = _raw_function("legacy")
    assessment = report_assessment(legacy)
    assert assessment["behavior_validation"]["status"] == "legacy"
    assert verdict_for_metrics(assessment["metrics"])["code"] == "OK"


# Quality verdict boundaries and evaluation scope agree in Python and the browser.


def assert_browser_agrees(expression, expected):
    script = verdict_js_classify_fn() + f"\nconsole.log(JSON.stringify({expression}));"
    assert json.loads(run_node(script)) == expected


@pytest.mark.parametrize("unavailable", [{"evaluated": False}, {"evaluation_stale": True}])
def test_unavailable_current_evaluation_does_not_reuse_saved_counts(unavailable):
    scores = {"evaluated": True, "n_items": 100, "n_scored": 100,
              "n_flagged": 20, "recall": .4, "ppv_at_pi0": .9, **unavailable}
    assert verdict_for_scores(scores)["label"] == "NOT EVALUATED"


@pytest.mark.parametrize("within_matches,coverage,precision,label", [
    (0, 0, 0, "NO MATCHES"),
    (1, 0, 0, "WEAK"),  # One matched case cannot establish five distinct matched tasks.
])
def test_explainer_badge_uses_within_parent_scope(within_matches, coverage, precision, label):
    scores = {
        "evaluated": True, "n_items": 1000, "n_flagged": 200,
        "recall": .5, "ppv_at_pi0": .8,
        "within_parent": "parent", "within_n_flagged": within_matches,
        "within_recall": coverage, "within_ppv_at_pi0": precision,
        "behavior_purity": 1.0,
    }
    original = deepcopy(scores)
    metrics = extract_metrics(scores)
    assert metrics["recall"] == coverage
    assert metrics["ppv_at_pi0"] == precision
    assert metrics["n_flagged"] == 200  # Preserve the separate corpus count.
    assert verdict_for_scores(scores)["label"] == label
    assert scores == original
    assert_browser_agrees(f"classifyFromMetrics({json.dumps(metrics)})", verdict_for_scores(scores))


@pytest.mark.parametrize("coverage,expected", [
    (.00999, "<1%"), (.01, "1.0%"), (.01001, ">1%"), (.00499, "<0.5%"), (None, "—"),
])
def test_coverage_format_keeps_quality_boundaries_unambiguous(coverage, expected):
    assert format_coverage_percent(coverage) == expected
    assert_browser_agrees(f"coveragePercent({json.dumps(coverage)})", expected)


@pytest.mark.parametrize("coverage,precision,adherence,expected", [
    (.01001, .70, .90, "GOOD"),
    (.01, .70, .90, "OK"),
    (.00501, .70, .90, "OK"),
    (.005, .70, .90, "WEAK"),
    (.02, .6999, .90, "OK"),
    (.02, .40, .90, "OK"),
    (.02, .3999, .90, "WEAK"),
    (.02, .70, .8999, "OK"),
    (.02, .70, .50, "OK"),
    (.02, .70, .4999, "WEAK"),
    (.02, .80, None, "NOT EVALUATED"),
])
def test_combined_thresholds_agree_in_browser(coverage, precision, adherence, expected):
    scores = {"evaluated": True, "n_items": 1000, "n_flagged": 100,
              "recall": coverage, "ppv_at_pi0": precision, "behavior_purity": adherence,
              "behavior_n_judged": 10, "behavior_n_completed": 10}
    attach_task_support(SimpleNamespace(
        name="boundary", source="def detect(item): return True",
        description="Observable decision", mechanism="Observable decision", scores=scores,
        dependency_bundle=None,
    ))
    actual = verdict_for_scores(scores)
    assert actual["label"] == expected
    if adherence is None:
        assert extract_metrics(scores)["behavior_purity"] == -1
    assert_browser_agrees(f"classifyFromMetrics({json.dumps(extract_metrics(scores))})", actual)


@pytest.mark.parametrize("tasks,repeats,label", [
    (4, 2, "OK"), (5, 1, "GOOD"),
])
def test_good_requires_matches_on_five_distinct_failed_tasks(tasks, repeats, label):
    fn = SimpleNamespace(
        name="support", source="def detect(item): return True",
        description="Observable decision", mechanism="Observable decision", dependency_bundle=None,
        scores={"evaluated": True, "n_items": 100, "n_flagged": tasks * repeats,
                "recall": .2, "ppv_at_pi0": .9, "behavior_purity": 1.0},
    )
    attach_task_support(fn, task_n=tasks, repeat_n=repeats)
    original = deepcopy(fn.scores)
    metrics = extract_metrics(fn.scores)
    assert metrics["n_confirmed_failed_tasks"] == tasks
    assert metrics["n_matched_failed_attempts"] == tasks * repeats
    actual = verdict_for_scores(fn.scores)
    assert actual["label"] == label
    assert fn.scores == original
    assert_browser_agrees(f"classifyFromMetrics({json.dumps(metrics)})", actual)


@pytest.mark.parametrize("overrides,label,provisional", [
    ({}, "GOOD", True),
    ({"estimated_n_matched_failed_tasks": 4.99}, "OK", False),
    ({"estimated_n_matched_failed_tasks": 5.0}, "GOOD", True),
    ({"estimated_n_matched_failed_tasks": "8"}, "OK", False),
    ({"task_support_estimate_status": "unavailable"}, "OK", False),
    ({"task_support_comparison_basis": "observed"}, "OK", False),
    ({"n_matched_failed_tasks": 5, "estimated_n_matched_failed_tasks": 4}, "GOOD", False),
])
def test_provisional_task_support_comparison_agrees_in_browser(overrides, label, provisional):
    metrics = {
        "evaluation_status": "evaluated", "n_flagged": 4, "recall": .18,
        "ppv_at_pi0": .9, "behavior_purity": 1.0, "n_matched_failed_tasks": 2,
        "estimated_n_matched_failed_tasks": 8.0,
        "task_support_estimate_status": "provisional",
        "task_support_comparison_basis": "estimated", **overrides,
    }
    actual = verdict_for_metrics(metrics)
    assert actual["label"] == label
    assert ("Provisional Good" in actual["reason"]) is provisional
    assert_browser_agrees(f"classifyFromMetrics({json.dumps(metrics)})", actual)


@pytest.mark.parametrize("support", [None, 4.5, "5"])
def test_unknown_or_invalid_support_cannot_promote_good(support):
    metrics = {"recall": .2, "ppv_at_pi0": .9, "behavior_purity": 1.0,
               "n_matched_failed_tasks": support}
    actual = verdict_for_metrics(metrics)
    assert actual["code"] == "OK"
    assert_browser_agrees(f"classifyFromMetrics({json.dumps(metrics)})", actual)


# Coverage affects grades, but does not exclude precise detectors from review or reports.


def scores_at(recall, /, **overrides):
    return {
        **contingency(recall * 10000, 0, (1 - recall) * 10000, 10000),
        "evaluated": True, "n_scored": 20000, "sampled": False, "eval_mode": "full",
        "evaluation_complete": True, "errors": 0, "behavior_purity": 1,
        **overrides,
    }


def function_at(recall, /, **overrides):
    return FeatureFunction(name="rare", source="def rare(item): return True",
                           description="A rare behavior", scores=scores_at(recall, **overrides))


@pytest.mark.parametrize("recall", [0, .005])
def test_global_coverage_never_excludes_a_precise_detector(recall):
    scores = scores_at(recall)
    before = deepcopy(scores)
    assert failure_coverage_eligibility(scores)["status"] == "pass"
    decision = judge_prefilter(scores)
    assert decision["status"] == ("skip" if recall == 0 else "pass")
    assert decision["criteria"] == (["no_fires"] if recall == 0 else [])
    assert scores == before


def test_removing_coverage_gate_does_not_bypass_negative_association():
    result = judge_prefilter(scores_at(.009, fpr=.02, ppv_at_pi0=.4))
    assert result["status"] == "skip"
    assert result["criteria"] == ["negative_association"]


@pytest.mark.parametrize("scope,upper", [
    ({"sampled": True}, .02), ({"eval_truncated": True}, None),
])
def test_precise_narrow_sample_does_not_need_a_coverage_lower_bound(scope, upper):
    scores = scores_at(.0001, recall_ci_low=0, recall_ci_high=upper, **scope)
    assert failure_coverage_eligibility(scores)["status"] == "pass"
    decision = judge_prefilter(scores)
    assert decision["status"] == "pass"
    assert decision["criteria"] == []


def test_precise_within_parent_detector_is_not_subject_to_global_coverage_floor():
    scores = scores_at(.0001, within_parent="parent", within_n=100,
                       within_a=50, within_b=0, within_c=0, within_d=50,
                       within_recall=1, within_fpr=0, within_ppv_at_pi0=1,
                       within_n_flagged=50)
    result = judge_prefilter(scores)
    assert result["status"] == "pass"
    assert result["measured"]["failure_coverage"]["recall"] == .0001


@pytest.mark.parametrize("scores,status", [
    ({}, "not_evaluated"),
    (scores_at(.005, evaluation_stale=True), "not_evaluated"),
    (scores_at(0, a=0, c=0), "inconclusive"),
    *((scores_at(.005, **changes), "inconclusive") for changes in (
        {"errors": 1}, {"assessment_coverage": "unknown"},
        {"n_flagged": -1}, {"n_items": 0}, {"a": None},
    )),
])
def test_unknown_invalid_incomplete_or_stale_coverage_is_never_usable(scores, status):
    assert failure_coverage_eligibility(scores)["status"] == status


def test_new_and_refreshed_reports_retain_rare_judged_cards_and_keep_cohort_counts():
    rare = function_at(.005)
    boundary = function_at(.01)
    boundary.name = "boundary"
    library = FunctionLibrary(functions=[rare, boundary])
    before = deepcopy(library.to_dict())
    saved = build_data(library)
    assert len(saved["functions"]) == 2
    fresh = build_data(library)
    refreshed = refresh_report_data(saved, library)
    for report in (fresh, refreshed):
        assert [fn["name"] for fn in report["functions"]] == ["boundary", "rare"]
        assert report["meta"]["n_functions"] == 2
        assert report["meta"]["n_failures"] == 10000
        assert report["meta"]["n_items"] == 20000
    assert library.to_dict() == before
    assert len(saved["functions"]) == 2


def test_report_writer_keeps_low_coverage_scores_visible(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.reports.failure_cards.try_dataset_from_dir", lambda _: None)
    monkeypatch.setattr("autofa.reports.failure_cards.try_dataset_for_library", lambda _: None)
    library = FunctionLibrary(functions=[function_at(.015)])
    out = write_failure_cards(library, out=tmp_path / "failure_cards.html")
    data = _extract_autofa_data(out.read_text())
    assert [fn["name"] for fn in data["functions"]] == ["rare"]


@pytest.mark.parametrize("recall,overrides", [
    (.005, {"evaluation_scope_complete": True}),
    # A historical sample without a complete frame, despite a tight interval.
    (0, {"recall_ci_low": 0, "recall_ci_high": .001}),
])
def test_sampled_low_coverage_reports_remain_visible(recall, overrides):
    fn = function_at(recall, sampled=True, **overrides)
    library = FunctionLibrary(functions=[fn])
    before = deepcopy(library.to_dict())
    assert failure_coverage_eligibility(fn.scores)["status"] == "pass"
    assert [row["name"] for row in build_data(library)["functions"]] == ["rare"]
    assert library.to_dict() == before


@pytest.mark.parametrize("changes", [{"errors": 1}, {"recall": None}])
def test_reports_keep_unknown_or_incomplete_coverage_visible(changes):
    library = FunctionLibrary(functions=[function_at(.005, **changes)])
    assert len(build_data(library)["functions"]) == 1


def test_changed_detector_does_not_hide_based_on_stale_coverage():
    fn = function_at(.005)
    fn.scores["evaluation_claim_hash"] = claim_hash(
        source=fn.source, description=fn.description, mechanism=fn.description)
    library = FunctionLibrary(functions=[fn])
    saved = build_data(library)
    fn.source = "def rare(item): return False"
    assert len(build_data(library)["functions"]) == 1
    assert len(refresh_report_data(saved, library)["functions"]) == 1
