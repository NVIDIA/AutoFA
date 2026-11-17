# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior validation provenance and completeness decide rankings, badges and reports."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.remote_library_api import _json_payload
from autofa.metrics import claim_contract, claude_judge
from autofa.metrics.claim_calibration import summarize_judge_feedback
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.completeness import (
    comparable_evaluations,
    completed_evaluation,
    incomplete_evidence,
)
from autofa.metrics.pareto import (
    library_frontier_metrics,
    library_frontier_points,
    point_from_scores,
    rank_functions,
)
from autofa.metrics.verdicts import (
    behavior_ranking_eligibility,
    behavior_validation,
    current_behavior_scores,
    extract_metrics,
    resolved_behavior_purity,
    resolved_behavior_specificity,
    verdict_for_scores,
    verdict_js_classify_fn,
)
from autofa.reports.failure_cards import (
    _extract_autofa_data,
    build_data,
    refresh_report_data,
    render_html,
)
from autofa.ui.explorer import _build_tree

from support.task_support import attach_task_support, report_renderers, run_node


def _scores(**overrides):
    return {
        "evaluated": True, "n_items": 100, "recall": 0.4,
        "precision": 0.9, "ppv_at_pi0": 0.9, "pct_caught": 0.4,
        "judge_backend": "claude_agent_sdk", "behavior_purity": 1.0,
        "behavior_n_judged": 4, "behavior_n_completed": 4,
        "behavior_n_yes": 4, "behavior_n_no": 0,
        "behavior_n_uncertain": 0, "behavior_n_errors": 0,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(),
        "claim_hash": claim_hash(source="def f(item): return True",
                                 description="Claim", mechanism="Claim"),
        **overrides,
    }


def _claim_fn(scores, name="f"):
    """A detector whose claim matches the claim hash recorded by `_scores`."""
    return FeatureFunction(name=name, source="def f(item): return True", description="Claim",
                           mechanism="Claim", scores=scores)


def _review_js(html):
    """HTML escaping plus the review summary renderers embedded in a report."""
    escaping = html[html.index("function escapeHtml("):html.index("function pct(")]
    return escaping + html[html.index("function reviewValidationStatus("):
                           html.index("function renderCard(")]


def test_quality_badge_keeps_mechanism_validation_provenance_separate():
    legacy = _scores(judge_contract_version=None)
    assert resolved_behavior_purity(legacy) == 1.0  # Historical result is preserved.
    assert behavior_validation(legacy)["status"] == "legacy"
    assert verdict_for_scores(legacy)["label"] == "OK"

    unjudged = _scores(behavior_n_judged=0, behavior_n_completed=0)
    assert verdict_for_scores(unjudged)["label"] == "NOT EVALUATED"
    assert behavior_validation(unjudged)["status"] == "unjudged"
    assert not behavior_validation(unjudged)["ranking_eligible"]
    # Numeric quality alone does not establish the five-task breadth floor.
    assert verdict_for_scores(_scores())["label"] == "OK"
    supported = _claim_fn(_scores())
    attach_task_support(supported)
    assert verdict_for_scores(supported.scores)["label"] == "GOOD"
    supported.scores["judge_contract_version"] = None
    assert verdict_for_scores(supported.scores)["label"] == "OK"

    rejected = _scores(behavior_purity=0.25, behavior_n_yes=1, behavior_n_no=3)
    assert verdict_for_scores(rejected)["label"] == "WEAK"
    assert behavior_validation(rejected)["status"] == "mixed"
    assert behavior_validation(rejected)["rejected"] == 3


def test_prior_judge_contract_requires_revalidation(monkeypatch):
    old_version = "behavior-claim-v3"
    with monkeypatch.context() as patch:
        patch.setattr(claim_contract, "JUDGE_CONTRACT_VERSION", old_version)
        old = _scores(judge_contract_version=old_version)
    original = deepcopy(old)
    validation = behavior_validation(old)
    assert validation["status"] == "legacy"
    assert not validation["ranking_eligible"]
    assert not behavior_ranking_eligibility(old, ["behavior_purity"])["eligible"]
    current = current_behavior_scores(old, source="def f(item): return True",
                                      description="Claim", mechanism="Claim")
    assert behavior_validation(current)["status"] == "stale"
    assert resolved_behavior_purity(current) is None
    assert old == original


def test_missing_specificity_preserves_confirmed_claim_but_excludes_that_ranking_axis():
    scores = _scores(behavior_specificity=None)
    assert behavior_validation(scores)["status"] == "confirmed"
    assert resolved_behavior_purity(scores) == 1
    assert resolved_behavior_specificity(scores) is None
    assert point_from_scores(scores, ["behavior_purity"]) == (1.0,)
    assert point_from_scores(scores, ["behavior_purity", "behavior_specificity"]) is None
    assert not behavior_ranking_eligibility(scores, ["behavior_specificity"])["eligible"]


EXAMPLE_REVIEWS = [
    ("confirmed", None, "this example was not reviewed by the judge"),
    ("confirmed", "yes", "confirmed the described behavior and its contribution"),
    ("mixed", "no", "rejected this example"),
    ("stale", "yes", "historical review"),
]


def test_example_review_never_infers_validation_from_sample_score():
    functions = []
    for status, verdict, _ in EXAMPLE_REVIEWS:
        rows = [{"item_id": "different-example", "verdict": "yes"}]
        if verdict:
            rows.append({"item_id": "selected", "verdict": verdict,
                         "rationale": "Evidence contains <script>markup</script>."})
        functions.append({"behavior_validation": {
            "status": status, "sampled": 10, "confirmed": 10,
            "judge_contract_version": JUDGE_CONTRACT_VERSION, "verdicts": rows,
        }})
    script = (_review_js(render_html(build_data(FunctionLibrary())))
              + "\nconsole.log(JSON.stringify(" + json.dumps(functions)
              + '.map(fn => renderItemValidation(fn, "selected"))));')
    outputs = json.loads(run_node(script))
    for (status, verdict, expected), output in zip(EXAMPLE_REVIEWS, outputs, strict=True):
        assert expected in output, status
        assert "<script>" not in output
        if verdict and status not in {"legacy", "stale", "skipped_pre_filter"}:
            assert "&lt;script&gt;" in output, status


@pytest.mark.parametrize("version,refresh", [
    pytest.param("behavior-claim-v3", True, id="old-contract-refreshed"),
    pytest.param(JUDGE_CONTRACT_VERSION, False, id="current-contract"),
])
def test_saved_confirmed_status_cannot_promote_an_old_contract_in_report_renderers(
    version, refresh,
):
    fn = _claim_fn(_scores(behavior_specificity=.8, behavior_verdicts=[{
        "item_id": "selected", "verdict": "yes", "rationale": "The required output is missing.",
    }]))
    library = FunctionLibrary(functions=[fn])
    saved = build_data(library)
    # Old published reports froze the then-current `confirmed` status alongside
    # their old contract. Refresh intentionally preserves these historical data.
    fn.scores["judge_contract_version"] = version
    saved["functions"][0]["behavior_validation"]["judge_contract_version"] = version
    saved["functions"][0]["behavior_validation"]["status"] = "confirmed"
    original = deepcopy(saved)
    original_scores = deepcopy(fn.scores)
    data = refresh_report_data(saved, library) if refresh else saved
    html = render_html(data)
    payload = _extract_autofa_data(html)
    assert payload["functions"][0]["metrics"] == original["functions"][0]["metrics"]
    assert payload["functions"][0]["behavior_validation"]["status"] == "confirmed"
    script = (report_renderers(html) + "\nconst fn = " + json.dumps(payload["functions"][0])
              + ";const before = JSON.stringify(fn); console.log(JSON.stringify({"
                "item: renderItemValidation(fn, 'selected'),"
                "summary: renderBehaviorSummary(fn.behavior_validation, fn.judge_rationale),"
                "card: renderCard(fn, 1, 1), unchanged: before === JSON.stringify(fn)}));")
    result = json.loads(run_node(script))
    assert result["unchanged"]
    current = version == JUDGE_CONTRACT_VERSION
    confirmed = "confirmed the described behavior and its contribution"
    assert (confirmed in result["item"]) is current
    assert ("matched the description and contributed" in result["summary"]) is current
    if not current:
        assert "historical review" in result["item"]
        assert "OK" in result["card"] and "GOOD" not in result["card"]
    assert saved == original and fn.scores == original_scores


@pytest.mark.parametrize("reused", [False, True])
def test_report_displays_computed_scores_without_promoting_limited_reviews(reused):
    scores = _scores(behavior_specificity=.8, behavior_sample={
        "actual_n": 4, "target_n": 4 if reused else 10, "limited": True,
        "reused_n": 4 if reused else 0,
    })
    fn = _claim_fn(scores, name="scored")
    pending = FeatureFunction(name="pending", description="Unscored candidate",
                              source="def f(item): return False")
    library = FunctionLibrary(functions=[fn, pending])
    original = deepcopy(library.to_dict())
    data = _extract_autofa_data(render_html(build_data(library)))
    # Embedded records retain candidates for refresh; visibility is a separate filter.
    assert [f["name"] for f in data["functions"]] == ["scored", "pending"]
    assert data["functions"][1]["verdict"]["code"] == "UNSCORED"
    assert data["meta"]["n_hidden_functions"] == 1
    assert data["meta"]["n_functions"] == data["meta"]["n_pending_scores"] == 1
    validation = data["functions"][0]["behavior_validation"]
    assert validation["status"] == ("reused_evidence" if reused else "limited_evidence")
    assert not validation["ranking_eligible"]
    assert data["functions"][0]["metrics"]["behavior_purity"] == 1
    assert point_from_scores(scores, ["behavior_purity", "behavior_specificity"]) is None
    assert library.to_dict() == original


def test_stale_review_keeps_coverage_visible_and_withholds_adherence():
    fn = _claim_fn(_scores(behavior_specificity=.8, judge_policy_hash="previous-evidence-policy"))
    original = deepcopy(fn.to_dict())
    card, = _extract_autofa_data(render_html(build_data(FunctionLibrary(functions=[fn]))))[
        "functions"]
    assert card["metrics"]["recall"] == .4
    assert card["metrics"]["behavior_purity"] == -1
    assert card["behavior_validation"]["status"] == "stale"
    assert card["verdict"]["code"] == "UNSCORED"
    assert fn.to_dict() == original


def test_saved_policy_verdicts_and_order_survive_render_refresh_and_upload(monkeypatch):
    with monkeypatch.context() as old_policy:
        old_policy.setattr(claude_judge, "EVIDENCE_VERSION", "judge-evidence-v10")
        saved_policy = judge_policy_hash()
    functions = []
    for name, recall, adherence in (("weak", .9, .3), ("ok", .8, .8), ("good", .2, 1)):
        fn = _claim_fn(_scores(
            recall=recall, behavior_purity=adherence, behavior_specificity=.8,
            behavior_n_judged=5, behavior_n_completed=5, behavior_n_yes=5,
        ), name=name)
        attach_task_support(fn)
        fn.scores["judge_policy_hash"] = saved_policy
        if name == "good":
            fn.scores["behavior_sample"] = {
                "target_n": 10, "limited": True, "reused_n": 5, "phase": "development",
            }
        functions.append(fn)
    pending = deepcopy(functions[0])
    pending.name = "unjudged"
    pending.scores.update(behavior_n_judged=0, behavior_n_completed=0, behavior_n_yes=0,
                          behavior_purity=None, behavior_specificity=None, behavior_verdicts=[])
    library = FunctionLibrary(functions=[pending, *functions])
    before = deepcopy(library.to_dict())
    data = _extract_autofa_data(render_html(build_data(library)))
    expected_names = ["good", "ok", "weak", "unjudged"]
    expected_codes = ["GOOD", "OK", "WEAK", "UNSCORED"]
    assert [fn["name"] for fn in data["functions"]] == expected_names
    assert [fn["verdict"]["code"] for fn in data["functions"]] == expected_codes
    assert all(fn["behavior_validation"]["status"] == "historical"
               for fn in data["functions"][:3])
    assert all(not fn["ranking"]["eligible"] for fn in data["functions"])
    # Repair reports produced by the earlier publisher, which withheld old review scores.
    masked = deepcopy(data)
    for fn in masked["functions"][:3]:
        fn["metrics"]["behavior_purity"] = -1
        fn["behavior_validation"]["status"] = "stale"
    refreshed = refresh_report_data(masked, library)
    assert [fn["verdict"]["code"] for fn in refreshed["functions"]] == expected_codes
    uploaded = json.loads(_json_payload(library.to_dict()))
    assert [fn["name"] for fn in uploaded["functions"]] == expected_names
    assert {fn["name"]: fn for fn in uploaded["functions"]} == {
        fn["name"]: fn for fn in before["functions"]
    }

    html = render_html(refreshed)
    helpers = html[html.index("const VERDICT_PRESENTATIONS"):html.index("const RAW =")]
    renderer = html[html.index("function escapeHtml("):html.index("function boot(")]
    script = ("const STATE = {selection: null}; const document = {addEventListener() {}};\n"
              + helpers + renderer + "\nconst rows = orderReportFunctions("
              + json.dumps(list(reversed(refreshed["functions"]))) + ");\n"
              + "console.log(JSON.stringify(rows.map((fn, i) => ({name: fn.name, "
                "code: classifyFromMetrics(fn.metrics, fn.evaluation).code, "
                "card: renderCard(fn, i, rows.length), "
                "item: renderItemValidation(fn, 'item-0-0')}))));")
    result = json.loads(run_node(script))
    assert [fn["name"] for fn in result] == expected_names
    assert [fn["code"] for fn in result] == expected_codes
    assert "Historical review" in result[0]["card"]
    assert library.to_dict() == before


@pytest.mark.parametrize("stale_field", ["behavior_evidence_stale", "evaluation_stale"])
def test_cached_historical_report_cannot_revive_invalidated_evidence(stale_field):
    fn = _claim_fn(_scores(behavior_specificity=.8))
    attach_task_support(fn)
    fn.scores["judge_policy_hash"] = "a" * 64
    library = FunctionLibrary(functions=[fn])
    saved = build_data(library)
    assert saved["functions"][0]["verdict"]["code"] == "GOOD"
    fn.scores[stale_field] = True
    card = refresh_report_data(saved, library)["functions"][0]
    assert card["verdict"]["code"] == "UNSCORED"
    assert card["metrics"]["behavior_purity"] == -1


@pytest.mark.parametrize("missing", ["recall", "precision"])
def test_report_does_not_turn_missing_association_scores_into_zero(missing):
    scores = _scores(behavior_specificity=.8)
    scores.pop(missing)
    fn = _claim_fn(scores, name="pending")
    data = _extract_autofa_data(render_html(build_data(FunctionLibrary(functions=[fn]))))
    record, = data["functions"]
    assert record["metrics"].get(missing) is None
    assert data["meta"]["n_pending_scores"] == 1
    if missing != "precision":  # Grading uses normalized precision, not raw precision.
        assert record["verdict"]["code"] == "UNSCORED"
        assert data["meta"]["n_functions"] == 0
        assert data["meta"]["n_hidden_functions"] == 1


@pytest.mark.parametrize("prior_judgment", [False, True])
def test_score_prefilter_skip_is_unjudged_without_erasing_history(prior_judgment):
    prefilter = {"status": "skip", "reason": "Failure association is below the threshold.",
                 "criteria": {"min_ppv_at_pi0": 0.4}, "metrics": {"ppv_at_pi0": 0.2},
                 "version": 1}
    rows = [{"item_id": "old-confirmed", "verdict": "yes"},
            {"item_id": "old-error", "verdict": "error", "rationale": "Previous request failed"}]
    scores = (_scores(behavior_specificity=0.9, behavior_verdicts=rows, behavior_purity=0.5,
                      behavior_n_judged=2, behavior_n_completed=1, behavior_n_yes=1,
                      behavior_n_errors=1) if prior_judgment else
              {"evaluated": True, "n_items": 100, "ppv_at_pi0": 0.2, "recall": 0.3})
    scores["judge_prefilter"] = prefilter
    original = deepcopy(scores)
    validation = behavior_validation(scores)
    assert validation["status"] == "skipped_pre_filter"
    assert validation["judge_prefilter"] == prefilter
    assert validation["reason"] == prefilter["reason"]
    assert validation["ranking_eligible"] is False
    assert validation["judgment_history_only"] is prior_judgment
    assert validation["verdicts"] == (rows if prior_judgment else [])
    assert resolved_behavior_purity(scores) is None
    assert resolved_behavior_specificity(scores) is None
    assert extract_metrics(scores)["behavior_purity"] == -1.0
    assert extract_metrics(scores)["behavior_specificity"] == -1.0
    assert verdict_for_scores(scores)["label"] == "NOT EVALUATED"
    assert point_from_scores(scores, ["behavior_purity", "behavior_specificity"]) is None

    feedback = summarize_judge_feedback(scores, name="candidate")
    assert feedback["status"] == "skipped_pre_filter"
    assert feedback["judge_prefilter"] == prefilter
    assert feedback["reason"] == prefilter["reason"]
    assert feedback["findings"] == []
    assert scores == original


def test_nonrejecting_prefilter_preserves_normal_judgment_behavior():
    status = "inconclusive"
    scores = _scores(behavior_specificity=0.9,
                     judge_prefilter={"status": status, "reason": "No rejection"})
    fn = _claim_fn(scores)
    attach_task_support(fn)
    validation = behavior_validation(scores)
    assert validation["status"] == "confirmed" and validation["ranking_eligible"]
    assert resolved_behavior_purity(scores) == 1.0
    assert resolved_behavior_specificity(scores) == 0.9
    assert verdict_for_scores(scores)["label"] == "GOOD"
    assert summarize_judge_feedback(scores, name="candidate")["judge_prefilter"]["status"] == status


def test_prefilter_skip_with_archived_purity_is_excluded_from_recall_only_ranking():
    rejected = FeatureFunction(name="rejected", source="def f(item): return True",
                               description="Claim", mechanism="Claim")
    rejected.scores = _scores(
        recall=0.9, behavior_verdicts=[{"item_id": "archived", "verdict": "yes"}],
        judge_prefilter={"status": "skip", "reason": "Low association"},
    )
    original = deepcopy(rejected.scores)
    unjudged = FeatureFunction(name="unjudged", source="def f(item): return True",
                              description="Unjudged candidate")
    unjudged.scores = {"evaluated": True, "n_items": 100, "recall": 0.3}
    cfg = Config(objective_axes=["recall"], objective_weights=[1])
    rows = rank_functions([rejected.to_dict(), unjudged.to_dict()], config=cfg)
    assert [row["name"] for row in rows] == ["unjudged", "rejected"]
    assert rows[0]["ranking"]["eligible"] is True
    assert rows[1]["ranking"]["eligible"] is False
    assert rows[1]["ranking"]["status"] == "skipped_pre_filter"
    assert rejected.scores == original


def test_prefilter_for_an_old_claim_does_not_reject_the_current_claim():
    claim = {"source": "def f(item): return True", "description": "Claim", "mechanism": "Claim"}
    scores = _scores(judge_prefilter={"status": "skip", "claim_hash": "old-claim"})
    current = current_behavior_scores(scores, **claim)
    assert current["judge_prefilter_stale"] is True
    assert behavior_validation(current)["status"] == "confirmed"
    assert resolved_behavior_purity(current) == 1.0
    assert "judge_prefilter_stale" not in scores
    scores["judge_prefilter"]["claim_hash"] = claim_hash(**claim)
    assert behavior_validation(current_behavior_scores(scores, **claim))["status"] == (
        "skipped_pre_filter"
    )


def test_failure_card_marks_prefilter_skip_and_keeps_archived_judgments():
    fn = FeatureFunction(name="candidate", source="def f(item): return True", description="Claim")
    old_rows = [{"item_id": "old", "verdict": "yes"}]
    fn.scores = _scores(behavior_specificity=0.9, behavior_verdicts=old_rows,
                        behavior_n_judged=1, behavior_n_completed=1, behavior_n_yes=1,
                        judge_prefilter={"status": "skip", "reason": "Low failure association"})
    card = build_data(FunctionLibrary(functions=[fn]))["functions"][0]
    assert card["behavior_validation"]["status"] == "skipped_pre_filter"
    assert card["behavior_validation"]["judgment_history_only"] is True
    assert card["behavior_validation"]["verdicts"] == old_rows
    assert card["metrics"]["behavior_purity"] == card["metrics"]["behavior_specificity"] == -1.0
    assert card["ranking"]["eligible"] is False
    assert card["verdict"]["label"] == "NOT EVALUATED"
    assert fn.scores["behavior_purity"] == 1.0 and fn.scores["behavior_verdicts"] == old_rows


def test_changed_claim_hides_stale_purity_and_defers_quality_badge():
    claim = {"source": "def f(item): return True", "description": "Fails to act",
             "mechanism": "The assistant abandons the required action."}
    scores = _scores(claim_hash=claim_hash(**claim))
    current = current_behavior_scores(scores, **claim)
    assert behavior_validation(current)["status"] == "confirmed"
    claim["description"] += " revised"
    stale = current_behavior_scores(scores, **claim)
    assert stale["behavior_purity"] == 1.0  # Retain original result for inspection.
    assert resolved_behavior_purity(stale) is None
    assert behavior_validation(stale)["status"] == "stale"
    assert verdict_for_scores(stale)["label"] == "NOT EVALUATED"
    assert "behavior_judgment_stale" not in scores


def test_report_preserves_component_reviews_and_separates_processing_errors():
    fn = FeatureFunction(name="f", description="Failure", mechanism="Required action omitted",
                         source="def f(item): return True")
    reviews = [
        {"item_id": "a", "verdict": "no", "claim_assessments": [
            {"requirement_id": "consequence", "status": "contradicted",
             "rationale": "A later tool call completes the action."}]},
        {"item_id": "b", "verdict": "uncertain"},
        {"item_id": "c", "verdict": "error", "validation_error": "invalid citation"},
    ]
    fn.scores = _scores(
        behavior_purity=0.0, behavior_n_judged=3, behavior_n_completed=1,
        behavior_n_yes=0, behavior_n_no=1, behavior_n_uncertain=1, behavior_n_errors=1,
        behavior_verdicts=reviews,
        claim_hash=claim_hash(source=fn.source, description=fn.description, mechanism=fn.mechanism),
        claim_contract={"requirements": [{"id": "consequence", "text": fn.mechanism}]},
    )
    data = build_data(FunctionLibrary(functions=[fn]))
    record = data["functions"][0]
    validation = record["behavior_validation"]
    assert validation["confirmed"] == 0
    assert validation["rejected"] == validation["uncertain"] == validation["processing_errors"] == 1
    assert validation["sampled"] == 3
    assert validation["verdicts"] == reviews
    assert validation["claim_contract"] == fn.scores["claim_contract"]
    assert record["metrics"]["behavior_validation_status"] == "mixed"
    # Processing failures leave the review unfinished; they are not negative judgments.
    assert record["verdict"]["label"] == "NOT EVALUATED"


def test_processing_failure_is_distinct_from_evidence_uncertainty():
    common = {"behavior_n_yes": 0, "behavior_n_completed": 0, "behavior_purity": None}
    error = behavior_validation(_scores(**common, behavior_n_errors=4))
    uncertain = behavior_validation(_scores(**common, behavior_n_uncertain=4))
    assert error["status"] == "error" and error["uncertain"] == 0
    assert uncertain["status"] == "uncertain" and uncertain["processing_errors"] == 0


@pytest.mark.parametrize("final_verdict,check_status", [
    ("no", "rejected"), ("uncertain", "uncertain"),
])
def test_report_scores_final_verification_only(final_verdict, check_status):
    row = {
        "item_id": "reviewed", "verdict": final_verdict, "verification_status": check_status,
        "primary_review_path": "/primary", "verification_review_path": "/primary/confirmation",
        "initial_review": {"verdict": "yes", "rationale": "Provisional support", "citations": []},
    }
    completed = int(final_verdict in {"yes", "no"})
    fn = FeatureFunction(name="f", source="def f(item): return True", description="Claim")
    fn.scores = _scores(
        behavior_n_judged=1, behavior_n_completed=completed,
        behavior_n_yes=int(final_verdict == "yes"), behavior_n_no=int(final_verdict == "no"),
        behavior_n_uncertain=int(final_verdict == "uncertain"),
        behavior_n_errors=int(final_verdict == "error"),
        behavior_purity=float(final_verdict == "yes") if completed else None,
        behavior_verdicts=[row],
    )
    record = build_data(FunctionLibrary(functions=[fn]))["functions"][0]
    validation = record["behavior_validation"]
    assert validation["sampled"] == 1
    assert validation["confirmed"] == int(final_verdict == "yes")
    assert validation["verdicts"][0] == row
    assert record["metrics"]["behavior_purity"] == (
        float(final_verdict == "yes") if completed else -1.0
    )


def test_malformed_saved_strategy_is_stale_instead_of_crashing():
    claim = {"source": "def f(item): return True", "description": "Claim", "mechanism": "Claim"}
    scores = _scores(claim_hash=claim_hash(**claim), judge_objective_strategy=[])
    assert current_behavior_scores(scores, **claim)["behavior_judgment_stale"] is True
    assert behavior_validation(scores)["status"] == "stale"


@pytest.mark.parametrize("overrides", [
    pytest.param({"judge_policy_hash": None}, id="missing-policy"),
    pytest.param({"judge_policy_hash": "previous-static-policy"}, id="previous-policy"),
    pytest.param({"claim_hash": None}, id="missing-claim-provenance"),
    pytest.param({"claim_hash": "A" * 64}, id="malformed-claim-provenance"),
])
def test_current_contract_without_current_policy_or_provenance_cannot_confirm_or_rank(overrides):
    fn = _claim_fn(_scores(behavior_specificity=0.9, **overrides))
    current = current_behavior_scores(
        fn.scores, source=fn.source, description=fn.description, mechanism=fn.mechanism,
    )
    assert current["behavior_judgment_stale"] is True
    assert behavior_validation(fn.scores)["status"] == "stale"
    assert resolved_behavior_purity(fn.scores) is None
    assert resolved_behavior_specificity(fn.scores) is None
    assert verdict_for_scores(fn.scores)["label"] == "NOT EVALUATED"
    assert point_from_scores(fn.scores, ["behavior_purity", "behavior_specificity"]) is None
    assert library_frontier_points([fn], ["behavior_purity"]) == []
    card = build_data(FunctionLibrary(functions=[fn]))["functions"][0]
    assert card["behavior_validation"]["status"] == "stale"
    assert card["metrics"]["behavior_purity"] == -1.0
    assert fn.scores["behavior_purity"] == 1.0
    assert "behavior_judgment_stale" not in fn.scores


def test_terminal_views_resolve_edited_claim_without_changing_saved_scores():
    fn = FeatureFunction(name="f", source="def f(item): return True", description="Claim")
    fn.scores = _scores()
    fn.description = "Changed claim"
    library = FunctionLibrary(functions=[fn])
    node = _build_tree(library)[0]
    assert node.data["_verdict"]["label"] == "NOT EVALUATED"
    assert node.data["scores"]["behavior_judgment_stale"] is True
    assert fn.scores["behavior_purity"] == 1.0
    assert "behavior_judgment_stale" not in fn.scores


def test_report_displays_candidates_without_granting_validated_ranking():
    fn = FeatureFunction(name="pending", source="def f(item): return True", description="Claim")
    fn.scores = _scores(behavior_n_yes=1, behavior_n_completed=1, behavior_n_uncertain=3,
                        behavior_purity=.25, behavior_specificity=.9)
    library = FunctionLibrary(functions=[fn])
    card = build_data(library)["functions"][0]
    assert card["ranking"]["eligible"] is False
    assert card["ranking"]["decisive_coverage"] == .25


def test_prompt_only_change_invalidates_unchanged_claim(monkeypatch):
    claim = {"source": "def f(item): return True", "description": "Claim", "mechanism": "Claim"}
    scores = _scores(claim_hash=claim_hash(**claim))
    assert behavior_validation(scores)["status"] == "confirmed"
    monkeypatch.setattr(
        claude_judge, "JUDGE_SYSTEM", claude_judge.JUDGE_SYSTEM + "\nStricter policy.",
    )
    assert scores["claim_hash"] == claim_hash(**claim)
    current = current_behavior_scores(scores, **claim)
    assert behavior_validation(current)["status"] == "stale"
    fn = {"name": "old", **claim, "scores": scores}
    fresh = {"name": "fresh", **claim, "scores": _scores(behavior_purity=0.5)}
    ranked = rank_functions([fn, fresh], config=Config(objective_axes=["behavior_purity"],
                                                       objective_weights=[1]))
    assert ranked[0]["name"] == fresh["name"]
    assert ranked[0]["ranking"]["eligible"]


def test_policy_hash_tracks_evidence_and_contract_version(monkeypatch):
    previous = judge_policy_hash()
    for version_field in ("EVIDENCE_VERSION", "JUDGE_CONTRACT_VERSION"):
        with monkeypatch.context() as patch:
            patch.setattr(claude_judge, version_field, "changed-version")
            assert judge_policy_hash() != previous, version_field


def test_historical_run_configuration_does_not_invalidate_matching_policy():
    scores = _scores(judge_model="recorded-model", judge_rubric_hash="recorded-run-configuration",
                     judge_effort="high", judge_thinking="adaptive")
    validation = behavior_validation(scores)
    assert validation["status"] == "confirmed"
    assert validation["judge_model"] == "recorded-model"
    assert validation["judge_effort"] == "high"
    assert validation["judge_thinking"] == "adaptive"
    assert resolved_behavior_purity(scores) == 1.0
    # Legacy records without the effort settings report them as unspecified.
    legacy = behavior_validation(_scores(judge_model="recorded-model"))
    assert legacy["judge_effort"] is None and legacy["judge_thinking"] is None


def test_report_uses_stable_claim_hash_with_distinct_compiled_contract():
    fn = FeatureFunction(name="f", source="def f(item): return True",
                         description="The agent abandons the required action.")
    contract = build_claim_contract(
        source=fn.source, description=fn.description, mechanism=fn.description,
        defining_clauses=[{
            "id": "clause_1", "source_field": "description",
            "quote": "abandons the required action",
            "required_review_scope": "local",
            "predicate": "The required action remains abandoned.",
        }],
    )
    assert contract["contract_id"] != contract["claim_hash"]
    fn.scores = _scores(claim_hash=contract["claim_hash"], claim_contract=contract)
    record = build_data(FunctionLibrary(functions=[fn]))["functions"][0]
    assert record["behavior_validation"]["status"] == "confirmed"
    assert record["behavior_validation"]["claim_contract"] == contract
    assert record["metrics"]["behavior_purity"] == 1.0


def test_ranking_excludes_stale_and_known_legacy_judgments():
    current = FeatureFunction(
        name="current", source="def f(item): return True", description="Claim",
    )
    current.scores = _scores(behavior_purity=0.5, behavior_n_yes=2, behavior_n_no=2)
    current.scores["claim_hash"] = claim_hash(
        source=current.source, description=current.description, mechanism=current.description,
    )
    legacy = FeatureFunction(name="legacy", source=current.source, description=current.description)
    legacy.scores = _scores(judge_contract_version=None, behavior_purity=1.0)
    stale = FeatureFunction(name="stale", source=current.source, description=current.description)
    stale.scores = {**current.scores, "behavior_purity": 1.0}
    stale.description = "Changed claim"
    axes = ["behavior_purity"]
    assert point_from_scores({"behavior_purity": 0.9}, axes) == (0.9,)
    assert point_from_scores(
        {"behavior_purity": 0.9, "behavior_judgment_stale": True}, axes,
    ) is None
    assert library_frontier_points([legacy, stale, current], axes) == [(0.5,)]
    cfg = Config(objective_axes=axes, objective_weights=[1.0], behavior_strictness="causal")
    ranked = rank_functions([fn.to_dict() for fn in [legacy, stale, current]], config=cfg)
    assert ranked[0]["name"] == "current"


def test_browser_badges_match_backend_across_validation_states():
    cases = [
        _scores(), _scores(judge_contract_version=None),
        _scores(behavior_judgment_stale=True),
        _scores(behavior_n_judged=0, behavior_n_completed=0),
        _scores(behavior_purity=0.25, behavior_n_yes=1, behavior_n_no=3),
        _scores(behavior_purity=0.75, behavior_n_yes=3, behavior_n_uncertain=1),
        _scores(behavior_purity=None, behavior_n_yes=0, behavior_n_completed=0,
                behavior_n_uncertain=4),
        _scores(behavior_purity=None, behavior_n_yes=0, behavior_n_completed=0,
                behavior_n_errors=4),
        _scores(judge_prefilter={"status": "skip", "reason": "Low association"}),
        _scores(ppv_at_pi0=0.2, judge_prefilter={"status": "skip"}),
        _scores(judge_prefilter={"status": "pass"}),
        _scores(behavior_specificity=0.2),
        _scores(behavior_specificity=0.4),
    ]
    script = (verdict_js_classify_fn() + "\nconsole.log(JSON.stringify("
              + json.dumps([extract_metrics(case) for case in cases])
              + ".map(classifyFromMetrics)));")
    assert json.loads(run_node(script)) == [verdict_for_scores(case) for case in cases]


# Incomplete judgments remain candidates, never fabricated quality measurements.

CLAIM = {"source": "def detect(item): return True", "description": "Claim", "mechanism": "Claim"}
AXES = ["ppv_at_pi0", "recall", "behavior_purity", "behavior_specificity"]


def _judged_scores(yes=1, no=0, uncertain=0, errors=0, **overrides):
    total = yes + no + uncertain + errors
    return {
        "evaluated": True, "n_items": 100, "ppv_at_pi0": .9, "precision": .9,
        "recall": .5, "pct_caught": .3, "judge_backend": "claude_agent_sdk",
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(), "claim_hash": claim_hash(**CLAIM),
        "behavior_n_judged": total, "behavior_n_completed": yes + no,
        "behavior_n_yes": yes, "behavior_n_no": no,
        "behavior_n_uncertain": uncertain, "behavior_n_errors": errors,
        "behavior_purity": yes / total if yes + no else None,
        "behavior_specificity": .9 if yes + no else None, **overrides,
    }


@pytest.mark.parametrize("remaining", [{"uncertain": 9}, {"errors": 9}])
def test_one_confirmation_does_not_turn_unresolved_reviews_into_mismatches(remaining):
    result = _judged_scores(**remaining)
    validation = behavior_validation(result)
    assert validation["status"] == "incomplete"
    assert validation["decisive"] == 1
    assert validation["decisive_coverage"] == .1
    assert not validation["ranking_eligible"]
    assert result["behavior_purity"] == .1  # The conservative rate is unchanged.
    # Processing errors cannot establish a weak detector; completed uncertainty
    # still contributes to the existing conservative adherence measurement.
    assert verdict_for_scores(result)["label"] == (
        "NOT EVALUATED" if remaining.get("errors") else "WEAK"
    )
    assert point_from_scores(result, AXES) is None


def test_no_decisive_reviews_have_no_quality_point():
    result = _judged_scores(yes=0, uncertain=10)
    assert point_from_scores(result, AXES) is None
    assert not behavior_validation(result)["ranking_eligible"]
    assert behavior_validation(result)["decisive_coverage"] == 0
    assert verdict_for_scores(result)["label"] == "NOT EVALUATED"


def test_rejected_claim_still_has_a_real_zero_measurement():
    result = _judged_scores(yes=0, no=10)
    assert behavior_validation(result)["ranking_eligible"]
    assert point_from_scores(result, ["behavior_purity"]) == (0.0,)
    assert verdict_for_scores(result)["label"] == "WEAK"


@pytest.mark.parametrize("partial", [{"errors": 1}, {"assessment_coverage": .5}])
def test_dataset_errors_cannot_rank_even_when_behavior_review_confirms(partial):
    fn = {"name": "pending", **CLAIM, "scores": {**_judged_scores(), **partial}}
    assert library_frontier_points([fn], AXES) == []
    candidate = rank_functions([fn])[0]
    assert not candidate["ranking"]["eligible"]
    assert candidate["ranking"]["status"] == "evaluation_incomplete"
    assert library_frontier_metrics([fn])["n_research_candidates"] == 1


def test_unevaluated_function_is_candidate_even_for_association_only_objectives():
    fn = {"name": "new", **CLAIM, "scores": {}}
    cfg = Config(objective_axes=["recall"], objective_weights=[1])
    candidate = rank_functions([fn], config=cfg)[0]
    assert not candidate["ranking"]["eligible"]
    assert candidate["ranking"]["status"] == "evaluation_incomplete"


@pytest.mark.parametrize("overrides", [
    {"behavior_judgment_stale": True}, {"judge_policy_hash": "old-policy"},
    {"behavior_specificity": None},
])
def test_unavailable_requested_behavior_measurements_are_unranked(overrides):
    assert point_from_scores(_judged_scores(**overrides), AXES) is None


def test_real_function_requires_provenance_but_raw_numerical_helpers_remain_usable():
    raw = {"behavior_purity": .8, "evaluated": True}
    assert point_from_scores(raw, ["behavior_purity"]) == (.8,)
    fn = {"name": "candidate", **CLAIM, "scores": raw}
    assert library_frontier_points([fn], ["behavior_purity"]) == []
    result = rank_functions([fn], config=Config(objective_axes=["behavior_purity"],
                                                objective_weights=[1]))
    assert not result[0]["ranking"]["eligible"]


def test_research_candidates_remain_visible_but_cannot_displace_ranked_functions():
    candidate = {"name": "pending", **CLAIM, "scores": _judged_scores(uncertain=9)}
    measured = {"name": "measured", **CLAIM,
                "scores": _judged_scores(yes=0, no=10, ppv_at_pi0=.4, recall=.1)}
    original = deepcopy([candidate, measured])
    cfg = Config(behavior_strictness="causal", objective_axes=AXES, objective_weights=[25] * 4)
    ranked = rank_functions([candidate, measured], config=cfg)
    assert [row["name"] for row in ranked] == ["measured", "pending"]
    assert [row["ranking"]["eligible"] for row in ranked] == [True, False]
    assert [candidate, measured] == original
    metrics = library_frontier_metrics([candidate, measured], config=cfg)
    assert metrics["points"] == [(.4, .1, 0.0, .9)]
    assert metrics["n_rank_eligible"] == metrics["n_research_candidates"] == 1


def test_association_only_objectives_do_not_require_behavior_validation():
    candidate = {"name": "candidate", **CLAIM, "scores": _judged_scores(yes=0, uncertain=10)}
    cfg = Config(objective_axes=AXES, objective_weights=[50, 50, 0, 0])
    metrics = library_frontier_metrics([candidate], config=cfg)
    assert metrics["objective_axes"] == ["ppv_at_pi0", "recall"]
    assert metrics["points"] == [(.9, .5)]
    assert rank_functions([candidate], config=cfg)[0]["ranking"]["eligible"]
    assert behavior_ranking_eligibility(candidate["scores"], ["recall"])["eligible"]


def test_incomplete_behavior_blocks_only_comparisons_requiring_it():
    result = _judged_scores(uncertain=9)
    assert not incomplete_evidence(result)  # Association evaluation is independently complete.
    assert incomplete_evidence(result, axes=AXES)
    assert not incomplete_evidence(result, axes=["recall"])
    snapshot = {"eval_mode": "full", "objective_axes": AXES,
                "results": [{"scores": result}]}
    assert not completed_evaluation(snapshot)
    assert completed_evaluation({**snapshot, "objective_axes": ["recall"]})
    assert completed_evaluation({**snapshot, "results": [{"scores": _judged_scores(yes=0, no=10)}]})


def test_unknown_processing_and_stale_cases_cannot_establish_behavior_plateau():
    for result in [_judged_scores(yes=0, uncertain=10), _judged_scores(errors=9),
                   _judged_scores(behavior_judgment_stale=True)]:
        assert not completed_evaluation({"objective_axes": AXES,
                                         "results": [{"scores": result}]})


def test_comparable_evaluations_keep_completed_history_without_mutation():
    complete = {"eval_mode": "full", "evaluation_complete": True, "frontier_value": .9}
    partial = {"eval_mode": "full", "evaluation_complete": False, "frontier_value": .4}
    history = [complete, partial]
    before = deepcopy(history)
    assert comparable_evaluations(history) == [complete]
    assert history == before
