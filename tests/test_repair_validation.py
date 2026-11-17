# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review retries and quality views count only complete, current and valid reviews."""

import json
import shutil
import subprocess
from copy import deepcopy

import pytest

from autofa.mcp import tools_eval, tools_library
from autofa.metrics.adaptive_review import merge_expanded_review
from autofa.metrics.claim_compiler import compiler_policy_hash
from autofa.metrics.claim_contract import build_claim_contract
from autofa.metrics.claude_judge import compatible_judge_policy, judge_policy_hash
from autofa.metrics.repair_validation import merge_processing_retry
from autofa.metrics.review_failures import processing_failure
from autofa.metrics.verdicts import (
    current_behavior_scores,
    extract_metrics,
    verdict_for_scores,
    verdict_js_classify_fn,
)
from autofa.objectives import apply_preset, discovery_strategy
from autofa.reports.failure_cards import build_data
from autofa.session.discovery_target import (
    discovery_good_target_for_runtime,
    discovery_grade_feedback,
)

from support.repair_review import _runtime
from support.task_support import _function

CLAIM = {"description": "The assistant writes incorrect rewards.",
         "mechanism": "It guesses an undocumented rate.", "source": "def detect(item): True"}


def _reviews():
    provenance = {"claim_hash": "claim", "judge_policy_hash": "policy",
                  "judge_contract_version": "v2", "judge_model": "original"}
    previous = {
        **provenance, "claim_contract": {"contract_id": "frozen"},
        "behavior_specificity": 0.8,
        "behavior_verdicts": [
            {"item_id": "yes", "verdict": "yes", "rationale": "Supported.",
             "claim_contract_id": "frozen"},
            {"item_id": "no", "verdict": "no", "rationale": "Contradicted.",
             "claim_contract_id": "frozen"},
            {"item_id": "unknown", "verdict": "uncertain", "rationale": "Missing fact.",
             "claim_contract_id": "frozen"},
            {"item_id": "error", "verdict": "error", "rationale": "Timed out."},
        ],
    }
    fresh = {
        **provenance, "behavior_specificity": 0.5,
        "verdicts": [{"item_id": "error", "verdict": "yes", "claim_contract_id": "frozen"}],
    }
    return previous, fresh


def test_retry_preserves_denominator_and_all_semantic_counterexamples():
    before, fresh = _reviews()
    # A changed search strategy is not part of the frozen review policy.
    before["judge_objective_strategy"], fresh["judge_objective_strategy"] = "precision", "recall"
    original = deepcopy(before)
    result = merge_processing_retry(before, fresh, {"error"})
    assert result["n_judged"] == 4 and result["n_yes"] == 2
    assert result["n_no"] == result["n_uncertain"] == 1 and result["n_errors"] == 0
    assert result["behavior_purity"] == 0.5
    assert result["behavior_specificity"] == pytest.approx(0.7)
    assert result["verdicts"][:3] == before["behavior_verdicts"][:3]
    assert before == original


def test_retry_recomputes_incomplete_review_count_over_preserved_full_cohort():
    before, fresh = _reviews()
    before["behavior_verdicts"][2]["completion_status"] = "needs_evidence_review"
    before["behavior_n_review_incomplete"] = 1
    fresh["verdicts"][0].update(verdict="uncertain", completion_status="needs_evidence_review")
    fresh["n_review_incomplete"] = 1
    result = merge_processing_retry(before, fresh, {"error"})
    assert result["n_review_incomplete"] == 2
    assert result["n_uncertain"] == 2
    assert result["verdicts"][2] == before["behavior_verdicts"][2]


# The saved evidence provenance key differs from the fresh payload's key.
@pytest.mark.parametrize("saved,retried,match", [
    ({}, {"claim_hash": "changed"}, "claim_hash"),
    ({"judge_thinking_budget_tokens": 2048}, {}, "judge_thinking_budget_tokens"),
    ({"judge_evidence_version": "original"}, {"evidence_version": None}, "judge_evidence_version"),
    ({"judge_evidence_version": "original"}, {"evidence_version": "other"},
     "judge_evidence_version"),
], ids=["changed_claim", "unknown_vs_explicit_setting", "missing_evidence", "changed_evidence"])
def test_retry_refuses_changed_or_unknown_review_provenance(saved, retried, match):
    before, fresh = _reviews()
    before.update(saved)
    fresh.update(retried)
    with pytest.raises(ValueError, match=match):
        merge_processing_retry(before, fresh, {"error"})


def test_retry_preserves_identical_explicit_review_settings():
    settings = {"judge_effort": "medium", "judge_thinking": "adaptive",
                "judge_objective_strategy": "recall", "judge_evidence_version": "v4"}
    before, fresh = _reviews()
    before.update(settings)
    fresh.update(settings, evidence_version="v4")
    result = merge_processing_retry(before, fresh, {"error"})
    assert {key: result[key] for key in settings} == settings


def test_retry_preserves_legacy_unspecified_settings_without_inventing_provenance():
    before, fresh = _reviews()
    fields = ("judge_model", "judge_effort", "judge_thinking")
    for payload in (before, fresh):
        for field in fields:
            payload.pop(field, None)
    result = merge_processing_retry(before, fresh, {"error"})
    assert result["n_judged"] == 4
    assert all(result.get(field) is None for field in fields)


@pytest.mark.parametrize("item_id", ["no", "unseen"])
def test_retry_cannot_replace_semantic_or_unrecorded_items(item_id):
    before, fresh = _reviews()
    fresh["verdicts"][0]["item_id"] = item_id
    with pytest.raises(ValueError, match="Only processing errors"):
        merge_processing_retry(before, fresh, {item_id})


@pytest.mark.parametrize("change,match", [
    ({"claim_contract_id": "weaker"}, "same frozen"),  # A different frozen decomposition.
    ({"processing_status": "error"}, "Processing status"),  # Semantic output marked failed.
    ({"item_id": None}, "item IDs"),
])
def test_retry_rejects_invalid_retried_rows(change, match):
    before, fresh = _reviews()
    fresh["verdicts"][0].update(change)
    with pytest.raises(ValueError, match=match):
        merge_processing_retry(before, fresh, {"error"})


def test_failed_retry_excludes_remaining_errors_and_keeps_the_original_frozen_contract():
    before, fresh = _reviews()
    fresh["verdicts"][0].update(verdict="error", claim_contract_id="fallback")
    fresh["claim_contract"] = {"contract_id": "fallback"}
    result = merge_processing_retry(before, fresh, {"error"})
    assert result["claim_contract"] == before["claim_contract"]
    assert result["behavior_purity"] == pytest.approx(1 / 3) and result["n_errors"] == 1
    assert result["n_yes"] == result["n_no"] == result["n_uncertain"] == 1


@pytest.mark.parametrize("payload,key,value", [
    ("before", "behavior_n_judged", 100), ("fresh", "n_judged", True),
    # The recorded completion count must match the rows' completion_status.
    ("before", "behavior_n_review_incomplete", 1),
])
def test_retry_refuses_truncated_cohorts_and_inconsistent_counts(payload, key, value):
    before, fresh = _reviews()
    (before if payload == "before" else fresh)[key] = value
    with pytest.raises(ValueError, match="cohort count"):
        merge_processing_retry(before, fresh, {"error"})


def test_all_error_fallback_retains_original_rubric_provenance():
    before, fresh = _reviews()
    before.update(judge_rubric_hash="original", judge_evidence_version="v4")
    fresh["verdicts"][0]["verdict"] = "error"
    result = merge_processing_retry(before, fresh, {"error"})
    assert result["judge_rubric_hash"] == "original" and result["evidence_version"] == "v4"


@pytest.mark.parametrize("bad", [2, "0.5"])
def test_retry_rejects_invalid_specificity_values(bad):
    before, fresh = _reviews()
    fresh["behavior_specificity"] = bad
    with pytest.raises(ValueError, match="specificity"):
        merge_processing_retry(before, fresh, {"error"})


def test_retry_rejects_changed_evidence_under_the_same_item_id():
    before, fresh = _reviews()
    before["behavior_verdicts"][-1]["snapshot_id"] = "old-content"
    fresh["verdicts"][0]["snapshot_id"] = "new-content"
    with pytest.raises(ValueError, match="evidence snapshot"):
        merge_processing_retry(before, fresh, {"error"})


def test_retry_preserves_item_editorial_advice_instead_of_replacing_it_with_heuristics():
    before, fresh = _reviews()
    cal = {"status": "overclaim", "claim_note": "The mechanism overstates certainty.",
           "detector_note": "Behavior is visible.", "rewrite_hint": "Inspect the causal premise."}
    before["item_calibrations"] = [{"item_id": "unknown", "claim_calibration": cal}]
    fresh["item_calibrations"] = [{"item_id": "error", "claim_calibration": cal}]
    result = merge_processing_retry(before, fresh, {"error"})
    assert result["claim_calibration"] == cal
    assert "Inspect the causal premise" in result["calibration_hint"]
    assert len(result["item_calibrations"]) == 2
    fresh["item_calibrations"][0]["item_id"] = "unseen"
    with pytest.raises(ValueError, match="calibration falls outside"):
        merge_processing_retry(before, fresh, {"error"})


def test_expansion_aggregate_excludes_processing_errors_without_losing_counterexamples():
    original = {"verdicts": [{"item_id": "yes", "verdict": "yes"}]}
    added = {"verdicts": [{"item_id": "unknown", "verdict": "uncertain"},
                          {"item_id": "error", "verdict": "error"}]}
    result = merge_expanded_review(original, added)
    assert result["behavior_purity"] == .5
    assert result["n_errors"] == result["n_uncertain"] == 1
    assert result["n_no"] == 0 and len(result["verdicts"]) == 3


def _compiled(coverage=True, predicate="The written rewards are incorrect.", **claim):
    claim = {**CLAIM, **claim}
    return build_claim_contract(**claim, defining_clauses=[{
        "id": "wrong_rewards", "source_field": "description", "quote": claim["description"],
        "required_review_scope": "local", "predicate": predicate,
    }], compilation={"policy_hash": compiler_policy_hash(), "model": "test", "exclusions": [],
                     **({"coverage_review": {"approved": True, "issues": []}}
                        if coverage else {})})


def _compile_failure():
    """A failed shared compiler stage leaves every row an unreviewed processing error."""
    base = build_claim_contract(**CLAIM)
    failure = {**processing_failure("The audit rejected every draft.",
                                    kind="claim_decomposition_error"),
               "stage": "claim_compilation", "cache_key": "shared", "explicit_retry_used": False}
    provenance = {"claim_hash": base["contract_id"], "judge_policy_hash": "policy",
                  "judge_contract_version": "v2"}
    previous = {
        **provenance, "claim_contract": base,
        "claim_compilation": {"error": "claim compilation failed: rejected", "failure": failure},
        "behavior_verdicts": [
            {"item_id": key, "verdict": "error", "processing_failure": failure,
             "snapshot_id": f"{key}-uncompiled", "claim_contract_id": base["contract_id"]}
            for key in ("a", "b")
        ],
    }
    return previous, provenance, base, failure


def _crashed_batch():
    """judge_function's fallback after a batch crashed before compiling the claim."""
    previous, provenance, base, failure = _compile_failure()
    error = "claude judge unavailable: No space left on device"
    del previous["claim_compilation"]
    previous["behavior_verdicts"] = [
        {"item_id": key, "verdict": "error", "processing_status": "error",
         "validation_error": error, "rationale": error,
         "claim_contract_id": base["contract_id"], "claim_assessments": []}
        for key in ("a", "b")
    ]
    return previous, provenance, base, failure


# A crashed batch saved no compile receipt; its retried stage still decides recovery.
COMPILE_RETRIES = pytest.mark.parametrize("setup,retried", [
    (_compile_failure, ("a",)), (_crashed_batch, ("a", "b")),
], ids=["failed_compile", "crashed_batch"])


@COMPILE_RETRIES
def test_failed_compile_retry_records_the_spent_shared_stage_retry(setup, retried):
    previous, provenance, base, failure = setup()
    spent = {**failure, "explicit_retry_used": True,
             "retry_policy": "requires_backend_or_configuration_change"}
    fresh = {**provenance, "claim_contract": base,
             "claim_compilation": {"error": "claim compilation failed: again", "failure": spent},
             "verdicts": [{"item_id": key, "verdict": "error", "processing_failure": spent,
                           "snapshot_id": f"{key}-uncompiled",
                           "claim_contract_id": base["contract_id"]} for key in retried]}
    result = merge_processing_retry(previous, fresh, set(retried))
    # The shared stage's newer outcome decides whether the claim may be reworded.
    assert result["claim_compilation"]["failure"] == spent
    assert result["claim_contract"] == base and result["n_errors"] == 2


@COMPILE_RETRIES
def test_recovered_compile_retry_freezes_its_first_approved_contract(setup, retried):
    previous, provenance, _, _ = setup()
    compiled = _compiled()
    receipt = {**compiled["compilation"], "error": None, "failure": None, "cache_key": "shared"}

    def fresh(keys):
        return {**provenance, "claim_contract": compiled, "claim_compilation": receipt,
                "verdicts": [{"item_id": key, "verdict": "uncertain",
                              "snapshot_id": f"{key}-compiled",
                              "claim_contract_id": compiled["contract_id"]} for key in keys]}

    result = merge_processing_retry(previous, fresh(retried), set(retried))
    assert result["claim_contract"] == compiled and result["claim_compilation"] == receipt
    if retried == ("a",):
        assert result["n_uncertain"] == result["n_errors"] == 1
        # The retained row never assessed the frozen contract; retrying it later is
        # its first review, not changed evidence.
        current = {**provenance, "claim_contract": compiled, "claim_compilation": receipt,
                   "behavior_verdicts": result["verdicts"]}
        result = merge_processing_retry(current, fresh(["b"]), {"b"})
        assert result["claim_contract"] == compiled and result["claim_compilation"] == receipt
    assert result["n_uncertain"] == 2 and result["n_errors"] == 0


@pytest.mark.parametrize("change", ["unaudited", "other_claim", "frozen_contract"])
def test_only_a_failed_compile_stage_adopts_an_approved_contract_for_its_claim(change):
    previous, provenance, _, _ = _compile_failure()
    compiled = (_compiled(coverage=False) if change == "unaudited" else
                _compiled(description="The assistant writes incorrect totals.")
                if change == "other_claim" else _compiled())
    if change == "frozen_contract":
        # A frozen decomposition is never swapped for a different one.
        previous["claim_contract"] = _compiled(predicate="The rewards differ from the rate.")
    fresh = {**provenance, "claim_contract": compiled,
             "claim_compilation": {**compiled["compilation"], "error": None, "failure": None},
             "verdicts": [{"item_id": "a", "verdict": "uncertain", "snapshot_id": "a-compiled",
                           "claim_contract_id": compiled["contract_id"]}]}
    with pytest.raises(ValueError, match="frozen evidence snapshot|same frozen claim contract"):
        merge_processing_retry(previous, fresh, {"a"})


def _read_quality(rt, fn):
    """Exercise actual MCP, report and stopping read paths without new evaluation."""
    inspected = tools_library.get_library_function(fn.name)
    feedback = tools_library.get_function_feedback(fn.name)
    listed = tools_library.list_library_functions()
    assert inspected["ok"] and feedback["ok"] and listed["ok"]
    target = discovery_good_target_for_runtime(rt, minimum=1)
    bulletin = discovery_grade_feedback(rt, target=target)
    return {
        "labels": [inspected["verdict"]["code"], feedback["verdict"]["code"],
                   listed["functions"][0]["verdict"]["code"],
                   tools_eval._current_verdict(rt, fn)["code"]],
        "grades": [inspected["discovery_grade"]["code"],
                   feedback["discovery_grade"]["code"],
                   listed["functions"][0]["discovery_grade"]["code"],
                   tools_eval._current_discovery_grade(rt, fn)["code"],
                   listed["discovery_grades"]["functions"][0]["code"],
                   bulletin["summary"]["functions"][0]["code"]],
        "target": target,
    }


@pytest.mark.parametrize("preset,sampled", [("precision", True), ("recall", False)])
def test_objective_switch_preserves_same_quality_across_cli_report_mcp_and_minimum(
    tmp_path, monkeypatch, preset, sampled,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = _function("sampled_support" if sampled else "good")
    rt.library.functions = [fn]
    rt.config = apply_preset(rt.config, preset)
    before = deepcopy(fn.to_dict())

    quality = _read_quality(rt, fn)
    assert quality["labels"] == ["GOOD"] * 4
    assert quality["grades"] == ["OUTSTANDING"] * 6
    assert quality["target"]["qualifying_failure_modes"] == [fn.name]
    assert quality["target"]["minimum_met"]
    report = build_data(rt.library)
    assert report["functions"][0]["verdict"]["code"] == "GOOD"
    assert fn.to_dict() == before

    # Supplying the active search strategy never changes the frozen review policy.
    strict = current_behavior_scores(
        fn.scores, source=fn.source, description=fn.description, mechanism=fn.mechanism,
        objective_strategy=discovery_strategy(rt.config), behavior_strictness="causal",
    )
    assert verdict_for_scores(strict)["code"] == "GOOD"


@pytest.mark.parametrize("invalidated", ["judge_policy", "claim"])
def test_quality_views_still_require_current_claim_and_valid_active_judge_policy(
    tmp_path, monkeypatch, invalidated,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = _function("sampled_support")
    rt.library.functions = [fn]
    rt.config = apply_preset(rt.config, "precision")
    if invalidated == "judge_policy":
        rt.config.behavior_strictness = "contextual"
    else:
        fn.description += " An additional unreviewed condition."
    before = deepcopy(fn.to_dict())

    quality = _read_quality(rt, fn)
    assert quality["labels"] == ["UNSCORED"] * 4
    assert quality["grades"] == ["UNSCORED"] * 6
    assert not quality["target"]["minimum_met"]
    assert quality["target"]["qualifying_failure_modes"] == []
    assert fn.to_dict() == before


@pytest.mark.parametrize("issue", ["error", "unread"])
def test_partial_review_cannot_meet_target_even_when_completed_scores_are_good(
    tmp_path, monkeypatch, issue,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = _function("good")
    rt.library.functions = [fn]
    fn.scores.update(behavior_n_judged=6, behavior_n_completed=5)
    if issue == "error":
        fn.scores["behavior_n_errors"] = 1
        fn.scores["behavior_verdicts"].append({"item_id": "pending", "verdict": "error"})
    else:
        fn.scores.update(behavior_n_uncertain=1, behavior_n_review_incomplete=1)
        fn.scores["behavior_verdicts"].append({
            "item_id": "pending", "verdict": "uncertain",
            "completion_status": "needs_evidence_review",
        })
    quality = _read_quality(rt, fn)
    assert quality["labels"] == ["UNSCORED"] * 4
    assert not quality["target"]["minimum_met"]
    assert fn.scores["behavior_purity"] == 1.0  # Measured completed judgments stay available.
    metrics = extract_metrics(fn.scores)
    assert metrics["behavior_processing_pending"]
    assert "incomplete" in verdict_for_scores(fn.scores)["reason"].lower()
    if shutil.which("node"):
        script = (verdict_js_classify_fn()
                  + '\nconsole.log(JSON.stringify(classifyFromMetrics('
                  + json.dumps(metrics) + ')));')
        rendered = json.loads(subprocess.check_output(["node", "-e", script], text=True))
        assert rendered["code"] == "UNSCORED"


def test_binary_evidence_upgrade_preserves_compiler_but_invalidates_old_reviews():
    # The binary projection upgrade changes review identity, not claim semantics.
    previous = "d12927a00d12c8e4ea6b3ac24a89b1dda36a8fc9a9867383c3e36e74a1d60cb3"
    assert compiler_policy_hash("causal") == (
        "2b59e23a60bcc867d7779c07f0767dd096c2e67a3c11b91c1d662653f5f1604a")
    assert judge_policy_hash(behavior_strictness="causal") == (
        "2836da5a7c8c7633fbaab2a89e76cf136a3f467a51f8bd164b602cb74039b070")
    assert not compatible_judge_policy(previous, behavior_strictness="causal")
