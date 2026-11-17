# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounded repair plans distinguish false hits, missing facts, and failed reviews."""

from copy import deepcopy
from inspect import signature

import pytest

from autofa.mcp import tools_data, tools_library
from autofa.metrics.claim_calibration import DEVELOPMENT_REVIEW_GUIDANCE, summarize_judge_feedback
from autofa.metrics.claim_contract import build_claim_contract, claim_hash
from autofa.metrics.repair_feedback import (
    MAX_CITATIONS,
    MAX_EXAMPLES,
    MAX_EXCERPT_CHARS,
    build_repair_feedback,
)

from support.repair_review import CLAIM, assessment, citation, reviewed_function


def requirement(fn, requirement_id):
    """The first review row's assessment of ``requirement_id``."""
    return next(a for a in fn["scores"]["behavior_verdicts"][0]["claim_assessments"]
                if a["requirement_id"] == requirement_id)


def test_groups_all_conditions_with_stable_ids_bounded_examples_and_full_scope():
    fn = reviewed_function(*(["contradicted"] * 8 + ["unknown"] * 4 + ["supported"] * 2))
    original = deepcopy(fn)
    result = build_repair_feedback(fn)
    assert fn == original
    assert result["claim_hash"] == claim_hash(**CLAIM)
    assert result["validation"]["sampled"] == 14
    assert len(result["diagnostic_item_ids"]) == 14
    by_status = {row["status"]: row for row in result["findings"]
                 if row["requirement_id"] == "clause_1"}
    assert {key: value["total_examples"] for key, value in by_status.items()} == {
        "supported": 2, "contradicted": 8, "unknown": 4,
    }
    negative = by_status["contradicted"]
    assert negative["examples_omitted"] == 8 - MAX_EXAMPLES
    assert len(negative["examples"]) == MAX_EXAMPLES
    assert negative["item_ids"] == [f"item-{i:03}" for i in range(8)]
    assert negative["requirement"] == "Authoritative rates are unavailable."
    assert negative["claim_quote"] == "authoritative rates are unavailable"
    assert negative["examples"][0]["counterevidence"] == [
        citation("An authoritative source was supplied."),
    ]
    assert negative["examples"][0]["citations"] == [citation()]
    assert result["recommended_action"] == "tighten_detector"
    assert "split_detector" in result["allowed_actions"]
    assert "retire" not in result["allowed_actions"]
    assert result["target_bucket"] == "missing-rate"
    assert result["details_tool"] == {
        "name": "get_function_feedback", "arguments": {"name": "detector"},
    }
    fn["scores"]["behavior_verdicts"].reverse()
    assert build_repair_feedback(fn) == result
    smaller = build_repair_feedback(reviewed_function("contradicted", "unknown", "supported"))
    assert {row["id"] for row in smaller["findings"]} == {row["id"] for row in result["findings"]}


def test_unknown_only_preserves_missing_facts_without_inventing_false_hits_or_a_rewrite():
    result = build_repair_feedback(reviewed_function("unknown", "unknown"))
    assert result["allowed_actions"] == ["gather_evidence"]
    assert result["recommended_action"] == "gather_evidence"
    assert {f["status"] for f in result["findings"]} == {"unknown"}
    assert "upstream rate lookup is missing" in result["investigations"][0]["question"]
    assert "hard negative" not in result["investigations"][0]["question"]


def test_available_unread_evidence_does_not_authorize_narrowing_supported_observations():
    fn = reviewed_function("unknown")
    row = fn["scores"]["behavior_verdicts"][0]
    row["completion_status"] = "needs_evidence_review"
    for requirement in row["claim_assessments"]:
        if requirement["requirement_id"] == "clause_1":
            requirement.update(unknown_reason="review_incomplete",
                               rationale="The required original interval is not fully read.")
        else:
            requirement["status"] = "supported"
    fn["scores"]["claim_calibration"] = {
        "status": "overclaim", "rewrite_hint": "Narrow to the event I read.",
    }
    result = build_repair_feedback(fn)
    assert result["review_completion_only"] is True
    assert result["allowed_actions"] == ["gather_evidence"]
    assert result["recommended_action"] == "gather_evidence"
    assert "calibration_suggestion" not in result
    assert "does not justify narrowing" in " ".join(result["notes"])


def test_supported_development_sample_can_be_retained_without_independent_confirmation():
    fn = reviewed_function("supported", "supported")
    fn["scores"].update(behavior_review_phase="development",
                        behavior_sample={"target_n": 5, "limited": True})
    original = deepcopy(fn)
    result = build_repair_feedback(fn)
    assert result["validation"]["status"] == "limited_evidence"
    assert result["validation"]["confirmed"] == 2
    assert result["recommended_action"] == "retain"
    assert result["allowed_actions"] == []
    assert "review_phase='confirmation'" in " ".join(result["notes"])
    assert DEVELOPMENT_REVIEW_GUIDANCE in " ".join(result["notes"])
    summary = summarize_judge_feedback(fn["scores"], name=fn["name"])
    assert summary["next_steps"] == [DEVELOPMENT_REVIEW_GUIDANCE]
    assert summary["status"] == "limited_evidence"
    assert summary["sample"]["limited"] is True
    assert fn == original


# Semantic verdicts share one path and processing errors another; neither yields advice.
@pytest.mark.parametrize("old_status", ["contradicted", "error"])
def test_score_prefilter_skip_preserves_history_without_semantic_repair_advice(old_status):
    fn = reviewed_function(old_status)
    fn["scores"]["judge_prefilter"] = {
        "status": "skip", "reason": "Weak benchmark association", "claim_hash": claim_hash(**CLAIM),
    }
    original = deepcopy(fn)
    result = build_repair_feedback(fn)
    assert result["validation"]["status"] == "skipped_pre_filter"
    assert result["judge_prefilter"] == fn["scores"]["judge_prefilter"]
    assert result["reason"] == "Weak benchmark association"
    assert result["findings"] == result["investigations"] == result["allowed_actions"] == []
    assert result["recommended_action"] == "reevaluate"
    assert "remove" in " ".join(result["next_steps"])
    assert "defer_function_repair" in " ".join(result["next_steps"])
    assert result["diagnostic_item_ids"] == ["item-000"]
    assert result["judgment_history_only"] is True
    assert "Call judge_function" not in " ".join(result["notes"])
    assert fn == original


@pytest.mark.parametrize("supported_requirement,has_citation,may_revise", [
    ("description", True, True), ("actor_attribution", True, False),
    ("description", False, False),
])
def test_supported_observation_with_unknown_premise_can_offer_a_narrower_claim(
    supported_requirement, has_citation, may_revise,
):
    fn = reviewed_function("unknown")
    requirement(fn, supported_requirement).update({"status": "supported",
        "rationale": "The submitted value is directly visible.",
        "citations": [citation("I wrote 400 points.")] if has_citation else [],
    })
    requirement(fn, "failure_contribution").update({"status": "supported",
        "rationale": "The invented rate produced the incorrect submitted value.",
        "citations": [citation("The submitted value uses 8% instead of the documented 4%.")],
    })
    result = build_repair_feedback(fn)
    assert result["recommended_action"] == "gather_evidence"
    assert ("revise_claim" in result["allowed_actions"]) is may_revise
    assert "tighten_detector" not in result["allowed_actions"]
    assert "retire" not in result["allowed_actions"]
    if may_revise:
        assert "does not confirm the original claim" in " ".join(result["notes"])


@pytest.mark.parametrize("status,verdict,recommended", [
    ("unknown", "uncertain", "gather_evidence"),
    ("contradicted", "no", "tighten_detector"),
])
def test_observed_behavior_with_unproven_causality_cannot_be_repaired_by_narrowing_prose(
    status, verdict, recommended,
):
    fn = reviewed_function("supported")
    scores = fn["scores"]
    row = scores["behavior_verdicts"][0]
    contribution = requirement(fn, "failure_contribution")
    contribution.update({
        "status": status,
        "rationale": (
            "The record omits whether the invented rate affected the delivered result."
            if status == "unknown" else
            "The invented rate was corrected; an unrelated missing attachment caused failure."
        ),
        "counterevidence": [] if status == "unknown" else [citation("The rate was corrected.")],
    })
    row["verdict"] = verdict
    scores.update({
        "behavior_n_yes": 0, "behavior_n_no": int(verdict == "no"),
        "behavior_n_uncertain": int(verdict == "uncertain"),
        "behavior_n_completed": int(verdict == "no"),
        "behavior_purity": 0.0 if verdict == "no" else None,
    })
    original = deepcopy(fn)

    result = build_repair_feedback(fn)

    assert fn == original
    assert result["review_detail_complete"]
    assert result["recommended_action"] == recommended
    # Grounded revisions remain possible, but cannot remove the acceptance check.
    assert "revise_claim" in result["allowed_actions"]
    notes = " ".join(result["notes"])
    assert "failure_contribution acceptance check remains mandatory" in notes
    assert "A narrower claim may describe those supported facts" not in notes
    assert len(result["investigations"]) == 1
    investigation = result["investigations"][0]
    assert contribution["rationale"] in investigation["question"]
    assert investigation["arguments"]["item_id"] == "item-000"
    if status == "unknown":
        assert "Gather evidence of the specific task-breaking defect" in notes
        assert "tighten_detector" not in result["allowed_actions"]
        assert "retire" not in result["allowed_actions"]
    else:
        assert "excludes recovered or unrelated matches" in notes


def test_processing_errors_never_reuse_provisional_or_partial_semantic_findings():
    fn = reviewed_function("error")
    row = fn["scores"]["behavior_verdicts"][0]
    row["initial_review"] = {"verdict": "yes", "claim_assessments": [assessment("supported")]}
    row["claim_assessments"] = [assessment("contradicted")]
    fn["scores"]["claim_calibration"] = {"status": "good", "rewrite_hint": "Keep it."}
    result = build_repair_feedback(fn)
    assert result["allowed_actions"] == ["retry_review"]
    assert result["recommended_action"] == "retry_review"
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["status"] == "processing_error"
    assert finding["examples"][0]["citations"] == []
    assert finding["examples"][0]["rationale"] == "SDK review timed out"
    assert "calibration_suggestion" not in result


def test_confirmed_claim_needs_no_repair_and_retains_supported_clauses():
    result = build_repair_feedback(reviewed_function("supported", "supported"))
    assert result["recommended_action"] == "retain"
    assert result["allowed_actions"] == []
    assert result["investigations"] == []
    assert result["findings"][0]["status"] == "supported"
    assert result["findings"][0]["total_examples"] == 2


@pytest.mark.parametrize("changed", ["source", "legacy"])
def test_stale_and_legacy_reviews_keep_diagnostic_history_without_semantic_repair_advice(changed):
    fn = reviewed_function("contradicted", "supported")
    if changed == "legacy":
        fn["scores"]["judge_contract_version"] = "old-version"
    else:
        fn["source"] += " changed"
    result = build_repair_feedback(fn)
    assert result["validation"]["status"] in {"stale", "legacy"}
    assert result["diagnostic_item_ids"] == ["item-000", "item-001"]
    assert result["findings"] == result["investigations"] == []
    assert result["allowed_actions"] == ["gather_evidence"]
    assert "judge_function" in result["notes"][-1]


@pytest.mark.parametrize("field,tool", [
    ("response_text", "read_field_chunk"), ("trajectory_text", "get_trajectory"),
    ("detector_source", "get_function_feedback"),
])
def test_investigations_use_existing_tools_and_real_arguments(field, tool):
    result = build_repair_feedback(reviewed_function("contradicted", field=field))
    investigation = result["investigations"][0]
    assert investigation["issue_type"] == "counterexample"
    assert investigation["tool"] == tool
    module = tools_library if tool == "get_function_feedback" else tools_data
    signature(getattr(module, tool)).bind(**investigation["arguments"])
    assert investigation["arguments"]["item_id"] == "item-000"


def test_long_citations_are_explicit_excerpts_with_original_locations_and_total_counts():
    fn = reviewed_function("contradicted")
    condition = requirement(fn, "clause_1")
    quote = "Evidence " * 1000
    condition["citations"] = [citation(quote)] * (MAX_CITATIONS + 2)
    result = build_repair_feedback(fn)
    example = next(f for f in result["findings"]
                   if f["requirement_id"] == "clause_1")["examples"][0]
    assert example["total_citations"] == MAX_CITATIONS + 2
    assert len(example["citations"]) == MAX_CITATIONS
    copied = example["citations"][0]
    assert copied["quote"] == quote[:MAX_EXCERPT_CHARS]
    assert copied["end"] == len(quote) + 5 and copied["quote_chars"] == len(quote)
    assert copied["quote_truncated"] is True
    assert condition["citations"][0]["quote"] == quote


def test_component_findings_preserve_the_specific_blocking_condition():
    fn = reviewed_function("contradicted")
    condition = requirement(fn, "mechanism")
    condition["status"] = "supported"
    condition["components"] = [{**assessment("contradicted"),
                                "claim": "authoritative rates are unavailable"}]
    result = build_repair_feedback(fn)
    component = next(f for f in result["findings"] if f.get("component"))
    assert component["requirement_id"] == "mechanism"
    assert component["requirement"] == condition["components"][0]["claim"]
    assert component["status"] == "contradicted"
    assert result["recommended_action"] == "tighten_detector"
    assert "retire" in result["allowed_actions"]


@pytest.mark.parametrize("status", ["supported", "contradicted"])
def test_missing_review_rows_cannot_justify_retain_or_retire(status):
    fn = reviewed_function(status, status)
    fn["scores"]["behavior_verdicts"].pop()
    result = build_repair_feedback(fn)
    assert result["review_detail_complete"] is False
    assert result["recommended_action"] != "retain"
    assert "retire" not in result["allowed_actions"]
    assert "gather_evidence" in result["allowed_actions"]
    assert result["validation"]["sampled"] == 2
    assert result["findings"][0]["total_examples"] == 1


@pytest.mark.parametrize("mutation", [
    "contradicted", "duplicate_item", "wrong_contract", "failed_processing",
])
def test_inconsistent_saved_confirmation_cannot_become_retain_or_repair_proof(mutation):
    fn = reviewed_function("supported")
    row = fn["scores"]["behavior_verdicts"][0]
    if mutation == "contradicted":
        row["claim_assessments"][0]["status"] = mutation
    elif mutation == "duplicate_item":
        fn["scores"]["behavior_verdicts"].append(deepcopy(row))
    elif mutation == "wrong_contract":
        row["claim_contract_id"] = "different-contract"
    else:
        row["processing_status"] = "error"
    result = build_repair_feedback(fn)
    assert result["findings"] == []
    assert result["recommended_action"] == "gather_evidence"
    assert result["allowed_actions"] == ["gather_evidence"]
    assert result["review_integrity_errors"] and not result["review_detail_complete"]


def test_malformed_saved_counts_are_diagnostic_errors_not_actionable_feedback():
    fn = reviewed_function("contradicted")
    fn["scores"]["behavior_n_judged"] = -1
    result = build_repair_feedback(fn)
    assert result["validation"]["status"] == "invalid"
    assert result["findings"] == [] and result["allowed_actions"] == ["gather_evidence"]


def test_claim_compiler_rejection_can_be_revised_but_transport_error_cannot():
    fn = reviewed_function("error")
    # A passed score prefilter keeps the normal processing-error repair.
    fn["scores"]["judge_prefilter"] = {"status": "pass", "claim_hash": claim_hash(**CLAIM)}
    transport = build_repair_feedback(fn)
    assert transport["allowed_actions"] == ["retry_review"]
    assert transport["recommended_action"] == "retry_review"
    fn["scores"]["claim_compilation"] = {
        "error": "claim compilation failed validation: coverage audit rejected the draft"
    }
    feedback = build_repair_feedback(fn)
    assert "revise_claim" in feedback["allowed_actions"]
    assert feedback["recommended_action"] == "revise_claim"


def termination_gap_function():
    """Synthetic missing tool contract; no benchmark-specific evidence or judge call."""
    fn = reviewed_function("unknown")
    claim = {
        "source": "def detector(item): return item.get('last_signal') == 'FINISH'",
        "description": "The assistant terminates while a required operation remains pending.",
        "mechanism": "The assistant treats a termination signal as successful completion "
                     "and leaves the required operation unapplied.",
    }
    contract = build_claim_contract(**claim)
    fn.update(claim)
    scores = fn["scores"]
    scores.update(claim_hash=claim_hash(**claim), claim_contract=contract)
    row = scores["behavior_verdicts"][0]
    row.update(claim_contract_id=contract["contract_id"], claim_assessments=[
        {"requirement_id": requirement["id"], "status": "unknown",
         "unknown_reason": "evidence_unavailable",
         "rationale": "The trace contains FINISH, but its termination contract is not exported.",
         "citations": [], "counterevidence": []}
        for requirement in contract["requirements"]
    ])
    return fn


def test_missing_termination_contract_recommends_evidence_not_an_engine_rewrite():
    fn = termination_gap_function()
    before = deepcopy(fn)
    feedback = build_repair_feedback(fn)
    assert feedback["review_detail_complete"]
    assert feedback["recommended_action"] == "gather_evidence"
    assert feedback["diagnostic_disposition"] == "defer_claim"
    assert feedback["allowed_actions"] == ["gather_evidence"]
    for finding, investigation in zip(
        feedback["findings"], feedback["investigations"], strict=True,
    ):
        assert finding["issue_type"] == investigation["issue_type"] == "missing_evidence"
        assert "LLM rewrite cannot supply missing evidence" in investigation["question"]
        assert investigation["action"] == "defer_claim"
        assert investigation["tool"] == "get_function_feedback"
        signature(tools_library.get_function_feedback).bind(**investigation["arguments"])
    assert fn == before
