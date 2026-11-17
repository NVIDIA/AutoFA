# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The claim judge enforces frozen claim gates, independent confirmation and safe concurrency."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from autofa.core.library import FeatureFunction
from autofa.metrics import claim_compiler, claude_judge
from autofa.metrics.claim_compiler import compiler_policy_hash
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.judge_checkpoints import JudgeInterruptedError
from autofa.metrics.judge_evidence import EVIDENCE_VERSION, EvidenceReview, make_snapshot
from autofa.metrics.review_assessments import reusable_contract
from autofa.metrics.verdicts import current_behavior_scores, verdict_for_scores

from support.judge import _trace
from support.task_support import attach_task_support

pytestmark = pytest.mark.usefixtures("stub_claim_compiler")

SOURCE = "def detect(item): return True"
TEXT = "Initial claim. Later counterevidence."
PRIVATE_REASONING = "PRIVATE REVIEW REASONING: the old device failure must have persisted."
STRATEGIES = ["precision", "balanced", "recall"]
RATES = {
    "description": "The assistant writes incorrect rewards.",
    "mechanism": "The assistant invents rates because authoritative rates are unavailable.",
    "source": SOURCE,
}
SCOPE_CLAIM = {
    "description": "The assistant omits the requested total.",
    "mechanism": "The total is missing. A rater may prefer the other reply.",
    "source": SOURCE,
}
STRUCTURAL_CLAIM = "The run contains no separately recorded assistant answer message."


def _row(review, status="supported", *, quote=None, rationale=PRIVATE_REASONING):
    """A complete assessment row giving every requirement ``status``, citing read evidence."""
    reads = {field: review.read(field) for field in review.fields}
    citation = review.cite("response_text", quote or reads["response_text"]["text"])
    contract = review.snapshot["claim_contract"]
    clause_ids = [clause["id"] for clause in contract.get("defining_clauses") or []]
    return {
        "item_id": review.snapshot["item_id"], "snapshot_id": review.snapshot["snapshot_id"],
        "claim_contract_id": contract["contract_id"], "scope": "local",
        "claim_assessments": [{
            "requirement_id": requirement["id"], "status": status, "rationale": rationale,
            "citations": [{"citation_id": citation["citations"][0]["citation_id"]}],
            "counterevidence": [],
            **({"defining_requirement_ids": clause_ids}
               if status != "supported" and requirement["kind"] in {"claim", "review_check"}
               else {}),
        } for requirement in contract["requirements"]],
    }


def _reply(review, status="supported", *, specificity=0.8, **kwargs):
    return json.dumps({"verdicts": [_row(review, status, **kwargs)],
                       "behavior_specificity": specificity})


def _record_prompts(monkeypatch, **reply):
    """Answer every review stage with ``_reply`` and record each stage's prompt."""
    prompts = []
    monkeypatch.setattr(claude_judge, "_run_review", lambda *, review, prompt, **_: (
        prompts.append(prompt) or _reply(review, **reply)))
    return prompts


def _assessment(row, key):
    return next(a for a in row["claim_assessments"] if a["requirement_id"] == key)


def _review_and_row(tmp_path, contract, item, quote=None):
    review = EvidenceReview(make_snapshot(
        item, item_id="case", source=SOURCE, claim_contract=contract,
    ), tmp_path / "review")
    return review, _row(review, quote=quote)


def _rate_review(tmp_path):
    return _review_and_row(
        tmp_path, build_claim_contract(**RATES),
        {"response_text": "The documented rate is 4%. I applied 4% and wrote 400 points."},
        "The documented rate is 4%.",
    )


def _scope_review(tmp_path, profile="causal"):
    contract = build_claim_contract(
        **SCOPE_CLAIM, behavior_strictness=profile,
        defining_clauses=[{
            "id": "missing_total", "source_field": "description",
            "quote": SCOPE_CLAIM["description"], "predicate": SCOPE_CLAIM["description"],
            "required_review_scope": "local",
        }],
        compilation={
            "policy_hash": compiler_policy_hash(profile), "model": "test",
            "exclusions": [{
                "source_field": "mechanism", "quote": "A rater may prefer the other reply.",
                "reason": "Grader commentary is not a universal item condition.",
            }],
            "coverage_review": {"approved": True, "issues": []},
        },
    )
    return _review_and_row(tmp_path, contract, {
        "prompt_text": "Return the total.", "response_text": "Here are the individual values.",
    })


def _parse(review, row, **extra):
    return claude_judge.parse_session_result(
        json.dumps({"verdicts": [row], **extra}), [{"item_id": "case"}],
        reviews={"case": review}, source=SOURCE,
    )


def _verdict(review, row):
    return _parse(review, row)["verdicts"][0]


def _judge(tmp_path, count=1, *, label=True, prefix="case", **kwargs):
    return claude_judge.judge_behavior_with_claude(
        name="claim", description="An action was permanently dropped.",
        mechanism="The assistant treats a failed call as successful.", source=SOURCE,
        items=[{"_item_id": f"{prefix}-{i}", "response_text": TEXT} for i in range(count)],
        assigned_labels=[label] * count, cwd=tmp_path, model="test", max_chars=0, **kwargs,
    )


# The live parser enforces claim gates after resolving every evidence reference.


def test_direct_empty_judge_returns_unjudged_current_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_judge, "_run_review", lambda **kwargs: pytest.fail(
        "empty evidence must not start a judge review"))
    result = claude_judge.judge_behavior_with_claude(
        name="empty", description="An incorrect action occurred.", mechanism="",
        source=SOURCE, items=[], assigned_labels=[], cwd=tmp_path, model="test",
    )
    assert result["behavior_purity"] is result["behavior_specificity"] is None
    assert "judge" not in result
    for key in ("n_judged", "n_yes", "n_no", "n_uncertain", "n_completed", "n_errors"):
        assert result[key] == 0
    assert result["verdicts"] == result["sessions"] == []
    assert result["judge_contract_version"] == JUDGE_CONTRACT_VERSION
    assert result["evidence_version"] == EVIDENCE_VERSION
    assert result["judge_policy_hash"] == claude_judge.judge_policy_hash()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("requirement,status,expected", [
    ("description", "contradicted", "no"), ("failure_contribution", "unknown", "uncertain"),
])
def test_defining_checks_gate_confirmation(tmp_path, requirement, status, expected):
    review, row = _rate_review(tmp_path)
    _assessment(row, requirement).update(
        status=status, rationale="The required premise does not hold.",
    )
    row["correct"] = True
    result = _parse(review, row, behavior_purity=1, claim_calibration={"status": "good"})
    assert result["verdicts"][0]["verdict"] == expected
    assert result["n_yes"] == result["n_errors"] == 0
    assert requirement in result["verdicts"][0]["rationale"]
    assert result["behavior_purity"] == (0 if expected == "no" else None)


def test_supported_requirements_confirm_despite_legacy_boolean_or_editorial_label(tmp_path):
    review, row = _rate_review(tmp_path)
    row["correct"] = False
    result = _parse(review, row, claim_calibration={"status": "mixed"})
    assert result["n_yes"] == 1 and result["behavior_purity"] == 1


@pytest.mark.parametrize("status,field", [
    ("contradicted", "counterevidence"), ("unknown", "citations"),
])
def test_invented_citation_is_a_processing_error_not_a_semantic_verdict(tmp_path, status, field):
    review, row = _rate_review(tmp_path)
    row["claim_assessments"][0].update({"status": status, field: [{"citation_id": "invented"}]})
    result = _parse(review, row)
    assert result["n_errors"] == 1
    assert result["n_no"] == result["n_uncertain"] == result["n_yes"] == 0
    assert "unknown citation ID" in result["verdicts"][0]["validation_error"]


@pytest.mark.parametrize("status, expected", [
    ("unknown", "uncertain"), ("supported", "error"), ("contradicted", "no"),
])
def test_incomplete_exhaustive_review_can_abstain_without_weakening_scope(
    tmp_path, status, expected,
):
    review, row = _rate_review(tmp_path)
    # Keep the cited prefix read, but leave the remainder of the stored field unread.
    review.reads = [r for r in review.reads if r["field"] == "detector_source"]
    review.read("response_text", 0, len("The documented rate is 4%."))
    row["scope"] = "whole_trace"
    for assessment in row["claim_assessments"]:
        assessment["status"] = "unknown" if status == "contradicted" else status
        assessment["rationale"] = "An exhaustive review cannot be finished within the budget."
        if status == "contradicted" and assessment["requirement_id"] == "mechanism":
            assessment.update(
                status="contradicted",
                rationale="The cited documented rate contradicts the claim that authoritative "
                          "rates were unavailable, regardless of the unread later actions.",
            )
    result = _parse(review, row)
    assert result["verdicts"][0]["verdict"] == expected
    assert result["verdicts"][0]["scope"] == "whole_trace"


def test_component_contradiction_is_in_the_derived_explanation(tmp_path):
    review, row = _rate_review(tmp_path)
    assessment = _assessment(row, "mechanism")
    assessment["components"] = [{
        "claim": "authoritative rates are unavailable", "status": "contradicted",
        "rationale": "The rates were documented.", "citations": assessment["citations"],
    }]
    result = _parse(review, row)
    assert result["n_no"] == 1
    assert "authoritative rates are unavailable" in result["verdicts"][0]["rationale"]
    component = result["verdicts"][0]["claim_assessments"][1]["components"][0]
    assert component["citations"][0]["quote"] == "The documented rate is 4%."


@pytest.mark.parametrize("raw", [
    "broken JSON", '{"verdicts": []}', '{"verdicts": 1}',
    '{"verdicts": [{"item_id":"case","correct":true}]}',
], ids=["unparseable", "item_missing", "verdicts_not_a_list", "legacy_boolean"])
def test_production_contract_never_falls_back_to_a_legacy_verdict(tmp_path, raw):
    review, _ = _rate_review(tmp_path)
    result = claude_judge.parse_session_result(raw, [{"item_id": "case"}],
                                               reviews={"case": review})
    assert result["n_errors"] == 1 and result["n_yes"] == 0
    assert result["verdicts"][0]["validation_error"]


@pytest.mark.parametrize("contract", [{}, None])
def test_present_but_invalid_contract_never_enters_legacy_parser(tmp_path, contract):
    review, row = _rate_review(tmp_path)
    review.snapshot["claim_contract"] = contract
    row["correct"] = True
    result = _parse(review, row)
    assert result["n_errors"] == 1 and result["n_yes"] == 0


def test_duplicate_assessments_for_one_item_are_a_processing_error(tmp_path):
    review, row = _rate_review(tmp_path)
    result = claude_judge.parse_session_result(
        json.dumps({"verdicts": [row, deepcopy(row)]}), [{"item_id": "case"}],
        reviews={"case": review},
    )
    assert result["n_errors"] == 1 and result["n_yes"] == 0
    assert "exactly one assessment" in result["verdicts"][0]["validation_error"]


def test_contract_and_roles_remain_available_after_compaction(tmp_path):
    contract = build_claim_contract(
        description="Agent drops an action.", mechanism="", source=SOURCE,
    )
    item = {"response_text": json.dumps({"reward": 0, "steps": [
        {"source": "user", "tool_calls": [{"name": "malformed工具", "arguments": {}}]},
        {"source": "user", "message": "It works after retrying."},
        {"source": "assistant", "message": "Glad to hear it."},
    ]})}
    review = EvidenceReview(make_snapshot(
        item, item_id="case", source=SOURCE, claim_contract=contract,
    ), tmp_path / "review")
    text = review.read("trajectory_text")["text"]
    assert "source=user" in text and "It works after retrying." in text
    assert "source=assistant" in text and "malformed工具" in text
    assert '"reward"' not in text
    review.compacted({})
    assert review.status()["claim_contract"] == contract
    assert review.manifest()["claim_contract"] == contract


def test_claim_changes_change_evidence_identity_without_changing_trace():
    item = {"response_text": "The assistant retried successfully."}
    contracts = [build_claim_contract(description=d, mechanism="", source=SOURCE)
                 for d in ("A transient error occurred.", "The action was permanently dropped.")]
    snapshots = [make_snapshot(item, item_id="case", source=SOURCE, claim_contract=c)
                 for c in contracts]
    assert snapshots[0]["fields"] == snapshots[1]["fields"]
    assert snapshots[0]["snapshot_id"] != snapshots[1]["snapshot_id"]


# Compiled exclusions cannot return as independent blockers in generic checks.


def test_excluded_rater_evidence_requires_scope_correction_not_completed_uncertainty(tmp_path):
    review, row = _scope_review(tmp_path)
    _assessment(row, "causal_link").update(
        status="unknown", unknown_reason="evidence_unavailable",
        rationale="The competing answer and the rater's reasoning are unavailable.",
    )
    result = _verdict(review, row)
    assert result["verdict"] == "error"
    assert result["completion_status"] == "processing_error"
    assert "compilation.exclusions" in result["validation_error"]

    # Once the out-of-scope demand is corrected, the otherwise supported claim
    # passes. The parser never silently turns an invalid assessment into support.
    _assessment(row, "causal_link").update(
        status="supported", unknown_reason=None,
        rationale="The task defect is established; excluded grader commentary is not required.",
    )
    assert _verdict(review, row)["verdict"] == "yes"


@pytest.mark.parametrize("references, error", [
    ([], "defining_requirement_ids"),
    (["exclusion_1"], "defining_requirement_ids"),
    (["missing_total"], "same status"),
], ids=["no_reference", "exclusion_reference", "supported_clause"])
def test_blocker_must_map_to_a_frozen_clause_with_the_same_status(tmp_path, references, error):
    review, row = _scope_review(tmp_path)
    _assessment(row, "causal_link").update(
        status="unknown", defining_requirement_ids=references,
    )
    result = _verdict(review, row)
    assert result["verdict"] == "error"
    assert error in result["validation_error"]


def test_actual_defining_condition_still_blocks_without_marking_other_clauses(tmp_path):
    review, row = _scope_review(tmp_path)
    _assessment(row, "missing_total").update(status="unknown")
    _assessment(row, "outcome_and_recovery").update(
        status="unknown", defining_requirement_ids=["missing_total"],
    )
    assert _verdict(review, row)["verdict"] == "uncertain"


@pytest.mark.parametrize("profile,acceptance,status,verdict", [
    ("causal", "failure_contribution", "unknown", "uncertain"),
    ("contextual", "failure_relevance", "contradicted", "no"),
])
def test_policy_only_missing_contribution_remains_mandatory(
    tmp_path, profile, acceptance, status, verdict,
):
    review, row = _scope_review(tmp_path, profile)
    _assessment(row, acceptance).update(
        status=status, rationale="The omission is evidenced but its effect on the task is not.",
    )
    assert _verdict(review, row)["verdict"] == verdict


def test_changed_scope_prompt_reviews_again_without_recompiling_claim(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    calls = _record_prompts(monkeypatch)
    with monkeypatch.context() as older:
        older.setattr(claude_judge, "judge_system_prompt", lambda *a: "Prior review policy")
        older.setattr(claude_judge, "verification_system_prompt", lambda *a: "Prior verifier")
        first = _judge(tmp_path)
        repeated = _judge(tmp_path, compiled_contract=first["claim_contract"])
        assert first["n_yes"] == repeated["n_yes"] == 1
        assert len(calls) == 2 and repeated["reused_stages"] == 2
    current = _judge(tmp_path, compiled_contract=first["claim_contract"])
    assert current["n_yes"] == 1 and current["n_errors"] == 0
    assert len(calls) == 4 and current["reused_stages"] == 0
    assert current["claim_contract"] == first["claim_contract"]
    assert len(stub_claim_compiler) == 1


# A provisional confirmation needs a fresh, independently audited second review.


def test_initial_and_confirmation_share_configured_effort(tmp_path, monkeypatch):
    calls = []

    def run(*, review, model, effort, system_prompt=None, **kwargs):
        assert kwargs["thinking"] == "enabled" and kwargs["thinking_budget_tokens"] == 2048
        calls.append((model, effort, system_prompt))
        return _reply(review)

    monkeypatch.setattr(claude_judge, "_run_review", run)
    result = _judge(tmp_path, effort="medium", thinking="enabled", thinking_budget_tokens=2048)
    assert result["n_yes"] == 1
    assert calls == [("test", "medium", None),
                     ("test", "medium", claude_judge.VERIFICATION_SYSTEM)]


@pytest.mark.parametrize("first,expected", [("contradicted", "no"), ("malformed", "error")])
def test_nonconfirmations_do_not_trigger_a_second_session(tmp_path, monkeypatch, first, expected):
    reviews = []

    def run(*, review, **kwargs):
        reviews.append(review)
        return "invalid JSON" if first == "malformed" else _reply(review, first)

    monkeypatch.setattr(claude_judge, "_run_review", run)
    result = _judge(tmp_path)
    assert len(reviews) == 1
    assert result["verdicts"][0]["verdict"] == expected
    assert not (reviews[0].directory / "confirmation").exists()


@pytest.mark.parametrize("second,expected", [
    ("supported", "yes"), ("contradicted", "no"), ("sdk_error", "error"),
])
def test_verifier_controls_final_verdict_and_keeps_provisional_evidence(
    tmp_path, monkeypatch, second, expected,
):
    reviews = []

    def run(*, review, system_prompt=None, **kwargs):
        assert not review.reads and not review.citations and not review.notes
        prompt = kwargs["prompt"]
        assert review.snapshot["snapshot_id"] in prompt
        assert review.snapshot["claim_contract"]["contract_id"] in prompt
        reviews.append(review)
        if len(reviews) == 1:
            review.save_notes("Private initial notes must not become verifier evidence.")
            return _reply(review, quote="Initial claim.")
        assert system_prompt == claude_judge.VERIFICATION_SYSTEM
        assert PRIVATE_REASONING not in prompt
        assert "Private initial notes" not in prompt
        assert '"claim_assessments"' not in prompt and '"citation_id"' not in prompt
        assert review is not reviews[0]
        assert review.snapshot == reviews[0].snapshot
        assert review.directory == reviews[0].directory / "confirmation"
        assert not (reviews[0].directory / "verdict.json").exists()
        if second == "sdk_error":
            raise RuntimeError("confirmation context exhausted")
        return _reply(review, second, quote="Later counterevidence.")

    monkeypatch.setattr(claude_judge, "_run_review", run)
    result = _judge(tmp_path)
    assert len(reviews) == 2
    final = result["verdicts"][0]
    assert final["verdict"] == expected
    assert final["initial_review"]["verdict"] == "yes"
    assert final["initial_review"]["claim_assessments"][0]["rationale"] == PRIVATE_REASONING
    assert final["initial_review"]["citations"][0]["quote"] == "Initial claim."
    assert final["primary_review_path"] == str(reviews[0].directory)
    assert final["verification_review_path"] == final["review_path"] == str(reviews[1].directory)
    assert final["verification_status"] == {
        "yes": "confirmed", "no": "rejected", "error": "error",
    }[expected]
    assert (result["n_yes"], result["n_no"], result["n_errors"]) == (
        int(expected == "yes"), int(expected == "no"), int(expected == "error"))
    assert [session["stage"] for session in result["sessions"]] == ["initial", "verification"]
    assert (reviews[0].directory / "initial-verdict.json").is_file()
    assert json.loads((reviews[0].directory / "verdict.json").read_text()) == final
    if expected != "error":
        assert final["citations"][0]["quote"] == "Later counterevidence."
    else:
        assert result["behavior_purity"] is None


def test_primary_citation_ids_cannot_satisfy_independent_verification(tmp_path, monkeypatch):
    primary_reply = None
    reviews = []

    def run(*, review, **kwargs):
        nonlocal primary_reply
        reviews.append(review)
        if len(reviews) == 1:
            primary_reply = _reply(review)
            return primary_reply
        assert not review.citations
        # Even reading the same content does not issue this review a citation ID.
        review.read("detector_source")
        review.read("response_text")
        return primary_reply

    monkeypatch.setattr(claude_judge, "_run_review", run)
    result = _judge(tmp_path)
    assert result["n_yes"] == 0 and result["n_errors"] == 1
    final = result["verdicts"][0]
    assert "unknown citation ID" in final["validation_error"]
    assert final["initial_review"]["verdict"] == "yes"
    assert Path(final["review_path"]) == reviews[1].directory


@pytest.mark.parametrize("initial_verdict,changed,error", [
    ("yes", "snapshot_id", "snapshot_id"),
    ("yes", "claim_contract_id", "claim_contract_id"),
    ("no", None, "only a provisional confirmation"),
])
def test_confirmation_requires_a_provisional_yes_for_the_identical_item_and_claim(
    initial_verdict, changed, error,
):
    initial = {"item_id": "item", "snapshot_id": "snapshot", "claim_contract_id": "contract",
               "verdict": initial_verdict}
    verification = {**initial, "verdict": "yes"}
    if changed:
        verification[changed] = "different"
    with pytest.raises(ValueError, match=error):
        claude_judge.combine_confirmation(initial, verification)


def test_assigned_label_stays_in_archival_bookkeeping_but_never_reaches_either_judge(
    tmp_path, monkeypatch,
):
    prompts = _record_prompts(monkeypatch)
    private_label = "PRIVATE-DETECTOR-LABEL"
    result = _judge(tmp_path, label=private_label)
    assert len(prompts) == 2
    assert not any(private_label in p or "assigned_label" in p for p in prompts)
    assert result["verdicts"][0]["assigned_label"] == private_label
    assert result["verdicts"][0]["initial_review"]["assigned_label"] == private_label


# Search preferences and policy provenance.


@pytest.mark.parametrize("module, policy", [
    ("claim_compiler", "COMPILER_SYSTEM"), ("claude_judge", "VERIFICATION_SYSTEM"),
])
def test_compiler_and_verifier_policy_changes_invalidate_judging_provenance(
    monkeypatch, module, policy,
):
    before = claude_judge.judge_policy_hash()
    monkeypatch.setattr(f"autofa.metrics.{module}.{policy}", "A changed policy.")
    assert claude_judge.judge_policy_hash() != before


def test_changing_search_preference_reuses_both_saved_review_stages(tmp_path, monkeypatch):
    calls = _record_prompts(monkeypatch, specificity=0.2)
    first = _judge(tmp_path, objective_strategy="precision")
    assert first["n_yes"] == 1  # A low detail diagnostic does not reject a supported claim.
    repeated = _judge(tmp_path, objective_strategy="precision")
    assert len(calls) == 2 and repeated["reused_stages"] == 2
    changed = _judge(tmp_path, objective_strategy="recall")
    assert len(calls) == 2 and changed["reused_stages"] == 2
    assert first["judge_policy_hash"] == changed["judge_policy_hash"]
    assert first["judge_rubric_hash"] == changed["judge_rubric_hash"]


def test_new_judge_prompt_is_identical_across_search_presets():
    kwargs = dict(name="claim", description="Claim", mechanism="Mechanism", source=SOURCE,
                  evidence=[])
    assert len({claude_judge.build_judge_prompt(**kwargs, objective_strategy=strategy)
                for strategy in STRATEGIES}) == 1


# Reviews made under an earlier judge policy remain history, never current support.
OBSOLETE_JUDGE_POLICIES = {
    "behavioral": "95d4d13c94fdb469cbfc27b553468806b349043408117bc2d72b8dca7d59ced0",
    "causal": "0078c867dba05270b9c45921e877b19d689a7b93ccb9c0712d0740538c4e007d",
}


def _structural_contract(profile, policy_hash=None):
    return build_claim_contract(
        source=SOURCE, description=STRUCTURAL_CLAIM, mechanism=STRUCTURAL_CLAIM,
        behavior_strictness=profile,
        defining_clauses=[{"id": "clause_1", "source_field": "description",
                           "quote": STRUCTURAL_CLAIM, "predicate": STRUCTURAL_CLAIM,
                           "required_review_scope": "whole_trace"}],
        compilation={"policy_hash": policy_hash or compiler_policy_hash(profile),
                     "model": "test", "exclusions": [],
                     "coverage_review": {"approved": True, "issues": []}},
    )


@pytest.mark.parametrize("profile,strategy", [
    ("behavioral", "balanced"), ("causal", "precision"),
])
def test_completed_prior_policy_preserves_history_but_requires_fresh_review(profile, strategy):
    claim = {"source": SOURCE, "description": "A required message is missing",
             "mechanism": "The assistant omits the required response"}
    fn = FeatureFunction(name="missing", **claim)
    fn.scores = {"evaluated": True, "n_items": 100, "n_flagged": 10, "recall": .1,
                 "ppv_at_pi0": .9, "behavior_purity": 1., "claim_hash": claim_hash(**claim),
                 "judge_behavior_strictness": profile, "judge_objective_strategy": strategy}
    attach_task_support(fn, task_n=5)
    fn.scores["judge_policy_hash"] = OBSOLETE_JUDGE_POLICIES[profile]
    original = deepcopy(fn.scores)
    current = current_behavior_scores(fn.scores, **claim, behavior_strictness=profile)
    assert verdict_for_scores(current)["code"] == "UNSCORED"
    assert current["behavior_judgment_stale"]
    assert current["judge_policy_hash"] == original["judge_policy_hash"]
    assert fn.scores == original
    # Old completed history may not reuse these prompts.
    assert not claude_judge.compatible_judge_policy(
        original["judge_policy_hash"], behavior_strictness=profile,
    )


@pytest.mark.parametrize("profile", ["behavioral", "causal"])
def test_contract_from_another_compiler_policy_is_recompiled_not_reused(profile):
    current = _structural_contract(profile)
    stale = _structural_contract(profile, policy_hash="0" * 64)
    claim = dict(source=SOURCE, description=STRUCTURAL_CLAIM, mechanism=STRUCTURAL_CLAIM,
                 behavior_strictness=profile)
    assert claim_compiler.validate_compiled_contract(current, **claim) == current
    with pytest.raises(ValueError, match="stale compiler policy provenance"):
        claim_compiler.validate_compiled_contract(stale, **claim)
    # A saved stale contract takes the fresh compilation path instead of failing validation.
    assert reusable_contract({"claim_contract": current}) == current
    assert reusable_contract({"claim_contract": stale}) is None


@pytest.mark.parametrize("profile,strategy,compiled,structured", [
    ("behavioral", "balanced", False, False), ("causal", "recall", True, True),
])
def test_obsolete_operation_cannot_resume_with_its_old_judge_policy(
    tmp_path, monkeypatch, profile, strategy, compiled, structured,
):
    def unexpected(**kwargs):
        pytest.fail("obsolete policy must be rejected before compilation or paid review")

    monkeypatch.setattr(claude_judge, "compile_claim_contract", unexpected)
    monkeypatch.setattr(claude_judge, "_run_review", unexpected)
    response = json.dumps(_trace()) if structured else "Reasoning without an ending"
    kwargs = {
        "name": "missing", "description": STRUCTURAL_CLAIM, "mechanism": STRUCTURAL_CLAIM,
        "source": SOURCE, "items": [{"_item_id": "item", "response_text": response}],
        "assigned_labels": [True], "cwd": tmp_path, "model": "test",
        "behavior_strictness": profile, "objective_strategy": strategy,
        "_resume_policy_hash": OBSOLETE_JUDGE_POLICIES[profile],
        **({"compiled_contract": _structural_contract(profile)} if compiled else {}),
    }
    with pytest.raises(ValueError, match="(?i)policy"):
        claude_judge.judge_behavior_with_claude(**kwargs)


# Concurrent judging preserves evidence isolation, results and recovery.


def test_bounded_overlap_and_checkpoint_reuse_across_worker_counts(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    workers = 2
    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    active = peak = 0

    def review(*, review, **kwargs):
        nonlocal active, peak
        assert len(stub_claim_compiler) == 1
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=5)
            return _reply(review, "contradicted")
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(claude_judge, "_run_review", review)
    count = workers * 2
    result = _judge(tmp_path, count, workers=workers)
    assert peak == workers and result["n_no"] == count
    progress_path = next((tmp_path / "behavior_judge").glob("*/progress.json"))
    progress = json.loads(progress_path.read_text())
    assert progress["completed"] == count and progress["errors"] == 0
    assert progress["state"] == "complete" and progress["active_items"] == []

    monkeypatch.setattr(claude_judge, "_run_review", lambda **kwargs: pytest.fail(
        "Changing concurrency must reuse completed reviews"))
    resumed = _judge(tmp_path, count, workers=1)
    assert resumed["reused_stages"] == count and resumed["n_no"] == count
    assert len(stub_claim_compiler) == 1


def test_out_of_order_completion_keeps_input_order(tmp_path, monkeypatch):
    second_saved = threading.Event()
    completed = []
    original_write = claude_judge.atomic_write_json

    def write(path, value):
        original_write(path, value)
        if path.name == "verdict.json" and path.parent.name.isdigit():
            completed.append(int(path.parent.name))
            if path.parent.name == "1":
                second_saved.set()

    def review(*, review, **kwargs):
        if review.snapshot["item_id"] == "case-0":
            assert second_saved.wait(timeout=5)
        return _reply(review, "contradicted")

    monkeypatch.setattr(claude_judge, "atomic_write_json", write)
    monkeypatch.setattr(claude_judge, "_run_review", review)
    result = _judge(tmp_path, 3, workers=2)
    ids = [f"case-{i}" for i in range(3)]
    assert completed[0] == 1
    assert [v["item_id"] for v in result["verdicts"]] == ids
    assert [v["item_id"] for v in result["sessions"]] == ids
    pack = json.loads((tmp_path / "behavior_judge/claim.evidence.json").read_text())
    assert [v["item_id"] for v in pack["evidence"]] == ids


def test_parallel_confirmations_have_fresh_isolated_context(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    primary = {}

    def review(*, review, prompt, system_prompt=None, **kwargs):
        item_id = review.snapshot["item_id"]
        assert PRIVATE_REASONING not in prompt
        assert not review.reads and not review.citations
        if system_prompt is None:
            primary[item_id] = review
        else:
            assert primary[item_id] is not review
            assert primary[item_id].directory != review.directory
            assert primary[item_id].snapshot == review.snapshot
        barrier.wait(timeout=5)
        status = "contradicted" if system_prompt and item_id == "case-1" else "supported"
        return _reply(review, status)

    monkeypatch.setattr(claude_judge, "_run_review", review)
    result = _judge(tmp_path, 2, workers=2)
    assert [v["verdict"] for v in result["verdicts"]] == ["yes", "no"]
    assert [(v["item_id"], v["stage"]) for v in result["sessions"]] == [
        (f"case-{i}", stage) for i in range(2) for stage in ["initial", "verification"]
    ]
    assert len({v["review_path"] for v in result["sessions"]}) == 4


@pytest.mark.parametrize("peer_status", ["contradicted", "supported"])
def test_disconnect_stops_new_items_and_keeps_peer_checkpoint(
    tmp_path, monkeypatch, peer_status,
):
    barrier = threading.Barrier(2)
    interrupted = threading.Event()
    calls = []
    original_progress = claude_judge.write_progress

    def progress(directory, data, **updates):
        original_progress(directory, data, **updates)
        if data["state"] == "interrupted":
            interrupted.set()

    def review(*, review, **kwargs):
        item_id = review.snapshot["item_id"]
        calls.append(item_id)
        barrier.wait(timeout=5)
        if item_id == "case-0":
            raise ConnectionResetError("connection reset")
        assert interrupted.wait(timeout=5)
        return _reply(review, peer_status)

    monkeypatch.setattr(claude_judge, "write_progress", progress)
    monkeypatch.setattr(claude_judge, "_run_review", review)
    with pytest.raises(JudgeInterruptedError):
        _judge(tmp_path, 4, workers=2)
    assert sorted(calls) == ["case-0", "case-1"]
    progress_path = next((tmp_path / "behavior_judge").glob("*/progress.json"))
    saved = json.loads(progress_path.read_text())
    assert saved["state"] == "interrupted"
    assert saved["completed"] == int(peer_status == "contradicted")
    assert saved["active_items"] == []
    assert (progress_path.parent / "1/review-verdict.json").exists()
    assert not (progress_path.parent / "1/confirmation").exists()
    assert not (tmp_path / "behavior_judge/claim.verdict.json").exists()

    calls.clear()

    def recover(*, review, **kwargs):
        calls.append(review.snapshot["item_id"])
        return _reply(review, "contradicted")

    monkeypatch.setattr(claude_judge, "_run_review", recover)
    result = _judge(tmp_path, 4, workers=3)
    assert result["n_no"] == 4 and result["reused_stages"] == 1
    expected = ["case-0", "case-2", "case-3"]
    if peer_status == "supported":
        expected.append("case-1")  # Only its independent confirmation remains.
    assert sorted(calls) == sorted(expected)


@pytest.mark.parametrize("during_drain", [False, True])
def test_ctrl_c_cancels_active_sdk_requests_and_waits_for_cleanup(
    tmp_path, monkeypatch, during_drain,
):
    started = [threading.Event(), threading.Event()]
    cleaned = [threading.Event(), threading.Event()]
    original_wait = claude_judge.concurrent.futures.wait
    original_shutdown = ThreadPoolExecutor.shutdown
    injected = False

    async def session(*, review, **kwargs):
        index = int(review.snapshot["item_id"].split("-")[-1])
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned[index].set()

    # For the drain case, fail one worker without spending transport retries;
    # leave its peer in the real _run_review cancellation monitor.
    original_review = claude_judge._run_review

    def review(**kwargs):
        if during_drain and kwargs["review"].snapshot["item_id"] == "case-0":
            started[0].set()
            assert started[1].wait(timeout=5)
            cleaned[0].set()
            raise ConnectionResetError("connection reset")
        return original_review(**kwargs)

    def wait(*args, **kwargs):
        nonlocal injected
        if not during_drain and not injected:
            assert all(event.wait(timeout=5) for event in started)
            injected = True
            raise KeyboardInterrupt
        return original_wait(*args, **kwargs)

    def shutdown(pool, *args, **kwargs):
        nonlocal injected
        if during_drain and not injected:
            injected = True
            raise KeyboardInterrupt
        return original_shutdown(pool, *args, **kwargs)

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", session)
    monkeypatch.setattr(claude_judge, "_run_review", review)
    monkeypatch.setattr(claude_judge.concurrent.futures, "wait", wait)
    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", shutdown)
    with pytest.raises(KeyboardInterrupt):
        _judge(tmp_path, 2, workers=2)
    assert injected and all(event.is_set() for event in cleaned)
    progress = json.loads(next((tmp_path / "behavior_judge").glob("*/progress.json")).read_text())
    assert progress["state"] == "interrupted" and progress["active_items"] == []
    assert not (tmp_path / "behavior_judge/claim.verdict.json").exists()


def test_invalid_concurrency_rejected_before_starting(tmp_path):
    with pytest.raises(ValueError, match="integer between 1 and 8"):
        _judge(tmp_path, 4, workers=9)
    assert not (tmp_path / "behavior_judge").exists()
