# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge feedback, diagnoses, labels and citation references reach the next iteration intact."""

import json
from copy import deepcopy
from inspect import signature
from types import SimpleNamespace

import pytest

from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import NullLLM
from autofa.data.parquet_store import Dataset
from autofa.mcp import tool_results, tools_data, tools_eval, tools_library
from autofa.metrics import claude_judge
from autofa.metrics.claim_calibration import (
    attach_calibration,
    parse_claim_calibration,
    summarize_judge_feedback,
)
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.judge_evidence import make_snapshot
from autofa.metrics.repair_feedback import build_repair_feedback
from autofa.metrics.review_diagnosis import build_review_diagnosis
from autofa.metrics.verdicts import verdict_for_scores

from support.repair_review import CALIBRATION, DESCRIPTION, MECHANISM, SOURCE, _result, _runtime
from support.task_support import _function as scored_function
from support.task_support import attach_task_support


def _assert_compact_feedback(result):
    """Budget target and Good-qualification directions apart from review findings."""
    discovery = dict(result["discovery_feedback"])
    discovery["next_steps"] = list(discovery["next_steps"])
    for name in ("grade_gap", "qualification_gap"):
        gap = discovery.pop(name)
        summary = discovery.pop(name + "_summary")
        assert len(json.dumps(gap)) < 6_000
        assert len(summary) < 2_000
        if discovery["next_steps"]:
            discovery["next_steps"][0] = discovery["next_steps"][0].replace(" " + summary, "")
    original_payload = {**result, "discovery_feedback": discovery}
    assert len(json.dumps(original_payload)) < 25_000, {
        key: len(json.dumps(value)) for key, value in original_payload.items()
    }
    assert len(json.dumps(result)) < 40_000


def _reply(review, status, *, quote="I guess the value.",
           rationale="The earlier derivation is unavailable.", diagnosis=None,
           calibration=CALIBRATION):
    review.read("detector_source")
    review.read("response_text")
    citation = review.cite("response_text", quote)["citations"][0]
    contract = review.snapshot["claim_contract"]
    defining_ids = [clause["id"] for clause in contract["defining_clauses"]]
    return json.dumps({
        "verdicts": [{
            "item_id": review.snapshot["item_id"],
            "snapshot_id": review.snapshot["snapshot_id"],
            "claim_contract_id": contract["contract_id"], "scope": "local",
            "claim_assessments": [{
                "requirement_id": requirement["id"], "status": status, "rationale": rationale,
                "citations": [{"citation_id": citation["citation_id"]}],
                **({"defining_requirement_ids": defining_ids}
                   if status != "supported" and requirement["kind"] in {"claim", "review_check"}
                   else {}),
                **({"unknown_reason": "evidence_unavailable", "evidence_inspection": {
                    "status": "unavailable",
                    "ranges": [{"field": "response_text", "start": 0,
                                "end": len(review.fields["response_text"])}],
                    "reason": "The response is inspected; derivation inputs were not retained.",
                }} if status == "unknown" else {}),
                **({"diagnosis": diagnosis} if diagnosis else {}),
            } for requirement in contract["requirements"]],
        }],
        "behavior_specificity": 0.8,
        **({"claim_calibration": calibration} if calibration else {}),
    })


def resubmit(fn, **changes):
    """Submit ``fn`` through the discovery tool, optionally with changed arguments."""
    return tools_library.submit_function(**{
        "name": fn.name, "source": fn.source, "description": fn.description,
        "mechanism": fn.mechanism, **changes,
    })


@pytest.mark.usefixtures("stub_claim_compiler")
@pytest.mark.parametrize("outcome", ["unknown", "error"])
def test_final_uncertain_review_retains_calibration_but_errors_do_not(
    tmp_path, monkeypatch, outcome,
):
    # The confirmation stage decides; single-stage replies are covered with the evidence tests.
    calls = []

    def review(*, review, **kwargs):
        calls.append(review)
        if len(calls) == 1:
            return _reply(review, "supported")
        return "invalid JSON" if outcome == "error" else _reply(review, outcome)

    monkeypatch.setattr(claude_judge, "_run_review", review)
    result = claude_judge.judge_behavior_with_claude(
        name="guess", description=DESCRIPTION, mechanism=MECHANISM, source=SOURCE,
        items=[{"_item_id": "case", "response_text": "I guess the value."}],
        assigned_labels=[True], cwd=tmp_path, model="test", max_chars=0,
    )
    assert len(calls) == 2
    assert result["behavior_purity"] is result["behavior_specificity"] is None
    if outcome == "unknown":
        assert result["n_uncertain"] == 1 and result["n_errors"] == 0
        assert result["claim_calibration"] == CALIBRATION
        assert result["item_calibrations"] == [
            {"item_id": "case", "claim_calibration": CALIBRATION},
        ]
    else:
        assert result["n_errors"] == 1 and result["n_uncertain"] == 0
        assert result["claim_calibration"] is result["calibration_hint"] is None
        assert result["item_calibrations"] == []


def test_fallback_never_turns_unknown_or_processing_failure_into_a_mismatch():
    result = attach_calibration({"behavior_purity": None, "verdicts": [
        {"verdict": "uncertain", "rationale": "Could not establish the causal premise."},
        {"verdict": "error", "rationale": "keyword regex superficial failure"},
    ]})
    assert result["claim_calibration"]["status"] == "mixed"
    failed = attach_calibration({"behavior_purity": None, "verdicts": [
        {"verdict": "error", "rationale": "No review completed."},
    ]}, {"claim_calibration": CALIBRATION})
    assert failed["claim_calibration"] is failed["calibration_hint"] is None


@pytest.mark.usefixtures("stub_claim_compiler")
def test_structured_diagnosis_survives_real_parser_submission_and_saved_feedback(
    tmp_path, monkeypatch,
):
    rt, fn = _runtime(tmp_path, monkeypatch, automatic_eligible=True)
    diagnosis = {
        "observed_facts": "The assistant guesses a value.",
        "unsupported_premise": "The derivation was available to the assistant.",
        "competing_explanation": "The trace may omit derivation inputs.",
        "evidence_needed": "Inputs available before the choice.",
        "evidence_availability": "unavailable",
        "availability_reason": "That input was not retained.",
    }
    monkeypatch.setattr(claude_judge, "_run_review", lambda *, review, **kwargs: _reply(
        review, "unknown", diagnosis=diagnosis))
    result = resubmit(fn)
    assert result["ok"] and result["judge_refresh"]["refreshed"]
    finding = result["judge_feedback"]["findings"][0]
    assert finding["requirements"][0]["requirement_id"] == "failure_contribution"
    assert {key: finding["diagnosis"][key] for key in diagnosis} == diagnosis
    assert finding["diagnosis"]["next_action"]["kind"] == "defer_claim"
    saved = FunctionLibrary.load(rt.working_library_path()).get(fn.name)
    assert saved.scores["behavior_n_uncertain"] > 0
    for row in saved.scores["behavior_verdicts"]:
        assert all(a["diagnosis"] == diagnosis for a in row["claim_assessments"])
    details = tools_library.get_function_feedback(fn.name)
    for row in details["feedback"]["behavior_validation"]["verdicts"]:
        assert all(a["diagnosis"] == diagnosis for a in row["claim_assessments"])


def _bind_review_snapshots(result, args):
    for index, (row, item) in enumerate(zip(result["verdicts"], args["items"], strict=True)):
        row["item_id"] = item["_item_id"]
        row["snapshot_id"] = make_snapshot(
            item, item_id=item["_item_id"], source=args["source"],
            failure_localisation=args["failure_localisations"][index],
            claim_contract=result["claim_contract"],
        )["snapshot_id"]
    return result


def test_item_feedback_survives_judge_evaluate_disk_and_explicit_retrieval(tmp_path, monkeypatch):
    rt, fn = _runtime(tmp_path, monkeypatch)
    expected = _result(fn, ["no", "uncertain", "error", "uncertain"])
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude",
                        lambda **kwargs: _bind_review_snapshots(expected, kwargs))
    # This persistence regression deliberately covers four different verdict rows;
    # request that cohort explicitly instead of depending on the default minimum.
    result = tools_eval.judge_function(fn.name, evidence_n=4)
    assert result["item_calibrations"] == expected["item_calibrations"]
    assert result["judge_feedback"]["details_tool"] == {
        "name": "get_function_feedback", "arguments": {"name": fn.name},
    }
    assert {f["verdict"] for f in result["judge_feedback"]["findings"]} == {
        "no", "uncertain", "error",
    }
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert fn.scores["item_calibrations"] == expected["item_calibrations"]
    saved = FunctionLibrary.load(rt.working_library_path()).get(fn.name)
    assert saved.scores["item_calibrations"] == expected["item_calibrations"]
    feedback = tools_library.get_function_feedback(fn.name)
    assert feedback["item_calibrations"] == expected["item_calibrations"]
    assert feedback["feedback"]["behavior_validation"]["verdicts"] == expected["verdicts"]


@pytest.mark.parametrize("specificity", [None, float("nan")])
def test_unavailable_specificity_is_not_invented_for_confirmed_claim(
    tmp_path, monkeypatch, specificity,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    reviewed = _result(fn, ["yes"] * 4)
    reviewed["behavior_specificity"] = specificity
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", lambda **kwargs: reviewed)
    result = tools_eval.judge_function(fn.name)
    assert result["ok"] and result["n_yes"] == 4
    assert result["behavior_purity"] == 1.0
    assert result["behavior_specificity"] is None
    assert result["feedback"]["behavior_specificity"] is None
    saved = FunctionLibrary.load(rt.working_library_path()).get(fn.name)
    assert saved.scores["behavior_specificity"] is None
    assert saved.scores["behavior_purity"] == 1.0


@pytest.fixture
def paginated_feedback(tmp_path, monkeypatch):
    rt, fn = _runtime(tmp_path, monkeypatch)
    expected = _result(fn, ["yes", "no", "uncertain", "error"])
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", lambda **kwargs: expected)
    assert tools_eval.judge_function(fn.name)["ok"]
    return rt, fn, expected


def test_feedback_pages_keep_matching_calibrations_and_full_review_summary(paginated_feedback):
    _, fn, expected = paginated_feedback
    saved = json.dumps(fn.scores, sort_keys=True)
    pages = [tools_library.get_function_feedback(fn.name, offset=offset, limit=2)
             for offset in (0, 2, 4)]
    for offset, page in zip((0, 2, 4), pages, strict=True):
        assert page["ok"], page
        validation = page["feedback"]["behavior_validation"]
        rows = expected["verdicts"][offset:offset + 2]
        assert validation["verdicts"] == rows
        ids = {row["item_id"] for row in rows}
        assert page["item_calibrations"] == [
            row for row in expected["item_calibrations"] if row["item_id"] in ids
        ]
        assert validation["sampled"] == 4
        assert all(validation[key] == 1 for key in (
            "confirmed", "rejected", "uncertain", "processing_errors",
        ))
        assert validation["claim_contract"] == expected["claim_contract"]
        assert page["claim_calibration"] == expected["claim_calibration"]
        assert page["feedback"]["behavior_purity"] == .25
        assert page["judge_feedback"] == pages[0]["judge_feedback"]
        assert page["pagination"] == {
            "offset": offset, "limit": 2, "total": 4,
            "next_offset": 2 if offset == 0 else None,
        }
    assert json.dumps(fn.scores, sort_keys=True) == saved


@pytest.mark.parametrize("item_id", ["case-2", "missing"])
def test_feedback_item_filter_keeps_global_counts_and_filters_calibration(
    paginated_feedback, item_id,
):
    _, fn, expected = paginated_feedback
    page = tools_library.get_function_feedback(fn.name, item_id=item_id, limit=1)
    assert page["ok"], page
    rows = [row for row in expected["verdicts"] if row["item_id"] == item_id]
    validation = page["feedback"]["behavior_validation"]
    assert validation["verdicts"] == rows
    assert validation["sampled"] == 4
    assert validation["uncertain"] == validation["processing_errors"] == 1
    assert page["pagination"] == {
        "offset": 0, "limit": 1, "total": len(rows), "next_offset": None,
    }
    assert page["item_calibrations"] == [
        row for row in expected["item_calibrations"] if row["item_id"] == item_id
    ]
    if rows:
        after = tools_library.get_function_feedback(fn.name, item_id=item_id, offset=1, limit=1)
        assert after["feedback"]["behavior_validation"]["verdicts"] == []
        assert after["item_calibrations"] == []
        assert after["pagination"]["total"] == 1


@pytest.mark.parametrize("arguments", [{"offset": -1}, {"limit": 51}])
def test_feedback_rejects_invalid_pagination(tmp_path, monkeypatch, arguments):
    _, fn = _runtime(tmp_path, monkeypatch)
    response = tools_library.get_function_feedback(fn.name, **arguments)
    assert not response["ok"]
    assert "offset" in response["error"] and "limit" in response["error"]


@pytest.mark.parametrize("outcomes,refreshed", [
    (["uncertain"] * 4, True), (["error"] * 4, False),
])
def test_auto_submit_delivers_compact_actionable_feedback_and_honest_refresh_status(
    tmp_path, monkeypatch, outcomes, refreshed,
):
    rt, fn = _runtime(tmp_path, monkeypatch, automatic_eligible=True)
    fn.scores.update(judge_backend="claude_agent_sdk", behavior_n_judged=4)

    def judge(**kwargs):
        return _result(rt.library.get(fn.name), outcomes)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    result = resubmit(fn, description=fn.description + " Revised claim.")
    assert result["ok"]
    assert result["judge_refresh"]["refreshed"] is refreshed
    assert result["judge_refresh"]["semantic_reviews"] == 4 - outcomes.count("error")
    assert result["judge_feedback"]["uncertain"] == outcomes.count("uncertain")
    assert result["judge_feedback"]["processing_errors"] == outcomes.count("error")
    assert result["judge_feedback"]["next_steps"]
    _assert_compact_feedback(result)
    assert "FULL-REVIEW-DETAIL" not in json.dumps(result)
    # The compact response does not repeat the policy or mutate saved assessments.
    assert "judge_prefilter" not in result["feedback"]["behavior_validation"]
    full_feedback = tools_library.get_function_feedback(fn.name)
    assert (full_feedback["feedback"]["behavior_validation"]["judge_prefilter"]
            == result["feedback"]["judge_prefilter"])
    if not refreshed:
        assert result["judge_refresh"]["status"] == "error"


@pytest.mark.parametrize("prior", ["new", "stale"])
def test_default_submit_returns_immediate_judgment_for_eligible_new_or_stale_fires(
    tmp_path, monkeypatch, prior,
):
    rt, fn = _runtime(tmp_path, monkeypatch, automatic_eligible=True)
    calls = []

    def judge(**kwargs):
        calls.append(kwargs)
        ids = [item["_item_id"] for item in kwargs["items"]]
        return _result(rt.library.get(fn.name), ["yes"] * len(ids), item_ids=ids)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    if prior == "new":
        rt.library = FunctionLibrary()
    elif prior == "stale":
        assert tools_eval.judge_function(fn.name)["ok"]
        fn.scores["judge_policy_hash"] = "previous-evidence-projection"
        calls.clear()
    result = resubmit(fn)
    assert result["ok"] and result["evaluated"]
    assert len(calls) == 1
    assert len(calls[0]["items"]) == 3
    assert result["judge_refresh"]["refreshed"]
    assert result["judge_feedback"]["confirmed"] == 3
    assert result["example_fail_ids"]
    scores = rt.library.get(fn.name).scores
    assert scores["ppv_at_pi0"] >= .7
    assert scores["n_matched_failed_tasks"] == 5


def test_explicit_standard_review_includes_passing_matches(tmp_path, monkeypatch):
    rt, fn = _runtime(tmp_path, monkeypatch)
    for item in rt.dataset.items:
        item["is_correct"] = True
    reviewed = []

    def judge(**kwargs):
        reviewed.extend(kwargs["items"])
        ids = [item["_item_id"] for item in kwargs["items"]]
        return _result(fn, ["yes"] * len(ids), item_ids=ids)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    # Passing matches remain valid adherence evidence in a deliberate review;
    # they cannot supply the failed-task breadth required for automatic review.
    result = tools_eval.judge_function(fn.name, review_mode="standard", evidence_n=4)
    assert result["ok"] and result["n_judged"] == 4
    assert len(reviewed) == 4 and all(item["is_correct"] for item in reviewed)
    assert result["judge_feedback"]["confirmed"] == 4


@pytest.mark.parametrize("outcome", ["uncertain", "error"])
def test_unchanged_current_submit_does_not_repeat_semantic_or_failed_reviews(
    tmp_path, monkeypatch, outcome,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude",
                        lambda **kwargs: _bind_review_snapshots(
                            _result(rt.library.get(fn.name), [outcome] * 4), kwargs,
                        ))
    assert tools_eval.judge_function(fn.name)["ok"]
    previous_rows = fn.scores["behavior_verdicts"]

    def unexpected_judge(**kwargs):
        pytest.fail("An unchanged reviewed version must not be automatically retried.")

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", unexpected_judge)
    result = resubmit(fn)
    assert result["ok"] and result["evaluated"]
    assert result["judge_refresh"] is None
    assert rt.library.get(fn.name).scores["behavior_verdicts"] == previous_rows


def test_source_only_edit_replaces_prior_confirmation_with_immediate_new_review(
    tmp_path, monkeypatch,
):
    rt, fn = _runtime(tmp_path, monkeypatch, automatic_eligible=True)
    calls = []

    def judge(**kwargs):
        calls.append(kwargs["source"])
        outcomes = ["yes"] * 4 if len(calls) == 1 else ["uncertain"] * 4
        return _result(rt.library.get(fn.name), outcomes)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    assert tools_eval.judge_function(fn.name)["ok"]
    changed_source = SOURCE.replace("item['response_text']", "item['response_text'].lower()")
    result = resubmit(fn, source=changed_source)
    assert calls == [SOURCE, changed_source]
    assert result["judge_refresh"]["refreshed"]
    assert result["judge_feedback"]["confirmed"] == 0
    assert result["judge_feedback"]["uncertain"] == 4
    stored = FunctionLibrary.load(rt.working_library_path()).get(fn.name)
    assert stored.source == changed_source
    assert stored.scores["behavior_n_yes"] == 0
    assert stored.scores["behavior_purity"] is None


@pytest.mark.parametrize("skip", ["no_fires", "empty"])
def test_submit_skips_judging_when_no_evaluated_fires_are_available(
    tmp_path, monkeypatch, skip,
):
    rt, fn = _runtime(tmp_path, monkeypatch)

    def unexpected_judge(*args, **kwargs):
        pytest.fail("This submission has no evaluated fires to judge.")

    monkeypatch.setattr(tools_eval, "judge_function", unexpected_judge)
    if skip == "no_fires":
        for item in rt.dataset.items:
            item["response_text"] = "A fully derived answer."
    else:
        rt.dataset.items = []
    result = resubmit(fn)
    assert result["ok"]
    if skip == "no_fires":
        assert result["judge_refresh"]["skipped"]
        assert result["judge_feedback"]["status"] == "skipped_pre_filter"
    else:
        assert result.get("judge_refresh") is None
        # Unevaluated submissions are saved under the shared NOT EVALUATED label.
        assert result["verdict"] == verdict_for_scores({})
        assert rt.library.get(fn.name) is not None
    assert result["evaluated"] is (skip == "no_fires")


def test_default_submit_exposes_judge_failure_without_claiming_review_success(
    tmp_path, monkeypatch,
):
    rt, fn = _runtime(tmp_path, monkeypatch, automatic_eligible=True)

    def fail_judge(**kwargs):
        raise RuntimeError("Judge service unavailable.")

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", fail_judge)
    result = resubmit(fn)
    assert result["ok"] and result["evaluated"]
    assert result["judge_refresh"]["refreshed"] is False
    assert result["judge_refresh"]["semantic_reviews"] == 0
    assert result["judge_feedback"]["processing_errors"] == 3
    assert "Judge service unavailable" in result["judge_refresh"]["error"]
    assert rt.library.get(fn.name).scores["behavior_purity"] is None


def test_stale_feedback_is_retrievable_as_history_but_not_current_repair_advice():
    fn = FeatureFunction(name="guess", source=SOURCE, description=DESCRIPTION, mechanism=MECHANISM)
    result = _result(fn, ["no"])
    scores = {"judge_contract_version": JUDGE_CONTRACT_VERSION,
              "claim_hash": result["claim_hash"], "judge_policy_hash": "old-policy",
              "behavior_n_judged": 1, "behavior_n_no": 1,
              "behavior_verdicts": result["verdicts"], "claim_calibration": CALIBRATION,
              "calibration_hint": "HISTORICAL REWRITE ADVICE"}
    feedback = summarize_judge_feedback(scores, name=fn.name)
    assert feedback["status"] == "stale"
    assert feedback["findings"] == []
    assert "claim_calibration" not in feedback
    assert "judge_function" in feedback["next_steps"][0]
    hint = tools_library._feedback_hint(scores)
    assert "judge_function" in hint and "HISTORICAL" not in hint


def test_pareto_status_keeps_uncertain_candidate_out_of_validated_top_x(tmp_path, monkeypatch):
    rt, fn = _runtime(tmp_path, monkeypatch)
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    rt.scores_history.clear()
    expected = _result(fn, ["uncertain"] * 4)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", lambda **kwargs: expected)
    tools_eval.judge_function(fn.name)
    rt.config.objective_axes = ["behavior_purity"]
    rt.config.objective_weights = [100]
    status = tools_library.pareto_status()
    assert status["top_x"] == [fn.name]
    assert status["validated_top_x"] == []
    assert [row["name"] for row in status["research_candidates"]] == [fn.name]
    candidate = status["research_candidates"][0]
    assert not candidate["eligible"] and candidate["status"] == "uncertain"


def test_compact_feedback_includes_decisive_nested_component():
    fn = FeatureFunction(name="guess", source=SOURCE, description=DESCRIPTION, mechanism=MECHANISM)
    result = _result(fn, ["no"])
    result["verdicts"][0]["claim_assessments"] = [{
        "requirement_id": "mechanism", "status": "supported", "rationale": "Broadly related.",
        "components": [{"claim": "The input rate was absent", "status": "contradicted",
                        "rationale": "The rate was documented in the preceding user message."}],
    }]
    feedback = summarize_judge_feedback({
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": result["judge_policy_hash"], "claim_hash": result["claim_hash"],
        "claim_contract": result["claim_contract"], "behavior_n_judged": 1,
        "behavior_n_no": 1, "behavior_verdicts": result["verdicts"],
    }, name=fn.name)
    requirement = feedback["findings"][0]["requirements"][0]
    assert requirement["status"] == "contradicted"
    assert "rate was absent" in requirement["requirement"]
    assert "preceding user message" in requirement["rationale"]

# Review diagnosis: counterexamples and evidence gaps get distinct, executable next steps.


def _function(diagnosis, *, unknown_reason="ambiguous_evidence", extra_clauses=0):
    claim = {
        "source": "def detector(item):\n    return True",
        "description": "The assistant omits the requested deliverable.",
        "mechanism": "The assistant withholds a deliverable it could have supplied.",
    }
    contract = build_claim_contract(**claim, defining_clauses=[{
        "id": f"premise_{index}", "source_field": "mechanism",
        "quote": "a deliverable it could have supplied",
        "predicate": f"Required evidence distinction {index}",
        "required_review_scope": "local",
    } for index in range(extra_clauses)] or None)
    assessments = [{
        "requirement_id": requirement["id"], "status": "unknown",
        "unknown_reason": unknown_reason, "rationale": "The required premise is not established.",
        "citations": [],
    } for requirement in contract["requirements"]]
    acceptance = next(a for a in assessments if a["requirement_id"] == "failure_contribution")
    acceptance["diagnosis"] = diagnosis
    # Deliberately place the policy blocker after the old two-requirement cap.
    assessments = [a for a in assessments if a is not acceptance] + [acceptance]
    row = {
        "item_id": "item-1", "verdict": "uncertain", "processing_status": "completed",
        "claim_contract_id": contract["contract_id"], "claim_assessments": assessments,
        "rationale": "The behavior is visible; the claim's missing premise is unresolved.",
    }
    return {
        "name": "candidate", **claim,
        "scores": {
            "judge_contract_version": JUDGE_CONTRACT_VERSION,
            "judge_policy_hash": claude_judge.judge_policy_hash(),
            "claim_hash": claim_hash(**claim),
            "claim_contract": contract, "behavior_verdicts": [row],
            "behavior_n_judged": 1, "behavior_n_yes": 0, "behavior_n_no": 0,
            "behavior_n_uncertain": 1, "behavior_n_errors": 0, "behavior_n_completed": 0,
            "behavior_purity": None,
        },
    }


WITHHELD = {
    "observed_facts": "The assistant says it does not know the answer.",
    "unsupported_premise": "The assistant had the information needed to answer.",
    "competing_explanation": "It may have lacked the information and avoided guessing.",
    "evidence_needed": "The information available to the assistant at the time of its reply.",
    "evidence_availability": "unknown",
    "availability_reason": "The review did not establish whether that information was retained.",
}
ATTACHMENT = {
    "observed_facts": "The user requests OCR and the assistant asks for an upload.",
    "unsupported_premise": "An attachment was uploaded and omitted from the assistant's input.",
    "competing_explanation": "The user may not have uploaded an attachment.",
    "evidence_needed": "Upload records and the actual input delivered to the assistant.",
    "evidence_availability": "unavailable",
    "availability_reason": "Neither record is retained in this example.",
}


@pytest.mark.parametrize("diagnosis,reason,action", [
    (WITHHELD, "ambiguous_evidence", "check_evidence_availability"),
    (ATTACHMENT, "evidence_unavailable", "defer_claim"),
])
def test_submission_summary_preserves_policy_blocker_and_discriminating_fact(
    diagnosis, reason, action,
):
    fn = _function(diagnosis, unknown_reason=reason)
    before = deepcopy(fn)
    summary = summarize_judge_feedback(fn["scores"], name=fn["name"])
    requirements = summary["findings"][0]["requirements"]
    assert requirements[0]["requirement_id"] == "failure_contribution"
    assert len(summary["findings"][0]["unresolved_requirements"]) == len(
        fn["scores"]["claim_contract"]["requirements"]
    )
    assert len(requirements) <= 3
    actual = summary["findings"][0]["diagnosis"]
    assert {key: actual[key] for key in diagnosis} == diagnosis
    assert actual["next_action"]["kind"] == action
    assert diagnosis["evidence_needed"] in summary["next_steps"][0]
    assert actual["next_action"]["instruction_ref"] == "summary.next_steps[0]"
    assert summary["confirmed"] == 0 and summary["uncertain"] == 1
    assert fn == before
    repair = build_repair_feedback(fn)
    first = repair["investigations"][0]
    assert first["action"] == action
    assert first["diagnosis"]["unsupported_premise"] == diagnosis["unsupported_premise"]
    assert diagnosis["evidence_needed"] in first["next_step"]
    if action == "defer_claim":
        assert first["tool"] == "get_function_feedback"
        signature(tools_library.get_function_feedback).bind(**first["arguments"])
    else:
        signature(tools_data.get_item).bind(**first["arguments"])


def test_legacy_diagnosis_marks_missing_structure_instead_of_inventing_facts():
    fn = _function({}, unknown_reason="evidence_unavailable")
    summary = summarize_judge_feedback(fn["scores"], name=fn["name"])
    diagnosis = summary["findings"][0]["diagnosis"]
    assert diagnosis["provenance"] == "saved_assessment_fallback"
    assert diagnosis["evidence_availability"] == "unavailable"
    assert "competing_explanation" in diagnosis["incomplete_fields"]
    assert diagnosis["next_action"]["kind"] == "defer_claim"
    assert diagnosis["evidence_needed"] == "The required premise is not established."


def test_many_unresolved_clauses_have_complete_manifest_and_visible_policy_blocker():
    fn = _function(WITHHELD, extra_clauses=6)
    summary = summarize_judge_feedback(fn["scores"], name=fn["name"])
    finding = summary["findings"][0]
    required = {a["requirement_id"] for a in fn["scores"]["behavior_verdicts"][0][
        "claim_assessments"
    ]}
    assert {a["requirement_id"] for a in finding["unresolved_requirements"]} == required
    assert finding["requirements"][0]["requirement_id"] == "failure_contribution"
    assert len(finding["requirements"]) == 3
    assert finding["requirements_omitted"] == len(required) - 3
    assert finding["diagnosis"]["unsupported_premise"] == WITHHELD["unsupported_premise"]
    assert finding["details_tool"]["arguments"] == {"name": "candidate", "item_id": "item-1"}


def test_identical_gaps_share_diagnosis_without_reusing_another_items_read():
    fn = _function(WITHHELD)
    row = fn["scores"]["behavior_verdicts"][0]
    duplicate = deepcopy(row)
    duplicate["item_id"] = "item-2"
    fn["scores"]["behavior_verdicts"].append(duplicate)
    fn["scores"].update(behavior_n_judged=2, behavior_n_uncertain=2)
    findings = summarize_judge_feedback(fn["scores"], name=fn["name"])["findings"]
    assert len(findings) == 2
    second = findings[1]["diagnosis"]
    assert second["same_as_item_id"] == "item-1"
    assert second["next_action"]["arguments"]["item_id"] == "item-2"


def test_available_unread_evidence_keeps_original_budget_and_actual_citation_tool():
    row = {"item_id": "case", "claim_assessments": []}
    assessment = {
        "status": "unknown", "unknown_reason": "review_incomplete",
        "rationale": "The remaining trajectory has not been read.",
        "citations": [{"field": "trajectory_text", "quote": "Still working"}],
        "diagnosis": {**WITHHELD, "evidence_availability": "unavailable",
                      "tool": "invented_tool", "arguments": {"secret": True}},
    }
    diagnosis = build_review_diagnosis(row, assessment, requirement="Required claim", name="f")
    assert diagnosis["evidence_availability"] == "available"
    assert diagnosis["availability_source"] == "assessment_unknown_reason"
    assert diagnosis["overridden_fields"] == ["evidence_availability", "availability_reason"]
    action = diagnosis["next_action"]
    assert action["tool"] == "get_trajectory"
    signature(tools_data.get_trajectory).bind(**action["arguments"])
    assert "invented_tool" not in str(diagnosis)


@pytest.mark.parametrize("field,tool", [
    ("response_text", "read_field_chunk"),
    ("unknown_judge_projection", "get_item"),
    ("detector_source", "get_function_feedback"),
])
def test_next_action_reads_real_surfaces_for_synthetic_judge_citations(monkeypatch, field, tool):
    from autofa import trace

    monkeypatch.setattr(tools_data, "_load_item_raw", lambda _: {
        "_item_id": "case", "response_text": "The retained answer.",
    })
    monkeypatch.setattr(trace, "coverage", lambda _: {"status": "available"})
    calls = []

    def get_feedback(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(tools_library, "get_function_feedback", get_feedback)
    diagnosis = build_review_diagnosis(
        {"item_id": "case"}, {
            "status": "unknown", "unknown_reason": "ambiguous_evidence",
            "rationale": "Inspect the source.", "citations": [{"field": field}],
        }, requirement="Required premise", name="candidate",
    )
    action = diagnosis["next_action"]
    assert action["tool"] == tool
    module = tools_library if tool == "get_function_feedback" else tools_data
    if tool in {"get_item", "get_trajectory"}:
        # These loaders require a Runtime; bind their real contracts, while the
        # synthetic field and actual source read paths are dispatched below.
        signature(getattr(module, tool)).bind(**action["arguments"])
    else:
        result = getattr(module, tool)(**action["arguments"])
        assert result["ok"], result
    if tool == "read_field_chunk":
        assert action["arguments"]["limit"] == 4000
    if calls:
        assert calls == [{"name": "candidate", "item_id": "case"}]


def test_missing_item_identity_returns_function_feedback_instead_of_an_invalid_item_read():
    diagnosis = build_review_diagnosis(
        {}, {"status": "unknown", "rationale": "Unresolved"},
        requirement="Claim", name="candidate",
    )
    assert diagnosis["next_action"]["tool"] == "get_function_feedback"
    assert diagnosis["next_action"]["arguments"] == {"name": "candidate"}
    assert diagnosis["details_tool"]["arguments"] == {"name": "candidate"}


def test_calibration_preserves_full_advice_but_never_overrides_missing_premise():
    fn = _function(ATTACHMENT, unknown_reason="evidence_unavailable")
    advice = {
        "status": "mixed", "claim_note": "A premise is unsupported.",
        "detector_note": "The visible wording is a proxy.",
        "rewrite_hint": "Require wording that implies a file exists. " * 30,
    }
    fn["scores"]["claim_calibration"] = advice
    parsed = parse_claim_calibration(fn["scores"])
    assert parsed["rewrite_hint"] == advice["rewrite_hint"].strip()
    summary = summarize_judge_feedback(fn["scores"], name=fn["name"])
    assert "rewrite_hint" in summary["claim_calibration"]["fields_truncated"]
    assert (summary["findings"][0]["diagnosis"]["next_action"]["kind"]
            == "defer_claim")
    assert fn["scores"]["claim_calibration"] == advice


def _contradicted_function(requirement_id, rationale):
    fn = _function({})
    row = fn["scores"]["behavior_verdicts"][0]
    # Shape of the completed item-503 review: a clear negative finding with no
    # structured diagnosis, rather than a review that needs missing evidence.
    row.update(item_id="item-503", verdict="no", rationale=rationale)
    for assessment in row["claim_assessments"]:
        assessment.update(
            status="supported", rationale="The other requirement is supported.",
            citations=[{"field": "response_text", "quote": "The retained evidence."}],
        )
        assessment.pop("unknown_reason", None)
        assessment.pop("diagnosis", None)
    rejected = next(a for a in row["claim_assessments"]
                    if a["requirement_id"] == requirement_id)
    rejected.update(
        status="contradicted", rationale=rationale,
        citations=[{"field": "response_text", "quote": "The requested content continues."}],
    )
    fn["scores"].update(
        behavior_n_no=1, behavior_n_uncertain=0, behavior_n_completed=1, behavior_purity=0.0,
    )
    return fn


@pytest.mark.parametrize("requirement_id,rationale,action", [
    (
        "description",
        "The reply continues the requested content. A brief clarification softens the "
        "wording, but does not withhold or replace the requested deliverable.",
        "tighten_detector",
    ),
    (
        "failure_contribution",
        "The visible mistake was corrected before the final answer. A different, "
        "unrelated error caused the task to fail.",
        "revise_claim",
    ),
])
def test_contradictions_reach_submission_and_repair_as_counterexamples(
    requirement_id, rationale, action,
):
    fn = _contradicted_function(requirement_id, rationale)
    before = deepcopy(fn)
    summary = summarize_judge_feedback(fn["scores"], name=fn["name"])
    diagnosis = summary["findings"][0]["diagnosis"]
    assert diagnosis["assessment_status"] == "contradicted"
    assert diagnosis["diagnosis_kind"] == "counterexample"
    assert diagnosis["observed_facts"] == rationale
    assert diagnosis["next_action"]["kind"] == action
    instruction = summary["next_steps"][0]
    assert diagnosis["next_action"]["tool"] == "read_field_chunk"
    signature(tools_data.read_field_chunk).bind(**diagnosis["next_action"]["arguments"])
    repair = build_repair_feedback(fn)
    assert repair["review_detail_complete"], repair["review_integrity_errors"]
    investigation = repair["investigations"][0]
    assert investigation["issue_type"] == "counterexample"
    assert investigation["action"] == action
    assert instruction == f"Item item-503: {investigation['next_step']}"
    assert repair["recommended_action"] == "tighten_detector"
    assert summary["rejected"] == 1 and summary["uncertain"] == 0
    assert fn == before


def test_contradiction_routing_cannot_be_overridden_by_old_unknown_metadata():
    # An unknown would route review_incomplete to available evidence; a contradiction does not.
    assessment = {
        "requirement_id": "description", "status": "contradicted",
        "rationale": "The requested answer is present in the reply.",
        "unknown_reason": "review_incomplete",
        "diagnosis": {**WITHHELD, "evidence_availability": "unavailable"},
    }
    before = deepcopy(assessment)
    diagnosis = build_review_diagnosis(
        {"item_id": "item-503"}, assessment, requirement="Withholds the answer", name="candidate",
    )
    action = diagnosis["next_action"]
    assert action["kind"] == "tighten_detector"
    assert diagnosis["evidence_availability"] == "unavailable"
    assert assessment == before


def test_unknown_policy_premise_does_not_hide_an_observed_behavior_counterexample():
    fn = _contradicted_function("description", "The requested deliverable is present.")
    row = fn["scores"]["behavior_verdicts"][0]
    acceptance = next(a for a in row["claim_assessments"]
                      if a["requirement_id"] == "failure_contribution")
    acceptance.update(
        status="unknown", unknown_reason="evidence_unavailable", diagnosis=ATTACHMENT,
        rationale="Attachment-delivery records were not retained.",
    )
    summary = summarize_judge_feedback(fn["scores"], name=fn["name"])
    finding = summary["findings"][0]
    assert finding["requirements"][0]["requirement_id"] == "description"
    assert finding["diagnosis"]["next_action"]["kind"] == "tighten_detector"
    assert any(a["requirement_id"] == "failure_contribution" and a["status"] == "unknown"
               for a in finding["unresolved_requirements"])
    repair = build_repair_feedback(fn)
    assert repair["review_detail_complete"], repair["review_integrity_errors"]
    assert repair["investigations"][0]["action"] == "tighten_detector"
    assert any(i["action"] == "defer_claim" for i in repair["investigations"])


def test_identical_editorial_diagnosis_does_not_merge_contradiction_with_unknown():
    fn = _contradicted_function("failure_contribution", "The behavior did not cause failure.")
    row = fn["scores"]["behavior_verdicts"][0]
    acceptance = next(a for a in row["claim_assessments"]
                      if a["requirement_id"] == "failure_contribution")
    acceptance["diagnosis"] = WITHHELD
    uncertain = deepcopy(row)
    uncertain.update(item_id="unknown-item", verdict="uncertain")
    other = next(a for a in uncertain["claim_assessments"]
                 if a["requirement_id"] == "failure_contribution")
    other.update(status="unknown", unknown_reason="ambiguous_evidence")
    fn["scores"]["behavior_verdicts"].append(uncertain)
    fn["scores"].update(behavior_n_judged=2, behavior_n_uncertain=1)
    findings = summarize_judge_feedback(fn["scores"], name=fn["name"])["findings"]
    assert len(findings) == 2
    assert findings[0]["diagnosis"]["next_action"]["kind"] == "revise_claim"
    assert findings[1]["diagnosis"]["next_action"]["kind"] == "check_evidence_availability"
    assert "same_as_item_id" not in findings[1]["diagnosis"]

# Library reads report current quality labels without rewriting saved evidence.


def library_reads(name):
    """The discovery reads that report a detector's current quality verdict."""
    return (tools_library.get_library_function(name), tools_library.get_function_feedback(name),
            tools_library.list_library_functions())


@pytest.mark.parametrize("state, code, label", [
    ("good", "GOOD", "GOOD"), ("unscored", "UNSCORED", "NOT EVALUATED"),
])
def test_library_reads_return_shared_quality_verdicts(tmp_path, monkeypatch, state, code, label):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = scored_function(state)
    rt.library.functions = [fn]
    before = deepcopy(fn.scores)
    expected = verdict_for_scores(fn.scores)
    assert expected["code"] == code
    assert expected["label"] == label

    inspected, feedback, listed = library_reads(fn.name)
    assert inspected["ok"] and feedback["ok"] and listed["ok"]
    assert inspected["verdict"] == expected
    assert inspected["function"]["scores"]["verdict"] == expected
    assert inspected["function"]["scores"]["badge"] == label
    assert feedback["verdict"] == expected
    assert listed["functions"][0]["verdict"] == expected
    assert fn.scores == before


@pytest.mark.parametrize("stale_reason", ["claim", "judge_policy"])
def test_reads_replace_saved_good_label_without_mutating_library_or_history(
    tmp_path, monkeypatch, stale_reason,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = scored_function("good")
    rt.library.functions = [fn]
    saved_verdict = verdict_for_scores(fn.scores)
    assert saved_verdict["code"] == "GOOD"
    fn.scores.update(verdict=saved_verdict, badge=saved_verdict["label"])
    rt.scores_history = [{"name": fn.name, "scores": deepcopy(fn.scores)}]
    if stale_reason == "claim":
        fn.description += " The assistant also claims to have verified its answer."
    else:
        # A different acceptance standard requires its own current review.
        rt.config.behavior_strictness = "contextual"
    library_path = rt.persist_library()
    history_path = rt.persist_scores_history()
    before = deepcopy(rt.library.to_dict())
    history_before = deepcopy(rt.scores_history)
    library_bytes = library_path.read_bytes()
    history_bytes = history_path.read_bytes()

    inspected, feedback, listed = library_reads(fn.name)
    expected = {
        **verdict_for_scores({}),
        "reason": {
            "claim": "Stale judgment — claim or detector changed.",
            "judge_policy": "Stale judgment — judge policy changed.",
        }[stale_reason],
    }
    assert inspected["verdict"] == expected
    assert inspected["function"]["scores"]["verdict"] == expected
    assert inspected["function"]["scores"]["badge"] == "NOT EVALUATED"
    assert feedback["verdict"] == expected
    assert listed["functions"][0]["verdict"] == expected
    assert rt.library.to_dict() == before
    assert rt.scores_history == history_before
    assert library_path.read_bytes() == library_bytes
    assert history_path.read_bytes() == history_bytes


@pytest.mark.parametrize("auto_evaluate", [False, True])
def test_unchanged_resubmit_classifies_current_scores_after_optional_evaluation(
    tmp_path, monkeypatch, auto_evaluate,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    fn.scores = scored_function("good").scores
    attach_task_support(fn)
    saved_verdict = verdict_for_scores(fn.scores)
    assert saved_verdict["code"] == "GOOD"
    fn.scores.update(verdict=saved_verdict, badge=saved_verdict["label"])
    before = deepcopy(fn.scores)
    calls = []

    def fail_evaluation(**kwargs):
        assert auto_evaluate, "Saving without evaluation must not run an evaluation"
        calls.append(kwargs)
        current = rt.library.get(fn.name)
        assert current.scores == before  # Exact resubmission retained the existing review.
        current.scores["evaluation_stale"] = True
        return {"ok": False, "error": "Evaluation interrupted"}

    monkeypatch.setattr(tools_eval, "evaluate_functions", fail_evaluation)
    submitted = resubmit(fn, auto_evaluate=auto_evaluate)
    assert submitted["ok"] and submitted["evaluated"] is False, submitted
    assert len(calls) == int(auto_evaluate)
    if auto_evaluate:
        assert submitted["verdict"] == {
            **verdict_for_scores({}),
            "reason": "Evaluation is stale; score the current detector.",
        }
        assert submitted["verdict"]["label"] == "NOT EVALUATED"
        assert rt.library.get(fn.name).scores["evaluation_stale"] is True
    else:
        assert submitted["verdict"] == saved_verdict
        assert rt.library.get(fn.name).scores == before


def test_submit_reports_quality_after_automatic_evaluation_and_review(tmp_path, monkeypatch):
    rt, fn = _runtime(tmp_path, monkeypatch)
    rt.dataset.items[-1]["response_text"] = "A derived value."
    calls = []

    def review(**kwargs):
        calls.append(kwargs)
        return _result(rt.library.get(fn.name), ["yes"] * len(kwargs["items"]))

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", review)
    submitted = resubmit(fn)
    assert submitted["ok"] and submitted["evaluated"], submitted
    assert calls
    expected = verdict_for_scores(rt.library.get(fn.name).scores)
    # Three failed tasks remain an OK seed even with perfect aggregate adherence.
    assert expected["code"] == "OK"
    assert submitted["verdict"] == expected
    assert tools_library.get_library_function(fn.name)["verdict"] == expected
    assert tools_library.get_function_feedback(fn.name)["verdict"] == expected
    assert tools_library.list_library_functions()["functions"][0]["verdict"] == expected


def test_sampled_projection_is_used_and_explained_in_agent_tool_responses(tmp_path, monkeypatch):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = scored_function("sampled_support")
    rt.library.functions = [fn]
    before = deepcopy(fn.scores)

    inspected, feedback, listed = library_reads(fn.name)
    for verdict in (inspected["verdict"], feedback["verdict"],
                    listed["functions"][0]["verdict"], tools_eval._current_verdict(rt, fn)):
        assert verdict["code"] == "GOOD"
    scores = inspected["function"]["scores"]
    assert scores["n_confirmed_failed_tasks"] == 2
    assert scores["task_support_comparison"] == pytest.approx(8)
    assert scores["task_support_comparison_basis"] == "estimated"
    assert "task_support" not in scores  # Detailed receipt is not repeated in tool previews.
    assert feedback["feedback"]["n_confirmed_failed_tasks"] == 2
    assert feedback["feedback"]["task_support_comparison"] == pytest.approx(8)
    assert feedback["feedback"]["task_support_comparison_basis"] == "estimated"
    assert tools_eval._current_discovery_grade(rt, fn)["code"] == "OUTSTANDING"
    assert fn.scores == before

# Presentation references save bandwidth while preserving exact immutable evidence.


def _pointer(root, pointer):
    value = root
    for part in pointer.removeprefix("#/").split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _expanded(value, root):
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            return deepcopy(_pointer(root, value["$ref"]))
        return {key: _expanded(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_expanded(item, root) for item in value]
    return value


def _verdict(item_id):
    citation = {"citation_id": "shared/id~suffix", "item_id": item_id,
                "field": "response_text", "quote": "Verbatim evidence λ \\n " * 200,
                "start": 17, "end": 4017, "complete": True}
    changed = {**citation, "start": 18, "complete": False}
    return {
        "item_id": item_id, "verdict": "yes", "rationale": "Exact saved rationale.",
        "citations": [deepcopy(citation)],
        "claim_assessments": [
            {"requirement_id": "action", "citations": [deepcopy(citation)]},
            {"requirement_id": "consequence", "citations": [deepcopy(citation), changed]},
        ],
        "initial_review": {"rationale": "Different initial reasoning.",
                           "citations": [deepcopy(citation)]},
    }


@pytest.mark.parametrize("tool", ["library", "feedback"])
def test_review_references_reconstruct_exact_evidence_and_survive_library_changes(
    tmp_path, monkeypatch, tool,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    fn.scores = {"behavior_verdicts": [_verdict("case-0"), _verdict("case-1")],
                 "behavior_n_judged": 2, "behavior_n_yes": 2}
    fn.failure_localisations = {"case-0": {"passage": "Exact localised passage."}}
    before = deepcopy(fn.to_dict())
    library_path = tmp_path / "original-library.json"
    rt.library.save(library_path)
    saved_library = library_path.read_bytes()
    if tool == "library":
        response = tools_library.get_library_function(fn.name)
        path = "#/function/scores/behavior_verdicts"
        expected = before["scores"]["behavior_verdicts"]
    else:
        response = tools_library.get_function_feedback(fn.name, offset=1, limit=1)
        path = "#/feedback/behavior_validation/verdicts"
        expected = before["scores"]["behavior_verdicts"][1:]
        assert response["pagination"] == {
            "offset": 1, "limit": 1, "total": 2, "next_offset": None,
        }
    rows = _pointer(response, path)
    assert _expanded(rows, response) == expected
    assert rows[0]["claim_assessments"][0]["citations"][0] == {
        "$ref": path + "/0/citations/0",
    }
    # Reusing an ID never merges different offsets or completeness evidence.
    changed = rows[0]["claim_assessments"][1]["citations"][1]
    assert changed["start"] == 18 and not changed["complete"]
    assert len(json.dumps(rows)) < len(json.dumps(expected)) * .6
    assert fn.to_dict() == before
    assert library_path.read_bytes() == saved_library
    assert [row["item_id"] for row in rows] == [row["item_id"] for row in expected]

    # The full-result checkpoint is independent of future library state.
    fn.scores.clear()
    rt.library.functions.clear()
    offset, pages = 0, []
    while True:
        page = tool_results.read_tool_result(response["full_result"]["result_id"], offset=offset)
        assert page["ok"]
        pages.append(page["text"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    original = json.loads("".join(pages))
    assert _pointer(original, path) == expected
    expanded = _expanded(response, response)
    expanded.pop("full_result")
    expanded.pop("citation_references")
    assert expanded == original


def test_review_checkpoint_failure_returns_full_original_and_pointer_keys_are_escaped(
    tmp_path, monkeypatch,
):
    _runtime(tmp_path, monkeypatch)
    original = {"review/~key": {"verdicts": [_verdict("case-0")]}}
    before = deepcopy(original)
    response = tools_library.reference_review_response(
        original, verdict_path=("review/~key", "verdicts"),
    )
    citation = response["review/~key"]["verdicts"][0]["claim_assessments"][0]["citations"][0]
    assert citation["$ref"] == "#/review~1~0key/verdicts/0/citations/0"
    assert _expanded(response["review/~key"], response) == original["review/~key"]
    assert original == before

    def unavailable(*args, **kwargs):
        raise OSError("Synthetic storage failure")

    monkeypatch.setattr(tool_results, "save_tool_result", unavailable)
    unchanged = tools_library.reference_review_response(
        original, verdict_path=("review/~key", "verdicts"),
    )
    assert unchanged is original and unchanged == before


def test_formal_and_supplemental_references_share_one_unmodified_full_snapshot(monkeypatch):
    citation = {"citation_id": "same", "field": "response_text",
                "start": 0, "end": 900, "quote": "x" * 900}
    verdict = {
        "item_id": "case", "citations": [citation],
        "claim_assessments": [{"citations": [citation]}],
        "initial_review": {"citations": [citation]},
    }
    original = {
        "feedback": {"behavior_validation": {"verdicts": [verdict]}},
        "diagnostic_review": {"verdicts": [{**deepcopy(verdict), "item_id": "other"}]},
    }
    before = deepcopy(original)
    saved = []

    def save(response):
        saved.append(deepcopy(response))
        return {"result_id": "immutable-original"}

    monkeypatch.setattr("autofa.mcp.tool_results.save_tool_result", save)
    result = tools_library.reference_review_response(
        original, verdict_path=("feedback", "behavior_validation", "verdicts"),
        extra_verdict_paths=(("diagnostic_review", "verdicts"),))
    assert original == before
    assert saved == [before]
    assert result["citation_references"]["count"] == 4
    assert result["full_result"]["result_id"] == "immutable-original"
    for path in (("feedback", "behavior_validation", "verdicts"),
                 ("diagnostic_review", "verdicts")):
        node = result
        for key in path:
            node = node[key]
        row = node[0]
        expected = {"$ref": "#/" + "/".join(path) + "/0/citations/0"}
        assert row["claim_assessments"][0]["citations"][0] == expected
        assert row["initial_review"]["citations"][0] == expected
        assert row["citations"][0] == citation


def test_smoke_histogram_references_preserve_long_values_and_scalar_counts(monkeypatch):
    long_value = "Complete trace λ \\n " * 1000
    items = [{"_item_id": name, "response_text": value} for name, value in (
        ("long-a", long_value), ("short", "ordinary label"), ("long-b", long_value),
    )]

    class Traces(Dataset):
        def __init__(self):
            super().__init__(parquet_paths=[])

        def iter_batches(self, *, batch_size=512, max_items=None, columns=None):
            yield items[:max_items]

    rt = SimpleNamespace(dataset=Traces(), library=FunctionLibrary(), config=SimpleNamespace())
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: rt)
    monkeypatch.setattr(tools_eval, "_llm_adapter", NullLLM)
    response = tools_eval.smoke_test_feature(
        "def detect(item):\n    return item['response_text']",
        item_ids=[x["_item_id"] for x in items],
    )
    assert response["ok"] and response["errors"] == 0
    assert response["histogram"] == {"ordinary label": 1}
    assert response["histogram_references"] == [{"value_ref": "#/results/0/value", "count": 2}]
    reference = response["histogram_references"][0]
    assert _pointer(response, reference["value_ref"]) == long_value
    assert [row["value"] for row in response["results"]] == [x["response_text"] for x in items]
    assert json.dumps(response).count(json.dumps(long_value)[1:-1]) == 2

    # Unbundled diagnostics can return non-scalars. JSON converts a tuple to an
    # array, so its original histogram spelling must remain available verbatim.
    tuple_result = tools_eval.smoke_test_feature(
        "def detect(item):\n    return (item['response_text'],)", item_ids=["long-a"],
    )
    assert tuple_result["errors"] == 0
    assert tuple_result["histogram"] == {str((long_value,)): 1}
    assert "histogram_references" not in tuple_result
