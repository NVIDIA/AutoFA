# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete, fresh and compact evidence, isolated sessions and verifiable judge output."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.core import localisation
from autofa.data import evidence_refs
from autofa.data.evidence_refs import compact_trace, expand_trace
from autofa.data.trace_navigation import TraceNavigation
from autofa.metrics import claude_judge, judge_evidence
from autofa.metrics.claim_compiler import compiler_policy_hash
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.judge_evidence import MAX_READ_CHARS, EvidenceReview, make_snapshot
from autofa.metrics.judge_trajectory import judge_trajectory_text
from autofa.metrics.pareto import point_from_scores
from autofa.metrics.verdicts import (
    behavior_validation,
    current_behavior_scores,
    resolved_behavior_purity,
)
from autofa.session import repair_workflow
from autofa.util.cancellation import Cancellation, cancellation_scope
from autofa.util.model_budget import model_call_budget, reserve_model_call

SOURCE = 'def detect(item, llm_client=None): return "guess" in item["response_text"]'

pytestmark = pytest.mark.usefixtures("stub_claim_compiler")


def _review(tmp_path, *, source=SOURCE, claim_contract=None, **fields):
    snapshot = make_snapshot(fields or {"response_text": "I guessed."}, item_id="item",
                             source=source, claim_contract=claim_contract)
    return EvidenceReview(snapshot, tmp_path / "review")


def _row(review, citations, **changes):
    """A verdict row for a review's current snapshot."""
    return {"item_id": "item", "snapshot_id": review.snapshot["snapshot_id"],
            "correct": True, "scope": "local", "citations": citations, **changes}


def _parse(review, data, **kwargs):
    reply = data if isinstance(data, str) else json.dumps(data)
    return claude_judge.parse_session_result(
        reply, [{"item_id": "item"}], reviews={"item": review}, **kwargs,
    )


def _read_all(review, field):
    result = review.read(field)
    text = result["text"]
    while result["next_offset"] is not None:
        result = review.read(field, result["next_offset"])
        text += result["text"]
    return text


def _verdict(review, *, field="response_text", correct=True):
    _read_all(review, "detector_source")
    passage = review.read(field)
    row = {
        "item_id": review.snapshot["item_id"],
        "snapshot_id": review.snapshot["snapshot_id"],
        "correct": correct,
        "scope": "local",
        "rationale": "Evidence in the original text.",
        "citations": [
            {"field": field, "start": 0, "end": len(passage["text"]), "quote": passage["text"]}
        ],
    }
    if contract := review.snapshot.get("claim_contract"):
        row.update(_claim_fields(contract, row["citations"], correct))
    return row


def _claim_fields(contract, citations, correct=True):
    defining_ids = [r["id"] for r in contract["requirements"] if r["kind"] == "defining_clause"]
    return {
        "claim_contract_id": contract["contract_id"],
        "claim_assessments": [
            {
                "requirement_id": requirement["id"],
                "status": "supported" if correct is True else (
                    "contradicted" if correct is False else "unknown"
                ),
                "rationale": "Checked this requirement against the original evidence.",
                "citations": citations, "counterevidence": [],
                **({"defining_requirement_ids": defining_ids}
                   if defining_ids and correct is not True
                   and requirement["kind"] in {"claim", "review_check"} else {}),
            }
            for requirement in contract["requirements"]
        ],
    }


def _assessment(verdict, requirement_id):
    return next(row for row in verdict["claim_assessments"]
                if row["requirement_id"] == requirement_id)


def _test_contract(source=SOURCE):
    return build_claim_contract(
        description="A guessed final answer", mechanism="Ships an admitted guess", source=source,
        defining_clauses=[
            {"id": "clause_1", "source_field": "description", "quote": "A guessed final answer",
             "required_review_scope": "local",
             "predicate": "A guessed final answer"},
            {"id": "clause_2", "source_field": "mechanism", "quote": "Ships an admitted guess",
             "required_review_scope": "local",
             "predicate": "Ships an admitted guess"},
        ],
        compilation={"policy_hash": compiler_policy_hash(), "model": "test", "exclusions": [],
                     "coverage_review": {"approved": True, "issues": []}},
    )


def test_archive_preserves_long_prompt_flat_response_and_full_source(tmp_path):
    item = {
        "prompt_text": "p" * 30_000 + "MANDATORY_LAST_INSTRUCTION",
        "response_text": "r" * 230_000 + "ACTUAL_FINAL_ANSWER",
        "reasoning_text": "reason",
        "is_correct": False,
        "reward": 0,
        "judge_text": "GRADE",
    }
    source = SOURCE + "\n#" + "s" * 6000 + "SOURCE_END"
    snapshot = make_snapshot(item, item_id="id", source=source)
    review = EvidenceReview(snapshot, tmp_path / "review")
    persisted = json.loads((review.directory / "evidence.json").read_text())
    for field in ("prompt_text", "response_text", "reasoning_text"):
        assert persisted["fields"][field] == item[field]
        assert _read_all(review, field) == item[field]
    assert _read_all(review, "detector_source") == source
    assert "GRADE" not in json.dumps(persisted["fields"])
    assert "reward" not in persisted["fields"]
    assert len(review.initial_context(1000)) < 5000
    assert "complete_fields" not in json.loads(review.initial_context(1000))


def test_archive_preserves_dialogue_tools_and_limitations_but_filters_envelope_outcomes():
    steps = [
        {"source": "user", "message": "FOLLOWUP_INSTRUCTION"},
        {
            "source": "assistant",
            "message": "reward=0 is valid tool input",
            "tool_calls": [{"name": "tool", "arguments": {"reward": 0}}],
        },
    ]
    envelope = {
        "steps": steps,
        "agent": {"tool_definitions": [{"name": "TOOL_SCHEMA"}]},
        "evidence_scope": {"limitations": {"encrypted reasoning unavailable": 1}},
        "reward": 0,
        "final_metrics": {"accuracy": 0},
        "metadata": {"resolved": False, "source": "original"},
    }
    snapshot = make_snapshot({"response_text": json.dumps(envelope)}, item_id="id", source=SOURCE)
    saved = json.loads(snapshot["fields"]["response_text"])
    assert saved["steps"] == steps
    assert saved["agent"] == envelope["agent"]
    assert snapshot["source_limitations"] == [envelope["evidence_scope"]]
    assert "reward" not in saved and "final_metrics" not in saved
    assert saved["metadata"] == {"source": "original"}
    assert "response_text.reward" in snapshot["redacted_metadata"]


def test_snapshot_identity_tracks_evidence_source_and_compact_format(monkeypatch):
    a = make_snapshot({"response_text": "a"}, item_id="id", source=SOURCE)
    assert a == make_snapshot({"response_text": "a"}, item_id="id", source=SOURCE)
    assert (
        a["snapshot_id"]
        != make_snapshot({"response_text": "b"}, item_id="id", source=SOURCE)["snapshot_id"]
    )
    assert (
        a["snapshot_id"]
        != make_snapshot({"response_text": "a"}, item_id="id", source=SOURCE + "\n")["snapshot_id"]
    )
    projection_hash = judge_evidence.evidence_projection_hash()
    monkeypatch.setattr(
        evidence_refs, "EVIDENCE_REFS_VERSION", evidence_refs.EVIDENCE_REFS_VERSION + 1,
    )
    assert make_snapshot({"response_text": "a"}, item_id="id", source=SOURCE)["snapshot_id"] != (
        a["snapshot_id"])
    assert judge_evidence.evidence_projection_hash() != projection_hash


def test_unicode_reads_and_paged_search_recover_original_offsets(tmp_path):
    text = "理由 " + ("start GAUSS end " * 30_000) + "真实结尾"
    review = _review(tmp_path, response_text=text)
    first = review.read("response_text", limit=MAX_READ_CHARS * 10)
    assert len(first["text"]) == MAX_READ_CHARS
    assert _read_all(review, "response_text") == text
    found = review.search("response_text", "gauss", limit=2)
    assert len(found["matches"]) == 2 and found["next_offset"] is not None
    for match in found["matches"]:
        assert text[match["match_start"] : match["match_end"]] == "GAUSS"
    later = review.search("response_text", "gauss", found["next_offset"], limit=2)
    assert later["matches"][0]["match_start"] > found["matches"][-1]["match_start"]
    end = review.read("response_text", len(text))
    assert end["is_final"] and end["text"] == ""
    with pytest.raises(ValueError):
        review.read("response_text", len(text) + 1)


def test_small_inputs_are_inlined_completely_and_count_as_read(tmp_path):
    review = _review(tmp_path, prompt_text="Task", response_text="Answer")
    context = json.loads(review.initial_context(5000))
    assert context["complete_fields"]["response_text"]["text"] == "Answer"
    assert review.covers("response_text", 0, 6)
    assert review.covers("detector_source", 0, len(SOURCE))


def test_source_inlining_respects_the_serialized_evidence_budget(tmp_path):
    source = SOURCE + '\n# "Quoted detail"\n' * 120
    review = _review(tmp_path, source=source, response_text="x" * 50_000)
    budget = len(json.dumps({"detector_source": source}, ensure_ascii=False, sort_keys=True))
    assert "complete_fields" not in json.loads(review.initial_context(budget - 1))
    assert review.reads == []
    context = json.loads(review.initial_context(budget))
    assert set(context["complete_fields"]) == {"detector_source"}
    delivered = context["complete_fields"]["detector_source"]
    assert (delivered["text"], delivered["start"], delivered["end"], delivered["next_offset"]) == (
        source, 0, len(source), None,
    )
    assert review.reads == [
        {"field": "detector_source", "start": 0, "end": len(source), "epoch": 0},
    ]
    assert not review.covers("response_text", 0, 1)


def test_large_source_still_requires_complete_paged_reads(tmp_path):
    source = SOURCE + "\n#" + "s" * (MAX_READ_CHARS + 1000)
    review = _review(tmp_path, source=source)
    assert "complete_fields" not in json.loads(review.initial_context(50_000))
    review.read("detector_source")
    passage = review.read("response_text")
    verdict = _row(review, [{"field": "response_text", "start": 0, "end": len(passage["text"]),
                             "quote": passage["text"]}])
    assert review.validate(verdict) == "detector source was not fully read"
    review.read("detector_source", offset=MAX_READ_CHARS)
    assert review.validate(verdict) is None


def test_inlined_source_does_not_refresh_citations_after_compaction(tmp_path):
    review = _review(tmp_path, response_text="I guessed." + "x" * 50_000)
    context = json.loads(review.initial_context(5000))
    assert set(context["complete_fields"]) == {"detector_source"}
    passage = review.read("response_text", limit=10)
    verdict = _row(review, [{"field": "response_text", "start": 0, "end": 10,
                             "quote": passage["text"]}])
    assert review.validate(verdict) is None
    review.compacted({"trigger": "auto"})
    assert not review.covers("detector_source", 0, len(SOURCE), current=True)
    assert "latest compaction" in review.validate(verdict)
    # Redelivering code updates only its own receipt, never cited trace evidence.
    review.initial_context(5000)
    assert review.covers("detector_source", 0, len(SOURCE), current=True)
    assert "latest compaction" in review.validate(verdict)
    review.read("response_text", limit=10)
    assert review.validate(verdict) is None


def test_compaction_preserves_notes_but_requires_fresh_cited_passages(tmp_path):
    review = _review(tmp_path)
    verdict = _verdict(review)
    assert review.validate(verdict) is None
    review.save_notes("Check whether this early guess was corrected. response_text:0-10")
    review.compacted({"trigger": "auto"})
    assert "corrected" in review.status()["notes"]
    assert "latest compaction" in review.validate(verdict)
    assert review.covers("response_text", 0, len(review.fields["response_text"]))
    review.read("response_text")
    assert review.validate(verdict) is None


@pytest.mark.parametrize(
    "change",
    ["quote", "offset", "snapshot", "unread", "scope", "item_id", "field_type"],
)
def test_invalid_evidence_reference_cannot_produce_a_confirmed_verdict(tmp_path, change):
    review = _review(tmp_path, claim_contract=_test_contract())
    verdict = _verdict(review)
    if change == "quote":
        verdict["citations"][0]["quote"] = "invented"
    elif change == "offset":
        verdict["citations"][0]["end"] = 9999
    elif change == "snapshot":
        verdict["snapshot_id"] = "different"
    elif change == "unread":
        review.reads = [r for r in review.reads if r["field"] != "response_text"]
    elif change == "item_id":
        verdict.pop("item_id")
    elif change == "field_type":
        verdict["citations"][0]["field"] = {}
    else:
        verdict["scope"] = "guess"
    result = _parse(review, {"verdicts": [verdict]})
    assert result["n_yes"] == 0 and result["n_errors"] == 1
    assert result["verdicts"][0]["validation_error"]


def _exhaustive_review(tmp_path, *, recovered=False):
    description = "The shell hangs and never regains a prompt before the run ends."
    contract = build_claim_contract(
        description=description, mechanism="", source=SOURCE, defining_clauses=[
            {"id": "hang", "source_field": "description", "quote": "The shell hangs",
             "predicate": "A foreground command stalls the shell.",
             "required_review_scope": "local"},
            {"id": "no_recovery", "source_field": "description",
             "quote": "never regains a prompt before the run ends",
             "predicate": "There is no recovered shell prompt at any later point in the run.",
             "required_review_scope": "whole_trace"},
        ],
    )
    ending = ("The prompt returns after Ctrl-C; the next command writes solution.txt."
              if recovered else "Still waiting at the last recorded event.")
    review = _review(
        tmp_path, claim_contract=contract, prompt_text="Create solution.txt.",
        response_text="apt-get | tail stalls\n" + "unread intervening events\n" * 2000 + ending,
        final_message=("task_complete: true" if recovered
                       else "Send Ctrl-C and try another command. task_complete: false"),
    )
    verdict = _verdict(review)
    review.search("response_text", "prompt")
    review.read("response_text", len(review.fields["response_text"]) - 100, 100)
    review.read("final_message")
    return review, verdict


def _read_stored_fields(review):
    for field in review.fields:
        if review.snapshot["manifest"][field]["origin"] == "stored":
            _read_all(review, field)


def _parse_review(review, verdict):
    return _parse(review, {"verdicts": [verdict]}, source=SOURCE)


@pytest.mark.parametrize("scope,missing_ending", [("local", False), ("whole_trace", True)])
def test_exhaustive_support_cannot_bypass_coverage_by_scope_or_another_unknown(
    tmp_path, scope, missing_ending,
):
    review, verdict = _exhaustive_review(tmp_path)
    verdict["scope"] = scope
    if missing_ending:
        _assessment(verdict, "failure_contribution").update(
            status="unknown", rationale="EOF does not prove task termination.")
    result = _parse_review(review, verdict)
    assert result["n_yes"] == result["n_uncertain"] == 0
    assert result["n_errors"] == 1
    error = result["verdicts"][0]["validation_error"]
    assert ("require scope=whole_trace" if scope == "local" else "unread content") in error
    assert result["behavior_purity"] is None


@pytest.mark.parametrize("status,expected,scope", [
    ("contradicted", "no", "local"), ("unknown", "uncertain", "whole_trace"),
])
def test_exhaustive_clause_allows_local_counterexample_or_honest_abstention(
    tmp_path, status, expected, scope,
):
    review, verdict = _exhaustive_review(tmp_path, recovered=status == "contradicted")
    verdict["scope"] = scope
    assessment = _assessment(verdict, "no_recovery")
    assessment["status"] = status
    if status == "contradicted":
        citation = review.cite("response_text", "The prompt returns after Ctrl-C")
        assessment["citations"] = citation["citations"]
    result = _parse_review(review, verdict)
    assert result["verdicts"][0]["verdict"] == expected
    assert result["verdicts"][0]["validation_error"] is None
    # The unreviewed middle remains unreviewed; a negative/unknown is not a claim
    # to have established semantic absence across it.
    assert not review.covers("response_text", 0, len(review.fields["response_text"]))


def test_counterexample_does_not_waive_separate_supported_exhaustive_claim(tmp_path):
    review, verdict = _exhaustive_review(tmp_path)
    verdict["scope"] = "whole_trace"
    _assessment(verdict, "hang")["status"] = "contradicted"
    result = _parse_review(review, verdict)
    assert result["n_no"] == 0
    assert result["n_errors"] == 1
    assert "unread content" in result["verdicts"][0]["validation_error"]


def test_counterexample_still_requires_valid_read_citations(tmp_path):
    review, verdict = _exhaustive_review(tmp_path, recovered=True)
    verdict["scope"] = "whole_trace"
    citation = review.cite("response_text", "The prompt returns after Ctrl-C")["citations"][0]
    _assessment(verdict, "no_recovery").update(
        status="contradicted", citations=[{**citation, "quote": "invented"}])
    result = _parse_review(review, verdict)
    assert result["n_no"] == 0
    assert result["n_errors"] == 1


def test_whole_trace_completion_uses_original_fields_not_derived_duplicates(tmp_path):
    review, verdict = _exhaustive_review(tmp_path)
    assessment = _assessment(verdict, "no_recovery")
    assessment.update(status="unknown", unknown_reason="review_incomplete")
    pending = review.completion_plan([assessment])
    assert pending[0]["unread_ranges"]
    assert all(review.snapshot["manifest"][span["field"]]["origin"] == "stored"
               for span in pending[0]["unread_ranges"])

    _read_stored_fields(review)
    assert any(not review.covers(field, 0, len(value))
               for field, value in review.fields.items()
               if review.snapshot["manifest"][field]["origin"] == "derived")
    assert review.completion_plan([assessment]) == []
    # A completed original read can support exhaustive review without forcing
    # another pass through extracted reasoning/input/trajectory copies.
    assessment.update(status="supported")
    assessment.pop("unknown_reason")
    verdict["scope"] = "whole_trace"
    assert _parse_review(review, verdict)["n_yes"] == 1


@pytest.mark.parametrize("inspection,reason,gap", [
    (None, "review_incomplete", "whole-trace verdict has unread or unavailable artifact 0"),
    # A known unavailable artifact is not retried, but still blocks positive absence.
    ({"status": "unavailable", "reason": "File not retained."}, "evidence_unavailable", None),
], ids=["retrievable", "unavailable"])
def test_whole_trace_artifact_gap_is_retried_only_while_retrievable_and_never_waived(
    tmp_path, inspection, reason, gap,
):
    review, verdict = _exhaustive_review(tmp_path)
    review.snapshot.setdefault("artifact_manifest", {})["catalog"] = [
        {"artifact_idx": 0, "name": "chart.png", "mime_type": "image/png"},
    ]
    if inspection:
        review.artifact_inspections[0] = inspection
    _read_stored_fields(review)
    assessment = _assessment(verdict, "no_recovery")
    assessment.update(status="unknown", unknown_reason=reason)
    pending = review.completion_plan([assessment])
    if gap:
        assert pending[0]["unread_ranges"] == [] and pending[0]["artifact_gap"] == gap
    else:
        assert pending == []
    assessment.update(status="supported")
    assessment.pop("unknown_reason")
    verdict["scope"] = "whole_trace"
    result = _parse_review(review, verdict)
    assert result["n_errors"] == 1
    assert "artifact 0" in result["verdicts"][0]["validation_error"]


def test_local_unknown_does_not_require_reading_unrelated_trace(tmp_path):
    review, verdict = _exhaustive_review(tmp_path)
    assessment = _assessment(verdict, "failure_contribution")
    assessment.update(status="unknown", unknown_reason="ambiguous_evidence")
    assessment["evidence_inspection"] = {
        "status": "complete", "ranges": [],
        "reason": "The necessary authoritative result is absent.",
    }
    assert not review.covers("response_text", 0, len(review.fields["response_text"]))
    assert review.completion_plan([assessment]) == []


def test_complete_exhaustive_review_does_not_invent_missing_task_termination(tmp_path):
    review, verdict = _exhaustive_review(tmp_path)
    verdict["scope"] = "whole_trace"
    for field in review.fields:
        _read_all(review, field)
    _assessment(verdict, "failure_contribution").update(
        status="unknown", rationale="The final command has no result.")
    result = _parse_review(review, verdict)
    assert result["verdicts"][0]["verdict"] == "uncertain"
    assert result["n_errors"] == 0


@pytest.mark.parametrize("prior_attempt", ["supported", "unknown", "contradicted"])
def test_direct_submitted_defect_is_local_but_does_not_waive_other_claim_clauses(
    tmp_path, prior_attempt,
):
    description = "After a failed general implementation, submits a lookup of visible examples."
    contract = build_claim_contract(
        description=description, mechanism="", source=SOURCE, defining_clauses=[
            {"id": "prior_attempt", "source_field": "description",
             "quote": "a failed general implementation", "predicate": "A real attempt failed.",
             "required_review_scope": "local"},
            {"id": "lookup", "source_field": "description",
             "quote": "submits a lookup of visible examples",
             "predicate": "The submitted implementation only maps the visible inputs.",
             "required_review_scope": "local"},
        ],
    )
    history = {"supported": "The earlier general implementation crashed.",
               "unknown": "No evidence of the earlier implementation is available.",
               "contradicted": "The first implementation was this lookup."}[prior_attempt]
    review = _review(
        tmp_path, claim_contract=contract, prompt_text="Implement square(n) for every integer.",
        response_text="An unrelated earlier log.\n" * 2000,
        final_message=history + " Submitted: def square(n): return {1: 1, 2: 4}[n]. "
                                "Samples pass. task_complete: true",
    )
    verdict = _verdict(review, field="final_message")
    review.read("prompt_text")
    _assessment(verdict, "prior_attempt")["status"] = prior_attempt
    result = _parse_review(review, verdict)
    assert result["verdicts"][0]["verdict"] == {
        "supported": "yes", "unknown": "uncertain", "contradicted": "no",
    }[prior_attempt]
    assert result["n_errors"] == 0
    assert not review.covers("response_text", 0, len(review.fields["response_text"]))


def _judge(tmp_path, items, **kwargs):
    return claude_judge.judge_behavior_with_claude(
        name="guess", mechanism="Ships an admitted guess", description="A guessed final answer",
        source=SOURCE, items=items, assigned_labels=[True] * len(items), cwd=tmp_path,
        model="test", **kwargs,
    )


def test_each_rollout_has_an_isolated_session_and_verdicts_are_aggregated(tmp_path, monkeypatch):
    calls = []
    items = [{"_item_id": key, "response_text": key} for key in ("FIRST", "SECOND", "THIRD")]

    async def session(*, prompt, cwd, review, **kwargs):
        calls.append(cwd)
        item_id = review.snapshot["item_id"]
        assert item_id in prompt
        assert all(
            other["_item_id"] not in prompt for other in items if other["_item_id"] != item_id
        )
        value = {"FIRST": True, "SECOND": False, "THIRD": None}[item_id]
        verdict = _verdict(review, correct=value)
        return json.dumps(
            {"verdicts": [verdict], "behavior_purity": 1, "behavior_specificity": 0.8}
        )

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", session)
    result = _judge(tmp_path, items)
    assert len(set(calls)) == 4  # Three initial reviews and FIRST's confirmation.
    assert result["n_yes"] == result["n_no"] == result["n_uncertain"] == 1
    assert result["behavior_purity"] == pytest.approx(1 / 3)
    assert result["n_completed"] == 2
    assert result["evidence_version"] == judge_evidence.EVIDENCE_VERSION
    assert result["behavior_specificity"] == pytest.approx(0.8)
    for verdict in result["verdicts"]:
        directory = Path(verdict["review_path"])
        assert (directory / "evidence.json").is_file()
        assert (directory / "coverage.json").is_file()
        assert (directory / "review-verdict.json").is_file()


@pytest.mark.parametrize("correct,value", [
    (True, "missing"), (False, "0.9"), (True, True), (True, float("nan")),
    (False, -0.1), (True, 1.1), (False, 0), (True, 1),
])
def test_specificity_never_uses_code_heuristics_or_changes_literal_verdict(
    tmp_path, correct, value,
):
    review = _review(tmp_path, claim_contract=_test_contract())
    data = {"verdicts": [_verdict(review, correct=correct)]}
    if value != "missing":
        data["behavior_specificity"] = value
    result = _parse(review, data, source=SOURCE)
    assert result["verdicts"][0]["verdict"] == ("yes" if correct else "no")
    assert result["behavior_purity"] == float(correct)
    expected = float(value) if type(value) in (int, float) and 0 <= value <= 1 else None
    assert result["behavior_specificity"] == expected


@pytest.mark.parametrize("raw", ["not JSON", "{broken JSON}", "{}"])
def test_unusable_judge_reply_has_no_specificity_fallback(tmp_path, raw):
    review = _review(tmp_path, claim_contract=_test_contract())
    result = _parse(review, raw, source=SOURCE)
    assert result["behavior_specificity"] is None
    assert result["n_yes"] == 0


def test_one_failure_is_not_retried_or_replaced_by_a_shortened_judge(tmp_path, monkeypatch):
    called = []

    async def session(*, review, **kwargs):
        item_id = review.snapshot["item_id"]
        called.append(item_id)
        if item_id == "a":
            raise RuntimeError("context budget exceeded")
        return json.dumps({"verdicts": [_verdict(review)]})

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", session)
    result = _judge(
        tmp_path,
        [{"_item_id": "a", "response_text": "one"}, {"_item_id": "b", "response_text": "two"}],
        workers=4,
    )
    assert sorted(called) == ["a", "b", "b"]  # No retry for a; b gets confirmation.
    assert result["n_judged"] == 2 and result["n_yes"] == result["n_errors"] == 1
    assert result["n_uncertain"] == 0
    assert "context budget exceeded" in result["verdicts"][0]["rationale"]


def test_nested_event_loop_and_immutable_review_history(tmp_path, monkeypatch):
    async def session(*, review, **kwargs):
        return json.dumps({"verdicts": [_verdict(review)]})

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", session)

    async def invoke():
        return _judge(tmp_path, [{"_item_id": "a", "response_text": "original"}])

    first = asyncio.run(invoke())
    saved_path = Path(first["verdicts"][0]["review_path"]) / "evidence.json"
    original = saved_path.read_bytes()
    second = _judge(tmp_path, [{"_item_id": "a", "response_text": "changed"}])
    assert first["verdicts"][0]["review_path"] != second["verdicts"][0]["review_path"]
    assert saved_path.read_bytes() == original


@pytest.mark.parametrize("reply", ["invalid JSON", "uncited", "explicit_null"])
def test_unfinished_reviews_have_no_scores_but_uncertain_reviews_keep_advice(
    tmp_path, monkeypatch, reply,
):
    calibration = {"status": "mixed", "claim_note": "The causal premise is unestablished.",
                   "rewrite_hint": "Inspect the missing earlier reasoning before revising."}
    async def session(*, review, **kwargs):
        if reply == "invalid JSON":
            return reply
        verdict = _verdict(review, correct=None if reply == "explicit_null" else True)
        if reply == "uncited":
            verdict["claim_assessments"][0]["citations"] = []
        else:
            # An explicit abstention must not be overridden by an old output alias.
            verdict.update(correct=None, present=True)
            for assessment in verdict["claim_assessments"]:
                assessment["evidence_inspection"] = {
                    "status": "complete", "ranges": [],
                    "reason": "The inspected answer does not establish the causal premise.",
                }
        return json.dumps(
            {
                "verdicts": [verdict],
                "behavior_specificity": 1,
                "claim_calibration": calibration,
            }
        )

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", session)
    result = _judge(tmp_path, [{"response_text": "answer"}])
    assert result["n_completed"] == 0
    assert result["n_uncertain"] == int(reply == "explicit_null")
    assert result["n_errors"] == int(reply != "explicit_null")
    assert result["behavior_purity"] is result["behavior_specificity"] is None
    if reply == "explicit_null":
        assert result["claim_calibration"]["rewrite_hint"] == calibration["rewrite_hint"]
        assert "missing earlier reasoning" in result["calibration_hint"]
        assert result["item_calibrations"][0]["claim_calibration"] == result["claim_calibration"]
    else:
        assert result["claim_calibration"] is result["calibration_hint"] is None
        assert result["item_calibrations"] == []


def test_mismatched_labels_and_duplicate_ids_fail_before_any_review(tmp_path):
    with pytest.raises(ValueError, match="one assigned label"):
        claude_judge.judge_behavior_with_claude(
            name="n", mechanism="m", description="", source=SOURCE, items=[{}],
            assigned_labels=[], cwd=tmp_path, model="test",
        )
    with pytest.raises(ValueError, match="duplicate judge"):
        _judge(tmp_path, [{"_item_id": "same"}, {"_item_id": "same"}])


@pytest.fixture
def judge_sdk_tools(monkeypatch):
    import claude_agent_sdk as sdk

    from autofa.harness import claude_cli

    monkeypatch.setattr(claude_cli, "ensure_claude_auth", lambda **kwargs: None)
    captured = {}
    create = sdk.create_sdk_mcp_server

    def capture(**kwargs):
        captured.update({t.name: t.handler for t in kwargs["tools"]})
        return create(**kwargs)

    monkeypatch.setattr(sdk, "create_sdk_mcp_server", capture)
    return sdk, captured


@pytest.mark.parametrize("stop,effort", [("end_turn", "medium"), ("max_tokens", None)])
def test_sdk_tools_hooks_compaction_and_termination_are_observed(
    tmp_path, monkeypatch, judge_sdk_tools, stop, effort,
):
    sdk, captured_tools = judge_sdk_tools
    review = _review(tmp_path)

    async def query(*, prompt, options):
        prompts = [message async for message in prompt]
        assert prompts[0]["message"]["content"] == "review this"
        assert options.tools == [] and options.setting_sources == []
        expected_tools = {
            "read_evidence", "search_evidence", "cite_evidence", "interval_events",
            "retrieve_supporting_passages",
            "review_interval", "list_artifacts", "prepare_artifact_support",
            "inspect_artifact", "read_artifact", "view_artifact", "transcribe_artifact",
            "cite_artifact", "review_status", "save_review_notes",
        }
        assert options.strict_mcp_config
        assert set(captured_tools) == expected_tools
        assert set(options.allowed_tools) == {
            f"mcp__judge_evidence__{name}" for name in expected_tools
        }
        assert options.max_turns == 64
        assert options.model == "test" and options.effort == effort
        assert options.thinking == ({"type": "adaptive"} if effort else None)
        yield sdk.SystemMessage(subtype="init", data={"session_id": "test-session"})
        handler = captured_tools["read_evidence"]
        response = await handler({"field": "response_text"})
        assert json.loads(response["content"][0]["text"])["text"] == "I guessed."
        citation = await captured_tools["cite_evidence"](
            {"field": "response_text", "quote": "guessed"}
        )
        issued = json.loads(citation["content"][0]["text"])["citations"][0]
        assert issued["quote"] == "guessed" and issued["start"] == 2 and issued["end"] == 9
        invalid = await handler({"field": "reward"})
        assert invalid["isError"] is True
        await captured_tools["save_review_notes"]({"notes": "check for recovery"})
        hook = options.hooks["PreCompact"][0].hooks[0]
        await hook({"trigger": "auto", "transcript_path": "session.jsonl"}, None, {})
        yield sdk.SystemMessage(subtype="compact_boundary", data={"trigger": "auto"})
        status = await captured_tools["review_status"]({})
        assert json.loads(status["content"][0]["text"])["epoch"] == 1
        yield sdk.ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=3,
            session_id="test-session", stop_reason=stop, total_cost_usd=0.01,
            result='{"verdicts": []}',
        )

    monkeypatch.setattr(sdk, "query", query)
    # One request shows the session protocol; repairing the uncompiled reply is covered elsewhere.
    monkeypatch.setattr(claude_judge, "MAX_VERDICT_REPAIRS", 0)
    kwargs = {"prompt": "review this", "cwd": review.directory, "model": "test",
              "cli_path": None, "review": review, "effort": effort}
    if stop == "end_turn":
        assert asyncio.run(claude_judge._run_claude_judge_session(**kwargs)) == '{"verdicts": []}'
    else:
        with pytest.raises(RuntimeError, match="incomplete judge"):
            asyncio.run(claude_judge._run_claude_judge_session(**kwargs))
    assert review.session["session_id"] == "test-session" and review.epoch == 1
    assert review.session["judge_model"] == "test"
    assert review.session["judge_effort"] == effort
    assert review.session["judge_thinking"] == ("adaptive" if effort else None)
    events = [
        json.loads(line)["event"]
        for line in (review.directory / "audit.jsonl").read_text().splitlines()
    ]
    assert "pre_compact" in events and "session_result" in events


def test_tool_citations_avoid_manual_offsets_in_long_unicode_traces(tmp_path):
    quote = "I'm running out of time. I'll write a solution."
    prefix = "理由 🧪\n" + "x" * 188_539
    review = _review(tmp_path, claim_contract=_test_contract(),
                     reasoning_text=prefix + quote + "\nActual ending.")
    _read_all(review, "detector_source")
    review.read("reasoning_text", len(prefix) - 50, 200)
    citation = review.cite("reasoning_text", quote)["citations"][0]
    assert citation["start"] == len(prefix)
    assert citation["end"] == len(prefix) + len(quote)
    row = _row(review, [{"citation_id": citation["citation_id"]}], correct=False,
               rationale="Checked context and ending.")
    row.update(_claim_fields(review.snapshot["claim_contract"], row["citations"], False))
    result = _parse(review, {"verdicts": [row]})
    assert result["verdicts"][0]["verdict"] == "no"
    assert result["verdicts"][0]["citations"] == [citation]
    assert result["behavior_purity"] == 0


def test_repeated_quotes_have_distinct_paged_references_and_require_no_guessing(tmp_path):
    review = _review(tmp_path, response_text="guess then correct; guess then ship; guess")
    _read_all(review, "response_text")
    first = review.cite("response_text", "guess", limit=1)
    second = review.cite("response_text", "guess", offset=first["next_offset"], limit=1)
    third = review.cite("response_text", "guess", offset=second["next_offset"], limit=1)
    citations = [part["citations"][0] for part in (first, second, third)]
    assert [c["start"] for c in citations] == [0, 20, 37]
    assert len({c["citation_id"] for c in citations}) == 3
    assert third["next_offset"] is None
    longer = review.cite("response_text", "guess then ship")["citations"]
    assert len(longer) == 1 and longer[0]["start"] == 20


def test_citation_tool_never_reads_unseen_or_compacted_evidence(tmp_path):
    review = _review(tmp_path, response_text="read hidden read")
    review.read("response_text", 0, 4)
    review.read("response_text", 12, 4)
    before = list(review.reads)
    assert len(review.cite("response_text", "read")["citations"]) == 2
    for quote in ("hidden", "read hidden read", "invented", "READ"):
        with pytest.raises(ValueError, match="exact quote not found"):
            review.cite("response_text", quote)
    assert review.reads == before
    review.compacted({})
    with pytest.raises(ValueError, match="latest compaction"):
        review.cite("response_text", "read")
    review.read("response_text", 12, 4)
    assert review.cite("response_text", "read")["citations"][0]["start"] == 12
    for field in ("detector_source", "reward", {}):
        with pytest.raises(ValueError, match="evidence field"):
            review.cite(field, "read")


@pytest.mark.parametrize("change", ["unknown", "conflict", "whole_trace"])
def test_citation_ids_preserve_evidence_validation(tmp_path, change):
    review = _review(tmp_path, claim_contract=_test_contract(),
                     response_text="I guessed. Later correction.", prompt_text="Task")
    _read_all(review, "detector_source")
    review.read("response_text", 0, 10)
    citation = review.cite("response_text", "guessed")["citations"][0]
    row = _row(review, [{"citation_id": citation["citation_id"]}])
    row.update(_claim_fields(review.snapshot["claim_contract"], row["citations"]))
    if change == "unknown":
        row["citations"][0]["citation_id"] = "invented"
    elif change == "conflict":
        row["citations"][0]["quote"] = "different"
    else:
        row["scope"] = "whole_trace"
    parsed = _parse(review, {"verdicts": [row]})
    assert parsed["n_errors"] == 1 and parsed["n_yes"] == 0
    assert parsed["verdicts"][0]["validation_error"]


def test_search_issues_exact_case_preserving_citations(tmp_path):
    review = _review(tmp_path, response_text="I GUESSED, then verified.")
    match = review.search("response_text", "guessed")["matches"][0]
    assert match["citation"]["quote"] == "GUESSED"
    resolved = review.resolve_citations(
        {"citations": [{"citation_id": match["citation"]["citation_id"]}]}
    )
    assert resolved["citations"] == [match["citation"]]


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_long_trace_review_accepts_first_reply_with_inlined_source(
    tmp_path, monkeypatch, judge_sdk_tools, harness,
):
    from autofa.metrics import codex_judge

    sdk, tools = judge_sdk_tools
    source = SOURCE + "\n#" + "source detail " * 150
    quote = "I guessed."
    trace = "x" * 190_000 + quote
    contract = _test_contract(source)
    calls = []
    reviews = []
    original_specs = claude_judge._evidence_tool_specs

    def specs(review):
        reviews.append(review)
        return original_specs(review)

    monkeypatch.setattr(claude_judge, "_evidence_tool_specs", specs)

    async def reply(prompt, call_tool, resume):
        assert resume is None  # Neither stage should need a missing-source repair.
        calls.append(prompt)
        context, _ = json.JSONDecoder().raw_decode(prompt.split("TRACE:\n", 1)[1])
        assert set(context["complete_fields"]) == {"detector_source"}
        assert context["complete_fields"]["detector_source"]["text"] == source
        assert trace not in prompt and len(prompt) < 40_000
        # Only trace evidence needs a tool call; the required source was delivered.
        passage = await call_tool("read_evidence", {
            "field": "response_text", "offset": 190_000,
        })
        citation = (await call_tool("cite_evidence", {
            "field": "response_text", "quote": quote,
        }))["citations"][0]
        references = [{"citation_id": citation["citation_id"]}]
        return json.dumps({"verdicts": [{
            "item_id": "rollout", "snapshot_id": passage["snapshot_id"],
            "correct": True, "scope": "local", "citations": references,
            **_claim_fields(contract, references),
        }]})

    async def query(*, prompt, options):
        messages = [message async for message in prompt]

        async def call_tool(name, args):
            result = await tools[name](args)
            assert not result.get("isError")
            return json.loads(result["content"][0]["text"])

        result = await reply(messages[0]["message"]["content"], call_tool, options.resume)
        yield sdk.ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=1, session_id=f"stage-{len(calls)}", stop_reason="end_turn",
            result=result,
        )

    async def codex_turn(**kwargs):
        async def call_tool(name, args):
            result = await kwargs["call_tool"](name, args)
            assert result["success"]
            return json.loads(result["contentItems"][0]["text"])

        result = await reply(kwargs["prompt"], call_tool, kwargs.get("resume"))
        return SimpleNamespace(
            text=result, thread_id=f"stage-{len(calls)}", turn_id="turn",
            status="completed", usage={}, error=None, error_code=None, retryable=False,
        )

    monkeypatch.setattr(sdk, "query", query)
    monkeypatch.setattr(codex_judge, "_run_turn", codex_turn)
    result = claude_judge.judge_behavior_with_claude(
        name="guess", description="A guessed final answer", mechanism="Ships an admitted guess",
        source=source, items=[{"_item_id": "rollout", "response_text": trace}],
        assigned_labels=[True], cwd=tmp_path, model="test", compiled_contract=contract,
        harness=harness, max_chars=5000,
    )
    assert result["n_yes"] == 1 and result["n_errors"] == 0
    assert len(calls) == 2  # Initial review and independent verification.
    assert [session["stage"] for session in result["sessions"]] == ["initial", "verification"]
    for review in reviews:
        assert [read for read in review.reads if read["field"] == "detector_source"] == [
            {"field": "detector_source", "start": 0, "end": len(source), "epoch": 0},
        ]
        assert not (review.directory / "repair-1.txt").exists()


@pytest.mark.parametrize("citation_error", ["offset", "detector_source"])
def test_invalid_citations_are_repaired_in_same_session_and_archived(
    tmp_path, monkeypatch, judge_sdk_tools, citation_error,
):
    sdk, tools = judge_sdk_tools
    calls = []
    quote = "Given the time, I'll implement this greedy algorithm"
    text = "理由 " + "x" * 190_000 + quote + "."
    issued = None
    session_id = None

    async def call_tool(name, **args):
        result = await tools[name](args)
        assert not result.get("isError")
        return json.loads(result["content"][0]["text"])

    async def query(*, prompt, options):
        nonlocal issued, session_id
        messages = [m async for m in prompt]
        calls.append(options.resume)
        if options.resume is None:
            assert options.max_turns == 64 and options.resume is None
            session_id = "original-session" if len(calls) == 1 else "confirmation-session"
            await call_tool("read_evidence", field="detector_source")
            passage = await call_tool("read_evidence", field="response_text", offset=190_000)
            citation = {"field": "response_text", "start": 190_016, "end": 190_050, "quote": quote}
            if citation_error == "detector_source":
                citation = {"field": "detector_source", "start": 0, "end": 6, "quote": SOURCE[:6]}
        else:
            assert len(calls) in (2, 4) and options.resume == session_id
            assert options.max_turns == 60
            error = (
                "citation quote does not match source" if citation_error == "offset"
                else "invalid citation field or offsets"
            )
            assert error in messages[0]["message"]["content"]
            passage = await call_tool("review_status")
            # A compaction during repair still requires rereading the cited evidence.
            yield sdk.SystemMessage(subtype="compact_boundary", data={"trigger": "auto"})
            await call_tool("read_evidence", field="response_text", offset=190_000)
            issued = (await call_tool("cite_evidence", field="response_text", quote=quote))[
                "citations"
            ][0]
            citation = {"citation_id": issued["citation_id"]}
        row = {
            "item_id": "rollout", "snapshot_id": passage["snapshot_id"],
            "correct": True, "scope": "local", "citations": [citation],
            **_claim_fields(_test_contract(), [citation]),
        }
        yield sdk.ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=4, session_id=session_id, stop_reason="end_turn",
            total_cost_usd=0.01,
            result=json.dumps({"verdicts": [row], "behavior_specificity": 0.8}),
        )

    monkeypatch.setattr(sdk, "query", query)
    result = _judge(tmp_path, [{"_item_id": "rollout", "response_text": text}])
    assert calls == [None, "original-session", None, "confirmation-session"]
    assert result["n_completed"] == 1 and result["behavior_purity"] == 1
    assert result["behavior_specificity"] == 0.8
    assert result["verdicts"][0]["citations"] == [issued]
    assert [s["stage"] for s in result["sessions"]] == ["initial", "verification"]
    for session in result["sessions"]:
        assert session["compactions"] == 1 and len(session["attempts"]) == 2
        path = Path(session["review_path"])
        assert (path / "reply-attempt-0.txt").is_file()
        assert (path / "reply-attempt-1.txt").read_text() == (path / "reply.txt").read_text()
        assert "cite_evidence" in (path / "repair-1.txt").read_text()
        archived = json.loads((path / "review-verdict.json").read_text())
        assert archived["citations"][0]["quote"] == quote


@pytest.mark.parametrize("mode,expected_calls", [("invalid", 3), ("abstain", 1), ("sdk_error", 1)])
def test_repairs_are_bounded_and_do_not_force_scores(
    tmp_path, monkeypatch, judge_sdk_tools, mode, expected_calls,
):
    sdk, tools = judge_sdk_tools
    calls = []

    async def query(*, prompt, options):
        _ = [m async for m in prompt]
        calls.append(options.resume)
        if len(calls) > 1:
            assert options.resume == "original-session"
        row = {"item_id": "rollout", "correct": None if mode == "abstain" else True}
        if mode == "abstain":
            await tools["read_evidence"]({"field": "detector_source"})
            evidence = await tools["read_evidence"]({"field": "response_text"})
            snapshot = json.loads(evidence["content"][0]["text"])["snapshot_id"]
            row.update(snapshot_id=snapshot, scope="local")
            row.update(_claim_fields(_test_contract(), [], None))
            for assessment in row["claim_assessments"]:
                assessment["evidence_inspection"] = {
                    "status": "complete", "ranges": [],
                    "reason": "The inspected answer remains ambiguous.",
                }
        yield sdk.ResultMessage(
            subtype="error_during_execution" if mode == "sdk_error" else "success",
            duration_ms=1, duration_api_ms=1, is_error=mode == "sdk_error",
            num_turns=4, session_id="original-session", stop_reason="end_turn",
            result=json.dumps({"verdicts": [row]}),
        )

    monkeypatch.setattr(sdk, "query", query)
    result = _judge(tmp_path, [{"_item_id": "rollout", "response_text": "answer"}])
    assert len(calls) == expected_calls
    assert result["n_completed"] == 0
    assert result["n_uncertain"] == int(mode == "abstain")
    assert result["n_errors"] == int(mode != "abstain")
    assert result["behavior_purity"] is result["behavior_specificity"] is None


# Lossless source sharing reduces review text without weakening read coverage.

BODY = "UNIQUE_MESSAGE_BODY " + "Complete original text. " * 200
ARGUMENT = "UNIQUE_ARGUMENT_VALUE " + "Full argument. " * 100
AUXILIARY = "UNIQUE_AUXILIARY_CORRECTION"
REPEATED = "Repeated finding"


def _trace():
    part = {"part_idx": 0, "part_type": "text", "text": BODY,
            "chars": len(BODY), "lane": ""}
    event = {"event_idx": 7, "lane": "", "kind": "message", "role": "assistant",
             "parts": [part], "n_parts": 1, "provider_extension": {"state": "recorded"}}
    call = {"event_idx": 7, "lane": "", "call_idx": 0, "tool_name": "test",
            "args_full": json.dumps({"value": ARGUMENT}), "status": "completed"}
    step = {
        "step_id": 7, "source": "assistant", "message": BODY,
        "metadata": {key: value for key, value in event.items() if key != "parts"},
        "parts": [part],
        "source_parts": [{"part_idx": 0, "start": 0, "end": len(BODY), "field": "message"}],
        "tool_calls": [{**call, "function_name": "test", "arguments": call["args_full"],
                        "source_call_index": 0}],
    }
    return {
        "steps": [step],
        "evidence_scope": {"source": "original API event parts and tool arguments",
                           "event_lane": "", "limitations": {}},
        "execution": {"events": [event, {
            "event_idx": 7, "lane": "worker", "kind": "message", "role": "assistant",
            "parts": [{"part_idx": 0, "part_type": "text", "text": AUXILIARY}],
        }], "tool_calls": [call], "extra_resources": {"novel_records": [{"value": "retained"}]}},
    }


@pytest.mark.parametrize("encoded,wrapped,already_compact", [
    (False, False, False), (True, True, False), (False, False, True),
], ids=["object", "encoded-wrapped", "already-compact"])
def test_snapshot_shares_only_mirrors_and_preserves_source_records(
    encoded, wrapped, already_compact,
):
    original = _trace()
    value = (compact_trace(original, include_rendered=True) if already_compact
             else deepcopy(original))
    if wrapped:
        value = [value]
    if encoded:
        value = json.dumps(value)
    before = deepcopy(value)
    snapshot = _response_snapshot(value)
    text = snapshot["fields"]["response_text"]
    saved = json.loads(text)
    assert expand_trace(saved[0] if wrapped else saved) == original
    assert text.count("UNIQUE_MESSAGE_BODY") == 1
    assert text.count("UNIQUE_ARGUMENT_VALUE") == 1
    assert AUXILIARY in text and "novel_records" in text
    assert snapshot["manifest"]["response_text"]["origin"] == "stored"
    assert value == before
    # The readable trajectory does not expand source mirrors either.
    rendered = snapshot["fields"]["trajectory_text"]
    assert rendered.count("UNIQUE_MESSAGE_BODY") == 1
    assert rendered.count("UNIQUE_ARGUMENT_VALUE") == 1
    assert "source_references" in rendered and '"event_idx": 7' in rendered
    assert "source=assistant" in rendered and "whole-trace read" in rendered


def test_whole_trace_still_requires_unique_stored_content(tmp_path):
    review = _review(tmp_path, response_text=_trace())
    _read_all(review, "detector_source")
    _read_all(review, "trajectory_text")
    citations = review.cite("trajectory_text", "UNIQUE_MESSAGE_BODY")["citations"]
    verdict = _row(review, citations, scope="whole_trace")
    assert review.validate(verdict) == "whole-trace verdict has unread content in response_text"

    text = review.fields["response_text"]
    start = text.index(AUXILIARY)
    review.read("response_text", 0, start)
    review.read("response_text", start + len(AUXILIARY), len(text))
    assert review.validate(verdict) == "whole-trace verdict has unread content in response_text"
    review.read("response_text", start, len(AUXILIARY))
    assert review.validate(verdict) is None


def test_rebased_quote_and_citation_retain_source_occurrence(tmp_path):
    original = json.dumps(_trace())
    quote = "UNIQUE_MESSAGE_BODY"
    start = original.index(quote)
    selected = {"source_sha256": localisation.text_hash(SOURCE), "passages": [
        localisation.passage("response_text", original, start, start + len(quote),
                             method="literal"),
    ]}
    snapshot = _response_snapshot(original, failure_localisation=selected)
    text = snapshot["fields"]["response_text"]
    rebased, = snapshot["failure_localisation"]["passages"]
    assert text[rebased["start"]:rebased["end"]] == quote
    assert rebased["field_sha256"] == localisation.text_hash(text)
    assert rebased["start"] != start
    location, = TraceNavigation({"response_text": text}).locate(rebased)
    assert location["source_event"] == {"lane": "", "event_idx": 7, "part_idx": 0}

    review = EvidenceReview(snapshot, tmp_path / "review")
    assert not review.covers("response_text", rebased["start"], rebased["end"])
    _read_all(review, "response_text")
    citation, = review.cite("response_text", quote)["citations"]
    assert citation["start"] == rebased["start"]
    assert citation["end"] == rebased["end"]


def _response_snapshot(value, **kwargs):
    return make_snapshot({"response_text": value}, item_id="item", source=SOURCE, **kwargs)


def _repeated_trace(messages):
    original = _trace()
    example_event, example_step = original["execution"]["events"][0], original["steps"][0]
    original["steps"] = []
    original["execution"]["events"] = []
    original["execution"]["tool_calls"] = []
    for index, message in enumerate(messages):
        event, step = deepcopy(example_event), deepcopy(example_step)
        event["event_idx"] = index
        event["parts"][0].update(text=message, chars=len(message))
        step.update(step_id=index, message=message, parts=deepcopy(event["parts"]), tool_calls=[],
                    metadata={key: value for key, value in event.items() if key != "parts"})
        step["source_parts"][0]["end"] = len(message)
        original["steps"].append(step)
        original["execution"]["events"].append(event)
    return original


def _selected_localisation(text, quote):
    start = text.rindex(quote)
    selected = localisation.passage("response_text", text, start, start + len(quote),
                                    method="literal")
    # Archived navigation is advisory and can be stale or forged. Recompute it.
    selected["trace_locations"] = [{"source_event": {"lane": "", "event_idx": 0}}]
    return {"source_sha256": localisation.text_hash(SOURCE), "passages": [selected]}


def _localised_snapshot(text, quote):
    return _response_snapshot(text, failure_localisation=_selected_localisation(text, quote))


@pytest.mark.parametrize("messages,event_idx", [
    ([REPEATED] * 10, 9),  # A late selected event survives before the passage limit.
    ([f"{REPEATED}; then {REPEATED}; finally {REPEATED}"], 0),  # A repeat within one part.
])
def test_compaction_preserves_the_selected_repeated_occurrence(messages, event_idx):
    original = json.dumps(_repeated_trace(messages))
    snapshot = _localised_snapshot(original, REPEATED)
    rebased, = snapshot["failure_localisation"]["passages"]
    location, = TraceNavigation(snapshot["fields"]).locate(rebased)
    assert location["source_event"] == {"lane": "", "event_idx": event_idx, "part_idx": 0}
    start = messages[event_idx].rindex(REPEATED)
    assert (location["start"], location["end"]) == (start, start + len(REPEATED))


def test_rebase_checks_source_offsets_without_scanning_thousands_of_repeats(monkeypatch):
    message = ('🙂 "source" ' + REPEATED + "\n") * 10_000
    original = json.dumps(_repeated_trace([message]))
    locate = judge_evidence._source_position
    calls = 0

    def counted(*args):
        nonlocal calls
        calls += 1
        return locate(*args)

    monkeypatch.setattr(judge_evidence, "_source_position", counted)
    snapshot = _localised_snapshot(original, REPEATED)
    rebased, = snapshot["failure_localisation"]["passages"]
    location, = TraceNavigation(snapshot["fields"]).locate(rebased)
    assert location["start"] == message.rindex(REPEATED)
    assert calls < 10  # Source checks must scale with representations, not repeats.


def test_compaction_preserves_auxiliary_lane_and_repeat_within_part():
    message = f"{REPEATED}; later {REPEATED}"
    original = _repeated_trace([message])
    auxiliary = deepcopy(original["execution"]["events"][0])
    auxiliary["lane"] = auxiliary["parts"][0]["lane"] = "worker"
    original["execution"]["events"].append(auxiliary)
    text = json.dumps(original)
    snapshot = _localised_snapshot(text, REPEATED)
    rebased, = snapshot["failure_localisation"]["passages"]
    location, = TraceNavigation(snapshot["fields"]).locate(rebased)
    assert location["source_event"] == {"lane": "worker", "event_idx": 0, "part_idx": 0}
    # Auxiliary locators do not expose within-part offsets; the selected raw
    # quote must still be its last occurrence, not the earlier identical phrase.
    assert rebased["start"] == snapshot["fields"]["response_text"].rindex(REPEATED)


@pytest.mark.parametrize("invalid", ["hash", "offset"])
def test_compaction_drops_unverified_original_localisation(invalid):
    text = json.dumps(_repeated_trace([REPEATED] * 10))
    selected = _selected_localisation(text, REPEATED)
    if invalid == "hash":
        selected["passages"][0]["field_sha256"] = "stale"
    else:
        selected["passages"][0]["start"] += 1
    snapshot = _response_snapshot(text, failure_localisation=selected)
    assert snapshot["failure_localisation"]["status"] == "unavailable"
    assert snapshot["failure_localisation"]["passages"] == []


def test_compaction_drops_ambiguous_auxiliary_tool_occurrence():
    quote = "Repeated tool argument"
    original = _repeated_trace(["Unrelated conversation"])
    original["execution"]["events"].append({
        "event_idx": 0, "lane": "worker", "kind": "tool_call", "parts": [],
    })
    original["execution"]["tool_calls"] = [
        {"event_idx": 0, "lane": "worker", "call_idx": index,
         "tool_name": "test", "args_full": quote} for index in (0, 1)
    ]
    text = json.dumps(original)
    snapshot = _localised_snapshot(text, quote)
    assert snapshot["failure_localisation"]["passages"] == []


def test_existing_compact_sources_expand_before_outcome_redaction():
    original = _trace()
    original["execution"]["rollout"] = {"reward": 0, "status": "completed"}
    original["reward"] = 1
    compact = compact_trace(original)
    snapshot = _response_snapshot(compact)
    restored = expand_trace(json.loads(snapshot["fields"]["response_text"]))
    assert "reward" not in restored
    assert "reward" not in restored["execution"]["rollout"]
    assert restored["steps"] == original["steps"]


def test_invalid_refs_remain_original_evidence_with_explicit_limitation():
    original = compact_trace(_trace())
    original["_autofa_evidence_refs"]["steps"][0]["event_idx"] = 999
    snapshot = _response_snapshot(original)
    assert json.loads(snapshot["fields"]["response_text"]) == original
    assert snapshot["manifest"]["response_text"]["origin"] == "stored"
    assert "trajectory_text" not in snapshot["fields"]
    assert any("Invalid compact source references" in entry.get("limitation", "")
               for entry in snapshot["source_limitations"])


@pytest.mark.parametrize("value", ["Plain original text", {"steps": [{"message": "legacy"}]},
                                    [{"source": "user", "message": "original dialogue"}]])
def test_unknown_and_legacy_shapes_are_not_rewritten(value):
    snapshot = _response_snapshot(value)
    text = snapshot["fields"]["response_text"]
    assert text == value if isinstance(value, str) else json.loads(text) == value


# Missing actors remain unknown in the judge's complete event view.


@pytest.mark.parametrize("wrapper", [lambda steps: {"steps": steps},
                                     lambda steps: json.dumps({"steps": steps})])
def test_missing_step_actor_does_not_become_assistant(wrapper):
    # JSON inside text or tool arguments does not gain or change roles either.
    payload = '{"messages":[{"content":"keep exactly"}]}'
    trace = wrapper([{"message": payload, "tool_calls": [
        {"name": "malformed工具", "arguments": {"file_text": payload}},
    ]}])
    text = judge_trajectory_text(trace)
    assert "source=unknown" in text and '"source": "unknown"' in text
    assert text.count("source=unknown") == 1 and json.dumps(payload) in text
    assert "assistant" not in text and "malformed工具" in text


@pytest.mark.parametrize("value,actor", [
    ({"steps": [{"source": "assistant", "message": "Full original text."}]}, "assistant"),
    ({"steps": [{"role": "tool", "message": "Full original text."}]}, "tool"),
    ({"choices": [{"message": {"content": "Full original text."}}]}, "assistant"),
    ({"message": "Full original text."}, "unknown"),
    ([{"message": "Full original text."}], "unknown"),
], ids=["step-source", "step-role", "chat-choice", "bare-step", "bare-step-list"])
def test_declared_or_implied_actor_is_kept_and_otherwise_unknown(value, actor):
    text = judge_trajectory_text(value)
    assert f"source={actor}" in text and "Full original text." in text


def test_atif_user_call_and_later_recovery_keep_order_and_never_modify_input():
    trace = [{"schema_version": "ATIF-v1.5", "steps": [
        {"source": "user", "tool_calls": [{"name": "malformed", "arguments": {}}]},
        {"source": "tool", "observation": "Unknown function"},
        {"source": "user", "tool_calls": [{"name": "correct_name", "arguments": {}}]},
        {"source": "assistant", "message": "Recovery acknowledged."},
    ]}]
    original = deepcopy(trace)
    text = judge_trajectory_text(trace)
    assert trace == original
    assert text.count("source=user") == 2
    assert text.index("malformed") < text.index("Unknown function") < text.index("correct_name")
    assert "Recovery acknowledged." in text


@pytest.mark.parametrize("wrapper", [lambda messages: messages,
                                     lambda messages: {"messages": messages}])
def test_roleless_chat_messages_and_calls_are_unknown(wrapper):
    text = judge_trajectory_text(wrapper([
        {"role": "user", "content": "Please fix it."},
        {"content": "I will try.", "tool_calls": [{"name": "try_fix", "arguments": {}}]},
        {"type": "function_call", "name": "another_fix", "arguments": "{}"},
    ]))
    assert "source=user" in text
    assert text.count("source=unknown") == 2
    assert "source=assistant" not in text
    assert "try_fix" in text and "another_fix" in text


def test_captured_model_call_distinguishes_response_from_unknown_request_history():
    text = judge_trajectory_text({"ng_trajectory": {"model_calls": [{
        "request": {"messages": [{"content": "Unknown participant."},
                                 {"role": "user", "content": "Known customer."}]},
        "response": {"output": [{"type": "message", "content": "Model reply."}]},
    }]}})
    assert "source=unknown" in text and "Unknown participant." in text
    assert "source=user" in text and "Known customer." in text
    assert "source=assistant" in text and "Model reply." in text


@pytest.mark.parametrize("value", ["plain answer", {"steps": [None]},
                                   {"choices": [{"message": {}}, {"message": {}}]}])
def test_unsupported_or_partial_parse_adds_no_derived_view(value):
    assert judge_trajectory_text(value) is None


# Optional supporting navigation is lazy, scoped, and independent of verdicts.

LLM_SOURCE = (
    'def detect(item, llm_client): return llm_client.classify("Guess?", item["response_text"])'
)
ITEM = {"_item_id": "one", "response_text": "I guessed. Then I checked.", "reward": 0}


def _record(*, located=False):
    return {
        "version": localisation.LOCALISATION_VERSION,
        "item_id": "one", "source_sha256": localisation.text_hash(LLM_SOURCE),
        "assigned_label": "yes", "status": "located" if located else "deferred",
        "evidence_policy": "deferred_optional_v1",
        "passages": [localisation.passage("response_text", ITEM["response_text"], 0, 10,
                                          method="llm")] if located else [],
    }


def _snapshot(record):
    return make_snapshot(ITEM, item_id="one", source=LLM_SOURCE, failure_localisation=record)


def _navigation_review(tmp_path, record):
    return EvidenceReview(_snapshot(record), tmp_path / "review")


def test_optional_highlights_preserve_snapshot_and_saved_review_identity(tmp_path):
    pending, completed = _snapshot(_record()), _snapshot(_record(located=True))
    assert pending["snapshot_id"] == completed["snapshot_id"] == _snapshot(None)["snapshot_id"]
    review = EvidenceReview(pending, tmp_path / "review")
    review.read("response_text")
    review.cite("response_text", "I guessed.")
    restored = EvidenceReview.restore(completed, review.directory)
    assert restored.reads == review.reads and restored.citations == review.citations
    assert restored.snapshot["failure_localisation"]["passages"]
    changed = make_snapshot({**ITEM, "response_text": "Different evidence"}, item_id="one",
                            source=LLM_SOURCE, failure_localisation=_record(located=True))
    with pytest.raises(ValueError, match="saved evidence"):
        EvidenceReview.restore(changed, review.directory)


def test_request_is_lazy_cached_and_does_not_mark_evidence_read(tmp_path):
    review = _navigation_review(tmp_path, _record())
    calls = []

    def resolve():
        calls.append(1)
        return _record(located=True)

    review._supporting_passage_resolver = resolve
    context = json.loads(review.initial_context(0))
    assert "retrieve_supporting_passages" in context["instructions"]
    specs = {name: (schema, call) for name, _, schema, call
             in claude_judge._evidence_tool_specs(review)}
    schema, call = specs["retrieve_supporting_passages"]
    assert schema == {"type": "object", "properties": {}, "required": [],
                      "additionalProperties": False}
    assert calls == []
    first = call()
    assert first["passages"][0]["quote"] == "I guessed."
    first["passages"].clear()
    assert call()["passages"] and calls == [1]
    assert not review.reads and not review.citations
    review.compacted({})
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    restored._supporting_passage_resolver = lambda: pytest.fail("Saved navigation must be reused")
    assert restored.supporting_passages()["passages"]
    assert not restored.reads and not restored.citations
    invalid = claude_judge._evidence_tool_result(
        "retrieve_supporting_passages", call, {"item_id": "other"},
    )
    assert invalid["isError"] and calls == [1]


@pytest.mark.parametrize("mismatch", ["item", "source"])
def test_request_rejects_unbound_navigation(tmp_path, mismatch):
    review = _navigation_review(tmp_path, _record())
    wrong = _record(located=True)
    wrong["item_id" if mismatch == "item" else "source_sha256"] = "other"
    review._supporting_passage_resolver = lambda: wrong
    with pytest.raises(ValueError, match="different"):
        review.supporting_passages()
    assert review._supporting_passage_result is None


def test_resolver_binds_current_input_and_shares_cache_between_review_stages(monkeypatch):
    from autofa.mcp import runtime

    requests, reservations, runtimes = [], [], []

    class Localiser:
        def localise(self, *args, **kwargs):
            reserve_model_call()
            requests.append((args, kwargs))
            return '{"quotes":["I guessed."]}'

    def get_runtime():
        runtimes.append(1)
        return SimpleNamespace(ensure_hub_llm=lambda: Localiser())

    def resolve(record, item, source, client, *, reviewed_source=False):
        assert item == ITEM and source == LLM_SOURCE and record == _record()
        assert reviewed_source
        client.localise("Locate supporting passages", item["response_text"])
        return _record(located=True)

    monkeypatch.setattr(runtime, "get_runtime", get_runtime)
    monkeypatch.setattr(localisation, "resolve_deferred_localisation", resolve, raising=False)
    item, record = deepcopy(ITEM), _record()
    callback = claude_judge._supporting_passage_resolver(
        item=item, record=record, source=LLM_SOURCE,
    )
    assert requests == runtimes == []
    item["response_text"] = "Mutated after review selected"
    record["assigned_label"] = "no"
    with model_call_budget(lambda: reservations.append(1)):
        assert callback() == callback() == _record(located=True)
    assert len(requests) == len(reservations) == len(runtimes) == 1


def test_cancellation_blocks_extraction_and_does_not_cache_partial_result(tmp_path):
    review = _navigation_review(tmp_path, _record())
    cancellation = Cancellation()

    def cancelled():
        cancellation.cancel()
        return _record(located=True)

    review._supporting_passage_resolver = cancelled
    with cancellation_scope(cancellation), pytest.raises(asyncio.CancelledError):
        review.supporting_passages()
    assert review._supporting_passage_result is None
    review._supporting_passage_resolver = lambda: pytest.fail("Cancelled requests cannot dispatch")
    with cancellation_scope(cancellation), pytest.raises(asyncio.CancelledError):
        review.supporting_passages()


def test_optional_backend_error_keeps_original_evidence_available_and_can_retry(tmp_path):
    import httpx
    from openai import APIConnectionError

    review = _navigation_review(tmp_path, _record())
    calls = []

    def resolve():
        calls.append(1)
        if len(calls) == 1:
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        return _record(located=True)

    review._supporting_passage_resolver = resolve
    first = claude_judge._evidence_tool_result(
        "retrieve_supporting_passages", review.supporting_passages, {},
    )
    assert first["isError"] and json.loads(first["content"][0]["text"])["retryable"]
    assert review._supporting_passage_result is None
    assert review.read("response_text")["text"] == ITEM["response_text"]
    second = claude_judge._evidence_tool_result(
        "retrieve_supporting_passages", review.supporting_passages, {},
    )
    assert not second.get("isError") and len(calls) == 2
    assert json.loads(second["content"][0]["text"])["passages"]


def test_navigation_tool_does_not_swallow_budget_exhaustion(tmp_path):
    class BudgetStop(BaseException):
        """A non-Exception stop signal, like a hard model-call budget."""

    review = _navigation_review(tmp_path, _record())

    def resolve():
        raise BudgetStop("model call budget exhausted")

    review._supporting_passage_resolver = resolve
    with pytest.raises(BudgetStop):
        claude_judge._evidence_tool_result(
            "retrieve_supporting_passages", review.supporting_passages, {},
        )
    assert review._supporting_passage_result is None


def test_missing_or_stale_record_cannot_authorize_extraction(tmp_path):
    resolver = claude_judge._supporting_passage_resolver
    assert resolver(item=ITEM, record=None, source=LLM_SOURCE) is None
    assert resolver(item=ITEM, record={**_record(), "item_id": "other"}, source=LLM_SOURCE) is None
    assert _navigation_review(tmp_path, None).supporting_passages()["status"] == "unavailable"


def test_initial_and_confirmation_reviews_share_requested_navigation_only(tmp_path, monkeypatch):
    requests, sessions = [], []

    def resolve(record, item, source, client, **kwargs):
        requests.append(item["_item_id"])
        return _record(located=True)

    async def session(*, review, **kwargs):
        sessions.append(review)
        result = review.supporting_passages()
        assert result["passages"][0]["quote"] == "I guessed."
        assert not review.reads and not review.citations
        review.read("detector_source")
        review.read("response_text")
        citation = review.cite("response_text", "I guessed.")["citations"][0]
        return json.dumps({"verdicts": [{
            "item_id": "one", "snapshot_id": review.snapshot["snapshot_id"],
            "claim_contract_id": review.snapshot["claim_contract"]["contract_id"],
            "claim_assessments": [
                {"requirement_id": requirement["id"], "status": "supported",
                 "rationale": "The original response explicitly admits guessing.",
                 "citations": [citation]}
                for requirement in review.snapshot["claim_contract"]["requirements"]
            ],
            "correct": True, "scope": "local", "citations": [citation],
        }], "behavior_specificity": 0.8})

    monkeypatch.setattr(localisation, "resolve_deferred_localisation", resolve, raising=False)
    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", session)
    kwargs = {"name": "guess", "mechanism": "guesses", "description": "guess",
              "source": LLM_SOURCE, "items": [ITEM], "assigned_labels": [True],
              "cwd": tmp_path, "model": "claude-test", "max_chars": 0}
    result = claude_judge.judge_behavior_with_claude(**kwargs, failure_localisations=[_record()])
    assert result["behavior_purity"] == 1.0 and result["n_completed"] == 1
    assert len(sessions) == 2 and requests == ["one"]
    assert sessions[0].citations is not sessions[1].citations
    # Enrichment after review neither reruns the judge nor changes its confirmation.
    again = claude_judge.judge_behavior_with_claude(
        **kwargs, failure_localisations=[_record(located=True)],
    )
    assert again["behavior_purity"] == 1.0 and len(sessions) == 2 and requests == ["one"]


def test_requested_passages_are_reused_by_finalization_without_another_call(tmp_path, monkeypatch):
    from autofa.core.sandbox import run_detector
    from autofa.mcp import runtime

    calls = []

    class Detector:
        def classify(self, *args, **kwargs):
            return "yes"

        def localise(self, *args, **kwargs):
            calls.append((args, kwargs))
            return '{"quotes":["I guessed."]}'

    client, record = Detector(), {}
    assert run_detector(LLM_SOURCE, ITEM, llm=client, localisation=record) == "yes"
    assert not calls and record["deferred_localisation"]["status"] == "pending"
    monkeypatch.setattr(runtime, "get_runtime",
                        lambda: SimpleNamespace(ensure_hub_llm=lambda: client))
    cache = tmp_path / "localisation_cache"
    review = _navigation_review(tmp_path, record)
    assert review.snapshot["failure_localisation"]["status"] == "deferred"
    review._supporting_passage_resolver = claude_judge._supporting_passage_resolver(
        item=ITEM, record=record, source=LLM_SOURCE, cache_dir=cache,
    )
    before = deepcopy(record)
    assert review.supporting_passages()["passages"][0]["quote"] == "I guessed."
    assert len(calls) == 1 and record == before
    saved = localisation.resolve_deferred_localisation(
        record, ITEM, LLM_SOURCE, client, cache_dir=cache,
    )
    assert saved["passages"][0]["quote"] == "I guessed." and len(calls) == 1
    assert _snapshot(saved)["snapshot_id"] == review.snapshot["snapshot_id"]
    # A revision cannot use a previous example's content-addressed receipt.
    with pytest.raises(ValueError, match="different item, source or content"):
        localisation.resolve_deferred_localisation(
            record, {**ITEM, "response_text": "Changed"}, LLM_SOURCE, client, cache_dir=cache,
        )
    assert len(calls) == 1


# Evidence projection changes invalidate saved judgments and processing retries.

CLAIM = {
    "source": "def guess(item): return 'guess' in item['response_text']",
    "description": "The assistant submits a guess.",
    "mechanism": "The assistant substitutes a guess for a derived value.",
}
GUESS_ITEM = {"_item_id": "case", "response_text": "I guess this value."}


def _saved_function(policy_hash, verdict="yes"):
    return {"name": "guess", **CLAIM, "scores": {
        "claim_hash": claim_hash(**CLAIM), "judge_policy_hash": policy_hash,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_evidence_version": judge_evidence.EVIDENCE_VERSION,
        "judge_model": "test", "judge_backend": "claude_agent_sdk",
        "judge_behavior_strictness": "causal",
        "behavior_purity": 1.0 if verdict == "yes" else None,
        "behavior_n_judged": 1, "behavior_n_yes": int(verdict == "yes"),
        "behavior_n_errors": int(verdict == "error"),
        "behavior_verdicts": [{"item_id": "case", "verdict": verdict}],
    }}


def _plan_retry(directory, fn):
    feedback = {
        "claim_hash": fn["scores"]["claim_hash"], "allowed_actions": ["retry_review"],
        "diagnostic_item_ids": ["case"],
        "findings": [{"id": "failed-review", "status": "processing_error",
                      "examples": [{"item_id": "case"}]}],
    }
    return repair_workflow.plan_repair(
        directory, fn, feedback, action="retry_review",
        rationale="Retry the failed processing without changing the claim.",
        finding_ids=["failed-review"], evidence_ids=["case"],
    )


def _stub_reviews(monkeypatch):
    """Replace judge sessions with complete reads that support every requirement."""
    calls = []

    def review_all(*, review, **kwargs):
        calls.append(review.snapshot)
        for field in review.fields:
            review.read(field)
        contract = review.snapshot["claim_contract"]
        citation = review.cite("response_text", GUESS_ITEM["response_text"])["citations"][0]
        return json.dumps({"verdicts": [{
            "item_id": "case", "snapshot_id": review.snapshot["snapshot_id"],
            "claim_contract_id": contract["contract_id"], "scope": "local",
            "claim_assessments": [{
                "requirement_id": row["id"], "status": "supported",
                "rationale": "Stub semantic assessment of the recorded guess.",
                "citations": [{"citation_id": citation["citation_id"]}], "counterevidence": [],
            } for row in contract["requirements"]],
        }]})

    monkeypatch.setattr(claude_judge, "_run_review", review_all)
    return calls


def test_saved_search_preference_keeps_scores_without_relabeling_provenance():
    strategy = "precision"
    saved_hash = claude_judge.judge_policy_hash()
    fn = _saved_function(saved_hash)
    fn["scores"]["judge_objective_strategy"] = strategy
    for requested in ("precision", "balanced", "recall"):
        scores = current_behavior_scores(fn["scores"], **CLAIM, objective_strategy=requested)
        assert behavior_validation(scores)["status"] == "confirmed"
        assert resolved_behavior_purity(scores) == 1.0
        assert scores["judge_policy_hash"] == saved_hash
        assert scores["judge_objective_strategy"] == strategy
    assert current_behavior_scores(fn["scores"], **{**CLAIM, "description": "Changed claim"})[
        "behavior_judgment_stale"]
    assert current_behavior_scores(fn["scores"], **CLAIM, behavior_strictness="behavioral")[
        "behavior_judgment_stale"]


@pytest.mark.parametrize("component", ["EVIDENCE_VERSION", "compiler_policy_hash"])
def test_policy_from_older_evidence_or_compiler_is_not_compatible(monkeypatch, component):
    with monkeypatch.context() as old:
        value = getattr(claude_judge, component)
        old.setattr(claude_judge, component,
                    (lambda *args, **kwargs: "obsolete-policy") if callable(value) else "obsolete")
        saved = claude_judge.judge_policy_hash()
    assert not claude_judge.compatible_judge_policy(saved)


def test_projection_field_policy_changes_invalidate_scores_and_active_retries(
    tmp_path, monkeypatch,
):
    saved_policy = claude_judge.judge_policy_hash()
    fn = _saved_function(saved_policy)
    assert behavior_validation(fn["scores"])["status"] == "confirmed"
    failed = _saved_function(saved_policy, verdict="error")
    plan = _plan_retry(tmp_path, failed)
    assert repair_workflow.validation_context(tmp_path, failed)["plan_id"] == plan["plan_id"]
    before = judge_evidence.evidence_projection_hash()
    # Content-field changes are covered by the stage-reuse test below.
    monkeypatch.setattr(judge_evidence, "_OUTCOMES",
                        judge_evidence._OUTCOMES | {"new_projection_field"})
    assert judge_evidence.evidence_projection_hash() != before
    assert claude_judge.judge_policy_hash() != saved_policy
    scores = current_behavior_scores(fn["scores"], **CLAIM)
    assert behavior_validation(scores)["status"] == "stale"
    assert resolved_behavior_purity(scores) is None
    assert point_from_scores(scores, ["behavior_purity"]) is None
    with pytest.raises(ValueError, match="Judge policy changed"):
        repair_workflow.validation_context(tmp_path, failed)


def test_current_projection_preserves_system_roles_and_tool_results(tmp_path):
    item = {
        "system_text": "Require explicit confirmation before changing a value.",
        "prompt_text": "Please change the value.",
        "response_text": json.dumps({"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "Value changed."},
            {"type": "text", "text": "I can see the new value now."},
        ]}]}),
    }
    snapshot = make_snapshot(item, item_id="case", source=CLAIM["source"])
    assert snapshot["fields"]["system_text"] == item["system_text"]
    messages = json.loads(snapshot["fields"]["input_messages"])
    assert [message["role"] for message in messages] == ["system", "user"]
    # Anthropic tool results stay tool events, followed by the customer's own text.
    trajectory = snapshot["fields"]["trajectory_text"]
    assert "source=tool" in trajectory and "source=user" in trajectory
    assert trajectory.index("Value changed.") < trajectory.index("I can see the new value now.")
    assert snapshot["projection_hash"] == judge_evidence.evidence_projection_hash()
    review = EvidenceReview(snapshot, tmp_path / "review")
    assert review.manifest()["projection_hash"] == snapshot["projection_hash"]


def test_policy_change_reruns_judge_stages_but_reuses_trace_blind_contract(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    calls = _stub_reviews(monkeypatch)
    first = _judge(tmp_path, [GUESS_ITEM])
    repeated = _judge(tmp_path, [GUESS_ITEM])
    assert first["n_yes"] == repeated["n_yes"] == 1
    assert first["verdicts"][0]["verification_status"] == "confirmed"
    # A confirmed review without a judged specificity never invents one.
    assert first["behavior_purity"] == 1 and first["behavior_specificity"] is None
    assert len(calls) == 2 and repeated["reused_stages"] == 2

    monkeypatch.setattr(judge_evidence, "DEFAULT_CONTENT_FIELDS",
                        judge_evidence.DEFAULT_CONTENT_FIELDS | {"new_content_field"})
    current = _judge(tmp_path, [GUESS_ITEM], compiled_contract=first["claim_contract"])
    assert current["n_yes"] == 1 and current["n_errors"] == 0
    assert len(calls) == 4 and current["reused_stages"] == 0
    assert len(stub_claim_compiler) == 1
    assert current["claim_contract"] == first["claim_contract"]
    assert current["claim_compilation"]["reused"] is True
    assert current["judge_policy_hash"] != first["judge_policy_hash"]
    assert current["judge_rubric_hash"] != first["judge_rubric_hash"]
    assert calls[0]["projection_hash"] != calls[2]["projection_hash"]
    assert calls[0]["snapshot_id"] != calls[2]["snapshot_id"]
