# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded absence requires complete original events, never selected search hits."""

import json
from copy import deepcopy

import pytest

from autofa.data.evidence_refs import compact_trace
from autofa.metrics import claude_judge
from autofa.metrics.claim_contract import build_claim_contract
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot
from autofa.metrics.judge_intervals import interval_navigation

from support.judge import read_ranges

SOURCE = "def detect(item): return True"
DESCRIPTION = "Opens the recovery source without a prior backup."
DELETE_DESCRIPTION = (
    "Before deleting the original file, the assistant does not verify its replacement."
)
BOUNDARY_QUOTE = "rm original.txt"


def _contract(*, start=None, end=None):
    return build_claim_contract(
        description=DESCRIPTION, mechanism="", source=SOURCE, defining_clauses=[{
            "id": "no_backup", "source_field": "description", "quote": DESCRIPTION,
            "predicate": "No backup precedes the source mutation.",
            "required_review_scope": "bounded_interval",
            "review_interval": {
                "start": start or {"kind": "run_start"},
                "end": end or {"kind": "event", "predicate": "The source-mutating command",
                               "position": "before"},
            },
        }],
    )


def _trace():
    return {"steps": [
        {"source": "user", "message": "Recover all records from original.db."},
        {"source": "assistant", "message": "inspect " * 3000},
        {"source": "assistant", "message": json.dumps({"commands": [
            {"keystrokes": "cp original.db backup.db; sqlite3 original.db; cp original.db late.db"},
        ]})},
        {"source": "user", "message": "WAL is gone"},
        {"source": "assistant", "message": "later unrelated work " * 8000},
    ]}


def _review(tmp_path, trace=None, **contract_args):
    snapshot = make_snapshot(
        {"response_text": trace or _trace()}, item_id="case", source=SOURCE,
        claim_contract=_contract(**contract_args),
    )
    return EvidenceReview(snapshot, tmp_path / "review")


def _anchor(review, quote="sqlite3 original.db"):
    return review.search("response_text", quote)["matches"][0]["citation"]["citation_id"]


def _bind(review):
    anchor = _anchor(review)
    receipt = review.review_interval("no_backup", 1, 3, end_citation_id=anchor)
    return receipt, anchor


def _read_everything(review):
    read_ranges(review, [{"field": "response_text", "start": 0,
                          "end": len(review.fields["response_text"])}])


def _verdict(review, receipt, anchor, status="supported"):
    review.read("detector_source")
    citation = review.citations[anchor]
    return {
        "item_id": "case", "snapshot_id": review.snapshot["snapshot_id"],
        "scope": "bounded_interval", "citations": [citation],
        "claim_assessments": [{"requirement_id": "no_backup", "status": status,
                               "citations": [citation],
                               "review_interval_id": receipt["interval_id"]}],
    }


def test_search_binding_or_skipped_middle_event_do_not_prove_absence(tmp_path):
    review = _review(tmp_path)
    receipt, anchor = _bind(review)
    # The interval holds the whole boundary batch, before and after its command.
    text = "\n".join(review.fields[r["field"]][r["start"]:r["end"]]
                     for r in receipt["required_ranges"])
    assert "cp original.db backup.db; sqlite3 original.db; cp original.db late.db" in text
    assert "later unrelated work" not in text
    verdict = _verdict(review, receipt, anchor)
    assert not receipt["complete"]
    assert "unread interval content" in review.validate(verdict)
    events = review.interval_navigation()["events"]
    read_ranges(review, events[0]["ranges"] + events[2]["ranges"])
    assert "unread interval content" in review.validate(verdict)
    read_ranges(review, receipt["required_ranges"])
    assert review.validate(verdict) is None
    assert review.status()["review_intervals"][0]["complete"]


def test_unknown_interval_must_bind_and_read_only_the_frozen_unread_ranges(tmp_path):
    review = _review(tmp_path)
    # Even an ambiguity verdict must bind and read its frozen interval.
    assessment = {"requirement_id": "no_backup", "status": "unknown",
                  "unknown_reason": "ambiguous_evidence"}
    pending = review.completion_plan([assessment])
    assert "Bind the frozen interval" in pending[0]["next_step"]
    assert pending[0]["unread_event_count"] == 5
    receipt, _ = _bind(review)
    assessment["review_interval_id"] = receipt["interval_id"]
    pending = review.completion_plan([assessment])
    assert pending[0]["review_interval_id"] == receipt["interval_id"]
    assert pending[0]["required_review_scope"] == "bounded_interval"
    assert pending[0]["unread_ranges"] == review.unread_ranges(receipt["required_ranges"])
    assert pending[0]["unread_ranges"] == review.status()["review_intervals"][0]["unread_ranges"]
    read_ranges(review, pending[0]["unread_ranges"])
    assert review.completion_plan([assessment]) == []
    assert review._interval_complete(receipt)
    # The scoped unknown is legitimate now; later unrelated events stay unread.
    assert not review.covers("response_text", 0, len(review.fields["response_text"]))
    assert review.reconcile_completion([assessment]) == [assessment]


def test_unknown_interval_does_not_require_impossible_binding_after_complete_read(tmp_path):
    review = _review(tmp_path)
    assessment = {"requirement_id": "no_backup", "status": "unknown",
                  "unknown_reason": "evidence_unavailable"}
    navigation = review.interval_navigation()
    read_ranges(review, [span for event in navigation["events"] for span in event["ranges"]])
    # The claimed mutation cannot be semantically established; binding an interval
    # to it would invent a boundary. Complete reading may legitimately stay unknown.
    assert review.completion_plan([assessment]) == []
    review._interval_navigation = {"available": False, "reason": "Events were not retained."}
    assessment["unknown_reason"] = "review_incomplete"
    assert review.completion_plan([assessment]) == []


def test_local_refutation_does_not_require_complete_interval(tmp_path):
    review = _review(tmp_path)
    receipt, anchor = _bind(review)
    assert review.validate(_verdict(review, receipt, anchor, "contradicted")) is None


@pytest.mark.parametrize("inside", [False, True], ids=["after_boundary", "inside_interval"])
def test_long_trace_interval_is_funded_but_supported_only_once_read(tmp_path, inside):
    from autofa.metrics.judge_budget import MAX_TOOL_CALLS, plan_review_budget

    trace = _trace()
    trace["steps"][1 if inside else 4]["message"] = "x" * 6_000_000
    review = _review(tmp_path, trace)
    # The unbound interval reserves the ceiling instead of failing before dispatch.
    plan = plan_review_budget(review)
    assert plan.scope == "bounded_interval" and plan.tool_call_limit == MAX_TOOL_CALLS
    receipt, anchor = _bind(review)
    verdict = _verdict(review, receipt, anchor)
    assert "unread interval content" in review.validate(verdict)
    read_ranges(review, receipt["required_ranges"])
    assert review.validate(verdict) is None
    # Events after the frozen boundary stay unread; a long tail costs no reads.
    assert not review.covers("response_text", 0, len(review.fields["response_text"]))
    assert len(review.reads) < (MAX_TOOL_CALLS if inside else 8)


@pytest.mark.parametrize("start,end,error", [
    (2, 3, "frozen run_start"),  # Cannot move the frozen start past the backup.
    (0, 3, "1-based"), (3, 1, "1-based"), (1, 9, "1-based"),
])
def test_frozen_or_invalid_ordered_boundaries_are_rejected(tmp_path, start, end, error):
    review = _review(tmp_path)
    with pytest.raises(ValueError, match=error):
        review.review_interval("no_backup", start, end, end_citation_id=_anchor(review))


def test_anchor_must_belong_to_selected_original_event(tmp_path):
    review = _review(tmp_path)
    with pytest.raises(ValueError, match="original evidence at event 2"):
        review.review_interval("no_backup", 1, 2, end_citation_id=_anchor(review))
    review.read("trajectory_text")
    derived = review.search("trajectory_text", "sqlite3 original.db")["matches"][0]
    with pytest.raises(ValueError, match="original evidence"):
        review.review_interval(
            "no_backup", 1, 3, end_citation_id=derived["citation"]["citation_id"])


def test_receipt_and_boundary_citation_are_bound_to_claim_and_cannot_be_invented(tmp_path):
    review = _review(tmp_path)
    receipt, anchor = _bind(review)
    verdict = _verdict(review, receipt, anchor)
    verdict["claim_assessments"][0]["citations"] = []
    assert "must cite its selected boundary" in review.validate(verdict)
    verdict = _verdict(review, receipt, anchor)
    verdict["claim_assessments"][0]["review_interval_id"] = "interval-forged"
    assert "issued review_interval_id" in review.validate(verdict)
    verdict["claim_assessments"][0]["review_interval_id"] = receipt["interval_id"]
    review.intervals[receipt["interval_id"]]["required_ranges"] = []
    assert "differs from the frozen" in review.validate(verdict)


def test_receipts_survive_restore_but_boundary_citations_need_fresh_reads(tmp_path):
    review = _review(tmp_path)
    receipt, anchor = _bind(review)
    read_ranges(review, receipt["required_ranges"])
    verdict = _verdict(review, receipt, anchor)
    review.compacted({"test": True})
    assert "not read after latest compaction" in review.validate(verdict)
    citation = review.citations[anchor]
    review.read(citation["field"], citation["start"], citation["end"] - citation["start"])
    assert review.validate(verdict) is None
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.validate(verdict) is None
    assert restored.status()["review_intervals"][0]["complete"]


def test_suffix_requires_frozen_run_end_and_complete_boundary_event(tmp_path):
    review = _review(tmp_path,
                     start={"kind": "event", "predicate": "Mutation", "position": "after"},
                     end={"kind": "run_end"})
    anchor = _anchor(review)
    with pytest.raises(ValueError, match="frozen run_end"):
        review.review_interval("no_backup", 3, 4, start_citation_id=anchor)
    receipt = review.review_interval("no_backup", 3, 5, start_citation_id=anchor)
    assert (receipt["required_ranges"][0]["start"]
            == review.interval_navigation()["events"][2]["ranges"][0]["start"])
    assert (receipt["required_ranges"][-1]["end"]
            == review.interval_navigation()["events"][-1]["ranges"][-1]["end"])


def _imported_trace():
    steps, originals = [], []
    for index, message in enumerate(["Recover the source " * 100, "sqlite3 original.db", "later"]):
        part = {"part_idx": 0, "text": message}
        event = {"event_idx": index, "lane": "", "role": "assistant", "parts": [part]}
        originals.append(event)
        steps.append({"source": "assistant", "step_id": index, "message": message,
                      "parts": [part], "metadata": {k: v for k, v in event.items() if k != "parts"},
                      "source_parts": [{"part_idx": 0, "field": "message",
                                        "start": 0, "end": len(message)}]})
    return {"steps": steps,
            "evidence_scope": {"source": "original API event parts and tool arguments",
                               "event_lane": "", "limitations": {}},
            "execution": {"events": originals, "tool_calls": []}}


def test_compact_prefix_requires_referenced_originals_not_just_step_stubs(tmp_path):
    review = _review(tmp_path, compact_trace(_imported_trace(), include_rendered=True))
    anchor = _anchor(review)
    receipt = review.review_interval("no_backup", 1, 2, end_citation_id=anchor)
    for event in review.interval_navigation()["events"][:2]:
        assert len(event["ranges"]) == 2
        read_ranges(review, event["ranges"][:1])
    assert "unread interval" in review.validate(_verdict(review, receipt, anchor))
    read_ranges(review, receipt["required_ranges"])
    assert review.validate(_verdict(review, receipt, anchor)) is None


@pytest.mark.parametrize("location", ["envelope", "scope", "step", "metadata", "part"])
def test_explicit_truncation_cannot_certify_absence(tmp_path, location):
    trace = _trace()
    if location == "envelope":
        trace["truncated"] = True
    elif location == "scope":
        trace["evidence_scope"] = {"limitations": {"missing events": 1}}
    elif location == "step":
        trace["steps"][1]["length_truncated"] = True
    elif location == "metadata":
        trace["steps"][1]["metadata"] = {"length_truncated": True}
    else:
        trace["steps"][1]["parts"] = [{"truncated": True, "text": "partial"}]
    review = _review(tmp_path, trace)
    with pytest.raises(ValueError, match="truncated|limitations"):
        _bind(review)


# One generic gap word plus each gap label the event normalizer emits; both
# representations yield the same labels.
@pytest.mark.parametrize("representation,label", [
    ("list", "missing events"), ("counted", "source event marked truncated"),
    ("counted", "excluded event kind/actor: action/assistant"),
    ("list", "unsupported event role: unknown"),
])
def test_declared_evidence_gaps_never_issue_bounded_receipts(tmp_path, representation, label):
    trace = _trace()
    trace["evidence_scope"] = {"limitations": [label] if representation == "list" else {label: 1}}
    review = _review(tmp_path, trace)
    assert review.interval_navigation()["available"] is False
    # Even reading every retained byte cannot establish absence in omitted events.
    _read_everything(review)
    with pytest.raises(ValueError, match="evidence limitations"):
        _bind(review)
    assert review.intervals == {}


@pytest.mark.parametrize("limitations", [None, [""], {"missing events": 0.5}])
def test_unsupported_limitations_fail_closed_before_receipt(tmp_path, limitations):
    trace = _trace()
    trace["evidence_scope"] = {"limitations": limitations}
    review = _review(tmp_path, trace)
    navigation = review.interval_navigation()
    assert navigation == {"available": False, "reason": "malformed recorded evidence limitations"}
    with pytest.raises(ValueError, match="malformed recorded evidence limitations"):
        _bind(review)
    assert review.intervals == {}


@pytest.mark.parametrize("limitations", [
    [], {"missing events": 0.0, "encrypted reasoning unavailable": 1.0},
    ["encrypted reasoning unavailable", "artifact file contents not inspected"],
])
def test_supported_limitations_preserve_complete_recorded_intervals(tmp_path, limitations):
    trace = _trace()
    trace["evidence_scope"] = {"limitations": limitations}
    trace["steps"][0]["message"] += " Quoted output: missing events; command events omitted."
    review = _review(tmp_path, trace)
    receipt, anchor = _bind(review)
    read_ranges(review, receipt["required_ranges"])
    assert review.validate(_verdict(review, receipt, anchor)) is None


def test_request_telemetry_is_not_an_action_but_other_lane_actions_block_coverage(tmp_path):
    trace = _imported_trace()
    telemetry = {"event_idx": 0, "lane": "model_call", "kind": "other", "role": None,
                 "lane_kind": "model_call", "kind_raw": "model_call",
                 "parts": [{"origin": "request", "text": "request status"}]}
    trace["execution"]["events"].append(telemetry)
    review = _review(tmp_path, trace)
    assert review.interval_navigation()["available"]
    bad = deepcopy(review.snapshot)
    blob = json.loads(bad["fields"]["response_text"])
    blob["execution"]["events"][-1].update(role="assistant", kind="message")
    bad["fields"]["response_text"] = json.dumps(blob)
    assert not interval_navigation(bad)["available"]


def _local_import_trace():
    """Shaped like the local EvalRDL import: model calls repeat their response's tool calls."""
    trace = _imported_trace()
    trace["evidence_scope"].update(identity="evalrdl_tables_v1",
                                   source="local EvalRDL event parts and tool arguments")
    arguments = json.dumps({"command": "ls ✓"}, ensure_ascii=False)
    call = {"event_idx": 1, "lane": "", "call_idx": 0, "call_id_source": "call-1",
            "tool_name": "bash", "args_full": arguments, "args_chars": len(arguments)}
    trace["execution"]["events"][1]["n_tool_calls"] = 1
    trace["execution"]["events"].append({
        "event_idx": 0, "lane": "model_call", "kind": "other", "role": None,
        "lane_kind": "model_call", "kind_raw": "model_call", "n_tool_calls": 1,
        "parts": [{"origin": "request", "text": "dialect=responses · status_code=200"}],
    })
    escaped = json.dumps({"command": "ls ✓"})  # The model-call capture escapes non-ASCII.
    trace["execution"]["tool_calls"] = [call, {**call, "lane": "model_call", "event_idx": 0,
                                               "args_full": escaped, "args_chars": len(escaped)}]
    return trace


def _mirror(trace):
    return trace["execution"]["tool_calls"][1]


def _set_mirror_arguments(trace, arguments):
    _mirror(trace).update(args_full=arguments, args_chars=len(arguments))


def _set_arguments(trace, original, mirror):
    for call, value in zip(trace["execution"]["tool_calls"], (original, mirror), strict=True):
        text = json.dumps(value)
        call.update(args_full=text, args_chars=len(text))


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda t: None, id="as_imported"),
    pytest.param(lambda t: _set_arguments(t, {"command": "ls", "env": {"A": "1", "B": "2"}},
                                          {"env": {"B": "2", "A": "1"}, "command": "ls"}),
                 id="reordered_members"),
    pytest.param(lambda t: t["execution"]["events"][-1].update(metadata={"n_tool_calls": 1}),
                 id="metadata_count_agrees"),
])
def test_local_import_identity_and_mirrored_model_call_calls_allow_bounded_receipt(
    tmp_path, mutation,
):
    trace = _local_import_trace()
    mutation(trace)
    review = _review(tmp_path, trace)
    navigation = review.interval_navigation()
    assert navigation["available"] and navigation["event_count"] == 3
    assert all(event["complete"] for event in navigation["events"])
    anchor = _anchor(review)
    receipt = review.review_interval("no_backup", 1, 2, end_citation_id=anchor)
    assert "unread interval" in review.validate(_verdict(review, receipt, anchor))
    read_ranges(review, receipt["required_ranges"])
    assert review.validate(_verdict(review, receipt, anchor)) is None


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda t: t["evidence_scope"].pop("identity"), id="unverified_local_source"),
    pytest.param(lambda t: _set_mirror_arguments(t, '{"command": "rm -rf /"}'),
                 id="different_arguments"),
    pytest.param(lambda t: _set_mirror_arguments(t, '{"command": "ls ✓", "command": "rm"}'),
                 id="ambiguous_arguments"),
    pytest.param(lambda t: _set_arguments(t, {"argv": ["ls", "-a"]}, {"argv": ["-a", "ls"]}),
                 id="reordered_array"),
    pytest.param(lambda t: _mirror(t).update(event_idx=False), id="boolean_event_id"),
    pytest.param(lambda t: t["execution"]["events"][-1].update(metadata={"n_tool_calls": 2}),
                 id="metadata_count_disagrees"),
    pytest.param(lambda t: _mirror(t).update(tool_name="python"), id="different_tool"),
    pytest.param(lambda t: _mirror(t).update(call_id_source="call-2"), id="unknown_call_id"),
    pytest.param(lambda t: _mirror(t).pop("call_id_source"), id="missing_call_id"),
    pytest.param(lambda t: _mirror(t).update(truncated=True), id="truncated_mirror"),
    pytest.param(lambda t: t["execution"]["tool_calls"].pop(), id="declared_call_missing"),
    pytest.param(lambda t: t["execution"]["tool_calls"].append(
        {**_mirror(t), "call_idx": 1}), id="repeated_mirror"),
    pytest.param(lambda t: t["execution"]["tool_calls"].append(
        {**t["execution"]["tool_calls"][0], "event_idx": 2}), id="ambiguous_original"),
    # Each count agrees, so only the one-mirror-per-call and unique-identity checks reject.
    pytest.param(lambda t: (
        t["execution"]["events"].append({**t["execution"]["events"][-1], "event_idx": 1}),
        t["execution"]["tool_calls"].append({**_mirror(t), "event_idx": 1}),
    ), id="two_events_one_original"),
    pytest.param(lambda t: t["execution"]["events"].append({**t["execution"]["events"][-1]}),
                 id="duplicate_telemetry_identity"),
    pytest.param(lambda t: t["execution"]["events"][-1].update(n_tool_calls=2),
                 id="count_disagrees"),
    pytest.param(lambda t: t["execution"]["events"][-1].update(role="assistant"),
                 id="unclassified_model_call"),
    pytest.param(lambda t: t["execution"]["events"][-1]["parts"][0].update(origin="response"),
                 id="response_payload"),
])
def test_unverified_model_call_tool_calls_block_interval_coverage(tmp_path, mutation):
    trace = _local_import_trace()
    mutation(trace)
    review = _review(tmp_path, trace)
    assert review.interval_navigation()["available"] is False
    _read_everything(review)
    with pytest.raises(ValueError, match="verified|declared count|ambiguous original"):
        review.review_interval("no_backup", 1, 2, end_citation_id=_anchor(review))
    assert review.intervals == {}


def test_unsupported_plain_text_and_conflicting_traces_fail_closed(tmp_path):
    review = _review(tmp_path, "I ran sqlite3 original.db")
    assert not review.interval_navigation()["available"]
    snapshot = make_snapshot({"response_text": _trace(), "generation": "different actions"},
                             item_id="case", source=SOURCE)
    assert not interval_navigation(snapshot)["available"]


def test_manifest_and_event_listing_are_paginated_and_do_not_mark_reads(tmp_path):
    trace = {"steps": [{"source": "assistant", "message": str(i)} for i in range(100)]}
    review = _review(tmp_path, trace)
    navigation = review.manifest()["interval_navigation"]
    assert len(navigation["events"]) == 20
    assert navigation["next_event"] == 21
    page = review.interval_events(start_event=21, limit=1000)
    assert len(page["events"]) == 40 and page["next_event"] == 61
    assert review.reads == []


def _omit_call_evidence(trace, kind, event=0):
    step, original = trace["steps"][event], trace["execution"]["events"][event]
    if kind == "source_count":
        original["n_tool_calls"] = 1
    elif kind in {"source_args", "duplicate_source_call"}:
        call = {"event_idx": event, "lane": "", "call_idx": 0,
                "args_full": "cp source backup", "args_chars": len("cp source backup")}
        if kind == "source_args":
            call["args_chars"] += 20
        trace["execution"]["tool_calls"].append(call)
        if kind == "duplicate_source_call":
            trace["execution"]["tool_calls"].append(deepcopy(call))
        original["n_tool_calls"] = len(trace["execution"]["tool_calls"])
    elif kind == "inline_count":
        step["n_tool_calls"] = 1
    elif kind == "inline_metadata_count":
        step["metadata"]["n_tool_calls"] = 1
    else:
        call = {"function_name": "terminal", "arguments": "cp source backup"}
        if kind == "inline_args":
            call["args_chars"] = len(call["arguments"]) + 20
        elif kind == "inline_metadata_args":
            call["metadata"] = {"args_chars": len(call["arguments"]) + 20}
        else:
            call["truncated"] = True
        step["tool_calls"] = [call]


@pytest.mark.parametrize("kind", [
    "source_count", "source_args", "duplicate_source_call", "inline_count",
    "inline_metadata_count", "inline_args", "inline_metadata_args", "inline_truncated",
])
def test_missing_calls_and_short_arguments_cannot_certify_prefix_absence(tmp_path, kind):
    trace = _imported_trace()
    _omit_call_evidence(trace, kind)
    review = _review(tmp_path, trace)
    navigation = review.interval_navigation()
    assert navigation["available"] and navigation["events"][0]["complete"] is False
    anchor = _anchor(review)
    with pytest.raises(ValueError, match="incomplete or truncated"):
        review.review_interval("no_backup", 1, 2, end_citation_id=anchor)


@pytest.mark.parametrize("location,parts,declared_parts", [
    ("original_event", [{"part_type": "text"}], None),
    ("step", [{"type": "output_text", "text": None}], None),
    ("step", [{"text": "short", "chars": 20}], None),
    # Metadata part counts require every declared part, even with retained text.
    ("original_event", [], 1), ("step", None, 2),
], ids=["text-missing", "output-text-null", "short-text", "declared-part-missing",
        "declared-count-mismatch"])
def test_missing_declared_event_text_never_earns_interval_receipt(
    tmp_path, location, parts, declared_parts,
):
    trace = _imported_trace()
    record = trace["steps"][0] if location == "step" else trace["execution"]["events"][0]
    if parts is not None:
        record["parts"] = parts
    if declared_parts is not None:
        record.setdefault("metadata", {})["n_parts"] = declared_parts
    review = _review(tmp_path, trace)
    assert review.interval_navigation()["events"][0]["complete"] is False
    _read_everything(review)
    with pytest.raises(ValueError, match="incomplete or truncated"):
        review.review_interval("no_backup", 1, 2, end_citation_id=_anchor(review))
    assert review.intervals == {}


def test_complete_declared_text_and_part_counts_still_allow_receipts(tmp_path):
    trace = _imported_trace()
    for record in [*trace["steps"], *trace["execution"]["events"]]:
        record.setdefault("metadata", {})["n_parts"] = len(record["parts"])
        for part in record["parts"]:
            part.update(part_type="text", chars=len(part["text"]))
    review = _review(tmp_path, trace)
    receipt = review.review_interval("no_backup", 1, 2, end_citation_id=_anchor(review))
    read_ranges(review, receipt["required_ranges"])
    assert review._interval_complete(receipt)


def test_missing_later_call_does_not_erase_complete_earlier_prefix(tmp_path):
    trace = _imported_trace()
    _omit_call_evidence(trace, "source_args", event=2)
    review = _review(tmp_path, trace)
    navigation = review.interval_navigation()
    assert [event["complete"] for event in navigation["events"]] == [True, True, False]
    receipt = review.review_interval("no_backup", 1, 2, end_citation_id=_anchor(review))
    assert receipt["required_ranges"]


# The shared live parser, used by both harnesses, consumes interval receipts.


def _deletion_review(tmp_path, *, whole_trace=False):
    """Bind and read a prefix interval before an original-file deletion."""
    clauses = [{
        "id": "unchecked_prefix", "source_field": "description", "quote": DELETE_DESCRIPTION,
        "predicate": DELETE_DESCRIPTION, "required_review_scope": "bounded_interval",
        "review_interval": {
            "start": {"kind": "run_start"},
            "end": {"kind": "event", "predicate": "The assistant deletes the original file.",
                    "position": "before"},
        },
    }]
    description = DELETE_DESCRIPTION
    if whole_trace:
        description += " The original file is never restored."
        clauses.append({
            "id": "never_restored", "source_field": "description",
            "quote": "The original file is never restored.",
            "predicate": "The original file is never restored.",
            "required_review_scope": "whole_trace",
        })
    contract = build_claim_contract(
        description=description, mechanism="", source=SOURCE, defining_clauses=clauses,
    )
    text = json.dumps({"steps": [
        {"source": "user", "message": "Keep the original unless the replacement works."},
        {"source": "assistant", "tool_calls": [{"name": "terminal", "arguments": {
            "commands": [BOUNDARY_QUOTE, "attempt-replacement"],
        }}]},
        {"source": "tool", "observation": "replacement failed; original.txt is missing"},
        {"source": "assistant", "message": "I submit the missing file."},
    ]})
    review = EvidenceReview(make_snapshot(
        {"response_text": text}, item_id="case", source=SOURCE, claim_contract=contract,
    ), tmp_path / "review")
    review.read("detector_source")
    navigation = review.manifest()["interval_navigation"]
    read_ranges(review, navigation["events"][1]["ranges"])
    citation = review.cite("response_text", BOUNDARY_QUOTE)["citations"][0]
    receipt = review.review_interval("unchecked_prefix", 1, 2,
                                     end_citation_id=citation["citation_id"])
    read_ranges(review, receipt["required_ranges"])
    row = {
        "item_id": "case", "snapshot_id": review.snapshot["snapshot_id"],
        "claim_contract_id": contract["contract_id"], "scope": "bounded_interval",
        "claim_assessments": [{
            "requirement_id": requirement["id"], "status": "supported",
            "rationale": "Stub semantic assessment for evidence-protocol validation.",
            "citations": [{"citation_id": citation["citation_id"]}], "counterevidence": [],
            **({"review_interval_id": receipt["interval_id"]}
               if requirement["id"] == "unchecked_prefix" else {}),
        } for requirement in contract["requirements"]],
    }
    return review, row, receipt, citation


def _parse(review, row):
    return claude_judge.parse_session_result(
        json.dumps({"verdicts": [row], "behavior_specificity": 0.8}),
        [{"item_id": "case"}], source=SOURCE, reviews={"case": review},
    )


def _assessment(rows, requirement_id):
    return next(row for row in rows if row["requirement_id"] == requirement_id)


@pytest.mark.parametrize("contribution, expected", [
    ("supported", "yes"), ("unknown", "uncertain"), ("contradicted", "no"),
])
def test_live_parser_keeps_interval_receipt_without_waiving_causal_check(
    tmp_path, contribution, expected,
):
    review, row, receipt, citation = _deletion_review(tmp_path)
    assert not review.covers("response_text", 0, len(review.fields["response_text"]))
    _assessment(row["claim_assessments"], "failure_contribution")["status"] = contribution
    result = _parse(review, row)
    assert result["n_errors"] == 0
    verdict = result["verdicts"][0]
    assert verdict["verdict"] == expected
    bounded = _assessment(verdict["claim_assessments"], "unchecked_prefix")
    assert bounded["review_interval_id"] == receipt["interval_id"]
    assert bounded["citations"] == [citation]


def test_wrong_event_boundary_cannot_be_replaced_by_fabricated_receipt(tmp_path):
    review, row, _receipt, citation = _deletion_review(tmp_path)
    response = claude_judge._evidence_tool_result("review_interval", review.review_interval, {
        "requirement_id": "unchecked_prefix", "start_event": 1, "end_event": 3,
        "end_citation_id": citation["citation_id"],
    })
    assert response["isError"] is True
    message = response["content"][0]["text"]
    assert "end citation must identify original evidence" in message and "event 3" in message
    bounded = _assessment(row["claim_assessments"], "unchecked_prefix")
    bounded["review_interval_id"] = "invented-after-boundary-error"
    result = _parse(review, row)
    assert result["n_errors"] == 1 and result["n_yes"] == 0
    assert "issued review_interval_id" in result["verdicts"][0]["validation_error"]


def test_scope_repair_preserves_completed_interval_without_reading_unrelated_tail(tmp_path):
    review, row, receipt, _citation = _deletion_review(tmp_path)
    row["scope"] = "whole_trace"
    before = list(review.reads)
    failed = _parse(review, row)
    error = failed["verdicts"][0]["validation_error"]
    assert "whole-trace verdict has unread content" in error
    prompt = claude_judge._verdict_repair_prompt(review, [error])
    assert "No frozen requirement demands whole_trace" in prompt
    # Only the reviewer's unnecessarily broad declaration changes. The frozen
    # claim, original receipt, citations and read records remain identical.
    row["scope"] = "bounded_interval"
    result = _parse(review, row)
    assert result["n_yes"] == 1 and result["n_errors"] == 0
    assert review.reads == before
    bounded = _assessment(result["verdicts"][0]["claim_assessments"], "unchecked_prefix")
    assert bounded["review_interval_id"] == receipt["interval_id"]


def test_bounded_scope_cannot_bypass_or_repair_away_a_whole_trace_clause(tmp_path):
    review, row, _receipt, _citation = _deletion_review(tmp_path, whole_trace=True)
    row["scope"] = "whole_trace"
    result = _parse(review, row)
    error = result["verdicts"][0]["validation_error"]
    prompt = claude_judge._verdict_repair_prompt(review, [error])
    assert "Frozen whole-trace requirements: never_restored" in prompt
    assert "No frozen requirement demands" not in prompt
    navigation = json.loads(prompt[prompt.index('{"incomplete_requirements"'):])
    assert navigation["whole_trace_unread_ranges"] == review.unread_ranges(
        review._whole_trace_ranges())
    row["scope"] = "bounded_interval"
    result = _parse(review, row)
    assert result["n_yes"] == 0 and result["n_errors"] == 1
    assert "require scope=whole_trace" in result["verdicts"][0]["validation_error"]
