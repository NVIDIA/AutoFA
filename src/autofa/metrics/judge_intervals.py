# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original-source ranges for exhaustive review of a bounded conversation interval.

This deliberately supports explicit ordered step envelopes only. Normalized search
views, guessed actors, and unorderable auxiliary executions cannot certify absence.
Whole boundary events are retained, including every command in a command batch.
"""

from __future__ import annotations

import json
import math
from typing import Any

from autofa.data.evidence_refs import expand_trace
from autofa.data.input_view import RESPONSE_FIELDS
from autofa.data.source_completeness import event_content_unavailable, generation_cutoff
from autofa.data.trace_navigation import _json_children


def _members(text: str, start: int) -> dict:
    return {key: (value, lo, hi) for key, value, lo, hi in _json_children(text, start)}


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON keys make interval order ambiguous")
    return result


def _limited(record: dict) -> bool:
    """Read schema flags only; words inside command/output text are not flags."""
    return (any(record.get(key) not in (None, False, 0, "")
                for key in ("truncated", "length_truncated", "content_truncated",
                            "missing", "incomplete"))
            or record.get("complete") is False or record.get("prefix_complete") is False)


def _count_mismatch(record: dict, key: str, actual: int) -> bool:
    expected = record.get(key)
    return expected is not None and (type(expected) is not int or expected != actual)


def _check_limitations(limits: Any) -> None:
    """Validate importer declarations before certifying complete recorded intervals."""
    if isinstance(limits, dict):
        labels = []
        for label, count in limits.items():
            if (not isinstance(label, str) or not label.strip()
                    or not isinstance(count, (int, float)) or count < 0
                    or (isinstance(count, float)
                        and (not math.isfinite(count) or not count.is_integer()))):
                raise ValueError("malformed recorded evidence limitations")
            if count:
                labels.append(label)
    elif isinstance(limits, list):
        labels = limits
    else:
        raise ValueError("malformed recorded evidence limitations")
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("malformed recorded evidence limitations")
        label = label.lower()
        # These are source declarations, never words in commands or quoted outputs.
        gap = any(word in label for word in ("truncat", "missing", "omitted", "gap",
                                             "incomplete"))
        event_gap = (any(word in label for word in ("event", "command", "trajectory", "trace",
                                                    "step", "tool call", "tool_call"))
                     and any(word in label for word in ("unavailable", "excluded", "unsupported")))
        if gap or event_gap:
            raise ValueError("recorded evidence limitations prevent complete interval coverage")


def _incomplete_call(record: dict) -> bool:
    metadata = record.get("metadata", {})
    if (_limited(record) or not isinstance(metadata, dict) or _limited(metadata)):
        return True
    for source in (record, metadata):
        expected = source.get("args_chars")
        if expected is None:
            continue
        arguments = record.get("args_full", record.get("arguments"))
        if (type(expected) is not int or expected < 0 or not isinstance(arguments, str)
                or len(arguments) != expected):
            return True
    return False


def _incomplete_event(
    record: dict, *, inline_calls: bool = True, verified_cutoff: bool = False,
) -> bool:
    metadata, parts = record.get("metadata", {}), record.get("parts", [])
    def limited(value):
        return _limited({key: item for key, item in value.items() if key != "length_truncated"}
                        if verified_cutoff else value)

    if (limited(record) or not isinstance(metadata, dict) or limited(metadata)
            or record.get("has_encrypted_reasoning") or metadata.get("has_encrypted_reasoning")
            or not isinstance(parts, list) or _count_mismatch(record, "n_parts", len(parts))
            or _count_mismatch(metadata, "n_parts", len(parts))
            or any(_incomplete_part(part) for part in parts)):
        return True
    if not inline_calls:
        return False
    calls = record.get("tool_calls", [])
    return (not isinstance(calls, list)
            or _count_mismatch(record, "n_tool_calls", len(calls))
            or _count_mismatch(metadata, "n_tool_calls", len(calls))
            or any(not isinstance(call, dict) or _incomplete_call(call) for call in calls))


def _incomplete_part(part: Any) -> bool:
    if not isinstance(part, dict) or _limited(part):
        return True
    text, chars = part.get("text"), part.get("chars")
    if part.get("part_type", part.get("type")) in {"text", "output_text"}:
        if not isinstance(text, str):
            return True
    # A declared length requires the corresponding retained text; otherwise an
    # absent/null text field can silently certify the event as fully reviewed.
    return chars is not None and (
        type(chars) is not int or chars < 0 or not isinstance(text, str) or len(text) != chars
    )


def _context_capture(event: dict) -> bool:
    if event.get("is_copied_context") is True:
        return True
    parts = event.get("parts")
    if (event.get("kind") == "other" and event.get("kind_raw") == "model_call"
            and event.get("lane_kind") == "model_call" and event.get("role") is None
            and not event.get("n_tool_calls") and isinstance(parts, list) and parts
            and all(isinstance(part, dict) and part.get("origin") == "request"
                    for part in parts)):
        # Importer-declared request telemetry/context, not an assistant execution.
        # The lane name alone is insufficient to identify these records.
        return True
    if event.get("kind") != "other" or event.get("role") != "harness":
        return False
    if not isinstance(parts, list) or not parts:
        return False
    for part in parts:
        try:
            capture = json.loads(part.get("text", ""))
        except (ValueError, AttributeError, TypeError):
            return False
        if not isinstance(capture, dict) or not {
            "message_ids", "messages_first_seen_here",
        } <= capture.keys():
            return False
    return True


def interval_navigation(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return exact stored ranges, or an explicit inability to certify order.

    The interval covers recorded conversation events, not file contents or model
    request pools (which repeat context rather than record executed actions).
    Those remain available to the judge for the other causal requirements.
    """
    try:
        return _navigation(snapshot)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return {"available": False, "reason": str(exc)}


def _verified_generation_cutoffs(snapshot: dict, envelope: dict) -> set[tuple[str, int]]:
    """Accept provider generation cutoffs only against a complete original inventory.

    No source byte or snapshot identity is rewritten. This narrow exception needs
    provider finish metadata, exact retained parts and source counts. It never
    treats a bare truncation declaration as evidence that nothing was omitted.
    """
    from autofa.metrics.judge_structure import trace_structure

    execution = envelope.get("execution")
    if not isinstance(execution, dict):
        return set()
    events = execution.get("events", [])
    if (not isinstance(events, list) or not events
            or any(not isinstance(event, dict) or event_content_unavailable(event)
                   for event in events)):
        return set()
    cutoffs = {(event.get("lane"), event.get("event_idx"))
               for event in events if generation_cutoff(event)}
    if not cutoffs:
        return set()
    structure = trace_structure(snapshot["fields"], {
        key for key, value in snapshot["manifest"].items() if value["origin"] == "stored"
    })
    return cutoffs if structure and structure.get("available") else set()


def _navigation(snapshot: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    fields, manifest = snapshot["fields"], snapshot["manifest"]
    candidates = [(key, fields[key]) for key in RESPONSE_FIELDS
                  if key in fields and manifest[key]["origin"] == "stored" and fields[key]]
    if not candidates or len({value for _, value in candidates}) != 1:
        raise ValueError("a single unambiguous stored ordered trace is required")
    field, text = candidates[0]
    envelope = json.loads(text, object_pairs_hook=_unique_object)
    start = len(text) - len(text.lstrip())
    if isinstance(envelope, list) and len(envelope) == 1:
        _, envelope, start, _ = next(_json_children(text, start))
    if not isinstance(envelope, dict) or not isinstance(envelope.get("steps"), list):
        raise ValueError("bounded coverage requires an explicit ordered steps envelope")
    if _limited(envelope):
        raise ValueError("stored trace is incomplete or truncated")
    expanded = expand_trace(envelope)  # Validate every reference, even outside this interval.
    steps = expanded["steps"]
    if not steps or any(not isinstance(step, dict) for step in steps):
        raise ValueError("missing or malformed ordered steps")
    scope = envelope.get("evidence_scope", {})
    if not isinstance(scope, dict) or _limited(scope) or scope.get("trajectory_source"):
        raise ValueError("trace scope does not establish a complete ordered conversation")
    cutoffs = _verified_generation_cutoffs(snapshot, expanded)
    _check_limitations(scope.get("limitations", {}))
    members = _members(text, start)
    events = []
    for index, _step, lo, hi in _json_children(text, members["steps"][1]):
        events.append({"event_index": index + 1,
                       "source": steps[index].get("source", "unknown"),
                       "ranges": [{"field": field, "start": lo, "end": hi}],
                       "complete": not _incomplete_event(
                           steps[index], verified_cutoff=(scope.get("event_lane"),
                                                          steps[index].get("step_id")) in cutoffs)})
    execution = expanded.get("execution")
    if execution is not None:
        if (not isinstance(execution, dict)
                or scope.get("source") != "original API event parts and tool arguments"
                or not isinstance(scope.get("event_lane"), str)):
            raise ValueError("execution records have no verified conversation identity")
        availability = execution.get("availability", {})
        if not isinstance(availability, dict):
            raise ValueError("malformed execution availability")
        for name in ("events", "tool_calls"):
            resource = availability.get(name, {})
            if not isinstance(resource, dict):
                raise ValueError(f"malformed {name} availability")
            if resource.get("status") not in (None, "available"):
                raise ValueError(f"{name} were not completely retrieved")
        ids = [step.get("step_id") for step in steps]
        if (any(type(value) is not int or value < 0 for value in ids)
                or ids != sorted(set(ids))):
            raise ValueError("step source identities are missing, repeated or out of order")
        by_id = dict(zip(ids, events, strict=True))
        resources = _members(text, members["execution"][1])
        seen = set()
        original_events, call_counts, call_ids = {}, {}, set()
        for name in ("events", "tool_calls"):
            if name not in resources:
                continue
            if not isinstance(resources[name][0], list):
                raise ValueError(f"malformed execution {name}")
            for _, record, lo, hi in _json_children(text, resources[name][1]):
                if not isinstance(record, dict):
                    raise ValueError(f"malformed execution {name} record")
                if name == "events" and _context_capture(record):
                    continue
                event_id, lane = record.get("event_idx"), record.get("lane")
                if (lane != scope["event_lane"] or type(event_id) is not int
                        or event_id not in by_id):
                    raise ValueError("execution actions fall outside the verified ordered steps")
                if name == "events":
                    if event_id in seen:
                        raise ValueError("ambiguous original event identity")
                    seen.add(event_id)
                    original_events[event_id] = record
                else:
                    call_counts[event_id] = call_counts.get(event_id, 0) + 1
                    call_id = record.get("call_idx")
                    if call_id is not None:
                        if (type(call_id) is not int or call_id < 0
                                or (event_id, call_id) in call_ids):
                            by_id[event_id]["complete"] = False
                        else:
                            call_ids.add((event_id, call_id))
                event = by_id[event_id]
                event["ranges"].append({"field": field, "start": lo, "end": hi})
                incomplete = (_incomplete_event(
                    record, inline_calls=False, verified_cutoff=(lane, event_id) in cutoffs,
                ) if name == "events"
                              else _incomplete_call(record))
                if incomplete:
                    event["complete"] = False
        if seen != set(ids):
            raise ValueError("some ordered steps have no original execution event")
        for event_id, record in original_events.items():
            if _count_mismatch(record, "n_tool_calls", call_counts.get(event_id, 0)):
                by_id[event_id]["complete"] = False
    return {
        "available": True, "field": field, "event_count": len(events), "events": events,
        **({"generation_cutoff_events": sorted(index for lane, index in cutoffs
                                                if lane == scope["event_lane"]),
            "generation_cutoff_note": "Provider generation ended before normal completion; "
                "the original event inventory and retained text parts are complete. "
                "This verifies recorded content only; termination cause and causal claims "
                "still require their own evidence."} if cutoffs else {}),
        "coverage": "Complete recorded conversation events, including whole boundary batches. "
                    "Request-context pools and external artifacts are not executions; inspect "
                    "them separately when needed for requirement, recovery or causal evidence.",
    }


def merge_ranges(ranges: list[dict]) -> list[dict]:
    merged = []
    for span in sorted(ranges, key=lambda row: (row["field"], row["start"], row["end"])):
        if (merged and merged[-1]["field"] == span["field"]
                and span["start"] <= merged[-1]["end"] + 2):
            # Adjacent JSON records are separated by a comma and optional space.
            # Reading those extra bytes permits sequential paging instead of one
            # tool call per small event; it never removes any required content.
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged
