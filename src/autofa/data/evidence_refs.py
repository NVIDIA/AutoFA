# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless references from imported step views to canonical execution evidence.

Only the importer-defined event identity and verified source-part offsets justify
sharing content. Equal text in unrelated events is never a reference. Legacy and
unknown envelopes retain their representation; explicit invalid refs fail closed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

EVIDENCE_REFS_VERSION = 1
_MARKER = "_autofa_evidence_refs"
_SOURCE = "original API event parts and tool arguments"
_TEXT_FIELDS = frozenset({"message", "observation", "reasoning_content"})
_EVENT_FIELDS = _TEXT_FIELDS | {"metadata", "parts"}
_CALL_INLINE = frozenset({"function_name", "arguments", "source_call_index", "result_event_idx"})


def _same(left: Any, right: Any) -> bool:
    """JSON values must match types too: True and 1 are different evidence."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(value, right[key])
                                                  for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _unique(rows: Any, *, calls: bool = False) -> dict[tuple, dict]:
    """Index explicit source identities, withholding every ambiguous identity."""
    result, ambiguous = {}, set()
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict):
            continue
        lane, event = row.get("lane"), row.get("event_idx")
        call = row.get("call_idx")
        if not isinstance(lane, str) or not _integer(event) or (calls and not _integer(call)):
            continue
        key = (lane, event, call) if calls else (lane, event)
        if key in result:
            ambiguous.add(key)
        result[key] = row
    return {key: row for key, row in result.items() if key not in ambiguous}


def _scope(envelope: dict) -> str | None:
    scope = envelope.get("evidence_scope")
    if (isinstance(scope, dict) and scope.get("source") == _SOURCE
            and not scope.get("trajectory_source") and isinstance(scope.get("event_lane"), str)):
        return scope["event_lane"]
    return None


def _part_text(step: dict, event: dict, field: str) -> str:
    """Rebuild a joined field only from complete, contiguous source-part mappings."""
    parts, mappings = event.get("parts"), step.get("source_parts")
    if not isinstance(parts, list) or not isinstance(mappings, list):
        raise ValueError("missing source parts for evidence reference")
    by_id = {}
    for part in parts:
        if not isinstance(part, dict) or not _integer(part.get("part_idx")):
            raise ValueError("invalid source part identity in evidence reference")
        if part["part_idx"] in by_id:
            raise ValueError("ambiguous source part identity in evidence reference")
        by_id[part["part_idx"]] = part
    selected = [mapping for mapping in mappings
                if isinstance(mapping, dict) and mapping.get("field") == field]
    if not selected:
        raise ValueError("missing source-part mapping for evidence reference")
    for mapping in selected:
        if not all(_integer(mapping.get(key)) for key in ("part_idx", "start", "end")):
            raise ValueError("invalid source-part offsets in evidence reference")
    selected.sort(key=lambda mapping: mapping["start"])
    chunks, end, seen = [], 0, set()
    for mapping in selected:
        part = by_id.get(mapping["part_idx"], {})
        text = part.get("text")
        start = end + (2 if chunks else 0)
        if (not isinstance(text, str) or mapping["part_idx"] in seen
                or mapping["start"] != start or mapping["end"] != start + len(text)):
            raise ValueError("noncontiguous source-part mapping in evidence reference")
        chunks.append(text)
        seen.add(mapping["part_idx"])
        end = mapping["end"]
    return "\n\n".join(chunks)


def _event_value(step: dict, event: dict, field: str) -> Any:
    if field == "metadata":
        return {key: value for key, value in event.items() if key != "parts"}
    if field == "parts":
        if "parts" not in event:
            raise ValueError("missing parts in evidence reference")
        return event["parts"]
    return _part_text(step, event, field)


def _call_value(call: dict, field: str) -> Any:
    key = "args_full" if field == "arguments" else field
    if key not in call:
        raise ValueError("missing source call field in evidence reference")
    return call[key]


def compact_trace(envelope: dict, *, include_rendered: bool = False) -> dict:
    """Copy an imported envelope, replacing exact mirrored step fields with refs.

    The default retains the established raw step content/call argument fields.
    Judge snapshots may additionally reference rendered fields, using verified
    source-part joins. Canonical execution and all unmatched extensions remain
    untouched. Expanding the result reproduces the full input dictionary exactly.
    """
    out = expand_trace(envelope)
    lane = _scope(out)
    steps, execution = out.get("steps"), out.get("execution")
    if lane is None or not isinstance(steps, list) or not isinstance(execution, dict):
        return out
    events = _unique(execution.get("events"))
    calls = _unique(execution.get("tool_calls"), calls=True)
    refs = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not _integer(step.get("step_id")):
            continue
        event_idx = step["step_id"]
        event = events.get((lane, event_idx))
        if event is None:
            continue
        ref = {"index": index, "lane": lane, "event_idx": event_idx}
        fields = []
        candidates = ["metadata", "parts"] + (sorted(_TEXT_FIELDS) if include_rendered else [])
        for field in candidates:
            if field not in step:
                continue
            try:
                value = _event_value(step, event, field)
            except ValueError:
                continue
            if _same(step[field], value):
                fields.append(field)
                del step[field]
        if fields:
            ref["fields"] = fields
        call_refs = []
        step_calls = step.get("tool_calls")
        for call_index, call in enumerate(step_calls if isinstance(step_calls, list) else []):
            if not isinstance(call, dict) or not _integer(call.get("source_call_index")):
                continue
            call_idx = call["source_call_index"]
            original = calls.get((lane, event_idx, call_idx))
            if original is None:
                continue
            fields = []
            for field in list(call):
                if field in _CALL_INLINE and not (include_rendered and field == "arguments"):
                    continue
                key = "args_full" if field == "arguments" else field
                if key in original and _same(call[field], original[key]):
                    fields.append(field)
                    del call[field]
            if fields:
                call_refs.append({"index": call_index, "call_idx": call_idx, "fields": fields})
        if call_refs:
            ref["calls"] = call_refs
        if "fields" in ref or call_refs:
            refs.append(ref)
    if refs:
        out[_MARKER] = {"version": EVIDENCE_REFS_VERSION, "steps": refs}
    return out


def _fields(value: Any, allowed: frozenset[str] | None = None) -> list[str]:
    if (not isinstance(value, list) or not value or any(not isinstance(key, str) for key in value)
            or len(set(value)) != len(value)
            or (allowed is not None and not set(value) <= allowed)):
        raise ValueError("invalid fields in evidence reference")
    return value


def _target(rows: Any, index: Any) -> dict:
    if (not isinstance(rows, list) or not _integer(index) or index >= len(rows)
            or not isinstance(rows[index], dict)):
        raise ValueError("invalid target index in evidence reference")
    return rows[index]


def _restore(target: dict, field: str, value: Any) -> None:
    if field in target:
        raise ValueError("evidence reference conflicts with inline target")
    target[field] = deepcopy(value)


def expand_trace(envelope: dict) -> dict:
    """Return an independent full envelope; reject malformed explicit references.

    Expansion is restricted to documented step fields and fields of the explicitly
    identified original tool call. It never follows arbitrary paths or URLs.
    """
    if not isinstance(envelope, dict):
        raise ValueError("evidence envelope must be an object")
    out = deepcopy(envelope)
    if _MARKER not in out:
        return out
    marker = out.pop(_MARKER)
    if (not isinstance(marker, dict) or set(marker) != {"version", "steps"}
            or type(marker["version"]) is not int or marker["version"] != EVIDENCE_REFS_VERSION
            or not isinstance(marker["steps"], list) or not marker["steps"]):
        raise ValueError("invalid evidence reference version or envelope")
    lane, execution = _scope(out), out.get("execution")
    if lane is None or not isinstance(execution, dict):
        raise ValueError("evidence references require an imported source execution")
    events = _unique(execution.get("events"))
    calls = _unique(execution.get("tool_calls"), calls=True)
    seen = set()
    for ref in marker["steps"]:
        if (not isinstance(ref, dict) or not {"index", "lane", "event_idx"} <= ref.keys()
                or not set(ref) <= {"index", "lane", "event_idx", "fields", "calls"}
                or not ("fields" in ref or "calls" in ref)):
            raise ValueError("invalid step evidence reference")
        step = _target(out.get("steps"), ref["index"])
        if (ref["index"] in seen or ref["lane"] != lane or not _integer(ref["event_idx"])
                or not _integer(step.get("step_id"))
                or step["step_id"] != ref["event_idx"]):
            raise ValueError("conflicting step identity in evidence reference")
        seen.add(ref["index"])
        event = events.get((lane, ref["event_idx"]))
        if event is None:
            raise ValueError("missing or ambiguous source event in evidence reference")
        if "fields" in ref:
            for field in _fields(ref["fields"], _EVENT_FIELDS):
                _restore(step, field, _event_value(step, event, field))
        if "calls" not in ref:
            continue
        call_refs, seen_calls = ref["calls"], set()
        if not isinstance(call_refs, list) or not call_refs:
            raise ValueError("invalid call evidence references")
        for call_ref in call_refs:
            if not isinstance(call_ref, dict) or set(call_ref) != {"index", "call_idx", "fields"}:
                raise ValueError("invalid call evidence reference")
            call = _target(step.get("tool_calls"), call_ref["index"])
            if (call_ref["index"] in seen_calls or not _integer(call_ref["call_idx"])
                    or not _integer(call.get("source_call_index"))
                    or call["source_call_index"] != call_ref["call_idx"]):
                raise ValueError("conflicting call identity in evidence reference")
            seen_calls.add(call_ref["index"])
            original = calls.get((lane, ref["event_idx"], call_ref["call_idx"]))
            if original is None:
                raise ValueError("missing or ambiguous source call in evidence reference")
            for field in _fields(call_ref["fields"]):
                if field in _CALL_INLINE - {"arguments"}:
                    raise ValueError("invalid normalized call field in evidence reference")
                _restore(call, field, _call_value(original, field))
    return out
