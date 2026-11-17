# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered overlapping windows for an oversized judgement context."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .trace_digest import (
    DEFAULT_JUDGEMENT_CHARACTER_BUDGET,
    build_trace_judgement_context,
    serialized_characters,
)

DEFAULT_OVERLAP_UNITS = 1
EVENT_CHUNK_OVERLAP_CHARACTERS = 256
_FIT_SENTINEL = 9_999
_HASH_SENTINEL = "0" * 64


class TraceWindowingError(ValueError):
    """An atomic trace unit cannot fit in one judgement window."""


def build_trace_judgement_windows(
    digest: dict[str, Any],
    *,
    max_serialized_characters: int = DEFAULT_JUDGEMENT_CHARACTER_BUDGET,
) -> list[dict[str, Any]]:
    """Split oversized context into the fewest fitting windows."""
    if max_serialized_characters < 1:
        raise ValueError("max_serialized_characters must be positive")
    context = build_trace_judgement_context(digest)
    if serialized_characters(context) <= max_serialized_characters:
        return [context]
    events = context.get("ordered_trace")
    if not isinstance(events, list) or not events:
        raise TraceWindowingError("oversized judgement context has no ordered events")
    events, chunk_sources = _chunk_oversized_events(
        events,
        max_serialized_characters=max_serialized_characters,
    )

    units = _atomic_units(events)
    anchor_unit_indexes = _anchor_unit_indexes(units)
    task_anchor = _flatten(
        unit for index, unit in enumerate(units) if index in anchor_unit_indexes["task"]
    )
    terminal_anchor = _flatten(
        unit for index, unit in enumerate(units) if index in anchor_unit_indexes["terminal"]
    )
    local_units = [
        unit for index, unit in enumerate(units) if index not in anchor_unit_indexes["all"]
    ]
    auxiliary_lanes = context.get("auxiliary_lanes") or []
    trace_resources = context.get("trace_resources") or {}
    base = {
        "coverage": context.get("coverage"),
        "source_judgement_characters": serialized_characters(context),
        "task_anchor": task_anchor,
        "terminal_anchor": terminal_anchor,
        "failure_index": _failure_index(context.get("error_extract")),
        "outcome_evidence": context.get("outcome_evidence") or {},
        "verifier_artifacts": context.get("verifier_artifacts") or [],
        "_event_chunk_sources": chunk_sources,
    }
    drafts: list[tuple[list[dict[str, Any]], list[int], dict[str, Any]]] = []
    if not local_units:
        drafts.append(([], [], {}))
    else:
        try:
            segments = _optimal_segments(
                base,
                local_units,
                max_serialized_characters=max_serialized_characters,
                overlap_units=DEFAULT_OVERLAP_UNITS,
            )
        except TraceWindowingError:
            segments = _optimal_segments(
                base,
                local_units,
                max_serialized_characters=max_serialized_characters,
                overlap_units=0,
            )
        previous_event_refs: set[tuple[str, int]] = set()
        for segment in segments:
            local_events = _flatten(segment)
            event_refs = set(_lane_event_refs(local_events))
            overlap_event_indexes = sorted(
                event_idx for _, event_idx in previous_event_refs & event_refs
            )
            drafts.append((local_events, overlap_event_indexes, {}))
            previous_event_refs = event_refs

    for lane in auxiliary_lanes:
        if not isinstance(lane, dict):
            continue
        lane_events = lane.get("ordered_trace")
        if not isinstance(lane_events, list) or not lane_events:
            continue
        expanded, sources = _chunk_oversized_events(
            lane_events,
            max_serialized_characters=max_serialized_characters,
        )
        if sources:
            base["_event_chunk_sources"].extend(sources)
        metadata = {key: value for key, value in lane.items() if key != "ordered_trace"}
        for selected in _pack_units(
            _atomic_units(expanded),
            lambda rows, metadata=metadata: {
                "auxiliary_lanes": [{**metadata, "ordered_trace": _flatten(rows)}],
            },
            base=base,
            max_serialized_characters=max_serialized_characters,
        ):
            drafts.append((
                [],
                [],
                {"auxiliary_lanes": [{**metadata, "ordered_trace": _flatten(selected)}]},
            ))

    resource_units = _resource_window_units(
        trace_resources,
        base=base,
        max_serialized_characters=max_serialized_characters,
    )
    for selected in _pack_units(
        resource_units,
        _resource_window_extra,
        base=base,
        max_serialized_characters=max_serialized_characters,
    ):
        drafts.append(([], [], _resource_window_extra(selected)))

    windows: list[dict[str, Any]] = []
    for index, (local_events, overlap_event_indexes, extra) in enumerate(drafts):
        window = _finalize_window(
            base,
            local_events,
            index=index,
            count=len(drafts),
            overlap_event_indexes=overlap_event_indexes,
            extra=extra,
        )
        if serialized_characters(window) > max_serialized_characters:
            raise TraceWindowingError(
                f"final window {index} exceeds budget after metadata was added"
            )
        windows.append(window)
    return windows


def _optimal_segments(
    base: dict[str, Any],
    units: list[list[dict[str, Any]]],
    *,
    max_serialized_characters: int,
    overlap_units: int,
) -> list[list[list[dict[str, Any]]]]:
    """Split units into the fewest fitting windows."""
    segments: list[list[list[dict[str, Any]]]] = []
    start = 0
    while start < len(units):
        overlap_indexes = (
            _event_indexes(_flatten(units[start : start + overlap_units]))
            if start > 0 and overlap_units
            else []
        )
        selected_end = None
        for end in range(start + 1, len(units) + 1):
            candidate = _window_payload(
                base,
                _flatten(units[start:end]),
                index=_FIT_SENTINEL,
                count=_FIT_SENTINEL,
                overlap_event_indexes=overlap_indexes,
                window_sha256=_HASH_SENTINEL,
            )
            if serialized_characters(candidate) > max_serialized_characters:
                break
            selected_end = end
        if selected_end is None:
            event_indexes = _event_indexes(units[start])
            raise TraceWindowingError(
                "no window segmentation fits the budget; "
                f"atomic unit {start} event_indexes={event_indexes}"
            )
        if selected_end < len(units) and selected_end - start <= overlap_units:
            raise TraceWindowingError(
                f"window overlap prevents progress from atomic unit {start}"
            )
        segments.append(units[start:selected_end])
        start = (
            selected_end
            if selected_end == len(units)
            else selected_end - overlap_units
        )
    return segments


def _pack_units(
    units: list[list[dict[str, Any]]],
    extra_builder: Any,
    *,
    base: dict[str, Any],
    max_serialized_characters: int,
) -> list[list[list[dict[str, Any]]]]:
    """Greedily pack non-primary evidence units into bounded windows."""
    packed: list[list[list[dict[str, Any]]]] = []
    start = 0
    while start < len(units):
        selected_end = None
        for end in range(start + 1, len(units) + 1):
            candidate = _window_payload(
                base,
                [],
                index=_FIT_SENTINEL,
                count=_FIT_SENTINEL,
                overlap_event_indexes=[],
                window_sha256=_HASH_SENTINEL,
                extra=extra_builder(units[start:end]),
            )
            if serialized_characters(candidate) > max_serialized_characters:
                break
            selected_end = end
        if selected_end is None:
            raise TraceWindowingError(
                f"non-primary evidence unit {start} exceeds the window budget"
            )
        packed.append(units[start:selected_end])
        start = selected_end
    return packed


def _resource_window_units(
    resources: Any,
    *,
    base: dict[str, Any],
    max_serialized_characters: int,
) -> list[list[dict[str, Any]]]:
    if not isinstance(resources, dict):
        return []
    units: list[list[dict[str, Any]]] = []
    target = max(128, min(8_000, max_serialized_characters // 16))
    for resource, rows in sorted(resources.items()):
        if not isinstance(rows, list):
            continue
        for row_index, row in enumerate(rows):
            unit = [{"resource": str(resource), "row": row}]
            candidate = _window_payload(
                base,
                [],
                index=_FIT_SENTINEL,
                count=_FIT_SENTINEL,
                overlap_event_indexes=[],
                window_sha256=_HASH_SENTINEL,
                extra=_resource_window_extra([unit]),
            )
            if serialized_characters(candidate) <= max_serialized_characters:
                units.append(unit)
                continue
            source = json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            source_sha256 = hashlib.sha256(source.encode()).hexdigest()
            ranges = [
                (start, min(len(source), start + target))
                for start in range(0, len(source), target)
            ]
            for chunk_index, (start, end) in enumerate(ranges):
                units.append([{
                    "resource": str(resource),
                    "row": {
                        "resource_idx": (
                            row.get("resource_idx", row_index)
                            if isinstance(row, dict)
                            else row_index
                        ),
                        "resource_chunk": {
                            "encoding": "canonical_json",
                            "source_sha256": source_sha256,
                            "source_characters": len(source),
                            "index": chunk_index,
                            "count": len(ranges),
                            "character_start": start,
                            "character_end": end,
                            "sha256": hashlib.sha256(source[start:end].encode()).hexdigest(),
                        },
                        "value": source[start:end],
                    },
                }])
    return units


def _resource_window_extra(
    units: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for item in _flatten(units):
        grouped.setdefault(str(item["resource"]), []).append(item["row"])
    return {"trace_resources": grouped}


def _atomic_units(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Keep each tool call with its result."""
    chunked_event_indexes = {
        event_idx
        for event in events
        if event.get("event_chunk")
        for event_idx in [(event.get("event_ref") or {}).get("event_idx")]
        if isinstance(event_idx, int)
    }

    def linked_result(event: dict[str, Any]) -> int | None:
        calls = event.get("tool_calls") or []
        if any(
            isinstance(call, dict)
            and call.get("result_event_idx") in chunked_event_indexes
            for call in calls
        ):
            return None
        return _last_linked_result(event)

    units: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(events):
        last_result = linked_result(events[index])
        cursor = index + 1
        while cursor < len(events) and last_result is not None:
            event_idx = (events[cursor].get("event_ref") or {}).get("event_idx")
            if isinstance(event_idx, int) and event_idx > last_result:
                break
            nested = linked_result(events[cursor])
            if nested is not None:
                last_result = max(last_result, nested)
            cursor += 1
        units.append(events[index:cursor])
        index = cursor
    return units


def _last_linked_result(event: dict[str, Any]) -> int | None:
    own_idx = (event.get("event_ref") or {}).get("event_idx")
    results = [
        call["result_event_idx"]
        for call in event.get("tool_calls") or []
        if isinstance(call, dict)
        and isinstance(call.get("result_event_idx"), int)
        and (not isinstance(own_idx, int) or call["result_event_idx"] > own_idx)
    ]
    return max(results) if results else None


def _anchor_unit_indexes(
    units: list[list[dict[str, Any]]],
) -> dict[str, set[int]]:
    """Keep the task instructions and final model response visible in every window."""
    refs = [unit[0].get("event_ref") or {} for unit in units]
    task: set[int] = set()
    for index, ref in enumerate(refs):
        chunk = units[index][0].get("event_chunk") or {}
        first_chunk = not chunk or chunk.get("index") == 0
        if ref.get("role") in {"system", "developer"} and first_chunk:
            task.add(index)
        elif (
            ref.get("role") == "user"
            and ref.get("kind") != "tool_result"
            and first_chunk
        ):
            task.add(index)
            break
    task = task or {0}
    model_units = [
        index for index, ref in enumerate(refs) if ref.get("role") in {"agent", "assistant"}
    ]
    terminal = {model_units[-1]} if model_units else set()
    return {"task": task, "terminal": terminal, "all": task | terminal}


def _chunk_oversized_events(
    events: list[dict[str, Any]],
    *,
    max_serialized_characters: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace oversized events with exact overlapping canonical-JSON chunks."""
    threshold = max(512, max_serialized_characters * 3 // 5)
    target = max(128, min(8_000, max_serialized_characters // 16))
    serialized = {
        id(event): json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for event in events
    }
    linked_callers: dict[int, list[dict[str, Any]]] = {}
    oversized_argument_results: dict[int, set[int]] = {}
    argument_context_limit = max(512, min(4_000, max_serialized_characters // 8))
    for caller in events:
        for call in caller.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            result_event_idx = call.get("result_event_idx")
            if not isinstance(result_event_idx, int):
                continue
            call_context = {
                key: call.get(key)
                for key in (
                    "event_idx",
                    "call_idx",
                    "tool_name",
                    "status",
                    "result_event_idx",
                    "scope",
                )
                if call.get(key) is not None
            }
            arguments = call.get("args")
            if arguments is not None:
                argument_source = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if len(argument_source) <= argument_context_limit:
                    call_context["args"] = arguments
                else:
                    oversized_argument_results.setdefault(id(caller), set()).add(
                        result_event_idx
                    )
                    call_context["args_reference"] = {
                        "caller_source_sha256": hashlib.sha256(
                            serialized[id(caller)].encode()
                        ).hexdigest(),
                        "sha256": hashlib.sha256(argument_source.encode()).hexdigest(),
                        "characters": len(argument_source),
                    }
            linked_callers.setdefault(result_event_idx, []).append(
                {
                    "caller_event_ref": dict(caller.get("event_ref") or {}),
                    "call": call_context,
                }
            )
    linked_oversized = {
        id(event)
        for unit in _atomic_units(events)
        if sum(len(serialized[id(event)]) for event in unit) > threshold
        and all(len(serialized[id(event)]) <= threshold for event in unit)
        for event in unit
    }
    chunked_event_indexes = {
        event_idx
        for event in events
        if len(serialized[id(event)]) > threshold or id(event) in linked_oversized
        for event_idx in [(event.get("event_ref") or {}).get("event_idx")]
        if isinstance(event_idx, int)
    }
    linked_argument_callers = {
        caller_id
        for caller_id, result_indexes in oversized_argument_results.items()
        if result_indexes & chunked_event_indexes
    }
    expanded: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for event in events:
        source = serialized[id(event)]
        if (
            len(source) <= threshold
            and id(event) not in linked_oversized
            and id(event) not in linked_argument_callers
        ):
            expanded.append(event)
            continue
        overlap = min(EVENT_CHUNK_OVERLAP_CHARACTERS, max(0, target // 4))
        step = target - overlap
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < len(source):
            end = min(len(source), start + target)
            ranges.append((start, end))
            if end == len(source):
                break
            start += step
        source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        event_ref = dict(event.get("event_ref") or {})
        context = {
            "tool_calls": [
                {
                    key: call.get(key)
                    for key in (
                        "event_idx",
                        "call_idx",
                        "tool_name",
                        "status",
                        "result_event_idx",
                        "scope",
                    )
                    if call.get(key) is not None
                }
                for call in event.get("tool_calls") or []
                if isinstance(call, dict)
            ]
        }
        event_idx = event_ref.get("event_idx")
        if isinstance(event_idx, int) and linked_callers.get(event_idx):
            context["linked_callers"] = linked_callers[event_idx]
        sources.append(
            {
                "event_ref": event_ref,
                "sha256": source_sha256,
                "characters": len(source),
                "chunk_count": len(ranges),
            }
        )
        for index, (start, end) in enumerate(ranges):
            value = source[start:end]
            expanded.append(
                {
                    "event_ref": event_ref,
                    "event_chunk": {
                        "encoding": "canonical_json",
                        "source_sha256": source_sha256,
                        "source_characters": len(source),
                        "index": index,
                        "count": len(ranges),
                        "character_start": start,
                        "character_end": end,
                        "overlap_before": 0 if index == 0 else overlap,
                        "sha256": hashlib.sha256(value.encode()).hexdigest(),
                        "context": context,
                    },
                    "text": {
                        "value": value,
                        "truncated": False,
                        "characters": len(value),
                        "sha256": hashlib.sha256(value.encode()).hexdigest(),
                    },
                    "full_reason": "event_chunk",
                    "tool_calls": [],
                }
            )
    return expanded, sources


def _failure_index(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {
            "tool_name": item.get("tool_name"),
            "event_idxs": item.get("event_idxs") or [item.get("event_idx")],
            "repeat_count": item.get("repeat_count", 1),
        }
        for item in value
        if isinstance(item, dict)
    ]


def _finalize_window(
    base: dict[str, Any],
    local_events: list[dict[str, Any]],
    *,
    index: int,
    count: int,
    overlap_event_indexes: list[int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    window = _window_payload(
        base,
        local_events,
        index=index,
        count=count,
        overlap_event_indexes=overlap_event_indexes,
        window_sha256=None,
        extra=extra,
    )
    rendered = json.dumps(window, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    window["window"]["sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
    return window


def _window_payload(
    base: dict[str, Any],
    local_events: list[dict[str, Any]],
    *,
    index: int,
    count: int,
    overlap_event_indexes: list[int],
    window_sha256: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk_sources = base.get("_event_chunk_sources") or []
    public_base = {key: value for key, value in base.items() if not key.startswith("_")}
    visible_events = [
        *(base.get("task_anchor") or []),
        *(base.get("terminal_anchor") or []),
        *local_events,
        *[
            event
            for lane in (extra or {}).get("auxiliary_lanes", [])
            if isinstance(lane, dict)
            for event in lane.get("ordered_trace", [])
            if isinstance(event, dict)
        ],
    ]
    visible_source_hashes = {
        str(chunk["source_sha256"])
        for event in visible_events
        for chunk in [event.get("event_chunk")]
        if isinstance(chunk, dict) and chunk.get("source_sha256")
    }
    visible_sources = [
        source
        for source in chunk_sources
        if str(source.get("sha256")) in visible_source_hashes
    ]
    auxiliary_events = [
        event
        for lane in (extra or {}).get("auxiliary_lanes", [])
        if isinstance(lane, dict)
        for event in lane.get("ordered_trace", [])
        if isinstance(event, dict)
    ]
    if auxiliary_events:
        event_indexes = _event_indexes(auxiliary_events)
        window_scope = "auxiliary_lane"
        window_lane = str(
            ((extra or {}).get("auxiliary_lanes") or [{}])[0].get("lane") or ""
        )
    elif (extra or {}).get("trace_resources"):
        event_indexes = []
        window_scope = "trace_resources"
        window_lane = None
    else:
        event_indexes = _event_indexes(local_events)
        window_scope = "primary_lane"
        window_lane = str(
            ((local_events or base.get("task_anchor") or [{}])[0].get("event_ref") or {}).get(
                "lane"
            ) or ""
        )
    if not event_indexes and window_scope == "primary_lane":
        event_indexes = sorted(
            {
                *_event_indexes(base.get("task_anchor") or []),
                *_event_indexes(base.get("terminal_anchor") or []),
            }
        )
    window = {
        **public_base,
        **(extra or {}),
        **(
            {
                "event_chunking": {
                    "version": 1,
                    "overlap_characters": EVENT_CHUNK_OVERLAP_CHARACTERS,
                    "sources": visible_sources,
                }
            }
            if visible_sources
            else {}
        ),
        "window": {
            "index": index,
            "count": count,
            "scope": window_scope,
            "lane": window_lane,
            "start_event_idx": event_indexes[0] if event_indexes else None,
            "end_event_idx": event_indexes[-1] if event_indexes else None,
            "overlap_event_indexes": overlap_event_indexes,
        },
        "ordered_trace": local_events,
    }
    if window_sha256 is not None:
        window["window"]["sha256"] = window_sha256
    return window


def _event_indexes(events: list[dict[str, Any]]) -> list[int]:
    return [
        event_idx
        for event in events
        if isinstance(event, dict)
        for event_idx in [(event.get("event_ref") or {}).get("event_idx")]
        if isinstance(event_idx, int)
    ]


def _lane_event_refs(events: list[dict[str, Any]]) -> list[tuple[str, int]]:
    return [
        (str(ref.get("lane") or ""), event_idx)
        for event in events
        if isinstance(event, dict)
        for ref in [event.get("event_ref") or {}]
        for event_idx in [ref.get("event_idx")]
        if isinstance(event_idx, int)
    ]


def _flatten(units: Any) -> list[dict[str, Any]]:
    return [event for unit in units for event in unit]
