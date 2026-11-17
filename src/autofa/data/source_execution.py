# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain imported execution records independently of conversation step views."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from copy import deepcopy
from typing import Any

from autofa.data.execution_policy import (
    is_evaluation_record,
    is_evaluation_table,
    sanitize_metadata,
)
from autofa.data.source_completeness import EVENT_CONTENT_LIMITATION, event_content_unavailable
from autofa.data.structured import decode_container

_COLLECTIONS = (
    "events", "tool_calls", "attempts", "artifacts", "tool_definitions", "provenance", "relations",
    "prompt_messages", "prompt_summaries",
)
# Captured requests name pooled messages and tools by hash (EventText references).
_REQUEST_POOLS = {
    "prompt_messages": ("request_messages", "msg_hash"),
    "tool_definitions": ("request_tools", "sha256"),
}
_WITHHELD = " Evaluation records were withheld."


def _request_hashes(events: list[dict], field: str) -> set[str]:
    found: set[str] = set()
    for event in events:
        captured = event.get("event_text") if isinstance(event, dict) else None
        values = decode_container(captured.get(field)) if isinstance(captured, dict) else None
        if isinstance(values, list):
            found.update(value for value in values if isinstance(value, str))
    return found


def _withheld_requests(
    events: list[dict], withheld: Callable[[Any], bool],
) -> dict[str, set[str]]:
    """Pooled request entries that only withheld events sent, by pool and hash."""
    removed: list[dict] = []
    kept: list[dict] = []
    for event in events:
        (removed if withheld(event) else kept).append(event)
    if not removed:
        return {}
    return {name: _request_hashes(removed, field) - _request_hashes(kept, field)
            for name, (field, _) in _REQUEST_POOLS.items()}


def _pooled(name: str, row: Any, withheld: dict[str, set[str]]) -> bool:
    key = row.get(_REQUEST_POOLS[name][1]) if name in withheld and isinstance(row, dict) else None
    return isinstance(key, str) and key in withheld[name]


def execution_record(
    metadata: dict, events: list[dict], calls: list[dict], *, execution: dict | None = None,
) -> dict[str, Any]:
    """Keep provider extensions and distinguish unavailable resources from recorded emptiness."""
    if execution is None:
        execution = {
            "rollout": metadata, "events": events, "tool_calls": calls,
            "availability": {
                "rollout": {"status": "available", "reason": "Rollout metadata retrieved."},
                "events": {"status": "partial", "reason": "Only the conversation lane was read."},
                "tool_calls": {
                    "status": "partial", "reason": "Only the conversation lane was read.",
                },
            },
        }
    result = {"rollout": sanitize_metadata(execution.get("rollout", metadata), rollout=True)}
    availability = deepcopy(execution.get("availability", {}))
    # Calls may omit the actor even when their parent event identifies a grader.
    excluded_events = {
        (row.get("lane") or "", row.get("event_idx"))
        for row in execution.get("events", []) if is_evaluation_record(row)
    }
    # Pooled request messages and tools that only excluded events sent go with them.
    withheld = _withheld_requests(execution.get("events", []), is_evaluation_record)
    # A lane-less artifact is unambiguous when exactly one execution event part
    # references it by index and no evaluation event part does.
    part_lanes: dict[tuple[Any, Any, Any], set[tuple[str, bool]]] = {}
    for event in execution.get("events", []):
        for part in event.get("parts") or []:
            if isinstance(part, dict) and type(part.get("artifact_idx")) is int:
                key = (event.get("event_idx"), part.get("part_idx"), part["artifact_idx"])
                part_lanes.setdefault(key, set()).add(
                    (event.get("lane") or "", is_evaluation_record(event))
                )

    def referenced_lane(row: dict) -> str | None:
        key = (row.get("event_idx"), row.get("part_idx"), row.get("artifact_idx"))
        owners = part_lanes.get(key)
        if not owners or len(owners) != 1:
            return None
        lane, evaluation = next(iter(owners))
        return None if evaluation else lane

    def has_evaluation_parent(row: dict) -> bool:
        if type(row.get("event_idx")) is not int:
            return False
        lane = row.get("lane")
        if lane is None:
            lane = referenced_lane(row)
        parents = [identity for identity in excluded_events
                   if identity[1] == row["event_idx"]
                   and (lane is None or lane == identity[0])]
        return bool(parents)

    for name in _COLLECTIONS:
        rows = execution.get(name, [])
        if name == "artifacts":
            rows = [
                {**row, "lane": referenced_lane(row)}
                if row.get("lane") is None and referenced_lane(row) is not None
                else row
                for row in rows
            ]
        selected = [row for row in rows if not is_evaluation_record(row)
                    and not (name == "artifacts" and has_evaluation_parent(row))
                    and not (name == "tool_calls" and
                             (row.get("lane") or "", row.get("event_idx")) in excluded_events)
                    and not _pooled(name, row, withheld)]
        result[name] = sanitize_metadata(selected)
        if name == "events":
            for event in result[name]:
                _sanitize_snapshot(event)
        availability.setdefault(name, {
            "status": "unavailable", "reason": "This resource was not included in the import.",
        })
        if len(selected) != len(rows):
            reason = availability[name].get("reason")
            availability[name]["reason"] = (
                f"{reason or ''}{_WITHHELD}".strip()
            )
            if name == "artifacts" and any(has_evaluation_parent(row) for row in rows):
                availability[name] = {
                    "status": "partial",
                    "reason": "Artifacts linked to evaluation events or ambiguous evaluation "
                              "parents were withheld; execution artifacts remain available.",
                }
    unavailable_content = any(event_content_unavailable(event) for event in result["events"])
    if unavailable_content:
        availability["events"] = {
            "status": "partial",
            "reason": EVENT_CONTENT_LIMITATION,
        }
    if any(row.get("truncated") for row in result["artifacts"]):
        availability["artifacts"] = {
            "status": "partial", "reason": "The source reports truncated artifact content.",
        }
    extra = execution.get("extra_resources", {})
    result["extra_resources"] = {
        name: sanitize_metadata(rows) for name, rows in extra.items()
        if not is_evaluation_table(name)
    }
    availability.setdefault("extra_resources", {
        "status": "unavailable", "reason": "This import did not inspect additional source tables.",
    })
    result["availability"] = availability
    return result


def withhold_model_calls(execution: dict, call_ids: Collection[str]) -> set[str]:
    """Withhold the exported ``model_call`` events of these calls; return the ids withheld.

    An event export names a model call by the capture's ``model_call_id``
    (``source_event_id``) but not by whose call it was. Their tool calls, and the
    request messages and tools only they sent, are withheld with them.
    """
    events = execution.get("events")
    if not call_ids or not isinstance(events, list):
        return set()

    def withheld_event(event: Any) -> bool:
        return (isinstance(event, dict)
                and "model_call" in (event.get("kind_raw"), event.get("lane_kind"))
                and isinstance(event.get("source_event_id"), str)
                and event["source_event_id"] in call_ids)

    removed = [event for event in events if withheld_event(event)]
    if not removed:
        return set()
    withheld = _withheld_requests(events, withheld_event)
    identities = {(event.get("lane") or "", event.get("event_idx")) for event in removed}

    def withheld_row(name: str, row: Any) -> bool:
        if name == "events":
            return withheld_event(row)
        if name in _REQUEST_POOLS:
            return _pooled(name, row, withheld)
        return (isinstance(row, dict)
                and (row.get("lane") or "", row.get("event_idx")) in identities)

    availability = execution.get("availability")
    for name in ("events", "tool_calls", "artifacts", *_REQUEST_POOLS):
        rows = execution.get(name)
        if not isinstance(rows, list):
            continue
        kept = [row for row in rows if not withheld_row(name, row)]
        if len(kept) == len(rows):
            continue
        execution[name] = kept
        status = availability.get(name) if isinstance(availability, dict) else None
        if (isinstance(status, dict) and isinstance(status.get("reason"), str)
                and not status["reason"].endswith(_WITHHELD)):
            status["reason"] += _WITHHELD
    return {event["source_event_id"] for event in removed}


def _sanitize_snapshot(event: dict) -> None:
    """Filter known harness capture envelopes without interpreting conversation text."""
    if event.get("kind") != "other" or event.get("role") != "harness":
        return
    for part in event.get("parts", []):
        text = part.get("text")
        if not isinstance(text, str):
            continue
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            continue
        if not isinstance(value, dict) or not {
            "message_ids", "messages_first_seen_here",
        }.issubset(value):
            continue
        clean = sanitize_metadata(value)
        if clean != value:
            part["text"] = json.dumps(clean, ensure_ascii=False)
            part["source_chars"] = part.get("chars")
            part["chars"] = len(part["text"])
            part["metadata_filtered"] = True
            if isinstance(part.get("text_head"), str):
                part["text_head"] = part["text"][:len(part["text_head"])]
