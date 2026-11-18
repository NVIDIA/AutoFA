# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified event structure for structural facts, never semantic trace summaries.

An exported reasoning event can contain hundreds of thousands of characters.
Whether a separate assistant message exists is a property of the complete event
inventory, not of the words in that reasoning. Keep that distinction explicit:
this view cannot establish recovery, correctness, termination, or causation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from autofa.data.evidence_refs import expand_trace
from autofa.data.input_view import RESPONSE_FIELDS
from autofa.data.source_completeness import EVENT_CONTENT_LIMITATION, TRUNCATED_EVENT_LIMITATION
from autofa.data.trace_navigation import _verified_source_identity
from autofa.metrics.judge_intervals import (
    _check_limitations,
    _context_capture,
    _limited,
    _unique_object,
)

# EVENT_CONTENT_LIMITATION describes retained event records with incomplete
# *content*, not omitted records. No other partial-availability reason is waived.
_COMPLETE_DECLARATION = "Exported table retrieved completely; evaluation records are excluded."
_CONTENT_ONLY_LIMITS = {
    TRUNCATED_EVENT_LIMITATION,
    "source event has encrypted reasoning",
}
_ROLES = {"user", "system", "developer", "assistant", "output", "tool", "function"}
_KINDS = {"message", "reasoning", "tool_call", "tool_result"}


def _count(value: Any, name: str, actual: int | None = None) -> int:
    if type(value) is not int or value < 0 or (actual is not None and value != actual):
        raise ValueError(f"{name} is missing, malformed, or disagrees with retained records")
    return value


def _limits(scope: dict) -> None:
    limits = scope.get("limitations", {})
    # Verify the complete representation before removing the two known
    # content-only declarations. Other missing/truncated event declarations fail.
    if isinstance(limits, dict):
        for key, value in limits.items():
            _count(value, f"limitation count for {key}")
        limits = {key: value for key, value in limits.items() if key not in _CONTENT_ONLY_LIMITS}
    elif isinstance(limits, list):
        limits = [key for key in limits if key not in _CONTENT_ONLY_LIMITS]
    _check_limitations(limits)


def trace_structure(fields: dict[str, str], stored_fields: set[str]) -> dict | None:  # noqa: C901
    """Return checked structure, or unavailability; never guess from aggregate counts."""
    candidates = [(key, fields[key]) for key in RESPONSE_FIELDS
                  if key in stored_fields and key in fields and fields[key]]
    if not candidates:
        return None
    try:
        if len({value for _, value in candidates}) != 1:
            raise ValueError("multiple different stored response representations")
        field, text = candidates[0]
        if not text.lstrip().startswith(("{", "[")):
            return None
        envelope = json.loads(text, object_pairs_hook=_unique_object)
        if isinstance(envelope, list) and len(envelope) == 1:
            envelope = envelope[0]
        if not isinstance(envelope, dict) or not isinstance(envelope.get("execution"), dict):
            return None  # A flat text has no structural absence certificate.
        if _limited(envelope):
            raise ValueError("the stored envelope is incomplete")
        envelope = expand_trace(envelope)
        execution = envelope["execution"]
        if _limited(execution):
            raise ValueError("execution inventory is declared incomplete")
        scope = envelope.get("evidence_scope")
        if not isinstance(scope, dict) or _limited(scope) or scope.get("trajectory_source"):
            raise ValueError("the source does not establish a complete execution inventory")
        _limits(scope)
        availability = execution.get("availability", {}).get("events", {})
        if not isinstance(availability, dict) or _limited(availability) or not (
            availability.get("status") == "available"
            or (availability.get("status") in {None, "partial"}
                and availability.get("reason") in {_COMPLETE_DECLARATION, EVENT_CONTENT_LIMITATION})
        ):
            raise ValueError("the event inventory is unavailable or only partially imported")
        rollout, events = execution.get("rollout"), execution.get("events")
        if not isinstance(rollout, dict) or not isinstance(events, list):
            raise ValueError("missing original rollout counts or event records")
        if _limited(rollout):
            raise ValueError("the original rollout is declared incomplete")
        conversation_lane = scope.get("event_lane")
        mixed_lanes = (isinstance(conversation_lane, str)
                       and any(isinstance(event, dict)
                               and event.get("lane") != conversation_lane for event in events))
        if mixed_lanes and not _verified_source_identity(scope):
            raise ValueError("multiple execution lanes have no verified conversation identity")
        records, identities = [], set()
        assistant_messages = reasoning_events = 0
        conversation_events = conversation_messages = conversation_reasoning = 0
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("malformed event")
            index = _count(event.get("event_idx"), "event_idx")
            lane = event.get("lane")
            if not isinstance(lane, str) or (lane, index) in identities:
                raise ValueError("missing or ambiguous event identity")
            identities.add((lane, index))
            role, kind = event.get("role"), event.get("kind")
            if event.get("is_copied_context"):
                raise ValueError("copied context does not establish executed message counts")
            request_context = (
                mixed_lanes and lane != conversation_lane and role is None
                and kind == "other" and _context_capture(event)
            )
            if not request_context and (role not in _ROLES or kind not in _KINDS):
                raise ValueError("unsupported event kind or actor; cannot classify all events")
            metadata = event.get("metadata", {})
            parts = event.get("parts")
            if not isinstance(metadata, dict) or not isinstance(parts, list):
                raise ValueError("missing event parts or malformed metadata")
            if any(record.get(key) for record in (event, metadata)
                   for key in ("missing", "incomplete", "truncated")):
                raise ValueError("event structure is declared incomplete")
            if any(record.get(key) is False for record in (event, metadata)
                   for key in ("complete", "prefix_complete")):
                raise ValueError("event structure is declared incomplete")
            if request_context and (_limited(event) or _limited(metadata)):
                raise ValueError("request-context inventory is declared incomplete")
            for record in (event, metadata):
                if record.get("n_tool_calls") is not None:
                    _count(record["n_tool_calls"], "event n_tool_calls", 0)
                if record.get("tool_calls") not in (None, []):
                    raise ValueError("inline tool-call inventory requires original call review")
            _count(event.get("n_parts"), "n_parts", len(parts))
            if metadata.get("n_parts") is not None:
                _count(metadata["n_parts"], "metadata.n_parts", len(parts))
            part_rows, part_ids = [], set()
            for part in parts:
                if not isinstance(part, dict):
                    raise ValueError("malformed event part")
                if _limited(part):
                    raise ValueError("event part is declared incomplete")
                part_id = _count(part.get("part_idx"), "part_idx")
                if part_id in part_ids:
                    raise ValueError("duplicate part identity")
                part_ids.add(part_id)
                if part.get("event_idx") != index or part.get("lane") != lane:
                    raise ValueError("part identity disagrees with its event")
                part_type, channel = part.get("part_type"), part.get("channel")
                if part_type not in {"text", "image", "audio", "video", "file"}:
                    raise ValueError("unsupported part type")
                if channel not in {"content", "reasoning"}:
                    raise ValueError("missing or unsupported part channel")
                if request_context and channel != "content":
                    raise ValueError("request telemetry contains unclassified reasoning")
                if kind == "reasoning" and channel != "reasoning":
                    raise ValueError("a reasoning event also contains non-reasoning content")
                if kind == "reasoning" and role not in {"assistant", "output"}:
                    raise ValueError("a reasoning event has an unsupported actor")
                chars = _count(part.get("chars"), "part chars")
                body = part.get("text")
                if not isinstance(body, str) or len(body) != chars:
                    raise ValueError("declared part text is not fully retained")
                part_rows.append({"part_idx": part_id, "part_type": part_type,
                                  "channel": channel, "chars": chars,
                                  "sha256": hashlib.sha256(body.encode()).hexdigest()})
            if part_ids != set(range(len(parts))):
                raise ValueError("event part identities contain gaps")
            assistant_messages += int(kind == "message" and role in {"assistant", "output"})
            reasoning_events += int(kind == "reasoning")
            if lane == conversation_lane:
                conversation_events += 1
                conversation_messages += int(kind == "message" and role in {"assistant", "output"})
                conversation_reasoning += int(kind == "reasoning")
            if kind == "tool_call":
                raise ValueError("tool-call event semantics are not supported by this inventory")
            records.append({
                "event_idx": index, "lane": lane, "kind": kind, "role": role,
                "parts": part_rows,
                **({"classification": "request_context"} if request_context else {}),
                **{key: event[key] for key in ("finish_reason", "length_truncated",
                                              "has_encrypted_reasoning", "completion_tokens")
                   if event.get(key) is not None},
            })
        # Source aggregates must agree with the original events, rather than
        # serving as unverified substitutes for those events.
        # Importer rollups describe the selected conversation lane; the retained
        # events table additionally includes model-call telemetry and other lanes.
        # Every extra record was still classified and checked above, and assistant
        # messages in other lanes count in the all-event result (never as absence).
        _count(rollout.get("n_events"), "n_events",
               conversation_events if mixed_lanes else len(events))
        _count(rollout.get("n_assistant_turns"), "n_assistant_turns",
               conversation_messages if mixed_lanes else assistant_messages)
        if rollout.get("n_reasoning_segments") is not None:
            _count(rollout["n_reasoning_segments"], "n_reasoning_segments",
                   conversation_reasoning if mixed_lanes else reasoning_events)
        calls = execution.get("tool_calls")
        if not isinstance(calls, list) or calls:
            raise ValueError("tool-bearing execution needs the original call inventory")
        _count(rollout.get("n_tool_calls"), "n_tool_calls", 0)
        return {
            "version": 2 if mixed_lanes else 1,
            "available": True, "event_inventory_complete": True,
            "source_field": field, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "n_events": len(events), "n_assistant_message_events": assistant_messages,
            "n_reasoning_events": reasoning_events, "events": records,
            **({"conversation_lane": conversation_lane,
                "n_conversation_events": conversation_events,
                "n_conversation_assistant_message_events": conversation_messages}
               if mixed_lanes else {}),
            **({"completion_tokens": _count(rollout["completion_tokens"], "completion_tokens")}
               if rollout.get("completion_tokens") is not None else {}),
            "scope": (
                ("Complete retained event structure checked against source counts. "
                 if not mixed_lanes else
                 "Complete retained event structure checked across all exported lanes. "
                 "Source rollup counts match the selected conversation lane; auxiliary "
                 "request-context records are retained and verified separately. ")
                + "Message "
                "counts concern separately recorded message events, not words inside reasoning "
                "or artifacts. Content truncation, termination, recovery, correctness and "
                "causal contribution require their own evidence. No text or artifact was "
                "marked read by generating this inventory."
            ),
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        return {"version": 1, "available": False,
                "reason": "The retained source cannot certify complete event structure; inspect "
                          "the original records and preserve uncertainty about missing events."}
