# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA's stored events, one at a time, as event rows. Every event is JSON with an
`event_type`; model calls and their results are paired by the tool call's id."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

from ..common import dumps, part, text_parts, tool_call

#: Events the model saw that the harness wrote: execution feedback, errors, loop guards.
HARNESS_SAID = frozenset({"Error", "Feedback", "LoopGuardTriggered", "Notification"})

#: Events the model's generated code wrote as its own words, before `LLMResponse` carried them.
CODE_SAID = {"Message": "message", "Reasoning": "reasoning"}

#: The human's prompt as the two older session stores spelled it.
USER_SAID = frozenset({"SessionUserMessage", "TUIUserInput"})

#: `self.message()`: what the agent showed the person, never the model.
AGENT_SAID = frozenset({"AgentMessage", "TUIAgentMessage"})

#: What the harness rendered into the prompt: the method's task text and the system prompt.
RENDERED = frozenset({"Task", "SystemPrompt"})

STATUS = {"complete": "ok", "error": "error", "cancelled": "cancelled"}


def timestamp(raw, offset: float | None):
    """NOOA stamps events with naive local time; the session's start names the writer's offset,
    so a stamp it covers is stated in UTC."""
    if not isinstance(raw, str) or not raw:
        return None
    if offset is None:
        return raw
    try:
        at = datetime.fromisoformat(raw)
    except ValueError:
        return raw
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone(timedelta(seconds=offset)))
    return at.astimezone(UTC).isoformat()


def seconds(start, end) -> float | None:
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None


def usage_columns(usage) -> dict:
    """NOOA's normalized usage; `input_tokens` already includes the cache reads."""
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "cached_tokens": usage.get("cached_input_tokens"),
        "cache_creation_tokens": usage.get("cache_write_input_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
    }


def call_args(arguments) -> str:
    return arguments if isinstance(arguments, str) else dumps(arguments)


def response_row(rec: dict, outcome: dict, request_idx: int, at) -> dict:
    """One model response: its parts in the order the provider returned them."""
    parts, calls = [], []
    for p in rec.get("parts") or ():
        kind = p.get("kind") if isinstance(p, dict) else None
        if kind == "text":
            parts += text_parts(p.get("text"))
        elif kind == "reasoning":
            if p.get("text"):
                parts.append(part(p["text"], channel="reasoning", origin="reasoning"))
            if p.get("native"):
                parts.append(
                    part(None, channel="reasoning", origin="native", part_type="encrypted")
                )
        elif kind == "tool_call":
            args = call_args(p.get("arguments"))
            done = outcome.get(p.get("id")) or {}
            calls.append(
                tool_call(
                    p.get("name"),
                    p.get("id"),
                    args,
                    status=done.get("status"),
                    started_at=done.get("started") or at,
                    ended_at=done.get("at"),
                    latency_s=seconds(done.get("started") or at, done.get("at")),
                )
            )
        else:
            parts.append(part(dumps(p), part_type=kind or "other"))
    if calls:
        kind = "tool_call"
    elif parts and all(p["channel"] == "reasoning" for p in parts):
        kind = "reasoning"
    else:
        kind = "message"
    return {
        "role": "assistant",
        "kind": kind,
        "kind_raw": None,
        "parts": parts,
        "tool_calls": calls,
        "model_name": rec.get("model_name") or None,
        "finish_reason": rec.get("finish_reason"),
        "response_id": rec.get("id"),
        "request_idx": request_idx,
        **usage_columns(rec.get("usage")),
    }


def output_row(rec: dict) -> dict:
    """A cell's execution: what it printed, raised and returned, answering the call that ran it."""
    parts = text_parts(rec.get("stdout"), channel="tool_result", origin="stdout")
    parts += text_parts(rec.get("stderr"), channel="tool_result", origin="stderr")
    parts += text_parts(rec.get("error"), channel="tool_result", origin="error")
    if rec.get("value") is not None:
        parts.append(part(dumps(rec["value"]), channel="tool_result", origin="value"))
    for image in rec.get("images") or ():
        parts += text_parts(image, channel="tool_result", origin="show")
    return {
        "role": "tool",
        "kind": "tool_result",
        "kind_raw": rec.get("execution_status"),
        "parts": parts,
        "tool_calls": [],
        "answers_call_ids": [rec["tool_call_id"]] if rec.get("tool_call_id") else [],
    }


def prefill_row(rec: dict, outcome: dict, at) -> dict:
    """A cell the harness ran before the model's turn, which the model then sees."""
    args = call_args(rec.get("arguments") or {})
    done = outcome.get(rec.get("tool_call_id")) or {}
    return {
        "role": "harness",
        "kind": "tool_call",
        "kind_raw": (rec.get("metadata") or {}).get("prefill_type") or "prefill",
        "parts": [],
        "tool_calls": [
            tool_call(
                rec.get("name"),
                rec.get("tool_call_id"),
                args,
                status=done.get("status"),
                started_at=at,
                ended_at=done.get("at"),
                latency_s=seconds(at, done.get("at")),
            )
        ],
    }


def item_row(rec: dict) -> dict:
    """An item put on one of the agent's channels: the person's prompt, a parent's brief, or a
    delivery from the agent's own work."""
    body = rec.get("item_json")
    try:
        body = json.loads(body) if isinstance(body, str) else body
    except ValueError:
        pass
    source = str(rec.get("source") or "")
    channel = rec.get("channel") or ""
    if source == "user" or source.startswith("parent:"):
        return {
            "role": "user",
            "kind": "message",
            "kind_raw": None if channel == "user_messages" else channel,
            "parts": text_parts(body, origin="parent" if source.startswith("parent:") else "body"),
            "tool_calls": [],
        }
    return {
        "role": "harness",
        "kind": "message",
        "kind_raw": channel or None,
        "parts": text_parts(body, origin=source or "item"),
        "tool_calls": [],
    }


#: Keys every stored event carries around what it says.
ENVELOPE = frozenset({"event_type", "id", "metadata", "status", "tag", "timestamp"})


def said_row(rec: dict, role: str, kind: str, kind_raw: str | None, origin: str) -> dict:
    """Words an event carries as `content`; one without them is kept whole."""
    said = rec.get("content")
    if said is None:
        said = dumps({k: v for k, v in rec.items() if k not in ENVELOPE})
    return {
        "role": role,
        "kind": kind,
        "kind_raw": kind_raw,
        "parts": text_parts(said, origin=origin),
        "tool_calls": [],
    }


def summary_row(rec: dict) -> dict:
    """The agent's recap of the events it collapsed, and which ones (`replaced_range`)."""
    parts = text_parts(rec.get("summary_text"), origin="summary")
    rest = {k: rec.get(k) for k in ("summary_tag", "replaced_range", "children_tags", "doc")}
    parts.append(part(dumps(rest), origin="summary_range"))
    return {
        "role": "assistant",
        "kind": "message",
        "kind_raw": "summary",
        "parts": parts,
        "tool_calls": [],
    }


def rendered_row(rec: dict) -> dict:
    return {
        "role": "system",
        "kind": "system",
        "kind_raw": rec.get("event_type"),
        "parts": text_parts(rec.get("prompt") or rec.get("content"), origin="rendered"),
        "tool_calls": [],
    }


def bookkeeping_row(rec, kind_raw: str) -> dict:
    return {
        "role": "harness",
        "kind": "system",
        "kind_raw": kind_raw,
        "parts": [part(dumps(rec), origin="bookkeeping")],
        "tool_calls": [],
    }
