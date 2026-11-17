# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi's session-file messages: `user` and `assistant` with typed content blocks, and `toolResult`."""

from __future__ import annotations

import json

from ..common import dumps, event_kind, part, text_parts, tool_call


def pi_message(msg: dict, outcome: dict[str, str]) -> dict:
    """A pi message: `user`/`assistant` with typed content blocks, or a `toolResult`.
    `outcome` is each call's result as its `toolResult` states it."""
    role = msg.get("role")
    parts, calls = [], []
    for block in msg.get("content") if isinstance(msg.get("content"), list) else ():
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "thinking":
            parts.append(part(block.get("thinking"), channel="reasoning", origin="thinking"))
        elif kind == "toolCall":
            args = block.get("arguments")
            text = args if isinstance(args, str) else dumps(args)
            calls.append(
                tool_call(
                    block.get("name"), block.get("id"), text, status=outcome.get(block.get("id"))
                )
            )
        elif kind == "text":
            if block.get("text"):
                parts.append(
                    part(
                        block.get("text"),
                        channel="tool_result" if role == "toolResult" else "content",
                    )
                )
        else:
            parts.append(part(dumps(block), part_type=kind or "other"))
    if isinstance(msg.get("content"), str):
        parts += text_parts(msg["content"])
    usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
    row = {
        "role": role,
        **event_kind("tool_result" if role == "toolResult" else "message"),
        "parts": parts,
        "tool_calls": calls,
        "model_name": msg.get("model"),
        "finish_reason": msg.get("stopReason"),
    }
    if usage:
        # `input` excludes what was read from cache; the provider's prompt count includes it.
        cached = usage.get("cacheRead") or 0
        row |= {
            "prompt_tokens": (usage.get("input") or 0) + cached + (usage.get("cacheWrite") or 0),
            "completion_tokens": usage.get("output"),
            "cached_tokens": cached,
        }
    if role == "toolResult" and msg.get("toolCallId"):
        row["answers_call_ids"] = [msg["toolCallId"]]
    return row


def stdout_session(text: str) -> str | None:
    """The session id pi's JSON-mode stdout opens with, its `session` record; None otherwise."""
    if not isinstance(text, str) or not text.startswith('{"type":"session"'):
        return None
    try:
        head = json.loads(text.split("\n", 1)[0])
    except ValueError:
        return None
    return str(head["id"]) if isinstance(head, dict) and head.get("id") else None
