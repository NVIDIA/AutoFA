# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Chat Completions: a `messages` list."""

from __future__ import annotations

from ..common import args_text, text_parts, tool_call


def chat_messages(conversation) -> list[dict]:
    """One turn per message of an OpenAI-shaped `conversation` list."""
    out = []
    for m in conversation or ():
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        parts = []
        for r in (
            (m.get("reasoning_content") or ())
            if isinstance(m.get("reasoning_content"), list)
            else ()
        ):
            parts += text_parts(r, channel="reasoning")
        if isinstance(m.get("reasoning_content"), str):
            parts += text_parts(m["reasoning_content"], channel="reasoning")
        parts += text_parts(
            m.get("content"), channel="tool_result" if role == "tool" else "content"
        )
        calls = []
        for c in m.get("tool_calls") or ():
            if not isinstance(c, dict):
                continue
            fn = c.get("function") or {}
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if args is None:
                # stirrup (gdpval history.json) puts `name` and `arguments` on the call itself.
                args = c.get("arguments")
            calls.append(
                tool_call(
                    (fn.get("name") if isinstance(fn, dict) else None) or c.get("name"),
                    c.get("id") or c.get("tool_call_id"),
                    args_text(args),
                )
            )
        kind = "tool_result" if role == "tool" else ("tool_call" if calls else "message")
        out.append(
            {
                "role": role,
                "kind": kind,
                "parts": parts,
                "tool_calls": calls,
                "answers": [m["tool_call_id"]] if m.get("tool_call_id") else [],
            }
        )
    return out
