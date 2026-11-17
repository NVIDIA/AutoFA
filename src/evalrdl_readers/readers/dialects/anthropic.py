# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages: a message of typed content blocks."""

from __future__ import annotations

from ..common import dumps, part, text_parts, tool_call


def anthropic_message(message: dict) -> dict:
    """The parts, the tool calls and the kind of one Anthropic-format message record."""
    parts: list[dict] = []
    calls: list[dict] = []
    answers: list[str] = []
    content = (message or {}).get("content")
    if isinstance(content, str):
        parts += text_parts(content)
        content = ()
    for block in content or ():
        if not isinstance(block, dict):
            parts.append(part(block))
            continue
        kind = block.get("type")
        if kind == "text":
            parts += text_parts(block.get("text"))
        elif kind == "thinking":
            parts.append(part(block.get("thinking"), channel="reasoning"))
            if block.get("signature"):
                # Signed reasoning; feeds production's `has_encrypted_reasoning`.
                parts.append(
                    part(
                        None, channel="reasoning", origin="encrypted_content", part_type="encrypted"
                    )
                )
        elif kind == "tool_use":
            args = dumps(block.get("input")) if block.get("input") is not None else ""
            calls.append(tool_call(block.get("name"), block.get("id"), args))
        elif kind == "tool_result":
            body = block.get("content")
            if isinstance(body, list):
                for b in body:
                    parts += text_parts(
                        b.get("text") if isinstance(b, dict) else b,
                        channel="tool_result",
                        origin="observation",
                    )
            else:
                parts += text_parts(body, channel="tool_result", origin="observation")
            if block.get("tool_use_id"):
                answers.append(block["tool_use_id"])
        else:
            # Unknown block types keep their own spelling.
            parts.append(part(dumps(block), part_type=kind or "other"))
    kind = "tool_call" if calls else "tool_result" if answers else "message"
    return {"parts": parts, "tool_calls": calls, "kind": kind, "answers": answers}
