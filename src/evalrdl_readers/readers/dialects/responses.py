# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenAI Responses API: one input or output item."""

from __future__ import annotations

from ..common import inline_artifact, part, text_parts, tool_call


def responses_parts(item: dict) -> list[dict]:
    """Every part of one responses-api output item."""
    out: list[dict] = []
    content = item.get("content")
    if isinstance(content, str):
        out += text_parts(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                out.append(part(block))
                continue
            if block.get("text") == "" and not block.get("image_url"):
                continue
            p = part(block.get("text"), part_type=block.get("type") or "text")
            art = inline_artifact(block["image_url"]) if block.get("image_url") else None
            if art is not None:
                p["artifact"] = art
                if art.get("mime"):
                    p["media_type"] = art["mime"]
            out.append(p)
    for block in item.get("summary") or ():
        if isinstance(block, dict):
            out.append(part(block.get("text"), channel="reasoning", origin="summary"))
    if item.get("encrypted_content") is not None:
        out.append(
            part(None, channel="reasoning", origin="encrypted_content", part_type="encrypted")
        )
    if item.get("output") is not None:
        out.append(part(item["output"], channel="tool_result", origin="observation"))
    return out


def responses_calls(item: dict) -> list[dict]:
    """The tool call one responses-api item IS, or nothing. Columns of `tool_calls`."""
    if item.get("type") != "function_call":
        return []
    args = item.get("arguments")
    text = "" if args is None else str(args)
    return [tool_call(item.get("name"), item.get("call_id"), text, status=item.get("status"))]
