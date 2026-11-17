# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP spans carrying OpenInference attributes: the conversation of an instrumented agent."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from ..common import args_text, event_kind, split_inline_think, text_parts, tool_call

_OI_MSG = re.compile(r"^llm\.(input|output)_messages\.(\d+)\.message\.(.+)$")
_OI_CALL = re.compile(r"^tool_calls\.(\d+)\.tool_call\.(?:function\.)?(name|arguments|id)$")


def otlp_attrs(span: dict) -> dict:
    """OTLP attribute list -> flat dict; a value is a one-key union, not always stringValue."""
    out: dict = {}
    for a in span.get("attributes") or []:
        v = a.get("value") or {}
        for k in ("stringValue", "intValue", "boolValue", "doubleValue"):
            if k in v:
                out[a.get("key")] = v[k]
                break
    return out


def otlp_spans(rec: dict) -> list[dict]:
    return [
        s
        for rs in rec.get("resourceSpans") or []
        for ss in rs.get("scopeSpans") or []
        for s in ss.get("spans") or []
    ]


def nano_iso(ns) -> str | None:
    try:
        return datetime.fromtimestamp(int(ns) / 1e9, UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _as_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _oi_messages(attrs: dict, which: str) -> list[dict]:
    by_idx: dict[int, dict] = {}
    for k, v in attrs.items():
        m = _OI_MSG.match(str(k))
        if not m or m.group(1) != which:
            continue
        msg = by_idx.setdefault(int(m.group(2)), {"tool_calls": {}})
        tc = _OI_CALL.match(m.group(3))
        if tc:
            msg["tool_calls"].setdefault(int(tc.group(1)), {})[tc.group(2)] = v
        else:
            msg[m.group(3)] = v
    return [by_idx[i] for i in sorted(by_idx)]


def _is_llm(attrs: dict) -> bool:
    return attrs.get("openinference.span.kind") == "LLM" or "llm.model_name" in attrs


def openinference_rows(spans: list[dict]) -> dict | None:
    """The conversation of an OpenInference trace, as event rows; None if it has no LLM span.

    Every LLM span re-sends the whole history, so the conversation is the LAST one's input
    plus output messages; walking every span would repeat each message once per call. Per-span
    tokens attach to the assistant turns in start order; the prompt total is the last span's
    (cumulative) and the completion total is summed.
    """
    llm = sorted(
        ((s, otlp_attrs(s)) for s in spans if _is_llm(otlp_attrs(s))),
        key=lambda p: _as_int(p[0].get("startTimeUnixNano")) or 0,
    )
    if not llm:
        return None
    status_by_call: dict[str, str] = {}
    for s in spans:
        a = otlp_attrs(s)
        cid = a.get("tool_call.id") or a.get("tool.id")
        if cid:
            status_by_call[str(cid)] = {1: "ok", 2: "error"}.get(
                (s.get("status") or {}).get("code"), "unknown"
            )
    last_span, last = llm[-1]
    model = last.get("llm.model_name")
    rows = []
    for i, msg in enumerate(_oi_messages(last, "input") + _oi_messages(last, "output")):
        role = (msg.get("role") or "").lower() or None
        content = msg.get("content")
        calls = [msg["tool_calls"][j] for j in sorted(msg["tool_calls"])]
        row = {
            "source_event_id": f"{last_span.get('spanId') or 'llm'}:{i:04d}",
            "read_seq": f"{i:09d}",
            "started_at": None,
            "role": role,
            "tool_calls": [],
            "model_name": None,
        }
        if role == "system":
            row.update(event_kind("system"), parts=text_parts(content or None))
        elif role == "tool":
            cid = msg.get("tool_call_id")
            row.update(
                event_kind("tool_result"),
                parts=text_parts(content or None, channel="tool_result", origin="observation"),
                **({"answers_call_ids": [str(cid)]} if cid else {}),
            )
        elif role == "assistant" and calls:
            row.update(event_kind("tool_call"), parts=text_parts(content or None), model_name=model)
            for c in calls:
                args = c.get("arguments")
                cid = str(c.get("id") or "") or None
                row["tool_calls"].append(
                    tool_call(
                        c.get("name"),
                        cid,
                        args_text(args, 1 << 20),
                        status=status_by_call.get(cid or "", "unknown"),
                    )
                )
        else:
            parts = split_inline_think(text_parts(content or None))
            reasoning = bool(parts) and all(p.get("channel") == "reasoning" for p in parts)
            row.update(
                event_kind("reasoning" if reasoning else "message"),
                parts=parts,
                model_name=model if role == "assistant" else None,
            )
        rows.append(row)
    assistant = [
        r for r in rows if r["role"] == "assistant" and r["kind"] in ("message", "tool_call")
    ]
    for (span, a), row in zip(llm, assistant, strict=False):
        row["prompt_tokens"] = _as_int(a.get("llm.token_count.prompt"))
        row["completion_tokens"] = _as_int(a.get("llm.token_count.completion"))
        row["cached_tokens"] = _as_int(a.get("llm.token_count.prompt_details.cache_read"))
        row["reasoning_tokens"] = _as_int(a.get("llm.token_count.completion_details.reasoning"))
        row["started_at"] = nano_iso(span.get("startTimeUnixNano"))
        row["ended_at"] = nano_iso(span.get("endTimeUnixNano"))
        row["request_idx"] = next(i for i, p in enumerate(llm) if p[0] is span)
    completion = [_as_int(a.get("llm.token_count.completion")) for _s, a in llm]
    return {
        "rows": rows,
        "prompt_tokens": _as_int(last.get("llm.token_count.prompt")),
        "completion_tokens": sum(c for c in completion if c is not None)
        if any(c is not None for c in completion)
        else None,
    }
