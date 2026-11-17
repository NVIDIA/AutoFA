# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""opencode's `generation.json`: the clock and outcome of each tool call."""

from __future__ import annotations

from datetime import UTC, datetime

from ...common import as_float, dig


def _opencode_timing(docs) -> dict:
    """Per-tool-call clock and outcome from opencode `generation.json`, keyed by `tool_call_id`."""
    out: dict[str, dict] = {}
    for doc in docs:
        rec = doc.json()
        if isinstance(rec, dict):
            out.update(tool_call_timing(dig(rec, "observations", "records")))
    return out


def tool_call_timing(records) -> dict:
    """Per-tool-call clock and outcome from Gym's tool-call observations, keyed by
    `tool_call_id`: opencode's `observations.records`, or `ng_trajectory.tool_calls`."""
    out: dict[str, dict] = {}
    for r in records or ():
        if not isinstance(r, dict) or r.get("kind") != "tool_call":
            continue
        cid = r.get("tool_call_id")
        if not cid:
            continue
        out[str(cid)] = {
            "started_at": _epoch(r.get("started_at")),
            "ended_at": _epoch(r.get("completed_at")),
            "latency_s": (as_float(r.get("duration_ms")) or 0.0) / 1000.0
            if r.get("duration_ms") is not None
            else None,
            # Whether the tool ran; the item's own `status` only says it was emitted.
            "status": r.get("status"),
        }
    return out


def _epoch(value) -> str | None:
    """An epoch second, as the ISO-8601 string the column holds."""
    seconds = as_float(value)
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()
