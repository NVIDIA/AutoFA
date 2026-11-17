# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What is computed from a source's rows, the same over every source."""

from __future__ import annotations

from datetime import datetime


def ts_seconds(value: str | None) -> float | None:
    """Timestamp string -> seconds: numeric strings (monotonic/epoch floats) or ISO."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None


def timeline(data: dict, event_cap: int) -> dict:
    """Timing for a timeline view: per-event start + duration, tool-call spans, rollup.

    `derivation`: span (both ends in source), to_next (start-only ATIF steps), point (lone
    stamp). Carries no texts.
    """
    ro = data["rollout"]
    total_events = data["total_events"]
    events = data["events"]
    calls = data["tool_calls"]

    for ev in events:
        ev["_t0"], ev["_t1"] = ts_seconds(ev["started_at"]), ts_seconds(ev["ended_at"])
    for c in calls:
        c["_t0"], c["_t1"] = ts_seconds(c["started_at"]), ts_seconds(c["ended_at"])
    ro_t0, ro_t1 = ts_seconds(ro["started_at"]), ts_seconds(ro["finished_at"])
    stamps = [t for r in (*events, *calls) for t in (r["_t0"], r["_t1"]) if t is not None]
    stamps += [t for t in (ro_t0, ro_t1) if t is not None]

    base = {
        "record_schema": ro["record_schema"],
        "status": ro["status"],
        "rollout": {
            "started_at": ro["started_at"],
            "finished_at": ro["finished_at"],
            "total_s": None if ro_t0 is None or ro_t1 is None else round(ro_t1 - ro_t0, 6),
        },
        "total_events": total_events,
        "event_cap": event_cap,
    }
    if not stamps:
        return {
            **base,
            "n_timed_events": 0,
            "span_s": None,
            "events": [],
            "tool_calls": [],
            "rollup": None,
        }
    t0 = min(stamps)
    span = max(stamps) - t0

    # start-only ATIF steps: bar = delta to next timed event start; last stays open
    to_next = (ro["record_schema"] or "").startswith("atif")
    out_events = []
    # Per (agent_ref, lane): a `to_next` width is the gap to the next step of the same sequence.
    for ag, ln in sorted({(e["agent_ref"], e["lane"]) for e in events}):
        ag_evs = [
            e for e in events if e["agent_ref"] == ag and e["lane"] == ln and e["_t0"] is not None
        ]
        for i, ev in enumerate(ag_evs):
            if ev["_t1"] is not None:
                duration, derivation = ev["_t1"] - ev["_t0"], "span"
            elif to_next and i + 1 < len(ag_evs):
                duration, derivation = ag_evs[i + 1]["_t0"] - ev["_t0"], "to_next"
            elif to_next:
                duration, derivation = None, "open"
            else:
                duration, derivation = None, "point"
            out_events.append(
                {
                    "agent_ref": ag,
                    "lane": ln,
                    "event_idx": ev["event_idx"],
                    "kind": ev["kind"],
                    "role": ev["role"],
                    "start_s": round(ev["_t0"] - t0, 6),
                    "duration_s": None if duration is None else round(max(duration, 0), 6),
                    "derivation": derivation,
                    "n_tool_calls": ev["n_tool_calls"],
                    "completion_tokens": ev["completion_tokens"],
                }
            )
    out_calls = [
        {
            "agent_ref": c["agent_ref"],
            "lane": c["lane"],
            "event_idx": c["event_idx"],
            "call_idx": c["call_idx"],
            "tool_name": c["tool_name"],
            "status": c["status"],
            "start_s": round(c["_t0"] - t0, 6),
            "duration_s": None if c["_t1"] is None else round(max(c["_t1"] - c["_t0"], 0), 6),
            "result_event_idx": c["result_event_idx"],
        }
        for c in calls
        if c["_t0"] is not None
    ]

    def is_model(e):
        return e["kind"] == "reasoning" or (
            e["kind"] == "message" and e["role"] in ("assistant", "agent")
        )

    model_spans = [e["duration_s"] for e in out_events if e["derivation"] == "span" and is_model(e)]
    n_model = sum(1 for e in events if is_model(e))
    tool_spans = [c["duration_s"] for c in out_calls if c["duration_s"] is not None]
    total_s = (
        base["rollout"]["total_s"] if base["rollout"]["total_s"] is not None else round(span, 6)
    )
    model_s = round(sum(model_spans), 6) if model_spans else None
    tool_s = round(sum(tool_spans), 6) if tool_spans else None
    unattributed = None
    if total_s and (model_s is not None or tool_s is not None):
        unattributed = round(max(total_s - (model_s or 0) - (tool_s or 0), 0), 6)
    return {
        **base,
        "n_timed_events": len(out_events),
        "span_s": round(span, 6),
        "span_basis": "rollout" if base["rollout"]["total_s"] is not None else "observed",
        "events": out_events,
        "tool_calls": out_calls,
        "rollup": {
            "total_s": total_s,
            "model_s": model_s,
            "tool_s": tool_s,
            "unattributed_s": unattributed,
            "n_model_events": n_model,
            "n_model_events_timed": len(model_spans),
            "n_tool_calls": len(calls),
            "n_tool_calls_timed": len(tool_spans),
        },
    }
