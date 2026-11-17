# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA's sessions: each root session with every child session under it is one rollout, keyed by
the root's session id. A child names its parent; the parent's events name the child only after
it ran (its usage, its close), never the call that started it, so a child is a lane of the
rollout and no edge ties it to a call."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from evalrdl_readers.core import Answer, Sources, Unreadable

from ..common import part
from .shape import (
    AGENT_SAID,
    CODE_SAID,
    HARNESS_SAID,
    RENDERED,
    STATUS,
    USER_SAID,
    bookkeeping_row,
    item_row,
    output_row,
    prefill_row,
    rendered_row,
    response_row,
    said_row,
    summary_row,
    timestamp,
)

_START = frozenset({"SessionStarted", "TUISessionStart"})


def _sid(doc) -> str:
    return doc.name.split(".", 1)[0]


def _read(doc) -> tuple[list[dict], list[tuple]]:
    """Every stored event in insertion order, and the agent's snapshots."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.db"
        path.write_bytes(doc.bytes())
        db = sqlite3.connect(path)
        try:
            rows = db.execute(
                "SELECT insertion_order, event_type, status, data FROM events "
                "ORDER BY insertion_order"
            ).fetchall()
            snaps = db.execute(
                "SELECT snapshot_id, created_at, data FROM snapshots ORDER BY created_at"
            ).fetchall()
        finally:
            db.close()
    events = []
    for order, event_type, status, data in rows:
        rec = json.loads(data)
        if not isinstance(rec, dict):
            raise ValueError(f"event {order} is not an object")
        rec.setdefault("event_type", event_type)
        rec["_order"], rec["_status"] = order, status
        events.append(rec)
    return events, snaps


def _outcomes(events: list[dict], offset) -> tuple[dict, dict]:
    """Per tool call id: the response that made it, and when and how its cell finished."""
    made_by, outcome = {}, {}
    for rec in events:
        at = timestamp(rec.get("timestamp"), offset)
        et = rec["event_type"]
        if et == "LLMResponse":
            for p in rec.get("parts") or ():
                if isinstance(p, dict) and p.get("kind") == "tool_call" and p.get("id"):
                    made_by[p["id"]] = rec.get("id")
        elif et == "ToolCallEvent" and rec.get("tool_call_id"):
            outcome.setdefault(rec["tool_call_id"], {})["started"] = at
            if not rec.get("llm_response_id"):
                made_by.setdefault(rec["tool_call_id"], rec.get("id"))
        elif et == "PythonOutput" and rec.get("tool_call_id"):
            status = rec.get("execution_status")
            outcome.setdefault(rec["tool_call_id"], {}).update(
                status=STATUS.get(status, status) if status is not None else None, at=at
            )
    return made_by, outcome


def _session_rows(sid: str, events: list[dict], snaps, lane, actor: dict, offset):
    """Every event of one session as (row, lane) with its parent named by id."""
    made_by, outcome = _outcomes(events, offset)
    prefill = {
        r["tool_call_id"]
        for r in events
        if r["event_type"] == "ToolCallEvent" and (r.get("metadata") or {}).get("prefill")
    }
    context, bookkeeping = ("context", sid), ("bookkeeping", sid)
    requests = 0
    for rec in events:
        et, eid = rec["event_type"], rec.get("id") or f"{sid}:{rec['_order']}"
        at = timestamp(rec.get("timestamp"), offset)
        meta = rec.get("metadata") or {}
        into, parent = lane, None
        raw = {k: v for k, v in rec.items() if not k.startswith("_")}
        if et == "LLMResponse":
            row = response_row(rec, outcome, requests, at)
            requests += 1
            if rec.get("dynamic_context"):
                yield (
                    {
                        "source_event_id": f"{eid}:context",
                        "parent_event_id": eid,
                        "started_at": at,
                        "role": "system",
                        "kind": "system",
                        "kind_raw": "dynamic_context",
                        "parts": [part(rec["dynamic_context"], origin="dynamic_context")],
                        "tool_calls": [],
                        "read_seq": f"{rec['_order']:09d}.1",
                        **actor,
                    },
                    context,
                )
        elif et == "ToolCallEvent" and meta.get("prefill"):
            row, into = prefill_row(rec, outcome, at), context
        elif et == "ToolCallEvent":
            row, into = (
                bookkeeping_row(raw, "synthetic" if meta.get("synthetic") else et),
                bookkeeping,
            )
            parent = rec.get("llm_response_id")
        elif et == "PythonOutput":
            row = output_row(rec)
            parent = made_by.get(rec.get("tool_call_id"))
            if rec.get("tool_call_id") in prefill:
                into = context
        elif et == "ItemAdmitted":
            row = item_row(rec)
        elif et in USER_SAID:
            row = said_row(rec, "user", "message", None, "body")
        elif et in AGENT_SAID:
            row = said_row(rec, "assistant", "message", "agent_message", "message")
        elif et in CODE_SAID:
            row = said_row(rec, "assistant", CODE_SAID[et], et, "code")
        elif et in HARNESS_SAID:
            row = said_row(rec, "harness", "message", et, "feedback")
        elif et == "Summary":
            row = summary_row(rec)
        elif et in RENDERED:
            row, into = rendered_row(rec), context
        else:
            row, into = bookkeeping_row(raw, et), bookkeeping
        if rec.get("_status") not in (None, "active"):
            row["kind_raw"] = f"{row.get('kind_raw') or row['kind']}:{rec['_status']}"
        yield (
            {
                "source_event_id": eid,
                "parent_event_id": parent,
                "started_at": at,
                "read_seq": f"{rec['_order']:09d}",
                **row,
                **({} if into == bookkeeping else actor),
            },
            into,
        )
    for snap_id, created, data in snaps:
        yield (
            {
                "source_event_id": snap_id,
                "started_at": created,
                "read_seq": f"~{created}",
                **bookkeeping_row(json.loads(data), "snapshot"),
            },
            bookkeeping,
        )


def root_sessions(src: Sources, i: int = 0):
    """The session id of every store at `src`'s `i`th read that starts a session naming no
    parent among them, sorted; a store that cannot be read is the reader's to report."""
    starts = {}
    for doc in src.docs(i):
        try:
            events, _ = _read(doc)
        except (sqlite3.Error, ValueError, OSError):
            continue
        start = next((r for r in events if r["event_type"] in _START), None)
        if start is not None:
            starts[_sid(doc)] = start.get("parent_id")
    return sorted(sid for sid, parent in starts.items() if parent not in starts)


def nooa_sessions(src: Sources):
    """Every session in the folder, each child folded into its root's rollout as a lane."""
    yield from _tree(src, root_lane=None, run=True)


def nooa_session_lanes(src: Sources):
    """The same rows where a host runs NOOA and keeps its own account of the turns: the root
    session is the `session` lane, an alternate account, and names its session id for the host
    to join; the run is the host's."""
    yield from _tree(src, root_lane="session", run=False)


def _tree(src: Sources, root_lane: str | None, run: bool):
    sessions: dict[str, dict] = {}
    for doc in src.docs(0):
        sid = _sid(doc)
        try:
            events, snaps = _read(doc)
        except (sqlite3.Error, ValueError, OSError) as e:
            yield Unreadable("rollout.events", where=doc.relpath, detail=f"not a NOOA store: {e}")
            continue
        start = next((r for r in events if r["event_type"] in _START), None)
        if start is None:
            yield Unreadable(
                "rollout.events",
                where=doc.relpath,
                detail="the store records no session start, so it is not a session",
            )
            continue
        sessions[sid] = {"start": start, "events": events, "snaps": snaps}

    def root_of(sid: str) -> str:
        seen = {sid}
        while (p := sessions[sid]["start"].get("parent_id")) in sessions and p not in seen:
            sid = p
            seen.add(sid)
        return sid

    models: Counter = Counter()
    run_first = None
    times: dict[str, list] = {}
    for sid in sorted(sessions, key=lambda s: (sessions[s]["start"].get("depth") or 0, s)):
        s = sessions[sid]
        start, root = s["start"], root_of(sid)
        offset = start.get("utc_offset")
        child = root != sid
        lane = ("subagent", sid) if child else (root_lane, sid) if root_lane else None
        actor = {
            "actor": (start.get("name") or "subagent") if child else "main",
            "agent_depth": start.get("depth"),
        }
        for row, into in _session_rows(sid, s["events"], s["snaps"], lane, actor, offset):
            if row.get("started_at") and row.get("kind_raw") != "snapshot":
                times.setdefault(root, []).append(row["started_at"])
            if row.get("model_name"):
                models[row["model_name"]] += 1
            yield Answer(
                "rollout.events",
                row,
                key=(root,),
                input_key=f"{sid}:{row['source_event_id']}",
                lane=into,
            )
        if not models and start.get("model"):
            models[start["model"]] = 0
    if not run:
        for root in sorted({root_of(sid) for sid in sessions}):
            yield Answer("rollout.nooa_log", root, key=(root,), keys={"nooa_session": root})
        return
    for root, ts in times.items():
        yield Answer("rollout.started_at", min(ts), key=(root,))
        yield Answer("rollout.finished_at", max(ts), key=(root,))
        run_first = min(run_first or min(ts), min(ts))
    if run_first:
        yield Answer("run.started_at", run_first)
    if models:
        yield Answer("run.model", models.most_common(1)[0][0])
