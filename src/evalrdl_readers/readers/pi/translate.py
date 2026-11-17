# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi's session file and stdout. Each is keyed by the session id it states, the only name
for its trial that pi writes; the host joins that id to the trial."""

from __future__ import annotations

from evalrdl_readers.core import Answer, Sources, Unreadable

from ..common import bookkeeping_event, dumps, event_kind, part, records, stdout_split, text_parts
from ..dialects.pi import pi_message


def _key(sid: str) -> tuple[str]:
    return (f"pi_session:{sid}",)


def pi_session(src: Sources):
    """pi's own transcript: every message with its id, its parent and its clock; the summary a
    compaction replaced the history with; model and thinking-level switches as bookkeeping."""
    for doc in src.docs(0):
        sid = None
        outcome = {}
        for _line, rec, _err in doc.jsonl():
            msg = rec.get("message") if isinstance(rec, dict) else None
            if isinstance(msg, dict) and msg.get("role") == "toolResult" and msg.get("toolCallId"):
                outcome[msg["toolCallId"]] = "error" if msg.get("isError") else "completed"
        for item in records(doc, "rollout.events"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            if sid is None:
                if rec.get("type") != "session" or not rec.get("id"):
                    yield Unreadable(
                        "rollout.events",
                        where=f"{doc.relpath}:{line}",
                        detail="a pi session file that does not open with its `session` "
                        "record names no session, so no trial",
                    )
                    break
                sid = str(rec["id"])
                yield Answer("rollout.pi_log", sid, key=_key(sid), keys={"pi_session": sid})
            eid = rec.get("id") or f"{sid}:{line:06d}"
            if rec.get("type") == "message" and isinstance(rec.get("message"), dict):
                row = {
                    "source_event_id": eid,
                    "parent_event_id": rec.get("parentId"),
                    "started_at": rec.get("timestamp"),
                    **pi_message(rec["message"], outcome),
                }
                lane = ("session", sid)
            elif rec.get("type") == "compaction":
                # The summary pi replaced its history with; the turns after it build on it.
                rest = {
                    k: v
                    for k, v in rec.items()
                    if k not in ("summary", "id", "parentId", "timestamp")
                }
                row = {
                    "source_event_id": eid,
                    "parent_event_id": rec.get("parentId"),
                    "started_at": rec.get("timestamp"),
                    "role": "harness",
                    **event_kind("compaction"),
                    "parts": text_parts(rec.get("summary"), origin="summary")
                    + [part(dumps(rest), origin="bookkeeping")],
                    "tool_calls": [],
                }
                lane = ("session", sid)
            else:
                row = bookkeeping_event(rec, eid, rec.get("timestamp"))
                row["parent_event_id"] = rec.get("parentId")
                lane = ("bookkeeping", sid)
            yield Answer("rollout.events", row, key=_key(sid), input_key=f"{sid}:{line}", lane=lane)


#: pi's stream records that restate its session file, message by message and tool by tool.
PI_STREAM = frozenset(
    {
        "session",
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "message_start",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
    }
)


def pi_stdout(src: Sources):
    """What only pi's stdout says: compaction, automatic retry, queued input and anything a
    newer pi adds; and text that is not an event, such as a crash. In stream order, the only
    order these records have."""
    for doc in src.docs(0):
        events, text = stdout_split(doc)
        sid = next(
            (str(r["id"]) for _, r in events if r.get("type") == "session" and r.get("id")), None
        )
        if sid is None:
            yield Unreadable(
                "rollout.events",
                where=doc.relpath,
                detail="a pi stdout with no `session` record names no session, so no trial; "
                "its events are not landed",
            )
            continue
        for line, rec in events:
            if rec.get("type") in PI_STREAM:
                continue
            yield Answer(
                "rollout.events",
                bookkeeping_event(rec, f"pi.txt:{line:06d}"),
                key=_key(sid),
                input_key=f"{sid}:stdout:{line}",
                lane=("bookkeeping", sid),
            )
        if text:
            yield Answer(
                "rollout.events",
                bookkeeping_event(
                    {"text": "\n".join(t for _, t in text)},
                    f"pi.txt:{text[0][0]:06d}",
                    kind="stdout_text",
                ),
                key=_key(sid),
                input_key=f"{sid}:stdout:text",
                lane=("bookkeeping", sid),
            )
