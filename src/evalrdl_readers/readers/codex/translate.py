# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""codex's rollout file and stdout. Each is keyed by the session id it states, the only name
for its trial that codex writes; the host joins that id to the trial."""

from __future__ import annotations

from evalrdl_readers.core import Answer, Sources, Unreadable

from ..common import bookkeeping_event, event_kind, records, stdout_split
from ..dialects.sessions import session_shape


def _key(sid: str) -> tuple[str]:
    return (f"codex_session:{sid}",)


def _session_of(rec: dict) -> str | None:
    sid = (rec.get("payload") or {}).get("id") if rec.get("type") == "session_meta" else None
    return str(sid) if sid else None


def codex_log(src: Sources):
    """The session a rollout file names itself by, from its `session_meta` header."""
    for doc in src.docs(0):
        for item in records(doc, "rollout.codex_log"):
            if isinstance(item, Unreadable):
                yield item
                break
            sid = _session_of(item[1])
            if sid:
                yield Answer("rollout.codex_log", sid, key=_key(sid), keys={"codex_session": sid})
            break


def codex_session(src: Sources):
    """codex's rollout file, record by record, through the shared codex decoders."""
    for doc in src.docs(0):
        sid = None
        for item in records(doc, "rollout.events"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            if sid is None:
                sid = _session_of(rec)
                if sid is None:
                    yield Unreadable(
                        "rollout.events",
                        where=f"{doc.relpath}:{line}",
                        detail="a codex rollout that does not open with `session_meta` "
                        "names no session, so no trial",
                    )
                    break
            dec = src.decode(rec, "rollout.events")
            if isinstance(dec, Unreadable):
                yield dec
                continue
            got = session_shape(dec, rec)
            yield Answer(
                "rollout.events",
                {
                    "source_event_id": got["id"] or f"{sid}:{line:06d}",
                    "started_at": rec.get("timestamp"),
                    "role": got["role"],
                    **event_kind(got["kind"]),
                    "parts": got["parts"],
                    "tool_calls": got["tool_calls"],
                    "model_name": got["model_name"],
                    **({"answers_call_ids": got["answers"]} if got["answers"] else {}),
                },
                key=_key(sid),
                input_key=f"{sid}:{line}",
                via=dec.name,
                lane=(got["lane"], sid),
            )


#: codex stdout records that restate its rollout file.
CODEX_STREAM_PREFIXES = ("thread.", "turn.", "item.")


def codex_stdout(src: Sources):
    """What only codex's stdout says: its errors (a reconnect, a refused request) and text that
    is not an event. Items, turns and the thread restate the rollout file."""
    for doc in src.docs(0):
        events, text = stdout_split(doc)
        sid = next(
            (
                str(r["thread_id"])
                for _, r in events
                if r.get("type") == "thread.started" and r.get("thread_id")
            ),
            None,
        )
        if sid is None:
            yield Unreadable(
                "rollout.events",
                where=doc.relpath,
                detail="a codex stdout with no `thread.started` names no session, so no trial; "
                "its events are not landed",
            )
            continue
        for line, rec in events:
            if str(rec.get("type") or "").startswith(CODEX_STREAM_PREFIXES):
                continue
            yield Answer(
                "rollout.events",
                bookkeeping_event(rec, f"codex.txt:{line:06d}"),
                key=_key(sid),
                input_key=f"{sid}:stdout:{line}",
                lane=("bookkeeping", sid),
            )
        if text:
            yield Answer(
                "rollout.events",
                bookkeeping_event(
                    {"text": "\n".join(t for _, t in text)},
                    f"codex.txt:{text[0][0]:06d}",
                    kind="stdout_text",
                ),
                key=_key(sid),
                input_key=f"{sid}:stdout:text",
                lane=("bookkeeping", sid),
            )
