# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model proxy's own record of each call, `<bench>/model_traffic.jsonl`.

The launcher's adapter proxy appends one record per model call: when it started, how it ended, the
usage the provider reported and, where capture was on, the request body and the reply. The
request is the whole prompt the model saw, so an agent that compacts or rewrites its history
shows it here and nowhere else in the folder.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from evalrdl_readers.core import Answer, At, Card, Input, Sources, Unreadable

from ..common import args_text, dig, dumps, event_kind, part, records, tool_call
from ..prompts import PromptEncoder, dialect_of, fields_named, unread_messages
from .landmarks import RUN

#: The route the proxy says it sent upstream -> the dialect of the captured body. The body is
#: the upstream one, so a `/v1/messages` call the proxy translated is a chat request.
ROUTE_DIALECT = {
    "chat_completions": "chat",
    "chat_completions_enable_thinking": "chat",
    "anthropic_messages": None,
}

#: The request path, for a proxy that states no route.
PATH_DIALECT = {
    "/chat/completions": "chat",
    "/v1/chat/completions": "chat",
    "/responses": "responses",
    "/v1/responses": "responses",
}


def _dialect(rec: dict, body) -> tuple[str | None, str]:
    """The captured body's dialect, and why there is none."""
    if not isinstance(body, dict):
        return None, "not a JSON object"
    route = dig(rec, "reasoning_request", "effective_route")
    if route in ROUTE_DIALECT:
        stated = ROUTE_DIALECT[route]
        if stated is None:
            return None, f"sent upstream as {route}, a dialect the store does not hold"
        return dialect_of(stated, body), ""
    path = (rec.get("path") or "").rstrip("/")
    if path.endswith("/messages"):
        return (
            None,
            "sent to a Messages path with no route stated, so chat and Messages bodies cannot be told apart",
        )
    return dialect_of(PATH_DIALECT.get(path), body), f"carries neither {fields_named()}"


#: The proxy's suffix on a body it cut at its `max_content_chars`.
TRUNCATED = "...[truncated, "


def _iso(ts) -> str | None:
    """The proxy writes `time.time()`; a copy may hold an ISO string already."""
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")
    return ts if isinstance(ts, str) else None


def _ended(start, latency_ms) -> str | None:
    if not isinstance(start, (int, float)) or not isinstance(latency_ms, (int, float)):
        return None
    return _iso(start + latency_ms / 1000.0)


def _calls(rec: dict) -> list[dict]:
    out = []
    for tc in rec.get("tool_calls_full") or ():
        if not isinstance(tc, dict):
            continue
        out.append(tool_call(tc.get("name"), tc.get("id") or None, args_text(tc.get("arguments"))))
    return out


def _reply_parts(rec: dict) -> list[dict]:
    parts = []
    if rec.get("reasoning_content"):
        parts.append(part(rec["reasoning_content"], channel="reasoning", origin="reply"))
    if rec.get("message_content"):
        parts.append(part(rec["message_content"], origin="reply"))
    return parts


def _outcome(rec: dict) -> str | None:
    """The finish reason of a call that finished, else why it did not."""
    if rec.get("finish_reason"):
        return rec["finish_reason"]
    if rec.get("error_type"):
        return rec["error_type"]
    status = rec.get("status_code")
    return f"http_{status}" if isinstance(status, int) and status >= 400 else None


def _tool_def(tool, offered_by: str, idx: int, text: str) -> dict:
    fn = tool.get("function") if isinstance(tool, dict) else None
    name = (tool.get("name") if isinstance(tool, dict) else None) or (
        fn.get("name") if isinstance(fn, dict) else None
    )
    return {
        "def_idx": idx,
        "name": name,
        "tool_type": tool.get("type") if isinstance(tool, dict) else None,
        "def_chars": len(text),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "offered_by": offered_by,
        "def_json": text,
    }


def nel_traffic(src: Sources):
    """One `model_call` event per proxy record, carrying the request it sent where the body
    was captured whole. The file is appended in call order across every rollout of the run,
    so each rollout's encoder sees its own calls in the order they were made."""
    # Written only where traffic capture was on.
    if not src.has(0):
        return
    encoders: dict[tuple, PromptEncoder] = {}
    defs: dict[tuple, set[str]] = {}
    for item in records(src.doc(0), "rollout.events"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        idx, rep = rec.get("problem_idx"), rec.get("repeat")
        if idx is None:
            yield Unreadable(
                "rollout.events",
                where=f"{src.doc(0).relpath}:{line}",
                detail="a proxy record with no `problem_idx` names no rollout",
            )
            continue
        key = (f"{idx}/{rep}" if rep is not None else str(idx),)
        cid = rec.get("request_id") or rec.get("model_traffic_request_id") or f"line:{line}"
        usage = rec.get("usage") if isinstance(rec.get("usage"), dict) else {}
        event = {
            "source_event_id": cid,
            **event_kind("model_call"),
            "started_at": _iso(rec.get("timestamp")),
            "ended_at": _ended(rec.get("timestamp"), rec.get("latency_ms")),
            "latency_s": rec["latency_ms"] / 1000.0
            if isinstance(rec.get("latency_ms"), (int, float))
            else None,
            "model_name": rec.get("model") or None,
            "finish_reason": _outcome(rec),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": usage.get("cached_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "tool_calls": _calls(rec),
            "parts": _reply_parts(rec),
        }
        raw = rec.get("request_body")
        if raw is not None:
            body = None
            if isinstance(raw, dict):
                body = raw
            elif isinstance(raw, str) and TRUNCATED not in raw[-80:]:
                try:
                    body = json.loads(raw)
                except ValueError:
                    pass
            cut = isinstance(raw, str) and TRUNCATED in raw[-80:]
            dialect, why = _dialect(rec, body)
            if dialect is None:
                yield Unreadable(
                    "rollout.events",
                    where=f"{src.doc(0).relpath}:{line}",
                    detail=f"call {cid}'s request body is "
                    + ("cut at the proxy's content limit" if cut else why)
                    + "; its request is not landed",
                )
            else:
                if why := unread_messages(body, dialect):
                    yield Unreadable(
                        "rollout.events",
                        where=f"{src.doc(0).relpath}:{line}",
                        detail=f"call {cid}'s request {why}",
                    )
                cols, new = encoders.setdefault(key, PromptEncoder()).encode(body, dialect)
                event |= cols
                for msg in new:
                    yield Answer("rollout.prompt_messages", msg, key=key)
                seen = defs.setdefault(key, set())
                for tool in body.get("tools") if isinstance(body.get("tools"), list) else ():
                    text = dumps(tool)
                    if text not in seen:
                        seen.add(text)
                        yield Answer(
                            "rollout.tool_defs", _tool_def(tool, cid, len(seen) - 1, text), key=key
                        )
        yield Answer("rollout.events", event, key=key, input_key=cid, lane=("model_call", ""))


NEL_TRAFFIC = Input(
    name="nel_traffic",
    via=nel_traffic,
    reads=[At(RUN, "model_traffic.jsonl")],
    answers={
        "rollout.events": Card.ANY(
            because="the proxy writes the file only where traffic capture was on, and a rollout "
            "whose agent never reached the model has no call in it",
        ),
        "rollout.prompt_messages": Card.ANY(
            because="`request_body` is kept only where the proxy was told to capture bodies",
        ),
        "rollout.tool_defs": Card.ANY(
            because="the tools a captured request offered; a request may offer none",
        ),
    },
)
