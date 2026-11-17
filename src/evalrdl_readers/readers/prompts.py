# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model call's request, kept so it reassembles exactly, with each message stored once.

A long conversation re-sends its history on every call, so a rollout's requests together run to
tens or hundreds of millions of characters. A translator therefore never answers a request
verbatim: the reader result holds every answer of a run until the write step, so verbatim
requests would all stay alive at once. Instead `PromptEncoder` turns each request into compact
columns for its `model_call` event (`event_texts`) and the messages this rollout has not sent
before (`prompt_messages`).

A message is named by a hash of its own JSON, so two inputs that read the same message give it
the same name without coordinating, and a reparse renames nothing.

    enc = PromptEncoder()                    # one per rollout
    cols, new = enc.encode(request, "chat")  # per call, in the producer's order
    enc.release()                            # when the record those calls came from ends
    # cols: request_dialect, request_messages, request_tools, request_params
    # new:  [{"msg_hash": ..., "msg_json": ...}], the messages not seen before in this rollout

To reassemble: `json.loads(request_params)`; then set the dialect's message field to the
`prompt_messages.msg_json` of each `msg_hash` in `request_messages`, in order; then set `tools` to
the `tool_def_texts.def_json` of each `tool_defs.sha256` in `request_tools`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from evalrdl_schema import CURRENT_DIALECTS

from .common import dumps

#: The dialects a reader may state, and the request field each keeps its messages in: the
#: current schema's, which says where `request_messages` go back.
DIALECT_FIELDS = CURRENT_DIALECTS

#: The request fields that carry content rather than parameters, in either dialect.
PAYLOAD_FIELDS = frozenset({*DIALECT_FIELDS.values(), "tools"})

#: The roles of a request's instructions: what the harness says, not the model or a user.
INSTRUCTION_ROLES = ("system", "developer")

#: The roles a harness sends in a request: its instructions and the user turns.
SENT_ROLES = (*INSTRUCTION_ROLES, "user")

#: The roles of a message item, in either dialect.
MESSAGE_ROLES = (*SENT_ROLES, "assistant")

#: Hex characters of a message's SHA-256 kept as its name: 64 bits, so two distinct messages
#: of one rollout share a name with odds near 1e-19 per pair.
NAME_LENGTH = 16


def message_name(msg_json: str) -> str:
    """The name of a message, from its JSON."""
    return hashlib.sha256(msg_json.encode()).hexdigest()[:NAME_LENGTH]


def tool_name(tool: Any) -> str:
    """A tool definition's `tool_defs.sha256`: the digest of its JSON, as readers land it."""
    return hashlib.sha256(dumps(tool).encode()).hexdigest()


def dialect_of(stated: Any, request: dict) -> str | None:
    """The request's dialect: the producer's own word when it is one this module knows, else
    the one whose message field the request carries; None for neither."""
    if stated in DIALECT_FIELDS:
        return stated
    for dialect, field in DIALECT_FIELDS.items():
        if field in request:
            return dialect
    return None


#: LiteLLM, with its preview features on, appends a Gemini thought signature to a tool call id
#: (`<id>__thought__<base64>`), up to hundreds of kilobytes; the id is the part before it.
THOUGHT_SEPARATOR = "__thought__"


def producer_id(v: Any) -> str:
    """A producer's item or call id without a signature carried in it."""
    return str(v).split(THOUGHT_SEPARATOR, 1)[0]


def fields_named() -> str:
    """The dialects' message fields, as a finding names them."""
    return " nor ".join(f"`{f}`" for f in DIALECT_FIELDS.values())


def unread_messages(request: dict, dialect: str) -> str | None:
    """Why a reader of `dialect` leaves messages of `request` unread: it carries another dialect's
    message list, which only its parameters keep. None where it carries none."""
    foreign = [
        f for d, f in DIALECT_FIELDS.items() if d != dialect and isinstance(request.get(f), list)
    ]
    if not foreign:
        return None
    named = ", ".join(f"`{f}`" for f in foreign)
    return f"is stated {dialect!r} but carries {named}; it is landed whole, those messages unread"


def request_items(request: Any, stated: Any = None) -> list:
    """A request's message items, from the field its dialect (`dialect_of`) keeps them in; []
    for a request in no dialect this module knows, or whose field holds no list. Whoever lands
    the request reports what no reader of it reads (`unread_messages`)."""
    if not isinstance(request, dict):
        return []
    dialect = dialect_of(stated, request)
    items = request.get(DIALECT_FIELDS[dialect]) if dialect else None
    return items if isinstance(items, list) else []


def shared_item_keys(item: dict) -> list:
    """The names a request item shares with the same item in a transcript, in either dialect:
    the producer's item id, and the call id a tool call and its result carry (`call_id` on a
    Responses item, `tool_call_id` on a chat tool message)."""
    keys = [("id", producer_id(item["id"]))] if item.get("id") else []
    if item.get("call_id") and item.get("type"):
        keys.append((item["type"], producer_id(item["call_id"])))
    if item.get("role") == "tool" and item.get("tool_call_id"):
        keys.append(("function_call_output", producer_id(item["tool_call_id"])))
    return keys


def item_text(item: dict) -> str | None:
    """The text of a message item in either dialect, its text parts joined by a newline; None
    for an item that is not a message (a call, a result) or holds no text."""
    if not isinstance(item, dict) or item.get("type") not in (None, "message"):
        return None
    if item.get("role") not in MESSAGE_ROLES:
        return None
    content = item.get("content")
    if isinstance(content, str):
        return content
    texts = [
        p["text"] for p in content or () if isinstance(p, dict) and isinstance(p.get("text"), str)
    ] if isinstance(content, list) else []  # fmt: skip
    return "\n".join(texts) or None


def item_keys(item: dict) -> list[str]:
    """The `item_id` keys an item of a transcript or of a request is known by, in either
    dialect: its producer id, else, for a tool result, `function_call_output:<call_id>`. A
    chat assistant message is known by the ids of the tool calls it carries."""
    if not isinstance(item, dict):
        return []
    if item.get("id"):
        return [producer_id(item["id"])]
    if item.get("type") == "function_call_output" and item.get("call_id"):
        return [f"function_call_output:{producer_id(item['call_id'])}"]
    if item.get("role") == "tool" and item.get("tool_call_id"):
        return [f"function_call_output:{producer_id(item['tool_call_id'])}"]
    if item.get("type") == "function_call" and item.get("call_id"):
        return [producer_id(item["call_id"])]
    return [
        producer_id(tc["id"])
        for tc in item.get("tool_calls") or ()
        if isinstance(tc, dict) and tc.get("id")
    ]


class PromptEncoder:
    """Encodes one rollout's requests. Give it every call of the rollout, in order, and nothing
    of another rollout: which messages are new is a fact about the rollout."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._prev_items: list = []
        self._prev_names: list[str] = []
        self._prev_tools: Any = None
        self._prev_tool_names: list[str] = []

    def release(self) -> None:
        """Forget the previous request, which otherwise stays alive with every rollout's encoder
        until the read ends. Names and what is new do not change: a later request's messages
        are named again, and those this rollout has sent stay sent."""
        self._prev_items, self._prev_names = [], []

    def encode(self, request: dict, dialect: str) -> tuple[dict, list[dict]]:
        """The event columns for `request`, and the messages it is the first to send."""
        field = DIALECT_FIELDS[dialect]
        items = request.get(field)
        listed = isinstance(items, list)
        names: list[str] = []
        new: list[dict] = []
        if listed:
            prev_items, prev_names = self._prev_items, self._prev_names
            for i, item in enumerate(items):
                # A request re-sends the history before it: a message equal to the previous
                # request's at the same position keeps its name without being serialised.
                if i < len(prev_items) and item == prev_items[i]:
                    names.append(prev_names[i])
                    continue
                text = dumps(item)
                name = message_name(text)
                if name not in self._seen:
                    self._seen.add(name)
                    new.append({"msg_hash": name, "msg_json": text})
                names.append(name)
            self._prev_items, self._prev_names = items, names
        tools = request.get("tools")
        # An absent, empty or malformed `tools` stays in the parameters, exactly as written.
        by_name = isinstance(tools, list) and bool(tools)
        if by_name:
            if tools != self._prev_tools:
                self._prev_tools, self._prev_tool_names = tools, [tool_name(t) for t in tools]
            tool_names = self._prev_tool_names
        params = {
            k: v
            for k, v in request.items()
            if not (k == "tools" and by_name) and not (k == field and listed)
        }
        cols = {
            "request_dialect": dialect,
            "request_messages": json.dumps(names) if listed else None,
            "request_tools": json.dumps(tool_names) if by_name else None,
            "request_params": dumps(params),
        }
        return cols, new
