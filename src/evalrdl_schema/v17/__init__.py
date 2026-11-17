# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""store-v17: v16 plus the exact request each captured model call sent.

A model call's request is the whole prompt the model saw: its messages in order, the tools
offered and every other field. An agent rewrites its history between calls (it compacts it,
trims it, or edits a turn), so a call's request is not a prefix of the rollout's conversation,
and nothing else in the store says what it was.

The request is the payload of its `model_call` event, so it lives on that event, in
`event_texts`: a texts-tier sibling of `events`, keyed like it. The request lists its messages
by hash. `prompt_messages` holds each distinct message once per rollout, so a long conversation
that re-sends its history on every call stores each message once. Both tables are the texts
tier, written to the object store only. Runs written before hold no rows.
"""

from __future__ import annotations

from ..dsl import Column, Dtype, Grain, Schema, Table, evolve
from ..v16 import V16

_ROLLOUT = (
    Column("run_id", Dtype.STR, "the run"),
    Column("task_id", Dtype.STR, "the rollout's task"),
    Column("repeat", Dtype.INT, "the rollout's repeat"),
    Column("agent_ref", Dtype.STR, "the rollout's agent"),
)
_EVENT_KEY = ("run_id", "task_id", "repeat", "agent_ref", "lane", "event_idx")
_MSG_KEY = ("run_id", "task_id", "repeat", "agent_ref", "msg_idx")

#: The `request_dialect` values a reader may state, and the request field holding the messages.
DIALECTS = {"chat": "messages", "responses": "input"}


V17: Schema = evolve(
    V16,
    "store-v17",
    note="each captured model call's request, on its event, its messages content-addressed",
    tables=(
        Table(
            "event_texts",
            Grain.TEXTS,
            key=_EVENT_KEY,
            columns=_ROLLOUT
            + (
                Column("lane", Dtype.STR, "the event's lane, as `events.lane`"),
                Column(
                    "event_idx",
                    Dtype.INT,
                    "the event's position in its lane, as `events.event_idx`",
                ),
                Column(
                    "request_dialect",
                    Dtype.STR,
                    "the API the call's request was written for, as the reader states it: "
                    "`chat` (messages under `messages`) or `responses` (input items under "
                    "`input`). It decides where `request_messages` go back and how a consumer "
                    "renders the prompt. NULL where the event carries no captured request",
                ),
                Column(
                    "request_messages",
                    Dtype.STR,
                    "the request's messages (or input items) in order, as a JSON array of "
                    "`prompt_messages.msg_hash`. NULL where the request's message field is not a "
                    "list (a Responses `input` given as one string stays in `request_params`)",
                ),
                Column(
                    "request_tools",
                    Dtype.STR,
                    "the tools the request offered, in order, as a JSON array of "
                    "`tool_defs.sha256`; NULL where it offered none",
                ),
                Column(
                    "request_params",
                    Dtype.STR,
                    "every other field of the request (model, sampling, `tool_choice`, "
                    "`chat_template_kwargs`, ...), as the producer wrote them, as JSON",
                ),
            ),
            means="an event's long values, beside it in the texts tier: for a `model_call` "
            "event, the request it sent",
        ),
        Table(
            "prompt_messages",
            Grain.TEXTS,
            key=_MSG_KEY,
            columns=_ROLLOUT
            + (
                Column(
                    "msg_idx",
                    Dtype.INT,
                    "the message's position among the rollout's distinct prompt messages. A row "
                    "number only: requests name a message by `msg_hash`, never by this",
                ),
                Column(
                    "msg_hash",
                    Dtype.STR,
                    "the first 16 hex characters of the SHA-256 of `msg_json`: the message's "
                    "name, the same from every input and every reparse",
                ),
                Column("msg_json", Dtype.STR, "the message as the producer wrote it, as JSON"),
            ),
            means="one distinct message a rollout's captured requests sent, once however many "
            "requests re-sent it",
        ),
    ),
)
