# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym's harness-neutral account of who did what (`ng_trajectory`): each sub-agent an agent
invocation naming its parent and the tool call that started it, with its own conversation, and
each tool call with its clock."""

from __future__ import annotations

from dataclasses import replace

from evalrdl_readers.core import Answer, At, Card, Input, Key, Sources, Unreadable

from ..common import dig, dumps, event_kind, part, primary_stream, records, text_parts
from ..prompts import INSTRUCTION_ROLES, SENT_ROLES
from .agents.opencode import tool_call_timing
from .calls import call_items, stated_dialects, trajectory_calls
from .landmarks import RUN, STREAMS
from .record import _gym_key, _responses_events


def invocations(rec: dict) -> list[dict]:
    return [
        inv
        for inv in dig(rec, "ng_trajectory", "invocations") or ()
        if isinstance(inv, dict) and inv.get("invocation_id")
    ]


def trajectory_timing(rec: dict) -> dict:
    """Each tool call's clock and outcome, keyed by its call id."""
    return tool_call_timing(dig(rec, "ng_trajectory", "tool_calls"))


def spawns(rec: dict) -> set[str]:
    """The tool calls that started a sub-agent."""
    return {
        str(inv["spawned_by_tool_call_id"])
        for inv in invocations(rec)
        if inv.get("parent_invocation_id") and inv.get("spawned_by_tool_call_id")
    }


def subagent_calls(rec: dict) -> dict[str, str]:
    """The model calls a sub-agent made, and which sub-agent made each."""
    return {
        str(ref["model_call_id"]): str(inv["invocation_id"])
        for inv in invocations(rec)
        if inv.get("parent_invocation_id")
        for ref in inv.get("model_calls") or ()
        if isinstance(ref, dict) and ref.get("model_call_id")
    }


def named_spawns(events, started: set[str]):
    """Each event, naming the sub-agents its tool call started."""
    for ev in events:
        if not isinstance(ev, Answer):
            yield ev
            continue
        ids = [
            tc["call_id_source"]
            for tc in ev.value.get("tool_calls") or ()
            if tc.get("call_id_source") in started
        ]
        yield replace(ev, refs={**ev.refs, "spawned": ids}) if ids else ev


def _split(conversation: list) -> tuple[list, list]:
    """What the sub-agent was given (its leading system, developer and user turns), and the rest."""
    n = 0
    for it in conversation:
        if not (isinstance(it, dict) and it.get("role") in SENT_ROLES):
            break
        n += 1
    return conversation[:n], conversation[n:]


def _system_turns(rec: dict, inv: dict) -> list[dict]:
    """The system and developer turns opening a sub-agent's first request, which its
    conversation does not keep."""
    if any(
        isinstance(it, dict) and it.get("role") in INSTRUCTION_ROLES
        for it in inv.get("conversation") or ()
    ):
        return []
    first = next(
        (
            str(ref["model_call_id"])
            for ref in inv.get("model_calls") or ()
            if isinstance(ref, dict) and ref.get("model_call_id")
        ),
        None,
    )
    call = next(
        (
            c
            for c in trajectory_calls(rec)
            if isinstance(c, dict) and str(c.get("model_call_id")) == first
        ),
        None,
    )
    out = []
    for it in call_items(call, stated_dialects(rec)):
        if not (isinstance(it, dict) and it.get("role") in INSTRUCTION_ROLES):
            break
        out.append(it)
    return out


def _child_events(rec: dict, key: tuple, line: int, inv: dict, timing: dict, started: set):
    iid = str(inv["invocation_id"])
    lane = ("subagent", iid)
    spawned_by = inv.get("spawned_by_tool_call_id")
    yield Answer(
        "rollout.events",
        {
            "source_event_id": f"{iid}:spawn",
            "read_seq": f"{line:09d}:",
            "started_at": dig(timing, str(spawned_by), "started_at") if spawned_by else None,
            "role": "harness",
            "kind": "system",
            "kind_raw": "subagent_spawn",
            "parts": [
                part(
                    dumps(
                        {k: v for k, v in inv.items() if k not in ("conversation", "model_calls")}
                    ),
                    origin="subagent_meta",
                )
            ],
            "tool_calls": [],
            "actor": "subagent",
        },
        key=key,
        input_key=f"{line}:{iid}:spawn",
        keys={"agent_call": str(spawned_by)} if spawned_by else {},
        lane=lane,
    )
    for j, it in enumerate(_system_turns(rec, inv)):
        yield Answer(
            "rollout.events",
            {
                "source_event_id": f"{iid}:system:{j}",
                "read_seq": f"{line:09d}:0!{j:09d}",
                "role": it.get("role"),
                **event_kind("message"),
                "parts": text_parts(it.get("content")),
                "side": "request",
                "actor": "subagent",
            },
            key=key,
            input_key=f"{line}:{iid}:system:{j}",
            lane=lane,
        )
    given, rest = _split(list(inv.get("conversation") or ()))
    exchange = {"responses_create_params": {"input": given}, "response": {"output": rest}}
    for ev in named_spawns(_responses_events(exchange, key, line, timing), started):
        v = ev.value
        yield replace(
            ev,
            value=v
            | {
                # Item ids restart in each invocation.
                "source_event_id": f"{iid}:{v['source_event_id']}",
                "read_seq": f"{line:09d}:{v['read_seq'].split(':', 1)[1]}",
                "actor": "subagent",
            },
            input_key=f"{line}:{iid}:{ev.input_key}",
            lane=lane,
        )


def gym_invocations(src: Sources):
    doc = primary_stream(src.docs(0))
    if doc is None:
        return
    for item in records(doc, "rollout.events"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        key = _gym_key(rec, line)
        timing = trajectory_timing(rec)
        started = spawns(rec)
        for inv in invocations(rec):
            if inv.get("parent_invocation_id"):
                yield from _child_events(rec, key, line, inv, timing, started)


GYM_INVOCATIONS = Input(
    name="gym_invocations",
    via=gym_invocations,
    reads=(At(RUN, STREAMS),),
    mints=(
        Key(
            "agent_call",
            optional=True,
            because="the tool call that started a sub-agent, as its invocation names it; only a "
            "sub-agent's opening row carries one, and an integration that cannot link a "
            "sub-agent to its call names none",
        ),
    ),
    names=(
        Key(
            "spawned",
            optional=True,
            because="a sub-agent's tool call names the sub-agents it started in turn",
        ),
    ),
    answers={
        "rollout.events": Card.ANY(
            because=(
                "`ng_trajectory` is attached only when observability is on, and most rollouts "
                "start no sub-agent"
            ),
        ),
    },
)
