# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rollout's conversation: the stream's transcript, the system turns the proxy added, and
Gym's per-call capture (`ng_model_call_capture`, `ng_trajectory`)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from evalrdl_readers.core import Answer, At, Card, Input, Key, Sources, Unreadable
from evalrdl_schema.text import text_key

from ..common import (
    args_text,
    dig,
    dumps,
    event_kind,
    part,
    primary_stream,
    records,
    text_parts,
    tool_call,
)
from ..prompts import (
    INSTRUCTION_ROLES,
    PAYLOAD_FIELDS,
    SENT_ROLES,
    PromptEncoder,
    fields_named,
    item_keys,
    item_text,
    producer_id,
    shared_item_keys,
    unread_messages,
)
from .agents.opencode import _opencode_timing
from .agents.tau2 import _tau2_messages
from .calls import (
    call_dialect,
    call_items,
    captured_calls,
    endpoint,
    stated_dialects,
    trajectory_calls,
)
from .invocations import named_spawns, spawns, subagent_calls, trajectory_timing
from .landmarks import RUN, STREAMS
from .record import _gym_key, _responses_events

# ------------------------------------------------------------------- translators


def capture_calls(call: dict) -> list[dict]:
    """The tool calls a captured model call produced. Columns of `tool_calls`."""
    out = []
    for tc in call.get("tool_calls") or ():
        if not isinstance(tc, dict):
            continue
        out.append(
            tool_call(
                tc.get("name") or dig(tc, "function", "name"),
                tc.get("call_id") or tc.get("id"),
                args_text(tc.get("arguments")),
            )
        )
    return out


def _request_envelope(call: dict, req: dict | None) -> list[dict]:
    """The call's parameters as one `request` part; sampling parameters are not columns."""
    req = req or {}
    # Inline fields first, since most calls have no sidecar; fields that already have
    # a column are left out.
    bits = [
        f"{k}={call[k]}"
        for k in (
            "dialect",
            "response_status",
            "status_code",
            "tokens_reasoning",
            "tokens_total",
            "latency_ttft_ms",
            "cache_hit",
            "cached_tokens",
            "cache_creation_tokens",
            "error_category",
        )
        if call.get(k) is not None
    ]
    bits += [f"{k}={req[k]}" for k in sorted(req) if k not in PAYLOAD_FIELDS and req[k] is not None]
    tools = [
        ((t.get("function") or {}).get("name") or t.get("name"))
        for t in (req.get("tools") or [])
        if isinstance(t, dict)
    ]
    if tools:
        bits.append(f"{len(tools)} tool(s) offered: " + ", ".join(str(t) for t in tools))
    return [part(" · ".join(bits), origin="request")] if bits else []


def _answer_items(rec: dict | None) -> list[dict]:
    """The items a call's response answered with, in either dialect: `output`, or each
    choice's message."""
    resp = (rec or {}).get("response")
    if not isinstance(resp, dict):
        return []
    items = resp.get("output") or [
        c.get("message") for c in resp.get("choices") or () if isinstance(c, dict)
    ]
    return [it for it in items or () if isinstance(it, dict)]


def _produced_ids(rec: dict | None) -> list:
    """The `item_id` keys of what a call's response produced: a sidecar record or a
    `ng_trajectory.model_calls` entry, which both keep the response."""
    return [k for it in _answer_items(rec) for k in item_keys(it)]


def _call_args(call: dict) -> dict:
    """A tool call's arguments, in either dialect, as an object; {} where they do not parse."""
    raw = (
        (call.get("function") or {}).get("arguments")
        if "function" in call
        else call.get("arguments")
    )
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _answer_text_keys(rec: dict | None) -> list[str]:
    """The `text_key`s of what a call's response said that no id states: each message it
    answered with that carries no id, and each string a tool call it made passes on (a user
    simulator's message is its tool's `content`)."""
    out: list[str] = []
    for it in _answer_items(rec):
        texts = [] if it.get("id") else [item_text(it)]
        calls = it.get("tool_calls") or ([it] if it.get("type") == "function_call" else [])
        for c in calls if isinstance(calls, list) else ():
            if isinstance(c, dict):
                texts += [v for v in _call_args(c).values() if isinstance(v, str)]
        for t in texts:
            if (k := text_key(t)) and k not in out:
                out.append(k)
    return out


def _sidecar_slim(rec: dict) -> dict:
    """What `_request_envelope` and `_produced_ids` read of a sidecar record; captures run to GBs."""
    req = rec.get("request")
    if isinstance(req, dict):
        req = {k: v for k, v in req.items() if k not in PAYLOAD_FIELDS} | (
            {
                "tools": [
                    {"name": (t.get("function") or {}).get("name") or t.get("name")}
                    for t in req["tools"]
                    if isinstance(t, dict)
                ]
            }
            if req.get("tools")
            else {}
        )
    return {
        "request": req,
        "response": {"output": [{"id": k} for k in _produced_ids(rec)]},
        "text_keys": _answer_text_keys(rec),
    }


def _exchange_requests(rec: dict):
    """(model_call_id, claimed, items) for each captured call of the rollout's own agent, in
    order: whether the exchange holds what it answered, and its request's items. A sub-agent's
    calls and a tau2 record's are left out: their requests are not the exchange's."""
    if _tau2_messages(rec) is not None:
        return
    outputs = list(dig(rec, "response", "output") or ())
    for act in rec.get("ng_activations") or ():
        outputs += dig(act, "response", "output") or () if isinstance(act, dict) else ()
    held = {k for it in outputs for k in item_keys(it)}
    delegated = subagent_calls(rec)
    stated = stated_dialects(rec)
    for call in trajectory_calls(rec):
        if not isinstance(call, dict) or not call.get("model_call_id"):
            continue
        if call["model_call_id"] in delegated:
            continue
        claims = any(k in held for it in _answer_items(call) for k in item_keys(it))
        yield call["model_call_id"], claims, call_items(call, stated)


def _sent_first(rec: dict) -> dict[str, list[str]]:
    """For each captured call the exchange claims, the `item_id` keys of the items its
    request carries that no earlier call's request did. A call another actor made (a judge,
    a side call) is left out."""
    seen: set[str] = set()
    sent: dict[str, list[str]] = {}
    for cid, claims, items in _exchange_requests(rec):
        new = []
        for it in items:
            for k in item_keys(it):
                if k not in seen:
                    seen.add(k)
                    new.append(k)
        if claims and new:
            sent[cid] = new
    return sent


def _sent_first_texts(rec: dict) -> dict[str, list[str]]:
    """For each captured call the exchange claims, the `text_key`s of the messages without an
    id its request carries that no earlier claimed call's request did."""
    seen: set[str] = set()
    sent: dict[str, list[str]] = {}
    for cid, claims, items in _exchange_requests(rec):
        if not claims:
            continue
        new = []
        for it in items:
            if isinstance(it, dict) and not item_keys(it) and (k := text_key(item_text(it))):
                if k not in seen:
                    seen.add(k)
                    new.append(k)
        if new:
            sent[cid] = new
    return sent


def _inline_event(call: dict, sidecar: dict) -> dict:
    """A `model_call` event from the inline capture's account of the call."""
    return {
        "source_event_id": call["model_call_id"],
        # No store kind for a wire call; `lane_kind` marks it.
        **event_kind("model_call"),
        "started_at": call.get("started_at"),
        "ended_at": call.get("completed_at"),
        "latency_s": (
            call["latency_total_ms"] / 1000.0 if call.get("latency_total_ms") is not None else None
        ),
        # Time to first token; independent of `latency_s`.
        "latency_ttft_ms": call.get("latency_ttft_ms"),
        "model_name": call.get("model") or dig(call, "model_ref", "name"),
        # `chat` fills finish_reason, `responses` fills response_status;
        # error_category is the only reason on unfinished calls.
        "finish_reason": (
            call.get("finish_reason") or call.get("response_status") or call.get("error_category")
        ),
        "prompt_tokens": call.get("tokens_in"),
        "completion_tokens": call.get("tokens_out"),
        "cached_tokens": call.get("cached_tokens"),
        "reasoning_tokens": call.get("tokens_reasoning"),
        "request_idx": call.get("call_index"),
        # Joins to the serving log; resolves only in some folders, so not an edge.
        "response_id": call.get("response_id"),
        # No content parts: reasoning text duplicates the conversation lane.
        # `tool_calls` stays; its `call_id_source` joins to the conversation.
        "tool_calls": capture_calls(call),
        # Sidecar adds only request params; its payloads restate the conversation.
        "parts": _request_envelope(call, (sidecar.get(call["model_call_id"]) or {}).get("request")),
    }


#: Whose call wrote a tau2 message, by the message's role.
_TAU2_ACTOR = {"assistant": "main", "user": "user_simulator"}


def gym_inline_calls(src: Sources):
    """Model-call events from the inline capture, joined to the sidecar by `model_call_id`, each
    carrying the request it sent.

    Inline and sidecar are complements: Gym strips payloads from the inline copy. Only
    `ng_trajectory.model_calls` keeps a call's request whole; it is encoded onto the inline
    event of the same call, and the messages it is the first to send become
    `rollout.prompt_messages` rows. A request with no inline event to land on, or in a dialect
    no reader states, is reported rather than dropped."""
    # One input for both files: no Merge combines halves of one row across inputs.
    sidecar: dict[str, dict] = {}
    # Both sidecar depths; an `At` whose index is not read stays UNREAD.
    for doc in list(src.docs(1)) + list(src.docs(2)):
        for item in records(doc, "rollout.events"):
            if isinstance(item, Unreadable):
                yield item
                continue
            _line, rec = item
            if rec.get("model_call_id"):
                sidecar[rec["model_call_id"]] = _sidecar_slim(rec)
    # Which messages are new is a fact about the rollout, which may span several records.
    encoders: dict[tuple, PromptEncoder] = {}
    # A record rewritten after a retry repeats the calls of the one before it.
    named: set[tuple] = set()
    for doc in src.docs(0):
        for item in records(doc, "rollout.events"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            key = _gym_key(rec, line)
            inline = captured_calls(rec)
            stated = stated_dialects(rec)
            requests: dict[str, dict] = {}
            outputs: dict[str, list] = {}
            answered: dict[str, list] = {}
            enc = encoders.setdefault(key, PromptEncoder())
            for ci, call in enumerate(trajectory_calls(rec)):
                if not isinstance(call, dict):
                    continue
                cid = call.get("model_call_id")
                if cid:
                    outputs[cid] = _produced_ids(call)
                    answered[cid] = _answer_text_keys(call)
                request = call.get("request")
                if not isinstance(request, dict) or not request:
                    continue
                # The one report of a request not landed, or of messages no reader of it takes:
                # every reader reads the stated field alike, an unlanded request's too.
                if cid not in stated:
                    yield Unreadable(
                        "rollout.events",
                        where=f"{doc.relpath}:{line}",
                        detail=f"model call {cid or f'at position {ci}'} has a request in "
                        "`ng_trajectory.model_calls` but no call in `ng_model_call_capture.calls` "
                        "to land it on",
                    )
                    continue
                dialect = call_dialect(call, stated)
                if dialect is None:
                    yield Unreadable(
                        "rollout.events",
                        where=f"{doc.relpath}:{line}",
                        detail=f"model call {cid}'s request carries neither {fields_named()}, "
                        f"and its stated dialect {stated[cid]!r} is not one a reader knows; its "
                        "request is not landed",
                    )
                    continue
                if why := unread_messages(request, dialect):
                    yield Unreadable(
                        "rollout.events",
                        where=f"{doc.relpath}:{line}",
                        detail=f"model call {cid}'s request {why}",
                    )
                requests[cid], new = enc.encode(request, dialect)
                for msg in new:
                    yield Answer("rollout.prompt_messages", msg, key=key)
            sent = _sent_first(rec)
            sent_texts = _sent_first_texts(rec)
            # A tau2 transcript is its system prompt alone; its messages name their call.
            msgs = _tau2_messages(rec)
            tau2 = msgs is not None
            wrote = {
                dig(m, "raw_data", "id"): _TAU2_ACTOR.get(m.get("role"))
                for m in msgs or ()
                if isinstance(m, dict) and dig(m, "raw_data", "id")
            }
            delegated = subagent_calls(rec)
            # Endpoints keep separate prompt histories, so each is its own lane.
            split = rec.get("ng_activations") or len({endpoint(c) for c in inline}) > 1
            for call in inline:
                cid = call["model_call_id"]
                repeated = (key, cid) in named
                named.add((key, cid))
                # A sub-agent's items are its own lane's, and their ids restart in it.
                own = tau2 or cid in delegated
                produced = [] if own else _produced_ids(sidecar.get(cid)) or outputs.get(cid, [])
                said = (
                    []
                    if own
                    else answered.get(cid) or (sidecar.get(cid) or {}).get("text_keys") or []
                )
                yield Answer(
                    "rollout.events",
                    _inline_event(call, sidecar)
                    | requests.get(cid, {})
                    | ({"actor": actor} if (actor := wrote.get(call.get("response_id"))) else {})
                    | ({"actor": "subagent"} if cid in delegated else {}),
                    key=key,
                    input_key=f"{cid}@{line}" if repeated else cid,
                    keys={"model_call_id": cid}
                    | ({"response_id": call["response_id"]} if call.get("response_id") else {}),
                    refs=({"produced": produced} if produced else {})
                    | ({"sent_in": sent[cid]} if sent.get(cid) else {})
                    | ({"produced_by_text": said} if said else {})
                    | ({"sent_in_by_text": sent_texts[cid]} if sent_texts.get(cid) else {}),
                    lane=("model_call", endpoint(call) if split else ""),
                )
            enc.release()


def _after_output(line: int, i: int) -> str:
    """A `read_seq` after output item `i` of the record and before the next; `-1` is before
    the first output, after the input."""
    return f"{line:09d}:1:{i:09d}:~" if i >= 0 else f"{line:09d}:0:~"


def _before_input(line: int) -> str:
    """A `read_seq` before the record's input: `!` sorts before the `:` every item has there."""
    return f"{line:09d}!"


def _own_call_turns(items: list, where: dict, produced: int, line: int):
    """(position, item) for each system, developer or user item of a request whose call is a
    turn of the exchange and produced its output item `produced` first, placed just after the
    exchange item before it in the request. One before any such item goes before the input if
    it is a leading system turn, else before the first exchange item the request names, or
    before `produced` where the request names none."""
    named = [next((where[k] for k in shared_item_keys(it) if k in where), None) for it in items]
    first = min((i for i in named if i is not None), default=produced)
    last, leading = None, True
    for j, (it, i) in enumerate(zip(items, named, strict=True)):
        if i is not None:
            last = i if last is None else max(last, i)
            continue
        role = it.get("role")
        leading = leading and role in INSTRUCTION_ROLES
        if role not in SENT_ROLES:
            continue
        if last is not None:
            yield j, _after_output(line, last), it
        else:
            yield j, _before_input(line) if leading else _after_output(line, first - 1), it


def _other_call_turns(items: list, line: int):
    """(position, item) for each system turn of a request whose call the exchange does not
    name, so has no place among its outputs: the leading ones before the input, the rest
    after it."""
    leading = True
    for j, it in enumerate(items):
        if it.get("role") not in INSTRUCTION_ROLES:
            leading = False
            continue
        yield j, _before_input(line) if leading else _after_output(line, -1), it


def _lifted_turn(
    call: dict, ci: int, j: int, it: dict, read_seq: str, key: tuple, line: int, refs: dict
) -> Answer:
    """Item `j` of the request of trajectory call `ci`, a turn the model saw that the exchange
    does not hold, as a main-lane event at `read_seq`."""
    return Answer(
        "rollout.events",
        {
            "source_event_id": f"reminder:{ci}:{j}",
            "read_seq": read_seq + f"{j:09d}",
            "started_at": call.get("started_at"),
            "request_idx": line,
            "role": it.get("role"),
            **event_kind("message"),
            "parts": text_parts(it.get("content")),
            "side": "request",
        },
        key=key,
        input_key=f"{line}:reminder:{ci}:{j}",
        refs=refs,
    )


def _request_turns(rec: dict, key: tuple, line: int):
    """Turns the harness added to a model call's request, which the model saw and neither
    `responses_create_params.input` nor `response.output` holds. A request is `input`
    (Responses) or `messages` (chat).

    Whose call it is decides what it adds. A call whose output the exchange holds is one of its
    turns, and adds its system, developer and user turns (an agent that returns tool results
    asynchronously sends them as user turns). Where the exchange holds some calls' output, the
    rest are another actor's (a judge) and add nothing; where it holds none, a call adds its
    system turns only. A tau2 record's calls are the agent's and the user simulator's, and its
    conversation is its own result, so they add nothing.

    Each turn names the model call whose request first carried it (`sent_in`), where that call
    is an event of its own."""
    if _tau2_messages(rec) is not None:
        return
    calls = trajectory_calls(rec)
    output = [it if isinstance(it, dict) else {} for it in dig(rec, "response", "output") or ()]
    where = {k: i for i, it in enumerate(output) for k in shared_item_keys(it)}
    known = {
        str(it.get("content"))
        for it in [*(dig(rec, "responses_create_params", "input") or ()), *output]
        if isinstance(it, dict) and "content" in it
    }
    stated = stated_dialects(rec)
    # The first exchange item each call produced; None for a call the exchange does not name.
    produced = [
        min((where[("id", it.get("id"))] for it in dig(c, "response", "output") or ()
             if isinstance(it, dict) and ("id", it.get("id")) in where), default=None)
        if isinstance(c, dict) else None
        for c in calls
    ]  # fmt: skip
    named_any = any(p is not None for p in produced)
    delegated = subagent_calls(rec)
    for ci, (call, first) in enumerate(zip(calls, produced, strict=True)):
        if not isinstance(call, dict) or (first is None and named_any):
            continue
        if call.get("model_call_id") in delegated:
            continue
        items = [it for it in call_items(call, stated) if isinstance(it, dict)]
        turns = (
            _other_call_turns(items, line)
            if first is None
            else _own_call_turns(items, where, first, line)
        )
        for j, at, it in turns:
            text = str(it.get("content"))
            # A request replays the conversation so far; a turn already placed is not new.
            if text in known:
                continue
            known.add(text)
            yield _lifted_turn(
                call, ci, j, it, at, key, line,
                {"sent_in": [call["model_call_id"]]} if call.get("model_call_id") in stated else {},
            )  # fmt: skip


def _interactive_reminder_events(rec: dict, key: tuple, line: int, output_order: dict):
    """Candidate system reminders ordered against every activation's output. The root
    response repeats only the last activation, so it cannot place earlier reminders."""
    stated = stated_dialects(rec)
    seen = {
        str(it.get("content"))
        for it in dig(rec, "responses_create_params", "input") or ()
        if isinstance(it, dict) and it.get("role") in INSTRUCTION_ROLES
    }
    for ci, call in enumerate(trajectory_calls(rec)):
        if not isinstance(call, dict):
            continue
        positions = [
            output_order[it["id"]]
            for it in dig(call, "response", "output") or ()
            if isinstance(it, dict) and it.get("id") in output_order
        ]
        opening = not positions
        seq = _after_output(line, -1)
        if positions:
            prefix, index = min(positions).rsplit(":", 1)
            seq = f"{prefix}:{int(index) - 1:09d}:~"
        leading = True
        for j, it in enumerate(call_items(call, stated)):
            if not isinstance(it, dict) or it.get("role") not in INSTRUCTION_ROLES:
                leading = False
                continue
            text = str(it.get("content"))
            # A request replays the conversation so far; a turn already placed is not new.
            if text in seen:
                continue
            seen.add(text)
            at = _before_input(line) if opening and leading else seq
            yield _lifted_turn(call, ci, j, it, at, key, line, {})


def _tool_defs(rec: dict, key: tuple):
    """Each distinct tool definition offered, first-seen: the rollout record's
    `responses_create_params.tools`, then each request's.

    `response.tools` is the server's echo of those, re-serialised (`defer_loading` added on
    tau3), so it is read only where `responses_create_params` lists none."""
    offers = [(None, dig(rec, "responses_create_params", "tools") or dig(rec, "response", "tools"))]
    for call in trajectory_calls(rec):
        if isinstance(call, dict):
            offers.append((call.get("model_call_id"), dig(call, "request", "tools")))
    seen: set[str] = set()
    for offered_by, tools in offers:
        for tool in tools if isinstance(tools, list) else ():
            text = dumps(tool)
            if text in seen:
                continue
            seen.add(text)
            fn = tool.get("function") if isinstance(tool, dict) else None
            name = (tool.get("name") if isinstance(tool, dict) else None) or (
                fn.get("name") if isinstance(fn, dict) else None
            )
            yield Answer(
                "rollout.tool_defs",
                {
                    "def_idx": len(seen) - 1,
                    "name": name,
                    "tool_type": tool.get("type") if isinstance(tool, dict) else None,
                    "def_chars": len(text),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "offered_by": offered_by,
                    "def_json": text,
                },
                key=key,
            )


def _continues(step: dict) -> bool:
    """Whether a step delivered its input to the next activation: `continue_episode: true`,
    or `kind: "continue"` in the later shape."""
    return bool(step.get("continue_episode")) or step.get("kind") == "continue"


def _interactive_events(rec: dict, key: tuple, line: int, timing: dict):
    """Each activation is a delta; its preceding step supplies the next delivered input.

    The final response is repeated at the record root for verification. Reading that copy as
    another exchange would both omit earlier activations and count the last response twice.
    Array order is the producer's causal order, independent of the model servers' clocks.
    """
    steps = {
        step["activation_id"]: step
        for step in rec.get("ng_steps") or ()
        if isinstance(step, dict) and "activation_id" in step
    }
    previous = None
    output_order = {}
    candidate_models = set()
    for position, activation in enumerate(rec["ng_activations"]):
        if not isinstance(activation, dict) or not isinstance(activation.get("response"), dict):
            yield Unreadable(
                "rollout.events",
                where=f"record {line}, activation {position}",
                detail="interactive activation has no response object",
            )
            continue
        response = activation["response"]
        if response.get("model"):
            candidate_models.add(response["model"])
        step = steps.get(previous) or {}
        params = (
            rec.get("responses_create_params")
            if position == 0
            else (step.get("responses_create_params") if _continues(step) else None)
        )
        exchange = {"responses_create_params": params, "response": response}
        for event in _responses_events(exchange, key, line, timing, item_keys=True):
            value = event.value
            # Keep native item ids for cross-account links; scope only ids we minted.
            side, index = event.input_key.rsplit(":", 2)[1:]
            items = (params or {}).get("input") if side == "input" else response.get("output")
            item = items[int(index)]
            native_id = (
                producer_id(item["id"]) if isinstance(item, dict) and item.get("id") else None
            )
            value = value | {
                "source_event_id": native_id or f"activation:{line}:{position}:{side}:{index}",
                "read_seq": f"{line:09d}:{position:09d}:" + value["read_seq"].split(":", 1)[1],
                "request_idx": position,
            }
            if side == "input" and position:
                value["actor"] = "harness" if step.get("synthetic") else "environment"
                if step.get("synthetic"):
                    value["parts"] = value["parts"] + [
                        part("Synthetic continuation", origin="harness")
                    ]
            if side == "output" and native_id:
                output_order[native_id] = value["read_seq"]
            yield replace(
                event, value=value, input_key=f"{line}:activation:{position}:{side}:{index}"
            )
        previous = activation.get("activation_id")

    # Only the candidate endpoint can add reminders to its conversation. Simulator/judge
    # system prompts belong to their own captured requests, not to the candidate's input.
    candidate_calls = {
        call["model_call_id"] for call in captured_calls(rec) if endpoint(call) in candidate_models
    }
    calls = [
        call
        for call in trajectory_calls(rec)
        if isinstance(call, dict) and call.get("model_call_id") in candidate_calls
    ]
    scoped = rec | {"ng_trajectory": {"model_calls": calls}}
    yield from _interactive_reminder_events(scoped, key, line, output_order)


def gym_transcript(src: Sources):
    """The responses-api transcript from the rollout stream (all gym shapes but gdpval).

    Of a tau2 record only the system prompt: `gym_tau2` reads the conversation."""
    doc = primary_stream(src.docs(0))
    if doc is None:
        return
    timing = _opencode_timing(src.docs(1))
    for item in records(doc, "rollout.events"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        own = timing | trajectory_timing(rec)
        if rec.get("ng_activations"):
            yield from named_spawns(
                _interactive_events(rec, _gym_key(rec, line), line, own), spawns(rec)
            )
            yield from _tool_defs(rec, _gym_key(rec, line))
            continue
        events = named_spawns(
            _responses_events(rec, _gym_key(rec, line), line, own, item_keys=True), spawns(rec)
        )
        if _tau2_messages(rec) is not None:
            events = (
                ev
                for ev in events
                if ev.value["side"] == "input" and ev.value["role"] in INSTRUCTION_ROLES
            )
        yield from events
        yield from _request_turns(rec, _gym_key(rec, line), line)
        yield from _tool_defs(rec, _gym_key(rec, line))


# ------------------------------------------------------------------------- inputs

GYM_TRANSCRIPT = Input(
    name="gym_transcript",
    via=gym_transcript,
    reads=[At(RUN, STREAMS), At(RUN, "opencode/*/generation.json")],
    mints=(
        Key(
            "item_id",
            optional=True,
            because="the producer's id of an output item of the exchange, or a tool result's "
            "call id; an input item has none",
        ),
        Key(
            "text_key",
            optional=True,
            because="the text-v1 key of a message of the exchange, either side; a call, a result "
            "or an empty message has none. Derived from content, not stated: what joins on it "
            "is inferred",
        ),
    ),
    names=(
        Key(
            "sent_in",
            optional=True,
            because="a turn lifted from a request names the captured call that first sent it",
        ),
        Key(
            "spawned",
            optional=True,
            because="a tool call that started a sub-agent names it, where `ng_trajectory` "
            "links the sub-agent to its call",
        ),
    ),
    answers={
        "rollout.events": Card.ANY(
            because=(
                "a responses-api rollout record usually is its `response.output`, but a rollout "
                "whose sandbox never started is written with empty `input` and `output`, and "
                "gdpval carries no history here"
            ),
        ),
        "rollout.tool_defs": Card.ANY(
            because=(
                "`responses_create_params.tools` (else its echo `response.tools`) and each "
                "captured request's `tools`; a single-turn benchmark offers none, and a record's "
                "`responses_create_params` can list none while its captured requests do"
            ),
        ),
    },
)

GYM_INLINE_CALLS = Input(
    name="gym_inline_calls",
    via=gym_inline_calls,
    mints=(
        Key(
            "model_call_id",
            optional=True,
            because="Gym's id of a captured call, on every inline call; a run without "
            "observability captures none",
        ),
        Key(
            "response_id",
            optional=True,
            because="the model server's id of the call's completion (`chatcmpl-...`); a call "
            "that failed before a response has none",
        ),
    ),
    names=(
        Key(
            "produced",
            optional=True,
            because="the capture sidecar lists the output items a call's response produced; "
            "a call without a sidecar names none",
        ),
        Key(
            "sent_in",
            optional=True,
            because="a call's request names the transcript items it carries for the first "
            "time; a call the exchange does not claim names none",
        ),
        Key(
            "produced_by_text",
            optional=True,
            because="a call names the text keys of what its response said that no id states: "
            "messages without an id, and the strings its tool calls pass on",
        ),
        Key(
            "sent_in_by_text",
            optional=True,
            because="a claimed call names the text keys of the messages without an id its "
            "request carries for the first time",
        ),
    ),
    # Sidecar dir name is set by wrapper config (`model_call_capture_dir`) and may sit one
    # level under a benchmark dir, so match Gym's file name at both depths.
    # `.capture.incomplete` files are empty cut-off markers, ignored.
    reads=[At(RUN, STREAMS), At(RUN, "*/*.capture.jsonl"), At(RUN, "*/*/*.capture.jsonl")],
    answers={
        "rollout.events": Card.ANY(
            because="`ng_model_call_capture` is attached only when observability is on; the rest carry "
            "no `ng_*` key at all",
        ),
        "rollout.prompt_messages": Card.ANY(
            because="only `ng_trajectory.model_calls` keeps a call's request whole. MEASURED over "
            "the reader corpus: 106,950 of its 145,784 calls carry one (96,465 chat, 10,485 "
            "Responses, none in another dialect) and 38,834 none, an older Gym keeping the first "
            "call's only; a record without the trajectory sends none",
        ),
    },
)
