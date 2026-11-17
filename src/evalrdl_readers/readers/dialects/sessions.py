# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An agent's session log: claude-code and codex records, interleaved within one file."""

from __future__ import annotations

from evalrdl_readers.core import Decoder

from ..common import dumps, part, text_parts, tool_call
from .anthropic import anthropic_message
from .responses import responses_calls, responses_parts


def _record_type(*names: str):
    """The decider for a session record that names its own shape in `type`."""
    return lambda rec: isinstance(rec, dict) and rec.get("type") in names


SESSION_MESSAGE = Decoder(
    "anthropic-message",
    decides=lambda rec: isinstance(rec, dict) and isinstance(rec.get("message"), dict),
    because=(
        "claude-code writes `{type, message: {role, content: [typed blocks]}}`; the "
        "`message` key is the recognition, since no other shape here has one"
    ),
)
SESSION_ATTACHMENT = Decoder(
    "session-attachment",
    decides=_record_type("attachment"),
    because=(
        "claude-code spills a skill listing or a file it read into a record of its own, "
        "`{type: attachment, attachment: {type, content}}`, with no `message` and its text "
        "one key down"
    ),
)
SESSION_LAST_PROMPT = Decoder(
    "session-last-prompt",
    decides=_record_type("last-prompt"),
    because=(
        "the resume marker `{type, lastPrompt, sessionId}` claude-code appends so a session "
        "can be continued: harness bookkeeping, not a turn, whose `lastPrompt` is a truncated "
        "rendering of a prompt the file already holds in full"
    ),
)
SESSION_QUEUE_OP = Decoder(
    "session-queue-operation",
    decides=_record_type("queue-operation"),
    because=(
        "a prompt entering or leaving claude-code's input queue, `{type, operation, "
        "timestamp, sessionId, content?}`: harness bookkeeping, since the enqueued `content` "
        "repeats a `user` turn already in the session lane while the queue timestamps are a "
        "fact only these records hold"
    ),
)
SESSION_SYSTEM = Decoder(
    "session-system",
    decides=_record_type("system"),
    because=(
        "claude-code's marker that it compacted the conversation, `{type: system, subtype: "
        "compact_boundary, compactMetadata}`. The turn after a compaction names it as its "
        "parent, and it alone records what the compaction cost and the `logicalParentUuid` "
        "that re-parents across the boundary"
    ),
)
CODEX_RESPONSE_ITEM = Decoder(
    "codex-response-item",
    decides=_record_type("response_item"),
    because=(
        "codex wraps one responses-API item in `{timestamp, ordinal, type, payload}`; "
        "`payload` is the item, so the responses dialect reads it"
    ),
)
CODEX_EVENT_MSG = Decoder(
    "codex-event-msg",
    decides=_record_type("event_msg"),
    because=(
        "codex's own account of the turns it took, `{payload: {type, ...}}`, which renders "
        "a function_call and its output as one CommandExecution: a second account of the "
        "same minutes, so it goes in its own lane"
    ),
)

CODEX_COMPACTED = Decoder(
    "codex-compacted",
    decides=_record_type("compacted"),
    because=(
        "codex's record that it compacted the conversation, `{type: compacted, payload: "
        "{message}}`: the summary it replaced the history with, which no other record holds"
    ),
)
CODEX_SETUP = Decoder(
    "codex-setup",
    decides=_record_type("session_meta", "turn_context", "world_state"),
    because=(
        "codex's session header, the settings of each turn (model, reasoning effort, "
        "sandbox and approval policy, base instructions) and, from 0.15x, the state of the "
        "world it renders into the prompt (environments, instructions files): harness "
        "bookkeeping, not a turn"
    ),
)
CODEX_TOKEN_USAGE = Decoder(
    "codex-token-usage",
    decides=_record_type("token_usage_record"),
    because=(
        "codex's record, from 0.15x, of what one response cost, `{type: token_usage_record, "
        "payload: {response_id, usage, turn_token_usage, thread_token_usage}}`: bookkeeping, "
        "since the `token_count` event beside it carries the same usage"
    ),
)

#: Every shape a session record may have, shared by the three session inputs.
SESSION_CANDIDATES = (
    SESSION_MESSAGE,
    SESSION_ATTACHMENT,
    SESSION_LAST_PROMPT,
    SESSION_QUEUE_OP,
    SESSION_SYSTEM,
    CODEX_RESPONSE_ITEM,
    CODEX_EVENT_MSG,
    CODEX_COMPACTED,
    CODEX_SETUP,
    CODEX_TOKEN_USAGE,
)

#: codex's UI item types, in the store's closed event vocabulary.
CODEX_ITEM_KIND = {
    "UserMessage": "message",
    "AgentMessage": "message",
    "Reasoning": "reasoning",
    "CommandExecution": "tool_call",
    "FileChange": "tool_result",
}


def _codex_item(item: dict) -> dict:
    """One completed codex UI item: its parts, its calls and what it is."""
    kind = item.get("type")
    if kind == "Reasoning":
        return {
            "parts": text_parts(item.get("summary_text"), channel="reasoning", origin="summary"),
            "tool_calls": [],
        }
    if kind == "CommandExecution":
        cmd = " ".join(str(c) for c in item.get("command") or ())
        return {
            "parts": text_parts(
                item.get("aggregated_output"), channel="tool_result", origin="observation"
            ),
            "tool_calls": [
                tool_call("exec_command", item.get("id"), cmd, status=item.get("status"))
            ],
        }
    if kind == "FileChange":
        diffs = [
            c.get("unified_diff")
            for c in (item.get("changes") or {}).values()
            if isinstance(c, dict)
        ]
        return {
            "parts": text_parts(
                [item.get("stdout"), *diffs], channel="tool_result", origin="observation"
            ),
            "tool_calls": [],
        }
    blocks = item.get("content")
    if isinstance(blocks, list):
        # Not `part(part_type=...)`: codex's `Text`/`text` are not store part types.
        blocks = [b.get("text") if isinstance(b, dict) else b for b in blocks]
    return {"parts": text_parts(blocks), "tool_calls": []}


#: Where codex's older turn log keeps a turn's text: the reasoning summary in `text`, a message
#: in `message`, the turn's last answer in `last_agent_message`.
CODEX_REASONING_EVENTS = {"agent_reasoning": "summary", "agent_reasoning_raw_content": "raw"}


def _turn_log_parts(payload: dict) -> list[dict]:
    kind = payload.get("type")
    if kind in CODEX_REASONING_EVENTS:
        return text_parts(
            payload.get("text"), channel="reasoning", origin=CODEX_REASONING_EVENTS[kind]
        )
    return text_parts(payload.get("message") or payload.get("last_agent_message"))


def session_shape(dec, rec: dict) -> dict:
    """One session record as `{id, role, kind, parts, tool_calls, answers, model_name, lane}`.

    `answers` holds the call ids a result answers; the bridge resolves them to positions.
    `answers=None` means not known (codex UI lane), unlike `[]`, which means answers none."""
    if dec.name == "anthropic-message":
        msg = rec.get("message") or {}
        got = anthropic_message(msg)
        return {
            "id": rec.get("uuid"),
            "role": msg.get("role"),
            "kind": got["kind"],
            "parts": got["parts"],
            "tool_calls": got["tool_calls"],
            "answers": got["answers"],
            "model_name": msg.get("model"),
            "lane": "session",
        }
    if dec.name == "session-attachment":
        att = rec.get("attachment") or {}
        return {
            "id": rec.get("uuid"),
            "role": "attachment",
            "kind": att.get("type") or "attachment",
            "parts": text_parts(att.get("content"), origin="attachment"),
            "tool_calls": [],
            "answers": [],
            "model_name": None,
            "lane": "session",
        }
    if dec.name == "session-system":
        meta = rec.get("compactMetadata") or {}
        return {
            "id": rec.get("uuid"),
            "role": "harness",
            "kind": rec.get("subtype") or "system",
            "parts": [
                part(
                    dumps(
                        {
                            "content": rec.get("content"),
                            "compactMetadata": meta,
                            "logicalParentUuid": rec.get("logicalParentUuid"),
                        }
                    ),
                    origin="boundary",
                )
            ],
            "tool_calls": [],
            "answers": [],
            "model_name": None,
            "lane": "session",
        }
    if dec.name == "codex-compacted":
        return {
            "id": None,
            "role": "harness",
            "kind": "compacted",
            "parts": text_parts((rec.get("payload") or {}).get("message"), origin="summary"),
            "tool_calls": [],
            "answers": [],
            "model_name": None,
            "lane": "session",
        }
    if dec.name in (
        "session-queue-operation",
        "session-last-prompt",
        "codex-setup",
        "codex-token-usage",
    ):
        # Bookkeeping: the whole record as JSON; no column holds its parameters.
        return {
            "id": None,
            "role": "harness",
            "kind": rec.get("operation") or rec.get("type"),
            "parts": [part(dumps(rec), origin="bookkeeping")],
            "tool_calls": [],
            "answers": [],
            "model_name": None,
            "lane": "bookkeeping",
        }
    payload = rec.get("payload") or {}
    if dec.name == "codex-response-item":
        # A `function_call_output` names the `call_id` it answers.
        return {
            "id": payload.get("id"),
            "role": payload.get("role") or payload.get("type"),
            "kind": payload.get("type"),
            "parts": responses_parts(payload),
            "tool_calls": responses_calls(payload),
            "answers": (
                [payload["call_id"]]
                if payload.get("type") == "function_call_output" and payload.get("call_id")
                else []
            ),
            "model_name": None,
            "lane": "session",
        }
    item = payload.get("item") or {}
    got = (
        _codex_item(item)
        if payload.get("type") == "item_completed"
        else {"parts": _turn_log_parts(payload), "tool_calls": []}
    )
    return {
        "id": item.get("id"),
        "role": item.get("type") or payload.get("type"),
        "kind": CODEX_ITEM_KIND.get(item.get("type"), payload.get("type")),
        "parts": got["parts"],
        "tool_calls": got["tool_calls"],
        "answers": None,
        "model_name": None,
        "lane": "codex-ui",
    }
