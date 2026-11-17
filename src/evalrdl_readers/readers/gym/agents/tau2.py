# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tau2's own trajectory, `result.messages`: the agent, the user simulator and the tools."""

from __future__ import annotations

import re

from evalrdl_readers.core import Answer, At, Card, Input, Key, Sources, Unreadable

from ...common import args_text, dig, event_kind, primary_stream, records, text_parts, tool_call
from ..landmarks import RUN, STREAMS
from ..record import _gym_key

# ------------------------------------------------------------------- translators


def _tau2_messages(rec: dict) -> list | None:
    """tau2's own trajectory, every actor in order; `None` on any other record shape."""
    msgs = dig(rec, "result", "messages")
    return msgs if isinstance(msgs, list) and msgs else None


_THINK_TAG = re.compile(r"</?think>")


def _tau2_calls(msg: dict) -> list[dict]:
    out = []
    for c in msg.get("tool_calls") or ():
        if not isinstance(c, dict):
            continue
        out.append(tool_call(c.get("name"), c.get("id"), args_text(c.get("arguments"), 1 << 20)))
    return out


def gym_tau2(src: Sources):
    """tau2's `result.messages`: the agent, the user simulator and the tools, in order.

    `response.output` is the agent's side only; the simulator's own tool calls change the
    environment the agent is scored on, and the reasoning is only in `raw_data`."""
    doc = primary_stream(src.docs(0))
    if doc is None:
        return
    for item in records(doc, "rollout.events"):
        if isinstance(item, Unreadable):
            continue
        line, rec = item
        msgs = _tau2_messages(rec)
        if msgs is None:
            continue
        key = _gym_key(rec, line)
        # Each assistant turn is its own model call, so each carries its own usage; the
        # record's line would count only the first call once per rollout.
        call_no = 0
        for i, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "assistant":
                call_no += 1
            calls = _tau2_calls(msg)
            raw = msg.get("raw_data") if isinstance(msg.get("raw_data"), dict) else {}
            # The server wraps it in the `<think>` tags the bridge strips from inline thinking.
            parts = [
                p
                for c in raw.get("choices") or ()
                if isinstance(c, dict)
                for p in text_parts(
                    _THINK_TAG.sub("", dig(c, "message", "reasoning_content") or "").strip()
                    or None,
                    channel="reasoning",
                )
            ]
            parts += text_parts(
                msg.get("content") or None, channel="tool_result" if role == "tool" else "content"
            )
            usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
            yield Answer(
                "rollout.events",
                {
                    "source_event_id": f"tau2:{i}",
                    # After the system prompt, which is this record's input side.
                    "read_seq": f"{line:09d}:1:{i:09d}",
                    "started_at": msg.get("timestamp"),
                    "request_idx": call_no,
                    "role": role,
                    **event_kind(
                        "tool_result" if role == "tool" else "tool_call" if calls else "message"
                    ),
                    # The simulator's tokens are not the agent's: assistant turns only. It is
                    # another model, and tau2 reports its cost apart (`user_cost`).
                    **(
                        {
                            "model_name": raw.get("model"),
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                        }
                        if role == "assistant"
                        else {}
                    ),
                    "parts": parts,
                    "tool_calls": calls,
                    **(
                        {"answers_call_ids": [msg["id"]]}
                        if role == "tool" and msg.get("id")
                        else {}
                    ),
                },
                key=key,
                input_key=f"{line}:tau2:{i}",
                refs={"produced": [raw["id"]]} if raw.get("id") else {},
            )


# ------------------------------------------------------------------------- inputs

GYM_TAU2 = Input(
    name="gym_tau2",
    via=gym_tau2,
    reads=[At(RUN, STREAMS)],
    names=(
        Key(
            "produced",
            optional=True,
            because="a model-written message keeps its completion in `raw_data`, whose `id` "
            "is the captured call's `response_id`; a scripted turn or a tool result has none",
        ),
    ),
    answers={
        "rollout.events": Card.ANY(
            because=(
                "only tau2 records carry `result.messages`, tau2's own trajectory of every "
                "actor; `response.output` holds only the agent's side, so the user simulator's "
                "tool calls and results and the `raw_data` reasoning are only here"
            ),
        )
    },
)
