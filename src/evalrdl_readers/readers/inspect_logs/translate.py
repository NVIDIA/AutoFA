# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""inspect_ai's log files, read: each sample's score, task, repeat, prompt, usage and
conversation, and the run's model and task count."""

from __future__ import annotations

import json

from evalrdl_readers.core import Answer, Sources

from ..common import as_float, dig, event_kind
from ..dialects.chat import chat_messages


def in_logs(src: Sources):
    """One input for all log files, since a run-grain question may be answered only once.

    Wanted: one input per log file; the model reduction (one distinct value, or none) is
    hidden in Python here.
    """
    models, tasks = set(), []
    for doc in src.docs(0):
        body = doc.json() or {}
        ev = body.get("eval") or {}
        models.add(ev.get("model"))
        tasks.append(ev.get("task"))
        for sample in body.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            key = (f"{ev.get('task')}/{sample.get('id')}/{sample.get('epoch')}",)
            yield Answer("rollout.score", as_float(dig(sample, "score", "value")), key=key)
            yield Answer("rollout.task", f"{ev.get('task')}:{sample.get('id')}", key=key)
            yield Answer("rollout.repeat", sample.get("epoch"), key=key)
            yield Answer("rollout.prompt", sample.get("input"), key=key)
            usage = dig(sample, "output", "usage") or {}
            yield Answer("rollout.n_prompt_tokens", usage.get("input_tokens"), key=key)
            yield Answer("rollout.n_completion_tokens", usage.get("output_tokens"), key=key)
            if ev.get("created") is not None:
                yield Answer("rollout.started_at", str(ev["created"]), key=key)
            yield from _events(sample.get("messages"), key)
    yield Answer(
        "run.model",
        next(iter(models)) if len(models) == 1 else None,
        note=None
        if len(models) == 1
        else f"{len(models)} distinct models across {len(tasks)} logs",
    )
    yield Answer("run.n_tasks_declared", len(tasks))


def _openai(m):
    """inspect names a call's function bare, with its arguments as an object."""
    if not isinstance(m, dict) or not m.get("tool_calls"):
        return m
    calls = []
    for c in m["tool_calls"]:
        fn = c.get("function") if isinstance(c, dict) else None
        if isinstance(fn, str):
            args = c.get("arguments")
            c = {
                **c,
                "function": {
                    "name": fn,
                    "arguments": args
                    if isinstance(args, str) or args is None
                    else json.dumps(args, ensure_ascii=False),
                },
            }
        calls.append(c)
    return {**m, "tool_calls": calls}


def _events(messages, key: tuple):
    open_calls: list[str] = []
    for i, t in enumerate(chat_messages([_openai(m) for m in messages or ()])):
        open_calls += [c["call_id_source"] for c in t["tool_calls"] if c.get("call_id_source")]
        if t["role"] == "tool" and not t["answers"] and open_calls:
            # inspect's tool message names no call; it answers the oldest one still open.
            t = {**t, "answers": [open_calls[0]]}
        for a in t["answers"]:
            if a in open_calls:
                open_calls.remove(a)
        yield Answer(
            "rollout.events",
            {
                "source_event_id": f"message:{i:09d}",
                "read_seq": f"{i:09d}",
                "role": t["role"],
                **event_kind(t["kind"]),
                "parts": t["parts"],
                "tool_calls": t["tool_calls"],
                **({"answers_call_ids": t["answers"]} if t["answers"] else {}),
            },
            key=key,
            input_key=f"message:{i:09d}",
        )
