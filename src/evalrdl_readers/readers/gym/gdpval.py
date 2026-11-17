# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gdpval's deliverables tree: one `task_<uuid>/repeat_<n>` directory per rollout."""

from __future__ import annotations

from evalrdl_readers.core import Answer, At, Card, Input, Key, Sources

from ..common import artifact_row, event_kind, text_parts
from ..dialects.chat import chat_messages
from ._pre import GYM_IGNORE_FILES
from .landmarks import DELIVERABLES

# ------------------------------------------------------------- gdpval translators


def deliverables_dir(task_id, repeat) -> str:
    """`task_<task_id>/repeat_<n>`: where a gdpval rollout's files are, relative to the cache."""
    return f"task_{task_id}/repeat_{repeat}"


def _deliv_key(site) -> tuple[str]:
    """`<task_uuid>/repeat_<n>` from the site path; the task dir has no marker file, so no landmark."""
    return ("/".join(site.path.split("/")[-2:]),)


def _clock(value) -> str | None:
    return None if value is None else str(value)


def gdpval_history(src: Sources):
    """The gdpval agent's own multi-turn transcript, in a lane beside the stream's record."""
    if not src.has(0):
        return
    body = src.doc(0).json()
    if not isinstance(body, list):
        return
    key = _deliv_key(src.site)
    task, _, repeat = key[0].partition("/")
    # `task_<uuid>/repeat_<n>`: the directory names the task and the repeat, which a stream
    # record naming this directory also states; one no record names has only these.
    if task.startswith("task_") and repeat.startswith("repeat_") and repeat[7:].isdigit():
        yield Answer("rollout.task", task[5:], key=key)
        yield Answer("rollout.repeat", int(repeat[7:]), key=key)
    for i, msg in enumerate(body):
        if not isinstance(msg, dict):
            continue
        turn = chat_messages([msg])[0]
        # Stirrup returns a tool's output as a `user` turn carrying the call's id.
        result = bool(msg.get("tool_call_id")) and msg.get("role") != "assistant"
        # One assistant turn is one model call; its completion counts its reasoning.
        usage = msg.get("token_usage") if isinstance(msg.get("token_usage"), dict) else {}
        answered = usage.get("answer")
        tokens = (
            {}
            if answered is None
            else {
                "prompt_tokens": usage.get("input"),
                "completion_tokens": answered + (usage.get("reasoning") or 0),
                "reasoning_tokens": usage.get("reasoning"),
            }
        )
        yield Answer(
            "rollout.events",
            {
                "source_event_id": str(i),
                # Zero-padded so it sorts numerically.
                "read_seq": f"{i:09d}",
                # Stirrup's clock is the process's monotonic one.
                "started_at": _clock(
                    msg.get("tool_start_time" if result else "request_start_time")
                ),
                "ended_at": _clock(msg.get("tool_end_time" if result else "request_end_time")),
                "role": msg.get("role"),
                **event_kind(
                    "tool_result"
                    if result
                    else "tool_call"
                    if turn["tool_calls"]
                    else msg.get("kind") or msg.get("role")
                ),
                "parts": text_parts(
                    msg.get("content"), channel="tool_result" if result else "content"
                ),
                "tool_calls": turn["tool_calls"],
                **tokens,
                **({"answers_call_ids": turn["answers"]} if result and turn["answers"] else {}),
            },
            key=key,
            input_key=str(i),
            lane=("history", ""),
        )


def gdpval_metadata(src: Sources):
    if not src.has(0):
        return
    body = src.doc(0).json() or {}
    key = _deliv_key(src.site)
    usage = body.get("token_usage") or []
    yield Answer("rollout.n_model_calls", len(usage), key=key)


def gdpval_finish(src: Sources):
    if not src.has(0):
        return
    body = src.doc(0).json() or {}
    yield Answer(
        "rollout.stop_reason", (body.get("reason") or "")[:200] or None, key=_deliv_key(src.site)
    )


def gdpval_repeat(src: Sources):
    """The repeat directory itself, whatever it holds: its name is the join to the stream."""
    key = _deliv_key(src.site)
    yield Answer("rollout.deliverables_dir", key[0], key=key, keys={"deliv_dir": key[0]})


def gdpval_deliverables(src: Sources):
    key = _deliv_key(src.site)
    for doc in src.docs(0):
        if doc.name in GYM_IGNORE_FILES:
            continue
        yield Answer(
            "rollout.artifacts",
            artifact_row(doc, role="output"),
            key=key,
            input_key=doc.relpath,
        )


def gdpval_inputs(src: Sources):
    key = _deliv_key(src.site)
    for doc in src.docs(0):
        yield Answer(
            "rollout.artifacts",
            artifact_row(doc, role="reference"),
            key=key,
            input_key=doc.relpath,
        )


# ------------------------------------------------------------------------- inputs

GDPVAL_HISTORY = Input(
    name="gdpval_history",
    via=gdpval_history,
    reads=[At(DELIVERABLES, "history.json")],
    answers={
        "rollout.events": Card.ANY(
            because=(
                "`history.json` is the stirrup agent's own transcript. The Hermes agent writes "
                "none: its transcript is the stream record's `response`, and its repeat "
                "directory holds only the deliverables"
            ),
        ),
        "rollout.task": Card.ANY(
            because=(
                "the directory is `task_<uuid>/repeat_<n>` and the uuid is the task. Where a "
                "stream record names the directory the two agree; where none does it is the "
                "only task id the rollout has"
            ),
        ),
        "rollout.repeat": Card.ANY(
            because="the `repeat_<n>` half of the same directory name",
        ),
    },
)

GDPVAL_METADATA = Input(
    name="gdpval_metadata",
    via=gdpval_metadata,
    reads=[At(DELIVERABLES, "metadata.json")],
    answers={
        "rollout.n_model_calls": Card.ANY(
            because="only the stirrup agent writes `metadata.json`; a Hermes repeat directory holds only the deliverables"
        )
    },
)

GDPVAL_REPEAT = Input(
    name="gdpval_repeat",
    via=gdpval_repeat,
    reads=[At(DELIVERABLES, "*")],
    answers={
        "rollout.deliverables_dir": Card.ONE(
            because="one directory per repeat, and its name is the answer",
        )
    },
    mints=(
        Key(
            "deliv_dir",
            optional=True,  # only gdpval writes a deliverables cache
            because=(
                "`task_<uuid>/repeat_<n>`, the name of the directory, which the stream states "
                "as its record's `task_id` and `_ng_rollout_index`. It is the only join between "
                "the deliverables tree and the rollout stream, and the directory has it whether "
                "it holds the stirrup agent's metadata.json or only the files the agent produced"
            ),
        ),
    ),
)

GDPVAL_FINISH = Input(
    name="gdpval_finish",
    via=gdpval_finish,
    reads=[At(DELIVERABLES, "finish_params.json")],
    answers={
        "rollout.stop_reason": Card.ANY(
            because="written when the agent calls its finish tool; absent where it did not"
        )
    },
)

GDPVAL_DELIVERABLES = Input(
    name="gdpval_deliverables",
    via=gdpval_deliverables,
    reads=[At(DELIVERABLES, "*")],
    answers={
        "rollout.artifacts": Card.AT_LEAST_ONE(
            because="a gdpval repeat directory exists because the agent produced a file into it",
        )
    },
)

GDPVAL_INPUTS = Input(
    name="gdpval_inputs",
    via=gdpval_inputs,
    reads=[At(DELIVERABLES, "reference_files/**")],
    answers={
        "rollout.artifacts": Card.ANY(
            because="`reference_files/` holds the input documents the task supplied; a task with none is legal",
        )
    },
)
