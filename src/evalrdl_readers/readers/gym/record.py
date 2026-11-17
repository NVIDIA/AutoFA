# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rollout record: `responses_create_params` and `response`, in the stream, its shards and
its failures sidecar."""

from __future__ import annotations

import json
from dataclasses import replace

from evalrdl_readers.core import Answer, At, Card, Input, Key, Sources, Unreadable
from evalrdl_schema.text import text_key

from ..common import (
    FAILURES,
    MATERIALIZED_INPUTS,
    as_float,
    dig,
    event_kind,
    failure_message,
    failure_reward,
    field_artifact,
    has_value,
    primary_stream,
    records,
    without_reference_fields,
)
from ..dialects.responses import responses_calls, responses_parts
from ..prompts import item_keys as item_key_of
from ..prompts import item_text, producer_id
from .calls import captured_calls, endpoint
from .gdpval import deliverables_dir
from .landmarks import RUN, SHARDS, STREAMS

# ------------------------------------------------------------------- translators


def _producer_task(rec: dict) -> str | None:
    """The producer's own task id, under whichever key the harness wrote it."""

    # native episode protocol: TaskId.task_id from the environment server
    tid = dig(rec, "_ng_task_id", "task_id")
    if isinstance(tid, str) and tid.strip():
        return tid

    # legacy path with verifier_metadata.task_id
    tid = dig(rec, "verifier_metadata", "task_id")
    if isinstance(tid, str) and tid.strip():
        return tid

    # legacy path with flat-row format; same candidate keys order as in
    # `nemo_gym.rollout_collection._trajectory_identity` @ 5287fdc
    for key in ("task_id", "problem_id", "instance_id"):
        tid = rec.get(key)
        if isinstance(tid, str) and tid.strip():
            return tid
    return None


def _position(rec: dict, line: int) -> str:
    """`t<task>-r<repeat>`, the record's position in the stream that wrote it."""
    t, r = rec.get("_ng_task_index"), rec.get("_ng_rollout_index")
    return f"line:{line}" if t is None and r is None else f"t{t}-r{r}"


def _gym_key(rec: dict, line: int, stream: str = "") -> tuple[str]:
    """`<task id>-r<repeat>` where the producer names the task, one key space for every
    stream; else the position, prefixed by the stream that wrote it."""
    task, r = _producer_task(rec), rec.get("_ng_rollout_index")
    if task is not None and r is not None:
        return (f"{task}-r{r}",)
    base = _position(rec, line)
    return (f"{stream}#{base}" if stream else base,)


def _stream_keys(docs) -> dict[str, tuple]:
    """Each rollout key the stream holds, by the position it holds it at."""
    doc = primary_stream(docs)
    if doc is None:
        return {}
    out: dict[str, tuple] = {}
    for item in records(doc, "rollout.score"):
        if isinstance(item, Unreadable):
            continue
        line, rec = item
        out.setdefault(_position(rec, line), _gym_key(rec, line))
    return out


def _sidecar_key(rec: dict, line: int, stream: dict[str, tuple]) -> tuple:
    """A sidecar record's own producer key, else the key the stream gives its position."""
    if _producer_task(rec) is not None:
        return _gym_key(rec, line)
    pos = _position(rec, line)
    return (stream.get(pos) if not pos.startswith("line:") else None) or _gym_key(rec, line)


def _rollout_facts(rec: dict, key: tuple, line: int):
    """Rollout-level facts, shared by the merged, shard and combined streams."""
    yield Answer("rollout.score", as_float(rec.get("reward")), key=key)
    yield Answer("rollout.repeat", rec.get("_ng_rollout_index"), key=key)
    yield Answer("rollout.stage", rec.get("stage_index"), key=key)
    # Per-record model; may differ from the run's.
    yield Answer("rollout.agent_model", dig(rec, "response", "model"), key=key)
    # Constant per run; not `rollouts.agent_ref` (a PK column held empty).
    yield Answer("rollout.agent_name", dig(rec, "agent_ref", "name"), key=key)
    # The main stream's own error; the failures sidecar may be empty when this is set.
    err = rec.get("error_kind")
    if err is not None:
        yield Answer("rollout.error", err, key=key)
    # The agent's own exit class: an outcome of a rollout that ran and was scored.
    if rec.get("agent_error_kind") is not None:
        yield Answer("rollout.stop_reason", rec["agent_error_kind"], key=key)


#: The endpoint Gym names the policy model under; a judge or simulator has its own.
POLICY_ENDPOINT = "policy_model"


def _token_facts(rec: dict, key: tuple):
    """The rollout's token totals from the policy endpoint's captured calls.

    Used when `response.usage` is missing, has no token counts, or has both at zero (we take
    zeros as missing data, not real counts).
    """
    usage = dig(rec, "response", "usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if (
            isinstance(input_tokens, int)
            and isinstance(output_tokens, int)
            and any((input_tokens, output_tokens))
        ):
            # usage has both token counts and they are not zero: the events already carry them.
            return
    calls = captured_calls(rec)
    if len({endpoint(c) for c in calls}) > 1:
        # if there are multiple endpoints we rely on typical Gym naming
        # to exclude calls to judge models, user simulators etc
        calls = [c for c in calls if endpoint(c) == POLICY_ENDPOINT]
    counted = [
        c
        for c in calls
        if isinstance(c.get("tokens_in"), int) and isinstance(c.get("tokens_out"), int)
    ]
    if not counted:
        return
    prompt = sum(c["tokens_in"] for c in counted)
    completion = sum(c["tokens_out"] for c in counted)
    yield Answer("rollout.n_prompt_tokens", prompt, key=key)
    yield Answer("rollout.n_completion_tokens", completion, key=key)
    yield Answer("rollout.n_tokens", prompt + completion, key=key)


def _stamp(calls: list[dict], timing: dict) -> list[dict]:
    """Put the sidecar's clock on the calls the stream minted."""
    for call in calls:
        known = timing.get(str(call.get("call_id_source")))
        if known:
            call.update({k: v for k, v in known.items() if v is not None})
    return calls


def _responses_events(
    rec: dict,
    key: tuple,
    request_idx: int = 0,
    timing: dict | None = None,
    item_keys: bool = False,
):
    """One event per item of `responses_create_params.input`, then per `response.output`.

    With `item_keys`, an output item carries its key `item_id`, its producer id or for a tool
    result its call id, by which the model calls that produced it and were sent it name it; and
    a message item, either side, its `text_key`, by which calls name a text no id states."""
    params = rec.get("responses_create_params") or {}
    response = rec.get("response") or {}
    usage = response.get("usage") or {}
    # The response's usage and finish state, on its first output only.
    owned = {
        "finish_reason": response.get("status"),
        "length_truncated": response.get("status") == "incomplete",
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "cached_tokens": dig(usage, "input_tokens_details", "cached_tokens"),
        "reasoning_tokens": dig(usage, "output_tokens_details", "reasoning_tokens"),
    }
    for rank, (side, items) in enumerate(
        (("input", params.get("input")), ("output", response.get("output")))
    ):
        if not isinstance(items, list):
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                item = {"content": item}
            yield Answer(
                "rollout.events",
                {
                    "source_event_id": producer_id(item["id"]) if item.get("id") else f"{side}:{i}",
                    # The lane order; zero-padded so it sorts as a string.
                    "read_seq": f"{request_idx:09d}:{rank}:{i:09d}",
                    "started_at": None,
                    # One responses record is one model invocation.
                    "request_idx": request_idx,
                    "role": item.get("role") or side,
                    **event_kind(item.get("type") or side),
                    # Sampling parameters belong on the model_call event, not on turns.
                    **({"model_name": response.get("model")} if side == "output" else {}),
                    **(owned if side == "output" and i == 0 else {}),
                    "parts": responses_parts(item),
                    "tool_calls": _stamp(responses_calls(item), timing or {}),
                    # Producer call id, never a position: `to_store._pair_calls` resolves it
                    # within the lane, since other lanes reuse the same ids.
                    **(
                        {"answers_call_ids": [item["call_id"]]}
                        if item.get("type") == "function_call_output" and item.get("call_id")
                        else {}
                    ),
                    "side": side,
                },
                key=key,
                keys=(
                    ({"item_id": own[0]} if side == "output" and (own := item_key_of(item)) else {})
                    | ({"text_key": tk} if (tk := text_key(item_text(item))) else {})
                    if item_keys
                    else {}
                ),
                # A rollout may hold several records; `request_idx` keeps names unique.
                input_key=f"{request_idx}:{side}:{i}",
            )


def gym_rollouts(src: Sources):
    doc = primary_stream(src.docs(0))
    if doc is None:
        return
    for item in records(doc, "rollout.score"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        key = _gym_key(rec, line)
        yield from _rollout_facts(rec, key, line)
        yield from _token_facts(rec, key)
        keys = {"rollout_position": _position(rec, line)}
        # The path is where the agent wrote; a wrapper may move and renumber it.
        task, repeat = rec.get("task_id"), rec.get("_ng_rollout_index")
        if rec.get("deliverables_dir") and task is not None and repeat is not None:
            keys["deliv_dir"] = deliverables_dir(task, repeat)
        yield Answer(
            "rollout.task",
            _producer_task(rec) or rec.get("_ng_task_index"),
            key=key,
            keys=keys,
        )


def gym_outcomes(src: Sources):
    """Retain an exact allowlist of candidate and verifier outcome fields."""
    doc = primary_stream(src.docs(0))
    if doc is None:
        return
    for item in records(doc, "rollout.artifacts"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        key = _gym_key(rec, line)
        if has_value(rec.get("test_output")):
            yield Answer(
                "rollout.artifacts",
                field_artifact(
                    str(rec["test_output"]),
                    kind="log",
                    role="verifier",
                    name="test-output.log",
                ),
                key=key,
                input_key=f"{line}:test-output",
            )
        evaluation = {
            name: (without_reference_fields(rec[name]) if name == "test_results" else rec[name])
            for name in ("evaluation_completed", "test_results")
            if name in rec and rec[name] is not None
        }
        if evaluation:
            yield Answer(
                "rollout.artifacts",
                field_artifact(
                    json.dumps(
                        evaluation,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    kind="data",
                    role="verifier",
                    name="evaluation-result.json",
                ),
                key=key,
                input_key=f"{line}:evaluation-result",
            )
        if has_value(rec.get("model_patch")):
            yield Answer(
                "rollout.artifacts",
                field_artifact(
                    str(rec["model_patch"]),
                    kind="patch",
                    role="output",
                    name="model.patch",
                ),
                key=key,
                input_key=f"{line}:model-patch",
            )


def gym_failures(src: Sources):
    """The failure sidecar: one record per try the harness could not finish.

    A try the stream later finished is an earlier attempt of that rollout: it stays an
    attempt row and its transcript an `attempt` lane, and the stream decides the rest.
    Components use different message keys, one per record: `common.FAILURE_MESSAGE_KEYS`."""
    # The stream is read only for a failure to place: most runs have none, and the stream is
    # the run's largest file.
    stream: dict[str, tuple] | None = None
    finished: set[tuple] = set()
    seq: dict[tuple, int] = {}
    for doc in src.docs(0):
        for item in records(doc, "rollout.error"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            if stream is None:
                stream = _stream_keys(src.docs(1))
                finished = set(stream.values())
            key = _sidecar_key(rec, line, stream)
            kind = rec.get("_ng_failure_class") or rec.get("error_kind") or rec.get("error")
            idx = seq.get(key, 0)
            seq[key] = idx + 1
            if key not in finished:
                if kind is None:
                    yield Unreadable(
                        "rollout.error",
                        where=f"line {line}",
                        detail="a failure record naming no failure class. It is "
                        "in the failures file, so something went wrong; "
                        "a constant here would say what",
                    )
                else:
                    yield Answer("rollout.error", kind, key=key)
                # Failure records also carry the score.
                yield Answer("rollout.score", failure_reward(rec), key=key)
                yield Answer(
                    "rollout.task",
                    _producer_task(rec) or rec.get("_ng_task_index"),
                    key=key,
                )
                yield Answer("rollout.repeat", rec.get("_ng_rollout_index"), key=key)
            # The same record is also an attempt row.
            yield Answer(
                "rollout.attempts",
                {
                    "attempt_idx": idx,
                    "failure_class": kind,
                    # The producer's flag where it writes one; else terminal iff no stream
                    # record finished this rollout, as production derives it.
                    "is_terminal": (
                        rec["_ng_failure_terminal"]
                        if "_ng_failure_terminal" in rec
                        else key not in finished
                    ),
                    "error_message": failure_message(rec),
                },
                key=key,
                input_key=f"{doc.name}:{line}",
            )
            # Failure records carry the transcript too: the rollout's own where it is the
            # only account, an attempt lane beside the finished one where it is not.
            lane = None if key not in finished else ("attempt", str(idx))
            for ev in _responses_events(rec, key, line):
                yield ev if lane is None else replace(ev, lane=lane)


def gym_shards(src: Sources):
    """Per-shard streams: the same records as the merged stream, deduplicated within a file.

    No transcript: these records are the merged stream's, and only its rows are rollouts."""
    seen = set()
    for doc in src.docs(0):
        stem = doc.name.rsplit(".", 1)[0]
        if any(stem.endswith(suf) for suf in (FAILURES, MATERIALIZED_INPUTS)):
            continue
        for item in records(doc, "rollout.score"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            key = _gym_key(rec, line, stream=stem)
            if (stem, key) in seen:
                continue
            seen.add((stem, key))
            yield from _rollout_facts(rec, key, line)


def gym_combined(src: Sources):
    seen = set()
    for item in records(src.doc(0), "rollout.score"):
        if isinstance(item, Unreadable):
            yield item
            continue
        line, rec = item
        key = _gym_key(rec, line, stream="combined")
        if key in seen:
            continue
        seen.add(key)
        yield from _rollout_facts(rec, key, line)


# ------------------------------------------------------------------------- inputs

GYM_ROLLOUTS = Input(
    name="gym_rollouts",
    via=gym_rollouts,
    reads=[At(RUN, STREAMS)],
    answers={
        "rollout.score": Card.ONE(
            because="a rollout record exists because the harness produced one, and reward is the reason it was written"
        ),
        "rollout.error": Card.ANY(
            because="the harness's verdict in the main stream rather than the failures sidecar; most "
            "runs write it nowhere, and a run whose sandbox never came up carries it here beside "
            "an empty sidecar"
        ),
        "rollout.task": Card.ONE(
            because="`_ng_task_index` is a reserved row key on every record; the producer's own "
            "task identity (`_ng_task_id.task_id`, `verifier_metadata.task_id`, `instance_id`, "
            "`task_id` or `problem_id`, depending on the row format) outranks that ordinal. Not "
            "`agent_ref.name`, which names the agent"
        ),
        "rollout.repeat": Card.ONE(
            because="`_ng_rollout_index` is a reserved row key gym writes on every record"
        ),
        "rollout.stage": Card.ANY(
            because="only the gdpval multistage driver writes `stage_index`; a single-stage run has no "
            "stage"
        ),
        "rollout.stop_reason": Card.ANY(
            because="`agent_error_kind` is the agent's own exit class (`other`, `stuck_in_loop`, `context_window`, `agent_launch_failed`) on a record the harness went on to score, so it says why the agent stopped, not that the rollout failed"
        ),
        "rollout.agent_name": Card.ANY(
            because="`agent_ref.name`, the agent the harness drove; only the nemo-gym shapes write it "
            "and the rest carry no `agent_ref` key"
        ),
        "rollout.agent_model": Card.ONE(
            because=(
                "`response.model` is on every responses-api record, and it can differ from every "
                "run-level source in the same tree"
            ),
        ),
        "rollout.n_prompt_tokens": Card.ANY(
            because="the policy endpoint's captured `tokens_in`, used only when `response.usage` has no token counts; with counts, the rollup sums the events' tokens"
        ),
        "rollout.n_completion_tokens": Card.ANY(
            because="the policy endpoint's captured `tokens_out`, used only when `response.usage` has no token counts"
        ),
        "rollout.n_tokens": Card.ANY(because="the sum of the two counts above"),
    },
    mints=(
        Key(
            "rollout_position",
            optional=True,  # a run that attempted nothing has no positions
            because="`t<N>-r<M>` from `_ng_task_index`/`_ng_rollout_index`: the position of a rollout in the run, whatever its record is keyed by",
        ),
    ),
)

GYM_OUTCOMES = Input(
    name="gym_outcomes",
    via=gym_outcomes,
    reads=[At(RUN, STREAMS)],
    answers={
        "rollout.artifacts": Card.ANY(
            because=(
                "only records with allowlisted evaluator outputs carry portable test logs, "
                "structured test results, or the candidate patch"
            ),
        ),
    },
)

GYM_FAILURES = Input(
    name="gym_failures",
    via=gym_failures,
    # The stream too: a failure record for a key the stream finished is an earlier try.
    reads=[At(RUN, f"*{FAILURES}.jsonl"), At(RUN, STREAMS)],
    answers={
        "rollout.error": Card.ANY(
            because=(
                "the sidecar is always created and is usually empty: rows land in it only with "
                "`route_failures_to_sidecar`"
            ),
        ),
        "rollout.score": Card.ANY(
            because=(
                "every failure record carries a reward under one of the two disjoint spellings "
                "in `common.FAILURE_REWARD_KEYS`, but the sidecar is often empty"
            ),
        ),
        "rollout.events": Card.ANY(
            because=(
                "a failure record carries the same responses-api transcript a success does, and "
                "an empty sidecar carries none"
            ),
        ),
        "rollout.attempts": Card.ANY(
            because=(
                "the same records, counted as what they are. `_ng_failure_class` is on every "
                "one; `error_message` and `_ng_failure_terminal` are not, so a demanded count on "
                "either would be a claim about the rest"
            ),
        ),
        "rollout.task": Card.ANY(
            because=(
                "answered only for a try the stream never finished, whose failure record is "
                "its only account, by the same spellings as gym_rollouts"
            ),
        ),
        "rollout.repeat": Card.ANY(
            because=("`_ng_rollout_index`, for the same failure-only tries"),
        ),
    },
)

# The shard files and combined.jsonl hold the merged stream's own records, so they describe
# its rollouts and never create one (`Rows`), and carry no transcript: a copy under the same
# key collides on every event identity, and in a lane of its own stores the transcript again.
GYM_SHARDS = Input(
    name="gym_shards",
    via=gym_shards,
    reads=[At(SHARDS, "shard_*.jsonl")],
    answers={
        "rollout.score": Card.ONE(because="a shard line is a rollout the shard scored"),
        "rollout.error": Card.ANY(
            because="the harness's verdict in the main stream rather than the failures sidecar; most "
            "runs write it nowhere, and a run whose sandbox never came up carries it here beside "
            "an empty sidecar"
        ),
        "rollout.repeat": Card.ONE(
            because="`_ng_rollout_index` is a reserved row key gym writes on every record"
        ),
        "rollout.stage": Card.ANY(
            because="only the gdpval multistage driver writes `stage_index`; a single-stage run has no "
            "stage"
        ),
        "rollout.stop_reason": Card.ANY(
            because="`agent_error_kind` is the agent's own exit class (`other`, `stuck_in_loop`, `context_window`, `agent_launch_failed`) on a record the harness went on to score, so it says why the agent stopped, not that the rollout failed"
        ),
        "rollout.agent_name": Card.ANY(
            because="`agent_ref.name`, the agent the harness drove; only the nemo-gym shapes write it and the rest carry no `agent_ref` key"
        ),
        "rollout.agent_model": Card.ONE(
            because="`response.model` is on every responses-api record, and it can differ from every "
            "run-level source in the same tree"
        ),
    },
)

GYM_COMBINED = Input(
    name="gym_combined",
    via=gym_combined,
    reads=[At(SHARDS, "combined.jsonl")],
    answers={
        "rollout.score": Card.ONE(
            because="combined.jsonl is the shards concatenated, one line per rollout"
        ),
        "rollout.error": Card.ANY(
            because="the harness's verdict in the main stream rather than the failures sidecar; most "
            "runs write it nowhere, and a run whose sandbox never came up carries it here beside "
            "an empty sidecar"
        ),
        "rollout.repeat": Card.ONE(
            because="`_ng_rollout_index` is a reserved row key gym writes on every record"
        ),
        "rollout.stage": Card.ANY(
            because="only the gdpval multistage driver writes `stage_index`; a single-stage run has no "
            "stage"
        ),
        "rollout.stop_reason": Card.ANY(
            because="`agent_error_kind` is the agent's own exit class (`other`, `stuck_in_loop`, `context_window`, `agent_launch_failed`) on a record the harness went on to score, so it says why the agent stopped, not that the rollout failed"
        ),
        "rollout.agent_name": Card.ANY(
            because="`agent_ref.name`, the agent the harness drove; only the nemo-gym shapes write it and the rest carry no `agent_ref` key"
        ),
        "rollout.agent_model": Card.ONE(
            because="`response.model` is on every responses-api record, and it can differ from every "
            "run-level source in the same tree"
        ),
    },
)
