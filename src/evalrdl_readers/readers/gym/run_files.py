# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym's own run-grain files: the plan, the aggregates, the dataset, the shard rollup, the
certification report and the quality pass's verdicts."""

from __future__ import annotations

from evalrdl_readers.core import ROOT, Answer, At, Card, Input, Sources, Unreadable

from ..common import (
    AGGREGATE_METRICS,
    FAILURES,
    MATERIALIZED_INPUTS,
    MULTISTAGE_STATE,
    as_float,
    dig,
    dumps,
    event_kind,
    records,
    text_parts,
)
from .landmarks import RUN, SHARDS, STREAMS
from .record import _position, _sidecar_key, _stream_keys

# ------------------------------------------------------------------- translators


def gym_plan(src: Sources):
    """The materialized inputs: one line per planned rollout.

    Not used for `run.n_expected`: these lines are what the run prepared, not what it was
    commissioned to do. A scalar `num_rollouts` in results.yml would fix that.

    A line takes the key of the rollout at its position: the plan and the stream may name a
    task under different spellings, or one of them under none. The stream holds a finished
    rollout and the failures sidecar one it never finished; a position neither holds keeps the
    position, as the verdicts name it."""
    held: dict[str, tuple] | None = None
    for doc in src.docs(0):
        for item in records(doc, "rollout.prompt"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            if held is None:
                held = _held_keys(src.docs(1))
            yield Answer(
                "rollout.prompt",
                rec.get("problem") or rec.get("question"),
                key=held.get(pos := _position(rec, line)) or (pos,),
            )


def _held_keys(docs) -> dict[str, tuple]:
    """Each rollout key the stream or the failures sidecar holds, by its position."""
    stream = _stream_keys(docs)
    out = dict(stream)
    for doc in docs:
        if not doc.name.endswith(f"{FAILURES}.jsonl"):
            continue
        for item in records(doc, "rollout.prompt"):
            if isinstance(item, Unreadable):
                continue
            line, rec = item
            out.setdefault(_position(rec, line), _sidecar_key(rec, line, stream))
    return out


def gym_aggregate(src: Sources):
    if not src.has(0):
        return
    body = src.doc(0).json()
    groups = body if isinstance(body, list) else []
    means = [
        dig(g, "key_metrics", "mean/reward")
        for g in groups
        if isinstance(g, dict) and dig(g, "key_metrics", "mean/reward") is not None
    ]
    yield Answer("run.score", means[0] if len(means) == 1 else None)
    yield Answer("run.n_agents", len(groups))


def gym_multistage(src: Sources):
    stages = set()
    for doc in src.docs(0):
        for item in records(doc, "run.n_stages"):
            if isinstance(item, Unreadable):
                yield item
                continue
            _line, rec = item
            stages.add(rec.get("stage_index"))
    yield Answer("run.n_stages", len(stages) or None)


def gym_shard_run(src: Sources):
    """The sharded runner's own manifest: `tasks.rollout_count` is tasks x repeats."""
    body = src.doc(0).json() or {}
    n = dig(body, "tasks", "rollout_count")
    if isinstance(n, int) and not isinstance(n, bool):
        yield Answer("run.n_expected", n)


def gym_verification(src: Sources):
    """The certification pass's own completion report, written beside the rundir."""
    if not src.has(0):
        return
    body = src.doc(0).json() or {}
    if body.get("expected_rollouts") is not None:
        yield Answer("run.n_expected", body.get("expected_rollouts"))


def gym_planned(src: Sources):
    """How many rollouts the run prepared: one materialized input line each."""
    n = sum(doc.count_lines() for doc in src.docs(0))
    if n:
        yield Answer("run.n_planned", n)


def gym_dataset_name(src: Sources):
    """The prepared dataset's own name, from the metrics summary written beside it."""
    names = [(doc.json() or {}).get("name") for i in (0, 1) for doc in src.docs(i)]
    names = sorted({n for n in names if isinstance(n, str) and n.strip()})
    if names:
        yield Answer(
            "run.benchmark",
            names[0] if len(names) == 1 else None,
            note=None if len(names) == 1 else f"summaries name {names}",
        )


def gym_dataset(src: Sources):
    # Counted, not parsed: the prepared dataset can be gigabytes and only its size is asked.
    n = sum(doc.count_lines() for i in (0, 1) for doc in src.docs(i))
    if n:
        yield Answer("run.n_dataset_rows", n)


def gym_shard_metrics(src: Sources):
    body = src.doc(0).json() or {}
    yield Answer("run.score", as_float(body.get("mean_osworld_score")))


def gym_verdicts(src: Sources):
    """The harness's own health report, one line per rollout it checked."""
    stream = _stream_keys(src.docs(1))
    for doc in src.docs(0):
        for item in records(doc, "rollout.health"):
            if isinstance(item, Unreadable):
                yield item
                continue
            line, rec = item
            key = _sidecar_key(rec, line, stream)
            yield Answer("rollout.repeat", rec.get("_ng_rollout_index"), key=key)
            yield Answer(
                "rollout.health",
                {
                    "verdict": rec.get("verdict"),
                    "checks_failed": [
                        f.get("check") for f in rec.get("findings") or () if isinstance(f, dict)
                    ],
                    "checks_unobserved": list(rec.get("unobserved") or ()),
                },
                key=key,
            )
            yield Answer(
                "rollout.events",
                {
                    "source_event_id": f"verdict:{line}",
                    "read_seq": f"{0:09d}",
                    **event_kind("verdict"),
                    "parts": (
                        text_parts(f"verdict: {rec.get('verdict')}")
                        + text_parts([dumps(f, 1 << 16) for f in rec.get("findings") or ()])
                        + text_parts([f"unobserved: {u}" for u in rec.get("unobserved") or ()])
                    ),
                },
                key=key,
                input_key=f"verdict:{line}",
                lane=("verdict", ""),
            )


# ------------------------------------------------------------------------- inputs

GYM_VERDICTS = Input(
    name="gym_verdicts",
    via=gym_verdicts,
    # The stream too: a verdict names its rollout by position, the stream by its key.
    reads=[At(RUN, "rollout_verdicts.jsonl"), At(RUN, STREAMS)],
    answers={
        "rollout.health": Card.ANY(
            because=("written by the quality pass, which not every run runs"),
        ),
        "rollout.repeat": Card.ANY(
            because=(
                "`_ng_rollout_index` is a reserved row key on every verdict record, but most "
                "runs write no verdict file"
            ),
        ),
        "rollout.events": Card.ANY(
            because=(
                "one verdict event per line, where the quality pass ran; a rollout the stream "
                "wrote twice is checked twice (`rollout_duplicate_identity`)"
            ),
        ),
    },
)

GYM_PLAN = Input(
    name="gym_plan",
    via=gym_plan,
    # The stream too: a plan line takes the key the stream gives its position.
    reads=[At(RUN, f"*{MATERIALIZED_INPUTS}.jsonl"), At(RUN, STREAMS)],
    answers={
        "rollout.prompt": Card.ANY(
            because=(
                "not written on a `resume_from_cache` hit, and never under the gdpval "
                "multistage driver"
            ),
        )
    },
)

GYM_AGGREGATE = Input(
    name="gym_aggregate",
    via=gym_aggregate,
    reads=[At(RUN, f"*{AGGREGATE_METRICS}.json")],
    answers={
        "run.score": Card.ANY(because="not written when `disable_aggregation` is set"),
        "run.n_agents": Card.ANY(because="same file, same absence"),
    },
)

GYM_MULTISTAGE = Input(
    name="gym_multistage",
    via=gym_multistage,
    reads=[At(RUN, f"*{MULTISTAGE_STATE}.jsonl")],
    answers={
        "run.n_stages": Card.ANY(because="only the gdpval multistage orchestrator writes it"),
    },
)

GYM_SHARD_RUN = Input(
    name="gym_shard_run",
    via=gym_shard_run,
    reads=[At(SHARDS, "run.json")],
    answers={
        "run.n_expected": Card.ANY(
            because=(
                "the sharded runner's manifest states `tasks.rollout_count`, the tasks times "
                "the repeats it was given; an interrupted runner may not have written it"
            ),
        ),
    },
)

GYM_VERIFICATION = Input(
    name="gym_verification",
    via=gym_verification,
    reads=[At(ROOT, "verification/completion.json")],
    answers={
        "run.n_expected": Card.ANY(
            because=(
                "written by the certification pass, which most runs do not run. Where it exists "
                "it is the only scalar in a nemo-gym tree stating the expected count: "
                "`results.yml: num_rollouts` and `eval_metadata.json: num_task_repeats` are null"
            ),
        ),
    },
)

GYM_DATASET = Input(
    name="gym_dataset",
    via=gym_dataset,
    reads=[
        At(RUN, "*/preprocessed_datasets/*.jsonl"),
        At(RUN, "preprocessed_datasets/*.jsonl"),
    ],
    answers={
        "run.n_dataset_rows": Card.ANY(
            because=(
                "skipped when `reuse_existing_data_preparation` is on, the wrapper's default. "
                "Two read patterns: the directory sits under the stem in some layouts and "
                "directly under `artifacts/` in others"
            ),
        )
    },
)

GYM_PLANNED = Input(
    name="gym_planned",
    via=gym_planned,
    reads=[At(RUN, f"*{MATERIALIZED_INPUTS}.jsonl")],
    answers={
        "run.n_planned": Card.ANY(
            because=(
                "what the run prepared, which is not what it was commissioned to do "
                "(run.n_expected); absent on a cache-resumed run and under the multistage driver"
            ),
        ),
    },
)

GYM_DATASET_NAME = Input(
    name="gym_dataset_name",
    via=gym_dataset_name,
    reads=[
        At(RUN, "*/preprocessed_datasets/*.json"),
        At(RUN, "preprocessed_datasets/*.json"),
    ],
    answers={
        "run.benchmark": Card.ANY(
            because=(
                "`benchmark_metrics.json: name` is the prepared dataset's name, the only "
                "benchmark signal in a gym-only folder; a guess that drops the agent variant, "
                "so it ranks last"
            ),
        ),
    },
)

GYM_SHARD_METRICS = Input(
    name="gym_shard_metrics",
    via=gym_shard_metrics,
    reads=[At(SHARDS, "metrics.json")],
    answers={
        "run.score": Card.ONE(
            because=(
                "the sharded runner writes its own rollup, keyed by the benchmark's name, which "
                "need not match the name the catalog files the run under"
            ),
        )
    },
)
