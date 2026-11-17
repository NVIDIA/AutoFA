# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoFA (auto-failure-analysis): one row per rollout in gym_adapter's canonical columns.

Each rollout is read from one lane, its conversation: the lane with the most user and
assistant messages, then the most events, the unnamed lane first on a tie. Texts are whole. Written batch by batch, one
parquet row group each, so memory is one batch and the file grows as it is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl
from polars.io.plugins import register_io_source

from evalrdl_schema.tables import RunTables

from .api import ROLLOUT, RUN, Exporter, Only

COLUMNS = (
    "model_id",
    "benchmark",
    "problem_id",
    "run_idx",
    "is_correct",
    "reward",
    "system_text",
    "prompt_text",
    "response_text",
    "reasoning_text",
    "lane",
)
SCHEMA: dict[str, pl.DataType] = {
    "model_id": pl.Utf8,
    "benchmark": pl.Utf8,
    "problem_id": pl.Utf8,
    "run_idx": pl.Int64,
    "is_correct": pl.Boolean,
    "reward": pl.Float64,
    "system_text": pl.Utf8,
    "prompt_text": pl.Utf8,
    "response_text": pl.Utf8,
    "reasoning_text": pl.Utf8,
    "lane": pl.Utf8,
}

#: Characters of text per batch, and rows per parquet row group: together they bound memory.
#: On a 117,590-rollout MMLU-ProX run: 3.2 s and 1.0 GB peak (batches of 50 rollouts, one file
#: written at the end: 112 s and 7.0 GB, over the serve's limit).
BATCH_CHARS = 32 << 20
ROW_GROUP = 1024

_KEY = ["task_id", "repeat", "agent_ref"]
_LKEY = [*_KEY, "lane"]
_SPEAKERS = ["user", "assistant", "agent"]


def produce(tables: RunTables, run: Mapping, only: Only, out: Path) -> int:
    """prompt: first user message; response: last assistant/agent message plus a `tools:` line;
    system: the system and developer messages; reasoning: every reasoning part, whole; all from
    the rollout's conversation lane. is_correct: reward >= 0.5 and status != 'failed'."""
    rollouts = tables.scan("rollouts").collect()
    if only is not None:
        rollouts = rollouts.filter(
            (pl.col("task_id") == only[0]) & (pl.col("repeat") == int(only[1]))
        )
    rollouts = rollouts.sort(_KEY)
    model_id, benchmark = run.get("model"), run.get("benchmark")
    n = 0

    def batches(with_columns, predicate, n_rows, batch_size):
        nonlocal n
        for batch, t in tables.batches_by(rollouts, _chars(tables, rollouts), BATCH_CHARS):
            frame = _batch(t, batch, model_id, benchmark)
            n += frame.height
            yield frame

    register_io_source(io_source=batches, schema=SCHEMA).sink_parquet(
        out, engine="streaming", row_group_size=ROW_GROUP
    )
    return n


def _chars(tables: RunTables, rollouts: pl.DataFrame) -> pl.Series:
    """Text characters per rollout of `rollouts`, in its order: what sizes a batch."""
    per = (
        tables.scan("parts")
        .group_by(_KEY)
        .agg(pl.col("chars").fill_null(0).sum().alias("_chars"))
        .collect(engine="streaming")
    )
    return rollouts.join(per, on=_KEY, how="left", maintain_order="left")["_chars"].fill_null(0)


def _batch(
    tables: RunTables, rollouts: pl.DataFrame, model_id: str | None, benchmark: str | None
) -> pl.DataFrame:
    def scan(table: str) -> pl.DataFrame:
        return tables.scan(table).collect(engine="streaming")

    events = scan("events")
    speaks = (pl.col("kind") == "message") & pl.col("role").is_in(_SPEAKERS)
    lane = (
        events.group_by(_LKEY)
        .agg(speaks.sum().alias("_msgs"), pl.len().alias("_n"))
        .sort([*_KEY, "_msgs", "_n", "lane"], descending=[False, False, False, True, True, False])
        .unique(_KEY, keep="first", maintain_order=True)
        .select(_LKEY)
    )
    events = events.join(lane, on=_LKEY, how="semi")
    parts = scan("parts").join(lane, on=_LKEY, how="semi")
    pc = (
        parts.join(scan("part_texts"), on=[*_LKEY, "event_idx", "part_idx"], how="left")
        .join(events, on=[*_LKEY, "event_idx"], how="inner")
        .with_columns(pl.col("text").fill_null(""))
        .sort([*_KEY, "event_idx", "part_idx"])
    )
    content = pc.filter(pl.col("channel") == "content")

    def message(roles: list[str], pick: str, name: str, sep: str) -> pl.DataFrame:
        at = (
            events.filter((pl.col("kind") == "message") & pl.col("role").is_in(roles))
            .group_by(_KEY)
            .agg(getattr(pl.col("event_idx"), pick)().alias("_idx"))
        )
        return (
            content.join(at, on=_KEY, how="inner")
            .filter(pl.col("event_idx") == pl.col("_idx"))
            .group_by(_KEY, maintain_order=True)
            .agg(pl.col("text").str.join(sep).alias(name))
        )

    system = (
        content.filter(pl.col("kind") == "message")
        .filter(pl.col("role").is_in(["system", "developer"]))
        .group_by(_KEY, maintain_order=True)
        .agg(pl.col("text").str.join("\n\n").alias("system_text"))
    )
    prompt = message(["user"], "min", "prompt_text", "\n")
    resp = message(["assistant", "agent"], "max", "_resp_body", "\n")
    reasoning = (
        pc.filter((pl.col("channel") == "reasoning") & (pl.col("text") != ""))
        .group_by(_KEY, maintain_order=True)
        .agg(pl.col("text").str.join("\n\n").alias("reasoning_text"))
    )
    tools = (
        scan("tool_calls")
        .join(lane, on=_LKEY, how="semi")
        .filter(pl.col("tool_name").is_not_null())
        .sort([*_KEY, "event_idx", "call_idx"])
        .group_by(_KEY, maintain_order=True)
        .agg(pl.col("tool_name").str.join(", ").alias("_tools"))
    )

    out = rollouts.join(lane, on=_KEY, how="left")
    for part in (system, prompt, resp, tools, reasoning):
        out = out.join(part, on=_KEY, how="left")
    out = out.with_columns(
        pl.col("system_text", "prompt_text", "_resp_body", "reasoning_text").fill_null(""),
        pl.col("lane").fill_null(""),
    ).with_columns(
        pl.when(pl.col("_tools").is_not_null())
        .then(pl.col("_resp_body") + pl.lit("\ntools: ") + pl.col("_tools"))
        .otherwise(pl.col("_resp_body"))
        .alias("response_text"),
        pl.lit(model_id, dtype=pl.Utf8).alias("model_id"),
        pl.lit(benchmark, dtype=pl.Utf8).alias("benchmark"),
        pl.col("task_id").cast(pl.Utf8).alias("problem_id"),
        pl.col("repeat").cast(pl.Int64).alias("run_idx"),
        (
            pl.col("reward").is_not_null()
            & (pl.col("reward") >= 0.5)
            & (pl.col("status") != "failed")
        ).alias("is_correct"),
    )
    return out.sort(_KEY).select(list(COLUMNS)).cast(SCHEMA, strict=False)


AUTOFA = Exporter(
    name="autofa",
    title="AutoFA",
    media_type="application/vnd.apache.parquet",
    suffix=".parquet",
    scopes=(RUN, ROLLOUT),
    reads={
        "rollouts": (*_KEY, "reward", "status"),
        "events": (*_LKEY, "event_idx", "kind", "role"),
        "parts": (*_LKEY, "event_idx", "part_idx", "channel", "chars"),
        "part_texts": (*_LKEY, "event_idx", "part_idx", "text"),
        "tool_calls": (*_LKEY, "event_idx", "call_idx", "tool_name"),
    },
    produce=produce,
    means=(
        "AutoFA (auto-failure-analysis): one parquet, one row per rollout, canonical columns "
        "model_id, benchmark, problem_id, run_idx, is_correct, reward, system_text, prompt_text, "
        "response_text, reasoning_text, lane. Each rollout is read from its conversation lane "
        "(named in `lane`): the one with the most user and assistant messages. is_correct is "
        "reward>=0.5 and status!='failed'. system_text = the system and developer messages; "
        "prompt_text = the first user message; response_text = the last assistant message plus "
        "a `\\ntools: ...` line; reasoning_text = every reasoning part, joined `\\n\\n`, uncut."
    ),
)
