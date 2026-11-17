# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound rich-trace hydration while sharing table reads between nearby rollouts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from evalrdl_schema.tables import RunTables

# These bound the captured text before normalization; one larger rollout is still
# read whole. Python objects and reconstructed requests add overhead to this budget.
TEXT_BUDGET = 8 << 20
ROLLOUT_LIMIT = 64


def _text_weights(source: RunTables) -> Counter[tuple[str, int]]:
    weights = Counter()
    estimates = {
        "part_texts": pl.col("text").str.len_bytes().fill_null(0),
        "tool_call_texts": pl.col("args_full").str.len_bytes().fill_null(0),
        "tool_def_texts": pl.col("def_json").str.len_bytes().fill_null(0),
        "artifact_texts": pl.col("text").str.len_bytes().fill_null(0),
        "prompt_messages": pl.col("msg_json").str.len_bytes().fill_null(0),
        "event_texts": pl.sum_horizontal(
            pl.col("request_messages", "request_tools", "request_params")
            .str.len_bytes().fill_null(0).cast(pl.Int64)
        ),
    }
    for table, estimate in estimates.items():
        rows = (source.scan(table).group_by("task_id", "repeat")
                .agg(estimate.cast(pl.Int64).sum().alias("weight"))
                .collect(engine="streaming"))
        for task, repeat, weight in rows.iter_rows():
            weights[task, repeat] += max(0, weight)
    return weights


class _BatchTables:
    """Use schema-projected tables once per batch, with exact nullable scope rules."""

    def __init__(self, source: RunTables, rows: list[dict[str, Any]]):
        self.directory, self.schema, self.reads = source.directory, source.schema, source.reads
        self.source = source
        self.frames = {}
        self.tasks = list({row["problem_id"] for row in rows})
        self.repeats = list({row["run_idx"] for row in rows})
        self.pairs = pl.DataFrame(
            [(row["problem_id"], row["run_idx"]) for row in rows],
            schema={"task_id": pl.String, "repeat": pl.Int64}, orient="row",
        ).to_struct()

    def _pairs(self, task: str = "task_id", repeat: str = "repeat") -> pl.Expr:
        return pl.struct(pl.col(task).alias("task_id"), pl.col(repeat).alias("repeat")).is_in(
            self.pairs.implode()
        )

    def scan(self, table: str) -> pl.LazyFrame:
        if table not in self.frames:
            query = self.source.scan(table)
            columns = self.reads[table]
            if table == "relations":
                def endpoint(side):
                    task, repeat = f"{side}_task_id", f"{side}_repeat"
                    return (self._pairs(task, repeat)
                            | (pl.col(task).is_in(self.tasks) & pl.col(repeat).is_null()))
                query = query.filter(endpoint("frm") | endpoint("to"))
            elif "task_id" in columns and "repeat" in columns:
                scope = self._pairs()
                if table == "provenance":
                    scope = (scope | (pl.col("task_id").is_null()
                                      & (pl.col("repeat").is_null()
                                         | pl.col("repeat").is_in(self.repeats)))
                             | (pl.col("repeat").is_null()
                                & pl.col("task_id").is_in(self.tasks)))
                query = query.filter(scope)
            self.frames[table] = query.collect(engine="streaming")
        return self.frames[table].lazy()


def rollout_batches(
    source: RunTables, parquet: pq.ParquetFile, *,
    text_budget: int = TEXT_BUDGET, rollout_limit: int = ROLLOUT_LIMIT,
) -> Iterator[tuple[list[dict[str, Any]], _BatchTables]]:
    """Yield compact rows and their bounded shared source without dropping global provenance."""
    weights = _text_weights(source)
    pending, size = [], 0
    for record in parquet.iter_batches(batch_size=1):
        row = record.to_pylist()[0]
        weight = weights[row["problem_id"], row["run_idx"]]
        if pending and (len(pending) >= rollout_limit or size + weight > text_budget):
            yield pending, _BatchTables(source, pending)
            pending, size = [], 0
        pending.append(row)
        size += weight
    if pending:
        yield pending, _BatchTables(source, pending)
