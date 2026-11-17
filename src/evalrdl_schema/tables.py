# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`RunTables`: one run partition, as the declared schema says it reads."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import polars as pl

from . import CURRENT, Schema, polars_schemas

#: What addresses one rollout inside a run partition.
ROLLOUT_KEY = ("task_id", "repeat", "agent_ref")

#: The tables a run partition holds per rollout. `runs` is the catalog's row, passed as `run`.
READABLE = frozenset(t.name for t in CURRENT.tables if set(ROLLOUT_KEY) <= set(t.column_names))


class NoSuchRollouts(LookupError):
    """A selection named rollouts or tasks the run does not have."""


class RunTables:
    """Lazy scans of exactly the declared tables and columns of one partition directory.

    Every column is cast to its declared dtype; one the partition was written without (an
    older schema version) reads as its declared backfill, and a present NULL in a column with a
    backfill reads as the backfill, as `canonical_view_select` does. A table the partition does
    not have reads as empty.
    """

    def __init__(
        self,
        directory: Path,
        reads: Mapping[str, tuple[str, ...]],
        *,
        schema: Schema = CURRENT,
        keys: pl.DataFrame | None = None,
    ):
        self.directory = Path(directory)
        self.reads = reads
        self.schema = schema
        self._dtypes = polars_schemas(schema)
        self._keys = keys

    def scan(self, table: str) -> pl.LazyFrame:
        if table not in self.reads:
            raise KeyError(f"{table!r} is not in this exporter's reads: {sorted(self.reads)}")
        return self._scan(table, self.reads[table])

    def _scan(self, table: str, cols: tuple[str, ...]) -> pl.LazyFrame:
        need = cols if self._keys is None else tuple(dict.fromkeys((*cols, *ROLLOUT_KEY)))
        dtypes = self._dtypes[table]
        f = self.directory / f"{table}.parquet"
        if not f.is_file():
            lf = pl.LazyFrame(schema={c: dtypes[c] for c in need})
        else:
            lf = pl.scan_parquet(f)
            present = set(lf.collect_schema().names())
            lf = lf.select([self._column(table, c, c in present) for c in need])
        if self._keys is not None:
            # The range lets parquet statistics skip row groups: partitions are sorted by key.
            ids = self._keys["task_id"]
            lf = (
                lf.filter(pl.col("task_id").is_between(pl.lit(ids.min()), pl.lit(ids.max())))
                .filter(pl.col("task_id").is_in(ids.implode()))
                .join(self._keys.lazy(), on=list(ROLLOUT_KEY), how="semi")
            )
            if need != cols:
                lf = lf.select(cols)
        return lf

    def _column(self, table: str, name: str, present: bool) -> pl.Expr:
        dtype = self._dtypes[table][name]
        backfill = self.schema.table(table).column(name).backfill
        if not present:
            return pl.lit(backfill, dtype=dtype).alias(name)
        expr = pl.col(name).cast(dtype)
        return expr if backfill is None else expr.fill_null(pl.lit(backfill, dtype=dtype))

    def scoped(self, rollouts: pl.DataFrame) -> RunTables:
        """The same tables, holding only the rows of `rollouts` (any frame with the key)."""
        missing = [c for c in ROLLOUT_KEY if c not in rollouts.columns]
        if missing:
            raise KeyError(
                f"scoping needs the rollout key {ROLLOUT_KEY}; the frame lacks {missing}"
            )
        keys = rollouts.select(ROLLOUT_KEY)
        return RunTables(self.directory, self.reads, schema=self.schema, keys=keys)

    def select(
        self, rollouts: Iterable[tuple[str, int]] = (), task_ids: Iterable[str] = ()
    ) -> RunTables:
        """The same tables, holding only the `(task_id, repeat)` rollouts and every repeat of
        `task_ids`. NoSuchRollouts names those the partition does not have."""
        pairs = list(dict.fromkeys((str(t), int(r)) for t, r in rollouts))
        tasks = list(dict.fromkeys(str(t) for t in task_ids))
        have = self._scan("rollouts", ROLLOUT_KEY).collect()
        dtypes = self._dtypes["rollouts"]
        wanted = pl.DataFrame(
            pairs, schema={"task_id": dtypes["task_id"], "repeat": dtypes["repeat"]}, orient="row"
        )
        missing = [
            f"{t}:{r}" for t, r in wanted.join(have, on=["task_id", "repeat"], how="anti").rows()
        ]
        known = set(have["task_id"].to_list())
        missing += [f"task_id={t}" for t in tasks if t not in known]
        if missing:
            raise NoSuchRollouts(missing)
        chosen = have.filter(pl.col("task_id").is_in(tasks)).vstack(
            have.join(wanted, on=["task_id", "repeat"], how="semi")
        )
        return self.scoped(chosen.unique(ROLLOUT_KEY).sort(ROLLOUT_KEY))

    def batches(
        self, rollouts: pl.DataFrame, size: int
    ) -> Iterator[tuple[pl.DataFrame, RunTables]]:
        """`rollouts` in slices of `size`, each with the tables scoped to it."""
        for i in range(0, rollouts.height, size):
            batch = rollouts.slice(i, size)
            yield batch, self.scoped(batch)

    def batches_by(
        self, rollouts: pl.DataFrame, weights: pl.Series, budget: int
    ) -> Iterator[tuple[pl.DataFrame, RunTables]]:
        """`rollouts` in consecutive slices whose `weights` sum to at most `budget` (at least one
        rollout each), each with the tables scoped to it."""
        start, total = 0, 0
        for i, w in enumerate(weights.to_list()):
            if i > start and total + w > budget:
                yield (
                    rollouts.slice(start, i - start),
                    self.scoped(rollouts.slice(start, i - start)),
                )
                start, total = i, 0
            total += w
        if start < rollouts.height:
            batch = rollouts.slice(start)
            yield batch, self.scoped(batch)
