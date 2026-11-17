# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The exporter contract: what an exporter declares, and what it is handed."""

from __future__ import annotations

import io
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import polars as pl

from evalrdl_schema import CURRENT, Refusal, parquet_row_group_rows, widest_value
from evalrdl_schema.tables import READABLE, RunTables

RUN = "run"
ROLLOUT = "rollout"
SCOPES = (RUN, ROLLOUT)

#: `(task_id, repeat)` of the one rollout a rollout-scoped export is for; None for a whole run.
Only = tuple[str, int] | None


#: Polars' default rows per row group; fewer where values are long.
ROW_GROUP_ROWS = 512 * 512


def write_parquet(frame: pl.DataFrame) -> bytes:
    """`frame` as Parquet, its row groups bounded by bytes so no page passes the format's limit."""
    buf = io.BytesIO()
    frame.write_parquet(
        buf, row_group_size=parquet_row_group_rows(widest_value(frame), ROW_GROUP_ROWS)
    )
    return buf.getvalue()


@dataclass(frozen=True)
class Exporter:
    """One export format. Checked against the CURRENT schema when it is declared.

    `build(tables, run, only)` turns one run into a frame: `tables` scans exactly `reads`,
    `run` is the run's catalog row, `only` narrows it to one rollout. The frames of every run
    asked for are concatenated and handed to `write`.

    `produce(tables, run, only, out)` instead writes one run's export to the file `out` itself
    and returns its row count: a format of several tables, streamed batch by batch. An exporter
    declares `build` or `produce`, not both.
    """

    name: str
    title: str
    media_type: str
    suffix: str
    scopes: tuple[str, ...]
    reads: Mapping[str, tuple[str, ...]]
    build: Callable[[RunTables, Mapping, Only], pl.DataFrame] | None = None
    write: Callable[[pl.DataFrame], bytes] = write_parquet
    means: str = ""
    produce: Callable[[RunTables, Mapping, Only, Path], int] | None = None

    def __post_init__(self) -> None:
        where = f"Exporter({self.name!r})"
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.name):
            raise Refusal(f"{where}: the name is a URL segment: lowercase, digits, - and _")
        if not self.scopes or set(self.scopes) - set(SCOPES):
            raise Refusal(f"{where}: scopes {self.scopes!r} must be drawn from {SCOPES}")
        if not self.suffix.startswith("."):
            raise Refusal(f"{where}: suffix {self.suffix!r} must start with '.'")
        if (self.build is None) == (self.produce is None):
            raise Refusal(f"{where}: declares one of build and produce")
        if not self.reads:
            raise Refusal(f"{where}: reads nothing")
        for table, cols in self.reads.items():
            if table not in READABLE:
                raise Refusal(
                    f"{where}: reads {table!r}; an exporter reads the rollout-keyed tables "
                    f"{sorted(READABLE)}, and gets the run's own row as `run`"
                )
            declared = CURRENT.table(table).column_names
            unknown = [c for c in cols if c not in declared]
            if unknown:
                raise Refusal(
                    f"{where}: {table}.{unknown} is not declared in {CURRENT.version}; "
                    f"an exporter reads the declared schema, never a file's own columns"
                )
            if len(set(cols)) != len(cols):
                raise Refusal(f"{where}: {table} names a column twice")
        reads = MappingProxyType({t: tuple(c) for t, c in self.reads.items()})
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "scopes", tuple(self.scopes))

    def to_file(self, tables: RunTables, run: Mapping, only: Only, out: Path) -> int:
        """One run's export written to `out`; its row count."""
        if self.produce is not None:
            return self.produce(tables, run, only, Path(out))
        frame = self.build(tables, run, only)
        Path(out).write_bytes(self.write(frame))
        return frame.height

    def describe(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "media_type": self.media_type,
            "suffix": self.suffix,
            "scopes": list(self.scopes),
            "reads": {t: list(c) for t, c in self.reads.items()},
            "means": self.means,
        }


class ExportTooLarge(Exception):
    """A whole-run export needed more memory than its process was allowed."""
