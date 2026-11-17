# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""trace: every rollout-keyed table of a run, whole, as a ZIP of one parquet file per table."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from evalrdl_schema import CURRENT, parquet_row_group_rows, widest_value
from evalrdl_schema.tables import READABLE, ROLLOUT_KEY, RunTables

from .api import ROLLOUT, ROW_GROUP_ROWS, RUN, Exporter, Only

#: Bumped when the layout of the ZIP or of `manifest.json` changes.
VERSION = 1

#: Rollouts per batch: a run's texts can outgrow the serve's memory.
BATCH = 100

TABLES = tuple(sorted(READABLE))


class _AppendOnly:
    """A file zipfile cannot seek: it puts each member's CRC and sizes after its data rather than
    going back to its header, so bytes are final once written and can be read while it writes."""

    def __init__(self, f):
        self.write, self.flush = f.write, f.flush


def _member(zf: zipfile.ZipFile, name: str, src: Path) -> None:
    # A fixed timestamp: the same run exports to the same bytes.
    with src.open("rb") as f, zf.open(zipfile.ZipInfo(name), "w", force_zip64=True) as dst:
        shutil.copyfileobj(f, dst, 1 << 20)


def produce(tables: RunTables, run: Mapping, only: Only, out: Path) -> int:
    rollouts = tables.scan("rollouts").collect()
    if only is not None:
        rollouts = rollouts.filter(
            (pl.col("task_id") == only[0]) & (pl.col("repeat") == int(only[1]))
        )
    rollouts = rollouts.sort(ROLLOUT_KEY)
    batches = [scoped for _, scoped in tables.batches(rollouts, BATCH)] or [tables.scoped(rollouts)]
    counts: dict[str, int] = {}
    per_rollout: list[pl.DataFrame] = []
    with (
        tempfile.TemporaryDirectory(dir=out.parent) as tmp,
        out.open("wb") as raw,
        zipfile.ZipFile(_AppendOnly(raw), "w", allowZip64=True) as zf,
    ):
        for table in TABLES:
            key = list(tables.schema.table(table).key)
            parts, widest = [], 0
            for i, scoped in enumerate(batches):
                frame = scoped.scan(table).collect(engine="streaming").sort(key)
                per_rollout.append(
                    frame.group_by(ROLLOUT_KEY).len().with_columns(table=pl.lit(table))
                )
                parts.append(Path(tmp) / f"{table}.{i}.parquet")
                w = widest_value(frame)
                widest = max(widest, w)
                frame.write_parquet(
                    parts[-1], row_group_size=parquet_row_group_rows(w, ROW_GROUP_ROWS)
                )
            whole = Path(tmp) / f"{table}.parquet"
            pl.scan_parquet(parts).sink_parquet(
                whole, row_group_size=parquet_row_group_rows(widest, ROW_GROUP_ROWS)
            )
            counts[table] = pl.scan_parquet(whole).select(pl.len()).collect().item()
            _member(zf, f"{table}.parquet", whole)
            for f in (*parts, whole):
                f.unlink()
        rows = (
            pl.concat(per_rollout).pivot("table", index=ROLLOUT_KEY, values="len")
            if rollouts.height
            else None
        )
        manifest = Path(tmp) / "manifest.json"
        manifest.write_text(json.dumps(_manifest(run, tables, rollouts, counts, rows), indent=1))
        _member(zf, "manifest.json", manifest)
    return rollouts.height


def _manifest(run, tables, rollouts, counts, rows) -> dict:
    listed = rollouts.select(*ROLLOUT_KEY, "n_events", "n_tool_calls")
    if rows is not None:
        listed = listed.join(rows, on=list(ROLLOUT_KEY), how="left")
    return {
        "exporter": "trace",
        "exporter_version": VERSION,
        "run_id": run.get("run_id"),
        "schema": tables.schema.version,
        "n_rollouts": rollouts.height,
        "tables": {t: {"rows": counts[t], "file": f"{t}.parquet"} for t in TABLES},
        "artifact_bytes": "not included: artifacts.parquet lists them, artifact_texts their texts",
        "rollouts": [
            {
                **{k: r[k] for k in (*ROLLOUT_KEY, "n_events", "n_tool_calls")},
                "rows": {t: r.get(t) or 0 for t in TABLES},
            }
            for r in listed.iter_rows(named=True)
        ],
    }


TRACE = Exporter(
    name="trace",
    title="trace tables",
    media_type="application/zip",
    suffix=".zip",
    scopes=(RUN, ROLLOUT),
    reads={t: CURRENT.table(t).column_names for t in TABLES},
    produce=produce,
    means=(
        "Every rollout-keyed table of the run (rollouts, attempts, events, parts, tool_calls, "
        "artifacts and the part/tool-call/artifact texts, and any such table the schema adds), "
        "read through the current schema, one `<table>.parquet` per table plus `manifest.json`: "
        "the schema version read through, rows per table, and per rollout its `n_events` / "
        "`n_tool_calls` rollups and the rows each table holds for it. Artifact bytes are not "
        "included. A run-scoped download takes `?rollout=<task_id>:<repeat>` and `?task_id=`, "
        "each repeatable, to export only those rollouts."
    ),
)
