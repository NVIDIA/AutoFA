# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every exporter must pass: it reads only declared columns, and builds a file from a run
written at every declared schema version."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from evalrdl_schema import CURRENT, VERSIONS, Schema, polars_schemas
from evalrdl_schema.tables import READABLE, RunTables

from .api import ROLLOUT, RUN, Exporter


def check_declaration(exp: Exporter) -> None:
    for table, cols in exp.reads.items():
        assert table in READABLE, table
        assert set(cols) <= set(CURRENT.table(table).column_names), (table, cols)
    assert set(exp.scopes) <= {RUN, ROLLOUT}


def sample_rows(n_rollouts: int = 3) -> dict[str, list[dict]]:
    """`n_rollouts` rollouts of two messages and a tool call each, keyed for every table."""
    key = lambda i: dict(run_id="r", task_id=f"t{i}", repeat=0, agent_ref="a")  # noqa: E731
    rows: dict[str, list[dict]] = {t: [] for t in READABLE}
    for i in range(n_rollouts):
        rows["rollouts"].append(dict(key(i), reward=float(i % 2), status="ok"))
        for e, (role, text) in enumerate([("user", f"ask {i}"), ("assistant", f"answer {i}")]):
            ev = dict(key(i), lane="", event_idx=e)
            rows["events"].append(dict(ev, kind="message", role=role))
            rows["parts"].append(dict(ev, part_idx=0, channel="content"))
            rows["part_texts"].append(dict(ev, part_idx=0, text=text))
        rows["tool_calls"].append(dict(key(i), lane="", event_idx=1, call_idx=0, tool_name="sh"))
    return rows


def write_partition(directory: Path, version: Schema, rows: dict[str, list[dict]]) -> Path:
    """A run partition written at `version`: each table with exactly that version's columns."""
    directory.mkdir(parents=True, exist_ok=True)
    schemas = polars_schemas(version)
    for table, rs in rows.items():
        if table in schemas:
            cols = schemas[table]
            pl.DataFrame(
                [{c: r.get(c) for c in cols} for r in rs], schema=cols, orient="row"
            ).write_parquet(directory / f"{table}.parquet")
    return directory


def check_builds_at_every_version(exp: Exporter, tmp: Path) -> None:
    run = {"run_id": "r", "model": "m", "benchmark": "b"}
    for version in VERSIONS:
        d = write_partition(tmp / version.version, version, sample_rows())
        for scope in exp.scopes:
            only = None if scope == RUN else ("t1", 0)
            out = d / f"{exp.name}-{scope}{exp.suffix}"
            rows = exp.to_file(RunTables(d, exp.reads), run, only, out)
            assert rows == (3 if scope == RUN else 1), (version.version, scope, rows)
            assert out.stat().st_size, (version.version, scope)
