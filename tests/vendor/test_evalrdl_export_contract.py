# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The exporter contract, against this package alone.

Every registered exporter reads only what the CURRENT schema declares, and builds on a
partition written at any declared version: `RunTables` projects the older ones.
"""

from __future__ import annotations

import json
import subprocess
import sys

import polars as pl
import pytest

from evalrdl_export import PUBLIC, RUN, Exporter
from evalrdl_export.conformance import (
    check_builds_at_every_version,
    check_declaration,
    sample_rows,
    write_partition,
)
from evalrdl_schema import CURRENT, V11, Refusal
from evalrdl_schema.tables import ROLLOUT_KEY, RunTables


def _exporter(**kw) -> Exporter:
    base = dict(
        name="probe",
        title="Probe",
        media_type="text/csv",
        suffix=".csv",
        scopes=(RUN,),
        reads={"rollouts": ("task_id", "repeat", "reward")},
        build=lambda tables, run, only: tables.scan("rollouts").collect(),
    )
    return Exporter(**{**base, **kw})


# ── the contract, checked when an exporter is declared ──


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_every_public_exporter_reads_only_declared_columns(name):
    assert PUBLIC[name].name == name
    check_declaration(PUBLIC[name])


def test_a_column_the_schema_does_not_declare_is_refused():
    with pytest.raises(Refusal, match=r"rollouts\.\['score'\] is not declared"):
        _exporter(reads={"rollouts": ("task_id", "score")})


def test_the_runs_table_is_refused_because_the_run_row_is_passed_in():
    with pytest.raises(Refusal, match="gets the run's own row"):
        _exporter(reads={"runs": ("model",)})


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(name="Auto FA"), "URL segment"),
        (dict(scopes=("corpus",)), "scopes"),
        (dict(scopes=()), "scopes"),
        (dict(suffix="csv"), "suffix"),
        (dict(reads={}), "reads nothing"),
        (dict(build=None), "one of build and produce"),
        (dict(produce=lambda tables, run, only, out: 0), "one of build and produce"),
    ],
)
def test_a_malformed_declaration_is_refused(kw, match):
    with pytest.raises(Refusal, match=match):
        _exporter(**kw)


# ── RunTables: the declared schema, whatever version the partition was written at ──


def _partition(tmp_path, version, rows):
    return write_partition(tmp_path, version, rows)


_rows = sample_rows


def test_a_column_an_older_partition_lacks_reads_as_its_backfill(tmp_path):
    v11 = _partition(tmp_path, V11, _rows())
    assert "lane" not in pl.read_parquet_schema(v11 / "events.parquet")
    got = RunTables(v11, {"events": ("task_id", "lane", "event_idx")}).scan("events").collect()
    assert got.columns == ["task_id", "lane", "event_idx"]
    assert got.schema["lane"] == pl.Utf8
    assert got["lane"].to_list() == [CURRENT.table("events").column("lane").backfill] * 6


def test_a_null_in_a_backfilled_column_reads_as_the_backfill(tmp_path):
    rows = _rows()
    rows["events"][0]["lane"] = None
    d = _partition(tmp_path, CURRENT, rows)
    got = RunTables(d, {"events": ("lane",)}).scan("events").collect()
    assert got["lane"].null_count() == 0


def test_a_table_the_partition_does_not_have_reads_empty_and_typed(tmp_path):
    d = _partition(tmp_path, CURRENT, {"rollouts": _rows()["rollouts"]})
    got = RunTables(d, {"artifacts": ("task_id", "name")}).scan("artifacts").collect()
    assert got.height == 0 and got.schema == {"task_id": pl.Utf8, "name": pl.Utf8}


def test_a_table_outside_the_declared_reads_cannot_be_scanned(tmp_path):
    d = _partition(tmp_path, CURRENT, _rows())
    with pytest.raises(KeyError, match="events"):
        RunTables(d, {"rollouts": ("task_id",)}).scan("events")


def test_a_scope_keeps_only_its_rollouts_and_the_declared_columns(tmp_path):
    d = _partition(tmp_path, CURRENT, _rows(5))
    tables = RunTables(d, {"events": ("event_idx", "role"), "rollouts": ("task_id", "repeat")})
    with pytest.raises(KeyError, match="agent_ref"):
        tables.scoped(tables.scan("rollouts").collect())
    tables = RunTables(d, {"events": ("event_idx", "role"), "rollouts": ROLLOUT_KEY})
    seen = []
    for batch, scoped in tables.batches(tables.scan("rollouts").collect(), 2):
        ev = scoped.scan("events").collect()
        assert ev.columns == ["event_idx", "role"]
        assert ev.height == 2 * batch.height
        seen += batch["task_id"].to_list()
    assert seen == [f"t{i}" for i in range(5)]


@pytest.mark.parametrize("name", sorted(PUBLIC))
def test_every_public_exporter_builds_on_every_declared_version(tmp_path, name):
    check_builds_at_every_version(PUBLIC[name], tmp_path)


# ── a package of its own ──

PROBE = """
import sys, json
before = set(sys.modules)
import evalrdl_export
print(json.dumps(sorted(set(sys.modules) - before)))
"""


def test_the_package_loads_no_service_and_no_server():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True)
    top = {m.split(".")[0] for m in json.loads(out.stdout)}
    ours = {m for m in top if m.startswith("evalrdl")}
    assert ours == {"evalrdl_export", "evalrdl_schema"}, sorted(ours)
    heavy = {"duckdb", "psycopg", "boto3", "botocore", "starlette", "uvicorn"}
    assert not sorted(top & heavy)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
