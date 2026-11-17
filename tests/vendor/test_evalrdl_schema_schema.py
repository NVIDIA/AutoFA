# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The schema is declared, versioned, and only ever grows; it needs nothing but the standard
library."""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import pairwise

import pytest

import evalrdl_schema as S
from evalrdl_schema.dsl import Add, Column, Dtype, Grain, Refusal, Schema, Table, evolve


def test_the_package_loads_only_the_standard_library():
    probe = (
        "import sys, json; before = set(sys.modules); import evalrdl_schema; "
        "print(json.dumps(sorted(set(sys.modules) - before)))"
    )
    loaded = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    foreign = sorted({m.split(".")[0] for m in loaded} - set(sys.stdlib_module_names))
    assert foreign == ["evalrdl_schema"], foreign


def test_every_version_is_an_additive_extension_of_the_one_before():
    """The chain is what lets a partition outlive the schema it was written at."""
    for older, newer in pairwise(S.VERSIONS):
        d = S.diff(older, newer)
        assert d.is_compatible, f"{newer.version} breaks {older.version}: {d}"
        assert not d.removed and not d.retyped and not d.removed_tables


def test_a_key_column_added_later_carries_the_value_older_rows_take():
    """`lane` joined the key of the five trace tables. A v11 row has no lane column at
    all, so without a declared constant its key would hold a NULL and the row would stop
    being addressable."""
    for table in ("events", "parts", "part_texts", "tool_calls", "tool_call_texts"):
        key = S.CURRENT.table(table).key
        assert "lane" in key, table
        assert "lane" not in S.V11.table(table).key, table
        assert S.CURRENT.table(table).column("lane").backfill == S.LANE_DEFAULT is not None


def test_every_column_added_after_v11_says_what_it_holds():
    """v11 is grandfathered: its dict carried prose on 23 of 171 columns and inventing the
    rest would add words, not information. Everything added since must explain itself."""
    v11 = S.V11.columns()
    silent = []
    for table in S.CURRENT.tables:
        for col in table.columns:
            if col.name not in v11.get(table.name, ()) and not col.means.strip():
                silent.append(f"{table.name}.{col.name}")
    assert not silent, f"added after v11 and unexplained: {silent}"


def test_a_widened_key_is_not_applied_inline():
    """`ADD PRIMARY KEY` takes ACCESS EXCLUSIVE on a populated table. The generator emits
    it as a scheduled block, never as a statement the migration runner will execute."""
    from evalrdl_schema.backends import migration_sql

    for _, sql in migration_sql(S.V11, S.V12):
        for line in sql.splitlines():
            if "ADD PRIMARY KEY" in line:
                assert line.lstrip().startswith("--"), line


def test_each_migration_alters_one_table():
    """The runner applies a file in one transaction, and an ALTER holds ACCESS EXCLUSIVE until
    it commits. Two tables in one file would keep the first locked while the runner waits for
    the second, and every read of the first would queue behind it."""
    import re

    from evalrdl_schema.backends import migration_sql

    for slug, sql in migration_sql(S.V11, S.V12):
        tables = set(re.findall(r'^(?:ALTER TABLE|CREATE TABLE IF NOT EXISTS) "(\w+)"', sql, re.M))
        assert len(tables) == 1, (slug, tables)


# ----------------------------------------------------------------- the DSL's refusals


def _base() -> Schema:
    return Schema(
        "t-v1",
        (
            Table(
                "e", Grain.TRACE, ("id", "idx"), (Column("id", Dtype.STR), Column("idx", Dtype.INT))
            ),
        ),
    )


def test_dropping_a_column_is_refused():
    with pytest.raises(Refusal, match="drops column"):
        S.check_additive(
            _base(), Schema("t-v2", (Table("e", Grain.TRACE, ("id",), (Column("id", Dtype.STR),)),))
        )


def test_retyping_a_column_is_refused():
    with pytest.raises(Refusal, match="retypes"):
        S.check_additive(
            _base(),
            Schema(
                "t-v2",
                (
                    Table(
                        "e",
                        Grain.TRACE,
                        ("id", "idx"),
                        (Column("id", Dtype.STR), Column("idx", Dtype.STR)),
                    ),
                ),
            ),
        )


def test_widening_a_key_without_a_backfill_is_refused():
    with pytest.raises(Refusal, match="must name the value"):
        evolve(
            _base(),
            "t-v2",
            add=(Add("e", (Column("lane", Dtype.STR, "x"),), key=("id", "lane", "idx")),),
        )


def test_widening_a_key_with_a_backfill_is_allowed():
    out = evolve(
        _base(),
        "t-v2",
        add=(Add("e", (Column("lane", Dtype.STR, "x", backfill=""),), key=("id", "lane", "idx")),),
    )
    assert out.table("e").key == ("id", "lane", "idx")


def test_narrowing_a_key_is_refused():
    with pytest.raises(Refusal, match="never narrowed"):
        evolve(_base(), "t-v2", add=(Add("e", key=("id",)),))


def test_redeclaring_a_column_is_refused():
    with pytest.raises(Refusal, match="already exists"):
        evolve(_base(), "t-v2", add=(Add("e", (Column("idx", Dtype.INT),)),))


def test_a_key_naming_no_column_is_refused():
    with pytest.raises(Refusal, match="no column for"):
        Table("e", Grain.TRACE, ("nope",), (Column("id", Dtype.STR),))


def test_a_backfill_of_the_wrong_type_is_refused():
    with pytest.raises(Refusal, match="is not a int"):
        Column("n", Dtype.INT, "x", backfill="zero")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))


def test_a_vocabulary_version_only_adds_and_a_changed_meaning_is_refused():
    from evalrdl_schema import vocab

    assert vocab.CURRENT.version == "vocab-v5"
    # An inferred kind says so, and orders nothing; a stated one names no rule.
    inferred = {k: e.inferred_from for k, e in vocab.CURRENT.edges.items() if e.inferred_from}
    assert inferred == {"produced_by_text": "text-v1", "sent_in_by_text": "text-v1"}
    assert all(vocab.CURRENT.edges[k].earlier is None for k in inferred)
    assert not any(e.inferred_from for e in vocab.V2.edges.values())
    assert vocab.V1.words.items() <= vocab.V2.words.items() and not vocab.V1.edges
    assert {k: e.earlier for k, e in vocab.edge_kinds().items()} == {
        "produced": "frm",
        "sent_in": "frm",
        "consumed": "to",
        "produced_by_text": None,
        "sent_in_by_text": None,
        "telemetry_parent": "to",
        "served_request": None,
        "spawned": "frm",
    }
    produced = vocab.V2.edges["produced"]
    with pytest.raises(ValueError, match="a changed meaning is a new word"):
        vocab.evolve(
            vocab.V2,
            "vocab-x",
            "test",
            edges=(vocab.EdgeKind("produced", produced.means, "to", produced.because),),
        )
    dropped = vocab.Vocabulary("vocab-x", "test", {**vocab.V2.words}, {})
    with pytest.raises(ValueError, match="changes or drops edge kind"):
        vocab.check_additive(vocab.V2, dropped)
