# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The IR mapped onto each engine. The only place a dtype becomes a library's type."""

from __future__ import annotations

from .dsl import Dtype, Refusal, Schema

DUCK = {Dtype.STR: "VARCHAR", Dtype.INT: "BIGINT", Dtype.FLOAT: "DOUBLE",
        Dtype.BOOL: "BOOLEAN"}
POSTGRES = {Dtype.STR: "text", Dtype.INT: "bigint", Dtype.FLOAT: "double precision",
            Dtype.BOOL: "boolean"}


def polars_schemas(schema: Schema) -> dict[str, dict]:
    """The `{table: {column: pl.DataType}}` the store has always been written from."""
    import polars as pl

    to_pl = {Dtype.STR: pl.Utf8, Dtype.INT: pl.Int64, Dtype.FLOAT: pl.Float64,
             Dtype.BOOL: pl.Boolean}
    return {t.name: {c.name: to_pl[c.dtype] for c in t.columns} for t in schema.tables}


#: Parquet states a page's size as an i32. Polars cuts a row group into pages of equal row
#: counts sized by the average value, so a run of long values can fill one page past 2 GiB. A
#: page lies inside one row group, so row groups of at most this many bytes bound every page.
PARQUET_ROW_GROUP_BYTES = 1 << 30


def widest_value(frame) -> int:
    """The longest string or binary value of a polars DataFrame or LazyFrame, in bytes."""
    import polars as pl

    schema = frame.collect_schema()
    names = [n for n, t in schema.items() if t in (pl.String, pl.Binary)]
    if not names:
        return 0
    got = frame.select(pl.max_horizontal(
        (pl.col(n).str.len_bytes() if schema[n] == pl.String else pl.col(n).bin.size()).max()
        for n in names
    ))
    got = got.collect() if isinstance(got, pl.LazyFrame) else got
    return int(got.item() or 0)


def parquet_row_group_rows(widest: int, most: int) -> int:
    """Rows per row group so that no column of one, at `widest` bytes a value, passes
    `PARQUET_ROW_GROUP_BYTES`."""
    return max(1, min(most, PARQUET_ROW_GROUP_BYTES // max(widest, 1)))


def keys(schema: Schema) -> dict[str, list[str]]:
    return {t.name: list(t.key) for t in schema.tables}


def grains(schema: Schema) -> dict[str, str]:
    return {t.name: t.grain.value for t in schema.tables}


def sql_literal(value) -> str:
    """A declared constant as SQL; refuses `nan`/`inf`, which have no portable literal."""
    import math

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and not math.isfinite(value):
        raise Refusal(
            f"backfill {value!r} has no portable SQL literal; a column whose absent value "
            f"is not a number should declare NULL"
        )
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def pg_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def create_table_sql(table) -> str:
    """`CREATE TABLE IF NOT EXISTS` for one declared table, key included."""
    cols = [f"    {pg_ident(c.name)} {POSTGRES[c.dtype]}" for c in table.columns]
    key = ", ".join(pg_ident(k) for k in table.key)
    cols.append(f"    PRIMARY KEY ({key})")
    return (f"CREATE TABLE IF NOT EXISTS {pg_ident(table.name)} (\n"
            + ",\n".join(cols) + "\n);")


def add_column_sql(table: str, column) -> str:
    """`ADD COLUMN IF NOT EXISTS` (catalog-only), with the backfill as DEFAULT.

    The DEFAULT keeps later rows non-NULL, which a column joining the key requires.
    """
    out = (f"ALTER TABLE {pg_ident(table)} "
           f"ADD COLUMN IF NOT EXISTS {pg_ident(column.name)} {POSTGRES[column.dtype]}")
    if column.backfill is not None:
        out += f" DEFAULT {sql_literal(column.backfill)}"
    return out + ";"


def migration_sql(old: Schema, new: Schema) -> list[tuple[str, str]]:
    """Postgres DDL from `old` to `new`, as `(slug, sql)` steps.

    One step per table: each file is one transaction, and an ALTER holds ACCESS EXCLUSIVE until
    commit. TEXTS-grain tables are S3-only and get no DDL.
    """
    from .dsl import Grain, diff

    d = diff(old, new)

    def in_pg(name: str) -> bool:
        return new.table(name).grain is not Grain.TEXTS

    head = [f"-- {old.version} -> {new.version}. GENERATED from",
            f"-- evalrdl/storage/schema/{new.version.removeprefix('store-')}/__init__.py by",
            "-- `python -m evalrdl.storage.schema.emit`. Do not edit.", ""]
    steps: list[tuple[str, str]] = []
    for name in filter(in_pg, d.added_tables):
        steps.append((f"create_{name}", "\n".join(head + [create_table_sql(new.table(name))]) + "\n"))
    by_table: dict[str, list[str]] = {}
    for table, column in d.added:
        if not in_pg(table):
            continue
        by_table.setdefault(table, []).append(add_column_sql(table, new.table(table).column(column)))
    for table in sorted(by_table):
        body = head + by_table[table]
        if table in d.widened:
            old_key = ", ".join(old.table(table).key)
            new_key = ", ".join(new.table(table).key)
            body += [
                "",
                f"-- The key widens from ({old_key}) to ({new_key}).",
                "-- ADD PRIMARY KEY takes ACCESS EXCLUSIVE on a populated table, so it is NOT run",
                "-- here. The ADD COLUMN above gives every existing row the backfill (catalog-only",
                "-- since PG 11), so the new key is already unique and non-NULL; schedule with the",
                "-- write path quiesced:",
                f"--   ALTER TABLE {pg_ident(table)} DROP CONSTRAINT IF EXISTS {pg_ident(table + '_pkey')};",
                f"--   ALTER TABLE {pg_ident(table)} ADD PRIMARY KEY "
                f"({', '.join(pg_ident(k) for k in new.table(table).key)});",
            ]
        steps.append((f"columns_{table}", "\n".join(body) + "\n"))
    return steps
