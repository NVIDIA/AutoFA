# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stored schema as data: tables, columns, dtypes, keys.

Versions form a chain built by `evolve`, which only adds, so any partition in the chain is
readable under the newest (missing columns project as typed NULL). Engine-free; see `backends.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class Refusal(Exception):
    """A declaration that cannot mean what it says. Raised at import, never at runtime."""


class Dtype(Enum):
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


class Grain(Enum):
    """Which ladder rung a table's rows sit on. Fixes the key prefix and the read tier."""

    RUN = "run"
    TRACE = "trace"
    TEXTS = "texts"


@dataclass(frozen=True)
class Column:
    """`backfill` is the value in partitions written before the column existed (None = NULL).

    A column joining the key must name a non-NULL backfill, or rows lose their address.
    """

    name: str
    dtype: Dtype
    means: str = ""
    backfill: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise Refusal(f"Column({self.name!r}): a column name must be an identifier")
        if not isinstance(self.dtype, Dtype):
            raise Refusal(f"Column({self.name!r}): dtype must be a Dtype, got {self.dtype!r}")
        want = {Dtype.STR: str, Dtype.INT: int, Dtype.FLOAT: (int, float), Dtype.BOOL: bool}
        if self.backfill is not None and not isinstance(self.backfill, want[self.dtype]):
            raise Refusal(
                f"Column({self.name!r}): backfill {self.backfill!r} is not a {self.dtype.value}"
            )


@dataclass(frozen=True)
class Table:
    name: str
    grain: Grain
    key: tuple[str, ...]
    columns: tuple[Column, ...]
    means: str = ""

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise Refusal(f"Table({self.name!r}): declares {dupes} more than once")
        missing = [k for k in self.key if k not in names]
        if missing:
            raise Refusal(
                f"Table({self.name!r}): key names {missing}, which the table has no column for"
            )
        if not self.key:
            raise Refusal(f"Table({self.name!r}): a table without a key cannot be addressed")
        if len(set(self.key)) != len(self.key):
            raise Refusal(
                f"Table({self.name!r}): key {self.key} names a column twice, which emits "
                f"a PRIMARY KEY no engine will accept"
            )

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name}.{name}")

    def dtypes(self) -> dict[str, Dtype]:
        return {c.name: c.dtype for c in self.columns}


@dataclass(frozen=True)
class Schema:
    version: str
    tables: tuple[Table, ...]
    note: str = ""

    def __post_init__(self) -> None:
        names = [t.name for t in self.tables]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise Refusal(f"Schema({self.version!r}): declares table {dupes} more than once")

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tables)

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    def columns(self) -> dict[str, tuple[str, ...]]:
        return {t.name: t.column_names for t in self.tables}


# ------------------------------------------------------------------------ evolution

@dataclass(frozen=True)
class Add:
    """Columns appended to a table, and optionally a widened key.

    A widened key's gained columns must be new and carry a `backfill`.
    """

    table: str
    columns: tuple[Column, ...] = ()
    key: tuple[str, ...] | None = None


def evolve(base: Schema, version: str, *,
           add: tuple[Add, ...] = (),
           tables: tuple[Table, ...] = (),
           note: str = "") -> Schema:
    """`base` plus new columns and tables; any other change needs a new chain."""
    if version == base.version:
        raise Refusal(f"evolve: {version!r} is already the version of the schema it extends")
    by_name = {t.name: t for t in base.tables}
    for item in add:
        if item.table not in by_name:
            raise Refusal(
                f"evolve({version!r}): Add names table {item.table!r}, which {base.version} "
                f"does not have. A new table goes in `tables=`, not `add=`."
            )
        target = by_name[item.table]
        clash = sorted(set(c.name for c in item.columns) & set(target.column_names))
        if clash:
            raise Refusal(
                f"evolve({version!r}): {item.table}.{clash} already exists in {base.version}. "
                f"Redeclaring a column cannot change it; only a new chain can."
            )
        widened = target.key if item.key is None else item.key
        if item.key is not None:
            gained = set(item.key) - set(target.key)
            if not set(target.key) <= set(item.key):
                raise Refusal(
                    f"evolve({version!r}): the new key for {item.table!r} drops "
                    f"{sorted(set(target.key) - set(item.key))}. A key may be widened, "
                    f"never narrowed -- rows already written would collide."
                )
            fresh = {c.name: c for c in item.columns}
            for col in sorted(gained):
                if col not in fresh:
                    raise Refusal(
                        f"evolve({version!r}): {item.table}.{col} joins the key but is not "
                        f"among the columns this step adds"
                    )
                if fresh[col].backfill is None:
                    raise Refusal(
                        f"evolve({version!r}): {item.table}.{col} joins the key, so it must "
                        f"name the value every {base.version} row takes. Without it those "
                        f"rows have a NULL in their key and stop being addressable."
                    )
        by_name[item.table] = replace(
            target, columns=target.columns + item.columns, key=widened)
    for new in tables:
        if new.name in by_name:
            raise Refusal(
                f"evolve({version!r}): table {new.name!r} already exists in {base.version}"
            )
        by_name[new.name] = new
    out = Schema(version=version, tables=tuple(by_name.values()), note=note)
    check_additive(base, out)
    return out


@dataclass(frozen=True)
class Diff:
    added_tables: tuple[str, ...]
    removed_tables: tuple[str, ...]
    added: tuple[tuple[str, str], ...]
    removed: tuple[tuple[str, str], ...]
    retyped: tuple[tuple[str, str], ...]
    narrowed: tuple[str, ...]
    #: compatible when the gained columns backfill
    widened: tuple[str, ...]
    regrained: tuple[str, ...]
    reordered: tuple[str, ...]
    #: key column order moved: a different index prefix
    rekeyed: tuple[str, ...]
    #: backfill changed: repoints every existing row
    refilled: tuple[tuple[str, str], ...]

    @property
    def is_compatible(self) -> bool:
        return not (self.removed_tables or self.removed or self.retyped or self.narrowed
                    or self.regrained or self.reordered or self.rekeyed or self.refilled)


def _subsequence(small: tuple[str, ...], large: tuple[str, ...]) -> bool:
    """Every element of `small`, in order, somewhere in `large`."""
    it = iter(large)
    return all(c in it for c in small)


def diff(old: Schema, new: Schema) -> Diff:
    o, n = {t.name: t for t in old.tables}, {t.name: t for t in new.tables}
    added, removed, retyped, narrowed, widened = [], [], [], [], []
    regrained, reordered, rekeyed, refilled = [], [], [], []
    for name in sorted(set(o) & set(n)):
        ot, nt = o[name], n[name]
        oc, nc = ot.dtypes(), nt.dtypes()
        added += [(name, c) for c in nt.column_names if c not in oc]
        removed += [(name, c) for c in ot.column_names if c not in nc]
        retyped += [(name, c) for c in sorted(set(oc) & set(nc)) if oc[c] != nc[c]]
        if ot.grain is not nt.grain:
            regrained.append(name)
        # Declared order is the projection and on-disk sort order.
        kept = [c for c in nt.column_names if c in oc]
        if kept != list(ot.column_names):
            reordered.append(name)
        for c in sorted(set(oc) & set(nc)):
            if ot.column(c).backfill != nt.column(c).backfill:
                refilled.append((name, c))
        if ot.key != nt.key:
            if set(ot.key) - set(nt.key):
                narrowed.append(name)
            elif not _subsequence(ot.key, nt.key):
                # Old key columns keep their relative order; new ones may go anywhere.
                rekeyed.append(name)
            else:
                widened.append(name)
    return Diff(
        added_tables=tuple(sorted(set(n) - set(o))),
        removed_tables=tuple(sorted(set(o) - set(n))),
        added=tuple(added), removed=tuple(removed),
        retyped=tuple(retyped), narrowed=tuple(narrowed), widened=tuple(widened),
        regrained=tuple(regrained), reordered=tuple(reordered), rekeyed=tuple(rekeyed),
        refilled=tuple(refilled),
    )


def check_additive(old: Schema, new: Schema) -> None:
    d = diff(old, new)
    if d.is_compatible:
        return
    why = []
    if d.removed_tables:
        why.append(f"drops table(s) {list(d.removed_tables)}")
    if d.removed:
        why.append(f"drops column(s) {[f'{t}.{c}' for t, c in d.removed]}")
    if d.retyped:
        why.append(f"retypes column(s) {[f'{t}.{c}' for t, c in d.retyped]}")
    if d.narrowed:
        why.append(f"narrows the key of table(s) {list(d.narrowed)}")
    if d.rekeyed:
        why.append(f"reorders the key of table(s) {list(d.rekeyed)}")
    if d.regrained:
        why.append(f"changes the grain of table(s) {list(d.regrained)}")
    if d.reordered:
        why.append(f"reorders existing columns of table(s) {list(d.reordered)}")
    if d.refilled:
        why.append(
            f"changes the backfill of {[f'{t}.{c}' for t, c in d.refilled]}, which "
            f"repoints every row already written"
        )
    raise Refusal(
        f"{new.version} is not an additive extension of {old.version}: it "
        + "; ".join(why)
        + ". Partitions already written at the older version would stop being readable."
    )
