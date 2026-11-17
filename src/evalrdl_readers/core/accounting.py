# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks that run once the rows are settled, and the accounting of the read.

Whole-table landings are checked against the schema's columns, `Compare`s are tested per
row, declared mints are verified, every file in the tree is either read, hidden by a
reasoned `Ignore`, or reported UNREAD, and `fired` records which declarations did anything.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from .findings import Finding, Kind
from .input import Given
from .landmarks import Site
from .merge import is_mapping
from .result import ReadResult
from .tree import Index

if TYPE_CHECKING:
    from .reader import Reader


def check_landed_columns(reader: Reader, out: ReadResult) -> None:
    """Check the keys of whole-table answers against the table's columns."""
    if reader.schema is None:
        return
    unknown: dict[tuple[str, str, str], list[str]] = {}
    unshaped: dict[tuple[str, str, str], list[str]] = {}
    for (qname, _key), rows in out.rows.items():
        q = reader.questions.get(qname)
        if q is None:
            continue
        table, _, column = q.lands.partition(".")
        if column:
            # Column landings store the value whole; see `store_schema.NO_HOME`.
            continue
        cols = reader.schema.get(table, ())
        for row in rows:
            if not is_mapping(row.value):
                unshaped.setdefault(
                    (qname, type(row.value).__name__, row.input), []
                ).append(row.name)
                continue
            for k in row.value:
                # A key naming another table carries this row's child rows.
                if k not in cols and k not in reader.schema:
                    unknown.setdefault((qname, str(k), row.input), []).append(row.name)
    for (qname, col, who), names in sorted(unknown.items()):
        lands = reader.questions[qname].lands
        out.findings.append(
            Finding(
                Kind.MISSING,
                qname,
                lands,
                f"{len(names)} row(s) carry the key {col!r} and the table "
                f"{lands.partition('.')[0]!r} has no column for it. Its columns are "
                f"{list(reader.schema[lands.partition('.')[0]])}. {qname!r} lands in a "
                f"WHOLE TABLE, so its answer IS a row of that table and every key of "
                f"it is a column — a key the table does not have is dropped by the "
                f"write step without a word. First: {sorted(names)[0]}. Either the "
                f"column is spelt differently in the store, or the store has no home "
                f"for this fact.",
                who,
            )
        )
    for (qname, kind, who), names in sorted(unshaped.items()):
        lands = reader.questions[qname].lands
        out.findings.append(
            Finding(
                Kind.MISSING,
                qname,
                lands,
                f"{len(names)} answer(s) carry a {kind}, and {qname!r} lands in the "
                f"whole table {lands!r} — where the answer IS the row and its keys "
                f"are the columns. A value with no keys fills no column. Land it in "
                f"one column with lands={lands}.<column>, or yield a mapping. "
                f"First: {sorted(names)[0]}.",
                who,
            )
        )


def compare(reader: Reader, out: ReadResult) -> None:
    """Check each `Compare` per row: an absent side is UNCHECKED, an uncomparable one a loss."""
    for c in reader.compares:
        keys = sorted(
            {k for (q, k) in out.rows if q in (c.left, c.right)},
            key=lambda k: tuple(map(repr, k)),
        )
        for key in keys:
            out.fired[f"compare:{c.left}~{c.right}"] += 1
            lrows = out.rows.get((c.left, key), [])
            rrows = out.rows.get((c.right, key), [])
            absent = [n for n, rows in ((c.left, lrows), (c.right, rrows)) if not rows]
            if absent:
                out.findings.append(
                    Finding(
                        Kind.UNCHECKED, c.left, f"{key!r}",
                        f"{' and '.join(absent)} answered nothing here, so "
                        f"{c.left!r} {c.phrase} {c.right!r} was not checked — "
                        f"{c.because}",
                        c.right,
                    )
                )
                continue
            lv, rv = lrows[0].value, rrows[0].value
            try:
                holds = c.holds(lv, rv)
            except TypeError:
                holds = None
            if holds is None or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                for v in (lv, rv)
            ):
                out.findings.append(
                    Finding(
                        Kind.MISSING, c.left, f"{key!r}",
                        f"{c.left!r}={lv!r} and {c.right!r}={rv!r} cannot be "
                        f"compared, so the claim that one is {c.phrase} the other "
                        f"could not be tested — {c.because}",
                        c.right,
                    )
                )
                continue
            if not holds:
                out.findings.append(
                    Finding(
                        Kind.MISSING, c.left, f"{key!r}",
                        f"{c.left!r}={lv!r} is not {c.phrase} {c.right!r}={rv!r} — "
                        f"{c.because}",
                        c.right,
                    )
                )


def check_mints(reader: Reader, answers: list[Given], out: ReadResult) -> None:
    minting = {prov.name for prov in reader.inputs if prov.mints}
    by_input: dict[str, list[Given]] = {}
    for g in answers:
        if g.input in minting and g.keys:
            by_input.setdefault(g.input, []).append(g)
    for prov in reader.inputs:
        mine = by_input.get(prov.name, ())
        for k in prov.mints:
            n = len({g.keys[k.space] for g in mine if k.space in g.keys})
            out.minted[k.space] = n
            if n == 0 and not k.optional:
                out.findings.append(
                    Finding(
                        Kind.COVERAGE,
                        f"mint:{k.space}",
                        prov.name,
                        f"Input({prov.name!r}) declares it mints {k.space!r} and "
                        f"emitted no key for it. A declaration nothing verifies is a "
                        f"worse foundation than no declaration, because a check built "
                        f"on it reports success — {k.because}",
                        prov.name,
                    )
                )


def account_for_files(reader: Reader, idx: Index, sites: dict[str, list[Site]], out: ReadResult
) -> None:
    hidden_by: dict[str, list[str]] = {}
    for ig in reader.ignored:
        for site in sites.get(ig.landmark.name, []):
            for e in idx.files_at(site.path, ig.segments):
                hidden_by.setdefault(e.path, []).append(f"{ig!r}: {ig.because}")
                out.fired[f"ignore:{ig.landmark.name}/{ig.pattern}"] += 1
    for path, who in sorted(hidden_by.items()):
        out.hidden[path] = who[0]
        if len(who) > 1:
            out.findings.append(
                Finding(
                    Kind.CONTESTED,
                    "ignore",
                    path,
                    f"hidden by {len(who)} declarations — {'; '.join(sorted(who))}",
                )
            )
        if path in out.read_paths:
            out.findings.append(
                Finding(
                    Kind.CONTESTED,
                    "ignore",
                    path,
                    f"is both read by an input and hidden by {who[0]}",
                )
            )
    for e in idx.files:
        if e.path in out.read_paths or e.path in out.hidden:
            continue
        out.findings.append(
            Finding(
                Kind.UNREAD,
                "unread",
                e.path,
                "no input reads this file and no Ignore accounts for it. An "
                "unexamined file stays loud.",
            )
        )


def record_firing(reader: Reader, out: ReadResult) -> None:
    """Stamp the declarations whose firing is visible in the result."""
    for name, sites in out.sites.items():
        if sites:
            out.fired[f"landmark:{name}"] += len(sites)
    sources: Counter = Counter()
    positioned: Counter = Counter()
    for (qname, _key), rows in out.rows.items():
        out.fired[f"question:{qname}"] += len(rows)
        for row in rows:
            sources[row.source] += 1
            if row.position is not None:
                positioned[(qname, row.input)] += 1
    for source, n in sources.items():
        prov, _, decoder = source.partition("/")
        if decoder:
            out.fired[f"decoder:{prov}/{decoder}"] += n
    for (qname, input_name), n in positioned.items():
        contrib = reader.contributes.get((qname, input_name))
        if contrib is not None:
            out.fired[f"contribute:{contrib.to}<-{contrib.input}"] += n
    for space, n in out.minted.items():
        if n:
            out.fired[f"mint:{space}"] += n
