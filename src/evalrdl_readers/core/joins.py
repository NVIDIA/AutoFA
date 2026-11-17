# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 5 of a read: joins across inputs by a declared key space.

`Relate` records an edge from each row to the rows its key names; `Link` records an edge
between rows of one stream, from the values one input's rows name to the rows of another
carrying them as their key. Both run at the grain of the rows they write from, after that
grain is settled and before it is numbered.
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable
from typing import TYPE_CHECKING

from .attach import Link, Relate, link_namer
from .declare import Card, _Expectation
from .findings import Finding, Kind
from .input import Given
from .merge import _MISSING
from .questions import Grain
from .result import Edge, ReadResult, Row

if TYPE_CHECKING:
    from .reader import Reader


def _rows_of(out: ReadResult, question: str) -> list[tuple[tuple, list[Row]]]:
    """(key, rows) of one question, in key order."""
    return sorted(
        ((key, rows) for (qname, key), rows in out.rows.items() if qname == question),
        key=lambda kv: tuple(map(repr, kv[0])),
    )


def attach(reader: Reader, grain: Grain, answers: list[Given], out: ReadResult) -> None:
    for att in reader.attachments:
        if isinstance(att, Relate) and reader.questions[att.frm].grain is grain:
            _relate(reader, att, out)
        elif isinstance(att, Link) and reader.questions[att.to].grain is grain:
            _link(reader, att, out)


def _identity(reader: Reader, question: str, row: Row) -> str | None:
    """The row's own identity value, as its `Contribute` declares it, as an end is kept
    (`_named`)."""
    contrib = reader.contributes.get((question, row.input))
    for col in contrib.identity_columns if contrib else ():
        got = row.value.get(col) if hasattr(row.value, "get") else None
        if got is not None:
            return _named(got)
    return None


# An end is a key of the store, not a payload: a producer's id this long (a tool call id that
# carries its encoded reasoning) is kept by digest, matched or not; a miss keeps it whole in the
# edge's reason, a match in the row it reaches.
NAMED_MAX = 256


def _named(v) -> str:
    s = str(v)
    return s if len(s) <= NAMED_MAX else "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def _link(reader: Reader, ln: Link, out: ReadResult) -> None:
    """Per row key where both inputs gave rows: join the namer's `refs` to the minter's
    `keys`, record an edge per pair, and hold each end to its declared count. A row key
    where either input gave none is not checked: the source is absent there."""
    space = ln.on.space
    names_frm = (
        link_namer(ln, reader.by_name[ln.frm_input], reader.by_name[ln.to_input])
        is reader.by_name[ln.frm_input]
    )
    label = f"Link({ln.word}, {ln.frm_input} -> {ln.to_input}) on {space}"
    for key, rows in _rows_of(out, ln.to):
        frm_rows = [r for r in rows if r.input == ln.frm_input]
        to_rows = [r for r in rows if r.input == ln.to_input]
        if not frm_rows or not to_rows:
            out.fired[f"link:{ln.word}:{ln.frm_input}->{ln.to_input}:unchecked"] += 1
            continue
        out.fired[f"link:{ln.word}:{ln.frm_input}->{ln.to_input}"] += 1
        namers, minters = (frm_rows, to_rows) if names_frm else (to_rows, frm_rows)
        by_value: dict[Hashable, list[Row]] = {}
        for r in minters:
            if space in r.keys:
                by_value.setdefault(r.keys[space], []).append(r)
        # A pair once, however many of the row's values reach it; with how many rows its value
        # reached, where more than one did.
        pairs: dict[tuple, tuple[Row, Row, str]] = {}
        for r in sorted(namers, key=lambda r: r.stable):
            for v in r.refs.get(ln.word, ()):
                hits = by_value.get(v, [])
                if not hits and ln.inferred:
                    out.fired[f"link:{ln.word}:{ln.frm_input}->{ln.to_input}:unmatched"] += 1
                    continue
                if not hits:
                    end = ln.to_input if names_frm else ln.frm_input
                    reason = f"{space}={v!r} matched no row of {end!r} — {ln.because}"
                    out.findings.append(
                        Finding(
                            Kind.ORPHAN,
                            ln.to,
                            r.name,
                            reason,
                            ln.to_input if names_frm else ln.frm_input,
                        )
                    )
                    if names_frm:
                        # A reference that reaches nothing is a stored edge with no far end.
                        out.edges.append(
                            Edge(
                                "link",
                                ln.word,
                                ln.to,
                                r.name,
                                ln.to,
                                None,
                                reason,
                                frm_key=key,
                                to_key=None,
                                frm_lane=r.lane,
                                frm_site=r.site,
                                frm_id=_identity(reader, ln.to, r),
                                # The value it named: two misses of one row stay two edges.
                                to_id=_named(v),
                            )
                        )
                others = [m for m in hits if m is not r]
                if ln.inferred and len(others) > ln.max_matches:
                    out.fired[f"link:{ln.word}:{ln.frm_input}->{ln.to_input}:too_common"] += 1
                    continue
                many = (
                    f"one of {len(others)} rows carrying {space}={_named(v)!r} — {ln.because}"
                    if len(others) > 1
                    else ""
                )
                for m in others:
                    f, t = (r, m) if names_frm else (m, r)
                    pairs.setdefault((f.stable, t.stable), (f, t, many))
        reach: dict[tuple[str, str], int] = {}
        got: dict[tuple[str, str], int] = {}
        for f, t, many in pairs.values():
            reach[f.stable] = reach.get(f.stable, 0) + 1
            got[t.stable] = got.get(t.stable, 0) + 1
            out.edges.append(
                Edge(
                    "link",
                    ln.word,
                    ln.to,
                    f.name,
                    ln.to,
                    t.name,
                    many,
                    frm_key=key,
                    to_key=key,
                    frm_lane=f.lane,
                    to_lane=t.lane,
                    frm_site=f.site,
                    to_site=t.site,
                    frm_id=_identity(reader, ln.to, f),
                    to_id=_identity(reader, ln.to, t),
                )
            )
        carries = (
            (lambda r: bool(r.refs.get(ln.word))) if names_frm else (lambda r: space in r.keys)
        )
        carries_to = (
            (lambda r: space in r.keys) if names_frm else (lambda r: bool(r.refs.get(ln.word)))
        )
        for r in frm_rows:
            if carries(r):
                _hold(
                    _Expectation(ln.cardinality, ln.because),
                    reach.get(r.stable, 0),
                    f"{label}, from {r.name}",
                    r,
                    ln.to,
                    ln.to_input,
                    out,
                )
        if ln.to_card is not None:
            for r in to_rows:
                if carries_to(r):
                    _hold(
                        _Expectation(ln.to_card, ln.because),
                        got.get(r.stable, 0),
                        f"{label}, at {r.name}",
                        r,
                        ln.to,
                        ln.frm_input,
                        out,
                    )


def _hold(
    expect: _Expectation, n: int, where: str, row: Row, question: str, other: str, out: ReadResult
) -> None:
    err = expect.check(n, where)
    if err:
        out.findings.append(Finding(Kind.MISSING, question, row.name, err, other))
        row.note(err)


def _relate(reader: Reader, att: Relate, out: ReadResult) -> None:
    space = att.on.space
    targets: dict[Hashable, list[Row]] = {}
    for (qname, _key), rows in out.rows.items():
        if qname != att.to:
            continue
        for row in rows:
            if space in row.keys:
                targets.setdefault(row.keys[space], []).append(row)
    for group in targets.values():
        group.sort(key=lambda r: r.stable)
    received: dict[tuple[str, str], int] = {}
    for _key, rows in _rows_of(out, att.frm):
        for row in rows:
            out.fired[f"relate:{att.frm}->{att.to}"] += 1
            k = row.keys.get(space, _MISSING)
            matches = [] if k is _MISSING else targets.get(k, [])
            reason = (
                f"carries no {space!r} key — {att.because}"
                if k is _MISSING
                else f"{space}={k!r} matched nothing — {att.because}"
            )
            err = _Expectation(att.cardinality, att.because).check(
                len(matches), f"Relate({att.frm} -> {att.to}) on {space}"
            )
            if err:
                # A declared count not met is a loss; ORPHAN would not affect `ok`.
                detail = reason if not matches else err
                out.findings.append(Finding(Kind.MISSING, att.frm, row.name, detail, att.to))
                row.note(detail)
            elif not matches and att.cardinality is Card.ANY:
                out.findings.append(Finding(Kind.ORPHAN, att.frm, row.name, reason, att.to))
                row.note(reason)
            if not matches:
                out.edges.append(
                    Edge(
                        "relate",
                        space,
                        att.frm,
                        row.name,
                        att.to,
                        None,
                        reason,
                        frm_key=row.key,
                        to_key=None,
                        frm_lane=row.lane,
                        frm_site=row.site,
                    )
                )
                continue
            for m in matches:
                # Skip self-edges.
                if m.key == row.key and m.name == row.name:
                    continue
                received.setdefault(m.stable, 0)
                received[m.stable] = received.get(m.stable, 0) + 1
                out.edges.append(
                    Edge(
                        "relate",
                        space,
                        att.frm,
                        row.name,
                        att.to,
                        m.name,
                        frm_key=row.key,
                        to_key=m.key,
                        frm_lane=row.lane,
                        to_lane=m.lane,
                        frm_site=row.site,
                        to_site=m.site,
                    )
                )
    _check_to_card(reader, att, targets, received, out)


def _check_to_card(
    reader: Reader,
    att: Relate,
    targets: dict[Hashable, list[Row]],
    received: dict[tuple[str, str], int],
    out: ReadResult,
) -> None:
    """Check how many `frm` rows each `to` row received; `to_card=None` skips it."""
    if att.to_card is None:
        return
    expect = _Expectation(att.to_card, att.because)
    seen: set[tuple[str, str]] = set()
    for k in sorted(targets, key=repr):
        for row in targets[k]:
            if row.stable in seen:
                continue
            seen.add(row.stable)
            n = received.get(row.stable, 0)
            err = expect.check(n, f"Relate({att.frm} -> {att.to}) on {att.on.space}, at {row.name}")
            if err:
                out.findings.append(Finding(Kind.MISSING, att.to, row.name, err, att.frm))
                row.note(err)
