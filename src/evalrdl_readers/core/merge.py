# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steps 4, 6 and 7 of a read, one grain at a time: settle, number, fold.

Settling turns the answers to one row into that row: one answer is the row, several answers
to a `Merge.ONE` question are ranked by the declared `Order`, several to a `Merge.UNION`
stream are deduplicated by identity and all kept. Stream rows are then numbered per lane,
and folds lift a grain's answers one rung up the ladder as answers of the grain above.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, Any

from .errors import Refusal
from .findings import Finding, Kind
from .input import Given
from .questions import DEPTH, Grain, Merge, Question
from .result import ReadResult, Row

if TYPE_CHECKING:
    from .reader import Reader


def is_mapping(value: Any) -> bool:
    """`isinstance(value, Mapping)`, with the common case answered before the ABC is asked."""
    return type(value) is dict or isinstance(value, Mapping)


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


_RUNS = re.compile(r"\d+|\D+")


def _natural(text: str) -> tuple:
    """A string split into text and number runs, so digits compare as numbers."""
    if text.isdecimal():
        return ((0, int(text), ""),)
    return tuple([
        (0, int(part), "") if part.isdigit() else (1, 0, part)
        for part in _RUNS.findall(text)
    ]) or ((1, 0, ""),)


def _sort_key(value: Any) -> tuple[int, Any]:
    """A total order over values of mixed type, so a sort never raises."""
    if value is _MISSING or value is None:
        return (3, ())
    if isinstance(value, bool):
        return (1, _natural(str(value)))
    if isinstance(value, (int, float)):
        return (0, value)
    if isinstance(value, str):
        return (1, _natural(value))
    return (2, _natural(repr(value)))


def _first(pair: tuple) -> Any:
    return pair[0]


def settle(reader: Reader, grain: Grain, given: list[Given], out: ReadResult) -> None:
    groups: dict[tuple[str, tuple], list[Given]] = {}
    for g in given:
        groups.setdefault((g.question, g.key), []).append(g)

    # A describing input may fill rows an enumerating input made, never create them.
    backed: dict[str, set[tuple]] = {}
    for (qname, key), got in groups.items():
        decl = reader.rows_decl.get(qname)
        if decl is None:
            continue
        if any(g.input in decl.may_create for g in got):
            backed.setdefault(qname, set()).add(key)
            out.fired[f"rows:{decl.question}"] += 1

    # A grain-level declaration backs a key, not a question.
    grain_decl = reader.rows_by_grain.get(grain)
    grain_backed: set[tuple] = set()
    if grain_decl is not None:
        grain_backed = {
            g.key for g in given if g.input in grain_decl.may_create
        }
        if grain_backed:
            out.fired[f"rows:{grain_decl.question.name}"] += len(grain_backed)

    for (qname, key), got in sorted(
        groups.items(), key=lambda kv: (kv[0][0], tuple(map(repr, kv[0][1])))
    ):
        q = reader.questions[qname]
        decl = reader.rows_decl.get(qname)
        if decl is None and grain_decl is not None:
            decl, unbacked = grain_decl, key not in grain_backed
        else:
            unbacked = decl is not None and key not in backed.get(qname, ())
        if unbacked:
            names = (
                f"no input in {list(decl.may_create)} — the inputs that may create "
                f"rows at {grain.name} grain —"
                if isinstance(decl.question, Grain)
                else f"no input in {list(decl.may_create)}"
            )
            for g in sorted({g.input for g in got}):
                out.findings.append(
                    Finding(
                        Kind.UNBACKED, qname, f"{key!r}",
                        f"{g!r} describes this row and {names} produced it, so it is "
                        f"not a row — {decl.because}",
                        g,
                    )
                )
            continue
        got = sorted(got, key=lambda g: (g.source, g.input, g.input_key))
        if len(got) == 1:
            out.rows[(qname, key)] = [Row.of(got[0])]
        elif q.merge is Merge.ONE:
            out.rows[(qname, key)] = _pick(reader, q, key, got, out)
        else:
            out.rows[(qname, key)] = _union(reader, q, key, got, out)


def _pick(reader: Reader, q: Question, key: tuple, got: list[Given], out: ReadResult
) -> list[Row]:
    order = reader.order_by_question.get(q.name)
    if order is not None:
        out.fired[f"order:{order.question}"] += 1
    same = [g for g in got if g.source == got[0].source]
    if len(same) == len(got):
        sites = sorted({g.site for g in got})
        if len(sites) > 1:
            # One key across two sites: the key does not distinguish them.
            raise Refusal(
                f"{q.name!r} at {key!r} was answered {len(got)} times by the single "
                f"source {got[0].source!r}, at {len(sites)} different sites "
                f"{sites[:4]}. A landmark resolved to more than one site and the row "
                f"key does not say which — so these are not one row answered twice, "
                f"they are two rows with one name. Put the site in the key, or "
                f"narrow the landmark."
            )
        # Same source, one site: a duplicate record (e.g. resumed run); a loss.
        kept, dropped = got[0], got[1:]
        listed = ", ".join(f"{g.input_key}={g.value!r}" for g in dropped[:4])
        detail = (
            f"{kept.source} answered this row {len(got)} times at one site — "
            f"{kept.input_key}={kept.value!r} was kept and {len(dropped)} more were "
            f"dropped ({listed}). One source answering one row twice is a duplicate "
            f"record, not two sources disagreeing, so no Order settles it and none "
            f"can: a resumed run writes the same case twice and both copies are "
            f"equally authoritative. The published row count is now smaller than the "
            f"records read, which is the number `counts=` claims."
        )
        out.findings.append(Finding(Kind.MISSING, q.name, f"{key!r}", detail,
                                    kept.source))
        row = Row.of(kept)
        row.note(detail)
        if order is not None and order.keep_loser:
            out.losers[(q.name, key)] = [Row.of(g) for g in dropped]
        return [row]
    if order is None:  # pragma: no cover - refused at construction
        raise Refusal(f"{q.name!r}: no order, and {len(got)} sources answered")
    ranked = sorted(got, key=lambda g: (order.rank(g.source), g.input, g.input_key))
    win, lost = ranked[0], ranked[1:]
    row = Row.of(win)
    # A null beating a value is kept but reported as a loss.
    spoke = [g for g in lost if g.value is not None]
    if win.value is None and spoke:
        said = ", ".join(f"{g.source}={g.value!r}" for g in spoke)
        detail = (
            f"{win.source} won this settle carrying no value, and {said}. That value "
            f"is gone and no declaration chose to lose it: an order ranks the sources "
            f"that stated the fact, and None is the absence of one, not a rival "
            f"statement of it. If {win.source} means there is genuinely nothing here, "
            f"say it with Unreadable or do not answer at all — then the source that "
            f"did answer is the answer. If it does not mean that, the order is "
            f"ranking the wrong source first — {order.because}"
        )
        out.findings.append(Finding(Kind.MISSING, q.name, f"{key!r}", detail,
                                    win.source))
        row.note(detail)
    else:
        for loser in lost:
            if loser.value is None:
                continue
            if loser.value != win.value:
                out.findings.append(
                    Finding(
                        Kind.CONTESTED,
                        q.name,
                        f"{key!r}",
                        f"{win.source}={win.value!r} beat "
                        f"{loser.source}={loser.value!r} — {order.because}",
                        win.source,
                    )
                )
    if order.keep_loser:
        out.losers[(q.name, key)] = [Row.of(g) for g in lost]
    return [row]


def _union(reader: Reader, q: Question, key: tuple, got: list[Given], out: ReadResult
) -> list[Row]:
    """Dedupe a stream by identity, scoped per lane."""
    rows: list[Row] = []
    by_identity: dict[Hashable, Row] = {}
    for g in sorted(got, key=lambda g: (g.input, g.input_key)):
        row = Row.of(g)
        ident = _column(reader, q, g, "identity", out)
        if ident is _MISSING:
            ident = ("<no identity>", g.input, g.input_key)
        ident = (g.lane, ident)
        if ident in by_identity:
            first = by_identity[ident]
            if first.input == row.input:
                # Within one input a shared identity loses a row; across inputs it is a note.
                out.findings.append(
                    Finding(
                        Kind.MISSING,
                        q.name,
                        f"{key!r}",
                        f"{row.input} gave two rows the identity {ident!r} "
                        f"({first.name} and {row.name}); {first.name} was kept and "
                        f"{row.name} is gone. An identity deduplicates rows from "
                        f"DIFFERENT inputs describing one thing; within one input, "
                        f"two rows sharing one identity are either two rows the "
                        f"column cannot tell apart or one thing the producer wrote "
                        f"twice, and nothing here can say which. Whether their "
                        f"values matched is not the question — the equal-value path "
                        f"wrote only a note, and a note changes neither `ok` nor "
                        f"`quiet`: 350 turns in, 46 rows out, 304 gone, green. A row "
                        f"that was read and is not published is a loss.",
                        row.input,
                    )
                )
                first.note(f"deduped against {row.name} on identity {ident!r}")
                continue
            if first.value != row.value:
                # MISSING, not CONTESTED: no Order settled this, sort position did.
                out.findings.append(
                    Finding(
                        Kind.MISSING,
                        q.name,
                        f"{key!r}",
                        f"identity {ident!r} is claimed by {first.name} and "
                        f"{row.name} with different values; {first.name} was kept and "
                        f"{row.name}'s value is gone. {q.name!r} is Merge.UNION, which "
                        f"refuses an Order, so nothing chose between them — sort "
                        f"position did. Give the two rows distinct identities, or make "
                        f"the question Merge.ONE and rank the sources.",
                        row.input,
                    )
                )
            first.note(f"deduped against {row.name} on identity {ident!r}")
            continue
        by_identity[ident] = row
        rows.append(row)
    return rows


def _lane_column(row: Row, declared) -> Any:
    """Read a lane's `order_by` column(s) off one row; `_MISSING` if any is absent."""
    cols = declared if isinstance(declared, tuple) else (declared,)
    value = row.value
    if not is_mapping(value):
        return _MISSING
    got = tuple(value.get(c, _MISSING) for c in cols)
    if any(v is _MISSING for v in got):
        return _MISSING
    return got[0] if len(got) == 1 else got


def _column(reader: Reader, q: Question, g: Given | Row, which: str, out: ReadResult) -> Any:
    """Read the identity or order_by column out of one contributed row."""
    contrib = reader.contributes.get((q.name, g.input))
    if contrib is None:  # pragma: no cover - refused at construction
        raise Refusal(f"{q.name!r}: {g.input!r} contributes with no declaration")
    cols = (contrib.identity_columns if which == "identity"
            else (contrib.order_by,))
    col = getattr(contrib, which)
    value = g.value
    if not is_mapping(value):
        raise Refusal(
            f"Input({g.input!r}) answered {q.name!r}, a Merge.UNION stream whose "
            f"{which} is the column {col!r}, with a {type(value).__name__}. A row in a "
            f"stream is a mapping, so its declared columns can be addressed."
        )
    absent = [c for c in cols if c not in value]
    if absent:
        out.findings.append(
            Finding(
                Kind.MISSING,
                q.name,
                g.name,
                f"{which} column(s) {absent} absent from this row, so it cannot be "
                f"{'deduped' if which == 'identity' else 'interleaved'} — "
                f"{contrib.because}",
                g.input,
            )
        )
        return _MISSING
    if len(cols) == 1:
        return value[cols[0]]
    # A tuple, never a joined string: missing fields would collide as "None".
    return tuple(value[c] for c in cols)


def assign_positions(reader: Reader, grain: Grain, out: ReadResult) -> None:
    """Number stream rows per lane; `lane=None` is the single unnamed lane."""
    for (qname, key), rows in out.rows.items():
        q = reader.questions.get(qname)
        if q is None or q.grain is not grain or q.merge is not Merge.UNION:
            continue
        by_lane: dict[str | None, list] = {}
        for row in rows:
            by_lane.setdefault(row.lane, []).append(row)
        settled: list = []
        for lane_kind, lane_rows in ((k[0] if isinstance(k, tuple) else k, v)
                                    for k, v in by_lane.items()):
            # `Lane.order_by` wins over the per-contribution column; a lane is one producer's
            # sequence, so the declaration of the producer that wrote it.
            lane = reader.lane_for(lane_kind, lane_rows[0].input) if lane_kind else None
            declared = getattr(lane, "order_by", None)
            settled.extend(_number(reader, q, lane_rows, out, declared))
        out.rows[(qname, key)] = settled


def _number(reader: Reader, q: Question, rows: list[Row], out: ReadResult,
            declared=None) -> list[Row]:
    primary = []
    for row in rows:
        sort_value = (_lane_column(row, declared)
                      if declared is not None
                      else _column(reader, q, row, "order_by", out))
        primary.append((1 if sort_value is _MISSING else 0, _sort_key(sort_value), row.input))
    # Natural sort on `input_key` so unpadded indexes tie-break as numbers; only a key
    # shared by two rows ever reaches the tie-break, so only those rows compute it.
    tied = Counter(primary)
    decorated = [
        (key + (_natural(row.input_key),) if tied[key] > 1 else key, row)
        for key, row in zip(primary, rows, strict=True)
    ]
    decorated.sort(key=_first)
    ordered = [row for _k, row in decorated]
    for i, row in enumerate(ordered):
        row.position = i
    return ordered


def fold(reader: Reader, grain: Grain, by_grain: dict[Grain, list[Given]], out: ReadResult
) -> None:
    folds = [f for f in reader.folds if reader.questions[f.over].grain is grain]
    if not folds:
        return
    # Subjects are lower-grain keys that survived settling.
    produced = {
        key for (qname, key) in out.rows if reader.questions[qname].grain is grain
    }
    subjects: set[tuple] = {g.key for g in by_grain[grain]} & produced
    for f in folds:
        parent_grain = reader.questions[f.into].grain
        want = DEPTH[parent_grain]
        groups: dict[tuple, list[Any]] = {}
        counts: dict[tuple, int] = {}
        for key in subjects:
            parent = key[:want]
            counts[parent] = counts.get(parent, 0) + 1
            groups.setdefault(parent, [])
        folded = 0
        for (qname, key), rows in out.rows.items():
            if qname != f.over or not rows:
                continue
            values = groups.setdefault(key[:want], [])
            for row in rows:
                values.append(row.value)
            folded += len(rows)
        if folded:
            out.fired[f"fold:{f.into}<-{f.over}"] += folded
        for parent in sorted(groups, key=lambda k: tuple(map(repr, k))):
            has, value, notes = f.apply(groups[parent], counts.get(parent, 0))
            for n in notes:
                out.findings.append(
                    Finding(Kind.COVERAGE, f.into, f"{parent!r}", n, f.source)
                )
            if not has:
                continue
            by_grain[parent_grain].append(
                Given(
                    question=f.into,
                    value=value,
                    key=parent,
                    input="fold",
                    input_key=f.over,
                    source=f.source,
                    site=f"fold:{f.over}",
                )
            )
