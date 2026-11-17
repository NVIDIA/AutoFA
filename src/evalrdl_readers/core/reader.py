# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reader: declarations, the checks over them at construction, and one read.

Unstated choices are refused at construction; what only a real folder can show becomes a
`Finding` during the read. A read runs in steps, each in its own module: landmarks
(`landmarks`), every input at every site (`translate`), then per grain lowest first
settle, join, number, carried edges and fold (`merge`, `joins`), then the checks and
accounting that need the whole result (`accounting`).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

from .accounting import (
    account_for_files,
    check_landed_columns,
    check_mints,
    compare,
    record_firing,
)
from .alias import alias_rollouts
from .attach import Contribute, Lane, Link, Relate, Unlinked, link_namer
from .declare import Card, Compare, Fold, Ignore, Order, Population, Rows
from .errors import Refusal
from .findings import Finding, Kind, sort_findings
from .input import Given, Input
from .joins import attach
from .landmarks import ROOT, Landmark, resolve_landmarks
from .merge import assign_positions, fold, settle
from .questions import DEPTH, LOWEST_FIRST, PARENT, Grain, Merge, Question
from .result import ReadResult
from .translate import report_unchecked, run_inputs
from .tree import Index, LocalTree, Tree


def _grouped_inputs(reader: str, inputs) -> tuple[tuple[tuple[str, tuple], ...], tuple]:
    """Split `inputs=` into producer groups and the flat tuple everything else uses."""
    items = tuple(inputs)
    pairs = [x for x in items if not isinstance(x, Input)]
    if not pairs:
        return (), items
    if len(pairs) != len(items):
        loose = [x.name for x in items if isinstance(x, Input)]
        raise Refusal(
            f"Reader({reader!r}): inputs mixes bare Input objects {loose} with "
            f"(producer, inputs) groups. Half a grouping is worse than none: the "
            f"ungrouped inputs would silently have no producer while the rest look "
            f"accounted for. Group all of them or none."
        )
    groups: list[tuple[str, tuple]] = []
    flat: list = []
    where: dict[str, str] = {}
    for item in items:
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                and item[0].strip()):
            raise Refusal(
                f"Reader({reader!r}): a grouped `inputs=` entry is "
                f"(producer_name, (Input, ...)), got {item!r}"
            )
        producer, members = item[0], tuple(item[1])
        if not members:
            raise Refusal(
                f"Reader({reader!r}): group {producer!r} declares no inputs. A producer "
                f"this reader reads nothing from is a name in the stack that can never "
                f"produce a row."
            )
        if producer in dict(groups):
            raise Refusal(
                f"Reader({reader!r}): producer {producer!r} names two groups. The stack "
                f"is ordered outermost-first, so one producer appearing twice makes the "
                f"order meaningless."
            )
        for i in members:
            name = getattr(i, "name", None)
            if where.get(name):
                raise Refusal(
                    f"Reader({reader!r}): input {name!r} is in group {where[name]!r} "
                    f"and in group {producer!r}. Exactly one producer wrote the bytes "
                    f"it reads."
                )
            where[name] = producer
        groups.append((producer, members))
        flat.extend(members)
    return tuple(groups), tuple(flat)


class Reader:
    """A set of inputs, and everything that may be said about them; checked when built.

    With `schema={table: [column, ...]}` every `lands=` is checked at construction.

    Arguments:
        name: The reader's name, recorded on every run it reads.
        landmarks: Every landmark its paths are relative to.
        inputs: Its inputs; or `((PRODUCER, (Input, ...)), ...)`, grouped by the producer
            that wrote the bytes each reads.
        questions: Every question its inputs answer.
        counts: Which rollouts its rows are drawn from; required.
        orders: Who wins where two sources answer one `Merge.ONE` question.
        rows: Which inputs may create rows; required for every question below run grain.
        folds: Totals lifted one grain up.
        compare: Relations that must hold between two answered questions.
        ignored: Files read on purpose for nothing.
        attach: `Contribute`, `Link`, `Relate` and `Unlinked` declarations.
        lanes: The lane kinds its stream rows may name.
        schema: `{table: [column, ...]}` that every landing is resolved against; `None` checks
            nothing.
    """

    def __init__(
        self,
        name: str,
        landmarks: Iterable[Landmark],
        inputs: Iterable[Input] | Iterable[tuple[str, Iterable[Input]]],
        questions: Iterable[Question],
        counts: Population | None = None,
        orders: Iterable[Order] = (),
        rows: Iterable[Rows] = (),
        folds: Iterable[Fold] = (),
        compare: Iterable[Compare] = (),
        ignored: Iterable[Ignore] = (),
        attach: Iterable[Any] = (),
        lanes: Iterable[Lane] = (),
        schema: Mapping[str, Iterable[str]] | None = None,
    ):
        if not isinstance(name, str) or not name.strip():
            raise Refusal(f"Reader: needs a name, got {name!r}")
        self.name = name
        self.landmarks = tuple(landmarks)
        self.input_groups, self.inputs = _grouped_inputs(name, inputs)
        self.declared_stack = tuple(p for p, _ in self.input_groups)
        self.producer_of = {i.name: p for p, ins in self.input_groups for i in ins}
        self.question_list = tuple(questions)
        if not isinstance(counts, Population):
            raise Refusal(
                f"Reader({name!r}): `counts=` is not declared. Say which rollouts this "
                f"reader's rows are drawn from — one of "
                f"{[p.name for p in Population]}. There is no default: measured on one "
                f"real folder, four readers that each skipped this question produced 0, "
                f"45 and 503 rollouts, and every one of them passed every other check."
            )
        self.counts = counts
        self.orders = tuple(orders)
        rows = tuple(rows)
        self.rows = rows
        self.mounts: tuple = ()
        for r in rows:
            if not isinstance(r, Rows):
                raise Refusal(
                    f"Reader({name!r}): rows takes Rows objects, got {type(r).__name__}"
                )
        self.rows_decl = {r.question: r for r in rows if not isinstance(r.question, Grain)}
        self.rows_by_grain = {r.question: r for r in rows if isinstance(r.question, Grain)}
        if len(self.rows_decl) + len(self.rows_by_grain) != len(rows):
            raise Refusal(f"Reader({name!r}): two Rows for one question or grain")
        self.folds = tuple(folds)
        for f in self.folds:
            if not isinstance(f, Fold):
                raise Refusal(
                    f"Reader({name!r}): folds takes Fold objects, got {type(f).__name__}"
                )
        self.compares = tuple(compare)
        self.ignored = tuple(ignored)
        self.attachments = tuple(attach)
        self.lanes = tuple(lanes)
        self.lane_kinds: dict[str, Lane] = {}
        for ln in self.lanes:
            if not isinstance(ln, Lane):
                raise Refusal(
                    f"Reader({name!r}): lanes takes Lane objects, got {type(ln).__name__}"
                )
            if ln.kind in self.lane_kinds:
                raise Refusal(f"Reader({name!r}): two lanes of kind {ln.kind!r}")
            self.lane_kinds[ln.kind] = ln
        #: (input, kind) -> the Lane a mounted producer declared for its own rows of that kind,
        #: where it orders them differently from the host's.
        self.lane_by_input: dict[tuple[str, str], Lane] = {}
        if schema is None:
            self.schema = None
        else:
            if not isinstance(schema, Mapping):
                raise Refusal(
                    f"Reader({name!r}): schema= is a {{table: [column, ...]}} mapping of "
                    f"the tables the application actually has, got "
                    f"{type(schema).__name__}"
                )
            self.schema = {t: tuple(cols) for t, cols in schema.items()}
            if not self.schema:
                raise Refusal(
                    f"Reader({name!r}): schema= is empty. A mapping with no tables would "
                    f"refuse every landing; pass None to say this reader is standalone."
                )
        self._validate()

    # -- validation --------------------------------------------------------

    def _validate(self) -> None:
        self._check_landmarks()
        self._check_inputs()
        self._check_questions()
        self._check_lands()
        self._check_folds()
        self._check_compares()
        self._check_sources_and_orders()
        self._check_keys()
        self._check_attachments()

    def _check_landmarks(self) -> None:
        if not self.landmarks:
            raise Refusal(f"Reader({self.name}): no landmarks; nothing can be addressed")
        for lm in self.landmarks:
            if not isinstance(lm, Landmark):
                raise Refusal(
                    f"Reader({self.name}): landmarks takes Landmark objects, got "
                    f"{type(lm).__name__}"
                )
        names = [lm.name for lm in self.landmarks]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise Refusal(
                f"Reader({self.name}): two landmarks named {dupes}. A landmark's name is "
                f"printed in every finding about it, so two of them could not be told apart."
            )
        known = {id(lm) for lm in self.landmarks} | {id(ROOT)}
        for lm in self.landmarks:
            if lm.within is not None and id(lm.within) not in known:
                raise Refusal(
                    f"Reader({self.name}): Landmark({lm.name}) is within "
                    f"{lm.within.name!r}, which this reader does not declare. Add it to "
                    f"`landmarks=`."
                )
        self._landmarks_by_id = known
        for ig in self.ignored:
            if not isinstance(ig, Ignore):
                raise Refusal(
                    f"Reader({self.name}): `ignored` is a list of Ignore declarations, not "
                    f"a list of glob strings. A bare string would be matched against the "
                    f"whole relative path, so its `*` would cross `/` and its reach would "
                    f"have nothing to do with what it appeared to say."
                )
            if id(ig.landmark) not in known:
                raise Refusal(
                    f"Reader({self.name}): {ig!r} is anchored to a landmark this reader "
                    f"does not declare"
                )

    def _check_inputs(self) -> None:
        if not self.inputs:
            raise Refusal(f"Reader({self.name}): no inputs")
        for p in self.inputs:
            if not isinstance(p, Input):
                raise Refusal(
                    f"Reader({self.name}): inputs takes Input objects, got "
                    f"{type(p).__name__}"
                )
        names = [p.name for p in self.inputs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise Refusal(
                f"Reader({self.name}): two inputs named {dupes}. The winning source is "
                f"stored beside the value and printed in every finding, so the record "
                f"could not tell them apart afterwards."
            )
        self.by_name = {p.name: p for p in self.inputs}
        for p in self.inputs:
            for a in p.reads:
                if id(a.landmark) not in self._landmarks_by_id:
                    raise Refusal(
                        f"Input({p.name!r}) reads at Landmark({a.landmark.name}), which "
                        f"this reader does not declare. Add it to `landmarks=`."
                    )

    def _check_questions(self) -> None:
        for q in self.question_list:
            if not isinstance(q, Question):
                raise Refusal(
                    f"Reader({self.name}): questions takes Question objects, got "
                    f"{type(q).__name__}"
                )
        names = [q.name for q in self.question_list]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise Refusal(f"Reader({self.name}): two questions named {dupes}")
        self.questions = {q.name: q for q in self.question_list}
        for p in self.inputs:
            for q in p.answers:
                if q not in self.questions:
                    raise Refusal(
                        f"Input({p.name!r}) answers {q!r}, which this reader does not "
                        f"declare as a Question. A fact with no declared grain, merge and "
                        f"landing is a column nobody is responsible for."
                    )
        answered = {q for p in self.inputs for q in p.answers}
        folded = {f.into for f in self.folds}
        for q in self.questions:
            if q not in answered and q not in folded:
                raise Refusal(
                    f"Reader({self.name}): Question({q!r}) is declared and nothing answers "
                    f"it. Remove it, or name the input that fills it."
                )

    def _check_lands(self) -> None:
        self._resolve_lands()
        claimed: dict[str, str] = {}
        for q in self.question_list:
            for slot in q.slots():
                if slot in claimed:
                    raise Refusal(
                        f"Reader({self.name}): {q.name!r} and {claimed[slot]!r} both land "
                        f"in {slot!r}. `lands` is checked injective, so one fact cannot "
                        f"acquire two names in two readers."
                    )
                claimed[slot] = q.name
        # A whole table and a column of that table are the same landing said twice.
        for slot, owner in sorted(claimed.items()):
            if "." not in slot:
                for other, other_owner in sorted(claimed.items()):
                    if other.startswith(f"{slot}.") :
                        raise Refusal(
                            f"Reader({self.name}): {owner!r} lands in the whole table "
                            f"{slot!r} and {other_owner!r} lands in {other!r}, which is "
                            f"part of it. `lands` is checked injective."
                        )

    def _resolve_lands(self) -> None:
        """Every landing names a known table and column; whole tables finish in
        `_check_landed_columns`."""
        if self.schema is None:
            return
        for q in self.question_list:
            for slot in q.slots():
                table, _, column = slot.partition(".")
                if table not in self.schema:
                    raise Refusal(
                        f"Reader({self.name}): Question({q.name!r}) lands in {slot!r} and "
                        f"the registered schema has no table {table!r}. It has "
                        f"{sorted(self.schema)}. A landing nothing resolves is a column "
                        f"the write step drops in silence."
                    )
                if column and column not in self.schema[table]:
                    raise Refusal(
                        f"Reader({self.name}): Question({q.name!r}) lands in {slot!r} and "
                        f"{table!r} has no column {column!r}. Its columns are "
                        f"{list(self.schema[table])}. Either the column is spelt "
                        f"differently in the store, or the store has no home for this "
                        f"fact — and a fact with no home is not stored by declaring that "
                        f"it is."
                    )

    def _check_folds(self) -> None:
        seen: set[tuple[str, str]] = set()
        for f in self.folds:
            if not isinstance(f, Fold):
                raise Refusal(
                    f"Reader({self.name}): folds takes Fold objects, got {type(f).__name__}"
                )
            for side in ("into", "over"):
                qname = getattr(f, side)
                if qname not in self.questions:
                    raise Refusal(
                        f"Fold(into={f.into!r}): {side}={qname!r} is not a Question this "
                        f"reader declares"
                    )
            into_q, over_q = self.questions[f.into], self.questions[f.over]
            parent = PARENT.get(over_q.grain)
            if parent is None:
                raise Refusal(
                    f"Fold(into={f.into!r}): {f.over!r} is at {over_q.grain.name} grain, "
                    f"which is the top of the ladder — there is nothing above it to fold into"
                )
            if into_q.grain is not parent:
                raise Refusal(
                    f"Fold(into={f.into!r}): {f.over!r} is at {over_q.grain.name} grain, so "
                    f"it folds into {parent.name} grain; {f.into!r} is declared at "
                    f"{into_q.grain.name}. A fold lifts one rung of the ladder."
                )
            if into_q.merge is not Merge.ONE:
                raise Refusal(
                    f"Fold(into={f.into!r}): that question is {into_q.merge.name}. A fold "
                    f"produces one answer at the grain above, so its question is settled "
                    f"with Merge.ONE."
                )
            if (f.into, f.over) in seen:
                raise Refusal(f"Reader({self.name}): Fold({f.into!r} <- {f.over!r}) twice")
            seen.add((f.into, f.over))
            # With `over` alone at its grain the gap is zero by construction, so
            # `Card.NONE` could never fire.
            if f.expect.card is Card.NONE:
                siblings = sorted(
                    n for n, q in self.questions.items()
                    if q.grain is over_q.grain and n != f.over
                )
                if not siblings:
                    raise Refusal(
                        f"Fold(into={f.into!r}, over={f.over!r}): "
                        f"expect=Card.NONE says do not publish this total over an "
                        f"incomplete population, and this fold cannot tell a complete "
                        f"population from an incomplete one. The subjects it counts are "
                        f"the rows some input named at {over_q.grain.name} grain, and "
                        f"{f.over!r} is the only question there — so every subject is one "
                        f"that answered and the gap is zero whatever the folder holds. "
                        f"Either answer a second question at {over_q.grain.name} grain "
                        f"from the input that ENUMERATES the rows (an id is enough: the "
                        f"rows it names but {f.over!r} does not become the gap), or "
                        f"declare expect=Card.ANY and say in the reason that this reader "
                        f"cannot know the denominator."
                    )
            if f.aggregate:
                stated = [p.name for p in self.inputs if f.into in p.answers]
                if stated:
                    raise Refusal(
                        f"Fold(into={f.into!r}, op={f.op.name}) is an aggregate — a property "
                        f"of a population — and {sorted(stated)} also state {f.into!r} "
                        f"directly, which is a property of a document. An Order would rank "
                        f"them and publish one, which is not a winner: it is the other "
                        f"question deleted. Two questions are sharing one name; give them two."
                    )

    def _check_compares(self) -> None:
        seen: set[tuple[str, str]] = set()
        for c in self.compares:
            if not isinstance(c, Compare):
                raise Refusal(
                    f"Reader({self.name}): compare takes Compare objects, got "
                    f"{type(c).__name__}"
                )
            left = self._question_or_refuse(c.left, "Compare.left")
            right = self._question_or_refuse(c.right, "Compare.right")
            if left.grain is not right.grain:
                raise Refusal(
                    f"Compare({c.left!r}, {c.right!r}): they are at {left.grain.name} and "
                    f"{right.grain.name} grain, so there is no row the two answers are "
                    f"both about. A claim across grains is a Fold and then a Compare."
                )
            for q in (left, right):
                if q.merge is not Merge.ONE:
                    raise Refusal(
                        f"Compare({c.left!r}, {c.right!r}): {q.name!r} is {q.merge.name}, "
                        f"so it settles a stream of rows and has no one value to compare. "
                        f"Fold it to a number first."
                    )
            if (c.left, c.right) in seen or (c.right, c.left) in seen:
                raise Refusal(
                    f"Reader({self.name}): two Compares over {c.left!r} and {c.right!r}"
                )
            seen.add((c.left, c.right))

    def _sources_for(self, question: str) -> tuple[str, ...]:
        """Every source name that could answer `question` in this reader."""
        out: list[str] = []
        for p in self.inputs:
            if question in p.answers:
                out.extend(p.sources_for(question))
        out.extend(f.source for f in self.folds if f.into == question)
        return tuple(sorted(set(out)))

    def _check_sources_and_orders(self) -> None:
        by_question: dict[str, Order] = {}
        for o in self.orders:
            if not isinstance(o, Order):
                raise Refusal(
                    f"Reader({self.name}): orders takes Order objects, got {type(o).__name__}"
                )
            if o.question in by_question:
                raise Refusal(
                    f"Reader({self.name}): two orders for {o.question!r}. One question has "
                    f"one ranking."
                )
            by_question[o.question] = o
            if o.question not in self.questions:
                raise Refusal(
                    f"Order({o.question!r}): not a Question this reader declares"
                )
            if self.questions[o.question].merge is Merge.UNION:
                raise Refusal(
                    f"Order({o.question!r}): {o.question!r} is Merge.UNION — two inputs "
                    f"hold different parts of one fact, so all of them land and declaring "
                    f"an order between them is refused."
                )
            possible = self._sources_for(o.question)
            unknown = sorted(set(o.order) - set(possible))
            if unknown:
                raise Refusal(
                    f"Order({o.question!r}) names {unknown}, which cannot answer it. The "
                    f"sources that can are {list(possible)}."
                )
            missing = sorted(set(possible) - set(o.order))
            if missing:
                raise Refusal(
                    f"Order({o.question!r}) does not mention {missing} — it is not a total "
                    f"order over the sources that can answer it."
                )
        # Every keyed question, UNION and single-input included, must say who may create
        # rows; run grain has no key.
        for qname, q in self.questions.items():
            if DEPTH[q.grain] == 0:
                continue
            answering = sorted(
                p.name for p in self.inputs if qname in p.answers
            )
            if not answering:
                continue
            decl = self.rows_decl.get(qname)
            if decl is None and q.grain in self.rows_by_grain:
                continue
            if decl is None:
                at_grain = sorted(
                    p.name
                    for p in self.inputs
                    if any(self.questions[a].grain is q.grain for a in p.answers)
                )
                raise Refusal(
                    f"Reader({self.name}): {qname!r} is answered by {answering} and "
                    f"nothing says which inputs may CREATE rows at {q.grain.name} "
                    f"grain. An Order picks a winner among sources that answered the same "
                    f"row; it does nothing about rows only one source names, so a summary "
                    f"can invent them, and a stream's Contribute says who contributes, "
                    f"never which keys they may bring. Add to rows=:\n"
                    f"    Rows(question=Grain.{q.grain.name}, "
                    f"may_create=({', '.join(repr(a) for a in at_grain[:1])},), "
                    f'because="...")\n'
                    f"  ...which covers every question at that grain, or name this one "
                    f"question to override it."
                )
            unknown = sorted(set(decl.may_create) - set(answering))
            if unknown:
                raise Refusal(
                    f"Rows({qname!r}): names {unknown}, which cannot answer it. The "
                    f"inputs that can are {answering}."
                )
        for grain, decl in self.rows_by_grain.items():
            at_grain = {
                p.name
                for p in self.inputs
                if any(self.questions[a].grain is grain for a in p.answers if a in self.questions)
            }
            unknown = sorted(set(decl.may_create) - at_grain)
            if unknown:
                raise Refusal(
                    f"Rows(Grain.{grain.name}): names {unknown}, which answer nothing at "
                    f"{grain.name} grain. The inputs that do are {sorted(at_grain)}."
                )

        self.order_by_question = by_question
        for qname, q in self.questions.items():
            if q.merge is not Merge.ONE:
                continue
            possible = self._sources_for(qname)
            if len(possible) > 1 and qname not in self.order_by_question:
                raise Refusal(
                    f"Reader({self.name}): {qname!r} can be answered by {list(possible)} "
                    f"and no order is declared. Declaration order would decide it "
                    f"silently. Add:\n"
                    f"    Order(question={qname!r}, order=("
                    + ", ".join(repr(s) for s in possible)
                    + "), keep_loser=True, because=\"...\")"
                )

    def _check_keys(self) -> None:
        minters: dict[str, list[str]] = {}
        for p in self.inputs:
            for k in p.mints:
                minters.setdefault(k.space, []).append(p.name)
        for space, who in sorted(minters.items()):
            if len(who) > 1:
                raise Refusal(
                    f"Reader({self.name}): key space {space!r} is minted by {sorted(who)}. "
                    f"Exactly one minter per key space: two makes the index "
                    f"order-dependent."
                )
        self.minters = {space: who[0] for space, who in minters.items()}
        for att in self.attachments:
            space = getattr(getattr(att, "on", None), "space", None)
            if space is None:
                continue
            if space not in self.minters:
                raise Refusal(
                    f"Reader({self.name}): {type(att).__name__} joins on key space "
                    f"{space!r}, which no input mints. Without a minter the join "
                    f"silently returns nothing — measured, 0 of 117 matched, with no "
                    f"exception and no empty table. Declare "
                    f"mints=(Key({space!r}, because=...),) on the input that reads it."
                )

    def _check_links(self) -> None:
        """Each Link joins two contributing inputs, one minting its space and one naming its
        word; every word an input names is linked or declared unlinked."""
        linked: set[tuple[str, str]] = set()
        seen: set[tuple] = set()
        for ln in self.links:
            sig = (ln.to, ln.frm_input, ln.to_input, ln.on.space)
            if sig in seen:
                raise Refusal(f"Link({ln.word!r}) joins {sig} twice")
            seen.add(sig)
            for inp in (ln.frm_input, ln.to_input):
                if (ln.to, inp) not in self.contributes:
                    raise Refusal(
                        f"Link({ln.word!r}): {inp!r} contributes no rows to {ln.to!r}"
                    )
            frm, to = self.by_name[ln.frm_input], self.by_name[ln.to_input]
            namer = link_namer(ln, frm, to)
            if namer is None:
                raise Refusal(
                    f"Link({ln.word!r}, {ln.frm_input!r} -> {ln.to_input!r}) on "
                    f"{ln.on.space!r}: one end mints the space (mints=) and the other "
                    f"names the word (names=); here they do not"
                )
            linked.add((namer.name, ln.word))
        for p in self.inputs:
            for k in p.names:
                if (p.name, k.space) not in linked and (p.name, k.space) not in self.unlinked:
                    raise Refusal(
                        f"Reader({self.name}): Input({p.name!r}) names rows by "
                        f"{k.space!r} and nothing says what they reach. Link it, or "
                        f"declare Unlinked({k.space!r}, {p.name!r}, because=...): a "
                        f"reference left undecided joins nothing, silently."
                    )

    def _check_attachments(self) -> None:
        self.contributes: dict[tuple[str, str], Contribute] = {}
        self.links: list[Link] = []
        self.unlinked: dict[tuple[str, str], Unlinked] = {}
        for att in self.attachments:
            if isinstance(att, Contribute):
                q = self._question_or_refuse(att.to, "Contribute.to")
                if q.merge is not Merge.UNION:
                    raise Refusal(
                        f"Contribute(to={att.to!r}): that question is {q.merge.name}. "
                        f"Contributing adds rows to a stream; a question two inputs "
                        f"state the same fact about is Merge.ONE and settled by an Order."
                    )
                p = self.by_name.get(att.input)
                if p is None:
                    raise Refusal(
                        f"Contribute(to={att.to!r}): no input named {att.input!r}"
                    )
                if att.to not in p.answers:
                    raise Refusal(
                        f"Contribute(to={att.to!r}, input={att.input!r}): that "
                        f"input does not answer {att.to!r}"
                    )
                if (att.to, att.input) in self.contributes:
                    raise Refusal(
                        f"Contribute(to={att.to!r}, input={att.input!r}) twice"
                    )
                self.contributes[(att.to, att.input)] = att
            elif isinstance(att, Relate):
                frm = self._question_or_refuse(att.frm, "Relate.frm")
                to = self._question_or_refuse(att.to, "Relate.to")
                if att.same_rollout and (frm.grain is not Grain.ROLLOUT or to.grain is not Grain.ROLLOUT):
                    raise Refusal(
                        f"Relate({att.frm!r} -> {att.to!r}, same_rollout=True): both ends "
                        f"must be ROLLOUT-grain questions; an alias names rollouts, and a row "
                        f"below a rollout is moved with its rollout, not on its own."
                    )
                if DEPTH[to.grain] < DEPTH[frm.grain]:
                    raise Refusal(
                        f"Relate({att.frm!r} -> {att.to!r}): the target is at "
                        f"{to.grain.name} grain, above {frm.grain.name}, so it is not "
                        f"settled when this edge is resolved. The merge walks lowest first."
                    )
                # `frm` did not mint these ids, so each asserts existence; Card.ANY hides a miss.
                minter = self.minters.get(att.on.space)
                answering = {p.name for p in self.inputs if att.frm in p.answers}
                if att.cardinality is Card.ANY and minter not in answering:
                    raise Refusal(
                        f"Relate({att.frm!r} -> {att.to!r}) joins on key space "
                        f"{att.on.space!r}, which {att.frm!r} does not mint — "
                        f"{minter!r} does. Every {att.frm!r} row carrying that key is the "
                        f"producer saying the thing it names exists, so Card.ANY is not a "
                        f"weaker claim, it is no claim: a join that resolves nothing "
                        f"passes it, forever, with `ok` True. Declare Card.ONE or "
                        f"Card.AT_LEAST_ONE. Card.ANY is for the other shape, where this "
                        f"side mints the id and the other end may honestly not exist."
                    )
            elif isinstance(att, Link):
                q = self._question_or_refuse(att.to, "Link.to")
                if q.merge is not Merge.UNION:
                    raise Refusal(
                        f"Link({att.word!r}, to={att.to!r}): that question is "
                        f"{q.merge.name}; a link joins rows of one stream"
                    )
                self.links.append(att)
            elif isinstance(att, Unlinked):
                p = self.by_name.get(att.input)
                if p is None or att.word not in {k.space for k in p.names}:
                    raise Refusal(
                        f"Unlinked({att.word!r}, {att.input!r}): that input does not "
                        f"declare names=(Key({att.word!r}, ...),)"
                    )
                self.unlinked[(att.input, att.word)] = att
            else:
                raise Refusal(
                    f"Reader({self.name}): `attach` takes Contribute, Relate, Link or "
                    f"Unlinked, got {type(att).__name__}"
                )
        self._check_links()
        for qname, q in self.questions.items():
            if q.merge is not Merge.UNION:
                continue
            for p in self.inputs:
                if qname in p.answers and (qname, p.name) not in self.contributes:
                    raise Refusal(
                        f"Input({p.name!r}) answers {qname!r}, which is Merge.UNION, and "
                        f"declares no Contribute. A stream needs the column this input's "
                        f"own rows carry their identity in and the one they are ordered by; "
                        f"without them its rows cannot be interleaved. Add:\n"
                        f"    Contribute(to={qname!r}, input={p.name!r}, order_by=..., "
                        f"identity=..., because=\"...\")"
                    )

    def _question_or_refuse(self, name: str, what: str) -> Question:
        q = self.questions.get(name)
        if q is None:
            raise Refusal(f"{what}={name!r}: not a Question this reader declares")
        return q

    def mount(self, *mounts: Any) -> Reader:
        """This reader with every `Mount` placed in it, as one reader under this name."""
        from .mount import mount

        return mount(self, *mounts)

    def lane_for(self, kind: str, input_name: str | None) -> Lane | None:
        """The Lane a row of `kind` follows: its own producer's, else the reader's."""
        return self.lane_by_input.get((input_name, kind)) or self.lane_kinds.get(kind)

    # -- reading -----------------------------------------------------------

    def detect(self, tree: Tree | str | os.PathLike[str]) -> tuple[bool, list[str]]:
        """Whether every landmark resolves in its declared number; returns (ok, complaints)."""
        idx = Index(_as_tree(tree))
        res = resolve_landmarks(idx, self.landmarks)
        return res.ok, [c[2] for c in res.complaints]

    def locate(self, landmark_name: str, tree: Tree | str | os.PathLike[str]) -> list[str]:
        """Tree-relative paths where `landmark_name` resolved; empty when undetected.

        Each path is the directory the landmark matched, relative to the tree root
        (``""`` means the root itself).  Use this to find data at the correct depth
        rather than re-deriving it from the filesystem.
        """
        idx = Index(_as_tree(tree))
        res = resolve_landmarks(idx, self.landmarks)
        return [s.path for s in res.sites.get(landmark_name, [])]

    def read(self, tree: Tree | str | os.PathLike[str]) -> ReadResult:
        """Read a folder (a path or a `Tree`): rows, losers, edges and findings."""
        t = _as_tree(tree)
        idx = Index(t)
        out = ReadResult(reader=self.name, tree=getattr(t, "name", str(tree)),
                         counts=self.counts, producer_of=dict(self.producer_of),
                         declared_stack=self.declared_stack)

        res = resolve_landmarks(idx, self.landmarks)
        out.sites = dict(res.sites)
        for lm_name, where, err in res.complaints:
            out.findings.append(Finding(Kind.MISSING, f"landmark:{lm_name}", where, err))
        if not res.ok:
            out.claimed = False
            out.findings = sort_findings(out.findings)
            return out
        report_unchecked(self, res.sites, out)

        answers = alias_rollouts(self, run_inputs(self, idx, res.sites, out), out, res.sites)

        by_grain: dict[Grain, list[Given]] = {g: [] for g in LOWEST_FIRST}
        for g in answers:
            by_grain[self.questions[g.question].grain].append(g)

        for grain in LOWEST_FIRST:
            settle(self, grain, by_grain[grain], out)
            attach(self, grain, answers, out)
            assign_positions(self, grain, out)
            fold(self, grain, by_grain, out)

        check_landed_columns(self, out)
        compare(self, out)
        check_mints(self, answers, out)
        account_for_files(self, idx, res.sites, out)
        record_firing(self, out)
        out.findings = sort_findings(out.findings)
        return out


def _as_tree(tree: Tree | str | os.PathLike[str]) -> Tree:
    if isinstance(tree, (str, os.PathLike)):
        return LocalTree(tree)
    return tree
