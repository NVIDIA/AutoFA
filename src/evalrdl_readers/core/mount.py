# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Mount`: one reader's tree placed inside another's, declared at the seam.

A mounted reader is written as if it read a folder of its own. `mount(host, Mount(...))`
places it at a site of the host and returns one flat `Reader`; neither reader names the
other, and everything that depends on both lives on the `Mount`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from .declare import At, Ignore, Join, Order, Rows, _Expectation
from .errors import Refusal, need_reason
from .input import Input
from .landmarks import ROOT, Landmark
from .questions import DEPTH, Grain, Merge, Question
from .reader import Reader

MOUNT_ORDER = "the host's sources rank above the mounted reader's, the Mount's default"
WRAP_ORDER = "the mounted reader wraps the host, so its sources rank first, the Mount's default"


@dataclass(frozen=True)
class Mount:
    """`reader`'s root landmark, placed at `where` under the host's `within`.

    `rollout`: a host key space whose value is the mount site's `row`; every rollout of the
    mounted reader then IS the host rollout that carries it. `drop_run`: the mounted
    reader's run-grain inputs are not read here, for this reason; required where a host
    holds several mount sites, since one run cannot be answered from each. `wraps`: the
    mounted reader is the outer producer, so its sources rank first. `inputs`, `orders`,
    `rows`, `attach`, `ignored`, `questions`: declarations that need both readers, and so
    belong to neither.
    """

    reader: Reader
    """The mounted reader, written as if it read a folder of its own (root `where=("",)`)."""
    within: Landmark
    """The host landmark it is placed under."""
    where: tuple[str, ...]
    """Where under `within` its root sits, as a landmark's `where`."""
    expect: _Expectation
    """How many mount sites per `within` site, and why."""
    because: str
    """Where in the folder it shows that one producer ran inside the other."""
    rollout: Join | None = None
    """A host key space whose value is the mount site's `row`: each mounted rollout is that
    host rollout."""
    drop_run: str | None = None
    """Why the mounted reader's run-grain inputs are not read here; required where a host
    holds several mount sites."""
    wraps: bool = False
    """The mounted reader is the outer producer, so its sources rank first."""
    inputs: tuple[Input, ...] | tuple[tuple[str, tuple[Input, ...]], ...] = ()
    """Seam inputs, that read a part only meaningful with both producers' knowledge;
    grouped by producer as on `Reader(inputs=)`."""
    orders: tuple[Order, ...] = ()
    """Seam orders; one replaces the merged order for its question."""
    rows: tuple[Rows, ...] = ()
    """Seam `Rows`; one replaces the merged one for its question."""
    attach: tuple = ()
    """Seam attachments: `Contribute`, `Link`, `Relate`, `Unlinked`."""
    ignored: tuple[Ignore, ...] = ()
    """Seam ignores."""
    questions: tuple[Question, ...] = ()
    """Questions only the seam answers."""

    def __post_init__(self) -> None:
        if not isinstance(self.reader, Reader):
            raise Refusal(f"Mount: reader is a Reader, got {type(self.reader).__name__}")
        what = f"Mount({self.reader.name})"
        if self.reader.mounts:
            raise Refusal(
                f"{what}: the reader already mounts others. Mount each reader into the "
                f"host directly, so every seam is declared once and at one level."
            )
        if not isinstance(self.within, Landmark):
            raise Refusal(f"{what}.within: the host Landmark object, got {self.within!r}")
        object.__setattr__(self, "where", tuple(self.where))
        if not self.where:
            raise Refusal(f"{what}.where: no locations declared")
        if not isinstance(self.expect, _Expectation):
            raise Refusal(
                f"{what}: expect is not declared. How many mount sites there are per "
                f"`within` site is the host's fact: expect=Card.ANY(because=...)."
            )
        need_reason(
            self.because,
            what,
            "A mount is a claim that one producer ran inside "
            "another; the reason says where that is visible.",
        )
        if self.rollout is not None and not isinstance(self.rollout, Join):
            raise Refusal(f"{what}.rollout: a Join naming a host key space")
        if self.drop_run is not None:
            need_reason(
                self.drop_run,
                f"{what}.drop_run",
                "Dropping the run-grain inputs hides their files.",
            )
        for name in ("inputs", "orders", "rows", "attach", "ignored", "questions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class Placed:
    """A mount as expanded into the host: the placed root landmark and the input names."""

    mount: Mount
    root: Landmark
    inputs: frozenset[str] = field(default_factory=frozenset)


def mount(host: Reader, *mounts: Mount) -> Reader:
    """`host` with every mount placed in it, as one `Reader` under the host's name."""
    if host.mounts:
        raise Refusal(f"{host.name}.mount(): already mounted; pass every Mount in one call")
    if not mounts:
        raise Refusal(f"{host.name}.mount(): no Mount given")
    acc = _Decls.of(host)
    placed: list[Placed] = []
    seen: dict[int, Landmark] = {}
    for m in mounts:
        if not any(m.within is lm for lm in (*acc.landmarks, ROOT)):
            raise Refusal(
                f"Mount({m.reader.name}).within is Landmark({m.within.name}), which the "
                f"host does not declare"
            )
        inner, root = _placed(m, seen)
        names = frozenset(p.name for _, ins in inner.groups for p in ins)
        acc = acc.join(inner, m)
        placed.append(Placed(m, root, names))
    out = acc.build(host)
    out.mounts = tuple(placed)
    for p in placed:
        if p.mount.rollout is not None and p.mount.rollout.space not in out.minters:
            raise Refusal(
                f"Mount({p.mount.reader.name}).rollout joins on {p.mount.rollout.space!r}, "
                f"which no host input mints. The host names its own rollouts; declare "
                f"mints=(Key({p.mount.rollout.space!r}, because=...),) on the input that "
                f"carries the key."
            )
    return out


# ---------------------------------------------------------------------- placing


def _placed(m: Mount, seen: dict[int, Landmark]) -> tuple[_Decls, Landmark]:
    """The mounted reader's declarations with its landmarks re-rooted at the mount.

    `seen` maps every landmark placed by an earlier mount to its placed copy, so the seam's
    own inputs and ignores may address any reader mounted before it; it is extended here.
    """
    r = m.reader
    tops = [lm for lm in r.landmarks if lm.within is None or lm.within is ROOT]
    if len(tops) != 1:
        raise Refusal(
            f"Mount({r.name}): the reader has {len(tops)} top-level landmarks "
            f"{[lm.name for lm in tops]}. A mounted reader has one root, the directory the "
            f"mount places; nest the others `within` it."
        )
    top = tops[0]
    if top.where != ("",):
        raise Refusal(
            f"Mount({r.name}): its root Landmark({top.name}) looks in {top.where}. A "
            f'mountable reader\'s root is the folder itself, where=("",); where it sits '
            f"in a host is the Mount's `where`."
        )
    root = replace(
        top, within=None if m.within is ROOT else m.within, where=m.where, expect=m.expect
    )
    remap: dict[int, Landmark] = {id(top): root, id(ROOT): root}
    for lm in r.landmarks:
        if lm is top:
            continue
        chain, node = [], lm
        while id(node) not in remap:
            chain.append(node)
            node = node.within
        for n in reversed(chain):
            remap[id(n)] = replace(n, within=remap[id(n.within)])
    landmarks = tuple(remap[id(lm)] for lm in r.landmarks)
    own = dict(remap)
    del own[id(ROOT)]
    seen.update(own)

    def placer(table: dict[int, Landmark]):
        def at(a: At) -> At:
            return replace(a, landmark=table[id(a.landmark)]) if id(a.landmark) in table else a

        def moved(p: Input) -> Input:
            return replace(p, reads=tuple(at(a) for a in p.reads))

        def ignore(ig: Ignore) -> Ignore:
            return replace(ig, landmark=table[id(ig.landmark)]) if id(ig.landmark) in table else ig

        return moved, ignore

    # The reader's own ROOT is the mount site; the seam's ROOT stays the host's.
    moved, ignore = placer(remap)
    seam_moved, seam_ignore = placer(dict(seen))

    groups = _groups(r)
    dropped: list[Ignore] = []
    kept: list[tuple[str, tuple[Input, ...]]] = []
    run_q = {q for q, spec in r.questions.items() if DEPTH[spec.grain] == 0}
    for producer, ins in groups:
        keep = []
        for p in ins:
            at_run = [q for q in p.answers if q in run_q]
            if m.drop_run is not None and at_run:
                if len(at_run) != len(p.answers):
                    raise Refusal(
                        f"Mount({r.name}).drop_run: Input({p.name!r}) answers run-grain "
                        f"{sorted(at_run)} and also "
                        f"{sorted(set(p.answers) - set(at_run))}. Split it, so dropping "
                        f"the run's facts does not drop a rollout's."
                    )
                dropped += [
                    Ignore(remap.get(id(a.landmark), a.landmark), a.pattern, because=m.drop_run)
                    for a in p.reads
                ]
                continue
            keep.append(moved(p))
        if keep:
            kept.append((producer, tuple(keep)))
    gone = {p.name for _, ins in groups for p in ins} - {p.name for _, ins in kept for p in ins}
    extra = _groups_of(m.inputs, f"Mount({r.name}).inputs")
    inner = _Decls(
        landmarks=landmarks,
        groups=tuple(kept) + tuple((pr, tuple(seam_moved(p) for p in ins)) for pr, ins in extra),
        grouped=bool(r.input_groups),
        questions=_questions(
            tuple(
                q
                for q in r.question_list
                if not (
                    gone
                    and q.name in run_q
                    and not _answered(q.name, kept, extra)
                    and not any(f.into == q.name for f in r.folds)
                )
            ),
            m.questions,
            r.name,
        ),
        orders=tuple(_drop_sources(o, gone) for o in r.orders),
        rows=tuple(_drop_rows(rw, gone) for rw in r.rows),
        folds=r.folds,
        compare=r.compares,
        ignored=tuple(ignore(ig) for ig in r.ignored)
        + tuple(seam_ignore(ig) for ig in m.ignored)
        + tuple(dropped),
        attach=r.attachments + m.attach,
        lanes=r.lanes,
    )
    return inner, root


def _answered(q: str, *group_sets) -> bool:
    return any(q in p.answers for groups in group_sets for _, ins in groups for p in ins)


def _drop_sources(o: Order, gone: set[str]) -> Order | None:
    if not gone:
        return o
    left = tuple(s for s in o.order if s.split("/", 1)[0] not in gone)
    return replace(o, order=left) if len(left) > 1 else None


def _drop_rows(rw: Rows, gone: set[str]) -> Rows | None:
    if not gone:
        return rw
    left = tuple(s for s in rw.may_create if s not in gone)
    return replace(rw, may_create=left) if left else None


def _groups(r: Reader) -> tuple[tuple[str, tuple[Input, ...]], ...]:
    return r.input_groups or (("", r.inputs),)


def _groups_of(items, what: str) -> tuple[tuple[str, tuple[Input, ...]], ...]:
    out = []
    for item in items:
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
            raise Refusal(f"{what}: (producer, (Input, ...)) groups, got {item!r}")
        out.append((item[0], tuple(item[1])))
    return tuple(out)


# ---------------------------------------------------------------------- merging


@dataclass(frozen=True)
class _Decls:
    landmarks: tuple
    groups: tuple
    grouped: bool
    questions: tuple
    orders: tuple
    rows: tuple
    folds: tuple
    compare: tuple
    ignored: tuple
    attach: tuple
    lanes: tuple
    lane_by_input: tuple = ()

    @classmethod
    def of(cls, r: Reader) -> _Decls:
        return cls(
            r.landmarks,
            _groups(r),
            bool(r.input_groups),
            r.question_list,
            r.orders,
            r.rows,
            r.folds,
            r.compares,
            r.ignored,
            r.attachments,
            r.lanes,
        )

    def join(self, inner: _Decls, m: Mount) -> _Decls:
        name = m.reader.name
        if self.grouped != inner.grouped:
            raise Refusal(
                f"Mount({name}): one reader groups its inputs by producer and the other "
                f"does not, so the mounted inputs would have no place in the stack"
            )
        taken = {lm.name for lm in self.landmarks}
        clash = sorted(taken & {lm.name for lm in inner.landmarks})
        if clash:
            raise Refusal(f"Mount({name}): landmarks {clash} are declared by both readers")
        outer, lower = (inner, self) if m.wraps else (self, inner)
        return _Decls(
            landmarks=self.landmarks + inner.landmarks,
            groups=_stack(outer.groups, lower.groups),
            grouped=self.grouped,
            questions=_questions(self.questions, inner.questions, name),
            orders=(),
            rows=(),
            folds=_unique(self.folds, inner.folds, lambda f: (f.into, f.over)),
            compare=_unique(self.compare, inner.compare, lambda c: frozenset((c.left, c.right))),
            ignored=self.ignored + inner.ignored,
            attach=self.attach + inner.attach,
            lanes=_lanes(self.lanes, inner.lanes, name),
            lane_by_input=self.lane_by_input + _own_lanes(self.lanes, inner, name),
        )._ranked(self, inner, outer, m)

    def _ranked(self, host: _Decls, inner: _Decls, outer: _Decls, m: Mount) -> _Decls:
        """Orders and Rows over both sides, then the Mount's own on top."""
        qs = {q.name: q for q in self.questions}
        lower = host if outer is inner else inner
        orders: dict[str, Order] = {}
        for qname, q in qs.items():
            if q.merge is not Merge.ONE:
                continue
            first, second = _ranking(outer, qname), _ranking(lower, qname)
            if not first and not second:
                continue
            if first and second:
                ranked = first[0] + tuple(x for x in second[0] if x not in first[0])
                if len(ranked) > len(first[0]):
                    orders[qname] = Order(
                        question=qname,
                        order=ranked,
                        keep_loser=True,
                        because=WRAP_ORDER if m.wraps else MOUNT_ORDER,
                    )
                elif first[1] is not None:
                    orders[qname] = first[1]
            elif len((first or second)[0]) > 1:
                orders[qname] = (first or second)[1]
        for o in m.orders:
            orders[o.question] = o
        rows: dict = {}
        keys = {rw.question for side in (host, inner) for rw in side.rows if rw is not None}
        for key in keys:
            names: list[str] = []
            because = ""
            for side in (host, inner):
                got = _may_create(side, key, qs)
                names += [n for n in got[0] if n not in names]
                because = because or got[1]
            if names:
                rows[key] = Rows(question=key, may_create=tuple(names), because=because)
        for rw in m.rows:
            rows[rw.question] = rw
        return replace(self, orders=tuple(orders.values()), rows=tuple(rows.values()))

    def build(self, host: Reader) -> Reader:
        groups = self.groups if self.grouped else ()
        flat = tuple(p for _, ins in self.groups for p in ins)
        out = Reader(
            name=host.name,
            schema=host.schema,
            counts=host.counts,
            landmarks=self.landmarks,
            inputs=groups if groups else flat,
            questions=self.questions,
            orders=tuple(o for o in self.orders if o),
            rows=tuple(r for r in self.rows if r),
            folds=self.folds,
            compare=self.compare,
            ignored=self.ignored,
            attach=self.attach,
            lanes=self.lanes,
        )
        out.lane_by_input = dict(self.lane_by_input)
        return out


def _stack(outer, lower) -> tuple:
    """Producer groups, outer first; one producer named on both sides is one group."""
    out: dict[str, list] = {}
    for producer, ins in (*outer, *lower):
        out.setdefault(producer, []).extend(ins)
    return tuple((p, tuple(ins)) for p, ins in out.items())


def _questions(a: Iterable, b: Iterable, name: str) -> tuple:
    by = {q.name: q for q in a}
    for q in b:
        mine = by.get(q.name)
        if mine is None:
            by[q.name] = q
        elif mine is not q and (mine.grain, mine.merge, mine.lands) != (q.grain, q.merge, q.lands):
            raise Refusal(
                f"Mount({name}): Question({q.name!r}) is declared by both readers with "
                f"different grain, merge or landing. One name is one fact."
            )
    return tuple(by.values())


def _unique(a: tuple, b: tuple, key) -> tuple:
    seen = {key(x) for x in a}
    return a + tuple(x for x in b if key(x) not in seen)


def _lanes(a: tuple, b: tuple, name: str) -> tuple:
    by = {ln.kind: ln for ln in a}
    out = list(a)
    for ln in b:
        mine = by.get(ln.kind)
        if mine is None:
            out.append(ln)
        elif (mine.identity, mine.actor) != (ln.identity, ln.actor):
            raise Refusal(
                f"Mount({name}): both readers declare a {ln.kind!r} lane with a different "
                f"identity or actor. One kind names its rows and their author one way; only "
                f"the order of a producer's own sequence may be its own."
            )
    return tuple(out)


def _own_lanes(a: tuple, inner: _Decls, name: str) -> tuple:
    """(input, kind) -> Lane for each lane the mounted reader orders differently from the
    host: a lane instance is one producer's sequence, so its rows keep their writer's order."""
    by = {ln.kind: ln for ln in a}
    differ = [ln for ln in inner.lanes if ln.kind in by and by[ln.kind] != ln]
    return tuple(((p.name, ln.kind), ln) for ln in differ for _, ins in inner.groups for p in ins)


def _sources(d: _Decls, qname: str) -> tuple[str, ...]:
    out: list[str] = []
    for _, ins in d.groups:
        for p in ins:
            if qname in p.answers:
                out.extend(p.sources_for(qname))
    out.extend(f.source for f in d.folds if f.into == qname)
    return tuple(sorted(set(out)))


def _ranking(d: _Decls, qname: str) -> tuple[tuple[str, ...], Order | None] | None:
    """One side's ranked sources for `qname`, and its own Order if it declared one."""
    possible = _sources(d, qname)
    if not possible:
        return None
    own = next((o for o in d.orders if o is not None and o.question == qname), None)
    if own is not None:
        return tuple(s for s in own.order if s in possible), own
    return possible, None


def _may_create(d: _Decls, key, qs: dict) -> tuple[tuple[str, ...], str]:
    rows = {rw.question: rw for rw in d.rows if rw is not None}
    if key in rows:
        return rows[key].may_create, rows[key].because
    if isinstance(key, Grain):
        return (), ""
    q = qs.get(key)
    grain_rows = rows.get(q.grain) if q is not None else None
    if grain_rows is None:
        return (), ""
    answering = {p.name for _, ins in d.groups for p in ins if key in p.answers}
    return tuple(n for n in grain_rows.may_create if n in answering), grain_rows.because
