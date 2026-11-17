# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 2 of a read: every input, at every site of its landmark.

A translator's output is checked here, answer by answer, and stamped with the provenance it
cannot forge: its input, its decoder, its site and the file it was parsing. A crash in a
translator becomes a finding about that site; a crash in this repo's code is a `Refusal`.
"""

from __future__ import annotations

import dataclasses
import marshal
import os
import pickle
import threading
import time
from collections import Counter
from collections.abc import Hashable, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import TYPE_CHECKING

from . import procs
from .declare import Card
from .errors import Refusal
from .findings import Finding, Kind
from .input import Answer, Doc, Given, Input, Sources, Unreadable, _NoDoc
from .landmarks import Site
from .questions import DEPTH, Grain, Question
from .result import ReadResult
from .shared import SharedReads
from .tree import Index

if TYPE_CHECKING:
    from .reader import Reader


#: Exceptions that only this repo's code raises, never a producer's bytes.
_CODE_DEFECTS = (NameError, ImportError, SyntaxError, RecursionError)


def _is_code_defect(exc: BaseException) -> bool:
    """Whether a translator failed on this repo's code rather than on the bytes.

    `AttributeError` counts only when raised on a module; on data it is a producer shape.
    """
    if isinstance(exc, _CODE_DEFECTS):
        return True
    if isinstance(exc, AttributeError):
        return isinstance(getattr(exc, "obj", None), ModuleType)
    return False


def _guarded(prov, src, site, out):
    """Turn a translator crash into a finding about this site, not the end of the read.

    `Refusal` and code defects propagate; `_NoDoc` becomes a MISSING finding.
    """
    it = prov.via(src)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Refusal:
            raise
        except _NoDoc as gone:
            out.findings.append(
                Finding(Kind.MISSING, ",".join(sorted(prov.answers)), site.path,
                        f"{gone} This site yields nothing and the read goes on; the "
                        f"counts this input declares are checked against that nothing. "
                        f"If the file is honestly optional here, reach it with "
                        f"first({gone.i}) or has({gone.i}) and say so.", prov.name)
            )
            return
        except Exception as exc:
            if _is_code_defect(exc):
                raise Refusal(
                    f"Input({prov.name!r}) raised {type(exc).__name__}: {exc}. That is a "
                    f"name this repo failed to resolve, not a fact about the producer's "
                    f"bytes, so it is refused rather than recorded as an unreadable "
                    f"record. Reading on would report a code defect as missing data."
                ) from exc
            out.findings.append(
                Finding(Kind.UNREADABLE, ",".join(sorted(prov.answers)), site.path,
                        f"the translator raised {type(exc).__name__}: {exc}", prov.name)
            )
            return


def report_unchecked(reader: Reader, sites: dict[str, list[Site]], out: ReadResult) -> None:
    """Report declared counts skipped because their landmark resolved to no site."""
    demanded = (Card.ONE, Card.AT_LEAST_ONE, Card.NONE)
    for lm in reader.landmarks:
        if sites.get(lm.name):
            continue
        skipped: list[str] = []
        for p in reader.inputs:
            if p.landmark.name != lm.name:
                continue
            skipped += [
                f"{p.name}.{q}={e.card.name}"
                for q, e in sorted(p.answers.items())
                if e.card in demanded
            ]
        if not skipped:
            continue
        out.findings.append(
            Finding(
                Kind.UNCHECKED,
                f"landmark:{lm.name}",
                lm.name,
                f"resolved to no site here, so {len(skipped)} declared count(s) "
                f"under it were never evaluated: {', '.join(sorted(skipped))}. The "
                f"landmark says this may happen — {lm.expect.because} — so this is "
                f"not a loss; it is the list of things this read is NOT claiming "
                f"about this folder.",
                lm.name,
            )
        )


def run_inputs(
    reader: Reader, idx: Index, sites: dict[str, list[Site]], out: ReadResult
) -> list[Given]:
    """Every answer every input gave, in input order and then site order.

    Each record of a file several inputs read is parsed once and handed to the others; where
    the file outgrows what is kept, those inputs run side by side, so it is also read once.
    Answers are joined in input order whatever ran first.
    """
    # Every match is known before any translator runs, so a file several Docs will open
    # is read and parsed once.
    plan = [
        (prov, [(site, [idx.files_at(site.path, at.segments) for at in prov.reads])
                for site in sites.get(prov.landmark.name, [])])
        for prov in reader.inputs
    ]
    shared = SharedReads(Counter(
        e.path for _prov, at_sites in plan for _site, hits_by_at in at_sites
        for hits in hits_by_at for e in hits
    ))
    lock = threading.Lock()
    results: list[list[Given] | None] = [None] * len(plan)
    groups = _sharing_groups(plan, shared)
    for group in groups:
        if len(group) > 1 and _in_processes(group, plan, shared):
            parts = _run_in_processes(reader, idx, plan, group, out, shared)
            if parts is not None:
                for k, given in zip(group, parts, strict=False):
                    results[k] = given
                continue
        parts = _run_group(reader, idx, plan, group, shared, lock, out)
        for k, given in zip(group, parts, strict=False):
            results[k] = given
    out.line_starts.update(shared.line_starts)
    return [g for r in results for g in (r or ())]


def _run_group(reader: Reader, idx: Index, plan, group: list[int], shared: SharedReads,
               lock: threading.Lock, out: ReadResult) -> list[list[Given]]:
    """The answers of each input of `group`, in its order: one after another, or side by side
    where their shared files outgrow what is kept."""
    if len(group) == 1 or not (THREADS or _outgrows_budget(group, plan, shared)):
        return [_run_input(reader, idx, plan[k], shared, lock, out) for k in group]

    def run(k: int) -> list[Given]:
        try:
            return _run_input(reader, idx, plan[k], shared, lock, out)
        finally:
            shared.leave()

    shared.side_by_side(len(group))
    with ThreadPoolExecutor(max_workers=len(group)) as pool:
        futures = [pool.submit(run, k) for k in group]
        return [f.result() for f in futures]


def _outgrows_budget(group: list[int], plan, shared: SharedReads) -> bool:
    """Whether the files the group shares may not fit what is kept: one after another, each
    input would read and parse again what the first could not keep."""
    sizes = {e.path: e.size or 0
             for k in group for _site, hits_by_at in plan[k][1]
             for hits in hits_by_at for e in hits if shared.shares(e.path)}
    return sum(sizes.values()) > shared.budget // 2


#: Run each input of a group whose shared files reach PROCESS_MIN_BYTES in a process of its own,
#: this many at once, so the group's Python runs on as many cores. 0 or 1: threads, as before.
PROCESSES = int(os.environ.get("EVALRDL_READ_PROCESSES", "0") or 0)
PROCESS_MIN_BYTES = int(os.environ.get("EVALRDL_READ_PROCESS_MIN_BYTES", str(1 << 30)) or 0)

#: Set once a child reports it came from a multithreaded forkserver: this process reads in
#: threads from then on, since the server it would start children from stays the same.
_REFUSED = False


def _in_processes(group: list[int], plan, shared: SharedReads) -> bool:
    if PROCESSES < 2 or _REFUSED or not procs.available():
        return False
    sizes = {e.path: e.size or 0
             for k in group for _site, hits_by_at in plan[k][1]
             for hits in hits_by_at for e in hits if shared.shares(e.path)}
    return sum(sizes.values()) >= PROCESS_MIN_BYTES


class _TreeOnly:
    """What a `Doc` reaches of the `Index`: the tree. A child needs no index of its own."""

    __slots__ = ("tree",)

    def __init__(self, tree):
        self.tree = tree


def _process_work(reader: Reader, idx: Index, plan,
                  parts_of: list[list[int]]) -> tuple[tuple[str, str], list[bytes]] | None:
    """Each subset's work, pickled: where to import the reader from, the tree, and the subset's
    own plan entries by input position. None when the reader is no module's attribute or the
    tree does not pickle; those groups read in threads."""
    ref = procs.locate(reader)
    if ref is None:
        return None
    try:
        return ref, [pickle.dumps((ref, reader.name, idx.tree,
                              [(k, plan[k][0].name, plan[k][1]) for k in ks]),
                             protocol=pickle.HIGHEST_PROTOCOL)
                for ks in parts_of]
    except (pickle.PicklingError, TypeError, AttributeError):
        return None


def _run_part(work: bytes):
    """Some inputs of a group in a child process: their answers and what they added.

    They share one parse of their files among themselves, as a group does in one process, and
    nothing with the other processes.
    """
    ref, name, tree, items = pickle.loads(work)
    reader = procs.load(ref)
    if reader.name != name:
        raise Refusal(f"{ref[0]}.{ref[1]} is reader {reader.name!r} in a child, not {name!r}")
    plan = {}
    for k, input_name, at_sites in items:
        prov = reader.inputs[k]
        if prov.name != input_name:
            raise Refusal(f"input {k} of {name!r} is {prov.name!r} in a child, not {input_name!r}")
        plan[k] = (prov, at_sites)
    ks = list(plan)
    out = ReadResult(reader=name, tree=getattr(tree, "name", ""))
    own = SharedReads(Counter(
        e.path for k in ks for _site, hits_by_at in plan[k][1] for hits in hits_by_at
        for e in hits
    ))
    given = _run_group(reader, _TreeOnly(tree), plan, ks, own, threading.Lock(), out)
    return (_pack(given), out.findings, +out.fired, +out.input_seconds, out.read_paths,
            own.line_starts)


_GIVEN_FIELDS = tuple(f.name for f in dataclasses.fields(Given))


def _pack(given: list[list[Given]]):
    """Answers as one `marshal` blob of field tuples: answers are JSON-shaped, and marshal
    writes and reads them several times faster than pickle. Pickled when a value is not."""
    try:
        return marshal.dumps([[tuple(getattr(g, f) for f in _GIVEN_FIELDS) for g in part]
                              for part in given])
    except ValueError:
        return given


def _unpack(packed) -> list[list[Given]]:
    if not isinstance(packed, bytes):
        return packed
    return [[Given(*fields) for fields in part] for part in marshal.loads(packed)]


def _run_in_processes(reader: Reader, idx: Index, plan, group: list[int], out: ReadResult,
                      shared: SharedReads) -> list[list[Given]] | None:
    """The inputs of `group` dealt over child processes; merged in input order, as one after
    another would add them, with answer strings made this read's one copy again. None when
    the group cannot go to processes and reads in threads instead."""
    global _REFUSED
    n = min(PROCESSES, len(group))
    parts_of = [group[i::n] for i in range(n)]
    work = _process_work(reader, idx, plan, parts_of)
    if work is None:
        return None
    ref, work = work
    try:
        parts = procs.run_parts(_run_part, work, n, preload=[ref[0]])
    except procs.ForkRefused as e:
        _REFUSED = True
        procs.refused(None, f"{e}; reading in threads")
        return None
    except procs.ChildLost as e:
        # Most often killed for memory: threads hold one parse of the shared files, not one each.
        procs.log.warning("%s; reading the group in threads", e,
                          extra={"event": "read_child_lost", "reader": reader.name})
        return None
    by_input: dict[int, list[Given]] = {}
    for i, ks in enumerate(parts_of):
        packed, findings, fired, seconds, paths, starts = parts[i]
        parts[i] = None
        given = _unpack(packed)
        del packed
        out.line_starts.update(starts)
        out.findings.extend(findings)
        out.fired.update(fired)
        out.input_seconds.update(seconds)
        out.read_paths |= paths
        for k, part in zip(ks, given, strict=False):
            kept = []
            for g in part:
                value = shared.one_copy(g.value)
                kept.append(g if value is g.value else dataclasses.replace(g, value=value))
            by_input[k] = kept
    return [by_input[k] for k in group]


#: Run every group of inputs that share a file side by side. Off (the default), only a group
#: whose shared files outgrow the kept-record budget does, so each line is read once.
THREADS = os.environ.get("EVALRDL_READ_THREADS", "0").strip().lower() in ("1", "true", "yes")


def _sharing_groups(plan, shared: SharedReads) -> list[list[int]]:
    """Inputs joined by a file they both read, in plan order; the rest alone."""
    parent = list(range(len(plan)))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first_reader: dict[str, int] = {}
    for k, (_prov, at_sites) in enumerate(plan):
        for _site, hits_by_at in at_sites:
            for hits in hits_by_at:
                for e in hits:
                    if not shared.shares(e.path):
                        continue
                    j = first_reader.setdefault(e.path, k)
                    parent[find(k)] = find(j)
    groups: dict[int, list[int]] = {}
    for k in range(len(plan)):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=lambda g: g[0])


def _run_input(reader: Reader, idx: Index, item, shared: SharedReads, lock: threading.Lock,
               out: ReadResult) -> list[Given]:
    """One input at every site of its landmark."""
    prov, at_sites = item
    answers: list[Given] = []
    patterns = tuple(repr(at) for at in prov.reads)
    counters: dict[tuple[str, str], int] = {}
    for site, hits_by_at in at_sites:
        # Last-parsed file; stamped on answers as their location.
        cursor: dict = {"path": None, "line": None}
        by_at: list[list[Doc]] = []
        for at, hits in zip(prov.reads, hits_by_at, strict=False):
            if hits:
                with lock:
                    out.fired[f"read:{prov.name}@{at.landmark.name}/{at.pattern}"] += len(hits)
            # A path counts as read when its Doc is opened, not when matched.
            by_at.append([Doc(idx, e, site, out.read_paths, cursor, shared)
                          for e in hits])
        src = Sources(site, by_at, prov.name, patterns, prov.decoders, cursor)
        t0 = time.perf_counter()
        given, said = _translate(reader, prov, src, site, counters, out, shared)
        with lock:
            out.input_seconds[prov.name] += time.perf_counter() - t0
            if given or said:
                out.fired[f"input:{prov.name}"] += len(given) + said
        _check_parse_errors(prov, src, site, said, out)
        for docs in by_at:
            for doc in docs:
                shared.done(doc.path, id(doc))
        answers.extend(given)
        _check_value_cardinality(reader, prov, site, given, out)
    return answers


def _check_parse_errors(prov: Input, src: Sources, site: Site, said: int, out: ReadResult) -> None:
    """Flag parse errors the translator did not report as `Unreadable`."""
    bad = [(d, line) for d in src.all for line in d.bad_lines]
    if len(bad) <= said:
        return
    where: dict[str, list[int]] = {}
    for d, line in bad:
        where.setdefault(d.relpath, []).append(line)
    listed = "; ".join(
        f"{name} lines {sorted(lines)}" for name, lines in sorted(where.items())
    )
    out.findings.append(
        Finding(
            Kind.UNREADABLE,
            ",".join(sorted(prov.answers)),
            site.path,
            f"{len(bad)} record(s) could not be parsed and the translator reported "
            f"{said}. The rest were dropped without a word — {listed}. A translator "
            f"that ignores the error half of Doc.jsonl() turns half a file into a "
            f"shorter file; yield an Unreadable for each one.",
            prov.name,
        )
    )


def _translate(
    reader: Reader,
    prov: Input,
    src: Sources,
    site: Site,
    counters: dict[tuple[str, str], int],
    out: ReadResult,
    shared: SharedReads,
) -> tuple[list[Given], int]:
    given: list[Given] = []
    said = 0
    seen: set[tuple[str, tuple, str, str]] = set()
    site_name = str(site)
    questions = reader.questions
    decoder_names = prov.decoders.names if prov.decoders is not None else None
    for a in _guarded(prov, src, site, out):
        if isinstance(a, Unreadable):
            said += 1
            if a.question not in prov.answers:
                raise Refusal(
                    f"Input({prov.name!r})@{site} reported an unreadable record "
                    f"for {a.question!r}, which it does not declare."
                )
            out.findings.append(
                Finding(Kind.UNREADABLE, a.question, f"{site.path}/{a.where}",
                        a.detail, prov.name)
            )
            continue
        if not isinstance(a, Answer):
            raise Refusal(
                f"Input({prov.name!r}): the translator yielded "
                f"{type(a).__name__}, not an Answer or an Unreadable"
            )
        if a.question not in prov.answers:
            raise Refusal(
                f"Input({prov.name!r})@{site} answered {a.question!r}, which it does "
                f"not declare. A translator may only answer questions its input "
                f"declares a count for — otherwise it becomes the authority for a "
                f"question nobody ranked it on."
            )
        q = questions[a.question]
        named = {k.space for k in prov.names}
        for word in a.refs or ():
            if word not in named:
                raise Refusal(
                    f"Input({prov.name!r})@{site} named rows by {word!r}, which it does "
                    f"not declare. Declare names=(Key({word!r}, because=...),) on the "
                    f"input, and Link that word or say it is Unlinked: a reference nobody "
                    f"declared joins nothing, silently."
                )
        if a.lane is not None:
            kind = a.lane[0] if isinstance(a.lane, tuple) else a.lane
            lane = reader.lane_for(kind, prov.name)
            if lane is None:
                raise Refusal(
                    f"Input({prov.name!r})@{site} put a row in a lane of kind "
                    f"{kind!r}, which this reader does not declare. Declare it: "
                    f"Lane(kind={kind!r}, order_by=..., identity=..., because=...). "
                    f"A lane nobody declared is a free string, and the set of "
                    f"sequences a reader can produce should be readable off the page."
                )
            # Default `actor` from the lane; a row's own actor wins.
            if lane.actor is not None and (type(a.value) is dict
                                           or isinstance(a.value, MutableMapping)):
                a.value.setdefault("actor", lane.actor)
        key = _check_grain(prov, q, a, site)
        source = prov.name
        at_path, at_line = src.at_now
        if decoder_names is None:
            if a.via is not None:
                raise Refusal(
                    f"Input({prov.name!r}) stamped decoder {a.via!r} on an answer "
                    f"and declares no `decoders=`. A decoder that is not declared "
                    f"cannot be ranked against another."
                )
        else:
            if a.via is None:
                raise Refusal(
                    f"Input({prov.name!r}) declares decoders "
                    f"{list(decoder_names)} and yielded an answer to "
                    f"{a.question!r} with no `via=`. The format is a property of the "
                    f"record, so every record says which decoder read it."
                )
            if a.via not in decoder_names:
                raise Refusal(
                    f"Input({prov.name!r}) stamped undeclared decoder {a.via!r}; "
                    f"the candidates are {list(decoder_names)}"
                )
            source = f"{prov.name}/{a.via}"
        pkey = a.input_key
        if pkey is None:
            ck = (a.question, site_name)
            n = counters.get(ck, 0)
            pkey = str(n)
            counters[ck] = n + 1
        pkey = str(pkey)
        fp = (a.question, key, source, pkey)
        if fp in seen:
            raise Refusal(
                f"Input({prov.name!r})@{site} gave two answers to {a.question!r} "
                f"for row {key!r} under one name {pkey!r}. A row's name is counted "
                f"within one input's own output, so it has to be unique there."
            )
        seen.add(fp)
        keys = dict(a.keys) if a.keys else {}
        for space in keys:
            if not isinstance(space, str) or not space.strip():
                raise Refusal(
                    f"Input({prov.name!r}): a key space needs a name, got {space!r}"
                )
        given.append(
            Given(
                question=a.question,
                value=shared.one_copy(a.value),
                key=key,
                input=prov.name,
                input_key=pkey,
                source=source,
                site=site_name,
                keys=keys,
                note=a.note,
                lane=a.lane,
                refs={s: tuple(v) if isinstance(v, (list, tuple)) else (v,)
                      for s, v in (a.refs or {}).items()},
                at_path=at_path,
                at_line=at_line,
            )
        )
    return given, said


def _check_grain(prov: Input, q: Question, a: Answer, site: Site) -> tuple[Hashable, ...]:
    want = DEPTH[q.grain]
    key = a.key
    if key is None:
        key = ()
    if not isinstance(key, tuple):
        raise Refusal(
            f"Input({prov.name!r})@{site}: the key of an answer to {a.question!r} "
            f"is a tuple naming the row's address, got {type(key).__name__} "
            f"({key!r}). Write key=({key!r},)."
        )
    if len(key) != want:
        raise Refusal(
            f"Input({prov.name!r})@{site} answered {a.question!r}, declared at "
            f"{q.grain.name} grain, with a key of {len(key)} component(s) {key!r}; "
            f"{q.grain.name} grain takes {want}. Grain is checked against the answer, "
            f"not guessed from the name — a rollout-grain question answered without a "
            f"row key would otherwise be silently treated as run-grain."
        )
    return key


def _check_value_cardinality(
    reader: Reader, prov: Input, site: Site, given: list[Given], out: ReadResult
) -> None:
    """Check each declared count per row key of the grain, counting non-`None` values."""
    questions = reader.questions
    universe: dict[Grain, set[tuple]] = {}
    # question -> row key -> [answers with a value, answers without]
    counted: dict[str, dict[tuple, list[int]]] = {}
    for g in given:
        universe.setdefault(questions[g.question].grain, set()).add(g.key)
        tally = counted.setdefault(g.question, {}).setdefault(g.key, [0, 0])
        tally[g.value is None] += 1
    for qname, expect in prov.answers.items():
        if expect.card is Card.ANY:
            continue
        per_key = counted.get(qname, {})
        keys = universe.get(questions[qname].grain, set()) | set(per_key)
        if not keys:
            err = expect.check(0, f"{prov.name}.{qname}")
            if err:
                out.findings.append(Finding(Kind.MISSING, qname, str(site), err, prov.name))
            continue
        for key in keys:
            n, blank = per_key.get(key, (0, 0))
            if expect.holds(n, blank):
                continue
            err = expect.check(n, f"{prov.name}.{qname} at {key!r}", blank)
            out.findings.append(Finding(Kind.MISSING, qname, str(site), err, prov.name))
