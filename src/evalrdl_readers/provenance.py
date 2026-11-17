# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance of a read: which input, reader, producer, file and lines settled each fact, and
the edges the readers stated between rows. Built from a `ReadResult` and its `Reader` alone,
so the intake's bridge and the local provenance explorer derive it the same way."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable
from typing import Any

from .core.joins import _named


def question_index(readers) -> dict[str, dict[str, Any]]:
    """reader name -> {question name -> Question}, from the reader objects themselves."""
    return {r.name: {q.name: q for q in r.question_list} for r in readers}


def reads_index(reader) -> dict[str, str]:
    """input name -> its declared read patterns, `landmark:pattern` joined."""
    return {
        p.name: "|".join(f"{a.landmark.name}:{a.pattern}" for a in p.reads) for p in reader.inputs
    }


def lane_kind(lane) -> str | None:
    """The declared kind half of a lane."""
    if lane is None:
        return None
    return str(lane[0]) if isinstance(lane, tuple) else str(lane)


def lane_name(lane) -> str | None:
    """`(kind, instance)` as the column's `kind/instance` string; `None` is the unnamed lane."""
    if lane is None:
        return None
    if isinstance(lane, tuple):
        kind, _, inst = (lane + ("",))[0], None, (lane[1] if len(lane) > 1 else "")
        return f"{kind}/{inst}" if inst else str(kind)
    return str(lane)


def relation_rows(
    result, run_id: str, asserted_by: str, asserted_at: str, ident=lambda k0: (str(k0), None)
) -> list[dict]:
    """Every edge the readers stated, landed as a row -- resolved or not."""

    # Unnamed lane is "" here, not NULL: NULL keys never join.
    def _lane(v):
        return lane_name(v) or ""

    def _half(name, key, ident_value):
        # `ikey` is in the key: an end naming nothing is "" (with resolved=False), never NULL.
        # A row named by a long producer id is held by digest, as an identity value is.
        inp, _, ikey = (name or "").partition("/")
        task, repeat = ident(key[0]) if key and len(key) > 0 else (None, None)
        return inp or None, ident_value or _named(ikey), task, repeat

    out = []
    for e in result.edges:
        fi, fk, ftask, frep = _half(e.frm_name, e.frm_key, e.frm_id)
        ti, tk, ttask, trep = _half(e.to_name, e.to_key or (), e.to_id)
        out.append(
            {
                "run_id": run_id,
                "space": e.space,
                "kind": e.kind,
                "frm_question": e.frm_question,
                "frm_task_id": ftask,
                "frm_repeat": frep,
                "frm_lane": _lane(e.frm_lane),
                "frm_input": fi,
                "frm_key": fk,
                "to_question": e.to_question,
                "to_task_id": ttask,
                "to_repeat": trep,
                "to_lane": _lane(e.to_lane),
                "to_input": ti,
                "to_key": tk,
                "frm_site": e.frm_site or "",
                "to_site": e.to_site or "",
                "resolved": e.to_name is not None,
                "reason": e.reason or None,
                "asserted_by": asserted_by,
                "asserted_at": asserted_at,
            }
        )
    return out


def provenance_rows(
    result, reader, qindex, run_id: str, ident=lambda k0: (str(k0), None)
) -> list[dict]:
    """One row per (question, input, FILE): who owns this fact, and what else it claimed."""
    seen: dict[tuple, dict] = {}
    keys_of: dict[tuple, set] = defaultdict(set)
    reads = reads_index(reader)
    for (qname, key), rows in result.rows.items():
        q = qindex.get(qname)
        for row in rows:
            # Keyed by file, not site, so each value is attributable to one input.
            slot = (qname, row.input, row.at_path or row.site)
            rec = seen.setdefault(
                slot,
                {
                    "run_id": run_id,
                    "question": qname,
                    "input": row.input,
                    "site": row.site,
                    "path": row.at_path,
                    "file": row.at_path or "",
                    "first_line": row.at_line,
                    "last_line": row.at_line,
                    "reader": result.reader,
                    "producer": result.producer_of.get(row.input),
                    "reads": reads.get(row.input),
                    "lands": getattr(q, "lands", None),
                    "task_id": None,
                    "repeat": None,
                    "n_values": 0,
                },
            )
            rec["n_values"] += 1
            if row.at_line is not None:
                lo, hi = rec["first_line"], rec["last_line"]
                rec["first_line"] = row.at_line if lo is None else min(lo, row.at_line)
                rec["last_line"] = row.at_line if hi is None else max(hi, row.at_line)
            keys_of[slot].add(key)
    for slot, rec in seen.items():
        keys = keys_of[slot]
        # Set only when the record covers exactly one rollout.
        if len(keys) == 1:
            (only,) = keys
            if len(only) == 1:
                rec["task_id"], rec["repeat"] = ident(only[0])
    return [seen[k] for k in sorted(seen)]


def rollout_ident(result, qindex) -> Callable[[Hashable], tuple[str, int]]:
    """A rollout key's first component -> the `(task_id, repeat)` it is stored under."""
    # Rollout keys (`repeat` is the only key column a question lands in).
    repeat_of: dict[Hashable, int] = {}
    for (qname, key), rows in result.rows.items():
        q = qindex.get(qname)
        if q is None or q.lands != "rollouts.repeat" or len(key) != 1:
            continue
        for row in rows:
            if isinstance(row.value, int) and not isinstance(row.value, bool):
                repeat_of[key[0]] = row.value

    # A rollout is (task, repeat) where the reader names both and no other key claims the
    # same pair; otherwise the reader's own key, which is unique by construction.
    task_of: dict[Hashable, str] = {}
    for (qname, key), rows in result.rows.items():
        q = qindex.get(qname)
        if q is None or q.lands != "rollouts.task_name" or len(key) != 1:
            continue
        for row in rows:
            if row.value is not None and str(row.value).strip():
                task_of[key[0]] = str(row.value)
    # A namespace every task name carries (switchyard's `<dataset>__`) is the benchmark's, not
    # the task's: dropped from the key so an instance joins itself under another harness.
    for (qname, _key), rows in result.rows.items():
        q = qindex.get(qname)
        if q is None or q.lands != "runs.task_namespace":
            continue
        for row in rows:
            pfx = f"{row.value}__" if row.value else None
            if pfx:
                task_of = {
                    k0: t[len(pfx) :] if t.startswith(pfx) else t for k0, t in task_of.items()
                }
    claimed = Counter((t, int(repeat_of.get(k0, 0))) for k0, t in task_of.items())

    def ident(k0: Hashable) -> tuple[str, int]:
        repeat = int(repeat_of.get(k0, 0))
        task = task_of.get(k0)
        if task is not None and claimed[(task, repeat)] == 1:
            return task, repeat
        return str(k0), repeat

    return ident
