# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 2b of a read: one rollout named two ways becomes one row.

A `Relate(..., same_rollout=True)` says the rollout its `frm` rows sit in and the rollout its
`to` rows sit in are the same rollout, found by a shared key space. MEASURED on gdpval: the
stream keys a rollout by `_ng_task_index`/`_ng_rollout_index`, the deliverables directory by
`task_<uuid>/repeat_<n>`, the verdicts sidecar by the indices again, and only the stream
record states both -- so no key all three files can compute exists, and without this every
task was stored twice. An edge could name the pair; only moving the answers makes it one row,
and it has to happen before settling, or `Order` and `Rows` never see the two as one row.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from .attach import Relate
from .findings import Finding, Kind
from .input import Given
from .landmarks import Site
from .questions import Grain
from .result import ReadResult

if TYPE_CHECKING:
    from .reader import Reader


def alias_rollouts(
    reader: Reader,
    answers: list[Given],
    out: ReadResult,
    sites: Mapping[str, list[Site]] | None = None,
) -> list[Given]:
    """`answers` with every aliased rollout key replaced by the key it is an alias of."""
    alias: dict[Hashable, Hashable] = {}
    for placed in reader.mounts:
        if placed.mount.rollout is not None:
            _alias_mount(
                reader, placed, answers, (sites or {}).get(placed.root.name, []), alias, out
            )
    for att in reader.attachments:
        if not (isinstance(att, Relate) and att.same_rollout):
            continue
        space = att.on.space
        targets: dict[Hashable, set] = {}
        for g in answers:
            if g.question == att.to and space in g.keys and g.key:
                targets.setdefault(g.keys[space], set()).add(g.key[0])
        claims: dict[Hashable, set] = {}
        for g in answers:
            if g.question == att.frm and space in g.keys and g.key:
                claims.setdefault(g.key[0], set()).update(targets.get(g.keys[space], ()))
        for k, to_keys in sorted(claims.items(), key=lambda kv: repr(kv[0])):
            to_keys.discard(k)
            if len(to_keys) == 1:
                alias[k] = next(iter(to_keys))
                out.fired[f"alias:{att.frm}->{att.to}"] += 1
            elif len(to_keys) > 1:
                out.findings.append(
                    Finding(
                        Kind.CONTESTED,
                        att.frm,
                        repr(k),
                        f"{space} names {len(to_keys)} rollouts, {sorted(map(repr, to_keys))}; "
                        f"the rollout keeps its own key rather than being guessed into one -- "
                        f"{att.because}",
                        att.to,
                    )
                )
    if not alias:
        return answers

    def final(k: Hashable) -> Hashable:
        seen = {k}
        while k in alias and alias[k] not in seen:
            k = alias[k]
            seen.add(k)
        return k

    return [
        replace(g, key=(final(g.key[0]),) + tuple(g.key[1:])) if g.key and g.key[0] in alias else g
        for g in answers
    ]


def _alias_mount(
    reader: Reader,
    placed,
    answers: list[Given],
    mount_sites: list[Site],
    alias: dict[Hashable, Hashable],
    out: ReadResult,
) -> None:
    """Each mounted rollout becomes the host rollout whose `rollout` key names its mount site."""
    space = placed.mount.rollout.space
    rollout = {q for q, spec in reader.questions.items() if spec.grain is Grain.ROLLOUT}
    targets: dict[Hashable, set] = {}
    for g in answers:
        if g.input not in placed.inputs and g.question in rollout and space in g.keys and g.key:
            targets.setdefault(g.keys[space], set()).add(g.key[0])
    by_path = sorted(mount_sites, key=lambda s: -len(s.path))
    claims: dict[Hashable, str] = {}
    for g in answers:
        if g.input in placed.inputs and g.question in rollout and g.key and g.key[0] not in claims:
            path = g.site.partition("@")[2]
            path = "" if path == "." else path
            site = next(
                (
                    s
                    for s in by_path
                    if not s.path or path == s.path or path.startswith(s.path + "/")
                ),
                None,
            )
            if site is not None:
                claims[g.key[0]] = site.row
    for k, row in sorted(claims.items(), key=lambda kv: repr(kv[0])):
        to_keys = targets.get(row, set()) - {k}
        label = f"mount:{placed.mount.reader.name}"
        if len(to_keys) == 1:
            alias[k] = next(iter(to_keys))
            out.fired[f"alias:{label}"] += 1
        elif not to_keys:
            out.findings.append(
                Finding(
                    Kind.ORPHAN,
                    label,
                    repr(k),
                    f"its mount site {row!r} names no host rollout on {space!r}, so it stays "
                    f"a rollout of its own -- {placed.mount.because}",
                )
            )
        else:
            out.findings.append(
                Finding(
                    Kind.CONTESTED,
                    label,
                    repr(k),
                    f"its mount site {row!r} names {len(to_keys)} host rollouts on {space!r}, "
                    f"{sorted(map(repr, to_keys))}; it keeps its own key -- "
                    f"{placed.mount.because}",
                )
            )
