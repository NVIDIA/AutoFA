# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Landmarks: directories found by a marker file, nested via `within`.

A reader claims a folder when every landmark resolves in its declared count.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .declare import Card, _Expectation
from .errors import Refusal
from .patterns import AnyOf, Says, marker_names, parse_pattern
from .tree import Index


@dataclass(frozen=True, eq=False)
class Landmark:
    """A named kind of directory, identified by `marker` (one segment, `AnyOf` or `Says`).

    `where` lists rigid places to look relative to `within` (`""` = the enclosing directory).
    """

    name: str
    """Its name in sites (`name@path`) and findings; unique in the reader."""
    marker: str | AnyOf | Says
    """The file that identifies the directory: one path segment, `AnyOf` or `Says`."""
    where: tuple[str, ...]
    """Directories to look in, relative to the enclosing landmark; `""` is that directory."""
    expect: _Expectation
    """How many to find per enclosing directory, and why."""
    within: Landmark | None = None
    """The enclosing landmark, as the object; `None` is the tree root."""
    _markers: tuple[str, ...] = field(init=False, repr=False, compare=False, default=())
    _where: tuple[tuple[str, ...], ...] = field(
        init=False, repr=False, compare=False, default=()
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise Refusal(f"Landmark: needs a name, got {self.name!r}")
        what = f"Landmark({self.name})"
        object.__setattr__(self, "_markers", marker_names(self.marker, f"{what}.marker"))
        if not isinstance(self.where, (tuple, list)) or not self.where:
            raise Refusal(
                f"{what}.where: no locations declared. A landmark is found by looking in "
                f"declared places; searching the tree for its marker is the recursive "
                f"wildcard this design refuses."
            )
        object.__setattr__(self, "where", tuple(self.where))
        segs = []
        for w in self.where:
            if w == "":
                segs.append(())
                continue
            segs.append(parse_pattern(w, f"{what}.where"))
        if len(set(segs)) != len(segs):
            raise Refusal(f"{what}.where: a location appears twice — {self.where}")
        object.__setattr__(self, "_where", tuple(segs))
        if not isinstance(self.expect, _Expectation):
            raise Refusal(
                f"{what}: expect is not declared. How many of this landmark there should "
                f"be per enclosing site is what makes detection a check rather than a "
                f"guess, and the count carries the reason: expect=Card.ONE(because=...)."
            )
        if self.within is not None and not isinstance(self.within, Landmark):
            raise Refusal(
                f"{what}.within: a nested landmark names its parent with the parent "
                f"object, got {self.within!r}"
            )
        seen = {self.name}
        node = self.within
        while node is not None:
            if node.name in seen:
                raise Refusal(f"{what}.within: a cycle through {node.name!r}")
            seen.add(node.name)
            node = node.within

    @property
    def says(self) -> Says | None:
        """The declared content test, if any."""
        return self.marker if isinstance(self.marker, Says) else None

    @property
    def markers(self) -> tuple[str, ...]:
        return self._markers

    @property
    def where_segments(self) -> tuple[tuple[str, ...], ...]:
        return self._where

@dataclass(frozen=True)
class Site:
    """A resolved landmark directory. Key rows by `row`; `last_segment` collides across siblings."""

    landmark: str
    """The landmark's name."""
    path: str
    """The directory, relative to the tree; `""` is the root."""
    last_segment: str
    """The last component of `path`, on its own. Not a row key: see `row`."""
    anchor: Site | None = None
    """The enclosing site."""
    matched: tuple[str, ...] = ()
    """The `where` pattern this site was found by, segment by segment."""

    @property
    def row(self) -> str:
        """The wildcard-matched segments of the path, joined (e.g. `task_5/repeat_0`)."""
        base = self.anchor.path if self.anchor is not None else ""
        rel = self.path[len(base):].strip("/").split("/") if self.path != base else []
        if not self.matched:
            return "/".join(rel) if rel else self.last_segment
        picked = [
            seg for seg, pat in zip(rel, self.matched, strict=False) if any(c in pat for c in "*?[")
        ]
        return "/".join(picked) if picked else "/".join(rel)

    @property
    def chain(self) -> tuple[str, ...]:
        """Last segments of this site and its ancestors, outermost first; unique only when
        every `where` is one segment deep."""
        out: list[str] = []
        node: Site | None = self
        while node is not None:
            out.append(node.last_segment)
            node = node.anchor
        return tuple(reversed(out))

    def __str__(self) -> str:
        return f"{self.landmark}@{self.path or '.'}"


ROOT = Landmark(
    name="_root",
    marker="*",
    where=("",),
    expect=Card.ONE(
        because=(
            "every tree has exactly one root; this landmark exists so a file at the top "
            "of the tree is anchored to a name like every other file"
        ),
    ),
)
"""The tree root, so every file is anchored to some landmark; rarely named."""


@dataclass
class Resolution:
    """Resolved sites, rejected candidates, and complaints."""

    sites: dict[str, list[Site]] = field(default_factory=dict)
    rejected: list[tuple[str, str, str]] = field(default_factory=list)
    complaints: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.complaints


def _order_landmarks(landmarks: Iterable[Landmark]) -> list[Landmark]:
    """Order so parents resolve before children."""
    remaining = list(landmarks)
    out: list[Landmark] = []
    placed: set[int] = set()
    while remaining:
        progress = False
        for lm in list(remaining):
            parent = lm.within
            if parent is None or id(parent) in placed or parent is ROOT:
                out.append(lm)
                placed.add(id(lm))
                remaining.remove(lm)
                progress = True
        if not progress:  # pragma: no cover - cycles are refused at construction
            raise Refusal(
                f"landmarks {[lm.name for lm in remaining]} cannot be ordered by `within`; "
                f"a parent is missing from the reader's landmarks"
            )
    return out


def _enclosing(d: str) -> Iterable[str]:
    """The directories strictly enclosing `d`, outermost first; the tree root is not one."""
    i = d.find("/")
    while i != -1:
        yield d[:i]
        i = d.find("/", i + 1)


def resolve_landmarks(idx: Index, landmarks: Iterable[Landmark]) -> Resolution:
    """Resolve every landmark, parents first, checking `expect` per anchor.

    Nearest wins: a match nested inside another match of the same landmark is rejected.
    """
    res = Resolution()
    root_site = Site(ROOT.name, "", "", None)
    res.sites[ROOT.name] = [root_site]
    for lm in _order_landmarks(landmarks):
        if lm.name == ROOT.name:
            continue
        anchors = res.sites.get(lm.within.name, []) if lm.within is not None else [root_site]
        found: list[Site] = []
        for anchor in anchors:
            candidates: list[str] = []
            by_pat: dict[str, tuple[str, ...]] = {}
            for pat in lm.where_segments:
                for d in idx.dirs_at(anchor.path, pat):
                    candidates.append(d)
                    by_pat.setdefault(d, pat)
            candidates = sorted(set(candidates))
            named = [d for d in candidates if idx.has_marker(d, lm.markers)]
            for d in candidates:
                if d not in named:
                    res.rejected.append(
                        (lm.name, d, f"no {'|'.join(lm.markers)} in this directory")
                    )
            qualifying = named
            if lm.says is not None:
                qualifying = []
                for d in named:
                    ok, complaint = idx.marker_says(d, lm.says)
                    if complaint:
                        res.complaints.append((lm.name, d, complaint))
                    elif ok:
                        qualifying.append(d)
                    else:
                        res.rejected.append(
                            (lm.name, d,
                             f"{'|'.join(lm.markers)} does not say "
                             f"{lm.says.contains!r} in its first 256 KB")
                        )
            kept: list[str] = []
            qualified = set(qualifying)
            for d in qualifying:
                outer = next((o for o in _enclosing(d) if o in qualified), None)
                if outer is not None:
                    res.rejected.append(
                        (lm.name, d, f"nearest wins: inside {outer or '.'}/")
                    )
                else:
                    kept.append(d)
            here = [
                Site(lm.name, d, d.rsplit("/", 1)[-1] if d else "", anchor,
                     by_pat.get(d, ()))
                for d in kept
            ]
            err = lm.expect.check(len(here), f"Landmark({lm.name}) under {anchor}")
            if err:
                res.complaints.append((lm.name, str(anchor), err))
            found.extend(here)
        res.sites[lm.name] = found
    return res
