# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rigid landmark-relative path patterns, and marker matching (`AnyOf`, `Says`).

Matched segment by segment: `*` never crosses `/`; only a trailing `**` after a named
directory does."""

from __future__ import annotations

from collections.abc import Iterable

from .errors import Refusal, need_reason

GLOB_CHARS = "*?["


def is_literal(segment: str) -> bool:
    """Whether a segment names exactly one thing: `fnmatchcase` on it is plain equality."""
    return not any(c in segment for c in GLOB_CHARS)


def parse_pattern(pattern: object, what: str, *, subtree: bool = False) -> tuple[str, ...]:
    """Validate one rigid relative pattern and return it as segments.

    `subtree=True` permits a trailing `**` (reads of an envelope, and ignores).
    """
    if not isinstance(pattern, str):
        raise Refusal(f"{what}: pattern must be a string, got {type(pattern).__name__}")
    if pattern.startswith("/"):
        raise Refusal(
            f"{what}: {pattern!r} is absolute. Every path is relative to a landmark, "
            f"which is what lets the tree move without breaking the declaration."
        )
    if "\\" in pattern:
        raise Refusal(f"{what}: {pattern!r} — separators are `/`, on every platform")
    segs = tuple(s for s in pattern.split("/") if s != "")
    if not segs:
        raise Refusal(f"{what}: empty pattern")
    if any(s == ".." for s in segs):
        raise Refusal(f"{what}: {pattern!r} escapes its landmark")
    if any(s == "." for s in segs):
        raise Refusal(f"{what}: {pattern!r} — `.` is not a path segment here")
    for i, seg in enumerate(segs):
        if "**" in seg and seg != "**":
            raise Refusal(
                f"{what}: {pattern!r} — `**` is a whole path segment or nothing. "
                f"`{seg}` mixes a recursive wildcard into a name, so its reach is unreadable."
            )
        if seg != "**":
            continue
        if not subtree:
            raise Refusal(
                f"{what}: {pattern!r} — `**` is not allowed here. A recursive wildcard "
                f"names files at several depths that mean several different things."
            )
        if i != len(segs) - 1:
            raise Refusal(
                f"{what}: {pattern!r} — a `**` must be trailing. `{pattern}` is a selector, "
                f"not an address: it matches files at several depths that mean several "
                f"different things. Name the directory and write `.../**`."
            )
        if i == 0:
            raise Refusal(
                f"{what}: {pattern!r} — a `**` must name the directory whose subtree it "
                f"covers, and there is no segment before it."
            )
        prev = segs[i - 1]
        if any(c in prev for c in GLOB_CHARS):
            raise Refusal(
                f"{what}: {pattern!r} — a `**` must name the directory whose subtree it "
                f"covers. The segment before it is {prev!r}, which is itself a glob, so the "
                f"pattern says `every subtree whose name happens to match`. Write the "
                f"directory out."
            )
    return segs


class AnyOf:
    """Alternative spellings of one marker (or literal); flat, never nested.

    Arguments:
        names: Two or more spellings.
    """

    __slots__ = ("names",)

    def __init__(self, *names: str):
        if len(names) < 2:
            raise Refusal(
                "AnyOf(...): needs at least two spellings; an AnyOf of one is a marker "
                "with extra steps"
            )
        for n in names:
            if isinstance(n, AnyOf):
                raise Refusal(
                    "AnyOf(...): is flat. Nesting builds an expression whose failure can "
                    "only report that it was false, instead of naming the spelling that "
                    "was missing."
                )
            if not isinstance(n, str) or not n.strip():
                raise Refusal(f"AnyOf(...): {n!r} is not a marker name")
        if len(set(names)) != len(names):
            raise Refusal(f"AnyOf{names!r}: a spelling appears twice")
        self.names = tuple(names)

    def __iter__(self) -> Iterable[str]:
        return iter(self.names)


HEAD_BYTES = 256 * 1024
"""How much of a marker a `Says` may look at: the first 256 KB, and not a byte more.

Fixed, not a knob. Across every marker file in the corpus the discriminating field sits in
the first 17,739 bytes — usually the first few hundred, because it is the first key of the
first record even in a 5 MB stream. 256 KB is fourteen times the worst real case, which is
the point: a bound nobody has to tune is a bound nobody argues about.
"""


class Says:
    """A marker that must also contain a literal substring in its first 256 KB (`HEAD_BYTES`).

    Picks among flavours the reader already declares; never computes a location.
    Literal only: a truncated window may not parse.

    Arguments:
        marker: The marker file, one segment or an `AnyOf`.
        contains: A literal of at least three characters it must contain, or an `AnyOf` of them.
        because: Why the literal tells this producer's file from another's.
    """

    __slots__ = ("marker", "contains", "because", "names")

    def __init__(self, marker: str | AnyOf, contains: str | AnyOf, because: str):
        if isinstance(marker, Says):
            raise Refusal(
                "Says(...): is flat. A marker that has to say two things is two landmarks."
            )
        self.names = marker_names(marker, "Says(...)")
        wanted = tuple(contains) if isinstance(contains, AnyOf) else (contains,)
        for contains in wanted:
            if not isinstance(contains, str) or not contains.strip():
                raise Refusal(
                    f"Says(...): `contains` must be a literal to look for, got "
                    f"{contains!r}. There is no negative form: absence is not evidence, "
                    f"and a cancelled run is missing almost everything."
                )
            if len(contains) < 3:
                raise Refusal(
                    f"Says(..., contains={contains!r}): too short to identify anything. A "
                    f"literal this common will match a folder that is not this producer's."
                )
        need_reason(
            because,
            "Says",
            "A content test with no reason is a guess about somebody else's file format.",
        )
        self.marker, self.contains, self.because = marker, wanted, because


def marker_names(marker: object, what: str) -> tuple[str, ...]:
    """Normalise a marker (a name, or an `AnyOf` of names) to the spellings it accepts."""
    if isinstance(marker, Says):
        return marker.names
    names = tuple(marker) if isinstance(marker, AnyOf) else (marker,)
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise Refusal(f"{what}: marker must be a file name, got {name!r}")
        segs = parse_pattern(name, what)
        if len(segs) != 1:
            raise Refusal(
                f"{what}: marker {name!r} must be one path segment. A marker says which "
                f"file makes a directory this kind of landmark; where to look is `where`."
            )
    return names
