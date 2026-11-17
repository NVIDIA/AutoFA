# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The store's closed vocabularies, versioned: the words a stored value may be, and what each means.

A run records the version it was decoded with (`runs.vocab_version`). A version only adds: a
word, once in a vocabulary, keeps its meaning in every later version (checked at import), so the
newest version answers for any word an older run holds. A changed meaning is a new word.

Closed word sets carry an `other` escape and a `*_raw` literal, so an unknown source value never
fails ingest. Edge kinds are the `space` of a stated relation between two events; each says
what the relation is, and which end it implies came first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType

#: Which end of an edge a kind implies came first: the row the edge leaves, or the row it names.
ENDS = ("frm", "to")


@dataclass(frozen=True)
class EdgeKind:
    """A kind of edge between two events, read from `frm` to `to`.

    `earlier` is the end the relation implies happened first, or None where it implies no
    order; `because` says why it does. `inferred_from` names the rule an inferred kind is
    matched by (a `text_key` version); None for a kind a producer states."""

    name: str
    means: str
    earlier: str | None
    because: str
    inferred_from: str | None = None

    def __post_init__(self) -> None:
        if self.earlier is not None and self.earlier not in ENDS:
            raise ValueError(f"EdgeKind({self.name!r}): earlier is 'frm', 'to' or None")
        if not self.means.strip() or not self.because.strip():
            raise ValueError(f"EdgeKind({self.name!r}): says what it means, and why")


@dataclass(frozen=True)
class Vocabulary:
    version: str
    note: str
    words: Mapping[str, frozenset[str]]
    edges: Mapping[str, EdgeKind] = field(default_factory=dict)


def evolve(
    older: Vocabulary,
    version: str,
    note: str,
    *,
    words: Mapping[str, frozenset[str]] = MappingProxyType({}),
    edges: tuple[EdgeKind, ...] = (),
) -> Vocabulary:
    """`older` plus new words and edge kinds; refuses to change or remove anything."""
    merged = dict(older.words)
    for name, add in words.items():
        merged[name] = merged.get(name, frozenset()) | frozenset(add)
    kinds = dict(older.edges)
    for e in edges:
        if e.name in kinds and kinds[e.name] != e:
            raise ValueError(
                f"{version}: edge kind {e.name!r} is in {older.version} with another meaning; "
                f"a changed meaning is a new word"
            )
        kinds[e.name] = e
    return Vocabulary(version, note, MappingProxyType(merged), MappingProxyType(kinds))


def check_additive(older: Vocabulary, newer: Vocabulary) -> None:
    for name, ws in older.words.items():
        if not ws <= newer.words.get(name, frozenset()):
            raise ValueError(f"{newer.version} drops words of {name!r} that {older.version} has")
    for name, e in older.edges.items():
        if newer.edges.get(name) != e:
            raise ValueError(f"{newer.version} changes or drops edge kind {name!r}")


V1 = Vocabulary(
    "vocab-v1",
    note="event kinds, part types, artifact kinds and the branch that resolved a run's cell",
    words=MappingProxyType(
        {
            "event_kind": frozenset({"message", "reasoning", "tool_call", "tool_result", "system"}),
            "part_type": frozenset({"text", "image", "encrypted", "binary"}),
            "artifact_kind": frozenset(
                {"document", "image", "patch", "log", "terminal_session", "data"}
            ),
            "resolved_by": frozenset({"exact", "scraped", "marker", "default"}),
        }
    ),
)

V2 = evolve(
    V1,
    "vocab-v2",
    note="the kinds of edge a reader states between two events",
    edges=(
        EdgeKind(
            "produced",
            means="the model call `frm` produced the item `to` in its response",
            earlier="frm",
            because="a response comes after the request that asked for it",
        ),
        EdgeKind(
            "sent_in",
            means="the turn `frm` was first sent to a model in the request of call `to`",
            earlier="frm",
            because="a request carries only what existed when it was sent",
        ),
        EdgeKind(
            "consumed",
            means="the model request `frm` was built from the event `to`",
            earlier="to",
            because="a request is built from what existed when it was sent",
        ),
    ),
)

V3 = evolve(
    V2,
    "vocab-v3",
    note="kinds of edge inferred from equal texts where no producer states an id",
    edges=(
        EdgeKind(
            "produced_by_text",
            means="the model call `frm` answered with a text equal to the turn `to`, as text-v1 "
            "compares texts; inferred from content, not an id either producer stated, and one "
            "of several where the text recurs",
            earlier=None,
            because="an inference orders nothing; the call's clock and the turn's lane do",
            inferred_from="text-v1",
        ),
        EdgeKind(
            "sent_in_by_text",
            means="the turn `frm` has a text equal to one the request of call `to` first sent, as "
            "text-v1 compares texts; inferred from content, not an id either producer stated, "
            "and one of several where the text recurs",
            earlier=None,
            because="an inference orders nothing; the call's clock and the turn's lane do",
            inferred_from="text-v1",
        ),
    ),
)

V4 = evolve(
    V3,
    "vocab-v4",
    note="verified telemetry parentage and captured model request associations",
    edges=(
        EdgeKind(
            "telemetry_parent",
            means=(
                "the observed span `frm` names span `to` as its same-trace parent in another lane"
            ),
            earlier="to",
            because="a child span begins after the parent whose context it received",
        ),
        EdgeKind(
            "served_request",
            means=(
                "the engine span `frm` observes the request whose response the model call `to` "
                "captured"
            ),
            earlier=None,
            because="these are observations of one request on independent producer clocks",
        ),
    ),
)

V5 = evolve(
    V4,
    "vocab-v5",
    note="the edge from an agent's tool call to the sub-agent it started",
    edges=(
        EdgeKind(
            "spawned",
            means="the tool call in event `frm` started the sub-agent whose lane opens with `to`",
            earlier="frm",
            because="a sub-agent runs only once the call that starts it was made",
        ),
    ),
)

#: Oldest first. Each adds to the one before, checked at import.
VERSIONS: tuple[Vocabulary, ...] = (V1, V2, V3, V4, V5)

#: What new intake and every reparse records.
CURRENT: Vocabulary = VERSIONS[-1]

for _older, _newer in pairwise(VERSIONS):
    check_additive(_older, _newer)


def edge_kinds() -> Mapping[str, EdgeKind]:
    """Every edge kind any version defines; the newest version, since versions only add."""
    return CURRENT.edges
