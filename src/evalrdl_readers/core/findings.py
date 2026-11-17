# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Findings: what a real folder got wrong, returned as data alongside the rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(str, Enum):  # noqa: UP042 (StrEnum changes str() and format())
    """The closed list of things a read can report."""

    MISSING = "missing"
    """A declared cardinality was not met: a landmark, a read, or an answer.

    Also the two places a value is gone and no declaration chose to lose it: a settle
    won by an answer carrying `None` while another source carried a value, and a key of
    a row landing in a whole table that the table has no column for."""

    CONTESTED = "contested"
    """Two sources said different things about one thing, and an order picked one.
    Also: a file hidden — or claimed — by two declarations."""

    COVERAGE = "coverage"
    """Two sources agreed on the value and disagreed on how much they covered; a fold
    published a total over an incomplete population; a declared mint minted nothing."""

    ORPHAN = "orphan"
    """A join found nothing to attach to, or an edge has one end unnameable. Never a
    blank: the row is kept and the reason is recorded on it."""

    UNREAD = "unread"
    """A file in the tree that no input read and no `Ignore` accounted for."""

    UNBACKED = "unbacked"
    """A source described a row that no enumerating source produced.

    A rollup naming 267 trials where 18 directories exist is not a disagreement an Order
    can settle: an Order picks a winner among sources that answered the *same* row, and
    does nothing about rows only one of them named. Without this, the rollup quietly
    invents 249 rollouts and the read is clean."""

    UNREADABLE = "unreadable"
    """A record a translator was handed and could not read. Dropping one silently is how
    half a file becomes a shorter file, so there is a word for it."""

    UNCHECKED = "unchecked"
    """A declared check that never ran, and what it would have checked.

    A landmark declaring `Card.ANY` that resolves to zero sites takes every count
    underneath it with it: the input never runs, so nothing is counted, so nothing is
    reported. Measured over the 53 corpus folders: **286 written `Card.ONE` and
    `Card.AT_LEAST_ONE` claims were never evaluated, across 37 folders — against 285
    `MISSING` findings actually emitted.** The library skipped as many checks as it ran
    and said nothing about either fact, and three of eight authors found it
    independently.

    Deliberately **not** in `LOSS`, and the choice was not made to keep the corpus
    green: measured, none of the 8 folders that report `ok` is affected either way. It
    is out because a landmark declaring `Card.ANY(because=...)` has already said in
    writing that it may be absent, and when it is absent there is nothing underneath it
    to have lost. Making that a failure would make `Card.ANY` on a landmark mean nothing
    — the trap `Relate` refuses in its own message — and would leave the author the same
    two bad options they started with: check nothing, or claim no folder. What was
    missing was never the verdict. It was the sentence.
    """


LOSS = frozenset({Kind.MISSING, Kind.COVERAGE, Kind.UNREAD, Kind.UNREADABLE,
                  Kind.UNBACKED})
"""The kinds that mean something is unaccounted for.

`CONTESTED`, `ORPHAN` and `UNCHECKED` are deliberately not here. An `Order` that fires emits a
CONTESTED every time it works, and a join records its misses as ORPHANs on purpose.
Counting those as failure marks a correct reader broken, and readers then stop using the
features that produce them. `UNCHECKED` is out for a different reason: it does not say
something went wrong, it says something was never looked at, which is news a reader has
to see and not a verdict on the folder. It still costs `quiet`.
"""


@dataclass(frozen=True, order=True)
class Finding:
    """One thing a folder got wrong, as data; a read with findings still has rows."""

    kind: Kind
    """What kind of thing."""
    question: str
    """The question, or `unread`, `landmark:<name>`, `reader:<name>`."""
    where: str
    """The row key, site or path it is about."""
    detail: str
    """What happened, with the declaration's own reason."""
    source: str = ""
    """The input or source it concerns, where one does."""

    def line(self) -> str:
        """The finding on one line, as reports print it."""
        src = f" [{self.source}]" if self.source else ""
        return f"[{self.kind.value:9}] {self.question} @ {self.where}{src}: {self.detail}"


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Canonical order, so two reads of one folder are byte-identical."""
    return sorted(findings, key=lambda f: (f.kind.value, f.question, f.where, f.source, f.detail))
