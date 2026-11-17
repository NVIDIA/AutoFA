# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Questions (named stored facts), the grain ladder, and landings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .errors import Refusal, need_reason


class Grain(Enum):
    """Fixed grain ladder: deeper data adds rows, never grains, so merging terminates."""

    RUN = "run"
    """The run; key `()`."""
    ROLLOUT = "rollout"
    """A rollout; a key of one component."""
    EVENT = "event"
    """An event of a rollout; two components."""
    PART = "part"
    """A content part of an event; three components."""
    CALL = "call"
    """A tool call an event made; three components."""


#: A run holds rollouts, a rollout holds events, an event holds parts and calls.
PARENT: Mapping[Grain, Grain] = {
    Grain.ROLLOUT: Grain.RUN,
    Grain.EVENT: Grain.ROLLOUT,
    Grain.PART: Grain.EVENT,
    Grain.CALL: Grain.EVENT,
}

#: Row-key length per grain; a fold's subjects group by `key[:-1]`.
DEPTH: Mapping[Grain, int] = {
    Grain.RUN: 0,
    Grain.ROLLOUT: 1,
    Grain.EVENT: 2,
    Grain.PART: 3,
    Grain.CALL: 3,
}

LOWEST_FIRST: tuple[Grain, ...] = (
    Grain.PART,
    Grain.CALL,
    Grain.EVENT,
    Grain.ROLLOUT,
    Grain.RUN,
)


class Merge(Enum):
    """How two answers to one question are settled."""

    ONE = "one"
    """Two inputs state the *same* fact. One wins, the loser is recorded. An order is
    required; unstated is refused."""

    UNION = "union"
    """Two inputs hold *different parts* of one fact. All land, sorted by a declared
    key. Declaring an order is refused."""


#: Default table list; the real schema is passed as `Reader(schema=)`. Grains are
#: independent of tables.
TABLES: tuple[str, ...] = (
    "runs",
    "rollouts",
    "attempts",
    "events",
    "parts",
    "tool_calls",
    "artifacts",
    "part_texts",
    "tool_call_texts",
    "artifact_texts",
)


def _check_landing(land: object, what: str) -> None:
    """Check a landing's shape only; table/column existence needs the schema."""
    if not isinstance(land, str) or not land.strip():
        raise Refusal(f"{what}: a landing is `table` or `table.column`, got {land!r}")
    if land.count(".") > 1:
        raise Refusal(f"{what}: {land!r} — a landing is `table` or `table.column`")
    for part in land.split("."):
        if not part.strip():
            raise Refusal(
                f"{what}: {land!r} — a landing is `table` or `table.column`, and neither "
                f"half may be blank"
            )


@dataclass(frozen=True)
class Question:
    """A named fact we store, settled at `grain` and combined by `merge`.

    `lands` is `table.column`, or `table` when the answer is a whole row (its keys name the
    columns, checked during the read). Existence is checked against the reader's schema.
    """

    name: str
    """The fact's name, by convention `grain.fact` (`rollout.score`)."""
    grain: Grain
    """The rung it is settled at, which fixes the length of its row key."""
    merge: Merge
    """How two answers to one row combine."""
    lands: str
    """Where it is stored: `table.column`, or `table` for a whole row."""
    means: str
    """What the fact is, in a sentence."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise Refusal(f"Question: needs a name, got {self.name!r}")
        if not isinstance(self.grain, Grain):
            raise Refusal(
                f"Question({self.name!r}): grain must be one of "
                f"{[g.name for g in Grain]}, got {self.grain!r}"
            )
        if not isinstance(self.merge, Merge):
            raise Refusal(
                f"Question({self.name!r}): merge must be Merge.ONE or Merge.UNION — how "
                f"two answers combine is a decision with no default"
            )
        _check_landing(self.lands, f"Question({self.name!r}).lands")
        need_reason(
            self.means,
            f"Question({self.name!r})",
            "A stored fact whose meaning is not written down is a column nobody can "
            "argue with.",
        )

    def slots(self) -> tuple[str, ...]:
        """The landing slots this question claims."""
        return (self.lands,)
