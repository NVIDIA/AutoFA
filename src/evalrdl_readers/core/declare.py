# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small declarations: counts, orders, folds, comparisons, keys, decoders, ignores.

None has a default; an unstated choice is a `Refusal` at construction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .errors import Refusal, need_reason
from .patterns import parse_pattern
from .questions import Grain

if TYPE_CHECKING:
    from .landmarks import Landmark

class Population(str, Enum):  # noqa: UP042 (StrEnum changes str() and format())
    """Which rollouts a reader's rows are drawn from; no default.

    Warning: nothing verifies `counts=`; do not build on it.
    """

    COMPLETED = "completed"
    """Only rollouts that ran to the end and produced a result."""

    ATTEMPTED = "attempted"
    """Completed plus the ones that started and failed."""

    PREPARED = "prepared"
    """Everything the producer set up to run, whether it ran or not."""


class Card(str, Enum):  # noqa: UP042 (StrEnum changes str() and format())
    """A cardinality; called with a reason (`Card.ONE(because=...)`) it is a count, what
    every `expect=` and `answers=` takes.

    Counts answers, not sources. Bare members are only legal where the enclosing
    declaration carries the reason (`Relate` cardinality).
    """

    NONE = "none"
    """None at all."""
    ONE = "one"
    """Exactly one."""
    AT_LEAST_ONE = "at_least_one"
    """One or more."""
    ANY = "any"
    """Any number, none included."""

    def __call__(self, because: object = None) -> _Expectation:
        return _Expectation(self, need_reason(
            because,
            f"Card.{self.name}(...)",
            "A cardinality with no reason is a guess wearing a number.",
        ))


@dataclass(frozen=True)
class _Expectation:
    """A count plus its reason; built only via `Card.X(because=...)`."""

    card: Card
    because: str

    def __post_init__(self) -> None:
        if not isinstance(self.card, Card):
            raise Refusal(
                f"expect: the count must be one of {[c.name for c in Card]}, "
                f"got {self.card!r}"
            )
        need_reason(
            self.because,
            f"Card.{getattr(self.card, 'name', self.card)}(...)",
            "A cardinality with no reason is a guess wearing a number.",
        )

    def holds(self, n: int, blank: int = 0) -> bool:
        """Whether `n` answers (and `blank` empty ones) meet this count; `check` says why not."""
        if self.card is Card.ONE:
            return n == 1
        if self.card is Card.AT_LEAST_ONE:
            return n >= 1
        if self.card is Card.NONE:
            return not (n + blank)
        return True

    def check(self, n: int, what: str, blank: int = 0) -> str | None:
        """Return the complaint, or None. `Card.NONE` also counts blank answers."""
        empty = f" ({blank} answered with no value)" if blank else ""
        if self.card is Card.ONE and n != 1:
            return f"{what}: expected exactly one, found {n}{empty} — {self.because}"
        if self.card is Card.AT_LEAST_ONE and n < 1:
            return f"{what}: expected at least one, found none{empty} — {self.because}"
        if self.card is Card.NONE and n + blank:
            return f"{what}: expected none, found {n + blank} — {self.because}"
        return None


@dataclass(frozen=True)
class At:
    """A rigid relative file pattern, anchored to a landmark."""

    landmark: Landmark
    """The landmark the pattern is relative to, as the object."""
    pattern: str
    """A file under it; `*` matches within one segment, a trailing `/**` covers a named
    directory."""
    _segs: tuple[str, ...] = field(init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        from .landmarks import Landmark

        if not isinstance(self.landmark, Landmark):
            raise Refusal(
                f"At(..., {self.pattern!r}): the first argument is the Landmark object "
                f"itself, not {self.landmark!r}. References are objects wherever a Python "
                f"object exists, so the interpreter checks them."
            )
        segs = parse_pattern(
            self.pattern, f"At({self.landmark.name}, {self.pattern!r})", subtree=True
        )
        object.__setattr__(self, "_segs", segs)

    @property
    def segments(self) -> tuple[str, ...]:
        return self._segs

    def __repr__(self) -> str:
        return f"At({self.landmark.name}, {self.pattern!r})"


@dataclass(frozen=True)
class Ignore:
    """Files deliberately not read; anchored, rigid and reasoned like a read."""

    landmark: Landmark
    """The landmark the pattern is relative to, as the object."""
    pattern: str
    """The files it accounts for, written as an `At` pattern."""
    because: str
    """Why nothing in them is stored."""
    _segs: tuple[str, ...] = field(init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        from .landmarks import Landmark

        if not isinstance(self.landmark, Landmark):
            raise Refusal(
                f"Ignore(..., {self.pattern!r}): the first argument is the Landmark object"
            )
        need_reason(
            self.because,
            f"Ignore({self.landmark.name}, {self.pattern!r})",
            "An ignore is the one construct that hides a file; without a reason it hides "
            "it for no reason.",
        )
        segs = parse_pattern(
            self.pattern,
            f"Ignore({self.landmark.name}, {self.pattern!r})",
            subtree=True,
        )
        object.__setattr__(self, "_segs", segs)

    @property
    def segments(self) -> tuple[str, ...]:
        return self._segs

    def __repr__(self) -> str:
        return f"Ignore({self.landmark.name}, {self.pattern!r})"


@dataclass(frozen=True)
class Order:
    """Who wins where two sources answer one question, and why.

    Ranks source names (`input`, `input/decoder`, `fold:question`), not input objects.
    A `None` winning over a value is stored but reported `MISSING`.
    """

    question: str
    """The `Merge.ONE` question it settles."""
    order: tuple[str, ...]
    """Every source that can answer it, best first."""
    keep_loser: bool
    """Whether a beaten value is kept beside the winner, in `ReadResult.losers`."""
    because: str
    """Why each source outranks the next."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", tuple(self.order))
        if not isinstance(self.question, str) or not self.question.strip():
            raise Refusal(f"Order: question must be a name, got {self.question!r}")
        if len(self.order) < 2:
            raise Refusal(
                f"Order({self.question!r}): an order of {len(self.order)} answers nothing"
            )
        if len(set(self.order)) != len(self.order):
            raise Refusal(f"Order({self.question!r}): a name appears twice — {self.order}")
        for name in self.order:
            if not isinstance(name, str) or not name.strip():
                raise Refusal(f"Order({self.question!r}): {name!r} is not a source name")
        if not isinstance(self.keep_loser, bool):
            raise Refusal(
                f"Order({self.question!r}): keep_loser must be declared true or false — "
                f"whether the beaten value is kept is a decision, not a detail"
            )
        need_reason(
            self.because,
            f"Order({self.question!r})",
            "An order with no reason is declaration order with a nicer name.",
        )

    def rank(self, source: str) -> int:
        """Rank of `source`."""
        if source in self.order:
            return self.order.index(source)
        raise Refusal(f"Order({self.question!r}): {source!r} is not ranked")


@dataclass(frozen=True)
class Rows:
    """Which inputs may create rows for a question (per reader, since questions are shared)."""

    question: str | Grain
    """One question's name, or a `Grain` for every question at that grain."""
    may_create: tuple[str, ...]
    """The inputs whose answers may bring a row into being."""
    because: str
    """Why these inputs are the enumeration."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "may_create", tuple(self.may_create))
        if not isinstance(self.question, Grain) and (
            not isinstance(self.question, str) or not self.question.strip()
        ):
            raise Refusal(
                f"Rows: question must be a question name or a Grain, got {self.question!r}"
            )
        if not self.may_create:
            raise Refusal(
                f"Rows({self.question!r}): names nobody. A question no input may "
                f"create rows for has no rows."
            )
        for n in self.may_create:
            if not isinstance(n, str) or not n.strip():
                raise Refusal(f"Rows({self.question!r}): {n!r} is not an input name")
        if len(set(self.may_create)) != len(self.may_create):
            raise Refusal(f"Rows({self.question!r}): a name appears twice")
        need_reason(
            self.because,
            f"Rows({self.question!r})",
            "Which sources are the enumeration is the whole question here.",
        )


class Op(str, Enum):  # noqa: UP042 (StrEnum changes str() and format())
    """Closed vocabulary of fold operations."""

    SUM = "sum"
    """The sum of the numbers."""
    MEAN = "mean"
    """Their mean."""
    MIN = "min"
    """The smallest."""
    MAX = "max"
    """The largest."""
    COUNT = "count"
    """How many subjects answered."""
    DISTINCT = "distinct"
    """The distinct values, as a sorted tuple."""


#: Ops whose value depends on the population; one colliding with a stated number is refused.
AGGREGATE_OPS = frozenset({Op.SUM, Op.MEAN, Op.MIN, Op.MAX, Op.COUNT})

#: Legal `Fold.expect` cards (it counts non-answering subjects).
FOLD_GAP_CARD = (Card.NONE, Card.ANY)


@dataclass(frozen=True)
class Fold:
    """N answers at one grain become one at the grain above.

    `expect` counts subjects that did not answer: `Card.NONE` withholds an incomplete
    total, `Card.ANY` publishes it with the shortfall noted.
    """

    into: str
    """The question one grain up that receives the total."""
    over: str
    """The question whose answers are folded."""
    op: Op
    """How they combine."""
    expect: _Expectation
    """Subjects that did not answer: `Card.NONE(...)` withholds a partial total and
    `Card.ANY(...)` publishes it; any other count is refused."""
    because: str
    """Why the total means something."""

    def __post_init__(self) -> None:
        if not isinstance(self.op, Op):
            raise Refusal(
                f"Fold(into={self.into!r}): op must be one of {[o.name for o in Op]}. The "
                f"vocabulary is closed; a callable here would be a decision no reviewer "
                f"can see in a diff."
            )
        if not isinstance(self.expect, _Expectation):
            raise Refusal(
                f"Fold(into={self.into!r}): expect is not declared. Write "
                f"expect=Card.NONE(because=...) or Card.ANY(because=...)."
            )
        if self.expect.card not in FOLD_GAP_CARD:
            raise Refusal(
                f"Fold(into={self.into!r}): expect counts the subjects that did NOT answer "
                f"{self.over!r}, so only {[c.name for c in FOLD_GAP_CARD]} mean anything "
                f"here (got {self.expect.card.name})"
            )
        if self.into == self.over:
            raise Refusal(f"Fold: into and over are the same question {self.into!r}")
        need_reason(
            self.because,
            f"Fold(into={self.into!r})",
            "A fold with no reason is an arithmetic choice nobody ranked.",
        )

    @property
    def source(self) -> str:
        """Source name of the folded value."""
        return f"fold:{self.over}"

    @property
    def aggregate(self) -> bool:
        return self.op in AGGREGATE_OPS

    def apply(self, values: list[Any], n_subjects: int) -> tuple[bool, Any, list[str]]:
        """Return (has_value, value, notes) over the low-grain `values`."""
        notes: list[str] = []
        gaps = n_subjects - len(values)
        if gaps and self.expect.card is Card.NONE:
            return (
                False,
                None,
                [
                    f"{self.into}: {self.op.value} over {self.over} covers {len(values)} of "
                    f"{n_subjects} subjects; {gaps} did not answer, so no total is "
                    f"published — {self.expect.because}"
                ],
            )
        if not values:
            return (
                False,
                None,
                [
                    f"{self.into}: none of the {n_subjects} subjects answered {self.over} "
                    f"— {self.expect.because}"
                ],
            )
        if gaps:
            notes.append(
                f"{self.into}: PARTIAL {self.op.value} over {len(values)} of {n_subjects} "
                f"subjects; {gaps} did not answer {self.over} — {self.expect.because}"
            )
        if self.op in (Op.SUM, Op.MEAN, Op.MIN, Op.MAX):
            bad = [v for v in values if isinstance(v, bool) or not isinstance(v, (int, float))]
            if bad:
                return (
                    False,
                    None,
                    notes
                    + [
                        f"{self.into}: {self.op.value} needs numbers and {len(bad)} of "
                        f"{len(values)} answers to {self.over} are not (e.g. {bad[0]!r})"
                    ],
                )
        if self.op is Op.SUM:
            return True, sum(values), notes
        if self.op is Op.MEAN:
            return True, sum(values) / len(values), notes
        if self.op is Op.MIN:
            return True, min(values), notes
        if self.op is Op.MAX:
            return True, max(values), notes
        if self.op is Op.COUNT:
            return True, len(values), notes
        seen: list[Any] = []
        for v in values:
            if v not in seen:
                seen.append(v)
        seen.sort(key=repr)
        return True, tuple(seen), notes


#: `Compare.must` vocabulary.
RELATIONS = ("equal", "at_least", "at_most")


@dataclass(frozen=True)
class Compare:
    """A relation that must hold between two answered questions, read left to right.

    `must` is one of `RELATIONS`. Fails as `MISSING`; a side with no answer is `UNCHECKED`.
    E.g. `Compare("run.n_scored", "run.n_expected", must="equal", ...)`.
    """

    left: str
    """A question this reader answers."""
    right: str
    """Another one."""
    must: str
    """`"equal"`, `"at_least"` or `"at_most"`, read left to right."""
    because: str
    """Why the relation holds for this producer."""

    def __post_init__(self) -> None:
        for side in ("left", "right"):
            v = getattr(self, side)
            if not isinstance(v, str) or not v.strip():
                raise Refusal(
                    f"Compare: {side} must be the name of a question this reader "
                    f"declares, got {v!r}"
                )
        if self.left == self.right:
            raise Refusal(
                f"Compare({self.left!r}): both sides are the same question, which holds "
                f"whatever the folder says"
            )
        if self.must not in RELATIONS:
            raise Refusal(
                f"Compare({self.left!r}, {self.right!r}): must={self.must!r} is not one "
                f"of {list(RELATIONS)}. It is written as a word rather than an operator "
                f"because `<=` in a declaration reads as an expression, and this is a "
                f"claim somebody is making about a producer."
            )
        need_reason(
            self.because,
            f"Compare({self.left!r}, {self.right!r})",
            "Two numbers that must agree is a claim about the producer, not arithmetic.",
        )

    def holds(self, left: Any, right: Any) -> bool:
        if self.must == "equal":
            return left == right
        if self.must == "at_least":
            return left >= right
        return left <= right

    @property
    def phrase(self) -> str:
        return {"equal": "equal", "at_least": "at least",
                "at_most": "at most"}[self.must]


@dataclass(frozen=True)
class Key:
    """A key space an input mints; declared because distinct id spaces can share a name."""

    space: str
    """The key space's name; on `Input.names`, the relation word."""
    because: str
    """Why these values are ids of one kind."""
    optional: bool = False
    """True where the producer writes this id only sometimes.

    One reader covers seventeen folders and two of them carry a task uuid. Without this
    flag, declaring the key space turns the other fifteen red — so people delete the
    declaration and write the join as data nothing checks. An optional mint that emits
    nothing is not a finding; a required one still is. The reason is still required, so
    "sometimes absent" stays a claim somebody wrote down.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.space, str) or not self.space.strip():
            raise Refusal(f"Key: a key space needs a name, got {self.space!r}")
        if not isinstance(self.optional, bool):
            raise Refusal(
                f"Key({self.space!r}): optional must be true or false — whether a "
                f"producer always writes this id is a fact, not a detail"
            )
        need_reason(
            self.because,
            f"Key({self.space!r})",
            "A mint with no reason cannot be told from a column name that happens to "
            "look like an id.",
        )


@dataclass(frozen=True)
class Join:
    """A join on a key space (never an input), so execution order cannot matter."""

    space: str
    """The key space both ends meet on."""

    def __post_init__(self) -> None:
        if not isinstance(self.space, str) or not self.space.strip():
            raise Refusal(f"Join: a key space needs a name, got {self.space!r}")


@dataclass(frozen=True)
class Decoder:
    """One candidate way to read a record; its `name` is stamped as `input/name`.

    `decides` is an optional predicate used by `Sources.decode`: no match is
    `Unreadable`, several is a `Refusal`. Without it the translator picks via `via=`.
    """

    name: str
    """Stamped on what it reads, as `input/name`."""
    because: str
    """Why records are written this way."""
    decides: Callable[[Any], bool] | None = None
    """A predicate that recognises a record of this shape, for `Sources.decode`."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise Refusal(f"Decoder: needs a name, got {self.name!r}")
        if self.decides is not None and not callable(self.decides):
            raise Refusal(
                f"Decoder({self.name!r}): decides is the predicate that recognises a "
                f"record as this shape, got {type(self.decides).__name__}"
            )
        if "/" in self.name:
            raise Refusal(
                f"Decoder({self.name!r}): a `/` separates an input from its decoder in "
                f"the stamped source name, so a decoder name may not contain one"
            )
        need_reason(
            self.because,
            f"Decoder({self.name!r})",
            "A decoder with no reason cannot be ranked against another.",
        )


@dataclass(frozen=True)
class Decoders:
    """Candidate decoders, chosen per record; an unordered set (rank with `Order`)."""

    candidates: tuple[Decoder, ...]
    """The decoders a record may be read by."""
    because: str
    """Why one input meets more than one shape."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates:
            raise Refusal("Decoders: no candidates")
        for c in self.candidates:
            if not isinstance(c, Decoder):
                raise Refusal(
                    f"Decoders: candidates are Decoder objects, got {type(c).__name__}"
                )
        names = [c.name for c in self.candidates]
        if len(set(names)) != len(names):
            raise Refusal(f"Decoders: a candidate name appears twice — {sorted(names)}")
        need_reason(
            self.because,
            "Decoders",
            "Naming two ways to read one file is a claim about the data.",
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.candidates)
