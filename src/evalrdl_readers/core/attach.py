# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attachments: `Contribute` (rows to a stream), `Relate` (edges), and `Lane`. Joins name a key space, never an input, so order never matters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .declare import Card, Join
from .errors import Refusal, need_reason


@dataclass(frozen=True)
class Contribute:
    """One input adds rows to a stream question, with its own ids and clock.

    `identity` (a column or tuple of columns, compared as a tuple) and `order_by` name
    columns in this input's own rows; the merge interleaves inputs afterwards.
    """

    to: str
    """The `Merge.UNION` question the rows are added to."""
    input: str
    """The input whose answers they are."""
    order_by: str
    """The column of its rows that orders them."""
    identity: str | tuple[str, ...]
    """The column, or tuple of columns, that names a row; equal identities are one row."""
    because: str
    """Why these columns order and name the rows."""

    def __post_init__(self) -> None:
        for field_name in ("to", "input", "order_by"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise Refusal(
                    f"Contribute(to={self.to!r}): {field_name} must be a name, got {value!r}"
                )
        ident = self.identity
        if not isinstance(ident, (str, tuple, list, type(None))):
            raise Refusal(
                f"Contribute(to={self.to!r}): identity must be a column name or a "
                f"tuple of them, got {type(ident).__name__}"
            )
        cols = (ident,) if isinstance(ident, str) else tuple(ident or ())
        if not cols:
            raise Refusal(
                f"Contribute(to={self.to!r}): identity names no column. A stream whose "
                f"rows have no identity cannot be deduplicated, so two inputs describing "
                f"one event become two events."
            )
        for c in cols:
            if not isinstance(c, str) or not c.strip():
                raise Refusal(
                    f"Contribute(to={self.to!r}): identity must be a column name or a "
                    f"tuple of them, got {c!r}"
                )
        if len(set(cols)) != len(cols):
            raise Refusal(
                f"Contribute(to={self.to!r}): a column appears twice in identity — "
                f"{list(cols)}"
            )
        object.__setattr__(self, "identity", ident if isinstance(ident, str) else cols)
        need_reason(
            self.because,
            f"Contribute(to={self.to!r}, input={self.input!r})",
            "Adding rows to somebody else's stream is a claim about what those rows are.",
        )

    @property
    def identity_columns(self) -> tuple[str, ...]:
        """`identity` as a tuple."""
        return (self.identity,) if isinstance(self.identity, str) else self.identity


class _Undeclared:
    """Sentinel default telling "not written" apart from an explicit `None`."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<undeclared>"


_UNDECLARED = _Undeclared()


@dataclass(frozen=True)
class Relate:
    """An edge between two rows, matched on a key space; an unnameable end is an `ORPHAN`.

    `cardinality`: bare `Card`, `to` rows per `frm` row (reason is `because`).
    `to_card`: `frm` rows per `to` row, or `None` for no claim; required, no default.
    `same_rollout`: the two ends are one rollout named two ways. Where the join resolves to
    exactly one `to` rollout, every answer under the `frm` rollout's key is moved to the
    `to` rollout's key before anything settles, so the two describe one row (`alias.py`).
    """

    frm: str
    """The question whose rows the edge leaves."""
    to: str
    """The question whose rows it reaches."""
    on: Join
    """The key space the two ends meet on."""
    cardinality: Card
    """How many `to` rows each `frm` row reaches, as a bare `Card`."""
    because: str
    """What the edge says, and why the counts hold."""
    to_card: Card | None = _UNDECLARED
    """How many `frm` rows each `to` row receives, or `None` for no claim; must be written."""
    same_rollout: bool = False
    """The two ends are one rollout named two ways, and are read as one."""

    def __post_init__(self) -> None:
        if self.to_card is _UNDECLARED:
            raise Refusal(
                f"Relate({self.frm!r} -> {self.to!r}): to_card is not declared. It is the "
                f"same question as `cardinality` from the other end: how many {self.frm!r} "
                f"rows each {self.to!r} row must receive. Write to_card=None if the claim "
                f"is not made — right for a leaf of a DAG, wrong for a sub-agent nobody "
                f"spawned. It has no default because a check that is available and not "
                f"required is a suggestion, and the measured compliance rate with the "
                f"suggestions in this library is near zero. Writing to_card=None is a "
                f"decision somebody made; omitting it would be one nobody made."
            )
        for field_name in ("frm", "to"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise Refusal(f"Relate: {field_name} must be a question name, got {value!r}")
        if self.frm == self.to:
            raise Refusal(f"Relate({self.frm!r}): both ends are the same question")
        if not isinstance(self.on, Join):
            raise Refusal(
                f"Relate({self.frm!r} -> {self.to!r}): `on` is a Join naming a key space"
            )
        if not isinstance(self.cardinality, Card):
            raise Refusal(
                f"Relate({self.frm!r} -> {self.to!r}): cardinality must be declared, as a "
                f"bare Card — its reason is this Relate's own `because`"
            )
        if self.to_card is not None and not isinstance(self.to_card, Card):
            raise Refusal(
                f"Relate({self.frm!r} -> {self.to!r}): to_card is a bare Card or None. "
                f"None says how many {self.frm!r} rows each {self.to!r} row must receive "
                f"is a claim this declaration does not make — which is a decision, and is "
                f"why it has to be written."
            )
        need_reason(
            self.because,
            f"Relate({self.frm!r} -> {self.to!r})",
            "An edge with no reason cannot be told from a coincidence of column names.",
        )


@dataclass(frozen=True)
class Link:
    """An edge between two rows of one stream, matched on a key space that one input mints and
    the other names under a relation word.

    Rows of one input name, under the relation `word` (`Answer.refs`), values that rows of
    the other carry as their key in `on`'s space. `word` is what the edge is, a word of the application's vocabulary. `frm_input` and
    `to_input` are the inputs whose rows the edge leaves and reaches. The end that mints the
    space is named; the other names it (one input may be both ends, its `frm` rows naming).
    Checked per row key, and only where both inputs gave that key rows: a source a folder
    does not hold is not checked, one it holds is. `cardinality`: `to` rows per `frm` row
    taking part; `to_card`: `frm` rows per `to` row taking part, or `None` for no claim
    (required, as on `Relate`).

    `inferred`: the space is not an id either producer states but one the reader derives from
    content (a text's key), so a value matching nothing states nothing: no finding, no edge, a
    count in `fired`. An inferred link claims no counts (`Card.ANY`, `to_card=None` or ANY),
    links every row a value reaches, and none where it reaches more than `max_matches`: a
    content that common (boilerplate) identifies nothing, so it is only counted.
    """

    word: str
    """What the edge is, stored as the relation's `space`; the naming end's `refs` key."""
    to: str
    """The stream both ends contribute to."""
    frm_input: str
    """The input whose rows the edge leaves."""
    to_input: str
    """The input whose rows it reaches."""
    on: Join
    """The key space one end mints and the other names."""
    cardinality: Card
    """How many `to` rows each taking-part `frm` row reaches, as a bare `Card`."""
    because: str
    """What the edge says about the two rows, and why the counts hold."""
    to_card: Card | None = _UNDECLARED
    """How many `frm` rows each taking-part `to` row receives, or `None`; must be written."""
    inferred: bool = False
    """The space is derived from content, not stated: a miss is counted, never a finding."""
    max_matches: int = 8
    """For an inferred link, a value reaching more rows than this links none."""

    def __post_init__(self) -> None:
        for field_name in ("word", "to", "frm_input", "to_input"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise Refusal(f"Link: {field_name} must be a name, got {value!r}")
        name = f"Link({self.word!r}, {self.frm_input!r} -> {self.to_input!r})"
        if not isinstance(self.on, Join):
            raise Refusal(f"{name}: `on` is a Join naming a key space")
        if not isinstance(self.cardinality, Card):
            raise Refusal(f"{name}: cardinality is a bare Card; its reason is `because`")
        if self.to_card is _UNDECLARED:
            raise Refusal(
                f"{name}: to_card is not declared. Say how many {self.frm_input!r} rows "
                f"each {self.to_input!r} row must receive, or to_card=None if that claim "
                f"is not made."
            )
        if self.to_card is not None and not isinstance(self.to_card, Card):
            raise Refusal(f"{name}: to_card is a bare Card or None")
        if self.inferred and (
            self.cardinality is not Card.ANY or self.to_card not in (None, Card.ANY)
        ):
            raise Refusal(
                f"{name}: an inferred link claims no counts: a text that recurs, or is not "
                "sent on, is no loss. Use Card.ANY and to_card=None or Card.ANY."
            )
        if not isinstance(self.max_matches, int) or self.max_matches < 1:
            raise Refusal(f"{name}: max_matches is a count of rows, at least 1")
        need_reason(self.because, name,
                    "what the edge says about the two rows, and why the counts hold")


@dataclass(frozen=True)
class Unlinked:
    """A relation word an input's rows name others by that this reader joins to nothing,
    and why.

    A named word is linked or declared unlinked: a reference nobody joins is otherwise
    indistinguishable from a join that silently stopped matching."""

    word: str
    """The relation word the input names rows by."""
    input: str
    """The input that names them."""
    because: str
    """Why nothing in this reader is there to reach."""

    def __post_init__(self) -> None:
        for field_name in ("word", "input"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise Refusal(f"Unlinked: {field_name} must be a name, got {value!r}")
        need_reason(self.because, f"Unlinked({self.word!r}, {self.input!r})",
                    "why rows naming this way have nothing here to reach")


def link_namer(ln: Link, frm, to):
    """The end of a link whose rows name: the other end mints the link's space, and this
    one names its word. One input as both ends names from its `frm` rows."""
    space, word = ln.on.space, ln.word
    def mints(p) -> bool:
        return space in {k.space for k in p.mints}
    def names(p) -> bool:
        return word in {k.space for k in p.names}
    if frm is to:
        return frm if mints(frm) and names(frm) else None
    if mints(to) and names(frm):
        return frm
    if mints(frm) and names(to):
        return to
    return None


ATTACHMENTS = (Contribute, Relate, Link, Unlinked)


@dataclass(frozen=True)
class Lane:
    """One ordered sequence inside a row (trace, sub-agent turns, monitor), declared by kind.

    Kinds are declared, instances are data; an undeclared kind is refused. `actor` may be None.
    """

    kind: str
    """The lane's kind; an answer names it as `lane=KIND` or `lane=(KIND, INSTANCE)`."""
    order_by: str | tuple[str, ...]
    """The column, or tuple of columns, its rows are numbered by; where a mounted reader orders
    its own rows of this kind differently from the host, each producer's rows keep its own."""
    identity: str | tuple[str, ...]
    """The column, or tuple of columns, that names a row within the lane."""
    because: str
    """Why these rows are one sequence and not part of another."""
    actor: str | None = None
    """The `actor` its rows get where they state none; `None` for a lane no agent wrote."""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise Refusal(f"Lane: needs a kind, got {self.kind!r}")
        if "/" in self.kind:
            raise Refusal(
                f"Lane({self.kind!r}): a `/` separates a kind from its instance in a "
                f"lane name, so a kind may not contain one"
            )
        for field_name in ("order_by", "identity"):
            value = getattr(self, field_name)
            cols = (value,) if isinstance(value, str) else tuple(value or ())
            if not cols or any(not isinstance(c, str) or not c.strip() for c in cols):
                raise Refusal(
                    f"Lane({self.kind!r}): {field_name} names the column(s) its rows "
                    f"carry it in, got {value!r}"
                )
        need_reason(self.because, f"Lane({self.kind!r})",
                    "why these rows are one sequence and not part of another")
