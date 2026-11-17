# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a read produces: settled `Row`s, `Edge`s between them, and the `ReadResult`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .declare import Population
from .errors import Refusal
from .findings import LOSS, Finding, Kind
from .input import Given
from .landmarks import Site


@dataclass(slots=True)
class Row:
    """One settled row; its name `(input, input_key)` is independent of other inputs."""

    question: str
    """The question it answers."""
    key: tuple[Hashable, ...]
    """Its row key."""
    value: Any
    """The settled value."""
    input: str
    """The input that read it."""
    input_key: str
    """The producer's id, or a number counted within the input."""
    source: str
    """The source name that won: `input`, `input/decoder` or `fold:question`."""
    site: str
    """The site it was read at."""
    at_path: str | None = None
    """The file it came from."""
    at_line: int | None = None
    """The line, where the file has lines."""
    keys: Mapping[str, Hashable] = field(default_factory=dict)
    """The key-space values it is named by."""
    lane: str | tuple[str, str] | None = None
    """Its lane: `None`, `KIND` or `(KIND, INSTANCE)`."""
    refs: Mapping[str, tuple] = field(default_factory=dict)
    """By relation word, the ids it names other rows by."""
    position: int | None = None
    """Its place in its stream and lane, assigned last from `order_by`."""
    notes: tuple[str, ...] = ()
    """Remarks: an `Answer.note`, a join that missed."""

    @property
    def name(self) -> str:
        """`input/input_key`: the row's name, independent of every other input."""
        return f"{self.input}/{self.input_key}"

    @property
    def stable(self) -> tuple[str, str]:
        """`(input, input_key)`."""
        return (self.input, self.input_key)

    def note(self, text: str) -> None:
        """Add a remark."""
        self.notes = self.notes + (text,)

    @classmethod
    def of(cls, g: Given) -> Row:
        """The row an answer settles into."""
        return cls(
            question=g.question,
            key=g.key,
            value=g.value,
            input=g.input,
            input_key=g.input_key,
            source=g.source,
            site=g.site,
            at_path=g.at_path,
            at_line=g.at_line,
            keys=dict(g.keys),
            lane=g.lane,
            refs=dict(g.refs),
            notes=(g.note,) if g.note else (),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    """An edge between two rows, or why one end has no name. Records full addresses."""

    kind: str
    """`link` or `relate`: the declaration that made it."""
    space: str
    """The relation word, or the key space a `Relate` joins on."""
    frm_question: str
    """The question of the row it leaves."""
    frm_name: str
    """That row's name."""
    to_question: str
    """The question of the row it reaches."""
    to_name: str | None
    """That row's name; `None` where nothing matched, and `reason` says why."""
    reason: str = ""
    """Why the far end has no name; empty where it has one."""
    frm_key: tuple = ()
    """The row key of the row it leaves."""
    to_key: tuple | None = None
    """The row key of the row it reaches."""
    frm_lane: str | tuple[str, str] | None = None
    """The lane the row it leaves sits in."""
    to_lane: str | tuple[str, str] | None = None
    """The lane the row it reaches sits in.

    An endpoint without its lane is worse than no endpoint: two lanes of one rollout can
    share ids and disagree on positions, so an index resolved against the wrong lane
    mispairs every row of the pair and says nothing about it."""
    frm_site: str = ""
    """The site the row it leaves was read at."""
    to_site: str = ""
    """The site the row it reaches was read at.

    A row's name is counted within one input's output at one site, so the name alone
    repeats across sites."""
    frm_id: str | None = None
    """The identity value of the row it leaves, where the edge was stated by identity."""
    to_id: str | None = None
    """The identity value of the row it reaches, where the edge was stated by identity.

    A row's name may be a position in its file; the identity is the producer's own id, which
    is what the row stores and what the edge names."""


@dataclass
class ReadResult:
    """What one folder produced. Findings are data; a read with findings still has rows."""

    reader: str
    """The reader that read the folder; `<none>` where none claimed it."""
    tree: str
    """The folder's name."""
    claimed: bool = True
    """Whether every landmark resolved in its declared count."""
    sites: dict[str, list[Site]] = field(default_factory=dict)
    """Per landmark, the directories it resolved to."""
    rows: dict[tuple[str, tuple[Hashable, ...]], list[Row]] = field(default_factory=dict)
    """Per `(question, key)`, the settled rows."""
    losers: dict[tuple[str, tuple[Hashable, ...]], list[Row]] = field(default_factory=dict)
    """Per `(question, key)`, the values an `Order` with `keep_loser=True` beat."""
    edges: list[Edge] = field(default_factory=list)
    """Every edge between rows, those with no far end included."""
    findings: list[Finding] = field(default_factory=list)
    """What the folder got wrong, in a fixed order."""
    read_paths: set[str] = field(default_factory=set)
    """Every file some input opened."""
    input_seconds: Counter = field(default_factory=Counter)
    """Seconds each input spent translating, summed over its sites."""
    hidden: dict[str, str] = field(default_factory=dict)
    """Per file an `Ignore` accounts for, that `Ignore` and its reason."""
    minted: dict[str, int] = field(default_factory=dict)
    """Per declared key space, how many ids were minted."""
    counts: Population | None = None
    """Which rollouts these rows are drawn from, as the reader declared it.

    Published so two readers of one producer that disagree disagree visibly."""
    producer_of: dict[str, str] = field(default_factory=dict)
    """Per input, the producer that wrote the bytes it reads, from the reader's grouping.

    `Row.input` already names the input; this is the only thing that says who wrote what
    that input read, and it is a declaration, not an inference."""
    declared_stack: tuple[str, ...] = ()
    """The producer group names, outermost first, as the reader declared them.

    It is documentation: nothing here checks it against the folder and no column is
    derived from it."""
    fired: Counter = field(default_factory=Counter)
    """How many times each declaration did something in this folder, by declaration id.

    Ids are like `read:<input>@<landmark>/<pattern>` and `order:<question>`. A declaration
    absent here read nothing, checked nothing and settled nothing — over one folder that is
    ordinary, and over every folder we hold it is dead code that still has to be maintained
    and still looks like a claim. Nothing here is a finding: which absences are defects is a
    judgement about the corpus, and it is made outside one read."""
    line_starts: dict[str, Sequence[int]] = field(default_factory=dict)
    """Per file several inputs read in one pass, the byte offset of each line's start."""

    # -- accessors ---------------------------------------------------------

    def stream(self, question: str, key: Sequence[Hashable] = ()) -> list[Row]:
        """Every settled row for one question about one thing, in position order."""
        return list(self.rows.get((question, tuple(key)), []))

    def value(self, question: str, key: Sequence[Hashable] = (), default: Any = None) -> Any:
        """The settled value of a `Merge.ONE` question."""
        got = self.rows.get((question, tuple(key)))
        if not got:
            return default
        if len(got) != 1:
            raise Refusal(
                f"{question!r} at {tuple(key)!r} settled {len(got)} rows; it is a stream. "
                f"Use .stream()."
            )
        return got[0].value

    def keys_of(self, question: str) -> list[tuple[Hashable, ...]]:
        """Every row key the question settled, sorted."""
        return sorted(
            (k for (q, k) in self.rows if q == question), key=lambda k: tuple(map(repr, k))
        )

    def of_kind(self, kind: Kind) -> list[Finding]:
        """The findings of one kind."""
        return [f for f in self.findings if f.kind is kind]

    @property
    def ok(self) -> bool:
        """Nothing is lost; the pipeline gate. `CONTESTED`/`ORPHAN` are news, not loss."""
        return self.claimed and not any(f.kind in LOSS for f in self.findings)

    @property
    def quiet(self) -> bool:
        """Nothing to report at all, including the findings that are only news."""
        return self.claimed and not self.findings

    def report(self) -> str:
        """A line naming the reader and the folder, then one line per finding."""
        pop = f", counting {self.counts.value} rollouts" if self.counts else ""
        head = (f"{self.reader} @ {self.tree}: "
                f"{'claimed' if self.claimed else 'NOT CLAIMED'}{pop}")
        return "\n".join([head] + [f.line() for f in self.findings])

    def to_dict(self) -> dict[str, Any]:
        """A deterministic, comparable rendering of everything this read decided."""
        return {
            "reader": self.reader,
            "counts": self.counts.value if self.counts else None,
            "claimed": self.claimed,
            "sites": {
                name: [s.path for s in sites] for name, sites in sorted(self.sites.items())
            },
            "rows": [
                {
                    "question": q,
                    "key": [repr(k) for k in key],
                    "rows": [
                        {
                            "name": r.name,
                            "source": r.source,
                            "position": r.position,
                            "value": repr(r.value),
                            "notes": list(r.notes),
                        }
                        for r in rows
                    ],
                }
                for (q, key), rows in sorted(
                    self.rows.items(), key=lambda kv: (kv[0][0], tuple(map(repr, kv[0][1])))
                )
            ],
            "edges": [
                {
                    "kind": e.kind,
                    "space": e.space,
                    "frm": [e.frm_question, [repr(k) for k in e.frm_key], e.frm_name],
                    "to": [
                        e.to_question,
                        None if e.to_key is None else [repr(k) for k in e.to_key],
                        e.to_name,
                    ],
                    "reason": e.reason,
                }
                for e in sorted(
                    self.edges,
                    key=lambda e: (
                        e.kind,
                        e.space,
                        e.frm_question,
                        tuple(map(repr, e.frm_key)),
                        e.frm_name,
                        e.to_name or "",
                    ),
                )
            ],
            "findings": [f.line() for f in self.findings],
            "hidden": dict(sorted(self.hidden.items())),
            "minted": dict(sorted(self.minted.items())),
            "fired": dict(sorted(self.fired.items())),
        }
