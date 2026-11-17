# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inputs, their answers, and `Sources`: the only object a translator is handed.

A translator can reach only the files its own `reads` matched.
"""

from __future__ import annotations

import json
import os

try:
    import orjson
except ImportError:  # the core runs on the standard library alone
    orjson = None

try:
    import evalrdl_readers_native
except ImportError:  # optional, as orjson is: `_loads_merged` is the reference
    evalrdl_readers_native = None

#: A value at least this large is parsed with equal strings, numbers, lists and objects made one
#: object: a long conversation re-sends its history on every call, so one record repeats each
#: message many times over.
MERGE_BYTES = int(os.environ.get("EVALRDL_READ_MERGE_BYTES", 16 << 20))


def _loads(raw: bytes | str):
    """orjson where it can, `json` where only it can: NaN, Infinity, integers past 64 bits."""
    if len(raw) >= MERGE_BYTES:
        if evalrdl_readers_native is not None and os.environ.get("EVALRDL_READERS_NATIVE") != "0":
            try:
                return evalrdl_readers_native.loads_merged(raw)
            except Exception:  # whatever it cannot reproduce exactly, the reference parses
                pass
        return _loads_merged(raw)
    if orjson is not None:
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            pass
    return json.loads(raw)


def _loads_merged(raw: bytes | str):
    """`json.loads` with each equal value one object, built bottom-up so a repeat never lives
    longer than its own parse. Equality is by identity of already-merged children, so values
    of different types never merge. A float merges only when finite and non-zero, where equal
    is the same bits."""
    strs: dict[str, str] = {}
    ints: dict[int, int] = {}
    floats: dict[float, float] = {}
    lists: dict[tuple, list] = {}
    objs: dict[tuple, dict] = {}

    def merged(v):
        t = type(v)
        if t is str:
            return strs.setdefault(v, v)
        if t is int:
            return ints.setdefault(v, v)
        if t is float:
            return floats.setdefault(v, v) if v and v - v == 0.0 else v
        if t is list:
            v = [merged(e) for e in v]
            return lists.setdefault(tuple(map(id, v)), v)
        return v

    def hook(pairs):
        d = {strs.setdefault(k, k): merged(v) for k, v in pairs}
        return objs.setdefault(tuple((k, id(v)) for k, v in d.items()), d)

    return json.loads(raw, object_pairs_hook=hook)


from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .declare import At, Decoders, Key, _Expectation
from .errors import Refusal
from .landmarks import Landmark, Site
from .shared import BLANK, EOF, PARSED, WHOLE, SharedReads, thaw
from .tree import Entry, Index


@dataclass(frozen=True)
class Unreadable:
    """Yielded by a translator for a record it could not read; reports a loss, fills nothing."""

    question: str
    """The question the record would have answered."""
    where: str
    """Where the record is, such as `path:line`."""
    detail: str
    """What was wrong with it."""

    def __post_init__(self) -> None:
        for name in ("question", "where", "detail"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise Refusal(
                    f"Unreadable: `{name}` must say something, got {v!r}. A loss nobody "
                    f"can locate is barely better than a loss nobody reports."
                )


@dataclass
class Answer:
    """What a translator yields; the runtime stamps its source, so it cannot be forged.

    `key`: row key, length = question grain depth (`()` at run grain). `input_key`: the
    producer's row id, else counted within this input. `keys`: key-space values carried.
    `refs`: by relation word, the key values of the rows this row names that way, several
    where it names several; the input declares the word in `names`. `via`: the decoder that
    read this record."""

    question: str
    """The question answered; its input must declare it."""
    value: Any
    """The value; for a `Merge.UNION` question, a mapping of the landing table's columns."""
    key: tuple[Hashable, ...] | None = None
    """The row key, as long as the question's grain is deep; `()` or `None` at run grain."""
    input_key: str | None = None
    """The producer's own id for the row; otherwise a number counted within this input."""
    keys: Mapping[str, str | int] = field(default_factory=dict)
    """The key-space values this row is named by (`{space: id}`), one id each."""
    via: str | None = None
    """The decoder that read this record, for an input with `decoders`."""
    note: str | None = None
    """A remark kept on the row (`Row.notes`)."""
    lane: str | tuple[str, str] | None = None
    """The ordered sequence of the row this answer belongs to: `KIND` or `(KIND, INSTANCE)`.

    `None` is the one unnamed lane. A lane is not an agent: a serving or monitor lane has
    none."""
    refs: Mapping[str, tuple[str | int, ...] | list[str | int]] = field(default_factory=dict)
    """By relation word, the ids of the rows this row names that way; the input declares the
    word in `names`."""

    def __post_init__(self) -> None:
        for space, value in (self.keys or {}).items():
            if isinstance(value, str) and value.strip():
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                continue
            if isinstance(value, str):
                raise Refusal(
                    f"Answer({self.question!r}): key space {space!r} was given "
                    f"{value!r}. A blank id joins every row that also has a blank one, "
                    f"silently. A key a producer did not write is a key this answer does "
                    f"not carry — leave the space out of `keys`."
                )
            raise Refusal(
                f"Answer({self.question!r}): key space {space!r} was given "
                f"{value!r}. A key space names ONE thing, and every join looks the value "
                f"up whole — a tuple of three references matches nothing, forever, while "
                f"every structural check still passes. One answer names one thing; a set "
                f"of references is several answers, one per reference, carrying the same "
                f"row key."
            )
        for space, values in (self.refs or {}).items():
            vals = tuple(values) if isinstance(values, (list, tuple)) else (values,)
            for v in vals:
                if not (
                    (isinstance(v, str) and v.strip())
                    or (isinstance(v, int) and not isinstance(v, bool))
                ):
                    raise Refusal(
                        f"Answer({self.question!r}): refs[{space!r}] holds {v!r}. A "
                        f"reference names one thing by a non-blank id; one the producer "
                        f"did not write is left out."
                    )


@dataclass(frozen=True, slots=True)
class Given:
    """An `Answer` after the runtime has stamped provenance on it."""

    question: str
    """As on the `Answer`."""
    value: Any
    """As on the `Answer`."""
    key: tuple[Hashable, ...]
    """The row key, `()` at run grain."""
    input: str
    """The input that made it."""
    input_key: str
    """The producer's id, or a number counted within the input."""
    source: str
    """The source name an `Order` ranks: `input`, or `input/decoder`."""
    site: str
    """The site it was read at, as `landmark@path`."""
    keys: Mapping[str, Hashable] = field(default_factory=dict)
    """As on the `Answer`."""
    note: str | None = None
    """As on the `Answer`."""
    lane: str | tuple[str, str] | None = None
    """As on the `Answer`."""
    refs: Mapping[str, tuple] = field(default_factory=dict)
    """As on the `Answer`."""
    at_path: str | None = None
    """The file whose bytes were last parsed when this answer was made.

    `source` names the INPUT and has never named a file: measured, `Row.source ==
    Row.input` on all 1,272 rows probed, so the finest location provenance could record
    was a glob. A translator reads a file and then yields about it, so the last file
    parsed is the file the answer is about -- stamped by the runtime rather than by the
    translator, for the same reason `source` is: a translator cannot forge it."""
    at_line: int | None = None
    """The 1-based line the record came from, where the file has lines.

    `jsonl()` yields one and every reader discarded it. A whole-file `json()` honestly has
    none, and this stays None there rather than claiming line 1."""

    @property
    def name(self) -> str:
        """The stable name of this row: `(input, input_key)`, rendered."""
        return f"{self.input}/{self.input_key}"


class Doc:
    """One matched file with parsing helpers; no way to open another file.

    Parse errors are counted in `bad_lines`; the runtime reports any not matched by a
    yielded `Unreadable`.

    Attributes:
        path: The file's path relative to the tree.
        bad_lines: The lines this file could not be read at, in the order it gave up.
    """

    __slots__ = (
        "path",
        "_relpath",
        "_name",
        "_size",
        "_index",
        "_cache",
        "bad_lines",
        "_recorded",
        "_opened",
        "_cursor",
        "_shared",
    )

    def __init__(
        self,
        index: Index,
        entry: Entry,
        site: Site,
        opened: set[str] | None = None,
        cursor: dict | None = None,
        shared: SharedReads | None = None,
    ):
        self.path = entry.path
        self._relpath = entry.path[len(site.path) + 1 :] if site.path else entry.path
        self._name = entry.path.rsplit("/", 1)[-1]
        self._size = entry.size
        self._index = index
        self._cache: dict[str, Any] = {}
        self._recorded: set[Hashable] = set()
        self._opened = opened
        self._cursor = cursor
        self._shared = shared
        self.bad_lines: list[int] = []
        """Every line this Doc could not read, in the order it gave up on them.

        `jsonl()` adds the line a record failed on. `json()` adds the line
        the document failed on — the whole file is the record there, so there is exactly
        one to lose, and losing it silently is the same defect one level up. `text()`
        adds the line a byte was replaced on."""

    def _read(self) -> None:
        """Mark as read for `UNREAD`; any accessor counts (metadata too), not only parsing."""
        if self._opened is not None:
            self._opened.add(self.path)

    def _parsed_now(self, line: int | None) -> None:
        """Stamp this file (and line) as the one answers are now being made from."""
        if self._cursor is not None:
            self._cursor["path"], self._cursor["line"] = self.path, line

    @property
    def relpath(self) -> str:
        """This file's path relative to its site."""
        self._read()
        return self._relpath

    @property
    def name(self) -> str:
        """This file's basename."""
        self._read()
        return self._name

    @property
    def size(self) -> int:
        """Length in bytes, from the index (no read)."""
        self._read()
        return self._size

    def _lost(self, line: int, tag: Hashable) -> None:
        """Record one unreadable thing; idempotent per `tag`."""
        if tag in self._recorded:
            return
        self._recorded.add(tag)
        self.bad_lines.append(line)

    def bytes(self, limit: int | None = None) -> bytes:
        """The file's bytes, or its first `limit` of them."""
        k = f"b{limit}"
        if k not in self._cache:
            self._cache[k] = self._index.tree.read(self.path, limit)
        self._read()
        self._parsed_now(None)
        return self._cache[k]

    def text(self, limit: int | None = None) -> str:
        """Decode as UTF-8 with replacement; replacements are recorded as losses unless `limit=`."""
        raw = self.bytes(limit)
        out = raw.decode("utf-8", "replace")
        if limit is None:
            lost = out.count("\ufffd") - raw.count(b"\xef\xbf\xbd")
            if lost > 0:
                head = out.split("\ufffd", 1)[0]
                self._lost(head.count("\n") + 1, "text")
        return out

    def json(self, default: Any = None) -> Any:
        """Parse as JSON; a failure is recorded and returns `default`."""
        shared = self._shared
        kept = shared.recall(self.path, WHOLE) if shared is not None else None
        if kept is not None:
            self._read()
            self._parsed_now(None)
            if kept[0] == PARSED:
                return thaw(kept[1])
            self._lost(kept[1], "json")
            return default
        try:
            value = _loads(self.bytes())
        except ValueError as e:
            line = getattr(e, "lineno", 1) or 1
            if shared is not None:
                shared.parsed(self.path, WHOLE, failed=line)
            self._lost(line, "json")
            return default
        except OSError:
            self._lost(1, "json")
            return default
        if shared is not None:
            shared.parsed(self.path, WHOLE, value)
        return value

    def _raw_lines(self) -> Iterator[bytes]:
        """The file split on `\\n` only, one line at a time.

        Not `splitlines()`: it also breaks on \\x0b, \\x0c, \\x1c-\\x1e, \\x85, legal in JSON
        strings. Streamed from the tree where it can stream, so a translator that stops
        after the first record never loads the rest, and a large file is never held whole
        next to a list of its own lines.
        """
        data = self._cache.get("bNone")
        stream = getattr(self._index.tree, "open", None)
        if data is None and stream is not None:
            f = stream(self.path)
            self._read()
            self._parsed_now(None)
            with f:
                for line in f:
                    yield line[:-1] if line[-1:] == b"\n" else line
            return
        data = self.bytes()
        start = 0
        find = data.find
        while True:
            end = find(b"\n", start)
            if end == -1:
                yield data[start:]
                return
            yield data[start:end]
            start = end + 1

    def _jsonl_shared(
        self, shared: SharedReads, opener
    ) -> Iterator[tuple[int, dict | None, str | None]]:
        """`jsonl()` over a path other Docs read too: each line is parsed once, by whichever
        Doc reaches it first."""
        self._read()
        self._parsed_now(None)
        i = 0
        while True:
            i += 1
            got = shared.take(self.path, i, opener, _loads, id(self))
            if got is EOF:
                return
            if got[0] == BLANK:
                continue
            if got[0] == PARSED:
                self._parsed_now(i)
                yield i, got[1], None
                continue
            self._lost(i, ("jsonl", i))
            yield i, None, got[1]

    def count_lines(self) -> int:
        """Non-blank lines, unparsed: for an answer that is only a count. A malformed line counts."""
        return sum(1 for raw in self._raw_lines() if raw.strip())

    def jsonl(self) -> Iterator[tuple[int, dict | None, str | None]]:
        """Yield (line_no, payload, error); bad or non-object lines are errors, recorded."""
        shared = self._shared
        path = self.path
        opener = getattr(self._index.tree, "open", None)
        if shared is not None and opener is not None and shared.shares(path):
            yield from self._jsonl_shared(shared, opener)
            return
        for i, raw in enumerate(self._raw_lines(), start=1):
            kept = shared.recall(path, i) if shared is not None else None
            if kept is None:
                if not raw.strip():
                    continue
                try:
                    payload = _loads(raw)
                except ValueError as e:
                    err = f"unparseable record: {e}"
                else:
                    if isinstance(payload, dict):
                        if shared is not None:
                            shared.parsed(path, i, payload)
                        self._parsed_now(i)
                        yield i, payload, None
                        continue
                    err = f"record is a {type(payload).__name__}, not an object"
                if shared is not None:
                    shared.parsed(path, i, failed=err)
            elif kept[0] == PARSED:
                self._parsed_now(i)
                yield i, thaw(kept[1]), None
                continue
            else:
                err = kept[1]
            self._lost(i, ("jsonl", i))
            yield i, None, err


class _NoDoc(Exception):
    """`doc(i)` matched nothing: becomes a finding for this site, not a `Refusal`."""

    def __init__(self, input_name: str, site: Site, i: int, pattern: str):
        self.input_name = input_name
        self.site = site
        self.i = i
        self.pattern = pattern
        super().__init__(
            f"{input_name}@{site}: reads[{i}] ({pattern}) matched no file, and the "
            f"translator asked for exactly one."
        )


class Sources:
    """What a translator is handed: one site, and the docs its own `reads` matched there.

    Attributes:
        site: The `Site` this run of the input is for; key rows by `site.row`.
        input: The input's name.
    """

    __slots__ = ("site", "input", "_by_at", "_patterns", "_decoders", "_cursor")

    def __init__(
        self,
        site: Site,
        by_at: list[list[Doc]],
        input_name: str,
        patterns: tuple[str, ...] = (),
        decoders=None,
        cursor: dict | None = None,
    ):
        self.site = site
        self.input = input_name
        self._by_at = by_at
        self._patterns = patterns
        self._decoders = decoders
        self._cursor = cursor if cursor is not None else {"path": None, "line": None}

    @property
    def at_now(self) -> tuple[str | None, int | None]:
        """(path, line) most recently parsed."""
        return self._cursor.get("path"), self._cursor.get("line")

    def decode(self, record, question: str):
        """Return the one declared decoder that recognises `record`, else `Unreadable`."""
        cands = [
            d
            for d in (self._decoders.candidates if self._decoders else ())
            if d.decides is not None
        ]
        if not cands:
            raise Refusal(
                f"Input({self.input!r})@{self.site}: decode() needs decoders that "
                f"declare `decides=`. Either give the candidates a decider or keep "
                f"stamping `via=` yourself."
            )
        claimed = [d for d in cands if d.decides(record)]
        if len(claimed) == 1:
            return claimed[0]
        if not claimed:
            return Unreadable(
                question=question,
                where=str(self.site),
                detail=(
                    f"no decoder recognises this record; the candidates are "
                    f"{[d.name for d in cands]}. A record nothing can read is a "
                    f"record, not an absence."
                ),
            )
        raise Refusal(
            f"Input({self.input!r})@{self.site}: {len(claimed)} decoders claim one "
            f"record — {[d.name for d in claimed]}. Two recognisers that both fire make "
            f"the stamped format depend on declaration order, which is the thing a "
            f"decoder name exists to stop."
        )

    def _pattern(self, i: int) -> str:
        return self._patterns[i] if i < len(self._patterns) else f"reads[{i}]"

    def docs(self, i: int = 0) -> tuple[Doc, ...]:
        """Every file `reads[i]` matched, in path order."""
        return tuple(self._by_at[i])

    def doc(self, i: int = 0) -> Doc:
        """The single file `reads[i]` matched; several is a `Refusal`, none ends this site
        with a finding. Use `has(i)` for optional files."""
        group = self._by_at[i]
        if not group:
            raise _NoDoc(self.input, self.site, i, self._pattern(i))
        if len(group) != 1:
            raise Refusal(
                f"{self.input}@{self.site}: reads[{i}] ({self._pattern(i)}) matched "
                f"{len(group)} files and the translator asked for exactly one. That is "
                f"a pattern too broad for what it is being asked to name — narrow it, "
                f"or read them all with docs({i})."
            )
        return group[0]

    def has(self, i: int = 0) -> bool:
        """Whether `reads[i]` matched any file."""
        return bool(self._by_at[i])

    @property
    def all(self) -> tuple[Doc, ...]:
        """Every file every read matched."""
        return tuple(d for group in self._by_at for d in group)


Translator = Callable[[Sources], Iterable["Answer | Unreadable"]]
"""The ordinary code of a reader: a function from its input's `Sources` to answers."""


@dataclass(frozen=True, eq=False)
class Input:
    """One named unit inside a reader, run once per site of its single landmark.

    `name` is stored as the source of every answer, so it must be unique. `via` is the
    translator; several `reads` mean the same content in several places, never two kinds.
    """

    name: str
    """Stamped as the source of every answer; unique in the reader, no `/`."""
    via: Translator
    """The translator."""
    reads: tuple[At, ...]
    """The files it reads, all under one landmark; it runs once per site of it."""
    answers: Mapping[str, _Expectation]
    """Per question, how many answers with a value it owes per row, and why."""
    decoders: Decoders | None = None
    """The ways its records may be written, chosen per record."""
    mints: tuple[Key, ...] = ()
    """The key spaces its rows are named in (`Answer.keys`)."""
    names: tuple[Key, ...] = ()
    """The relation words its rows name other rows by (`Answer.refs`), each a `Key` whose
    `space` is the word.

    A reader links every word an input names or says it does not."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise Refusal(f"Input: needs a name, got {self.name!r}")
        if "/" in self.name:
            raise Refusal(
                f"Input({self.name!r}): a `/` separates an input from its decoder in "
                f"the stamped source name, so an input name may not contain one"
            )
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "mints", tuple(self.mints))
        object.__setattr__(self, "names", tuple(self.names))
        for k in self.mints + self.names:
            if not isinstance(k, Key):
                raise Refusal(
                    f"Input({self.name!r}): mints and names take Key objects, got "
                    f"{type(k).__name__}"
                )
        object.__setattr__(self, "answers", dict(self.answers))
        if not self.reads:
            raise Refusal(
                f"Input({self.name!r}): reads nothing. An input with no read has no "
                f"content to be one kind of."
            )
        for a in self.reads:
            if not isinstance(a, At):
                raise Refusal(
                    f"Input({self.name!r}): reads takes At(landmark, pattern), got "
                    f"{type(a).__name__}"
                )
        landmarks = {a.landmark.name for a in self.reads}
        if len(landmarks) != 1:
            raise Refusal(
                f"Input({self.name!r}) reads from landmarks {sorted(landmarks)} — an "
                f"input runs once per site of ONE landmark. Split it in two and "
                f"declare an Order where they meet."
            )
        patterns = [a.pattern for a in self.reads]
        if len(set(patterns)) != len(patterns):
            raise Refusal(f"Input({self.name!r}): a read pattern appears twice")
        if not self.answers:
            raise Refusal(
                f"Input({self.name!r}): answers nothing. An input that answers no "
                f"question reads bytes nobody asked for."
            )
        for q, e in self.answers.items():
            if not isinstance(q, str) or not q.strip():
                raise Refusal(f"Input({self.name!r}): {q!r} is not a question name")
            if not isinstance(e, _Expectation):
                raise Refusal(
                    f"Input({self.name!r}).answers[{q!r}]: expect is not declared. How "
                    f"many answers there should be is declared per input — one "
                    f"producer's unscored rollout is a bug, another's is legal. Write "
                    f"{q!r}: Card.ONE(because=...)."
                )
        if self.decoders is not None and not isinstance(self.decoders, Decoders):
            raise Refusal(
                f"Input({self.name!r}): decoders takes Decoders(candidates=, "
                f"because=), got {type(self.decoders).__name__}"
            )
        for k in self.mints:
            if not isinstance(k, Key):
                raise Refusal(
                    f"Input({self.name!r}): mints takes Key(space, because=), got "
                    f"{type(k).__name__}"
                )
        spaces = [k.space for k in self.mints]
        if len(set(spaces)) != len(spaces):
            raise Refusal(
                f"Input({self.name!r}): mints the same key space twice — {sorted(spaces)}"
            )

    @property
    def landmark(self) -> Landmark:
        return self.reads[0].landmark

    def sources_for(self, question: str) -> tuple[str, ...]:
        """Source names this input may stamp on answers to `question`."""
        if self.decoders is not None:
            return tuple(f"{self.name}/{n}" for n in self.decoders.names)
        return (self.name,)

    def __repr__(self) -> str:
        return f"<Input {self.name}>"
