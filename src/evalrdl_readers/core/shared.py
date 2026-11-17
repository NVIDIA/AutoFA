# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records that several `Doc`s of one read parse: parsed once, then recalled.

Inputs never see each other, so two inputs reading one file each parse it. The reader knows
before any translator runs how many Docs every path will get. A record parsed while another
Doc of its path is still to come is kept, and later Docs rebuild it instead of parsing it.
Everything kept for a path is dropped when its last Doc is done. A read also keeps one copy of
each string its inputs answer (`one_copy`), dropped with the read.

Kept records are `marshal` blobs, rebuilt per Doc, so no two Docs ever share an object. Within
one large record equal values may be one object (`input._loads_merged`), and `marshal` keeps
that: a translator reads what it is handed and never writes it. JSON parses only to dict,
list, str, int, float, bool and None, all of which `marshal` round-trips exactly, and
rebuilding is several times cheaper than parsing. Measured on gym folders, where four inputs
parse one rollout stream, it takes a fifth to a third off a read; where the other Docs of a
path never parse it the blobs are wasted, which costs a few percent.

A record of LARGE bytes or more is held by at most LARGE_SLOTS Docs at once. A Doc keeps its
slot until the record it last handed out is small again, since its caller still holds that
record while it asks for the next; a Doc waiting for a slot counts as waiting, so the Doc
ahead never waits on it for room.
"""

from __future__ import annotations

import marshal
import os
import threading
from array import array
from collections import Counter
from collections.abc import Callable, Hashable
from typing import Any, BinaryIO

#: Most bytes of kept records held at once for one path. Past it, inputs running side by side
#: wait for the slowest; one after another, records are parsed again.
BUDGET = 512 << 20

#: Raw or kept bytes from which a record takes a slot: inputs side by side would otherwise each
#: hold their own copy of it at the same time.
LARGE = int(os.environ.get("EVALRDL_READ_LARGE_BYTES", 64 << 20))
LARGE_SLOTS = int(os.environ.get("EVALRDL_READ_LARGE_SLOTS", "1"))

PARSED = "parsed"
FAILED = "failed"

#: The whole document of `Doc.json()`, as opposed to a line of `Doc.jsonl()`.
WHOLE = "whole"

BLANK = "blank"
#: `SharedReads.take` past the last line.
EOF = object()


class _Stream:
    """One path several Docs read line by line: where each line is, and the lines still owed.

    Whoever first needs line i reads it (sequentially, or by its span if it is behind) and
    parses it, under the lock; the other Docs take the kept blob. Nothing rescans the file.
    """

    __slots__ = ("lock", "f", "fd", "spans", "pos", "eof", "owed", "last", "peek")

    def __init__(self, f: BinaryIO):
        self.lock = threading.Lock()
        self.f = f
        self.fd = f.fileno()
        self.spans: list[tuple[int, int]] = []
        self.pos = 0
        self.eof = False
        #: line -> (outcome, Docs still to take it)
        self.owed: dict[int, tuple[tuple, int]] = {}
        #: Doc -> the last line it asked for
        self.last: dict[int, int] = {}
        #: (line, raw) read ahead to learn its size, handed to the next read of that line
        self.peek: tuple[int, bytes] | None = None


class SharedReads:
    """Per-read store of records more than one Doc may parse; see the module docstring."""

    def __init__(self, pending: Counter, budget: int = BUDGET):
        #: path -> byte offset of each line's start, kept after the stream closes
        self.line_starts: dict[str, array] = {}
        self._pending = pending
        self._readers = Counter(pending)
        self._kept: dict[str, dict[Hashable, tuple]] = {}
        self._cost: dict[str, int] = {}
        self._budget = budget
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running = 0
        self._waiting = 0
        self._gen = 0
        #: paths whose reader ahead waits until what is kept falls to half the budget
        self._draining: set[str] = set()
        self._streams: dict[str, _Stream] = {}
        #: one copy of each string answered in this read; dropped with the read
        self._strings: dict[str, str] = {}
        self._large = threading.BoundedSemaphore(LARGE_SLOTS)
        #: threads holding a slot -> whether the record they last handed out is large
        self._large_held: dict[int, bool] = {}
        self._gated = 0

    @property
    def budget(self) -> int:
        return self._budget

    def shares(self, path: str) -> bool:
        """Whether more than one Doc of this read opens `path`, finished or not."""
        return self._readers.get(path, 0) >= 2

    def take(self, path: str, i: int, opener: Callable[[str], BinaryIO],
             parse: Callable[[bytes], Any], who: int = 0) -> Any:
        """Line `i` of `path`: EOF, (BLANK,), (PARSED, value) or (FAILED, detail).

        A PARSED value is this Doc's own object: parsed here, or rebuilt from a blob.
        """
        me = threading.get_ident()
        if self._large_held.get(me) is False:
            self._release_large()
        with self._lock:
            st = self._streams.get(path)
            if st is None:
                st = self._streams[path] = _Stream(opener(path))
        while True:
            need_slot = False
            with st.lock:
                st.last[who] = i
                owed = st.owed.get(i)
                if owed is not None:
                    outcome, left = owed
                    if (outcome[0] == PARSED and len(outcome[1]) >= LARGE
                            and me not in self._large_held):
                        need_slot = True
                    else:
                        if left <= 1:
                            del st.owed[i]
                            self._charge(path, -_cost(outcome))
                        else:
                            st.owed[i] = (outcome, left - 1)
                        break
                else:
                    left = self._pending.get(path, 0) - 1
                    # Only the Doc ahead of every other waits: one behind moves on, which is what
                    # frees what is kept.
                    ahead = i > len(st.spans)
                    if (left <= 0 or not ahead
                            or not (self._full(path) or path in self._draining)
                            or not self._may_wait()):
                        if _size(st, i) >= LARGE and me not in self._large_held:
                            need_slot = True
                        else:
                            return self._parse_and_keep(path, st, i, parse, left)
            if need_slot:
                self._acquire_large()
                continue
            self._wait_for_room(path)
        if outcome[0] == PARSED:
            self._handed(len(outcome[1]) >= LARGE)
            return (PARSED, thaw(outcome[1]))
        self._handed(False)
        return outcome

    def _handed(self, large: bool) -> None:
        me = threading.get_ident()
        if me in self._large_held:
            self._large_held[me] = large

    def _acquire_large(self) -> None:
        """Hold a slot for a large record. A thread blocked here counts as waiting, so the one
        ahead never waits on it for room."""
        me = threading.get_ident()
        if me in self._large_held:
            return
        with self._lock:
            self._gated += 1
        try:
            self._large.acquire()
        finally:
            with self._lock:
                self._gated -= 1
        self._large_held[me] = True

    def _release_large(self) -> None:
        me = threading.get_ident()
        if self._large_held.pop(me, None) is not None:
            self._large.release()

    def _parse_and_keep(self, path: str, st: _Stream, i: int,
                        parse: Callable[[bytes], Any], left: int) -> Any:
        value = self._parse_line(st, i, parse)
        if value is EOF or value[0] != PARSED:
            kept = value
        else:
            try:
                kept = (PARSED, marshal.dumps(value[1]))
            except (ValueError, RecursionError):
                kept = None
        if kept is not None and kept is not EOF and left > 0:
            if self._charge(path, _cost(kept), cap=True):
                st.owed[i] = (kept, left)
        self._handed(_span(st, i) >= LARGE)
        return value

    def one_copy(self, value: Any) -> Any:
        """`value` with each string replaced by this read's one copy of it: answers repeat the
        same prompts, tool outputs and definitions across rollouts. Containers are updated in
        place, a tuple is rebuilt; nothing else is touched."""
        strings = self._strings

        def walk(o):
            t = type(o)
            if t is str:
                return strings.setdefault(o, o)
            if t is dict:
                for k, v in o.items():
                    o[k] = walk(v)
            elif t is list:
                for i, v in enumerate(o):
                    o[i] = walk(v)
            elif t is tuple:
                return tuple(walk(v) for v in o)
            return o

        return walk(value)

    def side_by_side(self, n: int) -> None:
        """`n` inputs now run as threads: the one ahead may wait for the others."""
        with self._lock:
            self._running += n

    def leave(self) -> None:
        """A thread counted by `side_by_side` is finished."""
        self._release_large()
        with self._cond:
            self._running -= 1
            self._wake()

    def _may_wait(self) -> bool:
        """Whether another thread is still moving, so a full budget will drain."""
        with self._lock:
            return self._running - self._waiting - self._gated > 1

    def _full(self, path: str) -> bool:
        return self._cost.get(path, 0) >= self._budget

    def _wait_for_room(self, path: str) -> None:
        """Wait once for the others to take what is kept, then re-check. Never when every other
        thread waits too: this one reads on and re-parses what does not fit, so no deadlock."""
        with self._cond:
            self._waiting += 1
            gen = self._gen
            try:
                if self._running - self._waiting - self._gated > 0 and (
                        self._full(path) or path in self._draining):
                    self._draining.add(path)
                    self._cond.wait(0.5)
            finally:
                # A wake already counted this thread as moving again.
                if gen == self._gen:
                    self._waiting -= 1

    def _wake(self) -> None:
        """Under `_lock`: every waiter re-checks, and counts as moving until it waits again."""
        self._gen += 1
        self._waiting = 0
        self._cond.notify_all()

    @staticmethod
    def _parse_line(st: _Stream, i: int, parse: Callable[[bytes], Any]) -> Any:
        raw = _line(st, i)
        if raw is None:
            return EOF
        if not raw.strip():
            return (BLANK,)
        try:
            obj = parse(raw)
        except ValueError as e:
            return (FAILED, f"unparseable record: {e}")
        if isinstance(obj, dict):
            return (PARSED, obj)
        return (FAILED, f"record is a {type(obj).__name__}, not an object")

    def _charge(self, path: str, cost: int, cap: bool = False) -> bool:
        """Add `cost` to what is kept of `path`; with `cap`, refuse once its budget is reached,
        and a record larger than the whole budget. The last record kept may pass it."""
        with self._lock:
            if cap and (self._cost.get(path, 0) >= self._budget or cost > self._budget):
                return False
            self._cost[path] = self._cost.get(path, 0) + cost
            # Woken at half, not per record: the one ahead resumes with room for many lines.
            if path in self._draining and self._cost[path] < self._budget // 2:
                self._draining.discard(path)
                self._wake()
            return True

    def recall(self, path: str, record: Hashable) -> tuple | None:
        """A kept (PARSED, blob) or (FAILED, detail) for this record, if any."""
        kept = self._kept.get(path)
        return None if kept is None else kept.get(record)

    def parsed(self, path: str, record: Hashable, value: Any = None,
               failed: Any = None) -> None:
        """A Doc parsed this record itself: to `value`, or failing with `failed`."""
        if self._pending.get(path, 0) < 2:
            return
        if failed is not None:
            outcome, cost = (FAILED, failed), 64
        else:
            try:
                blob = marshal.dumps(value)
            except (ValueError, RecursionError):
                return
            outcome, cost = (PARSED, blob), len(blob)
        if not self._charge(path, cost, cap=True):
            return
        self._kept.setdefault(path, {})[record] = outcome


    def done(self, path: str, who: int = 0) -> None:
        """One Doc of `path` is finished; the lines kept past where it stopped no longer wait
        for it, and the last Doc frees what was kept."""
        self._release_large()
        st = self._streams.get(path)
        if st is not None:
            with st.lock:
                stop = st.last.pop(who, 0)
                for line in [n for n in st.owed if n > stop]:
                    outcome, left = st.owed[line]
                    if left <= 1:
                        del st.owed[line]
                        self._charge(path, -_cost(outcome))
                    else:
                        st.owed[line] = (outcome, left - 1)
        with self._lock:
            n = self._pending.get(path, 0) - 1
            self._pending[path] = n
            if n > 0:
                return
            self._kept.pop(path, None)
            st = self._streams.pop(path, None)
            self._cost.pop(path, None)
            self._draining.discard(path)
            self._wake()
        if st is not None:
            if st.eof:
                self.line_starts[path] = array("Q", (s for s, _ in st.spans))
            st.f.close()


def _cost(outcome: tuple) -> int:
    return len(outcome[1]) if outcome[0] == PARSED else 64


def _size(st: _Stream, i: int) -> int:
    """Length of line `i`, reading up to it if its span is not known yet."""
    if i > len(st.spans):
        raw = _line(st, i)
        if raw is not None:
            st.peek = (i, raw)
    return _span(st, i)


def _span(st: _Stream, i: int) -> int:
    return st.spans[i - 1][1] - st.spans[i - 1][0] if i <= len(st.spans) else 0


def _line(st: _Stream, i: int) -> bytes | None:
    """Raw line `i` (1-based) of a stream, without its newline; None past the end."""
    if st.peek is not None and st.peek[0] == i:
        raw, st.peek = st.peek[1], None
        return raw
    if i <= len(st.spans):
        start, end = st.spans[i - 1]
        return os.pread(st.fd, end - start, start)
    raw = b""
    while len(st.spans) < i:
        if st.eof:
            return None
        raw = st.f.readline()
        if not raw:
            st.eof = True
            return None
        start = st.pos
        st.pos += len(raw)
        if raw[-1:] == b"\n":
            raw = raw[:-1]
        st.spans.append((start, start + len(raw)))
    return raw


def thaw(blob: bytes) -> Any:
    return marshal.loads(blob)
