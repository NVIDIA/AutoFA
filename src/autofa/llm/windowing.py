# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliberate text windows for LLM classify — no magic 4000-char habit.

Detectors pass a full surface (or a step-selected excerpt) and name the
evidence they need via ``focus``. Truncation is a policy, not a hand-rolled
slice in detector source.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TypedDict

FOCUS_CHOICES = frozenset({"auto", "head", "tail", "ends", "full"})

# Default budget for one classify call. Large enough for real CoT; small enough
# that hub requests stay practical.
DEFAULT_MAX_CHARS = 16_000
# For ``ends``: keep opening context and closing decision when we must cut.
_ENDS_HEAD_SHARE = 0.40
_MAX_RECEIPTS = 64
_MAX_RECEIPT_CHARS = 2_000_000


class WindowSourceFragment(TypedDict):
    """Exact source text with zero-based Python character offsets."""

    source: str
    start: int
    end: int
    text: str


class WindowOmission(TypedDict):
    """Analyzer metadata, never text authored in the original source."""

    source: str
    original_chars: int
    omitted_ranges: list[list[int]]
    focus: str
    origin: str


class WindowEvidence(TypedDict):
    """Completeness applies to windowing, not original-source availability."""

    complete: bool
    source_fragments: list[WindowSourceFragment]
    analyzer_window_metadata: list[WindowOmission]


@dataclass(frozen=True)
class _Piece:
    """An exact source span and its position in the legacy rendered string."""

    text: str
    source: str
    start: int
    left: int
    right: int
    # Each JSON serialization escapes the preceding representation once.
    encodings: tuple[bool, ...] = ()


@dataclass(frozen=True)
class _Window:
    text: str
    pieces: tuple[_Piece, ...]
    omissions: tuple[WindowOmission, ...]

    def evidence(self) -> WindowEvidence:
        return {
            "complete": not self.omissions,
            "source_fragments": [
                {"source": p.source, "start": p.start, "end": p.start + len(p.text),
                 "text": p.text} for p in self.pieces
            ],
            "analyzer_window_metadata": [
                {**m, "omitted_ranges": [list(span) for span in m["omitted_ranges"]]}
                for m in self.omissions
            ],
        }


class _WindowText(str):
    """String compatibility, including callers outside detector execution."""

    def __new__(cls, window: _Window):
        value = super().__new__(cls, window.text)
        value.window = window
        return value

    def __getnewargs__(self):
        return (self.window,)


_WINDOWS: ContextVar[list[_Window] | None] = ContextVar("autofa_text_windows", default=None)


@contextmanager
def window_provenance() -> Iterator[None]:
    """Keep receipts through string formatting/JSON within one detector call.

    Nested helper/dependency calls share that scope. A new item never inherits
    another item's receipts. Only bounded excerpts, never full omitted input, are
    retained. Arbitrary edits to strings cannot preserve their source locations.
    """
    if _WINDOWS.get() is not None:
        yield
        return
    token = _WINDOWS.set([])
    try:
        yield
    finally:
        _WINDOWS.reset(token)


def _encode(text: str, encodings: tuple[bool, ...]) -> str:
    for ascii_only in encodings:
        text = json.dumps(text, ensure_ascii=ascii_only)[1:-1]
    return text


def _representations(text: str, max_chars: int) -> Iterator[tuple[str, tuple[bool, ...]]]:
    """Recognize nested JSON packing without guessing from marker text.

    Recorded windows contain omission separators, so escaping grows until the
    representation cannot fit in the supplied input. Deduplicate equivalent
    ASCII/non-ASCII paths rather than imposing a depth that could lose provenance.
    """
    pending = [(text, ())]
    seen = {text}
    while pending:
        rendered, encodings = pending.pop()
        if len(rendered) > max_chars:
            continue
        yield rendered, encodings
        for ascii_only in (False, True):
            encoded = _encode(rendered, (ascii_only,))
            if len(encoded) <= max_chars and encoded not in seen:
                seen.add(encoded)
                pending.append((encoded, (*encodings, ascii_only)))


def _known_pieces(text: str) -> tuple[list[_Piece], list[WindowOmission]]:
    """Resolve only exact, recorded windows; literal omission markers are data."""
    windows = list(_WINDOWS.get() or [])
    if isinstance(text, _WindowText):
        windows.append(text.window)
    source = hashlib.sha256(text.encode()).hexdigest()[:16]
    matches = []
    for window in windows:
        for rendered, encoding in _representations(window.text, len(text)):
            if not rendered:
                continue
            offset = text.find(rendered)
            while offset >= 0:
                matches.append((offset, offset + len(rendered), window, encoding))
                offset = text.find(rendered, offset + len(rendered))
    pieces, omissions = [], []
    cursor = 0
    # Prefer the widest recorded window when windows overlap or are nested.
    for left, right, window, encoding in sorted(matches, key=lambda m: (m[0], -m[1])):
        if left < cursor:
            continue
        if left > cursor:
            pieces.append(_Piece(text[cursor:left], source, cursor, cursor, left))
        for piece in window.pieces:
            start = left + len(_encode(window.text[:piece.left], encoding))
            end = left + len(_encode(window.text[:piece.right], encoding))
            pieces.append(replace(piece, left=start, right=end,
                                  encodings=piece.encodings + encoding))
        omissions.extend(window.omissions)
        cursor = right
    if cursor < len(text):
        pieces.append(_Piece(text[cursor:], source, cursor, cursor, len(text)))
    return pieces, omissions


def _clip_piece(piece: _Piece, left: int, right: int, output_offset: int) -> _Piece | None:
    left, right = max(left, piece.left), min(right, piece.right)
    if left >= right:
        return None
    if piece.encodings:
        # Never expose half an escaped character as if it were original code.
        boundaries = [0]
        for char in piece.text:
            boundaries.append(boundaries[-1] + len(_encode(char, piece.encodings)))
        start = bisect_left(boundaries, left - piece.left)
        end = bisect_right(boundaries, right - piece.left) - 1
        left, right = piece.left + boundaries[start], piece.left + boundaries[end]
    else:
        start, end = left - piece.left, right - piece.left
    if start >= end:
        return None
    return replace(piece, text=piece.text[start:end], start=piece.start + start,
                   left=left + output_offset, right=right + output_offset)


def _window(text: str, *, focus: str, max_chars: int) -> _Window:
    if not text:
        return _Window("", (), ())
    focus = normalize_focus(focus)
    budget = max(64, int(max_chars))
    if focus == "auto":
        focus = "tail" if _looks_like_json_blob(text) else "ends"
    ranges = [(0, len(text))]
    marker = ""
    if len(text) > budget:
        if focus in {"full", "head"}:
            marker = "\n…[tail omitted]"
            ranges = [(0, budget - len(marker))]
        elif focus == "tail" or budget - len("\n…[middle omitted]…\n") < 64:
            marker = "[head omitted]…\n"
            ranges = [(len(text) - budget + len(marker), len(text))]
            focus = "tail"
        else:
            marker = "\n…[middle omitted]…\n"
            available = budget - len(marker)
            head = max(32, int(available * _ENDS_HEAD_SHARE))
            ranges = [(0, head), (len(text) - (available - head), len(text))]
    pieces, omissions = _known_pieces(text)
    output, selected = "", []
    for left, right in ranges:
        if marker and (output or focus == "tail"):
            output += marker
        shift = len(output) - left
        for piece in pieces:
            clipped = _clip_piece(piece, left, right, shift)
            if clipped is not None:
                selected.append(clipped)
        output += text[left:right]
    if marker and focus in {"full", "head"}:
        output += marker
    if marker:
        source = hashlib.sha256(text.encode()).hexdigest()[:16]
        boundaries = [0, *[x for pair in ranges for x in pair], len(text)]
        omitted = [[left, right] for left, right in zip(
            boundaries[::2], boundaries[1::2], strict=True,
        )
                   if left < right]
        omissions.append({"source": source, "original_chars": len(text),
                          "omitted_ranges": omitted, "focus": focus,
                          "origin": "autofa.text_window"})
    # Repeated/nested use of a window need not duplicate its metadata.
    unique = {json.dumps(m, sort_keys=True): m for m in omissions}
    return _Window(output, tuple(selected), tuple(unique.values()))


def text_window_evidence(
    text: str, *, focus: str = "auto", max_chars: int = DEFAULT_MAX_CHARS,
) -> WindowEvidence:
    """Return exact source fragments plus separate analyzer omission metadata.

    ``complete`` describes windowing only, not whether the original source is
    complete or sufficient for a claim. Offsets are Python character offsets in
    the identified input. Use ``require_complete=True`` on classifier calls for
    a claim that requires the whole supplied input; unresolved evidence abstains.
    """
    return _window(text, focus=focus, max_chars=max_chars).evidence()


def normalize_focus(focus: str | None) -> str:
    f = (focus or "auto").strip().lower()
    if f not in FOCUS_CHOICES:
        raise ValueError(
            f"focus must be one of {sorted(FOCUS_CHOICES)}, got {focus!r}"
        )
    return f


def _looks_like_json_blob(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(("[", "{"))


def text_window(
    text: str,
    *,
    focus: str = "auto",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Return a char window of ``text`` under ``focus``.

    - ``full`` — whole string, or an explicitly marked prefix if over budget
    - ``head`` — prefix
    - ``tail`` — suffix
    - ``ends`` — prefix + suffix with an ellipsis marker (keeps setup and wrap-up)
    - ``auto`` — JSON/trajectory blobs → ``tail`` (payload is at the end);
      plain prose → ``ends`` (not a blind prefix)

    The default budget is 16,000 characters, with a minimum of 64. Even ``full``
    remains budget-limited. Empty input returns ''; invalid focus raises ValueError
    for nonempty input. This is a window, not proof of complete evidence review.
    """
    if not text:
        return ""
    window = _window(text, focus=focus, max_chars=max_chars)
    if not window.omissions:
        return str(text)
    receipts = _WINDOWS.get()
    if receipts is not None:
        if (len(receipts) >= _MAX_RECEIPTS
                or sum(len(w.text) for w in receipts) + len(window.text) > _MAX_RECEIPT_CHARS):
            raise ValueError(
                "Text-window provenance budget exceeded; use fewer or smaller evidence windows. "
                "Receipts cannot be discarded without losing omission provenance."
            )
        receipts.append(window)
    return _WindowText(window)
