# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream validated JSON without repeatedly rescanning large scalar tokens."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import suppress
from typing import Any, BinaryIO

import ijson

_SCALAR_END = re.compile(rb'[\s{}\[\],:"]')
_TOKEN_END_BYTES = b' \t\r\n{}[]:,"'


def _string_end(data: bytes, start: int, end: int, escaped: bool) -> tuple[int | None, bool]:
    """Find an unescaped quote using linear scans, preserving split escape pairs.

    This only frames parser feeds; ijson still validates escapes, UTF-8 and JSON.
    Avoid a regex repeating per character, which can allocate a large match stack.
    """
    quote = data.find(b'"', start, end)
    while quote >= 0:
        previous = quote - 1
        while previous >= start and data[previous] == 92:
            previous -= 1
        backslashes = quote - previous - 1 + int(previous < start and escaped)
        if backslashes % 2 == 0:
            return quote + 1, False
        quote = data.find(b'"', quote + 1, end)
    previous = end - 1
    while previous >= start and data[previous] == 92:
        previous -= 1
    backslashes = end - previous - 1 + int(previous < start and escaped)
    return None, bool(backslashes % 2)


def _feed_boundaries(data: bytes, buf_size: int, state: list[bool]) -> Iterator[bytes]:
    """Feed at most one pending scalar or one ordinary buffer before yielding.

    An adaptive read can end a huge string and contain a dense suffix. Splitting
    at that scalar's boundary lets the consumer drain its event before parsing
    the suffix, whose event batches stay bounded by ``buf_size``.
    """
    start, size = 0, len(data)
    while start < size:
        in_string, escaped, in_scalar = state
        if in_string:
            boundary, escaped = _string_end(data, start, size, escaped)
            end = boundary if boundary is not None else size
            in_string = boundary is None
        elif in_scalar:
            boundary = _SCALAR_END.search(data, start)
            end = boundary.start() + 1 if boundary is not None else size
            in_string = data[end - 1] == 34
        else:
            end = min(start + buf_size, size)
            pos = start
            while pos < end:
                quote = data.find(b'"', pos, end)
                if quote < 0:
                    break
                boundary, escaped = _string_end(data, quote + 1, end, False)
                if boundary is None:
                    in_string = True
                    break
                pos = boundary
        state[:] = [in_string, escaped, not in_string and data[end - 1] not in _TOKEN_END_BYTES]
        yield data[start:end]
        start = end


def parse_json(
    source: BinaryIO, *, with_prefix: bool = False, buf_size: int = 64 * 1024,
    use_float: bool = False,
) -> Iterator[tuple[str, Any] | tuple[str, str, Any]]:
    """Yield ijson events, growing reads while a token remains incomplete.

    YAJL rescans an unfinished token when more bytes arrive. Fixed-size reads
    therefore make large strings quadratic to parse. Geometric read growth
    bounds this rescanning, and normal-sized reads resume whenever events appear.
    Large feeds stop at a scalar boundary before parsing any dense suffix, so
    queued events stay bounded independently of the adaptive read size. Memory
    is bounded by the largest pending token/input chunk plus one ordinary event
    batch, rather than the complete document. The caller owns and bounds ``source``
    (for example, to one JSONL record); every requested read size is positive.

    Exhausting the iterator validates the complete document, including its end.
    Closing it early releases the parser without validating an unread suffix or
    replacing the consumer's exception with an incomplete-document error.
    """
    if not isinstance(buf_size, int) or buf_size <= 0:
        raise ValueError("JSON buffer size must be a positive integer")
    events = ijson.sendable_list()
    factory = ijson.parse_coro if with_prefix else ijson.basic_parse_coro
    parser = factory(events, use_float=use_float)
    read_size = buf_size
    state = [False, False, False]  # unfinished string, escape pair, or unquoted scalar
    try:
        while chunk := source.read(read_size):
            had_events = False
            for feed in _feed_boundaries(chunk, buf_size, state):
                parser.send(feed)
                had_events = bool(events)
                if had_events:
                    read_size = buf_size
                yield from events
                events.clear()
            del chunk, feed
            if not had_events:
                read_size *= 2
        # EOF both validates completeness and can finish a number token.
        parser.close()
        parser = None
        yield from events
    finally:
        events.clear()
        if parser is not None:
            # Cancellation or a consumer/source error is not a JSON EOF.
            with suppress(Exception):
                parser.close()
