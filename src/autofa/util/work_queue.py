# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered worker execution with bounded, replenished input buffering."""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from contextvars import copy_context

from autofa.util.cancellation import check_cancelled


def ordered_bounded_map[Input, Output](  # noqa: C901
    fn: Callable[[Input], Output],
    iterable: Iterable[Input],
    *,
    workers: int,
    max_pending: int,
    max_bytes: int,
    size_of: Callable[[Input], int],
    key_of: Callable[[Input], Hashable] | None = None,
    executor: Executor | None = None,
    observe: Callable[[dict], None] | None = None,
) -> Iterator[Output]:
    """Map a stream using one pool, refilling after each ordered result.

    Both running and completed-but-unconsumed jobs count toward the limits.
    ``size_of`` estimates retained input bytes; ``key_of`` can identify shared
    input storage, charged once until every pending owner has been consumed.
    Equal keys must describe the same storage and size.

    There can be one lookahead input beyond the admitted byte budget: its size
    cannot be discovered without reading it. A single oversized input is always
    admitted intact when the queue is empty. Jobs sharing that input may run
    together. Results keep input order, so a slow early job can still exert
    backpressure once the bounded window fills.

    A single worker without an executor runs inline without prefetching or probes.
    Its sole current input is the same irreducible minimum as an oversized row.

    Closing or failing closes the source, cancels queued jobs, and waits for
    running jobs to finish (including any checkpoint writes in ``fn``).
    A supplied executor is borrowed: only this iterator's jobs are cancelled or
    waited for, and the caller remains responsible for shutting down the pool.
    """
    if workers < 1 or max_pending < 1 or max_bytes < 1:
        raise ValueError("workers, max_pending, and max_bytes must be positive")

    source = iter(iterable)
    del iterable
    pending: deque[tuple[Future[Output], Hashable, Input]] = deque()
    ownership: dict[Hashable, tuple[int, int]] = {}
    pending_bytes = 0
    lookahead: tuple[Input, int, Hashable] | None = None
    exhausted = False
    pool = executor
    active: Future[Output] | None = None

    def run(value):
        check_cancelled()
        result = fn(value)
        check_cancelled()
        return result

    def next_result():
        nonlocal active
        active = pending.popleft()[0]
        try:
            result = active.result()
            check_cancelled()
            return result
        finally:
            # An asynchronous interrupt can escape result() while its worker
            # still runs. Keep that popped future until cleanup can wait for it.
            if active.done():
                active = None

    try:
        if workers == 1 and executor is None:
            def serial(value):
                if observe is not None:
                    # Inline execution has no retained queue and needs no size probe.
                    observe({"pending_jobs": 1, "retained_input_bytes": None,
                             "running_jobs": 1, "workers": 1, "queue_limit_bytes": max_bytes,
                             "byte_limited": False})
                return run(value)

            # map does not retain consumed values/results in a generator local.
            yield from map(serial, source)
            return
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=workers)
        while True:
            check_cancelled()
            while len(pending) < max_pending:
                check_cancelled()
                if lookahead is None:
                    if exhausted:
                        break
                    try:
                        value = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    try:
                        size = int(size_of(value))
                        if size < 0:
                            raise ValueError("input byte size must not be negative")
                        lookahead = (value, size, key_of(value) if key_of else object())
                    finally:
                        del value
                key = lookahead[2]
                owned_size, owners = ownership.get(key, (lookahead[1], 0))
                additional_bytes = 0 if owners else owned_size
                if pending and additional_bytes and pending_bytes + additional_bytes > max_bytes:
                    del key
                    break
                # Retain the admitted value even when a completed scalar result
                # no longer owns it. This also prevents identity keys being
                # reused before their byte ownership has been released.
                pending.append((pool.submit(copy_context().run, run, lookahead[0]),
                                key, lookahead[0]))
                ownership[key] = (owned_size, owners + 1)
                pending_bytes += additional_bytes
                lookahead = None
                del key

            if not pending:
                break
            if observe is not None:
                observe({"pending_jobs": len(pending), "retained_input_bytes": pending_bytes,
                         "running_jobs": sum(f.running() for f, _, _ in pending),
                         "workers": workers, "queue_limit_bytes": max_bytes,
                         "byte_limited": lookahead is not None})
            key = pending[0][1]
            owned_size, owners = ownership[key]
            if owners == 1:
                del ownership[key]
                pending_bytes -= owned_size
            else:
                ownership[key] = (owned_size, owners - 1)
            del key
            # Do not keep a local Future/result while suspended at yield: a
            # completed result can itself contain the whole original trace.
            yield next_result()
    finally:
        failing = sys.exc_info()[0] is not None
        lookahead = None
        key = None
        ownership.clear()
        cleanup_futures = []
        while pending:
            cleanup_futures.append(pending.popleft()[0])
        if active is not None:
            cleanup_futures.append(active)
        for future in cleanup_futures:
            future.cancel()
        try:
            if pool is not None and executor is None:
                pool.shutdown(wait=True, cancel_futures=True)
            else:
                # Filter cancelled futures: wait() expects executor notification
                # after cancellation, which a borrowed executor need not send.
                wait([future for future in cleanup_futures if not future.done()])
        finally:
            cleanup_futures.clear()
            active = future = None
            try:
                close = getattr(source, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException:
                        if not failing:
                            raise
            finally:
                del source
                close = None
