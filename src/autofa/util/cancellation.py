# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cooperative cancellation shared by one tool's worker threads and model calls."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class Cancellation:
    """A tool-local token; cancelling one session never stops another session."""

    def __init__(self, stop_requested: Callable[[], bool | str] | None = None) -> None:
        self.event = threading.Event()
        self.lock = threading.RLock()
        self.stop_requested = stop_requested
        self._last_poll = 0.0
        self.supervisor_requested = False
        self.supervisor_action: str | None = None
        self._local = threading.local()

    @property
    def guard_depth(self) -> int:
        return getattr(self._local, "guard_depth", 0)

    def poll(self) -> None:
        now = time.monotonic()
        if self.stop_requested is not None and now - self._last_poll >= 0.25:
            self._last_poll = now
            if action := self.stop_requested():
                self.request_supervisor_stop(action if isinstance(action, str) else "stop")

    def request_supervisor_stop(self, action: str) -> None:
        """Remember the observed intent even if the controller releases it later."""
        self.supervisor_requested = True
        self.supervisor_action = action
        self.cancel()

    def cancel(self) -> None:
        # A save already underway may finish its checkpoint. No state-changing
        # save may begin after cancellation, even if an HTTP response arrives
        # later. Recovery may still record receipts for work already completed.
        with self.lock:
            self.event.set()


_CURRENT: ContextVar[Cancellation | None] = ContextVar("autofa_cancellation", default=None)


def current_cancellation() -> Cancellation | None:
    return _CURRENT.get()


@contextmanager
def cancellation_scope(cancellation: Cancellation) -> Iterator[None]:
    token = _CURRENT.set(cancellation)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def check_cancelled() -> None:
    cancellation = _CURRENT.get()
    if cancellation is not None and cancellation.guard_depth:
        return
    if cancellation is not None:
        cancellation.poll()
    if cancellation is not None and cancellation.event.is_set():
        # BaseException prevents detector/judge error handlers from converting
        # an interrupted operation into negative labels or processing results.
        raise asyncio.CancelledError("AutoFA tool cancelled")


@contextmanager
def cancellation_guard() -> Iterator[None]:
    """Observe cancellation before a checkpoint, never between its nested saves.

    Enter the outer guard before the first related ledger/library mutation.
    Its successful entry is the commit boundary; later cancellation is deferred
    until the next unguarded check. Actual I/O exceptions still propagate.
    """
    cancellation = _CURRENT.get()
    if cancellation is None:
        yield
    else:
        with cancellation.lock:
            check_cancelled()
            cancellation._local.guard_depth = cancellation.guard_depth + 1
            try:
                yield
            finally:
                cancellation._local.guard_depth -= 1


async def cancellable[T](
    awaitable: Awaitable[T], *, cancel_event: threading.Event | None = None,
) -> T:
    """Cancel an async child session when its enclosing sync tool is stopped."""
    cancellation = _CURRENT.get()
    if cancellation is None and cancel_event is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            check_cancelled()
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("AutoFA judge batch cancelled")
            await asyncio.wait({task}, timeout=0.1)
        # Completion wins a cancellation race: the child may already have
        # committed a mutation or produced a meaningful failure. Preserve that
        # outcome instead of replacing it with a late cancellation signal.
        return await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
