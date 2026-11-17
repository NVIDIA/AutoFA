# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reader child processes: how they start, and how long they may go without progress.

A child forked from a process with other threads inherits every lock those threads held at
that instant, held by a thread that does not exist in the child; the first time it needs one
it waits forever. So children start from a forkserver, a fresh single-threaded interpreter
that forks nothing else, and a direct fork is taken only by `context`, only when this process
has exactly one OS thread. Children that stop returning are ended (`ReadStalled`), and one
that exits without its answer is reported (`ChildLost`), instead of waited on.

Events go to the `evalrdl_readers.core.procs` logger with an `event` field in `extra`
(`read_fork_refused`, `read_child_lost`), so an application's log handler carries them as
its own.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing
import os
import sys
from collections.abc import Callable, Sequence
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from typing import Any

log = logging.getLogger(__name__)

#: How reader children start: `forkserver` (the default), or `fork`, taken only while this
#: process has one OS thread and refused to forkserver otherwise.
START_METHOD = os.environ.get("EVALRDL_READ_START_METHOD", "forkserver").strip() or "forkserver"

#: Seconds a pool may run with no subset completing before it is ended. 0: never. A subset is
#: some of a read's inputs, so it takes at most what the whole read takes in one process; the
#: default is half again over the longest such read measured, 4,731 s.
STALL_S = float(os.environ.get("EVALRDL_READ_STALL_S", "7200") or 0)

#: Imported last in the forkserver; records how many threads the server has once loaded.
_PRELOADED = "evalrdl_readers.core._preloaded"


class ReadStalled(Exception):
    """No reader child returned within the stall window; the children are terminated."""

    def __init__(self, stall_s: float, done: int, parts: int):
        self.stall_s, self.done, self.parts = stall_s, done, parts
        super().__init__(
            f"no reader process finished within {stall_s:g} s ({done} of {parts} subsets "
            f"done); the processes were terminated"
        )


class ForkRefused(Exception):
    """A child found it was forked from a process with more than one thread."""


def thread_count() -> int | None:
    """OS threads of this process, native ones included; None where it cannot be counted."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


def single_threaded() -> bool:
    """Whether this process has exactly one OS thread. False where that cannot be counted."""
    return thread_count() == 1


def refused(threads: int | None, why: str) -> None:
    log.warning(
        "reader processes: not forking from a process with %s threads; %s",
        threads,
        why,
        extra={"event": "read_fork_refused", "threads": threads},
    )


def available() -> bool:
    """Whether reader children can start here at all."""
    return "forkserver" in multiprocessing.get_all_start_methods()


def context(preload: Sequence[str] = ()):
    """The multiprocessing context reader children start from, or None where there is none.

    `preload` names modules the forkserver imports once, before it forks anything; it takes
    effect when the server starts. `__main__` is never preloaded: an application's entry
    module may start threads on import, and the server would then fork with them.
    """
    methods = multiprocessing.get_all_start_methods()
    if START_METHOD == "fork" and "fork" in methods:
        if single_threaded():
            return multiprocessing.get_context("fork")
        refused(thread_count(), "starting from a forkserver instead")
    if "forkserver" not in methods:
        return None
    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload([m for m in preload if m != "__main__"] + [_PRELOADED])
    return ctx


def locate(obj: Any) -> tuple[str, str] | None:
    """(module, attribute) of an imported module that holds `obj` at its top level, so a
    child can import it rather than unpickle it; None when no module does."""
    hits = []
    for name, mod in list(sys.modules.items()):
        if name in ("__main__", "__mp_main__") or mod is None:
            continue
        for attr, value in list(getattr(mod, "__dict__", {}).items()):
            if value is obj:
                hits.append((name, attr))
    return min(hits, key=lambda h: (len(h[0]), h)) if hits else None


def load(ref: tuple[str, str]) -> Any:
    return getattr(importlib.import_module(ref[0]), ref[1])


class ChildLost(Exception):
    """A child exited without returning its part: killed (out of memory) or crashed."""


def _child(conn, fn: Callable[[Any], Any], arg: Any) -> None:
    """In a child: `fn(arg)` sent back, or what it raised; refused if the forkserver it came
    from had more than one thread."""
    try:
        threads = getattr(sys.modules.get(_PRELOADED), "THREADS", 1)
        if threads is not None and threads > 1:
            raise ForkRefused(f"the forkserver had {threads} threads when it forked this child")
        out = ("ok", fn(arg))
    except BaseException as e:  # noqa: BLE001 - the parent raises it
        out = ("err", e)
    try:
        conn.send(out)
    except Exception as e:  # noqa: BLE001 - a result or exception that does not pickle
        conn.send(("err", RuntimeError(f"a reader process could not return its part: {e}")))
    conn.close()


def run_parts(
    fn: Callable[[Any], Any],
    args: Sequence[Any],
    processes: int,
    stall_s: float | None = None,
    preload: Sequence[str] = (),
) -> list[Any]:
    """`fn` over `args`, each in a child of its own, up to `processes` at once; results in
    `args` order.

    The first child exception is raised here, and `ChildLost` for a child that exits without
    an answer. If no child returns for `stall_s` seconds (`STALL_S` by default) the children
    are terminated and `ReadStalled` is raised.
    """
    stall = STALL_S if stall_s is None else stall_s
    ctx = context(preload)
    if ctx is None:
        raise RuntimeError("no process start method is available here")
    results: list[Any] = [None] * len(args)
    pending = list(enumerate(args))
    running: dict[connection.Connection, tuple[int, BaseProcess]] = {}
    done = 0
    try:
        while pending or running:
            while pending and len(running) < max(1, processes):
                i, a = pending.pop(0)
                r, w = ctx.Pipe(duplex=False)
                p = ctx.Process(target=_child, args=(w, fn, a), daemon=True)
                p.start()
                w.close()
                running[r] = (i, p)
            ready = connection.wait(list(running), timeout=stall if stall > 0 else None)
            if not ready:
                raise ReadStalled(stall, done, len(args))
            for r in [c for c in running if c in ready]:
                i, p = running.pop(r)
                try:
                    kind, value = r.recv()
                except EOFError:
                    p.join()
                    raise ChildLost(
                        f"reader process {p.pid} exited with {p.exitcode} without its part"
                    ) from None
                finally:
                    r.close()
                p.join()
                if kind == "err":
                    raise value
                results[i] = value
                done += 1
    finally:
        for _i, p in running.values():
            p.terminate()
        for r, (_i, p) in running.items():
            p.join()
            r.close()
    return results
