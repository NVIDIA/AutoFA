# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrency must survive tiny reads without unbounded trace retention."""

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from autofa.util.work_queue import ordered_bounded_map


def _map(fn, values, **kwargs):
    options = dict(workers=2, max_pending=2, max_bytes=100, size_of=lambda _: 1)
    options.update(kwargs)
    return ordered_bounded_map(fn, values, **options)


def test_window_refills_before_remaining_jobs_finish():
    third_started = threading.Event()

    def work(value):
        if value == 1:
            assert third_started.wait(2), "the old window was drained before refilling"
        if value == 2:
            third_started.set()
        return value

    assert list(_map(work, range(4))) == list(range(4))


def test_single_worker_runs_inline_without_prefetching_or_sizing():
    caller = threading.get_ident()
    reads, events = [], []

    def source():
        for value in range(4):
            reads.append(value)
            yield value

    def work(value):
        assert threading.get_ident() == caller
        return value

    def unused(value):
        raise AssertionError("unbuffered work must not scan large traces for a size/key probe")

    results = _map(work, source(), workers=1, size_of=unused, key_of=unused,
                   observe=events.append)
    assert next(results) == 0
    assert reads == [0]
    assert list(results) == [1, 2, 3]
    assert len(events) == 4
    assert all(event["retained_input_bytes"] is None for event in events)


def test_nested_maps_share_one_executor_and_leave_it_usable():
    first_parents = threading.Barrier(2)
    lock = threading.Lock()
    active = peak = 0
    calls = []

    def work(stage, value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if stage == "parent" and value in (0, 1):
                first_parents.wait(timeout=2)
            calls.append((stage, value))
            return value + 1
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        parents = _map(lambda value: work("parent", value), range(12), executor=executor)

        def source():
            try:
                yield from parents
            finally:
                parents.close()

        assert list(_map(
            lambda value: work("child", value), source(), executor=executor,
        )) == list(range(2, 14))
        assert peak == 2
        assert len(calls) == 24
        assert executor.submit(lambda: "still usable").result(timeout=2) == "still usable"


@pytest.mark.parametrize("borrowed", [False, True])
def test_close_closes_source_cancels_waiting_jobs_and_waits_only_for_own_running_work(
    monkeypatch, borrowed,
):
    unrelated_release, own_release = threading.Event(), threading.Event()
    running, cancelled = threading.Event(), threading.Event()
    finished, called, closed = [], [], []
    original_cancel = Future.cancel

    def cancel(future):
        result = original_cancel(future)
        if result:
            cancelled.set()
        return result

    monkeypatch.setattr(Future, "cancel", cancel)

    def source():
        try:
            yield from range(6)
        finally:
            closed.append(True)

    def work(value):
        called.append(value)
        if value in (1, 2):
            if value == 2:
                running.set()
            assert own_release.wait(2)
            finished.append(value)
        return value

    with ThreadPoolExecutor(max_workers=3) as executor:
        unrelated = executor.submit(unrelated_release.wait, 5)
        try:
            options = {"executor": executor} if borrowed else {}
            results = _map(work, source(), max_pending=6, **options)
            assert next(results) == 0
            assert running.wait(2)
            with ThreadPoolExecutor(max_workers=1) as consumer:
                close = consumer.submit(results.close)
                try:
                    assert cancelled.wait(2)
                    assert not close.done() and not finished
                finally:
                    own_release.set()
                close.result(timeout=2)
            assert sorted(finished) == [1, 2] and sorted(called) == [0, 1, 2]
            assert closed == [True]
            assert not unrelated.done()
            assert executor.submit(lambda: 7).result(timeout=2) == 7
        finally:
            own_release.set()
            unrelated_release.set()


def test_borrowed_executor_waits_for_popped_future_when_result_is_interrupted(monkeypatch):
    running, interrupted = threading.Event(), threading.Event()
    release, checkpoint = threading.Event(), threading.Event()
    failure = KeyboardInterrupt("interrupted while awaiting result")

    def work(value):
        running.set()
        assert release.wait(2)
        checkpoint.set()
        return value

    with ThreadPoolExecutor(max_workers=1) as executor:
        original_submit = executor.submit

        def submit(fn, *args, **kwargs):
            future = original_submit(fn, *args, **kwargs)

            def result(timeout=None):
                assert running.wait(2)
                interrupted.set()
                raise failure

            # The queue wraps work to propagate cancellation context.
            if not interrupted.is_set():
                monkeypatch.setattr(future, "result", result)
            return future

        monkeypatch.setattr(executor, "submit", submit)
        results = _map(work, [0], executor=executor, workers=1)
        with ThreadPoolExecutor(max_workers=1) as consumer:
            first = consumer.submit(next, results)
            try:
                assert interrupted.wait(2)
                assert not first.done()
            finally:
                release.set()
            with pytest.raises(KeyboardInterrupt) as caught:
                first.result(timeout=2)
        assert caught.value is failure
        assert checkpoint.is_set()
        assert executor.submit(lambda: 7).result(timeout=2) == 7


def test_byte_budget_keeps_completed_inputs_charged_and_bounds_lookahead():
    first_release, second_done, lookahead_read = (
        threading.Event(), threading.Event(), threading.Event(),
    )
    started, read = [], []

    def source():
        for value in range(5):
            read.append(value)
            if value == 2:
                lookahead_read.set()
            yield value

    def work(value):
        started.append(value)
        if value == 0:
            assert first_release.wait(2)
        if value == 1:
            second_done.set()
        return value

    results = _map(work, source(), max_pending=5, max_bytes=20, size_of=lambda _: 10)
    with ThreadPoolExecutor(max_workers=1) as consumer:
        first = consumer.submit(next, results)
        try:
            assert second_done.wait(2)
            assert lookahead_read.wait(2)
            assert started == [0, 1]
            assert read == [0, 1, 2]  # At most one unadmitted input is retained.
        finally:
            first_release.set()
        assert first.result(timeout=2) == 0
        assert list(results) == [1, 2, 3, 4]


def test_oversized_input_runs_intact_without_admitting_another_input():
    started = []
    values = [b"x" * 101, b"y" * 10, b"z" * 10]

    def work(value):
        started.append(value)
        return value

    results = _map(work, values, size_of=len)
    assert next(results) is values[0]
    assert started == [values[0]]
    assert list(results) == values[1:]


def test_shared_inputs_count_bytes_once_including_oversized_rows():
    second_started = threading.Event()
    row = b"x" * 101

    def work(pair):
        number, value = pair
        if number == 0:
            assert second_started.wait(2)
        else:
            second_started.set()
        return value

    assert list(_map(
        work, [(0, row), (1, row)], size_of=lambda pair: len(pair[1]),
        key_of=lambda pair: id(pair[1]),
    )) == [row, row]


def test_shared_bytes_stay_owned_until_last_job_is_consumed():
    first_row, second_row = b"x" * 70, b"y" * 70
    started = []

    def work(pair):
        started.append(pair[0])
        return pair[0]

    results = _map(
        work, [(0, first_row), (1, first_row), (2, second_row)],
        max_pending=3, size_of=lambda pair: len(pair[1]), key_of=lambda pair: id(pair[1]),
    )
    assert next(results) == 0
    assert next(results) == 1
    assert started == [0, 1]
    assert list(results) == [2]


@pytest.mark.parametrize("origin", ["source", "worker", "size"])
def test_original_errors_propagate_and_source_closes(origin):
    failure = RuntimeError("original failure")
    closed = []

    def source():
        try:
            yield 0
            if origin == "source":
                raise failure
            yield 1
        finally:
            closed.append(True)

    def work(value):
        if origin == "worker":
            raise failure
        return value

    def size_of(value):
        if origin == "size":
            raise failure
        return 1

    with pytest.raises(RuntimeError) as caught:
        list(_map(work, source(), size_of=size_of))
    assert caught.value is failure
    assert closed == [True]


def test_source_close_failure_does_not_replace_original_worker_error():
    failure = RuntimeError("worker error")

    def source():
        try:
            yield 0
            yield 1
        finally:
            raise ValueError("close failed")

    def work(value):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        list(_map(work, source()))
    assert caught.value is failure


class _Payload:
    pass


@pytest.mark.parametrize("workers", [1, 2])
def test_consumed_inputs_and_results_are_released_while_iterator_is_open(workers):
    refs = []

    def source():
        for _ in range(8):
            value = _Payload()
            refs.append(weakref.ref(value))
            yield value
            del value

    results = _map(lambda value: value, source(), workers=workers)
    first = next(results)
    assert first is refs[0]()
    del first
    gc.collect()
    if workers > 1:
        # The inline source itself can still own its suspended current input.
        assert refs[0]() is None
    next(results)
    gc.collect()
    assert refs[0]() is None
    if workers > 1:
        assert refs[1]() is None
    results.close()
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_retained_worker_exception_does_not_keep_unrelated_inputs_alive():
    refs = []

    def source():
        for _ in range(6):
            value = _Payload()
            refs.append(weakref.ref(value))
            yield value
            del value

    def work(value):
        raise RuntimeError("failed")

    with pytest.raises(RuntimeError) as caught:
        list(_map(work, source(), workers=2, max_pending=6))
    assert caught.value.__traceback__ is not None
    gc.collect()
    # The original error deliberately retains its own detector frame. The
    # queue must not make this keep every other prefetched trace alive too.
    assert all(ref() is None for ref in refs[1:])


def test_nonpositive_limits_are_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        list(_map(lambda value: value, [1], max_bytes=0))
