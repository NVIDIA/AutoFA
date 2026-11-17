# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finished detector batches must not remain reachable through saved errors."""

from __future__ import annotations

import gc
import threading
import weakref
from collections import Counter
from types import SimpleNamespace

import pytest

from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import DetectorAbstained, RegexTimeoutError
from autofa.mcp import tools_eval
from autofa.metrics.association import StreamingBucketAccumulator
from autofa.session.recovery import RunInterruptedError

from support.evaluation import _scoring_runtime


class _Payload:
    def __init__(self):
        self.trace = bytearray(1024 * 1024)


@pytest.mark.parametrize("workers,error_type,nested", [
    (1, DetectorAbstained, "cause"), (3, RegexTimeoutError, "context"),
], ids=["inline-abstain-cause", "pool-timeout-context"])
def test_saved_failures_release_finished_batch_frames(
    tmp_path, monkeypatch, workers, error_type, nested,
):
    _scoring_runtime(tmp_path, monkeypatch)
    refs = []
    fn = FeatureFunction(name="flag", description="", source="def flag(item): return False")

    def track():
        payload = _Payload()
        refs.append(weakref.ref(payload))
        return payload

    def detector(source, item, **kwargs):
        # This object is only reachable from the completed detector's frame.
        local_payload = track()
        assert local_payload.trace
        try:
            raise ValueError("inner failure")
        except ValueError as inner:
            if nested == "cause":
                raise error_type("missing evidence") from inner
            raise error_type("missing evidence") from None

    monkeypatch.setattr(tools_eval, "run_detector", detector)

    def score_batch():
        pairs = [(fn, {"_item_id": str(i), "trace_payload": track()}) for i in range(4)]
        scored = tools_eval._score_items_concurrently(pairs, workers=workers, llm=None)
        # The sampled evaluator retains scalar verdicts until reweighting, not
        # the original row. Keeping exceptions here used to retain all rows.
        return [(str(item["_item_id"]), pred, error) for _, item, pred, error in scored]

    verdicts = score_batch()
    gc.collect()
    assert len(refs) == 8 and all(ref() is None for ref in refs)
    assert len(verdicts) == 4
    for _, pred, error in verdicts:
        assert pred is None and isinstance(error, error_type)
        assert str(error) == "missing evidence"
        assert error.__traceback__ is error.__context__ is error.__cause__ is None


def test_detachment_breaks_cyclic_exception_chains():
    first, second = ValueError("first"), ValueError("second")
    first.__cause__ = second
    second.__context__ = first
    assert tools_eval._detach_detector_error(first) is first
    assert first.__cause__ is None
    assert second.__context__ is None


@pytest.mark.parametrize("workers", [1, 3])
def test_detached_outcomes_keep_abstentions_and_execution_errors_distinct(
    tmp_path, monkeypatch, workers,
):
    _scoring_runtime(tmp_path, monkeypatch)
    fn = FeatureFunction(name="flag", description="", source="def flag(item): return False")

    def detector(source, item, **kwargs):
        if item["_item_id"] == "abstain":
            raise DetectorAbstained("Evidence missing")
        if item["_item_id"] == "error":
            raise RegexTimeoutError("regex timed out")
        return True

    monkeypatch.setattr(tools_eval, "run_detector", detector)
    pairs = [(fn, {"_item_id": key, "is_correct": False})
             for key in ("abstain", "error", "positive")]
    scored = tools_eval._score_items_concurrently(pairs, workers=workers, llm=None)
    acc = StreamingBucketAccumulator()
    tools_eval._apply_llm_results(
        scored, accs={fn.name: acc}, within_accs={}, parent_fn=None, parent_fires_by_id={},
    )
    assert acc.n_items == acc.n_abstained == acc.errors == 1
    assert acc.finalize()[0]["n_fail_caught"] == 1
    assert "RegexTimeoutError: regex timed out" in str(acc.error_examples)


@pytest.mark.parametrize("workers", [1, 3])
def test_transient_errors_still_interrupt_with_original_cause_and_reuse_checkpoints(
    tmp_path, monkeypatch, workers,
):
    _scoring_runtime(tmp_path, monkeypatch)
    fn = FeatureFunction(name="flag", description="", kind="llm",
                         source="def flag(item): return False")
    calls = Counter()
    lock = threading.Lock()

    def detector(source, item, **kwargs):
        key = item["_item_id"]
        with lock:
            calls[key] += 1
            attempt = calls[key]
        if key == "two" and attempt == 1:
            raise ConnectionResetError("connection reset")
        return True

    monkeypatch.setattr(tools_eval, "run_detector", detector)
    pairs = [(fn, {"_item_id": key}) for key in ("one", "two")]
    with pytest.raises(RunInterruptedError) as interrupted:
        tools_eval._score_items_concurrently(pairs, workers=workers, llm=None)
    assert isinstance(interrupted.value.__cause__, ConnectionResetError)
    assert interrupted.value.__cause__.__traceback__ is not None
    scored = tools_eval._score_items_concurrently(pairs, workers=workers, llm=None)
    assert all(pred is True and error is None for _, _, pred, error in scored)
    assert calls == {"one": 1, "two": 2}


class _ScanFinished(Exception):
    pass


def test_judge_preserves_exact_phase_boundaries_when_extending_repair_scan(tmp_path, monkeypatch):
    from autofa.session import repair_workflow

    rt = _scoring_runtime(tmp_path, monkeypatch)
    # The extended scan limit ends inside the last four-row batch.
    rt.config.max_eval_limit = scan_limit = 11
    rt.run.eval_limit = 5
    fn = FeatureFunction(name="flag", description="", source="def flag(item): return True")
    rt.library = FunctionLibrary(functions=[fn])
    rows = [{"_item_id": str(i), "is_correct": False} for i in range(12)]
    closed, scan_buffers = [], []

    def iter_batches(**kwargs):
        arrow_buffer = _Payload()
        scan_buffers.append(weakref.ref(arrow_buffer))
        try:
            yield from (rows[i:i + 4] for i in range(0, len(rows), 4))
        finally:
            closed.append(True)

    rt.dataset = SimpleNamespace(iter_batches=iter_batches)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    monkeypatch.setattr(tools_eval, "_prefilter_decision", lambda *args: {"status": "review"})
    monkeypatch.setattr(repair_workflow, "validation_context", lambda *args, **kwargs: {
        "exclusion_item_ids": [str(i) for i in range(5)],
        "judge_objective_strategy": "balanced", "judge_behavior_strictness": "causal",
    })
    calls = []

    def score(pairs, **kwargs):
        rows = list(pairs)
        calls.append([item["_item_id"] for _, item in rows])
        yield from ((fn, item, True, None) for fn, item in rows)

    def stop_before_paid_review(*args, **kwargs):
        assert closed == [True]
        assert all(ref() is None for ref in scan_buffers)
        raise _ScanFinished

    monkeypatch.setattr(tools_eval, "_iter_scored_items", score)
    monkeypatch.setattr(tools_eval, "sample_fire_bucket_evidence", stop_before_paid_review)
    with pytest.raises(_ScanFinished):
        tools_eval.judge_function(fn.name)
    assert [len(batch) for batch in calls] == [5, scan_limit - 5]
    assert [item_id for batch in calls for item_id in batch] == [str(i) for i in range(scan_limit)]


@pytest.mark.parametrize("workers", [1, 4])
def test_streaming_scoring_releases_consumed_rows(tmp_path, monkeypatch, workers):
    _scoring_runtime(tmp_path, monkeypatch)
    refs = []
    fn = FeatureFunction(name="flag", description="", source="def flag(item): return True")

    def pairs():
        for i in range(80):
            payload = _Payload()
            refs.append(weakref.ref(payload))
            yield fn, {"_item_id": str(i), "trace_payload": payload}

    monkeypatch.setattr(tools_eval, "run_detector", lambda *args, **kwargs: True)
    scored = tools_eval._iter_scored_items(pairs(), workers=workers, llm=None)
    for i, (_, item, pred, error) in enumerate(scored):
        assert pred is True and error is None
        assert item["_item_id"] == str(i)
        if i >= 20:
            assert refs[i - 20]() is None
    item = None
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_explicit_small_chunk_size_survives_eval_dispatch(tmp_path, monkeypatch):
    rt = _scoring_runtime(tmp_path, monkeypatch)
    fn = FeatureFunction(name="flag", description="", source="def flag(item): return True")
    rt.library = FunctionLibrary(functions=[fn])
    monkeypatch.setattr(tools_eval, "_evaluate_streaming", lambda *args, **kwargs: kwargs)
    assert tools_eval.evaluate_functions(mode="full", chunk_size=2)["chunk_size"] == 2


def test_judge_scan_keeps_workers_busy_across_single_row_reads(tmp_path, monkeypatch):
    from autofa.metrics.claim_contract import claim_hash
    from autofa.session import repair_workflow

    rt = _scoring_runtime(tmp_path, monkeypatch)
    rt.run.run_id = "single-row-review"
    rt.config.llm_eval_workers = 4
    rt.run.eval_limit = rt.config.max_eval_limit = 8
    fn = FeatureFunction(name="flag", description="", kind="llm",
                         source="def flag(item): return True")
    fingerprint = claim_hash(source=fn.source, description="", mechanism=fn.name)
    # Development review requires a current evaluation panel. Start without
    # checkpoint files so the review still exercises concurrent detector work.
    fn.scores.update(
        evaluated=True, evaluation_run_id=rt.run.run_id, evaluation_claim_hash=fingerprint,
        evaluation_item_ids=[str(i) for i in range(8)], sampled=True,
        n_scored=8, sample_n=8, sample_frame_n=12,
    )
    rt.library = FunctionLibrary(functions=[fn])
    closed, seen = [], []
    gate, lock = threading.Event(), threading.Lock()

    def batches(**kwargs):
        try:
            for i in range(12):
                yield [{"_item_id": str(i), "is_correct": False}]
        finally:
            closed.append(True)

    def detector(source, item, **kwargs):
        with lock:
            seen.append(item["_item_id"])
            if len(seen) == 4:
                gate.set()
        assert gate.wait(5), "Judge scan serialized single-row reads"
        return True

    def stop_before_review(pool, **kwargs):
        assert closed == [True]
        # Scan all eight concurrently, but retain only the bounded review
        # reservoir rather than pinning every full trajectory until judging.
        assert len(pool) == kwargs["n"] == 3
        assert {item["_item_id"] for item, _ in pool} <= {str(i) for i in range(8)}
        raise _ScanFinished

    rt.dataset = SimpleNamespace(iter_batches=batches)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    monkeypatch.setattr(tools_eval, "_prefilter_decision", lambda *args: {
        "status": "review", "claim_hash": fingerprint,
    })
    monkeypatch.setattr(repair_workflow, "validation_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(tools_eval, "run_detector", detector)
    monkeypatch.setattr(tools_eval, "sample_fire_bucket_evidence", stop_before_review)
    with pytest.raises(_ScanFinished):
        tools_eval.judge_function(fn.name)
    assert sorted(seen) == [str(i) for i in range(8)]


def test_fallback_selected_scan_closes_custom_reader_on_early_exit():
    class Batches:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return [{"_item_id": "one"}, {"_item_id": "two"}]

        def close(self):
            self.closed = True

    batches = Batches()
    dataset = SimpleNamespace(iter_batches=lambda **kwargs: batches)
    rows = tools_eval._fetch_items_by_id(dataset, {"one", "two"}, max_items=None)
    assert next(rows)["_item_id"] == "one"
    rows.close()
    assert batches.closed
