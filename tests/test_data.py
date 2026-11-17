# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet datasets and tools load, sample and stream rows within memory and payload bounds."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import weakref
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.data import parquet_store as store
from autofa.data.gym_adapter import _from_parsed
from autofa.data.parquet_store import (
    Dataset,
    get_item,
    infer_models_benchmarks,
    item_memory_bytes,
    make_item_id,
    sample_items,
    truncate_item,
)
from autofa.data.row_index import RowLocation
from autofa.diagnostics import Diagnostics
from autofa.mcp import tools_data
from autofa.mcp.runtime import Runtime
from autofa.mcp.server import (
    MCP_HARD_CEILING_CHARS,
    MCP_SOFT_WARN_CHARS,
    clamp_mcp_payload,
)
from autofa.metrics.association import StreamingBucketAccumulator
from autofa.util import memory
from autofa.util.memory import (
    MemoryInfo,
    apply_memory_tuning,
    estimate_bytes_per_row,
    read_memory,
    row_group_bytes_per_row,
    suggest_chunk_size,
)


def _write_tiny_parquet(path: Path, n: int = 25, *, split: int = 15,
                        row_group_size: int | None = None) -> None:
    pq.write_table(pa.table({
        "model_id": ["m"] * n,
        "benchmark": ["bench-a" if i < split else "bench-b" for i in range(n)],
        "problem_id": [str(i) for i in range(n)],
        "run_idx": [0] * n,
        "is_correct": [i % 3 != 0 for i in range(n)],
        "response_text": [f"answer {i} " + ("ok" * 20) for i in range(n)],
        "reasoning_text": [f"think {i}" for i in range(n)],
    }), path, row_group_size=row_group_size)


def _write(path: Path, texts: list[str], *, row_group_size: int | None = None) -> None:
    rows = [
        {"model_id": "m", "benchmark": "b", "problem_id": str(i), "run_idx": 0,
         "response_text": text, "is_correct": bool(i % 2)}
        for i, text in enumerate(texts)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


# --- Sampling, truncation and indexed reads ---

def test_truncate_and_id():
    item = {"prompt_text": "a" * 50 + "MID" + "b" * 50, "response_text": "ok", "model_id": "m",
            "benchmark": "b", "problem_id": "1", "run_idx": 0, "_item_id": "m|b|1|0"}
    trunc = truncate_item(item, max_chars=20)
    text, meta = trunc["prompt_text"], trunc["prompt_text__meta"]
    assert text.startswith("a") and text.endswith("b") and "MID" not in text
    assert (meta["truncated"], meta["total_chars"]) == (True, 103)
    assert "read_field_chunk" in meta["full_read"]
    assert make_item_id(item) == "m|b|1|0"


def test_truncate_trajectory_digest():
    steps = [
        {"source": "system", "message": "You are an agent."},
        {"source": "user", "message": "Fix the bug."},
        {"source": "agent", "reasoning_content": "I will inspect the repo first.",
         "tool_calls": [{"function_name": "terminal",
                         "arguments": {"command": "ls", "is_input": False}}],
         "observation": "file.py\nREADME.md\n"},
        {"source": "agent", "reasoning_content": "Now patch the file.", "message": "Applying fix.",
         "tool_calls": [{"function": {
             "name": "file_editor", "arguments": '{"path": "file.py", "command": "str_replace"}',
         }}]},
    ]
    raw = json.dumps([{
        "schema_version": "ATIF-v1.5",
        "agent": {"name": "OpenHands-SDK", "tool_definitions": [
            {"function": {"name": "terminal"}}, {"function": {"name": "file_editor"}},
        ]},
        "steps": steps, "final_metrics": {"total_prompt_tokens": 100},
    }])
    trunc = truncate_item({"_item_id": "m|b|1|0", "response_text": raw, "prompt_text": "fix it"},
                          max_chars=80)
    text = trunc["response_text"]
    assert text.lower().startswith("[trajectory digest")
    assert "schema_version" not in text


def test_sample_items_reaches_every_benchmark(tmp_path: Path):
    # Rows are ordered by benchmark so prefix sampling would miss later benches.
    path = tmp_path / "ordered.parquet"
    benches = ["bench-early", "bench-mid", "bench-late"]
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "m", "benchmark": bench, "problem_id": f"{bench}-{i}", "run_idx": 0,
         "is_correct": i % 2 == 0, "response_text": f"answer for {bench} {i} " + ("x" * 40),
         "reasoning_text": ""}
        for bench in benches for i in range(30)
    ]), path, row_group_size=15)  # Small row groups so random access spans groups.
    ds = Dataset(parquet_paths=[path], model_id="m")
    out = sample_items(ds, n_fail=9, n_pass=9, prefer_informative=False, by_benchmark=True,
                       seed=7, max_total=40)
    assert {str(it.get("benchmark")) for it in out["items"]} == set(benches)


def test_get_item_via_identity_index(tmp_path: Path):
    path = tmp_path / "indexed.parquet"
    _write_tiny_parquet(path, n=100, split=50, row_group_size=10)
    item_id = "m|bench-b|99|0"
    result = get_item(Dataset(parquet_paths=[path]), item_id, max_chars=500)
    assert result["ok"] is True and result["item"]["_item_id"] == item_id
    assert "answer 99" in result["item"]["response_text"]
    assert (path.parent / f"{path.stem}.autofa_id_index.json").is_file()


def test_iter_batches_and_peek_chunk_apply_the_benchmark_filter(tmp_path: Path):
    path = tmp_path / "tiny.parquet"
    _write_tiny_parquet(path, n=25)
    ds = Dataset(parquet_paths=[path], model_id="m", benchmarks=["bench-a"])
    batches = list(ds.iter_batches(batch_size=5))
    assert sum(len(b) for b in batches) == 15  # only bench-a
    assert all(len(b) <= 5 for b in batches)
    assert all(it["benchmark"] == "bench-a" for b in batches for it in b)
    assert ds.peek_chunk(2, chunk_size=5, max_chars=40)["n"] == 5
    peek_past = ds.peek_chunk(99, chunk_size=5)
    assert peek_past["n"] == 0 and peek_past["exhausted"] is True


def test_streaming_accumulator_keeps_all_fail_ids():
    preds = ["ok", "ok"] + ["bug"] * 10 + ["ok"]
    labels = [True, True] + [False] * 10 + [True]
    acc = StreamingBucketAccumulator()  # unlimited fails by default
    for i, (pred, ok_label) in enumerate(zip(preds, labels, strict=True)):
        acc.update(pred, is_correct=ok_label, item_id=f"id-{i}")
    stream_rows = {r["bucket"]: r for r in acc.finalize()}
    assert len(stream_rows["bug"]["example_fail_ids"]) == 10


# --- Memory: summaries, streaming scans and indexed reads ---

def _observe(monkeypatch):
    """Record parquet opens, closes, batch reads and Python conversions; forbid group reads."""
    original = pq.ParquetFile
    state = {"opens": [], "closes": [], "schemas": [], "batches": [], "rows": [],
             "conversions": []}

    class TrackedRow(dict):
        pass

    class ObservedBatch:
        def __init__(self, batch):
            self.batch = batch

        def __getattr__(self, name):
            return getattr(self.batch, name)

        def slice(self, *args, **kwargs):
            return ObservedBatch(self.batch.slice(*args, **kwargs))

        def take(self, *args, **kwargs):
            return ObservedBatch(self.batch.take(*args, **kwargs))

        def to_pylist(self):
            state["conversions"].append((self.batch.num_rows, self.batch.nbytes))
            rows = [TrackedRow(row) for row in self.batch.to_pylist()]
            state["rows"].extend(weakref.ref(row) for row in rows)
            return rows

    class ObservedParquet:
        def __init__(self, path, *args, **kwargs):
            self.path = Path(path)
            self.file = original(path, *args, **kwargs)
            state["opens"].append(self.path)

        def __getattr__(self, name):
            return getattr(self.file, name)

        @property
        def schema_arrow(self):
            state["schemas"].append(self.path)
            return self.file.schema_arrow

        def iter_batches(self, **kwargs):
            state["batches"].append(kwargs)
            for batch in self.file.iter_batches(**kwargs):
                yield ObservedBatch(batch)

        def close(self):
            state["closes"].append(self.path)
            self.file.close()

        def read_row_group(self, *args, **kwargs):
            raise AssertionError("point reads must not materialize an entire row group")

    monkeypatch.setattr(pq, "ParquetFile", ObservedParquet)
    return state


def test_summary_streams_exact_statistics_without_materializing_corpus(tmp_path, monkeypatch):
    path = tmp_path / "stats.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "z", "benchmark": "a", "problem_id": "0", "is_correct": True,
         "generation": "", "reasoning_text": "r"},
        {"model_id": "a", "benchmark": "b", "problem_id": "1", "is_correct": False,
         "generation": "abc", "reasoning_text": None},
        {"model_id": "z", "benchmark": "a", "problem_id": "2", "is_correct": False,
         "generation": "x" * 100, "reasoning_text": "rrr"},
        {"model_id": "a", "benchmark": "b", "problem_id": "3", "is_correct": True,
         "generation": "x" * 11, "reasoning_text": ""},
        {"model_id": "a", "benchmark": "b", "problem_id": "4", "is_correct": False,
         "generation": "x" * 5, "reasoning_text": "rr"},
    ]), path)
    dataset = Dataset([path])

    def forbidden(**kwargs):
        raise AssertionError("info must not call load_items")

    monkeypatch.setattr(dataset, "load_items", forbidden)
    info = dataset.info()
    assert (info["n_items"], info["n_fail"], info["n_pass"], info["fail_rate"]) == (5, 3, 2, 0.6)
    assert info["models"] == ["a", "z"]
    assert list(info["benchmarks"]) == ["b", "a"]
    assert info["benchmarks"]["b"] == {"n": 3, "fail": 2, "fail_rate": 0.6667}
    assert info["text_field_stats"]["response_text"] == {
        "nonempty_pct": 80.0, "median_len": 11, "p95_len": 11,
    }
    assert dataset.info(scan_limit=2)["n_items"] == 2


def test_summary_and_deprecated_loader_release_rows(monkeypatch):
    class TrackedRow(dict):
        pass

    dataset = Dataset([])
    refs = []
    calls = []

    def batches(*, max_items=None, **kwargs):
        calls.append(max_items)
        for i in range(min(20, max_items or 20)):
            row = TrackedRow(problem_id=str(i), response_text="x" * 1000)
            refs.append(weakref.ref(row))
            yield [row]

    monkeypatch.setattr(dataset, "iter_batches", batches)
    assert dataset.info(scan_limit=10)["n_items"] == 10
    gc.collect()
    assert all(ref() is None for ref in refs)
    items = dataset.load_items(limit=3)
    assert len(items) == 3
    del items
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert calls == [10, 3]


def test_streaming_batches_bound_actual_bytes_and_keep_large_traces_intact(tmp_path, monkeypatch):
    path = tmp_path / "large.parquet"
    texts = ["a" * 1000, "🍋" * 5000, "b" * 9000, "c" * 80_000,
             "d" * 9000, "e" * 1000, "f" * 9000]
    _write(path, texts)
    monkeypatch.setattr(store, "MAX_BATCH_BYTES", 32 * 1024)
    # Simulate compressed/dictionary metadata that misses a large outlier.
    monkeypatch.setattr("autofa.util.memory.row_group_bytes_per_row", lambda *args: 1024)
    batches = list(Dataset([path]).iter_batches(batch_size=20))
    assert [row["response_text"] for batch in batches for row in batch] == texts
    assert len(batches) > 1
    for batch in batches:
        assert len(batch) == 1 or sum(item_memory_bytes(row) for row in batch) <= 32 * 1024
    assert any(len(batch) == 1 and len(batch[0]["response_text"]) == 80_000 for batch in batches)


def test_arrow_rows_respect_caller_cap_metadata_cost_and_projection(tmp_path, monkeypatch):
    path = tmp_path / "large.parquet"
    _write(path, [str(i) + "x" * 80_000 for i in range(8)], row_group_size=4)
    monkeypatch.setattr(store, "MAX_BATCH_BYTES", 64 * 1024)
    calls = _observe(monkeypatch)["batches"]
    dataset = Dataset([path])
    assert len(list(dataset.iter_batches(batch_size=1, max_items=1))) == 1
    assert [call["batch_size"] for call in calls] == [1]
    calls.clear()
    batches = list(dataset.iter_batches(batch_size=100))
    assert sum(map(len, batches)) == 8
    assert all(call["batch_size"] == 1 for call in calls)
    assert [call["row_groups"] for call in calls] == [[0], [1]]
    calls.clear()
    projected = list(dataset.iter_batches(
        batch_size=100, columns=["model_id", "benchmark", "problem_id", "run_idx"],
    ))
    assert sum(map(len, projected)) == 8
    assert all(call["batch_size"] > 1 for call in calls)
    assert all("response_text" not in row for batch in projected for row in batch)


def test_peek_chunk_assembles_logical_window_across_byte_limited_batches(tmp_path, monkeypatch):
    path = tmp_path / "peek.parquet"
    _write(path, [str(i) + "x" * 20_000 for i in range(11)])
    monkeypatch.setattr(store, "MAX_BATCH_BYTES", 32 * 1024)
    dataset = Dataset([path])
    first = dataset.peek_chunk(0, chunk_size=4, max_chars=20)
    middle = dataset.peek_chunk(1, chunk_size=4, max_chars=20)
    last = dataset.peek_chunk(2, chunk_size=4, max_chars=20)
    assert [row["problem_id"] for row in first["items"]] == ["0", "1", "2", "3"]
    assert [row["problem_id"] for row in middle["items"]] == ["4", "5", "6", "7"]
    assert [row["problem_id"] for row in last["items"]] == ["8", "9", "10"]
    assert first["exhausted"] is False and last["exhausted"] is True
    assert all(len(row["response_text"]) < 200 for row in middle["items"])


def test_indexed_reads_keep_group_scope_and_request_order(tmp_path, monkeypatch):
    path = tmp_path / "indexed.parquet"
    texts = [str(i) + "x" * 20_000 for i in range(12)]
    _write(path, texts, row_group_size=4)
    monkeypatch.setattr(store, "MAX_BATCH_BYTES", 32 * 1024)
    calls = _observe(monkeypatch)["batches"]
    dataset = Dataset([path])
    locations = [RowLocation(path, 1, 3), RowLocation(path, 0, 1),
                 RowLocation(path, 1, 3), RowLocation(path, 2, 99)]
    rows = dataset.read_locations(locations)
    assert [row["response_text"] for row in rows] == [texts[7], texts[1], texts[7]]
    assert rows[0] is not rows[2]
    assert [call["row_groups"] for call in calls] == [[1], [0]]
    assert all(call["batch_size"] == 1 for call in calls)


def test_python_byte_estimate_does_not_double_count_aliases():
    value = "x" * 10_000
    shared = {"response_text": value, "generation": value}
    distinct = {"response_text": value, "generation": "y" * 10_000}
    assert item_memory_bytes(distinct) - item_memory_bytes(shared) >= 10_000


def test_missing_projection_on_a_shard_does_not_load_trace_columns(tmp_path, monkeypatch):
    first = tmp_path / "ids.parquet"
    second = tmp_path / "traces.parquet"
    pq.write_table(pa.table({"problem_id": ["a", "b"]}), first)
    pq.write_table(pa.table({"response_text": ["x" * 80_000, "y" * 80_000]}), second)
    calls = _observe(monkeypatch)["batches"]
    dataset = Dataset([first, second])
    rows = [row for batch in dataset.iter_batches(columns=["problem_id"]) for row in batch]
    assert len(rows) == 4
    assert [row.get("problem_id") for row in rows] == ["a", "b", None, None]
    assert all("response_text" not in row for row in rows)
    assert [call["columns"] for call in calls] == [["problem_id"], []]


def test_dictionary_encoded_traces_are_byte_sliced_before_python_conversion(tmp_path, monkeypatch):
    path = tmp_path / "dictionary.parquet"
    text = "trace" * 4000
    _write(path, [text] * 32)
    group = pq.ParquetFile(path).metadata.row_group(0)
    response = next(group.column(i) for i in range(group.num_columns)
                    if group.column(i).path_in_schema == "response_text")
    assert "RLE_DICTIONARY" in response.encodings
    monkeypatch.setattr(store, "MAX_BATCH_BYTES", 128 * 1024)
    observed = _observe(monkeypatch)
    calls, conversions = observed["batches"], observed["conversions"]
    dataset = Dataset([path])
    rows = [row for batch in dataset.iter_batches(batch_size=32) for row in batch]
    assert [row["response_text"] for row in rows] == [text] * 32
    assert calls[0]["batch_size"] > max(n for n, _ in conversions)
    assert all(n == 1 or byte_count * 4 <= store.MAX_BATCH_BYTES
               for n, byte_count in conversions)

    conversions.clear()
    rows = dataset.read_locations([RowLocation(path, 0, i) for i in range(32)])
    assert [row["problem_id"] for row in rows] == [str(i) for i in range(32)]
    assert [row["response_text"] for row in rows] == [text] * 32
    assert all(n == 1 or byte_count * 4 <= store.MAX_BATCH_BYTES
               for n, byte_count in conversions)


def test_python_conversion_rechecks_skewed_slice_bytes(tmp_path, monkeypatch):
    path = tmp_path / "skewed.parquet"
    texts = ["a"] * 4 + ["x" * 30_000] * 2 + ["b"] * 2
    _write(path, texts)
    monkeypatch.setattr(store, "MAX_BATCH_BYTES", 128 * 1024)
    conversions = _observe(monkeypatch)["conversions"]
    rows = [row for batch in Dataset([path]).iter_batches(batch_size=8) for row in batch]
    assert [row["response_text"] for row in rows] == texts
    assert any(n > 1 for n, _ in conversions)
    assert any(n == 1 for n, _ in conversions)
    assert all(n == 1 or byte_count * 4 <= store.MAX_BATCH_BYTES
               for n, byte_count in conversions)


# --- Indexed row-location streaming reuses readers and releases yielded rows ---

def _write_grouped_traces(path: Path, *, rows: int = 12, model: str = "m", benchmark: str = "b"):
    pq.write_table(pa.Table.from_pylist([
        {"model_id": model, "benchmark": benchmark, "problem_id": str(i),
         "run_idx": 0, "generation": f"{i}:" + "x" * 1000,
         "messages": [{"role": "assistant", "content": str(i)}]}
        for i in range(rows)
    ]), path, row_group_size=4)


def test_indexed_stream_preserves_arbitrary_order_and_independent_duplicates(tmp_path, monkeypatch):
    path = tmp_path / "sample.parquet"
    _write_grouped_traces(path)
    monkeypatch.setattr(store, "MAX_BATCH_BYTES", 32 * 1024)
    observed = _observe(monkeypatch)
    locations = [RowLocation(path, 1, 3), RowLocation(path, 1, 0),
                 RowLocation(path, 1, 0), RowLocation(path, 0, 2),
                 RowLocation(path, 50, 0), RowLocation(path, 0, -1),
                 RowLocation(path, 0, 4)]
    rows = list(Dataset([path]).iter_locations(locations))
    assert [row["problem_id"] for row in rows] == ["7", "4", "4", "2"]
    rows[1]["messages"][0]["content"] = "changed"
    assert rows[2]["messages"][0]["content"] == "4"
    assert observed["opens"] == observed["schemas"] == observed["closes"] == [path]


def test_indexed_stream_filters_and_aliases(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in (
        "first.parquet", "other_model.parquet", "other_bench.parquet",
    )]
    _write_grouped_traces(paths[0], rows=1)
    _write_grouped_traces(paths[1], rows=1, model="other")
    _write_grouped_traces(paths[2], rows=1, benchmark="other")
    observed = _observe(monkeypatch)
    locations = [RowLocation(path, 0, 0) for path in paths]
    dataset = Dataset(paths, model_id="m", benchmarks=["b"])
    rows = list(dataset.iter_locations(locations))
    assert len(rows) == 1
    assert rows[0]["response_text"] == rows[0]["generation"] == "0:" + "x" * 1000
    assert observed["opens"] == observed["schemas"] == observed["closes"] == paths


def test_indexed_stream_closes_on_early_exit_or_processing_failure(tmp_path, monkeypatch):
    path = tmp_path / "sample.parquet"
    _write_grouped_traces(path)
    observed = _observe(monkeypatch)

    def locations():
        yield RowLocation(path, 0, 0)
        pytest.fail("early close must not consume the next request")

    rows = Dataset([path]).iter_locations(locations())
    assert next(rows)["problem_id"] == "0"
    assert observed["closes"] == []
    rows.close()
    assert observed["closes"] == [path]
    gc.collect()
    assert all(ref() is None for ref in observed["rows"])

    def fail(row, path):
        raise ValueError("broken normalization")

    dataset = Dataset([path])
    monkeypatch.setattr(dataset, "_make_item_id", fail)
    with pytest.raises(ValueError, match="broken normalization"):
        list(dataset.iter_locations([RowLocation(path, 0, 0)]))
    assert observed["closes"] == [path, path]


def test_indexed_stream_decodes_forward_once_without_retaining_yielded_rows(
    tmp_path, monkeypatch,
):
    path = tmp_path / "sample.parquet"
    _write_grouped_traces(path)
    monkeypatch.setattr(store, "MAX_BATCH_BYTES", 32 * 1024)
    observed = _observe(monkeypatch)
    rows = Dataset([path]).iter_locations(
        RowLocation(path, i // 4, i % 4) for i in range(12)
    )
    for i in range(12):
        row = next(rows)
        assert row["problem_id"] == str(i)
        assert sum(ref() is not None for ref in observed["rows"]) == 1
        del row
    rows.close()
    assert all(ref() is None for ref in observed["rows"])
    assert observed["opens"] == observed["schemas"] == observed["closes"] == [path]
    assert [kwargs["row_groups"] for kwargs in observed["batches"]] == [[0], [1], [2]]
    assert all(kwargs["batch_size"] <= 2 for kwargs in observed["batches"])


def test_materialized_reads_reuse_each_file_and_restore_interleaved_order(tmp_path, monkeypatch):
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    _write_grouped_traces(first)
    _write_grouped_traces(second)
    observed = _observe(monkeypatch)
    rows = Dataset([first, second]).read_locations([
        RowLocation(first, 1, 1), RowLocation(second, 2, 2),
        RowLocation(first, 0, 0), RowLocation(second, 1, 2),
    ])
    assert [row["problem_id"] for row in rows] == ["5", "10", "0", "6"]
    assert observed["opens"] == observed["schemas"] == observed["closes"] == [first, second]


def test_single_early_point_read_does_not_decode_a_full_batch(tmp_path, monkeypatch):
    path = tmp_path / "sample.parquet"
    _write_grouped_traces(path)
    observed = _observe(monkeypatch)
    row, = Dataset([path]).read_locations([RowLocation(path, 0, 0)])
    assert row["problem_id"] == "0"
    assert observed["batches"][0]["batch_size"] == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_indexed_reads_accept_string_location_values(tmp_path, monkeypatch, streaming):
    path = tmp_path / "sample.parquet"
    _write_grouped_traces(path)
    observed = _observe(monkeypatch)
    dataset = Dataset([path])
    locations = [RowLocation(str(path), "1", "2"), RowLocation(path, 0, 0),
                 RowLocation(str(path), "1", "2")]
    read = dataset.iter_locations if streaming else dataset.read_locations
    rows = list(read(locations))
    assert [row["problem_id"] for row in rows] == ["6", "0", "6"]
    assert rows[0] is not rows[2]
    assert observed["opens"] == observed["schemas"] == observed["closes"] == [path]


# --- Memory probes size chunks from RAM and parquet row-group estimates ---

@pytest.mark.parametrize(("available", "row_bytes", "expected"), [
    (128 * 1024**2, 96 * 1024, 128),  # rounds down to fit 18% of RAM after the reserve
    (16 * 1024**3, 32 * 1024, 4096),  # clamped to max_size
    (128 * 1024**3, 8 * 1024**2, 32),  # abundant RAM never builds a gigabyte window
    (1024**2, 2 * 1024**2, 1),  # low RAM has no minimum of 64 rows or artificial budget
    (0, 2 * 1024**2, 1),  # zero available RAM is exhausted, not unknown
    (None, 96 * 1024, 256),  # unknown RAM uses a conservative 32 MiB window
])
def test_suggest_chunk_size_fits_memory_budget(available, row_bytes, expected):
    assert suggest_chunk_size(available_bytes=available, bytes_per_row=row_bytes) == expected


def test_apply_auto_then_refine_stays_auto(monkeypatch):
    monkeypatch.setattr(memory, "read_memory", lambda: MemoryInfo(2 * 1024**3, 4 * 1024**3, "proc"))
    cfg = SimpleNamespace(eval_chunk_size=0)
    assert apply_memory_tuning(cfg).source == "auto"
    first = cfg.eval_chunk_size
    monkeypatch.setattr(memory, "estimate_bytes_per_row", lambda paths: 8 * 1024**2)
    assert apply_memory_tuning(cfg, parquet_paths=[]).source == "auto"
    assert 0 < cfg.eval_chunk_size < first


def test_apply_pinned_config(monkeypatch):
    monkeypatch.setattr(memory, "read_memory", lambda: MemoryInfo(0, 4 * 1024**3, "proc"))
    cfg = SimpleNamespace(eval_chunk_size=256)
    # Later calls must not override a pin even with parquet paths.
    for paths in (None, []):
        assert apply_memory_tuning(cfg, parquet_paths=paths).source == "config"
        assert cfg.eval_chunk_size == 256


def test_linux_zero_available_does_not_fall_back_to_memfree(monkeypatch):
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_a, **_kw: "MemAvailable: 0 kB\nMemFree: 1024 kB\nMemTotal: 2048 kB\n",
    )
    assert read_memory() == MemoryInfo(0, 2048 * 1024, "proc")


def _no_proc(monkeypatch):
    def absent(*_a, **_kw):
        raise FileNotFoundError

    monkeypatch.setattr(Path, "read_text", absent)


VM_STAT = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"


@pytest.mark.parametrize(("vm_output", "available", "source"), [
    (VM_STAT + "Pages free: 100.\nPages inactive: 300.\nPages speculative: 50.\n"
     "Pages purgeable: 60.\nFile-backed pages: 200.\nPages occupied by compressor: 2000.\n",
     300 * 16384, "vm_stat"),
    # Unknown or zero availability is never inferred as half of total RAM.
    ("malformed", None, "unknown"),
    (VM_STAT + "Pages free: 0.\n", 0, "vm_stat"),
])
def test_darwin_counts_free_and_reclaimable_pages_without_double_counting(
    monkeypatch, vm_output, available, source,
):
    _no_proc(monkeypatch)
    monkeypatch.setattr(memory.sys, "platform", "darwin")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["timeout"] == 2
        return SimpleNamespace(
            stdout=str(48 * 1024**3) if args[0].endswith("sysctl") else vm_output
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert read_memory() == MemoryInfo(available, 48 * 1024**3, source)
    assert len(calls) == 2


def test_darwin_probe_timeout_is_bounded_fallback(monkeypatch):
    _no_proc(monkeypatch)
    monkeypatch.setattr(memory.sys, "platform", "darwin")

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    assert read_memory() == MemoryInfo(None, None, "unknown")


def test_sysconf_without_available_pages_preserves_unknown(monkeypatch):
    _no_proc(monkeypatch)
    monkeypatch.setattr(memory.sys, "platform", "unsupported")

    def sysconf(name):
        if name == "SC_AVPHYS_PAGES":
            raise ValueError(name)
        return {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 8 * 1024**2}[name]

    monkeypatch.setattr(os, "sysconf", sysconf)
    assert read_memory() == MemoryInfo(None, 32 * 1024**3, "sysconf")


def test_parquet_estimate_uses_heaviest_uncompressed_row_group_in_any_shard(tmp_path):
    # One heavy, highly compressible row in its own row group of the sixth shard: an
    # estimate from compressed size, a corpus average or the first five shards is too low.
    paths = []
    for index in range(6):
        path = tmp_path / f"shard-{index}.parquet"
        rows = ["small"] * 9 + ["x" * 1024**2] if index == 5 else ["small"]
        pq.write_table(pa.table({"trace_json": rows}), path, row_group_size=1,
                       compression="zstd", use_dictionary=False)
        paths.append(path)
    assert paths[5].stat().st_size < 32 * 1024
    assert estimate_bytes_per_row(paths) >= 4 * 1024**2


def test_row_group_estimate_respects_selected_nested_columns(tmp_path):
    path = tmp_path / "nested.parquet"
    pq.write_table(pa.table({"id": [1], "trace": [{"text": "x" * 1024**2}]}), path)
    with pq.ParquetFile(path) as parquet:
        group = parquet.metadata.row_group(0)
        assert row_group_bytes_per_row(group) >= 4 * 1024**2
        assert row_group_bytes_per_row(group, ["trace"]) >= 4 * 1024**2
        assert row_group_bytes_per_row(group, ["id"]) == 16 * 1024
        assert row_group_bytes_per_row(group, []) == 16 * 1024


def test_parquet_estimate_handles_missing_corrupt_and_empty_files(tmp_path):
    corrupt = tmp_path / "broken.parquet"
    corrupt.write_bytes(b"bad parquet")
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"trace_json": pa.array([], type=pa.string())}), empty)
    assert estimate_bytes_per_row([tmp_path / "missing.parquet", corrupt, empty]) == 96 * 1024


# --- MCP benchmark selectors narrow the run's selected corpus ---

@pytest.fixture
def runtime(tmp_path, monkeypatch, request):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "autofa"))
    path = tmp_path / "mixed.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "m", "benchmark": benchmark, "problem_id": f"task-{task}",
         "run_idx": repeat, "is_correct": task == 2,
         "response_text": "Synthetic shared search evidence"}
        for benchmark in ("tau2", "banking", "outside")
        for task in range(3) for repeat in range(2)
    ]), path)
    cfg = Config()
    run = RunConfig.create([path], "m", ["tau2", "banking"], cfg)
    # Leave the backing dataset broad to verify the run boundary independently
    # of the dataset's own benchmark policy.
    rt = Runtime(dataset=Dataset(parquet_paths=[path], model_id="m"),
                 library=FunctionLibrary(), config=cfg, run=run)
    monkeypatch.setattr(tools_data, "get_runtime", lambda: rt)
    if getattr(request, "param", "indexed") == "fallback":
        monkeypatch.setattr(tools_data.ps, "_sample_via_index", lambda *args, **kwargs: None)
    return rt


def _read_tools(**selection):
    sample = tools_data.sample_items(n_fail=20, n_pass=20, prefer_informative=False,
                                     **selection)
    search = tools_data.search_items(query="shared search", limit=40, **selection)
    return sample, search


@pytest.mark.parametrize("runtime,selection", [
    ("indexed", {"benchmark": "banking"}),
    ("indexed", {"benchmark": "banking", "benchmarks": []}),  # Empty list does not override.
    ("indexed", {"benchmark": "tau2", "benchmarks": ["banking"]}),  # The list wins.
    ("fallback", {"benchmark": "banking"}),
], indirect=["runtime"], ids=["indexed", "empty-list-keeps-single", "list-overrides-single",
                              "fallback"])
def test_explicit_benchmark_filter_is_honored_by_sampling_and_search(runtime, selection):
    sample, search = _read_tools(**selection)
    for result in (sample, search):
        assert result["ok"]
        assert {row["benchmark"] for row in result["items"]} == {"banking"}
    assert (sample["n_fail"], sample["n_pass"], search["n"]) == (2, 1, 6)


def test_empty_filter_uses_the_run_scope(runtime):
    sample, search = _read_tools(benchmarks=[])
    for result in (sample, search):
        assert result["ok"]
        assert {row["benchmark"] for row in result["items"]} == {"tau2", "banking"}
    assert (sample["n_fail"], sample["n_pass"], search["n"]) == (4, 2, 12)


def test_explicit_filters_cannot_expand_the_run_scope(runtime):
    for result in _read_tools(benchmarks=["banking", "outside"]):
        assert not result["ok"] and "outside the selected run corpus" in result["error"]


def test_explicit_filter_still_respects_the_dataset_policy(runtime):
    runtime.dataset.benchmarks = ["tau2"]
    for result in _read_tools(benchmark="banking"):
        assert result["ok"] and result["items"] == []


def test_unrestricted_run_can_select_an_explicit_benchmark(runtime):
    runtime.run.benchmarks = []
    for result in _read_tools(benchmark="outside"):
        assert result["ok"]
        assert {row["benchmark"] for row in result["items"]} == {"outside"}


# --- Dataset tools preserve byte-bounded reads through their own consumers ---

class TraceRow(dict):
    __slots__ = ("__weakref__",)


class StubDataset:
    rows = 100

    def info(self, **kwargs):
        return {"n_items": self.rows}

    def estimate_matching_rows(self):
        return self.rows


def _tool_runtime(monkeypatch, dataset):
    rt = SimpleNamespace(
        dataset=dataset, run=SimpleNamespace(eval_limit=5000),
        config=SimpleNamespace(eval_chunk_size=1, feature_model="test",
                               text_truncate_chars=1000),
        llm_readiness={"available": False, "configured": False},
    )
    monkeypatch.setattr(tools_data, "get_runtime", lambda: rt)
    return rt


def test_parse_coverage_consumes_stream_without_retaining_the_sample(monkeypatch):
    refs = []
    closed = []

    class Streamed(StubDataset):
        def load_items(self, **kwargs):
            pytest.fail("A streamed summary must not fall back to eager loading")

        def iter_batches(self, *, batch_size, max_items):
            assert max_items == 40
            try:
                for i in range(max_items):
                    assert sum(ref() is not None for ref in refs) <= 2
                    row = TraceRow(_item_id=str(i), response_text="x" * 4096)
                    refs.append(weakref.ref(row))
                    yield [row]
            finally:
                closed.append(True)

    def coverage(items):
        assert not isinstance(items, list)
        return {"n_sampled": sum(1 for _ in items)}

    _tool_runtime(monkeypatch, Streamed())
    monkeypatch.setattr("autofa.data.corpus_health.measure_parse_coverage", coverage)
    result = tools_data.dataset_info()
    assert result["ok"]
    assert result["trajectory_coverage"]["n_sampled"] == 40
    assert closed == [True]
    assert all(ref() is None for ref in refs)


def test_stream_error_does_not_retry_by_loading_full_sample(monkeypatch):
    class Unavailable(StubDataset):
        def iter_batches(self, **kwargs):
            # A generator: the error is raised while dataset_info consumes the stream.
            raise OSError("dataset unavailable")
            yield  # pragma: no cover

        def load_items(self, **kwargs):
            pytest.fail("A failed stream must not trigger an eager reload")

    _tool_runtime(monkeypatch, Unavailable())
    with pytest.raises(OSError, match="dataset unavailable"):
        tools_data.dataset_info()


def test_tool_chunk_sizes_do_not_inflate_small_memory_windows(monkeypatch):
    class Chunked(StubDataset):
        def peek_chunk(self, index, *, chunk_size, max_chars):
            return {"chunk_index": index, "chunk_size": chunk_size}

    _tool_runtime(monkeypatch, Chunked())
    chunks = tools_data.dataset_chunks()
    assert chunks["chunk_size"] == 1
    assert chunks["estimated_chunks"] == 100
    assert tools_data.peek_chunk(chunk_index=3)["chunk_size"] == 1


@pytest.mark.parametrize(
    ("configured_sample", "effective_sample"), [(0, 0), (50_000, 20_000)],
)
def test_dataset_info_reports_configured_capped_policy(
    monkeypatch, configured_sample, effective_sample,
):
    class Small(StubDataset):
        rows = 10

        def info(self, **kwargs):
            return {**super().info(), "existing_field": "preserved"}

        def iter_batches(self, **kwargs):
            yield []

    rt = _tool_runtime(monkeypatch, Small())
    rt.config.llm_eval_sample = configured_sample
    result = tools_data.dataset_info()
    assert result["ok"] and result["existing_field"] == "preserved"
    assert result["default_evaluation_mode"] == "capped"
    # Zero is kept as configured; large values clamp to the bounded LLM sample.
    assert result["default_llm_eval_sample"] == effective_sample


# --- Column aliases, the gym adapter and get_item payload clamping ---

def test_column_aliases_on_load(tmp_path: Path):
    path = tmp_path / "alias.parquet"
    pq.write_table(pa.table({
        "model_id": ["m1", "m1"], "benchmark": ["tb", "tb"], "task_id": ["t0", "t1"],
        "resolved": [False, True], "instruction": ["do a", "do b"],
        "generation": ["ans a", "ans b"],
    }), path)
    items = Dataset(parquet_paths=[path], model_id="m1").load_items(limit=10)
    assert [(it["problem_id"], it["is_correct"]) for it in items] == [("t0", False), ("t1", True)]
    assert (items[0]["prompt_text"], items[0]["response_text"]) == ("do a", "ans a")
    meta = infer_models_benchmarks([path])
    assert (meta["models"][0][0], meta["benchmarks"][0][0], meta["n_fail"]) == ("m1", "tb", 1)


def test_from_parsed_preserves_observations_and_tool_args(monkeypatch):
    monkeypatch.setattr("autofa.diagnostics._DIAGNOSTICS", Diagnostics())
    obj = {"task_id": "task-obs-1", "is_resolved": False, "question": "fix the bug", "steps": [
        {"reasoning": "I should inspect the file", "message": "Looking at main.py",
         "tool_calls": [{"name": "bash", "arguments": {"command": "cat main.py"}}],
         "observation": "raise RuntimeError('boom')"},
        {"message": "Trying a patch", "tool_call_names": ["edit_file"],
         "observation": "patched ok"},
    ]}
    row = _from_parsed(obj, model_id="m", benchmark="tb", line_no=0)
    # Must not collapse to the last message and bare tool names.
    for fragment in ("Looking at main.py", "cat main.py", "RuntimeError('boom')", "patched ok"):
        assert fragment in row["response_text"]
    assert "I should inspect the file" in row["reasoning_text"]


def test_clamp_mcp_payload_hard_ceiling_returns_valid_json(monkeypatch):
    monkeypatch.setattr("autofa.diagnostics._DIAGNOSTICS", Diagnostics())
    # Without a run workspace, retain the overflow fallback rather than losing JSON.
    monkeypatch.setattr("autofa.mcp.runtime._RUNTIME", None)
    huge = {"ok": True, "blob": "x" * (MCP_HARD_CEILING_CHARS + 1000)}
    parsed = json.loads(clamp_mcp_payload(huge, "get_item"))
    assert (parsed["ok"], parsed["error"], parsed["tool"]) == (
        False, "response_too_large", "get_item",
    )
    assert parsed["total_chars"] > MCP_HARD_CEILING_CHARS
    assert "read_field_chunk" in parsed["read_with"]


def test_clamp_mcp_payload_soft_warn_keeps_full_payload(monkeypatch):
    monkeypatch.setattr("autofa.diagnostics._DIAGNOSTICS", Diagnostics())
    result = {"ok": True, "item_id": "abc", "blob": "y" * (MCP_SOFT_WARN_CHARS + 500)}
    parsed = json.loads(clamp_mcp_payload(result, "get_item"))
    assert parsed["ok"] is True and parsed["blob"] == result["blob"]
