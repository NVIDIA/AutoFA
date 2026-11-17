# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Status log reads stay bounded, turn outcomes are recorded, and MCP status tools work."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tracemalloc

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rich.console import Console

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.mcp.server import TOOL_CALLABLES
from autofa.session import status_log
from autofa.session.status_log import (
    append_status,
    bind_activity_listener,
    clear_status_listener,
    emit_runtime_status,
    latest_turn_status,
    read_status,
    record_turn_status,
    write_status_markdown,
)
from autofa.ui.activity import AgentActivity


@pytest.mark.parametrize("terminated", [True, False], ids=["legacy-json", "partial-json"])
def test_oversized_status_preserves_evidence_and_bounds_reader_memory(tmp_path, terminated):
    path = tmp_path / status_log.STATUS_JSONL
    known = {"source": "harness", "ts": "earlier",
             "context": {"turn_status": "finished"}, "message": "Earlier turn finished"}
    prefix = json.dumps(known).encode() + b"\n"
    with path.open("wb") as stream:
        stream.write(prefix)
        stream.write(b'{"source":"agent","message":"')
        for _ in range(192):
            stream.write(b"x" * status_log.STATUS_READ_CHUNK_BYTES)
        if terminated:
            stream.write(b'"}\n')
    size = path.stat().st_size
    with path.open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    tracemalloc.start()
    try:
        rows = read_status(tmp_path, limit=0)
        latest = latest_turn_status(tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The log is twelve times the record limit; readers never hold it whole.
    assert peak < 6 * status_log.MAX_STATUS_RECORD_BYTES
    assert len(rows) == 2 and rows[0] == known
    observation = rows[1]
    assert observation["context"] == {
        "status": "unavailable", "reason": "oversized_record", "artifact": str(path),
        "byte_start": len(prefix), "byte_end": size - int(terminated),
        "bytes": size - int(terminated) - len(prefix),
        "limit_bytes": status_log.MAX_STATUS_RECORD_BYTES, "newline_terminated": terminated,
    }
    assert latest["status"] == "unknown"
    assert latest["last_known_turn"] == {"status": "finished", "recorded_at": "earlier"}
    assert latest["latest_unavailable_record"] == observation
    assert latest["unavailable_records"] == 1
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == before
    assert read_status(tmp_path, limit=1) == [observation]
    assert str(path) in write_status_markdown(tmp_path).read_text()


def test_reverse_status_boundaries_and_newer_harness_receipt(tmp_path, monkeypatch):
    # Tiny chunks exercise separators at either edge and records spanning chunks.
    monkeypatch.setattr(status_log, "STATUS_READ_CHUNK_BYTES", 7)
    monkeypatch.setattr(status_log, "MAX_STATUS_RECORD_BYTES", 300)
    older = {"message": "🖼 " * 10, "source": "agent"}
    latest = {"source": "harness", "context": {"turn_status": "started"}}
    path = tmp_path / status_log.STATUS_JSONL
    old_bytes = json.dumps(older, ensure_ascii=False).encode()
    oversized = b'{"message":"' + b"x" * 300 + b'"}'
    path.write_bytes(old_bytes + b"\n\n" + oversized + b"\n" + json.dumps(latest).encode())
    rows = read_status(tmp_path, limit=0)
    assert rows[0] == older and rows[2] == latest
    assert rows[1]["context"]["byte_start"] == len(old_bytes) + 2
    assert rows[1]["context"]["bytes"] == len(oversized)
    assert latest_turn_status(tmp_path) == {"status": "started", "recorded_at": ""}


def test_runtime_status_reaches_bound_activity_listener():
    activity = AgentActivity(Console(force_terminal=False, no_color=True))
    bind_activity_listener(activity)
    try:
        emit_runtime_status("Hello from listener", source="agent")
        assert "Hello from listener" in activity.status_line
    finally:
        clear_status_listener()


def test_report_status_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    run = RunConfig(run_id="status-test", parquet_paths=[], model_id="m", benchmarks=["b"])
    set_runtime(Runtime(dataset=Dataset(parquet_paths=[], model_id="m", benchmarks=["b"]),
                        library=FunctionLibrary(), run=run, config=Config()))
    # Call through the MCP registry so the tools' registration is covered too.
    assert TOOL_CALLABLES["report_status"]("Evidence matches claim on regex-log")["ok"]
    listed = TOOL_CALLABLES["list_status_updates"](limit=5)
    assert listed["ok"] and listed["n"] >= 1
    assert listed["updates"][-1]["message"] == "Evidence matches claim on regex-log"
    assert listed["updates"][-1]["source"] == "agent"


def test_large_context_written_once_as_original_encoded_bytes(tmp_path):
    context = {'long': 'large\u2028context' * 1500}
    encoded = json.dumps(context, ensure_ascii=False, default=str).encode('utf-8')
    first = status_log.append_status(tmp_path, 'one', context=context, notify=False)
    artifact = tmp_path / first['context']['artifact']
    assert artifact.read_bytes() == encoded
    assert first['context']['sha256'] == hashlib.sha256(encoded).hexdigest()
    before = artifact.stat().st_mtime_ns
    status_log.append_status(tmp_path, 'two', context=context, notify=False)
    assert artifact.stat().st_mtime_ns == before
    assert (tmp_path / status_log.STATUS_JSONL).stat().st_size < 1024


@pytest.mark.parametrize("result_fields,expected", [
    ({}, "finished"),
    ({"terminal_reason": "aborted_tools"}, "interrupted"),
    ({"terminal_reason": "max_turns"}, "incomplete"),
    ({"stop_reason": "max_tokens"}, "incomplete"),
    ({"is_error": True, "errors": ["provider error"]}, "failed"),
    ("cancel", "interrupted"),
    ("exception", "failed"),
    ("no_result", "incomplete"),
])
def test_actual_harness_records_sdk_outcomes(
    tmp_path, monkeypatch, result_fields, expected,
):
    import claude_agent_sdk as sdk

    from autofa.harness.claude import ClaudeHarness
    from autofa.ui.activity import AgentActivity

    async def fake_query(**kwargs):
        assert latest_turn_status(tmp_path)["status"] == "started"
        if result_fields == "cancel":
            raise asyncio.CancelledError
        if result_fields == "exception":
            raise RuntimeError("provider error")
        if isinstance(result_fields, dict):
            fields = {"subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
                      "is_error": False, "num_turns": 1, "session_id": "session"}
            yield sdk.ResultMessage(**{**fields, **result_fields})

    monkeypatch.setattr(sdk, "query", fake_query)
    monkeypatch.setattr(AgentActivity, "start", lambda self: None)
    monkeypatch.setattr(AgentActivity, "stop", lambda self, **kwargs: None)
    harness = ClaudeHarness.__new__(ClaudeHarness)
    harness.model = "test"
    harness.cli_path = None
    harness.reasoning_options = {}
    result = asyncio.run(harness.run_turn(
        "Inspect logs", cwd=str(tmp_path), system_prompt="test",
        resume="session", show_result=False,
    ))
    assert result.session_id == "session"
    assert result.stopped is (expected == "interrupted")
    assert latest_turn_status(tmp_path)["status"] == expected


def test_status_is_not_inferred_from_agent_claims_or_an_old_success(tmp_path):
    assert latest_turn_status(tmp_path) == {"status": "unknown"}
    append_status(tmp_path, "All done", source="agent", context={"turn_status": "finished"})
    assert latest_turn_status(tmp_path) == {"status": "unknown"}
    record_turn_status(tmp_path, "finished")
    record_turn_status(tmp_path, "started")
    assert latest_turn_status(tmp_path)["status"] == "started"
    record_turn_status(tmp_path, "interrupted")
    assert latest_turn_status(tmp_path)["status"] == "interrupted"
    record_turn_status(tmp_path, "started")
    record_turn_status(tmp_path, "finished")
    assert latest_turn_status(tmp_path)["status"] == "finished"


def test_status_logging_failure_does_not_stop_analysis(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("read only")

    monkeypatch.setattr("autofa.session.status_log.append_status", fail)
    record_turn_status(tmp_path, "started")
    assert latest_turn_status(tmp_path) == {"status": "unknown"}


@pytest.mark.parametrize("status", ["interrupted", None])
def test_empty_report_keeps_counts_and_execution_status(tmp_path, status):
    from autofa.reports.failure_cards import write_failure_cards

    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "m", "benchmark": "b", "problem_id": "a", "is_correct": False},
        {"model_id": "m", "benchmark": "b", "problem_id": "b", "is_correct": True},
    ]), path)
    if status:
        record_turn_status(tmp_path, status)
    library = FunctionLibrary(model_id="m", benchmark="b")
    report = write_failure_cards(library, out=tmp_path / "failure_cards.html",
                                 dataset=Dataset([path], model_id="m", benchmarks=["b"]))
    html = report.read_text()
    data = json.loads(re.search(r'<script[^>]*id="autofa-data"[^>]*>(.*?)</script>', html, re.S)[1])
    assert data["meta"]["n_items"] == 2
    assert data["meta"]["n_failures"] == 1
    assert data["meta"]["n_functions"] == 0
    assert data["meta"]["agent_execution"]["status"] == (status or "unknown")
    assert library.functions == []
