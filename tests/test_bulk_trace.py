# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bulk trace import keeps execution evidence, request pools and caches exact and bounded."""

import copy
import errno
import hashlib
import io
import json
import shutil
import sqlite3
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import trace
from autofa.data import source_export
from autofa.data.bulk_trace import TraceArchive
from autofa.data.evidence_refs import expand_trace
from autofa.data.remote_api import DataApiError
from autofa.data.source_errors import SourceDataError
from autofa.data.source_export import materialize_source_traces, normalize_events
from autofa.data.source_request_references import (
    RequestReferenceResolver,
    attach_event_requests,
)
from autofa.data.trace_fields import TRACE_FIELDS, derive_trace_fields
from autofa.data.trace_navigation import trace_events

from support.traces import Api, compact, record

# --- archive integrity and bounded import ---

def normalized(tmp_path, items=None, *, task="task/a", lane=""):
    """Import rollouts through the bulk archive and normalize one conversation lane."""
    with TraceArchive(tmp_path, api=Api(items or [rich_record()]), run_id="run") as archive:
        raw = archive.rollout(task, 0, lane=lane)
    return normalize_events(**raw, conversation_lane=lane), raw


@pytest.mark.parametrize("failure", [
    "extra_member", "version", "wrong_count", "wrong_per_rollout_count", "wrong_run",
    "duplicate_event", "orphan_call", "short_text", "rollup", "rollup_over_empty_tables",
])
def test_corrupt_export_never_becomes_a_completed_index(tmp_path, failure):
    def mutate(tables, manifest):
        if failure == "extra_member":
            tables["../escape"] = []
        elif failure == "version":
            manifest["exporter_version"] = 2
        elif failure == "wrong_count":
            manifest["tables"]["events"]["rows"] += 1
        elif failure == "wrong_per_rollout_count":
            manifest["rollouts"][0]["rows"]["events"] += 1
        elif failure == "wrong_run":
            tables["events"][0]["run_id"] = "another-run"
        elif failure == "duplicate_event":
            tables["events"][1] = copy.deepcopy(tables["events"][0])
        elif failure == "orphan_call":
            tables["tool_calls"][0]["event_idx"] = 999
        elif failure == "short_text":
            tables["part_texts"][0]["text"] = "x"
        elif failure == "rollup":
            tables["rollouts"][0]["n_events"] += 1
        else:
            # Accurately manifested empty tables cannot be certified by a positive rollup.
            for table in ("events", "parts", "part_texts", "tool_calls", "tool_call_texts"):
                tables[table] = []
                manifest["tables"][table]["rows"] = 0
                manifest["rollouts"][0]["rows"][table] = 0
            for rollup in (tables["rollouts"][0], manifest["rollouts"][0]):
                rollup.update(n_events=0, n_tool_calls=1)

    with pytest.raises(DataApiError):
        TraceArchive(tmp_path, api=Api(mutate=mutate), run_id="run")
    for name in ("index.json", "trace.sqlite", "trace.partial", "download.json", "trace.zip"):
        assert not (tmp_path / name).exists()


def test_duplicate_zip_members_are_rejected(tmp_path):
    class Duplicate(Api):
        def download_trace_export(self, run_id, dest, **kwargs):
            super().download_trace_export(run_id, dest, **kwargs)
            with zipfile.ZipFile(dest, "a") as archive:
                with pytest.warns(UserWarning):
                    archive.writestr("manifest.json", "{}")

    with pytest.raises(DataApiError, match="duplicate"):
        TraceArchive(tmp_path, api=Duplicate(), run_id="run")


@pytest.mark.parametrize("records,lane,match", [
    pytest.param([record(agent="a"), record(agent="b")], "", "ambiguous", id="two_agents"),
    pytest.param(None, "missing", "lane", id="unknown_lane"),
])
def test_rollout_selection_fails_instead_of_combining_or_guessing(tmp_path, records, lane, match):
    with TraceArchive(tmp_path, api=Api(records), run_id="run") as archive:
        with pytest.raises(DataApiError, match=match):
            archive.rollout("task/a", 0, lane=lane)


def test_integrity_checked_cache_rebuilds_index_and_redownloads_changed_zip(tmp_path):
    api = Api()
    with TraceArchive(tmp_path, api=api, run_id="run") as archive:
        expected = archive.rollout("task/a", 0, lane="")
    (tmp_path / "trace.sqlite").write_bytes(b"corrupt index")
    with TraceArchive(tmp_path, api=api, run_id="run") as archive:
        assert archive.rollout("task/a", 0, lane="") == expected
    assert len(api.downloads) == 1
    (tmp_path / "trace.zip").write_bytes(b"corrupt zip")
    with TraceArchive(tmp_path, api=api, run_id="run") as archive:
        assert archive.rollout("task/a", 0, lane="") == expected
    assert len(api.downloads) == 2


def test_import_uses_bounded_batches_and_one_archive_and_keeps_long_reasoning(
    tmp_path, monkeypatch
):
    item = record(lane="session/main")
    part = item["events"][-1]["parts"][0]
    part.update(channel="reasoning", text="x" * 90001, chars=90001)
    api = Api([item, record(task="other")])
    original = compact(tmp_path / "compact.parquet", lane="session/main")
    real = pq.ParquetFile
    batch_sizes = []

    class BoundedParquet:
        def __init__(self, *args, **kwargs):
            self.file = real(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def iter_batches(self, *, batch_size, **kwargs):
            batch_sizes.append(batch_size)
            assert kwargs.get("columns") is not None or batch_size <= 32
            return self.file.iter_batches(batch_size=batch_size, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", BoundedParquet)
    out = materialize_source_traces(
        original, api=api, run_id="run", catalog={"run_id": "run", "n_rollouts": 2},
    )
    row = pq.read_table(out).to_pylist()[0]
    assert "x" * 90001 in row["response_text"]
    assert row["reward"] == 0.0 and row["is_correct"] is False
    assert api.downloads == [("run", [("task/a", 0)])]
    assert batch_sizes


def test_explicit_export_generation_refreshes_byte_identical_compact_input(tmp_path):
    original = compact(tmp_path / "compact.parquet")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"export_generation": "first"}))
    api = Api()
    first = materialize_source_traces(original, api=api, run_id="run", catalog={"run_id": "run"})
    source.write_text(json.dumps({"export_generation": "second"}))
    second = materialize_source_traces(original, api=api, run_id="run", catalog={"run_id": "run"})
    assert first != second and first.exists()
    assert len(api.downloads) == 2


def assert_failed_import_retries(tmp_path, api, repair, *, kept, error=DataApiError, match=None):
    """A failed import never leaves an index; only a completed, valid download is kept."""
    with pytest.raises(error, match=match):
        TraceArchive(tmp_path, api=api, run_id="run")
    assert not (tmp_path / "index.json").exists()
    assert (tmp_path / "download.json").exists() is kept
    assert (tmp_path / "trace.zip").exists() is kept
    repair()
    with TraceArchive(tmp_path, api=api, run_id="run") as archive:
        rollout = archive.rollout("task/a", 0, lane="")
    assert len(rollout["events"]) == 5 and len(rollout["calls"]) == 2
    assert len(api.downloads) == (1 if kept else 2)


def damage_parquet_page(dest):
    with zipfile.ZipFile(dest) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    damaged = bytearray(members["events.parquet"])
    parquet = pq.ParquetFile(io.BytesIO(damaged))
    offset = parquet.metadata.row_group(0).column(0).data_page_offset
    damaged[offset:offset + 12] = b"\xff" * 12
    members["events.parquet"] = damaged
    # Rewriting the archive makes its CRC valid, so only decoding
    # the Parquet page detects this completed download's corruption.
    with zipfile.ZipFile(dest, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


@pytest.mark.parametrize("failure,match", [
    ("bad_zip", "not a zip file"), ("bad_parquet_page", "Invalid trace Parquet"),
])
def test_invalid_completed_download_is_discarded_so_backend_repair_can_retry(
    tmp_path, failure, match,
):
    class Repaired(Api):
        broken = True

        def download_trace_export(self, run_id, dest, **kwargs):
            if self.broken and failure == "bad_zip":
                self.downloads.append((run_id, kwargs.get("rollouts")))
                dest.write_bytes(b"not a ZIP despite HTTP 200")
                return dest
            super().download_trace_export(run_id, dest, **kwargs)
            if self.broken:
                damage_parquet_page(dest)
            return dest

    api = Repaired()
    assert_failed_import_retries(
        tmp_path, api, lambda: setattr(api, "broken", False), kept=False, match=match,
    )


@pytest.mark.parametrize("failure_stage", ["open", "read"])
def test_parquet_local_io_failure_preserves_download(tmp_path, monkeypatch, failure_stage):
    real = pq.ParquetFile

    class UnreadableParquet:
        def __init__(self, *args, **kwargs):
            if failure_stage == "open":
                raise OSError(errno.EIO, "local read failed")
            self.file = real(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def iter_batches(self, **kwargs):
            raise OSError(errno.EIO, "local read failed")

    monkeypatch.setattr(pq, "ParquetFile", UnreadableParquet)
    assert_failed_import_retries(
        tmp_path, Api(), lambda: monkeypatch.setattr(pq, "ParquetFile", real),
        kept=True, match="local trace index",
    )


@pytest.mark.parametrize("failure,error,kept", [
    pytest.param(KeyboardInterrupt(), KeyboardInterrupt, True, id="interrupt"),
    pytest.param(sqlite3.OperationalError("disk full"), DataApiError, True, id="disk_full"),
    # An unexpected archive EOF means the download itself is bad; local failures are not.
    pytest.param(EOFError("member ended early"), DataApiError, False, id="archive_eof"),
])
def test_index_failure_keeps_only_a_completed_valid_download(
    tmp_path, monkeypatch, failure, error, kept,
):
    real = TraceArchive._index_archive

    def interrupted(*args):
        raise failure

    monkeypatch.setattr(TraceArchive, "_index_archive", interrupted)
    assert_failed_import_retries(
        tmp_path, Api(), lambda: monkeypatch.setattr(TraceArchive, "_index_archive", real),
        kept=kept, error=error,
    )


def many_compact_records(path, count):
    records = []
    for i in range(count):
        item = record(task=f"task-{i}")
        item["events"] = [item["events"][0], item["events"][-1]]
        item["calls"] = []
        item["metadata"].update(n_events=2, n_tool_calls=0)
        records.append(item)
    pq.write_table(pa.Table.from_pylist([{
        "problem_id": f"task-{i}", "run_idx": 0, "lane": "",
        "prompt_text": "prompt", "response_text": "summary", "reasoning_text": "",
    } for i in range(count)]), path)
    return records


BATCH = 5  # SELECTION_BATCH_SIZE (500) patched down so batching needs only a few rollouts.


@pytest.fixture
def small_batches(monkeypatch):
    monkeypatch.setattr("autofa.data.bulk_trace.SELECTION_BATCH_SIZE", BATCH)


@pytest.mark.parametrize("count", [BATCH, 2 * BATCH + 1])
def test_large_subsets_are_split_without_a_whole_run_request(tmp_path, small_batches, count):
    original = tmp_path / "compact.parquet"
    api = Api(many_compact_records(original, count))
    result = materialize_source_traces(original, api=api, run_id="run",
                                       catalog={"run_id": "run", "n_rollouts": 10000})
    assert pq.ParquetFile(result).metadata.num_rows == count
    assert [len(selected) for _, selected in api.downloads] == [
        min(BATCH, count - i) for i in range(0, count, BATCH)
    ]
    assert [identity for _, selected in api.downloads for identity in selected] == [
        (f"task-{i}", 0) for i in range(count)
    ]
    manifest = json.loads((result.parent / "manifest.json").read_text())
    exports = manifest["trace_exports"]
    assert len(exports) == len(api.downloads)
    assert manifest["trace_export_sha256"] == (
        exports[0]["sha256"] if len(exports) == 1 else None
    )
    expected_sources = {
        tuple(identity): export["sha256"]
        for export in exports for identity in export["rollouts"]
    }
    for row in pq.read_table(result).to_pylist():
        trace = json.loads(row["response_text"])
        assert trace["execution"]["source"]["trace_export_sha256"] == expected_sources[
            row["problem_id"], row["run_idx"]
        ]


def test_unknown_catalog_size_is_selected_and_failed_later_batch_resumes(tmp_path, small_batches):
    original = tmp_path / "compact.parquet"

    class Interrupted(Api):
        fail = True

        def download_trace_export(self, run_id, dest, *, rollouts=None, **kwargs):
            assert rollouts is not None
            if self.fail and rollouts[0][0] == f"task-{BATCH}":
                self.downloads.append((run_id, rollouts))
                raise DataApiError("temporary export failure", status=503)
            return super().download_trace_export(run_id, dest, rollouts=rollouts, **kwargs)

    api = Interrupted(many_compact_records(original, BATCH + 1))
    with pytest.raises(DataApiError, match="temporary export failure"):
        materialize_source_traces(original, api=api, run_id="run", catalog={"run_id": "run"})
    api.fail = False
    result = materialize_source_traces(original, api=api, run_id="run", catalog={"run_id": "run"})
    assert pq.ParquetFile(result).metadata.num_rows == BATCH + 1
    assert [len(selected) for _, selected in api.downloads] == [BATCH, 1, 1]


def test_known_complete_selection_still_uses_one_run_export(tmp_path, small_batches):
    original = tmp_path / "compact.parquet"
    api = Api(many_compact_records(original, BATCH + 1))
    materialize_source_traces(original, api=api, run_id="run",
                              catalog={"run_id": "run", "n_rollouts": BATCH + 1})
    assert api.downloads == [("run", None)]


# --- execution metadata ---

def rich_record():
    item = record()
    item["metadata"].update(provider_extension={"region": "local", "reward": 0})
    event = item["events"][1]
    event.update(model_name="model", prompt_tokens=100, finish_reason="tool_calls",
                 source_event_id="source-1", provider_extension={"queue_ms": 12})
    item["calls"][0].update(error_type="ProcessExit", call_id="call-1")
    auxiliary = copy.deepcopy(item["events"][4])
    auxiliary.update(lane="worker/second", kind="other", role="harness")
    auxiliary["parts"][0].update(lane="worker/second", text="A" * 90001, chars=90001)
    item["events"].append(auxiliary)
    grader = copy.deepcopy(item["events"][0])
    grader.update(lane="judge/verdict", actor="grader")
    grader["parts"][0].update(lane="judge/verdict", text="PRIVATE_VERDICT", chars=15)
    item["events"].append(grader)
    definition = json.dumps({"name": "terminal", "parameters": {
        "type": "object", "properties": {"reward": {"type": "number"}},
    }})
    item["tables"] = {
        "attempts": [{"attempt_idx": 0, "failure_class": "TimeoutError", "is_terminal": False,
                      "error_message": "transport timeout"}],
        "artifacts": [{"artifact_idx": 0, "name": "terminal.log", "kind": "log",
                       "mime": "text/plain", "event_idx": 3, "size_bytes": 90001}],
        "artifact_texts": [{"artifact_idx": 0, "text": "L" * 90001}],
        "tool_defs": [{"def_idx": 0, "name": "terminal", "def_chars": len(definition),
                       "offered_by": "harness"}],
        "tool_def_texts": [{"def_idx": 0, "def_json": definition}],
        "event_texts": [{"event_idx": 1, "lane": "", "request_dialect": "openai",
                         "request_messages": json.dumps([{"role": "user", "content": "Task"}]),
                         "request_tools": json.dumps([json.loads(definition)]),
                         "request_params": json.dumps({"temperature": 0.7, "api_key": "SECRET"})}],
        "provider_metrics": [{"sample_idx": 0, "queue_ms": 120, "status": "retry", "reward": 0}],
        "grader_results": [{"text": "PRIVATE_EXTRA_VERDICT", "reward": 0}],
        "prompt_messages": [{"msg_idx": 0, "msg_hash": "message-hash",
                             "msg_json": json.dumps({"role": "user", "content": "Full prompt",
                                                     "reward": 0, "api_key": "SECRET"})}],
        "prompt_summaries": [{"event_idx": 1, "lane": "", "class": "compaction",
                              "api_key": "SECRET"}],
        "provenance": [{"task_id": None, "repeat": None, "agent_ref": None,
                        "path": "/source/trace.json", "reader": "reader-v17"}],
        "relations": [{"task_id": None, "repeat": None, "agent_ref": None,
                       "frm_task_id": "task/a", "frm_repeat": 0, "frm_lane": "",
                       "to_task_id": "task/a", "to_repeat": 0, "to_lane": "worker/second",
                       "kind": "fork", "resolved": True}],
    }
    return item


def test_full_execution_evidence_and_other_lanes_survive_without_target_fields(tmp_path):
    trace, _ = normalized(tmp_path)
    execution = trace["execution"]
    encoded = json.dumps(trace)
    for private in ("PRIVATE_VERDICT", "PRIVATE_EXTRA_VERDICT", "SECRET"):
        assert private not in encoded
    assert len(trace["steps"]) == 5
    assert len(execution["events"]) == 6
    assert execution["events"][-1]["parts"][0]["text"] == "A" * 90001
    assert execution["rollout"]["provider_extension"] == {"region": "local"}
    assert "status" not in execution["rollout"] and "is_scored" not in execution["rollout"]
    step = trace["steps"][1]
    assert step["metadata"]["source_event_id"] == "source-1"
    assert step["metadata"]["provider_extension"] == {"queue_ms": 12}
    assert step["tool_calls"][0]["status"] == "failed"
    assert execution["attempts"][0]["failure_class"] == "TimeoutError"
    assert execution["artifacts"][0]["text"] == "L" * 90001
    definition = json.loads(execution["tool_definitions"][0]["def_json"])
    assert "reward" in definition["parameters"]["properties"]
    request = execution["events"][1]["event_text"]
    assert json.loads(request["request_messages"])[0]["content"] == "Task"
    assert json.loads(request["request_params"]) == {"temperature": 0.7}
    assert execution["provenance"][0]["reader"] == "reader-v17"
    assert execution["relations"][0]["resolved"] is True
    assert execution["prompt_summaries"][0]["class"] == "compaction"
    assert "api_key" not in execution["prompt_summaries"][0]
    for resource in ("prompt_summaries", "events", "artifacts", "prompt_messages",
                     "extra_resources.provider_metrics"):
        assert execution["availability"][resource]["status"] == "available"
    assert trace["steps"][3]["source_parts"][0]["part_idx"] == 0
    model_call = trace["model_call_evidence"]["calls"][0]
    assert (model_call["source_event_index"], model_call["source_lane"]) == (1, "")
    assert model_call["response_recorded"] is False
    assert trace["initial_request"]["messages"][0]["content"] == "Task"
    assert json.loads(execution["prompt_messages"][0]["msg_json"]) == {
        "role": "user", "content": "Full prompt",
    }
    metrics = execution["extra_resources"]["provider_metrics"][0]
    assert metrics["queue_ms"] == 120 and metrics["status"] == "retry"
    assert "reward" not in metrics
    assert "grader_results" not in execution["extra_resources"]


def test_optional_resources_distinguish_absent_tables_from_recorded_empty_tables(tmp_path):
    execution = normalized(tmp_path, [record()])[0]["execution"]
    assert execution["attempts"] == [] and execution["provenance"] == []
    assert execution["availability"]["attempts"]["status"] == "available"
    for resource in ("provenance", "relations", "event_texts"):
        assert execution["availability"][resource]["status"] == "unavailable"


@pytest.mark.parametrize("gap,partial", [
    ("missing_full_text", ["artifacts", "tool_definitions"]),
    ("source_truncated", ["events", "artifacts"]),
    ("rollup_exceeds_exported_table", ["artifacts"]),
])
def test_incomplete_resources_are_reported_partial(tmp_path, gap, partial):
    item = rich_record()
    if gap == "missing_full_text":
        item["tables"]["artifact_texts"] = []
        item["tables"]["tool_def_texts"] = []
    elif gap == "source_truncated":
        item["events"][1]["length_truncated"] = True
        item["tables"]["artifacts"][0]["truncated"] = True
    else:
        item["metadata"]["n_artifacts"] = 2
    availability = normalized(tmp_path, [item])[0]["execution"]["availability"]
    assert [availability[resource]["status"] for resource in partial] == ["partial"] * len(partial)


@pytest.mark.parametrize("resource", ["tool_calls", "tool_defs"])
def test_shortened_optional_or_auxiliary_evidence_is_rejected(tmp_path, resource):
    item = rich_record()
    if resource == "tool_calls":
        call = copy.deepcopy(item["calls"][0])
        call.update(lane="worker/second", event_idx=4, args_full="short")
        item["calls"].append(call)
    else:
        item["tables"]["tool_def_texts"][0]["def_json"] = "{}"
    with pytest.raises(DataApiError, match="shortened"):
        normalized(tmp_path, [item])


def test_adapter_upgrade_reuses_legacy_raw_download_and_preserves_old_output(tmp_path, monkeypatch):
    original = compact(tmp_path / "compact.parquet")
    api = Api([rich_record()])
    first = materialize_source_traces(original, api=api, run_id="run", catalog={"run_id": "run"})
    old = first.read_bytes()
    bulk = next(tmp_path.glob("source-bulk-*"))
    shutil.move(bulk, first.parent / "bulk")  # Layout of adapters before shared raw caches.
    monkeypatch.setattr(source_export, "ADAPTER_VERSION", source_export.ADAPTER_VERSION + 1)
    current = materialize_source_traces(original, api=api, run_id="run", catalog={"run_id": "run"})
    assert current != first and first.read_bytes() == old
    assert len(api.downloads) == 1
    trace = json.loads(pq.read_table(current).to_pylist()[0]["response_text"])
    assert trace["execution"]["attempts"][0]["error_message"] == "transport timeout"


def test_non_conversation_event_text_is_retained_and_validated():
    item = record()
    event = item["events"][-1]
    event.update(kind="transport", role="harness", retry_after=2)
    trace = normalize_events(item["metadata"], item["events"], item["calls"])
    assert len(trace["steps"]) == 4
    assert trace["execution"]["events"][-1]["retry_after"] == 2
    assert trace["execution"]["events"][-1]["parts"][0]["text"] == "done"
    event["parts"][0]["text"] = "cut"
    with pytest.raises(DataApiError, match="shortened"):
        normalize_events(item["metadata"], item["events"], item["calls"])


def test_copied_context_remains_available_without_replaying_it_as_a_conversation_step():
    item = record()
    item["events"][-1]["is_copied_context"] = True
    trace = normalize_events(item["metadata"], item["events"], item["calls"])
    assert len(trace["steps"]) == 4
    assert trace["execution"]["events"][-1]["is_copied_context"] is True
    assert trace["execution"]["events"][-1]["parts"][0]["text"] == "done"


def test_grader_parent_calls_are_excluded_even_without_call_actor_metadata(tmp_path):
    item = record()
    item["events"][1]["actor"] = "grader"
    trace, _ = normalized(tmp_path, [item])
    execution = trace["execution"]
    assert len(trace["steps"]) == 4 and len(execution["events"]) == 4
    assert [call["event_idx"] for call in execution["tool_calls"]] == [2]


def test_explicit_grader_parts_do_not_enter_step_text_or_raw_event_content():
    item = record()
    part = copy.deepcopy(item["events"][-1]["parts"][0])
    part.update(part_idx=1, actor="grader", text="PRIVATE_VERDICT", chars=15)
    item["events"][-1]["parts"].append(part)
    item["events"][-1]["n_parts"] += 1
    trace = normalize_events(item["metadata"], item["events"], item["calls"])
    assert trace["steps"][-1]["message"] == "done"
    assert "PRIVATE_VERDICT" not in json.dumps(trace)


def test_artifact_of_grader_event_cannot_leak_verdict_through_generic_log():
    item = record()
    item["events"][1]["actor"] = "grader"
    # Another lane reusing the grader's event index must not make its artifact look safe.
    execution = {
        "rollout": item["metadata"], "tool_calls": item["calls"],
        "events": [*item["events"], {
            "event_idx": 1, "lane": "safe-worker", "kind": "message", "parts": [],
        }],
        "artifacts": [{"event_idx": 1, "kind": "log", "text": "PRIVATE_VERDICT"}],
    }
    trace = normalize_events(item["metadata"], item["events"], item["calls"], execution=execution)
    assert "PRIVATE_VERDICT" not in json.dumps(trace)
    assert trace["execution"]["artifacts"] == []
    assert trace["execution"]["availability"]["artifacts"]["status"] == "partial"


@pytest.mark.parametrize("grader_reference", [False, True])
def test_lane_less_prompt_image_is_withheld_only_when_a_grader_event_references_it(
    grader_reference,
):
    item = record()
    events = [
        {"lane": "", "event_idx": 0, "role": "user", "kind": "message",
         "parts": [{"part_idx": 0, "part_type": "image", "artifact_idx": 0}]},
        {"lane": "verdict", "event_idx": 0, "actor": "harness", "kind": "other",
         "parts": ([{"part_idx": 0, "artifact_idx": 0}] if grader_reference else [])},
    ]
    execution = {
        "rollout": item["metadata"], "events": events, "tool_calls": [],
        "artifacts": [{"artifact_idx": 0, "event_idx": 0, "part_idx": 0, "kind": "image",
                       "origin": "inline", "mime": "image/jpeg"}],
    }
    trace = normalize_events(item["metadata"], item["events"], item["calls"], execution=execution)
    artifacts = trace["execution"]["artifacts"]
    if grader_reference:
        assert artifacts == []
        assert trace["execution"]["availability"]["artifacts"]["status"] == "partial"
    else:
        assert [row["artifact_idx"] for row in artifacts] == [0] and artifacts[0]["lane"] == ""


def test_compact_cache_preserves_detector_views_and_reuses_source_archive(tmp_path, monkeypatch):
    source = compact(tmp_path / "source.parquet")
    api = Api([rich_record()])
    with monkeypatch.context() as patch:
        patch.setattr(source_export, "compact_trace", lambda value: value)
        patch.setattr(source_export, "ADAPTER_VERSION", 11)
        old_path = source_export.materialize_source_traces(
            source, api=api, run_id="run", catalog={"run_id": "run"},
        )
    old_bytes = old_path.read_bytes()
    new_path = source_export.materialize_source_traces(
        source, api=api, run_id="run", catalog={"run_id": "run"},
    )
    old = pq.read_table(old_path).to_pylist()[0]
    new = pq.read_table(new_path).to_pylist()[0]
    original, stored = (json.loads(row["response_text"]) for row in (old, new))
    assert old_path != new_path and old_path.read_bytes() == old_bytes
    assert len(api.downloads) == 1
    assert expand_trace(stored) == original
    assert len(new["response_text"]) < len(old["response_text"])
    assert trace.steps(new) == trace.steps(old)
    assert trace.record(new) == trace.record(old)
    assert trace.events(new) == trace.events(old)
    assert trace_events(new) == trace_events(old)
    old_fields, new_fields = derive_trace_fields(old), derive_trace_fields(new)
    assert {k: new_fields.get(k) for k in TRACE_FIELDS} == {
        k: old_fields.get(k) for k in TRACE_FIELDS
    }
    # Existing raw JSON detectors keep the original normalized conversation API.
    for before, after in zip(original["steps"], stored["steps"], strict=True):
        for key in ("source", "step_id", "message", "observation", "reasoning_content"):
            assert after.get(key) == before.get(key)
        for old_call, new_call in zip(
            before.get("tool_calls", []), after.get("tool_calls", []), strict=True,
        ):
            for key in ("function_name", "arguments", "source_call_index", "result_event_idx"):
                assert new_call.get(key) == old_call.get(key)


# --- remote request pools ---

def request_record(*, task="task/a", lane="", content="Recorded task", tool="inspect"):
    item = record(task=task, lane=lane)
    messages = [
        {"role": "system", "content": "Read before editing"},
        {"role": "user", "content": content},
    ]
    definition = {
        "type": "function",
        "function": {"name": tool, "parameters": {"type": "object"}},
    }
    encoded = [json.dumps(message) for message in messages]
    refs = [hashlib.sha256(value.encode()).hexdigest()[:16] for value in encoded]
    tool_json = json.dumps(definition)
    tool_ref = hashlib.sha256(tool_json.encode()).hexdigest()
    item["tables"] = {
        "event_texts": [
            {
                "lane": lane, "event_idx": index, "request_dialect": "chat",
                "request_messages": json.dumps(refs),
                "request_tools": json.dumps([tool_ref]),
                "request_params": '{"temperature":0.2}',
            }
            for index in (1, 4)
        ],
        "prompt_messages": [
            {"msg_idx": index, "msg_hash": ref, "msg_json": payload}
            for index, (ref, payload) in enumerate(zip(refs, encoded, strict=True))
        ],
        "tool_defs": [{"def_idx": 0, "sha256": tool_ref, "def_chars": len(tool_json)}],
        "tool_def_texts": [{"def_idx": 0, "def_json": tool_json}],
    }
    return item, messages, definition


def test_remote_only_import_resolves_messages_and_tools_without_raw_fallback(tmp_path):
    item, messages, definition = request_record()
    original = copy.deepcopy(item)
    api = Api([item])  # Any per-rollout or raw-source HTTP read raises in this fake.
    path = materialize_source_traces(
        compact(tmp_path / "compact.parquet"), api=api, run_id="run",
        catalog={"run_id": "run", "n_rollouts": 1},
    )
    row = pq.read_table(path).to_pylist()[0]
    calls = trace.model_calls(row)
    assert len(calls) == 2
    assert all(call["request"] == messages and call["tools"] == [definition] for call in calls)
    assert all(call["request_recorded"] and not call["response_recorded"] for call in calls)
    assert "Recorded task" in row["prompt_text"]
    assert [step.step_id for step in trace.steps(row)] == [0, 1, 2, 3, 4]
    evidence = json.loads(row["response_text"])
    notice = "provider response envelopes not recorded in event request captures"
    manifest = json.loads((path.parent / "manifest.json").read_text())
    assert evidence["evidence_scope"]["notices"][notice] == 2
    assert manifest["notices"][notice] == 2
    assert notice not in manifest["limitations"]
    assert "unsupported tool definitions retained in raw source" not in manifest["limitations"]
    assert item == original
    assert api.downloads == [("run", None)]


def test_request_attachment_preserves_lane_and_rollout_pools(tmp_path):
    main, messages, definition = request_record(lane="main")
    worker, worker_messages, worker_definition = request_record(
        lane="worker", content="Worker task", tool="worker_tool",
    )
    main["events"].extend(worker["events"])
    main["calls"].extend(worker["calls"])
    # Pool row identities are rollout-level; assign distinct indices for the worker.
    for table, key, offset in (("prompt_messages", "msg_idx", 2),
                               ("tool_defs", "def_idx", 1),
                               ("tool_def_texts", "def_idx", 1)):
        for row in worker["tables"][table]:
            row[key] += offset
    for table, rows in worker["tables"].items():
        main["tables"][table].extend(rows)
    other, _, _ = request_record(task="other", content="Other rollout", tool="other_tool")
    # Even deliberately conflicting names in another rollout cannot contaminate this one.
    for pool, hash_key in (("prompt_messages", "msg_hash"), ("tool_defs", "sha256")):
        for index, row in enumerate(other["tables"][pool]):
            row[hash_key] = main["tables"][pool][index][hash_key]
    result, raw = normalized(tmp_path, [main, other], lane="main")
    calls = trace.model_calls({"response_text": json.dumps(result)})
    assert [(call["source_lane"], call["source_event_index"]) for call in calls] == [
        ("main", 1), ("main", 4), ("worker", 1), ("worker", 4),
    ]
    for call in calls:
        assert call["request"] == (messages if call["source_lane"] == "main" else worker_messages)
        assert call["tools"] == ([definition] if call["source_lane"] == "main"
                                 else [worker_definition])
    assert result["initial_request"]["messages"] == messages
    assert all(event["lane"] == "main" for event in raw["events"])


@pytest.mark.parametrize("field,failure", [
    ("request_messages", "missing"), ("request_messages", "ambiguous"),
    ("request_tools", "invalid_json"), ("request_tools", "missing_text"),
])
def test_unresolved_remote_references_fail_explicitly(tmp_path, field, failure):
    item, _, _ = request_record()
    if field == "request_messages":
        table, index, text, reference = "prompt_messages", "msg_idx", "msg_json", "msg_hash"
        rows = item["tables"][table]
    else:
        table, index, text, reference = "tool_defs", "def_idx", "def_json", "sha256"
        rows = item["tables"]["tool_def_texts"]
    if failure == "missing":
        item["tables"]["event_texts"][0][field] = json.dumps([
            "f" * (16 if field == "request_messages" else 64),
        ])
    elif failure == "ambiguous":
        duplicate = {**item["tables"][table][0], index: 10}
        if field == "request_messages":
            duplicate[text] = '{"role":"user","content":"Conflicting"}'
        else:
            item["tables"]["tool_def_texts"].append({index: 10, text: '{"name":"other"}'})
            duplicate["def_chars"] = len('{"name":"other"}')
        item["tables"][table].append(duplicate)
        assert duplicate[reference] == item["tables"][table][0][reference]
    else:
        rows[0][text] = "{invalid" if failure == "invalid_json" else None
        if field == "request_tools" and failure == "invalid_json":
            item["tables"][table][0]["def_chars"] = len(rows[0][text])
    expected = "invalid" if failure == "invalid_json" else (
        "missing" if failure == "missing_text" else failure
    )
    with pytest.raises(SourceDataError, match=f"{expected} {field}"):
        normalized(tmp_path, [item])


def test_reference_does_not_fall_back_to_another_rollouts_pool(tmp_path):
    item, _, _ = request_record()
    other, _, _ = request_record(task="other")
    item["tables"]["prompt_messages"].clear()
    with pytest.raises(SourceDataError, match="missing request_messages"):
        normalized(tmp_path, [item, other])


def test_legacy_inline_requests_remain_unchanged_and_need_no_pool(tmp_path):
    item, messages, definition = request_record()
    for captured in item["tables"]["event_texts"]:
        captured["request_messages"] = json.dumps(messages)
        captured["request_tools"] = json.dumps([definition])
    for table in ("prompt_messages", "tool_defs", "tool_def_texts"):
        item["tables"][table].clear()
    result, raw = normalized(tmp_path, [item])
    request = raw["events"][1]["event_text"]
    assert request["request_messages"] == json.dumps(messages)
    assert request["request_tools"] == json.dumps([definition])
    assert "request_messages_raw" not in request
    assert result["initial_request"] == {"messages": messages, "tools": [definition]}


@pytest.mark.parametrize("messages", [
    pytest.param(["A literal message", "Another literal message"], id="legacy_literal_strings"),
    pytest.param([{"role": "user", "content": ["a" * 16, "b" * 64]}], id="nested_content"),
])
def test_ordinary_inline_strings_and_content_are_not_interpreted_as_references(messages):
    captured = {"request_messages": json.dumps(messages), "request_tools": "[]"}
    resolved = RequestReferenceResolver([], []).resolve(captured)
    assert resolved["request_messages"] == messages
    assert resolved["request_messages_raw"] == captured["request_messages"]
    assert resolved["request_tools"] == []


def test_duplicate_pool_values_with_equal_json_are_unambiguous():
    reference = "a" * 16
    resolver = RequestReferenceResolver([
        {"msg_hash": reference, "msg_json": '{"role":"user","content":"Task"}'},
        {"msg_hash": reference, "msg_json": '{ "content": "Task", "role": "user" }'},
    ], [])
    assert resolver.resolve({"request_messages": json.dumps([reference])})[
        "request_messages"
    ] == [{"role": "user", "content": "Task"}]


def test_archived_requests_stay_compact_and_unique_messages_are_pooled(tmp_path):
    content = "Repeated context marker " + "x" * 30000
    item, _, _ = request_record(content=content)
    request = item["tables"]["event_texts"][0]
    item["tables"]["event_texts"] = [
        {**request, "event_idx": event["event_idx"]} for event in item["events"]
    ]
    result, raw = normalized(tmp_path, [item])
    for event in raw["events"]:
        assert event["event_text"]["request_messages"] == request["request_messages"]
        assert content not in json.dumps(event)
    assert len(result["model_call_evidence"]["messages"]) == 2
    assert len(result["model_call_evidence"]["calls"]) == 5
    # Unique pool + original pool + initial request, never one history per event.
    assert len(json.dumps(result)) < 5 * len(content)


def test_shared_attachment_never_combines_request_lanes_or_mutates_rows():
    keys = ("run_id", "task_id", "repeat", "agent_ref", "lane", "event_idx")
    base = {"run_id": "run", "task_id": "task", "repeat": 0, "agent_ref": "agent",
            "event_idx": 1, "request_messages": '["literal"]'}
    requests = [{**base, "lane": lane} for lane in ("main", "worker")]
    original = copy.deepcopy(requests)
    events = {tuple(row[key] for key in keys): {} for row in requests}
    attach_event_requests(requests, events, keys)
    assert [event["event_text"]["lane"] for event in events.values()] == ["main", "worker"]
    assert requests == original
    with pytest.raises(SourceDataError, match="Duplicate model request"):
        attach_event_requests([requests[0], requests[0]], events, keys)
    with pytest.raises(SourceDataError, match="Orphan model request"):
        attach_event_requests([{**requests[0], "lane": "missing"}], events, keys)
