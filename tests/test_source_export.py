# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source export fidelity, optional execution resources and raw-download cache reuse."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.data.evalrdl_efficiency import locate_bulk_indexes
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.source_export import (
    _get,
    _reuse_raw_downloads,
    materialize_efficiency_tables,
    materialize_source_traces,
    normalize_events,
)
from autofa.data.source_normalization import _validate_identities
from autofa.data.source_resources import fetch_execution_resource, recover_execution_resources
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trajectory import parse_trajectory
from autofa.mcp import tools_data

from support.traces import Api as BulkApi
from support.traces import compact, fixture, write_trace_archive


@pytest.mark.parametrize("change", ["canonical_run_id", "run_id", "generation", "legacy"])
def test_raw_downloads_are_reused_only_for_the_same_run_generation(tmp_path, change):
    old = tmp_path / "source-v13-old"
    inputs = old / "initial-inputs"
    inputs.mkdir(parents=True)
    raw = inputs / "capture.jsonl"
    raw.write_bytes(b'{"raw": "original"}\n')
    receipt = raw.with_suffix(".sha256")
    receipt.write_text(hashlib.sha256(raw.read_bytes()).hexdigest())
    manifest = {"base_url": "https://example.test", "run_id": "run", "canonical_run_id": "run",
                "export_sha256": "subset"}
    source, generation = {"export_generation": "generation"}, "generation"
    if change == "canonical_run_id":  # Requested by an alias of the same canonical run.
        manifest["run_id"] = "alias"
    elif change == "generation":
        source["export_generation"] = "different"
    elif change == "legacy":  # Caches without a generation must match the export digest.
        source, generation = {}, None
    elif change == "run_id":
        manifest["canonical_run_id"] = "different"
    (old / "manifest.json").write_text(json.dumps(manifest))
    (old / "source.json").write_text(json.dumps(source))
    dest = tmp_path / "source-v14-new" / "initial-inputs"
    # Another export digest of the same run generation is just a different subset.
    digest = "other" if change in ("canonical_run_id", "legacy") else "subset"
    _reuse_raw_downloads(dest, tmp_path, "https://example.test", "run", digest, generation)
    if change in ("run_id", "generation", "legacy"):
        assert not dest.exists()
        return
    assert (dest / raw.name).samefile(raw)
    assert (dest / receipt.name).read_text() == receipt.read_text()
    # Cache replacement leaves source artifacts unchanged.
    (dest / receipt.name).write_text("new checksum")
    replacement = dest / "replacement.partial"
    replacement.write_bytes(b"replacement")
    replacement.replace(dest / raw.name)
    assert raw.read_bytes() == b'{"raw": "original"}\n'
    assert receipt.read_text() != "new checksum"


def test_full_content_pairing_and_no_grader_metadata():
    trace = normalize_events(*fixture())
    blob = json.dumps(trace)
    parsed = parse_trajectory(blob)
    assert parsed.ok and parsed.n_steps == 5 and parsed.n_tool_calls == 2
    assert parsed.agent_name == "hermes"
    assert "x" * 6000 in blob
    assert trace["steps"][3]["paired_event_idx"] == 1
    assert trace["steps"][3]["paired_call_idx"] == 0
    assert trace["steps"][1]["tool_calls"][0]["result_event_idx"] == 3
    assert '"reward"' not in blob and '"is_scored"' not in blob
    assert "status" not in trace["execution"]["rollout"]
    assert trace["steps"][1]["tool_calls"][0]["status"] == "failed"
    assert trace["execution"]["tool_calls"][0]["status"] == "failed"
    fields = derive_trace_fields({"response_text": blob, "reasoning_text": ""})
    assert "Traceback" in fields["observations_text"]
    assert fields["final_message"] == "done"
    assert "terminal" in fields["actions_text"]


def test_harbor_agent_and_output_roles_keep_messages_tool_calls_and_observations():
    meta, events, calls = fixture()
    events[0]["kind"] = "system"
    events[0]["role"] = "system"
    events[1]["kind"] = "message"
    events[1]["role"] = "agent"
    events[3]["role"] = "output"  # A tool result with an output role is an observation.
    events[-1]["role"] = "agent"
    trace = normalize_events(meta, events, calls)
    parsed = parse_trajectory(json.dumps(trace))
    assert parsed.n_tool_calls == 2 and parsed.n_agent_steps == 3
    assert trace["evidence_scope"]["limitations"] == {}
    assert trace["steps"][0]["message"] == "fix it"
    assert trace["steps"][3]["source"] == "tool"
    assert trace["steps"][3]["observation"].startswith("Traceback:")
    assert trace["steps"][3]["paired_event_idx"] == 1
    assert derive_trace_fields({"response_text": json.dumps(trace)})["final_message"] == "done"


@pytest.mark.parametrize("target", ["event", "part"])
def test_auxiliary_lane_events_or_parts_are_rejected_if_server_ignores_lane_filter(target):
    meta, events, calls = fixture()
    (events[0] if target == "event" else events[0]["parts"][0])["lane"] = "verdict"
    with pytest.raises(DataApiError, match="auxiliary lane"):
        normalize_events(meta, events, calls)


# False is present but falsy: it must be validated, not skipped as an absent count.
@pytest.mark.parametrize("count", [False, "0"], ids=["falsy_bool", "string"])
def test_only_absent_event_call_counts_may_be_inferred(count):
    meta, events, calls = fixture()
    events[0]["n_tool_calls"] = count
    with pytest.raises(DataApiError, match="valid n_tool_calls"):
        normalize_events(meta, events, calls)


def test_nullable_event_counts_do_not_hide_missing_rollout_calls():
    meta, events, calls = fixture()
    for event in events:
        event["n_tool_calls"] = None
    with pytest.raises(DataApiError, match="tool-call count"):
        normalize_events(meta, events, calls[:-1])


@pytest.mark.parametrize("case", [
    "event_count", "duplicate_event", "duplicate_call", "short_args", "absent_args",
    "missing_part", "missing_calls", "duplicate_part",
])
def test_incomplete_data_fails_visibly(case):
    meta, events, calls = fixture()
    if case == "event_count":
        meta["n_events"] += 1
    elif case == "duplicate_event":
        events[2]["event_idx"] = 1
    elif case == "duplicate_call":
        calls[1] = copy.deepcopy(calls[0])
    elif case == "short_args":
        calls[0]["args_full"] = "{}"
    elif case == "absent_args":  # args_head alone is a preview, not evidence
        del calls[0]["args_full"]
    elif case == "missing_part":
        events[0]["parts"] = []
    elif case == "missing_calls":
        events[1]["n_tool_calls"] = 2
    elif case == "duplicate_part":
        events[0]["parts"] *= 2
        events[0]["n_parts"] = 2
    with pytest.raises(DataApiError):
        normalize_events(meta, events, calls)


def test_limitations_are_visible_without_invented_text():
    meta, events, calls = fixture()
    meta["n_artifacts"] = 2
    events[0]["has_encrypted_reasoning"] = True
    events[3]["length_truncated"] = True
    events[0]["parts"].append({"part_idx": 1, "part_type": "image"})
    events[0]["n_parts"] += 1
    trace = normalize_events(meta, events, calls)
    limits = trace["evidence_scope"]["limitations"]
    assert limits["encrypted reasoning unavailable"] == 1
    assert limits["artifact file contents not inspected"] == 2
    assert limits["non-text part not inspected: image"] == 1
    assert limits["source event marked truncated"] == 1


class Api:
    base_url = "https://example.test"

    def __init__(self):
        self.meta, self.events, self.calls = fixture()
        self.requests = []
        self.downloads = []

    def download_trace_export(self, run_id, dest, *, progress=None, rollouts=None):
        assert rollouts == [("task/a", 0)]
        self.downloads.append(run_id)
        return write_trace_archive(dest, [{"metadata": self.meta, "events": self.events,
                                          "calls": self.calls}], run_id=run_id)

    def _get_json(self, path, query=None):
        if path.startswith("/v1/runs/"):
            return {"run_id": "run"}
        self.requests.append((path, query))
        assert path.startswith("/v1/rollouts/run/task%2Fa/0")
        return self.meta


def test_part_identity_mismatch_fails():
    meta, events, calls = fixture()
    events[0]["parts"][0]["task_id"] = "different"
    with pytest.raises(DataApiError, match="part identity"):
        _validate_identities(meta, events, calls)


def test_only_transient_get_errors_are_retried(monkeypatch):
    delays = []
    monkeypatch.setattr("autofa.data.source_export.time.sleep", delays.append)
    api = Api()
    responses = iter([DataApiError("overloaded", status=503), {"ok": True}])

    def get(*args):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    api._get_json = get
    assert _get(api, "/test") == {"ok": True}
    assert delays == [1]
    responses = iter([DataApiError("not found", status=404)])
    with pytest.raises(DataApiError, match="not found"):
        _get(api, "/test")
    assert delays == [1]


def test_materialization_preserves_labels_rows_and_original_and_resumes(tmp_path):
    original = tmp_path / "dataset.parquet"
    row = {"problem_id": "task/a", "run_idx": 0, "prompt_text": "question",
           "response_text": "summary", "reasoning_text": "capped reasoning",
           "is_correct": False, "reward": 0.0, "model_id": "model"}
    pq.write_table(pa.Table.from_pylist([row]), original)
    original_bytes = original.read_bytes()
    (tmp_path / "source.json").write_text('{"run_id":"run"}')
    api = Api()
    out = materialize_source_traces(original, api=api, run_id="run")
    assert out != original and original.read_bytes() == original_bytes
    actual = pq.read_table(out).to_pylist()
    assert len(actual) == 1
    for key in row.keys() - {"response_text", "reasoning_text"}:
        assert actual[0][key] == row[key]
    assert actual[0]["reasoning_text"] == ""
    assert parse_trajectory(actual[0]["response_text"]).n_tool_calls == 2
    assert api.downloads == ["run"] and not api.requests
    manifest = json.loads((out.parent / "manifest.json").read_text())
    assert manifest["n_events"] == 5 and manifest["n_tool_calls"] == 2
    assert (out.parent / "source.json").is_file()
    api.requests.clear()
    assert materialize_source_traces(original, api=api, run_id="run") == out
    assert not api.requests
    # A corrupt final output is rebuilt from checked per-rollout caches.
    out.write_bytes(b"corrupt")
    materialize_source_traces(original, api=api, run_id="run")
    assert pq.read_table(out).num_rows == 1 and not api.requests
    assert api.downloads == ["run"]


def test_efficiency_tables_skip_raw_source_recovery_and_keep_receipts_per_subset(tmp_path):
    full = tmp_path / "dataset.parquet"
    subset = tmp_path / "dataset.selection-one.parquet"
    pq.write_table(pa.Table.from_pylist([{"problem_id": "task/a", "run_idx": 0, "reward": 0.0}]),
                   full)
    pq.write_table(pa.Table.from_pylist([
        {"problem_id": "task/a", "run_idx": 0, "selection": "one"},
    ]), subset)
    original_bytes = full.read_bytes()
    api = Api()
    assert materialize_efficiency_tables(full, api=api, run_id="run") == full
    assert full.read_bytes() == original_bytes
    assert api.downloads == ["run"] and not api.requests
    marker = json.loads((tmp_path / "efficiency-tables.json").read_text())
    assert marker["run_id"] == "run" and len(marker["trace_exports"]) == 1
    assert not list(tmp_path.glob("source-v*"))
    assert not (tmp_path / "initial-inputs").exists()
    materialize_efficiency_tables(subset, api=Api(), run_id="run")
    assert (tmp_path / "dataset.selection-one.parquet.efficiency-tables.json").is_file()
    for export in (full, subset):
        run = SimpleNamespace(parquet_paths=[export], reference_data=[{
            "run_id": "run", "export_url": "https://example.test/v1/runs/run/export/autofa",
            "parquet_path": str(export),
        }])
        assert len(locate_bulk_indexes(run)) == 1


def summary_export(tmp_path):
    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "problem_id": "task/a", "run_idx": 0, "response_text": "summary", "reasoning_text": "",
    }]), path)
    return path


def test_concurrent_builders_share_one_verified_cache(tmp_path):
    original = summary_export(tmp_path)
    api = Api()
    started = threading.Barrier(2)
    download = api.download_trace_export

    def slow_download(*args, **kwargs):
        time.sleep(0.2)
        return download(*args, **kwargs)

    api.download_trace_export = slow_download

    def build():
        started.wait()
        return materialize_source_traces(original, api=api, run_id="run")

    with ThreadPoolExecutor(2) as pool:
        first, second = pool.map(lambda _: build(), range(2))

    assert first == second
    assert pq.read_table(first).num_rows == 1
    assert api.downloads == ["run"]
    assert not list(first.parent.glob("*.partial"))


def test_legacy_run_alias_uses_catalog_canonical_identity(tmp_path):
    original = summary_export(tmp_path)
    api = Api()
    out = materialize_source_traces(original, api=api, run_id="legacy-alias")
    manifest = json.loads((out.parent / "manifest.json").read_text())
    assert manifest["run_id"] == "legacy-alias"
    assert manifest["canonical_run_id"] == "run"
    assert all(path.startswith("/v1/rollouts/run/") for path, _ in api.requests)


# --- optional execution resources ---

def record():
    return {"metadata": {"task_id": "task", "repeat": 0}, "trace": {"execution": {}}}


class ResourceApi:
    base_url = "https://api.test"

    def __init__(self):
        self.calls = []
        self.payloads = {
            "provenance": {"rows": [{"run_id": "run", "task_id": "task", "repeat": 0,
                                      "path": "/source.jsonl", "reader": "reader-v2",
                                      "api_key": "SECRET", "reward": "SECRET"}],
                           "n": 1, "grain": "rollout"},
            "relations": {"edges": [{"frm_task_id": "task", "frm_repeat": 0,
                                      "to_task_id": "child", "to_repeat": 0,
                                      "kind": "fork", "resolved": True}], "n": 1},
            "prompt_summaries": {"calls": [{"event_idx": 2, "lane": "", "class": "compaction",
                                             "chars": 50, "n_messages": 2, "reward": "SECRET"}],
                                 "n_model_calls": 2, "n_with_request": 1},
        }

    def get_trace_resource(self, run_id, task_id, repeat, resource):
        assert (run_id, task_id, repeat) == ("run", "task", 0)
        self.calls.append(resource)
        payload = self.payloads[resource]
        if isinstance(payload, Exception):
            raise payload
        return copy.deepcopy(payload)


def recover(target, api, tmp_path):
    recover_execution_resources(target, api=api, run_id="run", root=tmp_path)
    return target["trace"]["execution"]


def test_optional_resources_preserve_metadata_and_do_not_invent_full_requests(tmp_path):
    target, api = record(), ResourceApi()
    api.payloads["provenance"]["truncated"] = True
    execution = recover(target, api, tmp_path)
    assert execution["provenance"][0]["reader"] == "reader-v2"
    assert execution["relations"][0]["resolved"] is True
    assert execution["prompt_summaries"][0]["class"] == "compaction"
    assert execution["resource_metadata"]["prompt_summaries"] == {
        "n_model_calls": 2, "n_with_request": 1,
    }
    assert "SECRET" not in json.dumps(execution)
    assert "model_call_evidence" not in target["trace"]
    assert "initial_request" not in target["trace"]
    assert "summaries only" in execution["availability"]["prompt_summaries"]["reason"]
    # The source's own truncation flag is disclosed instead of reported as complete.
    assert {name: value["status"] for name, value in execution["availability"].items()} == {
        "provenance": "partial", "relations": "available", "prompt_summaries": "available",
    }
    assert execution["resource_metadata"]["provenance"]["truncated"] is True


def test_successful_fetches_are_cached_for_later_import_attempts(tmp_path):
    api = ResourceApi()
    first = recover(record(), api, tmp_path)
    second = recover(record(), api, tmp_path)
    assert first == second
    assert api.calls == ["provenance", "relations", "prompt_summaries"]


def test_missing_reader_is_explicit_and_does_not_prevent_later_recovery(tmp_path):
    target = record()
    execution = recover(target, object(), tmp_path)
    assert execution["availability"]["provenance"]["status"] == "unavailable"
    assert not list(tmp_path.glob("*.json"))
    execution = recover(target, ResourceApi(), tmp_path)
    assert execution["availability"]["provenance"]["status"] == "available"


def test_existing_complete_archive_resources_are_not_fetched_again(tmp_path):
    target, api = record(), ResourceApi()
    target["trace"]["execution"] = {
        "provenance": [{"reader": "bulk-reader"}],
        "availability": {"provenance": {"status": "available", "reason": "bulk export"}},
    }
    execution = recover(target, api, tmp_path)
    assert execution["provenance"] == [{"reader": "bulk-reader"}]
    assert api.calls == ["relations", "prompt_summaries"]


def test_transient_failures_are_raised_and_not_cached_as_absence(tmp_path):
    target, api = record(), ResourceApi()
    api.payloads["provenance"] = DataApiError("retry later", status=503)
    with pytest.raises(DataApiError, match="retry later"):
        recover(target, api, tmp_path)
    assert not list(tmp_path.glob("*.json"))
    assert "provenance" not in target["trace"]["execution"]["availability"]
    assert recover(target, ResourceApi(), tmp_path)["provenance"]


@pytest.mark.parametrize("resource,payload", [
    ("provenance", {"rows": [{"run_id": "other"}], "n": 1}),
    ("provenance", {"rows": [{"task_id": "other"}], "n": 1}),
    ("provenance", {"rows": [], "n": 1}),
    ("provenance", {"run_id": "other", "rows": [], "n": 0}),
    ("relations", {"edges": [{"frm_task_id": "other", "to_task_id": "another"}], "n": 1}),
    ("prompt_summaries", {"calls": [None]}),
], ids=["row_other_run", "row_other_task", "count_mismatch", "payload_other_run",
        "relation_elsewhere", "non_record"])
def test_malformed_or_cross_rollout_resources_are_rejected(tmp_path, resource, payload):
    api = ResourceApi()
    api.payloads[resource] = payload
    with pytest.raises(DataApiError, match="Source|Invalid source"):
        recover(record(), api, tmp_path)


@pytest.mark.parametrize("mutation", ["invalid_json", "wrong_identity"])
def test_malformed_cached_records_are_refetched(tmp_path, mutation):
    api = ResourceApi()
    recover(record(), api, tmp_path)
    for path in tmp_path.glob("*.json"):
        cached = json.loads(path.read_text())
        if mutation == "invalid_json":
            path.write_text("{bad")
            continue
        cached["identity"][1] = "other-run"
        path.write_text(json.dumps(cached))
    recover(record(), api, tmp_path)
    assert len(api.calls) == 6


def test_partial_import_is_retained_when_optional_endpoint_is_unavailable(tmp_path):
    target, api = record(), ResourceApi()
    target["trace"]["execution"] = {
        "provenance": [{"path": "/retained.json"}],
        "availability": {"provenance": {"status": "partial", "reason": "partial export"}},
    }
    api.payloads["provenance"] = DataApiError("unsupported", status=404)
    execution = recover(target, api, tmp_path)
    assert execution["provenance"] == [{"path": "/retained.json"}]
    assert execution["availability"]["provenance"]["status"] == "partial"


def test_evaluation_edges_are_excluded_by_endpoint_provenance(tmp_path):
    api = ResourceApi()
    api.payloads["relations"] = {"edges": [{"frm_task_id": "task", "frm_repeat": 0,
                                            "to_lane": "judge/verdict", "reason": "SECRET"}],
                                 "n": 1}
    execution = recover(record(), api, tmp_path)
    assert execution["relations"] == []
    assert "SECRET" not in json.dumps(execution)


def test_recovery_can_select_one_resource_without_fetching_the_others(tmp_path):
    api, target = ResourceApi(), record()
    recover_execution_resources(target, api=api, run_id="run", root=tmp_path,
                                resources=("relations",))
    assert api.calls == ["relations"]
    assert set(target["trace"]["execution"]["availability"]) == {"relations"}


@pytest.fixture
def imported_item(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    source = compact(tmp_path / "dataset.parquet")
    api = BulkApi()
    output = materialize_source_traces(source, api=api, run_id="run",
                                       catalog={"run_id": "run"})
    assert api.downloads == [("run", [("task/a", 0)])]
    item = pq.read_table(output).to_pylist()[0]
    monkeypatch.setattr(tools_data, "_load_item_raw", lambda key: item if key == "a" else None)
    monkeypatch.setattr(tools_data, "get_runtime", lambda: SimpleNamespace(
        config=SimpleNamespace(data_api_base_url=api.base_url),
    ))
    return item


def source_api(monkeypatch, respond):
    """Answer on-demand source requests with ``respond(n)``; return the requested paths."""
    calls = []

    def get(api, path, query=None):
        calls.append(path)
        return respond(len(calls))

    monkeypatch.setattr(DataApiClient, "_get_json", get)
    return calls


def no_source_requests(n):
    pytest.fail("invalid request must not access source")


def test_agent_fetch_is_explicit_scoped_cached_and_does_not_mutate_inputs(
    imported_item, monkeypatch,
):
    original = copy.deepcopy(imported_item)
    calls = source_api(monkeypatch, lambda n: {"rows": [{
        "run_id": "run", "task_id": "task/a", "repeat": 0, "reader": "reader-v2",
        "api_key": "SECRET",
    }], "n": 1})
    result = tools_data.get_trace_record("a", section="provenance")
    assert result["availability"]["status"] == "unavailable" and calls == []
    for _ in range(2):
        result = tools_data.get_trace_record("a", section="provenance", fetch_missing=True)
        assert result["ok"] and result["availability"]["status"] == "available"
        assert result["value"][0]["reader"] == "reader-v2"
        assert result["resource_metadata"] == {"n": 1}
        assert "SECRET" not in json.dumps(result)
    assert calls == ["/v1/rollouts/run/task%2Fa/0/provenance"]
    assert imported_item == original
    assert tools_data.get_trace_record("a", section="provenance")["value"] == []


def test_supplemental_text_is_pageable_without_repeating_requests(imported_item, monkeypatch):
    text = "long provenance " * 1000
    calls = source_api(monkeypatch, lambda n: {"rows": [{"path": text}], "n": 1})
    offset, chunks = 0, []
    while True:
        result = tools_data.get_trace_record(
            "a", section="provenance", fetch_missing=True, pointer="/0/path",
            offset=offset, max_chars=1000,
        )
        assert result["ok"]
        chunks.append(result["value"])
        if result["is_final"]:
            break
        offset = result["next_offset"]
    assert "".join(chunks) == text
    assert len(calls) == 1


@pytest.mark.parametrize("status", [404, 503])
def test_agent_fetch_reports_absence_or_failure_and_only_caches_absence(
    imported_item, monkeypatch, status,
):
    def unavailable(n):
        raise DataApiError("source unavailable", status=status)

    calls = source_api(monkeypatch, unavailable)
    for _ in range(2):
        result = tools_data.get_trace_record("a", section="relations", fetch_missing=True)
        if status == 503:
            assert not result["ok"] and result["error"] == "source unavailable"
        else:
            assert result["ok"] and result["availability"]["status"] == "unavailable"
    assert len(calls) == (2 if status == 503 else 1)
    assert calls[0] == "/v1/rollouts/run/task%2Fa/0/edges"


@pytest.mark.parametrize("change", ["origin", "identity", "revision", "source"])
def test_on_demand_fetch_validates_recorded_identity_before_network(
    imported_item, monkeypatch, change,
):
    envelope = json.loads(imported_item["response_text"])
    execution = envelope["execution"]
    if change == "origin":
        execution["source"]["base_url"] = "https://untrusted.test"
    elif change == "identity":
        execution["rollout"]["repeat"] = True
    elif change == "revision":  # older imports lack a valid export revision
        execution["source"]["trace_export_sha256"] = "invalid"
    else:
        execution.pop("source")
    imported_item["response_text"] = json.dumps(envelope)
    source_api(monkeypatch, no_source_requests)
    result = tools_data.get_trace_record("a", section="provenance", fetch_missing=True)
    assert not result["ok"] and result["error"]


def test_new_export_revision_does_not_reuse_older_supplemental_metadata(imported_item, monkeypatch):
    calls = source_api(monkeypatch, lambda n: {"calls": [{"class": f"version-{n}"}]})
    first = tools_data.get_trace_record("a", section="prompt_summaries", fetch_missing=True)
    envelope = json.loads(imported_item["response_text"])
    envelope["execution"]["source"]["trace_export_sha256"] = "b" * 64
    imported_item["response_text"] = json.dumps(envelope)
    second = tools_data.get_trace_record("a", section="prompt_summaries", fetch_missing=True)
    assert first["value"] != second["value"] and len(calls) == 2
    assert calls[0] == "/v1/rollouts/run/task%2Fa/0/prompts"


def test_on_demand_reader_reuses_exported_metadata_without_source_access(tmp_path):
    resource, rows = "prompt_summaries", [{"class": "compaction"}]
    execution = {"availability": {resource: {"status": "available"}}, resource: rows}
    api = ResourceApi()
    fetch_execution_resource(execution, resource, api=api, root=tmp_path)
    assert execution[resource] == rows and api.calls == []


@pytest.mark.parametrize("kwargs", [
    {"section": "events"}, {"section": "provenance", "event_idx": 0},
], ids=["unfetchable_section", "event_selector"])
def test_invalid_explicit_fetch_does_not_reach_source(imported_item, monkeypatch, kwargs):
    source_api(monkeypatch, no_source_requests)
    assert not tools_data.get_trace_record("a", fetch_missing=True, **kwargs)["ok"]
