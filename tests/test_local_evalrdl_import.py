# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local EvalRDL import: stores, shared reader goldens, batching, artifacts and wizard labels."""

import base64
import hashlib
import io
import json
import os
import shutil
import urllib.request
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from autofa import trace
from autofa.core.sandbox import run_detector
from autofa.data import local_import
from autofa.data.artifact_access import capture_artifact_reads, validate_artifact_reads
from autofa.data.artifact_store import ArtifactStore, ensure_artifact
from autofa.data.gym_adapter import materialize_eval_source
from autofa.data.input_view import parsed_input
from autofa.data.local_import import detect_reader, import_local_run, reference_for_import
from autofa.data.local_table_batches import _BatchTables, rollout_batches
from autofa.data.local_trace_store import resolve_store
from autofa.data.parquet_store import Dataset
from autofa.data.reference_data import event_url, rollout_url
from autofa.data.source_errors import SourceDataError
from autofa.data.sources import EvalSource, discover_eval_sources
from autofa.data.trace_navigation import trace_events
from autofa.wizard import dataset
from evalrdl_schema import CURRENT, polars_schemas
from evalrdl_schema.tables import RunTables
from evalrdl_trace.source import RolloutKey
from evalrdl_trace.tables import TablesSource

# Readers warn about synthetic runs' missing optional metadata.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

@pytest.fixture(autouse=True)
def offline_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(dataset, "prepare_step", lambda: None)
    monkeypatch.setattr(dataset, "console", Console(file=io.StringIO()))

    def no_network(*args, **kwargs):
        raise AssertionError("Local imports must not make network requests")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)


def gym_run(root: Path, *, text="tool output", repeat=2, edit=None) -> Path:
    """Write a NeMo Gym run; ``edit`` mutates the record in place or returns replacements."""
    root.mkdir(parents=True)
    record = {
        "_ng_task_index": 0, "_ng_rollout_index": repeat, "reward": 0.0,
        "verifier_metadata": {"task_id": "task/with ünicode"},
        "responses_create_params": {
            "model": "record-model",
            "input": [{"type": "message", "role": "user", "content": "Read the file"}],
        },
        "response": {
            "model": "record-model", "status": "completed",
            "output": [
                {"type": "function_call", "call_id": "call-a", "name": "bash",
                 "arguments": json.dumps({"command": "cat file"})},
                {"type": "function_call_output", "call_id": "call-a", "output": text},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Cannot finish"}]},
            ],
        },
    }
    records = (edit(record) if edit else None) or [record]
    (root / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records),
                                         encoding="utf-8")
    (root / "eval_metadata.json").write_text(json.dumps({"model": "model", "dataset": "toy"}))
    return root


def harbor_run(root: Path, *, text="observation") -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "jobs_dir": "jobs",
        "agents": [{"name": "agent", "model_name": "harbor-model"}],
    }))
    trial = root / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({
        "task_name": "harbor/task", "started_at": "2026-01-01T00:00:00Z",
        "agent_info": {"name": "agent", "model_info": {"name": "harbor-model"}},
        "verifier_result": {"rewards": {"reward": 0.75}},
    }))
    (trial / "agent" / "trajectory.json").write_text(json.dumps({
        "schema_version": "ATIF-v1.6",
        "steps": [
            {"step_id": 1, "source": "user", "message": "Read the file"},
            {"step_id": 2, "source": "agent", "message": "Reading",
             "tool_calls": [{"tool_call_id": "call-a", "function_name": "bash",
                             "arguments": {"command": "cat file"}}],
             "observation": {"results": [{"source_call_id": "call-a", "content": text}]}},
            {"step_id": 3, "source": "agent", "message": "Finished"},
        ],
    }))
    return root


def first_row(path):
    return pq.read_table(path).to_pylist()[0]


# Importer and managed stores


def test_efficiency_import_skips_rich_trace_dataset(tmp_path, monkeypatch):
    run = gym_run(tmp_path / "gym")

    monkeypatch.setattr("autofa.data.local_import._write_rich_dataset",
                        lambda *a, **kw: pytest.fail("rich trace dataset should not be built"))
    result = import_local_run(run, efficiency_only=True)
    manifest = json.loads((result.parquet_path.parent / "manifest.json").read_text())
    assert result.dataset_kind == manifest["dataset_kind"] == "compact"
    assert manifest["artifact_files"] == []
    assert pq.ParquetFile(result.parquet_path).metadata.num_rows == 1


def test_gym_keeps_full_text_identity_and_offline_viewer_tables(tmp_path):
    full = "α\n" * 40000 + "EXACT_END"
    run = gym_run(tmp_path / "gym", text=full)
    selected = discover_eval_sources(run)
    assert len(selected) == 1 and selected[0].kind == "evalrdl"
    result = import_local_run(run)
    row = first_row(result.parquet_path)
    assert row["model_id"] == "model" and row["benchmark"] == "toy"
    assert row["problem_id"] == "task/with ünicode" and row["run_idx"] == 2
    assert row["is_correct"] is False and row["reward"] == 0.0
    assert full in trace.observations(row)
    assert "Read the file" in row["prompt_text"]
    envelope = json.loads(row["response_text"])
    assert envelope["evidence_scope"]["identity"] == "evalrdl_tables_v1"
    assert any(s.get("source_parts") for s in envelope["steps"])
    store = resolve_store(result.store_id)
    shutil.rmtree(run)
    source = TablesSource(store / "tables")
    events = source.events(RolloutKey(result.run_id, row["problem_id"], 2), parts=True)
    assert any(p.get("text") == full for event in events for p in event["parts"])
    selected = discover_eval_sources(result.parquet_path)[0]
    assert materialize_eval_source(selected) == result.parquet_path
    assert [s.path for s in discover_eval_sources(store)] == [result.parquet_path]
    with pytest.raises(ValueError, match="normalized run tables"):
        discover_eval_sources(store / "tables")
    provenance = pq.read_table(store / "tables" / "provenance.parquet").to_pylist()
    assert len(envelope["execution"]["provenance"]) == len(provenance)


def test_gym_retains_only_allowlisted_portable_outcome_artifacts(tmp_path):
    from autofa.taxonomy.sources import LocalOutcomeSource

    def outcome(record):
        record.update({
            "test_output": "FAILED test_feature.py::test_case\nKeyError: 'language'",
            "test_results": {
                "passed": 2,
                "referencePatch": "NESTED SECRET",
                "details": {"golden.patch": "DOTTED SECRET"},
            },
            "evaluation_completed": False,
            "model_patch": "diff --git a/a.py b/a.py\n+candidate\n",
            "golden_patch": "SECRET REFERENCE PATCH",
            "reference_patch": "OTHER SECRET REFERENCE PATCH",
        })

    run = gym_run(tmp_path / "gym", edit=lambda record: (outcome(record), [record])[1])
    result = import_local_run(run)
    row = first_row(result.parquet_path)
    artifacts = LocalOutcomeSource(result.parquet_path.parent).artifacts(
        row["problem_id"], row["run_idx"],
    )
    by_name = {artifact["name"]: artifact for artifact in artifacts}
    assert by_name["test-output.log"]["text"].startswith("FAILED")
    assert json.loads(by_name["evaluation-result.json"]["text"]) == {
        "evaluation_completed": False,
        "test_results": {"details": {}, "passed": 2},
    }
    assert by_name["model.patch"]["text"].endswith("+candidate\n")
    assert "SECRET REFERENCE PATCH" not in json.dumps(artifacts)
    assert "OTHER SECRET" not in json.dumps(artifacts)


def test_artifact_parent_filter_is_scoped_to_one_rollout() -> None:
    artifact_keys = CURRENT.table("artifacts").key
    conversion = SimpleNamespace(tables={
        "events": [
            {
                "run_id": "run",
                "task_id": "evaluated",
                "repeat": 0,
                "lane": "",
                "event_idx": 0,
                "actor": "grader",
            },
            {
                "run_id": "run",
                "task_id": "candidate",
                "repeat": 0,
                "lane": "",
                "event_idx": 0,
                "actor": "agent",
            },
        ],
        "parts": [],
        "tool_calls": [],
        "artifacts": [
            {
                "run_id": "run",
                "task_id": task_id,
                "repeat": 0,
                "agent_ref": "",
                "artifact_idx": 0,
                "lane": "",
                "event_idx": 0,
            }
            for task_id in ("evaluated", "candidate")
        ],
    })

    allowed = local_import._capturable_artifact_keys(conversion, artifact_keys)

    assert ("run", "candidate", 0, "", 0) in allowed
    assert ("run", "evaluated", 0, "", 0) not in allowed


def test_evaluator_launcher_uses_top_level_outcomes_when_details_are_empty(tmp_path):
    from autofa.taxonomy.sources import LocalOutcomeSource

    run = tmp_path / "evaluator" / "swe-pro"
    run.mkdir(parents=True)
    (run / "eval-run.json").write_text(json.dumps({
        "run_id": "launcher-run",
        "timestamp": "2026-01-01T00:00:00Z",
        "n_results": 1,
        "config": {"benchmark": "swe-pro", "model": "candidate-model"},
        "benchmark": {"scores": {"pass@1": {"value": 0.0}}},
    }))
    record = {
        "problem_idx": 0,
        "repeat": 0,
        "reward": 0.0,
        "metadata": {"task_id": "instance_owner__repo-v1"},
        "scoring_details": {},
        "test_output": "compiler: cannot use candidate value",
        "test_results": {},
        "evaluation_completed": False,
        "golden_patch": "SECRET REFERENCE PATCH",
    }
    (run / "results.jsonl").write_text(json.dumps(record) + "\n")
    (run / "trajectories.jsonl").write_text(json.dumps({
        **record,
        "test_results": [{
            "passed": False,
            "golden_patch": "NESTED SECRET",
        }],
        "model_patch": "diff --git a/a.go b/a.go\n+candidate\n",
    }) + "\n")

    result = import_local_run(run)
    artifacts = LocalOutcomeSource(result.parquet_path.parent).artifacts(
        "instance_owner__repo-v1", 0,
    )
    by_name = {artifact["name"]: artifact for artifact in artifacts}
    assert by_name["test-output.log"]["text"] == record["test_output"]
    assert json.loads(by_name["evaluation-result.json"]["text"]) == {
        "evaluation_completed": False,
        "test_results": [{"passed": False}],
    }
    assert by_name["model.patch"]["text"].endswith("+candidate\n")
    assert "SECRET REFERENCE PATCH" not in json.dumps(artifacts)


def test_harbor_tool_results_labels_metadata_and_artifacts(tmp_path):
    full = "recorded observation\n" * 4000
    run = harbor_run(tmp_path / "harbor", text=full)
    artifacts = run / "trial" / "artifacts"
    artifacts.mkdir()
    artifact_text = "untruncated local artifact\n" * 4000
    (artifacts / "tool.log").write_text(artifact_text)
    result = import_local_run(run, benchmark="terminal-bench", model_id="tested-model")
    row = first_row(result.parquet_path)
    assert row["model_id"] == "tested-model" and row["benchmark"] == "terminal-bench"
    assert row["is_correct"] is True and row["reward"] == 0.75  # Exporter semantics.
    assert full.strip() in trace.observations(row)
    envelope = json.loads(row["response_text"])
    assert any(step.get("observation") == full for step in envelope["steps"])
    assert envelope["execution"]["artifacts"][0]["text"] == artifact_text
    manifest = json.loads((result.parquet_path.parent / "manifest.json").read_text())
    captured = manifest["artifact_files"][0]
    assert (result.parquet_path.parent / captured["path"]).read_text() == artifact_text
    assert captured["task_id"] == "harbor/task"


def test_harbor_retains_complete_verifier_reports_logs_and_spills(tmp_path):
    from autofa.taxonomy.sources import LocalOutcomeSource

    run = harbor_run(tmp_path / "harbor")
    trial = run / "trial"
    result_path = trial / "result.json"
    result_doc = json.loads(result_path.read_text())
    result_doc["verifier_result"]["diagnostics"] = {
        "summary": "typed failure",
        "reference_patch": "SECRET REFERENCE PATCH",
        "referencePatch": "CAMEL SECRET REFERENCE PATCH",
        "golden.patch": "DOTTED SECRET REFERENCE PATCH",
    }
    result_doc["verifier_result"]["golden_patch"] = "ANOTHER SECRET"
    result_path.write_text(json.dumps(result_doc))
    verifier = trial / "verifier"
    verifier.mkdir()
    (verifier / "ctrf.json").write_text(json.dumps({
        "reportFormat": "CTRF",
        "specVersion": "1.0.0",
        "results": {
            "summary": {"tests": 1, "passed": 0, "failed": 1},
            "tests": [{
                "name": "test",
                "status": "failed",
                "referencePatch": "REPORT SECRET",
            }],
        },
    }))
    (verifier / "test-stdout.txt").write_text("KeyError: 'language'\n")
    (verifier / "run-script-stderr.txt").write_text("cannot use &fakeNative{}\n")
    tool_results = trial / "agent" / "sessions" / "projects" / "p" / "s" / "tool-results"
    tool_results.mkdir(parents=True)
    (tool_results / "result.txt").write_text("complete spilled result")

    imported = import_local_run(run)
    artifacts = LocalOutcomeSource(imported.parquet_path.parent).artifacts("harbor/task", 0)
    by_name = {artifact["name"]: artifact for artifact in artifacts}
    verifier_result = json.loads(by_name["verifier-result.json"]["text"])
    assert verifier_result["diagnostics"] == {"summary": "typed failure"}
    assert "SECRET REFERENCE PATCH" not in by_name["verifier-result.json"]["text"]
    assert "CAMEL SECRET" not in by_name["verifier-result.json"]["text"]
    assert "DOTTED SECRET" not in by_name["verifier-result.json"]["text"]
    assert "ANOTHER SECRET" not in by_name["verifier-result.json"]["text"]
    assert "REPORT SECRET" not in by_name["ctrf.json"]["text"]
    assert by_name["test-stdout.txt"]["text"] == "KeyError: 'language'\n"
    assert by_name["run-script-stderr.txt"]["text"] == "cannot use &fakeNative{}\n"
    assert by_name["result.txt"]["role"] == "trace"
    item = first_row(imported.parquet_path)
    spilled = trace.artifact(item, by_name["result.txt"]["artifact_idx"])
    assert "complete spilled result" in "".join(
        segment["text"] for segment in spilled["segments"]
    )


def test_cache_is_portable_and_invalidates_for_changed_bytes_or_metadata_override(
    tmp_path, monkeypatch,
):
    source = gym_run(tmp_path / "run")
    first = import_local_run(source)
    assert import_local_run(source) == first
    moved = tmp_path / "moved"
    shutil.copytree(source, moved)
    assert import_local_run(moved) == first
    overridden = import_local_run(source, model_id="explicit", benchmark="other")
    assert overridden.store_id != first.store_id and overridden.run_id == first.run_id
    assert pq.read_table(overridden.parquet_path)["model_id"].to_pylist() == ["explicit"]
    (source / "eval_metadata.json").write_text('{"model": "new-model"}')
    changed = import_local_run(source)
    assert changed.store_id != first.store_id
    assert resolve_store(first.store_id).is_dir()
    # An importer revision creates a new snapshot without mutating the previous import.
    previous_bytes = first.parquet_path.read_bytes()
    monkeypatch.setattr(local_import, "IMPORT_VERSION", local_import.IMPORT_VERSION + 1)
    revised = import_local_run(moved)
    assert revised.store_id != first.store_id and revised.run_id == first.run_id
    assert first.parquet_path.read_bytes() == previous_bytes


def test_model_call_captures_linked_from_a_shared_store_are_read(tmp_path, monkeypatch):
    def captured_call(record):
        record["ng_model_call_capture"] = {"calls": [
            {"model_call_id": "call-1", "dialect": "chat", "call_index": 0},
        ]}

    capture = json.dumps({
        "model_call_id": "call-1",
        "request": {"temperature": 0.37,
                    "messages": [{"role": "user", "content": "LINKED_REQUEST_MESSAGE"}]},
    }) + "\n"
    shared = tmp_path / "shared-captures"
    shared.mkdir()
    (shared / "call-1.capture.jsonl").write_text(capture)
    runs = {}
    for kind in ("regular", "linked"):
        run = gym_run(tmp_path / kind, edit=captured_call)
        captures = run / "artifacts" / "model_calls"
        captures.mkdir(parents=True)
        if kind == "regular":
            (captures / "call-1.capture.jsonl").write_text(capture)
        else:
            (captures / "call-1.capture.jsonl").symlink_to(shared / "call-1.capture.jsonl")
        # A separate profile per import keeps the second from reusing the first snapshot.
        monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / f"{kind}-home"))
        runs[kind] = import_local_run(run)

    regular, linked = runs["regular"], runs["linked"]
    assert linked.store_id == regular.store_id
    assert linked.parquet_path != regular.parquet_path
    row = first_row(linked.parquet_path)
    assert row == first_row(regular.parquet_path)
    # The rollout does not hold this parameter; only the linked capture does. Gym's
    # inline record keeps the call's messages, so the capture's copy is not imported.
    assert "temperature=0.37" in row["response_text"]
    assert "LINKED_REQUEST_MESSAGE" not in json.dumps(row)


@pytest.mark.parametrize(("name", "target"), [
    ("notes.txt", "secret.txt"),
    ("notes.capture.jsonl", "secret.txt"),
    ("rollouts.jsonl", "other.capture.jsonl"),
])
def test_links_to_other_files_outside_the_run_folder_are_refused(tmp_path, name, target):
    outside = tmp_path / target
    outside.write_text("{}\n")
    run = gym_run(tmp_path / "gym")
    (run / name).unlink(missing_ok=True)
    # Whoever made the folder chooses the link's name, but not its target's; a capture
    # linked under another name would be read as run records, not as a capture.
    (run / name).symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the selected run folder"):
        import_local_run(run)


def test_linked_capture_read_as_another_input_is_refused(tmp_path):
    outside = tmp_path / "other.capture.jsonl"
    outside.write_text('{"model_call_id": "call-1"}\n')
    run = harbor_run(tmp_path / "harbor")
    (run / "trial" / "artifacts").mkdir()
    # Harbor keeps every file under artifacts/, so the link would be stored in full.
    (run / "trial" / "artifacts" / "notes.capture.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="was read as run data"):
        import_local_run(run)
    assert not list((Path(os.environ["AUTOFA_HOME"]) / "sources").glob("*/manifest.json"))


def test_corrupt_or_incomplete_store_never_reused(tmp_path):
    result = import_local_run(gym_run(tmp_path / "run"))
    table = result.parquet_path.parent / "tables" / "part_texts.parquet"
    table.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        resolve_store(result.store_id)
    with pytest.raises(ValueError):
        resolve_store("../tables")
    manifest = result.parquet_path.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    data["status"] = "importing"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest"):
        resolve_store(result.store_id)


def test_source_sidecar_cannot_redirect_an_unrelated_selected_parquet(tmp_path):
    result = import_local_run(gym_run(tmp_path / "run"))
    sibling = result.parquet_path.with_name("unrelated.parquet")
    shutil.copyfile(result.parquet_path, sibling)
    with pytest.raises(ValueError, match="does not belong"):
        reference_for_import(sibling)
    sidecar = result.parquet_path.parent / "source.json"
    reference = json.loads(sidecar.read_text())
    reference["run_id"] = "wrong-run"
    sidecar.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="disagree"):
        reference_for_import(result.parquet_path)


def test_store_without_dataset_kind_must_be_re_imported(tmp_path):
    result = import_local_run(gym_run(tmp_path / "run"))
    manifest = result.parquet_path.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    del data["dataset_kind"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(SourceDataError, match="predates the current store format"):
        reference_for_import(result.parquet_path)
    with pytest.raises(SystemExit, match="re-import the run"):
        dataset.reuse_reference_data([result.reference_data()])


def test_quarantined_record_persists_diagnostics_without_completed_dataset(tmp_path):
    run = gym_run(tmp_path / "run")
    with (run / "rollouts.jsonl").open("a") as handle:
        handle.write('{"_ng_task_index": broken\n')
    with pytest.raises(SourceDataError, match="quarantined"):
        import_local_run(run)
    records = list((tmp_path / "home" / "sources" / "quarantine").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["quarantined_rollouts"]
    assert not list((tmp_path / "home" / "sources").glob("*/dataset.parquet"))


def test_mixed_directory_keeps_gym_harbor_and_legacy_inputs(tmp_path):
    root = tmp_path / "mixed"
    gym_run(root / "gym")
    harbor_run(root / "harbor")
    (root / "legacy.jsonl").write_text(json.dumps({
        "task_id": "legacy", "is_resolved": False,
        "steps": [{"source": "assistant", "message": "Legacy trace"}],
    }) + "\n")
    sources = discover_eval_sources(root)
    assert {(s.kind, s.path.name) for s in sources} == {
        ("evalrdl", "gym"), ("evalrdl", "harbor"), ("jsonl", "legacy.jsonl"),
    }
    legacy = next(s for s in sources if s.kind == "jsonl")
    row = first_row(materialize_eval_source(legacy))
    assert row["problem_id"] == "legacy" and "Legacy trace" in row["response_text"]


def test_selected_jsonl_does_not_import_neighboring_rollout_stream(tmp_path):
    run = gym_run(tmp_path / "run")
    (run / "unrelated.jsonl").write_text('{"_ng_task_index": malformed\n')
    result = import_local_run(run / "rollouts.jsonl")
    assert pq.ParquetFile(result.parquet_path).metadata.num_rows == 1


def reuse(result, monkeypatch, **kwargs):
    seen = {}
    monkeypatch.setattr(dataset, "_finish_selection", lambda paths, **kw:
                        seen.update(paths=paths, **kw) or seen)
    return dataset.reuse_reference_data([result.reference_data()], **kwargs)


def test_saved_local_reference_reopens_without_original_logs(tmp_path, monkeypatch):
    run = gym_run(tmp_path / "run")
    result = import_local_run(run)
    shutil.rmtree(run)
    recovered = reuse(result, monkeypatch)
    assert recovered["paths"] == [result.parquet_path]
    assert recovered["reference_data"][0]["store_id"] == result.store_id


def test_saved_compact_reference_hydrates_traces_for_discovery(tmp_path, monkeypatch):
    result = import_local_run(gym_run(tmp_path / "run"), efficiency_only=True)
    recovered = reuse(result, monkeypatch, efficiency_only=False)
    assert not recovered["paths"][0].is_relative_to(result.parquet_path.parent)
    assert "tool output" in first_row(recovered["paths"][0])["response_text"]
    assert recovered["reference_data"][0]["parquet_path"] == str(recovered["paths"][0])


def test_recorded_requests_resolve_message_and_tool_hashes_without_truncation(tmp_path):
    content = "Long captured request 🧭\n" * 4000
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
    }}}

    def capture(record):
        record["ng_model_call_capture"] = {"calls": [
            {"model_call_id": "model-call-1", "dialect": "chat", "call_index": 0},
        ]}
        record["ng_trajectory"] = {"model_calls": [
            {"model_call_id": "model-call-1", "request": {
                "messages": [{"role": "user", "content": content}], "tools": [tool],
                "model": "captured-model", "temperature": 0.25,
            }},
        ]}

    row = first_row(import_local_run(gym_run(tmp_path / "gym", edit=capture)).parquet_path)
    envelope = json.loads(row["response_text"])
    evidence = envelope["model_call_evidence"]
    call = evidence["calls"][0]
    assert evidence["messages"][call["request"][0]]["content"] == content
    assert evidence["tools"][call["tools"]][0] == tool
    assert call["request_metadata"]["temperature"] == 0.25
    assert len({e["lane"] for e in envelope["execution"]["events"]}) == 2


@pytest.mark.parametrize("captured_simulator", [False, True])
def test_greeting_before_first_user_keeps_exported_task_input(tmp_path, captured_simulator):
    system = "You are the support agent."
    task = '[{"role":"system","content":"Literal customer question"}]'

    def greeting(record):
        record["responses_create_params"]["input"] = [
            {"type": "message", "role": "system", "content": system},
        ]
        record["response"]["output"] = [
            {"type": "message", "role": "assistant", "content": "How may I help?"},
            {"type": "message", "role": "user", "content": task},
            {"type": "message", "role": "assistant", "content": "Let me check."},
        ]
        record["result"] = {"messages": record["response"]["output"]}
        if captured_simulator:
            record["ng_model_call_capture"] = {"calls": [
                {"model_call_id": "simulator", "dialect": "chat", "call_index": 0},
            ]}
            record["ng_trajectory"] = {"model_calls": [{
                "model_call_id": "simulator", "request": {"messages": [
                    {"role": "system", "content": "SIMULATOR_PRIVATE_INSTRUCTIONS"},
                ]},
            }]}

    row = first_row(import_local_run(gym_run(tmp_path / "gym", edit=greeting)).parquet_path)
    assert row["prompt_text"] == task
    view = parsed_input(row)
    assert view["origin"] == "frontier_compact_export"
    assert view["messages"] == [
        {"role": "system", "content": system}, {"role": "user", "content": task},
    ]
    assert "SIMULATOR_PRIVATE_INSTRUCTIONS" not in view["text"]
    envelope = json.loads(row["response_text"])
    assert "initial_request" not in envelope
    assert [step["source"] for step in envelope["steps"]] == [
        "system", "assistant", "user", "assistant",
    ]
    if captured_simulator:
        evidence = envelope["model_call_evidence"]
        assert evidence["calls"][0]["source_lane"] == "model_call"
        assert evidence["messages"][0]["content"] == "SIMULATOR_PRIVATE_INSTRUCTIONS"


def test_complete_leading_context_preserves_developer_and_all_input_turns(tmp_path):
    messages = [
        {"role": "system", "content": "System rules"},
        {"role": "developer", "content": "Developer instructions"},
        {"role": "user", "content": "Initial question"},
        {"role": "user", "content": "More context"},
    ]

    def context(record):
        record["responses_create_params"]["input"] = [
            {"type": "message", **message} for message in messages
        ]

    row = first_row(import_local_run(gym_run(tmp_path / "gym", edit=context)).parquet_path)
    assert parsed_input(row)["messages"] == messages


def test_local_dataset_produces_exact_local_event_and_part_links(tmp_path):
    result = import_local_run(gym_run(tmp_path / "gym"))
    item = Dataset([result.parquet_path]).load_items(limit=1)[0]
    observation = next(e for e in trace_events(item) if e["source"] == "tool")
    part = observation["sections"][0]["source_parts"][0]
    source = {**observation["source_event"], "part_idx": part["part_idx"]}
    url = event_url(rollout_url(result.reference_data(), problem_id=item["problem_id"],
                               run_idx=item["run_idx"]), source)
    assert url.startswith(f"/traces/{result.store_id}/#/rollout/{result.run_id}/")
    assert f"/e{source['event_idx']}-p{source['part_idx']}?lane=" in url
    assert "task%252Fwith%20%C3%BCnicode/2/" in url


def test_public_harbor_dispatch_does_not_claim_an_unrelated_config(tmp_path):
    (tmp_path / "config.json").write_text('{"job_name": "other application"}')
    assert detect_reader(tmp_path) is None


# Bounded table hydration


def test_many_rollouts_share_disk_reads_without_mixing_evidence(tmp_path, monkeypatch):
    def twelve_rollouts(seed):
        records = []
        for i in range(12):
            row = deepcopy(seed)
            row["_ng_task_index"] = i // 2
            row["_ng_rollout_index"] = i % 2
            row["verifier_metadata"]["task_id"] = f"task {i // 2}"
            row["response"]["output"][1]["output"] = f"unique observation {i} END"
            records.append(row)
        return records

    run = gym_run(tmp_path / "run", edit=twelve_rollouts)
    scans = Counter()
    original_scan = RunTables.scan

    def scan(source, table):
        scans[table] += 1
        return original_scan(source, table)

    monkeypatch.setattr(RunTables, "scan", scan)
    result = import_local_run(run)
    rows = pq.read_table(result.parquet_path).to_pylist()
    assert len(rows) == 12
    for row in rows:
        number = int(row["problem_id"].split()[-1]) * 2 + row["run_idx"]
        observations = trace.observations(row)
        assert observations.splitlines()[-1] == f"unique observation {number} END"
        assert observations.count("unique observation") == 1
        provenance = json.loads(row["response_text"])["execution"]["provenance"]
        assert any(p["task_id"] is None for p in provenance)
        assert all(p["task_id"] in {None, row["problem_id"]} for p in provenance)
    # Export, batch sizing and hydration read once each, not once per rollout.
    assert scans["part_texts"] <= 3
    assert pq.ParquetFile(result.parquet_path).num_row_groups == 1


def write_tables(directory, values):
    dtypes = polars_schemas(CURRENT)
    for name, rows in values.items():
        cols = dtypes[name]
        pl.DataFrame([{c: row.get(c) for c in cols} for row in rows], schema=cols,
                     orient="row").write_parquet(directory / f"{name}.parquet")
    return RunTables(directory, {t.name: tuple(t.column_names) for t in CURRENT.tables})


def test_batch_scope_retains_global_provenance_and_relations_with_nullable_repeats(tmp_path):
    source = write_tables(tmp_path, {
        "provenance": [
            {"file": "global", "task_id": None, "repeat": None},
            {"file": "task", "task_id": "a", "repeat": None},
            {"file": "repeat", "task_id": None, "repeat": 1},
            {"file": "exact", "task_id": "a", "repeat": 1},
            {"file": "wrong-pair", "task_id": "a", "repeat": 2},
            {"file": "other", "task_id": "c", "repeat": 1},
        ],
        "relations": [
            {"reason": "from", "frm_task_id": "a", "frm_repeat": None},
            {"reason": "to", "to_task_id": "b", "to_repeat": 2},
            {"reason": "wrong-pair", "frm_task_id": "a", "frm_repeat": 2},
        ],
    })
    batch = _BatchTables(source, [{"problem_id": "a", "run_idx": 1},
                                  {"problem_id": "b", "run_idx": 2}])
    assert set(batch.scan("provenance").collect()["file"]) == {
        "global", "task", "repeat", "exact",
    }
    assert set(batch.scan("relations").collect()["reason"]) == {"from", "to"}


def test_text_budget_counts_full_payloads_without_size_metadata(tmp_path):
    source = write_tables(tmp_path, {
        "prompt_messages": [{"task_id": "large", "repeat": 0, "msg_json": "x" * 9000}],
    })
    path = tmp_path / "compact.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"problem_id": task, "run_idx": 0} for task in ("first", "large", "last", "end")
    ]), path)
    batches = list(rollout_batches(source, pq.ParquetFile(path),
                                   text_budget=1000, rollout_limit=2))
    assert [[r["problem_id"] for r in rows] for rows, _ in batches] == [
        ["first"], ["large"], ["last", "end"],
    ]
    assert batches[1][1].scan("prompt_messages").collect()["msg_json"][0] == "x" * 9000


# Detector artifacts captured from local runs


@pytest.fixture
def local_artifacts(tmp_path, monkeypatch):
    def no_remote(self):
        raise AssertionError("Local artifacts must not construct a remote source adapter")

    monkeypatch.setattr(ArtifactStore, "_provider", no_remote)
    original = harbor_run(tmp_path / "run")
    directory = original / "trial" / "artifacts"
    directory.mkdir()
    body = b"First line\nIMPORTANT evidence\n"
    (directory / "evidence.txt").write_bytes(body)
    pil = pytest.importorskip("PIL.Image")
    with io.BytesIO() as stream:
        pil.new("RGB", (8, 6), "red").save(stream, format="PNG")
        image_bytes = stream.getvalue()
    (directory / "image.png").write_bytes(image_bytes)
    result = import_local_run(original)
    item = first_row(result.parquet_path)
    records = trace.artifacts(item)
    text_idx = next(row["artifact_idx"] for row in records if row["name"].endswith("evidence.txt"))
    image_idx = next(row["artifact_idx"] for row in records if row["name"].endswith("image.png"))
    shutil.rmtree(original)
    return item, result.parquet_path.parent, text_idx, image_idx, body, image_bytes


def test_local_text_image_and_detector_share_verified_bytes_and_receipts(local_artifacts):
    item, directory, text_idx, image_idx, body, image_bytes = local_artifacts
    with capture_artifact_reads() as receipts:
        evidence = trace.artifact(item, text_idx)
        image = trace.artifact_image(item, image_idx)
    assert evidence["status"] == "available" and evidence["verified"] and evidence["complete"]
    assert "".join(s["text"] for s in evidence["segments"]) == body.decode()
    assert image["status"] == "available" and image["data_url"].startswith("data:image/png;base64,")
    assert image["artifact_sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert validate_artifact_reads(item, receipts)
    source = ("from autofa.trace import artifact\ndef detect(item):\n"
              f"    segments = artifact(item, {text_idx})['segments']\n"
              "    return 'IMPORTANT' in ''.join(s['text'] for s in segments)\n")
    localisation = {}
    assert run_detector(source, item, localisation=localisation) is True
    assert validate_artifact_reads(item, localisation["artifact_reads"])
    # Repair cache damage using captured originals, without the deleted input directory.
    Path(evidence["download_path"]).write_bytes(b"damaged cached content")
    assert trace.artifact(item, text_idx)["sha256"] == hashlib.sha256(body).hexdigest()
    assert validate_artifact_reads(item, receipts)
    assert directory.is_dir()


def test_inline_input_image_is_captured_after_original_logs_disappear(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    with io.BytesIO() as stream:
        pil.new("RGB", (8, 6), "blue").save(stream, format="PNG")
        image_bytes = stream.getvalue()
    original = harbor_run(tmp_path / "run")
    trajectory = original / "trial" / "agent" / "trajectory.json"
    document = json.loads(trajectory.read_text())
    document["steps"][0]["message"] = [
        {"type": "input_text", "text": "Inspect this image"},
        {
            "type": "input_image",
            "image_url": (
                "data:image/png;base64,"
                + base64.b64encode(image_bytes).decode("ascii")
            ),
        },
    ]
    trajectory.write_text(json.dumps(document))

    result = import_local_run(original)
    item = first_row(result.parquet_path)
    image_idx = next(
        row["artifact_idx"]
        for row in trace.artifacts(item)
        if row.get("mime") == "image/png"
    )
    shutil.rmtree(original)

    image = trace.artifact_image(item, image_idx)
    assert image["status"] == "available" and image["verified"]
    assert image["artifact_sha256"] == hashlib.sha256(image_bytes).hexdigest()


@pytest.mark.parametrize("mutation", [
    {"source": {"store_id": "../escape"}},
    {"source": {"store_id": "f" * 64}},
    {"source": {"run_id": "foreign-run"}},
    {"source": {"run_id": "foreign-run"}, "rollout": {"run_id": "foreign-run"}},
    {"rollout": {"task_id": "foreign-task"}},
    {"rollout": {"agent_ref": "foreign-agent"}},
])
def test_local_artifacts_refuse_foreign_source_or_rollout(local_artifacts, mutation):
    item, _, text_idx, _, _, _ = local_artifacts
    envelope = json.loads(item["response_text"])
    for section in ("source", "rollout"):
        envelope["execution"][section].update(mutation.get(section, {}))
    result = ensure_artifact({**item, "response_text": json.dumps(envelope)}, text_idx)
    assert result["status"] in {"error", "unavailable"} and not result["verified"]
    assert "path" not in result


@pytest.mark.parametrize(
    "mutation", ["missing", "digest", "size", "path", "symlink", "agent", "ambiguous"],
)
def test_local_artifacts_validate_captured_inventory_and_bytes(local_artifacts, tmp_path, mutation):
    item, directory, text_idx, _, _, _ = local_artifacts
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = next(row for row in manifest["artifact_files"] if row["artifact_idx"] == text_idx)
    artifact = directory / record["path"]
    if mutation == "missing":
        manifest["artifact_files"].remove(record)
    elif mutation == "digest":
        artifact.write_bytes(b"different captured bytes")
    elif mutation == "size":
        record["size_bytes"] += 1
    elif mutation == "path":
        record["path"] = "../../outside.txt"
    elif mutation == "symlink":
        outside = tmp_path / "outside"
        artifact.rename(outside)
        artifact.symlink_to(outside)
    elif mutation == "agent":
        record["agent_ref"] = "foreign-agent"
    else:
        manifest["artifact_files"].append(dict(record))
    manifest_path.write_text(json.dumps(manifest))
    result = ensure_artifact(item, text_idx)
    assert result["status"] in {"error", "unavailable"} and not result["verified"]
    assert "path" not in result


# Wizard labels for local runs


def wire_prompt(monkeypatch, pipe):
    def application(**kwargs):
        app = Application(input=pipe, output=DummyOutput(), **kwargs)
        app.ttimeoutlen = app.timeoutlen = 0.01  # Resolve a lone Escape without waiting.
        return app

    monkeypatch.setattr("prompt_toolkit.application.Application", application)


def test_harbor_wizard_labels_missing_names_without_mutating_existing_store(tmp_path, monkeypatch):
    run = harbor_run(tmp_path / "harbor")
    original = import_local_run(run)
    original_bytes = original.parquet_path.read_bytes()
    source_bytes = (run / "config.json").read_bytes()
    monkeypatch.setattr(dataset, "browse_filesystem", lambda *a, **kw: run)
    with create_pipe_input() as pipe:
        wire_prompt(monkeypatch, pipe)
        pipe.send_text("tested-model\rMy benchmark / v2\r")
        selected = dataset._run_local_wizard(default_folder=str(run))

    assert (selected.model_id, selected.benchmarks) == ("tested-model", ["My benchmark / v2"])
    assert selected.source_paths == [str(run)]
    [ref] = selected.reference_data
    assert ref["store_id"] != original.store_id and ref["run_id"] == original.run_id
    assert resolve_store(ref["store_id"]) / "dataset.parquet" == selected.parquet_paths[0]
    row = first_row(selected.parquet_paths[0])
    assert row["model_id"] == "tested-model" and row["benchmark"] == "My benchmark / v2"
    assert json.loads(row["response_text"])["execution"]["source"]["store_id"] == ref["store_id"]
    run_row = first_row(selected.parquet_paths[0].parent / "tables/runs.parquet")
    assert run_row["model"] == "tested-model" and run_row["benchmark"] == "My benchmark / v2"
    assert original.parquet_path.read_bytes() == original_bytes
    assert (run / "config.json").read_bytes() == source_bytes
    assert pq.read_table(original.parquet_path)["model_id"].to_pylist() == ["unknown"]


def test_only_missing_field_is_offered_and_known_name_survives(tmp_path, monkeypatch):
    run = harbor_run(tmp_path / "harbor")
    (run / "eval_metadata.json").write_text(json.dumps({"benchmark": "Raw-Name / 2026"}))
    seen = []
    monkeypatch.setattr(dataset, "_prompt_local_metadata", lambda source, missing: (
        seen.append((source, missing)) or {"model_id": "tested-model"}
    ))
    paths = dataset._prepare_local_sources(discover_eval_sources(run))
    assert seen == [(run, ["model_id"])]
    row = first_row(paths[0])
    assert row["model_id"] == "tested-model" and row["benchmark"] == "Raw-Name / 2026"


@pytest.mark.parametrize("answer", [{}, None], ids=["skip", "back"])
def test_optional_labels_do_not_change_unknown_snapshot(tmp_path, monkeypatch, answer):
    # Harbor's agent model reference is not mistaken for the tested model.
    run = harbor_run(tmp_path / "harbor")
    original = import_local_run(run)
    monkeypatch.setattr(dataset, "_prompt_local_metadata", lambda *args: answer)
    result = dataset._prepare_local_sources(discover_eval_sources(run))
    assert result == ([original.parquet_path] if answer == {} else None)
    row = first_row(original.parquet_path)
    assert row["model_id"] == row["benchmark"] == "unknown"
    assert len(list(original.parquet_path.parent.parent.glob("*/manifest.json"))) == 1


def test_each_selected_source_gets_its_own_labels_and_reference(tmp_path, monkeypatch):
    first = harbor_run(tmp_path / "first", text="First source")
    second = harbor_run(tmp_path / "second", text="Second source")
    seen = []

    def label(source, missing):
        seen.append(source)
        return {"model_id": source.name + "-model", "benchmark": source.name + "-benchmark"}

    monkeypatch.setattr(dataset, "_prompt_local_metadata", label)
    paths = dataset._prepare_local_sources([
        *discover_eval_sources(first), *discover_eval_sources(second),
    ])
    assert seen == [first, second]
    refs = [reference_for_import(path) for path in paths]
    assert refs[0]["store_id"] != refs[1]["store_id"]
    for source, path, ref in zip((first, second), paths, refs, strict=True):
        row = first_row(path)
        assert row["model_id"] == source.name + "-model"
        assert row["benchmark"] == source.name + "-benchmark"
        execution_source = json.loads(row["response_text"])["execution"]["source"]
        assert execution_source["store_id"] == ref["store_id"]


def test_existing_managed_and_legacy_parquets_do_not_prompt(tmp_path, monkeypatch):
    managed = import_local_run(harbor_run(tmp_path / "harbor"))
    legacy = tmp_path / "legacy.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "problem_id": "task", "model_id": "unknown", "benchmark": "unknown",
        "prompt_text": "Task", "response_text": "Attempt", "is_correct": False,
    }]), legacy)
    monkeypatch.setattr(dataset, "_prompt_local_metadata", lambda *a:
                        pytest.fail("Existing datasets must stay unchanged"))
    paths = dataset._prepare_local_sources([
        EvalSource(kind="parquet", path=managed.parquet_path),
        EvalSource(kind="parquet", path=legacy),
    ])
    assert paths == [managed.parquet_path, legacy]


@pytest.mark.parametrize(("keys", "expected"), [("\r\r", {}), ("draft\x1b", None)],
                         ids=["skip", "escape_goes_back"])
def test_optional_metadata_keyboard_navigation(tmp_path, monkeypatch, keys, expected):
    with create_pipe_input() as pipe:
        wire_prompt(monkeypatch, pipe)
        pipe.send_text(keys)
        assert dataset._prompt_local_metadata(
            tmp_path / "source", ["model_id", "benchmark"],
        ) == expected
