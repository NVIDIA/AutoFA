# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace archives, source traces, bulk imports, artifacts, source model calls and fake urllib
responses."""

import copy
import hashlib
import io
import json
import tempfile
import warnings
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from autofa.data.artifact_store import ArtifactStore
from autofa.data.remote_api import DataApiClient
from autofa.data.source_inputs import recover_initial_inputs


# Build the real trace ZIP contract from hydrated test rollout records.
def write_trace_archive(dest, records, *, run_id="run", mutate=None):
    tables = {
        name: []
        for name in (
            "rollouts",
            "events",
            "parts",
            "part_texts",
            "tool_calls",
            "tool_call_texts",
            "attempts",
            "artifacts",
            "artifact_texts",
            "tool_defs",
            "tool_def_texts",
        )
    }
    for record in records:
        for name in record.get("tables", {}):
            tables.setdefault(name, [])
    listed = []
    for original in records:
        record = copy.deepcopy(original)
        meta = record["metadata"]
        meta.setdefault("agent_ref", "")
        ident = {k: meta[k] for k in ("run_id", "task_id", "repeat", "agent_ref")}
        before = {t: len(rows) for t, rows in tables.items()}
        tables["rollouts"].append(meta)
        for ev in record.get("events", []):
            parts = ev.pop("parts", [])
            ev = {**ident, "lane": "", **ev}
            tables["events"].append(ev)
            for part in parts:
                part = {**ident, "lane": ev["lane"], "event_idx": ev["event_idx"], **part}
                text = part.pop("text", None)
                tables["parts"].append(part)
                if text is not None:
                    tables["part_texts"].append(
                        {
                            **{k: part[k] for k in (*ident, "lane", "event_idx", "part_idx")},
                            "text": text,
                        }
                    )
        for call in record.get("calls", []):
            call = {**ident, "lane": "", **call}
            args = call.pop("args_full", None)
            tables["tool_calls"].append(call)
            if args is not None:
                tables["tool_call_texts"].append(
                    {
                        **{k: call[k] for k in (*ident, "lane", "event_idx", "call_idx")},
                        "args_full": args,
                    }
                )
        for name, rows in record.get("tables", {}).items():
            tables[name].extend({**ident, **row} for row in rows)
        listed.append(
            {
                **{k: meta[k] for k in ("task_id", "repeat", "agent_ref")},
                "n_events": meta["n_events"],
                "n_tool_calls": meta["n_tool_calls"],
                "rows": {t: len(rows) - before[t] for t, rows in tables.items()},
            }
        )
    manifest = {
        "exporter": "trace",
        "exporter_version": 1,
        "schema": "store-v16",
        "run_id": run_id,
        "n_rollouts": len(listed),
        "rollouts": listed,
        "tables": {t: {"file": f"{t}.parquet", "rows": len(rows)} for t, rows in tables.items()},
    }
    if mutate:
        mutate(tables, manifest)
    with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(dest, "w") as archive:
        for name, rows in tables.items():
            path = Path(temp) / f"{name}.parquet"
            # Empty tables still declare the full identity fields.
            table = (
                pa.Table.from_pylist([
                    {key: row.get(key) for key in dict.fromkeys(k for r in rows for k in r)}
                    for row in rows
                ])
                if rows
                else pa.table(
                    {
                        "run_id": pa.array([], type=pa.string()),
                        "task_id": pa.array([], type=pa.string()),
                        "repeat": pa.array([], type=pa.int64()),
                        "agent_ref": pa.array([], type=pa.string()),
                    }
                )
            )
            pq.write_table(table, path)
            archive.write(path, f"{name}.parquet")
        archive.writestr("manifest.json", json.dumps(manifest))
    return Path(dest)


# Synthetic full source trace shared by public and remote adapter tests.
def fixture():
    ident = {"run_id": "run", "task_id": "task/a", "repeat": 0}
    metadata = {**ident, "n_events": 5, "n_tool_calls": 2, "agent_ref": "hermes",
                "n_artifacts": 0, "reward": 0.0, "is_scored": True, "status": "failed"}

    def event(idx, kind, role, text=None, calls=0):
        parts = [] if text is None else [{**ident, "event_idx": idx, "part_idx": 0,
            "part_type": "text", "channel": "content", "text": text, "chars": len(text)}]
        return {**ident, "event_idx": idx, "kind": kind, "role": role,
                "n_parts": len(parts), "parts": parts, "n_tool_calls": calls}

    events = [event(0, "message", "user", "fix it"),
              event(1, "tool_call", None, calls=1),
              event(2, "tool_call", None, calls=1),
              event(3, "tool_result", "tool", "Traceback: " + "x" * 6000),
              event(4, "message", "assistant", "done")]
    events[3].update(paired_event_idx=1, paired_call_idx=0)
    calls = [{**ident, "event_idx": i, "call_idx": 0, "tool_name": "terminal",
              "args_full": '{"command":"test"}', "args_chars": 18,
              "result_event_idx": 3 if i == 1 else None, "status": "failed"} for i in (1, 2)]
    return metadata, events, calls


# Bulk trace.
def record(*, lane="", task="task/a", agent="hermes"):
    meta, events, calls = fixture()
    for row in (meta, *events, *calls):
        row.update(task_id=task, agent_ref=agent)
    for event in events:
        event["lane"] = lane
        for part in event["parts"]:
            part.update(task_id=task, agent_ref=agent, lane=lane)
    for call in calls:
        call["lane"] = lane
    return {"metadata": meta, "events": events, "calls": calls}


class Api:
    base_url = "https://api.test"
    # Match the real client: any eager optional fetch must reach the failing GET below.
    get_trace_resource = DataApiClient.get_trace_resource

    def __init__(self, records=None, *, mutate=None):
        self.records = records if records is not None else [record()]
        self.mutate = mutate
        self.downloads = []

    def download_trace_export(self, run_id, dest, *, progress=None, rollouts=None):
        self.downloads.append((run_id, rollouts))
        records = self.records
        if rollouts is not None:
            selected = set(rollouts)
            records = [
                r
                for r in records
                if (r["metadata"]["task_id"], r["metadata"]["repeat"]) in selected
            ]
        path = write_trace_archive(dest, records, run_id=run_id, mutate=self.mutate)
        if progress:
            progress(path.stat().st_size, path.stat().st_size)
        return path

    def _get_json(self, *args, **kwargs):
        raise AssertionError("bulk import must not request individual rollout traces")


def compact(path, *, task="task/a", lane="", **fields):
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "problem_id": task,
                    "run_idx": 0,
                    "lane": lane,
                    "prompt_text": "prompt",
                    "response_text": "compact",
                    "reasoning_text": "cut",
                    "reward": 0.0,
                    "is_correct": False,
                    **fields,
                }
            ]
        ),
        path,
    )
    return path


# Artifact access.
BODY = b"First line\nIMPORTANT evidence\n"


def item(body=BODY, **metadata):
    artifact = {"artifact_idx": 2, "name": "evidence.txt", "mime": "text/plain",
                "size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), **metadata}
    return {"response_text": json.dumps({"steps": [], "execution": {
        "rollout": {"run_id": "run", "task_id": "task", "repeat": 0},
        "source": {"kind": "trace_api", "base_url": "https://api.example",
                   "trace_export_sha256": "a" * 64, "export_generation": "generation"},
        "artifacts": [{key: value for key, value in artifact.items() if value is not None}],
    }})}


class Provider:
    base_url = "https://api.example"

    def __init__(self, body=BODY):
        self.body, self.calls = body, 0

    def download_artifact(self, run_id, task_id, repeat, artifact_idx, dest):
        assert (run_id, task_id, repeat, artifact_idx) == ("run", "task", 0, 2)
        self.calls += 1
        dest.write_bytes(self.body)
        return dest


def serve(monkeypatch, home, body=BODY):
    """Route the default artifact store under ``home`` to a counting provider."""
    provider = Provider(body)
    monkeypatch.setenv("AUTOFA_HOME", str(home))
    monkeypatch.setattr(ArtifactStore, "_provider", lambda self: provider)
    return provider


# Artifact archive.
def zip_bytes(entries, *, compression=zipfile.ZIP_DEFLATED):
    output = io.BytesIO()
    with (warnings.catch_warnings(),
          zipfile.ZipFile(output, "w", compression=compression) as package):
        warnings.simplefilter("ignore", UserWarning)
        for name, data in entries:
            package.writestr(name, data)
    return output.getvalue()


# Trace execution.
def execution_item():
    event = {
        "event_idx": 41, "lane": "", "kind": "tool_call", "role": "assistant",
        "model_name": "example-model", "finish_reason": "tool_calls",
        "started_at": "2026-10-07T01:02:03Z", "latency_s": 1.7,
        "prompt_tokens": 812, "completion_tokens": 42,
        "provider_extension": {"cache": ["cold"]},
        "parts": [{"part_idx": 0, "part_type": "reasoning", "channel": "analysis",
                   "text": "Inspect the return code. " + "large evidence " * 10_000},
                  {"part_idx": 1, "part_type": "image", "artifact_idx": 3}],
        "event_text": {"request_messages": '[{"role":"user","content":"Task"}]',
                       "request_params": '{"temperature":0.2}'},
    }
    call = {"event_idx": 41, "call_idx": 0, "tool_name": "test", "lane": "",
            "args_full": '{"command":"pytest"}', "status": "error", "latency_s": 0.5,
            "result_event_idx": 43, "call_id_source": "call_1"}
    execution = {
        "rollout": {"status": "error", "started_at": "2026-10-07T01:02:00Z",
                    "provider_extension": {"worker": [2]}},
        "events": [event, {"event_idx": 42, "lane": "worker/2", "kind": "lifecycle",
                           "role": "harness", "parts": [], "status": "retrying"}],
        "tool_calls": [call],
        "attempts": [{"attempt_idx": 0, "failure_class": "timeout",
                      "is_terminal": False, "error_message": "retry"}],
        "artifacts": [{"artifact_idx": 3, "mime": "image/png", "size_bytes": 27},
                      {"artifact_idx": 4, "name": "log.txt", "text": ""}],
        "tool_definitions": [{"def_idx": 0, "name": "test",
                              "def_json": '{"name":"test","parameters":{"type":"object"}}'}],
        "provenance": [{"path": "source.jsonl", "first_line": 17, "reader": "example"}],
        "relations": [{"kind": "parent", "frm_key": "worker/2", "to_key": "main",
                       "resolved": True}],
        "availability": {"events": {"status": "available", "lanes": ["", "worker/2"]},
                         "artifacts": {"status": "partial", "reason": "Image text not available"},
                         "artifact_texts": {"status": "partial", "reason": "One text file"}},
        "raw_trajectory_metadata": {"schema": "source-v2", "runtime": {"worker": 2}},
        "prompt_summaries": [{"base_event_idx": 40, "kept": 9, "dropped": 2, "added": 1}],
        "resource_metadata": {"prompt_summaries": {"n_model_calls": 2, "cursor": None}},
        "future_collection": [{"raw": "keep this too"}],
    }
    return {"response_text": {"execution": execution, "steps": [
        {"step_id": 41, "source": "assistant", "reasoning_content": "Inspect the return code.",
         "tool_calls": [{"name": "test", "arguments": '{"command":"pytest"}'}],
         "metadata": {k: v for k, v in event.items() if k != "parts"}, "parts": event["parts"]},
        {"step_id": 43, "source": "tool", "observation": {"exit_code": 1},
         "metadata": {"event_idx": 43, "status": "error"}},
    ]}}


# Source model calls.
def prepare(tmp_path, monkeypatch, raw, *, steps=None):
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(raw) + "\n")
    monkeypatch.setattr("autofa.data.source_inputs._source_record", lambda *args, **kwargs: source)
    target = {
        "metadata": {"record_schema": "responses-api", "task_id": "0", "repeat": 0,
                     "raw_record_ref": "raw.jsonl:0"},
        "trace": {"agent": {}, "steps": steps or [],
                  "evidence_scope": {"limitations": {"source rollout contains no events": 1}}},
    }
    recover_initial_inputs([target], api=None, run_id="run", root=tmp_path)
    return target


# Grader model calls.
QUESTION = "Hey"
REPLY = "Hello! I'm your assistant. How can I help you today?"
BASELINE = "Heyyy! So good to see you!"
JUDGE_SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses provided "
    "by two AI assistants. Output your final verdict: Assistant A is better: [[A>>B]], "
    "Tie: [[A=B]], Assistant B is better: [[B>>A]]."
)


def _model_call(call_id, server, request, text, *, item):
    return {
        "model_call_id": call_id, "duration_ms": 5.0,
        "request": {"input": request, "temperature": 0.0},
        "response": {"output": [{"id": item, "type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": text},
        ]}], "model": f"{server}-v1", "status": "completed"},
        "response_metadata": {"model_ref": {"type": "responses_api_models", "name": server},
                              "model": f"{server}-v1", "status_code": 200},
        "token_stats": {"prompt_tokens": 16, "completion_tokens": 23},
    }


def _comparison(first, second, verdict, *, item):
    answers = (f"<|Conversation History|>\n{QUESTION}\n\n"
               f"<|The Start of Assistant A's Answer|>\n{first}\n"
               f"<|The End of Assistant A's Answer|>\n\n"
               f"<|The Start of Assistant B's Answer|>\n{second}\n"
               f"<|The End of Assistant B's Answer|>")
    return _model_call(verdict, "judge_model", [
        {"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": answers},
    ], f"B sounds natural.\nMy final verdict is {verdict}", item=item)


def _rollout(calls):
    """A NeMo Gym record: the policy's exchange, its captured calls and their inline account."""
    policy = [call for call in calls
              if call["response_metadata"]["model_ref"]["name"] == "policy_model"]
    return {
        "responses_create_params": {"input": [{"role": "user", "content": QUESTION}]},
        "response": {"output": [item for call in policy for item in call["response"]["output"]]},
        "reward": 0.0, "_ng_task_index": 0, "_ng_rollout_index": 0,
        "ng_trajectory": {"model_calls": calls},
        "ng_model_call_capture": {"calls": [
            {"model_call_id": call["model_call_id"], "call_index": index, "dialect": "responses",
             "model_ref": call["response_metadata"]["model_ref"],
             "model": call["response"]["model"]}
            for index, call in enumerate(calls)
        ]},
    }


def _policy():
    return _model_call("policy", "policy_model", [{"role": "user", "content": QUESTION}], REPLY,
                       item="msg-policy")


def arena_record(*, policy_call=True):
    """NeMo Gym arena rollout: one policy call, then two A/B-swapped judge calls."""
    judges = [_comparison(REPLY, BASELINE, "Assistant B is better: [[B>>A]]", item="msg-ab"),
              _comparison(BASELINE, REPLY, "Assistant A is better: [[A>>B]]", item="msg-ba")]
    return _rollout([_policy(), *judges] if policy_call else judges)


# Fake urllib responses.
class UrlResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict | None = None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            out, self._body = self._body, b""
            return out
        out, self._body = self._body[:n], self._body[n:]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
