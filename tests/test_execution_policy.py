# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution metadata survives normalization while external targets stay separate."""

import json
from copy import deepcopy

import pytest

from autofa import trace
from autofa.data.execution_policy import (
    is_evaluation_record,
    is_evaluation_table,
    sanitize_metadata,
)
from autofa.data.source_model_calls import recover_calls, recover_event_requests
from autofa.data.source_snapshots import recover_request_snapshots
from autofa.data.source_trajectory import read_raw_trajectory, recover_raw_trajectory
from autofa.data.structured import message_steps, messages


def test_status_and_usage_are_execution_evidence_not_external_outcomes():
    raw = {
        "status": "failed", "error": {"code": "timeout"},
        "metrics": {"output_tokens": 100, "reward": 0, "is_correct": False},
        "metadata": {"provider_extension": {"gpu": "h100", "score": 0.1}},
        "headers": {"request-id": "request-1", "authorization": "SECRET"},
        "api-key": "SECRET", "credentials": {"access_token": "SECRET"},
        "metadata_json": json.dumps({"latency_ms": 20, "grader": "SECRET"}),
    }
    original = deepcopy(raw)
    cleaned = sanitize_metadata(raw)
    assert cleaned == {
        "status": "failed", "error": {"code": "timeout"}, "metrics": {"output_tokens": 100},
        "metadata": {"provider_extension": {"gpu": "h100"}},
        "headers": {"request-id": "request-1"}, "metadata_json": cleaned["metadata_json"],
    }
    assert json.loads(cleaned["metadata_json"]) == {"latency_ms": 20}
    cleaned["metadata"]["provider_extension"]["gpu"] = "changed"
    assert raw == original


def test_rollout_targets_are_scoped_separately_from_execution_records():
    raw = {
        "status": "failed", "success": False,
        "metrics": {"resolved": False, "duration_ms": 20, "reward": 0},
        "events": [{"status": "failed", "duration_ms": 1}],
        "attempts": [{"status": "failed", "error": "connection reset"}],
        "other": [{"actor": "grader", "text": "SECRET"}, {"actor": "agent", "text": "ok"}],
        "extension": {"lane_kind": "verdict", "text": "SECRET"},
    }
    cleaned = sanitize_metadata(raw, rollout=True)
    assert "status" not in cleaned and "success" not in cleaned
    assert cleaned["metrics"] == {"duration_ms": 20}
    assert cleaned["events"][0]["status"] == "failed"
    assert cleaned["attempts"][0]["status"] == "failed"
    assert cleaned["other"] == [{"actor": "agent", "text": "ok"}]
    assert "extension" not in cleaned
    # Known correctness aliases do not hide tool or relation status on execution records.
    metadata = {"is_resolved": True, "tb_is_resolved": False,
                "resolved": True, "status": "failed"}
    assert sanitize_metadata(metadata) == {"resolved": True, "status": "failed"}
    assert sanitize_metadata(metadata, rollout=True) == {}


@pytest.mark.parametrize("record, expected", [
    ({"actor": "grader"}, True), ({"lane": "evaluation/results"}, True),
    ({"role": "assistant", "text": "ask a judge"}, False),
    ({"source": "/code/evaluation/protocol.py", "status": "failed"}, False),
    # NeMo Gym model calls name the model server they were made through.
    ({"response_metadata": {"model_ref": {"type": "responses_api_models",
                                          "name": "judge_model"}}}, True),
    ({"response_metadata": json.dumps({"model_ref": {"name": "Judge-Model"}})}, True),
    ({"response_metadata": {"model_ref": {"name": "policy_model"}, "model": "judge-llm"}},
     False),
    ({"model_ref": {"name": "judge_ensemble_policy"}}, False),
])
def test_only_explicit_evaluation_provenance_is_excluded(record, expected):
    assert is_evaluation_record(record) is expected


@pytest.mark.parametrize("name, expected", [
    ("scores", True), ("grader_results", True), ("tool_results", False),
])
def test_only_explicit_target_tables_are_excluded(name, expected):
    assert is_evaluation_table(name) is expected


def test_tool_payloads_and_model_content_are_opaque_evidence():
    payload = {"reward": 7, "status": "failed", "api_key": "literal task parameter"}
    raw = {"tool_calls": [{"name": "inspect", "status": "failed", "arguments": payload}],
           "content": payload, "output": payload, "observation": payload,
           "parameters": {"properties": {"reward": {"type": "number"}}}}
    assert sanitize_metadata(raw) == raw
    # A trajectory step's message stays opaque even when it looks like a provider envelope.
    step = {"source": "tool", "message": {"role": "grader", "reward": 7, "status": "failed"}}
    assert sanitize_metadata(step) == step
    wrapper = {"results": [{"source_call_id": "c", "content": payload,
                            "status": "failed", "reward": "EXTERNAL"}],
               "metrics": {"duration_ms": 1, "score": "EXTERNAL"}}
    assert sanitize_metadata({"observation": wrapper}) == {"observation": {
        "results": [{"source_call_id": "c", "content": payload, "status": "failed"}],
        "metrics": {"duration_ms": 1},
    }}


def test_encoded_metadata_and_provider_message_envelopes_have_scoped_filtering():
    message = {"role": "assistant", "content": {"reward": "visible"}, "reward": "SECRET",
               "status": "incomplete", "authorization": "SECRET"}
    raw = {"request_params": json.dumps({"messages": [message], "temperature": 0.2}),
           "response": {"choices": [{"message": message}]},
           "metadata": json.dumps({"usage": {"tokens": 2}, "grader": "SECRET"}),
           "function": json.dumps({"name": "tool", "arguments": {"reward": "visible"},
                                   "api_key": "SECRET"})}
    cleaned = sanitize_metadata(raw)
    assert "SECRET" not in json.dumps(cleaned)
    assert json.loads(cleaned["request_params"])["messages"][0]["content"] == {"reward": "visible"}
    assert cleaned["response"]["choices"][0]["message"]["status"] == "incomplete"
    assert json.loads(cleaned["function"])["arguments"] == {"reward": "visible"}
    # An encoded-metadata field that holds plain text is kept verbatim, not rejected.
    assert sanitize_metadata({"request": "{not json"}) == {"request": "{not json"}


def test_transport_uri_credentials_are_removed_without_scrubbing_evidence_urls():
    signed = "https://user:SECRET@storage.test/log?part=1&X-Amz-Signature=SECRET&token=SECRET"
    assert sanitize_metadata({"uri": signed}) == {"uri": "https://storage.test/log?part=1"}
    assert sanitize_metadata({"content": signed, "arguments": {"url": signed}}) == {
        "content": signed, "arguments": {"url": signed},
    }


def test_message_extensions_survive_without_surrounding_grader_data():
    raw = {"messages": [{"role": "assistant", "content": {"reward": "visible"},
                         "status": "incomplete", "provider_extension": {"cache_hit": True},
                         "reward": "SECRET"}],
           "grader": "SECRET", "api_key": "SECRET"}
    parsed, warnings = messages(raw)
    assert not warnings
    assert parsed[0]["content"] == {"reward": "visible"}
    assert parsed[0]["status"] == "incomplete"
    assert "SECRET" not in json.dumps(parsed)
    steps, _ = message_steps(parsed[0])
    assert steps[0]["metadata"]["provider_extension"] == {"cache_hit": True}


def test_model_calls_preserve_parameters_and_source_metadata_without_targets():
    request = [{"role": "user", "content": "Task"}]
    output = {"role": "assistant", "content": "Done"}
    calls = [{
        "request": {"messages": request, "model": "model-v1", "temperature": 0.2,
                    "authorization": "SECRET"},
        "response": {"choices": [{"message": output, "finish_reason": "length"}],
                     "status": "incomplete", "metrics": {"reward": "SECRET"}},
        "status": "failed", "error": {"code": "timeout"}, "source_event_index": 10,
        "grader": "SECRET",
    }, {
        "request": {"messages": [*request, output], "model": "model-v1"},
        "response": {"output": [{"role": "assistant", "content": "Finished"}]},
    }]
    original = deepcopy(calls)
    recovered = recover_calls(calls)
    first = recovered["calls"][0]
    assert first["request_metadata"] == {"model": "model-v1", "temperature": 0.2}
    assert "message" not in first["response_metadata"]["choices"][0]
    assert first["metadata"]["error"] == {"code": "timeout"}
    assert first["metadata"]["source_event_index"] == 10
    assert "SECRET" not in json.dumps(recovered)
    assert [step.get("message") for step in recovered["steps"]] == ["Task", "Done", "Finished"]
    assert calls == original


def test_request_snapshot_preserves_request_and_event_parameters():
    body = {"message_ids": ["m"], "messages_first_seen_here": {
        "m": {"role": "user", "content": "Task", "provider_extension": 1},
    }, "temperature": 0.7, "reward": "SECRET", "api_key": "SECRET"}
    text = json.dumps(body)
    event = {"event_idx": 2, "kind": "other", "role": "harness", "actor": "agent",
             "n_parts": 1, "status": "completed",
             "parts": [{"part_type": "text", "text": text, "chars": len(text)}]}
    recovered = recover_request_snapshots([event])
    call = recovered["calls"][0]
    assert call["request_metadata"] == {"temperature": 0.7}
    assert call["metadata"]["status"] == "completed"
    assert recovered["messages"][0]["provider_extension"] == 1
    assert "SECRET" not in json.dumps(call)


def test_event_text_requests_preserve_lane_identity_without_replaying_history():
    initial = [{"role": "user", "content": "Task"}]
    later = [*initial, {"role": "assistant", "content": "First answer"}]

    def captured(index, lane, content):
        return {"event_idx": index, "lane": lane, "role": "assistant", "kind": "message",
                "event_text": {"request_dialect": "openai",
                               "request_messages": json.dumps(content),
                               "request_tools": json.dumps([{"name": "tool", "parameters": {
                                   "properties": {"reward": {"type": "number"}},
                               }}]),
                               "request_params": json.dumps({"temperature": 0.2,
                                                             "api_key": "SECRET"})}}

    recovered = recover_event_requests([
        captured(2, "", later), captured(0, "", initial), captured(0, "worker", initial),
        captured(1, "grader", initial),
    ])
    assert "steps" not in recovered
    assert len(recovered["messages"]) == 2
    calls = recovered["calls"]
    assert [(call["source_lane"], call["source_event_index"]) for call in calls] == [
        ("", 0), ("", 2), ("worker", 0),
    ]
    assert [call["initial_request_recorded"] for call in calls] == [True, False, True]
    assert all(call["response_recorded"] is False and call["response"] == [] for call in calls)
    assert calls[0]["request_metadata"] == {"temperature": 0.2}
    assert "reward" in recovered["tools"][0][0]["parameters"]["properties"]
    assert "SECRET" not in json.dumps(recovered)


def test_event_request_is_not_initial_when_earlier_output_or_context_exists():
    output = {"event_idx": 1, "lane": "", "role": "assistant"}
    capture = {"event_idx": 2, "lane": "", "role": "assistant", "event_text": {
        "request_messages": '[{"role":"user","content":"After compaction"}]',
    }}
    first = recover_event_requests([output, capture])["calls"][0]
    assert first["initial_request_recorded"] is False
    capture["event_text"]["request_messages"] = '[{"role":"assistant","content":"Earlier"}]'
    assert recover_event_requests([capture])["calls"][0]["initial_request_recorded"] is False


def test_raw_fallback_keeps_execution_extensions_and_envelope_metadata(tmp_path):
    raw = {"problem_idx": "task", "repeat": 0, "reward": "SECRET", "trajectory": {
        "schema_version": "ATIF-v1.7", "status": "failed", "duration_ms": 123,
        "is_resolved": False, "provider_extension": {"region": "eu", "score": "SECRET"},
        "agent": {"name": "agent"},
        "steps": [{"source": "agent", "message": "Run tests", "status": "completed",
                   "usage": {"tokens": 10, "reward": "SECRET"},
                   "tool_calls": [{"name": "test", "arguments": {"reward": "visible"},
                                   "status": "failed", "score": "SECRET"}],
                   "observation": {"results": [{"source_call_id": "c", "content": "failed",
                                                "exit_code": 1, "reward": "SECRET"}]}}],
    }}
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(raw) + "\n")
    recovered = read_raw_trajectory(path, 0, task="task", repeat=0)
    step = recovered["steps"][0]
    assert step["usage"] == {"tokens": 10}
    assert step["tool_calls"][0]["arguments"] == {"reward": "visible"}
    assert step["observation"]["results"][0]["exit_code"] == 1
    assert recovered["metadata"]["duration_ms"] == 123
    assert "status" not in recovered["metadata"] and "is_resolved" not in recovered["metadata"]
    assert "SECRET" not in json.dumps(recovered)
    record = {"metadata": {"raw_record_ref": "raw.jsonl:0", "task_id": "task", "repeat": 0},
              "trace": {"steps": [], "execution": {"events": ["existing"]},
                        "evidence_scope": {"limitations": {}}}}
    assert recover_raw_trajectory(record, api=None, run_id="run", root=tmp_path,
                                  source_files={"raw.jsonl": path})
    execution = record["trace"]["execution"]
    assert execution["events"] == ["existing"]
    assert execution["raw_trajectory_metadata"]["duration_ms"] == 123
    assert execution["recovered_steps_source"] == "steps"
    assert "recovered_steps" not in execution  # no duplicate persisted bodies
    item = {"response_text": record["trace"]}
    assert trace.steps(item)[0].metadata["status"] == "completed"
    assert trace.record(item)["recovered_steps"][0]["usage"] == {"tokens": 10}
    assert trace.coverage(item)["recovered_steps"]["status"] == "partial"
