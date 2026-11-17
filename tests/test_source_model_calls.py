# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw-record recovery keeps model calls and trajectories exact without labels or replay."""

import copy
import json
import zipfile
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import trace as trace_api
from autofa.core.sandbox import run_detector
from autofa.data.input_view import parsed_input
from autofa.data.parquet_store import Dataset
from autofa.data.remote_api import DataApiError
from autofa.data.source_export import materialize_source_traces, normalize_events
from autofa.data.source_inputs import _reconcile_request_capture_evidence, recover_initial_inputs
from autofa.data.source_model_calls import recover_calls, recover_model_calls
from autofa.data.source_trajectory import read_raw_trajectory, recover_raw_trajectory
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trajectory import parse_trajectory
from autofa.metrics.judge_evidence import make_snapshot

from support.traces import prepare, write_trace_archive


def record():
    first = [{"role": "system", "content": "System instruction"},
             {"role": "user", "content": "Fix it"}]
    action = {"role": "assistant", "content": "<think>Full reasoning tail</think>",
              "tool_calls": [{"id": "call-1", "type": "function", "function": {
                  "name": "terminal", "arguments": '{"command":"test", "reward":0}',
              }}]}
    result = {"role": "tool", "tool_call_id": "call-1", "content": "Observed test output"}
    return {
        "_ng_task_index": 0, "_ng_rollout_index": 0,
        "reward": 1, "grader": "SECRET_GRADER",
        "ng_trajectory": {"model_calls": [
            {"request": {"messages": first, "authorization": "SECRET_TOKEN", "tools": []},
             "response": {"choices": [{"message": action}],
                          "metrics": {"latency_ms": 12, "reward": "SECRET_METRICS"}}},
            {"request": {"messages": [*first, action, result], "tools": []},
             "response": {"choices": [{"message": {"role": "assistant", "content": "Done"}}]},
             "response_metadata": {"provider_region": "eu", "grader": "SECRET_METADATA"}},
        ]},
    }


CONTEXT_NOTICE = "model requests with changed context retained separately"
ENVELOPE_NOTICE = "provider response envelopes not recorded in event request captures"
TOOLS = "unsupported tool definitions retained in raw source"


class EmptyEventsApi:
    """Bulk export whose rollouts have no events, only references to raw source records."""

    base_url = "https://api.test"

    def __init__(self, *rollouts):
        self.rollouts = rollouts

    def _get_json(self, path, query=None):
        assert path == "/v1/runs/run", f"Unexpected per-rollout request: {path}"
        return {"run_id": "run"}

    def download_trace_export(self, run_id, dest, *, progress=None, rollouts=None):
        assert run_id == "run"
        assert rollouts == [(row["task_id"], row["repeat"]) for row in self.rollouts]
        self.archive = write_trace_archive(dest, [
            {"metadata": {"run_id": "run", "n_events": 0, "n_tool_calls": 0, **row},
             "events": [], "calls": []} for row in self.rollouts
        ])
        return self.archive


# --- raw model calls ---

def test_missing_events_recovered_and_repeated_requests_not_replayed(tmp_path, monkeypatch):
    recovered = prepare(tmp_path, monkeypatch, record())
    trace = recovered["trace"]
    assert recovered["raw_trajectory_recovered"]
    assert recovered["n_model_calls_recovered"] == 2
    assert [s["source"] for s in trace["steps"]] == [
        "system", "user", "assistant", "tool", "assistant",
    ]
    assert "source rollout contains no events" not in trace["evidence_scope"]["limitations"]
    calls = trace["model_call_evidence"]
    assert len(calls["messages"]) == 5
    assert calls["calls"][1]["request"] == [0, 1, 2, 3]
    assert trace["steps"][2]["tool_calls"][0]["call_id"] == trace["steps"][3]["call_id"]
    assert trace["steps"][2]["tool_calls"][0]["result_event_idx"] == 3
    blob = json.dumps(trace)
    assert "SECRET" not in blob
    assert "reward" in blob  # Actual model-visible tool argument, not a label.
    item = {"response_text": blob, "reasoning_text": ""}
    fields = derive_trace_fields(item)
    assert fields["reasoning_text"].endswith("Full reasoning tail")
    assert fields["final_message"] == "Done"
    assert "Observed test output" in fields["observations_text"]
    detector = 'def check(item):\n    return "Full reasoning tail" in item["reasoning_text"]'
    assert run_detector(detector, item)
    snapshot = make_snapshot(item, item_id="test", source=detector)
    assert "Full reasoning tail" in snapshot["fields"]["reasoning_text"]
    execution = trace_api.record(item)
    assert len(execution["recovered_steps"]) == 5
    assert execution["availability"]["recovered_steps"]["status"] == "partial"
    assert "recovered_steps" not in trace["execution"]  # reference, no duplicated stored history


def test_real_capture_shape_responses_without_later_requests(tmp_path, monkeypatch):
    raw = record()
    del raw["ng_trajectory"]["model_calls"][1]["request"]
    recovered = prepare(tmp_path, monkeypatch, raw)
    trace = recovered["trace"]
    assert len(trace["steps"]) == 4
    assert trace["steps"][-1]["message"] == "Done"
    limits = trace["evidence_scope"]["limitations"]
    assert limits["raw model request not recorded"] == 1
    assert limits["raw tool results not recorded"] == 1
    assert all("observation" not in step for step in trace["steps"])


def test_existing_events_preserved_with_later_request_snapshot(tmp_path, monkeypatch):
    raw = record()
    raw["ng_trajectory"]["model_calls"][1]["request"]["messages"][0] = {
        "role": "system", "content": "CHANGED_CONTEXT",
    }
    steps = [{"source": "assistant", "message": "Original event"}]
    recovered = prepare(tmp_path, monkeypatch, raw, steps=steps)
    assert recovered["trace"]["steps"] == steps
    assert "raw_trajectory_recovered" not in recovered
    evidence = recovered["trace"]["model_call_evidence"]
    idx = evidence["calls"][1]["request"][0]
    assert evidence["messages"][idx]["content"] == "CHANGED_CONTEXT"
    # Changed context is reported as information, never as missing evidence.
    scope = recovered["trace"]["evidence_scope"]
    assert scope["notices"][CONTEXT_NOTICE] == 1
    assert not any("changed request context" in reason for reason in scope["limitations"])


def test_responses_api_reasoning_function_output_and_media(tmp_path):
    raw = {"ng_trajectory": {"model_calls": [{
        "request": {"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Inspect"},
            {"type": "input_image", "image_url": "image-reference"},
        ]}]},
        "response": {"output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "r" * 80000}]},
            {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        ]},
    }, {
        "request": {"input": [{"type": "function_call_output", "call_id": "c1", "output": "Data"}]},
        "response": {"output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "Final"},
        ]}]},
    }]}}
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(raw) + "\n")
    result = recover_model_calls(path, 0)
    assert result["steps"][1]["reasoning_content"] == "r" * 80000
    assert result["steps"][-2]["observation"] == "Data"
    assert result["steps"][-1]["message"] == "Final"
    assert "image-reference" in json.dumps(result["messages"])
    assert result["limitations"]["raw non-text content retained but not inspected"] == 1


def test_initial_request_alone_is_not_claimed_as_recovered_trajectory(tmp_path, monkeypatch):
    raw = record()
    raw["ng_trajectory"]["model_calls"] = [{"request": {"messages": [
        {"role": "user", "content": "Input only"},
    ]}}]
    recovered = prepare(tmp_path, monkeypatch, raw)
    assert recovered["trace"]["steps"] == []
    assert "raw_trajectory_recovered" not in recovered
    assert recovered["trace"]["evidence_scope"]["limitations"]["raw model response not recorded"]


def test_ambiguous_response_choices_fail_visibly(tmp_path):
    raw = record()
    choices = raw["ng_trajectory"]["model_calls"][0]["response"]["choices"]
    choices.append({"message": {"role": "assistant", "content": "Alternative execution"}})
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(DataApiError, match="one execution sequence"):
        recover_model_calls(path, 0)


@pytest.mark.parametrize("bad_second", [False, True])
def test_source_preparation_writes_recovered_rows_and_manifest(tmp_path, monkeypatch, bad_second):
    source = tmp_path / "raw.jsonl"
    first = record()
    second = record()
    second["_ng_rollout_index"] = 2 if bad_second else 1
    line = (json.dumps(first) + "\n").encode()
    source.write_bytes(line + (json.dumps(second) + "\n").encode())
    downloads = []

    def source_record(*args, **kwargs):
        offset = args[3]
        downloads.append(offset)
        selected = source.read_bytes()[offset:].split(b"\n", 1)[0] + b"\n"
        dest = tmp_path / f"selected-{offset}.jsonl"
        dest.write_bytes(selected)
        return dest

    monkeypatch.setattr("autofa.data.source_inputs._source_record", source_record)

    api = EmptyEventsApi(*({
        "task_id": "0", "repeat": repeat, "record_schema": "responses-api",
        "raw_record_ref": f"raw.jsonl:{len(line) if repeat else 0}",
    } for repeat in range(2)))
    rows = [{"problem_id": "0", "run_idx": i, "response_text": "Summary",
             "reasoning_text": "CAPPED", "is_correct": bool(i), "reward": float(i)}
            for i in range(2)]
    export = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist(rows), export)
    original = export.read_bytes()
    if bad_second:
        with pytest.raises(DataApiError, match="repeat identity"):
            materialize_source_traces(export, api=api, run_id="run")
        assert not list(tmp_path.glob("source-v*/dataset.*"))
        assert export.read_bytes() == original
        return
    output = materialize_source_traces(export, api=api, run_id="run")
    loaded = Dataset(parquet_paths=[output]).load_items()
    assert len(loaded) == 2
    for i, row in enumerate(loaded):
        assert row["is_correct"] == rows[i]["is_correct"]
        assert "CAPPED" not in row["reasoning_text"]
        assert len(json.loads(row["response_text"])["steps"]) == 5
        assert "Full reasoning tail" in derive_trace_fields(row)["reasoning_text"]
    manifest = json.loads((output.parent / "manifest.json").read_text())
    assert manifest["n_model_calls_recovered"] == 4
    assert manifest["n_raw_trajectories_recovered"] == 2
    assert downloads == [0, len(line)]
    assert export.read_bytes() == original


# --- request context comparison ---

def action(arguments, *, dialect="tool_calls", call_id="call-1"):
    function = {"name": "terminal", "arguments": arguments}
    if dialect == "tool_calls":
        return {"role": "assistant", "content": "Run the check", "tool_calls": [
            {"id": call_id, "type": "function", "function": function},
        ]}
    if dialect == "function_call":
        return {"role": "assistant", "content": "Run the check",
                "function_call": {**function, "call_id": call_id}}
    return {"type": "function_call", "call_id": call_id, **function}


def calls(original, repeated, *, first_input=None, next_input=None, continuation=None):
    first_input = first_input or {"role": "user", "content": "Inspect the files"}
    next_input = next_input or first_input
    result = {"role": "tool", "tool_call_id": "call-1", "content": "checked"}
    return [
        {"request": {"messages": [first_input]}, "response": {"output": [original]}},
        {"request": {"messages": [next_input, repeated, result, *(continuation or [])]},
         "response": {"output": [{"role": "assistant", "content": "Done"}]}},
    ]


@pytest.mark.parametrize("dialect", ["tool_calls", "function_call", "responses"])
def test_argument_formatting_is_a_continuation_without_replaying_execution(dialect):
    original = action('{"command":"check","options":[1,true]}', dialect=dialect)
    repeated = action('{ "options": [1, true], "command": "check" }', dialect=dialect)
    # This later action exists only in the request tail. A formatting-only prefix
    # change must not misclassify the whole request and lose this new execution.
    new_action = action('{"command":"inspect"}', call_id="call-2")
    new_result = {"role": "tool", "tool_call_id": "call-2", "content": "inspected"}
    records = calls(original, repeated, continuation=[new_action, new_result])
    before = copy.deepcopy(records)
    recovered = recover_calls(records)
    assert recovered["notices"] == {}
    assert CONTEXT_NOTICE not in recovered["limitations"]
    assert records == before
    executed = [call for step in recovered["steps"] for call in step.get("tool_calls", [])]
    # Legacy function_call envelopes do not expose an outer call identity.
    assert [call.get("call_id") for call in executed] == [
        None if dialect == "function_call" else "call-1", "call-2",
    ]
    assert [call["arguments"]["command"] for call in executed] == ["check", "inspect"]
    tail = [step for step in recovered["steps"] if step.get("model_call_index") == 1]
    assert all(step["evidence_origin"] != "request_context" for step in tail)


def test_formatted_tool_arguments_keep_both_exact_raw_snapshots():
    raw_first = '{"command":"check","options":[1,true]}'
    raw_next = '{ "options": [1, true], "command": "check" }'
    recovered = recover_calls(calls(action(raw_first), action(raw_next)))
    response_id = recovered["calls"][0]["response"][0]
    request_id = recovered["calls"][1]["request"][1]
    assert response_id != request_id
    for index, raw in [(response_id, raw_first), (request_id, raw_next)]:
        stored = recovered["messages"][index]["tool_calls"][0]
        assert stored["arguments_raw"] == raw
        assert stored["function"]["arguments_raw"] == raw
        assert stored["arguments"] == json.loads(raw)
    # The execution still refers to the actual response, not its later restatement.
    execution = next(step for step in recovered["steps"] if step.get("tool_calls"))
    assert execution["message_index"] == response_id
    assert execution["tool_calls"][0]["arguments_raw"] == raw_first


def _raw_arguments(message, raw):
    message = copy.deepcopy(message)
    message["tool_calls"][0]["function"]["arguments_raw"] = raw
    return message


@pytest.mark.parametrize("original,repeated", [
    (action('{"command":"check"}'), action('{"command":"delete"}')),
    # Strings inside arguments and raw-named provider metadata are payloads, not formatting.
    (action(json.dumps({"script": '{"a":1}', "arguments_raw": '{"x":1}'})),
     action(json.dumps({"script": '{ "a": 1 }', "arguments_raw": '{"x":1}'}))),
    # Ambiguous or inconsistent arguments are never normalized equal.
    (action('{"command":"check","command":"check"}'), action('{"command":"check"}')),
    (action('{"command":NaN}'), action('{ "command": NaN }')),
    (action({"command": 1}), _raw_arguments(action({"command": 1}), '{"command":true}')),
], ids=["arguments", "payload_string", "duplicate_keys", "nan", "raw_true_is_not_one"])
def test_significant_changes_are_notices_without_replaying_old_actions(original, repeated):
    recovered = recover_calls(calls(original, repeated))
    assert recovered["notices"] == {CONTEXT_NOTICE: 1}
    assert "changed request context retained separately" not in recovered["limitations"]
    assert [step["source"] for step in recovered["steps"]] == [
        "user", "assistant", "tool", "assistant",
    ]
    first = recovered["messages"][recovered["calls"][0]["response"][0]]
    second = recovered["messages"][recovered["calls"][1]["request"][1]]
    assert first != second
    assert recovered["steps"][1]["tool_calls"][0]["name"] == "terminal"


def test_identical_response_messages_are_still_distinct_recorded_executions():
    repeated = {"role": "assistant", "content": "Checking"}
    records = [
        {"request": {"messages": [{"role": "user", "content": "Inspect"}]},
         "response": {"output": [repeated]}},
        {"request": {"messages": [{"role": "user", "content": "Inspect"}, repeated]},
         "response": {"output": [repeated]}},
    ]
    recovered = recover_calls(records)
    assert recovered["notices"] == {}
    assert len([step for step in recovered["steps"] if step["source"] == "assistant"]) == 2


def test_unsupported_raw_tool_definitions_remain_real_limitations():
    records = calls(action("{}"), action("{}"))
    records[0]["request"]["tools"] = {"name": "terminal"}
    recovered = recover_calls(records)
    assert recovered["limitations"][TOOLS] == 1
    assert recovered["notices"] == {}


# --- reported evidence gaps ---

def recover(tmp_path, monkeypatch, raw, *, missing_capture_messages=False):
    ident = {"run_id": "run", "task_id": "0", "repeat": 0}
    metadata = {**ident, "record_schema": "responses-api", "raw_record_ref": "raw.jsonl:0",
                "n_events": 2, "n_tool_calls": 0}
    events = [{**ident, "event_idx": i, "kind": "message", "role": "assistant",
               "n_parts": 0, "n_tool_calls": 0, "parts": [], "event_text": {
                   "request_messages": (None if missing_capture_messages else
                                        json.dumps([{"role": "user", "content": "Fix it"}])),
                   "request_tools": {"unsupported": "container"},
               }} for i in range(2)]
    trace = normalize_events(metadata, events, [])
    assert trace["evidence_scope"]["limitations"][TOOLS] == 2
    assert trace["evidence_scope"]["notices"][ENVELOPE_NOTICE] == 2
    trace["evidence_scope"]["limitations"]["encrypted reasoning unavailable"] = 3
    steps = copy.deepcopy(trace["steps"])
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(raw) + "\n")
    monkeypatch.setattr("autofa.data.source_inputs._source_record", lambda *a, **kw: source)
    target = {"metadata": metadata, "trace": trace}
    recover_initial_inputs([target], api=None, run_id="run", root=tmp_path)
    assert trace["steps"] == steps
    assert trace["evidence_scope"]["limitations"]["encrypted reasoning unavailable"] == 3
    return trace


def test_full_raw_recovery_removes_only_superseded_request_capture_gaps(tmp_path, monkeypatch):
    trace = recover(tmp_path, monkeypatch, record())
    scope = trace["evidence_scope"]
    assert TOOLS not in scope["limitations"]
    assert ENVELOPE_NOTICE not in scope.get("notices", {})
    assert scope["superseded_request_capture_limitations"] == {TOOLS: 2}
    assert all(call["response_recorded"] for call in trace["model_call_evidence"]["calls"])


def test_recovering_one_call_cannot_clear_two_capture_gaps(tmp_path, monkeypatch):
    raw = record()
    raw["ng_trajectory"]["model_calls"] = raw["ng_trajectory"]["model_calls"][:1]
    scope = recover(tmp_path, monkeypatch, raw)["evidence_scope"]
    assert scope["limitations"][TOOLS] == 2
    assert scope["notices"][ENVELOPE_NOTICE] == 2


@pytest.mark.parametrize("capture_lanes,raw_lane", [
    (["main", "worker"], None),  # Raw calls without lanes cannot certify several lanes.
    (["main", "main"], "worker"),
])
def test_call_count_cannot_establish_recovery_across_lanes(capture_lanes, raw_lane):
    raw = record()["ng_trajectory"]["model_calls"]
    if raw_lane is not None:
        for call in raw:
            call["lane"] = raw_lane
    trace = {
        "model_call_evidence": {"calls": [{"source_lane": lane} for lane in capture_lanes]},
        "evidence_scope": {
            "limitations": {TOOLS: 2}, "request_capture_limitations": {TOOLS: 2},
            "notices": {ENVELOPE_NOTICE: 2},
        },
    }
    original = copy.deepcopy(trace["evidence_scope"])
    _reconcile_request_capture_evidence(trace, recover_calls(raw))
    assert trace["evidence_scope"] == original


def test_missing_raw_response_remains_a_limitation(tmp_path, monkeypatch):
    raw = record()
    del raw["ng_trajectory"]["model_calls"][1]["response"]
    scope = recover(tmp_path, monkeypatch, raw)["evidence_scope"]
    assert scope["limitations"]["raw model response not recorded"] == 1
    assert scope["notices"][ENVELOPE_NOTICE] == 2


def test_empty_raw_request_cannot_clear_missing_message_gap(tmp_path, monkeypatch):
    raw = record()
    raw["ng_trajectory"]["model_calls"][1]["request"]["messages"] = []
    scope = recover(tmp_path, monkeypatch, raw, missing_capture_messages=True)["evidence_scope"]
    assert scope["limitations"]["event model request messages not recorded"] == 2


@pytest.mark.parametrize("tools,remaining", [(None, 2), ({"bad": "schema"}, 3)])
def test_null_or_invalid_raw_tools_cannot_clear_tool_capture_warning(
    tmp_path, monkeypatch, tools, remaining,
):
    raw = record()
    raw["ng_trajectory"]["model_calls"][1]["request"]["tools"] = tools
    trace = recover(tmp_path, monkeypatch, raw)
    assert trace["evidence_scope"]["limitations"][TOOLS] == remaining
    assert "tools" not in trace["model_call_evidence"]["calls"][1]


# --- raw trajectories ---

def raw_record():
    return {
        "problem_idx": "task", "repeat": 2, "reward": 1,
        "expected_answer": "SECRET_ANSWER", "scoring_details": {"solution": "SECRET_GRADE"},
        "trajectory": [{
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "terminal-agent", "extra": {"reward": "SECRET_AGENT"}},
            "final_metrics": {"resolved": True}, "extra": {"status": "SECRET_STATUS"},
            "steps": [
                {"step_id": 1, "source": "system", "message": "Instructions"},
                {"step_id": 2, "source": "user", "message": "Fix λ task"},
                {"step_id": 3, "source": "agent", "message": "Working",
                 "reasoning_content": "Inspect the task",
                 "extra": {"scorer": "SECRET_STEP"}, "metrics": {"reward": 1},
                 "tool_calls": [{"tool_call_id": "call-1", "function_name": "run",
                                 "arguments": '{"command":"echo hello", "literal":"{x}"}',
                                 "score": "SECRET_CALL"}],
                 "observation": {"results": [
                     {"source_call_id": "call-1", "content": "hello",
                      "extra": {"reward": "SECRET_OBSERVATION_RESULT"}},
                 ], "extra": {"grade": "SECRET_OBSERVATION"}}},
            ],
        }],
    }


def write_record(tmp_path, raw):
    source = tmp_path / "source.jsonl"
    prefix = json.dumps({"ignored": "λ prefix"}, ensure_ascii=False).encode() + b"\n"
    source.write_bytes(prefix + json.dumps(raw, ensure_ascii=False).encode() + b"\n")
    return source, len(prefix)


@pytest.mark.parametrize("shape", ["wrapped", "native", "nested_encoded", "steps", "root"])
def test_shape_based_recovery_preserves_conversation_and_excludes_grading(tmp_path, shape):
    raw = raw_record()
    envelope = raw["trajectory"][0]
    if shape == "root":  # Root steps with the record's explicit identity.
        raw.update(raw.pop("trajectory")[0])
    elif shape == "native":
        raw["trajectory"] = envelope
    elif shape == "nested_encoded":
        envelope["steps"] = json.dumps(envelope["steps"])
        raw["trajectory"] = json.dumps(json.dumps([json.dumps(envelope)]))
    elif shape == "steps":
        raw["trajectory"] = envelope["steps"]
    source, offset = write_record(tmp_path, raw)
    recovered = read_raw_trajectory(source, offset, task="task", repeat=2)
    assert recovered is not None
    steps = recovered["steps"]
    assert [s["step_id"] for s in steps] == [1, 2, 3]
    assert [s["source"] for s in steps] == ["system", "user", "agent"]
    assert steps[1]["message"] == "Fix λ task"
    assert steps[2]["reasoning_content"] == "Inspect the task"
    call = steps[2]["tool_calls"][0]
    assert call["arguments"] == {"command": "echo hello", "literal": "{x}"}
    assert call["call_id"] == "call-1"
    assert steps[2]["observation"]["results"][0]["content"] == "hello"
    assert "SECRET" not in json.dumps(recovered)


@pytest.mark.parametrize("mutation", [
    {"problem_idx": "different"}, {"task_id": "contradiction"},
    "repeat",  # Missing identity fails closed.
])
def test_raw_identity_must_be_present_and_match_without_conflicting_aliases(tmp_path, mutation):
    raw = raw_record()
    if isinstance(mutation, str):
        del raw[mutation]
    else:
        raw.update(mutation)
    source, offset = write_record(tmp_path, raw)
    with pytest.raises(DataApiError, match="identity"):
        read_raw_trajectory(source, offset, task="task", repeat=2)


def test_corrupt_tail_and_mid_record_offset_are_rejected(tmp_path):
    source, offset = write_record(tmp_path, raw_record())
    with pytest.raises(DataApiError, match="boundary"):
        read_raw_trajectory(source, offset + 1, task="task", repeat=2)
    source.write_bytes(source.read_bytes().rstrip() + b" trailing\n")
    with pytest.raises(DataApiError, match="complete JSON"):
        read_raw_trajectory(source, offset, task="task", repeat=2)


def test_multiple_trajectories_are_not_partially_recovered(tmp_path):
    raw = raw_record()
    raw["trajectory"].append(copy.deepcopy(raw["trajectory"][0]))
    source, offset = write_record(tmp_path, raw)
    with pytest.raises(DataApiError, match="parsed completely"):
        read_raw_trajectory(source, offset, task="task", repeat=2)


# A grader actor is excluded even when its step has a conversation source.
@pytest.mark.parametrize("field,value", [("source", "grader"), ("actor", "judge")])
def test_grader_steps_are_not_imported_as_model_content(tmp_path, field, value):
    raw = raw_record()
    raw["trajectory"][0]["steps"][2][field] = value
    source, offset = write_record(tmp_path, raw)
    with pytest.raises(DataApiError, match="role/actor"):
        read_raw_trajectory(source, offset, task="task", repeat=2)


def test_recovery_retains_scope_tools_and_does_not_fabricate_model_requests(tmp_path):
    source, offset = write_record(tmp_path, raw_record())
    record = {
        "metadata": {"record_schema": "unrelated-label", "task_id": "task", "repeat": 2,
                     "raw_record_ref": f"record.jsonl:{offset}"},
        "trace": {"steps": [], "agent": {"tool_definitions": [{"name": "run"}]},
                  "evidence_scope": {"source": "API", "limitations": {
                      "source rollout contains no events": 1,
                  }}},
    }
    assert recover_raw_trajectory(
        record, api=None, run_id="run", root=tmp_path, source_files={"record.jsonl": source},
    )
    trace = record["trace"]
    assert len(trace["steps"]) == 3 and record["raw_trajectory_recovered"]
    assert trace["agent"]["tool_definitions"] == [{"name": "run"}]
    assert trace["evidence_scope"]["source"] == "API"
    assert trace["evidence_scope"]["trajectory_source"] == "raw record trajectory"
    assert "source rollout contains no events" not in trace["evidence_scope"]["limitations"]
    assert "initial_request" not in trace and "model_call_evidence" not in trace
    assert "SECRET" not in json.dumps(record)


def test_existing_trace_is_untouched(tmp_path):
    record = {"metadata": {"raw_record_ref": "source.jsonl:0"}, "trace": {
        "steps": [{"source": "user", "message": "Existing"}],
        "evidence_scope": {"limitations": {}},
    }}
    before = copy.deepcopy(record)
    assert not recover_raw_trajectory(record, api=None, run_id="run", root=tmp_path)
    assert record == before


def test_no_raw_conversation_is_explicit(tmp_path):
    source, offset = write_record(tmp_path, {"problem_idx": "task", "repeat": 2})
    record = {"metadata": {"task_id": "task", "repeat": 2,
                           "raw_record_ref": f"source.jsonl:{offset}"},
              "trace": {"steps": [], "evidence_scope": {"limitations": {}}}}
    assert not recover_raw_trajectory(
        record, api=None, run_id="run", root=tmp_path, source_files={"source.jsonl": source},
    )
    assert record["trace"]["steps"] == []
    assert record["trace"]["evidence_scope"]["limitations"] == {
        "raw source contains no recognized conversation trajectory": 1,
    }


def test_source_materialization_recovers_raw_trace_when_api_events_are_empty(
    tmp_path, monkeypatch,
):
    raw = raw_record()
    raw["trajectory"][0]["steps"][1]["source"] = "developer"
    source, offset = write_record(tmp_path, raw)
    selected = tmp_path / "selected.jsonl"
    selected.write_bytes(source.read_bytes()[offset:])

    def fetch(api, run_id, name, raw_offset, root, **kwargs):
        assert (run_id, name, raw_offset) == ("run", "source.jsonl", offset)
        return selected

    monkeypatch.setattr("autofa.data.source_inputs._source_record", fetch)

    export = tmp_path / "export.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "problem_id": "task", "run_idx": 2, "response_text": "", "reasoning_text": "",
        "prompt_text": "", "reward": 1.0,
    }]), export)
    api = EmptyEventsApi({
        "task_id": "task", "repeat": 2, "record_schema": "otlp-openinference",
        "agent_ref": "main", "n_artifacts": 0, "raw_record_ref": f"source.jsonl:{offset}",
    })
    prepared = materialize_source_traces(export, api=api, run_id="run", catalog={"run_id": "run"})
    row = pq.read_table(prepared).to_pylist()[0]
    parsed = parse_trajectory(row["response_text"])
    assert parsed.ok and parsed.n_steps == 3 and parsed.n_tool_calls == 1
    assert row["reward"] == 1.0
    assert "SECRET" not in row["response_text"]
    assert parsed_input(row)["messages"] == [
        {"role": "system", "content": "Instructions"},
        {"role": "developer", "content": "Fix λ task"},
    ]
    manifest = json.loads((prepared.parent / "manifest.json").read_text())
    assert manifest["n_raw_trajectories_recovered"] == 1
    assert "task/user input unavailable" not in manifest["limitations"]
    with zipfile.ZipFile(api.archive) as archive:
        assert pq.read_table(BytesIO(archive.read("events.parquet"))).num_rows == 0
        original_rollout = pq.read_table(BytesIO(archive.read("rollouts.parquet"))).to_pylist()[0]
        assert original_rollout["raw_record_ref"] == f"source.jsonl:{offset}"
