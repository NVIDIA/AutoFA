# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same evidence, system text included, must reach import, detector, judge and MCP consumers."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import trace
from autofa.core.sandbox import run_detector
from autofa.data.gym_adapter import materialize_gym
from autofa.data.input_view import parsed_input
from autofa.data.parquet_store import Dataset, search_items
from autofa.data.remote_api import DataApiError
from autofa.data.source_export import materialize_source_traces, normalize_events
from autofa.data.source_inputs import _read_input_record, initial_request
from autofa.data.source_model_calls import recover_model_calls
from autofa.data.source_snapshots import recover_request_snapshots
from autofa.data.structured import decode_container, input_view
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trajectory import parse_trajectory
from autofa.mcp import tools_data
from autofa.mcp.server import TOOL_CALLABLES
from autofa.metrics.behavior import item_trace_for_judge
from autofa.metrics.judge_evidence import make_snapshot

from support.traces import write_trace_archive


def encoded(value, depth):
    for _ in range(depth):
        value = json.dumps(value, ensure_ascii=False)
    return value


def raw_file(tmp_path, raw):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(raw))
    return path


def write_rows(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.mark.parametrize("depth", [0, 4])
def test_structured_columns_and_nested_arguments_round_trip(tmp_path, depth):
    name = "arbitrary.namespace/tool"
    arguments = {"code": '{"literal": "JSON source"}', "flags": [1, None, 7], "enabled": True,
                 "nested": {"λ": "value"}}
    steps = [
        {"source": "user", "message": "Task"},
        {"source": "assistant", "tool_calls": [
            {"id": "call", "function": {"name": name, "arguments": encoded(arguments, depth)}},
        ]},
        {"source": "tool", "observation": encoded({"status": "ok"}, depth)},
    ]
    messages = [{"role": "developer", "content": "Rules"}, {"role": "user", "content": "Task"}]
    path = write_rows(tmp_path / "input.parquet", [{
        "problem_id": "x", "run_idx": 0, "prompt_text": encoded(messages, depth),
        "response_text": encoded({"steps": steps}, depth),
    }])
    loaded = Dataset(parquet_paths=[path]).load_items()[0]
    assert loaded["response_text"] == pq.read_table(path).to_pylist()[0]["response_text"]
    assert [m["role"] for m in trace.input_messages(loaded)] == ["developer", "user"]
    call = trace.tool_calls(loaded)[0]
    assert call["name"] == name and call["call_id"] == "call"
    assert call["arguments"] == arguments
    assert trace.steps(loaded)[2].observation_data == {"status": "ok"}
    assert run_detector(
        """
from autofa.trace import tool_calls
def check(item):
    return (item['input_messages'][0]['role'] == 'developer'
            and tool_calls(item)[0]['arguments']['flags'][2] == 7)
""",
        loaded,
    )
    snapshot = make_snapshot(loaded, item_id="test", source="def check(item): return True")
    assert json.loads(snapshot["fields"]["input_messages"])[0]["role"] == "developer"


@pytest.mark.parametrize("shape", ["string", "messages", "encoded_messages", "instructions"])
def test_initial_request_formats_agree_with_model_call_capture(tmp_path, shape):
    prompt = "λ task"
    value = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    request = {"input": value, "tools": [{"type": "function", "name": "any_tool"}]}
    if shape == "string":
        request["input"] = prompt
    elif shape == "encoded_messages":
        request["input"] = encoded(value, 3)
    elif shape == "messages":
        request["messages"] = request.pop("input")
    elif shape == "instructions":
        request["instructions"] = "System rules"
    path = raw_file(tmp_path, {
        "_ng_task_index": 0, "_ng_rollout_index": 0,
        "ng_trajectory": {"model_calls": [{
            "request": request, "response": {"output": [{"role": "assistant", "content": "Done"}]},
        }]},
    })
    initial = initial_request(_read_input_record(path, 0), task="0", repeat=0)
    assert prompt in initial["prompt"]
    assert initial["limitations"] == []
    calls = recover_model_calls(path, 0)
    item = {"response_text": json.dumps({"steps": calls["steps"], "model_call_evidence": calls})}
    assert trace.input_messages(item) == initial["messages"]
    assert trace.input_tools(item) == initial["tools"]
    assert trace.input_text(item) == initial["prompt"]
    assert trace.model_calls(item)[0]["request"] == initial["messages"]


@pytest.mark.parametrize("field", ["request", "model_calls", "ng_trajectory"])
def test_encoded_capture_boundaries(tmp_path, field):
    request = {"messages": [{"role": "user", "content": "Task"}]}
    calls = [{"request": encoded(request, 2) if field == "request" else request,
              "response": {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}}]
    trajectory = {"model_calls": encoded(calls, 2) if field == "model_calls" else calls}
    path = raw_file(tmp_path, {
        "_ng_task_index": 0,
        "ng_trajectory": encoded(trajectory, 2) if field == "ng_trajectory" else trajectory,
    })
    assert "Task" in initial_request(_read_input_record(path, 0), task="0", repeat=0)["prompt"]
    recovered = recover_model_calls(path, 0)
    assert len(recovered["calls"]) == 1
    assert recovered["steps"][-1]["message"] == "Done"


def test_anthropic_blocks_and_arbitrary_tool_names_preserve_evidence(tmp_path):
    request = {"system": [{"type": "text", "text": "Rules"}], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Look"},
                                     {"type": "image", "source": {"data": "IMAGE"}}]},
    ]}
    action = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Plan"},
        {"type": "tool_use", "id": "c1", "name": "custom_tool", "input": {"x": 42}},
    ]}
    output = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "Output", "is_error": True},
    ]}
    path = raw_file(tmp_path, {"ng_trajectory": {"model_calls": [
        {"request": request, "response": action},
        {"request": {**request, "messages": [*request["messages"], action, output]},
         "response": {"role": "assistant", "content": "Done"}},
    ]}})
    recovered = recover_model_calls(path, 0)
    steps = recovered["steps"]
    assert [s["source"] for s in steps] == ["system", "user", "assistant", "tool", "assistant"]
    call = steps[2]["tool_calls"][0]
    assert call["arguments"] == {"x": 42}
    assert call["result_event_idx"] == 3
    assert steps[3]["is_error"] is True
    assert steps[2]["reasoning_content"] == "Plan"
    assert "IMAGE" in json.dumps(recovered["messages"])
    assert recovered["limitations"]["raw non-text content retained but not inspected"] == 1


@pytest.mark.parametrize("literal", ['{"question":"literal JSON task"}', '"quoted"'])
def test_literal_json_tasks_are_not_reinterpreted(literal):
    view = parsed_input({"prompt_text": literal})
    assert view["messages"] == [{"role": "user", "content": literal}]
    assert view["text"] == literal


def test_encoded_input_only_request_reaches_detector_and_mcp(monkeypatch):
    item = {"input": encoded({"input": [{"role": "user", "content": "Task"}]}, 3)}
    expected = [{"role": "user", "content": "Task"}]
    assert parsed_input(item)["messages"] == expected
    assert run_detector(
        "def check(item): return item['input_messages'][0]['content'] == 'Task'", item,
    )
    monkeypatch.setattr(tools_data, "_load_item_raw", lambda _: item)
    assert tools_data.get_input("id")["value"] == expected


def test_encoded_request_prefix_blocks_agree_across_capture_and_mcp(tmp_path, monkeypatch):
    blocks = [{"type": "text", "text": '{"literal": "JSON inside instructions"}'}]
    request = {"system": encoded(blocks, 3), "input": "Task"}
    path = raw_file(tmp_path, {"ng_trajectory": {"model_calls": [{"request": request}]}})
    initial = initial_request(_read_input_record(path, 0), task="0", repeat=0)
    recovered = recover_model_calls(path, 0)
    item = {"response_text": {"steps": recovered["steps"], "model_call_evidence": recovered}}
    assert initial["messages"][0]["content"] == blocks
    assert initial["messages"][0]["content_raw"] == request["system"]
    assert trace.input_messages(item) == initial["messages"]
    assert trace.model_calls(item)[0]["request"] == initial["messages"]
    monkeypatch.setattr(tools_data, "_load_item_raw", lambda _: item)
    page = tools_data.read_parsed_field("id", "input_messages", "/0/content/0/text")
    assert page["value"] == blocks[0]["text"]


def test_invalid_and_unknown_structures_are_explicit():
    bad = '{"steps":'
    assert not parse_trajectory(bad).ok
    assert parse_trajectory(bad).warning
    step = {"source": "assistant", "tool_calls": [{"name": "x", "arguments": '{"broken":'}]}
    parsed = parse_trajectory({"steps": [step]})
    assert parsed.ok and parsed.limitations
    assert trace.tool_calls({"response_text": {"steps": [step]}})[0]["arguments"] == '{"broken":'
    assert decode_container(encoded({"x": 1}, 10)) != {"x": 1}
    assert input_view([{"role": "unexpected", "content": "Do not drop silently"}])["limitations"]


def test_mcp_parsed_fields_page_without_corrupting_nested_json(monkeypatch):
    content = "λ🚀" * 1000
    arguments = {"a/b~c": {"body": content}}
    step = {"source": "assistant",
            "tool_calls": [{"function_name": "tool", "arguments": json.dumps(arguments)}]}
    item = {"prompt_text": encoded([{"role": "user", "content": content}], 2),
            "response_text": json.dumps({"steps": [step]})}
    monkeypatch.setattr(tools_data, "_load_item_raw", lambda _: item)
    for name in ("get_input", "get_model_calls", "read_parsed_field"):
        assert name in TOOL_CALLABLES
    small = tools_data.get_input("id", max_chars=256)
    assert not small["ok"] and small["child_pointers"] == ["/0"]
    assert small["field"] == "input_messages"
    for field, pointer in [
        ("input_messages", "/0/content"),
        ("trajectory_steps", "/0/tool_calls/0/arguments/a~1b~0c/body"),
    ]:
        offset, parts = 0, []
        while True:
            page = tools_data.read_parsed_field("id", field, pointer, offset=offset, max_chars=256)
            assert page["ok"]
            parts.append(page["value"])
            if page["is_final"]:
                break
            offset = page["next_offset"]
        assert "".join(parts) == content
    assert not tools_data.read_parsed_field("id", "input_messages", "/-1")["ok"]
    assert tools_data.get_trajectory_step("id", 1)["step"]["tool_calls"][0]["arguments"] == (
        arguments
    )


def test_model_call_mcp_paging_and_no_grader_metadata(tmp_path, monkeypatch):
    calls = [
        {"request": {"input": "first", "authorization": "SECRET"},
         "response": {"output": [{"role": "assistant", "content": "answer"}]}},
        {"request": {"input": "second"},
         "response": {"output": [{"role": "assistant", "content": "final"}]}},
    ]
    path = raw_file(tmp_path, {"ng_trajectory": {"model_calls": calls}, "grader": "SECRET"})
    recovered = recover_model_calls(path, 0)
    assert "SECRET" not in json.dumps(recovered)
    item = {"response_text": json.dumps({"steps": recovered["steps"],
                                         "model_call_evidence": recovered})}
    monkeypatch.setattr(tools_data, "_load_item_raw", lambda _: item)
    first = tools_data.get_model_calls("id", limit=1)
    assert first["ok"] and first["next_offset"] == 1
    assert first["value"][0]["request"][0]["content"] == "first"
    last = tools_data.read_parsed_field("id", "model_calls", "/1/request/0/content")
    assert last["value"] == "second"


def test_local_gym_capture_and_parquet_input_projection(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.data.gym_adapter.converted_root", lambda: tmp_path / "converted")
    path = raw_file(tmp_path, {
        "_ng_task_index": 0, "_ng_rollout_index": 2,
        "ng_trajectory": {"model_calls": [{
            "request": {"instructions": "Rules", "input": "Task"},
            "response": {"output": [{"role": "assistant", "content": "Done"}]},
        }]},
    })
    row = Dataset(parquet_paths=[materialize_gym(path)]).load_items()[0]
    assert row["run_idx"] == 2
    assert trace.input_messages(row)[0]["content"] == "Rules"
    assert trace.final_message(row) == "Done"
    assert derive_trace_fields(row)["input_text"].endswith("Task")


def test_encoded_multimodal_blocks_and_tool_schemas():
    blocks = [{"type": "input_text", "text": "Inspect"},
              {"type": "input_image", "image_url": "image-reference"}]
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    view = input_view(
        [{"role": "user", "content": encoded(blocks, 2)}],
        tools=[{"type": "function",
                "function": encoded({"name": "any", "parameters": encoded(schema, 2)}, 2)}],
    )
    assert view["messages"][0]["content"] == blocks
    assert "Inspect" in view["text"]
    assert view["tools"][0]["function"]["parameters"] == schema
    assert view["limitations"] == ["non-text content retained but not inspected"]


def test_atif_tool_result_wrappers_and_mixed_anthropic_messages():
    item = {"response_text": {"steps": [{
        "source": "agent",
        "tool_calls": [{"tool_call_id": "c", "function_name": "custom", "arguments": "{}"}],
        "observation": {"results": [{"source_call_id": "c", "content": '{"result": 3}'}]},
    }]}}
    assert trace.steps(item)[0].observation_data["results"][0]["content"] == {"result": 3}
    assert trace.tool_calls(item)[0]["call_id"] == "c"
    parsed = parse_trajectory([{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c", "content": "Output"},
        {"type": "text", "text": "Continue"},
        {"type": "tool_result", "tool_use_id": "d", "content": "More output"},
    ]}])
    assert [s["source"] for s in parsed._steps] == ["tool", "user", "tool"]
    assert parsed._steps[1]["message"] == "Continue"


def snapshot_event(index, ids, new):
    text = json.dumps({"message_ids": ids, "messages_first_seen_here": new,
                       "reward": "SECRET", "authorization": "SECRET"})
    return {"event_idx": index, "kind": "other", "role": "harness", "actor": "main",
            "n_parts": 1, "parts": [{"text": text, "chars": len(text), "part_type": "text"}]}


def test_request_snapshot_schema_restores_inputs_without_replaying_history():
    initial = {"m0": {"role": "developer", "content": "Instructions"},
               "m1": {"role": "user", "content": "Task"}}
    events = [
        snapshot_event(0, list(initial), initial),
        {"event_idx": 1, "kind": "message", "role": "assistant", "n_parts": 1,
         "parts": [{"part_idx": 0, "part_type": "text", "text": "Done", "chars": 4}]},
        snapshot_event(2, ["m0", "m1", "m2"], {"m2": {"role": "assistant", "content": "Done"}}),
    ]
    envelope = normalize_events({"n_events": 3, "n_tool_calls": 0}, events, [])
    item = {"response_text": json.dumps(envelope), "prompt_text": "Task"}
    assert len(trace.steps(item)) == 1
    assert trace.input_messages(item) == list(initial.values())
    assert [len(call["request"]) for call in trace.model_calls(item)] == [2, 3]
    assert "SECRET" not in json.dumps(envelope)
    assert "tool definitions not recorded" in parsed_input(item)["limitations"][0]


@pytest.mark.parametrize("failure", ["missing", "conflict", "truncated"])
def test_snapshot_inconsistencies_fail_visibly(failure):
    events = [snapshot_event(0, ["m0"], {"m0": {"role": "user", "content": "Task"}})]
    if failure == "missing":
        events.append(snapshot_event(1, ["m0", "unknown"], {}))
    elif failure == "conflict":
        events.append(snapshot_event(1, ["m0"], {"m0": {"role": "user", "content": "Different"}}))
    else:
        events[0]["parts"][0]["chars"] += 1
    with pytest.raises(DataApiError):
        recover_request_snapshots(events)


def test_later_snapshot_is_not_claimed_as_initial_input():
    events = [{"role": "assistant", "kind": "message"},
              snapshot_event(1, ["m0"], {"m0": {"role": "user", "content": "Later"}})]
    evidence = recover_request_snapshots(events)
    evidence.pop("event_indices")
    item = {"prompt_text": "Original",
            "response_text": json.dumps({"steps": [], "model_call_evidence": evidence})}
    assert parsed_input(item)["text"] == "Original"


def test_snapshot_source_materialization_updates_prompt(tmp_path):
    identity = {"run_id": "run", "task_id": "task", "repeat": 0}
    event = snapshot_event(0, ["m"], {"m": {"role": "developer", "content": "FULL_CONTEXT"}})
    event.update(identity)
    for part in event["parts"]:
        part.update(identity, event_idx=0, part_idx=0)

    class Api:
        base_url = "https://api.test"

        def _get_json(self, path, query=None):
            assert path == "/v1/runs/run"
            return {"run_id": "run"}

        def download_trace_export(self, run_id, dest, *, progress=None, rollouts=None):
            return write_trace_archive(dest, [{
                "metadata": {**identity, "n_events": 1, "n_tool_calls": 0,
                             "record_schema": "frontier-agentic-v1"},
                "events": [event], "calls": [],
            }])

    path = write_rows(tmp_path / "export.parquet", [{
        "problem_id": "task", "run_idx": 0, "response_text": "", "reasoning_text": "",
        "prompt_text": "Only the task", "is_correct": True,
    }])
    prepared = materialize_source_traces(path, api=Api(), run_id="run")
    row = Dataset(parquet_paths=[prepared]).load_items()[0]
    assert row["prompt_text"] == "[Initial model request: developer]\nFULL_CONTEXT"
    assert trace.input_messages(row) == [{"role": "developer", "content": "FULL_CONTEXT"}]


def test_judge_redacts_outcomes_inside_encoded_envelopes():
    item = {"response_text": encoded(
        {"steps": [{"source": "assistant", "message": "Done"}], "reward": "SECRET_OUTCOME"}, 3,
    )}
    snapshot = make_snapshot(item, item_id="id", source="def check(item): return True")
    assert "SECRET_OUTCOME" not in json.dumps(snapshot["fields"])
    assert "response_text.reward" in snapshot["redacted_metadata"]


def test_parquet_input_and_tools_columns_are_available_to_detectors(tmp_path):
    schema = {"type": "object", "properties": {"code": {"type": "string"}}}
    path = write_rows(tmp_path / "input.parquet", [{
        "problem_id": "task", "run_idx": 0, "response_text": "Done",
        "input": encoded([{"role": "user", "content": "Task"}], 2),
        "tools": encoded([{"name": "custom", "parameters": encoded(schema, 2)}], 2),
    }])
    row = Dataset(parquet_paths=[path]).load_items()[0]
    assert trace.input_messages(row) == [{"role": "user", "content": "Task"}]
    assert trace.input_tools(row)[0]["parameters"] == schema
    assert run_detector(
        "def check(item):\n"
        "    return item['input_tools'][0]['parameters']['properties']['code']['type'] == 'string'",
        row,
    )


def test_export_system_text_reaches_detectors_and_full_judge_evidence(tmp_path):
    system = "Keep the required format.\n" + "λ" * 20_000 + "\nSYSTEM_TAIL"
    prompt = '{"question":"literal task"}'
    path = write_rows(tmp_path / "export.parquet", [{
        "model_id": "m", "benchmark": "b", "problem_id": "task", "run_idx": 0,
        "system_text": system, "prompt_text": prompt, "response_text": "Answer",
        "reasoning_text": "Reasoning", "is_correct": False,
        "judge_text": "PRIVATE_GRADER_RESULT",
    }])
    dataset = Dataset([path])
    loaded = dataset.load_items()[0]
    assert loaded["system_text"] == system
    assert search_items(dataset, query="SYSTEM_TAIL")["n"] == 1
    assert parsed_input(loaded)["messages"] == [
        {"role": "system", "content": system}, {"role": "user", "content": prompt},
    ]
    assert run_detector(
        """
def check(item):
    return (item['system_text'].endswith('SYSTEM_TAIL')
            and item['input_messages'][0]['role'] == 'system'
            and item['input_messages'][1]['role'] == 'user'
            and len(item['input_messages'][0]['content']) > 20000)
""",
        loaded,
    )
    snapshot = make_snapshot(loaded, item_id="task", source="def check(item): return True")
    assert snapshot["fields"]["system_text"] == system
    assert json.loads(snapshot["fields"]["input_messages"])[0]["content"] == system
    assert "judge_text" not in snapshot["fields"] and "is_correct" not in snapshot["fields"]
    excerpt = item_trace_for_judge(loaded, max_chars=3000)
    assert "## input_text" in excerpt and "SYSTEM_TAIL" in excerpt
    assert "PRIVATE_GRADER_RESULT" not in excerpt


@pytest.mark.parametrize("capture", ["initial_request", "model_call_evidence"])
def test_complete_initial_requests_override_export_summary_for_detectors_and_judges(capture):
    messages = [{"role": "system", "content": "Captured system"},
                {"role": "user", "content": "Captured task"}]
    if capture == "initial_request":
        evidence = {"initial_request": {"messages": messages}}
    else:
        evidence = {"model_call_evidence": {"messages": messages, "tools": [], "calls": [
            {"index": 0, "request_recorded": True, "request": [0, 1]},
        ]}}
    item = {"system_text": "STALE_SUMMARY_SYSTEM", "prompt_text": "STALE_SUMMARY_TASK",
            "response_text": json.dumps({**evidence, "steps": [
                {"source": "assistant", "message": "Answer"},
            ]})}
    view = parsed_input(item)
    assert view["messages"] == messages
    assert "STALE_SUMMARY" not in view["text"]
    excerpt = item_trace_for_judge(item)
    assert "Captured system" in excerpt and "Captured task" in excerpt
    assert "STALE_SUMMARY" not in excerpt


def test_structured_prompt_context_is_not_duplicated():
    messages = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Task"}]
    view = parsed_input({"system_text": "Rules", "prompt_text": json.dumps(messages)})
    assert view["messages"] == messages


# Context is distinct when either its role or its content differs from system_text.
@pytest.mark.parametrize("role,content", [
    ("developer", "System rules"), ("system", "Other system rules"),
])
def test_distinct_system_and_structured_context_both_reach_the_judge(role, content):
    messages = [{"role": role, "content": content}, {"role": "user", "content": "Task"}]
    item = {"system_text": "System rules", "prompt_text": json.dumps(messages)}
    view = parsed_input(item)
    assert view["messages"] == [{"role": "system", "content": "System rules"}, *messages]
    excerpt = item_trace_for_judge(item)
    assert f"[Exported input: {role}]\n{content}" in excerpt
    assert "[Exported input: system]\nSystem rules" in excerpt


def test_captured_system_summary_does_not_consume_an_extra_judge_budget():
    system = "SYSTEM_RULES " * 2500
    request = {"messages": [
        {"role": "system", "content": system}, {"role": "user", "content": "Task"},
    ]}
    answer = "a" * 600 + " CRITICAL_FAILURE_EVIDENCE " + "b" * 9000
    item = {
        "prompt_text": input_view(request)["text"],
        "response_text": json.dumps({"initial_request": request, "steps": [
            {"source": "assistant", "message": answer},
        ]}),
    }
    baseline = item_trace_for_judge(item, max_chars=6000)
    migrated = item_trace_for_judge({**item, "system_text": system}, max_chars=6000)
    assert migrated == baseline
    assert "CRITICAL_FAILURE_EVIDENCE" in migrated
    assert "## system_text" not in migrated


def test_generic_distinct_question_is_not_discarded_as_a_duplicate():
    excerpt = item_trace_for_judge({
        "prompt_text": "Formatting instructions", "question": "Distinct question",
        "system_text": "System rules", "response_text": "Answer",
    })
    assert "Formatting instructions" in excerpt and "Distinct question" in excerpt
    assert "System rules" in excerpt


def test_system_text_changes_invalidate_the_parsed_input_cache():
    item = {"system_text": "First system", "prompt_text": "Same task"}
    assert parsed_input(item)["messages"][0]["content"] == "First system"
    item["system_text"] = "Second system"
    assert parsed_input(item)["messages"][0]["content"] == "Second system"


def test_legacy_prompt_projection_remains_byte_exact():
    prompt = "  λ task\n"
    view = parsed_input({"system_text": "   ", "prompt_text": prompt})
    assert view["messages"] == [{"role": "user", "content": prompt}]
    assert view["text"] == prompt and view["origin"] == "prompt_text"


def test_system_only_export_keeps_role_and_missing_task_limitation():
    view = parsed_input({"system_text": "Only rules", "prompt_text": ""})
    assert view["messages"] == [{"role": "system", "content": "Only rules"}]
    assert "task input unavailable" in view["limitations"]


def test_system_input_does_not_displace_localised_response_evidence():
    source = """
import re
def detect(item):
    return bool(re.search('RULE', item['system_text']) and
                re.search('RULE', item['response_text']))
"""
    record = {}
    assert run_detector(source, {"system_text": "RULE", "response_text": "RULE"},
                        localisation=record)
    assert record["passages"][0]["field"] == "response_text"
