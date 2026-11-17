# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression cases from the input/preprocessing audit (F11 archive excluded)."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import trace
from autofa.core.sandbox import run_detector
from autofa.data import gym_adapter, parquet_store
from autofa.data.corpus_health import measure_parse_coverage
from autofa.data.row_index import DuplicateItemIdError, identity_index_cache_path
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trajectory import parse_trajectory
from autofa.mcp import tools_data
from autofa.metrics.behavior import item_trace_for_judge


def _row(record):
    return gym_adapter._row_from_obj(record, model_id="m", benchmark="b", line_no=0)


def _item(*steps, **fields):
    return {"response_text": json.dumps({"steps": list(steps)}), **fields}


def _parquet(tmp_path, rows):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    return parquet_store.Dataset(parquet_paths=[path])


def test_raw_import_preserves_long_text_and_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(gym_adapter, "converted_root", lambda: tmp_path / "converted")
    reasoning = "x" * 60_000 + "REAL_END"
    output = [{"type": "reasoning", "summary": reasoning}]
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"resolved": False, "response": {"output": output}}) + "\n")
    converted = gym_adapter.materialize_gym(path)
    row = pq.read_table(converted).to_pylist()[0]
    assert row["reasoning_text"] == reasoning
    assert json.loads(row["response_text"]) == output


def test_identity_field_does_not_route_raw_record_to_parsed_adapter():
    row = _row({"task_id": "t", "resolved": True, "reward": 1.0,
                "responses_create_params": {"input": "THE_TASK"}, "response": "THE_ANSWER"})
    assert row["problem_id"] == "t"
    assert row["prompt_text"] == "THE_TASK"
    assert row["response_text"] == "THE_ANSWER"
    assert row["is_correct"] is True and row["reward"] == 1.0


def test_missing_outcome_remains_unknown():
    row = _row({"response": "answer"})
    assert row["is_correct"] is None and row["reward"] is None


def test_parsed_import_keeps_roles_tool_failures_and_extensions(tmp_path, monkeypatch):
    monkeypatch.setattr(gym_adapter, "converted_root", lambda: tmp_path / "converted")
    record = {"task_id": "t", "is_resolved": False, "steps": [
        {"source": "user", "message": "FOLLOWUP_REQUIREMENT"},
        {"source": "assistant", "reasoning_content": "REASON", "tool_calls": [
            {"id": "call-1", "function": {"name": "bash", "arguments": "test"}}]},
        {"source": "tool", "message": "TOOL_FAILURE", "tool_call_id": "call-1"},
        {"source": "assistant", "message": "FINAL_ANSWER", "custom": {"keep": True}},
    ]}
    path = tmp_path / "parsed.jsonl"
    path.write_text(json.dumps(record) + "\n")
    row = pq.read_table(gym_adapter.materialize_gym(path)).to_pylist()[0]
    steps = trace.steps(row)
    assert len(steps) == 4
    assert steps[0].source == "user" and steps[0].message == "FOLLOWUP_REQUIREMENT"
    assert steps[2].observation == "TOOL_FAILURE"
    assert "TOOL_FAILURE" in trace.observations(row)
    assert trace.final_message(row) == "FINAL_ANSWER"
    saved = json.loads(row["response_text"])["steps"]
    assert saved[1]["tool_calls"][0]["id"] == "call-1"
    assert saved[3]["custom"] == {"keep": True}


@pytest.mark.parametrize("bad", ['{"response":', '[]', '{"task_id": "t"}', '{"steps": [{}, null]}'])
def test_invalid_jsonl_records_fail_with_file_and_line(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(gym_adapter, "converted_root", lambda: tmp_path / "converted")
    path = tmp_path / "input.jsonl"
    path.write_text('{"response": "valid"}\n\n' + bad + "\n")
    with pytest.raises(ValueError, match=r"input.jsonl:3:"):
        gym_adapter.materialize_gym(path)
    assert not list((tmp_path / "converted").glob("*/dataset.parquet"))


def test_repeat_identity_round_trips_to_the_right_rollout(tmp_path):
    rows = [_row({"task_id": "same", "_ng_rollout_index": i,
                  "steps": [{"message": f"ANSWER_{i}"}]})
            for i in range(2)]
    ds = _parquet(tmp_path, rows)
    loaded = ds.load_items(limit=10)
    assert [r["run_idx"] for r in loaded] == [0, 1]
    assert len({r["_item_id"] for r in loaded}) == 2
    for i, row in enumerate(loaded):
        fetched = parquet_store.get_item(ds, row["_item_id"], max_chars=None)["item"]
        assert trace.final_message(fetched) == f"ANSWER_{i}"


@pytest.mark.parametrize("identity", [
    {"repeat": -1}, {"repeat": True}, {"repeat": 1, "run_idx": 2},
])
def test_invalid_repeat_identity_rejected(identity):
    with pytest.raises(ValueError):
        _row({"steps": [], **identity})


def test_jsonl_identity_is_unique_across_files(tmp_path):
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path in paths:
        path.write_text('{"response": "answer"}\n')
    rows = list(gym_adapter.iter_gym_rows(paths))
    assert rows[0]["problem_id"] != rows[1]["problem_id"]  # missing-identity fallback
    for path in paths:
        path.write_text('{"task_id": "same", "steps": []}\n')
    with pytest.raises(ValueError, match=r"b.jsonl:1: duplicate.*a.jsonl:1"):
        list(gym_adapter.iter_gym_rows(paths))


@pytest.mark.parametrize("operation", ["index", "lookup", "sample", "stream"])
def test_duplicate_native_ids_never_fall_back_to_an_arbitrary_row(tmp_path, operation):
    row = _row({"task_id": "same", "steps": [{"message": "first"}]})
    ds = _parquet(tmp_path, [row, {**row, "response_text": "second"}])
    with pytest.raises(DuplicateItemIdError, match="duplicate item ID"):
        if operation == "index":
            ds.identity_index()
        elif operation == "lookup":
            parquet_store.get_item(ds, ds._make_item_id(row, ds.parquet_paths[0]))
        elif operation == "sample":
            parquet_store.sample_items(ds)
        else:
            list(ds.iter_batches(batch_size=1))


def test_invalid_cached_identity_index_is_rebuilt(tmp_path):
    ds = _parquet(tmp_path, [_row({"task_id": "t", "steps": []})])
    ds.identity_index()
    cache = identity_index_cache_path(ds.parquet_paths)
    saved = json.loads(cache.read_text())
    saved["rows"].append(saved["rows"][0])
    cache.write_text(json.dumps(saved))
    assert len(ds.identity_index(refresh=True).rows) == 1


def test_late_schema_key_has_same_health_helpers_and_detector_result():
    item = {"response_text": json.dumps({"metadata": "x" * 5000, "steps": [
        {"source": "assistant", "reasoning_content": "TAIL_EVIDENCE"},
    ]})}
    assert parse_trajectory(item["response_text"]).ok
    assert measure_parse_coverage([item], record_diagnostics=False)["parse_rate"] == 1.0
    assert trace.steps(item)[0].reasoning == "TAIL_EVIDENCE"
    assert "TAIL_EVIDENCE" in derive_trace_fields(item)["reasoning_text"]
    assert run_detector('def detect(item, llm_client=None):\n'
                        '    return "TAIL_EVIDENCE" in item.get("reasoning_text", "")', item)


def test_cached_parse_cannot_be_mutated_by_a_previous_caller():
    item = _item({"source": "assistant", "reasoning_content": "ORIGINAL",
                  "tool_calls": [{"name": "bash", "arguments": {"command": "test"}}]})
    parsed = parse_trajectory(item["response_text"])
    parsed._steps[0]["reasoning_content"] = "CHANGED"
    trace.tool_calls(item)[0]["arguments"]["command"] = "changed"
    assert trace.steps(item)[0].reasoning == "ORIGINAL"
    assert trace.tool_calls(item)[0]["arguments"]["command"] == "test"


@pytest.mark.parametrize("data", [
    pytest.param([{"steps": [{"message": "a"}]}, {"steps": [{"message": "b"}]}],
                 id="two_envelopes"),
    pytest.param({"steps": [{"message": "a"}, None]}, id="null_step"),
])
def test_unsupported_shapes_are_rejected_without_partial_evidence(data):
    item = {"response_text": json.dumps(data)}
    result = parse_trajectory(item["response_text"])
    assert not result.ok and result.warning and "partial parse" in result.warning
    assert not result._steps and not trace.steps(item)
    assert "final_message" not in derive_trace_fields(item)


@pytest.mark.parametrize("focus,kept,lost", [("closing", "FINAL_DECISION", "OPENING"),
                                            ("opening", "OPENING", "FINAL_DECISION")])
def test_oversized_selected_step_preserves_requested_end(focus, kept, lost):
    item = _item({"source": "assistant",
                  "reasoning_content": "OPENING" + "x" * 20_000 + "FINAL_DECISION"})
    excerpt = trace.evidence_text(item, surface="reasoning", focus=focus, max_chars=1000)
    assert kept in excerpt and lost not in excerpt
    assert "omitted" in excerpt and len(excerpt) <= 1000


@pytest.mark.parametrize("surface,focus,expected,absent", [
    ("reasoning", "opening", "REASON", "DONE"),
    ("observations", "closing", "TEST_FAILED", "REASON"),
    ("actions", "closing", "bash", "REASON"), ("final", "closing", "DONE", "TEST_FAILED"),
])
def test_step_windows_respect_surface(surface, focus, expected, absent):
    item = _item({"source": "assistant", "reasoning_content": "REASON",
                  "tool_calls": [{"name": "bash", "arguments": "test"}]},
                 {"source": "tool", "observation": "TEST_FAILED"},
                 {"source": "assistant", "message": "DONE"})
    excerpt = trace.evidence_text(item, surface=surface, focus=focus, max_steps=5)
    assert expected in excerpt and absent not in excerpt


def test_native_surface_is_not_replaced_by_other_trajectory_content():
    item = _item({"source": "assistant", "reasoning_content": "OTHER"},
                 reasoning_text="NATIVE_REASON")
    assert "NATIVE_REASON" in trace.evidence_text(item, surface="reasoning", focus="closing")
    assert trace.evidence_text(item, surface="absent_field", focus="closing") == ""


def test_detector_view_pages_exact_derived_unicode_text(monkeypatch):
    item = _item({"source": "assistant", "reasoning_content": "理由" * 50_000 + "END",
                  "message": "DONE"}, reasoning_text="", data={"text": "理由"})
    monkeypatch.setattr(tools_data, "_load_item_raw", lambda _: item)
    assert tools_data.get_field_meta("id", "reasoning_text")["total_chars"] == 0
    meta = tools_data.get_field_meta("id", "reasoning_text", view="detector")
    expected = derive_trace_fields(item)["reasoning_text"]
    assert meta["total_chars"] == len(expected)
    assert meta["stored_chars"] == 0 and meta["provenance"] == "trajectory"
    offset, chunks = 0, []
    while True:
        part = tools_data.read_field_chunk("id", "reasoning_text", offset=offset,
                                            limit=8000, view="detector")
        chunks.append(part["text"])
        if part["is_final"]:
            break
        assert "view='detector'" in part["full_read"]
        offset = part["next_offset"]
    assert "".join(chunks) == expected
    final = tools_data.read_field_chunk("id", "final_message", view="detector")
    assert final["text"] == "DONE" and final["is_final"]
    assert not tools_data.get_field_meta("id", "reasoning_text", view="invalid")["ok"]
    # Structured fields share one unicode serialization between metadata and chunks.
    chunk = tools_data.read_field_chunk("id", "data")
    assert tools_data.get_field_meta("id", "data")["total_chars"] == chunk["total_chars"] == (
        len(chunk["text"])
    )


def test_judge_window_keeps_short_reasoning_between_large_fields():
    item = {"prompt_text": "p" * 4000, "reasoning_text": "CRUCIAL_REASONING_EVIDENCE",
            "response_text": "a" * 6000 + "ACTUAL_END"}
    excerpt = item_trace_for_judge(item, max_chars=5000)
    assert "CRUCIAL_REASONING_EVIDENCE" in excerpt and "ACTUAL_END" in excerpt
    assert "prompt_text" in excerpt and "total_chars=" in excerpt and "omitted" in excerpt
    assert len(excerpt) <= 5000


def test_identical_prompt_and_reasoning_keep_their_distinct_roles():
    excerpt = item_trace_for_judge({"prompt_text": "I give up", "reasoning_text": "I give up"})
    assert "## prompt_text" in excerpt and "## reasoning_text" in excerpt


def test_judge_window_deduplicates_aliases_and_covers_all_surfaces():
    item = _item({"source": "assistant", "reasoning_content": "r" * 9000 + "REASON_END",
                  "tool_calls": [{"name": "bash", "arguments": "test"}]},
                 {"source": "tool", "observation": "FAIL"},
                 {"source": "assistant", "message": "DONE"},
                 prompt_text="UNIQUE_PROMPT", question="UNIQUE_PROMPT")
    excerpt = item_trace_for_judge(item, max_chars=1000)
    assert excerpt.count("UNIQUE_PROMPT") == 1 and "REASON_END" in excerpt
    for field in ("reasoning_text", "actions_text", "observations_text", "final_message"):
        assert f"## {field}" in excerpt
    assert "schema_version" not in excerpt and len(excerpt) <= 1000
    with pytest.raises(ValueError, match="budget too small"):
        item_trace_for_judge(item, max_chars=64)
