# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-parsing audits sample every row and reject dropped, misplaced or reinterpreted evidence."""

import json
import random
from copy import deepcopy
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools import validate_input_parsing as audit


def test_random_sample_spans_row_groups_and_is_reproducible(tmp_path):
    schema = pa.schema([("problem_id", pa.string()), ("run_idx", pa.int64())],
                       metadata={b"source": b"audit-fixture"})
    rows = [{"problem_id": str(i), "run_idx": i} for i in range(103)]
    source = tmp_path / "export.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), source, row_group_size=7)
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    selected = audit.sample_export(source, first, samples=13, seed=20260929, run_id="run")
    assert selected == sorted(random.Random("20260929:run").sample(range(103), 13))
    assert audit.sample_export(source, second, samples=13, seed=20260929, run_id="run") == selected
    table = pq.read_table(first)
    assert table.schema == schema
    assert table.to_pylist() == [rows[i] for i in selected]


def _source():
    return {"events": [
        {"event_idx": 0, "kind": "message", "parts": [
            {"part_idx": 1, "part_type": "text", "text": "second"},
            {"part_idx": 0, "part_type": "text", "text": "first"},
            {"part_idx": 2, "part_type": "reasoning", "text": "reason"},
        ]},
        {"event_idx": 1, "kind": "tool_result", "parts": [
            {"part_idx": 0, "part_type": "text", "text": '{"result": [1, 2]}'},
        ]},
    ]}


def _steps():
    return SimpleNamespace(_steps=[
        {"step_id": 0, "source": "assistant", "message": "first\n\nsecond",
         "reasoning_content": "reason"},
        {"step_id": 1, "source": "tool", "observation": {"result": [1, 2]}},
    ])


@pytest.mark.parametrize("damage", [None, "backup_only", "part_order"])
def test_audit_rejects_text_preserved_only_in_wrong_place(damage):
    steps = _steps()
    if damage is None:
        assert audit._source_steps_checked(_source(), steps, {0, 1}) == 4
        return
    if damage == "backup_only":
        steps._steps[0]["message_raw"] = steps._steps[0]["message"]
        steps._steps[0]["message"] = ""
    else:
        steps._steps[0]["message"] = "second\n\nfirst"
    with pytest.raises(AssertionError):
        audit._source_steps_checked(_source(), steps, {0, 1})


@pytest.mark.parametrize("truncate", [False, True])
def test_audit_reads_every_mcp_page_and_rejects_truncated_text(monkeypatch, truncate):
    expected = {"key/with~escapes": ["x" * 100_005]}
    seen = []

    def read(item_id, field, pointer="", **kwargs):
        seen.append((pointer, kwargs["offset"]))
        value = expected
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
        page = audit.tools_data._parsed_page(value, pointer=pointer, **kwargs)
        if page["ok"] and isinstance(value, str) and truncate:
            page["value"] = page["value"][:-1]
        return page

    monkeypatch.setattr(audit.tools_data, "read_parsed_field", read)
    if truncate:
        with pytest.raises(AssertionError):
            audit._assert_mcp_value("item", "input_messages", expected)
    else:
        audit._assert_mcp_value("item", "input_messages", expected)
        assert ("/key~1with~0escapes/0", 100_000) in seen


def _raw_trajectory_fixture():
    source = [
        {"step_id": 0, "source": "system", "message": "System instructions"},
        {"step_id": 1, "source": "user", "message": "The task"},
        {"step_id": 2, "source": "agent", "message": "Running", "reasoning_content": "Plan",
         "tool_calls": [{"tool_call_id": "c1", "function_name": "python",
                         "arguments": '{"code":"{\\"literal\\":1}"}'}],
         "observation": {"results": [{"source_call_id": "c1", "content": '{"value":1}'}]}},
    ]
    actual = deepcopy(source)
    actual[2]["tool_calls"][0].update(
        name="python", call_id="c1", arguments={"code": '{"literal":1}'},
    )
    actual[2]["observation"]["results"][0]["content"] = {"value": 1}
    record = {"problem_idx": "task", "repeat": 0, "trajectory": [{"steps": source}]}
    metadata = {"task_id": "task", "repeat": 0}
    view = {"messages": [{"role": "system", "content": "System instructions"},
                         {"role": "user", "content": "The task"}]}
    return record, metadata, SimpleNamespace(_steps=actual), view


@pytest.mark.parametrize("damage", [None, "args", "task"])
def test_raw_trajectory_oracle_counts_originals_and_rejects_changed_values(damage):
    record, metadata, parsed, view = _raw_trajectory_fixture()
    if damage is None:
        assert audit._check_raw_trajectory(record, metadata, parsed, view) == {
            "raw_steps_checked": 3, "raw_tool_calls_checked": 1,
            "raw_tool_results_checked": 1, "raw_input_messages_checked": 2,
        }
        return
    if damage == "args":
        parsed._steps[2]["tool_calls"][0]["arguments"] = json.dumps({"code": '{"literal":1}'})
    else:
        metadata["task_id"] = "other task"
    with pytest.raises(AssertionError):
        audit._check_raw_trajectory(record, metadata, parsed, view)
