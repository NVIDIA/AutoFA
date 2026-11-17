# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only API audit of export -> source recovery -> detector/MCP parsing.

Run with .venv/bin/python tools/validate_input_parsing.py --run RUN_ID ... --output DIR.
Artifacts stay in DIR; the summary contains counts and provenance, not task contents.
No model inference is required. Exit nonzero on retrieval or parsing inconsistencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from autofa import trace
from autofa.config import RunConfig, load_config
from autofa.core.library import FunctionLibrary
from autofa.core.sandbox import run_detector
from autofa.data.input_view import parsed_input
from autofa.data.parquet_store import Dataset
from autofa.data.remote_api import DataApiClient
from autofa.data.source_export import materialize_source_traces
from autofa.data.structured import decode_container
from autofa.data.trajectory import parse_trajectory
from autofa.mcp import tools_data
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics.judge_evidence import make_snapshot


def _independent_container(value):
    """Audit oracle: decode boundary containers, preserving literal JSON scalars."""
    original = value
    for _ in range(8):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return original
    return value if isinstance(value, (dict, list)) else original


def _check_raw_trajectory(record, metadata, parsed, view):
    """Independently check recovered ATIF against the original JSONL payload."""
    identities = [record[k] for k in ("problem_idx", "problem_id", "task_id") if k in record]
    repeats = [record[k] for k in ("repeat", "run_idx") if k in record]
    assert identities and all(str(v) == metadata["task_id"] for v in identities)
    assert repeats and all(type(v) is int and v == metadata["repeat"] for v in repeats)
    raw = _independent_container(record.get("trajectory", record))
    if isinstance(raw, list):
        if len(raw) == 1 and "steps" in _independent_container(raw[0]):
            raw = _independent_container(raw[0])
    source_steps = _independent_container(raw["steps"] if isinstance(raw, dict) else raw)
    assert len(parsed._steps) == len(source_steps), "raw trajectory lost steps"
    checked = {"raw_steps_checked": 0, "raw_tool_calls_checked": 0,
               "raw_tool_results_checked": 0, "raw_input_messages_checked": 0}
    leading = []
    for source, actual in zip(source_steps, parsed._steps, strict=True):
        source = _independent_container(source)
        role = str(source.get("source") or source.get("role") or "assistant").lower()
        if role in {"function", "environment"}:
            role = "tool"
        assert actual["source"] == role, "raw trajectory role changed"
        assert actual["step_id"] == source["step_id"], "raw step identity changed"
        for field in ("message", "reasoning_content", "reasoning", "thought"):
            if field in source:
                assert actual.get(field) == source[field], f"raw {field} changed"
        source_calls = _independent_container(source.get("tool_calls")) or []
        actual_calls = actual.get("tool_calls") or []
        assert len(actual_calls) == len(source_calls), "raw tool calls lost"
        for raw_call, call in zip(source_calls, actual_calls, strict=True):
            function = _independent_container(raw_call.get("function"))
            function = function if isinstance(function, dict) else raw_call
            expected_name = (raw_call.get("function_name") or function.get("name")
                             or raw_call.get("name") or "unknown")
            expected_args = raw_call.get("arguments")
            if expected_args is None:
                expected_args = raw_call.get("args")
            if expected_args is None:
                expected_args = function.get("arguments", function.get("input"))
            assert call["name"] == expected_name, "raw tool name changed"
            assert call["arguments"] == _independent_container(expected_args), "raw args changed"
            identity = raw_call.get("call_id", raw_call.get("id", raw_call.get("tool_call_id")))
            if identity is not None:
                assert call["call_id"] == identity, "raw tool-call identity changed"
            checked["raw_tool_calls_checked"] += 1
        observation = _independent_container(source.get("observation"))
        if isinstance(observation, dict) and isinstance(observation.get("results"), list):
            results = actual["observation"]["results"]
            assert len(results) == len(observation["results"]), "raw tool results lost"
            for raw_result, result in zip(observation["results"], results, strict=True):
                assert result.get("source_call_id") == raw_result.get("source_call_id")
                assert result.get("content") == _independent_container(raw_result.get("content"))
                checked["raw_tool_results_checked"] += 1
        elif observation is not None:
            assert actual.get("observation") == observation, "raw observation changed"
        checked["raw_steps_checked"] += 1
        if len(leading) == checked["raw_steps_checked"] - 1 and role in {
            "system", "developer", "user",
        }:
            leading.append({"role": role, "content": source.get("message", "")})
    if leading:
        assert view["messages"] == leading, "raw leading input messages changed or lost"
        checked["raw_input_messages_checked"] = len(leading)
    return checked


def _raw_trajectory_checked(prepared, envelope, original, parsed, view):
    """Read the exact cached source record using JSON, independent of streaming recovery."""
    scope = envelope.get("evidence_scope", {})
    if scope.get("trajectory_source") != "raw record trajectory":
        return {}
    metadata = original["metadata"]
    reference = metadata["raw_record_ref"]
    assert scope["raw_trajectory_reference"] == reference, "raw trajectory reference changed"
    name, _, offset = reference.rpartition(":")
    assert name.endswith(".jsonl") and offset.isdigit(), "unsupported raw reference"
    source = prepared.parent / "initial-inputs" / (hashlib.sha256(name.encode()).hexdigest()
                                                   + ".jsonl")
    with source.open("rb") as handle:
        if int(offset):
            handle.seek(int(offset) - 1)
            assert handle.read(1) == b"\n", "raw trajectory offset is not a record boundary"
        record = json.loads(handle.readline())
    return _check_raw_trajectory(record, metadata, parsed, view)


def _assert_mcp_value(item_id, field, expected, pointer=""):
    """Compare complete structures through public MCP paging, including long text."""
    offset = 0
    while True:
        page = tools_data.read_parsed_field(
            item_id, field, pointer, offset=offset, limit=200, max_chars=100_000,
        )
        if not page["ok"]:
            assert isinstance(expected, (dict, list)), page
            keys = list(expected) if isinstance(expected, dict) else list(range(len(expected)))
            selected = keys[offset:offset + 200]
            paths = [pointer + "/" + str(k).replace("~", "~0").replace("/", "~1")
                     for k in selected]
            assert page.get("child_pointers") == paths, "MCP lost structured child pointers"
            assert page["total"] == len(expected), "MCP lost container entries"
            for key, path in zip(selected, paths, strict=True):
                _assert_mcp_value(item_id, field, expected[key], path)
            offset += len(selected)
            if offset >= len(expected):
                return
            continue
        if isinstance(expected, dict):
            keys = list(expected)[offset:offset + 200]
            want = {key: expected[key] for key in keys}
        elif isinstance(expected, list):
            want = expected[offset:offset + 200]
        elif isinstance(expected, str):
            want = expected[offset:offset + 100_000]
        else:
            want = expected
        assert page["value"] == want, f"MCP changed {field}{pointer} at offset {offset}"
        assert page["value_type"] == type(expected).__name__, "MCP changed field type"
        total = len(expected) if isinstance(expected, (dict, list, str)) else 1
        assert page["total"] == total, "MCP reported incorrect field size"
        end = offset + (len(want) if isinstance(want, (dict, list, str)) else 1)
        assert page["is_final"] == (end >= total), "MCP stopped paging prematurely"
        assert page["next_offset"] == (end if end < total else None), "MCP skipped field content"
        if page["is_final"]:
            assert page["next_offset"] is None
            return
        assert page["next_offset"] > offset, "MCP paging did not advance"
        offset = page["next_offset"]


def _source_steps_checked(original, parsed, retained_indices):
    """Compare source parts at their own event, never against backups or other steps."""
    by_id = {step.get("step_id"): step for step in parsed._steps}
    checked = 0
    for event in original["events"]:
        if event["event_idx"] not in retained_indices:
            continue
        step = by_id[event["event_idx"]]
        content, reasoning = [], []
        for part in sorted(event["parts"], key=lambda p: p["part_idx"]):
            if part["part_type"] not in {"text", "reasoning"}:
                continue
            is_reasoning = (part.get("channel") in {"reasoning", "analysis"}
                            or part["part_type"] == "reasoning"
                            or event["kind"] == "reasoning")
            (reasoning if is_reasoning else content).append(part["text"])
            checked += 1
        if reasoning:
            assert step["reasoning_content"] == "\n\n".join(reasoning), "source reasoning changed"
        if content:
            expected = "\n\n".join(content)
            if step["source"] == "tool":
                expected = _independent_container(expected)
                assert step["observation"] == expected, "source tool output changed"
            else:
                assert step["message"] == expected, "source message changed"
    return checked


def sample_export(
    source: Path, dest: Path, *, samples: int, seed: int, run_id: str,
) -> list[int]:
    """Sample uniformly without replacement across the entire export, in source order."""
    pf = pq.ParquetFile(source)
    if not pf.metadata.num_rows:
        raise ValueError("Cannot sample an empty export")
    if samples < 1:
        raise ValueError("samples must be positive")
    indices = sorted(random.Random(f"{seed}:{run_id}").sample(
        range(pf.metadata.num_rows), min(samples, pf.metadata.num_rows),
    ))
    # Read only row groups containing sampled rows, retaining Arrow types/metadata.
    chunks = []
    offset = 0
    for group in range(pf.metadata.num_row_groups):
        size = pf.metadata.row_group(group).num_rows
        local = [i - offset for i in indices if offset <= i < offset + size]
        if local:
            chunks.append(pf.read_row_group(group).take(pa.array(local, type=pa.int64())))
        offset += size
    pq.write_table(pa.concat_tables(chunks), dest)
    return indices


def audit(api: DataApiClient, run_id: str, root: Path, samples: int, seed: int = 0) -> dict:
    catalog = api.get_run(run_id)
    root.mkdir(parents=True, exist_ok=True)
    export = root / "export.parquet"
    api.download_autofa_export(run_id, export)
    pf = pq.ParquetFile(export)
    sample = root / "sample.parquet"
    indices = sample_export(export, sample, samples=samples, seed=seed, run_id=run_id)
    selection = {
        "seed": seed,
        "method": "uniform rows without replacement; Random(f'{seed}:{run_id}')",
        "export_rows": pf.metadata.num_rows,
        "row_indices": indices,
        "identities": pq.read_table(sample, columns=["problem_id", "run_idx"]).to_pylist(),
    }
    announced = root / "announced-selection.json"
    if announced.exists():
        assert selection == json.loads(announced.read_text()), "Announced sample changed"
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    prepared = materialize_source_traces(sample, api=api, run_id=run_id, catalog=catalog)
    ds = Dataset(parquet_paths=[prepared])
    set_runtime(
        Runtime(
            dataset=ds,
            library=FunctionLibrary(),
            config=load_config(),
            run=RunConfig(model_id="*", run_id="parsing-audit", parquet_paths=[str(prepared)]),
        )
    )
    rows = []
    for item in ds.load_items(limit=samples):
        iid = item["_item_id"]
        parsed = parse_trajectory(item["response_text"])
        view = parsed_input(item)
        if view["origin"] in {"captured_initial_request", "recorded_agent_request"}:
            assert item["prompt_text"] == view["text"], "stored prompt omits captured input"
        envelope = decode_container(item["response_text"])
        expected = sum(len(s.get("tool_calls", [])) for s in envelope["steps"])
        actual = trace.tool_calls(item)
        assert len(actual) == expected, "lost tool calls"
        cache_key = hashlib.sha256(
            json.dumps(
                [run_id, item["problem_id"], item["run_idx"]],
            ).encode()
        ).hexdigest()
        # Adapter upgrades deliberately reuse original source caches from older versions.
        candidates = [prepared.parent / (cache_key + ".json"),
                      *sorted(prepared.parent.parent.glob(f"source-v*/{cache_key}.json"))]
        original_path = next(p for p in candidates if p.is_file())
        original = json.loads(original_path.read_text())
        raw_checks = _raw_trajectory_checked(prepared, envelope, original, parsed, view)
        retained_indices = {s.get("step_id") for s in envelope["steps"]}
        original_calls = sorted(
            [c for c in original["calls"] if c["event_idx"] in retained_indices],
            key=lambda c: (c["event_idx"], c["call_idx"]),
        )
        if original_calls:
            assert len(actual) == len(original_calls), "source API/tool count mismatch"
            for source_call, parsed_call in zip(original_calls, actual, strict=True):
                expected_args = _independent_container(source_call["args_full"])
                assert parsed_call["arguments"] == expected_args, "tool argument changed"
                assert parsed_call["name"] == source_call["tool_name"], "tool name changed"
                source_step = parsed._steps[parsed_call["step"] - 1]
                assert source_step["step_id"] == source_call["event_idx"], "tool event changed"
                assert parsed_call["source_call_index"] == source_call["call_idx"]
                assert parsed_call["result_event_idx"] == source_call.get("result_event_idx")
        checked_parts = _source_steps_checked(original, parsed, retained_indices)
        for call in actual:
            assert call["arguments"] == decode_container(call["arguments"]), "encoded arguments"
        assert not envelope["steps"] or parsed.ok, "source trajectory failed parsing"
        mcp = tools_data.get_trajectory(iid)
        assert mcp.get("ok_parse") == parsed.ok, "MCP/detector parse disagreement"
        assert mcp["steps_total"] == len(envelope["steps"]), "MCP lost steps"
        input_meta = tools_data.read_parsed_field(iid, "input_parse")
        assert input_meta["value"]["origin"] == view["origin"], "MCP input provenance mismatch"
        for field, value in (("input_messages", view["messages"]),
                             ("input_tools", view["tools"]), ("input_text", view["text"]),
                             ("input_parse", {"origin": view["origin"],
                                              "limitations": view["limitations"]})):
            _assert_mcp_value(iid, field, value)
        for section in ("messages", "tools", "text"):
            page = tools_data.get_input(iid, section)
            assert page["origin"] == view["origin"]
            assert page["limitations"] == view["limitations"]
        detector = """
from autofa.trace import input_messages, tool_calls
def check(item):
    return item['input_messages'] == input_messages(item) and all(
        c['arguments'] is None or isinstance(c['arguments'], (dict, list, str, int, float, bool))
        for c in tool_calls(item))
"""
        assert run_detector(detector, item), "detector input does not match helper"
        snapshot = make_snapshot(item, item_id=iid, source=detector)
        assert json.loads(snapshot["fields"]["input_messages"]) == view["messages"]
        calls = trace.model_calls(item)
        if calls:
            assert view["origin"] == "captured_initial_request" or not calls[0]["request_recorded"]
        for index, call in enumerate(calls):
            _assert_mcp_value(iid, "model_calls", call, f"/{index}")
        rows.append(
            {
                "task_id": item["problem_id"],
                **raw_checks,
                "repeat": item["run_idx"],
                "parsed": parsed.ok,
                "source_api_events": len(original["events"]),
                "source_api_tool_calls": len(original["calls"]),
                "trajectory_source": envelope.get("evidence_scope", {}).get(
                    "trajectory_source", "normalized API events",
                ),
                "source_text_parts_checked": checked_parts,
                "mcp_input_fields_checked": 4,
                "mcp_model_calls_checked": len(calls),
                "steps": parsed.n_steps,
                "tool_calls": len(actual),
                "tool_names": sorted({c["name"] for c in actual}),
                "input_messages": len(view["messages"]),
                "input_origin": view["origin"],
                "input_limitations": view["limitations"],
                "parse_limitations": parsed.limitations,
                "source_limitations": envelope.get("evidence_scope", {}).get("limitations", {}),
                "model_calls": len(envelope.get("model_call_evidence", {}).get("calls", [])),
            }
        )
    manifest = json.loads((prepared.parent / "manifest.json").read_text())
    return {
        "run_id": run_id,
        "harness": catalog.get("harness"),
        "benchmark": catalog.get("benchmark"),
        "selection": selection,
        "adapter_version": manifest["adapter_version"],
        "export_sha256": manifest["export_sha256"],
        "output_sha256": manifest["output_sha256"],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api = DataApiClient(load_config().data_api_base_url, timeout=45, max_attempts=2)
    report = {
        "checked_at": datetime.now(UTC).isoformat(), "seed": args.seed,
        "samples_per_run": args.samples, "runs": [], "errors": [],
    }
    for run in args.run:
        try:
            result = audit(api, run, args.output / run, max(1, args.samples), seed=args.seed)
            report["runs"].append(result)
            print(json.dumps(result), flush=True)
        except Exception as exc:
            error = {"run_id": run, "error": f"{type(exc).__name__}: {exc}"}
            report["errors"].append(error)
            print(json.dumps(error), flush=True)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
