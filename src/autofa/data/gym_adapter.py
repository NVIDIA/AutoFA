# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Gym / terminal-bench JSONL → canonical AutoFA parquet."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from autofa.config import autofa_home
from autofa.data.sources import EvalSource, _gym_input_files

CANONICAL_COLUMNS = (
    "model_id",
    "benchmark",
    "problem_id",
    "run_idx",
    "is_correct",
    "prompt_text",
    "response_text",
    "reasoning_text",
    "reward",
    "input_json",
)


def converted_root() -> Path:
    root = autofa_home() / "converted"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_gym_metadata(dir_path: Path) -> dict[str, Any]:
    path = Path(dir_path) / "eval_metadata.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _cache_key(inputs: list[Path], meta_path: Path | None) -> str:
    h = hashlib.sha1()
    # Bump when conversion semantics change so stale flattened parquets rebuild.
    h.update(b"v6-grader-calls-excluded\n")
    for p in sorted(inputs, key=lambda x: str(x)):
        st = p.stat()
        h.update(str(p.resolve()).encode())
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
    if meta_path and meta_path.is_file():
        st = meta_path.stat()
        h.update(str(meta_path.resolve()).encode())
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
    return h.hexdigest()[:16]


def materialize_eval_source(
    source: EvalSource,
    *,
    force: bool = False,
    efficiency_only: bool = False,
) -> Path:
    """Return canonical data, recovering original traces for compact API exports."""
    if source.kind == "evalrdl":
        from autofa.data.local_import import import_local_run

        return import_local_run(
            source.path,
            force=force,
            efficiency_only=efficiency_only,
        ).parquet_path
    if source.kind == "parquet":
        from autofa.data.local_import import reference_for_import
        from autofa.data.local_trace_store import resolve_store

        if reference := reference_for_import(source.path):
            return resolve_store(reference["store_id"]) / "dataset.parquet"
        from autofa.data.local_source import materialize_local_parquet

        return materialize_local_parquet(source.path)
    if source.kind == "gym_dir":
        from autofa.data.local_import import detect_reader, import_local_run

        if detect_reader(source.path):
            return import_local_run(
                source.path,
                force=force,
                efficiency_only=efficiency_only,
            ).parquet_path
        return materialize_gym(source.path, force=force)
    if source.kind == "jsonl":
        from autofa.data.local_import import detect_reader, import_local_run

        if detect_reader(source.path):
            return import_local_run(
                source.path,
                force=force,
                efficiency_only=efficiency_only,
            ).parquet_path
        return materialize_gym(source.path, force=force)
    raise ValueError(f"unknown source kind: {source.kind}")


def materialize_gym(path: Path | str, *, force: bool = False) -> Path:
    """Convert a gym dir or JSONL file to ``~/.autofa/converted/<hash>/dataset.parquet``."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        inputs = _gym_input_files(path)
        meta_path = path / "eval_metadata.json"
        meta = load_gym_metadata(path)
    elif path.is_file() and path.suffix == ".jsonl":
        inputs = [path]
        meta_path = path.parent / "eval_metadata.json"
        meta = load_gym_metadata(path.parent) if meta_path.is_file() else {}
    else:
        raise FileNotFoundError(f"not a gym dir or jsonl: {path}")

    if not inputs:
        raise FileNotFoundError(f"no JSONL inputs under {path}")

    key = _cache_key(inputs, meta_path if meta_path.is_file() else None)
    out_dir = converted_root() / key
    out_parquet = out_dir / "dataset.parquet"
    if out_parquet.is_file() and not force:
        return out_parquet

    rows = list(iter_gym_rows(inputs, metadata=meta))
    if not rows:
        raise ValueError(f"no rows converted from {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    # Ensure stable column order
    present = [c for c in CANONICAL_COLUMNS if c in table.column_names]
    extra = [c for c in table.column_names if c not in present]
    table = table.select(present + extra)
    pq.write_table(table, out_parquet)
    (out_dir / "source.json").write_text(
        json.dumps(
            {
                "source": str(path),
                "inputs": [str(p) for p in inputs],
                "n_rows": len(rows),
                "n_fail": sum(1 for r in rows if r.get("is_correct") is False),
                "n_unknown": sum(1 for r in rows if r.get("is_correct") is None),
                "model_id": rows[0].get("model_id"),
                "benchmark": rows[0].get("benchmark"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_parquet


def iter_gym_rows(
    inputs: list[Path] | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield canonical item dicts from gym JSONL file(s)."""
    if isinstance(inputs, Path):
        inputs = [inputs]
    meta = metadata or {}
    model_id = _meta_model(meta)
    benchmark = _meta_benchmark(meta)

    seen: dict[tuple[Any, ...], str] = {}
    record_index = 0
    for path in inputs:
        with Path(path).open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError("expected a JSON object")
                    row = _row_from_obj(
                        obj, model_id=model_id, benchmark=benchmark, line_no=record_index,
                    )
                    if row is None:
                        raise ValueError("unsupported rollout schema (expected steps or response)")
                    identity = tuple(row[k] for k in (
                        "model_id", "benchmark", "problem_id", "run_idx",
                    ))
                    if identity in seen:
                        raise ValueError(
                            f"duplicate rollout identity {identity!r}; first at {seen[identity]}"
                        )
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no + 1}: {exc}") from exc
                seen[identity] = f"{path}:{line_no + 1}"
                record_index += 1
                yield row


def _meta_model(meta: dict[str, Any]) -> str:
    raw = meta.get("model") or meta.get("model_id") or "unknown"
    s = str(raw).strip()
    # nvidia/nvidia/nemotron-3-ultra → last segment is enough for catalog
    return s.split("/")[-1] if "/" in s else s or "unknown"


def _meta_benchmark(meta: dict[str, Any]) -> str:
    for key in ("dataset", "benchmark"):
        v = meta.get(key)
        if v:
            return str(v)
    return "gym"


def _row_from_obj(
    obj: dict[str, Any],
    *,
    model_id: str,
    benchmark: str,
    line_no: int,
) -> dict[str, Any] | None:
    if "ng_trajectory" in obj:
        from autofa.data.source_inputs import initial_request
        from autofa.data.source_model_calls import recover_calls
        from autofa.data.structured import decode_container

        trajectory = decode_container(obj["ng_trajectory"])
        if not isinstance(trajectory, dict):
            raise ValueError("ng_trajectory must be an object")
        calls = decode_container(trajectory.get("model_calls"))
        if not isinstance(calls, list):
            raise ValueError("model_calls must be an array")
        task = str(obj.get("_ng_task_index", _raw_problem_id(obj) or f"row-{line_no}"))
        repeat = _run_idx(obj)
        initial = initial_request(obj, task=task, repeat=repeat)
        recovered = recover_calls(calls)
        recovered.pop("excluded_call_ids")  # No exported copies of them exist here.
        envelope = {"steps": recovered.pop("steps"),
                    "evidence_scope": {"limitations": recovered.pop("limitations")},
                    "model_call_evidence": recovered}
        correct, reward = _outcome(obj)
        return {"model_id": model_id, "benchmark": benchmark, "problem_id": task,
                "run_idx": repeat, "is_correct": correct, "reward": reward,
                "prompt_text": initial["prompt"], "reasoning_text": "",
                "response_text": json.dumps(envelope, ensure_ascii=False),
                "input_json": json.dumps({"messages": initial.get("messages", []),
                                          "tools": initial["tools"]}, ensure_ascii=False)}
    # Parsed trajectories
    if "steps" in obj:
        return _from_parsed(obj, model_id=model_id, benchmark=benchmark, line_no=line_no)
    # Raw rollouts
    if "responses_create_params" in obj or "response" in obj:
        return _from_raw(obj, model_id=model_id, benchmark=benchmark, line_no=line_no)
    return None


def _normalize_gym_step(step: dict[str, Any]) -> dict[str, Any]:
    """Adapt gym step fields so ``format_actions`` / ``parse_trajectory`` can consume them."""
    # Keep original fields, including user turns, call IDs and dialect extensions.
    out: dict[str, Any] = dict(step)
    src = step.get("source")
    if isinstance(src, str) and src.strip():
        out["source"] = src.strip()
    else:
        # Gym traces often omit source; treat as agent so messages render.
        out["source"] = "agent"

    msg = step.get("message")
    if isinstance(msg, str) and msg.strip():
        out["message"] = msg

    reasoning = step.get("reasoning_content") or step.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        out["reasoning"] = reasoning.strip()
        out["reasoning_content"] = reasoning.strip()

    tool_calls: list[dict[str, Any]] = []
    raw_tcs = step.get("tool_calls")
    if raw_tcs is not None and (
        not isinstance(raw_tcs, list) or any(not isinstance(tc, dict) for tc in raw_tcs)
    ):
        raise ValueError("tool_calls must be an array of objects")
    if isinstance(raw_tcs, list) and raw_tcs:
        for tc in raw_tcs:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
            name = tc.get("name") or tc.get("function_name") or (fn.get("name") if fn else None)
            if not name:
                continue
            args = tc.get("arguments")
            if args is None and tc.get("args") is not None:
                args = tc.get("args")
            if args is None and fn is not None:
                args = fn.get("arguments")
            tool_calls.append({**tc, "name": str(name), "arguments": {} if args is None else args})
    else:
        names = step.get("tool_call_names")
        if isinstance(names, list):
            for n in names:
                if n:
                    tool_calls.append({"name": str(n), "arguments": {}})
    if tool_calls and len(tool_calls) == len(raw_tcs or tool_calls):
        out["tool_calls"] = tool_calls

    if "observation" in step and step["observation"] not in (None, ""):
        out["observation"] = step["observation"]
    elif out["source"].lower() in {"tool", "environment"} and msg is not None:
        out["observation"] = msg

    if "step_id" in step:
        out["step_id"] = step["step_id"]
    return out


def _run_idx(obj: dict[str, Any]) -> int:
    values = []
    for key in ("run_idx", "repeat", "_ng_rollout_index"):
        value = obj.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not (
            isinstance(value, int) or isinstance(value, str) and value.isdecimal()
        ):
            raise ValueError(f"{key} must be a nonnegative integer, got {value!r}")
        index = int(value)
        if index < 0:
            raise ValueError(f"{key} must be nonnegative, got {index}")
        values.append(index)
    if len(set(values)) > 1:
        raise ValueError("conflicting rollout indices: run_idx/repeat/_ng_rollout_index")
    return values[0] if values else 0


def _outcome(obj: dict[str, Any]) -> tuple[bool | None, float | None]:
    labels = []
    for key in ("is_resolved", "resolved", "is_correct"):
        value = obj.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = {"true": True, "false": False, "1": True, "0": False}.get(value.lower())
        if value is None or value not in (True, False):
            raise ValueError(f"{key} must be a boolean")
        labels.append(bool(value))
    if len(set(labels)) > 1:
        raise ValueError("conflicting outcome labels")
    reward = obj.get("reward")
    if reward is not None:
        try:
            reward = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward must be numeric") from exc
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
    correct = labels[0] if labels else (reward >= 1.0 if reward is not None else None)
    if reward is None and correct is not None:
        reward = float(correct)
    return correct, reward


def _from_parsed(
    obj: dict[str, Any],
    *,
    model_id: str,
    benchmark: str,
    line_no: int,
) -> dict[str, Any]:
    from autofa.diagnostics import get_diagnostics

    problem_id = _raw_problem_id(obj) or f"row-{line_no}"
    is_correct, reward = _outcome(obj)
    prompt_value = obj.get("question") or ""
    prompt = (prompt_value if isinstance(prompt_value, str)
              else json.dumps(prompt_value, ensure_ascii=False))
    steps = obj.get("steps")
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError("steps must be an array of objects")
    reasoning_parts: list[str] = []
    normalized: list[dict[str, Any]] = []
    n_observations = 0

    for step in steps:
        reasoning = step.get("reasoning_content") or step.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            reasoning_parts.append(reasoning.strip())
        norm = _normalize_gym_step(step)
        if "observation" in norm:
            n_observations += 1
        normalized.append(norm)

    if n_observations:
        get_diagnostics().info(
            "gym_observations_preserved",
            f"Preserved {n_observations} step observation(s) while converting gym trajectory",
            context={"n_observations": n_observations, "problem_id": problem_id},
        )

    envelope = {"steps": normalized}
    for key in ("agent", "schema_version"):
        if key in obj:
            envelope[key] = obj[key]
    response = json.dumps(envelope, ensure_ascii=False)

    return {
        "model_id": model_id,
        "benchmark": benchmark,
        "problem_id": problem_id,
        "run_idx": _run_idx(obj),
        "is_correct": is_correct,
        "prompt_text": prompt,
        "response_text": response,
        "reasoning_text": "\n\n".join(reasoning_parts),
        "reward": reward,
    }


def _from_raw(
    obj: dict[str, Any],
    *,
    model_id: str,
    benchmark: str,
    line_no: int,
) -> dict[str, Any]:
    is_correct, reward_f = _outcome(obj)

    problem_id = _raw_problem_id(obj) or f"row-{line_no}"
    prompt = _raw_prompt(obj)
    response = _raw_response(obj)
    reasoning = _raw_reasoning(obj)

    from autofa.data.structured import decode_container, input_view

    request = decode_container(obj.get("responses_create_params"))
    view = input_view(request, tools=request.get("tools")) if isinstance(request, dict) else None
    return {
        "model_id": model_id,
        "benchmark": benchmark,
        "problem_id": problem_id,
        "run_idx": _run_idx(obj),
        "is_correct": is_correct,
        "prompt_text": prompt,
        "response_text": response,
        "reasoning_text": reasoning,
        "reward": reward_f,
        "input_json": (json.dumps({"messages": view["messages"], "tools": view["tools"]},
                                  ensure_ascii=False) if view else None),
    }


def _raw_problem_id(obj: dict[str, Any]) -> str:
    for key in ("task_id", "problem_id", "trial_name", "instance_id"):
        if obj.get(key) is not None:
            return str(obj[key])
    params = obj.get("responses_create_params")
    if isinstance(params, dict):
        meta = params.get("metadata")
        if isinstance(meta, dict):
            for key in ("task_name", "instance_id", "task_id"):
                if meta.get(key):
                    return str(meta[key])
    cfg = obj.get("instance_config")
    if isinstance(cfg, dict):
        for key in ("task_name", "instance_id", "task_id"):
            if cfg.get(key):
                return str(cfg[key])
        info = cfg.get("problem_info")
        if isinstance(info, dict) and info.get("instance_id"):
            return str(info["instance_id"])
        body = cfg.get("body")
        if isinstance(body, dict):
            meta = body.get("metadata")
            if isinstance(meta, dict) and meta.get("task_name"):
                return str(meta["task_name"])
    return ""


def _raw_prompt(obj: dict[str, Any]) -> str:
    params = obj.get("responses_create_params")
    text = _extract_input_text(params)
    if text:
        return text
    cfg = obj.get("instance_config")
    if isinstance(cfg, dict):
        body = cfg.get("body")
        if isinstance(body, dict):
            text = _extract_input_text(body)
            if text:
                return text
        info = cfg.get("problem_info")
        if isinstance(info, str) and info.strip():
            return info.strip()
        if isinstance(info, dict):
            for key in ("question", "prompt", "instruction", "description"):
                if info.get(key):
                    return str(info[key])
    return ""


def _extract_input_text(container: Any) -> str:
    if not isinstance(container, dict):
        return ""
    inp = container.get("input")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        parts: list[str] = []
        for item in inp:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        t = block.get("text") or block.get("content")
                        if t:
                            parts.append(str(t))
            elif item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(p for p in parts if p).strip()
    return ""


def _raw_response(obj: dict[str, Any]) -> str:
    resp = obj.get("response")
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        out = resp.get("output")
        if out is not None:
            try:
                return json.dumps(out, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(out)
        try:
            return json.dumps(resp, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(resp)
    return json.dumps(resp, ensure_ascii=False)


def _raw_reasoning(obj: dict[str, Any]) -> str:
    from autofa.data.execution_policy import is_evaluation_record

    parts: list[str] = []
    capture = obj.get("ng_model_call_capture")
    if isinstance(capture, dict):
        calls = capture.get("calls")
        if isinstance(calls, list):
            for call in calls:
                # A grader's reasoning is about the outcome, not the rollout.
                if not isinstance(call, dict) or is_evaluation_record(call):
                    continue
                rc = call.get("reasoning_content")
                if isinstance(rc, str) and rc.strip():
                    parts.append(rc.strip())
    resp = obj.get("response")
    if isinstance(resp, dict):
        for item in resp.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                summary = item.get("summary") or item.get("content") or item.get("text")
                if isinstance(summary, str) and summary.strip():
                    parts.append(summary.strip())
                elif isinstance(summary, list):
                    for block in summary:
                        if isinstance(block, dict) and block.get("text"):
                            parts.append(str(block["text"]))
    return "\n\n".join(parts)
