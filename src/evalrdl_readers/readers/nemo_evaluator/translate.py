# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from evalrdl_readers.core import Answer, Sources

from ..common import dig, yaml_doc


def ne_run_config(src: Sources):
    if not src.has(0):
        return
    body = yaml_doc(src.doc(0))
    yield Answer("run.producer", body.get("framework_name"))
    task = dig(body, "config", "params", "task")
    if task is not None:
        yield Answer("run.benchmark", task)
    model = dig(body, "target", "api_endpoint", "model_id")
    if model:
        yield Answer("run.model", model)


def ne_results_yml(src: Sources):
    """Two shapes: the small NEL one flat, the launcher one nesting `config` and `target`."""
    if not src.has(0):
        return
    body = yaml_doc(src.doc(0))
    yield Answer("run.benchmark", body.get("benchmark"))
    model = dig(body, "target", "api_endpoint", "model_id") or body.get("model")
    if model:
        yield Answer("run.model", model)
    if body.get("num_rollouts") is not None:
        yield Answer("run.n_expected", body.get("num_rollouts"))
    yield Answer("run.adapter_version", dig(body, "metadata", "versioning", "nemo_evaluator"))


def ne_metadata(src: Sources):
    if not src.has(0):
        return
    body = yaml_doc(src.doc(0))
    cluster = dig(body, "launcher_resolved_config", "cluster", "name")
    if cluster:
        yield Answer("run.cluster", cluster)
    user = dig(body, "launcher_resolved_config", "execution", "username")
    if user:
        yield Answer("run.source_user", user)
    out = _output_dir(body)
    yield Answer("run.output_dir_name", out.rstrip("/").rsplit("/", 1)[-1] if out else None)


def _output_dir(node, depth: int = 0):
    if depth > 8:
        return None
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "output_dir" and isinstance(v, str) and v.startswith("/lustre"):
                return v
            found = _output_dir(v, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _output_dir(v, depth + 1)
            if found:
                return found
    return None


def ne_export_config(src: Sources):
    """The launcher's mlflow tags. The benchmark tag is kept verbatim: the registry owns every
    spelling."""
    if not src.has(0):
        return
    tags = dig(yaml_doc(src.doc(0)), "export", "mlflow", "tags", default={}) or {}
    yield Answer("run.benchmark", tags.get("benchmark"))
    yield Answer("run.model", tags.get("model"))
    yield Answer("run.cluster", tags.get("cluster"))


def ne_eval_metadata(src: Sources):
    if not src.has(0):
        return
    body = src.doc(0).json() or {}
    if body.get("date"):
        yield Answer("run.started_at", body.get("date"))
    yield Answer("run.suite", body.get("benchmark"))
    yield Answer("run.agent_harness", dig(body, "harness", "name"))
    if body.get("model"):
        yield Answer("run.model", body.get("model"))


def ne_eval_factory_metrics(src: Sources):
    if not src.has(0):
        return
    body = src.doc(0).json() or {}
    raw = dig(body, "evaluation", "start_time")
    runs = dig(body, "response_stats", "inference_run_times", default={}) or {}
    first = runs.get("0") if isinstance(runs, dict) else None
    good = first.get("run_start") if isinstance(first, dict) else None
    if isinstance(raw, str) and "%f" in raw and good:
        yield Answer(
            "run.started_at",
            good,
            note=(
                f"evaluation.start_time is malformed ({raw!r} carries a literal %f), so "
                f"response_stats.inference_run_times.0.run_start is used instead"
            ),
        )
        return
    if isinstance(raw, str) and "%f" in raw:
        raw = None
    yield Answer("run.started_at", good or raw)
