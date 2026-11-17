# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy workflow section of the shared ``autofa run`` configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autofa.data.reference_data import parse_evalrdl_run_url
from autofa.run_config import _integer, _keys, _mapping, _text


@dataclass(frozen=True)
class TaxonomySourceSpec:
    api_runs: tuple[str, ...]
    evalrdl_url: str | None
    local_folder: Path | None
    api_base_url: str | None


@dataclass(frozen=True)
class SelectionSpec:
    per_run: int
    sample: str
    seed: int
    cache: Path


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    model: str | None
    harness: str | None
    params: tuple[str, ...]
    workers: int
    output: Path


@dataclass(frozen=True)
class ReportSpec:
    output: Path
    publish: str | None


@dataclass(frozen=True)
class ClassificationRunSpec:
    first_pass: str
    second_pass: str
    judges: tuple[JudgeSpec, ...]
    report: ReportSpec


@dataclass(frozen=True)
class TaxonomyWorkflowSpec:
    judge_base_url: str | None
    judge_api_key_env: str | None
    source: TaxonomySourceSpec
    selection: SelectionSpec
    runs: tuple[ClassificationRunSpec, ...]


def _required_text(mapping: dict[str, Any], key: str, prefix: str) -> str:
    return _text(mapping.get(key), f"{prefix}.{key}")


def _optional_text(mapping: dict[str, Any], key: str, prefix: str) -> str | None:
    value = mapping.get(key)
    return None if value is None else _text(value, f"{prefix}.{key}")


def _positive(mapping: dict[str, Any], key: str, prefix: str) -> int:
    value = _integer(mapping.get(key), f"{prefix}.{key}")
    if value < 1:
        raise ValueError(f"{prefix}.{key} must be a positive integer")
    return value


def _path(mapping: dict[str, Any], key: str, prefix: str, base: Path) -> Path:
    value = Path(_required_text(mapping, key, prefix)).expanduser()
    return (base / value).resolve() if not value.is_absolute() else value.resolve()


def parse_taxonomy_workflow(root: dict[str, Any], base: Path) -> TaxonomyWorkflowSpec:
    """Parse ``workflow: classification`` from a shared version-2 run config."""
    source = _mapping(root.get("source"), "source")
    _keys(source, {"api_runs", "evalrdl_url", "local_folder"}, "source")
    raw_api_runs = source.get("api_runs")
    api_runs: tuple[str, ...] = ()
    if raw_api_runs is not None:
        if not isinstance(raw_api_runs, list) or not raw_api_runs or not all(
            isinstance(run_id, str) and run_id.strip() for run_id in raw_api_runs
        ):
            raise ValueError("source.api_runs must contain run IDs")
        api_runs = tuple(run_id.strip() for run_id in raw_api_runs)
    evalrdl_url = source.get("evalrdl_url")
    if evalrdl_url is not None:
        if not isinstance(evalrdl_url, str) or not evalrdl_url.strip():
            raise ValueError("source.evalrdl_url must be a nonblank string")
        evalrdl_url = evalrdl_url.strip()
    local_folder = source.get("local_folder")
    if local_folder is not None:
        if not isinstance(local_folder, str) or not local_folder.strip():
            raise ValueError("source.local_folder must be a nonblank string")
        folder = Path(local_folder).expanduser()
        local_folder = (base / folder).resolve() if not folder.is_absolute() else folder.resolve()
    if sum((bool(api_runs), evalrdl_url is not None, local_folder is not None)) != 1:
        raise ValueError(
            "source must define exactly one of api_runs, evalrdl_url, or local_folder"
        )
    api_base_url = None
    if evalrdl_url is not None:
        api_base_url, run_id, _export_url = parse_evalrdl_run_url(evalrdl_url)
        api_runs = (run_id,)
    source_spec = TaxonomySourceSpec(
        api_runs=api_runs,
        evalrdl_url=evalrdl_url,
        local_folder=local_folder,
        api_base_url=api_base_url,
    )

    selection = _mapping(root.get("selection"), "selection")
    _keys(selection, {"per_run", "sample", "seed", "cache"}, "selection")
    sample = _required_text(selection, "sample", "selection")
    if sample not in {"first", "random"}:
        raise ValueError("selection.sample must be first or random")
    seed = selection.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("selection.seed must be a nonnegative integer")
    selection_spec = SelectionSpec(
        per_run=_positive(selection, "per_run", "selection"),
        sample=sample,
        seed=seed,
        cache=_path(selection, "cache", "selection", base),
    )

    taxonomy = _mapping(root.get("taxonomy"), "taxonomy")
    _keys(
        taxonomy,
        {"judge_base_url", "judge_api_key_env", "runs"},
        "taxonomy",
    )
    raw_runs = taxonomy.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("taxonomy.runs must contain at least one run")
    runs = []
    for run_index, raw_run in enumerate(raw_runs, 1):
        run_prefix = f"taxonomy.runs[{run_index}]"
        configured = _mapping(raw_run, run_prefix)
        _keys(
            configured,
            {"first_pass", "second_pass", "judges", "report"},
            run_prefix,
        )
        first_pass = _required_text(configured, "first_pass", run_prefix)
        second_pass = _required_text(configured, "second_pass", run_prefix)
        if first_pass not in {"llm", "agentic"}:
            raise ValueError(f"{run_prefix}.first_pass must be llm or agentic")
        if second_pass not in {"off", "llm", "agentic"}:
            raise ValueError(f"{run_prefix}.second_pass must be off, llm or agentic")
        raw_judges = configured.get("judges")
        if not isinstance(raw_judges, list) or not raw_judges:
            raise ValueError(f"{run_prefix}.judges must contain at least one judge")
        judges = []
        for judge_index, raw_judge in enumerate(raw_judges, 1):
            prefix = f"{run_prefix}.judges[{judge_index}]"
            judge = _mapping(raw_judge, prefix)
            _keys(
                judge,
                {"name", "model", "harness", "params", "workers", "output"},
                prefix,
            )
            harness = judge.get("harness")
            if harness is not None and (
                not isinstance(harness, str) or harness not in {"claude", "codex"}
            ):
                raise ValueError(f"{prefix}.harness must be claude or codex")
            if harness is not None and "agentic" not in {first_pass, second_pass}:
                raise ValueError(f"{prefix}.harness applies to agentic passes only")
            params = judge.get("params", [])
            if not isinstance(params, list) or not all(
                isinstance(value, str)
                and bool(value.partition("=")[1])
                and bool(value.partition("=")[0].strip())
                for value in params
            ):
                raise ValueError(f"{prefix}.params must contain KEY=VALUE strings")
            if params and "llm" not in {first_pass, second_pass}:
                raise ValueError(f"{prefix}.params apply to llm passes only")
            model = _optional_text(judge, "model", prefix)
            if model is not None and "llm" not in {first_pass, second_pass}:
                raise ValueError(f"{prefix}.model applies to llm passes only")
            judges.append(JudgeSpec(
                name=_required_text(judge, "name", prefix),
                model=model,
                harness=harness,
                params=tuple(params),
                workers=_positive(judge, "workers", prefix),
                output=_path(judge, "output", prefix, base),
            ))
        raw_report = _mapping(configured.get("report"), f"{run_prefix}.report")
        _keys(raw_report, {"output", "publish"}, f"{run_prefix}.report")
        publish = raw_report.get("publish")
        if publish is not None and (
            not isinstance(publish, str) or not publish.strip()
        ):
            raise ValueError(f"{run_prefix}.report.publish must be a nonblank string")
        runs.append(ClassificationRunSpec(
            first_pass=first_pass,
            second_pass=second_pass,
            judges=tuple(judges),
            report=ReportSpec(
                output=_path(raw_report, "output", f"{run_prefix}.report", base),
                publish=publish.strip() if isinstance(publish, str) else None,
            ),
        ))

    outputs = [judge.output for run in runs for judge in run.judges]
    outputs.extend(run.report.output for run in runs)
    if len(outputs) != len(set(outputs)):
        raise ValueError("taxonomy judge and report output paths must be unique")

    return TaxonomyWorkflowSpec(
        judge_base_url=(
            value.rstrip("/")
            if (value := _optional_text(taxonomy, "judge_base_url", "taxonomy"))
            else None
        ),
        judge_api_key_env=_optional_text(taxonomy, "judge_api_key_env", "taxonomy"),
        source=source_spec,
        selection=selection_spec,
        runs=tuple(runs),
    )
