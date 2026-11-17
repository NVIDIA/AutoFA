# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest for taxonomy classification results."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from autofa.data.cohort import Cohort
from autofa.util.atomic import atomic_write_json
from evalrdl_schema import CURRENT

from .classifier import PROTOCOL_VERSION, TaxonomyClassifier
from .providers import effective_reasoning_effort, provider_configuration_sha256
from .record import read_classifications

EXPORTER = "autofa-taxonomy"
EXPORTER_VERSION = 0
log = logging.getLogger(__name__)


def manifest_path(results: Path | str) -> Path:
    """Return the manifest path beside a results file."""
    path = Path(results)
    return path.with_name(f"{path.stem}.manifest.json")


def write_manifest(
    results: Path | str,
    *,
    cohort: Cohort | None,
    classifier: TaxonomyClassifier,
    selection: Mapping[str, Any],
    workers: int,
    parent_results: Sequence[Path | str] = (),
    workflow_config: Mapping[str, Any] | None = None,
) -> Path:
    """Write provenance and configuration for a classification result."""
    results = Path(results)
    records = read_classifications(results)
    provider = classifier.provider
    agentic = classifier.agentic_provider
    primary_method = (
        classifier.second_pass
        if classifier.saved_first_pass is not None
        else classifier.first_pass
    )
    judge = _judge(primary_method, provider=provider, agentic=agentic)
    destination = manifest_path(results)
    previous = _read_manifest(destination)
    parents = _parent_manifests(parent_results)
    sources = {source["run_id"]: source for source in cohort.sources} if cohort else {}
    references: list[dict[str, Any]] = []
    inherited_rollouts: list[dict[str, Any]] = []
    inherited_runs: list[dict[str, Any]] = []
    for parent in parents:
        references = _merge_references(references, parent.get("reference_data") or [])
        inherited_rollouts = _merge_rollouts(
            inherited_rollouts, parent.get("rollouts") or []
        )
        inherited_runs = _merge_runs(inherited_runs, parent.get("runs") or [])
    references = _merge_references(
        references,
        list(cohort.datasets) if cohort else [],
    )
    rollouts = _rollouts(records, cohort)
    runs = _runs(records, sources)
    if previous:
        inherited_rollouts = _merge_rollouts(
            inherited_rollouts, previous.get("rollouts") or []
        )
        inherited_runs = _merge_runs(inherited_runs, previous.get("runs") or [])
        references = _merge_references(previous.get("reference_data"), references)
        rollouts = _merge_rollouts(previous.get("rollouts"), rollouts)
        runs = _merge_runs(previous.get("runs"), runs)
    rollouts = _merge_rollouts(inherited_rollouts, rollouts)
    runs = _merge_runs(inherited_runs, runs)
    selected = {**(previous.get("selection") or {}), **selection}
    if parents:
        selected["parent_results"] = [
            {
                "file": str(parent.get("results", {}).get("file") or ""),
                "sha256": str(parent["results"]["sha256"]),
            }
            for parent in parents
        ]
    if cohort is not None:
        if "cohort_id" not in selection:
            selected["cohort_id"] = cohort.id
    document = {
        "exporter": EXPORTER,
        "exporter_version": EXPORTER_VERSION,
        "schema": CURRENT.version,
        "runs": runs,
        "rollouts": rollouts,
        "reference_data": references,
        "taxonomy": {
            "id": classifier.taxonomy.taxonomy_id,
            "name": classifier.taxonomy.name,
            "version": classifier.taxonomy.version,
            "sha256": classifier.taxonomy.content_sha256,
            "protocol_version": PROTOCOL_VERSION,
            "first_pass": classifier.first_pass,
            "second_pass": classifier.second_pass,
            **(
                {
                    "second_pass_judge": _judge(
                        classifier.second_pass,
                        provider=provider,
                        agentic=agentic,
                    )
                }
                if classifier.second_pass != "off"
                else {}
            ),
        },
        "judge": judge,
        "selection": selected,
        "results": {
            "file": results.name,
            "sha256": _file_sha256(results),
            "records": len(records),
            "status": dict(Counter(str(record.get("status") or "unknown") for record in records)),
            "workers": workers,
        },
        **({"workflow_config": dict(workflow_config)} if workflow_config else {}),
    }
    atomic_write_json(destination, document)
    return destination


def _judge(method: str, *, provider: Any, agentic: Any) -> dict[str, Any]:
    return _agentic_judge(agentic) if method == "agentic" else _chat_judge(provider)


def _agentic_judge(agentic: Any) -> dict[str, Any]:
    return {
        "provider": "agentic",
        "model": getattr(agentic, "model", None),
        "harness": getattr(agentic, "harness", None),
        "reasoning_effort": getattr(agentic, "effort", None),
        "configured_max_turns": getattr(agentic, "max_turns", None),
        "configuration_sha256": (
            agentic.configuration_sha256() if agentic is not None else None
        ),
    }


def _chat_judge(provider: Any) -> dict[str, Any]:
    return {
        "provider": "chat",
        "model": getattr(provider, "model", None),
        "base_url": getattr(provider, "endpoint", None),
        "api_key_env": getattr(provider, "api_key_env", None),
        "api": getattr(provider, "api", None),
        "timeout_seconds": getattr(provider, "timeout_seconds", None),
        "max_retries": getattr(provider, "max_retries", None),
        "backoff_seconds": getattr(provider, "backoff_seconds", None),
        "reasoning_effort": effective_reasoning_effort(provider),
        # Providers take output limits from request_options; keep the key for stable manifests.
        "max_tokens": None,
        "configuration_sha256": provider_configuration_sha256(provider),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict) or document.get("exporter") != EXPORTER:
        return {}
    return document


def _parent_manifests(results_paths: Sequence[Path | str]) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in results_paths:
        results = Path(value)
        if not results.is_file():
            log.warning(
                "cannot inherit taxonomy provenance: parent results are missing: %s",
                results,
            )
            continue
        document = _read_manifest(manifest_path(results))
        if not document:
            log.warning(
                "cannot inherit taxonomy provenance: parent manifest is missing or invalid: %s",
                manifest_path(results),
            )
            continue
        result_info = document.get("results")
        expected = result_info.get("sha256") if isinstance(result_info, Mapping) else None
        if not isinstance(expected, str) or _file_sha256(results) != expected:
            log.warning(
                "cannot inherit taxonomy provenance: parent manifest does not match %s",
                results,
            )
            continue
        if expected in seen:
            continue
        parents.append(document)
        seen.add(expected)
    return parents


def _merge_rollouts(previous: Any, current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[str, str, int] | None:
        try:
            return str(row["run_id"]), str(row["task_id"]), int(row["repeat"])
        except (KeyError, TypeError, ValueError):
            return None

    return _merge_records(previous, current, key)


def _merge_runs(previous: Any, current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _merge_records(
        previous,
        current,
        lambda row: str(row["run_id"]) if row.get("run_id") is not None else None,
    )


def _merge_references(previous: Any, current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*(previous if isinstance(previous, list) else []), *current]:
        if not isinstance(item, Mapping):
            continue
        reference = dict(item)
        identity = json.dumps(reference, ensure_ascii=False, sort_keys=True, default=str)
        if identity not in seen:
            references.append(reference)
            seen.add(identity)
    return references


def _merge_records(
    previous: Any,
    current: list[dict[str, Any]],
    key: Callable[[Mapping[str, Any]], Any],
) -> list[dict[str, Any]]:
    merged: dict[Any, dict[str, Any]] = {}
    for item in [*(previous if isinstance(previous, list) else []), *current]:
        if not isinstance(item, Mapping) or (identity := key(item)) is None:
            continue
        row = merged.setdefault(identity, {})
        row.update({name: value for name, value in item.items() if value is not None})
    return list(merged.values())


def populations(document: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Read run population totals from a classification manifest."""
    result: dict[str, dict[str, int]] = {}
    for run in document.get("runs") or []:
        if not isinstance(run, Mapping) or not isinstance(run.get("run_id"), str):
            continue
        population = run.get("population")
        if not isinstance(population, Mapping):
            continue
        keys = ("rows", "scored", "zero_reward", "positive_reward", "unscored")
        values = {key: population.get(key) for key in keys}
        if all(type(value) is int and value >= 0 for value in values.values()):
            result[run["run_id"]] = values
    return result


def _rollouts(records: list[dict[str, Any]], cohort: Cohort | None) -> list[dict[str, Any]]:
    selected = {
        (row["run_id"], row["task_id"], int(row["repeat"])): row
        for row in (cohort.rows if cohort else [])
    }
    for record in records:
        source = record.get("source") or {}
        key = (
            source.get("run_id"),
            source.get("task_id"),
            source.get("repeat"),
        )
        if all(value is not None for value in key):
            selected[key] = {**selected.get(key, {}), **source}
    return [
        {
            "run_id": str(run_id),
            "task_id": str(task_id),
            "repeat": int(repeat),
            "agent_ref": row.get("agent_ref"),
            "lane": str(row.get("lane") or ""),
        }
        for (run_id, task_id, repeat), row in selected.items()
    ]


def _runs(
    records: list[dict[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        source = record.get("source") or {}
        run_id = str(source.get("run_id") or "")
        if run_id:
            metadata.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "benchmark": source.get("benchmark"),
                    "model": source.get("model"),
                },
            )
    for run_id, source in sources.items():
        run = metadata.setdefault(run_id, {"run_id": run_id})
        population = source.get("population")
        if isinstance(population, Mapping):
            run["population"] = dict(population)
        for field in (
            "kind",
            "base_url",
            "store_id",
            "dataset_contract",
            "metadata_watermark",
        ):
            if source.get(field) is not None:
                run[field] = source[field]
    return list(metadata.values())


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
