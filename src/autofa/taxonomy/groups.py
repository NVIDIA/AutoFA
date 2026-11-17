# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build saved rollout selections from taxonomy classifications."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from autofa.data.cohort import Cohort, load_cohort
from autofa.data.outcomes import is_discovery_success
from autofa.taxonomy.manifest import manifest_path
from autofa.taxonomy.record import has_usable_first_pass, read_classifications
from autofa.taxonomy.report import (
    aggregate_classification_rows,
    classification_judge_views,
    latest_classification_rows,
)

GROUPS = ("all", "accepted", "review", "no-supported", "judge-disagreement")
EVIDENCE_LEVELS = ("complete", "partial", "insufficient")
_EVIDENCE_RANK = {"insufficient": 0, "partial": 1, "complete": 2}


def _is_filterable_result(
    row: dict[str, Any],
    *,
    include_review_failed: bool,
) -> bool:
    """Keep failed reviews retryable without exposing them to normal handoff."""
    if not has_usable_first_pass(row):
        return False
    return row.get("status") == "ok" or (
        include_review_failed and row.get("status") == "review_failed"
    )


def resolve_classification_paths(paths: list[Path]) -> list[Path]:
    """Resolve result files from explicit JSONL paths or result folders."""
    resolved: list[Path] = []
    for value in paths:
        path = value.expanduser().resolve()
        if path.is_file():
            resolved.append(path)
            continue
        if path.is_dir():
            resolved.extend(
                candidate
                for candidate in sorted(path.rglob("*.jsonl"))
                if manifest_path(candidate).is_file()
            )
            continue
        raise ValueError(f"Taxonomy results do not exist: {path}")
    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        raise ValueError("No taxonomy classification results were found")
    return resolved


def classification_facets(
    paths: list[Path],
    *,
    include_review_failed: bool = False,
) -> dict[str, Any]:
    """Return filter choices available in taxonomy classification results."""
    paths = resolve_classification_paths(paths)
    for path in paths:
        _read_manifest(path)
    rows = latest_classification_rows([
        row
        for path in paths
        for row in read_classifications(path)
    ])
    rows = [
        row
        for row in rows
        if _is_filterable_result(
            row,
            include_review_failed=include_review_failed,
        )
    ]
    if not rows:
        raise ValueError("Taxonomy results contain no successful classifications")
    labels: dict[str, str] = {}
    for row in rows:
        code = str(row.get("selected_code") or row.get("selected_leaf_id") or "")
        if code:
            labels.setdefault(code, str(row.get("selected_label") or code))
    return {
        "paths": paths,
        "judges": list(classification_judge_views(
            rows,
            include_review_failed=include_review_failed,
        )),
        "labels": labels,
        "evidence": sorted({
            str((row.get("diagnosis") or {}).get("evidence_support") or "")
            for row in rows
        } & set(EVIDENCE_LEVELS), key=_EVIDENCE_RANK.__getitem__),
    }


def classification_rollout_identities(
    paths: list[Path],
    *,
    judge: str | None = None,
    group: str = "all",
    labels: list[str] | None = None,
    evidence: list[str] | None = None,
    include_review_failed: bool = False,
) -> set[tuple[str, str, int]]:
    """Return filtered rollout identities without requiring a saved local cohort."""
    paths = resolve_classification_paths(paths)
    if group not in GROUPS:
        raise ValueError(f"Unknown taxonomy group: {group}")
    if judge and group == "judge-disagreement":
        raise ValueError("Judge disagreement requires the consensus view")
    rows = latest_classification_rows([
        row
        for path in paths
        for row in read_classifications(path)
    ])
    rows = [
        row
        for row in rows
        if _is_filterable_result(
            row,
            include_review_failed=include_review_failed,
        )
    ]
    if not rows:
        raise ValueError("Taxonomy results contain no successful classifications")
    compatibility = {
        (str(row.get("taxonomy_sha256") or ""), str(row.get("protocol_version") or ""))
        for row in rows
    }
    if len(compatibility) != 1:
        raise ValueError("Taxonomy results use incompatible taxonomy or protocol versions")
    available = classification_judge_views(
        rows,
        include_review_failed=include_review_failed,
    )
    if judge and judge not in available:
        raise ValueError(
            f"Judge {judge!r} was not found. Available judges: {', '.join(available)}"
        )
    facets = classification_facets(
        paths,
        include_review_failed=include_review_failed,
    )
    label_set = set(_resolve_labels(labels or [], facets["labels"]))
    evidence_set = set(evidence or [])
    return {
        (str(view["run_id"]), str(view["task_id"]), int(view["repeat"]))
        for view in aggregate_classification_rows(
            rows,
            judge_view=judge,
            include_review_failed=include_review_failed,
        )
        if _matches(
            view,
            group=group,
            labels=label_set,
            evidence=evidence_set,
        )
    }


def saved_selection_id(paths: list[Path]) -> str | None:
    """Return the one saved selection the results were classified from, if any."""
    selection_ids = {
        str((_read_manifest(path).get("selection") or {}).get("cohort_id") or "")
        for path in resolve_classification_paths(paths)
    }
    selection_ids.discard("")
    return selection_ids.pop() if len(selection_ids) == 1 else None


def cohort_from_classifications(
    paths: list[Path],
    *,
    judge: str | None = None,
    group: str = "all",
    labels: list[str] | None = None,
    evidence: list[str] | None = None,
    include_review_failed: bool = False,
) -> Cohort:
    """Select classified rollouts and preserve their original source selection."""
    if not paths:
        raise ValueError("At least one taxonomy results file is required")
    paths = resolve_classification_paths(paths)
    if group not in GROUPS:
        raise ValueError(f"Unknown taxonomy group: {group}")
    if judge and group == "judge-disagreement":
        raise ValueError("Judge disagreement requires the consensus view")

    selection_id = saved_selection_id(paths)
    if selection_id is None:
        raise ValueError("Taxonomy results must reference one saved selection")
    parent = load_cohort(selection_id)

    facets = classification_facets(
        paths,
        include_review_failed=include_review_failed,
    )
    label_set = set(_resolve_labels(labels or [], facets["labels"]))
    evidence_set = set(evidence or [])
    selected = classification_rollout_identities(
        paths,
        judge=judge,
        group=group,
        labels=labels,
        evidence=evidence,
        include_review_failed=include_review_failed,
    )
    base_rows = {
        (str(row["run_id"]), str(row["task_id"]), int(row["repeat"])): row
        for row in parent.rows
    }
    missing = selected - set(base_rows)
    if missing:
        raise ValueError("Taxonomy results include rollouts outside their saved selection")
    selected_rows = [
        row
        for row in parent.rows
        if (str(row["run_id"]), str(row["task_id"]), int(row["repeat"])) in selected
    ]
    selected_runs = {str(row["run_id"]) for row in selected_rows}
    taxonomy_filter = {
        "results": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in paths
        ],
        "judge": judge or "consensus",
        "group": group,
        "labels": sorted(label_set),
        "evidence": sorted(evidence_set),
    }
    return Cohort(
        runs_where=parent.runs_where,
        where=parent.where,
        sources=[
            dict(source)
            for source in parent.sources
            if str(source["run_id"]) in selected_runs
        ],
        rows=selected_rows,
        latest=parent.latest,
        per_run=0,
        sample="first",
        seed=0,
        created_at=datetime.now(UTC).isoformat(),
        parent_selection=parent.id,
        taxonomy_filter=taxonomy_filter,
    )


def add_success_controls(
    cohort: Cohort,
    *,
    api: Any = None,
    activity: Callable[[str], None] | None = None,
) -> Cohort:
    """Add up to one same-source successful control per selected failure."""
    if not cohort.taxonomy_filter:
        return cohort

    from autofa.data.cohort import select_cohort
    from autofa.data.local_trace_store import resolve_store
    from autofa.data.remote_api import DataApiClient

    rows = [dict(row) for row in cohort.rows]
    previous_controls = cohort.taxonomy_filter.get("controls") or {}
    recorded_controls = {
        (
            str(item["run_id"]),
            str(item["task_id"]),
            int(item["repeat"]),
        )
        for item in previous_controls.get("rollouts") or []
        if isinstance(item, dict)
        and item.get("run_id") is not None
        and item.get("task_id") is not None
        and item.get("repeat") is not None
    }
    control_identities = set(recorded_controls)
    for source in cohort.sources:
        run_id = str(source["run_id"])
        run_rows = [row for row in rows if str(row["run_id"]) == run_id]
        successes = [
            row
            for row in run_rows
            if (
                str(row["run_id"]),
                str(row["task_id"]),
                int(row["repeat"]),
            )
            in recorded_controls
        ]
        failures = [row for row in run_rows if row not in successes]
        needed = len(failures) - len(successes)
        if needed <= 0:
            control_identities.update(
                (
                    str(row["run_id"]),
                    str(row["task_id"]),
                    int(row["repeat"]),
                )
                for row in successes
            )
            continue
        if activity:
            activity(
                f"Taxonomy selection for run {run_id} has "
                f"{len(successes)} successful control(s); "
                f"selecting up to {needed} from the original run."
            )
        if source["kind"] == "remote":
            source_api = api
            if source_api is None or source_api.base_url != source["base_url"]:
                source_api = DataApiClient(source["base_url"])
            controls = select_cohort(
                api_runs=[run_id],
                rewards=[">=0.5"],
                per_run=needed,
                sample="random",
                seed=cohort.seed,
                api=source_api,
            )
        else:
            controls = select_cohort(
                folders=[resolve_store(source["store_id"]) / "dataset.parquet"],
                rewards=[">=0.5"],
                per_run=needed,
                sample="random",
                seed=cohort.seed,
            )
        existing = {
            (str(row["run_id"]), str(row["task_id"]), int(row["repeat"]))
            for row in rows
        }
        added = 0
        for control in controls.rows:
            control = dict(control)
            control["run_id"] = run_id
            identity = (run_id, str(control["task_id"]), int(control["repeat"]))
            if identity not in existing and is_discovery_success(control):
                rows.append(control)
                existing.add(identity)
                control_identities.add(identity)
                added += 1
        control_identities.update(
            (
                str(row["run_id"]),
                str(row["task_id"]),
                int(row["repeat"]),
            )
            for row in successes
        )
        available = len(successes) + added
        if not available:
            raise ValueError(
                f"Taxonomy group discovery requires successful controls from run {run_id}, "
                "but none were found in the original run"
            )
        if available < len(failures) and activity:
            activity(
                f"Run {run_id} has only {available} successful control(s) for "
                f"{len(failures)} taxonomy failure(s); continuing with all available controls."
            )

    taxonomy_filter = dict(cohort.taxonomy_filter)
    taxonomy_filter["controls"] = {
        "outcome": "discovery pass (reward >= 0.5 in canonical EvalRDL data)",
        "strategy": "same-run",
        "max_per_failure": 1,
        "count": len(control_identities),
        "rollouts": [
            {"run_id": run_id, "task_id": task_id, "repeat": repeat}
            for run_id, task_id, repeat in sorted(control_identities)
        ],
    }
    return Cohort(
        runs_where=cohort.runs_where,
        where=cohort.where,
        sources=[dict(source) for source in cohort.sources],
        rows=rows,
        latest=cohort.latest,
        per_run=cohort.per_run,
        sample=cohort.sample,
        seed=cohort.seed,
        created_at=datetime.now(UTC).isoformat(),
        parent_selection=cohort.parent_selection,
        taxonomy_filter=taxonomy_filter,
        limit=cohort.limit,
    )


def _resolve_labels(values: list[str], labels: dict[str, str]) -> list[str]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for code, label in labels.items():
        by_label[label.casefold()].append(code)
    resolved = []
    for value in values:
        if value in labels:
            resolved.append(value)
            continue
        matches = by_label.get(value.casefold(), [])
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Taxonomy label {value!r} is ambiguous; use one of: {', '.join(matches)}"
            )
        choices = [*labels, *labels.values()]
        suggestions = get_close_matches(value, choices, n=3, cutoff=0.5)
        suffix = f". Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown taxonomy label or code {value!r}{suffix}")
    return list(dict.fromkeys(resolved))


def _read_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Taxonomy results do not exist: {path}")
    sidecar = manifest_path(path)
    if not sidecar.is_file():
        raise ValueError(f"Taxonomy manifest does not exist: {sidecar}")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read taxonomy manifest {sidecar}: {exc}") from exc
    expected = str((manifest.get("results") or {}).get("sha256") or "")
    if not expected or expected != _sha256(path):
        raise ValueError(f"Taxonomy results do not match their manifest: {path}")
    return manifest


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _matches(
    view: dict[str, Any],
    *,
    group: str,
    labels: set[str],
    evidence: set[str],
) -> bool:
    selection = view["selected_code"] or view.get("selected_leaf_id")
    if labels and selection not in labels:
        return False
    if evidence and view["evidence_support"] not in evidence:
        return False
    if group == "accepted":
        return view["decision"] == "accept"
    if group == "review":
        return view["decision"] == "review"
    if group == "no-supported":
        return view["selected_code"] == "no_supported_failure"
    if group == "judge-disagreement":
        return len({
            str(judgment["selected_code"] or judgment.get("selected_leaf_id") or "")
            for judgment in view["judgments"]
        }) > 1
    return True
