# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed detector breadth across source-identified failed tasks.

The denominator is always the selected corpus, never the evaluated sample. A
sample therefore gives a lower bound, without extrapolating row sample weights
to tasks. Unknown failed-task identities make that denominator unknowable: keep
the known match counts, but leave the ranking metric unavailable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from autofa.core.dependencies import review_source
from autofa.core.localisation import text_hash
from autofa.metrics.association import matches_selected_bucket, selected_bucket
from autofa.metrics.claim_contract import claim_hash


def _field(function: Any, name: str, default: Any = None) -> Any:
    return function.get(name, default) if isinstance(function, Mapping) else getattr(
        function, name, default,
    )


def _count(value: Any) -> int | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        return None
    return int(value)


def current_task_recall_scores(
    scores: Mapping[str, Any] | None, *, source: str, description: str, mechanism: str,
) -> dict[str, Any]:
    """Hide unavailable or stale task scores in source-bearing read views.

    Preserve historical association scores and do not mutate the saved record.
    Missing task identity can make this metric unavailable without making an
    otherwise current attempt-level evaluation stale.
    """
    result = dict(scores or {})
    if "task_recall" not in result and "task_recall_scope" not in result:
        return result
    fingerprint = claim_hash(source=source, description=description, mechanism=mechanism)
    hashes = (result.get("task_recall_claim_hash"), result.get("evaluation_claim_hash"))
    changed = bool(result.get("evaluation_stale") or any(
        saved is not None and saved != fingerprint for saved in hashes
    ))
    if changed or any(saved != fingerprint for saved in hashes):
        result.update({
            "task_recall": None, "task_recall_scope": "unavailable",
            "task_recall_reason": (
                "Task coverage has no current detector/claim evaluation provenance."
            ),
        })
        if changed:
            result["evaluation_stale"] = True
    elif result.get("task_recall_scope") not in {"full_selected_corpus", "observed_lower_bound"}:
        result["task_recall"] = None
        result["task_recall_scope"] = "unavailable"
    if isinstance(result.get("feedback"), Mapping):
        result["feedback"] = {**result["feedback"], **{
            key: value for key, value in result.items() if key.startswith("task_recall")
        }}
    return result


def build_task_recall(
    fn: Any, inventory: Mapping[str, Any], *, run_id: str,
) -> dict[str, Any]:
    """Measure only the fresh evaluation localisations supplied by the caller.

    Call immediately after evaluation, before later judge probes append evidence.
    The metric is an observed match rate, not causal confirmation. Precision and
    behavioral review remain separate requirements for ranking and progress.
    No dataset rows, traces, or model calls are needed here.
    """
    source = review_source(fn)
    fingerprint = claim_hash(
        source=source, description=_field(fn, "description", "") or "",
        mechanism=_field(fn, "mechanism", "") or _field(fn, "description", "")
        or _field(fn, "name", ""),
    )
    scores = _field(fn, "scores", {}) or {}
    failed = inventory.get("task_by_failed_item_id")
    failed = failed if isinstance(failed, Mapping) else {}
    known_tasks = {task for task in failed.values() if isinstance(task, str) and task.strip()}
    unknown_tasks = sum(not isinstance(task, str) or not task.strip() for task in failed.values())
    result = {
        "task_recall": None,
        "task_recall_scope": "unavailable",
        "n_matched_failed_tasks": 0,
        "n_failed_tasks": len(known_tasks),
        "n_matched_failed_attempts": 0,
        "n_matched_failed_attempts_without_task_identity": 0,
        "task_recall_corpus_fingerprint": inventory.get("corpus_fingerprint"),
        "task_recall_claim_hash": fingerprint,
    }

    def unavailable(reason: str) -> dict[str, Any]:
        return {**result, "task_recall_reason": reason}

    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or scores.get("evaluation_run_id") != run_id
            or scores.get("evaluation_claim_hash") != fingerprint):
        return unavailable("The current detector and claim have no current-run evaluation.")
    if (inventory.get("inventory_complete") is not True
            or not inventory.get("corpus_fingerprint")
            or _count(inventory.get("n_failed_attempts")) != len(failed)
            or _count(inventory.get("n_failed_tasks")) != len(known_tasks)):
        return unavailable("The source-verified task inventory is incomplete or inconsistent.")

    selection = selected_bucket(scores)
    item_ids = set(inventory.get("item_ids") or [])
    source_hash = text_hash(source)
    records = _field(fn, "failure_localisations", {}) or {}
    positives = {
        item_id for item_id, record in records.items()
        if item_id in item_ids and isinstance(record, Mapping)
        and record.get("item_id") == item_id
        and record.get("source_sha256") == source_hash
        and "assigned_label" in record
        and matches_selected_bucket(record["assigned_label"], selection)
    }
    failed_matches = positives & failed.keys()
    matched_tasks = {failed[item_id] for item_id in failed_matches
                     if isinstance(failed[item_id], str) and failed[item_id].strip()}
    result.update({
        "n_matched_failed_tasks": len(matched_tasks),
        "n_matched_failed_attempts": len(failed_matches),
        "n_matched_failed_attempts_without_task_identity": sum(
            not isinstance(failed[item_id], str) or not failed[item_id].strip()
            for item_id in failed_matches
        ),
    })
    if unknown_tasks:
        return unavailable(
            f"{unknown_tasks} failed rollouts lack source task identities; the total failed-task "
            "denominator is unknown. Known task counts remain descriptive only."
        )
    if _count(inventory.get("n_unknown_outcomes")) != 0:
        return unavailable("Unknown source outcomes prevent a complete failed-task denominator.")
    if not known_tasks:
        return unavailable("The selected corpus contains no source-identified failed tasks.")

    unresolved = any(scores.get(key) for key in (
        "n_unresolved", "n_insufficient", "errors", "n_errors", "parent_errors",
    )) or (scores.get("evaluation_complete") is False
           or scores.get("dataset_evaluation_complete") is False
           or scores.get("assessment_coverage", 1) != 1)
    census = (
        scores.get("sampled") is False and scores.get("evaluation_scope_complete") is True
        and not scores.get("eval_truncated") and not unresolved
        and _count(scores.get("n_items")) == len(item_ids) == _count(inventory.get("n_items"))
        and _count(scores.get("n_flagged")) == len(positives)
        and _count(scores.get("n_fail_caught", scores.get("a"))) == len(failed_matches)
    )
    result.update({
        "task_recall": len(matched_tasks) / len(known_tasks),
        "task_recall_scope": "full_selected_corpus" if census else "observed_lower_bound",
    })
    if not census:
        result["task_recall_reason"] = (
            "Observed distinct failed-task matches divided by all failed tasks in the selected "
            "corpus. Sampling, unresolved results, or incomplete match records can miss tasks; "
            "this is a lower bound, not a sample-weighted estimate or causal confirmation."
        )
    return result
