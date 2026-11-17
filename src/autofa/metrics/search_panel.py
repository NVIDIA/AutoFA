# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comparable discovery breadth on the fixed LLM scoring panel.

Project already evaluated matches onto the same item IDs for every engine. This
does not execute detectors or extrapolate row weights into task coverage. Saved
corpus metrics, independent review, and confirmed coverage remain separate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from autofa.metrics.task_recall import build_task_recall


def build_search_panel(
    fn: Any, inventory: Mapping[str, Any], *, item_ids: Sequence[str], run_id: str,
    attempted_item_ids: Sequence[str] | None = None,
    allocation_version: str | None = None,
) -> dict[str, Any]:
    """Freeze breadth from fresh evaluation evidence, before judge probes add matches.

    ``item_ids`` is the fixed reservoir allocation, even if a later point read
    failed to retrieve some rows. Such a partial read cannot silently create a
    smaller, favorable comparison cohort. ``None`` for attempted IDs means the
    caller evaluated every row in the stream that supplied this allocation.
    """
    scores = fn.get("scores", {}) if isinstance(fn, Mapping) else fn.scores
    scores = scores or {}
    result = {
        "version": 1, "available": False, "scope": "fixed_scoring_panel",
        "claim_hash": scores.get("evaluation_claim_hash"), "run_id": run_id,
        "corpus_fingerprint": inventory.get("corpus_fingerprint"),
        "task_recall": None,
        "allocation_version": allocation_version,
    }

    def unavailable(reason: str) -> dict[str, Any]:
        return {**result, "reason": reason}

    ids = set(item_ids)
    if (not ids or len(ids) != len(item_ids)
            or any(not isinstance(item_id, str) or not item_id for item_id in ids)
            or not ids <= set(inventory.get("item_ids") or [])):
        return unavailable("The fixed scoring panel is missing or has inconsistent source IDs.")
    result.update(
        n_items=len(ids),
        cohort_fingerprint=hashlib.sha256(json.dumps(
            [inventory.get("corpus_fingerprint"), sorted(ids), allocation_version],
            separators=(",", ":"),
        ).encode()).hexdigest(),
    )
    if attempted_item_ids is not None and not ids <= set(attempted_item_ids):
        return unavailable(
            "Some fixed-panel rows were not attempted. Recover the same evaluation before "
            "comparing breadth; missing rows do not become detector misses."
        )
    failed = inventory.get("task_by_failed_item_id")
    if not isinstance(failed, Mapping):
        return unavailable("Source task identities are unavailable for the scoring panel.")
    selected_failed = {item_id: task for item_id, task in failed.items() if item_id in ids}
    selected_inventory = {
        **inventory, "item_ids": sorted(ids), "n_items": len(ids),
        "task_by_failed_item_id": selected_failed,
        "n_failed_attempts": len(selected_failed),
        "n_failed_tasks": len({task for task in selected_failed.values() if task is not None}),
    }
    if "unknown_outcome_item_ids" in inventory:
        selected_inventory["n_unknown_outcomes"] = len(
            ids & set(inventory["unknown_outcome_item_ids"])
        )
    breadth = build_task_recall(fn, selected_inventory, run_id=run_id)
    result.update({key: breadth[key] for key in (
        "task_recall", "n_failed_tasks", "n_matched_failed_tasks",
    )})
    if breadth["task_recall"] is None:
        return unavailable(breadth.get("task_recall_reason", "Panel breadth is unavailable."))
    # Never shrink the denominator to resolved rows. Abstentions retain observed
    # lower-bound coverage; execution errors cannot yield a useful search score.
    errors = any(scores.get(key) for key in (
        "errors", "n_errors", "parent_errors",
    ))
    if errors:
        result["task_recall"] = None
        return unavailable(
            "Recover detector or dependency execution errors before ranking breadth."
        )
    result.update(
        available=True, observed_lower_bound=bool(scores.get("n_unresolved")
                                                  or scores.get("n_insufficient")),
    )
    return result


def current_search_panel(scores: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate the saved panel without mutating or inventing historical evidence."""
    panel = scores.get("search_panel")
    if (not isinstance(panel, Mapping) or panel.get("version") != 1
            or panel.get("available") is not True
            or panel.get("scope") != "fixed_scoring_panel"
            or scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or not panel.get("claim_hash")
            or panel["claim_hash"] != scores.get("evaluation_claim_hash")
            or not panel.get("run_id") or panel["run_id"] != scores.get("evaluation_run_id")
            or not panel.get("corpus_fingerprint")
            or panel["corpus_fingerprint"] != scores.get("task_recall_corpus_fingerprint")
            or not isinstance(panel.get("cohort_fingerprint"), str)
            or not panel["cohort_fingerprint"]
            or not isinstance(panel.get("observed_lower_bound"), bool)):
        return None
    counts = [panel.get(key) for key in (
        "n_items", "n_failed_tasks", "n_matched_failed_tasks",
    )]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 or int(value) != value for value in counts):
        return None
    n_items, failed, matched = counts
    recall = panel.get("task_recall")
    if (not 0 < failed <= n_items or not 0 <= matched <= failed
            or isinstance(recall, bool) or not isinstance(recall, (int, float))
            or not math.isfinite(recall) or not math.isclose(recall, matched / failed)):
        return None
    return dict(panel)
