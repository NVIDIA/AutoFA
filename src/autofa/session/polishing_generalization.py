# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen per-revision diagnostic samples outside previously flagged successes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from autofa.metrics.association import matches_selected_bucket
from autofa.metrics.judge_checkpoints import fingerprint


def diagnostic_controls(controls: dict, diagnostic: dict | None) -> dict:
    """Expose extra cases through the existing paged diagnostic feedback."""
    rows = {row["item_id"]: row for row in controls.get("controls") or []}
    rows.update({row["item_id"]: row for row in (diagnostic or {}).get(
        "additional_controls") or []})
    return {**controls, "controls": list(rows.values()), "item_ids": sorted(rows)}


def _valid_history(saved: Any, controls: dict, name: str) -> bool:
    return (isinstance(saved, dict) and saved.get("name") == name
            and saved.get("run_id") == controls.get("run_id")
            and saved.get("source_signature") == controls.get("source_signature")
            and saved.get("controls_fingerprint") == controls.get("controls_fingerprint")
            and isinstance(saved.get("additional_controls"), list)
            and saved.get("additional_controls_fingerprint") == fingerprint({
                "controls": saved["additional_controls"],
                "sampling": saved.get("generalization_sample"),
            }))


def _history(directory: Path, name: str, controls: dict):
    for path in sorted(directory.glob("*.json")):
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        if not _valid_history(saved, controls, name):
            continue
        yield path, saved


def _retain_problems(retained: dict, saved: dict, selection: Any) -> None:
    outcomes = {row["item_id"]: row for row in saved.get("results") or []}
    for row in saved["additional_controls"]:
        outcome = outcomes.get(row["item_id"])
        if not outcome:
            continue
        categories = set(row.get("categories") or [])
        categories.discard("fresh_unflagged_success")
        categories.add("retained_diagnostic_problem")
        if outcome.get("status") != "value":
            categories.add("execution_error" if outcome.get("status") == "error"
                           else "unresolved_prediction")
        elif matches_selected_bucket(outcome.get("prediction"), selection):
            categories.add("introduced_success_match")
        else:
            continue
        categories.update(retained.get(row["item_id"], {}).get("categories") or [])
        retained[row["item_id"]] = {**deepcopy(row), "categories": sorted(categories)}


def retained_diagnostic_problems(rt: Any, name: str, controls: dict) -> list[dict]:
    """Carry discovered problems across a subsequent full-validation checkpoint."""
    retained: dict[str, dict] = {}
    directory = Path(rt.run.run_dir()) / "polishing_evaluations"
    for _, saved in _history(directory, name, controls):
        _retain_problems(retained, saved, controls["selection"])
    return list(retained.values())


def prepare_diagnostic_controls(
    rt: Any, name: str, claim: str, controls: dict, result_path: Path,
) -> dict:
    """Freeze on first evaluation; retries recover exactly the same extra cases.

    Other revisions consume diagnostic sample tasks, not independent review
    allocations. Only newly matched or unresolved cases carry forward. Unchanged
    negative controls rotate out so subsequent repairs do not grow toward a census.
    The caller holds the detector operation lock and checkpoints this allocation
    before executing predictions.
    """
    used_tasks: set[str] = set()
    retained: dict[str, dict] = {}
    for path, saved in _history(result_path.parent, name, controls):
        if path == result_path:
            return {key: saved[key] for key in (
                "additional_controls", "additional_controls_fingerprint", "generalization_sample")}
        _retain_problems(retained, saved, controls["selection"])
        for row in saved["additional_controls"]:
            if row.get("task_id") is not None:
                used_tasks.add(row["task_id"])
    pool = controls.get("unflagged_success_pool")
    pool_by_id = {row["item_id"]: row for row in pool or []}
    desired = len(controls.get("panel_item_ids") or [])
    known_tasks = {row["task_id"] for row in controls.get("controls") or []
                   if row.get("task_id") is not None}
    eligible = [row for row in pool_by_id.values()
                if row.get("task_id") is not None and row["task_id"] not in used_tasks]
    seed = getattr(rt.run, "exploration_base_seed", None)
    ordered = sorted(eligible, key=lambda row: (
        row["task_id"] not in known_tasks, fingerprint([
            seed, claim, controls.get("controls_fingerprint"), row["item_id"]])))
    # Prefer tasks outside retained checks, balance benchmarks, and select at
    # most one rollout per task. Passing rollouts of known tasks remain useful
    # diagnostic contrasts when every task also has a failed attempt.
    queues: dict[str, list[dict]] = {}
    for row in ordered:
        queues.setdefault(str(row.get("benchmark") or "unknown"), []).append(row)
    selected, tasks, by_benchmark = [], set(), {}
    unchanged = claim == controls.get("claim_hash")
    while queues and len(selected) < desired and not unchanged:
        benchmark = min(queues, key=lambda key: (
            queues[key][-1]["task_id"] in known_tasks, by_benchmark.get(key, 0), key))
        row = queues[benchmark].pop()
        if not queues[benchmark]:
            del queues[benchmark]
        if row["task_id"] in tasks:
            continue
        tasks.add(row["task_id"])
        selected.append(row)
        by_benchmark[benchmark] = by_benchmark.get(benchmark, 0) + 1
    extra = dict(retained)
    for row in selected:
        extra[row["item_id"]] = {**deepcopy(row), "categories": ["fresh_unflagged_success"],
                                 "review_findings": []}
    sampling = {
        "version": 1, "requested_n": 0 if unchanged else desired, "selected_n": len(selected),
        "distinct_task_n": len(tasks), "eligible_task_n": len({row["task_id"] for row in eligible}),
        "new_task_n": len(tasks - known_tasks), "existing_task_n": len(tasks & known_tasks),
        "previously_checked_task_n": len(used_tasks), "retained_problem_n": len(retained),
        "status": ("unchanged_revision" if unchanged else "unavailable" if pool is None
                   else "complete" if len(selected) == desired else "exhausted"),
        "scope": "diagnostic controls; not a population sample or independent review evidence",
        "limitation": ("The saved validation checkpoint has no unflagged-success inventory."
                       if pool is None else
                       "No additional source-verified distinct tasks remain in this pool."
                       if not unchanged and len(selected) < desired else None),
    }
    rows = [extra[item_id] for item_id in sorted(extra)]
    return {"additional_controls": rows, "generalization_sample": sampling,
            "additional_controls_fingerprint": fingerprint({"controls": rows,
                                                            "sampling": sampling})}
