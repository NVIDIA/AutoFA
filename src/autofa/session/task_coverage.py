# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality-gated, observed task coverage for discovery, without running detectors."""

from __future__ import annotations

from collections import Counter
from math import isclose, isfinite
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.core.library import FunctionLibrary
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.library_coverage import library_failure_coverage
from autofa.metrics.verdicts import current_behavior_scores
from autofa.objectives import discovery_strategy
from autofa.session.discovery_target import (
    MIN_BEHAVIOR_PURITY,
    MIN_STANDARDIZED_PRECISION,
    _completed_review,
)
from autofa.session.family_coverage import cached_task_inventory

# Useful narrow seeds can establish search progress before qualifying as Good.
MIN_PROGRESS_CONFIRMED_TASKS = 2


def _number(value: Any, *, integer: bool = False) -> float | int | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0):
        return None
    if integer:
        return int(value) if int(value) == value else None
    return float(value)


def _outcome_total(scores: dict, metric: str, positive: float, resolved: float) -> float | None:
    """Recover the weighted outcome denominator from saved abstention bounds."""
    bounds = (scores.get("unresolved_bounds") or scores.get("abstention_bounds") or {})
    interval = bounds.get(metric) if isinstance(bounds, dict) else None
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return None
    low, high = (_number(value) for value in interval)
    if low is None or high is None or not 0 <= low <= high <= 1 or high - low >= 1:
        return None
    total = resolved / (1 - high + low)
    if total <= 0 or not isclose(low, positive / total):
        return None
    return total


def _association(scores: dict, inventory: dict, positives: set[str]) -> dict | None:
    """Require every selected row attempted, retaining honest abstentions.

    Sampled contingency cells are weighted population estimates. Their outcome
    totals, including abstentions, stay weighted; actual confirmed task counts
    below are never extrapolated from those estimates.
    """
    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or scores.get("sampled") not in {True, False}
            or any(_number(scores.get(key, 0), integer=True) != 0 for key in (
                "errors", "n_errors", "parent_errors",
            ))):
        return None
    cells = [_number(scores.get(key)) for key in ("a", "b", "c", "d")]
    if any(value is None for value in cells):
        return None
    a, b, c, d = cells
    resolved = _number(scores.get("n_scored"), integer=True)
    abstained = _number(scores.get("n_insufficient", 0), integer=True)
    if (resolved is None or abstained is None or resolved <= 0
            or _number(scores.get("n_unresolved", abstained), integer=True) != abstained
            or _number(scores.get("n_items")) is None
            or not isclose(scores["n_items"], sum(cells))
            or _number(scores.get("n_flagged")) is None
            or not isclose(scores["n_flagged"], a + b)
            or _number(scores.get("n_fail_caught", a)) is None
            or not isclose(scores.get("n_fail_caught", a), a)
            or (not abstained and scores.get("evaluation_complete") is False)):
        return None
    attempted = resolved + abstained
    if not attempted <= inventory["n_items"]:
        return None
    failed = inventory["task_by_failed_item_id"]
    positive_failed = len(positives & failed.keys())
    positive_passed = len(positives - failed.keys())
    if scores["sampled"]:
        selected = scores.get("evaluation_item_ids")
        if (not isinstance(selected, list)
                or any(not isinstance(item_id, str) for item_id in selected)
                or len(set(selected)) != len(selected)
                or len(selected) != attempted
                or _number(scores.get("sample_n"), integer=True) != attempted
                or not set(selected) <= inventory["task_by_item_id"].keys()
                or not (set(selected) & failed.keys())
                or not (set(selected) - failed.keys())):
            return None
        frame_n = _number(scores.get("sample_frame_n"), integer=True)
        if frame_n is None or not attempted <= frame_n <= inventory["n_items"]:
            return None
        # A subsequent confirmation review can scan and record new matches
        # outside the evaluated panel. They support task coverage but do not
        # retroactively change that panel's association estimates.
        panel_positives = positives & set(selected)
        positive_failed = len(panel_positives & failed.keys())
        positive_passed = len(panel_positives - failed.keys())
        # Positive weighted cells cannot contradict the observed positive records.
        if ((a == 0) != (positive_failed == 0) or (b == 0) != (positive_passed == 0)
                or a + 1e-8 < positive_failed or b + 1e-8 < positive_passed):
            return None
    else:
        frame_n = attempted
        if (not isclose(resolved, sum(cells)) or not isclose(a, positive_failed)
                or not isclose(b, positive_passed)):
            return None
    n_fail, n_pass = a + c, b + d
    if abstained:
        n_fail = _outcome_total(scores, "recall", a, n_fail)
        n_pass = _outcome_total(scores, "fpr", b, n_pass)
        # Full scans can recover even a completely abstained outcome stratum
        # directly from the source inventory.
        if scores.get("evaluation_scope_complete") is True:
            n_fail, n_pass = inventory["n_failed_attempts"], inventory["n_passed_attempts"]
    if (n_fail is None or n_pass is None or n_fail <= 0 or n_pass <= 0
            or n_fail + 1e-8 < a + c or n_pass + 1e-8 < b + d
            or not isclose(n_fail + n_pass, frame_n)
            or (scores.get("evaluation_scope_complete") is True
                and (not isclose(n_fail, inventory["n_failed_attempts"])
                     or not isclose(n_pass, inventory["n_passed_attempts"])))):
        return None
    if (scores.get("evaluation_scope_complete") is not True
            and scores.get("eval_mode") != "capped"):
        return None
    return contingency(a, b, n_fail - a, n_pass - b)


def task_coverage_snapshot(rt: Any, *, include_task_ids: bool = False) -> dict[str, Any]:
    """Union current claim confirmations from detectors passing discovery quality gates."""
    result = {
        "behavior_strictness": strictness_for_config(rt.config),
        "quality_thresholds": {
            "ppv_at_pi0": MIN_STANDARDIZED_PRECISION,
            "behavior_purity": MIN_BEHAVIOR_PURITY,
            "min_confirmed_failed_tasks": MIN_PROGRESS_CONFIRMED_TASKS,
        },
        "available": False, "corpus_fingerprint": None, "n_failed_tasks": 0,
        "n_confirmed_failed_tasks": 0, "confirmed_task_coverage": None, "detectors": [],
        "interpretation": (
            "Observed confirmed coverage across source-identified failed tasks, counted once. "
            f"Eligible detectors meet the Good precision (>={MIN_STANDARDIZED_PRECISION:.0%}) "
            f"and adherence (>={MIN_BEHAVIOR_PURITY:.0%}) thresholds, "
            "and confirmations on at least two failed tasks. Capped/sample evaluations and "
            "development reviews can guide search; these counts are lower bounds, not "
            "extrapolated census recall or untouched validation."
        ),
    }
    if include_task_ids:
        result["confirmed_failed_task_ids"] = []
    try:
        inventory = cached_task_inventory(rt)
        if inventory.get("inventory_complete") is not True:
            raise ValueError("Source outcome inventory is incomplete")
        failed = inventory["task_by_failed_item_id"]
        all_items = inventory["task_by_item_id"]
        unknown = set(inventory.get("unknown_outcome_item_ids") or ())
        if (len(all_items) != inventory["n_items"]
                or len(failed) != inventory["n_failed_attempts"]
                or not failed.keys() <= all_items.keys()
                or len(unknown) != inventory["n_unknown_outcomes"]
                or not unknown <= all_items.keys() or unknown & failed.keys()
                or len(all_items) - len(failed) - len(unknown) != inventory["n_passed_attempts"]):
            raise ValueError("Source outcome inventory is inconsistent")
    except Exception as exc:  # Source failures leave discovery feedback explicitly unavailable.
        result["reason"] = f"Task coverage unavailable: {type(exc).__name__}: {exc}"
        return result
    strategy = discovery_strategy(rt.config)
    tasks_by_name = {}
    for fn in rt.library.functions:
        source = review_source(fn)
        description, mechanism = fn.description or "", fn.mechanism or fn.description or fn.name
        fingerprint = claim_hash(source=source, description=description, mechanism=mechanism)
        scores = current_behavior_scores(fn.scores, source=source, description=description,
                                         mechanism=mechanism, objective_strategy=strategy,
                                         behavior_strictness=strictness_for_config(rt.config))
        row = {
            "name": fn.name, "eligible": False, "reason": None,
            "n_confirmed_failed_tasks": 0, "n_exclusive_confirmed_failed_tasks": 0,
            "evaluation_sampled": bool(scores.get("sampled")),
            "current_claim_hash": fingerprint, "selected_evaluation_complete": False,
            "corpus_evaluation_complete": False, "review_complete": False,
        }
        result["detectors"].append(row)
        try:
            # Include passes to verify all reviewed positive cases, including controls.
            coverage = library_failure_coverage(
                FunctionLibrary(functions=[fn]), all_items.keys() - unknown,
                run_id=rt.run.run_id, objective_strategy=strategy, include_item_ids=True,
                behavior_strictness=strictness_for_config(rt.config),
                task_by_item_id=failed,
            )["detectors"][0]
            positives = set(coverage["known_matched_failed_item_ids"])
            association = _association(scores, inventory, positives)
            current = (coverage["evaluation_current"] and
                       scores.get("evaluation_claim_hash") == fingerprint)
            row["selected_evaluation_complete"] = current and association is not None
            row["corpus_evaluation_complete"] = bool(
                row["selected_evaluation_complete"] and scores.get("sampled") is False
                and scores.get("evaluation_scope_complete") is True
                and scores["n_scored"] + scores.get("n_insufficient", 0) == inventory["n_items"]
            )
            reviewed = _completed_review(scores, fingerprint)
            if reviewed:
                reviewed = {v["item_id"] for v in scores["behavior_verdicts"]} <= positives
            row["review_complete"] = reviewed
            from autofa.metrics.review_scheduling import current_review_schedule

            row["review_deferred"] = current_review_schedule(scores) is not None
            confirmed = {failed[item_id] for item_id in coverage["review_confirmed_failed_item_ids"]
                         if item_id in failed and failed[item_id] is not None}
            row["n_confirmed_failed_tasks"] = len(confirmed)
            if not row["selected_evaluation_complete"]:
                row["reason"] = "Needs a current evaluation with every selected row attempted."
            elif association["ppv_at_pi0"] < MIN_STANDARDIZED_PRECISION:
                row["reason"] = (
                    f"Standardized precision is below the Good {MIN_STANDARDIZED_PRECISION:.0%} "
                    "threshold."
                )
            elif not reviewed:
                row["reason"] = "Needs a current completed claim review."
            elif scores["behavior_purity"] < MIN_BEHAVIOR_PURITY:
                row["reason"] = (
                    f"Detector adherence is below the Good {MIN_BEHAVIOR_PURITY:.0%} threshold."
                )
            elif len(confirmed) < MIN_PROGRESS_CONFIRMED_TASKS:
                row["reason"] = "Needs confirmed failures on at least two different tasks."
            else:
                row["eligible"] = True
                tasks_by_name[fn.name] = confirmed
        except (TypeError, ValueError, KeyError, OverflowError):
            row["reason"] = "Current evaluation or review evidence is malformed."
    counts = Counter(task for tasks in tasks_by_name.values() for task in tasks)
    n_failed = len({task for task in failed.values() if task is not None})
    result.update(
        available=True, corpus_fingerprint=inventory.get("corpus_fingerprint"),
        n_failed_tasks=n_failed, n_confirmed_failed_tasks=len(counts),
        confirmed_task_coverage=len(counts) / n_failed if n_failed else None,
        n_unknown_outcomes=len(unknown),
    )
    for row in result["detectors"]:
        row["n_exclusive_confirmed_failed_tasks"] = sum(
            counts[task] == 1 for task in tasks_by_name.get(row["name"], ())
        )
    if include_task_ids:
        result["confirmed_failed_task_ids"] = sorted(counts)
    return result


def task_coverage_prompt(snapshot: dict[str, Any]) -> str:
    """Compact live search feedback without changing per-detector recall semantics."""
    if not snapshot.get("available"):
        return ("Confirmed task coverage is unavailable; restore source identities and current "
                "evaluation/review evidence before claiming broader coverage.")
    return (
        "Library breadth — quality-supported confirmed task coverage: "
        f"{snapshot['n_confirmed_failed_tasks']}/"
        f"{snapshot['n_failed_tasks']} failed tasks (observed lower bound). "
        "Seek additional confirmed tasks with sample_items(coverage='unconfirmed_tasks'). "
        "Investigate another mechanism or a supported top-down/bottom-up extension, while "
        "preserving useful parents and seeds. Repeated rollouts and overlapping detectors do "
        "not add task coverage. Overlaps and subsets can still add a distinct supported "
        "explanation or remedy; report that discovery/quality progress separately from "
        "growth in this union. Preserve the Good quality floors: normalized precision "
        f">={MIN_STANDARDIZED_PRECISION:.0%} and detector adherence >={MIN_BEHAVIOR_PURITY:.0%}; "
        "independently review new matches. Per-detector recall remains a separate metric."
    )
