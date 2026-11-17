# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined matches above a chosen detector grade, without rerunning detectors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from math import isfinite
from typing import Any

from autofa.core.dependencies import review_source
from autofa.core.library import FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.metrics.association import matches_selected_bucket, selected_bucket
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
from autofa.metrics.discovery_grades import grade_for_metrics
from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS
from autofa.metrics.verdicts import QUALITY_THRESHOLDS
from autofa.reports.ordering import rank_display_functions

_GOOD_GRADES = {"GOOD", "EXCELLENT", "OUTSTANDING"}


def _count(value: Any) -> int | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0 or int(value) != value):
        return None
    return int(value)


def _hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _inventory_valid(inventory: Any) -> bool:
    if (not isinstance(inventory, Mapping) or inventory.get("inventory_complete") is not True
            or not inventory.get("corpus_fingerprint")):
        return False
    failed, all_items = (inventory.get(key) for key in (
        "task_by_failed_item_id", "task_by_item_id",
    ))
    if not isinstance(failed, Mapping) or not isinstance(all_items, Mapping):
        return False
    if any(not isinstance(item_id, str) or not item_id for item_id in all_items):
        return False
    unknown = inventory.get("unknown_outcome_item_ids", [])
    if (not isinstance(unknown, (list, tuple, set))
            or any(not isinstance(item_id, str) for item_id in unknown)):
        return False
    unknown = set(unknown)
    return (
        _count(inventory.get("n_items")) == len(all_items)
        and _count(inventory.get("n_failed_attempts")) == len(failed)
        and failed.keys() <= all_items.keys()
        and all(all_items[item_id] == task for item_id, task in failed.items())
        and _count(inventory.get("n_unknown_outcomes")) == len(unknown)
        and unknown <= all_items.keys() and not unknown & failed.keys()
        and _count(inventory.get("n_passed_attempts"))
        == len(all_items) - len(failed) - len(unknown)
    )


def library_quality_coverage(
    library: FunctionLibrary,
    inventory: Mapping[str, Any] | None,
    *,
    run_id: str | None = None,
    behavior_strictness: str | None = None,
    objective_strategy: str | None = None,
    saved_assessment: bool = False,
    minimum_grade: str = "OK",
) -> dict[str, Any]:
    """Union detector matches, never individual review cases or summed percentages.

    Eligibility defaults to OK-or-better using the same grade classifier as the
    CLI; minimum_grade="GOOD" retains the Good-or-better cohort. Sampled or
    incomplete positive inventories give a lower bound against the whole known
    failure corpus. Full-corpus claims require consistent, complete match records.
    """
    if not isinstance(minimum_grade, str) or minimum_grade.upper() not in {"OK", "GOOD"}:
        raise ValueError("minimum_grade must be OK or GOOD")
    minimum_grade = minimum_grade.upper()
    qualifying_grades = _GOOD_GRADES | {"OK"} if minimum_grade == "OK" else _GOOD_GRADES
    cohort_label = "OK-or-better" if minimum_grade == "OK" else "Good-or-better"
    excluded_grades = "Weak and unscored" if minimum_grade == "OK" else "OK, Weak and unscored"
    grades, invalid = {}, []
    for fn in library.functions:
        try:
            row = rank_display_functions(
                [vars(fn)], behavior_strictness=behavior_strictness,
                objective_strategy=objective_strategy, review_history=library.review_history,
                saved_assessment=saved_assessment,
            )[0]
            grades[fn.name] = grade_for_metrics(row["metrics"])["code"]
        except (KeyError, TypeError, ValueError, OverflowError):
            grades[fn.name] = "UNSCORED"
            invalid.append(fn.name)
    result: dict[str, Any] = {
        "version": 2, "minimum_grade": minimum_grade,
        "available": False, "status": "unavailable",
        "coverage": None, "task_coverage": None, "task_status": "unavailable",
        "n_good_detectors": sum(code in _GOOD_GRADES for code in grades.values()),
        "n_qualifying_detectors": sum(code in qualifying_grades for code in grades.values()),
        "n_excluded_detectors": 0,
        "n_eligible_detectors": 0, "n_excluded_good_detectors": 0,
        "n_matched_failed_attempts": None, "n_failed_attempts": None,
        "n_matched_failed_tasks": None, "n_failed_tasks": None,
        "n_failed_attempts_without_task_identity": None,
        "n_unknown_outcomes": None,
        "eligible_detector_names": [], "detectors": [], "excluded_detectors": [],
        "unavailable_grade_detector_names": invalid,
        "corpus_fingerprint": None, "cohort_fingerprint": None, "comparison_key": None,
        "behavior_strictness": behavior_strictness,
        "assessment_scope": "saved_report" if saved_assessment else "current_policy",
        "reason": None,
        "interpretation": (
            f"Percentage of failed rollouts matched by at least one current {cohort_label} "
            f"detector. Overlaps count once; {excluded_grades} detectors are excluded. "
            "Matches are detector predictions, not individually judge-confirmed causes. "
            "Distinct-task coverage also deduplicates repeated rollouts. Sampled or incomplete "
            "evaluations provide observed lower bounds against the full failure corpus, "
            "not extrapolated estimates. Unseen cases are not assumed negative."
        ),
    }
    if not _inventory_valid(inventory):
        result["reason"] = "Source failure inventory is unavailable or incomplete."
        return result
    assert inventory is not None
    failed = inventory["task_by_failed_item_id"]
    all_items = inventory["task_by_item_id"]
    corpus = inventory["corpus_fingerprint"]
    task_map = {
        item_id: task for item_id, task in failed.items()
        if isinstance(task, str) and task.strip()
    }
    failed_tasks = set(task_map.values())
    unknown_task_n = len(failed) - len(task_map)
    result.update(
        corpus_fingerprint=corpus, n_failed_attempts=len(failed),
        n_failed_tasks=len(failed_tasks), n_unknown_outcomes=inventory["n_unknown_outcomes"],
        n_failed_attempts_without_task_identity=unknown_task_n,
    )
    matched: set[str] = set()
    complete = True
    for fn in library.functions:
        if grades.get(fn.name) not in qualifying_grades:
            continue
        scores = fn.scores or {}
        source = review_source(fn)
        fingerprint = claim_hash(
            source=source, description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
        )
        exclusion = None
        if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
                or scores.get("evaluation_claim_hash") != fingerprint
                or (run_id is not None and scores.get("evaluation_run_id") != run_id)):
            exclusion = "The detector has no current evaluation for this run and claim."
        elif scores.get("task_recall_corpus_fingerprint") != corpus:
            exclusion = "The saved evaluation does not establish this source corpus."
        if exclusion:
            result["excluded_detectors"].append({"name": fn.name, "reason": exclusion})
            complete = False
            continue
        selection, source_hash = selected_bucket(scores), text_hash(source)
        positives = {
            item_id for item_id, record in (fn.failure_localisations or {}).items()
            if isinstance(record, Mapping) and record.get("item_id") == item_id
            and record.get("source_sha256") == source_hash and "assigned_label" in record
            and matches_selected_bucket(record["assigned_label"], selection)
        }
        # A later sampled evaluation replaces the localisation inventory. The
        # existing task-support receipt preserves same-claim matches outside its
        # panel, without reviving examples rechecked as negatives or errors.
        receipt, panel = scores.get("task_support"), scores.get("evaluation_item_ids")
        if (scores.get("sampled") is True and isinstance(panel, list)
                and all(isinstance(item_id, str) for item_id in panel)
                and isinstance(receipt, Mapping) and receipt.get("version") == 1
                and receipt.get("claim_hash") == fingerprint
                and receipt.get("evaluation_run_id") == scores.get("evaluation_run_id")
                and receipt.get("corpus_fingerprint") == corpus
                and receipt.get("selection") == list(selection)
                and isinstance(receipt.get("failed_matched_task_by_item_id"), Mapping)):
            evaluated_ids = set(panel)
            positives.update(
                item_id for item_id, task in receipt["failed_matched_task_by_item_id"].items()
                if item_id not in evaluated_ids and isinstance(task, str) and task.strip()
                and failed.get(item_id) == task
            )
        failed_matches = positives & failed.keys()
        matched.update(failed_matches)
        census = (
            scores.get("sampled") is False
            and scores.get("evaluation_scope_complete") is True
            and not scores.get("eval_truncated")
            and not any(scores.get(key) for key in (
                "n_unresolved", "n_insufficient", "errors", "n_errors", "parent_errors",
            ))
            and scores.get("evaluation_complete") is not False
            and scores.get("dataset_evaluation_complete") is not False
            and scores.get("assessment_coverage", 1) == 1
            and not inventory["n_unknown_outcomes"]
            and _count(scores.get("n_items")) == inventory["n_items"]
            and _count(scores.get("n_scored")) == inventory["n_items"]
            and positives <= all_items.keys()
            and _count(scores.get("n_flagged")) == len(positives)
            and _count(scores.get("n_fail_caught", scores.get("a"))) == len(failed_matches)
        )
        complete &= census
        result["eligible_detector_names"].append(fn.name)
        result["detectors"].append({
            "name": fn.name, "grade": grades[fn.name], "claim_hash": fingerprint,
            "evaluation_run_id": scores.get("evaluation_run_id"),
            "evaluation_sampled": bool(scores.get("sampled")),
            "match_inventory_complete": census,
            "n_matched_failed_attempts": len(failed_matches),
            "n_matched_failed_tasks": len({task_map[item_id] for item_id in failed_matches
                                          if item_id in task_map}),
        })
    result["eligible_detector_names"].sort()
    result["n_eligible_detectors"] = len(result["eligible_detector_names"])
    result["n_excluded_detectors"] = len(result["excluded_detectors"])
    result["n_excluded_good_detectors"] = sum(
        grades[row["name"]] in _GOOD_GRADES for row in result["excluded_detectors"]
    )
    matched_tasks = {task_map[item_id] for item_id in matched if item_id in task_map}
    result.update(n_matched_failed_attempts=len(matched), n_matched_failed_tasks=len(matched_tasks))
    if inventory["n_unknown_outcomes"]:
        result["reason"] = (
            f"Coverage unavailable: {inventory['n_unknown_outcomes']} rollouts have unknown "
            "benchmark outcomes, so the complete failure denominator is not established."
        )
        return result
    if not failed:
        result["reason"] = "The selected corpus has no known failed rollouts."
        return result
    if result["n_excluded_detectors"] and not result["n_eligible_detectors"]:
        result["reason"] = (
            f"{cohort_label} detectors lack current match evidence for this source corpus."
        )
        return result
    status = "exact" if complete else "observed_lower_bound"
    result.update(
        available=True, status=status, coverage=len(matched) / len(failed),
        task_coverage=(len(matched_tasks) / len(failed_tasks)
                       if failed_tasks and not unknown_task_n else None),
        task_status=status if failed_tasks and not unknown_task_n else "unavailable",
    )
    if complete:
        # Only census unions support numerical round deltas. Sampling expansion
        # or newly recorded diagnostic matches must not masquerade as improvement.
        result["cohort_fingerprint"] = _hash([corpus, "full_failure_corpus"])
        result["comparison_key"] = _hash([
            2, corpus, behavior_strictness, saved_assessment, "qualified_match_union",
            minimum_grade,
            QUALITY_THRESHOLDS, MIN_GOOD_FAILED_TASKS, JUDGE_CONTRACT_VERSION,
        ])
    else:
        result["reason"] = (
            "Observed lower bound: qualifying detectors include sampled or incomplete "
            "match inventories; their individual estimated percentages cannot be added."
        )
    if unknown_task_n:
        result["task_reason"] = (
            f"Task percentage unavailable: {unknown_task_n} failed rollouts "
            "lack source task identity."
        )
    return result


def library_quality_coverage_for_runtime(
    rt: Any, *, cached_only: bool = False, minimum_grade: str = "OK",
) -> dict[str, Any]:
    """Use the runtime's existing identity cache; renderers never initiate I/O."""
    from autofa.behavior_policy import strictness_for_config
    from autofa.session.family_coverage import cached_task_inventory

    reason = None
    try:
        inventory = (getattr(rt, "_family_failed_task_inventory", None) if cached_only
                     else cached_task_inventory(rt))
    except Exception as exc:
        inventory = None
        reason = f"Source failure inventory unavailable: {type(exc).__name__}: {exc}"
    result = library_quality_coverage(
        rt.library, inventory, run_id=getattr(rt.run, "run_id", None),
        behavior_strictness=strictness_for_config(rt.config),
        minimum_grade=minimum_grade,
    )
    if reason:
        result["reason"] = reason
    return result
