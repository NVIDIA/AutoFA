# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule automatic adherence review without inventing a behavioral verdict.

Current score estimates choose where to spend review effort. They neither
permanently reject a detector nor override current per-case judge findings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from math import isfinite
from typing import Any

from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS, task_support_assessment

_REVIEW_PRIORITY = (
    "Follow current per-case judge findings and active repair constraints first; "
    "score-based scheduling does not replace those findings."
)
_DIAGNOSTIC_REVIEW = (
    "When a small review would resolve a specific uncertainty, explicitly call "
    "judge_function with review_mode='diagnostic' for at most 3 matched examples. "
    "Selected diagnostic cases do not establish population detector adherence."
)
_DELIBERATE_OK_REVIEW = (
    "If this is a deliberately useful OK candidate, explicitly call judge_function with "
    "review_mode='standard' to review adherence without lowering the Good requirements."
)
_PRECISION_REPAIR = (
    "Inspect matched passing examples and supported failed matches to distinguish false "
    "matches from a mechanism that occurred but was recovered or incidental. Either fix "
    "false matches while preserving supported breadth, or specialize the claim and "
    "predicate to a coherent failure mechanism for higher precision, accepting and "
    "remeasuring lower coverage. Do not select arbitrary task IDs or keywords to raise scores. "
    "Smoke-test the evidenced change on counterexamples and supported positives, then "
    "reevaluate normalized precision, coverage and distinct failed-task support."
)


def _rate(value: Any) -> float | None:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and 0 <= value <= 1):
        return float(value)
    return None


def _count(value: Any) -> int | None:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and value >= 0 and int(value) == value):
        return int(value)
    return None


def _weighted_count(value: Any) -> float | None:
    """Population estimates may be fractional; observed row counts cannot be."""
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and value >= 0):
        return float(value)
    return None


def automatic_review_schedule(scores: dict[str, Any] | None) -> dict[str, Any]:
    """Choose automatic review effort from current normalized precision.

    Callers resolve detector/claim currency before calling. Normalized precision
    uses the same scope and thresholds as quality grading. Coverage and matched
    failed-task support remain visible for discovery feedback and grading; neither
    prevents reviewing a precise detector with observed matches.
    """
    from autofa.metrics.verdicts import QUALITY_THRESHOLDS

    s = scores or {}
    explains = bool(s.get("within_parent") or s.get("explains"))
    coverage_key = ("within_recall"
                    if explains and s.get("within_recall") is not None else "recall")
    precision_key = ("within_ppv_at_pi0"
                     if explains and s.get("within_ppv_at_pi0") is not None else "ppv_at_pi0")
    coverage, precision = _rate(s.get(coverage_key)), _rate(s.get(precision_key))
    try:
        support = task_support_assessment(s)
    except (TypeError, ValueError, OverflowError, KeyError):
        # Invalid saved provenance supplies no breadth, even if saved counts look good.
        support = task_support_assessment({})
    comparison = support["task_support_comparison"]
    provisional = bool(
        s.get("sampled") or s.get("evaluation_scope_complete") is False
        or s.get("eval_truncated") or s.get("metrics_scope") == "resolved_items"
        or support["task_support_comparison_basis"] == "estimated"
    )
    measured = {
        "ppv_at_pi0": precision,
        "recall": coverage,
        "scope": "within_parent" if explains else "corpus",
        "provisional": provisional,
        "n_matched_failed_tasks": support["n_matched_failed_tasks"],
        "estimated_n_matched_failed_tasks": support["estimated_n_matched_failed_tasks"],
        "matched_task_support_status": support["matched_task_support_status"],
        "task_support_comparison": comparison,
        "task_support_comparison_basis": support["task_support_comparison_basis"],
        "task_support_estimate_status": support["task_support_estimate_status"],
        "task_support_estimate_reason": support["task_support_estimate_reason"],
        "min_good_failed_tasks": MIN_GOOD_FAILED_TASKS,
    }
    ok_precision = QUALITY_THRESHOLDS["ppv_at_pi0"]["ok"]
    good_precision = QUALITY_THRESHOLDS["ppv_at_pi0"]["good"]

    def result(status: str, reason: str, criteria: list[str], *steps: str) -> dict[str, Any]:
        if provisional:
            reason += " This schedule is provisional because evidence is sampled or incomplete."
        return {
            "status": status,
            "reason": reason,
            "next_steps": [
                _REVIEW_PRIORITY, *steps,
                "This scheduling choice is not a behavioral verdict or permanent rejection. "
                "Reassess after a meaningful detector revision or new evaluation evidence.",
            ],
            "criteria": criteria,
            "measured": measured,
        }

    if s.get("evaluated") is not True or s.get("evaluation_stale"):
        return result(
            "deferred", "Current detector evaluation is missing or stale.",
            ["evaluation_unavailable"],
            "Evaluate the current detector and claim before allocating automatic adherence "
            "review; historical scores do not measure this revision.",
        )
    if any(s.get(key) for key in ("errors", "n_errors", "parent_errors")):
        return result(
            "deferred", "Recover detector evaluation errors before automatic adherence review.",
            ["evaluation_errors"],
            "Inspect recorded detector, dependency or provider errors, repair their cause, "
            "smoke-test the implementation, and reevaluate. Execution errors are not evidence "
            "against the stated behavior.",
        )
    count_key = "n_scored" if "n_scored" in s else "n_items"
    scored_count = (_count(s.get(count_key)) if count_key == "n_scored"
                    else _weighted_count(s.get(count_key)))
    if (s.get("metrics_scope") == "unrepresented_strata"
            or precision is None
            or count_key in s and (scored_count or 0) <= 0
            or any(key in s and _weighted_count(s[key]) is None
                   for key in ("n_items", "n_flagged"))):
        return result(
            "deferred",
            "Current normalized precision or resolved prediction measurements are unavailable.",
            ["invalid_measurements"],
            "Inspect evaluation scope, missing population strata and unresolved predictions. "
            "Repair invalid measurements and reevaluate before spending automatic review effort; "
            "raw precision cannot replace normalized ppv_at_pi0.",
        )
    if s.get("n_flagged") == 0:
        return result(
            "deferred", "No observed matches are available for adherence review.",
            ["no_observed_matches"],
            "Evaluate representative examples or revise the detector using supported evidence. "
            "A detector with no matched examples cannot receive adherence review.",
        )
    if precision < ok_precision:
        return result(
            "deferred",
            f"Normalized precision {precision:.2f} is below {ok_precision:.2f}; "
            "defer automatic standard adherence review and investigate precision.",
            ["precision_below_ok"], _PRECISION_REPAIR, _DIAGNOSTIC_REVIEW,
        )
    if precision < good_precision:
        return result(
            "deferred",
            f"Normalized precision {precision:.2f} is below the Good threshold "
            f"{good_precision:.2f}; standard adherence review is not automatic.",
            ["precision_below_good"],
            _PRECISION_REPAIR, _DIAGNOSTIC_REVIEW, _DELIBERATE_OK_REVIEW,
        )
    return result(
        "standard",
        f"Normalized precision ≥{good_precision:.2f} justifies automatic standard adherence "
        "review. Coverage and task support affect detector grading, not review eligibility.",
        ["good_precision"],
        "Proceed with standard adherence review within the existing evidence and budget "
        "limits. Meeting these score conditions does not establish detector adherence.",
    )


def current_review_schedule(scores: dict[str, Any] | None) -> dict[str, Any] | None:
    """Refresh a saved deferral only when its evaluation provenance is still current.

    Callers preserve existing current judge findings ahead of this scheduling view.
    Recompute instead of trusting saved thresholds, reasons or support counts.
    """
    s = scores or {}
    saved = s.get("review_schedule")
    if (not isinstance(saved, Mapping) or saved.get("status") != "deferred"
            or s.get("evaluated") is not True or s.get("evaluation_stale")
            or not saved.get("evaluation_claim_hash")
            or saved["evaluation_claim_hash"] != s.get("evaluation_claim_hash")):
        return None
    if ("evaluation_run_id" in saved
            and saved["evaluation_run_id"] != s.get("evaluation_run_id")):
        return None
    if saved.get("cohort_fingerprint") is not None:
        panel = s.get("search_panel")
        if (not isinstance(panel, Mapping)
                or saved["cohort_fingerprint"] != panel.get("cohort_fingerprint")):
            return None
    if saved.get("evaluation_cohort_hash") is not None:
        panel_ids = s.get("evaluation_item_ids") or []
        if (not isinstance(panel_ids, list)
                or any(not isinstance(item_id, str) for item_id in panel_ids)):
            return None
        cohort_hash = hashlib.sha256(json.dumps(
            sorted(panel_ids), separators=(",", ":"),
        ).encode()).hexdigest()
        if saved["evaluation_cohort_hash"] != cohort_hash:
            return None
    refreshed = automatic_review_schedule(s)
    return refreshed if refreshed["status"] == "deferred" else None
