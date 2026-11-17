# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap association checks before spending calls on behavior judging.

A skip is a scheduling decision for this evaluation, not a behavioral verdict or
an instruction to delete a candidate. Missing evidence never justifies a skip.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from autofa.metrics.association import REFERENCE_PI0
from autofa.metrics.completeness import incomplete_evidence

POLICY_VERSION = "judge-prefilter-v5"


def _number(value: Any, *, probability: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not isfinite(number) or number < 0 or (probability and number > 1):
        return None
    return number


def _is_census(data: dict[str, Any]) -> bool:
    return (
        (data.get("eval_mode") == "full" or data.get("evaluation_scope_complete") is True)
        and data.get("sampled") is False
        and not data.get("eval_truncated")
        and data.get("evaluation_scope_complete") is not False
        and data.get("dataset_evaluation_complete") is not False
    )


def failure_coverage_eligibility(scores: dict[str, Any] | None) -> dict[str, Any]:
    """Validate global measurements without imposing a coverage threshold.

    Coverage never excludes narrow detectors from adherence review. Grades still
    apply their independent coverage requirements.
    """
    data = scores or {}
    recall = _number(data.get("recall"), probability=True)
    n_items = _number(data.get("n_items"))
    n_flagged = _number(data.get("n_flagged"))
    a, c = (_number(data.get(key)) for key in ("a", "c"))
    n_fail = _number(a + c) if a is not None and c is not None else None
    census = _is_census(data)
    measured = {
        "scope": "corpus", "metric": "failure_recall", "recall": recall,
        "estimated": not census,
        "coverage": "census" if census else "limited",
        "n_items": n_items, "n_flagged": n_flagged, "n_fail": n_fail,
    }

    def result(status: str, reason: str) -> dict[str, Any]:
        return {"status": status, "reason": reason, "measured": measured}

    if data.get("evaluated") is not True and data.get("evaluation_complete") is not True:
        return result("not_evaluated", "No completed detector evaluation is recorded.")
    if data.get("judge_prefilter_stale") or data.get("evaluation_stale"):
        return result("not_evaluated", "The detector or claim changed after its evaluation.")
    try:
        incomplete = incomplete_evidence(data)
    except (TypeError, ValueError):
        incomplete = True
    if incomplete:
        return result("inconclusive", "Detector evaluation is incomplete or contains errors.")
    for key in ("assessment_coverage", "recall", "fpr", "ppv_at_pi0",
                "within_recall", "within_fpr", "within_ppv_at_pi0"):
        if key in data and _number(data[key], probability=True) is None:
            return result("inconclusive", "Evaluation rates are missing or invalid.")
    n_scored = _number(data.get("n_scored", data.get("n_items")))
    if (recall is None or n_items is None or n_items <= 0
            or n_scored is None or n_scored <= 0
            or n_flagged is None or n_flagged > n_items
            or n_fail is None or n_fail <= 0 or n_fail > n_items):
        return result("inconclusive", "Global failure-coverage measurements are missing or invalid")
    if a is not None and a > n_flagged:
        return result("inconclusive", "Failure and matched-example counts are inconsistent.")
    return result(
        "pass", "Global measurements are usable; coverage does not gate adherence review.",
    )


def judge_prefilter(scores: dict[str, Any] | None) -> dict[str, Any]:
    """Return a reproducible, JSON-safe decision without changing scores.

    Full, unsampled census metrics can establish an association directly. Capped
    or sampled results need conservative bounds: FPR lower > recall upper.
    Explainers use their within-parent metrics exclusively, including zeroes.
    """
    data = scores or {}
    within = bool(data.get("within_parent") or data.get("explains"))
    prefix = "within_" if within else ""
    rates = {
        name: _number(data.get(prefix + name), probability=True)
        for name in ("recall", "fpr", "ppv_at_pi0")
    }
    bounds = {
        name: _number(data.get(prefix + name), probability=True)
        for name in ("recall_ci_low", "recall_ci_high", "fpr_ci_low", "fpr_ci_high")
    }
    n_items = _number(data.get("within_n" if within else "n_items"))
    n_flagged = _number(data.get(prefix + "n_flagged"))
    if within and prefix + "n_flagged" not in data:
        fraction = _number(data.get("within_pct_caught"), probability=True)
        if fraction is not None and n_items is not None:
            n_flagged = fraction * n_items
    cells = [_number(data.get(prefix + key)) for key in ("a", "b", "c", "d")]
    n_fail = _number(cells[0] + cells[2]) if None not in cells else None
    n_pass = _number(cells[1] + cells[3]) if None not in cells else None
    census = _is_census(data)
    failure_coverage = failure_coverage_eligibility(data)
    measured = {
        "scope": "within_parent" if within else "corpus",
        "coverage": "census" if census else "limited",
        "n_items": n_items,
        "n_flagged": n_flagged,
        "n_fail": n_fail,
        "n_pass": n_pass,
        **rates,
        **bounds,
        "ppv_upper_bound": None,
        "reference_pi0": REFERENCE_PI0,
        "failure_coverage": failure_coverage["measured"],
    }

    def result(status: str, reason: str, *criteria: str) -> dict[str, Any]:
        return {
            "policy_version": POLICY_VERSION, "status": status, "reason": reason,
            "criteria": list(criteria), "measured": measured,
        }

    if data.get("evaluated") is not True and data.get("evaluation_complete") is not True:
        return result("not_evaluated", "No completed detector evaluation is recorded.")
    try:
        incomplete = incomplete_evidence(data)
    except (TypeError, ValueError):
        incomplete = True
    if incomplete:
        return result("inconclusive", "Detector evaluation is incomplete or contains errors.")
    if ("assessment_coverage" in data
            and _number(data["assessment_coverage"], probability=True) is None):
        return result("inconclusive", "Detector assessment coverage is invalid.")
    if failure_coverage["status"] in {"inconclusive", "not_evaluated"}:
        return result(failure_coverage["status"], failure_coverage["reason"])
    n_scored = _number(data.get("n_scored", data.get("n_items")))
    if n_scored is None or n_scored <= 0:
        return result("inconclusive", "No valid positive count of evaluated examples is recorded.")
    if n_items is None or n_flagged is None or n_flagged > n_items:
        return result("inconclusive", "Matched-example counts are missing or invalid.")
    # An explicitly malformed rate is never evidence, even when the fire count is zero.
    if any(prefix + name in data and value is None for name, value in rates.items()):
        return result("inconclusive", "Association metrics contain missing or invalid values.")
    if n_flagged == 0:
        if census:
            return result("skip", "No matched examples were found in the full census.", "no_fires")
        return result("inconclusive", "No matches in a limited evaluation do not prove zero fires.")
    if any(value is None for value in rates.values()):
        return result("inconclusive", "Association metrics are missing for the evaluated scope.")

    recall, fpr = rates["recall"], rates["fpr"]
    assert recall is not None and fpr is not None
    # With no members of either label class, its zero rate is not an association measurement.
    if n_fail is None or n_pass is None or n_fail <= 0 or n_pass <= 0:
        return result("inconclusive", "Association requires evaluated failures and passes.")
    if fpr <= recall:
        return result("pass", "No applicable association criterion excludes behavior judging.")

    if not census:
        if (data.get("sampled") is True
                and (data.get("eval_truncated")
                     or data.get("dataset_evaluation_complete") is False
                     or ("evaluation_scope_complete" in data
                         and data["evaluation_scope_complete"] is not True))):
            return result(
                "inconclusive",
                "A sample from an incomplete corpus frame cannot establish an association skip.",
            )
        rlo, rhi, flo, fhi = (bounds[name] for name in (
            "recall_ci_low", "recall_ci_high", "fpr_ci_low", "fpr_ci_high",
        ))
        if (any(value is None for value in (rlo, rhi, flo, fhi))
                or not rlo <= recall <= rhi or not flo <= fpr <= fhi):
            return result("inconclusive", "Limited evaluation needs valid recall and FPR bounds.")
        denominator = rhi * REFERENCE_PI0 + flo * (1 - REFERENCE_PI0)
        # No positive lower FPR bound gives no finite evidence of low precision.
        upper = rhi * REFERENCE_PI0 / denominator if denominator > 0 else 1.0
        measured["ppv_upper_bound"] = upper
        if flo <= rhi:
            return result("inconclusive", "Confidence bounds do not establish a skip criterion.")

    evidence = "full-corpus metrics" if census else "conservative confidence bounds"
    return result("skip", f"False-positive rate exceeds failure recall ({evidence}).",
                  "negative_association")
