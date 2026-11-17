# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic quality-first display order shared by CLI and reports.

Display ordering never changes optimizer objectives, eligibility or saved scores.
"""

from __future__ import annotations

import json
from math import isfinite
from typing import Any

REPORT_AXES = ("recall", "ppv_at_pi0", "behavior_purity")
REPORT_SCORE_AXES = ("recall", "ppv_at_pi0", "precision")
_VERDICT_ORDER = {code: rank for rank, code in enumerate(
    ("GOOD", "OK", "WEAK", "EMPTY", "UNSCORED")
)}


def _metric(function: dict[str, Any], axis: str) -> float | None:
    value = (function.get("metrics") or {}).get(axis)
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and 0 <= value <= 1):
        return float(value)
    return None


def has_report_scores(function: dict[str, Any]) -> bool:
    """Keep measured detector results visible independently of review availability.

    Missing or stale adherence affects the verdict, not whether an evaluated
    detector appears. Historical review scores must not be promoted to current
    judgments merely to keep a saved result visible.
    """
    return all(_metric(function, axis) is not None for axis in REPORT_SCORE_AXES)


def _display_verdict(function: dict[str, Any]) -> str:
    from autofa.metrics.verdicts import verdict_for_metrics

    metrics = function.get("metrics") or {}
    evaluation = function.get("evaluation")
    # Older compact reports have only metrics. When evaluation metadata exists,
    # honor its pending/stale/zero-resolved state even if numeric values remain.
    # A metrics-level unavailable state must not be revived by that metadata.
    if isinstance(evaluation, dict) and metrics.get("evaluation_status") != "unscored":
        return verdict_for_metrics(metrics, scores={"evaluated": True, **evaluation})["code"]
    return verdict_for_metrics(metrics)["code"]


def order_report_functions(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Good, OK, Weak, No matches, Not evaluated; highest coverage within a tier.

    Normalized precision, adherence and name break coverage ties. Eligibility for
    discovery optimization remains separate from the visible quality verdict.
    Saved verdict strings are ignored so ordering follows the current classifier.
    """
    def key(function: dict[str, Any]) -> tuple:
        values = tuple(_metric(function, axis) for axis in REPORT_AXES)
        return (
            _VERDICT_ORDER[_display_verdict(function)],
            *(-(value if value is not None else -1) for value in values),
            str(function.get("name") or ""),
        )

    return sorted(functions, key=key)


def visible_report_functions(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Present OK-or-better findings without removing other saved candidates."""
    return [fn for fn in order_report_functions(functions)
            if _display_verdict(fn) in {"GOOD", "OK"}]


def report_ordering_js() -> str:
    """Keep the rendered order authoritative when a host rearranges report data."""
    return (
        "const REPORT_VERDICT_ORDER = " + json.dumps(_VERDICT_ORDER) + ";\n"
        + "const REPORT_AXES = " + json.dumps(REPORT_AXES) + ";\n"
        + r"""
function orderReportFunctions(functions) {
  return [...functions].sort((left, right) => {
    const a = left.metrics || {}, b = right.metrics || {};
    const quality = REPORT_VERDICT_ORDER[classifyFromMetrics(a, left.evaluation).code]
      - REPORT_VERDICT_ORDER[classifyFromMetrics(b, right.evaluation).code];
    if (quality) return quality;
    for (const axis of REPORT_AXES) {
      const difference = (coverageRate(b[axis]) ?? -1) - (coverageRate(a[axis]) ?? -1);
      if (difference) return difference;
    }
    const aName = String(left.name || ''), bName = String(right.name || '');
    return aName < bName ? -1 : aName > bName ? 1 : 0;
  });
}
function visibleReportFunctions(functions) {
  return orderReportFunctions(functions).filter(fn =>
    ['GOOD', 'OK'].includes(classifyFromMetrics(fn.metrics || {}, fn.evaluation).code));
}
"""
    )


def report_assessment(
    function: dict[str, Any], *, review_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Present saved reviews without making them eligible for current discovery.

    A policy revision does not erase the assessment of an unchanged detector.
    Restore its saved measurements only in the presentation metrics, with an
    explicit historical label. Changed claims, missing provenance and reviews
    without completed judgments continue through the ordinary current-policy path.
    """
    from autofa.behavior_policy import STRICTNESS_LABELS, strictness_for_scores
    from autofa.core.dependencies import review_source
    from autofa.metrics.claim_contract import (
        JUDGE_CONTRACT_VERSION,
        _validate_contract,
        claim_hash,
        contract_behavior_strictness,
    )
    from autofa.metrics.claude_judge import compatible_judge_policy
    from autofa.metrics.task_support import build_task_support
    from autofa.metrics.verdicts import (
        behavior_validation,
        current_behavior_scores,
        extract_report_metrics,
        judge_prefilter_skipped,
    )

    raw = function.get("scores") or {}
    source = review_source(function)
    description = function.get("description") or ""
    mechanism = function.get("mechanism") or description or function.get("name") or ""
    scores = current_behavior_scores(
        {**raw, **build_task_support(function, review_history=review_history)},
        source=source, description=description, mechanism=mechanism,
    )
    metrics = extract_report_metrics(scores)
    validation = behavior_validation(scores)
    result = {"scores": scores, "metrics": metrics, "behavior_validation": validation}

    def rate(value: Any) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and isfinite(value) and 0 <= value <= 1)

    def positive_count(value: Any) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and isfinite(value) and value > 0 and value == int(value))

    fingerprint = raw.get("judge_policy_hash")
    strategy = raw.get("judge_objective_strategy", "balanced")
    current_claim = claim_hash(source=source, description=description, mechanism=mechanism)
    if (raw.get("judge_contract_version") != JUDGE_CONTRACT_VERSION
            or not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
            or not isinstance(strategy, str) or strategy not in {"precision", "balanced", "recall"}
            or raw.get("claim_hash") != current_claim
            or raw.get("evaluation_stale")
            or (raw.get("evaluation_claim_hash") is not None
                and raw["evaluation_claim_hash"] != current_claim)
            or raw.get("behavior_judgment_stale") or raw.get("behavior_evidence_stale")
            or raw.get("judge_backend") == "heuristic" or judge_prefilter_skipped(scores)
            or not positive_count(raw.get("behavior_n_judged"))
            or not positive_count(raw.get("behavior_n_completed"))
            or not any(positive_count(raw.get(key)) for key in ("behavior_n_yes", "behavior_n_no"))
            or not rate(raw.get("behavior_purity"))):
        return result
    try:
        strictness = strictness_for_scores(raw)
        if compatible_judge_policy(fingerprint, behavior_strictness=strictness):
            return result
        contract = raw.get("claim_contract")
        _validate_contract(contract)
        if (contract_behavior_strictness(contract) != strictness
                or contract.get("claim_hash", contract["contract_id"]) != raw["claim_hash"]):
            return result
    except (KeyError, TypeError, ValueError):
        return result
    # Honor invalidations already recorded by the run. This is provenance
    # checking, not a replay of today's stricter evidence receipt validation.
    rows = raw.get("behavior_verdicts", [])
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict):
            return result
        for review in (row, row.get("initial_review")):
            if review is None:
                continue
            if not isinstance(review, dict):
                return result
            content = review.get("content_validation")
            if (review.get("verification_status") == "historical"
                    or ("claim_contract_id" in review
                        and review["claim_contract_id"] != contract["contract_id"])
                    or (content is not None and (
                        not isinstance(content, dict)
                        or content.get("status") in {"changed", "unavailable", "invalid", "stale"}
                    ))):
                return result
    metrics.update(
        behavior_purity=float(raw["behavior_purity"]),
        behavior_validation_status="historical",
        behavior_ranking_eligible=False,
        assessment_scope="saved_review",
    )
    if rate(raw.get("behavior_specificity")):
        metrics["behavior_specificity"] = float(raw["behavior_specificity"])
    validation.update(
        status="historical",
        label=f"Saved review — {STRICTNESS_LABELS[strictness]} (earlier evidence policy)",
        ranking_eligible=False,
        assessment_scope="saved_review",
    )
    return result


def rank_display_functions(
    functions: list[dict[str, Any]], *, behavior_strictness: str | None = None,
    review_history: dict[str, Any] | None = None,
    objective_strategy: str | None = None,
    saved_assessment: bool = False,
) -> list[dict[str, Any]]:
    """Prepare raw library rows for the same display ordering as compact reports.

    Resolve current claim/review provenance without copying trace payloads or
    mutating saved scores. Only the active judge policy affects provenance here;
    optimizer strategy and its ranking eligibility do not select display order.
    """
    from autofa.core.dependencies import review_source
    from autofa.metrics.pareto import function_ranking_eligibility
    from autofa.metrics.task_support import build_task_support
    from autofa.metrics.verdicts import current_behavior_scores, extract_report_metrics

    rows = []
    for function in functions:
        if saved_assessment:
            assessment = report_assessment(function, review_history=review_history)
        else:
            scores = current_behavior_scores(
                {**(function.get("scores") or {}),
                 **build_task_support(function, review_history=review_history)},
                source=review_source(function),
                description=function.get("description") or "",
                mechanism=(function.get("mechanism") or function.get("description")
                           or function.get("name") or ""),
                objective_strategy=objective_strategy,
                behavior_strictness=behavior_strictness,
            )
            assessment = {"scores": scores, "metrics": extract_report_metrics(scores)}
        rows.append({
            **function, **assessment,
            "ranking": function_ranking_eligibility(
                function, REPORT_AXES, behavior_strictness=behavior_strictness,
            ),
        })
    return order_report_functions(rows)


def order_library_functions(
    functions: list[dict[str, Any]], *, review_history: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the original saved rows in display order without rewriting evidence.

    Ranking resolves current display metrics in temporary rows. Publication keeps
    each original row, including scores and provenance, exactly as saved.
    """
    return [row["function"] for row in rank_display_functions(
        [{**function, "function": function} for function in functions],
        review_history=review_history,
        saved_assessment=True,
    )]
