# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-only discovery goals above the shared public Good quality threshold.

These grades guide further search. They do not change CLI/report badges,
ranking, or the Good detector count used by the stopping rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS, task_support_assessment
from autofa.metrics.verdicts import (
    QUALITY_THRESHOLDS,
    extract_metrics,
    verdict_for_metrics,
)

EXCELLENT_COVERAGE = 0.10
OUTSTANDING_COVERAGE = 0.20
GRADE_CODES = ("OUTSTANDING", "EXCELLENT", "GOOD", "OK", "WEAK", "EMPTY", "UNSCORED")


def grade_for_metrics(metrics: dict[str, Any]) -> dict[str, str]:
    """Grade an already resolved metric view, preserving all shared quality gates."""
    verdict = verdict_for_metrics(metrics)
    if verdict["code"] != "GOOD":
        return verdict
    coverage = metrics["recall"]  # The shared classifier has validated this rate.
    for threshold, code, symbol in (
        (OUTSTANDING_COVERAGE, "OUTSTANDING", "●●●"),
        (EXCELLENT_COVERAGE, "EXCELLENT", "●●"),
    ):
        if coverage > threshold:
            return {**verdict, "code": code, "label": code, "symbol": symbol,
                    "reason": (f"Meets all Good requirements with Coverage >{threshold:.0%}. "
                               + verdict["reason"])}
    return verdict


def grade_for_scores(scores: dict[str, Any] | None) -> dict[str, str]:
    """Grade current scores without exposing or duplicating their review payloads."""
    try:
        metrics = extract_metrics(scores)
    except (TypeError, ValueError, OverflowError):
        # Malformed historical counts need inspection, not a fabricated grade.
        return verdict_for_metrics({})
    return grade_for_metrics(metrics)


def grade_gap_for_scores(  # noqa: C901
    scores: dict[str, Any] | None, *, target_grade: str | None = None,
) -> dict[str, Any]:
    """Describe requirements for a requested grade, defaulting to the next milestone.

    This is directional advice, subordinate to current per-case findings. Missing
    measurements remain unavailable; diagnostic reviews and saved task counts cannot
    supply adherence or source-verified matched breadth.
    """
    if target_grade is not None:
        if not isinstance(target_grade, str) or target_grade.upper() not in {
            "OK", "GOOD", "EXCELLENT", "OUTSTANDING",
        }:
            raise ValueError("target_grade must be OK, GOOD, EXCELLENT, or OUTSTANDING")
        target_grade = target_grade.upper()
    s = scores or {}
    try:
        metrics = extract_metrics(s)
    except (TypeError, ValueError, OverflowError):
        metrics = {}
    current = grade_for_metrics(metrics)["code"]
    next_grade = {
        "WEAK": "OK", "OK": "GOOD", "GOOD": "EXCELLENT",
        "EXCELLENT": "OUTSTANDING", "OUTSTANDING": None,
        "UNSCORED": "OK", "EMPTY": "OK",
    }[current]
    target = target_grade or next_grade
    threshold_grade = target or "OUTSTANDING"
    tier = "ok" if threshold_grade == "OK" else "good"
    coverage_required = {
        "EXCELLENT": EXCELLENT_COVERAGE, "OUTSTANDING": OUTSTANDING_COVERAGE,
    }.get(threshold_grade, QUALITY_THRESHOLDS["recall"][tier])

    def rate(value: Any) -> float | None:
        return (float(value) if isinstance(value, (int, float))
                and not isinstance(value, bool) and isfinite(value) and 0 <= value <= 1
                else None)

    explains = bool(s.get("within_parent") or s.get("explains"))
    scope = "within_parent" if explains else "corpus"
    current_evaluation = (
        metrics.get("evaluation_status") != "unscored"
        and s.get("evaluated") is not False and not s.get("evaluation_stale")
    )
    coverage = (rate(metrics.get("recall"))
                if current_evaluation and metrics.get("coverage_available") is not False else None)
    precision = (rate(metrics.get("ppv_at_pi0"))
                 if current_evaluation
                 and metrics.get("normalized_precision_available") is not False
                 else None)
    adherence = rate(metrics.get("behavior_purity"))
    review_status = metrics.get("behavior_validation_status", "invalid")
    # extract_metrics already resolves whether the aggregate is available.
    # Partial/uncertain reviews can still have a measured aggregate; keep it,
    # while the existing per-case diagnosis governs evidence recovery.
    if review_status == "legacy" and tier == "good":
        adherence = None
    try:
        support = task_support_assessment(s)
    except (TypeError, ValueError, OverflowError, KeyError):
        support = task_support_assessment({})
    breadth = support["task_support_comparison"]
    provisional = bool(
        s.get("sampled") or s.get("evaluation_scope_complete") is False
        or s.get("eval_truncated") or s.get("metrics_scope") == "resolved_items"
        or support["task_support_comparison_basis"] == "estimated"
    )
    labels = {
        "recall": "coverage", "ppv_at_pi0": "normalized precision",
        "behavior_purity": "detector adherence",
        "matched_failed_tasks": "distinct failed-task support",
    }
    requirements = [
        ("recall", coverage, coverage_required, bool(QUALITY_THRESHOLDS["recall"]["strict"])),
        ("ppv_at_pi0", precision, QUALITY_THRESHOLDS["ppv_at_pi0"][tier],
         bool(QUALITY_THRESHOLDS["ppv_at_pi0"]["strict"])),
        ("behavior_purity", adherence, QUALITY_THRESHOLDS["behavior_purity"][tier],
         bool(QUALITY_THRESHOLDS["behavior_purity"]["strict"])),
        ("matched_failed_tasks", breadth, None if tier == "ok" else MIN_GOOD_FAILED_TASKS, False),
    ]
    dimensions = []
    for key, value, required, strict in requirements:
        met = (True if required is None else None if value is None
               else value > required if strict else value >= required)
        dimension = {
            "key": key, "label": labels[key], "current": value, "required": required,
            "operator": None if required is None else ">" if strict else ">=",
            "met": met,
            "status": ("not_required" if required is None else "unavailable" if value is None
                       else "met" if met else "below_requirement"),
            "next_steps": [],
        }
        if key == "matched_failed_tasks":
            dimension.update(
                observed=support["n_matched_failed_tasks"],
                estimated=support["estimated_n_matched_failed_tasks"],
                basis=support["task_support_comparison_basis"],
            )
        if met is not True:
            if key in {"recall", "ppv_at_pi0"} and value is None:
                advice = (
                    f"Recover current {labels[key]} by inspecting evaluation errors, unresolved "
                    "predictions and missing strata, then evaluate the current detector. "
                    "An unavailable measurement is not a measured zero."
                )
            elif key == "recall":
                advice = (
                    "Inspect missed failed examples for broader variants of the same coherent "
                    "mechanism. Broaden an existing detector or synthesize a shared abstraction "
                    "from related detector seeds when the evidence supports it; preserve the "
                    "supported version and useful individual originals. Compare the deduplicated "
                    "union of matched tasks on the same scoring cohort, without summing overlaps "
                    "or OR-ing unrelated causes. Remeasure coverage, precision and adherence, "
                    "and independently validate newly included matches across distinct tasks, "
                    "including passing and boundary examples. Keep targeted diagnostics separate "
                    "from population estimates and complete the planned final full-corpus "
                    "evaluation."
                )
                if provisional:
                    advice += (
                        " Coverage is provisional while scoring is sampled or incomplete; "
                        "unvisited or unresolved tasks are not observed misses. Obtain "
                        "representative, resolved evidence before inferring corpus breadth."
                    )
            elif key == "ppv_at_pi0":
                advice = (
                    "Inspect matched passing cases and supported failed matches. Either fix "
                    "false matches while preserving supported breadth, or specialize the claim "
                    "and predicate to a coherent mechanism for higher normalized precision, "
                    "accepting and remeasuring lower coverage. Benchmark outcomes alone are not "
                    "mechanism labels; do not add arbitrary task IDs or keyword exceptions."
                )
            elif key == "behavior_purity" and review_status in {"uncertain", "error", "incomplete"}:
                advice = (
                    "Inspect current per-case diagnosis for missing, unread or ambiguous "
                    "evidence and processing errors. Recover the needed source evidence or "
                    "resume the frozen review within its existing budget; do not count "
                    "uncertainty as a false match or repeat favorable cases to erase it."
                )
            elif key == "behavior_purity" and value is None:
                from autofa.metrics.review_scheduling import automatic_review_schedule

                schedule = automatic_review_schedule(s)
                advice = (
                    "Obtain a current standard adherence review within the existing budget."
                    if schedule["status"] == "standard" else
                    "Follow review_schedule before requesting standard adherence review. "
                    "A bounded diagnostic can guide a specific repair but cannot establish "
                    "adherence; improve the blocking scores or deliberately review a useful "
                    "OK candidate when its normalized precision reaches the OK floor."
                )
                advice += " Missing or historical reviews are not poor adherence scores."
            elif key == "behavior_purity":
                advice = (
                    "Use get_function_feedback/get_function_repair to inspect rejected claim "
                    "requirements, counterexamples and later recovery. Repair a mismatched "
                    "predicate or revise the claim to the mechanism the evidence supports; "
                    "test the distinction on counterexamples and supported positives before "
                    "requesting more reviews. Follow the actual diagnosis and active repair "
                    "constraints; additional favorable cases do not repair a known mismatch."
                )
            elif breadth is None:
                advice = (
                    "Recover source task identities and the current matched-failure receipt. "
                    "Bare saved counts and additional judge confirmations cannot establish "
                    "distinct failed-task support."
                )
            elif provisional:
                advice = (
                    "Inspect observed matched tasks and the validated sampling projection. "
                    "A small sample does not establish narrow corpus support; obtain the planned "
                    "representative or final evaluation before deciding to broaden. Repeated "
                    "rollouts and additional judge confirmations do not add distinct tasks."
                )
            else:
                advice = (
                    "Seek matches on additional distinct failed tasks through evidenced variants "
                    "of the same coherent mechanism. Preserve the supported narrow candidate "
                    "while testing the extension. Repeated rollouts of existing tasks and "
                    "additional judge confirmations do not satisfy this breadth requirement."
                )
            dimension["next_steps"] = [advice]
        dimensions.append(dimension)

    blocked = [dimension for dimension in dimensions if dimension["met"] is not True]

    def describe(dimension: dict[str, Any]) -> str:
        key, value, required = (dimension[name] for name in ("key", "current", "required"))
        measured = ("unavailable" if value is None else f"{value:.2%}" if key == "recall"
                    else f"{value:g}" if key == "matched_failed_tasks" else f"{value:.2f}")
        boundary = f"{required:.0%}" if key == "recall" else (
            f"{required:g}" if key == "matched_failed_tasks" else f"{required:.2f}")
        # Preserve the OK half-percent threshold instead of rounding it to zero.
        if key == "recall" and required < .01:
            boundary = f"{required:.1%}"
        if value is not None and value < required:
            displayed = float(measured.rstrip("%")) / (100 if key == "recall" else 1)
            if displayed >= required:
                measured = "<" + boundary
        operator = ">" if dimension["operator"] == ">" else "≥"
        return f"{dimension['label']} ({measured}; requires {operator}{boundary})"

    target_met = not blocked
    if target is None:
        summary = "Outstanding requirements are met; no higher discovery grade is defined."
    elif target_met:
        summary = f"{target.title()} requirements are already met; retain this supported detector."
    elif len(blocked) == 1:
        summary = (f"To reach {target.title()}, the only blocking dimension is "
                   + describe(blocked[0]) + ".")
    elif blocked:
        summary = (f"To reach {target.title()}, blocking dimensions are "
                   + "; ".join(describe(dimension) for dimension in blocked) + ".")
    if current in {"UNSCORED", "EMPTY"}:
        summary += " Establish a measurable initial grade; unavailable evidence is not Weak."
    if provisional:
        summary += " Qualification is provisional because scoring is sampled or incomplete."
    return {
        "current_grade": current, "next_grade": next_grade, "target_grade": target,
        "target_met": target_met, "scope": scope,
        "provisional": provisional, "dimensions": dimensions,
        "bottlenecks": [dimension["key"] for dimension in blocked],
        "summary": summary,
        "next_steps": [step for dimension in blocked for step in dimension["next_steps"]],
    }


def grading_table() -> str:
    """Compact agent guidance, sharing the actual score and task-support floors."""
    coverage = QUALITY_THRESHOLDS["recall"]
    precision = QUALITY_THRESHOLDS["ppv_at_pi0"]
    adherence = QUALITY_THRESHOLDS["behavior_purity"]
    return (
        "| Criterion | Weak | OK | Good ● | Excellent ●● | Outstanding ●●● |\n"
        "|---|---|---|---|---|---|\n"
        f"| Coverage | ≤{coverage['ok']:.1%} | >{coverage['ok']:.1%} | "
        f">{coverage['good']:.0%} | >{EXCELLENT_COVERAGE:.0%} | >{OUTSTANDING_COVERAGE:.0%} |\n"
        f"| Normalized precision | <{precision['ok']:.2f} | ≥{precision['ok']:.2f} | "
        f"≥{precision['good']:.2f} | ≥{precision['good']:.2f} | ≥{precision['good']:.2f} |\n"
        f"| Detector adherence | <{adherence['ok']:.2f} | ≥{adherence['ok']:.2f} | "
        f"≥{adherence['good']:.2f} | ≥{adherence['good']:.2f} | ≥{adherence['good']:.2f} |\n"
        "| Matched distinct failed tasks (estimated during sampled scoring) | "
        "No minimum | No minimum | "
        f"≥{MIN_GOOD_FAILED_TASKS} | ≥{MIN_GOOD_FAILED_TASKS} | "
        f"≥{MIN_GOOD_FAILED_TASKS} |\n\n"
        "OK and higher require every condition in their column; any score below the OK floor "
        "makes a detector Weak. During sampled scoring, compare the estimated full-corpus "
        "matched task support with the five-task floor and mark the grade provisional. "
        "Keep observed matches and reviewed confirmations visible separately. Final qualification "
        "requires matches on at least five distinct failed tasks after the full scan. "
        "Judge adherence is a separate quality requirement; the reviewed subset need not "
        "contain five confirmed tasks. "
        "An unavailable estimate cannot substitute for observed support; otherwise Good scores "
        "without sufficient support remain OK. No matches and Not evaluated retain their "
        "existing availability meanings. "
        "Repeated rollouts of a task count once. Use these grades to guide broader "
        "supported discovery."
    )


def agent_quality_guidance() -> str:
    """Describe the agent's breadth and quality target."""
    precision = QUALITY_THRESHOLDS["ppv_at_pi0"]["good"]
    adherence = QUALITY_THRESHOLDS["behavior_purity"]["good"]
    return (
        "Use pareto_status.target_grade and the live round assessment for the active target. "
        "Pursue that target until the runtime reports a revised target. Do not infer a "
        "different target from round or grade counts. "
        "Build a library of different supported failure mechanisms and informative subtypes "
        "across tasks, rather than a collection of variants of one claim. "
        "Aim for the requested detector count at the active grade, "
        "while treating Good and Excellent detectors as successful, valuable discoveries. "
        f"Outstanding ●●● detectors require Coverage >{OUTSTANDING_COVERAGE:.0%}, "
        f"normalized precision ≥{precision:.2f}, detector adherence ≥{adherence:.2f}, and "
        f"support for at least {MIN_GOOD_FAILED_TASKS} distinct failed tasks: use the "
        "estimated full-corpus matched task support for provisional sampled grades, and "
        "actual matched failed tasks for final qualification. "
        f"Excellent ●● requires Coverage >{EXCELLENT_COVERAGE:.0%} with the same quality "
        "and support floors.\n\n"
        "Aim for Outstanding detectors where a coherent, well-supported mechanism explains "
        "many failures. Excellent and Good detectors can "
        "identify more specific decisions that break the final deliverable and suggest "
        "concrete remedies. Both broad families and specific mechanisms belong in the "
        "library, including when their coverage overlaps or one detector's matches are a "
        "subset of another's. A deep dive can explain a specific damaging decision within "
        "a broader family without adding any new matched tasks. Retain it when that "
        "distinction adds a supported explanation, diagnostic use or remedy, and validate "
        "its own precision, adherence and task support. Equivalent duplicates do not add "
        "useful discovery progress.\n\n"
        "When a detector meets the Good or Excellent requirements, keep it and count it as "
        "useful progress. The active grade is an aspiration, not a retention threshold. "
        "Assess a candidate against its own grade's requirements; do not discard a supported "
        "Good detector merely because its Coverage is below Excellent or Outstanding. "
        "Broaden it when the evidence supports a coherent broader mechanism, "
        "preserving the supported version while testing the broader candidate. Discovering "
        "another useful mechanism is also productive progress.\n\n"
        "Choose the next investigation based on the evidence and its potential to improve "
        "the library. Do not sacrifice precision or adherence to reach a higher coverage "
        "tier. No fixed mix of grades is required. Follow the runtime's round and stopping "
        "instructions.\n\n"
        "Preserve supported narrow seeds; retain coverage and independently review new "
        "matches when broadening or synthesizing related mechanisms. Compare candidates on "
        "the same scoring panel, and distinguish sampled observations from full-corpus "
        "Coverage. A small sampled bucket alone does not establish narrow corpus support; "
        "inspect the extrapolation and its uncertainty before choosing an action. Repeated "
        "rollouts do not add distinct-task support. Reuse valid "
        "existing reviews to assess adherence; do not require five reviewed confirmations "
        "merely to meet the distinct-task breadth floor. Do not trade away precision or "
        "adherence to increase breadth."
    )


def agent_quality_reminder() -> str:
    """Short policy reference for routine feedback; the protocol owns the full goal."""
    return (
        "Follow pareto_status.target_grade and the runtime's round and stopping instructions. "
        "Keep useful Good and Excellent findings. "
        "The active grade is an aspiration, not a retention threshold. Investigate distinct "
        "mechanisms or informative subtypes, including overlaps. Preserve precision and adherence. "
        "See autofa_protocol.md for quality guidance and the grading table."
    )


def grade_summary(
    functions: Iterable[Any], *, behavior_strictness: str | None = None,
    review_history: dict[str, Any] | None = None,
    objective_strategy: str | None = None,
) -> dict[str, Any]:
    """Compact current grades; these counts do not establish stopping eligibility."""
    from autofa.reports.ordering import rank_display_functions

    rows = rank_display_functions(
        [dict(function) if isinstance(function, Mapping) else vars(function)
         for function in functions],
        behavior_strictness=behavior_strictness, review_history=review_history,
        objective_strategy=objective_strategy,
    )
    counts = dict.fromkeys(GRADE_CODES, 0)
    summaries = []
    for row in rows:
        grade = grade_for_metrics(row["metrics"])
        counts[grade["code"]] += 1
        summaries.append({"name": row.get("name"), "code": grade["code"],
                          "label": grade["label"], "symbol": grade["symbol"],
                          "coverage": row["metrics"].get("recall"),
                          "n_matched_failed_tasks": row["metrics"].get("n_matched_failed_tasks"),
                          "estimated_n_matched_failed_tasks": row["metrics"].get(
                              "estimated_n_matched_failed_tasks"),
                          "n_confirmed_failed_tasks": row["metrics"].get(
                              "n_confirmed_failed_tasks"),
                          "estimated_n_confirmed_failed_tasks": row["metrics"].get(
                              "estimated_n_confirmed_failed_tasks"),
                          "task_support_comparison_basis": row["metrics"].get(
                              "task_support_comparison_basis", "unavailable")})
    return {"counts": counts, "functions": summaries,
            "n_good_or_better": sum(counts[code] for code in ("GOOD", "EXCELLENT", "OUTSTANDING")),
            "n_outstanding": counts["OUTSTANDING"]}
