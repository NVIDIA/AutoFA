# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score-based investigations, subordinate to current per-case review evidence.

This advice changes neither detector acceptance nor scores. Task recall measures
observed matches, not causal confirmations; sampled coverage remains a lower bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from autofa.behavior_policy import strictness_for_scores, validate_behavior_strictness
from autofa.metrics.claim_calibration import (
    review_issue_guidance,
    review_issue_types,
    review_policy_guidance,
)
from autofa.metrics.search_panel import current_search_panel
from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS, task_support_assessment


def _rate(value: Any) -> float | None:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and 0 <= value <= 1):
        return float(value)
    return None


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def build_discovery_feedback(  # noqa: C901
    scores: Mapping[str, Any], *, judge_feedback: Mapping[str, Any] | None = None,
    behavior_strictness: str | None = None,
    target_grade: str | None = None,
) -> dict[str, Any]:
    """Connect current metrics to useful investigations without forcing an edit.

    Callers must resolve current evaluation/claim provenance before calling. Pass
    an existing ``summarize_judge_feedback`` result to retain its detailed findings;
    an explicit active policy still checks that summary against the saved review.
    Show its per-case findings and repair instructions before this advisory block.
    Good quality thresholds are shared with the displays. Breadth informs the
    investigation alongside evidence and quality; it does not require an edit.
    """
    from autofa.metrics.discovery_grades import (
        agent_quality_reminder,
        grade_for_scores,
        grade_gap_for_scores,
    )
    strictness = (strictness_for_scores(scores) if behavior_strictness is None
                  else validate_behavior_strictness(behavior_strictness))
    if judge_feedback is None or behavior_strictness is not None:
        from autofa.metrics.verdicts import behavior_validation

        try:
            current_validation = behavior_validation(
                dict(scores), include_details=False, behavior_strictness=strictness,
            )
        except (TypeError, ValueError, OverflowError):
            current_validation = {"status": "invalid"}
        # A caller-provided summary cannot turn evidence from another policy into
        # current support. Keep other caller details when the saved policy agrees.
        if judge_feedback is None or current_validation["status"] in {"stale", "invalid"}:
            judge_feedback = current_validation
    validation = dict(judge_feedback)
    status = validation.get("status")
    explains = bool(scores.get("within_parent") or scores.get("explains"))
    precision = _rate(
        scores.get("within_ppv_at_pi0")
        if explains and scores.get("within_ppv_at_pi0") is not None else scores.get("ppv_at_pi0")
    )
    purity = _rate(scores.get("behavior_purity"))
    support = task_support_assessment(scores)
    support_comparison = support["task_support_comparison"]
    corpus_breadth = _rate(scores.get("task_recall"))
    if scores.get("task_recall_scope") == "unavailable":
        corpus_breadth = None
    panel = current_search_panel(scores)
    limited_corpus = bool(
        scores.get("sampled") or scores.get("evaluation_scope_complete") is False
        or scores.get("task_recall_scope") == "observed_lower_bound"
        or _positive(scores.get("n_insufficient")))
    # Low corpus coverage on a small panel describes evaluation effort as well as
    # detector behavior. Only a matched panel or an actual census can inform breadth.
    breadth = panel["task_recall"] if panel else None if limited_corpus else corpus_breadth
    grading_scores = dict(scores)
    if behavior_strictness is not None and status == "stale":
        # The active policy check above can invalidate an otherwise well-formed
        # saved review. Neither the grade nor its upgrade advice may reuse it.
        grading_scores["behavior_judgment_stale"] = True
    grade_gap = grade_gap_for_scores(grading_scores, target_grade=target_grade)
    # The active aspiration can require broader coverage even when the immediate
    # useful improvement is qualifying for Good. Keep both views on the same
    # provenance-resolved scores and shared thresholds.
    qualification_gap = grade_gap_for_scores(grading_scores, target_grade="GOOD")
    qualification_summary = (
        "Current scores meet Good requirements; preserve coverage, normalized precision, "
        "detector adherence and distinct-task support while pursuing higher grades."
        if qualification_gap["target_met"] else
        "Good qualification: " + qualification_gap["summary"]
    )
    if qualification_gap["target_met"] and qualification_gap["provisional"]:
        qualification_summary += " This qualification remains provisional until validation."
    requirements = {dimension["key"]: dimension["required"]
                    for dimension in grade_gap["dimensions"]}
    required_precision = requirements["ppv_at_pi0"]
    required_purity = requirements["behavior_purity"]
    required_support = requirements["matched_failed_tasks"]
    quality_floor = "OK" if grade_gap["target_grade"] == "OK" else "Good"
    result: dict[str, Any] = {
        "advisory": True,
        "quality_goal": agent_quality_reminder(),
        "discovery_grade": grade_for_scores(grading_scores),
        "grade_gap": grade_gap,
        "grade_gap_summary": grade_gap["summary"],
        "qualification_gap": qualification_gap,
        "qualification_gap_summary": qualification_summary,
        "behavior_strictness": strictness,
        "status": "inspect_scores",
        "signals": {
            "standardized_precision": precision,
            "behavior_purity": purity,
            "task_recall": corpus_breadth,
            "task_recall_scope": scores.get("task_recall_scope"),
            "task_recall_reason": scores.get("task_recall_reason"),
            **support,
            # Preserve the evaluated-cohort count paired with task_recall. The
            # support comparison can additionally include source-verified matches
            # inspected during review outside that frozen evaluation cohort.
            "n_matched_failed_tasks": scores.get("n_matched_failed_tasks"),
            "min_good_failed_tasks": MIN_GOOD_FAILED_TASKS,
            "n_failed_tasks": scores.get("n_failed_tasks"),
            "review_status": status,
            "search_panel": panel,
        },
        "next_steps": [],
        "issue_types": [],
        "caveats": [
            "Detailed current judge findings and active repair constraints take precedence. "
            "Scores suggest investigations; they do not establish why a detector needs repair.",
            "Task recall counts observed matches on distinct failed tasks, "
            "not review-confirmed claims. "
            "Repeated rollouts do not add task breadth. Pass/fail labels alone do not establish "
            "whether the specific mechanism occurred.",
        ],
    }
    if limited_corpus:
        result["caveats"].append(
            "Task coverage is an observed lower bound from limited or unresolved scoring. "
            "Unvisited tasks are not detector misses; low observed coverage does not establish "
            "low full-corpus coverage.")
    if support["task_support_comparison_basis"] == "estimated":
        result["caveats"].append(
            "The provisional task-support grade compares estimated full-corpus matched "
            "failed tasks with the five-task floor. This extrapolates sampled task matches using "
            "the sampling design, accounting for benchmark strata and repeated rollouts; "
            "it is not an observed count or a confidence bound. Final qualification still "
            "needs five observed distinct failed-task matches. Detector adherence uses the "
            "reviewed subset separately; unreviewed matches are not confirmations.")
    if panel:
        result["caveats"].append(
            f"Breadth advice uses a fixed {panel['n_items']}-example scoring panel; compare "
            "candidates only when cohort_fingerprint matches. Corpus task coverage above "
            "remains separate; panel coverage is not an estimate of full-corpus task coverage.")
    if status in {"limited_evidence", "reused_evidence", "no_fresh_evidence"}:
        result["caveats"].append(
            "Available current reviews remain limited or reused evidence; they do not establish "
            "fresh independent validation.")

    diagnostics: list[str] = []

    def recommend(state: str, *steps: str) -> dict[str, Any]:
        result.update(status=state, next_steps=[*steps, *diagnostics])
        if result["next_steps"]:
            result["next_steps"][0] += " " + qualification_summary
        else:
            result["next_steps"] = [qualification_summary]
        if grade_gap["summary"] not in qualification_summary:
            result["next_steps"][0] += " " + grade_gap["summary"]
        return result

    if scores.get("evaluated") is not True or scores.get("evaluation_stale"):
        return recommend("evaluate_current", "Evaluate the current detector before interpreting "
                         "historical association or task coverage as evidence about this revision.")
    if any(_positive(scores.get(key)) for key in ("errors", "parent_errors")):
        result["issue_types"] = ["detector_execution_failure"]
        result["error_examples"] = scores.get("error_examples") or []
        return recommend(
            "recover_evaluation", "Inspect the recorded detector or dependency execution errors. "
            "For parsing, schema or call/result-link failures, compare extraction with the raw "
            "source and repair the implementation, then smoke-test and reevaluate it. For "
            "transport or provider failures, restore the failed service and resume evaluation. "
            "Neither is evidence against the mechanism; do not change the claim or switch "
            "implementation type merely to hide an error.")
    if _positive(scores.get("n_insufficient")):
        result["issue_types"] = ["detector_abstention"]
        diagnostics.append(
            "Inspect abstained examples for an extraction limit versus "
            "facts genuinely absent from the source. Recover relevant action, consequence and "
            "later repair context where available; repair the extractor if it omitted them. "
            "Retain genuine unknowns: changing to an LLM cannot supply missing source evidence. "
            "Abstentions alone do not justify broadening or narrowing the predicate.")
        result["caveats"].append(
            "Association metrics describe resolved predictions. Abstained examples remain "
            "unknown; unresolved panel task coverage is an observed lower bound, not evidence "
            "that abstained tasks are detector misses.")
        # A partial evaluation can still guide investigation from resolved evidence.
        # No such evidence exists when every prediction abstained, and historical
        # metrics cannot establish it when the resolved count is unavailable.
        if not _positive(scores.get("n_scored")):
            return recommend("inspect_evidence")
    if scores.get("eval_truncated") and not scores.get("sampled"):
        return recommend("inspect_evaluation_scope", "Inspect why the intended evaluation was "
                         "truncated and complete interrupted work if appropriate. Keep unvisited "
                         "tasks distinct from observed detector misses.")
    if status == "skipped_pre_filter":
        return recommend("inspect_association", "Inspect the score pre-filter reason and actual "
                         "matched and missed examples before changing the detector. Saved judge "
                         "scores are historical for this evaluation; reevaluate after any edit.")
    if validation.get("review_schedule"):
        result["review_schedule"] = validation["review_schedule"]
        return recommend("review_deferred", *validation["review_schedule"]["next_steps"])
    if status in {"stale", "legacy", "unjudged", "invalid", "no_fresh_evidence"}:
        return recommend("review_current", f"Obtain a current {strictness} claim review "
                         "before treating "
                         "purity as support for generalization. Preserve evidence and budget "
                         "limits; historical or missing judgments are not poor-quality scores.")
    rows = scores.get("behavior_verdicts") or []
    incomplete = any(isinstance(row, Mapping)
                     and row.get("completion_status") == "needs_evidence_review" for row in rows)
    if (incomplete or status in {"incomplete", "uncertain", "error"}
            or any(_positive(validation.get(key)) for key in
                   ("uncertain", "processing_errors", "review_incomplete"))):
        issues = review_issue_types(dict(scores), validation)
        if incomplete and "review_incomplete" not in issues:
            issues.append("review_incomplete")
        result["issue_types"] = list(dict.fromkeys([*result["issue_types"], *issues]))
        return recommend(
            "inspect_review_evidence",
            "Follow the distinct per-case findings before interpreting aggregate purity; do not "
            "reset an unchanged claim's budget or treat uncertainty as a false hit.",
            *(review_issue_guidance(issue, behavior_strictness=strictness) for issue in issues),
        )
    if precision is None or purity is None:
        return recommend("inspect_scores", "Recover current precision and behavioral purity "
                         "measurements before choosing a score-driven revision.")

    actions = []
    if purity < required_purity:
        result["issue_types"] = list(dict.fromkeys([
            *result["issue_types"], *review_issue_types(dict(scores), validation),
        ]))
        actions.append(
            "Inspect rejected claim requirements, competing interpretations and relevant "
            "later recovery. Check whether the detector mixes mechanisms or mismatches its "
            "stated predicate before revising or splitting. Do not blindly narrow the claim "
            "or waive its required conditions. A known counterexample is not merely a small "
            "sample problem: inspect get_function_feedback/get_function_repair and test "
            "the evidenced distinction against that case and supported positives before "
            "requesting more reviews. More favorable cases do not repair the mismatch. "
            + review_policy_guidance(strictness))
    if precision < required_precision:
        actions.append(
            "Inspect flagged passing tasks and relevant missed failures to identify evidenced "
            "boundaries. Check whether the mechanism occurred and was recovered or incidental "
            "before sharpening the predicate; benchmark labels alone are not mechanism labels. "
            "Either repair false matches while preserving breadth, or specialize the claim and "
            "predicate to a coherent subtype, accepting lower coverage for higher precision. "
            "Reevaluate on the same panel and inspect lost supported matches; do not add "
            "task-specific exceptions.")
    if actions:
        actions[-1] += (
            " Test the evidenced distinction with smoke_test_feature while rechecking supported "
            "positives across tasks. For a parent, pair seed support and supported new variants "
            "with similar traces whose surface cues or unresolved outcome occur without the "
            "claimed behavior. Repair that distinction without task-specific "
            "exceptions or automatically narrowing the family. These author controls use the "
            "existing workflow and do not establish independent validation.")
        return recommend(
            "inspect_causal_claim" if purity < required_purity else "inspect_boundaries", *actions,
        )
    if required_support is not None and support_comparison is None:
        diagnostics.append(
            "Source-verified distinct failed-task support is unavailable. Recover task "
            "identities and the current detector's saved matches before comparing its breadth "
            "with the five-task floor. Additional judge reviews do not repair missing task "
            "identity or evaluation provenance.")
    elif required_support is not None and support_comparison < required_support:
        if limited_corpus:
            diagnostics.append(
                f"Provisional Good requires estimated matches across {MIN_GOOD_FAILED_TASKS} "
                "distinct failed tasks, or that many observed task matches. Current support "
                "does not establish that floor, but this limited scan is not "
                "evidence of narrow full-corpus coverage. Complete the survivor's planned final "
                "evaluation at the existing final-validation handoff; do not request extra "
                "judge reviews solely to fill the task-breadth requirement.")
        else:
            diagnostics.append(
                f"Good also requires matches across {MIN_GOOD_FAILED_TASKS} distinct "
                "failed tasks. This candidate currently matches fewer tasks: keep its supported "
                "narrow seed. To improve its task support, investigate broader variants or "
                "synthesis only where a coherent mechanism is supported by the evidence. "
                "Investigating a distinct useful mechanism is also an option. Reviewing repeated "
                "rollouts of those same tasks cannot satisfy the support requirement.")
    elif (required_support is not None
          and support["task_support_comparison_basis"] == "estimated"):
        diagnostics.append(
            f"Estimated full-corpus matched task support meets the {MIN_GOOD_FAILED_TASKS}-task "
            "floor for provisional grading. Preserve this candidate for the planned final "
            "full-corpus evaluation. Fewer than five currently observed matches alone do not "
            "justify broadening its predicate. Detector adherence is assessed on the reviewed "
            "subset; do not require five judge confirmations for the task-breadth floor.")
    if breadth is None:
        if corpus_breadth is not None and limited_corpus:
            return recommend(
                "inspect_evaluation_scope", "Comparable panel breadth is unavailable. Recover "
                "the fixed scoring panel before judging breadth; do not broaden a detector "
                "merely because its limited scan observed few of all corpus tasks.")
        return recommend("inspect_task_identity", "Establish task coverage from verified source "
                         "identities before judging breadth. Attempt-level recall cannot show "
                         "that the detector generalizes across different tasks.")
    return recommend(
        "choose_next_investigation",
        "Current normalized precision and reviewed detector adherence meet the "
        f"{quality_floor} thresholds. "
        "Retain this supported detector. Choose the next investigation from the evidence: "
        "a supported extension across different tasks, a distinct useful mechanism, or repair "
        "of a concrete quality issue. "
        "Related supported seeds may suggest synthesis; overlapping or subset detectors are "
        "useful when they add a distinct supported explanation or remedy. Report those "
        "discoveries and quality improvements separately from additional confirmed task "
        "coverage; equivalent duplicates add neither. If extending or synthesizing, preserve "
        "supported seed matches and independently review new task matches; do not broaden by "
        "joining unrelated causes.",
    )
