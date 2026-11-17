# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current validation facts for repair handoffs and deterministic closing summaries."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.metrics.claim_calibration import summarize_judge_feedback
from autofa.metrics.discovery_grades import grade_for_scores, grade_gap_for_scores
from autofa.metrics.verdicts import verdict_for_scores
from autofa.reports.ordering import rank_display_functions
from autofa.util.atomic import atomic_write_json

_METRICS = {"recall": "coverage", "ppv_at_pi0": "normalized_precision",
            "behavior_purity": "adherence"}
_SCOPE_LABELS = {"full_dataset": "full scan", "sampled": "sampled estimate",
                 "partial": "partial scan", "not_evaluated": "not evaluated",
                 "unknown": "scan scope unknown"}


def evaluation_scope(scores: dict) -> str:
    """Absence of a sample flag alone is not evidence of a complete corpus scan."""
    if scores.get("evaluated") is not True or scores.get("evaluation_stale"):
        return "not_evaluated"
    if scores.get("sampled"):
        return "sampled"
    if scores.get("eval_truncated") or scores.get("evaluation_scope_complete") is False:
        return "partial"
    if scores.get("evaluation_scope_complete") is True and scores.get("sampled") is False:
        return "full_dataset"
    return "unknown"


def _saved_baseline(rt: Any) -> dict:
    try:
        receipt = json.loads((rt.run.run_dir() / "round_validation.json").read_text())
    except (OSError, ValueError):
        return {}
    state = receipt.get("refinement", {}) if isinstance(receipt, dict) else {}
    return state if isinstance(state, dict) and state.get("run_id") == rt.run.run_id else {}


def current_validation_summary(rt: Any) -> dict:
    """Read current claim/policy projections; never execute detectors or fetch traces."""
    from autofa.session.discovery_target import (
        agent_target_grade_for_runtime,
        discovery_good_target_for_runtime,
    )

    strictness = strictness_for_config(rt.config)
    rows = rank_display_functions(
        [vars(fn) for fn in rt.library.functions], behavior_strictness=strictness,
        review_history=rt.library.review_history,
    )
    baseline = _saved_baseline(rt)
    old_grades = {row["name"]: row["code"]
                  for row in (baseline.get("baseline_grades") or {}).get("functions", [])}
    target = discovery_good_target_for_runtime(rt)
    detectors = []
    for row in rows:
        scores = row["scores"]
        gap = grade_gap_for_scores(scores, target_grade="GOOD")
        # This projection hides stale/unavailable numbers using the same quality
        # requirements as ordinary feedback; no separate grading thresholds.
        values = {d["key"]: d["current"] for d in gap["dimensions"]}
        review = summarize_judge_feedback(scores, name=row["name"],
                                          behavior_strictness=strictness)
        controls = (baseline.get("controls") or {}).get(row["name"]) or {}
        old_scores = controls.get("initial_score_changes") or controls.get("score_changes") or {}
        changes = {}
        for metric, label in _METRICS.items():
            old = (old_scores.get(metric) or {}).get("before")
            current = values.get(metric)
            if old is not None:
                changes[label] = {"before": old, "after": current,
                                  "delta": current - old if current is not None else None}
        findings = []
        for finding in review.get("findings") or []:
            diagnosis = finding.get("diagnosis") or {}
            findings.append({
                "item_id": finding.get("item_id"), "verdict": finding.get("verdict"),
                "reason": finding.get("rationale"),
                "blocker": (finding.get("requirements") or [{}])[0],
                "next_action": diagnosis.get("next_action"),
            })
        keys = (["round_review", "selection_review", "final_review"] if scores.get("sampled")
                else ["final_review", "round_review", "selection_review"])
        receipt = {}
        # A saved receipt is historical when its claim/policy no longer applies.
        reviewed_ids = {r.get("item_id") for r in scores.get("behavior_verdicts") or []}
        for key in keys:
            candidate = scores.get(key) or {}
            policy_matches = candidate.get("policy_hash") == scores.get("judge_policy_hash")
            if (candidate.get("claim_hash") == scores.get("claim_hash")
                    and candidate.get("policy_hash")
                    and policy_matches
                    and set(candidate.get("item_ids") or []) == reviewed_ids
                    and review["status"] not in {"stale", "legacy", "unjudged"}):
                receipt = candidate
                break
        detectors.append({
            "name": row["name"], "grade": verdict_for_scores(scores)["code"],
            "agent_grade": grade_for_scores(scores)["code"],
            "previous_agent_grade": old_grades.get(row["name"]),
            **{label: values.get(metric) for metric, label in _METRICS.items()},
            "task_support": values.get("matched_failed_tasks"),
            "scope": evaluation_scope(scores), "sample_n": scores.get("sample_n"),
            "sample_frame_n": scores.get("sample_frame_n"),
            "resolved_n": scores.get("n_scored"),
            "abstentions": scores.get("n_insufficient", 0),
            "execution_errors": scores.get("errors", scores.get("n_errors", 0)),
            "blocking_dimensions": [d for d in gap["dimensions"] if d["met"] is not True],
            "score_changes": changes,
            "review": {**{key: review.get(key) for key in (
                "status", "sampled", "confirmed", "rejected", "uncertain",
                "processing_errors", "review_incomplete", "reason",
            )}, "completion": receipt.get("status"),
                "findings": findings},
            "details_action": {"tool": "get_function_feedback", "arguments": {"name": row["name"]}},
        })
    target_grade = agent_target_grade_for_runtime(rt)
    grade_order = ["GOOD", "EXCELLENT", "OUTSTANDING"]
    target_count = (sum(row["agent_grade"] in grade_order[grade_order.index(target_grade):]
                        for row in detectors) if target_grade in grade_order else None)
    return {
        "version": 1, "run_id": rt.run.run_id, "round": rt.run.round,
        "judge_policy": strictness, "source": "current_saved_library",
        "good_count": target["n_validated_failure_modes"],
        "requested_minimum": target["min_failure_modes"], "minimum_met": target["minimum_met"],
        "active_target_grade": target_grade, "active_target_count": target_count,
        "grade_counts": dict(Counter(row["grade"] for row in detectors)),
        "scope_counts": dict(Counter(row["scope"] for row in detectors)),
        "detectors": detectors,
    }


def render_validation_summary(summary: dict, *, for_agent: bool = False) -> str:
    """Use exact persisted measurements, with public grades capped at Good."""
    def rate(value: Any, *, percent: bool = False) -> str:
        return "unavailable" if value is None else (f"{value:.1%}" if percent else f"{value:.2f}")

    scopes = ", ".join(f"{count} {_SCOPE_LABELS[scope]}"
                       for scope, count in summary["scope_counts"].items())
    completion = summary.get("final_validation")
    target_line = (
        f"Active agent target: {summary['requested_minimum']} "
        f"{summary['active_target_grade'] or 'see pareto_status.target_grade'}; "
        f"currently {summary['active_target_count']} at that grade or better."
        if for_agent else
        f"{summary['good_count']}/{summary['requested_minimum']} Good-or-better detectors; "
        + ("minimum met." if summary["minimum_met"] else "minimum unmet.")
    )
    if completion and not for_agent:
        target_line = (
            f"{completion['n_validated_failure_modes']}/{summary['requested_minimum']} "
            "Good-or-better detectors with completed full-dataset validation; "
            + ("minimum met." if completion["minimum_met"] else "minimum unmet.")
        )
    lines = [
        "Current saved validation results (supersede earlier estimates and remembered scores):",
        target_line,
        "Evaluation scope: " + (scopes or "no detectors") + ".",
    ]
    for row in summary["detectors"]:
        scope = _SCOPE_LABELS[row["scope"]]
        if row["scope"] == "sampled" and row["sample_n"] is not None:
            scope += f" {row['sample_n']}/{row['sample_frame_n'] or '?'} attempts"
        grade = row["agent_grade"] if for_agent else row["grade"]
        grade = {"EMPTY": "NO MATCHES", "UNSCORED": "NOT EVALUATED"}.get(grade, grade)
        review = row["review"]
        lines.append(
            f"- {row['name']}: {grade}; Coverage {rate(row['coverage'], percent=True)}; "
            f"normalized precision {rate(row['normalized_precision'])}; "
            f"adherence {rate(row['adherence'])}; {scope}; "
            f"review {review['status']} ({review['confirmed']} confirmed, "
            f"{review['rejected']} rejected, {review['uncertain']} inconclusive, "
            f"{review['processing_errors']} errors)"
            + (f"; review completion: {review['completion']}" if review['completion'] else "")
            + f"; {row['abstentions']} abstentions, {row['execution_errors']} execution errors."
        )
        changes = ", ".join(f"{key.replace('_', ' ')} {rate(change['before'])} → "
                            f"{rate(change['after'])}"
                            for key, change in row["score_changes"].items())
        if changes:
            lines.append("  Since pre-validation baseline: " + changes + ".")
        blockers = ", ".join(d["label"] + (" unavailable" if d["current"] is None else
                             f" {d['current']:.3g} (requires {d['operator']} {d['required']})")
                             for d in row["blocking_dimensions"])
        if blockers:
            lines.append("  Good blockers: " + blockers + ".")
        if for_agent:
            for finding in review["findings"]:
                blocker = finding["blocker"]
                lines.append(f"  Review {finding['item_id']} ({finding['verdict']}): "
                             + str(blocker.get("rationale") or finding["reason"] or
                                   "Inspect the saved review for the claim requirement."))
            lines.append(f"  Full evidence: get_function_feedback(name={row['name']!r}).")
    if for_agent:
        lines.append("Base repairs and any closing summary on these current results. "
                     "A completed review may reject a claim or remain inconclusive. "
                     "Do not describe sampled/partial detectors as fully evaluated. "
                     "The runtime prints and saves the authoritative closing status "
                     "after validation.")
    return "\n".join(lines)


def save_validation_summary(rt: Any, *, completion: dict | None = None) -> dict:
    """Persist the detailed closing digest without dumping it into the CLI."""
    summary = current_validation_summary(rt)
    if completion is not None:
        summary["final_validation"] = completion
    atomic_write_json(rt.run.run_dir() / "validation_summary.json", summary)
    return summary
