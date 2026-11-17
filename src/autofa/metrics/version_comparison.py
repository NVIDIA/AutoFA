# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare preserved detector claims on identified saved evidence, without executing them."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.config import Config
from autofa.core.dependencies import review_source
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, _validate_contract, claim_hash
from autofa.metrics.discovery_grades import GRADE_CODES, grade_for_scores, grade_gap_for_scores
from autofa.metrics.live_progress import _evaluation_cohort
from autofa.metrics.repair_tradeoff import _review_sample, compare_repair_coverage
from autofa.metrics.verdicts import behavior_validation, current_behavior_scores
from autofa.objectives import discovery_strategy
from autofa.session import repair_workflow


def _identity(value: Any) -> str | None:
    return (hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
            .hexdigest() if value is not None else None)


def _resolved(scores: dict) -> bool:
    """Completeness within the actual scoring cohort, including a fixed sample."""
    cells = [scores.get(key) for key in ("a", "b", "c", "d")]
    n_items = scores.get("n_items")
    counts = [*cells, n_items]
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 for value in counts)
            or not n_items or not math.isclose(sum(cells), n_items)):
        return False
    if scores.get("sampled") is False and (
        int(n_items) != n_items
        or scores.get("n_scored", n_items) != n_items
    ):
        return False
    return bool(
        scores.get("evaluated") is True and not scores.get("evaluation_stale")
        and scores.get("evaluation_scope_complete") is True
        and not scores.get("eval_truncated")
        and scores.get("evaluation_complete") is not False
        and scores.get("dataset_evaluation_complete") is not False
        and scores.get("metrics_scope") not in {"resolved_items", "unrepresented_strata"}
        and scores.get("assessment_coverage", 1) == 1
        and all(scores.get(key, 0) == 0 for key in (
            "n_unresolved", "n_insufficient", "errors", "n_errors", "parent_errors",
        ))
    )


def version_summary(
    fingerprint: str, function: dict, current_version: str, *, config=None,
    run_id: str | None = None, target_grade: str | None = None,
) -> dict[str, Any]:
    config = config or Config()
    saved = function.get("scores") or {}
    expected_claim = claim_hash(
        source=review_source(function), description=function.get("description") or "",
        mechanism=function.get("mechanism") or function.get("description") or function["name"],
    )
    scores = current_behavior_scores(
        saved, source=review_source(function), description=function.get("description") or "",
        mechanism=function.get("mechanism") or function.get("description") or function["name"],
        objective_strategy=discovery_strategy(config),
        behavior_strictness=strictness_for_config(config),
    )
    contract = scores.get("claim_contract")
    contract_matches = False
    if isinstance(contract, dict):
        try:
            _validate_contract(contract)
            contract_matches = (
                contract.get("source_sha256")
                == hashlib.sha256(review_source(function).encode()).hexdigest()
                and contract.get("claim_hash", contract["contract_id"]) == expected_claim
            )
        except (KeyError, TypeError, ValueError):
            pass
    if scores.get("judge_contract_version") == JUDGE_CONTRACT_VERSION and not contract_matches:
        scores["behavior_judgment_stale"] = True
    evaluation_matches = bool(saved.get("evaluated") is True
                              and saved.get("evaluation_claim_hash") == expected_claim
                              and not saved.get("evaluation_stale"))
    current_run = run_id is None or saved.get("evaluation_run_id") == run_id
    if not evaluation_matches or not current_run:
        scores["evaluation_stale"] = True
    try:
        validation = behavior_validation(scores, include_details=False)
    except (TypeError, ValueError, OverflowError):
        validation = {"status": "invalid", "ranking_eligible": False}
    gap = grade_gap_for_scores(scores, target_grade=target_grade)
    dimensions = {dimension["key"]: dimension for dimension in gap["dimensions"]}
    cohort = _evaluation_cohort(scores) if evaluation_matches and current_run else None
    review_cohort = (_review_sample(scores) if validation["status"] not in {
        "stale", "invalid", "legacy", "unjudged",
    } else None)
    return {
        "version_hash": fingerprint, "current": fingerprint == current_version,
        "claim_hash": expected_claim,
        "description": function.get("description") or "",
        "mechanism": function.get("mechanism") or "",
        "metrics": {key: saved.get(key) for key in (
            "a", "b", "c", "d", "recall", "ppv_at_pi0", "behavior_purity",
            "n_items", "n_unresolved", "errors", "sampled", "evaluation_scope_complete",
            "evaluation_run_id", "behavior_n_judged", "behavior_n_yes", "behavior_n_no",
            "behavior_n_uncertain", "behavior_n_errors",
        )},
        "evaluation_matches_claim": evaluation_matches,
        "validation_status": validation.get("status"),
        "review_sample": {key: (saved.get("behavior_sample") or {}).get(key) for key in (
            "actual_n", "eligible_n", "reused_n", "limited", "reason",
        )},
        "grade": grade_for_scores(scores),
        "grade_gap": gap,
        "quality_metrics": {key: dimension["current"] for key, dimension in dimensions.items()},
        "task_support": {key: dimensions["matched_failed_tasks"][key]
                         for key in ("current", "observed", "estimated", "basis")},
        "evaluation": {
            "current": evaluation_matches and current_run, "current_run": current_run,
            "complete": _resolved(scores),
            "cohort_fingerprint": _identity(cohort),
            "corpus_fingerprint": saved.get("task_recall_corpus_fingerprint"),
            "scope": cohort[2] if cohort else "unidentified",
            "sampled": saved.get("sampled"), "provisional": gap["provisional"],
            "sample_n": saved.get("sample_n"), "sample_frame_n": saved.get("sample_frame_n"),
            "n_scored": saved.get("n_scored"), "metric_scope": gap["scope"],
        },
        "review": {
            "status": validation["status"], "ranking_eligible": validation["ranking_eligible"],
            "current_claim": contract_matches and validation["status"] not in {
                "stale", "invalid", "legacy", "unjudged",
            },
            "cohort_fingerprint": _identity(review_cohort),
            "claim_hash": saved.get("claim_hash"),
            "policy_hash": saved.get("judge_policy_hash"),
        },
    }


def _comparison(
    before: dict, after: dict, prior: dict, current: dict, *, config, run_id: str | None,
) -> dict[str, Any]:
    old_eval, new_eval = prior["evaluation"], current["evaluation"]
    same_cohort = bool(old_eval["cohort_fingerprint"]
                       and old_eval["cohort_fingerprint"] == new_eval["cohort_fingerprint"])
    # Parent-conditioned metrics need their own identified denominator. The
    # association cohort alone cannot establish that a changed parent is equal.
    same_scope = old_eval["metric_scope"] == new_eval["metric_scope"] == "corpus"
    old_scores, new_scores = before.get("scores") or {}, after.get("scores") or {}
    # Identity establishes the cohort; consistent outcome denominators are an
    # additional check against mislabeled complete or differently weighted data.
    same_denominators = all(
        isinstance(old_scores.get(left), (int, float))
        and isinstance(old_scores.get(right), (int, float))
        and isinstance(new_scores.get(left), (int, float))
        and isinstance(new_scores.get(right), (int, float))
        and math.isclose(old_scores[left] + old_scores[right],
                         new_scores[left] + new_scores[right])
        for left, right in (("a", "c"), ("b", "d"))
    )
    comparable = bool(same_cohort and same_scope and old_eval["complete"]
                      and new_eval["complete"] and old_eval["current"] and new_eval["current"]
                      and same_denominators)
    same_review = bool(prior["review"]["cohort_fingerprint"]
                       and prior["review"]["cohort_fingerprint"]
                       == current["review"]["cohort_fingerprint"])
    changes = {}
    for key in prior["quality_metrics"]:
        old_value, new_value = (row["quality_metrics"][key] for row in (prior, current))
        available = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) for value in (old_value, new_value))
        metric_comparable = comparable and (key != "behavior_purity" or same_review)
        if key == "matched_failed_tasks":
            metric_comparable = metric_comparable and (
                prior["task_support"]["basis"] == current["task_support"]["basis"]
                # Observed sample receipts can grow through targeted reviews,
                # independently of the fixed scoring panel.
                and (not old_eval["sampled"] or prior["task_support"]["basis"] == "estimated"))
        delta = new_value - old_value if available and metric_comparable else None
        changes[key] = {
            "before": old_value, "after": new_value, "delta": delta,
            "comparable": bool(available and metric_comparable),
            "direction": ("unavailable" if delta is None else "regression" if delta < 0
                          else "gain" if delta > 0 else "unchanged"),
        }
    old_grade, new_grade = prior["grade"]["code"], current["grade"]["code"]
    ranked = {"WEAK", "OK", "GOOD", "EXCELLENT", "OUTSTANDING"}
    grade_direction = "unavailable"
    if comparable and old_grade in ranked and new_grade in ranked:
        lower_grade = GRADE_CODES.index(new_grade) > GRADE_CODES.index(old_grade)
        grade_direction = ("unchanged" if old_grade == new_grade else
                           "regression" if lower_grade else "gain")
    reason = None if comparable else (
        "Compare current, fully resolved scores with the same identified association cohort "
        "and metric denominator. Equal counts or a shared projected breadth panel are "
        "insufficient; unavailable comparisons do not establish a regression."
    )
    coverage_evidence = None
    if comparable:
        evidence = compare_repair_coverage(before, after, config=config, run_id=run_id)
        coverage_evidence = {key: evidence[key] for key in (
            "comparison_complete", "reason", "n_removed_matches", "n_no_longer_observed_matches",
            "evidence_counts", "evidence", "requires_coverage_inspection",
        )}
    return {
        "prior_version_hash": prior["version_hash"],
        "current_version_hash": current["version_hash"],
        "comparable": comparable, "same_association_cohort": same_cohort, "reason": reason,
        "provisional": old_eval["provisional"] or new_eval["provisional"],
        "metric_changes": changes,
        "claim_grade": {"before": old_grade, "after": new_grade, "direction": grade_direction},
        "judge_comparison": {
            "same_sample_and_settings": same_review,
            "causal_improvement_established": False,
            "interpretation": (
                "Grades describe the evidence for each written claim. Different judge samples "
                "do not establish a numerical adherence gain or loss caused by the revision; "
                "even a matched-sample difference is descriptive."
            ),
        },
        "coverage_evidence": coverage_evidence,
    }


def compare_function_versions(
    before: dict, after: dict, *, config, run_id: str | None = None,
    target_grade: str | None = None,
) -> dict[str, Any]:
    """Read a provenance-aware comparison without scoring, judging or choosing a version."""
    current_version = repair_workflow._version(after)
    prior = version_summary(repair_workflow._version(before), before, current_version,
                     config=config, run_id=run_id, target_grade=target_grade)
    current = version_summary(current_version, after, current_version, config=config,
                       run_id=run_id, target_grade=target_grade)
    return _comparison(before, after, prior, current, config=config, run_id=run_id)


def version_recommendation(name: str, rows: list[dict], comparisons: list[dict],
                    current_version: str) -> dict[str, Any]:
    qualifying = []
    by_version = {row["version_hash"]: row for row in rows}
    for comparison in comparisons:
        prior = by_version[comparison["prior_version_hash"]]
        if (not comparison["comparable"] or prior["grade"]["code"] not in {
                "GOOD", "EXCELLENT", "OUTSTANDING",
        } or not prior["review"]["current_claim"]):
            continue
        directions = [change["direction"] for change in comparison["metric_changes"].values()]
        grade_regressed = (comparison["claim_grade"]["direction"] == "regression"
                           and "regression" in directions)
        dominated = (comparison["claim_grade"]["direction"] == "unchanged"
                     and "regression" in directions and "gain" not in directions)
        if grade_regressed or dominated:
            qualifying.append((prior, comparison))
    if qualifying:
        # Use shared grade order, then observed coverage to retain useful breadth.
        prior, comparison = min(qualifying, key=lambda pair: (
            GRADE_CODES.index(pair[0]["grade"]["code"]),
            -(pair[0]["quality_metrics"]["recall"] or 0), pair[0]["version_hash"],
        ))
        return {
            "action": "restore", "version_hash": prior["version_hash"],
            "reason": (
                "A preserved Good-or-better claim has current supporting provenance and "
                "the same complete scoring cohort; a comparable metric regressed and "
                "the current version has a lower claim grade or no offsetting measured gains. "
                "Inspect the listed changes and retained counterexamples, then explicitly "
                "select the supported version if it still fits the intended mechanism."
            ),
            "comparison": comparison,
            "tool": {"name": "restore_function_version", "arguments": {
                "name": name, "version_hash": prior["version_hash"],
                "expected_current_version_hash": current_version,
            }, "reason_required": True},
            "automatic_restore": False,
        }
    current = by_version.get(current_version)
    inspect = (not current or not current["evaluation"]["current"]
               or current["grade"]["code"] == "UNSCORED") or any(
        not comparison["comparable"] or any(
            change["direction"] == "regression"
            for change in comparison["metric_changes"].values()
        ) or comparison["claim_grade"]["direction"] in {"regression", "unavailable"}
        for comparison in comparisons
    )
    return {
        "action": "inspect" if inspect else "keep_current", "version_hash": current_version or None,
        "reason": (
            "The evidence is incomparable or shows a tradeoff without a supported dominant "
            "prior version. Inspect provenance, claim scope and the active objective before "
            "choosing; missing evidence is not a measured regression."
            if inspect else
            "No comparable regression establishes a better preserved Good-or-better version. "
            "Keep the current implementation while following its target-grade feedback."
        ),
        "automatic_restore": False,
    }
