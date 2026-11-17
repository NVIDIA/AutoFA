# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage and objective costs of a repair, using saved observations only."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.core.localisation import text_hash
from autofa.metrics.association import matches_selected_bucket, selected_bucket
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash, derive_claim_verdict
from autofa.metrics.pareto import _scalar_score
from autofa.metrics.verdicts import current_behavior_scores
from autofa.objectives import (
    OBJECTIVE_FIELDS,
    discovery_strategy,
    weights_by_axis,
)


def _current_scores(function: Mapping[str, Any], config) -> dict[str, Any]:
    return current_behavior_scores(
        function.get("scores"), source=review_source(function),
        description=function.get("description") or "",
        mechanism=function.get("mechanism") or function.get("description")
        or function.get("name") or "",
        objective_strategy=discovery_strategy(config),
        behavior_strictness=strictness_for_config(config),
    )


def _evaluation_current(function: Mapping[str, Any]) -> bool:
    scores = function.get("scores") or {}
    return bool(scores.get("evaluated") and not scores.get("evaluation_stale")
                and scores.get("evaluation_claim_hash") == claim_hash(
                    source=review_source(function), description=function.get("description") or "",
                    mechanism=function.get("mechanism") or function.get("description")
                    or function.get("name") or ""))


def _census(scores: Mapping[str, Any]) -> bool:
    return (scores.get("evaluation_scope_complete") is True
            and scores.get("sampled") is False and not scores.get("eval_truncated")
            and _count(scores.get("n_items")) is not None and scores["n_items"] > 0
            and not any(scores.get(key) for key in (
                "n_unresolved", "n_errors", "errors", "parent_errors", "n_insufficient"))
            and scores.get("evaluation_complete") is not False
            and scores.get("assessment_coverage", 1) == 1
            and scores.get("dataset_evaluation_complete") is not False)


def _positives(function: Mapping[str, Any]) -> set[str]:
    if not _evaluation_current(function):
        return set()
    scores = function.get("scores") or {}
    selection = selected_bucket(scores)
    source_hash = text_hash(review_source(function))
    return {
        item_id for item_id, row in (function.get("failure_localisations") or {}).items()
        if isinstance(item_id, str) and isinstance(row, dict) and row.get("item_id") == item_id
        and row.get("source_sha256") == source_hash
        and "assigned_label" in row and matches_selected_bucket(row["assigned_label"], selection)
    }


def _match_inventory_complete(scores: Mapping[str, Any], positives: set[str]) -> bool:
    count = scores.get("n_flagged")
    if count is None and all(_count(scores.get(key)) is not None for key in ("a", "b")):
        count = scores["a"] + scores["b"]
    return _count(count) == len(positives)


def _count(value: Any) -> int | None:
    """Association counts are serialized as floats, including exact census counts."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        return None
    return int(value)


def _review_sample(scores: Mapping[str, Any]) -> tuple | None:
    """Only comparable saved judge samples support a descriptive purity delta."""
    if (scores.get("behavior_judgment_stale")
            or scores.get("judge_contract_version") != JUDGE_CONTRACT_VERSION):
        return None
    rows = scores.get("behavior_verdicts") or []
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get("item_id"), str)
                       or not row["item_id"] for row in rows):
        return None
    item_ids = [row["item_id"] for row in rows]
    if len(set(item_ids)) != len(item_ids):
        return None
    keys = ("judge_model", "judge_harness", "judge_policy_hash")
    if any(not scores.get(key) for key in keys):
        return None
    return (tuple(sorted(item_ids)), *(scores.get(key) for key in keys),
            scores.get("judge_effort"), scores.get("judge_rubric_hash"),
            scores.get("judge_thinking"), scores.get("judge_thinking_budget_tokens"),
            scores.get("behavior_review_phase"))


def _reviewed_matches(scores: dict[str, Any], positives: set[str]) -> dict[str, str]:
    if (scores.get("behavior_judgment_stale")
            or scores.get("judge_contract_version") != JUDGE_CONTRACT_VERSION):
        return {}
    contract = scores.get("claim_contract")
    if (not isinstance(contract, dict) or not contract.get("contract_id")
            or contract.get("claim_hash", contract["contract_id"]) != scores.get("claim_hash")):
        return {}
    grouped: dict[str, list[dict]] = {}
    for row in scores.get("behavior_verdicts") or []:
        if (isinstance(row, dict) and isinstance(row.get("item_id"), str)
                and row["item_id"] in positives):
            grouped.setdefault(row["item_id"], []).append(row)
    reviewed = {}
    for item_id, rows in grouped.items():
        if len(rows) != 1:
            continue
        row = rows[0]
        if (row.get("processing_status") != "completed"
                or row.get("claim_contract_id") != contract["contract_id"]):
            continue
        try:
            derived = derive_claim_verdict(contract, row)
            if row.get("verdict") == "no" and derived is False:
                reviewed[item_id] = "no"
            initial = row.get("initial_review")
            if (row.get("verdict") == "yes" and derived is True
                    and row.get("verification_status") == "confirmed"
                    and isinstance(initial, dict) and initial.get("verdict") == "yes"
                    and initial.get("item_id") == item_id and row.get("snapshot_id")
                    and initial.get("snapshot_id") == row["snapshot_id"]
                    and initial.get("claim_contract_id") == contract["contract_id"]
                    and derive_claim_verdict(contract, initial) is True):
                reviewed[item_id] = "yes"
        except (ValueError, TypeError, KeyError):
            continue
    return reviewed


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and 0 <= value <= 1 else None


def _objective(scores: dict[str, Any], config) -> float | None:
    weights = {axis: weight for axis, weight in weights_by_axis(config).items() if weight > 0}
    if ("behavior_purity" in weights and (scores.get("behavior_judgment_stale")
            or scores.get("judge_contract_version") != JUDGE_CONTRACT_VERSION)):
        return None
    values = [_number(scores.get(axis)) for axis in weights]
    if any(value is None for value in values):
        return None
    return _scalar_score(values, weights=list(weights.values()))


def compare_repair_coverage(
    before: Mapping[str, Any], after: Mapping[str, Any], *, config,
    objective_settings: Mapping[str, Any] | None = None,
    failed_item_ids: Iterable[str] | None = None, run_id: str | None = None,
) -> dict[str, Any]:
    """Describe lost positives without assuming that every removed match was wrong.

    A missing positive record means a removed match only after a complete, resolved
    census of the same run. Old yes/no reviews describe the baseline claim, never
    ground truth for a rewritten claim. This diagnostic does not promote a repair.
    """
    if objective_settings:
        config = replace(config, **{key: objective_settings[key] for key in OBJECTIVE_FIELDS
                                    if key in objective_settings})
    old, new = _current_scores(before, config), _current_scores(after, config)
    old_current = _evaluation_current(before) and (
        run_id is None or old.get("evaluation_run_id") == run_id)
    new_current = _evaluation_current(after) and (
        run_id is None or new.get("evaluation_run_id") == run_id)
    current = old_current and new_current
    same_run = bool(old.get("evaluation_run_id")
                    and old.get("evaluation_run_id") == new.get("evaluation_run_id"))
    prior = _positives(before) if old_current else set()
    latest = _positives(after) if new_current else set()
    failures = set(failed_item_ids) if failed_item_ids is not None else None
    comparable = bool(current and same_run and _census(old) and _census(new)
                      and old.get("n_items") == new.get("n_items")
                      and _match_inventory_complete(old, prior)
                      and _match_inventory_complete(new, latest))
    valid_cells = all(_count(scores.get(key)) is not None
                      for scores in (old, new) for key in ("a", "b", "c", "d"))
    comparable = comparable and valid_cells and (
        old["a"] + old["c"], old["b"] + old["d"]) == (
        new["a"] + new["c"], new["b"] + new["d"])
    comparable = comparable and all(
        sum(scores[key] for key in ("a", "b", "c", "d")) == scores["n_items"]
        for scores in (old, new))
    if comparable and failures is not None:
        comparable = (old["a"] + old["c"] == len(failures)
                      and len(prior & failures) == old["a"]
                      and len(latest & failures) == new["a"])
    retained = prior & latest if current and same_run else set()
    absent = prior - latest if current and same_run else set()
    added = latest - prior if current and same_run else set()
    reviewed = _reviewed_matches(old, prior)
    removed = absent if comparable else set()
    groups = {
        "previously_confirmed_removed": sorted(i for i in removed if reviewed.get(i) == "yes"),
        "previously_rejected_removed": sorted(i for i in removed if reviewed.get(i) == "no"),
        "unresolved_or_unreviewed_removed": sorted(
            i for i in removed if reviewed.get(i) not in {"yes", "no"}),
        "previously_confirmed_retained": sorted(i for i in retained if reviewed.get(i) == "yes"),
        "removal_unknown": sorted(absent - removed),
    }
    metrics = {
        axis: {"before": _number(old.get(axis)) if old_current else None,
               "after": _number(new.get(axis)) if new_current else None}
        for axis in ("ppv_at_pi0", "recall", "task_recall", "behavior_purity")
    }
    for key, scores in (("before", old), ("after", new)):
        if (scores.get("behavior_judgment_stale")
                or scores.get("judge_contract_version") != JUDGE_CONTRACT_VERSION):
            metrics["behavior_purity"][key] = None
    for row in metrics.values():
        row["delta"] = (row["after"] - row["before"]
                        if comparable and row["before"] is not None and row["after"] is not None
                        else None)
    same_review_sample = (_review_sample(old) is not None
                          and _review_sample(old) == _review_sample(new))
    if not same_review_sample:
        metrics["behavior_purity"]["delta"] = None
    metrics["recall"]["scope"] = "Resolved predictions; inspect fixed-denominator coverage below."
    same_task_corpus = bool(old.get("task_recall_corpus_fingerprint")
                           and old.get("task_recall_corpus_fingerprint")
                           == new.get("task_recall_corpus_fingerprint"))
    task_comparable = same_task_corpus and all(
        scores.get("task_recall_scope") == "full_selected_corpus" for scores in (old, new)
    )
    if not task_comparable:
        metrics["task_recall"]["delta"] = None
    metrics["task_recall"]["scope"] = {
        "before": old.get("task_recall_scope"), "after": new.get("task_recall_scope"),
        "same_task_corpus": same_task_corpus,
    }
    old_value, new_value = (_objective(old, config), _objective(new, config))
    if not comparable:
        old_value = new_value = None
    active_weights = weights_by_axis(config)
    objective_comparable = comparable and (
        not active_weights.get("behavior_purity", 0) or same_review_sample) and (
        not active_weights.get("task_recall", 0) or task_comparable)
    delta = (new_value - old_value if objective_comparable and old_value is not None
             and new_value is not None else None)
    recall_drop = metrics["recall"]["delta"]
    task_recall_drop = metrics["task_recall"]["delta"]
    failure_coverage = None
    if failures is not None:
        prior_failed, latest_failed = prior & failures, latest & failures
        observed_comparable = current and same_run
        retained_failed = prior_failed & latest_failed if observed_comparable else set()
        absent_failed = prior_failed - latest_failed if observed_comparable else set()
        added_failed = latest_failed - prior_failed if observed_comparable else set()
        old_recall = len(prior_failed) / len(failures) if old_current and failures else None
        new_recall = len(latest_failed) / len(failures) if new_current and failures else None
        failure_coverage = {
            "n_failed": len(failures),
            "before": len(prior_failed) if old_current else None,
            "after": len(latest_failed) if new_current else None,
            "n_retained": len(retained_failed) if observed_comparable else None,
            "n_no_longer_observed": len(absent_failed) if observed_comparable else None,
            "n_newly_observed": len(added_failed) if observed_comparable else None,
            "before_recall_lower_bound": old_recall,
            "after_recall_lower_bound": new_recall,
            "observed_recall_delta": (new_recall - old_recall
                                      if observed_comparable and old_recall is not None
                                      and new_recall is not None else None),
            "exact": bool(comparable and old["a"] + old["c"] == len(failures)
                          and len(prior_failed) == old["a"]
                          and len(latest_failed) == new["a"]),
            "scored_failure_counts": {
                "before": _count(old.get("a")) if old_current else None,
                "after": _count(new.get("a")) if new_current else None,
            },
            "receipts_match_scored_failure_counts": (
                old_current and new_current and len(prior_failed) == old.get("a")
                and len(latest_failed) == new.get("a")),
            "evidence": {"no_longer_observed": sorted(absent_failed)[:10],
                         "newly_observed": sorted(added_failed)[:10],
                         "retained": sorted(retained_failed)[:10]},
            "interpretation": (
                "Current saved matches divided by the same dataset failure count. "
                "Saved match receipts may include observations beyond the latest scoring pass; "
                "compare scored_failure_counts before interpreting a discrepancy. "
                "Missing matches may be unresolved, errors or unassessed items; they are "
                "not proven negative predictions. These are coverage observations, not "
                "causal confirmations or an automatic version choice."
            ),
        }
    requires_inspection = bool(groups["previously_confirmed_removed"]
                               or (failure_coverage and failure_coverage["n_no_longer_observed"])
                               or (recall_drop is not None and recall_drop < 0)
                               or (task_recall_drop is not None and task_recall_drop < 0)
                               or (delta is not None and delta < 0))
    return {
        "comparison_complete": comparable,
        "reason": (None if comparable else
                   "Requires current, complete resolved censuses of the same run and cohort, "
                   "including all positive match records."),
        "metrics": metrics,
        "failure_coverage": failure_coverage,
        "evaluation_counts": {
            side: {key: scores.get(key) for key in ("a", "b", "c", "d", "n_items", "n_flagged")}
            for side, scores in (("before", old), ("after", new))
        },
        "uncertainty": {
            side: {key: scores.get(key) for key in (
                "n_unresolved", "errors", "n_insufficient", "sampled",
                "assessment_coverage", "recall_ci_low", "recall_ci_high",
                "unresolved_bounds", "behavior_n_judged", "behavior_n_uncertain",
                "behavior_n_errors",
            )} for side, scores in (("before", old), ("after", new))
        },
        "judge_comparison": {
            "same_sample_and_settings": same_review_sample,
            "causal_improvement_established": False,
            "interpretation": (
                "Different judge samples or settings do not establish a purity improvement. "
                "Even a matched-sample delta describes the two written claims; independent "
                "repair validation remains separate."
            ),
        },
        "objective": {"rank": config.objective_rank,
                      "weights": weights_by_axis(config),
                      "settings_frozen": bool(objective_settings),
                      "delta_comparable": bool(objective_comparable),
                      "delta_unavailable_reason": (
                          "The causal-purity scores use different or unidentified judge samples "
                          "or settings." if comparable and active_weights.get("behavior_purity")
                          and not same_review_sample else None),
                      "before": old_value, "after": new_value, "delta": delta,
                      "scope": "Observed scores; this is not independent repair validation."},
        "n_baseline_matches": len(prior) if old_current else None,
        "n_current_matches": len(latest) if new_current else None,
        "n_retained_matches": len(retained) if current and same_run else None,
        "n_removed_matches": len(removed) if comparable else None,
        "n_no_longer_observed_matches": len(absent) if current and same_run else None,
        "n_newly_observed_matches": len(added) if current and same_run else None,
        "evidence_counts": {key: len(value) for key, value in groups.items()},
        "evidence": {key: value[:10] for key, value in groups.items()},
        "requires_coverage_inspection": requires_inspection,
        "automatic_promotion": False,
        "next_step": (
            "Inspect removed confirmed examples as well as removed false positives. Compare "
            "the selected objective and keep a useful specific mechanism as a child of a "
            "supported parent. Preserve an unsupported parent as an open hypothesis; higher "
            "precision alone does not establish improvement."
            if requires_inspection else
            "Check the claim review and coverage comparison before choosing this version."
        ),
    }
