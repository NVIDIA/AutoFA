# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provisional discovery order, separate from validated ranking and stopping gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.config import resolve_objective_axes
from autofa.core.dependencies import review_source
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
from autofa.metrics.pareto import _scalar_score
from autofa.metrics.search_association import search_association
from autofa.metrics.search_panel import current_search_panel
from autofa.metrics.verdicts import current_behavior_scores, judge_prefilter_skipped
from autofa.objectives import discovery_strategy, weights_by_axis

CAVEAT = (
    "Provisional search guidance from current observed evidence, including limited or reused "
    "development reviews. Uncertain reviews stay in the purity denominator. This is not "
    "independent validation, keeper eligibility, or confirmed task-coverage progress. Sampled "
    "corpus task recall remains an observed lower bound. Iterative task breadth uses "
    "the same fixed scoring panel across detector engines, never mixed evaluation scopes; "
    "repeated rollouts never add task breadth. "
    "When association_scope is worst_case_abstentions, abstained failed attempts count as "
    "misses and abstained passing attempts as false positives."
)


def _number(value: Any, *, count: bool = False) -> float | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0
            or (count and int(value) != value)):
        return None
    return float(value)


def _review_evidence(scores: dict, fingerprint: str) -> dict | None:
    """Count exact current per-case outcomes; aggregate purity alone is not evidence."""
    contract, rows = scores.get("claim_contract"), scores.get("behavior_verdicts")
    if (scores.get("judge_contract_version") != JUDGE_CONTRACT_VERSION
            or scores.get("claim_hash") != fingerprint or scores.get("behavior_judgment_stale")
            or scores.get("behavior_evidence_stale")
            or scores.get("judge_backend") == "heuristic"
            or not isinstance(contract, dict) or not contract.get("contract_id")
            or contract.get("claim_hash", contract["contract_id"]) != fingerprint
            or not isinstance(rows, list) or not rows):
        return None
    keys = ("behavior_n_judged", "behavior_n_completed", "behavior_n_yes", "behavior_n_no",
            "behavior_n_uncertain", "behavior_n_errors")
    counts = [_number(scores.get(key), count=True) for key in keys]
    if any(value is None for value in counts):
        return None
    total, completed, yes, no, uncertain, errors = counts
    if (total != len(rows) or yes + no + uncertain != total or errors
            or completed != yes + no or not completed):
        return None
    sample = scores.get("behavior_sample") or {}
    if (not isinstance(sample, dict)
            or ("actual_n" in sample and _number(sample["actual_n"], count=True) != total)
            or _number(sample.get("target_n", total), count=True) is None
            or _number(sample.get("reused_n", 0), count=True) is None):
        return None
    observed, ids, unfinished = {"yes": 0, "no": 0, "uncertain": 0}, set(), 0
    for row in rows:
        if not isinstance(row, dict):
            return None
        item_id, verdict = row.get("item_id"), row.get("verdict")
        if (not isinstance(item_id, str) or not item_id or item_id in ids
                or not isinstance(verdict, str) or verdict not in observed
                or row.get("validation_error")
                or row.get("processing_status") == "error"
                or row.get("completion_status") not in {"complete", "needs_evidence_review"}
                or row.get("claim_contract_id") != contract["contract_id"]
                or not row.get("snapshot_id")):
            return None
        ids.add(item_id)
        observed[verdict] += 1
        unfinished += row["completion_status"] == "needs_evidence_review"
        if verdict != "uncertain" and row["completion_status"] != "complete":
            return None
        if verdict == "yes":
            initial = row.get("initial_review") or {}
            if (row.get("verification_status") != "confirmed" or not isinstance(initial, dict)
                    or initial.get("verdict") != "yes" or initial.get("item_id") != item_id
                    or initial.get("claim_contract_id") != contract["contract_id"]
                    or initial.get("snapshot_id") != row["snapshot_id"]):
                return None
    if ([observed[key] for key in ("yes", "no", "uncertain")] != [yes, no, uncertain]
            or _number(scores.get("behavior_n_review_incomplete", 0), count=True) != unfinished):
        return None
    return {"reviewed": int(total), "confirmed": int(yes), "rejected": int(no),
            "uncertain": int(uncertain), "purity_lower_bound": yes / total,
            "limited": bool(sample.get("limited")), "reused_n": sample.get("reused_n", 0),
            "phase": sample.get("phase"), "target_n": sample.get("target_n", int(total))}


def search_score(function: Any, config: Any, *, run_id: str | None = None) -> dict[str, Any]:
    """Score current research evidence without promoting a function into validated ranking."""
    data = function if isinstance(function, Mapping) else vars(function)
    axes = resolve_objective_axes(config)
    result = {"eligible": False, "status": "unavailable", "score": None,
              "objective_values": {}, "evidence": None, "provisional": True,
              "caveat": CAVEAT}

    def unavailable(reason: str) -> dict:
        return {**result, "reason": reason}

    source = review_source(data)
    description = data.get("description") or ""
    mechanism = data.get("mechanism") or description or data.get("name") or ""
    fingerprint = claim_hash(source=source, description=description, mechanism=mechanism)
    try:
        scores = current_behavior_scores(
            data.get("scores"), source=source, description=description, mechanism=mechanism,
            objective_strategy=discovery_strategy(config),
            behavior_strictness=strictness_for_config(config),
        )
    except (TypeError, ValueError):
        return unavailable("Saved evaluation or review metadata is malformed.")
    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or scores.get("evaluation_claim_hash") != fingerprint
            or (run_id is not None and scores.get("evaluation_run_id") != run_id)):
        return unavailable("No current evaluation of this exact detector and claim.")
    try:
        association = search_association(scores)
    except (TypeError, ValueError, OverflowError):
        association = None
    if association is None:
        return unavailable("Selected evaluation is incomplete, errored, or cannot conservatively "
                           "account for abstentions. Existing evidence remains inspectable.")
    if judge_prefilter_skipped(scores):
        return unavailable("The current association pre-filter rejected this candidate.")
    try:
        evidence = _review_evidence(scores, fingerprint)
    except (TypeError, ValueError, OverflowError):
        evidence = None
    if evidence is None:
        return unavailable("Current, consistent per-case claim reviews are unavailable, stale, "
                           "unjudged, or contain processing errors.")
    result["evidence"] = evidence
    result["association_scope"] = association.get("search_association_scope", "observed_scores")
    result["evaluation_sampled"] = bool(scores.get("sampled"))
    result["task_recall_scope"] = scores.get("task_recall_scope")
    if "task_recall" in axes:
        panel = current_search_panel(scores)
        if panel is not None:
            association["task_recall"] = panel["task_recall"]
            result["task_recall_scope"] = panel["scope"]
            result["breadth_cohort"] = {
                key: panel[key] for key in (
                    "scope", "cohort_fingerprint", "corpus_fingerprint", "n_items",
                    "n_failed_tasks", "n_matched_failed_tasks", "observed_lower_bound",
                )
            }
        else:
            return unavailable(
                "No current comparable scoring panel. Evaluate this detector on the fixed "
                "panel with the current settings before comparing breadth; saved corpus "
                "coverage remains inspectable. Existing evaluation checkpoints can be reused."
            )
    if association.get("search_abstention_n"):
        result["abstention_n"] = association["search_abstention_n"]
    values = {}
    for axis in axes:
        value = (evidence["purity_lower_bound"] if axis == "behavior_purity"
                 else association.get(axis))
        if axis in {"precision", "raw_ppv"}:
            value = association.get(axis, association.get("raw_ppv", association.get("precision")))
        if axis == "recall" and value is None:
            value = association.get("sensitivity")
        value = _number(value)
        if value is None or (axis in {"ppv_at_pi0", "task_recall", "recall", "sensitivity",
                                     "behavior_purity", "behavior_specificity",
                                     "precision", "raw_ppv", "fpr", "pct_caught"} and value > 1):
            return unavailable(f"The active objective {axis!r} has no current measured value.")
        values[axis] = value
    weights = weights_by_axis(config)
    scalar = _scalar_score(list(values.values()),
                           weights=[weights.get(axis, 1.0) for axis in axes])
    return {**result, "eligible": True, "status": "provisional", "score": scalar,
            "objective_values": values, "reason": "Current evidence can guide exploration."}


def search_ranking(functions: Sequence[Any], config: Any, *, run_id: str | None = None) -> dict:
    """Order provisional candidates separately from display and validated rankings."""
    axes = resolve_objective_axes(config)
    ranked, unranked = [], []
    for fn in functions:
        name = fn.get("name") if isinstance(fn, Mapping) else fn.name
        row = {"name": name, **search_score(fn, config, run_id=run_id)}
        row.pop("caveat")  # The shared caveat is returned once for the entire order.
        (ranked if row["eligible"] else unranked).append(row)
    if "task_recall" in axes and len({
        row["breadth_cohort"]["cohort_fingerprint"] for row in ranked
    }) > 1:
        # Do not pick whichever panel makes a candidate look best, nor compare
        # breadth measured on different scoring panels.
        for row in ranked:
            row.update(
                eligible=False, status="incomparable_cohorts", score=None,
                reason="Candidates use different breadth cohorts. Evaluate them with the same "
                       "fixed-panel settings before comparing search scores; existing "
                       "evaluation checkpoints can be reused.",
            )
        unranked.extend(ranked)
        ranked = []
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return {"ranked": ranked, "unranked": unranked, "objective_axes": axes, "caveat": CAVEAT}
