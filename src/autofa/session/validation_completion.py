# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify completion without changing the public detector-quality thresholds."""

from __future__ import annotations

from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.session.discovery_target import discovery_good_target_for_runtime


def _validation_problem(rt: Any, fn: Any, measured: dict) -> str | None:
    from autofa.mcp.tools_eval import (
        _review_settings,
        _review_settings_match,
        judge_evidence_sample_size,
    )
    from autofa.metrics.claude_judge import compatible_judge_policy, judge_policy_hash
    from autofa.metrics.final_review import COHORT_POLICY, _matches
    from autofa.metrics.final_validation_selection import _evaluation_problem
    from autofa.metrics.judge_checkpoints import fingerprint

    scores = fn.scores or {}
    if not measured.get("corpus_evaluation_complete"):
        return "Complete current full-dataset predictions before confirming this Good detector."
    if problem := _evaluation_problem(rt, fn):
        return problem
    receipt = scores.get("final_review") or {}
    policy = judge_policy_hash(behavior_strictness=strictness_for_config(rt.config))
    if (receipt.get("status") != "complete" or receipt.get("scope") != "full_corpus"
            or receipt.get("cohort_policy") != COHORT_POLICY
            or receipt.get("claim_hash") != scores.get("evaluation_claim_hash")
            or receipt.get("policy_hash") != policy
            or not compatible_judge_policy(
                scores.get("judge_policy_hash"),
                behavior_strictness=strictness_for_config(rt.config))
            or scores.get("behavior_evidence_stale")
            or scores.get("behavior_judgment_stale")
            or scores.get("behavior_n_errors") or scores.get("behavior_n_review_incomplete")):
        return "Finish the current representative full-dataset adherence review."
    _, _, settings = _review_settings(rt.config)
    if not _review_settings_match(scores, settings):
        return "Refresh the representative review using the current reviewer settings."
    try:
        labels = _matches(scores)
    except (TypeError, ValueError):
        return "Restore complete current match membership before confirming final validation."
    base_seed = getattr(rt.run, "exploration_base_seed", None)
    seed = int(base_seed if base_seed is not None else
               getattr(rt.run, "exploration_seed", 0) or 0)
    expected_identity = fingerprint({
        "claim": scores["evaluation_claim_hash"],
        "settings": {key: value for key, value in settings.items()
                     if key != "judge_objective_strategy"},
        "matches": {key: str(value) for key, value in labels.items()},
        "cohort_policy": COHORT_POLICY, "seed": seed, "scope": "full_corpus", "policy": policy,
    })
    cohort = receipt.get("item_ids") or []
    reviewed = [row.get("item_id") for row in scores.get("behavior_verdicts") or []]
    if (receipt.get("identity") != expected_identity or not receipt.get("operation_id")
            or not cohort or len(cohort) != len(set(cohort))
            or len(reviewed) != len(cohort) or set(reviewed) != set(cohort)
            or not set(cohort) <= labels.keys()
            or receipt.get("matched_n") != len(labels)
            or receipt.get("required_n") != judge_evidence_sample_size(len(labels))
            or len(cohort) != receipt.get("required_n")):
        return "Refresh representative review for the current full-dataset match population."
    return None


def validated_good_target_for_runtime(rt: Any) -> dict:
    """Apply the ordinary Good grade, then require its final measurements to exist.

    Discovery may display provisional Good estimates. Those labels remain useful,
    but cannot certify that final validation finished. This adds no precision,
    adherence, coverage, task-support or independent-confirmation threshold.
    """
    public = discovery_good_target_for_runtime(rt)
    functions = {fn.name: fn for fn in rt.library.functions}
    rows, names, blockers = [], [], []
    for row in public["failure_mode_eligibility"]:
        reason = row["reason"]
        if row["eligible"]:
            reason = _validation_problem(rt, functions[row["name"]], row["measured"])
            if reason:
                blockers.append({"name": row["name"], "reason": reason})
        rows.append({**row, "eligible": reason is None, "reason": reason})
        if reason is None:
            names.append(row["name"])
    minimum = public["min_failure_modes"]
    return {
        **public, "n_qualifying_failure_modes": len(names), "qualifying_failure_modes": names,
        "n_validated_failure_modes": len(names), "validated_failure_modes": names,
        "minimum_met": len(names) >= minimum,
        "remaining_failure_modes": max(0, minimum - len(names)),
        "failure_mode_eligibility": rows,
        "pending_failure_modes": [row for row in rows if not row["eligible"]],
        "provisional_good_count": public["n_validated_failure_modes"],
        "validation_blockers": blockers,
        "validation_scope": "current_full_corpus_and_representative_review",
    }
