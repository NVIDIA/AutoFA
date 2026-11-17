# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared persistence of judge results for discovery and final report review."""

from __future__ import annotations

import math
from copy import deepcopy

from autofa.metrics.judge_backend import backend_name


def apply_review_result(scores: dict, result: dict, *, judge_harness: str) -> None:
    """Update causal metrics and provenance without touching association measurements."""
    from autofa.metrics.review_assessments import retain_assessment

    retain_assessment(scores)
    scores.pop("final_review", None)
    has_judgment = result.get("backend") != "heuristic" and bool(
        result.get("n_completed", result.get("n_judged"))
    )
    purity = float(result["behavior_purity"]) if has_judgment else None
    specificity = result.get("behavior_specificity")
    specificity = (
        float(specificity)
        if has_judgment
        and type(specificity) in (int, float)
        and 0 <= specificity <= 1
        and math.isfinite(specificity)
        else None
    )
    scores["behavior_purity"] = purity
    scores["behavior_specificity"] = specificity
    if result.get("specificity_rationale"):
        scores["specificity_rationale"] = result.get("specificity_rationale")
    scores["judge_rationale"] = result.get("rationale")
    scores["judge_backend"] = result.get("backend", backend_name(judge_harness))
    scores["judge_harness"] = result.get("judge_harness", judge_harness)
    scores["behavior_n_judged"] = result.get("n_judged")
    scores["behavior_n_yes"] = result.get("n_yes")
    scores["behavior_n_no"] = result.get("n_no")
    scores["behavior_n_uncertain"] = result.get("n_uncertain")
    scores["behavior_n_completed"] = result.get("n_completed")
    from autofa.metrics.review_failures import review_failures

    scores["behavior_n_errors"] = result.get("n_errors", 0)
    scores["behavior_processing_failures"] = review_failures(result.get("verdicts") or [])
    scores["behavior_n_review_incomplete"] = result.get("n_review_incomplete", 0)
    historical = list(scores.get("behavior_verdict_history") or [])
    for row in scores.get("behavior_verdicts") or []:
        previous = row.get("historical_review") if isinstance(row, dict) else None
        if previous and previous not in historical:
            historical.append(deepcopy(previous))
    if historical:
        scores["behavior_verdict_history"] = historical
    scores["behavior_verdicts"] = result.get("verdicts") or []
    scores["judge_evidence_version"] = result.get("evidence_version")
    scores["judge_contract_version"] = result.get("judge_contract_version")
    scores["judge_model"] = result.get("judge_model")
    scores["judge_effort"] = result.get("judge_effort")
    scores["judge_thinking"] = result.get("judge_thinking")
    scores["judge_thinking_budget_tokens"] = result.get("judge_thinking_budget_tokens")
    scores["judge_rubric_hash"] = result.get("judge_rubric_hash")
    scores["judge_policy_hash"] = result.get("judge_policy_hash")
    scores["judge_objective_strategy"] = result.get("judge_objective_strategy")
    scores["judge_behavior_strictness"] = result.get("judge_behavior_strictness")
    scores["claim_hash"] = result.get("claim_hash")
    scores["claim_contract"] = result.get("claim_contract")
    scores["claim_compilation"] = result.get("claim_compilation")
    scores.pop("behavior_judgment_stale", None)
    if any(
        isinstance(row, dict)
        and (row.get("historical_review") or row.get("verification_status") == "historical")
        for row in scores["behavior_verdicts"]
    ):
        scores["behavior_evidence_stale"] = True
    else:
        scores.pop("behavior_evidence_stale", None)
