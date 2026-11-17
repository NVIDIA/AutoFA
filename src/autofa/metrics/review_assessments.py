# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve separate review cohorts without pooling their judgments or scores."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from autofa.metrics.judge_checkpoints import fingerprint

_HISTORY = "behavior_review_assessments"
_EXTRA_FIELDS = {
    "claim_hash", "claim_contract", "claim_compilation", "claim_calibration",
    "item_calibrations", "calibration_hint", "specificity_rationale",
    "final_review", "round_review", "selection_review",
}


def assessment_scores(scores: dict) -> dict:
    """Copy only review provenance/results, never population metrics or other cohorts."""
    return deepcopy({key: value for key, value in scores.items()
                     if (key.startswith(("behavior_", "judge_")) or key in _EXTRA_FIELDS)
                     and key not in {_HISTORY, "behavior_verdict_history"}})


def assessment_record(scores: dict) -> dict:
    """The saved form of one cohort's results, which also defines its assessment_id."""
    saved = assessment_scores(scores)
    if not (saved.get("behavior_evidence_stale") and any(
            isinstance(row, dict) and (row.get("historical_review")
                                       or row.get("verification_status") == "historical")
            for row in saved.get("behavior_verdicts") or [])):
        # A pending representative review marks only the displayed aggregate stale.
        # The saved cohort keeps what its own rows establish.
        saved.pop("behavior_evidence_stale", None)
    return saved


def retain_assessment(scores: dict) -> None:
    """Archive the current assessment before replacement, including negative findings."""
    if not scores.get("behavior_verdicts") or not scores.get("claim_hash"):
        return
    saved = assessment_record(scores)
    identity = fingerprint(saved)
    history = list(scores.get(_HISTORY) or [])
    if not any(row.get("assessment_id") == identity for row in history):
        history.append({"assessment_id": identity, "scores": saved})
    scores[_HISTORY] = history


def archived_assessments(scores: dict) -> list[dict]:
    return [row["scores"] for row in scores.get(_HISTORY) or []
            if isinstance(row, dict) and isinstance(row.get("scores"), dict)]


def restore_assessment(scores: dict, saved: dict) -> None:
    """Select one whole assessment; keep other cohorts and population scores separate."""
    retain_assessment(scores)
    for key in assessment_scores(scores):
        scores.pop(key, None)
    scores.update(assessment_scores(saved))


def cohort_ids(scores: dict) -> set[str]:
    return {row["item_id"] for row in scores.get("behavior_verdicts") or []
            if isinstance(row, dict) and isinstance(row.get("item_id"), str)}


def completed_assessment(scores: dict) -> bool:
    """Substantive uncertainty is complete; processing/source failures are not."""
    rows = scores.get("behavior_verdicts") or []
    return bool(rows) and not any(scores.get(key) for key in (
        "behavior_evidence_stale", "behavior_judgment_stale", "behavior_n_errors",
        "behavior_n_review_incomplete",
    )) and all(isinstance(row, dict) and row.get("verdict") in {"yes", "no", "uncertain"}
               and row.get("completion_status") != "needs_evidence_review"
               and row.get("processing_status") != "error" for row in rows)


def _approved_contract(contract: dict) -> bool:
    compilation = contract.get("compilation") or {}
    return bool(contract.get("defining_clauses")) and (
        (compilation.get("coverage_review") or {}).get("approved") is True)


def stale_compiled_contract(contract: Any) -> bool:
    """An approved contract frozen under another compiler policy must be recompiled."""
    from autofa.metrics.claim_compiler import compiler_policy_hash
    from autofa.metrics.claim_contract import contract_behavior_strictness

    if not isinstance(contract, dict) or not _approved_contract(contract):
        return False
    try:
        current = compiler_policy_hash(contract_behavior_strictness(contract))
    except ValueError:
        return True
    return (contract.get("compilation") or {}).get("policy_hash") != current


def reusable_contract(scores: dict) -> dict | None:
    """A failed compiler's base contract is an identity, not an approved rubric.

    A contract frozen under another compiler policy is recompiled, not reused.
    """
    contract = scores.get("claim_contract") or {}
    if not _approved_contract(contract) or stale_compiled_contract(contract):
        return None
    return contract


def assessment_summary(scores: dict) -> dict[str, Any]:
    """Expose a requested historical cohort separately from the active assessment."""
    ids = sorted(cohort_ids(scores))
    sample = {key: value for key, value in (scores.get("behavior_sample") or {}).items()
              if key in {"phase", "target_n", "actual_n", "limited", "scan_scope", "n_scanned",
                         "independence_scope", "selection", "population_n"}}
    return {"assessment_id": fingerprint(assessment_record(scores)),
            "item_ids": ids[:20], "n_items": len(ids), "item_ids_truncated": len(ids) > 20,
            "behavior_purity": scores.get("behavior_purity"),
            "n_yes": scores.get("behavior_n_yes"), "n_no": scores.get("behavior_n_no"),
            "n_uncertain": scores.get("behavior_n_uncertain"),
            "n_errors": scores.get("behavior_n_errors", 0),
            "n_review_incomplete": scores.get("behavior_n_review_incomplete", 0),
            "complete": completed_assessment(scores),
            "sample": sample,
            "review_paths": list(dict.fromkeys(row["review_path"] for row in
                                               scores.get("behavior_verdicts") or []
                                               if row.get("review_path")))[:5]}
