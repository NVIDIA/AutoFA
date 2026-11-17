# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep incomplete evidence out of plateau decisions."""

from __future__ import annotations

from collections.abc import Sequence


def incomplete_evidence(scores: dict, *, axes: Sequence[str] | None = None) -> bool:
    """Recognize both current execution errors and saved partial-evidence scores."""
    incomplete = (
        scores.get("evaluation_complete") is False
        or any(scores.get(key, 0) for key in (
            "errors", "parent_errors", "n_insufficient",
        ))
        or (scores.get("assessment_coverage") is not None
            and scores["assessment_coverage"] < 1)
    )
    if incomplete or axes is None:
        return incomplete
    from autofa.metrics.verdicts import behavior_ranking_eligibility

    return not behavior_ranking_eligibility(scores, axes)["eligible"]


def completed_association_rejection(scores: dict) -> bool:
    """A current census can finish a rejected experiment without a behavior review.

    This changes attempt accounting only. Behavior ranking must still exclude the
    unjudged candidate. Recheck the metrics so a saved skip cannot mask later errors.
    """
    from autofa.metrics.judge_prefilter import POLICY_VERSION, judge_prefilter

    saved = scores.get("judge_prefilter")
    if (not isinstance(saved, dict) or saved.get("policy_version") != POLICY_VERSION
            or saved.get("status") != "skip" or scores.get("judge_prefilter_stale")):
        return False
    fingerprint = saved.get("claim_hash")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
            or scores.get("evaluation_claim_hash") != fingerprint):
        return False
    measured = saved.get("measured")
    if not isinstance(measured, dict) or measured.get("coverage") != "census":
        return False
    decision = judge_prefilter(scores)
    return decision["status"] == "skip" and decision["measured"]["coverage"] == "census"


def completed_evaluation(snapshot: dict) -> bool:
    """Only comparable, fully assessed evaluations feed frontier statistics.

    Legacy snapshots without completeness metadata remain usable unless they
    explicitly record errors, sampling or a capped evaluation.
    """
    if incomplete_evidence(snapshot):
        return False
    if snapshot.get("eval_mode") == "capped" and not snapshot.get("evaluation_complete"):
        return False
    return all(
        not incomplete_evidence(result.get("scores") or {}, axes=snapshot.get("objective_axes"))
        and not result.get("sampled")
        for result in snapshot.get("results", [])
    )


def comparable_evaluations(history: Sequence[dict]) -> list[dict]:
    """Select completed evaluations without mutating archived snapshots."""
    return [snapshot for snapshot in history if completed_evaluation(snapshot)]
