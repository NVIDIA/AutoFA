# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select automatic census work without changing grades or population estimates.

A skipped candidate remains a sampled candidate, not a fully validated detector.
Receipts certify only that this exact scored version was intentionally deferred.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.judge_checkpoints import fingerprint
from autofa.metrics.verdicts import (
    current_behavior_scores,
    extract_metrics,
    metric_tier,
    verdict_for_scores,
)
from autofa.util.cancellation import cancellation_guard, check_cancelled

_SELECTION_VERSION = 1
CONTEXT_REFRESH_REASON = ("Evaluation source/settings changed or cannot be verified; "
                          "refresh checkpoint scoring.")
# Presentation rebuilds task-support fields. Bind the raw measurements and review
# that decide OK eligibility, not those derived fields or feedback formatting.
_SCORE_KEYS = (
    "evaluated", "evaluation_stale", "evaluation_scope_complete",
    "evaluation_checkpoint_complete", "eval_mode",
    "evaluation_claim_hash", "evaluation_run_id", "evaluation_context",
    "evaluation_execution_version", "evaluation_item_ids", "sampled", "sample_n",
    "sample_frame_n", "n_items", "n_scored", "n_unresolved", "n_flagged", "n_rows_scored",
    "errors", "n_errors", "n_insufficient", "evaluation_complete",
    "dataset_evaluation_complete", "parent_errors", "eval_truncated", "metrics_scope",
    "bucket", "buckets", "sampling_limitation", "strata", "unrepresented_strata",
    "recall", "ppv_at_pi0", "precision", "raw_ppv", "purity", "within_parent", "explains",
    "within_recall", "within_ppv_at_pi0", "within_n_flagged", "round_validation",
    "behavior_purity", "behavior_specificity", "behavior_n_judged", "behavior_n_yes",
    "behavior_n_no", "behavior_n_completed", "behavior_n_uncertain", "behavior_n_errors",
    "behavior_n_review_incomplete", "behavior_verdicts", "behavior_evidence_stale",
    "behavior_judgment_stale", "behavior_sample", "judge_backend",
    "claim_hash", "claim_contract", "judge_contract_version", "judge_policy_hash",
    "judge_behavior_strictness", "judge_evidence_version", "judge_prefilter",
    "selection_review", "round_review", "final_review",
)


def _score_fingerprint(scores: dict) -> str:
    values = {key: scores.get(key) for key in _SCORE_KEYS}
    # Claim projection may explicitly write False where the raw save omitted it.
    values["behavior_judgment_stale"] = bool(scores.get("behavior_judgment_stale"))
    return fingerprint(values)


def _claim(fn: Any) -> str:
    return claim_hash(source=review_source(fn), description=fn.description or "",
                      mechanism=fn.mechanism or fn.description or fn.name)


def display_final_validation_skip(
    scores: dict | None, *, source: str, description: str, mechanism: str,
    behavior_strictness: str | None = None,
) -> dict | None:
    """Read a source/score/policy-bound skip for saved report presentation.

    Runtime callers additionally verify the live corpus with
    :func:`current_final_validation_skip` before treating work as settled.
    """
    from autofa.metrics.claude_judge import judge_policy_hash

    s = scores or {}
    receipt = s.get("final_validation_selection")
    if not isinstance(receipt, dict):
        return None
    strictness = behavior_strictness or receipt.get("judge_behavior_strictness")
    if strictness not in {"behavioral", "contextual", "causal"}:
        return None
    if (receipt.get("version") != _SELECTION_VERSION or receipt.get("status") != "skipped"
            or receipt.get("claim_hash") != claim_hash(
                source=source, description=description, mechanism=mechanism)
            or receipt.get("judge_behavior_strictness") != strictness
            or receipt.get("policy_hash") != judge_policy_hash(behavior_strictness=strictness)
            or receipt.get("score_fingerprint") != _score_fingerprint(s)
            or not receipt.get("source_signature")
            or receipt.get("source_signature") != s.get("evaluation_context")
            or receipt.get("run_id") != s.get("evaluation_run_id")
            or s.get("evaluation_stale")):
        return None
    return dict(receipt)


def current_final_validation_skip(rt: Any, fn: Any) -> dict | None:
    """An intentional skip settles work only for the current dataset and claim."""
    receipt = display_final_validation_skip(
        fn.scores, source=review_source(fn), description=fn.description or "",
        mechanism=fn.mechanism or fn.description or fn.name,
        behavior_strictness=strictness_for_config(rt.config),
    )
    if receipt is None or receipt.get("run_id") != rt.run.run_id:
        return None
    return receipt if _evaluation_problem(rt, fn) is None else None


def _unfinished_selection(rt: Any, fn: Any) -> dict | None:
    """Finish an admitted census/review even when census scores downgrade it.

    Scoring changes the score fingerprint deliberately. The reservation instead
    freezes the candidate, corpus/settings, active policy and originating run.
    Final review closes it; a changed candidate never inherits that admission.
    """
    return _current_selection(rt, fn, statuses={"selected"})


def _current_selection(rt: Any, fn: Any, *, statuses: set[str]) -> dict | None:
    """Verify an admission against its current detector, source and review policy."""
    from autofa.metrics.claude_judge import judge_policy_hash

    receipt = (fn.scores or {}).get("final_validation_selection")
    strictness = strictness_for_config(rt.config)
    if (not isinstance(receipt, dict) or receipt.get("version") != _SELECTION_VERSION
            or receipt.get("status") not in statuses or receipt.get("claim_hash") != _claim(fn)
            or receipt.get("run_id") != rt.run.run_id
            or receipt.get("judge_behavior_strictness") != strictness
            or receipt.get("policy_hash") != judge_policy_hash(behavior_strictness=strictness)
            or not receipt.get("source_signature")
            or receipt.get("source_signature") != (fn.scores or {}).get("evaluation_context")
            or _evaluation_problem(rt, fn) is not None):
        return None
    return receipt


def _save_selection(rt: Any, fn: Any, *, status: str, reason: str, grade: str) -> dict:
    from autofa.metrics.claude_judge import judge_policy_hash

    strictness = strictness_for_config(rt.config)
    scores = dict(fn.scores or {})
    receipt = {
        "version": _SELECTION_VERSION, "name": fn.name, "status": status,
        "reason": reason, "grade": grade, "claim_hash": _claim(fn),
        "run_id": rt.run.run_id,
        "judge_behavior_strictness": strictness,
        "policy_hash": judge_policy_hash(behavior_strictness=strictness),
        "source_signature": scores.get("evaluation_context"),
        "corpus_fingerprint": scores.get("task_recall_corpus_fingerprint"),
        "score_fingerprint": _score_fingerprint(scores),
        "evaluation_scope": "sampled" if scores.get("sampled") else "full_dataset",
    }
    scores["final_validation_selection"] = receipt
    previous = fn.scores
    with cancellation_guard(), rt._library_lock:
        try:
            fn.scores = scores
            rt.persist_library()
        except BaseException:
            fn.scores = previous
            raise
    return receipt


def recover_final_validation_selection(
    rt: Any, *, run_id: str, names: list[str], claims: dict[str, str],
    source_signature: str, policy_hash: str,
) -> dict[str, Any]:
    """Restore legacy admissions proved by the caller's saved validation history.

    Recovery creates ordinary, finite selection receipts, not a lasting exemption
    from the current grade threshold. The migration must first persist its recovery
    plan; retries preserve matching admissions and never reopen completed ones.
    """
    from autofa.metrics.claude_judge import judge_policy_hash

    recovered, unresolved = [], []
    functions = {fn.name: fn for fn in rt.library.functions}
    active_policy = judge_policy_hash(behavior_strictness=strictness_for_config(rt.config))
    for name in dict.fromkeys(names):
        check_cancelled()
        fn = functions.get(name)
        reason = None
        if run_id != rt.run.run_id:
            reason = "Saved validation admission belongs to a different run."
        elif not policy_hash or policy_hash != active_policy:
            reason = "Review policy changed since the saved validation admission."
        elif fn is None:
            reason = "The previously admitted detector is no longer in the library."
        elif claims.get(name) != _claim(fn):
            reason = "The detector changed since the saved validation admission."
        elif not source_signature or source_signature != (fn.scores or {}).get(
            "evaluation_context"
        ):
            reason = "Evaluation source/settings differ from the saved validation admission."
        else:
            reason = _evaluation_problem(rt, fn)
        if reason:
            unresolved.append({"name": name, "reason": reason})
            continue
        current = _current_selection(rt, fn, statuses={"selected", "complete"})
        if current and current["status"] == "complete":
            continue
        if current is None:
            _save_selection(
                rt, fn, status="selected",
                grade=verdict_for_scores(_current_scores(rt, fn))["code"],
                reason="Resume the previously admitted interrupted full validation.",
            )
        recovered.append(name)
    return {"recovered_names": recovered, "unresolved": unresolved}


def _current_scores(rt: Any, fn: Any) -> dict:
    return current_behavior_scores(
        fn.scores, source=review_source(fn), description=fn.description or "",
        mechanism=fn.mechanism or fn.description or fn.name,
        behavior_strictness=strictness_for_config(rt.config),
    )


def _all_predictions_abstained(scores: dict) -> bool:
    """A finished evidence limitation is different from an unfinished/error scan."""
    abstained = scores.get("n_insufficient")
    return bool(
        checkpoint_scoring_complete(scores)
        and isinstance(abstained, (int, float)) and not isinstance(abstained, bool)
        and isfinite(abstained) and abstained > 0 and int(abstained) == abstained
        and scores.get("n_unresolved") == abstained
        and all(scores.get(key) == 0 for key in ("n_items", "n_scored", "n_flagged"))
        and not any(scores.get(key) for key in (
            "errors", "n_errors", "parent_errors",
        ))
    )


def checkpoint_scoring_complete(scores: dict) -> bool:
    """Recognize a finished requested frame without treating it as a census.

    Every frame needs the explicit completion receipt written only after its
    selected rows finish.
    """
    if scores.get("evaluation_scope_complete") is True:
        return scores.get("evaluation_checkpoint_complete") is True
    return bool(
        scores.get("eval_mode") == "capped"
        and scores.get("eval_truncated") is True
        and scores.get("evaluation_checkpoint_complete") is True
    )


def _evaluation_problem(rt: Any, fn: Any) -> str | None:
    from autofa.mcp.tools_eval import (
        DETECTOR_EXECUTION_VERSION,
        _evaluation_context,
        is_llm_detector,
    )

    scores = fn.scores or {}
    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or not checkpoint_scoring_complete(scores)
            or scores.get("evaluation_claim_hash") != _claim(fn)
            or scores.get("evaluation_run_id") != rt.run.run_id
            or (is_llm_detector(fn) and scores.get("evaluation_execution_version")
                != DETECTOR_EXECUTION_VERSION)):
        return "Complete current discovery/checkpoint scoring before selecting final validation."
    try:
        context = _evaluation_context(rt)
    except (OSError, ValueError, TypeError) as exc:
        return f"Current evaluation source cannot be verified: {exc}"
    if not context or scores.get("evaluation_context") != context:
        return CONTEXT_REFRESH_REASON
    return None


def prepare_final_validation(
    rt: Any, *, names: list[str] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Freeze automatic census candidates using current OK-or-better grades.

    Missing adherence on an otherwise OK candidate gets one representative review
    of its existing scored matches. No detector executes here. Frozen review
    cohorts/checkpoints resume normally; infrastructure failures remain pending.
    """
    from autofa.metrics.final_review import final_review_functions

    selected, skipped, unresolved, pending_reviewed = [], [], [], []
    functions = [fn for fn in rt.library.functions if names is None or fn.name in names]
    for fn in functions:
        check_cancelled()
        if _unfinished_selection(rt, fn):
            selected.append(fn.name)
            continue
        previous = current_final_validation_skip(rt, fn)
        if previous:
            skipped.append(previous)
            continue
        problem = _evaluation_problem(rt, fn)
        if problem:
            unresolved.append({"name": fn.name, "reason": problem})
            continue
        if _all_predictions_abstained(fn.scores):
            reason = (
                "Automatic full validation skipped: every attempted prediction abstained "
                "because evidence was insufficient. Coverage, precision and adherence remain "
                "unavailable; inspect the recorded evidence gaps before revising this candidate."
            )
            skipped.append(_save_selection(
                rt, fn, status="skipped", reason=reason, grade="UNSCORED"))
            continue
        scores = _current_scores(rt, fn)
        grade = verdict_for_scores(scores)["code"]
        if grade == "UNSCORED":
            metrics = extract_metrics(scores)
            if (scores.get("n_scored") == 0 or not metrics["coverage_available"]
                    or not metrics["normalized_precision_available"]):
                unresolved.append({"name": fn.name, "reason":
                    "Usable coverage and normalized precision are unavailable. Restore "
                    "failed scoring or missing population evidence before final selection."})
                continue
            tiers = [metric_tier(key, metrics.get(key)) for key in ("recall", "ppv_at_pi0")]
            if any(tier == 0 for tier in tiers):
                reason = ("Automatic full validation skipped: measured coverage or normalized "
                          "precision is below OK. Improve this candidate before a full scan.")
                skipped.append(_save_selection(
                    rt, fn, status="skipped", reason=reason, grade=grade))
                continue
            if any(tier is None for tier in tiers):
                unresolved.append({"name": fn.name, "reason":
                    "Usable coverage and normalized precision are required for final selection."})
                continue
            _save_selection(rt, fn, status="reviewing", grade=grade,
                            reason="Checking pending adherence on existing checkpoint matches.")
            reviewed = final_review_functions(
                scope="selection_checkpoint", names=[fn.name], progress=progress,
            )
            pending_reviewed.append(fn.name)
            scores = _current_scores(rt, fn)
            grade = verdict_for_scores(scores)["code"]
            if grade == "UNSCORED":
                # Substantive uncertainty is an actionable result, not a reason
                # to census an unqualified candidate or repeat the same review.
                settled_uncertainty = (
                    scores.get("behavior_n_judged", 0) > 0
                    and scores.get("behavior_n_uncertain", 0) > 0
                    and not scores.get("behavior_n_errors")
                    and not scores.get("behavior_n_review_incomplete")
                    and not scores.get("behavior_evidence_stale")
                    and not scores.get("behavior_judgment_stale")
                )
                if settled_uncertainty:
                    reason = ("Automatic full validation skipped: checkpoint review completed "
                              "but the available evidence does not establish adherence. Inspect "
                              "the review findings before revising or gathering specific evidence.")
                    skipped.append(_save_selection(
                        rt, fn, status="skipped", reason=reason, grade=grade))
                else:
                    reasons = [row["reason"] for row in reviewed.get("unresolved", [])]
                    unresolved.append({"name": fn.name, "reason": "; ".join(reasons) or
                        "Checkpoint review remains unresolved; resume it before a full scan."})
                continue
        if grade in {"GOOD", "OK"}:
            selected.append(fn.name)
            _save_selection(rt, fn, status="selected", grade=grade,
                            reason="OK-or-better candidate selected for full-dataset verification.")
        else:
            reason = (f"Automatic full validation skipped: current grade is "
                      f"{'No matches' if grade == 'EMPTY' else 'Weak'}. "
                      "Scores and review history are retained; reconsider after a revision.")
            skipped.append(_save_selection(rt, fn, status="skipped", reason=reason, grade=grade))
    return {"ok": not unresolved, "selected_names": selected, "skipped": skipped,
            "unresolved": unresolved, "pending_reviewed": pending_reviewed}
