# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared quality verdicts and metric formatting for CLI and reports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from math import isfinite
from typing import Any

from autofa.metrics.association import pct_caught_from_scores
from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS, task_support_assessment

# Shared quality thresholds; optimizer ranking remains independent.
# Coverage bounds are strict; precision and adherence bounds are inclusive.
QUALITY_THRESHOLDS = {
    "recall": {"good": 0.01, "ok": 0.005, "strict": True},
    "ppv_at_pi0": {"good": 0.70, "ok": 0.40, "strict": False},
    "behavior_purity": {"good": 0.90, "ok": 0.50, "strict": False},
}
VERDICT_PRESENTATIONS = {
    "GOOD": {"label": "GOOD", "symbol": "●", "tone": "good"},
    "OK": {"label": "OK", "symbol": "◐", "tone": "warn"},
    "WEAK": {"label": "WEAK", "symbol": "○", "tone": "bad"},
    "EMPTY": {"label": "NO MATCHES", "symbol": "○", "tone": "neutral"},
    "UNSCORED": {"label": "NOT EVALUATED", "symbol": "…", "tone": "neutral"},
}
_VERDICT_REASONS = {
    "GOOD": "Coverage >1%, normalized precision ≥0.70, detector adherence ≥0.90, and "
            f"matches on at least {MIN_GOOD_FAILED_TASKS} distinct failed tasks.",
    "OK": "Coverage >0.5%, normalized precision ≥0.40, and detector adherence ≥0.50; "
          "at least one score is below the Good threshold.",
    "WEAK": "Coverage ≤0.5%, normalized precision <0.40, or detector adherence <0.50.",
    "EMPTY": "No matches in the evaluated cohort.",
    "UNSCORED": "Quality grade unavailable; inspect evaluation and review status.",
}
_LEGACY_REVIEW_REASON = (
    "Scores meet Good thresholds, but detector adherence comes from a historical review. "
    "Review the current claim before treating this detector as Good."
)


def _verdict(code: str) -> dict[str, str]:
    return {"code": code, **VERDICT_PRESENTATIONS[code], "reason": _VERDICT_REASONS[code]}


def _evaluation_status(scores: dict[str, Any] | None) -> str:
    s = scores or {}
    if (s.get("evaluated") is False or not has_been_evaluated(s) or s.get("evaluation_stale")
            or s.get("n_scored") == 0 or s.get("n_items") == 0):
        return "unscored"
    # Partial evaluation still has usable measurements. Its limitations remain
    # visible alongside the numbers rather than becoming another badge.
    return "evaluated"


def _rate(value: Any) -> float | None:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and 0 <= value <= 1):
        return float(value)
    return None


def _count(value: Any) -> int | None:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and value >= 0 and int(value) == value):
        return int(value)
    return None


def _support_reason(support: int | None, estimate: float | None = None) -> str:
    if estimate is not None:
        return (f"Scores meet Good thresholds, but estimated full-dataset matched support is "
                f"{estimate:.1f} distinct failed tasks ({support or 0} observed). "
                f"Good requires at least {MIN_GOOD_FAILED_TASKS}.")
    detail = ("Distinct flagged failed-task support is unavailable" if support is None
              else f"Detector matches {support} distinct failed tasks")
    return (f"Scores meet Good thresholds. {detail}; Good requires at least "
            f"{MIN_GOOD_FAILED_TASKS} distinct failed tasks matched by the detector.")


def _support_estimate(metrics: dict[str, Any]) -> float | None:
    """Only the validated sampling assessment can enable a provisional verdict."""
    value = metrics.get("estimated_n_matched_failed_tasks")
    if (metrics.get("task_support_estimate_status") == "provisional"
            and metrics.get("task_support_comparison_basis") == "estimated"
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and value >= 0):
        return float(value)
    return None


def _estimated_support_reason(support: int | None, estimate: float) -> str:
    return (f"Provisional Good: scores meet Good thresholds and estimated full-dataset "
            f"matched support is {estimate:.1f} distinct failed tasks "
            f"({support or 0} observed). Final qualification requires matches on "
            f"at least {MIN_GOOD_FAILED_TASKS} observed distinct failed tasks.")


def format_coverage_percent(value: Any) -> str:
    """Keep rounding from implying a different quality tier at either boundary."""
    rate = _rate(value)
    if rate is None:
        return "—"
    for threshold in (QUALITY_THRESHOLDS["recall"]["good"], QUALITY_THRESHOLDS["recall"]["ok"]):
        # One decimal percent can round a value on either side onto the cutoff.
        if rate != threshold and threshold - 0.0005 <= rate <= threshold + 0.0005:
            return f"{'>' if rate > threshold else '<'}{threshold * 100:g}%"
    return f"{rate:.1%}"


def format_flagged(scores: dict[str, Any] | None) -> str:
    """Population flagged rate alongside actual panel counts, never weighted counts."""
    s = scores or {}
    unavailable = s.get("metrics_scope") == "unrepresented_strata"
    rate = None if unavailable else pct_caught_from_scores(s)
    percent = "—" if rate is None else "<0.1%" if 0 < rate < .0005 else f"{rate:.1%}"
    if is_sampled(s):
        observed = _count(s.get("n_rows_scored"))
        if observed is None:
            bucket = s.get("bucket")
            row = next((row for row in s.get("buckets") or []
                        if isinstance(row, dict) and row.get("bucket") == bucket), {})
            observed = _count(row.get("n_rows_scored"))
        if observed is None and s.get("n_flagged") == 0:
            observed = 0
        total = _count(s.get("sample_n"))
        scope = "sampled attempts"
    else:
        observed = _count(s.get("n_flagged"))
        total = _count(s.get("n_scored", s.get("n_items")))
        scope = "resolved attempts"
    if observed is not None and total is not None and total > 0 and observed <= total:
        # In a stratified panel this observed fraction need not equal the weighted
        # rate; explicitly identify the percentage as a population estimate.
        estimate = "estimate; " if is_sampled(s) and not unavailable else ""
        return f"{percent} ({estimate}{observed}/{total} {scope})"
    return percent


def format_scoring_status(scores: dict[str, Any] | None) -> str:
    """Explain prediction availability independently of the causal review."""
    s = scores or {}
    selection_detail = s.get("final_validation_skip_detail")
    if _evaluation_status(s) == "unscored":
        return selection_detail or (
            "Scoring unavailable" if has_been_evaluated(s) else "Scoring pending")
    if s.get("metrics_scope") == "unrepresented_strata":
        return "Scoring partial (sampled panel; population strata missing)"
    partial = s.get("eval_truncated") or s.get("evaluation_scope_complete") is False
    scope = "sampled panel" if is_sampled(s) else "full scan" if (
        s.get("evaluation_scope_complete") is True
    ) else "evaluated cohort"
    if s.get("metrics_scope") == "resolved_items":
        scope += "; unresolved predictions excluded"
    detail = f"Scoring {'partial' if partial else 'complete'} ({scope})"
    return detail + (f" · {selection_detail}" if selection_detail else "")


def metric_tier(name: str, value: Any) -> int | None:
    """Return Weak=0, OK=1, Good=2; missing measurements have no tier."""
    rate = _rate(value)
    thresholds = QUALITY_THRESHOLDS.get(name)
    if rate is None or thresholds is None:
        return None
    for tier, key in ((2, "good"), (1, "ok")):
        above = rate > thresholds[key] if thresholds["strict"] else rate >= thresholds[key]
        if above:
            return tier
    return 0


def judge_prefilter_skipped(scores: dict[str, Any] | None) -> bool:
    """Whether the current evaluation rejected this candidate before judging."""
    s = scores or {}
    prefilter = s.get("judge_prefilter")
    return (isinstance(prefilter, dict) and prefilter.get("status") == "skip"
            and not s.get("judge_prefilter_stale"))


def invalid_claim_provenance(scores: dict[str, Any] | None) -> bool:
    """A current-contract judgment must identify the exact claim it reviewed."""
    from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION

    s = scores or {}
    fingerprint = s.get("claim_hash")
    return (s.get("judge_contract_version") == JUDGE_CONTRACT_VERSION
            and (not isinstance(fingerprint, str) or len(fingerprint) != 64
                 or any(char not in "0123456789abcdef" for char in fingerprint)))


def stale_judge_policy(scores: dict[str, Any] | None) -> bool:
    """Current-contract scores require proof of the current static judge policy.

    An approved contract frozen by another compiler policy is stale as well.
    """
    from autofa.behavior_policy import strictness_for_scores
    from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, contract_behavior_strictness
    from autofa.metrics.claude_judge import compatible_judge_policy
    from autofa.metrics.review_assessments import stale_compiled_contract

    s = scores or {}
    strategy = s.get("judge_objective_strategy", "balanced")
    try:
        strictness = strictness_for_scores(s)
        contract = s.get("claim_contract")
        if isinstance(contract, dict) and (
            contract_behavior_strictness(contract) != strictness
            or stale_compiled_contract(contract)
        ):
            return True
    except ValueError:
        return True
    return (s.get("judge_contract_version") == JUDGE_CONTRACT_VERSION
            and (not isinstance(strategy, str)
                 or strategy not in {"precision", "balanced", "recall"}
                 or not compatible_judge_policy(
                     s.get("judge_policy_hash"), behavior_strictness=strictness)))


def current_behavior_scores(
    scores: dict[str, Any] | None, *, source: str, description: str, mechanism: str,
    objective_strategy: str | None = None,
    behavior_strictness: str | None = None,
) -> dict[str, Any]:
    """Resolve saved claim provenance without changing the historical evidence."""
    from autofa.metrics.claim_contract import claim_hash

    result = dict(scores or {})
    if "task_recall" in result or "task_recall_scope" in result:
        from autofa.metrics.task_recall import current_task_recall_scores

        result = current_task_recall_scores(
            result, source=source, description=description, mechanism=mechanism,
        )
    current_hash = claim_hash(source=source, description=description, mechanism=mechanism)
    prefilter = result.get("judge_prefilter")
    if isinstance(prefilter, dict):
        result["judge_prefilter_stale"] = (
            prefilter.get("claim_hash") is not None and prefilter["claim_hash"] != current_hash
        )
    saved_hash = result.get("claim_hash")
    if invalid_claim_provenance(result):
        result["behavior_judgment_stale"] = True
    elif saved_hash:
        result["behavior_judgment_stale"] = (
            saved_hash != current_hash
            or stale_judge_policy(result)
        )
    elif stale_judge_policy(result):
        result["behavior_judgment_stale"] = True
    if behavior_strictness is not None and result.get("judge_contract_version"):
        if result.get("judge_behavior_strictness", "causal") != behavior_strictness:
            result["behavior_policy_stale"] = True
            result["behavior_judgment_stale"] = True
        else:
            result.pop("behavior_policy_stale", None)
    # Selection is a scheduling receipt, not a grade or review verdict. Resolve
    # its saved source/score/policy provenance before displaying it on a copy.
    from autofa.metrics.final_validation_selection import display_final_validation_skip

    selection = display_final_validation_skip(
        result, source=source, description=description, mechanism=mechanism,
        behavior_strictness=behavior_strictness,
    )
    result.pop("final_validation_skip_detail", None)
    if selection:
        retained = {
            "sampled": "retained sampled scores",
            "full_dataset": "existing full-dataset scores retained",
        }.get(selection.get("evaluation_scope"), "existing results retained")
        reason = str(selection["reason"]).removeprefix("Automatic full validation skipped: ")
        result["final_validation_skip_detail"] = (
            f"Full validation skipped: {reason.rstrip('.')}; {retained}."
        )
    return result


def behavior_validation(
    scores: dict[str, Any] | None, *, include_details: bool = True,
    behavior_strictness: str | None = None,
) -> dict[str, Any]:
    """Describe saved review evidence, optionally requiring the active review policy."""
    from autofa.behavior_policy import validate_behavior_strictness
    from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION

    if behavior_strictness is not None:
        validate_behavior_strictness(behavior_strictness)
    s = scores or {}
    version = s.get("judge_contract_version")
    total = int(s.get("behavior_n_judged") or 0)
    counts = {
        "confirmed": int(s.get("behavior_n_yes") or 0),
        "rejected": int(s.get("behavior_n_no") or 0),
        "uncertain": int(s.get("behavior_n_uncertain") or 0),
        "processing_errors": int(s.get("behavior_n_errors") or 0),
    }
    decisive = counts["confirmed"] + counts["rejected"]
    sample = s.get("behavior_sample") or {}
    target = sample.get("target_n", total)
    sample_limited = (bool(sample.get("limited")) or type(target) is not int
                      or target < 0 or total < target)
    complete = (total > 0 and sum(counts.values()) == total and decisive == total
                and not sample_limited)
    if judge_prefilter_skipped(s):
        status, label = "skipped_pre_filter", "Not judged — rejected by score pre-filter"
    elif s.get("behavior_policy_stale") or (behavior_strictness is not None and version
          and s.get("judge_behavior_strictness", "causal") != behavior_strictness):
        status, label = "stale", "Stale judgment — judge policy changed"
    elif invalid_claim_provenance(s):
        status, label = "stale", "Stale judgment — claim provenance missing or malformed"
    elif stale_judge_policy(s):
        status, label = "stale", "Stale judgment — judge policy changed or provenance missing"
    elif s.get("behavior_evidence_stale"):
        status, label = "stale", "Historical judgment — reviewed rollout changed or is unavailable"
    elif s.get("behavior_judgment_stale"):
        status, label = "stale", "Stale judgment — claim or detector changed"
    elif version != JUDGE_CONTRACT_VERSION and (total or resolved_behavior_purity(s) is not None):
        status, label = "legacy", "Legacy behavior score — root-cause attribution not validated"
    elif sample_limited and not total:
        status, label = "no_fresh_evidence", "No fresh validation evidence"
    elif sample.get("reused_n"):
        status, label = "reused_evidence", "Scored on previously reviewed evidence"
    elif sample_limited:
        status, label = "limited_evidence", "Limited validation evidence"
    elif not version or not total:
        status, label = "unjudged", "Mechanism not judged"
    elif complete and counts["confirmed"] == total:
        status = "confirmed"
        label = {
            "behavioral": "Stated behavior confirmed on reviewed sample",
            "contextual": "Behavior and failure relevance confirmed on reviewed sample",
            "causal": "Claim and root-cause attribution confirmed on reviewed sample",
        }.get(s.get("judge_behavior_strictness", "causal"), "Review policy unavailable")
    elif counts["rejected"]:
        status, label = "mixed", "Claim rejected for some reviewed items"
    elif counts["confirmed"]:
        status, label = "incomplete", "Claim only partially validated"
    elif counts["processing_errors"] == total:
        status, label = "error", "Judge processing failed"
    else:
        status, label = "uncertain", "Claim not established by available evidence"
    result = {
        "status": status, "label": label, "judge_contract_version": version,
        "judge_model": s.get("judge_model"),
        "judge_harness": s.get("judge_harness", "claude"),
        "judge_effort": s.get("judge_effort"),
        "judge_thinking": s.get("judge_thinking"),
        "judge_thinking_budget_tokens": s.get("judge_thinking_budget_tokens"),
        "judge_rubric_hash": s.get("judge_rubric_hash"),
        "judge_policy_hash": s.get("judge_policy_hash"),
        "judge_objective_strategy": s.get("judge_objective_strategy", "balanced"),
        "judge_behavior_strictness": s.get("judge_behavior_strictness", "causal"),
        "claim_hash": s.get("claim_hash"), "sampled": total, **counts,
        "sample": dict(sample),
        "decisive": decisive,
        "decisive_coverage": decisive / total if total else 0.0,
        "ranking_eligible": complete and status in {"confirmed", "mixed"},
    }
    if isinstance(s.get("judge_prefilter"), dict):
        result["judge_prefilter"] = dict(s["judge_prefilter"])
        result["judge_prefilter_stale"] = bool(s.get("judge_prefilter_stale"))
    if status == "skipped_pre_filter":
        result.update(
            reason=s["judge_prefilter"].get("reason"),
            next_steps=["Improve benchmark association and reevaluate the detector, or remove it. "
                        "Judging can proceed after the score pre-filter passes."],
            judgment_history_only=bool(total or s.get("behavior_verdicts")),
        )
    if status in {"unjudged", "stale", "legacy"}:
        from autofa.metrics.review_scheduling import current_review_schedule

        schedule = current_review_schedule(s)
        if schedule is not None:
            result.update(
                label="Automatic adherence review deferred",
                review_schedule=schedule, reason=schedule["reason"],
                next_steps=schedule["next_steps"],
            )
    if include_details:
        result.update(
            claim_contract=s.get("claim_contract"), verdicts=s.get("behavior_verdicts") or [],
        )
    return result


def behavior_ranking_eligibility(
    scores: dict[str, Any] | None, axes: Sequence[str],
) -> dict[str, Any]:
    """Separate research candidates from comparable values on requested behavior axes.

    A zero from a completed, rejected review is a measured score. Missing, stale,
    uncertain or failed reviews are not zero-quality measurements. Plain numerical
    helper inputs remain supported; real function records require review provenance.
    """
    s = scores or {}
    requested = set(axes) & {"behavior_purity", "behavior_specificity"}
    validation = behavior_validation(s, include_details=False)
    status = validation["status"]
    if status == "skipped_pre_filter":
        return {"eligible": False, "status": status,
                "decisive_coverage": validation["decisive_coverage"],
                "reason": "Not judged — rejected by score pre-filter. "
                          "Improve association and reevaluate, or remove the detector."}
    eligible = not requested
    reason = "Requested objectives do not require behavior validation."
    if requested:
        has_review = any(s.get(key) is not None for key in (
            "judge_backend", "judge_contract_version", "behavior_n_judged", "claim_hash",
            "behavior_judgment_stale", "behavior_requires_validation",
        ))
        values = [s.get(axis) for axis in requested]
        if not has_review and any(value is not None for value in values):
            values = [0.0 if value is None else value for value in values]
        available = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and isfinite(value) and 0 <= value <= 1 for value in values)
        eligible = available and (not has_review or validation["ranking_eligible"])
        if eligible:
            reason = "Requested behavior scores have complete decisive reviews."
            if not has_review:
                status, reason = "numeric", "Explicit numerical objective values."
        else:
            reason = ("Behavior review is incomplete or unavailable; retain as a research "
                      "candidate, outside the validated ranking.")
    if eligible and "task_recall" in axes:
        task_recall = s.get("task_recall")
        if (s.get("evaluation_stale") or s.get("task_recall_scope") == "unavailable"
                or not (isinstance(task_recall, (int, float))
                        and not isinstance(task_recall, bool)
                        and isfinite(task_recall) and 0 <= task_recall <= 1)):
            return {"eligible": False, "status": "task_coverage_unavailable",
                    "behavior_status": status,
                    "decisive_coverage": validation["decisive_coverage"],
                    "reason": "Distinct-task recall is unavailable; evaluate with verified "
                              "source task identities before ranking on task coverage. "
                              "Attempt recall cannot substitute for task recall."}
        if (any(key in s for key in (
                "task_recall_scope", "evaluation_claim_hash", "behavior_requires_validation",
            )) and (s.get("task_recall_scope") != "full_selected_corpus"
                    or s.get("sampled") is True
                    or s.get("evaluation_scope_complete") is False)):
            return {"eligible": False, "status": "task_coverage_awaiting_census",
                    "behavior_status": status,
                    "decisive_coverage": validation["decisive_coverage"],
                    "reason": "The corpus task-coverage frontier requires complete corpus "
                              "measurements. Use provisional common-panel search_ranking during "
                              "discovery; defer the corpus scan to final validation. This does "
                              "not reject the detector or its existing causal evidence."}
    return {"eligible": eligible, "status": status,
            "decisive_coverage": validation["decisive_coverage"], "reason": reason}


def precision_from_scores(scores: dict[str, Any] | None) -> float:
    """Raw precision among fires: n_fail / n_flagged (bucket purity).

    Distinct from ppv@π₀ (standardized at π₀=0.30).
    """
    if not scores:
        return 0.0
    if scores.get("raw_ppv") is not None:
        return float(scores["raw_ppv"])
    if scores.get("precision") is not None:
        return float(scores["precision"])
    if scores.get("purity") is not None:
        return float(scores["purity"])
    a = float(scores.get("a") or scores.get("n_fail_caught") or 0)
    b = float(scores.get("b") or 0)
    n = a + b
    if n <= 0 and scores.get("n_flagged"):
        n = float(scores["n_flagged"])
    return (a / n) if n else 0.0


def has_been_evaluated(scores: dict[str, Any] | None) -> bool:
    """True if association metrics were written by evaluate_functions."""
    if not scores:
        return False
    if scores.get("evaluated") is True:
        return True
    # Contingency cells / n_items are only set by evaluate.
    if scores.get("n_items") is not None or scores.get("a") is not None:
        return True
    if scores.get("feedback") and isinstance(scores["feedback"], dict):
        fb = scores["feedback"]
        if fb.get("recall") is not None and scores.get("n_items") is not None:
            return True
    return False


def resolved_behavior_purity(scores: dict[str, Any] | None) -> float | None:
    """Return judged behavior_purity, or None if the behavior judge never ran.

    ``evaluate_functions`` used to stamp ``behavior_purity=0.0`` as a placeholder when
    unjudged — that looked like a failed purity score and agents dropped functions.
    """
    if not scores:
        return None
    if (
        scores.get("judge_backend") == "heuristic"
        or judge_prefilter_skipped(scores)
        or scores.get("behavior_judgment_stale")
        or scores.get("behavior_evidence_stale")
        or invalid_claim_provenance(scores)
        or stale_judge_policy(scores)
        or scores.get("behavior_n_judged") == 0
        or scores.get("behavior_n_completed") == 0
    ):
        return None
    judged = (
        scores.get("judge_backend") is not None
        or scores.get("behavior_n_judged") is not None
        or scores.get("judge_rationale") is not None
        or scores.get("claim_calibration") is not None
        or scores.get("specificity_rationale") is not None
    )
    raw = scores.get("behavior_purity")
    if judged:
        if raw is None:
            return 0.0
        return float(raw)
    # Unjudged: ignore missing or placeholder 0.0 from evaluate.
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val == 0.0:
        return None
    # Non-zero without judge metadata (legacy) — treat as a real score.
    return val


def _display_behavior_purity(scores: dict[str, Any] | None) -> float | None:
    """A missing saved measurement is pending, not the legacy scoring fallback zero."""
    s = scores or {}
    if _rate(s.get("behavior_purity")) is None:
        return None
    return _rate(resolved_behavior_purity(s))


def format_behavior_purity(scores: dict[str, Any] | None) -> str:
    bp = _display_behavior_purity(scores)
    if bp is None:
        return "beh=—"
    legacy = behavior_validation(scores)["status"] == "legacy"
    return f"beh={bp:.2f}" + (" (legacy)" if legacy else "")


def resolved_behavior_specificity(scores: dict[str, Any] | None) -> float | None:
    """Return judged behavior_specificity, or None if never scored by the judge.

    Distinct from purity: measures whether the detector isolates a concrete causal
    decision (1), with broad symptoms and structural proxies at the low end (0).
    """
    if not scores:
        return None
    if (
        scores.get("judge_backend") == "heuristic"
        or judge_prefilter_skipped(scores)
        or scores.get("behavior_judgment_stale")
        or scores.get("behavior_evidence_stale")
        or invalid_claim_provenance(scores)
        or stale_judge_policy(scores)
        or scores.get("behavior_n_judged") == 0
        or scores.get("behavior_n_completed") == 0
    ):
        return None
    judged = (
        scores.get("judge_backend") is not None
        or scores.get("behavior_n_judged") is not None
        or scores.get("specificity_rationale") is not None
        or (
            scores.get("behavior_specificity") is not None
            and scores.get("judge_rationale") is not None
        )
    )
    raw = scores.get("behavior_specificity")
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not judged:
        # Heuristic stamp without judge metadata still counts when explicitly set.
        return val
    return val


def format_behavior_specificity(scores: dict[str, Any] | None) -> str:
    sp = resolved_behavior_specificity(scores)
    if sp is None:
        return "spec=—"
    return f"spec={sp:.2f}"


def _diagnostic_review_detail(scores: dict[str, Any]) -> str:
    if scores.get("behavior_n_judged") or not scores.get("diagnostic_review"):
        return ""
    from autofa.metrics.diagnostic_review import diagnostic_scores

    supplemental = diagnostic_scores(scores)
    if supplemental is None:
        return ""
    n = _count(supplemental.get("behavior_n_judged")) or 0
    errors = _count(supplemental.get("behavior_n_errors")) or 0
    incomplete = _count(supplemental.get("behavior_n_review_incomplete")) or 0
    completed = max(0, n - errors - incomplete)
    pending_n = errors + incomplete
    pending = (f" {pending_n} control{'s' if pending_n != 1 else ''} "
               f"still {'need' if pending_n != 1 else 'needs'} review recovery."
               if pending_n else "")
    return (f"Development controls reviewed: {completed}; adherence unavailable.{pending} "
            "Selected examples are excluded from adherence and confirmed support.")


def _review_status_detail(
    scores: dict[str, Any], validation: dict[str, Any], *, diagnostic_detail: str = "",
) -> str:
    """Explain missing adherence without treating inconclusive reviews as pending."""
    status = validation["status"]
    if validation.get("review_schedule"):
        detail = "Automatic review deferred: " + validation["review_schedule"]["reason"]
        return detail + (" " + diagnostic_detail if diagnostic_detail else "")
    if status in {"stale", "legacy"}:
        return validation["label"] + "."
    if diagnostic_detail:
        return diagnostic_detail
    if status == "skipped_pre_filter":
        return "Review skipped: " + str(validation.get("reason") or validation["label"])
    errors = validation["processing_errors"]
    uncertain = validation["uncertain"]
    incomplete = int(scores.get("behavior_n_review_incomplete") or 0)
    if not incomplete:
        incomplete = sum(
            row.get("completion_status") == "needs_evidence_review"
            for row in scores.get("behavior_verdicts") or [] if isinstance(row, dict)
        )
    if errors or incomplete:
        parts = []
        if errors:
            parts.append(f"{errors} processing error{'s' if errors != 1 else ''}")
        if incomplete:
            parts.append(f"{incomplete} case{'s' if incomplete != 1 else ''} with unread evidence")
        if uncertain > incomplete:
            parts.append(f"{uncertain - incomplete} inconclusive case(s)")
        prefix = "Review failed" if errors == validation["sampled"] else "Review incomplete"
        return prefix + ": " + ", ".join(parts) + "."
    if uncertain:
        unknown_requirements = {
            assessment.get("requirement_id")
            for row in scores.get("behavior_verdicts") or [] if isinstance(row, dict)
            for assessment in row.get("claim_assessments") or [] if isinstance(assessment, dict)
            if assessment.get("status") == "unknown"
        }
        subject = ("causal evidence" if "failure_contribution" in unknown_requirements
                   else "failure relevance" if "failure_relevance" in unknown_requirements
                   else "claim evidence")
        return (
            f"Review completed: {subject} unresolved "
            f"({uncertain} inconclusive case{'s' if uncertain != 1 else ''})."
        )
    if not validation["sampled"]:
        if status == "no_fresh_evidence":
            return "Review unavailable: no eligible evidence was allocated."
        return "Review pending: no current judge verdict."
    return "Review score unavailable; inspect the saved assessments."


def _unavailable_reason(
    scores: dict[str, Any], metrics: dict[str, Any], validation: dict[str, Any],
) -> str:
    if scores.get("final_validation_skip_detail") and (
        metrics["evaluation_status"] == "unscored"
        or not metrics["coverage_available"] or not metrics["normalized_precision_available"]
        or _rate(metrics["behavior_purity"]) is None
    ):
        return str(scores["final_validation_skip_detail"])
    if scores.get("metrics_scope") == "unrepresented_strata":
        return str(scores.get("sampling_limitation") or
                   "Population rates unavailable: scoring did not represent every stratum.")
    if metrics["evaluation_status"] == "unscored":
        if scores.get("evaluation_stale"):
            return "Evaluation is stale; score the current detector."
        if scores.get("n_scored") == 0:
            return (
                "Evaluation has no resolved predictions; inspect execution errors and abstentions."
            )
        return "Evaluation pending: detector scores are not available yet."
    missing = []
    if not metrics["coverage_available"]:
        missing.append("coverage")
    if not metrics["normalized_precision_available"]:
        missing.append("normalized precision")
    if missing:
        return "Evaluation has no usable " + " or ".join(missing) + "; inspect scoring results."
    if metrics.get("behavior_processing_pending") or _rate(metrics["behavior_purity"]) is None:
        return _review_status_detail(
            scores, validation, diagnostic_detail=metrics["diagnostic_review_detail"])
    return ""


def extract_metrics(scores: dict[str, Any] | None) -> dict[str, Any]:
    s = scores or {}
    population_unavailable = s.get("metrics_scope") == "unrepresented_strata"
    bp = _display_behavior_purity(s)
    sp = resolved_behavior_specificity(s)
    validation = behavior_validation(s, include_details=False)
    # Explainers: prefer within-parent stratum metrics for recall/ppv gates.
    explains = bool(s.get("within_parent") or s.get("explains"))
    recall_value = (s.get("within_recall") if explains and s.get("within_recall") is not None
                    else s.get("recall"))
    ppv_value = (s.get("within_ppv_at_pi0")
                 if explains and s.get("within_ppv_at_pi0") is not None
                 else s.get("ppv_at_pi0"))
    recall = None if population_unavailable else _rate(recall_value)
    ppv = None if population_unavailable else _rate(ppv_value)
    support = task_support_assessment(s)
    matched_attempts = _count(s.get("n_matched_failed_attempts"))
    if support["matched_task_support_status"] == "verified":
        # Reviews can discover additional matches outside the scoring panel.
        # Display attempts and tasks from the same verified evidence inventory.
        matched_attempts = len(s["task_support"]["failed_matched_task_by_item_id"])
    metrics = {
        "evaluation_status": _evaluation_status(s),
        "coverage_available": recall is not None,
        "normalized_precision_available": ppv is not None,
        "n_flagged": s.get("n_flagged"),
        "badge_n_flagged": (s.get("within_n_flagged")
                            if explains and s.get("within_n_flagged") is not None
                            else s.get("n_flagged")),
        "pct_caught": None if population_unavailable else pct_caught_from_scores(s),
        "recall": recall if recall is not None else 0.0,
        "precision": None if population_unavailable else precision_from_scores(s),
        "ppv_at_pi0": ppv if ppv is not None else 0.0,
        # Only real judgments use 0.0; formatters omit the -1.0 unjudged sentinel.
        "behavior_purity": float(bp) if bp is not None else -1.0,
        "behavior_specificity": float(sp) if sp is not None else -1.0,
        "behavior_validation_status": validation["status"],
        "behavior_decisive_coverage": validation["decisive_coverage"],
        "behavior_ranking_eligible": validation["ranking_eligible"],
        "behavior_processing_pending": bool(
            s.get("behavior_n_errors") or s.get("behavior_n_review_incomplete")
            or any(isinstance(row, dict) and (
                row.get("verdict") == "error" or row.get("processing_status") == "error"
                or row.get("completion_status") == "needs_evidence_review"
            ) for row in s.get("behavior_verdicts") or [])
        ),
        "scoring_detail": format_scoring_status(s),
        "flagged_detail": format_flagged(s),
        "diagnostic_review_detail": _diagnostic_review_detail(s),
        **support,
        # These are observed counts, never the weighted association estimates.
        "n_matched_failed_attempts": matched_attempts,
    }
    metrics["unavailable_reason"] = _unavailable_reason(s, metrics, validation)
    return metrics


def extract_report_metrics(scores: dict[str, Any] | None) -> dict[str, Any]:
    """Report completeness must not mistake formatting defaults for measured zeros."""
    s = scores or {}
    metrics = extract_metrics(s)
    if not metrics["coverage_available"]:
        metrics["recall"] = None
    if not metrics["normalized_precision_available"]:
        metrics["ppv_at_pi0"] = None
    if not (
        any(s.get(key) is not None for key in ("raw_ppv", "precision", "purity"))
        or all(s.get(key) is not None for key in ("a", "b"))
        or all(s.get(key) is not None for key in ("n_fail_caught", "n_flagged"))
    ):
        metrics["precision"] = None
    return metrics


def verdict_for_scores(scores: dict[str, Any] | None) -> dict[str, str]:
    """Combined quality verdict, preserving unavailable and zero-match states."""
    return verdict_for_metrics(extract_metrics(scores), scores=scores)


def verdict_for_metrics(
    m: dict[str, Any], *, scores: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Use the lowest measured tier; review/scoring eligibility is unchanged.

    Explicit match counts take precedence over rounded or legacy metrics. Without
    counts, a positive flagged fraction, coverage or precision establishes a match.
    """
    status = _evaluation_status(scores) if scores is not None else m.get("evaluation_status")
    recall = _rate(m.get("recall"))
    unscored = {**_verdict("UNSCORED")}
    if m.get("unavailable_reason"):
        unscored["reason"] = str(m["unavailable_reason"])
    if status == "unscored" or m.get("coverage_available") is False or recall is None:
        return unscored
    matches = m.get("badge_n_flagged", m.get("n_flagged"))
    if (isinstance(matches, (int, float)) and not isinstance(matches, bool)
            and isfinite(matches) and matches >= 0):
        has_matches = matches > 0
    else:
        has_matches = any((_rate(m.get(key)) or 0) > 0
                          for key in ("pct_caught", "recall", "precision"))
    if not has_matches:
        return _verdict("EMPTY")
    if m.get("normalized_precision_available") is False or m.get("behavior_processing_pending"):
        return unscored
    tiers = [metric_tier(name, m.get(name)) for name in QUALITY_THRESHOLDS]
    if any(tier is None for tier in tiers):
        return unscored
    tier = min(tiers)
    if tier == 2 and m.get("behavior_validation_status") == "legacy":
        return {**_verdict("OK"), "reason": _LEGACY_REVIEW_REASON}
    support = _count(m.get("n_matched_failed_tasks"))
    if tier == 2 and (support is None or support < MIN_GOOD_FAILED_TASKS):
        estimate = _support_estimate(m)
        if estimate is not None and estimate >= MIN_GOOD_FAILED_TASKS:
            return {**_verdict("GOOD"), "reason": _estimated_support_reason(support, estimate)}
        return {**_verdict("OK"), "reason": _support_reason(support, estimate)}
    return _verdict(("WEAK", "OK", "GOOD")[tier])


def is_sampled(scores: dict[str, Any] | None) -> bool:
    """Whether these metrics are estimates from a subsample rather than exact."""
    s = scores or {}
    if s.get("sampled"):
        return True
    fb = s.get("feedback")
    return bool(isinstance(fb, dict) and fb.get("sampled"))


def sampled_marker(scores: dict[str, Any] | None, *, short: bool = False) -> str:
    """Suffix marking sampled estimates, so they never read as exact counts."""
    if not is_sampled(scores):
        return ""
    s = scores or {}
    n = s.get("sample_n")
    frame = s.get("sample_frame_n")
    if short:
        return " ~est"
    if n and frame:
        return f"  ~estimate (stratified sample n={n}/{frame})"
    return "  ~estimate (stratified sample)"


def format_interval(scores: dict[str, Any] | None, metric: str) -> str:
    """``[lo, hi]`` 95% Wilson CI for ``recall`` / ``fpr`` / ``raw_ppv``, or empty.

    Explainers display the within-parent stratum (see ``extract_metrics``), so the
    interval must come from the same stratum or it would not bracket the number
    next to it.
    """
    s = scores or {}
    prefix = ""
    if metric in {"recall", "fpr"} and (s.get("within_parent") or s.get("explains")):
        within_key = f"within_{metric}"
        if s.get(within_key) is not None or (
            metric == "recall" and s.get("within_recall") is not None
        ):
            prefix = "within_"
            # No within-stratum interval available ⇒ show no interval rather than
            # the corpus-wide one, which brackets a different number.
            if s.get(f"within_{metric}_ci_low") is None:
                return ""
    lo = s.get(f"{prefix}{metric}_ci_low")
    hi = s.get(f"{prefix}{metric}_ci_high")
    if lo is None or hi is None:
        return ""
    return f"[{float(lo):.2f},{float(hi):.2f}]"


def _fisher_snippet(scores: dict[str, Any] | None) -> str:
    s = scores or {}
    p = s.get("within_fisher_p") if (s.get("within_parent") or s.get("explains")) else None
    if p is None:
        p = s.get("fisher_p")
    if p is None:
        return ""
    p = float(p)
    if p < 0.001:
        return "fisher_p<0.001"
    return f"fisher_p={p:.3g}"


def format_task_recall(scores: dict[str, Any] | None) -> str:
    """Observed task coverage, explicitly distinguished from attempt estimates."""
    s = scores or {}
    value = s.get("task_recall")
    if (s.get("evaluation_stale") or s.get("task_recall_scope") == "unavailable"
            or not isinstance(value, (int, float)) or isinstance(value, bool)
            or not isfinite(value) or not 0 <= value <= 1):
        return "—"
    text = f"{value:.1%}"
    caught, total = s.get("n_matched_failed_tasks"), s.get("n_failed_tasks")
    if isinstance(caught, int) and isinstance(total, int) and total > 0:
        text += f" ({caught}/{total} tasks)"
    if s.get("task_recall_scope") == "observed_lower_bound":
        text += " observed lower bound"
    return text


def format_coverage_support(metrics: dict[str, Any]) -> str:
    """Observed support stays separate from percentage estimates and repeat counts."""
    parts = []
    attempts = _count(metrics.get("n_matched_failed_attempts"))
    tasks = _count(metrics.get("n_matched_failed_tasks"))
    support = _count(metrics.get("n_confirmed_failed_tasks"))
    if attempts is not None:
        parts.append(f"{attempts} observed failed rollouts")
    if tasks is not None:
        parts.append(f"{tasks} distinct failed tasks matched")
    parts.append("confirmed tasks unavailable" if support is None
                 else f"{support} confirmed failed tasks")
    estimate = _support_estimate(metrics)
    if estimate is not None:
        parts.append(f"~{estimate:.1f} matched failed tasks estimated in full dataset")
    return " · ".join(parts)


def format_metrics_line(scores: dict[str, Any] | None, *, short: bool = False) -> str:
    """One-line metrics with clear names (not cryptic bp=)."""
    v = verdict_for_scores(scores)
    if _evaluation_status(scores) == "unscored":
        return f"{v['symbol']} {v['label']} — {v['reason']}"
    m = extract_metrics(scores)
    recall = f"{m['recall']:.2f}" if m["coverage_available"] else "—"
    ppv = f"{m['ppv_at_pi0']:.2f}" if m["normalized_precision_available"] else "—"
    precision = f"{m['precision']:.2f}" if m["precision"] is not None else "—"
    beh = format_behavior_purity(scores)
    spec = format_behavior_specificity(scores)
    rec_ci = format_interval(scores, "recall")
    fpr_ci = format_interval(scores, "fpr")
    s = scores or {}
    fpr = s.get("within_fpr") if (s.get("within_parent") or s.get("explains")) else None
    if fpr is None:
        fpr = s.get("fpr")
    fpr_bit = ""
    if fpr is not None:
        fpr_bit = f"  fpr={float(fpr):.3f}{fpr_ci}"
    fish = _fisher_snippet(scores)
    fish_bit = f"  {fish}" if fish else ""
    if short:
        return (
            f"{v['symbol']}{v['label']}  "
            f"%caught={m['flagged_detail']}  "
            f"prec={precision}  "
            f"attempt_recall={recall}{rec_ci}  "
            f"({format_coverage_support(m)})  "
            f"task_recall={format_task_recall(scores)}  "
            f"ppv@π₀={ppv}  "
            f"{beh}  {spec}"
            f"{sampled_marker(scores, short=True)}"
            + f" · {m['scoring_detail']}"
            + (f" · {v['reason']}" if v["code"] == "UNSCORED" else "")
            + (f" · {m['diagnostic_review_detail']}"
               if v["code"] != "UNSCORED" and m["diagnostic_review_detail"] else "")
        )
    beh_long = (
        beh.replace("beh=", "behavior_purity=")
        if beh != "beh=—"
        else "behavior_purity=unavailable"
    )
    spec_long = (
        spec.replace("spec=", "behavior_specificity=")
        if spec != "spec=—"
        else "behavior_specificity=unavailable"
    )
    recall = f"{m['recall']:.3f}" if m["coverage_available"] else "—"
    ppv = f"{m['ppv_at_pi0']:.3f}" if m["normalized_precision_available"] else "—"
    return (
        f"{v['symbol']} {v['label']} ({v['reason']})  ·  "
        f"%caught={m['flagged_detail']}  "
        f"precision={precision}  "
        f"attempt_recall={recall}{rec_ci}  "
        f"({format_coverage_support(m)})  "
        f"task_recall={format_task_recall(scores)}  "
        f"ppv@π₀={ppv}"
        f"{fpr_bit}{fish_bit}  "
        f"{beh_long}  {spec_long}"
        f"{sampled_marker(scores)}"
        f" · {m['scoring_detail']}"
        + (f" · {m['diagnostic_review_detail']}"
           if v["code"] != "UNSCORED" and m["diagnostic_review_detail"] else "")
    )


def verdict_style(code: str) -> str:
    """prompt_toolkit semantic style for the shared quality verdict."""
    tone = VERDICT_PRESENTATIONS.get(code, {}).get("tone", "neutral")
    return {"good": "good", "warn": "ok", "bad": "bad", "neutral": "dim"}[tone]


def verdict_legend() -> str:
    return ("GOOD: coverage >1%, normalized precision ≥0.70, adherence ≥0.90 · "
            f"matches on ≥{MIN_GOOD_FAILED_TASKS} distinct failed tasks · "
            "OK: coverage >0.5%, normalized precision ≥0.40, adherence ≥0.50 · "
            "WEAK: any score below the OK thresholds · NO MATCHES: no evaluated matches · "
            "NOT EVALUATED: evaluation or review scores unavailable. "
            "The lowest score tier determines the verdict; insufficient matched task "
            "support caps it at OK. During sampled evaluation, validated full-dataset task "
            "support estimates can qualify as provisional Good; final qualification requires "
            "observed matched failed tasks. Detector adherence is assessed separately.")


def verdict_js_classify_fn() -> str:
    """Browser counterpart with labels, reasons, colors and boundary shared above."""
    declarations = (
        "const VERDICT_PRESENTATIONS = " + json.dumps(VERDICT_PRESENTATIONS, ensure_ascii=False)
        + ";\nconst VERDICT_REASONS = " + json.dumps(_VERDICT_REASONS, ensure_ascii=False)
        + ";\nconst QUALITY_THRESHOLDS = " + json.dumps(QUALITY_THRESHOLDS) + ";\n"
        + "const LEGACY_REVIEW_REASON = " + json.dumps(_LEGACY_REVIEW_REASON) + ";\n"
        + f"const MIN_GOOD_FAILED_TASKS = {MIN_GOOD_FAILED_TASKS};\n"
    )
    return declarations + r"""
function qualityVerdict(code) {
  return {code, ...VERDICT_PRESENTATIONS[code], reason: VERDICT_REASONS[code]};
}
function matchedSupportReason(support, estimate = null) {
  if (estimate !== null) {
    return `Scores meet Good thresholds, but estimated full-dataset matched support is `
      + `${estimate.toFixed(1)} distinct failed tasks (${support || 0} observed). `
      + `Good requires at least ${MIN_GOOD_FAILED_TASKS}.`;
  }
  const detail = support === null ? 'Distinct flagged failed-task support is unavailable'
    : `Detector matches ${support} distinct failed tasks`;
  return `Scores meet Good thresholds. ${detail}; Good requires at least `
    + `${MIN_GOOD_FAILED_TASKS} distinct failed tasks matched by the detector.`;
}
function matchedSupportEstimate(m) {
  const value = m.estimated_n_matched_failed_tasks;
  return m.task_support_estimate_status === 'provisional'
    && m.task_support_comparison_basis === 'estimated'
    && typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}
function estimatedSupportReason(support, estimate) {
  return `Provisional Good: scores meet Good thresholds and estimated full-dataset `
    + `matched support is ${estimate.toFixed(1)} distinct failed tasks `
    + `(${support || 0} observed). Final qualification requires matches on `
    + `at least ${MIN_GOOD_FAILED_TASKS} observed distinct failed tasks.`;
}
function verdictClass(code) {
  return (VERDICT_PRESENTATIONS[code] || {}).tone || 'neutral';
}
function cardClass(code) { return verdictClass(code); }
function coverageRate(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
    ? value : null;
}
function metricTier(name, value) {
  const rate = coverageRate(value);
  const thresholds = QUALITY_THRESHOLDS[name];
  if (rate === null || !thresholds) return null;
  if (thresholds.strict ? rate > thresholds.good : rate >= thresholds.good) return 2;
  if (thresholds.strict ? rate > thresholds.ok : rate >= thresholds.ok) return 1;
  return 0;
}
function metricClass(name, value) {
  const tier = metricTier(name, value);
  return tier === null ? 'neutral' : ['bad', 'warn', 'green'][tier];
}
function coveragePercent(value) {
  const rate = coverageRate(value);
  if (rate === null) return '—';
  for (const threshold of [QUALITY_THRESHOLDS.recall.good, QUALITY_THRESHOLDS.recall.ok]) {
    if (rate !== threshold && rate >= threshold - 0.0005 && rate <= threshold + 0.0005) {
      return (rate > threshold ? '>' : '<') + (threshold * 100) + '%';
    }
  }
  return (rate * 100).toFixed(1) + '%';
}
function classifyFromMetrics(m, evaluation = null) {
  m = m || {};
  const unscoredVerdict = () => ({
    ...qualityVerdict('UNSCORED'),
    ...(m.unavailable_reason ? {reason: m.unavailable_reason} : {})
  });
  let unscored = m.evaluation_status === 'unscored';
  // Array.map supplies an index as its second argument, not evaluation metadata.
  if (evaluation && typeof evaluation === 'object' && !Array.isArray(evaluation)) {
    unscored = unscored || evaluation.evaluated === false || !!evaluation.evaluation_stale
      || evaluation.n_scored === 0 || evaluation.n_items === 0;
  }
  const recall = coverageRate(m.recall);
  if (unscored || m.coverage_available === false || recall === null) {
    return unscoredVerdict();
  }
  const matches = m.badge_n_flagged === undefined ? m.n_flagged : m.badge_n_flagged;
  const hasMatches = typeof matches === 'number' && Number.isFinite(matches) && matches >= 0
    ? matches > 0 : ['pct_caught', 'recall', 'precision'].some(key => coverageRate(m[key]) > 0);
  if (!hasMatches) return qualityVerdict('EMPTY');
  const tiers = Object.keys(QUALITY_THRESHOLDS).map(name => metricTier(name, m[name]));
  if (m.normalized_precision_available === false || m.behavior_processing_pending
      || tiers.some(tier => tier === null)) {
    return unscoredVerdict();
  }
  const tier = Math.min(...tiers);
  if (tier === 2 && m.behavior_validation_status === 'legacy') {
    return {...qualityVerdict('OK'), reason: LEGACY_REVIEW_REASON};
  }
  const rawSupport = m.n_matched_failed_tasks;
  const support = typeof rawSupport === 'number' && Number.isInteger(rawSupport)
    && rawSupport >= 0 ? rawSupport : null;
  if (tier === 2 && (support === null || support < MIN_GOOD_FAILED_TASKS)) {
    const estimate = matchedSupportEstimate(m);
    if (estimate !== null && estimate >= MIN_GOOD_FAILED_TASKS) {
      return {...qualityVerdict('GOOD'), reason: estimatedSupportReason(support, estimate)};
    }
    return {...qualityVerdict('OK'), reason: matchedSupportReason(support, estimate)};
  }
  return qualityVerdict(['WEAK', 'OK', 'GOOD'][tier]);
}
"""
