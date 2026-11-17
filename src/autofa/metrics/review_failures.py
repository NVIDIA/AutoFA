# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reviewer execution failures never establish a detector counterexample."""

from __future__ import annotations

from autofa.util.model_errors import model_error_diagnostic, safe_error_detail
from autofa.util.transient import is_transient_error


def processing_failure(reason, *, kind=None) -> dict:
    exc = reason if isinstance(reason, Exception) else RuntimeError(str(reason))
    diagnostic = model_error_diagnostic(exc)
    text = str(reason).lower()
    if kind is None:
        from autofa.metrics.judge_budget import review_budget_failure

        if review_budget_failure(reason):
            kind = "review_budget_exhausted"
        elif "refusal" in text or "safeguards" in text or "reasoning_extraction" in text:
            kind = "backend_refusal"
        elif diagnostic.get("status_code") in {401, 403} or any(
            term in text for term in ("unauthorized", "authentication", "permission denied")
        ):
            kind = "authentication"
        elif is_transient_error(exc):
            kind = "transport"
        elif "context" in text and any(term in text for term in ("exceeded", "too long")):
            kind = "context_limit"
        elif any(term in text for term in ("invalid", "verdict", "citation", "scope annotation")):
            kind = "invalid_reply"
        else:
            kind = "execution_error"
    guidance = {
        "review_budget_exhausted": (
            "Required review work exceeded a bounded resource limit. Inspect the saved workload "
            "estimate and budget usage; preserve the frozen claim, cohort and completed judgments. "
            "Recover only unfinished stages after addressing evidence projection or reviewer "
            "capacity. Unread evidence is not a detector counterexample."
        ),
        "claim_input_ambiguity": (
            "Compilation is blocked by conflicting or ambiguous original claim wording. "
            "Inspect the quoted compiler issues and clarify the description/mechanism "
            "without weakening defining conditions. Submit that grounded changed claim "
            "for fresh evaluation and review. More examples or predicate-only edits "
            "cannot resolve this shared compilation block."
        ),
        "claim_decomposition_error": (
            "The compiler could not produce an approved decomposition after bounded repair. "
            "This does not reject the detector or establish ambiguity in its original claim. "
            "Preserve the claim and inspect the saved draft/audit. One explicit shared-stage "
            "recovery is available; repeated review requests, more examples, or predicate-only "
            "edits reuse the failure. A corrected compiler/backend configuration can reopen it. "
            "If that recovery also fails, a meaning-preserving claim revision is allowed."
        ),
        "backend_refusal": (
            "The review backend refused the request. Keep this candidate and its completed "
            "reviews; do not rewrite its claim to evade the refusal. Check backend eligibility "
            "or select an authorized compatible reviewer before retrying this failed stage."
        ),
        "authentication": "Restore reviewer credentials or permissions, then resume failed stages.",
        "transport": ("Restore the review connection, then resume the frozen cohort "
                      "and saved stages."),
        "context_limit": (
            "Resume with supported review context limits and evidence tools; preserve the frozen "
            "claim and required evidence instead of treating unread evidence as a rejection."
        ),
        "invalid_reply": (
            "The reviewer did not produce a valid evidence-backed reply after bounded schema "
            "repair. Retry only the failed stage once; retain completed verdicts and the claim."
        ),
        "execution_error": (
            "Inspect the reviewer execution diagnostic and correct the service/configuration "
            "failure before retrying the failed stage. Keep completed verdicts."
        ),
    }
    from autofa.metrics.judge_budget import review_budget_failure

    capacity = kind == "review_budget_exhausted" and review_budget_failure(reason) == "capacity"
    recovery = guidance[kind]
    if kind == "review_budget_exhausted" and not capacity:
        recovery = ("This review attempt exhausted its bounded allowance. One explicit "
                    "unfinished-stage retry is available; preserve the frozen claim, "
                    "cohort and completed judgments. Never repeat a consumed stage retry.")
    return {
        "kind": kind, "detector_rejected": False, "diagnostic": diagnostic,
        **({"budget_scope": "capacity" if capacity else "attempt"}
           if kind == "review_budget_exhausted" else {}),
        "retry_policy": "resume_after_recovery" if kind == "transport" else
                        "requires_claim_or_backend_change" if kind == "claim_input_ambiguity" else
                        "requires_backend_or_configuration_change" if
                        kind == "backend_refusal" or capacity else
                        "one_explicit_stage_retry",
        "recovery_action": recovery,
        "detail": safe_error_detail(reason),
    }


def review_failures(rows) -> list[dict]:
    failures = []
    shared = {}
    for row in rows:
        if row.get("verdict") != "error" and row.get("processing_status") != "error":
            continue
        failure = {"item_id": row.get("item_id"),
                   **(row.get("processing_failure") or processing_failure(
                       row.get("validation_error") or row.get("rationale")
                       or "Reviewer execution failed"))}
        # Capacity cannot be repaired by rerolling the same oversized workload.
        # Per-attempt exhaustion retains the already bounded single stage retry.
        if failure.get("kind") == "review_budget_exhausted":
            from autofa.metrics.judge_budget import review_budget_failure

            capacity = (failure.get("budget_scope") == "capacity"
                        or review_budget_failure(failure.get("detail", "")) == "capacity")
            failure["retry_policy"] = ("requires_backend_or_configuration_change" if capacity
                                       else "one_explicit_stage_retry")
        key = failure.get("cache_key") if failure.get("stage") == "claim_compilation" else None
        if key:
            if key not in shared:
                shared[key] = {**failure, "item_ids": [], "affected_items": 0}
                failures.append(shared[key])
            elif failure.get("explicit_retry_used") and not shared[key].get("explicit_retry_used"):
                # Rows retained from the first attempt share the stage's later outcome.
                shared[key].update({field: value for field, value in failure.items()
                                    if field not in {"item_id", "affected_items"}})
            shared[key]["item_ids"].append(row.get("item_id"))
            shared[key]["affected_items"] += 1
        else:
            failures.append(failure)
    return failures


def requires_review_reconfiguration(failure: dict) -> bool:
    """Keep hard capacity failures distinct from a bounded recoverable attempt."""
    if failure.get("kind") == "review_budget_exhausted":
        from autofa.metrics.judge_budget import review_budget_failure

        return (failure.get("budget_scope") == "capacity"
                or review_budget_failure(failure.get("detail", "")) == "capacity")
    return failure.get("retry_policy") == "requires_backend_or_configuration_change"
