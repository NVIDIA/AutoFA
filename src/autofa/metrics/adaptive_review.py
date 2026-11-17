# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory sample requests; acceptance of each original claim is unchanged."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DIAGNOSTIC_REVIEW_GUIDANCE = """

## Actionable diagnostic feedback
For each unknown or contradicted requirement, optionally include `diagnosis`:
{"observed_facts":"...", "unsupported_premise":"...", "competing_explanation":"...",
 "evidence_needed":"...", "evidence_availability":"available|unavailable|unknown",
 "availability_reason":"..."}.
Use concise, specific facts. Distinguish an available unread source from absent
or unsupported evidence. Name a retained field or artifact only if verified.
State when no competing explanation can be established. Identify the exact fact
that would discriminate explanations: a related fact, reference answer, or wording
proxy does not establish the missing premise. If the required source is absent,
advise deferring that claim; a predicate or prose rewrite cannot prove it.
Calibration is editorial and cannot substitute a proxy or waive the review policy.

You may optionally attach `additional_sample_request` to an item verdict:
{"reason":"across_task_consistency|sample_representativeness",
 "question":"The specific unresolved question about other tasks",
 "expected_evidence":"What another distinct task would establish",
 "preserve_original_findings":true}.
Request this only when a small or unrepresentative sample leaves a concrete
question about the detector across tasks. Additional cases cannot resolve a
missing fact about this case: retain its uncertainty or counterexample. Do not
request sampling to reach a grade, dilute negative findings, or substitute for
reading available evidence. The runtime may review one extra batch, with total
reviews capped at twice the initial target, using the same claim and matched
scoring pool. It may decline when no additional eligible tasks are available.
These advisory fields do not change the claim, verdict rules, or evidence scope.
"""


def normalize_sample_request(value: Any) -> dict | None:
    """Reject incomplete requests and unsupported kinds without altering a verdict."""
    if not isinstance(value, dict):
        return None
    if value.get("reason") not in {"across_task_consistency", "sample_representativeness"}:
        return None
    if value.get("preserve_original_findings") is not True:
        return None
    if any(not isinstance(value.get(key), str) or not value[key].strip()
           or len(value[key]) > 1600 for key in ("question", "expected_evidence")):
        return None
    return {key: value[key] for key in
            ("reason", "question", "expected_evidence", "preserve_original_findings")}


def sample_request(result: dict) -> dict | None:
    """Use the first valid question, in deterministic cohort order."""
    if result.get("n_errors") or result.get("n_review_incomplete"):
        return None
    for row in result.get("verdicts") or []:
        if isinstance(row, dict):
            request = normalize_sample_request(row.get("additional_sample_request"))
            if request:
                return request
    return None


def merge_expanded_review(original: dict, additional: dict) -> dict:
    """Append new findings without replacing or averaging away the original ones."""
    original_ids = {row["item_id"] for row in original["verdicts"]}
    if original_ids & {row["item_id"] for row in additional["verdicts"]}:
        raise ValueError("An additional review cannot replace original findings")
    for key in ("claim_hash", "judge_policy_hash", "judge_model", "judge_harness",
                "judge_effort", "judge_thinking", "judge_thinking_budget_tokens"):
        if original.get(key) != additional.get(key):
            raise ValueError(f"Expanded review changed its frozen {key}")
    result = deepcopy(original)
    rows = result["verdicts"] + deepcopy(additional["verdicts"])
    result["verdicts"] = rows
    for key in ("sessions", "item_calibrations"):
        result[key] = result.get(key, []) + deepcopy(additional.get(key) or [])
    for key, status in (("n_yes", "yes"), ("n_no", "no"),
                        ("n_uncertain", "uncertain"), ("n_errors", "error")):
        result[key] = sum(row.get("verdict") == status for row in rows)
    result["n_judged"] = len(rows)
    result["n_completed"] = result["n_yes"] + result["n_no"]
    result["n_review_incomplete"] = sum(
        row.get("completion_status") == "needs_evidence_review" for row in rows)
    result["behavior_purity"] = (
        result["n_yes"] / (len(rows) - result["n_errors"]) if result["n_completed"] else None)
    result["reused_stages"] = original.get("reused_stages", 0) + additional.get("reused_stages", 0)
    specifics = [(value.get("behavior_specificity"), value.get("n_judged", 0))
                 for value in (original, additional)
                 if type(value.get("behavior_specificity")) in (int, float)]
    if specifics and sum(weight for _, weight in specifics):
        result["behavior_specificity"] = (
            sum(score * weight for score, weight in specifics)
            / sum(weight for _, weight in specifics))
    calibrations = [row["claim_calibration"] for row in result["item_calibrations"]
                    if row.get("claim_calibration")]
    if calibrations and len({value["status"] for value in calibrations}) == 1:
        result["claim_calibration"] = calibrations[0]
    elif calibrations:
        result["claim_calibration"] = {
            "status": "mixed",
            "claim_note": "Claim calibration differs across the retained and additional reviews.",
            "detector_note": "Inspect item_calibrations and the unchanged per-item findings.",
            "rewrite_hint": "Resolve the specific disagreements before changing the claim.",
        }
    else:
        result["claim_calibration"] = None
    # Do not carry editorial advice from only the first batch into the combined result.
    from autofa.metrics.claim_calibration import attach_calibration

    result.pop("calibration_hint", None)
    result = attach_calibration(result)
    result["rationale"] = (
        f"{result['n_yes']}/{len(rows)} labels confirmed; {result['n_no']} rejected; "
        f"{result['n_uncertain']} uncertain; {result['n_errors']} processing errors. "
        "One additional diagnostic batch; original findings retained.")
    return result
