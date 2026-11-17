# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge processing retries without erasing the original validation denominator."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from autofa.metrics.claim_calibration import attach_calibration, parse_claim_calibration
from autofa.metrics.review_assessments import reusable_contract


def _review_rows(payload: dict[str, Any], key: str, *, prefix: str = "") -> list[dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Processing retry requires a nonempty list of recorded verdicts")
    normalized = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("item_id"), str)
                or not row["item_id"].strip()):
            raise ValueError("Processing retry verdicts require nonempty string item IDs")
        row = deepcopy(row)
        if row.get("verdict") == "processing_error":
            row["verdict"] = "error"
        if (not isinstance(row.get("verdict"), str)
                or row["verdict"] not in {"yes", "no", "uncertain", "error"}):
            raise ValueError("Processing retry contains an invalid verdict")
        if row.get("processing_status") == "error" and row["verdict"] != "error":
            raise ValueError("Processing status conflicts with semantic verdict")
        completion = row.get("completion_status")
        if (completion == "needs_evidence_review" and row["verdict"] != "uncertain"
                or completion == "processing_error" and row["verdict"] != "error"):
            raise ValueError("Review completion status conflicts with semantic verdict")
        normalized.append(row)
    labels = [row["verdict"] for row in normalized]
    counts = {"n_judged": len(rows), "n_yes": labels.count("yes"), "n_no": labels.count("no"),
              "n_uncertain": labels.count("uncertain"), "n_errors": labels.count("error"),
              "n_review_incomplete": sum(row.get("completion_status") == "needs_evidence_review"
                                         for row in normalized),
              "n_completed": labels.count("yes") + labels.count("no")}
    for count, expected in counts.items():
        key = prefix + count
        if key in payload and (type(payload[key]) is not int or payload[key] != expected):
            raise ValueError(f"Processing retry cohort count {key} disagrees with recorded rows")
    return normalized


def _specificity(value: Any) -> float | None:
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Processing retry contains invalid behavior_specificity")
    return float(value)


def _compilation_outcome(
    previous: dict[str, Any], fresh: dict[str, Any], before: list[dict[str, Any]],
) -> tuple[Any, Any]:
    """Select the frozen contract and compiler receipt that the merged cohort assesses.

    A failed shared compiler stage froze no contract and reviewed no evidence. Its
    retry reports the stage's newer outcome for the same claim: another failure,
    such as a spent explicit retry, or a first approved contract. A first batch
    that crashed before compiling left the same state without a compiler receipt.
    Item-only retries reuse the frozen contract and keep its original receipt.
    """
    contract = previous.get("claim_contract")
    compilation = previous.get("claim_compilation")
    if (not isinstance(contract, dict) or contract.get("defining_clauses")
            or any(row["verdict"] != "error" for row in before)):
        return contract, compilation
    fresh_contract = fresh.get("claim_contract") or {}
    fresh_compilation = fresh.get("claim_compilation") or {}
    if (fresh_contract.get("contract_id") == contract.get("contract_id")
            and (fresh_compilation.get("failure") or {}).get("stage") == "claim_compilation"):
        return contract, fresh_compilation
    if (not fresh_compilation.get("error") and reusable_contract(fresh) is not None
            and fresh_contract.get("claim_hash")
            == contract.get("claim_hash", contract.get("contract_id"))):
        return fresh_contract, fresh_compilation
    return contract, compilation


def merge_processing_retry(  # noqa: C901
    previous: dict[str, Any], fresh: dict[str, Any], retry_ids: set[str],
) -> dict[str, Any]:
    """Replace only failed reviews of an unchanged claim under the identical policy."""
    for key in ("claim_hash", "judge_policy_hash", "judge_contract_version"):
        if not previous.get(key) or previous.get(key) != fresh.get(key):
            raise ValueError(f"Processing retry changed {key}; a fresh validation is required")
    if previous.get("judge_harness") != fresh.get("judge_harness"):
        raise ValueError("Processing retry changed judge_harness; a fresh validation is required")
    if previous.get("judge_backend") != fresh.get("backend"):
        raise ValueError("Processing retry changed judge backend; a fresh validation is required")
    for key in ("judge_model", "judge_effort", "judge_thinking", "judge_thinking_budget_tokens"):
        if previous.get(key) != fresh.get(key):
            raise ValueError(f"Processing retry changed {key}; a fresh validation is required")
    if previous.get("judge_behavior_strictness") != fresh.get("judge_behavior_strictness"):
        raise ValueError("Processing retry changed judge policy")
    before = _review_rows(previous, "behavior_verdicts", prefix="behavior_")
    updates = _review_rows(fresh, "verdicts")
    for old_key, new_key in (("judge_rubric_hash", "judge_rubric_hash"),
                             ("judge_evidence_version", "evidence_version")):
        if previous.get(old_key) is not None and previous[old_key] != fresh.get(new_key):
            # A batch-level exception may have no rubric metadata. Retain the old
            # provenance when no new semantic judgment was produced at all.
            if (fresh.get(new_key) is not None
                    or any(row["verdict"] != "error" for row in updates)):
                raise ValueError(
                    f"Processing retry changed {old_key}; fresh validation is required"
                )
    indexed = {row["item_id"]: row for row in before}
    replacements = {row["item_id"]: row for row in updates}
    if (not retry_ids or len(indexed) != len(before) or len(replacements) != len(updates)
            or set(replacements) != retry_ids):
        raise ValueError("Processing retry must return exactly its reserved distinct item IDs")
    if any(indexed.get(key, {}).get("verdict") != "error" for key in retry_ids):
        raise ValueError("Only processing errors can be retried; semantic judgments stay recorded")
    contract, compilation = _compilation_outcome(previous, fresh, before)
    if not isinstance(contract, dict) or not contract.get("contract_id"):
        raise ValueError("Processing retry requires the original frozen claim contract")
    contract_id = contract["contract_id"]
    # A snapshot embeds the contract it was prepared for. A row that failed before
    # assessing the frozen contract, such as at shared compilation, has no reviewed
    # evidence snapshot to compare.
    if any(indexed[key].get("snapshot_id")
           and replacements[key]["verdict"] != "error"
           and indexed[key].get("claim_contract_id") in {None, contract_id}
           and indexed[key]["snapshot_id"] != replacements[key].get("snapshot_id")
           for key in retry_ids):
        raise ValueError("Processing retry changed the frozen evidence snapshot")
    if any(row["verdict"] != "error" and row.get("claim_contract_id") != contract_id
           for row in before + updates):
        raise ValueError("Processing retry must assess the same frozen claim contract")
    rows = [deepcopy(replacements.get(str(row.get("item_id")), row)) for row in before]
    counts = {status: sum(row.get("verdict") == status for row in rows)
              for status in ("yes", "no", "uncertain", "error")}
    if sum(counts.values()) != len(rows):
        raise ValueError("Original validation contains an invalid verdict")
    old_count = sum(row.get("verdict") in {"yes", "no"} for row in before)
    new_count = sum(row.get("verdict") in {"yes", "no"} for row in updates)
    weighted = [(_specificity(previous.get("behavior_specificity")), old_count),
                (_specificity(fresh.get("behavior_specificity")), new_count)]
    weighted = [(score, count) for score, count in weighted if score is not None and count]
    specificity = (sum(score * count for score, count in weighted)
                   / sum(count for _, count in weighted)) if weighted else None
    calibrations = []
    for payload, permitted in ((previous, set(indexed) - retry_ids), (fresh, retry_ids)):
        seen = set()
        item_calibrations = payload.get("item_calibrations") or []
        if not isinstance(item_calibrations, list):
            raise ValueError("Item calibrations must be a list")
        for row in item_calibrations:
            if (not isinstance(row, dict) or not isinstance(row.get("item_id"), str)
                    or row["item_id"] not in (
                set(indexed) if payload is previous else retry_ids
            )):
                raise ValueError("Item calibration falls outside the processing retry cohort")
            item_id = row["item_id"]
            if item_id in seen:
                raise ValueError("Duplicate item calibration in processing retry")
            seen.add(item_id)
            if item_id not in permitted:
                continue
            cal = parse_claim_calibration(row)
            final = replacements.get(item_id, indexed.get(item_id))
            if cal and final["verdict"] != "error":
                calibrations.append({"item_id": item_id, "claim_calibration": cal})
    decisive = counts["yes"] + counts["no"]
    purity = counts["yes"] / (len(rows) - counts["error"]) if decisive else None
    result = {
        **fresh, "verdicts": rows, "n_judged": len(rows), "n_completed": decisive,
        "n_yes": counts["yes"], "n_no": counts["no"], "n_uncertain": counts["uncertain"],
        "n_review_incomplete": sum(row.get("completion_status") == "needs_evidence_review"
                                   for row in rows),
        "n_errors": counts["error"], "behavior_purity": purity,
        "behavior_specificity": specificity, "item_calibrations": calibrations,
        "claim_calibration": None, "calibration_hint": None,
        "rationale": (f"{counts['yes']}/{len(rows)} claims confirmed; {counts['no']} rejected; "
                      f"{counts['uncertain']} uncertain; {counts['error']} processing errors. "
                      "Replaced failed reviews; retained previous semantic assessments."),
        "processing_retry": {"item_ids": sorted(retry_ids), "retained": len(before) - len(updates)},
    }
    result["claim_contract"] = deepcopy(contract)
    result["claim_compilation"] = deepcopy(compilation)
    for old_key, new_key in (("judge_rubric_hash", "judge_rubric_hash"),
                             ("judge_evidence_version", "evidence_version")):
        if not result.get(new_key) and previous.get(old_key):
            result[new_key] = previous[old_key]
    advice = [row["claim_calibration"] for row in calibrations]
    if advice and len({cal["status"] for cal in advice}) == 1:
        result["claim_calibration"] = advice[0]
    elif advice:
        result["claim_calibration"] = {
            "status": "mixed", "claim_note": "Calibration differs across the preserved cohort.",
            "detector_note": "See item_calibrations for the individual assessments.",
            "rewrite_hint": "Review the differing examples before changing the claim.",
        }
    return attach_calibration(result)
