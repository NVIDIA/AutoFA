# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actionable review diagnoses without inventing missing evidence or new verdicts."""

from __future__ import annotations

from typing import Any

from autofa.core.fields import DEFAULT_CONTENT_FIELDS

DIAGNOSIS_FIELDS = (
    "observed_facts", "unsupported_premise", "competing_explanation",
    "evidence_needed", "availability_reason",
)
EDITORIAL_LIMIT = (
    "Editorial advice is not proof: a related fact or proxy does not establish a missing premise. "
    "Verify the required fact or defer the claim; revisions need fresh review under the "
    "selected Judge Policy."
)


def excerpt(value: Any, *, limit: int) -> tuple[str, int | None]:
    """Return a bounded string with an explicit original length when abbreviated."""
    text = value.strip() if isinstance(value, str) else ""
    return (text, None) if len(text) <= limit else (text[:limit] + "…", len(text))


def excerpt_fields(data: dict[str, Any], fields: tuple[str, ...], *, limit: int) -> dict:
    result = dict(data)
    truncated = {}
    for key in fields:
        result[key], length = excerpt(data.get(key), limit=limit)
        if length:
            truncated[key] = {"original_chars": length}
    if truncated:
        result["fields_truncated"] = truncated
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _source_action(row: dict, assessment: dict, *, name: str) -> tuple[str, dict]:
    """Only citation-backed field names become reads; model advice is never a tool call."""
    item_id = _text(row.get("item_id"))
    if not item_id:
        return "get_function_feedback", {"name": name}
    refs = [
        ref
        for key in ("counterevidence", "citations")
        for ref in (assessment.get(key) if isinstance(assessment.get(key), list) else [])
        if isinstance(ref, dict)
    ]
    field = next((_text(ref.get("field")) for ref in refs
                  if _text(ref.get("field"))), None)
    if field == "trajectory_text":
        return "get_trajectory", {
            "item_id": item_id, "include_observations": True, "offset": 0, "limit": 50,
        }
    if field in {"trace_structure", "trace_commands"}:
        return "get_trace_record", {
            "item_id": item_id, "section": "coverage", "max_chars": 4000,
        }
    if field in DEFAULT_CONTENT_FIELDS:
        return "read_field_chunk", {
            "item_id": item_id, "field": field, "view": "detector",
            "offset": 0, "limit": 4000,
        }
    if field == "detector_source":
        return "get_function_feedback", {"name": name, "item_id": item_id}
    return "get_item", {"item_id": item_id, "include_trace": True}


def build_review_diagnosis(
    row: dict[str, Any], assessment: dict[str, Any], *,
    requirement: str, name: str, max_chars: int = 360,
) -> dict[str, Any]:
    """Use judge-provided facts, with explicit gaps for older unstructured reviews.

    This is advice, not an acceptance check. Raw assessments and diagnoses remain
    untouched in saved verdicts and get_function_feedback.
    """
    raw = assessment.get("diagnosis")
    raw = raw if isinstance(raw, dict) else {}
    contradicted = assessment.get("status") == "contradicted"
    rationale = _text(assessment.get("rationale")) or _text(row.get("rationale"))
    supported = [
        _text(part.get("rationale"))
        for part in row.get("claim_assessments") or []
        if isinstance(part, dict) and part.get("status") == "supported"
        and _text(part.get("rationale"))
    ]
    # A counterexample is a resolved rejection, even if old diagnostic metadata
    # contains an unknown_reason. Source availability does not change that verdict.
    reason = assessment.get("unknown_reason") if not contradicted else None
    availability = raw.get("evidence_availability")
    if availability not in {"available", "unavailable", "unknown"}:
        availability = "unknown"
    # The assessment's explicit reason takes precedence over contradictory advice.
    if reason == "review_incomplete":
        availability = "available"
    elif reason == "evidence_unavailable":
        availability = "unavailable"
    fallbacks = {
        "observed_facts": " ".join(dict.fromkeys(supported)) or (
            "The saved review does not separate observed facts from its explanation."
        ),
        "unsupported_premise": requirement,
        "competing_explanation": (
            "The saved review does not name a competing explanation. Do not invent one; "
            "inspect the full assessment."
        ),
        "evidence_needed": rationale or (
            f"The saved review does not specify which recorded fact would resolve: {requirement}"
        ),
        "availability_reason": {
            "available": "The assessment declares available evidence not yet fully reviewed.",
            "unavailable": "The assessment declares the required evidence unavailable.",
            "unknown": (
                "The saved review does not establish whether the required source is retained."
            ),
        }[availability],
    }
    if contradicted:
        fallbacks.update(
            observed_facts=rationale or fallbacks["observed_facts"],
            evidence_needed=(
                "The cited observations supporting the recorded counterexample, "
                "including any relevant later context and recovery."
            ),
            availability_reason={
                "available": "The diagnosis declares the counterexample evidence available.",
                "unavailable": (
                    "The saved contradiction remains recorded, but its source is declared "
                    "unavailable for inspection."
                ),
                "unknown": (
                    "The review records a counterexample; current access to its source "
                    "has not been established."
                ),
            }[availability],
        )
    diagnosis = {key: _text(raw.get(key)) or value for key, value in fallbacks.items()}
    diagnosis["assessment_status"] = assessment.get("status")
    diagnosis["diagnosis_kind"] = "counterexample" if contradicted else "evidence_gap"
    diagnosis["evidence_availability"] = availability
    diagnosis["provenance"] = "judge" if raw else "saved_assessment_fallback"
    if (raw.get("evidence_availability") in {"available", "unavailable", "unknown"}
            and raw["evidence_availability"] != availability):
        diagnosis["availability_reason"] = fallbacks["availability_reason"]
        diagnosis["availability_source"] = "assessment_unknown_reason"
        diagnosis["overridden_fields"] = ["evidence_availability", "availability_reason"]
    diagnosis["incomplete_fields"] = [key for key in fallbacks if not _text(raw.get(key))]
    diagnosis = excerpt_fields(diagnosis, DIAGNOSIS_FIELDS, limit=max_chars)
    details = {
        "name": "get_function_feedback",
        "arguments": {"name": name},
    }
    if _text(row.get("item_id")):
        details["arguments"]["item_id"] = row["item_id"]
    diagnosis["details_tool"] = details
    missing = diagnosis["evidence_needed"]
    unavailable = (
        "If the required source is absent or cannot establish this fact, retain the unknown "
        "and defer this claim. Rewriting the detector or reviewing other cases cannot prove "
        "this item's missing premise."
    )
    if contradicted:
        tool, arguments = _source_action(row, assessment, name=name)
        contribution = assessment.get("requirement_id") in {
            "failure_contribution", "failure_relevance",
        }
        next_action = {
            "kind": "revise_claim" if contribution else "tighten_detector",
            "tool": tool, "arguments": arguments,
            "instruction": (
                "The review records a counterexample, not an unresolved evidence gap. "
                "Inspect the cited contradiction and relevant later context or recovery; "
                "paginate as needed. "
                + (
                    "Revise the predicate to exclude recovered or unrelated incidents, "
                    "or revise the claimed mechanism only when the trace supports a "
                    "different contribution to failure. Retire the claim if no supported "
                    "distinction remains. Do not remove the selected Judge Policy's "
                    "requirement to make a rejected claim pass."
                    if contribution else
                    "Identify the observed condition that separates this hard negative "
                    "from a true match and repair the predicate. If the claim itself "
                    "misdescribes the behavior, revise or retire it using the recorded "
                    "evidence rather than changing implementations without addressing "
                    "the counterexample."
                )
                + " Any revision needs fresh evaluation and review."
            ),
            "if_unavailable": (
                "If the cited source cannot be retrieved, preserve the recorded "
                "contradiction and inspect the full saved assessment. Do not relabel it "
                "unknown, clear it, or retry the unchanged detector as a way to overturn it."
            ),
        }
    elif availability == "unavailable":
        next_action = {
            "kind": "defer_claim", "tool": details["name"], "arguments": details["arguments"],
            "instruction": (
                f"Required evidence unavailable: {missing} Inspect the saved assessment; "
                "defer this claim unless a retained source can establish that fact."
            ),
        }
    else:
        tool, arguments = _source_action(row, assessment, name=name)
        next_action = {
            "kind": ("inspect_evidence" if availability == "available"
                     else "check_evidence_availability"),
            "tool": tool, "arguments": arguments,
            "instruction": (
                f"Inspect the cited source and determine whether it records: {missing} "
                "Read the relevant later context and recovery; paginate as needed. "
                "The cited source being available does not itself establish the missing fact."
            ),
        }
        if reason == "review_incomplete":
            next_action["instruction"] += (
                " Complete unread ranges only within the original frozen review budget; "
                "do not resubmit an unchanged claim to reset it."
            )
    next_action.setdefault("if_unavailable", unavailable)
    diagnosis["next_action"] = next_action
    return diagnosis
