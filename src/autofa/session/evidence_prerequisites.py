# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap, source-bound evidence checks before spending on corpus predictions.

These checks establish whether declared source material is retained, not whether
its content proves a claim. A missing field on a selected item says nothing about
uninspected items or about whether the underlying real-world event occurred.
"""

from __future__ import annotations

import json
from contextlib import closing
from typing import Any

from autofa.core.fields import DEFAULT_CONTENT_FIELDS

MAX_SUPPORT_ITEMS = 3
EVIDENCE_PREREQUISITE_SCHEMA = {
    "type": "array", "maxItems": 8, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["fact", "item_id", "field"],
        "properties": {
            "fact": {"type": "string", "minLength": 1},
            "item_id": {"type": "string", "minLength": 1},
            "field": {"type": "string", "enum": sorted(DEFAULT_CONTENT_FIELDS)},
            "pointer": {"type": "string", "default": ""},
        },
    },
}


def validate_prerequisites(raw: Any) -> list[dict]:
    """Validate only explicit source declarations; never parse claims for keywords."""
    if not isinstance(raw, list) or len(raw) > 8:
        raise ValueError("evidence_prerequisites must be a list with at most eight source checks")
    result = []
    for row in raw:
        if (not isinstance(row, dict)
                or set(row) - {"fact", "item_id", "field", "pointer"}
                or any(not isinstance(row.get(key), str) or not row[key].strip()
                       for key in ("fact", "item_id", "field"))):
            raise ValueError("Each evidence prerequisite needs fact, item_id, and field strings")
        pointer = row.get("pointer", "")
        if (row["field"] not in DEFAULT_CONTENT_FIELDS
                or not isinstance(pointer, str) or (pointer and not pointer.startswith("/"))):
            raise ValueError("Use a retained content field and an empty or RFC 6901 pointer")
        if max(len(row["fact"]), len(row["item_id"]), len(pointer)) > 2000:
            raise ValueError("Evidence prerequisites must use compact facts and exact source IDs")
        result.append({**row, "pointer": pointer})
    if len({row["item_id"] for row in result}) > MAX_SUPPORT_ITEMS:
        raise ValueError("Evidence prerequisites may inspect at most three distinct support items")
    return result


def _source_items(rt: Any, ids: set[str]):
    """Point-read original evidence; never fall back to scanning complete traces."""
    index = rt.dataset.identity_index()
    locations = [index.by_id[item_id] for item_id in sorted(ids) if item_id in index.by_id]
    if not locations:
        return
    with closing(rt.dataset.iter_locations(locations)) as items:
        yield from items


def _declared_checks(rt: Any, prerequisites: list[dict]) -> list[dict]:
    from autofa.metrics.judge_evidence import _content_value, make_snapshot

    ids = {row["item_id"] for row in prerequisites}
    available = {}
    try:
        for item in _source_items(rt, ids):
            item_id = str(item.get("_item_id") or "")
            snapshot = None
            for index, requirement in enumerate(prerequisites):
                if requirement["item_id"] != item_id:
                    continue
                field = requirement["field"]
                if field in item:
                    # Preserve the declared original JSON path. The judge's
                    # compact trace representation can rearrange nested records.
                    value = _content_value(item[field], [], field)
                else:
                    if snapshot is None:
                        snapshot = make_snapshot(item, item_id=item_id, source="")
                    value = snapshot["fields"].get(field)
                try:
                    if requirement["pointer"]:
                        value = json.loads(value) if isinstance(value, str) else value
                        for token in requirement["pointer"].split("/")[1:]:
                            token = token.replace("~1", "/").replace("~0", "~")
                            if isinstance(value, list):
                                if not token.isdigit():
                                    raise KeyError(token)
                                value = value[int(token)]
                            else:
                                value = value[token]
                    # False and zero are retained evidence, not missing values.
                    present = value is not None and value not in ("", [], {})
                except (KeyError, IndexError, TypeError, ValueError):
                    present = False
                available[index] = "retained" if present else "not_retained"
    except (OSError, ValueError, AttributeError) as exc:
        return [{**row, "availability": "unknown", "availability_reason":
                 f"Source could not be checked: {exc}"} for row in prerequisites]
    return [
        {**row, "availability": available.get(index, "unknown"),
         "availability_reason": (
             "The declared source location is retained; its meaning still needs review."
             if available.get(index) == "retained" else
             "This selected item's retained reviewable source lacks the declared location."
             if index in available else
             "The selected item could not be loaded from the current source index."
         ),
         "next_action": {
             "tool": "get_field_meta",
             "arguments": {"item_id": row["item_id"], "field": row["field"]},
             "instruction": f"Check the retained source for this required fact: {row['fact']}",
             "if_unavailable": "Use another genuinely supported item or defer this claim; "
                               "a code rewrite cannot supply a missing source fact.",
         }}
        for index, row in enumerate(prerequisites)
    ]


def _known_gaps(rt: Any, fn: Any) -> list[dict]:
    """Reuse only exact current reviews, and verify the selected original source."""
    from autofa.behavior_policy import strictness_for_config
    from autofa.core.dependencies import review_source
    from autofa.metrics.claim_contract import derive_claim_verdict
    from autofa.metrics.review_content import RetainedReviewContent
    from autofa.metrics.review_diagnosis import build_review_diagnosis
    from autofa.metrics.verdicts import behavior_validation, current_behavior_scores

    try:
        scores = current_behavior_scores(
            fn.scores, source=review_source(fn), description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
            behavior_strictness=strictness_for_config(rt.config),
        )
        validation = behavior_validation(
            scores, behavior_strictness=strictness_for_config(rt.config))
    except (KeyError, TypeError, ValueError, OverflowError):
        return []  # Malformed historical reviews cannot establish a spending prerequisite.
    if (validation["status"] in {"stale", "legacy", "unjudged", "invalid", "skipped_pre_filter"}
            or validation["confirmed"] or validation["rejected"] or validation["processing_errors"]
            or scores.get("errors") or scores.get("parent_errors")):
        return []
    if scores.get("evaluation_run_id") not in (None, rt.run.run_id):
        return []  # Imported reviews must not prevent exploration of a different corpus.
    rows = scores.get("behavior_verdicts") or []
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or row.get("verdict") != "uncertain"
                   or not isinstance(row.get("item_id"), str) for row in rows)):
        return []
    if validation["uncertain"] != len(rows) or validation["sampled"] != len(rows):
        return []
    contract = scores.get("claim_contract")
    try:
        if any(derive_claim_verdict(contract, row) is not None for row in rows):
            return []
    except (KeyError, TypeError, ValueError):
        return []
    requirements = {
        row["id"]: row["text"] for row in (contract or {}).get(
            "requirements", []) if isinstance(row, dict) and "id" in row and "text" in row
    }
    gaps = []
    selected = []
    for row in rows:
        assessments = [a for a in row.get("claim_assessments") or []
                       if a.get("status") == "unknown"]
        if not assessments or any(a.get("unknown_reason") != "evidence_unavailable"
                                  for a in assessments):
            return []  # Reading/ambiguity or processing work may still resolve the review.
        if len(selected) < MAX_SUPPORT_ITEMS and row.get("snapshot_id"):
            selected.append(row)
    if not selected:
        return []
    retained = RetainedReviewContent([fn])
    ids = {row["item_id"] for row in selected}
    try:
        for item in _source_items(rt, ids):
            retained.observe(item)
    except (OSError, ValueError, AttributeError):
        return []  # An outage is not proof that a source prerequisite is absent.
    for row in selected:
        if retained.snapshots[fn.name].get(row["item_id"]) != row["snapshot_id"]:
            return []  # New or unavailable evidence invalidates the old preflight decision.
        for assessment in row["claim_assessments"]:
            if assessment.get("status") == "unknown":
                gaps.append(build_review_diagnosis(
                    row, assessment, requirement=requirements.get(
                        assessment["requirement_id"], assessment["requirement_id"]),
                    name=fn.name,
                ))
    return gaps


def evidence_preflight(rt: Any, fn: Any) -> dict | None:
    """Return an actionable pending check before an otherwise redundant evaluation."""
    prerequisites = validate_prerequisites((fn.meta or {}).get("evidence_prerequisites", []))
    checks = _declared_checks(rt, prerequisites) if prerequisites else []
    gaps = _known_gaps(rt, fn) if fn.scores else []
    missing = [row for row in checks if row["availability"] != "retained"]
    if not missing and not gaps:
        return ({
            "ok": True, "name": fn.name, "status": "declared_sources_retained",
            "evaluation_allowed": True, "source_checks": checks,
            "feedback": "The declared source locations are retained on these selected items. "
                        "This does not establish the facts; semantic review still applies.",
        } if checks else None)
    unknown = any(row["availability"] == "unknown" for row in missing)
    feedback = (
        "Evidence prerequisite check is pending: restore source access or correct the selected "
        "source reference, then retry. Source unavailability does not prove absence."
        if unknown else
        "The selected evidence does not establish a required premise. Inspect the exact "
        "evidence checks below before paying for more corpus predictions. Provide retained "
        "support, revise the claim to what is evidenced, or defer it."
    )
    return {
        "ok": True, "name": fn.name, "status": "evidence_prerequisite_pending",
        "evaluation_allowed": False, "provisional": True, "controls": [],
        "feedback": feedback, "source_checks": checks, "known_review_gaps": gaps,
        "scope": "Selected original support/review items only; not a corpus-wide absence claim.",
        "next_action": (
            missing[0].get("next_action") if missing and missing[0].get("next_action") else
            gaps[0]["next_action"] if gaps else
            {"tool": "get_item", "arguments": {"item_id": missing[0]["item_id"]},
             "instruction": "Restore source access and inspect this support item, then retry."}
        ),
        "details_tool": {"tool": "get_function_feedback", "arguments": {"name": fn.name}},
    }
