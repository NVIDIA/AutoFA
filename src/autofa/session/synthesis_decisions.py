# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auditable discovery decisions, without another judge or extra search budget.

The agent supplies the scientific interpretation. This module checks its shape,
source references and executed diagnostics; author controls never establish a
causal grade. An untested, insufficiently defined relation remains an open lead.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

DECISION_OUTCOMES = (
    "no_shared_relation", "incompatible_causes", "evidence_contradicts",
    "insufficient_definition", "evidence_unavailable", "implementation_failed",
    "classifier_limited",
)
UNRESOLVED_OUTCOMES = frozenset({
    "insufficient_definition", "evidence_unavailable", "implementation_failed",
    "classifier_limited",
})
FAMILY_DECISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["outcome", "reason", "next_action"],
    "properties": {
        "outcome": {"type": "string", "enum": list(DECISION_OUTCOMES)},
        "shared_relation": {"type": "string", "maxLength": 4000},
        "observable_conditions": {"type": "array", "maxItems": 12,
                                  "items": {"type": "string", "minLength": 1,
                                            "maxLength": 1000}},
        "unavailable_evidence": {"type": "array", "maxItems": 12,
                                 "items": {"type": "string", "minLength": 1,
                                           "maxLength": 1000}},
        "support_item_ids": {"type": "array", "maxItems": 24, "uniqueItems": True,
                             "items": {"type": "string", "minLength": 1}},
        "negative_control_ids": {"type": "array", "maxItems": 24, "uniqueItems": True,
                                 "items": {"type": "string", "minLength": 1}},
        "reason": {"type": "string", "minLength": 1, "maxLength": 4000},
        "next_action": {"type": "string", "minLength": 1, "maxLength": 2000},
        "experiment_id": {"type": "string", "minLength": 1},
        "comparison_id": {"type": "string", "minLength": 1},
        "source_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
    },
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f"decision.{name} must be {'a nonempty' if required else 'a'} "
                         f"string of at most {limit} characters")
    return value.strip()


def checked_decision(rt: Any, raw: Any, rows: list[dict], *, required: bool = False) -> dict | None:  # noqa: C901
    """Check decision citations and diagnostics, not the agent's causal conclusion."""
    if raw is None:
        if required:
            raise ValueError("no_coherent_family requires a structured decision: outcome, reason "
                             "and next_action. Breadth alone is not a refutation; use "
                             "insufficient_definition for an untested shared relation.")
        return None
    if not isinstance(raw, dict):
        raise ValueError("decision must be an object")
    unknown = set(raw) - set(FAMILY_DECISION_SCHEMA["properties"])
    if unknown:
        raise ValueError("Unknown decision fields: " + ", ".join(sorted(unknown)))
    outcome = raw.get("outcome")
    if outcome not in DECISION_OUTCOMES:
        raise ValueError("decision.outcome must be one of: " + ", ".join(DECISION_OUTCOMES))
    value = {
        "outcome": outcome,
        "shared_relation": _text(raw.get("shared_relation", ""), "shared_relation", 4000),
        "reason": _text(raw.get("reason"), "reason", 4000, required=True),
        "next_action": _text(raw.get("next_action"), "next_action", 2000, required=True),
    }
    for name, limit in (("observable_conditions", 12), ("unavailable_evidence", 12),
                        ("support_item_ids", 24),
                        ("negative_control_ids", 24)):
        entries = raw.get(name, [])
        if (not isinstance(entries, list) or len(entries) > limit
                or any(not isinstance(item, str) or not item.strip() for item in entries)
                or len(set(entries)) != len(entries)):
            raise ValueError(f"decision.{name} must be at most {limit} distinct nonempty strings")
        value[name] = [_text(item, name, 1000, required=True) for item in entries]
    cited = {row["item_id"]: row for row in rows}
    supports, negatives = set(value["support_item_ids"]), set(value["negative_control_ids"])
    if supports & negatives:
        raise ValueError("A decision control cannot be both supporting and negative")
    if (supports | negatives) - set(cited):
        raise ValueError("Decision control IDs must cite this call's checked evidence")
    if any(cited[item]["role"] != "support" for item in supports):
        raise ValueError("Decision support IDs must cite evidence marked support")
    if any(cited[item]["role"] not in {"boundary", "unrelated"} for item in negatives):
        raise ValueError("Negative controls must cite boundary or unrelated evidence")
    if outcome in {"insufficient_definition", "implementation_failed", "classifier_limited"}:
        if not value["shared_relation"] or not supports:
            raise ValueError("A supported unresolved relation needs shared_relation and "
                             "support_item_ids; otherwise explain no_shared_relation")
    if outcome in {"incompatible_causes", "evidence_contradicts"} and not value["shared_relation"]:
        raise ValueError("Specify the shared_relation contradicted by the inspected evidence")
    if outcome == "evidence_unavailable" and (
            not value["unavailable_evidence"]
            or not any(row["role"] == "uncertain" for row in rows)):
        raise ValueError("evidence_unavailable needs unavailable_evidence and checked uncertain "
                         "source evidence; preserve the actual missing premise")
    for key in ("source_sha256", "experiment_id", "comparison_id"):
        if key in raw:
            value[key] = _text(raw[key], key, 256, required=True)
    source = value.get("source_sha256")
    if source and not re.fullmatch(r"[a-f0-9]{64}", source):
        raise ValueError("decision.source_sha256 must be a SHA-256 hex digest")
    if (outcome in {"implementation_failed", "classifier_limited"}
            or value.get("experiment_id") or value.get("comparison_id")):
        if not source or not supports or not negatives or not value["observable_conditions"]:
            raise ValueError("A tested decision needs source_sha256, observable_conditions, "
                             "support_item_ids and negative_control_ids")
    if outcome == "implementation_failed" and not value.get("experiment_id"):
        raise ValueError("implementation_failed requires a saved smoke experiment_id")
    if outcome == "classifier_limited" and not value.get("comparison_id"):
        raise ValueError("classifier_limited requires a completed classifier comparison_id")
    if value.get("experiment_id"):
        from autofa.session.diagnostic_receipts import read_diagnostic_receipt

        receipt = read_diagnostic_receipt(rt, value["experiment_id"])
        _check_trial(receipt, value, rt, complete=receipt.get("complete") is True)
        if outcome == "implementation_failed" and receipt.get("checks_passed") is not False:
            raise ValueError("The saved smoke experiment does not establish a control failure")
        value["experiment"] = {
            key: receipt.get(key) for key in (
                "source_sha256", "execution_identity", "item_ids", "checks_passed", "complete",
            )
        }
    if value.get("comparison_id"):
        from autofa.session.classifier_profiles import read_classifier_comparison

        receipt = read_classifier_comparison(rt, value["comparison_id"])
        _check_trial(receipt, value, rt, complete=receipt.get("status") == "complete")
        if outcome == "classifier_limited" and receipt.get("outcome") != "capability_improved":
            raise ValueError("The saved classifier comparison did not demonstrate improvement")
        value["comparison"] = {
            key: receipt.get(key) for key in ("profile", "outcome", "item_ids", "status")
        }
    value["evidence_scope"] = (
        "Agent interpretation of checked source evidence; controls are "
        "development diagnostics, not independent causal validation."
    )
    return value


def _check_trial(receipt: dict, decision: dict, rt: Any, *, complete: bool) -> None:
    if (receipt.get("run_id") != rt.run.run_id or not complete
            or receipt.get("source_sha256") != decision["source_sha256"]):
        raise ValueError("Decision experiment must be complete and match this run and source")
    controls = set(decision["support_item_ids"]) | set(decision["negative_control_ids"])
    expected = receipt.get("expected_outputs") or {}
    if (not controls <= set(receipt.get("item_ids") or [])
            or any(expected.get(item) is not True for item in decision["support_item_ids"])
            or any(expected.get(item) is not False for item in decision["negative_control_ids"])):
        raise ValueError("Decision controls do not match the saved experiment's declared controls")


def record_decision(state: dict, decision: dict, rows: list[dict], *, unresolved: bool) -> dict:
    """Persist one record per checked evidence/relation/outcome, not per name or round."""
    evidence_key = _digest(sorted([
        {"item_id": row["item_id"], "task_id": row.get("task_id"),
         "quotes": sorted((q["field"], q["quote"]) for q in row.get("quotes", []))}
        for row in rows
    ], key=lambda row: row["item_id"]))
    identifier = _digest([evidence_key, decision["shared_relation"], decision["outcome"],
                          decision.get("source_sha256"), decision.get("experiment_id"),
                          decision.get("comparison_id")])
    records = state.setdefault("decision_records", {})
    previous = records.get(identifier, {})
    revisions = list(previous.get("earlier_interpretations", []))
    if previous.get("decision") and previous["decision"] != decision:
        revisions.append({"round": previous["last_round"], "decision": previous["decision"]})
    record = {
        "id": identifier, "evidence_key": evidence_key, "decision": decision,
        "first_round": previous.get("first_round", state["round"]),
        "last_round": state["round"], "requests": previous.get("requests", 0) + 1,
        "item_ids": [row["item_id"] for row in rows],
        "status": "unresolved" if unresolved else "recorded",
        "earlier_interpretations": revisions,
    }
    records[identifier] = record
    if unresolved:
        state["pending_synthesis_decision"] = identifier
    else:
        for prior in records.values():
            if (prior["evidence_key"] == evidence_key and prior["status"] == "unresolved"
                    and prior["decision"]["shared_relation"] == decision["shared_relation"]):
                prior["status"] = "superseded_by_evidence_decision"
        if records.get(state.get("pending_synthesis_decision"), {}).get("status") != "unresolved":
            state.pop("pending_synthesis_decision", None)
    return record


def pending_decision(state: dict) -> dict | None:
    """Read actionable unresolved guidance without granting budget or counting progress."""
    record = state.get("decision_records", {}).get(state.get("pending_synthesis_decision"))
    if not record or record.get("status") != "unresolved":
        return None
    decision = record["decision"]
    outcome = decision["outcome"]
    repeats = sum(row.get("requests", 1) for row in state["decision_records"].values()
                  if row["evidence_key"] == record["evidence_key"]
                  and row["decision"]["outcome"] == "insufficient_definition")
    if outcome == "insufficient_definition":
        instruction = (
            "Operationalize the proposed relation: state observable conditions, implement a "
            "bounded candidate, and smoke-test supporting examples and negative controls. "
            "Record the candidate as a family with prior_decision_id; breadth alone is not "
            "evidence against it."
        )
        if repeats > 1:
            instruction += (" The same evidence has already reached this unresolved conclusion. "
                            "Choose a different supported subset or a concrete testable "
                            "distinction; "
                            "do not repeat the same prose rejection. No new allowance is granted.")
    elif outcome == "evidence_unavailable":
        instruction = ("Retrieve the missing decisive evidence or explicitly defer an already "
                       "recorded family; unavailable evidence does not refute the mechanism.")
    else:
        instruction = (
            "Inspect the saved diagnostic failure and the source evidence. Repair the "
            "implementation or use classifier_profiles for a bounded capability comparison "
            "or adoption when warranted; "
            "keep this mechanism unresolved. Assess or explicitly defer the recorded family."
        )
    return {
        "id": record["id"], "outcome": outcome, "shared_relation": decision["shared_relation"],
        "item_ids": record["item_ids"], "reason": decision["reason"],
        "proposed_next_action": decision["next_action"], "instruction": instruction,
        "requests_for_same_evidence": (
            repeats if outcome == "insufficient_definition" else record["requests"]),
        "counts_as_completed_investigation": False,
        "reference": f"family_discovery.json#/decision_records/{record['id']}",
    }


def continue_decision(state: dict, rows: list[dict], family_name: str,
                      identifier: str = "") -> str | None:
    """Link an operationalized family to an earlier open decision without deleting it."""
    if not identifier:
        return None
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{64}", identifier):
        raise ValueError("prior_decision_id must be a saved decision ID")
    pending = state.get("decision_records", {}).get(identifier)
    if not pending or pending.get("status") != "unresolved":
        raise ValueError("prior_decision_id must reference an unresolved saved decision")
    if not set(pending["decision"]["support_item_ids"]) <= {row["item_id"] for row in rows}:
        raise ValueError("Operationalizing a prior decision must preserve its checked support")
    pending.update(status="under_investigation", family_name=family_name)
    if state.get("pending_synthesis_decision") == identifier:
        state.pop("pending_synthesis_decision", None)
    return identifier


def finish_decision(state: dict, survey: dict, outcome: str) -> None:
    """Keep the outcome linked to the native family assessment or explicit deferral."""
    record = state.get("decision_records", {}).get(survey.get("decision_id"))
    if record is not None:
        record.update(status=outcome, completed_round=state["round"])
