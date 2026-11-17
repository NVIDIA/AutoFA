# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned failure claims and deterministic aggregation of their assessments.

The contract is built before reviewing any datapoint. It preserves literal claim
text and can freeze source-anchored defining clauses supplied by a compiler that
has not seen the datapoints. Judges assess each defining clause, the complete
source text and fixed checks; no per-item ``optional`` flag weakens requirements.

This module validates structure and aggregates recorded assessments. Evidence
resolution and citation validation remain the responsibility of EvidenceReview;
it cannot establish whether a model's semantic assessment is itself accurate.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from autofa.behavior_policy import acceptance_requirement_id, validate_behavior_strictness

JUDGE_CONTRACT_VERSION = "behavior-claim-v6"

# Product acceptance is separate from faithful compilation of the literal claim.
# A true description of an incident is insufficient to explain why a task failed.
FAILURE_CONTRIBUTION = {
    "id": "failure_contribution",
    "kind": "acceptance_check",
    "text": (
        "Establish that the detected behavior explains a root cause of the task failure, "
        "rather than merely a symptom or incidental error. An earlier decision may qualify "
        "even when the visible failure occurs later; the last error need not be the cause. "
        "Restating failure, expressing doubt, or submitting without another check is not "
        "itself an explanation. A category that only restates the outcome is not a causal "
        "mechanism. A causal family may cover alternative technical mistakes when its "
        "shared causal connection is explicit. Establish the actual member mechanism "
        "for this item; do not demand every alternative or one identical technical "
        "mistake across items. Breadth alone does not invalidate a causal explanation. "
        "Identify the concrete mistaken decision or omission and "
        "show how it produced or left uncorrected the specific task-breaking defect. "
        "Identify the task requirement left unmet, independent evidence of the "
        "actual final defect or incompletion, and the causal chain from this behavior to "
        "that consequence. Independent evidence need not include a grader result or an "
        "observed test failure: an explicit task requirement and inspected evidence of the "
        "submitted artifact or state can establish a defect when the violation follows "
        "directly from its construction. Cite the requirement, the concrete violation, "
        "and evidence that this was the submitted state. Passing a narrower local check "
        "does not negate a distinct violated requirement. Suspected hidden tests, the "
        "assistant's unsupported self-assessment, and unfinished intermediate artifacts "
        "are not proof. Assess every other defining clause independently. "
        "Inspect later recovery and competing explanations. The behavior "
        "need not be the sole cause, but its occurrence, wasted turns, statistical association, "
        "or an external failure label alone do not establish a contribution. A recovered "
        "incident with no remaining effect on task completion, including recovery followed "
        "by an unrelated mistake, is contradicted. A successful task is contradicted. "
        "A lasting effect such as budget exhaustion preventing a required action needs "
        "direct evidence; do not infer it from retry counts. Missing final-state or causal "
        "evidence is unknown. A record stopping at a recoverable intermediate error does "
        "not itself establish task termination, abandonment or budget exhaustion. Without "
        "evidence the task actually ended or the consequence became irreversible, mark "
        "that attribution unknown. This condition is mandatory even when the original claim "
        "describes only a transient error or makes no final-outcome assertion; it cannot "
        "be supported as not applicable. Cite the incident, the ending, and evidence of "
        "the connection or recovery. Do not invent a different failure mechanism to "
        "justify this detector's match."
    ),
}

FAILURE_RELEVANCE = {
    "id": "failure_relevance",
    "kind": "acceptance_check",
    "text": (
        "Establish an observable mistake, unsupported decision, or premature abandonment "
        "and a plausible connection to failure grounded in this item's task and trace. "
        "Identify the actual problematic behavior and why it puts a task requirement at "
        "risk; deadline language, doubt, a keyword, repetition, or a failed benchmark label "
        "alone are insufficient. A shared behavioral pattern may span different technical "
        "defects and tasks. Do not demand a proven causal chain to a specific final defect "
        "or proof that this behavior caused the benchmark outcome. Review relevant later "
        "recovery and contrary evidence: a harmless mention, a sound decision, or complete "
        "recovery with no remaining task-relevant problem contradicts this check. A passing "
        "outcome alone does not erase an evidenced task-relevant mistake, but must not be "
        "ignored where it resolves the alleged problem. An observed premature decision "
        "can qualify without proving actual task termination; do not infer abandonment "
        "from a record that merely stops. Missing evidence of the behavior or its contextual "
        "relevance is unknown. Stronger causes, consequences and exhaustive assertions "
        "explicitly stated in the claim remain mandatory and must be established separately."
    ),
}


def acceptance_requirement_ids(behavior_strictness: str = "causal") -> tuple[str, ...]:
    """Additional product checks, separate from the literal claim's requirements."""
    requirement = acceptance_requirement_id(behavior_strictness)
    return (requirement,) if requirement is not None else ()


def contract_behavior_strictness(contract: dict[str, Any]) -> str:
    """Unannotated v6 contracts retain their original causal interpretation."""
    return validate_behavior_strictness(contract.get("behavior_strictness", "causal"))

# These checks are part of the versioned contract, not trace-dependent extraction.
# "Supported" means the original claim survives the check. A dimension that the
# original text does not constrain does not create an additional failure claim.
_CHECKS = (
    (
        "prerequisites",
        "Check every prerequisite and contextual condition asserted by the complete "
        "description and mechanism, including instructions, authoritative information, "
        "and tool availability where claimed. Do not replace a claimed absence of "
        "authority with mere calculation or writing. Preserve alternatives, examples, "
        "and quantifiers. If the claim imposes no prerequisite, explain that rather "
        "than inventing one; necessary unavailable evidence is unknown.",
    ),
    (
        "actor_attribution",
        "Verify the actor responsible for each claimed action using original roles "
        "and actor-specific tool schemas where relevant. A user or simulator action "
        "does not support a claim about the assistant. Missing actor or schema evidence "
        "is unknown only when needed to establish the original claim.",
    ),
    (
        "causal_link",
        "Check every causal connection asserted by the complete description and "
        "mechanism. Co-occurring actions, symptoms, or grader outcomes alone do not "
        "establish causation. If the claim asserts no causal connection, explain "
        "that rather than demanding one. Preserve natural quantifiers: an individual "
        "trace need not establish a population frequency, but must establish the "
        "claimed local failure and its defining conditions.",
    ),
    (
        "outcome_and_recovery",
        "Check each claimed consequence against event order, later corrections, "
        "and the actual ending. A recovered call can support a claim that an invalid "
        "call occurred, but does not establish permanent abandonment or an incorrect "
        "final state. Do not add an outcome requirement absent from the original "
        "claim. Claimed incorrect values require independent evidence they are "
        "incorrect; the assistant's disputed assumptions do not establish the "
        "correct reference value. Internal inconsistency alone does not prove an "
        "externally incorrect state when a necessary premise remains unknown.",
    ),
    (
        "counterevidence",
        "Review contrary instructions, authoritative sources, later actions, and "
        "other relevant context against ALL defining parts of the complete claim. "
        "Record decisive counterevidence and explain any reconciliation. An actual "
        "contradiction to a defining condition is contradicted; missing decisive "
        "evidence is unknown. Neither may be relegated to a calibration suggestion "
        "while marking the claim supported. Do not redefine a defining condition "
        "as incidental for an individual trace.",
    ),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(payload: dict[str, Any]) -> str:
    return _digest(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _requirements(
    description: str, mechanism: str, behavior_strictness: str,
) -> list[dict[str, str]]:
    acceptance = {"failure_contribution": FAILURE_CONTRIBUTION,
                  "failure_relevance": FAILURE_RELEVANCE}
    return [
        {"id": key, "kind": "claim", "text": value}
        for key, value in (("description", description), ("mechanism", mechanism))
        if value.strip()
    ] + [{"id": key, "kind": "review_check", "text": text} for key, text in _CHECKS] + [
        dict(acceptance[key]) for key in acceptance_requirement_ids(behavior_strictness)
    ]


def _defining_clauses(
    clauses: Any, *, description: str, mechanism: str,
) -> list[dict[str, Any]]:
    if not isinstance(clauses, list) or not clauses:
        raise ValueError("defining_clauses must be a nonempty list")
    required_keys = {"id", "source_field", "quote", "predicate", "required_review_scope"}
    source_fields = {"description": description, "mechanism": mechanism}
    used_ids = {"description", "mechanism", FAILURE_CONTRIBUTION["id"], FAILURE_RELEVANCE["id"],
                *(key for key, _ in _CHECKS)}
    normalized = []
    for clause in clauses:
        if (not isinstance(clause, dict) or not required_keys <= clause.keys()
                or clause.keys() - required_keys - {"review_interval"}):
            raise ValueError(
                "defining clause must contain only id, source_field, quote, predicate, "
                "required_review_scope and conditional review_interval"
            )
        if any(not isinstance(clause[key], str) or not clause[key].strip()
               for key in required_keys):
            raise ValueError("defining clause fields must be nonempty strings")
        clause_id = clause["id"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", clause_id) or clause_id in used_ids:
            raise ValueError(
                "defining clause IDs must be unique and disjoint from base requirements"
            )
        source_field = clause["source_field"]
        if source_field not in source_fields:
            raise ValueError("defining clause source_field must be description or mechanism")
        if clause["quote"] not in source_fields[source_field]:
            raise ValueError("defining clause quote must be an exact substring of its source field")
        scope = clause["required_review_scope"]
        if scope not in {"local", "bounded_interval", "whole_trace"}:
            raise ValueError("required_review_scope must be local, bounded_interval or whole_trace")
        item = dict(clause)
        if scope == "bounded_interval":
            item["review_interval"] = _review_interval(clause.get("review_interval"))
        elif "review_interval" in clause:
            raise ValueError("review_interval is only permitted for bounded_interval scope")
        used_ids.add(clause_id)
        normalized.append(item)
    return normalized


def _review_interval(value: Any) -> dict[str, dict[str, str]]:
    """Freeze semantic boundaries; item-specific event binding belongs to evidence review."""
    if not isinstance(value, dict) or set(value) != {"start", "end"}:
        raise ValueError("bounded_interval requires review_interval with only start and end")
    interval = {}
    for side, absolute_kind in (("start", "run_start"), ("end", "run_end")):
        boundary = value[side]
        if not isinstance(boundary, dict):
            raise ValueError(f"review_interval {side} must be a boundary object")
        kind = boundary.get("kind")
        if kind == absolute_kind:
            if set(boundary) != {"kind"}:
                raise ValueError(f"review_interval {absolute_kind} boundary contains only kind")
        elif kind == "event":
            if set(boundary) != {"kind", "predicate", "position"}:
                raise ValueError("review_interval event needs only kind, predicate and position")
            predicate = boundary["predicate"]
            if not isinstance(predicate, str) or not predicate.strip():
                raise ValueError("review_interval event predicate must be a nonempty string")
            if boundary["position"] not in ("before", "after"):
                raise ValueError("review_interval event position must be before or after")
        else:
            raise ValueError(f"review_interval {side} kind must be {absolute_kind} or event")
        interval[side] = dict(boundary)
    if interval["start"]["kind"] == "run_start" and interval["end"]["kind"] == "run_end":
        raise ValueError("run_start to run_end requires whole_trace, not bounded_interval")
    return interval


def _contract_payload(
    *, description: str, mechanism: str, source_sha256: str,
    defining_clauses: list[dict[str, Any]] | None = None,
    compilation: dict[str, Any] | None = None,
    behavior_strictness: str = "causal",
) -> dict[str, Any]:
    profile = validate_behavior_strictness(behavior_strictness)
    requirements = _requirements(description, mechanism, profile)
    payload = {
        "version": JUDGE_CONTRACT_VERSION,
        "description": description,
        "mechanism": mechanism,
        "source_sha256": source_sha256,
        "requirements": requirements,
    }
    # Preserve already-frozen v6 causal identities. Other profiles are explicitly
    # named and cannot reuse a causal contract or its evidence checkpoints.
    if profile != "causal":
        payload["behavior_strictness"] = profile
        payload["claim_hash"] = _identity(_contract_payload(
            description=description, mechanism=mechanism, source_sha256=source_sha256,
        ))
    if defining_clauses is not None:
        clauses = _defining_clauses(
            defining_clauses, description=description, mechanism=mechanism,
        )
        payload = {
            **payload,
            "claim_hash": payload.get("claim_hash", _identity(payload)),
            "defining_clauses": clauses,
            "requirements": requirements + [
                {"id": clause["id"], "kind": "defining_clause", "text": clause["predicate"],
                 "source_field": clause["source_field"], "quote": clause["quote"],
                 "required_review_scope": clause["required_review_scope"],
                 **({"review_interval": _review_interval(clause["review_interval"])}
                    if "review_interval" in clause else {})}
                for clause in clauses
            ],
        }
        if compilation is not None:
            payload["compilation"] = _compilation_metadata(
                compilation, description=description, mechanism=mechanism,
            )
    elif compilation is not None:
        raise ValueError("compilation metadata requires defining clauses")
    return payload


def _compilation_metadata(
    metadata: Any, *, description: str, mechanism: str,
) -> dict[str, Any]:
    required = {"policy_hash", "model", "exclusions"}
    if (not isinstance(metadata, dict)
            or not required <= metadata.keys()
            or metadata.keys() - required - {"coverage_review"}):
        raise ValueError(
            "compilation needs policy_hash, model, exclusions and optional coverage_review"
        )
    policy_hash = metadata["policy_hash"]
    if not isinstance(policy_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", policy_hash):
        raise ValueError("compilation policy_hash must be a SHA-256 hex digest")
    model = metadata["model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("compilation model must be a nonempty string")
    result = {
        "policy_hash": policy_hash, "model": model,
        "exclusions": _source_reasons(
            metadata["exclusions"], description=description, mechanism=mechanism,
        ),
    }
    if "coverage_review" in metadata:
        result["coverage_review"] = validate_coverage_review(
            metadata["coverage_review"], description=description, mechanism=mechanism,
        )
    return result


def _source_reasons(
    entries: Any, *, description: str, mechanism: str,
) -> list[dict[str, str]]:
    if not isinstance(entries, list):
        raise ValueError("source explanations must be a list")
    source_fields = {"description": description, "mechanism": mechanism}
    normalized = []
    for exclusion in entries:
        if (not isinstance(exclusion, dict)
                or set(exclusion) != {"source_field", "quote", "reason"}):
            raise ValueError("exclusion must contain only source_field, quote, reason")
        if any(not isinstance(v, str) or not v.strip() for v in exclusion.values()):
            raise ValueError("exclusion fields must be nonempty strings")
        field = exclusion["source_field"]
        if field not in source_fields or exclusion["quote"] not in source_fields[field]:
            raise ValueError("exclusion needs an exact source quote from description or mechanism")
        normalized.append(dict(exclusion))
    return normalized


def validate_coverage_review(
    review: Any, *, description: str, mechanism: str,
) -> dict[str, Any]:
    """Validate a trace-blind completeness audit without inferring its approval."""
    if not isinstance(review, dict) or set(review) != {"approved", "issues"}:
        raise ValueError("coverage review must contain only approved and issues")
    approved, issues = review["approved"], review["issues"]
    if type(approved) is not bool or not isinstance(issues, list):
        raise ValueError("coverage review needs a boolean approval and an issues list")
    if approved == bool(issues):
        raise ValueError(
            "coverage approval requires no issues; rejection requires at least one issue"
        )
    normalized = _source_reasons(issues, description=description, mechanism=mechanism)
    return {"approved": approved, "issues": normalized}


def build_claim_contract(
    *, description: str, mechanism: str, source: str = "",
    defining_clauses: list[dict[str, Any]] | None = None,
    compilation: dict[str, Any] | None = None,
    behavior_strictness: str = "causal",
) -> dict[str, Any]:
    """Build the same source-anchored contract for every sampled item.

    The returned JSON-compatible object is immutable by convention; aggregation
    rejects modifications to its text, checks, or identity. Empty claims can still
    be hashed for stale-score detection but cannot produce a substantive verdict.
    Compiled clauses receive their own mandatory assessments. Their contract ID
    includes the compiled predicates, while claim_hash preserves the base identity
    used to detect changes to the literal claim and detector source.
    """
    for key, value in (("description", description), ("mechanism", mechanism), ("source", source)):
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a string")
    payload = _contract_payload(
        description=description, mechanism=mechanism, source_sha256=_digest(source),
        defining_clauses=defining_clauses, compilation=compilation,
        behavior_strictness=behavior_strictness,
    )
    return {**payload, "contract_id": _identity(payload)}


def claim_hash(*, description: str, mechanism: str, source: str = "") -> str:
    """Stable detector/claim fingerprint, independent of selected review strictness.

    Keep the historical v6 fingerprint so changing review policy does not invalidate
    identical detector predictions. Contracts and review policy hashes bind strictness.
    """
    return build_claim_contract(
        description=description, mechanism=mechanism, source=source,
    )["contract_id"]


def _validate_contract(contract: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(contract, dict) or contract.get("version") != JUDGE_CONTRACT_VERSION:
        raise ValueError("unsupported claim contract version")
    description, mechanism = contract.get("description"), contract.get("mechanism")
    if not isinstance(description, str) or not isinstance(mechanism, str):
        raise ValueError("claim contract must preserve the literal description and mechanism")
    source_hash = contract.get("source_sha256")
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(char not in "0123456789abcdef" for char in source_hash)):
        raise ValueError("claim contract is missing its detector source hash")
    if "defining_clauses" in contract and contract["defining_clauses"] is None:
        raise ValueError("defining_clauses must be a nonempty list")
    if "compilation" in contract and contract["compilation"] is None:
        raise ValueError("compilation metadata must be an object")
    payload = _contract_payload(
        description=description, mechanism=mechanism, source_sha256=source_hash,
        defining_clauses=contract.get("defining_clauses"), compilation=contract.get("compilation"),
        behavior_strictness=contract_behavior_strictness(contract),
    )
    if contract != {**payload, "contract_id": _identity(payload)}:
        raise ValueError("claim contract was modified after construction")
    if not description.strip() and not mechanism.strip():
        raise ValueError("claim contract has no description or mechanism to assess")
    return payload["requirements"]


def _assessment_status(assessment: Any, label: str) -> str:
    if not isinstance(assessment, dict):
        raise ValueError(f"{label} must be an assessment object")
    if any(key in assessment for key in ("required", "applicable", "not_applicable")):
        raise ValueError(f"{label} cannot weaken required claim conditions")
    status = assessment.get("status")
    if not isinstance(status, str) or status not in {"supported", "contradicted", "unknown"}:
        raise ValueError(f"{label} status must be supported, contradicted, or unknown")
    rationale = assessment.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{label} needs a nonempty rationale")
    reason = assessment.get("unknown_reason")
    if reason is not None and (
        status != "unknown" or not isinstance(reason, str)
        or reason not in {"review_incomplete", "evidence_unavailable", "ambiguous_evidence"}
    ):
        raise ValueError(f"{label} unknown_reason must describe an unknown assessment: "
                         "review_incomplete, evidence_unavailable, or ambiguous_evidence")
    citations = assessment.get("citations")
    if not isinstance(citations, list) or any(not isinstance(c, dict) for c in citations):
        raise ValueError(f"{label} citations must be a list of evidence references")
    if status != "unknown" and not citations:
        raise ValueError(f"{label} needs citations for a {status} assessment")
    counterevidence = assessment.get("counterevidence", [])
    if (not isinstance(counterevidence, list)
            or any(not isinstance(c, dict) for c in counterevidence)):
        raise ValueError(f"{label} counterevidence must be a list of evidence references")
    return status


def validate_compiled_assessment_scope(
    contract: dict[str, Any], row: dict[str, Any],
) -> None:
    """Bind live summary/check blockers to audited defining conditions.

    Frozen v6 contracts and claim fingerprints remain unchanged. The live judge
    policy versions this additional assessment requirement; historical assessment
    readers must not reinterpret saved rows as if they supplied these references.
    Product acceptance checks remain independently mandatory and need no mapping
    to a literal clause.
    """
    compilation = contract.get("compilation") or {}
    if not (
        contract.get("defining_clauses")
        and compilation.get("coverage_review", {}).get("approved") is True
    ):
        return
    requirements = {r["id"]: r for r in contract["requirements"]}
    defining_ids = {key for key, r in requirements.items() if r["kind"] == "defining_clause"}
    assessments = {a["requirement_id"]: a for a in row["claim_assessments"]}
    for key, assessment in assessments.items():
        requirement = requirements[key]
        if requirement["kind"] not in {"claim", "review_check"}:
            continue
        for finding in [assessment, *assessment.get("components", [])]:
            references = finding.get("defining_requirement_ids")
            needs_mapping = finding["status"] != "supported"
            if references is None and not needs_mapping:
                continue
            if (
                not isinstance(references, list)
                or any(not isinstance(ref, str) or ref not in defining_ids for ref in references)
                or len(set(references)) != len(references)
                or (needs_mapping and not references)
            ):
                raise ValueError(
                    f"{key}: provide defining_requirement_ids identifying the frozen "
                    "defining condition(s) behind this finding. Generic checks and "
                    "source summaries cannot add conditions from compilation.exclusions. "
                    "Assess a policy-only concern in the separate acceptance check."
                )
            if needs_mapping and not any(
                assessments[ref]["status"] == finding["status"] for ref in references
            ):
                raise ValueError(
                    f"{key}: its {finding['status']} finding must be recorded on at least "
                    "one referenced defining condition with the same status. Assess only "
                    "the actual affected condition; do not weaken or mark every clause "
                    "unknown. If the concern is excluded commentary, remove that extra "
                    "condition from this check; acceptance checks remain mandatory."
                )


def derive_claim_verdict(contract: dict[str, Any], row: dict[str, Any]) -> bool | None:
    """Derive truth from complete assessments, never a model-provided ``correct``.

    A defining contradiction rejects the claim even when other conditions remain
    unknown. With no contradiction, any unknown makes the verdict uncertain.
    Malformed assessments are processing errors, raised for the caller's bounded
    repair flow. Citation content must also pass EvidenceReview before acceptance.
    """
    requirements = _validate_contract(contract)
    if not isinstance(row, dict) or row.get("claim_contract_id") != contract["contract_id"]:
        raise ValueError("claim_contract_id does not match the frozen claim contract")
    assessments = row.get("claim_assessments")
    if not isinstance(assessments, list):
        raise ValueError("claim_assessments must list every required claim assessment")
    by_id = {}
    required_ids = {requirement["id"] for requirement in requirements}
    for assessment in assessments:
        key = assessment.get("requirement_id") if isinstance(assessment, dict) else None
        if not isinstance(key, str) or key not in required_ids:
            raise ValueError(f"unknown claim assessment requirement_id: {key!r}")
        if key in by_id:
            raise ValueError(f"duplicate claim assessment: {key}")
        by_id[key] = assessment
    missing = required_ids - by_id.keys()
    if missing:
        raise ValueError(f"missing claim assessments: {', '.join(sorted(missing))}")

    statuses = []
    for requirement in requirements:
        key = requirement["id"]
        assessment = by_id[key]
        statuses.append(_assessment_status(assessment, key))
        components = assessment.get("components", [])
        if not isinstance(components, list):
            raise ValueError(f"{key} components must be a list")
        if components and requirement["kind"] != "claim":
            raise ValueError(f"{key} is not a literal claim to subdivide")
        for index, component in enumerate(components):
            label = f"{key} component {index + 1}"
            statuses.append(_assessment_status(component, label))
            if "components" in component:
                raise ValueError(f"{label} cannot contain nested components")
            text = component.get("claim")
            if not isinstance(text, str) or not text.strip() or text not in requirement["text"]:
                raise ValueError(f"{label} claim must be an exact substring of the original clause")

    if "contradicted" in statuses:
        return False
    if "unknown" in statuses:
        return None
    return True
