# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic repair advice grounded in the current, final claim assessments.

This is diagnostic feedback, not a new judgment or an automatic claim rewrite.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from math import isfinite
from typing import Any

from autofa.behavior_policy import acceptance_requirement_id, strictness_for_scores
from autofa.core.dependencies import review_source
from autofa.metrics.claim_calibration import (
    DEVELOPMENT_REVIEW_GUIDANCE,
    parse_claim_calibration,
    review_issue_guidance,
    review_issue_type,
)
from autofa.metrics.claim_contract import claim_hash, derive_claim_verdict
from autofa.metrics.review_diagnosis import EDITORIAL_LIMIT, build_review_diagnosis
from autofa.metrics.verdicts import behavior_validation, current_behavior_scores

SCHEMA_VERSION = "detector-repair-feedback-v1"
MAX_EXAMPLES = 3
MAX_CITATIONS = 3
MAX_EXCERPT_CHARS = 800
_STATUSES = {"supported", "contradicted", "unknown"}


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(serialized).hexdigest()


def _references(value: Any) -> tuple[list[dict[str, Any]], int]:
    refs = [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []
    result = []
    for ref in refs[:MAX_CITATIONS]:
        # Preserve exact citation locations and source text; mark any excerpt explicitly.
        copied = {k: deepcopy(ref[k]) for k in ("citation_id", "field", "start", "end", "quote")
                  if k in ref}
        if isinstance(copied.get("quote"), str) and len(copied["quote"]) > MAX_EXCERPT_CHARS:
            copied["quote_chars"] = len(copied["quote"])
            copied["quote"] = copied["quote"][:MAX_EXCERPT_CHARS]
            copied["quote_truncated"] = True
        result.append(copied)
    return result, len(refs)


def _example(
    row: dict[str, Any], assessment: dict[str, Any], *, requirement: str, name: str,
) -> dict[str, Any]:
    rationale = str(assessment.get("rationale") or row.get("rationale") or "")
    citations, n_citations = _references(assessment.get("citations"))
    counterevidence, n_counter = _references(assessment.get("counterevidence"))
    result = {
        "item_id": str(row["item_id"]), "rationale": rationale[:MAX_EXCERPT_CHARS],
        "citations": citations, "counterevidence": counterevidence,
        "total_citations": n_citations, "total_counterevidence": n_counter,
    }
    if len(rationale) > MAX_EXCERPT_CHARS:
        result["rationale_truncated"] = True
        result["rationale_chars"] = len(rationale)
    if assessment.get("status") in {"unknown", "contradicted"}:
        result["diagnosis"] = build_review_diagnosis(
            row, assessment, requirement=requirement, name=name,
        )
    return result


def _investigation(
    finding: dict[str, Any], *, behavior_strictness: str = "causal",
) -> dict[str, Any]:
    example = finding["examples"][0]
    item_id = example["item_id"]
    status = finding["status"]
    if status == "processing_error":
        question = "Can the item evidence be loaded before retrying the failed review?"
    elif status == "unknown" and finding.get("unknown_reason") == "review_incomplete":
        question = (
            "Which available ranges remain unread in saved coverage? Use only any remaining "
            "original review budget with the frozen boundaries. If exhausted, retain the "
            "inconclusive verdict and defer or gather new evidence; do not resubmit the "
            "unchanged claim to reset the budget."
        )
    elif finding["requirement_id"] == "failure_contribution":
        if status == "contradicted":
            question = (
                "Which observed condition excludes recovered or unrelated incidents from this "
                "detector? What evidence would establish a different concrete cause of the "
                "task-breaking defect, if the claim needs revision?"
            )
        else:
            question = (
                "What evidence connects the detected behavior to a concrete task-breaking "
                "defect, after accounting for recovery and competing causes?"
            )
    elif status == "contradicted":
        question = (
            f"What observed condition separates this hard negative from '{finding['requirement']}'?"
        )
    else:
        question = f"Which recorded facts establish or refute '{finding['requirement']}'?"
    issue_type = review_issue_type(status, finding.get("unknown_reason"))
    if issue_type:
        question += " " + review_issue_guidance(
            issue_type, behavior_strictness=behavior_strictness,
        )
    question += f" Recorded review: {example['rationale']}"
    # Judge-only trajectory_text has no corresponding raw item field. Use the real
    # trajectory API, rather than suggesting an invalid read_field_chunk field.
    refs = example["counterevidence"] + example["citations"]
    field = next((ref.get("field") for ref in refs if ref.get("field")), None)
    if field == "trajectory_text":
        tool, arguments = "get_trajectory", {
            "item_id": item_id, "include_observations": True, "offset": 0, "limit": 50,
        }
    elif field and field != "detector_source":
        tool, arguments = "read_field_chunk", {
            "item_id": item_id, "field": field, "view": "detector", "offset": 0, "limit": 80000,
        }
        question += (
            " Locate the cited quote and paginate if needed; frozen-review offsets may differ."
        )
    else:
        tool, arguments = "get_item", {"item_id": item_id, "include_trace": True}
    result = {"finding_id": finding["id"], "issue_type": issue_type,
              "question": question, "tool": tool, "arguments": arguments}
    if example.get("diagnosis"):
        diagnosis = example["diagnosis"]
        action = diagnosis["next_action"]
        # Route counterexamples to repair and evidence gaps to retrieval/defer;
        # availability metadata must not turn a rejected claim into an unknown.
        result.update(
            diagnosis=diagnosis, action=action["kind"],
            tool=action["tool"], arguments=action["arguments"],
            next_step=action["instruction"], if_unavailable=action["if_unavailable"],
        )
    return result


def claim_compilation_rejected(scores: dict) -> bool:
    """A recorded compiler validation failure can require rewriting the claim itself.

    Ambiguous original wording can be clarified at once. A decomposition failure
    first uses its shared explicit retry; if that also fails, a meaning-preserving
    rewording is the remaining escape.
    """
    compilation = scores.get("claim_compilation") or {}
    failure = compilation.get("failure") or {}
    if failure:
        return (failure.get("kind") == "claim_input_ambiguity"
                or (failure.get("kind") == "claim_decomposition_error"
                    and failure.get("explicit_retry_used") is True))
    error = compilation.get("error")
    return isinstance(error, str) and "claim compilation failed validation:" in error


def has_complete_causal_support(
    function: dict[str, Any], *, objective_strategy: str, behavior_strictness: str = "causal",
) -> bool:
    """Check complete current assessments, not just an aggregate purity score.

    Development support authorizes an independent confirmation attempt; it does
    not itself establish independent confirmation or validate a repair outcome.
    """
    from autofa.metrics.claude_judge import compatible_judge_policy

    scores = function.get("scores") or {}
    if (scores.get("judge_behavior_strictness", "causal") != behavior_strictness
            or not compatible_judge_policy(
                scores.get("judge_policy_hash"), behavior_strictness=behavior_strictness)
            or scores.get("behavior_judgment_stale")
            or scores.get("behavior_n_review_incomplete")):
        return False
    feedback = build_repair_feedback(function)
    findings = [row for row in feedback["findings"] if row["status"] != "quality_gap"]
    return bool(
        feedback["review_detail_complete"]
        and not feedback["review_integrity_errors"]
        and feedback["recommended_action"] == "retain"
        and findings and all(row["status"] == "supported" for row in findings)
        and (acceptance_requirement_id(behavior_strictness) is None or any(
            row["requirement_id"] == acceptance_requirement_id(behavior_strictness)
            for row in findings))
    )


def detector_execution_findings(function: dict[str, Any]) -> list[dict[str, Any]]:
    """Recorded execution failures authorize code repair, not semantic rejection."""
    scores = function.get("scores") or {}
    fingerprint = claim_hash(
        source=review_source(function), description=function.get("description") or "",
        mechanism=function.get("mechanism") or function.get("description")
        or function.get("name") or "",
    )
    count = scores.get("errors")
    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or scores.get("evaluation_claim_hash") != fingerprint
            or isinstance(count, bool) or not isinstance(count, (int, float))
            or not isfinite(count) or count <= 0 or int(count) != count):
        return []
    examples = scores.get("error_examples")
    if not isinstance(examples, list):
        return []
    findings = []
    for row in examples:
        if (not isinstance(row, dict) or not isinstance(row.get("item_id"), str)
                or not row["item_id"].strip() or not isinstance(row.get("error"), str)
                or not row["error"].strip()):
            continue
        item_id, error = row["item_id"], row["error"]
        finding = {
            "id": "execution_" + _fingerprint([fingerprint, item_id, error])[:16],
            "status": "execution_error", "issue_type": "detector_execution_failure",
            "requirement_id": "detector_execution",
            "requirement": "Execute the stated detector on the recorded item",
            "item_ids": [item_id], "total_examples": 1, "examples_omitted": 0,
            "examples": [{"item_id": item_id, "error": error[:MAX_EXCERPT_CHARS],
                          "rationale": error[:MAX_EXCERPT_CHARS], "citations": [],
                          "counterevidence": []}],
        }
        if not any(saved["id"] == finding["id"] for saved in findings):
            findings.append(finding)
    return sorted(findings, key=lambda finding: finding["id"])


def detector_quality_findings(
    function: dict[str, Any], feedback: dict[str, Any],
) -> list[dict[str, Any]]:
    """Measured quality gaps authorize predicate experiments, not causal rejection."""
    from autofa.metrics.claude_judge import compatible_judge_policy
    from autofa.metrics.discovery_grades import grade_gap_for_scores

    scores = function.get("scores") or {}
    findings = [row for row in feedback["findings"] if row["status"] != "quality_gap"]
    if (not feedback["review_detail_complete"] or feedback["review_integrity_errors"]
            or feedback["recommended_action"] != "retain" or not findings
            or any(row["status"] != "supported" for row in findings)
            or scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or scores.get("evaluation_claim_hash") != feedback["claim_hash"]
            or scores.get("behavior_judgment_stale") or scores.get("behavior_n_review_incomplete")
            or not scores.get("evaluation_run_id")
            or not scores.get("task_recall_corpus_fingerprint")
            or scores.get("errors") or scores.get("parent_errors")
            or not compatible_judge_policy(
                scores.get("judge_policy_hash"),
                behavior_strictness=strictness_for_scores(scores))):
        return []
    panel = scores.get("evaluation_item_ids")
    if scores.get("sampled") is True:
        if (not isinstance(panel, list) or not panel
                or any(not isinstance(item, str) or not item for item in panel)
                or len(panel) != len(set(panel))):
            return []
        cohort = _fingerprint(sorted(panel))
    elif scores.get("sampled") is False and scores.get("evaluation_scope_complete") is True:
        panel = None
        cohort = scores["task_recall_corpus_fingerprint"]
    else:
        return []
    # Bucket examples were produced by this evaluation. They identify controls,
    # not mechanism labels; a passing benchmark outcome is not a contradiction.
    item_ids = sorted({item for bucket in scores.get("buckets") or []
                       if isinstance(bucket, dict)
                       for key in ("example_fail_ids", "example_pass_ids")
                       for item in bucket.get(key) or []
                       if isinstance(item, str) and item and (panel is None or item in panel)})
    if not item_ids:
        return []
    gap = grade_gap_for_scores(scores, target_grade="OUTSTANDING")
    qualification = grade_gap_for_scores(scores, target_grade="GOOD")
    result = []
    for dimension in sorted(gap["dimensions"],
                            key=lambda row: row["key"] not in qualification["bottlenecks"]):
        if (dimension["key"] not in {"recall", "ppv_at_pi0", "matched_failed_tasks"}
                or dimension["status"] != "below_requirement"):
            continue
        result.append({
            "id": "quality_" + _fingerprint([
                feedback["claim_hash"], scores["evaluation_run_id"], cohort,
                dimension, item_ids,
            ])[:16],
            "status": "quality_gap", "issue_type": "detector_quality_improvement",
            "requirement_id": dimension["key"],
            "requirement": "Improve " + dimension["label"] + " while preserving the claim",
            "dimension": deepcopy(dimension), "cohort_fingerprint": cohort,
            "item_ids": item_ids, "total_examples": len(item_ids),
            "examples_omitted": max(0, len(item_ids) - MAX_EXAMPLES),
            "examples": [{"item_id": item, "rationale": (
                "Evaluated cohort control; inspect the trace before proposing a distinction. "
                "Benchmark outcomes do not establish whether the mechanism occurred."
            ), "citations": [], "counterevidence": []} for item in item_ids[:MAX_EXAMPLES]],
        })
    return result


def build_repair_feedback(function: dict[str, Any]) -> dict[str, Any]:
    """Combine separate execution and claim findings without changing saved evidence."""
    result = _build_review_feedback(function)
    from autofa.metrics.diagnostic_review import diagnostic_scores

    supplemental = diagnostic_scores(function.get("scores") or {})
    explicit = ((function.get("scores") or {}).get("diagnostic_review") or {}).get(
        "review_mode") == "diagnostic"
    if (supplemental is not None
            and supplemental.get("claim_hash") == result["claim_hash"]
            and (result["validation"]["status"] not in {
                "stale", "legacy", "unjudged", "invalid", "skipped_pre_filter",
            } or (explicit and result["validation"]["status"] != "skipped_pre_filter"))):
        diagnostic = _build_review_feedback({**function, "scores": supplemental})
        scope = "bounded_development_diagnostic" if explicit else "outcome_diagnostic"
        diagnostic["evidence_scope"] = scope
        diagnostic["excluded_from_adherence"] = True
        diagnostic["excluded_from_confirmed_support"] = True
        id_map = {}
        for finding in diagnostic["findings"]:
            old_id = finding["id"]
            finding["id"] = id_map[old_id] = "diagnostic_" + old_id
            finding["evidence_scope"] = scope
        for investigation in diagnostic["investigations"]:
            investigation["finding_id"] = id_map[investigation["finding_id"]]
            investigation["evidence_scope"] = scope
        result["diagnostic_feedback"] = diagnostic
        # Keep independently labeled findings actionable by the existing repair
        # planner; do not merge verdicts, counts, or integrity checks into the score.
        result["findings"].extend(diagnostic["findings"])
        result["investigations"].extend(diagnostic["investigations"])
        issue = diagnostic["recommended_action"]
        if issue != "retain":
            result["notes"].append(
                "Separate diagnostic examples have unresolved findings. They are excluded "
                "from adherence and confirmed support, but their counterexamples or evidence "
                "gaps must be addressed before retaining the detector."
            )
            actions = [action for action in diagnostic["allowed_actions"]
                       if action not in {"retire", "retry_review"}]
            result["allowed_actions"] = list(dict.fromkeys(
                result["allowed_actions"] + actions + ["gather_evidence"]))
            if issue == "tighten_detector":
                result["recommended_action"] = issue
            elif result["recommended_action"] == "retain":
                result["recommended_action"] = "gather_evidence"
    elif (function.get("scores") or {}).get("diagnostic_evidence_stale"):
        result["notes"].append(
            "Supplemental outcome evidence changed or is unavailable. Its saved findings "
            "are historical; restore or review the frozen diagnostic cohort before retention."
        )
        if result["recommended_action"] == "retain":
            result["recommended_action"] = "gather_evidence"
            result["allowed_actions"] = ["gather_evidence"]
    quality = detector_quality_findings(function, result)
    if quality:
        result["findings"].extend(quality)
        result["allowed_actions"].append("refine_detector")
        result["quality_refinement"] = {
            "action": "refine_detector", "finding_ids": [row["id"] for row in quality],
            "preserve_claim": True, "requires_new_validation": True,
        }
        result["investigations"].extend({
            "finding_id": row["id"], "tool": "get_item",
            "arguments": {"item_id": row["item_ids"][0]},
            "question": " ".join(row["dimension"]["next_steps"]) + (
                " After inspecting the controls and identifying a supported distinction, use "
                "plan_function_repair with action='refine_detector', "
                f"finding_ids={[row['id']]!r}, the inspected evidence_ids from this finding, "
                "and your evidence-based rationale; then submit and compare the revised "
                "predicate. The reviewed claim remains supported."
            ),
        } for row in quality)
        result["notes"].append(
            "The reviewed claim remains supported. A measured quality gap permits "
            "refine_detector to test a better predicate without inventing a causal "
            "counterexample. Preserve the literal claim and detector relationships, compare "
            "on the same scoring cohort, and review the changed detector with fresh evidence "
            "under the existing repair budget. This does not validate an improvement."
        )
    execution = detector_execution_findings(function)
    if not execution:
        return result
    result["findings"] = execution + result["findings"]
    result["allowed_actions"] = list(dict.fromkeys([
        *result["allowed_actions"], "repair_implementation",
    ]))
    result["recommended_action"] = "repair_implementation"
    result["investigations"].insert(0, {
        "finding_id": execution[0]["id"], "tool": "get_item",
        "arguments": {"item_id": execution[0]["item_ids"][0]},
        "question": "Inspect the recorded execution error against the source. Repair parsing "
                    "or linking defects without changing the claim; for a transient provider "
                    "failure restore the service and retry evaluation instead.",
    })
    result["notes"].append(
        "Execution failures are separate from claim findings and do not contradict a "
        "supported mechanism. repair_implementation preserves the claim. After the ordinary "
        "repair budget, two bounded implementation attempts remain available with reproduced "
        "regression checks. Smoke-test the error cases and supported controls, reevaluate, and "
        "review the changed implementation under the selected validation policy. Saved reviews "
        "remain historical; fixing execution alone does not validate the revised detector."
    )
    return result


def _build_review_feedback(function: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Group final findings without altering claims, scores, or saved judgments."""
    name = str(function.get("name") or "")
    description = str(function.get("description") or "")
    mechanism = str(function.get("mechanism") or description or name)
    source = review_source(function)
    fingerprint = claim_hash(source=source, description=description, mechanism=mechanism)
    scores = current_behavior_scores(
        function.get("scores"), source=source, description=description, mechanism=mechanism,
    )
    integrity_errors = []
    try:
        mode = strictness_for_scores(scores)
    except ValueError as exc:
        mode = "causal"
        integrity_errors.append({"error": str(exc)})
    try:
        for key in ("behavior_n_judged", "behavior_n_yes", "behavior_n_no",
                    "behavior_n_uncertain", "behavior_n_errors", "behavior_n_completed"):
            if key in scores and (type(scores[key]) is not int or scores[key] < 0):
                raise ValueError(f"Invalid saved review count: {key}")
        validation = behavior_validation(scores, include_details=False)
    except (TypeError, ValueError, OverflowError) as exc:
        validation = {"status": "invalid", "label": "Malformed saved review counts"}
        integrity_errors.append({"error": str(exc)})
    raw_rows = scores.get("behavior_verdicts") or []
    if not isinstance(raw_rows, list):
        raw_rows = []
        integrity_errors.append({"error": "Saved verdicts must be a list."})
    rows = [row for row in raw_rows if isinstance(row, dict)
            and isinstance(row.get("item_id"), str) and row["item_id"].strip()]
    if len(rows) != len(raw_rows):
        integrity_errors.append({"error": "Saved verdict rows have missing or malformed item IDs."})
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "function_name": name, "claim_hash": fingerprint,
        "validation": validation, "findings": [],
        "review_integrity_errors": integrity_errors, "review_detail_complete": False,
        "diagnostic_item_ids": sorted({str(row["item_id"]) for row in rows}),
        "target_bucket": scores.get("bucket"),
        "target_bucket_note": (
            "Findings assess the function's written description and mechanism on the selected "
            "bucket. A categorical bucket label is not a separate category definition."
        ),
        "allowed_actions": [], "recommended_action": "gather_evidence", "investigations": [],
        "details_tool": {"name": "get_function_feedback", "arguments": {"name": name}},
        "limits": {"examples_per_finding": MAX_EXAMPLES, "citations_per_example": MAX_CITATIONS,
                   "excerpt_chars": MAX_EXCERPT_CHARS},
        "notes": [
            "Examples and excerpts are bounded; total_examples counts all matching reviewed items. "
            "Use get_function_feedback for the full assessments and citations.",
            "Unknown requirements are evidence gaps, not demonstrated false hits. Supported "
            "clauses do not by themselves confirm the whole claim or its causal explanation.",
            ("The failure_contribution acceptance check remains mandatory for every revised "
             "or split claim. Supported defining behavior alone is insufficient; narrowing "
             "the prose to occurrence cannot remove the need to explain a root cause."
             if mode == "causal" else
             "Every revised claim must satisfy the selected " + mode + " validation standard. "
             "A rewrite requires fresh review and does not validate the original claim. "
             "Causal assertions are never waived merely by changing the profile."),
            "Separate a weak hypothesis from a detector mismatch, an incidental "
            "claim condition, missing evidence, or a processing error. A low purity alone "
            "does not identify which needs repair. Apply the selected validation standard; "
            "do not invent an internal belief to make the explanation sound deeper. Test a revised "
            "predicate on inspected positive and contrast item IDs before corpus evaluation.",
        ],
    }
    if validation["status"] == "skipped_pre_filter":
        result.update(
            judge_prefilter=validation["judge_prefilter"], reason=validation.get("reason"),
            recommended_action="reevaluate", next_steps=[
                "If an active repair plan freezes this candidate, call defer_function_repair "
                "before revising or removing it.",
                *validation["next_steps"],
            ],
            judgment_history_only=validation.get("judgment_history_only", False),
        )
        result["notes"].append(
            "Judging was skipped by the score pre-filter. Improve benchmark association and "
            "reevaluate, or remove the detector. Diagnostic IDs and saved judgments are "
            "historical; they do not justify a semantic repair or a processing-review retry "
            "for this evaluation."
        )
        return result
    if validation["status"] in {"stale", "legacy", "unjudged", "invalid"}:
        result["allowed_actions"] = ["gather_evidence"]
        result["notes"].append(
            "Current semantic repair findings are unavailable. Call judge_function on the current "
            "claim before using historical judgments to justify a repair. "
            "Diagnostic IDs are history."
        )
        return result

    contract = scores.get("claim_contract") or {}
    if not isinstance(contract, dict):
        contract = {}
    if contract.get("claim_hash", contract.get("contract_id")) != fingerprint:
        integrity_errors.append({"error": "Saved contract does not match the current claim."})
        result["allowed_actions"] = ["gather_evidence"]
        result["notes"].append(
            "Recover the matching frozen contract or re-judge the current claim."
        )
        return result
    raw_requirements = contract.get("requirements") or []
    if not isinstance(raw_requirements, list):
        raw_requirements = []
    requirements = {r["id"]: r for r in raw_requirements
                    if isinstance(r, dict) and isinstance(r.get("id"), str)}
    groups: dict[tuple[str, ...], dict[str, Any]] = {}

    def add(row, assessment, requirement_id, text, *, component=False):
        status = assessment["status"]
        key = (requirement_id, text, status)
        reason = assessment.get("unknown_reason")
        if reason:
            key += (reason,)
        if key not in groups:
            groups[key] = {
                "id": "finding_" + _fingerprint([SCHEMA_VERSION, fingerprint, *key])[:16],
                "status": status, "requirement_id": requirement_id, "requirement": text,
                "issue_type": review_issue_type(status, reason),
                "examples_by_item": {},
            }
            if reason:
                groups[key]["unknown_reason"] = reason
            if component:
                groups[key]["component"] = True
            requirement = requirements.get(requirement_id, {})
            if requirement.get("quote"):
                groups[key]["claim_quote"] = requirement["quote"]
                groups[key]["source_field"] = requirement.get("source_field")
        example = _example(
            row, {**assessment, "requirement_id": requirement_id}, requirement=text, name=name,
        )
        # Duplicate rows should neither inflate item counts nor make ordering change IDs/advice.
        examples = groups[key]["examples_by_item"]
        old = examples.get(example["item_id"])
        if old is None or _fingerprint(example) < _fingerprint(old):
            examples[example["item_id"]] = example

    final_outcomes = []
    repeated_ids = {item_id for item_id, count in Counter(row["item_id"] for row in rows).items()
                    if count > 1}
    for row in rows:
        if row["item_id"] in repeated_ids:
            integrity_errors.append({"item_id": row["item_id"], "error": "Duplicate item verdict."})
            continue
        outcome = row.get("verdict")
        if isinstance(outcome, str) and outcome in {"error", "processing_error"}:
            final_outcomes.append("error")
            add(row, {"status": "processing_error", "rationale": row.get("validation_error")
                      or row.get("rationale") or "Review processing failed."},
                "review_processing", "A completed evidence review")
            continue  # Never turn a failed or provisional assessment into a semantic finding.
        try:
            if row.get("processing_status") not in {None, "completed"}:
                raise ValueError("Processing status conflicts with semantic verdict")
            derived = derive_claim_verdict(contract, row)
            expected = "yes" if derived is True else "no" if derived is False else "uncertain"
            if outcome != expected:
                raise ValueError("Saved verdict conflicts with the required condition assessments")
        except (TypeError, ValueError, KeyError) as exc:
            integrity_errors.append({"item_id": row["item_id"], "error": str(exc)})
            continue
        final_outcomes.append(outcome)
        assessments = [a for a in row.get("claim_assessments") or []
                       if isinstance(a, dict) and a.get("status") in _STATUSES]
        for assessment in assessments:
            requirement_id = str(assessment.get("requirement_id") or "full_claim")
            text = str(requirements.get(requirement_id, {}).get("text") or requirement_id)
            add(row, assessment, requirement_id, text)
            for component in assessment.get("components") or []:
                if isinstance(component, dict) and component.get("status") in _STATUSES:
                    add(row, component, requirement_id, str(component.get("claim") or text),
                        component=True)

    priorities = {"contradicted": 0, "unknown": 1, "processing_error": 2, "supported": 3}
    acceptance_id = acceptance_requirement_id(mode)
    for group in sorted(groups.values(), key=lambda g: (
        g["status"] == "supported",
        priorities[g["status"]],
        g["requirement_id"] != acceptance_id, g["id"],
    )):
        examples = group.pop("examples_by_item")
        group["total_examples"] = len(examples)
        group["item_ids"] = sorted(examples)
        group["examples"] = [examples[key] for key in sorted(examples)[:MAX_EXAMPLES]]
        group["examples_omitted"] = max(0, len(examples) - MAX_EXAMPLES)
        result["findings"].append(group)
        if group["status"] != "supported":
            result["investigations"].append(_investigation(group, behavior_strictness=mode))

    complete_details = (
        not integrity_errors
        and len(result["diagnostic_item_ids"]) == validation["sampled"]
        and all(final_outcomes.count(outcome) == validation[count] for outcome, count in (
            ("yes", "confirmed"), ("no", "rejected"), ("uncertain", "uncertain"),
            ("error", "processing_errors"),
        ))
    )
    result["review_detail_complete"] = complete_details
    statuses = {group["status"] for group in result["findings"]}
    contribution_statuses = {
        group["status"] for group in result["findings"]
        if group["requirement_id"] == "failure_contribution"
    }
    actions = result["allowed_actions"]
    if "contradicted" in statuses:
        actions.extend(["tighten_detector", "revise_claim", "gather_evidence"])
        result["recommended_action"] = "tighten_detector"
        if "yes" in final_outcomes and "no" in final_outcomes:
            actions.append("split_detector")
        if complete_details and final_outcomes and set(final_outcomes) == {"no"}:
            actions.append("retire")
        result["notes"].append(
            "Contradicted examples are hard negatives for the current full claim. "
            "A revised or split "
            "claim must preserve supported facts and be re-evaluated; these actions are options, "
            "not newly inferred failure mechanisms."
        )
        if "contradicted" in contribution_statuses:
            result["notes"].append(
                "The causal contribution is contradicted on reviewed examples. Tighten the "
                "detector using evidence that excludes recovered or unrelated matches. "
                "Revising the claim requires a separately supported causal explanation and "
                "fresh review; changing only the wording or retaining the observed symptom "
                "does not repair the attribution."
            )
    if "unknown" in statuses:
        if "gather_evidence" not in actions:
            actions.append("gather_evidence")
        completion_only = all(group.get("unknown_reason") == "review_incomplete"
                              for group in result["findings"] if group["status"] == "unknown")
        result["review_completion_only"] = completion_only and "contradicted" not in statuses
        supported_claim = not completion_only and any(
            group["status"] == "supported"
            and (group["requirement_id"] == "description"
                 or requirements.get(group["requirement_id"], {}).get("kind") == "defining_clause")
            and any(example["citations"] for example in group["examples"])
            for group in result["findings"]
        )
        if supported_claim and "revise_claim" not in actions:
            actions.append("revise_claim")
        if completion_only:
            result["notes"].append(
                "All unresolved requirements cite unfinished review of available evidence. "
                "Inspect saved review coverage within any remaining original review budget. "
                "If exhausted, retain the inconclusive verdict and defer or gather new evidence; "
                "do not resubmit the unchanged claim to reset the budget. Unread content "
                "does not justify narrowing or splitting the detector.")
        elif "unknown" in contribution_statuses:
            result["notes"].append(
                "The causal contribution remains unknown. Gather evidence of the specific "
                "task-breaking defect and how the detected decision produced or left it "
                "uncorrected; inspect later recovery and competing causes. A claim revision "
                "may explain a different cause only with its own evidence and fresh review. "
                "Dropping an unestablished causal premise to describe behavior alone does "
                "not satisfy failure_contribution or confirm the original claim."
            )
        elif supported_claim and contribution_statuses == {"supported"}:
            result["notes"].append(
                "Some defining behavior has cited support while other conditions remain unknown. "
                "A narrower claim may describe those supported facts without the unestablished "
                "premise only if it still explains a concrete root cause. Record and "
                "re-evaluate it as a changed claim, including failure_contribution. This "
                "does not confirm the original claim."
            )
    if "processing_error" in statuses:
        failure = (scores.get("claim_compilation") or {}).get("failure")
        rejected = claim_compilation_rejected(scores)
        # A settled compiler failure that only a reworded claim can reopen would
        # re-raise on every unchanged retry; do not offer that loop.
        if not (failure and rejected):
            actions.append("retry_review")
            result["recommended_action"] = "retry_review"
        if rejected:
            if "revise_claim" not in actions:
                actions.append("revise_claim")
            result["recommended_action"] = "revise_claim"
            result["notes"].append(
                "Claim compilation failed validation. A grounded claim revision is allowed; "
                "it must receive fresh evaluation and review and does not validate the old claim."
            )
        if failure:
            result["notes"].append(failure["recovery_action"])
            result["compilation_block"] = failure
    if (complete_details and (validation["status"] == "confirmed"
                             or scores.get("behavior_review_phase") == "development")
            and final_outcomes and set(final_outcomes) == {"yes"}):
        result["recommended_action"] = "retain"
        result["allowed_actions"] = []
        if scores.get("behavior_review_phase") == "development":
            result["notes"].append(
                "The reviewed development examples support the full claim under the selected "
                "validation standard. The sample size alone does not require a repair. "
                + DEVELOPMENT_REVIEW_GUIDANCE)
    elif not actions:
        actions.append("gather_evidence")
        result["notes"].append(
            "Detailed final assessments are missing; retrieve them before repair."
        )
    if not complete_details:
        if "gather_evidence" not in result["allowed_actions"]:
            result["allowed_actions"].append("gather_evidence")
        result["notes"].append(
            "Stored final review rows do not cover the reported counts. Retrieve missing "
            "details before concluding the sample supports retaining or retiring this detector."
        )
    if integrity_errors:
        result["recommended_action"] = "gather_evidence"
        result["notes"].append(
            "Malformed or inconsistent saved rows are excluded from repair findings. "
            "Recover the original complete judgment or re-judge before acting on those rows."
        )
    if (not integrity_errors and result["recommended_action"] == "gather_evidence"
            and result["investigations"]
            and result["investigations"][0].get("action") == "defer_claim"):
        result["diagnostic_disposition"] = "defer_claim"
        result["notes"].append(
            "The required source is unavailable. The gather_evidence workflow action may "
            "only investigate whether a separate retained source can supply the missing fact; "
            "it does not authorize repeated judgment of unchanged evidence. Otherwise defer "
            "this claim and investigate an evidenced mechanism."
        )

    calibration = parse_claim_calibration(scores)
    if (calibration and not integrity_errors
            and not result.get("review_completion_only")
            and any(outcome != "error" for outcome in final_outcomes)):
        result["calibration_suggestion"] = {
            **calibration,
            "authority": "editorial suggestion; does not override claim assessments",
            "limitations": EDITORIAL_LIMIT,
        }
    return result
