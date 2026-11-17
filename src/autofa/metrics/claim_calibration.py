# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Soft claim-calibration feedback (mechanism vs evidence).

Does not gate keepers — nudges the discover agent to tone claims when the detector
is fine but description/mechanism over- or under-states what the fires show.
"""

from __future__ import annotations

from typing import Any

from autofa.behavior_policy import (
    acceptance_requirement_id,
    strictness_for_scores,
    validate_behavior_strictness,
)
from autofa.metrics.review_diagnosis import (
    DIAGNOSIS_FIELDS,
    EDITORIAL_LIMIT,
    build_review_diagnosis,
    excerpt_fields,
)

# Structured statuses the judge (or a heuristic fallback) may return.
CALIBRATION_STATUSES = frozenset(
    {
        "good",
        "overclaim",
        "underclaim",
        "detector_shaped",
        "mixed",
    }
)

DEVELOPMENT_REVIEW_GUIDANCE = (
    "Retain supported development findings with their sample limitations; they can support "
    "Good when the shared quality criteria are met. A smaller or reserved sample is not a "
    "contradiction. Independent confirmation is optional and useful when fresh evidence of "
    "generalization is needed. If seeking it, address evidenced mismatches, freeze the "
    "candidate, then call judge_function(name, review_phase='confirmation') for reserved "
    "tasks. Repeated development reviews do not establish independent confirmation. "
    "A small sample does not erase a known counterexample or substantive uncertainty; "
    "additional positive reviews cannot repair an already demonstrated mismatch."
)


def normalize_calibration_status(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ok": "good",
        "fine": "good",
        "aligned": "good",
        "too_strong": "overclaim",
        "overclaims": "overclaim",
        "overstated": "overclaim",
        "too_weak": "underclaim",
        "underclaims": "underclaim",
        "understated": "underclaim",
        "vague": "underclaim",
        "trigger": "detector_shaped",
        "fires_when": "detector_shaped",
        "partial": "mixed",
    }
    s = aliases.get(s, s)
    return s if s in CALIBRATION_STATUSES else None


def parse_claim_calibration(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract claim_calibration from judge JSON (top-level or nested)."""
    if not isinstance(data, dict):
        return None
    raw = data.get("claim_calibration")
    if raw is None and any(
        k in data for k in ("calibration_status", "claim_note", "rewrite_hint")
    ):
        raw = data
    if not isinstance(raw, dict):
        return None
    status = normalize_calibration_status(
        raw.get("status") or raw.get("calibration_status")
    )
    if not status:
        return None
    # Saved assessments retain the full advice. Only response summaries abbreviate it,
    # with explicit truncation metadata and a retrieval path.
    note = str(raw.get("claim_note") or raw.get("note") or "").strip()
    detector_note = str(raw.get("detector_note") or "").strip()
    rewrite = str(raw.get("rewrite_hint") or raw.get("hint") or "").strip()
    return {
        "status": status,
        "claim_note": note,
        "detector_note": detector_note,
        "rewrite_hint": rewrite,
    }


def synthesize_calibration_from_verdicts(
    verdicts: list[dict[str, Any]],
    *,
    purity: float,
) -> dict[str, Any] | None:
    """Heuristic fallback when the judge omits claim_calibration."""
    semantic = [
        v for v in verdicts if v.get("verdict") in {"yes", "no", "uncertain"}
        and v.get("processing_status") != "error"
    ]
    if not semantic:
        return None
    n = len(semantic)
    n_yes = sum(1 for v in semantic if v.get("verdict") == "yes")
    n_no = sum(1 for v in semantic if v.get("verdict") == "no")
    n_uncertain = sum(1 for v in semantic if v.get("verdict") == "uncertain")
    contribution_gaps = [
        a for v in semantic for a in v.get("claim_assessments") or []
        if a.get("requirement_id") == "failure_contribution"
        and a.get("status") in {"contradicted", "unknown"}
    ]
    if contribution_gaps:
        return {
            "status": "mixed",
            "claim_note": "Some fires do not establish a contribution to the task failure.",
            "detector_note": "Occurrence of the behavior alone is insufficient.",
            "rewrite_hint": (
                "Inspect the final unmet requirement, later recovery and competing causes. "
                "Exclude recovered unrelated incidents; obtain missing causal evidence for "
                "unresolved cases. Narrowing the prose to a transient behavior cannot waive "
                "the mandatory failure_contribution check."
            ),
        }
    relevance_gaps = [
        a for v in semantic for a in v.get("claim_assessments") or []
        if a.get("requirement_id") == "failure_relevance"
        and a.get("status") in {"contradicted", "unknown"}
    ]
    if relevance_gaps:
        return {
            "status": "mixed",
            "claim_note": "Some fires do not establish the required contextual relevance.",
            "detector_note": "Behavior occurrence or a failed benchmark label alone "
                             "is insufficient.",
            "rewrite_hint": review_policy_guidance("contextual") + " Obtain missing context "
                            "for unresolved cases; revised claims need their own review.",
        }
    if not n_no and (n_uncertain or n < len(verdicts)):
        return {
            "status": "mixed",
            "claim_note": (
                f"{n_yes} claims confirmed; {n_uncertain} unresolved evidence reviews; "
                f"{len(verdicts) - n} processing failures. No reviewed claim was contradicted."
            ),
            "detector_note": (
                "Unresolved evidence and processing failures do not prove a false hit."
            ),
            "rewrite_hint": (
                "Inspect unknown requirement assessments and their evidence gaps; retry "
                "processing failures. Change the claim or detector only when traces justify it."
            ),
        }
    rationales = " ".join(str(v.get("rationale") or "") for v in semantic).lower()
    over_cues = (
        "overclaim" in rationales
        or "too strong" in rationales
        or "stronger than" in rationales
        or "exaggerat" in rationales
        or "not that extreme" in rationales
        or "only partially" in rationales
    )
    under_cues = (
        "underclaim" in rationales
        or "too vague" in rationales
        or "too weak" in rationales
        or "underspecif" in rationales
        or "broader than claim" in rationales
    )
    detector_cues = (
        "fires when" in rationales
        or "keyword" in rationales
        or "superficial" in rationales
        or "regex" in rationales and "not" in rationales
    )

    if over_cues and n_yes >= max(1, n // 2):
        return {
            "status": "overclaim",
            "claim_note": (
                "Several fires match a weaker version of the story; the written "
                "mechanism/description looks stronger than the evidence."
            ),
            "detector_note": "Detector often fires on related behavior; claim may overstate it.",
            "rewrite_hint": (
                "Tone down mechanism/description to what the majority of fires "
                "actually show; keep the detector if metrics are strong."
            ),
        }
    if under_cues and n_yes >= max(1, n // 2):
        return {
            "status": "underclaim",
            "claim_note": "Evidence shows a sharper failure than the written claim names.",
            "detector_note": "Fires look coherent; claim undersells the mechanism.",
            "rewrite_hint": (
                "Sharpen description/mechanism to the concrete behavior in the traces "
                "without inventing details that are not there."
            ),
        }
    if detector_cues or (purity < 0.45 and n_no > 0 and n_no >= n_yes):
        return {
            "status": "detector_shaped" if detector_cues else "mixed",
            "claim_note": (
                "Claim and labeled evidence disagree often — or fires look like "
                "shallow detector hits rather than the stated failure story."
            ),
            "detector_note": "Tighten the predicate and/or rewrite the claim to match fires.",
            "rewrite_hint": (
                "get_item on no-verdict example_fail_ids; align description/mechanism "
                "to those traces or retune the detector."
            ),
        }
    if purity >= 0.7 and n_yes >= max(1, int(0.7 * n)):
        return {
            "status": "good",
            "claim_note": "Claim strength looks calibrated to the labeled evidence.",
            "detector_note": "Fires generally exhibit the stated failure.",
            "rewrite_hint": "",
        }
    if 0.45 <= purity < 0.7:
        return {
            "status": "mixed",
            "claim_note": "Some fires support the claim; others only partially match.",
            "detector_note": "Detector quality is mixed relative to the written story.",
            "rewrite_hint": (
                "Either narrow the claim to the clean fires or tighten the detector "
                "so false story-matches drop out."
            ),
        }
    return {
        "status": "mixed",
        "claim_note": "Claim ↔ evidence alignment is unclear from the sample.",
        "detector_note": "",
        "rewrite_hint": "Spot-check example_fail_ids and rewrite claim or detector.",
    }


def format_calibration_hint(cal: dict[str, Any] | None) -> str | None:
    """One soft line for MCP ``hint`` / message (None if good or empty)."""
    if not cal:
        return None
    status = normalize_calibration_status(cal.get("status"))
    if not status or status == "good":
        return None
    note = str(cal.get("claim_note") or "").strip()
    rewrite = str(cal.get("rewrite_hint") or "").strip()
    labels = {
        "overclaim": "Claim calibration: mechanism/description looks a bit too strong",
        "underclaim": "Claim calibration: mechanism/description undersells the evidence",
        "detector_shaped": "Claim calibration: claim reads detector-shaped vs the traces",
        "mixed": "Claim calibration: review evidence gaps and mismatches",
    }
    head = labels.get(status, f"Claim calibration: {status}")
    parts = [head]
    if note:
        parts.append(note)
    if rewrite:
        parts.append(f"Rewrite: {rewrite}")
    return " — ".join(parts)


def attach_calibration(
    result: dict[str, Any],
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ensure ``result`` has claim_calibration (parsed or synthesized)."""
    verdicts = list(result.get("verdicts") or [])
    if verdicts and not any(
        v.get("verdict") in {"yes", "no", "uncertain"}
        and v.get("processing_status") != "error" for v in verdicts
    ):
        result.update(claim_calibration=None, calibration_hint=None)
        return result
    unfinished = sum(v.get("completion_status") == "needs_evidence_review" for v in verdicts)
    if unfinished and not any(v.get("verdict") == "no" for v in verdicts):
        # Unread evidence is review work, not an observed flaw in the predicate.
        # Do not pass through unsupported editorial narrowing from that same review.
        cal = {
            "status": "mixed",
            "claim_note": f"{unfinished} reviews still need available evidence to be read. "
                          "No reviewed claim was contradicted.",
            "detector_note": "Incomplete review is not evidence of a false detector match.",
            "rewrite_hint": "Inspect saved unread ranges within any remaining original review "
                            "budget. If exhausted, keep the verdict inconclusive and defer or "
                            "gather new evidence. Do not resubmit the unchanged claim to reset "
                            "the budget. Change the detector only for an evidenced mismatch.",
        }
        result.update(claim_calibration=cal, calibration_hint=format_calibration_hint(cal))
        return result
    cal = parse_claim_calibration(data) if data else None
    if cal is None:
        cal = parse_claim_calibration(result)
    if cal is None:
        cal = synthesize_calibration_from_verdicts(
            verdicts,
            purity=float(result.get("behavior_purity") or 0.0),
        )
    result["claim_calibration"] = cal
    result["calibration_hint"] = format_calibration_hint(cal)
    return result


_REVIEW_ISSUE_GUIDANCE = {
    "counterexample": (
        "First inspect contradicted requirements and cited counterevidence with "
        "get_function_feedback/get_function_repair. A small sample does not make a saved "
        "counterexample disappear: do not request more positive reviews to dilute it. "
        "Compare the detector's "
        "parsed values, linked results and decisions with the source: repair implementation "
        "errors without changing the claim. For a genuine counterexample, revise the predicate "
        "or mechanism using the evidenced distinction, preserving supported positives and "
        "checking later recovery. Evaluate and re-judge any changed candidate; a revision "
        "cannot silently waive a condition of its frozen claim. Test the distinction on "
        "the saved counterexamples and supported positives before requesting fresh reviews."
    ),
    "review_incomplete": (
        "Inspect saved unread ranges and use only any remaining original review budget. "
        "If exhausted, retain the inconclusive verdict and defer or gather new evidence. "
        "Do not resubmit the unchanged claim to reset its budget. Unread evidence does "
        "not establish a false positive or an overclaim."
    ),
    "missing_evidence": (
        "Retrieve the source facts identified as unavailable, including authoritative tool or "
        "termination semantics when the claim depends on them. Inspect the full relevant trace "
        "and recovery context. An LLM rewrite cannot supply missing evidence; missing export "
        "fields do not establish that an action never occurred. If evidence cannot be obtained, "
        "retain the unknown and defer that claim, or investigate a narrower supported claim "
        "with fresh evaluation and review under the selected review standard."
    ),
    "ambiguous_evidence": (
        "Inspect the cited competing interpretations and seek evidence that distinguishes "
        "them. An unresolved claim is not a proven false hit. Changing implementation type "
        "does not resolve ambiguity by itself; retain uncertainty when the source cannot "
        "establish the required distinction. More favorable reviews on other items cannot "
        "resolve this item's ambiguity."
    ),
    "unresolved_review": (
        "Inspect unknown requirements and their saved rationale to determine whether evidence "
        "is unavailable, unread or ambiguous. An unresolved claim is not a proven false hit. "
        "Do not infer the cause from low purity or switch implementation type without an "
        "evidenced reason. "
        "A narrower claim still needs its required evidence and fresh evaluation and review."
    ),
    "review_processing_failure": (
        "Inspect processing errors and recover the failed review operation before interpreting "
        "purity. A reviewer timeout or invalid response is not a detector counterexample; "
        "restore the failing service or request and retry only within the original review "
        "budget and frozen cohort. Do not reset an unchanged claim's budget."
    ),
}


def review_issue_type(status: Any, unknown_reason: Any = None) -> str | None:
    """Describe saved assessment states, without classifying free-text rationales."""
    if status == "contradicted":
        return "counterexample"
    if status == "processing_error":
        return "review_processing_failure"
    if status == "unknown":
        return {"review_incomplete": "review_incomplete",
                "evidence_unavailable": "missing_evidence",
                "ambiguous_evidence": "ambiguous_evidence"}.get(
                    unknown_reason if isinstance(unknown_reason, str) else None,
                    "unresolved_review")
    return None


def review_policy_guidance(behavior_strictness: str = "causal") -> str:
    """The acceptance threshold, separate from diagnosis and implementation repair."""
    mode = validate_behavior_strictness(behavior_strictness)
    return {
        "behavioral": (
            "Behavioral review checks the full stated behavior; association with failed tasks "
            "is separate. Do not add a mandatory causal link unless the claim asserts it. "
            "Passing or recovered cases can match a pure behavioral claim."
        ),
        "contextual": (
            "Contextual review needs an observable mistake or harmful shortcut with a "
            "trace-supported connection to difficulty or failure. Inspect later recovery; "
            "exclude fully recovered or unrelated incidents. Complete reconstruction of the "
            "final defect is not required unless the claim asserts it."
        ),
        "causal": (
            "Every positive needs evidence of contribution to the task failure. Narrowing "
            "to occurrence cannot waive the mandatory failure_contribution check."
        ),
    }[mode]


def review_issue_guidance(issue: str, *, behavior_strictness: str = "causal") -> str:
    guidance = _REVIEW_ISSUE_GUIDANCE[issue]
    if issue in {"counterexample", "missing_evidence", "unresolved_review"}:
        guidance += " " + review_policy_guidance(behavior_strictness)
    return guidance


def review_issue_types(scores: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    """Keep mixed review problems separate; never treat historical rows as current advice."""
    if validation.get("status") in {"stale", "legacy", "unjudged", "invalid",
                                     "skipped_pre_filter", "no_fresh_evidence"}:
        return []
    issues = set()
    if validation.get("rejected"):
        issues.add("counterexample")
    if validation.get("processing_errors") or validation.get("status") == "error":
        issues.add("review_processing_failure")
    for row in scores.get("behavior_verdicts") or []:
        if not isinstance(row, dict) or row.get("verdict") not in {"no", "uncertain"}:
            continue
        if row.get("completion_status") == "needs_evidence_review":
            issues.add("review_incomplete")
        for assessment in row.get("claim_assessments") or []:
            if not isinstance(assessment, dict):
                continue
            for part in [assessment, *(assessment.get("components") or [])]:
                if not isinstance(part, dict):
                    continue
                issue = review_issue_type(part.get("status"), part.get("unknown_reason"))
                if issue:
                    issues.add(issue)
    if validation.get("review_incomplete"):
        issues.add("review_incomplete")
    if ((validation.get("uncertain") or validation.get("status") == "uncertain")
            and not issues.intersection({"review_incomplete", "missing_evidence",
                                         "ambiguous_evidence", "unresolved_review"})):
        issues.add("unresolved_review")
    return [issue for issue in _REVIEW_ISSUE_GUIDANCE if issue in issues]


def summarize_judge_feedback(
    scores: dict[str, Any], *, name: str, behavior_strictness: str | None = None,
) -> dict[str, Any]:
    """Bounded iteration feedback; full evidence remains in get_function_feedback."""
    from autofa.metrics.verdicts import behavior_validation

    strictness = (strictness_for_scores(scores) if behavior_strictness is None
                  else validate_behavior_strictness(behavior_strictness))
    validation = behavior_validation(
        scores, include_details=False, behavior_strictness=strictness,
    )
    status = validation["status"]
    summary = {
        key: validation.get(key) for key in (
            "status", "label", "sampled", "confirmed", "rejected", "uncertain",
            "processing_errors", "decisive", "decisive_coverage", "ranking_eligible",
            "sample",
        )
    }
    summary["behavior_strictness"] = strictness
    summary["details_tool"] = {"name": "get_function_feedback", "arguments": {"name": name}}
    if "judge_prefilter" in validation:
        summary["judge_prefilter"] = validation["judge_prefilter"]
        summary["judge_prefilter_stale"] = validation["judge_prefilter_stale"]
    if status == "skipped_pre_filter":
        summary.update(
            reason=validation.get("reason"), next_steps=validation["next_steps"], findings=[],
            judgment_history_only=validation.get("judgment_history_only", False),
        )
        return summary
    if validation.get("review_schedule"):
        summary.update(
            reason=validation["reason"], next_steps=validation["next_steps"], findings=[],
            review_schedule=validation["review_schedule"],
        )
        return _attach_diagnostic_feedback(scores, summary, name=name, strictness=strictness)
    actions = []
    incomplete = sum(row.get("completion_status") == "needs_evidence_review"
                     for row in scores.get("behavior_verdicts") or [])
    summary["review_incomplete"] = incomplete
    summary["review_phase"] = scores.get("behavior_review_phase")
    summary["issue_types"] = review_issue_types(scores, summary)
    if status in {"stale", "legacy", "unjudged"}:
        actions.append(f"Call judge_function for a current full-claim {strictness} "
                       "review before using purity.")
    else:
        actions.extend(review_issue_guidance(issue, behavior_strictness=strictness)
                       for issue in summary["issue_types"])
    if scores.get("behavior_review_phase") == "development" and status not in {
        "stale", "legacy", "unjudged",
    }:
        actions.append(DEVELOPMENT_REVIEW_GUIDANCE)
    elif status in {"limited_evidence", "no_fresh_evidence"}:
        actions.append(
            "Validation has fewer fresh examples than its target. Retain available scores "
            "as limited evidence, outside the validated ranking. Obtain additional fresh "
            "evidence to complete fresh validation."
        )
    if status == "reused_evidence":
        actions.append(
            "The current detector was scored on previously reviewed matches. These scores "
            "are descriptive and do not establish fresh validation or ranking eligibility."
        )
    summary["next_steps"] = actions
    summary["findings"] = []
    # Stale and legacy feedback remains accessible as history, never current advice.
    if status in {"stale", "legacy", "unjudged"}:
        return _attach_diagnostic_feedback(scores, summary, name=name, strictness=strictness)
    contract = scores.get("claim_contract") or {}
    requirements = {r["id"]: r for r in contract.get("requirements", []) if r.get("id")}
    calibrations = {
        row.get("item_id"): row.get("claim_calibration")
        for row in scores.get("item_calibrations") or []
    }
    aggregate_calibration = parse_claim_calibration(scores)
    findings = summary["findings"]
    # Show at least one example of each present outcome before filling extra slots.
    rows = scores.get("behavior_verdicts") or []
    ordered = []
    for outcome in ("no", "uncertain", "error"):
        ordered.extend([r for r in rows if r.get("verdict") == outcome][:1])
    ordered.extend(r for r in rows if r not in ordered and r.get("verdict") != "yes")
    summary["findings_omitted"] = max(0, len(ordered) - 3)
    shared_diagnoses = {}
    for row in ordered[:3]:
        item_id = row.get("item_id")
        finding = excerpt_fields({
            "item_id": item_id, "verdict": row.get("verdict"),
            "completion_status": row.get("completion_status", "complete"),
            "rationale": row.get("rationale"),
        }, ("rationale",), limit=200)
        assessments = []
        for assessment in (row.get("claim_assessments") or []) if row["verdict"] != "error" else []:
            components = [a for a in assessment.get("components") or []
                          if a.get("status") != "supported"]
            requirement = requirements.get(assessment.get("requirement_id"), {})
            for component in components:
                assessments.append(excerpt_fields({
                    "requirement_id": assessment.get("requirement_id"),
                    "requirement": str(component.get("claim") or ""),
                    "status": component.get("status"),
                    "unknown_reason": component.get("unknown_reason"),
                    "issue_type": review_issue_type(
                        component.get("status"), component.get("unknown_reason")),
                    "rationale": str(component.get("rationale") or ""),
                    "diagnosis": build_review_diagnosis(
                        row, {**component, "requirement_id": assessment.get("requirement_id")},
                        requirement=str(component.get("claim") or ""), name=name,
                    ),
                }, ("requirement", "rationale"), limit=400))
            if assessment.get("status") == "supported":
                continue
            assessments.append(excerpt_fields({
                "requirement_id": assessment.get("requirement_id"),
                "requirement": str(requirement.get("text") or ""),
                "status": assessment.get("status"),
                "unknown_reason": assessment.get("unknown_reason"),
                "issue_type": review_issue_type(
                    assessment.get("status"), assessment.get("unknown_reason")),
                "rationale": str(assessment.get("rationale") or ""),
                "diagnosis": build_review_diagnosis(
                    row, assessment, requirement=str(
                        requirement.get("text") or assessment.get("requirement_id") or ""
                    ), name=name,
                ),
            }, ("requirement", "rationale"), limit=400))
        decisive_status = "contradicted" if row["verdict"] == "no" else "unknown"
        acceptance_id = acceptance_requirement_id(strictness)
        assessments.sort(key=lambda a: (
            a["status"] != decisive_status,
            a["requirement_id"] != acceptance_id,
        ))
        # Lead with the finding that determines the verdict; prefer the policy
        # blocker among equal statuses. An unknown policy premise must not hide an
        # already contradicted behavior behind evidence-retrieval advice.
        # List every unresolved clause even when its full text is not expanded.
        # Exact full assessments remain accessible without repeating a large diagnosis
        # and its next action for every clause of the same reviewed item.
        finding["unresolved_requirements"] = [{
            "requirement_id": assessment["requirement_id"],
            "status": assessment["status"],
            "unknown_reason": assessment["unknown_reason"],
        } for assessment in assessments]
        finding["requirements_omitted"] = max(0, len(assessments) - 3)
        finding["requirements"] = [
            {key: value for key, value in assessment.items() if key != "diagnosis"}
            for assessment in assessments[:3]
        ]
        if assessments:
            diagnosis = {
                key: value for key, value in assessments[0]["diagnosis"].items()
                if key != "details_tool"
            }
            key = tuple(diagnosis.get(field) for field in (
                *DIAGNOSIS_FIELDS, "evidence_availability", "assessment_status",
                "diagnosis_kind",
            )) + (diagnosis["next_action"]["kind"],)
            # Share only exact, unabridged diagnoses; similar prefixes are not
            # equivalent evidence gaps. Retain each item's own executable read.
            shared_item = (shared_diagnoses.get(key)
                           if not diagnosis.get("fields_truncated") else None)
            if shared_item is not None:
                action = diagnosis["next_action"]
                finding["diagnosis"] = {
                    "same_as_item_id": shared_item,
                    "evidence_availability": diagnosis["evidence_availability"],
                    "next_action": {
                        key: value for key, value in action.items()
                        if key in {"kind", "tool", "arguments"}
                    },
                }
            else:
                finding["diagnosis"] = diagnosis
                if not diagnosis.get("fields_truncated"):
                    shared_diagnoses[key] = item_id
        finding["details_tool"] = {
            "name": "get_function_feedback", "arguments": {"name": name, "item_id": item_id},
        }
        cal = parse_claim_calibration({"claim_calibration": calibrations.get(item_id)})
        if cal and cal == aggregate_calibration:
            finding["claim_calibration_ref"] = "summary.claim_calibration"
        elif cal:
            finding["claim_calibration"] = {
                **excerpt_fields(cal, ("claim_note", "detector_note", "rewrite_hint"), limit=400),
                "authority": EDITORIAL_LIMIT,
            }
        findings.append(finding)
    cal = aggregate_calibration
    if cal:
        summary["claim_calibration"] = {
            **excerpt_fields(cal, ("claim_note", "detector_note", "rewrite_hint"), limit=400),
            "authority": EDITORIAL_LIMIT,
        }
    if findings and findings[0].get("diagnosis"):
        primary = findings[0]
        action = primary["diagnosis"]["next_action"]
        instruction = action.pop("instruction", None)
        if instruction:
            # The submission hint consumes next_steps. Put the concrete evidence
            # question before general safeguards, without copying its full text.
            summary["next_steps"].insert(0, f"Item {primary['item_id']}: {instruction}")
            action["instruction_ref"] = "summary.next_steps[0]"
    return _attach_diagnostic_feedback(scores, summary, name=name, strictness=strictness)


def _attach_diagnostic_feedback(scores: dict, summary: dict, *, name: str, strictness: str) -> dict:
    from autofa.metrics.diagnostic_review import diagnostic_scores

    saved = scores.get("diagnostic_review")
    explicit = isinstance(saved, dict) and saved.get("review_mode") == "diagnostic"
    supplemental = diagnostic_scores(scores)
    if supplemental is not None:
        diagnostic = summarize_judge_feedback(
            supplemental, name=name, behavior_strictness=strictness)
        diagnostic["excluded_from_adherence"] = True
        diagnostic["excluded_from_confirmed_support"] = True
        summary["diagnostic_feedback"] = diagnostic
        missing_main_review = summary["status"] == "unjudged" or (
            explicit and summary["status"] in {"stale", "legacy"})
        if missing_main_review and diagnostic.get("sampled"):
            scope = (
                "A bounded diagnostic reviewed selected development examples. Inspect "
                "its per-case findings to choose the next repair; "
                if explicit else
                "The scoring panel supplied no matching review cases. Inspect the "
                "separate selected development reviews for the actual claim diagnosis; "
            )
            steps = summary["next_steps"] if summary.get("review_schedule") else []
            summary["next_steps"] = [scope +
                "these do not establish population coverage or the main adherence score. "
                "Repeating this completed cohort does not add evidence.", *steps]
        if diagnostic["findings"]:
            scope = "bounded diagnostic" if explicit else "outcome-diagnostic"
            summary["next_steps"].insert(
                0, f"Inspect the separate {scope} findings before retaining this "
                "detector. Supplemental counterexamples and evidence gaps remain actionable; "
                "these controls do not change the adherence estimate or confirmed support.")
    elif scores.get("diagnostic_evidence_stale"):
        summary["diagnostic_feedback"] = {
            "status": "stale", "excluded_from_adherence": True,
            "excluded_from_confirmed_support": True,
        }
        review = f"judge_function({name!r}, review_mode='diagnostic')" if explicit \
            else "judge_function"
        summary["next_steps"].insert(
            0, "Supplemental evidence changed or is unavailable. Restore the source or call "
            f"{review} to recheck the frozen diagnostic cohort before retention.")
    return summary
