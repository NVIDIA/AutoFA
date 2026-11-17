# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read historical repair lessons without transferring verdicts between claims."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from difflib import unified_diff
from typing import Any

from autofa.core.dependencies import digest, review_source
from autofa.core.localisation import text_hash
from autofa.metrics.claim_contract import claim_hash, derive_claim_verdict
from autofa.session import repair_workflow
from autofa.session.repair_versions import collect_versions

MAX_LESSONS = 8
_MAX_FINDINGS_PER_CATEGORY = 2
_MAX_TEXT = 400
_SCORE_KEYS = (
    "a", "b", "c", "d", "recall", "ppv_at_pi0", "behavior_purity", "task_recall",
    "n_items", "n_unresolved", "errors", "evaluation_run_id", "evaluation_scope_complete",
)
_CATEGORIES = {
    "contradicted": "counterexample", "unknown": "missing_fact",
    "execution_error": "evaluation_error", "processing_error": "review_processing_error",
}


def _pointer(*parts: Any) -> str:
    return "#/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _snapshots(data: dict, name: str):
    """Yield full saved records, including earlier observations of the same code."""
    for parent, state in sorted((data.get("repairs") or {}).items()):
        prefix = ("repairs", parent)
        if isinstance(state.get("baseline"), dict):
            yield state["baseline"], _pointer(*prefix, "baseline"), None
        for index, plan in enumerate(state.get("history") or []):
            path = (*prefix, "history", index)
            for key in ("before", "after"):
                if isinstance(plan.get(key), dict):
                    yield plan[key], _pointer(*path, key), plan
            for child, candidate in sorted((plan.get("candidates") or {}).items()):
                for key in ("after", "after_validation"):
                    if isinstance(candidate.get(key), dict):
                        yield candidate[key], _pointer(*path, "candidates", child, key), plan
    for fingerprint, snapshot in sorted(((data.get("versions") or {}).get(name) or {}).items()):
        if isinstance(snapshot, dict):
            if repair_workflow._version(snapshot) != fingerprint:
                raise ValueError("Saved detector version does not match its snapshot")
            yield snapshot, _pointer("versions", name, fingerprint), None


def _scope(function: dict) -> dict:
    source = review_source(function)
    return {
        "version_hash": repair_workflow._version(function),
        "claim_hash": claim_hash(**repair_workflow._claim(function)),
        "source_sha256": text_hash(source),
        "description": function.get("description") or "",
        "mechanism": function.get("mechanism") or "",
    }


def _record(scope: dict, finding: dict, example: dict, *, reference: str,
            policy: Any = None, evidence_scope: str = "development_review") -> dict:
    return {
        **scope, "category": _CATEGORIES[finding["status"]], "status": finding["status"],
        "finding_id": finding.get("id"),
        "item_id": example["item_id"], "requirement_id": finding.get("requirement_id"),
        "requirement": finding.get("requirement"), "claim_quote": finding.get("claim_quote"),
        "source_field": finding.get("source_field"),
        "unknown_reason": finding.get("unknown_reason"),
        "rationale": example.get("rationale") or example.get("error") or "",
        "citations": deepcopy(example.get("citations") or []),
        "counterevidence": deepcopy(example.get("counterevidence") or []),
        "judge_policy_hash": policy, "snapshot_id": example.get("snapshot_id"),
        "evidence_scope": evidence_scope, "historical": True, "history_refs": [reference],
    }


def _snapshot_findings(function: dict, reference: str):
    scope = _scope(function)
    scores = function.get("scores") or {}
    if (scores.get("evaluated") is True
            and scores.get("evaluation_claim_hash") == scope["claim_hash"]):
        for index, example in enumerate(scores.get("error_examples") or []):
            if (isinstance(example, dict) and example.get("item_id")
                    and isinstance(example.get("error"), str) and example["error"]):
                yield _record(
                    scope, {"status": "execution_error", "requirement_id": "detector_execution"},
                    example, reference=f"{reference}/scores/error_examples/{index}",
                    evidence_scope="detector_execution",
                )
    containers = [(scores, "behavior_verdicts", f"{reference}/scores", "development_review")]
    diagnostic = scores.get("diagnostic_review")
    if isinstance(diagnostic, dict):
        containers.append((diagnostic, "verdicts", f"{reference}/scores/diagnostic_review",
                           "diagnostic_review"))
    for container, rows_key, path, evidence_scope in containers:
        if container.get("claim_hash") != scope["claim_hash"]:
            continue
        rows = container.get(rows_key) or []
        if not isinstance(rows, list):
            continue
        counts = Counter(row.get("item_id") for row in rows
                         if isinstance(row, dict) and isinstance(row.get("item_id"), str))
        contract = container.get("claim_contract") or {}
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or not isinstance(row.get("item_id"), str)
                    or not row["item_id"] or counts[row["item_id"]] != 1):
                continue
            row_ref = f"{path}/{rows_key}/{index}"
            if row.get("verdict") in {"error", "processing_error"}:
                yield _record(
                    scope, {"status": "processing_error", "requirement_id": "review_processing"},
                    {**row, "rationale": row.get("validation_error") or row.get("rationale")},
                    reference=row_ref, policy=container.get("judge_policy_hash"),
                    evidence_scope=evidence_scope,
                )
                continue
            contract_claim = (contract.get("claim_hash", contract.get("contract_id"))
                              if isinstance(contract, dict) else None)
            if (not isinstance(contract, dict) or contract_claim != scope["claim_hash"]
                    or contract.get("source_sha256") != scope["source_sha256"]
                    or row.get("processing_status") not in {None, "completed"}):
                continue
            try:
                truth = derive_claim_verdict(contract, row)
                if row.get("verdict") != ("yes" if truth is True else
                                          "no" if truth is False else "uncertain"):
                    continue
            except (ValueError, TypeError, KeyError):
                continue
            requirements = {r["id"]: r for r in contract["requirements"]}
            for assessment in row["claim_assessments"]:
                requirement = requirements[assessment["requirement_id"]]
                parts = [assessment, *(assessment.get("components") or [])]
                for part in parts:
                    if part.get("status") not in {"contradicted", "unknown"}:
                        continue
                    finding = {
                        **part, "requirement_id": assessment["requirement_id"],
                        "requirement": part.get("claim") or requirement.get("text"),
                        "claim_quote": requirement.get("quote"),
                        "source_field": requirement.get("source_field"),
                    }
                    yield _record(
                        scope, finding, {**row, **part}, reference=row_ref,
                        policy=container.get("judge_policy_hash"), evidence_scope=evidence_scope,
                    )


def saved_version_findings(data: dict, name: str) -> list[dict]:
    """Return all source-scoped negative/gap/error evidence, without truncation.

    These are historical observations, not findings eligible for a current repair
    plan. Consumers must check the selected code, literal claim and evidence
    freshness before applying an observation to another version.
    """
    records: dict[str, dict] = {}

    def add(record):
        identity = digest({key: value for key, value in record.items() if key != "history_refs"})
        if identity in records:
            records[identity]["history_refs"] = sorted(set(
                records[identity]["history_refs"] + record["history_refs"]))
        else:
            records[identity] = record

    for function, reference, plan in _snapshots(data, name):
        if function.get("name") != name:
            continue
        for record in _snapshot_findings(function, reference):
            add(record)
        # The planned-before feedback preserves findings from older judge policies,
        # and earlier observations of a version whose latest scores changed.
        if not plan or not reference.endswith("/before"):
            continue
        scope = _scope(function)
        feedback = plan.get("feedback") or {}
        if feedback.get("claim_hash") != scope["claim_hash"]:
            continue
        for index, finding in enumerate(feedback.get("findings") or []):
            if not isinstance(finding, dict) or finding.get("status") not in _CATEGORIES:
                continue
            examples = [example for example in finding.get("examples") or []
                        if isinstance(example, dict) and isinstance(example.get("item_id"), str)]
            example_ids = {example["item_id"] for example in examples}
            # Feedback bounds its excerpts but keeps all affected item IDs.
            # Preserve those IDs for a targeted check without inventing a quote.
            examples.extend({"item_id": item_id, "rationale": ""}
                            for item_id in finding.get("item_ids") or []
                            if isinstance(item_id, str) and item_id not in example_ids)
            for example in examples:
                if isinstance(example, dict) and example.get("item_id"):
                    add(_record(
                        scope, finding, example,
                        reference=f"{reference.rsplit('/', 1)[0]}/feedback/findings/{index}",
                        policy=plan.get("judge_policy_hash"),
                        evidence_scope=finding.get("evidence_scope", "recorded_repair_feedback"),
                    ))
    return sorted(records.values(), key=lambda row: (
        row["version_hash"], row["category"], row["item_id"], row.get("requirement_id") or "",
        row["rationale"], digest(row),
    ))


def _excerpt(value: Any) -> str:
    text = str(value or "")
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "…"


def _compact_finding(record: dict) -> dict:
    result = {key: deepcopy(record.get(key)) for key in (
        "category", "status", "finding_id", "item_id", "requirement_id", "unknown_reason",
        "judge_policy_hash", "snapshot_id", "evidence_scope",
    )}
    result.update(requirement=_excerpt(record.get("requirement")),
                  rationale=_excerpt(record.get("rationale")),
                  history_refs=record["history_refs"][:3])
    for key in ("citations", "counterevidence"):
        refs = record.get(key) or []
        result[key] = [{**{k: ref[k] for k in ("citation_id", "field", "start", "end") if k in ref},
                        "quote": _excerpt(ref.get("quote"))}
                       for ref in refs[:2] if isinstance(ref, dict)]
    return result


def _compact_comparison(comparison: dict) -> dict:
    """Metric directions suffice here; listing versions carries coverage examples."""
    return {key: deepcopy(comparison.get(key)) for key in (
        "prior_version_hash", "current_version_hash", "comparable", "reason",
        "same_association_cohort", "provisional", "metric_changes", "claim_grade",
        "judge_comparison",
    )}


def _edits(before: dict, after: dict) -> dict:
    lines = list(unified_diff(
        str(before.get("source") or "").splitlines(),
        str(after.get("source") or "").splitlines(), lineterm="", n=1,
    ))
    return {
        "source_diff_excerpt": _excerpt("\n".join(lines[:12])),
        "source_diff_total_lines": len(lines),
        "claim_changes": {
            key: {"before": _excerpt(before.get(key)), "after": _excerpt(after.get(key))}
            for key in ("description", "mechanism") if before.get(key) != after.get(key)
        },
        "frozen_dependencies_changed": any(
            (before.get("meta") or {}).get(key) != (after.get("meta") or {}).get(key)
            for key in ("deps", "dependency_bundle")
        ),
    }


def version_lessons(rt, name: str, current_function: dict | None = None, *,
                    limit: int = MAX_LESSONS) -> dict:
    """Derive bounded lessons from the existing ledger; never execute or re-judge."""
    if type(limit) is not int or not 1 <= limit <= MAX_LESSONS:
        raise ValueError(f"limit must be an integer from 1 to {MAX_LESSONS}")
    if current_function is None:
        current = rt.library.get(name)
        current_function = current.to_dict() if current else None
    with repair_workflow._ledger(rt.run.run_dir()) as data:
        return _version_lessons(
            data, name, current_function, config=rt.config, run_id=rt.run.run_id,
            history_path=str(rt.run.run_dir() / "detector_repairs.json"), limit=limit,
        )


def _version_lessons(data, name, current, *, config, run_id, history_path, limit):
    from autofa.metrics.version_comparison import compare_function_versions

    versions = collect_versions(data, name, current)
    findings = saved_version_findings(data, name)
    current_hash = repair_workflow._version(current) if current else ""
    by_version: dict[str, dict] = {}
    for function, reference, plan in _snapshots(data, name):
        if function.get("name") != name:
            continue
        fingerprint = repair_workflow._version(function)
        group = by_version.setdefault(fingerprint, {
            "function": function, "history_refs": [], "attempts": [],
        })
        group["history_refs"].append(reference)
        # Only the changed candidate owns an attempt's rationale; before feedback
        # remains a separate lesson with its original version and claim.
        before = (plan or {}).get("before")
        if (plan and isinstance(before, dict)
                and repair_workflow._version(before) != fingerprint):
            attempt = {
                "plan_id": plan.get("plan_id"), "action": plan.get("action"),
                "phase": plan.get("phase"), "rationale": _excerpt(plan.get("rationale")),
                "deferral_reason": _excerpt(plan.get("deferral_reason")),
                "finding_ids": list(plan.get("finding_ids") or [])[:8],
                "evidence_ids": list(plan.get("evidence_ids") or [])[:8],
                "before_version_hash": repair_workflow._version(before),
                "changed_fields": [key for key in (
                    "source", "description", "mechanism", "depends_on", "explains", "meta",
                ) if before.get(key) != function.get(key)],
                "edits": _edits(before, function),
                "comparison": _compact_comparison(compare_function_versions(
                    before, versions.get(fingerprint, function), config=config, run_id=run_id,
                )),
            }
            if not any(row["plan_id"] == attempt["plan_id"] for row in group["attempts"]):
                group["attempts"].append(attempt)
    if current:
        # A failed current candidate need not have been replaced/archived yet.
        # This reference is explicitly a live tool record, not a fabricated ledger path.
        live_ref = f"get_library_function({name!r})"
        findings.extend(_snapshot_findings(current, live_ref))
        by_version.setdefault(current_hash, {
            "function": current, "history_refs": [live_ref], "attempts": [],
        })
    lessons = []
    for fingerprint, group in sorted(by_version.items()):
        function = versions.get(fingerprint, group["function"])
        # A saved version and its current library copy describe the same evidence.
        unique = {}
        for row in findings:
            if row["version_hash"] == fingerprint:
                identity = digest({key: value for key, value in row.items()
                                   if key != "history_refs"})
                unique.setdefault(identity, row)
        rows = list(unique.values())
        comparison = (_compact_comparison(compare_function_versions(
            function, current, config=config, run_id=run_id))
                      if current and fingerprint != current_hash else None)
        selected = []
        counts = Counter(row["category"] for row in rows)
        for category in _CATEGORIES.values():
            selected.extend([row for row in rows if row["category"] == category]
                            [:_MAX_FINDINGS_PER_CATEGORY])
        if not rows and not group["attempts"] and comparison is None:
            continue
        scope = _scope(function)
        lessons.append({
            **scope, "description": _excerpt(scope["description"]),
            "mechanism": _excerpt(scope["mechanism"]), "current": fingerprint == current_hash,
            "applicability": ("same_version_requires_freshness_check"
                              if fingerprint == current_hash
                              else "different_version_requires_applicability_check"),
            "historical_only": True, "affects_current_grade": False,
            "scores": {key: (function.get("scores") or {}).get(key) for key in _SCORE_KEYS},
            "comparison_to_current": comparison, "attempts": group["attempts"][-3:],
            "total_attempts": len(group["attempts"]),
            "findings": [_compact_finding(row) for row in selected],
            "finding_counts": dict(counts), "total_findings": len(rows),
            "omitted_findings": len(rows) - len(selected),
            "evidence_ids": sorted({row["item_id"] for row in selected}),
            "history_refs": sorted(set(group["history_refs"]))[:6],
        })
    # Keep concrete counterexamples/errors ahead of descriptive comparisons, and
    # stable hashes break ties independently of archive dictionary insertion order.
    lessons.sort(key=lambda row: (
        not row["finding_counts"].get("counterexample"),
        not row["finding_counts"].get("evaluation_error"),
        not row["total_findings"], not row["total_attempts"], row["version_hash"],
    ))
    return {
        "name": name, "current_version_hash": current_hash, "lessons": lessons[:limit],
        "total_lessons": len(lessons), "omitted_lessons": max(0, len(lessons) - limit),
        "history_path": history_path, "limit": limit,
        "note": ("Historical lessons preserve failed approaches and evidence. They do not "
                 "validate or reject another version. Check its literal claim, saved item "
                 "content and actual matches before applying a lesson. Full records remain "
                 "at history_path and history_refs."),
    }
