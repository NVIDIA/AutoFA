# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One concrete recovery investigation, checkpointed with existing stall state.

This is advice backed by saved evidence, not an extra investigation allowance or
an alternative judge. Only family conclusions and completed detector attempts
advance it; sampling, paging, status writes and round changes do not.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.metrics.review_diagnosis import build_review_diagnosis
from autofa.metrics.verdicts import current_behavior_scores
from autofa.objectives import discovery_strategy
from autofa.session.family_coverage import cached_task_inventory
from autofa.session.synthesis_decisions import pending_decision


def _key(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _read(rt: Any, name: str) -> dict:
    try:
        value = json.loads((rt.run.run_dir() / name).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _comparisons(families: dict) -> list[dict]:
    return [comparison for batch in [*families.get("history", []), families]
            for comparison in batch.get("comparisons", [])]


def _comparison_key(comparison: dict) -> str:
    # A repeated round or annotation is not another source investigation.
    survey, result = comparison.get("survey") or {}, comparison.get("assessment") or {}
    return _key([survey.get("family_name"), survey.get("detector_name"),
                 survey.get("evidence"), result])


def _assessed_comparison(rt: Any, assignment: dict, comparison: dict) -> bool:
    """Recognize current saved assessments."""
    from autofa.metrics.claim_contract import claim_hash

    survey, result = comparison.get("survey") or {}, comparison.get("assessment") or {}
    name = survey.get("detector_name") or survey.get("family_name")
    coverage = result.get("coverage") or {}
    fn = rt.library.get(name)
    if (result.get("outcome") != "assessed"
            or survey.get("outcome") != "record_family"
            or result.get("family_name") != survey.get("family_name")
            or result.get("detector_name") != name or fn is None
            or not result.get("notes") or not isinstance(result.get("scores"), dict)
            or not isinstance(result.get("review"), dict)):
        return False
    fingerprint = claim_hash(source=review_source(fn), description=fn.description or "",
                             mechanism=fn.mechanism or fn.description or fn.name)
    scores = fn.scores or {}
    return (
        result.get("claim_hash") == coverage.get("claim_hash") == fingerprint
        and coverage.get("evaluation_run_id") == rt.run.run_id
        and bool(assignment.get("baseline_corpus"))
        and coverage.get("corpus_fingerprint") == assignment["baseline_corpus"]
        and scores.get("evaluated") is True and not scores.get("evaluation_stale")
        and scores.get("evaluation_claim_hash") == fingerprint
        and scores.get("evaluation_run_id") == rt.run.run_id
    )


def _completion(rt: Any, assignment: dict, state: dict, comparisons: list[dict]) -> dict | None:
    name = assignment.get("detector_name")
    for record in state["records"].values():
        if (name and record["name"] == name
                and record["attempt"] not in assignment["baseline_attempts"]):
            return {"outcome": "completed_detector_investigation", "detector_name": name,
                    "grade": record["grade"], "attempt": record["attempt"]}
    for comparison in comparisons:
        if _comparison_key(comparison) in assignment["baseline_comparisons"]:
            continue
        survey, result = comparison.get("survey") or {}, comparison.get("assessment") or {}
        rows = survey.get("evidence") or []
        linked = (bool(assignment.get("family_name"))
                  and survey.get("family_name") == assignment["family_name"])
        linked |= bool(set(assignment.get("item_ids", []))
                       & {row.get("item_id") for row in rows})
        if not linked:
            continue
        outcome = result.get("outcome")
        if outcome == "assessed":
            # assess_family already checked the evidence and current evaluation.
            # Do not make recovery re-record that family or spend another attempt.
            if not _assessed_comparison(rt, assignment, comparison):
                continue
        elif outcome == "deferred":
            if not result.get("blocker") or not result.get("next_step"):
                continue
        elif outcome == "no_coherent_family":
            # Prompt-only surveys cannot settle whether a delivered answer is
            # correct. An explicit unavailable-evidence finding uses defer_family.
            relevant = [row for row in rows
                        if row.get("item_id") in assignment.get("item_ids", [])]
            if not relevant or not all(any(
                quote.get("field") in {"response_text", "final_message", "trajectory_text"}
                for quote in row.get("quotes", [])
            ) for row in relevant):
                continue
        else:
            continue
        return {"outcome": outcome, "family_name": survey.get("family_name"),
                "notes": str(result.get("notes") or result.get("blocker") or "")[:900],
                "next_step": str(result.get("next_step") or "")[:600],
                **({"decision": result["decision"]} if result.get("decision") else {})}
    return None


def _review_assignment(rt: Any, *, excluded: set[str], preferred: str | None) -> dict | None:
    functions = sorted(rt.library.functions, key=lambda fn: fn.name != preferred)
    if not preferred:
        functions.reverse()
    for fn in functions:
        if fn.name in excluded or (preferred and fn.name != preferred):
            continue
        scores = current_behavior_scores(
            fn.scores, source=review_source(fn), description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
            objective_strategy=discovery_strategy(rt.config),
            behavior_strictness=strictness_for_config(rt.config),
        )
        if scores.get("behavior_judgment_stale") or scores.get("behavior_evidence_stale"):
            continue
        for row in scores.get("behavior_verdicts") or []:
            if row.get("verdict") not in {"no", "uncertain"}:
                continue
            decisive = "contradicted" if row["verdict"] == "no" else "unknown"
            findings = [part for part in row.get("claim_assessments") or []
                        if part.get("status") in {"unknown", "contradicted"}]
            issue = next(iter(sorted(findings, key=lambda part: part["status"] != decisive)), {})
            diagnosis = build_review_diagnosis(
                row, issue, requirement=issue.get("requirement_id") or "the reviewed claim",
                name=fn.name, max_chars=600,
            )
            return {"kind": "resolve_review_finding", "detector_name": fn.name,
                    "item_ids": [row["item_id"]] if row.get("item_id") else [],
                    "assessment_status": diagnosis["assessment_status"],
                    "diagnosis_kind": diagnosis["diagnosis_kind"],
                    "distinction": diagnosis["unsupported_premise"],
                    "observed": diagnosis["observed_facts"],
                    "competing_explanation": diagnosis["competing_explanation"],
                    "evidence_needed": diagnosis["evidence_needed"],
                    "evidence_availability": diagnosis["evidence_availability"],
                    "next_action": diagnosis["next_action"],
                    "details_tool": diagnosis["details_tool"]}
    return None


def _recovery_tasks(rt: Any, ledger: dict, history: list[dict]) -> tuple[list[str], str]:
    """Choose distinct failed tasks, rotating away from completed recovery evidence.

    Exposure is not a negative finding. It is only a sampling preference, and
    unknown task identities never establish that another rollout is a new task.
    """
    try:
        inventory = cached_task_inventory(rt)
    except (AttributeError, OSError, ValueError):
        inventory = {}
    failed = inventory.get("task_by_failed_item_id") or {}
    task_by_item = inventory.get("task_by_item_id") or failed
    identities = inventory.get("identity_by_item_id") or {}
    seen = set(ledger.get("discovery_item_ids", []))
    seen.update(ledger.get("discovery_task_groups", {}))
    tried = {item for entry in history if entry["result"].get("outcome") != "superseded"
             for item in entry.get("item_ids", [])}
    seen.update(tried)
    seen_tasks = {task_by_item[item] for item in seen if task_by_item.get(item)}
    tried_tasks = {task_by_item[item] for item in tried if task_by_item.get(item)}
    for group in ledger.get("discovery_task_groups", {}).values():
        try:
            benchmark, task = json.loads(group)
        except (TypeError, ValueError):
            continue
        seen_tasks.add(f"{benchmark or 'unknown'}:{task}")

    def benchmark(item: str) -> str:
        # Use the verified identity projection, not parsing an opaque item ID.
        return str(identities.get(item, {}).get("benchmark") or "")

    visited_benchmarks: dict[str, int] = {}
    for item in tried:
        group = benchmark(item)
        if group:
            visited_benchmarks[group] = visited_benchmarks.get(group, 0) + 1
    items, tasks = [], set()
    while len(items) < 2:
        candidates = [item for item, task in failed.items()
                      if item not in items and (task is None or task not in tasks)]
        if not candidates:
            break
        item = min(candidates, key=lambda candidate: (
            candidate in tried or bool(failed[candidate] in tried_tasks),
            candidate in seen or bool(failed[candidate] in seen_tasks),
            visited_benchmarks.get(benchmark(candidate), 0), candidate,
        ))
        items.append(item)
        if failed[item] is not None:
            tasks.add(failed[item])
        group = benchmark(item)
        if group:
            visited_benchmarks[group] = visited_benchmarks.get(group, 0) + 1
    selection = (
        "not_previously_exposed_tasks"
        if items and ledger.get("history_complete") is True
        and all(failed[item] is not None and item not in seen
                and failed[item] not in seen_tasks for item in items)
        else "revisit_or_exposure_unknown"
    )
    return items, selection


def _new_assignment(rt: Any, families: dict, history: list[dict]) -> dict:
    previous = history[-1] if history else None
    excluded = {entry.get("detector_name") for entry in history
                if entry["result"].get("outcome") != "superseded"}
    survey = families.get("survey") or {}
    pending = families.get("phase") == "assess_family"
    if not pending and (unresolved := pending_decision(families)):
        return {
            "kind": "operationalize_shared_relation", "decision_id": unresolved["id"],
            "item_ids": unresolved["item_ids"], "distinction": unresolved["shared_relation"],
            "diagnosis_kind": unresolved["outcome"],
            "next_action": {"tool": "family_discovery", "arguments": {"action": "status"},
                            "instruction": unresolved["instruction"]},
            "source": unresolved["reference"],
        }
    preferred = (survey.get("detector_name") or survey.get("family_name")) if pending else None
    assignment = _review_assignment(rt, excluded=excluded, preferred=preferred)
    blocked = None
    if (assignment and assignment["evidence_availability"] == "unavailable"
            and assignment["diagnosis_kind"] == "evidence_gap"):
        if pending:
            assignment["next_action"] = {
                "tool": "family_discovery",
                "arguments": {"action": "defer_family", "family_name": survey["family_name"],
                              "blocker": assignment["evidence_needed"],
                              "next_step": "Locate a retained source establishing: "
                                           + assignment["distinction"]},
                "instruction": "Preserve the unsupported premise as an open lead; "
                               "do not treat unavailable evidence as a contradiction.",
            }
        else:
            blocked, assignment = assignment, None
    if assignment is not None:
        if assignment["detector_name"] == preferred:
            assignment["family_name"] = survey.get("family_name")
    else:
        items = [row["item_id"] for row in survey.get("evidence") or []
                 if row.get("item_id")] if pending else []
        selection = "pending_family_evidence"
        if not items:
            ledger = _read(rt, "review_evidence.json")
            items, selection = _recovery_tasks(rt, ledger, history)
        assignment = {
            "kind": "check_mechanism_in_final_evidence", "item_ids": items[:2],
            "task_selection": selection,
            "distinction": str(survey.get("mechanism") or survey.get("description") or "")
                if pending else (
                    "Identify an operation or interpretation that changes the required result; "
                    "distinguish an actual defect from a complete-looking but incorrect answer, "
                    "a recovered incident, or a passing case with the same surface behavior."
                ),
            "next_action": {
                "tool": "get_trace_record" if items else "family_discovery",
                "arguments": ({"item_id": items[0], "section": "coverage", "max_chars": 4000}
                              if items else {"action": "status"}),
                "instruction": (
                    "Locate the latest relevant user request and the final assistant response "
                    "by their roles and event positions; do not assume a fixed step index. "
                    "Retrieve the relevant full text and any necessary artifact, follow paging "
                    "to the decisive result, and check the proposed distinction."
                ),
            },
        }
        if pending:
            assignment["family_name"] = survey.get("family_name")
            assignment["detector_name"] = preferred
    if blocked:
        assignment["unavailable_lead"] = {
            "detector_name": blocked["detector_name"], "distinction": blocked["distinction"],
            "evidence_needed": blocked["evidence_needed"], "details_tool": blocked["details_tool"],
            "instruction": "Keep this lead unresolved unless its missing premise becomes "
                           "available. Investigate a different observable mechanism now.",
        }
    if previous:
        assignment["previous_result"] = previous["result"]
        assignment["previous_investigation"] = {
            key: previous.get(key) for key in
            ("id", "round", "kind", "family_name", "detector_name", "item_ids", "distinction")
        }
        assignment["direction_guidance"] = (
            "Cite the previous investigation and its saved outcome. After an unsuccessful "
            "pass, investigate different task evidence or a different supported mechanism; "
            "do not repeat the same rejected claim on another rollout of the same task. "
            "If revisiting those cases, identify the new evidence or testable distinction "
            "that addresses the recorded blocker. Finish the existing bounded refinement "
            "and validation pass; declaring a new lead or retrieving examples is not "
            "completed recovery work."
        )
    return assignment


def update_recovery_assignment(rt: Any, state: dict, *, active: bool) -> dict | None:
    """Mutate only the caller's already-locked stall checkpoint, without new I/O writes."""
    if not active:
        state.pop("recovery_assignment", None)
        # A successful phase does not erase earlier negative investigations.
        # Keep their lessons for the next recovery phase and for round feedback.
        return None
    families = _read(rt, "family_discovery.json")
    if (families.get("run_id") != rt.run.run_id
            or families.get("behavior_strictness", "causal") != strictness_for_config(rt.config)):
        families = {}
    comparisons = _comparisons(families)
    from autofa.metrics.claude_judge import judge_policy_hash

    policy = judge_policy_hash(behavior_strictness=strictness_for_config(rt.config))
    try:
        corpus = cached_task_inventory(rt).get("corpus_fingerprint")
    except (AttributeError, OSError, ValueError):
        corpus = None

    def detector_snapshot(name: str | None) -> str | None:
        fn = next((fn for fn in rt.library.functions if fn.name == name), None)
        # A source-only implementation repair still needs its original
        # counterexample. A changed defining claim needs a new investigation.
        return (_key([fn.description, fn.mechanism]) if fn is not None else None)

    assignment = state.get("recovery_assignment")
    history = state.setdefault("recovery_history", [])
    if assignment:
        if (unresolved := pending_decision(families)) and families.get("phase") != "assess_family":
            # Refresh advice only. Repeated unsupported dismissals are not completed
            # recovery work and do not renew the assignment's baseline or allowance.
            assignment["pending_synthesis_decision"] = unresolved
            assignment["next_action"] = {
                "tool": "family_discovery", "arguments": {"action": "status"},
                "instruction": unresolved["instruction"],
            }
        else:
            assignment.pop("pending_synthesis_decision", None)
        changed = (
            assignment.get("baseline_review_policy") != policy
            or (corpus is not None and assignment.get("baseline_corpus") != corpus)
            or assignment.get("baseline_detector") != detector_snapshot(
                assignment.get("detector_name"))
        )
        result = _completion(rt, assignment, state, [] if changed else comparisons)
        if result is None and changed:
            result = {"outcome": "superseded", "notes":
                      "The detector, review policy or corpus changed; use current evidence. "
                      "This is not a completed investigation or quality improvement."}
        if result is None:
            return {key: value for key, value in assignment.items()
                    if not key.startswith("baseline_")}
        history.append({**{key: assignment.get(key) for key in (
            "id", "round", "kind", "family_name", "detector_name", "distinction", "item_ids",
        )}, "result": result, "corpus_fingerprint": assignment.get("baseline_corpus"),
            "review_policy": assignment.get("baseline_review_policy")})
    relevant_history = [entry for entry in history
                        if entry["result"].get("outcome") == "superseded"
                        or (entry.get("corpus_fingerprint") in {None, corpus}
                            and entry.get("review_policy") in {None, policy})]
    assignment = _new_assignment(rt, families, relevant_history)
    assignment["round"] = rt.run.round
    assignment.update(baseline_review_policy=policy, baseline_corpus=corpus,
                      baseline_detector=detector_snapshot(assignment.get("detector_name")),
                      baseline_attempts=list(state["processed"]),
                      baseline_comparisons=[_comparison_key(row) for row in comparisons])
    assignment["id"] = _key([state.get("quality_generation", 0), len(history), assignment])[:16]
    state["recovery_assignment"] = assignment
    return {key: value for key, value in assignment.items() if not key.startswith("baseline_")}


def recovery_round_context(rt: Any) -> dict | None:
    """Read the saved recovery lessons without creating assignments or counting work."""
    state = _read(rt, "discovery_stall.json")
    history = state.get("recovery_history") or []
    assignment = state.get("recovery_assignment")
    unresolved = pending_decision(_read(rt, "family_discovery.json"))
    if not history and not assignment and not unresolved:
        return None
    lessons = []
    for index, entry in reversed(list(enumerate(history))):
        if (entry.get("result") or {}).get("outcome") == "superseded":
            continue
        lessons.append({
            **{key: entry.get(key) for key in (
                "round", "kind", "family_name", "detector_name", "item_ids", "distinction",
            )},
            "result": entry.get("result"), "historical": True,
            "reference": f"discovery_stall.json#/recovery_history/{index}",
        })
        if len(lessons) == 2:
            break
    return {
        "pending_synthesis_decision": unresolved,
        "completed_investigations": lessons,
        "pending": ({key: assignment.get(key) for key in (
            "id", "round", "kind", "family_name", "detector_name", "item_ids", "distinction",
        )} if assignment else None),
        "guidance": (
            "Use these saved outcomes and the family/mode ledger to choose this round's "
            "exploration direction. After a negative pass, cite what was ruled out and "
            "check different failed tasks, an underexplored dataset segment, or a different "
            "observable mechanism. Supported subtypes and recorded synthesis leads are "
            "also useful; do not invent a family solely to change direction. Revisit a "
            "blocked claim only with new evidence or a testable distinction addressing "
            "its blocker. Complete investigation, round validation and the bounded "
            "refinement exchange; a new lead, another rollout, or more retrieval alone "
            "does not establish recovery or library improvement."
        ),
    }
