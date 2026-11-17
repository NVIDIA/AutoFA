# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence of recovery work, independent of how often an agent asks to advance."""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

from autofa.core.dependencies import execution_bundle, review_source
from autofa.metrics.claim_contract import claim_hash
from autofa.session.recovery import read_json


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _claim(fn: Any) -> str:
    return claim_hash(source=review_source(fn), description=fn.description or "",
                      mechanism=fn.mechanism or fn.description or fn.name)


def _evidence(scores: dict) -> dict:
    """Ignore timestamps, status chatter and regenerated narrative summaries."""
    result = {key: scores.get(key) for key in (
        "evaluation_claim_hash", "evaluation_context", "sampled", "sample_ids",
        "evaluation_item_ids", "n_scored", "n_flagged", "n_fail_caught", "n_insufficient",
        "recall", "ppv_at_pi0", "behavior_purity", "buckets",
        "evaluation_scope_complete", "task_recall_corpus_fingerprint",
        "evaluation_stale", "errors",
        "behavior_judgment_stale", "behavior_n_errors", "behavior_n_review_incomplete",
        "judge_policy_hash",
    )}
    result["final_review"] = {key: (scores.get("final_review") or {}).get(key)
                              for key in ("status", "claim_hash", "scope", "identity")}
    result["behavior_verdicts"] = sorted(
        ({key: row.get(key) for key in (
            "item_id", "verdict", "claim_contract_id", "processing_status", "completion_status",
        )} for row in scores.get("behavior_verdicts", []) if isinstance(row, dict)),
        key=lambda row: str(row["item_id"]),
    )
    result["buckets"] = sorted(({
        "bucket": row.get("bucket"),
        **{key: sorted(row.get(key) or [])
           for key in ("example_fail_ids", "example_pass_ids")},
    } for row in scores.get("buckets") or [] if isinstance(row, dict)),
        key=lambda row: str(row["bucket"]))
    for key in ("sample_ids", "evaluation_item_ids"):
        if isinstance(result[key], list):
            result[key] = sorted(result[key])
    return result


def _rows(rows: list) -> list:
    return sorted(rows, key=lambda row: str(row.get("item_id", "")))


def _smoke_measurements(rows: list) -> list:
    """Author expectations and display annotations are not execution evidence."""
    return _rows([{key: row[key] for key in ("item_id", "status", "value", "reason", "error")
                   if key in row} for row in rows])


def _source(source: str) -> str:
    try:
        return ast.dump(ast.parse(source), include_attributes=False)
    except (SyntaxError, TypeError):
        return str(source)


def _smoke_bindings(args: dict, result: dict, current: dict) -> dict:
    """Bind execution evidence to its tested revision, never just its item IDs."""
    from autofa.session.repair_workflow import _version

    name = result.get("name") or args.get("name")
    if name:
        fn = current.get(name)
        if (fn is None or args.get("name") != name or result.get("name") != name
                or result.get("version_hash") != _version(fn.to_dict())
                or result.get("claim_hash") != _claim(fn)):
            return {}
        return {name: {"version_hash": result["version_hash"],
                       "claim_hash": result["claim_hash"]}}
    source = args.get("source")
    if not source or args.get("depends_on") or args.get("uses_helpers"):
        return {}
    # Anonymous diagnostics do not freeze dependency snapshots or a claim. Only
    # standalone current code can be identified from their durable request.
    return {name: {"source_hash": _hash(_source(source))}
            for name, fn in current.items()
            if not execution_bundle(fn) and not fn.depends_on
            and not (fn.meta or {}).get("deps") and _source(fn.source) == _source(source)}


def recovery_work(rt: Any, names: list[str], controls: dict | None = None) -> dict:
    """Snapshot completed measurements and evidence-backed limitation decisions.

    Hash content rather than operation IDs: a retry, renamed file, repeated scan,
    or paraphrased deferral cannot manufacture another recovery investigation.
    """
    directory = rt.run.run_dir()
    work: dict[str, str] = {}
    tests: dict[str, dict] = {}
    current = {fn.name: fn for fn in rt.library.functions if fn.name in names}
    related_ids = {str(item) for entry in (controls or {}).values()
                   for item in (entry.get("item_ids") or [])}
    related_ids.update(str(item) for fn in current.values()
                       for item in (fn.scores or {}).get("evaluation_item_ids", []))
    related_ids.update(str(row.get("item_id")) for fn in current.values()
                       for row in (fn.scores or {}).get("behavior_verdicts", []))
    for name, fn in current.items():
        scores = fn.scores or {}
        if (scores.get("evaluation_claim_hash") == _claim(fn)
                and scores.get("evaluated") and not scores.get("evaluation_stale")
                and not scores.get("errors") and scores.get("n_scored", 0) > 0):
            key = _hash([name, _evidence(scores)])
            work[key] = f"Current evaluation/review: {name}"
            tests[key] = {"name": name, "item_ids": []}
    for path in (directory / "polishing_evaluations").glob("*.json"):
        result = read_json(path)
        if (result.get("name") in current and result.get("run_id") == rt.run.run_id
                and result.get("complete") and result.get("n", 0) > 0
                and not result.get("errors")):
            key = _hash(["checks", result.get("name"), result.get("claim_hash"),
                         _rows(result.get("results") or [])])
            work[key] = f"Completed regression checks: {result['name']}"
            if result.get("claim_hash") == _claim(current[result["name"]]):
                tests[key] = {"name": result["name"], "item_ids": []}
    for path in (directory / "operations").glob("*.json"):
        operation = read_json(path)
        result = operation.get("result") or {}
        if (operation.get("tool") != "smoke_test_feature"
                or operation.get("state") != "complete" or not isinstance(result, dict)
                or not result.get("ok") or result.get("errors")
                or not result.get("results")
                or not any(str(row.get("item_id")) in related_ids
                           for row in result["results"])):
            continue
        args = operation.get("args") or {}
        # Smoke tests may explore a possible refinement without submitting it.
        # Their outcomes are still useful negative evidence; no model call is required.
        source = (_source(args["source"]) if args.get("source") else
                  result.get("source_sha256") or result.get("source_hash"))
        identity = (["named", args.get("name") or result.get("name"),
                     result.get("version_hash"), result.get("claim_hash")]
                    if args.get("name") or result.get("name") else
                    ["source", source, sorted(args.get("depends_on") or []),
                     args.get("uses_helpers") or {}])
        key = _hash(["diagnostic_measurement_v2", identity,
                     _smoke_measurements(result["results"])])
        work[key] = "Completed smoke diagnostic on retained evidence"
        bindings = {**tests.get(key, {}).get("bindings", {}),
                    **_smoke_bindings(args, result, current)}
        tests[key] = {"name": None, "bindings": bindings,
                      "item_ids": [str(row.get("item_id")) for row in result["results"]]}
    deferred: dict[str, dict] = {}
    ledger = read_json(directory / "detector_repairs.json")
    repairs = ledger.get("repairs") or {}
    for name, fn in current.items():
        plans = (repairs.get(name) or {}).get("history") or []
        if not plans:
            continue
        plan = plans[-1]
        if (plan.get("phase") != "deferred" or not plan.get("deferral_reason")
                or not plan.get("finding_ids") or not plan.get("evidence_ids")
                or plan.get("before_claim_hash") != _claim(fn)
                or _evidence((plan.get("before") or {}).get("scores") or {})
                != _evidence(fn.scores or {})):
            continue
        token = _hash(["limitation", name, plan["before_claim_hash"],
                       sorted(plan["finding_ids"]), sorted(plan["evidence_ids"]),
                       _evidence(fn.scores or {})])
        work[token] = f"Evidence-backed repair limitation: {name}"
        deferred[name] = {"reason": plan["deferral_reason"],
                          "finding_ids": plan["finding_ids"], "evidence_ids": plan["evidence_ids"],
                          "work_id": token}
    for name, saved in (ledger.get("polishing_deferrals") or {}).items():
        fn = current.get(name)
        history = (repairs.get(name) or {}).get("history") or []
        if (fn is None or saved.get("claim_hash") != _claim(fn)
                or saved.get("measurement_hash") != _hash(_evidence(fn.scores or {}))
                or not saved.get("finding_ids") or not saved.get("evidence_ids")
                or (history and history[-1].get("phase") in {"planned", "awaiting_validation"})):
            continue
        token = _hash(["limitation", name, saved["claim_hash"], saved["measurement_hash"],
                       sorted(saved["finding_ids"]), sorted(saved["evidence_ids"])])
        work[token] = f"Evidence-backed repair limitation: {name}"
        deferred[name] = {key: saved[key] for key in ("reason", "finding_ids", "evidence_ids")}
        deferred[name]["work_id"] = token
    return {"version": 2, "work": work, "deferred": deferred, "tests": tests}


def remember_recovery_work(current: dict, previous: dict | None) -> dict:
    """Restoring a known version is not a new experiment; limitations remain current only."""
    return {**current, "work": {**(previous or {}).get("work", {}), **current["work"]}}


def recovery_state_fingerprint(rt: Any) -> str:
    """An exhausted decision is valid only for the same current library and target."""
    ledger = read_json(rt.run.run_dir() / "detector_repairs.json")
    active = {name: [(row.get("plan_id"), row.get("phase"))
                     for row in entry.get("history", [])[-1:]]
              for name, entry in ledger.get("repairs", {}).items()}
    return _hash([
        "detector_bound_measurement_recovery_proof_v3",
        getattr(rt.run, "min_failure_modes", None),
        (getattr(rt, "_family_failed_task_inventory", None) or {}).get("corpus_fingerprint"),
        sorted((fn.name, _claim(fn), _hash(_evidence(fn.scores or {})))
               for fn in rt.library.functions), active,
    ])


def limitation_findings(fn: Any) -> list[dict]:
    """Current measurements can justify a deferral without a contradicted causal finding."""
    from autofa.metrics.repair_feedback import build_repair_feedback

    feedback = build_repair_feedback(fn.to_dict())
    findings = list(feedback["findings"])
    scores = fn.scores or {}
    if (scores.get("evaluation_claim_hash") != _claim(fn) or scores.get("evaluation_stale")
            or not scores.get("evaluated")):
        return findings
    items = sorted({str(item) for item in scores.get("evaluation_item_ids") or []}
                   | {str(item) for bucket in scores.get("buckets") or []
                      for key in ("example_fail_ids", "example_pass_ids")
                      for item in bucket.get(key) or []})
    if items:
        findings.append({"id": "measurement_" + _hash(_evidence(scores))[:16],
                         "status": "measured", "requirement_id": "polishing_disposition",
                         "requirement": "Explain the tested repair or evidence limitation",
                         "item_ids": items,
                         "note": "These are evaluated controls, not causal confirmations."})
    return findings


def public_limitation_findings(fn: Any, *, offset: int, limit: int,
                              item_id: str | None = None) -> list[dict]:
    """Expose references without duplicating complete assessments or large ID inventories."""
    result = []
    for finding in limitation_findings(fn):
        ids = [item for item in finding.get("item_ids") or []
               if item_id is None or item == item_id]
        result.append({
            **{key: finding.get(key) for key in ("id", "status", "requirement", "note")},
            "item_ids": ids[offset:offset + limit], "evidence_total": len(ids),
            "next_offset": offset + limit if offset + limit < len(ids) else None,
        })
    return result


def record_limitation(rt: Any, fn: Any, *, finding_ids: list[str], evidence_ids: list[str],
                      reason: str) -> dict:
    from autofa.session.polishing_guard import polishing_repair_id
    from autofa.session.repair_workflow import record_polishing_deferral

    selected = [row for row in limitation_findings(fn) if row["id"] in finding_ids]
    available = {str(item) for row in selected for item in row.get("item_ids", [])}
    if (not finding_ids or not evidence_ids or {row["id"] for row in selected} != set(finding_ids)
            or not set(evidence_ids) <= available):
        raise ValueError("Deferral requires current finding IDs and their recorded evidence IDs; "
                         "inspect get_function_feedback.recovery_findings")
    return record_polishing_deferral(
        rt.run.run_dir(), fn.to_dict(), reason=reason, finding_ids=finding_ids,
        evidence_ids=evidence_ids, measurement_hash=_hash(_evidence(fn.scores or {})),
        polishing_id=polishing_repair_id(rt, fn.name),
    )


def polishing_response_assessment(rt: Any, state: dict) -> dict:
    """Do not turn repeated unchanged conclusions into additional patience rounds."""
    from autofa.behavior_policy import strictness_for_config
    from autofa.metrics.discovery_grades import grade_summary
    from autofa.session.discovery_target import discovery_good_target_for_runtime

    now = recovery_work(rt, state["frozen_names"], state.get("controls"))
    previous = state.get("recovery_work")
    first_work = set(state.get("recovery_initial_work", (previous or now).get("work", {})))
    previous_work = set((previous or {}).get("work", {}))
    tests = {key: details for key, details in now["tests"].items() if key not in first_work}
    supported_deferrals = {}
    from autofa.metrics.repair_feedback import build_repair_feedback

    for name, limitation in now["deferred"].items():
        fn = next(fn for fn in rt.library.functions if fn.name == name)
        feedback = build_repair_feedback(fn.to_dict())
        unavailable = [row["finding_id"] for row in feedback["investigations"]
                       if row.get("action") == "defer_claim"
                       and row.get("finding_id") in limitation["finding_ids"]]
        relevant_tests = [key for key, details in tests.items() if details["name"] == name
                          or (name in details.get("bindings", {})
                              and set(details["item_ids"]) & set(limitation["evidence_ids"]))]
        if unavailable or relevant_tests:
            supported_deferrals[name] = {
                **limitation, "tested_work_ids": relevant_tests,
                "unavailable_finding_ids": unavailable,
            }
    unsupported_limits = {row["work_id"] for name, row in now["deferred"].items()
                          if name not in supported_deferrals}
    fresh = {key: value for key, value in now["work"].items()
             if key not in previous_work and key not in unsupported_limits}
    summary = grade_summary(
        rt.library.functions, behavior_strictness=strictness_for_config(rt.config),
        review_history=rt.library.review_history,
    )
    outstanding = [row["name"] for row in summary["functions"]
                   if row["name"] in state["frozen_names"]
                   and row["code"] not in {"GOOD", "EXCELLENT", "OUTSTANDING"}]
    target = discovery_good_target_for_runtime(rt)
    minimum_met = target.get("minimum_met", False)
    from autofa.session.detector_service import infrastructure_block

    blocked = infrastructure_block(rt)
    capacity_exhausted = bool(
        not blocked and not minimum_met and not outstanding and state["frozen_names"]
        and set(state["frozen_names"]) == {fn.name for fn in rt.library.functions}
        and len(set(state["frozen_names"])) < target["min_failure_modes"]
    )
    limitations = supported_deferrals
    if capacity_exhausted:
        limitations = {"library_capacity": {
            "reason": "Every admitted detector is Good or better, but the frozen library "
                      "contains fewer candidates than the requested minimum. Final polishing "
                      "cannot add new detector identities.",
            "candidate_count": len(state["frozen_names"]),
            "requested_minimum": target["min_failure_modes"],
        }}
    exhausted = not blocked and (capacity_exhausted or (bool(outstanding)
                                     and all(name in supported_deferrals for name in outstanding)))
    # The first full checkpoint always gets its closing verification. Older
    # saved polishing states establish a work baseline without inventing history.
    baseline = previous is None
    can_close = not blocked and bool(rt.run.round < rt.run.max_rounds or minimum_met or baseline
                                    or fresh or exhausted)
    missing = [name for name in outstanding if name not in supported_deferrals]
    actions = [
        f"get_function_feedback(name={name!r}); test a concrete repair or record a "
        "repair deferral citing its current finding IDs and evidence IDs"
        for name in missing
    ]
    return {
        "can_close": can_close, "substantive": bool(fresh) and not baseline,
        "baseline": baseline, "work_ids": sorted(fresh) if not baseline else [],
        "work_completed": list(dict.fromkeys(fresh.values())) if not baseline else [],
        "exhausted": exhausted, "capacity_exhausted": capacity_exhausted,
        "infrastructure_blocked": bool(blocked), "service": blocked,
        "limitations": limitations if exhausted else {},
        "outstanding_actions": actions,
        "next_action": (blocked["next_action"] if blocked else
                        "No new completed repair or evidence decision was recorded. "
                        "This round remains open; repeating retain/defer text does not spend "
                        "patience. "
                        + (actions[0] + ". " if actions else "Inspect validation blockers. ")
                        + "A failed repair experiment counts once. If every remaining repair is "
                        "unsupported, call defer_function_repair with current finding_ids, "
                        "evidence_ids and the specific limitation; the runtime can "
                        "then end recovery explicitly under the stall setting."),
        "snapshot": remember_recovery_work(now, previous),
    }
