# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local, bounded detector repair history and explicit saved-version selection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autofa.behavior_policy import (
    BEHAVIOR_STRICTNESSES,
    strictness_for_scores,
    validate_behavior_strictness,
)
from autofa.core.dependencies import review_source
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.claude_judge import compatible_judge_policy, judge_policy_hash

SCHEMA_VERSION = "detector-repair-v1"
MAX_ATTEMPTS = 3
MAX_IMPLEMENTATION_ATTEMPTS = 2
ACTIONS = frozenset({
    "tighten_detector", "revise_claim", "split_detector", "gather_evidence",
    "retry_review", "retire", "repair_implementation", "refine_detector",
})
_LOCK = threading.RLock()
_ACTIVE = {"planned", "awaiting_validation"}
_UNSET = object()


def _current_plan_policy(plan: dict) -> bool:
    # Plans and scores saved before strictness was recorded are not current.
    if plan.get("judge_behavior_strictness") not in BEHAVIOR_STRICTNESSES:
        return False
    return compatible_judge_policy(
        plan.get("judge_policy_hash"),
        behavior_strictness=plan.get("judge_behavior_strictness"),
    )


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _claim(function: dict) -> dict:
    return {
        "source": review_source(function),
        "description": function.get("description") or "",
        "mechanism": (function.get("mechanism") or function.get("description")
                      or function.get("name") or ""),
    }


def _version(function: dict) -> str:
    """Include execution dependencies as well as literal claim/source provenance."""
    fields = {key: function.get(key) for key in (
        "name", "source", "description", "mechanism", "kind", "llm_prompt",
        "depends_on", "explains",
    )}
    fields["deps"] = (function.get("meta") or {}).get("deps")
    fields["dependency_bundle"] = (function.get("meta") or {}).get("dependency_bundle")
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def _ids(values: Any) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Evidence IDs must be a list of nonempty strings")
    return list(dict.fromkeys(values))


def _review_ids(function: dict) -> list[str]:
    return sorted({str(row["item_id"]) for row in
                   (function.get("scores") or {}).get("behavior_verdicts", [])
                   if row.get("item_id")})


@contextmanager
def _ledger(run_dir: Path, *, write: bool = False):
    directory = Path(run_dir)
    path = directory / "detector_repairs.json"
    if not write and not path.exists():
        yield {"schema_version": SCHEMA_VERSION, "repairs": {}}
        return
    directory.mkdir(parents=True, exist_ok=True)
    with _LOCK, (directory / ".repair_workflow.lock").open("a") as lock:
        try:
            import fcntl
        except ImportError:  # Windows retains the in-process lock and atomic replacement.
            fcntl = None
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text()) if path.exists() else {
                "schema_version": SCHEMA_VERSION, "repairs": {},
            }
            if data.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported repair workflow schema")
            yield data
            if write:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        "w", dir=directory, prefix=".repair-", delete=False,
                    ) as temp:
                        temp_path = Path(temp.name)
                        json.dump(data, temp, ensure_ascii=False, allow_nan=False, indent=2)
                        temp.flush()
                        os.fsync(temp.fileno())
                    os.replace(temp_path, path)
                finally:
                    if temp_path and temp_path.exists():
                        temp_path.unlink()
        finally:
            if fcntl:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _locate(data: dict, name: str, plan_id: str | None = None):
    direct = data["repairs"].get(name)
    if not plan_id and direct and direct["history"]:
        return direct, direct["history"][-1]
    for state in data["repairs"].values():
        if not state["history"]:
            continue
        plan = state["history"][-1]
        if plan_id:
            if plan["plan_id"] == plan_id:
                return state, plan
        elif name == state["name"] or name in plan["candidates"]:
            return state, plan
    return None, None


def _attempts_used(state: dict) -> int:
    # Processing recovery does not expose a fresh validation sample or authorize
    # a changed candidate, so it does not spend the bounded repair budget.
    return sum(plan["action"] != "retry_review" for plan in state["history"])


def repair_attempt_budget(state: dict, *, polishing_id: str | None = None) -> dict:
    """Expose the current allowance while retaining the lifetime attempt count."""
    attempts = (sum(plan["action"] != "retry_review"
                    and not plan.get("implementation_allowance")
                    and plan.get("polishing_id") == polishing_id
                    for plan in state.get("history", []))
                if polishing_id else sum(plan["action"] != "retry_review"
                                         and not plan.get("implementation_allowance")
                                         for plan in state.get("history", [])))
    return {
        "scope": "polishing_round" if polishing_id else "discovery",
        "polishing_id": polishing_id, "max_attempts": MAX_ATTEMPTS,
        "attempts_used": attempts, "remaining": max(0, MAX_ATTEMPTS - attempts),
    }


def implementation_repair_budget(state: dict) -> dict:
    """A lifetime allowance for demonstrated code defects, independent of search."""
    used = sum(bool(plan.get("implementation_allowance")) for plan in state.get("history", []))
    return {"scope": "implementation_recovery", "max_attempts": MAX_IMPLEMENTATION_ATTEMPTS,
            "attempts_used": used, "remaining": max(0, MAX_IMPLEMENTATION_ATTEMPTS - used)}


def implementation_diagnostic_finding(function: dict, diagnostic: dict | None) -> dict | None:
    """Runtime smoke receipts establish a reproducible mismatch, not causal validity."""
    if not diagnostic:
        return None
    if (diagnostic.get("version_hash") != _version(function)
            or diagnostic.get("ok") is not True or not diagnostic.get("results")):
        raise ValueError("Implementation diagnostics must check the current saved version")
    failed = [row for row in diagnostic["results"] if row.get("status") == "error"
              or row.get("matches_expected") is False]
    if not failed:
        raise ValueError("Implementation recovery needs a reproduced execution or output error")
    return {"id": "implementation_" + hashlib.sha256(json.dumps(
        [diagnostic["version_hash"], failed], sort_keys=True).encode()).hexdigest()[:16],
        "status": "implementation_error", "issue_type": "reproduced_implementation_error",
        "item_ids": [row["item_id"] for row in failed], "examples": _copy(failed)}


def record_implementation_check(run_dir: Path, function: dict, plan_id: str, result: dict) -> dict:
    """Keep failed proposals in the ledger without replacing the live baseline."""
    with _ledger(run_dir, write=True) as data:
        state, plan = _locate(data, function["name"], plan_id)
        if not plan or plan["phase"] != "planned" or not plan.get("implementation_diagnostics"):
            raise ValueError("No pending implementation regression check")
        receipt = {"version_hash": _version(function), "result": _copy(result),
                   "passed": result.get("ok") is True and result.get("checks_passed") is True
                   and not result.get("errors") and not result.get("abstentions")}
        plan["implementation_check"] = receipt
        if not receipt["passed"]:
            plan["failed_proposal"] = _copy(function)
            plan["deferral_reason"] = "Implementation regression checks failed; baseline retained."
            _touch(state, plan, "deferred")
        return _copy(receipt)


def _check_polishing_id(run_dir: Path, name: str, polishing_id: str) -> None:
    """An internal caller cannot spend a historical or invented phase allowance."""
    try:
        phase = json.loads((run_dir / "round_validation.json").read_text())["refinement"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("The final polishing phase is unavailable") from exc
    if (not isinstance(phase, dict) or phase.get("mode") != "polishing"
            or phase.get("polishing_id") != polishing_id
            or phase.get("status") not in {"awaiting_response", "validating_response"}
            or name not in (phase.get("frozen_names") or [])):
        raise ValueError("The repair does not belong to the active final polishing phase")


def _touch(state: dict, plan: dict, phase: str) -> None:
    if phase == "deferred" and not repair_attempt_budget(
        state, polishing_id=plan.get("polishing_id"),
    )["remaining"]:
        phase = "exhausted"
    plan["phase"] = phase
    plan["updated_at"] = _now()
    state["state"] = state["status"] = phase


def _candidate(function: dict) -> dict:
    return {
        "after": _copy(function), "after_claim_hash": claim_hash(**_claim(function)),
        "after_version_hash": _version(function), "phase": "awaiting_validation",
        "reservation": None, "result": None, "outcome": None,
    }


def _expose(state: dict) -> dict:
    result = _copy(state)
    result["attempts_used"] = _attempts_used(result)
    result["implementation_budget"] = implementation_repair_budget(result)
    result["latest"] = result["history"][-1] if result["history"] else None
    result["attempt_budget"] = repair_attempt_budget(
        result, polishing_id=(result["latest"] or {}).get("polishing_id"),
    )
    return result


def _split_outcome(plan: dict) -> dict:
    """The final child cannot stand in for the outcome of the complete split."""
    candidates = list(plan["candidates"].values())
    outcomes = [candidate["outcome"] for candidate in candidates if candidate.get("outcome")]
    pending = len(candidates) < 2 or len(outcomes) != len(candidates)
    counts = {key: sum(outcome[key] for outcome in outcomes)
              for key in ("n_judged", "n_yes", "n_no", "n_uncertain", "n_errors")}
    if pending:
        status = "awaiting_validation"
    elif any(o["review_status"] == "insufficient_new_evidence" for o in outcomes):
        status = "insufficient_new_evidence"
    elif any(o["review_status"] == "reused_evidence" for o in outcomes):
        status = "reused_evidence"
    elif counts["n_errors"]:
        status = "processing_error"
    elif counts["n_uncertain"]:
        status = "uncertain"
    else:
        status = "complete"
    complete = status == "complete"
    return {
        "review_status": status, **counts, "n_candidates": len(candidates),
        "n_candidates_reviewed": len(outcomes),
        "candidate_outcomes": {name: _copy(candidate["outcome"])
                               for name, candidate in plan["candidates"].items()},
        "all_reviewed_confirmed": complete and all(o["all_reviewed_confirmed"] for o in outcomes),
        "original_claim_validated": complete and all(
            o["original_claim_validated"] for o in outcomes
        ),
        "claim_preserved": bool(outcomes) and all(o["claim_preserved"] for o in outcomes),
        "comparison": "split_detector_validation", "sample_kind": "fresh_repair_holdout",
        "automatic_promotion": False,
    }


def get_repair(run_dir: Path, name: str) -> dict:
    with _ledger(run_dir) as data:
        state, _ = _locate(data, name)
        if state is None:
            # Read access to archived split children must survive later parent attempts.
            # This lookup is deliberately absent from mutation/validation routing.
            for archived in data["repairs"].values():
                for plan in reversed(archived["history"]):
                    if name in plan["candidates"]:
                        view = _expose(archived)
                        view.update(latest=_copy(plan), state=plan["phase"], status=plan["phase"],
                                    historical_link=True, requested_name=name,
                                    linked_plan_id=plan["plan_id"])
                        return view
            return {}
        return _expose(state)


def list_repairs(run_dir: Path) -> list[dict]:
    with _ledger(run_dir) as data:
        return [_expose(state) for state in data["repairs"].values()]


def select_version(
    run_dir: Path, name: str, *, version_hash: str, reason: str,
    current_function: dict | None, expected_current_version_hash: str,
    apply: Callable[[dict, list[dict]], None] | None = None,
) -> dict:
    """Select an archived scored snapshot without rewriting repair outcomes.

    The caller can persist the selected library entry under the ledger lock.
    Failed persistence leaves the selection unrecorded. Historic scores remain
    historic; selection neither spends nor replenishes the repair budget.
    """
    from autofa.session.repair_versions import collect_versions

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Selecting a saved version requires an explicit reason")
    current_version = _version(current_function) if current_function else ""
    if current_function and current_function.get("name") != name:
        raise ValueError("Current function name does not match the selection")
    if current_version != expected_current_version_hash:
        raise ValueError("Current detector version changed; list versions again")
    with _ledger(run_dir, write=True) as data:
        for state in data["repairs"].values():
            latest = state["history"][-1] if state["history"] else None
            if (latest and latest["phase"] in _ACTIVE
                    and (name == state["name"] or name in latest["candidates"])):
                raise ValueError("Finish or explicitly defer the active repair before selecting")
        versions = collect_versions(data, name, current_function)
        selected = versions.get(version_hash)
        if selected is None:
            raise ValueError("Unknown scored detector version")
        # Keep the replaced entry available to undo this explicit selection.
        archive = data.setdefault("versions", {}).setdefault(name, {})
        if current_function and (current_function.get("scores") or {}).get("evaluated"):
            archive[current_version] = _copy(current_function)
        archive[version_hash] = _copy(selected)
        receipt = {
            "version_hash": version_hash, "previous_version_hash": current_version,
            "reason": reason.strip(), "selected_at": _now(),
            "fresh_validation": False,
        }
        data.setdefault("version_selections", {}).setdefault(name, []).append(receipt)
        if apply is not None:
            # Persistence also carries the global review-exposure history. Pass
            # the snapshot already protected by this lock instead of reopening
            # the same repair ledger from Runtime.persist_library.
            apply(_copy(selected), [_expose(state) for state in data["repairs"].values()])
        return {"function": _copy(selected), "selection": _copy(receipt)}


def plan_repair(  # noqa: C901
    run_dir: Path, function: dict, feedback: dict, *, action: str, rationale: str,
    finding_ids: list[str], evidence_ids: list[str],
    objective_strategy: str | None = None, objective_settings: dict | None = None,
    behavior_strictness: str | None = None, polishing_id: str | None = None,
    implementation_diagnostics: dict | None = None,
) -> dict:
    """Freeze a grounded decision and baseline before any submitted edit."""
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Function name is required")
    if polishing_id is not None:
        _check_polishing_id(Path(run_dir), name, polishing_id)
    if action not in ACTIONS or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("A supported repair action and explicit rationale are required")
    fingerprint = claim_hash(**_claim(function))
    if feedback.get("claim_hash") != fingerprint:
        raise ValueError("Repair feedback is stale for this function/claim")
    finding_ids, evidence_ids = _ids(finding_ids), _ids(evidence_ids)
    diagnostic_finding = implementation_diagnostic_finding(function, implementation_diagnostics)
    if diagnostic_finding:
        if action != "repair_implementation":
            raise ValueError("Implementation diagnostics only authorize repair_implementation")
        feedback = _copy(feedback)
        feedback["findings"] = [*feedback.get("findings", []), diagnostic_finding]
        feedback["allowed_actions"] = list(dict.fromkeys([
            *feedback.get("allowed_actions", []), "repair_implementation"]))
        if not finding_ids:
            finding_ids = [diagnostic_finding["id"]]
            evidence_ids = diagnostic_finding["item_ids"]
    findings = feedback.get("findings") or []
    valid_findings = {f.get("id") for f in findings}
    valid_evidence = set(_ids(feedback.get("diagnostic_item_ids") or []))
    valid_evidence.update(item_id for f in findings for item_id in _ids(f.get("item_ids") or []))
    valid_evidence.update(str(e["item_id"]) for f in findings
                          for e in f.get("examples", []) if e.get("item_id"))
    if not set(finding_ids) <= valid_findings or not set(evidence_ids) <= valid_evidence:
        raise ValueError("Repair references unknown finding or evidence IDs")
    if action != "gather_evidence" and (not finding_ids or not evidence_ids):
        raise ValueError("This repair action requires finding and evidence references")
    selected = [f for f in findings if f.get("id") in finding_ids]
    selected_items = {item_id for f in selected for item_id in _ids(f.get("item_ids") or [])}
    selected_items.update(str(e["item_id"]) for f in selected
                          for e in f.get("examples", []) if e.get("item_id"))
    if not set(evidence_ids) <= selected_items:
        raise ValueError("Evidence IDs must belong to the selected findings")
    required_status = {
        "tighten_detector": {"contradicted"}, "split_detector": {"contradicted"},
        "retire": {"contradicted"}, "revise_claim": {"contradicted", "unknown"},
        "retry_review": {"processing_error"},
        "repair_implementation": {"execution_error", "implementation_error"},
        "refine_detector": {"quality_gap"},
    }.get(action)
    if action == "revise_claim":
        from autofa.metrics.repair_feedback import claim_compilation_rejected

        if claim_compilation_rejected(function.get("scores") or {}):
            required_status = required_status | {"processing_error"}
    if required_status and not any(f.get("status") in required_status for f in selected):
        raise ValueError("Selected finding status does not support the repair action")
    if action == "repair_implementation":
        from autofa.metrics.repair_feedback import detector_execution_findings

        recorded = {finding["id"]: finding for finding in detector_execution_findings(function)}
        if diagnostic_finding:
            recorded[diagnostic_finding["id"]] = diagnostic_finding
        if (not set(finding_ids) <= recorded.keys()
                or not set(evidence_ids) <= {
                    item_id for key in finding_ids for item_id in recorded[key]["item_ids"]
                }):
            raise ValueError("Implementation repair requires current recorded execution errors "
                             "or reproduced implementation diagnostics")
    allowed = feedback.get("allowed_actions")
    if isinstance(allowed, list) and action not in allowed:
        raise ValueError("Repair action is not supported by the current findings")
    scores = function.get("scores") or {}
    if objective_strategy is None:
        objective_strategy = scores.get("judge_objective_strategy", "balanced")
    behavior_strictness = validate_behavior_strictness(
        behavior_strictness or strictness_for_scores(scores))
    if action == "refine_detector":
        from autofa.metrics.repair_feedback import (
            build_repair_feedback,
            has_complete_causal_support,
        )

        current_feedback = build_repair_feedback(function)
        recorded = {row["id"]: row for row in current_feedback["findings"]
                    if row["status"] == "quality_gap"}
        if (not has_complete_causal_support(
                function, objective_strategy=objective_strategy,
                behavior_strictness=behavior_strictness)
                or not set(finding_ids) <= recorded.keys()
                or not set(evidence_ids) <= {
                    item for key in finding_ids for item in recorded[key]["item_ids"]
                }):
            raise ValueError("Quality refinement requires current supported claim review and "
                             "recorded evaluation findings and evidence")
    policy_hash = judge_policy_hash(behavior_strictness=behavior_strictness)
    retry_ids = []
    if action == "retry_review":
        if (scores.get("claim_hash") != fingerprint
                or not _current_plan_policy(scores)
                or strictness_for_scores(scores) != behavior_strictness):
            raise ValueError("Processing retry requires the same current claim and judge policy")
        # A processing retry belongs to its original policy and review identity.
        # Changing search weights does not alter either or discard completed work.
        policy_hash = scores["judge_policy_hash"]
        objective_strategy = scores.get("judge_objective_strategy", "balanced")
        grouped: dict[str, list[str]] = {}
        for row in scores.get("behavior_verdicts", []):
            if row.get("item_id"):
                grouped.setdefault(str(row["item_id"]), []).append(row.get("verdict"))
        retry_ids = sorted(key for key, labels in grouped.items()
                           if len(labels) == 1 and labels[0] in {"error", "processing_error"})
        if not retry_ids or not set(evidence_ids) <= set(retry_ids):
            raise ValueError("Only recorded processing errors may be retried")
        retry_ids = list(evidence_ids)

    from autofa.session.review_evidence import review_context

    portable_context = review_context(run_dir, name, related_names=[
        value for value in [function.get("explains")] + list(function.get("depends_on") or [])
        if isinstance(value, str) and value
    ])
    with _ledger(run_dir, write=True) as data:
        if polishing_id is not None:
            _check_polishing_id(Path(run_dir), name, polishing_id)
        state = data["repairs"].get(name)
        owner, owned_plan = _locate(data, name)
        if (owner is not None and owner["name"] != name and owned_plan["phase"] in _ACTIVE):
            raise ValueError("Finish or defer the active parent split before repairing its child")
        implementation_allowance = False
        if state and action != "retry_review" and not repair_attempt_budget(
            state, polishing_id=polishing_id,
        )["remaining"]:
            implementation_allowance = (
                action == "repair_implementation" and diagnostic_finding is not None
                and implementation_repair_budget(state)["remaining"] > 0)
            if not implementation_allowance:
                qualifier = "in this polishing round" if polishing_id else "per function"
                raise ValueError(f"Repair attempt limit reached ({MAX_ATTEMPTS} {qualifier}); "
                                 "bounded implementation recovery requires reproduced checks "
                                 "and its own remaining allowance")
        if state and state["state"] in _ACTIVE:
            raise ValueError("Finish or explicitly defer the active repair before another plan")
        if state is None:
            state = data["repairs"][name] = {
                "name": name, "schema_version": SCHEMA_VERSION,
                "baseline": _copy(function), "baseline_claim_hash": fingerprint,
                "baseline_version_hash": _version(function), "max_attempts": MAX_ATTEMPTS,
                "history": [], "reviewed_item_ids": [],
            }
        # A split child was constructed using the parent's recorded evidence.
        # Starting its own workflow must not make those cases fresh again, even
        # when a later parent attempt has moved the split into archived history.
        inherited_ids = {
            item_id for parent in data["repairs"].values()
            if parent["name"] != name and any(
                name in prior["candidates"] for prior in parent["history"]
            )
            for item_id in parent["reviewed_item_ids"]
        }
        from autofa.session.repair_versions import collect_versions

        archived_review_ids = {
            item_id for version in collect_versions(data, name, function).values()
            for item_id in _review_ids(version)
        }
        state["reviewed_item_ids"] = sorted(
            set(state["reviewed_item_ids"]) | set(_review_ids(function)) | valid_evidence
            | inherited_ids | archived_review_ids | set(portable_context["reviewed_item_ids"])
        )
        plan = {
            "plan_id": uuid.uuid4().hex, "name": name,
            "attempt": _attempts_used(state) + int(action != "retry_review"), "action": action,
            "rationale": rationale.strip(), "finding_ids": finding_ids,
            "evidence_ids": evidence_ids, "feedback": _copy(feedback),
            "before": _copy(function), "before_claim_hash": fingerprint,
            "before_version_hash": _version(function), "judge_policy_hash": policy_hash,
            "judge_objective_strategy": objective_strategy,
            "judge_behavior_strictness": behavior_strictness,
            "objective_settings": _copy(objective_settings),
            **({"polishing_id": polishing_id} if polishing_id else {}),
            "after": None, "after_claim_hash": None, "after_version_hash": None,
            "candidates": {}, "retry_item_ids": retry_ids,
            "reservation": None, "result": None, "outcome": None,
            "created_at": _now(),
            **({"implementation_allowance": True} if implementation_allowance else {}),
            **({"implementation_diagnostics": _copy(implementation_diagnostics)}
               if implementation_diagnostics else {}),
        }
        state["history"].append(plan)
        if polishing_id:
            plan["attempt_budget"] = repair_attempt_budget(state, polishing_id=polishing_id)
        if action in {"retry_review", "gather_evidence"}:
            candidate = _candidate(function)
            plan["candidates"][name] = candidate
            for key in ("after", "after_claim_hash", "after_version_hash"):
                plan[key] = _copy(candidate[key])
            _touch(state, plan, "awaiting_validation")
        else:
            _touch(state, plan, "planned")
        return _copy(plan)


def record_submission(
    run_dir: Path, function: dict, repair_plan_id: str | None = None, *,
    current_function: dict | None | object = _UNSET,
    apply: Callable[[list[dict]], None] | None = None,
) -> dict | None:
    """Bind a candidate and, optionally, persist it under the same ledger lock.

    Library submissions supply ``apply`` so a failed save leaves the repair plan
    unsubmitted. The callback receives staged repair history, avoiding a nested
    read of this locked ledger. Baseline archives share the same transaction.
    """
    archive = (apply is not None and isinstance(current_function, dict)
               and (current_function.get("scores") or {}).get("evaluated") is True)
    if (not (Path(run_dir) / "detector_repairs.json").exists()
            and not repair_plan_id and not archive):
        if apply is not None:
            apply([])
        return None
    with _ledger(run_dir, write=True) as data:
        result = _record_submission(data, function, repair_plan_id, current_function)
        if archive:
            data.setdefault("versions", {}).setdefault(current_function["name"], {})[
                _version(current_function)] = _copy(current_function)
        if apply is not None:
            apply([_expose(state) for state in data["repairs"].values()])
        return result


def _record_submission(  # noqa: C901
    data: dict, function: dict, repair_plan_id: str | None,
    current_function: dict | None | object,
) -> dict | None:
    name = function.get("name")
    state, plan = _locate(data, name, repair_plan_id)
    if plan is None or plan["phase"] not in _ACTIVE:
        if repair_plan_id:
            raise ValueError("Unknown or inactive repair plan")
        # Reject an unplanned rewrite before submit_function replaces the
        # reviewed library entry. Validation alone is too late: evaluation
        # has already run and the previous scores have been discarded.
        prior = plan
        if prior is None:
            prior = next((old for saved in data["repairs"].values()
                          for old in reversed(saved["history"])
                          if name in old["candidates"]), None)
        if prior is not None:
            saved = prior["candidates"].get(name)
            expected_version = (saved["after_version_hash"] if saved
                                else prior["before_version_hash"])
            selections = (data.get("version_selections") or {}).get(name) or []
            if (selections and isinstance(current_function, dict)
                    and selections[-1]["version_hash"] == _version(current_function)):
                expected_version = selections[-1]["version_hash"]
            if (_version(function) != expected_version
                    or (isinstance(current_function, dict)
                        and _version(current_function) != _version(function))):
                raise ValueError("Changing a repaired function needs a new active repair plan")
        return None
    if name != state["name"] and plan["action"] != "split_detector":
        raise ValueError("Only split_detector can link a differently named candidate")
    if name != state["name"] and name in data["repairs"]:
        raise ValueError("Split child already has a separate repair workflow; use a new name")
    if name != state["name"] and any(
        other["name"] != state["name"]
        and any(name in prior["candidates"] for prior in other["history"])
        for other in data["repairs"].values()
    ):
        raise ValueError("Split child name already belongs to another repair workflow")
    existing = plan["candidates"].get(name)
    if current_function is not _UNSET:
        if existing:
            expected_version = existing["after_version_hash"]
        elif name == state["name"]:
            expected_version = plan["before_version_hash"]
        else:
            expected_version = None
        if expected_version is None:
            if current_function is not None:
                raise ValueError("Split child would overwrite an untracked existing function")
        elif (current_function is None
              or _version(current_function) != expected_version):
            raise ValueError("Current function changed after the repair plan or submission")
    if plan["action"] in {"gather_evidence", "retry_review", "retire"}:
        if _version(function) != plan["before_version_hash"]:
            raise ValueError("This action does not authorize changing the function")
        return _copy(plan)
    if (plan["action"] in {"tighten_detector", "repair_implementation", "refine_detector"}
            and any(function.get(key) != plan["before"].get(key)
                    for key in ("description", "mechanism"))):
        raise ValueError(f"{plan['action']} must preserve the literal original claim")
    if plan["action"] in {"repair_implementation", "refine_detector"} and any(
        function.get(key) != plan["before"].get(key) for key in ("depends_on", "explains")
    ):
        raise ValueError(f"{plan['action']} must preserve detector relationships")
    if (plan["action"] in {"repair_implementation", "refine_detector"}
            and review_source(function) == review_source(plan["before"])):
        raise ValueError(f"{plan['action']} must change the executed detector or helpers")
    if _version(function) == plan["before_version_hash"]:
        raise ValueError("Submit a changed candidate version before validating this repair")
    if (plan["action"] == "split_detector"
            and _version({**function, "name": state["name"]}) == plan["before_version_hash"]):
        raise ValueError("Split candidates must change the detector or claim beyond its name")
    if existing:
        if existing["after_version_hash"] == _version(function):
            return _copy(plan)
        raise ValueError("Candidate is frozen; plan another bounded attempt to change it")
    if plan.get("implementation_diagnostics"):
        check = plan.get("implementation_check") or {}
        if not check.get("passed") or check.get("version_hash") != _version(function):
            raise ValueError("Implementation regression checks must pass before replacement")
    candidate = plan["candidates"][name] = _candidate(function)
    if name == state["name"] or plan["after"] is None:
        for key in ("after", "after_claim_hash", "after_version_hash"):
            plan[key] = _copy(candidate[key])
    _touch(state, plan, "awaiting_validation")
    return _copy(plan)



def _ordinary_confirmation_allowed(
    function: dict, plan: dict, *, objective_strategy: str | None,
    behavior_strictness: str | None = None,
    judge_settings: dict | None, allocation: dict | None,
    review_key: str | None = None, run_id: str | None = None,
    review_phase: str = "confirmation",
) -> bool:
    """A parked, unreviewed repair may review its unchanged supported candidate.

    This never reopens the attempt or consumes another repair attempt. Failed,
    incomplete and interrupted reserved validations still require their original
    recovery path; they cannot acquire a new holdout through ordinary judging.
    """
    candidate = plan["candidates"].get(function.get("name"))
    strategy = plan.get("judge_objective_strategy")
    strictness = plan.get("judge_behavior_strictness")
    if (plan["phase"] not in {"deferred", "exhausted"}
            or (behavior_strictness is not None and behavior_strictness != strictness)
            or not candidate or candidate["phase"] != "awaiting_validation"
            or candidate["after_version_hash"] != _version(function)
            or candidate["after_claim_hash"] != claim_hash(**_claim(function))
            or not _current_plan_policy(plan)
            or any(candidate.get(key) is not None or plan.get(key) is not None
                   for key in ("reservation", "result", "outcome"))):
        return False
    scores = function.get("scores") or {}
    request = scores.get("ordinary_confirmation_request") or {}
    if not isinstance(request, dict):
        return False
    if (request.get("origin_run_id") == run_id
            and request.get("review_key") != review_key):
        return False
    if (not judge_settings or any(scores.get(key) != value
                                  for key, value in judge_settings.items()
                                  if key != "judge_objective_strategy")):
        return False
    from autofa.metrics.repair_feedback import build_repair_feedback, has_complete_causal_support

    if scores.get("behavior_review_phase") == "development" and has_complete_causal_support(
        function, objective_strategy=strategy, behavior_strictness=strictness,
    ):
        return True
    rows = scores.get("behavior_verdicts") or []
    if (scores.get("behavior_review_phase") != review_phase or not allocation
            or not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows)
            or {row.get("item_id") for row in rows} != set(allocation["item_ids"])):
        return False
    # Successful confirmation reaches completed-operation reuse. A changed
    # evidence_n or judge configuration cannot manufacture another fresh cohort.
    if has_complete_causal_support(
            function, objective_strategy=strategy, behavior_strictness=strictness):
        return True
    if allocation["operation_completed"]:
        return False
    feedback = build_repair_feedback(function)
    return bool(
        feedback["review_detail_complete"] and not feedback["review_integrity_errors"]
        and feedback["validation"]["status"] not in {"stale", "legacy", "invalid", "unjudged"}
        and not scores.get("behavior_judgment_stale")
        and (scores.get("behavior_n_errors") or feedback.get("review_completion_only"))
        and feedback["findings"]
        and all(row["status"] in {"supported", "processing_error"}
                or (row["status"] == "unknown" and row.get("unknown_reason") == "review_incomplete")
                for row in feedback["findings"])
    )


def validation_context(
    run_dir: Path, function: dict, *, resume_reserved: bool = False,
    allow_deferred_confirmation: bool = False, objective_strategy: str | None = None,
    confirmation_review_key: str | None = None, judge_settings: dict | None = None,
    allow_deferred_development: bool = False, development_review_key: str | None = None,
    behavior_strictness: str | None = None,
) -> dict | None:
    name = function.get("name")
    allocation = None
    if allow_deferred_confirmation and confirmation_review_key:
        from autofa.session.review_evidence import find_review_allocation

        # Read before taking the repair lock, matching the established lock order.
        allocation = find_review_allocation(
            run_dir, name, review_key=confirmation_review_key, phase="confirmation",
        )
    development_allocation = None
    if allow_deferred_development and development_review_key:
        from autofa.session.review_evidence import find_review_allocation

        development_allocation = find_review_allocation(
            run_dir, name, review_key=development_review_key, phase="development",
        )
    with _ledger(run_dir) as data:
        state, plan = _locate(data, name)
        if plan is None:
            if any(name in old["candidates"] for saved in data["repairs"].values()
                   for old in saved["history"]):
                raise ValueError(
                    "Previously repaired child needs a new active repair plan to judge"
                )
            return None
        if (behavior_strictness is not None
                and plan.get("judge_behavior_strictness") != behavior_strictness
                and plan["phase"] in _ACTIVE):
            raise ValueError("Judge policy changed; defer this repair before a new review")
        if plan["phase"] not in _ACTIVE:
            if allow_deferred_development and _ordinary_confirmation_allowed(
                function, plan, objective_strategy=objective_strategy,
                behavior_strictness=behavior_strictness,
                judge_settings=judge_settings, allocation=development_allocation,
                review_key=development_review_key, run_id=Path(run_dir).name,
                review_phase="development",
            ):
                return None
            if allow_deferred_confirmation and _ordinary_confirmation_allowed(
                function, plan, objective_strategy=objective_strategy,
                behavior_strictness=behavior_strictness,
                judge_settings=judge_settings, allocation=allocation,
                review_key=confirmation_review_key, run_id=Path(run_dir).name,
            ):
                return None
            candidate = plan["candidates"].get(name)
            if (plan["phase"] == "completed" and candidate
                    and (candidate.get("outcome") or {}).get("all_reviewed_confirmed")
                    and candidate["after_version_hash"] == _version(function)
                    and _current_plan_policy(plan)):
                # Ordinary revalidation does not create another repair outcome or
                # retroactively change this completed attempt's held-out evidence.
                return None
            raise ValueError("Unresolved repair needs a new active plan before another judgment")
        candidate = plan["candidates"].get(name)
        if plan["phase"] == "planned" or candidate is None:
            raise ValueError("Repair needs a submitted candidate before validation")
        if candidate["after_version_hash"] != _version(function):
            raise ValueError("Candidate changed after repair submission")
        if candidate["phase"] != "awaiting_validation":
            raise ValueError("This candidate has already been reviewed")
        if not _current_plan_policy(plan):
            raise ValueError("Judge policy changed; defer this plan and create a fresh attempt")
        reservation = candidate["reservation"]
        if reservation is not None and not resume_reserved:
            raise ValueError("Validation already reserved; record or defer the interrupted review")
        return {
            "plan_id": plan["plan_id"], "name": name, "action": plan["action"],
            "judge_objective_strategy": plan.get("judge_objective_strategy"),
            "judge_behavior_strictness": plan.get("judge_behavior_strictness"),
            "judge_policy_hash": plan["judge_policy_hash"],
            # A resumed review must keep the same held-out sample. Its reserved
            # IDs already belong to reviewed_item_ids, but were not excluded
            # when this review began; preserve that original distinction.
            "exclusion_item_ids": list(reservation["excluded_item_ids"]
                                       if reservation is not None
                                       else state["reviewed_item_ids"]),
            "reserved_item_ids": (list(reservation["item_ids"])
                                  if reservation is not None else None),
            "sample": _copy(reservation.get("sample")) if reservation is not None else None,
            "retry_item_ids": list(plan["retry_item_ids"]),
            "candidate_claim_hash": candidate["after_claim_hash"],
            "candidate_version_hash": candidate["after_version_hash"],
            "sample_kind": (reservation["sample_kind"] if reservation is not None else
                            "processing_retry" if plan["action"] == "retry_review"
                            else "fresh_repair_holdout"),
        }


def reserve_validation(
    run_dir: Path, name: str, item_ids: list[str], *, expected_plan_id: str | None = None,
    resume_existing: bool = False, sample: dict | None = None,
) -> dict:
    ids = _ids(item_ids)
    if len(ids) != len(item_ids):
        raise ValueError("Validation IDs must be unique")
    with _ledger(run_dir, write=True) as data:
        state, plan = _locate(data, name)
        if expected_plan_id is not None and (plan is None or plan["plan_id"] != expected_plan_id):
            raise ValueError("Repair plan changed before evidence could be reserved")
        if plan is None or plan["phase"] != "awaiting_validation":
            raise ValueError("No repair candidate is awaiting validation")
        candidate = plan["candidates"].get(name)
        if candidate is None or candidate["phase"] != "awaiting_validation":
            raise ValueError("No unreviewed repair candidate matches this function")
        if not _current_plan_policy(plan):
            raise ValueError("Judge policy changed; defer this plan and create a fresh attempt")
        if candidate["reservation"] is not None:
            if not resume_existing:
                raise ValueError("Validation has already been reserved")
            if expected_plan_id is None:
                raise ValueError("Resuming reserved validation requires the original plan ID")
            if ids != candidate["reservation"]["item_ids"]:
                raise ValueError("Resumed validation must use the same reserved evidence IDs")
            return _copy(candidate["reservation"])
        retry = plan["action"] == "retry_review"
        if retry and not set(ids) <= set(plan["retry_item_ids"]):
            raise ValueError(
                "Processing retry contains an item without a recorded processing error"
            )
        reused = (sample or {}).get("reused_n", 0) if not retry else 0
        if reused and (reused != len(ids) or (sample or {}).get("eligible_n") != 0
                       or not set(ids) <= set(state["reviewed_item_ids"])):
            raise ValueError("Reused scoring requires an exhausted fresh pool "
                             "and recorded examples")
        if not retry and not reused and set(ids) & set(state["reviewed_item_ids"]):
            raise ValueError("Repair validation must exclude previously reviewed repair evidence")
        reservation = {
            "item_ids": ids, "reserved_at": _now(),
            "excluded_item_ids": list(state["reviewed_item_ids"]),
            "sample_kind": ("processing_retry" if retry else
                            "reused_review_evidence" if reused else "fresh_repair_holdout"),
            "sample": _copy(sample),
        }
        candidate["reservation"] = reservation
        plan["reservation"] = _copy(reservation)
        state["reviewed_item_ids"] = sorted(set(state["reviewed_item_ids"]) | set(ids))
        return _copy(reservation)


def record_validation(
    run_dir: Path, function: dict, result: dict, *, eligible_items: int,
    expected_plan_id: str | None = None,
) -> dict | None:
    name = function.get("name")
    with _ledger(run_dir, write=True) as data:
        state, plan = _locate(data, name)
        if expected_plan_id is not None and (plan is None or plan["plan_id"] != expected_plan_id):
            raise ValueError("Repair plan changed while validation was running")
        if plan is None:
            return None
        if plan["phase"] not in _ACTIVE:
            raise ValueError("Repair plan is no longer active; do not apply this validation")
        candidate = plan["candidates"].get(name)
        if candidate is None or candidate["after_version_hash"] != _version(function):
            raise ValueError("Repair validation does not match the submitted candidate")
        reservation = candidate["reservation"]
        if reservation is None or candidate["result"] is not None:
            raise ValueError("Validation must have one unconsumed reservation")
        rows = result.get("verdicts") or []
        ids = [row.get("item_id") for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(reservation["item_ids"]):
            raise ValueError("Validation verdicts do not match the reserved evidence IDs")
        labels = [row.get("verdict") for row in rows]
        if any(label not in {"yes", "no", "uncertain", "error", "processing_error"}
               for label in labels):
            raise ValueError("Unrecognized repair validation verdict")
        if not _current_plan_policy(plan):
            raise ValueError("Judge policy changed during repair validation")
        if rows and (result.get("claim_hash") != candidate["after_claim_hash"]
                     or result.get("judge_policy_hash") != plan["judge_policy_hash"]
                     or result.get("judge_behavior_strictness")
                     != plan.get("judge_behavior_strictness")):
            raise ValueError("Validation claim or policy provenance does not match the repair")
        if type(eligible_items) is not int or eligible_items < len(ids):
            raise ValueError("Eligible evidence count cannot be smaller than reviewed evidence")
        counts = {"n_yes": labels.count("yes"), "n_no": labels.count("no"),
                  "n_uncertain": labels.count("uncertain"),
                  "n_errors": labels.count("error") + labels.count("processing_error")}
        expected_counts = {**counts, "n_judged": len(ids),
                           "n_completed": counts["n_yes"] + counts["n_no"]}
        if any(key in result and result[key] != value for key, value in expected_counts.items()):
            raise ValueError("Validation summary counts disagree with item verdicts")
        required = result.get("required_items", len(ids))
        if type(required) is not int or required < 0:
            raise ValueError("Required evidence count must be a nonnegative integer")
        reviewed_counts = {"n_judged": len(ids), **counts}
        combined_rows = rows
        if plan["action"] == "retry_review" and rows:
            # The caller merges fresh error retries with unchanged prior semantic rows.
            # A successful retried subset must not conceal an existing NO/UNKNOWN.
            combined_rows = (function.get("scores") or {}).get("behavior_verdicts") or []
            combined = {row.get("item_id"): row for row in combined_rows}
            before_rows = (plan["before"].get("scores") or {}).get("behavior_verdicts") or []
            before = {row.get("item_id"): row for row in before_rows}
            if (len(combined) != len(combined_rows) or set(combined) != set(before)
                    or any(combined.get(row["item_id"]) != row for row in rows)
                    or any(combined[key] != row for key, row in before.items()
                           if key not in ids)):
                raise ValueError("Processing retry must retain prior semantic reviews unchanged")
            labels = [row.get("verdict") for row in combined_rows]
            counts = {"n_yes": labels.count("yes"), "n_no": labels.count("no"),
                      "n_uncertain": labels.count("uncertain"),
                      "n_errors": labels.count("error") + labels.count("processing_error")}
        current_sample = (function.get("scores") or {}).get("behavior_sample") or {}
        if (reservation.get("sample") or {}).get("reused_n") or current_sample.get("reused_n"):
            review_status = "reused_evidence"
        elif (not combined_rows or len(combined_rows) < required
              or (reservation.get("sample") or {}).get("limited") or current_sample.get("limited")):
            review_status = "insufficient_new_evidence"
        elif counts["n_errors"]:
            review_status = "processing_error"
        elif counts["n_uncertain"]:
            review_status = "uncertain"
        else:
            review_status = "complete"
        claim_preserved = all(function.get(key) == plan["before"].get(key)
                              for key in ("description", "mechanism"))
        outcome = {
            "review_status": review_status, "n_judged": len(combined_rows), **counts,
            "reviewed_counts": reviewed_counts,
            "eligible_items": eligible_items, "required_items": required,
            "claim_preserved": claim_preserved,
            "comparison": "same_claim_detector_repair" if claim_preserved
            else "new_claim_validation",
            "all_reviewed_confirmed": review_status == "complete" and bool(combined_rows)
            and counts["n_yes"] == len(combined_rows),
            "original_claim_validated": claim_preserved and plan["action"] != "retry_review"
            and review_status == "complete" and bool(rows)
            and counts["n_yes"] == len(rows),
            "sample_kind": reservation["sample_kind"],
            "automatic_promotion": False,
        }
        candidate.update(result=_copy(result), outcome=outcome,
                         after_validation=_copy(function), reviewed_at=_now(),
                         phase="completed" if review_status == "complete" else "deferred")
        plan["result"], plan["outcome"] = _copy(result), _copy(outcome)
        candidates = list(plan["candidates"].values())
        if plan["action"] == "split_detector":
            plan["result"] = {"candidate_results": {
                child: _copy(review["result"]) for child, review in plan["candidates"].items()
            }}
            plan["outcome"] = _split_outcome(plan)
        if plan["action"] == "split_detector" and (
            len(candidates) < 2 or any(c["phase"] == "awaiting_validation" for c in candidates)
        ):
            _touch(state, plan, "awaiting_validation")
        else:
            _touch(state, plan, "completed" if all(c["phase"] == "completed" for c in candidates)
                   else "deferred")
        return _copy(plan)


def defer_repair(run_dir: Path, name: str, reason: str) -> dict | None:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Deferring a repair requires a reason")
    with _ledger(run_dir, write=True) as data:
        state, plan = _locate(data, name)
        if plan is None or plan["phase"] not in _ACTIVE:
            return None
        plan["deferral_reason"] = reason.strip()
        _touch(state, plan, "deferred")
        return _copy(plan)


def record_retirement(run_dir: Path, name: str, reason: str, status: str) -> dict | None:
    if not (Path(run_dir) / "detector_repairs.json").exists():
        return None
    with _ledger(run_dir, write=True) as data:
        state, plan = _locate(data, name)
        if plan is None or plan["phase"] not in _ACTIVE:
            return None
        plan["retirement"] = {"name": name, "reason": reason, "status": status,
                              "recorded_at": _now()}
        plan["outcome"] = {"review_status": "retired", "automatic_promotion": False}
        _touch(state, plan, "completed" if plan["action"] == "retire" else "deferred")
        return _copy(plan)


def record_polishing_deferral(
    run_dir: Path, function: dict, *, reason: str, finding_ids: list[str],
    evidence_ids: list[str], measurement_hash: str, polishing_id: str | None,
) -> dict:
    """Record a grounded disposition without allocating another repair or judge."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A polishing deferral requires its concrete evidence limitation")
    name = function["name"]
    if not polishing_id:
        raise ValueError("Direct evidence-backed deferral is available during final polishing")
    with _ledger(run_dir, write=True) as data:
        _check_polishing_id(Path(run_dir), name, polishing_id)
        state, plan = _locate(data, name)
        if plan is not None and plan["phase"] in _ACTIVE:
            plan["deferral_reason"] = reason.strip()
            _touch(state, plan, "deferred")
        disposition = {
            "name": name, "claim_hash": claim_hash(**_claim(function)),
            "measurement_hash": measurement_hash, "reason": reason.strip(),
            "finding_ids": _ids(finding_ids), "evidence_ids": _ids(evidence_ids),
            "polishing_id": polishing_id,
        }
        data.setdefault("polishing_deferrals", {})[name] = disposition
        return _copy(disposition)
