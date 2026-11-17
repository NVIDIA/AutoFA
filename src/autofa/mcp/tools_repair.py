# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit detector repair plans grounded in the current judge feedback."""

from __future__ import annotations

from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.mcp.runtime import Runtime, err, get_runtime, ok
from autofa.metrics.repair_feedback import build_repair_feedback
from autofa.metrics.repair_tradeoff import compare_repair_coverage
from autofa.metrics.version_lessons import version_lessons
from autofa.session import repair_workflow
from autofa.session.polishing_guard import (
    active_polishing,
    polishing_repair_id,
    polishing_tool_guard,
)


def _round_refinement_candidate(rt: Runtime, function: dict[str, Any]) -> bool:
    """Allow feedback-driven repair of the exact version validated in this round."""
    from autofa.core.dependencies import review_source
    from autofa.metrics.claim_contract import claim_hash
    from autofa.session.round_validation import active_round_refinement

    refinement = active_round_refinement(rt)
    if refinement is None:
        return False
    name = function["name"]
    if rt.library.get(name) is None:
        return False
    polishing = active_polishing(rt)
    if polishing is not None and name in polishing["frozen_names"]:
        return True
    fingerprint = claim_hash(
        source=review_source(function), description=function.get("description") or "",
        mechanism=function.get("mechanism") or function.get("description") or name,
    )
    claims = refinement.get("evaluated_claims")
    return isinstance(claims, dict) and claims.get(name) == fingerprint


def review_next_action(
    rt: Runtime, function: dict[str, Any], *, evidence_n: int | None = None,
) -> dict[str, Any]:
    """Offer an executable prerequisite using saved workflow and evidence only."""
    try:
        return _review_next_action(rt, function, evidence_n=evidence_n)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"tool": "get_library_function", "arguments": {"name": function["name"]},
                "reason": f"Workflow evidence is unavailable: {exc}. Inspect the retained "
                          "candidate and restore its original workflow evidence before reviewing."}


def _saved_review(function: dict[str, Any], review_key: str, item_ids: list[str]) -> dict:
    """Summarize a completed operation's saved cohort; it may no longer be authoritative."""
    from autofa.metrics.review_assessments import (
        archived_assessments,
        assessment_summary,
        cohort_ids,
    )

    scores = function.get("scores") or {}
    for saved in (scores, *reversed(archived_assessments(scores))):
        operation = saved.get("behavior_review_operation") or {}
        if (cohort_ids(saved) == set(item_ids)
                and operation.get("review_key", review_key) == review_key):
            return {"status": "completed", "authoritative": saved is scores,
                    **assessment_summary(saved)}
    return {"status": "completed", "authoritative": False, "item_ids": sorted(item_ids)}


def _unfinished_review_request(
    rt: Runtime, function: dict[str, Any], settings: dict, plan_id: str | None,
) -> dict | None:
    """Recover the judge_function request that resumes the current unfinished cohort."""
    from autofa.mcp.tools_eval import _development_support_key, _review_key
    from autofa.metrics.review_assessments import cohort_ids
    from autofa.session.review_evidence import find_review_allocation

    scores = function.get("scores") or {}
    operation = scores.get("behavior_review_operation") or {}
    saved_key, phase = operation.get("review_key"), operation.get("phase")
    if (operation.get("scope") != "standard" or operation.get("origin_run_id") != rt.run.run_id
            or not isinstance(saved_key, str) or phase not in {"development", "confirmation"}):
        return None
    allocation = find_review_allocation(rt.run.run_dir(), function["name"],
                                        review_key=saved_key, phase=phase)
    if allocation is None or allocation["operation_completed"]:
        return None
    target = (scores.get("behavior_sample") or {}).get("target_n")
    largest = max(len(cohort_ids(scores)), target if type(target) is int else 0)
    # The saved key encodes the request size; find the request that reopens it.
    for evidence_n in (None, *range(1, largest + 1)):
        key = _review_key(function, settings, evidence_n, plan_id)
        if saved_key in {key, _development_support_key(key)}:
            return {"name": function["name"], "review_phase": phase,
                    **({"evidence_n": evidence_n} if evidence_n is not None else {})}
    return None


def review_recovery_action(
    rt: Runtime, function: dict[str, Any], *, plan_id: str | None = None,
) -> dict[str, Any] | None:
    """Recover the current assessment's processing failures, never by replaying a cohort."""
    from autofa.mcp.tools_eval import _review_settings
    from autofa.metrics.review_failures import review_failures

    failures = review_failures((function.get("scores") or {}).get("behavior_verdicts") or [])
    if not failures:
        return None
    name = function["name"]
    retryable = [row for row in failures
                 if row.get("retry_policy") != "requires_backend_or_configuration_change"]
    # Resuming the frozen request retries per-item errors. It reuses refusals and settled
    # compiler failures, which need retry_review or a configuration change instead.
    if any(row.get("stage") != "claim_compilation" for row in retryable):
        _, _, settings = _review_settings(rt.config)
        request = _unfinished_review_request(rt, function, settings, plan_id)
        if request is not None:
            return {"tool": "judge_function", "arguments": request,
                    "reason": "The current assessment has review processing failures. Resume "
                              "its own frozen review to retry them; completed judgments and "
                              "its cohort are kept."}
    problem = "The current assessment has review processing failures"
    recovery = ("plan retry_review for those failed stages; processing retries do not use the "
                "repair budget." if retryable else
                "follow their recorded recovery action; a repeated review reuses a backend "
                "refusal.")
    if plan_id is not None:
        return {"tool": "defer_function_repair",
                "arguments": {"name": name, "reason": problem + "."},
                "reason": f"{problem}. Defer this attempt, then {recovery}"}
    return {"tool": "get_function_repair" if retryable else "get_function_feedback",
            "arguments": {"name": name}, "reason": f"{problem}; {recovery}"}


def _review_next_action(rt: Runtime, function: dict[str, Any], *, evidence_n: int | None) -> dict:  # noqa: C901
    from autofa.mcp.tools_eval import _find_compatible_allocation, _review_settings
    from autofa.objectives import discovery_strategy
    from autofa.session.family_discovery import (
        _load,
        family_comparison_pending,
        family_progress_summary,
    )

    name = function["name"]
    feedback = build_repair_feedback(function)
    workflow = repair_workflow.get_repair(rt.run.run_dir(), name)
    budget = repair_workflow.repair_attempt_budget(
        workflow, polishing_id=polishing_repair_id(rt, name),
    )
    latest = workflow.get("latest") or {}
    _, _, settings = _review_settings(rt.config)
    request = (function.get("scores") or {}).get("ordinary_confirmation_request") or {}
    if not isinstance(request, dict):
        raise ValueError("Saved ordinary confirmation request is malformed")
    if request.get("origin_run_id") == rt.run.run_id:
        evidence_n = request.get("evidence_n")
    review_args = {"name": name, "review_phase": "confirmation"}
    if evidence_n is not None:
        review_args["evidence_n"] = evidence_n

    def action(tool, arguments, reason):
        return {"tool": tool, "arguments": arguments, "reason": reason}

    def investigation_action(investigation, context):
        specific = investigation.get("next_step")
        reason = " ".join(part for part in (
            specific, context, investigation["question"],
        ) if part)
        result = action(investigation["tool"], investigation["arguments"], reason)
        if investigation.get("action"):
            result["diagnostic_disposition"] = investigation["action"]
        if investigation.get("if_unavailable"):
            result["if_unavailable"] = investigation["if_unavailable"]
        return result

    if (feedback["recommended_action"] == "repair_implementation"
            and workflow.get("state") not in repair_workflow._ACTIVE):
        finding = next(row for row in feedback["findings"] if row["status"] == "execution_error")
        if not budget["remaining"] and not repair_workflow.implementation_repair_budget(
            workflow,
        )["remaining"]:
            return action("get_function_repair", {"name": name},
                          "The repair budget is exhausted. Preserve the execution-error evidence "
                          "and inspect saved versions or defer this candidate.")
        return action("get_item", {"item_id": finding["item_ids"][0]},
                      "Inspect the recorded execution error before planning repair_implementation "
                      "with its finding and item IDs. A supported claim does not block a code "
                      "repair; a separate two-attempt implementation allowance remains usable "
                      "after exploration attempts run out. Transient provider failures need an "
                      "evaluation retry instead.")

    try:
        confirmation_key, _ = _find_compatible_allocation(
            rt, function, settings, evidence_n, None, "confirmation",
        )
        context = repair_workflow.validation_context(
            rt.run.run_dir(), function, resume_reserved=True,
            allow_deferred_confirmation=True, objective_strategy=discovery_strategy(rt.config),
            behavior_strictness=strictness_for_config(rt.config),
            confirmation_review_key=confirmation_key,
            judge_settings=settings,
        )
    except (ValueError, OSError) as exc:
        investigation = next(iter(feedback["investigations"]), None)
        reason = f"{exc}. "
        scores = function.get("scores") or {}
        saved_settings = {key: scores.get(key) for key in settings}
        changed_settings = {key: saved_settings[key] for key, value in settings.items()
                            if key != "judge_objective_strategy" and saved_settings[key] != value}
        if changed_settings and feedback["recommended_action"] == "retain":
            reason += ("Restore the judge settings used for this supported candidate before "
                       f"confirmation: {changed_settings}. A new repair plan is not required.")
        elif workflow.get("state") == "exhausted" and not budget["remaining"]:
            reason += ("The repair attempt budget is exhausted. Inspect the saved findings; "
                       "retain this as a diagnostic or move to another candidate. "
                       "An unchanged judgment cannot reset the repair budget.")
        elif latest.get("phase") == "planned":
            return action("get_library_function", {"name": name}, reason +
                          "Inspect the frozen baseline, then submit the planned change with "
                          "its repair_plan_id before evaluation or review.")
        else:
            reason += ("Inspect the current findings before selecting an allowed repair action. "
                       "Do not repeat this judgment or infer support from historical scores.")
        return investigation_action(investigation, reason) if investigation \
            else action("get_function_feedback", {"name": name}, reason)

    if context and (context.get("reserved_item_ids") is not None
                    or context.get("retry_item_ids")):
        phase = (context.get("sample") or {}).get("phase") or "auto"
        return action("judge_function", {"name": name, "review_phase": phase},
                      "Resume the existing reserved review with its original evidence and budget.")

    if context:
        from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS

        completed_review = None
        pending = active_polishing(rt) is None and family_comparison_pending(rt.run.run_dir())
        for phase in ("development", "confirmation"):
            key, allocation = _find_compatible_allocation(
                rt, function, settings, evidence_n, context["plan_id"], phase,
            )
            if (phase == "development" and not pending and type(evidence_n) is int
                    and evidence_n >= MIN_GOOD_FAILED_TASKS
                    and (allocation is None or allocation["operation_completed"])):
                # As in judge_function, an explicit post-comparison expansion is its
                # own operation; offering it again must not replay a completed one.
                key, allocation = _find_compatible_allocation(
                    rt, function, settings, evidence_n, context["plan_id"], phase,
                    support=True,
                )
            if allocation and not allocation["operation_completed"]:
                return action("judge_function", {**review_args, "review_phase": phase},
                              "Resume the existing allocated review with its original evidence "
                              "and budget before changing or deferring this candidate.")
            if allocation and phase == "development":
                completed_review = _saved_review(function, key, allocation["item_ids"])
        from autofa.metrics.review_scheduling import current_review_schedule

        schedule = current_review_schedule(function.get("scores"))
        if schedule is not None:
            return action("defer_function_repair", {
                "name": name, "reason": schedule["reason"],
            }, "This unreserved repair is below the automatic review requirements. "
               "Defer the attempt before making another change. Follow the measured grade "
               "gaps and per-case findings: improve coverage, normalized precision, adherence "
               "or distinct-task support as needed. Explicit diagnostic review or deliberate "
               "useful-OK validation remains available; no adherence result was inferred.")
        if (feedback["recommended_action"] != "repair_implementation"
                and completed_review is not None):
            # Family development reviews leave the plan awaiting fresh confirmation,
            # which records its outcome. Repeating that review only returns it.
            saved = ("The active repair's development review already completed; repeating it "
                     "returns the saved completed_review and adds no evidence. ")
            unavailable = (((function.get("scores") or {}).get("confirmation_availability")
                            or {}).get("status") == "confirmation_unavailable")
            result = review_recovery_action(rt, function, plan_id=context["plan_id"])
            if result is not None:
                result["reason"] = saved + result["reason"]
            elif feedback["recommended_action"] == "retain" and not (pending or unavailable):
                result = action("judge_function", review_args, saved +
                                "Request independent confirmation on fresh tasks to record this "
                                "repair's outcome, or defer it to keep the development result.")
            elif feedback["recommended_action"] == "retain":
                result = action("defer_function_repair", {
                    "name": name,
                    "reason": ("Fresh confirmation waits for the family comparison"
                               if pending else "No unexposed tasks remain for confirmation")
                              + "; the supported development review is retained.",
                }, saved + "Defer this attempt; the detector and its judgments are preserved, "
                   "and a deferred supported candidate can request ordinary confirmation later.")
            else:
                result = action("defer_function_repair", {
                    "name": name,
                    "reason": "Current review findings do not establish full support for "
                              "this candidate.",
                }, saved + "Defer this attempt, then " + (
                    "plan a grounded repair from the current findings within the remaining "
                    "budget." if budget["remaining"] else
                    "retain it as a diagnostic or move to another candidate; the repair "
                    "budget is exhausted."))
            result["completed_review"] = completed_review
            return result
        if feedback["recommended_action"] != "repair_implementation":
            return action(
                "judge_function",
                {"name": name, "review_phase": "auto",
                 **({"evidence_n": evidence_n} if evidence_n is not None else {})},
                "Complete the active repair's review of this frozen candidate before "
                "starting another plan. Do not change or duplicate the submitted candidate; "
                "use defer_function_repair first if abandoning this attempt.",
            )
        return action("defer_function_repair", {
            "name": name,
            "reason": "The submitted candidate still has recorded execution errors; inspect "
                      "them before a new bounded implementation repair.",
        }, "This unreserved repair attempt needs an implementation fix before another review. "
           "Defer it, then use the current execution findings for a new plan within the "
           "remaining repair budget.")

    from autofa.metrics.review_scheduling import current_review_schedule

    schedule = current_review_schedule(function.get("scores"))
    if schedule is not None:
        investigation = next(iter(feedback["investigations"]), None)
        if investigation:
            return investigation_action(investigation, schedule["reason"])
        return action(
            "get_function_feedback", {"name": name},
            " ".join([schedule["reason"], *schedule["next_steps"]])
            + " A family can be assessed using the saved evaluation and scheduling decision; "
              "missing adherence does not require an automatic judge call.",
        )

    family = family_progress_summary(rt.run.run_dir())
    if (family and family["phase"] != "specialize"
            and not _round_refinement_candidate(rt, function)):
        if family["phase"] == "survey":
            seeds = family.get("synthesis_seed_functions") or []
            if seeds:
                return action("get_function_feedback", {"name": seeds[0]},
                              "Inspect the scheduled seeds' supported cases before recording "
                              "a coherent synthesis or explaining incompatible causes.")
            return action("sample_items", {"coverage": "unconfirmed_tasks", "distinct_tasks": True},
                          "Family comparison still needs a checked survey. Inspect distinct tasks, "
                          "then record_family or no_coherent_family with their actual evidence.")
        family_name = family.get("family_name")
        detector_name = family.get("detector_name") or family_name
        candidate = rt.library.get(detector_name)
        if candidate is None:
            return action("family_discovery", {"action": "status"},
                          "Retrieve the recorded family and its detector_name. If an existing "
                          "detector implements this claim, link it by passing detector_name "
                          "to assess_family; do not duplicate it just to match the family name. "
                          "Otherwise implement the recorded claim before assessment.")
        survey = (_load(rt.run.run_dir()) or {}).get("survey") or {}
        if any(key in survey and (getattr(candidate, key) or "").strip() != survey[key].strip()
               for key in ("description", "mechanism")):
            return action("family_discovery", {"action": "status"},
                          "The current claim differs from the recorded family. Retrieve its "
                          "survey, then record_family with the revised claim and checked evidence "
                          "using the same family_name and detector_name before assess_family; "
                          "another judgment does not update the survey.")
        from autofa.core.dependencies import review_source
        from autofa.metrics.claim_contract import claim_hash

        scores = candidate.scores or {}
        fingerprint = claim_hash(source=review_source(candidate),
                                 description=candidate.description or "",
                                 mechanism=candidate.mechanism or candidate.description
                                 or candidate.name)
        if (not scores.get("evaluated") or scores.get("evaluation_stale")
                or scores.get("evaluation_claim_hash") != fingerprint
                or scores.get("evaluation_run_id") != rt.run.run_id):
            return action("evaluate_functions", {"names": [detector_name]},
                          "The recorded family needs a current evaluation before assessment.")
        family_feedback = build_repair_feedback(candidate.to_dict())
        if (family_feedback["validation"]["status"] == "unjudged"
                and detector_name == name):
            return action("judge_function", {"name": name, "review_phase": "development"},
                          "Review the recorded family's first-pass matches within the existing "
                          "development budget before assessing it.")
        return action("get_function_feedback", {"name": detector_name},
                      "Inspect the saved evaluation and claim feedback, then call "
                      "family_discovery(action='assess_family', family_name="
                      f"{family_name!r}, detector_name={detector_name!r}, "
                      "assessment=your evidence-based assessment). "
                      "Do not repeat completed development review to satisfy this gate.")

    unavailable = (function.get("scores") or {}).get("confirmation_availability") or {}
    if unavailable.get("status") == "confirmation_unavailable":
        return action("get_function_feedback", {"name": name},
                      "No unexposed task evidence was available. Retain the diagnostic findings "
                      "and work on another candidate, or collect new source tasks before retrying "
                      "the original request. Repeating it now cannot establish freshness.")
    if (latest.get("phase") in {"deferred", "exhausted"}
            and (function.get("scores") or {}).get("behavior_review_phase") == "confirmation"):
        return action("judge_function", review_args,
                      "Resume or retrieve the same ordinary confirmation operation with its "
                      "frozen task cohort and original per-item review budget.")
    if not context and feedback["investigations"]:
        investigation = feedback["investigations"][0]
        return investigation_action(
            investigation,
            "Inspect this current finding before selecting a grounded repair action. "
            "Repeating the unchanged judgment does not resolve its evidence gap.",
        )
    phase = "confirmation" if feedback["recommended_action"] == "retain" else "development"
    return action("judge_function", review_args if phase == "confirmation"
                  else {"name": name, "review_phase": phase},
                  "The frozen candidate can request independent confirmation; prior development "
                  "support and the deferred repair remain unchanged." if phase == "confirmation"
                  else "Review the current candidate before choosing a grounded repair action.")


def repair_rejection_next_action(
    rt: Runtime, function: dict[str, Any], rejection: str,
) -> dict[str, Any]:
    """Resolve the rejected repair request before proposing unrelated family work."""
    name = function["name"]
    try:
        workflow = repair_workflow.get_repair(rt.run.run_dir(), name)
        if workflow.get("state") in repair_workflow._ACTIVE:
            latest = workflow.get("latest") or {}
            action = review_next_action(rt, function)
            action["reason"] = (
                f"Repair request rejected: {rejection}. The active plan "
                f"{latest.get('plan_id')} must be completed or explicitly deferred first. "
                + action["reason"]
            )
            action["alternative"] = {
                "tool": "defer_function_repair",
                "arguments": {
                    "name": workflow.get("name") or name,
                    "reason": "Defer this attempt to inspect current findings and choose "
                              "a different repair.",
                },
                "reason": "Use this only if abandoning the active attempt; then retrieve "
                          "current findings before planning another bounded repair.",
            }
            return action
        feedback = build_repair_feedback(function)
        if (feedback["recommended_action"] == "retain"
                and rejection == "Selected finding status does not support the repair action"):
            return review_next_action(rt, function)
        return {
            "tool": "get_function_repair", "arguments": {"name": name},
            "reason": f"Repair request rejected: {rejection}. Retrieve the current findings, "
                      "allowed actions and repair history. Use finding_ids from those findings "
                      "and evidence_ids belonging to the selected findings; do not retry the "
                      "unchanged request or change the candidate to resolve a reference error.",
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "tool": "get_library_function", "arguments": {"name": name},
            "reason": f"Repair workflow evidence is unavailable: {exc}. Restore access to "
                      "the saved repair before changing the candidate or starting another plan.",
        }


def compact_plan(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep immutable full snapshots on disk without duplicating them in every reply."""
    if not plan:
        return None
    result = {key: plan.get(key) for key in (
        "plan_id", "name", "attempt", "action", "phase", "rationale", "finding_ids",
        "evidence_ids", "before_claim_hash", "before_version_hash", "after_claim_hash",
        "after_version_hash", "reservation", "outcome", "deferral_reason", "retirement",
        "objective_settings", "polishing_id", "attempt_budget",
        "implementation_allowance",
    )}
    result["candidates"] = {
        name: {key: candidate.get(key) for key in (
            "phase", "after_claim_hash", "after_version_hash", "reservation", "outcome",
        )}
        for name, candidate in (plan.get("candidates") or {}).items()
    }
    if diagnostic := plan.get("implementation_diagnostics"):
        result["implementation_diagnostics"] = {
            "version_hash": diagnostic["version_hash"], "item_ids": diagnostic["item_ids"],
            "interpretation": diagnostic["interpretation"],
        }
    if check := plan.get("implementation_check"):
        result["implementation_check"] = {
            "version_hash": check["version_hash"], "passed": check["passed"],
        }
    return result


def compact_workflow(workflow: dict[str, Any] | None) -> dict[str, Any] | None:
    if not workflow:
        return None
    return {
        **{key: workflow.get(key) for key in (
            "name", "state", "status", "baseline_claim_hash", "max_attempts",
            "attempts_used", "reviewed_item_ids", "historical_link", "requested_name",
            "linked_plan_id", "attempt_budget",
            "implementation_budget",
        )},
        "latest": compact_plan(workflow.get("latest")),
        "history": [compact_plan(plan) for plan in workflow.get("history", [])],
    }


def get_function_repair(name: str) -> dict[str, Any]:
    """Get grounded findings, concrete investigation calls, and bounded repair history."""
    rt = get_runtime()
    fn = rt.library.get(name)
    workflow = repair_workflow.get_repair(rt.run.run_dir(), name)
    if fn is None and not workflow:
        return err(f"unknown function: {name}")
    latest = (workflow or {}).get("latest") or {}
    before = latest.get("before")
    comparison = None
    if fn and isinstance(before, dict):
        from autofa.metrics.library_coverage import failure_cohort

        try:
            comparison = compare_repair_coverage(
                before, fn.to_dict(), config=rt.config,
                objective_settings=latest.get("objective_settings"),
                failed_item_ids=failure_cohort(rt.dataset)["failed_item_ids"],
                run_id=rt.run.run_id,
            )
        except Exception as exc:  # noqa: BLE001
            comparison = {
                "comparison_complete": False, "automatic_promotion": False,
                "reason": f"Coverage comparison unavailable: {exc}",
            }
    try:
        lessons = version_lessons(rt, name, fn.to_dict() if fn else None)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        lessons = {
            "lessons": [], "available": False, "reason": f"Historical lessons unavailable: {exc}",
            "history_path": str(rt.run.run_dir() / "detector_repairs.json"),
        }
    return ok(
        name=name,
        feedback=build_repair_feedback(fn.to_dict()) if fn else None,
        workflow=compact_workflow(workflow),
        repair_budget=repair_workflow.repair_attempt_budget(
            workflow, polishing_id=polishing_repair_id(rt, name),
        ),
        implementation_budget=repair_workflow.implementation_repair_budget(workflow),
        implementation_recovery=(
            "Parsing, extraction or linking defects can use repair_implementation even when "
            "the claim passed review. For a nonthrowing defect, supply implementation_checks "
            "mapping inspected item IDs to expected outputs, finding_ids=[], and explain the "
            "concrete defective operation. The runtime reproduces the bug and tests repairs "
            "before replacing the baseline. The two extra attempts cannot change the claim; "
            "these author controls do not establish causal validity."),
        coverage_comparison=comparison,
        retained_lessons=lessons,
        history_path=str(rt.run.run_dir() / "detector_repairs.json"),
        next_action=review_next_action(rt, fn.to_dict()) if fn else None,
        version_tools={"list": {"name": "list_function_versions", "arguments": {"name": name}},
                       "restore": "restore_function_version"},
        message=("Inspect coverage_comparison and choose a grounded repair action before "
                 "changing the detector or claim. list_function_versions(name) compares "
                 "archived versions; restoration requires an explicit version and reason."),
    )


def plan_function_repair(
    name: str, action: str, rationale: str,
    finding_ids: list[str], evidence_ids: list[str],
    implementation_checks: dict[str, bool | int | float | str | None] | None = None,
) -> dict[str, Any]:
    """Record tighten_detector, refine_detector, revise_claim, split_detector, gather_evidence,
    repair_implementation, retry_review, or retire, citing findings and item IDs
    from get_function_repair. repair_implementation needs recorded execution errors
    and preserves the description, mechanism and detector relationships.
    refine_detector uses measured quality-gap findings for a currently supported
    claim, preserving that claim and its relationships while testing its predicate.

    Plans retain the original claim and reviews. The repair-plan workflow allows
    at most three attempts during discovery; each runtime-assigned final polishing
    round exposes a separate three-plan allowance while retaining the complete
    history. New validation examples are reserved separately from recorded repair examples.
    After the exploration allowance, two additional implementation repairs per
    detector are available. For parsing, extraction or linking defects that do not
    throw an exception, provide implementation_checks={item_id: expected_output},
    finding_ids=[], and describe the exact defective operation in rationale.
    The runtime reproduces the mismatch on the current version, preserves supported
    controls, and rejects a proposed replacement if regression checks fail. These
    author controls do not establish causal validity or authorize a changed mechanism.
    Processing retries preserve the current version and do not consume that budget.
    A plan proposes work; it does not establish that a detector has improved.
    """
    rt = get_runtime()
    if blocked := polishing_tool_guard(
        rt, "plan_function_repair", {"name": name, "action": action},
    ):
        return blocked
    fn = rt.library.get(name)
    if fn is None:
        return err(f"unknown function: {name}")
    if action in {"tighten_detector", "refine_detector", "revise_claim", "split_detector"}:
        from autofa.session.family_discovery import (
            STATE_FILE,
            family_progress_summary,
            initialize_family_discovery,
        )

        workflow = repair_workflow.get_repair(rt.run.run_dir(), name)
        # Let the existing workflow explain an active-plan conflict. Scheduling
        # must not strand work already authorized by a saved repair plan.
        active = workflow and workflow.get("state") in repair_workflow._ACTIVE
        if not active and (rt.run.run_dir() / STATE_FILE).exists():
            state = initialize_family_discovery(rt)
            if (state.get("synthesis_required")
                    and not _round_refinement_candidate(rt, fn.to_dict())):
                progress = family_progress_summary(rt.run.run_dir())
                return err(
                    "Inspect supported seeds before starting another detector repair. "
                    + progress["next_action"], family_discovery=progress,
                    next_action=review_next_action(rt, fn.to_dict()),
                )
    try:
        from autofa.objectives import discovery_strategy, snapshot_objectives

        diagnostics = None
        if implementation_checks is not None and action != "repair_implementation":
            raise ValueError("implementation_checks only apply to repair_implementation")
        if action == "repair_implementation":
            workflow = repair_workflow.get_repair(rt.run.run_dir(), name)
            budget = repair_workflow.repair_attempt_budget(
                workflow, polishing_id=polishing_repair_id(rt, name))
            if implementation_checks is not None or not budget["remaining"]:
                if workflow.get("state") in repair_workflow._ACTIVE:
                    raise ValueError("Finish or defer the active repair before another plan")
                extra = repair_workflow.implementation_repair_budget(workflow)
                if not budget["remaining"] and not extra["remaining"]:
                    raise ValueError("Implementation recovery allowance exhausted")
                from autofa.mcp.implementation_recovery import diagnose_implementation

                diagnostics = diagnose_implementation(
                    rt, fn.to_dict(), evidence_ids, implementation_checks)
        plan = repair_workflow.plan_repair(
            rt.run.run_dir(), fn.to_dict(), build_repair_feedback(fn.to_dict()),
            action=action, rationale=rationale, finding_ids=finding_ids,
            evidence_ids=evidence_ids,
            objective_strategy=discovery_strategy(rt.config),
            behavior_strictness=strictness_for_config(rt.config),
            objective_settings=snapshot_objectives(rt.config),
            polishing_id=polishing_repair_id(rt, name),
            implementation_diagnostics=diagnostics,
        )
    except ValueError as exc:
        return err(str(exc), next_action=repair_rejection_next_action(rt, fn.to_dict(), str(exc)))
    return ok(
        name=name, plan=compact_plan(plan),
        message=("Plan recorded. Apply the proposed change with submit_function and evaluate it. "
                 "Follow the returned review schedule: resume existing reservations, otherwise "
                 "use discovery_feedback.grade_gap and per-case findings to address the actual "
                 "blocking dimensions (coverage, normalized precision, adherence or distinct-task "
                 "support). Request diagnostic or standard review when it answers the identified "
                 "question. For split children pass repair_plan_id. "
                 "Evidence gathering and processing retries may keep the current version. "
                 "Retirement uses remove_function with an explicit reason. "
                 "If evidence cannot be obtained, use defer_function_repair."),
    )


def defer_function_repair(
    name: str, reason: str, finding_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Park an unresolved attempt without rejecting the failure mode.

    During final polishing, finding_ids and evidence_ids from
    get_function_feedback.recovery_findings can record a current limitation without
    creating another repair plan or spending a repair/judge allowance. Explain the
    failed experiment or missing evidence in reason; this is not a score ceiling
    inferred from a few selected cases.
    """
    rt = get_runtime()
    if blocked := polishing_tool_guard(rt, "defer_function_repair", {"name": name}):
        return blocked
    try:
        if finding_ids is not None or evidence_ids is not None:
            from autofa.session.polishing_progress import record_limitation

            fn = rt.library.get(name)
            if fn is None:
                return err(f"unknown function: {name}")
            limitation = record_limitation(rt, fn, finding_ids=finding_ids or [],
                                          evidence_ids=evidence_ids or [], reason=reason)
            return ok(name=name, limitation=limitation,
                      message="Current evidence-backed polishing limitation recorded; no new "
                              "repair attempt or review was allocated.")
        plan = repair_workflow.defer_repair(rt.run.run_dir(), name, reason)
    except ValueError as exc:
        return err(str(exc))
    return ok(name=name, plan=compact_plan(plan),
              message="Repair deferred. The detector and prior judgments are preserved.")
