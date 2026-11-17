# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate current Good LLM detectors before the next agent turn or completion."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.discovery_grades import grade_summary
from autofa.objectives import discovery_strategy
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import (
    Cancellation,
    cancellation_guard,
    cancellation_scope,
    check_cancelled,
)

TURN_VALIDATION_FRACTION = .8
_GOOD = {"GOOD", "EXCELLENT", "OUTSTANDING"}


def _claim(fn: Any) -> str:
    return claim_hash(source=review_source(fn), description=fn.description or "",
                      mechanism=fn.mechanism or fn.description or fn.name)


def _receipt(rt: Any) -> dict:
    try:
        saved = json.loads((rt.run.run_dir() / "turn_validation.json").read_text())
    except (OSError, ValueError):
        return {}
    return (saved if isinstance(saved, dict) and saved.get("version") == 1
            and saved.get("run_id") == rt.run.run_id else {})


def _review_context(rt: Any) -> str:
    from autofa.mcp.tools_eval import _review_settings
    from autofa.metrics.claude_judge import judge_policy_hash
    from autofa.metrics.judge_checkpoints import fingerprint

    _, _, settings = _review_settings(rt.config)
    return fingerprint({"settings": settings, "policy": judge_policy_hash(
        behavior_strictness=strictness_for_config(rt.config))})


def _review_state(fn: Any) -> str:
    """Notice an explicit review/recovery without using incidental display changes."""
    from autofa.metrics.judge_checkpoints import fingerprint

    scores = fn.scores or {}
    return fingerprint({key: scores.get(key) for key in (
        "round_review", "behavior_verdicts", "behavior_n_errors", "behavior_n_review_incomplete",
        "behavior_evidence_stale", "behavior_judgment_stale", "judge_policy_hash",
    )})


def _can_retry_review(fn: Any) -> bool:
    """Allow one stage recovery, never reroll a refusal or impossible review workload."""
    from autofa.metrics.review_failures import requires_review_reconfiguration, review_failures

    scores = fn.scores or {}
    receipt = scores.get("round_review") or {}
    if (receipt.get("retry_generation", 0) >= 1
            or (receipt.get("recovery") or {}).get("status") == "blocked"):
        return False
    failures = scores.get("behavior_processing_failures") or review_failures(
        scores.get("behavior_verdicts") or [])
    blocked = {item_id for row in failures
               if requires_review_reconfiguration(row)
               for item_id in row.get("item_ids", [row.get("item_id")])}
    unfinished_sibling = any(
        row.get("completion_status") == "needs_evidence_review"
        and row.get("item_id") not in blocked
        for row in scores.get("behavior_verdicts") or []
    )
    # A refused stage must not suppress the permitted retry of a different
    # malformed/incomplete stage. final_review excludes the refused item IDs.
    return (not failures or unfinished_sibling
            or any(not requires_review_reconfiguration(row) for row in failures))


def _blocked_message(blocked: dict) -> str:
    return (
        "TURN-END REVIEW BLOCKED — completed checkpoint scoring and successful judgments "
        "are saved. These are review-processing or evidence-availability problems, not "
        "detector rejections. Completed confirmations and rejections remain valid; address "
        "substantive counterexamples separately. Inspect get_function_feedback for the examples "
        "and the recovery action below. Do not rewrite a detector merely to avoid a backend "
        "refusal or repeatedly request the same blocked review. Unchanged blockers will "
        "not trigger automatic retries. A blocked review cannot establish final qualification. "
        "Continue other authorized discovery or refinement work; the runtime still owns "
        "the stopping decision.\n" + json.dumps({
            name: {key: row.get(key) for key in ("reason", "processing_failures")}
            for name, row in blocked.items()
        }, default=str)
    )


def _settled_review_uncertainty(fn: Any) -> bool:
    """An inconclusive completed review differs from unfinished processing."""
    from autofa.metrics.final_validation_selection import _all_predictions_abstained

    scores = fn.scores or {}
    receipt = scores.get("round_review") or {}
    if (receipt.get("scope") != "round_checkpoint" or receipt.get("claim_hash") != _claim(fn)
            or receipt.get("evidence_unavailable") or receipt.get("status") != "unavailable"):
        return False
    return (_all_predictions_abstained(scores)
            or (bool(receipt.get("identity")) and scores.get("behavior_n_uncertain", 0) > 0
                and not scores.get("behavior_n_errors")
                and not scores.get("behavior_n_review_incomplete")
                and not scores.get("behavior_evidence_stale")))


def _score_and_review(rt, selected, receipt, *, retry_review: bool, console: Console | None):
    from autofa.mcp.tools_eval import (
        _evaluation_context,
        _reusable_evaluation,
        evaluate_round_functions,
    )
    from autofa.metrics.final_review import final_review_functions
    from autofa.ui.evaluation_progress import evaluation_progress

    names = [fn.name for fn in selected]
    path = rt.run.run_dir() / "turn_validation.json"
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"), BarColumn(),
        MofNCompleteColumn(), TimeElapsedColumn(), TextColumn("{task.fields[detector]}"),
        console=console, refresh_per_second=4,
    ) as progress:
        task = progress.add_task("Turn-end validation 80%: scoring", total=None, detector="")

        def update(event):
            progress.update(task, completed=event["completed"], total=event["total"],
                            detector=event.get("detector", ""))

        with evaluation_progress(update):
            scored = evaluate_round_functions(names=names, fraction=TURN_VALIDATION_FRACTION)
        if not scored.get("ok"):
            raise RuntimeError("Turn-end 80% validation is incomplete; its selection is saved "
                               "for Resume. " + str(scored.get("error") or "Scoring failed."))
        incomplete = [fn.name for fn in selected
                      if not _reusable_evaluation(rt, fn, fraction=TURN_VALIDATION_FRACTION)]
        if incomplete:
            raise RuntimeError(
                "Turn-end 80% validation did not finish its current scoring cohort for "
                + ", ".join(incomplete) + ". Its selection is saved for Resume; inspect "
                "get_function_feedback for execution errors or missing source evidence.")
        with cancellation_guard():
            receipt["phase"] = "review"
            receipt["evaluation_context"] = _evaluation_context(rt)
            atomic_write_json(path, receipt)
        progress.update(task, description="Turn-end validation 80%: reviewing",
                        completed=0, total=len(names))
        review = final_review_functions(
            names=names, scope="round_checkpoint", progress=update,
            **({"retry_incomplete": True} if retry_review else {}),
        )
    return scored, review


def validate_turn(rt: Any, *, pending_only: bool = False, console: Console | None = None) -> dict:
    """Score and review one frozen selection without advancing a discovery round.

    Persist selection before scoring: a cancelled scan may already have lowered
    a detector's grade, but its same-claim review still belongs to this checkpoint.
    Existing evaluation receipts reuse current >=80% scans and cached predictions.
    """
    from autofa.mcp.tools_eval import (
        _evaluation_context,
        _evaluation_parent_identity,
        _reusable_evaluation,
        evaluation_uses_llm,
    )
    from autofa.session.operation_admission import admit_operation
    from autofa.session.round_validation import refresh_validated_target, validation_feedback
    from autofa.session.status_log import emit_runtime_status

    saved = _receipt(rt)
    if pending_only and saved.get("status") not in {"pending", "review_blocked"}:
        return {"ok": True, "status": "not_required"}
    with admit_operation("advance_round", {}) as busy:
        if busy:
            raise RuntimeError("Turn-end 80% validation is waiting for active detector work: "
                               + busy["error"])
        check_cancelled()
        summary = grade_summary(
            rt.library.functions, behavior_strictness=strictness_for_config(rt.config),
            review_history=rt.library.review_history,
            objective_strategy=discovery_strategy(rt.config),
        )
        grades = {row["name"]: row["code"] for row in summary["functions"]}
        context = _evaluation_context(rt) if rt.library.functions else None
        review_context = _review_context(rt) if rt.library.functions else None
        pending = saved.get("claims", {}) if saved.get("status") == "pending" else {}
        # Keep blockers per detector: validating a new keeper must not erase a
        # different detector's saved recovery state or make it retry each turn.
        blocked = {fn.name: row for fn in rt.library.functions
                   if (row := saved.get("blocked_reviews", {}).get(fn.name))
                   and row.get("claim") == _claim(fn)
                   and row.get("parent") == _evaluation_parent_identity(rt, fn)}
        selected, reused, resuming = [], [], set()
        for fn in rt.library.functions:
            unfinished = (fn.name in pending and pending[fn.name] == _claim(fn)
                          and saved.get("parents", {}).get(fn.name)
                          == _evaluation_parent_identity(rt, fn))
            blocker = blocked.get(fn.name)
            if blocker:
                changed = (blocker.get("review_context") != review_context
                           or blocker.get("evaluation_context") != context
                           or blocker.get("review_state") != _review_state(fn))
                unfinished = (unfinished or changed
                              or pending_only and _can_retry_review(fn))
                if not unfinished:
                    continue
            if not unfinished and (pending_only or grades[fn.name] not in _GOOD):
                continue
            if not evaluation_uses_llm(fn, runtime=rt):
                continue
            if not unfinished and _reusable_evaluation(rt, fn, fraction=TURN_VALIDATION_FRACTION):
                reused.append(fn.name)
                continue
            selected.append(fn)
            if unfinished:
                resuming.add(fn.name)
        if not selected:
            if (saved.get("status") == "pending"
                    or saved.get("status") == "review_blocked"
                    and saved.get("blocked_reviews") != blocked):
                with cancellation_guard():
                    atomic_write_json(rt.run.run_dir() / "turn_validation.json",
                                      {**saved, "status": "review_blocked" if blocked
                                       else "superseded", "blocked_reviews": blocked,
                                       "phase": "review" if blocked else "superseded"})
            if pending_only and blocked:
                return {"ok": False, "status": "review_blocked", "blocked_reviews": blocked,
                        "message": _blocked_message(blocked), "reused_names": reused}
            return {"ok": True, "status": "not_required", "reused_names": reused}
        previous = {row["name"]: row for row in saved.get("before", {}).get("functions", [])}
        before = {"functions": [
            blocked.get(fn.name, {}).get("before")
            or previous.get(fn.name, {"name": fn.name, "code": grades[fn.name]})
            if fn.name in resuming else {"name": fn.name, "code": grades[fn.name]}
            for fn in selected
        ]}
        names = [fn.name for fn in selected]
        receipt = {"version": 1, "run_id": rt.run.run_id, "round": rt.run.round,
                   "fraction": TURN_VALIDATION_FRACTION, "status": "pending",
                   "phase": "scoring", "names": names,
                   "claims": {fn.name: _claim(fn) for fn in selected},
                   "parents": {fn.name: _evaluation_parent_identity(rt, fn) for fn in selected},
                   "evaluation_context": context, "before": before,
                   "blocked_reviews": blocked}
        path = rt.run.run_dir() / "turn_validation.json"
        with cancellation_guard():
            atomic_write_json(path, receipt)
        emit_runtime_status(
            f"Turn-end validation: checking {len(names)} Good-or-better LLM detectors on 80% "
            "of the dataset; current larger scans and completed predictions are reused.",
            source="turn_validation", context={"names": names, "fraction": .8},
        )
        scored, review = _score_and_review(
            rt, selected, receipt, console=console,
            retry_review=bool(resuming and (saved.get("phase") == "review" or blocked)))
        check_cancelled()
        # The first scan can resolve endpoint capabilities and refine its context.
        # Freeze the completed scoring identity, not the pre-scan estimate.
        context = receipt["evaluation_context"]
        unresolved = {row["name"] for row in review.get("unresolved", [])}
        unfinished = [fn.name for fn in selected
                      if fn.name in unresolved and not _settled_review_uncertainty(fn)]
        if not review.get("ok") and not unresolved:
            unfinished = names  # An unclassified failure is not a completed review.
        reasons = {row["name"]: row.get("reason") for row in review.get("unresolved", [])}
        original = {row["name"]: row for row in before["functions"]}
        for fn in selected:
            blocked.pop(fn.name, None)
            if fn.name in unfinished:
                scores = fn.scores or {}
                blocked[fn.name] = {
                    "claim": _claim(fn), "parent": _evaluation_parent_identity(rt, fn),
                    "evaluation_context": context, "review_context": review_context,
                    "review_state": _review_state(fn), "before": original[fn.name],
                    "reason": reasons.get(fn.name) or review.get("error")
                    or "Review did not establish a completed result; inspect saved diagnostics.",
                    "processing_failures": scores.get("behavior_processing_failures") or [],
                }
        feedback = validation_feedback(rt, before)
        feedback["detectors"] = [row for row in feedback["detectors"] if row["name"] in names]
        message = (
            "TURN-END 80% VALIDATION — current Good-or-better LLM detectors were checked "
            "on the larger representative dataset panel. These results replace their "
            "provisional scores. Inspect each qualification_gap and get_function_feedback "
            "for the examples behind downgrades; repair the blocking dimensions or retain "
            "a supported version. The fixed discovery comparison panel is unchanged. "
            "This checkpoint does not advance the round or establish full-dataset completion. "
            "Continue the current discovery or refinement work under the runtime policy.\n"
            + json.dumps({"detectors": feedback["detectors"],
                          "review_unresolved": review.get("unresolved", []),
                          "review_deferred": review.get("deferred", [])}, default=str)
        )
        if blocked:
            message += "\n\n" + _blocked_message(blocked)
        with cancellation_guard():
            refresh_validated_target(rt)
            receipt.update(status="review_blocked" if blocked else "complete",
                           phase="review" if blocked else "complete",
                           ok=not blocked, blocked_reviews=blocked,
                           evaluated_names=scored.get("evaluated_names", []),
                           reused_names=scored.get("reused_names", []),
                           review_unresolved=review.get("unresolved", []),
                           feedback=feedback, message=message)
            atomic_write_json(path, receipt)
        emit_runtime_status(
            ("Turn-end 80% scoring saved; review blocked for " + ", ".join(blocked)
             + ". Recovery findings are returning to the agent." if blocked else
             "Turn-end 80% validation complete; findings are returning to the agent."),
            source="turn_validation", context={"names": names, "fraction": .8},
        )
        return receipt


async def turn_validation_prompt(
    rt: Any, *, pending_only: bool = False, console: Console | None = None,
) -> str | None:
    """Keep checkpoint work cancellable and deliver its findings before continuation."""
    from autofa.session.supervision import check_lifecycle, stop_requested, supervisor_cancelled

    check_lifecycle(rt.run.run_dir(), phase="validation")
    cancellation = Cancellation(
        lambda: stop_requested(rt.run.run_dir()))
    with cancellation_scope(cancellation):
        worker = asyncio.create_task(
            asyncio.to_thread(validate_turn, rt, pending_only=pending_only, console=console))
    try:
        result = await asyncio.shield(worker)
    except asyncio.CancelledError:
        if supervisor_cancelled(cancellation):
            check_lifecycle(rt.run.run_dir(), phase="validation")
        cancelling = asyncio.create_task(asyncio.to_thread(cancellation.cancel))
        draining = asyncio.gather(cancelling, worker, return_exceptions=True)
        while not draining.done():
            try:
                await asyncio.shield(draining)
            except asyncio.CancelledError:
                continue
        draining.result()
        raise
    return result.get("message")
