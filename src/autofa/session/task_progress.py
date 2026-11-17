# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist task-coverage progress without changing detector ranking metrics."""

from __future__ import annotations

from typing import Any

from autofa.metrics.final_validation_selection import current_final_validation_skip
from autofa.session.research_status import read_research_status, write_research_status
from autofa.session.task_coverage import task_coverage_snapshot


def _attempt_complete(
    history: list[dict], coverage: dict, final_validation_skips: dict[str, str] | None = None,
) -> bool:
    """Judge only the current claims attempted in this batch, including capped scans."""
    latest = {}
    for row in history:
        claims = row.get("evaluated_claims") or {}
        for name in row.get("evaluated_names") or []:
            latest[name] = claims.get(name)
    if not latest:
        return False
    detectors = {row["name"]: row for row in coverage.get("detectors", [])}
    for name, claim in latest.items():
        if claim and (final_validation_skips or {}).get(name) == claim:
            # Selective final validation explicitly closed this same attempt.
            # Do not invent a completed review or add confirmed task coverage.
            continue
        detector = detectors.get(name, {})
        if not (claim and detector.get("current_claim_hash") == claim
                and detector.get("selected_evaluation_complete") is True
                and (detector.get("review_complete") is True
                     or detector.get("review_deferred") is True)):
            return False
    return True


def observe_task_progress(
    rt: Any, *, coverage: dict | None = None, initialize: bool = False,
) -> dict[str, Any]:
    """Share one durable high-water checkpoint between the driver and MCP status.

    A read repeated without new work cannot consume patience. Histories without a
    matching checkpoint establish a baseline rather than inventing historical plateaus.
    Current library coverage must exceed its best previous union size: swapping,
    losing and restoring tasks, overlaps, and repeated rollouts never earn gains.
    """
    coverage = coverage if coverage is not None else task_coverage_snapshot(
        rt, include_task_ids=True,
    )
    directory = rt.run.run_dir()
    status = read_research_status(directory)
    previous = dict(status.get("task_coverage_progress") or {})
    # Older checkpoints saved turn-finalization deferrals here; nothing reads them now.
    previous.pop("pending_attempts", None)
    fingerprint = coverage.get("corpus_fingerprint")
    available = bool(coverage.get("available") and fingerprint)
    public_coverage = {key: value for key, value in coverage.items()
                       if key != "confirmed_failed_task_ids"}
    if previous.get("version") == 1 and not available:
        # Missing source evidence cannot establish a changed corpus or erase
        # its high-water mark, pending work, or already-accounted history.
        return {
            **previous, "progress_blocked": True, "improved": False,
            "confirmed_task_gain": 0, "counts_toward_patience": False,
            "baseline_initialized": False, "task_coverage": public_coverage,
        }
    same_scope = (previous.get("version") == 1 and fingerprint
                  and previous.get("corpus_fingerprint") == fingerprint
                  and previous.get("behavior_strictness", "causal")
                  == coverage.get("behavior_strictness", "causal"))
    valid = (same_scope
             and previous.get("quality_thresholds") == coverage.get("quality_thresholds"))
    history = rt.scores_history
    current = int(coverage.get("n_confirmed_failed_tasks", 0))
    state = dict(previous) if valid else {
        "version": 1, "corpus_fingerprint": fingerprint,
        "behavior_strictness": coverage.get("behavior_strictness", "causal"),
        "quality_thresholds": coverage.get("quality_thresholds"),
        "best_confirmed_failed_tasks": current,
        "experiments_since_improve": 0, "completed_attempts": 0,
        "history_cursor": len(history), "latest_attempt_complete": False,
    }
    if same_scope and not valid:
        # New quality floors reset coverage patience, not unfinished claim reviews
        # or the accounting that prevents completed attempts from being replayed.
        for key in ("history_cursor", "latest_attempt_complete", "completed_attempts"):
            if key in previous:
                state[key] = previous[key]
    cursor = min(int(state.get("history_cursor", 0)), len(history))
    pending = history[cursor:]
    final_validation_skips = {
        fn.name: receipt["claim_hash"]
        for fn in getattr(getattr(rt, "library", None), "functions", [])
        if (receipt := current_final_validation_skip(rt, fn)) is not None
    }
    improved = bool(valid and coverage.get("available")
                    and current > state["best_confirmed_failed_tasks"])
    delta = max(0, current - state["best_confirmed_failed_tasks"]) if improved else 0
    complete = bool(coverage.get("available") and pending
                    and _attempt_complete(pending, coverage, final_validation_skips))
    if improved:
        state["best_confirmed_failed_tasks"] = current
        state["experiments_since_improve"] = 0
    if not initialize:
        completed = int(complete)
        if completed:
            state["completed_attempts"] += completed
            if not improved:
                state["experiments_since_improve"] += completed
        if pending:
            state["latest_attempt_complete"] = complete
        if complete:
            state["history_cursor"] = len(history)
    else:
        completed = 0
    state["current_confirmed_failed_tasks"] = current
    state["progress_blocked"] = bool(
        not coverage.get("available") or not state["latest_attempt_complete"]
    )
    status["task_coverage_progress"] = state
    write_research_status(directory, status)
    # IDs are internal provenance, not useful bulk in a status response/prompt.
    return {
        **state, "improved": improved, "confirmed_task_gain": delta,
        "counts_toward_patience": bool(completed),
        "baseline_initialized": not valid,
        "task_coverage": public_coverage,
    }


def final_validation_handoff(
    rt: Any, progress: dict, target: dict, *, ready: bool, allow_pending: bool = False,
    discovery_target: dict | None = None,
) -> dict:
    """Request a final census for supported claims without a complete corpus scan."""
    if discovery_target is not None:
        # At the ordinary round boundary, the driver uses the same Good labels
        # as the UI. A valid sampled support estimate can qualify even when the
        # stricter progress ledger has fewer observed confirmations. Validation
        # rechecks the same public grade requirements before completion.
        supported = [row for row in discovery_target["failure_mode_eligibility"]
                     if row.get("eligible")]
        candidates = [row for row in supported
                      if (row.get("measured") or {}).get("corpus_evaluation_complete") is not True]
        candidate_minimum = discovery_target["minimum_met"]
    else:
        # Plateau-driven handoffs retain their existing policy.
        detectors = progress["task_coverage"].get("detectors", [])
        supported = [row for row in detectors if row.get("eligible")]
        candidates = [row for row in supported if row.get("corpus_evaluation_complete") is not True]
        candidate_minimum = (not target["minimum_met"]
                             and len(supported) >= target["min_failure_modes"])
    handoff = bool(ready and (allow_pending or not progress["progress_blocked"])
                   and candidate_minimum and candidates)
    names = [row["name"] for row in candidates] if handoff else []
    boundary_guidance = (
        "FINAL VALIDATION — the discovery target reached the planned round boundary. "
        "Freeze these definitions and return your summary now. The driver will run "
        "full-dataset scoring and representative review for eligible OK-or-better "
        "detectors in the saved library, then "
        "recheck the current grades and completion decision. Reuse applicable reviews; "
        "do not start a separate full scan. If validation falls short, the runtime will "
        "return its findings for further discovery within the configured stopping policy."
    )
    progress_guidance = (
        "FINAL VALIDATION — capped discovery reached a stopping boundary with enough "
        "quality-supported candidates. Freeze these definitions and run "
        f"evaluate_functions(names={names!r}, mode='full') once, then check pareto_status "
        "and the actual qualifying-mode minimum. Keep current reviews; judge only if "
        "missing, stale, or an additional review is explicitly needed. If the census "
        "does not confirm the minimum, continue capped discovery/repair. Do not repeat "
        "a full scan of these unchanged versions."
    )
    return {
        "ready_for_final_validation": handoff,
        "final_validation_candidates": names,
        "final_validation_guidance": (
            (boundary_guidance if discovery_target is not None else progress_guidance)
            if handoff else None
        ),
    }
