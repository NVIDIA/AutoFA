# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver-owned validation at completed discovery round boundaries."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.metrics.discovery_feedback import build_discovery_feedback
from autofa.metrics.discovery_grades import grade_summary
from autofa.reports.ordering import rank_display_functions
from autofa.session.discovery_target import (
    agent_target_grade_for_runtime,
    discovery_good_target_for_runtime,
    failure_mode_target_for_runtime,
)
from autofa.session.status_log import emit_runtime_status
from autofa.ui.evaluation_progress import evaluation_progress
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import check_cancelled


def checkpoint_fraction(rt: Any, *, campaign: dict | None = None) -> float:
    """The stored round is zero-based; the final planned round skips the 30% step."""
    from autofa.session.campaign_policy import checkpoint_fraction as campaign_fraction

    fraction = campaign_fraction(rt, status=campaign)
    return fraction if fraction is not None else (
        1.0 if rt.run.round + 1 >= rt.run.max_rounds else 0.3)


def refinement_guidance() -> str:
    """Shared closing instructions for live feedback and resumed harness turns."""
    return (
        "Do the improvement work in this round. For promising candidates below Good, "
        "use qualification_gap to address the measured blockers to Good before pursuing "
        "the additional breadth of the active target. Keep Excellent and Outstanding as "
        "the higher aspiration and preserve supported Good-or-better versions. "
        "Choose a concrete, evidence-backed refinement and test it on the supplied "
        "counterexamples and supported matches across tasks. Compare revisions on the "
        "unchanged scoring panel; targeted regression controls do not establish population "
        "rates. Validate newly added scope separately when broadening. If you identify a "
        "feasible refinement, test it before advancing or explain the specific missing "
        "evidence, failed experiment, or supported trade-off that prevents it. Finding the "
        "behavior in selected passing examples does not establish an unfixable score "
        "ceiling. Missing scores, source evidence, and execution errors need their own "
        "recovery actions, not an automatic predicate rewrite. In validation_response, "
        "name the candidates investigated, the blocking dimensions, the tested change "
        "and results, or the concrete evidence for retaining/deferring each candidate; "
        "state any untested refinement explicitly. Do not postpone feasible improvements "
        "to the final scan. The last planned round still automatically validates "
        "OK-or-better detectors on 100% of the dataset, after bounded review of promising "
        "pending candidates. Other candidates retain their measured scores and explicit "
        "exclusion reasons. Full-scan scores determine final grades, and a downgrade "
        "below the minimum returns discovery to improvement under the runtime's stopping "
        "policy. Never preserve a provisional label over contrary validation results."
    )


def active_round_refinement(rt: Any) -> dict[str, Any] | None:
    """Return only this run's current, unfinished validation repair exchange."""
    try:
        receipt = json.loads((rt.run.run_dir() / "round_validation.json").read_text())
        if not isinstance(receipt, dict):
            return None
        state = receipt.get("refinement") or {}
    except (OSError, ValueError, TypeError):
        return None
    if (not isinstance(state, dict) or state.get("version") != 1
            or state.get("run_id") != rt.run.run_id or state.get("round") != rt.run.round
            or state.get("status") not in {"awaiting_response", "validating_response"}):
        return None
    return state


def round_refinement_prompt(rt: Any) -> str | None:
    """A paused/resumed harness must finish the exchange before automatic completion."""
    state = active_round_refinement(rt)
    if not state:
        return None
    from autofa.session.validation_summary import (
        current_validation_summary,
        render_validation_summary,
    )

    measurements = "\n\n" + render_validation_summary(
        current_validation_summary(rt), for_agent=True,
    )
    if state.get("mode") == "polishing":
        from autofa.session.prompts import POST_VALIDATION_POLISHING_GUIDANCE

        return (
            f"ROUND {rt.run.round + 1} FINAL POLISHING — this round remains open. "
            + POST_VALIDATION_POLISHING_GUIDANCE
            + " Inspect get_function_feedback for each affected detector's score changes, "
            "groups of problematic datapoints, item IDs and review explanations. "
            "You can revise and evaluate repeatedly before closing this repair exchange. "
            "Call advance_round without a response to retrieve the saved findings. "
            "When the revisions are ready, call advance_round(validation_response='tested "
            "repairs and results, or specific evidence for retaining/deferring each detector'). "
            "That closes polishing with full verification of the selected revisions; "
            "unchanged full scans are reused. "
            + ("Your closing response is saved; call advance_round to resume verification."
               if state["status"] == "validating_response" else "")
            + measurements
        )
    return (
        f"ROUND {rt.run.round + 1} CLOSING REFINEMENT — validation has returned its findings "
        "and this round remains open for one focused repair pass. "
        + refinement_guidance() + " Inspect the checkpoint "
        "feedback and get_function_feedback for the retained detectors. Repair, restore, "
        "retain, or defer them based on that evidence. Test the supplied problematic "
        "datapoints as separate regression controls; keep the 200-example comparison panel "
        "unchanged. Do not start a new family exploration in this closing pass. Finish with "
        "advance_round(validation_response=\"what changed and what the checks showed, or "
        "why the versions were retained/deferred\"). The runtime revalidates revisions "
        "before advancing and deciding whether discovery can finish. If validation was "
        "interrupted after saving your response, advance_round resumes it. "
        + ("Call advance_round without a response to retrieve the pending validation feedback "
           "if it is no longer in context." if state["status"] == "awaiting_response" else
           "Your response is saved; call advance_round to resume its interrupted validation.")
        + measurements
    )


def validation_feedback(rt: Any, before: dict | None = None) -> dict[str, Any]:
    """Expose changed grades and existing dimension-specific advice after validation."""
    from autofa.session.polishing_guard import active_polishing
    from autofa.session.validation_controls import validation_controls
    from autofa.session.validation_summary import evaluation_scope

    strictness = strictness_for_config(rt.config)
    summary = grade_summary(
        rt.library.functions, behavior_strictness=strictness,
        review_history=rt.library.review_history,
    )
    old = {row["name"]: row for row in (before or {}).get("functions", [])}
    current = {row["name"]: row for row in summary["functions"]}
    views = {row["name"]: row for row in rank_display_functions(
        [vars(fn) for fn in rt.library.functions], behavior_strictness=strictness,
        review_history=rt.library.review_history,
    )}
    target_grade = agent_target_grade_for_runtime(rt)
    polishing = active_polishing(rt)

    def measured_rate(metrics: dict, key: str) -> float | None:
        value = metrics.get(key)
        return (float(value) if isinstance(value, (int, float))
                and not isinstance(value, bool) and 0 <= value <= 1 else None)

    detectors = []
    for fn in rt.library.functions:
        row = current[fn.name]
        scores, metrics = views[fn.name]["scores"], views[fn.name]["metrics"]
        feedback = build_discovery_feedback(
            scores, behavior_strictness=strictness, target_grade=target_grade,
        )
        detector = {
            "name": fn.name,
            "previous_grade": old.get(fn.name, {}).get("code"),
            "grade": row["code"],
            "coverage": measured_rate(metrics, "recall"),
            "normalized_precision": measured_rate(metrics, "ppv_at_pi0"),
            "adherence": measured_rate(metrics, "behavior_purity"),
            "evaluation_scope": evaluation_scope(scores),
            "validation_sample": scores.get("round_validation"),
            "feedback": {key: feedback.get(key) for key in (
                "status", "grade_gap_summary", "qualification_gap",
                "qualification_gap_summary", "next_steps",
            )},
            "evidence_action": f"get_function_feedback(name={fn.name!r})",
        }
        if polishing:
            from autofa.session.polishing_feedback import polishing_feedback

            detector["polishing_feedback"] = polishing_feedback(rt, fn, summary_only=True)
        else:
            detector["regression_checks"] = validation_controls(rt, fn, scores=scores)
        detectors.append(detector)
    counts = ", ".join(f"{n} {grade.title()}" for grade, n in summary["counts"].items() if n)
    from autofa.session.prompts import POST_VALIDATION_POLISHING_GUIDANCE

    return {
        "grades": summary,
        "detectors": detectors,
        "refinement_guidance": (POST_VALIDATION_POLISHING_GUIDANCE if polishing
                                else refinement_guidance()),
        "message": (
            f"Validation updated the library: {counts or 'no detectors'}. "
            + ("Polish the admitted detectors to Good or better. Inspect the datapoints and "
               "groups behind each downgrade, test revisions on all cumulative controls plus "
               "the fixed panel, and close with validation_response for full verification. "
               "No new detectors or family investigations belong in this phase."
               if polishing else
            "Use these current scores and review findings when deciding what to preserve "
            "or improve. For promising candidates below Good, address qualification_gap "
            "before the higher target's additional breadth. Test identified feasible "
            "refinements in this round or cite specific evidence for deferring them; "
            "an untested score ceiling is not a completed improvement investigation. "
            "Inspect get_function_feedback for cited counterexamples and "
            "unresolved evidence; the feedback identifies the blocking dimensions. "
            "Keep useful validated versions while investigating improvements.")
        ),
    }


def refresh_validated_target(rt: Any, *, final: bool = False) -> dict[str, Any]:
    """Replace a stale provisional count without inventing a new stopping policy."""
    path = rt.run.run_dir() / "discovery_target.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    if final:
        from autofa.session.validation_completion import validated_good_target_for_runtime

        current = validated_good_target_for_runtime(rt)
    else:
        current = discovery_good_target_for_runtime(rt)
        for key in ("validation_scope", "validation_blockers", "retained_results_complete",
                    "campaign_complete", "campaign"):
            previous.pop(key, None)
    saved = {**previous, **current, "validation_checked": True,
             "final_validation_target": failure_mode_target_for_runtime(rt)}
    # Completion also depends on round/budget policy. Validation may revoke an
    # existing completion decision, but never establishes that boundary itself.
    saved["goal_met"] = bool(final and previous.get("goal_met") and current["minimum_met"])
    saved.pop("agent_grade_feedback", None)
    atomic_write_json(path, saved)
    return saved


def validate_round(rt: Any) -> dict[str, Any]:
    """Run a bounded-panel/full-corpus checkpoint before changing the round cursor."""
    from autofa.mcp.tools_eval import evaluate_round_functions, final_evaluate_functions
    from autofa.metrics.final_review import final_review_functions
    from autofa.session.campaign_policy import CampaignBoundaryReached, check_campaign_boundary
    from autofa.session.control import apply_pending_updates
    from autofa.session.operation_admission import admit_operation
    from autofa.session.supervision import check_lifecycle
    from autofa.session.usage import usage_scope

    apply_pending_updates(rt)
    check_lifecycle(rt.run.run_dir(), phase="validation")
    campaign = check_campaign_boundary(rt)
    if campaign.get("finalize_due"):
        raise CampaignBoundaryReached(campaign)
    from autofa.session.detector_service import infrastructure_block

    if blocked := infrastructure_block(rt):
        return {"ok": False, "advanced": False, "status": "infrastructure_blocked",
                "service": blocked, "error": "Detector service failure left semantic work "
                "unresolved. Recover it before treating this checkpoint as completed."}
    fraction = checkpoint_fraction(rt, campaign=campaign)
    label = f"{fraction:.0%}"
    before = grade_summary(
        rt.library.functions, behavior_strictness=strictness_for_config(rt.config),
        review_history=rt.library.review_history,
    )
    previous_scores = {fn.name: deepcopy(fn.scores or {}) for fn in rt.library.functions}
    previous_refinement = active_round_refinement(rt)
    if not rt.library.functions:
        return {"ok": True, "fraction": fraction, "message": "No retained detectors to validate."}
    # Reserve the whole library while selecting, scoring and reviewing one
    # version set. Nested detector operations share this owner.
    with (usage_scope(run_dir=rt.run.run_dir(), phase="round_validation",
                      round_index=rt.run.round), admit_operation("advance_round", {}) as busy):
        if busy:
            return {**busy, "advanced": False}
        check_cancelled()
        emit_runtime_status(
            f"Round {rt.run.round + 1} validation: scoring a {label} dataset target.",
            source="round_validation", context={"fraction": fraction, "phase": "scoring"},
        )
        last_completed = -1

        def progress(event: dict) -> None:
            nonlocal last_completed
            completed = event["completed"]
            # Avoid thousands of durable status rows while keeping visible work.
            if completed - last_completed < 25 and event["phase"] != "saving":
                return
            last_completed = completed
            total = event["total"]
            emit_runtime_status(
                f"Round validation {label}: {completed}/{total if total is not None else '?'} "
                f"predictions · {event['detector']}",
                source="round_validation", context={**event, "fraction": fraction},
            )

        selection = None
        if fraction == 1.0:
            from autofa.metrics.final_validation_selection import prepare_final_validation

            selection = prepare_final_validation(rt)
            emit_runtime_status(
                f"Full validation: {len(selection['selected_names'])} detectors selected; "
                f"{len(selection.get('skipped', []))} skipped; "
                f"{len(selection.get('unresolved', []))} awaiting usable scores or review.",
                source="round_validation", context={"fraction": fraction, "phase": "selection"},
            )
        with evaluation_progress(progress):
            result = (final_evaluate_functions(names=selection["selected_names"])
                      if selection is not None else evaluate_round_functions(fraction=fraction))
        if not result.get("ok"):
            return {**result, "fraction": fraction, "advanced": False}
        check_cancelled()
        if blocked := infrastructure_block(rt):
            return {"ok": False, "advanced": False, "fraction": fraction,
                    "status": "infrastructure_blocked", "service": blocked,
                    "error": "Validation encountered an unavailable detector service; "
                    "saved predictions remain available, but discovery is not exhausted."}
        if fraction == 1.0:
            # Freeze selection across scoring and review. A full scan can lower
            # the grade; that must not silently drop its still-required review.
            review = final_review_functions(names=selection["selected_names"])
        else:
            # Cheap full-scanned detectors keep their stronger measurements,
            # while this early checkpoint still obeys iterative review budgets.
            review = final_review_functions(scope="round_checkpoint")
        # Save the full-validation evidence before any repair can replace scores.
        # Keep the original exchange baseline while refreshing cumulative controls.
        refinement = previous_refinement
        if fraction == 1.0 and not campaign.get("continue_discovery"):
            from autofa.session.polishing import prepare_polishing

            refinement = prepare_polishing(
                rt, before=before, previous_scores=previous_scores, previous=refinement,
            )
            if previous_refinement:
                refinement.update({key: previous_refinement[key] for key in (
                    "status", "response", "evaluated_claims",
                ) if key in previous_refinement})
            # Feedback must use the Good polishing target immediately.
            atomic_write_json(rt.run.run_dir() / "round_validation.json",
                              {"refinement": refinement})
        feedback = validation_feedback(rt, before)
        refresh_validated_target(rt, final=fraction == 1.0)
        receipt = {
            "ok": True, "round": rt.run.round, "fraction": fraction,
            "checkpoint_kind": "campaign_intermediate" if campaign.get("continue_discovery")
            else "closing",
            "review_unresolved": [*(review.get("unresolved") or []),
                                  *((selection or {}).get("unresolved") or [])], **feedback,
        }
        if selection is not None:
            receipt["final_validation_selection"] = selection
        if refinement:
            receipt["refinement"] = refinement
        atomic_write_json(rt.run.run_dir() / "round_validation.json", receipt)
        emit_runtime_status(
            f"Round {rt.run.round + 1} validation ({label}) complete. " + feedback["message"],
            source="round_validation", context={"fraction": fraction, "phase": "complete"},
        )
        return receipt
