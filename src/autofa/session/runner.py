# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start an AutoFA agent session for a run."""

from __future__ import annotations

import asyncio
import json
import logging
from itertools import chain, islice
from pathlib import Path
from typing import Any

from rich.console import Console

from autofa.config import Config, RunConfig, load_config
from autofa.harness.claude import get_harness
from autofa.harness.settings import HARNESS_NAMES, harness_cli
from autofa.mcp.runtime import load_runtime_from_run
from autofa.mcp.server import build_mcp_server, tool_names
from autofa.session.family_discovery import (
    family_discovery_prompt,
    initialize_family_discovery,
)
from autofa.session.guidance import guidance_prompt
from autofa.session.llm_readiness import check_llm_readiness, llm_readiness_guidance
from autofa.session.prompts import (
    build_system_prompt,
    format_continue_kickoff,
    format_discover_kickoff,
    write_session_workspace,
)
from autofa.session.recovery import exclusive_run
from autofa.ui.errors import set_action_stage
from autofa.ui.settings import scoring_summary
from autofa.ui.theme import GREEN, WARN
from autofa.util.atomic import atomic_write_json

console = Console()


def _dataset_health(dataset: Any) -> tuple[dict[str, Any], str]:
    """Keep preflight traces out of the frame that lives for the whole session."""
    from autofa.data.corpus_health import detect_corpus_domain, measure_parse_coverage

    batches = dataset.iter_batches(batch_size=32, max_items=40)
    try:
        sample = list(islice(chain.from_iterable(batches), 40))
        coverage = measure_parse_coverage(sample, record_diagnostics=True)
        return coverage, detect_corpus_domain(sample, coverage)
    finally:
        if callable(close := getattr(batches, "close", None)):
            close()


@exclusive_run
def run_session(  # noqa: C901
    run: RunConfig,
    *,
    cfg: Config | None = None,
    auto_discover: bool = True,
    continue_mode: bool = False,
    max_rounds: int | None = None,
    min_failure_modes: int | None = None,
    top_x: int | None = None,
    exit_after_discovery: bool = False,
    publish_on_exit: bool = True,
) -> None:
    if max_rounds is not None and (
        type(max_rounds) is not int or max_rounds < max(1, run.round)
    ):
        raise ValueError("The total round target must be a positive integer at least "
                         f"the current round ({run.round})")
    if top_x is not None:
        if top_x <= 0:
            raise ValueError("top_x must be a positive integer")
    cfg = cfg or load_config()
    from autofa.session.classifier_profiles import initialize_classifier_policy
    from autofa.session.usage import initialize_run_accounting

    initialize_classifier_policy(cfg, run)
    initialize_run_accounting(run.run_dir(), pricing=getattr(run, "usage_pricing", {}) or None)
    from autofa.session.managed_launch import bind_run

    bind_run(run.run_dir(), run.run_id)
    from autofa.preflight.runner import run_preflight
    from autofa.session.supervision import check_lifecycle

    check_lifecycle(run.run_dir(), phase="validation")
    run_preflight(run, cfg)
    from autofa.estimates import save_estimate_context

    save_estimate_context(run, cfg)
    set_action_stage("initialize analysis", analysis_run=run.run_id, run_dir=str(run.run_dir()))
    from autofa.session.execution import record_execution

    record_execution(run.run_dir())
    from autofa.util.memory import apply_memory_tuning

    tune = apply_memory_tuning(cfg, parquet_paths=run.parquet_paths)
    rt = load_runtime_from_run(run, cfg)
    cfg = rt.config
    from autofa.session.campaign_policy import (
        CampaignBoundaryReached,
        campaign_guidance,
        check_campaign_boundary,
    )
    from autofa.session.control import (
        apply_pending_updates,
        consume_control_requests,
        submit_update,
    )

    applied = apply_pending_updates(rt)
    explicit = {key: value for key, value in {
        "max_rounds": max_rounds, "min_failure_modes": min_failure_modes,
    }.items() if value is not None}
    if explicit:
        # Reconcile interrupted commits before explicit CLI goals supersede old
        # queued settings. Use the same transaction/revision path as live edits.
        if any(row.get("state") in {"deferred", "applying", "uncertain"} for row in applied):
            raise RuntimeError(
                "Pending control changes need reconciliation before Resume overrides")
        from uuid import uuid4

        request = submit_update(run.run_dir(), explicit, request_id="resume-" + uuid4().hex,
                                expected_revision=run.control_revision)
        results = apply_pending_updates(rt)
        result = next((row for row in results if row.get("id") == request["id"]), {})
        if result.get("state") != "applied":
            raise RuntimeError("Explicit Resume goals were not applied: " + str(result))
    if top_x is not None:
        run.top_x = top_x
        run.save()
    rt.persist_library()
    if continue_mode:
        from autofa.session.validation_recovery import recover_interrupted_validation

        recover_interrupted_validation(rt)
    initialize_family_discovery(rt)

    run_dir = run.run_dir()
    console.print(f"[{GREEN}]Run saved under[/] [bold]{run_dir}[/bold]")
    console.print(
        f"[dim]  working library: {run_dir / 'library.json'}\n"
        f"  scores history:  {run_dir / 'scores_history.json'}\n"
        f"  durable catalog: ~/.autofa/library/{{model}}/{{benchmark}}/library.json "
        f"(also written on exit + /save)\n"
        "  /save publishes to the configured remote library; /exit or /quit also ends the session\n"
        f"  {tune.summary()}[/dim]"
    )
    console.print(
        scoring_summary(cfg) + ".",
        style="dim", markup=False,
    )

    harness = get_harness(
        cfg.harness, cfg.model, cli_path=harness_cli(cfg, cfg.harness), cfg=cfg,
        thinking=cfg.thinking, effort=cfg.effort, thinking_budget_tokens=cfg.thinking_budget_tokens,
    )
    if run.harness != cfg.harness:
        run.harness = cfg.harness
        run.save()

    starting_campaign = check_campaign_boundary(rt)
    if starting_campaign.get("finalize_due"):
        rt.llm_readiness = {"available": None, "reason": "campaign_boundary"}
    else:
        try:
            from autofa.session.llm_readiness import check_and_recover_readiness

            rt.llm_readiness = check_and_recover_readiness(
                rt, console=console, probe=check_llm_readiness)
            cfg = rt.config
        except CampaignBoundaryReached:
            rt.llm_readiness = {"available": None, "reason": "campaign_boundary"}
        if (getattr(run, "require_llm_detectors", False)
                and rt.llm_readiness.get("available") is not True
                and rt.llm_readiness.get("reason") != "campaign_boundary"):
            raise RuntimeError("Required detector endpoint readiness failed: "
                               + str(rt.llm_readiness.get("reason", rt.llm_readiness)))
    if (getattr(run, "require_llm_detectors", False) and not rt.library.functions
            and rt.llm_readiness.get("available") is not True):
        raise RuntimeError("Required detector readiness was not verified before the campaign "
                           "boundary; no analysis was completed. Review limits/access and Resume.")

    # Diagnostics + data-health preflight
    from autofa.diagnostics import get_diagnostics

    diag = get_diagnostics()
    diag.configure(run_dir)
    try:
        set_action_stage("check dataset health")
        coverage, run.corpus_domain = _dataset_health(rt.dataset)
        run.save()
        console.print(
            f"[dim]Data health: parse_rate={coverage.get('parse_rate', 0):.0%} "
            f"· dialects={coverage.get('dialects') or 'plain'} "
            f"· domain={run.corpus_domain} "
            f"· n_sampled={coverage.get('n_sampled')}[/dim]"
        )
        for line in diag.preflight_lines()[:8]:
            style = WARN if line.startswith("[CRIT") or line.startswith("[WARN") else "dim"
            console.print(f"[{style}]{line}[/{style}]")
        console.print(
            f"[dim]  logs: {run_dir / 'autofa.log'} · "
            f"{run_dir / 'diagnostics.jsonl'}[/dim]"
        )
    except Exception as e:  # noqa: BLE001
        # Invalid input must not proceed into discovery as if the preflight passed.
        raise RuntimeError(f"Dataset readiness check failed: {e}") from e

    cwd = write_session_workspace(
        run.run_dir(),
        top_x=run.top_x,
        max_rounds=run.max_rounds,
        min_failure_modes=run.min_failure_modes,
        exploration_ssot=bool(getattr(run, "exploration_ssot", True)),
        exploration_seed=int(getattr(run, "exploration_seed", 0) or 0),
        exploration_templates=bool(getattr(run, "exploration_templates", False)),
        corpus_domain=str(getattr(run, "corpus_domain", "unknown")),
        analysis_guidance=guidance_prompt(run, audience="discovery"),
        objective_config=cfg,
    )
    mcp = build_mcp_server("autofa") if cfg.harness == "claude" else True
    allowed = [f"mcp__autofa__{n}" for n in tool_names()] + [
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "AskUserQuestion",
    ]

    ssot = bool(getattr(run, "exploration_ssot", True))
    seed = int(getattr(run, "exploration_seed", 0) or 0)
    templates = bool(getattr(run, "exploration_templates", False))
    domain = str(getattr(run, "corpus_domain", "unknown"))
    if continue_mode or (run.seed_library_path and rt.library.functions):
        initial = format_continue_kickoff(
            top_x=run.top_x,
            max_rounds=run.max_rounds,
            min_failure_modes=run.min_failure_modes,
            exploration_ssot=ssot,
            exploration_seed=seed,
            exploration_templates=templates,
            corpus_domain=domain,
            run_dir=run.run_dir(),
            llm_available=(rt.llm_readiness or {}).get("available"),
            objective_config=rt.config,
        )
    elif auto_discover:
        initial = format_discover_kickoff(
            top_x=run.top_x,
            max_rounds=run.max_rounds,
            min_failure_modes=run.min_failure_modes,
            exploration_ssot=ssot,
            exploration_seed=seed,
            exploration_templates=templates,
            corpus_domain=domain,
            run_dir=run.run_dir(),
            llm_available=(rt.llm_readiness or {}).get("available"),
            objective_config=rt.config,
        )
    else:
        initial = None

    session_path = run.run_dir() / "session.json"

    def saved_session_id() -> str | None:
        # Harnesses save their session identity during each turn. A campaign
        # boundary can leave run_session before it returns the latest identity.
        try:
            saved_session = json.loads(session_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if saved_session.get("harness", "claude") == cfg.harness:
            return saved_session.get("session_id")
        return None
    if continue_mode:
        from autofa.session.recovery import resume_prompt

        initial = resume_prompt(run_dir, rt.library.functions, fallback=initial or "Continue",
                                round_idx=run.round)
        initial = family_discovery_prompt(run_dir, objective_config=cfg) + "\n" + initial

    from autofa.session.discovery_target import (
        DiscoveryProgress,
        discovery_good_target_for_runtime,
        discovery_grade_feedback,
        inactivity_recovery_prompt,
        minimum_target_prompt,
    )

    progress = DiscoveryProgress(rt)
    from autofa.session.discovery_stall import discovery_stall_recovery, observe_discovery_stall
    from autofa.session.task_progress import observe_task_progress

    initial_progress = observe_task_progress(rt, initialize=True)
    initial_stall = observe_discovery_stall(rt, initialize=True)
    initial_feedback = discovery_grade_feedback(
        rt, target=discovery_good_target_for_runtime(rt),
        task_coverage=initial_progress["task_coverage"],
    )
    if initial:
        # Prefix after Resume has reconstructed its prompt, using live round/settings.
        initial = initial_feedback["message"] + "\n\n" + initial
        if recovery := discovery_stall_recovery(initial_stall):
            initial = recovery["prompt"] + "\n\n" + initial
    from autofa.session.finalization import finalize_discovery
    from autofa.session.round_stagnation import round_stagnation_status
    from autofa.session.round_validation import (
        refresh_validated_target,
        round_refinement_prompt,
        validation_feedback,
    )

    automatic_validation_handoff = False

    async def validate_agent_turn() -> str | None:
        from autofa.session.turn_validation import turn_validation_prompt
        from autofa.session.validation_recovery import (
            complete_recovered_validation,
            pending_recovered_validation,
        )

        try:
            await consume_control_requests(rt)
            if check_campaign_boundary(rt).get("finalize_due"):
                return None
            from autofa.session.detector_service import infrastructure_block

            if infrastructure_block(rt):
                return None  # Continuation owns the finite service-recovery exchange.
            if round_stagnation_status(rt).get("review_capacity_block"):
                return None  # Let continuation honor the saved operational stop first.
            feedback = await turn_validation_prompt(rt, console=console)
            if not feedback and pending_recovered_validation(rt):
                await finalize_discovery(console)
                complete_recovered_validation(rt)
            if feedback and (current_guidance := campaign_guidance(rt)):
                feedback = current_guidance + "\n\n" + feedback
            return feedback or None
        finally:
            await asyncio.to_thread(_flush_quality_snapshot, rt)

    async def finish_discovery() -> str | None:
        nonlocal automatic_validation_handoff
        from copy import deepcopy

        from autofa.behavior_policy import strictness_for_config
        from autofa.metrics.discovery_grades import grade_summary
        from autofa.session.report_highlights import prepare_report_highlights_async

        reconsider = automatic_validation_handoff
        automatic_validation_handoff = False
        previous_scores = {fn.name: deepcopy(fn.scores or {}) for fn in rt.library.functions}
        before = grade_summary(
            rt.library.functions, behavior_strictness=strictness_for_config(rt.config),
            review_history=rt.library.review_history,
        )
        await consume_control_requests(rt)
        campaign = check_campaign_boundary(rt)
        closing_revision = getattr(run, "control_revision", 0)
        from autofa.session.detector_service import infrastructure_block

        if blocked := infrastructure_block(rt):
            # A down service cannot certify exhaustion or force another paid
            # full scan. Retain the saved measurements and render an honest,
            # resumable partial result; the outer driver owns waiting/extension.
            rt.persist_library()
            report = write_run_report(run, cfg=rt.config)
            target = discovery_good_target_for_runtime(rt)
            target.update(goal_met=False, blocked=True, stop_reason="infrastructure_blocked",
                          service=blocked)
            atomic_write_json(run_dir / "discovery_target.json", target)
            atomic_write_json(run_dir / "finalization_outcome.json", {
                "version": 1, "state": "blocked", "report_generated": True,
                "report_path": str(report), "retained_results_complete": False,
                "resumable": True, "stop_reason": "infrastructure_blocked", "service": blocked,
            })
            return None
        boundary = None
        try:
            await finalize_discovery(console)
        except CampaignBoundaryReached as exc:
            boundary = exc
            rt.persist_library()
            atomic_write_json(run_dir / "finalization_outcome.json", {
                "version": 1, "state": "completed_with_gaps", "report_generated": False,
                "retained_results_complete": False, "stop_reason": exc.reason,
                "campaign": exc.status,
            })
        # Only goal edits are admitted after final validation. A newly arriving
        # detector contribution must wait for another native validation boundary.
        apply_pending_updates(rt)
        validated = refresh_validated_target(rt, final=True)
        current = check_campaign_boundary(rt)
        if ((campaign.get("finalize_due") and current.get("continue_discovery"))
                or (getattr(run, "control_revision", 0) != closing_revision
                    and not current.get("finalize_due"))):
            # A live extension, or an unconfirmed quality target, continues in
            # this session without inventing extra review/repair allowance.
            return continue_discovery()
        if campaign.get("finalize_due") or boundary is not None:
            current = boundary.status if boundary is not None else current
            validated.update(goal_met=bool(validated["minimum_met"]
                                           and current.get("coverage_target_met", True)),
                             stop_reason=current.get("stop_reason"), campaign=current,
                             campaign_complete=True)
            atomic_write_json(run_dir / "discovery_target.json", validated)
            if boundary is None:
                try:
                    await prepare_report_highlights_async(rt, console=console)
                except CampaignBoundaryReached:
                    pass  # Rendering/publishing saved evidence remains available.
            return None
        console.print(
            f"[cyan]Validated library: {validated['n_validated_failure_modes']}/"
            f"{validated['min_failure_modes']} Good-or-better detectors with completed "
            "full-dataset validation.[/cyan]"
        )
        if not reconsider or validated["minimum_met"]:
            await prepare_report_highlights_async(rt, console=console)
            return None
        # A failed automatic handoff returns existing detectors to polishing,
        # rather than reopening family discovery. Explicit user stops still stop.
        from autofa.session.polishing import prepare_polishing

        quality_stall = observe_discovery_stall(rt)
        if quality_stall["should_stop"]:
            # A genuine cost stop may retain an incomplete library, but the
            # provisional public grades must not overwrite this verification.
            # In particular, do not re-enter continue_discovery's public goal.
            stop_reason = quality_stall.get("stop_reason") or "stalled"
            validated.update(goal_met=False, stalled=stop_reason == "stalled",
                             blocked=stop_reason == "review_capacity_exhausted",
                             stop_reason=stop_reason,
                             quality_stall=quality_stall)
            atomic_write_json(run_dir / "discovery_target.json", validated)
            console.print(
                "[yellow]Final validation did not confirm the requested minimum. "
                + ("Discovery remains blocked by exhausted review capacity; "
                   if stop_reason == "review_capacity_exhausted" else
                   "Discovery remains stopped under the configured stall policy; ")
                + "the saved target records the unmet minimum and verification gaps.[/yellow]"
            )
            await prepare_report_highlights_async(rt, console=console)
            return None
        if rt.library.functions:
            path = rt.run.run_dir() / "round_validation.json"
            try:
                receipt = json.loads(path.read_text())
            except (OSError, ValueError):
                receipt = {}
            state = prepare_polishing(
                rt, before=before, previous_scores=previous_scores,
                previous=receipt.get("refinement"),
            )
            atomic_write_json(path, {"ok": True, "round": rt.run.round, "fraction": 1.0,
                                     "refinement": state})
            feedback = validation_feedback(rt, before)
            atomic_write_json(path, {"ok": True, "round": rt.run.round, "fraction": 1.0,
                                     **feedback, "refinement": state})
        next_work = continue_discovery()
        if next_work is None:
            await prepare_report_highlights_async(rt, console=console)
            return None
        feedback = validation_feedback(rt)
        incomplete = validated.get("validation_blockers") or []
        verification_feedback = (
            " Final verification remains pending for otherwise Good detectors: "
            + "; ".join(f"{row['name']}: {row['reason']}" for row in incomplete)
            + " Recover the indicated evaluation/review evidence before changing the predicate."
            if incomplete else ""
        )
        return (
            "Full-dataset validation changed the provisional library assessment. "
            "Discovery is continuing under the runtime's stopping policy. "
            + feedback["message"] + verification_feedback + "\n\n" + next_work
            + "\nValidation findings: " + json.dumps(feedback["detectors"], default=str)
        )

    def continue_discovery() -> str | None:
        nonlocal automatic_validation_handoff
        from autofa.session.supervision import check_lifecycle

        check_lifecycle(run_dir, phase="validation" if run.campaign_policy else "discovery")
        apply_pending_updates(rt)
        campaign = check_campaign_boundary(rt)
        if campaign.get("finalize_due"):
            automatic_validation_handoff = False
            target = discovery_good_target_for_runtime(rt)
            target.update(goal_met=False, stop_reason=campaign["stop_reason"], campaign=campaign)
            atomic_write_json(run_dir / "discovery_target.json", target)
            rt.persist_library()
            return None
        from autofa.session.detector_service import infrastructure_block, recovery_turn_guidance

        if blocked := infrastructure_block(rt):
            automatic_validation_handoff = False
            target = discovery_good_target_for_runtime(rt)
            target.update(goal_met=False, blocked=True, stop_reason="infrastructure_blocked",
                          service=blocked)
            atomic_write_json(run_dir / "discovery_target.json", target)
            return recovery_turn_guidance(rt)
        from autofa.mcp.tools_library import _pareto_status
        from autofa.session.polishing_guard import active_polishing
        from autofa.session.status_log import emit_runtime_status

        pending = active_polishing(rt)
        if pending and pending.get("transition_pending"):
            # The cursor and validation receipt are separate durable files. A
            # crash between their writes must not reopen family discovery.
            decision = _pareto_status()
            if not decision["goal_met"] and not (decision.get("quality_stall") or {}).get(
                "should_stop"
            ):
                from autofa.session.polishing import continue_polishing

                receipt = json.loads((run_dir / "round_validation.json").read_text())
                continue_polishing(rt, receipt)

        if (not round_stagnation_status(rt).get("review_capacity_block")
                and (refinement := round_refinement_prompt(rt))):
            automatic_validation_handoff = False
            current_guidance = campaign_guidance(rt, status=campaign)
            return current_guidance + "\n\n" + refinement if current_guidance else refinement

        initialize_family_discovery(rt)
        status = _pareto_status()
        target = discovery_good_target_for_runtime(rt)
        family = status.get("family_discovery") or {}
        synthesis_pending = bool(family.get("synthesis_pending"))
        stalled = progress.stalled()
        target["synthesis_pending"] = synthesis_pending
        target.update({key: status[key] for key in (
            "goal_met", "progress_blocked", "experiments_since_improve", "patience",
            "ready_for_final_validation", "final_validation_candidates",
            "discovery_wait_for_plateau",
        )})
        target["stalled"] = stalled and not status["goal_met"]
        quality_stall = observe_discovery_stall(rt)
        target["quality_stall"] = quality_stall
        recovery = (None if status["goal_met"] or status["ready_for_final_validation"]
                    else discovery_stall_recovery(quality_stall))
        target["discovery_recovery"] = recovery
        target["stop_reason"] = (
            (quality_stall.get("stop_reason") or "stalled")
            if quality_stall["should_stop"] and not status["goal_met"] else None
        )
        target["blocked"] = target["stop_reason"] == "review_capacity_exhausted"
        target["final_validation_target"] = status.get("final_validation_target")
        grade_feedback = discovery_grade_feedback(
            rt, target=target, task_coverage=status.get("task_coverage"),
        )
        target["agent_grade_feedback"] = grade_feedback
        atomic_write_json(run_dir / "discovery_target.json", target)
        # Only the runtime stopping condition ends automatic discovery here.
        # An idle or textual refusal is a recovery signal, not permission to finalize.
        if status["goal_met"]:
            automatic_validation_handoff = True
            return None
        automatic_validation_handoff = False
        if target["stop_reason"] in {"stalled", "review_capacity_exhausted"}:
            working = rt.persist_library()
            stop_detail = quality_stall.get("reason") or (
                "Completed extra recovery rounds without validated library improvement"
            )
            message = (
                f"Discovery terminated: {stop_detail}. "
                f"{target['n_validated_failure_modes']}/{target['min_failure_modes']} "
                "Good detectors; "
                + ("the minimum remains unmet. " if not target["minimum_met"] else "")
                + "Final full-dataset scoring and review will run before saving the report. "
                f"Library saved to {working}."
            )
            emit_runtime_status(message, source="discovery_target", context=target)
            console.print(message)
            return None
        if stalled:
            working = rt.persist_library()
            message = (
                "Automatic discovery is recovering after two turns without discovery progress: "
                f"{target['n_validated_failure_modes']}/{target['min_failure_modes']} "
                "Good detectors. The runtime stopping condition is not met; "
                "the next turn will receive concrete recovery guidance. "
                f"Library saved to {working}."
            )
            emit_runtime_status(message, source="discovery_target", context=target)
            from autofa.ui.errors import show_action_notice

            show_action_notice(message, console=console)
        else:
            emit_runtime_status(
                f"Continuing discovery: {target['n_validated_failure_modes']}/"
                f"{target['min_failure_modes']} Good detectors",
                source="discovery_target", context=target,
            )
        from autofa.session.diversity import detector_exploration

        exploration = detector_exploration(
            [fn.to_dict() for fn in rt.library.functions],
            llm_available=(rt.llm_readiness or {}).get("available"),
        )
        if status["ready_for_final_validation"]:
            next_work = status["final_validation_guidance"]
        elif target["minimum_met"] and synthesis_pending:
            next_work = (
                "Finish the scheduled bottom-up inspection or parent "
                "assessment. Use the existing evaluation/review budget, or document an "
                "evidence-backed deferral."
            )
        elif not target["minimum_met"]:
            next_work = minimum_target_prompt(
                target, round_idx=run.round, max_rounds=run.max_rounds,
            )
        elif campaign.get("continue_discovery"):
            next_work = (
                "Continue exploring or refining supported detectors to improve combined "
                "failed-task coverage. The detector minimum and legacy planned round count "
                "are milestones; the active campaign horizon determines completion."
            )
        elif not status["discovery_wait_for_plateau"]:
            next_work = (
                "Finish the planned discovery rounds "
                f"({run.round}/{run.max_rounds}) while seeking broader supported coverage. "
                "Wait for coverage plateau is Off: at the round limit, check pareto_status "
                "and hand off validation when goal_met is true. Full-dataset scoring "
                "and review run before the runtime confirms completion."
            )
        else:
            from autofa.session.task_coverage import task_coverage_prompt

            next_work = (
                "Discovery is not complete. Continue seeking "
                "additional confirmed coverage across distinct tasks. "
                f"Completed attempts without coverage gain: "
                f"{status['experiments_since_improve']}/{status['patience']}. "
                + ("Resolve pending evaluations or reviews before claiming a plateau. "
                   if status["progress_blocked"] else "")
                + task_coverage_prompt(status["task_coverage"])
                + " Check pareto_status before finishing; a grade improvement alone "
                "does not establish a stopping boundary."
            )
        if stalled:
            next_work = inactivity_recovery_prompt() + "\n\n" + next_work
        if recovery:
            emit_runtime_status(
                recovery["message"], source="discovery_recovery", context=recovery,
            )
            next_work = recovery["prompt"] + "\n\n" + next_work
        prompt = (grade_feedback["message"] + "\n\n" + next_work
                  + "\n" + family_discovery_prompt(run_dir, objective_config=cfg)
                  + "\n" + exploration["message"])
        current_guidance = campaign_guidance(rt, status=campaign)
        return current_guidance + "\n\n" + prompt if current_guidance else prompt

    async def _go(kickoff: str | None) -> None:
        from autofa.session.polishing_guard import active_polishing
        from autofa.session.report_highlights import prepare_report_highlights_async
        from autofa.session.turn_validation import turn_validation_prompt
        from autofa.session.validation_recovery import (
            complete_recovered_validation,
            pending_recovered_validation,
        )

        await consume_control_requests(rt)
        campaign = check_campaign_boundary(rt)
        if campaign.get("finalize_due") or round_stagnation_status(rt).get("review_capacity_block"):
            # An unchanged Resume must not buy a turn merely to repeat blocked
            # review feedback, including after a crash during the closing commit.
            continue_discovery()
            kickoff = await finish_discovery()
            if exit_after_discovery and not kickoff:
                return
            pending_turn_feedback = None
        else:
            await prepare_report_highlights_async(rt, pending_only=True, console=console)
            pending_turn_feedback = await turn_validation_prompt(
                rt, pending_only=True, console=console)
        if pending_recovered_validation(rt) and not pending_turn_feedback:
            console.print(
                "[cyan]Resuming the interrupted final check before polishing existing "
                "detectors. Saved predictions and completed reviews will be reused.[/cyan]"
            )
            await finalize_discovery(console)
            complete_recovered_validation(rt)
        phase = active_polishing(rt)
        if kickoff and phase and phase.get("transition_pending") and not pending_turn_feedback:
            kickoff = continue_discovery()
            if kickoff is None:
                kickoff = await finish_discovery()
        if kickoff and (refinement := round_refinement_prompt(rt)):
            kickoff = refinement
        if kickoff and initial_stall["should_stop"] and not pending_turn_feedback:
            # Resume uses the durable round checkpoint. Do not pay for
            # another discovery turn when the configured guard already fired.
            if continue_discovery() is None:
                kickoff = await finish_discovery()
        if (kickoff and not pending_turn_feedback
                and not cfg.discovery_wait_for_plateau and run.round >= run.max_rounds):
            from autofa.mcp.tools_library import _pareto_status

            if _pareto_status()["goal_met"]:
                # Resume may already satisfy the stopping condition. Finalize
                # directly rather than paying for another discovery turn.
                next_work = continue_discovery()
                kickoff = next_work if next_work is not None else await finish_discovery()
        if pending_turn_feedback:
            kickoff = pending_turn_feedback + "\n\n" + (kickoff or "Continue the current work.")
        from autofa.session.supervision import check_lifecycle

        check_lifecycle(run_dir)
        if kickoff and (rt.llm_readiness or {}).get("reason") == "campaign_boundary":
            # Startup may have skipped the paid probe for an already closed
            # campaign. A live extension must perform it before discovery starts.
            from autofa.session.llm_readiness import check_and_recover_readiness

            rt.llm_readiness = check_and_recover_readiness(
                rt, console=console, probe=check_llm_readiness)
            if (getattr(run, "require_llm_detectors", False)
                    and rt.llm_readiness.get("available") is not True):
                raise RuntimeError("Required detector endpoint readiness failed: "
                                   + str(rt.llm_readiness.get("reason", rt.llm_readiness)))
        readiness_guidance = (
            "" if (rt.llm_readiness or {}).get("reason") == "campaign_boundary"
            else llm_readiness_guidance(rt.llm_readiness)
        )
        if (kickoff and "ACTIVE CAMPAIGN POLICY" not in kickoff
                and "ACTIVE DISCOVERY GOAL UPDATE" not in kickoff
                and (current_guidance := campaign_guidance(rt))):
            kickoff = current_guidance + "\n\n" + kickoff
        sid = await harness.run_session(
            cwd=str(cwd),
            system_prompt=build_system_prompt(
                top_x=run.top_x,
                max_rounds=run.max_rounds,
                min_failure_modes=run.min_failure_modes,
                exploration_ssot=ssot,
                exploration_seed=seed,
                exploration_templates=templates,
                corpus_domain=domain,
                analysis_guidance=guidance_prompt(run, audience="discovery"),
                objective_config=cfg, run_dir=run_dir,
            ) + readiness_guidance,
            initial_prompt=kickoff,
            mcp_servers={"autofa": mcp} if mcp is not None else {},
            allowed_tools=allowed,
            resume=saved_session_id(),
            discovery_continuation=continue_discovery,
            discovery_finalization=finish_discovery,
            turn_validation=validate_agent_turn,
            exit_after_discovery=exit_after_discovery,
        )
        from autofa.session.detector_service import infrastructure_block

        if publish_on_exit and not infrastructure_block(rt):
            await prepare_report_highlights_async(rt, console=console)
        if sid:
            atomic_write_json(session_path, {"session_id": sid, "harness": cfg.harness})

    async def _drive() -> None:
        kickoff = initial
        while True:
            try:
                await _go(kickoff)
            except CampaignBoundaryReached:
                # Keep a continuation granted during closing validation in the
                # same run/event loop, with the latest durable harness session.
                # Neither a denied dispatch nor a completed result is retried.
                kickoff = continue_discovery()
                if kickoff is None:
                    kickoff = await finish_discovery()
                if kickoff:
                    continue
            return

    completed = False
    try:
        set_action_stage(f"run {HARNESS_NAMES[cfg.harness]} discovery session")
        asyncio.run(_drive())
        completed = True
    finally:
        # Always keep the run directory up to date, and mirror into the durable catalog.
        working = rt.persist_library()
        _flush_quality_snapshot(rt)
        from autofa.config import library_root
        from autofa.data.library_provenance import save_catalog_library
        from autofa.data.remote_library_publish import (
            maybe_publish_run,
            published_library_is_current,
        )

        cat = rt.library.catalog_path(cfg.resolved_library_root() or library_root())
        from autofa.session.detector_service import infrastructure_block

        if (completed and publish_on_exit and infrastructure_block(rt)
                and cfg.remote_library_base_url):
            from autofa.data.discovery_delivery import saved_only_delivery

            try:
                receipt = saved_only_delivery(run, rt.config)
                console.print("Saved partial report delivery: " + str(receipt.get("status")),
                              markup=False)
            except Exception as exc:
                # Optional session-end delivery cannot erase an already saved
                # partial outcome. Required managed delivery has its own receipt.
                console.print("Saved partial report delivery pending ("
                              + type(exc).__name__ + ").", markup=False)
        elif completed and publish_on_exit and published_library_is_current(run, cfg):
            console.print("[dim]Already published; nothing changed since the last save.[/dim]")
        elif completed and publish_on_exit:
            maybe_publish_run(
                run,
                cfg,
                trigger="session_end",
                runtime=rt,
                console=console,
            )
        save_catalog_library(rt.library, cat, run)
        console.print(
            f"[{GREEN}]Saved[/] run library → {working}\n"
            f"[{GREEN}]Saved[/] catalog     → {cat}"
        )


def _flush_quality_snapshot(rt: Any) -> None:
    """Turn/session boundaries publish the latest saved state despite tool throttling."""
    from autofa.session.supervision import save_quality_snapshot

    try:
        save_quality_snapshot(rt, force=True)
    except Exception:
        logging.getLogger(__name__).warning("Unable to refresh saved supervision quality",
                                            exc_info=True)


def write_run_report(run: RunConfig, *, cfg: Config | None = None) -> Path:
    """Render saved results; discovery finalization owns full-corpus scoring."""
    cfg = cfg or load_config()
    rt = load_runtime_from_run(run, cfg)
    rt.persist_library()
    from autofa.config import library_root
    from autofa.data.library_provenance import save_catalog_library

    catalog = rt.library.catalog_path(cfg.resolved_library_root() or library_root())
    save_catalog_library(rt.library, catalog, run)
    from autofa.reports.failure_cards import write_failure_cards_with_status

    return write_failure_cards_with_status(
        rt.library,
        out=run.run_dir() / "failure_cards.html",
        dataset=rt.dataset,
        source_harness=run.source_harness,
        reference_data=run.reference_data,
        console=console,
    )


def require_batch_target(run: RunConfig, *, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    rt = load_runtime_from_run(run, cfg)
    from autofa.session.discovery_stall import stall_termination_allowed
    from autofa.session.discovery_target import failure_mode_target_for_runtime

    target = failure_mode_target_for_runtime(rt)
    target_path = run.run_dir() / "discovery_target.json"
    try:
        previous = json.loads(target_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    target["stalled"] = bool(previous.get("stalled")) and not target["minimum_met"]
    if previous.get("stop_reason") == "infrastructure_blocked":
        from autofa.session.detector_service import infrastructure_block

        if blocked := infrastructure_block(rt):
            target.update(goal_met=False, blocked=True, stop_reason="infrastructure_blocked",
                          service=blocked)
            atomic_write_json(target_path, target)
            return target  # Saved partial delivery remains allowed; the goal is unmet.
    if previous.get("stop_reason") == "stalled":
        # Keep the runtime decision made before final scoring/review. Those
        # stages can improve grades, but do not undo a legitimate cost stop.
        assessment = previous.get("quality_stall") or {}
        target["quality_stall"] = assessment
        if (rt.config.discovery_terminate_if_stalled and stall_termination_allowed(rt)
                and assessment.get("round_stagnation", {}).get("should_stop")
                and assessment.get("enabled")
                and assessment.get("should_stop")):
            target["stop_reason"] = "stalled"
    elif previous.get("stop_reason") == "review_capacity_exhausted":
        from autofa.session.round_stagnation import round_stagnation_status

        assessment = round_stagnation_status(rt)
        if assessment.get("review_capacity_block"):
            target.update(blocked=True, stop_reason="review_capacity_exhausted",
                          quality_stall=assessment)
    if previous.get("campaign_complete") and getattr(run, "campaign_policy", None):
        target.update(campaign_complete=True, stop_reason=previous.get("stop_reason"),
                      campaign=previous.get("campaign"), goal_met=previous.get("goal_met", False))
        atomic_write_json(target_path, target)
        return target  # A bounded campaign can finish honestly below its quality goal.
    atomic_write_json(target_path, target)
    if not target["minimum_met"] and target.get("stop_reason") == "review_capacity_exhausted":
        raise RuntimeError(
            "Batch discovery is blocked by exhausted review capacity: "
            f"{target['n_validated_failure_modes']}/{target['min_failure_modes']} validated. "
            f"Partial artifacts and completed judgments are saved under {run.run_dir()}. "
            "Inspect the saved review workload and unread evidence before arranging a supported "
            "recovery; unchanged Resume cannot replenish the consumed review budget."
        )
    if not target["minimum_met"] and target.get("stop_reason") != "stalled":
        raise RuntimeError(
            "Batch discovery stopped before meeting its failure-mode target: "
            f"{target['n_validated_failure_modes']}/{target['min_failure_modes']} validated. "
            f"Partial artifacts are saved under {run.run_dir()}; resume that run to continue."
        )
    return target
