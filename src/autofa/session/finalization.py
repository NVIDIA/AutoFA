# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finish dataset scoring and causal review before presenting final report scores."""

from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from autofa.ui.evaluation_progress import evaluation_progress
from autofa.util.cancellation import Cancellation, cancellation_scope


class _FinalETA(TimeRemainingColumn):
    def render(self, task: Task) -> Text:
        if task.total is None or (not task.finished and task.speed is None):
            return Text("estimating", style="dim")
        return super().render(task)


def evaluate_final_library(console: Console) -> dict:
    from autofa.mcp.runtime import get_runtime
    from autofa.session.operation_admission import admit_operation
    from autofa.session.supervision import check_lifecycle, lifecycle_cancellation
    from autofa.session.usage import usage_scope

    directory = get_runtime().run.run_dir()
    check_lifecycle(directory, phase="validation")
    # Freeze selection and scoring while honoring stop requests outside commits.
    with (usage_scope(run_dir=directory, phase="finalization"),
          lifecycle_cancellation(directory) as cancellation,
          admit_operation("advance_round", {}) as busy):
        if busy:
            raise RuntimeError("Final validation cannot start: " + busy["error"])
        try:
            return _evaluate_final_library(console)
        except asyncio.CancelledError:
            from autofa.session.supervision import supervisor_cancelled, supervisor_interruption

            if supervisor_cancelled(cancellation):
                raise supervisor_interruption(directory, cancellation) from None
            raise


def _evaluate_final_library(console: Console) -> dict:
    from autofa.mcp.runtime import get_runtime
    from autofa.mcp.tools_eval import final_evaluate_functions
    from autofa.metrics.final_validation_selection import CONTEXT_REFRESH_REASON
    from autofa.session.llm_readiness import recheck_saved_routing
    from autofa.session.validation_recovery import prepare_recovered_validation

    console.print(
        "[cyan]Validating OK-or-better detectors on the full dataset before finalizing "
        "results.[/cyan]\n"
        "[bold]Please wait. Keep this terminal open until scoring and final review finish "
        "and the results are saved.[/bold]\n"
        "[dim]The discovery sample limit no longer applies to selected detectors. "
        "Press Ctrl+C only to cancel "
        "final scoring; checkpoints will be kept for resuming.[/dim]"
    )
    recheck_saved_routing(get_runtime(), console=console)
    _refresh_final_evaluations(console)
    prepare_recovered_validation(get_runtime())
    selection = _prepare_final_selection(console)
    # A preceding detector's review can establish a previously unknown classifier API.
    # Recompute stale aggregates in their original scopes before reconsidering admission.
    stale = {row["name"] for row in selection.get("unresolved", [])
             if row.get("reason") == CONTEXT_REFRESH_REASON}
    if stale and (refreshed := _refresh_final_evaluations(console, names=stale)):
        retry = _prepare_final_selection(console, names=refreshed)
        selection["selected_names"].extend(retry["selected_names"])
        selection["skipped"].extend(retry["skipped"])
        selection["unresolved"] = [row for row in selection["unresolved"]
                                   if row["name"] not in refreshed] + retry["unresolved"]
    # The harness has closed its Live display by this point. Own the scoring
    # display here so progress remains visible for every harness and /stop path.
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]"), BarColumn(),
        MofNCompleteColumn(), TextColumn("predictions"), TimeElapsedColumn(),
        TextColumn("ETA"), _FinalETA(), TextColumn("{task.fields[detector]}"),
        console=console, refresh_per_second=4,
    ) as progress:
        task_id = progress.add_task("Preparing final scoring", total=None, detector="")

        def update(event: dict) -> None:
            rows = event["rows"]
            denominator = (f"{'~' if event['estimated'] else ''}{rows}"
                           if rows is not None else "?")
            detector = event["detector"]
            detail = (f"{detector} ({event['detector_completed']}/{denominator} examples)"
                      if detector else "")
            progress.update(
                task_id, total=event["total"], completed=event["completed"],
                description="Saving scores" if event["phase"] == "saving" else "Final scoring",
                detector=detail,
            )

        with evaluation_progress(update):
            result = final_evaluate_functions(names=selection["selected_names"])
        if result.get("ok") and result.get("n") == 0:
            progress.update(task_id, description="No new full scans required", total=0, completed=0)
    if not result.get("ok"):
        raise RuntimeError("Final full-dataset evaluation is incomplete: "
                           + str(result.get("error") or "inspect the saved detector scores"))
    console.print(
        f"[green]Final detector evaluation complete[/green] "
        f"[dim]({result.get('n', 0)} detector(s) required a full scan).[/dim]"
    )
    review = _review_final_library(console, names=selection["selected_names"],
                                  selection_unresolved=selection.get("unresolved"))
    from autofa.session.validation_completion import validated_good_target_for_runtime
    from autofa.session.validation_summary import save_validation_summary

    try:
        rt = get_runtime()
        save_validation_summary(rt, completion=validated_good_target_for_runtime(rt))
    except Exception:
        # Reporting happens after the measurements have committed. A diagnostic
        # write/projection failure must not turn them into failed validation.
        logging.getLogger(__name__).warning("Unable to write validation summary", exc_info=True)
        console.print("[yellow]Closing summary unavailable; detector scores and review "
                      "results remain saved in the library.[/yellow]")
    from autofa.util.atomic import atomic_write_json

    outcome = {
        "version": 1, "selected": selection["selected_names"],
        "skipped": selection.get("skipped", []), "scoring": result, "review": review,
        "retained_results_complete": bool(review.get("ok")) and not any(
            row.get("name") in selection["selected_names"]
            for row in review.get("unresolved", [])),
        "state": "completed_with_gaps" if not review.get("ok") or review.get("unresolved")
                 else "complete",
        "report_generated": False,
    }
    atomic_write_json(get_runtime().run.run_dir() / "finalization_outcome.json", outcome)
    return outcome


def _refresh_final_evaluations(console: Console, *, names: set[str] | None = None) -> list[str]:
    from autofa.mcp.runtime import get_runtime
    from autofa.session.evaluation_refresh import refresh_evaluation_contexts

    console.print("[cyan]Checking saved detector scores; refreshes reuse saved predictions "
                  "where available.[/cyan]")
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]"), BarColumn(),
        MofNCompleteColumn(), TextColumn("predictions"), TimeElapsedColumn(),
        TextColumn("ETA"), _FinalETA(), TextColumn("{task.fields[detector]}"),
        console=console, refresh_per_second=4,
    ) as progress:
        task_id = progress.add_task("Checking saved scores", total=None, detector="")

        def update(event: dict) -> None:
            progress.update(
                task_id, total=event["total"], completed=event["completed"],
                detector=event.get("detector", ""),
                description=("Saving refreshed scores" if event["phase"] == "saving"
                             else "Refreshing saved scores"),
            )

        result = refresh_evaluation_contexts(get_runtime(), progress=update, names=names)
        if not result["refreshed_names"]:
            progress.update(task_id, description="Saved scores checked", total=0, completed=0)
        return result["refreshed_names"]


def _prepare_final_selection(console: Console, *, names: list[str] | None = None) -> dict:
    from autofa.mcp.runtime import get_runtime
    from autofa.metrics.final_validation_selection import prepare_final_validation

    console.print("[cyan]Selecting final candidates; resolving promising pending reviews "
                  "within their existing sample budget.[/cyan]")
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]"), BarColumn(),
        MofNCompleteColumn(), TextColumn("detectors"), TimeElapsedColumn(),
        TextColumn("{task.fields[detector]}"), console=console, refresh_per_second=4,
    ) as progress:
        task_id = progress.add_task("Checking eligibility", total=None, detector="")

        def update(event: dict) -> None:
            progress.update(task_id, total=event["total"], completed=event["completed"],
                            detector=event.get("detector", ""))

        selection = prepare_final_validation(get_runtime(), progress=update, names=names)
    skipped = selection.get("skipped") or []
    console.print(Text(
        f"Full validation: {len(selection['selected_names'])} selected; {len(skipped)} skipped. "
        "Skipped candidates retain their measured scores and selection reasons.", style="dim",
    ))
    return selection


def _review_final_library(
    console: Console, *, names: list[str] | None = None,
    selection_unresolved: list[dict] | None = None,
) -> dict:
    from autofa.metrics.final_review import final_review_functions

    console.print("[cyan]Checking final detector reviews; completed judgments are reused.[/cyan]")
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]"), BarColumn(),
        MofNCompleteColumn(), TextColumn("detectors"), TimeElapsedColumn(),
        TextColumn("ETA"), _FinalETA(), TextColumn("{task.fields[detector]}"),
        console=console, refresh_per_second=4,
    ) as progress:
        task_id = progress.add_task("Final review", total=None, detector="")

        def update(event: dict) -> None:
            progress.update(
                task_id, total=event["total"], completed=event["completed"],
                detector=event.get("detector", ""),
            )

        result = final_review_functions(progress=update, names=names)
    unresolved = [*(result.get("unresolved") or []), *(selection_unresolved or [])]
    if unresolved or not result.get("ok"):
        console.print(
            "[yellow]Final scores saved with unresolved review or evidence gaps. "
            "The report identifies the unavailable scores and their reasons.[/yellow]"
        )
        for row in unresolved:
            console.print(Text(f"  {row['name']}: {row['reason']}", style="yellow"))
        if not unresolved:
            console.print(Text(str(result.get("error") or "Final review did not complete."),
                               style="yellow"))
        return {**result, "unresolved": unresolved}
    console.print(
        "[green]Final validation complete for selected detectors.[/green] "
        "[dim]Validated results and retained scores with their evaluation scope are saved; "
        "/library or the report shows the final results.[/dim]"
    )
    return {**result, "unresolved": []}


async def finalize_discovery(console: Console) -> None:
    """Keep the interactive prompt cancellable while scoring in a worker."""
    from autofa.session.supervision import (
        SupervisorInterrupted,
        supervisor_cancelled,
        supervisor_interruption,
    )

    def finish():
        # KeyboardInterrupt subclasses cannot escape a task created by to_thread:
        # asyncio would tear down the loop before its awaiting caller can handle it.
        try:
            return evaluate_final_library(console)
        except SupervisorInterrupted as exc:
            return exc

    cancellation = Cancellation()
    with cancellation_scope(cancellation):
        worker = asyncio.create_task(asyncio.to_thread(finish))
    try:
        result = await asyncio.shield(worker)
        if isinstance(result, SupervisorInterrupted):
            raise result
    except asyncio.CancelledError:
        if supervisor_cancelled(cancellation):
            from autofa.mcp.runtime import get_runtime

            raise supervisor_interruption(get_runtime().run.run_dir(), cancellation) from None
        # Drain the bounded in-flight work once. Another cancellation propagates;
        # finalization must never turn repeated Ctrl+C into a retry loop.
        try:
            await asyncio.to_thread(cancellation.cancel)
            await asyncio.shield(worker)
        except (asyncio.CancelledError, Exception):
            pass
        raise
