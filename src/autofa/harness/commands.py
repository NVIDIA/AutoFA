# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local slash-command handling (before Agent SDK)."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autofa.metrics.verdicts import verdict_legend
from autofa.ui.theme import GREEN

# Slash commands shown in /help and used for prompt autocomplete.
SESSION_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show this help"),
    ("/discover", "Find top failure modes, write detectors, then score them"),
    ("/validate", "Score detectors and run the behavior judge"),
    ("/compact", "Compact the current conversation using the selected harness"),
    ("/dashboard", "Interactive explorer (detectors → buckets → examples)"),
    ("/diagnostics", "Show data-health diagnostics for this run"),
    ("/clean", "Clear working library (/clean <name> removes one)"),
    ("/stop", "How to interrupt a running turn (Ctrl+C)"),
    ("/library", "List detectors and scores"),
    ("/save", "Save locally and publish the report to the configured remote library"),
    ("/goal", "Check Pareto goal (ppv@π₀ × recall × behavior_purity)"),
    ("/exit", "Save, end session, and publish to the configured remote library"),
    ("/quit", "Save, end session, and publish to the configured remote library"),
]


def session_command_names() -> list[str]:
    return [c for c, _ in SESSION_COMMANDS]


class SlashCommandCompleter(Completer):
    """Tab-complete slash commands; for `/clean ` also complete function names."""

    def __init__(self, commands: Iterable[str] | None = None) -> None:
        self.commands = list(commands or session_command_names())

    def get_completions(self, document, complete_event):  # noqa: ANN001
        text = document.text_before_cursor
        body = text.lstrip()
        # Complete on empty prompt, "/", or "/dash…"; skip plain English.
        if body and not body.startswith("/"):
            return

        if " " in body:
            cmd, _, rest = body.partition(" ")
            if cmd.lower() in {"/clean", "clean"}:
                prefix = rest
                for name in _library_function_names():
                    if name.lower().startswith(prefix.lower()):
                        yield Completion(
                            name,
                            start_position=-len(prefix),
                            display_meta="remove",
                        )
            return

        prefix = body
        for cmd in self.commands:
            if cmd.lower().startswith(prefix.lower()):
                meta = next((d for c, d in SESSION_COMMANDS if c == cmd), "")
                yield Completion(
                    cmd,
                    start_position=-len(prefix),
                    display_meta=meta,
                )


def _library_function_names() -> list[str]:
    try:
        from autofa.mcp.runtime import get_runtime

        return [fn.name for fn in get_runtime().library.functions]
    except Exception:  # noqa: BLE001
        return []


def handle_local_command(text: str, console: Console) -> str | None:
    """Return ``"exit"`` to leave, ``"handled"``, or None to pass to the agent."""
    raw = text.strip()
    if not raw:
        return None
    parts = raw.split()
    cmd = parts[0].lower()

    if cmd in {"/exit", "/quit", "exit", "quit"}:
        return "exit"

    if cmd in {"/help", "help", "?", "/?"}:
        _print_help(console)
        return "handled"

    if cmd in {"/dashboard", "dashboard"}:
        _open_dashboard(console)
        return "handled"

    if cmd in {"/clean", "clean"}:
        _clean_library(console, parts[1:] if len(parts) > 1 else [])
        return "handled"

    if cmd in {"/save", "save"}:
        _save_library(console)
        return "handled"

    if cmd in {"/goal", "goal"}:
        _show_goal(console)
        return "handled"

    if cmd in {"/library", "library"}:
        _show_library(console)
        return "handled"

    if cmd in {"/diagnostics", "diagnostics"}:
        _show_diagnostics(console)
        return "handled"

    if cmd in {"/stop", "stop"}:
        console.print(
            Panel(
                "While the agent is running, press [bold]Ctrl+C[/] once to cancel that turn.\n"
                "Then you can ask another question, run `/clean`, `/dashboard`, etc.\n\n"
                "[dim]You cannot type a new prompt mid-turn — discovery blocks the input "
                "until it finishes or you interrupt.[/dim]",
                title=f"[bold {GREEN}]/stop[/]",
                border_style=GREEN,
            )
        )
        return "handled"

    return None


def _clean_library(console: Console, args: list[str]) -> None:
    try:
        from autofa.mcp import tools_library
        from autofa.mcp.runtime import get_runtime

        rt = get_runtime()
        if args:
            name = args[0]
            out = tools_library.remove_function(
                name, reason="user /clean", status="rejected"
            )
            if out.get("ok"):
                console.print(
                    f"[{GREEN}]Removed[/] {name}  ·  {out.get('n_functions', 0)} left"
                )
            else:
                console.print(f"[yellow]{out.get('error')}[/yellow]")
            return
        n = len(rt.library.functions)
        if n == 0:
            console.print("[dim]Library already empty.[/dim]")
            return
        out = tools_library.clear_library(confirm=True)
        console.print(
            f"[{GREEN}]Cleared[/] {out.get('cleared', n)} detectors from the working library."
        )
        console.print(f"[dim]Saved → {rt.working_library_path()}[/dim]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Clean failed: {e}[/yellow]")


def _save_library(console: Console) -> None:
    try:
        from autofa.mcp import tools_library
        from autofa.mcp.runtime import get_runtime
        from autofa.session.report_highlights import prepare_report_highlights

        out = tools_library.library_save(to_catalog=True)
        if not out.get("ok"):
            console.print(f"[yellow]Save failed: {out.get('error')}[/yellow]")
            return
        prepared = prepare_report_highlights(get_runtime(), console=console)
        if prepared.get("status") != "not_required":
            out = tools_library.library_save(to_catalog=True)
            if not out.get("ok"):
                console.print(f"[yellow]Save failed: {out.get('error')}[/yellow]")
                return
        saved = out.get("saved") or []
        console.print(f"[{GREEN}]Saved[/] {len(saved)} path(s):")
        for p in saved:
            console.print(f"  [dim]·[/dim] {p}")
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Save failed: {e}[/yellow]")
        return

    try:
        from autofa.config import library_root
        from autofa.data.library_provenance import save_catalog_library
        from autofa.data.remote_library_publish import maybe_publish_run
        from autofa.mcp.runtime import get_runtime

        rt = get_runtime()
        # /save is an explicit manual publication. Reuse the existing wire value
        # so cloud and older configured servers accept its metadata.
        remote = maybe_publish_run(
            rt.run, rt.config, trigger="manual_cli", runtime=rt, console=console,
        )
        if remote is not None:
            # Publication assigns library/branch/version IDs. Keep Continue from
            # the local catalog attached to that newly published version.
            cat = rt.library.catalog_path(rt.config.resolved_library_root() or library_root())
            save_catalog_library(rt.library, cat, rt.run)
    except Exception as e:  # noqa: BLE001
        console.print(
            "[yellow]Local library saved, but remote publication or catalog sync failed: "
            f"{e}[/yellow]"
        )


def _show_diagnostics(console: Console) -> None:
    try:
        from autofa.diagnostics import get_diagnostics
        from autofa.mcp.runtime import get_runtime

        diag = get_diagnostics()
        try:
            rt = get_runtime()
            diag.configure(rt.run.run_dir())
        except Exception:  # noqa: BLE001
            pass
        summary = diag.summary()
        lines = diag.preflight_lines()
        if not lines:
            console.print(
                Panel(
                    "[dim]No diagnostics recorded yet for this process.\n"
                    f"Log path: {summary.get('log_path') or '(not configured)'}[/dim]",
                    title=f"[bold {GREEN}]/diagnostics[/]",
                    border_style=GREEN,
                )
            )
            return
        body = "\n".join(lines[:40])
        footer = (
            f"\n\n[dim]{summary['n_critical']} critical · "
            f"{summary['n_warning']} warning · {summary['n_info']} info"
            f"\nlog: {summary.get('log_path') or '—'}[/dim]"
        )
        console.print(
            Panel(
                body + footer,
                title=f"[bold {GREEN}]/diagnostics[/]",
                border_style=GREEN,
            )
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Diagnostics unavailable: {e}[/yellow]")


def _show_library(console: Console) -> None:
    try:
        from autofa.behavior_policy import strictness_for_config
        from autofa.mcp.runtime import get_runtime
        from autofa.metrics.verdicts import format_metrics_line, verdict_for_scores
        from autofa.reports.ordering import rank_display_functions
        from autofa.ui.theme import verdict_color

        rt = get_runtime()
        fns = rank_display_functions(
            [vars(fn) for fn in rt.library.functions],
            behavior_strictness=strictness_for_config(rt.config),
            review_history=rt.library.review_history,
        )
        if not fns:
            console.print("[dim]Library empty — run /discover first.[/dim]")
            return
        table = Table(
            show_header=True,
            header_style=f"bold {GREEN}",
            box=None,
            padding=(0, 1),
            title=f"[bold]Library[/] · {rt.library.model_id or '?'} / "
            f"{rt.library.benchmark or 'all'} · {len(fns)} fn",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("name", min_width=20)
        table.add_column("verdict", min_width=15, no_wrap=True)
        table.add_column("metrics")
        for i, fn in enumerate(fns, 1):
            scores = fn["scores"]
            v = verdict_for_scores(scores)
            color = verdict_color(v["code"])
            table.add_row(
                str(i),
                fn["name"],
                f"[{color}]{v['symbol']} {v['label']}[/{color}]",
                format_metrics_line(scores, short=True),
            )
        console.print(table)
        console.print(
            "[dim]/dashboard for buckets · /save and publish · /goal for Pareto status[/dim]"
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Library listing failed: {e}[/yellow]")


def _show_goal(console: Console) -> None:
    try:
        from autofa.mcp import tools_library
        from autofa.mcp.runtime import get_runtime
        from autofa.metrics.live_progress import runtime_display_progress_snapshot
        from autofa.metrics.progress import render_terminal_progress

        out = tools_library._pareto_status()
        if not out.get("ok", True) and out.get("error"):
            console.print(f"[yellow]{out.get('error')}[/yellow]")
            return

        rt = get_runtime()
        goal_met = bool(out.get("goal_met"))
        improved = out.get("improved")
        round_i = out.get("round", rt.run.round)
        max_r = out.get("max_rounds", rt.run.max_rounds)
        top = out.get("top_x") or []
        reason = out.get("reason")
        since = out.get("experiments_since_improve")
        patience = out.get("patience")

        body = Table.grid(padding=(0, 1))
        body.add_column(style="dim", width=18)
        body.add_column()
        if reason and not out.get("current_front"):
            body.add_row("status", reason)
        else:
            status = (
                f"[{GREEN}]goal met[/{GREEN}]"
                if goal_met
                else "[yellow]keep iterating[/yellow]"
            )
            body.add_row("status", status)
            if improved is True:
                body.add_row("task coverage", f"[{GREEN}]confirmed coverage increased[/{GREEN}]")
            elif improved is False:
                body.add_row("task coverage", "no additional confirmed coverage")
            else:
                body.add_row("task coverage", "n/a (baseline assessment)")
        body.add_row("round", f"{round_i} / {max_r}")
        minimum = out.get("min_failure_modes", 0)
        if minimum:
            body.add_row("Good detectors",
                         f"{out.get('n_validated_failure_modes', 0)} / {minimum}")
            if not out.get("minimum_met"):
                body.add_row("minimum", "Continue past the round budget until this target is met.")
        if out.get("n_functions") is not None:
            body.add_row("n_functions", str(out["n_functions"]))
        if since is not None and patience is not None:
            body.add_row("since_improve", f"{since} / patience {patience}")
        if top:
            body.add_row("top_x", ", ".join(str(x) for x in top))

        title = f"[bold {GREEN}]/goal[/]"
        if goal_met:
            subtitle = "[dim]recommend /save · /dashboard — or keep going if budget remains[/dim]"
        else:
            subtitle = "[dim]pivot within this round · finish its comparison batch to advance[/dim]"
        console.print(Panel(body, title=title, border_style=GREEN, subtitle=subtitle))
        render_terminal_progress(
            rt.scores_history, console, title="frontier over time",
            current=runtime_display_progress_snapshot(rt),
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Goal check failed: {e}[/yellow]")


def _reload_ui_modules() -> Any:
    """Reload explorer/verdicts so a long-lived session picks up code edits."""
    import autofa.metrics.verdicts as verdicts
    import autofa.ui.explorer as explorer
    import autofa.ui.syntax as syntax
    import autofa.ui.theme as theme

    # theme binds verdict presentations at import, so reload verdicts first.
    importlib.reload(verdicts)
    importlib.reload(theme)
    importlib.reload(syntax)
    return importlib.reload(explorer)


def _open_dashboard(console: Console) -> None:
    try:
        from autofa.mcp.runtime import get_runtime

        explorer = _reload_ui_modules()
        rt = get_runtime()
        title = None
        try:
            title = rt.run.explorer_title()
        except Exception:  # noqa: BLE001
            title = (
                f"{rt.library.model_id or '?'} / {rt.library.benchmark or 'all'}"
            )
        html_path = rt.run.run_dir() / "failure_cards.html"
        explorer.explore_library(
            rt.library,
            top_x=None,
            title=title,
            html_path=html_path,
            dataset=rt.dataset,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Dashboard unavailable: {e}[/yellow]")


def _print_help(console: Console) -> None:
    table = Table(show_header=True, header_style=f"bold {GREEN}", box=None, padding=(0, 2))
    table.add_column("Command", style=f"bold {GREEN}")
    table.add_column("Description")
    for a, b in SESSION_COMMANDS:
        if a == "/quit":
            continue
        table.add_row(a, b.replace("Ctrl+C", "[bold]Ctrl+C[/bold]"))

    console.print(
        Panel(
            table,
            title=f"[bold {GREEN}]AutoFA help[/]",
            border_style=GREEN,
            subtitle="[dim]%caught · precision · recall · ppv@π₀ · behavior_purity[/dim]",
        )
    )
    console.print(
        "[dim]bp = behavior_purity (selected judge harness). "
        f"{verdict_legend()} "
        "[bold]Ctrl+C[/bold] stops a running turn.[/dim]"
    )
