# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoFA CLI entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

from rich.console import Console
from rich.text import Text

from autofa.config import (
    Config,
    RunConfig,
    ensure_default_config,
    list_runs,
    load_config,
    load_dotenv,
    runs_root,
    validate_analysis_outputs,
    validate_min_failure_modes,
)
from autofa.core.library import FunctionLibrary
from autofa.data.efficiency_support import EfficiencyGate, check_prepared, efficiency_gate
from autofa.data.library_provenance import read_catalog_provenance
from autofa.data.remote_library_api import (
    RemoteLibraryRef,
    RemotePublication,
    client_from_config,
    library_service_name,
    materialize_remote_library,
)
from autofa.harness.settings import HarnessNotFoundError
from autofa.mcp.runtime import load_runtime_from_run
from autofa.run_config import AutoFAConfig, AutoFAExecution
from autofa.services import remote_library_enabled
from autofa.ui.banner import home_menu_header
from autofa.ui.errors import (
    action_outcome,
    begin_action,
    cancel_action,
    end_action,
    notice_fragments,
    set_action_stage,
    show_action_error,
    show_action_notice,
)
from autofa.ui.library_browser import browse_library, select_remote_library
from autofa.ui.menus import (
    MenuItem,
    browse_filesystem,
    is_menu_back,
    select_option,
    select_yes_no,
)
from autofa.wizard.dataset import reuse_reference_data, run_dataset_wizard

console = Console()


@dataclass(frozen=True)
class ContinueSeed:
    """Materialized Continue seed plus its complete publication lineage."""

    path: str
    remote_library_id: str | None = None
    remote_branch_id: str | None = None
    remote_branch_key: str | None = None
    remote_version_id: str | None = None
    remote_base_version_id: str | None = None
    source_run_id: str | None = None
    source_harness: str = "unknown"
    reference_data: tuple[dict[str, str], ...] = ()
    selection_id: str | None = None


def _minimum_failure_modes_arg(value: str) -> int:
    try:
        return validate_min_failure_modes(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "minimum failure modes must be a non-negative integer (0 disables it)"
        ) from exc


def _top_x_arg(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-x must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("top-x must be a positive integer")
    return number


def _has_cli_data_selection(args) -> bool:
    return any((
        getattr(args, "input", None),
        getattr(args, "api_run", None),
        getattr(args, "model", None),
        getattr(args, "benchmark", None),
        getattr(args, "runs_where", ""),
        getattr(args, "latest", 0),
        getattr(args, "task", None),
        getattr(args, "reward", None),
        getattr(args, "where", ""),
        getattr(args, "limit", 0),
        getattr(args, "per_run", 0),
        getattr(args, "sample", "first") != "first",
        getattr(args, "seed", 0),
        getattr(args, "judge_view", None),
        getattr(args, "group", "all") != "all",
        getattr(args, "label", None),
        getattr(args, "evidence", None),
        getattr(args, "list_taxonomy", False),
    ))


def _rounds_arg(value: str) -> int:
    if len(value) > 9 or not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("rounds must be a whole number from 1 to 999999999")
    return int(value)


def main(argv: list[str] | None = None) -> None:
    env_path = load_dotenv()
    incoming = list(sys.argv[1:] if argv is None else argv)
    if incoming[:1] == ["config"]:
        from autofa.setup_contract import main as config_main

        raise SystemExit(config_main(incoming[1:]))
    if incoming[:1] == ["publish"] and any(
        flag in incoming for flag in ("--saved-only", "--verify-only", "--json")
    ):
        from autofa.data.discovery_delivery import main as delivery_main

        raise SystemExit(delivery_main(incoming[1:]))
    if incoming[:1] == ["supervise"]:
        from autofa.supervise_cli import main as supervise_main

        raise SystemExit(supervise_main(incoming[1:]))
    if incoming[:1] == ["prepare"]:
        from autofa.data.cohort import main as prepare_main

        raise SystemExit(prepare_main(incoming[1:]))
    if incoming[:1] == ["classify"]:
        from autofa.taxonomy.cli import main as classify_main

        raise SystemExit(classify_main(incoming[1:]))
    if incoming[:1] == ["preflight"]:
        from autofa.preflight.cli import main as preflight_main

        raise SystemExit(preflight_main(incoming[1:]))
    parser = argparse.ArgumentParser(
        prog="autofa",
        description="AutoFA — automated failure discovery",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="home",
        help="home|new|run|continue|resume|library|dashboard|settings|publish|serve"
             "|login|logout|prepare|classify|preflight|supervise|config",
    )
    parser.add_argument(
        "--config",
        dest="run_config_path",
        default=None,
        help=(
            "YAML run configuration for autofa run, or classification results "
            "handoff for New/Continue"
        ),
    )
    parser.add_argument("--run", dest="run_id", default=None)
    parser.add_argument("--library", dest="library_path", default=None)
    parser.add_argument(
        "--remote-library",
        dest="remote_library_id",
        default=None,
        help="Remote library id to use as the Continue seed",
    )
    parser.add_argument(
        "--remote-branch",
        dest="remote_branch_id",
        default=None,
        help="Remote branch id (defaults to the logical library's latest branch)",
    )
    parser.add_argument(
        "--remote-version",
        dest="remote_version_id",
        default=None,
        help="Immutable remote version id (defaults to the branch's latest version)",
    )
    from autofa.data.cohort import add_cohort_arguments, add_taxonomy_filter_arguments

    add_cohort_arguments(parser, taxonomy_inputs=True)
    add_taxonomy_filter_arguments(parser)
    parser.add_argument("--optimization", choices=("balanced", "precision", "recall"),
                        default=None, help="Objective-weight preset for New/Continue")
    parser.add_argument("--rounds", type=_rounds_arg, default=None,
                        help="Total discovery-round target for New/Continue/Resume")
    parser.add_argument(
        "--api-base",
        dest="api_base_url",
        default=None,
        help="Override EvalRDL server base URL (default: config data_api_base_url)",
    )
    parser.add_argument(
        "--refresh-api", action="store_true",
        help="Redownload selected API runs and rebuild their trace caches for New/Continue",
    )
    parser.add_argument("--no-auto", action="store_true", help="Do not auto-start /discover")
    parser.add_argument(
        "--analysis-output",
        dest="analysis_outputs",
        action="append",
        choices=("efficiency", "agent"),
        default=None,
        help=("Override enabled analyses from Settings (initially failure discovery only); "
              "repeat to select both."),
    )
    parser.add_argument(
        "--min-failure-modes", type=_minimum_failure_modes_arg, default=None,
        help=("Minimum Good detectors (sampled grades may be provisional); extend discovery "
              "beyond rounds until reached (default: config, 0 disables; explicit budgets apply)"),
    )
    parser.add_argument(
        "--top-x",
        type=_top_x_arg,
        default=None,
        help="Candidate failure modes retained per discovery round (default: config)",
    )
    parser.add_argument(
        "--analysis-focus", choices=("model", "harness", "both"), default=None,
        help="Discovery focus for New/Continue (Resume preserves the saved scope)",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Cap rows for evaluate (debug)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one command and exit (no return to the home menu)",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port for autofa serve (automatic startup uses local_library_base_url)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Address autofa serve listens on; remote connections require the server token",
    )
    args = parser.parse_args(argv)
    cmd = args.command or "home"
    if cmd not in {"home", "new", "run", "continue", "resume", "library", "dashboard",
                   "settings", "publish", "serve", "login", "logout"}:
        parser.error(f"Unknown command: {cmd}")
    if args.rounds is not None:
        if cmd not in {"new", "continue", "resume"}:
            parser.error("--rounds applies to New/Continue/Resume only")
    if cmd == "run" and not args.run_config_path:
        parser.error("run requires --config")
    if args.run_config_path and cmd not in {"run", "new", "continue"}:
        parser.error("--config applies to Run or taxonomy New/Continue handoff")
    if args.run_config_path and cmd in {"new", "continue"}:
        if args.input:
            parser.error("--config supplies taxonomy result inputs; do not also pass --input")
        try:
            args.input = _taxonomy_handoff_inputs(args.run_config_path)
        except ValueError as exc:
            parser.error(str(exc))
    data_flags = _has_cli_data_selection(args)
    if data_flags and cmd not in {"new", "continue"}:
        parser.error("data selection flags apply only to autofa new or autofa continue")
    if cmd not in {"new", "continue", "home"} and any(
        getattr(args, key) is not None
        for key in ("analysis_focus", "optimization")
    ):
        parser.error(
            "Analysis overrides apply to New/Continue; Resume uses the saved scope"
        )

    try:
        cfg_path = (ensure_default_config(require_harness=False) if cmd == "serve"
                    else ensure_default_config())
    except HarnessNotFoundError as exc:
        parser.exit(1, f"autofa: {exc}\n")

    interactive = cmd in {
        None,
        "home",
        "new",
        "continue",
        "resume",
        "library",
        "dashboard",
        "settings",
    }
    status_lines: list[str] = []
    if interactive:
        cfg = load_config()
        from autofa.objectives import objective_summary
        from autofa.ui.settings import scoring_summary
        from autofa.util.memory import apply_memory_tuning

        tune = apply_memory_tuning(cfg)
        if env_path:
            status_lines.append(f"Loaded env: {env_path}")
        status_lines.append(f"Config: {cfg_path}")
        status_lines.append(scoring_summary(cfg))
        status_lines.append(
            f"detector_fields={cfg.detector_fields}  objective={objective_summary(cfg)}"
        )
        status_lines.append(tune.summary())

    with ExitStack() as lifecycle:
        if cmd not in {"serve", "login", "logout"}:
            from autofa.local_service import dashboard_url_from_config

            try:
                cfg = load_config()
                url = dashboard_url_from_config(cfg)
                from autofa_library_server.local import managed_dashboard

                owned = lifecycle.enter_context(managed_dashboard(url=url))
                args._local_library_url = url
                lifetime = "closes with this AutoFA" if owned else "using existing server"
                message = f"AutoFA Local Library Server: {url} ({lifetime})"
                status_lines.append(message)
                if cmd != "home":
                    text = Text(message)
                    text.stylize(f"link {url}", 0, message.index(" ("))
                    console.print(text)
            except Exception as exc:
                message = f"Local dashboard unavailable: {exc}"
                status_lines.append(message)
                console.print(message, style="yellow", markup=False)
        # Start the local server before cloud sign-in or identity network calls.
        if interactive and _is_default_tty_home(args):
            from autofa.local_service import is_local_service_url

            cfg = load_config()
            if remote_library_enabled(cfg) and (
                cfg.remote_library_mode == "server"
                or not is_local_service_url(cfg.remote_library_base_url)
            ):
                status_lines.append(_home_cloud_status_line(cfg))
        _run_command(cmd, args, interactive=interactive, status_lines=status_lines)


def _run_command(cmd, args, *, interactive: bool, status_lines: list[str]) -> None:
    if cmd == "home":
        _home(args, status_lines=status_lines)
        return

    # One-shot commands and menu actions share the same failure boundary.
    return_to_menu = not args.once and interactive and sys.stdin.isatty() and sys.stdout.isatty()
    state = _execute_action(cmd, lambda: _dispatch(cmd, args), wait=return_to_menu)
    if state in {"failed", "stopped_by_supervisor", "campaign_boundary"} and not return_to_menu:
        raise SystemExit(1)
    if state == "interrupted" and not return_to_menu:
        raise SystemExit(130)
    if return_to_menu:
        _home(args, status_lines=status_lines)


def _dispatch(cmd, args) -> None:
    if cmd == "new":
        _new(args)
    elif cmd == "run":
        _run_config(args)
    elif cmd == "continue":
        _continue(args)
    elif cmd == "resume":
        _resume(args)
    elif cmd == "settings":
        from autofa.ui.settings import run_settings

        run_settings()
    elif cmd == "library":
        browse_library()
    elif cmd == "dashboard":
        _dashboard(args)
    elif cmd == "publish":
        _publish(args)
    elif cmd == "serve":
        from autofa_library_server.local import serve

        serve(port=args.port, host=args.host)
    elif cmd == "login":
        _login()
    elif cmd == "logout":
        _logout()
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        sys.exit(2)


def _execute_action(name: str, callback, *, wait: bool = True) -> str:
    from autofa.session.campaign_policy import CampaignBoundaryReached
    from autofa.session.supervision import SupervisorInterrupted

    token = begin_action(name, console=console)
    state = "failed"
    try:
        callback()
        state = action_outcome()
        if not notice_fragments():
            show_action_notice(f"{name} finished.", console=console, warning=False)
    except CampaignBoundaryReached as exc:
        state = "campaign_boundary"
        show_action_notice(f"{name} reached the campaign limit: {exc.reason}. "
                           "Saved results remain available.", console=console)
    except SupervisorInterrupted as exc:
        state = "stopped_by_supervisor"
        show_action_notice(str(exc), console=console)
    except KeyboardInterrupt:
        state = "interrupted"
        show_action_notice(f"{name} interrupted by Ctrl+C.", console=console)
    except SystemExit as exc:
        if exc.code is None or exc.code == 0:
            state = "cancelled"
            show_action_notice(f"{name} cancelled.", console=console, warning=False)
        else:
            show_action_error(exc, console=console, wait=wait)
    except Exception as exc:  # noqa: BLE001
        show_action_error(exc, console=console, wait=wait)
    finally:
        end_action(token, console=console, state=state)
    return state


def _is_default_tty_home(
    args,
    *,
    stdin=None,
    stdout=None,
) -> bool:
    """Gate onboarding/status to a direct interactive home launch."""
    if (getattr(args, "command", None) or "home") != "home":
        return False
    if bool(getattr(args, "once", False)):
        return False
    return _is_interactive_terminal(stdin=stdin, stdout=stdout)


def _is_interactive_terminal(*, stdin=None, stdout=None) -> bool:
    """Return whether both CLI streams are attached to a terminal."""
    input_stream = stdin if stdin is not None else sys.stdin
    output_stream = stdout if stdout is not None else sys.stdout
    try:
        return bool(input_stream.isatty() and output_stream.isatty())
    except (AttributeError, OSError):
        return False


def _home_cloud_status_line(cfg: Config) -> str:
    from autofa.data.remote_library_onboarding import home_cloud_status

    try:
        return home_cloud_status(
            cfg,
            prompt_fn=select_yes_no,
            console=console,
        )
    except Exception:  # noqa: BLE001
        # Cloud onboarding/status must never prevent local CLI use.
        return (
            "AutoFA Cloud Library: unavailable "
            "· launch autofa login first to log in"
        )


def _login() -> None:
    from autofa.data.remote_library_auth import login_to_remote_library

    try:
        login_to_remote_library(load_config(), console=console)
    except KeyboardInterrupt:
        cancel_action("Login cancelled; no credential was saved.")


def _logout() -> None:
    from autofa.data.remote_library_auth import logout_from_remote_library

    logout_from_remote_library(load_config(), console=console)


def _home(args, *, status_lines: list[str] | None = None) -> None:
    """Persistent home menu (logo + options) until Quit."""
    from autofa.ui.settings import model_summary, run_settings, scoring_summary

    while True:
        cfg = load_config()
        current_status = [line for line in (status_lines or []) if (
            remote_library_enabled(cfg) or not line.startswith(("AutoFA Cloud Library:",
                                                                "AutoFA server:"))
        )]
        current_status = [
            scoring_summary(cfg) if line.startswith("Iterative LLM scoring:") else line
            for line in current_status
        ]
        remote_url = str(cfg.remote_library_base_url or "").strip()
        if remote_library_enabled(cfg) and remote_url:
            # Show the configured library server instead of the local loopback one.
            from autofa.ui.banner import REMOTE_LIBRARY_PREFIX

            # Replace the local server line (or its startup failure) in place; add the
            # link when neither is present. Full local details remain in Settings.
            local = ("AutoFA Local Library Server: ", "Local dashboard unavailable:")
            position = next((index for index, line in enumerate(current_status)
                             if line.startswith(local)), len(current_status))
            current_status = [line for line in current_status if not line.startswith(local)]
            current_status.insert(position, REMOTE_LIBRARY_PREFIX + remote_url)
        status = [model_summary(cfg), *current_status]
        header = home_menu_header(status)
        server_url = getattr(args, "_local_library_url", None)
        # Put the outcome first so it remains visible even on a small terminal.
        if notice_fragments():
            from autofa.ui.banner import home_status_fragments

            header = [
                ("class:title", "  AutoFA\n\n"), *notice_fragments(),
                *home_status_fragments(status), ("", "\n"),
            ]
        choice = select_option(
            "AutoFA",
            [
                MenuItem("New analysis", "1"),
                MenuItem("Continue from library", "2"),
                MenuItem("Classify failures by taxonomy", "classify"),
                *([MenuItem("Explore failure library", "local-server")] if server_url else []),
                MenuItem("Settings", "settings"),
                MenuItem("Quit", "q"),
            ],
            allow_cancel=True,
            clear_screen=True,
            header=header,
        )
        if choice is None or choice.value == "q":
            console.print("[dim]bye[/dim]")
            return

        actions = {
            "1": lambda: _new(args), "2": lambda: _continue(args),
            "classify": _classify,
            "settings": lambda: run_settings(runtime_info=status_lines),
            "local-server": lambda url=server_url: _open_library_server(url),
        }
        _execute_action(choice.label, actions[choice.value])


def _classify() -> None:
    """Launch the interactive fixed classification flow."""
    from autofa.taxonomy.cli import classify_main

    status = classify_main([])
    if status:
        raise SystemExit(status)


def _open_library_server(url: str) -> None:
    import webbrowser

    from autofa.local_service import validate_service_url

    url = validate_service_url(url)
    if webbrowser.open(url):
        show_action_notice("Opened the Local Library in your browser.", console=console,
                           warning=False)
    else:
        show_action_notice(f"Open {url} in your browser.", console=console, warning=False)


def _discovery_config(args) -> Config | None:
    from autofa.objectives import apply_preset, config_for_new_discovery
    from autofa.session.guidance import snapshot_guidance

    cfg = config_for_new_discovery(load_config())

    if getattr(args, "optimization", None):
        cfg = apply_preset(cfg, args.optimization)
    if getattr(args, "min_failure_modes", None) is not None:
        cfg.min_failure_modes = validate_min_failure_modes(args.min_failure_modes)
    if getattr(args, "top_x", None) is not None:
        cfg.top_x = args.top_x
    for key in ("analysis_outputs", "analysis_focus"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    if sys.stdin.isatty() and sys.stdout.isatty() and not any(
        getattr(args, key, None) is not None
        for key in ("analysis_outputs", "analysis_focus", "optimization")
    ):
        from autofa.ui.settings import configure_analysis

        cfg = configure_analysis(cfg)
        if cfg is None:
            return None
    try:
        cfg.analysis_outputs = validate_analysis_outputs(
            cfg.analysis_outputs, cfg.preflight_detectors,
        )
        snapshot_guidance(cfg)
    except ValueError as exc:
        console.print(f"Invalid analysis guidance: {exc}", style="red", markup=False)
        raise SystemExit(2) from exc
    return config_for_new_discovery(cfg)


def _show_analysis_guidance(run: RunConfig) -> None:
    from autofa.behavior_policy import STRICTNESS_LABELS
    from autofa.session.guidance import ANALYSIS_FOCUS_LABELS

    focus = run.analysis_focus
    console.print(f"Analysis target: {ANALYSIS_FOCUS_LABELS.get(focus, focus)}")
    console.print(f"Judge policy: {STRICTNESS_LABELS[run.behavior_strictness]}")
    minimum = run.min_failure_modes
    console.print(
        f"Minimum Good detectors: {minimum} (extra rounds if needed)"
        if minimum else "Minimum Good detectors: disabled"
    )
    if run.objective_settings:
        from autofa.objectives import config_for_run

        _show_optimization(config_for_run(Config(), run))

    if run.skills_root:
        console.print(f"Skills: {len(run.skill_snapshots)} active from {run.skills_root}",
                      markup=False)
    for skill in run.skill_snapshots:
        console.print(
            f"Skill: {skill['name']} ({len(skill['text'])} characters, "
            f"sha256={skill['sha256'][:12]})", markup=False,
        )
    for path in run.skill_exclusions:
        console.print(f"Skill excluded by user: {path}", style="yellow", markup=False)
    for path in run.skill_ignored_paths:
        console.print(f"Skill file not loaded (not Markdown): {path}",
                      style="yellow", markup=False)
    if packs := run.skill_unmatched_packs:
        console.print(f"Harness reference packs not loaded (harness does not match): "
                      f"{', '.join(packs)}. Set analysis_harness to select one.",
                      style="dim", markup=False)


def _show_optimization(cfg: Config, *, resumed: bool = False) -> None:
    from autofa.objectives import (
        OBJECTIVE_DISPLAY_LABELS,
        discovery_strategy,
        preset_name,
        weights_by_axis,
    )

    label = "Saved run optimization" if resumed else "Optimization"
    console.print(
        f"{label}: {preset_name(cfg)} · Discovery strategy: {discovery_strategy(cfg)}",
        style="bold cyan", markup=False,
    )
    console.print(" · ".join(
        f"{OBJECTIVE_DISPLAY_LABELS.get(axis, axis)}: {weight:.1%}"
        for axis, weight in weights_by_axis(cfg).items()
    ), markup=False)


def _remember_analyses(run: RunConfig, analyses: set[str]) -> None:
    run.analysis_outputs = sorted(analyses)
    run.save()


def _efficiency_gate(analyses: set[str], *, interactive: bool) -> EfficiencyGate | None:
    """Stop efficiency-only runs on unsupported data; ask before dropping it from others."""
    if "efficiency" not in analyses:
        return None
    combined = "agent" in analyses
    return EfficiencyGate(
        combined=combined,
        continue_without=_continue_without_efficiency if combined and interactive else None,
        choose=_choose_efficiency_continuation if interactive else None,
        notify=_show_efficiency_note,
    )


def _continue_without_efficiency(reason: str) -> bool:
    console.print(f"Efficiency metrics cannot run: {reason}", style="yellow", markup=False)
    return select_yes_no("Continue with failure discovery only?", default_yes=True) is True


_EFFICIENCY_CHOICES = {
    "both": "Continue with both analyses",
    "without": "Continue with failure discovery only",
    "continue": "Continue anyway",
    "stop": "Stop",
}


def _choose_efficiency_continuation(choices: tuple[str, ...]) -> str:
    choice = select_option(
        "Efficiency report",
        [MenuItem(_EFFICIENCY_CHOICES[value], value) for value in choices],
        allow_cancel=True,
        clear_screen=False,
    )
    return choice.value if choice is not None else "stop"


def _show_efficiency_note(message: str, warning: bool) -> None:
    console.print(message, style="yellow" if warning else "dim", markup=False)


def _without_efficiency(analyses: set[str], gate: EfficiencyGate | None) -> set[str]:
    return analyses - {"efficiency"} if gate is not None and not gate.enabled else analyses


def _run_without_agent(run: RunConfig, cfg: Config) -> None:
    if cfg.preflight_detectors:
        from autofa.preflight.runner import run_preflight
        from autofa.session.recovery import run_lock

        with run_lock(run.run_dir()):
            run_preflight(run, cfg)
    if "efficiency" in cfg.analysis_outputs:
        _generate_efficiency_metrics(run, cfg)


def _generate_efficiency_metrics(run: RunConfig, cfg: Config) -> bool:
    """Export the standalone efficiency report without starting an agent."""
    from autofa.reports.efficiency_card import resolve_pricing, write_efficiency_card

    pricing = resolve_pricing(cfg, run.model_id)
    with console.status("[green]Computing overall efficiency metrics…[/green]"):
        html_path, data_path, payload = write_efficiency_card(
            run,
            run.run_dir(),
            source_harness=getattr(run, "source_harness", "unknown"),
            reference_data=getattr(run, "reference_data", []),
            pricing=pricing,
        )
    url = html_path.resolve().as_uri()
    console.print(
        f"[green]Efficiency report saved:[/] [link={url}]{url}[/link]\n"
        f"[dim]  data: {data_path}\n"
        f"  rows: {int(payload.get('n_items') or 0):,} · "
        f"metrics: {len(payload.get('metrics') or [])}[/dim]"
    )
    for warning in payload.get("warnings") or []:
        console.print(f"[yellow]{warning}[/yellow]")
    if not webbrowser.open(url):
        console.print("[yellow]Could not open the efficiency report automatically.[/yellow]")
    return True


def _offer_efficiency_publication(run: RunConfig, cfg: Config) -> None:
    """Let an interactive efficiency-only run publish before returning home."""
    if not remote_library_enabled(cfg) or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    service = library_service_name(
        cfg.remote_library_base_url,
        mode=getattr(cfg, "remote_library_mode", "cloud"),
    )
    from autofa.data.remote_library_publish import maybe_publish_run

    while True:
        choice = select_option(
            "Efficiency report",
            [
                MenuItem(f"Publish to {service}", "publish"),
                MenuItem("Return to home", "home"),
            ],
            allow_cancel=True,
            clear_screen=False,
        )
        if choice is None or choice.value != "publish":
            return
        if maybe_publish_run(run, cfg, trigger="manual_cli", console=console) is not None:
            return


def _create_analysis_run(
    cfg: Config,
    selection,
    *,
    show_guidance: bool = True,
) -> RunConfig:
    from autofa.util.memory import apply_memory_tuning

    paths = selection.parquet_paths
    apply_memory_tuning(cfg, parquet_paths=paths)
    set_action_stage("create analysis run")
    run = RunConfig.create(
        paths,
        selection.model_id,
        selection.benchmarks,
        cfg,
        max_rounds=cfg.max_rounds,
        min_failure_modes=cfg.min_failure_modes,
        top_x=cfg.top_x,
        source_paths=selection.source_paths,
        source_kind=selection.source_kind,
        source_harness=selection.source_harness,
        reference_data=selection.reference_data,
        selection_id=getattr(selection, "selection_id", None),
    )
    console.print(f"[dim]Run dir: {run.run_dir()}[/dim]")
    set_action_stage("start analysis", analysis_run=run.run_id, run_dir=str(run.run_dir()))
    if show_guidance:
        _show_analysis_guidance(run)
    if selection.source_paths:
        console.print(
            f"[dim]source={selection.source_kind} · "
            f"{selection.source_paths[0]}"
            + (f" (+{len(selection.source_paths) - 1})" if len(selection.source_paths) > 1 else "")
            + "[/dim]"
        )
    console.print(
        f"[dim]eval_limit={run.eval_limit} rows (capped probe) · "
        f"eval_chunk_size={cfg.eval_chunk_size} "
        f"(full corpus streams in chunks)[/dim]"
    )
    return run


def _run_cli_dataset_wizard(args, *, api_base: str, api_name: str, efficiency_only: bool):
    return run_dataset_wizard(
        inputs=list(getattr(args, "input", None) or []),
        api_run_ids=list(getattr(args, "api_run", None) or []),
        models=list(getattr(args, "model", None) or []),
        benchmarks=list(getattr(args, "benchmark", None) or []),
        tasks=list(getattr(args, "task", None) or []),
        rewards=list(getattr(args, "reward", None) or []),
        runs_where=getattr(args, "runs_where", ""),
        where=getattr(args, "where", ""),
        latest=getattr(args, "latest", 0),
        per_run=getattr(args, "per_run", 0),
        limit=getattr(args, "limit", 0),
        sample=getattr(args, "sample", "first"),
        seed=getattr(args, "seed", 0),
        judge_view=getattr(args, "judge_view", None),
        group=getattr(args, "group", "all"),
        labels=list(getattr(args, "label", None) or []),
        evidence=list(getattr(args, "evidence", None) or []),
        list_taxonomy=getattr(args, "list_taxonomy", False),
        api_base_url=api_base,
        api_name=api_name,
        refresh_api=getattr(args, "refresh_api", False),
        efficiency_only=efficiency_only,
    )


def _prepare_new_config(args) -> AutoFAConfig | None:
    from autofa.estimates import prepare_estimator
    from autofa.wizard.depth import select_discovery_depth

    cfg = _discovery_config(args)
    if cfg is None:
        cancel_action("Discovery cancelled in analysis settings.")
        return None
    analyses = set(cfg.analysis_outputs)
    set_action_stage("select dataset")
    api_base = getattr(args, "api_base_url", None) or cfg.data_api_base_url
    gate = _efficiency_gate(analyses, interactive=_is_interactive_terminal())
    with efficiency_gate(gate):
        sel = _run_cli_dataset_wizard(
            args,
            api_base=api_base,
            api_name=cfg.data_api_name,
            efficiency_only=analyses == {"efficiency"} and not cfg.preflight_detectors,
        )
        if sel is not None:
            check_prepared(sel)
    analyses = _without_efficiency(analyses, gate)
    cfg.analysis_outputs = [name for name in cfg.analysis_outputs if name in analyses]
    if sel is None:
        if getattr(args, "list_taxonomy", False):
            return None
        cancel_action("Discovery cancelled in dataset selection.")
        return None
    run_agent = "agent" in analyses
    paths, model_id, benchmarks = sel.parquet_paths, sel.model_id, sel.benchmarks
    min_failure_modes = cfg.min_failure_modes
    if not run_agent:
        max_rounds = int(cfg.max_rounds or 3)
        top_x = int(cfg.top_x or 5)
    elif getattr(args, "rounds", None) is not None:
        max_rounds = args.rounds
        top_x = cfg.top_x
    elif getattr(args, "once", False) or not _is_interactive_terminal():
        max_rounds = int(cfg.max_rounds or 3)
        top_x = int(cfg.top_x or 5)
    else:
        depth = select_discovery_depth(
            default_rounds=cfg.max_rounds,
            default_top_x=cfg.top_x,
            default_min_failure_modes=cfg.min_failure_modes,
            min_failure_modes_override=getattr(args, "min_failure_modes", None),
            estimator=prepare_estimator(cfg, paths, model_id, benchmarks),
            auto_start_seconds=None if getattr(args, "no_auto", False) else 60,
        )
        if depth is None:
            cancel_action("Discovery cancelled in depth selection.")
            return None
        max_rounds = depth.max_rounds
        top_x = depth.top_x
        min_failure_modes = depth.min_failure_modes
    cfg.max_rounds = max_rounds
    cfg.top_x = top_x
    cfg.min_failure_modes = min_failure_modes
    if getattr(args, "eval_limit", None) is not None:
        cfg.default_eval_limit = args.eval_limit
    return AutoFAConfig(
        settings=cfg,
        source=sel,
        execution=AutoFAExecution(
            mode="session",
            auto_discover=not getattr(args, "no_auto", False),
            exit_after_discovery=(
                getattr(args, "once", False) or not _is_interactive_terminal()
            ),
            offer_efficiency_publication=True,
        ),
    )


def run_autofa(config: AutoFAConfig) -> RunConfig:
    from autofa.harness.base import BatchSessionCancelled
    from autofa.objectives import config_for_new_discovery
    from autofa.run_spec import materialize_autofa_config, preflight_autofa_config
    from autofa.session.runner import require_batch_target, run_session, write_run_report

    cfg = config_for_new_discovery(config.settings)
    config = replace(config, settings=cfg)
    from autofa.setup_contract import validate_launch_contract

    validate_launch_contract(config)
    set_action_stage("check AutoFA run readiness")
    preflight_autofa_config(config)
    set_action_stage("resolve AutoFA source")
    analyses = set(cfg.analysis_outputs)
    gate = _efficiency_gate(
        analyses,
        interactive=_is_interactive_terminal() and config.execution.mode != "batch",
    )
    with efficiency_gate(gate):
        selection = materialize_autofa_config(config)
        if selection is not config.source:
            check_prepared(selection)
    analyses = _without_efficiency(analyses, gate)
    cfg.analysis_outputs = [name for name in cfg.analysis_outputs if name in analyses]
    run_agent = "agent" in analyses
    run = _create_analysis_run(
        cfg,
        selection,
        show_guidance=run_agent,
    )
    _remember_analyses(run, analyses)
    if os.environ.get("AUTOFA_DRIVER_LAUNCH"):
        from autofa.session.managed_launch import bind_run

        bind_run(run.run_dir(), run.run_id)
    validate_launch_contract(config, run=run)
    if not run_agent:
        _run_without_agent(run, cfg)
        if "efficiency" in analyses and config.execution.offer_efficiency_publication:
            _offer_efficiency_publication(run, cfg)
        return run
    if config.execution.mode == "batch":
        try:
            run_session(
                run,
                cfg=cfg,
                auto_discover=config.execution.auto_discover,
                continue_mode=False,
                exit_after_discovery=True,
                publish_on_exit=False,
            )
        except BatchSessionCancelled as exc:
            raise KeyboardInterrupt from exc
        require_batch_target(run, cfg=cfg)
        report = write_run_report(run, cfg=cfg)
        console.print(f"[green]Report:[/green] {report}", markup=False)
        if config.execution.publish:
            from autofa.data.remote_library_publish import maybe_publish_run

            managed_discovery = (os.environ.get("AUTOFA_SETUP_CONTRACT")
                                 and set(run.analysis_outputs) == {"agent"})
            if managed_discovery:
                from autofa.data.discovery_delivery import saved_only_delivery
                from autofa.session.recovery import run_lock

                with run_lock(run.run_dir()):
                    delivered = saved_only_delivery(run, cfg)
                delivery_status = delivered.get("status") or "unknown"
                if delivery_status != "verified":
                    raise RuntimeError(f"Required cloud delivery remains {delivery_status}")
                console.print(f"[green]Verified report:[/green] {delivered['url']}")
            elif maybe_publish_run(run, cfg, trigger="session_end", console=console) is None:
                raise RuntimeError("Required batch publication failed")
        return run
    run_session(
        run,
        cfg=cfg,
        auto_discover=config.execution.auto_discover,
        continue_mode=False,
        exit_after_discovery=config.execution.exit_after_discovery,
    )
    return run


def _new(args) -> None:
    config = _prepare_new_config(args)
    if config is None:
        return
    run_autofa(config)


def _taxonomy_handoff_inputs(path: str | Path) -> list[str]:
    """Return configured classification outputs for a later New/Continue bucket."""
    from autofa.run_spec import load_run_spec
    from autofa.taxonomy.workflow_config import TaxonomyWorkflowSpec

    workflow = load_run_spec(path)
    if not isinstance(workflow, TaxonomyWorkflowSpec):
        raise ValueError(
            "New/Continue --config requires workflow: classification"
        )
    outputs = [
        str(judge.output)
        for configured in workflow.runs
        for judge in configured.judges
    ]
    if not outputs:
        raise ValueError("classification config contains no taxonomy result outputs")
    return outputs


def _run_config(args) -> None:
    from autofa.run_spec import load_autofa_config
    from autofa.taxonomy.workflow_config import TaxonomyWorkflowSpec

    config = load_autofa_config(args.run_config_path, load_config())
    if isinstance(config, TaxonomyWorkflowSpec):
        from autofa.taxonomy.cli import run_taxonomy_workflow

        status = run_taxonomy_workflow(config)
        if status:
            raise RuntimeError(f"taxonomy classification failed with exit status {status}")
        return
    run_autofa(config)


def _reuse_saved_selection(
    selection_id: str,
    references,
    *,
    api_base_url: str,
    api_name: str | None = None,
    efficiency_only: bool = False,
    refresh_api: bool = False,
):
    """Reuse an exact cohort, or explicitly offer a wider source-data fallback."""
    try:
        return run_dataset_wizard(
            selection_id=selection_id,
            api_base_url=api_base_url,
            **({"api_name": api_name} if api_name else {}),
            efficiency_only=efficiency_only,
            refresh_api=refresh_api,
        )
    except FileNotFoundError as exc:
        if not references:
            raise
        console.print(
            "[yellow]The exact saved rollout selection is unavailable. "
            "The original source references remain available, but reusing them may include "
            "additional rollouts.[/yellow]"
        )
        if not _is_interactive_terminal():
            raise FileNotFoundError(
                f"Saved selection {selection_id!r} is unavailable; "
                "source-data fallback requires interactive confirmation"
            ) from exc
        fallback = select_yes_no(
            "Reprepare the original source data instead?",
            default_yes=False,
            allow_back=True,
        )
        if is_menu_back(fallback) or not fallback:
            return None
        return reuse_reference_data(
            list(references),
            allowed_base_url=api_base_url,
            refresh_api=refresh_api,
            efficiency_only=efficiency_only,
        )


def _continue(args) -> None:
    from autofa.estimates import prepare_estimator
    from autofa.session.runner import run_session
    from autofa.util.memory import apply_memory_tuning
    from autofa.wizard.depth import select_discovery_depth

    cfg = _discovery_config(args)
    if cfg is None:
        return cancel_action("Continuation cancelled in analysis settings.")
    analyses = set(cfg.analysis_outputs)
    api_base = getattr(args, "api_base_url", None) or cfg.data_api_base_url
    explicit_data = _has_cli_data_selection(args)
    efficiency_only = analyses == {"efficiency"} and not cfg.preflight_detectors
    explicit_seed = bool(
        getattr(args, "library_path", None)
        or getattr(args, "remote_library_id", None)
    )
    gate = _efficiency_gate(analyses, interactive=_is_interactive_terminal())
    with efficiency_gate(gate):
        sel = None
        while sel is None:
            seed = _select_continue_seed(args, cfg)
            if seed is None:
                return cancel_action("Continuation cancelled in library selection.")

            if not explicit_data and (seed.selection_id or seed.reference_data):
                from autofa.data.reference_data import inspect_url

                links = [
                    link
                    for reference in seed.reference_data
                    if (link := inspect_url(reference))
                ]
                if links:
                    console.print("[dim]Reference data:[/dim]")
                    for link in links:
                        console.print(f"[dim]  · {link}[/dim]")
                while sel is None:
                    reuse = select_yes_no(
                        (
                            "Reuse the exact same prepared data for this continuation?"
                            if seed.selection_id
                            else "Reuse the same source data for this continuation?"
                        ),
                        default_yes=True,
                        allow_back=True,
                    )
                    if is_menu_back(reuse) or reuse is None:
                        break
                    sel = (
                        _reuse_saved_selection(
                            seed.selection_id,
                            seed.reference_data,
                            api_base_url=api_base,
                            api_name=cfg.data_api_name,
                            efficiency_only=efficiency_only,
                            refresh_api=getattr(args, "refresh_api", False),
                        )
                        if reuse and seed.selection_id
                        else reuse_reference_data(
                            list(seed.reference_data),
                            allowed_base_url=api_base,
                            refresh_api=getattr(args, "refresh_api", False),
                            efficiency_only=efficiency_only,
                        )
                        if reuse
                        else run_dataset_wizard(
                            api_base_url=api_base,
                            api_name=cfg.data_api_name,
                            refresh_api=getattr(args, "refresh_api", False),
                            efficiency_only=efficiency_only,
                        )
                    )
            else:
                sel = _run_cli_dataset_wizard(
                    args,
                    api_base=api_base,
                    api_name=cfg.data_api_name,
                    efficiency_only=efficiency_only,
                )
                if sel is None and getattr(args, "list_taxonomy", False):
                    return
            if sel is None and explicit_seed:
                return cancel_action("Continuation cancelled in dataset selection.")
        check_prepared(sel)
    analyses = _without_efficiency(analyses, gate)
    cfg.analysis_outputs = [name for name in cfg.analysis_outputs if name in analyses]
    run_agent = "agent" in analyses
    paths, model_id, benchmarks = sel.parquet_paths, sel.model_id, sel.benchmarks
    apply_memory_tuning(cfg, parquet_paths=paths)
    min_failure_modes = cfg.min_failure_modes
    if not run_agent:
        max_rounds = int(cfg.max_rounds or 3)
        top_x = int(cfg.top_x or 5)
    elif getattr(args, "rounds", None) is not None:
        max_rounds = args.rounds
        top_x = cfg.top_x
    elif getattr(args, "once", False) or not _is_interactive_terminal():
        max_rounds = int(cfg.max_rounds or 3)
        top_x = int(cfg.top_x or 5)
    else:
        depth = select_discovery_depth(
            default_rounds=cfg.max_rounds,
            default_top_x=cfg.top_x,
            default_min_failure_modes=cfg.min_failure_modes,
            min_failure_modes_override=getattr(args, "min_failure_modes", None),
            estimator=prepare_estimator(cfg, paths, model_id, benchmarks),
            auto_start_seconds=None if getattr(args, "no_auto", False) else 60,
        )
        if depth is None:
            return cancel_action("Continuation cancelled in depth selection.")
        max_rounds = depth.max_rounds
        top_x = depth.top_x
        min_failure_modes = depth.min_failure_modes
    run = RunConfig.create(
        paths,
        model_id,
        benchmarks,
        cfg,
        seed_library_path=seed.path,
        seed_source_run_id=seed.source_run_id,
        remote_library_id=seed.remote_library_id,
        remote_branch_id=seed.remote_branch_id,
        remote_branch_key=seed.remote_branch_key,
        remote_version_id=seed.remote_version_id,
        remote_base_version_id=seed.remote_base_version_id,
        max_rounds=max_rounds,
        min_failure_modes=min_failure_modes,
        top_x=top_x,
        source_paths=sel.source_paths,
        source_kind=sel.source_kind,
        source_harness=getattr(sel, "source_harness", "unknown"),
        reference_data=getattr(sel, "reference_data", []),
        selection_id=getattr(sel, "selection_id", None),
    )
    if args.eval_limit:
        run.eval_limit = args.eval_limit
    _remember_analyses(run, analyses)
    if run_agent:
        load_runtime_from_run(run, cfg).persist_library()
        _show_analysis_guidance(run)
    console.print(
        f"[dim]eval_limit={run.eval_limit} rows (capped probe) · "
        f"eval_chunk_size={cfg.eval_chunk_size} "
        f"(full corpus streams in chunks)[/dim]"
    )
    if not run_agent:
        _run_without_agent(run, cfg)
        return
    run_session(
        run,
        cfg=cfg,
        auto_discover=True,
        continue_mode=True,
        exit_after_discovery=getattr(args, "once", False) or not _is_interactive_terminal(),
    )


def _select_continue_seed(args, cfg: Config) -> ContinueSeed | None:
    """Resolve local or remote Continue seed at a testable UI/network seam."""
    local_path = getattr(args, "library_path", None)
    remote_id = getattr(args, "remote_library_id", None)
    remote_branch_id = getattr(args, "remote_branch_id", None)
    remote_version_id = getattr(args, "remote_version_id", None)
    if local_path and remote_id:
        raise ValueError("Use only one of --library and --remote-library")
    if not remote_id and (remote_branch_id or remote_version_id):
        raise ValueError("--remote-branch/--remote-version require --remote-library")
    if local_path:
        return _local_continue_seed(Path(local_path).expanduser().resolve())
    if remote_id:
        client = client_from_config(cfg)
        selected = client.resolve_library_ref(
            str(remote_id),
            branch_id=str(remote_branch_id) if remote_branch_id else None,
            version_id=str(remote_version_id) if remote_version_id else None,
        )
        path = materialize_remote_library(selected, client=client)
        console.print(f"[dim]Cached remote seed: {path}[/dim]")
        return _continue_seed_from_remote(path, selected)

    while True:
        mode = select_option(
            "Seed library",
            [
                MenuItem("Browse catalog (~/.autofa/library)", "catalog"),
                *([MenuItem(library_service_name(cfg.remote_library_base_url,
                                                mode=cfg.remote_library_mode), "remote")]
                  if remote_library_enabled(cfg) else []),
            ],
            allow_cancel=True,
        )
        if mode is None:
            return None
        if mode.value == "remote":
            client = client_from_config(cfg)
            selected = select_remote_library(client, title="Select remote seed library")
            if selected is None:
                continue
            path = materialize_remote_library(selected, client=client)
            console.print(f"[dim]Cached remote seed: {path}[/dim]")
            return _continue_seed_from_remote(path, selected)
        picked = browse_filesystem(
            cfg.resolved_library_root(),
            title="Select library.json",
            select_files=True,
            select_dirs=False,
            file_glob="library.json",
            allow_select_dir_with_matches=False,
        )
        if picked is not None:
            return _local_continue_seed(picked.resolve())


def _continue_seed_from_remote(path: Path, selected: RemoteLibraryRef) -> ContinueSeed:
    version_id = selected.version_id
    try:
        source_run = RunConfig.load(selected.source_run_id)
    except (OSError, ValueError, TypeError):
        source_run = None
    return ContinueSeed(
        path=str(path.resolve()),
        remote_library_id=selected.library_id,
        remote_branch_id=selected.branch_id,
        remote_branch_key=selected.branch_key,
        remote_version_id=version_id,
        remote_base_version_id=version_id,
        source_run_id=selected.source_run_id or None,
        source_harness=selected.version.source_harness,
        reference_data=tuple(
            reference.to_dict()
            for reference in selected.version.reference_data
        ),
        selection_id=source_run.selection_id if source_run is not None else None,
    )


def _working_library_run(path: Path) -> RunConfig | None:
    """Return the run that owns ``path`` when it is that run's working library.

    ``runs/<id>`` may be a symlink to another disk, so compare resolved run directories.
    """
    path = path.resolve()
    run_dir = path.parent
    if path.name != "library.json" or not (run_dir / "config.json").is_file():
        return None
    root = runs_root()
    run = _run_stored_at(root / run_dir.name, run_dir)
    if run is not None:
        return run
    # Scan only for a link whose target folder has another name; never resolve real runs.
    try:
        links = sorted(entry for entry in root.iterdir()
                       if entry.name != run_dir.name and entry.is_symlink())
    except OSError:
        return None
    return next((run for entry in links if (run := _run_stored_at(entry, run_dir))), None)


def _run_stored_at(entry: Path, run_dir: Path) -> RunConfig | None:
    """Load the run registered as ``entry`` when it resolves to ``run_dir``."""
    try:
        if entry.resolve() != run_dir:
            return None
        run = RunConfig.load(entry.name)
    except (OSError, RuntimeError, ValueError, TypeError):
        return None
    return run if run.run_id == entry.name else None


def _local_continue_seed(path: Path) -> ContinueSeed:
    """Read sidecar provenance, or the owning run's lineage for its working library.

    Any other catalog without a sidecar starts a fresh lineage.
    """
    provenance = read_catalog_provenance(path)
    source_run: RunConfig | None = None
    if provenance is not None:
        try:
            source_run = RunConfig.load(provenance.source_run_id)
        except (OSError, ValueError, TypeError):
            source_run = None
        return ContinueSeed(
            path=str(path.resolve()),
            remote_library_id=provenance.remote_library_id,
            remote_branch_id=provenance.remote_branch_id,
            remote_branch_key=provenance.remote_branch_key,
            remote_version_id=provenance.remote_version_id,
            remote_base_version_id=provenance.remote_base_version_id,
            source_run_id=provenance.source_run_id,
            source_harness=(
                source_run.source_harness
                if source_run is not None
                else provenance.source_harness
            ),
            reference_data=tuple(
                source_run.reference_data
                if source_run is not None and source_run.reference_data
                else provenance.reference_data
            ),
            selection_id=(
                source_run.selection_id
                if source_run is not None
                else provenance.selection_id
            ),
        )
    # A run's working library has no sidecar; its config.json holds the lineage.
    source_run = _working_library_run(path)
    if source_run is None:
        return ContinueSeed(path=str(path.resolve()))
    return ContinueSeed(
        path=str(path.resolve()),
        remote_library_id=source_run.remote_library_id,
        remote_branch_id=source_run.remote_branch_id,
        remote_branch_key=source_run.remote_branch_key or source_run.run_id,
        remote_version_id=source_run.remote_version_id,
        remote_base_version_id=source_run.remote_base_version_id,
        source_run_id=source_run.run_id,
        source_harness=source_run.source_harness,
        reference_data=tuple(source_run.reference_data),
        selection_id=source_run.selection_id,
    )


def _publish(args) -> RemotePublication | None:
    """Regenerate and publish a run's library and self-contained report."""
    from autofa.data.remote_library_publish import publish_run
    from autofa.mcp.runtime import load_runtime_from_run
    from autofa.session.report_highlights import prepare_report_highlights

    cfg = load_config()
    run_id = getattr(args, "run_id", None)
    if not run_id:
        runs = list_runs()
        if not runs:
            show_action_notice(
                "No runs found to publish. Start a discovery first.", console=console,
            )
            return None
        choice = select_option(
            "Publish run",
            [
                MenuItem(
                    run_name,
                    run_name,
                    detail=str(runs_root() / run_name),
                )
                for run_name in runs[:30]
            ],
            allow_cancel=True,
        )
        if choice is None:
            return None
        run_id = str(choice.value)

    run = RunConfig.load(str(run_id))
    runtime = load_runtime_from_run(run, cfg)
    prepare_report_highlights(runtime, console=console)
    remote = publish_run(run, cfg, trigger="manual_cli", runtime=runtime, console=console)
    url = (
        remote.dashboard_url
        or remote.failure_card_url
        or f"{cfg.remote_library_base_url.rstrip('/')}/libraries/{remote.library_id}"
    )
    console.print(f"[green]Published remote library:[/green] {url}")
    return remote


def _resume(args) -> None:
    from autofa.session.runner import run_session
    from autofa.util.memory import apply_memory_tuning

    cfg = load_config()
    run_id = args.run_id
    if not run_id:
        runs = list_runs()
        if not runs:
            show_action_notice("No runs found. Start a new discovery first.", console=console)
            return
        choice = select_option(
            "Resume run",
            [MenuItem(r, r, detail=str(runs_root() / r)) for r in runs[:30]],
            allow_cancel=True,
        )
        if choice is None:
            return cancel_action("Resume cancelled in run selection.")
        run_id = str(choice.value)
    run = RunConfig.load(run_id)
    if not run.analysis_outputs and not run.preflight_detectors:
        raise ValueError(
            "Cannot resume: this run predates recorded analysis outputs. Start a new run."
        )
    from autofa.session.supervision import check_lifecycle

    check_lifecycle(run.run_dir())
    from autofa.config import restore_harness_settings

    cfg = restore_harness_settings(cfg, run)
    resume_options = {}
    if getattr(args, "rounds", None) is not None:
        resume_options["max_rounds"] = args.rounds
    if getattr(args, "min_failure_modes", None) is not None:
        # Apply persistent overrides only after the selected runner acquires its
        # exclusive run lock; a refused Resume must not edit an active run.
        resume_options["min_failure_modes"] = validate_min_failure_modes(
            args.min_failure_modes,
        )
    if getattr(args, "top_x", None) is not None:
        resume_options["top_x"] = args.top_x
    missing = [
        path
        for path in run.parquet_paths
        if not Path(path).expanduser().is_file()
    ]
    gate = _efficiency_gate(set(run.analysis_outputs), interactive=_is_interactive_terminal())
    with efficiency_gate(gate):
        if missing:
            if not run.reference_data:
                raise FileNotFoundError(
                    "Cannot resume: bound parquet data is missing and this run has no Data API "
                    "reference. Restore these files or start a new discovery: " + ", ".join(missing)
                )
            console.print(
                f"[yellow]{len(missing)} bound parquet file(s) are missing.[/yellow]"
            )
            if not run.selection_id:
                console.print(
                    "[yellow]This older run did not save its rollout selection; "
                    "recovery can only reuse its source data.[/yellow]"
                )
            redownload = select_yes_no(
                (
                    "Reprepare the exact saved data for this run?"
                    if run.selection_id
                    else "Reprepare source data for this run?"
                ),
                default_yes=True,
                allow_back=True,
            )
            if is_menu_back(redownload) or not redownload:
                return cancel_action("Resume cancelled; missing data was not redownloaded.")
            efficiency_only = (
                set(run.analysis_outputs) == {"efficiency"}
                and not run.preflight_detectors
            )
            selection = (
                _reuse_saved_selection(
                    run.selection_id,
                    run.reference_data,
                    api_base_url=cfg.data_api_base_url,
                    efficiency_only=efficiency_only,
                )
                if run.selection_id
                else reuse_reference_data(
                    run.reference_data,
                    allowed_base_url=cfg.data_api_base_url,
                    efficiency_only=efficiency_only,
                )
            )
            if selection is None:
                return cancel_action("Resume cancelled in dataset selection.")
            run.parquet_paths = [
                str(path.resolve()) for path in selection.parquet_paths
            ]
            run.source_paths = list(selection.source_paths)
            run.source_kind = selection.source_kind
            run.source_harness = selection.source_harness
            run.reference_data = list(selection.reference_data)
            run.selection_id = getattr(selection, "selection_id", None)
            check_prepared(run)
            run.save()
        else:
            check_prepared(run)
    if gate is not None and not gate.enabled:
        run.analysis_outputs = [name for name in run.analysis_outputs if name != "efficiency"]
        run.save()
    tune = apply_memory_tuning(cfg, parquet_paths=run.parquet_paths)
    console.print(f"[dim]Resuming {run.run_dir()}[/dim]")
    console.print(f"[dim]{tune.summary()}[/dim]")
    from autofa.objectives import config_for_run

    _show_optimization(config_for_run(cfg, run), resumed=True)
    if (run.analysis_outputs or run.preflight_detectors) and "agent" not in run.analysis_outputs:
        cfg.analysis_outputs = list(run.analysis_outputs)
        cfg.preflight_detectors = list(run.preflight_detectors)
        _run_without_agent(run, cfg)
        return
    run_session(
        run,
        cfg=cfg,
        auto_discover=False,
        continue_mode=True,
        exit_after_discovery=bool(getattr(args, "once", False)),
        **resume_options,
    )


def _dashboard(args) -> None:
    from autofa.ui.explorer import explore_library

    def _open(
        lib: FunctionLibrary,
        *,
        title: str,
        html_dir: Path | None = None,
        dataset=None,
    ) -> None:
        html_path = (html_dir / "failure_cards.html") if html_dir is not None else None
        explore_library(
            lib, top_x=None, title=title, html_path=html_path, dataset=dataset
        )

    cfg = load_config()
    if args.run_id:
        run = RunConfig.load(args.run_id)
        rt = load_runtime_from_run(run, cfg)
        _open(
            rt.library,
            title=run.explorer_title(),
            html_dir=run.run_dir(),
            dataset=rt.dataset,
        )
        return
    if args.library_path:
        lib_path = Path(args.library_path).expanduser()
        lib = FunctionLibrary.load(lib_path)
        _open(
            lib,
            title=f"{lib.model_id or '?'} / {lib.benchmark or 'all'}",
            html_dir=lib_path.parent,
        )
        return
    runs = list_runs()
    if not runs:
        browse_library()
        return
    run = RunConfig.load(runs[0])
    lib_path = run.run_dir() / "library.json"
    if not lib_path.exists():
        show_action_notice(f"No library in latest run {run.run_id}. Resume discovery first.",
                           console=console)
        return
    rt = load_runtime_from_run(run, cfg)
    _open(
        rt.library,
        title=run.explorer_title(),
        html_dir=run.run_dir(),
        dataset=rt.dataset,
    )


if __name__ == "__main__":
    main()
