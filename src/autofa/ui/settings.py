# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive analysis controls using the existing arrow-key menus."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from shutil import get_terminal_size
from textwrap import wrap

import tomlkit
from rich.console import Console

from autofa.behavior_policy import (
    BEHAVIOR_STRICTNESSES,
    STRICTNESS_LABELS,
    validate_behavior_strictness,
)
from autofa.config import (
    Config,
    config_path,
    load_config,
    validate_analysis_outputs,
    validate_discovery_stall_rounds,
    validate_discovery_terminate_if_stalled,
    validate_discovery_wait_for_plateau,
    validate_taxonomy_agentic_max_turns,
    validate_taxonomy_first_pass,
    validate_taxonomy_second_pass,
)
from autofa.objectives import (
    OBJECTIVE_DISPLAY_LABELS,
    OBJECTIVE_LABELS,
    PRESETS,
    apply_preset,
    config_for_new_discovery,
    objective_summary,
    preset_name,
    weights_by_axis,
)
from autofa.preflight.registry import detectors
from autofa.session.analysis_skills import skill_inventory
from autofa.session.guidance import (
    ANALYSIS_FOCI,
    ANALYSIS_FOCUS_LABELS,
    snapshot_guidance,
)
from autofa.ui.errors import show_action_error
from autofa.ui.menus import (
    MenuItem,
    browse_filesystem,
    is_menu_back,
    select_option,
)

console = Console()
_SETTING_FIELDS = (
    "analysis_outputs", "preflight_detectors", "analysis_focus", "skill_paths",
    "skills_dir", "disabled_skill_paths", "objective_axes", "objective_rank", "objective_weights",
    "min_failure_modes", "discovery_wait_for_plateau", "discovery_terminate_if_stalled",
    "discovery_stall_rounds",
    "behavior_strictness",
)
_ENDPOINT_FIELDS = tuple(f"{role}_{field}" for role in ("feature", "judge")
                         for field in ("base_url", "api_key_env", "model")) + ("feature_api",)
_LIBRARY_FIELDS = ("remote_library_mode", "remote_library_base_url", "remote_library_token_env")
_EVAL_FIELDS = ("data_api_base_url", "data_api_name")
_TAXONOMY_FIELDS = (
    "taxonomy_first_pass",
    "taxonomy_second_pass",
    "taxonomy_agentic_max_turns",
)


def _skills_summary(cfg: Config, *, compact: bool = False) -> str:
    try:
        inventory = skill_inventory(cfg)
        if compact:
            return f"{len(inventory['active'])} active"
        return (f"{len(inventory['active'])} active, "
                f"{len(inventory['excluded'])} excluded, "
                f"{len(inventory['ignored'])} non-Markdown")
    except (OSError, ValueError):
        return "folder error (open Skills)"


def settings_summary(cfg: Config) -> str:
    return (
        f"Analysis target: {ANALYSIS_FOCUS_LABELS[cfg.analysis_focus]} "
        f"| Skills: {_skills_summary(cfg, compact=True)} "
        f"| Judge policy: {STRICTNESS_LABELS[cfg.behavior_strictness]}"
    )


def model_summary(cfg: Config) -> str:
    from autofa.harness.settings import HARNESS_NAMES, harness_model
    from autofa.ui.llm_settings import model_label

    model = model_label(harness_model(cfg.harness, cfg.model, cfg=cfg))
    return f"Model: {model} · {HARNESS_NAMES[cfg.harness]}"


def scoring_summary(cfg: Config) -> str:
    scope = (f"up to {cfg.llm_eval_sample:,} examples per detector"
             if cfg.llm_eval_sample > 0 else "full dataset")
    return f"Iterative LLM scoring: {scope}; final scoring: full dataset"


def system_info_lines(cfg: Config, runtime_info: list[str] | None = None) -> list[str]:
    """Use current settings plus nonsecret startup diagnostics, never environment values."""
    from autofa.config import site_config_path
    from autofa.harness.settings import behavior_judge_settings

    details = [
        f"Config: {config_path()}", model_summary(cfg),
        f"Judge model: {behavior_judge_settings(cfg)['model']}", settings_summary(cfg),
        f"Detector fields: {cfg.detector_fields}",
        scoring_summary(cfg),
        f"Optimization: {objective_summary(cfg)}",
        f"Objective axes: {', '.join(cfg.objective_axes)}",
        f"Minimum Good detectors: {cfg.min_failure_modes}",
        "Continue until coverage plateaus: " + (
            "On" if cfg.discovery_wait_for_plateau else "Off"
        ),
        "Terminate if stalled: " + (
            "On (extra rounds only)" if cfg.discovery_terminate_if_stalled else "Off"
        ),
        f"Stalled extra rounds: {cfg.discovery_stall_rounds}",
    ]
    if deployment := site_config_path():
        details.append(f"Deployment defaults: {deployment} (saved settings take priority)")
    details.extend(line for line in (runtime_info or []) if line.startswith(
        ("Loaded env:", "mem:", "AutoFA Local Library", "AutoFA Cloud Library:",
         "Local dashboard unavailable:")
    ))
    if not any(line.startswith("mem:") for line in details):
        from autofa.util.memory import apply_memory_tuning

        details.append(apply_memory_tuning(replace(cfg)).summary())
    return details


def show_system_information(cfg: Config, runtime_info: list[str] | None = None) -> None:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea

    from autofa.ui.pt_app import run_app
    from autofa.ui.theme import MENU_STYLE

    body = TextArea(text="\n\n".join(system_info_lines(cfg, runtime_info)),
                    read_only=True, scrollbar=True, wrap_lines=True)
    keys = KeyBindings()

    @keys.add("escape")
    @keys.add("q")
    @keys.add("c-c")
    @keys.add("enter")
    def back(event):
        event.app.exit()

    app = Application(layout=Layout(HSplit([
        Window(FormattedTextControl([
            ("class:title", "  System information\n"),
            ("class:hint", "  ↑/↓ or PgUp/PgDn scroll · Enter/Esc back\n\n"),
        ]), height=3), body,
    ]), focused_element=body), key_bindings=keys, style=MENU_STYLE, full_screen=True)
    run_app(app)


def save_analysis_settings(cfg: Config) -> None:
    """Update only the menu-owned fields; retain comments and unknown settings."""
    cfg = config_for_new_discovery(cfg)
    validate_behavior_strictness(cfg.behavior_strictness)
    validate_discovery_wait_for_plateau(cfg.discovery_wait_for_plateau)
    validate_discovery_terminate_if_stalled(cfg.discovery_terminate_if_stalled)
    validate_discovery_stall_rounds(cfg.discovery_stall_rounds)
    validate_analysis_outputs(cfg.analysis_outputs, cfg.preflight_detectors)
    snapshot_guidance(cfg)
    weights_by_axis(cfg)
    _save_fields(cfg, _SETTING_FIELDS)


def save_taxonomy_settings(cfg: Config) -> None:
    """Validate and save only the taxonomy menu fields."""
    validate_taxonomy_first_pass(cfg.taxonomy_first_pass)
    validate_taxonomy_second_pass(cfg.taxonomy_second_pass)
    validate_taxonomy_agentic_max_turns(cfg.taxonomy_agentic_max_turns)
    _save_fields(cfg, _TAXONOMY_FIELDS)


def _save_fields(cfg: Config, fields: tuple[str, ...]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = (
        tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    )
    for key in fields:
        document[key] = getattr(cfg, key)
    path.write_text(tomlkit.dumps(document), encoding="utf-8")


def configure_endpoints(cfg: Config) -> Config:
    from autofa.harness.settings import HARNESS_NAMES, behavior_judge_settings, harness_model
    from autofa.ui.llm_settings import edit_agent, model_label

    while True:
        harness = HARNESS_NAMES[cfg.harness]
        judge_config = behavior_judge_settings(cfg)
        judge_name = HARNESS_NAMES[judge_config["harness"]]
        selected = select_option("LLM endpoints", [
            MenuItem("AutoFA harness", "harness",
                     f"{harness} · {model_label(harness_model(cfg.harness, cfg.model, cfg=cfg))}"),
            MenuItem("Judge endpoint", "judge",
                     f"{judge_name} · {model_label(judge_config['model'])}"),
            MenuItem("Detector LLM endpoint", "detector", model_label(cfg.feature_model)),
            MenuItem("Back", "back"),
        ], allow_back=True)
        if not selected or is_menu_back(selected) or selected.value == "back":
            return cfg
        if selected.value == "detector":
            cfg = configure_detector_endpoint(cfg)
        else:
            judge = selected.value == "judge"
            fields = (("behavior_judge_harness", "behavior_judge_model", "behavior_judge_effort",
                       "behavior_judge_thinking", "behavior_judge_thinking_budget_tokens") if judge
                      else ("harness", "model", "thinking", "effort", "thinking_budget_tokens",
                            "behavior_judge_model", "behavior_judge_thinking",
                            "behavior_judge_effort"))
            cfg = edit_agent(cfg, judge=judge,
                             save=lambda candidate, fields=fields: _save_fields(candidate, fields))


def configure_detector_endpoint(cfg: Config) -> Config:
    from autofa.ui.llm_settings import edit_endpoints

    return edit_endpoints(cfg, save=lambda candidate: _save_fields(candidate, _ENDPOINT_FIELDS))


def configure_eval_server(cfg: Config, *, persistent: bool = True) -> Config:
    from autofa.ui.service_settings import edit_eval_server

    return edit_eval_server(
        cfg, save=(lambda candidate: _save_fields(candidate, _EVAL_FIELDS)) if persistent
        else lambda candidate: None,
    )


def configure_library_server(cfg: Config, *, persistent: bool = True) -> Config:
    from autofa.ui.library_settings import edit_library_server

    return edit_library_server(
        cfg,
        save=(lambda candidate: _save_fields(candidate, _LIBRARY_FIELDS)) if persistent
        else lambda candidate: None,
    )


def configure_objectives(cfg: Config, *, persistent: bool = False) -> Config:
    """One keyboard slider surface; persist each valid change before displaying it."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    from autofa.ui.menus import prepare_step
    from autofa.ui.pt_app import run_app
    from autofa.ui.theme import MENU_STYLE

    draft, index, error = replace(config_for_new_discovery(cfg)), 0, ""
    axes = list(OBJECTIVE_LABELS)

    def commit(candidate: Config) -> None:
        nonlocal draft, error
        try:
            candidate = config_for_new_discovery(candidate)
            if persistent:
                save_analysis_settings(candidate)
            draft, error = candidate, ""
        except (OSError, ValueError) as exc:
            error = str(exc)

    def render():
        weights = weights_by_axis(draft)
        raw = dict(zip(draft.objective_axes, draft.objective_weights or
                       [100.0 / len(draft.objective_axes)] * len(draft.objective_axes),
                       strict=True))
        labels = ["Back", f"Preset                 < {preset_name(draft)} >"]
        for axis in axes:
            value = max(0, min(100, raw.get(axis, 0)))
            filled = round(value / 100 * 12)
            bar = "#" * filled + "-" * (12 - filled)
            label = OBJECTIVE_DISPLAY_LABELS.get(axis, axis)
            labels.append(f"{label:24} [{bar}] {value:5.1f}  ({weights.get(axis, 0):.0%})")
        lines = [("class:title", "  Optimization weights" +
                  (" (auto-save)" if persistent else "") + "\n\n")]
        for row, label in enumerate(labels):
            style = "class:selected" if row == index else "class:item"
            lines.append((style, ("> " if row == index else "  ") + label + "\n"))
        if error:
            lines.append(("class:hint", "\n  " + error + "\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def up(event):
        nonlocal index
        index = (index - 1) % (len(axes) + 2)

    @kb.add("down")
    def down(event):
        nonlocal index
        index = (index + 1) % (len(axes) + 2)

    def adjust(delta: int) -> None:
        if index == 1:
            names = list(PRESETS)
            current = preset_name(draft).lower()
            pos = names.index(current) if current in names else -1
            commit(apply_preset(draft, names[(pos + delta) % len(names)]))
        elif index >= 2:
            raw = dict(zip(draft.objective_axes, draft.objective_weights or
                           [100.0 / len(draft.objective_axes)] * len(draft.objective_axes),
                           strict=True))
            axis = axes[index - 2]
            raw[axis] = max(0.0, min(100.0, raw.get(axis, 0.0) + delta * 5))
            commit(replace(draft, objective_axes=axes,
                           objective_weights=[raw.get(a, 0.0) for a in axes],
                           objective_rank="geometric_mean"))

    @kb.add("left")
    def left(event):
        adjust(-1)

    @kb.add("right")
    def right(event):
        adjust(1)

    @kb.add("enter")
    def enter(event):
        if index == 0:
            event.app.exit()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def back(event):
        event.app.exit()

    prepare_step()
    app = Application(layout=Layout(HSplit([Window(FormattedTextControl(render))])),
                      key_bindings=kb, style=MENU_STYLE, full_screen=False)
    run_app(app)
    return draft


def select_objective_preset(cfg: Config) -> Config:
    effective = config_for_new_discovery(cfg)
    names = list(PRESETS)
    current = preset_name(effective).lower()
    options = [MenuItem(name.capitalize(), name) for name in names]
    if current not in names:
        options.append(MenuItem("Custom (Settings)", "custom"))
    choice = select_option("Optimization preset", options,
                           default_index=names.index(current) if current in names else 3,
                           allow_back=True)
    if choice is None or is_menu_back(choice):
        return cfg
    if choice.value == "custom":
        return effective
    return apply_preset(effective, str(choice.value))


def select_analysis_focus(default: str) -> str | None:
    choices = [
        MenuItem("Model failures", "model", "reasoning, decisions, verification, tool use"),
        MenuItem("Harness failures", "harness", "tools, environment, budgets, feedback"),
        MenuItem(ANALYSIS_FOCUS_LABELS["both"], "both", "model, harness, and their interactions"),
    ]
    selected = select_option("What should AutoFA focus on?", choices,
                             default_index=ANALYSIS_FOCI.index(default), allow_back=True)
    return None if selected is None or is_menu_back(selected) else str(selected.value)


def _skills_settings(cfg: Config, *, persistent: bool = False) -> None:
    from autofa.session.analysis_skills import snapshot_skills

    while True:
        try:
            inventory = skill_inventory(cfg)
        except (OSError, ValueError) as exc:
            show_action_error(exc, console=console, context="Read skills folder")
            inventory = {"root": cfg.skills_dir or str(Path.cwd()), "active": [],
                         "automatic": [], "excluded": [], "ignored": []}
        root = Path(inventory["root"])
        enabled = set(inventory["active"])
        excluded = set(inventory["excluded"])
        additional = {str(Path(p).expanduser().resolve()) for p in cfg.skill_paths}
        available = sorted(set(inventory["automatic"]) | additional | excluded)
        choices = []
        for path in available:
            name = str(Path(path).relative_to(root)) if Path(path).is_relative_to(root) else path
            choices.append(MenuItem(f"[{'x' if path in enabled else ' '}] {name}", path,
                                    "missing" if not Path(path).is_file() else ""))
        choices.extend([MenuItem("Enable all Markdown", "all"),
                        MenuItem("Select skills folder", "folder"),
                        MenuItem("Use default skills folder", "default")])
        choices.extend([MenuItem("Add Markdown file", "file"),
                        MenuItem("View skill", "view"), MenuItem("Back", "back")])
        footer = [("", line + "\n") for line in wrap(
            f"Folder: {root}. All Markdown active by default; harness/<pack>/ references "
            "load only for a matching harness. "
            f"{len(enabled)} active, {len(excluded)} excluded, "
            f"{len(inventory['ignored'])} non-Markdown files not loaded.",
            width=max(20, get_terminal_size((80, 24)).columns - 4),
        )]
        selected = select_option("Skills", choices, allow_back=True, footer=footer)
        if selected is None or is_menu_back(selected) or selected.value == "back":
            return
        try:
            candidate = replace(cfg)
            if selected.value in {"folder", "default"}:
                if selected.value == "folder":
                    path = browse_filesystem(root if root.is_dir() else Path.cwd(),
                                             title="Skills folder", select_files=False,
                                             select_dirs=True)
                    if path is None:
                        continue
                    candidate.skills_dir = str(path.resolve())
                else:
                    candidate.skills_dir = ""
            elif selected.value == "all":
                excluded.clear()
            elif selected.value in {"file", "view"}:
                path = browse_filesystem(root, title="Select skill", select_files=True,
                                         select_dirs=False, file_glob="*.md",
                                         allow_select_dir_with_matches=False)
                if path is None:
                    continue
                path = str(path.resolve())
                if selected.value == "view":
                    with console.pager(styles=False):
                        console.print(snapshot_skills([path])[0]["text"], markup=False,
                                      highlight=False)
                    continue
                additional.add(path)
                excluded.discard(path)
            elif selected.value in enabled:
                excluded.add(selected.value)
            else:
                excluded.discard(selected.value)
            candidate.skill_paths = sorted(additional)
            candidate.disabled_skill_paths = sorted(
                str(Path(p).relative_to(root)) if Path(p).is_relative_to(root) else p
                for p in excluded
            )
            snapshot_guidance(candidate)
            if persistent:
                save_analysis_settings(candidate)
            cfg.skill_paths = candidate.skill_paths
            cfg.skills_dir = candidate.skills_dir
            cfg.disabled_skill_paths = candidate.disabled_skill_paths
        except (OSError, ValueError) as exc:
            show_action_error(exc, console=console, context="Save skills settings")


def configure_preflight(cfg: Config, *, persistent: bool = False) -> Config:
    """Build detector toggles from the same registry used by the CLI and runner."""
    draft = replace(config_for_new_discovery(cfg))
    while True:
        choice = select_option("Preflight", [
            MenuItem("Back", "back"),
            *(MenuItem(detector.label, detector.name,
                       "On" if detector.name in draft.preflight_detectors else "Off")
              for detector in detectors()),
        ], allow_back=True)
        if choice is None or is_menu_back(choice) or choice.value == "back":
            return draft
        enabled = set(draft.preflight_detectors)
        enabled.symmetric_difference_update({choice.value})
        candidate = replace(draft, preflight_detectors=sorted(enabled))
        try:
            validate_analysis_outputs(candidate.analysis_outputs, candidate.preflight_detectors)
            if persistent:
                save_analysis_settings(candidate)
            draft = candidate
        except (OSError, ValueError) as exc:
            show_action_error(exc, console=console, context="Preflight settings")


def configure_analysis(cfg: Config, *, persistent: bool = False, analysis_only: bool = False,  # noqa: C901
                       runtime_info: list[str] | None = None) -> Config | None:
    """Autosave defaults; keep per-run overrides isolated until Continue."""
    draft = replace(config_for_new_discovery(cfg))
    while True:
        options = [MenuItem("Back" if persistent or analysis_only else "Continue", "done")]
        if analysis_only:
            options.extend([
                MenuItem("Efficiency metrics", "efficiency",
                         "On" if "efficiency" in draft.analysis_outputs else "Off"),
                MenuItem("Failure discovery", "agent",
                         "On" if "agent" in draft.analysis_outputs else "Off"),
                MenuItem("Preflight", "preflight",
                         ", ".join(d.label for d in detectors()
                                   if d.name in draft.preflight_detectors) or "Off"),
            ])
        else:
            enabled = [label for name, label in (
                ("efficiency", "Efficiency metrics"), ("agent", "Failure discovery"),
            ) if name in draft.analysis_outputs]
            if draft.preflight_detectors:
                enabled.append("Preflight")
            options.extend([
                MenuItem("Analyses to run", "analyses", ", ".join(enabled) or "None"),
                MenuItem("Analysis target", "focus", ANALYSIS_FOCUS_LABELS[draft.analysis_focus]),
                MenuItem("Skills", "skills", _skills_summary(draft)),
                MenuItem("Judge policy", "strictness",
                         STRICTNESS_LABELS[draft.behavior_strictness]),
                MenuItem("Continue until coverage plateaus", "plateau",
                         "On" if draft.discovery_wait_for_plateau else "Off"),
                MenuItem("Terminate if stalled", "stalled",
                         "On (extra rounds only)" if draft.discovery_terminate_if_stalled
                         else "Off"),
                MenuItem("Stalled extra rounds", "stall-rounds",
                         str(draft.discovery_stall_rounds)),
                MenuItem("Minimum Good detectors", "minimum",
                         str(draft.min_failure_modes)),
                MenuItem("Optimization weights" if persistent else "Optimization preset",
                         "objective", preset_name(draft)),
            ])
            if persistent:
                from autofa.ui.taxonomy_settings import workflow_label

                options.append(
                    MenuItem(
                        "Taxonomy classification",
                        "taxonomy",
                        workflow_label(
                            draft.taxonomy_first_pass,
                            draft.taxonomy_second_pass,
                        ),
                    )
                )
                options.append(MenuItem("LLM endpoints", "endpoints"))
            options.append(MenuItem("EvalRDL remote server", "eval-server",
                                    draft.data_api_base_url or "not configured"))
            options.append(MenuItem("AutoFA Cloud / library server", "library-server",
                                    draft.remote_library_base_url or "not configured"))
            if persistent:
                options.append(MenuItem("System information", "system-info",
                                        "config, memory, connection details"))
        selected = select_option("Analyses to run" if analysis_only else
                                 "Settings (auto-save)" if persistent else "Analysis settings",
                                 options, allow_back=True)
        if selected is None or is_menu_back(selected) or selected.value == "back":
            return draft if persistent or analysis_only else None
        candidate = replace(draft)
        if selected.value == "analyses":
            candidate = configure_analysis(
                draft, persistent=persistent, analysis_only=True,
            ) or draft
        elif selected.value in {"efficiency", "agent"}:
            enabled = set(draft.analysis_outputs)
            enabled.symmetric_difference_update({selected.value})
            if not enabled and not draft.preflight_detectors:
                show_action_error(ValueError("Keep at least one analysis enabled."),
                                  console=console, context="Analysis settings")
                continue
            candidate.analysis_outputs = sorted(enabled)
        elif selected.value == "preflight":
            candidate = configure_preflight(draft, persistent=persistent)
        elif selected.value == "focus":
            focus = select_analysis_focus(draft.analysis_focus)
            if focus is not None:
                candidate.analysis_focus = focus
        elif selected.value == "minimum":
            minimum = select_min_failure_modes(draft.min_failure_modes)
            if minimum is not None:
                candidate.min_failure_modes = minimum
        elif selected.value == "plateau":
            wait = select_discovery_wait_for_plateau(draft.discovery_wait_for_plateau)
            if wait is not None:
                candidate.discovery_wait_for_plateau = wait
        elif selected.value == "stalled":
            terminate = select_discovery_terminate_if_stalled(draft.discovery_terminate_if_stalled)
            if terminate is not None:
                candidate.discovery_terminate_if_stalled = terminate
        elif selected.value == "stall-rounds":
            rounds = select_discovery_stall_rounds(draft.discovery_stall_rounds)
            if rounds is not None:
                candidate.discovery_stall_rounds = rounds
        elif selected.value == "strictness":
            strictness = select_behavior_strictness(draft.behavior_strictness)
            if strictness is not None:
                candidate.behavior_strictness = strictness
        elif selected.value == "skills":
            _skills_settings(candidate, persistent=persistent)
        elif selected.value == "objective":
            candidate = (configure_objectives(draft, persistent=True) if persistent
                         else select_objective_preset(draft))
        elif selected.value == "taxonomy":
            from autofa.ui.taxonomy_settings import configure_taxonomy

            candidate = configure_taxonomy(
                draft,
                save=save_taxonomy_settings,
            )
        elif selected.value == "endpoints":
            candidate = configure_endpoints(draft)
        elif selected.value == "library-server":
            candidate = configure_library_server(draft, persistent=persistent)
        elif selected.value == "eval-server":
            candidate = configure_eval_server(draft, persistent=persistent)
        elif selected.value == "system-info":
            show_system_information(draft, runtime_info)
        elif selected.value == "done":
            if persistent or analysis_only:
                return draft
            try:
                validate_analysis_outputs(draft.analysis_outputs, draft.preflight_detectors)
                snapshot_guidance(draft)
                return draft
            except (OSError, ValueError) as exc:
                show_action_error(exc, console=console, context="Validate analysis settings")
        if candidate != draft:
            try:
                if persistent and selected.value not in {
                    "skills", "endpoints", "library-server", "eval-server",
                    "preflight", "analyses", "taxonomy",
                }:
                    save_analysis_settings(candidate)
                draft = candidate
            except (OSError, ValueError) as exc:
                show_action_error(exc, console=console, context="Save analysis settings")


def run_settings(*, runtime_info: list[str] | None = None) -> None:
    configure_analysis(load_config(), persistent=True, runtime_info=runtime_info)


def select_behavior_strictness(default: str) -> str | None:
    descriptions = {
        "behavioral": "Check that the described behavior occurred; verify any claimed causes.",
        "contextual": "Also require a plausible link to failure and check later recovery.",
        "causal": (
            "Default. Require evidence the behavior contributed to failure and was not recovered."
        ),
    }
    choice = select_option(
        "Judge policy",
        [MenuItem(STRICTNESS_LABELS[value], value, descriptions[value])
         for value in BEHAVIOR_STRICTNESSES],
        default_index=BEHAVIOR_STRICTNESSES.index(default), allow_back=True,
    )
    return None if choice is None or is_menu_back(choice) else str(choice.value)


def select_min_failure_modes(default: int) -> int | None:
    options = sorted({0, 1, 2, 3, 4, 5, 8, 10, default})
    choice = select_option(
        "Minimum Good detectors (extra rounds if needed)",
        [MenuItem(f"{count} Good detectors" if count else "Disabled", count)
         for count in options],
        default_index=options.index(default), allow_back=True,
    )
    return None if choice is None or is_menu_back(choice) else int(choice.value)


def select_discovery_wait_for_plateau(default: bool) -> bool | None:
    choice = select_option(
        "Continue until coverage plateaus (ordinary discovery)",
        [
            MenuItem("Off", False,
                     "Default. Finish planned rounds once the minimum Good count is met."),
            MenuItem("On", True,
                     "Wait for coverage to plateau and pending evaluations/reviews to finish."),
        ],
        default_index=int(default), allow_back=True,
    )
    return None if choice is None or is_menu_back(choice) else choice.value


def select_discovery_terminate_if_stalled(default: bool) -> bool | None:
    choice = select_option(
        "Terminate if stalled",
        [
            MenuItem("Off", False,
                     "Follow the round and detector targets without an extra-round stall stop."),
            MenuItem("On", True,
                     "Default. Only after the planned rounds, stop after the selected number of "
                     "completed, validated recovery rounds without meaningful library "
                     "improvement, even if the minimum Good count is unmet. "
                     "Stalling cannot end the selected planned rounds early. "
                     "Final dataset scoring and judge review still run."),
        ],
        default_index=int(default), allow_back=True,
    )
    return None if choice is None or is_menu_back(choice) else choice.value


def select_discovery_stall_rounds(default: int) -> int | None:
    options = sorted({1, 2, 3, 4, 5, 6, 8, 10, default})
    choice = select_option(
        "Stalled extra rounds",
        [MenuItem(f"{count} recovery rounds", count,
                  "Consecutive completed, validated extra rounds without meaningful library "
                  "improvement. Starts at zero after planned rounds; failed probes only "
                  "prompt recovery. Applies when Terminate if stalled is On.")
         for count in options],
        default_index=options.index(default), allow_back=True,
    )
    return None if choice is None or is_menu_back(choice) else int(choice.value)
