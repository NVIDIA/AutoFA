# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery depth picker with workload-aware estimates and optional auto-start."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import wrap

from rich.console import Console

from autofa.config import Config, validate_min_failure_modes
from autofa.defaults import default_value
from autofa.estimates import Estimator, Workload
from autofa.ui.menus import MenuItem, prepare_step, select_option
from autofa.ui.theme import GREEN

console = Console()

@dataclass(frozen=True)
class DiscoveryDepth:
    max_rounds: int
    top_x: int
    label: str
    est_min_minutes: int
    est_max_minutes: int
    min_failure_modes: int = default_value("min_failure_modes")
    est_median_minutes: int = 0
    est_rough: bool = False

    @property
    def estimate_label(self) -> str:
        from autofa.estimates import Estimate

        return Estimate(self.est_min_minutes, self.est_max_minutes, "",
                        self.est_median_minutes, rough=self.est_rough).time_label


def _timing_header(estimator: Estimator, minimum: int):
    from prompt_toolkit.application.current import get_app_session

    width = max(20, get_app_session().output.get_size().columns - 4)
    estimate = estimator.estimate(1)
    model = estimator.signature.get("model", "selected model")
    judge = estimator.signature.get("judge_model", "selected judge")
    if not estimate.high_minutes:
        history = "Without recorded timings, ETA calibrates as this run progresses."
    elif estimate.rough:
        history = "Without enough matching runs, timings are only a rough historical guide."
    else:
        history = "Timings use matching past runs; their observed range is not a guarantee."
    notes = [
        ("class:accent", "Timing is approximate"),
        ("class:hint", "Models, log size/depth and validation work affect duration."),
        ("class:hint", f"Extra rounds to reach {minimum} Good detectors can extend the run."
         if minimum else "No minimum Good-detector target; timing covers the selected rounds."),
        ("class:hint", history),
        ("class:hint", "Live ETA refreshes each second, using measured pace "
         "after 2 completed rounds."),
        ("class:hint", "Cost varies with model pricing, token usage and extra rounds; "
         "no reliable upfront estimate."),
        ("class:dim", f"Discovery: {model} · Judge: {judge}"),
    ]
    return [(style, f"  {line}\n") for style, text in notes for line in wrap(text, width=width)] + [
        ("", "\n")]


def select_discovery_depth(
    *,
    default_rounds: int | None = None,
    default_top_x: int | None = None,
    default_min_failure_modes: int | None = None,
    min_failure_modes_override: int | None = None,
    estimator: Estimator | None = None,
    auto_start_seconds: float | None = None,
) -> DiscoveryDepth | None:
    """Ask how deep to search. Returns None if cancelled."""
    if default_rounds is None:
        default_rounds = default_value("max_rounds")
    if default_top_x is None:
        default_top_x = default_value("top_x")
    if default_min_failure_modes is None:
        default_min_failure_modes = default_value("min_failure_modes")
    validate_min_failure_modes(default_min_failure_modes)
    if min_failure_modes_override is not None:
        validate_min_failure_modes(min_failure_modes_override)
    estimator = estimator or Estimator(Config(), Workload())
    prepare_step()
    presets: list[tuple[str, int, int, str]] = [
        ("quick", 1, default_top_x, "Quick look"),
        ("standard", 3, default_top_x, "Standard"),
        ("indepth", 5, default_top_x, "In-depth"),
        ("deep", 8, default_top_x, "Deep dive"),
        ("custom", 0, default_top_x, "Custom rounds…"),
    ]
    # Prefer matching default_rounds in the list
    default_idx = 1
    for i, (_, r, _, _) in enumerate(presets):
        if r == default_rounds:
            default_idx = i
            break

    items: list[MenuItem] = []
    for key, rounds, top_x, title in presets:
        if key == "custom":
            items.append(MenuItem(f"{title}", key, detail=""))
            continue
        estimate = estimator.estimate(rounds, top_x)
        items.append(
            MenuItem(
                f"{title}",
                key,
                detail=(f"  {rounds} round{'s' if rounds != 1 else ''}  ·  "
                        f"{estimate.time_label}"),
            )
        )

    min_failure_modes = (default_min_failure_modes if min_failure_modes_override is None
                         else min_failure_modes_override)
    while True:
        choice = select_option(
            "How deep should this search go?",
            items,
            default_index=1 if auto_start_seconds is not None else default_idx,
            allow_cancel=True,
            header=lambda: _timing_header(estimator, min_failure_modes),
            clear_screen=True,
            auto_select_seconds=auto_start_seconds,
            footer=[("class:hint", "  Timing covers selected rounds, excluding any extension.\n"
                     f"  Minimum Good detectors: {min_failure_modes} (change in Settings).\n")],
        )
        auto_start_seconds = None  # Custom/back navigation never re-arms the timer.
        if choice is None:
            return None

        top_x = default_top_x
        if choice.value == "custom":
            rounds = _pick_custom_rounds(default_rounds, estimator, default_top_x)
            if rounds is None:
                continue  # Esc from custom → depth presets
            label = f"Custom ({rounds} rounds)"
        else:
            rounds = next(r for k, r, _, _ in presets if k == choice.value)
            label = next(t for k, _, _, t in presets if k == choice.value)

        estimate = estimator.estimate(rounds, top_x)
        depth = DiscoveryDepth(
            max_rounds=rounds,
            top_x=top_x,
            label=label,
            est_min_minutes=estimate.low_minutes,
            est_max_minutes=estimate.high_minutes,
            est_median_minutes=estimate.median_minutes,
            est_rough=estimate.rough,
            min_failure_modes=min_failure_modes,
        )
        minimum = (
            f" · continue until {depth.min_failure_modes} Good failure detectors"
            if depth.min_failure_modes else " · no minimum Good-detector target"
        )
        console.print(
            f"[{GREEN}]Depth:[/{GREEN}] {depth.label}  ·  "
            f"[bold]{depth.max_rounds}[/bold] selected rounds  ·  "
            f"{depth.estimate_label}{minimum}"
        )
        return depth


def _pick_custom_rounds(default_rounds: int, estimator: Estimator, top_x: int) -> int | None:
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.processors import BeforeInput

    from autofa.ui.pt_app import run_app
    from autofa.ui.theme import MENU_STYLE

    value = Buffer(multiline=False)
    control = BufferControl(value, input_processors=[BeforeInput("  Rounds: ")])
    error = ""
    selected = None
    kb = KeyBindings()

    def parse() -> int | None:
        text = value.text.strip()
        if not text:
            return default_rounds
        # Bound input length before int conversion; accept any practical positive count.
        if len(text) <= 9 and text.isascii() and text.isdecimal() and int(text) > 0:
            return int(text)
        return None

    @kb.add("enter")
    def accept(event):
        nonlocal selected, error
        selected = parse()
        if selected is None:
            error = "Enter a whole number from 1 to 999999999."
            event.app.invalidate()
        else:
            event.app.exit()

    @kb.add("escape")
    @kb.add("c-c")
    def back(event):
        event.app.exit()

    def clear_error(_buffer):
        nonlocal error
        error = ""

    value.on_text_changed += clear_error

    def preview():
        n = parse()
        estimate = estimator.estimate(n, top_x) if n else None
        message = error or (estimate.time_label if estimate else "Enter a positive number")
        return [("class:warning" if error else "class:hint", "  " + message)]

    app = Application(
        layout=Layout(HSplit([
            Window(FormattedTextControl([
                ("class:title", "  Custom discovery rounds\n"),
                ("class:hint", f"  Type a number · Enter confirms (blank = {default_rounds})\n"
                 "  Esc goes back · auto-start is paused while entering a number\n\n"),
            ]), height=4),
            Window(control, height=1),
            Window(FormattedTextControl(preview), height=2),
        ]), focused_element=control),
        key_bindings=kb, style=MENU_STYLE, full_screen=True,
    )
    run_app(app)
    return selected
