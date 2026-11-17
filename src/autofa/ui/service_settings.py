# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settings form for an optional evaluation server."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from prompt_toolkit.application import Application
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Button, TextArea

from autofa.config import Config
from autofa.services import eval_server_name, validate_service_address
from autofa.ui.menus import prepare_step
from autofa.ui.pt_app import run_app
from autofa.ui.theme import MENU_STYLE


class _EvalServerForm:
    def __init__(self, cfg: Config, save: Callable[[Config], None]):
        self.original, self.result, self.save = cfg, cfg, save
        self.error = ""
        self.url = TextArea(text=cfg.data_api_base_url, multiline=False)
        self.name = TextArea(text=cfg.data_api_name, multiline=False)
        save_button = Button("Save", handler=self.commit,
                             left_symbol="", right_symbol="")
        cancel_button = Button("Cancel", handler=lambda: self.app.exit(),
                               left_symbol="", right_symbol="")
        fields = [self.url, self.name, save_button, cancel_button]
        keys = KeyBindings()

        def move(delta):
            index = next(i for i, field in enumerate(fields) if self.app.layout.has_focus(field))
            self.app.layout.focus(fields[(index + delta) % len(fields)])

        for key, delta in (("tab", 1), ("down", 1), ("s-tab", -1), ("up", -1)):
            keys.add(key)(lambda event, delta=delta: move(delta))
        keys.add("enter", filter=~(has_focus(save_button) | has_focus(cancel_button)))(
            lambda event: move(1))
        keys.add("c-s")(lambda event: self.commit())
        for key in ("escape", "c-c", "c-d"):
            keys.add(key)(lambda event: self.app.exit())

        def row(label, field):
            return VSplit([Window(FormattedTextControl(f"  {label}"), width=19), field], height=1)

        content = HSplit([
            Window(FormattedTextControl([("class:title", "  EvalRDL remote server")]), height=1),
            Window(height=1), row("Server address", self.url), row("Menu name", self.name),
            Window(height=1), Window(FormattedTextControl([("class:hint",
                "  Use an EvalRDL-compatible run catalog and AutoFA export API. "
                "Clear the address to hide the remote dataset menu item.")]),
                height=3, wrap_lines=True),
            Window(FormattedTextControl(lambda: [("class:bad", f"  {self.error}")]),
                   height=2, wrap_lines=True),
            VSplit([save_button, cancel_button], padding=2),
            Window(FormattedTextControl([("class:hint",
                "  Tab/↑↓ move · Ctrl+S save · Esc cancel")]), height=1),
        ])
        self.app = Application(layout=Layout(content, focused_element=self.url),
                               key_bindings=keys, style=MENU_STYLE, full_screen=False)

    def commit(self):
        try:
            candidate = replace(self.original, data_api_base_url=validate_service_address(
                self.url.text, name="EvalRDL remote server"), data_api_name=self.name.text)
            candidate.data_api_name = eval_server_name(candidate)
            if candidate != self.original:
                self.save(candidate)
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            return
        self.result = candidate
        self.app.exit()


def edit_eval_server(cfg: Config, *, save: Callable[[Config], None]) -> Config:
    form = _EvalServerForm(cfg, save)
    prepare_step()
    run_app(form.app)
    return form.result
