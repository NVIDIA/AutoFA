# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure report publication and full-trace transfers to another AutoFA server."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Button, TextArea

from autofa.config import Config
from autofa.defaults import default_value
from autofa.services import validate_service_address
from autofa.ui.llm_settings import _Choice
from autofa.ui.menus import prepare_step
from autofa.ui.pt_app import run_app
from autofa.ui.theme import MENU_STYLE


def validate_library_url(value: str, *, mode: str) -> str:
    """Accept an HTTP(S) address, without embedded credentials or browser state."""
    return validate_service_address(value, name="library server", origin_only=mode == "server")


class _LibraryServerForm:
    def __init__(self, cfg: Config, save: Callable[[Config], None]):
        self.original, self.result, self.save = cfg, cfg, save
        self.error = ""
        self.url = TextArea(text=cfg.remote_library_base_url, multiline=False)
        self.token_env = TextArea(text=cfg.remote_library_token_env, multiline=False)
        self.previous_mode = cfg.remote_library_mode
        self.saved_fields = {
            cfg.remote_library_mode: (cfg.remote_library_base_url, cfg.remote_library_token_env),
        }
        self.mode = _Choice(
            [("cloud", "Cloud · lightweight report"),
             ("server", "AutoFA server · report and full local traces")],
            cfg.remote_library_mode, self.change_mode,
        )
        save_button = Button("Save", handler=self.commit, width=10,
                             left_symbol="", right_symbol="")
        cancel_button = Button("Cancel", handler=lambda: self.app.exit(), width=10,
                               left_symbol="", right_symbol="")

        def row(label, field):
            area = Window(field.control, height=1) if isinstance(field, _Choice) else field
            return VSplit([Window(FormattedTextControl(f"  {label}"), width=19), area], height=1)

        fields = [self.mode.control, self.url, self.token_env, save_button, cancel_button]
        bindings = KeyBindings()

        def move(delta):
            index = next((i for i, field in enumerate(fields)
                          if self.app.layout.has_focus(field)), -1)
            self.app.layout.focus(fields[(index + delta) % len(fields)])

        for key, delta in (("tab", 1), ("down", 1), ("s-tab", -1), ("up", -1)):
            bindings.add(key)(lambda event, delta=delta: move(delta))
        bindings.add("enter", filter=~(has_focus(save_button) | has_focus(cancel_button)))(
            lambda event: move(1))
        bindings.add("c-s")(lambda event: self.commit())
        for key in ("escape", "c-c", "c-d"):
            bindings.add(key)(lambda event: self.app.exit())
        content = HSplit([
            Window(FormattedTextControl([
                ("class:title", "  AutoFA Cloud / library server")]), height=1),
            Window(height=1), row("Publication", self.mode), row("Server address", self.url),
            row("Token variable", self.token_env), Window(height=1),
            Window(FormattedTextControl(self.hint), height=4, wrap_lines=True),
            ConditionalContainer(Window(FormattedTextControl(
                lambda: [("class:bad", f"  {self.error}")]), height=2, wrap_lines=True),
                filter=Condition(lambda: bool(self.error))),
            VSplit([save_button, cancel_button], padding=2), Window(height=1),
            Window(FormattedTextControl([("class:hint",
                "  Tab/↑↓ move · ←→ select · Ctrl+S save · Esc cancel")]), height=1),
        ])
        self.app = Application(layout=Layout(content, focused_element=self.mode.control),
                               key_bindings=bindings, style=MENU_STYLE, full_screen=False)

    def change_mode(self):
        self.saved_fields[self.previous_mode] = (self.url.text, self.token_env.text)
        self.previous_mode = self.mode.value
        fallback = (("", "AUTOFA_LIBRARY_TOKEN") if self.mode.value == "server" else
                    (default_value("remote_library_base_url"),
                     default_value("remote_library_token_env")))
        self.url.text, self.token_env.text = self.saved_fields.get(self.mode.value, fallback)
        self.error = ""

    def hint(self):
        if self.mode.value == "server":
            text = ("Transfers the report and full local logs to another `autofa serve` machine. "
                    "Set the token variable in your shell or profile .env to that server's "
                    "token. The token itself is never saved in autofa.toml. "
                    "Your local dashboard stays available.")
        else:
            text = ("Publishes a lightweight report with available EvalRDL trace links. "
                    "Use autofa login or the configured token variable to sign in. "
                    "Your local dashboard stays available.")
        return [("class:hint", f"  {text} Clear the address to disable remote publication.")]

    def candidate(self) -> Config:
        url = validate_library_url(self.url.text, mode=self.mode.value)
        token_env = self.token_env.text.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env):
            raise ValueError("Enter an environment variable name, e.g. AUTOFA_LIBRARY_TOKEN.")
        if self.mode.value == "server" and token_env == "GITLAB_TOKEN":
            raise ValueError("Use a server token variable, e.g. AUTOFA_LIBRARY_TOKEN.")
        return replace(self.original, remote_library_mode=self.mode.value,
                       remote_library_base_url=url, remote_library_token_env=token_env)

    def commit(self):
        try:
            candidate = self.candidate()
            if candidate != self.original:
                self.save(candidate)
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            return
        self.result = candidate
        self.app.exit()


def edit_library_server(cfg: Config, *, save: Callable[[Config], None]) -> Config:
    form = _LibraryServerForm(cfg, save)
    prepare_step()
    run_app(form.app)
    return form.result
