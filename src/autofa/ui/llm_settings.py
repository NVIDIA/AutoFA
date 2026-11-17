# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact forms for the discovery harness, behavior judge and detector API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from urllib.parse import urlsplit

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Button, TextArea

from autofa.config import Config
from autofa.defaults import harness_default
from autofa.harness.model_catalog import ModelCatalog, load_model_catalog_async
from autofa.harness.settings import (
    HARNESS_NAMES,
    behavior_judge_settings,
    harness_cli,
    harness_model,
    judge_harness,
)
from autofa.harness.thinking import EFFORT_LEVELS
from autofa.llm.endpoints import NO_AUTH, endpoint_for_config, validate_base_url
from autofa.ui.menus import prepare_step
from autofa.ui.pt_app import run_app
from autofa.ui.theme import MENU_STYLE

_CUSTOM_MODEL = "__autofa_custom_model__"
_MODEL_LABELS: dict[str, str] = {
    "gcp/google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "openai/openai/gpt-6-luna": "GPT-6 Luna",
}


def model_label(model: str) -> str:
    return _MODEL_LABELS.get(model, model)


class _Choice:
    def __init__(self, options, value, changed=None):
        self.options = list(options)
        if value not in dict(self.options):
            self.options.append((value, value))
        self.value, self.changed = value, changed
        bindings = KeyBindings()
        for key, delta in (("left", -1), ("right", 1), (" ", 1)):
            bindings.add(key)(lambda event, delta=delta: self.move(delta))
        self.control = FormattedTextControl(self.text, focusable=True, key_bindings=bindings)

    def move(self, delta):
        values = [value for value, _ in self.options]
        self.value = values[(values.index(self.value) + delta) % len(values)]
        if self.changed:
            self.changed()

    def text(self):
        style = "class:selected" if get_app().layout.has_focus(self.control) else ""
        return [(style, f"‹ {dict(self.options)[self.value]} ›")]


class _AgentForm:
    def __init__(self, cfg: Config, save: Callable[[Config], None], *, judge: bool,
                 catalog: ModelCatalog | None = None):
        self.original, self.result, self.save, self.judge = cfg, cfg, save, judge
        self.error = ""
        self.catalog = ModelCatalog()
        self.loading = catalog is None
        harnesses = [("auto", "Same as AutoFA")] if judge else []
        self.harness = _Choice(harnesses + list(HARNESS_NAMES.items()),
                               cfg.behavior_judge_harness if judge else cfg.harness,
                               self.change_harness)
        self.active_harness = judge_harness(cfg) if judge else cfg.harness
        self.catalogs: dict[str, ModelCatalog] = {}
        self.drafts: dict[str, tuple[str, str, str, str]] = {}
        self.catalog_task = None
        self.updating = False
        saved_model = (behavior_judge_settings(cfg)["model"] if judge else
                       harness_model(cfg.harness, cfg.model, cfg=cfg))
        self.model_ids: dict[str, str] = {}
        self.model = _Choice(((saved_model, model_label(saved_model)),
                              (_CUSTOM_MODEL, "Custom model ID…")), saved_model,
                             self.change_model)
        self.custom_model = TextArea(text=saved_model, multiline=False)
        self.custom_model.buffer.on_text_changed += lambda _: self.change_model()
        prefix = "behavior_judge_" if judge else ""
        thinking = (harness_default(self.active_harness, prefix + "thinking", cfg)
                    if self.active_harness == "codex" else
                    getattr(cfg, prefix + "thinking"))
        self.thinking = _Choice([], thinking, self.change_thinking)
        self.effort = _Choice([], getattr(cfg, prefix + "effort"))
        self.budget = TextArea(text=str(getattr(cfg, prefix + "thinking_budget_tokens")),
                               multiline=False)
        self.sync_reasoning()
        if catalog is not None:
            self.apply_catalog(catalog)
        save_button = Button("Save", handler=self.commit, width=10,
                             left_symbol="", right_symbol="")
        cancel = Button("Cancel", handler=lambda: self.app.exit(), width=10,
                        left_symbol="", right_symbol="")

        def row(label, field):
            area = Window(field.control, height=1) if isinstance(field, _Choice) else field
            return VSplit([Window(FormattedTextControl(f"  {label}"), width=18), area], height=1)

        def shown(container, condition):
            return ConditionalContainer(container, filter=Condition(condition))

        rows = [row("Harness", self.harness),
                row("Model", self.model),
                shown(row("Model ID", self.custom_model),
                      lambda: self.model.value == _CUSTOM_MODEL),
                shown(row("Thinking", self.thinking),
                      lambda: self.active_harness == "claude"),
                shown(row("Effort", self.effort), self.show_effort),
                shown(row("Token budget", self.budget), self.show_budget),
        ]

        def fields():
            result = [self.harness.control]
            result.append(self.model.control)
            if self.model.value == _CUSTOM_MODEL:
                result.append(self.custom_model)
            if self.active_harness == "claude":
                result.append(self.thinking.control)
            if self.show_effort():
                result.append(self.effort.control)
            if self.show_budget():
                result.append(self.budget)
            return result + [save_button, cancel]

        bindings = KeyBindings()

        def move(delta):
            controls = fields()
            index = next((i for i, field in enumerate(controls)
                          if self.app.layout.has_focus(field)), -1)
            self.app.layout.focus(controls[(index + delta) % len(controls)])

        for key, delta in (("tab", 1), ("down", 1), ("s-tab", -1), ("up", -1)):
            bindings.add(key)(lambda event, delta=delta: move(delta))
        bindings.add("enter", filter=~(has_focus(save_button) | has_focus(cancel)))(
            lambda event: move(1))
        bindings.add("c-s")(lambda event: self.commit())
        for key in ("escape", "c-c", "c-d"):
            bindings.add(key)(lambda event: self.app.exit())
        content = HSplit([
            Window(FormattedTextControl([("class:title", "  Judge endpoint" if judge
                                          else "  AutoFA harness")]), height=1),
            Window(height=1), *rows, Window(height=1),
            Window(FormattedTextControl(self.hint), height=3, wrap_lines=True),
            ConditionalContainer(Window(FormattedTextControl(
                lambda: [("class:bad", f"  {self.error}")]), height=2, wrap_lines=True),
                filter=Condition(lambda: bool(self.error))),
            VSplit([save_button, cancel], padding=2), Window(height=1),
            Window(FormattedTextControl([("class:hint",
                "  Tab/↑↓ move · ←→ select · Ctrl+S save · Esc cancel")]), height=1),
        ])
        self.app = Application(layout=Layout(content, focused_element=fields()[0]),
                               key_bindings=bindings, style=MENU_STYLE, full_screen=False)
        if catalog is None:
            self.app.pre_run_callables.append(self.start_loading)

    def start_loading(self):
        if self.catalog_task is not None:
            self.catalog_task.cancel()
        self.catalog_task = self.app.create_background_task(self.load_models(self.active_harness))

    async def load_models(self, harness):
        cli_path = harness_cli(self.original, harness)
        catalog = await load_model_catalog_async(cli_path, harness=harness)
        self.catalogs[harness] = catalog
        if self.active_harness == harness:
            self.apply_catalog(catalog)
            self.app.invalidate()

    def change_harness(self):
        selected = (self.original.harness if self.harness.value == "auto"
                    else self.harness.value)
        if selected == self.active_harness:
            return
        self.drafts[self.active_harness] = (
            self.selected_model_id(), self.thinking.value, self.effort.value, self.budget.text)
        self.active_harness = selected
        prefix = "behavior_judge_" if self.judge else ""
        defaults = tuple(harness_default(selected, prefix + name, self.original)
                         for name in ("model", "thinking", "effort", "thinking_budget_tokens"))
        model, thinking, effort, budget = self.drafts.get(selected, defaults)
        self.updating = True
        self.catalog, self.loading = ModelCatalog(), True
        self.model_ids = {}
        self.model.options = ([(model, model_label(model))] if model else []) + [
            (_CUSTOM_MODEL, "Custom model ID…")]
        self.model.value = model or _CUSTOM_MODEL
        self.custom_model.text = model
        self.thinking.value, self.effort.value = thinking, effort
        self.budget.text = str(budget)
        self.updating = False
        self.error = ""
        self.sync_reasoning()
        if self.active_harness in self.catalogs:
            if self.catalog_task is not None:
                self.catalog_task.cancel()
            self.apply_catalog(self.catalogs[self.active_harness])
        else:
            self.start_loading()

    def selected_model_id(self):
        if self.model.value == _CUSTOM_MODEL:
            return self.custom_model.text.strip()
        return self.model_ids.get(self.model.value, self.model.value)

    def selected_model(self):
        return self.catalog.find(self.selected_model_id())

    def apply_catalog(self, catalog):
        current = self.selected_model_id()
        manual = self.model.value == _CUSTOM_MODEL
        self.catalog, self.loading = catalog, False
        self.catalogs[self.active_harness] = catalog
        self.model.options = [(m.value, m.display_name) for m in catalog.models]
        self.model.options.append((_CUSTOM_MODEL, "Custom model ID…"))
        self.model_ids = {m.value: m.value for m in catalog.models}
        for model in catalog.models:
            _MODEL_LABELS[model.value] = model.display_name
            if model.resolved_model:
                _MODEL_LABELS[model.resolved_model] = model.display_name
        match = catalog.find(current)
        if match is not None and not manual:
            self.model.value = match.value
            # Opening a form must not replace a pinned model ID with a moving alias.
            self.model_ids[match.value] = current
        else:
            self.model.value = _CUSTOM_MODEL
            self.custom_model.text = current
        prefix = "behavior_judge_" if self.judge else ""
        original_model = (self.original.behavior_judge_model if self.judge
                          else self.original.model)
        edited = (current != original_model
                  or self.thinking.value != getattr(self.original, prefix + "thinking")
                  or self.effort.value != getattr(self.original, prefix + "effort"))
        # A user may select a model before its metadata arrives. Reconcile that
        # draft now, while leaving an untouched saved configuration unchanged.
        self.sync_reasoning(adjust=edited and self.selected_model() is not None)

    def change_model(self):
        self.error = ""
        if hasattr(self, "thinking") and not self.updating:
            self.sync_reasoning(adjust=True)

    def change_thinking(self):
        self.error = ""
        self.sync_reasoning()

    def reasoning_choices(self):
        model = self.selected_model()
        capability = model.supports_adaptive_thinking if model is not None else None
        # Missing metadata does not mean the saved mode is unsupported. Keep
        # its normal label until the harness reports the actual capability.
        adaptive = capability is True or (capability is None and self.thinking.value == "adaptive")
        modes = [("default", "Model default"),
                 ("adaptive" if adaptive else "enabled", "On")]
        # Adaptive models can require thinking; the harness does not advertise
        # which ones allow disabling it. Do not promise an ineffective Off switch.
        if not adaptive:
            modes.append(("disabled", "Off (if supported)"))
        levels = (model.effort_levels if model is not None else
                  (() if self.active_harness == "codex" else EFFORT_LEVELS[1:]))
        if model is not None and model.supports_effort is False:
            levels = ()
        efforts = [("default", "Model default")] + [
            (v, v.capitalize()) for v in levels
            if v != "default" and (self.active_harness == "codex" or v in EFFORT_LEVELS)]
        return modes, efforts

    def sync_reasoning(self, *, adjust=False):
        adjust = adjust and self.selected_model() is not None
        modes, efforts = self.reasoning_choices()
        for choice, options in ((self.thinking, modes), (self.effort, efforts)):
            if choice.value not in dict(options):
                if adjust:
                    choice.value = (modes[1][0] if choice is self.thinking
                                    and choice.value in ("adaptive", "enabled") else "default")
                else:
                    options.append((choice.value, f"{choice.value.capitalize()} (saved)"))
            choice.options = options

    def show_effort(self):
        model = self.selected_model()
        return ((self.active_harness == "codex" or self.thinking.value in ("default", "adaptive"))
                and (model is None or model.supports_effort is not False))

    def show_budget(self):
        model = self.selected_model()
        return (self.active_harness == "claude" and self.thinking.value == "enabled"
                and (model is None or model.supports_adaptive_thinking is not True))

    def hint(self):
        name = "Claude Code" if self.active_harness == "claude" else "Codex"
        if self.loading:
            text = f"Loading {name} models…"
        elif self.catalog.error:
            text = self.catalog.error + "\n  Enter a custom model ID, or reopen this menu to retry."
        else:
            text = (f"Checks detector behavior using {name}; independent of the discovery model."
                    if self.judge else f"Runs discovery and writes detectors using {name}.")
            model = self.selected_model()
            if model is not None and model.supports_adaptive_thinking:
                text += ("\n  Adaptive thinking: adjust it with effort. "
                         "Some models require thinking.")
            elif self.show_budget():
                text += "\n  Token budget limits thinking; minimum 1,024 tokens."
            elif model is None:
                text += f"\n  Custom model: available controls depend on your {name} setup."
        return [("class:hint", f"  {text}")]

    def default_judge_settings(self):
        # An automatic judge follows the new harness, keeping its own model.
        configured = harness_default(self.active_harness, "behavior_judge_model", self.original)
        preferred = self.catalog.find(configured)
        effort = harness_default(self.active_harness, "behavior_judge_effort", self.original)
        if preferred is not None and effort not in preferred.effort_levels:
            effort = preferred.default_effort or "default"
        thinking = harness_default(self.active_harness, "behavior_judge_thinking", self.original)
        if (thinking == "adaptive" and preferred
                and preferred.supports_adaptive_thinking is False):
            thinking = "enabled"
        return {
            "behavior_judge_model": configured,
            "behavior_judge_thinking": thinking,
            "behavior_judge_effort": effort,
            "behavior_judge_thinking_budget_tokens": harness_default(
                self.active_harness, "behavior_judge_thinking_budget_tokens", self.original),
        }

    def commit(self):
        try:
            model = self.selected_model_id()
            if not model:
                raise ValueError("Enter a model ID.")
            try:
                budget = int(self.budget.text)
            except ValueError:
                raise ValueError("Enter a whole-number thinking token budget.") from None
            if budget < 1024:
                raise ValueError("Thinking token budget must be at least 1,024.")
            modes, efforts = self.reasoning_choices()
            thinking = (harness_default(self.active_harness,
                                       "behavior_judge_thinking" if self.judge else "thinking",
                                       self.original)
                        if self.active_harness == "codex" else self.thinking.value)
            effort = self.effort.value
            # Preserve existing settings on an unchanged save; model changes use
            # supported controls and never carry max effort over to a smaller model.
            prefix = "behavior_judge_" if self.judge else ""
            values = {prefix + "thinking": thinking, prefix + "effort": effort,
                      prefix + "thinking_budget_tokens": budget}
            values["behavior_judge_model" if self.judge else "model"] = model
            if self.judge:
                values["behavior_judge_harness"] = self.harness.value
            else:
                values["harness"] = self.active_harness
                if (self.active_harness != self.original.harness
                        and self.original.behavior_judge_harness == "auto"):
                    values.update(self.default_judge_settings())
            candidate = replace(self.original, **values)
            if candidate != self.original:
                if thinking not in dict(modes):
                    raise ValueError("Choose a thinking setting supported by this model.")
                if (self.show_effort() and self.selected_model() is not None
                        and effort not in dict(efforts)):
                    raise ValueError("Choose an effort level supported by this model.")
                self.save(candidate)
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            return
        self.result = candidate
        self.app.exit()


def edit_agent(cfg: Config, *, save: Callable[[Config], None], judge: bool = False) -> Config:
    form = _AgentForm(cfg, save, judge=judge)
    prepare_step()
    run_app(form.app)
    return form.result


def _provider(url: str, cfg: Config | None = None) -> int:
    default = (cfg or Config()).hub_base_url.strip().rstrip("/")
    if default and url.rstrip("/") == default:
        return 0
    try:
        if urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"}:
            return 2
    except ValueError:
        pass
    return 1


class _EndpointForm:
    def __init__(self, cfg: Config, save: Callable[[Config], None]):
        self.original, self.result, self.save = cfg, cfg, save
        self.error = ""
        self.url = TextArea(text=cfg.feature_base_url or cfg.hub_base_url, multiline=False)
        self.model = TextArea(text=cfg.feature_model, multiline=False)
        key = cfg.feature_api_key_env or cfg.hub_api_key_env
        self.key = TextArea(text="" if key == NO_AUTH else key, multiline=False)
        self.initial = self.values()
        self.provider = _provider(self.url.text, cfg)
        self.api = _Choice((("auto", "Auto (prefer Decisions)"),
                            ("responses", "Responses"),
                            ("decisions", "Decisions (required for labels)"),
                            ("chat_completions", "Chat Completions")), cfg.feature_api)
        # Existing explicit classifier overrides remain configurable in TOML.
        self.shared_classifier_model = (not cfg.judge_model or (
            cfg.judge_model == cfg.feature_model
            and not cfg.judge_base_url and not cfg.judge_api_key_env))
        self.url.buffer.on_text_changed += lambda _: self._url_changed()

        provider_keys = KeyBindings()
        for key_name, delta in (("left", -1), ("right", 1), (" ", 1)):
            provider_keys.add(key_name)(lambda event, delta=delta: self.choose_provider(delta))
        self.provider_control = FormattedTextControl(
            self.provider_text, focusable=True, key_bindings=provider_keys,
        )
        self.save_button = Button("Save", handler=self.commit, width=10,
                                  left_symbol="", right_symbol="")
        cancel = Button("Cancel", handler=self.cancel, width=10,
                        left_symbol="", right_symbol="")

        def row(label, area):
            return VSplit([Window(FormattedTextControl(f"  {label}"), width=18), area], height=1)

        content = HSplit([
            Window(FormattedTextControl([("class:title", "  Detector LLM endpoint")]), height=1),
            Window(height=1),
            VSplit([Window(FormattedTextControl("  Provider"), width=18),
                    Window(self.provider_control, height=1)], height=1),
            Window(height=1),
            row("Server URL", self.url),
            row("Model", self.model),
            row("Key variable", self.key),
            row("API preference", Window(self.api.control, height=1)),
            Window(height=1),
            Window(FormattedTextControl(self.hint), height=2, wrap_lines=True),
            ConditionalContainer(Window(
                FormattedTextControl(lambda: [("class:bad", f"  {self.error}")]),
                height=2, wrap_lines=True,
            ), filter=Condition(lambda: bool(self.error))),
            Window(height=1),
            VSplit([self.save_button, cancel], padding=2),
            Window(height=1),
            Window(FormattedTextControl([
                ("class:hint", "  Tab/↑↓ move · ←→ select · Ctrl+S save · Esc cancel"),
            ]), height=1),
        ])
        bindings = KeyBindings()

        def move_focus(delta):
            fields = [self.provider_control, self.url, self.model, self.key,
                      self.api.control, self.save_button, cancel]
            index = next(i for i, field in enumerate(fields) if self.app.layout.has_focus(field))
            self.app.layout.focus(fields[(index + delta) % len(fields)])

        @bindings.add("tab")
        @bindings.add("down")
        def next_field(event):
            move_focus(1)

        @bindings.add("s-tab")
        @bindings.add("up")
        def previous_field(event):
            move_focus(-1)

        @bindings.add("enter", filter=~(has_focus(self.save_button) | has_focus(cancel)))
        def enter(event):
            move_focus(1)

        @bindings.add("c-s")
        def save_form(event):
            self.commit()

        @bindings.add("escape")
        @bindings.add("c-c")
        @bindings.add("c-d")
        def cancel_form(event):
            self.cancel()

        self.app = Application(layout=Layout(content, focused_element=self.provider_control),
                               key_bindings=bindings, style=MENU_STYLE, full_screen=False)

    def values(self) -> tuple[str, str, str]:
        return self.url.text.strip(), self.model.text.strip(), self.key.text.strip()

    def _url_changed(self):
        self.provider = _provider(self.url.text.strip(), self.original)

    def provider_text(self):
        result = []
        for index, label in enumerate(("Configured default", "Other provider", "Local server")):
            result.append(("class:selected" if index == self.provider else "class:dim",
                           f" {label} "))
            result.append(("", "  "))
        return result

    def choose_provider(self, delta: int):
        target = (self.provider + delta) % 3
        if target == 0:
            cfg = self.original
            self.url.text, self.model.text, self.key.text = (
                cfg.hub_base_url, cfg.feature_model, cfg.hub_api_key_env,
            )
        else:
            self.url.text = "http://localhost:8000/v1" if target == 2 else ""
            self.model.text = ""
            self.key.text = "" if target == 2 else "LLM_API_KEY"
        self.provider = target
        self.error = ""

    def hint(self):
        focus = self.app.layout.current_control
        if focus is self.api.control:
            text = {
                "auto": "Checks this server and model. Prefers Decisions for labels; "
                        "Responses for text. Chat handles unsupported APIs or request formats.",
                "decisions": "Requires Decisions for labels. Text and quote extraction "
                             "use Responses when supported.",
                "responses": "Use Responses for labels, text and quote extraction.",
                "chat_completions": "Use Chat Completions for labels and generated text.",
            }[self.api.value]
        elif focus is self.key.control:
            text = "Environment variable name, e.g. LLM_API_KEY. Blank = no API key."
        elif focus is self.url.control:
            text = "API root including /v1; omit /decisions, /responses or /chat/completions."
        elif focus is self.model.control:
            text = "Exact model ID served by this endpoint."
        elif focus is self.provider_control:
            text = ("Local: connect to a running vLLM or SGLang server." if self.provider == 2 else
                    "Other provider: any OpenAI-compatible server." if self.provider == 1 else
                    "Connects to the configured inference endpoint."
                    if self.original.hub_base_url.strip() else
                    "No default endpoint is configured; choose Other provider or Local server.")
        else:
            text = "Used by LLM detectors."
        return [("class:hint", f"  {text}")]

    def candidate(self) -> Config:
        url, model, key = self.values()
        if not url:
            raise ValueError("Enter the API root URL of an OpenAI-compatible endpoint.")
        url = validate_base_url(url)
        if not model:
            raise ValueError("Enter the model ID served by this endpoint.")
        candidate = replace(self.original)
        candidate.feature_api = self.api.value
        if self.values() != self.initial:
            candidate.feature_base_url = url
            candidate.feature_model = model
            candidate.feature_api_key_env = key or NO_AUTH
        if self.shared_classifier_model:
            candidate.judge_model = model
        endpoint_for_config(candidate)
        endpoint_for_config(candidate, "judge")
        return candidate

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

    def cancel(self):
        self.app.exit()


def edit_endpoints(cfg: Config, *, save: Callable[[Config], None]) -> Config:
    form = _EndpointForm(cfg, save)
    prepare_step()
    run_app(form.app)
    return form.result
