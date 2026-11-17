# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive terminal menus: cancellation, countdowns, folder browser and app runner."""

import asyncio
import json
import logging
import sys
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application, create_app_session, get_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import DummyOutput

from autofa.config import Config
from autofa.data.sources import recognized_log_folder
from autofa.estimates import Estimator, Workload
from autofa.ui import menus
from autofa.ui.menus import MENU_BACK
from autofa.ui.pt_app import run_app
from autofa.wizard import depth

NEMO_GYM = "NeMo Gym log folder recognized — use this folder?"
HARBOR = "Harbor log folder recognized — use this folder?"


def press(app, key, event=None):
    binding = next(b for b in app.key_bindings.bindings if b.keys == (key,))
    binding.handler(event or SimpleNamespace(app=SimpleNamespace(exit=lambda: None)))


def rendered_text(app):
    return "".join(t for _, t in app.layout.container.children[0].content.text())


@pytest.mark.parametrize(("options", "choice", "allow_back"), [
    ({"allow_back": True}, MENU_BACK, True),  # nested wizard steps can go back
    ({"default_yes": False}, None, False),  # confirmations safely cancel on Escape
])
def test_yes_no_returns_back_or_cancels_without_choosing(monkeypatch, options, choice, allow_back):
    captured = {}

    def fake_select_option(*args, **kwargs):
        captured.update(kwargs)
        return choice

    monkeypatch.setattr(menus, "select_option", fake_select_option)

    assert menus.select_yes_no("Continue?", **options) is choice
    assert captured["allow_back"] is allow_back
    assert captured["allow_cancel"] is not allow_back


@pytest.mark.parametrize("key", [Keys.Escape, Keys.ControlC])
def test_real_cancel_bindings_return_to_previous_step(monkeypatch, key):
    monkeypatch.setattr(menus, "run_app", lambda app: press(app, key))
    result = menus.select_option("Nested", ["one", "two"], allow_cancel=False,
                                 allow_back=True, clear_screen=False)
    assert result is MENU_BACK


def test_unreadable_folder_explains_problem_in_browser(tmp_path, monkeypatch):
    import os

    def denied(path):
        raise PermissionError("test permission denied")

    monkeypatch.setattr(os, "scandir", denied)

    def inspect(app):
        assert "Cannot read this folder" in rendered_text(app)
        assert "permission denied" in rendered_text(app)
        press(app, Keys.Escape)

    monkeypatch.setattr(menus, "run_app", inspect)
    assert menus.browse_filesystem(tmp_path, clear_screen=False) is None


def test_empty_multi_selection_stays_open_with_explanation(monkeypatch):
    def inspect(app):
        exited = []
        event = SimpleNamespace(app=SimpleNamespace(exit=lambda: exited.append(True),
                                                    invalidate=lambda: None))
        press(app, Keys.Up, event)
        press(app, Keys.ControlM, event)
        assert exited == []
        assert "Select at least one item" in rendered_text(app)
        press(app, Keys.Escape, event)

    monkeypatch.setattr(menus, "run_app", inspect)
    assert menus.select_items_multi([menus.MenuItem("a", "a"), menus.MenuItem("b", "b")],
                                    clear_screen=False) is None


# Unattended start: the countdown chooses the current highlight unless the user cancels.


@pytest.mark.parametrize("key,expected", [(None, "standard"), ("q", None)])
def test_real_countdown_selects_current_highlight(monkeypatch, key, expected):
    monkeypatch.setattr(menus, "prepare_step", lambda: None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        def run(app):
            async def interact():
                await asyncio.sleep(0.02)
                if key:
                    pipe.send_text(key)

            app.pre_run_callables.append(lambda: app.create_background_task(interact()))
            app.run()

        monkeypatch.setattr(menus, "run_app", run)
        result = menus.select_option("Depth", [menus.MenuItem("Quick", "quick"),
                                                menus.MenuItem("Standard", "standard")],
                                     default_index=1, auto_select_seconds=0.12)
    assert (result.value if result else None) == expected


def test_custom_back_does_not_restart_countdown(monkeypatch):
    choices = iter([menus.MenuItem("Custom", "custom"), None])
    timers = []

    def pick(*args, **kwargs):
        timers.append(kwargs.get("auto_select_seconds"))
        if len(timers) == 1:
            assert kwargs["default_index"] == 1  # Standard starts highlighted.
        return next(choices)

    monkeypatch.setattr(depth, "select_option", pick)
    monkeypatch.setattr(depth, "_pick_custom_rounds", lambda *a: None)
    monkeypatch.setattr(depth, "prepare_step", lambda: None)
    assert depth.select_discovery_depth(default_rounds=8, auto_start_seconds=60) is None
    assert timers == [60, None]


@pytest.mark.parametrize("keys,expected", [
    ("17\r", 17), ("\x1b", None), ("0\r\x15" + "9\r", 9), ("2.5\r\x15" + "12\r", 12),
], ids=["typed", "escape", "zero_rejected", "fraction_rejected"])
def test_custom_rounds_accepts_typed_number_and_validates(monkeypatch, keys, expected):
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        def run(app):
            app.ttimeoutlen = app.timeoutlen = 0.01  # Deliver a lone Escape promptly.
            pipe.send_text(keys)
            app.run()

        monkeypatch.setattr("autofa.ui.pt_app.run_app", run)
        assert depth._pick_custom_rounds(3, Estimator(Config(), Workload(), []), 5) == expected


def test_countdown_opens_highlighted_custom_without_a_timer(monkeypatch):
    monkeypatch.setattr(menus, "prepare_step", lambda: None)
    monkeypatch.setattr(depth, "prepare_step", lambda: None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        def run_menu(app):
            async def navigate():
                await asyncio.sleep(0.02)
                pipe.send_text("\x1b[F")  # End: highlight Custom; let the timer choose it.

            app.pre_run_callables.append(lambda: app.create_background_task(navigate()))
            app.run()

        def run_custom(app):
            assert not app.pre_run_callables  # No timer while typing a custom value.
            pipe.send_text("23\r")
            app.run()

        monkeypatch.setattr(menus, "run_app", run_menu)
        monkeypatch.setattr("autofa.ui.pt_app.run_app", run_custom)
        selected = depth.select_discovery_depth(
            estimator=Estimator(Config(), Workload(), []), auto_start_seconds=0.3,
        )
    assert selected.max_rounds == 23


# Log-folder recognition in the dataset browser.


@pytest.mark.parametrize("files,expected", [
    ({"rollouts.jsonl": '{"_ng_task_index":0}'}, NEMO_GYM),
    ({"run_config.yml": "framework_name: nemo_gym"}, NEMO_GYM),
    ({"config.json": '{"jobs_dir":"jobs"}'}, HARBOR),
    # A Harbor job is recognized from a config prefix, without parsing it all.
    ({"config.json": json.dumps({"jobs_dir": "jobs", "extra": "x" * 70000})}, HARBOR),
    ({"eval_metadata.json": '{"model":"m","dataset":"b"}'}, None),
    ({"rollouts.jsonl": '{"_ng_task_index":0}', "invoice.pdf": "{}"}, None),
    ({"rollouts.jsonl": '{"_ng_task_index":0}', "output.swebench.jsonl": "{}"}, None),
    ({"config.json": '{"job_name":"unrelated"}',
      "artifacts/nested-harbor/config.json": '{"jobs_dir":"jobs"}'}, None),
], ids=["gym_rollouts", "gym_config", "harbor", "large_harbor_config", "metadata_only",
        "mixed_download", "foreign_stream", "nested_harbor"])
def test_log_folder_signatures(tmp_path, files, expected):
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    assert recognized_log_folder(tmp_path) == expected


def test_launcher_layout_with_long_config_and_arbitrary_artifacts(tmp_path):
    # Matches Downloads/autofa-logs/<job>/artifacts, including a framework
    # declaration beyond the former 8 KiB read limit and scheduler artifacts.
    root = tmp_path / 'autofa-logs'
    job = root / 'job-id'
    artifacts = job / 'artifacts'
    artifacts.mkdir(parents=True)
    (job / 'logs').mkdir()
    (job / 'run.sub').write_text('scheduler job')
    (artifacts / 'run_config.yml').write_text('# config\n' * 1200 + 'framework_name: nemo_gym')
    (artifacts / 'evaluator_rollouts.jsonl').touch()
    (artifacts / 'cache').mkdir()
    (artifacts / 'report.html').write_text('<html></html>')
    for i in range(260):
        (artifacts / f'chart-{i}.png').touch()
    for folder in (root, job, artifacts):
        assert recognized_log_folder(folder) == NEMO_GYM
    # A wrapper with multiple jobs is a collection, not one recognized run.
    second = root / 'another-job' / 'artifacts'
    second.mkdir(parents=True)
    (second / 'run_config.yml').write_text('framework_name: nemo_gym')
    assert recognized_log_folder(root) is None


def test_log_folder_recognized_inside_browser(tmp_path, monkeypatch):
    (tmp_path / "samples.jsonl").write_text('{"_ng_task_index": 0}')
    (tmp_path / "eval_metadata.json").write_text('{"model": "example"}')
    monkeypatch.setattr(menus, "prepare_step", lambda: None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        def run(app):
            assert "NeMo Gym log folder recognized" in rendered_text(app)
            pipe.send_text("\r")
            app.run()

        monkeypatch.setattr(menus, "run_app", run)
        result = menus.browse_filesystem(tmp_path, folder_description=recognized_log_folder)
    assert result == tmp_path


# The prompt_toolkit app runner under a live asyncio loop.

# A synchronous caller runs the menu directly; a caller inside a running loop uses a
# worker thread, whose error must still propagate after the streams are restored.
@pytest.mark.parametrize("inside_running_loop,exit_with_error", [(False, False), (True, True)])
def test_background_output_suspends_menu_and_restores_streams(
    monkeypatch, inside_running_loop, exit_with_error
):
    messages = ("background stdout", "background stderr", "background log")
    writes: list[tuple[str, bool]] = []
    delivered = asyncio.Event()

    class RecordingOutput(DummyOutput):
        def write_raw(self, data):
            if any(message in data for message in messages):
                writes.append((data, app._running_in_terminal))
                if all(message in "".join(text for text, _ in writes) for message in messages):
                    delivered.set()

    output = RecordingOutput()
    # The async entry point runs in another thread, where prompt_toolkit falls
    # back to its default session. Give that session the same simulated terminal.
    monkeypatch.setattr(get_app_session(), "_output", output)
    original_streams = sys.stdout, sys.stderr

    def write_messages():
        print(messages[0])
        print(messages[1], file=sys.stderr)
        # Match the managed server's fallback stderr logging without involving
        # pytest's root logger handlers, which intentionally capture messages.
        logger = logging.Logger("autofa-test-background")
        logger.warning(messages[2])

    async def exercise():
        try:
            await asyncio.to_thread(write_messages)
            await asyncio.wait_for(delivered.wait(), timeout=3)
        except Exception as exc:
            app.exit(exception=exc)
        else:
            app.exit(exception=ValueError("menu failed") if exit_with_error else None)

    with create_pipe_input() as pipe:
        app = Application(
            layout=Layout(Window(FormattedTextControl("Menu remains visible"))),
            input=pipe,
            output=output,
        )
        app.pre_run_callables.append(lambda: app.create_background_task(exercise()))

        async def call_from_loop():
            run_app(app)

        def run():
            if inside_running_loop:
                asyncio.run(call_from_loop())
            else:
                run_app(app)

        if exit_with_error:
            with pytest.raises(ValueError, match="menu failed"):
                run()
        else:
            run()

    assert (sys.stdout, sys.stderr) == original_streams
    assert all(message in "".join(text for text, _ in writes) for message in messages)
    assert all(renderer_suspended for _, renderer_suspended in writes)
