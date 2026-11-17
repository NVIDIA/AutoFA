# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI failures, stops and rejected commands have explicit outcomes and keep their cause."""

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from autofa import cli
from autofa.config import Config
from autofa.data.preparation import PreparedData
from autofa.data.remote_api import DataApiError
from autofa.run_config import AutoFAExecution
from autofa.session.supervision import SupervisorInterrupted
from autofa.ui import errors
from autofa.ui.menus import MenuItem


@pytest.mark.parametrize("failure", [SystemExit, RuntimeError])
def test_home_waits_for_acknowledgement_and_logs_cause_before_redraw(
    tmp_path, monkeypatch, failure,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "load_config", Config)
    monkeypatch.setattr(errors.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(errors.sys.stdout, "isatty", lambda: True)
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=200))
    events = []

    def choose(*args, **kwargs):
        if not events:
            events.append("first menu")
            return MenuItem("New analysis", "1")
        assert events == ["first menu", "acknowledged"]
        header = "".join(text for _, text in kwargs["header"])
        assert "Full source retrieval failed [details]" in header
        assert "cli-errors.log" in header
        events.append("next menu")
        return MenuItem("Quit", "q")

    def fail(args):
        try:
            raise DataApiError("Wrong task [expected]")
        except DataApiError as cause:
            raise failure("Full source retrieval failed [details]") from cause

    def acknowledge(prompt):
        assert "Press Enter" in prompt
        assert "Full source retrieval failed [details]" in output.getvalue()
        assert str(tmp_path / "cli-errors.log") in output.getvalue().replace("\n", "")
        log = (tmp_path / "cli-errors.log").read_text()
        assert f"{DataApiError.__name__}: Wrong task [expected]" in log
        assert f"{failure.__name__}: Full source retrieval failed [details]" in log
        events.append("acknowledged")

    monkeypatch.setattr(cli, "select_option", choose)
    monkeypatch.setattr(cli, "_new", fail)
    monkeypatch.setattr("builtins.input", acknowledge)
    cli._home(SimpleNamespace())
    assert events == ["first menu", "acknowledged", "next menu"]


@pytest.mark.parametrize("code", [None, 0])
def test_successful_exit_does_not_report_an_error(tmp_path, monkeypatch, code):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "load_config", Config)
    choices = iter([MenuItem("New", "1"), MenuItem("Quit", "q")])
    monkeypatch.setattr(cli, "select_option", lambda *a, **kw: next(choices))

    def cancel(args):
        raise SystemExit(code)

    monkeypatch.setattr(cli, "_new", cancel)
    monkeypatch.setattr(cli, "show_action_error", lambda *a, **kw: pytest.fail("Not an error"))
    cli._home(SimpleNamespace())
    assert not (tmp_path / "cli-errors.log").exists()


def test_classify_menu_propagates_a_failed_cli_status(monkeypatch):
    monkeypatch.setattr("autofa.taxonomy.cli.classify_main", lambda argv: 1)
    with pytest.raises(SystemExit) as caught:
        cli._classify()
    assert caught.value.code == 1


def test_noninteractive_errors_append_to_log_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(errors.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("Must not wait without a TTY"))
    console = Console(file=StringIO())
    errors.show_action_error(ValueError("First failure"), console=console)
    errors.show_action_error(RuntimeError("Second failure"), console=console)
    log = (tmp_path / "cli-errors.log").read_text()
    assert "ValueError: First failure" in log
    assert "RuntimeError: Second failure" in log


def test_unwritable_log_still_shows_exception_chain(monkeypatch):
    monkeypatch.setattr(errors.sys.stdin, "isatty", lambda: False)

    def unavailable_home():
        raise OSError("Read-only filesystem")

    monkeypatch.setattr(errors, "autofa_home", unavailable_home)
    output = StringIO()
    try:
        raise ValueError("Original failure")
    except ValueError as cause:
        error = RuntimeError("Wrapped failure")
        error.__cause__ = cause
    errors.show_action_error(error, console=Console(file=output, width=200))
    assert "Could not save error details: Read-only filesystem" in output.getvalue()
    assert "ValueError: Original failure" in output.getvalue()
    assert "RuntimeError: Wrapped failure" in output.getvalue()


def test_exit_notice_cannot_hide_an_agent_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    output = Console(file=StringIO())

    def fail_then_close():
        errors.show_action_error(RuntimeError("provider refused model"), console=output, wait=False)
        errors.show_action_notice("Session closed", console=output)

    assert cli._execute_action("Discovery", fail_then_close, wait=False) == "failed"
    assert "provider refused model" in "".join(t for _, t in errors.notice_fragments())


@pytest.mark.parametrize("reason", ["cancel", "interrupt"])
def test_cancel_and_interrupt_have_explicit_outcomes(tmp_path, monkeypatch, reason):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))

    def stop():
        if reason == "interrupt":
            raise KeyboardInterrupt
        errors.cancel_action("Discovery cancelled in dataset selection.")

    state = cli._execute_action("New analysis", stop)
    assert state == ("cancelled" if reason == "cancel" else "interrupted")
    assert errors.notice_fragments()
    assert not (tmp_path / "cli-errors.log").exists()


def test_supervisor_stop_has_release_guidance_not_ctrl_c(tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(cli, 'show_action_notice', lambda text, **kw: messages.append(text))

    def stop():
        raise SupervisorInterrupted(tmp_path / 'run-id', 'pause')

    assert cli._execute_action('resume', stop, wait=False) == 'stopped_by_supervisor'
    assert any('pause' in message and 'supervise request --run run-id release' in message
               for message in messages)
    assert not any('Ctrl+C' in message for message in messages)


def test_once_setup_failure_logs_details_and_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "ensure_default_config", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(cli, "load_config", Config)
    monkeypatch.setattr(cli, "_home", lambda *a, **kw: pytest.fail("Must not return to home"))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("Must not wait in --once"))

    def fail(args):
        raise DataApiError("Original trace API failed")

    monkeypatch.setattr(cli, "_new", fail)
    with pytest.raises(SystemExit) as caught:
        cli.main(["new", "--once"])
    assert caught.value.code == 1
    assert "Original trace API failed" in (tmp_path / "cli-errors.log").read_text()


def test_data_selection_flags_are_rejected_for_unrelated_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "ensure_default_config", lambda: tmp_path / "config.toml")
    with pytest.raises(SystemExit) as caught:
        cli.main(["settings", "--reward", "<=0"])
    assert caught.value.code == 2
    assert "data selection flags apply only" in capsys.readouterr().err


@pytest.mark.parametrize('argv', [
    ['research'], ['continue', '--patience', '6'], ['publish', '--snapshot'],
])
def test_retired_overnight_commands_rejected_before_setup(monkeypatch, argv):
    monkeypatch.setattr(cli, 'ensure_default_config', lambda **kw: pytest.fail('setup'))
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_retired_programmatic_execution_mode_is_rejected():
    with pytest.raises(ValueError, match='session or batch'):
        AutoFAExecution(mode='research')


@pytest.mark.parametrize("once_in_terminal", [False, True])
def test_unattended_new_uses_cli_depth_without_picker(monkeypatch, once_in_terminal):
    cfg = Config(analysis_outputs=["agent"], max_rounds=2, top_x=3, min_failure_modes=4)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    # Either a non-interactive stream or --once in a terminal skips the depth picker.
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda **kwargs: once_in_terminal)
    monkeypatch.setattr("autofa.session.guidance.snapshot_guidance", lambda _: {})
    selection = PreparedData(parquet_paths=[], model_id="model", benchmarks=["benchmark"])
    monkeypatch.setattr(cli, "_run_cli_dataset_wizard", lambda *args, **kwargs: selection)
    monkeypatch.setattr("autofa.run_spec.preflight_autofa_config", lambda config: None)
    monkeypatch.setattr(
        "autofa.wizard.depth.select_discovery_depth",
        lambda **kwargs: pytest.fail("Unattended discovery must not open the depth picker"),
    )
    monkeypatch.setattr(
        "autofa.estimates.prepare_estimator",
        lambda *args: pytest.fail("Unattended discovery does not need a depth estimate"),
    )
    created = {}
    run = SimpleNamespace(save=lambda: None)

    def create_run(*args, **kwargs):
        created.update(kwargs)
        return run

    monkeypatch.setattr(cli, "_create_analysis_run", create_run)
    started = []
    monkeypatch.setattr(
        "autofa.session.runner.run_session",
        lambda selected, **kwargs: started.append((selected, kwargs)),
    )
    cli._new(SimpleNamespace(
        analysis_outputs=["agent"], analysis_focus=None, optimization=None, min_failure_modes=0,
        top_x=1, keep_going=False, hours=None, eval_limit=17, no_auto=False,
        once=once_in_terminal,
    ))

    assert cfg.max_rounds == 2
    assert cfg.top_x == 1
    assert cfg.min_failure_modes == 0
    assert cfg.default_eval_limit == 17
    assert created == {"show_guidance": True}
    assert started == [(run, {
        "cfg": cfg, "auto_discover": True, "continue_mode": False, "exit_after_discovery": True,
    })]


@pytest.mark.parametrize("picker", ["menu", "rollouts", "browser", "multi", "paths"])
def test_unexpected_picker_exit_is_an_error_not_cancellation(monkeypatch, picker, tmp_path):
    from autofa.ui import menus, rollout_table

    monkeypatch.setattr(menus, "run_app", lambda app: None)
    monkeypatch.setattr(rollout_table, "run_app", lambda app: None)
    choices = [MenuItem("a", "a"), MenuItem("b", "b")]
    with pytest.raises(RuntimeError, match="closed"):
        if picker == "menu":
            menus.select_option("test", choices, clear_screen=False)
        elif picker == "rollouts":
            rollout_table.select_rollout_runs([{"run_id": "run"}])
        elif picker == "browser":
            menus.browse_filesystem(tmp_path, clear_screen=False)
        elif picker == "multi":
            menus.select_items_multi(choices, clear_screen=False)
        else:
            (tmp_path / "a").touch()
            (tmp_path / "b").touch()
            menus.select_paths_multi([tmp_path / "a", tmp_path / "b"], clear_screen=False)
