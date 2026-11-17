# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-launch selection requires authenticated harnesses and saves a coherent profile."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest
import tomlkit

from autofa import cli, config, defaults
from autofa.harness import claude_cli, settings

from support.config import save_config


def _executable(path: Path, *, logged_in: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '#!/bin/sh\ncase "$*" in "login status"|"auth status") ;; *) exit 99 ;; esac\n'
        f'exit {0 if logged_in else 1}\n'
    )
    path.chmod(0o755)
    return path


@pytest.fixture
def installation(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    monkeypatch.setenv("PATH", str(binaries))
    monkeypatch.setenv("AUTOFA_HOME", str(home / ".autofa"))
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "config/autofa.toml"))
    for name in ("AUTOFA_CODEX_CLI", "CODEX_CLI", "AUTOFA_CLAUDE_CLI", "CLAUDE_CLI",
                 "CLAUDE_CODE_EXECPATH", *claude_cli.API_KEY_ENV_NAMES):
        # Track originally absent variables too, including values later loaded by dotenv.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(claude_cli, "DEFAULT_CLI_CANDIDATES", (
        "claude", str(home / ".local/bin/claude"),
    ))
    # AutoFA normally installs this dependency even without a configured harness.
    sdk_dir = tmp_path / "sdk"
    _executable(sdk_dir / "_bundled/claude", logged_in=False)
    monkeypatch.setattr(claude_cli, "_sdk_package_dir", lambda: sdk_dir)
    monkeypatch.chdir(tmp_path)
    return binaries


@pytest.mark.parametrize("installed,expected", [
    (("claude",), "claude"), (("claude", "codex"), "codex"),
])
def test_first_launch_persists_harness_models_and_reasoning(installation, installed, expected):
    for name in installed:
        _executable(installation / name)
    cfg = config.load_config()
    saved = tomllib.loads(config.config_path().read_text())
    assert cfg.harness == saved["harness"] == expected
    assert cfg.behavior_judge_harness == "auto"
    for field, value in defaults.default_value("harness_defaults")[expected].items():
        assert getattr(cfg, field) == saved[field] == value
    # Installing/removing a harness later must not change the saved selection.
    before = config.config_path().read_text()
    for path in installation.iterdir():
        path.unlink()
    _executable(installation / ("codex" if expected == "claude" else "claude"))
    assert config.load_config() == cfg
    assert config.config_path().read_text() == before


def test_neither_installed_reports_setup_without_saving_and_allows_retry(installation):
    with pytest.raises(settings.HarnessNotFoundError) as caught:
        config.ensure_default_config()
    for instruction in ("codex login", "AUTOFA_CODEX_CLI", "AUTOFA_CLAUDE_CLI"):
        assert instruction in str(caught.value)
    assert not config.config_path().exists()
    _executable(installation / "claude")
    assert config.load_config().harness == "claude"


@pytest.mark.parametrize("harness,location", [("codex", "path"), ("claude", "home")])
def test_non_executable_files_do_not_count_as_installed(installation, harness, location):
    directory = installation if location == "path" else Path.home() / ".local/bin"
    binary = _executable(directory / harness)
    binary.chmod(0o644)
    with pytest.raises(settings.HarnessNotFoundError):
        config.ensure_default_config()
    assert not config.config_path().exists()


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_installations_in_user_bin_are_detected_off_path(installation, harness):
    _executable(Path.home() / ".local/bin" / harness)
    assert config.load_config().harness == harness


@pytest.mark.parametrize("variable,harness", [
    ("AUTOFA_CODEX_CLI", "codex"), ("CODEX_CLI", "codex"),
    ("AUTOFA_CLAUDE_CLI", "claude"), ("CLAUDE_CLI", "claude"),
    ("CLAUDE_CODE_EXECPATH", "claude"),
])
def test_explicit_binary_locations_are_detected(installation, monkeypatch, variable, harness):
    binary = _executable(installation.parent / "custom/agent")
    monkeypatch.setenv(variable, str(binary))
    assert config.load_config().harness == harness


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_packaged_cli_and_profile_overrides_are_used(installation, monkeypatch, harness):
    document = tomlkit.parse(defaults.defaults_text())
    document[f"{harness}_cli"] = str(_executable(installation.parent / "custom/agent"))
    document["harness_defaults"][harness]["model"] = "custom-discovery"
    document["harness_defaults"][harness]["behavior_judge_model"] = "custom-judge"
    text = tomlkit.dumps(document)
    monkeypatch.setattr(defaults, "defaults_text", lambda: text)
    cfg = config.load_config()
    assert cfg.harness == harness
    assert cfg.model == "custom-discovery"
    assert cfg.behavior_judge_model == "custom-judge"


def test_explicit_api_key_keeps_bundled_sdk_setup_working(installation, monkeypatch):
    binary = claude_cli._sdk_package_dir() / "_bundled/claude"
    binary.write_text('#!/bin/sh\n[ -n "$ANTHROPIC_API_KEY" ]\n')
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert config.load_config().harness == "claude"


@pytest.mark.parametrize("codex_login,claude_login,expected", [
    (False, True, "claude"), (False, False, None),
])
def test_selection_requires_login_not_just_installation(
    installation, codex_login, claude_login, expected,
):
    _executable(installation / "codex", logged_in=codex_login)
    _executable(installation / "claude", logged_in=claude_login)
    if expected:
        assert config.load_config().harness == expected
    else:
        with pytest.raises(settings.HarnessNotFoundError, match="not signed in"):
            config.load_config()
        assert not config.config_path().exists()
        _executable(installation / "claude")
        assert config.load_config().harness == "claude"


@pytest.mark.parametrize("problem,reason", [
    (subprocess.TimeoutExpired("auth status", 5), "timed out"),
    (OSError("private diagnostic"), "could not run"),
    (subprocess.CompletedProcess([], 2), "could not verify"),
])
def test_failed_auth_checks_are_bounded_and_do_not_expose_output(
    installation, monkeypatch, problem, reason,
):
    _executable(installation / "codex")
    calls = []

    def status(command, **kwargs):
        calls.append(command)
        assert kwargs == {
            "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL, "timeout": 5, "check": False,
        }
        if isinstance(problem, Exception):
            raise problem
        return problem

    monkeypatch.setattr(settings.subprocess, "run", status)
    with pytest.raises(settings.HarnessNotFoundError) as caught:
        config.load_config()
    assert reason in str(caught.value)
    assert "private diagnostic" not in str(caught.value)
    assert [command[1:] for command in calls] == [["login", "status"], ["auth", "status"]]
    assert not config.config_path().exists()


def test_api_key_without_executable_sdk_is_not_a_harness(installation, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(claude_cli, "_sdk_package_dir", lambda: None)
    with pytest.raises(settings.HarnessNotFoundError):
        config.ensure_default_config()


def test_existing_config_is_loaded_even_without_harnesses(installation, monkeypatch):
    cfg = config.Config(harness="claude", model="custom-discovery", effort="high",
                        behavior_judge_model="custom-judge",
                        library_root=str(installation.parent / "library"))
    save_config(cfg)
    monkeypatch.setattr(settings, "detect_default_harness", lambda **kw: pytest.fail("Re-detected"))
    before = config.config_path().read_text()
    assert config.load_config() == cfg
    assert config.config_path().read_text() == before


def test_legacy_config_is_imported_without_detection(installation, monkeypatch):
    legacy = Path.home() / ".autofa/config.toml"
    legacy.parent.mkdir()
    original = '# Legacy selection\nmodel = "custom-claude"\neffort = "high"\n'
    legacy.write_text(original)
    target = installation.parent / "project/autofa.toml"
    monkeypatch.delenv("AUTOFA_CONFIG")
    monkeypatch.delenv("AUTOFA_HOME")
    monkeypatch.setattr(config, "config_path", lambda **kwargs: target)
    monkeypatch.setattr(settings, "detect_default_harness", lambda **kw: pytest.fail("Re-detected"))
    cfg = config.load_config()
    assert cfg.harness == "claude"
    assert cfg.model == "custom-claude"
    assert cfg.effort == "high"
    assert legacy.read_text() == original


def test_cli_missing_harness_exits_cleanly_before_starting_services(
    installation, monkeypatch, capsys,
):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda: pytest.fail("Startup continued"))
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == 1
    stderr = capsys.readouterr().err
    assert "No authenticated harness found" in stderr
    assert "Traceback" not in stderr
    assert not config.config_path().exists()


def test_help_works_without_harness_or_config(installation, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "usage: autofa" in capsys.readouterr().out
    assert not config.config_path().exists()


def test_cli_loads_dotenv_before_detection(installation, monkeypatch):
    binary = _executable(installation.parent / "custom/agent")
    dotenv = installation.parent / ".env"
    dotenv.write_text(f'AUTOFA_CLAUDE_CLI="{binary}"\n')
    monkeypatch.setattr(cli, "load_dotenv", lambda: config.load_dotenv(dotenv))

    def stop_after_setup():
        saved = tomllib.loads(config.config_path().read_text())
        assert saved["harness"] == "claude"
        raise SystemExit(0)

    monkeypatch.setattr(cli, "load_config", stop_after_setup)
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == 0
