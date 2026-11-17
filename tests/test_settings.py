# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analysis, endpoint and harness settings: validation, menu edits, per-run snapshots and resume."""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from contextlib import contextmanager
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace

import pytest
import tomlkit
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from autofa import cli, config
from autofa.behavior_policy import BEHAVIOR_STRICTNESSES
from autofa.config import Config, config_path, ensure_default_config, load_config
from autofa.core.library import FunctionLibrary
from autofa.harness.model_catalog import HarnessModel, ModelCatalog
from autofa.harness.settings import behavior_judge_settings
from autofa.llm.endpoints import endpoint_for_config
from autofa.mcp import runtime
from autofa.objectives import config_for_run, config_from_run_dir
from autofa.ui import llm_settings, settings
from autofa.ui.llm_settings import _AgentForm
from autofa.ui.menus import MENU_BACK, MenuItem
from autofa.ui.pt_app import run_app
from autofa.wizard import depth

from support.config import save_config

LEFT, RIGHT, SAVE, CANCEL, CLEAR = "\x1b[D", "\x1b[C", "\x13", "\x03", "\x01\x0b"
pytestmark = pytest.mark.usefixtures("installed_codex")


def _choices(monkeypatch, values):
    remaining = iter(values)

    def choose(title, options, **kwargs):
        value = next(remaining)
        if value == "escape":
            return MENU_BACK
        return next(option for option in options if option.value == value)

    monkeypatch.setattr(settings, "select_option", choose)


def test_interactive_discovery_gets_a_per_run_target_picker(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    ensure_default_config()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    _choices(monkeypatch, ["focus", "harness", "done"])
    assert cli._discovery_config(SimpleNamespace()).analysis_focus == "harness"
    assert load_config().analysis_focus == "both"


def test_failed_autosave_does_not_show_unsaved_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    ensure_default_config()

    def fail_save(cfg):
        raise OSError("Disk full")

    monkeypatch.setattr(settings, "save_analysis_settings", fail_save)
    _choices(monkeypatch, ["focus", "harness", "done"])
    result = settings.configure_analysis(Config(), persistent=True)
    assert result.analysis_focus == "both"
    assert load_config().analysis_focus == "both"


@pytest.mark.parametrize("persistent", [False, True])
def test_analysis_toggles_preserve_per_run_and_saved_defaults(tmp_path, monkeypatch, persistent):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    save_config(Config(analysis_outputs=["efficiency", "agent"]))
    path = ensure_default_config()
    path.write_text(path.read_text() + '\n# Keep custom settings\ncustom_key = "value"\n')
    before = path.read_text()
    original = load_config()
    # Leaving the submenu with Escape still keeps the toggles made inside it.
    _choices(monkeypatch, ["analyses", "agent", "escape", "done"])

    cfg = settings.configure_analysis(original, persistent=persistent)

    assert cfg.analysis_outputs == ["efficiency"]
    assert set(original.analysis_outputs) == {"efficiency", "agent"}
    if persistent:
        assert load_config().analysis_outputs == ["efficiency"]
        assert '# Keep custom settings\ncustom_key = "value"' in path.read_text()
    else:
        assert set(load_config().analysis_outputs) == {"efficiency", "agent"}
        assert path.read_text() == before


def test_analysis_toggles_keep_at_least_one_enabled(monkeypatch):
    errors = []
    monkeypatch.setattr(settings, "show_action_error", lambda error, **kwargs: errors.append(error))
    _choices(monkeypatch, ["analyses", "efficiency", "agent", "done", "done"])

    cfg = settings.configure_analysis(Config(analysis_outputs=["efficiency", "agent"]))

    assert cfg.analysis_outputs == ["agent"]
    assert len(errors) == 1
    assert "Keep at least one analysis enabled" in str(errors[0])


@pytest.mark.parametrize("outputs", [[], ["unknown"]])
def test_invalid_analysis_defaults_are_rejected(tmp_path, monkeypatch, outputs):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    path = ensure_default_config()
    document = tomlkit.parse(path.read_text())
    document["analysis_outputs"] = outputs
    path.write_text(tomlkit.dumps(document))

    with pytest.raises(ValueError, match="analysis_outputs"):
        load_config()


# Stall, plateau and Judge Policy settings are saved per run and edited without side effects.

SETTINGS = {
    # name: (default, settings-menu key, selectable values)
    "discovery_terminate_if_stalled": (True, "stalled", (False, True)),
    "discovery_wait_for_plateau": (False, "plateau", (False, True)),
    "discovery_stall_rounds": (3, "stall-rounds", (2, 5)),
    "behavior_strictness": ("causal", "strictness", BEHAVIOR_STRICTNESSES),
}


def _other(name, value):
    return next(choice for choice in SETTINGS[name][2] if choice != value)


def test_settings_migration_adds_defaults_and_preserves_explicit_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    header = ("# Preserve custom settings\nmax_rounds = 7\ndiscovery_plateau_patience = 19\n"
              "discovery_wait_for_plateau = true\nmin_failure_modes = 0\n")
    path.write_text(header)
    expected = {name: spec[0] for name, spec in SETTINGS.items()}
    expected["discovery_wait_for_plateau"] = True
    loaded = config.load_config()
    assert {name: getattr(loaded, name) for name in SETTINGS} == expected
    assert (loaded.max_rounds, loaded.min_failure_modes) == (7, 0)
    # Plateau patience stays separate from stall rounds.
    assert loaded.discovery_plateau_patience == 19
    assert path.read_text().startswith(header)
    stored = tomllib.loads(path.read_text())
    assert {name: stored[name] for name in SETTINGS} == expected
    assert (stored["discovery_plateau_patience"], stored["min_failure_modes"]) == (19, 0)
    before = path.read_bytes()
    config.load_config()
    assert path.read_bytes() == before


# Each setting is copied by its own line in RunConfig.create, config_for_run and
# config_from_run_dir; non-default values catch a dropped copy that defaults would hide.
@pytest.mark.parametrize("name,value", [
    ("discovery_terminate_if_stalled", False), ("discovery_wait_for_plateau", True),
    ("discovery_stall_rounds", 5), ("behavior_strictness", "contextual"),
])
def test_run_choice_survives_settings_change_and_resume_while_continue_uses_new_choice(
    tmp_path, monkeypatch, name, value,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    changed = _other(name, value)
    save_config(config.Config(**{name: value}))
    run = config.RunConfig.create([], "model", ["bench"], config.load_config())
    assert json.loads((run.run_dir() / "config.json").read_text())[name] == value
    current = config.Config(**{name: changed})
    save_config(current)
    saved = config.RunConfig.load(run.run_id)
    assert getattr(saved, name) == value
    assert getattr(config_for_run(config.load_config(), saved), name) == value
    assert getattr(config_from_run_dir(run.run_dir()), name) == value
    monkeypatch.setattr(runtime, "set_runtime", lambda rt: None)
    assert getattr(runtime.load_runtime_from_run(saved, current).config, name) == value
    assert getattr(current, name) == changed

    seed = run.run_dir() / "library.json"
    FunctionLibrary(model_id="model", benchmark="bench").save(seed)
    before = seed.read_bytes()
    continued = config.RunConfig.create(
        [], "model", ["bench"], current, seed_library_path=str(seed),
        seed_source_run_id=run.run_id,
    )
    assert getattr(config.RunConfig.load(continued.run_id), name) == changed
    assert getattr(config.RunConfig.load(run.run_id), name) == value
    assert seed.read_bytes() == before


def test_legacy_run_uses_defaults_without_rewriting_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    defaults = {name: spec[0] for name, spec in SETTINGS.items()}
    cfg = config.Config(**{name: _other(name, default) for name, default in defaults.items()},
                        min_failure_modes=0)
    run = config.RunConfig.create([], "model", [], cfg)
    path = run.run_dir() / "config.json"
    data = json.loads(path.read_text())
    for name in [*SETTINGS, "min_failure_modes"]:
        del data[name]
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    saved = config.RunConfig.load(run.run_id)
    assert saved.min_failure_modes == 1
    for resolved in (saved, config_for_run(cfg, saved), config_from_run_dir(run.run_dir())):
        assert {name: getattr(resolved, name) for name in SETTINGS} == defaults
    assert path.read_bytes() == before
    saved.save()
    assert {name: json.loads(path.read_text())[name] for name in SETTINGS} == defaults


@pytest.mark.parametrize("name,value", [
    *((name, "typo") for name in SETTINGS),
    ("discovery_stall_rounds", 0),
    ("min_failure_modes", -1),
    ("min_failure_modes", True),  # Booleans are rejected rather than coerced.
])
def test_invalid_values_cannot_enter_configs_or_runs(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    with pytest.raises(ValueError, match=name):
        config.Config(**{name: value})
    with pytest.raises(ValueError, match=name):
        config.RunConfig("run", [], "model", **{name: value})
    (tmp_path / "config.toml").write_text(f"{name} = {json.dumps(value)}\n")
    with pytest.raises(ValueError, match=name):
        config.load_config()
    cfg = config.Config()
    setattr(cfg, name, value)
    with pytest.raises(ValueError, match=name):
        save_config(cfg)
    run = config.RunConfig("run", [], "model")
    setattr(run, name, value)
    with pytest.raises(ValueError, match=name):
        run.save()


@pytest.mark.parametrize("key,name,previous,value", [
    ("plateau", "discovery_wait_for_plateau", True, False),  # A falsy choice is still saved.
    ("stalled", "discovery_terminate_if_stalled", False, True),
    # A saved custom value outside the standard choices still opens the picker.
    ("stall-rounds", "discovery_stall_rounds", 7, 5),
    ("strictness", "behavior_strictness", "causal", "behavioral"),
    ("minimum", "min_failure_modes", 1, 8),
])
def test_menu_autosaves_choice_and_preserves_other_settings(
    tmp_path, monkeypatch, key, name, previous, value,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    save_config(config.Config(discovery_plateau_patience=17, **{name: previous}))
    path = config.config_path()
    path.write_text(path.read_text() + '\n# Extension\ncustom_key = "keep me"\n')
    _choices(monkeypatch, [key, value, "done"])
    settings.run_settings()
    loaded = config.load_config()
    assert getattr(loaded, name) == value
    assert loaded.discovery_plateau_patience == 17
    assert '# Extension\ncustom_key = "keep me"' in path.read_text()


def test_per_analysis_selection_is_isolated_and_escape_discards_it(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    before = config.ensure_default_config().read_bytes()
    original = config.Config()
    _choices(monkeypatch, ["stall-rounds", 5, "done"])
    assert settings.configure_analysis(original).discovery_stall_rounds == 5
    _choices(monkeypatch, ["stall-rounds", 5, "escape"])
    assert settings.configure_analysis(original) is None
    assert original.discovery_stall_rounds == 3
    assert config.config_path().read_bytes() == before


# The discovery minimum is explicit, validated, and retained on resume.


def test_minimum_survives_config_and_run_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    # Zero disables the minimum, so it must not fall back to the default.
    save_config(config.Config(min_failure_modes=0))
    cfg = config.load_config()
    assert cfg.min_failure_modes == 0
    run = config.RunConfig.create([], "model", [], cfg, max_rounds=3)
    assert config.RunConfig.load(run.run_id).min_failure_modes == 0
    save_config(config.Config(min_failure_modes=1))
    assert config.RunConfig.load(run.run_id).min_failure_modes == 0
    override = config.RunConfig.create([], "model", [], config.Config(), min_failure_modes=0)
    assert config.RunConfig.load(override.run_id).min_failure_modes == 0


@pytest.fixture
def depth_ui(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(depth, "console", Console(file=output, width=200))
    monkeypatch.setattr(depth, "prepare_step", lambda: None)
    return output


def test_depth_wizard_uses_saved_minimum_without_prompt(monkeypatch, depth_ui):
    calls = []

    def pick(title, items, **kwargs):
        calls.append(title)
        return next(item for item in items if item.value == "standard")

    monkeypatch.setattr(depth, "select_option", pick)
    selected = depth.select_discovery_depth(default_min_failure_modes=0)
    assert (selected.max_rounds, selected.min_failure_modes) == (3, 0)
    assert len(calls) == 1


def test_depth_wizard_honors_cli_override_without_second_prompt(monkeypatch, depth_ui):
    calls = []

    def pick(*args, **kwargs):
        calls.append(True)
        return MenuItem("Standard", "standard")

    monkeypatch.setattr(depth, "select_option", pick)
    selected = depth.select_discovery_depth(min_failure_modes_override=0)
    assert selected.min_failure_modes == 0
    assert len(calls) == 1


def test_cli_rejects_invalid_minimum_before_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "ensure_default_config", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(cli, "_new", lambda _: pytest.fail("Must not start discovery"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["new", "--min-failure-modes", "-1", "--once"])
    assert exc.value.code == 2


def test_cli_minimum_override_reaches_discovery_config(monkeypatch):
    monkeypatch.setattr(cli, "load_config", config.Config)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    # Zero disables the minimum, so a falsy override must still replace the default.
    cfg = cli._discovery_config(SimpleNamespace(min_failure_modes=0))
    assert cfg.min_failure_modes == 0


@pytest.fixture
def resumable(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(
        "autofa.util.memory.apply_memory_tuning",
        lambda *a, **kw: SimpleNamespace(summary=lambda: "test"),
    )


@pytest.mark.parametrize("override,expected", [(None, 5), (0, 0)])
def test_resume_retains_or_explicitly_updates_saved_minimum(
    resumable, monkeypatch, override, expected,
):
    from autofa.session.recovery import exclusive_run

    run = config.RunConfig("resume", [], "model", min_failure_modes=5,
                           analysis_outputs=["agent"])
    run.save()
    monkeypatch.setattr(cli, "load_config", lambda: config.Config(min_failure_modes=3))
    seen = []

    @exclusive_run
    def locked_runner(selected, *, min_failure_modes=None, **kwargs):
        # CLI must delegate persistence to the locked runner, not pre-edit it.
        assert config.RunConfig.load(selected.run_id).min_failure_modes == 5
        if min_failure_modes is not None:
            selected.min_failure_modes = config.validate_min_failure_modes(min_failure_modes)
            selected.save()
        seen.append(selected)

    monkeypatch.setattr("autofa.session.runner.run_session", locked_runner)
    cli._resume(SimpleNamespace(run_id=run.run_id, min_failure_modes=override))
    assert seen[0].min_failure_modes == expected
    assert config.RunConfig.load(run.run_id).min_failure_modes == expected


def test_resume_passes_top_x_to_locked_runner(resumable, monkeypatch):
    from autofa.session.recovery import exclusive_run

    run = config.RunConfig("resume-top-x", [], "model", top_x=5, analysis_outputs=["agent"])
    run.save()
    monkeypatch.setattr(cli, "load_config", config.Config)
    seen = []

    @exclusive_run
    def locked_runner(selected, *, top_x=None, **kwargs):
        assert config.RunConfig.load(selected.run_id).top_x == 5
        seen.append((selected.run_id, top_x))

    monkeypatch.setattr("autofa.session.runner.run_session", locked_runner)
    cli._resume(SimpleNamespace(run_id=run.run_id, min_failure_modes=None, top_x=1))
    assert seen == [(run.run_id, 1)]
    assert config.RunConfig.load(run.run_id).top_x == 5


def test_real_session_runner_preserves_top_x_when_preflight_fails(tmp_path, monkeypatch):
    from autofa.session import runner

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = config.RunConfig("runner-top-x", [], "model", top_x=5)
    path = run.save()
    before = path.read_bytes()

    def failed_preflight(*args, **kwargs):
        raise RuntimeError("preflight failed")

    monkeypatch.setattr("autofa.preflight.runner.run_preflight", failed_preflight)
    with pytest.raises(RuntimeError, match="preflight failed"):
        runner.run_session.__wrapped__(run, cfg=config.Config(), top_x=1)
    assert config.RunConfig.load(run.run_id).top_x == 5
    assert path.read_bytes() == before


def test_resume_cannot_change_minimum_of_an_active_run(resumable, monkeypatch):
    from autofa.session.recovery import exclusive_run, run_lock

    run = config.RunConfig("already-running", [], "model", min_failure_modes=5,
                           analysis_outputs=["agent"])
    path = run.save()
    before = path.read_bytes()
    monkeypatch.setattr(cli, "load_config", config.Config)

    @exclusive_run
    def locked_runner(*args, **kwargs):
        pytest.fail("An active run must reject Resume before applying overrides")

    monkeypatch.setattr("autofa.session.runner.run_session", locked_runner)
    with run_lock(run.run_dir()), pytest.raises(RuntimeError, match="already active"):
        cli._resume(SimpleNamespace(run_id=run.run_id, min_failure_modes=0))
    assert path.read_bytes() == before


# LLM endpoint and harness forms.


def claude_config(**overrides):
    """Existing Claude installations retain their explicit model and thinking choices."""
    settings = {
        "harness": "claude", "model": "claude-opus-5-5",
        "behavior_judge_model": "claude-sonnet-5-5",
        "behavior_judge_thinking": "adaptive", "behavior_judge_effort": "medium",
    }
    return Config(**{**settings, **overrides})


@pytest.fixture(autouse=True)
def harness_catalog(monkeypatch):
    catalog = ModelCatalog((
        HarnessModel("opus", "Opus from harness", "claude-opus-5-5",
                     supports_effort=True, effort_levels=("low", "medium", "high", "max"),
                     supports_adaptive_thinking=True),
        HarnessModel("sonnet", "Sonnet from harness", "claude-sonnet-5-5",
                     supports_effort=True, effort_levels=("low", "medium", "high"),
                     supports_adaptive_thinking=True),
        HarnessModel("brand-new-model", "New model from harness",
                     supports_effort=True, effort_levels=("low", "high"),
                     supports_adaptive_thinking=True),
        HarnessModel("haiku", "Haiku from harness", "claude-haiku-4-5",
                     supports_effort=False, supports_adaptive_thinking=False),
    ))

    async def load(cli_path=None, *, harness="claude"):
        if harness == "codex":
            return ModelCatalog((HarnessModel(
                "codex-only-model", "Codex model", supports_effort=True,
                effort_levels=("low", "ultra")),))
        return catalog

    monkeypatch.setattr(llm_settings, "load_model_catalog_async", load)
    return catalog


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "autofa.toml"))
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    # A deployment-provided default, so the form opens on "Configured default".
    cfg = claude_config(hub_base_url="https://hub.example/v1",
                        hub_api_key_env="HUB_API_KEY")  # pragma: allowlist secret
    save_config(cfg)
    return cfg


def edit(cfg, keys):
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(keys)
        return settings.configure_detector_endpoint(cfg)


def test_form_without_a_default_endpoint_opens_on_other_provider():
    from autofa.ui.llm_settings import _provider

    hub = "https://hub.example/v1"
    assert _provider("", Config()) == 1
    assert _provider("http://localhost:8000/v1", Config()) == 2
    assert _provider(hub + "/", Config(hub_base_url=hub)) == 0


def test_local_server_setup_needs_only_a_model_and_save(config_file):
    path = config_path()
    path.write_text(path.read_text() + '\n# Keep this comment\ncustom_option = "keep"\n')
    edit(config_file, LEFT + "\t\tlocal-model" + SAVE)
    cfg = load_config()
    for role in ("feature", "judge"):
        endpoint = endpoint_for_config(cfg, role)
        assert endpoint.base_url == "http://localhost:8000/v1"
        assert endpoint.api_key_env == "none"
        assert endpoint.model == "local-model"
    assert '# Keep this comment\ncustom_option = "keep"' in path.read_text()
    assert cfg.model == config_file.model


def test_other_provider_fields_share_one_form(config_file):
    edit(config_file, RIGHT + "\thttps://provider.example/prefix/v1/\tdetector-model\t"
         + CLEAR + "PROVIDER_KEY" + SAVE)
    cfg = load_config()
    for role in ("feature", "judge"):
        endpoint = endpoint_for_config(cfg, role)
        assert endpoint.base_url == "https://provider.example/prefix/v1"
        assert endpoint.api_key_env == "PROVIDER_KEY"
        assert endpoint.model == "detector-model"


def test_api_preference_round_trips_and_applies_to_both_api_roles(config_file):
    assert config_file.feature_api == "auto"
    cfg = edit(config_file, "\t" * 4 + RIGHT * 2 + SAVE)
    assert cfg == load_config()
    assert cfg.feature_api == "decisions"
    assert endpoint_for_config(cfg).api == endpoint_for_config(cfg, "judge").api == "decisions"
    assert cfg.feature_model == config_file.feature_model
    assert cfg.feature_request_options == config_file.feature_request_options
    # The detector API never changes the harness roles.
    assert cfg.model == config_file.model
    assert cfg.behavior_judge_model == config_file.behavior_judge_model


def test_changing_main_endpoint_preserves_existing_classifier_override(config_file):
    config_file.judge_base_url = "https://judge.example/v1"
    config_file.judge_api_key_env = "JUDGE_KEY"
    config_file.judge_model = "independent-model"
    config_file.judge_request_options = {"max_tokens": 4096}
    save_config(config_file)
    edit(config_file, LEFT + "\t\tlocal-model" + SAVE)
    cfg = load_config()
    assert endpoint_for_config(cfg, "judge") == endpoint_for_config(config_file, "judge")


def test_restore_configured_endpoint_without_reentering_model_or_key(config_file):
    config_file.feature_base_url = "http://localhost:8000/v1"
    config_file.feature_api_key_env = "none"
    config_file.feature_model = config_file.judge_model = "local-model"
    save_config(config_file)
    edit(config_file, RIGHT + SAVE)
    cfg = load_config()
    assert cfg.feature_base_url == config_file.hub_base_url
    assert cfg.feature_api_key_env == config_file.hub_api_key_env
    assert cfg.feature_model == "local-model"
    assert cfg.judge_model == "local-model"


@pytest.mark.parametrize("keys", [
    LEFT + "\t\tlocal-model" + CANCEL,
    RIGHT + "\thttps://provider.example/v1/chat/completions\tmodel" + SAVE + CANCEL,
], ids=["cancelled", "invalid_url"])
def test_invalid_or_cancelled_setup_does_not_save(config_file, keys):
    before = config_path().read_bytes()
    assert edit(config_file, keys) == config_file
    assert config_path().read_bytes() == before


def test_failed_save_keeps_current_endpoint(config_file, monkeypatch):
    def fail(*args):
        raise OSError("Disk full")

    monkeypatch.setattr(settings, "_save_fields", fail)
    assert edit(config_file, LEFT + "\t\tlocal-model" + SAVE + CANCEL) == config_file
    assert load_config().feature_base_url == ""


def test_unchanged_form_preserves_inheritance_and_request_options(config_file):
    config_file.feature_request_options = {"max_tokens": 2048}
    config_file.judge_model = "different-model-on-the-same-server"
    save_config(config_file)
    before = config_path().read_bytes()
    assert edit(config_file, SAVE) == config_file
    assert config_path().read_bytes() == before


def edit_agent(cfg, keys, *, judge=False):
    fields = (("behavior_judge_harness", "behavior_judge_model", "behavior_judge_effort",
               "behavior_judge_thinking", "behavior_judge_thinking_budget_tokens") if judge else
              ("harness", "model", "thinking", "effort", "thinking_budget_tokens",
               "behavior_judge_model", "behavior_judge_thinking", "behavior_judge_effort"))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(("\t" if judge else "") + keys)
        return llm_settings.edit_agent(cfg, judge=judge,
                                       save=lambda c: settings._save_fields(c, fields))


@contextmanager
def codex_form(cfg, catalog, *, judge=False):
    """Open the agent form and switch its discovery or judge harness to Codex."""
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        form = _AgentForm(cfg, save_config, judge=judge, catalog=catalog)
        form.start_loading = lambda: None  # Load catalogs explicitly, without a running app.
        form.harness.move(-1 if judge else 1)  # Judge: "Same as AutoFA" -> Codex.
        asyncio.run(form.load_models("codex"))
        yield form, pipe


def test_harness_model_saves_without_changing_judge_or_detector(config_file):
    path = config_path()
    path.write_text(path.read_text() + '\n# Keep this comment\ncustom_option = "keep"\n')
    edit_agent(config_file, "\t" + RIGHT + SAVE)
    cfg = load_config()
    assert cfg.harness == "claude"
    assert cfg.model == "sonnet"
    assert cfg.behavior_judge_model == config_file.behavior_judge_model
    assert endpoint_for_config(cfg) == endpoint_for_config(config_file)
    assert '# Keep this comment\ncustom_option = "keep"' in path.read_text()


def test_judge_model_and_effort_are_independent_of_discovery_and_classifier(config_file):
    edit_agent(config_file, LEFT + "\t\t" + RIGHT + SAVE, judge=True)
    cfg = load_config()
    assert cfg.behavior_judge_model == "opus"
    assert cfg.behavior_judge_effort == "high"
    assert cfg.model == config_file.model
    assert cfg.judge_model == config_file.judge_model


def test_existing_custom_model_survives_open_save_and_harness_switch(config_file):
    config_file.model = "company-claude-deployment"
    save_config(config_file)
    before = config_path().read_bytes()
    assert edit_agent(config_file, RIGHT + LEFT + SAVE) == config_file
    assert edit_agent(config_file, SAVE, judge=True) == config_file
    assert config_path().read_bytes() == before


def test_agent_cancel_and_save_failure_preserve_settings(config_file, monkeypatch):
    assert edit_agent(config_file, "\t" + RIGHT + CANCEL) == config_file

    def fail(*args):
        raise OSError("Disk full")

    monkeypatch.setattr(settings, "_save_fields", fail)
    assert edit_agent(config_file, "\t" + RIGHT + SAVE + CANCEL) == config_file
    assert load_config().model == config_file.model


def test_three_submenus_route_to_their_own_settings(config_file, monkeypatch):
    calls = []

    def agent(cfg, *, judge, save):
        calls.append(judge)
        cfg = (replace(cfg, behavior_judge_model="claude-opus-5-5") if judge else
               replace(cfg, model="claude-sonnet-5-5"))
        save(cfg)
        return cfg

    _choices(monkeypatch, ["harness", "judge", "detector", "back"])
    monkeypatch.setattr(llm_settings, "edit_agent", agent)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(LEFT + "\t\tlocal-model" + SAVE)
        cfg = settings.configure_endpoints(config_file)
    assert calls == [False, True]
    assert cfg == load_config()
    assert cfg.model == "claude-sonnet-5-5"
    assert cfg.behavior_judge_model == "claude-opus-5-5"
    assert cfg.feature_model == "local-model"


def test_discovery_thinking_and_effort_are_saved_independently(config_file):
    cfg = edit_agent(config_file, "\t\t" + RIGHT + "\t" + RIGHT * 3 + SAVE)
    assert (cfg.thinking, cfg.effort) == ("adaptive", "high")
    assert (cfg.behavior_judge_thinking, cfg.behavior_judge_effort) == ("adaptive", "medium")
    assert cfg == load_config()


def test_legacy_judge_model_uses_budget_and_clears_unsupported_effort(config_file):
    # Sonnet -> newly discovered model -> Haiku. The On setting becomes budgeted, and a
    # budget below the minimum is rejected until it is corrected in place.
    cfg = edit_agent(config_file, RIGHT * 2 + "\t\t" + CLEAR + "100" + SAVE
                     + CLEAR + "2048" + SAVE, judge=True)
    assert cfg.behavior_judge_model == "haiku"
    assert cfg.behavior_judge_thinking == "enabled"
    assert cfg.behavior_judge_effort == "default"
    assert cfg.behavior_judge_thinking_budget_tokens == 2048
    assert cfg == load_config()


def test_failed_catalog_preserves_saved_model_and_allows_manual_entry(config_file, monkeypatch):
    async def fail(*args, **kwargs):
        return ModelCatalog(error="Harness unavailable")

    monkeypatch.setattr(llm_settings, "load_model_catalog_async", fail)
    before = config_path().read_bytes()
    assert edit_agent(config_file, SAVE) == config_file
    assert config_path().read_bytes() == before
    cfg = edit_agent(config_file, "\t\t" + CLEAR + "custom-deployment" + SAVE)
    assert cfg.model == "custom-deployment"


def test_late_catalog_response_does_not_replace_the_selected_harness(harness_catalog):
    form = _AgentForm(claude_config(), lambda _: None, judge=False, catalog=harness_catalog)
    original_choices = form.model.options.copy()
    asyncio.run(form.load_models("codex"))
    assert form.harness.value == "claude"
    assert form.model.options == original_choices
    assert "codex" in form.catalogs


def test_closing_form_cancels_pending_catalog_query(config_file, monkeypatch):
    state = {"started": False, "closed": False}

    async def pending(*args, **kwargs):
        state["started"] = True
        try:
            await asyncio.Event().wait()
        finally:
            state["closed"] = True

    monkeypatch.setattr(llm_settings, "load_model_catalog_async", pending)
    assert edit_agent(config_file, CANCEL) == config_file
    assert state == {"started": True, "closed": True}


def test_late_capabilities_clear_unsupported_effort_for_an_edited_model(harness_catalog):
    form = _AgentForm(claude_config(effort="max"), lambda _: None, judge=False)
    form.model.move(1)  # Custom model while discovery is still pending.
    form.custom_model.text = "haiku"
    assert form.effort.value == "max"
    form.apply_catalog(harness_catalog)
    assert form.selected_model_id() == "haiku"
    assert form.effort.value == "default"
    assert not form.show_effort()


def test_loading_preserves_adaptive_controls_and_does_not_rewrite_custom_model(harness_catalog):
    form = _AgentForm(claude_config(), lambda _: None, judge=True)

    def controls():
        return (form.thinking.value, dict(form.thinking.options)[form.thinking.value],
                form.effort.value, form.show_effort(), form.show_budget())

    expected = ("adaptive", "On", "medium", True, False)
    assert controls() == expected
    # Selecting a custom deployment before metadata arrives must not change
    # adaptive thinking into budgeted thinking or drop its explicit effort.
    form.model.move(1)
    form.custom_model.text = "team-deployment"
    assert controls() == expected
    form.custom_model.text = "claude-sonnet-5-5"
    form.apply_catalog(harness_catalog)
    assert controls() == expected
    assert form.selected_model_id() == "claude-sonnet-5-5"


def test_codex_selection_saves_extended_effort_and_switches_automatic_judge(
    config_file, harness_catalog,
):
    config_file.harness_defaults["codex"].update({
        "model": "codex-only-model", "behavior_judge_model": "codex-only-model",
    })
    with codex_form(config_file, harness_catalog) as (form, pipe):
        form.effort.move(2)
        pipe.send_text(SAVE)
        run_app(form.app)
    cfg = load_config()
    assert cfg.harness == "codex" and cfg.model == "codex-only-model"
    assert cfg.effort == "ultra" and cfg.thinking == "default"
    assert cfg.behavior_judge_harness == "auto"
    assert cfg.behavior_judge_model == "codex-only-model"
    assert cfg.behavior_judge_thinking == "default"
    assert endpoint_for_config(cfg) == endpoint_for_config(config_file)


def test_judge_harness_can_be_pinned_independently(config_file, harness_catalog):
    config_file.harness_defaults["codex"]["behavior_judge_model"] = "codex-only-model"
    with codex_form(config_file, harness_catalog, judge=True) as (form, pipe):
        form.effort.move(2)
        pipe.send_text(SAVE)
        run_app(form.app)
    cfg = load_config()
    assert cfg.harness == "claude" and cfg.model == config_file.model
    assert cfg.behavior_judge_harness == "codex"
    assert cfg.behavior_judge_model == "codex-only-model"
    assert cfg.behavior_judge_effort == "ultra"
    assert cfg.behavior_judge_thinking == "default"


def test_new_codex_defaults_use_live_catalog_and_keep_judge_independent(config_file, monkeypatch):
    calls = []
    catalog = ModelCatalog((
        HarnessModel("gpt-6-astra", "Astra from installed Codex", supports_effort=True,
                     effort_levels=("low", "medium", "high")),
        HarnessModel("gpt-6.1-sol", "Sol from installed Codex", supports_effort=True,
                     effort_levels=("medium", "high")),
        HarnessModel("future-codex-model", "New runtime model", supports_effort=True,
                     effort_levels=("low", "ultra")),
    ))

    async def load(cli_path=None, *, harness):
        calls.append((harness, cli_path))
        return catalog

    monkeypatch.setattr(llm_settings, "load_model_catalog_async", load)
    cfg = Config(codex_cli="/configured/codex")
    save_config(cfg)
    assert edit_agent(cfg, SAVE).model == cfg.model
    judge = edit_agent(cfg, SAVE, judge=True)
    assert behavior_judge_settings(judge) == behavior_judge_settings(cfg)
    # A newly advertised model remains selectable without another preset change.
    changed = edit_agent(judge, "\t" + RIGHT * 2 + SAVE)
    assert changed.model == "future-codex-model"
    assert changed.behavior_judge_model == cfg.behavior_judge_model
    assert changed == load_config()
    assert calls == [("codex", "/configured/codex")] * 3


@pytest.mark.parametrize("judge", [False, True])
def test_switch_to_codex_prefers_role_model_over_cli_default(
    config_file, harness_catalog, monkeypatch, judge,
):
    async def load(cli_path=None, *, harness):
        assert harness == "codex"
        return ModelCatalog(tuple(
            HarnessModel(model, model, supports_effort=True,
                         effort_levels=("medium", "high"))
            for model in ("gpt-6.1-sol", "gpt-6-astra")
        ))

    monkeypatch.setattr(llm_settings, "load_model_catalog_async", load)
    with codex_form(config_file, harness_catalog, judge=judge) as (form, pipe):
        assert form.selected_model_id() == ("gpt-6.1-sol" if judge else "gpt-6-astra")
        pipe.send_text(SAVE)
        run_app(form.app)

    cfg = load_config()
    assert cfg.behavior_judge_model == "gpt-6.1-sol"
    if judge:
        assert cfg.harness == "claude" and cfg.model == config_file.model
        assert cfg.behavior_judge_harness == "codex"
    else:
        assert cfg.harness == "codex" and cfg.model == "gpt-6-astra"
        assert cfg.behavior_judge_harness == "auto"


def test_harness_switch_uses_saved_profile_defaults_for_custom_deployments(
    config_file, harness_catalog, monkeypatch,
):
    config_file.harness_defaults["codex"].update({
        "model": "private-discovery", "thinking": "default", "effort": "ultra",
        "thinking_budget_tokens": 2048, "behavior_judge_model": "private-judge",
        "behavior_judge_thinking": "default", "behavior_judge_effort": "high",
        "behavior_judge_thinking_budget_tokens": 3072,
    })
    save_config(config_file)
    configured = load_config()

    async def load(cli_path=None, *, harness):
        # The runtime does not advertise the private deployment.
        return ModelCatalog((HarnessModel("runtime-default", "Runtime default"),))

    monkeypatch.setattr(llm_settings, "load_model_catalog_async", load)
    with codex_form(configured, harness_catalog) as (form, _):
        assert form.selected_model_id() == "private-discovery"
        assert form.effort.value == "ultra"
        assert form.budget.text == "2048"
        form.app.exit = lambda: None
        form.commit()
        assert not form.error

    cfg = load_config()
    assert cfg.model == "private-discovery" and cfg.effort == "ultra"
    assert cfg.thinking_budget_tokens == 2048
    assert cfg.behavior_judge_model == "private-judge"
    assert cfg.behavior_judge_effort == "high"
    assert cfg.behavior_judge_thinking_budget_tokens == 3072
