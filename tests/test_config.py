# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config and .env discovery, packaged TOML defaults, migration, validation and harness settings."""

from __future__ import annotations

import fnmatch
import json
import os
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomlkit

from autofa import cli, config, defaults
from autofa.config import (
    Config,
    global_env_path,
    load_config,
    load_dotenv,
    migrate_config_toml,
    normalize_objective_name,
    parse_exploration_seed,
    resolve_objective_axes,
)
from autofa.harness.settings import behavior_judge_settings, harness_model

from support.config import save_config

# Config tests must not depend on the developer's installed harnesses.
pytestmark = pytest.mark.usefixtures("installed_codex")


# Config discovery per checkout, profile and deployment preset


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    source = root / "src/autofa"
    source.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "autofa"\n')
    monkeypatch.setattr(config, "__file__", str(source / "config.py"))
    monkeypatch.chdir(root)
    monkeypatch.delenv("AUTOFA_CONFIG", raising=False)
    monkeypatch.delenv("AUTOFA_HOME", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return root, home


def _full_config(path: Path, **values) -> str:
    """Write the packaged defaults with ``values``, a user comment and an unknown key."""
    document = tomlkit.parse(defaults.defaults_text())
    for name, value in values.items():
        document[name] = value
    text = tomlkit.dumps(document) + '\n# Keep my note\nprivate_extension = "retained"\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def test_new_worktree_imports_corrected_global_scoring_sample(project, tmp_path, monkeypatch):
    root, home = project
    global_path = home / ".autofa/config.toml"
    # Home settings (including unknown keys and comments) are copied once, then read.
    global_text = _full_config(global_path, llm_eval_sample=200, top_x=9)
    # An older worktree can still have a different explicit configuration.
    previous = tmp_path / "previous-worktree/autofa.toml"
    previous_text = _full_config(previous, llm_eval_sample=1600)
    monkeypatch.chdir(root / "src")

    cfg = config.load_config()

    assert config.config_path() == root / "autofa.toml"
    assert cfg.llm_eval_sample == 200 and cfg.top_x == 9
    assert (root / "autofa.toml").read_text() == global_text
    assert global_path.read_text() == global_text
    assert previous.read_text() == previous_text
    # Later changes in the global defaults must not replace the worktree's choice.
    _full_config(global_path, llm_eval_sample=1600)
    assert config.load_config().llm_eval_sample == 200


# "auto" dispatches like "responses" once the options come from an existing file.
@pytest.mark.parametrize("origin,policy", [
    ("fresh", "auto"),
    ("corrected_global", "chat_completions"),
    ("local_over_stale_global", "responses"),
])
def test_reasoning_off_survives_config_creation_settings_save_and_api_dispatch(
    project, origin, policy,
):
    from openai import OpenAI

    from autofa.llm.endpoints import endpoint_for_config, improvement_endpoint_for_config
    from autofa.llm.hub import HubLLM
    from autofa.ui.settings import save_analysis_settings

    try:
        import httpx
    except ImportError:
        import httpx2 as httpx

    root, home = project
    global_path = home / ".autofa/config.toml"
    local_path = root / "autofa.toml"
    off = {"reasoning_effort": "none"}
    if origin != "fresh":
        options = off if origin == "corrected_global" else {}
        _full_config(global_path, feature_request_options=options, judge_request_options=options)
        if origin == "local_over_stale_global":
            _full_config(local_path, feature_request_options=off, judge_request_options=off)
    global_before = global_path.read_text() if global_path.exists() else None

    cfg = config.load_config()
    assert config.config_path() == local_path
    assert endpoint_for_config(cfg).request_options == off
    assert improvement_endpoint_for_config(cfg).request_options == off
    assert endpoint_for_config(cfg, "judge").request_options == (
        {} if origin == "fresh" else off
    )
    cfg.analysis_focus = "harness"
    save_analysis_settings(cfg)
    cfg = config.load_config()
    cfg.feature_api = policy
    save_config(cfg)
    reloaded = config.load_config()
    assert reloaded.analysis_focus == "harness"
    assert (global_path.read_text() if global_path.exists() else None) == global_before

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json={
                "id": "reasoning-off", "object": "response", "created_at": 0,
                "model": "test/model", "status": "completed", "output": [{
                    "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "yes", "annotations": []}],
                }],
            })
        return httpx.Response(200, json={
            "id": "reasoning-off", "object": "chat.completion", "created": 0,
            "model": "test/model", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "yes"},
            }],
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        for endpoint in (endpoint_for_config(reloaded), improvement_endpoint_for_config(reloaded)):
            api = OpenAI(api_key="test", base_url="https://provider.example/v1",
                         http_client=transport)
            llm = HubLLM(api, model=endpoint.model, request_options=endpoint.request_options,
                         api=endpoint.api)
            assert llm.classify("Explain the evidence briefly.", "retained evidence") == "yes"
    assert len(requests) == 2
    if policy == "chat_completions":
        assert all(request["reasoning_effort"] == "none" for request in requests)
    else:
        assert all(request["reasoning"] == {"effort": "none"} for request in requests)


@pytest.mark.parametrize("role,options", [
    ("feature", {}),
    ("judge", {"extra_body": {"reasoning": {"enabled": False}}}),
])
def test_explicit_api_reasoning_options_survive_config_migration_independently(
    project, role, options,
):
    from autofa.llm.endpoints import endpoint_for_config

    root, _ = project
    document = tomlkit.document()
    document["harness"] = "codex"
    document[f"{role}_request_options"] = options
    (root / "autofa.toml").write_text(tomlkit.dumps(document))

    save_config(config.load_config())
    reloaded = config.load_config()

    other_role = "judge" if role == "feature" else "feature"
    assert endpoint_for_config(reloaded, role).request_options == options
    assert endpoint_for_config(reloaded, other_role).request_options == (
        {} if other_role == "judge" else {"reasoning_effort": "none"}
    )


def test_local_settings_survive_stale_global_and_settings_save(project, monkeypatch):
    from autofa.ui.settings import save_analysis_settings

    root, home = project
    path = root / "autofa.toml"
    path.write_text(
        "llm_eval_sample = 200\ndiscovery_stall_rounds = 4\ndiscovery_plateau_patience = 7\n")
    global_path = home / ".autofa/config.toml"
    global_text = _full_config(global_path, llm_eval_sample=1600, discovery_stall_rounds=9,
                               discovery_plateau_patience=12)
    # Launching outside the checkout still uses the installed source's settings.
    monkeypatch.chdir(home)
    expected = {"llm_eval_sample": 200, "discovery_stall_rounds": 4,
                "discovery_plateau_patience": 7}

    cfg = config.load_config()
    assert config.config_path() == path
    assert {key: getattr(cfg, key) for key in expected} == expected
    cfg.analysis_focus = "harness"
    save_analysis_settings(cfg)
    reloaded = config.load_config()
    assert reloaded.analysis_focus == "harness"
    assert tomllib.loads(path.read_text())["llm_eval_sample"] == 200
    save_config(reloaded)
    reloaded = config.load_config()
    assert {key: getattr(reloaded, key) for key in expected} == expected
    assert global_path.read_text() == global_text


# A missing local value migrates to the packaged default and an intentional one is
# kept; neither reads the stale global value.
@pytest.mark.parametrize("local,setting,expected", [
    ("# Older local configuration\nmin_failure_modes = 1\n", "llm_eval_sample", 200),
    ("llm_eval_sample = 0\n", "llm_eval_sample", 0),
    ("discovery_plateau_patience = 6\n", "discovery_stall_rounds", 3),
], ids=["missing_sample", "intentional_full_scan", "missing_stall_rounds"])
def test_missing_local_settings_use_defaults_not_stale_global(
    project, local, setting, expected,
):
    root, home = project
    global_path = home / ".autofa/config.toml"
    global_text = _full_config(global_path, llm_eval_sample=1600, discovery_stall_rounds=9)
    path = root / "autofa.toml"
    path.write_text(local)

    assert getattr(config.load_config(), setting) == expected
    assert tomllib.loads(path.read_text())[setting] == expected
    assert global_path.read_text() == global_text


@pytest.mark.parametrize("override", ["AUTOFA_CONFIG", "AUTOFA_HOME"])
def test_explicit_scoring_profile_beats_both_worktree_and_global(project, monkeypatch, override):
    root, home = project
    local_path = root / "autofa.toml"
    local_text = _full_config(local_path, llm_eval_sample=1600)
    global_path = home / ".autofa/config.toml"
    global_text = _full_config(global_path, llm_eval_sample=1600)
    selected_path = home / "selected-profile/config.toml"
    _full_config(selected_path, llm_eval_sample=200)
    monkeypatch.setenv(override, str(
        selected_path if override == "AUTOFA_CONFIG" else selected_path.parent,
    ))

    assert config.config_path() == selected_path
    cfg = config.load_config()
    assert cfg.llm_eval_sample == 200
    save_config(cfg)
    assert config.load_config().llm_eval_sample == 200
    assert local_path.read_text() == local_text
    assert global_path.read_text() == global_text


def test_explicit_config_overrides_profile_and_creates_parent(project, monkeypatch):
    root, home = project
    monkeypatch.setenv("AUTOFA_HOME", str(home / "profile"))
    path = root / "custom/sub/config.toml"
    monkeypatch.setenv("AUTOFA_CONFIG", str(path))
    assert config.config_path() == path
    save_config(config.Config(top_x=2))
    assert config.load_config().top_x == 2
    assert not (home / "profile/config.toml").exists()


def test_installed_tool_finds_project_config_or_repo_before_home(project, tmp_path, monkeypatch):
    _, home = project
    monkeypatch.setattr(config, "__file__", str(tmp_path / "site-packages/autofa/config.py"))
    root = tmp_path / "other"
    root.mkdir()
    (root / ".git").write_text("gitdir: elsewhere")
    nested = root / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    assert config.config_path() == root / "autofa.toml"
    config.ensure_default_config()
    assert config.project_root() == root
    monkeypatch.chdir(home)
    assert config.config_path() == home / ".autofa/config.toml"


def test_explicit_project_config_takes_priority_over_editable_checkout(
    project, tmp_path, monkeypatch,
):
    root = tmp_path / "other-project"
    root.mkdir()
    (root / "autofa.toml").write_text(defaults.defaults_text())
    monkeypatch.chdir(root)
    assert config.config_path() == root / "autofa.toml"


def test_relative_skill_paths_resolve_beside_config(project, monkeypatch):
    from autofa.session.guidance import snapshot_guidance

    root, _ = project
    skills = root / "skills"
    skills.mkdir()
    (skills / "harness.md").write_text("Harness context")
    (root / "extra.md").write_text("Additional context")
    text = defaults.defaults_text().replace('skills_dir = ""', 'skills_dir = "skills"')
    text = text.replace('skill_paths = []', 'skill_paths = ["extra.md"]', 1)
    (root / "autofa.toml").write_text(text)
    monkeypatch.chdir(root / "src")
    cfg = config.load_config()
    assert cfg.skills_dir == str(skills)
    assert len(snapshot_guidance(cfg)["skill_snapshots"]) == 2


def test_invalid_legacy_config_is_not_copied_or_silently_replaced(project):
    root, home = project
    legacy = home / ".autofa/config.toml"
    legacy.parent.mkdir()
    legacy.write_text('analysis_focus = "unterminated')
    with pytest.raises(tomllib.TOMLDecodeError):
        config.load_config()
    assert not (root / "autofa.toml").exists()
    assert legacy.read_text() == 'analysis_focus = "unterminated'


def test_agent_prompt_identifies_actual_config_source(project):
    from autofa.session.prompts import build_system_prompt

    root, _ = project
    prompt = build_system_prompt(exploration_ssot=False)
    assert f"Live config (from {root / 'autofa.toml'})" in prompt


@pytest.fixture
def deployment(project, monkeypatch):
    root, _ = project
    preset = root / "config/internal.toml"
    preset.parent.mkdir()
    preset.write_text(
        'harness = "codex"\n'
        'data_api_name = "Team Eval Server"\n'
        'data_api_base_url = "https://eval.example.org"\n'
        'remote_library_base_url = "https://library.example.org"\n'
        'hub_base_url = "https://inference.example.org/v1"\n'
        'hub_api_key_env = "TEAM_API_KEY"\n'
        'feature_model = "team-detector"\n'
        'judge_model = "team-judge"\n'
    )
    monkeypatch.delenv("AUTOFA_SITE_CONFIG")
    return preset


def test_editable_install_automatically_loads_preset_and_codex(deployment, project, monkeypatch):
    from autofa.harness import settings

    root, home = project
    original = deployment.read_text()
    # Launching outside the checkout still uses the installed source's preset.
    monkeypatch.chdir(home)
    monkeypatch.setattr(settings, "detect_default_harness",
                        lambda **kw: pytest.fail("Ignored deployment harness"))
    cfg = config.load_config()
    assert config.site_config_path() == deployment
    assert config.config_path() == root / "autofa.toml"
    preset = tomllib.loads(original)
    assert {key: getattr(cfg, key) for key in preset} == preset
    assert deployment.read_text() == original


def test_site_defaults_fill_missing_fields_but_preserve_saved_choices(deployment, project):
    root, _ = project
    path = root / "autofa.toml"
    path.write_text(
        '# Personal settings\nharness = "claude"\n'
        'data_api_base_url = ""\nremote_library_base_url = "https://personal.example.org"\n'
        'remote_library_token_env = "MY_PAT"\n'
        'data_api_name = "Personal Eval"\nfeature_model = "personal-detector"\n'
    )
    cfg = config.load_config()
    assert cfg.harness == "claude"
    assert cfg.data_api_base_url == ""
    assert cfg.remote_library_base_url == "https://personal.example.org"
    assert cfg.remote_library_token_env == "MY_PAT"
    assert not hasattr(cfg, "remote_library_token")  # Names the env var, never the token.
    assert cfg.data_api_name == "Personal Eval"
    assert cfg.feature_model == "personal-detector"
    assert cfg.hub_base_url == "https://inference.example.org/v1"
    before = path.read_text()
    assert before.startswith("# Personal settings")
    # Once saved, values remain personal settings even if deployment defaults change.
    deployment.write_text('hub_base_url = "https://updated.example.org/v1"\n')
    assert config.load_config() == cfg
    assert path.read_text() == before


def test_site_can_scope_discovery_options_without_changing_shared_judge(deployment):
    deployment.write_text('harness = "codex"\n'
                          'improvement_request_options = { reasoning_effort = "low" }\n')
    loaded = config.load_config()
    assert loaded.improvement_request_options == {"reasoning_effort": "low"}
    assert loaded.judge_request_options == {}


def test_imported_legacy_home_config_keeps_claude_and_disabled_services(deployment, project):
    root, home = project
    legacy = home / ".autofa/config.toml"
    legacy.parent.mkdir()
    original = 'data_api_base_url = ""\nremote_library_base_url = ""\n'
    legacy.write_text(original)
    cfg = config.load_config()
    assert cfg.harness == "claude"  # Pre-harness configs used Claude.
    # The implicit Claude harness keeps Claude's models, not the preset harness's models.
    assert (cfg.model, cfg.behavior_judge_model) == (
        defaults.harness_default("claude", "model"),
        defaults.harness_default("claude", "behavior_judge_model"),
    )
    assert tomllib.loads((root / "autofa.toml").read_text())["harness"] == "claude"
    assert cfg.data_api_base_url == cfg.remote_library_base_url == ""
    assert cfg.hub_base_url == "https://inference.example.org/v1"
    assert legacy.read_text() == original


def test_explicit_site_file_replaces_checkout_preset(deployment, project, monkeypatch):
    _, home = project
    site = home / "managed.toml"
    original = 'data_api_base_url = "https://managed.example.org"\n'
    site.write_text(original)
    monkeypatch.setenv("AUTOFA_SITE_CONFIG", str(site))
    monkeypatch.setenv("AUTOFA_CONFIG", str(home / "personal.toml"))
    cfg = config.load_config()
    assert config.site_config_path() == site
    assert cfg.data_api_base_url == "https://managed.example.org"
    assert cfg.remote_library_base_url == ""  # Checkout preset was not also loaded.
    assert site.read_text() == original


def test_empty_site_variable_disables_checkout_preset(deployment, monkeypatch):
    monkeypatch.setenv("AUTOFA_SITE_CONFIG", "")
    assert config.site_config_path() is None
    cfg = config.load_config()
    assert cfg.data_api_base_url == cfg.remote_library_base_url == ""


@pytest.mark.parametrize("packaged", [False, True])
def test_public_installation_has_no_implicit_remote_services(
    project, tmp_path, monkeypatch, packaged,
):
    root, _ = project
    monkeypatch.delenv("AUTOFA_SITE_CONFIG")
    if packaged:
        monkeypatch.setattr(config, "__file__", str(tmp_path / "site-packages/autofa/config.py"))
        # A similarly named file in the caller's repo must not become a site preset.
        (root / "config").mkdir()
        (root / "config/internal.toml").write_text(
            'data_api_base_url = "https://unrelated.example.org"\n'
        )
    assert config.site_config_path() is None
    cfg = config.load_config()
    assert cfg.data_api_base_url == cfg.remote_library_base_url == ""


@pytest.mark.parametrize("contents,error", [
    (None, "Cannot read deployment defaults"),
    ('library_root = "private-path"', "Unsupported deployment settings"),
    ('harness = "unknown"', "must be codex or claude"),
])
def test_invalid_explicit_site_file_fails_before_creating_config(
    project, monkeypatch, contents, error,
):
    root, home = project
    site = home / "managed.toml"
    if contents is not None:
        site.write_text(contents)
    monkeypatch.setenv("AUTOFA_SITE_CONFIG", str(site))
    with pytest.raises(ValueError, match=error):
        config.load_config()
    assert not (root / "autofa.toml").exists()
    if contents is not None:
        assert site.read_text() == contents


@pytest.fixture
def home_screen(project, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)

    from autofa_library_server import local

    monkeypatch.setattr(local, "managed_dashboard", lambda **kwargs: nullcontext(False))
    monkeypatch.setattr(
        cli, "_home_cloud_status_line",
        lambda cfg: pytest.fail("One-shot startup must not contact the cloud"),
    )
    return project


def test_home_and_system_information_refresh_config_source_and_scoring(home_screen, monkeypatch):
    from autofa.ui import settings

    checkout, home = home_screen
    local_path = checkout / "autofa.toml"
    _full_config(local_path, llm_eval_sample=200)
    updated_path = home / "new-profile.toml"
    headers = []
    info = []

    def capture_header(lines):
        headers.append(lines)
        return []

    def select(title, options, **kwargs):
        value = "settings" if len(headers) == 1 else "q"
        return next(option for option in options if option.value == value)

    def change_settings(*, runtime_info):
        _full_config(updated_path, llm_eval_sample=0)
        monkeypatch.setenv("AUTOFA_CONFIG", str(updated_path))
        info.extend(settings.system_info_lines(config.load_config(), runtime_info))

    monkeypatch.setattr(cli, "home_menu_header", capture_header)
    monkeypatch.setattr(cli, "select_option", select)
    monkeypatch.setattr(cli, "notice_fragments", lambda: [])
    monkeypatch.setattr(cli, "_execute_action", lambda label, callback: callback())
    monkeypatch.setattr(settings, "run_settings", change_settings)
    cli.main(["home", "--once"])

    assert len(headers) == 2
    initial = "Iterative LLM scoring: up to 200 examples per detector; final scoring: full dataset"
    assert initial in headers[0]
    updated = "Iterative LLM scoring: full dataset; final scoring: full dataset"
    assert f"Config: {updated_path}" in info
    for lines in (headers[1], info):
        assert updated in lines and initial not in lines



@pytest.mark.parametrize("remote_url,library_line,dashboard_fails", [
    ("", "  Library: http://127.0.0.1:8765", False),
    ("https://library.example.com/", "  Library: https://library.example.com", False),
    ("https://library.example.com", "  Library: https://library.example.com", True),
])
def test_home_header_shows_configured_library_without_scoring(
        home_screen, monkeypatch, remote_url, library_line, dashboard_fails):
    from autofa.ui import banner
    from autofa_library_server import local

    checkout, _home = home_screen
    _full_config(checkout / "autofa.toml", remote_library_base_url=remote_url)
    if dashboard_fails:
        def unavailable(**kwargs):
            raise OSError("port in use")

        monkeypatch.setattr(local, "managed_dashboard", unavailable)
    headers = []

    def capture_header(lines):
        headers.append("".join(text for _style, text, *_ in banner.home_menu_header(lines)))
        return []

    monkeypatch.setattr(cli, "home_menu_header", capture_header)
    monkeypatch.setattr(cli, "select_option", lambda title, options, **kwargs: None)
    monkeypatch.setattr(cli, "notice_fragments", lambda: [])
    cli.main(["home", "--once"])

    assert library_line + "\n" in headers[0]
    assert headers[0].count("Library") == 1
    assert "Iterative LLM scoring" not in headers[0]


def test_remote_library_link_rejects_unsafe_address():
    from autofa.ui import banner

    with pytest.raises(ValueError, match="credentials"):
        banner.home_status_fragments(
            [banner.REMOTE_LIBRARY_PREFIX
             + "https://user:secret@example.com"])  # pragma: allowlist secret

# Packaged defaults, migration and validation


def _raw_defaults():
    return tomllib.loads(defaults.defaults_text())


def _edit_defaults(monkeypatch, **values):
    document = tomlkit.parse(defaults.defaults_text())
    for name, value in values.items():
        document[name] = value
    text = tomlkit.dumps(document)
    monkeypatch.setattr(defaults, "defaults_text", lambda: text)


def test_parse_exploration_seed_auto():
    assert parse_exploration_seed("auto") is None
    assert parse_exploration_seed("AUTO") is None
    assert parse_exploration_seed(None) is None
    assert parse_exploration_seed(42) == 42
    assert parse_exploration_seed("7") == 7


@pytest.mark.parametrize("seed,expected", [('"auto"', None), ('0', 0)])
def test_existing_exploration_seed_survives_default_change(tmp_path, monkeypatch, seed, expected):
    path = tmp_path / "config.toml"
    path.write_text(f'harness = "claude"\nexploration_seed = {seed}\n')
    monkeypatch.setenv("AUTOFA_CONFIG", str(path))
    assert load_config().exploration_seed == expected
    stored = tomllib.loads(path.read_text())["exploration_seed"]
    assert stored == ("auto" if expected is None else expected)
    assert migrate_config_toml(path) is False


@pytest.mark.parametrize("workers", [None, 8])  # 8 is the inclusive upper bound.
def test_judge_workers_migration_is_independent_of_llm_eval_workers(
    tmp_path, monkeypatch, workers,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    content = "llm_eval_sample = 100\nllm_eval_workers = 2\n"
    if workers is not None:
        content += f"judge_workers = {workers}\n"
    path.write_text(content)
    assert migrate_config_toml(path) is True
    cfg = load_config()
    assert cfg.judge_workers == (Config().judge_workers if workers is None else workers)
    assert cfg.llm_eval_workers == 2
    assert migrate_config_toml(path) is False
    assert path.read_text().count("judge_workers =") == 1


@pytest.mark.parametrize(("saved", "expected"), [
    ('hub_base_url = "https://hub.example/v1"\n', "INFERENCE_HUB_API_KEY"),
    ('hub_base_url = "https://hub.example/v1"\n'
     'hub_api_key_env = "TEAM_KEY"\n', "TEAM_KEY"),  # pragma: allowlist secret
    ("llm_eval_workers = 2\n", ""),
])
def test_hub_key_migration_keeps_the_legacy_name_only_for_a_saved_hub_url(
    tmp_path, monkeypatch, saved, expected,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    path.write_text(saved)
    migrate_config_toml(path)
    assert load_config().hub_api_key_env == expected


def test_hub_key_migration_keeps_a_site_preset_key(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    preset = tmp_path / "site.toml"
    preset.write_text('hub_base_url = "https://site.example/v1"\n'
                      'hub_api_key_env = "SITE_KEY"\n')  # pragma: allowlist secret
    monkeypatch.setenv("AUTOFA_SITE_CONFIG", str(preset))
    path = tmp_path / "config.toml"
    path.write_text('hub_base_url = "https://hub.example/v1"\n')
    migrate_config_toml(path)
    cfg = load_config()
    assert (cfg.hub_base_url, cfg.hub_api_key_env) == ("https://hub.example/v1", "SITE_KEY")


def test_behavior_judge_migration_preserves_independent_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    path.write_text('model = "discovery-model"\njudge_model = "hub-model"\n'
                    'behavior_judge_model = "custom-judge"\nbehavior_judge_effort = "low"\n')
    cfg = load_config()
    assert (cfg.behavior_judge_model, cfg.behavior_judge_effort) == ("custom-judge", "low")
    assert cfg.model == "discovery-model" and cfg.judge_model == "hub-model"
    before = path.read_text()
    assert migrate_config_toml(path) is False
    assert path.read_text() == before
    # Per-harness profile tables may repeat these names; the root keys must
    # remain valid TOML and retain their independent user-selected settings.
    stored = tomllib.loads(before)
    assert (stored["behavior_judge_model"], stored["behavior_judge_effort"]) == (
        "custom-judge", "low"
    )


@pytest.mark.parametrize("effort", ["default", "max"])
def test_load_config_accepts_sdk_behavior_judge_efforts(tmp_path, monkeypatch, effort):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(f'behavior_judge_effort = "{effort}"\n')
    assert load_config().behavior_judge_effort == effort


@pytest.mark.parametrize("key,value", [
    ("judge_workers", "0"),
    ("judge_workers", "9"),
    ("behavior_judge_model", '"   "'),
    ("thinking", '"unknown"'),
    ("effort", '"unknown"'),
    ("thinking_budget_tokens", "1023"),
    ("thinking_budget_tokens", "2048.0"),  # In range but not an integer.
    ("behavior_judge_effort", '"unknown"'),
    ("taxonomy_first_pass", '"agentc"'),
    ("taxonomy_second_pass", "[]"),
])
def test_invalid_settings_fail_before_running(tmp_path, monkeypatch, key, value):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "config.toml"))
    (tmp_path / "config.toml").write_text(f"{key} = {value}\n")
    message = "judge_workers must be an integer between 1 and 8" if key == "judge_workers" else key
    with pytest.raises(ValueError, match=message):
        load_config()


def test_thinking_migration_preserves_saved_controls(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "config.toml"))
    path = tmp_path / "config.toml"
    path.write_text('thinking = "enabled"\nthinking_budget_tokens = 2048\n'
                    'behavior_judge_thinking = "default"\nbehavior_judge_effort = "default"\n')
    cfg = load_config()
    assert cfg.thinking == "enabled" and cfg.thinking_budget_tokens == 2048
    assert cfg.behavior_judge_thinking == cfg.behavior_judge_effort == "default"
    saved = tomllib.loads(path.read_text())
    for key in ("thinking", "effort", "thinking_budget_tokens", "behavior_judge_thinking",
                "behavior_judge_effort", "behavior_judge_thinking_budget_tokens"):
        assert saved[key] == getattr(cfg, key)
    assert migrate_config_toml(path) is False


def test_normalize_objective_names_and_resolve_active_axes():
    assert normalize_objective_name("precision") == "ppv_at_pi0"
    assert normalize_objective_name("ppv") == "ppv_at_pi0"
    assert normalize_objective_name("purity") == "behavior_purity"
    cfg = Config(objective_axes=["precision", "task_recall", "purity"],
                 objective_weights=[1, 0, 1])
    assert resolve_objective_axes(cfg) == ["ppv_at_pi0", "behavior_purity"]


def test_migration_fills_all_missing_root_defaults_without_polluting_custom_tables(tmp_path):
    path = tmp_path / "config.toml"
    # Commented or nested copies of a root name do not hide the missing root setting.
    path.write_text(
        '# My explicit preferences\ntop_x = 11\n# exploration_ssot is absent at the root\n'
        '[feature_request_options]\n# Keep this provider knob\ntemperature = 0.3\n'
        '[feature_request_options.extra_body]\nenable_thinking = false\n'
        '[custom]\nexploration_ssot = false\njudge_workers = 1\n[custom.ui]\nwidth = 120\n'
    )
    assert migrate_config_toml(path)
    raw = path.read_text()
    migrated = tomllib.loads(raw)
    assert {field.name for field in fields(Config)} <= migrated.keys()
    assert migrated["top_x"] == 11
    assert migrated["exploration_ssot"] is True
    assert migrated["judge_workers"] == Config().judge_workers
    assert migrated["feature_request_options"] == {
        "temperature": 0.3, "extra_body": {"enable_thinking": False},
    }
    assert migrated["custom"] == {
        "exploration_ssot": False, "judge_workers": 1, "ui": {"width": 120},
    }
    assert "# My explicit preferences" in raw
    assert "# Keep this provider knob" in raw
    assert not migrate_config_toml(path)
    assert path.read_text() == raw


def test_partial_harness_profile_and_unknown_tables_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "config.toml"))
    path = tmp_path / "config.toml"
    path.write_text(
        'harness = "codex"\n'
        '[harness_defaults.codex]\nmodel = "my-codex-profile"\n'
        '[custom]\nnotes = ["keep this"]\n'
    )
    assert migrate_config_toml(path)
    migrated = tomllib.loads(path.read_text())
    assert migrated["harness_defaults"]["codex"]["model"] == "my-codex-profile"
    assert migrated["model"] == "my-codex-profile"
    assert migrated["custom"]["notes"] == ["keep this"]
    assert not migrate_config_toml(path)


def test_invalid_toml_is_never_rewritten_by_migration(tmp_path):
    path = tmp_path / "config.toml"
    original = 'feature_model = "unfinished\n'
    path.write_text(original)
    assert migrate_config_toml(path) is False
    assert path.read_text() == original


# The packaged TOML document is the source of every application default.


def test_every_config_field_is_declared_in_toml_and_matches_its_value():
    raw = _raw_defaults()
    assert {field.name for field in fields(config.Config)} == set(raw)
    expected = dict(raw)
    if expected["exploration_seed"] == "auto":
        expected["exploration_seed"] = None
    # Config normalizes the policy, filling version, reserves and cost handling.
    from autofa.session.campaign_policy import validate_campaign_policy

    expected["campaign_policy"] = validate_campaign_policy(expected["campaign_policy"])
    assert asdict(config.Config()) == expected


def test_packaged_defaults_are_readable_and_included_in_package_data():
    resource = resources.files("autofa").joinpath("defaults.toml")
    assert resource.is_file()
    assert tomllib.loads(resource.read_text(encoding="utf-8")) == _raw_defaults()
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    patterns = project["tool"]["setuptools"]["package-data"]["autofa"]
    assert any(fnmatch.fnmatch("defaults.toml", pattern) for pattern in patterns)


def test_mutable_defaults_do_not_leak_between_configs_or_resource_reads():
    first, second = config.Config(), config.Config()
    expected = asdict(second)
    for field in fields(config.Config):
        value = getattr(first, field.name)
        if isinstance(value, list):
            value.append("changed for this instance")
        elif isinstance(value, dict):
            value["instance_only"] = True
    first.harness_defaults["claude"]["model"] = "changed-nested-model"
    assert asdict(second) == expected
    assert asdict(config.Config()) == expected
    profiles = defaults.default_value("harness_defaults")
    profiles["claude"]["model"] = "changed-resource-copy"
    assert defaults.default_value("harness_defaults") == expected["harness_defaults"]


def test_auto_seed_default_is_normalized_from_toml(monkeypatch):
    _edit_defaults(monkeypatch, exploration_seed="auto")
    assert defaults.default_value("exploration_seed") is None
    assert config.Config().exploration_seed is None


def test_edited_toml_drives_config_new_file_and_missing_key_migration(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "fresh.toml"))
    edited = {"judge_workers": 6, "sample_fail_default": 17,
              "feature_model": "test-provider/detector-next"}
    _edit_defaults(monkeypatch, **edited)
    assert {key: getattr(config.Config(), key) for key in edited} == edited
    generated = tomllib.loads(config.ensure_default_config().read_text())
    assert {key: generated[key] for key in edited} == edited
    assert {key: getattr(config.load_config(), key) for key in edited} == edited

    # Explicit settings survive, even where the canonical default changed.
    existing = tmp_path / "existing.toml"
    existing.write_text('harness = "codex"\ntop_x = 2\njudge_workers = 2\n')
    assert config.migrate_config_toml(existing)
    migrated = tomllib.loads(existing.read_text())
    assert (migrated["top_x"], migrated["judge_workers"]) == (2, 2)
    assert migrated["sample_fail_default"] == 17
    assert migrated["feature_model"] == "test-provider/detector-next"
    before = existing.read_text()
    assert not config.migrate_config_toml(existing)
    assert existing.read_text() == before
    monkeypatch.setenv("AUTOFA_CONFIG", str(existing))
    assert config.load_config().judge_workers == 2


def test_concurrent_first_launch_never_overwrites_the_config_it_lost_the_race_to(
    tmp_path, monkeypatch,
):
    path = tmp_path / "autofa.toml"
    dumps = tomlkit.dumps

    def racing_dumps(document):
        # Another launch creates the file after this one saw it was absent.
        if not path.exists():
            path.write_text('judge_workers = 2\n')
        return dumps(document)

    monkeypatch.setattr(tomlkit, "dumps", racing_dumps)
    assert config.ensure_default_config(path=path) == path
    assert tomllib.loads(path.read_text())["judge_workers"] == 2


def test_evaluation_caps_follow_edited_toml_unless_explicitly_supplied(monkeypatch):
    from autofa.mcp.tools_eval import resolve_eval_limit

    _edit_defaults(monkeypatch, default_eval_limit=17, max_eval_limit=80)
    assert resolve_eval_limit(None) == 17
    assert resolve_eval_limit(200) == 80
    assert resolve_eval_limit(None, cfg_default=3, cfg_max=4) == 3


def test_sparse_endpoint_config_uses_toml_defaults_and_retains_explicit_endpoint(monkeypatch):
    from autofa.llm.endpoints import endpoint_for_config

    _edit_defaults(monkeypatch, hub_base_url="https://custom.example/v1",
                   hub_api_key_env="TEST_PROVIDER_TOKEN", feature_model="provider/new-model",
                   feature_api="decisions", feature_request_options={"reasoning_effort": "low"})
    endpoint = endpoint_for_config(SimpleNamespace())
    assert (endpoint.base_url, endpoint.api_key_env, endpoint.model, endpoint.api) == (
        "https://custom.example/v1", "TEST_PROVIDER_TOKEN", "provider/new-model", "decisions"
    )
    assert endpoint.request_options == {"reasoning_effort": "low"}
    explicit = endpoint_for_config(SimpleNamespace(
        feature_base_url="http://127.0.0.1:8000/v1", feature_api_key_env="none",
        feature_model="local-model", feature_api="chat_completions", feature_request_options={},
    ))
    assert (explicit.base_url, explicit.api_key_env, explicit.model, explicit.api) == (
        "http://127.0.0.1:8000/v1", "none", "local-model", "chat_completions"
    )
    assert explicit.request_options == {}


def test_edited_harness_profile_reaches_defaults_config_and_codex_without_reimport(monkeypatch):
    from autofa.harness import codex

    document = tomlkit.parse(defaults.defaults_text())
    document["harness_defaults"]["codex"]["model"] = "provider/new-codex-model"
    document["harness_defaults"]["codex"]["effort"] = "high"
    text = tomlkit.dumps(document)
    monkeypatch.setattr(defaults, "defaults_text", lambda: text)
    assert defaults.harness_default("codex", "model") == "provider/new-codex-model"
    assert config.Config().harness_defaults["codex"]["model"] == "provider/new-codex-model"
    monkeypatch.setattr(codex, "_find_codex_cli", lambda _: "/fake/codex")
    harness = codex.CodexHarness()
    assert (harness.model, harness.effort) == ("provider/new-codex-model", "high")
    explicit = codex.CodexHarness(model="explicit-model", effort="low")
    assert (explicit.model, explicit.effort) == ("explicit-model", "low")


# Harness selection and per-harness defaults


def test_judge_can_use_a_different_harness_from_discovery():
    cfg = Config(harness="codex", behavior_judge_harness="claude", claude_cli="/custom/claude",
                 behavior_judge_model="claude-sonnet-5-5", behavior_judge_thinking="adaptive")
    settings = behavior_judge_settings(cfg)
    assert settings["harness"] == "claude"
    assert settings["model"] == "claude-sonnet-5-5"
    assert settings["thinking"] == "adaptive"
    assert settings["cli_path"] == "/custom/claude"


def test_codex_configuration_roundtrip_accepts_extended_effort(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "autofa.toml"))
    cfg = Config(harness="codex", model="gpt-6-luna", effort="ultra",
                 behavior_judge_model="gpt-6.1-sol", behavior_judge_effort="ultra")
    save_config(cfg)
    loaded = load_config()
    assert loaded.harness == "codex" and loaded.effort == "ultra"
    assert loaded.behavior_judge_harness == "auto"
    # Claude's SDK has no "ultra" effort, so the same judge effort fails there.
    save_config(replace(cfg, behavior_judge_harness="claude"))
    with pytest.raises(ValueError, match="behavior_judge_effort"):
        load_config()


@pytest.mark.parametrize("field", ["harness", "behavior_judge_harness"])
def test_unknown_harness_configuration_is_rejected(tmp_path, monkeypatch, field):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "autofa.toml"))
    save_config(replace(Config(), **{field: "missing"}))
    with pytest.raises(ValueError, match=field):
        load_config()


def test_codex_workspace_has_protocol_skills_and_live_ledger(tmp_path):
    from autofa.session.modes_ledger import upsert_mode
    from autofa.session.prompts import write_session_workspace

    directory = write_session_workspace(tmp_path, objective_config=Config(harness="codex"))
    assert "You run on Codex" in (directory / "autofa_protocol.md").read_text()
    assert (directory / ".agents/skills/discover/SKILL.md").is_file()
    assert (directory / "AGENTS.md").is_file()
    assert not (directory / "CLAUDE.md").exists()
    upsert_mode(directory, name="probe_mode", description="Test mode", mechanism="Test cause",
                status="open", reason="Needs evidence", detector={"name": "probe_mode"})
    assert "probe_mode" in (directory / "AGENTS.md").read_text()


def test_codex_context_handoff_does_not_read_claude_transcripts(tmp_path, monkeypatch):
    import claude_agent_sdk

    from autofa.session.recovery import context_limit_error, context_recovery_prompt

    monkeypatch.setattr(claude_agent_sdk, "get_session_messages", lambda *args, **kwargs:
                        pytest.fail("Wrong harness transcript"))
    assert context_limit_error("Codex context is full. Start a new session.")
    handoff = context_recovery_prompt(tmp_path, prompt="Continue discovery", session_id="codex-id",
                                      harness="codex")
    assert "codex-id" in handoff and "Continue discovery" in handoff
    assert list(Path(tmp_path).glob("context_recovery/*/handoff.json"))


def test_saved_claude_models_and_thinking_are_not_replaced_by_new_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "autofa.toml"))
    previous = Config(
        harness="claude", model="company-claude-deployment",
        behavior_judge_harness="claude", behavior_judge_model="claude-sonnet-5-5",
        behavior_judge_thinking="adaptive", behavior_judge_effort="high",
    )
    save_config(previous)
    assert load_config() == previous


def test_harness_fallbacks_follow_toml_edits_without_module_reload(monkeypatch):
    from autofa.harness import claude, codex

    document = tomlkit.parse(defaults.defaults_text())
    document["harness_defaults"]["codex"].update({
        "model": "configured-discovery", "behavior_judge_model": "configured-judge",
        "effort": "high", "behavior_judge_effort": "ultra",
    })
    document["harness_defaults"]["claude"].update({
        "model": "configured-claude", "thinking": "enabled", "effort": "high",
        "thinking_budget_tokens": 2048,
    })
    monkeypatch.setattr(defaults, "defaults_text", lambda: tomlkit.dumps(document))
    assert harness_model("codex", "configured-claude") == "configured-discovery"
    settings = behavior_judge_settings(SimpleNamespace(harness="codex"))
    assert (settings["model"], settings["effort"]) == ("configured-judge", "ultra")

    monkeypatch.setattr(codex, "CodexHarness", lambda **kwargs: kwargs)
    assert claude.get_harness("codex", "configured-claude") == {
        "model": "configured-discovery", "cli_path": None, "effort": "high",
    }
    monkeypatch.setattr(claude, "ensure_claude_auth", lambda **kwargs: None)
    harness = claude.ClaudeHarness()
    assert harness.model == "configured-claude"
    assert harness.reasoning_options == {"thinking": {"type": "enabled", "budget_tokens": 2048}}


def test_harness_factory_uses_profile_defaults_without_replacing_explicit_controls(monkeypatch):
    from autofa.harness import claude, codex

    cfg = Config(codex_cli="/profile/codex", harness_defaults={"codex": {
        "model": "profile-discovery", "behavior_judge_model": "profile-judge",
        "effort": "ultra", "behavior_judge_effort": "high",
    }})
    monkeypatch.setattr(codex, "CodexHarness", lambda **kwargs: kwargs)
    monkeypatch.setattr(claude, "ensure_claude_auth", lambda **kwargs: pytest.fail("Claude auth"))
    assert claude.get_harness("codex", "claude-opus-5-5", cfg=cfg) == {
        "model": "profile-discovery", "effort": "ultra", "cli_path": "/profile/codex",
    }
    explicit = claude.get_harness("codex", "private-model", effort="low", cfg=cfg)
    assert explicit["model"] == "private-model" and explicit["effort"] == "low"
    assert behavior_judge_settings(replace(cfg, behavior_judge_model="claude-sonnet-5-5"))[
        "model"] == "profile-judge"


def test_edited_claude_default_uses_matching_gateway_alias_only(monkeypatch):
    from autofa.harness.claude import resolve_model

    document = tomlkit.parse(defaults.defaults_text())
    monkeypatch.setattr(defaults, "defaults_text", lambda: tomlkit.dumps(document))
    monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "gateway-opus")
    monkeypatch.delenv("ANTHROPIC_DEFAULT_SONNET_MODEL", raising=False)
    document["harness_defaults"]["claude"]["model"] = "claude-sonnet-5-5"
    # Without a gateway alias for its tier, even the default keeps its literal id.
    assert resolve_model("claude-sonnet-5-5") == "claude-sonnet-5-5"
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "gateway-sonnet")
    assert resolve_model("claude-sonnet-5-5") == "sonnet"
    assert resolve_model("claude-opus-5-5") == "claude-opus-5-5"
    document["harness_defaults"]["claude"]["model"] = "private-deployment"
    assert resolve_model("private-deployment") == "private-deployment"


# Shared .env credentials across independent checkouts


@pytest.fixture
def dotenv_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for key in (
        "AUTOFA_HOME", "INFERENCE_HUB_API_KEY", "DOTENV_LOCAL", "DOTENV_PARENT",
        "DOTENV_PROCESS", "DOTENV_EMPTY", "DOTENV_LITERAL", "DOTENV_QUOTED", "DOTENV_ESCAPED",
    ):
        # setenv records the original state, so teardown also removes values load_dotenv sets.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    project = tmp_path / "checkout"
    (project / "src" / "autofa").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'autofa'\n")
    monkeypatch.chdir(project)
    return project


def write_global(contents: str) -> Path:
    path = global_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def test_precedence_and_supplementing_parent_files(
    dotenv_project: Path, monkeypatch: pytest.MonkeyPatch,
):
    nested = dotenv_project / "nested"
    nested.mkdir()
    nearest = nested / ".env"
    nearest.write_text(
        "DOTENV_PROCESS=nearest\nDOTENV_EMPTY=nearest\nDOTENV_LOCAL=nearest\n"
        "INFERENCE_HUB_API_KEY=project-test-key\n"
    )
    (dotenv_project / ".env").write_text(
        "DOTENV_PROCESS=parent\nDOTENV_LOCAL=parent\nDOTENV_PARENT=parent\n"
        "INFERENCE_HUB_API_KEY=parent-test-key\n"
    )
    write_global(
        "DOTENV_PROCESS=global\nDOTENV_EMPTY=global\nDOTENV_LOCAL=global\n"
        "DOTENV_PARENT=global\nDOTENV_LITERAL=global-only\n"
        "INFERENCE_HUB_API_KEY=shared-test-key\n"
    )
    monkeypatch.setenv("DOTENV_PROCESS", "existing")
    monkeypatch.setenv("DOTENV_EMPTY", "")
    monkeypatch.chdir(nested)

    assert load_dotenv() == nearest
    assert os.environ["DOTENV_PROCESS"] == "existing"
    assert os.environ["DOTENV_EMPTY"] == ""
    assert os.environ["DOTENV_LOCAL"] == "nearest"
    assert os.environ["DOTENV_PARENT"] == "parent"
    assert os.environ["DOTENV_LITERAL"] == "global-only"
    assert os.environ["INFERENCE_HUB_API_KEY"] == "project-test-key"


def test_explicit_profile_uses_its_shared_file(
    dotenv_project: Path, monkeypatch: pytest.MonkeyPatch,
):
    write_global("INFERENCE_HUB_API_KEY=default-test-key\nDOTENV_LOCAL=default\n")
    profile = dotenv_project / "profile"
    monkeypatch.setenv("AUTOFA_HOME", str(profile))
    assert global_env_path() == profile / ".env"
    assert not profile.exists()  # Finding the path must not create a profile.
    shared = write_global("INFERENCE_HUB_API_KEY=profile-test-key\n")

    assert load_dotenv() == shared
    assert os.environ["INFERENCE_HUB_API_KEY"] == "profile-test-key"
    assert "DOTENV_LOCAL" not in os.environ


def test_explicit_path_does_not_load_other_files_or_fall_back(dotenv_project: Path):
    write_global("INFERENCE_HUB_API_KEY=shared-test-key\n")
    (dotenv_project / ".env").write_text("DOTENV_LOCAL=project\n")
    selected = dotenv_project / "selected.env"
    selected.write_text("DOTENV_LITERAL=explicit\n")

    assert load_dotenv(dotenv_project / "missing.env") is None
    assert load_dotenv(selected) == selected
    assert os.environ["DOTENV_LITERAL"] == "explicit"
    assert "INFERENCE_HUB_API_KEY" not in os.environ
    assert "DOTENV_LOCAL" not in os.environ


def test_common_dotenv_syntax_preserves_literal_values(dotenv_project: Path):
    # A fresh checkout without a project file loads the shared file.
    assert not (dotenv_project / ".env").exists()
    shared = write_global(
        "  # Shared credentials\n"
        "  export INFERENCE_HUB_API_KEY = 'test key#fragment' # explanation\n"
        'export\tDOTENV_QUOTED="value # inside quotes"\t# explanation\n'
        "DOTENV_LOCAL = token#fragment # explanation\n"
        "DOTENV_PARENT = value\t# explanation\n"
        "DOTENV_EMPTY=\n"
        "DOTENV_LITERAL=$(do-not-execute)\n"
        'DOTENV_ESCAPED="value \\"quoted\\"" # comment with "quotes"\n'
        "ignored line without equals\n"
    )

    assert load_dotenv() == shared
    assert os.environ["INFERENCE_HUB_API_KEY"] == "test key#fragment"
    assert os.environ["DOTENV_QUOTED"] == "value # inside quotes"
    assert os.environ["DOTENV_LOCAL"] == "token#fragment"
    assert os.environ["DOTENV_PARENT"] == "value"
    assert os.environ["DOTENV_EMPTY"] == ""
    assert os.environ["DOTENV_LITERAL"] == "$(do-not-execute)"
    assert os.environ["DOTENV_ESCAPED"] == r'value \"quoted\"'


def test_project_boundary_does_not_load_unrelated_parent_file(dotenv_project: Path):
    (dotenv_project.parent / ".env").write_text("INFERENCE_HUB_API_KEY=unrelated\n")

    assert load_dotenv() is None
    assert "INFERENCE_HUB_API_KEY" not in os.environ
