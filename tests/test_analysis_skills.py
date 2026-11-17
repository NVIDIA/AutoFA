# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analysis skill documents are snapshotted per run, matched to harness packs and fail visibly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa import config
from autofa.cli import _show_analysis_guidance
from autofa.config import Config, RunConfig, load_config
from autofa.session.analysis_skills import (
    SKILL_MAX_BYTES,
    SKILLS_MAX_BYTES,
    skill_inventory,
    skills_directory,
    snapshot_configured_skills,
    snapshot_skills,
)
from autofa.session.guidance import guidance_prompt
from autofa.ui import settings

from support.config import save_config

pytestmark = pytest.mark.usefixtures("installed_codex")


@pytest.mark.parametrize("name, content, error", [
    ("skill.pdf", b"data", "Unsupported skill format"),
    ("skill.md", b"\xff", "UTF-8"),
    ("skill.md", b" ", "empty"),
    ("skill.md", b"a" * (SKILL_MAX_BYTES + 1), "silently truncated"),
])
def test_invalid_skills_fail_visibly(tmp_path, name, content, error):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(ValueError, match=error):
        snapshot_skills([str(path)])


def test_missing_skills_fail_and_duplicate_paths_are_only_read_once(tmp_path):
    path = tmp_path / "skill.md"
    with pytest.raises(ValueError, match="Cannot read"):
        snapshot_skills([str(path)])
    path.write_text("Harness context")
    assert len(snapshot_skills([str(path), str(path)])) == 1


def test_total_skill_budget_rejects_instead_of_dropping_files(tmp_path):
    paths = []
    for index in range(SKILLS_MAX_BYTES // SKILL_MAX_BYTES + 1):
        path = tmp_path / f"skill-{index}.md"
        path.write_text("a" * SKILL_MAX_BYTES)
        paths.append(str(path))
    with pytest.raises(ValueError, match="total"):
        snapshot_skills(paths)


def test_skill_menu_autosaves_enable_and_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    directory = tmp_path / "skills"
    directory.mkdir()
    path = directory / "harness.md"
    path.write_text("Tool contracts")
    save_config(Config(skills_dir=str(directory)))
    value = str(path)
    remaining = iter(["skills", value, "back", "done"])

    def choose(title, options, **kwargs):
        value = next(remaining)
        if value == "back":
            assert load_config().disabled_skill_paths == ["harness.md"]
            assert skill_inventory(load_config())["active"] == []
        return next(option for option in options if option.value == value)

    monkeypatch.setattr(settings, "select_option", choose)
    settings.run_settings()
    remaining = iter([value, "back"])
    monkeypatch.setattr(settings, "select_option", lambda *args, **kwargs:
                        SimpleNamespace(value=next(remaining)))
    settings._skills_settings(load_config(), persistent=True)
    assert load_config().disabled_skill_paths == []
    assert skill_inventory(load_config())["active"] == [str(path)]


def test_checkout_documents_load_unchanged_with_only_the_matching_pack(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1] / "skills"
    paths = sorted(root.rglob("*.md"))
    assert paths, "The supplied repository skills must be present"
    packs = root / "harness"
    general = {str(p) for p in paths if not p.is_relative_to(packs)}
    codex = {str(p) for p in (packs / "codex").rglob("*.md")}
    if (packs / "README.md").is_file():
        codex.add(str(packs / "README.md"))
    pack_ids = sorted(path.name for path in packs.iterdir() if path.is_dir())
    snapshots = snapshot_configured_skills(Config(), ["codex"])
    assert snapshots["skills_root"] == str(root)
    assert snapshots["skill_exclusions"] == []
    assert {s["path"] for s in snapshots["skill_snapshots"]} == general | codex
    assert snapshots["skill_unmatched_packs"] == [pack for pack in pack_ids if pack != "codex"]
    for skill in snapshots["skill_snapshots"]:
        raw = Path(skill["path"]).read_bytes()
        assert skill["text"] == raw.decode("utf-8-sig")
        assert skill["sha256"] == hashlib.sha256(raw).hexdigest()
        assert skill["name"] == str(Path(skill["path"]).relative_to(root))


def _write_packs(root, *names):
    for name in names:
        (root / "harness" / name).mkdir(parents=True)
        (root / "harness" / name / "tool-calling.md").write_text(f"{name} pack contract")
    (root / "harness" / "README.md").write_text("Pack index")
    (root / "notes.md").write_text("User analysis notes")


@pytest.mark.parametrize("harnesses, loaded", [
    (["openai_api_agent"], set()),  # "pi" inside another token is not a match.
    (["pi-coding-agent"], {"pi"}),
    (["mixed", "Hermes_Agent", "codex"], {"codex", "hermes"}),
])
def test_harness_packs_load_only_for_a_whole_token_match(tmp_path, harnesses, loaded):
    _write_packs(tmp_path, "codex", "opencode", "pi", "hermes")
    snapshot = snapshot_configured_skills(Config(skills_dir=str(tmp_path)), harnesses)
    names = {skill["name"] for skill in snapshot["skill_snapshots"]}
    assert names == {"notes.md", *(f"harness/{pack}/tool-calling.md" for pack in loaded),
                     *(["harness/README.md"] if loaded else [])}
    assert snapshot["skill_unmatched_packs"] == sorted(
        {"codex", "opencode", "pi", "hermes"} - loaded)
    prompt = guidance_prompt(SimpleNamespace(**snapshot))
    assert "User analysis notes" in prompt
    assert ("Pack index" in prompt) == bool(loaded)
    assert ("pi pack contract" in prompt) == ("pi" in loaded)


@pytest.mark.parametrize("override, loaded", [
    ("codex, hermes", {"codex", "hermes"}), ("none", set()),
])
def test_analysis_harness_setting_replaces_the_source_harness(tmp_path, override, loaded):
    _write_packs(tmp_path, "codex", "pi", "hermes")
    cfg = Config(skills_dir=str(tmp_path), analysis_harness=override)
    snapshot = snapshot_configured_skills(cfg, ["codex_agent"])
    packs = {Path(skill["name"]).parent.name for skill in snapshot["skill_snapshots"]
             if skill["name"].count("/") == 2}
    assert packs == loaded
    assert snapshot["skill_unmatched_packs"] == sorted({"codex", "pi", "hermes"} - loaded)


def test_explicit_skill_path_inside_a_pack_always_loads_without_duplicating_files(tmp_path):
    _write_packs(tmp_path, "hermes")
    selected = tmp_path / "harness" / "hermes" / "tool-calling.md"
    # An explicit path to an automatically loaded file is not snapshotted twice.
    automatic = tmp_path / "notes.md"
    cfg = Config(skills_dir=str(tmp_path), skill_paths=[str(selected), str(automatic)])
    snapshot = snapshot_configured_skills(cfg, ["lmarena_v3_benchmark_agent"])
    assert sorted(skill["name"] for skill in snapshot["skill_snapshots"]) == [
        "harness/hermes/tool-calling.md", "notes.md",
    ]


def test_run_snapshots_only_matching_packs_and_resume_keeps_saved_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    root = tmp_path / "skills"
    _write_packs(root, "codex", "hermes")
    cfg = Config(skills_dir=str(root))
    run = RunConfig.create([], "test", [], cfg, source_harness="lmarena_v3_benchmark_agent")
    assert [skill["name"] for skill in run.skill_snapshots] == ["notes.md"]
    assert run.skill_unmatched_packs == ["codex", "hermes"]
    # A mixed run matches each source's recorded harness.
    mixed = RunConfig.create([], "test", [], cfg, source_harness="mixed", reference_data=[
        {"run_id": "a", "export_url": "https://data.example/api/runs/a/export",
         "source_harness": "hermes_agent"},
    ])
    assert "hermes pack contract" in guidance_prompt(mixed)
    assert "codex pack contract" not in guidance_prompt(mixed)
    # Runs saved before pack matching keep every snapshotted pack on Resume, so the
    # judge guidance (part of saved review identities) does not change mid-run.
    legacy = RunConfig.create([], "test", [], cfg, source_harness="codex_agent")
    config_path = legacy.run_dir() / "config.json"
    data = json.loads(config_path.read_text())
    data["skill_snapshots"] = snapshot_configured_skills(
        Config(skills_dir=str(root), analysis_harness="codex hermes"))["skill_snapshots"]
    del data["skill_unmatched_packs"]
    config_path.write_text(json.dumps(data))
    for path in (root / "harness").rglob("*.md"):
        path.unlink()
    restored = RunConfig.load(legacy.run_id)
    assert restored.skill_unmatched_packs == []
    for text in ("codex pack contract", "hermes pack contract", "Pack index"):
        assert text in guidance_prompt(restored)


def test_recursive_markdown_is_snapshotted_with_visible_exclusions(tmp_path):
    harness = tmp_path / "harness" / "example"
    references = tmp_path / "references"
    harness.mkdir(parents=True)
    references.mkdir()
    (harness / "tools.md").write_text("Unique tool contract")
    (references / "feedback.MD").write_text("Ambiguous feedback reference")
    (tmp_path / "unused.md").write_text("Disabled document")
    (tmp_path / "unsupported.pdf").write_bytes(b"not Markdown")
    cfg = Config(skills_dir=str(tmp_path), disabled_skill_paths=["unused.md"])
    snapshot = snapshot_configured_skills(cfg, ["example_agent"])
    prompt = guidance_prompt(SimpleNamespace(**snapshot))
    assert "Unique tool contract" in prompt
    assert "Ambiguous feedback reference" in prompt
    assert "Disabled document" not in prompt
    assert snapshot["skill_exclusions"] == [str(tmp_path / "unused.md")]
    assert snapshot["skill_ignored_paths"] == [str(tmp_path / "unsupported.pdf")]
    (tmp_path / "new.md").write_text("New file needs no enable action")
    assert len(snapshot_configured_skills(cfg, ["example_agent"])["skill_snapshots"]) == 3


def test_snapshots_survive_source_edits_deletion_and_new_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    root = tmp_path / "skills"
    root.mkdir()
    source = root / "context.md"
    source.write_text("Original harness behavior")
    selected = tmp_path / "contract.md"
    selected.write_text("Harness knob: distinct validation errors")
    run = RunConfig.create([], "test", [],
                           Config(skills_dir=str(root), skill_paths=[str(selected)]))
    source.write_text("Changed harness behavior")
    (root / "new.md").write_text("New context")
    expected = guidance_prompt(run)
    source.unlink()
    selected.unlink()
    restored = RunConfig.load(run.run_id)
    assert guidance_prompt(restored) == expected
    assert "Original harness behavior" in expected
    assert "distinct validation errors" in expected
    assert "Changed harness behavior" not in expected
    assert "New context" not in expected


def test_excluded_bad_file_does_not_block_but_active_bad_file_fails(tmp_path):
    (tmp_path / "bad.md").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        snapshot_configured_skills(Config(skills_dir=str(tmp_path)))
    cfg = Config(skills_dir=str(tmp_path), disabled_skill_paths=["bad.md"])
    assert snapshot_configured_skills(cfg)["skill_snapshots"] == []


def test_missing_folder_and_unscanned_symlink_directory_fail_visibly(tmp_path):
    with pytest.raises(ValueError, match="Skills folder is missing"):
        skill_inventory(Config(skills_dir=str(tmp_path / "missing")))
    (tmp_path / "cycle").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked skills directory"):
        skill_inventory(Config(skills_dir=str(tmp_path)))


def test_traversal_errors_fail_instead_of_silently_omitting_content(tmp_path, monkeypatch):
    def fail_scan(root, onerror):
        onerror(PermissionError("unreadable subtree"))

    monkeypatch.setattr("autofa.session.analysis_skills.os.walk", fail_scan)
    with pytest.raises(ValueError, match="Cannot scan skills folder"):
        skill_inventory(Config(skills_dir=str(tmp_path)))


def test_cli_reports_loaded_excluded_and_unsupported_files(tmp_path, capsys):
    (tmp_path / "loaded.md").write_text("Context")
    (tmp_path / "disabled.md").write_text("Excluded")
    (tmp_path / "notes.txt").write_text("Not Markdown")
    cfg = Config(skills_dir=str(tmp_path), disabled_skill_paths=["disabled.md"])
    run = RunConfig("run", [], "model", **snapshot_configured_skills(cfg))
    _show_analysis_guidance(run)
    output = capsys.readouterr().out
    assert "1 active" in output
    assert "Skill: loaded.md" in output
    assert "Skill excluded by user" in output
    assert "not loaded (not Markdown)" in output


def test_installed_autofa_uses_local_folder_then_home(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "__file__",
                        str(tmp_path / "installed" / "autofa" / "config.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    assert skills_directory() == tmp_path / "home" / "skills"
    local = tmp_path / "skills"
    local.mkdir()
    assert skills_directory() == local
    custom = tmp_path / "custom"
    custom.mkdir()
    assert skills_directory(Config(skills_dir=str(custom))) == custom


def test_select_folder_autosaves_and_enable_all_clears_exclusions(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    root = tmp_path / "skills"
    root.mkdir()
    (root / "notes.md").write_text("Notes")
    cfg = Config(skills_dir=str(tmp_path / "missing"))
    save_config(cfg)
    remaining = iter(["folder", str(root / "notes.md"), "all", "back"])
    monkeypatch.setattr(settings, "browse_filesystem", lambda *args, **kwargs: root)
    monkeypatch.setattr(settings, "select_option", lambda *args, **kwargs:
                        SimpleNamespace(value=next(remaining)))
    settings._skills_settings(cfg, persistent=True)
    restored = load_config()
    assert restored.skills_dir == str(root)
    assert restored.disabled_skill_paths == []
    assert skill_inventory(restored)["active"] == [str(root / "notes.md")]
