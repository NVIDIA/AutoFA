# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The portable driver keeps durable campaign records, caches and snapshots honest."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "agent-skills" / "autofa-driver"
SPEC = importlib.util.spec_from_file_location(
    "driver_skill_helper", SKILL / "scripts" / "driver.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_helper_runs_bookkeeping_on_python39(tmp_path):
    configured = os.environ.get("AUTOFA_DRIVER_TEST_PYTHON39")
    candidates = [configured] if configured else [shutil.which("python3.9"), "/usr/bin/python3"]
    interpreter = None
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            version = subprocess.run([candidate, "-c", "import sys; print(sys.version[:3])"],
                                     capture_output=True, text=True, check=True)
            if version.stdout.strip() == "3.9":
                interpreter = candidate
                break
    if interpreter is None:
        if configured:
            pytest.fail("AUTOFA_DRIVER_TEST_PYTHON39 must point to Python 3.9")
        pytest.skip("Python 3.9 unavailable; CI supplies AUTOFA_DRIVER_TEST_PYTHON39")
    goal = tmp_path / "goal.json"
    goal.write_text(json.dumps({"objective": "Inspect saved findings", "workflow": "report"}))
    command = [interpreter, str(SKILL / "scripts/driver.py")]
    campaign = str(tmp_path / "campaign")
    created = subprocess.run([*command, "init", "--campaign", campaign, "--goal", str(goal)],
                             capture_output=True, text=True, check=True)
    state = json.loads(created.stdout)
    assert state["goal"]["objective"] == "Inspect saved findings"
    shown = subprocess.run([*command, "show", "--campaign", campaign],
                           capture_output=True, text=True, check=True)
    assert json.loads(shown.stdout) == state
    # The supervisor service also loads on the portable interpreter.
    subprocess.run([interpreter, str(SKILL / "scripts/supervisor.py"), "--help"],
                   capture_output=True, check=True)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    goal = write(tmp_path / "goal.json", {
        "objective": "Classify the supplied runs", "workflow": "classification",
        "budgets": {"experiments": 2}, "timezone": "Europe/Zurich",
    })
    driver.state_action(SimpleNamespace(command="init", campaign=root, goal=goal))
    return root


def test_goal_revisions_do_not_clear_stop_or_reinterpret_workflow(campaign, tmp_path):
    before = driver.read(campaign / "campaign.json")["revision"]
    stopped = driver.state_action(SimpleNamespace(command="stop", campaign=campaign,
                                                  reason="User requested saved results only"))
    assert stopped["revision"] > before  # Stop and reopen change the authorization revision.
    patch = write(tmp_path / "revision.json", {"budgets": {"rounds_added": 14}})
    state = driver.state_action(SimpleNamespace(command="revise", campaign=campaign,
                                                patch=patch, reason="User revised future scope"))
    assert state["state"] == "stopped"
    assert state["goal"]["workflow"] == "classification"
    assert state["goal"]["budgets"] == {"experiments": 2, "rounds_added": 14}
    assert state["history"][-1]["before"]["budgets"] == {"experiments": 2}
    revised = state["revision"]
    state = driver.state_action(SimpleNamespace(command="reopen", campaign=campaign,
                                                reason="User explicitly resumed"))
    assert state["state"] == "active"
    assert state["history"][-1]["before"] == "stopped"
    assert state["revision"] > revised  # Reopen itself fences out pre-stop approvals.


def test_invalid_revision_leaves_record_intact(campaign, tmp_path):
    original = (campaign / "campaign.json").read_bytes()
    patch = write(tmp_path / "bad.json", {"objective": ""})
    with pytest.raises(ValueError, match="objective"):
        driver.state_action(SimpleNamespace(command="revise", campaign=campaign,
                                            patch=patch, reason="Invalid input"))
    assert (campaign / "campaign.json").read_bytes() == original
    with pytest.raises(ValueError, match="outside"):
        driver.campaign_path(SKILL / "state")


def test_shared_spending_allocations_do_not_multiply_campaign_budget(campaign, tmp_path):
    patch = write(tmp_path / "allocations.json", {"budgets": {"spend_limit_usd": 100,
        "run_allocations_usd": {"first": 60, "second": 60}}})
    with pytest.raises(ValueError, match="exceed"):
        driver.state_action(SimpleNamespace(command="revise", campaign=campaign,
                                            patch=patch, reason="Invalid allocation"))
    assert driver.read(campaign / "campaign.json")["goal"]["budgets"] == {"experiments": 2}


def test_preferences_are_explicit_and_separate_from_campaigns(campaign, tmp_path):
    path = tmp_path / "preferences/settings.json"
    settings = write(tmp_path / "selection.json", {
        "discovery": {"escalation": ["baseline", "medium"], "api_key_env": "PROVIDER_KEY"}})
    original = (campaign / "campaign.json").read_bytes()
    output = driver.preferences(SimpleNamespace(path=path, settings_file=settings,
                                                reason="User selected defaults for future work"))
    assert output["preferences"]["discovery"]["escalation"] == ["baseline", "medium"]
    assert (campaign / "campaign.json").read_bytes() == original
    settings.write_text('{"discovery":{"api_key":"do-not-store"}}')
    with pytest.raises(ValueError, match="secret"):
        driver.preferences(SimpleNamespace(path=path, settings_file=settings,
                                            reason="Incorrect credential storage"))


@pytest.fixture
def cached_runtime(campaign, tmp_path):
    source = tmp_path / "release"
    source.mkdir()
    native = source / "registration.py"
    native.write_text("WORKFLOWS = {'new-future-workflow': 'inspect'}\n")
    marker = tmp_path / "must-not-exist"
    executable = tmp_path / "autofa"
    executable.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    executable.chmod(0o755)
    config = write(tmp_path / "config.json", {"provider": "development"})
    identity = driver.identify(str(executable), source, [str(config)])
    assert not marker.exists()
    identity_path = write(tmp_path / "identity.json", identity)
    evidence = tmp_path / "help.txt"
    evidence.write_text("future-workflow accepts a source and emits assignments\n")
    mapping = write(tmp_path / "map.json", {"capabilities": [{
        "name": "new-future-workflow", "surface": "CLI",
        "entrypoint": [str(executable), "future-workflow"],
        "purpose": "A newly added analysis", "evidence": [str(evidence)],
        "completion": {"artifact": "assignments.json"},
    }]})
    driver.cache_action(SimpleNamespace(command="cache-put", campaign=campaign,
                                       identity=identity_path, map=mapping))
    return SimpleNamespace(source=source, native=native, config=config, evidence=evidence,
                           executable=executable, marker=marker)


def test_new_workflow_cache_is_evidence_bound_without_dispatch(campaign, cached_runtime):
    cached = driver.cache_action(SimpleNamespace(command="cache-get", campaign=campaign))
    assert cached["map"]["capabilities"][0]["name"] == "new-future-workflow"
    assert not cached_runtime.marker.exists()
    cached_runtime.evidence.write_text("same command, changed lifecycle contract\n")
    with pytest.raises(ValueError, match="evidence changed"):
        driver.cache_action(SimpleNamespace(command="cache-get", campaign=campaign))


@pytest.mark.parametrize("changed", ["native", "config", "executable"])
def test_cache_invalidates_on_content_not_version_label(campaign, cached_runtime, changed):
    path = getattr(cached_runtime, changed)
    path.write_text(path.read_text() + "\nchanged\n")
    with pytest.raises(ValueError, match="runtime/source/config changed"):
        driver.cache_action(SimpleNamespace(command="cache-get", campaign=campaign))
    assert not cached_runtime.marker.exists()


def test_executable_symlink_release_change_invalidates_identity(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "cli.py").write_text("# same version\n")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_text("first")
    second.write_text("second")
    link = tmp_path / "autofa"
    link.symlink_to(first)
    before = driver.identify(str(link), package)
    link.unlink()
    link.symlink_to(second)
    assert driver.refresh_identity(before)["fingerprint"] != before["fingerprint"]


def test_receipts_are_idempotent_and_preserve_decisions(campaign, tmp_path):
    # A Unicode line separator inside a JSON line must not split the ledger record.
    reason = "Test a causal\u2028hypothesis"
    path = write(tmp_path / "receipt.json", {
        "id": "candidate-01", "state": "proposed", "reason": reason,
        "source_hash": "baseline", "exposed_examples": ["source-a/item-7"],
    })
    args = SimpleNamespace(campaign=campaign, kind="intervention", receipt=path)
    assert driver.record(args)["recorded"]
    assert not driver.record(args)["recorded"]
    receipt = driver.read(path)
    receipt["source_hash"] = "changed"
    write(path, receipt)
    with pytest.raises(ValueError, match="Conflicting"):
        driver.record(args)
    receipt.update(state="rejected", reason="Retained positive regressed")
    write(path, receipt)
    assert driver.record(args)["recorded"]
    lines = (campaign / "interventions.jsonl").read_bytes().split(b"\n")
    rows = [json.loads(line) for line in lines if line]
    assert len(rows) == 2
    assert rows[0]["receipt"]["source_hash"] == "baseline"
    assert rows[0]["receipt"]["reason"] == reason
    assert rows[1]["receipt"]["state"] == "rejected"


def test_observational_snapshot_handles_same_names_and_verifies_bytes(campaign, tmp_path):
    files = []
    for label in ("a", "b"):
        folder = tmp_path / label
        folder.mkdir()
        files.append(write(folder / "status.json", {"run": label}))
    result = driver.snapshot(SimpleNamespace(campaign=campaign, label="progress", file=files))
    assert result["consistent_native_checkpoint"] is False
    manifest = Path(result["path"]) / "manifest.json"
    assert driver.verify(manifest)["artifacts"] == 2
    copied = Path(result["path"]) / result["artifacts"][0]["path"]
    copied.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        driver.verify(manifest)


@pytest.mark.parametrize("name, content", [
    ("predictions.sqlite", b"anything"), ("database", b"SQLite format 3\0bytes"),
    ("predictions-wal", b"committed WAL"),
])
def test_snapshot_refuses_inconsistent_database_copies(campaign, tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(ValueError, match="native locked SQLite"):
        driver.snapshot(SimpleNamespace(campaign=campaign, label="bad", file=[path]))
    assert not list((campaign / "snapshots").iterdir())


def test_snapshot_does_not_publish_a_torn_copy(campaign, tmp_path, monkeypatch):
    path = write(tmp_path / "moving.json", {"round": 1})
    original = driver.shutil.copyfile

    def racing_copy(source, destination):
        original(source, destination)
        source.write_text('{"round":2}')

    monkeypatch.setattr(driver.shutil, "copyfile", racing_copy)
    with pytest.raises(ValueError, match="changed during snapshot"):
        driver.snapshot(SimpleNamespace(campaign=campaign, label="moving", file=[path]))
    assert not list((campaign / "snapshots").iterdir())


def test_cli_verification_reports_failure_without_claiming_success(tmp_path, capsys):
    manifest = write(tmp_path / "manifest.json", {"artifacts": []})
    assert driver.main(["verify", "--manifest", str(manifest)]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert "error" in json.loads(output.err)


def test_driver_package_stays_out_of_native_analysis_context(tmp_path, monkeypatch):
    from autofa.config import Config
    from autofa.session.analysis_skills import snapshot_configured_skills

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    snapshot = snapshot_configured_skills(Config(skills_dir=str(ROOT / "skills")), ["codex"])
    assert snapshot["skill_snapshots"]
    for item in snapshot["skill_snapshots"]:
        assert not Path(item["path"]).is_relative_to(SKILL)
        assert "# AutoFA driver" not in item["text"]


def test_torn_receipt_tail_is_preserved_and_recovered(campaign, tmp_path):
    path = write(tmp_path / 'receipt.json', {'id': 'one', 'state': 'saved', 'reason': 'first'})
    args = SimpleNamespace(campaign=campaign, kind='intervention', receipt=path)
    driver.record(args)
    ledger = campaign / 'interventions.jsonl'
    with ledger.open('ab') as stream:
        stream.write(b'{"receipt": "torn\xe2')
    write(path, {'id': 'two', 'state': 'saved', 'reason': 'second'})
    assert driver.record(args)['recorded']
    assert len([json.loads(line) for line in ledger.read_bytes().split(b'\n') if line]) == 2
    assert next(campaign.glob('*.torn-*')).read_bytes() == b'{"receipt": "torn\xe2'
