# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation preparation is isolated and never executes a model or seeds a detector."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "autofa_autonomy_eval", Path(__file__).parents[1] / "tools" / "autofa_autonomy_eval.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
# Settings the tests pass or prepare() adds; the Config test double declares exactly these.
CONFIG_FIELDS = (
    "max_rounds", "classifier_profiles", "objective_axes", "objective_weights", "objective_rank",
    "min_failure_modes", "discovery_wait_for_plateau", "skill_paths", "analysis_outputs",
    "preflight_detectors", "exploration_seed", "feature_api_key_env", "judge_api_key_env",
    "remote_library_token_env", "feature_request_options",
)


@pytest.fixture
def prepared(tmp_path):
    roots = []
    for name in ("baseline", "updated"):
        root = tmp_path / name
        package = root / "src" / "autofa"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "config.py").write_text(
            "from dataclasses import dataclass\n"
            "@dataclass(init=False)\nclass Config:\n"
            + "".join(f"    {field}: object = None\n" for field in CONFIG_FIELDS)
            + "    def __init__(self, **kwargs): pass\n")
        (package / "run_config.py").write_text(
            "class Options:\n    def __init__(self, *args, **kwargs): pass\n"
            "AutoFAConfig = AutoFAExecution = SourceSpec = Options\n")
        (package / "cli.py").write_text(
            "from pathlib import Path\nfrom types import SimpleNamespace\n"
            "def run_autofa(config):\n"
            "    return SimpleNamespace(run_id='synthetic', run_dir=lambda: Path.cwd() / 'run')\n")
        roots.append(root)
    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset.parquet").write_text("synthetic fixture, no inference")
    settings = {"max_rounds": 2, "classifier_profiles": [],
                "objective_axes": ["ppv_at_pi0", "task_recall", "behavior_purity"],
                "objective_weights": [20, 50, 30], "objective_rank": "geometric_mean"}
    return evaluation.prepare(tmp_path / "comparison", baseline=roots[0], updated=roots[1],
                              source=source, settings=settings, seeds=[1, 2], timeout_seconds=30,
                              mode="controlled")


def test_preparation_has_no_results_or_shared_state(prepared):
    manifest = json.loads(prepared.read_text())
    assert len(manifest["arms"]) == 4
    assert manifest["scope"] == "unassisted_raw_data"
    for arm in manifest["arms"]:
        directory = Path(arm["directory"])
        assert list(directory.iterdir()) == [directory / "settings.json"]
        assert not (directory / "home").exists()
        assert json.loads((directory / "settings.json").read_text())["skill_paths"] == []


def test_execute_native_test_double_once_and_record_scoped_result(prepared):
    result = evaluation.execute(prepared, python=sys.executable, arm_name="updated", seed=1)
    assert result["status"] == "completed"
    arm = prepared.parent / "updated-1"
    native = json.loads((arm / "native-result.json").read_text())
    assert native["run_id"] == "synthetic" and native["harness_candidate_injections"] == 0
    assert native["external_interventions"] == "not_audited"
    with pytest.raises(FileExistsError):
        evaluation.execute(prepared, python=sys.executable, arm_name="updated", seed=1)
    assert not (prepared.parent / "baseline-1" / "execution.json").exists()


@pytest.mark.parametrize("payload", [None, "{}"], ids=["missing", "incomplete"])
def test_successful_exit_without_valid_native_result_is_not_completion(
    prepared, monkeypatch, payload,
):
    directory = prepared.parent / "updated-1"
    if payload is not None:
        (directory / "native-result.json").write_text(payload)
    monkeypatch.setattr(evaluation.subprocess, "Popen",
                        lambda *args, **kwargs: SimpleNamespace(wait=lambda **kwargs: 0))
    result = evaluation.execute(prepared, python=sys.executable, arm_name="updated", seed=1)
    assert result["status"] == "missing_result"
    assert result["returncode"] == 0
    assert result["result_error"]
    assert json.loads((directory / "execution.json").read_text()) == result


@pytest.mark.parametrize("changed", ["source", "code", "settings"])
def test_changed_experiment_inputs_are_rejected_before_launch(prepared, changed):
    manifest = json.loads(prepared.read_text())
    arm = next(a for a in manifest["arms"] if a["name"] == "updated" and a["seed"] == 1)
    if changed == "source":
        target = Path(manifest["source"]) / "dataset.parquet"
    elif changed == "code":
        target = Path(arm["checkout"]) / "src" / "autofa" / "__init__.py"
    else:
        target = Path(arm["directory"]) / "settings.json"
    target.write_text('{}' if changed == "settings" else "changed")
    with pytest.raises(ValueError, match="changed"):
        evaluation.execute(prepared, python=sys.executable, arm_name="updated", seed=1)
    assert not (Path(arm["directory"]) / "execution.json").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_timeout_or_user_interrupt_stops_owned_process_and_keeps_receipt(
    prepared, monkeypatch, interrupt,
):
    calls = []

    class Process:
        pid = 12345

        def wait(self, *, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                if interrupt:
                    raise KeyboardInterrupt
                raise evaluation.subprocess.TimeoutExpired("native", timeout)
            return 130

    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *a, **kw: Process())
    signals = []
    monkeypatch.setattr(evaluation.os, "killpg", lambda *args: signals.append(args))
    result = evaluation.execute(prepared, python=sys.executable, arm_name="baseline", seed=1)
    assert result["status"] == ("interrupted" if interrupt else "timed_out")
    assert signals == [(12345, evaluation.signal.SIGINT)]
    assert (prepared.parent / "baseline-1" / "execution.json").is_file()


@pytest.mark.parametrize("credentials", [
    {"api_key": "private-credential"},
    {"classifier_profiles": [{"name": "approved", "feature_model": "approved",
                              "feature_request_options": {"extra_body": {"vendor": {
                                  "nested": [{"refresh_token": "private-credential"}]}}}}]},
    {"judge_request_options": {"extra_body": {"vendor": {
        "extra_headers": {"x-private": "private-credential"}}}}},
])
def test_literal_credentials_rejected_before_any_preparation_write(tmp_path, credentials):
    settings = {"max_rounds": 2, "classifier_profiles": [], **credentials}
    output = tmp_path / "comparison"
    # Intentionally absent runtime/source paths: credential rejection precedes any
    # scanning or creation, including duplicate settings and effective-settings records.
    with pytest.raises(ValueError, match="Literal credentials") as failure:
        evaluation.prepare(output, baseline=tmp_path / "baseline", updated=tmp_path / "updated",
                           source=tmp_path / "source", settings=settings, seeds=[1],
                           timeout_seconds=30, mode="default-experience")
    assert "private-credential" not in str(failure.value)
    assert not output.exists()


def test_allowed_provider_names_and_generation_controls_survive_preparation(prepared):
    manifest = json.loads(prepared.read_text())
    checkouts = {arm["name"]: Path(arm["checkout"]) for arm in manifest["arms"]}
    settings = {**manifest["settings"], "feature_api_key_env": "PERMITTED_PROVIDER_KEY",
                "judge_api_key_env": "PERMITTED_JUDGE_KEY", "remote_library_token_env": "LIB_KEY",
                "feature_request_options": {"max_tokens": 512, "reasoning_effort": "medium"},
                "classifier_profiles": [{"name": "allowed", "feature_model": "allowed-model",
                                         "feature_api_key_env": "OTHER_PROVIDER_KEY"}]}
    path = evaluation.prepare(prepared.parent / "another", baseline=checkouts["baseline"],
                              updated=checkouts["updated"], source=Path(manifest["source"]),
                              settings=settings, seeds=[1], timeout_seconds=30, mode="controlled")
    assert json.loads(path.read_text())["settings"] == settings


def test_settings_outside_either_config_are_rejected_before_any_arm(prepared):
    manifest = json.loads(prepared.read_text())
    checkouts = {arm["name"]: Path(arm["checkout"]) for arm in manifest["arms"]}
    # Only the baseline still declares the retired field; the updated arm would fail on it.
    baseline_config = checkouts["baseline"] / "src" / "autofa" / "config.py"
    baseline_config.write_text(baseline_config.read_text().replace(
        "    def __init__", "    objective_primary: object = None\n    def __init__"))
    output = prepared.parent / "another"
    with pytest.raises(ValueError, match="not Config fields in both checkouts: objective_primary"):
        evaluation.prepare(output, baseline=checkouts["baseline"], updated=checkouts["updated"],
                           source=Path(manifest["source"]),
                           settings={**manifest["settings"], "objective_primary": "ppv_at_pi0"},
                           seeds=[1], timeout_seconds=30, mode="controlled")
    assert not output.exists()


# Each failure before a child starts has its own unlock: a corrected rerun is not a reroll.
@pytest.mark.parametrize("failure", ["process_launch", "log_open"])
def test_failed_start_does_not_lock_unstarted_arm(prepared, failure):
    directory = prepared.parent / "updated-1"
    log = directory / "native.log"
    if failure == "log_open":
        log.mkdir()
        python, error = sys.executable, IsADirectoryError
    else:
        python, error = "/missing/autofa-python", FileNotFoundError
    with pytest.raises(error):
        evaluation.execute(prepared, python=python, arm_name="updated", seed=1)
    assert not (directory / "execution.json").exists()
    if failure == "log_open":
        log.rmdir()
    assert evaluation.execute(prepared, python=sys.executable, arm_name="updated", seed=1)[
        "status"] == "completed"
