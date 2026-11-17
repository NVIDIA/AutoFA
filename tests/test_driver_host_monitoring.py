# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The host-monitored driver fences launches, reviews and delivery to the acknowledged host."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agent-skills/autofa-driver/scripts"
SPEC = importlib.util.spec_from_file_location("host_test_supervisor", SCRIPTS / "supervisor.py")
supervision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervision)
host = supervision.host_controller


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "campaign"
    goal = write(tmp_path / "goal.json", {
        "objective": "Find supported failures and deliver a verified report",
        "delivery": {"destination": "local"},
    })
    supervision.records.state_action(SimpleNamespace(command="init", campaign=root, goal=goal))
    config = {
        "supervision_mode": "host", "native": {"argv": [sys.executable], "cwd": str(tmp_path)},
        "host": {"kind": "claude", "executable": sys.executable, "session_id": "outer-session"},
        "runs": [], "host_grace_seconds": 30, "review_seconds": 10, "poll_seconds": 1,
        "allowed_actions": ["steer", "update"],
    }
    write(root / "supervision/config.json", config)
    guard = supervision.Supervisor(root)
    guard.acquire()
    contract = write(tmp_path / "contract.json", {"execution": {"publish": False}})
    cfg = tmp_path / "autofa.toml"
    cfg.write_text("max_rounds = 3\n")
    fake = tmp_path / "fake.py"
    fake.write_text("raise SystemExit(7)\n")
    spec = {"launch_id": "a" * 32, "argv": [sys.executable, str(fake)], "cwd": str(tmp_path),
            "env": {"AUTOFA_CONFIG": str(cfg), "AUTOFA_SETUP_CONTRACT": str(contract)},
            "config_files": {str(path): host.records.digest(path) for path in (cfg, contract)}}
    yield SimpleNamespace(root=root, config=config, guard=guard, generation=guard.generation,
                          spec=spec, fake=fake, tmp=tmp_path)
    guard._publication_pool.shutdown(wait=True)
    guard._delivery_pool.shutdown(wait=True)


def ack(setup, *, assessment=None):
    observation = host.observe(setup.root, setup.generation)
    return host.acknowledge(setup.root, setup.generation, observation["observation_id"],
                            session_id="outer-session", task_id="foreground-wait",
                            assessment=assessment)


def launch_receipt(setup, receipt: dict) -> Path:
    return setup.root / "supervision/launches" / receipt["launch_id"] / "receipt.json"


def wait_owner(setup, pid: int) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        owner = host.document(setup.root / "supervision/owner.json")
        if owner.get("pid") == pid:
            break
        time.sleep(.05)
    assert owner["pid"] == pid
    return owner


def bound_run(setup) -> Path:
    """Bind a native run to a campaign that pauses when its host is lost."""
    setup.config["on_host_loss"] = "pause"
    write(setup.root / "supervision/config.json", setup.config)
    run = setup.tmp / "bound"
    write(run / "supervision/driver_launch.json", {"campaign_path": str(setup.root),
          "campaign_id": host.document(setup.root / "campaign.json")["id"], "launch_id": "b" * 32})
    return run


def wait_receipt(path: Path, state: set[str]) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = host.document(path)
        if result["state"] in state:
            return result
        time.sleep(.05)
    pytest.fail(f"Managed launch did not reach {state}: {result}")


def test_non_posix_liveness_does_not_signal_launch_worker(monkeypatch):
    def kill(*args):
        pytest.fail("A liveness check must not signal a worker on non-POSIX systems")

    monkeypatch.setattr(host, "os", SimpleNamespace(name="nt", kill=kill))
    identity = {"hostname": host.socket.gethostname(), "pid": 123, "started_epoch": 456}
    assert host.process_alive(identity) is None


@pytest.mark.parametrize(("started", "expected"), [(456, True), (789, False)])
def test_posix_liveness_checks_process_birth_identity(monkeypatch, started, expected):
    signaled = []
    monkeypatch.setattr(host, "os", SimpleNamespace(
        name="posix", kill=lambda pid, sig: signaled.append((pid, sig)),
    ))
    monkeypatch.setattr(host, "process_identity", lambda pid: {"started_epoch": started})
    identity = {"hostname": host.socket.gethostname(), "pid": 123, "started_epoch": 456}
    assert host.process_alive(identity) is expected
    assert signaled == [(123, 0)]


def test_launch_requires_consumed_host_observation_and_fenced_session(setup):
    with pytest.raises(ValueError, match="acknowledgment"):
        host.launch(setup.root, setup.generation, setup.spec)
    observation = host.observe(setup.root, setup.generation)
    with pytest.raises(ValueError, match="different host"):
        host.acknowledge(setup.root, setup.generation, observation["observation_id"],
                         session_id="another-session", task_id="task")
    ack(setup)
    assert host.host_available(setup.root, setup.generation)
    assert not (setup.root / "supervision/launches").exists()
    setup.guard.acquire()
    with pytest.raises(ValueError, match="owner changed"):
        host.launch(setup.root, setup.generation, setup.spec)


def test_host_review_no_secondary_model_and_heartbeat_does_not_postpone_review(setup):
    receipt = ack(setup, assessment="No native work yet; preparation remains supervised.")
    setup.guard.tick()
    assert setup.guard.worker is None
    receipt2 = ack(setup)
    assert receipt2["next_review_at_epoch"] == receipt["next_review_at_epoch"]
    assert host.snapshot(setup.root, setup.generation,
                         now=receipt["next_review_at_epoch"] + 1)["review_due"]


def test_real_preparation_failure_reported_before_any_run_exists(setup):
    ack(setup)
    receipt = host.launch(setup.root, setup.generation, setup.spec)
    done = wait_receipt(launch_receipt(setup, receipt), {"exited", "launch_failed"})
    assert done["exit_code"] == 7
    assert done["failure"]["preparation_failed"] is True
    setup.guard.observe()
    result = host.observe(setup.root, setup.generation)
    assert result["runs"] == {}
    assert any(event["kind"] == "launch" and event["exit_code"] == 7 for event in result["events"])
    assert result["launches"][0]["log_path"]
    # Retrying the same request inspects the receipt instead of launching again.
    assert host.launch(setup.root, setup.generation, setup.spec)["worker"] == done["worker"]


def test_native_binding_uses_launch_id_and_survives_guard_takeover(setup):
    setup.fake.write_text('''import os
from pathlib import Path
from autofa.session.managed_launch import bind_run, preparation_event
run = Path(os.environ["AUTOFA_CONFIG"]).parent / "actual-run"
run.mkdir()
(run / "config.json").write_text("{}")
preparation_event("converting_sources")
bind_run(run)
''')
    ack(setup)
    receipt = host.launch(setup.root, setup.generation, setup.spec)
    done = wait_receipt(launch_receipt(setup, receipt), {"exited", "launch_failed"})
    assert done["exit_code"] == 0, Path(done["log_path"]).read_text()
    assert done["run_ids"] == ["actual-run"]
    unrelated = setup.tmp / "newest-unrelated-run"
    unrelated.mkdir()
    events, bindings = host.launch_observations(setup.root)
    assert [row["run_id"] for row in bindings] == ["actual-run"]
    setup.guard.acquire()
    setup.generation = setup.guard.generation
    ack(setup)
    _, reattached = host.launch_observations(setup.root)
    assert reattached == bindings


def test_changed_contract_and_unagreed_publication_are_rejected_before_spawn(setup):
    ack(setup)
    path = Path(setup.spec["env"]["AUTOFA_SETUP_CONTRACT"])
    write(path, {"execution": {"publish": True}})
    with pytest.raises(ValueError, match="configuration changed"):
        host.launch(setup.root, setup.generation, setup.spec)
    setup.spec["config_files"][str(path)] = host.records.digest(path)
    with pytest.raises(ValueError, match="publication differs"):
        host.launch(setup.root, setup.generation, setup.spec)
    assert not (setup.root / "supervision/launches").exists()


@pytest.mark.parametrize("url", ["", "http://localhost:8080", "http://[::1]:8080",
                                 "http://0.0.0.0:8080"])
def test_cloud_destination_rejects_missing_or_loopback_library(setup, url):
    campaign = host.document(setup.root / "campaign.json")
    campaign["goal"]["delivery"]["destination"] = "cloud"
    write(setup.root / "campaign.json", campaign)
    ack(setup)
    path = Path(setup.spec["env"]["AUTOFA_SETUP_CONTRACT"])
    write(path, {"execution": {"publish": True}, "settings": {"remote_library_base_url": url}})
    setup.spec["config_files"][str(path)] = host.records.digest(path)
    with pytest.raises(ValueError, match="non-loopback"):
        host.launch(setup.root, setup.generation, setup.spec)


def test_user_stop_terminates_owned_preparation_without_native_run(setup):
    setup.fake.write_text("import time\ntime.sleep(30)\n")
    ack(setup)
    path = launch_receipt(setup, host.launch(setup.root, setup.generation, setup.spec))
    wait_receipt(path, {"running"})
    supervision.records.state_action(SimpleNamespace(command="stop", campaign=setup.root,
                                                     reason="User requested stop"))
    done = wait_receipt(path, {"exited"})
    assert done["exit_code"] != 0
    assert done["run_ids"] == []


def test_native_admission_pauses_on_lost_host_even_when_guard_is_gone(setup):
    from autofa.session.managed_launch import enforce_supervision
    from autofa.session.supervision import SupervisorInterrupted

    run = bound_run(setup)
    ack(setup)
    enforce_supervision(run)
    saved = host.document(setup.root / "supervision/host-ack.json")
    saved["acknowledged_at_epoch"] -= 120
    write(setup.root / "supervision/host-ack.json", saved)
    with pytest.raises(SupervisorInterrupted):
        enforce_supervision(run)
    assert host.document(run / "supervision/lifecycle.json")["action"] == "pause"


def test_concurrent_user_stop_wins_over_host_loss_pause(setup, monkeypatch):
    from autofa.session import supervision as native
    from autofa.session.managed_launch import enforce_supervision

    run = bound_run(setup)
    original = native.lifecycle_request

    def racing_request(directory, action, **kwargs):
        original(directory, "stop")
        return original(directory, action, **kwargs)

    monkeypatch.setattr(native, "lifecycle_request", racing_request)
    with pytest.raises(native.SupervisorInterrupted) as raised:
        enforce_supervision(run)
    assert raised.value.action == "stop"
    assert host.document(run / "supervision/lifecycle.json")["action"] == "stop"
    # Saved-only publication is permitted; reconnect never erases the user's stop.
    enforce_supervision(run, phase="publication")
    ack(setup)
    assert host.document(run / "supervision/lifecycle.json")["action"] == "stop"


def test_summary_ack_is_archived_and_bound_to_delivered_hash(setup):
    observation = host.observe(setup.root, setup.generation)
    receipt = host.acknowledge(setup.root, setup.generation, observation["observation_id"],
                               session_id="outer-session", task_id="task", summary_sha256="d" * 64)
    ack(setup)
    archived = host.document(setup.root / "supervision/host-acks" / (receipt["id"] + ".json"))
    assert archived["summary_sha256"] == "d" * 64


def test_new_ack_cannot_upgrade_fence_for_a_delayed_old_action(setup):
    first = ack(setup)
    old_fence = host.document(Path(first["fence_file"]))
    campaign = host.document(setup.root / "campaign.json")
    campaign["revision"] += 1
    write(setup.root / "campaign.json", campaign)
    second = ack(setup)
    assert first["fence_file"] != second["fence_file"]
    assert host.document(Path(first["fence_file"])) == old_fence
    assert host.document(Path(second["fence_file"]))["revision"] == campaign["revision"]


def test_portable_host_sources_parse_with_python39_grammar():
    for filename in ("host_controller.py", "managed_launch.py", "supervisor.py"):
        ast.parse((SCRIPTS / filename).read_text(), feature_version=(3, 9))


def test_python39_service_cli_watches_real_pre_run_failure_and_rejects_second_owner(setup):
    python = shutil.which("python3.9")
    if python is None:
        pytest.skip("Python 3.9 is not installed")
    command = [python, str(SCRIPTS / "supervisor.py")]
    owner = subprocess.Popen([*command, "run", "--campaign", str(setup.root)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        generation = wait_owner(setup, owner.pid)["generation"]

        def cli(name, *arguments):
            result = subprocess.run([*command, name, "--campaign", str(setup.root),
                                     "--generation", generation, *arguments],
                                    capture_output=True, text=True, timeout=10, check=False)
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)

        seen = cli("observe")
        cli("ack", "--observation-id", seen["observation_id"], "--session-id", "outer-session",
            "--task-id", "synthetic-host-task", "--assessment", "Ready to observe preparation")
        spec = write(setup.tmp / "launch.json", setup.spec)
        launched = cli("launch", "--spec-file", str(spec))
        assert wait_receipt(launch_receipt(setup, launched), {"exited"})["exit_code"] == 7
        seen = cli("watch", "--after", seen["fingerprint"], "--timeout", "2")
        assert seen["launches"][0]["failure"]["preparation_failed"]
        duplicate = subprocess.run([*command, "run", "--campaign", str(setup.root)],
                                   capture_output=True, timeout=5, check=False)
        assert duplicate.returncode == 2
        assert host.document(setup.root / "supervision/owner.json")["pid"] == owner.pid
    finally:
        owner.terminate()
        owner.communicate(timeout=5)


def test_source_diagnostics_retain_errno_without_secret_urls(tmp_path, monkeypatch):
    import errno

    from autofa.data.source_errors import SourceDataError
    from autofa.data.source_inputs import _source_file

    events = []
    monkeypatch.setattr("autofa.data.source_inputs.preparation_event",
                        lambda stage, **kwargs: events.append({"stage": stage, **kwargs}))

    class Api:
        timeout = 1
        max_attempts = 1

        def _get_json(self, *args):
            return {"size": 10}

        def _url(self, *args):
            return "https://example.test/SECRET?signature=PRIVATE"

        def _request_with_retries(self, operation, **kwargs):
            return operation()

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "SECRET https://signed.test/PRIVATE")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(SourceDataError) as raised:
        _source_file(Api(), "dataset", "raw", tmp_path)
    assert raised.value.diagnostic["errno"] == errno.ENOSPC
    assert raised.value.diagnostic["retryable"] is False
    assert "SECRET" not in str(raised.value) + json.dumps(events)
    assert "PRIVATE" not in str(raised.value) + json.dumps(events)
    assert events[-1]["stage"] == "source_preparation_failed"


def test_real_cli_can_acknowledge_authorized_saved_delivery_after_guard_stops(setup):
    campaign = host.document(setup.root / "campaign.json")
    campaign["goal"]["delivery"] = {"destination": "cloud", "publish_on_stop": True}
    write(setup.root / "campaign.json", campaign)
    setup.fake.write_text('''import json, sys
args = sys.argv[1:]
if args[:2] == ["supervise", "status"]:
 print(json.dumps({"run_id":"R1", "campaign":{"active":False},
  "snapshots":{"finalization_outcome":{"record":{"state":"completed"}}}}))
elif args[0] == "publish":
 print(json.dumps({"run_id":"R1", "status":"verified",
  "url":"https://library.example.test/libraries/L?version_id=V",
  "snapshot":{"library_sha256":"L", "report_sha256":"R"},
  "metrics":{"coverage":0.8}, "finalization":{"state":"completed"}}))
else: print(json.dumps({"ok":True}))
''')
    config = setup.config.copy()
    config["native"] = {"argv": [sys.executable, str(setup.fake)], "cwd": str(setup.tmp)}
    config["runs"] = [{"run_id": "R1"}]
    write(setup.root / "supervision/config.json", config)
    command = [sys.executable, str(SCRIPTS / "supervisor.py")]

    def cli(script, name, *arguments, success=True):
        result = subprocess.run([sys.executable, str(SCRIPTS / script), name,
                                 "--campaign", str(setup.root), *arguments],
                                capture_output=True, text=True, timeout=10, check=False)
        if success:
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        assert result.returncode != 0
        return result

    guard = subprocess.Popen([*command, "run", "--campaign", str(setup.root)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        generation = wait_owner(setup, guard.pid)["generation"]
        cli("driver.py", "stop", "--reason", "Stop native work; saved delivery is authorized")
        guard.communicate(timeout=5)
        stopped_owner = host.document(setup.root / "supervision/owner.json")
        assert stopped_owner["state"] == "stopped"
        cli("driver.py", "delivery", "reconcile", "--run", "R1", "--allow-publish")
        summary = write(setup.tmp / "summary.json", {
            "outcome": "complete",
            "text": "Findings https://library.example.test/libraries/L?version_id=V",
            "findings": ["Synthetic evidence"], "next_actions": [],
            "cost": {"status": "unknown", "reason": "Synthetic test"},
        })
        prepared = cli("driver.py", "delivery", "summary", "--summary", str(summary))
        observation = cli("supervisor.py", "observe", "--generation", generation)
        assert observation["delivery_only"] is True
        assert observation["supervision_available"] is False
        receipt = cli("supervisor.py", "ack", "--generation", generation,
                      "--observation-id", observation["observation_id"],
                      "--session-id", "outer-session", "--task-id", "final-message",
                      "--summary-sha256", prepared["sha256"])
        assert receipt["scope"] == "saved_delivery"
        assert "fence_file" not in receipt
        assert host.document(setup.root / "supervision/owner.json") == stopped_owner
        assert host.document(setup.root / "campaign.json")["state"] == "stopped"
        proof = cli("driver.py", "delivery", "check")
        assert proof["complete"] is True, proof
        cli("supervisor.py", "launch", "--generation", generation,
            "--spec-file", str(write(setup.tmp / "launch.json", setup.spec)), success=False)
        cli("driver.py", "complete", "--reason", "Verified saved findings delivered")
        assert host.document(setup.root / "supervision/owner.json") == stopped_owner
    finally:
        if guard.poll() is None:
            guard.terminate()
            guard.communicate(timeout=5)


def test_stopped_delivery_does_not_grant_new_lease_or_accept_unprepared_summary(setup):
    setup.guard.acquire()
    setup.generation = setup.guard.generation
    campaign = host.document(setup.root / "campaign.json")
    campaign["state"] = "stopped"
    write(setup.root / "campaign.json", campaign)
    with pytest.raises(ValueError, match="not authorized"):
        host.observe(setup.root, setup.generation)
    campaign["goal"]["delivery"]["publish_on_stop"] = True
    write(setup.root / "campaign.json", campaign)
    observation = host.observe(setup.root, setup.generation)
    with pytest.raises(ValueError, match="prepared summary"):
        host.acknowledge(setup.root, setup.generation, observation["observation_id"],
                         session_id="outer-session", task_id="task", summary_sha256="a" * 64)
    assert host.host_available(setup.root, setup.generation) is False


def test_bound_run_status_uses_prepared_profile_after_global_profile_changes(setup):
    setup.fake.write_text('''import json, os, sys
from pathlib import Path
if sys.argv[1] == "launch":
 from autofa.session.managed_launch import bind_run
 run = Path(__file__).parent / "frozen-run"
 run.mkdir(); (run / "config.json").write_text("{}")
 bind_run(run)
else:
 print(json.dumps({"run_id":"frozen-run", "config":os.environ.get("AUTOFA_CONFIG"),
  "home":os.environ.get("AUTOFA_HOME"), "campaign":{"active":False}}))
''')
    native = {"argv": [sys.executable, str(setup.fake)], "cwd": str(setup.tmp),
              "env": {"AUTOFA_CONFIG": str(setup.tmp / "old.toml"),
                      "AUTOFA_HOME": str(setup.tmp / "old-home")}}
    setup.guard.config["native"] = native
    write(setup.root / "supervision/config.json", setup.guard.config)
    setup.spec["argv"] = [*native["argv"], "launch"]
    setup.spec["env"]["AUTOFA_HOME"] = str(setup.tmp / "scoped-home")
    ack(setup)
    receipt = host.launch(setup.root, setup.generation, setup.spec)
    done = wait_receipt(launch_receipt(setup, receipt), {"exited"})
    assert done["exit_code"] == 0, Path(done["log_path"]).read_text()
    setup.guard.observe()
    setup.guard.config["native"]["env"]["AUTOFA_CONFIG"] = str(setup.tmp / "changed.toml")
    setup.guard.config["native"]["env"]["AUTOFA_HOME"] = str(setup.tmp / "changed-home")
    result = setup.guard.native(["supervise", "status", "--run", "frozen-run"])
    assert result["config"] == setup.spec["env"]["AUTOFA_CONFIG"]
    assert result["home"] == setup.spec["env"]["AUTOFA_HOME"]
    assert setup.guard.config["runs"][0]["native"]["env"]["AUTOFA_CONFIG"] == result["config"]


def test_stopped_guard_observes_until_saved_delivery_or_bounded_blocker(setup, monkeypatch):
    campaign = host.document(setup.root / "campaign.json")
    campaign.update(state="stopped", revision=1)
    campaign["goal"]["delivery"]["publish_on_stop"] = True
    write(setup.root / "campaign.json", campaign)
    setup.guard.config["runs"] = [{"run_id": "R1"}]
    setup.guard.config["stop_delivery_grace_seconds"] = 1
    calls = []
    monkeypatch.setattr(setup.guard, "observe", lambda: calls.append("observe"))
    monkeypatch.setattr(setup.guard, "watchdog", lambda: True)
    monkeypatch.setattr(setup.guard, "poll_publication", lambda: calls.append("saved-delivery"))
    setup.guard.tick()
    assert not setup.guard.stopping
    assert calls == ["observe", "saved-delivery"]
    state = setup.guard.state()
    state["stop_delivery"]["deadline_epoch"] = 0
    setup.guard.save(state)
    setup.guard.tick()
    assert setup.guard.stopping
    assert setup.guard.state()["stop_delivery"]["state"] == "blocked"
    assert any(event["kind"] == "stopped_delivery" and event["state"] == "blocked"
               for event in setup.guard.state()["pending_events"])


def test_default_monitoring_is_ten_minutes_and_host_loss_does_not_pause_native(setup):
    setup.guard.config.pop("review_seconds")
    write(setup.root / "supervision/config.json", setup.guard.config)
    assert host.monitoring_policy(setup.root)["review_seconds"] == 600
    assert host.monitoring_policy(setup.root)["on_host_loss"] == "continue"
    setup.guard.config["runs"] = [{"run_id": "R1"}]
    state = setup.guard.state()
    state["observations"] = {"R1": {"campaign": {"continue_discovery": True}}}
    setup.guard.save(state)
    called = []
    setup.guard.native = lambda args: called.append(args) or {"exit_code": 0, "ok": True}
    assert not host.host_available(setup.root, setup.generation)
    assert setup.guard.watchdog()
    assert called == []
    host.configure_monitoring(setup.root, "configure", generation=setup.generation,
                              on_host_loss="pause")
    assert setup.guard.watchdog()
    assert called[0][4] == "pause"
    fence = host.document(Path(called[0][-1]))
    assert fence["scope"] == "guard"


def test_paused_monitoring_keeps_observation_publication_and_user_stop(setup, monkeypatch):
    setup.guard.config["runs"] = [{"run_id": "R1"}]
    state = setup.guard.state()
    state["observations"] = {"R1": {"campaign": {"continue_discovery": True}}}
    setup.guard.save(state)
    before = host.document(setup.root / "campaign.json")
    owner = host.document(setup.root / "supervision/owner.json")
    ack(setup)
    host.configure_monitoring(setup.root, "pause", generation=setup.generation,
                              on_host_loss="pause")
    calls = []
    monkeypatch.setattr(setup.guard, "observe", lambda: calls.append("observe"))
    monkeypatch.setattr(setup.guard, "poll_publication", lambda: calls.append("publish"))
    monkeypatch.setattr(setup.guard, "native", lambda args: calls.append(args) or
                        {"exit_code": 0, "ok": True})
    setup.guard.tick()
    assert calls == ["observe", "publish"]
    assert not host.snapshot(setup.root, setup.generation)["review_due"]
    assert host.document(setup.root / "campaign.json") == before
    assert host.document(setup.root / "supervision/owner.json") == owner
    supervision.records.state_action(SimpleNamespace(command="stop", campaign=setup.root,
                                                     reason="Stop actual analysis"))
    setup.guard.watchdog()
    assert calls[-1][4] == "stop"


def test_live_interval_changes_review_deadline_without_new_goal_or_process(setup):
    first = ack(setup, assessment="Initial inspection")
    campaign = host.document(setup.root / "campaign.json")
    owner = host.document(setup.root / "supervision/owner.json")
    changed = host.configure_monitoring(setup.root, "configure", generation=setup.generation,
                                        review_seconds=1800)
    assert changed["revision"] == 0
    assert changed["native_execution_changed"] is False
    current = host.snapshot(setup.root, setup.generation)
    assert current["next_review_at_epoch"] == first["last_review_at_epoch"] + 1800
    assert host.host_available(setup.root, setup.generation)
    assert setup.guard.review_deadline(setup.guard.state()) == (
        first["last_review_at_epoch"] + 1800)
    host.configure_monitoring(setup.root, "configure", generation=setup.generation,
                              review_seconds=1)
    assert setup.guard.review_deadline(setup.guard.state()) == first["last_review_at_epoch"] + 1
    assert host.document(setup.root / "campaign.json") == campaign
    assert host.document(setup.root / "supervision/owner.json") == owner


def test_resume_monitoring_in_new_session_requires_new_observation_and_fence(setup):
    first = ack(setup, assessment="First host review")
    observation = host.observe(setup.root, setup.generation)
    campaign = host.document(setup.root / "campaign.json")
    host.configure_monitoring(setup.root, "pause", generation=setup.generation)
    with pytest.raises(ValueError, match="Stale observation"):
        host.acknowledge(setup.root, setup.generation, observation["observation_id"],
                         session_id="outer-session", task_id="old")
    resumed = host.configure_monitoring(setup.root, "resume", generation=setup.generation,
                                        session_id="next-session")
    assert resumed["revision"] == 2
    assert not host.host_available(setup.root, setup.generation)
    next_observation = host.observe(setup.root, setup.generation)
    with pytest.raises(ValueError, match="different host"):
        host.acknowledge(setup.root, setup.generation, next_observation["observation_id"],
                         session_id="outer-session", task_id="old")
    second = host.acknowledge(setup.root, setup.generation, next_observation["observation_id"],
                              session_id="next-session", task_id="new", assessment="Reattached")
    assert host.host_available(setup.root, setup.generation)
    assert host.document(Path(first["fence_file"]))["monitoring_revision"] == 0
    assert host.document(Path(second["fence_file"]))["monitoring_revision"] == 2
    assert host.document(setup.root / "campaign.json") == campaign


def test_explicit_pause_loss_grace_covers_configured_review_interval(setup):
    setup.config.update(review_seconds=600, on_host_loss="pause")
    write(setup.root / "supervision/config.json", setup.config)
    first = ack(setup, assessment="Ten-minute review")
    now = first["acknowledged_at_epoch"] + 590
    owner = host.document(setup.root / "supervision/owner.json")
    owner["heartbeat_epoch"] = now
    write(setup.root / "supervision/owner.json", owner)
    assert host.host_available(setup.root, setup.generation, now=now)
    owner["heartbeat_epoch"] = now + 50
    write(setup.root / "supervision/owner.json", owner)
    assert not host.host_available(setup.root, setup.generation, now=now + 50)


def test_paused_reviewer_never_starts_paid_review_but_keeps_delivery(setup, monkeypatch):
    setup.guard.config["supervision_mode"] = "reviewer"
    setup.guard.config["runs"] = [{"run_id": "R1"}]
    host.configure_monitoring(setup.root, "pause", generation=setup.generation)
    called = []
    monkeypatch.setattr(setup.guard, "observe", lambda: called.append("observe"))
    monkeypatch.setattr(setup.guard, "watchdog", lambda: True)
    monkeypatch.setattr(setup.guard, "poll_publication", lambda: called.append("publish"))
    monkeypatch.setattr(setup.guard, "begin_review", lambda: pytest.fail("Paid review started"))
    setup.guard.tick()
    assert called == ["observe", "publish"]


def test_monitoring_cli_validates_owner_without_restarting_native(setup):
    command = [sys.executable, str(SCRIPTS / "supervisor.py"), "monitoring"]
    bad = subprocess.run([*command, "pause", "--campaign", str(setup.root),
                          "--generation", "wrong"], capture_output=True, text=True, check=False)
    assert bad.returncode != 0
    assert not (setup.root / "supervision/monitoring.json").exists()
    good = subprocess.run([*command, "configure", "--campaign", str(setup.root),
                           "--generation", setup.generation, "--review-seconds", "900",
                           "--on-host-loss", "continue"],
                          capture_output=True, text=True, check=False)
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["review_seconds"] == 900
    assert not (setup.root / "supervision/launches").exists()


def test_watch_ignores_fresh_countdown_cost_timestamps_and_acknowledgments(setup):
    setup.guard.config["runs"] = [{"run_id": "R1"}]
    row = {"round": 2, "grade_counts": {"GOOD": 1}, "phase": "discovery",
           "turn": {"state": "running"},
           "campaign": {"active": False, "continue_discovery": True, "seconds_remaining": 3600},
           "usage": {"observed_at": "initial", "totals": {"input_tokens": 1000}},
           "detector_service": {"status": "ready", "infrastructure_blocked": False,
                                "incident_id": "route", "updated_at": "initial"}}
    setup.guard.native = lambda args: {"exit_code": 0, **row}
    setup.guard.observe()
    initial = host.observe(setup.root, setup.generation)
    ack(setup, assessment="Healthy round progress; next review at the agreed interval")
    after_ack = host.observe(setup.root, setup.generation)
    assert after_ack["fingerprint"] == initial["fingerprint"]
    assert after_ack["events"] == []
    for index in range(3):
        row["campaign"]["seconds_remaining"] -= 15
        row["usage"]["observed_at"] = str(index)
        row["usage"]["totals"]["input_tokens"] += 100
        row["detector_service"]["updated_at"] = str(index)
        setup.guard.observe()
        fresh = host.observe(setup.root, setup.generation)
        assert fresh["fingerprint"] == initial["fingerprint"]
        assert fresh["runs"]["R1"]["usage"]["observed_at"] == str(index)
        assert not fresh["review_due"]
    timed = host.observe(setup.root, setup.generation, after=initial["fingerprint"], timeout=.05)
    assert timed["fingerprint"] == initial["fingerprint"]
    streamed = list(host.watch_stream(setup.root, setup.generation,
                                      after=initial["fingerprint"], duration=1))
    assert [event.get("kind") for event in streamed] == ["watch_expired"]


def test_review_timer_emits_once_and_substantive_ack_does_not_echo(setup):
    ack(setup, assessment="Healthy progress")
    initial = host.observe(setup.root, setup.generation)
    saved = host.document(setup.root / "supervision/host-ack.json")
    saved["last_review_at_epoch"] -= 11
    write(setup.root / "supervision/host-ack.json", saved)
    due = host.observe(setup.root, setup.generation, after=initial["fingerprint"], timeout=1)
    assert due["review_due"]
    assert due["fingerprint"] != initial["fingerprint"]
    repeated = host.observe(setup.root, setup.generation)
    assert repeated["fingerprint"] == due["fingerprint"]
    ack(setup, assessment="Reviewed latest evidence; native work remains healthy")
    reset = host.observe(setup.root, setup.generation)
    assert not reset["review_due"]
    assert reset["fingerprint"] == due["fingerprint"]


@pytest.mark.parametrize("change", ["round", "error", "service", "coverage", "boundary"])
def test_important_native_changes_wake_once_before_periodic_review(setup, change):
    setup.guard.config["runs"] = [{"run_id": "R1"}]
    row = {"round": 0, "turn": {"state": "running"},
           "campaign": {"active": False, "continue_discovery": True, "coverage": .1},
           "detector_service": {"status": "ready", "infrastructure_blocked": False,
                                "incident_id": "route"}}
    setup.guard.native = lambda args: {"exit_code": 0, **row}
    setup.guard.observe()
    ack(setup, assessment="Initial state reviewed")
    initial = host.observe(setup.root, setup.generation)
    if change == "round":
        row["round"] = 1
    elif change == "error":
        row["turn"]["state"] = "failed"
    elif change == "service":
        row["detector_service"].update(status="infrastructure_blocked", infrastructure_blocked=True)
    elif change == "coverage":
        row["campaign"]["coverage"] = .4
    else:
        row["campaign"].update(stop_reason="spend_reserve", continue_discovery=False)
    setup.guard.observe()
    event = host.observe(setup.root, setup.generation, after=initial["fingerprint"], timeout=1)
    assert event["fingerprint"] != initial["fingerprint"]
    assert event["review_due"] and event["events"]
    for _ in range(2):
        setup.guard.observe()
        unchanged = host.observe(setup.root, setup.generation)
        assert unchanged["fingerprint"] == event["fingerprint"]
    ack(setup, assessment="Inspected the new event; intervention only if needed")
    assert host.observe(setup.root, setup.generation)["fingerprint"] == event["fingerprint"]
