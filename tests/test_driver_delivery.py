# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The portable driver cannot complete by merely recording a URL or a queued message."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "delivery_under_test",
    Path(__file__).resolve().parents[1] / "agent-skills/autofa-driver/scripts/delivery.py",
)
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def campaign(tmp_path):
    write(
        tmp_path / "campaign.json",
        {
            "id": "campaign",
            "revision": 0,
            "state": "active",
            "history": [],
            "goal": {"objective": "Find failures", "delivery": {"destination": "cloud"}},
        },
    )
    delivery.configure(tmp_path, {"argv": ["autofa"]}, ["run"])
    return tmp_path


def verified():
    return {
        "status": "verified",
        "run_id": "run",
        "url": "https://example.test/libraries/L?version_id=V",
        "version_id": "V",
        "snapshot": {"library_sha256": "L", "report_sha256": "R"},
        "metrics": {"coverage": 0.8},
        "finalization": {"state": "completed"},
    }


def summarize(campaign, outcome="complete"):
    return delivery.prepare_summary(campaign, {
        "outcome": outcome, "text": "Findings: " + verified()["url"],
        "findings": ["Supported saved finding"], "next_actions": [],
        "cost": {"status": "unknown", "reason": "No saved usage"},
    })


def acknowledge(campaign, summary, *, generation="generation", owner=None):
    """Record the current host's acknowledgment of the prepared final summary."""
    write(campaign / "supervision/owner.json",
          owner or {"generation": "generation", "state": "stopped"})
    write(campaign / "supervision/config.json", {"host": {"session_id": "session"}})
    write(campaign / "supervision/host-acks/final.json", {
        "id": "ack", "observation_id": "observation", "summary_sha256": summary["sha256"],
        "channel": "foreground", "generation": generation, "session_id": "session",
    })


def test_cloud_mapping_and_explicit_local_only():
    assert delivery.execution_settings({"delivery": {"destination": "cloud"}}) == {"publish": True}
    assert delivery.execution_settings({"delivery": {"destination": "local"}}) == {"publish": False}
    with pytest.raises(ValueError, match="Agree"):
        delivery.execution_settings({})


def test_freeform_publication_receipt_is_not_completion(campaign, monkeypatch):
    write(campaign / "delivery/run.json", verified())
    monkeypatch.setattr(
        delivery, "_invoke", lambda *a, **kw: {"run_id": "run", "status": "pending"}
    )
    assert not delivery.completion_evidence(campaign)["complete"]


def test_native_safe_retry_clears_outer_uncertainty(campaign, monkeypatch):
    write(campaign / "delivery/run.json", {"run_id": "run", "status": "publication_uncertain"})
    calls = []

    def invoke(*a, publish, **kw):
        calls.append(publish)
        return verified() if publish else {"status": "pending", "run_id": "run", "retry_safe": True}

    monkeypatch.setattr(delivery, "_invoke", invoke)
    assert delivery.reconcile_delivery(campaign, "run", allow_publish=True)["status"] == "verified"
    assert calls == [False, True]


def test_user_stop_suppresses_unagreed_upload(campaign, monkeypatch):
    state = json.loads((campaign / "campaign.json").read_text())
    state["state"] = "stopped"
    write(campaign / "campaign.json", state)
    calls = []

    def invoke(*a, publish, **kw):
        calls.append(publish)
        return {"status": "pending", "run_id": "run", "retry_safe": True}

    monkeypatch.setattr(delivery, "_invoke", invoke)
    delivery.reconcile_delivery(campaign, "run", allow_publish=True)
    assert calls == [False]


def test_completion_requires_exact_summary_and_current_host_ack(campaign, monkeypatch):
    monkeypatch.setattr(delivery, "_invoke", lambda *a, **kw: verified())
    delivery.reconcile_delivery(campaign, "run")
    summary = summarize(campaign)
    assert not delivery.completion_evidence(campaign)["complete"]
    acknowledge(campaign, summary, generation="stale", owner={"generation": "current"})
    assert not delivery.completion_evidence(campaign)["complete"]
    acknowledge(campaign, summary, generation="current", owner={"generation": "current"})
    assert delivery.completion_evidence(campaign)["complete"]
    # A reattached host session must acknowledge the summary itself.
    write(campaign / "supervision/monitoring.json", {"host_session_id": "second", "revision": 1})
    assert not delivery.completion_evidence(campaign)["complete"]
    ack = json.loads((campaign / "supervision/host-acks/final.json").read_text())
    write(campaign / "supervision/host-acks/final.json",
          {**ack, "session_id": "second", "monitoring_revision": 1})
    assert delivery.completion_evidence(campaign)["complete"]
    monkeypatch.setattr(
        delivery,
        "_invoke",
        lambda *a, **kw: {**verified(), "snapshot": {"library_sha256": "changed"}},
    )
    assert not delivery.completion_evidence(campaign)["complete"]


def test_driver_complete_uses_same_validator(campaign, monkeypatch):
    driver = delivery.records
    monkeypatch.setattr(driver, "delivery_helper", lambda: delivery)
    monkeypatch.setattr(
        delivery, "_invoke", lambda *a, **kw: {"status": "pending", "run_id": "run"}
    )
    with pytest.raises(ValueError, match="Delivery is incomplete"):
        driver.state_action(SimpleNamespace(command="complete", campaign=campaign, reason="done"))
    assert json.loads((campaign / "campaign.json").read_text())["state"] == "active"


def test_launch_builder_overrides_accidental_local_publish_setting(tmp_path):
    output = tmp_path / "native.json"
    receipt = delivery.prepare_run_config(
        {"delivery": {"destination": "cloud"}},
        {
            "version": 2,
            "source": {"parquet": ["dataset.parquet"]},
            "execution": {"publish": False, "require_llm_detectors": True},
        },
        output,
    )
    spec = json.loads(output.read_text())
    assert spec["execution"] == {"publish": True, "require_llm_detectors": True}
    assert receipt["execution"]["publish"] is True


def test_native_output_is_bounded_without_capturing_unbounded_pipes(tmp_path):
    import sys

    result = delivery._invoke(
        {"argv": [sys.executable, "-c", "print('X' * 3000000)"]}, "run", publish=False
    )
    assert result["status"] == "blocked" and "bound" in result["reason"]


@pytest.mark.parametrize("after", ["reopen", "revise", "artifact_change"])
def test_completed_delivery_is_idempotent_but_contract_changes_invalidate(
    campaign, monkeypatch, after
):
    driver = delivery.records
    monkeypatch.setattr(driver, "delivery_helper", lambda: delivery)
    monkeypatch.setattr(delivery, "_invoke", lambda *a, **kw: verified())
    delivery.reconcile_delivery(campaign, "run")
    acknowledge(campaign, summarize(campaign))
    args = SimpleNamespace(command="complete", campaign=campaign, reason="Delivered findings")
    complete = driver.state_action(args)
    assert complete["revision"] == 1
    assert complete["completion"]["contract_revision"] == 0
    assert complete["completion"]["completed_revision"] == 1
    assert delivery.completion_evidence(campaign)["complete"]
    # The original delivered message remains accepted when the stopped guard
    # is replaced, without fabricating a second user-message acknowledgment.
    write(campaign / "supervision/owner.json", {"generation": "replacement", "state": "running"})
    write(campaign / "supervision/config.json", {"host": {"session_id": "new-session"}})
    assert delivery.completion_evidence(campaign)["complete"]
    saved = (campaign / "campaign.json").read_bytes()
    assert driver.state_action(args) == complete
    assert (campaign / "campaign.json").read_bytes() == saved
    if after == "artifact_change":
        monkeypatch.setattr(
            delivery,
            "_invoke",
            lambda *a, **kw: {**verified(), "snapshot": {"library_sha256": "new checkpoint"}},
        )
    elif after == "reopen":
        driver.state_action(SimpleNamespace(command="reopen", campaign=campaign, reason="New work"))
    else:
        patch = campaign / "goal-patch.json"
        write(patch, {"objective": "Changed objective"})
        driver.state_action(
            SimpleNamespace(command="revise", campaign=campaign, patch=patch, reason="Changed goal")
        )
    assert not delivery.completion_evidence(campaign)["complete"]
    with pytest.raises(ValueError, match="Delivery is incomplete"):
        driver.state_action(args)


def test_supervisor_completion_proof_uses_same_accepted_contract(campaign, monkeypatch):
    driver = delivery.records
    monkeypatch.setattr(driver, "delivery_helper", lambda: delivery)
    monkeypatch.setattr(delivery, "_invoke", lambda *a, **kw: verified())
    delivery.reconcile_delivery(campaign, "run")
    summary = summarize(campaign)
    write(campaign / "supervision/config.json", {"delivery": {"channel": "test-channel"}})
    message_id = "final-summary-" + summary["sha256"]
    write(
        campaign / "supervision/outbox/final.json",
        {
            "id": message_id,
            "message": summary["summary"]["text"],
            "state": "delivered",
            "delivery_receipt": {
                "id": message_id,
                "delivered": True,
                "channel": "test-channel",
                "remote_id": "remote-message",
            },
        },
    )
    proof = delivery.completion_evidence(campaign)
    assert proof["complete"]
    state = json.loads((campaign / "campaign.json").read_text())
    state.update(
        state="complete",
        revision=1,
        completion={"outcome": "complete", "delivery": {**proof, "completed_revision": 1}},
    )
    write(campaign / "campaign.json", state)
    assert delivery.completion_evidence(campaign)["complete"]


def test_delivery_uses_managed_run_binding_instead_of_mutable_global_config(campaign, monkeypatch):
    frozen = {"argv": ["frozen-autofa"], "env": {"AUTOFA_CONFIG": "/prepared/autofa.toml"}}
    write(
        campaign / "supervision/config.json",
        {
            "native": {"argv": ["old-autofa"], "env": {"AUTOFA_CONFIG": "/old/autofa.toml"}},
            "runs": [{"run_id": "run", "native": frozen}],
        },
    )
    seen = []

    def invoke(native, *a, **kw):
        seen.append(native)
        return verified()

    monkeypatch.setattr(delivery, "_invoke", invoke)
    delivery.reconcile_delivery(campaign, "run", native={"argv": ["caller-old-autofa"]})
    assert seen == [frozen]


@pytest.mark.parametrize(
    ("source", "timeout"),
    [("default", 900), ("global", 1800.5), ("managed_run", 1200), ("delivery_run", 2700)],
)
def test_slow_healthy_delivery_uses_resolved_binding_timeout(
    campaign, monkeypatch, source, timeout,
):
    native = {"argv": ["bound-autofa"], "timeout_seconds": timeout}
    if source == "global":
        delivery.configure(campaign, native, ["run"])
    elif source in {"managed_run", "delivery_run"}:
        path = campaign / ("supervision/config.json" if source == "managed_run"
                           else "delivery/config.json")
        write(path, {"native": {"argv": ["global-autofa"], "timeout_seconds": 60},
                     "runs": [{"run_id": "run", "native": native}], "run_ids": ["run"]})
    seen = []

    def slow_native(argv, **kwargs):
        seen.append((argv, kwargs["timeout"]))
        # Both readback and publishing can legitimately exceed the old one-minute limit.
        if kwargs["timeout"] < 90:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        result = ({"status": "pending", "run_id": "run", "retry_safe": True}
                  if "--verify-only" in argv else verified())
        kwargs["stdout"].write(json.dumps(result).encode())

    monkeypatch.setattr(delivery.subprocess, "run", slow_native)
    receipt = delivery.reconcile_delivery(campaign, "run", allow_publish=True)
    assert receipt["status"] == "verified"
    assert [value for _, value in seen] == [timeout, timeout]
    assert "--verify-only" in seen[0][0]
    assert "--verify-only" not in seen[1][0]
    if source != "default":
        assert all(argv[0] == "bound-autofa" for argv, _ in seen)


@pytest.mark.parametrize(("source", "timeout"), [
    # Each binding source is validated before native work starts; together the rows
    # cover a boolean, a non-finite, a non-positive and a non-numeric timeout.
    ("global", True), ("managed_run", float("nan")), ("delivery_run", 0), ("delivery_run", "900"),
])
def test_delivery_rejects_invalid_resolved_timeouts_before_invoking(
    campaign, monkeypatch, source, timeout,
):
    native = {"argv": ["autofa"], "timeout_seconds": timeout}
    if source == "global":
        original = (campaign / "delivery/config.json").read_bytes()
        with pytest.raises(ValueError, match="timeout_seconds.*positive finite"):
            delivery.configure(campaign, native, ["run"])
        assert (campaign / "delivery/config.json").read_bytes() == original
        write(campaign / "delivery/config.json", {"native": native, "run_ids": ["run"]})
    else:
        path = campaign / ("supervision/config.json" if source == "managed_run"
                           else "delivery/config.json")
        write(path, {"native": {"argv": ["global-autofa"]},
                     "runs": [{"run_id": "run", "native": native}], "run_ids": ["run"]})
    monkeypatch.setattr(delivery, "_invoke", lambda *a, **kw: pytest.fail("Must not invoke native"))
    with pytest.raises(ValueError, match="timeout_seconds.*positive finite|not JSON compliant"):
        delivery.reconcile_delivery(campaign, "run", allow_publish=True)
    assert not (campaign / "delivery/run.json").exists()


@pytest.mark.parametrize("authorized", [False, True])
def test_stopped_saved_work_requires_explicit_authorization_and_partial_summary(
    campaign,
    monkeypatch,
    authorized,
):
    state = json.loads((campaign / "campaign.json").read_text())
    state["state"] = "stopped"
    state["goal"]["delivery"]["publish_on_stop"] = authorized
    write(campaign / "campaign.json", state)
    stopped = {
        **verified(),
        "finalization": {
            "state": "stopped",
            "stop_reason": "user_stop",
            "report_generated": True,
            "retained_results_complete": False,
        },
    }
    monkeypatch.setattr(delivery, "_invoke", lambda *a, **kw: stopped)
    delivery.reconcile_delivery(campaign, "run")
    acknowledge(campaign, summarize(campaign, "partial"))
    assert delivery.completion_evidence(campaign)["complete"] is authorized
