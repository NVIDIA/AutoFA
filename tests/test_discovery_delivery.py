# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved-only delivery verifies the real immutable artifacts without paid work."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.discovery_delivery import saved_only_delivery
from autofa.data.remote_library_api import RemoteLibraryClient, RemoteLibraryError
from autofa.mcp.runtime import Runtime
from autofa_library_server.config import Settings
from autofa_library_server.main import create_app

from support.library_server import saved_run as saved_run


def connect(monkeypatch, client, server, token, calls):
    """Route the real client through the in-process library server, recording requests."""

    def request(method, url, *, body=None, headers=None, **kwargs):
        calls.append((method, url))
        response = server.request(method, url, content=body, headers={
            **(headers or {}), "Authorization": f"Bearer {token}"})
        if response.status_code >= 400:
            raise RemoteLibraryError(response.text, status=response.status_code)
        return response.content

    def download(version, destination):
        data = client._request("GET", version.library_url)
        assert hashlib.sha256(data).hexdigest() == version.library_sha256
        destination.write_bytes(data)
        return destination

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "download_version", download)


def fail_posts(monkeypatch, client, error, *, commit=False, once=False):
    """Make appends raise ``error``, after committing them on the server when ``commit``.

    Returns the real request and the list of failed append URLs.
    """
    request = client._request
    failed = []

    def intercepted(method, url, **kwargs):
        if method != "POST" or (once and failed):
            return request(method, url, **kwargs)
        failed.append(url)
        if commit:
            request(method, url, **kwargs)
        raise error

    monkeypatch.setattr(client, "_request", intercepted)
    return request, failed


def change_saved_recall(run, recall=0.12):
    library = FunctionLibrary.load(run.run_dir() / "library.json")
    library.get("saved").scores["recall"] = recall
    library.save(run.run_dir() / "library.json")


def posts(calls):
    return len([method for method, _ in calls if method == "POST"])


def agreed_contract(cfg, monkeypatch):
    # Destination checks read the saved contract; hashing the installed package is
    # covered by the setup-contract tests and is needlessly slow here.
    from autofa.setup_contract import resolved_contract

    monkeypatch.setattr("autofa.setup_contract.runtime_identity",
                        lambda: {"source_sha256": "installed source tree"})
    return resolved_contract(cfg)


@pytest.fixture
def saved_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    settings = Settings(data_dir=tmp_path / "server", library_dir=tmp_path / "library",
                        runs_dir=tmp_path / "runs", local_token="synthetic")
    client = RemoteLibraryClient(settings.public_url, token=settings.local_token)
    calls = []
    parquet = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "model_id": "model", "benchmark": "bench", "problem_id": "one", "is_correct": False,
        "response_text": "saved response",
    }]), parquet)
    run = RunConfig(run_id="saved", model_id="model", benchmarks=["bench"],
                    parquet_paths=[str(parquet)], analysis_outputs=["agent"])
    run.save()
    library = FunctionLibrary(model_id="model", benchmark="bench")
    library.upsert(FeatureFunction(
        name="saved", description="Saved finding", source="def detect(item):\n    return True\n",
        scores={"recall": 0.42, "precision": 0.9, "evaluated": True,
                "behavior_review_assessments": [{"evidence": "proof " * 4000}]},
    ))
    library.save(run.run_dir() / "library.json")
    (run.run_dir() / "finalization_outcome.json").write_text(json.dumps({
        "state": "completed_with_gaps", "report_generated": True,
        "retained_results_complete": False,
    }))
    cfg = Config(remote_library_base_url=client.base_url)
    monkeypatch.setattr(Runtime, "ensure_hub_llm",
                        lambda *a: pytest.fail("No inference in delivery"))
    with TestClient(create_app(settings), base_url=settings.public_url) as server:
        connect(monkeypatch, client, server, settings.local_token, calls)
        yield run, cfg, client, calls


def test_saved_only_publish_reads_back_and_reuses_exact_version(saved_delivery, monkeypatch):
    run, cfg, client, calls = saved_delivery
    result = saved_only_delivery(run, cfg, client=client)
    assert result["status"] == "verified"
    assert "version_id=" in result["url"] and "branch_id=" in result["url"]
    assert result["metrics"]["detectors"][0]["recall"] == 0.42
    assert result["finalization"]["state"] == "completed_with_gaps"
    again = saved_only_delivery(run, cfg, client=client)
    assert again["version_id"] == result["version_id"]
    assert posts(calls) == 1
    assert not (run.run_dir() / "publication_intent.json").exists()
    # The known immutable version is not hidden by the server's list window.
    monkeypatch.setattr(client, "list_versions", lambda *a: [])
    verified = saved_only_delivery(run, cfg, client=client, verify_only=True)
    assert verified["version_id"] == result["version_id"]


def test_lost_response_reconciles_committed_version_without_second_append(
    saved_delivery, monkeypatch,
):
    run, cfg, client, calls = saved_delivery
    fail_posts(monkeypatch, client, TimeoutError("reply lost"), commit=True)
    with pytest.raises(TimeoutError):
        saved_only_delivery(run, cfg, client=client)
    assert run.remote_library_id is None
    result = saved_only_delivery(run, cfg, client=client)
    assert result["status"] == "verified"
    assert posts(calls) == 1


def test_uncertain_absence_does_not_append_again(saved_delivery, monkeypatch):
    run, cfg, client, _ = saved_delivery
    _, attempts = fail_posts(monkeypatch, client, TimeoutError("unknown network outcome"))
    with pytest.raises(TimeoutError):
        saved_only_delivery(run, cfg, client=client)
    result = saved_only_delivery(run, cfg, client=client)
    assert result["status"] == "publication_uncertain"
    assert not result["retry_safe"] and len(attempts) == 1


def test_local_delivery_never_constructs_cloud_client(saved_delivery, monkeypatch):
    from autofa.data.discovery_delivery import local_saved_delivery

    run, cfg, _, calls = saved_delivery
    monkeypatch.setattr("autofa.data.discovery_delivery.client_from_config",
                        lambda *a: pytest.fail("Local-only must not access cloud"))
    result = local_saved_delivery(run, cfg, verify_only=False)
    assert result["status"] == "verified" and Path(result["url"]).exists()
    assert local_saved_delivery(run, cfg, verify_only=True)["status"] == "verified"
    assert calls == []


@pytest.mark.parametrize("outputs", [[], ["efficiency"]])
def test_saved_only_delivery_requires_a_discovery_only_run(saved_delivery, outputs):
    from autofa.data.discovery_delivery import local_saved_delivery

    run, cfg, client, calls = saved_delivery
    run.analysis_outputs = outputs
    with pytest.raises(ValueError, match="discovery-only"):
        saved_only_delivery(run, cfg, client=client)
    with pytest.raises(ValueError, match="discovery-only"):
        local_saved_delivery(run, cfg, verify_only=False)
    assert calls == []
    assert not (run.run_dir() / "delivery_attempt.json").exists()
    assert not (run.run_dir() / "local_delivery.json").exists()


def test_preparation_failure_is_safe_to_retry(saved_delivery, monkeypatch):
    run, cfg, client, calls = saved_delivery
    import autofa.data.discovery_delivery as module

    publish = module.publish_run

    def fail(*args, **kwargs):
        raise ValueError("local report failure")

    monkeypatch.setattr(module, "publish_run", fail)
    with pytest.raises(ValueError, match="local report"):
        module.saved_only_delivery(run, cfg, client=client)
    assert json.loads((run.run_dir() / "delivery_attempt.json").read_text())["status"] == "prepared"
    assert calls == []
    monkeypatch.setattr(module, "publish_run", publish)
    assert module.saved_only_delivery(run, cfg, client=client)["status"] == "verified"


def test_reconcile_old_attempt_then_publish_changed_checkpoint(saved_delivery, monkeypatch):
    run, cfg, client, calls = saved_delivery
    request, _ = fail_posts(monkeypatch, client, TimeoutError("reply lost"), commit=True)
    with pytest.raises(TimeoutError):
        saved_only_delivery(run, cfg, client=client)
    change_saved_recall(run)
    monkeypatch.setattr(client, "_request", request)
    pending = saved_only_delivery(run, cfg, client=client, verify_only=True)
    assert pending["status"] == "pending" and pending["retry_safe"]
    delivered = saved_only_delivery(run, cfg, client=client)
    assert delivered["status"] == "verified"
    assert delivered["metrics"]["detectors"][0]["recall"] == 0.12
    assert posts(calls) == 2


# Exercise the same real managed trace fixture that the lazy report server uses.
def test_managed_local_readback_preserves_native_sources(saved_run, monkeypatch):
    settings, store, library, _ = saved_run
    saved = json.loads((settings.runs_dir / "saved" / "config.json").read_text())
    run = RunConfig(run_id="saved", parquet_paths=[str(store / "dataset.parquet")],
                    model_id=library.model_id, benchmarks=[library.benchmark],
                    reference_data=saved["reference_data"], analysis_outputs=["agent"])
    run.save()
    cfg = Config(remote_library_base_url=settings.public_url)
    client = RemoteLibraryClient(settings.public_url, token=settings.local_token)
    with TestClient(create_app(settings), base_url=settings.public_url) as server:
        connect(monkeypatch, client, server, settings.local_token, [])
        result = saved_only_delivery(run, cfg, client=client)
        assert result["status"] == "verified"
        assert result["metrics"]["detectors"][0]["recall"] == 1.0


def test_saved_only_cli_bypasses_harness_bootstrap_and_holds_run_lock(
    saved_delivery, monkeypatch, capsys,
):
    import autofa.cli as cli
    import autofa.data.discovery_delivery as module
    from autofa.session.recovery import run_lock

    run, cfg, _, _ = saved_delivery
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "ensure_default_config",
                        lambda *a, **kw: pytest.fail("Saved-only must not bootstrap a harness"))
    monkeypatch.setattr(module, "load_config", lambda: cfg)

    def verify(saved, settings, *, verify_only):
        assert saved.run_id == run.run_id and verify_only
        with pytest.raises(RuntimeError, match="already active"):
            with run_lock(saved.run_dir()):
                pass
        return {"status": "verified", "run_id": saved.run_id}

    monkeypatch.setattr(module, "saved_only_delivery", verify)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["publish", "--run", run.run_id, "--saved-only", "--json", "--verify-only"])
    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


def test_managed_destination_drift_is_rejected_before_requests(saved_delivery, monkeypatch):
    from dataclasses import replace

    run, cfg, client, calls = saved_delivery
    (run.run_dir() / "setup_contract.json").write_text(
        json.dumps(agreed_contract(cfg, monkeypatch)))
    with pytest.raises(ValueError, match="agreed destination"):
        saved_only_delivery(run, replace(cfg, remote_library_token_env="DIFFERENT_CREDENTIAL"),
                            client=client)
    assert calls == []


def test_managed_destination_allows_code_upgrade_but_checks_saved_integrity(
    saved_delivery, monkeypatch,
):
    from autofa.core.dependencies import digest

    run, cfg, client, calls = saved_delivery
    agreed = agreed_contract(cfg, monkeypatch)
    agreed["runtime"]["source_sha256"] = "previous source tree"
    agreed["fingerprint"] = digest({key: agreed[key] for key in (
        "settings", "execution", "source", "required_capabilities", "runtime")})
    path = run.run_dir() / "setup_contract.json"
    path.write_text(json.dumps(agreed))
    assert saved_only_delivery(run, cfg, client=client)["status"] == "verified"
    calls.clear()
    agreed["runtime"]["source_sha256"] = "corrupted snapshot"
    path.write_text(json.dumps(agreed))
    with pytest.raises(ValueError, match="fingerprint changed"):
        saved_only_delivery(run, cfg, client=client, verify_only=True)
    assert calls == []


def test_user_stopped_snapshot_publishes_as_partial_without_final_validation(saved_delivery):
    run, cfg, client, _ = saved_delivery
    (run.run_dir() / "finalization_outcome.json").unlink()
    supervision = run.run_dir() / "supervision"
    supervision.mkdir(exist_ok=True)
    stop = {"action": "stop", "state": "requested", "requested_at": "2026-11-05T12:00:00Z"}
    (supervision / "lifecycle.json").write_text(json.dumps(stop))
    before = (run.run_dir() / "library.json").read_bytes()
    delivered = saved_only_delivery(run, cfg, client=client)
    assert delivered["status"] == "verified"
    assert delivered["finalization"]["state"] == "stopped"
    assert delivered["finalization"]["retained_results_complete"] is False
    assert delivered["finalization"]["report_generated"] is True
    assert not (run.run_dir() / "finalization_outcome.json").exists()
    assert json.loads((supervision / "lifecycle.json").read_text()) == stop
    # Native persistence normalizes the saved envelope but never changes detector evidence.
    after = json.loads((run.run_dir() / "library.json").read_bytes())
    assert json.loads(before)["functions"] == after["functions"]


def test_explicit_size_rejection_allows_changed_checkpoint_retry(saved_delivery, monkeypatch):
    run, cfg, client, calls = saved_delivery
    fail_posts(monkeypatch, client, RemoteLibraryError("Library exceeds server limit", status=413),
               once=True)
    with pytest.raises(RemoteLibraryError, match="server limit"):
        saved_only_delivery(run, cfg, client=client)
    receipt = json.loads((run.run_dir() / "delivery_attempt.json").read_text())
    assert receipt["status"] == "rejected"
    pending = saved_only_delivery(run, cfg, client=client, verify_only=True)
    assert pending["status"] == "pending" and pending["retry_safe"]
    change_saved_recall(run)
    assert saved_only_delivery(run, cfg, client=client)["status"] == "verified"
    assert posts(calls) == 1
