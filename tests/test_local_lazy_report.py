# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where published report examples come from: dashboard snapshots resolve one trusted local
example on demand (also on a real standalone server with an independent profile), while cloud
reports link exact source rollouts without reading or uploading their text."""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

import autofa
from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.local_trace_store import file_digest
from autofa.data.parquet_store import Dataset
from autofa.data.reference_data import (
    canonical_export_url,
    make_local_reference_data,
    make_reference_data,
)
from autofa.data.remote_library_api import (
    RemoteLibraryClient,
    RemoteLibraryError,
    RemotePublication,
    client_from_config,
)
from autofa.data.remote_library_publish import publish_run, published_library_is_current
from autofa.data.trace_navigation import NAVIGATION_VERSION
from autofa.data.trace_upload import TraceArchive
from autofa.mcp.runtime import Runtime
from autofa.reports.failure_cards import (
    _extract_autofa_data,
    build_data,
    build_report_assets_zip,
    render_html,
    write_failure_cards,
)
from autofa_library_server.lazy_report import KEY_PREFIX
from autofa_library_server.main import create_app

from support.library_server import RUN, STORE, TEXT, _base
from support.library_server import saved_run as saved_run

SERVER_TOKEN = "portable-test-server-token-" + "x" * 40


def _upload(client, settings, library, report, branch, source_run_id, assets=None):
    files = {"library": ("library.json", json.dumps(library), "application/json"),
             "failure_card": ("report.html", report, "text/html")}
    if assets is not None:
        files["report_assets"] = ("assets.zip", assets, "application/zip")
    response = client.post(
        "/api/v2/publications", headers={"Authorization": f"Bearer {settings.local_token}"},
        data={"metadata": json.dumps({"branch_key": branch, "source_run_id": source_run_id,
                                      "model_id": "local-model", "benchmarks": ["local-bench"]})},
        files=files,
    )
    assert response.status_code == 201, response.text
    return response.json()


def _catalog_report(app, client):
    completed = app.state.local_catalog.request_refresh()
    assert completed is not None and completed.wait(5)
    library, = client.get("/api/v2/libraries").json()["items"]
    report = client.get(f"/libraries/{library['id']}/report")
    assert report.status_code == 200
    return library, _extract_autofa_data(report.text)


def test_efficiency_publication_survives_report_replacement_and_rendering(saved_run, monkeypatch):
    from autofa.reports import efficiency_card
    from autofa.reports.failure_cards import render_offline_html

    settings, _, library, iid = saved_run
    data = build_data(library)
    efficiency = {"n_items": 2, "n_fail": 0, "n_pass": 2, "metrics": []}
    sections = {"tokens": {"metrics": ["input_tokens"], "plot": "data:image/png;base64,cGxvdA=="}}
    uploaded = efficiency_card._inject_overview(
        render_html(data), efficiency, sections=sections, layout="tabs",
    )
    monkeypatch.setattr(
        efficiency_card, "_evalrdl_sections",
        lambda *args, **kwargs: pytest.fail("Serving a saved report must not recompute metrics"),
    )
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        published = _upload(client, settings, library.to_dict(), uploaded, "efficiency", "saved")
        report = client.get(f"/libraries/{published['library_id']}/report")
        assert report.status_code == 200
        assert report.text.count("const originalBoot = boot;") == 1
        served = _extract_autofa_data(report.text)
        assert served["meta"]["local_lazy_items"]
        assert served["meta"]["efficiency_layout"] == "tabs"
        assert served["meta"]["efficiency_metrics"] == {**efficiency, "sections": sections}
        assert served["meta"]["n_failures"] == data["meta"]["n_failures"] == 1
        item = client.get(served["meta"]["items_base"] + "/" + served["item_index"][iid])
        assert item.status_code == 200 and "TAIL EVIDENCE" in item.json()["response_raw"]

    # Offline export uses the same renderer and must retain the efficiency UI too.
    (settings.runs_dir / "saved" / "efficiency.html").write_text(uploaded)
    offline = render_offline_html(settings.runs_dir / "saved", html_name="efficiency.html")
    assert offline.count("const originalBoot = boot;") == 1


def test_catalog_never_materializes_full_trajectories_and_resolves_one(saved_run, monkeypatch):
    settings, store, _, iid = saved_run
    reads = []
    original_batches, original_reads = Dataset.iter_batches, Dataset.read_locations

    def projected(self, **kwargs):
        assert kwargs.get("columns") is not None, "Catalog must not scan full trajectories"
        assert "response_text" not in kwargs["columns"]
        return original_batches(self, **kwargs)

    def point_read(self, locations, **kwargs):
        reads.append(locations)
        return original_reads(self, locations, **kwargs)

    monkeypatch.setattr(Dataset, "iter_batches", projected)
    monkeypatch.setattr(Dataset, "read_locations", point_read)
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        library, data = _catalog_report(app, client)
        assert reads == []
        assert data["items"] == {} and data["meta"]["local_lazy_items"]
        assert TEXT not in json.dumps(data)
        asset = app.state.repository.get_report_asset(
            library["latest_version_id"], "failure_cards_data/items/" + data["item_index"][iid],
        )
        assert asset["artifact_key"].startswith(KEY_PREFIX)
        assert asset["size_bytes"] < len(TEXT)
        url = data["meta"]["items_base"] + "/" + data["item_index"][iid]
        denied = url.replace(library["latest_version_id"] + "/", "invalid/")
        assert client.get(denied).status_code == 404 and reads == []
        response = client.get(url)
        assert response.status_code == 200
        item = response.json()
        assert item["item_id"] == iid and "TAIL EVIDENCE" in item["response_raw"]
        assert len(reads) == 1 and len(reads[0]) == 1
        evidence = item["localisations"]["detector"]["passages"][0]
        assert evidence["trace_locations"][0]["source_event"] == {
            "event_idx": 55, "part_idx": 7, "lane": "conversation",
        }
        assert "text=%F0%9F%99%82%20TAIL%20EVIDENCE" in evidence["trace_locations"][0]["source_url"]
        # Immutable store corruption fails closed without returning partial evidence.
        (store / "dataset.parquet").write_bytes(b"changed")
        assert client.get(url).status_code == 404
        assert len(reads) == 1


def test_uploaded_descriptor_is_plain_json_and_cannot_read_local_store(saved_run):
    settings, _, library, iid = saved_run
    payload = json.dumps({"version": 1, "store_id": STORE, "run_id": RUN,
                          "item_id": iid, "report_item_id": iid})
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("items/example.json", payload)
    report = render_html({"meta": {"items_base": "items"}, "functions": [],
                          "items": {}, "item_index": {iid: "example.json"}})
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        version = _upload(client, settings, library.to_dict(), report, "upload", "upload",
                          assets=archive.getvalue())
        asset = app.state.repository.get_report_asset(version["version_id"], "items/example.json")
        assert not asset["artifact_key"].startswith(KEY_PREFIX)
        assert app.state.artifact_store.get(asset["artifact_key"]).decode() == payload


def test_deleting_lazy_snapshot_preserves_the_imported_store(saved_run):
    settings, store, _, iid = saved_run
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        library, data = _catalog_report(app, client)
        asset = app.state.repository.get_report_asset(
            library["latest_version_id"], "failure_cards_data/items/" + data["item_index"][iid],
        )
        app.state.artifact_store.delete(asset["artifact_key"])
        assert (store / "dataset.parquet").is_file()
        url = data["meta"]["items_base"] + "/" + data["item_index"][iid]
        assert client.get(url).status_code == 404


@pytest.mark.parametrize("source_run_id", ["../saved", "saved"])
def test_unmatched_upload_cannot_resolve_local_sources(saved_run, source_run_id, monkeypatch):
    from autofa_library_server.lazy_report import report_for_saved_run

    settings, _, library, _ = saved_run
    payload = library.to_dict()
    if source_run_id == "saved":
        payload["functions"][0]["description"] = "a different uploaded detector"
    monkeypatch.setattr("autofa_library_server.lazy_report.resolve_store",
                        lambda *args: pytest.fail("Unmatched uploads cannot read a store"))
    assert report_for_saved_run(settings, source_run_id, payload) is None


@pytest.mark.parametrize("label", ["missing", True])
def test_unscored_local_cohort_preserves_unknown_labels(saved_run, label):
    from autofa_library_server.lazy_report import build_lazy_report

    _, store, _, _ = saved_run
    path = store / "dataset.parquet"
    row, = pq.read_table(path).to_pylist()
    row.pop("is_correct")
    if label != "missing":
        row["is_correct"] = label
    pq.write_table(pa.Table.from_pylist([row]), path)
    manifest = json.loads((store / "manifest.json").read_text())
    manifest["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (store / "manifest.json").write_text(json.dumps(manifest))
    report, _ = build_lazy_report(
        FunctionLibrary(model_id="local-model", benchmark="local-bench"),
        [make_local_reference_data(store_id=STORE, run_id=RUN)], source_harness="unknown",
    )
    meta = _extract_autofa_data(report.decode())["meta"]
    assert meta["n_items"] == 1
    if label == "missing":
        assert meta["n_failures"] is None and meta["base_fail_rate"] is None
    else:
        assert meta["n_failures"] == int(not label)
        assert meta["base_fail_rate"] == float(not label)


def test_renderer_change_creates_new_snapshot_and_asset_urls(saved_run, tmp_path, monkeypatch):
    from autofa_library_server import lazy_report

    settings, _, _, iid = saved_run
    renderers = tuple(tmp_path / path.name for path in lazy_report.renderer_files())
    for renderer in renderers:
        renderer.write_text("original renderer")
    monkeypatch.setattr(lazy_report, "renderer_files", lambda: renderers)
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        first, original = _catalog_report(app, client)
        # Any renderer module, not only the entry point, identifies the snapshot.
        (tmp_path / "recorded_relationships.py").write_text("updated renderer implementation")
        second, updated = _catalog_report(app, client)
        assert second["latest_version_id"] != first["latest_version_id"]
        assert second["version_count"] == first["version_count"] + 1
        assert updated["meta"]["items_base"] != original["meta"]["items_base"]
        assert updated["meta"]["local_report_renderer"] != original["meta"]["local_report_renderer"]
        for library, data in ((first, original), (second, updated)):
            asset = app.state.repository.get_report_asset(
                library["latest_version_id"], "failure_cards_data/items/" + data["item_index"][iid],
            )
            descriptor = json.loads(app.state.artifact_store.get(
                asset["artifact_key"].removeprefix(KEY_PREFIX),
            ))
            assert descriptor["renderer"] == data["meta"]["local_report_renderer"]
        third, _ = _catalog_report(app, client)
        assert third["latest_version_id"] == second["latest_version_id"]


# Publication to a standalone server process with its own profile.


@pytest.fixture
def receiver(tmp_path_factory):
    """A separate process can see only the stores explicitly sent over HTTP."""
    home = tmp_path_factory.mktemp("standalone-receiver")
    token_path = home / "server" / "token"
    token_path.parent.mkdir()
    token_path.write_text(SERVER_TOKEN)
    token_path.chmod(0o600)
    with socket.socket() as socket_:
        socket_.bind(("0.0.0.0", 0))
        port = socket_.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    environment = {
        **os.environ,
        "AUTOFA_HOME": str(home),
        "AUTOFA_CONFIG": str(home / "config.toml"),
        "PYTHONPATH": str(Path(autofa.__file__).resolve().parents[1]),
        "PYTHONUNBUFFERED": "1",
    }
    log = home / "server.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "autofa", "serve", "--host", "0.0.0.0", "--port", str(port)],
            cwd=home, env=environment, stdout=output, stderr=subprocess.STDOUT,
        )
        client = RemoteLibraryClient(url, token=SERVER_TOKEN, credential_source="server", timeout=1)
        try:
            deadline = time.monotonic() + 20
            while True:
                assert process.poll() is None, log.read_text()
                # Wait for bind before connecting: a readiness probe can claim
                # the released ephemeral port as its own source port on macOS.
                if "Uvicorn running on" not in log.read_text():
                    if time.monotonic() >= deadline:
                        pytest.fail("Standalone server did not bind: " + log.read_text())
                    time.sleep(0.05)
                    continue
                try:
                    assert client.me().auth_method == "local_token"
                    break
                except RemoteLibraryError:
                    if time.monotonic() >= deadline:
                        pytest.fail("Standalone server did not become ready: " + log.read_text())
                    time.sleep(0.05)
            yield SimpleNamespace(home=home, url=url, client=client, log=log)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture
def sender(saved_run, receiver, monkeypatch):
    _, store, library, item_id = saved_run
    home = store.parent.parent
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_harness"] = "gym"
    manifest_path.write_text(json.dumps(manifest))
    reference = json.loads((store / "source.json").read_text())
    reference["source_harness"] = "gym"
    (store / "source.json").write_text(json.dumps(reference))
    run = RunConfig(run_id="saved", parquet_paths=[str(store / "dataset.parquet")],
                    model_id="local-model", benchmarks=["local-bench"], source_harness="gym",
                    reference_data=[reference])
    run.save()
    def persist_library():
        path = run.run_dir() / "library.json"
        library.save(path)
        return path

    runtime = SimpleNamespace(library=library, dataset=Dataset(run.parquet_paths),
                              persist_library=persist_library)
    config = Config(remote_library_mode="server", remote_library_base_url=receiver.url,
                    remote_library_token_env="AUTOFA_LIBRARY_TOKEN")
    monkeypatch.setenv("AUTOFA_LIBRARY_TOKEN", SERVER_TOKEN)
    # A server reached through localhost must use its supplied credential, not
    # a different dashboard token belonging to the publishing profile.
    token_path = home / "server" / "token"
    token_path.parent.mkdir(exist_ok=True)
    token_path.write_text("sender-profile-token-" + "y" * 40)
    return SimpleNamespace(home=home, store=store, library=library, item_id=item_id, run=run,
                           runtime=runtime, config=config)


def _publish(sender):
    return publish_run(sender.run, sender.config, trigger="manual_cli", runtime=sender.runtime,
                       show_progress=False)


def _report(receiver, publication):
    report = receiver.client._request("GET", f"/libraries/{publication.library_id}/report")
    return _extract_autofa_data(report.decode())


def test_wrong_token_is_rejected_then_full_store_publishes_to_another_profile_and_is_reused(
    sender, receiver, monkeypatch,
):
    assert sender.home != receiver.home
    iterate = TraceArchive.__iter__
    with monkeypatch.context() as wrong:
        wrong.setenv("AUTOFA_LIBRARY_TOKEN", "wrong-server-token")

        def forbidden(*args, **kwargs):
            pytest.fail("Do not stream any trace data before authenticating the receiver")

        wrong.setattr(TraceArchive, "__iter__", forbidden)
        with pytest.raises(RemoteLibraryError) as error:
            _publish(sender)
        assert error.value.status == 401
        assert not (receiver.home / "sources").exists()
        assert receiver.client._request_json("GET", "/api/v2/libraries")["items"] == []
    # A captured 65 MiB artifact bypasses only the report ZIP limit, not auth.
    size = 65 * 1024 * 1024
    large = sender.store / "artifacts" / "large-temporary"
    digest = hashlib.sha256()
    piece = b"captured local artifact\n" * 40000
    with large.open("wb") as output:
        for start in range(0, size, len(piece)):
            output.write(piece[:size - start])
            digest.update(piece[:size - start])
    large_name = digest.hexdigest()
    large.rename(large.with_name(large_name))
    manifest_path = sender.store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_files"].append({"task_id": "large-artifact", "repeat": 0,
                                       "artifact_idx": 99, "sha256": large_name,
                                       "path": f"artifacts/{large_name}"})
    manifest_path.write_text(json.dumps(manifest))
    sent_archives = []

    def streamed(archive):
        sent_archives.append(archive.size)
        for chunk in iterate(archive):
            assert len(chunk) <= 1024 * 1024
            yield chunk

    monkeypatch.setattr(TraceArchive, "__iter__", streamed)
    client = client_from_config(sender.config)
    assert client.token == SERVER_TOKEN and client.credential_source == "server"
    published = _publish(sender)
    assert sent_archives[0] > 64 * 1024 * 1024
    transferred = receiver.home / "sources" / STORE
    received = transferred / "artifacts" / large_name
    assert received.stat().st_size == size and file_digest(received) == large_name
    manifest = json.loads((transferred / "manifest.json").read_text())
    source = json.loads((transferred / "source.json").read_text())
    assert source["parquet_path"] == str(transferred / "dataset.parquet")
    assert str(sender.home) not in json.dumps(source)
    assert not (receiver.home / "runs" / "saved").exists()
    assert len(sent_archives) == 1
    before = {path.relative_to(transferred): (path.stat().st_ino, path.stat().st_mtime_ns)
              for path in transferred.rglob("*") if path.is_file()}
    sender.library.functions[0].description = "Updated report description"
    repeated = _publish(sender)
    assert repeated.library_id == published.library_id
    assert repeated.branch_id == published.branch_id
    assert repeated.version_id != published.version_id
    assert len(sent_archives) == 1
    assert before == {
        path.relative_to(transferred): (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in transferred.rglob("*") if path.is_file()
    }

    # Removing all original locations catches accidental absolute-path reuse.
    sender.home.rename(sender.home.with_name(sender.home.name + "-disconnected"))
    data = _report(receiver, repeated)
    assert data["meta"]["local_lazy_items"] and not data["items"]
    item_url = data["meta"]["items_base"] + "/" + data["item_index"][sender.item_id]
    item = receiver.client._request_json("GET", item_url)
    assert "TAIL EVIDENCE" in item["response_raw"]
    passage = item["localisations"]["detector"]["passages"][0]
    location = passage["trace_locations"][0]
    assert location["source_event"] == {"event_idx": 55, "part_idx": 7, "lane": "conversation"}
    assert location["source_url"].startswith(f"/traces/{STORE}/#/")
    assert "text=%F0%9F%99%82%20TAIL%20EVIDENCE" in location["source_url"]
    full = receiver.client._request_json(
        "GET", _base() + "/part-text?event_idx=55&part_idx=7&lane=conversation",
    )
    assert full["text"] == TEXT
    artifact = receiver.client._request("GET", _base() + "/artifacts/0/download")
    assert artifact.startswith(b"\x89PNG")
    assert hashlib.sha256(artifact).hexdigest() == manifest["artifact_files"][0]["sha256"]
    assert receiver.client._request_json("GET", f"/api/v2/trace-stores/{STORE}")["run_id"] == RUN


def test_mixed_managed_and_ordinary_inputs_remain_inspectable_after_transfer(sender, receiver):
    """Inputs only partly covered by sent stores keep the full report for the rest."""
    generic = sender.home / "ordinary.parquet"
    text = "Ordinary trace " * 1000 + "END OF ORDINARY TRACE"
    pq.write_table(pa.Table.from_pylist([{
        "model_id": "local-model", "benchmark": "local-bench", "problem_id": "ordinary-task",
        "run_idx": 0, "is_correct": False, "response_text": text,
    }]), generic)
    generic_id = next(Dataset([generic]).iter_batches())[0]["_item_id"]
    sender.run.parquet_paths.append(str(generic))
    sender.runtime.dataset = Dataset(sender.run.parquet_paths)
    function = sender.library.functions[0]
    function.example_fail_ids.append(generic_id)
    function.scores.update(n_items=2, a=2)
    function.scores["buckets"] = [{"bucket": "True", "n": 2, "n_fail": 2, "n_pass": 0,
                                   "example_fail_ids": list(function.example_fail_ids)}]
    published = _publish(sender)
    sender.home.rename(sender.home.with_name(sender.home.name + "-disconnected"))
    data = _report(receiver, published)

    def item(item_id):
        if item_id in data["items"]:
            return data["items"][item_id]
        return receiver.client._request_json(
            "GET", data["meta"]["items_base"] + "/" + data["item_index"][item_id],
        )

    assert item(generic_id)["response_head"] == text
    assert "TAIL EVIDENCE" in item(sender.item_id)["response_raw"]
    assert item(sender.item_id)["source_data_url"].startswith(f"/traces/{STORE}/")


# Cloud reports link exact source rollouts without reading/uploading their text.


def _forbid_text_reads(monkeypatch, dataset, *, point_reads=True):
    """Allow only identity-column scans of the dataset."""
    original_batches = dataset.iter_batches

    def identity_batches(**kwargs):
        assert kwargs.get("columns"), "Publication must not load full trajectories"
        assert not {"prompt_text", "response_text", "reasoning_text"} & set(kwargs["columns"])
        return original_batches(**kwargs)

    monkeypatch.setattr(dataset, "iter_batches", identity_batches)
    if point_reads:
        monkeypatch.setattr(dataset, "read_locations", lambda *a, **k: pytest.fail("full-row read"))


def _remote(branch):
    return RemotePublication(library_id="library", branch_id="branch", version_id="version",
                             branch_key=branch)


def _saved_runtime(run, dataset, library):
    def persist_library():
        path = run.run_dir() / "library.json"
        library.save(path)
        return path

    return SimpleNamespace(library=library, dataset=dataset, persist_library=persist_library)


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    paths, references = [], []
    for run_id in ("run-a", "run-b"):
        path = tmp_path / f"{run_id}.parquet"
        pq.write_table(pa.Table.from_pylist([{
            "model_id": "m", "benchmark": "b", "problem_id": "suite/task", "run_idx": 0,
            "is_correct": False, "prompt_text": "PRIVATE_PROMPT",
            "response_text": "PRIVATE_TRAJECTORY " + run_id,
            "reasoning_text": "PRIVATE_REASONING",
        }]), path)
        paths.append(path)
        references.append(make_reference_data(
            run_id=run_id, parquet_path=path,
            export_url=canonical_export_url("https://runs.example.test", run_id),
        ))
    dataset = Dataset(paths, model_id="m", benchmarks=["b"])
    ids = [r["_item_id"] for batch in dataset.iter_batches() for r in batch]
    library = FunctionLibrary(model_id="m", benchmark="b", functions=[FeatureFunction(
        name="detector", source="def detector(item, llm_client=None):\n    return True\n",
        description="Detects failure", example_fail_ids=ids,
        scores={"n_items": 2, "a": 2, "c": 0, "recall": 1, "precision": 1,
                "ppv_at_pi0": 1, "behavior_purity": .8, "behavior_specificity": .8},
    )])
    run = RunConfig(run_id="local-run", model_id="m", benchmarks=["b"],
                    parquet_paths=[str(p) for p in paths], reference_data=references)
    return run, dataset, library, ids


def test_publication_links_every_source_without_loading_text(source_run, monkeypatch):
    run, dataset, library, ids = source_run
    _forbid_text_reads(monkeypatch, dataset, point_reads=False)
    local_report = run.run_dir() / "failure_cards.html"
    local_report.write_text("existing local report")

    uploaded = []

    def upload(**kwargs):
        assert kwargs["report_assets_zip"] is None
        assert kwargs["metadata"]["data_identity"] == {"version": 1, "sources": [
            {"kind": "evalrdl", "origin": "https://runs.example.test", "run_id": name}
            for name in ("run-a", "run-b")
        ]}
        html = Path(kwargs["failure_cards_html"]).read_text()
        data = _extract_autofa_data(html)
        assert not data["item_index"]
        assert not data["meta"]["items_base"]
        assert set(data["items"]) == set(ids)
        for iid, source in zip(ids, ("run-a", "run-b"), strict=True):
            item = data["items"][iid]
            assert item["source_only"] is True
            assert item["source_data_url"] == (
                f"https://runs.example.test/#/rollout/{source}/suite%252Ftask/0"
            )
        assert "PRIVATE_" not in html
        assert str(dataset.parquet_paths[0].parent) not in html
        uploaded.append(data)
        return _remote(kwargs["metadata"]["branch_key"])

    publish_run(run, Config(), trigger="session_end", show_progress=False,
                runtime=_saved_runtime(run, dataset, library),
                client=SimpleNamespace(upload_library=upload))
    assert len(uploaded) == 1
    assert local_report.read_text() == "existing local report"
    assert run.remote_version_id == "version"
    assert published_library_is_current(run, Config())
    receipt = json.loads((run.run_dir() / "publication.json").read_text())
    assert [row["run_id"] for row in receipt["data_identity"]["sources"]] == ["run-a", "run-b"]
    library.benchmark = "changed after publication"
    _saved_runtime(run, dataset, library).persist_library()
    assert not published_library_is_current(run, Config())


def test_publication_marks_incomplete_source_identity_instead_of_using_remote_subset(source_run):
    run, dataset, library, _ = source_run
    run.parquet_paths.append(str(run.run_dir() / "unavailable-local-input.parquet"))

    def upload(**kwargs):
        assert len(kwargs["metadata"]["reference_data"]) == 2
        assert "data_identity" in kwargs["metadata"]
        assert kwargs["metadata"]["data_identity"] is None
        return RemotePublication(library_id="isolated-library", branch_id="branch",
                                 version_id="version", branch_key=kwargs["metadata"]["branch_key"])

    publish_run(run, Config(), trigger="manual_cli", show_progress=False,
                runtime=_saved_runtime(run, dataset, library),
                client=SimpleNamespace(upload_library=upload))


@pytest.mark.parametrize("trigger,runtime_source", [("manual_cli", "active"),
                                                    ("session_end", "saved_run")])
def test_publication_preserves_incomplete_validation_without_running_evaluation(
    source_run, monkeypatch, trigger, runtime_source,
):
    from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
    from autofa.metrics.claude_judge import judge_policy_hash
    from autofa.session.exploration import resolve_exploration_seed

    run, dataset, library, _ = source_run
    # The scored LLM keeper is still sampled; the other candidates contain
    # completed uncertainty, processing errors, stale judgments, or no review.
    keeper = library.functions[0]
    keeper.kind = "llm"
    keeper.scores.update(evaluated=True, eval_mode="sample", sampled=True)
    for state in ("uncertain", "processing_error", "stale", "missing"):
        fn = deepcopy(keeper)
        fn.name = state
        fn.kind = "regex"
        fn.mechanism = "Removing the backup makes the subsequent write failure unrecoverable."
        fn.scores.update(
            eval_mode="full", sampled=False,
            behavior_purity=None, behavior_specificity=None,
            judge_contract_version=JUDGE_CONTRACT_VERSION,
            judge_policy_hash=judge_policy_hash(),
            claim_hash=claim_hash(
                source=fn.source, description=fn.description, mechanism=fn.mechanism,
            ),
            behavior_n_judged=2, behavior_n_yes=0, behavior_n_no=0,
            behavior_n_uncertain=2 if state == "uncertain" else 0,
            behavior_n_errors=2 if state == "processing_error" else 0,
        )
        if state == "stale":
            fn.scores.update(behavior_n_yes=2, behavior_purity=1, behavior_specificity=.8)
            fn.mechanism = "A changed mechanism that has not been validated."
        elif state == "missing":
            fn.scores = {}
        library.functions.append(fn)
    expected = deepcopy(library.to_dict())
    expected["exploration_state"] = {
        "version": 1, "base_seed": resolve_exploration_seed(run_id=run.run_id),
        "sequence_index": 0,
    }
    cfg = Config()
    rt = Runtime(dataset=dataset, library=library, run=run, config=cfg)
    rt.persist_library()
    run.save()
    monkeypatch.setattr("autofa.mcp.runtime._RUNTIME", None)

    def unexpected_work(*args, **kwargs):
        pytest.fail("Publishing saved results must not evaluate detectors or call the judge")

    for operation in ("evaluate_functions", "_judge_function"):
        monkeypatch.setattr(f"autofa.mcp.tools_eval.{operation}", unexpected_work)

    uploads = []

    def upload(**kwargs):
        assert json.loads(Path(kwargs["library_json"]).read_text()) == expected
        data = _extract_autofa_data(Path(kwargs["failure_cards_html"]).read_text())
        assert data["meta"]["n_items"] == data["meta"]["n_failures"] == 2
        # Publication retains every saved candidate and review above, while the
        # report displays only OK-or-better findings. Missing, stale and unresolved
        # reviews remain in the library without becoming active report findings.
        assert data["meta"]["n_functions"] == 1
        assert data["meta"]["n_pending_scores"] == 1
        assert data["meta"]["n_hidden_functions"] == 4
        assert {fn["name"] for fn in data["functions"]} == {
            "detector", "uncertain", "processing_error", "stale", "missing",
        }
        stale = next(fn for fn in data["functions"] if fn["name"] == "stale")
        assert stale["behavior_validation"]["status"] == "stale"
        assert stale["metrics"]["behavior_purity"] == -1
        assert stale["verdict"]["code"] == "UNSCORED"
        assert data["functions"][0]["metrics"]["behavior_purity"] == .8
        assert kwargs["metadata"]["publish_trigger"] == trigger
        uploads.append(data)
        return _remote(kwargs["metadata"]["branch_key"])

    for _ in range(2):
        publish_run(
            run, cfg, trigger=trigger, show_progress=False,
            runtime=rt if runtime_source == "active" else None,
            client=SimpleNamespace(upload_library=upload),
        )
        assert library.to_dict() == expected
        assert json.loads(rt.working_library_path().read_text()) == expected
        status = json.loads((run.run_dir() / "publication.json").read_text())
        assert status["state"] == "published" and status["trigger"] == trigger
    assert len(uploads) == 2


def test_publication_after_evidence_policy_upgrade_keeps_saved_detectors(source_run, monkeypatch):
    from autofa.metrics import claude_judge
    from autofa.metrics.claim_contract import (
        JUDGE_CONTRACT_VERSION,
        build_claim_contract,
        claim_hash,
    )

    run, dataset, library, _ = source_run
    fn = library.functions[0]
    with monkeypatch.context() as old_policy:
        old_policy.setattr(
            claude_judge, "EVIDENCE_VERSION", "judge-evidence-v10-bounded-completeness",
        )
        saved_hash = claude_judge.judge_policy_hash()
    assert saved_hash != claude_judge.judge_policy_hash()
    fn.scores.update(
        evaluated=True, eval_mode="full", sampled=False,
        judge_contract_version=JUDGE_CONTRACT_VERSION,
        judge_policy_hash=saved_hash,
        claim_hash=claim_hash(source=fn.source, description=fn.description,
                              mechanism=fn.description),
        claim_contract=build_claim_contract(source=fn.source, description=fn.description,
                                            mechanism=fn.description),
        behavior_n_judged=2, behavior_n_completed=2, behavior_n_yes=2, behavior_n_no=0,
        behavior_n_uncertain=0, behavior_n_errors=0,
    )
    cfg = Config()
    rt = Runtime(dataset=dataset, library=library, run=run, config=cfg)
    rt.persist_library()
    run.save()
    before = deepcopy(library.to_dict())

    def unexpected_work(*args, **kwargs):
        pytest.fail("Publishing must not evaluate or rejudge historical results")

    for operation in ("evaluate_functions", "_judge_function"):
        monkeypatch.setattr(f"autofa.mcp.tools_eval.{operation}", unexpected_work)

    uploads = []

    def upload(**kwargs):
        assert json.loads(Path(kwargs["library_json"]).read_text()) == before
        data = _extract_autofa_data(Path(kwargs["failure_cards_html"]).read_text())
        assert data["meta"]["n_functions"] == 1
        assert data["meta"]["n_pending_scores"] == 0
        card, = data["functions"]
        assert card["metrics"]["behavior_purity"] == fn.scores["behavior_purity"]
        assert card["behavior_validation"]["status"] == "historical"
        assert not card["behavior_validation"]["ranking_eligible"]
        assert card["verdict"]["code"] == "OK"
        uploads.append(data)
        return _remote(kwargs["metadata"]["branch_key"])

    publish_run(run, cfg, trigger="manual_cli", show_progress=False,
                client=SimpleNamespace(upload_library=upload))
    assert len(uploads) == 1
    assert json.loads(rt.working_library_path().read_text()) == before


@pytest.mark.parametrize("reference_kind", ["missing", "unbound"])
def test_standalone_report_keeps_unlinked_trajectories(source_run, reference_kind):
    run, dataset, library, ids = source_run
    references = [dict(r) for r in run.reference_data]
    if reference_kind == "missing":
        references.pop()
    else:
        del references[1]["parquet_path"]
    report = write_failure_cards(
        library, out=run.run_dir() / "failure_cards_cloud.html", dataset=dataset,
        reference_data=references, prefer_source_links=True,
    )
    data = _extract_autofa_data(report.read_text())
    assert set(data["items"]) == {ids[0]}
    assert set(data["item_index"]) == {ids[1]}
    assert data["meta"]["n_embedded_items"] == 2
    payload = build_report_assets_zip(report.parent, html_name=report.name)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert len(archive.namelist()) == 1
        item = json.loads(archive.read(archive.namelist()[0]))
        assert item["response_head"] == "PRIVATE_TRAJECTORY run-b"
        assert not item["source_data_url"]


@pytest.mark.parametrize("service_url", ["https://library.example.test", "http://127.0.0.1:8765"])
def test_publication_of_local_import_keeps_only_saved_evidence(
    source_run, monkeypatch, service_url,
):
    from autofa.core.localisation import detector_hash
    from autofa.data.reference_data import make_local_reference_data

    run, dataset, library, ids = source_run
    run.reference_data = [
        make_local_reference_data(store_id=letter * 64, run_id=f"local-{letter}", parquet_path=path)
        for letter, path in zip(("a", "b"), dataset.parquet_paths, strict=True)
    ]
    function = library.functions[0]
    function.failure_localisations = {ids[0]: {
        "detector_hash": detector_hash(function.source),
        "passages": [{"quote": "SAVED_EVIDENCE", "field": "response_text"}],
    }}
    _forbid_text_reads(monkeypatch, dataset)
    captured = []

    def upload(**kwargs):
        assert kwargs["metadata"]["reference_data"] == []
        html = Path(kwargs["failure_cards_html"]).read_text()
        data = _extract_autofa_data(html)
        assert data["meta"]["reference_data"] == []
        assert not data["item_index"]
        assert not data["meta"]["items_base"]
        assert set(data["items"]) == set(ids)
        assert all(item["source_only"] and not item["source_data_url"]
                   for item in data["items"].values())
        assert kwargs["report_assets_zip"] is None
        assert "SAVED_EVIDENCE" in html
        assert "PRIVATE_" not in html
        assert "local_evalrdl" not in html
        assert all(ref["store_id"] not in html for ref in run.reference_data)
        assert str(dataset.parquet_paths[0].parent) not in html
        captured.append(data)
        return _remote(kwargs["metadata"]["branch_key"])

    # Progress output is the CLI default; its report writer must keep the source-only flags.
    publish_run(run, Config(remote_library_base_url=service_url),
                trigger="session_end", show_progress=True,
                runtime=_saved_runtime(run, dataset, library),
                client=SimpleNamespace(upload_library=upload))
    assert len(captured) == 1
    assert all(ref["kind"] == "local_evalrdl" for ref in run.reference_data)


def test_cloud_report_never_loads_unlinked_trajectories(source_run, monkeypatch):
    run, dataset, library, ids = source_run
    references = [dict(r) for r in run.reference_data]
    references[1]["export_url"] = "https://runs.example.test/not-an-export"
    _forbid_text_reads(monkeypatch, dataset)
    report = write_failure_cards(
        library, out=run.run_dir() / "failure_cards_cloud.html", dataset=dataset,
        reference_data=references, source_links_only=True,
    )
    html = report.read_text()
    data = _extract_autofa_data(html)
    assert set(data["items"]) == set(ids)
    assert not data["item_index"]
    assert build_report_assets_zip(report.parent, html_name=report.name) is None
    assert all(item["source_only"] for item in data["items"].values())
    assert data["items"][ids[0]]["source_data_url"]
    assert not data["items"][ids[1]]["source_data_url"]
    assert "PRIVATE_" not in html
    assert data["meta"]["n_items"] == 2
    assert data["functions"][0]["name"] == "detector"


# Only current-version event links are trusted without the trace; version 2 is not.
@pytest.mark.parametrize("navigation_version", [None, 2, NAVIGATION_VERSION])
def test_cloud_report_uses_saved_event_links_without_reading_trace(
    source_run, monkeypatch, navigation_version,
):
    from autofa.core.localisation import detector_hash

    run, dataset, library, ids = source_run
    function = library.functions[0]
    record = {
        "detector_hash": detector_hash(function.source),
        "passages": [{"quote": "SAVED_EVIDENCE", "field": "response_text", "trace_locations": [{
            "source_event": {"event_idx": 8, "part_idx": 1, "lane": "agent"},
            "quote": "SAVED_EVIDENCE",
        }]}],
    }
    if navigation_version is not None:
        record["navigation_version"] = navigation_version
    function.failure_localisations = {ids[0]: record}
    monkeypatch.setattr(dataset, "read_locations", lambda *a, **k: pytest.fail("full-row read"))
    report = write_failure_cards(
        library, out=run.run_dir() / "failure_cards_cloud.html", dataset=dataset,
        reference_data=run.reference_data, source_links_only=True,
    )
    data = _extract_autofa_data(report.read_text())
    passage = data["functions"][0]["failure_localisations"][ids[0]]["passages"][0]
    assert passage["quote"] == "SAVED_EVIDENCE"
    if navigation_version != NAVIGATION_VERSION:
        assert passage["trace_locations"] == []
        assert data["items"][ids[0]]["source_data_url"]
    else:
        assert passage["trace_locations"][0]["source_url"].endswith(
            "/e8-p1?lane=agent&text=SAVED_EVIDENCE"
        )


def test_cloud_report_without_dataset_keeps_cards_and_unavailable_examples(source_run):
    _, _, library, ids = source_run
    data = build_data(
        library, source_links_only=True,
        items={ids[0]: {"response_head": "PRIVATE_FULL_TRACE", "source_data_url": "/traces/local"}},
    )
    assert set(data["items"]) == set(ids)
    assert all(item["source_only"] and not item["source_data_url"]
               for item in data["items"].values())
    assert data["meta"]["n_items"] == 2
    assert data["functions"][0]["name"] == "detector"
    assert "PRIVATE_" not in json.dumps(data)


def test_local_publication_of_generic_parquet_retains_portable_trajectories(source_run):
    run, dataset, library, ids = source_run
    run.reference_data = []
    captured = []

    def upload(**kwargs):
        html = Path(kwargs["failure_cards_html"]).read_text()
        data = _extract_autofa_data(html)
        assert set(data["item_index"]) == set(ids)
        with zipfile.ZipFile(io.BytesIO(kwargs["report_assets_zip"])) as archive:
            assert len(archive.namelist()) == 2
            assert all("PRIVATE_TRAJECTORY" in archive.read(name).decode()
                       for name in archive.namelist())
        captured.append(data)
        return _remote(kwargs["metadata"]["branch_key"])

    publish_run(run, Config(remote_library_base_url="http://127.0.0.1:8765"),
                trigger="manual_cli", show_progress=False,
                runtime=_saved_runtime(run, dataset, library),
                client=SimpleNamespace(upload_library=upload))
    assert len(captured) == 1
