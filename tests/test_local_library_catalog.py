# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dashboard discovers local saves without modifying the original run files."""

from __future__ import annotations

import json
import logging
import os
import threading
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autofa.config import Config
from autofa.core.library import FunctionLibrary
from autofa.data.reference_data import make_local_reference_data
from autofa.reports.failure_cards import _extract_autofa_data, build_data, render_html
from autofa.session.status_log import record_turn_status
from autofa_library_server.catalog import LocalCatalog
from autofa_library_server.config import Settings
from autofa_library_server.main import create_app
from autofa_library_server.storage import FilesystemArtifactStore

DATA_IDENTITY = {"version": 1, "sources": [{"kind": "local", "sha256": "a" * 64}]}


def save_library(path: Path, *, model="saved-model", benchmark="test", description="First"):
    payload = {
        "version": 1, "model_id": model, "benchmark": benchmark,
        "functions": [{
            "name": "saved_detector", "description": description,
            "source": "def detect(item):\n    return False\n",
            "scores": {"recall": .4, "precision": .9, "ppv_at_pi0": .9,
                       "behavior_purity": .8, "behavior_specificity": .8},
        }],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def save_provenance(path, digest=None, run_id="run-1"):
    payload = {"source_run_id": run_id}
    if digest is not None:
        payload["data_identity"] = {
            "version": 1, "sources": [{"kind": "local", "sha256": digest * 64}],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name("library.provenance.json").write_text(json.dumps(payload))


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(LocalCatalog, "refresh_interval", 0)
    return Settings(data_dir=tmp_path / "server", library_dir=tmp_path / "library")


def refresh_catalog(client):
    # Discovery is deliberately eventual: await one explicitly requested background
    # pass before asserting snapshot contents, never rely on request timing.
    completed = client.app.state.local_catalog.request_refresh()
    assert completed is not None and completed.wait(5)


def libraries(client, *, refresh=True):
    if refresh:
        refresh_catalog(client)
    response = client.get("/api/v2/libraries")
    assert response.status_code == 200, response.text
    return response.json()["items"]


def write_saved_report(library_path, *, items_base="items", item="example.json", body=""):
    """A saved report whose example index points at a sidecar asset next to the library."""
    report = library_path.with_name("failure_cards.html")
    report.write_text(
        '<html><script id="autofa-data" type="application/json">'
        + json.dumps({"meta": {"items_base": items_base}, "item_index": {"example": item}})
        + "</script>" + body + "</html>"
    )
    return report


def pause_catalog(monkeypatch, method, timeout=10):
    """Block ``LocalCatalog.<method>`` until released; ``started`` is set once it is entered."""
    started, release = threading.Event(), threading.Event()
    original = getattr(LocalCatalog, method)

    def paused(self, *args, **kwargs):
        started.set()
        assert release.wait(timeout)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LocalCatalog, method, paused)
    return started, release


def upload(client, settings, payload, report, **metadata):
    response = client.post(
        "/api/v2/publications", headers={"Authorization": f"Bearer {settings.local_token}"},
        data={"metadata": json.dumps({"branch_key": "run-1", "source_run_id": "run-1",
                                      "model_id": "saved-model", "benchmarks": ["test"],
                                      **metadata})},
        files={"library": ("library.json", json.dumps(payload), "application/json"),
               "failure_card": ("report.html", report, "text/html")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def version_path(library):
    return (
        f"/api/v2/libraries/{library['id']}/branches/{library['latest_branch_id']}"
        f"/versions/{library['latest_version_id']}"
    )


@pytest.mark.parametrize("source", ["evalrdl", "local"])
def test_catalog_groups_actual_source_data_instead_of_model_benchmark(settings, source):
    originals = {}
    for index, dataset_name in enumerate(("same-data", "same-data", "other-data")):
        path = settings.runs_dir / f"analysis-{index}" / "library.json"
        save_library(path, model="another-label" if index == 1 else "same-model")
        if source == "evalrdl":
            config = {"reference_data": [{
                "run_id": dataset_name,
                "export_url": f"https://eval.example/v1/runs/{dataset_name}/export/autofa",
            }]}
        else:
            # Different file locations with identical bytes must share a card.
            dataset = path.parent / "input.parquet"
            dataset.write_bytes(dataset_name.encode())
            config = {"parquet_paths": [str(dataset)]}
        path.with_name("config.json").write_text(json.dumps(config))
        for item in path.parent.iterdir():
            originals[item] = item.read_bytes()
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        items = libraries(client)
        assert len(items) == 2
        assert sorted(item["branch_count"] for item in items) == [1, 2]
        assert all(item["data_identity_status"] == "known" for item in items)
        assert len({json.dumps(item["data_identity"], sort_keys=True) for item in items}) == 2
    assert originals == {path: path.read_bytes() for path in originals}


@pytest.mark.parametrize(
    ("next_run", "next_digest", "expected_cards", "expected_branches"),
    [("run-b", "b", 2, 2), ("run-b", "a", 1, 2), ("run-a", "a", 1, 1)],
    ids=["new_run_new_data", "new_run_same_data", "same_run_same_data"],
)
def test_overwritten_durable_catalog_preserves_source_and_analysis_history(
    settings, next_run, next_digest, expected_cards, expected_branches,
):
    path = settings.library_dir / "saved-model" / "test" / "library.json"
    save_library(path)
    save_provenance(path, "a", "run-a")
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        original, = libraries(client)
        original_version = version_path(original)
        save_provenance(path, next_digest, next_run)
        save_library(path, description="Updated analysis")
        current = libraries(client)
        assert len(current) == expected_cards
        assert sum(item["branch_count"] for item in current) == expected_branches
        assert sum(item["version_count"] for item in current) == 2
        latest = max(current, key=lambda item: item["updated_at"])
        assert latest["source_run_id"] == next_run
        assert latest["data_identity"]["sources"][0]["sha256"] == next_digest * 64
        assert client.get(original_version).status_code == 200
        # Subsequent saves of the same analysis remain versions of its branch.
        save_library(path, description="Later update of the same analysis")
        current = libraries(client)
        assert len(current) == expected_cards
        assert sum(item["branch_count"] for item in current) == expected_branches
        assert sum(item["version_count"] for item in current) == 3


def test_catalog_identity_upgrade_preserves_unknown_analysis_branch(settings):
    path = settings.library_dir / "saved-model" / "test" / "library.json"
    save_library(path)
    save_provenance(path, run_id="run-a")
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        original, = libraries(client)
        assert original["data_identity_status"] == "unknown"
        save_provenance(path, "a", "run-a")
        current, = libraries(client)
        assert current["data_identity"] == DATA_IDENTITY
        assert current["id"] == original["id"]
        assert current["latest_branch_id"] == original["latest_branch_id"]
        assert current["branch_count"] == 1 and current["version_count"] == 2
        assert client.get(version_path(original)).status_code == 200


def test_status_refresh_preserves_enriched_snapshot_when_source_report_is_unavailable(settings):
    run = settings.runs_dir / "source-run"
    save_library(run / "library.json")
    report_path = run / "failure_cards.html"
    report_path.write_text(render_html({
        "meta": {"items_base": "items", "retained_marker": "original report"},
        "functions": [], "items": {}, "item_index": {"example": "example.json"},
    }))
    (run / "items").mkdir()
    (run / "items" / "example.json").write_text('{"text":"retained evidence"}')
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        initial, = libraries(client)
        report_path.unlink()
        record_turn_status(run, "interrupted")
        changed, = libraries(client)
        assert changed["latest_version_id"] != initial["latest_version_id"]
        response = client.get(version_path(changed) + "/failure-card")
        data = _extract_autofa_data(response.text)
        assert data["meta"]["retained_marker"] == "original report"
        assert data["meta"]["agent_execution"]["status"] == "interrupted"
        asset = app.state.repository.get_report_asset(
            changed["latest_version_id"], "items/example.json",
        )
        assert b"retained evidence" in app.state.artifact_store.get(asset["artifact_key"])


def test_local_trace_catalog_keeps_registry_provenance_and_embedded_report(settings):
    from autofa.data.library_provenance import CatalogProvenance, write_catalog_provenance

    library = settings.library_dir / "local-model" / "test" / "library.json"
    save_library(library, model="local-model")
    reference = make_local_reference_data(store_id="a" * 64, run_id="source-run")
    write_catalog_provenance(library, CatalogProvenance(
        source_run_id="analysis-run", reference_data=(reference,),
    ))
    for name, evidence in (("failure_cards_local.html", "embedded local evidence"),
                           ("failure_cards_cloud.html", "compact export")):
        library.with_name(name).write_text(render_html({
            "meta": {"reference_data": [reference], "model_id": evidence},
            "functions": [], "items": {},
        }))
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        item, = libraries(client)
        saved = client.get(version_path(item) + "/failure-card")
        payload = _extract_autofa_data(saved.text)
        assert payload["meta"]["reference_data"] == [reference]
        assert payload["meta"]["model_id"] == "embedded local evidence"
        # Dashboard template refresh also preserves the registry, without a pinned port.
        report = client.get(f"/libraries/{item['id']}/report")
        assert _extract_autofa_data(report.text)["meta"]["reference_data"] == [reference]


def test_catalog_uses_save_time_and_backfills_without_rerender(settings, monkeypatch):
    older = settings.runs_dir / "z-older" / "library.json"
    newer = settings.runs_dir / "a-newer" / "library.json"
    for path, model, timestamp in ((older, "older", 1700000000), (newer, "newer", 1800000000)):
        save_library(path, model=model)
        os.utime(path, (timestamp, timestamp))
    # A catalog copy's more recent mtime should not replace the original save time.
    catalog = settings.library_dir / "older" / "test" / "library.json"
    save_library(catalog, model="older")
    save_provenance(catalog, run_id="z-older")
    older.with_name("publication.json").write_text('{"state":"published","updated_at":"invalid"}')
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        items = libraries(client)
        assert [item["model_id"] for item in items] == ["newer", "older"]
        ids = {item["latest_version_id"] for item in items}
        # Clear the saved activity; the cached imports keep their report snapshots.
        with app.state.database.transaction() as connection:
            connection.execute("DELETE FROM local_source_activity")

    def no_rerender(*args, **kwargs):
        raise AssertionError("Backfilling dates must not regenerate reports")

    monkeypatch.setattr(LocalCatalog, "_report", no_rerender)
    with TestClient(create_app(settings), base_url=settings.public_url) as restarted:
        items = libraries(restarted)
        assert [item["model_id"] for item in items] == ["newer", "older"]
        assert {item["latest_version_id"] for item in items} == ids


def test_latest_library_uses_newest_publication_across_imported_branches(settings):
    for run_id, date, description in (
        ("a-newer", "2026-10-06T06:00:00Z", "Latest published detector"),
        ("z-older", "2026-07-30T14:00:00Z", "Older detector imported later"),
    ):
        path = settings.runs_dir / run_id / "library.json"
        save_library(path, description=description)
        save_provenance(path, "a", run_id)
        path.with_name("publication.json").write_text(json.dumps({
            "state": "published", "updated_at": date,
        }))
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        item = libraries(client)[0]
        assert item["source_run_id"] == "a-newer"
        assert item["branch_count"] == 2
        report = client.get(version_path(item) + "/library.json").json()
        assert report["functions"][0]["description"] == "Latest published detector"


def test_saved_catalogs_runs_and_corrupt_runs_are_discovered_read_only(settings):
    catalog = settings.library_dir / "durable-model" / "test" / "library.json"
    run_library = settings.runs_dir / "run-1" / "library.json"
    save_library(catalog, model="durable-model")
    saved = save_library(run_library)
    config = run_library.with_name("config.json")
    config.write_text(json.dumps({
        "run_id": "run-1", "benchmarks": ["test"], "harness": "test-harness",
        "remote_library_id": "previous-library-id", "remote_branch_id": "previous-branch-id",
    }))
    publication = run_library.with_name("publication.json")
    publication.write_text('{"state": "published", "version_id": "previous-version"}')
    corrupt = settings.runs_dir / "broken" / "library.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text('{"unfinished":')
    originals = {path: path.read_bytes() for path in (catalog, run_library, config, publication)}
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        items = libraries(client)
        assert {item["model_id"] for item in items} == {"durable-model", "saved-model"}
        run = next(item for item in items if item["model_id"] == "saved-model")
        assert client.get(version_path(run) + "/library.json").json() == saved
        assert "saved_detector" in client.get(f"/libraries/{run['id']}/report").text
        assert "durable-model" in client.get("/").text
        save_library(corrupt, model="repaired-model")
        assert {item["model_id"] for item in libraries(client)} == {
            "durable-model",
            "saved-model",
            "repaired-model",
        }
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert not catalog.with_name("failure_cards.html").exists()
    assert not run_library.with_name("failure_cards.html").exists()


def test_saved_updates_keep_old_versions_and_survive_restart(settings, monkeypatch):
    path = settings.runs_dir / "run-1" / "library.json"
    original = save_library(path)
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        first = libraries(client)[0]
        first_path = version_path(first)
        updated = save_library(path, description="Updated detector")
        second = libraries(client)[0]
        assert second["id"] == first["id"]
        assert second["version_count"] == 2
        assert second["latest_version_id"] != first["latest_version_id"]
        assert client.get(first_path + "/library.json").json() == original
        assert client.get(version_path(second) + "/library.json").json() == updated
        assert libraries(client)[0]["version_count"] == 2

    def no_rerender(*args, **kwargs):
        raise AssertionError("unchanged local files should not be rerendered on restart")

    monkeypatch.setattr(LocalCatalog, "_report", no_rerender)
    with TestClient(create_app(settings), base_url=settings.public_url) as restarted:
        assert libraries(restarted)[0]["version_count"] == 2
        path.unlink()
        assert libraries(restarted)[0]["version_count"] == 2
        assert restarted.get(first_path + "/library.json").json() == original


def test_custom_catalog_root_and_profile_runs_are_used(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    custom = tmp_path / "custom-catalog"
    monkeypatch.setenv("AUTOFA_HOME", str(profile))
    monkeypatch.setattr(LocalCatalog, "refresh_interval", 0)
    monkeypatch.setattr("autofa.config.load_config", lambda: Config(library_root=str(custom)))
    save_library(custom / "custom" / "test" / "library.json", model="custom")
    save_library(profile / "runs" / "mine" / "library.json", model="profile-run")
    save_library(tmp_path / "other-profile" / "runs" / "other" / "library.json", model="other")
    settings = Settings()
    assert settings.library_dir == custom
    assert settings.runs_dir == profile / "runs"
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        assert {item["model_id"] for item in libraries(client)} == {"custom", "profile-run"}


def test_saved_report_and_asset_changes_create_immutable_snapshots(settings):
    path = settings.runs_dir / "run-1" / "library.json"
    save_library(path)
    asset = path.parent / "items" / "example.json"
    asset.parent.mkdir()
    asset.write_text('{"steps": [{"text": "original evidence"}]}')
    report = write_saved_report(path, body="<body>Saved report</body>")
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        first = libraries(client)[0]
        assert "Saved report" in client.get(f"/libraries/{first['id']}/report").text
        original_asset = app.state.repository.get_report_asset(
            first["latest_version_id"], "items/example.json"
        )
        assert original_asset is not None
        asset.write_text('{"steps": [{"text": "updated evidence"}]}')
        second = libraries(client)[0]
        assert second["version_count"] == 2
        assert b"original evidence" in app.state.artifact_store.get(original_asset["artifact_key"])
        new_asset = app.state.repository.get_report_asset(
            second["latest_version_id"], "items/example.json"
        )
        assert b"updated evidence" in app.state.artifact_store.get(new_asset["artifact_key"])
        report.write_text("<html><body>New report</body></html>")
        assert libraries(client)[0]["version_count"] == 3


def test_durable_catalog_reuses_saved_cloud_report_on_restart(settings):
    path = settings.runs_dir / "run-1" / "library.json"
    library = save_library(path)
    cloud = path.with_name("failure_cards_cloud.html")
    data = build_data(FunctionLibrary.from_dict(library), items={"row-1": {
        "item_id": "row-1", "source_only": True,
        "source_data_url": "https://traces.example.test/#/rollout/source-run/task/0",
    }})
    cloud.write_text(render_html(data))
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        original = libraries(client)[0]
        saved = _extract_autofa_data(client.get(version_path(original) + "/failure-card").text)
        assert saved["items"] == data["items"]
    # The durable catalog reuses the richer snapshot from the run, rather than
    # replacing it with another empty report that happens to be imported later.
    catalog = settings.library_dir / "saved-model" / "test" / "library.json"
    save_provenance(catalog)
    catalog.write_text(json.dumps(library))
    with TestClient(create_app(settings), base_url=settings.public_url) as restarted:
        current = libraries(restarted)[0]
        assert current["id"] == original["id"]
        assert current["version_count"] == 1
        report = restarted.get(f"/libraries/{current['id']}/report").text
        assert _extract_autofa_data(report)["items"] == data["items"]
        assert libraries(restarted)[0]["version_count"] == 1
        # Revisit the durable copy after it already points at the enriched version.
        # This used to publish a placeholder because previous.version_id == latest.id.
        os.utime(catalog, ns=(catalog.stat().st_atime_ns, catalog.stat().st_mtime_ns + 1))
        assert libraries(restarted)[0]["version_count"] == 1
        assert _extract_autofa_data(
            restarted.get(f"/libraries/{current['id']}/report").text
        )["items"] == data["items"]
        # Deferring the same large saved report on a later scan must also preserve it.
        cloud.write_text(render_html(data) + " " * (1024 * 1024))
        libraries(restarted)
        count = libraries(restarted)[0]["version_count"]
        os.utime(cloud, ns=(cloud.stat().st_atime_ns, cloud.stat().st_mtime_ns + 1))
        assert libraries(restarted)[0]["version_count"] == count
        assert _extract_autofa_data(
            restarted.get(f"/libraries/{current['id']}/report").text
        )["items"] == data["items"]
        assert cloud.read_text() == render_html(data) + " " * (1024 * 1024)


@pytest.mark.parametrize("api_source", [False, True])
def test_catalog_selects_report_by_input_source(settings, api_source):
    path = settings.runs_dir / "run-1" / "library.json"
    save_library(path)
    if api_source:
        path.with_name("config.json").write_text(json.dumps({"reference_data": [{
            "run_id": "source-run",
            "export_url": "https://traces.example.test/v1/runs/source-run/export/autofa",
        }]}))
    local = path.with_name("failure_cards_local.html")
    local.write_text("<html><body>Full local trajectory</body></html>")
    cloud = path.with_name("failure_cards_cloud.html")
    cloud.write_text("<html><body>Source links only</body></html>")
    os.utime(cloud, (local.stat().st_mtime + 10, local.stat().st_mtime + 10))
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        item = libraries(client)[0]
        expected = cloud if api_source else local
        assert client.get(version_path(item) + "/failure-card").content == expected.read_bytes()


def test_catalog_reuses_reordered_packed_publication_but_keeps_changed_evidence_distinct(
    settings,
):
    from autofa.core.library_transport import pack_bucket_examples

    payload = {"version": 1, "model_id": "saved-model", "benchmark": "test", "functions": [
        {"name": name, "description": "Observable behavior", "kind": kind,
         "source": "def detect(item): return True",
         "scores": {"recall": coverage, "precision": .9, "ppv_at_pi0": .9,
                    "behavior_purity": 1, "behavior_specificity": 1,
                    "buckets": [{"bucket": "positive", "example_fail_ids": [name]}]},
         "failure_localisations": {name: {"passages": [{"quote": "original evidence"}]}}}
        for name, coverage, kind in (("narrow", .02, "python"), ("broad", .5, "python"))
    ]}
    outgoing = deepcopy(payload)
    outgoing["functions"].reverse()
    outgoing = pack_bucket_examples(outgoing)
    path = settings.runs_dir / "run-1" / "library.json"
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        assert libraries(client) == []
        uploaded = upload(client, settings, outgoing, "<html>Uploaded report</html>",
                          data_identity=DATA_IDENTITY)
        original = json.dumps(payload)
        save_provenance(path, "a")
        path.write_text(original)
        item, = libraries(client)
        assert item["version_count"] == 1
        assert item["latest_version_id"] == uploaded["version_id"]
        assert path.read_text() == original

        payload["functions"][0]["failure_localisations"]["narrow"]["passages"][0]["quote"] = (
            "changed evidence"
        )
        path.write_text(json.dumps(payload))
        revised, = libraries(client)
        assert revised["version_count"] == 2
        assert revised["latest_version_id"] != item["latest_version_id"]
        saved = client.get(version_path(revised) + "/library.json").json()
        assert saved["functions"][0]["failure_localisations"] == (
            payload["functions"][0]["failure_localisations"]
        )


@pytest.mark.parametrize("malformed", [
    None, {"functions": {}},
    {"functions": [{"name": "a", "description": "x", "source": "x"}] * 2},
    {"packed_bucket_examples": {"data": "corrupt"}},
], ids=["not_object", "functions_not_list", "duplicate_names", "corrupt_packing"])
def test_malformed_libraries_cannot_establish_same_content(malformed):
    from autofa_library_server.library_identity import same_library_content

    if isinstance(malformed, dict):
        malformed = {"version": 1, "model_id": "model", "functions": [], **malformed}
    assert not same_library_content(malformed, deepcopy(malformed))


def test_invalid_local_report_sidecars_do_not_escape_report_directory(settings):
    path = settings.runs_dir / "run-1" / "library.json"
    save_library(path)
    (path.parent.parent / "private.json").write_text('{"secret": "not-report-evidence"}')
    write_saved_report(path, items_base="..", item="private.json")
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        item = libraries(client)[0]
        report = client.get(f"/libraries/{item['id']}/report").text
        assert "saved_detector" in report
        assert "not-report-evidence" not in report


def test_changing_model_at_same_source_path_retains_both_libraries(settings):
    path = settings.runs_dir / "run-1" / "library.json"
    save_library(path, model="old-model")
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        assert libraries(client)[0]["model_id"] == "old-model"
        save_library(path, model="new-model")
        assert {item["model_id"] for item in libraries(client)} == {"old-model", "new-model"}


def test_catalog_report_generation_never_executes_detector_code(settings):
    path = settings.runs_dir / "run-1" / "library.json"
    payload = save_library(path)
    payload["functions"][0]["source"] = "raise RuntimeError('detector source must not execute')"
    path.write_text(json.dumps(payload))
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        item = libraries(client)[0]
        assert client.get(f"/libraries/{item['id']}/report").status_code == 200
        assert client.get(version_path(item) + "/library.json").json() == payload


def test_report_asset_replaced_during_indexing_is_refreshed(settings, monkeypatch):
    from autofa_library_server import catalog as catalog_module

    path = settings.runs_dir / "run-1" / "library.json"
    save_library(path)
    asset = path.parent / "items" / "example.json"
    asset.parent.mkdir()
    asset.write_text('{"text": "old evidence"}')
    write_saved_report(path)
    original_read = catalog_module._bounded_read
    replaced = False

    def replace_after_read(file_path, maximum):
        nonlocal replaced
        data = original_read(file_path, maximum)
        if file_path == asset and not replaced:
            replaced = True
            asset.write_text('{"text": "new evidence with a different size"}')
        return data

    monkeypatch.setattr(catalog_module, "_bounded_read", replace_after_read)
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        refresh_catalog(client)
        item = libraries(client)[0]
        record = app.state.repository.get_report_asset(
            item["latest_version_id"], "items/example.json"
        )
        assert b"new evidence" in app.state.artifact_store.get(record["artifact_key"])
        assert libraries(client)[0]["version_count"] == item["version_count"]


def test_newer_saved_report_is_not_hidden_by_matching_uploaded_library(settings):
    payload = {"version": 1, "model_id": "saved-model", "benchmark": "test", "functions": []}
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        upload(client, settings, payload, "<html>Old report</html>", data_identity=DATA_IDENTITY)
        path = settings.runs_dir / "run-1" / "library.json"
        save_provenance(path, "a")
        path.write_text(json.dumps(payload))
        path.with_name("failure_cards.html").write_text("<html>Updated saved report</html>")
        item = libraries(client)[0]
        assert item["version_count"] == 2
        assert "Updated saved report" in client.get(f"/libraries/{item['id']}/report").text


def test_saved_kind_is_indexed_unchanged_and_retired_kind_keeps_saved_snapshot(settings):
    path = settings.library_dir / "saved-model" / "test" / "library.json"
    payload = save_library(path)
    payload["functions"][0]["kind"] = "python"
    path.write_bytes(json.dumps(payload).encode())
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        item, = libraries(client)
        assert item["function_count"] == 1
        library = client.get(version_path(item) + "/library.json").json()
        assert library["functions"][0]["kind"] == "python"
        # A retired pre-GA label is not indexed; the saved snapshot remains served.
        payload["functions"][0]["kind"] = "hybrid"
        retired = json.dumps(payload).encode()
        path.write_bytes(retired)
        assert [(row["id"], row["latest_version_id"]) for row in libraries(client)] == [
            (item["id"], item["latest_version_id"]),
        ]
        assert client.get(version_path(item) + "/library.json").json() == library
        assert client.get(f"/libraries/{item['id']}/report").status_code == 200
    assert path.read_bytes() == retired


def test_unindexable_saved_library_is_warned_about_once_per_saved_file(settings, caplog):
    path = settings.library_dir / "saved-model" / "test" / "library.json"
    payload = save_library(path)
    payload["functions"][0]["kind"] = "hybrid"
    path.write_text(json.dumps(payload))
    with (caplog.at_level(logging.WARNING, logger="autofa_library_server.catalog"),
          TestClient(create_app(settings), base_url=settings.public_url) as client):
        assert libraries(client) == []
        assert libraries(client) == []
        warnings = [record for record in caplog.records
                    if record.getMessage().startswith("Unable to index local library")]
        assert len(warnings) == 1 and warnings[0].exc_info is None
        assert "Invalid function kind 'hybrid' for detector 'saved_detector'" in (
            warnings[0].getMessage()
        )
        # A changed save is indexed on the next pass.
        payload["functions"][0]["kind"] = "regex"
        path.write_text(json.dumps(payload))
        item, = libraries(client)
        assert item["function_count"] == 1


def test_local_library_size_limit_is_independent_of_upload_limit(settings):
    from dataclasses import replace

    save_library(settings.runs_dir / "run-1" / "library.json")
    settings = replace(settings, max_library_bytes=10)
    with TestClient(create_app(settings), base_url=settings.public_url) as client:
        assert libraries(client)[0]["function_count"] == 1


def test_archive_members_preserve_normal_keys_and_reject_traversal(tmp_path):
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    store.put("normal!/item.json", b"plain")
    store.put_archive("version/report-assets.zip", [("items/a.json", b"a"), ("items/b.json", b"b")])
    assert store.get("normal!/item.json") == b"plain"
    assert store.get("zip:version/report-assets.zip!/items/a.json") == b"a"
    for key in (
        "zip:../report-assets.zip!/items/a.json",
        "zip:version/report-assets.zip!/../secret.json",
        "zip:version/report-assets.zip!//secret.json",
    ):
        with pytest.raises(ValueError):
            store.get(key)
        with pytest.raises(ValueError):
            store.delete(key)
    store.delete("normal!/item.json")
    assert not (store.root / "normal!").exists()
    store.delete("zip:version/report-assets.zip!/items/a.json")
    store.delete("zip:version/report-assets.zip!/items/b.json")
    assert not (store.root / "version").exists()


def test_deleted_source_clears_pending_report_without_losing_snapshot(settings, monkeypatch):
    path = settings.runs_dir / "run-1" / "library.json"
    save_library(path)
    path.with_name("failure_cards.html").write_text("<html>" + " " * (1024 * 1024) + "</html>")
    started, release = pause_catalog(monkeypatch, "_enrich_pending", 5)
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        try:
            assert started.wait(2)
            assert len(libraries(client, refresh=False)) == 1
            path.unlink()
        finally:
            release.set()
        refresh_catalog(client)
        assert app.state.local_catalog.pending_reports() == 0
        assert libraries(client)[0]["version_count"] == 1


@pytest.mark.parametrize("origin,benchmark,before_digest,after_digest", [
    ("run", "démo", None, None), ("catalog", "test", "a", "a"), ("run", "test", "a", "b"),
], ids=["run-unicode-unknown", "catalog-known", "run-changed-data"])
def test_shared_lineage_survives_incomplete_peer_save(
    settings, origin, benchmark, before_digest, after_digest
):
    """Run and durable copies retain shared history through incomplete saves."""
    run = settings.runs_dir / "run-1" / "library.json"
    durable = settings.library_dir / "saved-model" / "benchmark" / "library.json"
    source, copy = (run, durable) if origin == "run" else (durable, run)
    for path in (source, copy):
        save_provenance(path, before_digest)
    # Index the chosen origin first, then establish the other file's snapshot
    # on its branch. This covers both directions of cross-kind branch reuse.
    save_library(source, benchmark=benchmark)
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as client:
        first, = libraries(client)
        save_library(copy, benchmark=benchmark)
        shared, = libraries(client)
        assert shared["id"] == first["id"]
        assert shared["latest_branch_id"] == first["latest_branch_id"]
        assert shared["version_count"] == shared["branch_count"] == 1
        with app.state.database.read() as connection:
            report_key = connection.execute(
                "SELECT report_key FROM library_versions WHERE id = ?",
                (first["latest_version_id"],),
            ).fetchone()[0]
        original_report = app.state.artifact_store.get(report_key)
        assert b"First" in original_report

        with app.state.local_catalog._scan_lock:
            # Model a real save between truncation and write. The healthy copy
            # must advance history even while every read of its peer fails.
            source.write_bytes(b"")
            save_library(copy, benchmark=benchmark, description="Changed")
            save_provenance(copy, after_digest)
        updated = libraries(client)
        if before_digest == after_digest:
            latest, = updated
            assert latest["id"] == first["id"]
            assert latest["latest_branch_id"] == first["latest_branch_id"]
            assert latest["version_count"] == 2
        else:
            # A changed dataset still gets its own history, despite sharing the
            # source run and previously referring to the same local branch.
            assert len(updated) == 2
            latest, = [item for item in updated if item["id"] != first["id"]]
            assert latest["latest_branch_id"] != first["latest_branch_id"]
            assert all(item["version_count"] == 1 for item in updated)
            assert latest["data_identity"]["sources"][0]["sha256"] == after_digest * 64
        assert all(item["branch_count"] == 1 for item in updated)
        assert latest["benchmarks"] == [benchmark]
        versions = {item["id"]: (item["latest_version_id"], item["version_count"])
                    for item in updated}

        with app.state.local_catalog._scan_lock:
            save_library(source, benchmark=benchmark, description="Changed")
            save_provenance(source, after_digest)
        recovered = libraries(client)
        assert {item["id"]: (item["latest_version_id"], item["version_count"])
                for item in recovered} == versions
        assert all(item["branch_count"] == 1 for item in recovered)
        assert app.state.artifact_store.get(report_key) == original_report


# Resumable execution status shown on saved reports.


def test_budget_failure_is_actionable_without_rejecting_detector():
    from autofa.metrics.judge_budget import ReviewBudgetError
    from autofa.metrics.review_failures import processing_failure

    result = processing_failure(ReviewBudgetError("tool-call budget exhausted"))
    assert result["kind"] == "review_budget_exhausted"
    assert result["detector_rejected"] is False
    assert result["recovery_action"]
