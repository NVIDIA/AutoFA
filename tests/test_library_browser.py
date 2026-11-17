# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library browser opens exact remote versions and publishes catalogs through their source run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.data.library_provenance import save_catalog_library
from autofa.data.remote_library_api import (
    RemoteBranch,
    RemoteLibrary,
    RemoteLibraryError,
    RemotePublication,
    RemoteVersion,
)
from autofa.ui.library_browser import (
    _matching_run,
    _publish_catalog,
    browse_local_library,
    browse_remote_library,
    list_catalog_entries,
)
from autofa.ui.menus import MenuItem


def test_remote_menu_and_browser_open_cached_library(monkeypatch, tmp_path: Path):
    remote = RemoteLibrary(id="remote-1", model_id="model-a", benchmarks=("bench-a",),
                           latest_branch_id="branch-1", latest_version_id="version-2")
    branch = RemoteBranch(id="branch-1", library_id="remote-1", branch_key="origin-run",
                          latest_version_id="version-2", latest_version_number=2, version_count=2)
    version = RemoteVersion(
        id="version-2", library_id="remote-1", branch_id="branch-1", branch_key="origin-run",
        version_number=2, source_run_id="continued-run", model_id="model-a",
        benchmarks=("bench-a",), function_count=5,
    )

    class FakeClient:
        def list_libraries(self):
            return [remote]

        def list_branches(self, library_id):
            assert library_id == "remote-1"
            return [branch]

        def list_versions(self, library_id, branch_id):
            assert (library_id, branch_id) == ("remote-1", "branch-1")
            return [version]

    choices = iter([MenuItem("selected", remote), MenuItem("selected version", version), None])
    monkeypatch.setattr("autofa.ui.library_browser.select_option",
                        lambda *args, **kwargs: next(choices))
    cached_path = tmp_path / "library.json"
    materialized = []
    monkeypatch.setattr("autofa.ui.library_browser.materialize_remote_library",
                        lambda ref, *, client: materialized.append(ref) or cached_path)
    opened = []
    monkeypatch.setattr("autofa.ui.library_browser._open_library",
                        lambda entry, **kwargs: opened.append(entry))

    browse_remote_library(FakeClient())  # type: ignore[arg-type]

    assert [ref.version_id for ref in materialized] == ["version-2"]
    (entry,) = opened
    assert (entry.model_id, entry.benchmark, entry.path) == ("model-a", "bench-a", cached_path)


def test_remote_browser_delete_matches_local_delete_flow(monkeypatch):
    remote = RemoteLibrary(id="remote-1", model_id="model-a", benchmarks=("bench-a",))

    class FakeClient:
        deleted: list[str] = []

        def list_libraries(self):
            return [] if self.deleted else [remote]

        def delete_library(self, library_id):
            self.deleted.append(library_id)

    client = FakeClient()
    monkeypatch.setattr("autofa.ui.library_browser.select_option",
                        lambda *args, **kwargs: MenuItem("delete", remote, action="delete"))
    monkeypatch.setattr("autofa.ui.library_browser.select_yes_no", lambda *args, **kwargs: True)

    browse_remote_library(client)  # type: ignore[arg-type]

    assert client.deleted == ["remote-1"]


def test_matching_run_resolves_catalog_provenance(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig(
        run_id="source-run", parquet_paths=[], model_id="model-a", benchmarks=["bench-a"],
        remote_library_id="library-1", remote_branch_id="branch-1", remote_branch_key="origin-run",
    )
    run.save()
    catalog_path = tmp_path / "catalog" / "model-a" / "bench-a" / "library.json"
    _, sidecar = save_catalog_library(
        FunctionLibrary(model_id="model-a", benchmark="bench-a"), catalog_path, run,
    )
    entry = list_catalog_entries(tmp_path / "catalog")[0]

    matched = _matching_run(entry)

    assert matched is not None
    assert (matched.run_id, matched.remote_branch_key) == ("source-run", "origin-run")
    # Catalog provenance comes only from the sidecar, never from content matching.
    sidecar.unlink()
    assert _matching_run(entry) is None


@pytest.mark.parametrize("error", [None, RemoteLibraryError("service unavailable")])
def test_local_browser_publishes_selected_catalog_and_shows_outcome(
    tmp_path: Path, monkeypatch, error,
):
    root = tmp_path / "catalog" / "model-a" / "bench-a"
    root.mkdir(parents=True)
    (root / "library.json").write_text(json.dumps(
        {"version": 1, "model_id": "model-a", "benchmark": "bench-a", "functions": []}))
    entry = list_catalog_entries(tmp_path / "catalog")[0]
    choices = iter([MenuItem("publish", entry, action="publish"), None])
    menu_calls = []

    def select(*args, **kwargs):
        menu_calls.append(kwargs)
        return next(choices)

    monkeypatch.setattr("autofa.ui.library_browser.select_option", select)
    published = []

    def publish(selected, cfg):
        published.append(selected)
        if error:
            raise error
        return RemotePublication(library_id="remote-1", branch_id="branch-1",
                                 version_id="version-1", branch_key="run-1",
                                 dashboard_url="https://autofa.example/remote-1")

    monkeypatch.setattr("autofa.ui.library_browser._publish_catalog", publish)

    browse_local_library(tmp_path / "catalog")

    assert published == [entry]
    feedback = "".join(text for _, text in menu_calls[1]["footer"])
    expected = "service unavailable" if error else "https://autofa.example/remote-1"
    assert expected in feedback


def test_publish_catalog_publishes_matching_run_from_browse_menu(monkeypatch):
    run = SimpleNamespace(run_id="run-1")
    remote = RemotePublication(library_id="remote-1", branch_id="branch-1",
                               version_id="version-1", branch_key="run-1")
    monkeypatch.setattr("autofa.ui.library_browser._matching_run", lambda selected: run)
    published = []
    monkeypatch.setattr("autofa.data.remote_library_publish.publish_run",
                        lambda selected, cfg, **kwargs: published.append(kwargs) or remote)
    monkeypatch.setattr("autofa.ui.library_browser.write_catalog_provenance",
                        lambda *args, **kwargs: None)

    entry = SimpleNamespace(path=Path("library.json"))
    result = _publish_catalog(entry, Config())  # type: ignore[arg-type]

    assert result is remote
    assert published[0]["trigger"] == "browse_menu"
