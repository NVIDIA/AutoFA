# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Practical CLI seams for publishing (cloud and local) and Continue remote seeds."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.cli import (
    ContinueSeed,
    _continue,
    _local_continue_seed,
    _publish,
    _resume,
    _reuse_saved_selection,
    _select_continue_seed,
)
from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.library_provenance import (
    catalog_provenance_path,
    read_catalog_provenance,
    save_catalog_library,
)
from autofa.data.remote_library_api import (
    RemoteBranch,
    RemoteDataReference,
    RemoteLibrary,
    RemoteLibraryClient,
    RemoteLibraryError,
    RemoteLibraryRef,
    RemotePublication,
    RemoteVersion,
)
from autofa.data.remote_library_publish import (
    maybe_publish_run,
    publish_run,
    remote_publish_target,
)
from autofa.local_service import service_token

EXPORT_URL = "https://eval.example.org/v1/runs/data-run-1/export/autofa"
REFERENCE = {"run_id": "data-run-1", "export_url": EXPORT_URL, "source_harness": "nemo-gym"}
CONTINUE_ARGS = dict(api_base_url=None, api_run_id=None, folder=None, keep_going=False,
                     hours=None, patience=None, max_experiments=None, eval_limit=None)
LINEAGE = {"remote_library_id": "remote-1", "remote_branch_id": "branch-1",
           "remote_branch_key": "origin-run", "remote_version_id": "version-2",
           "remote_base_version_id": "version-2"}
QUIET = SimpleNamespace(print=lambda *args, **kwargs: None)


def _lineage(value, keys=LINEAGE):
    return {key: getattr(value, key) for key in keys}


def _messages():
    messages = []
    return messages, SimpleNamespace(print=lambda message, **kwargs: messages.append(str(message)))


def _published_status(run, service_url, url):
    path = run.run_dir() / "publication.json"
    path.write_text(json.dumps({"state": "published", "service_url": service_url, "url": url}))
    return path


@pytest.mark.parametrize("interactive", [True, False])
def test_missing_saved_selection_widens_to_source_only_after_confirmation(
    monkeypatch, interactive,
):
    reference = {"run_id": "run",
                 "export_url": "https://eval.example.org/v1/runs/run/export/autofa"}
    recovered = SimpleNamespace(parquet_paths=[Path("/tmp/recovered.parquet")])
    monkeypatch.setattr("autofa.cli.run_dataset_wizard", lambda **kwargs: (_ for _ in ()).throw(
        FileNotFoundError("missing cohort")))
    monkeypatch.setattr("autofa.cli._is_interactive_terminal", lambda **kwargs: interactive)
    monkeypatch.setattr("autofa.cli.select_yes_no", lambda *args, **kwargs: True)
    seen = []
    monkeypatch.setattr("autofa.cli.reuse_reference_data",
                        lambda references, **kwargs: seen.append(references) or recovered)

    def select():
        return _reuse_saved_selection("missing-selection", [reference],
                                      api_base_url="https://eval.example.org")

    if interactive:
        assert select() is recovered
        assert seen == [[reference]]
    else:
        with pytest.raises(FileNotFoundError, match="interactive confirmation"):
            select()
        assert seen == []


def test_continue_remote_flag_materializes_cached_seed(tmp_path: Path, monkeypatch):
    cached = tmp_path / "cache" / "library.json"
    cached.parent.mkdir()
    cached.write_text('{"version":1,"functions":[]}', encoding="utf-8")
    selected = RemoteLibraryRef(
        library=RemoteLibrary(id="remote-1", latest_branch_id="branch-1",
                              latest_version_id="version-2"),
        branch=RemoteBranch(id="branch-1", library_id="remote-1", branch_key="origin-run",
                            latest_version_id="version-2"),
        version=RemoteVersion(
            id="version-2", library_id="remote-1", branch_id="branch-1",
            branch_key="origin-run", source_run_id="source-run-1", source_harness="nemo-gym",
            reference_data=(RemoteDataReference(run_id="data-run-1", export_url=EXPORT_URL,
                                                source_harness="nemo-gym"),),
        ),
    )

    class FakeClient:
        def resolve_library_ref(self, library_id, *, branch_id=None, version_id=None):
            assert (library_id, branch_id, version_id) == ("remote-1", "branch-1", "version-2")
            return selected

    monkeypatch.setattr("autofa.cli.client_from_config", lambda cfg: FakeClient())
    monkeypatch.setattr("autofa.cli.materialize_remote_library",
                        lambda selected, *, client: cached)
    args = SimpleNamespace(library_path=None, remote_library_id="remote-1",
                           remote_branch_id="branch-1", remote_version_id="version-2")
    seed = _select_continue_seed(args, Config())
    assert seed is not None
    assert seed.path == str(cached.resolve())
    assert _lineage(seed) == LINEAGE
    assert seed.source_harness == "nemo-gym"
    assert seed.reference_data[0]["run_id"] == "data-run-1"


def test_continue_persists_remote_overwrite_provenance(monkeypatch):
    seed = ContinueSeed(path="/tmp/cached/library.json", source_harness="nemo-gym",
                        reference_data=(dict(REFERENCE),), **LINEAGE)
    selection = SimpleNamespace(
        parquet_paths=[Path("/tmp/evals.parquet")], model_id="model-a", benchmarks=["bench-a"],
        source_paths=[], source_kind="parquet", source_harness="nemo-gym",
        reference_data=list(seed.reference_data),
    )
    created: dict = {}

    class FakeRun:
        eval_limit = 10
        analysis_focus = "both"
        behavior_strictness = "causal"
        min_failure_modes = 4
        objective_settings = {}
        # RunConfig always has these; a run without skills keeps their defaults.
        skills_root = ""
        skill_snapshots = skill_exclusions = skill_ignored_paths = skill_unmatched_packs = ()

        def save(self):
            return None

    def fake_create(*args, **kwargs):
        created.update(kwargs)
        return FakeRun()

    monkeypatch.setattr("autofa.cli.load_config", lambda: Config(min_failure_modes=4))
    monkeypatch.setattr("autofa.cli._select_continue_seed", lambda args, cfg: seed)
    monkeypatch.setattr("autofa.cli.run_dataset_wizard",
                        lambda **kwargs: pytest.fail("source wizard should be skipped"))
    monkeypatch.setattr("autofa.cli.reuse_reference_data", lambda references, **kwargs: selection)
    monkeypatch.setattr("autofa.cli.select_yes_no", lambda *args, **kwargs: True)
    monkeypatch.setattr("autofa.cli.RunConfig.create", fake_create)
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "autofa.wizard.depth.select_discovery_depth",
        lambda **kwargs: SimpleNamespace(max_rounds=3, top_x=5, min_failure_modes=4),
    )
    monkeypatch.setattr("autofa.cli.load_runtime_from_run", lambda run, cfg: SimpleNamespace(
        persist_library=lambda: Path("/tmp/working-library.json")))
    completed: list[tuple[object, dict]] = []
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda run, **kwargs: completed.append((run, kwargs)))

    _continue(SimpleNamespace(library_path=None, remote_library_id="remote-1",
                              analysis_outputs=["agent"], **CONTINUE_ARGS))

    assert created["seed_library_path"] == seed.path
    assert created["min_failure_modes"] == 4
    assert {key: created[key] for key in LINEAGE} == LINEAGE
    assert created["source_harness"] == "nemo-gym"
    assert created["reference_data"] == list(seed.reference_data)
    ((run, options),) = completed
    assert run.analysis_outputs == ["agent"]
    assert options["exit_after_discovery"] is True


@pytest.fixture
def resume(tmp_path, monkeypatch):
    """A saved run whose input cache is missing, with the session runner stubbed."""
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("autofa.cli.load_config", Config)
    monkeypatch.setattr("autofa.cli.select_yes_no", lambda *args, **kwargs: True)
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning",
                        lambda *args, **kwargs: SimpleNamespace(summary=lambda: "memory ok"))
    completed = []
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda selected, **kwargs: completed.append((selected, kwargs)))
    recovered = tmp_path / "recovered.parquet"
    recovered.write_bytes(b"parquet")

    def saved_run(run_id, reference, **kwargs):
        kwargs.setdefault("analysis_outputs", ["agent"])
        run = RunConfig(run_id=run_id, parquet_paths=[str(tmp_path / "missing.parquet")],
                        model_id="model-a", reference_data=[reference], **kwargs)
        run.save()
        return run

    def selection(reference_data, source_harness="nemo-gym", **kwargs):
        return SimpleNamespace(parquet_paths=[recovered], source_paths=[EXPORT_URL],
                               source_kind="remote_run", source_harness=source_harness,
                               reference_data=reference_data, **kwargs)

    return SimpleNamespace(recovered=recovered, completed=completed, saved_run=saved_run,
                           selection=selection)


@pytest.mark.parametrize("once", [False, True])
def test_resume_redownloads_exact_reference_when_cache_is_missing(resume, monkeypatch, once):
    run = resume.saved_run("resume-run", dict(REFERENCE))
    selection = resume.selection(
        [{**REFERENCE, "parquet_path": str(resume.recovered.resolve())}],
    )
    monkeypatch.setattr("autofa.cli.reuse_reference_data", lambda references, **kwargs: selection)

    _resume(SimpleNamespace(run_id=run.run_id, once=once))

    loaded = RunConfig.load(run.run_id)
    assert loaded.parquet_paths == [str(resume.recovered.resolve())]
    assert loaded.source_harness == "nemo-gym"
    assert loaded.reference_data == selection.reference_data
    (resumed, options), = resume.completed
    assert resumed.run_id == run.run_id
    assert options["auto_discover"] is False
    assert options["exit_after_discovery"] is once


def test_resume_refuses_a_run_without_recorded_analysis_outputs(resume):
    run = resume.saved_run("resume-pre-ga", dict(REFERENCE), analysis_outputs=[])

    with pytest.raises(ValueError, match="predates recorded analysis outputs"):
        _resume(SimpleNamespace(run_id=run.run_id))

    assert resume.completed == []
    assert RunConfig.load(run.run_id).parquet_paths == run.parquet_paths


def test_resume_reuses_saved_selection_without_widening(resume, monkeypatch):
    reference = {**REFERENCE, "parquet_path": str(resume.recovered)}
    run = resume.saved_run("resume-selection", reference, selection_id="selection-1")
    selection = resume.selection([reference], selection_id="selection-1")
    calls = []
    monkeypatch.setattr("autofa.cli.run_dataset_wizard",
                        lambda **kwargs: calls.append(kwargs) or selection)
    monkeypatch.setattr("autofa.cli.reuse_reference_data",
                        lambda *args, **kwargs: pytest.fail("saved selection was widened"))

    _resume(SimpleNamespace(run_id=run.run_id))

    assert calls[0]["selection_id"] == "selection-1"
    assert RunConfig.load(run.run_id).selection_id == "selection-1"


def test_resume_source_fallback_clears_missing_selection_id(resume, monkeypatch):
    reference = {"run_id": "data-run-1", "export_url": EXPORT_URL}
    run = resume.saved_run("resume-fallback", reference, selection_id="missing-selection")
    selection = resume.selection([{**reference, "parquet_path": str(resume.recovered)}],
                                 source_harness="unknown", selection_id=None)
    monkeypatch.setattr("autofa.cli._reuse_saved_selection", lambda *args, **kwargs: selection)

    _resume(SimpleNamespace(run_id=run.run_id))

    assert RunConfig.load(run.run_id).selection_id is None


def test_repeat_publish_appends_with_current_run_source_and_saved_branch_key(
    tmp_path: Path, monkeypatch,
):
    evidence_scope = "Original traces; artifacts not inspected."
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig(
        run_id="run-1", parquet_paths=["/private/local/evals.parquet"],
        source_paths=["/private/local/source.jsonl"], model_id="model-a",
        benchmarks=["bench-a"], source_harness="nemo-gym",
        reference_data=[{**REFERENCE, "parquet_path": "/private/local/evals.parquet"}],
        remote_library_id="remote-1", remote_branch_key="source-run-1",
    )
    run.reference_data[0]["evidence_scope"] = evidence_scope
    library = FunctionLibrary(model_id="model-a", benchmark="bench-a", functions=[FeatureFunction(
        name="detector", description="Detects a failure",
        source="def detector(item, llm_client=None):\n    return False\n",
        scores={"evaluated": True, "recall": .4, "precision": .9, "ppv_at_pi0": .9,
                "behavior_purity": .8, "behavior_specificity": .8},
    )])

    class FakeRuntime:
        dataset = object()

        def __init__(self):
            self.library = library

        def persist_library(self):
            path = run.run_dir() / "library.json"
            self.library.save(path)
            return path

    captured: list[dict] = []

    class FakeClient:
        def upload_library(self, *, metadata, library_json, failure_cards_html,
                           report_assets_zip):
            captured.append({
                "metadata": dict(metadata),
                "library": Path(library_json).read_text(encoding="utf-8"),
                "html": Path(failure_cards_html).read_text(encoding="utf-8"),
                "assets": report_assets_zip.decode(),
            })
            number = len(captured) + 2
            return RemotePublication(
                library_id="remote-1", branch_id="branch-1", version_id=f"version-{number}",
                branch_key="source-run-1", version_number=number, source_run_id="run-1",
                dashboard_url="https://example.test/libraries/remote-1",
            )

        def dashboard_url(self, library_id):
            return f"https://example.test/libraries/{library_id}"

    report_path = run.run_dir() / "failure_cards_cloud.html"

    def fake_write(library_arg, *, out, dataset, source_harness, reference_data,
                   prefer_source_links, source_links_only, console):
        assert (library_arg, dataset, source_harness) == (library, FakeRuntime.dataset,
                                                          "nemo-gym")
        assert reference_data[0]["evidence_scope"] == evidence_scope
        assert prefer_source_links is source_links_only is True
        Path(out).write_text("<html>live</html>", encoding="utf-8")
        return Path(out)

    def fake_assets(report_dir, *, html_name):
        assert Path(report_dir) == report_path.parent
        assert html_name == "failure_cards_cloud.html"
        return b"report-assets"

    monkeypatch.setattr("autofa.cli.load_config", lambda: Config())
    monkeypatch.setattr("autofa.data.remote_library_publish.client_from_config",
                        lambda cfg: FakeClient())
    monkeypatch.setattr("autofa.cli.RunConfig.load", lambda run_id: run)
    monkeypatch.setattr("autofa.mcp.runtime.load_runtime_from_run",
                        lambda selected_run, cfg: FakeRuntime())
    prepared = []
    monkeypatch.setattr("autofa.session.report_highlights.prepare_report_highlights",
                        lambda runtime, **kwargs: prepared.append(runtime.library))
    monkeypatch.setattr("autofa.reports.failure_cards.write_failure_cards_with_status", fake_write)
    monkeypatch.setattr("autofa.reports.failure_cards.build_report_assets_zip", fake_assets)

    first = _publish(SimpleNamespace(run_id="run-1"))
    second = _publish(SimpleNamespace(run_id="run-1"))
    assert first is not None and second is not None
    assert prepared == [library, library]
    metadata = [call["metadata"] for call in captured]
    assert [row["branch_key"] for row in metadata] == ["source-run-1"] * 2
    for row in metadata:
        assert (row["source_run_id"], row["harness"], row["source_harness"],
                row["publish_trigger"]) == ("run-1", "claude", "nemo-gym", "manual_cli")
    assert metadata[0]["reference_data"] == [REFERENCE]
    assert metadata[0]["library_id"] == "remote-1"
    assert "branch_id" not in metadata[0]
    assert metadata[1]["branch_id"] == "branch-1"
    assert json.loads(captured[0]["library"])["version"] == 1
    assert captured[0]["html"] == "<html>live</html>"
    assert captured[0]["assets"] == "report-assets"
    assert run.remote_version_id == "version-4"
    serialized = json.dumps(captured)
    assert "config.json" not in serialized
    assert "/private/local" not in serialized


def test_remote_publish_target_and_provenance_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    fresh = RunConfig.create([tmp_path / "data.parquet"], "model-a", ["bench-a"], Config())
    assert RunConfig.load(fresh.run_id).remote_branch_key == fresh.run_id
    assert remote_publish_target(fresh) == fresh.run_id

    RunConfig(
        run_id="continued-run", parquet_paths=[], model_id="model-a",
        remote_library_id="remote-1", remote_branch_id="branch-1",
        remote_version_id="version-2", remote_branch_key="origin-run",
    ).save()
    loaded = RunConfig.load("continued-run")
    assert _lineage(loaded) == LINEAGE
    assert remote_publish_target(loaded) == "origin-run"


def test_pre_ga_cloud_run_without_branch_key_refuses_to_publish(tmp_path: Path, monkeypatch):
    """A config saved before branch keys must be continued from its cloud version again."""
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run_dir = tmp_path / "runs" / "old-run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": "old-run", "parquet_paths": [], "model_id": "model-a",
        "remote_library_id": "lib-1", "remote_publish_run_id": "old",
    }))
    loaded = RunConfig.load("old-run")
    assert loaded.remote_library_id == "lib-1" and loaded.remote_branch_key is None
    with pytest.raises(RemoteLibraryError, match="library id but no branch key"):
        remote_publish_target(loaded)


def test_catalog_sidecar_round_trip_preserves_portable_library(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig(
        run_id="continued-run", parquet_paths=[], model_id="model-a", benchmarks=["bench-a"],
        source_harness="nemo-gym", reference_data=[dict(REFERENCE)],
        remote_library_id="library-1", remote_branch_id="branch-1",
        remote_branch_key="origin-run", remote_version_id="version-3",
        remote_base_version_id="version-2", selection_id="selection-1",
    )
    run.save()
    library = FunctionLibrary(model_id="model-a", benchmark="bench-a")
    path = tmp_path / "catalog" / "model-a" / "bench-a" / "library.json"

    saved, sidecar = save_catalog_library(library, path, run)
    assert saved == path
    assert sidecar == catalog_provenance_path(path)
    assert json.loads(path.read_text(encoding="utf-8")) == library.to_dict()
    assert "provenance" not in path.read_text(encoding="utf-8")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance == {
        "schema_version": 2,
        "source_run_id": "continued-run",
        "remote_library_id": "library-1",
        "remote_branch_id": "branch-1",
        "remote_branch_key": "origin-run",
        "remote_version_id": "version-3",
        "remote_base_version_id": "version-2",
        "source_harness": "nemo-gym",
        "reference_data": run.reference_data,
        "data_identity": {"version": 1, "sources": [
            {"kind": "evalrdl", "origin": "https://eval.example.org", "run_id": "data-run-1"},
        ]},
        "selection_id": "selection-1",
    }
    assert list(sidecar.parent.glob(".*.partial")) == []

    seed_keys = ("source_run_id", "remote_library_id", "remote_branch_id", "remote_branch_key",
                 "remote_version_id", "remote_base_version_id", "source_harness", "selection_id")
    seed = _local_continue_seed(path)
    assert _lineage(seed, seed_keys) == {key: provenance[key] for key in seed_keys}
    assert list(seed.reference_data) == run.reference_data

    (run.run_dir() / "config.json").unlink()
    assert _local_continue_seed(path).selection_id == "selection-1"

    # A schema v1 sidecar is rejected; a catalog outside a run starts a fresh lineage.
    sidecar.write_text(json.dumps({**provenance, "schema_version": 1}), encoding="utf-8")
    assert read_catalog_provenance(path) is None
    assert _local_continue_seed(path) == ContinueSeed(path=str(path.resolve()))


def test_continue_from_run_working_library_keeps_lineage(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig(
        run_id="working-run", parquet_paths=[], model_id="model-a", benchmarks=["bench-a"],
        source_harness="nemo-gym", reference_data=[dict(REFERENCE)],
        remote_library_id="library-1", remote_branch_id="branch-1",
        remote_branch_key="origin-run", remote_version_id="version-3",
        remote_base_version_id="version-2", selection_id="selection-1",
    )
    run.save()
    path = run.run_dir() / "library.json"
    FunctionLibrary(model_id="model-a", benchmark="bench-a").save(path)
    assert read_catalog_provenance(path) is None

    seed = _local_continue_seed(path)
    assert seed == ContinueSeed(
        path=str(path.resolve()),
        remote_library_id="library-1",
        remote_branch_id="branch-1",
        remote_branch_key="origin-run",
        remote_version_id="version-3",
        remote_base_version_id="version-2",
        source_run_id="working-run",
        source_harness="nemo-gym",
        reference_data=seed.reference_data,
        selection_id="selection-1",
    )
    assert list(seed.reference_data) == run.reference_data

    # runs/<id> may point at another disk; Continue keeps the same lineage there.
    storage = tmp_path / "storage"
    storage.mkdir()
    run.run_dir().rename(storage / "working-run")
    (tmp_path / "runs" / "working-run").symlink_to(storage / "working-run",
                                                   target_is_directory=True)
    assert _local_continue_seed(path.resolve()) == replace(seed, path=str(path.resolve()))
    # The target folder need not carry the run ID.
    (tmp_path / "runs" / "working-run").unlink()
    (storage / "working-run").rename(storage / "data")
    (tmp_path / "runs" / "working-run").symlink_to(storage / "data", target_is_directory=True)
    moved = (storage / "data" / "library.json").resolve()
    assert _local_continue_seed(moved) == replace(seed, path=str(moved))
    # The same files outside a registered run start a fresh lineage.
    unregistered = tmp_path / "copied-run"
    shutil.copytree(storage / "data", unregistered)
    assert _local_continue_seed(unregistered / "library.json") == ContinueSeed(
        path=str((unregistered / "library.json").resolve()))

    fresh = RunConfig(run_id="fresh-run", parquet_paths=[], model_id="model-a")
    fresh.save()
    fresh_path = fresh.run_dir() / "library.json"
    FunctionLibrary(model_id="model-a").save(fresh_path)
    fresh_seed = _local_continue_seed(fresh_path)
    assert fresh_seed.source_run_id == "fresh-run"
    assert fresh_seed.remote_branch_key == "fresh-run"
    assert fresh_seed.remote_library_id is None


def test_auto_publish_skips_without_token(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    cfg = Config(remote_library_base_url="https://library.example.org",
                 remote_library_token_env="AUTOFA_TEST_TOKEN")
    run = RunConfig(run_id="run-1", parquet_paths=[], model_id="model-a")
    monkeypatch.delenv("AUTOFA_TEST_TOKEN", raising=False)
    monkeypatch.setattr("autofa.data.remote_library_publish.client_from_config",
                        lambda cfg: (_ for _ in ()).throw(RemoteLibraryError("no credential")))
    monkeypatch.setattr("autofa.data.remote_library_publish.publish_run",
                        lambda *args, **kwargs: pytest.fail("publish should be skipped"))
    assert maybe_publish_run(run, cfg, trigger="session_end", console=QUIET) is None


def test_auto_publish_failure_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    cfg = Config(remote_library_base_url="https://library.example.org",
                 remote_library_token_env="AUTOFA_TEST_TOKEN")
    run = RunConfig(run_id="run-1", parquet_paths=[], model_id="model-a")
    monkeypatch.setenv("AUTOFA_TEST_TOKEN", "token")
    monkeypatch.setattr(
        "autofa.data.remote_library_publish.publish_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RemoteLibraryError("service unavailable")),
    )

    assert maybe_publish_run(run, cfg, trigger="session_end", console=QUIET) is None
    status = json.loads((run.run_dir() / "publication.json").read_text())
    assert status["state"] == "failed" and status["error"] == "service unavailable"
    assert status["retry_command"] == "autofa publish --run run-1"
    assert "service unavailable" in (tmp_path / "cli-errors.log").read_text()


@pytest.mark.parametrize("previous,interrupted", [
    ("none", False), ("current", False), ("stale", True),
])
def test_session_exit_invokes_best_effort_auto_publish(
    tmp_path: Path, monkeypatch, previous: str, interrupted: bool,
):
    from autofa.data.remote_library_publish import _publication_origin
    from autofa.mcp.runtime import Runtime
    from autofa.session.runner import run_session

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    cfg = Config(library_root=str(tmp_path / "catalog"))
    run = RunConfig(run_id="run-1", parquet_paths=[], model_id="model-a", benchmarks=["bench-a"])
    library = FunctionLibrary(model_id="model-a", benchmark="bench-a", functions=[])

    class FakeDataset:
        def iter_batches(self, **kwargs):
            return iter(())

    class FakeHarness:
        async def run_session(self, **kwargs):
            if interrupted:
                raise RuntimeError("Session interrupted before finalization")
            return None

    runtime = Runtime(dataset=FakeDataset(), library=library, run=run,  # type: ignore[arg-type]
                      config=cfg)
    if previous != "none":
        library_path = runtime.persist_library()
        fingerprint = hashlib.sha256(library_path.read_bytes()).hexdigest()
        (run.run_dir() / "publication.json").write_text(json.dumps({
            "state": "published",
            "service_url": _publication_origin(cfg.remote_library_base_url),
            "library_sha256": fingerprint if previous == "current" else "0" * 64,
            "data_identity": None,
        }))

    monkeypatch.setattr("autofa.session.runner.load_runtime_from_run",
                        lambda selected, selected_cfg: runtime)
    monkeypatch.setattr("autofa.session.runner.get_harness", lambda *args, **kwargs: FakeHarness())
    monkeypatch.setattr("autofa.session.runner.write_session_workspace",
                        lambda *args, **kwargs: run.run_dir())
    monkeypatch.setattr("autofa.session.runner.build_mcp_server", lambda *args, **kwargs: object())
    monkeypatch.setattr("autofa.session.runner.tool_names", lambda: [])
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning",
                        lambda *args, **kwargs: SimpleNamespace(summary=lambda: "memory ok"))
    monkeypatch.setattr(
        "autofa.diagnostics._DIAGNOSTICS",
        SimpleNamespace(configure=lambda path: None, preflight_lines=lambda: []),
    )
    published: list[tuple[RunConfig, object, str]] = []
    monkeypatch.setattr(
        "autofa.data.remote_library_publish.maybe_publish_run",
        lambda selected, selected_cfg, *, trigger, runtime, console: published.append(
            (selected, runtime, trigger)
        ),
    )

    if interrupted:
        with pytest.raises(RuntimeError, match="Session interrupted before finalization"):
            run_session(run, cfg=cfg, auto_discover=False, top_x=7)
    else:
        run_session(run, cfg=cfg, auto_discover=False, top_x=7)

    assert run.top_x == 7
    expected = [] if interrupted or previous == "current" else [(run, runtime, "session_end")]
    assert published == expected
    assert (run.run_dir() / "library.json").is_file()


# Local auto-publication uses the profile token and never crosses to a known other service.


def _no_cloud_credentials(*args, **kwargs):
    pytest.fail("Local library operations must not access cloud credentials")


def test_local_auto_publish_uses_profile_token_and_embedded_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setenv("GITLAB_TOKEN", "cloud-environment-token")
    monkeypatch.setattr("autofa.data.remote_library_credentials.resolve_credential",
                        _no_cloud_credentials)
    token = service_token(create=True)
    run = RunConfig(
        run_id="local-run", parquet_paths=[], model_id="model-a",
        remote_library_id="local-library", remote_branch_id="local-branch",
        remote_branch_key="local-run",
        reference_data=[{"run_id": "input-run", "export_url": "https://example.test/export",
                         "source_harness": "unknown"}],
    )
    run.save()
    _published_status(run, "http://127.0.0.1:8765",
                      "http://127.0.0.1:8765/libraries/local-library")
    library = FunctionLibrary(model_id="model-a", benchmark="bench-a")
    library_path = run.run_dir() / "library.json"
    library.save(library_path)
    runtime = SimpleNamespace(library=library, dataset=None, persist_library=lambda: library_path)
    reports = []

    def write_report(library_arg, **kwargs):
        reports.append(kwargs)
        kwargs["out"].write_text("<html>local evidence</html>")
        return kwargs["out"]

    uploads = []

    def upload(client, **kwargs):
        assert client.token == token
        assert client.credential_source == "local"
        uploads.append(kwargs)
        return RemotePublication(
            library_id="local-library", branch_id="local-branch",
            branch_key=kwargs["metadata"]["branch_key"],
            version_id="local-version", version_number=1,
            dashboard_url="http://127.0.0.1:8765/libraries/local-library",
        )

    monkeypatch.setattr("autofa.reports.failure_cards.write_failure_cards_with_status",
                        write_report)
    monkeypatch.setattr("autofa.reports.failure_cards.build_report_assets_zip",
                        lambda *args, **kwargs: b"assets")
    monkeypatch.setattr(RemoteLibraryClient, "upload_library", upload)
    result = maybe_publish_run(run, Config(remote_library_base_url="http://127.0.0.1:8765"),
                               trigger="session_end", runtime=runtime, console=QUIET)

    assert result is not None
    assert uploads[0]["metadata"]["reference_data"] == []
    assert reports[0]["prefer_source_links"] is True
    assert reports[0]["out"].name == "failure_cards_local.html"
    assert run.remote_version_id == "local-version"
    status = json.loads((run.run_dir() / "publication.json").read_text())
    assert (status["state"], status["service_url"]) == ("published", "http://127.0.0.1:8765")


@pytest.mark.parametrize("automatic", [False, True])
def test_publication_rejects_known_other_service_before_artifacts(monkeypatch, tmp_path, automatic):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    service_token(create=True)
    run = RunConfig(run_id="published-run", parquet_paths=[], model_id="model-a",
                    remote_library_id="cloud-library", remote_branch_id="cloud-branch")
    run.save()
    status_path = _published_status(run, "https://example.test",
                                    "https://example.test/libraries/cloud-library")
    before = status_path.read_bytes()
    monkeypatch.setattr(
        "autofa.data.remote_library_publish._publish_run",
        lambda *args, **kwargs: pytest.fail("Do not build artifacts or upload to another service"),
    )
    cfg = Config(remote_library_base_url="http://127.0.0.1:8765")
    if automatic:
        messages, console = _messages()
        assert maybe_publish_run(run, cfg, trigger="session_end", console=console) is None
        assert any("Start a new run" in message for message in messages)
    else:
        with pytest.raises(RemoteLibraryError, match="Start a new run"):
            publish_run(run, cfg, trigger="manual_cli")
    assert status_path.read_bytes() == before


def test_failed_retry_retains_known_publication_service(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig(run_id="published-run", parquet_paths=[], model_id="model-a",
                    remote_library_id="cloud-library")
    run.save()
    status_path = _published_status(run, "https://example.test:443",
                                    "https://example.test:443/libraries/cloud-library")

    def fail_upload(*args, **kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr("autofa.data.remote_library_publish._publish_run", fail_upload)
    with pytest.raises(RuntimeError, match="service unavailable"):
        publish_run(run, Config(remote_library_base_url="https://example.test",
                                remote_library_token_env="GITLAB_TOKEN"), trigger="manual_cli")
    status = json.loads(status_path.read_text())
    assert status["state"] == "failed"
    assert status["service_url"] == "https://example.test"
    with pytest.raises(RemoteLibraryError, match="Start a new run"):
        publish_run(run, Config(remote_library_base_url="http://127.0.0.1:8765"),
                    trigger="manual_cli")
