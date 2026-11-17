# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset identities and reference data name exact source data, never paths or sessions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.data.cohort import Cohort, cohorts_root, save_cohort
from autofa.data.dataset_identity import (
    dataset_mixture_id,
    normalize_data_identity,
    publication_data_identity,
)
from autofa.data.library_provenance import CatalogProvenance
from autofa.data.parquet_store import Dataset, make_item_id
from autofa.data.reference_data import (
    canonical_export_url,
    data_api_location,
    inspect_url,
    make_local_reference_data,
    make_reference_data,
    require_canonical_reference_data,
    rollout_url,
)
from autofa.data.remote_api import remote_cache_dir
from autofa.data.remote_library_publish import published_library_is_current
from autofa.data.sources import EvalSource
from autofa.mcp import runtime
from autofa.mcp.runtime import load_runtime_from_run
from autofa.reports.failure_cards import (
    _compact_item,
    build_data,
    render_html,
    try_dataset_for_library,
)
from autofa.wizard.dataset import _local_source_harness, reuse_reference_data
from autofa_library_server.data_identity import normalize_data_identity as server_normalize

DATA_API = "https://eval.example.org"


def reference(path, run_id="source-run", base=DATA_API):
    return make_reference_data(run_id=run_id, export_url=canonical_export_url(base, run_id),
                               source_harness="nemo-gym", parquet_path=path)


def _remote_run(root):
    paths = [root / name / "dataset.parquet" for name in ["first", "second"]]
    return RunConfig(
        run_id="analysis-1",
        parquet_paths=[str(path) for path in paths],
        model_id="*",
        benchmarks=["bench-b", "bench-a"],
        source_kind="remote_run",
        reference_data=[reference(path, f"source-{i}") for i, path in enumerate(paths)],
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


# --- Mixture names identify source sets, never become row filters ---

def test_remote_identity_ignores_order_cache_paths_and_analysis_run(tmp_path):
    first = _remote_run(tmp_path / "machine-a")
    second = _remote_run(tmp_path / "machine-b")
    second.run_id = "analysis-2"
    second.model_id = "ALL"
    second.parquet_paths.reverse()
    second.reference_data.reverse()
    second.benchmarks.reverse()
    second.reference_data.append(dict(second.reference_data[0]))
    assert first.library_model_id().startswith("mixture-")
    assert first.library_model_id() == second.library_model_id()
    assert first.benchmark_label() == second.benchmark_label()
    public_references = [
        {k: v for k, v in ref.items() if k != "parquet_path"} for ref in first.reference_data
    ]
    portable = replace(first, dataset_id="", reference_data=public_references)
    assert portable.library_model_id() == first.library_model_id()


def test_different_sources_origins_and_filters_get_different_ids(tmp_path):
    run = _remote_run(tmp_path)
    changed_sources = [dict(ref) for ref in run.reference_data]
    changed_sources[1] = reference(run.parquet_paths[1], "different-source")
    changed_origin = [dict(ref) for ref in run.reference_data]
    changed_origin[1] = reference(run.parquet_paths[1], "source-1",
                                  base="https://another.example.test")
    variants = [
        replace(run, dataset_id="", reference_data=changed_sources),
        replace(run, dataset_id="", reference_data=changed_origin),
        replace(run, dataset_id="", benchmarks=["bench-a"]),
        replace(run, dataset_id="", model_id="one-model"),
    ]
    names = [run.library_model_id(), *(variant.library_model_id() for variant in variants)]
    assert len(set(names)) == len(names)


def test_local_identity_uses_full_contents_not_paths_or_timestamps(tmp_path):
    contents = [b"a" * 3_000_000, b"another dataset"]
    for names in (["first", "second"], ["renamed-a", "renamed-b"]):
        for name, data in zip(names, contents, strict=True):
            (tmp_path / f"{name}.parquet").write_bytes(data)

    def identify(*names):
        return dataset_mixture_id(
            [str(tmp_path / f"{name}.parquet") for name in names],
            model_id="*", benchmarks=[], reference_data=[],
        )

    original = identify("first", "second")
    # Renamed copies selected in the opposite order name the same data.
    assert identify("renamed-b", "renamed-a") == original
    # A change in the middle of a large file must also change the identity.
    with (tmp_path / "renamed-a.parquet").open("r+b") as handle:
        handle.seek(1_500_000)
        handle.write(b"b")
    assert identify("renamed-a", "renamed-b") != original


def test_single_model_keeps_its_name_without_source_files_but_wildcard_gets_identity(tmp_path):
    run = _remote_run(tmp_path)
    run.parquet_paths = run.parquet_paths[:1]
    run.reference_data = run.reference_data[:1]
    assert run.library_model_id().startswith("mixture-")
    run.model_id = "one-model"
    assert run.library_model_id() == "one-model"
    run.dataset_id = "mixture-legacy"
    run.reference_data = []
    assert run.library_model_id() == "one-model"


def test_single_model_catalog_lookup_accepts_model_and_legacy_mixture(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    run = _remote_run(tmp_path)
    run.model_id = "one-model"
    run.benchmarks = [f"bench-{i}" for i in range(4)]
    for raw in run.parquet_paths:
        path = Path(raw)
        _write_rows(path, [{"model_id": run.model_id, "benchmark": benchmark,
                            "problem_id": path.parent.name, "is_correct": False}
                           for benchmark in run.benchmarks])
    run.dataset_id = dataset_mixture_id(
        run.parquet_paths, model_id=run.model_id,
        benchmarks=run.benchmarks, reference_data=run.reference_data,
    )
    run.save()
    for name in (run.model_id, run.dataset_id):
        library = FunctionLibrary(model_id=name, benchmark=run.benchmark_label())
        rebound = try_dataset_for_library(library)
        assert rebound.model_id == run.model_id
        assert len(rebound.load_items()) == 8


def test_persisted_identity_survives_unavailable_local_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    paths = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
    for path in paths:
        path.write_bytes(path.name.encode())
    run = RunConfig.create(paths, "*", ["bench-a"], Config())
    name = run.library_model_id()
    for path in paths:
        path.unlink()
    loaded = RunConfig.load(run.run_id)
    assert loaded.dataset_id == name
    assert loaded.library_model_id() == name
    assert loaded.model_id == "*"


@pytest.mark.parametrize("model_id", ["*", ""])
def test_legacy_run_and_catalog_lookup_use_mixture_identity(tmp_path, monkeypatch, model_id):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime, "_RUNTIME", None)
    run = _remote_run(tmp_path)
    run.model_id = model_id
    for i, raw in enumerate(run.parquet_paths):
        _write_rows(Path(raw), [{"model_id": f"model-{i}", "benchmark": f"bench-{'ab'[i]}",
                                 "problem_id": "task", "is_correct": False}])
    config_path = run.save()
    payload = json.loads(config_path.read_text())
    payload.pop("dataset_id")
    config_path.write_text(json.dumps(payload))

    loaded = RunConfig.load(run.run_id)
    rt = load_runtime_from_run(loaded, Config())
    name = rt.library.model_id
    assert name.startswith("mixture-")
    assert rt.dataset.model_id is None
    assert len(rt.dataset.load_items()) == 2
    assert rt.library.catalog_path(tmp_path / "catalog").parent.parent.name == name
    assert len(try_dataset_for_library(rt.library).load_items()) == 2
    loaded.save()
    assert json.loads(config_path.read_text())["dataset_id"] == name


def test_sidecar_identity_matches_explicit_references(tmp_path):
    run = _remote_run(tmp_path)
    name = run.library_model_id()
    for ref in run.reference_data:
        parent = Path(ref["parquet_path"]).parent
        parent.mkdir()
        (parent / "source.json").write_text(json.dumps({
            "base_url": DATA_API, "run_id": ref["run_id"],
        }))
    assert replace(run, dataset_id="", reference_data=[]).library_model_id() == name


# --- Publication groups source data, not model names, paths, or discovery sessions ---

def test_repeated_catalog_saves_reuse_only_unchanged_file_content_hash(tmp_path, monkeypatch):
    import autofa.data.dataset_identity as module

    source = tmp_path / "dataset.parquet"
    source.write_bytes(b"first")
    original = hashlib.file_digest
    calls = []

    def count_reads(handle, algorithm):
        calls.append(handle.name)
        return original(handle, algorithm)

    monkeypatch.setattr(module.hashlib, "file_digest", count_reads)
    run = RunConfig(run_id="analysis", model_id="model", parquet_paths=[str(source)])
    first = CatalogProvenance.from_run(run)
    assert CatalogProvenance.from_run(run).data_identity == first.data_identity
    assert calls == [str(source)]
    previous_stat = source.stat()
    source.write_bytes(b"other")  # Same size, changed content.
    # Force a new cache key even on filesystems with coarse timestamp updates.
    os.utime(source, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 2_000_000_000))
    assert CatalogProvenance.from_run(run).data_identity != first.data_identity
    assert calls == [str(source)] * 2


def test_remote_identity_uses_source_id_and_api_namespace_without_reading_cache(tmp_path):
    missing = tmp_path / "private-cache.parquet"
    ref = reference(missing, base="HTTPS://EVAL.EXAMPLE.ORG:443/api/")
    first = publication_data_identity([missing], reference_data=[ref])
    assert first == {"version": 1, "sources": [
        {"kind": "evalrdl", "origin": "https://eval.example.org/api", "run_id": "source-run"},
    ]}
    moved = tmp_path / "elsewhere.parquet"
    assert publication_data_identity([moved], reference_data=[
        reference(moved, base="https://eval.example.org/api"),
    ]) == first
    for changed in (reference(missing, run_id="different"),
                    reference(missing, base="https://other.example.org/api")):
        assert publication_data_identity([missing], reference_data=[changed]) != first


def test_unbound_remote_sidecar_matches_explicit_identity_without_hashing(tmp_path, monkeypatch):
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"cached remote export")
    (tmp_path / "source.json").write_text(json.dumps({
        "base_url": "HTTPS://EVAL.EXAMPLE.ORG:443/api/", "run_id": "source-run",
    }))
    expected = publication_data_identity([path], reference_data=[
        reference(path, base="https://eval.example.org/api"),
    ])
    monkeypatch.setattr(hashlib, "file_digest", lambda *a, **kw: pytest.fail("hashed export"))
    assert publication_data_identity([path], reference_data=[]) == expected


@pytest.mark.parametrize("binding,sidecar", [
    ("remote", '{"base_url":"https://other.example.org","run_id":"sidecar-source"}'),
    ("local", '{"base_url":"https://other.example.org","run_id":"sidecar-source"}'),
    ("none", "not json"),
    ("none", '{"base_url":"file:///private/data","run_id":"source-run"}'),
], ids=["explicit_remote", "explicit_local", "invalid_json", "unsafe_origin"])
def test_sidecar_never_overrides_bindings_or_local_content_identity(tmp_path, binding, sidecar):
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"data")
    refs = {
        "remote": [reference(path, "explicit-source")],
        "local": [make_local_reference_data(store_id="a" * 64, run_id="r", parquet_path=path)],
        "none": [],
    }[binding]
    expected = publication_data_identity([path], reference_data=refs)
    (tmp_path / "source.json").write_text(sidecar)
    assert publication_data_identity([path], reference_data=refs) == expected


def test_mixed_identity_covers_all_sources_and_is_order_independent(tmp_path):
    remote = tmp_path / "remote.parquet"
    local = tmp_path / "local.parquet"
    local.write_bytes(b"local content")
    refs = [reference(remote)]
    mixed = publication_data_identity([remote, local], reference_data=refs)
    assert {"kind": "local", "sha256": hashlib.sha256(b"local content").hexdigest()} in (
        mixed["sources"]
    )
    assert {row["kind"] for row in mixed["sources"]} == {"evalrdl", "local"}
    assert publication_data_identity([local, remote, local], reference_data=refs * 2) == mixed
    assert publication_data_identity([remote], reference_data=refs) != mixed
    local.unlink()
    assert publication_data_identity([remote, local], reference_data=refs) is None


def test_conflicting_or_unbound_references_do_not_guess_source_ownership(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"data")
    assert publication_data_identity([path], reference_data=[
        reference(path, "first"), reference(path, "second"),
    ]) is None
    assert publication_data_identity([path], reference_data=[reference(None)]) is None
    # A duplicate of a bound source without a path is ambiguous, not the same binding.
    assert publication_data_identity([path], reference_data=[
        reference(path), reference(None),
    ]) is None
    assert publication_data_identity([], reference_data=[reference(None)]) is not None
    assert publication_data_identity([], reference_data=[reference(None), {"broken": True}]) is None
    assert publication_data_identity([], reference_data=[]) is None


def test_analysis_settings_do_not_change_data_identity_and_provenance_roundtrips(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"data")
    earlier = RunConfig(run_id="first-analysis", model_id="m", parquet_paths=[str(path)],
                        eval_limit=200, max_rounds=3)
    later = RunConfig(run_id="second-analysis", model_id="m", parquet_paths=[str(path)],
                      eval_limit=None, max_rounds=18, harness="codex")
    first = CatalogProvenance.from_run(earlier)
    second = CatalogProvenance.from_run(later)
    assert first.data_identity == second.data_identity
    assert first.source_run_id != second.source_run_id
    assert CatalogProvenance.from_dict(first.to_dict()).data_identity == first.data_identity
    path.unlink()
    assert CatalogProvenance.from_run(later).to_dict()["data_identity"] is None


def test_publication_cache_tracks_data_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    path = tmp_path / "data.parquet"
    path.write_bytes(b"original dataset")
    cfg = Config(remote_library_base_url="https://library.example.org")
    run = RunConfig(run_id="analysis", model_id="m", parquet_paths=[str(path)])
    library = run.run_dir() / "library.json"
    library.write_text('{"model_id":"m","functions":[]}')
    receipt = {
        "state": "published", "service_url": cfg.remote_library_base_url,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "data_identity": publication_data_identity([path], reference_data=[]),
    }
    status = run.run_dir() / "publication.json"
    status.write_text(json.dumps(receipt))
    assert published_library_is_current(run, cfg)
    moved = tmp_path / "moved.parquet"
    shutil.move(path, moved)
    run.parquet_paths = [str(moved)]
    assert published_library_is_current(run, cfg)
    moved.write_bytes(b"changed dataset")
    assert not published_library_is_current(run, cfg)
    moved.unlink()
    status.write_text(json.dumps({**receipt, "data_identity": None}))
    assert published_library_is_current(run, cfg)  # Explicitly unknown matches unknown data.


@pytest.mark.parametrize("source", [
    {"kind": "local", "sha256": "a" * 63},
    {"kind": "local", "sha256": "a" * 64, "path": "/private/dataset"},
    {"kind": "evalrdl", "origin": "file:///private/dataset", "run_id": "x"},
], ids=["short_digest", "private_path", "file_origin"])
def test_unsafe_or_malformed_identity_is_rejected(source):
    with pytest.raises(ValueError):
        normalize_data_identity({"version": 1, "sources": [source]})


@pytest.mark.parametrize("origin,run_id,canonical_origin,canonical_run_id", [
    (" HTTPS://EVAL.EXAMPLE.ORG:443/api/ ", " run-1 ", "https://eval.example.org/api", "run-1"),
    ("https://eval.example.org:8443/api", "a.b-c", "https://eval.example.org:8443/api", "a.b-c"),
    ("https://[2001:db8::1]:443/", "run-1", "https://[2001:db8::1]", "run-1"),
    ("https://:password@eval.example.org", "run-1", None, None),
    ("https://eval.exa mple.org", "run-1", None, None),
    ("https://eval.example.org?token=secret", "run-1", None, None),
    ("https://eval.example.org:99999", "run-1", None, None),
    ("https://eval.example.org", "../run-1", None, None),
])
def test_client_server_canonicalization_contract(
    origin, run_id, canonical_origin, canonical_run_id,
):
    payload = {"version": 1, "sources": [{"kind": "evalrdl", "origin": origin, "run_id": run_id}]}
    for normalize in (normalize_data_identity, server_normalize):
        if canonical_origin is None:
            with pytest.raises(ValueError):
                normalize(payload)
        else:
            assert normalize(payload) == {"version": 1, "sources": [
                {"kind": "evalrdl", "origin": canonical_origin, "run_id": canonical_run_id},
            ]}


# --- Reference data survives saving, links reports to exact rollouts and fails closed ---

def _export_url(run_id: str, base: str = DATA_API) -> str:
    return canonical_export_url(base, run_id)


def test_run_config_preserves_remote_source_and_reference_data(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    parquet = tmp_path / "dataset.parquet"
    ref = reference(parquet, "data-run-1")
    run = RunConfig.create(
        [parquet], "model-a", ["bench-a"], Config(), source_paths=[_export_url("data-run-1")],
        source_kind="remote_run", source_harness="nemo-gym", reference_data=[ref],
    )
    assert run.source_paths == [_export_url("data-run-1")]
    loaded = RunConfig.load(run.run_id)
    assert loaded.source_harness == "nemo-gym"
    assert loaded.reference_data == [ref]


def test_legacy_run_config_backfills_remote_cache_sidecar(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    parquet = tmp_path / "remote-cache" / "dataset.parquet"
    parquet.parent.mkdir()
    parquet.write_bytes(b"parquet")
    (parquet.parent / "source.json").write_text(json.dumps({
        "run_id": "data-run-2", "base_url": DATA_API,
        "export_path": "/v1/runs/data-run-2/export/autofa", "catalog": {"harness": "nemo-gym"},
    }), encoding="utf-8")
    RunConfig(run_id="legacy-run", parquet_paths=[str(parquet)], model_id="model-a").save()
    loaded = RunConfig.load("legacy-run")
    assert loaded.source_harness == "nemo-gym"
    assert loaded.reference_data[0]["run_id"] == "data-run-2"
    assert loaded.reference_data[0]["parquet_path"] == str(parquet.resolve())


def test_report_item_links_to_exact_data_api_rollout_and_source_run(tmp_path: Path):
    second = tmp_path / "second.parquet"
    references = [reference(tmp_path / "first.parquet", "run-a"), reference(second, "run-b")]
    item = {
        "_item_id": "item-1", "_source_parquet": str(second), "problem_id": "suite/task",
        "run_idx": 3, "is_correct": False, "benchmark": "bench-a", "prompt_text": "prompt",
        "response_text": "response",
    }
    compact = _compact_item(item, reference_data=references)
    assert compact["source_data_url"] == f"{DATA_API}/#/rollout/run-b/suite%252Ftask/3"
    # A zero repeat is a real rollout index, not a missing one.
    assert rollout_url(references[0], problem_id="0", run_idx=0) == (
        f"{DATA_API}/#/rollout/run-a/0/0"
    )
    report = build_data(FunctionLibrary(model_id="model-a", benchmark="bench-a"),
                        source_harness="nemo-gym", reference_data=references)
    assert report["meta"]["source_harness"] == "nemo-gym"
    assert report["meta"]["reference_data"][1]["inspect_url"] == f"{DATA_API}/#/run/run-b"
    assert "parquet_path" not in report["meta"]["reference_data"][1]
    # The sandboxed report opens source links through its parent frame's bridge.
    assert "namespace: 'autofa.source-data'" in render_html(report)


@pytest.mark.parametrize("case", ["outside", "wrong-cohort", "unlisted"])
def test_local_reference_rejects_unowned_cohort_dataset(tmp_path: Path, monkeypatch, case: str):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("autofa.data.local_trace_store.resolve_store",
                        lambda value: tmp_path / "managed-store")
    cohort = Cohort(runs_where="", where="", sources=[],
                    rows=[{"run_id": "run", "task_id": case, "repeat": 0}])
    if case == "outside":
        recorded = tmp_path / "outside.parquet"
    else:
        recorded = cohorts_root() / cohort.id / "datasets" / "dataset.parquet"
    recorded.parent.mkdir(parents=True, exist_ok=True)
    recorded.write_bytes(b"parquet")
    if case == "wrong-cohort":
        cohort.datasets = [{"kind": "local_evalrdl", "store_id": "b" * 64, "run_id": "run",
                            "parquet_path": str(recorded)}]
    save_cohort(cohort)
    ref = make_local_reference_data(store_id="a" * 64, run_id="run",
                                    source_harness="nemo-gym", parquet_path=recorded)
    with pytest.raises(ValueError, match="does not own its recorded dataset"):
        require_canonical_reference_data([ref], [recorded])


def test_reference_urls_preserve_path_prefixed_data_api():
    export_url = _export_url("run-a", "https://example.test/data")
    ref = {"run_id": "run-a", "export_url": export_url}
    assert export_url == "https://example.test/data/v1/runs/run-a/export/autofa"
    assert data_api_location(ref) == ("https://example.test/data", "run-a")
    assert inspect_url(ref) == "https://example.test/data/#/run/run-a"
    with pytest.raises(ValueError, match="HTTP"):
        canonical_export_url("https://user:secret@example.test?token=x", "run-a")


def test_item_ids_separate_source_runs_but_ignore_cache_paths(tmp_path: Path):
    row = {"model_id": "model-a", "benchmark": "bench-a", "problem_id": "task-1", "run_idx": 0,
           "is_correct": False, "prompt_text": "prompt", "response_text": "response"}
    local = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
    for path in local:
        _write_rows(path, [row])
    item_ids = [item["_item_id"] for batch in Dataset(parquet_paths=local).iter_batches()
                for item in batch]
    assert len(set(item_ids)) == 2

    cached = [tmp_path / machine / "dataset.parquet" for machine in ("machine-a", "machine-b")]
    for path in cached:
        path.parent.mkdir()
        path.write_bytes(b"same export")
        (path.parent / "source.json").write_text(
            json.dumps({"run_id": "run-a", "base_url": DATA_API}), encoding="utf-8",
        )
    assert make_item_id(row, cached[0], include_source=True) == make_item_id(
        row, cached[1], include_source=True,
    )


def test_local_gym_source_harness_is_not_autofa_backend(tmp_path: Path):
    gym_dir = tmp_path / "gym-run"
    gym_dir.mkdir()
    assert _local_source_harness([EvalSource("gym_dir", gym_dir)]) == "nemo_gym"
    (gym_dir / "eval_metadata.json").write_text(
        json.dumps({"harness": "custom-eval-harness"}), encoding="utf-8",
    )
    assert _local_source_harness([EvalSource("gym_dir", gym_dir)]) == "custom-eval-harness"


def test_reference_reuse_fails_closed_for_partial_or_untrusted_provenance():
    valid = {"run_id": "run-a", "export_url": _export_url("run-a")}
    with pytest.raises(SystemExit, match="refusing partial reuse"):
        reuse_reference_data([valid, {"run_id": "broken"}])
    with pytest.raises(SystemExit, match="different Data API"):
        reuse_reference_data([valid], allowed_base_url="https://different.example")


def test_remote_cache_key_cannot_collide_after_run_id_sanitization(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    assert remote_cache_dir("a/b") != remote_cache_dir("a_b")
