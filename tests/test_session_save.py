# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session preflight releases its traces, and /save persists locally before publishing."""

import gc
import json
import weakref
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data import corpus_health
from autofa.data.library_provenance import catalog_provenance_path
from autofa.data.remote_library_api import RemoteLibraryError, RemotePublication
from autofa.harness.commands import handle_local_command
from autofa.mcp.runtime import Runtime
from autofa.session.exploration import resolve_exploration_seed
from autofa.session.runner import _dataset_health


@pytest.fixture
def snapshot_library(monkeypatch):
    """Saving any validation state must remain a persistence-only operation."""
    from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
    from autofa.metrics.claude_judge import judge_policy_hash

    def unexpected_work(*args, **kwargs):
        pytest.fail("Saving must not evaluate detectors, run a census, or call the judge")

    for operation in ("evaluate_functions", "_judge_function"):
        monkeypatch.setattr(f"autofa.mcp.tools_eval.{operation}", unexpected_work)

    functions = []
    for state in ("sampled_llm", "uncertain", "processing_error", "unjudged", "missing", "stale"):
        fn = FeatureFunction(
            name=state, description="Removes the only backup before restoring the data",
            mechanism="Deleting the backup prevents recovery after the subsequent failed write.",
            source=f"def {state}(item, llm_client=None):\n    return True\n",
            kind="llm" if state == "sampled_llm" else "regex",
        )
        fn.scores = {
            "evaluated": True, "sampled": state == "sampled_llm",
            "eval_mode": "sample" if state == "sampled_llm" else "full",
            "n_items": 8, "a": 4, "b": 0, "c": 4,
            "recall": .5, "precision": 1, "ppv_at_pi0": 1,
            "behavior_purity": None, "behavior_specificity": None,
        }
        if state in {"uncertain", "processing_error", "stale"}:
            fn.scores.update(
                judge_contract_version=JUDGE_CONTRACT_VERSION,
                judge_policy_hash=judge_policy_hash(),
                claim_hash=claim_hash(
                    source=fn.source, description=fn.description, mechanism=fn.mechanism,
                ),
                behavior_n_judged=4, behavior_n_yes=0, behavior_n_no=0,
                behavior_n_uncertain=4 if state == "uncertain" else 0,
                behavior_n_errors=4 if state == "processing_error" else 0,
            )
        if state == "stale":
            fn.scores.update(behavior_n_yes=4, behavior_purity=1, behavior_specificity=.8)
            fn.mechanism = "An edited mechanism awaiting explicit validation."
        elif state == "unjudged":
            fn.scores = {}
        elif state == "missing":
            fn.scores.update(recall=None, ppv_at_pi0=None)
        functions.append(fn)
    return FunctionLibrary(model_id="model-a", benchmark="bench-a", functions=functions)


def _persisted(library, run):
    """The saved form: unchanged scores plus the run's initial exploration cursor."""
    expected = deepcopy(library.to_dict())
    expected["exploration_state"] = {
        "version": 1, "base_seed": resolve_exploration_seed(run_id=run.run_id),
        "sequence_index": 0,
    }
    return expected


@pytest.mark.parametrize("mode,outcome", [
    ("server", "published"), ("cloud", "failed"), ("cloud", "skipped"),
])
def test_save_publishes_current_report_and_preserves_local_copy(
    tmp_path, monkeypatch, snapshot_library, mode, outcome,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    cfg = Config(
        library_root=str(tmp_path / "catalog"), remote_library_mode=mode,
        remote_library_base_url=f"https://{mode}.example.test",
    )
    run = RunConfig(run_id="run-1", parquet_paths=[], model_id="model-a", benchmarks=["bench-a"])
    library = snapshot_library
    expected_library = _persisted(library, run)
    rt = Runtime(dataset=SimpleNamespace(parquet_paths=[]), library=library, run=run, config=cfg)
    monkeypatch.setattr("autofa.mcp.runtime._RUNTIME", rt)
    catalog = library.catalog_path(cfg.resolved_library_root())
    uploads, reports = [], []

    class Client:
        def upload_library(self, **kwargs):
            assert rt.working_library_path().is_file()
            assert catalog.is_file()
            assert kwargs["metadata"]["publish_trigger"] == "manual_cli"
            assert json.loads(Path(kwargs["library_json"]).read_text()) == expected_library
            assert json.loads(catalog.read_text()) == expected_library
            assert Path(kwargs["failure_cards_html"]).read_text() == reports[-1]
            assert kwargs["report_assets_zip"] == b"report assets"
            uploads.append(kwargs)
            if outcome == "failed":
                raise RemoteLibraryError("service unavailable")
            return RemotePublication(
                library_id="library-1", branch_id="branch-1",
                branch_key=kwargs["metadata"]["branch_key"],
                version_id=f"version-{len(uploads)}", version_number=len(uploads),
                source_run_id=run.run_id,
                dashboard_url=f"{cfg.remote_library_base_url}/libraries/library-1",
            )

    client = Client()

    def configured_client(selected_cfg):
        assert selected_cfg is cfg
        assert rt.working_library_path().is_file() and catalog.is_file()
        if outcome == "skipped":
            raise RemoteLibraryError("no credential; run autofa login")
        return client

    def write_report(selected_library, *, out, dataset, **kwargs):
        assert selected_library is library and dataset is rt.dataset
        assert Path(out).name == (
            "failure_cards_local.html" if mode == "server" else "failure_cards_cloud.html"
        )
        reports.append(f"<html>Current report {len(reports) + 1}</html>")
        Path(out).write_text(reports[-1])
        return Path(out)

    monkeypatch.setattr("autofa.data.remote_library_publish.client_from_config", configured_client)
    monkeypatch.setattr(
        "autofa.reports.failure_cards.write_failure_cards_with_status", write_report,
    )
    monkeypatch.setattr(
        "autofa.reports.failure_cards.build_report_assets_zip", lambda *a, **k: b"report assets",
    )
    monkeypatch.setattr("autofa.data.trace_upload.server_capabilities", lambda selected: {})
    output = StringIO()
    console = Console(file=output, width=200)
    assert handle_local_command("/save", console) == "handled"
    assert rt.working_library_path().is_file() and catalog.is_file()
    assert library.to_dict() == expected_library
    assert json.loads(rt.working_library_path().read_text()) == expected_library
    assert json.loads(catalog.read_text()) == expected_library
    status = json.loads((run.run_dir() / "publication.json").read_text())
    assert status["state"] == outcome
    assert status["trigger"] == "manual_cli"
    if outcome == "published":
        assert run.remote_version_id == "version-1"
        assert cfg.remote_library_base_url in output.getvalue()
        # Saving again builds the report again and appends to the same remote branch.
        assert handle_local_command("/save", console) == "handled"
        assert len(uploads) == len(reports) == 2
        assert uploads[1]["metadata"]["library_id"] == "library-1"
        assert uploads[1]["metadata"]["branch_id"] == "branch-1"
        provenance = json.loads(catalog_provenance_path(catalog).read_text())
        assert provenance["remote_version_id"] == "version-2"
        assert RunConfig.load(run.run_id).remote_version_id == "version-2"
        assert library.to_dict() == expected_library
        assert json.loads(rt.working_library_path().read_text()) == expected_library
        assert json.loads(catalog.read_text()) == expected_library
    else:
        assert "local library is safe" in output.getvalue()
        assert len(uploads) == (1 if outcome == "failed" else 0)
        if outcome == "failed":
            assert "Retry: autofa publish --run run-1" in output.getvalue()


@pytest.mark.parametrize("failure", ["exception", "result"])
def test_failed_local_save_never_publishes(monkeypatch, failure):
    def save(**kwargs):
        if failure == "exception":
            raise OSError("disk full")
        return {"ok": False, "error": "disk full"}

    monkeypatch.setattr("autofa.mcp.tools_library.library_save", save)
    monkeypatch.setattr("autofa.data.remote_library_publish.maybe_publish_run",
                        lambda *a, **k: pytest.fail("Local save must complete before publication"))
    output = StringIO()
    assert handle_local_command("/save", Console(file=output)) == "handled"
    assert "Save failed: disk full" in output.getvalue()


def test_preflight_keeps_only_summary_and_domain(monkeypatch):
    class TraceRow(dict):
        pass

    refs = []
    closed = []

    class Dataset:
        def iter_batches(self, *, batch_size, max_items):
            assert batch_size == 32
            assert max_items == 40
            try:
                for i in range(40):
                    row = TraceRow(_item_id=str(i), response_text="x" * 10_000)
                    refs.append(weakref.ref(row))
                    yield [row]
            finally:
                closed.append(True)

    def coverage(items, *, record_diagnostics):
        assert record_diagnostics
        assert [row["_item_id"] for row in items] == [str(i) for i in range(40)]
        return {"n_sampled": len(items), "trajectory_capable": True}

    def domain(items, measured):
        assert len(items) == measured["n_sampled"] == 40
        return "terminal"

    monkeypatch.setattr(corpus_health, "measure_parse_coverage", coverage)
    monkeypatch.setattr(corpus_health, "detect_corpus_domain", domain)
    summary, inferred_domain = _dataset_health(Dataset())
    gc.collect()
    assert summary == {"n_sampled": 40, "trajectory_capable": True}
    assert inferred_domain == "terminal"
    assert closed == [True]
    assert all(ref() is None for ref in refs)


def test_preflight_still_handles_empty_corpus(monkeypatch):
    class Dataset:
        def iter_batches(self, **kwargs):
            return iter(())

    monkeypatch.setattr(corpus_health, "measure_parse_coverage", lambda items, **kwargs: {
        "n_sampled": len(items), "trajectory_capable": False,
    })
    monkeypatch.setattr(corpus_health, "detect_corpus_domain", lambda items, coverage: "general")
    summary, domain = _dataset_health(Dataset())
    assert summary["n_sampled"] == 0
    assert domain == "general"
