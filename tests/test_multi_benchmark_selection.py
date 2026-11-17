# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed-benchmark data stays intact through selection, publication and sampling."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.data.cohort import load_cohort
from autofa.data.parquet_store import Dataset
from autofa.data.remote_library_api import RemotePublication
from autofa.data.remote_library_publish import publish_run
from autofa.mcp import runtime, tools_data
from autofa.reports.failure_cards import _extract_autofa_data, try_dataset_for_library
from autofa.ui import rollout_table
from autofa.ui.menus import MenuItem
from autofa.wizard import dataset, remote_runs

BENCHMARKS = ["swebench-verified", "gdpval", "swebench-pro"]


@pytest.mark.parametrize("multiple_models,legacy_identity", [
    (True, False), (False, True),
], ids=["mixed-models-new-run", "one-model-saved-mixture"])
def test_mixed_remote_selection_reaches_runtime_and_publication(
    tmp_path, monkeypatch, multiple_models, legacy_identity,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime, "_RUNTIME", None)
    # This exercises mixed-run failure analysis; efficiency requires one run.
    cfg = Config(remote_full_traces=False, analysis_outputs=["agent"])
    runs = [
        {
            "run_id": f"run-{benchmark}",
            "benchmark": benchmark,
            "model": f"model-{benchmark}" if multiple_models else "model-a",
            "harness": "nemo-gym",
            "n_rollouts": 2,
        }
        for benchmark in BENCHMARKS
    ]
    downloaded = []

    class DataApi:
        base_url = "https://example.test"

        def list_runs(self):
            return runs

        def get_run(self, run_id):
            return next(run for run in runs if run["run_id"] == run_id)

        def download_autofa_export(self, run_id, dest, *, progress=None):
            downloaded.append(run_id)
            catalog = next(run for run in runs if run["run_id"] == run_id)
            pq.write_table(pa.Table.from_pylist([
                {
                    "model_id": catalog["model"],
                    "benchmark": catalog["benchmark"],
                    "problem_id": "shared-task-id",
                    "run_idx": repeat,
                    "is_correct": bool(repeat),
                    "reward": float(bool(repeat)),
                    "prompt_text": "Solve this task",
                    "response_text": "Attempted solution",
                }
                for repeat in range(2)
            ]), dest)
            return dest

    monkeypatch.setattr(remote_runs, "DataApiClient", lambda base_url: DataApi())
    monkeypatch.setattr(remote_runs, "load_config", lambda: cfg)
    monkeypatch.setattr("autofa.data.cohort.load_config", lambda: cfg)

    def choose_source(title, *args, **kwargs):
        assert title == "Dataset source", "Selected remote models must not be filtered again"
        return MenuItem("Remote", "remote")

    monkeypatch.setattr(dataset, "select_option", choose_source)
    monkeypatch.setattr(
        dataset, "select_items_multi",
        lambda *a, **kw: pytest.fail("Selected remote benchmarks must not be filtered again"),
    )

    # Check one run per search. Earlier selections are hidden by subsequent
    # filters and must still be included when Enter submits the last search.
    with create_pipe_input() as pipe:
        monkeypatch.setattr(
            rollout_table, "Application",
            lambda **kwargs: Application(input=pipe, output=DummyOutput(), **kwargs),
        )
        pipe.send_text(
            "benchmark:swebench-verified\r \t\x15"
            "benchmark:gdpval\r \t\x15"
            "benchmark:swebench-pro\r \r"
        )
        selection = dataset.run_dataset_wizard(api_base_url=DataApi.base_url)

    assert selection.model_id == ("*" if multiple_models else "model-a")
    assert set(selection.benchmarks) == set(BENCHMARKS)
    assert set(downloaded) == {run["run_id"] for run in runs}
    assert len(selection.parquet_paths) == len(set(selection.parquet_paths)) == 3
    assert {ref["run_id"] for ref in selection.reference_data} == set(downloaded)
    # The remote selection keeps the id of its saved cohort.
    saved = load_cohort(selection.selection_id)
    assert {source["run_id"] for source in saved.sources} == set(downloaded)

    run = RunConfig.create(
        selection.parquet_paths,
        selection.model_id,
        selection.benchmarks,
        cfg,
        source_paths=selection.source_paths,
        source_kind=selection.source_kind,
        source_harness=selection.source_harness,
        reference_data=selection.reference_data,
    )
    if legacy_identity:
        from autofa.data.dataset_identity import dataset_mixture_id

        run.dataset_id = dataset_mixture_id(
            run.parquet_paths, model_id=run.model_id,
            benchmarks=run.resolved_benchmarks(), reference_data=run.reference_data,
        )
        run.save()
        FunctionLibrary(model_id=run.dataset_id, benchmark=run.benchmark_label()).save(
            run.run_dir() / "library.json",
        )
    loaded = RunConfig.load(run.run_id)
    rt = runtime.load_runtime_from_run(loaded, cfg)
    items = [item for batch in rt.dataset.iter_batches() for item in batch]
    assert Counter(item["benchmark"] for item in items) == dict.fromkeys(BENCHMARKS, 2)
    assert len({item["_item_id"] for item in items}) == 6
    expected_label = "+".join(sorted(selection.benchmarks))
    assert rt.library.benchmark == expected_label
    library_name = run.library_model_id()
    if multiple_models:
        assert library_name.startswith("mixture-")
    else:
        assert library_name == "model-a"
    assert rt.library.model_id == library_name
    assert loaded.model_id == selection.model_id
    assert len(try_dataset_for_library(rt.library).load_items()) == 6

    uploads = []

    class LibraryApi:
        def upload_library(self, **kwargs):
            uploads.append(kwargs)
            return RemotePublication(library_id="mixed-library", branch_id="branch",
                                     version_id="version",
                                     branch_key=kwargs["metadata"]["branch_key"])

    publish_run(
        loaded, cfg, trigger="manual_cli", runtime=rt, client=LibraryApi(),
        show_progress=False,
    )
    assert len(uploads) == 1
    upload = uploads[0]
    assert upload["metadata"]["model_id"] == library_name
    assert set(upload["metadata"]["benchmarks"]) == set(BENCHMARKS)
    assert {ref["run_id"] for ref in upload["metadata"]["reference_data"]} == set(downloaded)
    assert FunctionLibrary.load(upload["library_json"]).benchmark == expected_label
    assert FunctionLibrary.load(upload["library_json"]).model_id == library_name
    report = _extract_autofa_data(upload["failure_cards_html"].read_text())
    assert report["meta"]["benchmark"] == expected_label
    assert report["meta"]["model_id"] == library_name
    assert report["meta"]["n_items"] == 6
    assert report["meta"]["n_failures"] == 3
    assert {ref["run_id"] for ref in report["meta"]["reference_data"]} == set(downloaded)


def test_managed_local_subset_is_saved_in_the_cohort(tmp_path, monkeypatch):
    parquet = tmp_path / "local.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "model-a", "benchmark": benchmark, "problem_id": f"task-{index}",
         "run_idx": 0, "is_correct": False}
        for index, benchmark in enumerate(BENCHMARKS)
    ]), parquet)
    reference = {"kind": "local_evalrdl", "store_id": "a" * 64, "run_id": "local-run",
                 "source_harness": "nemo-gym", "parquet_path": str(parquet)}
    calls = []
    prepared = SimpleNamespace(model_id="model-a", benchmarks=["gdpval"], source_paths=[],
                               source_kind="evalrdl", selection_id="exact-selection")
    monkeypatch.setattr(dataset, "pick_local_sources",
                        lambda **kwargs: ("evalrdl", [SimpleNamespace(path=parquet)]))
    monkeypatch.setattr(dataset, "_prepare_local_sources", lambda *args, **kwargs: [parquet])
    monkeypatch.setattr(dataset, "select_items_multi", lambda *args, **kwargs: ["gdpval"])
    monkeypatch.setattr("autofa.data.local_import.reference_for_import", lambda path: reference)
    monkeypatch.setattr("autofa.data.cohort.select_cohort",
                        lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr("autofa.data.cohort.prepare_data", lambda cohort, **kwargs: prepared)

    selection = dataset._run_local_wizard()

    assert calls == [{"folders": [parquet], "models": ["model-a"], "benchmarks": ["gdpval"]}]
    assert selection is prepared
    assert selection.selection_id == "exact-selection"
    assert selection.source_paths == [str(parquet.resolve())]


# Coverage investigations do not let a large benchmark hide a small one.


def _sampling_runtime(monkeypatch, tmp_path, rows, selected):
    path = tmp_path / "mixed.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    rt = SimpleNamespace(dataset=Dataset(parquet_paths=[path]), library=FunctionLibrary(),
                         config=Config(), run=SimpleNamespace(
                             run_id="mixed", resolved_benchmarks=lambda: selected))
    monkeypatch.setattr(tools_data, "get_runtime", lambda: rt)


def _failure(benchmark, task, repeat=0, text="Incorrect response."):
    return {"model_id": "m", "benchmark": benchmark, "problem_id": task, "run_idx": repeat,
            "is_correct": False, "response_text": text}


@pytest.mark.parametrize("prefer,selected", [
    (True, ["large", "small"]), (False, []),
], ids=["informative-selected-benchmarks", "random-unlabeled-run"])
def test_unmatched_mixed_sample_balances_benchmarks_without_counting_repeats(
    tmp_path, monkeypatch, prefer, selected,
):
    rows = [_failure("large", str(i), repeat, "Uncorrected failure in a long trace.")
            for i in range(100) for repeat in range(3)]
    rows += [_failure("small", "only", repeat, "Different failure.") for repeat in range(3)]
    rows += [{"model_id": "m", "benchmark": benchmark, "problem_id": "pass", "run_idx": 0,
              "is_correct": True, "response_text": "Correction succeeded."}
             for benchmark in ("large", "small")]
    _sampling_runtime(monkeypatch, tmp_path, rows, selected)
    for seed in range(10):
        result = tools_data.sample_items(n_fail=4, n_pass=2, coverage="unmatched", seed=seed,
                                        prefer_informative=prefer)
        assert result["ok"] and result["by_benchmark"], result
        failed = [row for row in result["items"] if not row["is_correct"]]
        assert {row["benchmark"] for row in failed} == {"large", "small"}
        assert len(failed) == result["n_failed_tasks"] == 4
        assert result["n_repeated_failed_attempts"] == 0
        assert result["n_pass"] == 2
    pooled = tools_data.sample_items(n_fail=4, n_pass=0, coverage="unmatched", by_benchmark=False)
    assert pooled["ok"] and not pooled["by_benchmark"]
    scoped = tools_data.sample_items(n_fail=4, n_pass=0, coverage="unmatched", benchmark="small")
    assert scoped["ok"] and not scoped["by_benchmark"]
    assert scoped["n_fail"] == scoped["n_failed_tasks"] == 1
    assert {row["benchmark"] for row in scoped["items"]} == {"small"}


@pytest.mark.parametrize("labeled,prefer", [
    (["large"], True), (["large", "small"], False),
], ids=["one-label", "two-labels"])
def test_missing_benchmarks_form_one_group_and_do_not_enable_balancing(
    tmp_path, monkeypatch, labeled, prefer,
):
    rows = [_failure(label, str(i)) for label in labeled for i in range(500)]
    rows += [_failure(label, f"unknown-{i}") for i, label in enumerate([None, ""] * 5)]
    _sampling_runtime(monkeypatch, tmp_path, rows, [])
    for seed in range(10):
        result = tools_data.sample_items(n_fail=12, n_pass=0, coverage="unmatched", seed=seed,
                                        prefer_informative=prefer)
        assert result["ok"], result
        assert result["by_benchmark"] == (len(labeled) > 1)
        unknown = sum(not item.get("benchmark") for item in result["items"])
        if len(labeled) > 1:
            assert unknown <= 4
