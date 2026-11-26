# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cohort selection and the prepare command choose rollouts from remote and local runs."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import cli
from autofa.data import cohort as cohort_module
from autofa.data.cohort import (
    Cohort,
    load_cohort,
    prepare_cohort,
    prepare_data,
    save_cohort,
    select_cohort,
)
from autofa.data.evalrdl_efficiency import collect_evalrdl_efficiency
from autofa.data.local_import import import_local_run, reference_for_import
from autofa.data.remote_api import DataApiClient, DataApiError, stage_export_subset
from autofa.taxonomy import groups
from autofa.ui import menus
from autofa.wizard import dataset, remote_runs
from autofa.wizard.dataset import run_dataset_wizard

from support.traces import UrlResponse

pytestmark = pytest.mark.filterwarnings("ignore:Local import retained:UserWarning")

CANON_UUID = "0123456789abcdef" * 2


def _row(task: str, repeat: int, run_id: str = "canon") -> dict:
    return {"run_id": run_id, "task_id": task, "repeat": repeat, "agent_ref": "agent",
            "reward": 0.0}


def _remote_cohort(rows: list[dict], **fields) -> Cohort:
    sources = [{"kind": "remote", "base_url": "https://api.test", "run_id": run_id}
               for run_id in dict.fromkeys(row["run_id"] for row in rows)]
    return Cohort(runs_where="", where="", sources=sources, rows=rows, **fields)


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


class FakeApi:
    """A catalog in which ``alias`` and ``CANON_UUID`` both name the run ``canon``."""

    base_url = "https://api.test"

    def __init__(self, pages: list[list[dict]], *, n_rollouts: int = 3,
                 run_uuids: dict[str, str] | None = None):
        self.pages = pages
        self.n_rollouts = n_rollouts
        self.run_uuids = {"canon": CANON_UUID} if run_uuids is None else run_uuids
        self.sql: list[str] = []
        self.lookups: list[str] = []

    def get_run(self, run_id: str) -> dict:
        self.lookups.append(run_id)
        canonical = "canon" if run_id in {"alias", CANON_UUID} else run_id
        run = {"run_id": canonical, "n_rollouts": self.n_rollouts}
        if canonical in self.run_uuids:
            run["run_uuid"] = self.run_uuids[canonical]
        return run

    def query_analytics(self, sql: str) -> dict:
        self.sql.append(sql)
        return {"rows": self.pages.pop(0), "dataset_contract": "contract"}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _gym_run(tmp_path: Path, rewards: tuple[float, ...]) -> Path:
    run = tmp_path / "gym"
    run.mkdir()
    (run / "rollouts.jsonl").write_text("".join(json.dumps({
        "_ng_task_index": 0, "_ng_rollout_index": repeat, "reward": reward,
        "responses_create_params": {"model": "model", "input": [
            {"role": "user", "content": "Read the file"},
        ]},
        "response": {"status": "completed", "output": [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Done"}],
        }]},
    }) + "\n" for repeat, reward in enumerate(rewards)))
    return run


def test_remote_rollouts_are_run_scoped_and_paged(monkeypatch) -> None:
    monkeypatch.setattr(cohort_module, "PAGE_SIZE", 2)
    api = FakeApi([[_row("t1", 0), _row("t1", 1)], [_row("t2", 0)]])
    cohort = select_cohort(where="reward <= 0", api_runs=["alias"], api=api)
    assert cohort.identities() == [("t1", 0), ("t1", 1), ("t2", 0)]
    assert f"WHERE run_uuid = '{CANON_UUID}' AND (reward <= 0)" in api.sql[0]
    assert (f"WHERE run_uuid = '{CANON_UUID}' AND (reward <= 0) "
            "AND (task_id, repeat, agent_ref) > ('t1', 1, 'agent')") in api.sql[1]
    assert cohort.sources[0]["run_id"] == "canon"
    assert cohort.sources[0]["run_uuid"] == CANON_UUID


@pytest.mark.parametrize("where", ["", "reward <= 0"], ids=["whole_run", "filtered"])
def test_a_run_uuid_and_its_run_id_select_the_same_cohort(where) -> None:
    def select(run: str) -> Cohort:
        return select_cohort(where=where, api_runs=[run], api=FakeApi([[_row("t1", 0)]]))

    by_uuid, by_run_id = select(CANON_UUID), select("canon")
    assert by_uuid.id == by_run_id.id
    assert by_uuid.sources == by_run_id.sources
    assert by_uuid.sources[0]["run_id"] == "canon"
    assert by_uuid.sources[0]["run_uuid"] == CANON_UUID
    # A selection saved before sources recorded run_uuid keeps its id.
    saved = by_uuid.to_json()
    saved["sources"] = [{k: v for k, v in source.items() if k != "run_uuid"}
                        for source in saved["sources"]]
    assert Cohort.from_json(saved).id == by_uuid.id


def test_run_without_a_catalog_uuid_falls_back_to_run_id() -> None:
    api = FakeApi([[_row("t1", 0)]], run_uuids={})
    cohort = select_cohort(where="reward <= 0", api_runs=["alias"], api=api)
    assert "WHERE run_id = 'canon' AND (reward <= 0)" in api.sql[0]
    assert "run_uuid" not in cohort.sources[0]


class RunIdLayoutApi(FakeApi):
    """A server whose trace tables have no run_uuid column yet."""

    def __init__(self, pages: list[list[dict]], error: DataApiError | None = None):
        super().__init__(pages)
        self.error = error or DataApiError(
            "API 400 querying analytics: {\"error\": \"Code: 47. DB::Exception: Unknown "
            "expression identifier `run_uuid` in scope SELECT\"}", status=400,
        )

    def query_analytics(self, sql: str) -> dict:
        if "run_uuid =" in sql:
            self.sql.append(sql)
            raise self.error
        return super().query_analytics(sql)


def test_trace_tables_without_run_uuid_fall_back_to_run_id(monkeypatch) -> None:
    monkeypatch.setattr(cohort_module, "PAGE_SIZE", 1)
    api = RunIdLayoutApi([[_row("t1", 0)], [_row("t2", 0)], []])
    cohort = select_cohort(where="reward <= 0", api_runs=[CANON_UUID], api=api)
    assert cohort.identities() == [("t1", 0), ("t2", 0)]
    assert f"WHERE run_uuid = '{CANON_UUID}' AND (reward <= 0)" in api.sql[0]
    # One rejected query, then every page filters by run_id.
    assert [sql.count("run_uuid =") for sql in api.sql] == [1, 0, 0, 0]
    assert all("WHERE run_id = 'canon' AND (reward <= 0)" in sql for sql in api.sql[1:])
    assert cohort.sources[0]["run_uuid"] == CANON_UUID


@pytest.mark.parametrize("error", [
    DataApiError("API 400 querying analytics: syntax error near reward", status=400),
    DataApiError(
        "API 400 querying analytics: Unknown expression identifier `rewrd` in scope "
        "SELECT task_id FROM di.rollouts WHERE run_uuid = 'x'", status=400,
    ),
    DataApiError("API 503 querying analytics: Missing columns: 'run_uuid'", status=503),
], ids=["other_bad_request", "other_unknown_identifier", "server_error"])
def test_other_query_errors_do_not_fall_back_to_run_id(error) -> None:
    api = RunIdLayoutApi([[_row("t1", 0)]], error)
    with pytest.raises(DataApiError) as raised:
        select_cohort(where="reward <= 0", api_runs=["canon"], api=api)
    assert raised.value is error
    assert len(api.sql) == 1


def test_run_filter_finds_runs_then_filters_their_rollouts() -> None:
    r1_uuid = "1" * 32
    api = FakeApi([[{"run_id": "r1", "run_uuid": r1_uuid}, {"run_id": "r2", "run_uuid": None}],
                   [_row("t1", 0, "r1")], [_row("t1", 0, "r2")]])
    cohort = select_cohort(runs_where="model = 'm'", where="reward <= 0", latest=2, api=api)
    assert [source["run_id"] for source in cohort.sources] == ["r1", "r2"]
    assert "SELECT run_id, any(run_uuid) AS run_uuid, max(run_ts)" in api.sql[0]
    assert "FROM di.runs WHERE run_id IS NOT NULL AND (model = 'm')" in api.sql[0]
    assert "ORDER BY selected_run_ts DESC NULLS LAST, run_id LIMIT 2" in api.sql[0]
    assert f"WHERE run_uuid = '{r1_uuid}' AND" in api.sql[1]
    assert "WHERE run_id = 'r2' AND" in api.sql[2]
    assert api.lookups == []


def test_explicit_runs_and_a_run_filter_keep_their_uuids() -> None:
    # The explicit run's uuid comes from its catalog lookup, not the filter query.
    api = FakeApi([[{"run_id": "canon"}], [_row("t1", 0)]])
    cohort = select_cohort(api_runs=[CANON_UUID], runs_where="model = 'm'", where="true",
                           api=api)
    assert "run_id IN ('canon')" in api.sql[0]
    assert f"WHERE run_uuid = '{CANON_UUID}' AND (true)" in api.sql[1]
    assert cohort.sources[0]["run_uuid"] == CANON_UUID


@pytest.mark.parametrize("saved_uuid", [False, True], ids=["saved_before_uuid", "saved_uuid"])
def test_whole_run_materialization_filters_by_run_uuid(saved_uuid) -> None:
    source = {"kind": "remote", "base_url": "https://api.test", "run_id": "canon",
              "selection": "all", "n_rollouts": 1}
    if saved_uuid:
        source["run_uuid"] = CANON_UUID
    api = FakeApi([[_row("t1", 0)]])
    cohort = Cohort(runs_where="", where="", sources=[source], rows=[])
    materialized = cohort_module._materialize_whole_run_rows(cohort, api=api)
    assert f"WHERE run_uuid = '{CANON_UUID}' ORDER BY" in api.sql[0]
    assert api.lookups == ([] if saved_uuid else ["canon"])
    assert materialized.sources[0]["run_uuid"] == CANON_UUID


def test_analytics_query_posts_sql(monkeypatch) -> None:
    requests = []
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: (
        requests.append(request) or UrlResponse(b'{"rows": []}')
    ))
    assert DataApiClient("https://api.test").query_analytics("SELECT 1") == {"rows": []}
    assert requests[0].full_url == "https://api.test/v1/analytics/query"
    assert json.loads(requests[0].data) == {"sql": "SELECT 1"}


@pytest.mark.parametrize("kwargs,match", [
    ({"folders": [Path("local")], "where": "length(task_id) > 3"}, "require remote EvalRDL runs"),
    ({"api_runs": ["run"], "sample": "random"}, "requires per_run"),
    ({"models": ["model"]}, "Catalog filters require latest"),
    ({"api_runs": ["run"], "latest": 1}, "cannot be combined"),
    ({"api_runs": ["run"], "rewards": ["<=nan"]}, "Invalid reward comparison"),
], ids=["where_on_local", "random_without_per_run", "catalog_without_latest",
        "latest_with_explicit_runs", "invalid_reward"])
def test_select_cohort_rejects_unsupported_filters(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        select_cohort(api=FakeApi([]), **kwargs)


def test_local_folder_uses_evalrdl_tables_and_keeps_provenance(tmp_path) -> None:
    run = _gym_run(tmp_path, (0.0, 1.0))
    cohort = select_cohort(failures=True, folders=[run])
    imported = import_local_run(run)
    assert [row["reward"] for row in cohort.rows] == [0.0]
    [prepared] = prepare_cohort(cohort)
    assert cohort.sources[0]["population"] == {
        "rows": 2, "scored": 2, "zero_reward": 1, "positive_reward": 1, "unscored": 0,
    }
    assert prepared["store_id"] == imported.store_id
    assert prepared["run_id"] == imported.run_id
    assert Path(prepared["parquet_path"]) != imported.parquet_path
    selected = pq.read_table(prepared["parquet_path"]).to_pylist()
    assert [(row["problem_id"], row["run_idx"]) for row in selected] == cohort.identities()
    assert reference_for_import(imported.parquet_path) == imported.reference_data()
    assert select_cohort(folders=[imported.parquet_path]).id == select_cohort(folders=[run]).id
    assert select_cohort(models=["other"], folders=[run]).sources == []


def test_reward_comparisons_combine_for_local_data(tmp_path) -> None:
    run = _gym_run(tmp_path, (0.0, 0.5, 1.0))
    cohort = select_cohort(folders=[run], rewards=[">0", "<1"])
    assert [row["reward"] for row in cohort.rows] == [0.5]


def test_local_efficiency_store_hydrates_traces_only_when_needed(tmp_path) -> None:
    compact = import_local_run(_gym_run(tmp_path, (0.0, 1.0)), efficiency_only=True)
    cohort = select_cohort(folders=[compact.parquet_path])
    [efficiency] = prepare_cohort(cohort, efficiency_only=True)
    assert efficiency["store_id"] == compact.store_id
    [discovery] = prepare_cohort(cohort)
    assert discovery["store_id"] == compact.store_id
    assert Path(discovery["parquet_path"]).parent != compact.parquet_path.parent
    row = pq.read_table(discovery["parquet_path"]).to_pylist()[0]
    assert json.loads(row["response_text"])["evidence_scope"]["identity"] == "evalrdl_tables_v1"


def test_local_efficiency_subset_owns_and_filters_its_tables(tmp_path) -> None:
    compact = import_local_run(_gym_run(tmp_path, (0.0, 1.0)), efficiency_only=True)
    cohort = select_cohort(folders=[compact.parquet_path], per_run=1)
    payload = collect_evalrdl_efficiency(prepare_data(cohort, efficiency_only=True))
    assert payload["n_items"] == 1
    assert payload["n_fail"] == 1


def test_dataset_wizard_reuses_saved_selection(tmp_path) -> None:
    cohort = select_cohort(failures=True, folders=[_gym_run(tmp_path, (0.0, 1.0))])
    save_cohort(cohort)
    prepared = run_dataset_wizard(selection_id=cohort.id)
    assert prepared.selection_id == cohort.id
    assert prepared.source_kind == "evalrdl"
    assert pq.read_table(prepared.parquet_paths[0]).num_rows == 1


def test_dataset_wizard_rejects_saved_selection_from_different_api() -> None:
    cohort = _remote_cohort([_row("task", 0, "run")])
    cohort.sources[0]["base_url"] = "https://other.example"
    save_cohort(cohort)
    with pytest.raises(SystemExit, match="different Data API"):
        run_dataset_wizard(selection_id=cohort.id, api_base_url="https://allowed.example")


def _plain_parquet(path: Path) -> Path:
    return _write_parquet(path, [{
        "model_id": "model", "benchmark": "benchmark", "problem_id": "task", "run_idx": 0,
        "is_correct": False, "prompt_text": "prompt", "response_text": "response",
    }])


def _input(tmp_path: Path, kind: str) -> str:
    """Build one input of ``kind``: plain Parquet, managed import, saved selection or folder."""
    if kind == "parquet":
        return str(_plain_parquet(tmp_path / "dataset.parquet"))
    if kind == "managed":
        return str(import_local_run(_gym_run(tmp_path, (0.0,))).parquet_path)
    if kind == "saved":
        cohort = Cohort(runs_where="", where="", sources=[], rows=[])
        save_cohort(cohort)
        return cohort.id
    folder = tmp_path / kind
    folder.mkdir()
    return str(folder)


def test_dataset_wizard_keeps_plain_parquet_input_compatibility(tmp_path) -> None:
    parquet = _input(tmp_path, "parquet")
    prepared = run_dataset_wizard(inputs=[parquet])
    assert prepared.parquet_paths == [Path(parquet)]
    assert prepared.source_kind == "parquet"


@pytest.mark.parametrize("kinds,options,match", [
    (["local"], {"labels": ["OWN-RSN"]}, "taxonomy filters require taxonomy results"),
    (["saved"], {"api_run_ids": ["run"]}, "cannot be combined"),
    (["parquet"], {"rewards": ["<=0"]}, "Rollout filters require managed EvalRDL data"),
    (["parquet", "managed"], {}, "Do not mix managed EvalRDL and legacy local inputs"),
], ids=["taxonomy_filter_on_folder", "saved_selection_with_api_run",
        "rollout_filter_on_plain_parquet", "managed_and_legacy_mixed"])
def test_dataset_wizard_rejects_invalid_input_combinations(tmp_path, kinds, options, match):
    with pytest.raises(ValueError, match=match):
        run_dataset_wizard(inputs=[_input(tmp_path, kind) for kind in kinds], **options)


def test_dataset_wizard_prepares_interactive_taxonomy_group(monkeypatch, tmp_path) -> None:
    run = _gym_run(tmp_path, (0.0, 1.0))
    failures = select_cohort(failures=True, folders=[run])
    with_controls = select_cohort(folders=[run])
    controls = []
    monkeypatch.setattr(dataset, "prepare_step", lambda: None)
    monkeypatch.setattr(dataset, "select_option", lambda *args, **kwargs: (
        menus.MenuItem("Taxonomy classification results", "taxonomy")
    ))
    monkeypatch.setattr(cohort_module, "_pick_taxonomy_cohort", lambda: failures)
    monkeypatch.setattr(groups, "add_success_controls",
                        lambda cohort, **kwargs: controls.append(cohort.id) or with_controls)
    prepared = dataset.run_dataset_wizard()
    assert controls == [failures.id]
    assert prepared.selection_id == with_controls.id
    assert pq.read_table(prepared.parquet_paths[0]).num_rows == 2


def test_per_run_random_sample_has_the_same_order_remotely_and_locally(tmp_path) -> None:
    api = FakeApi([[_row("t1", 0, "r1")], [_row("t1", 0, "r2")]])
    remote = select_cohort(api_runs=["r1", "r2"], per_run=1, sample="random", seed=7, api=api)
    assert len(remote.rows) == 2
    order = "ORDER BY SHA256(concat('7/', task_id, '/', toString(repeat))), agent_ref LIMIT 1"
    assert all(order in sql for sql in api.sql)

    run = _gym_run(tmp_path, (0.0,) * 5)
    every = select_cohort(folders=[run]).identities()
    local = select_cohort(folders=[run], per_run=2, sample="random", seed=7)

    def rank(identity):
        return hashlib.sha256(f"7/{identity[0]}/{identity[1]}".encode()).digest()

    assert local.identities() == sorted(every, key=rank)[:2]


def test_inputs_reject_missing_mixed_and_multiple_saved_selections(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        cohort_module.resolve_inputs([tmp_path / "missing"])
    first = Cohort(runs_where="", where="", sources=[], rows=[])
    second = _remote_cohort([_row("task", 0, "run")])
    save_cohort(first)
    save_cohort(second)
    with pytest.raises(ValueError, match="Do not mix"):
        cohort_module.resolve_inputs([first.id, _input(tmp_path, "local")])
    with pytest.raises(ValueError, match="one saved selection"):
        cohort_module.resolve_inputs([first.id, second.id])


def test_parent_input_expands_declared_local_runs(tmp_path) -> None:
    parent = tmp_path / "runs"
    (parent / "first").mkdir(parents=True)
    (parent / "second").mkdir()
    first = _gym_run(parent / "first", (0.0,)).resolve()
    second = _gym_run(parent / "second", (1.0,)).resolve()
    kind, inputs = cohort_module.resolve_inputs([parent])
    assert kind == "local"
    assert inputs == [first, second]
    selected = select_cohort(folders=inputs)
    assert {Path(source["path"]) for source in selected.sources} == {first, second}


def test_saved_cohort_can_be_resumed_without_requerying() -> None:
    cohort = select_cohort(api_runs=["run"], where="true", api=FakeApi([[_row("t1", 0, "run")]]))
    path = save_cohort(cohort)
    assert load_cohort(cohort.id).identities() == [("t1", 0)]
    assert load_cohort(path).id == cohort.id


def test_saved_cohort_merges_prepared_metadata_across_runs() -> None:
    rows = [_row("task", 0, "a"), _row("task", 0, "b")]
    for index, run_id in enumerate(["a", "b"]):
        cohort = _remote_cohort(rows)
        cohort.sources[index]["population"] = {"rows": 1}
        cohort.datasets = [{"run_id": run_id, "parquet_path": f"{run_id}.parquet"}]
        save_cohort(cohort)
    saved = load_cohort(cohort.id)
    assert {dataset["run_id"] for dataset in saved.datasets} == {"a", "b"}
    assert {source["run_id"]: source.get("population") for source in saved.sources} == {
        "a": {"rows": 1}, "b": {"rows": 1},
    }


def test_whole_run_refinement_keeps_the_saved_selection_as_parent(monkeypatch) -> None:
    original = _remote_cohort([])
    original.sources = [{"kind": "remote", "base_url": "https://api.test", "run_id": "run",
                         "selection": "all", "n_rollouts": 2}]
    scored = {"status": "ok", "ingest_status": "ok", "error": None, "is_scored": True}
    materialized = _remote_cohort(
        [{**_row("failed", 0, "run"), **scored},
         {**_row("passed", 0, "run"), **scored, "reward": 1.0}],
        parent_selection=original.id,
    )
    monkeypatch.setattr(cohort_module, "_materialize_whole_run_rows",
                        lambda cohort, api=None: materialized)
    refined = cohort_module.refine_cohort(original, rewards=["<=0"])
    assert [row["task_id"] for row in refined.rows] == ["failed"]
    assert refined.parent_selection == original.id


def _patch_remote_prepare(monkeypatch, materialize, *, full_traces: bool) -> None:
    monkeypatch.setattr(cohort_module, "materialize_remote_run", materialize)
    monkeypatch.setattr(cohort_module, "load_config",
                        lambda: SimpleNamespace(remote_full_traces=full_traces))


def test_remote_prepare_always_validates_selection_then_uses_trace_cache(
    monkeypatch, tmp_path,
) -> None:
    complete = _write_parquet(tmp_path / "dataset.parquet", [
        {"problem_id": "t1", "run_idx": 0, "reward": 0.0},
        {"problem_id": "t2", "run_idx": 0, "reward": 1.0},
    ])
    calls = []
    _patch_remote_prepare(monkeypatch, lambda run_id, **kw: complete, full_traces=True)
    monkeypatch.setattr(cohort_module, "stage_export_subset",
                        lambda path, ids: calls.append(ids) or tmp_path / "subset.parquet")
    monkeypatch.setattr(cohort_module, "materialize_source_traces",
                        lambda path, **kw: path.with_name("traces.parquet"))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "n_rows": 2, "n_events": 4, "n_tool_calls": 1, "limitations": {}, "agent_refs": [],
    }))
    api = FakeApi([[_row("t1", 0), _row("t2", 0)]], n_rollouts=2)
    cohort = select_cohort(api_runs=["alias"], where="true", api=api)
    [prepared] = prepare_cohort(cohort, api=api)
    assert calls == [[("t1", 0), ("t2", 0)]]
    assert cohort.sources[0]["population"]["zero_reward"] == 1
    assert prepared["run_id"] == "canon"
    assert prepared["run_uuid"] == CANON_UUID
    assert prepared["export_url"] == "https://api.test/v1/runs/canon/export/autofa"
    assert prepared["parquet_path"] == str(tmp_path / "traces.parquet")
    assert prepared["evidence_scope"].startswith("Original text traces: 2 rollouts")


def test_remote_prepare_refreshes_incomplete_cached_export(monkeypatch, tmp_path) -> None:
    stale = _write_parquet(tmp_path / "stale.parquet",
                           [{"problem_id": "t1", "run_idx": 0, "reward": 0.0}])
    fresh = _write_parquet(tmp_path / "fresh.parquet", [
        {"problem_id": "t1", "run_idx": 0, "reward": 0.0},
        {"problem_id": "t2", "run_idx": 0, "reward": 1.0},
    ])
    force_values = []

    def materialize(run_id, **kwargs):
        force_values.append(kwargs.get("force", False))
        return fresh if kwargs.get("force") else stale

    _patch_remote_prepare(monkeypatch, materialize, full_traces=False)
    api = FakeApi([[_row("t1", 0), _row("t2", 0)]], n_rollouts=2)
    [prepared] = prepare_cohort(select_cohort(api_runs=["alias"], where="true", api=api), api=api)
    assert force_values == [False, True]
    identities = pq.read_table(prepared["parquet_path"], columns=["problem_id", "run_idx"])
    assert identities.to_pylist() == [
        {"problem_id": "t1", "run_idx": 0}, {"problem_id": "t2", "run_idx": 0},
    ]
    assert prepared["evidence_scope"].startswith("Summary-only API export:")


def _three_row_export(tmp_path: Path) -> Path:
    return _write_parquet(tmp_path / "dataset.parquet", [
        {"problem_id": task, "run_idx": 0, "text": text}
        for task, text in [("t1", "a"), ("t2", "b"), ("t3", "c")]
    ])


def test_export_subset_keeps_selected_rows_in_order(tmp_path) -> None:
    source = _three_row_export(tmp_path)
    subset = stage_export_subset(source, [("t2", 0), ("t1", 0)])
    assert pq.read_table(subset).column("text").to_pylist() == ["b", "a"]
    assert stage_export_subset(source, [("t1", 0), ("t2", 0), ("t3", 0)]) == source
    reordered = stage_export_subset(source, [("t3", 0), ("t1", 0), ("t2", 0)])
    assert pq.read_table(reordered).column("text").to_pylist() == ["c", "a", "b"]


def test_export_subset_concurrent_builders_publish_atomically(tmp_path, monkeypatch) -> None:
    source = _three_row_export(tmp_path)
    original = pq.write_table
    barrier = threading.Barrier(2)
    temporary_paths: list[Path] = []

    def synchronized_write(table, path, **kwargs):
        temporary_paths.append(Path(path))
        barrier.wait(timeout=5)
        return original(table, path, **kwargs)

    monkeypatch.setattr(pq, "write_table", synchronized_write)
    selection = [("t2", 0), ("t1", 0)]
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: stage_export_subset(source, selection), range(2)))
    assert results[0] == results[1]
    assert len(set(temporary_paths)) == 2
    assert pq.read_table(results[0]).column("text").to_pylist() == ["b", "a"]
    assert not list(tmp_path.glob("*.partial"))


def test_autofa_prepare_dry_run_combines_simple_flags(monkeypatch, capsys, home) -> None:
    api = FakeApi([[{"run_id": "canon"}], [_row("t1", 0)]])
    monkeypatch.setattr(cohort_module, "DataApiClient", lambda base_url: api)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["prepare", "--api-run", "alias", "--model", "m", "--reward", "<=0",
                  "--task", "t1", "--per-run", "10", "--dry-run", "--api-base", "https://api.test"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "remote run canon: 1 rollout(s) selected" in out
    assert "model IN ('m')" in api.sql[0]
    assert "task_id IN ('t1')" in api.sql[1]
    assert "LIMIT 10" in api.sql[1]
    assert not (home / "cohorts").exists()


@pytest.mark.parametrize("kind,options,message", [
    ("saved", ["--api-run", "run"], "cannot be combined"),
    ("local", ["--label", "OWN-RSN"], "taxonomy filters require taxonomy results"),
], ids=["saved_selection_with_run_filter", "taxonomy_filter_on_folder"])
def test_prepare_rejects_invalid_input_combinations(tmp_path, capsys, kind, options, message):
    with pytest.raises(SystemExit):
        cohort_module.main(["--input", _input(tmp_path, kind), *options, "--dry-run"])
    assert message in capsys.readouterr().err


def test_local_dry_run_uses_temporary_import_storage(tmp_path, capsys, home) -> None:
    run = _gym_run(tmp_path, (0.0, 1.0))
    assert cohort_module.main(["--input", str(run), "--reward", "<=0", "--dry-run"]) == 0
    assert "local run" in capsys.readouterr().out
    assert not (home / "sources").exists()
    assert not (home / "cohorts").exists()


def test_prepare_does_not_save_an_empty_cohort(monkeypatch, capsys, home) -> None:
    monkeypatch.setattr(cohort_module, "DataApiClient", lambda base_url: FakeApi([[]]))
    assert cohort_module.main([
        "--api-run", "run", "--where", "true", "--api-base", "https://api.test",
    ]) == 1
    assert "No rollouts matched" in capsys.readouterr().err
    assert not (home / "cohorts").exists()


def _interactive_source(monkeypatch, value: str) -> None:
    monkeypatch.setattr(cohort_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cohort_module.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(menus, "select_option",
                        lambda *args, **kwargs: menus.MenuItem(value, value))


def test_prepare_without_a_source_uses_the_interactive_picker(monkeypatch, capsys, home) -> None:
    api = FakeApi([[_row("t1", 0)]])
    monkeypatch.setattr(cohort_module, "DataApiClient", lambda base_url: api)
    _interactive_source(monkeypatch, "runs")
    monkeypatch.setattr(cohort_module, "_pick_interactive_sources",
                        lambda *args, **kwargs: (["canon"], []))
    assert cohort_module.main(["--dry-run", "--api-base", "https://api.test"]) == 0
    assert "remote run canon: 3 rollout(s) selected" in capsys.readouterr().out
    assert api.sql == []
    assert not (home / "cohorts").exists()


def test_prepare_can_pick_taxonomy_results_interactively(monkeypatch, capsys, home) -> None:
    cohort = _remote_cohort([_row("task", 0, "run")], parent_selection="parent",
                            taxonomy_filter={"labels": ["X"]})
    _interactive_source(monkeypatch, "taxonomy")
    monkeypatch.setattr(cohort_module, "_pick_taxonomy_cohort", lambda: cohort)
    monkeypatch.setattr(cohort_module, "_pick_interactive_sources",
                        lambda *args, **kwargs: pytest.fail("run picker should not open"))

    def add_controls(selected, **kwargs):
        selected.rows.append({**_row("pass", 0, "run"), "reward": 1.0})
        return selected

    monkeypatch.setattr(groups, "add_success_controls", add_controls)
    assert cohort_module.main(["--dry-run"]) == 0
    assert "remote run run: 2 rollout(s) selected" in capsys.readouterr().out
    assert not (home / "cohorts").exists()


@pytest.mark.parametrize("choice,expected", [
    ("remote", (["first", "second"], [])),
    ("local", ([], ["gym"])),
])
def test_interactive_picker_routes_source_choice(monkeypatch, choice, expected) -> None:
    monkeypatch.setattr(menus, "select_option",
                        lambda *args, **kwargs: menus.MenuItem(choice, choice))
    monkeypatch.setattr(remote_runs, "pick_remote_runs",
                        lambda api: [{"run_id": "first"}, {"run_id": "second"}])
    monkeypatch.setattr(dataset, "pick_local_sources", lambda *, raw_only: (
        ("gym_dir", [SimpleNamespace(path=Path("gym"))]) if raw_only else None
    ))
    assert cohort_module._pick_interactive_sources(FakeApi([])) == expected


def test_interactive_taxonomy_picker_collects_subset_filters(monkeypatch, tmp_path) -> None:
    results = tmp_path / "results"
    expected = Cohort(runs_where="", where="", sources=[], rows=[], parent_selection="parent")
    choices = iter([menus.MenuItem("Judge A", "judge-a"), menus.MenuItem("Needs review", "review")])
    selected = iter([["OWN-RSN"], ["partial"]])
    captured = {}
    monkeypatch.setattr(menus, "browse_filesystem", lambda *args, **kwargs: results)
    monkeypatch.setattr(menus, "select_option", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(menus, "select_items_multi", lambda *args, **kwargs: next(selected))
    monkeypatch.setattr(groups, "classification_facets", lambda paths: {
        "paths": [results / "taxonomy_classifications.jsonl"], "judges": ["judge-a"],
        "labels": {"OWN-RSN": "Reasoning"}, "evidence": ["partial"],
    })
    monkeypatch.setattr(groups, "cohort_from_classifications",
                        lambda paths, **kwargs: captured.update(paths=paths, **kwargs) or expected)
    assert cohort_module._pick_taxonomy_cohort() is expected
    assert captured == {
        "paths": [results / "taxonomy_classifications.jsonl"], "judge": "judge-a",
        "group": "review", "labels": ["OWN-RSN"], "evidence": ["partial"],
    }
