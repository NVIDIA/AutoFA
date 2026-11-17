# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EvalRDL efficiency: selection gates, prepared-data readiness and token accounting."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import polars as pl
import pytest

from autofa import cli
from autofa.config import Config
from autofa.data import cohort as cohort_module
from autofa.data.cohort import Cohort, prepare_cohort
from autofa.data.efficiency_support import (
    EfficiencyGate,
    check_local_sources,
    check_prepared,
    content_warnings,
    efficiency_gate,
)
from autofa.data.evalrdl_efficiency import (
    EfficiencyReadiness,
    UnsupportedEfficiencySelection,
    collect_evalrdl_efficiency,
    validate_efficiency_run_selection,
    verify_efficiency_data,
)
from autofa.data.sources import EvalSource
from autofa.wizard.dataset import DatasetSelection

SELECTED = {"source": "model_config", "model_id": "selected-model"}


def _export_url(run_id: str) -> str:
    return f"https://runs.example.test/v1/runs/{run_id}/export/autofa"


def _run(root):
    """A prepared run whose tables are indexed the way a remote trace download stores them."""
    dataset = root / "dataset.parquet"
    if not dataset.is_file():
        pl.DataFrame({"model_id": ["m"]}).write_parquet(dataset)
    tables = root / "tables"
    if tables.is_dir():
        digest = root.name
        (root / "manifest.json").write_text(json.dumps({
            "trace_exports": [{"sha256": digest, "rollouts": None}],
        }))
        index = root.parent / f"source-bulk-{root.name}" / "trace.sqlite"
        index.parent.mkdir(exist_ok=True)
        (index.parent / "download.json").write_text(json.dumps({
            "run_id": "r", "archive_sha256": digest,
        }))
        with sqlite3.connect(index) as db:
            db.execute("CREATE TABLE records (tbl TEXT, payload TEXT)")
            for path in tables.glob("*.parquet"):
                db.executemany(
                    "INSERT INTO records VALUES (?, ?)",
                    [(path.stem, json.dumps(row)) for row in pl.read_parquet(path).to_dicts()],
                )
    return SimpleNamespace(
        reference_data=[{"run_id": "r", "export_url": _export_url("r"),
                         "parquet_path": str(dataset)}],
        parquet_paths=[str(dataset)],
        source_paths=[],
    )


def _table(root, name, **columns) -> None:
    """Write one EvalRDL table; identity columns default to run ``r``, repeat 0, one agent."""
    n = len(columns["task_id"])
    identity = {"run_id": ["r"] * n, "repeat": [0] * n, "agent_ref": ["agent"] * n}
    (root / "tables").mkdir(exist_ok=True)
    pl.DataFrame({**identity, **columns}).write_parquet(root / "tables" / f"{name}.parquet")


def _metrics(payload) -> dict:
    return {metric["name"]: metric for metric in payload["metrics"]}


def _provenance(payload) -> dict:
    return {row["name"]: row for row in payload["provenance"]}


# --- Which selections can produce an efficiency report ---

def test_efficiency_selection_requires_one_run(tmp_path) -> None:
    paths = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
    for path in paths:
        pl.DataFrame({"x": [1]}).write_parquet(path)
    selection = SimpleNamespace(
        reference_data=[{"run_id": path.stem, "export_url": _export_url(path.stem),
                         "parquet_path": str(path)} for path in paths],
        parquet_paths=paths,
    )
    with pytest.raises(UnsupportedEfficiencySelection, match="exactly one"):
        validate_efficiency_run_selection(selection)


def test_efficiency_selection_accepts_duplicate_references_to_one_run(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a"], reward=[1.0])
    selection = _run(tmp_path)
    selection.reference_data *= 2
    assert validate_efficiency_run_selection(selection).endswith(":run:r")


def test_efficiency_selection_rejects_unreferenced_paths(tmp_path) -> None:
    selection = SimpleNamespace(
        reference_data=[], parquet_paths=[tmp_path / "first.parquet", tmp_path / "second.parquet"],
    )
    with pytest.raises(ValueError, match="requires data imported through EvalRDL"):
        validate_efficiency_run_selection(selection)


# The report collector re-checks the run because the data can change after the early check.
@pytest.mark.parametrize("run_ids,reason,check", [
    (["first", "second"], "different evaluation run", verify_efficiency_data),
    (["first", "second"], "different evaluation run", collect_evalrdl_efficiency),
    ([], "no rollouts", verify_efficiency_data),
], ids=["mixed_runs_check", "mixed_runs_report", "empty_check"])
def test_prepared_data_check_and_report_reject_unusable_runs(
    tmp_path, run_ids, reason, check,
) -> None:
    n = len(run_ids)
    (tmp_path / "tables").mkdir()
    pl.DataFrame(
        {"run_id": run_ids, "task_id": ["a", "b"][:n], "repeat": [0] * n,
         "agent_ref": ["agent"] * n},
        schema={"run_id": pl.String, "task_id": pl.String, "repeat": pl.Int64,
                "agent_ref": pl.String},
    ).write_parquet(tmp_path / "tables" / "rollouts.parquet")
    with pytest.raises(UnsupportedEfficiencySelection, match=reason):
        check(_run(tmp_path))


def test_absent_tables_warn_and_compute_nothing(tmp_path) -> None:
    with pytest.raises(UnsupportedEfficiencySelection, match="trace-batch manifest"):
        collect_evalrdl_efficiency(_run(tmp_path))


def test_prepared_data_check_counts_what_the_report_will_show(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a", "b", "c"], reward=[1.0, 0.0, None],
           prompt_tokens=[10, None, None], completion_tokens=[4, None, None])
    _table(tmp_path, "events", task_id=["b"], lane=["model_call"], event_idx=[0],
           request_idx=[0], model_name=["m"], prompt_tokens=[20], completion_tokens=[5])
    run = _run(tmp_path)
    readiness = verify_efficiency_data(run)
    payload = collect_evalrdl_efficiency(run)
    cost = _metrics(payload)["simulated_cost_usd"]
    assert readiness == EfficiencyReadiness(n_attempts=3, n_pass=1, n_fail=1, n_with_tokens=2)
    assert (payload["n_items"], payload["n_pass"], payload["n_fail"]) == (3, 1, 1)
    assert cost["coverage"] * payload["n_items"] == pytest.approx(readiness.n_with_tokens)


def test_collector_filters_tables_to_selected_rollouts(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["keep", "omit"], reward=[0.0, 1.0])
    run = _run(tmp_path)
    pl.DataFrame({"model_id": ["m"], "problem_id": ["keep"], "run_idx": [0]}).write_parquet(
        run.parquet_paths[0],
    )
    payload = collect_evalrdl_efficiency(run)
    assert (payload["n_items"], payload["n_pass"], payload["n_fail"]) == (1, 0, 1)


def test_remote_trace_batches_for_one_run_are_combined(tmp_path) -> None:
    export = tmp_path / "source-v14"
    export.mkdir()
    pl.DataFrame({"model_id": ["m"]}).write_parquet(export / "dataset.parquet")
    (export / "manifest.json").write_text(json.dumps({"trace_exports": [
        {"sha256": "first", "rollouts": [["a", 0]]},
        {"sha256": "second", "rollouts": [["b", 0]]},
    ]}))
    for digest, task, reward in (("first", "a", 1.0), ("second", "b", 0.0)):
        root = tmp_path / "source-bulk-abc" / f"batch-{digest}"
        root.mkdir(parents=True)
        (root / "download.json").write_text(json.dumps({"run_id": "r", "archive_sha256": digest}))
        with sqlite3.connect(root / "trace.sqlite") as db:
            db.execute("CREATE TABLE records (tbl TEXT, payload TEXT)")
            db.execute("INSERT INTO records VALUES (?, ?)", ("rollouts", json.dumps({
                "run_id": "r", "task_id": task, "repeat": 0, "agent_ref": "agent",
                "reward": reward, "prompt_tokens": 10, "completion_tokens": 5, "n_tool_calls": 0,
            })))
    payload = collect_evalrdl_efficiency(_run(export))
    assert (payload["n_items"], payload["n_pass"], payload["n_fail"]) == (2, 1, 1)


# --- Token and cost accounting ---

def test_verdict_only_import_has_prominent_missing_rollout_warning(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=[None], repeat=[None], agent_ref=[None], reward=[None],
           prompt_tokens=[None], completion_tokens=[None], n_assistant_turns=[0],
           n_tool_calls=[0])
    _table(tmp_path, "events", task_id=[None], repeat=[None], agent_ref=[None],
           lane=["verdict"], event_idx=[0])
    payload = collect_evalrdl_efficiency(_run(tmp_path))
    assert payload["metrics"] == []
    assert len(payload["alerts"]) == 1
    assert "no scored results or efficiency measurements" in payload["alerts"][0]


def test_missing_cached_tokens_are_omitted_with_a_warning(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a", "b"], reward=[1.0, 0.0],
           prompt_tokens=[10, 30], completion_tokens=[4, 6], n_assistant_turns=[2, 3],
           n_tool_calls=[1, 0])
    _table(tmp_path, "events", task_id=["a", "a", "b", "a"], lane=["", "", "", "model_call"],
           event_idx=[0, 1, 0, 0], request_idx=[0, 0, 0, 0], cached_tokens=[None] * 4,
           reasoning_tokens=[2, 2, None, None], completion_tokens=[None, None, None, 6])
    payload = collect_evalrdl_efficiency(_run(tmp_path))
    metrics = _metrics(payload)
    assert "input_tokens" in metrics
    assert not {"cached_tokens", "uncached_tokens", "reported_cost_usd"} & metrics.keys()
    assert any("cached tokens" in warning for warning in payload["warnings"])
    cost = metrics["simulated_cost_usd"]
    # Missing cached tokens count as zero: (tokens × $0.15 + completion × $0.50) / 1e6.
    assert cost["pass"]["mean"] == pytest.approx((10 * 0.15 + 6 * 0.50) / 1_000_000)
    assert cost["fail"]["mean"] == pytest.approx((30 * 0.15 + 6 * 0.50) / 1_000_000)
    assert cost["sources"] == {"simulated_fallback": 2}
    assert not any("simulated cost was not computed" in warning for warning in payload["warnings"])
    assert (metrics["reasoning_tokens"]["overall"]["total"],
            metrics["reasoning_tokens"]["overall"]["n"]) == (2, 1)
    assert any("1 of 2" in warning for warning in payload["warnings"])
    assert payload["has_agent_activity"] is True
    tested = metrics["input_tokens"]
    assert tested["permutation_test"]["method"] == "problem_cluster_permutation_ks"
    assert 0 < tested["permutation_test"]["p_value"] <= 1
    provenance = _provenance(payload)
    assert (provenance["input_tokens"]["status"], provenance["input_tokens"]["present"]) == (
        "recorded", 2,
    )
    assert (provenance["cached_tokens"]["status"], provenance["cached_tokens"]["present"]) == (
        "unavailable", 0,
    )
    assert "uncached_tokens" not in provenance
    assert provenance["simulated_cost_usd"]["status"] == "derived"


def test_cached_tokens_use_the_cached_rate(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a"], reward=[1.0], prompt_tokens=[100],
           completion_tokens=[10])
    _table(tmp_path, "events", task_id=["a"], request_idx=[0], cached_tokens=[40])
    payload = collect_evalrdl_efficiency(_run(tmp_path), pricing={
        "source": "model_config", "input_usd_per_million": 3.0,
        "cached_usd_per_million": 0.5, "output_usd_per_million": 10.0,
    })
    cost = _metrics(payload)["simulated_cost_usd"]
    assert cost["overall"]["mean"] == (60 * 3 + 40 * 0.5 + 10 * 10) / 1_000_000
    assert cost["sources"] == {"model_config": 1}


def test_uncached_input_is_omitted_where_cached_tokens_are_missing(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a", "b"], reward=[1.0, 0.0],
           prompt_tokens=[100, 300], completion_tokens=[10, 10])
    _table(tmp_path, "events", task_id=["a", "b"], request_idx=[0, 0], cached_tokens=[40, None])
    payload = collect_evalrdl_efficiency(_run(tmp_path))
    uncached = _metrics(payload)["uncached_tokens"]["overall"]
    assert (uncached["n"], uncached["total"]) == (1, 60)
    assert _provenance(payload)["uncached_tokens"]["status"] == "derived"


def test_rollout_columns_are_used_and_event_copies_are_not_summed(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a", "b", "c"], reward=[1.0, 0.0, None],
           prompt_tokens=[10, 20, None], completion_tokens=[1, 1, None],
           total_tokens=[50, 21, None], n_assistant_turns=[1, 1, 1], n_tool_calls=[0, 0, 0],
           n_reasoning_segments=[3, 1, None])
    # history is a second account of a and b, and is the larger one for each.
    _table(tmp_path, "events",
           task_id=["a", "a", "a", "a", "a", "a", "b", "c", "b"],
           lane=["trace", "trace", "trace", "subagent/worker", "model_call", "history",
                 "trace", "trace", "history"],
           event_idx=[0, 1, 2, 0, 0, 0, 0, 0, 0],
           request_idx=[0, 0, 1, 0, 0, 0, 0, 0, 0],
           prompt_tokens=[None] * 7 + [999, None],
           cached_tokens=[10, 99, 5, 7, 1000, 100, 4, 3, 30],
           reasoning_tokens=[8, 8, None, None, 50, None, None, None, None],
           completion_tokens=[None, None, None, None, 9, None, None, None, None])
    payload = collect_evalrdl_efficiency(_run(tmp_path))
    metrics = _metrics(payload)
    # trace 10 (once) + 5 + subagent 7 = 22, against history 100.
    assert metrics["cached_tokens"]["pass"]["mean"] == 100
    # trace 4 against history 30.
    assert metrics["cached_tokens"]["fail"]["mean"] == 30
    assert metrics["reasoning_tokens"]["overall"]["total"] == 8
    # Transcript usage is already represented by EvalRDL rollups and is not a fallback.
    assert metrics["input_tokens"]["overall"]["total"] == 30
    # a's rollout completion excludes its eight reasoning tokens.
    assert metrics["total_tokens"]["pass"]["mean"] == 19
    assert metrics["total_tokens"]["fail"]["mean"] == 21
    assert metrics["total_tokens"]["overall"]["n"] == 2
    assert payload["has_agent_activity"] is False
    assert not {"n_reasoning_segments", "n_turns", "n_tool_calls",
                "final_context_tokens"} & metrics.keys()
    assert not any("tool-argument" in warning for warning in payload["warnings"])
    assert _provenance(payload)["agent_activity"]["status"] == "not_applicable"


def test_cached_token_rows_without_request_indexes_each_count_once(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a"], reward=[1.0], prompt_tokens=[10],
           completion_tokens=[2])
    _table(tmp_path, "events", task_id=["a", "a"], lane=["trace", "trace"], event_idx=[0, 1],
           request_idx=[None, None], cached_tokens=[12, 12])
    payload = collect_evalrdl_efficiency(_run(tmp_path))
    assert _metrics(payload)["cached_tokens"]["overall"]["total"] == 24


def test_remote_trace_index_supplies_rollout_totals(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a"], reward=[1.0], prompt_tokens=[120],
           completion_tokens=[40], n_assistant_turns=[1], n_tool_calls=[0],
           n_reasoning_segments=[1])
    _table(tmp_path, "events", task_id=["a"] * 3, lane=["", "model_call", "model_call"],
           event_idx=[0, 0, 1], request_idx=[0, 1, 0],
           model_name=[None, "selected-model", "policy-model"],
           cached_tokens=[None, 900, 10_000], reasoning_tokens=[15, 800, 10_000],
           completion_tokens=[None, 40, 10_000])
    payload = collect_evalrdl_efficiency(_run(tmp_path), pricing=SELECTED)
    metrics = _metrics(payload)
    assert {name: metrics[name]["overall"]["mean"] for name in (
        "input_tokens", "total_tokens", "cached_tokens", "answer_tokens", "output_tokens",
    )} == {"input_tokens": 120, "total_tokens": 160, "cached_tokens": 900,
           "answer_tokens": 25, "output_tokens": 40}
    assert metrics["reasoning_tokens"]["overall"]["total"] == 15


def test_selected_model_requests_fill_missing_rollout_tokens(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a"], reward=[1.0], prompt_tokens=[None],
           completion_tokens=[None])
    _table(tmp_path, "events", task_id=["a"] * 4, lane=["", "model_call", "model_call",
                                                        "model_call"],
           event_idx=[0, 0, 1, 2], request_idx=[0, 0, 1, 0],
           model_name=[None, "selected-model", "selected-model", "policy-model"],
           prompt_tokens=[0, 10, 20, 999], completion_tokens=[0, 1, 2, 999])
    metrics = _metrics(collect_evalrdl_efficiency(_run(tmp_path), pricing=SELECTED))
    assert metrics["input_tokens"]["overall"]["total"] == 30
    assert metrics["completion_tokens"]["overall"]["total"] == 3
    assert metrics["total_tokens"]["overall"]["total"] == 33


@pytest.mark.parametrize("auxiliary_model,include_selected", [
    ("policy-model", False), (None, True),
], ids=["other_model_only", "unnamed_beside_empty_selected"])
def test_missing_selected_model_usage_never_uses_auxiliary_requests(
    tmp_path, auxiliary_model, include_selected,
) -> None:
    _table(tmp_path, "rollouts", task_id=["a"], reward=[1.0], prompt_tokens=[100],
           completion_tokens=[10], n_tool_calls=[1])
    identity = {"run_id": "r", "task_id": "a", "repeat": 0, "agent_ref": "agent",
                "lane": "model_call"}
    usage = ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens")
    requests = [{**identity, "request_idx": 1, "event_idx": 1, "model_name": auxiliary_model,
                 **dict.fromkeys(usage, 900)}]
    if include_selected:
        requests.append({**identity, "request_idx": 0, "event_idx": 0,
                         "model_name": "selected-model", **dict.fromkeys(usage)})
    pl.DataFrame(requests).write_parquet(tmp_path / "tables" / "events.parquet")
    payload = collect_evalrdl_efficiency(_run(tmp_path), pricing={"model_id": "selected-model"})
    metrics = _metrics(payload)
    assert not {
        "cached_tokens", "reasoning_tokens", "prompt_tokens", "final_context_tokens",
    } & metrics.keys()
    assert metrics["input_tokens"]["overall"]["total"] == 100
    assert metrics["output_tokens"]["overall"]["total"] == 10
    assert metrics["simulated_cost_usd"]["overall"]["total"] == pytest.approx(0.00002)


def test_completion_semantics_are_classified_per_rollout(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["answer-only", "inclusive"], reward=[1.0, 0.0],
           prompt_tokens=[100, 100], completion_tokens=[2_895, 5_601])
    _table(tmp_path, "events", task_id=["answer-only", "inclusive"] * 2,
           lane=["trace", "trace", "model_call", "model_call"], event_idx=[0] * 4,
           request_idx=[0] * 4, model_name=[None, None, "selected-model", "selected-model"],
           reasoning_tokens=[2_706, 2_706, None, None],
           completion_tokens=[None, None, 5_601, 5_601])
    payload = collect_evalrdl_efficiency(_run(tmp_path), pricing=SELECTED)
    metrics = _metrics(payload)
    assert metrics["answer_tokens"]["overall"]["total"] == 5_790
    assert metrics["output_tokens"]["overall"]["total"] == 11_202
    assert metrics["total_tokens"]["overall"]["total"] == 11_402
    assert "1 answer-only, 1 reasoning-inclusive" in (
        _provenance(payload)["answer_tokens"]["detail"]
    )


def test_context_endpoints_use_only_one_parent_lane(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a", "b"], reward=[1.0, 0.0], prompt_tokens=[130, 70],
           completion_tokens=[5, 5], n_assistant_turns=[3, 2], n_tool_calls=[1, 1])
    _table(tmp_path, "events", task_id=["a", "a", "a", "b", "b"],
           lane=["trace", "trace", "subagent/worker", "", "trace"],
           event_idx=[0, 1, 0, 0, 0], request_idx=[0, 1, 0, 0, 0],
           prompt_tokens=[10, 20, 100, 30, 40])
    metrics = _metrics(collect_evalrdl_efficiency(_run(tmp_path)))
    # a uses its trace parent and excludes the child; b has two possible parent lanes.
    assert (metrics["prompt_tokens"]["overall"]["n"],
            metrics["prompt_tokens"]["overall"]["total"]) == (1, 10)
    assert (metrics["final_context_tokens"]["overall"]["n"],
            metrics["final_context_tokens"]["overall"]["total"]) == (1, 20)


def test_context_uses_single_ordered_selected_model_lane_without_request_ids(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a"], reward=[1.0], prompt_tokens=[30],
           completion_tokens=[3], n_tool_calls=[1])
    _table(tmp_path, "events", task_id=["a"] * 4, lane=["", "model_call", "model_call",
                                                        "model_call"],
           event_idx=[0, 0, 1, 2], request_idx=[0, None, None, None],
           model_name=[None, "selected-model", "selected-model", "policy-model"],
           prompt_tokens=[680_763, 10, 20, 999])
    metrics = _metrics(collect_evalrdl_efficiency(_run(tmp_path), pricing=SELECTED))
    assert metrics["prompt_tokens"]["overall"]["total"] == 10
    assert metrics["final_context_tokens"]["overall"]["total"] == 20


def test_agent_rollups_and_detailed_tools_use_evalrdl_accounts(tmp_path) -> None:
    _table(tmp_path, "rollouts", task_id=["a"], reward=[1.0], prompt_tokens=[100],
           completion_tokens=[10], n_assistant_turns=[7], n_tool_calls=[2])
    lanes = ["trace", "session", "session"]
    _table(tmp_path, "tool_calls", task_id=["a"] * 3, lane=lanes,
           tool_name=["terminal", "terminal", "python"], args_chars=[10, 10, 20])
    _table(tmp_path, "parts", task_id=["a"] * 3, lane=lanes, channel=["tool_result"] * 3,
           chars=[100, 100, 200])
    payload = collect_evalrdl_efficiency(_run(tmp_path))
    metrics = _metrics(payload)
    assert metrics["n_turns"]["overall"]["total"] == 7
    assert metrics["n_tool_calls"]["overall"]["total"] == 2
    assert metrics["avg_tool_calls_per_turn"]["overall"]["mean"] == 2 / 7
    assert metrics["tool_arg_chars"]["overall"]["total"] == 30
    assert metrics["tool_output_chars"]["overall"]["total"] == 300
    assert {row["name"]: row for row in payload["tool_types"]} == {
        "python": {"name": "python", "overall": 1, "pass": 1, "fail": 0},
        "terminal": {"name": "terminal", "overall": 1, "pass": 1, "fail": 0},
    }


# --- The efficiency gate before downloads and analysis ---

def _remote(run_id: str, n_rollouts: int | None = 3) -> dict:
    return {"kind": "remote", "base_url": "https://runs.example.test", "run_id": run_id,
            "selection": "all", "n_rollouts": n_rollouts}


def _no_download(monkeypatch) -> list[str]:
    downloads: list[str] = []
    monkeypatch.setattr(
        cohort_module, "_prepare_remote",
        lambda api, source, *args, **kwargs: downloads.append(source["run_id"]) or {},
    )
    monkeypatch.setattr(cohort_module, "save_cohort", lambda cohort: None)
    return downloads


@pytest.mark.parametrize("sources,reason", [
    ([_remote("first"), _remote("second")], "exactly one evaluation run"),
    ([_remote("empty", n_rollouts=0)], "no rollouts"),
], ids=["two_runs", "empty_run"])
def test_efficiency_only_selection_stops_before_download(monkeypatch, sources, reason) -> None:
    downloads = _no_download(monkeypatch)
    selection = Cohort(runs_where="", where="", sources=sources, rows=[])
    with efficiency_gate(EfficiencyGate()), pytest.raises(
        UnsupportedEfficiencySelection, match=reason,
    ):
        prepare_cohort(selection)
    assert downloads == []


@pytest.mark.parametrize("combined", [False, True], ids=["no_efficiency", "combined_run"])
def test_multi_run_selection_downloads_for_failure_analysis(monkeypatch, combined) -> None:
    downloads = _no_download(monkeypatch)
    reasons: list[str] = []
    gate = EfficiencyGate(
        combined=True, continue_without=lambda reason: reasons.append(reason) or True,
    ) if combined else None
    selection = Cohort(
        runs_where="", where="", sources=[_remote("first"), _remote("second")], rows=[],
    )
    with efficiency_gate(gate):
        prepare_cohort(selection)
    assert downloads == ["first", "second"]
    if combined:
        assert len(reasons) == 1 and "exactly one evaluation run" in reasons[0]
        assert not gate.enabled


def test_declined_combined_run_stops_and_explains_how_to_continue(tmp_path) -> None:
    gate = EfficiencyGate(combined=True, continue_without=lambda reason: False)
    sources = [EvalSource(kind="jsonl", path=tmp_path / f"{name}.jsonl") for name in "ab"]
    with efficiency_gate(gate), pytest.raises(
        UnsupportedEfficiencySelection, match="Turn off efficiency metrics",
    ):
        check_local_sources(sources)


def test_plain_parquet_is_rejected_before_import(tmp_path) -> None:
    parquet = tmp_path / "results.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(parquet)
    with efficiency_gate(EfficiencyGate()), pytest.raises(
        UnsupportedEfficiencySelection, match="not an EvalRDL run",
    ):
        check_local_sources([EvalSource(kind="parquet", path=parquet)])


@pytest.mark.parametrize("answer,expected", [(True, ["agent"]), (False, None)],
                         ids=["continue", "decline"])
def test_new_combined_run_on_unsupported_data_asks_before_analysis(
    monkeypatch, answer, expected,
) -> None:
    settings = Config()
    settings.analysis_outputs = ["efficiency", "agent"]
    questions: list[str] = []
    monkeypatch.setattr(cli, "_discovery_config", lambda args: settings)
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda **kwargs: True)
    monkeypatch.setattr(cli, "_run_cli_dataset_wizard", lambda *args, **kwargs: DatasetSelection(
        parquet_paths=[], model_id="model", benchmarks=["bench"],
    ))
    monkeypatch.setattr(
        cli, "select_yes_no", lambda question, **kwargs: questions.append(question) or answer,
    )
    args = SimpleNamespace(once=True, eval_limit=None, keep_going=False, hours=None)
    if expected is None:
        with pytest.raises(UnsupportedEfficiencySelection, match="EvalRDL"):
            cli._prepare_new_config(args)
    else:
        assert cli._prepare_new_config(args).settings.analysis_outputs == expected
    assert questions == ["Continue with failure discovery only?"]


def _prepared(monkeypatch, readiness: EfficiencyReadiness, **gate) -> tuple[EfficiencyGate, list]:
    notes: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "autofa.data.efficiency_support.verify_efficiency_data", lambda selection: readiness,
    )
    gate = EfficiencyGate(notify=lambda message, warning: notes.append((message, warning)), **gate)
    return gate, notes


@pytest.mark.parametrize("choice,enabled", [("both", True), ("without", False)])
def test_combined_run_without_comparison_offers_failure_analysis_only(
    monkeypatch, choice, enabled,
) -> None:
    offered = []
    gate, notes = _prepared(
        monkeypatch, EfficiencyReadiness(40, 40, 0, 40),
        combined=True, choose=lambda choices: offered.append(choices) or choice,
    )
    with efficiency_gate(gate):
        check_prepared(object())
    assert offered == [("both", "without", "stop")]
    assert gate.enabled is enabled
    message, warning = notes[-1]
    assert "no incorrect rollouts" in message and warning


def test_stopping_on_missing_comparison_raises_the_reason(monkeypatch) -> None:
    gate, _notes = _prepared(
        monkeypatch, EfficiencyReadiness(40, 0, 0, 40), choose=lambda choices: "stop",
    )
    with efficiency_gate(gate), pytest.raises(UnsupportedEfficiencySelection, match="reward"):
        check_prepared(object())


def test_noninteractive_run_continues_with_partial_content_warnings(monkeypatch) -> None:
    gate, notes = _prepared(monkeypatch, EfficiencyReadiness(100, 0, 0, 60))
    with efficiency_gate(gate):
        check_prepared(object())
    warnings = [message for message, warning in notes if warning]
    assert gate.enabled
    assert any("60 of 100 rollouts" in message for message in warnings)
    assert any("cannot compare correct and incorrect" in message for message in warnings)


def test_small_groups_warn_below_the_summary_minimum() -> None:
    [warning] = content_warnings(EfficiencyReadiness(110, 100, 10, 110))
    assert "at least 20 correct and 20 incorrect" in warning
    assert content_warnings(EfficiencyReadiness(40, 20, 20, 40)) == []
