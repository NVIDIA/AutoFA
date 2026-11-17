# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact frontier metrics, ranking and CLI progress that separates search from validation."""

from __future__ import annotations

from io import StringIO
from itertools import combinations
from math import prod
from random import Random

import pytest
from rich.console import Console

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.mcp import tools_eval
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics.live_progress import display_progress_snapshot
from autofa.metrics.pareto import (
    frontier_value,
    hypervolume,
    library_frontier_metrics,
    library_frontier_points,
    pareto_front,
    point_from_scores,
)
from autofa.metrics.progress import display_progress_rows, render_terminal_progress

from support.coverage import candidate
from support.evaluation import _FakeEvalDataset

PANEL = ["a1", "a2", "b1", "p1"]
TWO_AXES = Config(objective_axes=["ppv_at_pi0", "recall"], objective_weights=[1, 1])


def test_frontier_value_mean_of_front():
    points = [(0.8, 0.2), (0.2, 0.8), (0.1, 0.1)]  # third dominated
    front = pareto_front(points)
    assert len(front) == 2
    # Geometric means: sqrt(0.8*0.2)=0.4 for both frontier points → overall 0.4
    assert frontier_value(points, config=TWO_AXES) == pytest.approx(0.4)
    assert frontier_value([], config=TWO_AXES) == 0.0


@pytest.mark.parametrize("points", [[], [(0.8, 0.0), (0.0, 0.8)]])
def test_empty_or_zero_volume(points):
    assert hypervolume(points) == 0.0


def _inclusion_exclusion_volume(points, ref):
    """Independent oracle: alternate intersection volumes over all box subsets."""
    return sum(
        (-1) ** (size + 1) * prod(
            max(0.0, min(point[axis] for point in subset) - ref[axis])
            for axis in range(len(ref))
        )
        for size in range(1, len(points) + 1)
        for subset in combinations(points, size)
    )


# Random fronts include duplicates and dominated points.
@pytest.mark.parametrize("dimensions", [1, 2, 4])
def test_matches_independent_box_union(dimensions):
    rng = Random(42)
    ref = (0.0,) * dimensions
    for _ in range(20):
        points = [
            tuple(rng.randrange(11) / 10 for _ in range(dimensions))
            for _ in range(6)
        ]
        expected = _inclusion_exclusion_volume(points, ref)
        assert hypervolume(points) == pytest.approx(expected, abs=1e-12)




def test_library_frontier_points_skips_unscored():
    fns = [
        FeatureFunction(name="scored", description="d", source="def scored(item): return False",
                        scores={"ppv_at_pi0": 0.8, "recall": 0.4, "behavior_purity": 0.9,
                                "evaluated": True}),
        FeatureFunction(name="new", description="d", source="def new(item): return False",
                        scores={}),
    ]
    assert library_frontier_points(fns, axes=["ppv_at_pi0", "recall"]) == [(0.8, 0.4)]


@pytest.mark.parametrize("incomplete", [
    {"errors": 1}, {"assessment_coverage": 0.89},
])
def test_library_frontier_excludes_incomplete_evidence(incomplete):
    fn = FeatureFunction(name="partial", description="d", source="def partial(item): return True",
                         scores={"evaluated": True, "ppv_at_pi0": 1,
                                 "recall": 1, **incomplete})
    assert library_frontier_points([fn]) == []


def _two_item_runtime(source, run_id):
    fn = FeatureFunction(name="candidate", description="d", source=source)
    rt = Runtime(
        dataset=_FakeEvalDataset([
            {"_item_id": "a", "is_correct": False}, {"_item_id": "b", "is_correct": True},
        ]),
        library=FunctionLibrary(model_id="m", functions=[fn]),
        run=RunConfig(run_id=run_id, parquet_paths=[], model_id="m"), config=Config(),
    )
    set_runtime(rt)
    return rt, fn


@pytest.mark.parametrize("limit, broken, complete", [(None, False, True), (1, False, False),
                                                  (None, True, False)])
def test_evaluation_records_completeness_for_stopping(
    tmp_path, monkeypatch, limit, broken, complete,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    rt, fn = _two_item_runtime(
        "def candidate(item, llm_client=None):\n    "
        + ("raise ValueError('cannot assess')\n" if broken else "return True\n"),
        "complete",
    )
    assert tools_eval._evaluate_streaming([fn], chunk_size=2, max_items=limit)["ok"]
    assert rt.scores_history[-1]["evaluation_complete"] is complete


def test_unavailable_task_inventory_marks_breadth_unavailable_without_failing(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))

    def missing_inventory(rt):
        raise OSError("source index missing")

    monkeypatch.setattr(
        "autofa.session.family_coverage.cached_task_inventory", missing_inventory,
    )
    rt, fn = _two_item_runtime(
        "def candidate(item, llm_client=None):\n    return True\n", "no-inventory",
    )
    assert tools_eval._evaluate_streaming([fn], chunk_size=2)["ok"]
    scores = rt.library.get("candidate").scores
    assert scores["task_recall"] is None
    assert scores["task_recall_scope"] == "unavailable"
    assert "source index missing" in scores["task_recall_reason"]
    assert scores["search_panel"]["available"] is False


def test_partial_eval_frontier_uses_full_library(tmp_path, monkeypatch):
    """Regression: submit/partial eval must not set frontier from only the new fn."""
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    items = [{
        "model_id": "m", "benchmark": "b", "problem_id": str(i), "run_idx": 0,
        "is_correct": i % 2 == 0, "response_text": "TODO leak" if i % 5 == 0 else "clean",
        "_item_id": f"id-{i}",
    } for i in range(20)]
    keeper_scores = {"ppv_at_pi0": 0.9, "recall": 0.5, "behavior_purity": 0.8, "n_items": 20,
                     "evaluated": True}
    lib = FunctionLibrary(model_id="m", benchmark="b", functions=[
        FeatureFunction(name="keeper", description="already strong",
                        source="def keeper(item, llm_client=None):\n    return False\n",
                        scores=dict(keeper_scores)),
        FeatureFunction(name="new_fn", description="new candidate", source=(
            "def new_fn(item, llm_client=None):\n"
            "    return 'TODO' in (item.get('response_text') or '')\n")),
    ])
    run = RunConfig(run_id="partial-frontier", parquet_paths=[], model_id="m", benchmarks=["b"])
    rt = Runtime(
        dataset=_FakeEvalDataset(items), library=lib, run=run, scores_history=[],
        config=Config(objective_axes=["ppv_at_pi0", "recall"], objective_weights=[1, 1]),
    )
    set_runtime(rt)

    out = tools_eval.evaluate_functions(names=["new_fn"], limit=20)
    assert out.get("ok"), out
    assert [result["name"] for result in out["results"]] == ["new_fn"]
    snap = rt.scores_history[-1]
    assert snap["evaluated_names"] == ["new_fn"]
    assert snap["n_scored"] == 2
    assert len(snap["points"]) == 2
    # Full-library frontier, not the single new detector's point alone.
    expected = library_frontier_metrics(rt.library.functions, config=rt.config)
    assert snap["frontier_value"] == pytest.approx(expected["frontier_value"])
    alone = frontier_value([point_from_scores(
        rt.library.get("new_fn").scores or {}, rt.config.objective_axes,
    )], config=rt.config)
    assert alone < snap["frontier_value"]  # keeper lifts the library frontier


def test_terminal_logs_share_live_status_and_comparable_trend():
    cfg = Config(behavior_strictness="causal")
    # Legacy logs without display telemetry never invent a zero score.
    legacy = dict(display_progress_rows([{"frontier_value": 0, "points": []}]))
    assert legacy["best search score"].startswith("unavailable")
    assert legacy["validated frontier"].startswith("unavailable")
    first = display_progress_snapshot([candidate(panel_ids=PANEL)], cfg, run_id="r1")
    fn = candidate(reviewed=("yes", "yes"), matched=("a1", "b1"), panel_ids=PANEL)
    current = display_progress_snapshot([fn], cfg, run_id="r1")
    history = [{"display_progress": first}, {"display_progress": first}]
    rows = display_progress_rows(history, current=current)
    search = dict(rows)["best search score"]
    assert "1.000" in search and "↑+" in search and "provisional" in search
    assert dict(rows)["validated frontier"].startswith("pending")
    output = StringIO()
    render_terminal_progress(
        history, Console(file=output, width=180, no_color=True), current=current,
    )
    rendered = output.getvalue()
    for label, value in rows:
        assert label in rendered and value in rendered
