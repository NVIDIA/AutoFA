# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stratified LLM panels, local censuses, bounded concurrency and round sample rotation."""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config, RunConfig
from autofa.core.dependencies import build_bundle
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import DetectorAbstained, NullLLM
from autofa.data import parquet_store
from autofa.data.parquet_store import Dataset
from autofa.llm.hub import HubLLM
from autofa.mcp import tools_data, tools_eval, tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics import claude_judge
from autofa.metrics.association import StreamingBucketAccumulator
from autofa.metrics.sampling import (
    SCORING_SAMPLE_VERSION,
    StratifiedReservoir,
    wilson_interval,
)
from autofa.metrics.search_association import search_association
from autofa.metrics.search_panel import current_search_panel
from autofa.metrics.verdicts import resolved_behavior_purity, resolved_behavior_specificity
from autofa.session.family_discovery import STATE_FILE, initialize_family_discovery

from support.evaluation import (
    _LLM_SOURCE,
    _REGEX_SOURCE,
    _corpus,
    _FakeEvalDataset,
    _llm_flag,
    _record_calls,
    _runtime,
    _scoring_runtime,
)
from support.families import complete_round_batch

# Pure estimators.


def test_wilson_interval_brackets_and_bounds():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    # Never leaves the unit interval, even at the degenerate edges.
    for k, n in ((0, 10), (10, 10), (0, 1), (1, 1)):
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0
    # More data ⇒ tighter interval around the same proportion.
    narrow = wilson_interval(500, 1000)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
    # No data ⇒ maximally uninformative rather than a fake point estimate.
    assert wilson_interval(0, 0) == (0.0, 1.0)


# Stratified allocation.


def _fill(reservoir: StratifiedReservoir, key: str, n: int, prefix: str) -> None:
    for i in range(n):
        reservoir.observe(f"{prefix}-{i}", key=key)


def test_allocation_keeps_rare_strata_and_weights_sum_to_population():
    res = StratifiedReservoir(per_stratum_cap=500, seed=1)
    _fill(res, "big|fail", 900, "big")
    _fill(res, "rare|fail", 12, "rare")
    sample = res.allocate(target_n=100, min_per_stratum=5)

    assert sample.frame_n == 912
    assert set(sample.strata) == {"big|fail", "rare|fail"}
    # The rare stratum survives proportional allocation (~1 row) via the floor.
    assert sample.strata["rare|fail"]["sampled"] >= 5
    # Weights reconstruct the population, which is what makes estimates unbiased.
    assert sum(sample.weights.values()) == pytest.approx(912.0, rel=1e-6)
    assert sample.sample_n == len(sample.item_ids) == len(sample.weights)
    assert not sample.is_census()


def test_reweight_rescales_to_rows_actually_scored_and_zeroes_vanished_strata():
    """Losing rows after allocation must not silently shrink a stratum."""
    res = StratifiedReservoir(per_stratum_cap=100, seed=5)
    _fill(res, "a|fail", 100, "a")
    _fill(res, "b|fail", 100, "b")
    sample = res.allocate(target_n=20, min_per_stratum=5)
    assert sum(sample.weights.values()) == pytest.approx(200.0)

    # Half the "a" rows turn out to be unreachable on read.
    survivors = [i for i in sample.item_ids if sample.stratum_of[i] != "a|fail"]
    a_ids = [i for i in sample.item_ids if sample.stratum_of[i] == "a|fail"]
    survivors += a_ids[: len(a_ids) // 2]
    sample.reweight(survivors)

    assert sample.sample_n == len(survivors)
    # Population total is preserved, so the estimate stays unbiased.
    assert sum(sample.weights.values()) == pytest.approx(200.0)
    assert sample.strata["a|fail"]["sampled"] == len(a_ids) // 2

    # A stratum with no surviving rows contributes nothing rather than a stale weight.
    sample.reweight([i for i in survivors if sample.stratum_of[i] == "b|fail"])
    assert sample.strata["a|fail"]["sampled"] == sample.strata["a|fail"]["weight"] == 0.0
    assert sum(sample.weights.values()) == pytest.approx(100.0)


def test_duplicate_item_ids_are_not_double_counted():
    res = StratifiedReservoir(per_stratum_cap=50, seed=5)
    for _ in range(4):
        _fill(res, "dup|fail", 10, "d")  # same 10 ids, four times
    sample = res.allocate(target_n=100, min_per_stratum=1)
    # A repeated id is scored once, so it must appear once in the sample.
    assert len(sample.item_ids) == len(set(sample.item_ids)) == 10


def test_reservoir_is_uniform_within_a_stratum():
    """Reservoir must not degenerate to the first N ids seen."""
    res = StratifiedReservoir(per_stratum_cap=20, seed=7)
    _fill(res, "b|fail", 400, "x")
    sample = res.allocate(target_n=20, min_per_stratum=1)
    picked = [int(i.split("-")[1]) for i in sample.item_ids]
    assert len(picked) == 20
    assert max(picked) > 200, "reservoir only retained early rows"


def test_balanced_panel_preserves_benchmark_strata_and_weighted_population_rates():
    res = StratifiedReservoir(per_stratum_cap=200, seed=7)
    populations = {"large|fail": 900, "small|fail": 100,
                   "large|pass": 100, "small|pass": 1900}
    for key, n in populations.items():
        _fill(res, key, n, key)
    sample = res.allocate(target_n=200)
    assert sample.sample_n == 200
    assert {key: meta["sampled"] for key, meta in sample.strata.items()} == {
        "large|fail": 90, "small|fail": 10, "large|pass": 5, "small|pass": 95,
    }
    assert not sample.unrepresented_strata
    assert sum(sample.weights.values()) == pytest.approx(3000)
    acc = StreamingBucketAccumulator()
    for item_id in sample.item_ids:
        key = sample.stratum_of[item_id]
        acc.update(key.startswith("large|"), is_correct=key.endswith("|pass"),
                   item_id=item_id, weight=sample.weight_for(item_id))
    positive = next(row for row in acc.finalize() if row["bucket"] == "True")
    assert positive["recall"] == pytest.approx(.9)
    assert positive["precision"] == pytest.approx(.9)
    assert positive["pct_caught"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("failed,passed,target,expected", [
    (562, 3151, 200, (100, 100)),
    (12, 400, 200, (12, 188)),
    (400, 12, 200, (188, 12)),
    (12, 18, 200, (12, 18)),
    (0, 400, 200, (0, 200)),
    (200, 200, 1, (1, 0)),
], ids=["balanced", "fills-fail-shortage", "fills-pass-shortage", "census", "no-failures",
        "budget-of-one"])
def test_balanced_panel_fills_shortages_within_hard_budget(failed, passed, target, expected):
    res = StratifiedReservoir(per_stratum_cap=target, seed=2)
    _fill(res, "b|fail", failed, "f")
    _fill(res, "b|pass", passed, "p")
    sample = res.allocate(target_n=target)
    counts = tuple(int(sample.strata.get("b|" + outcome, {}).get("sampled", 0))
                   for outcome in ("fail", "pass"))
    assert counts == expected
    assert sample.sample_n == sum(expected) <= target
    if not sample.unrepresented_strata:
        assert sum(sample.weights.values()) == pytest.approx(failed + passed)
    assert sample.is_census() is (sample.sample_n == failed + passed)


def test_balanced_panel_reserves_each_benchmark_when_budget_allows():
    res = StratifiedReservoir(per_stratum_cap=20, seed=2)
    for i in range(8):
        _fill(res, f"bench{i}|fail", 20, f"f{i}")
    _fill(res, "bench0|pass", 400, "p")
    sample = res.allocate(target_n=10)
    assert sample.sample_n == 10
    assert not sample.unrepresented_strata
    assert sample.strata["bench0|pass"]["sampled"] == 2


# Weighted accumulation.


def test_weighted_accumulator_recovers_population_rates():
    """A 10% sample with weight 10 must reproduce the full-corpus recall."""
    exact = StreamingBucketAccumulator()
    for i in range(1000):
        is_fail = i % 2 == 0
        fires = is_fail and i % 4 == 0
        exact.update(fires, is_correct=not is_fail, item_id=f"e{i}")
    exact_row = next(b for b in exact.finalize() if b["bucket"] == "True")

    sampled = StreamingBucketAccumulator()
    for i in range(0, 1000, 10):
        is_fail = i % 2 == 0
        fires = is_fail and i % 4 == 0
        sampled.update(fires, is_correct=not is_fail, item_id=f"s{i}", weight=10.0)
    rows = sampled.finalize()
    row = next(b for b in rows if b["bucket"] == "True")

    assert row["recall"] == pytest.approx(exact_row["recall"], abs=0.02)
    assert row["sampled"] is True
    # Rows actually scored into this bucket, not the population they represent.
    assert row["n_rows_scored"] == 50
    assert row["n_fail"] == pytest.approx(500.0)
    assert row["recall_ci_low"] <= row["recall"] <= row["recall_ci_high"]
    assert row["raw_ppv_ci_low"] <= row["raw_ppv"] <= row["raw_ppv_ci_high"]
    # Interval reflects the 100 scored rows, not the 1000 they represent.
    assert row["n_eff_fail"] < 200


def test_unresolved_bounds_preserve_unequal_weights_and_parent_subset():
    overall, within = StreamingBucketAccumulator(), StreamingBucketAccumulator()
    # Weight, outcome, prediction/error and parent membership are independent.
    cases = [
        (2, False, True, None, True),
        (3, False, False, None, False),
        (5, True, False, None, True),
        (7, False, None, RuntimeError("Invalid label"), True),
        (11, True, None, RuntimeError("Refused"), False),
        (13, False, None, DetectorAbstained("Missing evidence"), False),
        (17, True, None, DetectorAbstained("Missing ending"), True),
    ]
    for index, (weight, correct, prediction, error, parent_fires) in enumerate(cases):
        item_id = str(index)
        for acc in (overall, within) if parent_fires else (overall,):
            if isinstance(error, DetectorAbstained):
                acc.note_abstention(is_correct=correct, weight=weight)
            elif error is not None:
                acc.note_error(is_correct=correct, weight=weight, error=str(error),
                               item_id=item_id)
            else:
                acc.update(prediction, is_correct=correct, item_id=item_id, weight=weight)

    global_scores = next(row for row in overall.finalize() if row["bucket"] == "True")
    parent_scores = next(row for row in within.finalize() if row["bucket"] == "True")
    assert overall.n_items == 3 and within.n_items == 2
    assert overall.errors == overall.n_abstained == 2
    assert within.errors == within.n_abstained == 1
    assert global_scores["recall"] == pytest.approx(2 / 5)
    assert parent_scores["recall"] == 1
    assert overall.unresolved_bounds(global_scores)["recall"] == pytest.approx([2 / 25, 22 / 25])
    assert overall.unresolved_bounds(global_scores)["fpr"] == pytest.approx([0, 28 / 33])
    assert within.unresolved_bounds(parent_scores)["recall"] == pytest.approx([2 / 9, 1])
    assert within.unresolved_bounds(parent_scores)["fpr"] == pytest.approx([0, 17 / 22])
    assert overall.abstention_bounds(global_scores)["recall"] == pytest.approx([2 / 18, 15 / 18])
    assert {row["bucket"] for row in overall.finalize()} == {"True", "False"}


# Detector classification.


# kind defaults to "regex"; an undeclared LLM detector would otherwise be scored on every
# row, one paid call each. Direct chat calls also take the checkpointed LLM path.
@pytest.mark.parametrize("source,kind,expected", [
    ("def r(item, llm_client=None):\n    return True\n", "regex", False),
    ("def d(i):\n    return 1\n", "llm", True),
    ("def u(item, llm_client=None):\n    return llm_client.classify('bad?', 'x') == 'yes'\n",
     "regex", True),
    ("def detect(item, llm_client=None):\n"
     "    return llm_client.chat.completions.create(model='model', messages=[])\n",
     "regex", True),
], ids=["unused-client", "declared", "undeclared-classify", "undeclared-chat"])
def test_is_llm_detector_sniffs_source_not_just_kind(source, kind, expected):
    fn = FeatureFunction(name="detector", description="", source=source, kind=kind)
    assert tools_eval.is_llm_detector(fn) is expected


# End-to-end evaluation.


def test_capped_panel_stays_fixed_across_rounds_and_full_census_clears_it(tmp_path, monkeypatch):
    fn = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[fn], n=120)
    calls = _record_calls(monkeypatch)
    assert tools_eval.evaluate_functions()["ok"]
    panel = set(fn.scores["evaluation_item_ids"])
    assert len(panel) == len(calls) == 20
    rt.run.round += 1
    rt.run.exploration_seed += 321
    assert tools_eval.evaluate_functions()["ok"]
    assert set(fn.scores["evaluation_item_ids"]) == panel
    assert len(calls) == 20
    fn.source += "\n# revised implementation\n"
    assert tools_eval.evaluate_functions()["ok"]
    assert set(fn.scores["evaluation_item_ids"]) == panel
    assert len(calls) == 40
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    assert "evaluation_item_ids" not in fn.scores
    assert fn.scores["n_scored"] == 120
    assert fn.scores["evaluation_scope_complete"] is True
    assert len(calls) == 140


def test_python_census_and_sampled_llm_share_search_panel_without_extra_execution(
    tmp_path, monkeypatch,
):
    python = FeatureFunction(name="plain", description="text match", source=_REGEX_SOURCE)
    llm = FeatureFunction(name="flag", description="semantic match", source=_LLM_SOURCE, kind="llm")
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[python, llm], n=120)
    # Explicitly test a corpus larger than the default iteration's scan frame.
    rt.run.eval_limit = 60
    calls = _record_calls(monkeypatch, lambda src, _: "llm" if src == _LLM_SOURCE else "python")
    assert tools_eval.evaluate_functions(names=[python.name], mode="full")["ok"]
    assert tools_eval.evaluate_functions(names=[llm.name])["ok"]
    py_panel, llm_panel = (current_search_panel(fn.scores) for fn in (python, llm))
    assert py_panel is not None and llm_panel is not None
    assert py_panel["cohort_fingerprint"] == llm_panel["cohort_fingerprint"]
    assert py_panel["task_recall"] == llm_panel["task_recall"]
    assert py_panel["n_failed_tasks"] == llm_panel["n_failed_tasks"] == 10
    assert python.scores["task_recall"] == .5
    assert llm.scores["task_recall"] < python.scores["task_recall"]
    assert python.scores["n_scored"] == 120
    assert llm.scores["n_scored"] == 20
    assert Counter(calls) == {"python": 120, "llm": 20}
    panel_before = dict(llm_panel)
    assert tools_eval.evaluate_functions(names=[llm.name], mode="full")["ok"]
    assert current_search_panel(llm.scores) == panel_before
    assert llm.scores["task_recall"] == .5
    # Full scoring uses cached panel predictions; panel projection adds no calls.
    assert Counter(calls) == {"python": 120, "llm": 120}


def test_full_scan_panel_uses_processed_rows_not_prefetched_parent_rows(tmp_path, monkeypatch):
    parent = FeatureFunction(name="flag", description="parent", source=_LLM_SOURCE, kind="llm")
    child = FeatureFunction(name="plain", description="child", source=_REGEX_SOURCE)
    rt = _runtime(tmp_path, monkeypatch, sample=8, workers=4, functions=[parent, child], n=40)
    rt.run.eval_limit = 12  # Below the parent worker's 16-row prefetch window.
    assert tools_eval.evaluate_functions(names=[child.name], within=parent.name)["ok"]
    capped = current_search_panel(child.scores)
    assert capped is not None
    assert tools_eval.evaluate_functions(names=[child.name], within=parent.name, mode="full")["ok"]
    assert current_search_panel(child.scores) == capped


def test_evaluation_seed_uses_saved_main_seed_across_rounds_and_continued_run_ids():
    # A falsy saved main seed is still the saved main seed.
    run = SimpleNamespace(
        run_id="first", exploration_seed=900, exploration_seed_fixed=False,
        exploration_base_seed=0, exploration_sequence_index=1,
    )
    assert tools_eval._evaluation_seed(run) == 0
    run.run_id = "continued"
    run.exploration_seed = 321
    run.exploration_sequence_index += 1
    assert tools_eval._evaluation_seed(run) == 0


def test_fresh_runs_reuse_default_scoring_panel_across_run_ids_and_workers(tmp_path, monkeypatch):
    panels, run_ids = [], []
    for workers, seed in ((1, 42), (3, 42), (2, 718)):
        fn = _llm_flag()
        rt = _runtime(tmp_path, monkeypatch, sample=20, workers=workers, functions=[fn], n=120)
        if seed != 42:
            rt.config.exploration_seed = seed
        rt.run = RunConfig.create([], "m", [], rt.config)
        run_ids.append(rt.run.run_id)
        assert rt.run.exploration_seed == seed
        assert rt.run.exploration_seed_fixed is True
        assert tools_eval.evaluate_functions()["ok"]
        panels.append(fn.scores["evaluation_item_ids"])
    assert len(set(run_ids)) == 3
    assert len(panels[0]) == 20
    assert panels[0] == panels[1]
    assert panels[0] != panels[2]


def test_scoring_panel_survives_round_rotation_and_continue(tmp_path, monkeypatch):
    fn = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[fn], n=120)
    rt.run = RunConfig.create([], "m", [], rt.config)
    monkeypatch.setattr(tools_library, "_detector_exploration", lambda: {"message": "Explore"})
    monkeypatch.setattr(tools_library, "emit_runtime_status", lambda *args, **kwargs: None)
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions",
                        lambda **kwargs: {"ok": True, "unresolved": []})
    monkeypatch.setattr(
        "autofa.session.discovery_target.failure_mode_target_for_runtime",
        lambda rt: {"min_failure_modes": rt.run.min_failure_modes},
    )
    calls = _record_calls(monkeypatch)
    assert tools_eval.evaluate_functions()["ok"]
    panel = fn.scores["evaluation_item_ids"]
    search_panel = fn.scores["search_panel"].copy()
    assert len(panel) == len(calls) == 20
    main_seed = rt.run.exploration_base_seed
    discovery_seeds = [rt.run.exploration_seed]
    audit_panel = None
    for _ in range(2):
        complete_round_batch(rt)
        previous_round, previous_seed = rt.run.round, rt.run.exploration_seed
        pending = tools_library.advance_round()
        assert pending["ok"] and pending["refinement_pending"] and not pending["advanced"]
        assert (rt.run.round, rt.run.exploration_seed) == (previous_round, previous_seed)
        controls = pending["round_validation"]["detectors"][0]["regression_checks"]
        assert controls["available"]
        assert set(controls["item_ids"]) <= set(fn.scores["evaluation_item_ids"])
        assert fn.scores["search_panel"] == search_panel
        spent_at_checkpoint = len(calls)
        assert tools_library.advance_round()["refinement_pending"]
        closed = tools_library.advance_round(
            validation_response="Retain the checked revision; the saved controls do not "
                                "establish a supported implementation change.",
        )
        assert closed["ok"] and closed["advanced"]
        assert rt.run.round == previous_round + 1
        assert len(calls) == spent_at_checkpoint  # No paid repeat for an unchanged closing pass.
        discovery_seeds.append(rt.run.exploration_seed)
        assert tools_eval.evaluate_functions()["ok"]
        assert fn.scores["search_panel"] == search_panel
        assert fn.scores["round_validation"]["complete"] is True
        if audit_panel is None:
            audit_panel = fn.scores["evaluation_item_ids"]
        assert fn.scores["evaluation_item_ids"] == audit_panel
    assert len(set(discovery_seeds)) == 3
    assert len(audit_panel) >= len(panel)
    spent = len(calls)
    assert spent == len(set(panel) | set(audit_panel))

    seed_path = rt.persist_library()
    previous = rt.run
    continued = RunConfig.create([], "m", [], rt.config, seed_library_path=str(seed_path))
    assert continued.run_id != previous.run_id
    assert continued.exploration_base_seed == main_seed
    assert continued.exploration_sequence_index == previous.exploration_sequence_index + 1
    assert continued.exploration_seed not in discovery_seeds
    rt = Runtime(dataset=rt.dataset, library=FunctionLibrary.load(seed_path),
                 run=continued, config=rt.config, scores_history=[])
    set_runtime(rt)
    assert tools_eval.evaluate_functions()["ok"]
    assert rt.library.get("flag").scores["evaluation_item_ids"] == panel
    assert len(calls) == spent + 20  # New run cache, identical discovery scoring panel.


@pytest.mark.parametrize("invalid_evaluation", ["missing", "stale", "other_run", "lost_panel"])
def test_development_judge_requires_current_llm_evaluation_before_scanning(
    tmp_path, monkeypatch, invalid_evaluation,
):
    fn = _llm_flag()
    _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[fn], n=120)
    if invalid_evaluation != "missing":
        assert tools_eval.evaluate_functions()["ok"]
        if invalid_evaluation == "stale":
            fn.source += "\n# revised detector\n"
        elif invalid_evaluation == "other_run":
            fn.scores["evaluation_run_id"] = "older-run"
        else:
            fn.scores.pop("evaluation_item_ids")

    def unexpected_scan(*args, **kwargs):
        pytest.fail("Development review must not silently buy a corpus scan")

    monkeypatch.setattr(tools_eval, "_iter_scored_items", unexpected_scan)
    result = tools_eval.judge_function(fn.name)
    assert not result["ok"], result
    assert result["evaluation_required"] is True


def test_development_judge_does_not_expand_previously_evaluated_prefix(tmp_path, monkeypatch):
    fn = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=0, workers=2, functions=[fn], n=120)
    rt.run.eval_limit = 40
    assert tools_eval.evaluate_functions()["ok"]
    assert "evaluation_item_ids" not in fn.scores
    assert fn.scores["n_scored"] == 40
    rt.run.eval_limit = 120

    def unexpected_detector_call(*args, **kwargs):
        pytest.fail("Review cannot expand an earlier evaluation after the run limit changes")

    monkeypatch.setattr(tools_eval, "run_detector", unexpected_detector_call)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", lambda **kwargs: {
        "behavior_purity": 1.0, "n_judged": len(kwargs["items"]), "verdicts": [],
    })
    result = tools_eval.judge_function(fn.name)
    assert result["ok"], result
    assert result["n_scanned"] == result["scan_limit"] == 40


def test_cancelled_evaluation_does_not_attach_late_scores(tmp_path, monkeypatch):
    from autofa.util.cancellation import Cancellation, cancellation_scope

    fn = _llm_flag(scores={"previous_score": 0.75})
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[fn])
    cancellation = Cancellation()

    def interrupted_breadth(*args, **kwargs):
        cancellation.cancel()
        return {"task_recall": 0.5}

    monkeypatch.setattr("autofa.metrics.task_recall.build_task_recall", interrupted_breadth)
    with cancellation_scope(cancellation), pytest.raises(asyncio.CancelledError):
        tools_eval.evaluate_functions()
    assert fn.scores == {"previous_score": 0.75}
    assert fn.failure_localisations == {}
    assert rt.scores_history == []


# The smaller panel keeps the three-review floor; the larger one scales past it.
@pytest.mark.parametrize("panel_size", [40, 160])
def test_family_review_reuses_nonprefix_panel_then_confirms_on_untouched_tasks(
    tmp_path, monkeypatch, panel_size,
):
    from autofa.util.atomic import atomic_write_json

    fn = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=panel_size, workers=2, functions=[fn], n=240)
    rt.config.behavior_judge_harness = "claude"
    state = initialize_family_discovery(rt)
    calls, reviewed = _record_calls(monkeypatch), []

    def judge(**kwargs):
        reviewed.append({item["_item_id"] for item in kwargs["items"]})
        return {"behavior_purity": 1.0, "behavior_specificity": .8,
                "n_judged": len(kwargs["items"]), "verdicts": []}

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    assert tools_eval.evaluate_functions()["ok"]
    panel = set(fn.scores["evaluation_item_ids"])
    assert len(panel) == len(calls) == panel_size
    panel_fires = sum("BAD" in item["response_text"] for item in rt.dataset._fake_items
                      if item["_item_id"] in panel)
    expected_reviews = max(3, (panel_fires + 9) // 10)

    development = tools_eval.judge_function(fn.name)
    assert development["ok"], development
    assert development["sample"]["scan_scope"] == "evaluated_llm_panel"
    assert development["sample"]["n_scanned"] == panel_size
    assert development["sample"]["actual_n"] == len(reviewed[0]) == expected_reviews
    assert "development_cap" not in development["sample"]
    assert (expected_reviews > 3) is (panel_size == 160)
    assert development["sample"]["limited"] is True
    assert reviewed[0] <= panel
    # This adapter has no point-read index. Reloading selected evidence using
    # n_scanned as a physical prefix would lose these nonprefix panel members.
    physical_prefix = {item["_item_id"] for item in rt.dataset._fake_items[:panel_size]}
    assert reviewed[0] - physical_prefix
    assert len(calls) == panel_size

    # Family workflow tests cover completing checked surveys. Isolate the
    # transition from bounded development feedback to final confirmation here.
    state["phase"] = "specialize"
    atomic_write_json(rt.run.run_dir() / STATE_FILE, state)
    confirmation = tools_eval.judge_function(fn.name, review_phase="confirmation")
    assert confirmation["ok"], confirmation
    assert confirmation["sample"]["scan_scope"] == "capped_corpus"
    assert confirmation["sample"]["n_scanned"] == 240
    assert confirmation["sample"]["limited"] is False
    assert confirmation["sample"]["independent_task_n"] == len(reviewed[1])
    assert reviewed[1] - panel
    assert reviewed[0].isdisjoint(reviewed[1])
    assert len(calls) == len(set(calls)) == 240


def test_non_firing_detector_has_zero_headline_recall(tmp_path, monkeypatch):
    fn = FeatureFunction(name="never", description="never fires",
                         source="def never(item, llm_client=None):\n    return False\n")
    rt = _runtime(tmp_path, monkeypatch, sample=40, workers=4, functions=[fn], n=120)
    result = tools_eval.evaluate_functions(names=[fn.name], mode="full")
    assert result["ok"]
    scores = rt.library.get(fn.name).scores
    assert scores["n_scored"] == 120
    assert scores["a"] == scores["b"] == scores["n_flagged"] == 0
    assert scores["c"] == scores["d"] == 60
    assert scores["recall"] == scores["ppv_at_pi0"] == scores["pct_caught"] == 0
    assert result["results"][0]["verdict"] == scores["verdict"]
    assert result["results"][0]["verdict"]["code"] == "EMPTY"
    assert rt.library.get(fn.name).example_fail_ids == []


@pytest.mark.parametrize("mode,workers", [("full", 1), ("capped", 4)])
def test_invalid_llm_answers_remain_unresolved_in_full_and_sampled_evaluation(
    tmp_path, monkeypatch, mode, workers,
):
    source = '''
def detect(item, llm_client=None):
    text = item['response_text']
    if text == 'missing':
        library.abstain('Missing causal evidence')
    return llm_client.classify_options(text, 'Does the behavior occur?') == 'yes'
'''
    fn = FeatureFunction(name="semantic", description="Semantic behavior", source=source)
    parent = FeatureFunction(name="parent", description="Parent population",
                             source="def detect(item): return True")
    rt = _runtime(tmp_path, monkeypatch, sample=12, workers=workers, functions=[fn, parent])
    # Separate strata guarantee both outcome classes contain errors and abstentions,
    # including when each sampled row represents five corpus rows.
    cases = [("yes", False), ("no", True), ("malformed", False),
             ("malformed", True), ("missing", False), ("missing", True)]
    rt.dataset = _FakeEvalDataset([
        {"_item_id": f"{group}-{index}", "benchmark": str(group), "is_correct": correct,
         "response_text": text}
        for group, (text, correct) in enumerate(cases) for index in range(10)
    ])
    rt.config.llm_eval_min_per_stratum = 1

    def complete(**kwargs):
        request = kwargs["messages"][0]["content"]
        if request.startswith("Locate passages"):
            answer = '{"quotes": []}'
        else:
            evidence = request.rsplit("--- text ---\n", 1)[1]
            answer = ("This is not one of the requested labels"
                      if evidence == "malformed" else evidence)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content=answer))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    llm = HubLLM(client, api="chat_completions")
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: llm)

    result = tools_eval.evaluate_functions(names=[fn.name], mode=mode, within=parent.name)

    assert result["ok"], result
    scores = rt.library.get(fn.name).scores
    attempted = 60 if mode == "full" else 12
    assert scores["errors"] == scores["n_insufficient"] == scores["n_scored"] == attempted // 3
    assert scores["n_unresolved"] == attempted * 2 // 3
    assert scores["n_items"] == 20  # weighted resolved population, never the failed calls
    assert scores["assessment_coverage"] == pytest.approx(1 / 3)
    assert scores["metrics_scope"] == "resolved_items"
    assert scores["evaluation_complete"] is False
    assert scores["a"] == scores["d"] == 10 and scores["b"] == scores["c"] == 0
    assert scores["recall"] == 1 and scores["fpr"] == 0
    assert scores["unresolved_bounds"]["recall"] == pytest.approx([1 / 3, 1])
    assert scores["unresolved_bounds"]["fpr"] == pytest.approx([0, 2 / 3])
    assert scores["abstention_bounds"]["recall"] == [0.5, 1]
    assert scores["abstention_bounds"]["fpr"] == [0, 0.5]
    assert scores["within_errors"] == scores["errors"]
    assert scores["within_n_insufficient"] == scores["n_insufficient"]
    assert scores["within_n_unresolved"] == scores["n_unresolved"]
    assert scores["within_assessment_coverage"] == scores["assessment_coverage"]
    assert scores["within_unresolved_bounds"] == scores["unresolved_bounds"]
    assert scores["judge_prefilter"]["status"] == "inconclusive"
    feedback = result["results"][0]["scores"]
    for key in ("metrics_scope", "assessment_coverage", "n_unresolved", "unresolved_bounds",
                "within_unresolved_bounds", "evaluation_complete"):
        assert feedback[key] == scores[key]
    assert all("exactly one" in e["error"] for e in scores["error_examples"])
    assert {row["bucket"] for row in scores["buckets"]} == {"True", "False"}
    assert all(not item_id.startswith(("2-", "3-", "4-", "5-"))
               for row in scores["buckets"]
               for item_id in row["example_fail_ids"] + row["example_pass_ids"])


def test_serial_detector_errors_are_not_negative_predictions(tmp_path, monkeypatch):
    fn = FeatureFunction(name="serial", description="Deterministic behavior", source='''
def detect(item):
    if item['response_text'] == 'broken':
        raise ValueError('Unsupported evidence format')
    return True
''')
    rt = _runtime(tmp_path, monkeypatch, sample=0, workers=1, functions=[fn])
    rt.dataset = _FakeEvalDataset([
        {"_item_id": "resolved", "response_text": "good", "is_correct": False},
        {"_item_id": "error", "response_text": "broken", "is_correct": False},
    ])
    result = tools_eval.evaluate_functions(names=[fn.name], mode="full")
    assert result["ok"], result
    scores = rt.library.get(fn.name).scores
    assert scores["errors"] == scores["n_unresolved"] == scores["n_scored"] == 1
    assert scores["n_insufficient"] == 0
    assert scores["recall"] == 1 and scores["c"] == 0
    assert scores["assessment_coverage"] == 0.5
    assert scores["unresolved_bounds"]["recall"] == [0.5, 1]
    assert "abstention_bounds" not in scores
    assert {row["bucket"] for row in scores["buckets"]} == {"True"}


def test_entirely_refused_llm_evaluation_has_no_scored_items(tmp_path, monkeypatch):
    fn = FeatureFunction(name="unavailable", description="Semantic behavior", source='''
def detect(item, llm_client=None):
    return llm_client.classify_options(item['response_text'], 'Apply the rubric') == 'yes'
''')
    rt = _runtime(tmp_path, monkeypatch, sample=0, workers=4, functions=[fn])
    rt.dataset = _FakeEvalDataset([
        {"_item_id": "fail", "response_text": "first", "is_correct": False},
        {"_item_id": "pass", "response_text": "second", "is_correct": True},
    ])
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
        message=SimpleNamespace(content=None, refusal="Cannot classify"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: response)))
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: HubLLM(client))

    result = tools_eval.evaluate_functions(names=[fn.name], mode="full")

    assert result["ok"], result
    scores = rt.library.get(fn.name).scores
    assert scores["n_insufficient"] == scores["n_unresolved"] == 2
    assert scores["errors"] == scores["n_scored"] == scores["n_items"] == 0
    assert result["results"][0]["verdict"] == scores["verdict"]
    assert result["results"][0]["verdict"]["code"] == "UNSCORED"
    assert scores["buckets"] == []
    assert scores["assessment_coverage"] == 0
    assert scores["unresolved_bounds"] == {"recall": [0, 1], "fpr": [0, 1]}
    assert scores["abstention_bounds"] == scores["unresolved_bounds"]
    assert scores["evaluation_complete"] is False
    assert scores["judge_prefilter"]["status"] == "inconclusive"


@pytest.mark.parametrize("llm_parent", [False, True])
def test_missing_backend_does_not_score_llm_detectors_as_negative(
    tmp_path, monkeypatch, llm_parent,
):
    detector = _llm_flag()
    plain = FeatureFunction(name="plain", description="text match", source=_REGEX_SOURCE)
    rt = _runtime(tmp_path, monkeypatch, sample=120, workers=4, functions=[detector, plain])
    monkeypatch.setattr(tools_eval, "_llm_adapter", NullLLM)
    detector.scores = {"previous_score": 0.75}

    result = tools_eval.evaluate_functions(
        names=[plain.name] if llm_parent else [detector.name],
        within=detector.name if llm_parent else None,
        mode="full",
    )

    assert result["ok"] is False
    assert result["code"] == "llm_backend_unavailable"
    assert detector.scores == {"previous_score": 0.75}
    assert rt.scores_history == []


@pytest.mark.parametrize("tool", ["judge", "smoke"])
def test_missing_backend_blocks_direct_judge_and_smoke(tmp_path, monkeypatch, tool):
    detector = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=120, workers=4, functions=[detector])
    monkeypatch.setattr(tools_eval, "_llm_adapter", NullLLM)
    result = (tools_eval.judge_function(detector.name) if tool == "judge"
              else tools_eval.smoke_test_feature(detector.source))
    assert result["ok"] is False
    assert result["code"] == "llm_backend_unavailable"
    assert detector.scores == {}
    assert not (rt.run.run_dir() / "detector_results.sqlite3").exists()


def test_missing_backend_cannot_write_detector_checkpoints(tmp_path, monkeypatch):
    detector = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=120, workers=4, functions=[detector])
    with pytest.raises(RuntimeError, match="checkpoints were not read or written"):
        tools_eval._score_items_concurrently(
            [(detector, _corpus(1)[0])], workers=1, llm=NullLLM(),
        )
    assert not (rt.run.run_dir() / "detector_results.sqlite3").exists()


def test_artifact_change_invalidates_cached_negative_detector_result(tmp_path, monkeypatch):
    import autofa.data.artifact_access as access

    source = (
        "from autofa.trace import artifact\n"
        "def flag(item, llm_client=None):\n"
        "    return artifact(item, 0)['segments'][0]['text'] == 'changed'\n"
    )
    fn = FeatureFunction(name="flag", description="artifact evidence", source=source, kind="llm")
    _runtime(tmp_path, monkeypatch, sample=0, workers=1, functions=[fn])
    evidence = {"status": "available", "sha256": "original",
                "segments": [{"locator": "line:1", "text": "original"}]}
    monkeypatch.setattr(access, "artifact_evidence",
                        lambda item, idx: access._record(evidence, idx))
    invocations = _record_calls(monkeypatch)
    pairs = [(fn, _corpus(1)[0])]
    assert tools_eval._score_items_concurrently(pairs, workers=1, llm=None)[0][2:] == (False, None)
    assert tools_eval._score_items_concurrently(pairs, workers=1, llm=None)[0][2:] == (False, None)
    assert len(invocations) == 1
    evidence.update(sha256="changed", segments=[{"locator": "line:1", "text": "changed"}])
    assert tools_eval._score_items_concurrently(pairs, workers=1, llm=None)[0][2:] == (True, None)
    assert len(invocations) == 2


@pytest.fixture
def image_detector_runtime(tmp_path, monkeypatch):
    """Exercise the nested request shape used by retained CharXiv images."""
    import base64
    import io
    import re

    from autofa.data.artifact_store import ArtifactStore

    image = pytest.importorskip("PIL.Image")

    def make_item(index, *, color="red", prefix=False):
        with io.BytesIO() as output:
            image.new("RGB", (8, 6), color).save(output, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        text = f"{'yes' if index % 2 else 'no'} chart-{index}"
        messages = [{"role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": url},
        ]}]
        if prefix:
            messages.insert(0, {"role": "system", "content": "A changed recorded request"})
        return {**_corpus(index + 1)[index], "benchmark": "charxiv",
                "response_text": json.dumps({
                    "agent_request": {"messages": json.dumps(messages)},
                    "steps": [{"role": "assistant", "content": "The blue bar is highest."}],
                })}

    class LocalLLM:
        def __init__(self):
            self.calls = []
            self.errors = set()
            self.localisations = 0

        def classify(self, prompt, text, *, images):
            assert len(images) == 1 and images[0]["status"] == "available"
            assert images[0]["data_url"].startswith("data:image/png;base64,")
            marker = re.search(r"(?:yes|no) chart-\d+", text).group()
            self.calls.append(marker)
            if marker in self.errors:
                raise ValueError("Malformed classifier response")
            return "yes" if marker.startswith("yes") else "no"

        def localise(self, *args, **kwargs):
            self.localisations += 1
            return '{"quotes":[]}'

    source = (
        "from autofa.trace import artifact_image\n"
        "def image_flag(item, llm_client):\n"
        "    image = artifact_image(item, 0)\n"
        "    return llm_client.classify('Read the chart', item['input_text'], "
        "images=[image]) == 'yes'\n"
    )
    fn = FeatureFunction(name="image_flag", description="Chart evidence", source=source,
                         kind="llm")
    rt = _runtime(tmp_path, monkeypatch, sample=8, workers=2, functions=[fn])
    rt.dataset = _FakeEvalDataset([make_item(i) for i in range(20)])
    llm = LocalLLM()
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: llm)
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "autofa"))
    monkeypatch.setattr(ArtifactStore, "_provider", lambda self: pytest.fail(
        "Retained image cache checks must never use a remote artifact provider"))
    return rt, fn, llm, make_item


def test_retained_image_predictions_and_judge_panel_are_reused(image_detector_runtime, monkeypatch):
    rt, fn, llm, _ = image_detector_runtime
    rt.config.behavior_judge_harness = "claude"
    initialize_family_discovery(rt)
    reviewed = []

    def judge(**kwargs):
        reviewed.append({item["_item_id"] for item in kwargs["items"]})
        return {"behavior_purity": 1.0, "behavior_specificity": 1.0,
                "n_judged": len(kwargs["items"]), "verdicts": []}

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    assert tools_eval.evaluate_functions()["ok"]
    panel = set(fn.scores["evaluation_item_ids"])
    calls = list(llm.calls)
    localisations = llm.localisations
    assert len(calls) == len(panel) == 8
    assert {text.split()[0] for text in calls} == {"yes", "no"}
    # Optional quote generation is deferred to report preparation. Artifact
    # read receipts still accompany every prediction and remain reusable.
    assert localisations == 0
    receipts = {item_id: row["artifact_reads"] for item_id, row
                in fn.failure_localisations.items()}
    assert len(receipts) == len(panel)
    assert tools_eval.evaluate_functions()["ok"]
    assert llm.calls == calls
    assert llm.localisations == localisations
    assert receipts == {item_id: row["artifact_reads"] for item_id, row
                        in fn.failure_localisations.items()}

    result = tools_eval.judge_function(fn.name)
    assert result["ok"], result
    assert result["sample"]["scan_scope"] == "evaluated_llm_panel"
    assert reviewed and reviewed[0] <= panel
    assert llm.calls == calls
    assert llm.localisations == localisations


@pytest.mark.parametrize("index,change", [(0, "bytes"), (1, "source")])
def test_retained_image_cache_revalidates_source_and_bytes(image_detector_runtime, index, change):
    from autofa.core.fields import get_field_policy, wrap_item_ban_sentinels
    from autofa.data.artifact_access import validate_artifact_reads
    from autofa.data.trace_fields import derive_trace_fields

    rt, fn, llm, make_item = image_detector_runtime
    original = make_item(index)
    localisations = {}
    first = tools_eval._score_items_concurrently(
        [(fn, original)], workers=1, llm=llm, localisations=localisations,
    )
    assert first[0][2:] == (bool(index), None)
    receipts = localisations[fn.name][original["_item_id"]]["artifact_reads"]
    assert tools_eval._score_items_concurrently(
        [(fn, original)], workers=1, llm=llm,
    )[0][2:] == (bool(index), None)
    assert len(llm.calls) == 1

    altered = make_item(index, **({"color": "blue"} if change == "bytes"
                                 else {"prefix": True}))
    payload = wrap_item_ban_sentinels(derive_trace_fields(altered), get_field_policy(rt.config))
    assert not validate_artifact_reads(payload, receipts)
    assert tools_eval._score_items_concurrently(
        [(fn, altered)], workers=1, llm=llm,
    )[0][2:] == (bool(index), None)
    assert len(llm.calls) == 2


def test_image_scoring_retry_only_reruns_failed_calls(image_detector_runtime):
    rt, fn, llm, _ = image_detector_runtime
    rt.config.llm_eval_sample = 0
    llm.errors = {"yes chart-1"}
    assert tools_eval.evaluate_functions()["ok"]
    assert fn.scores["errors"] == 1
    assert len(llm.calls) == 20
    llm.errors.clear()
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    assert fn.scores["errors"] == 0
    assert len(llm.calls) == 21
    assert llm.calls[-1] == "yes chart-1"


def test_no_evidence_does_not_become_a_heuristic_behavior_judgment(tmp_path, monkeypatch):
    fn = FeatureFunction(name="never", description="never fires",
                         source="def never(item, llm_client=None):\n    return False\n")
    _runtime(tmp_path, monkeypatch, sample=0, workers=1, functions=[fn], n=120)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    result = tools_eval.judge_function(fn.name)
    assert result["ok"]
    assert result["n_judged"] == 0
    assert result["behavior_purity"] is result["behavior_specificity"] is None
    assert resolved_behavior_purity(fn.scores) is None
    assert resolved_behavior_specificity(fn.scores) is None
    legacy = {"judge_backend": "heuristic", "behavior_purity": 0.7,
              "behavior_specificity": 0.8}
    assert resolved_behavior_purity(legacy) is None
    assert resolved_behavior_specificity(legacy) is None


@pytest.mark.parametrize("kind", ["llm", "regex"])
def test_judge_worker_pool_preserves_evidence_sample_and_bounds_concurrency(
    tmp_path, monkeypatch, kind,
):
    # Both runs must execute the fixture calls to measure worker concurrency.
    monkeypatch.setattr("autofa.metrics.detector_checkpoints.DetectorCheckpoints.get",
                        lambda self, key: None)

    fn = FeatureFunction(name="plain", description="test", source=_REGEX_SOURCE, kind=kind)
    rt = _runtime(tmp_path, monkeypatch, sample=40, workers=1, functions=[fn], n=120)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    active = peak = 0
    lock = threading.Lock()
    captured = []

    def detect(source, item, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.001 * (1 + int(item["problem_id"]) % 3))
            if int(item["problem_id"]) % 29 == 0:
                raise ValueError("fixture detector error")
            return item["response_text"] == "BAD"
        finally:
            with lock:
                active -= 1

    def judge(**kwargs):
        captured.append([item["_item_id"] for item in kwargs["items"]])
        return {"behavior_purity": 1.0, "behavior_specificity": 0.8,
                "n_judged": len(kwargs["items"]), "verdicts": []}

    monkeypatch.setattr(tools_eval, "run_detector", detect)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    if kind == "llm":
        assert tools_eval.evaluate_functions()["ok"]
        peak = 0
    serial = tools_eval.judge_function(fn.name)
    assert peak == 1
    peak = 0
    rt.config.llm_eval_workers = 4
    parallel = tools_eval.judge_function(fn.name)
    assert parallel["n_fires"] == serial["n_fires"]
    assert parallel["n_sampled"] == serial["n_sampled"]
    assert captured[0] == captured[1]
    assert 1 < peak <= 4 if kind == "llm" else peak == 1


def test_unfinished_judge_clears_previous_scores_and_exposes_uncertainty(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    from autofa.metrics.judge_evidence import EVIDENCE_VERSION

    fn = FeatureFunction(name="plain", description="test", source=_REGEX_SOURCE)
    rt = _runtime(tmp_path, monkeypatch, sample=0, workers=1, functions=[fn], n=120)
    rt.config.behavior_judge_harness = "claude"
    fn.scores.update(
        behavior_purity=1.0, behavior_specificity=0.9,
        claim_calibration={"status": "overclaim"}, calibration_hint="old hint",
    )
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    reviewed_items = []

    async def session(**kwargs):
        reviewed_items.append(kwargs["review"].snapshot["item_id"])
        raise RuntimeError("review budget exhausted")

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", session)
    result = tools_eval.judge_function(fn.name)
    assert result["ok"] and result["n_judged"] > 0
    assert len(reviewed_items) == result["n_judged"]
    assert result["n_completed"] == 0 and result["n_errors"] == result["n_judged"]
    assert result["n_uncertain"] == 0
    assert result["behavior_purity"] is result["behavior_specificity"] is None
    assert fn.scores["claim_calibration"] is fn.scores["calibration_hint"] is None
    assert fn.scores["behavior_n_uncertain"] == result["n_uncertain"]
    assert fn.scores["behavior_n_errors"] == result["n_errors"]
    assert fn.scores["judge_evidence_version"] == EVIDENCE_VERSION
    assert resolved_behavior_purity(fn.scores) is None
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert fn.scores["behavior_n_completed"] == 0
    assert fn.scores["behavior_n_uncertain"] == result["n_uncertain"]
    assert fn.scores["behavior_n_errors"] == result["n_errors"]
    assert resolved_behavior_purity(fn.scores) is None
    assert resolved_behavior_specificity(fn.scores) is None


def test_judge_claim_provenance_and_processing_errors_survive_evaluation(tmp_path, monkeypatch):
    from autofa.metrics.claim_contract import (
        JUDGE_CONTRACT_VERSION,
        build_claim_contract,
        claim_hash,
    )

    fn = FeatureFunction(name="plain", description="test", source=_REGEX_SOURCE)
    rt = _runtime(tmp_path, monkeypatch, sample=0, workers=1, functions=[fn], n=120)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    fingerprint = claim_hash(source=fn.source, description="test", mechanism="test")
    contract = build_claim_contract(
        source=fn.source, description="test", mechanism="test",
        defining_clauses=[{"id": "clause_1", "source_field": "description",
                           "quote": "test", "required_review_scope": "local",
                                            "predicate": "The stated test condition holds."}],
    )
    assert contract["claim_hash"] == fingerprint
    assert contract["contract_id"] != fingerprint
    compilation = {
        "reused": True, "error": None, "directory": "/original/compilation",
        "policy_hash": "compiler-policy", "model": "compiler-model",
        "exclusions": [{"source_field": "mechanism", "quote": "test", "reason": "Example"}],
        "sessions": [{"stage": "compilation", "session_id": "compiler-session",
                      "usage": {"input_tokens": 42}, "cost_usd": 0.012}],
    }
    reviews = [{
        "item_id": "a", "verdict": "uncertain", "claim_assessments": [],
        "verification_status": "uncertain", "primary_review_path": "/primary",
        "verification_review_path": "/primary/confirmation",
        "initial_review": {"item_id": "a", "verdict": "yes", "claim_assessments": []},
    }]

    def judge(**kwargs):
        count = len(kwargs["items"])
        return {
            "backend": "claude_agent_sdk", "behavior_purity": (count - 2) / count,
            "behavior_specificity": 0.8, "n_judged": count, "n_completed": count - 2,
            "n_yes": count - 2, "n_no": 0, "n_uncertain": 1, "n_errors": 1,
            "judge_contract_version": JUDGE_CONTRACT_VERSION, "claim_hash": fingerprint,
            "judge_model": "test-model", "judge_rubric_hash": "rubric-fingerprint",
            "judge_effort": "medium", "judge_thinking": "adaptive",
            "judge_policy_hash": claude_judge.judge_policy_hash(),
            "claim_contract": contract, "verdicts": reviews,
            "claim_compilation": compilation,
        }

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    result = tools_eval.judge_function(fn.name)
    assert result["n_errors"] == result["n_uncertain"] == 1
    assert result["claim_hash"] == fingerprint
    assert result["claim_contract"] == contract
    assert result["claim_compilation"] == compilation
    # A later evaluation keeps the judge provenance, in memory and on disk.
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert fn.scores["behavior_n_errors"] == fn.scores["behavior_n_uncertain"] == 1
    expected = {
        "behavior_verdicts": reviews, "claim_hash": fingerprint,
        "judge_contract_version": JUDGE_CONTRACT_VERSION, "claim_contract": contract,
        "claim_compilation": compilation, "judge_model": "test-model", "judge_effort": "medium",
        "judge_thinking": "adaptive", "judge_rubric_hash": "rubric-fingerprint",
        "judge_policy_hash": claude_judge.judge_policy_hash(),
    }
    saved = FunctionLibrary.load(rt.working_library_path()).get(fn.name)
    for key, value in expected.items():
        assert fn.scores[key] == saved.scores[key] == value, key


def test_batch_judge_failure_preserves_processing_error_counts(tmp_path, monkeypatch):
    from autofa.metrics.claim_contract import claim_hash
    from autofa.metrics.verdicts import behavior_validation

    fn = FeatureFunction(name="plain", description="test", source=_REGEX_SOURCE)
    _runtime(tmp_path, monkeypatch, sample=0, workers=1, functions=[fn], n=120)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)

    def judge(**kwargs):
        raise RuntimeError("snapshot preparation failed")

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    result = tools_eval.judge_function(fn.name)
    assert result["behavior_purity"] is result["behavior_specificity"] is None
    assert result["n_errors"] == result["n_judged"] == result["n_sampled"] > 0
    assert result["n_uncertain"] == result["n_completed"] == 0
    assert all(row["verdict"] == "error" for row in result["verdicts"])
    assert behavior_validation(fn.scores)["status"] == "error"
    assert result["claim_hash"] == claim_hash(
        source=fn.source, description=fn.description, mechanism=fn.description,
    )


def test_llm_detector_is_scored_on_a_subsample_and_marked(tmp_path, monkeypatch):
    fn = FeatureFunction(name="flag", description="llm", source=_LLM_SOURCE, kind="llm")
    rt = _runtime(tmp_path, monkeypatch, sample=40, workers=4, functions=[fn], n=120)

    out = tools_eval.evaluate_functions(names=["flag"], limit=120)
    assert out["ok"], out
    meta = out["llm_sample"]
    assert meta["sampled"] is True
    assert meta["sample_n"] == 40  # hard budget, including per-stratum floors
    assert meta["frame_n"] == 120

    scores = rt.library.get("flag").scores
    assert scores["sampled"] is True
    assert scores["sample_frame_n"] == 120
    assert scores["n_scored"] == meta["sample_n"]
    assert scores["sample_requested_n"] == meta["requested_n"] == meta["sample_n"]
    assert scores["task_support"]["sampling_design"]["sample_n"] == meta["sample_n"]
    assert scores["recall_ci_low"] <= scores["recall"] <= scores["recall_ci_high"]
    assert scores["feedback"]["sampled"] is True
    assert out["results"][0]["sampled"] is True

    # mode=full ignores llm_eval_sample and scores every row (end-of-loop path).
    full = tools_eval.evaluate_functions(names=["flag"], mode="full")
    assert full["ok"], full
    assert full["llm_sample"] is None
    scores = rt.library.get("flag").scores
    assert scores["sampled"] is False
    assert scores["n_scored"] == scores["n_items"] == 120
    assert scores["eval_mode"] == "full"


def test_sampled_zero_fire_retains_uncertainty_without_paid_rescoring(tmp_path, monkeypatch):
    fn = FeatureFunction(name="flag", description="Never matches", kind="llm",
                         source="def flag(item, llm_client=None): return False")
    _runtime(tmp_path, monkeypatch, sample=40, workers=4, functions=[fn], n=120)
    result = tools_eval.evaluate_functions(names=[fn.name], limit=120)
    assert result["ok"]
    scores = fn.scores
    assert scores["sampled"] and scores["recall"] == 0
    assert scores["n_items"] == pytest.approx(sum(scores[k] for k in ("a", "b", "c", "d")))
    assert scores["n_items"] == pytest.approx(120)
    assert scores["n_scored"] == result["llm_sample"]["sample_n"] < scores["n_items"]

    assert scores["recall_ci_low"] == 0
    assert scores["recall_ci_high"] > .01
    assert scores["judge_prefilter"]["status"] == "inconclusive"

    def forbidden(*args, **kwargs):
        pytest.fail("An empty cached panel must not trigger new paid detector calls")

    monkeypatch.setattr(tools_eval, "run_detector", forbidden)
    judged = tools_eval.judge_function(fn.name)
    assert judged["ok"] and judged["n_judged"] == 0
    assert fn.scores.get("behavior_purity") is None
    assert fn.scores["judge_prefilter"]["criteria"] == []


def test_llm_parent_gate_is_concurrent_and_preserves_within_metrics(tmp_path, monkeypatch):
    # Exercise scheduling on both passes; durable reuse has its own recovery tests.
    monkeypatch.setattr("autofa.metrics.detector_checkpoints.DetectorCheckpoints.get",
                        lambda self, key: None)

    parent = FeatureFunction(name="parent", description="", source=_LLM_SOURCE, kind="llm")
    child = FeatureFunction(name="child", description="", source=_REGEX_SOURCE)
    rt = _runtime(tmp_path, monkeypatch, sample=0, workers=1, functions=[parent, child], n=40)
    active = peak = 0
    lock = threading.Lock()
    original = tools_eval.run_detector

    def tracking(source, item, **kwargs):
        nonlocal active, peak
        if source != parent.source:
            return original(source, item, **kwargs)
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.003)
            if item["_item_id"] == "id-9":
                raise ValueError("parent error must not open the gate")
            return original(source, item, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(tools_eval, "run_detector", tracking)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    assert tools_eval.evaluate_functions(names=["child"], within="parent", mode="full")["ok"]
    serial = dict(child.scores)
    assert peak == 1
    peak = 0
    rt.config.llm_eval_workers = 4
    assert tools_eval.evaluate_functions(names=["child"], within="parent", mode="full")["ok"]
    assert 1 < peak <= 4
    assert child.scores == serial
    assert child.scores["within_n"] == 9


def test_llm_scoring_is_concurrent_but_accumulates_deterministically(tmp_path, monkeypatch):
    """Detector calls overlap; identical seeds still produce identical scores."""
    seen_threads: set[int] = set()
    barrier = threading.Barrier(4, timeout=5)
    original = tools_eval.run_detector

    def _tracking_run_detector(source, item, **kwargs):
        seen_threads.add(threading.get_ident())
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return original(source, item, **kwargs)

    fn = FeatureFunction(name="flag", description="", source=_LLM_SOURCE, kind="llm")
    rt = _runtime(tmp_path, monkeypatch, sample=40, workers=4, functions=[fn], n=120)
    monkeypatch.setattr(tools_eval, "run_detector", _tracking_run_detector)

    out = tools_eval.evaluate_functions(names=["flag"], limit=120)
    assert out["ok"], out
    assert len(seen_threads) > 1, "LLM scoring did not run on a worker pool"

    first = dict(rt.library.get("flag").scores)
    monkeypatch.setattr(tools_eval, "run_detector", original)
    rt2 = _runtime(tmp_path, monkeypatch, sample=40, workers=1, functions=[
        FeatureFunction(name="flag", description="", source=_LLM_SOURCE, kind="llm")
    ], n=120)
    second = tools_eval.evaluate_functions(names=["flag"], limit=120)
    assert second["ok"], second
    serial = rt2.library.get("flag").scores
    # Same seed and same sample ⇒ concurrency must not change the numbers.
    assert first["recall"] == pytest.approx(serial["recall"])
    assert first["sample_n"] == serial["sample_n"]


def test_unreachable_rows_do_not_bias_the_estimate(tmp_path, monkeypatch):
    """Rows lost between allocation and read are reweighted, not silently dropped."""
    fn = FeatureFunction(name="flag", description="", source=_LLM_SOURCE, kind="llm")
    rt = _runtime(tmp_path, monkeypatch, sample=40, workers=4, functions=[fn], n=120)

    original = tools_eval._fetch_items_by_id

    def _lossy(dataset, ids, *, max_items):
        # Loss uncorrelated with the detector's signal, as a dataset filter or an
        # index miss would be — that is the regime reweighting corrects.
        rng = random.Random(11)
        for item in original(dataset, ids, max_items=max_items):
            if rng.random() < 0.33:
                continue
            yield item

    monkeypatch.setattr(tools_eval, "_fetch_items_by_id", _lossy)
    out = tools_eval.evaluate_functions(names=["flag"], limit=120)
    assert out["ok"], out
    meta = out["llm_sample"]
    # sample_n reflects what was scored, not what was requested.
    assert meta["sample_n"] < meta["requested_n"]
    scores = rt.library.get("flag").scores
    assert scores["n_scored"] == meta["sample_n"]
    assert scores["sample_requested_n"] == meta["requested_n"]
    assert "sampling_design" not in scores["task_support"]
    assert scores["estimated_n_confirmed_failed_tasks"] is None
    # True corpus recall is 0.5 (every other failure carries BAD).
    assert scores["recall_ci_low"] <= 0.5 <= scores["recall_ci_high"]
    # Weights still reconstruct the frame, so counts are not deflated by the loss.
    assert scores["n_items"] == pytest.approx(120.0, rel=0.02)


def test_explainer_shows_the_within_parent_interval():
    """A CI must bracket the recall displayed next to it, not a different one."""
    from autofa.metrics.verdicts import format_interval

    plain = {"recall": 0.2, "recall_ci_low": 0.15, "recall_ci_high": 0.25}
    assert format_interval(plain, "recall") == "[0.15,0.25]"
    # Displayed recall is within_recall=0.80, so the corpus CI [0.15,0.25] is wrong.
    corpus_only = {**plain, "within_parent": "parent", "within_recall": 0.80}
    assert format_interval(corpus_only, "recall") == ""
    with_within = {**corpus_only, "within_recall_ci_low": 0.72, "within_recall_ci_high": 0.87}
    assert format_interval(with_within, "recall") == "[0.72,0.87]"


def test_build_data_rounds_weighted_floats_and_prefers_sample_frame(tmp_path):
    from autofa.reports.failure_cards import build_data

    fn = FeatureFunction(
        name="est", description="Often fails mid-debug without a verified fix",
        source="def est(item, llm_client=None):\n    return False\n", kind="llm",
        scores={
            "a": 63.63913043478261, "c": 223.36086956521632, "n_items": 499.9999999999985,
            "sample_frame_n": 500, "sampled": True, "recall": 0.22, "ppv_at_pi0": 0.76,
            "precision": 0.91, "pct_caught": 0.14, "bucket": "True",
            "buckets": [{"bucket": "True", "n": 70, "n_fail": 64, "n_pass": 6, "purity": 0.91,
                         "example_fail_ids": [], "example_pass_ids": []}],
        },
    )
    lib = FunctionLibrary(model_id="m", benchmark="b", functions=[fn])
    data = build_data(lib, dataset=None)
    assert data["meta"]["n_items"] == 500
    assert data["meta"]["n_failures"] == 287


@pytest.mark.parametrize("mode", ["full", "capped"])
def test_scoring_concurrency_respects_resident_byte_budget(tmp_path, monkeypatch, mode):
    fn = FeatureFunction(name="flag", description="flags BAD", kind="llm", source=_LLM_SOURCE)
    rt = _runtime(tmp_path, monkeypatch, sample=16, workers=4, functions=[fn], n=40)
    monkeypatch.setattr(parquet_store, "MAX_BATCH_BYTES", 100)
    monkeypatch.setattr(parquet_store, "item_memory_bytes", lambda _: 40)
    gate = threading.Event()
    lock = threading.Lock()
    started = active = peak = 0

    def detector(source, item, **kwargs):
        nonlocal started, active, peak
        with lock:
            started += 1
            active += 1
            peak = max(peak, active)
            number = started
            if started == 2:
                gate.set()
        try:
            if number <= 2:
                assert gate.wait(5)
            return "BAD" in item["response_text"]
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(tools_eval, "run_detector", detector)
    result = tools_eval.evaluate_functions(names=[fn.name], mode=mode, chunk_size=1)
    assert result["ok"]
    assert peak == 2
    assert rt.library.get(fn.name).scores["errors"] == 0


@pytest.mark.parametrize('sample,evaluated_limit,new_limit,expected_scan', [
    (0, 40, 120, 40),  # A raised run limit never buys detector calls past the scored prefix.
    (0, 120, 40, 120),
    (40, 120, 120, 40),
], ids=["smaller-prefix", "larger-prefix", "sampled-panel"])
def test_explicit_support_review_reuses_only_completed_llm_evaluation(
    tmp_path, monkeypatch, sample, evaluated_limit, new_limit, expected_scan,
):
    fn = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=sample, workers=2, functions=[fn], n=120)
    rt.run.eval_limit = evaluated_limit
    assert tools_eval.evaluate_functions()['ok']
    rt.run.eval_limit = new_limit

    def unexpected_detector_call(*args, **kwargs):
        pytest.fail('Support review must reuse the completed scoring footprint')

    monkeypatch.setattr(tools_eval, 'run_detector', unexpected_detector_call)
    monkeypatch.setattr(claude_judge, 'judge_behavior_with_claude', lambda **kwargs: {
        'behavior_purity': 1.0, 'n_judged': len(kwargs['items']), 'verdicts': [],
    })
    result = tools_eval.judge_function(fn.name, evidence_n=5, review_phase='development')
    assert result['ok'], result
    assert result['n_scanned'] == result['scan_limit'] == expected_scan
    assert result['sample']['support_review']['selected_failed_task_n'] >= 5


def test_balanced_evaluation_weights_population_and_shares_python_panel(tmp_path, monkeypatch):
    source = "def detector(item, llm_client=None): return True\n"
    llm = FeatureFunction(name="semantic", description="always matches", source=source, kind="llm")
    python = FeatureFunction(name="structured", description="always matches", source=source,
                             kind="python")
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[llm, python])
    # Failures are a skewed minority (64/400) split unevenly across two benchmarks.
    items = _corpus(400)
    for i, item in enumerate(items):
        item["benchmark"] = "majority" if i < 370 else "minority"
        item["is_correct"] = not (i < 60 or 370 <= i < 374)
    rt.dataset = _FakeEvalDataset(items)
    result = tools_eval.evaluate_functions(limit=400)
    assert result["ok"], result
    panel = set(llm.scores["evaluation_item_ids"])
    selected_outcomes = Counter(item["is_correct"] for item in items if item["_item_id"] in panel)
    assert selected_outcomes == {False: 10, True: 10}
    assert llm.scores["raw_ppv"] == pytest.approx(64 / 400)
    assert llm.scores["raw_ppv"] == pytest.approx(python.scores["raw_ppv"])
    assert llm.scores["recall"] == python.scores["recall"] == 1
    assert llm.scores["search_panel"]["n_failed_tasks"] == 10
    assert llm.scores["sample_allocation_version"] == SCORING_SAMPLE_VERSION
    assert (llm.scores["search_panel"]["cohort_fingerprint"]
            == python.scores["search_panel"]["cohort_fingerprint"])
    assert result["llm_sample"]["unrepresented_strata"] == []
    assert len(result["llm_sample"]["strata"]) == 4


def test_tiny_panel_keeps_budget_and_exposes_unmeasured_population(tmp_path, monkeypatch):
    fn = FeatureFunction(name="flag", description="always matches", kind="llm",
                         source="def flag(item, llm_client=None): return True\n")
    _runtime(tmp_path, monkeypatch, sample=1, workers=2, functions=[fn], n=120)
    result = tools_eval.evaluate_functions(limit=120)
    assert result["ok"], result
    scores = fn.scores
    assert scores["sample_n"] == 1
    assert len(scores["sample_unrepresented_strata"]) == 3
    assert scores["evaluation_complete"] is False
    assert scores["metrics_scope"] == "unrepresented_strata"
    assert scores["recall"] is scores["raw_ppv"] is scores["ppv_at_pi0"] is None
    assert [scores["recall_ci_low"], scores["recall_ci_high"]] == [0, 1]
    assert search_association(scores) is None


def test_allocation_upgrade_refreshes_aggregate_but_reuses_compatible_predictions(
    tmp_path, monkeypatch,
):
    from autofa.metrics.search_panel import build_search_panel
    from autofa.session.family_coverage import cached_task_inventory

    fn = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[fn], n=120)
    calls = _record_calls(monkeypatch)
    assert tools_eval.evaluate_functions(limit=120)["ok"]
    panel = list(fn.scores["evaluation_item_ids"])
    current_recall = fn.scores["recall"]
    expected_panel = deepcopy(fn.scores["search_panel"])
    historical = build_search_panel(
        fn, cached_task_inventory(rt), item_ids=panel, run_id=rt.run.run_id,
        allocation_version="legacy_proportional_v1",
    )
    assert historical["cohort_fingerprint"] != expected_panel["cohort_fingerprint"]
    fn.scores.update(sample_allocation_version="legacy_proportional_v1",
                     search_panel=historical, recall=.987654321)
    rt.persist_library()
    assert tools_eval.evaluate_functions(limit=120)["ok"]
    assert len(calls) == 20
    assert fn.scores["evaluation_item_ids"] == panel
    assert fn.scores["recall"] == current_recall
    assert fn.scores["sample_allocation_version"] == SCORING_SAMPLE_VERSION
    assert fn.scores["search_panel"] == expected_panel


# Local census vs LLM panel cost split.


def detectors():
    return (FeatureFunction(name="plain", description="local predicate", source=_REGEX_SOURCE),
            FeatureFunction(name="flag", description="semantic predicate", source=_LLM_SOURCE,
                            kind="llm"))


def track(monkeypatch):
    calls = Counter()
    original = tools_eval.run_detector

    def run(source, item, **kwargs):
        calls[source] += 1
        return original(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", run)
    return calls


def test_default_local_census_ignores_scan_cap_and_reuses_unchanged_scores(tmp_path, monkeypatch):
    plain, _ = detectors()
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[plain], n=120)
    rt.run.eval_limit = 60
    calls = track(monkeypatch)
    observed = []
    result = tools_eval._evaluate_functions(
        _result_observer=lambda fn, item, prediction, error:
        observed.append((fn.name, item["_item_id"], prediction, error)))
    assert result["ok"] and result["llm_sample"] is None, result
    assert calls[plain.source] == len(observed) == plain.scores["n_scored"] == 120
    assert plain.scores["eval_mode"] == "full"
    assert plain.scores["evaluation_scope_complete"] and not plain.scores["sampled"]
    scores = deepcopy(plain.scores)
    # An exact census keeps Wilson/Fisher association fields in scores and feedback.
    assert scores["recall_ci_low"] <= scores["recall"] <= scores["recall_ci_high"]
    assert scores["fpr_ci_low"] <= scores["fpr"] <= scores["fpr_ci_high"]
    assert scores["feedback"]["recall_ci"][0] == scores["recall_ci_low"]
    assert scores["feedback"]["fisher_p"] == scores["fisher_p"] is not None
    again = tools_eval._evaluate_functions()
    assert again["ok"] and again["reused_names"] == [plain.name]
    assert plain.scores == scores and calls[plain.source] == 120
    # Explicit debugging requests do not silently reuse larger-cohort scores.
    debug = tools_eval._evaluate_functions(limit=30)
    assert debug["ok"] and plain.scores["n_scored"] == 30
    assert not plain.scores["evaluation_scope_complete"] and plain.scores["eval_truncated"]
    assert calls[plain.source] == 150


def test_mixed_batch_keeps_separate_population_scopes_and_one_search_panel(tmp_path, monkeypatch):
    plain, llm = detectors()
    # Without LLM sampling the LLM detector is capped by the scan limit; the local one is not.
    rt = _runtime(tmp_path, monkeypatch, sample=0, workers=2, functions=[plain, llm], n=120)
    rt.run.eval_limit = 60
    calls = track(monkeypatch)
    result = tools_eval._evaluate_functions()
    assert result["ok"], result
    assert calls[plain.source] == plain.scores["n_scored"] == 120
    assert calls[llm.source] == llm.scores["n_scored"] == 60
    assert plain.scores["evaluation_scope_complete"]
    assert not llm.scores["evaluation_scope_complete"]
    assert llm.scores["sampled"] is False
    py_panel, llm_panel = map(lambda fn: current_search_panel(fn.scores), (plain, llm))
    assert py_panel["cohort_fingerprint"] == llm_panel["cohort_fingerprint"]
    assert py_panel["task_recall"] == llm_panel["task_recall"]


@pytest.mark.parametrize("mode,expected", [("capped", 20), ("full", 120)])
def test_llm_parent_and_local_child_execute_only_on_chosen_cohort(
    tmp_path, monkeypatch, mode, expected,
):
    plain, llm = detectors()
    plain.explains = llm.name
    _runtime(tmp_path, monkeypatch, sample=20, workers=3, functions=[plain, llm], n=120)
    calls = track(monkeypatch)
    result = tools_eval._evaluate_functions(names=[plain.name], mode=mode)
    assert result["ok"], result
    assert calls[llm.source] == calls[plain.source] == expected
    assert plain.scores["n_scored"] == expected
    assert plain.scores["sampled"] is (mode == "capped")
    assert plain.scores["within_parent_sampled"] is (mode == "capped")
    if mode == "capped":
        assert plain.scores["within_n"] == pytest.approx(plain.scores["n_flagged"])
        assert set(plain.scores["evaluation_item_ids"]) == set(
            row for bucket in plain.scores["buckets"]
            for key in ("example_fail_ids", "example_pass_ids") for row in bucket[key])
        # Gate predictions are durable even though the local child is cheap to rerun.
        assert tools_eval._evaluate_functions(names=[plain.name])["ok"]
        assert calls[llm.source] == expected
        assert calls[plain.source] == expected * 2


@pytest.mark.parametrize("snapshot", ["transitive", "pinned_parent"])
def test_llm_cost_includes_executed_snapshot_dependencies(tmp_path, monkeypatch, snapshot):
    plain, llm = detectors()
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[llm], n=120)
    rt.run.eval_limit = 60
    child = FeatureFunction(name="child", description="composed detector", kind="python",
                            source="def child(item): return library.fire('flag', item)")
    child.meta = {"dependency_bundle": build_bundle(rt.library, depends_on=[llm.name])}
    rt.library.upsert(child)
    if snapshot == "transitive":
        grandchild = FeatureFunction(
            name="grandchild", description="transitive composition", kind="python",
            source="def grandchild(item): return library.fire('child', item)",
            meta={"dependency_bundle": build_bundle(rt.library, depends_on=[child.name])})
        child = grandchild
    else:
        # Cost follows the snapshot even if the live parent's implementation changed.
        llm.source, llm.kind = plain.source, "python"
    rt.library.upsert(child)
    assert tools_eval.evaluation_uses_llm(child, runtime=rt)
    calls = track(monkeypatch)
    assert tools_eval._evaluate_functions(names=[child.name])["ok"]
    assert calls[child.source] == child.scores["n_scored"] == 20
    assert child.scores["sampled"] is True


def test_result_observer_keeps_errors_and_abstentions_distinct(tmp_path, monkeypatch):
    plain, _ = detectors()
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[plain], n=120)
    rt.run.eval_limit = 10
    original = tools_eval.run_detector

    def fail(source, item, **kwargs):
        if item["_item_id"] == "id-0":
            raise ValueError("bad parse")
        if item["_item_id"] == "id-1":
            raise DetectorAbstained("source missing")
        return original(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", fail)
    observed = {}
    out = tools_eval._evaluate_functions(_result_observer=lambda fn, item, pred, error:
                                         observed.update({item["_item_id"]: error}))
    assert out["ok"], out
    assert len(observed) == 120
    assert isinstance(observed["id-0"], ValueError)
    assert isinstance(observed["id-1"], DetectorAbstained)
    assert plain.scores["errors"] == plain.scores["n_insufficient"] == 1
    assert plain.scores["n_scored"] == 118


def test_observer_keeps_timeout_result_without_claiming_a_completed_census(tmp_path, monkeypatch):
    plain, _ = detectors()
    _runtime(tmp_path, monkeypatch, sample=200, workers=2, functions=[plain])
    original = tools_eval.run_detector

    def timeout(source, item, **kwargs):
        if item["_item_id"] == "id-2":
            raise TimeoutError("expensive regex")
        return original(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", timeout)
    observed = {}
    out = tools_eval._evaluate_functions(_result_observer=lambda fn, item, pred, error:
                                         observed.update({item["_item_id"]: (pred, error)}))
    assert not out["ok"] and "timed out" in out["error"]
    assert set(observed) == {"id-0", "id-1", "id-2"}
    assert isinstance(observed["id-2"][1], TimeoutError)
    assert observed["id-2"][0] is None
    assert not plain.scores.get("evaluation_scope_complete")
    assert not out["evaluated_names"]


def test_mixed_parent_batch_keeps_earlier_blocked_screen_visible(tmp_path, monkeypatch):
    plain, llm = detectors()
    child = FeatureFunction(name="child", description="local child", source=_REGEX_SOURCE,
                            explains=llm.name)
    _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[plain, llm, child])
    monkeypatch.setattr("autofa.session.candidate_screen.ensure_candidate_screen",
                        lambda rt, fn, **kw: {"evaluation_allowed": fn.name != plain.name,
                                             "status": "blocked" if fn.name == plain.name
                                                       else "not_required"})
    result = tools_eval._evaluate_functions(names=[plain.name, child.name])
    assert not result["ok"] and result["partial"]
    assert result["blocked_functions"] == [plain.name]
    assert result["candidate_screens"][plain.name]["status"] == "blocked"
    assert result["evaluated_names"] == [child.name]
    assert [row["name"] for row in result["results"]] == [child.name]
    assert result["eval_mode"] == "mixed"
    assert "llm_sample" not in result
    assert result["evaluation_groups"][1]["within"] == llm.name


def test_missing_parent_is_a_tool_error_before_scoring(tmp_path, monkeypatch):
    plain, _ = detectors()
    plain.explains = "absent"
    _runtime(tmp_path, monkeypatch, sample=200, workers=2, functions=[plain])
    calls = track(monkeypatch)
    result = tools_eval._evaluate_functions(names=[plain.name])
    assert not result["ok"] and "within parent not found" in result["error"]
    assert not calls


def test_round_audit_samples_costly_parent_with_child_and_keeps_search_panel(tmp_path, monkeypatch):
    plain, llm = detectors()
    child = FeatureFunction(name="child", description="parent scoped", source=_REGEX_SOURCE,
                            explains=llm.name)
    rt = _runtime(tmp_path, monkeypatch, sample=16, workers=3, functions=[plain, llm, child],
                  n=120)
    rt.run.eval_limit = 60
    calls = track(monkeypatch)
    result = tools_eval._evaluate_functions(names=[plain.name, child.name], _round_fraction=.3)
    assert result["ok"], result
    assert calls[plain.source] == 120 + calls[llm.source]  # child shares the local implementation
    assert 36 <= calls[llm.source] <= 52  # 30% audit plus the original 16-row comparison panel
    assert child.scores["within_parent"] == llm.name  # The explained parent applies by default.
    assert child.scores["n_scored"] == child.scores["sample_n"] == 36
    assert child.scores["sample_frame_n"] == 120
    assert child.scores["within_n"] == pytest.approx(child.scores["n_flagged"])
    assert child.scores["round_validation"]["complete"]
    assert current_search_panel(child.scores)["cohort_fingerprint"] == current_search_panel(
        plain.scores)["cohort_fingerprint"]


# Bounded lookahead.


def test_scoring_reaches_later_work_while_first_call_waits_and_preserves_cache(
    tmp_path, monkeypatch,
):
    _scoring_runtime(tmp_path, monkeypatch)
    fn = FeatureFunction(name="flag", description="", source="def flag(item): return True",
                         kind="llm")
    items = [{"_item_id": str(index)} for index in range(48)]
    later_completed = threading.Event()
    first_workers = threading.Barrier(8)
    guard = threading.Lock()
    active = peak = 0
    calls = Counter()

    def detector(source, item, **kwargs):
        nonlocal active, peak
        index = int(item["_item_id"])
        with guard:
            active += 1
            peak = max(peak, active)
            calls[index] += 1
        try:
            if index < 8:
                first_workers.wait(timeout=5)
            if index == 0:
                assert later_completed.wait(5), "slow head prevented workers from advancing"
            if index == 40:
                later_completed.set()
            return bool(index % 2)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(tools_eval, "run_detector", detector)
    pairs = [(fn, item) for item in items]
    first = list(tools_eval._iter_scored_items(pairs, workers=8, llm=None))
    assert [(item["_item_id"], pred, error) for _, item, pred, error in first] == [
        (str(index), bool(index % 2), None) for index in range(48)
    ]
    assert peak == 8
    assert calls == Counter(range(48))

    # A second full pass reuses every prediction, including those completed
    # beyond the slow first item before ordered consumption could progress.
    second = list(tools_eval._iter_scored_items(pairs, workers=8, llm=None))
    assert second == first
    assert calls == Counter(range(48))


def test_larger_lookahead_still_obeys_retained_input_byte_limit(tmp_path, monkeypatch):
    _scoring_runtime(tmp_path, monkeypatch)
    fn = FeatureFunction(name="flag", description="", source="def flag(item): return True")
    reads, calls = [], []

    def pairs():
        for index in range(8):
            reads.append(index)
            yield fn, {"_item_id": str(index)}

    def detector(source, item, **kwargs):
        calls.append(int(item["_item_id"]))
        return True

    monkeypatch.setattr(tools_eval, "run_detector", detector)
    monkeypatch.setattr(parquet_store, "item_memory_bytes",
                        lambda item: parquet_store.MAX_BATCH_BYTES // 2 + 1)
    scored = tools_eval._iter_scored_items(pairs(), workers=8, llm=None)
    first = next(scored)
    assert first[1]["_item_id"] == "0"
    assert calls == [0]  # Two retained rows would exceed the unchanged byte cap.
    assert reads == [0, 1]  # One unadmitted lookahead row is the existing contract.
    assert [item["_item_id"] for _, item, _, _ in scored] == [str(i) for i in range(1, 8)]
    assert calls == list(range(8))


@pytest.mark.parametrize("kind,updates", [("llm", True), ("regex", False)])
def test_serial_llm_refreshes_capacity_after_backoff_without_observing_local_scans(
    tmp_path, monkeypatch, kind, updates,
):
    _scoring_runtime(tmp_path, monkeypatch)
    config = tools_eval.get_runtime().config
    config.llm_eval_workers = 8
    config.llm_eval_max_workers = 1
    snapshot = tmp_path / "scoring_capacity.json"
    snapshot.write_text(json.dumps({"effective_workers": 8}))
    fn = FeatureFunction(name="flag", description="", source="def flag(item): return True",
                         kind=kind)
    monkeypatch.setattr(tools_eval, "run_detector", lambda *args, **kwargs: True)
    results = list(tools_eval._iter_scored_items(
        [(fn, {"_item_id": "one"})], workers=1, llm=None,
    ))
    assert results[0][2:] == (True, None)
    observed = json.loads(snapshot.read_text())
    if updates:
        assert observed["effective_workers"] == 1
        assert observed["configured_ceiling"] == 1
        assert observed["limiting_reason"] == "configured_ceiling"
    else:
        assert observed == {"effective_workers": 8}


# Discovery sample rotation.


@pytest.mark.parametrize("ssot, sampling_options, counts", [
    (False, {}, {}),
    (True, {"prefer_informative": False}, {"n_fail": 8, "n_pass": 4}),
])
def test_round_samples_visit_distinct_tasks_reproducibly_and_continue_the_sequence(
    tmp_path, monkeypatch, ssot, sampling_options, counts,
):
    expected_fail, expected_pass = counts.get("n_fail", 12), counts.get("n_pass", 2)
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "autofa"))
    monkeypatch.setattr(tools_library, "_detector_exploration", lambda: {"message": "Explore"})
    monkeypatch.setattr(tools_library, "emit_runtime_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "autofa.session.discovery_target.failure_mode_target_for_runtime", lambda rt: {},
    )
    path = tmp_path / "traces.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "m", "benchmark": "b", "problem_id": f"task-{task}",
         "run_idx": repeat, "is_correct": task >= 50,
         "response_text": "Synthetic trace " * (task + 4)}
        for task in range(100) for repeat in range(3)
    ]), path)
    cfg = Config(exploration_seed=42, exploration_ssot=ssot)
    dataset = Dataset(parquet_paths=[path], model_id="m", benchmarks=["b"])

    def fresh_run(*, seed_library_path=None):
        run = RunConfig.create([path], "m", ["b"], cfg,
                               seed_library_path=seed_library_path)
        rt = Runtime(dataset=dataset, library=FunctionLibrary(), config=cfg, run=run)
        set_runtime(rt)
        return rt

    def sample():
        result = tools_data.sample_items(**counts, **sampling_options)
        assert result["ok"] and result["distinct_tasks"]
        assert result["n_fail"] == result["n_failed_tasks"] == expected_fail
        assert result["n_pass"] == expected_pass
        assert sum(bool(item["is_correct"]) for item in result["items"]) == expected_pass
        assert result["n_repeated_failed_attempts"] == 0
        tasks = {item["source_task_identity"] for item in result["items"]}
        assert len(tasks) == len(result["items"]) == expected_fail + expected_pass
        return result["seed_used"], sorted(item["_item_id"] for item in result["items"]), tasks

    sequences = []
    for _ in range(2):
        rt = fresh_run()
        sequence = []
        explicit_sample = None
        for index in range(3):
            before = (rt.run.exploration_base_seed, rt.run.exploration_seed,
                      rt.run.exploration_sequence_index)
            overridden = tools_data.sample_items(seed=777, **counts, **sampling_options)
            assert (rt.run.exploration_base_seed, rt.run.exploration_seed,
                    rt.run.exploration_sequence_index) == before
            if explicit_sample is not None:
                assert overridden == explicit_sample
            explicit_sample = overridden
            sampled = sample()
            assert sampled[0] == rt.run.exploration_seed
            assert sample() == sampled  # Repeated reads in a round remain stable.
            sequence.append(sampled)
            if index < 2:
                complete_round_batch(rt)
                assert tools_library.advance_round()["ok"]
        assert len({row[0] for row in sequence}) == 3
        assert all(first[2] != second[2]
                   for first, second in zip(sequence, sequence[1:], strict=False))
        assert len(set.union(*(row[2] for row in sequence))) > expected_fail + expected_pass
        sequences.append(sequence)
    assert sequences[0] == sequences[1]  # A new run ID does not change a fixed main seed.

    # Keep an exported snapshot: advance_round also updates the working library.
    working_path = rt.persist_library()
    seed_path = tmp_path / "exported-library.json"
    seed_path.write_bytes(working_path.read_bytes())
    previous_index = rt.run.exploration_sequence_index
    complete_round_batch(rt)
    assert tools_library.advance_round()["ok"]
    uninterrupted_next = sample()
    for _ in range(2):
        continued = fresh_run(seed_library_path=str(seed_path))
        assert continued.run.exploration_sequence_index == previous_index + 1
        assert sample() == uninterrupted_next
        assert sample()[2] != sequences[0][0][2]

    # Continuing the newer working snapshot advances from its newer cursor.
    continued = fresh_run(seed_library_path=str(working_path))
    assert continued.run.exploration_sequence_index == previous_index + 2
    assert sample()[0] != uninterrupted_next[0]


def test_default_preference_keeps_informative_task_bias_without_freezing_failures(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "autofa"))
    path = tmp_path / "traces.parquet"
    # The full candidate pool fits the index oversample. Ranking it directly
    # used to select the same 12 longest failed tasks for every round seed.
    rows = [
        {"model_id": "m", "benchmark": "b", "problem_id": f"task-{task}",
         "run_idx": repeat, "is_correct": task >= 60,
         "response_text": ("Synthetic trace " * (task + 4) if task >= 12 else "[empty]")}
        for task in range(72) for repeat in range(2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    config = Config()
    run = RunConfig.create([path], "m", ["b"], config)
    dataset = Dataset(parquet_paths=[path], model_id="m", benchmarks=["b"])
    set_runtime(Runtime(dataset=dataset, library=FunctionLibrary(), config=config, run=run))

    from autofa.session.exploration import round_exploration_seed

    selected = []
    for index in range(3):
        result = tools_data.sample_items(seed=round_exploration_seed(42, index))
        assert result["ok"] and result["prefer_informative"]
        assert result["n_fail"] == result["n_failed_tasks"] == 12
        assert result["n_pass"] == 2
        assert result["n_repeated_failed_attempts"] == 0
        assert all(item["response_text"] != "[empty]" for item in result["items"])
        selected.append({item["problem_id"] for item in result["items"]
                         if not item["is_correct"]})
    assert all(first != second for first, second in zip(selected, selected[1:], strict=False))
    assert len(set.union(*selected)) > 12
