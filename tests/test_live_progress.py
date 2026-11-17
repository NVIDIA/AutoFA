# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live display progress and targets never alter validated attempts or discovery state."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.library import FunctionLibrary
from autofa.mcp import tools_eval
from autofa.mcp.runtime import Runtime
from autofa.metrics import claude_judge
from autofa.metrics.completeness import comparable_evaluations
from autofa.metrics.live_progress import (
    display_progress_snapshot,
    display_progress_stats,
    refresh_display_progress,
)
from autofa.session import discovery_target
from autofa.util import atomic

from support.coverage import candidate
from support.discovery import graded_candidate
from support.repair_review import _result, _runtime

PANEL = ["a1", "a2", "b1", "p1"]
LEGACY_STANDARD_AXES = ("ppv_at_pi0", "recall", "behavior_purity")


def _sampled_candidate(config):
    fn = candidate(panel_ids=PANEL, config=config)
    fn.scores.update(
        sampled=True, task_recall_scope="observed_lower_bound", evaluation_item_ids=list(PANEL),
        sample_n=len(PANEL), sample_frame_n=6, n_scored=len(PANEL), sample_strata=2,
    )
    return fn


def test_sampled_development_review_has_provisional_progress_without_validated_frontier():
    fn = candidate(reviewed=("yes", "yes"), panel_ids=PANEL)
    fn.scores.update(sampled=True, task_recall_scope="observed_lower_bound")
    before = deepcopy(fn.to_dict())
    snapshot = display_progress_snapshot([fn], Config(behavior_strictness="causal"), run_id="r1")
    assert snapshot["search"]["status"] == "available"
    assert snapshot["search"]["value"] == pytest.approx(.5 ** .5)
    assert snapshot["search"]["scope"] == "fixed_scoring_panel"
    assert snapshot["validated"] == {"status": "pending", "value": None, "n_ranked": 0}
    assert fn.to_dict() == before


def test_sampling_allocation_change_resets_trend_even_when_ids_coincide():
    cfg = Config(behavior_strictness="causal")
    fn = _sampled_candidate(cfg)
    first = display_progress_snapshot([fn], cfg, run_id="r1")
    fn.scores["sample_allocation_version"] = "outcome_balanced_v2"
    second = display_progress_snapshot([fn], cfg, run_id="r1")
    assert first["search"]["comparison_key"] != second["search"]["comparison_key"]
    stats = display_progress_stats([{"display_progress": first}, {"display_progress": second}])
    assert stats["search_values"] == [second["search"]["value"]]
    assert stats["search_delta"] is None


@pytest.mark.parametrize("change", ["panel", "run", "missing"])
def test_delta_and_best_do_not_cross_incompatible_or_unavailable_history(change):
    cfg = Config(behavior_strictness="causal")
    fn = candidate(reviewed=("yes", "yes"), panel_ids=PANEL)
    first = display_progress_snapshot([fn], cfg, run_id="r1")
    panel = ["a1", "b1", "c1", "p1"] if change == "panel" else PANEL
    fn = candidate(reviewed=("yes", "yes"), matched=("a1", "b1"), panel_ids=panel)
    if change == "run":
        fn.scores["evaluation_run_id"] = "r2"
        fn.scores["search_panel"]["run_id"] = "r2"
    second = display_progress_snapshot([fn], cfg, run_id="r2" if change == "run" else "r1")
    history = [{"display_progress": first}, {"display_progress": second}]
    if change == "missing":
        history.insert(1, {"frontier_value": 0})
    stats = display_progress_stats(history)
    assert stats["search_values"] == [second["search"]["value"]]
    assert stats["search_delta"] is None
    assert stats["best_search_value"] == second["search"]["value"]


def test_comparable_trend_and_current_removal():
    cfg = Config(behavior_strictness="causal")
    first = display_progress_snapshot([candidate(panel_ids=PANEL)], cfg, run_id="r1")
    fn = candidate(reviewed=("yes", "yes"), matched=("a1", "b1"), panel_ids=PANEL)
    current = display_progress_snapshot([fn], cfg, run_id="r1")
    history = [{"display_progress": first}, {"display_progress": current}]
    before = deepcopy(history)
    for stats in (display_progress_stats(history),
                  display_progress_stats(history, current=current)):
        assert stats["search_values"] == [first["search"]["value"], current["search"]["value"]]
        assert stats["search_delta"] == pytest.approx(1 - first["search"]["value"])
        assert stats["n_evaluations"] == 2
        assert stats["best_search_value"] == 1
    removed = display_progress_snapshot([], cfg, run_id="r1")
    assert display_progress_stats(history, current=removed)["search_values"] == []
    assert history == before


class _ObservedHistoryLock:
    """Signal when the worker reaches a real lock boundary, without sleeps."""

    def __init__(self, lock):
        self.lock = lock
        self.owner = threading.get_ident()
        self.worker_waiting = threading.Event()

    def __enter__(self):
        if threading.get_ident() != self.owner:
            self.worker_waiting.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


def test_review_waits_for_history_writer_before_selecting_and_mutating_latest_evaluation(
    tmp_path, monkeypatch,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    first = {"evaluated_names": ["old"], "evaluation_complete": False}
    newer = {"evaluated_names": ["new"], "evaluation_complete": False}
    rt.scores_history = [deepcopy(first)]
    lock = _ObservedHistoryLock(rt._scores_history_lock)
    rt._scores_history_lock = lock

    with ThreadPoolExecutor(max_workers=1) as pool:
        with lock:
            refreshed = pool.submit(refresh_display_progress, rt)
            assert lock.worker_waiting.wait(timeout=5)
            # No display dictionary may be inserted while another writer owns
            # the serialization lock, nor may a stale evaluation be selected.
            assert rt.scores_history == [first]
            rt.scores_history.append(deepcopy(newer))
            rt.persist_scores_history()
        refreshed.result(timeout=5)

    assert rt.scores_history[0] == first
    assert rt.scores_history[1]["display_progress"]["search"]["status"] == "pending"
    saved = json.loads((rt.run.run_dir() / "scores_history.json").read_text())
    assert saved == rt.scores_history


def test_evaluation_append_and_persist_wait_for_the_same_history_transaction(
    tmp_path, monkeypatch,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    first = {"evaluated_names": ["earlier"], "evaluation_complete": False}
    rt.scores_history = [deepcopy(first)]
    lock = _ObservedHistoryLock(rt._scores_history_lock)
    rt._scores_history_lock = lock

    with ThreadPoolExecutor(max_workers=1) as pool:
        with lock:
            evaluated = pool.submit(tools_eval.evaluate_functions, names=[fn.name], mode="full")
            assert lock.worker_waiting.wait(timeout=5)
            assert rt.scores_history == [first]
            # Simulate a refresh which is still serializing its selected row.
            rt.scores_history[0]["display_progress"] = {"version": 2, "search": {}}
            rt.persist_scores_history()
        result = evaluated.result(timeout=5)

    assert result["ok"], result
    assert len(rt.scores_history) == 2
    assert rt.scores_history[-1]["evaluated_names"] == [fn.name]
    assert rt.scores_history[0]["display_progress"] == {"version": 2, "search": {}}
    saved = json.loads((rt.run.run_dir() / "scores_history.json").read_text())
    assert saved == rt.scores_history


def test_review_refresh_cannot_replace_a_newer_evaluation_history(tmp_path, monkeypatch):
    rt = Runtime(
        dataset=None, library=FunctionLibrary(), config=Config(),
        run=SimpleNamespace(run_id="test", run_dir=lambda: tmp_path),
        scores_history=[{"evaluated_names": ["first"]}],
    )
    rt.persist_scores_history()
    review_flushed = threading.Event()
    release_review = threading.Event()
    evaluation_attempted = threading.Event()
    evaluation_writer_entered = threading.Event()
    writer_role = threading.local()
    original_temporary_file = atomic.tempfile.NamedTemporaryFile

    @contextmanager
    def controlled_temporary_file(*args, **kwargs):
        if writer_role.name == "evaluation":
            evaluation_writer_entered.set()
        with original_temporary_file(*args, **kwargs) as stream:
            yield stream
        if writer_role.name == "review":
            # Pause the real atomic writer after its payload has been serialized
            # and flushed, but before it replaces scores_history.json.
            review_flushed.set()
            assert release_review.wait(timeout=5)

    def review():
        writer_role.name = "review"
        refresh_display_progress(rt)

    def evaluation():
        writer_role.name = "evaluation"
        evaluation_attempted.set()
        rt.persist_scores_history()

    monkeypatch.setattr(atomic.tempfile, "NamedTemporaryFile", controlled_temporary_file)
    with ThreadPoolExecutor(max_workers=2) as pool:
        review_future = pool.submit(review)
        try:
            assert review_flushed.wait(timeout=5)
            rt.scores_history.append({"evaluated_names": ["second"]})
            evaluation_future = pool.submit(evaluation)
            assert evaluation_attempted.wait(timeout=5)
            # Serialization and replacement form one transaction. A second writer
            # must wait; otherwise the paused older payload can erase its result.
            assert not evaluation_writer_entered.wait(timeout=0.1)
        finally:
            release_review.set()
        review_future.result(timeout=5)
        evaluation_future.result(timeout=5)

    saved = json.loads((tmp_path / "scores_history.json").read_text())
    assert [row["evaluated_names"] for row in saved] == [["first"], ["second"]]
    assert saved == rt.scores_history
    assert saved[0]["display_progress"]["search"]["status"] == "pending"


def test_evaluation_and_judgment_refresh_telemetry_without_rewriting_attempts(
    tmp_path, monkeypatch,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert rt.scores_history[-1]["display_progress"]["search"]["status"] == "pending"
    baseline = [{key: value for key, value in row.items() if key != "display_progress"}
                for row in deepcopy(rt.scores_history)]
    comparable_before = deepcopy(comparable_evaluations(rt.scores_history))
    def judge(**kwargs):
        reviewed = _result(fn, ["yes"] * len(kwargs["items"]))
        pairs = zip(reviewed["verdicts"], kwargs["items"], strict=True)
        for index, (row, item) in enumerate(pairs):
            row.update(item_id=item["_item_id"], snapshot_id=f"s-{index}",
                       claim_contract_id=reviewed["claim_hash"], completion_status="complete",
                       verification_status="confirmed")
            row["initial_review"] = dict(row)
        return reviewed

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    result = tools_eval.judge_function(fn.name)
    assert result["ok"] and not result.get("skipped"), result
    assert result["search_score"]["eligible"], result["search_score"]
    snapshot = rt.scores_history[-1]["display_progress"]
    assert snapshot["search"]["status"] == "available"
    assert snapshot["search"]["value"] > 0
    assert [{key: value for key, value in row.items() if key != "display_progress"}
            for row in rt.scores_history] == baseline
    assert [row.get("evaluated_names") for row in comparable_evaluations(rt.scores_history)] == [
        row.get("evaluated_names") for row in comparable_before
    ]
    saved = json.loads((rt.run.run_dir() / "scores_history.json").read_text())
    assert len(saved) == len(baseline)
    assert saved[-1]["display_progress"] == snapshot


def test_telemetry_write_failure_does_not_require_repeating_a_review(tmp_path, monkeypatch):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.scores_history = [{"evaluated_names": ["guess"], "evaluation_complete": False}]

    def unavailable():
        raise OSError("test disk unavailable")

    monkeypatch.setattr(rt, "persist_scores_history", unavailable)
    refresh_display_progress(rt)  # The detector's result is already durable; no retry.
    assert len(rt.scores_history) == 1
    assert rt.scores_history[0]["display_progress"]["search"]["status"] == "pending"


def test_unknown_cohort_breaks_history_without_hiding_current_score():
    cfg = Config(behavior_strictness="causal", objective_axes=list(LEGACY_STANDARD_AXES))
    fn = _sampled_candidate(cfg)
    first = display_progress_snapshot([fn], cfg, run_id="r1")
    # A capped scan without its evaluated row identities has no known cohort.
    fn.scores.update(sampled=False, eval_mode="capped", evaluation_scope_complete=False,
                     eval_truncated=True)
    fn.scores.pop("evaluation_item_ids")
    unknown = display_progress_snapshot([fn], cfg, run_id="r1")
    assert unknown["search"]["status"] == "available"
    assert unknown["search"]["value"] == first["search"]["value"]
    assert unknown["search"]["comparison_key"] is None
    history = [{"display_progress": s} for s in (first, unknown)]
    stats = display_progress_stats(history)
    assert stats["search_values"] == []
    assert stats["search_delta"] is stats["best_search_value"] is None
    # Returning to the original known cohort cannot bridge the provenance gap.
    stats = display_progress_stats([*history, {"display_progress": first}])
    assert stats["search_values"] == [first["search"]["value"]]


def test_adding_an_alternative_preserves_best_score_but_real_losses_lower_it():
    cfg = Config(behavior_strictness="causal")
    leader = candidate("leader", reviewed=("yes", "yes"), panel_ids=PANEL, config=cfg)
    leader.scores["ppv_at_pi0"] = .8
    broader = candidate("broader", reviewed=("yes", "yes"), matched=("a1", "b1"),
                        panel_ids=PANEL, config=cfg)
    broader.scores["ppv_at_pi0"] = .1
    first = display_progress_snapshot([leader], cfg, run_id="r1")
    added = display_progress_snapshot([leader, broader], cfg, run_id="r1")
    assert first["search"]["value"] == pytest.approx(.8 ** .2 * .5 ** .5)
    assert added["search"]["value"] == first["search"]["value"]
    assert added["search"]["aggregation"] == "best_current_search_score"
    history = [{"display_progress": snapshot} for snapshot in (first, added)]
    assert display_progress_stats(history)["search_delta"] == 0

    leader.scores["ppv_at_pi0"] = .2
    weakened = display_progress_snapshot([leader, broader], cfg, run_id="r1")
    history.append({"display_progress": weakened})
    stats = display_progress_stats(history)
    assert weakened["search"]["value"] == pytest.approx(.1 ** .2)
    assert stats["search_delta"] < 0
    assert stats["best_search_value"] == first["search"]["value"]

    removed = display_progress_snapshot([leader], cfg, run_id="r1")
    history.append({"display_progress": removed})
    assert removed["search"]["value"] == pytest.approx(.2 ** .2 * .5 ** .5)
    assert display_progress_stats(history)["search_delta"] < 0


@pytest.mark.parametrize("round_idx,minimum,expected", [
    (2, 2, "EXCELLENT"), (3, 2, "GOOD"), (3, 1, "OUTSTANDING"),
])
def test_runtime_display_target_matches_agent_round_target_without_scans_or_writes(
    tmp_path, monkeypatch, round_idx, minimum, expected,
):
    from autofa.session import round_validation

    rt = SimpleNamespace(
        library=FunctionLibrary(functions=[
            graded_candidate(0, sampled=False, coverage=.05, tasks=5)]),
        config=Config(behavior_strictness="causal"),
        run=SimpleNamespace(run_id="fixture-run", round=round_idx, max_rounds=3,
                            min_failure_modes=minimum, run_dir=lambda: tmp_path),
    )
    monkeypatch.setattr(
        "autofa.session.family_coverage.failed_task_inventory",
        lambda *args, **kwargs: pytest.fail("Display targets must not load source inventory"))
    monkeypatch.setattr(
        discovery_target, "failure_mode_target_for_runtime",
        lambda *args, **kwargs: pytest.fail("Display targets must not run final qualification"))
    monkeypatch.setattr(
        round_validation, "failure_mode_target_for_runtime",
        lambda *args, **kwargs: pytest.fail("Display targets must not run final qualification"))
    before = deepcopy(rt.library.to_dict())
    keys = set(vars(rt))
    run_state = dict(vars(rt.run))
    raw = discovery_target.discovery_good_target_for_runtime(rt)
    original = deepcopy(raw)
    target = discovery_target.agent_failure_mode_target(
        raw, round_idx=round_idx, max_rounds=3)
    assert discovery_target.agent_target_grade_for_runtime(rt) == expected == target["target_grade"]
    assert raw == original
    assert rt.library.to_dict() == before
    assert set(vars(rt)) == keys
    assert vars(rt.run) == run_state


def test_display_target_failure_is_unavailable_and_cannot_interrupt_business_work(monkeypatch):
    def unavailable(rt):
        raise RuntimeError("malformed historical display evidence")

    monkeypatch.setattr(discovery_target, "discovery_good_target_for_runtime", unavailable)
    rt = SimpleNamespace()
    assert discovery_target.agent_target_grade_for_runtime(rt) is None
    assert vars(rt) == {}
