# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A discovery quota counts validated modes and drives actual follow-up turns."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.data.parquet_store import Dataset
from autofa.harness.claude import TurnResult
from autofa.mcp import tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import build_claim_contract, claim_hash
from autofa.metrics.discovery_grades import GRADE_CODES, grade_for_metrics, grade_summary
from autofa.session import runner
from autofa.session.discovery_stall import observe_discovery_stall
from autofa.session.discovery_target import (
    DiscoveryProgress,
    agent_failure_mode_target,
    discovery_good_target,
    discovery_good_target_for_runtime,
    discovery_grade_assessment,
    failure_mode_target,
    failure_mode_target_for_runtime,
)
from autofa.session.polishing import current_claims
from autofa.session.round_stagnation import complete_discovery_round

from support.discovery import _quality_stall_assessment, graded_candidate, keeper, source_inventory
from support.discovery import interactive as interactive
from support.discovery import interactive_runner as interactive_runner
from support.task_support import attach_sampled_task_support


def target(functions, minimum=3, *, inventory=None):
    return failure_mode_target(functions, minimum=minimum,
                               config=Config(behavior_strictness="causal"),
                               inventory=source_inventory() if inventory is None else inventory)


def evaluation_row(functions, *, complete=True):
    """A scores_history row naming the claims evaluate_functions attempted."""
    return {"evaluation_complete": complete,
            "evaluated_names": [fn.name for fn in functions],
            "evaluated_claims": current_claims(
                SimpleNamespace(library=SimpleNamespace(functions=functions)))}


def test_count_is_current_validated_modes_not_library_size_or_report_top_x():
    functions = [keeper(i) for i in range(4)]
    before = deepcopy([fn.to_dict() for fn in functions])
    progress = failure_mode_target(functions, minimum=3,
                                   config=Config(behavior_strictness="causal", top_x=1),
                                   inventory=source_inventory())
    assert progress["minimum_met"]
    assert progress["n_validated_failure_modes"] == 4
    assert progress["remaining_failure_modes"] == 0
    assert progress["validated_failure_modes"] == [f"mode_{i}" for i in range(4)]
    assert [fn.to_dict() for fn in functions] == before
    assert target(functions[:2])["remaining_failure_modes"] == 1
    assert target([], minimum=0)["minimum_met"]


@pytest.mark.parametrize("changes", [
    {"sampled": True}, {"errors": 1}, {"evaluation_claim_hash": "0" * 64},
    {"judge_policy_hash": "old"},
    # Only the contract-version check excludes an older review of an unchanged claim.
    {"judge_contract_version": "legacy"},
    {"behavior_n_yes": 0, "behavior_n_no": 4, "behavior_purity": 0.0},
    {"behavior_n_no": 0, "behavior_n_uncertain": 1},
    {"n_fail_caught": 0}, {"judge_prefilter": {"status": "skip"}},
    {**contingency(1, 8, 9, 2)},
], ids=["sampled", "errors", "stale_evaluation", "stale_policy", "legacy_review", "rejected",
        "uncertain", "no_failures", "prefilter_skip", "low_precision"])
def test_ineligible_candidates_cannot_fill_the_minimum(changes):
    result = target([keeper(**changes)], minimum=1)
    assert not result["minimum_met"]
    assert result["n_validated_failure_modes"] == 0
    assert result["pending_failure_modes"][0]["reason"]


def test_repaired_and_rejudged_candidate_can_fill_target():
    fn = keeper(behavior_n_no=0, behavior_n_uncertain=1)
    assert not target([fn], 1)["minimum_met"]
    fn.scores.update(behavior_n_uncertain=0, behavior_n_no=0)
    assert target([fn], 1)["minimum_met"]
    fn.mechanism += " A different claim."
    assert not target([fn], 1)["minimum_met"]


# Good coverage is exclusive at 1% (8/800), whatever the judge cost filter selects.
@pytest.mark.parametrize("caught,expected", [(8, False), (9, True)])
def test_good_coverage_count_is_independent_of_judge_cost_filter(caught, expected):
    fn = keeper(**contingency(caught, 0, 800 - caught, 800), n_scored=1600)
    inventory = source_inventory(800, 800)
    config = Config(behavior_strictness="causal")
    result = failure_mode_target([fn], minimum=1, config=config, inventory=inventory)
    assert result["minimum_met"] is expected


def test_minimum_two_needs_two_good_detectors_not_two_library_entries():
    inventory = source_inventory(800, 800)
    good = keeper(**contingency(49, 9, 751, 791), n_scored=1600)
    narrow = keeper(1, **contingency(8, 0, 792, 800), n_scored=1600)
    low_precision = keeper(2, **contingency(48, 9, 752, 791), n_scored=1600)
    progress = target([good, narrow, low_precision], minimum=2, inventory=inventory)
    assert not progress["minimum_met"]
    assert progress["n_qualifying_failure_modes"] == 1
    assert progress["remaining_failure_modes"] == 1
    # The inclusive precision floor admits 49/9 but not 48/9, whatever the recall.
    assert low_precision.name in {row["name"] for row in progress["pending_failure_modes"]}
    second_good = keeper(3, **contingency(56, 8, 744, 792), n_scored=1600)
    progress = target([good, narrow, low_precision, second_good], minimum=2, inventory=inventory)
    assert progress["minimum_met"]
    assert progress["qualifying_failure_modes"] == [good.name, second_good.name]


def test_discovery_counts_visible_projected_good_without_final_scan_or_extra_reviews():
    functions = [graded_candidate(0), graded_candidate(1, coverage=.3)]
    before = deepcopy([fn.to_dict() for fn in functions])
    cfg = Config(behavior_strictness="causal")
    result = discovery_good_target(functions, minimum=2, config=cfg)
    grades = grade_summary(functions, behavior_strictness="causal")
    assert result["minimum_met"]
    assert result["n_qualifying_failure_modes"] == grades["n_good_or_better"] == 2
    assert all(row["measured"]["n_confirmed_failed_datapoints"] == 2
               and row["measured"]["estimated_n_confirmed_failed_tasks"] == pytest.approx(8)
               and not row["measured"]["corpus_evaluation_complete"]
               for row in result["failure_mode_eligibility"])
    assert not failure_mode_target(functions, minimum=2, config=cfg,
                                   inventory=source_inventory())["minimum_met"]
    # Model aspirations remain separate and retain the validated projection.
    agent = agent_failure_mode_target(result)
    assert agent["n_qualifying_failure_modes"] == 1
    assert agent["qualifying_failure_modes"] == ["graded_1"]
    assert [fn.to_dict() for fn in functions] == before


@pytest.mark.parametrize("change", [
    {"evaluation_stale": True}, {"judge_policy_hash": "obsolete"},
    {"recall": .005}, {"ppv_at_pi0": .6}, {"task_support": None},
], ids=["stale", "policy", "coverage", "precision", "no_task_support"])
def test_discovery_cannot_count_stale_or_below_good_candidates(change):
    fn = graded_candidate()
    fn.scores.update(change)
    result = discovery_good_target([fn], minimum=1, config=Config(behavior_strictness="causal"))
    assert not result["minimum_met"]
    assert result["pending_failure_modes"][0]["reason"]


def test_discovery_honors_active_policy_and_current_claim():
    fn = graded_candidate()
    assert not discovery_good_target([fn], minimum=1,
                                     config=Config(behavior_strictness="contextual"))["minimum_met"]
    fn.description += " Different behavior."
    assert not discovery_good_target([fn], minimum=1,
                                     config=Config(behavior_strictness="causal"))["minimum_met"]


def test_discovery_deduplicates_public_good_detectors():
    first, second = graded_candidate(0), graded_candidate(1)
    second.description, second.mechanism = first.description, first.mechanism
    attach_sampled_task_support(second)
    result = discovery_good_target([first, second], minimum=2,
                                   config=Config(behavior_strictness="causal"))
    assert result["n_qualifying_failure_modes"] == 1
    assert "Duplicate" in result["pending_failure_modes"][0]["reason"]


def test_discovery_full_scan_still_requires_observed_task_support():
    low, good = graded_candidate(0, sampled=False, tasks=2), graded_candidate(1, sampled=False,
                                                                            tasks=5)
    low.scores.update(estimated_n_confirmed_failed_tasks=50,
                      task_support_comparison_basis="estimated",
                      task_support_estimate_status="provisional")
    result = discovery_good_target([low, good], minimum=1,
                                   config=Config(behavior_strictness="causal"))
    assert result["qualifying_failure_modes"] == [good.name]
    assert result["failure_mode_eligibility"][0]["measured"]["corpus_evaluation_complete"]


# The same claim, or a renamed predicate (even with a new docstring), is the same detector.
@pytest.mark.parametrize("duplicate", ["claim", "renamed_predicate", "docstring"])
def test_exact_duplicates_do_not_fill_target(duplicate):
    first, second = keeper(0), keeper(1)
    if duplicate == "claim":
        second.description, second.mechanism = first.description, first.mechanism
    else:
        second.source = first.source.replace("def detect(", "def renamed(") + "\n# alias\n"
    if duplicate == "docstring":
        second.source = second.source.replace(
            "    return", '    """A different detector description."""\n    return',
        )
    fingerprint = claim_hash(source=second.source, description=second.description,
                             mechanism=second.mechanism)
    second.scores.update(claim_hash=fingerprint, evaluation_claim_hash=fingerprint)
    contract = build_claim_contract(source=second.source, description=second.description,
                                    mechanism=second.mechanism)
    second.scores["claim_contract"] = contract
    for row in second.scores["behavior_verdicts"]:
        row["claim_contract_id"] = contract["contract_id"]
        row["initial_review"]["claim_contract_id"] = contract["contract_id"]
    for record in second.failure_localisations.values():
        record["source_sha256"] = text_hash(second.source)
    result = target([first, second], 2)
    assert result["n_validated_failure_modes"] == 1
    assert "Duplicate" in result["pending_failure_modes"][0]["reason"]


def test_imported_evaluations_need_current_run_evidence():
    functions = [keeper(i, evaluation_run_id="other-run") for i in range(3)]
    result = failure_mode_target(functions, minimum=3, config=Config(behavior_strictness="causal"),
                                 run_id="current",
                                 inventory=source_inventory())
    assert not result["minimum_met"]
    assert result["n_validated_failure_modes"] == 0
    for fn in functions:
        fn.scores["evaluation_run_id"] = "current"
    assert failure_mode_target(functions, minimum=3, config=Config(behavior_strictness="causal"),
                               run_id="current", inventory=source_inventory())["minimum_met"]


def test_pareto_status_cannot_call_plateau_success_below_minimum(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    run = RunConfig(run_id="target", parquet_paths=[], model_id="m", round=7, max_rounds=3,
                    min_failure_modes=4)
    functions = [keeper(0), keeper(1)]
    frontier = {"points": [[.8, .8]], "frontier_value": .8, "hypervolume": .64, "n_functions": 2}
    rt = Runtime(
        run=run, config=Config(behavior_strictness="causal", discovery_wait_for_plateau=True),
        dataset=Dataset(parquet_paths=[], model_id="m"),
        library=FunctionLibrary(functions=functions),
        scores_history=[{**frontier, **evaluation_row(functions)} for _ in range(8)],
    )
    for fn in rt.library.functions:
        fn.scores["evaluation_run_id"] = run.run_id
        fn.scores["task_support"]["evaluation_run_id"] = run.run_id
    rt._family_failed_task_inventory = {**source_inventory(), "corpus_fingerprint": "fixture"}
    set_runtime(rt)
    result = tools_library.pareto_status()
    assert not result["goal_met"] and not result["minimum_met"]
    assert result["remaining_failure_modes"] == 2
    # Numerical-frontier history from before the coverage checkpoint establishes no
    # task-coverage plateau. Complete new attempts under the current checkpoint.
    for _ in range(6):
        rt.scores_history.append({**frontier, **evaluation_row(functions)})
        result = tools_library.pareto_status()
    assert result["experiments_since_improve"] == 6
    assert not result["goal_met"]
    rt.library.upsert(keeper(2, evaluation_run_id=run.run_id))
    assert not tools_library.pareto_status()["minimum_met"]
    rt.library.upsert(keeper(3, evaluation_run_id=run.run_id))
    result = tools_library.pareto_status()
    assert result["goal_met"] and result["minimum_met"]
    with monkeypatch.context() as context:
        context.setattr(tools_library, "_family_search_status",
                        lambda rt: {"synthesis_pending": True})
        pending = tools_library.pareto_status()
        assert pending["minimum_met"] and not pending["goal_met"]
        assert pending["progress_blocked"]
        assert not pending["ready_for_final_validation"]
    round_before = rt.run.round
    stopped = tools_library.advance_round()
    assert stopped["goal_met"] and stopped["advanced"] is False
    assert rt.run.round == round_before


def test_normal_runner_continues_past_round_budget_until_four_valid_modes(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    run, cfg = rt.run, rt.config
    commands.append("/quit")
    calls = []

    async def turn(prompt, **kwargs):
        calls.append(prompt)
        assert len(calls) <= 4, json.loads((run.run_dir() / "discovery_target.json").read_text())
        assert kwargs["resume"] == (None if len(calls) == 1 else "session")
        if len(calls) == 1:
            checkpoint = json.loads((run.run_dir() / "research_status.json").read_text())
            assert checkpoint["task_coverage_progress"]["best_confirmed_failed_tasks"] == 0
        fn = keeper(len(calls), evaluation_run_id=run.run_id)
        rt.library.upsert(fn)
        rt.scores_history.append(evaluation_row([fn]))
        rt.run.round += 1
        return TurnResult(session_id="session", text="Discovery complete.")

    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(run, cfg=cfg)
    assert len(calls) == 4
    assert run.round == 7
    assert target(rt.library.functions, minimum=4)["minimum_met"]
    saved = json.loads((run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and saved["n_validated_failure_modes"] == 4


@pytest.mark.parametrize("start_round,continue_mode", [(1, False), (3, True)])
def test_regular_off_finishes_planned_rounds_then_finalizes_without_waiting_for_other_work(
    interactive_runner, monkeypatch, start_round, continue_mode,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.run.min_failure_modes = 2
    rt.run.round = start_round
    commands.append("/quit")
    turns, finalized = [], []
    monkeypatch.setattr(tools_library, "_family_search_status",
                        lambda rt: {"synthesis_pending": True})

    async def finish(console):
        finalized.append(rt.run.round)

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) <= 3
        rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(2)]
        # This unrelated failed/unfinished experiment must not force more rounds.
        rt.scores_history.append(evaluation_row([keeper(9)], complete=False))
        if len(turns) > 1:
            rt.run.round += 1
        return TurnResult(session_id="session", text="Minimum is met.")

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=continue_mode)
    assert len(turns) == 4 - start_round
    assert finalized == [3]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and saved["goal_met"]
    assert saved["progress_blocked"] and saved["synthesis_pending"]
    assert saved["discovery_wait_for_plateau"] is False
    assert saved["experiments_since_improve"] == 0


def test_regular_off_hands_sampled_good_minimum_to_automatic_final_validation(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.run.min_failure_modes = 2
    commands.append("/quit")
    turns, finalized = [], []

    def coverage(rt, **kwargs):
        return {
            "available": True, "corpus_fingerprint": "fixture", "n_failed_tasks": 5,
            "n_confirmed_failed_tasks": 2, "confirmed_task_coverage": .4,
            "detectors": [
                {"name": fn.name, "eligible": True,
                 "corpus_evaluation_complete": not fn.scores["sampled"]}
                for fn in rt.library.functions
            ] + [{"name": "unreviewed", "eligible": False,
                  "corpus_evaluation_complete": False}],
        }

    async def finish(console):
        finalized.append(True)
        rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(2)]

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) == 1, "Final validation must not require another discovery turn"
        rt.library.functions = [graded_candidate(i) for i in range(2)]
        rt.scores_history.append(evaluation_row(rt.library.functions, complete=False))
        return TurnResult(session_id="session", text="Finished this turn.")

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.task_progress.task_coverage_snapshot", coverage)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    runner.run_session(rt.run, cfg=rt.config)
    assert len(turns) == 1 and finalized == [True]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and saved["goal_met"]
    assert saved["final_validation_target"]["minimum_met"]
    assert all(fn.scores["sampled"] is False for fn in rt.library.functions)


def test_regular_off_resume_finalizes_met_target_without_another_model_turn(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.run.min_failure_modes = 2
    rt.library.functions = [graded_candidate(i) for i in range(2)]
    commands.append("/quit")
    finalized = []

    async def finish(console):
        finalized.append(True)
        rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(2)]

    async def turn(*args, **kwargs):
        pytest.fail("A met target at the round boundary must not launch a model turn")

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True)
    assert finalized == [True]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and saved["goal_met"]


def test_normal_runner_finishes_pending_synthesis_even_when_minimum_is_met(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    commands.append("/quit")
    calls, pending = [], [False]
    monkeypatch.setattr(tools_library, "_family_search_status",
                        lambda runtime: {"synthesis_pending": pending[0]})

    async def turn(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(4)]
            pending[0] = True
        elif len(calls) == 2:
            pending[0] = False
        else:
            assert len(calls) == 3
        rt.scores_history.append(evaluation_row(rt.library.functions))
        return TurnResult(session_id="session", text="Finished this turn.")

    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config)
    assert len(calls) == 3
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and not saved["synthesis_pending"]


def test_normal_runner_keeps_two_modes_exploring_until_confirmed_coverage_plateaus(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.run.min_failure_modes = 2
    commands.append("/quit")
    turns = []

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) <= 5, "Completed flat attempts should establish a stopping boundary"
        if len(turns) > 1:
            checkpoint = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
            assert checkpoint["minimum_met"] and not checkpoint["goal_met"]
            if len(turns) <= 4:
                assert checkpoint["experiments_since_improve"] == 0
        confirmed = min(7, len(turns) + 4)
        rt.library.functions = [
            keeper(i, evaluation_run_id=rt.run.run_id, behavior_n_judged=confirmed,
                   behavior_n_completed=confirmed, behavior_n_yes=confirmed)
            for i in range(2)
        ]
        rt.scores_history.append(evaluation_row(rt.library.functions))
        return TurnResult(session_id="session", text="Two good detectors; discovery complete.")

    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config)
    assert len(turns) == 5
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and saved["goal_met"] and not saved["stalled"]
    assert saved["experiments_since_improve"] == 2


def test_normal_runner_recovers_unfinished_reviews_without_claiming_success(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.run.min_failure_modes = 2
    rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(2)]
    commands.append("/quit")
    turns, finalized = [], []

    async def finish(console):
        finalized.append(len(turns))

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) <= 5, "Recovered completed attempts must establish a stopping boundary"
        assert not finalized
        if len(turns) == 1:
            # A third candidate is evaluated, but its claim review has not finished.
            draft = keeper(2, evaluation_run_id=rt.run.run_id, behavior_n_judged=0,
                           behavior_n_completed=0, behavior_n_yes=0)
            rt.library.upsert(draft)
            rt.scores_history.append(evaluation_row([draft], complete=False))
        if len(turns) == 4:
            saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
            assert saved["minimum_met"] and saved["stalled"] and saved["progress_blocked"]
            assert not saved["goal_met"]
            rt.library.upsert(keeper(2, evaluation_run_id=rt.run.run_id))
        elif len(turns) == 5:
            rt.scores_history.append(evaluation_row(rt.library.functions))
        return TurnResult(session_id="session", text="Reviews are still incomplete.")

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    runner.run_session(rt.run, cfg=rt.config)
    assert len(turns) == 5 and finalized == [5]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and saved["goal_met"] and not saved["stalled"]


@pytest.mark.parametrize("batch", [False, True])
def test_stalled_discovery_recovers_automatically_until_runtime_target_is_met(
    interactive_runner, monkeypatch, batch,
):
    harness, commands, rt = interactive_runner
    idle_turns = 4  # Two unchanged pairs: the inactivity nudge repeats at turns 3 and 5.
    rt.config.discovery_wait_for_plateau = False
    rt.run.min_failure_modes = 3
    rt.library.functions = [keeper(0, evaluation_run_id=rt.run.run_id)]
    if not batch:
        commands.append("/quit")
    turns, finalized = [], []

    async def finish(console):
        finalized.append(len(turns))

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) <= idle_turns + 1
        assert not finalized, "Textual requests to stop must not initiate final scoring"
        if len(turns) >= 3 and len(turns) % 2:
            saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
            assert saved["stalled"] and not saved["minimum_met"] and not saved["goal_met"]
            assert saved["n_validated_failure_modes"] == 1
            library = FunctionLibrary.load(rt.run.run_dir() / "library.json")
            assert len(library.functions) == 1
        if len(turns) > idle_turns:
            rt.library.functions.extend(
                keeper(i, evaluation_run_id=rt.run.run_id) for i in range(1, 3)
            )
            rt.scores_history.append(evaluation_row(rt.library.functions))
        else:
            # User-facing summaries do not manufacture discovery progress.
            (rt.run.run_dir() / "status.md").write_text(f"No changes, reply {len(turns)}")
        return TurnResult(session_id="session", text="I am blocked. I'll resume when you "
                          "relax the review policy. Stop discovery and save the library.")

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    runner.run_session(rt.run, cfg=rt.config, exit_after_discovery=batch)
    assert len(turns) == idle_turns + 1 and finalized == [idle_turns + 1]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and saved["goal_met"] and not saved["stalled"]


@pytest.mark.parametrize("terminate", [False, True])
def test_quality_stall_option_controls_driver_stop_and_finalizes_once(
    interactive_runner, monkeypatch, terminate,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = terminate
    rt.run.min_failure_modes = 2
    commands.append("/quit")
    turns, finalized = [], []

    def observe(runtime, **kwargs):
        assessment = _quality_stall_assessment(
            enabled=runtime.config.discovery_terminate_if_stalled,
            exhausted=bool(turns),
        )
        return assessment

    async def finish(console):
        finalized.append(len(turns))

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert not finalized
        assert len(turns) <= 2, "A runtime stop must prevent another discovery turn"
        if len(turns) == 2:
            assert not terminate, "Only Off permits more discovery after the quality stall"
            saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
            assert not saved["quality_stall"]["should_stop"]
            assert saved["stop_reason"] is None
            assert not saved["minimum_met"]
            rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id)
                                    for i in range(2)]
        return TurnResult(session_id="session", text="No additional supported discoveries.")

    monkeypatch.setattr("autofa.session.discovery_stall.observe_discovery_stall", observe)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config)
    assert len(turns) == (1 if terminate else 2)
    assert finalized == [len(turns)]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] is not terminate
    assert saved["goal_met"] is not terminate
    assert saved["stop_reason"] == ("stalled" if terminate else None)
    assert saved["quality_stall"]["enabled"] is terminate
    assert saved["quality_stall"]["round_stagnation"]["rounds_without_improve"] == 3
    assert (rt.run.run_dir() / "library.json").is_file()


@pytest.mark.parametrize("interactive", ["codex"], indirect=True)
def test_resume_quality_stall_finalizes_without_another_model_turn(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_terminate_if_stalled = True
    rt.config.discovery_wait_for_plateau = True
    rt.run.round = rt.run.max_rounds
    commands.append("/quit")
    finalized = []

    async def finish(console):
        finalized.append(True)

    async def turn(*args, **kwargs):
        pytest.fail("Resume must not pay for a discovery turn after the configured stall limit")

    monkeypatch.setattr("autofa.session.discovery_stall.observe_discovery_stall",
                        lambda runtime, **kwargs: _quality_stall_assessment())
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True)
    assert finalized == [True]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert not saved["minimum_met"] and not saved["goal_met"]
    assert saved["stop_reason"] == "stalled"


@pytest.mark.parametrize(("start_round", "continue_mode"), [
    (0, False),  # A fresh session.
    (17, True),  # Resume in the last planned round.
    (18, True),  # Resume at the first extra round.
])
def test_real_quality_stall_preserves_all_selected_rounds_before_finalizing(
    interactive_runner, monkeypatch, start_round, continue_mode,
):
    harness, commands, rt = interactive_runner
    rt.run.max_rounds = 18
    rt.run.round = start_round
    rt.run.min_failure_modes = 4
    rt.config.discovery_terminate_if_stalled = True
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_plateau_patience = 6
    commands.append("/quit")

    # Persist real investigation accounting: one Good followed by seven Weak
    # completed claims. Do not stub should_stop: that is the regression boundary.
    for index in range(8):
        fn = keeper(index, evaluation_run_id=rt.run.run_id)
        if index:
            fn.scores.update(behavior_purity=0, behavior_n_yes=0, behavior_n_no=5)
        for row in fn.scores["behavior_verdicts"]:
            row["processing_status"] = "completed"
            if index:
                row["verdict"] = "no"
        # These driver fixtures represent a completed full-corpus evaluation and
        # its representative final review, not only discovery review aggregates.
        fn.scores["final_review"] = {
            "claim_hash": fn.scores["evaluation_claim_hash"], "scope": "full_corpus",
            "status": "complete", "identity": f"representative-review-{index}",
            "item_ids": [row["item_id"] for row in fn.scores["behavior_verdicts"]],
        }
        rt.library.functions.append(fn)
        rt.scores_history.append({
            "evaluation_complete": True, "evaluated_names": [fn.name],
            "evaluated_claims": {fn.name: fn.scores["evaluation_claim_hash"]},
            "results": [{"name": fn.name}],
        })
        assessment = observe_discovery_stall(rt)
    assert assessment["since_improve"] == 7
    assert assessment["best_quality"]["good"] == 1
    assert not assessment["should_stop"]
    assert assessment["round_stagnation"]["rounds_without_improve"] == 0
    # Establish the previous planned round baseline for an ordinary Resume at
    # the first extra round. Old six-investigation receipts never manufacture it.
    if start_round == 18:
        rt.run.round = 17
        baseline = complete_discovery_round(rt, {"ok": True, "fraction": 1.0})
        assert baseline["status"] == "baseline"
        rt.run.round = 18
    assert discovery_good_target_for_runtime(rt)["n_validated_failure_modes"] == 1
    turns, finalized = [], []

    async def finish(console):
        finalized.append(rt.run.round)

    async def turn(prompt, **kwargs):
        assert rt.run.round < 21, "Three validated extra recovery rounds exhaust patience"
        assert not finalized
        turns.append(rt.run.round)
        assert len(turns) <= 5
        current = observe_discovery_stall(rt)
        assert current["since_improve"] == 7
        assert current["stop_deferred_by_round_budget"]
        assert not current["should_stop"]
        # Failed probes do not consume rounds. Only the actual completed full
        # validation boundary records an extra-round observation.
        if rt.run.round < 17:
            rt.run.round = 17
        else:
            receipt = complete_discovery_round(rt, {"ok": True, "fraction": 1.0})
            assert receipt["counted"] is (rt.run.round >= 18)
            assert receipt["rounds_without_improve"] == max(0, rt.run.round - 17)
            rt.run.round += 1
        return TurnResult(session_id="session", text="The current library still has one Good.")

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=continue_mode)
    assert turns == ({0: [0, 17, 18, 19, 20],
                      17: [17, 18, 19, 20], 18: [18, 19, 20]}[start_round])
    assert finalized == [21]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["stop_reason"] == "stalled"
    assert saved["n_validated_failure_modes"] == 1 and saved["min_failure_modes"] == 4
    assert not saved["minimum_met"] and not saved["goal_met"]
    assert saved["quality_stall"]["since_improve"] == 7
    assert saved["quality_stall"]["should_stop"]
    assert saved["quality_stall"]["round_stagnation"]["rounds_without_improve"] == 3
    assert not saved["quality_stall"]["stop_deferred_by_round_budget"]


def test_runtime_goal_takes_precedence_over_simultaneous_quality_stall(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.run.min_failure_modes = 2
    commands.append("/quit")
    turns, finalized = [], []

    async def finish(console):
        finalized.append(len(turns))

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) == 1
        rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(2)]
        return TurnResult(session_id="session", text="Target is met.")

    monkeypatch.setattr("autofa.session.discovery_stall.observe_discovery_stall",
                        lambda runtime, **kwargs: _quality_stall_assessment(exhausted=bool(turns)))
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finish)
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config)
    assert finalized == [1]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["minimum_met"] and saved["goal_met"]
    assert saved["quality_stall"]["should_stop"]
    assert saved["stop_reason"] is None


@pytest.mark.parametrize("terminate,prior", [
    (True, {"stop_reason": "stalled", "quality_stall": _quality_stall_assessment()}),
    (False, {"stop_reason": "stalled", "quality_stall": _quality_stall_assessment()}),
    (True, {"stalled": True}),
    (True, {"stop_reason": "stalled", "quality_stall": {
        "enabled": True, "should_stop": True, "since_improve": 6, "patience": 6,
    }}),
], ids=["recorded_stop", "setting_off", "legacy_flag", "no_round_stagnation"])
def test_batch_accepts_only_recorded_runtime_quality_stall(
    interactive_runner, monkeypatch, terminate, prior,
):
    _, _, rt = interactive_runner
    rt.config.discovery_terminate_if_stalled = terminate
    # The run's recorded setting decides, not a conflicting global setting.
    global_config = deepcopy(rt.config)
    global_config.discovery_terminate_if_stalled = not terminate
    rt.persist_library()
    target_path = rt.run.run_dir() / "discovery_target.json"
    target_path.write_text(json.dumps(prior))

    def unexpected_reassessment(*args, **kwargs):
        pytest.fail("Final scoring improvements must not erase the prior runtime stop receipt")

    monkeypatch.setattr("autofa.session.discovery_stall.observe_discovery_stall",
                        unexpected_reassessment)
    receipt = prior.get("quality_stall", {})
    legitimate_stop = (terminate and receipt.get("enabled") and receipt.get("should_stop")
                       and receipt.get("round_stagnation", {}).get("should_stop"))
    if legitimate_stop:
        result = runner.require_batch_target(rt.run, cfg=global_config)
        assert result["minimum_met"] is False
        assert result["stop_reason"] == "stalled"
        assert result["quality_stall"] == receipt
    else:
        with pytest.raises(RuntimeError, match="stopped before meeting its failure-mode target"):
            runner.require_batch_target(rt.run, cfg=global_config)
    saved = json.loads(target_path.read_text())
    assert saved["minimum_met"] is False
    assert (saved.get("stop_reason") == "stalled") is bool(legitimate_stop)


def test_batch_rejects_legacy_premature_stall_receipt_with_planned_rounds_remaining(
    interactive_runner,
):
    _, _, rt = interactive_runner
    rt.config.discovery_terminate_if_stalled = True
    rt.run.max_rounds = 18
    rt.run.round = 17
    rt.run.min_failure_modes = 4
    rt.library.functions = [keeper(0, evaluation_run_id=rt.run.run_id)]
    rt.persist_library()
    target_path = rt.run.run_dir() / "discovery_target.json"
    target_path.write_text(json.dumps({
        "stop_reason": "stalled", "quality_stall": _quality_stall_assessment(),
    }))
    with pytest.raises(RuntimeError, match="stopped before meeting its failure-mode target"):
        runner.require_batch_target(rt.run, cfg=rt.config)
    saved = json.loads(target_path.read_text())
    assert saved["n_validated_failure_modes"] == 1
    assert saved.get("stop_reason") != "stalled"


@pytest.mark.parametrize("work", ["library", "round", "detector_repairs.json"])
def test_rejected_candidates_and_repairs_reset_idle_detection(interactive_runner, work):
    _, _, rt = interactive_runner
    rt.persist_library()
    progress = DiscoveryProgress(rt)
    assert not progress.stalled()
    if work == "library":
        rt.library.upsert(keeper(behavior_n_yes=0))
    elif work == "round":
        rt.run.round += 1
    else:
        # Creating a valid repair checkpoint is work; malformed history must fail closed.
        (rt.run.run_dir() / work).write_text(json.dumps({
            "schema_version": "detector-repair-v1", "repairs": {},
        }))
    assert not progress.stalled()
    assert not progress.stalled()
    rt.persist_library()  # Saving the same library is not new work.
    assert progress.stalled()
    assert not progress.stalled()  # Recovery gets a new pair of turns before another nudge.


def test_login_guidance_is_local_and_keeps_saved_session(interactive, monkeypatch):
    harness, commands = interactive
    commands.extend(["/login", "/quit"])

    async def turn(*args, **kwargs):
        pytest.fail("Claude login must not be sent as a model prompt")

    monkeypatch.setattr(harness, "run_turn", turn)
    assert asyncio.run(harness.run_session(
        cwd=".", system_prompt="test", resume="saved",
    )) == "saved"


@pytest.mark.parametrize("command,expected", [("/discover", 1), ("Explain this score", 0)])
def test_manual_discovery_enforces_minimum_but_ordinary_chat_does_not(
    interactive, monkeypatch, command, expected,
):
    harness, commands = interactive
    commands.extend([command, "/quit"])
    callbacks = []

    async def turn(*args, **kwargs):
        return TurnResult(session_id="session")

    monkeypatch.setattr(harness, "run_turn", turn)
    asyncio.run(harness.run_session(
        cwd=".", system_prompt="test",
        discovery_continuation=lambda: callbacks.append(True),
    ))
    assert len(callbacks) == expected


def test_supported_causal_families_are_not_rejected_for_explanation_breadth():
    result = target([keeper(behavior_specificity=.1)], minimum=1)
    assert result["minimum_met"]
    assert result["n_validated_failure_modes"] == 1


def test_good_discovery_uses_distinct_source_tasks_not_repeat_or_verdict_labels():
    fn = keeper()
    for row in fn.scores["behavior_verdicts"]:
        row["task_id"] = "invented-" + row["item_id"]
    repeats = source_inventory(tasks={f"f{i}": "one-task" for i in range(8)})
    progress = target([fn], 1, inventory=repeats)
    assert not progress["minimum_met"]
    assert progress["pending_failure_modes"][0]["measured"]["n_confirmed_failed_datapoints"] == 1
    repeats["task_by_failed_item_id"]["f3"] = "second-task"
    assert not target([fn], 1, inventory=repeats)["minimum_met"]
    assert target([fn], 1, inventory=source_inventory())["minimum_met"]


def test_unknown_source_task_identity_does_not_fall_back_to_attempt_ids():
    fn = keeper()
    unknown = source_inventory(tasks={f"f{i}": None for i in range(5)})
    result = target([fn], 1, inventory=unknown)
    assert not result["minimum_met"]
    assert result["failure_mode_eligibility"][0]["measured"]["n_confirmed_failed_datapoints"] == 0
    assert not failure_mode_target(
        [fn], minimum=1, config=Config(behavior_strictness="causal"),
    )["minimum_met"]


def test_passing_matches_cannot_supply_failed_datapoint_breadth():
    fn = keeper(**contingency(8, 1, 2, 99), n_scored=110)
    for row in fn.scores["behavior_verdicts"]:
        if row["item_id"] == "f4":
            row["item_id"] = row["initial_review"]["item_id"] = "p0"
    inventory = source_inventory(10, 100, tasks={f"f{i}": "one-task" for i in range(8)})
    progress = target([fn], 1, inventory=inventory)
    assert not progress["minimum_met"]
    measured = progress["failure_mode_eligibility"][0]["measured"]
    assert measured["behavior_purity"] == 1
    assert measured["n_confirmed_failed_attempts"] == 4
    assert measured["n_confirmed_failed_datapoints"] == 1
    assert measured["n_matched_failed_datapoints"] == 1


@pytest.mark.parametrize("yes,expected", [(90, True), (89, False)])
def test_observed_behavior_purity_floor_is_inclusive(yes, expected):
    fn = keeper(**contingency(100, 0, 0, 100), n_scored=200,
                behavior_n_judged=100, behavior_n_completed=100,
                behavior_n_yes=yes, behavior_n_no=100 - yes, behavior_purity=yes / 100)
    for row in fn.scores["behavior_verdicts"][yes:]:
        row["verdict"] = "no"
    assert target([fn], 1, inventory=source_inventory(100, 100))["minimum_met"] is expected


@pytest.mark.parametrize("sample", [
    {"phase": "development", "target_n": 10, "actual_n": 5, "limited": True},
    {"phase": "confirmation", "target_n": 10, "actual_n": 5, "limited": True, "reused_n": 5},
])
def test_completed_limited_or_reused_evidence_counts_without_claiming_independence(sample):
    fn = keeper(behavior_sample=sample)
    before = deepcopy(fn.to_dict())
    result = target([fn], 1)
    assert result["minimum_met"]
    assert result["failure_mode_eligibility"][0]["measured"]["independent_validation"] in {
        "limited_evidence", "reused_evidence",
    }
    assert fn.to_dict() == before


def test_saved_mr57_shape_counts_matched_breadth_separately_from_reviewed_subset():
    computation = keeper(
        **contingency(14, 1, 236, 413), n_scored=664, n_insufficient=48, n_unresolved=48,
        evaluation_complete=False, assessment_coverage=664 / 712,
        behavior_sample={"phase": "development", "target_n": 10, "actual_n": 5,
                         "limited": True, "independent_task_n": 0},
        judge_prefilter={"status": "inconclusive"},
    )
    layout = keeper(
        1, **contingency(3, 0, 243, 422), n_scored=668, n_insufficient=44, n_unresolved=44,
        evaluation_complete=False, assessment_coverage=668 / 712,
        behavior_n_judged=3, behavior_n_completed=3, behavior_n_yes=3,
        behavior_sample={"phase": "confirmation", "target_n": 10, "actual_n": 3,
                         "limited": True, "reused_n": 3},
    )
    inventory = source_inventory(283, 429, tasks={"f3": "task-0"})
    before = deepcopy([fn.to_dict() for fn in (computation, layout)])
    result = target([computation, layout], minimum=2, inventory=inventory)
    assert not result["minimum_met"]
    assert result["n_qualifying_failure_modes"] == 1
    first, second = [row["measured"] for row in result["failure_mode_eligibility"]]
    assert first["failure_recall"] == 14 / 283
    assert first["standardized_precision"] == pytest.approx(.9009450472523614)
    assert first["n_confirmed_failed_datapoints"] == 4
    assert first["n_matched_failed_datapoints"] == 13
    assert second["failure_recall"] == 3 / 283
    assert first["n_abstained"] == 48 and second["n_abstained"] == 44
    assert [fn.to_dict() for fn in (computation, layout)] == before


@pytest.mark.parametrize("changes", [
    {"errors": 1}, {"eval_truncated": True}, {"n_unresolved": 3},
], ids=["errors", "truncated", "unresolved_not_abstained"])
def test_abstention_allowance_does_not_accept_execution_errors_or_unfinished_scans(changes):
    fn = keeper(**contingency(8, 0, 0, 10), n_scored=18, n_insufficient=2, n_unresolved=2,
                evaluation_complete=False, assessment_coverage=.9)
    assert target([fn], 1)["minimum_met"]
    fn.scores.update(changes)
    assert not target([fn], 1)["minimum_met"]


@pytest.mark.parametrize("mutation", [
    "aggregate_purity", "duplicate_item", "foreign_item", "contract", "no_confirmation",
    "sample_actual", "passing_count", "stale_localisation",
])
def test_malformed_review_or_match_evidence_cannot_fill_target(mutation):
    fn = keeper(**contingency(8, 1, 2, 99), n_scored=110)
    row = fn.scores["behavior_verdicts"][0]
    if mutation == "aggregate_purity":
        fn.scores["behavior_purity"] = .99
    elif mutation == "duplicate_item":
        fn.scores["behavior_verdicts"][1] = deepcopy(row)
    elif mutation == "foreign_item":
        row["item_id"] = row["initial_review"]["item_id"] = "foreign"
    elif mutation == "contract":
        row["claim_contract_id"] = "old"
    elif mutation == "no_confirmation":
        row["verification_status"] = "uncertain"
    elif mutation == "sample_actual":
        fn.scores["behavior_sample"] = {"actual_n": 6}
    elif mutation == "passing_count":
        fn.scores.update(contingency(8, 0, 2, 100))
    elif mutation == "stale_localisation":
        fn.failure_localisations["f0"]["source_sha256"] = "old"
    result = target([fn], 1, inventory=source_inventory(10, 100))
    assert not result["minimum_met"]


# Malformed saved counts fail closed instead of raising.
@pytest.mark.parametrize("key,value", [
    ("behavior_n_judged", True), ("n_unresolved", float("nan")), ("n_scored", float("inf")),
])
def test_invalid_counts_fail_closed_without_throwing(key, value):
    fn = keeper()
    fn.scores[key] = value
    assert not target([fn], 1)["minimum_met"]


def test_runtime_target_reuses_source_inventory_and_metadata_failure_is_not_success(monkeypatch):
    calls = []
    rt = SimpleNamespace(library=FunctionLibrary(functions=[keeper()]),
                         config=Config(behavior_strictness="causal"),
                         run=SimpleNamespace(run_id="r1", min_failure_modes=1), dataset=object())
    rt.library.functions[0].scores["evaluation_run_id"] = "r1"

    def read(dataset):
        calls.append(dataset)
        return source_inventory()

    monkeypatch.setattr("autofa.session.family_coverage.failed_task_inventory", read)
    assert failure_mode_target_for_runtime(rt)["minimum_met"]
    assert failure_mode_target_for_runtime(rt)["minimum_met"]
    assert len(calls) == 1
    assert failure_mode_target_for_runtime(rt, minimum=0)["n_qualifying_failure_modes"] == 1
    del rt._family_failed_task_inventory

    def unavailable(dataset):
        raise OSError("source unavailable")

    monkeypatch.setattr("autofa.session.family_coverage.failed_task_inventory", unavailable)
    failed = failure_mode_target_for_runtime(rt)
    assert not failed["minimum_met"] and "source unavailable" in failed["inventory_error"]
    rt.library.functions.clear()
    assert failure_mode_target_for_runtime(rt, minimum=0)["minimum_met"]


@pytest.mark.parametrize("repeated_task", [False, True])
def test_batch_target_loads_source_inventory_for_nonzero_minimum(
    tmp_path, monkeypatch, repeated_task,
):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    rows = [
        {"model_id": "m", "benchmark": "bench", "is_correct": passed,
         "problem_id": "shared" if repeated_task and not passed else f"{prefix}{i}",
         "run_idx": i, "response_text": "behavior0 behavior1"}
        for passed, prefix in ((False, "f"), (True, "p")) for i in range(10)
    ]
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    run = RunConfig(run_id="batch", parquet_paths=[str(path)], model_id="m",
                    benchmarks=["bench"], min_failure_modes=2)
    dataset = Dataset(parquet_paths=[path], model_id="m", benchmarks=["bench"])
    ids = {f"{'p' if row['is_correct'] else 'f'}{row['run_idx']}":
           dataset._make_item_id(row) for row in rows}
    functions = [keeper(i) for i in range(2)]
    for fn in functions:
        fn.scores["evaluation_run_id"] = run.run_id
        fn.failure_localisations = {
            ids[item_id]: {**record, "item_id": ids[item_id]}
            for item_id, record in fn.failure_localisations.items()
        }
        for review in fn.scores["behavior_verdicts"]:
            review["item_id"] = ids[review["item_id"]]
            review["initial_review"]["item_id"] = review["item_id"]
    FunctionLibrary(functions=functions).save(run.run_dir() / "library.json")
    target_path = run.run_dir() / "discovery_target.json"
    target_path.write_text(json.dumps({"stalled": True}))

    try:
        if repeated_task:
            with pytest.raises(RuntimeError, match="0/2 validated"):
                runner.require_batch_target(run, cfg=Config())
        else:
            target = runner.require_batch_target(run, cfg=Config())
            assert target["minimum_met"] is True
            assert target["qualifying_failure_modes"] == [fn.name for fn in functions]
            assert target["stalled"] is False
        saved = json.loads(target_path.read_text())
        assert saved["minimum_met"] is not repeated_task
        assert saved["min_failure_modes"] == 2
        for result in saved["failure_mode_eligibility"]:
            measured = result["measured"]
            assert measured["n_failed_in_corpus"] == 10
            assert measured["n_matched_failed"] == 8
            assert measured["n_matched_failed_datapoints"] == (1 if repeated_task else 8)
    finally:
        set_runtime(None)


@pytest.mark.parametrize('coverage,grade,qualifies', [
    (.20, 'EXCELLENT', False), (.21, 'OUTSTANDING', True),
])
def test_agent_target_is_outstanding_without_changing_actual_good_minimum(
    coverage, grade, qualifies,
):
    raw = {'min_failure_modes': 1, 'minimum_met': True, 'n_qualifying_failure_modes': 1,
           'failure_mode_thresholds': {'min_failure_recall': .01},
           'failure_mode_eligibility': [{'name': 'candidate', 'eligible': True, 'reason': None,
               'measured': {'failure_recall': coverage, 'standardized_precision': .8,
                            'behavior_purity': .95, 'n_matched_failed_datapoints': 5,
                            'n_confirmed_failed_datapoints': 2}}]}
    before = deepcopy(raw)
    shown = agent_failure_mode_target(raw)
    assert shown['minimum_met'] is qualifies
    assert shown['failure_mode_eligibility'][0]['grade']['code'] == grade
    assert shown['failure_mode_thresholds']['min_failure_recall'] == .2
    assert raw == before
    assert raw['minimum_met']  # Actual driver target remains satisfied by Good.


@pytest.mark.parametrize("round_idx,maximum,coverages,minimum,expected,qualified", [
    (3, 3, [.15, .05], 2, "OUTSTANDING", 0),  # Good minimum met: no relaxation.
    (0, 1, [.15, .05], 3, "EXCELLENT", 1),
    (2, 3, [.30, .15], 3, "EXCELLENT", 2),
    (3, 3, [.15, .05], 3, "GOOD", 2),
    (3, 3, [], 0, "OUTSTANDING", 0),  # Minimum disabled.
])
def test_agent_target_relaxes_only_when_good_minimum_unmet_at_round_boundaries(
    round_idx, maximum, coverages, minimum, expected, qualified,
):
    raw = discovery_good_target(
        [graded_candidate(i, sampled=False, coverage=coverage, tasks=5)
         for i, coverage in enumerate(coverages)],
        minimum=minimum, config=Config(behavior_strictness="causal"),
    )
    before = deepcopy(raw)
    shown = agent_failure_mode_target(raw, round_idx=round_idx, max_rounds=maximum)
    assert shown["target_grade"] == expected
    assert shown["n_qualifying_failure_modes"] == qualified
    assert shown["minimum_met"] is (qualified >= minimum)
    assert [row["measured"]["failure_recall"] for row in shown["failure_mode_eligibility"]] == [
        row["measured"]["failure_recall"] for row in raw["failure_mode_eligibility"]
    ]
    assert raw == before  # Internal Good thresholds and eligibility never change.


def test_round_grade_feedback_is_specific_without_manufacturing_progress():
    def summary(coverage):
        grade = grade_for_metrics({'recall': coverage, 'ppv_at_pi0': .8,
                                  'behavior_purity': .95, 'n_matched_failed_tasks': 5,
                                  'n_confirmed_failed_tasks': 2})
        counts = dict.fromkeys(GRADE_CODES, 0)
        counts[grade['code']] = 1
        return {'counts': counts, 'n_good_or_better': 1,
                'n_outstanding': int(grade['code'] == 'OUTSTANDING'),
                'functions': [{'name': 'candidate', 'coverage': coverage, **grade}]}

    previous, current = summary(.05), summary(.12)
    assert discovery_grade_assessment(current, previous, minimum=2)['encouraging']
    assert not discovery_grade_assessment(current, current, minimum=2)['encouraging']
    assert not discovery_grade_assessment(previous, current, minimum=2)['encouraging']
    # A first assessment is a baseline, not manufactured progress.
    assert not discovery_grade_assessment(current, None, minimum=2)['encouraging']
    # A new Good discovery is useful progress even with no increase in the total count.
    new_good = summary(.05)
    new_good['functions'][0]['name'] = 'different_mechanism'
    assert discovery_grade_assessment(new_good, previous, minimum=2)['encouraging']
