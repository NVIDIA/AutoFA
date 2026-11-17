# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery recovery and live round budgets reach agents without changing saved progress."""

import asyncio
import json
from copy import deepcopy

import pytest

from autofa.mcp import runtime as runtime_module
from autofa.mcp import server, tools_eval, tools_library
from autofa.session import discovery_stall
from autofa.session.discovery_target import discovery_grade_feedback
from autofa.session.family_discovery import (
    STATE_FILE,
    family_discovery,
    initialize_family_discovery,
)

from support.discovery import add, candidate, graded_candidate
from support.discovery import runtime as runtime
from support.families import complete_small_corpus_round, evaluated_family, record
from support.families import rt as rt
from support.repair_review import _runtime


@pytest.fixture
def active_runtime(runtime, monkeypatch):
    monkeypatch.setattr(runtime_module, "_RUNTIME", runtime)
    return runtime


def stagnate(rt, count=3):
    for index in range(count):
        add(rt, candidate(f"weak_{index}"))


def invoke(tool, *, harness="claude", args=None):
    result = asyncio.run(server.invoke_tool(tool, args or {}))
    if harness == "codex":
        response = server.codex_tool_response(result)
        assert response["success"]
        return json.loads(response["contentItems"][0]["text"])
    return json.loads(result["content"][0]["text"])


def invoke_ok(tool, args=None):
    payload = invoke(tool, args=args)
    assert payload["ok"], payload
    return payload


def forbid_detector_work(monkeypatch, message):
    def forbidden(*args, **kwargs):
        pytest.fail(message)

    monkeypatch.setattr(tools_eval, "evaluate_functions", forbidden)
    monkeypatch.setattr(tools_eval, "_judge_function", forbidden)


# Recovery guidance is attached to completed tool responses for both harnesses.


@pytest.mark.parametrize("harness,tool", [
    ("claude", "submit_function"), ("codex", "pareto_status"),
])
def test_current_recovery_reaches_successful_tool_responses(
    active_runtime, monkeypatch, harness, tool,
):
    rt = active_runtime
    stagnate(rt)
    original = {"ok": True, "name": "saved_detector", "score": .3}
    calls = []
    monkeypatch.setitem(server.TOOL_CALLABLES, tool,
                        lambda: calls.append(tool) or original)
    forbid_detector_work(monkeypatch, "Recovery guidance must not execute a detector or judge")
    result = invoke(tool, harness=harness)
    assignment = result["discovery_recovery"]["assignment"]
    assert assignment["detector_name"] == "weak_2"
    assert discovery_stall.observe_discovery_stall(rt)["recovery_assignment"] == assignment
    assert {key: result[key] for key in original} == original
    assert "discovery_recovery" not in original
    assert calls == [tool]
    # The committed operation receipt keeps the original result, without advice.
    for path in (rt.run.run_dir() / "operations").glob("*.json"):
        receipt = json.loads(path.read_text())
        assert receipt["state"] == "complete"
        assert receipt["result"] == original
    assert discovery_stall.observe_discovery_stall(rt)["completed_now"] == 0


@pytest.mark.parametrize("count,enabled", [(2, True), (3, False)])
def test_recovery_requires_enabled_investigation_trigger(
    active_runtime, monkeypatch, count, enabled,
):
    rt = active_runtime
    rt.config.discovery_terminate_if_stalled = enabled
    stagnate(rt, count)
    original = {"ok": True}
    monkeypatch.setitem(server.TOOL_CALLABLES, "submit_function", lambda: original)
    assert invoke("submit_function") == original


@pytest.mark.parametrize("tool,result", [
    ("pareto_status", {"ok": True, "goal_met": True}),
    ("submit_function", {"ok": False, "error": "Original validation failure"}),
    ("sample_items", {"ok": True, "items": []}),
])
def test_runtime_finalization_errors_and_read_tools_do_not_trigger_recovery(
    active_runtime, monkeypatch, tool, result,
):
    stagnate(active_runtime)
    monkeypatch.setitem(server.TOOL_CALLABLES, tool, lambda: result)

    def forbidden(*args, **kwargs):
        pytest.fail("This response should not observe or change discovery progress")

    monkeypatch.setattr(discovery_stall, "observe_discovery_stall", forbidden)
    assert invoke(tool) == result


def test_advisory_failure_cannot_replace_committed_success(active_runtime, monkeypatch):
    original = {"ok": True, "name": "committed_detector"}
    monkeypatch.setitem(server.TOOL_CALLABLES, "submit_function", lambda: original)

    def fail(*args, **kwargs):
        raise OSError("unavailable checkpoint")

    monkeypatch.setattr(discovery_stall, "observe_discovery_stall", fail)
    assert invoke("submit_function") == original
    receipt = json.loads(next((active_runtime.run.run_dir() / "operations").glob("*.json"))
                         .read_text())
    assert receipt["state"] == "complete"
    assert receipt["result"] == original


# Real family assessments settle recovery without another survey or paid review.


def pending_recovery(rt, monkeypatch):
    forbid_detector_work(
        monkeypatch, "Completing an assessment must not evaluate or judge another detector",
    )
    for index in range(3):
        fn = candidate(f"rejected_{index}")
        fn.scores["evaluation_run_id"] = rt.run.run_id
        add(rt, fn)
    assert record()["ok"]
    fn = evaluated_family(rt)
    status = discovery_stall.observe_discovery_stall(rt)
    assert status["completed_attempts"] == 3
    assignment = status["recovery_assignment"]
    assert assignment["detector_name"] == fn.name
    return fn, assignment


def assess(fn):
    result = family_discovery(
        "assess_family", family_name=fn.name,
        assessment="Inspected both rejected calls and later response: the required output "
                   "remains missing. The corrected passing case is a boundary.",
    )
    assert result["ok"]
    assert result["assessment"]["outcome"] == "assessed"
    return result


def test_real_assessment_closes_recovery_once_without_new_tasks_or_attempts(rt, monkeypatch):
    fn, original = pending_recovery(rt, monkeypatch)
    result = assess(fn)
    path = rt.run.run_dir() / STATE_FILE
    saved = json.loads(path.read_text())
    finished = discovery_stall.observe_discovery_stall(rt)
    assignment = finished["recovery_assignment"]
    assert assignment["id"] != original["id"]
    assert assignment["previous_result"]["outcome"] == "assessed"
    assert (finished["completed_attempts"], finished["completed_now"]) == (3, 0)
    assert finished["since_improve"] == 3
    assert json.loads(path.read_text()) == saved

    # Status, Resume, and a round change cannot replay the same recovery outcome,
    # create another comparison, or consume investigation allowance.
    assert family_discovery("status")["completed_comparisons"] == result["completed_comparisons"]
    assert discovery_stall.observe_discovery_stall(
        rt, initialize=True)["recovery_assignment"] == assignment
    rt.run.round += 1
    initialize_family_discovery(rt)
    repeated = discovery_stall.observe_discovery_stall(rt)
    assert repeated["recovery_assignment"] == assignment
    assert (repeated["completed_attempts"], repeated["completed_now"]) == (3, 0)
    checkpoint = json.loads((rt.run.run_dir() / "discovery_stall.json").read_text())
    assert len(checkpoint["recovery_history"]) == 1
    families = json.loads(path.read_text())
    assert sum(len(batch.get("comparisons", []))
               for batch in [*families["history"], families]) == 1


@pytest.mark.parametrize("change", ["source", "corpus"])
def test_saved_assessment_does_not_settle_recovery_after_losing_freshness(rt, monkeypatch, change):
    fn, original = pending_recovery(rt, monkeypatch)
    assess(fn)
    if change == "source":
        fn.source = "def rejected_required_call(item): return True"
    else:
        rt._family_failed_task_inventory["corpus_fingerprint"] = "changed"
    result = discovery_stall.observe_discovery_stall(rt)
    assignment = result["recovery_assignment"]
    if change == "corpus":
        assert assignment["previous_result"]["outcome"] == "superseded"
    else:
        assert assignment == original
    assert (result["completed_attempts"], result["completed_now"]) == (3, 0)


# The agent sees the live round budget without changing stopping or seed state.


def _advance_checked_round(monkeypatch):
    # These fixtures already supply measured grades. Exercise the required
    # closing feedback exchange without replacing them with new predictions.
    monkeypatch.setattr("autofa.session.round_validation.validate_round",
                        lambda rt: {"ok": True, "message": "Budget fixture checked."})
    pending = invoke_ok("advance_round")
    assert pending["refinement_pending"] and not pending["advanced"]
    return invoke_ok("advance_round", {
        "validation_response": "Retain the checked detectors and their measured grades.",
    })


def test_mcp_round_transition_cannot_overrun_a_completed_driver_target(tmp_path, monkeypatch):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.run.round, rt.run.max_rounds = 8, 8
    before = deepcopy(vars(rt.run))
    monkeypatch.setattr(tools_library, "_pareto_status", lambda: {
        "ok": True, "goal_met": True, "round": 8, "max_rounds": 8,
        "minimum_met": True, "n_validated_failure_modes": 2,
        "ready_for_final_validation": False, "final_validation_guidance": None,
        "final_validation_target": {"minimum_met": False},
    })
    response = invoke_ok("advance_round")
    assert (response["advanced"], response["goal_met"], response["round"]) == (False, True, 8)
    assert vars(rt.run) == before


def test_status_budget_uses_live_run_without_advancing_search(tmp_path, monkeypatch):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.run.round, rt.run.max_rounds = 10, 8
    rt.config.max_rounds = 20  # Existing runs retain their selected budget.
    rt.run.exploration_base_seed = 42
    rt.run.exploration_sequence_index = 19
    rt.run.exploration_seed = 123
    before = deepcopy((vars(rt.run), rt.library.to_dict(), rt.scores_history))

    response = invoke_ok("pareto_status")
    repeated = invoke_ok("pareto_status")

    budget = response["budget_feedback"]
    assert (budget["max_rounds"], budget["rounds_remaining"], budget["phase"]) == (8, 0, "overrun")
    assert repeated["budget_feedback"] == budget
    assert repeated["experiments_since_improve"] == response["experiments_since_improve"]
    assert (vars(rt.run), rt.library.to_dict(), rt.scores_history) == before
    # A warning past the limit does not authorize completion on its own.
    assert not response["minimum_met"] and not response["goal_met"]


def test_round_transition_reports_tapered_target_and_all_grade_counts(tmp_path, monkeypatch):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.run.round, rt.run.max_rounds, rt.run.min_failure_modes = 1, 3, 2
    rt.run.exploration_base_seed = 42
    rt.run.exploration_sequence_index = 31  # Continue need not start at seed index zero.
    fn = graded_candidate(0, sampled=False, coverage=.15, tasks=5)
    rt.library.functions = [fn]
    assert invoke_ok("pareto_status")["target_grade"] == "OUTSTANDING"
    complete_small_corpus_round(rt)
    scores = deepcopy(fn.scores)

    last = _advance_checked_round(monkeypatch)
    assert last["round"] == 2  # Displayed round 3 of 3 has just started.
    assert last["target_grade"] == "EXCELLENT"
    assert last["n_qualifying_failure_modes"] == 1
    assert last["agent_grade_feedback"]["summary"]["counts"]["EXCELLENT"] == 1
    assert rt.run.exploration_sequence_index == 32
    assert fn.scores == scores
    # Resume's first assessment uses the saved live round, not an initial seed index.
    resumed = discovery_grade_feedback(rt)
    assert (resumed["target_grade"], resumed["remaining_target"]) == ("EXCELLENT", 1)

    complete_small_corpus_round(rt)
    extension = _advance_checked_round(monkeypatch)
    assert extension["round"] == 3
    assert extension["target_grade"] == "GOOD"
    assert invoke_ok("pareto_status")["target_grade"] == "GOOD"


@pytest.mark.parametrize("round_idx", [2, 3])
def test_status_and_resume_do_not_relax_when_good_minimum_is_already_met(
    tmp_path, monkeypatch, round_idx,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.run.round, rt.run.max_rounds, rt.run.min_failure_modes = round_idx, 3, 2
    rt.library.functions = [
        graded_candidate(i, sampled=False, coverage=coverage, tasks=5)
        for i, coverage in enumerate((.15, .05))
    ]
    internal = tools_library._pareto_status()
    assert internal["minimum_met"]
    assert internal["n_qualifying_failure_modes"] == 2

    status = invoke_ok("pareto_status")
    assert status["target_grade"] == discovery_grade_feedback(rt)["target_grade"] == "OUTSTANDING"
    assert status["n_qualifying_failure_modes"] == 0
    assert status["goal_met"] is (round_idx >= 3)
