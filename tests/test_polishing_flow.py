# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A full checkpoint opens repairs of the admitted library, not another discovery loop."""

import asyncio
import json
from copy import deepcopy

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import runtime, server, tools_library
from autofa.session.discovery_target import (
    agent_failure_mode_target,
    agent_target_grade_for_runtime,
)
from autofa.session.family_discovery import (
    STATE_FILE,
    family_discovery,
    family_submission_guard,
)
from autofa.session.polishing import prepare_polishing
from autofa.session.polishing_guard import (
    active_polishing,
    polishing_repair_id,
    polishing_tool_guard,
)
from autofa.session.round_validation import active_round_refinement
from autofa.util.atomic import atomic_write_json

from support.evaluation import write_phase
from support.families import _complete_retained_batch
from support.families import batch_runtime as batch_runtime
from support.families import polishing_round as polishing_round


def test_full_checkpoint_keeps_frozen_library_and_reuses_feedback(polishing_round):
    rt, calls = polishing_round
    first = tools_library.advance_round()
    assert first["refinement_pending"] and not first["advanced"]
    phase = active_round_refinement(rt)
    assert phase["mode"] == "polishing" and phase["target_grade"] == "GOOD"
    assert phase["frozen_names"] == [f.name for f in rt.library.functions]
    assert "controls" in phase
    assert "controls" not in first["round_validation"]["refinement"]
    assert len(calls) == 1
    repeated = tools_library.advance_round()
    assert repeated["round_validation"] == first["round_validation"]
    assert len(calls) == 1
    status = tools_library.pareto_status()
    assert status["family_discovery"]["phase"] == "polishing"
    assert status["target_grade"] == "GOOD"


def test_polishing_closure_verifies_then_keeps_same_detectors_if_minimum_unmet(polishing_round):
    rt, calls = polishing_round
    tools_library.advance_round()
    original = deepcopy(active_round_refinement(rt))
    closed = tools_library.advance_round(validation_response="Checked the failed cases; retain.")
    assert closed["advanced"] and closed["phase"] == "polishing"
    assert len(calls) == 2
    assert rt.run.round == 2
    assert not closed["goal_met"] and closed["refinement_pending"]
    phase = active_round_refinement(rt)
    assert phase["frozen_names"] == original["frozen_names"]
    assert phase["baseline_grades"] == original["baseline_grades"]
    assert phase["polishing_id"] != original["polishing_id"]
    # No family-comparison quota or fresh corpus scan is required for its feedback.
    resumed = tools_library.advance_round()
    assert resumed["refinement_pending"] and len(calls) == 2


@pytest.mark.parametrize("stalled", [False, True])
def test_polishing_stops_after_verified_minimum_or_configured_stall(
    polishing_round, monkeypatch, stalled,
):
    rt, calls = polishing_round
    tools_library.advance_round()
    monkeypatch.setattr(tools_library, "_pareto_status", lambda: {
        "goal_met": not stalled, "ready_for_final_validation": False,
        "final_validation_guidance": "The full scan is current.",
    })
    if stalled:
        monkeypatch.setattr("autofa.session.discovery_stall.complete_discovery_round",
                            lambda *a: {"should_stop": True, "reason": "No validated improvement."})
    closed = tools_library.advance_round(validation_response="Repaired the measured blockers.")
    assert closed["advanced"] and closed["goal_met"] is not stalled
    assert closed["stop_reason"] == ("stalled" if stalled else None)
    assert not closed["refinement_pending"]
    assert active_round_refinement(rt) is None
    assert len(calls) == 2


def test_interrupted_polishing_verification_resumes_saved_response(polishing_round, monkeypatch):
    rt, calls = polishing_round
    tools_library.advance_round()
    original = deepcopy(active_round_refinement(rt))

    def interrupted(**kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr("autofa.mcp.tools_eval.final_evaluate_functions", interrupted)
    with pytest.raises(asyncio.CancelledError):
        tools_library.advance_round(validation_response="Verified the regression cases.")
    phase = active_round_refinement(rt)
    assert phase["status"] == "validating_response"
    assert phase["frozen_names"] == original["frozen_names"]
    assert phase["controls"] == original["controls"]
    assert rt.run.round == 1
    monkeypatch.setattr("autofa.mcp.tools_eval.final_evaluate_functions",
                        lambda **kwargs: {"ok": True})
    assert tools_library.advance_round()["advanced"]
    assert rt.run.round == 2 and len(calls) == 1


def test_polishing_capacity_block_preserves_incomplete_result_without_reopening(
    polishing_round, monkeypatch,
):
    rt, calls = polishing_round
    tools_library.advance_round()
    frozen = deepcopy(active_round_refinement(rt)["frozen_names"])
    blocked = {"should_stop": True, "stop_reason": "review_capacity_exhausted",
               "review_capacity_block": {"status": "blocked"},
               "reason": "Required evidence remains unread; preserve consumed review allowances.",
               "available": False, "counted": False}
    monkeypatch.setattr("autofa.session.discovery_stall.complete_discovery_round",
                        lambda *a: blocked)
    closed = tools_library.advance_round(validation_response="Retain the incomplete review.")
    assert closed["advanced"] and not closed["goal_met"]
    assert closed["stop_reason"] == "review_capacity_exhausted"
    assert closed["quality_stall"]["available"] is False
    assert not closed["refinement_pending"] and active_round_refinement(rt) is None
    assert [fn.name for fn in rt.library.functions] == frozen
    assert len(calls) == 2


def test_untested_edit_gets_diagnostic_feedback_before_full_verification(
    polishing_round, monkeypatch,
):
    rt, calls = polishing_round
    tools_library.advance_round()
    fn = rt.library.functions[0]
    fn.description += " Repaired scope."
    checks = []
    monkeypatch.setattr("autofa.session.polishing.unchecked_revisions",
                        lambda rt, state: [fn.name] if not checks else [])

    def diagnostics(**kwargs):
        checks.append(kwargs)
        return {"ok": True, "polishing_feedback": {fn.name: {"n": 900}}}

    monkeypatch.setattr("autofa.mcp.tools_eval.evaluate_functions", diagnostics)
    result = tools_library.advance_round(validation_response="Ready to finish.")
    assert not result["advanced"]
    assert checks == [{"names": [fn.name]}] and len(calls) == 1
    assert active_round_refinement(rt)["status"] == "awaiting_response"
    assert tools_library.advance_round(validation_response="Inspected the new checks.")["advanced"]
    assert len(calls) == 2


def test_unchecked_revision_preserves_controls_upgraded_during_evaluation(
    polishing_round, monkeypatch,
):
    """The closing caller must not overwrite the evaluator's migrated receipt."""
    rt, calls = polishing_round
    tools_library.advance_round()
    fn = rt.library.functions[0]
    fn.description += " Repaired scope."
    path = rt.run.run_dir() / "round_validation.json"
    before = json.loads(path.read_text())
    upgraded = {
        **before["refinement"]["controls"][fn.name],
        "unflagged_success_pool": [{"item_id": "new-control", "task_id": "new-task",
                                    "baseline_label": False}],
        "controls_fingerprint": "upgraded-metadata-receipt",
    }
    monkeypatch.setattr("autofa.session.polishing.unchecked_revisions",
                        lambda rt, state: [fn.name])

    def diagnostics(**kwargs):
        assert kwargs == {"names": [fn.name]}
        receipt = json.loads(path.read_text())
        receipt["refinement"]["controls"][fn.name] = upgraded
        receipt["metadata_upgrade_complete"] = True
        path.write_text(json.dumps(receipt))
        return {"ok": True, "polishing_feedback": {fn.name: {"n": 900}}}

    monkeypatch.setattr("autofa.mcp.tools_eval.evaluate_functions", diagnostics)
    result = tools_library.advance_round(validation_response="Ready to finish.")
    assert not result["advanced"]
    saved = json.loads(path.read_text())
    assert saved["refinement"]["controls"][fn.name] == upgraded
    assert saved["metadata_upgrade_complete"] is True
    assert saved["refinement"]["status"] == "awaiting_response"
    assert len(calls) == 1  # Only the initial validation, never an extra census.


def test_skipped_validation_cannot_erase_saved_controls(batch_runtime, monkeypatch):
    rt = batch_runtime
    _complete_retained_batch(rt)
    fn = rt.library.functions[0]
    previous = {
        "mode": "polishing", "frozen_names": [fn.name],
        "baseline_grades": {"functions": [{"name": fn.name, "code": "GOOD"}]},
        "controls": {fn.name: {"available": True, "item_ids": ["all-prior-cases"]}},
    }
    monkeypatch.setattr("autofa.session.validation_controls.collect_full_validation_controls",
                        lambda *a, **kw: {"available": False, "reason": "Skipped full scan."})
    out = prepare_polishing(rt, before={}, previous_scores={}, previous=previous)
    assert out["controls"][fn.name]["item_ids"] == ["all-prior-cases"]
    assert out["controls"][fn.name]["refresh_limitation"] == "Skipped full scan."
    assert out["baseline_grades"] == previous["baseline_grades"]
    assert json.loads(json.dumps(out))["frozen_names"] == [fn.name]


def test_resume_between_cursor_and_receipt_keeps_polishing_without_rescoring(polishing_round):
    rt, calls = polishing_round
    tools_library.advance_round()
    previous = deepcopy(active_round_refinement(rt))
    assert tools_library.advance_round(validation_response="Retained after checks.")["advanced"]
    path = rt.run.run_dir() / "round_validation.json"
    receipt = json.loads(path.read_text())
    receipt["refinement"] = {**previous, "status": "complete"}
    path.write_text(json.dumps(receipt))
    result = tools_library.advance_round()
    assert result["refinement_pending"] and not result["advanced"]
    assert len(calls) == 2
    state = active_round_refinement(rt)
    assert state["round"] == rt.run.round == 2
    assert state["frozen_names"] == previous["frozen_names"]
    assert "transition_pending" not in state


def test_provisional_good_does_not_claim_completion_before_polishing_closes(
    polishing_round, monkeypatch,
):
    rt, _ = polishing_round
    tools_library.advance_round()
    assert tools_library.advance_round(validation_response="Retain pending candidates.")["advanced"]
    from autofa.session.discovery_target import discovery_good_target_for_runtime

    target = discovery_good_target_for_runtime(rt)
    monkeypatch.setattr("autofa.session.discovery_target.discovery_good_target_for_runtime",
                        lambda rt: {**target, "minimum_met": True})
    status = tools_library._pareto_status()
    assert status["minimum_met"]
    assert not status["goal_met"]
    assert not status["ready_for_final_validation"]
    assert active_round_refinement(rt)["status"] == "awaiting_response"


# Closing validation repairs an admitted library without reopening discovery.


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    value = runtime.Runtime(
        dataset=Dataset(parquet_paths=[]),
        library=FunctionLibrary(functions=[FeatureFunction(
            name="existing", description="Claim", mechanism="Mechanism",
            source="def detect(item): return False",
        )]),
        config=Config(), run=RunConfig(run_id="polishing-guard", model_id="m", round=2,
                                     max_rounds=3, min_failure_modes=1, parquet_paths=[]),
    )
    monkeypatch.setattr(runtime, "_RUNTIME", value)
    write_phase(value)
    return value


# One case per guard rule: detector outside the admitted library, family mutation,
# split, evaluation scope and library-wide replacement.
@pytest.mark.parametrize("tool,args", [
    ("submit_function", {"name": "new_detector"}),
    ("family_discovery", {"action": "record_family"}),
    ("plan_function_repair", {"name": "existing", "action": "split_detector"}),
    ("evaluate_functions", {"names": ["existing", "new_detector"]}),
    ("library_load", {"path": "/library.json"}),
])
def test_closing_phase_blocks_new_discovery_mutations(rt, tool, args):
    before = rt.library.to_dict()
    response = polishing_tool_guard(rt, tool, args)
    assert response["ok"] is False and response["phase"] == "polishing"
    assert response["target_grade"] == "GOOD"
    assert rt.library.to_dict() == before


# Reads such as get_item take the same path as family_discovery status.
@pytest.mark.parametrize("tool,args", [
    ("restore_function_version", {"name": "existing"}),
    ("plan_function_repair", {"name": "existing", "action": "refine_detector"}),
    ("evaluate_functions", {}),
    ("family_discovery", {"action": "status"}),
])
def test_repair_restore_and_evidence_operations_remain_available(rt, tool, args):
    assert polishing_tool_guard(rt, tool, args) is None


@pytest.mark.parametrize("changes", [
    {"mode": "refinement"}, {"status": "complete"}, {"run_id": "previous"},
])
def test_inactive_or_earlier_validation_does_not_restrict_discovery(rt, changes):
    write_phase(rt, **changes)
    assert active_polishing(rt) is None
    assert polishing_tool_guard(rt, "submit_function", {"name": "new_detector"}) is None
    assert polishing_repair_id(rt, "existing") is None


def test_crash_after_cursor_save_keeps_frozen_scope_until_runtime_decides_continuation(rt):
    path = rt.run.run_dir() / "round_validation.json"
    write_phase(rt, round=rt.run.round - 1, status="complete", polishing_id="polishing-guard:1")
    before = path.read_bytes()
    recovered = active_polishing(rt)
    assert recovered["transition_pending"] is True
    assert recovered["round"] == rt.run.round
    assert recovered["status"] == "awaiting_response"
    assert recovered["polishing_id"] == "polishing-guard:2"
    assert recovered["frozen_names"] == ["existing"]
    blocked = polishing_tool_guard(rt, "submit_function", {"name": "new_detector"})
    assert blocked["phase"] == "polishing" and not blocked["ok"]
    assert agent_target_grade_for_runtime(rt) == "GOOD"
    assert path.read_bytes() == before  # Recovery observation is read-only.


def test_invoke_rejects_fresh_mutation_before_execution(rt, monkeypatch):
    calls = []
    monkeypatch.setitem(server.TOOL_CALLABLES, "submit_function",
                        lambda name: calls.append(name) or {"ok": True})
    response = asyncio.run(server.invoke_tool("submit_function", {"name": "new_detector"}))
    payload = json.loads(response["content"][0]["text"])
    assert not payload["ok"] and payload["phase"] == "polishing"
    assert calls == []
    assert not list((rt.run.run_dir() / "operations").glob("*.json"))


def test_direct_family_calls_cannot_mutate_and_status_cannot_restart_survey(rt):
    assert not (rt.run.run_dir() / STATE_FILE).exists()
    blocked = family_discovery("record_family", family_name="new_hypothesis")
    assert not blocked["ok"] and blocked["phase"] == "polishing"
    status = family_discovery("status")
    assert status["ok"] and status["phase"] == "polishing"
    assert not (rt.run.run_dir() / STATE_FILE).exists()


def test_direct_restore_and_helper_conversion_reject_before_mutation(rt):
    from autofa.mcp.tools_helpers import retain_as_helper
    from autofa.mcp.tools_repair_versions import restore_function_version

    before = rt.library.to_dict()
    blocked = restore_function_version("new_detector", "saved", "Restore", "")
    assert not blocked["ok"] and blocked["phase"] == "polishing"
    blocked = retain_as_helper("existing", "extracted", "Retain as code")
    assert not blocked["ok"] and blocked["phase"] == "polishing"
    assert rt.library.to_dict() == before


def test_frozen_revisions_do_not_consume_or_reset_family_search_state(rt):
    state = {"version": 3, "phase": "survey", "mutation_attempts": {"existing": 9}}
    path = rt.run.run_dir() / STATE_FILE
    atomic_write_json(path, state)
    for _ in range(4):
        assert family_submission_guard(rt, "existing", consume_attempt=True) is None
    assert family_submission_guard(rt, "new_detector", consume_attempt=True)
    assert json.loads(path.read_text()) == state


def test_polishing_targets_good_even_after_minimum_is_met(rt):
    target = {
        "min_failure_modes": 1, "minimum_met": True,
        "failure_mode_eligibility": [{
            "name": "existing", "eligible": True, "measured": {
                "failure_recall": .05, "standardized_precision": 1.,
                "behavior_purity": 1., "n_matched_failed_datapoints": 5,
            },
        }],
    }
    normal = agent_failure_mode_target(target, round_idx=2, max_rounds=3)
    polishing = agent_failure_mode_target(target, round_idx=2, max_rounds=3, target_grade="GOOD")
    assert normal["target_grade"] == "OUTSTANDING" and not normal["minimum_met"]
    assert polishing["target_grade"] == "GOOD" and polishing["minimum_met"]
    assert agent_target_grade_for_runtime(rt) == "GOOD"
