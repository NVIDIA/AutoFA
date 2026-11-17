# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round checkpoints, resume handoffs and final summaries use current, fully measured validation."""

import json
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from autofa.config import Config
from autofa.core.library import FunctionLibrary
from autofa.harness.claude import TurnResult
from autofa.mcp import tools_eval, tools_library
from autofa.metrics import claude_judge, final_review
from autofa.objectives import apply_preset
from autofa.session import round_validation, runner
from autofa.session.discovery_target import discovery_good_target_for_runtime
from autofa.session.validation_completion import _validation_problem as real_validation_problem
from autofa.session.validation_completion import validated_good_target_for_runtime
from autofa.session.validation_summary import (
    current_validation_summary,
    evaluation_scope,
    render_validation_summary,
    save_validation_summary,
)
from autofa.util.atomic import atomic_write_json

from support.discovery import _quality_stall_assessment, keeper
from support.discovery import interactive as interactive
from support.discovery import interactive_runner as interactive_runner
from support.evaluation import reply, runtime


@pytest.mark.parametrize("round_index,fraction", [(1, .3), (2, 1.0), (3, 1.0)])
def test_last_planned_round_and_extra_rounds_use_full_validation(round_index, fraction):
    rt = SimpleNamespace(run=SimpleNamespace(round=round_index, max_rounds=3))
    assert round_validation.checkpoint_fraction(rt) == fraction


def test_final_downgrade_resumes_then_finishes_with_current_minimum(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = False
    rt.run.min_failure_modes = 3
    commands.append("/quit")
    turns, validations, highlights = [], [], []

    async def prepare_highlights(runtime, *, pending_only=False, **kwargs):
        if not pending_only:
            highlights.append(len(validations))

    monkeypatch.setattr("autofa.session.report_highlights.prepare_report_highlights_async",
                        prepare_highlights)

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) <= 2
        if len(turns) == 1:
            rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(6)]
        else:
            assert not highlights  # A full checkpoint returning to repair is not a final report.
            saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
            assert saved["n_validated_failure_modes"] == 2
            assert not saved["minimum_met"] and not saved["goal_met"]
            assert "FINAL POLISHING" in prompt
            rt.library.functions[2] = keeper(2, evaluation_run_id=rt.run.run_id)
            closed = tools_library.advance_round(validation_response="Repaired and checked.")
            assert closed["advanced"]
        return TurnResult(session_id="same-session", text="Ready to validate.")

    async def validate(console):
        validations.append(True)
        if len(validations) == 1:
            for fn in rt.library.functions[2:]:
                fn.scores["ppv_at_pi0"] = .55

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", validate)
    monkeypatch.setattr(round_validation, "validate_round", lambda rt: {
        "ok": True, "message": "Mocked final verification of the restored keeper.",
    })
    runner.run_session(rt.run, cfg=rt.config, publish_on_exit=False)
    assert len(turns) == len(validations) == 2
    assert highlights == [2]
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert saved["n_validated_failure_modes"] == 3
    assert saved["minimum_met"] and saved["goal_met"] and saved["validation_checked"]


def test_resume_validates_provisional_target_before_skipping_discovery(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = False
    rt.run.min_failure_modes = 2
    rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(2)]
    commands.append("/quit")
    turns, validations = [], []

    async def validate(console):
        validations.append(True)
        if len(validations) == 1:
            rt.library.functions[1].scores["ppv_at_pi0"] = .5

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert "FINAL POLISHING" in prompt
        rt.library.functions[1] = keeper(1, evaluation_run_id=rt.run.run_id)
        assert tools_library.advance_round(validation_response="Repaired and checked.")["advanced"]
        return TurnResult(session_id="resumed", text="Repaired the precision gap.")

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", validate)
    monkeypatch.setattr(round_validation, "validate_round", lambda rt: {
        "ok": True, "message": "Mocked final verification of the restored keeper.",
    })
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True, publish_on_exit=False)
    assert len(turns) == 1 and len(validations) == 2


@pytest.mark.parametrize("round_index", [0, 2])
def test_round_checkpoint_scores_then_reviews_before_feedback(
    interactive_runner, monkeypatch, round_index,
):
    monkeypatch.setattr(
        "autofa.metrics.final_validation_selection.prepare_final_validation",
        lambda rt: {"selected_names": [fn.name for fn in rt.library.functions], "unresolved": []},
    )
    _, _, rt = interactive_runner
    rt.run.round = round_index
    rt.library.functions = [keeper(0, evaluation_run_id=rt.run.run_id)]
    events = []

    def score(**kwargs):
        events.append(("score", kwargs))
        fn = rt.library.functions[0]
        fn.scores.update(ppv_at_pi0=.5, sampled=round_index == 0)
        return {"ok": True}

    def review(**kwargs):
        assert events[0][0] == "score"
        events.append(("review", kwargs))
        return {"ok": True, "unresolved": []}

    monkeypatch.setattr(tools_eval, "evaluate_round_functions", score)
    monkeypatch.setattr(tools_eval, "final_evaluate_functions", score)
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", review)
    result = round_validation.validate_round(rt)
    assert result["ok"] and result["round"] == round_index
    assert result["grades"]["n_good_or_better"] == 0
    assert result["detectors"][0]["previous_grade"] in {"GOOD", "EXCELLENT", "OUTSTANDING"}
    assert result["detectors"][0]["grade"] == "OK"
    if round_index == 0:
        assert events[0][1] == {"fraction": .3}
        assert events[1][1] == {"scope": "round_checkpoint"}
    else:
        assert events[0][1] == events[1][1] == {"names": [rt.library.functions[0].name]}
        assert result["refinement"]["mode"] == "polishing"
    assert rt.run.round == round_index


def test_failed_scoring_does_not_review_or_advance_checkpoint(interactive_runner, monkeypatch):
    _, _, rt = interactive_runner
    rt.run.round = 0
    rt.library.functions = [keeper(0, evaluation_run_id=rt.run.run_id)]
    monkeypatch.setattr(tools_eval, "evaluate_round_functions",
                        lambda **kwargs: {"ok": False, "error": "Connection interrupted"})
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions",
                        lambda **kwargs: pytest.fail("Cannot review a failed scoring checkpoint"))
    result = round_validation.validate_round(rt)
    assert not result["ok"] and result["advanced"] is False
    assert rt.run.round == 0
    assert not (rt.run.run_dir() / "round_validation.json").exists()


def test_resume_after_polishing_cursor_save_gets_repair_prompt_first(
    interactive_runner, monkeypatch,
):
    from autofa.session.polishing import current_claims

    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = False
    rt.run.min_failure_modes = 2
    rt.run.round = rt.run.max_rounds
    rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(2)]
    rt.library.functions[1].scores["ppv_at_pi0"] = .55
    atomic_write_json(rt.run.run_dir() / "round_validation.json", {"refinement": {
        "version": 1, "mode": "polishing", "run_id": rt.run.run_id,
        "round": rt.run.round - 1, "status": "complete",
        "frozen_names": [f.name for f in rt.library.functions], "controls": {},
        "evaluated_claims": current_claims(rt),
    }})
    commands.append("/quit")
    turns = []

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) == 1 and "FINAL POLISHING" in prompt
        assert round_validation.active_round_refinement(rt)["round"] == rt.run.round
        rt.library.functions[1] = keeper(1, evaluation_run_id=rt.run.run_id)
        assert tools_library.advance_round(validation_response="Restored and checked.")["advanced"]
        return TurnResult(session_id="resumed", text="Repairs complete.")

    async def validate(console):
        return None

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", validate)
    monkeypatch.setattr(round_validation, "validate_round", lambda rt: {
        "ok": True, "message": "Mocked verification of restored detector.",
    })
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True, publish_on_exit=False)
    assert len(turns) == 1


def test_resume_does_not_skip_closing_exchange_after_good_minimum_is_met(
    interactive_runner, monkeypatch,
):
    harness, commands, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.run.min_failure_modes = 1
    rt.library.functions = [keeper(0, evaluation_run_id=rt.run.run_id)]
    path = rt.run.run_dir() / "round_validation.json"
    path.write_text(json.dumps({"refinement": {
        "version": 1, "run_id": rt.run.run_id, "round": rt.run.round,
        "status": "awaiting_response", "evaluated_claims": {},
    }}))
    commands.append("/quit")
    prompts = []

    async def turn(prompt, **kwargs):
        prompts.append(prompt)
        assert "CLOSING REFINEMENT" in prompt
        assert "validation_response" in prompt
        path.unlink()  # The tool-completion tests exercise durable acknowledgement.
        return TurnResult(session_id="closing", text="Retained the validated version.")

    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True, publish_on_exit=False)
    assert len(prompts) == 1


@pytest.mark.parametrize("change", ["run_id", "round", "status", "version", "malformed"])
def test_refinement_does_not_resume_stale_or_malformed_receipts(interactive_runner, change):
    _, _, rt = interactive_runner
    state = {"version": 1, "run_id": rt.run.run_id, "round": rt.run.round,
             "status": "awaiting_response"}
    state[change] = {"run_id": "old-run", "round": rt.run.round + 1, "status": "complete",
                     "version": 0, "malformed": None}[change]
    payload = [] if change == "malformed" else {"refinement": state}
    (rt.run.run_dir() / "round_validation.json").write_text(json.dumps(payload))
    assert round_validation.active_round_refinement(rt) is None


@pytest.mark.parametrize("abstain", ["one", "all"])
def test_real_evaluation_and_final_review_form_a_countable_round(
    tmp_path, monkeypatch, stub_claim_compiler, abstain,
):
    from autofa.session.discovery_stall import complete_discovery_round

    rt, fn = runtime(tmp_path, monkeypatch)
    selected = rt.dataset._fake_items if abstain == "all" else rt.dataset._fake_items[:1]
    for item in selected:
        item["response_text"] = "missing evidence"
    fn.source = fn.source.replace(
        "    return", "    if item.get('response_text') == 'missing evidence':\n"
        "        library.abstain('Source evidence is missing.')\n    return",
    )
    # Selection needs current discovery scores, not a stale earlier claim.
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    monkeypatch.setattr(claude_judge, "_run_review", lambda **kw: reply(kw["review"]))
    rt.run.round = rt.run.max_rounds - 1
    validated = round_validation.validate_round(rt)
    assert validated["ok"]
    assert fn.scores["n_insufficient"] == len(selected)
    if abstain == "all":
        assert validated["review_unresolved"] == []
        assert "final_review" not in fn.scores
        assert fn.scores["n_scored"] == fn.scores["n_flagged"] == fn.scores["errors"] == 0
    else:
        assert validated["review_unresolved"] == []
        assert fn.scores["selection_review"]["status"] == "complete"
        assert validated["final_validation_selection"]["skipped"][0]["name"] == fn.name
    baseline = complete_discovery_round(rt, validated)
    assert baseline["status"] == "baseline", baseline
    rt.run.round += 1
    validated = round_validation.validate_round(rt)
    completed = complete_discovery_round(rt, validated)
    assert not completed["counted"] and completed["rounds_without_improve"] == 0, completed
    assert completed["status"] == "no_new_recovery_work"


@pytest.fixture
def reviewed(tmp_path, monkeypatch, stub_claim_compiler, request):
    rt, fn = runtime(tmp_path, monkeypatch)
    if hasattr(request, "param"):
        rt.config = apply_preset(rt.config, request.param)
    rt.run.min_failure_modes = 1
    monkeypatch.setattr(claude_judge, "_run_review",
                        lambda **kwargs: reply(kwargs["review"], "supported"))
    assert final_review.final_review_functions()["ok"]
    assert discovery_good_target_for_runtime(rt)["minimum_met"]
    return rt, fn


@pytest.mark.parametrize("reviewed", ["precision"], indirect=True)
def test_current_final_receipt_can_reuse_completed_current_policy_reviews(reviewed, monkeypatch):
    rt, fn = reviewed
    fn.scores.pop("final_review")

    def forbidden(**kwargs):
        pytest.fail("Compatible completed evidence must not be judged again")

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbidden)
    assert final_review.final_review_functions()["ok"]
    assert fn.scores["judge_policy_hash"] == claude_judge.judge_policy_hash()
    assert fn.scores["final_review"]["policy_hash"] == claude_judge.judge_policy_hash()
    assert validated_good_target_for_runtime(rt)["minimum_met"]
    assert current_validation_summary(rt)["detectors"][0]["review"]["completion"] == "complete"


def test_honest_abstention_does_not_become_an_execution_error_or_block_good(reviewed):
    rt, fn = reviewed
    rt.dataset._fake_items[0]["response_text"] = "missing evidence"
    fn.source = fn.source.replace(
        "    return", "    if item.get('response_text') == 'missing evidence':\n"
        "        library.abstain('Source evidence is missing.')\n    return",
    )
    assert tools_eval.final_evaluate_functions(names=[fn.name])["ok"]
    assert final_review.final_review_functions(names=[fn.name])["ok"]
    assert fn.scores["n_insufficient"] == 1 and fn.scores["errors"] == 0
    assert validated_good_target_for_runtime(rt)["minimum_met"]


def test_provisional_or_stale_evidence_cannot_confirm_completion(reviewed):
    # Each case edits a fresh copy of one real reviewed result; the fixture is costly.
    rt, fn = reviewed
    current = deepcopy(fn.scores)
    for change in [{"sampled": True}, {"evaluation_context": "changed-source"},
                   {"errors": 1}, {"final_review": None}]:
        fn.scores = {**deepcopy(current), **change}
        assert discovery_good_target_for_runtime(rt)["minimum_met"], change
        target = validated_good_target_for_runtime(rt)
        assert not target["minimum_met"] and target["provisional_good_count"] == 1, change
    for change in [{"status": "running"}, {"policy_hash": "old-policy"},
                   {"item_ids": ["not-the-cohort"]}]:  # Stale or unfinished final reviews.
        fn.scores = deepcopy(current)
        fn.scores["final_review"].update(change)
        assert discovery_good_target_for_runtime(rt)["minimum_met"], change
        assert not validated_good_target_for_runtime(rt)["minimum_met"], change


def test_real_validation_qualifies_without_new_calls_and_summary_matches_html(reviewed):
    from autofa.reports.failure_cards import build_data

    rt, fn = reviewed
    before = deepcopy(fn.to_dict())
    target = validated_good_target_for_runtime(rt)
    assert target["minimum_met"]
    assert target["validated_failure_modes"] == [fn.name]
    assert fn.to_dict() == before
    summary = current_validation_summary(rt)
    report = build_data(rt.library)
    expected = {row["name"]: row["grade"] for row in summary["detectors"]}
    assert expected == {row["name"]: row["verdict"]["code"] for row in report["functions"]}
    assert expected[fn.name] == "GOOD"


# Harness, missing evidence and stall handling are independent; cover each value once.
@pytest.mark.parametrize("interactive,missing,stalled", [
    ("claude", "final_review", False), ("codex", "evaluation_context", True),
], indirect=["interactive"])
def test_finalization_returns_pending_good_to_polishing_before_success(
    interactive_runner, reviewed, monkeypatch, missing, stalled,
):
    harness, commands, _ = interactive_runner
    rt, fn = reviewed
    rt.config.harness = "codex" if type(harness).__name__ == "CodexHarness" else "claude"
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = stalled
    rt.run.round = rt.run.max_rounds
    # The judge may follow the selected harness. Establish its real current
    # receipt before injecting a post-finalization missing-evidence condition.
    assert final_review.final_review_functions()["ok"]
    completed = deepcopy(fn.scores)
    monkeypatch.setattr(runner, "load_runtime_from_run", lambda *args: rt)
    monkeypatch.setattr("autofa.session.validation_completion._validation_problem",
                        real_validation_problem)
    if stalled:
        monkeypatch.setattr("autofa.session.discovery_stall.observe_discovery_stall",
                            lambda *args, **kwargs: _quality_stall_assessment())
    commands.append("/quit")
    turns, finalizations = [], []

    async def finalize(console):
        finalizations.append(True)
        if len(finalizations) == 1:
            fn.scores[missing] = None

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) == 1
        assert "FINAL POLISHING" in prompt
        saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
        assert not saved["minimum_met"] and not saved["goal_met"]
        assert saved["provisional_good_count"] == 1
        assert discovery_good_target_for_runtime(rt)["minimum_met"]
        # Recover the exact previously completed measurement; real rescore/review
        # recovery is tested separately. No new predicate or grade is fabricated.
        fn.scores = deepcopy(completed)
        (rt.run.run_dir() / "round_validation.json").unlink()
        return TurnResult(session_id="recovered", text="Recovered current validation evidence.")

    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, publish_on_exit=False)
    saved = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    if stalled:
        assert not turns and len(finalizations) == 1
        assert not saved["minimum_met"] and not saved["goal_met"]
        assert saved["stop_reason"] == "stalled"
        assert saved["n_validated_failure_modes"] == 0
        assert saved["validation_blockers"][0]["name"] == fn.name
    else:
        assert len(turns) == 1 and len(finalizations) == 2
        assert saved["minimum_met"] and saved["goal_met"]
        assert saved["n_validated_failure_modes"] == 1


# The resume handoff and final status must use current saved measurements.


@pytest.fixture
def rt(tmp_path):
    return SimpleNamespace(
        config=Config(behavior_strictness="causal"), library=FunctionLibrary(),
        run=SimpleNamespace(run_dir=lambda: tmp_path, run_id="fixture-run", round=3,
                            max_rounds=3, min_failure_modes=3),
    )


def baseline(rt, fn):
    atomic_write_json(rt.run.run_dir() / "round_validation.json", {"refinement": {
        "version": 1, "mode": "polishing", "run_id": rt.run.run_id,
        "round": rt.run.round, "status": "awaiting_response", "frozen_names": [fn.name],
        "baseline_grades": {"functions": [{"name": fn.name, "code": "GOOD"}]},
        "controls": {fn.name: {"initial_score_changes": {
            "ppv_at_pi0": {"before": .57, "after": .61},
            "behavior_purity": {"before": 1., "after": 1.},
        }}},
    }})


def rejected(fn):
    fn.scores.update(behavior_n_yes=1, behavior_n_no=3, behavior_n_uncertain=1,
                     behavior_purity=.2)
    for row, verdict in zip(fn.scores["behavior_verdicts"],
                            ["yes", "no", "no", "no", "uncertain"], strict=True):
        row.update(verdict=verdict, rationale="The reply retracts its submission and resumes work.")
        row["initial_review"].update(verdict=verdict)


def test_resume_prompt_injects_refreshed_scores_and_specific_counterexamples(rt):
    fn = keeper(ppv_at_pi0=.76)
    rejected(fn)
    rt.library.functions = [fn]
    baseline(rt, fn)
    prompt = round_validation.round_refinement_prompt(rt)
    assert "normalized precision 0.57 → 0.76" in prompt
    assert "adherence 1.00 → 0.20" in prompt
    assert "Review f1 (no): The reply retracts its submission" in prompt
    # Reading current results must not overwrite saved prior evidence.
    assert json.loads((rt.run.run_dir() / "round_validation.json").read_text())[
        "refinement"]["controls"][fn.name]["initial_score_changes"]["ppv_at_pi0"]["after"] == .61


@pytest.mark.parametrize("changes,scope", [
    ({}, "full_dataset"),
    ({"sampled": True, "sample_n": 200, "sample_frame_n": 3000}, "sampled"),
    ({"eval_truncated": True}, "partial"),
    ({"evaluation_scope_complete": None}, "unknown"),
    ({"evaluation_stale": True}, "not_evaluated"),
])
def test_scope_requires_positive_full_scan_evidence(changes, scope):
    assert evaluation_scope(keeper(**changes).scores) == scope


def test_deterministic_final_summary_keeps_mixed_scopes(rt, capsys):
    rt.library.functions = [keeper(), keeper(1, sampled=True, sample_n=200,
                                             sample_frame_n=3000)]
    saved = save_validation_summary(rt)
    assert capsys.readouterr().out == ""
    assert saved == json.loads((rt.run.run_dir() / "validation_summary.json").read_text())
    assert saved["good_count"] == 2 and not saved["minimum_met"]
    assert saved["scope_counts"] == {"full_dataset": 1, "sampled": 1}
    assert "minimum unmet" in render_validation_summary(saved)


def test_stale_reviews_do_not_become_current_adherence_or_completed_reviews(rt):
    fn = keeper(behavior_evidence_stale=True)
    fn.scores["final_review"] = {"status": "complete", "claim_hash": fn.scores["claim_hash"]}
    rt.library.functions = [fn]
    row = current_validation_summary(rt)["detectors"][0]
    assert row["adherence"] is None
    assert row["review"]["completion"] is None
    assert row["review"]["status"] == "stale"
    assert row["review"]["findings"] == []


def test_completed_receipt_must_match_current_policy_and_review_cohort(rt):
    fn = keeper(sampled=True, sample_n=200, sample_frame_n=3000)
    rt.library.functions = [fn]
    receipt = {"status": "complete", "claim_hash": fn.scores["claim_hash"],
               "policy_hash": fn.scores["judge_policy_hash"],
               "item_ids": [r["item_id"] for r in fn.scores["behavior_verdicts"]]}
    fn.scores["final_review"] = {**receipt, "policy_hash": "old"}
    assert current_validation_summary(rt)["detectors"][0]["review"]["completion"] is None
    fn.scores["round_review"] = {**receipt, "item_ids": ["other-cohort"]}
    assert current_validation_summary(rt)["detectors"][0]["review"]["completion"] is None
    fn.scores["round_review"] = receipt
    assert current_validation_summary(rt)["detectors"][0]["review"]["completion"] == "complete"


def test_summary_write_failure_does_not_discard_completed_final_validation(rt, monkeypatch):
    from autofa.session import finalization

    completed = []
    monkeypatch.setattr("autofa.mcp.runtime.get_runtime", lambda: rt)
    monkeypatch.setattr(finalization, "_refresh_final_evaluations", lambda console: None)
    monkeypatch.setattr("autofa.session.validation_recovery.prepare_recovered_validation",
                        lambda rt: None)
    monkeypatch.setattr(finalization, "_prepare_final_selection",
                        lambda console: {"selected_names": []})
    monkeypatch.setattr("autofa.mcp.tools_eval.final_evaluate_functions",
                        lambda **kwargs: {"ok": True, "n": 0})
    def reviewed(*args, **kwargs):
        completed.append("review")
        return {"ok": True, "unresolved": []}

    monkeypatch.setattr(finalization, "_review_final_library", reviewed)

    def fail(*args, **kwargs):
        raise OSError("Summary destination unavailable")

    monkeypatch.setattr("autofa.session.validation_summary.save_validation_summary", fail)
    finalization._evaluate_final_library(Console(file=StringIO()))
    assert completed == ["review"]
