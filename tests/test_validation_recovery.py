# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interrupted and modern resumes enter polishing from verified checkpoints without rescoring."""

import json
import threading
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from autofa.behavior_policy import strictness_for_config
from autofa.config import Config
from autofa.core.library import FeatureFunction
from autofa.mcp import tools_eval
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.session import runner
from autofa.session.evaluation_refresh import refresh_evaluation_contexts
from autofa.session.polishing import current_claims, public_validation_receipt
from autofa.session.polishing_guard import active_polishing
from autofa.session.polishing_progress import _claim, _evidence, _hash, recovery_work
from autofa.session.round_stagnation import _policy, round_stagnation_status
from autofa.session.validation_controls import collect_full_validation_controls
from autofa.session.validation_recovery import (
    _historical_baseline,
    _interrupted_boundary,
    complete_recovered_validation,
    pending_recovered_validation,
    prepare_recovered_validation,
    recover_interrupted_validation,
)
from autofa.util.atomic import atomic_write_json

from support.discovery import interactive as interactive
from support.discovery import interactive_runner as interactive_runner
from support.discovery import keeper
from support.evaluation import MATCHED_N
from support.evaluation import polishing_runtime as polishing_runtime


@pytest.fixture
def interrupted_runtime(tmp_path, monkeypatch):
    fn = FeatureFunction(name="candidate", source="def detect(item): return True",
                         description="An omitted required action", kind="python")
    fn.scores = {"evaluation_context": "context", "task_recall_corpus_fingerprint": "corpus"}
    rt = SimpleNamespace(
        run=SimpleNamespace(run_id="old-run", round=2, max_rounds=3, run_dir=lambda: tmp_path),
        config=Config(), library=SimpleNamespace(functions=[fn]), scores_history=[],
        _scores_history_lock=threading.RLock(),
    )
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda rt: "context")
    monkeypatch.setattr("autofa.session.family_coverage.cached_task_inventory", lambda rt: {
        "inventory_complete": True, "corpus_fingerprint": "corpus"})
    receipt = {"ok": True, "round": 1, "fraction": .3, "refinement": {
        "version": 1, "run_id": "old-run", "round": 1, "status": "complete",
        "evaluated_claims": current_claims(rt)}}
    atomic_write_json(tmp_path / "round_validation.json", receipt)
    save_interrupted_first_final(rt)
    return rt, fn, receipt


def history(rt, *, recall=.4, grade="GOOD", mode="capped", when=1):
    strictness = strictness_for_config(rt.config)
    claim = current_claims(rt)["candidate"]
    return {"round": 1, "eval_mode": mode, "wall_clock": when,
            "evaluated_claims": {"candidate": claim}, "results": [{
                "name": "candidate", "verdict": {"code": grade}, "scores": {
                    "recall": recall, "ppv_at_pi0": .95, "behavior_purity": 1.0,
                    "search_panel": {"run_id": rt.run.run_id, "claim_hash": claim,
                                     "corpus_fingerprint": "corpus"},
                    "behavior_validation": {
                        "judge_behavior_strictness": strictness,
                        "judge_policy_hash": judge_policy_hash(behavior_strictness=strictness)},
                }},
            ]}


FINAL_STARTED_AT = datetime(2026, 10, 21, 12, tzinfo=UTC)


def save_interrupted_final_round(rt, *, started_at=FINAL_STARTED_AT):
    """Save the previous round's receipt and a final advance_round interrupted while scoring."""
    receipt = {
        "ok": True, "round": rt.run.round - 1, "fraction": .3,
        "review_unresolved": [], "message": "Saved round validation findings.",
        "refinement": {
            "version": 1, "run_id": rt.run.run_id, "round": rt.run.round - 1,
            "status": "complete", "evaluated_claims": current_claims(rt),
            "response": "Retained the measured detector versions.",
        },
    }
    atomic_write_json(rt.run.run_dir() / "round_validation.json", receipt)
    save_interrupted_first_final(rt, started_at)
    return receipt


def save_modern(rt, *, controls=None, status="awaiting_response"):
    state = {
        "version": 1, "mode": "polishing", "run_id": rt.run.run_id,
        "round": rt.run.round, "status": status, "target_grade": "GOOD",
        "frozen_names": [fn.name for fn in rt.library.functions],
        "polishing_id": f"{rt.run.run_id}:{rt.run.round}",
        "evaluated_claims": current_claims(rt), "controls": controls or {},
        "baseline_grades": {"functions": [{"name": "candidate", "code": "OK"}]},
        "response": "Keep the saved repair response.", "retained_detectors": ["candidate"],
    }
    receipt = {"round": rt.run.round, "fraction": 1.0, "refinement": state}
    atomic_write_json(rt.run.run_dir() / "round_validation.json", receipt)
    return receipt


def save_interrupted_first_final(rt, started_at=FINAL_STARTED_AT):
    atomic_write_json(rt.run.run_dir() / "operations" / "first-final.json", {
        "tool": "advance_round", "state": "interrupted", "started_at": started_at.isoformat(),
    })
    (rt.run.run_dir() / "status_log.jsonl").write_text(json.dumps({
        "ts": (started_at + timedelta(minutes=1)).isoformat(), "source": "round_validation",
        "context": {"phase": "scoring", "fraction": 1.0},
    }) + "\n")


def forbid_execution(*args, **kwargs):
    pytest.fail("Resume must reuse saved detector predictions and review evidence")


def forbid_rescoring(monkeypatch):
    """Tests that need a startup validation replace finalization after this call."""
    monkeypatch.setattr(tools_eval, "run_detector", forbid_execution)
    monkeypatch.setattr("autofa.metrics.claude_judge._run_review", forbid_execution)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", forbid_execution)


def use_saved_fixture_inventory(rt, monkeypatch):
    """The harness fixture has saved scores and identities, without a real corpus."""
    inventory = rt._family_failed_task_inventory
    monkeypatch.setattr("autofa.session.family_coverage.cached_task_inventory",
                        lambda runtime: inventory)
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda runtime: "fixture-context")
    for fn in rt.library.functions:
        fn.scores["evaluation_context"] = "fixture-context"


def resume(rt):
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True, publish_on_exit=False)


class PolishingTurnReached(Exception):
    """Stop the harness after observing its first resumed prompt."""


def test_resume_round_target_rejects_past_round(tmp_path):
    run = SimpleNamespace(run_dir=lambda: tmp_path, run_id="old", round=2, max_rounds=99)
    with pytest.raises(ValueError, match="total round target"):
        runner.run_session(run, continue_mode=True, max_rounds=1)
    assert run.max_rounds == 99


def test_migration_retains_current_scores_round_and_exact_saved_receipt(
    interrupted_runtime, monkeypatch,
):
    rt, fn, receipt = interrupted_runtime
    rt.scores_history = [history(rt)]
    original = deepcopy(fn.scores)
    # The opaque legacy context hash is recorded, never rebound during migration.
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda rt: "canonical")
    state = recover_interrupted_validation(rt)
    assert state["legacy_recovery"]["source_signature"] == "canonical"
    assert state["legacy_recovery"]["validation_pending"] is True
    assert state["mode"] == "polishing" and state["status"] == "awaiting_response"
    assert state["frozen_names"] == [fn.name]
    assert state["legacy_recovery"]["admitted_names"] == [fn.name]
    assert state["baseline_grades"]["functions"][0]["code"] == "GOOD"
    assert pending_recovered_validation(rt)
    assert fn.scores == original and rt.run.round == 2
    backup = rt.run.run_dir() / "round_validation.before_polishing_migration.json"
    assert json.loads(backup.read_text()) == receipt
    path = rt.run.run_dir() / "round_validation.json"
    first = path.read_bytes()
    assert recover_interrupted_validation(rt) is None
    assert path.read_bytes() == first
    # Summaries do not leak the potentially large historical evidence payload.
    public = public_validation_receipt(json.loads(first))
    assert "previous_scores" not in public["refinement"]["legacy_recovery"]


def test_baseline_is_last_checkpoint_before_census_not_best_or_later_sample(interrupted_runtime):
    rt, _, _ = interrupted_runtime
    rt.scores_history = [history(rt, recall=.7), history(rt, recall=.2, grade="OK", when=2),
                         history(rt, recall=.05, grade="WEAK", mode="full", when=3),
                         history(rt, recall=.9, when=4)]
    before, grades, admitted = _historical_baseline(rt, {}, "corpus")
    assert before["candidate"]["recall"] == .2
    assert grades["functions"][0]["code"] == "OK"
    assert admitted == ["candidate"]


@pytest.mark.parametrize("change", ["other_panel", "changed_policy", "later_round", "new_claim"])
def test_history_admission_requires_current_panel_policy_and_boundary(interrupted_runtime, change):
    rt, _, _ = interrupted_runtime
    entry, boundary = history(rt), {}
    scores = entry["results"][0]["scores"]
    if change == "other_panel":
        scores["search_panel"]["claim_hash"] = "old-source"
    elif change == "changed_policy":
        scores["behavior_validation"]["judge_policy_hash"] = "old-policy"
    else:  # Later discoveries do not inherit an earlier full-validation admission.
        boundary = {"round": 1, "claims": current_claims(rt)}
        if change == "later_round":
            entry["round"] = 2
        else:
            boundary["claims"] = {"candidate": "an-earlier-version"}
    rt.scores_history = [entry]
    before, grades, admitted = _historical_baseline(rt, boundary, "corpus")
    assert not admitted
    if change == "changed_policy":  # The measured history is preserved, not admitted.
        assert before and grades["functions"][0]["code"] == "GOOD"
        assert not grades["functions"][0]["policy_matches"]
    else:
        assert not before and grades["functions"][0]["code"] is None


@pytest.mark.parametrize("status", ["selected", "reviewing"])
def test_incomplete_review_return_keeps_recovery_pending(interrupted_runtime, status):
    rt, fn, _ = interrupted_runtime
    recover_interrupted_validation(rt)
    fn.scores["final_validation_selection"] = {"status": status}
    fn.scores["final_review" if status == "selected" else "selection_review"] = {
        "status": "unavailable", "reason": "Reviewer service disconnected"}
    path = rt.run.run_dir() / "round_validation.json"
    saved = path.read_bytes()
    with pytest.raises(RuntimeError, match="recovery is still pending"):
        complete_recovered_validation(rt)
    assert path.read_bytes() == saved
    assert pending_recovered_validation(rt)


def test_pending_recovery_rechecks_identity_even_without_admitted_names(interrupted_runtime):
    rt, fn, _ = interrupted_runtime
    recover_interrupted_validation(rt)
    assert pending_recovered_validation(rt)["admitted_names"] == []
    fn.source += "\n# changed"
    with pytest.raises(RuntimeError, match="no longer matches"):
        prepare_recovered_validation(rt)
    assert pending_recovered_validation(rt)


@pytest.mark.parametrize("change", ["ordinary_scan", "current_modern"])
def test_ordinary_discovery_invalid_or_current_checkpoints_do_not_enter_polishing(
    interrupted_runtime, change,
):
    rt, fn, receipt = interrupted_runtime
    if change == "ordinary_scan":
        # An ordinary full scan records no runtime full-validation event.
        (rt.run.run_dir() / "status_log.jsonl").unlink()
        rt.scores_history = [history(rt, mode="full")]
    else:
        fn.scores["evaluated"] = True
        receipt = save_modern(rt)
    path = rt.run.run_dir() / "round_validation.json"
    atomic_write_json(path, receipt)
    original = path.read_bytes()
    assert recover_interrupted_validation(rt) is None
    assert path.read_bytes() == original
    assert not pending_recovered_validation(rt)


def test_interrupted_boundary_needs_matching_runtime_scoring_event(interrupted_runtime):
    rt, _, receipt = interrupted_runtime
    event = {"ts": "2026-10-21T11:59:00Z", "source": "round_validation",
             "context": {"phase": "scoring", "fraction": 1.0}}
    path = rt.run.run_dir() / "status_log.jsonl"
    path.write_text(json.dumps(event) + "\n")
    assert _interrupted_boundary(rt, receipt) is None
    event["ts"] = "2026-10-21T12:01:00Z"
    path.write_text('[]\n{"truncated":\n' + json.dumps(event) + "\n")
    boundary = _interrupted_boundary(rt, receipt)
    assert boundary["kind"] == "interrupted_full_checkpoint"
    rt.scores_history = [history(rt, when=boundary["started_at"] - 1),
                         history(rt, recall=.9, when=boundary["started_at"] + 1)]
    before, _, _ = _historical_baseline(rt, boundary, "corpus")
    assert before["candidate"]["recall"] == .4


def test_first_interrupted_final_validation_recovers_without_earlier_receipt(interrupted_runtime):
    rt, _, _ = interrupted_runtime
    rt.run.round, rt.run.max_rounds = 0, 1
    (rt.run.run_dir() / "round_validation.json").unlink()
    save_interrupted_first_final(rt)
    state = recover_interrupted_validation(rt)
    assert state["mode"] == "polishing"
    assert state["legacy_recovery"]["boundary"]["kind"] == "interrupted_full_checkpoint"
    assert pending_recovered_validation(rt)


def test_malformed_existing_checkpoint_does_not_use_missing_receipt_recovery(interrupted_runtime):
    rt, _, _ = interrupted_runtime
    (rt.run.run_dir() / "round_validation.json").write_text("{}")
    save_interrupted_first_final(rt)
    assert _interrupted_boundary(rt, {}) is None


@pytest.mark.parametrize("modern", [False, True], ids=["interrupted", "modern"])
def test_unverified_corpus_preserves_saved_checkpoint(interrupted_runtime, monkeypatch, modern):
    rt, fn, _ = interrupted_runtime
    if modern:
        fn.scores.update(evaluated=True, task_recall_corpus_fingerprint="other")
        save_modern(rt)
        monkeypatch.setattr(tools_eval, "_evaluation_context", lambda rt: "canonical")
    else:
        monkeypatch.setattr("autofa.session.family_coverage.cached_task_inventory", lambda rt: {
            "inventory_complete": True, "corpus_fingerprint": "different"})
    path = rt.run.run_dir() / "round_validation.json"
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="different or unverified corpus"):
        recover_interrupted_validation(rt)
    assert path.read_bytes() == original


def test_modern_context_recovery_preserves_exchange_and_is_idempotent(
    interrupted_runtime, monkeypatch,
):
    rt, fn, _ = interrupted_runtime
    fn.scores["evaluated"] = True
    rt.scores_history = [history(rt)]
    receipt = save_modern(rt, status="validating_response")
    original_scores = deepcopy(fn.scores)
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda rt: "canonical")
    state = recover_interrupted_validation(rt)
    assert {key: state[key] for key in receipt["refinement"]} == receipt["refinement"]
    assert pending_recovered_validation(rt)["admitted_names"] == [fn.name]
    assert fn.scores == original_scores
    path = rt.run.run_dir() / "round_validation.json"
    saved = path.read_bytes()
    assert recover_interrupted_validation(rt) is None
    assert path.read_bytes() == saved
    backup = path.with_name("round_validation.before_context_recovery.json")
    assert json.loads(backup.read_text()) == receipt


@pytest.mark.parametrize("status", ["selected", "complete"])
def test_modern_saved_selection_preserves_admission_without_reopening_completed_review(
    interrupted_runtime, monkeypatch, status,
):
    rt, fn, _ = interrupted_runtime
    strictness = strictness_for_config(rt.config)
    fn.scores.update(evaluated=True, final_validation_selection={
        "version": 1, "status": status, "run_id": rt.run.run_id,
        "claim_hash": current_claims(rt)[fn.name], "source_signature": "context",
        "policy_hash": judge_policy_hash(behavior_strictness=strictness),
        "judge_behavior_strictness": strictness,
    })
    save_modern(rt)
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda rt: "canonical")
    state = recover_interrupted_validation(rt)
    assert state["legacy_recovery"]["admitted_names"] == ([fn.name] if status == "selected" else [])


# Real saved full validations: controls are rebuilt from cached predictions only.


def test_recovery_rebuilds_controls_without_replaying_full_scan(
    polishing_runtime, monkeypatch,
):
    rt, fn, before, panel = polishing_runtime
    rt.run.max_rounds = 3
    rt.run.round = 2
    # Save the real pre-census scores under the unchanged evaluated claim. The
    # current library continues to hold the stronger, complete full scan.
    rt.scores_history[0].update(round=1, eval_mode="capped")
    rt.scores_history[0]["results"][0]["scores"] = deepcopy(before)
    rt.scores_history[-1]["round"] = 2
    rt.persist_scores_history()
    # The final advance_round started after the capped round and stopped while scoring.
    save_interrupted_final_round(rt, started_at=datetime.fromtimestamp(
        rt.scores_history[0]["wall_clock"] + 1, UTC))
    latest = deepcopy(fn.to_dict())
    cache = rt.run.run_dir() / "detector_results.sqlite3"
    cached_predictions = cache.read_bytes()
    forbid_rescoring(monkeypatch)

    assert recover_interrupted_validation(rt)
    path = rt.run.run_dir() / "round_validation.json"
    pending = path.read_bytes()
    recover_interrupted_validation(rt)
    assert path.read_bytes() == pending
    complete_recovered_validation(rt)
    phase = active_polishing(rt)
    assert phase and phase["frozen_names"] == [fn.name]
    controls = phase["controls"][fn.name]
    assert controls["available"] and controls["complete"], controls
    assert set(controls["panel_item_ids"]) == panel
    assert controls["score_changes"]["ppv_at_pi0"] == {
        "before": before["ppv_at_pi0"], "after": latest["scores"]["ppv_at_pi0"],
    }
    assert any(row["new_since_checkpoint"] for row in controls["controls"])
    assert set(controls["groups"]["flagged_success_contrast"]) == {
        f"case-{i}" for i in range(0, MATCHED_N, 2)
    }
    assert fn.to_dict() == latest
    assert cache.read_bytes() == cached_predictions

    reused = tools_eval.evaluate_functions(names=[fn.name])
    assert reused["ok"] and reused["reused_names"] == [fn.name]
    assert reused["evaluated_names"] == []
    assert fn.to_dict() == latest
    assert cache.read_bytes() == cached_predictions

    migrated = path.read_bytes()
    recover_interrupted_validation(rt)
    assert path.read_bytes() == migrated
    assert fn.to_dict() == latest


def test_modern_recovery_rebuilds_executable_controls_from_cached_predictions(
    polishing_runtime, monkeypatch,
):
    rt, fn, before, panel = polishing_runtime
    current_context = tools_eval._evaluation_context(rt)
    controls = collect_full_validation_controls(rt, fn, previous_scores=before)
    controls["source_signature"] = "old-process-hash"
    old_finding = {"claim_hash": "old-source-claim", "verdict": "yes"}
    controls["controls"][0]["review_findings"].append(old_finding)
    historical_item = controls["controls"][0]["item_id"]
    fn.scores["evaluation_context"] = "old-process-hash"
    receipt = save_modern(rt, controls={fn.name: controls}, status="validating_response")
    forbid_rescoring(monkeypatch)
    state = recover_interrupted_validation(rt)
    archived = state["legacy_recovery"]["historical_controls"][fn.name]
    assert archived == json.loads(json.dumps(controls))
    assert fn.scores["evaluation_context"] == "old-process-hash"
    assert refresh_evaluation_contexts(rt)["refreshed_names"] == [fn.name]
    complete_recovered_validation(rt)
    state = active_polishing(rt)
    rebuilt = state["controls"][fn.name]
    assert rebuilt["available"] and rebuilt["complete"]
    assert rebuilt["source_signature"] == current_context
    assert set(rebuilt["panel_item_ids"]) == panel
    assert set(rebuilt["item_ids"]) == set(controls["item_ids"])
    assert rebuilt["initial_score_changes"] == controls["initial_score_changes"]
    retained = next(row for row in rebuilt["controls"] if row["item_id"] == historical_item)
    assert {**old_finding, "historical": True} in retained["historical_review_findings"]
    assert old_finding not in retained["review_findings"]
    assert retained["baseline_claim_hash"] == current_claims(rt)[fn.name]
    for key in ("status", "response", "polishing_id", "baseline_grades", "frozen_names"):
        assert state[key] == receipt["refinement"][key]
    assert not pending_recovered_validation(rt)
    assert tools_eval.evaluate_functions(names=[fn.name])["reused_names"] == [fn.name]


def test_historical_controls_require_verified_original_run_corpus_and_panel(polishing_runtime):
    rt, fn, before, _ = polishing_runtime
    controls = collect_full_validation_controls(rt, fn, previous_scores=before)
    controls.update(source_signature="old-process-hash")
    assert collect_full_validation_controls(rt, fn, historical_controls=controls)["available"]
    for field, value in (("run_id", "another-run"), ("corpus_fingerprint", "another-corpus"),
                         ("panel_item_ids", ["wrong-panel-item"])):
        result = collect_full_validation_controls(
            rt, fn, historical_controls={**controls, field: value})
        assert not result["available"], field


# The resumed session reaches its first polishing turn without new work.


def test_recovery_restores_polishing_before_family_initialization_and_first_turn(
    interactive_runner, monkeypatch,
):
    harness, _, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = False
    rt.run.round = rt.run.max_rounds - 1
    rt.run.min_failure_modes = 2
    rt.library.functions = [
        keeper(i, evaluation_run_id=rt.run.run_id, ppv_at_pi0=.55) for i in range(2)
    ]
    use_saved_fixture_inventory(rt, monkeypatch)
    save_interrupted_final_round(rt)
    original_scores = deepcopy([fn.scores for fn in rt.library.functions])
    original_round = rt.run.round
    events = []
    initialize = runner.initialize_family_discovery

    def initialize_after_migration(runtime):
        phase = active_polishing(runtime)
        assert phase and phase["frozen_names"] == [fn.name for fn in rt.library.functions]
        events.append("initialize")
        return initialize(runtime)

    async def validate(console):
        assert active_polishing(rt)
        events.append("validate")

    async def turn(prompt, **kwargs):
        events.append("turn")
        assert "FINAL POLISHING" in prompt
        assert active_polishing(rt)["round"] == original_round
        assert events.index("initialize") < events.index("validate") < events.index("turn")
        raise PolishingTurnReached

    forbid_rescoring(monkeypatch)
    monkeypatch.setattr(runner, "initialize_family_discovery", initialize_after_migration)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", validate)
    monkeypatch.setattr(harness, "run_turn", turn)
    with pytest.raises(PolishingTurnReached):
        resume(rt)
    assert events.count("turn") == events.count("validate") == 1
    assert rt.run.round == original_round
    assert [fn.scores for fn in rt.library.functions] == original_scores
    path = rt.run.run_dir() / "round_validation.json"
    migrated = path.read_bytes()
    with pytest.raises(PolishingTurnReached):
        resume(rt)
    assert events.count("turn") == 2 and events.count("validate") == 1
    assert path.read_bytes() == migrated
    assert rt.run.round == original_round
    assert [fn.scores for fn in rt.library.functions] == original_scores


def test_interrupted_startup_validation_keeps_recovery_pending_until_resume(
    interactive_runner, monkeypatch,
):
    harness, _, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = False
    fn = keeper(0, evaluation_run_id=rt.run.run_id, ppv_at_pi0=.55)
    rt.library.functions = [fn]
    use_saved_fixture_inventory(rt, monkeypatch)
    # A modern receipt whose detector context hash came from an older process.
    fn.scores["evaluation_context"] = "old-process-hash"
    save_modern(rt)
    original_scores = deepcopy(fn.scores)
    original_round = rt.run.round
    validations, turns = [], []

    async def validate(console):
        validations.append(True)
        assert pending_recovered_validation(rt)
        if len(validations) == 1:
            raise ConnectionResetError("Connection interrupted during resumed full validation")
        fn.scores["evaluation_context"] = "fixture-context"  # The refreshed modern hash.

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert "FINAL POLISHING" in prompt
        assert not pending_recovered_validation(rt)
        raise PolishingTurnReached

    forbid_rescoring(monkeypatch)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", validate)
    monkeypatch.setattr(harness, "run_turn", turn)
    with pytest.raises(ConnectionResetError, match="resumed full validation"):
        resume(rt)
    assert not turns
    assert pending_recovered_validation(rt)
    assert active_polishing(rt)["frozen_names"] == [fn.name]
    assert rt.run.round == original_round
    assert fn.scores == original_scores
    with pytest.raises(PolishingTurnReached):
        resume(rt)
    assert len(validations) == 2 and len(turns) == 1
    assert not pending_recovered_validation(rt)
    assert rt.run.round == original_round
    assert fn.scores == {**original_scores, "evaluation_context": "fixture-context"}


def save_completed_exchange(rt, monkeypatch, *functions):
    """A completed polishing exchange saved before its next round cursor."""
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = True
    rt.library.functions = list(functions)
    use_saved_fixture_inventory(rt, monkeypatch)
    receipt = save_modern(rt, status="complete")
    work = recovery_work(rt, receipt["refinement"]["frozen_names"])
    receipt["refinement"].update(recovery_work=work, recovery_initial_work=sorted(work["work"]))
    atomic_write_json(rt.run.run_dir() / "round_validation.json", receipt)
    forbid_rescoring(monkeypatch)
    return receipt, work


def assert_reopened(rt, prompt, receipt, work, scores, round_index):
    assert "FINAL POLISHING" in prompt
    phase = active_polishing(rt)
    assert phase["status"] == "awaiting_response" and phase["round"] == round_index
    assert phase["frozen_names"] == receipt["refinement"]["frozen_names"]
    assert phase["recovery_initial_work"] == sorted(work["work"])
    assert phase["recovery_work"]["work"] == work["work"]
    assert [fn.scores for fn in rt.library.functions] == scores


def test_resume_reopens_legacy_exhaustion_without_rescoring_or_new_work(
    interactive_runner, monkeypatch,
):
    harness, _, rt = interactive_runner
    rt.run.round, rt.run.max_rounds, rt.run.min_failure_modes = 14, 8, 4
    receipt, work = save_completed_exchange(
        rt, monkeypatch, keeper(0, evaluation_run_id=rt.run.run_id, ppv_at_pi0=.55),
        keeper(1, evaluation_run_id=rt.run.run_id))
    rt.run.round = 15
    inventory = rt._family_failed_task_inventory
    old_fingerprint = _hash([
        rt.run.min_failure_modes, inventory["corpus_fingerprint"],
        sorted((fn.name, _claim(fn), _hash(_evidence(fn.scores or {})))
               for fn in rt.library.functions), {},
    ])
    atomic_write_json(rt.run.run_dir() / "discovery_round_stagnation.json", {
        "version": 1, "run_id": rt.run.run_id, "policy": _policy(rt),
        "recovery_exhausted": True, "recovery_evidence_fingerprint": old_fingerprint,
        "rounds_without_improve": 0,
    })
    with monkeypatch.context() as old_policy:
        old_policy.setattr("autofa.session.polishing_progress.recovery_state_fingerprint",
                           lambda rt: old_fingerprint)
        assert round_stagnation_status(rt)["should_stop"]
    assert not round_stagnation_status(rt)["should_stop"]
    scores = deepcopy([fn.scores for fn in rt.library.functions])

    async def turn(prompt, **kwargs):
        assert_reopened(rt, prompt, receipt, work, scores, 15)
        raise PolishingTurnReached

    monkeypatch.setattr(harness, "run_turn", turn)
    with pytest.raises(PolishingTurnReached):
        resume(rt)
    assert rt.run.round == 15


def test_first_resumed_turn_sees_results_refreshed_during_startup(
    interactive_runner, monkeypatch,
):
    harness, _, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.config.discovery_terminate_if_stalled = False
    rt.run.round = rt.run.max_rounds - 1
    rt.run.min_failure_modes = 2
    fn = keeper(evaluation_run_id=rt.run.run_id, ppv_at_pi0=.55)
    rt.library.functions = [fn]
    use_saved_fixture_inventory(rt, monkeypatch)
    save_interrupted_final_round(rt)

    async def validate(console):
        # Startup validation rejects four of the five reviewed examples.
        fn.scores.update(ppv_at_pi0=.76, behavior_n_yes=1, behavior_n_no=3,
                         behavior_n_uncertain=1, behavior_purity=.2)
        for row, verdict in zip(fn.scores["behavior_verdicts"],
                                ["yes", "no", "no", "no", "uncertain"], strict=True):
            row["verdict"] = row["initial_review"]["verdict"] = verdict

    async def turn(prompt, **kwargs):
        assert "FINAL POLISHING" in prompt
        assert "normalized precision 0.76" in prompt and "adherence 0.20" in prompt
        raise PolishingTurnReached

    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", validate)
    monkeypatch.setattr(harness, "run_turn", turn)
    with pytest.raises(PolishingTurnReached):
        resume(rt)
