# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic final scans select keepers and keep intentional exclusions visible."""

import json
from copy import deepcopy

import pytest

from autofa.core.library import FeatureFunction
from autofa.mcp import tools_eval
from autofa.metrics import claude_judge, final_review
from autofa.metrics.final_validation_selection import (
    current_final_validation_skip,
    display_final_validation_skip,
    prepare_final_validation,
    recover_final_validation_selection,
)
from autofa.metrics.verdicts import current_behavior_scores, format_metrics_line, verdict_for_scores
from autofa.reports.failure_cards import build_data, refresh_report_data
from autofa.ui.activity import AgentActivity

from support.evaluation import _llm_flag, _runtime, judge_rows, record_reviews, reply
from support.task_support import _console, _render_card


def runtime(tmp_path, monkeypatch, *, matches=True):
    fn = _llm_flag(description="The agent guesses.", mechanism="It acts on a guess.")
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=1, functions=[fn], n=60)
    rt.config.judge_workers = 1
    if not matches:
        for item in rt.dataset._fake_items:
            item["response_text"] = "fine"
    assert tools_eval.evaluate_functions()["ok"]
    return rt, fn


def forbid(*args, **kwargs):
    pytest.fail("No detector execution or judge work is allowed for this candidate")


def _raise(error):
    def fail(*args, **kwargs):
        raise error

    return fail


def test_completed_capped_frame_can_enter_final_census(tmp_path, monkeypatch, stub_claim_compiler):
    fn = _llm_flag(description="The agent guesses.", mechanism="It acts on a guess.")
    plain = FeatureFunction(name="plain", description="Empty control",
                            source="def plain(item): return False")
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=1, functions=[fn, plain], n=60)
    rt.run.eval_limit = 40
    assert len(rt.dataset._fake_items) > rt.run.eval_limit
    assert tools_eval.evaluate_functions()["ok"]
    assert fn.scores["sample_n"] == 20
    assert fn.scores["sample_frame_n"] == 40
    assert fn.scores["evaluation_scope_complete"] is False
    assert fn.scores["eval_truncated"] is True
    assert fn.scores["evaluation_checkpoint_complete"] is True
    monkeypatch.setattr(claude_judge, "_run_review",
                        lambda **kw: reply(kw["review"], status="supported"))

    with monkeypatch.context() as blocked:
        blocked.setattr(tools_eval, "run_detector", forbid)
        result = prepare_final_validation(rt, names=[fn.name])
    assert result["ok"] and result["selected_names"] == [fn.name]
    assert fn.scores["selection_review"]["status"] == "complete"
    # Admission does not upgrade the capped estimates to full-corpus scores.
    assert fn.scores["evaluation_scope_complete"] is False
    assert tools_eval.final_evaluate_functions(names=[fn.name])["censused"] == [fn.name]
    assert fn.scores["evaluation_scope_complete"] is True
    assert fn.scores["sampled"] is False
    assert fn.scores["n_scored"] == len(rt.dataset._fake_items)


# A capped frame without a true completion receipt cannot enter the final census.
@pytest.mark.parametrize("state", ["missing_receipt", "incomplete"])
def test_unfinished_capped_frame_cannot_enter_final_census(tmp_path, monkeypatch, state):
    rt, fn = runtime(tmp_path, monkeypatch)
    rt.run.eval_limit = 40
    assert tools_eval.evaluate_functions()["ok"]
    assert fn.scores["evaluation_scope_complete"] is False
    if state == "missing_receipt":
        fn.scores.pop("evaluation_checkpoint_complete")
    else:
        fn.scores["evaluation_checkpoint_complete"] = False
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    result = prepare_final_validation(rt)
    assert not result["ok"]
    assert not result["selected_names"]
    assert result["unresolved"][0]["name"] == fn.name


def test_unjudged_precise_sample_gets_bounded_review_before_full_selection(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    original = deepcopy(fn.scores)
    calls = record_reviews(monkeypatch, status="supported")
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    out = prepare_final_validation(rt)
    assert out["ok"] and out["selected_names"] == [fn.name]
    assert out["pending_reviewed"] == [fn.name]
    assert len(set(calls)) == max(3, int(fn.scores["n_rows_scored"] * .1))
    assert set(calls) <= set(original["evaluation_item_ids"])
    assert fn.scores["sampled"] and fn.scores["sample_n"] == 20
    assert fn.scores["evaluation_item_ids"] == original["evaluation_item_ids"]
    assert fn.scores["selection_review"]["status"] == "complete"
    assert verdict_for_scores(fn.scores)["code"] in {"GOOD", "OK"}
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert prepare_final_validation(rt)["selected_names"] == [fn.name]


def test_useful_ok_pending_review_overrides_automatic_precision_deferral(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    fn.scores["ppv_at_pi0"] = .4
    monkeypatch.setattr(claude_judge, "_run_review",
                        lambda **kw: reply(kw["review"], status="supported"))
    out = prepare_final_validation(rt)
    assert out["ok"] and out["selected_names"] == [fn.name]
    assert fn.scores["behavior_purity"] == 1
    assert verdict_for_scores(fn.scores)["code"] == "OK"


@pytest.mark.parametrize("kind", ["empty", "low_precision", "low_coverage", "rejected"])
def test_skips_ineligible_scores_and_preserves_sample_without_more_calls(
    tmp_path, monkeypatch, stub_claim_compiler, kind,
):
    rt, fn = runtime(tmp_path, monkeypatch, matches=kind != "empty")
    if kind == "low_precision":
        fn.scores["ppv_at_pi0"] = .39
    elif kind == "low_coverage":
        fn.scores["recall"] = .005
    elif kind == "rejected":
        monkeypatch.setattr(claude_judge, "_run_review", lambda **kw: reply(kw["review"]))
        assert final_review.final_review_functions(scope="selection_checkpoint")["ok"]
        assert verdict_for_scores(fn.scores)["code"] == "WEAK"
    original = deepcopy(fn.scores)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    out = prepare_final_validation(rt)
    assert out["ok"] and out["selected_names"] == []
    assert out["skipped"][0]["name"] == fn.name
    assert current_final_validation_skip(rt, fn)
    assert {k: v for k, v in fn.scores.items() if k != "final_validation_selection"} == original
    assert prepare_final_validation(rt)["skipped"] == out["skipped"]


@pytest.mark.parametrize("change", ["source", "score", "policy", "corpus", "run"])
def test_skip_receipt_never_settles_changed_candidate_or_source(
    tmp_path, monkeypatch, change,
):
    rt, fn = runtime(tmp_path, monkeypatch, matches=False)
    assert prepare_final_validation(rt)["skipped"]
    assert current_final_validation_skip(rt, fn)
    if change == "source":
        fn.source += "\n# changed predicate\n"
    elif change == "score":
        fn.scores["ppv_at_pi0"] = .99
    elif change == "policy":
        rt.config.behavior_strictness = "behavioral"
    elif change == "corpus":
        rt.dataset._fake_items[0]["response_text"] = "new retained content"
    else:
        rt.run.run_id = "continued-run"
    assert current_final_validation_skip(rt, fn) is None


def test_saved_receipt_survives_display_projection_but_not_score_change(tmp_path, monkeypatch):
    from autofa.reports.ordering import report_assessment

    rt, fn = runtime(tmp_path, monkeypatch, matches=False)
    prepare_final_validation(rt)
    projected = report_assessment(vars(fn), review_history=rt.library.review_history)["scores"]
    assert display_final_validation_skip(
        projected, source=fn.review_source, description=fn.description, mechanism=fn.mechanism,
    )
    projected["recall"] = .01
    assert display_final_validation_skip(
        projected, source=fn.review_source, description=fn.description, mechanism=fn.mechanism,
    ) is None


@pytest.mark.parametrize("sampled,abstain", [
    pytest.param(False, True, id="full-abstentions"),
    pytest.param(True, False, id="sampled-errors"),
])
def test_unresolved_predictions_distinguish_evidence_abstentions_from_execution_errors(
    tmp_path, monkeypatch, sampled, abstain,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    action = ("library.abstain('Required source evidence is unavailable.')" if abstain else
              "raise ValueError('Parser failed')")
    fn.source = f"def flag(item, llm_client=None):\n    {action}\n"
    assert tools_eval.evaluate_functions(
        names=[fn.name], mode="capped" if sampled else "full",
    )["ok"]
    expected = 20 if sampled else len(rt.dataset._fake_items)
    assert fn.scores["n_scored"] == 0
    assert fn.scores["n_insufficient" if abstain else "errors"] == expected
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    out = prepare_final_validation(rt)
    assert out["selected_names"] == []
    assert verdict_for_scores(fn.scores)["code"] == "UNSCORED"
    if abstain:
        assert out["ok"] and not out["unresolved"]
        assert "every attempted prediction abstained" in out["skipped"][0]["reason"]
        assert current_final_validation_skip(rt, fn)
    else:
        assert not out["ok"] and not out["skipped"]
        assert out["unresolved"][0]["name"] == fn.name
        assert current_final_validation_skip(rt, fn) is None


@pytest.mark.parametrize("problem", [
    "missing_population_rates", "review_infrastructure_failure", "outdated_execution",
])
def test_unmeasurable_candidate_remains_unresolved_without_skip_receipt(
    tmp_path, monkeypatch, stub_claim_compiler, problem,
):
    rt, fn = runtime(tmp_path, monkeypatch, matches=problem != "outdated_execution")
    if problem == "missing_population_rates":
        # Missing population rates are not a measured low quality.
        rt.config.llm_eval_sample = 1
        assert tools_eval.evaluate_functions(names=[fn.name])["ok"]
        assert fn.scores["metrics_scope"] == "unrepresented_strata"
    elif problem == "outdated_execution":
        fn.scores["evaluation_execution_version"] = "obsolete"
    if problem == "review_infrastructure_failure":
        monkeypatch.setattr(claude_judge, "judge_behavior_with_claude",
                            _raise(OSError("Review endpoint unavailable")))
    else:
        monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
        monkeypatch.setattr(tools_eval, "run_detector", forbid)
    out = prepare_final_validation(rt)
    assert not out["ok"]
    assert out["selected_names"] == out["skipped"] == []
    assert out["unresolved"][0]["name"] == fn.name
    assert current_final_validation_skip(rt, fn) is None


def test_substantive_uncertainty_skips_full_scan_and_does_not_repeat_review(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)

    def inspected_uncertainty(*, review, **kwargs):
        result = json.loads(reply(review, status="unknown"))
        for assessment in result["verdicts"][0]["claim_assessments"]:
            assessment["evidence_inspection"] = {
                "status": "complete",
                "ranges": [{"field": "response_text", "start": 0,
                            "end": len(review.fields["response_text"])}],
                "reason": "The relevant original response was inspected and remains ambiguous.",
            }
        return json.dumps(result)

    monkeypatch.setattr(claude_judge, "_run_review", inspected_uncertainty)
    out = prepare_final_validation(rt)
    assert out["ok"] and out["selected_names"] == [] and out["skipped"]
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    assert prepare_final_validation(rt)["skipped"]


def test_pending_review_resumes_frozen_selection_and_saved_stages(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    from autofa.session.recovery import RunInterruptedError

    rt, fn = runtime(tmp_path, monkeypatch)
    calls = record_reviews(monkeypatch, interrupt_at=3, status="supported")
    with pytest.raises((KeyboardInterrupt, RunInterruptedError)):
        prepare_final_validation(rt)
    frozen = deepcopy(fn.scores["selection_review"])
    assert frozen["status"] == "running"
    resumed = record_reviews(monkeypatch, status="supported")
    out = prepare_final_validation(rt)
    assert out["ok"] and out["selected_names"] == [fn.name]
    assert fn.scores["selection_review"]["item_ids"] == frozen["item_ids"]
    assert fn.scores["selection_review"]["operation_id"] == frozen["operation_id"]
    assert (calls + resumed).count(calls[0]) == 2  # Completed stages were reused on resume.


def test_full_scan_downgrade_keeps_admission_until_original_final_review_finishes(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(claude_judge, "_run_review",
                        lambda **kw: reply(kw["review"], status="supported"))
    assert prepare_final_validation(rt)["selected_names"] == [fn.name]
    admission = deepcopy(fn.scores["final_validation_selection"])
    checkpoint_review = deepcopy(fn.scores["selection_review"])
    # Nondeterministic detector predictions on newly scanned rows reveal passing
    # matches. Use the real evaluator to check its score-reconstruction path.
    monkeypatch.setattr(tools_eval, "run_detector", lambda *args, **kw: True)
    assert tools_eval.final_evaluate_functions(names=[fn.name])["ok"]
    assert verdict_for_scores(fn.scores)["code"] == "WEAK"
    assert fn.scores["final_validation_selection"] == admission
    assert fn.scores["selection_review"] == checkpoint_review
    # Simulate process interruption after scoring and again inside final review.
    assert prepare_final_validation(rt)["selected_names"] == [fn.name]
    judge = claude_judge.judge_behavior_with_claude

    def interrupted(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", interrupted)
    with pytest.raises(KeyboardInterrupt):
        final_review.final_review_functions(names=[fn.name])
    frozen = deepcopy(fn.scores["final_review"])
    assert frozen["status"] == "running"
    assert prepare_final_validation(rt)["selected_names"] == [fn.name]
    assert fn.scores["selection_review"] == checkpoint_review
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    assert final_review.final_review_functions(names=[fn.name])["ok"]
    assert fn.scores["final_review"]["operation_id"] == frozen["operation_id"]
    assert fn.scores["final_validation_selection"]["status"] == "complete"
    assert prepare_final_validation(rt)["selected_names"] == []
    assert current_final_validation_skip(rt, fn)


def recovery_proof(rt, fn):
    return {
        "run_id": rt.run.run_id, "names": [fn.name],
        "claims": {fn.name: fn.scores["evaluation_claim_hash"]},
        "source_signature": fn.scores["evaluation_context"],
        "policy_hash": claude_judge.judge_policy_hash(
            behavior_strictness=rt.config.behavior_strictness,
        ),
    }


def test_legacy_downgrade_recovers_admission_until_review_completes_once(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(claude_judge, "_run_review",
                        lambda **kw: reply(kw["review"], status="supported"))
    assert prepare_final_validation(rt)["selected_names"] == [fn.name]
    proof = recovery_proof(rt, fn)
    monkeypatch.setattr(tools_eval, "run_detector", lambda *args, **kw: True)
    assert tools_eval.final_evaluate_functions(names=[fn.name])["ok"]
    assert verdict_for_scores(fn.scores)["code"] == "WEAK"
    # Legacy scoring did not preserve a per-detector admission across the scan.
    fn.scores.pop("final_validation_selection")
    original = deepcopy(fn.scores)
    assert prepare_final_validation(rt)["selected_names"] == []
    assert current_final_validation_skip(rt, fn)

    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    recovered = recover_final_validation_selection(rt, **proof)
    assert recovered == {"recovered_names": [fn.name], "unresolved": []}
    assert {k: v for k, v in fn.scores.items() if k != "final_validation_selection"} == original
    admission = deepcopy(fn.scores["final_validation_selection"])
    assert recover_final_validation_selection(rt, **proof) == recovered
    assert fn.scores["final_validation_selection"] == admission
    assert prepare_final_validation(rt)["selected_names"] == [fn.name]
    # The completed census is reused and the pending review closes the receipt.
    assert tools_eval.final_evaluate_functions(names=[fn.name])["ok"]
    assert final_review.final_review_functions(names=[fn.name])["ok"]
    assert fn.scores["final_validation_selection"]["status"] == "complete"
    completed = deepcopy(fn.scores)
    assert recover_final_validation_selection(rt, **proof) == {
        "recovered_names": [], "unresolved": [],
    }
    assert fn.scores == completed
    assert prepare_final_validation(rt)["selected_names"] == []


@pytest.mark.parametrize("change", [
    "claim", "corpus", "score_source", "policy", "run", "incomplete", "missing_detector",
])
def test_legacy_recovery_rejects_changed_or_unverified_admission(tmp_path, monkeypatch, change):
    rt, fn = runtime(tmp_path, monkeypatch, matches=False)
    proof = recovery_proof(rt, fn)
    if change == "claim":
        fn.source += "\n# revised predicate\n"
        assert tools_eval.evaluate_functions()["ok"]
    elif change == "corpus":
        rt.dataset._fake_items[0]["response_text"] = "new retained content"
    elif change == "score_source":
        fn.scores["evaluation_context"] = "different-source"
    elif change == "policy":
        rt.config.behavior_strictness = "behavioral"
    elif change == "run":
        rt.run.run_id = "different-run"
    elif change == "incomplete":
        fn.scores["evaluation_scope_complete"] = False
    else:
        rt.library.functions = []
    before = deepcopy(fn.scores)
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    result = recover_final_validation_selection(rt, **proof)
    assert result["recovered_names"] == []
    assert result["unresolved"][0]["name"] == fn.name
    assert fn.scores == before


def test_recovered_admission_does_not_cover_a_later_revision(tmp_path, monkeypatch):
    rt, fn = runtime(tmp_path, monkeypatch, matches=False)
    assert recover_final_validation_selection(rt, **recovery_proof(rt, fn))["recovered_names"]
    assert prepare_final_validation(rt)["selected_names"] == [fn.name]
    fn.source += "\n# later repair must qualify on its own current evidence\n"
    assert tools_eval.evaluate_functions()["ok"]
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    result = prepare_final_validation(rt)
    assert result["selected_names"] == []
    assert result["skipped"][0]["name"] == fn.name


# --- reporting intentional exclusions ----------------------------------------


def _skipped(tmp_path, monkeypatch, *, sampled=True):
    fn = _llm_flag(description="The agent guesses.", mechanism="It acts on a guess.")
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=1, functions=[fn], n=80)
    assert tools_eval.evaluate_functions(mode="capped" if sampled else "full")["ok"]
    judge_rows(rt, fn, monkeypatch)
    result = prepare_final_validation(rt)
    assert not result["selected_names"] and result["skipped"]
    return rt, fn


def test_render_skipped_validation_preserves_actual_scope_and_badge(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = _skipped(tmp_path, monkeypatch, sampled=False)
    before = deepcopy(rt.library.to_dict())
    scores = current_behavior_scores(fn.scores, source=fn.review_source,
                                     description=fn.description, mechanism=fn.mechanism)
    assert verdict_for_scores(scores)["code"] == "WEAK"
    report = _render_card(build_data(rt.library))
    cli = format_metrics_line(scores)
    activity = " ".join(AgentActivity(_console())._library_snapshot(force=True)["lines"])
    for rendered in (report, cli, activity):
        assert "Full validation skipped:" in rendered
        assert "existing full-dataset scores retained" in rendered
        assert "retained sampled scores" not in rendered
        assert "WEAK" in rendered
    assert rt.library.to_dict() == before


# A score change keeps the claim, so only the refreshed report's merge can drop the skip.
@pytest.mark.parametrize("changed", ["source", "scores", "policy"])
def test_changed_evidence_invalidates_displayed_skip(
    tmp_path, monkeypatch, stub_claim_compiler, changed,
):
    rt, fn = _skipped(tmp_path, monkeypatch)
    old_report = build_data(rt.library)
    assert "Full validation skipped:" in _render_card(old_report)
    if changed == "source":
        fn.source += "\n# revised implementation\n"
    elif changed == "scores":
        fn.scores["ppv_at_pi0"] = 0.5
    else:
        fn.scores["judge_behavior_strictness"] = "contextual"
    fresh = build_data(rt.library)
    assert "Full validation skipped:" not in _render_card(fresh)
    refreshed = refresh_report_data(old_report, rt.library)
    assert "Full validation skipped:" not in _render_card(refreshed)
