# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic, adaptive and outcome-control reviews never pollute scored reviews."""

import json
import zlib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.core.library import FunctionLibrary
from autofa.mcp import tools_eval, tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics import claude_judge
from autofa.metrics.adaptive_review import (
    merge_expanded_review,
    normalize_sample_request,
    sample_request,
)
from autofa.metrics.diagnostic_review import diagnostic_scores, known_failed_smoke_matches
from autofa.metrics.repair_feedback import build_repair_feedback
from autofa.metrics.review_content import RetainedReviewContent
from autofa.metrics.verdicts import behavior_validation
from autofa.session.family_discovery import initialize_family_discovery
from autofa.session.repair_workflow import plan_repair
from autofa.session.review_evidence import (
    allocate_review,
    find_review_allocation,
    task_group,
)

from support.repair_review import REQUEST, TIGHTER, plan, workflow
from support.repair_review import family_review as family_review
from support.repair_review import repair_runtime as repair_runtime

RECOVERED = {"status": "contradicted", "rationale": "The trace shows recovery."}
UNREAD = {"status": "unknown", "rationale": "Premise is not retained.",
          "unknown_reason": "evidence_unavailable",
          "evidence_inspection": {
              "status": "unavailable", "ranges": [],
              "reason": "Required premise is absent from the inspected record."}}


def judge_with(change=None, *, batch=None, first_only=False, request=None):
    """Wrap the stub judge, applying ``change`` to the causal requirements of matching reviews.

    A review matches when its prompt names ``batch`` (any batch by default); with
    ``first_only`` only the first review changes. ``request`` asks every reviewer for an
    additional sample. The wrapper records ``prompts`` and the ``changed`` item IDs.
    """
    run_review = claude_judge._run_review

    def review(**kwargs):
        review.prompts.append(kwargs["prompt"])
        data = json.loads(run_review(**kwargs))
        row = data["verdicts"][0]
        if (change and (batch is None or f'"batch": "{batch}"' in kwargs["prompt"])
                and not (first_only and review.changed)):
            review.changed.append(row["item_id"])
            mechanism_id = next(
                clause["id"] for clause in kwargs["review"].snapshot["claim_contract"][
                    "defining_clauses"] if clause["source_field"] == "mechanism")
            for assessment in row["claim_assessments"]:
                if assessment["requirement_id"] in {"causal_link", mechanism_id}:
                    assessment.update(deepcopy(change))
                    if assessment["requirement_id"] == "causal_link":
                        assessment["defining_requirement_ids"] = [mechanism_id]
        if request:
            row["additional_sample_request"] = request
        return json.dumps(data)

    review.prompts, review.changed = [], []
    return review


def interrupt_third_review(run_review):
    """Wrap ``run_review`` so the third review raises a connection reset."""
    attempts = []

    def review(**kwargs):
        attempts.append(kwargs["review"].snapshot["item_id"])
        if len(attempts) == 3:
            raise ConnectionResetError("interrupted review")
        return run_review(**kwargs)

    return review


def development_allocation(rt, fn, suffix=""):
    key = tools_eval._review_key(fn.to_dict(), tools_eval._review_settings(rt.config)[2], None)
    return find_review_allocation(rt.run.run_dir(), fn.name, review_key=key + suffix,
                                  phase="development")


@pytest.mark.parametrize("kwargs", [
    {"review_mode": "unknown"}, {"review_mode": "diagnostic", "review_phase": "confirmation"},
    {"review_mode": "diagnostic", "evidence_n": 4}, {"evidence_n": 0},
])
def test_invalid_modes_and_budgets_do_no_work(family_review, monkeypatch, kwargs):
    rt, fn, calls = family_review
    before = deepcopy(fn.scores)
    monkeypatch.setattr(tools_eval, "_iter_scored_items",
                        lambda *args, **kw: pytest.fail("invalid input must not scan"))
    result = tools_eval.judge_function(fn.name, **kwargs)
    assert not result["ok"], result
    assert fn.scores == before
    assert calls == []


def test_diagnostic_review_is_bounded_and_does_not_create_quality_scores(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    before = deepcopy(fn.scores)
    batches = []
    original = claude_judge.judge_behavior_with_claude

    def capture(**kwargs):
        batches.append(kwargs["review_batch_context"])
        return original(**kwargs)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", capture)
    result = tools_eval.judge_function(fn.name, review_mode="diagnostic", evidence_n=1)
    assert result["ok"], result
    assert result["n_judged"] == 1
    assert len(calls) == 2
    assert len(batches) == 1
    assert batches[0]["batch"] == "diagnostic"
    assert batches[0]["expansion_available"] is False
    assert result["sample"]["phase"] == "development"
    assert result["sample"]["review_mode"] == "diagnostic"
    assert result["sample"]["excluded_from_adherence"] is True
    assert "diagnostic_sample" not in result["sample"]
    assert {key: value for key, value in fn.scores.items()
            if key not in {"diagnostic_review", "judge_prefilter"}} == {
                key: value for key, value in before.items() if key != "judge_prefilter"}
    assert behavior_validation(fn.scores)["status"] == "unjudged"
    assert diagnostic_scores(fn.scores) is not None
    assert result["diagnostic_review"]["current"] is True
    assert result["diagnostic_review"]["excluded_from_repair_validation"] is True
    assert result["judge_feedback"]["diagnostic_feedback"]["confirmed"] == 1


def test_diagnostic_ignores_adaptive_request(family_review, monkeypatch):
    # Without evidence_n, only the diagnostic mode itself keeps expansion unavailable.
    rt, fn, calls = family_review
    monkeypatch.setattr("autofa.metrics.adaptive_review.sample_request", lambda result: REQUEST)
    result = tools_eval.judge_function(fn.name, review_mode="diagnostic")
    assert result["ok"], result
    assert result["n_judged"] == 3 and len(calls) == 6
    assert result["sample"]["judge_expansion"]["status"] == "not_available"


def test_diagnostic_interruption_resumes_frozen_cohort_and_completed_stages(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    original = claude_judge._run_review
    monkeypatch.setattr(claude_judge, "_run_review", interrupt_third_review(original))
    response = tools_eval.judge_function(fn.name, review_mode="diagnostic")
    assert not response["ok"] and response["resumable"], response
    allocated = development_allocation(rt, fn, ":diagnostic")
    assert allocated["sample_info"]["target_n"] == 3
    rt.dataset.items.reverse()
    monkeypatch.setattr(claude_judge, "_run_review", original)
    resumed = tools_eval.judge_function(fn.name, review_mode="diagnostic")
    assert resumed["ok"], resumed
    assert [row["item_id"] for row in resumed["verdicts"]] == allocated["item_ids"]
    assert calls == [item_id for item_id in allocated["item_ids"] for _ in range(2)]
    count = len(calls)
    repeated = tools_eval.judge_function(fn.name, review_mode="diagnostic")
    assert repeated["ok"], repeated
    assert len(calls) == count


@pytest.mark.parametrize("review_mode", ["diagnostic", "automatic"])
def test_frozen_standard_allocation_survives_other_review_modes(family_review, review_mode):
    rt, fn, calls = family_review
    key = tools_eval._review_key(fn.to_dict(), tools_eval._review_settings(rt.config)[2], None)
    allocate_review(rt.run.run_dir(), fn.name, rt.dataset.items, target_n=4,
                    seed=31, review_key=key, phase="development")
    frozen = development_allocation(rt, fn)
    if review_mode == "diagnostic":
        # A diagnostic review samples separately without mutating the frozen operation.
        diagnostic = tools_eval.judge_function(fn.name, review_mode="diagnostic")
        assert diagnostic["ok"] and diagnostic["n_judged"] == 3, diagnostic
        assert development_allocation(rt, fn) == frozen
        review_mode = "standard"
    # An automatic review resumes the frozen standard operation before deferring.
    resumed = tools_eval.judge_function(fn.name, review_mode=review_mode)
    assert resumed["ok"] and resumed["n_judged"] == 4, resumed
    assert [row["item_id"] for row in resumed["verdicts"]] == frozen["item_ids"]
    count = len(calls)
    repeated = tools_eval.judge_function(fn.name, review_mode=review_mode)
    assert repeated["ok"] and repeated["status"] == "review_already_completed", repeated
    assert len(calls) == count


def test_diagnostic_counterexamples_allow_grounded_repair_without_standard_review(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    monkeypatch.setattr(claude_judge, "_run_review", judge_with(RECOVERED))
    response = tools_eval.judge_function(fn.name, review_mode="diagnostic")
    assert response["ok"] and response["n_no"] == 3, response
    feedback = build_repair_feedback(fn.to_dict())
    assert feedback["validation"]["status"] == "unjudged"
    assert feedback["recommended_action"] == "tighten_detector"
    finding = next(row for row in feedback["findings"] if row["status"] == "contradicted")
    assert finding["evidence_scope"] == "bounded_development_diagnostic"
    planned = plan_repair(
        rt.run.run_dir(), fn.to_dict(), feedback, action="tighten_detector",
        rationale="Exclude observed recoveries.", finding_ids=[finding["id"]],
        evidence_ids=finding["item_ids"])
    assert planned["phase"] == "planned"


def test_diagnostic_requires_current_evaluation(family_review):
    rt, fn, calls = family_review
    fn.scores["evaluation_stale"] = True
    response = tools_eval.judge_function(fn.name, review_mode="diagnostic")
    assert not response["ok"] and response["evaluation_required"], response
    assert calls == []


def test_diagnostic_cannot_consume_processing_retry_and_automatic_resumes_it(repair_runtime):
    rt, statuses, calls = repair_runtime
    plan("retry_review", status="processing_error")
    diagnostic = tools_eval.judge_function("guess", review_mode="diagnostic")
    assert not diagnostic["ok"] and "standard mode" in diagnostic["error"], diagnostic
    assert calls == []
    statuses["case-2"] = "supported"
    result = tools_eval.judge_function("guess", review_mode="automatic")
    assert result["ok"], result
    assert set(calls) == {"case-2"}


@pytest.mark.parametrize("change", [None, "rollout"])
def test_diagnostic_review_survives_reevaluation_with_independent_content_freshness(
    family_review, change,
):
    rt, fn, calls = family_review
    response = tools_eval.judge_function(fn.name, review_mode="diagnostic")
    assert response["ok"], response
    original = deepcopy(fn.scores["diagnostic_review"])
    if change == "rollout":
        item_id = original["verdicts"][0]["item_id"]
        row = next(row for row in rt.dataset.items if row["_item_id"] == item_id)
        row["response_text"] += " I then recovered after checking the answer."
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    saved = fn.scores["diagnostic_review"]
    assert saved["claim_hash"] == original["claim_hash"]
    assert [row["item_id"] for row in saved["verdicts"]] == [
        row["item_id"] for row in original["verdicts"]]
    assert bool(fn.scores.get("diagnostic_evidence_stale")) is (change is not None)
    assert (diagnostic_scores(fn.scores) is not None) is (change is None)
    assert "behavior_verdicts" not in fn.scores
    assert len(calls) == 6


def test_automatic_deferral_preserves_scores_until_explicit_standard_review_clears_it(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    before = deepcopy(fn.scores)
    with monkeypatch.context() as patch:
        # Automatic deferral happens before any scan or candidate screen.
        patch.setattr(tools_eval, "_iter_scored_items",
                      lambda *args, **kw: pytest.fail("deferred review must not scan"))
        patch.setattr("autofa.session.candidate_screen.ensure_candidate_screen",
                      lambda *args, **kw: pytest.fail("deferred review must not screen"))
        deferred = tools_eval.judge_function(fn.name, review_mode="automatic")
    assert deferred["ok"] and deferred["status"] == "review_deferred", deferred
    assert calls == []
    assert fn.scores["review_schedule"]["evaluation_claim_hash"] == before["evaluation_claim_hash"]
    assert fn.scores["review_schedule"]["evaluation_run_id"] == rt.run.run_id
    assert {key: value for key, value in fn.scores.items()
            if key not in {"judge_prefilter", "review_schedule"}} == {
                key: value for key, value in before.items() if key != "judge_prefilter"}
    # An explicit standard review clears the deferral even when the review errors.
    monkeypatch.setattr(claude_judge, "_run_review", lambda **kwargs: "invalid judge response")
    result = tools_eval.judge_function(fn.name, review_mode="standard")
    assert result["ok"] and result["n_errors"] == 4, result
    assert "review_schedule" not in fn.scores


def test_diagnostic_preserves_prior_outcome_counterexample_without_score_pollution(
    family_review, tmp_path, monkeypatch,
):
    rt, fn, calls = family_review
    _prepare_all_pass_sample(rt, fn, tmp_path / "preview")
    monkeypatch.setattr(claude_judge, "_run_review",
                        judge_with(RECOVERED, batch="outcome_diagnostic"))
    assert tools_eval.judge_function(fn.name)["ok"]
    old = deepcopy(fn.scores["diagnostic_review"])
    assert old["n_no"] == 1
    before = deepcopy(fn.scores)
    for attempt in range(2):
        call_count = len(calls)
        result = tools_eval.judge_function(fn.name, review_mode="diagnostic")
        assert result["ok"] and result["n_judged"] == 3, result
        saved = fn.scores["diagnostic_review"]
        ids = [row["item_id"] for row in saved["verdicts"]]
        assert set(row["item_id"] for row in old["verdicts"]) <= set(ids)
        assert len(ids) == len(set(ids)) == saved["n_judged"]
        assert saved["n_no"] >= 1
        assert saved["sample"].get("latest_cohort_n", saved["n_judged"]) <= 3
        assert fn.scores["behavior_verdicts"] == before["behavior_verdicts"]
        assert fn.scores["behavior_purity"] == before["behavior_purity"]
        assert fn.scores["n_confirmed_failed_tasks"] == before["n_confirmed_failed_tasks"]
        assert build_repair_feedback(fn.to_dict())["recommended_action"] == "tighten_detector"
        if attempt:
            assert len(calls) == call_count
    retained_ids = {row["item_id"] for row in fn.scores["diagnostic_review"]["verdicts"]}
    # Rechecking a frozen standard cohort must preserve the separate diagnostic
    # cases, including when the main review's content freshness needs recovery.
    fn.scores["behavior_evidence_stale"] = True
    checked = tools_eval.judge_function(fn.name, review_mode="standard")
    assert checked["ok"] and checked["n_judged"] == 4, checked
    assert {row["item_id"] for row in fn.scores["diagnostic_review"]["verdicts"]} == retained_ids
    assert fn.scores["diagnostic_review"]["sample"]["latest_cohort_review_mode"] == (
        "outcome_diagnostic")
    assert fn.scores["diagnostic_review"]["n_no"] >= 1
    assert fn.scores["behavior_purity"] == before["behavior_purity"]
    assert fn.scores["n_confirmed_failed_tasks"] == before["n_confirmed_failed_tasks"]
    count = len(calls)
    completed = tools_eval.judge_function(fn.name, review_mode="standard")
    assert completed["status"] == "review_already_completed", completed
    assert len(calls) == count


# One bounded adaptive expansion keeps source scope and every original finding.


@pytest.mark.parametrize("value", [
    None, {**REQUEST, "reason": "reach_good"}, {**REQUEST, "question": ""},
    {**REQUEST, "preserve_original_findings": False},
])
def test_malformed_or_unsupported_requests_cannot_authorize_expansion(value):
    assert normalize_sample_request(value) is None


def test_processing_or_unfinished_reads_take_precedence_over_sampling():
    result = {"n_uncertain": 1, "verdicts": [{"additional_sample_request": REQUEST}]}
    assert sample_request(result) == REQUEST
    assert sample_request({**result, "n_errors": 1}) is None
    assert sample_request({**result, "n_review_incomplete": 1}) is None
    # A completed case can still leave a specific question about other tasks.
    assert sample_request({**result, "n_uncertain": 0, "n_yes": 1}) == REQUEST


def test_combined_result_retains_unknown_and_counterexample_and_counts_every_item():
    original = {
        "verdicts": [{"item_id": "a", "verdict": "uncertain", "rationale": "missing metadata"},
                     {"item_id": "b", "verdict": "no", "rationale": "later recovery"}],
        "claim_hash": "claim", "judge_policy_hash": "policy", "judge_model": "model",
        "n_judged": 2, "behavior_specificity": 0.4,
    }
    additional = {
        **original, "verdicts": [{"item_id": "c", "verdict": "yes"}],
        "n_judged": 1, "behavior_specificity": 1.,
    }
    before = deepcopy(original)
    merged = merge_expanded_review(original, additional)
    assert merged["verdicts"][:2] == original["verdicts"]
    assert merged["n_uncertain"] == merged["n_no"] == merged["n_yes"] == 1
    assert merged["behavior_purity"] == pytest.approx(1 / 3)
    assert merged["behavior_specificity"] == pytest.approx(.6)
    assert original == before
    with pytest.raises(ValueError, match="replace original"):
        merge_expanded_review(original, original)
    with pytest.raises(ValueError, match="frozen judge_policy_hash"):
        merge_expanded_review(original, {**additional, "judge_policy_hash": "changed"})


def test_real_judge_expands_once_preserves_uncertain_and_reuses_all_checkpoints(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    review = judge_with(UNREAD, first_only=True, request=REQUEST)
    monkeypatch.setattr(claude_judge, "_run_review", review)
    judged = tools_eval.judge_function(fn.name)
    assert judged["ok"], judged
    scores = fn.scores
    info = scores["behavior_sample"]["judge_expansion"]
    assert info["status"] == "complete"
    assert info["initial_target_n"] == 4
    assert scores["behavior_n_judged"] == 8
    assert scores["behavior_n_uncertain"] == 1
    assert scores["behavior_n_yes"] == 7
    assert scores["behavior_purity"] == pytest.approx(7 / 8)
    unknown = next(row for row in scores["behavior_verdicts"] if row["verdict"] == "uncertain")
    assert unknown["item_id"] == review.changed[0]
    # One unknown primary, seven confirmed cases, and a separate passing control.
    prompts = review.prompts
    assert len(calls) == len(prompts) == 17
    assert all('"batch": "initial"' in prompt and '"cohort_n": 4' in prompt
               and '"expansion_available": true' in prompt for prompt in prompts[:7])
    assert all('"batch": "additional"' in prompt and '"cohort_n": 8' in prompt
               and '"expansion_available": false' in prompt for prompt in prompts[7:15])
    assert all('"batch": "outcome_diagnostic"' in prompt for prompt in prompts[15:])
    assert scores["diagnostic_review"]["n_judged"] == 1
    before = deepcopy(scores["behavior_verdicts"])
    allocated = set(info["initial_item_ids"] + info["added_item_ids"])
    rt.dataset.items.reverse()
    # A repeat cannot create another batch, even with every reviewer requesting it.
    repeated = tools_eval.judge_function(fn.name)
    assert repeated["ok"], repeated
    assert repeated["status"] == "review_already_completed"
    assert repeated["judge_feedback"]
    assert len(calls) == 17
    assert fn.scores["behavior_verdicts"] == before
    assert {row["item_id"] for row in fn.scores["behavior_verdicts"]} == allocated


def test_interrupted_additional_batch_resumes_frozen_cohort_without_rejudging_original(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    requesting = judge_with(request=REQUEST)
    interrupt = [True]

    def interrupted(**kwargs):
        if interrupt[0] and len(calls) >= 10:
            raise ConnectionResetError("interrupted during additional review")
        return requesting(**kwargs)

    monkeypatch.setattr(claude_judge, "_run_review", interrupted)
    result = tools_eval.judge_function(fn.name)
    assert not result["ok"] and result["resumable"], result
    frozen = development_allocation(rt, fn)
    assert frozen["sample_info"]["judge_expansion"]["initial_target_n"] == 4
    assert len(frozen["item_ids"]) == 8
    assert len(calls) == 10
    interrupt[0] = False
    rt.dataset.items.reverse()
    recovered = tools_eval.judge_function(fn.name)
    assert recovered["ok"], recovered
    assert len(calls) == 18  # scored expansion plus one separately reviewed passing control
    assert {row["item_id"] for row in fn.scores["behavior_verdicts"]} == set(frozen["item_ids"])
    again = tools_eval.judge_function(fn.name)
    assert again["ok"] and again["status"] == "review_already_completed"
    assert len(calls) == 18


def test_additional_review_never_scores_outside_existing_llm_panel(family_review, monkeypatch):
    rt, fn, calls = family_review
    template = rt.dataset.items[1]
    # Every third item forms the saved LLM evaluation panel. Its three-case cohort
    # expands once, to six tasks, without scoring any item outside that panel.
    rt.dataset.items = [
        {**template, "_item_id": f"panel-{index}", "problem_id": f"task-{index}"}
        for index in range(90)
    ]
    rt._family_failed_task_inventory = None
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    panel = {row["_item_id"] for row in rt.dataset.items[::3]}
    fn.scores.update(sampled=True, evaluation_item_ids=sorted(panel))
    monkeypatch.setattr(tools_eval, "is_llm_detector", lambda _: True)
    monkeypatch.setattr(tools_eval, "_llm_backend_error", lambda: None)
    original_scoring = tools_eval._iter_scored_items
    scored = []

    def scoring(pairs, **kwargs):
        def observed():
            for function, item in pairs:
                scored.append(item["_item_id"])
                yield function, item
        return original_scoring(observed(), **kwargs)

    monkeypatch.setattr(tools_eval, "_iter_scored_items", scoring)
    monkeypatch.setattr(claude_judge, "_run_review", judge_with(request=REQUEST))
    result = tools_eval.judge_function(fn.name)
    assert result["ok"], result
    assert set(scored) == panel
    assert set(calls) <= panel
    sample = fn.scores["behavior_sample"]
    assert sample["scan_scope"] == "evaluated_llm_panel"
    assert sample["judge_expansion"]["initial_target_n"] == 3
    assert sample["actual_n"] == 6
    assert fn.scores["evaluation_item_ids"] == sorted(panel)


def test_finished_unknown_reviews_are_completed_operations_with_actionable_repeat_feedback(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    monkeypatch.setattr(claude_judge, "_run_review", judge_with(UNREAD))
    judged = tools_eval.judge_function(fn.name)
    assert judged["ok"], judged
    assert fn.scores["behavior_n_uncertain"] == 4
    assert not fn.scores["behavior_n_review_incomplete"]
    assert fn.scores["behavior_purity"] is None
    calls_before = list(calls)
    repeated = tools_eval.judge_function(fn.name)
    assert repeated["ok"] and repeated["status"] == "review_already_completed"
    assert repeated["judge_feedback"]["uncertain"] == 4
    assert calls == calls_before


def test_additional_provider_failure_preserves_original_results_and_pending_allocation(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    monkeypatch.setattr(claude_judge, "_run_review", judge_with(request=REQUEST))
    judge = claude_judge.judge_behavior_with_claude
    attempts = []

    def unavailable_additional_batch(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 2:
            raise RuntimeError("provider unavailable for the additional batch")
        return judge(**kwargs)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", unavailable_additional_batch)
    result = tools_eval.judge_function(fn.name)
    assert result["ok"], result
    assert fn.scores["behavior_n_yes"] == fn.scores["behavior_n_judged"] == 4
    sample = fn.scores["behavior_sample"]
    assert sample["judge_expansion"]["status"] == "blocked"
    assert sample["actual_n"] == 4 and sample["target_n"] == 8
    assert not development_allocation(rt, fn)["operation_completed"]
    original = deepcopy(fn.scores["behavior_verdicts"])
    assert len(calls) == 10  # original scored cohort and the separate passing control
    recovered = tools_eval.judge_function(fn.name)
    assert recovered["ok"], recovered
    assert fn.scores["behavior_verdicts"][:4] == original
    assert fn.scores["behavior_n_judged"] == 8
    assert fn.scores["behavior_sample"]["judge_expansion"]["status"] == "complete"
    assert len(calls) == 18


# Outcome-selected diagnostic controls never become adherence observations.


def _prepare_all_pass_sample(rt, fn, path):
    seed = int(rt.run.exploration_seed or 0) ^ (zlib.crc32(fn.name.encode()) & 0xFFFFFFFF)
    identity = ("_item_id", "benchmark", "problem_id", "task_id", "instance_id", "question_id")
    projected = [{key: row[key] for key in identity if key in row} for row in rt.dataset.items]
    base = allocate_review(
        path, fn.name, projected, target_n=4, seed=seed, review_key="preview")
    # Outcome metadata is hidden from the judge and does not select the scored cohort.
    for row in rt.dataset.items:
        row["is_correct"] = row["_item_id"] in base["item_ids"]
    rt._family_failed_task_inventory = None
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    return base


def test_real_supplement_is_actionable_but_never_changes_adherence_or_support(
    family_review, tmp_path, monkeypatch,
):
    rt, fn, calls = family_review
    base = _prepare_all_pass_sample(rt, fn, tmp_path / "preview")
    review = judge_with(RECOVERED, batch="outcome_diagnostic")
    monkeypatch.setattr(claude_judge, "_run_review", review)
    judged = tools_eval.judge_function(fn.name)
    assert judged["ok"], judged
    assert judged["n_judged"] == 4
    assert judged["behavior_purity"] == 1
    assert {row["item_id"] for row in judged["verdicts"]} == set(base["item_ids"])
    assert len(review.changed) == 1
    supplemental = fn.scores["diagnostic_review"]
    assert supplemental["n_no"] == 1
    assert supplemental["excluded_from_adherence"] is True
    assert supplemental["excluded_from_confirmed_support"] is True
    assert fn.scores["n_confirmed_failed_tasks"] == 0
    feedback = build_repair_feedback(fn.to_dict())
    assert feedback["review_detail_complete"] is True
    assert feedback["recommended_action"] == "tighten_detector"
    assert feedback["diagnostic_feedback"]["validation"]["rejected"] == 1
    assert any(row["evidence_scope"] == "outcome_diagnostic"
               for row in feedback["findings"] if row.get("evidence_scope"))
    assert judged["judge_feedback"]["diagnostic_feedback"]["rejected"] == 1
    # Completed operation reuses both original and supplemental checkpoints.
    count = len(calls)
    repeated = tools_eval.judge_function(fn.name)
    assert repeated["status"] == "review_already_completed"
    assert len(calls) == count
    assert fn.scores["diagnostic_review"]["verdicts"] == supplemental["verdicts"]
    assert repeated["diagnostic_review"]["n_no"] == 1
    finding = next(row for row in feedback["diagnostic_feedback"]["findings"]
                   if row["status"] == "contradicted")
    planned = plan_repair(
        rt.run.run_dir(), fn.to_dict(), feedback, action="tighten_detector",
        rationale="Exclude the recovered supplemental counterexample.",
        finding_ids=[finding["id"]], evidence_ids=finding["item_ids"])
    assert planned["phase"] == "planned"


@pytest.mark.parametrize("failure", ["interrupted", "provider"])
def test_failed_supplement_retains_primary_judgments_and_resumes_same_ids(
    family_review, tmp_path, monkeypatch, failure,
):
    rt, fn, calls = family_review
    base = _prepare_all_pass_sample(rt, fn, tmp_path / "preview")
    fail = [True]
    if failure == "interrupted":
        run_review = claude_judge._run_review

        def interrupt_diagnostic(**kwargs):
            if fail[0] and '"batch": "outcome_diagnostic"' in kwargs["prompt"]:
                raise ConnectionResetError("interrupted supplemental review")
            return run_review(**kwargs)

        monkeypatch.setattr(claude_judge, "_run_review", interrupt_diagnostic)
    else:
        judge = claude_judge.judge_behavior_with_claude

        def unavailable(**kwargs):
            if fail[0] and kwargs["review_batch_context"]["batch"] == "outcome_diagnostic":
                raise RuntimeError("Supplemental provider unavailable")
            return judge(**kwargs)

        monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", unavailable)
    result = tools_eval.judge_function(fn.name)
    assert len(calls) == 8
    frozen = development_allocation(rt, fn)
    assert frozen["item_ids"] == base["item_ids"]
    assert len(frozen["diagnostic_item_ids"]) == 1
    assert not frozen["operation_completed"]
    if failure == "interrupted":
        assert not result["ok"] and result["resumable"], result
    else:
        # A provider failure keeps the completed primary scores and a bounded retry.
        assert result["ok"] and result["behavior_purity"] == 1, result
        assert result["n_judged"] == 4
        assert result["sample"]["diagnostic_sample"]["status"] == "blocked"
    fail[0] = False
    rt.dataset.items.reverse()
    resumed = tools_eval.judge_function(fn.name)
    assert resumed["ok"], resumed
    assert len(calls) == 10
    assert resumed["n_judged"] == 4
    assert resumed["diagnostic_review"]["n_judged"] == 1
    assert [row["item_id"] for row in resumed["diagnostic_review"]["verdicts"]] == (
        frozen["diagnostic_item_ids"])


def test_supplemental_source_change_is_separately_stale(family_review, tmp_path):
    rt, fn, calls = family_review
    _prepare_all_pass_sample(rt, fn, tmp_path / "preview")
    assert tools_eval.judge_function(fn.name)["ok"]
    saved = deepcopy(fn.scores)
    assert diagnostic_scores(saved) is not None
    for key in ("behavior_judgment_stale", "behavior_policy_stale", "behavior_evidence_stale"):
        assert diagnostic_scores({**saved, key: True}) is None
    assert diagnostic_scores({
        **saved, "diagnostic_review": {**saved["diagnostic_review"], "evidence_stale": True},
    }) is None
    item_id = saved["diagnostic_review"]["verdicts"][0]["item_id"]
    row = next(row for row in rt.dataset.items if row["_item_id"] == item_id)
    row["response_text"] += " The assistant then recovered."
    retained = RetainedReviewContent([fn])
    for item in rt.dataset.items:
        retained.observe(item)
    current = retained.apply(fn.name, saved)
    assert current["diagnostic_evidence_stale"] is True
    assert current["diagnostic_review"]["evidence_stale"] is True
    assert current["behavior_evidence_stale"] is False
    assert current["behavior_purity"] == saved["behavior_purity"]
    assert diagnostic_scores(current) is None
    assert build_repair_feedback({**fn.to_dict(), "scores": current})[
        "recommended_action"] == "gather_evidence"
    count = len(calls)
    refreshed = tools_eval.judge_function(fn.name)
    assert refreshed["ok"], refreshed
    assert len(calls) == count + 2  # original scored cases remain cached
    assert fn.scores["behavior_verdicts"] == saved["behavior_verdicts"]
    assert fn.scores["diagnostic_review"]["verdicts"][0]["snapshot_id"] != (
        saved["diagnostic_review"]["verdicts"][0]["snapshot_id"])


def _empty_panel_with_smoke_controls(rt, fn):
    # The unchanged current detector matches author controls outside a two-row panel.
    rt.dataset.items[0]["response_text"] = "The assistant derives the answer."
    rt.dataset.items[1]["response_text"] = "The assistant computes the answer."
    assert tools_eval.smoke_test_feature(
        fn.source, item_ids=["case-2", "case-3", "case-4", "case-6"],
        expected_outputs={"case-2": True},
    )["ok"]
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    rt.run.eval_limit = 2
    fn.scores.update(
        evaluated=True, sampled=True, eval_mode="capped", evaluation_scope_complete=False,
        n_items=2, n_scored=2, n_flagged=0, a=0, b=0, c=1, d=1,
        recall=0, fpr=0, ppv_at_pi0=0, recall_ci_low=0, recall_ci_high=.95,
        fpr_ci_low=0, fpr_ci_high=.95, assessment_coverage=1,
        evaluation_item_ids=["case-0", "case-1"],
    )
    return fn.scores.copy()


def test_off_panel_smoke_matches_are_diagnostic_only_and_resume(family_review):
    rt, fn, calls = family_review
    original = _empty_panel_with_smoke_controls(rt, fn)
    judged = tools_eval.judge_function(fn.name)
    assert judged["ok"], judged
    assert judged["n_judged"] == 0
    assert judged["behavior_purity"] is None
    diagnostic = fn.scores["diagnostic_review"]
    assert diagnostic["n_judged"] == 3
    assert diagnostic["n_yes"] == 3
    assert diagnostic["sample"]["purpose"] == "known_smoke_matches"
    assert diagnostic["sample"]["selected_development_evidence"] is True
    assert diagnostic_scores(fn.scores) is not None
    assert fn.scores["behavior_verdicts"] == []
    assert not fn.scores["n_confirmed_failed_tasks"]
    for key in ("recall", "n_flagged", "a", "b", "c", "d", "evaluation_item_ids"):
        assert fn.scores[key] == original[key]
    assert judged["verdict"]["label"] != "GOOD"
    previous_calls = len(calls)
    rt.dataset.items.reverse()
    repeated = tools_eval.judge_function(fn.name)
    assert repeated["ok"], repeated
    assert repeated["status"] == "review_already_completed"
    assert len(calls) == previous_calls


@pytest.mark.parametrize("change", ["source", "evidence", "expected_only", "task_repeat"])
def test_known_smoke_controls_verify_actual_predictions_and_current_identity(
    family_review, change,
):
    rt, fn, _ = family_review
    for row in rt.dataset.items:
        row["is_correct"] = False
        if change == "task_repeat":
            row["problem_id"] = "one-task"
        elif change == "expected_only":
            row["response_text"] = "I compute the answer."
    rt._family_failed_task_inventory = None
    result = tools_eval.smoke_test_feature(
        fn.source, item_ids=["case-6", "case-7", "case-8"],
        expected_outputs={key: True for key in ["case-6", "case-7", "case-8"]},
    )
    assert result["ok"], result
    if change == "source":
        fn.source += "\n# Current repaired revision.\n"
    elif change == "evidence":
        for row in rt.dataset.items:
            row["response_text"] += " Later recovery."
    controls = known_failed_smoke_matches(rt, fn, client=None)
    assert len(controls) == (1 if change == "task_repeat" else 0)


def test_named_pinned_controls_remain_available_for_diagnostic_review(family_review):
    from autofa.core.dependencies import build_bundle
    from autofa.core.library import FeatureFunction

    rt, fn, _ = family_review

    def pinned(source):
        peer = FeatureFunction(name="peer", description="Peer", source=source)
        bundle = build_bundle(FunctionLibrary(functions=[peer]), depends_on=["peer"])
        return {"deps": {"peer": source}, "dependency_bundle": bundle}

    peer_source = fn.source
    fn.meta = pinned(peer_source)
    fn.source = ('def detect(item):\n'
                 '    text = item["response_text"]\n'
                 '    return library.fire("peer", item) and "final" in text\n')
    result = tools_eval.smoke_test_feature(name=fn.name, item_ids=["case-6", "case-7"])
    assert result["ok"]
    controls = known_failed_smoke_matches(rt, fn, client=None, frozen_ids=["case-6", "case-7"])
    assert {row["_item_id"] for row, value in controls if value is True} == {"case-6", "case-7"}
    fn.meta = pinned(peer_source + "\n# A different implementation must be checked again.\n")
    assert known_failed_smoke_matches(rt, fn, client=None, frozen_ids=["case-6", "case-7"]) == []


def test_off_panel_interruption_keeps_frozen_cohort(family_review, monkeypatch):
    rt, fn, calls = family_review
    _empty_panel_with_smoke_controls(rt, fn)
    review = claude_judge._run_review
    monkeypatch.setattr(claude_judge, "_run_review", interrupt_third_review(review))
    result = tools_eval.judge_function(fn.name)
    assert not result["ok"] and result["resumable"], result
    frozen = development_allocation(rt, fn, ":smoke-only")
    assert frozen["item_ids"] == []
    assert len(frozen["diagnostic_item_ids"]) == 3
    # Extra known positives cannot replace or enlarge an interrupted operation.
    assert tools_eval.smoke_test_feature(fn.source, item_ids=["case-9"])["ok"]
    monkeypatch.setattr(claude_judge, "_run_review", review)
    # Even a full rescan, which finds the off-panel matches, cannot reopen the cohort.
    rt.run.eval_limit = len(rt.dataset.items)
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    rt.dataset.items.reverse()
    resumed = tools_eval.judge_function(fn.name)
    assert resumed["ok"], resumed
    assert {row["item_id"] for row in resumed["diagnostic_review"]["verdicts"]} == set(
        frozen["diagnostic_item_ids"])
    assert len(calls) == 6


def test_automatic_submission_defers_empty_panel_then_explicit_review_resumes(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    _empty_panel_with_smoke_controls(rt, fn)
    # Keep the already-computed empty panel as the submission's evaluation; the
    # real judge, persistence, allocation, and checkpoint code run below.
    monkeypatch.setattr(tools_eval, "evaluate_functions", lambda **kwargs: {"ok": True})
    submitted = tools_library.submit_function(
        name=fn.name, source=fn.source, description=fn.description,
        mechanism=fn.mechanism, auto_evaluate=True,
    )
    assert submitted["ok"], submitted
    assert submitted["judge_refresh"]["status"] == "review_deferred"
    assert "diagnostic_review" not in rt.library.get(fn.name).scores
    assert calls == []
    # A deliberate standard review retains the existing smoke-control workflow.
    explicit = tools_eval.judge_function(fn.name, review_mode="standard")
    assert explicit["ok"], explicit
    assert rt.library.get(fn.name).scores["diagnostic_review"]["n_judged"] == 3
    assert explicit["judge_feedback"]["diagnostic_feedback"]["sampled"] == 3
    assert len(calls) == 6
    restored = Runtime(dataset=rt.dataset,
                       library=FunctionLibrary.load(rt.working_library_path()),
                       run=rt.run, config=rt.config)
    set_runtime(restored)
    repeated = tools_library.submit_function(
        name=fn.name, source=fn.source, description=fn.description,
        mechanism=fn.mechanism, auto_evaluate=True,
    )
    assert repeated["ok"], repeated
    assert len(calls) == 6
    assert restored.library.get(fn.name).scores["behavior_purity"] is None


def test_completed_smoke_diagnostic_revalidates_detector_model(family_review, monkeypatch):
    rt, fn, calls = family_review
    _empty_panel_with_smoke_controls(rt, fn)
    assert tools_eval.judge_function(fn.name)["ok"]
    previous_calls = len(calls)
    monkeypatch.setattr(tools_eval, "_llm_adapter",
                        lambda: SimpleNamespace(model="changed-detector-model"))
    result = tools_eval.judge_function(fn.name)
    assert not result["ok"]
    assert result["stale_diagnostic_controls"] is True
    assert len(calls) == previous_calls


def test_empty_unstarted_cohort_allows_later_smoke_matches(family_review):
    rt, fn, _ = family_review
    _empty_panel_with_smoke_controls(rt, fn)
    # Unindexed old smoke hits are used only after a freshness-checked reuse indexes them.
    for path in (rt.run.run_dir() / "smoke_results").glob("index-*.json"):
        path.unlink()
    assert known_failed_smoke_matches(rt, fn, client=None) == []
    first = tools_eval.judge_function(fn.name)
    assert first["ok"] and first["n_judged"] == 0, first
    assert development_allocation(rt, fn) is None
    reused = tools_eval.smoke_test_feature(fn.source, item_ids=["case-2", "case-3", "case-4"])
    assert reused["ok"] and reused["reused"] == 3, reused
    second = tools_eval.judge_function(fn.name)
    assert second["ok"] and second["diagnostic_review"]["n_judged"] == 3, second


def test_completed_controls_do_not_freeze_later_representative_review(family_review):
    rt, fn, _ = family_review
    _empty_panel_with_smoke_controls(rt, fn)
    first = tools_eval.judge_function(fn.name)
    assert first["ok"] and first["n_judged"] == 0, first
    rt.run.eval_limit = len(rt.dataset.items)
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    reviewed = tools_eval.judge_function(fn.name)
    assert reviewed["ok"] and reviewed["n_judged"] > 0, reviewed
    assert reviewed["behavior_purity"] == 1
    assert fn.scores["diagnostic_review"]["excluded_from_adherence"] is True


def test_smoke_limit_is_applied_after_reserved_and_represented_task_exclusion(family_review):
    rt, fn, _ = family_review
    ids = ["case-6", "case-7", "case-8", "case-9", "case-11", "case-12"]
    assert tools_eval.smoke_test_feature(fn.source, item_ids=ids)["ok"]
    excluded = {task_group(row) for row in rt.dataset.items if row["_item_id"] in ids[:3]}
    found = known_failed_smoke_matches(rt, fn, client=None, excluded_task_groups=excluded)
    assert {row["_item_id"] for row, _ in found} == set(ids[3:])


def test_submitted_repair_with_empty_panel_defers_until_explicit_development_review(
    repair_runtime, monkeypatch,
):
    rt, statuses, calls = repair_runtime
    attempt = plan("tighten_detector")
    fn = rt.library.get("guess")
    # Declared LLM kind exercises the repaired detector's bounded panel path;
    # this deterministic implementation keeps the regression provider-free.
    first = tools_library.submit_function(
        name=fn.name, source=TIGHTER, description=fn.description, mechanism=fn.mechanism,
        kind="llm", repair_plan_id=attempt["plan_id"], auto_evaluate=False,
    )
    assert first["ok"], first
    fn = rt.library.get("guess")
    _empty_panel_with_smoke_controls(rt, fn)
    statuses.clear()
    statuses["default"] = "supported"
    initialize_family_discovery(rt)
    calls.clear()
    monkeypatch.setattr(tools_eval, "evaluate_functions", lambda **kwargs: {"ok": True})
    submitted = tools_library.submit_function(
        name=fn.name, source=fn.source, description=fn.description, mechanism=fn.mechanism,
        kind="llm", repair_plan_id=attempt["plan_id"], auto_evaluate=True,
    )
    assert submitted["ok"], submitted
    assert submitted["judge_refresh"]["status"] == "review_deferred"
    assert calls == []
    explicit = tools_eval.judge_function(fn.name, review_mode="standard")
    assert explicit["ok"], explicit
    scores = rt.library.get(fn.name).scores
    assert scores["diagnostic_review"]["n_judged"] == 3
    assert scores["behavior_purity"] is None
    state = workflow(rt)["latest"]
    assert state["phase"] == "awaiting_validation"
    assert state["candidates"][fn.name]["reservation"] is None
    assert state["candidates"][fn.name]["outcome"] is None
    assert len(calls) == 6
