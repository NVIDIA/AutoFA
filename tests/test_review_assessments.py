# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completed requests and migrated operations cannot replace another cohort's assessment."""

import json
from copy import deepcopy

import pytest

from autofa.mcp import tools_eval, tools_repair
from autofa.metrics import claude_judge, final_review
from autofa.metrics.review_assessments import (
    archived_assessments,
    assessment_summary,
    cohort_ids,
    completed_assessment,
    retain_assessment,
)
from autofa.session.family_discovery import initialize_family_discovery
from autofa.session.review_evidence import (
    allocate_review,
    find_review_allocation,
    record_discovery_exposure,
)

from support.evaluation import forbid, judge_rows, reply, runtime
from support.repair_review import _finish_family_comparisons, plan, submit, workflow
from support.repair_review import family_review as family_review
from support.repair_review import repair_runtime as repair_runtime


def _uncertain(*, review, **kwargs):
    result = json.loads(reply(review, status="unknown"))
    for assessment in result["verdicts"][0]["claim_assessments"]:
        assessment["evidence_inspection"] = {
            "status": "complete", "reason": "Inspected evidence remains ambiguous.",
            "ranges": [{"field": "response_text", "start": 0,
                        "end": len(review.fields["response_text"])}],
        }
    return json.dumps(result)


@pytest.mark.parametrize("legacy,review_phase", [(False, "auto"), (True, "development")])
def test_default_replay_keeps_newer_five_case_assessment_even_when_score_is_lower(
    family_review, monkeypatch, legacy, review_phase,
):
    rt, fn, _ = family_review
    monkeypatch.setattr(tools_eval, "judge_evidence_sample_size",
                        lambda n, *, evidence_n=None: min(n, evidence_n or 1))
    monkeypatch.setattr(claude_judge, "_run_review",
                        lambda **kw: reply(kw["review"], status="supported"))
    first = tools_eval.judge_function(fn.name)
    assert first["ok"] and first["n_judged"] == 1, first
    first_ids = cohort_ids(fn.scores)
    monkeypatch.setattr(claude_judge, "_run_review",
                        lambda **kw: reply(kw["review"], status="contradicted"))
    expanded = tools_eval.judge_function(fn.name, evidence_n=5)
    assert expanded["ok"] and expanded["n_judged"] == 5, expanded
    assert fn.scores["behavior_purity"] < 1  # Never select whichever score is higher.
    assert archived_assessments(fn.scores)
    history = deepcopy(fn.scores["behavior_review_assessments"])
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert fn.scores["behavior_review_assessments"] == history
    if legacy:
        fn.scores.pop("behavior_review_assessments")
        fn.scores.pop("behavior_review_operation", None)
    before = deepcopy(fn.scores)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    monkeypatch.setattr(tools_eval, "_iter_scored_items", forbid)
    # Both phases name the same default-size development operation.
    result = tools_eval.judge_function(fn.name, review_phase=review_phase)
    assert result["ok"] and result["status"] == "review_already_completed", result
    assert fn.scores == before
    assert result["requested_review"]["item_ids"] == sorted(first_ids)
    assert result["requested_review"]["authoritative"] is False
    assert len(result["authoritative_review"]["item_ids"]) == 5


def _family_candidate(rt, statuses, *, repair_plan):
    # Thirty-two items keep default reviews at three cases.
    rt.dataset.items = rt.dataset.items[:32]
    # Passing attempts stop matching, so the precision schedule does not defer
    # the round checkpoint's representative review.
    for item in rt.dataset.items:
        if item["is_correct"] and item["_item_id"] != "case-0":
            item["response_text"] = "I computed an intermediate value."
    rt._family_failed_task_inventory = None
    record_discovery_exposure(rt.run.run_dir(), rt.dataset.items[:4])
    attempt = None
    if repair_plan:
        attempt = plan("tighten_detector")
        submit(attempt)
    else:
        assert tools_eval.evaluate_functions(names=["guess"], mode="full")["ok"]
    statuses.clear()  # Every later review supports the claim.
    initialize_family_discovery(rt)
    return rt.library.get("guess"), attempt


def _by_item(*statuses):
    """Reply by item index modulo 3; "malformed" is a reviewer processing failure and
    "inspected" an unknown requirement after complete evidence inspection."""
    def review(*, review, **kwargs):
        status = statuses[int(review.snapshot["item_id"].split("-")[1]) % 3]
        if status == "malformed":
            return "malformed judge response"
        return _uncertain(review=review) if status == "inspected" else reply(review, status=status)
    return review


@pytest.mark.parametrize(("repair_plan", "newer", "requested_n", "earlier_archive"), [
    (True, "larger_review", None, False), (False, "round_checkpoint", None, True),
    # The newer assessment's phase differs from the replayed development request.
    (False, "confirmation", None, False),
])
def test_completed_review_replay_keeps_newer_assessment_with_or_without_repair_plan(
    repair_runtime, monkeypatch, repair_plan, newer, requested_n, earlier_archive,
):
    rt, statuses, _ = repair_runtime
    fn, attempt = _family_candidate(rt, statuses, repair_plan=repair_plan)
    first = tools_eval.judge_function("guess", evidence_n=requested_n)
    assert first["ok"] and fn.scores["behavior_purity"] == 1, first
    first_ids = cohort_ids(fn.scores)
    if repair_plan:
        assert fn.scores["repair_validation"]["plan_id"] == attempt["plan_id"]
    monkeypatch.setattr(claude_judge, "_run_review",
                        _by_item("inspected", "supported", "supported"))
    if newer == "larger_review":
        assert tools_eval.judge_function("guess", evidence_n=9)["ok"]
    elif newer == "confirmation":
        _finish_family_comparisons(rt)
        assert tools_eval.judge_function("guess", review_phase="confirmation")["ok"]
    else:
        rt.run.exploration_base_seed = 12345  # A representative cohort unlike the first.
        receipt, _ = final_review._finalize_one(rt, fn, scope="round_checkpoint")
        assert receipt["status"] == "complete", receipt
    newer_ids = cohort_ids(fn.scores)
    assert newer_ids != first_ids and fn.scores["behavior_purity"] < 1
    if earlier_archive:
        # Earlier archives kept the pending representative review's aggregate marker.
        next(saved for saved in archived_assessments(fn.scores)
             if cohort_ids(saved) == first_ids)["behavior_evidence_stale"] = True
    before = deepcopy(fn.scores)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    monkeypatch.setattr(tools_eval, "_iter_scored_items", forbid)
    replay = tools_eval.judge_function("guess", evidence_n=requested_n)
    assert replay["ok"] and replay["status"] == "review_already_completed", replay
    assert fn.scores == before
    assert replay["requested_review"]["item_ids"] == sorted(first_ids)
    assert replay["requested_review"]["authoritative"] is False
    assert set(replay["authoritative_review"]["item_ids"]) == newer_ids
    if repair_plan:
        assert workflow(rt)["latest"]["phase"] == "awaiting_validation"
        # The newer assessment is unsupported: the attempt is deferred, not replayed.
        assert replay["next_action"]["tool"] == "defer_function_repair"
        saved = replay["next_action"]["completed_review"]
        assert saved["item_ids"] == sorted(first_ids) and saved["behavior_purity"] == 1
        assert saved["authoritative"] is False


@pytest.mark.parametrize("repair_plan,newer", [
    (True, "unread_evidence"), (False, "failed_review"),
    (True, "failed_checkpoint"), (False, "failed_checkpoint"),
])
def test_completed_review_replay_keeps_newer_unfinished_assessment(
    repair_runtime, monkeypatch, repair_plan, newer,
):
    rt, statuses, _ = repair_runtime
    fn, _ = _family_candidate(rt, statuses, repair_plan=repair_plan)
    assert tools_eval.judge_function("guess")["ok"] and fn.scores["behavior_purity"] == 1
    first_ids = cohort_ids(fn.scores)
    monkeypatch.setattr(claude_judge, "_run_review", _by_item(
        "contradicted", "unknown" if newer == "unread_evidence" else "malformed", "supported"))
    if newer == "failed_checkpoint":
        rt.run.exploration_base_seed = 12345  # A representative cohort unlike the first.
        receipt, _ = final_review._finalize_one(rt, fn, scope="round_checkpoint")
        assert receipt["status"] == "unavailable", receipt
    else:
        assert tools_eval.judge_function("guess", evidence_n=9)["ok"]
    newer_ids = cohort_ids(fn.scores)
    assert newer_ids != first_ids and not completed_assessment(fn.scores)
    before = deepcopy(fn.scores)
    judge, scan = claude_judge.judge_behavior_with_claude, tools_eval._iter_scored_items
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    monkeypatch.setattr(tools_eval, "_iter_scored_items", forbid)
    replay = tools_eval.judge_function("guess")
    # The older complete cohort must not replace an unfinished newer review.
    assert replay["ok"] and replay["status"] == "review_already_completed", replay
    assert fn.scores == before
    assert replay["requested_review"]["item_ids"] == sorted(first_ids)
    assert set(replay["authoritative_review"]["item_ids"]) == newer_ids
    assert replay["authoritative_review"]["complete"] is False
    next_action = replay.get("next_action")
    if newer == "unread_evidence":
        # Saved unread evidence stays unresolved on resume; the findings decide next.
        assert (next_action or {}).get("tool") == (
            "defer_function_repair" if repair_plan else None)
        return
    # Processing failures are recovered through the newer review's own operation.
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    monkeypatch.setattr(tools_eval, "_iter_scored_items", scan)
    retried = []

    def recovered(*, review, **kwargs):
        retried.append(review.snapshot["item_id"])
        return reply(review, status="supported")

    monkeypatch.setattr(claude_judge, "_run_review", recovered)
    if newer == "failed_review":
        assert next_action["arguments"] == {
            "name": "guess", "review_phase": "development", "evidence_n": 9}
        assert tools_eval.judge_function(**next_action["arguments"])["ok"]
    else:
        assert next_action["tool"] == ("defer_function_repair" if repair_plan
                                       else "get_function_repair")
        assert "retry_review" in next_action["reason"]
        if repair_plan:
            assert tools_repair.defer_function_repair(**next_action["arguments"])["ok"]
        finding = next(row for row in tools_repair.get_function_repair("guess")["feedback"][
            "findings"] if row["status"] == "processing_error")
        assert tools_repair.plan_function_repair(
            name="guess", action="retry_review", rationale="Retry the failed review stages.",
            finding_ids=[finding["id"]], evidence_ids=finding["item_ids"])["ok"]
        assert tools_eval.judge_function("guess")["ok"]
    assert set(retried) == {row["item_id"] for row in before["behavior_verdicts"]
                            if row["verdict"] == "error"}
    assert cohort_ids(fn.scores) == newer_ids and completed_assessment(fn.scores)


@pytest.mark.parametrize("failure", ["unavailable", "interrupted", "cannot_start"])
def test_completed_replay_keeps_newer_assessment_left_pending_by_a_failed_checkpoint(
    repair_runtime, monkeypatch, failure,
):
    rt, statuses, _ = repair_runtime
    fn, _ = _family_candidate(rt, statuses, repair_plan=False)
    assert tools_eval.judge_function("guess")["ok"] and fn.scores["behavior_purity"] == 1
    first_ids = cohort_ids(fn.scores)
    monkeypatch.setattr(claude_judge, "_run_review",
                        _by_item("inspected", "supported", "supported"))
    rt.run.exploration_base_seed = 12345  # A representative cohort unlike the first.
    receipt, _ = final_review._finalize_one(rt, fn, scope="round_checkpoint")
    assert receipt["status"] == "complete", receipt
    newer_ids = cohort_ids(fn.scores)
    assert newer_ids != first_ids and fn.scores["behavior_purity"] < 1
    purity = fn.scores["behavior_purity"]
    judge, matches = claude_judge.judge_behavior_with_claude, final_review._matches

    def broken(*args, **kwargs):
        if failure == "interrupted":
            raise KeyboardInterrupt
        raise (ValueError("Saved detector matches are malformed") if failure == "cannot_start"
               else RuntimeError("judge backend rejected the request"))

    if failure == "cannot_start":
        monkeypatch.setattr(final_review, "_matches", broken)
    else:
        monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", broken)
    rt.run.exploration_base_seed = 777  # The next round's checkpoint fails before results.
    try:
        final_review._finalize_one(rt, fn, scope="round_checkpoint")
    except KeyboardInterrupt:
        pass
    monkeypatch.setattr(final_review, "_matches", matches)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    assert cohort_ids(fn.scores) == newer_ids
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    monkeypatch.setattr(tools_eval, "_iter_scored_items", forbid)
    replay = tools_eval.judge_function("guess")
    assert replay["ok"] and replay["status"] == "review_already_completed", replay
    assert cohort_ids(fn.scores) == newer_ids and fn.scores["behavior_purity"] == purity
    # The response still reports the pending checkpoint's assessment as unfinished.
    assert replay["authoritative_review"]["complete"] is False
    assert "unfinished" in replay["message"]


@pytest.mark.parametrize("repair_plan,newer_complete", [(True, True), (False, False)])
def test_resumed_older_review_keeps_newer_checkpoint_authority(
    repair_runtime, monkeypatch, repair_plan, newer_complete,
):
    rt, statuses, _ = repair_runtime
    fn, _ = _family_candidate(rt, statuses, repair_plan=repair_plan)
    monkeypatch.setattr(claude_judge, "_run_review", _by_item("supported", "malformed",
                                                              "supported"))
    assert tools_eval.judge_function("guess")["n_errors"]
    first_ids = cohort_ids(fn.scores)
    monkeypatch.setattr(claude_judge, "_run_review", _by_item(
        "inspected", "supported" if newer_complete else "malformed", "supported"))
    rt.run.exploration_base_seed = 12345  # A representative cohort unlike the first.
    final_review._finalize_one(rt, fn, scope="round_checkpoint")
    newer = deepcopy(fn.scores["behavior_verdicts"])
    assert cohort_ids(fn.scores) != first_ids
    assert completed_assessment(fn.scores) is newer_complete
    monkeypatch.setattr(claude_judge, "_run_review", _by_item("supported", "supported",
                                                              "supported"))
    # Resuming the older unfinished review finishes it without making it current.
    resumed = tools_eval.judge_function("guess")
    assert resumed["status"] == "review_completed_authority_retained", resumed
    assert resumed["requested_review"]["item_ids"] == sorted(first_ids)
    assert resumed["requested_review"]["behavior_purity"] == 1
    assert fn.scores["behavior_verdicts"] == newer
    assert any(cohort_ids(saved) == first_ids and completed_assessment(saved)
               for saved in archived_assessments(fn.scores))
    if repair_plan:
        assert resumed["next_action"]["tool"] == "defer_function_repair"
        assert resumed["next_action"]["completed_review"]["item_ids"] == sorted(first_ids)


def test_archived_assessment_omits_pending_cohort_marker_but_keeps_changed_content():
    row = {"item_id": "a", "verdict": "yes", "completion_status": "complete"}
    scores = {"claim_hash": "claim", "behavior_verdicts": [row], "behavior_evidence_stale": True}
    shown = assessment_summary(scores)["assessment_id"]
    retain_assessment(scores)
    assert scores["behavior_evidence_stale"] is True  # The live view is unchanged.
    assert completed_assessment(archived_assessments(scores)[-1])
    # The id shown for the live assessment pages its archived copy.
    assert scores["behavior_review_assessments"][-1]["assessment_id"] == shown
    changed = {**row, "verification_status": "historical", "historical_review": deepcopy(row)}
    scores = {"claim_hash": "claim", "behavior_verdicts": [changed],
              "behavior_evidence_stale": True}
    shown = assessment_summary(scores)["assessment_id"]
    retain_assessment(scores)
    assert not completed_assessment(archived_assessments(scores)[-1])
    assert scores["behavior_review_assessments"][-1]["assessment_id"] == shown


@pytest.mark.parametrize("legacy", [False, True])
def test_frozen_round_uncertainty_restores_own_cohort_after_separate_standard_review(
    tmp_path, monkeypatch, stub_claim_compiler, legacy,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(tools_eval, "judge_evidence_sample_size", lambda n, **kw: min(n, 4))
    monkeypatch.setattr(claude_judge, "_run_review", _uncertain)
    receipt, _ = final_review._finalize_one(rt, fn, scope="round_checkpoint")
    assert receipt["status"] == "unavailable"
    frozen = deepcopy(receipt)
    frozen_rows = deepcopy(fn.scores["behavior_verdicts"])
    assert completed_assessment(fn.scores)
    corpus = rt.dataset._fake_items
    rt.dataset._fake_items = [item for item in corpus if item["_item_id"] not in frozen["item_ids"]]
    judge_rows(rt, fn, monkeypatch, count=4, runner=_uncertain)
    manual_ids = cohort_ids(fn.scores)
    rt.dataset._fake_items = corpus
    assert manual_ids.isdisjoint(frozen["item_ids"])
    assert fn.scores["round_review"] == frozen
    if legacy:
        fn.scores.pop("behavior_review_assessments", None)
        # Actual saved CharXiv shape after the old mismatched-cohort early return.
        fn.scores["behavior_evidence_stale"] = True
    else:
        monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    restored, _ = final_review._finalize_one(rt, fn, scope="round_checkpoint")
    assert restored["operation_id"] == frozen["operation_id"]
    assert restored["item_ids"] == frozen["item_ids"]
    assert fn.scores["behavior_verdicts"] == frozen_rows
    assert fn.scores["behavior_n_uncertain"] == 4
    assert fn.scores["behavior_n_errors"] == fn.scores["behavior_n_review_incomplete"] == 0
    assert not fn.scores.get("behavior_evidence_stale")
    assert completed_assessment(fn.scores)
    assert {row["item_id"] for row in fn.scores["diagnostic_review"]["verdicts"]} == manual_ids
    from autofa.metrics.final_validation_selection import prepare_final_validation

    selection = prepare_final_validation(rt)
    assert selection["ok"] and selection["skipped"] and not selection["selected_names"]


def test_archived_assessment_never_reuses_changed_source_content(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(claude_judge, "_run_review", _uncertain)
    receipt, _ = final_review._finalize_one(rt, fn, scope="round_checkpoint")
    corpus = rt.dataset._fake_items
    rt.dataset._fake_items = [item for item in corpus
                             if item["_item_id"] not in receipt["item_ids"]]
    judge_rows(rt, fn, monkeypatch, count=3, runner=_uncertain)
    rt.dataset._fake_items = corpus
    changed = next(item for item in corpus if item["_item_id"] == receipt["item_ids"][0])
    changed["response_text"] = "BAD changed original content"
    calls = []

    def reviewed(**kwargs):
        calls.append(kwargs["review"].snapshot["item_id"])
        return reply(kwargs["review"], status="contradicted")

    monkeypatch.setattr(claude_judge, "_run_review", reviewed)
    final_review._finalize_one(rt, fn, scope="round_checkpoint")
    assert calls and set(calls) == {changed["_item_id"]}
    assert fn.scores["behavior_n_no"] == 1
    assert fn.scores["behavior_n_uncertain"] == 2


def test_obsolete_family_review_remains_historical_and_requires_current_review(
    family_review, monkeypatch,
):
    rt, fn, calls = family_review
    obsolete = "26ff5754dbbdba019fbb069e07abb5b9228680154fd52b30e69b5c79d3486dec"
    settings = tools_eval._review_settings(rt.config)[2]
    with monkeypatch.context() as old:
        old.setattr(claude_judge, "judge_policy_hash", lambda **kwargs: obsolete)
        key = tools_eval._review_key(fn.to_dict(), settings, None)
    allocation = allocate_review(rt.run.run_dir(), fn.name, rt.dataset.items,
                                 target_n=3, seed=31, review_key=key, phase="development")
    original = find_review_allocation(
        rt.run.run_dir(), fn.name, review_key=key, phase="development",
    )
    assert not claude_judge.compatible_judge_policy(obsolete)
    result = tools_eval.judge_function(fn.name)
    assert result["ok"], result
    assert result["sample"]["target_n"] == 4
    assert calls
    assert [r["item_id"] for r in fn.scores["behavior_verdicts"]] != allocation["item_ids"]
    assert fn.scores["judge_policy_hash"] == claude_judge.judge_policy_hash()
    assert find_review_allocation(
        rt.run.run_dir(), fn.name, review_key=key, phase="development",
    ) == original
