# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repair planning, screening, family guards and parked-repair confirmation with judge reviews."""

import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.behavior_policy import acceptance_requirement_id
from autofa.config import RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_eval, tools_library, tools_repair
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics import claude_judge
from autofa.objectives import apply_preset
from autofa.reports.failure_cards import _extract_autofa_data, build_data, render_html
from autofa.session import candidate_screen, repair_workflow, review_evidence
from autofa.session.family_discovery import (
    STATE_FILE,
    family_comparison_pending,
    family_discovery,
    initialize_family_discovery,
)
from autofa.session.recovery import RunInterruptedError
from autofa.session.review_evidence import record_discovery_exposure, review_context
from autofa.util.atomic import atomic_write_json

from support.repair_review import (
    DESCRIPTION,
    MECHANISM,
    SOURCE,
    TIGHTER,
    _finish_family_comparisons,
    plan,
    submit,
    workflow,
)
from support.repair_review import repair_runtime as repair_runtime


def forbid(monkeypatch, *targets):
    """Fail if blocked work reaches a target: a dotted path or an (owner, name) pair."""
    def unexpected(*args, **kwargs):
        pytest.fail("Blocked work reached a paid, scanning or mutating call")

    for target in targets:
        if isinstance(target, str):
            monkeypatch.setattr(target, unexpected)
        else:
            monkeypatch.setattr(*target, unexpected)


def fresh_tasks(rt, n, prefix="fresh"):
    """Matching failed attempts, each on its own new task; resets the task inventory."""
    template = rt.dataset.items[1]
    rt._family_failed_task_inventory = None
    return [{**template, "_item_id": f"{prefix}-{i}", "problem_id": f"{prefix}-{i}"}
            for i in range(n)]


def test_exhausted_unplanned_submission_preserves_saved_review(repair_runtime, monkeypatch):
    rt, _, _ = repair_runtime
    for _ in range(3):
        plan("gather_evidence", status="unknown")
        repair_workflow.defer_repair(rt.run.run_dir(), "guess", "Need more evidence")
    before = deepcopy(rt.library.get("guess").to_dict())
    saved = rt.working_library_path().read_bytes()
    ledger = (rt.run.run_dir() / "detector_repairs.json").read_bytes()

    forbid(monkeypatch, (tools_eval, "evaluate_functions"), (tools_eval, "judge_function"))
    result = tools_library.submit_function(
        name="guess", source=SOURCE, description=DESCRIPTION, mechanism=MECHANISM + " Again.",
        auto_evaluate=True,
    )
    assert not result["ok"] and "new active repair plan" in result["error"]
    assert rt.library.get("guess").to_dict() == before
    assert rt.working_library_path().read_bytes() == saved
    assert (rt.run.run_dir() / "detector_repairs.json").read_bytes() == ledger


def test_tighten_validates_on_fresh_items_and_preserves_original_review(repair_runtime):
    rt, _, calls = repair_runtime
    original = deepcopy(rt.library.get("guess").to_dict())
    attempt = plan("tighten_detector")
    submit(attempt)
    judged = tools_eval.judge_function("guess")
    assert judged["ok"], judged
    reviewed = {row["item_id"] for row in judged["verdicts"]}
    assert reviewed and len(reviewed) == judged["n_judged"]
    assert reviewed.isdisjoint({"case-0", "case-1", "case-2", "case-3"})
    assert set(calls) == reviewed
    state = workflow(rt)
    assert state["baseline"] == original
    assert state["attempts_used"] == 1
    assert state["latest"]["before"]["source"] == SOURCE
    assert state["latest"]["after"]["source"] == TIGHTER
    assert rt.library.get("guess").scores["behavior_n_yes"] == len(reviewed)


@pytest.mark.parametrize("missing_reserved_item", [False, True],
                         ids=["reserved_sample_resumes", "missing_reserved_item"])
def test_interrupted_repair_reuses_reserved_sample_and_completed_reviews(
    repair_runtime, monkeypatch, missing_reserved_item,
):
    rt, _, calls = repair_runtime
    submit(plan("tighten_detector"))
    before_scores = deepcopy(rt.library.get("guess").scores)
    original_review = claude_judge._run_review
    attempts = []

    def disconnect_after_first_item(**kwargs):
        attempts.append(kwargs["review"].snapshot["item_id"])
        if len(attempts) == 3:
            raise ConnectionResetError("connection reset")
        return original_review(**kwargs)

    monkeypatch.setattr(claude_judge, "_run_review", disconnect_after_first_item)
    interrupted = tools_eval.judge_function("guess", evidence_n=4)
    assert not interrupted["ok"] and interrupted["resumable"]
    assert rt.library.get("guess").scores == before_scores
    saved = workflow(rt)
    reservation = saved["latest"]["reservation"]
    reserved_ids = reservation["item_ids"]
    assert reserved_ids and calls == [reserved_ids[0], reserved_ids[0]]

    # Dataset order, a smaller scan cap, and a new sample-size request cannot
    # redraw a reserved repair sample after reconnecting.
    rt.dataset.items.reverse()
    rt.run.eval_limit = 4

    rt.config = apply_preset(rt.config, "recall")
    if missing_reserved_item:
        rt.dataset.items = [item for item in rt.dataset.items
                            if item["_item_id"] != reserved_ids[-1]]
    resumed = tools_eval.judge_function("guess", evidence_n=12)
    if missing_reserved_item:
        assert not resumed["ok"] and "Reserved repair evidence is missing" in resumed["error"]
        assert workflow(rt) == saved
        assert rt.library.get("guess").scores == before_scores
        assert len(attempts) == 3
        return
    assert resumed["ok"], resumed
    assert [row["item_id"] for row in resumed["verdicts"]] == reserved_ids
    assert calls == [item_id for item_id in reserved_ids for _ in range(2)]
    assert workflow(rt)["latest"]["reservation"] == reservation
    assert workflow(rt)["attempts_used"] == 1
    assert workflow(rt)["latest"]["outcome"]["all_reviewed_confirmed"]


def test_processing_retry_only_retries_failed_items_and_preserves_semantic_reviews(repair_runtime):
    rt, statuses, calls = repair_runtime
    rt.config.model = "different-analysis-model"
    before = deepcopy(rt.library.get("guess").scores["behavior_verdicts"])
    plan("retry_review", status="processing_error")
    statuses["case-2"] = "supported"
    rt.run.eval_limit = 2  # The failed item lies beyond the initial scan.
    judged = tools_eval.judge_function("guess")
    assert judged["ok"], judged
    assert set(calls) == {"case-2"}
    scores = rt.library.get("guess").scores
    assert scores["behavior_n_judged"] == 4
    assert scores["behavior_n_yes"] == 2
    assert scores["behavior_n_no"] == scores["behavior_n_uncertain"] == 1
    assert scores["behavior_n_errors"] == 0
    retained = {row["item_id"]: row for row in scores["behavior_verdicts"]}
    for row in before:
        if row["item_id"] != "case-2":
            assert retained[row["item_id"]] == row
    assert workflow(rt)["latest"]["outcome"]["original_claim_validated"] is False


def test_processing_retry_remains_available_after_repair_budget_exhaustion(repair_runtime):
    rt, statuses, calls = repair_runtime
    rt.dataset.items = rt.dataset.items[:30]  # Three-case reviews: 29 matches.
    submit(plan("tighten_detector"))
    statuses["default"] = "unknown"
    assert tools_eval.judge_function("guess")["ok"]
    plan("gather_evidence", status="unknown")
    assert tools_eval.judge_function("guess")["ok"]
    plan("gather_evidence", status="unknown")
    statuses["default"] = "error"
    assert tools_eval.judge_function("guess")["ok"]
    assert workflow(rt)["state"] == "exhausted"
    assert workflow(rt)["attempts_used"] == 3
    before = deepcopy(rt.library.get("guess").scores)
    failed_ids = [row["item_id"] for row in before["behavior_verdicts"]]
    feedback = tools_repair.get_function_repair("guess")["feedback"]
    findings = [f["id"] for f in feedback["findings"] if f["status"] == "processing_error"]
    retry = tools_repair.plan_function_repair(
        name="guess", action="retry_review", rationale="Complete the failed processing.",
        finding_ids=findings, evidence_ids=failed_ids,
    )
    assert retry["ok"], retry
    assert workflow(rt)["attempts_used"] == 3
    statuses["default"] = "supported"
    calls.clear()
    rt.library = FunctionLibrary.load(rt.working_library_path())
    judged = tools_eval.judge_function("guess")
    assert judged["ok"], judged
    assert set(calls) == set(failed_ids)
    assert judged["n_yes"] == before["behavior_n_judged"]
    assert judged["n_errors"] == 0
    assert judged["sample"] == before["behavior_sample"]
    assert workflow(rt)["attempts_used"] == 3
    assert workflow(rt)["latest"]["outcome"]["all_reviewed_confirmed"]


@pytest.mark.parametrize("setting,value,provenance", [
    ("behavior_judge_effort", "high", "judge_effort"), (None, None, "judge_thinking"),
])
def test_processing_retry_rejects_changed_or_missing_judge_settings(
    repair_runtime, setting, value, provenance,
):
    rt, _, calls = repair_runtime
    if setting is None:  # Never guess a saved setting that is missing.
        rt.library.get("guess").scores.pop(provenance)
    plan("retry_review", status="processing_error")
    before = deepcopy(rt.library.get("guess").scores)
    if setting:
        setattr(rt.config, setting, value)
    result = tools_eval.judge_function("guess")
    assert not result["ok"] and provenance in result["error"]
    assert calls == []
    assert rt.library.get("guess").scores == before


# The retry gate above compares these saved settings, so failures and rescoring keep them.
def test_failed_judge_batch_records_configured_model_and_effort(repair_runtime, monkeypatch):
    rt, _, calls = repair_runtime
    rt.config.model = "analysis-model"
    rt.config.behavior_judge_model = "custom-judge-model"
    rt.config.behavior_judge_effort = "high"
    rt.dataset.items = rt.dataset.items[:4]

    def fail_batch(**kwargs):
        assert (kwargs["model"], kwargs["effort"]) == ("custom-judge-model", "high")
        raise RuntimeError("Judge batch failed")

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", fail_batch)
    result = tools_eval.judge_function("guess")
    assert result["ok"], result
    assert result["n_errors"] == result["n_judged"] == 3
    assert calls == []
    for payload in (result, rt.library.get("guess").scores):
        assert payload["judge_model"] == "custom-judge-model"
        assert payload["judge_effort"] == "high"
        assert payload["judge_thinking"] == "adaptive"


def test_rescoring_preserves_judge_harness_and_allows_processing_retry(repair_runtime):
    rt, statuses, _ = repair_runtime
    rt.config.behavior_judge_harness = "codex"  # A non-default judge harness.
    rt.config.behavior_judge_model = "gpt-6.1-sol"
    rt.dataset.items = rt.dataset.items[:4]
    judged = tools_eval.judge_function("guess", evidence_n=4)
    assert judged["ok"] and judged["n_errors"] == 1, judged
    before = deepcopy(rt.library.get("guess").scores)
    assert tools_eval.evaluate_functions(names=["guess"], mode="full")["ok"]
    rt.library = FunctionLibrary.load(rt.working_library_path())
    scores = rt.library.get("guess").scores
    assert (scores["judge_harness"], scores["judge_backend"]) == ("codex", "codex_app_server")
    assert scores["behavior_verdicts"] == before["behavior_verdicts"]
    plan("retry_review", status="processing_error")
    statuses["case-2"] = "supported"
    retried = tools_eval.judge_function("guess")
    assert retried["ok"] and retried["n_errors"] == 0, retried
    assert retried["n_judged"] == before["behavior_n_judged"]


def test_crashed_processing_retry_preserves_frozen_contract_and_semantic_rows(
    repair_runtime, monkeypatch,
):
    rt, _, calls = repair_runtime
    before = deepcopy(rt.library.get("guess").scores)
    plan("retry_review", status="processing_error")

    def fail_batch(**kwargs):
        assert kwargs["compiled_contract"] == before["claim_contract"]
        assert [item["_item_id"] for item in kwargs["items"]] == ["case-2"]
        raise RuntimeError("The judge batch crashed before producing a result.")

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", fail_batch)
    result = tools_eval.judge_function("guess")
    assert result["ok"], result
    assert calls == []
    current = rt.library.get("guess").scores
    assert current["claim_contract"] == before["claim_contract"]
    assert current["claim_compilation"] == before["claim_compilation"]
    assert all(current[key] == before[key] for key in (
        "behavior_n_judged", "behavior_n_yes", "behavior_n_no", "behavior_n_uncertain",
        "behavior_n_errors", "behavior_purity",
    ))
    retained = {row["item_id"]: row for row in current["behavior_verdicts"]}
    for row in before["behavior_verdicts"]:
        if row["item_id"] != "case-2":
            assert retained[row["item_id"]] == row
    assert retained["case-2"]["verdict"] == "error"
    assert "judge batch crashed" in retained["case-2"]["validation_error"]
    restored = FunctionLibrary.load(rt.working_library_path()).get("guess").scores
    assert restored["claim_contract"] == before["claim_contract"]
    assert restored["behavior_verdicts"] == current["behavior_verdicts"]


def test_unrecordable_validation_does_not_publish_yes_now_or_after_reevaluation(
    repair_runtime, monkeypatch,
):
    rt, _, calls = repair_runtime
    submit(plan("tighten_detector"))
    before = deepcopy(rt.library.get("guess").scores)

    def reject_recording(*args, **kwargs):
        raise OSError("The reserved review could not be validated.")

    monkeypatch.setattr("autofa.session.repair_workflow.record_validation", reject_recording)
    result = tools_eval.judge_function("guess")
    assert not result["ok"]
    assert calls
    assert rt.library.get("guess").scores == before
    attempted = json.loads(Path(result["attempted_review_path"]).read_text())
    assert attempted["result"]["n_yes"] == attempted["result"]["n_judged"] > 0
    assert "reserved review" in attempted["error"]
    rt.library = FunctionLibrary.load(rt.working_library_path())
    assert rt.library.get("guess").scores == before
    evaluated = tools_eval.evaluate_functions(names=["guess"], mode="full")
    assert evaluated["ok"], evaluated
    after = rt.library.get("guess").scores
    for key in ("behavior_n_yes", "behavior_verdicts", "claim_contract", "judge_backend"):
        assert after.get(key) == before.get(key)


def test_ordinary_rejudge_does_not_reuse_repair_sample_metadata(repair_runtime):
    rt, _, _ = repair_runtime
    submit(plan("tighten_detector"))
    assert tools_eval.judge_function("guess")["ok"]
    archived = deepcopy(workflow(rt))
    assert rt.library.get("guess").scores["repair_validation"]
    response = tools_eval.judge_function("guess")
    assert response["ok"], response
    assert "repair_validation" not in rt.library.get("guess").scores
    assert workflow(rt) == archived


def test_failed_library_save_preserves_prior_json_and_restores_scores(repair_runtime, monkeypatch):
    import os

    rt, _, calls = repair_runtime
    submit(plan("tighten_detector"))
    before = deepcopy(rt.library.get("guess").scores)
    saved = rt.working_library_path().read_bytes()
    replace = os.replace

    def fail_library_replace(source, destination):
        if Path(destination) == rt.working_library_path():
            raise OSError("The library could not be saved.")
        return replace(source, destination)

    monkeypatch.setattr("autofa.util.atomic.os.replace", fail_library_replace)
    response = tools_eval.judge_function("guess")
    assert not response["ok"] and "could not be saved" in response["error"]
    assert calls and rt.library.get("guess").scores == before
    assert rt.working_library_path().read_bytes() == saved
    assert not list(rt.run.run_dir().glob(".library.json.*.tmp"))
    # The valid completed review remains auditable despite the library save failure.
    assert workflow(rt)["latest"]["outcome"]["all_reviewed_confirmed"]


def test_review_does_not_overwrite_concurrent_score_updates(repair_runtime, monkeypatch):
    rt, _, _ = repair_runtime
    submit(plan("tighten_detector"))
    original = claude_judge.judge_behavior_with_claude

    def another_evaluation(**kwargs):
        result = original(**kwargs)
        rt.library.get("guess").scores["concurrent_evaluation"] = True
        rt.persist_library()
        return result

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", another_evaluation)
    response = tools_eval.judge_function("guess")
    assert not response["ok"] and "changed while" in response["error"]
    assert rt.library.get("guess").scores["concurrent_evaluation"] is True
    assert not rt.library.get("guess").scores.get("behavior_n_yes")
    assert workflow(rt)["latest"]["result"] is None


def test_cancelled_repair_review_does_not_commit_late_judge_result(repair_runtime, monkeypatch):
    import asyncio

    from autofa.util.cancellation import Cancellation, cancellation_scope

    rt, _, _ = repair_runtime
    submit(plan("tighten_detector"))
    before = deepcopy(rt.library.get("guess").scores)
    saved = rt.working_library_path().read_bytes()
    cancellation = Cancellation()

    def cancelled_review(**kwargs):
        cancellation.cancel()
        # A response racing with cancellation must never become persisted scores.
        return {"behavior_purity": 1.0, "n_judged": len(kwargs["items"])}

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", cancelled_review)
    with cancellation_scope(cancellation), pytest.raises(asyncio.CancelledError):
        tools_eval.judge_function("guess")

    assert rt.library.get("guess").scores == before
    assert rt.working_library_path().read_bytes() == saved
    assert workflow(rt)["latest"]["result"] is None
    assert workflow(rt)["latest"]["reservation"] is not None  # Resumable frozen cohort.


def test_repair_refuses_duplicate_item_identity_before_judging(repair_runtime):
    rt, _, calls = repair_runtime
    submit(plan("tighten_detector"))
    before = deepcopy(rt.library.get("guess").scores)
    rt.dataset.items[5]["_item_id"] = "case-4"
    response = tools_eval.judge_function("guess")
    assert not response["ok"], response
    assert "item IDs" in response["error"]
    assert calls == [] and rt.library.get("guess").scores == before
    assert workflow(rt)["latest"]["reservation"] is None


def test_partial_validation_scores_only_fresh_evidence(repair_runtime):
    from autofa.metrics.pareto import point_from_scores
    from autofa.metrics.verdicts import behavior_validation

    rt, _, calls = repair_runtime
    attempt = plan("tighten_detector")
    rt.dataset.items = rt.dataset.items[:12]
    baseline = deepcopy(workflow(rt)["baseline"])
    submit(attempt)
    # Eight fresh matches remain once the four baseline reviews are excluded.
    response = tools_eval.judge_function("guess", evidence_n=10)
    assert response["ok"], response
    assert response["n_judged"] == response["sample"]["actual_n"] == 8
    assert response["sample"]["eligible_n"] == 8
    assert response["sample"]["target_n"] == 10
    assert response["sample"]["limited"] and response["sample"]["reason"]
    assert response["n_scanned"] == 12
    assert len(set(calls)) == 8
    assert set(calls).isdisjoint({f"case-{i}" for i in range(4)})
    assert response["behavior_purity"] == 1.0
    state = workflow(rt)
    assert state["baseline"] == baseline
    assert state["attempts_used"] == 1
    assert state["latest"]["outcome"]["review_status"] == "insufficient_new_evidence"
    assert not state["latest"]["outcome"]["all_reviewed_confirmed"]
    # Saving, reloading, and association rescoring must not promote partial reviews.
    rt.library = FunctionLibrary.load(rt.working_library_path())
    assert tools_eval.evaluate_functions(names=["guess"], mode="full")["ok"]
    scores = rt.library.get("guess").scores
    assert scores["behavior_sample"] == response["sample"]
    validation = behavior_validation(scores)
    assert validation["status"] == "limited_evidence"
    assert not validation["ranking_eligible"]
    assert point_from_scores(scores, ["behavior_purity", "behavior_specificity"]) is None


def test_exhausted_fresh_pool_rejudges_matches_without_validating_repair(repair_runtime):
    rt, statuses, calls = repair_runtime
    attempt = plan("tighten_detector")
    baseline = deepcopy(workflow(rt)["baseline"])
    rt.dataset.items = rt.dataset.items[:4]
    submit(attempt)
    statuses.clear()
    judged = tools_eval.judge_function("guess")
    assert judged["ok"], judged
    assert set(calls) == {"case-1", "case-2", "case-3"}
    assert judged["behavior_purity"] == 1.0
    assert judged["sample"]["eligible_n"] == 0
    assert judged["sample"]["reused_n"] == judged["n_judged"] == 3
    assert judged["judge_feedback"]["status"] == "reused_evidence"
    assert not judged["judge_feedback"]["ranking_eligible"]
    state = workflow(rt)
    assert state["baseline"] == baseline
    assert state["latest"]["reservation"]["sample_kind"] == "reused_review_evidence"
    assert state["latest"]["outcome"]["review_status"] == "reused_evidence"
    assert not state["latest"]["outcome"]["all_reviewed_confirmed"]
    assert not state["latest"]["outcome"]["original_claim_validated"]


def test_save_preserves_completed_inconclusive_reviews(repair_runtime):
    rt, statuses, calls = repair_runtime
    submit(plan("tighten_detector"))
    statuses.clear()
    statuses["default"] = "unknown"
    judged = tools_eval.judge_function("guess")
    assert judged["ok"] and judged["n_uncertain"] == judged["n_judged"] > 0
    assert judged["n_no"] == judged["n_yes"] == 0
    # Uncertain validation stays unresolved, never false hits.
    assert judged["behavior_purity"] is None
    outcome = workflow(rt)["latest"]["outcome"]
    assert outcome["review_status"] == "uncertain"
    assert outcome["automatic_promotion"] is False
    before = deepcopy(rt.library.get("guess").to_dict())
    ledger = (rt.run.run_dir() / "detector_repairs.json").read_bytes()
    calls.clear()
    for _ in range(2):
        assert tools_library.library_save(to_catalog=False)["ok"]
    assert calls == []
    assert rt.library.get("guess").to_dict() == before
    assert FunctionLibrary.load(rt.working_library_path()).get("guess").to_dict() == before
    assert (rt.run.run_dir() / "detector_repairs.json").read_bytes() == ledger
    raw = build_data(rt.library)
    assert raw["functions"][0]["behavior_validation"]["status"] == "uncertain"
    data = _extract_autofa_data(render_html(raw))
    # The saved library retains incomplete reviews, while report cards show
    # only OK-or-better findings. Saving must not rejudge to force visibility.
    assert data["meta"]["n_functions"] == 0
    assert data["meta"]["n_hidden_functions"] == 1
    assert data["meta"]["n_pending_scores"] == 0


def record_scored_items(monkeypatch):
    """Record, in order, the item IDs that detector scoring yields."""
    scored = []
    original = tools_eval._iter_scored_items

    def score(pairs, **kwargs):
        for result in original(pairs, **kwargs):
            scored.append(result[1]["_item_id"])
            yield result

    monkeypatch.setattr(tools_eval, "_iter_scored_items", score)
    return scored


@pytest.mark.parametrize("eval_limit,scanned,judged", [(6, 64, 7), (16, 16, 3)],
                         ids=["fresh_evidence_shortage", "enough_fresh_evidence"])
def test_repair_review_expands_initial_scan_only_for_a_fresh_evidence_shortage(
    repair_runtime, monkeypatch, eval_limit, scanned, judged,
):
    rt, statuses, calls = repair_runtime
    submit(plan("tighten_detector"))
    statuses.clear()
    rt.run.eval_limit = eval_limit  # Six items hold only two unseen fires: below three.
    rt.config.max_eval_limit = len(rt.dataset.items)
    scored = record_scored_items(monkeypatch)

    def batches(*, max_items=None):
        items = rt.dataset.items[:max_items]
        for offset in range(0, len(items), 4):
            yield items[offset:offset + 4]

    monkeypatch.setattr(rt.dataset, "iter_batches", batches)
    response = tools_eval.judge_function("guess")

    assert response["ok"], response
    assert response["n_judged"] == judged  # Ten percent of the matches, at least three.
    assert response["n_scanned"] == response["scan_limit"] == scanned
    assert scored == [item["_item_id"] for item in rt.dataset.items[:scanned]]
    assert set(calls).isdisjoint({f"case-{i}" for i in range(4)})
    state = workflow(rt)
    assert state["attempts_used"] == 1
    assert state["latest"]["outcome"]["all_reviewed_confirmed"]


@pytest.mark.parametrize("tracked", [False, True])
def test_llm_repair_review_reuses_evaluated_panel_without_expanding_scan(
    repair_runtime, monkeypatch, tracked,
):
    rt, statuses, calls = repair_runtime
    attempt = plan("tighten_detector")
    # A 12-item LLM panel (200 by default) in the 64-item corpus: neither a fresh-evidence
    # shortage nor a request larger than the panel may expand the repair review's scan.
    rt.config.llm_eval_sample = 12
    submitted = tools_library.submit_function(
        name="guess", source=TIGHTER, description=DESCRIPTION, mechanism=MECHANISM,
        kind="llm", repair_plan_id=attempt["plan_id"], auto_evaluate=False,
    )
    assert submitted["ok"], submitted
    detector_calls = []
    original = tools_eval.run_detector

    def detect(source, item, **kwargs):
        detector_calls.append(item["_item_id"])
        return original(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", detect)
    assert tools_eval.evaluate_functions(names=["guess"])["ok"]
    panel = set(rt.library.get("guess").scores["evaluation_item_ids"])
    assert len(panel) == len(detector_calls) == 12
    if tracked:
        initialize_family_discovery(rt)
    statuses.clear()

    response = tools_eval.judge_function("guess", evidence_n=40)

    assert response["ok"], response
    assert response["n_scanned"] == response["scan_limit"] == 12
    assert response["sample"]["scan_scope"] == "evaluated_llm_panel"
    assert calls and set(calls) <= panel
    assert len(detector_calls) == 12  # Repair review only reads detector checkpoints.
    target = response["sample"]["target_n"]
    assert response["sample"]["limited"] == (response["n_judged"] < target)
    if tracked:
        # Explicit all-match requests cannot consume reserved confirmation tasks.
        assert response["sample"]["limited"]
        assert target == tools_eval.judge_evidence_sample_size(response["n_fires"], evidence_n=40)
        assert 0 < response["n_judged"] <= target
        assert workflow(rt)["latest"]["phase"] == "awaiting_validation"


def test_repair_scan_respects_maximum_and_reports_limited_review(repair_runtime):
    rt, _, calls = repair_runtime
    submit(plan("tighten_detector"))
    rt.run.eval_limit = 9
    rt.config.max_eval_limit = 10
    response = tools_eval.judge_function("guess", evidence_n=9)
    assert response["ok"], response
    assert response["n_scanned"] == response["scan_limit"] == 10
    assert response["sample"]["eligible_n"] == 6
    assert response["sample"]["target_n"] == 9
    assert response["n_judged"] == 6
    assert len(set(calls)) == 6
    assert set(calls).isdisjoint({f"case-{i}" for i in range(4)})
    assert response["judge_feedback"]["status"] == "limited_evidence"
    assert workflow(rt)["latest"]["outcome"]["review_status"] == "insufficient_new_evidence"


def test_submission_surfaces_repair_evidence_shortage(repair_runtime):
    rt, _, calls = repair_runtime
    attempt = plan("tighten_detector")
    rt.dataset.items = rt.dataset.items[:6]
    response = tools_library.submit_function(
        name="guess", source=TIGHTER, description=DESCRIPTION, mechanism=MECHANISM,
        repair_plan_id=attempt["plan_id"],
    )
    assert response["ok"] and response["evaluated"], response
    assert response["judge_refresh"]["deferred"] and calls == []
    next_action = tools_repair.get_function_repair("guess")["next_action"]
    assert next_action["tool"] == "defer_function_repair"
    assert "measured grade gaps" in next_action["reason"]
    # This small, low-precision repair needs a deliberate standard review.
    # Its new reservation still reports the original shortage honestly.
    reviewed = tools_eval.judge_function("guess", review_mode="standard")
    assert reviewed["ok"]
    assert reviewed["judge_feedback"]["status"] == "limited_evidence"
    assert reviewed["judge_feedback"]["sample"]["target_n"] == 3
    assert reviewed["judge_feedback"]["sample"]["actual_n"] == 2
    assert len(set(calls)) == 2
    comparison = response["coverage_comparison"]
    assert comparison["failure_coverage"]["n_failed"] == 4
    assert comparison["objective"]["settings_frozen"] is True
    assert comparison["automatic_promotion"] is False


def test_repair_coverage_reporting_failure_does_not_fail_completed_submission(
    repair_runtime, monkeypatch,
):
    rt, _, _ = repair_runtime
    attempt = plan("tighten_detector")

    def unavailable(name):
        raise OSError("Identity index unavailable for coverage reporting")

    monkeypatch.setattr(tools_repair, "get_function_repair", unavailable)
    response = tools_library.submit_function(
        name="guess", source=TIGHTER, description=DESCRIPTION, mechanism=MECHANISM,
        repair_plan_id=attempt["plan_id"],
    )
    assert response["ok"] and response["evaluated"], response
    assert rt.library.get("guess").source == TIGHTER
    assert response["coverage_comparison"]["comparison_complete"] is False
    assert "Identity index unavailable" in response["coverage_comparison"]["reason"]


def test_mcp_polishing_plan_exposes_new_scope_without_resetting_discovery_attempts(repair_runtime):
    rt, _, calls = repair_runtime
    for _ in range(3):
        plan("tighten_detector")
        assert tools_repair.defer_function_repair("guess", "Deferred tested approach")["ok"]
    original = workflow(rt)
    identifier = f"{rt.run.run_id}:{rt.run.round}"
    atomic_write_json(rt.run.run_dir() / "round_validation.json", {"refinement": {
        "version": 1, "run_id": rt.run.run_id, "round": rt.run.round, "mode": "polishing",
        "status": "awaiting_response", "polishing_id": identifier, "frozen_names": ["guess"],
    }})
    exposed = tools_repair.get_function_repair("guess")
    assert exposed["repair_budget"]["scope"] == "polishing_round"
    assert exposed["repair_budget"]["remaining"] == 3
    fourth = plan("tighten_detector")
    assert fourth["attempt"] == 4 and fourth["polishing_id"] == identifier
    assert fourth["attempt_budget"]["remaining"] == 2
    assert workflow(rt)["history"][:3] == original["history"]
    assert workflow(rt)["reviewed_item_ids"] == original["reviewed_item_ids"]
    assert calls == []


def _roll_over_to_polishing(rt):
    _finish_family_comparisons(rt)
    rt.run.round += 1
    initialize_family_discovery(rt)
    assert family_comparison_pending(rt.run.run_dir())
    atomic_write_json(rt.run.run_dir() / "round_validation.json", {"refinement": {
        "version": 1, "mode": "polishing", "run_id": rt.run.run_id, "round": rt.run.round,
        "status": "awaiting_response", "frozen_names": [fn.name for fn in rt.library.functions],
        "polishing_id": f"{rt.run.run_id}:{rt.run.round}", "target_grade": "GOOD",
    }})


def test_family_development_reserves_tasks_then_confirms_without_reusing_feedback(repair_runtime):
    rt, statuses, calls = repair_runtime
    statuses.clear()  # Every later review supports the claim.
    initialize_family_discovery(rt)
    development = tools_eval.judge_function("guess")
    assert development["ok"], development
    assert development["sample"]["phase"] == "development"
    assert development["sample"]["limited"]
    dev_ids = {row["item_id"] for row in development["verdicts"]}
    calls.clear()
    _roll_over_to_polishing(rt)  # Confirmation remains available in the next polishing round.
    confirmation = tools_eval.judge_function("guess", review_phase="confirmation")
    assert confirmation["ok"], confirmation
    assert confirmation["sample"]["phase"] == "confirmation"
    assert not confirmation["sample"]["limited"]
    confirmed_ids = {row["item_id"] for row in confirmation["verdicts"]}
    assert not dev_ids & confirmed_ids
    assert set(calls) == confirmed_ids
    assert dev_ids | confirmed_ids <= set(review_context(rt.run.run_dir(), "guess")[
        "reviewed_item_ids"
    ])
    assert rt.library.get("guess").scores["behavior_review_phase"] == "confirmation"


def test_completed_allocation_cannot_return_scores_from_another_profile(
    repair_runtime, monkeypatch,
):
    rt, _statuses, calls = repair_runtime
    fn = rt.library.get("guess")
    saved_profile = fn.scores["judge_behavior_strictness"]
    rt.config.behavior_strictness = "causal" if saved_profile != "causal" else "behavioral"
    initialize_family_discovery(rt)
    fn.scores["behavior_review_phase"] = "development"
    ids = [row["item_id"] for row in fn.scores["behavior_verdicts"]]
    monkeypatch.setattr(review_evidence, "find_review_allocation", lambda *args, **kwargs: {
        "operation_completed": True, "item_ids": ids, "sample_info": {"target_n": len(ids)},
    })
    screens = []

    def block(*args, **kwargs):
        screens.append(True)
        return {"status": "inconclusive", "evaluation_allowed": False}

    monkeypatch.setattr(candidate_screen, "ensure_candidate_screen", block)
    response = tools_eval.judge_function("guess", review_phase="development")
    assert response.get("status") != "review_already_completed"
    assert response["candidate_screen_blocked"]
    assert screens == [True]
    assert calls == []
    assert fn.scores["judge_behavior_strictness"] == saved_profile


def test_family_development_review_does_not_complete_a_repair_plan(repair_runtime):
    rt, statuses, _ = repair_runtime
    # Mimic metadata retained by the shared tool boundary when the baseline
    # counterexample was inspected. The repaired detector excludes that example.
    record_discovery_exposure(rt.run.run_dir(), rt.dataset.items[:4])
    attempt = plan("tighten_detector")
    submit(attempt)
    statuses.clear()  # Every later review supports the claim.
    initialize_family_discovery(rt)
    development = tools_eval.judge_function("guess")
    assert development["ok"], development
    state = workflow(rt)["latest"]
    assert state["phase"] == "awaiting_validation"
    assert state["candidates"]["guess"]["reservation"] is None
    assert state["candidates"]["guess"]["outcome"] is None
    _finish_family_comparisons(rt)
    confirmation = tools_eval.judge_function("guess", review_phase="confirmation")
    assert confirmation["ok"], confirmation
    state = workflow(rt)["latest"]
    assert state["phase"] == "completed", state
    assert state["outcome"]["all_reviewed_confirmed"]
    assert state["reservation"]["sample"]["phase"] == "confirmation"


def test_single_task_confirmation_skips_paid_review_and_retains_diagnostics(repair_runtime):
    rt, statuses, calls = repair_runtime
    # A new detector cohort has many attempts but only one actual task.
    rt.dataset.items = [{**row, "_item_id": f"same-{i}", "problem_id": "one-task"}
                        for i, row in enumerate(rt.dataset.items)]
    statuses.clear()  # Every later review supports the claim.
    initialize_family_discovery(rt)
    development = tools_eval.judge_function("guess")
    assert development["ok"], development
    before = deepcopy(rt.library.get("guess").scores)
    calls.clear()
    _finish_family_comparisons(rt)
    for _ in range(2):
        confirmation = tools_eval.judge_function("guess", review_phase="confirmation")
        assert confirmation["ok"] and confirmation["status"] == "confirmation_unavailable"
        assert confirmation["verdict"] == development["verdict"]
    assert calls == []
    assert rt.library.get("guess").scores["behavior_verdicts"] == before["behavior_verdicts"]
    assert rt.library.get("guess").scores["behavior_review_phase"] == "development"


@pytest.mark.parametrize("failure", ["interrupted", "processing_error"])
def test_confirmation_recovery_keeps_original_cohort_before_extending_scan(
    repair_runtime, monkeypatch, failure,
):
    rt, statuses, calls = repair_runtime
    rt.dataset.items = fresh_tasks(rt, 120)
    rt.run.eval_limit = 12
    rt.config.max_eval_limit = 120
    original = claude_judge.judge_behavior_with_claude
    cohorts = []

    def capture(**kwargs):
        cohorts.append([row["_item_id"] for row in kwargs["items"]])
        if failure == "interrupted" and len(cohorts) == 1:
            raise RunInterruptedError("Interrupted confirmation")
        return original(**kwargs)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", capture)
    statuses.clear()
    statuses["default"] = "error"
    first = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
    assert len(cohorts[0]) == 10
    assert first.get("resumable") if failure == "interrupted" else first["n_errors"] == 10

    rt.config = apply_preset(rt.config, "recall")
    statuses["default"] = "supported"
    rt.dataset.items.reverse()
    rt.run.eval_limit = 1
    calls.clear()
    resumed = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
    assert resumed["ok"], resumed
    assert cohorts[1] == cohorts[0]
    assert resumed["sample"]["target_n"] == 10
    assert resumed["sample"]["n_scanned"] == 10
    assert set(calls) == set(cohorts[0])
    calls.clear()
    rt.library.get("guess").scores["verdict"] = {"code": "OLD", "label": "OLD"}
    repeated = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
    assert repeated["status"] == "review_already_completed"
    assert repeated["verdict"] == resumed["verdict"]
    assert repeated["scores"]["verdict"] == repeated["verdict"]
    assert rt.library.get("guess").scores["verdict"]["code"] == "OLD"
    assert calls == []


def test_completed_review_refreshes_only_newly_accessible_image_evidence(
    repair_runtime, monkeypatch,
):
    from autofa.data import inline_images

    rt, statuses, calls = repair_runtime
    statuses.clear()  # Every later review supports the claim.
    rt.dataset.items = fresh_tasks(rt, 10)
    # The bytes were retained in the old run too; only their visual availability
    # changes. Text-only members of the same review cohort remain reusable.
    rt.dataset.items[0]["input_messages"] = [{"role": "user", "content": [{
        "type": "image_url", "image_url": {
            "url": "data:image/png;base64,"
                   "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
                   "/x8AAwMCAO+jRZkAAAAASUVORK5CYII=",
        },
    }]}]
    with monkeypatch.context() as legacy:
        legacy.setattr(inline_images, "image_inventory", lambda *args, **kwargs: ([], []))
        first = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
        assert first["ok"], first
        assert {row["item_id"] for row in first["verdicts"]} == {
            row["_item_id"] for row in rt.dataset.items
        }
        calls.clear()
        unchanged = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
        assert unchanged["status"] == "review_already_completed"
        assert calls == []

    original = deepcopy(rt.library.get("guess").scores)
    current = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
    assert current["ok"], current
    assert current.get("status") != "review_already_completed"
    assert calls == ["fresh-0", "fresh-0"]  # primary and independent verification
    assert current["sample"]["target_n"] == first["sample"]["target_n"]
    assert {row["item_id"] for row in current["verdicts"]} == {
        row["item_id"] for row in first["verdicts"]
    }
    scores = rt.library.get("guess").scores
    previous_image = next(row for row in original["behavior_verdicts"]
                          if row["item_id"] == "fresh-0")
    assert previous_image in scores["behavior_verdict_history"]
    previous_text = {row["item_id"]: row["snapshot_id"] for row in original["behavior_verdicts"]
                     if row["item_id"] != "fresh-0"}
    assert {row["item_id"]: row["snapshot_id"] for row in current["verdicts"]
            if row["item_id"] != "fresh-0"} == previous_text
    assert not scores.get("behavior_evidence_stale")


@pytest.mark.parametrize("source_failure", ["missing", "unavailable"])
def test_completed_review_cannot_reuse_unavailable_source(
    repair_runtime, monkeypatch, source_failure,
):
    rt, statuses, calls = repair_runtime
    statuses.clear()  # Every later review supports the claim.
    first = tools_eval.judge_function("guess", review_phase="confirmation")
    assert first["ok"], first
    saved = deepcopy(rt.library.get("guess").to_dict())
    history_path = rt.run.run_dir() / "review_evidence.json"
    history = history_path.read_bytes()
    calls.clear()
    if source_failure == "missing":
        missing_id = first["verdicts"][0]["item_id"]
        rt.dataset.items = [item for item in rt.dataset.items if item["_item_id"] != missing_id]
    else:
        def unavailable(*args, **kwargs):
            raise OSError("Source unavailable")

        monkeypatch.setattr(tools_eval, "_fetch_items_by_id", unavailable)
    repeated = tools_eval.judge_function("guess", review_phase="confirmation")
    assert not repeated["ok"]
    assert "evidence" in repeated["error"]
    assert calls == []
    assert rt.library.get("guess").to_dict() == saved
    assert history_path.read_bytes() == history


def test_continue_does_not_rejudge_imported_completed_confirmation_as_fresh(repair_runtime):
    rt, statuses, calls = repair_runtime
    statuses.clear()  # Every later review supports the claim.
    # Preserve identities for the baseline review before replacing this fixture's
    # source; otherwise its old opaque IDs correctly make independence unknown.
    record_discovery_exposure(rt.run.run_dir(), rt.dataset.items[:4])
    rt.dataset.items = fresh_tasks(rt, 10)
    first = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
    assert first["ok"] and not first["sample"]["limited"]
    calls.clear()
    continued = Runtime(
        dataset=rt.dataset, library=FunctionLibrary.from_dict(rt.library.to_dict()),
        run=RunConfig(run_id="continued", model_id="m", benchmarks=["b"], parquet_paths=[]),
        config=rt.config,
    )
    set_runtime(continued)
    continued.persist_library()
    repeated = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
    assert repeated["status"] == "confirmation_unavailable"
    assert repeated["sample"]["reused_n"] == 10
    assert repeated["sample"]["independent_task_n"] == 0
    assert calls == []


def test_legacy_reserved_repair_cannot_establish_independent_confirmation(repair_runtime):
    rt, statuses, _ = repair_runtime
    statuses.clear()  # Every later review supports the claim.
    attempt = plan("tighten_detector")
    submit(attempt)
    ids = [row["_item_id"] for row in rt.dataset.items[4:14]]
    sample = {"target_n": 10, "actual_n": 10, "reused_n": 0, "limited": False,
              "reason": None, "phase": "confirmation"}
    repair_workflow.reserve_validation(rt.run.run_dir(), "guess", ids,
                       expected_plan_id=attempt["plan_id"], sample=sample)
    initialize_family_discovery(rt)
    rt.library.review_history["history_complete"] = False
    rt.persist_library()
    record_discovery_exposure(rt.run.run_dir(), rt.dataset.items[4:14])
    result = tools_eval.judge_function("guess")
    assert result["ok"], result
    assert result["sample"]["limited"]
    assert result["sample"]["phase"] == "confirmation"
    outcome = workflow(rt)["latest"].get("outcome") or {}
    assert not outcome.get("original_claim_validated")
    assert not outcome.get("all_reviewed_confirmed")


# Repair guards while a family workflow is pending.


def pending_family(rt, *, detector_name="not_submitted"):
    initialize_family_discovery(rt)
    path = rt.run.run_dir() / STATE_FILE
    state = json.loads(path.read_text())
    state.update(phase="assess_family", survey={
        "family_name": "recorded_family", "detector_name": detector_name, "evidence": [],
    })
    path.write_text(json.dumps(state))


def test_active_candidate_rejection_resumes_its_review_before_unrelated_family(repair_runtime):
    rt, statuses, calls = repair_runtime
    attempt = plan("tighten_detector")
    submit(attempt)
    pending_family(rt)
    before = deepcopy(rt.library.get("guess").to_dict())
    saved_plan = workflow(rt)
    result = tools_library.submit_function(
        name="guess", source=TIGHTER + "\n# Another implementation change.\n",
        description=DESCRIPTION, mechanism=MECHANISM,
        repair_plan_id=attempt["plan_id"], auto_evaluate=False,
    )
    assert not result["ok"] and "Candidate is frozen" in result["error"], result
    assert rt.library.get("guess").to_dict() == before
    assert workflow(rt) == saved_plan
    assert calls == []
    next_action = result["next_action"]
    assert next_action["tool"] == "judge_function"
    assert next_action["arguments"] == {"name": "guess", "review_phase": "auto"}
    assert attempt["plan_id"] in next_action["reason"]
    assert "Do not change or duplicate" in next_action["reason"]
    assert next_action["alternative"]["tool"] == "defer_function_repair"
    assert next_action["alternative"]["arguments"]["name"] == "guess"
    # The proposed tool is executable with the frozen candidate, even while the
    # unrelated family remains unsubmitted. No new candidate or plan is needed.
    statuses.clear()  # Every later review supports the claim.
    reviewed = tools_eval.judge_function(**next_action["arguments"])
    assert reviewed["ok"], reviewed
    assert reviewed["n_judged"] > 0 and calls
    assert reviewed["sample"]["phase"] == "development"
    assert workflow(rt)["latest"]["plan_id"] == attempt["plan_id"]


def test_bad_repair_references_point_to_current_findings_without_mutation(repair_runtime):
    rt, _, calls = repair_runtime
    pending_family(rt)
    feedback = tools_repair.get_function_repair("guess")["feedback"]
    selected = next(row for row in feedback["findings"] if row["status"] == "contradicted")
    # Evidence from a different finding than the one selected.
    unselected = next(row for row in feedback["findings"] if row["status"] == "unknown")
    before = deepcopy(rt.library.get("guess").to_dict())
    result = tools_repair.plan_function_repair(
        "guess", "tighten_detector", "Use the cited finding.", [selected["id"]],
        [unselected["item_ids"][0]],
    )
    assert not result["ok"]
    assert result["next_action"]["tool"] == "get_function_repair"
    assert result["next_action"]["arguments"] == {"name": "guess"}
    assert "evidence_ids belonging to the selected findings" in result["next_action"]["reason"]
    assert rt.library.get("guess").to_dict() == before
    assert workflow(rt) == {}
    assert calls == []
    refreshed = tools_repair.get_function_repair(**result["next_action"]["arguments"])
    assert refreshed["ok"]
    assert selected in refreshed["feedback"]["findings"]


def test_active_planned_repair_rejection_exposes_existing_plan_and_deferral(repair_runtime):
    rt, _, calls = repair_runtime
    attempt = plan("tighten_detector")
    pending_family(rt)
    saved_plan = workflow(rt)
    result = tools_repair.plan_function_repair(
        "guess", "gather_evidence", "Start a different investigation.", [], [],
    )
    assert not result["ok"]
    assert result["next_action"]["tool"] == "get_library_function"
    assert result["next_action"]["arguments"] == {"name": "guess"}
    assert attempt["plan_id"] in result["next_action"]["reason"]
    assert "repair_plan_id" in result["next_action"]["reason"]
    assert workflow(rt) == saved_plan and calls == []
    alternative = result["next_action"]["alternative"]
    deferred = tools_repair.defer_function_repair(**alternative["arguments"])
    assert deferred["ok"]
    assert deferred["plan"]["plan_id"] == attempt["plan_id"]
    assert workflow(rt)["state"] == "deferred"


# Family comparison effort is advisory; repair guards and review samples remain.


def eligible_automatic_review_corpus(rt):
    """Supply nonmatching passing controls while retaining the same 63 matches."""
    rt.dataset.items.extend([
        {"_item_id": f"passing-control-{i}", "model_id": "m", "benchmark": "b",
         "problem_id": f"passing-control-{i}", "run_idx": 0, "is_correct": True,
         "response_text": "I derive the result."}
        for i in range(64)
    ])


def record_active_family(rt):
    result = family_discovery(
        action="record_family", family_name="guess", description=DESCRIPTION,
        mechanism=MECHANISM, evidence=[{
            "item_id": f"case-{i}", "role": "boundary" if i == 0 else "support",
            "causal_chain": "The guessed value substitutes for the required derivation.",
            "quotes": [{"field": "response_text", "quote": "I guess"}],
        } for i in (1, 2, 3, 4, 6, 7, 0)],
        boundary_notes="Contrast the intermediate guess with submitted final values.",
    )
    assert result["ok"], result
    assert result["phase"] == "assess_family"


def submit_unscored(**changes):
    return tools_library.submit_function(**{
        "name": "guess", "source": SOURCE, "description": DESCRIPTION,
        "mechanism": MECHANISM, "auto_evaluate": False, **changes,
    })


def attempts():
    return family_discovery()["mutation_attempts"].get("guess", 0)


def test_assessment_links_existing_evaluation_and_review_without_duplicate(
    repair_runtime, monkeypatch,
):
    rt, statuses, review_calls = repair_runtime
    statuses.clear()  # Every later review supports the claim.
    assert tools_eval.evaluate_functions(names=["guess"], mode="full")["ok"]
    assert tools_eval.judge_function("guess")["ok"]
    fn = rt.library.get("guess")
    original = deepcopy(fn.to_dict())
    scores_history = deepcopy(rt.scores_history)
    calls = len(review_calls)
    record_active_family(rt)

    forbid(monkeypatch, (tools_eval, "evaluate_functions"), (tools_eval, "judge_function"),
           (tools_library, "submit_function"))
    assessed = family_discovery(
        "assess_family", family_name="guess", detector_name=fn.name,
        assessment="The saved matches establish breadth; reviewed examples support the claim.",
    )
    assert assessed["ok"], assessed
    assert assessed["assessment"]["detector_name"] == fn.name
    assert assessed["completed_comparisons"] == 1
    assert fn.to_dict() == original
    assert rt.scores_history == scores_history
    assert len(review_calls) == calls
    assert [row.name for row in rt.library.functions] == ["guess"]


def test_failed_compile_and_invalid_repair_plan_do_not_spend_search_allocation(repair_runtime):
    rt, _, _ = repair_runtime
    record_active_family(rt)
    result = submit_unscored(source="def guess(: this is invalid")
    assert not result["ok"] and "compile failed" in result["error"]
    assert attempts() == 0
    result = tools_repair.plan_function_repair(
        name="guess", action="retry_review", rationale="Retry failed processing",
        finding_ids=["invented-finding"], evidence_ids=["invented-item"],
    )
    assert not result["ok"]
    assert attempts() == 0
    assert not (rt.run.run_dir() / "detector_repairs.json").exists()


def test_family_effort_does_not_replace_the_existing_repair_validation_guard(repair_runtime):
    rt, _, _ = repair_runtime
    record_active_family(rt)
    assert submit_unscored()["ok"]
    assert attempts() == 1
    proposed = plan("tighten_detector")
    assert attempts() == 1  # The plan does not double-charge its later submission.
    result = submit_unscored(source=TIGHTER, repair_plan_id=proposed["plan_id"])
    assert result["ok"], result
    assert attempts() == 2
    prior = rt.library.get("guess").to_dict()
    result = submit_unscored(source=TIGHTER + "\n# another expensive iteration")
    assert not result["ok"]
    assert "family_workflow_pending" not in result
    assert "repair" in result["error"].lower()
    assert attempts() == 2
    assert rt.library.get("guess").to_dict() == prior
    assert family_discovery()["phase"] == "assess_family"


def test_confirmation_waits_for_comparisons_then_keeps_normal_sample_target(repair_runtime):
    rt, statuses, calls = repair_runtime
    eligible_automatic_review_corpus(rt)
    statuses.clear()  # Every later review supports the claim.
    record_active_family(rt)
    assert tools_library.submit_function(
        name="guess", source=TIGHTER, description=DESCRIPTION, mechanism=MECHANISM,
    )["ok"]
    # 63 flagged examples: ceil(10%) = 7, without a separate family-search cap.
    assert rt.library.get("guess").scores["behavior_n_judged"] == 7
    calls.clear()
    result = tools_eval.judge_function("guess", review_phase="confirmation")
    assert not result["ok"] and result["family_workflow_pending"]
    assert calls == []
    _finish_family_comparisons(rt)
    result = tools_eval.judge_function("guess", review_phase="confirmation")
    assert result["ok"], result
    assert rt.library.get("guess").scores["behavior_review_phase"] == "confirmation"
    assert rt.library.get("guess").scores["behavior_n_judged"] == 7
    assert len(set(calls)) == 7


def test_family_launch_evaluates_and_judges_without_semantic_prechecks(repair_runtime):
    rt, statuses, review_calls = repair_runtime
    initialize_family_discovery(rt)
    evaluated = tools_eval.evaluate_functions(names=["guess"], mode="capped")
    assert evaluated["ok"], evaluated
    assert review_calls == []  # Evaluation runs no paid semantic precheck.
    statuses.clear()  # Every later review supports the claim.
    judged = tools_eval.judge_function("guess")
    assert judged["ok"], judged
    assert judged["n_judged"] > 0
    assert review_calls  # The unchanged normal causal judge still reviews candidate fires.


# Every expensive evaluation entry point respects a blocked candidate check.


def blocked_screen(*args, **kwargs):
    return {"ok": True, "status": "needs_revision", "evaluation_allowed": False,
            "provisional": True,
            "feedback": "The agent contemplated this action but never executed it."}


def test_direct_evaluation_cannot_bypass_failed_screen(repair_runtime, monkeypatch):
    rt, _, calls = repair_runtime
    before = deepcopy(rt.library.get("guess").scores)
    monkeypatch.setattr(candidate_screen, "ensure_candidate_screen", blocked_screen)

    forbid(monkeypatch, (tools_eval, "_evaluate_streaming"))
    result = tools_eval.evaluate_functions(names=["guess"], mode="stream")
    assert not result["ok"] and result["candidate_screen_blocked"]
    assert result["evaluated_names"] == []
    assert result["candidate_screens"]["guess"]["status"] == "needs_revision"
    assert rt.library.get("guess").scores == before
    assert calls == []


def test_unscored_save_does_not_bypass_later_screen(repair_runtime, monkeypatch):
    calls = []

    def reject(*args, **kwargs):
        calls.append("screen")
        return blocked_screen()

    monkeypatch.setattr(candidate_screen, "ensure_candidate_screen", reject)
    saved = submit_unscored()
    assert saved["ok"] and not saved["evaluated"]
    assert calls == []  # Saving remains a local, reversible operation.
    result = tools_eval.evaluate_functions(names=["guess"], mode="full")
    assert result["candidate_screen_blocked"] and calls == ["screen"]


def test_submit_explains_screen_failure_without_claiming_evaluation(repair_runtime, monkeypatch):
    rt, _, _ = repair_runtime
    monkeypatch.setattr(candidate_screen, "ensure_candidate_screen", blocked_screen)
    result = tools_library.submit_function(
        name="guess", source=SOURCE, description=DESCRIPTION, mechanism=MECHANISM,
    )
    assert result["ok"] and not result["evaluated"]
    assert result["candidate_screen_blocked"]
    assert result["candidate_screen"]["status"] == "needs_revision"
    assert "screen_function" in result["message"]
    assert rt.library.get("guess") is not None


def test_direct_judging_is_blocked_before_any_scan_or_review(repair_runtime, monkeypatch):
    rt, _, calls = repair_runtime
    monkeypatch.setattr(candidate_screen, "ensure_candidate_screen", blocked_screen)

    forbid(monkeypatch, (rt.dataset, "iter_batches"))
    result = tools_eval.judge_function("guess", review_phase="confirmation")
    assert not result["ok"] and result["candidate_screen_blocked"]
    assert calls == []


def test_mixed_census_scores_eligible_keeper_and_reports_blocked_candidate(
    repair_runtime, monkeypatch,
):
    rt, _, _ = repair_runtime
    other = FeatureFunction(name="other", source=SOURCE.replace("def guess", "def other"),
                            description=DESCRIPTION, mechanism=MECHANISM)
    rt.library.upsert(other)
    monkeypatch.setattr(candidate_screen, "ensure_candidate_screen", lambda rt, fn, **kwargs: (
        blocked_screen() if fn.name == "guess" else
        {"ok": True, "status": "passed", "evaluation_allowed": True, "provisional": True}
    ))
    result = tools_eval.evaluate_functions(mode="full")
    assert not result["ok"] and result["partial"]
    assert result["evaluated_names"] == ["other"]
    assert result["blocked_functions"] == ["guess"]
    assert other.scores["evaluated"] and other.scores["n_items"] == 64


def test_judge_confirmation_uses_genuine_question_identity_after_exposure(
    repair_runtime, tmp_path,
):
    import asyncio

    from autofa.mcp.server import invoke_tool

    prior, statuses, calls = repair_runtime
    rows = [{"item_id": f"case-{i}", "model_id": "m", "benchmark": "b",
             "question_id": f"question-{i // 2}", "is_correct": i >= 52,
             "response_text": "I guess the final value." if i < 52 else "A derived value."}
            for i in range(64)]
    path = tmp_path / "questions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    fn = FeatureFunction(name="guess", source=SOURCE, description=DESCRIPTION, mechanism=MECHANISM)
    rt = Runtime(dataset=Dataset(parquet_paths=[path], model_id="m", benchmarks=["b"]),
                 library=FunctionLibrary(functions=[fn]), config=prior.config,
                 run=RunConfig(run_id="question-review", model_id="m", parquet_paths=[str(path)]))
    rt.run.run_dir().mkdir(parents=True, exist_ok=True)
    # Complete family comparison so confirmation can use untouched task identities.
    initialize_family_discovery(rt)
    _finish_family_comparisons(rt)
    set_runtime(rt)
    actual = [row for batch in rt.dataset.iter_batches() for row in batch]
    exposed_attempts = {row["_item_id"] for row in actual[:2]}
    response = asyncio.run(invoke_tool("get_item", {"item_id": actual[0]["_item_id"]}))
    assert json.loads(response["content"][0]["text"])["ok"], response
    assert tools_eval.evaluate_functions(names=["guess"], mode="full")["ok"]
    statuses.clear()  # Every later review supports the claim.
    calls.clear()
    result = tools_eval.judge_function("guess", review_phase="confirmation")
    assert result["ok"], result
    assert not exposed_attempts.intersection(calls)
    assert fn.scores["behavior_sample"]["independent_task_n"] == 6  # ceil(10% of 52 fires)
    question_by_id = {row["_item_id"]: row["question_id"] for row in actual}
    assert len({question_by_id[item_id] for item_id in calls}) == 6


# Supported parked repairs confirm independently without rewriting their history.


def supported_deferred(repair_runtime, *, attempts=1, compared=True):
    """Park a supported tightened candidate after its development review.

    Returns ``rt, statuses, calls, development``; ``compared`` also finishes the
    family comparisons that otherwise gate its confirmation.
    """
    rt, statuses, calls = repair_runtime
    # Thirty-two items keep each development or confirmation review at three cases.
    rt.dataset.items = rt.dataset.items[:32]
    rt._family_failed_task_inventory = None
    # These tests exercise confirmation of a supported parked candidate. Keep
    # passing traces outside the tightened predicate so unrelated quality-gap
    # feedback does not correctly direct the agent to refine precision first.
    for item in rt.dataset.items:
        if item["is_correct"]:
            item["response_text"] = "I guess an intermediate value."
    record_discovery_exposure(rt.run.run_dir(), rt.dataset.items[:4])
    for _ in range(attempts - 1):
        plan("gather_evidence", status="unknown")
        repair_workflow.defer_repair(rt.run.run_dir(), "guess", "Investigate the missing cause")
    attempt = plan("tighten_detector")
    submit(attempt)
    statuses.clear()  # Every later review supports the claim.
    initialize_family_discovery(rt)
    development = tools_eval.judge_function("guess")
    assert development["ok"], development
    assert development["sample"]["phase"] == "development"
    assert tools_repair.get_function_repair("guess")["feedback"]["allowed_actions"] == []
    tools_repair.defer_function_repair("guess", "Complete the other family comparisons first")
    calls.clear()
    if compared:
        _finish_family_comparisons(rt)
    return rt, statuses, calls, development


def test_supported_deferred_confirms_once_on_new_tasks_without_completing_repair(repair_runtime):
    # Even an exhausted repair budget leaves ordinary confirmation available.
    rt, _, calls, development = supported_deferred(repair_runtime, attempts=3)
    ledger = rt.run.run_dir() / "detector_repairs.json"
    before = ledger.read_bytes()
    next_action = tools_repair.get_function_repair("guess")["next_action"]
    assert next_action["tool"] == "judge_function"
    assert next_action["arguments"] == {"name": "guess", "review_phase": "confirmation"}
    confirmed = tools_eval.judge_function(**next_action["arguments"])
    assert confirmed["ok"], confirmed
    assert confirmed["sample"]["independent_task_n"] > 0
    assert confirmed["sample"]["reused_n"] == 0
    dev_ids = {row["item_id"] for row in development["verdicts"]}
    confirmed_ids = {row["item_id"] for row in confirmed["verdicts"]}
    assert confirmed_ids.isdisjoint(dev_ids | {f"case-{i}" for i in range(4)})
    assert calls and set(calls) == confirmed_ids
    assert ledger.read_bytes() == before
    state = workflow(rt)
    assert state["state"] == "exhausted"
    assert state["attempts_used"] == 3
    assert state["latest"]["candidates"]["guess"]["outcome"] is None
    assert "repair_validation" not in rt.library.get("guess").scores
    calls.clear()
    repeated = tools_eval.judge_function("guess", review_phase="confirmation")
    assert repeated["ok"] and repeated["status"] == "review_already_completed", repeated
    assert calls == []
    assert ledger.read_bytes() == before


def test_deferred_confirmation_obeys_family_gate_before_paid_screen_or_review(
    repair_runtime, monkeypatch,
):
    rt, _, calls, _ = supported_deferred(repair_runtime, compared=False)
    ledger = rt.run.run_dir() / "detector_repairs.json"
    before = ledger.read_bytes()
    review_path = rt.run.run_dir() / "review_evidence.json"
    history = review_path.read_bytes()
    forbid(monkeypatch, "autofa.session.candidate_screen.ensure_candidate_screen",
           (tools_eval, "_score_items_concurrently"))
    for _ in range(2):
        blocked = tools_eval.judge_function("guess", review_phase="confirmation")
        assert not blocked["ok"] and blocked["family_workflow_pending"], blocked
        assert blocked["next_action"]["tool"] == "sample_items"
        assert "survey" in blocked["next_action"]["reason"]
    assert calls == []
    assert ledger.read_bytes() == before
    assert review_path.read_bytes() == history
    # Auto/development remains blocked, but reports the same executable family prerequisite.
    blocked = tools_eval.judge_function("guess")
    assert not blocked["ok"]
    assert blocked["next_action"]["tool"] == "sample_items"


# One mutation per eligibility check: candidate claim, complete causal support,
# semantic verdict and recorded repair-ledger state.
@pytest.mark.parametrize("mutation", ["source", "missing_acceptance", "contradicted", "outcome"])
def test_unsupported_or_changed_deferred_candidate_never_reaches_paid_work(
    repair_runtime, monkeypatch, mutation,
):
    rt, _, calls, _ = supported_deferred(repair_runtime)
    fn = rt.library.get("guess")
    scores = fn.scores
    ledger = rt.run.run_dir() / "detector_repairs.json"
    if mutation == "source":
        fn.source += " changed"
    elif mutation == "missing_acceptance":
        acceptance = acceptance_requirement_id(rt.config.behavior_strictness)
        assert acceptance is not None
        for row in scores["behavior_verdicts"]:
            row["claim_assessments"] = [a for a in row["claim_assessments"]
                                        if a["requirement_id"] != acceptance]
    elif mutation == "contradicted":
        scores["behavior_verdicts"][0]["verdict"] = "no"
    else:
        data = json.loads(ledger.read_text())
        data["repairs"]["guess"]["history"][-1]["candidates"]["guess"]["outcome"] = {
            "recorded": True}
        ledger.write_text(json.dumps(data))
    before = ledger.read_bytes()
    before_scores = deepcopy(scores)
    forbid(monkeypatch, "autofa.session.candidate_screen.ensure_candidate_screen")
    for _ in range(2):
        result = tools_eval.judge_function("guess", review_phase="confirmation")
        assert not result["ok"] and "Repair validation is not ready" in result["error"], result
        assert result["next_action"]["tool"] != "judge_function"
    assert calls == []
    assert fn.scores == before_scores
    assert ledger.read_bytes() == before


def test_pending_family_assessment_uses_saved_feedback_not_another_review(repair_runtime):
    rt, _, _, _ = supported_deferred(repair_runtime, compared=False)
    path = rt.run.run_dir() / STATE_FILE
    data = json.loads(path.read_text())
    data.update(phase="assess_family", survey={"family_name": "guess", "evidence": []})
    path.write_text(json.dumps(data))
    blocked = tools_eval.judge_function("guess", review_phase="confirmation")
    assert blocked["family_workflow_pending"]
    assert blocked["next_action"]["tool"] == "get_function_feedback"
    assert blocked["next_action"]["arguments"] == {"name": "guess"}
    assert "assess_family" in blocked["next_action"]["reason"]
    assert "Do not repeat completed development review" in blocked["next_action"]["reason"]
    rt.library.get("guess").scores["evaluation_stale"] = True
    assert tools_repair.get_function_repair("guess")["next_action"]["tool"] == "evaluate_functions"
    data["survey"]["description"] = "A different recorded claim"
    path.write_text(json.dumps(data))
    action = tools_repair.get_function_repair("guess")["next_action"]
    assert action["tool"] == "family_discovery" and action["arguments"] == {"action": "status"}
    assert "record_family with the revised claim" in action["reason"]


def test_blocked_repair_plan_for_supported_candidate_points_to_confirmation(repair_runtime):
    supported_deferred(repair_runtime)
    feedback = tools_repair.get_function_repair("guess")["feedback"]
    finding = feedback["findings"][0]
    rejected = tools_repair.plan_function_repair(
        "guess", "tighten_detector", "Try another repair", [finding["id"]],
        [finding["item_ids"][0]],
    )
    assert not rejected["ok"]
    assert rejected["next_action"]["tool"] == "judge_function"
    assert rejected["next_action"]["arguments"]["review_phase"] == "confirmation"


def test_ordinary_confirmation_processing_recovery_keeps_cohort_and_history(repair_runtime):
    rt, statuses, calls, _ = supported_deferred(repair_runtime)
    ledger = rt.run.run_dir() / "detector_repairs.json"
    before = ledger.read_bytes()
    statuses["default"] = "error"
    failed = tools_eval.judge_function("guess", evidence_n=12, review_phase="confirmation")
    assert failed["ok"] and failed["n_errors"] > 0, failed
    assert failed["sample"]["independent_task_n"] > 0
    frozen_ids = {row["item_id"] for row in failed["verdicts"]}
    assert ledger.read_bytes() == before
    calls.clear()
    next_action = tools_repair.get_function_repair("guess")["next_action"]
    assert next_action["tool"] == "judge_function"
    assert next_action["arguments"] == {
        "name": "guess", "review_phase": "confirmation", "evidence_n": 12,
    }
    statuses["default"] = "supported"
    recovered = tools_eval.judge_function(**next_action["arguments"])
    assert recovered["ok"] and recovered["n_errors"] == 0, recovered
    assert {row["item_id"] for row in recovered["verdicts"]} == frozen_ids
    assert set(calls) == frozen_ids
    assert recovered["sample"]["reused_n"] == 0  # Recovery retains original independence.
    assert ledger.read_bytes() == before
    calls.clear()
    repeated = tools_eval.judge_function(**next_action["arguments"])
    assert repeated["status"] == "review_already_completed"
    assert calls == []


def test_ordinary_confirmation_semantic_failure_cannot_reset_with_new_sample(
    repair_runtime, monkeypatch,
):
    rt, statuses, calls, _ = supported_deferred(repair_runtime)
    original_review = claude_judge._run_review

    def completed_review(*, review, **kwargs):
        reply = json.loads(original_review(review=review, **kwargs))
        for verdict in reply["verdicts"]:
            for assessment in verdict["claim_assessments"]:
                if assessment["status"] == "unknown":
                    assessment.update(
                        unknown_reason="evidence_unavailable",
                        evidence_inspection={
                            "status": "unavailable",
                            "ranges": [{"field": "response_text", "start": 0,
                                        "end": len(review.fields["response_text"])}],
                            "reason": "The response is inspected; its derivation is not retained.",
                        },
                    )
        return json.dumps(reply)

    monkeypatch.setattr(claude_judge, "_run_review", completed_review)
    statuses["default"] = "unknown"  # Completed, inspected, and still unsupported.
    result = tools_eval.judge_function("guess", review_phase="confirmation")
    assert result["ok"], result
    assert result["n_errors"] == result["n_review_incomplete"] == 0
    assert all(row["completion_status"] == "complete" for row in result["verdicts"])
    calls.clear()
    for evidence_n in [None, 30]:
        blocked = tools_eval.judge_function("guess", evidence_n=evidence_n,
                                          review_phase="confirmation")
        assert not blocked["ok"], blocked
        assert blocked["next_action"]["tool"] != "judge_function"
    assert calls == []


@pytest.mark.parametrize("setting", ["model", "evidence_n"])
def test_deferred_confirmation_cannot_change_settings_to_get_new_cohort(repair_runtime, setting):
    rt, _, calls, _ = supported_deferred(repair_runtime)
    confirmed = tools_eval.judge_function("guess", review_phase="confirmation")
    assert confirmed["ok"], confirmed
    calls.clear()
    kwargs = {}
    if setting == "model":
        rt.config.behavior_judge_model = "different-judge"
    else:
        kwargs["evidence_n"] = 30
    blocked = tools_eval.judge_function("guess", review_phase="confirmation", **kwargs)
    assert not blocked["ok"], blocked
    if setting != "evidence_n":
        assert "Restore the judge settings" in blocked["next_action"]["reason"]
    else:
        assert blocked["next_action"]["tool"] == "judge_function"
        assert "evidence_n" not in blocked["next_action"]["arguments"]
    assert calls == []


def test_repair_error_response_survives_unavailable_ledger(repair_runtime, monkeypatch):
    supported_deferred(repair_runtime, compared=False)

    def unavailable(*args, **kwargs):
        raise OSError("Temporarily unavailable repair ledger")

    monkeypatch.setattr(repair_workflow, "validation_context", unavailable)
    monkeypatch.setattr(repair_workflow, "get_repair", unavailable)
    blocked = tools_eval.judge_function("guess", review_phase="confirmation")
    assert not blocked["ok"]
    assert "Temporarily unavailable" in blocked["error"]
    assert blocked["next_action"]["tool"] == "get_library_function"


def test_interrupted_ordinary_confirmation_cannot_replace_its_reserved_operation(
    repair_runtime, monkeypatch,
):
    from autofa.session.review_evidence import find_review_allocation

    rt, _, _, _ = supported_deferred(repair_runtime)
    original = claude_judge.judge_behavior_with_claude
    entered = []

    def interrupted(**kwargs):
        entered.append({row["_item_id"] for row in kwargs["items"]})
        raise RunInterruptedError("Disconnected after the cohort was reserved")

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", interrupted)
    ledger = (rt.run.run_dir() / "detector_repairs.json").read_bytes()
    first = tools_eval.judge_function("guess", evidence_n=10, review_phase="confirmation")
    assert not first["ok"] and first["resumable"]
    assert rt.library.get("guess").scores["behavior_review_phase"] == "development"
    frozen_request = rt.library.get("guess").scores["ordinary_confirmation_request"]
    allocation = find_review_allocation(rt.run.run_dir(), "guess", phase="confirmation",
                                        review_key=frozen_request["review_key"])
    assert set(allocation["item_ids"]) == entered[0]
    rt.config = apply_preset(rt.config, "recall")
    blocked = tools_eval.judge_function("guess", evidence_n=12, review_phase="confirmation")
    assert not blocked["ok"] and len(entered) == 1
    assert blocked["next_action"]["arguments"] == {
        "name": "guess", "review_phase": "confirmation", "evidence_n": 10,
    }
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", original)
    resumed = tools_eval.judge_function(**blocked["next_action"]["arguments"])
    assert resumed["ok"], resumed
    assert {row["item_id"] for row in resumed["verdicts"]} == entered[0]
    assert (rt.run.run_dir() / "detector_repairs.json").read_bytes() == ledger


def test_deferred_confirmation_without_available_tasks_can_use_later_fresh_evidence(repair_runtime):
    rt, _, calls, _ = supported_deferred(repair_runtime)
    record_discovery_exposure(rt.run.run_dir(), rt.dataset.items)
    unavailable = tools_eval.judge_function("guess", review_phase="confirmation")
    assert unavailable["ok"] and unavailable["status"] == "confirmation_unavailable", unavailable
    assert not calls
    assert unavailable["next_action"]["tool"] == "get_function_feedback"
    assert "collect new source tasks" in unavailable["next_action"]["reason"]
    rt.dataset.items += fresh_tasks(rt, 20, "new")
    available = tools_eval.judge_function("guess", review_phase="confirmation")
    assert available["ok"] and available["sample"]["independent_task_n"] > 0, available
    assert set(calls) <= {f"new-{i}" for i in range(20)}


def test_exhausted_parked_repair_can_expand_development_support(repair_runtime):
    rt, _, calls, development = supported_deferred(repair_runtime, attempts=3, compared=False)
    ledger = rt.run.run_dir() / 'detector_repairs.json'
    before = ledger.read_bytes()
    target = len(development['verdicts']) + 5
    blocked = tools_eval.judge_function('guess', evidence_n=target, review_phase='development')
    assert not blocked['ok'] and not calls
    _finish_family_comparisons(rt)
    expanded = tools_eval.judge_function('guess', evidence_n=target, review_phase='development')
    assert expanded['ok'], expanded
    old_ids = {row['item_id'] for row in development['verdicts']}
    new_ids = {row['item_id'] for row in expanded['verdicts']}
    assert old_ids < new_ids
    assert set(calls) == new_ids - old_ids
    assert expanded['sample']['limited']
    assert expanded['sample']['target_n'] == len(new_ids) == target
    assert expanded['sample']['support_review']['selected_failed_task_n'] >= 5
    assert ledger.read_bytes() == before
    calls.clear()
    repeated = tools_eval.judge_function('guess', evidence_n=target, review_phase='development')
    assert repeated['status'] == 'review_already_completed'
    assert not calls
    assert ledger.read_bytes() == before


def test_support_expansion_does_not_bypass_parked_repair_safety_gates(repair_runtime, monkeypatch):
    rt, _, _, _ = supported_deferred(repair_runtime)
    fn = rt.library.get('guess')
    fn.scores.update(errors=1, error_examples=[{'item_id': 'case-1', 'error': 'ParseError'}])
    ledger = rt.run.run_dir() / 'detector_repairs.json'
    before = ledger.read_bytes()
    forbid(monkeypatch, 'autofa.session.candidate_screen.ensure_candidate_screen')
    result = tools_eval.judge_function('guess', evidence_n=5, review_phase='development')
    assert not result['ok']
    assert 'Repair validation is not ready' in result['error']
    assert ledger.read_bytes() == before
