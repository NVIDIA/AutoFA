# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repair state is bounded, grounded and independent of library/model mutation."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.mcp.tools_repair import review_next_action
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.repair_feedback import build_repair_feedback, has_complete_causal_support
from autofa.session import repair_workflow as workflow
from autofa.session.review_evidence import record_discovery_exposure, sync_review_history
from autofa.util.atomic import atomic_write_json

from support.repair_review import reviewed_function


def function(name="f", verdict="no"):
    fn = {"name": name, "source": "def f(item): return True", "description": "Claim",
          "mechanism": "Original defining cause and consequence", "kind": "regex",
          "depends_on": [], "meta": {}, "scores": {}}
    fn["scores"] = {
        "claim_hash": fingerprint(fn), "judge_policy_hash": judge_policy_hash(),
        "judge_behavior_strictness": "causal",
        "behavior_verdicts": [{"item_id": "diagnostic-1", "verdict": verdict}],
        "behavior_purity": 0.0, "n_items": 100,
    }
    return fn


def fingerprint(fn):
    return claim_hash(source=fn["source"], description=fn["description"],
                      mechanism=fn["mechanism"])


def feedback(fn):
    label = fn["scores"]["behavior_verdicts"][0]["verdict"]
    status = {"error": "processing_error", "uncertain": "unknown"}.get(label, "contradicted")
    return {"claim_hash": fingerprint(fn), "allowed_actions": sorted(workflow.ACTIONS),
            "findings": [{"id": "finding-1", "status": status,
                          "examples": [{"item_id": "diagnostic-1"}]}],
            "diagnostic_item_ids": [r["item_id"] for r in fn["scores"]["behavior_verdicts"]]}


def plan(directory, fn, action="tighten_detector", **overrides):
    return workflow.plan_repair(
        directory, fn, feedback(fn), **{
            "action": action, "rationale": "The cited condition is missing from these fires.",
            "finding_ids": ["finding-1"], "evidence_ids": ["diagnostic-1"], **overrides,
        },
    )


def submit(directory, fn, *, name=None, mechanism=None, repair_plan_id=None):
    candidate = deepcopy(fn)
    candidate["source"] += "\n# narrower detector"
    if name:
        candidate["name"] = name
    if mechanism:
        candidate["mechanism"] = mechanism
    workflow.record_submission(directory, candidate, repair_plan_id=repair_plan_id)
    return candidate


def result(fn, label="yes", item_id="holdout-1", **overrides):
    return {"claim_hash": fingerprint(fn), "judge_policy_hash": judge_policy_hash(),
            "judge_behavior_strictness": "causal",
            "verdicts": [{"item_id": item_id, "verdict": label}], **overrides}


def reserved(directory, ids=("holdout-1",)):
    """Plan, submit and reserve validation for f; return the plan and its candidate."""
    fn = function()
    repair = plan(directory, fn)
    candidate = submit(directory, fn)
    workflow.reserve_validation(directory, "f", list(ids))
    return repair, candidate


def close_repair(directory, fn, ending):
    """Stop f's repair at ``ending``; return the candidate and any recorded validation."""
    if ending == "exhausted":
        for _ in range(3):
            plan(directory, fn, action="gather_evidence")
            workflow.defer_repair(directory, "f", "Need more evidence")
        return fn, None
    if ending == "retired":
        plan(directory, fn, action="retire")
        workflow.record_retirement(directory, "f", "Unsupported cause", "rejected")
        return fn, None
    plan(directory, fn)
    candidate = submit(directory, fn)
    if ending == "shortage":
        # Too few new items never falls back to the diagnostic cases.
        workflow.reserve_validation(directory, "f", [])
        reviewed, eligible = {"verdicts": [], "n_judged": 0, "required_items": 10}, 4
    else:
        workflow.reserve_validation(directory, "f", ["holdout-1"])
        reviewed, eligible = result(candidate, ending), 1
    return candidate, workflow.record_validation(directory, candidate, reviewed,
                                                 eligible_items=eligible)


def supported_candidate(**overrides):
    fn = reviewed_function("supported", "supported", "supported")
    fn["scores"].update(
        evaluated=True, evaluation_claim_hash=fn["scores"]["claim_hash"],
        evaluation_run_id="quality-run", task_recall_corpus_fingerprint="quality-corpus",
        sampled=True, evaluation_scope_complete=True, n_items=200, n_flagged=10,
        recall=.042, ppv_at_pi0=.57,
        evaluation_item_ids=["item-000", "item-001", "item-002", "passing", "missed"],
        buckets=[{"bucket": True, "example_fail_ids": ["item-000"],
                  "example_pass_ids": ["passing"]},
                 {"bucket": False, "example_fail_ids": ["missed"]}],
        **overrides,
    )
    return fn


def propose(directory, fn, **overrides):
    """Plan a quality refinement from the supported candidate's precision gap."""
    feedback = build_repair_feedback(fn)
    finding = next(row for row in feedback["findings"]
                   if row["status"] == "quality_gap" and row["requirement_id"] == "ppv_at_pi0")
    return workflow.plan_repair(directory, fn, feedback, **{
        "action": "refine_detector", "rationale": "Test the evidenced request distinction.",
        "finding_ids": [finding["id"]], "evidence_ids": ["passing"], **overrides,
    })


def test_baseline_before_after_and_results_preserved_across_reload(tmp_path):
    fn = function()
    original = deepcopy(fn)
    first = plan(tmp_path, fn)
    candidate = submit(tmp_path, fn)
    submitted = deepcopy(candidate)
    ctx = workflow.validation_context(tmp_path, candidate)
    assert ctx["exclusion_item_ids"] == ["diagnostic-1"]
    assert ctx["plan_id"] == first["plan_id"]
    workflow.reserve_validation(tmp_path, "f", ["holdout-1"])
    completed = workflow.record_validation(tmp_path, candidate, result(candidate), eligible_items=5)
    assert completed["phase"] == "completed"
    assert completed["outcome"]["all_reviewed_confirmed"]
    assert completed["outcome"]["automatic_promotion"] is False
    candidate["scores"]["behavior_purity"] = 99
    fn["mechanism"] = "mutated outside the ledger"
    stored = workflow.get_repair(tmp_path, "f")
    assert stored["baseline"] == original
    assert stored["latest"]["before"] == original
    assert stored["latest"]["after"] == submitted
    assert stored["latest"]["candidates"]["f"]["result"]["verdicts"][0]["verdict"] == "yes"
    assert stored["reviewed_item_ids"] == ["diagnostic-1", "holdout-1"]
    assert (tmp_path / "detector_repairs.json").is_file()
    # A confirmed candidate returns to ordinary (non-repair) validation.
    assert workflow.validation_context(tmp_path, submitted) is None


@pytest.mark.parametrize("overrides", [
    {"finding_ids": ["invented"]}, {"rationale": " "}, {"action": "automatically_accept"},
    {"finding_ids": []},
], ids=["unknown_finding", "blank_rationale", "unknown_action", "no_finding"])
def test_ungrounded_decisions_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        plan(tmp_path, function(), **overrides)
    assert workflow.get_repair(tmp_path, "f") == {}


def test_stale_feedback_cannot_plan(tmp_path):
    fn = function()
    stale = feedback(fn)
    fn["source"] += "\n# changed"
    with pytest.raises(ValueError, match="stale"):
        workflow.plan_repair(tmp_path, fn, stale, action="gather_evidence", rationale="Inspect",
                             finding_ids=[], evidence_ids=[])


def test_planned_tightening_needs_edit_and_preserves_literal_claim(tmp_path):
    fn = function()
    plan(tmp_path, fn)
    with pytest.raises(ValueError, match="submitted"):
        workflow.validation_context(tmp_path, fn)
    with pytest.raises(ValueError, match="changed candidate"):
        workflow.record_submission(tmp_path, fn)
    with pytest.raises(ValueError, match="literal"):
        submit(tmp_path, fn, mechanism="Weaker claim")
    assert workflow.get_repair(tmp_path, "f")["latest"]["after"] is None


def test_claim_revision_is_not_confirmation_of_original_claim(tmp_path):
    fn = function()
    plan(tmp_path, fn, action="revise_claim")
    candidate = submit(tmp_path, fn, mechanism="Narrower observable claim")
    workflow.reserve_validation(tmp_path, "f", ["holdout-1"])
    completed = workflow.record_validation(tmp_path, candidate, result(candidate), eligible_items=1)
    assert completed["outcome"]["comparison"] == "new_claim_validation"
    assert not completed["outcome"]["original_claim_validated"]


@pytest.mark.parametrize("quality", [False, True], ids=["contradiction", "quality_refinement"])
def test_attempt_limit_persists_across_reload_and_deferral(tmp_path, quality):
    # Quality refinement spends the same bounded attempt budget as other repairs.
    fn, start = (supported_candidate(), propose) if quality else (function(), plan)
    for expected in range(1, 4):
        assert start(tmp_path, fn)["attempt"] == expected
        workflow.defer_repair(tmp_path, fn["name"], "Evidence not yet available")
        assert workflow.get_repair(tmp_path, fn["name"])["attempts_used"] == expected
    assert workflow.get_repair(tmp_path, fn["name"])["state"] == "exhausted"
    with pytest.raises(ValueError, match="attempt limit"):
        start(tmp_path, fn)


def polishing_phase(directory, identifier="run:3", **changes):
    atomic_write_json(directory / "round_validation.json", {"refinement": {
        "version": 1, "run_id": "run", "round": 3, "mode": "polishing",
        "status": "awaiting_response", "polishing_id": identifier,
        "frozen_names": ["f"], **changes,
    }})


def test_polishing_has_an_explicit_bounded_allowance_without_resetting_history(tmp_path):
    fn = function()
    for _ in range(3):
        plan(tmp_path, fn)
        workflow.defer_repair(tmp_path, "f", "Recorded discovery attempt")
    before = workflow.get_repair(tmp_path, "f")
    assert before["state"] == "exhausted"
    polishing_phase(tmp_path)
    for used in range(1, 4):
        attempt = plan(tmp_path, fn, polishing_id="run:3")
        assert attempt["attempt"] == 3 + used
        assert attempt["attempt_budget"] == {
            "scope": "polishing_round", "polishing_id": "run:3", "max_attempts": 3,
            "attempts_used": used, "remaining": 3 - used,
        }
        workflow.defer_repair(tmp_path, "f", "Validation did not establish improvement")
        saved = workflow.get_repair(tmp_path, "f")
        assert saved["attempts_used"] == 3 + used
        assert saved["history"][:3] == before["history"]
        assert saved["reviewed_item_ids"] == before["reviewed_item_ids"]
        assert saved["state"] == ("exhausted" if used == 3 else "deferred")
    # Reloading the same closing exchange cannot replenish its allowance.
    polishing_phase(tmp_path)
    with pytest.raises(ValueError, match="limit reached .*this polishing round"):
        plan(tmp_path, fn, polishing_id="run:3")
    # The runtime can explicitly assign a new polishing round. Old attempts
    # and evidence remain; the candidate never receives a silent global reset.
    polishing_phase(tmp_path, "run:4", round=4)
    next_attempt = plan(tmp_path, fn, polishing_id="run:4")
    assert next_attempt["attempt"] == 7
    assert next_attempt["attempt_budget"]["remaining"] == 2
    assert len(workflow.get_repair(tmp_path, "f")["history"]) == 7
    with pytest.raises(ValueError, match="active repair"):
        plan(tmp_path, fn, polishing_id="run:4")


def test_stale_polishing_allowance_is_rejected_without_a_ledger_edit(tmp_path):
    polishing_phase(tmp_path, polishing_id="prior:2")
    with pytest.raises(ValueError, match="active final polishing phase"):
        plan(tmp_path, function(), polishing_id="run:3")
    assert workflow.get_repair(tmp_path, "f") == {}


def test_processing_retry_does_not_spend_polishing_repair_allowance(tmp_path):
    state = {"history": [
        {"action": "tighten_detector"},
        {"action": "tighten_detector", "polishing_id": "run:3"},
        {"action": "retry_review", "polishing_id": "run:3"},
    ]}
    assert workflow.repair_attempt_budget(state, polishing_id="run:3")["remaining"] == 2
    assert workflow.repair_attempt_budget(state)["attempts_used"] == 2


def test_holdout_disjointness_and_crash_reservation_survive_restart(tmp_path):
    fn = function()
    plan(tmp_path, fn)
    candidate = submit(tmp_path, fn)
    with pytest.raises(ValueError, match="previously reviewed"):
        workflow.reserve_validation(tmp_path, "f", ["diagnostic-1"])
    workflow.reserve_validation(tmp_path, "f", ["holdout-1"])
    with pytest.raises(ValueError, match="already reserved"):
        workflow.validation_context(tmp_path, candidate)
    workflow.defer_repair(tmp_path, "f", "Interrupted validation")
    plan(tmp_path, candidate, action="gather_evidence")
    assert "holdout-1" in workflow.validation_context(tmp_path, candidate)["exclusion_item_ids"]


@pytest.mark.parametrize("ids", [["holdout-2", "holdout-1"], []])
def test_resume_preserves_reserved_sample_and_original_exclusions(tmp_path, ids):
    fn = function()
    repair = plan(tmp_path, fn)
    candidate = submit(tmp_path, fn)
    assert workflow.validation_context(tmp_path, candidate)["reserved_item_ids"] is None
    reserved = workflow.reserve_validation(
        tmp_path, "f", ids, expected_plan_id=repair["plan_id"],
    )
    before = workflow.get_repair(tmp_path, "f")
    # A new caller reads durable state after the interrupted review. It may
    # finish this exact reservation without consuming another repair attempt.
    ctx = workflow.validation_context(tmp_path, deepcopy(candidate), resume_reserved=True)
    assert ctx["reserved_item_ids"] == ids
    assert ctx["exclusion_item_ids"] == ["diagnostic-1"]
    assert ctx["sample_kind"] == "fresh_repair_holdout"
    assert workflow.reserve_validation(
        tmp_path, "f", ctx["reserved_item_ids"], expected_plan_id=ctx["plan_id"],
        resume_existing=True,
    ) == reserved
    assert workflow.get_repair(tmp_path, "f") == before
    # Caller edits must not alter the saved reservation or exclusion history.
    ctx["reserved_item_ids"].append("unreserved-item")
    assert workflow.get_repair(tmp_path, "f") == before
    rows = [{"item_id": item_id, "verdict": "yes"} for item_id in ids]
    completed = workflow.record_validation(
        tmp_path, candidate, result(candidate, verdicts=rows, required_items=2),
        eligible_items=len(ids), expected_plan_id=repair["plan_id"],
    )
    assert completed["phase"] == ("completed" if ids else "deferred")
    assert workflow.get_repair(tmp_path, "f")["attempts_used"] == 1


@pytest.mark.parametrize("ids,plan_id,match", [
    (["holdout-1", "holdout-2"], "original", "same reserved evidence"),
    (["holdout-2", "holdout-1"], None, "original plan ID"),
    (["holdout-2", "holdout-1"], "different-plan", "plan changed"),
], ids=["reordered_sample", "missing_plan_id", "other_plan"])
def test_resuming_reservation_requires_same_sample_and_plan_identity(
    tmp_path, ids, plan_id, match,
):
    repair, _ = reserved(tmp_path, ["holdout-2", "holdout-1"])
    before = workflow.get_repair(tmp_path, "f")
    with pytest.raises(ValueError, match=match):
        workflow.reserve_validation(
            tmp_path, "f", ids, resume_existing=True,
            expected_plan_id=repair["plan_id"] if plan_id == "original" else plan_id,
        )
    assert workflow.get_repair(tmp_path, "f") == before


def test_changed_candidate_or_dependencies_cannot_resume_or_use_reserved_validation(tmp_path):
    _, candidate = reserved(tmp_path)
    candidate["meta"]["deps"] = {"parent": "def parent(item): return False"}
    with pytest.raises(ValueError, match="Candidate changed"):
        workflow.validation_context(tmp_path, candidate, resume_reserved=True)
    with pytest.raises(ValueError, match="submitted candidate"):
        workflow.record_validation(tmp_path, candidate, result(candidate), eligible_items=1)


def test_resuming_does_not_allow_changed_policy(tmp_path, monkeypatch):
    repair, candidate = reserved(tmp_path)
    monkeypatch.setattr("autofa.metrics.claude_judge.evidence_projection_hash", lambda: "changed")
    with pytest.raises(ValueError, match="Judge policy changed"):
        workflow.validation_context(tmp_path, candidate, resume_reserved=True)
    with pytest.raises(ValueError, match="Judge policy changed"):
        workflow.reserve_validation(tmp_path, "f", ["holdout-1"],
                                    expected_plan_id=repair["plan_id"], resume_existing=True)


def test_processing_retry_reuses_only_failed_ids_and_resumes_as_retry(tmp_path):
    fn = function(verdict="error")
    repair = plan(tmp_path, fn, action="retry_review")
    assert workflow.validation_context(tmp_path, fn)["retry_item_ids"] == ["diagnostic-1"]
    with pytest.raises(ValueError, match="processing error"):
        workflow.reserve_validation(tmp_path, "f", ["new-item"])
    reservation = workflow.reserve_validation(tmp_path, "f", ["diagnostic-1"])
    ctx = workflow.validation_context(tmp_path, deepcopy(fn), resume_reserved=True)
    assert ctx["reserved_item_ids"] == ctx["retry_item_ids"] == ["diagnostic-1"]
    assert ctx["sample_kind"] == "processing_retry"
    assert ctx["exclusion_item_ids"] == reservation["excluded_item_ids"]
    assert workflow.reserve_validation(
        tmp_path, "f", ctx["reserved_item_ids"], expected_plan_id=repair["plan_id"],
        resume_existing=True,
    ) == reservation
    fresh = result(fn, item_id="diagnostic-1")
    fn["scores"]["behavior_verdicts"] = fresh["verdicts"]
    recorded = workflow.record_validation(tmp_path, fn, fresh, eligible_items=1,
                                          expected_plan_id=repair["plan_id"])
    assert recorded["outcome"]["sample_kind"] == "processing_retry"
    assert not recorded["outcome"]["original_claim_validated"]


@pytest.mark.parametrize("label,phase,status", [
    ("no", "completed", "complete"),
    ("uncertain", "deferred", "uncertain"),
    ("error", "deferred", "processing_error"),
    ("shortage", "deferred", "insufficient_new_evidence"),
])
def test_review_outcomes_do_not_promote_failed_or_unknown_repairs(tmp_path, label, phase, status):
    _, recorded = close_repair(tmp_path, function(), label)
    assert recorded["phase"] == phase
    assert recorded["outcome"]["review_status"] == status
    assert recorded["outcome"]["n_judged"] == (0 if label == "shortage" else 1)
    assert not recorded["outcome"]["all_reviewed_confirmed"]


@pytest.mark.parametrize("verdict,policy,match", [
    ("uncertain", None, "status"), ("error", "old-policy", "same current"),
], ids=["semantic_verdict", "changed_policy"])
def test_processing_retry_requires_a_processing_error_under_the_current_policy(
    tmp_path, verdict, policy, match,
):
    fn = function(verdict=verdict)
    if policy:
        fn["scores"]["judge_policy_hash"] = policy
    with pytest.raises(ValueError, match=match):
        plan(tmp_path, fn, action="retry_review")


def test_processing_retry_of_review_without_recorded_strictness_is_refused(tmp_path):
    fn = function(verdict="error")
    fn["scores"].pop("judge_behavior_strictness", None)
    with pytest.raises(ValueError, match="same current"):
        plan(tmp_path, fn, action="retry_review")


@pytest.mark.parametrize("bad", [
    {"n_yes": 99}, {"claim_hash": "stale"},
    {"verdicts": [{"item_id": "wrong-item", "verdict": "yes"}]},
    # Same reserved ID set, but one item would count as two confirmations.
    {"verdicts": [{"item_id": "holdout-1", "verdict": "yes"}] * 2},
], ids=["counts", "claim", "item", "duplicate_item"])
def test_malformed_or_mismatched_validation_fails_closed(tmp_path, bad):
    _, candidate = reserved(tmp_path)
    with pytest.raises(ValueError):
        workflow.record_validation(tmp_path, candidate, result(candidate, **bad), eligible_items=2)
    assert workflow.get_repair(tmp_path, "f")["latest"]["result"] is None


def test_retirement_and_evidence_gathering_preserve_baseline(tmp_path):
    fn = function()
    plan(tmp_path, fn, action="gather_evidence", finding_ids=[], evidence_ids=[])
    assert workflow.validation_context(tmp_path, fn)["sample_kind"] == "fresh_repair_holdout"
    workflow.defer_repair(tmp_path, "f", "Need authoritative rate")
    close_repair(tmp_path, fn, "retired")
    state = workflow.get_repair(tmp_path, "f")
    assert state["latest"]["phase"] == "completed"
    assert state["latest"]["outcome"]["review_status"] == "retired"
    assert state["baseline"] == fn


def test_planned_version_must_still_be_current_before_upsert(tmp_path):
    fn = function()
    plan(tmp_path, fn)
    candidate = {**fn, "source": fn["source"] + "\n# repaired"}
    changed_current = {**fn, "mechanism": "Changed outside this plan"}
    with pytest.raises(ValueError, match="Current function changed"):
        workflow.record_submission(tmp_path, candidate, current_function=changed_current)
    workflow.record_submission(tmp_path, candidate, current_function=fn)


def test_split_cannot_overwrite_untracked_existing_child(tmp_path):
    fn = function()
    repair = plan(tmp_path, fn, action="split_detector")
    child = {**fn, "name": "child", "source": fn["source"] + "\n# child"}
    with pytest.raises(ValueError, match="overwrite"):
        workflow.record_submission(tmp_path, child, repair["plan_id"], current_function=child)
    workflow.record_submission(tmp_path, child, repair["plan_id"], current_function=None)


def test_retry_preserves_aggregate_uncertainty_and_separates_reviewed_counts(tmp_path):
    fn = function(verdict="error")
    fn["scores"]["behavior_verdicts"].append({"item_id": "prior-unknown", "verdict": "uncertain"})
    plan(tmp_path, fn, action="retry_review")
    workflow.reserve_validation(tmp_path, "f", ["diagnostic-1"])
    fresh = result(fn, item_id="diagnostic-1")
    fn["scores"]["behavior_verdicts"][0] = fresh["verdicts"][0]
    recorded = workflow.record_validation(tmp_path, fn, fresh, eligible_items=1)
    assert recorded["phase"] == "deferred"
    assert recorded["outcome"]["n_uncertain"] == 1
    assert recorded["outcome"]["reviewed_counts"]["n_uncertain"] == 0
    assert not recorded["outcome"]["all_reviewed_confirmed"]
    assert not recorded["outcome"]["original_claim_validated"]


def test_split_child_cannot_open_competing_plan_until_parent_deferred(tmp_path):
    fn = function()
    parent_plan = plan(tmp_path, fn, action="split_detector")
    child = submit(tmp_path, fn, name="child", mechanism="Narrow child",
                   repair_plan_id=parent_plan["plan_id"])
    with pytest.raises(ValueError, match="parent split"):
        plan(tmp_path, child)
    workflow.defer_repair(tmp_path, "f", "Pause the split")
    child_plan = plan(tmp_path, child)
    assert workflow.get_repair(tmp_path, "child")["latest"]["plan_id"] == child_plan["plan_id"]
    child_candidate = submit(tmp_path, child)
    context = workflow.validation_context(tmp_path, child_candidate)
    assert context["plan_id"] == child_plan["plan_id"]


@pytest.mark.parametrize("ending", ["no", "exhausted"])
def test_closed_failed_repair_cannot_bypass_budget_with_ordinary_judge(tmp_path, ending):
    candidate, _ = close_repair(tmp_path, function(), ending)
    with pytest.raises(ValueError, match="new active plan"):
        workflow.validation_context(tmp_path, candidate)


@pytest.mark.parametrize("ending", ["exhausted", "yes"])
def test_inactive_repair_rejects_changed_submission_before_mutation(tmp_path, ending):
    candidate, _ = close_repair(tmp_path, function(), ending)
    before = (tmp_path / "detector_repairs.json").read_bytes()
    changed = {**candidate, "mechanism": "A reworded explanation of the same failure"}
    with pytest.raises(ValueError, match="new active repair plan"):
        workflow.record_submission(tmp_path, changed, current_function=candidate)
    assert (tmp_path / "detector_repairs.json").read_bytes() == before
    # Explaining how unchanged code works does not discard its existing review.
    metadata_only = {**candidate, "function_mechanism": "Read the visible response text"}
    assert workflow.record_submission(tmp_path, metadata_only, current_function=candidate) is None


def test_archived_split_child_stays_readable_but_needs_its_own_plan(tmp_path):
    fn = function()
    old = plan(tmp_path, fn, action="split_detector")
    child = submit(tmp_path, fn, name="child", mechanism="Narrow child",
                   repair_plan_id=old["plan_id"])
    workflow.defer_repair(tmp_path, "f", "Park incomplete split")
    plan(tmp_path, fn)
    archived = workflow.get_repair(tmp_path, "child")
    assert archived["historical_link"]
    assert archived["latest"]["plan_id"] == old["plan_id"]
    with pytest.raises(ValueError, match="new active repair plan"):
        workflow.validation_context(tmp_path, child)
    changed = {**child, "source": child["source"] + "\n# unplanned edit"}
    with pytest.raises(ValueError, match="new active repair plan"):
        workflow.record_submission(tmp_path, changed, current_function=child)
    assert workflow.record_submission(tmp_path, child, current_function=child) is None


def test_child_workflow_inherits_parent_diagnostic_and_validation_exclusions(tmp_path):
    parent = function()
    split = plan(tmp_path, parent, action="split_detector")
    child = submit(tmp_path, parent, name="child", mechanism="First failure",
                   repair_plan_id=split["plan_id"])
    sibling = submit(tmp_path, parent, name="sibling", mechanism="Second failure",
                     repair_plan_id=split["plan_id"])
    for candidate in (child, sibling):
        key = candidate["name"] + "-validation"
        workflow.reserve_validation(tmp_path, candidate["name"], [key])
        reviewed = result(candidate, "no", item_id=key)
        workflow.record_validation(tmp_path, candidate, reviewed, eligible_items=1)
        # Normal judging replaces a child's current score details. Parent repair
        # cases are now present only in the persistent lineage, not this feedback.
        candidate["scores"]["behavior_verdicts"] = deepcopy(reviewed["verdicts"])
    plan(tmp_path, parent, action="gather_evidence")  # Archives the split.
    parent_before = workflow.get_repair(tmp_path, "f")
    child_feedback = {
        "claim_hash": fingerprint(child), "allowed_actions": ["tighten_detector"],
        "diagnostic_item_ids": ["child-validation"],
        "findings": [{"id": "child-finding", "status": "contradicted",
                      "item_ids": ["child-validation"], "examples": []}],
    }
    workflow.plan_repair(
        tmp_path, child, child_feedback, action="tighten_detector",
        rationale="The child still fires without its defining failure",
        finding_ids=["child-finding"], evidence_ids=["child-validation"],
    )
    repaired_child = submit(tmp_path, child)
    context = workflow.validation_context(tmp_path, repaired_child)
    assert set(context["exclusion_item_ids"]) == {
        "diagnostic-1", "child-validation", "sibling-validation",
    }
    for known_id in ("diagnostic-1", "sibling-validation"):
        with pytest.raises(ValueError, match="previously reviewed"):
            workflow.reserve_validation(tmp_path, "child", [known_id])
    assert workflow.get_repair(tmp_path, "f") == parent_before
    workflow.reserve_validation(tmp_path, "child", ["new-child-holdout"])


@pytest.mark.parametrize("first_label,expected_phase", [
    ("yes", "completed"), ("no", "completed"), ("uncertain", "deferred"), ("error", "deferred"),
])
def test_split_outcome_aggregates_all_children_not_only_last_success(
    tmp_path, first_label, expected_phase,
):
    fn = function()
    repair = plan(tmp_path, fn, action="split_detector")
    first = submit(tmp_path, fn, name="first", mechanism="First failure",
                   repair_plan_id=repair["plan_id"])
    second = submit(tmp_path, fn, name="second", mechanism="Second failure",
                    repair_plan_id=repair["plan_id"])
    assert workflow.validation_context(tmp_path, first)["plan_id"] == repair["plan_id"]
    workflow.reserve_validation(tmp_path, "first", ["first-holdout"])
    partial = workflow.record_validation(
        tmp_path, first, result(first, first_label, item_id="first-holdout"), eligible_items=1,
    )
    assert not partial["outcome"]["all_reviewed_confirmed"]
    assert partial["outcome"]["review_status"] == "awaiting_validation"
    assert workflow.get_repair(tmp_path, "f")["state"] == "awaiting_validation"
    # Each child needs validation disjoint from its siblings'.
    with pytest.raises(ValueError, match="previously reviewed"):
        workflow.reserve_validation(tmp_path, "second", ["first-holdout"])
    workflow.reserve_validation(tmp_path, "second", ["second-holdout"])
    complete = workflow.record_validation(
        tmp_path, second, result(second, item_id="second-holdout"), eligible_items=1,
    )
    assert complete["phase"] == expected_phase
    assert workflow.get_repair(tmp_path, "f")["state"] == expected_phase
    assert len(workflow.get_repair(tmp_path, "second")["latest"]["candidates"]) == 2
    assert complete["outcome"]["n_judged"] == 2
    assert complete["outcome"]["n_yes"] == (2 if first_label == "yes" else 1)
    assert complete["outcome"]["all_reviewed_confirmed"] is (first_label == "yes")
    assert set(complete["result"]["candidate_results"]) == {"first", "second"}


def test_split_ownership_blocks_stale_two_parent_submission_race(tmp_path):
    one, two = function("one"), function("two")
    p1, p2 = plan(tmp_path, one, action="split_detector"), plan(
        tmp_path, two, action="split_detector",
    )
    child = {**one, "name": "shared_child", "source": one["source"] + "\n# split"}
    # Both callers observed no library child before either upsert; the ledger
    # still must give the name only one owner while library writes are pending.
    workflow.record_submission(tmp_path, child, p1["plan_id"], current_function=None)
    with pytest.raises(ValueError, match="another repair workflow"):
        workflow.record_submission(tmp_path, child, p2["plan_id"], current_function=None)
    assert workflow.validation_context(tmp_path, child)["plan_id"] == p1["plan_id"]


def test_late_retry_cannot_be_attributed_to_a_new_same_version_plan(tmp_path):
    fn = function(verdict="error")
    old = plan(tmp_path, fn, action="retry_review")
    workflow.reserve_validation(tmp_path, "f", ["diagnostic-1"],
                                expected_plan_id=old["plan_id"])
    workflow.defer_repair(tmp_path, "f", "Review interrupted")
    new = plan(tmp_path, fn, action="retry_review")
    with pytest.raises(ValueError, match="plan changed"):
        workflow.reserve_validation(tmp_path, "f", ["diagnostic-1"],
                                    expected_plan_id=old["plan_id"])
    workflow.reserve_validation(tmp_path, "f", ["diagnostic-1"],
                                expected_plan_id=new["plan_id"])
    reviewed = result(fn, item_id="diagnostic-1")
    fn["scores"]["behavior_verdicts"] = reviewed["verdicts"]
    with pytest.raises(ValueError, match="plan changed"):
        workflow.record_validation(tmp_path, fn, reviewed, eligible_items=1,
                                   expected_plan_id=old["plan_id"])
    assert workflow.get_repair(tmp_path, "f")["latest"]["result"] is None


def test_inflight_review_cannot_complete_an_inactive_plan(tmp_path):
    repair, candidate = reserved(tmp_path)
    workflow.defer_repair(tmp_path, "f", "Wait for more evidence")
    before = workflow.get_repair(tmp_path, "f")
    with pytest.raises(ValueError, match="no longer active"):
        workflow.record_validation(tmp_path, candidate, result(candidate), eligible_items=1,
                                   expected_plan_id=repair["plan_id"])
    assert workflow.get_repair(tmp_path, "f") == before


@pytest.mark.parametrize("operation", ["plan", "reserve"])
def test_concurrent_state_transitions_have_one_winner(tmp_path, operation):
    fn = function()
    if operation == "reserve":
        repair = plan(tmp_path, fn)
        submit(tmp_path, fn)
    barrier = Barrier(2)

    def attempt(index):
        barrier.wait(timeout=5)
        try:
            if operation == "plan":
                plan(tmp_path, fn)
            else:
                workflow.reserve_validation(tmp_path, "f", [f"holdout-{index}"],
                                            expected_plan_id=repair["plan_id"])
        except ValueError:
            return "rejected"
        return "accepted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, [1, 2])) == ["accepted", "rejected"]
    state = workflow.get_repair(tmp_path, "f")
    assert state["attempts_used"] == 1
    if operation == "reserve":
        assert len(state["latest"]["reservation"]["item_ids"]) == 1


def test_policy_refresh_requires_new_attempt_and_preserves_budget_and_exclusions(
    tmp_path, monkeypatch,
):
    old, candidate = reserved(tmp_path, ["old-holdout"])
    monkeypatch.setattr("autofa.metrics.claude_judge.evidence_projection_hash", lambda: "updated")
    with pytest.raises(ValueError, match="policy changed"):
        workflow.validation_context(tmp_path, candidate)
    workflow.defer_repair(tmp_path, "f", "Evidence policy changed")
    new = plan(tmp_path, candidate, action="gather_evidence")
    assert new["attempt"] == 2
    assert new["judge_policy_hash"] == judge_policy_hash()
    assert new["judge_policy_hash"] != old["judge_policy_hash"]
    context = workflow.validation_context(tmp_path, candidate)
    assert set(context["exclusion_item_ids"]) == {"diagnostic-1", "old-holdout"}
    workflow.defer_repair(tmp_path, "f", "Deferred second attempt")
    plan(tmp_path, candidate, action="gather_evidence")
    workflow.defer_repair(tmp_path, "f", "Deferred third attempt")
    with pytest.raises(ValueError, match="attempt limit"):
        plan(tmp_path, candidate, action="gather_evidence")


def test_claim_compilation_repair_requires_recorded_validation_failure(tmp_path):
    fn = function(verdict="error")
    with pytest.raises(ValueError, match="finding status"):
        plan(tmp_path, fn, action="revise_claim")
    fn["scores"]["claim_compilation"] = {
        "error": "claim compilation failed validation: coverage audit rejected the draft"
    }
    recorded = plan(tmp_path, fn, action="revise_claim")
    candidate = deepcopy(fn)
    candidate["mechanism"] = "A narrower directly testable behavior"
    workflow.record_submission(tmp_path, candidate)
    assert recorded["action"] == "revise_claim"
    assert workflow.validation_context(tmp_path, candidate)["action"] == "revise_claim"


def test_repair_keeps_original_strategy_metadata_but_accepts_same_review_policy(tmp_path):
    fn = function(verdict="error")
    fn["scores"].update(judge_objective_strategy="recall",
                        judge_policy_hash=judge_policy_hash())
    # A retry keeps the reviewed provenance even when the search strategy changed.
    recorded = plan(tmp_path, fn, action="retry_review", objective_strategy="precision")
    assert recorded["judge_objective_strategy"] == "recall"
    assert recorded["judge_policy_hash"] == fn["scores"]["judge_policy_hash"]
    context = workflow.validation_context(tmp_path, fn)
    assert context["judge_objective_strategy"] == "recall"
    workflow.reserve_validation(tmp_path, "f", ["diagnostic-1"])
    fresh = result(fn, item_id="diagnostic-1", judge_policy_hash=judge_policy_hash(),
                   judge_objective_strategy="precision")
    fn["scores"]["behavior_verdicts"] = deepcopy(fresh["verdicts"])
    completed = workflow.record_validation(tmp_path, fn, fresh, eligible_items=1)
    assert completed["phase"] == "completed"


def test_fresh_claim_repair_can_adopt_changed_strategy(tmp_path):
    fn = function()
    recorded = plan(tmp_path, fn, action="revise_claim", objective_strategy="recall")
    assert recorded["judge_objective_strategy"] == "recall"
    candidate = submit(tmp_path, fn, mechanism="An explicit shared causal family.")
    assert workflow.validation_context(tmp_path, candidate)["judge_objective_strategy"] == "recall"


def test_continue_history_excludes_archived_diagnostics_from_new_repair(tmp_path):
    previous = tmp_path / "previous"
    current = tmp_path / "continue"
    fn = function()
    first = plan(previous, fn)
    candidate = submit(previous, fn)
    workflow.reserve_validation(previous, "f", ["old-holdout"], expected_plan_id=first["plan_id"])
    workflow.record_validation(previous, candidate, result(candidate, item_id="old-holdout"),
                               eligible_items=1)
    library = FunctionLibrary(functions=[FeatureFunction.from_dict(candidate)])
    record_discovery_exposure(previous, ["read-during-discovery"])
    sync_review_history(library, previous)
    portable = FunctionLibrary.from_dict(library.to_dict())
    sync_review_history(portable, current)
    plan(current, candidate)
    revised = submit(current, candidate)
    context = workflow.validation_context(current, revised)
    assert set(context["exclusion_item_ids"]) == {
        "diagnostic-1", "old-holdout", "read-during-discovery",
    }
    with pytest.raises(ValueError, match="previously reviewed"):
        workflow.reserve_validation(current, "f", ["old-holdout"])


def test_same_task_or_legacy_limited_evidence_cannot_complete_repair(tmp_path):
    fn = function()
    plan(tmp_path, fn)
    candidate = submit(tmp_path, fn)
    workflow.reserve_validation(tmp_path, "f", ["holdout-1"], sample={
        "target_n": 1, "actual_n": 1, "eligible_n": 1, "limited": True,
        "task_group_limited": True, "independent_task_n": 0,
    })
    completed = workflow.record_validation(tmp_path, candidate, result(candidate), eligible_items=1)
    assert completed["phase"] == "deferred"
    assert completed["outcome"]["review_status"] == "insufficient_new_evidence"
    assert not completed["outcome"]["all_reviewed_confirmed"]
    assert not completed["outcome"]["original_claim_validated"]


def test_supported_review_exposes_measured_gaps_without_rejecting_any_claim():
    fn = supported_candidate()
    original = deepcopy(fn)
    feedback = build_repair_feedback(fn)
    gaps = [row for row in feedback["findings"] if row["status"] == "quality_gap"]
    assert {row["requirement_id"] for row in gaps} == {"recall", "ppv_at_pi0"}
    assert feedback["recommended_action"] == "retain"
    assert feedback["allowed_actions"] == ["refine_detector"]
    assert gaps[0]["requirement_id"] == "ppv_at_pi0"
    assert feedback["quality_refinement"]["requires_new_validation"]
    assert all(row["status"] in {"supported", "quality_gap"} for row in feedback["findings"])
    assert gaps[0]["item_ids"] == ["item-000", "missed", "passing"]
    assert has_complete_causal_support(fn, objective_strategy="balanced")
    assert fn == original


@pytest.mark.parametrize("change", [
    {"evaluation_claim_hash": "other-claim"}, {"evaluation_item_ids": None}, {"buckets": []},
], ids=["stale_evaluation", "unidentified_panel", "no_controls"])
def test_quality_refinement_requires_current_review_and_evaluated_controls(change):
    fn = supported_candidate()
    fn["scores"].update(change)
    feedback = build_repair_feedback(fn)
    assert "refine_detector" not in feedback["allowed_actions"]
    assert not any(row["status"] == "quality_gap" for row in feedback["findings"])


def test_quality_gap_cannot_authorize_contradiction_repair(tmp_path):
    with pytest.raises(ValueError, match="finding status"):
        propose(tmp_path, supported_candidate(), action="tighten_detector")


def test_quality_plan_requires_current_metric_cohort_and_active_policy(tmp_path):
    fn = supported_candidate()
    feedback = build_repair_feedback(fn)
    gap = next(row for row in feedback["findings"] if row["status"] == "quality_gap")
    fn["scores"]["ppv_at_pi0"] = .9
    fn["scores"]["recall"] = .3
    with pytest.raises(ValueError, match="current supported claim review"):
        workflow.plan_repair(
            tmp_path, fn, feedback, action="refine_detector", rationale="Old gap",
            finding_ids=[gap["id"]], evidence_ids=["passing"],
        )
    with pytest.raises(ValueError, match="current supported claim review"):
        propose(tmp_path, supported_candidate(), behavior_strictness="behavioral")


@pytest.mark.parametrize("change,edited,match", [
    ({"description": "Weaker claim"}, True, "preserve"),
    ({"explains": "parent"}, True, "preserve"),
    ({"kind": "python"}, False, "executed detector"),
], ids=["claim", "relationship", "metadata_only"])
def test_quality_revision_changes_execution_but_preserves_claim_and_relationships(
    tmp_path, change, edited, match,
):
    fn = supported_candidate()
    propose(tmp_path, fn)
    revised = {**deepcopy(fn), **change}
    if edited:
        revised["source"] += "\n# updated predicate"
    with pytest.raises(ValueError, match=match):
        workflow.record_submission(tmp_path, revised, current_function=fn)
    assert workflow.get_repair(tmp_path, fn["name"])["latest"]["after"] is None


def test_quality_revision_retains_baseline_and_requires_new_review(tmp_path):
    fn = supported_candidate()
    repair = propose(tmp_path, fn)
    before = deepcopy(fn)
    revised = {**deepcopy(fn), "source": fn["source"] + "\n# evidence-backed predicate"}
    workflow.record_submission(tmp_path, revised, current_function=fn)
    context = workflow.validation_context(tmp_path, revised)
    assert context["plan_id"] == repair["plan_id"]
    assert {"item-000", "item-001", "item-002", "passing", "missed"} <= set(
        context["exclusion_item_ids"])
    with pytest.raises(ValueError, match="previously reviewed"):
        workflow.reserve_validation(tmp_path, fn["name"], ["passing"])
    workflow.reserve_validation(tmp_path, fn["name"], ["fresh-task"])
    completed = workflow.record_validation(tmp_path, revised, {
        "claim_hash": claim_hash(**{key: revised[key]
                                    for key in ("source", "description", "mechanism")}),
        "judge_policy_hash": judge_policy_hash(), "judge_behavior_strictness": "causal",
        "verdicts": [{"item_id": "fresh-task", "verdict": "yes"}],
    }, eligible_items=10)
    assert completed["phase"] == "completed"
    assert completed["outcome"]["claim_preserved"]
    assert completed["outcome"]["automatic_promotion"] is False
    saved = workflow.get_repair(tmp_path, fn["name"])
    assert saved["baseline"] == before
    assert saved["attempts_used"] == 1
    assert saved["latest"]["before"] == before
    assert saved["latest"]["after"] == revised


def test_quality_feedback_next_action_explains_executable_refinement(tmp_path):
    rt = SimpleNamespace(config=Config(), run=SimpleNamespace(
        run_id="quality-run", run_dir=lambda: tmp_path,
    ))
    fn = supported_candidate()
    feedback = build_repair_feedback(fn)
    next_action = review_next_action(rt, fn)
    assert next_action["tool"] == "get_item", next_action
    assert "plan_function_repair" in next_action["reason"]
    assert "action='refine_detector'" in next_action["reason"]
    assert feedback["quality_refinement"]["finding_ids"][0] in next_action["reason"]
