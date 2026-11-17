# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Due synthesis gets real diagnostics, truthful decisions and ordinary independent review."""

import hashlib
import json
from copy import deepcopy

import pytest

from autofa.config import Config
from autofa.mcp import tools_eval, tools_library, tools_repair
from autofa.mcp.server import TOOL_CALLABLES
from autofa.metrics import claude_judge
from autofa.objectives import apply_preset
from autofa.session.family_discovery import STATE_FILE, family_discovery
from autofa.session.research_status import write_research_status

from support.families import assess_recorded, evidence, record, reviewed_seed
from support.families import rt as rt
from support.judge import judge_result
from support.judge import runtime as runtime

SOURCE = "def candidate(item): return False"
SEEDS = ["narrow_one", "narrow_two"]


def decision(outcome="insufficient_definition", **changes):
    return {
        "outcome": outcome,
        "shared_relation": "An adopted unsupported premise causes the required action to fail.",
        "observable_conditions": ["A specific adopted premise is contradicted by source evidence.",
                                  "Using it leaves the requested result defective."],
        "support_item_ids": ["a1", "b1"], "negative_control_ids": ["c1"],
        "reason": "Source cases support the relation, but its operational scope is not tested.",
        "next_action": "Implement the conditions and retain the recovered negative control.",
        **changes,
    }


def decline(value, **changes):
    return family_discovery(
        action="no_coherent_family", evidence=[evidence(), evidence("b1"),
                                                evidence("c1", "boundary")],
        boundary_notes="A corrected rejected call is outside the proposed mechanism.",
        assessment="The scope needs a concrete test instead of a label-only conclusion.",
        decision=value, **changes,
    )


def close_due_synthesis():
    result = decline(decision(
        "no_shared_relation", reason="Checked source cases do not establish one shared operation.",
    ), seed_functions=SEEDS)
    assert result["ok"], result


def saved(rt):
    return json.loads((rt.run.run_dir() / STATE_FILE).read_text())


def schedule(rt, *, limited=False):
    for name, item_id in zip(SEEDS, ("a1", "b1"), strict=True):
        reviewed_seed(rt, name, [item_id], limited=limited)
    write_research_status(rt.run.run_dir(), {
        "task_coverage_progress": {"experiments_since_improve": 1},
    })
    assert family_discovery()["synthesis_required"]


def smoke_receipt(rt, monkeypatch, *, source=SOURCE, expected=None):
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: rt)
    monkeypatch.setattr(rt.dataset, "iter_batches", lambda **kwargs: iter([rt.dataset.items]),
                        raising=False)
    result = tools_eval.smoke_test_feature(
        source=source, item_ids=["a1", "b1", "c1"],
        expected_outputs=expected or {"a1": True, "b1": True, "c1": False},
    )
    assert result["ok"], result
    return result


# A due synthesis checkpoint cannot be skipped by another repair or round.


def test_round_does_not_advance_until_due_synthesis_is_inspected(rt, monkeypatch):
    monkeypatch.setattr("autofa.session.round_validation.validate_round",
                        lambda rt: {"ok": True, "message": "Synthesis-only fixture checked."})
    schedule(rt, limited=True)
    rt.run.save = lambda: None
    rt.run.exploration_ssot = False
    rt.run.exploration_seed = 0
    rt.run.min_failure_modes = 0
    rt.run.max_rounds = 8
    monkeypatch.setattr(tools_library, "get_runtime", lambda: rt)
    monkeypatch.setattr(tools_library, "_detector_exploration", lambda: {"message": "Explore"})
    before = rt.run.round
    result = tools_library.advance_round()
    assert not result["ok"]
    assert result["family_discovery"]["synthesis_required"]
    assert rt.run.round == before
    close_due_synthesis()
    assert tools_library.advance_round()["refinement_pending"]
    assert tools_library.advance_round(validation_response="Retain the checked seeds.")["ok"]
    assert rt.run.round == before + 1
    assert not family_discovery()["synthesis_required"]


def test_due_synthesis_blocks_only_new_mutating_repair_plans(rt, monkeypatch):
    schedule(rt, limited=True)
    monkeypatch.setattr(tools_repair, "get_runtime", lambda: rt)
    calls = []

    def plan(*args, **kwargs):
        calls.append(kwargs)
        return {"plan_id": "new-plan", "candidates": {}}

    monkeypatch.setattr(tools_repair.repair_workflow, "plan_repair", plan)
    args = dict(name=SEEDS[0], action="revise_claim", rationale="Inspect a grounded mismatch.",
                finding_ids=["finding"], evidence_ids=["a1"])
    result = tools_repair.plan_function_repair(**args)
    assert not result["ok"]
    assert result["family_discovery"]["synthesis_required"]
    assert not calls
    assert not (rt.run.run_dir() / "detector_repairs.json").exists()
    # Retiring a detector is not a new mutation, so the due synthesis never blocks it.
    assert tools_repair.plan_function_repair(
        name=SEEDS[0], action="retire", rationale="Evidence action",
        finding_ids=[], evidence_ids=[],
    )["ok"]
    assert len(calls) == 1 and family_discovery()["synthesis_required"]
    close_due_synthesis()
    assert tools_repair.plan_function_repair(**args)["ok"]
    assert len(calls) == 2


# Uncertain broad hypotheses get actual diagnostics and retain truthful outcomes.


def test_new_negative_conclusions_require_structured_decisions(rt):
    result = decline(None)
    assert not result["ok"]
    assert result["missing_arguments"] == ["decision"]
    assert result["decision_schema"]["properties"]["outcome"]["enum"]
    assert not saved(rt).get("comparisons")


def test_supported_generic_hypothesis_cannot_close_scheduled_synthesis(rt):
    schedule(rt)
    before = rt.library.to_dict()
    result = decline(decision(), seed_functions=SEEDS)
    assert not result["ok"] and result["decision_recorded"]
    assert result["synthesis_required"]
    assert result["completed_comparisons"] == 0
    assert not result["pending_decision"]["counts_as_completed_investigation"]
    assert rt.library.to_dict() == before
    assert not saved(rt).get("synthesis_checks")


def test_repeated_reworded_refusals_are_audited_without_more_budget(rt):
    first = decline(decision())
    again = decline(decision(reason="A rephrased generic-scope objection."))
    assert first["pending_decision"]["id"] == again["pending_decision"]["id"]
    assert again["decision_record_count"] == 1
    assert again["pending_decision"]["requests_for_same_evidence"] == 2
    assert first["remaining_comparisons"] == again["remaining_comparisons"]
    history = saved(rt)["decision_records"][first["pending_decision"]["id"]]
    assert history["earlier_interpretations"][0]["decision"]["reason"] == decision()["reason"]
    assert family_discovery("status")["pending_decision"] == again["pending_decision"]
    # Paraphrasing the relation itself does not hide that the same evidence is repeated.
    paraphrased = decline(decision(shared_relation="Different wording for an uncertain parent."))
    assert paraphrased["pending_decision"]["requests_for_same_evidence"] == 3
    assert paraphrased["completed_comparisons"] == 0


def test_legitimate_source_rejection_does_not_force_paid_prototype(rt, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Source rejection must not run a detector or judge")

    monkeypatch.setattr(tools_eval, "smoke_test_feature", forbidden)
    result = decline(decision(
        "evidence_contradicts", reason="Checked source requirements contradict this parent.",
    ))
    assert result["ok"], result
    assert result["completed_comparisons"] == 1
    assert result["assessment"]["decision"]["outcome"] == "evidence_contradicts"


def test_operationalized_family_links_only_explicit_prior_decision(rt):
    pending = decline(decision())["pending_decision"]
    result = record(prior_decision_id=pending["id"])
    assert result["ok"], result
    assert result["pending_decision"] is None
    state = saved(rt)
    assert state["survey"]["decision_id"] == pending["id"]
    assert state["decision_records"][pending["id"]]["status"] == "under_investigation"
    assessed = assess_recorded(rt)
    assert assessed["ok"]
    assert saved(rt)["decision_records"][pending["id"]]["status"] == "evaluated"


@pytest.mark.parametrize("replacement", ["new_decision", "none"])
def test_deferral_closes_linked_decisions_and_preserves_family(rt, monkeypatch, replacement):
    pending = decline(decision())["pending_decision"]["id"]
    recorded = record(prior_decision_id=pending)
    assert recorded["ok"], recorded
    before = saved(rt)
    assert before["decision_records"][pending]["status"] == "under_investigation"
    library_before = rt.library.to_dict()

    deferred_decision = None
    if replacement == "new_decision":
        receipt = smoke_receipt(rt, monkeypatch)["diagnostic_receipt_id"]
        deferred_decision = decision(
            "implementation_failed", experiment_id=receipt,
            source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        )
    result = family_discovery(
        "defer_family", family_name="rejected_required_call",
        blocker="The implementation still needs investigation.",
        next_step="Revisit the operational conditions before rejecting the mechanism.",
        decision=deferred_decision,
    )
    assert result["ok"], result
    state = saved(rt)
    comparison = state["comparisons"][-1]
    linked_id = comparison["survey"]["decision_id"]
    assert (linked_id != pending) == (replacement == "new_decision")
    for identifier in {pending, linked_id}:
        audit = state["decision_records"][identifier]
        assert (audit["status"], audit["family_name"]) == ("deferred", "rejected_required_call")
    original = before["decision_records"][pending]["decision"]
    assert state["decision_records"][pending]["decision"] == original
    assert comparison["assessment"]["outcome"] == "deferred"
    assert result["completed_comparisons"] == recorded["completed_comparisons"] + 1
    assert family_discovery("status")["pending_decision"] is None
    assert rt.library.to_dict() == library_before
    assert not rt.scores_history


def test_unrelated_family_does_not_silently_resolve_pending_decision(rt):
    pending = decline(decision())["pending_decision"]
    assert record()["ok"]
    assert saved(rt)["decision_records"][pending["id"]]["status"] == "unresolved"


def test_autonomous_smoke_response_supplies_complete_family_deferral_contract(rt, monkeypatch):
    """A tested implementation failure defers, linking native tool outputs as they are."""
    before = rt.library.to_dict()
    pending = decline(decision())
    assert not pending["ok"] and pending["completed_comparisons"] == 0
    trial = smoke_receipt(rt, monkeypatch)
    assert not trial["checks_passed"]
    result = decline(decision(
        "implementation_failed", experiment_id=trial["diagnostic_receipt_id"],
        source_sha256=trial["source_sha256"],
        reason="The executed predicate misses both declared support controls.",
    ))
    assert result["ok"], result
    assert result["assessment"]["outcome"] == "deferred"
    assert result["completed_comparisons"] == 1
    assert result["pending_decision"] is None
    audit = result["assessment"]["decision"]
    assert audit["experiment_id"] == trial["diagnostic_receipt_id"]
    assert audit["experiment"]["checks_passed"] is False
    receipt_path = rt.run.run_dir() / "diagnostic_receipts" / f"{audit['experiment_id']}.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["source_sha256"] == trial["source_sha256"]
    assert receipt["results"] == trial["results"]
    assert rt.library.to_dict() == before and not rt.scores_history


@pytest.mark.parametrize("change", ["source", "expectations", "stale", "passes"])
def test_diagnostic_receipt_must_match_actual_failed_controls(rt, monkeypatch, change):
    source = ("def candidate(item): return 'corrected' not in item['response_text']"
              if change == "passes" else SOURCE)
    # The experiment declared a1 as a negative control, so it cannot certify a1 as support.
    expected = {"a1": False, "b1": True, "c1": False} if change == "expectations" else None
    receipt = smoke_receipt(
        rt, monkeypatch, source=source, expected=expected)["diagnostic_receipt_id"]
    proposal = decision("implementation_failed", experiment_id=receipt,
                        source_sha256=hashlib.sha256(source.encode()).hexdigest())
    if change == "source":
        proposal["source_sha256"] = "0" * 64
    elif change == "stale":
        rt.dataset.items[0]["response_text"] += " New decisive evidence."
    result = decline(proposal)
    assert not result["ok"], result
    assert not saved(rt).get("comparisons")


def test_unavailable_evidence_defers_without_inventing_supported_parent(rt):
    result = family_discovery(
        action="no_coherent_family",
        evidence=[evidence("a1", "uncertain"), evidence("b1", "uncertain")],
        boundary_notes="The omitted tool result prevents checking the claimed common effect.",
        assessment="No final result source is available to resolve the proposed mechanism.",
        decision={"outcome": "evidence_unavailable", "reason": "Original tool result missing.",
                  "unavailable_evidence": ["The original tool result and its failure cause."],
                  "next_action": "Retrieve the missing tool result if it becomes available."},
    )
    assert result["ok"], result
    assert result["assessment"]["outcome"] == "deferred"
    assert not rt.library.functions
    assert result["completed_comparisons"] == 1


def test_unavailable_evidence_requires_recorded_gap_not_just_a_label(rt):
    result = decline(decision("evidence_unavailable"))
    assert not result["ok"]
    assert "unavailable_evidence" in result["error"]


def test_exhausted_diagnostic_allowance_allows_honest_deferral(rt, monkeypatch):
    schedule(rt)
    monkeypatch.setattr("autofa.session.smoke_work.diagnostic_budget",
                        lambda rt: {"exhausted": True})
    result = decline(decision(), seed_functions=SEEDS)
    assert result["ok"] and not result["synthesis_required"]
    assert result["completed_comparisons"] == 1
    assert result["assessment"]["outcome"] == "deferred"
    assert result["assessment"]["decision"]["deferral_basis"] == "diagnostic_budget_exhausted"


def test_exhausted_comparison_budget_does_not_grant_an_extra_prototype(rt):
    rt.config = apply_preset(Config(), "precision")
    for name, item_id in zip(SEEDS, ("a1", "b1"), strict=True):
        reviewed_seed(rt, name, [item_id])
    assert record()["ok"]
    assert assess_recorded(rt)["synthesis_required"]
    result = decline(decision(), seed_functions=SEEDS)
    assert result["ok"] and not result["synthesis_required"]
    assert result["completed_comparisons"] == result["comparison_budget"] == 1
    assert result["recent_decisions"][-1]["status"] == "deferred"
    assert saved(rt)["history"][-1]["disposition"] == "deferred"


def test_historical_unstructured_decisions_remain_readable(rt):
    state = family_discovery()
    original = saved(rt)
    original["history"].append({"round": 0, "outcome": "synthesis_inspection_only",
                                "assessment": "Legacy inspected incompatible causes."})
    (rt.run.run_dir() / STATE_FILE).write_text(json.dumps(original))
    resumed = family_discovery("status")
    assert resumed["comparison_budget"] == state["comparison_budget"]
    assert saved(rt)["history"] == original["history"]
    assert not saved(rt).get("decision_records")


@pytest.mark.parametrize("outcome", ["capability_improved", "no_demonstrated_improvement"])
def test_classifier_limitation_requires_completed_comparison_improvement(rt, monkeypatch, outcome):
    source_hash = hashlib.sha256(SOURCE.encode()).hexdigest()
    comparison = {
        "run_id": rt.run.run_id, "source_sha256": source_hash,
        "status": "complete", "outcome": outcome, "profile": "fallback",
        "item_ids": ["a1", "b1", "c1"],
        "expected_outputs": {"a1": True, "b1": True, "c1": False},
    }
    monkeypatch.setattr("autofa.session.classifier_profiles.read_classifier_comparison",
                        lambda rt, identifier: comparison)
    result = decline(decision("classifier_limited", comparison_id="1" * 64,
                              source_sha256=source_hash))
    if outcome == "capability_improved":
        assert result["ok"], result
        assert result["assessment"]["outcome"] == "deferred"
        assert result["assessment"]["decision"]["comparison"]["outcome"] == outcome
    else:
        assert not result["ok"]
        assert "did not demonstrate improvement" in result["error"]
        assert not saved(rt)["comparisons"]


def test_cannot_claim_classifier_limit_from_a_failed_smoke_alone(rt, monkeypatch):
    receipt = smoke_receipt(rt, monkeypatch)["diagnostic_receipt_id"]
    result = decline(decision("classifier_limited", experiment_id=receipt,
                              source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest()))
    assert not result["ok"]
    assert "comparison_id" in result["error"]
    assert not saved(rt)["comparisons"]


# Bottom-up synthesis goes through ordinary independent detector evaluation and judging.


def test_native_bottom_up_submission_evaluates_new_cases_and_reviews_parent_independently(
    runtime, monkeypatch,
):
    # This regression uses frozen causal judge_result payloads and seed evidence.
    runtime.config.behavior_strictness = "causal"
    for index, item in enumerate(runtime.dataset.items):
        item["response_text"] = (
            "Unrecovered rejected call; required operation was never performed."
            if index < 10 else "Rejected call corrected; required operation completed."
        )
    first = reviewed_seed(runtime, "invalid_call", ["case-0"], limited=True)
    second = reviewed_seed(runtime, "missing_argument", ["case-1"], limited=True)
    originals = deepcopy([first.to_dict(), second.to_dict()])
    description = "Rejected calls leave required operations unperformed"
    mechanism = "An unrecovered rejected prerequisite call prevents the required operation."
    evidence_rows = [{
        "item_id": item_id, "role": role,
        "causal_chain": "A rejected prerequisite leaves the required operation undone."
        if role == "support" else "Correcting the rejected call completes the operation.",
        "quotes": [{"field": "response_text", "quote": quote}],
    } for item_id, role, quote in (
        ("case-0", "support", "Unrecovered rejected call"),
        ("case-1", "support", "Unrecovered rejected call"),
        ("case-10", "boundary", "Rejected call corrected"),
    )]
    recorded = TOOL_CALLABLES["family_discovery"](
        action="record_family", family_name="unrecovered_calls",
        description=description, mechanism=mechanism, evidence=evidence_rows,
        boundary_notes="Exclude calls corrected before the required operation completes.",
        seed_functions=[first.name, second.name],
    )
    assert recorded["ok"], recorded
    assert recorded["approach"] == "bottom_up"

    calls = []

    def reject_parent(**kwargs):
        calls.append(kwargs)
        parent = runtime.library.get("unrecovered_calls")
        assert kwargs["name"] == parent.name
        assert kwargs["mechanism"] == mechanism
        assert parent.depends_on == [] and parent.explains is None
        assert "dependency_bundle" not in parent.meta
        assert not parent.scores.get("behavior_verdicts")
        assert parent.scores["n_fail_caught"] == 10  # Seeds cover only two failures.
        result = judge_result(parent, kwargs["items"])
        return {**result, "n_yes": 0, "n_no": len(kwargs["items"]),
                "behavior_purity": 0.0,
                "verdicts": [{"item_id": item["_item_id"], "verdict": "no"}
                             for item in kwargs["items"]]}

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", reject_parent)
    submitted = tools_library.submit_function(
        name="unrecovered_calls", description=description, mechanism=mechanism,
        source="def unrecovered_calls(item):\n"
               "    return 'Unrecovered rejected call' in item['response_text']\n",
    )
    assert submitted["ok"] and submitted["evaluated"], submitted
    assert submitted["judge_refresh"]["refreshed"], submitted
    assert len(calls) == 1
    assert any(item["_item_id"] not in {"case-0", "case-1"} for item in calls[0]["items"])
    parent = runtime.library.get("unrecovered_calls")
    assert parent.scores["behavior_n_yes"] == 0
    assert parent.scores["behavior_n_no"] > 0
    assert all(row["verdict"] == "no" for row in parent.scores["behavior_verdicts"])
    assessed = family_discovery(action="assess_family", family_name=parent.name,
                                assessment="Reject the broader claim under its own causal review.")
    assert assessed["ok"], assessed
    assert assessed["assessment"]["review"]["rejected"] > 0
    assert assessed["assessment"]["review"]["confirmed"] == 0
    assert len(calls) == 1  # Recording and assessment add no separate reviewer.
    assert [first.to_dict(), second.to_dict()] == originals
