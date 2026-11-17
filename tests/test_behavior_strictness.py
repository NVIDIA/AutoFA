# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each Judge Policy keeps literal claims, binds its repairs, and never borrows other reviews."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

from autofa.behavior_policy import BEHAVIOR_STRICTNESSES as PROFILES
from autofa.behavior_policy import acceptance_requirement_id
from autofa.config import Config
from autofa.core.library import FunctionLibrary
from autofa.mcp import runtime, tools_data
from autofa.mcp.tools_eval import _find_compatible_allocation, _review_key
from autofa.metrics import claim_compiler, claude_judge
from autofa.metrics.claim_calibration import summarize_judge_feedback
from autofa.metrics.claim_contract import (
    acceptance_requirement_ids,
    build_claim_contract,
    claim_hash,
    derive_claim_verdict,
)
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.discovery_feedback import build_discovery_feedback
from autofa.metrics.library_coverage import library_failure_coverage
from autofa.metrics.pareto import rank_functions
from autofa.metrics.repair_feedback import build_repair_feedback, has_complete_causal_support
from autofa.metrics.repair_validation import merge_processing_retry
from autofa.metrics.search_panel import build_search_panel
from autofa.metrics.search_ranking import search_ranking
from autofa.metrics.task_recall import build_task_recall
from autofa.metrics.verdicts import behavior_validation
from autofa.session import repair_workflow as workflow
from autofa.session.discovery_target import failure_mode_target
from autofa.session.polishing import current_claims
from autofa.session.review_evidence import allocate_review
from autofa.session.task_coverage import task_coverage_snapshot
from autofa.session.task_progress import observe_task_progress
from autofa.ui.activity import AgentActivity
from autofa.ui.explorer import _build_tree

from support.discovery import keeper, source_inventory
from support.repair_review import strictness_function

# Reviews saved under the looser policy must never count once the stricter one is active.
SAVED, ACTIVE = "behavioral", "causal"

CLAIM = {
    "description": "The assistant abandons a required computation after invoking a deadline.",
    "mechanism": "It substitutes a placeholder for that computation.",
    "source": "def detect(item): return True",
}
TEXT = "I will skip the required computation and submit this placeholder."


def _draft():
    return {
        "defining_clauses": [
            {"id": f"clause_{index}", "source_field": field, "quote": CLAIM[field],
             "predicate": CLAIM[field], "required_review_scope": "local"}
            for index, field in enumerate(("description", "mechanism"), 1)
        ],
        "exclusions": [],
    }


def _contract(profile):
    return build_claim_contract(
        **CLAIM, behavior_strictness=profile, defining_clauses=_draft()["defining_clauses"],
        compilation={
            "policy_hash": claim_compiler.compiler_policy_hash(profile), "model": "test",
            "exclusions": [], "coverage_review": {"approved": True, "issues": []},
        },
    )


def _assessments(contract):
    return {
        "claim_contract_id": contract["contract_id"],
        "claim_assessments": [
            {"requirement_id": requirement["id"], "status": "supported",
             "rationale": "Original evidence establishes the stated condition.",
             "citations": [{"citation_id": "cite-1"}], "counterevidence": []}
            for requirement in contract["requirements"]
        ],
    }


@pytest.mark.parametrize("profile", PROFILES)
def test_profiles_keep_literal_claims_but_have_distinct_acceptance_requirements(profile):
    contract = _contract(profile)
    assert contract["description"] == CLAIM["description"]
    assert contract["mechanism"] == CLAIM["mechanism"]
    assert contract["defining_clauses"] == _draft()["defining_clauses"]
    assert tuple(r["id"] for r in contract["requirements"] if r["kind"] == "acceptance_check") == (
        acceptance_requirement_ids(profile)
    )
    assert contract["claim_hash"] == claim_hash(**CLAIM)
    assert derive_claim_verdict(contract, _assessments(contract)) is True
    # Relaxing product acceptance cannot excuse an explicitly asserted condition, and
    # supported literal claims cannot bypass the selected acceptance check.
    literal = contract["requirements"][0]["id"]
    for requirement_id in (literal, *acceptance_requirement_ids(profile)):
        for status, expected in (("unknown", None), ("contradicted", False)):
            row = _assessments(contract)
            next(a for a in row["claim_assessments"]
                 if a["requirement_id"] == requirement_id).update(status=status)
            assert derive_claim_verdict(contract, row) is expected, (requirement_id, status)


def test_profiles_separate_contracts_and_review_caches_without_invalidating_predictions():
    contracts = [_contract(profile) for profile in PROFILES]
    assert len({c["contract_id"] for c in contracts}) == 3
    assert {c["claim_hash"] for c in contracts} == {claim_hash(**CLAIM)}
    assert len({claim_compiler.compiler_policy_hash(p) for p in PROFILES}) == 3
    assert len({judge_policy_hash(behavior_strictness=p) for p in PROFILES}) == 3
    for profile, contract in zip(PROFILES, contracts, strict=True):
        assert claim_compiler.validate_compiled_contract(
            contract, **CLAIM, behavior_strictness=profile,
        ) == contract
        for other in set(PROFILES) - {profile}:
            with pytest.raises(ValueError, match="policy provenance"):
                claim_compiler.validate_compiled_contract(
                    contract, **CLAIM, behavior_strictness=other,
                )
    tampered = deepcopy(contracts[0])
    tampered["behavior_strictness"] = "contextual"
    with pytest.raises(ValueError, match="modified"):
        derive_claim_verdict(tampered, _assessments(tampered))


def test_looser_prompts_do_not_reintroduce_unconditional_causal_gates():
    for profile in ("behavioral", "contextual"):
        for text in (
            claim_compiler.compiler_system_prompt(profile),
            claim_compiler.coverage_system_prompt(profile),
            claude_judge.judge_system_prompt(profile),
            claude_judge.verification_system_prompt(profile),
        ):
            assert "failure_contribution" not in text
            assert "{{" not in text


def test_compiler_and_coverage_audit_receive_selected_policy(tmp_path, monkeypatch):
    profile = "behavioral"
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        is_coverage = kwargs["system_prompt"] == claim_compiler.coverage_system_prompt(profile)
        return json.dumps({"approved": True, "issues": []} if is_coverage else _draft())

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    contract = claim_compiler.compile_claim_contract(
        **CLAIM, model="test", directory=tmp_path / "compiler", behavior_strictness=profile,
    )
    assert [call["system_prompt"] for call in calls] == [
        claim_compiler.compiler_system_prompt(profile),
        claim_compiler.coverage_system_prompt(profile),
    ]
    assert claim_compiler.validate_compiled_contract(
        contract, **CLAIM, behavior_strictness=profile,
    ) == contract


def test_initial_and_verification_reviews_cannot_reuse_another_policy(tmp_path, monkeypatch):
    calls = []

    def run(*, review, system_prompt=None, **kwargs):
        calls.append((review, system_prompt))
        review.read("detector_source")
        review.read("response_text")
        citation = review.cite("response_text", TEXT)["citations"][0]
        contract = review.snapshot["claim_contract"]
        row = _assessments(contract)
        for assessment in row["claim_assessments"]:
            assessment["citations"] = [{"citation_id": citation["citation_id"]}]
        row.update(item_id="case", snapshot_id=review.snapshot["snapshot_id"], scope="local")
        return json.dumps({"verdicts": [row], "behavior_specificity": 0.8})

    monkeypatch.setattr(claude_judge, "_run_review", run)
    results = []
    for profile in PROFILES:
        args = dict(
            name="claim", **CLAIM, cwd=tmp_path, model="test",
            items=[{"_item_id": "case", "response_text": TEXT}], assigned_labels=[True],
            behavior_strictness=profile, compiled_contract=_contract(profile),
        )
        result = claude_judge.judge_behavior_with_claude(**args)
        results.append(result)
        assert result["n_yes"] == 1 and result["n_errors"] == 0
        assert result["judge_behavior_strictness"] == profile
        assert result["reused_stages"] == 0
        assert calls[-2][1] == (None if profile == "causal"
                               else claude_judge.judge_system_prompt(profile))
        assert calls[-1][1] == claude_judge.verification_system_prompt(profile)
        assert claude_judge.judge_behavior_with_claude(**args)["reused_stages"] == 2
    assert len(calls) == 6
    assert len({r["judge_policy_hash"] for r in results}) == 3
    assert {r["claim_hash"] for r in results} == {claim_hash(**CLAIM)}


def _plan_repair(directory, fn, profile, action="tighten_detector"):
    feedback = build_repair_feedback(fn)
    expected_status = "processing_error" if action == "retry_review" else "contradicted"
    finding = next(f for f in feedback["findings"] if f["status"] == expected_status)
    return workflow.plan_repair(
        directory, fn, feedback, action=action,
        rationale="Address the observed mismatch under the selected validation policy.",
        finding_ids=[finding["id"]], evidence_ids=[finding["examples"][0]["item_id"]],
        behavior_strictness=profile,
    )


def _submit_candidate(directory, fn, repair):
    changed = deepcopy(fn)
    changed["source"] += "\n# revised predicate"
    workflow.record_submission(directory, changed, repair_plan_id=repair["plan_id"])
    return changed


def _validation_result(fn, profile):
    return {
        "claim_hash": claim_hash(**{k: fn[k] for k in ("source", "description", "mechanism")}),
        "judge_policy_hash": judge_policy_hash(behavior_strictness=profile),
        "judge_behavior_strictness": profile,
        "verdicts": [{"item_id": "fresh-task", "verdict": "yes"}],
    }


@pytest.mark.parametrize("profile", ["behavioral", "causal"])
def test_complete_support_requires_the_selected_profiles_actual_assessments(profile):
    fn = strictness_function(profile)
    assert has_complete_causal_support(
        fn, objective_strategy="balanced", behavior_strictness=profile,
    )
    for other in set(PROFILES) - {profile}:
        assert not has_complete_causal_support(
            fn, objective_strategy="balanced", behavior_strictness=other,
        )
    required_id = acceptance_requirement_id(profile) or "description"
    assessments = fn["scores"]["behavior_verdicts"][0]["claim_assessments"]
    assessments[:] = [a for a in assessments if a["requirement_id"] != required_id]
    assert not has_complete_causal_support(
        fn, objective_strategy="balanced", behavior_strictness=profile,
    )


def test_repair_plan_and_completion_freeze_the_selected_profile(tmp_path):
    profile = "contextual"
    fn = strictness_function(profile, "contradicted")
    repair = _plan_repair(tmp_path, fn, profile)
    assert repair["judge_behavior_strictness"] == profile
    assert repair["judge_policy_hash"] == judge_policy_hash(behavior_strictness=profile)
    changed = _submit_candidate(tmp_path, fn, repair)
    context = workflow.validation_context(tmp_path, changed, behavior_strictness=profile)
    assert context["judge_behavior_strictness"] == profile
    workflow.reserve_validation(tmp_path, fn["name"], ["fresh-task"])
    completed = workflow.record_validation(tmp_path, changed, _validation_result(changed, profile),
                                           eligible_items=1)
    assert completed["phase"] == "completed"
    assert completed["outcome"]["all_reviewed_confirmed"]


def test_policy_change_cannot_resume_or_complete_an_old_reserved_cohort(tmp_path):
    fn = strictness_function(SAVED, "contradicted")
    repair = _plan_repair(tmp_path, fn, SAVED)
    changed = _submit_candidate(tmp_path, fn, repair)
    reservation = workflow.reserve_validation(tmp_path, fn["name"], ["fresh-task"])
    with pytest.raises(ValueError, match="policy changed"):
        workflow.validation_context(
            tmp_path, changed, resume_reserved=True, behavior_strictness=ACTIVE,
        )
    with pytest.raises(ValueError, match="provenance"):
        workflow.record_validation(tmp_path, changed, _validation_result(changed, ACTIVE),
                                   eligible_items=1)
    context = workflow.validation_context(
        tmp_path, changed, resume_reserved=True, behavior_strictness=SAVED,
    )
    assert context["reserved_item_ids"] == reservation["item_ids"]
    assert workflow.get_repair(tmp_path, fn["name"])["latest"]["result"] is None


def test_processing_retry_cannot_change_policy(tmp_path):
    fn = strictness_function(SAVED, "error")
    repair = _plan_repair(tmp_path, fn, SAVED, action="retry_review")
    assert repair["judge_behavior_strictness"] == SAVED
    workflow.defer_repair(tmp_path, fn["name"], "Change the validation policy.")
    with pytest.raises(ValueError, match="same current claim and judge policy"):
        _plan_repair(tmp_path, fn, ACTIVE, action="retry_review")
    # A fresh judge result names its backend under "backend", not "judge_backend".
    fresh = {**fn["scores"], "backend": fn["scores"]["judge_backend"],
             "judge_behavior_strictness": ACTIVE, "verdicts": []}
    with pytest.raises(ValueError, match="judge policy"):
        merge_processing_retry(fn["scores"], fresh, {"item-000"})


def test_historical_repair_profiles_default_to_causal(tmp_path):
    fn = strictness_function("causal", "contradicted")
    fn["scores"].pop("judge_behavior_strictness")
    feedback = build_repair_feedback(fn)
    finding = next(f for f in feedback["findings"] if f["status"] == "contradicted")
    repair = workflow.plan_repair(
        tmp_path, fn, feedback, action="tighten_detector", rationale="Preserve legacy policy.",
        finding_ids=[finding["id"]], evidence_ids=[finding["examples"][0]["item_id"]],
    )
    assert repair["judge_behavior_strictness"] == "causal"
    changed = _submit_candidate(tmp_path, fn, repair)
    assert workflow.validation_context(tmp_path, changed)["judge_behavior_strictness"] == "causal"


def test_old_contract_cannot_be_relabelled_as_support_under_another_policy():
    changed = strictness_function(SAVED)
    changed["scores"].update(
        judge_behavior_strictness=ACTIVE,
        judge_policy_hash=judge_policy_hash(behavior_strictness=ACTIVE),
    )
    assert not has_complete_causal_support(
        changed, objective_strategy="balanced", behavior_strictness=ACTIVE,
    )
    assert build_repair_feedback(changed)["recommended_action"] == "gather_evidence"


def test_explicit_causal_mode_shares_the_default_review_key(tmp_path):
    fn = strictness_function("causal")
    settings = {"judge_model": "test", "judge_objective_strategy": "balanced"}
    # Hand-built current key: it omits the default causal profile and the search preference.
    expected_key = hashlib.sha256(json.dumps({
        "claim": fn["scores"]["claim_hash"], "policy": judge_policy_hash(),
        "plan": None, "settings": {"judge_model": "test"}, "evidence_n": 2,
    }, sort_keys=True).encode()).hexdigest()
    allocation = allocate_review(
        tmp_path, fn["name"],
        [{"_item_id": f"task-{i}", "benchmark": "b", "problem_id": str(i)} for i in range(5)],
        target_n=2, seed=17, review_key=expected_key, phase="development",
    )
    assert _review_key(fn, settings, 2) == expected_key
    rt = SimpleNamespace(run=SimpleNamespace(run_id=tmp_path.name, run_dir=lambda: tmp_path))
    for profile in PROFILES:
        key, saved = _find_compatible_allocation(
            rt, fn, {**settings, "judge_behavior_strictness": profile,
                     "judge_objective_strategy": "recall"}, 2, None, "development",
        )
        if profile == "causal":
            assert key == expected_key
            assert saved["item_ids"] == allocation["item_ids"]
        else:
            assert key != expected_key
            assert saved is None


def _workflow(profile, *, same_task=False):
    inventory = source_inventory(tasks=(
        {f"f{i}": "same-task" for i in range(10)} if same_task else {"f1": "task-0"}
    ))
    inventory.update(
        corpus_fingerprint="corpus", item_ids=list(inventory["task_by_item_id"]),
        n_failed_tasks=len(set(inventory["task_by_failed_item_id"].values())),
    )
    functions = [keeper(index, evaluation_run_id="current", behavior_n_judged=6,
                        behavior_n_completed=6, behavior_n_yes=6) for index in range(2)]
    for fn in functions:
        contract = build_claim_contract(
            source=fn.source, description=fn.description, mechanism=fn.mechanism,
            behavior_strictness=profile,
        )
        fn.scores.update(
            claim_contract=contract, judge_behavior_strictness=profile,
            judge_policy_hash=judge_policy_hash(behavior_strictness=profile),
        )
        for row in fn.scores["behavior_verdicts"]:
            row["claim_contract_id"] = contract["contract_id"]
            row["initial_review"]["claim_contract_id"] = contract["contract_id"]
        fn.scores.update(build_task_recall(fn, inventory, run_id="current"))
        fn.scores["search_panel"] = build_search_panel(
            fn, inventory, item_ids=inventory["item_ids"], run_id="current",
        )
    return SimpleNamespace(
        config=Config(behavior_strictness=profile),
        run=SimpleNamespace(run_id="current"),
        library=FunctionLibrary(functions=functions),
        _family_failed_task_inventory=inventory,
    )


def test_selected_policy_supports_minimum_confirmed_task_coverage_and_ranking():
    rt = _workflow("contextual")
    before = deepcopy(rt.library.to_dict())
    inventory = rt._family_failed_task_inventory
    target = failure_mode_target(rt.library.functions, minimum=2, config=rt.config,
                                 run_id="current", inventory=inventory)
    assert target["minimum_met"]
    assert target["n_qualifying_failure_modes"] == 2
    coverage = task_coverage_snapshot(rt, include_task_ids=True)
    assert coverage["n_confirmed_failed_tasks"] == 5
    assert coverage["confirmed_task_coverage"] == pytest.approx(5 / 9)
    assert all(row["eligible"] for row in coverage["detectors"])
    provisional = search_ranking(rt.library.functions, rt.config, run_id="current")
    assert len(provisional["ranked"]) == 2
    assert all(row["score"] > 0 for row in provisional["ranked"])
    ranked = rank_functions([fn.to_dict() for fn in rt.library.functions], config=rt.config)
    assert [row["ranking"]["eligible"] for row in ranked] == [True, True]
    assert rt.library.to_dict() == before


def test_changing_policy_requires_new_reviews_before_minimum_progress_or_ranking():
    rt = _workflow(SAVED)
    before = deepcopy(rt.library.to_dict())
    rt.config.behavior_strictness = ACTIVE
    inventory = rt._family_failed_task_inventory
    target = failure_mode_target(rt.library.functions, minimum=2, config=rt.config,
                                 run_id="current", inventory=inventory)
    assert not target["minimum_met"]
    assert target["n_qualifying_failure_modes"] == 0
    assert task_coverage_snapshot(rt)["n_confirmed_failed_tasks"] == 0
    assert search_ranking(rt.library.functions, rt.config, run_id="current")["ranked"] == []
    assert not any(row["ranking"]["eligible"] for row in rank_functions(
        [fn.to_dict() for fn in rt.library.functions], config=rt.config,
    ))
    # Detector executions stay useful evidence; only their review eligibility changes.
    observed = library_failure_coverage(
        rt.library, inventory["task_by_failed_item_id"], run_id="current",
        behavior_strictness=ACTIVE, include_item_ids=True,
    )
    assert all(row["evaluation_current"] for row in observed["detectors"])
    assert all(len(row["known_matched_failed_item_ids"]) == 8 for row in observed["detectors"])
    assert all(row["review_confirmed_failed_item_ids"] == [] for row in observed["detectors"])
    assert rt.library.to_dict() == before


def test_relaxed_policies_still_require_support_across_different_tasks():
    rt = _workflow("behavioral", same_task=True)
    target = failure_mode_target(
        rt.library.functions, minimum=2, config=rt.config, run_id="current",
        inventory=rt._family_failed_task_inventory,
    )
    assert not target["minimum_met"]
    assert target["n_qualifying_failure_modes"] == 0
    assert task_coverage_snapshot(rt)["n_confirmed_failed_tasks"] == 0
    assert all(row["measured"]["n_confirmed_failed_attempts"] == 6
               for row in target["pending_failure_modes"])
    assert all(row["measured"]["n_confirmed_failed_datapoints"] == 1
               for row in target["pending_failure_modes"])


def test_task_progress_resets_policy_checkpoint_without_recounting_historical_attempts(tmp_path):
    rt = _workflow(SAVED)
    rt.run.run_dir = lambda: tmp_path
    rt.scores_history = []
    observe_task_progress(rt, initialize=True)
    evaluation = {"evaluation_complete": True,
                  "evaluated_names": [fn.name for fn in rt.library.functions],
                  "evaluated_claims": current_claims(rt)}
    for _ in range(2):
        rt.scores_history.append(deepcopy(evaluation))
        old_state = observe_task_progress(rt)
    assert old_state["completed_attempts"] == old_state["experiments_since_improve"] == 2
    assert old_state["best_confirmed_failed_tasks"] == 5
    rt.config.behavior_strictness = ACTIVE
    changed = observe_task_progress(rt, initialize=True)
    assert changed["baseline_initialized"]
    assert changed["behavior_strictness"] == ACTIVE
    assert changed["best_confirmed_failed_tasks"] == 0
    assert changed["completed_attempts"] == changed["experiments_since_improve"] == 0
    assert not changed["improved"]
    # Reviewing under the new policy can now establish progress from its own baseline.
    rt.library = _workflow(ACTIVE).library
    reviewed = observe_task_progress(rt)
    assert reviewed["improved"]
    assert reviewed["confirmed_task_gain"] == reviewed["best_confirmed_failed_tasks"] == 5
    assert reviewed["completed_attempts"] == 0


def test_live_activity_and_explorer_do_not_promote_reviews_under_a_different_policy(
    tmp_path, monkeypatch,
):
    rt = _workflow(ACTIVE)
    stale = _workflow(SAVED).library.functions[0]
    current = rt.library.functions[1]
    rt.library.functions = [stale, current]
    rt.run.run_dir = lambda: tmp_path
    rt.scores_history = []
    before = deepcopy(rt.library.to_dict())
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    activity = AgentActivity(Console(force_terminal=False, no_color=True))
    lines = [Text.from_markup(line).plain
             for line in activity._library_snapshot(force=True)["lines"]]
    assert current.name in lines[0] and "adherence=1.00" in lines[0]
    assert stale.name in lines[1] and "adherence=—" in lines[1]
    assert "GOOD" not in lines[1]
    nodes = _build_tree(rt.library, objective_config=rt.config)
    assert nodes[0].data["name"] == current.name
    assert "Research candidate" not in nodes[0].label
    assert nodes[1].data["name"] == stale.name
    assert "Research candidate" in nodes[1].label
    assert "behavior_purity=—" in nodes[1].detail
    assert rt.library.to_dict() == before


def test_library_coverage_tool_does_not_count_another_policys_confirmations(monkeypatch):
    rt = _workflow(SAVED)
    before = deepcopy(rt.library.to_dict())
    monkeypatch.setattr(tools_data, "get_runtime", lambda: rt)
    current = tools_data._library_coverage_snapshot(include_item_ids=True)
    assert all(row["n_review_confirmed_failed"] == 6 for row in current["detectors"])
    rt.config.behavior_strictness = ACTIVE
    changed = tools_data._library_coverage_snapshot(include_item_ids=True)
    assert all(row["n_review_confirmed_failed"] == 0 for row in changed["detectors"])
    assert all(row["review_confirmed_failed_item_ids"] == [] for row in changed["detectors"])
    assert all(row["n_known_matched_failed"] == 8 for row in changed["detectors"])
    assert rt.library.to_dict() == before


def test_raw_review_feedback_requires_active_policy_even_with_supplied_summary():
    scores = _workflow(SAVED).library.functions[0].scores
    before = deepcopy(scores)
    original = behavior_validation(scores)
    assert original["status"] == "confirmed"
    changed = behavior_validation(scores, behavior_strictness=ACTIVE)
    assert changed["status"] == "stale"
    assert changed["judge_behavior_strictness"] == SAVED
    assert not changed["ranking_eligible"]
    summary = summarize_judge_feedback(scores, name="mode_0", behavior_strictness=ACTIVE)
    assert summary["status"] == "stale" and summary["findings"] == []
    for supplied in (None, original, {"status": "confirmed", "ranking_eligible": True}):
        feedback = build_discovery_feedback(
            scores, judge_feedback=supplied, behavior_strictness=ACTIVE,
        )
        assert feedback["status"] == "review_current"
        assert feedback["signals"]["review_status"] == "stale"
    assert scores == before


def test_unannotated_saved_causal_review_does_not_become_relaxed_confirmation():
    active = "contextual"
    scores = _workflow("causal").library.functions[0].scores
    scores.pop("judge_behavior_strictness")
    before = deepcopy(scores)
    assert behavior_validation(scores)["status"] == "confirmed"
    assert behavior_validation(scores, behavior_strictness=active)["status"] == "stale"
    assert summarize_judge_feedback(
        scores, name="mode_0", behavior_strictness=active,
    )["status"] == "stale"
    assert scores == before
