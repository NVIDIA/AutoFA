# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution repairs are grounded, bounded edits; workflow feedback never changes scores."""

import json
from copy import deepcopy

import pytest

from autofa.core.dependencies import review_source
from autofa.mcp import tools_eval, tools_library, tools_repair
from autofa.metrics.claim_contract import build_claim_contract, claim_hash
from autofa.metrics.repair_feedback import build_repair_feedback, detector_execution_findings
from autofa.metrics.review_scores import apply_review_result
from autofa.session import repair_workflow as workflow

from support.repair_review import _result, _runtime, strictness_function

# Parses serialized JSON, so plain-text rows of the feedback dataset raise JSONDecodeError.
JSON_SOURCE = (
    "import json\n"
    "def guess(item, llm_client=None):\n"
    "    return 'guess' in json.loads(item['response_text'])\n"
)


def fingerprint(fn):
    return claim_hash(source=review_source(fn), description=fn["description"],
                      mechanism=fn["mechanism"])


def errored_function(profile="contextual"):
    fn = strictness_function(profile)
    fn["scores"].update(
        evaluated=True, evaluation_claim_hash=fingerprint(fn), errors=1,
        error_examples=[{"item_id": "parse-failure", "error": "JSONDecodeError: invalid JSON"}],
    )
    return fn


def plan(directory, fn, *, feedback=None, **kwargs):
    feedback = feedback or build_repair_feedback(fn)
    finding = next(row for row in feedback["findings"] if row["status"] == "execution_error")
    return workflow.plan_repair(
        directory, fn, feedback, action="repair_implementation",
        rationale="Parse the recorded input representation without changing the written claim.",
        finding_ids=[finding["id"]], evidence_ids=[finding["item_ids"][0]], **kwargs,
    )


def evaluated_runtime(tmp_path, monkeypatch, *, source=None, partial=False):
    rt, fn = _runtime(tmp_path, monkeypatch)
    if source:
        fn.source = source
    if partial:
        # A partial failure still has matches that ordinarily trigger automatic judging.
        rt.dataset.items[0]["response_text"] = json.dumps("I guess the value.")
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    return rt, fn


def submit_still_broken(fn, plan_id, *, auto_evaluate=False):
    return tools_library.submit_function(
        name=fn.name, source=fn.source.replace("json.loads(", "json.loads( "),
        description=fn.description, mechanism=fn.mechanism,
        repair_plan_id=plan_id, auto_evaluate=auto_evaluate,
    )


def reevaluated_broken_repair(tmp_path, monkeypatch):
    rt, fn = evaluated_runtime(tmp_path, monkeypatch, source=JSON_SOURCE, partial=True)
    proposed = plan(rt.run.run_dir(), fn.to_dict(), behavior_strictness="causal")
    assert submit_still_broken(fn, proposed["plan_id"])["ok"]
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    return rt, fn, proposed


def exhaust_search(rt, name):
    for _ in range(workflow.MAX_ATTEMPTS):
        response = tools_repair.plan_function_repair(
            name, "gather_evidence", "Inspect the saved cases", [], [])
        assert response["ok"], response
        assert tools_repair.defer_function_repair(name, "No further evidence obtained")["ok"]
    assert workflow.repair_attempt_budget(workflow.get_repair(rt.run.run_dir(), name))[
        "remaining"] == 0


def submit_repair(fn, source, proposal, **changes):
    return tools_library.submit_function(
        name=fn.name, source=source, description=fn.description,
        mechanism=changes.pop("mechanism", fn.mechanism), auto_evaluate=False,
        repair_plan_id=proposal["plan"]["plan_id"], **changes)


# Recorded execution errors, independent of semantic contradictions.

def test_supported_review_still_allows_grounded_execution_repair():
    fn = errored_function()
    before = deepcopy(fn)
    feedback = build_repair_feedback(fn)
    assert feedback["recommended_action"] == "repair_implementation"
    assert "repair_implementation" in feedback["allowed_actions"]
    assert "tighten_detector" not in feedback["allowed_actions"]
    assert feedback["review_detail_complete"]
    assert feedback["validation"]["confirmed"] == 1
    findings = detector_execution_findings(fn)
    assert {finding["status"] for finding in findings} == {"execution_error"}
    assert {item for finding in findings for item in finding["item_ids"]} == {"parse-failure"}
    assert not any(row["status"] == "contradicted" for row in feedback["findings"])
    assert fn == before


@pytest.mark.parametrize("patch", [
    {"evaluated": False}, {"evaluation_stale": True}, {"evaluation_claim_hash": "old"},
    {"errors": 0}, {"error_examples": [{"item_id": "", "error": "JSONDecodeError"}]},
], ids=["unevaluated", "stale_evaluation", "stale_claim", "no_errors", "unidentified_example"])
def test_execution_repair_rejects_unrecorded_or_stale_errors(tmp_path, patch):
    fn = errored_function()
    valid_feedback = build_repair_feedback(fn)
    fn["scores"].update(patch)
    assert not detector_execution_findings(fn)
    assert "repair_implementation" not in build_repair_feedback(fn)["allowed_actions"]
    # A client cannot pass formerly valid feedback to bypass the current score check.
    with pytest.raises(ValueError):
        plan(tmp_path, fn, feedback=valid_feedback)
    assert not workflow.get_repair(tmp_path, fn["name"])


def test_judge_processing_error_alone_does_not_authorize_editing_detector():
    fn = strictness_function("contextual", "error")
    fn["scores"].update(evaluated=True, evaluation_claim_hash=fingerprint(fn), errors=0)
    feedback = build_repair_feedback(fn)
    assert not detector_execution_findings(fn)
    assert "retry_review" in feedback["allowed_actions"]
    assert "repair_implementation" not in feedback["allowed_actions"]


def test_copied_execution_finding_cannot_substitute_a_different_recorded_error(tmp_path):
    fn = errored_function()
    feedback = build_repair_feedback(fn)
    fn["scores"]["error_examples"] = [
        {"item_id": "different-task", "error": "KeyError: new_input"},
    ]
    assert detector_execution_findings(fn)
    with pytest.raises(ValueError):
        plan(tmp_path, fn, feedback=feedback)
    assert not workflow.get_repair(tmp_path, fn["name"])


@pytest.mark.parametrize("field,value,source_changed", [
    ("llm_prompt", "Changed metadata without changing the source.", False),
    ("mechanism", "A different failure mechanism.", True),
    ("depends_on", ["new_parent"], True),
])
def test_implementation_repair_needs_an_execution_change_without_changing_claim_or_relations(
    tmp_path, field, value, source_changed,
):
    fn = errored_function()
    repair = plan(tmp_path, fn)
    changed = deepcopy(fn)
    if source_changed:
        changed["source"] += "\n# repair parser"
    changed[field] = value
    with pytest.raises(ValueError):
        workflow.record_submission(tmp_path, changed, repair["plan_id"], current_function=fn)
    assert workflow.get_repair(tmp_path, fn["name"])["latest"]["candidates"] == {}


def test_execution_repair_uses_existing_budget_and_requires_fresh_validation(tmp_path):
    profile = "causal"
    fn = errored_function(profile)
    original = deepcopy(fn)
    repair = plan(tmp_path, fn, behavior_strictness=profile)
    with pytest.raises(ValueError, match="must change"):
        workflow.record_submission(tmp_path, fn, repair["plan_id"], current_function=fn)
    changed = deepcopy(fn)
    changed["source"] += "\n# repair parser"
    workflow.record_submission(tmp_path, changed, repair["plan_id"], current_function=fn)
    context = workflow.validation_context(tmp_path, changed, behavior_strictness=profile)
    assert context["sample_kind"] == "fresh_repair_holdout"
    assert {"parse-failure", "item-000"} <= set(context["exclusion_item_ids"])
    assert context["judge_behavior_strictness"] == profile
    state = workflow.get_repair(tmp_path, fn["name"])
    assert state["baseline"] == original
    assert state["attempts_used"] == 1
    assert state["latest"]["result"] is None
    workflow.defer_repair(tmp_path, fn["name"], "Parser fix still needs independent review.")
    # Deferral and unchanged passing semantic reviews do not replenish the budget.
    for _ in range(workflow.MAX_ATTEMPTS - 1):
        plan(tmp_path, fn, behavior_strictness=profile)
        workflow.defer_repair(tmp_path, fn["name"], "Need a different implementation fix.")
    with pytest.raises(ValueError, match="limit"):
        plan(tmp_path, fn, behavior_strictness=profile)
    assert workflow.get_repair(tmp_path, fn["name"])["attempts_used"] == workflow.MAX_ATTEMPTS


# Repair tools on a live runtime.

def test_real_json_error_can_be_repaired_after_supported_review_and_prior_plan(
    tmp_path, monkeypatch,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    reviewed = strictness_function("causal")
    fn.description = reviewed["description"]
    fn.mechanism = reviewed["mechanism"]
    fn.source = (
        "import json\n"
        "def guess(item, llm_client=None):\n"
        "    text = json.loads(item['response_text'])\n"
        "    return 'guess' in text\n"
    )
    for item in rt.dataset.items:
        item["response_text"] = json.dumps("I guess the value.")
    rt.dataset.items[2]["response_text"] = "I guess the value."
    rt.dataset.items[3]["response_text"] = json.dumps("A derived value.")
    evaluated = tools_eval.evaluate_functions(names=[fn.name], mode="full")
    assert evaluated["ok"], evaluated
    assert fn.scores["errors"] == 1
    assert "JSONDecodeError" in fn.scores["error_examples"][0]["error"]
    gathered = tools_repair.plan_function_repair(
        name=fn.name, action="gather_evidence", rationale="Inspect the review evidence.",
        finding_ids=[], evidence_ids=[],
    )
    assert gathered["ok"], gathered
    assert tools_repair.defer_function_repair(fn.name, "Keep the diagnostic history.")["ok"]

    # A fully supported review must not hide a separately recorded parser failure.
    contract = build_claim_contract(
        source=fn.source, description=fn.description, mechanism=fn.mechanism,
        defining_clauses=reviewed["scores"]["claim_contract"]["defining_clauses"],
    )
    reviewed["scores"].update(claim_contract=contract, claim_hash=fingerprint(fn.to_dict()))
    reviewed["scores"]["behavior_verdicts"][0].update(
        item_id="case-0", claim_contract_id=contract["contract_id"],
    )
    fn.scores.update(reviewed["scores"])
    baseline = deepcopy(fn.to_dict())
    fixed_source = (
        "import json\n"
        "def guess(item, llm_client=None):\n"
        "    try:\n"
        "        text = json.loads(item['response_text'])\n"
        "    except json.JSONDecodeError:\n"
        "        text = item['response_text']\n"
        "    return 'guess' in text\n"
    )
    args = dict(name=fn.name, source=fixed_source, description=fn.description,
                mechanism=fn.mechanism, auto_evaluate=False)
    rejected = tools_library.submit_function(**args)
    assert not rejected["ok"]
    assert "repair plan" in rejected["error"]
    assert rt.library.get(fn.name).to_dict() == baseline
    feedback = tools_repair.get_function_repair(fn.name)
    finding = next(row for row in feedback["feedback"]["findings"]
                   if row["status"] == "execution_error")
    proposed = tools_repair.plan_function_repair(
        name=fn.name, action="repair_implementation", rationale="Accept the recorded text format.",
        finding_ids=[finding["id"]], evidence_ids=finding["item_ids"],
    )
    assert proposed["ok"], proposed
    submitted = tools_library.submit_function(**args, repair_plan_id=proposed["plan"]["plan_id"])
    assert submitted["ok"], submitted
    reevaluated = tools_eval.evaluate_functions(names=[fn.name], mode="full")
    assert reevaluated["ok"], reevaluated
    current = rt.library.get(fn.name).to_dict()
    assert current["scores"]["errors"] == 0
    assert current["scores"]["n_scored"] == 4
    assert "repair_implementation" not in build_repair_feedback(current)["allowed_actions"]
    state = workflow.get_repair(rt.run.run_dir(), fn.name)
    assert state["attempts_used"] == 2
    assert state["latest"]["before"] == baseline
    assert state["latest"]["result"] is None
    context = workflow.validation_context(rt.run.run_dir(), current)
    assert {"case-0", "case-2"} <= set(context["exclusion_item_ids"])
    assert context["sample_kind"] == "fresh_repair_holdout"


def test_still_broken_submitted_repair_advises_deferral_before_another_edit(
    tmp_path, monkeypatch,
):
    rt, fn = evaluated_runtime(tmp_path, monkeypatch, source=JSON_SOURCE)
    feedback = tools_repair.get_function_repair(fn.name)["feedback"]
    finding = next(row for row in feedback["findings"] if row["status"] == "execution_error")
    proposed = tools_repair.plan_function_repair(
        fn.name, "repair_implementation", "Handle the recorded input format.",
        [finding["id"]], finding["item_ids"],
    )
    assert proposed["ok"], proposed
    # The proposed parser change still fails; do not spend a judge call on it.
    submitted = submit_still_broken(fn, proposed["plan"]["plan_id"])
    assert submitted["ok"], submitted
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    response = tools_repair.get_function_repair(fn.name)
    assert response["workflow"]["state"] == "awaiting_validation"
    assert response["next_action"]["tool"] == "defer_function_repair"
    assert response["next_action"]["arguments"]["name"] == fn.name
    assert response["next_action"]["arguments"]["reason"]
    assert response["workflow"]["attempts_used"] == 1

    # Once a review has reserved evidence, recovery takes precedence over a new edit.
    workflow.reserve_validation(rt.run.run_dir(), fn.name, ["fresh-review-task"])
    response = tools_repair.get_function_repair(fn.name)
    assert response["next_action"]["tool"] == "judge_function"
    assert "original evidence and budget" in response["next_action"]["reason"]
    assert response["workflow"]["attempts_used"] == 1


def test_automatic_submission_cannot_review_a_still_broken_repair(tmp_path, monkeypatch):
    from autofa.metrics import claude_judge
    from autofa.session import review_evidence

    rt, fn = evaluated_runtime(tmp_path, monkeypatch, source=JSON_SOURCE, partial=True)
    proposed = plan(rt.run.run_dir(), fn.to_dict(), behavior_strictness="causal")
    calls = []

    def unexpected(*args, **kwargs):
        calls.append(True)
        raise AssertionError("No allocation or paid review is allowed for this broken repair")

    monkeypatch.setattr(review_evidence, "allocate_review", unexpected)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", unexpected)
    submitted = submit_still_broken(fn, proposed["plan_id"], auto_evaluate=True)
    assert submitted["ok"], submitted
    current = rt.library.get(fn.name)
    assert current.scores["errors"] == 3
    assert current.scores["n_flagged"] == 1
    for phase in ("auto", "development", "confirmation"):
        result = tools_eval.judge_function(fn.name, review_phase=phase)
        assert not result["ok"], result
        assert result["repair_execution_blocked"]
        assert result["next_action"]["tool"] == "defer_function_repair"
    state = workflow.get_repair(rt.run.run_dir(), fn.name)
    assert state["latest"]["candidates"][fn.name]["reservation"] is None
    assert state["latest"]["result"] is None
    assert calls == []


@pytest.mark.parametrize("recovery", ["reservation", "processing_retry", "allocation"])
def test_execution_errors_do_not_block_existing_review_recovery(
    tmp_path, monkeypatch, recovery,
):
    from autofa.session import candidate_screen, review_evidence

    rt, fn, _ = reevaluated_broken_repair(tmp_path, monkeypatch)
    if recovery == "reservation":
        workflow.reserve_validation(rt.run.run_dir(), fn.name, ["case-0"])
    elif recovery == "processing_retry":
        context = workflow.validation_context(rt.run.run_dir(), rt.library.get(fn.name).to_dict())
        context.update(retry_item_ids=["case-0"], action="retry_review")
        monkeypatch.setattr(workflow, "validation_context", lambda *args, **kwargs: context)
    else:
        monkeypatch.setattr(review_evidence, "find_review_allocation", lambda *args, **kwargs: {
            "item_ids": ["case-0"], "operation_completed": False,
        })
    screens = []

    def screen(*args, check_evidence_prerequisites=True):
        assert check_evidence_prerequisites is False
        screens.append(True)
        return {"evaluation_allowed": False}

    monkeypatch.setattr(candidate_screen, "ensure_candidate_screen", screen)
    result = tools_eval.judge_function(fn.name, review_phase="development")
    assert result.get("candidate_screen_blocked"), result
    assert "repair_execution_blocked" not in result
    assert screens == [True]


# Any change of review request, settings or plan gives a different review key.
@pytest.mark.parametrize("phase,allocation_state", [
    ("development", "pending"), ("confirmation", "pending"),
    ("development", "completed"), ("confirmation", "different_plan"),
])
def test_broken_repair_feedback_preserves_only_matching_pending_allocations(
    tmp_path, monkeypatch, phase, allocation_state,
):
    from autofa.session import review_evidence

    rt, fn, proposed = reevaluated_broken_repair(tmp_path, monkeypatch)
    current = rt.library.get(fn.name).to_dict()
    _, _, settings = tools_eval._review_settings(rt.config)
    plan_id = ("a different plan" if allocation_state == "different_plan"
               else proposed["plan_id"])
    review_key = tools_eval._review_key(current, settings, 4, plan_id)
    review_evidence.allocate_review(
        rt.run.run_dir(), fn.name, [rt.dataset.items[0]], target_n=1, seed=0,
        review_key=review_key, phase=phase,
    )
    if allocation_state == "completed":
        review_evidence.complete_review_allocation(
            rt.run.run_dir(), fn.name, review_key=review_key, phase=phase,
        )
    action = tools_repair.review_next_action(rt, current, evidence_n=4)
    if allocation_state == "pending":
        assert action["tool"] == "judge_function", action
        assert action["arguments"] == {
            "name": fn.name, "review_phase": phase, "evidence_n": 4,
        }
        assert "original evidence and budget" in action["reason"]
    else:
        assert action["tool"] == "defer_function_repair", action


# Execution fixes remain possible after search attempts, with precommit controls.

def test_execution_recovery_is_separate_bounded_and_failed_submission_keeps_baseline(
    tmp_path, monkeypatch,
):
    rt, fn = evaluated_runtime(tmp_path, monkeypatch, source=JSON_SOURCE)
    baseline = deepcopy(fn.to_dict())
    exhaust_search(rt, fn.name)
    for used in range(1, 3):
        feedback = tools_repair.get_function_repair(fn.name)["feedback"]
        finding = next(row for row in feedback["findings"] if row["status"] == "execution_error")
        proposal = tools_repair.plan_function_repair(
            fn.name, "repair_implementation", "Accept plain text in the recorded parser",
            [finding["id"]], finding["item_ids"])
        assert proposal["ok"], proposal
        assert proposal["plan"]["implementation_allowance"] is True
        result = submit_repair(fn, fn.source + "\n# still incorrect", proposal)
        assert not result["ok"] and result["baseline_retained"], result
        assert rt.library.get(fn.name).to_dict() == baseline
        state = workflow.get_repair(rt.run.run_dir(), fn.name)
        assert state["implementation_budget"]["attempts_used"] == used
        assert state["attempt_budget"]["attempts_used"] == 3
        assert state["latest"]["failed_proposal"]["source"].endswith("# still incorrect")
        assert not state["latest"]["implementation_check"]["passed"]
    refused = tools_repair.plan_function_repair(
        fn.name, "repair_implementation", "Try again", [finding["id"]], finding["item_ids"])
    assert not refused["ok"] and "allowance exhausted" in refused["error"]
    assert rt.library.get(fn.name).to_dict() == baseline


def test_nonthrowing_extraction_fix_does_not_require_rejected_causal_review(
    tmp_path, monkeypatch,
):
    rt, fn = _runtime(tmp_path, monkeypatch)
    fn.source = (
        "def guess(item, llm_client=None):\n"
        "    return 'guess' in item['response_text'] or 'guess' in item['prompt']\n")
    for item in rt.dataset.items:
        item["prompt"] = "Please guess the number."
    rt.dataset.items[1]["response_text"] = "A derived value."
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert fn.scores["errors"] == 0
    # A positive existing review adds a regression control, never a prerequisite
    # for the error finding or evidence that the new implementation is valid.
    fn.scores["behavior_verdicts"] = [{"item_id": "case-0", "verdict": "yes"}]
    exhaust_search(rt, fn.name)
    proposal = tools_repair.plan_function_repair(
        fn.name, "repair_implementation",
        "Extract the answer from response_text; prompt text is not the answer.", [], [],
        implementation_checks={"case-1": False})
    assert proposal["ok"], proposal
    assert proposal["plan"]["implementation_allowance"]
    assert proposal["plan"]["implementation_diagnostics"]["item_ids"][:2] == ["case-1", "case-0"]
    fixed = "def guess(item, llm_client=None):\n    return 'guess' in item['response_text']\n"
    result = submit_repair(fn, fixed, proposal)
    assert result["ok"], result
    state = workflow.get_repair(rt.run.run_dir(), fn.name)
    assert state["implementation_budget"]["remaining"] == 1
    assert state["attempt_budget"]["remaining"] == 0
    assert state["latest"]["implementation_check"]["passed"]
    assert state["latest"]["before"]["source"] != fixed
    assert rt.library.get(fn.name).source == fixed
    assert not rt.library.get(fn.name).scores  # Diagnostic expectations confer no scores.
    assert state["latest"]["phase"] == "awaiting_validation"
    assert state["latest"]["result"] is None


def test_extra_parser_repair_succeeds_with_error_only_controls(tmp_path, monkeypatch):
    # The recorded execution error is the only reproduction; no expected outputs are given.
    rt, fn = evaluated_runtime(tmp_path, monkeypatch, source=JSON_SOURCE)
    exhaust_search(rt, fn.name)
    finding = tools_repair.get_function_repair(fn.name)["feedback"]["findings"][0]
    proposal = tools_repair.plan_function_repair(
        fn.name, "repair_implementation", "The source field is plain text, not serialized JSON",
        [finding["id"]], finding["item_ids"])
    assert proposal["ok"], proposal
    result = submit_repair(fn, "def guess(item, llm_client=None):\n"
                           "    return 'guess' in item['response_text']\n", proposal)
    assert result["ok"], result
    state = workflow.get_repair(rt.run.run_dir(), fn.name)
    assert state["latest"]["implementation_check"]["passed"]
    assert state["latest"]["phase"] == "awaiting_validation"


def test_reproduced_fix_cannot_drop_supported_controls_or_change_claim(tmp_path, monkeypatch):
    rt, fn = evaluated_runtime(tmp_path, monkeypatch)
    fn.scores["behavior_verdicts"] = [{"item_id": "case-0", "verdict": "yes"}]
    proposal = tools_repair.plan_function_repair(
        fn.name, "repair_implementation", "Exclude the prompt-only extraction", [], [],
        implementation_checks={"case-1": False})
    assert proposal["ok"], proposal
    baseline = deepcopy(fn.to_dict())
    bad = "def guess(item, llm_client=None):\n    return False\n"
    rejected = submit_repair(fn, bad, proposal, mechanism="A newly invented mechanism")
    assert not rejected["ok"] and "preserve" in rejected["error"]
    assert rt.library.get(fn.name).to_dict() == baseline
    result = submit_repair(fn, bad, proposal)
    assert not result["ok"] and result["baseline_retained"]
    assert rt.library.get(fn.name).to_dict() == baseline
    rows = result["implementation_check"]["result"]["results"]
    assert next(row for row in rows if row["item_id"] == "case-0")["matches_expected"] is False


@pytest.mark.parametrize("action,checks", [
    ("repair_implementation", {"case-0": True}), ("repair_implementation", {}),
    ("revise_claim", {"case-0": False}),
], ids=["nonfailing_reproduction", "missing_reproduction", "hypothesis_rewrite"])
def test_extra_implementation_attempt_needs_a_failing_reproduction_of_an_execution_fix(
    tmp_path, monkeypatch, action, checks,
):
    rt, fn = evaluated_runtime(tmp_path, monkeypatch)
    exhaust_search(rt, fn.name)
    response = tools_repair.plan_function_repair(
        fn.name, action, "An asserted bug", [], [], implementation_checks=checks)
    assert not response["ok"]
    if action == "revise_claim":
        assert "only apply" in response["error"]
    state = workflow.get_repair(rt.run.run_dir(), fn.name)
    assert state["implementation_budget"]["attempts_used"] == 0
    assert len(state["history"]) == 3


# Targeted diagnostics reach the agent without changing scores.

def test_feedback_retrieves_supplemental_case_separately_from_adherence(tmp_path, monkeypatch):
    rt, fn = _runtime(tmp_path, monkeypatch)
    formal = _result(fn, ["yes"], item_ids=["formal"])
    formal.update(judge_behavior_strictness="causal", judge_objective_strategy="balanced",
                  judge_harness="claude", judge_model=rt.config.behavior_judge_model)
    apply_review_result(fn.scores, formal, judge_harness="claude")
    diagnostic = _result(fn, ["no"], item_ids=["supplement"])
    diagnostic.update(
        excluded_from_adherence=True, excluded_from_confirmed_support=True,
        sample={"purpose": "missing_outcome_control", "outcome": "failed"},
        judge_behavior_strictness="causal", judge_objective_strategy="balanced",
        judge_harness="claude", judge_model=rt.config.behavior_judge_model,
    )
    fn.scores["diagnostic_review"] = diagnostic
    before = deepcopy(fn.scores)
    rt.persist_library()

    response = tools_library.get_function_feedback(fn.name, item_id="supplement")
    assert response["ok"]
    extra = response["diagnostic_review"]
    assert [row["item_id"] for row in extra["verdicts"]] == ["supplement"]
    assert extra["verdicts"][0]["claim_assessments"]
    assert extra["excluded_from_adherence"] and extra["excluded_from_confirmed_support"]
    assert extra["current"] is True
    assert extra["pagination"]["total"] == 1
    assert response["feedback"]["behavior_validation"]["verdicts"] == []
    assert response["feedback"]["behavior_purity"] == 1
    assert fn.scores == before

    original = tools_library.get_function_feedback(fn.name, item_id="formal")
    assert original["diagnostic_review"]["verdicts"] == []
    assert original["feedback"]["behavior_validation"]["verdicts"][0]["item_id"] == "formal"

    fn.scores["diagnostic_evidence_stale"] = True
    historical = tools_library.get_function_feedback(fn.name, item_id="supplement")
    assert historical["diagnostic_review"]["current"] is False
    assert historical["diagnostic_review"]["verdicts"][0]["item_id"] == "supplement"
    assert historical["feedback"]["behavior_purity"] == 1
