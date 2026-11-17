# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-first workflow requires real evidence and persists independent of the agent."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import claude_agent_sdk
import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config
from autofa.data.parquet_store import Dataset as ParquetDataset
from autofa.mcp import server, tools_eval, tools_library
from autofa.metrics.claim_contract import claim_hash
from autofa.objectives import apply_preset
from autofa.session import family_discovery as module
from autofa.session.discovery_target import DiscoveryProgress
from autofa.session.family_discovery import (
    STATE_FILE,
    family_discovery,
    family_progress_summary,
    family_submission_guard,
    initialize_family_discovery,
)
from autofa.session.modes_ledger import get_mode
from autofa.session.research_status import write_research_status
from autofa.session.review_evidence import task_group
from autofa.ui.family_progress import family_progress_lines

from support.families import (
    assess_recorded,
    evaluated_family,
    evidence,
    expand_failures,
    record,
    reviewed_seed,
    survey_rows,
)
from support.families import rt as rt

NO_SHARED_RELATION = {
    "outcome": "no_shared_relation",
    "reason": "Checked source cases do not establish one shared operation.",
    "next_action": "Inspect a different supported relation.",
}


def test_workflow_is_persisted_without_blocking_subtype_submission(rt):
    initialize_family_discovery(rt)
    assert family_submission_guard(rt, "premature_specific_case") is None
    recorded = record()
    assert recorded["ok"]
    assert recorded["phase"] == "assess_family"
    assert recorded["n_supporting_tasks"] == 2
    assert recorded["n_inspected_tasks"] == 3
    assert family_submission_guard(rt, "rejected_required_call") is None
    assert family_submission_guard(rt, "missing_argument",
                                   explains="rejected_required_call") is None
    args = dict(action="assess_family", family_name="rejected_required_call",
                assessment="No fires in the first pass; investigate extraction then subtypes.")
    assert not family_discovery(**args)["ok"]
    evaluated_family(rt)
    result = family_discovery(**args)
    assert result["ok"]
    assert result["phase"] == "specialize"
    assert result["assessment"]["scores"]["recall"] == 0.0
    assert result["assessment"]["review"]["status"] == "unjudged"
    assert family_submission_guard(rt, "missing_argument",
                                   explains="rejected_required_call") is None
    assert initialize_family_discovery(rt)["phase"] == "specialize"


@pytest.mark.parametrize("change", [
    {"evaluation_claim_hash": "old"}, {"evaluation_stale": True},
], ids=["changed_claim", "stale"])
def test_stale_or_foreign_family_evaluation_does_not_unlock_specialization(rt, change):
    record()
    evaluated_family(rt).scores.update(change)
    result = family_discovery(action="assess_family", family_name="rejected_required_call",
                              assessment="Read the results.")
    assert not result["ok"]
    assert family_discovery()["phase"] == "assess_family"


def test_seeded_claim_must_match_the_recorded_family_before_assessment(rt):
    record()
    fn = evaluated_family(rt)
    original = fn.description
    fn.description = "A different, narrower claim"
    fn.scores["evaluation_claim_hash"] = claim_hash(source=fn.source,
                                                   description=fn.description,
                                                   mechanism=fn.mechanism)
    args = dict(action="assess_family", family_name=fn.name,
                assessment="Read a current evaluation of the wrong claim.")
    before = (rt.run.run_dir() / STATE_FILE).read_bytes()
    assert not family_discovery(**args)["ok"]
    assert (rt.run.run_dir() / STATE_FILE).read_bytes() == before
    assert record(description=fn.description,
                  scope_limit="The investigated cross-task hypothesis was revised.")["ok"]
    assert family_discovery(**args)["ok"]
    state = initialize_family_discovery(rt)
    assert state["history"][0]["survey"]["description"] == original


def test_differently_named_detector_is_assessed_without_duplicate_or_new_review(
    rt, monkeypatch,
):
    name = "actual_evaluated_detector"
    assert record()["ok"]
    fn = evaluated_family(rt)
    # Renaming only the library reference emulates an already evaluated detector:
    # names do not participate in the immutable source/description/mechanism claim.
    fn.name = name
    original = deepcopy(fn.to_dict())
    before_history = deepcopy(rt.scores_history)

    def no_paid_work(*args, **kwargs):
        pytest.fail("Family linking must not resubmit, evaluate or judge a detector")

    monkeypatch.setattr(tools_library, "submit_function", no_paid_work)
    monkeypatch.setattr(tools_eval, "evaluate_functions", no_paid_work)
    monkeypatch.setattr(tools_eval, "judge_function", no_paid_work)
    result = family_discovery(
        "assess_family", family_name="rejected_required_call", detector_name=name,
        assessment="Existing first-pass evaluation found no matches; extraction remains open.",
    )
    assert result["ok"], result
    assert result["completed_comparisons"] == 1
    assert result["assessment"]["detector_name"] == name
    assert result["assessment"]["claim_hash"] == original["scores"]["evaluation_claim_hash"]
    assert [row.name for row in rt.library.functions] == [name]
    assert fn.to_dict() == original
    assert rt.scores_history == before_history


def test_missing_same_name_detector_suggests_existing_exact_claim(rt):
    record()
    fn = evaluated_family(rt)
    fn.name = "existing_detector"
    result = family_discovery(
        "assess_family", family_name="rejected_required_call", assessment="Assess saved results.",
    )
    assert not result["ok"]
    assert result["matching_detectors"] == ["existing_detector"]
    assert result["next_action"] == {
        "tool": "get_library_function", "arguments": {"name": "existing_detector"},
    }
    assert family_discovery()["completed_comparisons"] == 0


def test_detector_alias_does_not_bypass_evaluation_freshness(rt):
    record(detector_name="existing_detector")
    fn = evaluated_family(rt)
    fn.name = "existing_detector"
    fn.scores["evaluation_run_id"] = "old_run"
    result = family_discovery(
        "assess_family", family_name="rejected_required_call", assessment="Read current results.",
    )
    assert not result["ok"]
    assert result["next_action"] == {
        "tool": "evaluate_functions", "arguments": {"names": [fn.name]},
    }
    assert family_discovery()["completed_comparisons"] == 0


def test_family_bad_arguments_expose_complete_action_requirements(rt):
    result = family_discovery("record_family", family_name="candidate")
    assert not result["ok"]
    assert result["missing_arguments"] == ["description", "mechanism", "evidence", "boundary_notes"]
    assert result["evidence_schema"]["items"]["required"] == [
        "item_id", "role", "causal_chain", "quotes",
    ]
    response = asyncio.run(server.invoke_tool("family_discovery", {
        "action": "record_family", "name": "guessed_name", "rationale": "guessed field",
    }))
    payload = json.loads(response["content"][0]["text"])
    assert not payload["ok"]
    assert "family_name" in payload["allowed_arguments"]
    assert "detector_name" in payload["allowed_arguments"]
    assert "name" not in payload["allowed_arguments"]
    assert payload["required_arguments"] == [
        "family_name", "description", "mechanism", "evidence", "boundary_notes",
    ]


def test_completed_survey_recurs_next_round_but_pending_work_is_not_discarded(rt):
    record()
    rt.run.round = 2
    assert initialize_family_discovery(rt)["phase"] == "assess_family"
    evaluated_family(rt)
    family_discovery(action="assess_family", family_name="rejected_required_call",
                     assessment="Read first-pass false positives and uncertain evidence.")
    state = initialize_family_discovery(rt)
    assert state["phase"] == "survey"
    assert state["round"] == 2
    assert state["history"][0]["assessment"]["family_name"] == "rejected_required_call"
    assert family_submission_guard(rt, "child") is None


def test_standalone_submission_is_not_forced_into_session_workflow(rt):
    assert family_submission_guard(rt, "new") is None
    assert not (rt.run.run_dir() / STATE_FILE).exists()


def test_advance_round_initializes_discovery_and_waits_for_completed_survey(
    rt, monkeypatch,
):
    rt.run.save = lambda: None
    rt.run.exploration_ssot = False
    rt.run.exploration_seed = 0
    rt.run.min_failure_modes = 0
    rt.run.max_rounds = 3
    monkeypatch.setattr(tools_library, "get_runtime", lambda: rt)
    monkeypatch.setattr(tools_library, "_detector_exploration", lambda: {"message": "Explore"})
    before = rt.run.round
    assert not tools_library.advance_round()["ok"]
    assert rt.run.round == before
    assert (rt.run.run_dir() / STATE_FILE).exists()
    completed = family_discovery(
        action="no_coherent_family", decision=NO_SHARED_RELATION,
        evidence=[evidence(), evidence("b1")],
        boundary_notes="Different causal requirements.",
        assessment="No coherent family in the inspected survey.",
    )
    assert completed["ok"]
    assert completed["exploration_exhausted"] == "available_failed_task_evidence_exhausted"
    result = tools_library.advance_round()
    assert result["ok"]
    assert rt.run.round == before + 1
    assert result["family_discovery_phase"] == "survey"
    assert family_discovery()["round"] == rt.run.round


def test_claude_family_schema_and_help_match_codex(monkeypatch):
    wrapped = []
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server",
                        lambda **kwargs: wrapped.extend(kwargs["tools"]))
    server.build_mcp_server()
    assert server.TOOL_CALLABLES["family_discovery"] is family_discovery
    claude = next(tool for tool in wrapped if tool.name == "family_discovery")
    codex = next(spec for spec in server.codex_tool_specs() if spec["name"] == claude.name)
    assert claude.input_schema == codex["inputSchema"]
    assert claude.description == codex["description"]
    args = {"action": "record_family", "family_name": "family", "detector_name": "existing",
            "description": "Observed claim", "mechanism": "Trace-supported mechanism",
            "evidence": [evidence()], "boundary_notes": "Inspect recovery"}
    for schema in (claude.input_schema, codex["inputSchema"]):
        jsonschema.validate(args, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**args, "evidence": [{"item_id": "a1"}]}, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**args, "action": "invented_action"}, schema)


def test_repeated_attempts_cannot_pretend_to_be_cross_task_support(rt):
    rows = [evidence(), evidence("a2"), evidence("c1", "boundary")]
    # Even an explicit fabricated task_id is ignored in favor of the dataset.
    rows[1]["task_id"] = "different-task"
    assert not record(evidence=rows)["ok"]
    assert not record(evidence=rows,
                      scope_limit="Support is limited to one task; test generalization.")["ok"]
    rows.append(evidence("b1", "unrelated"))
    result = record(evidence=rows,
                    scope_limit="Support is limited to one task; test generalization.")
    assert result["ok"]
    assert result["n_supporting_tasks"] == 1
    assert not result["cross_task_support_observed"]
    assessed = assess_recorded(rt)
    assert assessed["assessment"]["coverage"]["n_review_confirmed_failed_tasks"] == 0


@pytest.mark.parametrize("mutation,error", [
    (lambda rows: rows[0].update(item_id="foreign"), "selected dataset"),
    (lambda rows: rows[0]["quotes"][0].update(quote="invented quote"), "Quote not found"),
    (lambda rows: rows[0].update(quotes=[]), "1–6 quotes"),
    (lambda rows: rows[0].update(causal_chain=""), "causal_chain"),
    (lambda rows: rows[2].update(role="support"), "failed outcome"),
    (lambda rows: rows[1].update(item_id="a1"), "distinct"),
])
def test_invalid_evidence_never_advances_phase(rt, mutation, error):
    rows = [evidence(), evidence("b1"), evidence("c1", "boundary")]
    mutation(rows)
    result = record(evidence=rows)
    assert not result["ok"]
    assert error in result["error"]
    assert family_discovery()["phase"] == "survey"


def test_decoded_json_quotes_match_the_actual_record_not_only_serialization(rt):
    rt.dataset.items[0]["response_text"] = json.dumps({"content": "line one\nline two"})
    rows = [evidence(), evidence("b1"), evidence("c1", "boundary")]
    rows[0]["quotes"][0]["quote"] = "line one\nline two"
    assert record(evidence=rows)["ok"]


def test_no_coherent_family_explicitly_unlocks_rare_causes_without_validation_claim(rt):
    args = dict(action="no_coherent_family", decision=NO_SHARED_RELATION,
                evidence=[evidence(), evidence("b1", "unrelated")],
                boundary_notes="Requirements reveal distinct causes, not one shared causal rule.")
    assert not family_discovery(**args)["ok"]
    result = family_discovery(**args, assessment="Survey found different operations and no "
                              "meaningful common failure beyond the generic bad-result label.")
    assert result["ok"]
    assert (result["phase"], result["family_name"]) == ("specialize", None)
    assert family_submission_guard(rt, "supported_rare_cause") is None


def test_no_family_outcome_cannot_be_based_only_on_passing_tasks(rt):
    result = family_discovery(action="no_coherent_family", decision=NO_SHARED_RELATION,
                              evidence=[evidence("c1", "boundary")],
                              boundary_notes="Recovered error", scope_limit="One example",
                              assessment="No failures inspected")
    assert not result["ok"]
    assert "recorded failed item" in result["error"]


def test_blocked_family_is_preserved_as_open_without_fake_assessment(rt):
    record()
    args = dict(action="defer_family", family_name="rejected_required_call")
    assert not family_discovery(**args)["ok"]
    result = family_discovery(**args, blocker="Semantic endpoint unavailable",
                              next_step="Restore the configured backend then evaluate this family")
    assert result["ok"]
    assert result["phase"] == "specialize"
    assert result["assessment"]["outcome"] == "deferred"
    assert "scores" not in result["assessment"]
    mode = get_mode(rt.run.run_dir(), "rejected_required_call")
    assert mode["status"] == "open"
    assert "Semantic endpoint unavailable" in mode["reason"]


def test_cited_family_examples_are_excluded_from_later_reserved_confirmation(rt):
    assert record()["ok"]
    saved = json.loads((rt.run.run_dir() / "review_evidence.json").read_text())
    assert saved["discovery_item_ids"] == ["a1", "b1", "c1"]
    assert saved["discovery_task_groups"]["a1"] == task_group(rt.dataset.items[0])


def test_status_reads_do_not_fake_discovery_progress(rt):
    initialize_family_discovery(rt)
    path = rt.run.run_dir() / STATE_FILE
    before = path.stat().st_mtime_ns
    progress = DiscoveryProgress(rt)
    family_discovery()
    assert path.stat().st_mtime_ns == before
    assert not progress.stalled()
    assert progress.stalled()
    record()
    assert not progress.stalled()


@pytest.mark.parametrize("preset,tasks,budget", [
    ("precision", 3, 1), ("balanced", 6, 3), ("recall", 8, 5),
])
def test_preset_sets_survey_effort_and_comparison_budget_preserved_on_status_reads(
    rt, preset, tasks, budget,
):
    rt.config = apply_preset(Config(), preset)
    expand_failures(rt, 20)
    first = family_discovery()
    assert first["suggested_distinct_tasks"] == tasks
    assert first["comparison_budget"] == budget
    path = rt.run.run_dir() / STATE_FILE
    before = path.stat().st_mtime_ns
    assert family_discovery()["comparison_budget"] == budget
    assert path.stat().st_mtime_ns == before


def test_cross_run_state_is_never_silently_reused(rt):
    initialize_family_discovery(rt)
    rt.run.run_id = "new-run"
    result = family_discovery()
    assert not result["ok"]
    assert "different run" in result["error"]


def test_balanced_compares_three_families_before_specializing_even_when_recall_zero(rt):
    expand_failures(rt)
    for index, inspected in enumerate((range(6), range(2, 8), range(4, 10))):
        assert record(family_name=f"family_{index}", evidence=survey_rows(*inspected),
                      mechanism=f"Distinct prerequisite {index} left required output absent.")["ok"]
        result = assess_recorded(rt)
        assert result["completed_comparisons"] == index + 1
        assert result["phase"] == ("specialize" if index == 2 else "survey")
        assert result["comparisons"][index]["scores"]["recall"] == 0.0
    assert result["exploration_exhausted"] == "comparison_budget_completed"
    saved = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
    assert saved["version"] == 3
    assert [row["survey"]["family_name"] for row in saved["comparisons"]] == [
        "family_0", "family_1", "family_2",
    ]


def test_no_family_and_deferral_consume_one_real_comparison_not_the_entire_budget(rt):
    expand_failures(rt)
    result = family_discovery(action="no_coherent_family", decision=NO_SHARED_RELATION,
                              evidence=survey_rows(*range(6)),
                              boundary_notes="Contrasts inspected.", assessment="Different causes.")
    assert result["ok"]
    assert result["completed_comparisons"] == 1
    assert result["phase"] == "survey"
    assert record(family_name="another_family", evidence=survey_rows(*range(2, 8)))["ok"]
    result = family_discovery(action="defer_family", family_name="another_family",
                              blocker="Backend unavailable", next_step="Restore backend")
    assert result["ok"]
    assert result["completed_comparisons"] == 2
    assert result["phase"] == "survey"
    assert result["comparisons"][1]["outcome"] == "deferred"
    assert result["comparisons"][1]["scores"] is None


def test_reusing_old_evidence_or_scope_note_cannot_skip_real_comparison_effort(rt):
    expand_failures(rt)
    assert record(evidence=survey_rows(*range(6)))["ok"]
    assess_recorded(rt)
    assert not record(family_name="renamed_family", evidence=survey_rows(*range(6)),
                      scope_limit="Pretend these are all available tasks.")["ok"]
    assert not record(family_name="renamed_family", evidence=survey_rows(6, 7),
                      scope_limit="Only two inspected")["ok"]
    assert family_discovery()["completed_comparisons"] == 1


def test_missing_task_identities_have_bounded_effort_without_claiming_task_diversity(rt):
    expand_failures(rt, 3, task_identity=False)
    assert record(evidence=survey_rows(0, 1),
                  scope_limit="Task identities unavailable; only attempt diversity known.")["ok"]
    first = assess_recorded(rt)
    assert first["phase"] == "survey"
    assert first["n_previously_inspected_failed_tasks"] == 0
    assert first["n_previously_inspected_failed_items"] == 2
    assert record(family_name="last_family", evidence=survey_rows(2),
                  mechanism="A distinct prerequisite was omitted.",
                  scope_limit="One remaining failed item with unknown task identity.")["ok"]
    last = assess_recorded(rt)
    assert last["phase"] == "specialize"
    assert last["completed_comparisons"] == 2
    assert last["exploration_exhausted"] == "available_failed_task_evidence_exhausted"


def test_read_only_progress_summary_never_initializes_or_changes_state(rt):
    assert family_progress_summary(rt.run.run_dir()) is None
    initialize_family_discovery(rt)
    path = rt.run.run_dir() / STATE_FILE
    before = path.stat().st_mtime_ns
    assert family_progress_summary(rt.run.run_dir())["comparison_budget"] == 3
    assert path.stat().st_mtime_ns == before


def test_identical_claim_cannot_count_as_another_compared_family_under_a_new_name(rt):
    expand_failures(rt)
    assert record(evidence=survey_rows(*range(6)))["ok"]
    assess_recorded(rt)
    result = record(family_name="renamed_family", evidence=survey_rows(*range(2, 8)))
    assert not result["ok"]
    assert "already compared under another name" in result["error"]
    assert family_discovery()["completed_comparisons"] == 1


def test_question_identity_replaces_opaque_alias_when_recording_discovery_exposure(rt, tmp_path):
    path = tmp_path / "questions-with-attempt-ids.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"item_id": f"opaque_{i}", "model_id": "m", "benchmark": "bench:variant",
         "question_id": question, "is_correct": i == 3,
         "response_text": "Required action rejected."}
        for i, question in enumerate(("same:question", "same:question", "other", "boundary"))
    ]), path)
    rt.dataset = ParquetDataset(parquet_paths=[path])
    items = rt.dataset.load_items()
    evidence_rows = [{
        "item_id": item["_item_id"], "role": "boundary" if item["is_correct"] else "support",
        "causal_chain": "Rejected prerequisite left a required output absent.",
        "quotes": [{"field": "response_text", "quote": "Required action rejected."}],
    } for item in items]
    result = record(evidence=evidence_rows)
    assert result["ok"], result
    assert result["n_supporting_tasks"] == 2
    exposure = json.loads((rt.run.run_dir() / "review_evidence.json").read_text())
    groups = exposure["discovery_task_groups"]
    assert groups[items[0]["_item_id"]] == groups[items[1]["_item_id"]] == task_group({
        "benchmark": "bench:variant", "question_id": "same:question",
    })
    assert groups[items[2]["_item_id"]] == task_group({
        "benchmark": "bench:variant", "question_id": "other",
    })


def test_resume_preserves_scores_and_pending_work(rt):
    record()
    fn = evaluated_family(rt)
    before_scores = dict(fn.scores)
    path = rt.run.run_dir() / STATE_FILE
    status = family_discovery()
    assert status["phase"] == "assess_family"
    assert family_submission_guard(rt, fn.name) is None
    assert fn.scores == before_scores
    saved = json.loads(path.read_text())
    family_discovery()
    assert json.loads(path.read_text()) == saved


def test_status_advises_bottom_up_for_complementary_current_reviewed_seeds(rt):
    first = reviewed_seed(rt, "narrow_invalid_call", ["a1"], limited=True)
    second = reviewed_seed(rt, "narrow_argument", ["b1"])
    before = deepcopy([first.to_dict(), second.to_dict()])
    status = family_discovery()
    assert status["recommended_direction"] == "bottom_up"
    assert {row["name"] for row in status["synthesis_candidates"]} == {first.name, second.name}
    assert before == [first.to_dict(), second.to_dict()]


def test_unjudged_precision_alone_and_stale_review_cannot_seed_synthesis(rt):
    first = reviewed_seed(rt, "narrow_invalid_call", ["a1"])
    second = reviewed_seed(rt, "narrow_argument", ["b1"])
    first.scores.pop("behavior_verdicts")
    second.scores["judge_policy_hash"] = "old"
    status = family_discovery()
    assert status["recommended_direction"] == "top_down"
    assert status["synthesis_candidates"] == []
    result = record(seed_functions=[first.name, second.name])
    assert not result["ok"]
    assert "individually confirmed" in result["error"]


def test_bottom_up_reuses_evidence_after_comparisons_and_preserves_originals(rt):
    first = reviewed_seed(rt, "narrow_invalid_call", ["a1"], limited=True)
    second = reviewed_seed(rt, "narrow_argument", ["b1"])
    before = deepcopy([first.to_dict(), second.to_dict()])
    assert record(family_name="earlier_family")["ok"]
    assert assess_recorded(rt)["phase"] == "specialize"
    result = record(seed_functions=[first.name, second.name],
                    mechanism="Rejected prerequisites leave required actions unperformed.")
    assert result["ok"], result
    assert result["phase"] == "assess_family"
    assert result["approach"] == "bottom_up"
    assert result["seed_functions"] == [
        {"name": first.name, "claim_hash": first.scores["claim_hash"], "support_item_ids": ["a1"]},
        {"name": second.name, "claim_hash": second.scores["claim_hash"],
         "support_item_ids": ["b1"]},
    ]
    assessed = assess_recorded(rt)
    assert assessed["phase"] == "specialize"
    assert assessed["comparisons"][-1]["approach"] == "bottom_up"
    assert (assessed["comparisons"][-1]["seed_comparison"]
            == assessed["assessment"]["seed_comparison"])
    assert before == [first.to_dict(), second.to_dict()]
    parent = rt.library.get("rejected_required_call")
    assert parent.depends_on == []
    assert parent.explains is None


def test_bottom_up_repeated_tasks_need_scope_and_never_imply_breadth(rt):
    reviewed_seed(rt, "narrow_one", ["a1"])
    reviewed_seed(rt, "narrow_two", ["a2"])
    rows = [evidence(), evidence("a2"), evidence("c1", "boundary")]
    rows[1]["task_id"] = "invented-different-task"
    args = {"evidence": rows, "seed_functions": ["narrow_one", "narrow_two"]}
    assert "scope_limit" in record(**args)["error"]
    recorded = record(**args, scope_limit="Support comes from repeats of one task only.")
    assert recorded["ok"], recorded
    assert recorded["n_supporting_tasks"] == 1
    assert not recorded["cross_task_support_observed"]


@pytest.mark.parametrize("seeds,error", [
    ([], "at least two"), (["narrow_one", "narrow_one"], "distinct"),
    (["narrow_one", "missing"], "current evaluation"),
    (["narrow_one", "rejected_required_call"], "own seed"),
])
def test_bottom_up_rejects_invalid_seed_provenance(rt, seeds, error):
    reviewed_seed(rt, "narrow_one", ["a1"])
    assert error in record(seed_functions=seeds)["error"]
    assert family_discovery()["phase"] == "survey"


def test_bottom_up_requires_real_checked_current_support_from_every_seed(rt):
    reviewed_seed(rt, "narrow_one", ["a1"])
    reviewed_seed(rt, "narrow_two", ["b1"])
    rows = [evidence(), evidence("b1", "unrelated"), evidence("c1", "boundary")]
    result = record(seed_functions=["narrow_one", "narrow_two"], evidence=rows)
    assert not result["ok"]
    assert "Cite checked support evidence from seed 'narrow_two'" in result["error"]
    rows[1]["role"] = "support"
    rows[1]["quotes"][0]["quote"] = "fabricated causal evidence"
    assert "Quote not found" in record(seed_functions=["narrow_one", "narrow_two"],
                                        evidence=rows)["error"]


@pytest.mark.parametrize("change", ["claim", "review"])
def test_changed_seed_does_not_block_independent_parent_assessment(rt, change):
    first = reviewed_seed(rt, "narrow_one", ["a1"])
    second = reviewed_seed(rt, "narrow_two", ["b1"])
    recorded = record(seed_functions=[first.name, second.name])
    assert recorded["ok"]
    historical_seeds = deepcopy(recorded["seed_functions"])
    parent = evaluated_family(rt)
    parent_before = deepcopy(parent.to_dict())
    if change == "claim":
        first.description += " revised"
    else:
        first.scores["judge_policy_hash"] = "old"
    status = family_discovery()
    provenance = {"current": False, "stale_seed_functions": [first.name]}
    assert status["seed_provenance"] == provenance
    result = family_discovery(action="assess_family", family_name=parent.name,
                              assessment="Assess the parent's independent current evaluation.")
    assert result["ok"], result
    assert result["phase"] == "specialize"
    assert result["assessment"]["seed_provenance"] == provenance
    comparison = result["assessment"]["seed_comparison"]
    assert comparison["comparison_scope"] == (
        "unavailable_seed_revision" if change == "claim" else "partial_observed_matches")
    assert result["comparisons"][-1]["seed_functions"] == historical_seeds
    assert rt.library.get(parent.name) is parent
    assert parent.to_dict() == parent_before


def test_repaired_parent_can_rerecord_with_historical_missing_seed(rt):
    first = reviewed_seed(rt, "narrow_one", ["a1"])
    second = reviewed_seed(rt, "narrow_two", ["b1"])
    recorded = record(seed_functions=[first.name, second.name])
    assert recorded["ok"]
    historical_seeds = deepcopy(recorded["seed_functions"])
    parent = evaluated_family(rt)
    rt.library.remove(first.name)
    parent.description += " with independently repaired evidence extraction"
    parent.source += "\n# Current independent parent implementation"
    parent.scores["evaluation_claim_hash"] = claim_hash(
        source=parent.source, description=parent.description, mechanism=parent.mechanism,
    )
    # Explicitly refreshing the baseline still needs current reviewed seed evidence.
    refreshed = record(description=parent.description,
                       seed_functions=[first.name, second.name])
    assert not refreshed["ok"]
    assert "current evaluation" in refreshed["error"]
    revised = record(description=parent.description)
    assert revised["ok"], revised
    assert revised["seed_functions"] == historical_seeds
    assert revised["seed_provenance"] == {
        "current": False, "stale_seed_functions": [first.name],
    }
    result = family_discovery(action="assess_family", family_name=parent.name,
                              assessment="Assess the independently repaired parent.")
    assert result["ok"], result
    assert result["phase"] == "specialize"
    assert (result["assessment"]["seed_comparison"]["comparison_scope"]
            == "unavailable_seed_revision")
    assert result["seed_functions"] == historical_seeds
    saved = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
    assert saved["history"][-1]["survey"]["seed_functions"] == historical_seeds
    assert rt.library.get(parent.name) is parent


def test_seeded_library_still_allows_top_down_without_forcing_synthesis(rt):
    reviewed_seed(rt, "narrow_one", ["a1"])
    reviewed_seed(rt, "narrow_two", ["b1"])
    assert family_discovery()["recommended_direction"] == "bottom_up"
    recorded = record()
    assert recorded["ok"]
    assert recorded["approach"] == "top_down"
    assert recorded["seed_functions"] == []


def test_stalled_top_down_repairs_require_seed_inspection_before_another_comparison(rt):
    expand_failures(rt)
    reviewed_seed(rt, "narrow_one", ["failed_0"], limited=True)
    reviewed_seed(rt, "narrow_two", ["failed_1"], limited=True)
    assert record(evidence=survey_rows(*range(6)))["ok"]
    name = "rejected_required_call"
    family_submission_guard(rt, name, operation="repair", consume_attempt=True)
    assert not family_discovery()["synthesis_required"]
    family_submission_guard(rt, name, operation="repair", consume_attempt=True)
    status = family_discovery()
    assert status["synthesis_required"]
    assert status["synthesis_reason"] == "repeated_top_down_mutations"
    # A pending family can still be revised, evaluated and assessed normally.
    assert record(evidence=survey_rows(*range(6)), description="Revised current family")["ok"]
    assert family_submission_guard(rt, name, operation="repair") is None
    assert assess_recorded(rt)["synthesis_required"]
    assert not record(family_name="another_top_down", evidence=survey_rows(*range(2, 8)))["ok"]
    synthesized = record(family_name="shared_prerequisite", evidence=survey_rows(0, 1),
                         seed_functions=["narrow_one", "narrow_two"])
    assert synthesized["ok"], synthesized
    assert synthesized["approach"] == "bottom_up"
    assert not synthesized["synthesis_required"]
    assert synthesized["synthesis_pending"]
    assert synthesized["completed_synthesis_checks"] == 1
    assert assess_recorded(rt)["assessment"]["seed_comparison"]


def test_incompatible_seeds_can_be_deferred_with_checked_evidence_without_new_parent(rt):
    expand_failures(rt)
    reviewed_seed(rt, "narrow_one", ["failed_0"], limited=True)
    reviewed_seed(rt, "narrow_two", ["failed_1"], limited=True)
    for index, inspected in enumerate((range(6), range(2, 8))):
        assert record(family_name=f"top_down_{index}", evidence=survey_rows(*inspected),
                      mechanism=f"Necessary action {index} was omitted")["ok"]
        result = assess_recorded(rt)
    assert result["synthesis_required"]
    assert result["remaining_comparisons"] == 1
    assert result["synthesis_reason"] == "top_down_comparison_budget"
    before = [fn.to_dict() for fn in rt.library.functions]
    args = dict(action="no_coherent_family", decision=NO_SHARED_RELATION,
                seed_functions=["narrow_one", "narrow_two"],
                evidence=survey_rows(0, 1), boundary_notes="Distinct required operations.",
                assessment="Inspected both supports: a shared input-rejection symptom does "
                           "not establish the same violated requirement or necessary action.")
    forged = survey_rows(0, 1)
    forged[0]["quotes"][0]["quote"] = "invented shared mechanism"
    assert not family_discovery(**{**args, "evidence": forged})["ok"]
    assert family_discovery()["synthesis_required"]
    result = family_discovery(**args)
    assert result["ok"], result
    assert result["completed_comparisons"] == result["comparison_budget"] == 3
    assert result["phase"] == "specialize"
    assert not result["synthesis_required"]
    assert result["comparisons"][-1]["approach"] == "bottom_up"
    assert result["comparisons"][-1]["outcome"] == "no_coherent_family"
    assert [fn.to_dict() for fn in rt.library.functions] == before
    # Resume and a new round do not request the same unsuccessful synthesis again.
    path = rt.run.run_dir() / STATE_FILE
    persisted = path.read_text()
    assert not family_discovery()["synthesis_required"]
    assert path.read_text() == persisted
    rt.run.round += 1
    assert not family_discovery()["synthesis_required"]
    assert record(family_name="new_top_down", evidence=survey_rows(*range(6)),
                  mechanism="A different required action was omitted")["ok"]


@pytest.mark.parametrize("invalid", ["same_task", "stale_claim", "duplicate"])
def test_stall_never_schedules_impossible_or_stale_synthesis(rt, invalid):
    first = reviewed_seed(rt, "narrow_one", ["a1"])
    second = reviewed_seed(rt, "narrow_two", ["a2" if invalid == "same_task" else "b1"])
    if invalid == "stale_claim":
        second.description += " changed without a new review"
    elif invalid == "duplicate":
        second.source, second.description, second.mechanism = (
            first.source, first.description, first.mechanism,
        )
        second.scores["evaluation_claim_hash"] = first.scores["evaluation_claim_hash"]
        second.scores["claim_hash"] = first.scores["claim_hash"]
    write_research_status(rt.run.run_dir(), {
        "task_coverage_progress": {"experiments_since_improve": 2},
    })
    assert not family_discovery()["synthesis_required"]
    assert record()["ok"]


def test_stalled_task_coverage_schedules_once_per_current_task_support_not_rollout(rt):
    reviewed_seed(rt, "narrow_one", ["a1"])
    second = reviewed_seed(rt, "narrow_two", ["b1"])
    write_research_status(rt.run.run_dir(), {
        "task_coverage_progress": {"experiments_since_improve": 1},
    })
    status = family_discovery()
    assert status["synthesis_required"]
    assert status["synthesis_reason"] == "confirmed_task_coverage_stalled"
    assert family_discovery(
        action="no_coherent_family", decision=NO_SHARED_RELATION,
        seed_functions=["narrow_one", "narrow_two"],
        evidence=[evidence(), evidence("b1")], boundary_notes="Different operations.",
        assessment="These seed causes have incompatible required actions; no coherent parent.",
    )["ok"]
    # Another failed rollout is not new support for a synthesis opportunity.
    reviewed_seed(rt, "narrow_one", ["a1", "a2"])
    assert not family_discovery()["synthesis_required"]
    second.description += " stale current claim"
    assert not family_discovery()["synthesis_required"]


def test_new_round_does_not_hide_previous_top_down_work_or_outside_family_repairs(rt):
    assert record()["ok"]
    assert assess_recorded(rt)["phase"] == "specialize"
    rt.run.round += 1
    assert family_discovery()["phase"] == "survey"
    reviewed_seed(rt, "narrow_one", ["a1"], limited=True)
    reviewed_seed(rt, "narrow_two", ["b1"], limited=True)
    for _ in range(2):
        family_submission_guard(rt, "narrow_one", operation="repair", consume_attempt=True)
    assert family_discovery()["synthesis_required"]


def test_concurrent_family_proposals_recheck_phase_before_accepting(rt, monkeypatch):
    initialize_family_discovery(rt)
    started, overlapping, release = threading.Event(), threading.Event(), threading.Event()
    original = module._checked_evidence

    def checked(*args, **kwargs):
        if started.is_set():
            overlapping.set()
        started.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_checked_evidence", checked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(record, family_name="first_family")
        assert started.wait(3)
        second = pool.submit(record, family_name="second_family")
        try:
            assert not overlapping.wait(.1), "The second proposal entered the open transaction"
        finally:
            release.set()
        results = [first.result(timeout=3), second.result(timeout=3)]
    assert results[0]["ok"]
    assert not results[1]["ok"]
    saved = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
    assert saved["survey"]["family_name"] == "first_family"
    assert saved["phase"] == "assess_family"


def test_concurrent_mutation_reservations_preserve_every_consumed_attempt(rt):
    initialize_family_discovery(rt)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _: family_submission_guard(rt, "candidate", consume_attempt=True),
            range(24),
        ))
    assert all(result is None for result in results)
    saved = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
    assert saved["mutation_attempts"]["candidate"] == 24


@pytest.mark.parametrize("outcome", ["record_family", "no_coherent_family"])
def test_scheduled_synthesis_after_budget_is_inspection_not_bonus_comparison(rt, outcome):
    rt.config = apply_preset(Config(), "precision")
    reviewed_seed(rt, "narrow_one", ["a1"])
    reviewed_seed(rt, "narrow_two", ["b1"])
    reviewed_seed(rt, "another_seed", ["a2"])
    assert record(family_name="earlier_family")["ok"]
    status = assess_recorded(rt)
    assert status["completed_comparisons"] == status["comparison_budget"] == 1
    assert status["synthesis_required"]

    result = record(
        action=outcome, family_name="shared_parent",
        seed_functions=["narrow_one", "narrow_two"],
        mechanism="Required actions remain undone after their prerequisites are rejected.",
        assessment="The checked pair requires incompatible operations; other pairs are unexplored.",
        **({"decision": {"outcome": "incompatible_causes",
                         "shared_relation": "A shared rejected prerequisite operation.",
                         "reason": "Checked requirements instead require different operations.",
                         "next_action": "Preserve the seeds and inspect a different pair."}}
           if outcome == "no_coherent_family" else {}),
    )
    assert result["ok"], result
    assert result["completed_comparisons"] == result["comparison_budget"] == 1
    assert result["phase"] == "specialize"
    assert not result["synthesis_required"]
    assert result["completed_synthesis_checks"] == 1
    assert rt.library.get("shared_parent") is None
    assert not record(family_name="another_parent",
                      seed_functions=["narrow_one", "narrow_two"])["ok"]
    saved = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
    inspection = saved["synthesis_checks"][-1]
    assert inspection["inspected_seed_functions"] == ["narrow_one", "narrow_two"]
    assert family_discovery()["completed_synthesis_checks"] == 1

    if outcome == "record_family":
        reviewed_seed(rt, "later_seed", ["b1"])
        waiting = family_discovery()
        assert not waiting["synthesis_required"]
        assert waiting["deferred_synthesis"]["family_name"] == "shared_parent"
    rt.run.round += 1
    next_round = family_discovery()
    assert next_round["completed_comparisons"] == 0
    assert not next_round["synthesis_required"]
    assert next_round["deferred_synthesis"] is None
    if outcome == "record_family":
        assert result["synthesis_pending"]
        assert result["deferred_synthesis"]["family_name"] == "shared_parent"
        assert next_round["phase"] == "assess_family"
        assert next_round["family_name"] == "shared_parent"
        assert next_round["approach"] == "bottom_up"
        assert not record(family_name="ignored_saved_synthesis")["ok"]
        assessed = assess_recorded(rt)
        assert assessed["completed_comparisons"] == 1
        assert assessed["phase"] == "specialize"
        assert not assessed["synthesis_pending"]
    else:
        assert not result["synthesis_pending"]
        assert result["deferred_synthesis"] is None
        assert next_round["phase"] == "survey"


def test_bottom_up_is_pending_until_assessed_or_explicitly_deferred(rt):
    reviewed_seed(rt, "narrow_one", ["a1"])
    reviewed_seed(rt, "narrow_two", ["b1"])
    status = record(seed_functions=["narrow_one", "narrow_two"])
    assert status["synthesis_pending"] and not status["synthesis_required"]
    deferred = family_discovery(
        action="defer_family", family_name="rejected_required_call",
        blocker="The checked abstraction loses a supported required-action boundary.",
        next_step="Keep the two seeds and inspect another abstraction when budget allows.",
    )
    assert deferred["ok"] and not deferred["synthesis_pending"]


def test_scheduled_inspection_needs_distinct_tasks_from_the_selected_seed_support(rt):
    reviewed_seed(rt, "same_task_one", ["a1"])
    reviewed_seed(rt, "same_task_two", ["a2"])
    reviewed_seed(rt, "other_task", ["b1"])
    write_research_status(rt.run.run_dir(), {
        "task_coverage_progress": {"experiments_since_improve": 1},
    })
    assert family_discovery()["synthesis_required"]
    result = family_discovery(
        action="no_coherent_family", decision=NO_SHARED_RELATION,
        seed_functions=["same_task_one", "same_task_two"],
        evidence=[evidence("a1"), evidence("a2"), evidence("b1")],
        boundary_notes="The extra task is not confirmed support for either selected seed.",
        assessment="Two detectors on repeated rollouts are not cross-task synthesis.",
    )
    assert not result["ok"] and "two distinct failed tasks" in result["error"]
    assert family_discovery()["synthesis_required"]


@pytest.mark.parametrize("mode", ["behavioral", "causal"])
def test_synthesis_support_uses_selected_review_policy(rt, mode):
    rt.config = Config(behavior_strictness=mode)
    reviewed_seed(rt, "first_seed", ["a1"])
    reviewed_seed(rt, "second_seed", ["b1"])
    assert family_discovery()["recommended_direction"] == "bottom_up"
    rt.config = Config(behavior_strictness="causal" if mode != "causal" else "behavioral")
    assert family_discovery()["synthesis_candidates"] == []


def test_changed_review_policy_archives_family_assessment_without_relabeling(rt):
    rt.config = Config(behavior_strictness="causal")
    assert record()["ok"]
    original = initialize_family_discovery(rt)
    rt.config = Config(behavior_strictness="contextual")
    state = initialize_family_discovery(rt)
    assert state["behavior_strictness"] == "contextual"
    assert state["survey"] is None
    assert state["assessment"] is None
    archived = state["history"][-1]
    assert archived["reason"] == "behavior_strictness_changed"
    assert archived["behavior_strictness"] == "causal"
    assert archived["survey"] == original["survey"]


# One smoke test for the progress display: sampled and missing coverage stay distinct from exact.
def test_search_progress_distinguishes_sampled_observations_from_exact_added_coverage():
    lines = family_progress_lines({
        "phase": "survey", "completed_comparisons": 3, "comparison_budget": 3,
        "comparisons": [
            {"family_name": "first", "coverage": {
                "failure_recall": .03, "recall_scope": "full_selected_corpus",
                "n_matched_failed_tasks": 2, "matched_task_inventory_complete": True,
                "n_additional_failed_attempts": 8,
            }},
            {"family_name": "second", "coverage": {
                "failure_recall": .15, "recall_scope": "sample_estimate",
                "n_matched_failed_tasks": 3, "match_inventory_complete": False,
                "n_additional_failed_attempts": None,
                "n_observed_additional_failed_attempts": 4,
            }},
            {"family_name": "blocked", "outcome": "deferred"},
        ],
    })
    assert "3/3 comparisons" in lines[0]
    assert "recall=3.0%" in lines[1] and "added vs compared families=8" in lines[1]
    assert "recall=15.0% (sample)" in lines[2] and "failed tasks=≥3 observed" in lines[2]
    assert "added vs compared families=4 observed; total unknown" in lines[2]
    assert "deferred · coverage unavailable" in lines[3]
