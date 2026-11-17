# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source prerequisites are cheap, scoped, actionable and recover when sources change."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_eval, tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.judge_evidence import make_snapshot
from autofa.session import candidate_screen
from autofa.session.evidence_prerequisites import evidence_preflight, validate_prerequisites


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    rows = [
        {"_item_id": str(i), "problem_id": str(i), "benchmark": "b", "model_id": "m",
         "is_correct": False, "response_text": "The output is withheld.",
         "input": {"delivered": False, "count": 0}}
        for i in range(5)
    ]
    reads = []

    class IndexedDataset(Dataset):
        def identity_index(self, **kwargs):
            return SimpleNamespace(by_id={row["_item_id"]: i for i, row in enumerate(rows)})

        def iter_locations(self, locations, **kwargs):
            for location in locations:
                reads.append(rows[location]["_item_id"])
                yield deepcopy(rows[location])

        def iter_batches(self, **kwargs):
            pytest.fail("Source preflight must never fall back to a complete trace scan")

    fn = FeatureFunction(
        name="withheld", kind="python",
        source="def withheld(item, llm_client=None): return 'withheld' in item['response_text']",
        description="Withholds an answer despite having the required information.",
        mechanism="The original input supplied the necessary facts, but the answer is withheld.",
    )
    rt = Runtime(
        dataset=IndexedDataset(parquet_paths=[], model_id="m", benchmarks=["b"]),
        library=FunctionLibrary(functions=[fn]), config=Config(behavior_strictness="causal"),
        run=RunConfig(run_id="prerequisites", model_id="m", benchmarks=["b"], parquet_paths=[]),
    )
    set_runtime(rt)
    yield rt, fn, rows, reads
    set_runtime(None)


def _declare(fn, *, pointer="/authoritative_fact"):
    fn.meta["evidence_prerequisites"] = [{
        "fact": "The required information was included in the original input.",
        "item_id": "0", "field": "input", "pointer": pointer,
    }]


def _install_gap(fn, item):
    contract = build_claim_contract(source=fn.source, description=fn.description,
                                    mechanism=fn.mechanism)
    snapshot = make_snapshot(item, item_id=item["_item_id"], source=fn.source,
                             claim_contract=contract)
    row = {
        "item_id": item["_item_id"], "snapshot_id": snapshot["snapshot_id"],
        "claim_contract_id": contract["contract_id"], "verdict": "uncertain",
        "claim_assessments": [{
            "requirement_id": requirement["id"], "status": "unknown",
            "unknown_reason": "evidence_unavailable", "citations": [],
            "rationale": "The retained input does not include the information needed to answer.",
        } for requirement in contract["requirements"]],
    }
    fn.scores = {
        "evaluated": True, "behavior_verdicts": [row], "behavior_n_judged": 1,
        "behavior_n_uncertain": 1, "behavior_n_yes": 0, "behavior_n_no": 0,
        "behavior_n_errors": 0, "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(), "judge_behavior_strictness": "causal",
        "claim_hash": contract["contract_id"], "claim_contract": contract,
    }


def test_new_candidates_without_declarations_do_not_read_sources(runtime):
    rt, fn, _, reads = runtime
    assert evidence_preflight(rt, fn) is None
    assert reads == []


def test_missing_declared_source_blocks_before_corpus_calls_even_without_old_screen(runtime):
    rt, fn, rows, reads = runtime
    _declare(fn)
    rows[1]["input"]["authoritative_fact"] = "present in a different item"
    result = tools_eval.evaluate_functions(names=[fn.name])
    assert result["candidate_screen_blocked"]
    preflight = result["candidate_screens"][fn.name]
    assert preflight["status"] == "evidence_prerequisite_pending"
    assert preflight["source_checks"][0]["availability"] == "not_retained"
    assert preflight["next_action"]["tool"] == "get_field_meta"
    assert preflight["next_action"]["arguments"] == {"item_id": "0", "field": "input"}
    assert reads == ["0"]


@pytest.mark.parametrize("field", ["question", "input_text"])
def test_original_fields_and_derived_aliases_are_not_falsely_missing(runtime, field):
    rt, fn, rows, _ = runtime
    rows[0]["question"] = "What is the retained value?"
    fn.meta["evidence_prerequisites"] = [{
        "fact": "A user input was retained.", "item_id": "0", "field": field,
    }]
    assert evidence_preflight(rt, fn)["evaluation_allowed"]


def test_original_json_pointer_is_checked_without_rewriting_trace_structure(runtime):
    rt, fn, rows, _ = runtime
    rows[0]["response_text"] = json.dumps({"messages": [
        {"role": "user", "content": "Original source fact"},
    ]})
    fn.meta["evidence_prerequisites"] = [{
        "fact": "The source preserves the original message.",
        "item_id": "0", "field": "response_text", "pointer": "/messages/0/content",
    }]
    assert evidence_preflight(rt, fn)["evaluation_allowed"]


def test_pending_declaration_recovers_with_new_source_and_survives_saved_resume(runtime):
    rt, fn, rows, _ = runtime
    _declare(fn)
    assert evidence_preflight(rt, fn)["status"] == "evidence_prerequisite_pending"
    rt.persist_library()
    resumed = FunctionLibrary.load(rt.working_library_path()).get(fn.name)
    assert evidence_preflight(rt, resumed)["status"] == "evidence_prerequisite_pending"
    rows[0]["input"]["authoritative_fact"] = "new original source record"
    assert evidence_preflight(rt, resumed)["evaluation_allowed"]


def test_source_outage_remains_unknown_and_recovers(runtime, monkeypatch):
    rt, fn, _, _ = runtime
    _declare(fn, pointer="/count")  # Zero is retained source content, not absence.
    with monkeypatch.context() as unavailable:
        def fail(**kwargs):
            raise OSError("source is temporarily offline")
        unavailable.setattr(rt.dataset, "identity_index", fail)
        result = evidence_preflight(rt, fn)
    assert result["source_checks"][0]["availability"] == "unknown"
    assert not result["evaluation_allowed"]
    assert evidence_preflight(rt, fn)["evaluation_allowed"]
    assert fn.scores == {}


@pytest.mark.parametrize("availability,tool", [
    ("retained", "screen_function"), ("not_retained", "evaluate_functions"),
    ("unknown", "screen_function"),
])
def test_real_mcp_preflight_source_checks_cannot_become_fresh_confirmation(
    runtime, monkeypatch, availability, tool,
):
    import asyncio

    from autofa.mcp.server import invoke_tool
    from autofa.session.review_evidence import allocate_review, review_context

    rt, fn, rows, _ = runtime
    _declare(fn, pointer="/count" if availability == "retained" else "/missing_fact")
    if availability == "unknown":
        fn.meta["evidence_prerequisites"][0]["item_id"] = "not-present"
    # Cache the verified identity-only inventory just as family initialization
    # does, without reading the trace bodies a second time.
    rt._family_failed_task_inventory = {
        "identity_by_item_id": {
            row["_item_id"]: {key: row[key] for key in ("_item_id", "benchmark", "problem_id")}
            for row in rows
        },
    }
    rt.run.run_dir().mkdir(parents=True, exist_ok=True)
    (rt.run.run_dir() / "family_discovery.json").write_text(json.dumps({
        "version": 3, "run_id": rt.run.run_id,
    }))
    # A successful source-presence check may proceed to scoring. This test
    # exercises real preflight and MCP exposure; no model work is needed.
    monkeypatch.setattr(tools_eval, "_evaluate_streaming", lambda *args, **kwargs: {
        "ok": True, "evaluated_names": [fn.name],
    })
    response = asyncio.run(invoke_tool(
        tool, {"names": [fn.name]} if tool == "evaluate_functions" else {"name": fn.name},
    ))
    result = json.loads(response["content"][0]["text"])
    assert "_exposure_warning" not in result
    context = review_context(rt.run.run_dir(), fn.name)
    if availability == "unknown":
        assert context["reviewed_item_ids"] == []
        return
    assert context["reviewed_item_ids"] == ["0"]
    confirmation = allocate_review(
        rt.run.run_dir(), fn.name, [rows[0]], target_n=1, seed=0,
        review_key="post-preflight", phase="confirmation",
    )
    assert confirmation["sample_info"]["independent_task_n"] == 0
    assert confirmation["sample_info"]["fresh_item_n"] == 0
    assert confirmation["sample_info"]["limited"]


def test_unchanged_known_missing_premise_prevents_rescore_but_preserves_review_access(runtime):
    rt, fn, rows, reads = runtime
    _install_gap(fn, rows[0])
    before = deepcopy(fn.scores)
    result = tools_eval.evaluate_functions(names=[fn.name])
    assert result["candidate_screen_blocked"]
    preflight = result["candidate_screens"][fn.name]
    assert preflight["known_review_gaps"]
    assert preflight["known_review_gaps"][0]["evidence_availability"] == "unavailable"
    assert preflight["next_action"]["kind"] == "defer_claim"
    assert reads == ["0"]
    assert fn.scores == before
    assert candidate_screen.ensure_candidate_screen(
        rt, fn, check_evidence_prerequisites=False,
    )["evaluation_allowed"]


@pytest.mark.parametrize("change", ["source", "policy", "source_evidence"])
def test_new_claim_implementation_policy_or_source_cannot_inherit_old_gap(runtime, change):
    rt, fn, rows, _ = runtime
    _install_gap(fn, rows[0])
    if change == "source":
        fn.source += "\n# Implement a new condition using retained facts."
    elif change == "policy":
        rt.config.behavior_strictness = "behavioral"
    else:
        rows[0]["input"]["authoritative_fact"] = "new retained fact"
    assert evidence_preflight(rt, fn) is None


@pytest.mark.parametrize("change", [
    lambda scores: scores.update(errors=1),
    lambda scores: scores["behavior_verdicts"][0]["claim_assessments"][0].update(
        unknown_reason="ambiguous_evidence"),
    lambda scores: scores.update(behavior_n_yes=1),
    lambda scores: scores.update(judge_policy_hash="old-policy"),
    lambda scores: scores.update(evaluation_run_id="previous-corpus"),
], ids=["execution_error", "ambiguity", "partial_support", "old_policy", "different_run"])
def test_non_prerequisite_work_is_not_blocked(runtime, change):
    rt, fn, rows, _ = runtime
    _install_gap(fn, rows[0])
    change(fn.scores)
    assert evidence_preflight(rt, fn) is None


def test_full_evaluation_and_finalization_can_measure_even_unresolved_claims(runtime, monkeypatch):
    rt, fn, rows, _ = runtime
    _declare(fn)
    _install_gap(fn, rows[0])
    calls = []
    monkeypatch.setattr(tools_eval, "_evaluate_streaming", lambda *a, **kw: (
        calls.append(kw) or {"ok": True, "evaluated_names": [fn.name]}))
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert tools_eval._evaluate_functions(names=[fn.name], _finalizing=True)["ok"]
    assert len(calls) == 2


def test_real_submit_saves_candidate_and_returns_specific_source_action(runtime):
    rt, fn, _, _ = runtime
    _declare(fn)
    result = tools_library.submit_function(
        name=fn.name, source=fn.source, description=fn.description,
        mechanism=fn.mechanism, kind="python",
        evidence_prerequisites=fn.meta["evidence_prerequisites"],
    )
    assert result["ok"] and not result["evaluated"]
    assert result["next_action"]["tool"] == "get_field_meta"
    assert rt.library.get(fn.name).meta["evidence_prerequisites"]


def test_claim_change_clears_previous_declarations_but_code_repair_retains_them(runtime):
    rt, fn, _, _ = runtime
    _declare(fn)
    args = {"name": fn.name, "source": fn.source + "\n# implementation repair",
            "description": fn.description, "mechanism": fn.mechanism,
            "kind": "python", "auto_evaluate": False}
    assert tools_library.submit_function(**args)["ok"]
    assert rt.library.get(fn.name).meta["evidence_prerequisites"]
    assert tools_library.submit_function(**{
        **args, "mechanism": "Withholds a result; no claim that the needed facts were supplied.",
    })["ok"]
    assert "evidence_prerequisites" not in rt.library.get(fn.name).meta


@pytest.mark.parametrize("bad", [
    [{"fact": "fact", "item_id": "0", "field": "is_correct"}],
    [{"fact": "fact", "item_id": "0", "field": "input", "pointer": "not/a/pointer"}],
    [{"fact": "fact", "item_id": str(i), "field": "input"} for i in range(4)],
], ids=["outcome_field", "invalid_pointer", "too_many"])
def test_malformed_declarations_fail_before_library_mutation(runtime, bad):
    rt, fn, _, reads = runtime
    before = deepcopy(rt.library.to_dict())
    result = tools_library.submit_function(
        name=fn.name, source=fn.source, evidence_prerequisites=bad,
    )
    assert not result["ok"] and rt.library.to_dict() == before and not reads
    with pytest.raises(ValueError):
        validate_prerequisites(bad)


def test_both_harnesses_receive_explicit_prerequisite_schema():
    from autofa.mcp.server import _tool_input_schema

    schema = _tool_input_schema(tools_library.submit_function)
    prerequisite = schema["properties"]["evidence_prerequisites"]["anyOf"][0]
    assert prerequisite["items"]["required"] == ["fact", "item_id", "field"]
    assert "pointer" in prerequisite["items"]["properties"]
    assert "is_correct" not in json.dumps(prerequisite)
