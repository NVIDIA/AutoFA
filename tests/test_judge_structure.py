# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural counts never replace semantic reads; only reviews of changed evidence become stale."""

import hashlib
import json
from copy import deepcopy

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.data.source_completeness import EVENT_CONTENT_LIMITATION, TRUNCATED_EVENT_LIMITATION
from autofa.mcp import tools_eval
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics import claude_judge, judge_structure
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot
from autofa.metrics.judge_structure import trace_structure
from autofa.metrics.library_coverage import library_failure_coverage
from autofa.metrics.review_content import RetainedReviewContent
from autofa.metrics.verdicts import (
    behavior_validation,
    current_behavior_scores,
    resolved_behavior_purity,
)
from autofa.session.family_discovery import initialize_family_discovery

from support.judge import _trace


def install_reviews(fn, items):
    contract = build_claim_contract(source=fn.source, description=fn.description,
                                    mechanism=fn.mechanism)
    verdicts = []
    for item in items:
        item_id = item["_item_id"]
        snapshot = make_snapshot(item, item_id=item_id, source=fn.source,
                                 failure_localisation=fn.failure_localisations.get(item_id),
                                 claim_contract=contract)
        initial = {"item_id": item_id, "snapshot_id": snapshot["snapshot_id"],
                   "claim_contract_id": contract["contract_id"], "verdict": "yes",
                   "completion_status": "complete"}
        verdicts.append({**initial, "initial_review": dict(initial),
                         "verification_status": "confirmed"})
    fn.scores.update(
        behavior_verdicts=verdicts, behavior_purity=1.,
        behavior_n_judged=len(verdicts), behavior_n_completed=len(verdicts),
        behavior_n_yes=len(verdicts), behavior_n_no=0, behavior_n_uncertain=0,
        judge_contract_version=JUDGE_CONTRACT_VERSION, judge_policy_hash=judge_policy_hash(),
        judge_objective_strategy="balanced", judge_behavior_strictness="causal",
        claim_hash=contract["contract_id"], claim_contract=contract,
    )


@pytest.fixture
def retained_runtime(tmp_path, monkeypatch):
    rows = [{"_item_id": str(index), "problem_id": str(index), "benchmark": "b",
             "model_id": "m", "is_correct": False, "response_text": f"Failed attempt {index}"}
            for index in range(20)]

    class CurrentDataset(Dataset):
        def evaluation_fingerprint(self):
            return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

        def estimate_matching_rows(self):
            return len(rows)

        def iter_batches(self, *, max_items=None, **kwargs):
            yield deepcopy(rows[:max_items])

    fn = FeatureFunction(name="candidate", source="def candidate(item): return True",
                         description="Leaves the required operation undone",
                         mechanism="An invalid call prevents the required action")
    rt = Runtime(dataset=CurrentDataset(parquet_paths=[]),
                 library=FunctionLibrary(functions=[fn]),
                 config=Config(behavior_strictness="causal", llm_eval_sample=4,
                               llm_eval_min_per_stratum=1),
                 run=RunConfig(run_id="retained-content", model_id="m", benchmarks=["b"],
                               parquet_paths=[], eval_limit=20))
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    monkeypatch.setattr(tools_eval, "_llm_backend_error", lambda: None)
    set_runtime(rt)
    yield rt, fn, rows
    set_runtime(None)


def _inventory(trace):
    return trace_structure({"response_text": json.dumps(trace)}, {"response_text"})


@pytest.mark.parametrize("content_loss", [False, True])
def test_complete_structural_count_does_not_interpret_words_inside_reasoning(content_loss):
    trace = _trace()
    if content_loss:
        # An import declares retained records with lost content; the inventory stays countable.
        trace["execution"]["events"][1].pop("finish_reason")
        trace["evidence_scope"].pop("notices", None)
        trace["evidence_scope"]["limitations"] = {TRUNCATED_EVENT_LIMITATION: 1}
        trace["execution"]["availability"]["events"] = {
            "status": "partial", "reason": EVENT_CONTENT_LIMITATION}
    result = _inventory(trace)
    assert result["available"] is True
    assert result["event_inventory_complete"] is True
    assert result["n_assistant_message_events"] == 0
    assert result["n_reasoning_events"] == 1
    assert result["completion_tokens"] == 100000
    assert result["events"][1]["length_truncated"] is True
    assert len(json.dumps(result)) < 2500
    assert "Return a final answer." not in json.dumps(result)


def _trace_with_model_call(*, answer=False):
    """The export's rollout counts omit its separately retained model-call lane."""
    trace = _trace(answer=answer)
    text = "dialect=responses · status_code=200 · error_category=cancelled"
    trace["execution"]["events"].append({
        "event_idx": 0, "lane": "model_call", "lane_kind": "model_call",
        "kind": "other", "kind_raw": "model_call", "role": None, "n_parts": 1,
        "n_tool_calls": None, "finish_reason": "cancelled",
        "parts": [{"event_idx": 0, "lane": "model_call", "part_idx": 0,
                   "part_type": "text", "channel": "content", "origin": "request",
                   "text": text, "chars": len(text)}],
    })
    return trace


def _local_trace_with_model_call():
    """The local EvalRDL import declares its table identity, not the API source label."""
    trace = _trace_with_model_call()
    trace["evidence_scope"].update(identity="evalrdl_tables_v1",
                                   source="local EvalRDL event parts and tool arguments")
    return trace


@pytest.mark.parametrize("build", [_trace_with_model_call, _local_trace_with_model_call])
def test_conversation_rollups_do_not_count_separately_verified_model_call_telemetry(build):
    trace = build()
    # A prior answer copied into a request is not a new action. Conversely,
    # response payloads are rejected below rather than mistaken for telemetry.
    text = '{"messages": [{"role": "assistant", "content": "B"}]}'
    trace["execution"]["events"][-1]["parts"][0].update(text=text, chars=len(text))
    result = _inventory(trace)
    assert result["available"] is True
    assert result["n_events"] == 3
    assert result["conversation_lane"] == ""
    assert result["n_conversation_events"] == 2
    assert result["n_assistant_message_events"] == 0
    assert result["n_reasoning_events"] == 1
    assert result["events"][-1]["classification"] == "request_context"
    assert result["events"][-1]["finish_reason"] == "cancelled"
    assert result["events"][-1]["parts"][0]["sha256"]
    assert "termination" in result["scope"]  # The token cap is not a completeness gap.


def test_answer_in_another_retained_lane_prevents_absence_certificate():
    trace = _trace_with_model_call(answer=True)
    answer = trace["execution"]["events"][2]
    answer["lane"] = "assistant/other"
    answer["parts"][0]["lane"] = answer["lane"]
    trace["execution"]["rollout"].update(n_events=2, n_assistant_turns=0)
    result = _inventory(trace)
    assert result["available"] is True
    assert result["n_assistant_message_events"] == 1
    assert result["n_conversation_assistant_message_events"] == 0


def _event(trace, index):
    return trace["execution"]["events"][index]


@pytest.mark.parametrize("build, mutation", [
    pytest.param(_trace, lambda t: t.update(incomplete=True), id="envelope_incomplete"),
    pytest.param(_trace, lambda t: t["execution"].update(incomplete=True),
                 id="execution_incomplete"),
    pytest.param(_trace, lambda t: t["execution"]["rollout"].update(n_assistant_turns=1),
                 id="rollout_count_disagrees"),
    pytest.param(_trace, lambda t: _event(t, 1).update(event_idx=0), id="duplicate_event"),
    pytest.param(_trace, lambda t: _event(t, 1).update(role="unknown"), id="unknown_actor"),
    pytest.param(_trace, lambda t: _event(t, 1).update(metadata={"prefix_complete": False}),
                 id="prefix_incomplete"),
    pytest.param(_trace, lambda t: _event(t, 1).update(tool_calls=[{"name": "search"}]),
                 id="inline_tool_calls"),
    pytest.param(_trace, lambda t: _event(t, 1)["parts"][0].update(channel="content"),
                 id="content_in_reasoning_event"),
    pytest.param(_trace, lambda t: _event(t, 1)["parts"][0].update(chars=0),
                 id="part_text_not_retained"),
    # Importer-declared gaps and part identities that cannot account for every part.
    pytest.param(_trace, lambda t: _event(t, 1).update(incomplete=True),
                 id="event_declared_incomplete"),
    pytest.param(_trace, lambda t: _event(t, 1)["parts"][0].update(incomplete=True),
                 id="part_declared_incomplete"),
    pytest.param(_trace, lambda t: _event(t, 1).update(metadata={"n_parts": 2}),
                 id="part_count_disagrees"),
    pytest.param(_trace, lambda t: _event(t, 1)["parts"][0].update(part_idx=1),
                 id="part_index_gap"),
    pytest.param(_trace, lambda t: t["execution"]["tool_calls"].append({"event_idx": 1}),
                 id="uninventoried_tool_call"),
    pytest.param(_trace, lambda t: t["execution"]["availability"]["events"].update(
        status="unavailable"), id="events_unavailable"),
    pytest.param(_trace, lambda t: t["evidence_scope"].update(
        limitations=["some execution events excluded"]), id="events_excluded"),
    pytest.param(_trace_with_model_call, lambda t: t["evidence_scope"].pop("source"),
                 id="unverified_lane_source"),
    pytest.param(_local_trace_with_model_call, lambda t: t["evidence_scope"].pop("identity"),
                 id="unverified_local_lane_identity"),
    pytest.param(_trace_with_model_call, lambda t: _event(t, -1).update(is_copied_context=True),
                 id="copied_context"),
    pytest.param(_trace_with_model_call, lambda t: _event(t, -1).update(length_truncated=True),
                 id="truncated_request_context"),
    pytest.param(_trace_with_model_call, lambda t: _event(t, -1)["parts"][0].update(lane=""),
                 id="part_lane_disagrees"),
    pytest.param(_trace_with_model_call,
                 lambda t: _event(t, -1)["parts"][0].update(channel="reasoning"),
                 id="reasoning_in_telemetry"),
])
def test_unverified_or_unclassified_structure_never_certifies_absence(build, mutation):
    trace = build()
    mutation(trace)
    result = _inventory(trace)
    assert result["available"] is False
    assert "event_inventory_complete" not in result
    assert "n_assistant_message_events" not in result


def test_raw_counts_and_flat_response_are_not_structural_proof():
    assert trace_structure({"response_text": "No answer"}, {"response_text"}) is None
    assert _inventory({"metadata": {"n_assistant_turns": 0}}) is None
    assert trace_structure({"response_text": json.dumps(_trace())}, set()) is None


def test_snapshot_exposes_citable_inventory_without_marking_payload_or_artifacts_read(tmp_path):
    source = "def detect(item): return True"
    snapshot = make_snapshot(
        {"response_text": json.dumps(_trace()), "trace_structure": "UNTRUSTED"},
        item_id="example", source=source,
    )
    assert snapshot["manifest"]["trace_structure"]["origin"] == "derived"
    review = EvidenceReview(snapshot, tmp_path / "evidence")
    review.read("detector_source")
    inventory = review.read("trace_structure")
    assert json.loads(inventory["text"])["n_assistant_message_events"] == 0
    assert not review.covers("response_text", 0, 1)
    assert not review.artifact_reads
    verdict = {
        "item_id": "example", "snapshot_id": snapshot["snapshot_id"], "scope": "local",
        "citations": [{"field": "trace_structure", "start": 0, "end": len(inventory["text"]),
                      "quote": inventory["text"]}],
    }
    assert review.validate(verdict) is None
    verdict["scope"] = "whole_trace"
    assert "unread content" in review.validate(verdict)


def test_additive_structure_and_old_policy_recovery_preserve_original_snapshot(monkeypatch):
    kwargs = {"item_id": "example", "source": "def detect(item): return True"}
    plain = {"response_text": json.dumps(_trace())}
    mixed = {"response_text": json.dumps(_trace_with_model_call())}
    current = make_snapshot(plain, **kwargs)
    recovered = make_snapshot(mixed, **kwargs, _include_structure=False)
    with monkeypatch.context() as previous_policy:
        previous_policy.setattr(judge_structure, "trace_structure", lambda *_: None)
        previous = make_snapshot(plain, **kwargs)
        original = make_snapshot(mixed, **kwargs)
    assert "trace_structure" not in previous["fields"]
    assert current["snapshot_id"] == previous["snapshot_id"]
    assert current["projection_hash"] == previous["projection_hash"]
    for key in previous["fields"]:
        assert current["fields"][key] == previous["fields"][key]
    # Old-policy recovery reconstructs the original snapshot without new inventory.
    assert recovered == original
    assert "trace_structure" not in recovered["fields"]
    assert "structure_sha256" not in recovered


def test_frozen_global_clause_cannot_be_reinterpreted_using_structural_inventory(tmp_path):
    text = "There is no recovery anywhere in the run."
    source = "def detect(item): return True"
    contract = build_claim_contract(
        description=text, mechanism=text, source=source, defining_clauses=[
            {"id": "clause_1", "source_field": "description", "quote": text,
             "predicate": text, "required_review_scope": "whole_trace"},
        ],
    )
    snapshot = make_snapshot({"response_text": json.dumps(_trace())}, item_id="example",
                             source=source, claim_contract=contract)
    review = EvidenceReview(snapshot, tmp_path / "evidence")
    review.read("detector_source")
    inventory = review.read("trace_structure")
    verdict = {
        "item_id": "example", "snapshot_id": snapshot["snapshot_id"], "scope": "local",
        "claim_assessments": [{"requirement_id": "clause_1", "status": "supported"}],
        "citations": [{"field": "trace_structure", "start": 0, "end": len(inventory["text"]),
                      "quote": inventory["text"]}],
    }
    assert "supported exhaustive clauses require scope=whole_trace" in review.validate(verdict)


def _old_structure(*_):
    return {"version": 1, "available": False,
            "reason": "The retained source cannot certify complete event structure; inspect "
                      "the original records and preserve uncertainty about missing events."}


def _historical_structure(fields, stored):
    """Before the fix, only traces with a retained model-call lane lost their inventory."""
    return (_old_structure() if '"model_call"' in fields["response_text"]
            else trace_structure(fields, stored))


def _mixed_and_plain_items():
    return [{"_item_id": "mixed", "response_text": json.dumps(_trace_with_model_call())},
            {"_item_id": "plain", "response_text": json.dumps(_trace())}]


def test_fixed_mixed_lane_view_changes_snapshot_and_cannot_restore_an_old_stage(
    tmp_path, monkeypatch,
):
    item = {"response_text": json.dumps(_trace_with_model_call())}
    kwargs = {"item_id": "case", "source": "def detect(item): return True"}
    policy = claude_judge.judge_policy_hash()
    with monkeypatch.context() as previous:
        previous.setattr(judge_structure, "trace_structure", _old_structure)
        old = make_snapshot(item, **kwargs)
    directory = tmp_path / "old-review"
    EvidenceReview(old, directory)
    current = make_snapshot(item, **kwargs)
    assert json.loads(current["fields"]["trace_structure"])["available"]
    # Only the structural view changes, never the source text or global judge policy.
    assert current["snapshot_id"] != old["snapshot_id"]
    assert current["fields"]["response_text"] == old["fields"]["response_text"]
    assert current["projection_hash"] == old["projection_hash"]
    assert claude_judge.judge_policy_hash() == policy
    assert "structure_sha256" in current and "structure_sha256" not in old
    with pytest.raises(ValueError, match="saved evidence does not match"):
        EvidenceReview.restore(current, directory)
    assert json.loads((directory / "evidence.json").read_text()) == old


def test_retained_aggregate_refreshes_only_affected_item_and_preserves_history(monkeypatch):
    fn = FeatureFunction(name="missing_answer", source="def detect(item): return True",
                         description="No answer message is emitted.",
                         mechanism="Reasoning exhausts the output budget without an answer.")
    items = _mixed_and_plain_items()
    with monkeypatch.context() as previous:
        previous.setattr(judge_structure, "trace_structure", _historical_structure)
        install_reviews(fn, items)
    fn.scores["behavior_verdicts"][0].update(verdict="uncertain", verification_status="uncertain")
    before = deepcopy(fn.scores["behavior_verdicts"])
    retained = RetainedReviewContent([fn])
    for item in items:
        retained.observe(item)
    result = retained.apply(fn.name, fn.scores)

    assert not retained.missing_item_ids()
    assert result["behavior_evidence_stale"]
    assert result["behavior_verdicts"][0]["historical_review"] == before[0]
    assert result["behavior_verdicts"][0]["content_validation"]["status"] == "changed"
    assert result["behavior_verdicts"][1] == before[1]
    assert fn.scores["behavior_verdicts"] == before


def _review_from_structure(calls):
    def review(**kwargs):
        evidence = kwargs["review"]
        calls.append(evidence.snapshot["item_id"])
        evidence.read("detector_source")
        text = evidence.read("trace_structure")["text"]
        supported = json.loads(text)["available"]
        citation = evidence.cite("trace_structure", text)["citations"][0]
        contract = evidence.snapshot["claim_contract"]
        defining_ids = [clause["id"] for clause in contract["defining_clauses"]]
        return json.dumps({"verdicts": [{
            "item_id": evidence.snapshot["item_id"],
            "snapshot_id": evidence.snapshot["snapshot_id"],
            "claim_contract_id": contract["contract_id"], "scope": "local",
            "claim_assessments": [{
                "requirement_id": requirement["id"],
                "status": "supported" if supported else "unknown",
                "rationale": "Complete structure." if supported else "Inventory unavailable.",
                **({} if supported else {"unknown_reason": "evidence_unavailable"}),
                "citations": [{"citation_id": citation["citation_id"]}], "counterevidence": [],
                **({"defining_requirement_ids": defining_ids}
                   if not supported and requirement["kind"] in {"claim", "review_check"} else {}),
            } for requirement in contract["requirements"]],
        }], "behavior_specificity": 0.8})

    return review


def test_judge_refreshes_affected_stage_once_and_reuses_unaffected_reviews(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    items = _mixed_and_plain_items()
    calls = []
    monkeypatch.setattr(claude_judge, "_run_review", _review_from_structure(calls))

    def judge():
        return claude_judge.judge_behavior_with_claude(
            name="missing_answer", source="def detect(item): return True",
            description="No answer message is emitted.",
            mechanism="Reasoning exhausts the output budget without an answer.",
            items=items, assigned_labels=[True, True], cwd=tmp_path, model="test", workers=1,
        )

    with monkeypatch.context() as previous:
        previous.setattr(judge_structure, "trace_structure", _historical_structure)
        old = judge()
    assert old["n_yes"] == 1 and old["n_uncertain"] == 1
    assert calls == ["mixed", "plain", "plain"]
    calls.clear()
    current = judge()
    assert current["n_yes"] == 2 and current["n_uncertain"] == 0
    assert calls == ["mixed", "mixed"]
    calls.clear()
    assert judge()["n_yes"] == 2
    assert calls == []


def test_completed_allocation_refreshes_same_cohort_after_structure_fix(
    retained_runtime, monkeypatch, stub_claim_compiler,
):
    rt, fn, items = retained_runtime
    items[:] = items[:3]
    for index, item in enumerate(items):
        item["response_text"] = json.dumps(_trace_with_model_call() if index == 0 else _trace())
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    calls = []
    monkeypatch.setattr(claude_judge, "_run_review", _review_from_structure(calls))
    # Exercise the completed-allocation fast path, not just per-stage checkpoints.
    initialize_family_discovery(rt)
    with monkeypatch.context() as previous:
        previous.setattr(judge_structure, "trace_structure", _historical_structure)
        old = tools_eval.judge_function(fn.name)
    assert old["ok"], old
    assert fn.scores["behavior_n_uncertain"] == 1
    ledger = rt.run.run_dir() / "review_evidence.json"
    allocations = json.loads(ledger.read_text())["detectors"][fn.name]["allocations"]
    before = deepcopy(fn.scores["behavior_verdicts"])
    calls.clear()

    result = tools_eval.judge_function(fn.name)

    assert result["ok"] and not result.get("skipped"), result
    assert calls == ["0", "0"]  # Only the previously unavailable item needs review.
    assert fn.scores["behavior_n_yes"] == len(before)
    assert json.loads(ledger.read_text())["detectors"][fn.name]["allocations"] == allocations
    assert {row["item_id"] for row in fn.scores["behavior_verdicts"]} == {
        row["item_id"] for row in before
    }
    calls.clear()
    assert tools_eval.judge_function(fn.name)["skipped"]
    assert calls == []


def confirming_result(fn, items, **overrides):
    """A completed judge result confirming ``items`` under ``fn``'s current claim."""
    reviewed = deepcopy(fn)
    install_reviews(reviewed, items)
    count = len(items)
    return {
        "backend": "claude_agent_sdk", "behavior_purity": 1., "behavior_specificity": 1.,
        "n_judged": count, "n_completed": count, "n_yes": count, "n_no": 0,
        "n_uncertain": 0, "n_errors": 0, "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(), "judge_behavior_strictness": "causal",
        "claim_hash": reviewed.scores["claim_hash"],
        "claim_contract": reviewed.scores["claim_contract"],
        "verdicts": reviewed.scores["behavior_verdicts"], **overrides,
    }


def test_real_evaluation_excludes_changed_review_but_retains_unchanged_confirmation(
    retained_runtime,
):
    mode = "capped"
    rt, fn, items = retained_runtime
    fn.kind = "llm"
    assert tools_eval.evaluate_functions(mode=mode)["ok"]
    reviewed_ids = sorted(fn.failure_localisations)[:2]
    reviewed_items = [row for row in items if row["_item_id"] in reviewed_ids]
    install_reviews(fn, reviewed_items)
    original_verdicts = deepcopy(fn.scores["behavior_verdicts"])
    assert tools_eval.evaluate_functions(mode=mode)["ok"]
    assert fn.scores["behavior_evidence_stale"] is False
    assert resolved_behavior_purity(fn.scores) == 1.

    changed = reviewed_ids[0]
    next(row for row in items if row["_item_id"] == changed)["response_text"] += " Recovered later."
    assert tools_eval.evaluate_functions(mode=mode)["ok"]
    coverage = library_failure_coverage(rt.library, {row["_item_id"] for row in items},
                                        run_id=rt.run.run_id, include_item_ids=True)
    assert coverage["review_confirmed_failed_item_ids"] == reviewed_ids[1:]
    assert changed in coverage["known_matched_failed_item_ids"]
    saved = next(row for row in fn.scores["behavior_verdicts"] if row["item_id"] == changed)
    assert saved["historical_review"] == next(
        row for row in original_verdicts if row["item_id"] == changed)
    assert saved["content_validation"]["status"] == "changed"
    current = current_behavior_scores(fn.scores, source=fn.source,
                                      description=fn.description, mechanism=fn.mechanism)
    assert current["behavior_evidence_stale"] is True
    assert resolved_behavior_purity(current) is None
    assert behavior_validation(current)["status"] == "stale"
    assert fn.scores["sampled"] is True


def test_review_outside_capped_scan_is_validated_without_extra_detector_calls(retained_runtime):
    _, fn, items = retained_runtime
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    install_reviews(fn, [items[-1]])
    items[-1]["response_text"] = "Changed rollout with the same stable identity"
    assert tools_eval.evaluate_functions(limit=1)["ok"]
    assert fn.scores["n_scored"] == 1
    assert fn.scores["behavior_evidence_stale"] is True
    assert fn.scores["behavior_verdicts"][0]["content_validation"]["status"] == "changed"


def test_repeated_capped_scans_keep_original_review_localisation(retained_runtime):
    _, fn, items = retained_runtime
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    reviewed_id = items[-1]["_item_id"]
    original_localisation = deepcopy(fn.failure_localisations[reviewed_id])
    install_reviews(fn, [items[-1]])
    original = deepcopy(fn.scores["behavior_verdicts"])
    # Historical scores without source identity need a fresh scan; current full
    # scores would instead remain authoritative on an unchanged capped request.
    fn.scores.pop("evaluation_context")
    for _ in range(2):
        assert tools_eval.evaluate_functions(limit=1)["ok"]
        assert reviewed_id not in fn.failure_localisations
        assert fn.scores["behavior_review_localisations"][reviewed_id] == original_localisation
        assert fn.scores["behavior_verdicts"] == original
        assert fn.scores["behavior_evidence_stale"] is False
        assert resolved_behavior_purity(fn.scores) == 1.


def test_unavailable_review_is_historical_and_original_evidence_can_revalidate(retained_runtime):
    _, fn, items = retained_runtime
    install_reviews(fn, [items[0]])
    original = deepcopy(fn.scores["behavior_verdicts"])
    validator = RetainedReviewContent([fn])
    fn.scores = validator.apply(fn.name, fn.scores)
    assert fn.scores["behavior_evidence_stale"] is True
    assert fn.scores["behavior_verdicts"][0]["content_validation"]["status"] == "unavailable"
    validator = RetainedReviewContent([fn])
    validator.observe(items[0])
    scores = validator.apply(fn.name, fn.scores)
    assert scores["behavior_evidence_stale"] is False
    assert scores["behavior_verdicts"] == original


@pytest.mark.parametrize("retain_historical", [False, True])
def test_review_refresh_requires_current_results_to_clear_staleness(
    retained_runtime, monkeypatch, retain_historical,
):
    _, fn, items = retained_runtime
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    install_reviews(fn, [items[0]])
    historical = deepcopy(fn.scores["behavior_verdicts"][0])
    items[0]["response_text"] = "Changed evidence"
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    assert fn.scores["behavior_evidence_stale"]

    def judge(**kwargs):
        result = confirming_result(fn, kwargs["items"])
        if retain_historical:
            result["verdicts"][0] = deepcopy(fn.scores["behavior_verdicts"][0])
        return result

    monkeypatch.setattr("autofa.metrics.claude_judge.judge_behavior_with_claude", judge)
    assert tools_eval.judge_function(fn.name)["ok"]
    assert bool(fn.scores.get("behavior_evidence_stale")) is retain_historical
    assert historical in fn.scores["behavior_verdict_history"]
    assert resolved_behavior_purity(fn.scores) == (None if retain_historical else 1.)
    assert set(fn.scores["behavior_review_localisations"]) == {
        row["item_id"] for row in fn.scores["behavior_verdicts"]}


def test_first_processing_retry_freezes_old_and_new_review_locators(retained_runtime, monkeypatch):
    from autofa.metrics.judge_backend import backend_name
    from autofa.session import repair_workflow

    rt, fn, items = retained_runtime
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    retained_id, retry_id = items[-1]["_item_id"], items[0]["_item_id"]
    install_reviews(fn, [items[-1]])
    original_localisation = deepcopy(fn.failure_localisations[retained_id])
    judge_config, _, settings = tools_eval._review_settings(rt.config)
    fn.scores.update(settings, judge_backend=backend_name(judge_config["harness"]),
                     judge_harness=judge_config["harness"], behavior_n_judged=2,
                     behavior_n_errors=1)
    fn.scores["behavior_verdicts"].append({
        "item_id": retry_id, "verdict": "error", "processing_status": "error",
    })
    assert "behavior_review_localisations" not in fn.scores
    monkeypatch.setattr(repair_workflow, "validation_context", lambda *args, **kwargs: {
        "plan_id": "saved-retry", "retry_item_ids": [retry_id],
        "judge_objective_strategy": "balanced",
        "judge_behavior_strictness": settings["judge_behavior_strictness"],
    })
    monkeypatch.setattr(repair_workflow, "reserve_validation", lambda *args, **kwargs: None)
    monkeypatch.setattr(repair_workflow, "record_validation", lambda *args, **kwargs: {
        "plan_id": "saved-retry",
    })

    def judge(**kwargs):
        assert [item["_item_id"] for item in kwargs["items"]] == [retry_id]
        return confirming_result(fn, kwargs["items"], **settings, behavior_specificity=None,
                                 backend=backend_name(judge_config["harness"]))

    monkeypatch.setattr("autofa.metrics.claude_judge.judge_behavior_with_claude", judge)
    result = tools_eval.judge_function(fn.name)
    assert result["ok"], result
    assert fn.scores["behavior_review_localisations"][retained_id] == original_localisation
    assert (fn.scores["behavior_review_localisations"][retry_id]
            == fn.failure_localisations[retry_id])
    fn.scores.pop("evaluation_context")
    for _ in range(2):
        assert tools_eval.evaluate_functions(limit=1)["ok"]
        assert fn.scores["behavior_evidence_stale"] is False
        assert resolved_behavior_purity(fn.scores) == 1.


def test_processing_retry_of_a_stale_compiler_contract_requires_fresh_validation(
    retained_runtime, monkeypatch,
):
    from autofa.metrics.judge_backend import backend_name
    from autofa.session import repair_workflow

    rt, fn, items = retained_runtime
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    retry_id = items[0]["_item_id"]
    install_reviews(fn, [items[-1]])
    judge_config, _, settings = tools_eval._review_settings(rt.config)
    fn.scores.update(settings, judge_backend=backend_name(judge_config["harness"]),
                     judge_harness=judge_config["harness"], behavior_n_judged=2,
                     behavior_n_errors=1)
    fn.scores["behavior_verdicts"].append({
        "item_id": retry_id, "verdict": "error", "processing_status": "error",
    })
    fn.scores["claim_contract"] = build_claim_contract(
        source=fn.source, description=fn.description, mechanism=fn.mechanism,
        defining_clauses=[{"id": "clause_1", "source_field": "description",
                           "quote": fn.description, "predicate": fn.description,
                           "required_review_scope": "local"}],
        compilation={"policy_hash": "0" * 64, "model": "test", "exclusions": [],
                     "coverage_review": {"approved": True, "issues": []}},
    )
    monkeypatch.setattr(repair_workflow, "validation_context", lambda *args, **kwargs: {
        "plan_id": "saved-retry", "retry_item_ids": [retry_id],
        "judge_objective_strategy": "balanced",
        "judge_behavior_strictness": settings["judge_behavior_strictness"],
    })

    def unexpected(*args, **kwargs):
        pytest.fail("a retry that cannot merge must not compile or review")

    monkeypatch.setattr(repair_workflow, "reserve_validation", unexpected)
    monkeypatch.setattr("autofa.metrics.claude_judge.judge_behavior_with_claude", unexpected)
    result = tools_eval.judge_function(fn.name)
    assert not result["ok"]
    assert "earlier compiler policy" in result["error"]
    assert "fresh validation" in result["error"]
