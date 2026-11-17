# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family discovery runtimes, recorded families and completed round batches."""

import json
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.objectives import apply_preset, discovery_strategy
from autofa.session.exploration import resolve_exploration_seed, round_exploration_seed
from autofa.session.family_coverage import cached_task_inventory
from autofa.session.family_discovery import STATE_FILE, family_discovery
from autofa.session.round_validation import validate_round

from support.repair_review import _runtime


# Family discovery.
class FamilyDataset:
    parquet_paths = []

    def __init__(self):
        self.items = [
            {"_item_id": "a1", "problem_id": "a", "benchmark": "bench", "is_correct": False,
             "response_text": "Rejected invalid call; the required file was never created."},
            {"_item_id": "a2", "problem_id": "a", "benchmark": "bench", "is_correct": False,
             "response_text": "Rejected invalid call; the required file was never created."},
            {"_item_id": "b1", "problem_id": "b", "benchmark": "bench", "is_correct": False,
             "response_text": "Missing argument rejected; no required result was produced."},
            {"_item_id": "c1", "problem_id": "c", "benchmark": "bench", "is_correct": True,
             "response_text": "Rejected call corrected; required result produced."},
        ]

    def load_items(self, **kwargs):
        return self.items


@pytest.fixture
def rt(tmp_path, monkeypatch):
    from autofa.mcp import runtime

    rt = SimpleNamespace(
        dataset=FamilyDataset(), library=FunctionLibrary(),
        config=apply_preset(Config(), "balanced"), scores_history=[],
        run=SimpleNamespace(run_id="r1", round=1, run_dir=lambda: tmp_path,
                            exploration_base_seed=resolve_exploration_seed(run_id="r1"),
                            exploration_sequence_index=1),
        persist_library=lambda: None,
    )
    monkeypatch.setattr(runtime, "get_runtime", lambda: rt)
    return rt


def evidence(item_id="a1", role="support"):
    quote = {
        "a1": "Rejected invalid call", "a2": "Rejected invalid call",
        "b1": "Missing argument rejected", "c1": "Rejected call corrected",
    }[item_id]
    return {"item_id": item_id, "role": role,
            "causal_chain": "Required output depends on the call; rejection left it absent."
            if role == "support" else "Correction produced the required output.",
            "quotes": [{"field": "response_text", "quote": quote}]}


def record(**overrides):
    args = dict(action="record_family", family_name="rejected_required_call",
                description="Invalid tool calls prevent required operations",
                mechanism="The rejected call leaves a necessary operation undone.",
                evidence=[evidence(), evidence("b1"), evidence("c1", "boundary")],
                boundary_notes="Exclude calls corrected before the required output is produced.")
    args.update(overrides)
    return family_discovery(**args)


def evaluated_family(rt):
    state = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
    survey = state["survey"]
    fn = FeatureFunction(name=survey["family_name"], description=survey["description"],
                         mechanism=survey["mechanism"],
                         source="def rejected_required_call(item): return False")
    fn.scores = {"evaluated": True, "evaluation_run_id": "r1", "recall": 0.0,
                 "evaluation_claim_hash": claim_hash(source=fn.source,
                                                     description=fn.description,
                                                     mechanism=fn.mechanism)}
    rt.library.upsert(fn)
    return fn


def expand_failures(rt, count=12, *, task_identity=True):
    rt.dataset.items = [
        {"_item_id": f"failed_{i}", "benchmark": "bench", "is_correct": False,
         **({"problem_id": f"task_{i}"} if task_identity else {}),
         "response_text": f"Trace {i}: required action was rejected and output is absent."}
        for i in range(count)
    ] + [rt.dataset.items[-1]]


def survey_rows(*indices):
    return [
        {"item_id": f"failed_{i}", "role": "support",
         "causal_chain": "The rejected prerequisite left the required output absent.",
         "quotes": [{"field": "response_text", "quote": f"Trace {i}: required action"}]}
        for i in indices
    ] + [evidence("c1", "boundary")]


def assess_recorded(rt):
    fn = evaluated_family(rt)
    result = family_discovery(action="assess_family", family_name=fn.name,
                              assessment="Evaluated a rare, low-recall family; retain it.")
    assert result["ok"], result
    return result


def reviewed_seed(rt, name, item_ids, *, limited=False):
    """Use real saved localization and review provenance, not aggregate purity."""
    fn = FeatureFunction(name=name, description=f"Specific required call rejected: {name}",
                         mechanism=f"A rejected {name} prerequisite leaves output absent",
                         source=f"def {name}(item): return True")
    strictness = rt.config.behavior_strictness
    contract = build_claim_contract(source=fn.source, description=fn.description,
                                    mechanism=fn.mechanism, behavior_strictness=strictness)
    fingerprint = claim_hash(source=fn.source, description=fn.description, mechanism=fn.mechanism)
    strategy = discovery_strategy(rt.config)
    fn.scores = {
        "evaluated": True, "evaluation_run_id": rt.run.run_id,
        "evaluation_claim_hash": fingerprint, "recall": .05, "precision": 1.,
        "claim_hash": fingerprint, "claim_contract": contract,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(behavior_strictness=strictness),
        "judge_behavior_strictness": strictness,
        "judge_objective_strategy": strategy, "behavior_n_judged": len(item_ids),
        "behavior_n_yes": len(item_ids), "behavior_n_no": 0,
        "behavior_sample": {"limited": limited, "phase": "development"},
        "behavior_verdicts": [],
    }
    for item_id in item_ids:
        fn.failure_localisations[item_id] = {
            "item_id": item_id, "assigned_label": "True", "source_sha256": text_hash(fn.source),
        }
        initial = {"item_id": item_id, "snapshot_id": "snapshot-" + item_id,
                   "claim_contract_id": contract["contract_id"], "verdict": "yes"}
        fn.scores["behavior_verdicts"].append({
            **initial, "verification_status": "confirmed", "initial_review": dict(initial),
        })
    rt.library.upsert(fn)
    return fn


# Finish actual comparisons before tests inspect an unrelated round transition.
def complete_round_batch(rt):
    """Use actual fixture traces, inspecting new tasks each attempt."""
    failed_tasks = cached_task_inventory(rt)["task_by_failed_item_id"]
    rows = [{
        "item_id": item["_item_id"],
        "role": "support" if item["_item_id"] in failed_tasks else "boundary",
        "causal_chain": "The trace does not establish a shared causal failure mechanism.",
        "quotes": [{"field": "response_text", "quote": item["response_text"]}],
    } for item in rt.dataset.load_items()]
    # Repeated attempts of one task cannot substitute for distinct failed tasks.
    failures, seen = [], set()
    for row in rows:
        if row["item_id"] not in failed_tasks:
            continue
        identity = failed_tasks[row["item_id"]] or row["item_id"]
        if identity not in seen:
            seen.add(identity)
            failures.append(row)
    boundaries = [row for row in rows if row["role"] == "boundary"][:1]
    result = family_discovery()
    assert result["ok"], result
    for _ in range(result["comparison_budget"]):
        if result["phase"] == "specialize":
            break
        start = result["completed_comparisons"] * 2
        rotated = failures[start:] + failures[:start]
        selected = rotated[:result["suggested_distinct_tasks"]]
        result = family_discovery(
            action="no_coherent_family",
            decision={
                "outcome": "no_shared_relation",
                "reason": "Checked source cases do not establish one shared operation.",
                "next_action": "Inspect a different supported relation.",
            },
            evidence=selected + boundaries,
            boundary_notes="Inspected task outcomes do not establish one shared cause.",
            assessment="The available traces do not support one shared failure mechanism.",
        )
        assert result["ok"], result
    assert result["phase"] == "specialize"
    return result


def complete_small_corpus_round(rt):
    """The tiny feedback fixture exhausts its evidence in a single real comparison."""
    result = complete_round_batch(rt)
    assert result["exploration_exhausted"] == "available_failed_task_evidence_exhausted"


# Round batch completion.
@pytest.fixture
def batch_runtime(tmp_path, monkeypatch):
    rt, _ = _runtime(tmp_path, monkeypatch)
    rt.config = apply_preset(rt.config, "balanced")
    rt.dataset = FamilyDataset()
    expand_failures(rt, 20)
    rt.library.functions = []
    rt.run.round, rt.run.max_rounds, rt.run.min_failure_modes = 1, 8, 2
    rt.run.exploration_base_seed = 42
    rt.run.exploration_sequence_index = 37
    rt.run.exploration_seed = round_exploration_seed(42, 37)
    rt.scores_history = [{"round": 0, "note": "Preserve earlier score history."}]

    def unexpected_paid_review(*args, **kwargs):
        pytest.fail("Round comparison bookkeeping must not invoke a paid reviewer")

    monkeypatch.setattr("autofa.metrics.claude_judge._run_review", unexpected_paid_review)
    # These tests isolate family/seed bookkeeping. Actual checkpoint scoring,
    # feedback and completion are covered in test_round_validation_flow.
    monkeypatch.setattr("autofa.session.round_validation.validate_round",
                        lambda runtime: {"ok": True, "message": "Validated round fixture."})
    _save_checkpoints(rt)
    return rt


def _save_checkpoints(rt):
    rt.run.save()
    rt.persist_library()
    rt.persist_scores_history()


def _terminal_comparison(rt, index, outcome):
    rows = survey_rows(*range(index * 2, index * 2 + 6))
    if outcome == "no_coherent_family":
        result = family_discovery(
            action="no_coherent_family",
            decision={
                "outcome": "no_shared_relation",
                "reason": "Checked source cases do not establish one shared operation.",
                "next_action": "Inspect a different supported relation.",
            },
            evidence=rows,
            boundary_notes="Recovery and different task requirements were checked.",
            assessment="The inspected failures depend on incompatible causal operations.",
        )
    else:
        name = f"family_{index}"
        recorded = record(
            family_name=name, evidence=rows,
            mechanism=f"Distinct prerequisite {index} left the required output absent.",
        )
        assert recorded["ok"], recorded
        if outcome == "deferred":
            result = family_discovery(
                action="defer_family", family_name=name,
                blocker="Required execution backend is unavailable.",
                next_step="Restore that backend and revisit this open hypothesis.",
            )
        else:
            fn = evaluated_family(rt)
            fn.scores["evaluation_run_id"] = rt.run.run_id
            result = family_discovery(
                action="assess_family", family_name=name,
                assessment="The current evaluation has zero recall; retain the finding.",
            )
    assert result["ok"], result
    return result


def _complete_retained_batch(rt):
    for index, outcome in enumerate(("assessed", "no_coherent_family", "no_coherent_family")):
        _terminal_comparison(rt, index, outcome)
    assert rt.library.functions
    _save_checkpoints(rt)


# Polishing flow.
@pytest.fixture
def polishing_round(batch_runtime, monkeypatch):
    rt = batch_runtime
    _complete_retained_batch(rt)
    rt.run.max_rounds = 2
    calls = []

    def score(**kwargs):
        calls.append(("full", kwargs))
        rt.library.functions[0].scores.update(ppv_at_pi0=.55, sampled=False)
        return {"ok": True}

    monkeypatch.setattr("autofa.metrics.final_validation_selection.prepare_final_validation",
                        lambda rt: {"selected_names": [f.name for f in rt.library.functions],
                                    "unresolved": []})
    monkeypatch.setattr("autofa.mcp.tools_eval.final_evaluate_functions", score)
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions",
                        lambda **kwargs: {"ok": True, "unresolved": []})
    monkeypatch.setattr("autofa.session.round_validation.validate_round", validate_round)
    return rt, calls
