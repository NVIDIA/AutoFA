# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector repair workflows, reviewed functions, review exposure and improvement replies."""

import json
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.dependencies import review_source
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_eval, tools_library, tools_repair
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics import claude_judge
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.objectives import apply_preset
from autofa.session import family_discovery
from autofa.session.family_discovery import initialize_family_discovery
from autofa.session.repair_workflow import get_repair

# Detector repair workflow.
SOURCE = "def guess(item, llm_client=None):\n    return 'guess' in item['response_text']\n"
TIGHTER = ("def guess(item, llm_client=None):\n"
           "    text = item['response_text']\n"
           "    return 'guess' in text and 'final' in text\n")
DESCRIPTION = "The assistant submits an unverified guess."
MECHANISM = "The assistant substitutes a guessed value for a missing derived answer."


class ItemsDataset(Dataset):
    """In-memory attempts of model ``m`` on benchmark ``b``."""

    def __init__(self, items):
        super().__init__(parquet_paths=[], model_id="m", benchmarks=["b"])
        self.items = items

    def available_columns(self):
        return set(self.items[0])

    def estimate_matching_rows(self):
        return len(self.items)

    def iter_batches(self, *, batch_size=512, max_items=None, columns=None):
        yield self.items[:max_items]


def case(index, *, correct, text):
    """Attempt ``case-<index>``, the only attempt of task ``<index>``."""
    return {"_item_id": f"case-{index}", "model_id": "m", "benchmark": "b",
            "problem_id": str(index), "run_idx": 0, "is_correct": correct, "response_text": text}


def unreviewed_runtime(tmp_path, monkeypatch, fn):
    """Runtime over the 64-item corpus with a stubbed judge backend; nothing is reviewed yet.

    The stub answers per item from ``statuses`` (``default``: supported) and records
    every reviewed item ID in ``calls``. The caller must request ``stub_claim_compiler``.
    """
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    monkeypatch.setattr("autofa.metrics.improvements.classify_improvement",
                        lambda **kwargs: {"targets": [], "status": "insufficient_evidence"})
    # Tightening excludes case-0; make it a benchmark pass so the
    # repaired candidate still reaches the semantic-review workflow.
    items = [case(i, correct=i % 5 == 0, text="I guess the final value.") for i in range(64)]
    items[0]["response_text"] = "I guess an intermediate value."
    rt = Runtime(
        dataset=ItemsDataset(items), library=FunctionLibrary(functions=[fn]),
        run=RunConfig(run_id="repair-flow", model_id="m", benchmarks=["b"], parquet_paths=[]),
        config=apply_preset(Config(
            harness="claude", behavior_judge_model="claude-sonnet-5-5",
            behavior_judge_thinking="adaptive", judge_workers=1,
        ), "balanced"),
    )
    set_runtime(rt)
    calls = []
    statuses = {}

    def review(*, review, **kwargs):
        item_id = review.snapshot["item_id"]
        calls.append(item_id)
        status = statuses.get(item_id, statuses.get("default", "supported"))
        if status == "error":
            return "malformed judge response"
        for field in review.fields:
            review.read(field)
        text = review.fields["response_text"]
        citation = review.cite("response_text", text)["citations"][0]
        contract = review.snapshot["claim_contract"]
        mechanism_id = next(c["id"] for c in contract["defining_clauses"]
                            if c["source_field"] == "mechanism")
        assessments = [{
            "requirement_id": requirement["id"],
            "status": status if requirement["id"] in {"causal_link", mechanism_id}
            else "supported",
            **({"defining_requirement_ids": [mechanism_id]}
               if requirement["id"] == "causal_link" and status != "supported" else {}),
            "rationale": (
                "The visible reasoning establishes this condition." if status == "supported"
                else "The claimed origin of the guess is contradicted or unavailable."
            ),
            "citations": [{"citation_id": citation["citation_id"]}],
        } for requirement in contract["requirements"]]
        return json.dumps({
            "verdicts": [{"item_id": item_id, "snapshot_id": review.snapshot["snapshot_id"],
                          "claim_contract_id": contract["contract_id"], "scope": "local",
                          "claim_assessments": assessments}],
            "behavior_specificity": .8,
            "claim_calibration": {"status": "mixed", "claim_note": "Inspect the causal claim.",
                                  "detector_note": "The guess is visible.",
                                  "rewrite_hint": "Review the earlier derivation."},
        })

    monkeypatch.setattr(claude_judge, "_run_review", review)
    return rt, statuses, calls


@pytest.fixture
def repair_runtime(tmp_path, monkeypatch, stub_claim_compiler):
    """Runtime whose ``guess`` detector has a judged baseline on case-0..case-3.

    The baseline holds one contradicted, one unknown, one processing-error and one
    supported review, so repair plans can cite each finding status.
    """
    fn = FeatureFunction(name="guess", source=SOURCE, description=DESCRIPTION, mechanism=MECHANISM)
    rt, statuses, calls = unreviewed_runtime(tmp_path, monkeypatch, fn)
    statuses.update({"case-0": "contradicted", "case-1": "unknown", "case-2": "error"})
    all_items = rt.dataset.items
    rt.dataset.items = all_items[:4]
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    baseline_review = tools_eval.judge_function(fn.name, evidence_n=4)
    assert all(baseline_review[key] == 1 for key in (
        "n_yes", "n_no", "n_uncertain", "n_errors",
    )), baseline_review
    rt.dataset.items = all_items
    # Production runs have a fixed corpus. This fixture expands its in-memory
    # source after baseline evaluation, so discard that earlier identity cache.
    rt._family_failed_task_inventory = None
    calls.clear()
    return rt, statuses, calls


def plan(action, *, name="guess", status="contradicted"):
    response = tools_repair.get_function_repair(name)
    assert response["ok"], response
    finding = next(row for row in response["feedback"]["findings"] if row["status"] == status)
    result = tools_repair.plan_function_repair(
        name=name, action=action,
        rationale="Address this reviewed condition using its cited trace.",
        finding_ids=[finding["id"]], evidence_ids=[finding["examples"][0]["item_id"]],
    )
    assert result["ok"], result
    return result["plan"]


def workflow(rt, name="guess"):
    """Read full on-disk snapshots; the MCP response intentionally stays compact."""
    return get_repair(rt.run.run_dir(), name)


def submit(plan, *, name="guess", source=TIGHTER, description=DESCRIPTION, mechanism=MECHANISM):
    result = tools_library.submit_function(
        name=name, source=source, description=description, mechanism=mechanism,
        repair_plan_id=plan["plan_id"], auto_evaluate=False,
    )
    assert result["ok"], result
    evaluation = tools_eval.evaluate_functions(names=[name], mode="full")
    assert evaluation["ok"], evaluation
    return result


def _finish_family_comparisons(rt):
    # These tests isolate review reservations. The family workflow tests cover
    # reaching this phase through checked surveys and candidate assessments.
    path = rt.run.run_dir() / family_discovery.STATE_FILE
    state = json.loads(path.read_text())
    state["phase"] = "specialize"
    path.write_text(json.dumps(state))


# Judge iteration feedback.
CALIBRATION = {
    "status": "mixed", "claim_note": "The required causal link is not established.",
    "detector_note": "The guessed value is visible, its origin is not.",
    "rewrite_hint": "Inspect the preceding calculation before changing the claim.",
}


def _runtime(tmp_path, monkeypatch, *, automatic_eligible=False):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    monkeypatch.setattr("autofa.metrics.improvements.classify_improvement",
                        lambda **kwargs: {"targets": [], "status": "insufficient_evidence"})
    fn = FeatureFunction(name="guess", source=SOURCE, description=DESCRIPTION, mechanism=MECHANISM)
    rt = Runtime(
        dataset=ItemsDataset([case(i, correct=i == 3, text="I guess the value.")
                              for i in range(4)]),
        library=FunctionLibrary(functions=[fn]),
        run=RunConfig(run_id="feedback", model_id="m", benchmarks=["b"], parquet_paths=[]),
        config=Config(harness="claude", behavior_judge_model="claude-sonnet-5-5",
                      behavior_strictness="causal"),
    )
    if automatic_eligible:
        # Automatic review needs strong normalized association and matches on five
        # distinct failed tasks. Retain a matched passing example as a review control.
        rt.dataset.items.extend(case(i, correct=i >= 6, text="I guess the value." if i < 6
                                     else "A fully derived answer.") for i in range(4, 16))
    set_runtime(rt)
    return rt, fn


def _result(fn, outcomes, *, item_ids=None):
    contract = build_claim_contract(source=fn.source, description=fn.description,
                                    mechanism=fn.mechanism)
    rows = [
        {"item_id": item_ids[i] if item_ids is not None else f"case-{i}", "verdict": outcome,
         "rationale": "The required causal link could not be established.",
         "claim_assessments": [{
             "requirement_id": "causal_link", "status": "unknown" if outcome != "no"
             else "contradicted", "rationale": "No preceding calculation is available.",
         }], "initial_review": {"private": "FULL-REVIEW-DETAIL " * 2000}}
        for i, outcome in enumerate(outcomes)
    ]
    n_yes, n_no = outcomes.count("yes"), outcomes.count("no")
    return {
        "backend": "claude_agent_sdk", "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": claude_judge.judge_policy_hash(),
        "claim_hash": contract["contract_id"], "claim_contract": contract,
        "n_judged": len(rows), "n_yes": n_yes, "n_no": n_no,
        "n_uncertain": outcomes.count("uncertain"), "n_errors": outcomes.count("error"),
        "n_completed": n_yes + n_no,
        "behavior_purity": n_yes / len(rows) if n_yes + n_no else None,
        "behavior_specificity": 0.8 if n_yes + n_no else None,
        "rationale": "Completed item review outcomes.", "verdicts": rows,
        "claim_calibration": CALIBRATION if any(o != "error" for o in outcomes) else None,
        "item_calibrations": [{"item_id": row["item_id"], "claim_calibration": CALIBRATION}
                              for row in rows if row["verdict"] != "error"],
    }


# Family review mode, assessment and allocation.
REQUEST = {
    "reason": "across_task_consistency",
    "question": "Does the missing prerequisite recur across different tasks?",
    "expected_evidence": "A distinct task with retained prerequisite records.",
    "preserve_original_findings": True,
}


@pytest.fixture
def family_review(tmp_path, monkeypatch, stub_claim_compiler):
    """An evaluated, never-reviewed family detector matching all 32 items.

    Its default standard review therefore samples four cases (ten percent, rounded up).
    """
    fn = FeatureFunction(
        name="family_guess", source=SOURCE.replace("def guess(", "def family_guess("),
        description=DESCRIPTION, mechanism=MECHANISM)
    rt, _, calls = unreviewed_runtime(tmp_path, monkeypatch, fn)
    rt.dataset.items = rt.dataset.items[:32]
    initialize_family_discovery(rt)
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    return rt, fn, calls


# Repair feedback.
CLAIM = {
    "source": "def detector(item): return 'guess' in item['response_text']",
    "description": "The assistant submits an incorrect value.",
    "mechanism": "The assistant invents a rate because authoritative rates are unavailable.",
}


def citation(quote="The documented rate is 4%.", field="response_text"):
    return {"citation_id": "cite-1", "field": field, "start": 5,
            "end": 5 + len(quote), "quote": quote}


def assessment(status, *, field="response_text"):
    return {
        "requirement_id": "clause_1", "status": status,
        "rationale": {"supported": "The rate source is absent in this complete trace.",
                      "contradicted": "The authoritative rate is in the prompt.",
                      "unknown": "The upstream rate lookup is missing from the trace."}[status],
        "citations": [citation(field=field)],
        "counterevidence": [citation("An authoritative source was supplied.", field)]
        if status == "contradicted" else [],
    }


def reviewed_function(*statuses, field="response_text"):
    contract = build_claim_contract(**CLAIM, defining_clauses=[{
        "id": "clause_1", "source_field": "mechanism",
        "quote": "authoritative rates are unavailable",
        "required_review_scope": "local",
        "predicate": "Authoritative rates are unavailable.",
    }])
    outcomes = {"supported": "yes", "contradicted": "no", "unknown": "uncertain",
                "error": "error"}
    rows = []
    for i, status in enumerate(statuses):
        row = {"item_id": f"item-{i:03}", "verdict": outcomes[status],
               "claim_contract_id": contract["contract_id"],
               "processing_status": "error" if status == "error" else "completed",
               "rationale": "SDK review timed out" if status == "error" else status,
               "claim_assessments": [] if status == "error" else [
                   {**assessment(status, field=field), "requirement_id": requirement["id"]}
                   for requirement in contract["requirements"]
               ]}
        rows.append(row)
    yes, no = statuses.count("supported"), statuses.count("contradicted")
    return {"name": "detector", **CLAIM, "scores": {
        "bucket": "missing-rate", "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(), "judge_backend": "claude_agent_sdk",
        "claim_hash": claim_hash(**CLAIM), "claim_contract": contract,
        "behavior_verdicts": rows, "behavior_n_judged": len(rows),
        "behavior_n_yes": yes, "behavior_n_no": no,
        "behavior_n_uncertain": statuses.count("unknown"),
        "behavior_n_errors": statuses.count("error"), "behavior_n_completed": yes + no,
        "behavior_purity": yes / len(rows) if yes + no else None,
    }}


# Behavior strictness repairs.
def strictness_function(profile, status="supported"):
    fn = reviewed_function(status)
    scores = fn["scores"]
    contract = build_claim_contract(
        **{key: fn[key] for key in ("source", "description", "mechanism")},
        defining_clauses=scores["claim_contract"]["defining_clauses"],
        behavior_strictness=profile,
    )
    scores.update(
        claim_contract=contract, judge_behavior_strictness=profile,
        judge_policy_hash=judge_policy_hash(behavior_strictness=profile),
    )
    row = scores["behavior_verdicts"][0]
    row["claim_contract_id"] = contract["contract_id"]
    template = row["claim_assessments"][0] if status != "error" else None
    row["claim_assessments"] = ([{**template, "requirement_id": req["id"]}
                                 for req in contract["requirements"]] if template else [])
    return fn


# Round refinement repairs.
def _claim(fn):
    return claim_hash(source=review_source(fn), description=fn.description,
                      mechanism=fn.mechanism)


def _refinement(rt, **overrides):
    state = {
        "version": 1, "run_id": rt.run.run_id, "round": rt.run.round,
        "status": "awaiting_response",
        "evaluated_claims": {"guess": _claim(rt.library.get("guess"))},
        **overrides,
    }
    (rt.run.run_dir() / "round_validation.json").write_text(json.dumps({"refinement": state}))
    return state


def _synthesis_pending(rt, monkeypatch):
    # Freeze only the scheduling input. Plans still use real saved findings,
    # claim contracts, repair ledgers and the production receipt reader.
    state = family_discovery.initialize_family_discovery(rt)
    state.update(phase="specialize", comparisons=[{}, {}, {}], comparison_budget=3,
                 mutation_attempts={"guess": 7},
                 synthesis_required={"seed_functions": ["other_seed"]})
    path = rt.run.run_dir() / family_discovery.STATE_FILE
    path.write_text(json.dumps(state))
    monkeypatch.setattr(family_discovery, "initialize_family_discovery", lambda _rt: state)
    monkeypatch.setattr(family_discovery, "family_progress_summary", lambda *_: {
        "phase": "survey", "synthesis_seed_functions": ["other_seed"],
        "next_action": "Inspect supported seeds.",
    })
    return path


def _propose():
    feedback = tools_repair.get_function_repair("guess")["feedback"]
    finding = next(row for row in feedback["findings"] if row["status"] == "contradicted")
    return tools_repair.plan_function_repair(
        "guess", "tighten_detector", "Address the reviewed contradiction.",
        [finding["id"]], [finding["item_ids"][0]],
    )


# Review exposure tools.
class Identities:
    def __init__(self):
        self.rows = [{"_item_id": f"opaque-{i}", "benchmark": "b", "problem_id": str(i // 2)}
                     for i in range(12)]
        self.lookups = 0
        self.columns = []

    def iter_batches(self, *, columns, max_items=None):
        self.lookups += 1
        self.columns.extend(columns)
        yield self.rows[:max_items]

    def identity_index(self):
        self.lookups += 1
        return SimpleNamespace(by_id={row["_item_id"]: i for i, row in enumerate(self.rows)})

    def read_locations(self, locations):
        return [self.rows[location] for location in locations]


def exposure_runtime(tmp_path, monkeypatch, *, family=True):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path)
    rt = Runtime(dataset=Identities(), library=FunctionLibrary(), config=Config(),
                 run=RunConfig(run_id="run", model_id="m", parquet_paths=[]))
    rt.run.run_dir().mkdir(parents=True, exist_ok=True)
    if family:
        (rt.run.run_dir() / "family_discovery.json").write_text('{"version":1}')
    set_runtime(rt)
    return rt


# Improvements.
def _reply(targets=None, **changes):
    return {"status": "assessed", "targets": ["model", "harness"] if targets is None else targets,
            "rationale": "Wrong arguments receive indistinguishable errors.",
            "model_change": "Use the code field with a nonempty body.",
            "harness_change": "Distinguish wrong keys from empty code in validation errors.",
            "validation_test": "Replay wrong-key, empty, null, and valid-code requests.",
            "evidence_ids": ["item-1"], **changes}


def _fn():
    return FeatureFunction(name="bad_arguments", description="Ambiguous code errors",
                           mechanism="Wrong-key and empty-body requests get the same error.",
                           source="def bad_arguments(item, llm_client=None):\n    return True\n")
