# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery targets, stalls, grades, round stagnation and round feedback snapshots."""

import logging
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.data.parquet_store import Dataset
from autofa.harness.claude import ClaudeHarness
from autofa.mcp import tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.discovery_grades import GRADE_CODES
from autofa.metrics.task_support import build_task_support
from autofa.objectives import discovery_strategy
from autofa.session.discovery_stall import complete_discovery_round
from autofa.session.family_coverage import failed_task_inventory

from support.task_support import attach_sampled_task_support, attach_task_support


# Discovery target.
def source_inventory(failures=10, passes=10, *, tasks=None):
    failed = {f"f{i}": f"task-{i}" for i in range(failures)}
    if tasks is not None:
        failed.update(tasks)
    passed = {f"p{i}": f"passing-task-{i}" for i in range(passes)}
    all_items = {**failed, **passed}
    return {
        "inventory_complete": True, "n_items": failures + passes,
        "n_failed_attempts": failures, "n_passed_attempts": passes, "n_unknown_outcomes": 0,
        "task_by_failed_item_id": failed, "task_by_item_id": all_items,
        "identity_by_item_id": {key: {"_item_id": key} for key in all_items},
    }


def keeper(index=0, **changes):
    fn = FeatureFunction(
        name=f"mode_{index}", description=f"Mode {index}", mechanism=f"Mechanism {index}",
        source=f"def detect(item):\n    return 'behavior{index}' in item['response_text']\n",
    )
    contract = build_claim_contract(source=fn.source, description=fn.description,
                                    mechanism=fn.mechanism)
    fingerprint = claim_hash(source=fn.source, description=fn.description, mechanism=fn.mechanism)
    fn.scores = {
        **contingency(8, 0, 2, 10), "evaluated": True, "n_scored": 20,
        "eval_mode": "full", "sampled": False, "evaluation_scope_complete": True,
        "evaluation_run_id": "fixture-run", "task_recall_corpus_fingerprint": "fixture",
        "evaluation_claim_hash": fingerprint,
        "judge_backend": "claude_agent_sdk", "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(), "claim_hash": fingerprint,
        "claim_contract": contract,
        "behavior_n_judged": 5, "behavior_n_completed": 5,
        "behavior_n_yes": 5, "behavior_n_no": 0, "behavior_n_uncertain": 0,
        "behavior_n_errors": 0, "behavior_purity": 1.0, "behavior_specificity": .8,
        **changes,
    }
    for i in range(int(fn.scores["a"])):
        item_id = f"f{i}"
        fn.failure_localisations[item_id] = {
            "item_id": item_id, "source_sha256": text_hash(fn.source), "assigned_label": True,
        }
    for i in range(int(fn.scores["b"])):
        item_id = f"p{i}"
        fn.failure_localisations[item_id] = {
            "item_id": item_id, "source_sha256": text_hash(fn.source), "assigned_label": True,
        }
    fn.scores["behavior_verdicts"] = []
    for i in range(int(fn.scores["behavior_n_judged"])):
        initial = {"item_id": f"f{i}", "verdict": "yes", "snapshot_id": f"snapshot-{i}",
                   "claim_contract_id": contract["contract_id"], "completion_status": "complete"}
        fn.scores["behavior_verdicts"].append({
            **initial, "verification_status": "confirmed", "initial_review": dict(initial),
        })
    fn.scores.update(build_task_support(fn, {**source_inventory(),
                                            "corpus_fingerprint": "fixture"}))
    return fn


def graded_candidate(index=0, *, sampled=True, coverage=.08, tasks=2):
    fn = FeatureFunction(
        name=f"graded_{index}", description=f"Observable mistake {index}",
        mechanism=f"Mistaken state propagation {index}",
        source=f"def detect(item): return item.get('mistake_{index}', False)",
        scores={"recall": coverage, "ppv_at_pi0": 1., "behavior_purity": 1.,
                "n_flagged": tasks, "judge_behavior_strictness": "causal"},
    )
    if sampled:
        return attach_sampled_task_support(fn, task_n=tasks)
    fn.scores.update(sampled=False, evaluation_scope_complete=True)
    return attach_task_support(fn, task_n=tasks)


@pytest.fixture
def interactive(tmp_path, monkeypatch, request):
    from autofa import diagnostics
    from autofa.harness import claude, codex

    harness_module = codex if getattr(request, "param", "claude") == "codex" else claude
    harness_class = codex.CodexHarness if harness_module is codex else ClaudeHarness

    # The real runner configures process-wide diagnostics; keep its test run
    # from capturing the log destination of later sessions/tests.
    monkeypatch.setattr(diagnostics, "_DIAGNOSTICS", diagnostics.Diagnostics())
    logger = logging.getLogger("autofa")
    previous_handlers, previous_level = list(logger.handlers), logger.level
    monkeypatch.setattr("autofa.config.autofa_home", lambda: tmp_path)
    monkeypatch.setattr("prompt_toolkit.PromptSession", lambda **kwargs: object())
    monkeypatch.setattr("autofa.ui.menus.prepare_step", lambda: None)
    monkeypatch.setattr(claude, "describe_auth_mode", lambda *args: "test")
    harness = object.__new__(harness_class)
    harness.model, harness.cli_path, harness.effort = "test", None, None
    commands = []

    async def prompt(session):
        assert commands, "Unexpected user input while discovery should still be continuing"
        return commands.pop(0)

    monkeypatch.setattr(harness_module, "_async_prompt", prompt)
    yield harness, commands
    for handler in list(logger.handlers):
        if handler not in previous_handlers:
            logger.removeHandler(handler)
            handler.close()
    for handler in previous_handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)
    logger.setLevel(previous_level)


@pytest.fixture
def interactive_runner(interactive, tmp_path, monkeypatch):
    from autofa.session import runner

    harness, commands = interactive
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    harness_name = "codex" if type(harness).__name__ == "CodexHarness" else "claude"
    cfg = Config(behavior_strictness="causal", harness=harness_name,
                 library_root=str(tmp_path / "catalog"), min_failure_modes=4,
                 discovery_plateau_patience=2, discovery_wait_for_plateau=True)
    run = RunConfig(run_id="minimum", parquet_paths=[], model_id="m", round=3, max_rounds=3,
                    min_failure_modes=4)
    rt = Runtime(dataset=Dataset(parquet_paths=[], model_id="m"), run=run, config=cfg,
                 library=FunctionLibrary())
    rt._family_failed_task_inventory = {**source_inventory(), "corpus_fingerprint": "fixture"}
    set_runtime(rt)
    monkeypatch.setattr(runner, "load_runtime_from_run", lambda *args: rt)
    monkeypatch.setattr(runner, "get_harness", lambda *args, **kwargs: harness)
    monkeypatch.setattr(runner, "build_mcp_server", lambda *args: object())
    monkeypatch.setattr(runner, "check_llm_readiness", lambda *args, **kwargs: {})
    monkeypatch.setattr(tools_library, "_family_search_status",
                        lambda runtime: {"synthesis_pending": False})
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning",
                        lambda *args, **kwargs: SimpleNamespace(summary=lambda: "test"))
    monkeypatch.setattr("autofa.data.remote_library_publish.maybe_publish_run",
                        lambda *args, **kwargs: None)
    # These driver tests construct score snapshots without an executable corpus.
    # Real scorer/review provenance is covered by final-evaluation/review suites;
    # downgrade/resume tests replace this callback with their validation outcome.
    # Represent their completed full snapshots as verified review receipts. The
    # sampled/full distinction remains real; test_round_validation_flow exercises
    # the unmocked source identity and representative-cohort checks end to end.
    monkeypatch.setattr("autofa.session.validation_completion._validation_problem",
                        lambda rt, fn, measured: None if measured.get(
                            "corpus_evaluation_complete") else "Full validation is pending.")

    async def finalize(console):
        return None

    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    return harness, commands, rt


def _quality_stall_assessment(*, enabled=True, exhausted=True):
    return {
        "enabled": enabled, "should_stop": enabled and exhausted,
        "since_improve": 6 if exhausted else 0, "patience": 6,
        "completed_attempts": 7 if exhausted else 0,
        "round_stagnation": {
            "unit": "completed_extra_rounds", "patience": 3,
            "rounds_without_improve": 3 if exhausted else 0,
            "should_stop": enabled and exhausted,
        },
        "reason": "Three completed extra recovery rounds without validated library improvement"
        if exhausted else None,
    }


# Discovery stall.
def candidate(name, *, purity=0., precision=.6, reviewed=True, cohort="corpus"):
    fn = FeatureFunction(name=name, description=f"Mistake {name}", mechanism=f"Mechanism {name}",
                         source=f"def {name}(item): return item.get({name!r}, False)")
    contract = build_claim_contract(source=fn.source, description=fn.description,
                                    mechanism=fn.mechanism)
    fingerprint = contract["contract_id"]
    verdict = "yes" if purity else "no"
    row = {"item_id": "review", "snapshot_id": "snapshot", "claim_contract_id": fingerprint,
           "verdict": verdict, "processing_status": "completed", "completion_status": "complete"}
    if purity:
        row.update(verification_status="confirmed", initial_review=deepcopy(row))
    fn.scores = {
        "evaluated": True, "evaluation_claim_hash": fingerprint, "evaluation_run_id": "run",
        "evaluation_scope_complete": True, "evaluation_complete": True, "sampled": False,
        "n_items": 100, "n_scored": 100, "n_flagged": 5, "n_fail_caught": 5,
        "ppv_at_pi0": precision, "recall": .1, "precision": precision,
        "a": 5, "b": 2, "c": 45, "d": 48, "n_matched_failed_tasks": 5,
        "task_recall": .1, "task_recall_scope": "full_selected_corpus",
        "task_recall_corpus_fingerprint": cohort, "task_recall_claim_hash": fingerprint,
        # A census evaluation freezes its scoring panel over every corpus item.
        "search_panel": {
            "version": 1, "available": True, "scope": "fixed_scoring_panel",
            "claim_hash": fingerprint, "run_id": "run", "corpus_fingerprint": cohort,
            "cohort_fingerprint": f"census:{cohort}", "allocation_version": None,
            "n_items": 100, "n_failed_tasks": 50, "n_matched_failed_tasks": 5,
            "task_recall": .1, "observed_lower_bound": False,
        },
        "behavior_purity": purity if reviewed else None,
        "claim_hash": fingerprint, "claim_contract": contract,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(),
        "judge_objective_strategy": discovery_strategy(Config()),
        "behavior_n_judged": int(reviewed), "behavior_n_completed": int(reviewed),
        "behavior_n_yes": int(reviewed and bool(purity)),
        "behavior_n_no": int(reviewed and not purity), "behavior_n_uncertain": 0,
        "behavior_n_errors": 0, "behavior_verdicts": [row] if reviewed else [],
        "behavior_sample": {"phase": "development", "limited": True,
                            "actual_n": int(reviewed), "target_n": 1},
    }
    return fn


@pytest.fixture
def runtime(tmp_path):
    # round == max_rounds is the first extra round, so cost-accounting tests reach the
    # extra-round guard; test_discovery_stall covers planned-round boundaries explicitly.
    return SimpleNamespace(run=SimpleNamespace(run_id="run", round=3, max_rounds=3,
                                               run_dir=lambda: tmp_path),
                           config=Config(), library=FunctionLibrary(), scores_history=[])


def add(rt, fn):
    rt.library.functions.append(fn)
    rt.scores_history.append({"evaluated_names": [fn.name], "results": [{"name": fn.name}],
                              "evaluated_claims": {fn.name: fn.scores["evaluation_claim_hash"]}})


# Discovery grades.
def _function(name, coverage, *, tasks=5, repeats=1):
    fn = FeatureFunction(name=name, source="def detect(item): return True",
                         description="Observable unfinished work",
                         mechanism="Unfinished work prevents completion")
    fn.scores = {"evaluated": True, "n_items": 100, "n_flagged": 20,
                 "recall": coverage, "ppv_at_pi0": .9, "behavior_purity": 1.0}
    return attach_task_support(fn, task_n=tasks, repeat_n=repeats)


# Round stagnation.
@pytest.fixture
def rt(tmp_path):
    rows = [{"_item_id": str(index), "benchmark": "bench", "problem_id": str(index),
             "is_correct": index >= 50} for index in range(100)]
    inventory = failed_task_inventory(SimpleNamespace(load_items=lambda: rows))
    return SimpleNamespace(
        run=SimpleNamespace(run_id="quality-test", round=2, max_rounds=3,
                            run_dir=lambda: tmp_path),
        config=Config(), library=FunctionLibrary(), scores_history=[],
        _family_failed_task_inventory=inventory,
    )


def detector(rt, name, *, matches=range(5), coverage=.1, precision=.9, adherence=1.):
    fn = attach_task_support(FeatureFunction(
        name=name, source=f"def {name}(item): return item.get({name!r}, False)",
        description=f"Observable mistake {name}", mechanism=f"A distinct mechanism {name}",
        scores={"evaluated": True, "recall": coverage, "ppv_at_pi0": precision,
                "behavior_purity": adherence},
    ))
    matches = list(map(str, matches))
    fn.failure_localisations = {
        item_id: {"item_id": item_id, "source_sha256": text_hash(fn.source), "assigned_label": True}
        for item_id in matches
    }
    fn.scores.update(
        sampled=False, evaluation_scope_complete=True, evaluation_checkpoint_complete=True,
        n_items=100, n_scored=100,
        n_flagged=len(matches), n_fail_caught=sum(int(item) < 50 for item in matches),
        task_recall_corpus_fingerprint=rt._family_failed_task_inventory["corpus_fingerprint"],
    )
    fn.scores.update(build_task_support(fn, rt._family_failed_task_inventory))
    fn.scores["final_review"] = {
        "claim_hash": fn.scores["evaluation_claim_hash"], "scope": "full_corpus",
        "status": "complete", "identity": "representative-review",
        "item_ids": [row["item_id"] for row in fn.scores["behavior_verdicts"]],
    }
    return fn


def close(rt, *, fraction=1.0, ok=True):
    return complete_discovery_round(rt, {"ok": ok, "fraction": fraction})


def next_round(rt):
    rt.run.round += 1
    return close(rt)


# Round feedback.
def snapshot(*, score=.3, cohort="same-panel", confirmed=2, corpus="corpus"):
    counts = dict.fromkeys(GRADE_CODES, 0)
    counts["GOOD"] = 1
    return {
        "grades": {
            "counts": counts, "n_good_or_better": 1, "n_outstanding": 0,
            "functions": [{"name": "original", "code": "GOOD", "label": "GOOD",
                           "symbol": "●", "coverage": .05}],
        },
        "behavior_strictness": "causal",
        "search": {"value": score, "comparison_key": cohort},
        "task_breadth": {
            "corpus_fingerprint": corpus, "behavior_strictness": "causal",
            "quality_thresholds": {"precision": .7}, "n_confirmed_failed_tasks": confirmed,
            "n_failed_tasks": 100,
        },
    }
