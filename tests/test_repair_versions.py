# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved detector versions are compared, explained and restored without new validation."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.dependencies import digest, review_source
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.core.sandbox import NullLLM, run_detector
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_eval, tools_repair, tools_repair_versions
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.mcp.tools_library import submit_function
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.judge_evidence import make_snapshot
from autofa.metrics.repair_feedback import build_repair_feedback
from autofa.metrics.repair_tradeoff import compare_repair_coverage
from autofa.metrics.version_comparison import (
    compare_function_versions,
    version_recommendation,
    version_summary,
)
from autofa.metrics.version_lessons import saved_version_findings, version_lessons
from autofa.objectives import apply_preset, snapshot_objectives
from autofa.session import repair_workflow as workflow
from autofa.session.polishing_guard import polishing_tool_guard
from autofa.session.repair_versions import archive_scored_version, collect_versions
from autofa.util.atomic import atomic_write_json

from support.repair_review import reviewed_function
from support.task_support import attach_task_support

# Saved versions and restoration


def function(*, narrow=False, name="scope"):
    source = ('def detect(item):\n    return "omit" in item["response_text"]'
              + (' and "explicit" in item["response_text"]' if narrow else '') + '\n')
    fn = FeatureFunction(name=name, source=source, description="Omits a required case",
                         mechanism="A partial change leaves a required case uncorrected.")
    fp = claim_hash(source=source, description=fn.description, mechanism=fn.mechanism)
    fn.scores = {
        "evaluated": True, "evaluation_claim_hash": fp, "evaluation_run_id": "versions",
        "claim_hash": fp, "a": 47 if narrow else 68, "b": 41 if narrow else 51,
        "c": 345 if narrow else 310, "d": 997 if narrow else 973,
        "n_items": 1430 if narrow else 1402, "n_unresolved": 70 if narrow else 98,
        "errors": 0 if narrow else 5, "recall": 47 / 392 if narrow else 68 / 378,
        "behavior_purity": .5 if narrow else 11 / 13,
        "behavior_n_yes": 5 if narrow else 11,
        "behavior_n_no": 0 if narrow else 1,
        "behavior_n_uncertain": 5 if narrow else 1,
        "behavior_verdicts": [{"item_id": "reviewed", "verdict": "uncertain"}],
        "behavior_sample": {"actual_n": 10 if narrow else 13, "reused_n": 0,
                            "limited": False},
    }
    fn.failure_localisations = {"known": {"item_id": "known", "assigned_label": True}}
    return fn.to_dict()


def history(old, new, *, phase="exhausted"):
    plans = []
    for index in range(3):
        plans.append({
            "plan_id": f"attempt-{index}", "name": old["name"], "attempt": index + 1,
            "action": "tighten_detector", "phase": phase if index == 2 else "deferred",
            "before": deepcopy(old), "before_version_hash": workflow._version(old),
            "after": {**deepcopy(new), "scores": {}},
            "after_version_hash": workflow._version(new),
            "candidates": {old["name"]: {
                "after": {**deepcopy(new), "scores": {}},
                "after_validation": deepcopy(new),
                "after_version_hash": workflow._version(new),
                "phase": "deferred", "outcome": {"all_reviewed_confirmed": False},
            }},
        })
    return {
        "schema_version": workflow.SCHEMA_VERSION,
        "repairs": {old["name"]: {
            "name": old["name"], "baseline": deepcopy(old),
            "baseline_version_hash": workflow._version(old), "max_attempts": 3,
            "history": plans, "reviewed_item_ids": ["reviewed", "old-holdout"],
            "state": phase, "status": phase,
        }},
    }


def make_runtime(tmp_path, monkeypatch, *, run_id, config, functions=(), dataset=None):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    rt = Runtime(
        dataset=dataset or Dataset(parquet_paths=[]),
        library=FunctionLibrary(functions=[
            fn if isinstance(fn, FeatureFunction) else FeatureFunction.from_dict(fn)
            for fn in functions
        ]),
        run=RunConfig(run_id=run_id, model_id="m", benchmarks=["b"], parquet_paths=[]),
        config=config,
    )
    set_runtime(rt)
    return rt


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    rt = make_runtime(tmp_path, monkeypatch, run_id="versions", config=Config(),
                      functions=[function(narrow=True)])
    rt.persist_library()
    return rt


def save_history(rt, *, phase="exhausted"):
    data = history(function(), function(narrow=True), phase=phase)
    with workflow._ledger(rt.run.run_dir(), write=True) as ledger:
        ledger.update(data)
    return data


def restore(rt, old=None):
    old = old or function()
    current = rt.library.get(old["name"])
    return tools_repair_versions.restore_function_version(
        old["name"], workflow._version(old), "Keep the broader scored candidate for comparison.",
        workflow._version(current.to_dict()) if current else "",
    )


def test_collects_legacy_scored_snapshots_and_current_scores_without_mutation():
    old, new = function(), function(narrow=True)
    data = history(old, new)
    unscored = deepcopy(old)
    unscored["source"] += "\n# not evaluated"
    unscored["scores"] = {}
    data["repairs"]["scope"]["history"].append({"before": unscored, "candidates": {}})
    current = deepcopy(new)
    current["scores"]["errors"] = 2
    before = deepcopy(data)
    versions = collect_versions(data, "scope", current)
    assert set(versions) == {workflow._version(old), workflow._version(new)}
    assert versions[workflow._version(new)]["scores"]["errors"] == 2
    versions[workflow._version(old)]["scores"].clear()
    assert data == before


def test_collects_archived_split_children_without_creating_child_workflows():
    child = function(name="child")
    data = {"repairs": {"parent": {"history": [{"candidates": {"child": {
        "after_validation": child, "after_version_hash": workflow._version(child),
    }}}]}}}
    assert collect_versions(data, "child") == {workflow._version(child): child}


def test_archives_scored_versions_before_first_repair_only(tmp_path):
    old = function()
    archive_scored_version(tmp_path, old)
    unscored = function(narrow=True)
    unscored["scores"] = {}
    archive_scored_version(tmp_path, unscored)
    data = json.loads((tmp_path / "detector_repairs.json").read_text())
    assert data["repairs"] == {}
    assert collect_versions(data, "scope") == {workflow._version(old): old}


def test_listing_shows_coverage_and_review_cohorts_without_full_snapshots(runtime):
    save_history(runtime)
    response = tools_repair_versions.list_function_versions("scope")
    assert response["ok"]
    assert response["current_version_hash"] == workflow._version(function(narrow=True))
    rows = {row["version_hash"]: row for row in response["versions"]}
    prior = rows[workflow._version(function())]
    assert prior["metrics"]["a"] == 68
    assert prior["metrics"]["errors"] == 5
    assert prior["review_sample"]["actual_n"] == 13
    assert "source" not in prior and "function" not in prior


def test_selection_preserves_scores_budget_exclusions_and_is_reversible(runtime):
    saved = save_history(runtime)
    # Final polishing may restore scored versions without erasing failed edits.
    atomic_write_json(runtime.run.run_dir() / "round_validation.json", {"refinement": {
        "version": 1, "run_id": runtime.run.run_id, "round": runtime.run.round,
        "mode": "polishing", "status": "awaiting_response", "frozen_names": ["scope"],
        "polishing_id": f"{runtime.run.run_id}:{runtime.run.round}",
    }})
    for selected in (function(), function(narrow=True)):
        assert polishing_tool_guard(runtime, "restore_function_version", {"name": "scope"}) is None
        result = restore(runtime, selected)
        assert result["ok"], result
        assert runtime.library.get("scope").to_dict() == selected
        persisted = FunctionLibrary.load(runtime.working_library_path())
        assert persisted.get("scope").to_dict() == selected
        after = workflow.get_repair(runtime.run.run_dir(), "scope")
        assert after["attempts_used"] == 3
        assert after["state"] == "exhausted"
        assert after["history"] == saved["repairs"]["scope"]["history"]
        assert after["reviewed_item_ids"] == ["reviewed", "old-holdout"]
        with pytest.raises(ValueError, match="active plan"):
            workflow.validation_context(runtime.run.run_dir(), selected)


def test_restore_keeps_later_counterexamples_and_checks_exact_narrow_version(runtime, monkeypatch):
    old = function(name="detector")
    later = reviewed_function("contradicted", "unknown")
    later["scores"].update(
        evaluated=True, evaluation_claim_hash=later["scores"]["claim_hash"],
        evaluation_run_id=runtime.run.run_id,
    )
    later = FeatureFunction.from_dict(later).to_dict()
    runtime.library.upsert(FeatureFunction.from_dict(later))
    archive_scored_version(runtime.run.run_dir(), old)
    response = tools_repair_versions.restore_function_version(
        "detector", workflow._version(old), "Keep the supported original scope",
        workflow._version(later),
    )
    assert response["ok"], response
    assert response["counterexample_checks"]["item_n"] == 1
    lesson = next(row for row in response["retained_lessons"]["lessons"]
                  if row["version_hash"] == workflow._version(later))
    assert lesson["finding_counts"]["counterexample"] > 0
    assert lesson["claim_hash"] != old["scores"]["claim_hash"]
    assert lesson["applicability"] == "different_version_requires_applicability_check"
    assert lesson["mechanism"] == later["mechanism"]
    assert lesson["historical_only"] and not lesson["affects_current_grade"]
    assert runtime.library.get("detector").to_dict() == old

    # The actual suggested tool request checks this frozen version, rather than
    # copying the failed broad claim's verdict onto a restored narrow claim.
    action = response["counterexample_checks"]["actions"][0]
    assert action["tool"] == "smoke_test_feature"
    assert action["arguments"] == {
        "name": "detector", "expected_version_hash": workflow._version(old),
        "item_ids": ["item-000"],
    }
    rows = [{"_item_id": "item-000", "response_text": "guess with a supplied authoritative rate",
             "is_correct": False}]
    runtime.dataset = SimpleNamespace(
        parquet_paths=[], iter_batches=lambda **kwargs: iter([rows]),
    )
    monkeypatch.setattr(tools_eval, "_llm_adapter", NullLLM)
    result = tools_eval.smoke_test_feature(**action["arguments"])
    assert result["ok"] and result["claim_validation"] == "not_performed"
    assert result["results"][0]["matches_saved_bucket"] is False
    assert runtime.library.get("detector").to_dict() == old
    assert tools_eval.smoke_test_feature(**action["arguments"])["reused"] == 1


@pytest.mark.parametrize("target,error,reported", [
    # Historical guidance that cannot be read is reported beside the committed selection.
    ("autofa.metrics.version_lessons.version_lessons", OSError,
     "recovery_guidance_unavailable"),
    ("autofa.mcp.tools_repair_versions.upsert_mode", ValueError, "diagnostics"),
    ("autofa.mcp.tools_repair_versions._summary", ValueError, "diagnostics"),
])
def test_restore_advisory_failure_never_hides_committed_selection(
    runtime, monkeypatch, target, error, reported,
):
    save_history(runtime)

    def unavailable(*args, **kwargs):
        raise error("advisory output unavailable")

    monkeypatch.setattr(target, unavailable)
    response = restore(runtime)
    assert response["ok"] and response[reported]
    if error is OSError:
        assert "advisory output unavailable" in response[reported]
    assert runtime.library.get("scope").to_dict() == function()
    assert FunctionLibrary.load(runtime.working_library_path()).get("scope").to_dict() == function()


def test_restoration_check_actions_cover_every_saved_counterexample_with_bounded_panels(
    runtime, monkeypatch,
):
    monkeypatch.setattr("autofa.metrics.version_lessons.version_lessons", lambda *a, **kw: {})
    monkeypatch.setattr("autofa.metrics.version_lessons.saved_version_findings",
                        lambda *args: [
                            {"category": category, "item_id": str(index)}
                            for category, n in (("counterexample", 65), ("missing_fact", 90))
                            for index in range(n)
                        ])
    guidance = tools_repair_versions._restored_version_guidance(runtime, "scope", function())
    checks = guidance["counterexample_checks"]
    assert checks["item_n"] == 65
    assert [len(action["arguments"]["item_ids"]) for action in checks["actions"]] == [30, 30, 5]
    checked_ids = {item for action in checks["actions"]
                   for item in action["arguments"]["item_ids"]}
    assert len(checked_ids) == 65


def test_selection_waits_for_library_commit_before_locking_the_repair_ledger(
    runtime, monkeypatch,
):
    save_history(runtime)
    compiled, selecting = threading.Event(), threading.Event()
    compile_detector = tools_repair_versions.compile_detector
    select_version = workflow.select_version

    def compile_and_signal(*args, **kwargs):
        result = compile_detector(*args, **kwargs)
        compiled.set()
        return result

    def select_and_signal(*args, **kwargs):
        selecting.set()
        return select_version(*args, **kwargs)

    monkeypatch.setattr(tools_repair_versions, "compile_detector", compile_and_signal)
    monkeypatch.setattr(workflow, "select_version", select_and_signal)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with runtime._library_lock:
            future = pool.submit(restore, runtime)
            assert compiled.wait(5)
            # A commit can need the repair ledger while holding the library
            # lock. Selection must not hold that ledger while waiting for us.
            assert not selecting.wait(0.1)
        assert future.result(timeout=5)["ok"]
    assert selecting.is_set()


def test_active_repair_must_be_explicitly_deferred_before_selection(runtime):
    save_history(runtime, phase="awaiting_validation")
    before = runtime.working_library_path().read_bytes()
    result = restore(runtime)
    assert not result["ok"]
    assert "defer" in result["error"].lower() or "active" in result["error"].lower()
    assert runtime.working_library_path().read_bytes() == before


def test_restores_absent_function_without_new_attempt_or_review(runtime):
    archive_scored_version(runtime.run.run_dir(), function())
    runtime.library.remove("scope")
    result = restore(runtime)
    assert result["ok"], result
    assert runtime.library.get("scope").scores == function()["scores"]
    assert workflow.get_repair(runtime.run.run_dir(), "scope") == {}


def test_stale_expected_version_is_rejected_before_any_mutation(runtime):
    save_history(runtime)
    path = runtime.run.run_dir() / "detector_repairs.json"
    before = path.read_bytes()
    result = tools_repair_versions.restore_function_version(
        "scope", workflow._version(function()), "Compare", "",
    )
    assert not result["ok"]
    assert path.read_bytes() == before
    assert runtime.library.get("scope").to_dict() == function(narrow=True)


def test_current_field_policy_is_checked_before_selection(runtime):
    forbidden = function()
    forbidden["source"] = 'def detect(item): return item["is_correct"]\n'
    archive_scored_version(runtime.run.run_dir(), forbidden)
    path = runtime.run.run_dir() / "detector_repairs.json"
    before = path.read_bytes()
    result = restore(runtime, forbidden)
    assert not result["ok"]
    assert "disallowed" in result["error"] or "banned" in result["error"]
    assert path.read_bytes() == before


def test_frozen_helper_bundle_survives_without_current_helper_catalog(runtime):
    old = function()
    helper = {"role": "helper", "entrypoint": "extract", "source":
              'def extract(text): return "omit" in text\n', "helpers": {}, "detectors": {},
              "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
              "output_schema": {"type": "boolean"}}
    ref = digest(helper)
    bundle = {"version": 1, "helpers": {"extract": ref}, "detectors": {},
              "nodes": {ref: helper}}
    old["source"] = 'def detect(item): return helpers.call("extract", item["response_text"])\n'
    old["meta"] = {"dependency_bundle": bundle}
    old["uses_helpers"] = {"extract": ref}
    old["scores"]["evaluation_claim_hash"] = claim_hash(
        source=review_source(old), description=old["description"], mechanism=old["mechanism"],
    )
    archive_scored_version(runtime.run.run_dir(), old)
    result = restore(runtime, old)
    assert result["ok"], result
    restored = runtime.library.get("scope")
    assert restored.to_dict() == old
    assert not runtime.library.helpers.revisions
    assert run_detector(restored.source, {"response_text": "omit case"}, bundle=bundle) is True


def test_failed_library_save_does_not_select_or_replace_current_version(runtime, monkeypatch):
    save_history(runtime)
    path = runtime.run.run_dir() / "detector_repairs.json"
    before = path.read_bytes()
    saved_library = runtime.working_library_path().read_bytes()

    def fail(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runtime, "persist_library", fail)
    result = restore(runtime)
    assert not result["ok"]
    assert runtime.library.get("scope").to_dict() == function(narrow=True)
    assert runtime.working_library_path().read_bytes() == saved_library
    assert path.read_bytes() == before


@pytest.mark.parametrize("present", [True, False])
def test_failed_selection_ledger_commit_restores_persisted_entry_and_history(
    runtime, monkeypatch, present,
):
    save_history(runtime)
    if not present:
        runtime.library.remove("scope")
    runtime.persist_library()
    before = deepcopy(runtime.library)
    original = runtime.library.get("scope")
    path = runtime.run.run_dir() / "detector_repairs.json"
    ledger_before = path.read_bytes()
    replace = workflow.os.replace

    def fail_ledger_replace(source, target):
        if str(target) == str(path):
            assert FunctionLibrary.load(runtime.working_library_path()).get("scope") is not None
            raise OSError("selection ledger unavailable")
        return replace(source, target)

    monkeypatch.setattr(workflow.os, "replace", fail_ledger_replace)
    result = restore(runtime)

    assert not result["ok"] and "selection ledger unavailable" in result["error"]
    assert runtime.library.get("scope") is original
    assert runtime.library == before
    assert FunctionLibrary.load(runtime.working_library_path()) == before
    assert path.read_bytes() == ledger_before


def test_restore_uses_locked_repair_snapshot_and_preserves_exposure(runtime, monkeypatch):
    """Restoration must not reacquire the repair file lock during persistence."""
    save_history(runtime)

    def nested_read(*args, **kwargs):
        pytest.fail("Version restoration reopened the already-locked repair ledger")

    monkeypatch.setattr(workflow, "list_repairs", nested_read)
    result = restore(runtime)
    assert result["ok"], result
    history = runtime.library.review_history
    assert {"reviewed", "old-holdout"} <= set(
        history["detectors"]["scope"]["reviewed_item_ids"]
    )
    persisted = FunctionLibrary.load(runtime.working_library_path())
    assert persisted.review_history == history


def test_concurrent_score_update_during_compilation_is_not_overwritten(runtime, monkeypatch):
    save_history(runtime)
    path = runtime.run.run_dir() / "detector_repairs.json"
    before = path.read_bytes()
    original = tools_repair_versions.compile_detector

    def concurrent_update(*args, **kwargs):
        runtime.library.get("scope").scores["n_unresolved"] = 12
        return original(*args, **kwargs)

    monkeypatch.setattr(tools_repair_versions, "compile_detector", concurrent_update)
    result = restore(runtime)
    assert not result["ok"]
    assert runtime.library.get("scope").scores["n_unresolved"] == 12
    assert path.read_bytes() == before


def test_corrupt_archive_identifier_is_not_selectable(runtime):
    with workflow._ledger(runtime.run.run_dir(), write=True) as data:
        data["versions"] = {"scope": {"wrong-version": function()}}
    result = tools_repair_versions.list_function_versions("scope")
    assert not result["ok"]
    assert "does not match" in result["error"]


def test_selected_version_can_be_resubmitted_unchanged_but_not_rewritten(runtime):
    save_history(runtime)
    assert restore(runtime)["ok"]
    old = function()
    args = {key: old[key] for key in ("name", "source", "description", "mechanism", "kind")}
    result = submit_function(**args, auto_evaluate=False)
    assert result["ok"], result
    assert runtime.library.get("scope").scores == old["scores"]
    result = submit_function(**{**args, "source": args["source"] + "\n# new edit"},
                             auto_evaluate=False)
    assert not result["ok"]
    assert "repair plan" in result["error"].lower()
    assert runtime.library.get("scope").source == old["source"]
    assert workflow.get_repair(runtime.run.run_dir(), "scope")["attempts_used"] == 3


def test_first_repair_after_restoration_excludes_all_archived_reviews(runtime):
    old = function()
    old["scores"]["behavior_verdicts"] = [{"item_id": "old-review", "verdict": "uncertain"}]
    current = runtime.library.get("scope")
    current.scores["behavior_verdicts"] = [{"item_id": "later-review", "verdict": "uncertain"}]
    archive_scored_version(runtime.run.run_dir(), old)
    result = restore(runtime, old)
    assert result["ok"], result
    workflow.plan_repair(
        runtime.run.run_dir(), old,
        {"claim_hash": old["scores"]["claim_hash"], "findings": [],
         "diagnostic_item_ids": ["old-review"], "allowed_actions": ["gather_evidence"]},
        action="gather_evidence", rationale="Inspect the unresolved evidence.",
        finding_ids=[], evidence_ids=[],
    )
    context = workflow.validation_context(runtime.run.run_dir(), old)
    # MR57 also preserves exposure from the initial persisted library version.
    assert set(context["exclusion_item_ids"]) == {"old-review", "later-review", "reviewed"}


# Version comparison and recommendations


CONFIG = Config(behavior_strictness="causal")


def _graded(variant, *, coverage=.08, precision=.9, adherence=1., tasks=8):
    fn = FeatureFunction(
        name="scope", source=f"def detect(item): return True  # {variant}",
        description="An omitted required branch remains unresolved",
        mechanism="An omitted required branch prevents the requested operation",
    )
    fn.scores = {
        "evaluated": True, "n_items": 1000., "n_scored": 1000,
        "n_flagged": float(tasks), "a": float(tasks), "b": 0.,
        "c": 100. - tasks, "d": 900., "recall": coverage,
        "ppv_at_pi0": precision, "behavior_purity": adherence,
        "sampled": False, "evaluation_scope_complete": True,
        "evaluation_complete": True, "n_unresolved": 0, "errors": 0,
    }
    attach_task_support(fn, task_n=tasks)
    fn.scores.update(
        behavior_n_judged=tasks, behavior_n_completed=tasks,
        behavior_n_yes=tasks if adherence == 1 else tasks - 1,
        behavior_n_no=0 if adherence == 1 else 1,
        judge_model="review-model", judge_harness="test",
        task_recall_scope="full_selected_corpus",
        task_recall_claim_hash=fn.scores["evaluation_claim_hash"],
    )
    return fn.to_dict()


def _summary(fn, current, *, config=CONFIG, target="Outstanding"):
    return version_summary(workflow._version(fn), fn, workflow._version(current),
                           config=config, run_id="quality-test", target_grade=target)


def _compare(before, after, *, config=CONFIG):
    return compare_function_versions(before, after, config=config, run_id="quality-test",
                                     target_grade="Outstanding")


def _recommend(before, after):
    return version_recommendation(
        "scope", [_summary(before, after), _summary(after, after)], [_compare(before, after)],
        workflow._version(after),
    )


def test_summary_shows_current_grade_normalized_metrics_support_scope_and_active_gap():
    before, current = _graded("old"), _graded("new", coverage=.06, tasks=6)
    current["scores"]["precision"] = .1
    result = _summary(current, current)
    assert result["grade"]["code"] == "GOOD"
    assert result["quality_metrics"] == {
        "recall": .06, "ppv_at_pi0": .9, "behavior_purity": 1.,
        "matched_failed_tasks": 6,
    }
    assert result["task_support"]["observed"] == 6
    assert result["task_support"]["basis"] == "observed"
    assert result["evaluation"]["scope"] == "full_selected_corpus"
    assert result["evaluation"]["current"] and result["evaluation"]["complete"]
    assert result["review"]["current_claim"]
    assert result["grade_gap"]["target_grade"] == "OUTSTANDING"
    assert result["grade_gap"]["bottlenecks"] == ["recall"]
    assert result["evaluation"]["cohort_fingerprint"] == _summary(before, current)[
        "evaluation"]["cohort_fingerprint"]
    assert "source" not in result


def test_same_good_grade_with_lower_coverage_recommends_exact_preserved_version():
    before, after = _graded("old"), _graded("new", coverage=.06, tasks=6)
    original = deepcopy((before, after))
    comparison = _compare(before, after)
    assert comparison["comparable"]
    assert comparison["claim_grade"] == {"before": "GOOD", "after": "GOOD",
                                       "direction": "unchanged"}
    assert comparison["metric_changes"]["recall"]["delta"] == pytest.approx(-.02)
    assert comparison["metric_changes"]["recall"]["direction"] == "regression"
    assert comparison["metric_changes"]["matched_failed_tasks"]["delta"] == -2
    assert comparison["coverage_evidence"]["n_removed_matches"] == 2
    recommendation = _recommend(before, after)
    assert recommendation["action"] == "restore"
    assert recommendation["version_hash"] == workflow._version(before)
    assert recommendation["tool"]["arguments"]["expected_current_version_hash"] == (
        workflow._version(after))
    assert recommendation["tool"]["reason_required"]
    assert recommendation["automatic_restore"] is False
    assert (before, after) == original


@pytest.mark.parametrize("before,after,action", [
    ({"coverage": .15}, {"coverage": .08, "precision": .95}, "restore"),
    ({}, {"coverage": .06, "precision": .95}, "inspect"),
    ({"coverage": .06, "tasks": 6}, {}, "keep_current"),
    # A weak prior is not a retention-threshold exception.
    ({"precision": .3}, {"coverage": .06, "precision": .2}, "inspect"),
], ids=["grade_regression", "same_grade_precision_coverage_tradeoff", "improved_current",
        "weak_prior"])
def test_recommendation_inspects_tradeoffs_and_keeps_improvements(before, after, action):
    assert _recommend(_graded("old", **before), _graded("new", **after))["action"] == action


@pytest.mark.parametrize("changes,same_cohort", [
    ({"task_recall_corpus_fingerprint": None}, False), ({"errors": 1}, True),
    # An identical corpus fingerprint cannot mask inconsistent full-scan counts.
    ({"n_items": 999., "n_scored": 999, "d": 899.}, True),
], ids=["unidentified_corpus", "errors", "inconsistent_counts"])
def test_same_counts_cannot_establish_comparable_current_evidence(changes, same_cohort):
    before, after = _graded("old"), _graded("new", coverage=.06)
    after["scores"].update(changes)
    # A common projected breadth panel cannot rescue association comparability.
    for fn in (before, after):
        fn["scores"]["search_panel"] = {"cohort_fingerprint": "same-panel"}
    result = _compare(before, after)
    assert result["same_association_cohort"] is same_cohort
    assert not result["comparable"]
    assert result["metric_changes"]["recall"]["delta"] is None
    assert result["metric_changes"]["recall"]["direction"] == "unavailable"
    assert result["claim_grade"]["direction"] == "unavailable"
    assert _recommend(before, after)["action"] == "inspect"


def _sample(fn, *, shift=0):
    fn["scores"].update(
        sampled=True, n_scored=10, sample_n=10, sample_frame_n=1000,
        evaluation_item_ids=[f"sample-{i + shift}" for i in range(10)],
        sample_strata=2, sample_allocation_version="fixed-v1",
        task_recall_scope="observed_lower_bound",
    )


def test_same_identified_sample_can_compare_provisionally_without_counting_extra_probes():
    before, after = _graded("old"), _graded("new", coverage=.06, tasks=6)
    for fn in (before, after):
        _sample(fn)
    result = _compare(before, after)
    assert result["comparable"] and result["provisional"]
    assert result["metric_changes"]["recall"]["delta"] == pytest.approx(-.02)
    assert result["metric_changes"]["matched_failed_tasks"]["delta"] is None
    assert _recommend(before, after)["action"] == "restore"


def test_different_sample_with_equal_size_cannot_recommend_rollback():
    before, after = _graded("old"), _graded("new", coverage=.06)
    _sample(before)
    _sample(after, shift=10)
    assert not _compare(before, after)["comparable"]
    assert _recommend(before, after)["action"] == "inspect"


def test_grade_drop_from_a_different_judge_cohort_alone_cannot_recommend_restoring():
    before, after = _graded("old"), _graded("new", adherence=.8)
    after["scores"]["behavior_verdicts"][0]["item_id"] = "different-reviewed-item"
    result = _compare(before, after)
    assert result["claim_grade"]["direction"] == "regression"
    # Each cohort's purity stays visible, but no numeric delta is claimed across cohorts.
    assert result["metric_changes"]["behavior_purity"]["before"] == 1
    assert result["metric_changes"]["behavior_purity"]["after"] == .8
    assert result["metric_changes"]["behavior_purity"]["delta"] is None
    assert not result["judge_comparison"]["same_sample_and_settings"]
    assert not result["judge_comparison"]["causal_improvement_established"]
    assert _recommend(before, after)["action"] == "inspect"


def test_same_judge_cohort_and_settings_allows_only_descriptive_adherence_delta():
    before, after = _graded("old"), _graded("new", adherence=.9)
    result = _compare(before, after)
    assert result["judge_comparison"]["same_sample_and_settings"]
    assert result["metric_changes"]["behavior_purity"]["delta"] == pytest.approx(-.1)
    assert not result["judge_comparison"]["causal_improvement_established"]


@pytest.mark.parametrize("damage", ["claim", "contract_source", "missing_contract"])
def test_stale_or_borrowed_prior_review_cannot_recommend_restoration(damage):
    before, after = _graded("old"), _graded("new", coverage=.06)
    scores = before["scores"]
    if damage == "claim":
        scores["claim_hash"] = after["scores"]["claim_hash"]
    elif damage == "contract_source":
        scores["claim_contract"]["source_sha256"] = "0" * 64
    else:
        scores.pop("claim_contract")
    assert _summary(before, after)["grade"]["code"] not in {"GOOD", "EXCELLENT", "OUTSTANDING"}
    assert _recommend(before, after)["action"] != "restore"


def test_explicit_active_policy_is_applied_to_preserved_claims():
    before, after = _graded("old"), _graded("new", coverage=.06)
    result = _summary(before, after, config=Config(behavior_strictness="behavioral"))
    assert result["grade"]["code"] == "UNSCORED"
    assert result["review"]["status"] == "stale"


@pytest.mark.parametrize("scored", [True, False])
def test_listing_integrates_comparison_without_mutating_library_or_archive(
    tmp_path, monkeypatch, scored,
):
    before, after = _graded("old"), _graded("new", coverage=.06, tasks=6)
    if not scored:
        after["scores"] = {}  # Displayed, but never compared as a loss.
    rt = make_runtime(tmp_path, monkeypatch, run_id="quality-test", config=CONFIG,
                      functions=[after])
    monkeypatch.setattr(tools_repair_versions, "agent_target_grade_for_runtime",
                        lambda runtime: "OUTSTANDING")
    archive_scored_version(rt.run.run_dir(), before)
    ledger = rt.run.run_dir() / "detector_repairs.json"
    saved = ledger.read_bytes()
    response = tools_repair_versions.list_function_versions("scope")
    assert response["ok"] and len(response["versions"]) == 2
    if scored:
        assert response["recommendation"]["action"] == "restore"
        assert response["versions"][0]["grade_gap"]["target_grade"] == "OUTSTANDING"
    else:
        assert not response["comparisons"][0]["comparable"]
        assert response["recommendation"]["action"] == "inspect"
    assert rt.library.get("scope").to_dict() == after
    assert ledger.read_bytes() == saved


# Historical lessons


def _evaluated(*statuses):
    fn = reviewed_function(*statuses)
    fn["scores"].update(
        evaluated=True, evaluation_claim_hash=claim_hash(**workflow._claim(fn)),
        evaluation_run_id="quality-test", errors=0,
    )
    return FeatureFunction.from_dict(fn).to_dict()


def _ledger(*functions):
    return {
        "schema_version": workflow.SCHEMA_VERSION, "repairs": {},
        "versions": {"detector": {
            workflow._version(fn): deepcopy(fn) for fn in functions
        }},
    }


def _lessons_runtime(tmp_path, data, current):
    path = tmp_path / "detector_repairs.json"
    path.write_text(json.dumps(data))
    return SimpleNamespace(
        run=SimpleNamespace(run_dir=lambda: tmp_path, run_id="quality-test"),
        config=Config(), library=FunctionLibrary(
            functions=[FeatureFunction.from_dict(current)] if current else []),
    )


def _good(coverage, variant):
    matched = int(100 * coverage)
    fn = FeatureFunction(
        name="detector", source=f"def detect(item): return True  # {variant}",
        description="Observable unfinished work",
        mechanism="Unfinished work prevents completion",
        scores={
            "evaluated": True, "n_items": 200, "n_scored": 200,
            "sampled": False, "evaluation_scope_complete": True,
            "a": matched, "b": 1, "c": 100 - matched, "d": 99,
            "n_flagged": matched + 1, "recall": coverage,
            "ppv_at_pi0": matched / (matched + 1), "behavior_purity": 1.,
        },
    )
    attach_task_support(fn)
    fn.scores["evaluation_item_ids"] = [f"item-{index}" for index in range(200)]
    return fn.to_dict()


def test_full_findings_keep_counterexamples_gaps_and_errors_separate():
    fn = _evaluated("contradicted", "unknown", "error")
    fn["scores"].update(errors=1, error_examples=[{"item_id": "execution", "error": "KeyError: x"}])
    data = _ledger(fn)
    original = deepcopy(data)
    records = saved_version_findings(data, "detector")
    assert {row["category"] for row in records} == {
        "counterexample", "missing_fact", "evaluation_error", "review_processing_error",
    }
    assert {row["item_id"] for row in records if row["category"] == "counterexample"} == {
        "item-000",
    }
    assert {row["item_id"] for row in records if row["category"] == "missing_fact"} == {
        "item-001",
    }
    counterexample = next(row for row in records if row["category"] == "counterexample")
    assert counterexample["counterevidence"][0]["quote"] == "An authoritative source was supplied."
    assert counterexample["version_hash"] == workflow._version(fn)
    assert counterexample["claim_hash"] == fn["scores"]["claim_hash"]
    assert counterexample["historical"] is True
    assert counterexample["history_refs"][0].startswith("#/versions/detector/")
    assert data == original


def test_diagnostic_findings_keep_scope_and_do_not_change_adherence(tmp_path):
    fn = _evaluated("contradicted")
    diagnostic = fn["scores"]
    fn["scores"] = {
        "evaluated": True, "evaluation_claim_hash": diagnostic["claim_hash"],
        "evaluation_run_id": "quality-test", "behavior_purity": 1,
        "behavior_n_no": 0,
        "diagnostic_review": {**diagnostic, "verdicts": diagnostic["behavior_verdicts"],
                              "review_mode": "diagnostic"},
    }
    data = _ledger(fn)
    rt = _lessons_runtime(tmp_path, data, fn)
    before = rt.library.get("detector").to_dict()
    result = version_lessons(rt, "detector")
    assert result["lessons"][0]["finding_counts"]["counterexample"] > 0
    assert all(row["evidence_scope"] == "diagnostic_review"
               for row in result["lessons"][0]["findings"])
    assert rt.library.get("detector").to_dict() == before
    assert result["lessons"][0]["affects_current_grade"] is False


@pytest.mark.parametrize("mutation", ["provisional", "wrong_claim"])
def test_provisional_or_mismatched_reviews_never_become_counterexamples(mutation):
    fn = _evaluated("contradicted")
    if mutation == "provisional":
        fn["scores"]["behavior_verdicts"][0]["processing_status"] = "pending"
    else:
        fn["scores"]["claim_hash"] = "other"
    assert saved_version_findings(_ledger(fn), "detector") == []


def test_same_good_metric_loss_and_concrete_attempt_edit_are_retained(tmp_path):
    before, after = _good(.08, "baseline"), _good(.05, "candidate")
    plan = {
        "plan_id": "broaden-1", "action": "tighten_detector", "phase": "deferred",
        "rationale": "Exclude the concrete recovered cases.",
        "deferral_reason": "Lost useful coverage without improved precision.",
        "finding_ids": ["finding-a"], "evidence_ids": ["recovered-case"],
        "before": before, "before_version_hash": workflow._version(before),
        "after": after, "after_version_hash": workflow._version(after),
    }
    data = _ledger(before, after)
    data["repairs"]["detector"] = {"history": [plan], "reviewed_item_ids": ["recovered-case"]}
    rt = _lessons_runtime(tmp_path, data, after)
    original_ledger = (tmp_path / "detector_repairs.json").read_bytes()
    result = version_lessons(rt, "detector")
    old = next(row for row in result["lessons"]
               if row["version_hash"] == workflow._version(before))
    comparison = old["comparison_to_current"]
    assert comparison["claim_grade"] == {
        "before": "GOOD", "after": "GOOD", "direction": "unchanged",
    }
    assert comparison["metric_changes"]["recall"]["direction"] == "regression"
    assert comparison["metric_changes"]["recall"]["delta"] == pytest.approx(-.03)
    new = next(row for row in result["lessons"]
               if row["version_hash"] == workflow._version(after))
    attempt = new["attempts"][0]
    assert attempt["rationale"] == plan["rationale"]
    assert attempt["evidence_ids"] == ["recovered-case"]
    assert "candidate" in attempt["edits"]["source_diff_excerpt"]
    assert attempt["comparison"]["metric_changes"]["recall"]["direction"] == "regression"
    assert (tmp_path / "detector_repairs.json").read_bytes() == original_ledger


def test_earlier_same_version_plan_feedback_survives_newer_clean_snapshot(tmp_path):
    earlier = _evaluated(*(["contradicted"] * 8))
    latest = _evaluated("supported")
    assert workflow._version(earlier) == workflow._version(latest)
    feedback = build_repair_feedback(earlier)
    data = _ledger(latest)
    data["repairs"]["detector"] = {"history": [{
        "plan_id": "old-feedback", "before": latest,
        "feedback": feedback, "judge_policy_hash": earlier["scores"]["judge_policy_hash"],
    }]}
    records = saved_version_findings(data, "detector")
    assert any(row["category"] == "counterexample" for row in records)
    assert {row["item_id"] for row in records} == {f"item-{index:03}" for index in range(8)}
    assert all("/feedback/findings/" in row["history_refs"][0] for row in records)
    result = version_lessons(_lessons_runtime(tmp_path, data, latest), "detector")
    lesson = result["lessons"][0]
    assert lesson["applicability"] == "same_version_requires_freshness_check"
    assert lesson["finding_counts"]["counterexample"] > 0


def test_bounded_deterministic_summary_keeps_untruncated_full_findings(tmp_path):
    functions = []
    for index in range(11):
        fn = _good(.04, index)
        fn["scores"].update(errors=1, error_examples=[
            {"item_id": f"error-{index}-{item}", "error": f"Missing field {item}"}
            for item in range(5)
        ])
        functions.append(fn)
    data = _ledger(*functions)
    rt = _lessons_runtime(tmp_path, data, functions[-1])
    result = version_lessons(rt, "detector")
    assert len(result["lessons"]) == 8
    assert result["omitted_lessons"] == 3
    assert all(row["omitted_findings"] == 3 for row in result["lessons"])
    assert len(saved_version_findings(data, "detector")) == 55
    reversed_data = deepcopy(data)
    reversed_data["versions"]["detector"] = dict(reversed(
        list(reversed_data["versions"]["detector"].items())))
    other = version_lessons(_lessons_runtime(tmp_path, reversed_data, functions[-1]), "detector")
    assert result == other
    assert not (tmp_path / "version_lessons.json").exists()


def _get_repair(tmp_path, monkeypatch, data, current):
    rt = _lessons_runtime(tmp_path, data, current)
    monkeypatch.setattr(tools_repair, "get_runtime", lambda: rt)
    monkeypatch.setattr(tools_repair, "review_next_action", lambda *args: None)
    return tools_repair.get_function_repair("detector")


def test_unavailable_history_summary_does_not_hide_current_repair_feedback(tmp_path, monkeypatch):
    def broken(*args):
        raise ValueError("invalid archived counts")

    monkeypatch.setattr(tools_repair, "version_lessons", broken)
    result = _get_repair(tmp_path, monkeypatch, _ledger(), _evaluated("contradicted"))
    assert result["ok"] and result["retained_lessons"]["available"] is False
    assert any(row["status"] == "contradicted" for row in result["feedback"]["findings"])


# Repair coverage tradeoffs


def _localised(name, labels, *, bucket="positive", reviewed=None, precision=.8, recall=.4,
              purity=.8, strategy="balanced", source=None):
    source = source or f"def detect(item): return True  # {name}"
    claim = {"source": source, "description": "Destroys required input.",
             "mechanism": "Overwrites the only copy before preserving it."}
    contract = build_claim_contract(**claim)
    rows = []
    for item_id, verdict in (reviewed or {}).items():
        row = {
            "item_id": item_id, "snapshot_id": "snapshot-" + item_id,
            "claim_contract_id": contract["contract_id"], "verdict": verdict,
            "processing_status": "completed",
            "claim_assessments": [{
                "requirement_id": requirement["id"],
                "status": "supported" if verdict == "yes" else "contradicted",
                "rationale": "Original evidence supports this assessment.",
                "citations": [{"citation_id": "cite-1"}], "counterevidence": [],
            } for requirement in contract["requirements"]],
        }
        if verdict == "yes":
            row.update(initial_review=deepcopy(row), verification_status="confirmed")
        rows.append(row)
    count = sum(value not in (False, "False", "ok")
                and (bucket == "positive" or value == bucket) for value in labels.values())
    return {
        "name": name, **claim,
        "failure_localisations": {
            item_id: {"item_id": item_id, "assigned_label": label,
                      "source_sha256": text_hash(source)} for item_id, label in labels.items()
        },
        "scores": {
            "evaluated": True, "evaluation_claim_hash": claim_hash(**claim),
            "evaluation_run_id": "same-run", "evaluation_scope_complete": True,
            "sampled": False, "n_items": 20, "n_flagged": count,
            "a": count, "b": 0, "c": 10 - count, "d": 10, "bucket": bucket,
            "ppv_at_pi0": precision, "recall": recall, "behavior_purity": purity,
            "judge_contract_version": JUDGE_CONTRACT_VERSION,
            "judge_policy_hash": judge_policy_hash(), "judge_objective_strategy": strategy,
            "claim_hash": claim_hash(**claim), "claim_contract": contract,
            "behavior_verdicts": rows,
        },
    }


def _coverage(before, after, **kwargs):
    return compare_repair_coverage(before, after, config=CONFIG, **kwargs)


def _pair():
    labels = dict.fromkeys(["confirmed", "rejected", "unreviewed", "keep"], True)
    before = _localised("before", labels,
                       reviewed={"confirmed": "yes", "rejected": "no", "keep": "yes"})
    after = _localised("after", {"keep": True}, precision=1., recall=.1, purity=1.)
    return before, after


def test_repair_separates_removed_confirmations_rejections_and_unreviewed_matches():
    before, after = _pair()
    unchanged = deepcopy((before, after))
    result = _coverage(before, after)
    assert result["comparison_complete"] is True
    assert result["n_removed_matches"] == 3
    assert result["evidence"] == {
        "previously_confirmed_removed": ["confirmed"],
        "previously_rejected_removed": ["rejected"],
        "unresolved_or_unreviewed_removed": ["unreviewed"],
        "previously_confirmed_retained": ["keep"], "removal_unknown": [],
    }
    assert result["requires_coverage_inspection"] is True
    assert result["automatic_promotion"] is False
    assert result["metrics"]["ppv_at_pi0"]["delta"] > 0
    assert result["metrics"]["recall"]["delta"] < 0
    assert result["objective"]["delta"] is None
    assert result["judge_comparison"]["same_sample_and_settings"] is False
    assert result["judge_comparison"]["causal_improvement_established"] is False
    assert (before, after) == unchanged


def test_real_full_evaluations_preserve_float_count_repair_diagnostics(tmp_path, monkeypatch):
    item_ids = ["confirmed", "rejected", "unreviewed", "keep"] + [
        f"item-{i}" for i in range(4, 20)
    ]

    class RepairDataset(Dataset):
        def estimate_matching_rows(self):
            return len(item_ids)

        def iter_batches(self, *, max_items=None, **kwargs):
            yield [{"_item_id": item_id, "response_text": item_id, "is_correct": index >= 10,
                    "model_id": "m", "benchmark": "b", "problem_id": item_id}
                   for index, item_id in enumerate(item_ids)][:max_items]

    before = FeatureFunction.from_dict(_localised(
        "before", dict.fromkeys(item_ids[:4], True),
        source=("def detect(item): return item['response_text'] in "
                "('confirmed', 'rejected', 'unreviewed', 'keep')"),
        reviewed={"confirmed": "yes", "rejected": "no", "keep": "yes"},
    ))
    after = FeatureFunction.from_dict(_localised(
        "after", {"keep": True}, source="def detect(item): return item['response_text'] == 'keep'",
    ))
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    rt = make_runtime(tmp_path, monkeypatch, run_id="repair-census",
                      config=Config(behavior_strictness="causal"), functions=[before, after],
                      dataset=RepairDataset(parquet_paths=[]))
    items = {row["_item_id"]: row for batch in rt.dataset.iter_batches() for row in batch}
    for verdict in before.scores["behavior_verdicts"]:
        item_id = verdict["item_id"]
        snapshot = make_snapshot(
            items[item_id], item_id=item_id, source=before.source,
            failure_localisation=before.failure_localisations[item_id],
            claim_contract=before.scores["claim_contract"],
        )["snapshot_id"]
        verdict["snapshot_id"] = snapshot
        if "initial_review" in verdict:
            verdict["initial_review"]["snapshot_id"] = snapshot
    assert tools_eval.evaluate_functions(names=["before", "after"], mode="full")["ok"]
    assert before.scores["n_flagged"] == 4.0
    assert after.scores["n_flagged"] == 1.0
    assert isinstance(before.scores["n_items"], float)
    result = compare_repair_coverage(
        before.to_dict(), after.to_dict(), config=rt.config,
        failed_item_ids=item_ids[:10], run_id=rt.run.run_id,
    )
    assert result["comparison_complete"]
    assert result["n_removed_matches"] == 3
    assert result["evidence"]["previously_confirmed_removed"] == ["confirmed"]
    assert result["requires_coverage_inspection"]
    assert result["metrics"]["recall"]["delta"] == pytest.approx(-.3)


@pytest.mark.parametrize("field,value", [
    ("sampled", True), ("evaluation_claim_hash", "stale"), ("n_items", 19),
    ("n_items", float("nan")),
], ids=["sampled", "stale", "different_corpus", "non_census_counts"])
def test_partial_stale_or_different_corpus_cannot_establish_removed_matches(field, value):
    before, after = _pair()
    after["scores"][field] = value
    result = _coverage(before, after)
    assert result["comparison_complete"] is False
    assert result["n_removed_matches"] is None
    assert result["evidence"]["previously_confirmed_removed"] == []
    assert result["objective"]["delta"] is None
    assert all(metric["delta"] is None for metric in result["metrics"].values())


def test_missing_positive_record_is_not_a_negative_prediction():
    before, after = _pair()
    after["failure_localisations"].clear()
    result = _coverage(before, after)
    assert result["comparison_complete"] is False
    assert result["n_removed_matches"] is None
    assert result["evidence"]["removal_unknown"] == ["confirmed", "keep", "rejected", "unreviewed"]


def test_stale_or_provisional_judgment_cannot_label_removed_matches_confirmed():
    before, after = _pair()
    before["scores"]["judge_policy_hash"] = "old-policy"
    result = _coverage(before, after)
    assert result["comparison_complete"] is True
    assert result["evidence"]["previously_confirmed_removed"] == []
    assert result["evidence"]["previously_rejected_removed"] == []
    assert result["metrics"]["behavior_purity"]["delta"] is None
    assert result["objective"]["delta"] is None
    before, after = _pair()
    before["scores"]["behavior_verdicts"][0].pop("initial_review")
    result = _coverage(before, after)
    assert result["evidence"]["previously_confirmed_removed"] == []
    assert "confirmed" in result["evidence"]["unresolved_or_unreviewed_removed"]


def test_selected_categorical_bucket_excludes_other_labels_and_negative_matches():
    bucket = "input_loss"
    before = _localised("before", {"first": bucket, "second": bucket,
                                   "other": "bad_call", "negative": "ok"}, bucket=bucket)
    after = _localised("after", {"first": "bad_call", "second": bucket,
                                 "other": "bad_call", "negative": "ok"}, bucket=bucket)
    for fn, count in ((before, 2), (after, 1)):
        fn["scores"].update(bucket_mode="exact", n_flagged=count, a=count, c=10 - count)
    result = _coverage(before, after)
    assert result["comparison_complete"] is True
    assert result["n_baseline_matches"] == 2
    assert result["n_retained_matches"] == 1
    assert result["n_removed_matches"] == 1
    assert result["evidence"]["unresolved_or_unreviewed_removed"] == ["first"]


def test_objective_snapshot_survives_later_preset_changes_without_rescoring_reviews():
    before, after = _pair()
    frozen = snapshot_objectives(Config(
        objective_axes=["ppv_at_pi0", "recall", "behavior_purity"],
        objective_weights=[1, 1, 1], objective_rank="geometric_mean",
    ))
    result = compare_repair_coverage(
        before, after, config=apply_preset(Config(behavior_strictness="causal"), "precision"),
        objective_settings=frozen,
    )
    assert result["objective"]["settings_frozen"] is True
    assert list(result["objective"]["weights"].values()) == pytest.approx([1 / 3] * 3)
    assert result["objective"]["before"] == pytest.approx((.8 * .4 * .8) ** (1 / 3))
    assert result["objective"]["after"] == pytest.approx(.1 ** (1 / 3))
    assert result["automatic_promotion"] is False


def _coverage_pair():
    """The observed 68 -> 47 failure-match shrinkage, including passes and churn."""
    before = _localised("before", {**dict.fromkeys((f"f{i}" for i in range(68)), True),
                                  **dict.fromkeys((f"p{i}" for i in range(5)), True)},
                       precision=.7445, recall=.68, reviewed={"f0": "yes", "p0": "no"})
    after = _localised("after", {**dict.fromkeys((f"f{i}" for i in range(25, 72)), True),
                                "p0": True}, precision=.9097, recall=.47,
                      reviewed={"f25": "yes"})
    for fn, tp, fp in ((before, 68, 5), (after, 47, 1)):
        fn["scores"].update(n_items=150., n_scored=150, n_flagged=float(tp + fp),
                            a=float(tp), b=float(fp), c=float(100 - tp), d=float(50 - fp),
                            judge_model="model-a", judge_harness="codex", judge_effort="medium")
    return before, after, {f"f{i}" for i in range(100)}


def test_repair_exposes_68_to_47_loss_on_fixed_failure_denominator_and_bounded_examples():
    before, after, failures = _coverage_pair()
    result = _coverage(before, after, failed_item_ids=failures, run_id="same-run")
    coverage = result["failure_coverage"]
    assert result["comparison_complete"] is True  # Production counts are integral floats.
    assert coverage["exact"] is True
    assert (coverage["before"], coverage["after"], coverage["n_failed"]) == (68, 47, 100)
    assert (coverage["n_retained"], coverage["n_no_longer_observed"],
            coverage["n_newly_observed"]) == (43, 25, 4)
    assert coverage["observed_recall_delta"] == pytest.approx(-.21)
    assert len(coverage["evidence"]["no_longer_observed"]) == 10
    assert result["n_removed_matches"] == 29  # Includes four benchmark passes.
    assert result["metrics"]["ppv_at_pi0"]["delta"] > 0
    assert result["metrics"]["behavior_purity"]["delta"] is None
    assert result["objective"]["delta"] is None
    assert result["requires_coverage_inspection"] is True
    assert result["automatic_promotion"] is False


def test_unresolved_rows_keep_fixed_denominator_observations_without_proven_removed_matches():
    before, after, failures = _coverage_pair()
    after["scores"].update(n_unresolved=20, errors=12, n_insufficient=8, n_items=130.,
                           n_scored=130, c=33., evaluation_complete=False,
                           assessment_coverage=130 / 150, recall=47 / 80)
    result = _coverage(before, after, failed_item_ids=failures)
    assert result["comparison_complete"] is False
    assert result["n_removed_matches"] is None
    assert result["failure_coverage"]["n_no_longer_observed"] == 25
    assert result["failure_coverage"]["after_recall_lower_bound"] == .47
    assert result["failure_coverage"]["exact"] is False
    assert result["metrics"]["recall"]["after"] == 47 / 80
    assert result["metrics"]["recall"]["delta"] is None
    assert result["uncertainty"]["after"]["errors"] == 12
    assert result["requires_coverage_inspection"] is True


def test_fixed_failure_coverage_does_not_reuse_other_run_evaluation():
    before, after, failures = _coverage_pair()
    after["scores"]["evaluation_run_id"] = "another-run"
    result = _coverage(before, after, failed_item_ids=failures, run_id="same-run")
    assert result["failure_coverage"]["after"] is None
    assert result["failure_coverage"]["n_no_longer_observed"] is None
    assert result["failure_coverage"]["observed_recall_delta"] is None


def test_same_judge_ids_with_changed_model_do_not_establish_purity_improvement():
    before = _localised("before", {"same": True}, reviewed={"same": "yes"})
    after = _localised("after", {"same": True}, reviewed={"same": "yes"}, purity=1.)
    for fn in (before, after):
        fn["scores"].update(judge_model="model-a", judge_harness="codex")
    result = _coverage(before, after)
    assert result["judge_comparison"]["same_sample_and_settings"] is True
    assert result["metrics"]["behavior_purity"]["delta"] == pytest.approx(.2)
    assert result["judge_comparison"]["causal_improvement_established"] is False
    after["scores"]["judge_model"] = "model-b"
    result = _coverage(before, after)
    assert result["judge_comparison"]["same_sample_and_settings"] is False
    assert result["metrics"]["behavior_purity"]["delta"] is None
    assert result["objective"]["delta"] is None


@pytest.mark.parametrize("same_corpus,scope", [
    (True, "full_selected_corpus"), (False, "full_selected_corpus"),
    (True, "observed_lower_bound"),
])
def test_task_breadth_repair_delta_requires_matching_corpus(same_corpus, scope):
    before = _localised("before", {"same": True}, reviewed={"same": "yes"})
    after = _localised("after", {"same": True}, reviewed={"same": "yes"})
    for fn, breadth in ((before, .1), (after, .3)):
        fn["scores"].update(
            judge_model="model-a", judge_harness="codex", task_recall=breadth,
            task_recall_scope=scope,
            task_recall_claim_hash=fn["scores"]["evaluation_claim_hash"],
            task_recall_corpus_fingerprint="corpus-a",
        )
    if not same_corpus:
        after["scores"]["task_recall_corpus_fingerprint"] = "corpus-b"
    result = _coverage(before, after)
    assert result["metrics"]["recall"]["delta"] == 0
    if same_corpus and scope == "full_selected_corpus":
        assert result["metrics"]["task_recall"]["delta"] == pytest.approx(.2)
        assert result["objective"]["delta"] > 0
    else:
        assert result["metrics"]["task_recall"]["delta"] is None
        assert result["objective"]["delta"] is None


def test_supplied_failure_cohort_must_match_scored_failure_counts_for_exact_comparison():
    before, after, failures = _coverage_pair()
    failures.remove("f0")
    failures.add("different-failed-item")  # Same count alone does not establish the cohort.
    result = _coverage(before, after, failed_item_ids=failures)
    assert result["comparison_complete"] is False
    assert result["n_removed_matches"] is None
    assert result["metrics"]["recall"]["delta"] is None
    assert result["failure_coverage"]["receipts_match_scored_failure_counts"] is False


def test_repair_tool_exposes_saved_baseline_with_current_dataset_denominator(tmp_path, monkeypatch):
    before, after, failures = _coverage_pair()
    fn = FeatureFunction.from_dict(after)
    frozen = snapshot_objectives(Config())
    dataset = SimpleNamespace(parquet_paths=[], iter_batches=lambda **kwargs: iter([
        [{"_item_id": item_id, "is_correct": False} for item_id in failures]
        + [{"_item_id": f"p{i}", "is_correct": True} for i in range(50)]
    ]))
    runtime = SimpleNamespace(
        config=apply_preset(Config(behavior_strictness="causal"), "precision"), dataset=dataset,
        library=SimpleNamespace(get=lambda name: fn),
        run=SimpleNamespace(run_id="same-run", run_dir=lambda: tmp_path),
    )
    monkeypatch.setattr(tools_repair, "get_runtime", lambda: runtime)
    monkeypatch.setattr(tools_repair.repair_workflow, "get_repair", lambda *args: {
        "latest": {"before": before, "objective_settings": frozen}, "history": [],
    })
    response = tools_repair.get_function_repair("after")
    assert response["ok"]
    comparison = response["coverage_comparison"]
    assert comparison["failure_coverage"]["n_failed"] == 100
    assert comparison["failure_coverage"]["n_no_longer_observed"] == 25
    assert comparison["objective"]["settings_frozen"] is True
    assert "before" not in response["workflow"]["latest"]  # Full source stays on disk.

    def unavailable(*args, **kwargs):
        raise OSError("Identity index unavailable")

    monkeypatch.setattr("autofa.metrics.library_coverage.failure_cohort", unavailable)
    incomplete = tools_repair.get_function_repair("after")
    assert incomplete["ok"]
    assert incomplete["feedback"] == response["feedback"]
    assert incomplete["workflow"] == response["workflow"]
    assert incomplete["coverage_comparison"]["comparison_complete"] is False
    assert "Identity index unavailable" in incomplete["coverage_comparison"]["reason"]
