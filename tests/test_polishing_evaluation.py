# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation controls guide post-census repairs that keep all issues and leave metrics intact."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.core.library import FeatureFunction
from autofa.core.sandbox import DetectorAbstained
from autofa.mcp import tools_eval
from autofa.metrics.task_support import task_support_assessment
from autofa.session import family_coverage, polishing_evaluation
from autofa.session.polishing_census import _claim, polishing_census_diagnostics
from autofa.session.polishing_evaluation import (
    current_polishing_evaluation,
    run_polishing_evaluation,
)
from autofa.session.polishing_generalization import (
    prepare_diagnostic_controls,
    retained_diagnostic_problems,
)
from autofa.session.validation_controls import (
    collect_full_validation_controls,
    polishing_controls_identity,
    validation_controls,
)
from autofa.util.atomic import atomic_write_json

from support.evaluation import (
    _LLM_SOURCE,
    _REGEX_SOURCE,
    CORPUS_N,
    MATCHED_N,
    PANEL_N,
    _FakeEvalDataset,
    _llm_flag,
    _record_calls,
    _runtime,
    write_phase,
)
from support.evaluation import polishing_runtime as polishing_runtime

# In the polishing corpus, matched (and likewise unmatched) attempts split evenly
# into passes and failures.
MATCHED_EACH, UNMATCHED_EACH = MATCHED_N // 2, (CORPUS_N - MATCHED_N) // 2


def forbid(*args, **kwargs):
    pytest.fail("This operation must reuse saved predictions instead of executing")


@pytest.fixture
def controls_runtime(tmp_path, monkeypatch):
    fn = _llm_flag(description="Failure behavior",
                   mechanism="The required action is left undone")
    rt = _runtime(tmp_path, monkeypatch, sample=12, workers=1, functions=[fn])
    rt.dataset = _FakeEvalDataset([
        {"model_id": "m", "benchmark": "bench", "problem_id": str(i // 2), "run_idx": i % 2,
         "is_correct": i % 5 != 0, "response_text": "BAD" if i % 3 == 0 else "fine",
         "_item_id": f"id-{i}"} for i in range(120)
    ])
    assert tools_eval.evaluate_round_functions()["ok"]
    return rt, fn


def test_controls_are_observed_task_diverse_and_read_only(controls_runtime, monkeypatch):
    rt, fn = controls_runtime
    original = deepcopy(fn.to_dict())

    def forbidden(*args, **kwargs):
        pytest.fail("Building controls must not read source traces or execute detectors")

    monkeypatch.setattr(rt.dataset, "iter_batches", forbidden)
    monkeypatch.setattr(tools_eval, "run_detector", forbidden)
    result = validation_controls(rt, fn, limit=8)
    assert result["available"] and 1 <= len(result["controls"]) <= 8
    assert set(result["item_ids"]) <= set(fn.scores["evaluation_item_ids"])
    assert result["cohort"]["n"] == len(fn.scores["evaluation_item_ids"])
    assert result["cohort"]["scope"] == "round_checkpoint"
    tasks = [row["task_id"] for row in result["controls"]]
    assert len(tasks) == len(set(tasks))
    categories = {kind for row in result["controls"] for kind in row["categories"]}
    assert {"flagged_success_contrast", "matched_failed_attempt",
            "unflagged_failed_attempt"} <= categories
    assert "expected_outputs" not in result["baseline_smoke"]["arguments"]
    assert fn.to_dict() == original


@pytest.mark.parametrize("change", ["claim", "incomplete", "identity"])
def test_controls_require_current_validation_provenance(controls_runtime, change):
    rt, fn = controls_runtime
    if change == "claim":
        fn.description += " Revised claim"
    elif change == "incomplete":
        fn.scores["round_validation"]["complete"] = False
    else:
        rt._family_failed_task_inventory = None
    result = validation_controls(rt, fn)
    assert result["available"] is False and result["reason"]
    assert result["item_ids"] == [] and "baseline_smoke" not in result


@pytest.fixture
def large_controls_runtime(tmp_path, monkeypatch):
    fn = _llm_flag(description="Failure behavior",
                   mechanism="The required action is left undone")
    rt = _runtime(tmp_path, monkeypatch, sample=200, workers=3, functions=[fn])
    rt.dataset = _FakeEvalDataset([
        {"model_id": "m", "benchmark": "bench", "problem_id": str(i), "run_idx": 0,
         "is_correct": i % 2 == 0, "response_text": "BAD", "_item_id": f"case-{i}"}
        for i in range(120)
    ])
    assert tools_eval.final_evaluate_functions()["ok"]
    return rt, fn


def test_controls_default_to_half_configured_sample_and_run_in_batches(large_controls_runtime):
    rt, fn = large_controls_runtime
    # Half the sample (rounded down, minimum one); census mode uses half the default of 200.
    # Building controls is read-only, so one scored runtime serves every setting.
    for sample, expected in [(81, 40), (0, 100), (1, 1)]:
        rt.config.llm_eval_sample = sample
        result = validation_controls(rt, fn)
        assert result["available"] and result["limit"] == expected
        assert len(result["controls"]) == expected
        assert len({row["task_id"] for row in result["controls"]}) == expected
        batches = result["baseline_smoke_batches"]
        assert result["baseline_smoke"] == batches[0]
        assert [item_id for batch in batches for item_id in batch["arguments"]["item_ids"]
                ] == result["item_ids"]
        assert all(1 <= len(batch["arguments"]["item_ids"]) <= 30 for batch in batches)
    # Baseline and revised controls run in two smoke batches without changing scores.
    rt.config.llm_eval_sample = 80
    result = validation_controls(rt, fn)
    original_scores = deepcopy(fn.scores)
    assert len(result["baseline_smoke_batches"]) == 2
    for batch in result["baseline_smoke_batches"]:
        baseline = tools_eval.smoke_test_feature(**batch["arguments"])
        revised = tools_eval.smoke_test_feature(
            source=fn.source.replace("'BAD'", "'MISSING'"),
            item_ids=batch["arguments"]["item_ids"],
        )
        assert baseline["ok"] and revised["ok"]
        assert baseline["diagnostic_scope"] == revised["diagnostic_scope"] == "author_controls"
    assert fn.scores == original_scores


@pytest.fixture
def cheap_repair(tmp_path, monkeypatch):
    fn = FeatureFunction(name="plain", description="Failure behavior", source=_REGEX_SOURCE,
                         kind="python", mechanism="The required action is left undone")
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[fn])
    rt.run.eval_limit = 40
    rt.dataset = _FakeEvalDataset([
        {"model_id": "m", "benchmark": "bench", "problem_id": str(i), "run_idx": 0,
         "is_correct": i % 2 == 0, "response_text": "BAD" if i < 40 else "fine",
         "_item_id": f"case-{i}"} for i in range(80)
    ])
    assert tools_eval.evaluate_functions(limit=40)["ok"]
    before = deepcopy(fn.scores)
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    controls = collect_full_validation_controls(rt, fn, previous_scores=before)
    assert controls["available"] and controls["complete"], controls
    atomic_write_json(rt.run.run_dir() / "round_validation.json", {"refinement": {
        "version": 1, "mode": "polishing", "run_id": rt.run.run_id, "round": rt.run.round,
        "status": "awaiting_response", "frozen_names": [fn.name], "target_grade": "GOOD",
        "controls": {fn.name: controls}, "polishing_id": f"{rt.run.run_id}:{rt.run.round}",
    }})
    return rt, fn, controls


def test_cheap_repair_scans_once_and_detects_new_passing_matches(cheap_repair, monkeypatch):
    rt, fn, controls = cheap_repair

    def forbidden(*args, **kwargs):
        pytest.fail("Cheap repairs must not execute a separate diagnostic subset")

    monkeypatch.setattr(polishing_evaluation, "run_polishing_evaluation", forbidden)
    # An unchanged verified baseline reconstructs diagnostics without execution.
    with monkeypatch.context() as patch:
        patch.setattr(tools_eval, "run_detector", forbidden)
        unchanged = tools_eval.evaluate_functions(names=[fn.name])
    assert unchanged["ok"] and unchanged["reused_names"] == [fn.name]
    diagnostic = polishing_evaluation.current_polishing_evaluation(rt, fn, controls)
    assert diagnostic["complete"] and not diagnostic["errors"]
    passing = diagnostic["groups"]["passing_attempt"]
    assert not passing["lost_matches"] and not passing["added_matches"]

    fn.source = fn.source.replace("'BAD'", "'fine'")
    calls = _record_calls(monkeypatch)
    result = tools_eval.evaluate_functions(names=[fn.name])
    assert result["ok"], result
    assert Counter(calls) == Counter({f"case-{i}": 1 for i in range(80)})
    assert fn.scores["sampled"] is False
    assert fn.scores["evaluation_scope_complete"] is True
    assert fn.scores["n_items"] == 80
    diagnostic = polishing_evaluation.current_polishing_evaluation(rt, fn, controls)
    assert diagnostic["complete"], diagnostic
    passing = diagnostic["groups"]["passing_attempt"]
    assert passing["n"] == 40
    assert set(passing["lost_matches"]) == {f"case-{i}" for i in range(0, 40, 2)}
    assert set(passing["added_matches"]) == {f"case-{i}" for i in range(40, 80, 2)}
    assert result["polishing_feedback"][fn.name]

    # The next request must reuse both the population scores and diagnostics.
    scores = deepcopy(fn.scores)
    monkeypatch.setattr(tools_eval, "run_detector", forbidden)
    repeated = tools_eval.evaluate_functions(names=[fn.name])
    assert repeated["ok"] and repeated["reused_names"] == [fn.name]
    assert fn.scores == scores


def test_full_controls_keep_all_issues_and_original_fixed_panel(polishing_runtime):
    rt, fn, before, original_panel = polishing_runtime
    result = collect_full_validation_controls(rt, fn, previous_scores=before)
    assert result["available"] and result["complete"], result
    assert set(result["panel_item_ids"]) == original_panel
    assert len(result["panel_item_ids"]) == PANEL_N
    groups = result["groups"]
    assert len(groups["flagged_success_contrast"]) == MATCHED_EACH
    assert len(groups["matched_failed_attempt"]) == MATCHED_EACH
    assert len(groups["unflagged_failed_attempt"]) == UNMATCHED_EACH
    assert len(result["item_ids"]) > MATCHED_N + UNMATCHED_EACH
    pool = result["unflagged_success_pool"]
    assert pool and not {row["item_id"] for row in pool} & set(result["item_ids"])
    assert all(row["benchmark_outcome"] == "passed" and row["baseline_label"] == "False"
               and int(row["item_id"].split("-")[1]) >= MATCHED_N for row in pool)
    # Both outcomes of repeated tasks survive; task diversity never drops attempts.
    assert {"case-0", "case-1"} <= set(result["item_ids"])
    rows = {row["item_id"]: row for row in result["controls"]}
    assert rows["case-0"]["task_id"] == rows["case-1"]["task_id"]
    assert any(row["new_since_checkpoint"] for row in rows.values())
    assert result["score_changes"]["task_support_comparison"] == {
        "before": task_support_assessment(before)["task_support_comparison"],
        "after": task_support_assessment(fn.scores)["task_support_comparison"],
    }
    after = result["task_support_change"]["after"]
    assert after["task_support_comparison_basis"] == "observed"
    assert after["n_matched_failed_tasks"] == MATCHED_EACH


def test_full_controls_are_cumulative_even_when_revision_removes_a_false_match(polishing_runtime):
    rt, fn, before, _ = polishing_runtime
    old = collect_full_validation_controls(rt, fn, previous_scores=before)
    fn.source = _LLM_SOURCE.replace("'BAD'", "'fine'")
    assert tools_eval.final_evaluate_functions()["ok"]
    new = collect_full_validation_controls(rt, fn, previous_controls=old)
    assert new["available"] and new["complete"], new
    assert set(old["item_ids"]) <= set(new["item_ids"])
    rows = {row["item_id"]: row for row in new["controls"]}
    assert "flagged_success_contrast" in rows["case-0"]["categories"]
    assert rows["case-0"]["baseline_label"] == "False"
    assert new["panel_item_ids"] == old["panel_item_ids"]
    assert new["initial_score_changes"] == old["score_changes"]
    assert new["initial_score_changes"] != new["score_changes"]
    assert new["initial_validation_grade"] == old["baseline_grade"]
    assert new["initial_task_support_change"] == old["task_support_change"]
    assert new["task_support_change"]["after"]["n_matched_failed_tasks"] == UNMATCHED_EACH
    changes = new["full_validation_passing_match_changes"]
    assert changes["complete"] and changes["n"] == CORPUS_N // 2
    assert changes["previous_flagged"] == MATCHED_EACH and changes["flagged"] == UNMATCHED_EACH
    assert len(changes["lost_matches"]) == MATCHED_EACH
    assert len(changes["added_matches"]) == UNMATCHED_EACH
    assert len(new["groups"]["introduced_success_match"]) == UNMATCHED_EACH


def test_missing_saved_bucket_membership_is_explicit_not_invented(polishing_runtime):
    rt, fn, _, _ = polishing_runtime
    bucket = fn.scores["buckets"][0]
    omitted = bucket["example_pass_ids"].pop()
    controls = collect_full_validation_controls(rt, fn)
    assert controls["available"] and not controls["complete"]
    assert omitted in controls["groups"]["prediction_outcome_unavailable"]
    assert controls["unavailable_reasons"]
    assert not any(row.get("baseline_label") for row in controls["controls"]
                   if row["item_id"] == omitted)


def test_polishing_uses_cached_pool_and_leaves_scores_and_localisations_untouched(
    polishing_runtime, monkeypatch,
):
    rt, fn, before, _ = polishing_runtime
    controls = collect_full_validation_controls(rt, fn, previous_scores=before)
    original = deepcopy(fn.to_dict())
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    out = run_polishing_evaluation(rt, fn, controls)
    assert out["ok"] and out["complete"]
    assert out["n"] == len(controls["item_ids"])
    assert out["population_metrics"] is False and out["grade"] is None
    assert not any(key in out for key in ("precision", "recall", "behavior_purity"))
    assert out["groups"]["flagged_success_contrast"]["flagged"] == MATCHED_EACH
    assert not out["groups"]["flagged_success_contrast"]["lost_matches"]
    assert fn.to_dict() == original
    assert current_polishing_evaluation(rt, fn, controls) == out


def test_all_unresolved_ids_survive_capped_error_examples(polishing_runtime, monkeypatch):
    rt, fn, _, _ = polishing_runtime
    fn.source += "\n# new execution version"
    execute = tools_eval.run_detector

    def fail(source, item, **kwargs):
        number = int(item["_item_id"].split("-")[1])
        if number < 15:
            raise RuntimeError("bad parse")
        if number < 25:
            raise DetectorAbstained("missing evidence")
        return execute(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", fail)
    assert tools_eval.final_evaluate_functions()["ok"]
    assert len(fn.scores["error_examples"]) == 3
    controls = collect_full_validation_controls(rt, fn)
    assert controls["available"] and controls["complete"], controls
    assert set(controls["groups"]["unresolved_prediction"]) == {f"case-{i}" for i in range(25)}
    assert controls["unavailable_reasons"]  # Error/abstention types are not all retained.
    out = run_polishing_evaluation(rt, fn, controls)
    assert not out["ok"] and out["complete"]
    assert out["errors"] == 15 and out["abstentions"] == 10
    assert len(out["groups"]["unresolved_prediction"]["unresolved"]) == 25


def test_polishing_receipt_does_not_survive_changed_provenance(polishing_runtime, monkeypatch):
    rt, fn, _, _ = polishing_runtime
    controls = collect_full_validation_controls(rt, fn)
    receipt = run_polishing_evaluation(rt, fn, controls)
    assert receipt["ok"]
    # Retained cases and the panel share the controls fingerprint with the optional pool.
    for change in ("claim", "source", "success_pool"):
        changed = deepcopy(controls)
        with monkeypatch.context() as patch_change:
            if change == "claim":
                patch_change.setattr(fn, "description", fn.description + " A stronger claim.")
            elif change == "source":
                patch_change.setitem(rt.dataset._fake_items[0], "response_text", "Changed trace")
            else:
                changed["unflagged_success_pool"].pop()
            assert current_polishing_evaluation(rt, fn, changed) is None, change
        assert current_polishing_evaluation(rt, fn, controls) == receipt, change


def test_interrupted_polishing_retains_partial_diagnostics_and_reuses_predictions(
    polishing_runtime, monkeypatch,
):
    rt, fn, _, _ = polishing_runtime
    controls = collect_full_validation_controls(rt, fn)
    fn.source += "\n# targeted refinement"
    rt.config.llm_eval_workers = 1
    original_scores = deepcopy(fn.scores)
    execute = tools_eval.run_detector
    calls = []

    def interrupted(source, item, **kwargs):
        calls.append(item["_item_id"])
        if len(calls) == 10:
            raise KeyboardInterrupt
        return execute(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_polishing_evaluation(rt, fn, controls)
    partial = current_polishing_evaluation(rt, fn, controls)
    assert partial and not partial["complete"] and partial["completed"] > 0
    completed = {row["item_id"] for row in partial["results"]}
    assert fn.scores == original_scores
    fetch = tools_eval._fetch_items_by_id

    def interrupted_before_predictions(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(tools_eval, "_fetch_items_by_id", interrupted_before_predictions)
    with pytest.raises(KeyboardInterrupt):
        run_polishing_evaluation(rt, fn, controls)
    recovered = current_polishing_evaluation(rt, fn, controls)
    assert recovered["results"] == partial["results"]
    assert recovered["additional_controls"] == partial["additional_controls"]
    monkeypatch.setattr(tools_eval, "_fetch_items_by_id", fetch)
    calls.clear()

    def resumed(source, item, **kwargs):
        calls.append(item["_item_id"])
        return execute(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", resumed)
    out = run_polishing_evaluation(rt, fn, controls)
    assert out["ok"] and out["complete"]
    assert out["additional_controls"] == partial["additional_controls"]
    assert out["generalization_sample"] == partial["generalization_sample"]
    assert not completed.intersection(calls)
    assert fn.scores == original_scores


def test_changed_round_does_not_resample_frozen_polishing_panel(polishing_runtime):
    rt, fn, _, original_panel = polishing_runtime
    rt.run.round += 1
    rt.run.exploration_seed += 100
    controls = collect_full_validation_controls(rt, fn)
    assert controls["available"], controls
    assert set(controls["panel_item_ids"]) == original_panel
    rt.config.llm_eval_sample = 2 * PANEL_N
    assert not collect_full_validation_controls(rt, fn)["available"]


# Ordinary in-round evaluation automatically checks the cumulative repair set.


def activate(rt, fn, controls):
    write_phase(rt, frozen_names=[fn.name], controls={fn.name: controls})


def test_revision_automatically_tests_all_controls_then_only_scores_fixed_panel(
    polishing_runtime, monkeypatch,
):
    rt, fn, before, panel = polishing_runtime
    controls = collect_full_validation_controls(rt, fn, previous_scores=before)
    activate(rt, fn, controls)
    fn.source = fn.source.replace("'BAD'", "'NO_MATCH'")
    execute = tools_eval.run_detector
    seen = []
    scans = []
    streaming = tools_eval._evaluate_streaming

    def counted(source, item, **kwargs):
        seen.append(item["_item_id"])
        return execute(source, item, **kwargs)

    def stream(*args, **kwargs):
        scans.append(kwargs)
        return streaming(*args, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", counted)
    monkeypatch.setattr(tools_eval, "_evaluate_streaming", stream)
    out = tools_eval.evaluate_functions(names=[fn.name])
    assert out["ok"], out
    diagnostics = current_polishing_evaluation(rt, fn, controls)
    extras = {row["item_id"] for row in diagnostics["additional_controls"]}
    assert diagnostics["generalization_sample"]["selected_n"] == PANEL_N
    assert set(seen) == set(controls["item_ids"]) | extras
    # Panel calls reuse diagnostic checkpoints; added checks never enlarge the scoring panel.
    assert len(seen) == len(controls["item_ids"]) + len(extras) < CORPUS_N
    assert fn.scores["sampled"] is True and fn.scores["sample_n"] == PANEL_N
    assert set(fn.scores["evaluation_item_ids"]) == panel
    assert len(scans) == 1 and scans[0].get("max_items") is not None
    assert not scans[0].get("llm_census")
    assert diagnostics["complete"] and diagnostics["n"] > MATCHED_N + UNMATCHED_EACH
    for group in ("flagged_success_contrast", "matched_failed_attempt"):
        assert len(diagnostics["groups"][group]["lost_matches"]) == MATCHED_EACH
    monkeypatch.setattr(polishing_evaluation, "run_polishing_evaluation", forbid)
    assert tools_eval.evaluate_functions(names=[fn.name])["ok"]


def test_invalid_polishing_requests_fail_before_diagnostic_or_population_work(
    polishing_runtime, monkeypatch,
):
    rt, fn, _, _ = polishing_runtime
    activate(rt, fn, collect_full_validation_controls(rt, fn))
    monkeypatch.setattr(polishing_evaluation, "run_polishing_evaluation", forbid)
    monkeypatch.setattr(tools_eval, "_evaluate_functions", forbid)
    for arguments, error in [({"names": [fn.name, "does_not_exist"]}, "Unknown functions"),
                             ({"names": [fn.name], "limit": 20}, "Polishing keeps"),
                             ({"names": [fn.name], "within": "other_parent"}, "Polishing keeps")]:
        out = tools_eval.evaluate_functions(**arguments)
        assert not out["ok"] and error in out["error"], arguments


def test_explicit_full_request_is_not_routed_to_polishing_diagnostics(
    polishing_runtime, monkeypatch,
):
    rt, fn, _, _ = polishing_runtime
    activate(rt, fn, collect_full_validation_controls(rt, fn))
    monkeypatch.setattr(polishing_evaluation, "run_polishing_evaluation", forbid)
    assert tools_eval.evaluate_functions(names=[fn.name], mode="full")["ok"]
    assert fn.scores["eval_mode"] == "full" and fn.scores["evaluation_scope_complete"]


def test_candidate_screen_blocks_diagnostics_before_spending(polishing_runtime, monkeypatch):
    rt, fn, _, _ = polishing_runtime
    activate(rt, fn, collect_full_validation_controls(rt, fn))
    monkeypatch.setattr("autofa.session.candidate_screen.ensure_candidate_screen",
                        lambda *args, **kwargs: {"evaluation_allowed": False})
    monkeypatch.setattr(polishing_evaluation, "run_polishing_evaluation", forbid)
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    out = tools_eval.evaluate_functions(names=[fn.name])
    assert not out["ok"] and out["candidate_screen_blocked"]


def test_unavailable_controls_do_not_pretend_regression_cases_were_checked(
    polishing_runtime, monkeypatch,
):
    rt, fn, _, _ = polishing_runtime
    activate(rt, fn, {"available": False, "reason": "Original panel could not be verified."})
    monkeypatch.setattr(polishing_evaluation, "run_polishing_evaluation", forbid)
    out = tools_eval.evaluate_functions(names=[fn.name])
    assert out["ok"]
    # The wrapper still supplies ordinary estimates/current full scores, but
    # exposes a separate unavailable diagnostic result instead of a fake success.
    assert "Original panel could not be verified" in str(out)


# One cheap corpus scan supplies both population results and repair diagnostics.


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    fn = FeatureFunction(name="plain", description="Observable behavior",
                         source="def detect(item):\n    return True", kind="python")
    rt = SimpleNamespace(run=SimpleNamespace(run_id="run", run_dir=lambda: tmp_path))
    ids = list("abcdefgh")
    passed = set("abef")
    inventory = {"inventory_complete": True, "corpus_fingerprint": "corpus", "item_ids": ids,
                 "task_by_item_id": {key: f"task-{key}" for key in ids},
                 "passed_item_ids": sorted(passed),
                 "task_by_failed_item_id": {key: f"task-{key}" for key in ids if key not in passed},
                 "identity_by_item_id": {key: {"benchmark": "bench"} for key in ids}}
    monkeypatch.setattr(tools_eval, "_evaluation_context", lambda runtime: "source")
    monkeypatch.setattr(family_coverage, "cached_task_inventory", lambda runtime: inventory)
    controls = {
        "available": True, "complete": True, "scope": "targeted_polishing_controls",
        "name": fn.name, "claim_hash": _claim(fn), "run_id": "run",
        "source_signature": "source", "corpus_fingerprint": "corpus",
        "selection": [True, "exact"], "panel_item_ids": ["a", "b"],
        "item_ids": list("abcdgh"),
        "controls": [{"item_id": key, "task_id": f"task-{key}", "benchmark": "bench",
                      "benchmark_outcome": "passed" if key in passed else "failed",
                      "baseline_label": key in "ac", "categories": ["retained_case"]}
                     for key in "abcdgh"],
        "unflagged_success_pool": [
            {"item_id": key, "task_id": f"task-{key}", "benchmark": "bench",
             "benchmark_outcome": "passed", "baseline_label": False} for key in "ef"],
    }
    controls["controls_fingerprint"] = polishing_controls_identity(controls)
    return rt, fn, controls, inventory


def census(fn, inventory, *, matched="ce", errors=(), abstentions=()):
    failed = set(inventory["task_by_failed_item_id"])
    unresolved = set(errors) | set(abstentions)
    fn.scores = {
        "evaluated": True, "sampled": False, "evaluation_scope_complete": True,
        "evaluation_claim_hash": _claim(fn), "evaluation_context": "source",
        "evaluation_run_id": "run", "task_recall_corpus_fingerprint": "corpus",
        "n_scored": len(inventory["item_ids"]) - len(unresolved),
        "n_unresolved": len(unresolved), "errors": len(errors),
        "n_insufficient": len(abstentions), "buckets": [],
        "error_examples": [{"item_id": key, "error": "saved error"} for key in errors[:1]],
    }
    for label in (True, False):
        ids = [key for key in inventory["item_ids"] if key not in unresolved
               and (key in matched) == label]
        fn.scores["buckets"].append({
            "bucket": str(label), "n": float(len(ids)),
            "example_fail_ids": [key for key in ids if key in failed],
            "example_pass_ids": [key for key in ids if key not in failed],
        })


def test_full_observer_reports_added_and_removed_matches_without_mutating_scores(corpus):
    rt, fn, controls, inventory = corpus
    fn.source += "\n# repaired"
    census(fn, inventory)
    before = deepcopy(fn.to_dict())
    with polishing_census_diagnostics(rt, {fn.name: (fn, controls)}) as collector:
        for key in inventory["item_ids"]:
            collector.observe(fn, {"_item_id": key}, key in "ce", None)
        out = collector.finish({"evaluated_names": [fn.name]})[fn.name]
    assert out["complete"] and out["ok"]
    assert out["n"] == out["observed_in_scan"] == 8
    assert out["panel_n"] == 2 and out["execution_scope"] == "full_dataset"
    passing = out["groups"]["passing_attempt"]
    assert passing["n"] == 4
    assert passing["lost_matches"] == ["a"] and passing["added_matches"] == ["e"]
    assert out["population_metrics"] is False and out["grade"] is None
    assert not {"precision", "recall", "behavior_purity"} & out.keys()
    assert fn.to_dict() == before
    assert current_polishing_evaluation(rt, fn, controls) == out


def test_repeated_interruption_preserves_new_matches_and_unresolved_predictions(corpus):
    rt, fn, controls, inventory = corpus
    fn.source += "\n# repaired"
    with pytest.raises(KeyboardInterrupt), polishing_census_diagnostics(
        rt, {fn.name: (fn, controls)},
    ) as collector:
        collector.observe(fn, {"_item_id": "e"}, True, None)
        collector.observe(fn, {"_item_id": "f"}, None, ValueError("bad parse"))
        collector.observe(fn, {"_item_id": "a"}, None, DetectorAbstained("missing fact"))
        raise KeyboardInterrupt
    previous = current_polishing_evaluation(rt, fn, controls)
    assert previous["completed"] == 3 and not previous["complete"]
    assert previous["errors"] == previous["abstentions"] == 1
    assert previous["groups"]["passing_attempt"]["added_matches"] == ["e"]
    with pytest.raises(KeyboardInterrupt), polishing_census_diagnostics(
        rt, {fn.name: (fn, controls)},
    ):
        raise KeyboardInterrupt
    assert current_polishing_evaluation(rt, fn, controls)["results"] == previous["results"]

    census(fn, inventory, errors=("f",), abstentions=("a",))
    with polishing_census_diagnostics(rt, {fn.name: (fn, controls)}) as collector:
        out = collector.finish({"reused_names": [fn.name]})[fn.name]
    assert out["complete"] and not out["ok"]
    assert out["errors"] == out["abstentions"] == 1
    assert not out["unknown_outcomes"]
    assert set(out["groups"]["passing_attempt"]["unresolved"]) == {"a", "f"}
    assert "a" not in out["groups"]["passing_attempt"]["lost_matches"]


def test_saved_census_keeps_unresolved_types_explicit_when_details_were_not_retained(corpus):
    rt, fn, controls, inventory = corpus
    census(fn, inventory, errors=("a", "b"), abstentions=("f",))
    before = deepcopy(fn.scores)
    with polishing_census_diagnostics(rt, {fn.name: (fn, controls)}) as collector:
        out = collector.finish({"reused_names": [fn.name]})[fn.name]
    assert out["complete"] and not out["ok"]
    assert out["errors"] == 2 and out["abstentions"] == 1
    assert out["unknown_outcomes"] == 2 and out["control_limitations"]
    assert len(out["groups"]["passing_attempt"]["unresolved"]) == 3
    assert not out["groups"]["passing_attempt"]["lost_matches"]
    assert fn.scores == before


# Claim, source context and corpus are one saved-identity check.
@pytest.mark.parametrize("change", ["claim", "truncated_buckets"])
def test_incompatible_or_incomplete_saved_census_never_fabricates_diagnostics(corpus, change):
    rt, fn, controls, inventory = corpus
    census(fn, inventory)
    if change == "claim":
        fn.scores["evaluation_claim_hash"] = "old claim"
    else:
        fn.scores["buckets"][0]["example_pass_ids"].clear()
    with polishing_census_diagnostics(rt, {fn.name: (fn, controls)}) as collector:
        out = collector.finish({"reused_names": [fn.name]})[fn.name]
    assert not out["complete"] and not out["ok"] and out["error"]
    assert out["completed"] == 0


@pytest.mark.parametrize("observed,finish", [
    ("a", {}), ("abcdefgh", {"ok": False, "evaluated_names": [], "error": "Did not commit"}),
], ids=["partial-scan", "aborted-scan"])
def test_unfinished_scan_does_not_certify_or_borrow_a_previous_census(
    corpus, observed, finish,
):
    rt, fn, controls, inventory = corpus
    census(fn, inventory)
    with polishing_census_diagnostics(rt, {fn.name: (fn, controls)}) as collector:
        assert collector.finish({"reused_names": [fn.name]})[fn.name]["complete"]
    with polishing_census_diagnostics(rt, {fn.name: (fn, controls)}) as collector:
        for key in observed:
            collector.observe(fn, {"_item_id": key}, False, None)
        out = collector.finish(finish)[fn.name]
    # Keep previous exact outcomes, but do not certify this scan.
    assert out["completed"] == 8 and not out["complete"]


def test_source_content_change_during_scan_invalidates_completion(corpus, monkeypatch):
    rt, fn, controls, inventory = corpus
    census(fn, inventory)
    with polishing_census_diagnostics(rt, {fn.name: (fn, controls)}) as collector:
        for key in inventory["item_ids"]:
            collector.observe(fn, {"_item_id": key}, False, None)
        monkeypatch.setattr(tools_eval, "_evaluation_context", lambda runtime: "changed source")
        out = collector.finish({})[fn.name]
    assert not out["complete"] and "changed" in out["error"]
    assert current_polishing_evaluation(rt, fn, controls) is None


# New regression checks catch repairs that merely exchange passing matches.


@pytest.fixture
def sample_inputs(tmp_path):
    rt = SimpleNamespace(run=SimpleNamespace(run_dir=lambda: tmp_path, exploration_base_seed=41))
    controls = {
        "name": "example", "claim_hash": "baseline", "run_id": "run", "source_signature": "source",
        "controls_fingerprint": "controls", "selection": (True, "exact"),
        "panel_item_ids": ["old", "other"],
        "controls": [{"item_id": "old", "task_id": "old-task", "benchmark_outcome": "passed",
                      "categories": ["flagged_success_contrast"], "baseline_label": True}],
        "unflagged_success_pool": [
            {"item_id": f"task-{i}-rollout-{j}", "task_id": f"task-{i}",
             "benchmark": "a" if i % 2 else "b", "benchmark_outcome": "passed",
             "baseline_label": False, "baseline_claim_hash": "baseline"}
            for i in range(6) for j in range(2)],
    }
    return rt, controls, tmp_path / "polishing_evaluations"


def save_sample(directory, controls, claim, extra, outcomes=()):
    result = {key: controls[key] for key in (
        "name", "run_id", "source_signature", "controls_fingerprint")}
    atomic_write_json(directory / f"{claim}.json", {
        **result, **extra, "claim_hash": claim, "results": list(outcomes)})


def test_revision_sample_is_task_distinct_deterministic_and_frozen_on_retry(sample_inputs):
    rt, controls, directory = sample_inputs
    original = deepcopy(controls)
    first = prepare_diagnostic_controls(
        rt, "example", "revision", controls, directory / "revision.json")
    rows = first["additional_controls"]
    assert len(rows) == len({row["task_id"] for row in rows}) == 2
    assert {row["benchmark"] for row in rows} == {"a", "b"}
    assert all(row["baseline_label"] is False for row in rows)
    reordered = {**controls, "unflagged_success_pool": controls["unflagged_success_pool"][::-1]}
    assert prepare_diagnostic_controls(rt, "example", "revision", reordered,
                                       directory / "revision.json") == first
    save_sample(directory, controls, "revision", first)
    rt.run.exploration_base_seed = 99  # Recover saved allocation even if caller seed changes.
    assert prepare_diagnostic_controls(rt, "example", "revision", controls,
                                       directory / "revision.json") == first
    assert controls == original


def test_new_revision_rotates_tasks_and_retains_only_new_problem_cases(sample_inputs):
    rt, controls, directory = sample_inputs
    first = prepare_diagnostic_controls(rt, "example", "one", controls, directory / "one.json")
    positive, negative = first["additional_controls"]
    save_sample(directory, controls, "one", first, [
        {"item_id": positive["item_id"], "status": "value", "prediction": True},
        {"item_id": negative["item_id"], "status": "value", "prediction": False},
    ])
    second = prepare_diagnostic_controls(rt, "example", "two", controls, directory / "two.json")
    rows = second["additional_controls"]
    retained = [row for row in rows if "introduced_success_match" in row["categories"]]
    assert [row["item_id"] for row in retained] == [positive["item_id"]]
    assert "fresh_unflagged_success" not in retained[0]["categories"]
    assert negative["item_id"] not in {row["item_id"] for row in rows}
    fresh = [row for row in rows if row not in retained]
    assert len(fresh) == 2
    assert not {row["task_id"] for row in fresh} & {positive["task_id"], negative["task_id"]}
    assert second["generalization_sample"]["previously_checked_task_n"] == 2
    assert retained_diagnostic_problems(rt, "example", controls) == retained


def test_partial_failure_controls_survive_and_pool_exhaustion_is_explicit(sample_inputs):
    rt, controls, directory = sample_inputs
    controls["unflagged_success_pool"] = controls["unflagged_success_pool"][:2]
    first = prepare_diagnostic_controls(rt, "example", "one", controls, directory / "one.json")
    assert first["generalization_sample"]["status"] == "exhausted"
    assert first["generalization_sample"]["selected_n"] == 1  # Two rollouts of one task.
    row = first["additional_controls"][0]
    save_sample(directory, controls, "one", first, [
        {"item_id": row["item_id"], "status": "error", "prediction": None}])
    second = prepare_diagnostic_controls(rt, "example", "two", controls, directory / "two.json")
    assert second["generalization_sample"]["selected_n"] == 0
    assert second["generalization_sample"]["retained_problem_n"] == 1
    assert "execution_error" in second["additional_controls"][0]["categories"]
    assert second["generalization_sample"]["limitation"]


def test_missing_pool_and_unverified_tasks_do_not_claim_fresh_coverage(sample_inputs):
    rt, controls, directory = sample_inputs
    malformed = {key: value for key, value in controls.items()
                 if key != "unflagged_success_pool"}
    out = prepare_diagnostic_controls(rt, "example", "revision", malformed,
                                      directory / "malformed.json")
    assert out["generalization_sample"]["status"] == "unavailable"
    assert out["additional_controls"] == []
    for row in controls["unflagged_success_pool"]:
        row["task_id"] = None
    unknown = prepare_diagnostic_controls(rt, "example", "revision", controls,
                                          directory / "unknown.json")
    assert unknown["generalization_sample"]["status"] == "exhausted"
    assert unknown["generalization_sample"]["distinct_task_n"] == 0
