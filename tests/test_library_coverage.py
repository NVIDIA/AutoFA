# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library coverage counts observed cases: discovery coverage, OK-or-better union and headlines."""

from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rich.console import Console

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset, sample_items
from autofa.mcp import tools_data
from autofa.metrics.discovery_grades import grade_summary
from autofa.metrics.library_coverage import failure_cohort, library_failure_coverage
from autofa.metrics.library_quality_coverage import (
    library_quality_coverage,
    library_quality_coverage_for_runtime,
)
from autofa.metrics.live_progress import runtime_display_progress_snapshot
from autofa.metrics.progress import library_coverage_rows, render_terminal_progress
from autofa.metrics.task_support import build_task_support
from autofa.reports.failure_cards import build_data, refresh_report_data, render_html
from autofa.session.family_coverage import failed_task_inventory
from autofa.session.round_feedback import round_bulletin, round_change, round_snapshot
from autofa.ui.activity import AgentActivity

from support.coverage import corpus, coverage_detector, reviewed_detector
from support.discovery import snapshot
from support.task_support import attach_task_support, report_renderers, run_node

# Discovery coverage of the failure cohort and coverage-directed sampling.


def coverage(*functions, failures=("a", "b", "c", "d"), **kwargs):
    return library_failure_coverage(FunctionLibrary(functions=list(functions)), failures,
                                    run_id="r1", **kwargs)


def test_parent_child_coverage_is_union_with_explicit_additional_contributions():
    parent = reviewed_detector("invalid_call", ("a", "b", "c"), ("a", "b"))
    child = reviewed_detector("missing_argument", ("a", "b"), ("b",), parent="invalid_call")
    other = reviewed_detector("other", ("c", "d"), ("c", "d"))
    result = coverage(parent, child, other)
    assert result["n_known_matched_failed"] == 4
    assert result["n_review_confirmed_failed"] == 4
    assert result["observed_match_recall_lower_bound"] == 1
    by_name = {row["name"]: row for row in result["detectors"]}
    assert by_name["missing_argument"]["n_exclusive_matched_failed"] == 0
    assert by_name["missing_argument"]["n_additional_matched_beyond_parent"] == 0
    assert by_name["other"]["n_exclusive_matched_failed"] == 1
    assert by_name["invalid_call"]["n_exclusive_review_confirmed_failed"] == 1
    assert "known_matched_failed_item_ids" not in result


def test_examples_and_estimated_counts_cannot_invent_item_coverage():
    fn = reviewed_detector("sampled", ("a",), ("a",))
    fn.example_fail_ids = ["b", "c", "d"]
    fn.scores.update(sampled=True, n_fail_caught=400, a=400, behavior_purity=1,
                     behavior_n_yes=100)
    result = coverage(fn, include_item_ids=True)
    assert result["known_matched_failed_item_ids"] == ["a"]
    assert result["review_confirmed_failed_item_ids"] == ["a"]
    assert result["n_without_known_match"] == 3
    assert result["reviewed_confirmation_recall_lower_bound"] == .25


def test_passes_and_foreign_items_are_excluded_without_using_detector_as_ground_truth():
    matches = ("a", "pass", "foreign")
    result = coverage(reviewed_detector("broad", matches, matches))
    assert result["n_known_matched_failed"] == result["n_review_confirmed_failed"] == 1


def test_secondary_categorical_bucket_does_not_inflate_the_scored_detectors_coverage():
    bucket = "positive"
    fn = reviewed_detector("multi", ("a", "b"), ("a", "b"))
    fn.scores.update(bucket=bucket, buckets=[{"bucket": bucket}, {"bucket": "invalid_type"}])
    fn.failure_localisations["a"]["assigned_label"] = bucket
    fn.failure_localisations["b"]["assigned_label"] = "invalid_type"
    result = coverage(fn, include_item_ids=True)
    assert result["known_matched_failed_item_ids"] == ["a"]
    assert result["review_confirmed_failed_item_ids"] == ["a"]


# Stale evaluations or localisations lose the match; stale judging or a review that is not
# a final independent confirmation keeps the match but loses the confirmation.
@pytest.mark.parametrize(("target", "change", "matched"), [
    ("scores", {"evaluation_run_id": "old-run"}, 0),
    ("scores", {"evaluation_stale": True}, 0),
    ("localisation", {"source_sha256": "old-source"}, 0),
    ("scores", {"claim_hash": "old-claim"}, 1),
    ("scores", {"judge_contract_version": "old-contract"}, 1),
    ("verdict", {"verification_status": "uncertain"}, 1),
    ("verdict", {"claim_contract_id": "other-contract"}, 1),
    ("verdict", {"initial_review": {}}, 1),
], ids=["cross-run", "stale-evaluation", "old-source", "old-claim", "old-judge-contract",
        "unverified", "other-contract-review", "no-initial-review"])
def test_stale_or_unconfirmed_evidence_never_counts_as_confirmed_coverage(
    target, change, matched,
):
    fn = reviewed_detector("candidate", ("a",), ("a",))
    record = {"scores": fn.scores, "localisation": fn.failure_localisations["a"],
              "verdict": fn.scores["behavior_verdicts"][0]}[target]
    record.update(change)
    result = coverage(fn)
    assert result["n_known_matched_failed"] == matched
    assert result["n_review_confirmed_failed"] == 0


def test_strategy_change_preserves_confirmed_and_observed_matches():
    fn = reviewed_detector("candidate", ("a",), ("a",), strategy="precision")
    assert coverage(fn)["n_review_confirmed_failed"] == 1
    result = coverage(fn, objective_strategy="recall")
    assert result["n_known_matched_failed"] == 1
    assert result["n_review_confirmed_failed"] == 1


def test_empty_cohort_has_no_invented_recall():
    result = coverage(reviewed_detector("candidate", ("a",), ("a",)), failures=[])
    assert result["n_failed"] == 0
    assert result["observed_match_recall_lower_bound"] is None
    assert result["reviewed_confirmation_recall_lower_bound"] is None


def test_absent_parent_and_incomplete_evaluation_are_not_assumed_complete():
    fn = reviewed_detector("child", ("a",), parent="missing")
    fn.scores.update(evaluation_scope_complete=False, n_unresolved=3)
    result = coverage(fn)
    row = result["detectors"][0]
    assert result["n_known_matched_failed"] == 1
    assert result["n_without_known_match"] == 3
    assert row["evaluation_scope_complete"] is False
    assert row["n_detector_unresolved"] == 3
    assert row["parent_present"] is False
    assert row["n_additional_matched_beyond_parent"] is None


def test_metadata_fallback_streams_all_identities_and_keeps_unknown_separate():
    def batches(*, columns):
        assert "response_text" not in columns
        assert "is_correct" in columns
        yield [{"_item_id": "first", "is_correct": False}]
        yield [{"_item_id": "last", "is_correct": False}, {"_item_id": "unknown"}]

    cohort = failure_cohort(SimpleNamespace(parquet_paths=[], iter_batches=batches))
    assert cohort["failed_item_ids"] == {"first", "last"}
    assert cohort["n_unknown_outcomes"] == 1


def test_in_memory_sampling_fallback_respects_fail_exclusions():
    class MemoryDataset(Dataset):
        def load_items(self, *, limit=None):
            return [{"_item_id": "a", "is_correct": False},
                    {"_item_id": "b", "is_correct": False},
                    {"_item_id": "pass", "is_correct": True}]

    result = sample_items(MemoryDataset(parquet_paths=[]), n_fail=10, n_pass=10,
                          exclude_fail_item_ids={"a", "pass"})
    assert {row["_item_id"] for row in result["items"]} == {"b", "pass"}


def _parquet(tmp_path, rows):
    path = tmp_path / "traces.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return Dataset(parquet_paths=[path])


def _tool_runtime(monkeypatch, dataset, **run):
    rt = SimpleNamespace(
        dataset=dataset, library=FunctionLibrary(), config=Config(),
        run=SimpleNamespace(run_id="r1", resolved_benchmarks=lambda: ["b"], **run),
    )
    monkeypatch.setattr(tools_data, "get_runtime", lambda: rt)
    return rt


def _rollouts(task, n):
    return [{"problem_id": task, "run_idx": i, "is_correct": False, "model_id": "m",
             "benchmark": "b", "response_text": "Failed trace"} for i in range(n)]


@pytest.fixture
def dataset(tmp_path):
    rows = [
        {"problem_id": "f0", "is_correct": False, "model_id": "m", "benchmark": "b"},
        {"problem_id": "f1", "is_correct": False, "model_id": "m", "benchmark": "b"},
        {"problem_id": "f2", "is_correct": False, "model_id": "m", "benchmark": "b"},
        {"problem_id": "pass", "is_correct": True, "model_id": "m", "benchmark": "b"},
        {"problem_id": "unknown", "is_correct": None, "model_id": "m", "benchmark": "b"},
        {"problem_id": "foreign", "is_correct": False, "model_id": "else", "benchmark": "b"},
    ]
    for row in rows:
        row.update(run_idx=0, response_text="Meaningful trace " * 20)
    path = tmp_path / "traces.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return Dataset(parquet_paths=[path], model_id="m", benchmarks=["b"])


def test_failure_cohort_uses_metadata_and_excludes_unknown_outcomes(dataset, monkeypatch):
    monkeypatch.setattr(dataset, "iter_batches",
                        lambda **_: pytest.fail("should use identity index"))
    result = failure_cohort(dataset)
    assert result["failed_item_ids"] == {"m|b|f0|0", "m|b|f1|0", "m|b|f2|0"}
    assert result["n_items"] == 5
    assert result["n_passed"] == result["n_unknown_outcomes"] == 1


def test_sampling_exclusions_apply_before_draw_and_preserve_pass_controls(dataset):
    result = sample_items(dataset, n_fail=3, n_pass=3, prefer_informative=False,
                          exclude_fail_item_ids={"m|b|f0|0", "m|b|f1|0",
                                                 "m|b|unknown|0", "m|b|pass|0"})
    assert {item["_item_id"] for item in result["items"]} == {"m|b|f2|0", "m|b|pass|0"}
    assert result["n_fail"] == result["n_pass"] == 1


def test_sampling_filters_are_discovery_only_and_coverage_tool_is_read_only(dataset, monkeypatch):
    rt = _tool_runtime(monkeypatch, dataset, exploration_ssot=True, exploration_seed=5)
    rt.library.upsert(reviewed_detector("candidate", ("m|b|f0|0", "m|b|f1|0"), ("m|b|f0|0",)))
    rt.config = Config(behavior_strictness="causal")
    normal = tools_data.sample_items(n_fail=10, n_pass=1, prefer_informative=False)
    unmatched = tools_data.sample_items(n_fail=10, n_pass=1, prefer_informative=False,
                                         coverage="unmatched")
    unconfirmed = tools_data.sample_items(n_fail=10, n_pass=1, prefer_informative=False,
                                           coverage="unconfirmed")
    assert normal["n_fail"] == 4
    assert unmatched["n_fail"] == 1
    assert unconfirmed["n_fail"] == 2
    assert normal["n_pass"] == unmatched["n_pass"] == unconfirmed["n_pass"] == 1
    assert unmatched["n_excluded_known_failures"] == 2
    assert unconfirmed["n_excluded_known_failures"] == 1
    assert "known_matched_failed_item_ids" not in unmatched["library_coverage"]
    # Other evidence tools keep every corpus task.
    assert tools_data.peek_chunk(chunk_size=20)["n"] == 5
    assert tools_data.search_items(limit=20)["n"] == 5
    result = tools_data.library_coverage()
    assert result["ok"] is True
    assert result["n_failed"] == 3
    assert result["n_known_matched_failed"] == 2
    assert result["n_review_confirmed_failed"] == 1
    assert tools_data.sample_items(coverage="nonsense")["ok"] is False


def test_repeat_attempts_do_not_establish_cross_datapoint_generalization():
    repeats = reviewed_detector("repeats", ("a", "b", "c"), ("a", "b", "c"))
    broad = reviewed_detector("broad", ("a", "d"), ("a", "d"))
    result = coverage(repeats, broad, task_by_item_id={
        "a": "bench:one", "b": "bench:one", "c": "bench:one", "d": "bench:two",
    })
    assert result["n_failed"] == result["n_known_matched_failed"] == 4
    assert result["n_failed_tasks"] == result["n_known_matched_failed_tasks"] == 2
    assert result["observed_task_recall_lower_bound"] == 1
    by_name = {row["name"]: row for row in result["detectors"]}
    assert by_name["repeats"]["n_exclusive_matched_failed"] == 2
    assert by_name["repeats"]["n_exclusive_matched_failed_tasks"] == 0
    assert by_name["repeats"]["n_repeated_matched_failed_attempts"] == 2
    assert by_name["repeats"]["generalization_status"] == "single_task_diagnostic"
    assert by_name["broad"]["generalization_status"] == "cross_task_causal_evidence"
    assert by_name["broad"]["n_exclusive_matched_failed_tasks"] == 1


def test_unknown_task_identity_and_sampled_matches_are_not_generalization_evidence():
    fn = reviewed_detector("unknown", ("a", "b"))
    fn.scores.update(sampled=True, n_fail_caught=900)
    result = coverage(fn, task_by_item_id={"a": "bench:task"})
    assert result["n_failed_tasks"] == result["n_known_matched_failed_tasks"] == 1
    assert result["n_failed_attempts_without_task_identity"] == 3
    row = result["detectors"][0]
    assert row["n_matched_failed_attempts_without_task_identity"] == 1
    assert row["generalization_status"] == "task_identity_unknown"
    assert row["n_repeated_matched_failed_attempts"] == 0


@pytest.mark.parametrize("prefer_informative", [False, True])
def test_default_discovery_sample_covers_tasks_and_keeps_same_prompt_pass_control(
    tmp_path, monkeypatch, prefer_informative,
):
    rows = _rollouts("common", 80)
    rows += [{**rows[0], "problem_id": name} for name in ("rare1", "rare2")]
    rows += [{**rows[0], "run_idx": 80, "is_correct": True,
              "response_text": "Successful correction"}]
    _tool_runtime(monkeypatch, _parquet(tmp_path, rows))

    result = tools_data.sample_items(n_fail=10, n_pass=1,
                                    prefer_informative=prefer_informative)
    assert result["ok"] and result["distinct_tasks"]
    assert result["n_fail"] == result["n_failed_tasks"] == 3
    assert result["n_repeated_failed_attempts"] == 0
    assert result["n_pass"] == 1
    assert [row["source_task_identity"] for row in result["items"]].count("b:common") == 2

    repeats = tools_data.sample_items(n_fail=10, n_pass=1, distinct_tasks=False,
                                     prefer_informative=prefer_informative)
    assert repeats["ok"] and not repeats["distinct_tasks"]
    assert repeats["n_fail"] == 10
    assert repeats["n_repeated_failed_attempts"] == 10 - repeats["n_failed_tasks"]


def test_default_discovery_sample_keeps_missing_task_identity_explicit(tmp_path, monkeypatch):
    _tool_runtime(monkeypatch, _parquet(tmp_path, [
        {"benchmark": "b", "item_id": "opaque_attempt_1", "is_correct": False},
        {"benchmark": "b", "item_id": "opaque_attempt_2", "is_correct": False},
    ]))
    result = tools_data.sample_items(n_fail=10, n_pass=0)
    assert result["ok"] and result["distinct_tasks"]
    assert result["n_fail"] == result["n_failed_attempts_without_task_identity"] == 2
    assert result["n_failed_tasks"] == result["n_repeated_failed_attempts"] == 0
    assert all(row["source_task_identity"] is None for row in result["items"])


def test_unconfirmed_task_sampling_excludes_other_rollouts_but_keeps_pass_controls(
    tmp_path, monkeypatch,
):
    from autofa.session import task_coverage

    rows = _rollouts("covered", 8)
    rows += [{**rows[0], "problem_id": "new-task"},
             {**rows[0], "run_idx": 8, "is_correct": True},
             {**rows[0], "problem_id": "unknown-outcome", "is_correct": None}]
    _tool_runtime(monkeypatch, _parquet(tmp_path, rows))
    monkeypatch.setattr(task_coverage, "task_coverage_snapshot", lambda runtime, **kwargs: {
        "available": True, "confirmed_failed_task_ids": ["b:covered"],
        "n_confirmed_failed_tasks": 1,
    })
    result = tools_data.sample_items(n_fail=20, n_pass=5, coverage="unconfirmed_tasks",
                                    distinct_tasks=True, prefer_informative=False)
    assert result["ok"]
    assert result["n_fail"] == result["n_failed_tasks"] == result["n_pass"] == 1
    assert {row["_item_id"] for row in result["items"]} == {
        "m|b|new-task|0", "m|b|covered|8",
    }
    assert result["n_excluded_known_failures"] == 8
    assert "confirmed_failed_task_ids" not in result["task_coverage"]


def test_unconfirmed_task_sampling_reports_unavailable_coverage(dataset, monkeypatch):
    from autofa.session import task_coverage

    _tool_runtime(monkeypatch, dataset)
    monkeypatch.setattr(task_coverage, "task_coverage_snapshot", lambda *args, **kwargs: {
        "available": False, "reason": "Source task evidence unavailable",
    })
    result = tools_data.sample_items(coverage="unconfirmed_tasks")
    assert not result["ok"] and not result["available"]


def test_unmatched_samples_record_verified_task_exposure_through_mcp(
    dataset, tmp_path, monkeypatch,
):
    from autofa.mcp import runtime, server
    from autofa.session.family_discovery import STATE_FILE

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / STATE_FILE).write_text("{}")
    rt = _tool_runtime(monkeypatch, dataset, run_dir=lambda: run_dir)
    monkeypatch.setattr(runtime, "get_runtime", lambda: rt)
    result = tools_data.sample_items(n_fail=10, n_pass=0, coverage="unmatched")
    server._record_tool_exposure("sample_items", {}, result)
    history = json.loads((run_dir / "review_evidence.json").read_text())
    assert set(history["discovery_item_ids"]) == {"m|b|f0|0", "m|b|f1|0", "m|b|f2|0"}
    assert history["discovery_task_groups"]["m|b|f0|0"] == '["b","f0"]'


# The OK-or-better match union, not summed rates or reviewed cases.


def evaluate(inventory, *functions, **kwargs):
    kwargs.setdefault("behavior_strictness", "causal")
    return library_quality_coverage(FunctionLibrary(functions=list(functions)), inventory,
                                    **kwargs)


def _matches(start, stop):
    return [f"f-{i}-0" for i in range(start, stop)]


def test_grade_membership_matches_shared_classifier_all_three_good_tiers():
    inventory = corpus()
    functions = [
        coverage_detector("good", inventory, _matches(0, 5), coverage=.04),
        coverage_detector("excellent", inventory, _matches(3, 8), coverage=.15),
        coverage_detector("outstanding", inventory, _matches(6, 11), coverage=.3),
        coverage_detector("ok", inventory, _matches(10, 15), precision=.6),
        coverage_detector("weak", inventory, _matches(15, 20), adherence=.2),
        FeatureFunction(name="pending", description="", source="def pending(item): return True"),
    ]
    result = evaluate(inventory, *functions)
    assert result["eligible_detector_names"] == ["excellent", "good", "ok", "outstanding"]
    assert result["n_qualifying_detectors"] == result["n_eligible_detectors"] == 4
    assert result["n_good_detectors"] == 3
    assert result["coverage"] == 15 / 20
    good_only = evaluate(inventory, *functions, minimum_grade="GOOD")
    assert good_only["eligible_detector_names"] == ["excellent", "good", "outstanding"]
    assert good_only["n_qualifying_detectors"] == good_only["n_good_detectors"] == 3
    assert good_only["coverage"] == 11 / 20
    assert {row["code"] for row in grade_summary(functions)["functions"]} >= {
        "GOOD", "EXCELLENT", "OUTSTANDING", "OK", "WEAK", "UNSCORED",
    }


def test_excluded_ok_detectors_are_counted_without_inflating_good_counts():
    inventory = corpus()
    ok = coverage_detector("ok", inventory, ["f-0-0"])
    good = coverage_detector("good", inventory, _matches(0, 5))
    result = evaluate(inventory, ok, good, run_id="different-run")
    assert result["status"] == "unavailable"
    assert result["coverage"] is None
    assert result["n_qualifying_detectors"] == result["n_excluded_detectors"] == 2
    assert result["n_good_detectors"] == result["n_excluded_good_detectors"] == 1
    assert result["n_eligible_detectors"] == 0


def test_invalid_minimum_grade_is_rejected():
    with pytest.raises(ValueError, match="minimum_grade must be OK or GOOD"):
        evaluate(corpus(), minimum_grade="WEAK")


@pytest.mark.parametrize("change", [
    {"sampled": True, "sample_n": 200, "sample_frame_n": 2000, "n_items": 2000},
    {"n_insufficient": 1}, {"n_fail_caught": 6},
], ids=["sampled", "abstention", "inconsistent-counts"])
def test_sampled_incomplete_or_inconsistent_evaluations_are_lower_bounds(change):
    inventory = corpus()
    fn = coverage_detector("candidate", inventory, _matches(0, 5))
    fn.scores.update(change)
    result = evaluate(inventory, fn)
    assert result["n_good_detectors"] == 1
    assert result["status"] == "observed_lower_bound"
    assert result["coverage"] == .25
    assert result["comparison_key"] is result["cohort_fingerprint"] is None


def test_source_failure_set_excludes_passes_and_foreign_saved_matches():
    inventory = corpus()
    fn = coverage_detector("candidate", inventory, _matches(0, 5) + ["p-0", "foreign"])
    result = evaluate(inventory, fn)
    assert result["n_matched_failed_attempts"] == 5
    assert result["status"] == "observed_lower_bound"


@pytest.mark.parametrize(("change", "options"), [
    ({"evaluation_stale": True}, {}), ({"judge_policy_hash": "old"}, {}),
    ({}, {"behavior_strictness": "behavioral"}),
], ids=["stale-evaluation", "old-judge-policy", "other-strictness"])
def test_stale_or_invalid_scores_never_contribute(change, options):
    inventory = corpus()
    fn = coverage_detector("stale", inventory, _matches(0, 5))
    fn.scores.update(change)
    result = evaluate(inventory, fn, **options)
    assert result["n_eligible_detectors"] == 0
    assert result["n_matched_failed_attempts"] == 0


def test_different_corpus_not_interchangeable_even_when_ids_and_size_match():
    inventory = corpus()
    fn = coverage_detector("candidate", inventory, _matches(0, 5))
    result = evaluate({**inventory, "corpus_fingerprint": "different-source"}, fn)
    assert result["status"] == "unavailable"
    assert result["n_excluded_good_detectors"] == 1
    assert result["coverage"] is None


def test_excluded_good_detector_prevents_exactness_of_remaining_usable_union():
    inventory = corpus()
    a = coverage_detector("a", inventory, _matches(0, 5))
    b = coverage_detector("b", inventory, _matches(5, 10))
    b.scores["evaluation_run_id"] = "other-run"
    b.scores["task_support"]["evaluation_run_id"] = "other-run"
    result = evaluate(inventory, a, b, run_id="quality-test")
    assert result["n_good_detectors"] == 2
    assert result["n_eligible_detectors"] == result["n_excluded_good_detectors"] == 1
    assert result["status"] == "observed_lower_bound"
    assert result["coverage"] == .25


def test_missing_task_id_keeps_attempts_but_never_invents_task_denominator():
    inventory = corpus(unknown_task=True)
    fn = coverage_detector("candidate", inventory, _matches(0, 6))
    result = evaluate(inventory, fn)
    assert result["status"] == "exact"
    assert result["coverage"] == .3
    assert result["n_matched_failed_tasks"] == 5
    assert result["n_failed_attempts_without_task_identity"] == 1
    assert result["task_status"] == "unavailable"
    assert result["task_coverage"] is None


def test_unknown_source_does_not_turn_into_zero_coverage():
    result = evaluate({"inventory_complete": False})
    assert result["status"] == "unavailable"
    assert result["coverage"] is result["task_coverage"] is None
    assert result["n_failed_attempts"] is None


def test_empty_or_ungraded_library_has_exact_zero_on_known_failure_corpus():
    inventory = corpus()
    empty = evaluate(inventory)
    pending = evaluate(inventory, FeatureFunction(
        name="pending", description="", source="def pending(item): pass",
    ))
    assert empty["status"] == pending["status"] == "exact"
    assert empty["coverage"] == pending["coverage"] == 0
    assert empty["task_coverage"] == 0
    fn = coverage_detector("candidate", inventory, _matches(0, 5))
    assert empty["comparison_key"] == evaluate(inventory, fn)["comparison_key"]


def test_round_comparisons_require_same_source_and_policy_even_with_equal_rates():
    inventory = corpus()
    causal = evaluate(inventory)
    different_policy = evaluate(inventory, behavior_strictness="behavioral")
    different_source = evaluate({**inventory, "corpus_fingerprint": "other"})
    good_only = evaluate(inventory, minimum_grade="GOOD")
    assert len({result["comparison_key"] for result in (
        causal, different_policy, different_source, good_only,
    )}) == 4
    assert good_only["cohort_fingerprint"] == causal["cohort_fingerprint"]
    assert (evaluate(inventory, minimum_grade="good")["comparison_key"]
            == good_only["comparison_key"])
    assert evaluate(inventory, minimum_grade="ok")["comparison_key"] == causal["comparison_key"]


def test_unknown_benchmark_outcomes_leave_complete_failure_denominator_unavailable():
    inventory = corpus(unknown_outcome=True)
    fn = coverage_detector("candidate", inventory, _matches(0, 5))
    result = evaluate(inventory, fn)
    assert result["n_failed_attempts"] == 20
    assert result["n_unknown_outcomes"] == 1
    assert result["n_matched_failed_attempts"] == 5
    assert result["status"] == result["task_status"] == "unavailable"
    assert result["coverage"] is result["task_coverage"] is None
    assert result["comparison_key"] is None


def test_retained_off_panel_matches_survive_but_rechecked_negatives_do_not_revive():
    inventory = corpus()
    fn = coverage_detector("candidate", inventory, _matches(0, 7))
    fn.scores.update(sampled=True, evaluation_item_ids=["f-0-0", "f-1-0", "p-0"])
    # Only f-0 is a positive in the latest panel; f-1 was retested as negative.
    fn.failure_localisations = {"f-0-0": fn.failure_localisations["f-0-0"]}
    result = evaluate(inventory, fn)
    assert result["status"] == "observed_lower_bound"
    assert result["n_matched_failed_attempts"] == 6
    assert result["coverage"] == .3


def test_cached_runtime_read_never_loads_dataset_or_executes_detectors(monkeypatch):
    import autofa.session.family_coverage as family

    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering must not initiate source reads")

    monkeypatch.setattr(family, "cached_task_inventory", forbidden)
    inventory = corpus()
    rt = SimpleNamespace(config=Config(), run=SimpleNamespace(run_id="quality-test"),
                         library=FunctionLibrary(functions=[
                             coverage_detector("ok", inventory, ["f-0-0"])]))
    assert library_quality_coverage_for_runtime(rt, cached_only=True)["status"] == "unavailable"
    rt._family_failed_task_inventory = inventory
    assert library_quality_coverage_for_runtime(rt, cached_only=True)["coverage"] == 1 / 20
    # The requested cutoff reaches the metric: coverage_summary.json saves a Good-only cohort.
    good = library_quality_coverage_for_runtime(rt, cached_only=True, minimum_grade="GOOD")
    assert good["minimum_grade"] == "GOOD" and good["coverage"] == 0


def test_runtime_source_outage_reports_unavailable_without_mutating_library(monkeypatch):
    import autofa.session.family_coverage as family

    def unavailable(*args, **kwargs):
        raise OSError("source temporarily offline")

    monkeypatch.setattr(family, "cached_task_inventory", unavailable)
    rt = SimpleNamespace(config=Config(), run=SimpleNamespace(run_id="quality-test"),
                         library=FunctionLibrary())
    result = library_quality_coverage_for_runtime(rt)
    assert result["status"] == "unavailable"
    assert "temporarily offline" in result["reason"]


def test_report_uses_visible_saved_review_grades_without_relaxing_live_eligibility():
    inventory = corpus()
    fn = coverage_detector("historical", inventory, _matches(0, 5))
    fn.scores["judge_policy_hash"] = "0" * 64
    live = evaluate(inventory, fn)
    saved = evaluate(inventory, fn, saved_assessment=True)
    assert live["n_good_detectors"] == 0
    assert saved["n_good_detectors"] == 1
    assert saved["status"] == "exact"
    assert saved["coverage"] == .25
    assert saved["assessment_scope"] == "saved_report"


def test_grade_downgrade_and_removal_recompute_union_without_saved_headline_cache():
    inventory = corpus()
    a = coverage_detector("a", inventory, _matches(0, 5))
    b = coverage_detector("b", inventory, _matches(4, 9))
    library = FunctionLibrary(functions=[a, b])
    assert library_quality_coverage(library, inventory)["coverage"] == .45
    a.scores["behavior_purity"] = .5
    ok = library_quality_coverage(library, inventory)
    assert ok["coverage"] == .45
    assert ok["n_good_detectors"] == 1
    a.scores["behavior_purity"] = .4
    downgraded = library_quality_coverage(library, inventory)
    assert downgraded["coverage"] == .25
    assert downgraded["eligible_detector_names"] == ["b"]
    library.remove("b")
    removed = library_quality_coverage(library, inventory)
    assert removed["coverage"] == 0
    assert removed["status"] == "exact"


def test_malformed_detector_cannot_hide_other_valid_detector_coverage():
    inventory = corpus()
    valid = coverage_detector("valid", inventory, _matches(0, 5))
    malformed = coverage_detector("bad", inventory, _matches(5, 10))
    malformed.scores["behavior_n_judged"] = "not-a-count"
    result = evaluate(inventory, valid, malformed)
    assert result["coverage"] == .25
    assert result["eligible_detector_names"] == ["valid"]
    assert result["unavailable_grade_detector_names"] == ["bad"]


def test_no_failed_attempts_is_undefined_percentage_not_zero():
    inventory = failed_task_inventory(SimpleNamespace(load_items=lambda: [
        {"_item_id": "pass", "problem_id": "a", "benchmark": "b", "is_correct": True},
    ]))
    result = evaluate(inventory)
    assert result["status"] == "unavailable"
    assert result["n_failed_attempts"] == 0
    assert result["coverage"] is result["task_coverage"] is None


def test_categorical_bucket_selection_counts_only_the_scored_category():
    bucket = "positive"
    inventory = corpus()
    fn = coverage_detector("categorical", inventory, _matches(0, 10))
    fn.scores.update(bucket=bucket, buckets=[{"bucket": bucket}, {"bucket": "another"}],
                     bucket_mode="exact", n_flagged=5., n_fail_caught=5.)
    for index, record in enumerate(fn.failure_localisations.values()):
        record["assigned_label"] = bucket if index < 5 else "another"
    fn.scores.update(build_task_support(fn, inventory))
    result = evaluate(inventory, fn)
    assert result["status"] == "exact"
    assert result["coverage"] == .25
    assert result["n_matched_failed_attempts"] == 5


# The live headline and agent feedback show the selected detector match union.


@pytest.fixture
def rt(tmp_path, monkeypatch):
    from autofa.mcp import runtime

    inventory = corpus(repeats=2)
    functions = [
        coverage_detector("first", inventory, _matches(0, 8)),
        coverage_detector("second", inventory, _matches(4, 12) + ["f-4-1"]),
        coverage_detector("ok", inventory, _matches(10, 15), precision=.6),
        coverage_detector("weak", inventory, _matches(15, 20), adherence=.2),
    ]
    runtime_state = SimpleNamespace(
        library=FunctionLibrary(functions=functions), config=Config(behavior_strictness="causal"),
        scores_history=[],
        run=SimpleNamespace(run_id=functions[0].scores["evaluation_run_id"],
                            round=0, min_failure_modes=1, max_rounds=3, run_dir=lambda: tmp_path),
        _family_failed_task_inventory=inventory,
    )
    monkeypatch.setattr(runtime, "_RUNTIME", runtime_state)
    # Rendering must use the identity cache; no source I/O, evaluation, or judge.
    def no_work(*args, **kwargs):
        raise AssertionError("The headline must only read cached identities and saved results")
    monkeypatch.setattr("autofa.session.family_coverage.failed_task_inventory", no_work)
    return runtime_state


def test_terminal_and_agent_share_match_union_not_confirmation_subset(rt):
    before = deepcopy(rt.library.to_dict())
    live = AgentActivity(Console(force_terminal=False, no_color=True))._library_snapshot(
        force=True)["progress_lines"]
    assert live[0].startswith("Library coverage  40.0% · 16/40 failed rollouts")
    assert live[1].startswith("Task coverage  75.0% · 15/20 distinct failed tasks")
    assert rt.library.to_dict() == before
    current = runtime_display_progress_snapshot(rt)
    result = current["library_quality_coverage"]
    assert result["n_matched_failed_attempts"] == 16  # Only one review per detector.
    output = StringIO()
    render_terminal_progress([], Console(file=output, width=200, no_color=True), current=current)
    assert "40.0%" in output.getvalue() and "16/40" in output.getvalue()
    bulletin = round_bulletin(rt, grades=round_snapshot(rt)["grades"])
    assert bulletin["library_quality_coverage"] == result
    assert "16/40 failed rollouts" in bulletin["message"]


def test_round_delta_compares_exact_selected_unions_and_keeps_reviews_separate(rt):
    first = round_snapshot(rt)
    rt.library.functions.append(coverage_detector(
        "additional", rt._family_failed_task_inventory, _matches(12, 17),
    ))
    change = round_change(first, round_snapshot(rt))
    assert change["matched_failure_net_change"] == 2
    assert change["matched_task_net_change"] == 2
    assert change["confirmed_task_net_change"] in {0, None}


@pytest.mark.parametrize("status,key", [
    ("observed_lower_bound", "same"), ("exact", "different"),
])
def test_sampling_or_corpus_changes_cannot_manufacture_round_coverage_gain(status, key):
    first, second = snapshot(), snapshot()
    first["library_quality_coverage"] = {
        "status": "exact", "task_status": "exact", "comparison_key": "same",
        "n_matched_failed_attempts": 5, "n_matched_failed_tasks": 5,
    }
    second["library_quality_coverage"] = {
        "status": status, "task_status": status, "comparison_key": key,
        "n_matched_failed_attempts": 50, "n_matched_failed_tasks": 40,
    }
    change = round_change(first, second)
    assert change["matched_failure_net_change"] is None
    assert change["matched_task_net_change"] is None
    assert not change["encouraging"]


def test_explicit_good_cutoff_keeps_ok_matches_out_of_cli_and_tool(rt):
    default = tools_data.library_coverage()["library_quality_coverage"]
    good = tools_data.library_coverage(minimum_grade="GOOD")["library_quality_coverage"]
    assert default["minimum_grade"] == "OK"
    assert default["n_eligible_detectors"] == 3 and default["n_good_detectors"] == 2
    assert default["n_matched_failed_attempts"] == 16
    assert good["minimum_grade"] == "GOOD"
    assert good["n_eligible_detectors"] == good["n_good_detectors"] == 2
    assert good["n_matched_failed_attempts"] == 13
    assert "2 Good-or-better detectors" in dict(library_coverage_rows(good))["Library coverage"]


def test_headline_counts_only_included_detectors_and_shows_tiny_lower_bound():
    result = {
        "available": True, "status": "observed_lower_bound", "coverage": .00005,
        "n_matched_failed_attempts": 1, "n_failed_attempts": 20000,
        "minimum_grade": "GOOD", "n_good_detectors": 3,
        "n_qualifying_detectors": 4, "n_eligible_detectors": 1, "n_excluded_detectors": 3,
    }
    headline = dict(library_coverage_rows(result))["Library coverage"]
    assert "1 Good-or-better detector" in headline
    assert "3 Good-or-better" not in headline and "4 Good-or-better" not in headline
    assert headline.startswith("≥0.005%")


# The report headline uses the same deduplicated match union.


class IdentityDataset:
    """Full identity projection, deliberately without a trajectory loader."""

    def __init__(self):
        # Nine failed tasks; the first two have a second failed rollout.
        self.rows = [
            {"_item_id": f"item-{task}-{repeat}", "problem_id": f"task-{task}",
             "benchmark": "bench", "is_correct": False}
            for repeat, n_tasks in ((0, 9), (1, 2)) for task in range(n_tasks)
        ] + [
            {"_item_id": f"pass-{task}", "problem_id": f"pass-task-{task}",
             "benchmark": "bench", "is_correct": True}
            for task in range(3)
        ]

    def iter_batches(self, **kwargs):
        assert kwargs["max_items"] is None
        yield self.rows


def _report_library():
    dataset = IdentityDataset()
    inventory = failed_task_inventory(dataset)
    functions = []
    for name, matches in [
        ("first", [f"item-{task}-0" for task in range(6)] + ["item-0-1"]),
        ("second", [f"item-{task}-0" for task in range(3, 9)]),
        ("weak", ["item-1-1"]),
    ]:
        fn = FeatureFunction(name=name, source=f"def {name}(item): return True",
                             description="A concrete operation is missing.",
                             mechanism="The omission leaves the required output absent.")
        attach_task_support(fn, task_n=9, repeat_n=2)
        fn.failure_localisations = {
            item_id: row for item_id, row in fn.failure_localisations.items()
            if item_id in matches
        }
        # Adherence is based on one reviewed case; union coverage uses all matches.
        fn.scores["behavior_verdicts"] = [
            row for row in fn.scores["behavior_verdicts"] if row["item_id"] in matches
        ][:1]
        fn.scores.update(
            n_items=14, n_scored=14, sample_frame_n=14, n_flagged=len(matches),
            n_fail_caught=len(matches), a=len(matches), b=0, c=11-len(matches), d=3,
            recall=len(matches)/11, precision=1.0, ppv_at_pi0=1.0,
            pct_caught=len(matches)/14, behavior_purity=1.0 if name != "weak" else .1,
            behavior_n_judged=1, behavior_n_completed=1, behavior_n_yes=1,
            sampled=False, evaluation_scope_complete=True, evaluation_complete=True,
            task_recall_corpus_fingerprint=inventory["corpus_fingerprint"],
        )
        fn.scores.update(build_task_support(fn, inventory))
        functions.append(fn)
    return FunctionLibrary(model_id="model", benchmark="bench", functions=functions), dataset


def _data(library, dataset=None):
    # A single supplied inspector example must never become the corpus denominator.
    return build_data(library, dataset=dataset, items={"preview": {"item_id": "preview"}})


def _render_headline(data):
    return run_node(report_renderers(render_html(data)) + "console.log(renderLibraryCoverage("
                    + json.dumps(data["meta"].get("library_quality_coverage")) + "));")


def test_report_headline_deduplicates_good_matches_and_rollouts_not_review_sample():
    library, dataset = _report_library()
    data = _data(library, dataset)
    summary = data["meta"]["library_quality_coverage"]
    assert summary["status"] == "exact"
    assert summary["n_good_detectors"] == 2
    assert summary["n_matched_failed_attempts"] == 10
    assert summary["n_failed_attempts"] == 11
    assert summary["coverage"] == pytest.approx(10 / 11)
    assert summary["n_matched_failed_tasks"] == summary["n_failed_tasks"] == 9
    assert summary["task_coverage"] == 1
    headline = _render_headline(data)
    assert "90.9%" in headline and "100.0%" in headline
    assert "10 / 11 failed rollouts matched" in headline


@pytest.mark.parametrize("status,prefix,rate,display", [
    ("exact", "", .0001, "0.01%"), ("observed_lower_bound", "≥", .0099, "1.0%"),
])
def test_library_headline_formats_small_rates_without_detector_grade_boundaries(
    status, prefix, rate, display,
):
    library, dataset = _report_library()
    data = _data(library, dataset)
    data["meta"]["library_quality_coverage"].update(
        status=status, coverage=rate, task_coverage=rate,
    )
    headline = _render_headline(data)
    assert f">{prefix}{display}</div>" in headline
    assert ">0.0%</div>" not in headline


def test_missing_source_does_not_treat_embedded_examples_or_score_totals_as_inventory():
    library, _ = _report_library()
    data = _data(library)
    summary = data["meta"]["library_quality_coverage"]
    assert summary["status"] == "unavailable"
    assert summary["coverage"] is summary["task_coverage"] is None
    headline = _render_headline(data)
    assert "Unavailable" in headline
    assert "0.0%" not in headline


def test_source_inventory_failure_does_not_block_saved_report():
    library, dataset = _report_library()

    def fail(**kwargs):
        raise OSError("source is temporarily offline")

    dataset.iter_batches = fail
    data = _data(library, dataset)
    assert len(data["functions"]) == 3
    assert data["meta"]["library_quality_coverage"]["status"] == "unavailable"


def test_historical_saved_review_uses_same_good_selection_as_visible_cards(monkeypatch):
    library, dataset = _report_library()
    monkeypatch.setattr("autofa.metrics.claude_judge.judge_policy_hash",
                        lambda *args, **kwargs: "new-policy")
    data = _data(library, dataset)
    good_cards = {fn["name"] for fn in data["functions"] if fn["verdict"]["code"] == "GOOD"}
    assert good_cards == {"first", "second"}
    assert all(fn["behavior_validation"]["status"] == "historical"
               for fn in data["functions"])
    summary = data["meta"]["library_quality_coverage"]
    assert set(summary["eligible_detector_names"]) == good_cards
    assert summary["coverage"] == pytest.approx(10 / 11)


def test_unchanged_offline_refresh_retains_union_but_changed_detector_invalidates_it():
    library, dataset = _report_library()
    saved = _data(library, dataset)
    original = deepcopy(saved["meta"]["library_quality_coverage"])
    # Normal portable library serialization must not invalidate the saved union.
    restored = FunctionLibrary.from_dict(library.to_dict())
    assert refresh_report_data(saved, restored)["meta"]["library_quality_coverage"] == original
    restored.functions[0].source += "\n# updated detector"
    refreshed = refresh_report_data(saved, restored)
    assert refreshed["meta"]["library_quality_coverage"]["status"] == "unavailable"
    assert refreshed["meta"]["library_quality_coverage"]["coverage"] is None
