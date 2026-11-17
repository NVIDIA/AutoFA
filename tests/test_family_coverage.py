# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family comparisons use exact saved-match coverage."""

from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.core.library import FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.data.parquet_store import Dataset
from autofa.metrics.claim_contract import claim_hash
from autofa.session.family_coverage import (
    candidate_coverage,
    failed_task_inventory,
    seed_union_coverage,
    verified_identity_projection,
)

from support.coverage import MetadataDataset, detector, reviewed_detector
from support.coverage import rt as rt


def seed_refs(*functions):
    return [{"name": fn.name, "claim_hash": fn.scores["evaluation_claim_hash"]}
            for fn in functions]


# Saved-match coverage counts exact observed matches and never extrapolates samples.


def test_seed_union_comparison_distinguishes_overlap_retention_and_new_tasks(rt):
    rt.dataset.rows.append({"_item_id": "d1", "benchmark": "bench", "problem_id": "d",
                            "is_correct": False})
    left = detector(rt, "left", ["a1", "a2"], n_items=5)
    right = detector(rt, "right", ["a1", "b1"], n_items=5)
    parent = detector(rt, "parent", ["a1", "b1", "d1"], n_items=5)
    original = deepcopy(rt.library.to_dict())
    result = seed_union_coverage(rt, parent, seed_refs(left, right))
    assert result["comparison_complete"] and result["task_comparison_complete"]
    assert result["seed_union_matched_failed"] == result["parent_matched_failed"] == 3
    assert result["preserved_seed_matches"] == 2
    assert result["n_lost_failed_attempts"] == 1
    assert result["n_lost_failed_tasks"] == 0  # The lost attempt repeats task a.
    assert result["n_additional_failed_attempts"] == result["n_additional_failed_tasks"] == 1
    assert result["seed_union_matched_failed_tasks"] == 2
    assert result["parent_matched_failed_tasks"] == 3
    assert rt.library.to_dict() == original
    assert rt.dataset.reads == 1


@pytest.mark.parametrize("side,updates", [
    ("seed", {"sampled": True, "evaluation_item_ids": ["a1", "c1"]}),
    ("parent", {"n_unresolved": 1, "evaluation_complete": False}),
])
def test_partial_synthesis_comparison_reports_observations_not_exact_gain_or_loss(
    rt, side, updates,
):
    left = detector(rt, "left", ["a1", "a2"])
    right = detector(rt, "right", ["a1"])
    parent = detector(rt, "parent", ["a1", "b1"])
    (left if side == "seed" else parent).scores.update(updates)
    result = seed_union_coverage(rt, parent, seed_refs(left, right))
    assert result["comparison_scope"] == "partial_observed_matches"
    assert not result["comparison_complete"]
    assert result["seed_matches_not_observed_in_parent"] == 1
    assert result["additional_matched_failed"] == 1
    assert result["n_additional_failed_attempts"] is None
    assert result["n_additional_failed_tasks"] is None
    assert result["n_lost_failed_attempts"] is None
    assert result["n_lost_failed_tasks"] is None


@pytest.mark.parametrize("change", ["remove", "edit"])
def test_changed_or_missing_seed_cannot_silently_become_an_empty_baseline(rt, change):
    left = detector(rt, "left", ["a1"])
    right = detector(rt, "right", ["b1"])
    refs = seed_refs(left, right)
    parent = detector(rt, "parent", ["a1", "a2", "b1"])
    if change == "remove":
        rt.library.remove("left")
    else:
        left.description = "A different claim"
        left.scores["evaluation_claim_hash"] = claim_hash(
            source=left.source, description=left.description, mechanism=left.mechanism,
        )
    result = seed_union_coverage(rt, parent, refs)
    assert result["stale_seed_names"] == ["left"]
    assert not result["seed_claims_current"]
    assert result["comparison_scope"] == "unavailable_seed_revision"
    assert result["parent_matched_failed"] == 3
    assert result["additional_matched_failed"] is None
    assert result["seed_union_matched_failed"] is None
    assert result["n_additional_failed_attempts"] is None


def test_missing_parent_evaluation_does_not_imply_all_seed_matches_were_lost(rt):
    left = detector(rt, "left", ["a1"])
    right = detector(rt, "right", ["b1"])
    parent = detector(rt, "parent", [], evaluated=False)
    result = seed_union_coverage(rt, parent, seed_refs(left, right))
    assert result["seed_claims_current"]
    assert result["comparison_scope"] == "unavailable_evaluation"
    assert result["parent_matched_failed"] is None
    assert result["seed_matches_not_observed_in_parent"] is None


def test_seed_confirmations_never_become_parent_confirmations(rt):
    rt.dataset.rows.append({"_item_id": "d1", "benchmark": "bench", "problem_id": "d",
                            "is_correct": False})
    left = reviewed_detector("left", ["a1"], ["a1"])
    right = reviewed_detector("right", ["b1"], ["b1"])
    parent = reviewed_detector("parent", ["a1", "b1", "d1"])
    rt.library = FunctionLibrary(functions=[left, right, parent])
    parent.scores["behavior_purity"] = 1  # A headline number proves no individual case.
    result = seed_union_coverage(rt, parent, seed_refs(left, right))
    assert result["additional_matched_failed_tasks"] == 1
    assert result["n_review_confirmed_additional_failed"] == 0
    assert result["n_review_confirmed_additional_tasks"] == 0
    parent = reviewed_detector("parent", ["a1", "b1", "d1"], ["d1"])
    rt.library.upsert(parent)
    result = seed_union_coverage(rt, parent, seed_refs(left, right))
    assert result["n_review_confirmed_additional_failed"] == 1
    assert result["n_review_confirmed_additional_tasks"] == 1


def test_inventory_distinguishes_attempts_tasks_unknowns_and_benchmarks(rt):
    rt.dataset.rows += [
        {"_item_id": "d1", "benchmark": "other", "task_id": "a", "resolved": False},
        # An in-memory item_id alias is an attempt label, not a genuine task.
        {"_item_id": "opaque1", "item_id": "opaque", "accuracy": 0.0},
        {"_item_id": "unknown", "problem_id": "unknown", "is_correct": None},
    ]
    inventory = failed_task_inventory(rt.dataset)
    assert inventory["n_failed_attempts"] == 5
    assert inventory["n_failed_tasks"] == 3
    assert inventory["n_failed_attempts_without_task_identity"] == 1
    assert inventory["task_by_failed_item_id"]["opaque1"] is None
    assert inventory["task_by_failed_item_id"]["d1"] == "other:a"
    assert inventory["n_passed_attempts"] == 1
    assert inventory["n_unknown_outcomes"] == 1


def test_inventory_real_parquet_filters_and_aliases_without_trace_reads(tmp_path, monkeypatch):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "m", "benchmark": "b", "task_id": "x", "repeat": 0,
         "resolved": False, "response_text": "huge irrelevant trace"},
        {"model_id": "other", "benchmark": "b", "task_id": "y", "repeat": 0,
         "resolved": False, "response_text": "huge irrelevant trace"},
    ]), path)
    dataset = Dataset(parquet_paths=[path], model_id="m", benchmarks=["b"])
    original = pq.ParquetFile.iter_batches

    def checked(self, *args, **kwargs):
        assert "response_text" not in kwargs["columns"]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", checked)
    result = failed_task_inventory(dataset)
    assert result["n_items"] == 1
    assert list(result["task_by_failed_item_id"].values()) == ["b:x"]


def test_task_column_presence_is_per_shard_and_nulls_do_not_become_attempt_ids(tmp_path):
    genuine_path, opaque_path = tmp_path / "genuine.parquet", tmp_path / "opaque.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"benchmark": "b", "problem_id": "task", "item_id": "attempt_1", "run_idx": 0,
         "is_correct": False},
        {"benchmark": "b", "problem_id": "task", "item_id": "attempt_2", "run_idx": 1,
         "is_correct": False},
        {"benchmark": "b", "problem_id": None, "item_id": "attempt_3", "run_idx": 2,
         "is_correct": False},
        {"benchmark": "b", "problem_id": "boundary", "item_id": "attempt_4", "run_idx": 0,
         "is_correct": True},
    ]), genuine_path)
    pq.write_table(pa.Table.from_pylist([
        {"benchmark": "b", "item_id": "opaque", "is_correct": False},
    ]), opaque_path)
    # A real question_id survives an earlier opaque item_id alias.
    question_path = tmp_path / "questions.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"benchmark": "b", "item_id": "opaque_2", "question_id": "real_question",
         "is_correct": False, "response_text": "Do not read me"},
    ]), question_path)
    result = failed_task_inventory(
        Dataset(parquet_paths=[genuine_path, opaque_path, question_path]))
    assert result["n_failed_attempts"] == 5
    assert result["n_failed_tasks"] == 2
    assert result["n_failed_attempts_without_task_identity"] == 2
    assert sorted(task for task in result["task_by_item_id"].values() if task) == [
        "b:boundary", "b:real_question", "b:task", "b:task",
    ]


def test_inventory_rejects_duplicates_and_closes_generator(rt):
    rt.dataset.rows.append(dict(rt.dataset.rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        failed_task_inventory(rt.dataset)


def test_streaming_errors_never_retry_with_eager_trace_load():
    class Broken(MetadataDataset):
        parquet_paths = ["real.parquet"]

        def iter_batches(self, **kwargs):
            raise OSError("unreadable shard")

        def load_items(self):
            raise AssertionError("must not materialize trace corpus")

    with pytest.raises(OSError, match="unreadable shard"):
        failed_task_inventory(Broken([]))


def test_complete_census_float_counts_and_distinct_tasks(rt):
    fn = detector(rt, "family", ["a1", "a2"], pass_ids=["c1"])
    result = candidate_coverage(rt, fn)
    assert result["failure_recall"] == 2 / 3
    assert result["recall_scope"] == "full_selected_corpus"
    assert result["census_complete"] and result["match_inventory_complete"]
    assert result["n_known_matched_failed"] == 2
    assert result["n_matched_failed_tasks"] == 1
    assert result["n_additional_failed_attempts"] == 2
    assert result["known_matched_failed_item_ids"] == ["a1", "a2"]


def test_additional_matches_are_union_difference_and_inventory_is_cached(rt):
    previous = detector(rt, "previous", ["a1"])
    first = candidate_coverage(rt, previous)
    other = detector(rt, "other", ["a1", "b1"])
    current = detector(rt, "current", ["a1", "a2", "b1"])
    result = candidate_coverage(rt, current, [first, other])
    assert result["comparison_complete"]
    assert result["n_additional_failed_attempts"] == 1
    assert result["n_observed_additional_failed_attempts"] == 1
    assert rt.dataset.reads == 1


def test_additional_attempt_on_same_task_does_not_add_datapoint_coverage(rt):
    previous = detector(rt, "previous", ["a1"])
    current = detector(rt, "current", ["a1", "a2"])
    result = candidate_coverage(rt, current, [previous])
    assert result["n_additional_failed_attempts"] == 1
    assert result["n_additional_failed_tasks"] == 0
    assert result["n_repeated_matched_failed_attempts"] == 1
    assert result["generalization_status"] == "single_task_diagnostic"
    current = detector(rt, "current", ["a2", "b1"])
    result = candidate_coverage(rt, current, [previous])
    assert result["n_additional_failed_tasks"] == 1
    assert result["generalization_status"] == "cross_task_matches_observed"


def test_unknown_task_in_comparison_prevents_exact_additional_datapoint_claim(rt):
    rt.dataset.rows[0].pop("problem_id")
    previous = detector(rt, "previous", ["a1"])
    current = detector(rt, "current", ["b1"])
    result = candidate_coverage(rt, current, [previous])
    assert result["n_additional_failed_attempts"] == 1
    assert result["n_observed_additional_failed_tasks"] == 1
    assert result["n_additional_failed_tasks"] is None


def test_assessed_snapshot_retains_revision_after_library_repair(rt):
    previous = detector(rt, "previous", ["a1"])
    snapshot = candidate_coverage(rt, previous)
    previous.source = "def previous(item): return False"
    current = detector(rt, "current", ["a1", "b1"])
    result = candidate_coverage(rt, current, [snapshot])
    assert result["n_additional_failed_attempts"] == 1


@pytest.mark.parametrize("updates,scope", [
    ({"sampled": True, "n_flagged": 2.0}, "sample_estimate"),
    ({"n_items": 2.0, "evaluation_scope_complete": False}, "evaluated_subset"),
    ({"n_unresolved": 1}, "resolved_evaluated_attempts"),
])
def test_partial_or_sampled_evaluation_never_becomes_exact_coverage(rt, updates, scope):
    fn = detector(rt, "family", ["a1"], **updates)
    result = candidate_coverage(rt, fn)
    assert result["recall_scope"] == scope
    assert not result["match_inventory_complete"]
    assert result["n_additional_failed_attempts"] is None
    assert result["n_known_matched_failed"] == 1
    assert result["observed_failure_recall_lower_bound"] == 1 / 3


def test_partial_previous_matches_cannot_prove_new_coverage(rt):
    previous = detector(rt, "previous", ["a1"], sampled=True)
    current = detector(rt, "current", ["a1", "a2"])
    result = candidate_coverage(rt, current, [previous])
    assert result["n_observed_additional_failed_attempts"] == 1
    assert result["n_additional_failed_attempts"] is None
    assert not result["comparison_complete"]


@pytest.mark.parametrize("change", ["missing", "stale_source", "foreign_pass"])
def test_missing_or_invalid_positive_records_do_not_establish_complete_inventory(rt, change):
    if change == "foreign_pass":
        # A foreign positive record cannot stand in for a missing passing record.
        fn = detector(rt, "family", ["a1"], pass_ids=["foreign"])
    else:
        fn = detector(rt, "family", ["a1", "b1"])
    if change == "missing":
        del fn.failure_localisations["b1"]
    elif change == "stale_source":
        fn.failure_localisations["b1"]["source_sha256"] = "old"
    result = candidate_coverage(rt, fn)
    assert result["census_complete"]
    assert not result["match_inventory_complete"]
    assert result["n_known_matched_failed"] == 1
    assert result["n_additional_failed_attempts"] is None


@pytest.mark.parametrize("field,value", [
    ("evaluation_claim_hash", "old"), ("evaluation_run_id", "old"),
])
def test_stale_claim_or_different_run_has_no_current_matches(rt, field, value):
    fn = detector(rt, "family", ["a1"], **{field: value})
    result = candidate_coverage(rt, fn)
    assert not result["evaluation_current"]
    assert result["failure_recall"] is None
    assert result["n_known_matched_failed"] == 0
    assert result["n_additional_failed_attempts"] is None


def test_categorical_bucket_does_not_add_other_positive_categories(rt):
    fn = detector(rt, "family", ["a1"], bucket="tool_error", bucket_mode="exact")
    fn.failure_localisations["a1"]["assigned_label"] = "tool_error"
    fn.failure_localisations["b1"] = {
        "item_id": "b1", "assigned_label": "bad_artifact", "source_sha256": text_hash(fn.source),
    }
    result = candidate_coverage(rt, fn)
    assert result["known_matched_failed_item_ids"] == ["a1"]
    assert result["match_inventory_complete"]


def test_unknown_outcomes_or_tasks_never_become_failed_task_breadth(rt):
    rt.dataset.rows[0].pop("problem_id")
    rt.dataset.rows[1]["is_correct"] = None
    fn = detector(rt, "family", ["a1", "a2"])
    result = candidate_coverage(rt, fn)
    assert result["n_known_matched_failed"] == 1
    assert result["n_matched_failed_tasks"] == 0
    assert result["n_matched_failed_attempts_without_task_identity"] == 1
    assert not result["census_complete"]


def test_complete_matches_still_do_not_establish_unknown_task_breadth(rt):
    rt.dataset.rows[0].pop("problem_id")
    result = candidate_coverage(rt, detector(rt, "family", ["a1"]))
    assert result["match_inventory_complete"]
    assert not result["matched_task_inventory_complete"]
    assert result["n_matched_failed_tasks"] == 0


def test_unknown_seed_task_prevents_exact_task_gain_claim(rt):
    rt.dataset.rows[0].pop("problem_id")
    left = detector(rt, "left", ["a1"])
    right = detector(rt, "right", ["a2"])
    parent = detector(rt, "parent", ["a1", "b1"])
    result = seed_union_coverage(rt, parent, seed_refs(left, right))
    assert result["comparison_complete"] and not result["task_comparison_complete"]
    assert result["additional_matched_failed_tasks"] == 1
    assert result["n_additional_failed_tasks"] is None


def test_different_corpus_snapshot_and_missing_detector_are_unavailable(rt):
    first = candidate_coverage(rt, detector(rt, "previous", ["a1"]))
    first = deepcopy(first)
    first["corpus_fingerprint"] = "another corpus"
    fn = detector(rt, "current", ["b1"])
    result = candidate_coverage(rt, fn, [first, "deleted"])
    assert result["unavailable_comparisons"] == ["previous", "deleted"]
    assert result["n_additional_failed_attempts"] is None


def test_no_matches_is_complete_when_current_census_counts_are_zero(rt):
    result = candidate_coverage(rt, detector(rt, "family", []))
    assert result["match_inventory_complete"]
    assert result["n_additional_failed_attempts"] == 0


def test_verified_identity_ignores_display_aliases_and_returns_safe_copy(rt):
    projected = verified_identity_projection(rt, {
        "_item_id": "a1", "problem_id": "invented", "benchmark": "wrong",
        "task_id": "wrong", "response_text": "Do not retain this",
    })
    assert projected == {"_item_id": "a1", "benchmark": "bench", "problem_id": "a"}
    projected["problem_id"] = "mutated"
    assert verified_identity_projection(rt, "a1")["problem_id"] == "a"
    assert verified_identity_projection(rt, {"item_id": "missing", "problem_id": "invented"}) == {
        "_item_id": "missing",
    }
    assert rt.dataset.reads == 1
