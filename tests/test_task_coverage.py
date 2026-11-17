# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed-task breadth: detector recall/support receipts, library union and progress."""

import json
from copy import deepcopy
from math import comb
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.dependencies import review_source
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.mcp import tools_eval, tools_library
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.discovery_grades import grade_summary
from autofa.metrics.task_recall import build_task_recall, current_task_recall_scores
from autofa.metrics.task_support import (
    build_task_support,
    confirmed_failed_task_support,
    matched_failed_task_support,
    task_support_assessment,
)
from autofa.metrics.verdicts import current_behavior_scores, verdict_for_scores
from autofa.reports.failure_cards import build_data
from autofa.session.discovery_target import discovery_good_target, discovery_grade_assessment
from autofa.session.research_status import read_research_status, write_research_status
from autofa.session.task_coverage import task_coverage_prompt, task_coverage_snapshot
from autofa.session.task_progress import final_validation_handoff, observe_task_progress

from support.repair_review import _runtime
from support.task_support import _function, _render_card, attach_task_support

THRESHOLDS = {"ppv_at_pi0": .70, "behavior_purity": .90, "min_confirmed_failed_tasks": 2}


# Per-detector task recall stays separate from rollout-weighted association metrics.

@pytest.fixture
def inventory():
    return {
        "inventory_complete": True, "corpus_fingerprint": "selected-corpus",
        "n_items": 6, "n_failed_attempts": 5, "n_failed_tasks": 3,
        "n_failed_attempts_without_task_identity": 0, "n_unknown_outcomes": 0,
        "item_ids": ["a1", "a2", "a3", "b1", "c1", "pass"],
        "task_by_failed_item_id": {
            "a1": "bench:a", "a2": "bench:a", "a3": "bench:a",
            "b1": "bench:b", "c1": "bench:c",
        },
    }


def recall_detector(matches, *, scores=None, meta=None):
    fn = FeatureFunction(
        name="example", description="Shared requirement violation",
        mechanism="An incorrect transformation changes the required result",
        source="def detect(item):\n    return True\n", meta=meta or {},
    )
    source = review_source(fn)
    fn.scores = {
        "evaluated": True, "evaluation_run_id": "run",
        "evaluation_claim_hash": claim_hash(
            source=source, description=fn.description, mechanism=fn.mechanism,
        ),
        "n_items": 6, "n_flagged": len(matches), "n_fail_caught": len(matches),
        "sampled": False, "evaluation_scope_complete": True, "bucket": "positive",
        "recall": len(matches) / 5,
        **(scores or {}),
    }
    fn.failure_localisations = {
        item_id: {"item_id": item_id, "assigned_label": True,
                  "source_sha256": text_hash(source)} for item_id in matches
    }
    return fn


def test_repeated_rollouts_add_attempts_but_no_task_recall(inventory):
    one = recall_detector(["a1"])
    repeated = recall_detector(["a1", "a2", "a3"])
    first = build_task_recall(one, inventory, run_id="run")
    second = build_task_recall(repeated, inventory, run_id="run")
    assert first["task_recall"] == second["task_recall"] == 1 / 3
    assert second["n_matched_failed_attempts"] == 3
    assert second["n_matched_failed_tasks"] == 1
    assert first["task_recall_scope"] == second["task_recall_scope"] == "full_selected_corpus"
    assert one.scores["recall"] == .2
    assert repeated.scores["recall"] == .6
    broad = build_task_recall(recall_detector(["a1", "b1"]), inventory, run_id="run")
    assert broad["task_recall"] == 2 / 3


@pytest.mark.parametrize("scores", [
    {"sampled": True, "sample_n": 2, "n_fail_caught": 20, "n_flagged": 20},
    {"n_insufficient": 1},
    {"n_flagged": 2},  # A missing positive record is not a complete census.
], ids=["sampled", "abstention", "missing-positive-record"])
def test_partial_scoring_uses_full_task_denominator_as_lower_bound(inventory, scores):
    fn = recall_detector(["a1"], scores=scores)
    before = deepcopy(fn.to_dict())
    result = build_task_recall(fn, inventory, run_id="run")
    assert result["task_recall"] == 1 / 3
    assert result["task_recall_scope"] == "observed_lower_bound"
    assert fn.to_dict() == before


def test_unknown_failed_task_denominator_stays_unavailable(inventory):
    inventory["task_by_failed_item_id"]["a3"] = None
    inventory["n_failed_attempts_without_task_identity"] = 1
    result = build_task_recall(recall_detector(["a1", "a3"]), inventory, run_id="run")
    assert result["task_recall"] is None
    assert result["task_recall_scope"] == "unavailable"
    assert result["n_matched_failed_tasks"] == 1
    assert result["n_matched_failed_attempts"] == 2
    assert result["n_matched_failed_attempts_without_task_identity"] == 1


@pytest.mark.parametrize("change", [{"inventory_complete": False}, {"n_unknown_outcomes": 1}])
def test_unusable_source_inventory_cannot_produce_numeric_task_recall(inventory, change):
    inventory.update(change)
    result = build_task_recall(recall_detector(["a1"]), inventory, run_id="run")
    assert result["task_recall"] is None


def test_noncurrent_evaluations_are_unavailable(inventory):
    fn = recall_detector(["a1"], scores={"evaluation_run_id": "other"})
    result = build_task_recall(fn, inventory, run_id="run")
    assert result["task_recall"] is None
    assert result["n_matched_failed_tasks"] == 0


def test_counts_only_current_source_primary_positive_bucket_and_failed_items(inventory):
    fn = recall_detector(["a1", "a2", "a3", "b1", "c1", "pass"], scores={
        "bucket": "bad-transform", "n_fail_caught": 1, "n_flagged": 2,
    })
    for record in fn.failure_localisations.values():
        record["assigned_label"] = "bad-transform"
    fn.failure_localisations["a2"]["assigned_label"] = "different-mechanism"
    fn.failure_localisations["a3"]["assigned_label"] = False
    fn.failure_localisations["b1"]["source_sha256"] = "old-source"
    fn.failure_localisations["c1"]["item_id"] = "other-id"
    fn.failure_localisations["foreign"] = {
        "item_id": "foreign", "assigned_label": "bad-transform",
        "source_sha256": text_hash(fn.source),
    }
    result = build_task_recall(fn, inventory, run_id="run")
    assert result["task_recall"] == 1 / 3
    assert result["n_matched_failed_attempts"] == 1
    assert result["task_recall_scope"] == "full_selected_corpus"


# The selected exact category counts; a legacy synthetic "positive" counts only boolean fires;
# an explicit truthy aggregate counts every fire; and when an older primary selection picked
# the non-fire side, the first positive bucket is used instead.
@pytest.mark.parametrize(("scores", "labels", "matched"), [
    ({"bucket_mode": "exact"}, ("positive", "other", "negative"), 1),
    ({"buckets": [{"bucket": "positive"}, {"bucket": "other"}, {"bucket": "negative"}]},
     ("positive", "other", "negative"), 1),
    ({}, ("True", "other", "positive"), 1),
    ({"bucket_mode": "truthy"}, ("positive", "other", False), 2),
    ({"bucket": "False", "buckets": [{"bucket": "False"}, {"bucket": "bad"}]},
     ("bad", "other", False), 1),
], ids=["exact-mode", "historical-bucket-table", "legacy-synthetic-positive",
        "truthy-aggregate", "negative-primary-bucket"])
def test_task_recall_counts_only_the_selected_positive_bucket(inventory, scores, labels, matched):
    fn = recall_detector(["a1", "b1", "c1"],
                         scores={"n_fail_caught": matched, "n_flagged": matched, **scores})
    for item_id, label in zip(("a1", "b1", "c1"), labels, strict=True):
        fn.failure_localisations[item_id]["assigned_label"] = label
    result = build_task_recall(fn, inventory, run_id="run")
    assert result["task_recall"] == matched / 3
    assert result["task_recall_scope"] == "full_selected_corpus"


def test_pinned_dependency_change_invalidates_task_coverage(inventory):
    fn = recall_detector(["a1"], meta={"dependency_bundle": {
        "nodes": {"helper-ref": {"role": "helper", "source": "return 1"}},
    }})
    fn.scores.update(build_task_recall(fn, inventory, run_id="run"))
    assert fn.scores["task_recall"] == 1 / 3
    fn.meta["dependency_bundle"]["nodes"]["helper-ref"]["source"] = "return 2"
    fresh = build_task_recall(fn, inventory, run_id="run")
    assert fresh["task_recall"] is None
    view = current_task_recall_scores(
        fn.scores, source=review_source(fn), description=fn.description, mechanism=fn.mechanism,
    )
    assert view["task_recall"] is None and view["evaluation_stale"] is True
    assert fn.scores["task_recall"] == 1 / 3  # A read view never changes saved evidence.


def test_unavailable_identity_does_not_make_attempt_evaluation_stale(inventory):
    fn = recall_detector(["a1"])
    inventory["task_by_failed_item_id"]["a3"] = None
    fn.scores.update(build_task_recall(fn, inventory, run_id="run"))
    view = current_task_recall_scores(
        fn.scores, source=fn.source, description=fn.description, mechanism=fn.mechanism,
    )
    assert view["task_recall"] is None
    assert not view.get("evaluation_stale")
    assert view["recall"] == .2


def test_missing_task_provenance_does_not_rewrite_legacy_scores():
    fn = recall_detector(["a1"])
    before = deepcopy(fn.scores)
    assert current_task_recall_scores(
        fn.scores, source=fn.source, description=fn.description, mechanism=fn.mechanism,
    ) == before
    forged = {**fn.scores, "task_recall": 1, "task_recall_scope": "full_selected_corpus"}
    assert current_task_recall_scores(
        forged, source=fn.source, description=fn.description, mechanism=fn.mechanism,
    )["task_recall"] is None


# Quality task support needs verified cases and actual source identities.

def _unscored(**scores):
    return FeatureFunction(
        name="example", source="def detect(item): return True",
        description="The required operation remains unfinished",
        mechanism="Unfinished work prevents the required result", scores=scores,
    )


def supported(*, task_n=5, repeat_n=1, strictness="causal"):
    return attach_task_support(_unscored(judge_behavior_strictness=strictness),
                               task_n=task_n, repeat_n=repeat_n)


def receipt_inventory(fn):
    return {"inventory_complete": True, "corpus_fingerprint": "quality-test-corpus",
            "task_by_failed_item_id": dict(
                fn.scores["task_support"]["failed_matched_task_by_item_id"])}


@pytest.mark.parametrize("strictness", ["behavioral", "causal"])
def test_supported_tasks_are_deduplicated_and_saved_counts_cannot_override_them(strictness):
    fn = supported(task_n=5, repeat_n=3, strictness=strictness)
    assert confirmed_failed_task_support(fn.scores) == 5
    fn.scores["n_confirmed_failed_tasks"] = 999  # A saved scalar never overrides the receipt.
    assert confirmed_failed_task_support(fn.scores) == 5


@pytest.mark.parametrize("change", [
    {"task_support": None}, {"claim_hash": "different-claim"},
    {"judge_policy_hash": "old-policy"}, {"bucket": "another-category"},
    {"task_recall_corpus_fingerprint": "other-corpus"},
], ids=["no-receipt", "other-claim", "old-judge-policy", "other-category", "other-corpus"])
def test_stale_or_missing_provenance_cannot_supply_support(change):
    fn = supported()
    fn.scores.update(change)
    assert confirmed_failed_task_support(fn.scores) is None


def test_claim_or_implementation_change_invalidates_support_read_view():
    fn = supported()
    fn.source += " changed"
    view = current_behavior_scores(fn.scores, source=fn.source,
                                   description=fn.description, mechanism=fn.mechanism)
    assert confirmed_failed_task_support(view) is None
    assert build_task_support(fn)["n_confirmed_failed_tasks"] is None


@pytest.mark.parametrize("change", [
    {"verdict": "no"}, {"verification_status": "pending"}, {"snapshot_id": "other-snapshot"},
    {"claim_contract_id": "other-contract"}, {"initial_review": None}, {"assigned_label": False},
], ids=["rejected", "unverified", "other-snapshot", "other-contract", "no-initial-review",
        "unselected-label"])
def test_each_case_must_have_current_completed_confirmation(change):
    fn = supported()
    fn.scores["behavior_verdicts"][0].update(change)
    assert confirmed_failed_task_support(fn.scores) == 4


def test_conflicting_duplicate_review_does_not_count_favorable_row():
    fn = supported()
    fn.scores["behavior_verdicts"].append({**fn.scores["behavior_verdicts"][0], "verdict": "no"})
    assert confirmed_failed_task_support(fn.scores) == 4


def test_passing_unknown_and_foreign_tasks_never_count():
    fn = supported()
    source_inventory = receipt_inventory(fn)
    ids = list(source_inventory["task_by_failed_item_id"])
    # The current source says the first attempt passed and lacks task identity
    # for the second. An unrelated map entry has no matching detector record.
    source_inventory["task_by_failed_item_id"].pop(ids[0])
    source_inventory["task_by_failed_item_id"][ids[1]] = None
    source_inventory["task_by_failed_item_id"]["foreign"] = "foreign-task"
    result = build_task_support(fn, source_inventory)
    assert result["n_confirmed_failed_tasks"] == 3
    assert task_support_assessment({**fn.scores, **result})["n_matched_failed_tasks"] == 3
    assert result["task_support_comparison"] == 3


def test_match_breadth_is_independent_of_review_availability():
    fn = supported()
    fn.scores["judge_policy_hash"] = "old"
    result = task_support_assessment(fn.scores)
    assert result["n_matched_failed_tasks"] == result["task_support_comparison"] == 5
    assert result["n_confirmed_failed_tasks"] is None
    assert result["matched_task_support_status"] == "verified"


def test_breadth_receipt_requires_bound_corpus_and_valid_task_identity():
    fn = supported()
    fn.scores["task_support"]["corpus_fingerprint"] = None
    assert matched_failed_task_support(fn.scores) is None
    fn = supported()
    fn.scores["task_support"]["failed_matched_task_by_item_id"]["item-0-0"] = None
    assert matched_failed_task_support(fn.scores) is None


@pytest.mark.parametrize("label,mode,expected", [("positive", "exact", 0),
                                               (True, "legacy_boolean", 5)])
def test_receipt_uses_exact_selected_positive_bucket(label, mode, expected):
    fn = supported()
    source_inventory = receipt_inventory(fn)
    fn.scores.update(bucket="positive", bucket_mode=mode)
    for record in fn.failure_localisations.values():
        record["assigned_label"] = "other-category" if mode == "exact" else label
    assert build_task_support(fn, source_inventory)["n_confirmed_failed_tasks"] == expected


def test_only_current_localisations_establish_failed_matches():
    fn = supported()
    source_inventory = receipt_inventory(fn)
    rows = list(fn.failure_localisations.values())
    rows[0]["source_sha256"] = "old-source"
    rows[1]["item_id"] = "wrong-item"
    rows[2]["assigned_label"] = False
    assert build_task_support(fn, source_inventory)["n_confirmed_failed_tasks"] == 2


def historical(fn):
    saved = fn.scores.pop("task_support")
    groups = {item_id: json.dumps(["bench", task]) for item_id, task in
              saved["failed_matched_task_by_item_id"].items()}
    fn.scores.update(bucket="True", buckets=[{"bucket": "True", "example_fail_ids": list(groups)}])
    return {"schema_version": 1, "history_complete": True,
            "detectors": {fn.name: {"item_task_groups": groups}}}


def test_legacy_failed_examples_do_not_infer_task_identity_from_item_id():
    fn = supported()
    history = historical(fn)
    history["detectors"][fn.name]["item_task_groups"] = {}
    assert build_task_support(fn, review_history=history)["n_confirmed_failed_tasks"] is None
    assert build_task_support(fn, review_history={})["n_confirmed_failed_tasks"] is None


def test_legacy_receipt_counts_distinct_tasks_in_selected_bucket_only():
    fn = supported(repeat_n=2)
    history = historical(fn)
    # Two attempts of task 0 count once.
    fn.scores["buckets"][0]["example_fail_ids"] = ["item-0-0", "item-0-1", "item-1-0"]
    fn.scores["buckets"].append({"bucket": "other", "example_fail_ids": ["item-2-0"]})
    result = build_task_support(fn, review_history=history)
    assert result["n_confirmed_failed_tasks"] == 2


def test_full_receipt_is_not_replaced_by_incomplete_portable_history():
    fn = supported()
    receipt = deepcopy(fn.scores["task_support"])
    assert build_task_support(fn, review_history={})["task_support"] == receipt


def test_another_valid_contract_cannot_be_substituted():
    fn, other = supported(), supported(strictness="behavioral")
    fn.scores["claim_contract"] = other.scores["claim_contract"]
    assert confirmed_failed_task_support(fn.scores) is None


def test_sample_rescore_preserves_outside_panel_support_and_frozen_sample_recall():
    fn = supported(task_n=5)
    source_inventory = receipt_inventory(fn)
    fn.scores.update(sampled=True, evaluation_item_ids=["item-0-0", "item-1-0"],
                     task_recall=.05, n_matched_failed_tasks=1, n_failed_tasks=20)
    # Five cases were reviewed earlier. The new panel evaluates two, and only
    # the second still fires. The three cases outside it were not contradicted.
    fn.failure_localisations = {"item-1-0": fn.failure_localisations["item-1-0"]}
    result = build_task_support(fn, source_inventory)
    assert result["n_confirmed_failed_tasks"] == 4
    assert "item-0-0" not in result["task_support"]["failed_matched_task_by_item_id"]
    assert "n_matched_failed_tasks" not in result
    scores = {**fn.scores, **result}
    assert scores["n_matched_failed_tasks"] == 1
    assert scores["task_recall"] == scores["n_matched_failed_tasks"] / scores["n_failed_tasks"]
    # The three unchanged matches outside the panel still supply known breadth.
    assert task_support_assessment(scores)["n_matched_failed_tasks"] == 4
    assert scores["task_support_comparison"] == 4


@pytest.mark.parametrize("change,source_change", [
    ({"sampled": False}, {}), ({"evaluation_item_ids": None}, {}),
    # A changed source corpus cannot retain unchecked matches either.
    ({"task_recall_corpus_fingerprint": "new-corpus"}, {"corpus_fingerprint": "new-corpus"}),
], ids=["full-rescore", "unknown-panel", "changed-corpus"])
def test_full_rescore_or_unknown_panel_cannot_retain_old_unchecked_matches(change, source_change):
    fn = supported()
    source_inventory = {**receipt_inventory(fn), **source_change}
    fn.scores.update({"sampled": True, "evaluation_item_ids": ["item-0-0"], **change})
    fn.failure_localisations = {}
    assert build_task_support(fn, source_inventory)["n_confirmed_failed_tasks"] == 0


@pytest.mark.parametrize("change_policy", [False, True])
def test_mcp_views_recover_legacy_support_consistently_without_mutation(
    tmp_path, monkeypatch, change_policy,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = _function("good")
    rt.library.functions = [fn]
    rt.library.review_history = historical(fn)
    if change_policy:
        rt.config.behavior_strictness = "contextual"
    before = deepcopy(rt.library.to_dict())
    listed = tools_library.list_library_functions()["functions"][0]
    inspected = tools_library.get_library_function(fn.name)
    feedback = tools_library.get_function_feedback(fn.name)
    expected = "UNSCORED" if change_policy else "GOOD"
    support = None if change_policy else 5
    status = "unavailable" if change_policy else "verified"
    assert listed["verdict"]["code"] == expected
    assert inspected["verdict"]["code"] == expected
    assert feedback["verdict"]["code"] == expected
    assert tools_eval._current_verdict(rt, fn)["code"] == expected
    for scores in (listed["scores"], inspected["function"]["scores"], feedback["feedback"]):
        assert scores["n_confirmed_failed_tasks"] == support
        assert scores["confirmed_task_support_status"] == status
    assert rt.library.to_dict() == before


# Sampled-support extrapolation respects the actual probability panel and task identities.

def _case(population, panel, matches):
    """population maps opaque item IDs to (source task, source benchmark)."""
    fn = attach_task_support(_unscored(), task_n=1)
    row = fn.scores["behavior_verdicts"][0]
    localisation = fn.failure_localisations["item-0-0"]
    fn.failure_localisations = {
        item_id: {**localisation, "item_id": item_id} for item_id in matches
    }
    fn.scores["behavior_verdicts"] = [{
        **row, "item_id": item_id, "snapshot_id": f"snapshot-{item_id}",
        "initial_review": {**row["initial_review"], "item_id": item_id,
                           "snapshot_id": f"snapshot-{item_id}"},
    } for item_id in matches]
    item_ids = [*population, "pass"]
    selected = [*panel, "pass"]
    inventory = {
        "inventory_complete": True, "corpus_fingerprint": "quality-test-corpus",
        "item_ids": item_ids, "n_items": len(item_ids),
        "n_failed_attempts": len(population),
        "n_failed_tasks": len({task for task, _ in population.values()}),
        "n_passed_attempts": 1, "n_unknown_outcomes": 0,
        "task_by_failed_item_id": {item: task for item, (task, _) in population.items()},
        "identity_by_item_id": {
            item: {"_item_id": item, "benchmark": benchmark}
            for item, (_, benchmark) in population.items()
        },
    }
    inventory["identity_by_item_id"]["pass"] = {"_item_id": "pass", "benchmark": "bench"}
    fn.scores.update(
        sampled=True, sample_reason="llm_detector", sample_n=len(selected),
        sample_requested_n=len(selected),
        sample_frame_n=len(item_ids), evaluation_item_ids=selected,
        evaluation_scope_complete=True, eval_truncated=False,
        n_scored=len(selected), n_unresolved=0,
    )
    fn.scores.update(build_task_support(fn, inventory))
    return fn, inventory


def _simple():
    population = {f"opaque-{index}": (f"bench:task-{index}", "bench") for index in range(20)}
    return _case(population, list(population)[:5], list(population)[:2])


def test_repeated_attempts_change_inclusion_probability_and_count_only_once():
    population = {f"repeat-{i}": ("task-A", "bench") for i in range(3)}
    population.update({f"single-{i}": (f"task-{i}", "bench") for i in range(7)})
    panel = ["repeat-0", "repeat-1", "single-0"]
    fn, _ = _case(population, panel, panel)
    p_repeated_task = 1 - comb(7, 3) / comb(10, 3)
    expected = 1 / p_repeated_task + 10 / 3
    result = task_support_assessment(fn.scores)
    assert result["n_confirmed_failed_tasks"] == 2
    assert result["n_sampled_matched_failed_tasks"] == 2
    assert result["estimated_n_confirmed_failed_tasks"] == pytest.approx(expected)
    assert result["estimated_n_confirmed_failed_tasks"] < 5


def test_benchmarks_use_their_own_stratified_sampling_rates():
    population = {f"A{i}": (f"A:{i}", "A") for i in range(20)}
    population.update({f"B{i}": (f"B:{i}", "B") for i in range(10)})
    fn, _ = _case(population, ["A0", "A1", *[f"B{i}" for i in range(5)]], ["A0", "B0"])
    estimate = task_support_assessment(fn.scores)["estimated_n_confirmed_failed_tasks"]
    assert estimate == pytest.approx(20 / 2 + 10 / 5)


def test_projection_cannot_exceed_each_benchmarks_actual_failed_task_count():
    population = {"rare": ("A:rare", "A")}
    population.update({f"repeat-{i}": ("A:repeated", "A") for i in range(9)})
    population.update({f"B{i}": (f"B:{i}", "B") for i in range(10)})
    fn, _ = _case(population, ["rare", "B0", "B1"], ["rare", "B0"])
    # The rare A task has inverse probability 10, but A contains only two tasks.
    estimate = task_support_assessment(fn.scores)["estimated_n_confirmed_failed_tasks"]
    assert estimate == pytest.approx(2 + 5)


def test_unresolved_sampled_attempts_remain_in_the_sampling_denominator():
    fn, inventory = _simple()
    fn.scores.update(n_scored=5, n_unresolved=1, n_insufficient=1, evaluation_complete=False)
    result = build_task_support(fn, inventory)
    assert result["estimated_n_confirmed_failed_tasks"] == pytest.approx(8)
    assert result["n_sampled_failed_tasks"] == 5


def test_outside_panel_confirmations_are_not_extrapolated():
    fn, inventory = _simple()
    # A later targeted probe confirms a third task outside the random panel.
    row = deepcopy(fn.scores["behavior_verdicts"][0])
    row.update(item_id="opaque-9", snapshot_id="snapshot-outside")
    row["initial_review"].update(item_id="opaque-9", snapshot_id="snapshot-outside")
    fn.scores["behavior_verdicts"].append(row)
    fn.failure_localisations["opaque-9"] = {
        **fn.failure_localisations["opaque-0"], "item_id": "opaque-9",
    }
    result = build_task_support(fn, inventory)
    assert result["n_confirmed_failed_tasks"] == 3
    assert result["n_sampled_confirmed_failed_tasks"] == 2
    assert result["estimated_n_confirmed_failed_tasks"] == pytest.approx(8)


def test_outside_panel_review_of_same_task_does_not_confirm_sampled_match():
    population = {f"i{i}": (f"task-{i // 2}", "bench") for i in range(20)}
    fn, inventory = _case(population, ["i0", "i2", "i4"], ["i0", "i1", "i2"])
    fn.scores["behavior_verdicts"] = [row for row in fn.scores["behavior_verdicts"]
                                       if row["item_id"] != "i0"]
    result = build_task_support(fn, inventory)
    assert result["n_confirmed_failed_tasks"] == 2
    assert result["n_sampled_confirmed_failed_tasks"] == 1


def test_only_confirmed_sample_matches_count_but_all_matches_remain_visible():
    fn, inventory = _simple()
    fn.scores["behavior_verdicts"][0]["verdict"] = "no"
    result = build_task_support(fn, inventory)
    assert result["estimated_n_confirmed_failed_tasks"] == pytest.approx(4)
    assert result["estimated_n_matched_failed_tasks"] == pytest.approx(8)
    assert result["n_confirmed_failed_tasks"] == 1
    assert task_support_assessment({**fn.scores, **result})["n_matched_failed_tasks"] == 2
    assert result["task_support_comparison"] == pytest.approx(8)
    assert result["task_support_comparison_basis"] == "estimated"


def test_sampled_match_breadth_is_projected_without_claiming_unreviewed_confirmations():
    fn, inventory = _simple()
    fn.scores["behavior_verdicts"] = []
    result = build_task_support(fn, inventory)
    assert task_support_assessment({**fn.scores, **result})["n_matched_failed_tasks"] == 2
    assert result["n_confirmed_failed_tasks"] is None
    assert result["estimated_n_confirmed_failed_tasks"] is None
    assert result["estimated_n_matched_failed_tasks"] == pytest.approx(8)
    assert result["task_support_comparison"] == pytest.approx(8)
    assert result["task_support_estimate_status"] == "provisional"


def test_single_failed_task_rollouts_do_not_extrapolate_as_independent_tasks():
    population = {f"i{i}": ("one-task", "bench") for i in range(20)}
    fn, _ = _case(population, ["i0", "i1", "i2"], ["i0", "i1", "i2"])
    result = task_support_assessment(fn.scores)
    assert result["n_matched_failed_tasks"] == 1
    assert result["estimated_n_matched_failed_tasks"] == pytest.approx(1)
    assert result["task_support_comparison"] == 1


@pytest.mark.parametrize("key,value", [
    ("evaluation_stale", True), ("evaluation_claim_hash", "other"),
    ("task_recall_corpus_fingerprint", "other"), ("bucket", "other"),
])
def test_changed_match_receipt_disables_projection_and_observed_breadth(key, value):
    fn, _ = _simple()
    fn.scores.update({key: value, "estimated_n_matched_failed_tasks": 999,
                      "n_matched_failed_tasks": 999})
    result = task_support_assessment(fn.scores)
    assert matched_failed_task_support(fn.scores) is None
    assert result["n_matched_failed_tasks"] is None
    assert result["estimated_n_matched_failed_tasks"] is None
    assert result["task_support_comparison"] is None


@pytest.mark.parametrize("change", [
    {"sampled": False}, {"sample_reason": "manually_picked"}, {"n_scored": 3},
    {"behavior_judgment_stale": True},
    {"evaluation_item_ids": ["opaque-0", "opaque-1", "opaque-2", "opaque-3", "opaque-9", "pass"]},
], ids=["unsampled", "not-random", "count-mismatch", "stale-review", "changed-panel"])
def test_stale_or_changed_sampling_provenance_cannot_promote(change):
    fn, _ = _simple()
    fn.scores.update(change)
    fn.scores["estimated_n_confirmed_failed_tasks"] = 999
    assert task_support_assessment(fn.scores)["estimated_n_confirmed_failed_tasks"] is None


@pytest.mark.parametrize("change", [
    {"n_unknown_outcomes": 1}, {"n_failed_tasks": 99}, {"identity_by_item_id": {}},
], ids=["unknown-outcomes", "inconsistent-task-count", "missing-identities"])
def test_incomplete_source_inventory_disables_projection(change):
    fn, inventory = _simple()
    inventory.update(change)
    result = build_task_support(fn, inventory)
    assert result["estimated_n_confirmed_failed_tasks"] is None


def test_unknown_task_identity_or_unrepresented_failed_benchmark_disables_projection():
    fn, inventory = _simple()
    inventory["task_by_failed_item_id"]["opaque-19"] = None
    assert build_task_support(fn, inventory)["estimated_n_confirmed_failed_tasks"] is None
    fn, inventory = _simple()
    inventory["identity_by_item_id"]["opaque-19"]["benchmark"] = "unsampled-benchmark"
    assert build_task_support(fn, inventory)["estimated_n_confirmed_failed_tasks"] is None


def test_valid_saved_receipt_survives_read_only_views_without_inventory():
    fn, _ = _simple()
    before = deepcopy(fn.to_dict())
    assert build_task_support(fn)["estimated_n_confirmed_failed_tasks"] == pytest.approx(8)
    assert fn.to_dict() == before


def test_unknown_original_allocation_disables_projection():
    fn, inventory = _simple()
    fn.scores.pop("sample_requested_n")
    assert build_task_support(fn, inventory)["estimated_n_confirmed_failed_tasks"] is None


def test_legacy_actual_support_remains_available_without_inventing_projection():
    fn = attach_task_support(_unscored(), task_n=3)
    fn.scores.update(sampled=True, estimated_n_confirmed_failed_tasks=99)
    result = task_support_assessment(fn.scores)
    assert result["n_confirmed_failed_tasks"] == 3
    assert result["estimated_n_confirmed_failed_tasks"] is None
    assert result["task_support_comparison"] == 3
    assert result["task_support_comparison_basis"] == "observed"


def test_malformed_portable_sampling_design_is_unavailable_without_raising():
    fn, _ = _simple()
    task = next(iter(fn.scores["task_support"]["sampling_design"]["task_population"].values()))
    task["stratum"] = ["bench"]
    assert task_support_assessment(fn.scores)["estimated_n_confirmed_failed_tasks"] is None


def test_projection_never_demotes_or_discards_observed_confirmations():
    # A random panel of ten without matches projects zero, yet the five observed
    # confirmations outside it remain the comparison.
    population = {f"i{i}": (f"task-{i}", "bench") for i in range(20)}
    fn, _ = _case(population, [f"i{i}" for i in range(10)], [f"i{i}" for i in range(10, 15)])
    result = task_support_assessment(fn.scores)
    assert result["n_confirmed_failed_tasks"] == 5
    assert result["estimated_n_confirmed_failed_tasks"] == 0
    assert result["task_support_comparison"] == 5
    assert result["task_support_comparison_basis"] == "observed"


def test_full_corpus_scan_compares_actual_support_only():
    fn, inventory = _simple()
    fn.scores.update(sampled=False, n_scored=21)
    fn.scores.update(build_task_support(fn, inventory))
    result = task_support_assessment(fn.scores)
    assert result["estimated_n_confirmed_failed_tasks"] is None
    assert result["task_support_comparison"] == 2
    assert result["task_support_comparison_basis"] == "observed"


# Breadth counts matched source tasks, not the smaller judge-review subset. The
# aggregate shape mirrors a published regression: 36 failed matches on 23 tasks,
# 11 passed matches, and two positive judge reviews. All item identities, claims,
# and detector code are synthetic; no trace evidence is retained.

# These source tasks contributed 36 matched failed attempts in total.
PUBLISHED_TASK_MULTIPLICITIES = (1, 1, 2, 1, 3, 1, 1, 2, 2, 2, 1, 3,
                                 1, 1, 1, 1, 2, 2, 3, 2, 1, 1, 1)


def _reviewed_subset_fixture(multiplicities=PUBLISHED_TASK_MULTIPLICITIES, *, pass_n=11):
    fn = _unscored(
        bucket="True", bucket_mode="exact", sampled=False,
        eval_mode="full", evaluation_scope_complete=True, eval_truncated=False,
        n_items=1281.0, n_scored=1281, n_insufficient=81, n_unresolved=81,
        evaluation_complete=False, errors=0,
        recall=36 / 136, ppv_at_pi0=0.9219276533563954,
        behavior_purity=1.0, behavior_n_judged=2, behavior_n_yes=2,
        behavior_n_no=0, behavior_n_uncertain=0, behavior_n_completed=2,
        behavior_n_errors=0, behavior_n_review_incomplete=0,
    )
    attach_task_support(fn, task_n=len(multiplicities))
    # Judging a representative subset does not reduce the detector's matched-task breadth.
    fn.scores["behavior_verdicts"] = fn.scores["behavior_verdicts"][:2]
    failed = {f"item-{task}-{repeat}": f"task-{task}"
              for task, count in enumerate(multiplicities) for repeat in range(count)}
    passes = {f"passing-{index}": f"passed-task-{index}" for index in range(pass_n)}
    other_bucket = {f"nonmatch-{index}": f"unflagged-failed-task-{index}" for index in range(3)}
    source_hash = text_hash(review_source(fn))
    fn.failure_localisations = {
        item_id: {"item_id": item_id, "source_sha256": source_hash,
                  "assigned_label": item_id not in other_bucket}
        for item_id in {**failed, **passes, **other_bucket}
    }
    fn.scores.update(
        a=float(len(failed)), b=float(pass_n), c=100.0, d=1134.0,
        n_flagged=float(len(failed) + pass_n),
        n_matched_failed_tasks=len(multiplicities), n_matched_failed_attempts=len(failed),
        buckets=[{"bucket": "True", "bucket_mode": "exact",
                  "example_fail_ids": list(failed), "example_pass_ids": list(passes)}],
    )
    fn.scores.update(build_task_support(fn, {
        "inventory_complete": True, "corpus_fingerprint": "quality-test-corpus",
        "task_by_failed_item_id": {**failed, **other_bucket},
        "task_by_item_id": {**failed, **other_bucket, **passes},
    }))
    return fn


def test_full_scan_23_matched_tasks_and_two_reviewed_tasks_qualifies_as_good():
    fn = _reviewed_subset_fixture()
    before = deepcopy(fn.to_dict())

    support = task_support_assessment(fn.scores)
    receipt = fn.scores["task_support"]["failed_matched_task_by_item_id"]

    assert len(receipt) == 36
    assert len(set(receipt.values())) == 23
    assert support["n_matched_failed_tasks"] == 23
    assert support["n_confirmed_failed_tasks"] == 2
    assert support["task_support_comparison"] == 23
    assert support["task_support_comparison_basis"] == "observed"
    assert verdict_for_scores(fn.scores)["code"] == "GOOD"
    assert fn.to_dict() == before


def test_many_matched_repeats_do_not_satisfy_five_distinct_failed_tasks():
    fn = _reviewed_subset_fixture((12, 12, 12))
    support = task_support_assessment(fn.scores)
    assert len(fn.scores["task_support"]["failed_matched_task_by_item_id"]) == 36
    assert support["n_confirmed_failed_tasks"] == 2
    assert support["n_matched_failed_tasks"] == support["task_support_comparison"] == 3
    assert verdict_for_scores(fn.scores)["code"] == "OK"
    # Saved scalar counts cannot override the source-verified breadth.
    fn.scores["n_matched_failed_tasks"] = 1000
    fn.scores["n_confirmed_failed_tasks"] = 1000
    assert task_support_assessment(fn.scores)["task_support_comparison"] == 3
    assert verdict_for_scores(fn.scores)["code"] == "OK"


def test_breadth_does_not_replace_the_other_good_quality_requirements():
    fn = _reviewed_subset_fixture()
    fn.scores["behavior_purity"] = 0.5
    assert task_support_assessment(fn.scores)["task_support_comparison"] == 23
    assert verdict_for_scores(fn.scores)["code"] == "OK"


@pytest.mark.parametrize("state,observed,confirmed,comparison,coverage,agent_grade", [
    ("sampled_matches_without_confirmations", 3, 0, 12, .13, "EXCELLENT"),
    ("full_matches_with_two_confirmations", 15, 2, 15, .21, "OUTSTANDING"),
])
def test_matched_breadth_agrees_across_agent_report_and_actual_stopping(
    tmp_path, monkeypatch, state, observed, confirmed, comparison, coverage, agent_grade,
):
    rt, _ = _runtime(tmp_path, monkeypatch)
    fn = _function(state)
    fn.scores["recall"] = coverage
    rt.library.functions = [fn]
    before = deepcopy(fn.to_dict())

    support = task_support_assessment(fn.scores)
    assert support["n_matched_failed_tasks"] == observed
    assert support["n_confirmed_failed_tasks"] == confirmed
    assert support["task_support_comparison"] == pytest.approx(comparison)

    for verdict in (
        verdict_for_scores(fn.scores),
        tools_eval._current_verdict(rt, fn),
        tools_library.get_library_function(fn.name)["verdict"],
        tools_library.get_function_feedback(fn.name)["verdict"],
        tools_library.list_library_functions()["functions"][0]["verdict"],
    ):
        assert verdict["code"] == "GOOD", verdict
    assert tools_eval._current_discovery_grade(rt, fn)["code"] == agent_grade
    grades = grade_summary([fn], behavior_strictness="causal")
    assert grades["n_good_or_better"] == 1
    assert grades["functions"][0]["code"] == agent_grade
    target = discovery_good_target([fn], minimum=1, config=Config(behavior_strictness="causal"))
    assert target["minimum_met"]
    assert target["n_qualifying_failure_modes"] == 1
    rt.run.min_failure_modes = 1
    rt.run.round = rt.run.max_rounds = 8
    rt.config.discovery_wait_for_plateau = False
    status = tools_library.pareto_status()
    assert status["goal_met"]  # The authoritative driver completes on Good.
    assert status["minimum_met"] is (agent_grade == "OUTSTANDING")
    assert "final_validation_target" not in status

    report = build_data(FunctionLibrary(functions=[fn]))
    assert report["functions"][0]["verdict"]["code"] == "GOOD"
    card = _render_card(report)  # Execute the actual emitted report JavaScript in Node.
    assert "GOOD" in card
    assert "EXCELLENT" not in card and "OUTSTANDING" not in card
    assert fn.to_dict() == before


# Library task-search coverage rewards actual, current cross-task causal support.

def coverage_runtime(*functions, tasks=None):
    failed = {f"f{i}": f"task-{i}" for i in range(10)}
    failed.update(tasks or {})
    passed = {f"p{i}": f"passing-{i}" for i in range(10)}
    all_items = {**failed, **passed}
    inventory = {
        "inventory_complete": True, "corpus_fingerprint": "corpus-1", "n_items": 20,
        "n_failed_attempts": 10, "n_passed_attempts": 10, "n_unknown_outcomes": 0,
        "task_by_failed_item_id": failed, "task_by_item_id": all_items,
        "identity_by_item_id": {key: {"_item_id": key} for key in all_items},
    }
    return SimpleNamespace(library=FunctionLibrary(functions=list(functions)),
                           config=Config(behavior_strictness="causal"),
                           run=SimpleNamespace(run_id="run-1"),
                           _family_failed_task_inventory=inventory)


def detector(name="parent", *, confirmed=("f0", "f1"), matches=None, passes=(), **changes):
    matches = confirmed if matches is None else matches
    fn = FeatureFunction(name=name, description=f"Behavior {name}", mechanism=f"Cause {name}",
                         source=f"def {name}(item): return True")
    contract = build_claim_contract(source=fn.source, description=fn.description,
                                    mechanism=fn.mechanism)
    fingerprint = contract["contract_id"]
    fn.scores = {
        **contingency(len(matches), len(passes), 10-len(matches), 10-len(passes)),
        "evaluated": True, "n_scored": 20, "eval_mode": "full", "sampled": False,
        "evaluation_scope_complete": True, "evaluation_claim_hash": fingerprint,
        "evaluation_run_id": "run-1", "judge_backend": "claude_agent_sdk",
        "judge_contract_version": JUDGE_CONTRACT_VERSION, "judge_policy_hash": judge_policy_hash(),
        "claim_hash": fingerprint, "claim_contract": contract,
        "behavior_n_judged": len(confirmed), "behavior_n_completed": len(confirmed),
        "behavior_n_yes": len(confirmed), "behavior_n_no": 0, "behavior_n_uncertain": 0,
        "behavior_n_errors": 0, "behavior_purity": 1.0, **changes,
    }
    for item_id in (*matches, *passes):
        fn.failure_localisations[item_id] = {
            "item_id": item_id, "source_sha256": text_hash(fn.source), "assigned_label": True,
        }
    fn.scores["behavior_verdicts"] = []
    for item_id in confirmed:
        initial = {"item_id": item_id, "snapshot_id": f"snapshot-{item_id}", "verdict": "yes",
                   "claim_contract_id": fingerprint, "completion_status": "complete"}
        fn.scores["behavior_verdicts"].append({
            **initial, "verification_status": "confirmed", "initial_review": dict(initial),
        })
    return fn


def sampled(fn, *, abstain=False):
    """A balanced half-corpus sample: each actual row has population weight two."""
    fn.scores.update(
        **contingency(4, 0, 4 if abstain else 6, 10), n_scored=9 if abstain else 10,
        eval_mode="capped", sampled=True, sample_n=10, sample_frame_n=20,
        evaluation_item_ids=[f"{outcome}{i}" for outcome in ("f", "p") for i in range(5)],
    )
    if abstain:
        fn.scores.update(n_insufficient=1, n_unresolved=1, evaluation_complete=False,
                         unresolved_bounds={"recall": [.4, .6], "fpr": [0, 0]})
    return fn


def test_overlap_repeated_attempts_and_parents_are_counted_once_without_mutation():
    first = detector(confirmed=("f0", "f1", "f2"))
    child = detector("child", confirmed=("f0", "f1"))
    child.explains = first.name
    other = detector("other", confirmed=("f2", "f3"))
    rt = coverage_runtime(first, child, other, tasks={"f1": "task-0"})
    before = deepcopy(rt.library.to_dict())
    result = task_coverage_snapshot(rt, include_task_ids=True)
    assert result["available"]
    assert result["confirmed_failed_task_ids"] == ["task-0", "task-2", "task-3"]
    assert result["n_confirmed_failed_tasks"] == 3
    assert result["confirmed_task_coverage"] == 3 / 9
    rows = {row["name"]: row for row in result["detectors"]}
    assert rows["parent"]["eligible"] and rows["other"]["eligible"]
    assert not rows["child"]["eligible"]
    assert rows["parent"]["n_exclusive_confirmed_failed_tasks"] == 1
    assert rows["other"]["n_exclusive_confirmed_failed_tasks"] == 1
    assert rt.library.to_dict() == before
    assert "confirmed_failed_task_ids" not in task_coverage_snapshot(rt)


def test_completed_weighted_sample_and_development_review_supply_observed_coverage():
    fn = sampled(detector())
    fn.scores["behavior_sample"] = {"limited": True, "reused_n": 2, "actual_n": 2}
    result = task_coverage_snapshot(coverage_runtime(fn))
    assert result["n_confirmed_failed_tasks"] == 2
    assert result["confirmed_task_coverage"] == .2  # Never extrapolate 2 sampled tasks to 4.
    row = result["detectors"][0]
    assert row["eligible"] and row["evaluation_sampled"]
    assert row["selected_evaluation_complete"] and row["review_complete"]
    assert not row["corpus_evaluation_complete"]


def test_search_progress_uses_good_precision_floor_without_extra_tightening():
    fn = detector(matches=tuple(f"f{i}" for i in range(6)), passes=("p0",))
    assert .70 < fn.scores["ppv_at_pi0"] < .80
    result = task_coverage_snapshot(coverage_runtime(fn))
    assert result["n_confirmed_failed_tasks"] == 2
    assert result["detectors"][0]["eligible"]


def test_unsampled_capped_evaluation_needs_census_but_full_scan_does_not():
    fn = detector()
    fn.scores.update(**contingency(2, 0, 3, 5), n_scored=10,
                     eval_mode="capped", evaluation_scope_complete=False)
    row = task_coverage_snapshot(coverage_runtime(fn))["detectors"][0]
    assert row["eligible"] and row["selected_evaluation_complete"]
    assert not row["evaluation_sampled"] and not row["corpus_evaluation_complete"]
    fn.scores.update(**contingency(2, 0, 8, 10), n_scored=20,
                     eval_mode="full", evaluation_scope_complete=True)
    row = task_coverage_snapshot(coverage_runtime(fn))["detectors"][0]
    assert row["eligible"] and row["corpus_evaluation_complete"]


def test_confirmation_review_outside_scoring_panel_can_add_observed_tasks():
    fn = sampled(detector(confirmed=("f0", "f1", "f9")))
    result = task_coverage_snapshot(coverage_runtime(fn))
    assert result["n_confirmed_failed_tasks"] == 3
    assert result["detectors"][0]["eligible"]


@pytest.mark.parametrize("sample", [False, True])
def test_honest_abstentions_keep_current_supported_tasks(sample):
    fn = detector()
    if sample:
        sampled(fn, abstain=True)
    else:
        fn.scores.update(**contingency(2, 0, 6, 10), n_scored=18,
                         n_insufficient=2, n_unresolved=2, evaluation_complete=False)
    assert task_coverage_snapshot(coverage_runtime(fn))["n_confirmed_failed_tasks"] == 2


def test_abstentions_use_full_outcome_denominators_for_quality_gate():
    # Resolved-only precision appears high because most failures abstained.
    fn = detector(passes=("p0",))
    fn.scores.update(**contingency(2, 1, 0, 9), n_scored=12, n_insufficient=8,
                     n_unresolved=8, evaluation_complete=False)
    assert fn.scores["ppv_at_pi0"] > .8
    result = task_coverage_snapshot(coverage_runtime(fn))
    assert result["n_confirmed_failed_tasks"] == 0
    assert "precision" in result["detectors"][0]["reason"]


def test_intentionally_capped_frame_with_abstentions_uses_saved_weighted_bounds():
    fn = sampled(detector(), abstain=True)
    fn.scores.update(**contingency(2.4, 0, 2.4, 6), sample_frame_n=12,
                     evaluation_scope_complete=False, eval_truncated=True,
                     unresolved_bounds={"recall": [.4, .6], "fpr": [0, 0]})
    assert task_coverage_snapshot(coverage_runtime(fn))["n_confirmed_failed_tasks"] == 2
    fn.scores.pop("unresolved_bounds")
    assert task_coverage_snapshot(coverage_runtime(fn))["n_confirmed_failed_tasks"] == 0


@pytest.mark.parametrize("changes", [
    {"errors": 1}, {"evaluation_run_id": "another-run"}, {"judge_policy_hash": "old"},
    {"behavior_n_uncertain": 1}, {"n_flagged": 999}, {"n_scored": 21},
], ids=["errors", "other-run", "old-judge-policy", "review-incomplete",
        "inconsistent-counts", "more-scored-than-corpus"])
def test_stale_malformed_incomplete_evidence_cannot_supply_progress(changes):
    result = task_coverage_snapshot(coverage_runtime(detector(**changes)))
    assert result["available"]
    assert result["n_confirmed_failed_tasks"] == 0
    assert result["detectors"][0]["reason"]


@pytest.mark.parametrize("change", ["claim", "snapshot", "task"])
def test_source_and_review_provenance_cannot_invent_confirmed_tasks(change):
    fn = detector()
    rt = coverage_runtime(fn)
    if change == "claim":
        fn.mechanism += " Changed"
    elif change == "snapshot":
        fn.scores["behavior_verdicts"][0]["initial_review"]["snapshot_id"] = "other"
    else:
        rt._family_failed_task_inventory["task_by_failed_item_id"]["f0"] = None
    assert task_coverage_snapshot(rt)["n_confirmed_failed_tasks"] == 0


def test_low_purity_does_not_contribute_even_with_several_confirmed_tasks():
    fn = detector(confirmed=("f0", "f1", "f2"))
    fn.scores["behavior_verdicts"][-1].update(verdict="no", verification_status="rejected")
    fn.scores.update(behavior_n_yes=2, behavior_n_no=1, behavior_purity=2/3)
    row = task_coverage_snapshot(coverage_runtime(fn))["detectors"][0]
    assert row["review_complete"] and not row["eligible"]
    assert "adherence" in row["reason"]


def test_sample_completion_needs_actual_source_verified_selected_rows():
    fn = sampled(detector())
    fn.scores["evaluation_item_ids"][-1] = "f0"  # A duplicate replaces a selected row.
    assert task_coverage_snapshot(coverage_runtime(fn))["n_confirmed_failed_tasks"] == 0


def test_no_confirmations_means_zero_coverage_and_missing_source_is_unavailable():
    assert task_coverage_snapshot(coverage_runtime())["n_confirmed_failed_tasks"] == 0
    rt = coverage_runtime(detector())
    rt._family_failed_task_inventory["inventory_complete"] = False
    result = task_coverage_snapshot(rt, include_task_ids=True)
    assert not result["available"] and result["confirmed_failed_task_ids"] == []


def test_unknown_outcomes_do_not_prevent_search_of_known_failed_tasks():
    rt = coverage_runtime()
    inventory = rt._family_failed_task_inventory
    inventory["task_by_item_id"]["unknown"] = "unknown-task"
    inventory.update(n_items=21, n_unknown_outcomes=1, unknown_outcome_item_ids=["unknown"])
    result = task_coverage_snapshot(rt)
    assert result["available"] and result["n_failed_tasks"] == 10
    assert result["n_unknown_outcomes"] == 1


def test_live_prompt_reports_the_current_confirmed_task_union():
    prompt = task_coverage_prompt(task_coverage_snapshot(coverage_runtime(detector())))
    assert "2/10 failed tasks" in prompt


@pytest.mark.parametrize("second_tasks,net_growth", [
    (("f0", "f1", "f2", "f3", "f4"), 0),  # Useful subset.
    (("f3", "f4", "f5", "f6", "f7"), 2),  # Partial overlap adds two tasks.
])
def test_useful_detector_feedback_keeps_overlap_separate_from_union_growth(
    second_tasks, net_growth,
):
    parent = detector(confirmed=tuple(f"f{i}" for i in range(6)))
    child = detector("specific_decision", confirmed=second_tasks)
    rt = coverage_runtime(parent)
    for fn in (parent, child):
        fn.scores["task_recall_corpus_fingerprint"] = "corpus-1"
        fn.scores.update(build_task_support(fn, rt._family_failed_task_inventory))
    previous = grade_summary(rt.library.functions, behavior_strictness="causal")
    baseline = discovery_grade_assessment(
        previous, None, minimum=3, task_coverage=task_coverage_snapshot(rt),
    )
    previous["task_breadth"] = baseline["task_breadth"]
    rt.library.functions.append(child)
    current = grade_summary(rt.library.functions, behavior_strictness="causal")
    assert current["n_good_or_better"] == 2

    feedback = discovery_grade_assessment(
        current, previous, minimum=3, task_coverage=task_coverage_snapshot(rt),
    )

    assert feedback["encouraging"]
    assert feedback["confirmed_task_net_change"] == net_growth
    assert feedback["task_breadth"]["n_confirmed_failed_tasks"] == 6 + net_growth


@pytest.mark.parametrize("change", [
    {"available": False}, {"corpus_fingerprint": "other"},
])
def test_breadth_feedback_does_not_compare_unavailable_or_incompatible_unions(change):
    rt = coverage_runtime(detector())
    summary = grade_summary(rt.library.functions, behavior_strictness="causal")
    coverage = task_coverage_snapshot(rt)
    baseline = discovery_grade_assessment(summary, None, minimum=3, task_coverage=coverage)
    previous = {**summary, "task_breadth": baseline["task_breadth"]}
    feedback = discovery_grade_assessment(
        summary, previous, minimum=3, task_coverage={**coverage, **change},
    )
    assert feedback["confirmed_task_net_change"] is None


# Task union growth, sample completion, and durable shared progress accounting.

@pytest.fixture
def runtime(tmp_path):
    return SimpleNamespace(run=SimpleNamespace(run_dir=lambda: tmp_path), scores_history=[])


REVIEWED_PARENT = {"name": "parent", "current_claim_hash": "current",
                   "selected_evaluation_complete": True, "review_complete": True}


def coverage(*tasks, fingerprint="corpus", detectors=None):
    return {"available": True, "corpus_fingerprint": fingerprint,
            "n_failed_tasks": 10, "n_confirmed_failed_tasks": len(set(tasks)),
            "confirmed_task_coverage": len(set(tasks)) / 10,
            "confirmed_failed_task_ids": list(set(tasks)),
            "detectors": [dict(REVIEWED_PARENT)] if detectors is None else detectors}


def complete_sample(*, review=True, claim="current", complete=True):
    return coverage("a", "b", detectors=[{
        "name": "parent", "current_claim_hash": claim,
        "selected_evaluation_complete": complete, "review_complete": review,
    }])


def sample_evaluation():
    return {"evaluation_complete": False,
            "evaluated_names": ["parent"], "evaluated_claims": {"parent": "current"}}


def full_evaluation():
    return {**sample_evaluation(), "evaluation_complete": True}


def start_sampled_attempt(runtime, baseline=None):
    """Initialize progress, then record one sampled evaluation attempt."""
    observe_task_progress(runtime, coverage=baseline or complete_sample(), initialize=True)
    runtime.scores_history.append(sample_evaluation())


def test_only_net_distinct_union_growth_resets_patience(runtime):
    observe_task_progress(runtime, coverage=coverage("a", "b"), initialize=True)
    for tasks in [("a", "b", "b"), ("a",), ("a", "b"), ("b", "c")]:
        runtime.scores_history.append({**full_evaluation(), "frontier_value": 99})
        result = observe_task_progress(runtime, coverage=coverage(*tasks))
        assert result["improved"] is False
        assert result["best_confirmed_failed_tasks"] == 2
    assert result["experiments_since_improve"] == 4
    runtime.scores_history.append({**full_evaluation(), "frontier_value": 0.1})
    result = observe_task_progress(runtime, coverage=coverage("a", "b", "c"))
    assert result["improved"] is True
    assert result["confirmed_task_gain"] == 1
    assert result["experiments_since_improve"] == 0
    assert "confirmed_failed_task_ids" not in result["task_coverage"]


def test_resume_keeps_best_union_and_uncheckpointed_history_gets_fresh_baseline(runtime):
    runtime.scores_history = [full_evaluation() for _ in range(50)]
    result = observe_task_progress(runtime, coverage=coverage("a", "b"))
    assert result["baseline_initialized"] is True
    assert result["experiments_since_improve"] == 0
    runtime.scores_history.append(full_evaluation())
    observe_task_progress(runtime, coverage=coverage("a", "b", "c"))
    # A fresh process reads the same checkpoint, even with decreased coverage.
    resumed = SimpleNamespace(run=runtime.run, scores_history=list(runtime.scores_history))
    result = observe_task_progress(resumed, coverage=coverage("a"), initialize=True)
    assert result["baseline_initialized"] is False
    assert result["best_confirmed_failed_tasks"] == 3
    result = observe_task_progress(resumed, coverage=coverage("a", "b", "c"))
    assert result["improved"] is False
    assert result["experiments_since_improve"] == 0
    result = observe_task_progress(resumed, coverage=coverage("x", fingerprint="new"))
    assert result["baseline_initialized"] is True
    assert result["best_confirmed_failed_tasks"] == 1


@pytest.mark.parametrize("old_precision,tasks", [(None, ("a",)), (.8, ("a", "b", "c"))])
def test_changed_quality_thresholds_rebaseline_without_claiming_task_gain(
    runtime, old_precision, tasks,
):
    old = coverage("a", "b")
    if old_precision is not None:
        old["quality_thresholds"] = {**THRESHOLDS, "ppv_at_pi0": old_precision}
    observe_task_progress(runtime, coverage=old, initialize=True)
    runtime.scores_history.append(full_evaluation())
    previous = observe_task_progress(runtime, coverage=old)
    assert previous["experiments_since_improve"] == 1

    current = {**coverage(*tasks), "quality_thresholds": THRESHOLDS}
    resumed = observe_task_progress(runtime, coverage=current, initialize=True)
    assert resumed["baseline_initialized"] is True
    assert resumed["best_confirmed_failed_tasks"] == len(tasks)
    assert resumed["quality_thresholds"] == THRESHOLDS
    assert resumed["completed_attempts"] == 1
    assert resumed["experiments_since_improve"] == 0
    assert resumed["improved"] is False
    assert resumed["confirmed_task_gain"] == 0
    repeated = observe_task_progress(runtime, coverage=current)
    assert repeated["baseline_initialized"] is False
    assert repeated["counts_toward_patience"] is False
    assert repeated["improved"] is False


def test_same_quality_thresholds_preserve_pending_review_across_resume(runtime):
    complete = {**complete_sample(), "quality_thresholds": THRESHOLDS}
    pending = {**complete_sample(review=False), "quality_thresholds": dict(THRESHOLDS)}
    start_sampled_attempt(runtime, complete)
    before = observe_task_progress(runtime, coverage=pending)
    resumed = observe_task_progress(runtime, coverage=pending, initialize=True)
    assert resumed["baseline_initialized"] is False
    assert resumed["history_cursor"] == before["history_cursor"] == 0
    result = observe_task_progress(runtime, coverage=complete)
    assert result["completed_attempts"] == result["experiments_since_improve"] == 1
    assert result["history_cursor"] == 1
    assert observe_task_progress(runtime, coverage=complete)["completed_attempts"] == 1


def test_changed_quality_thresholds_preserve_pending_review_until_completed_once(runtime):
    old_thresholds = {**THRESHOLDS, "ppv_at_pi0": .8}
    old = {**complete_sample(), "quality_thresholds": old_thresholds}
    start_sampled_attempt(runtime, old)
    pending = {**complete_sample(review=False), "quality_thresholds": old_thresholds}
    before = observe_task_progress(runtime, coverage=pending)

    migrated = observe_task_progress(
        runtime, coverage={**pending, "quality_thresholds": THRESHOLDS}, initialize=True,
    )
    assert migrated["baseline_initialized"] is True
    assert migrated["history_cursor"] == before["history_cursor"] == 0
    assert migrated["latest_attempt_complete"] is False
    assert migrated["completed_attempts"] == migrated["experiments_since_improve"] == 0
    assert migrated["improved"] is False
    complete = {**complete_sample(), "quality_thresholds": THRESHOLDS}
    result = observe_task_progress(runtime, coverage=complete)
    assert result["completed_attempts"] == result["experiments_since_improve"] == 1
    assert result["history_cursor"] == 1
    repeated = observe_task_progress(runtime, coverage=complete)
    assert repeated["completed_attempts"] == repeated["experiments_since_improve"] == 1
    assert repeated["counts_toward_patience"] is False


def test_changed_quality_thresholds_wait_for_available_source_before_rebaseline(runtime):
    old = {**complete_sample(),
           "quality_thresholds": {**THRESHOLDS, "ppv_at_pi0": .8}}
    start_sampled_attempt(runtime, old)
    observe_task_progress(runtime, coverage={**old, **complete_sample(review=False)})
    checkpoint = read_research_status(runtime.run.run_dir())["task_coverage_progress"]
    outage = {**complete_sample(), "available": False, "quality_thresholds": THRESHOLDS}
    result = observe_task_progress(runtime, coverage=outage)
    assert result["progress_blocked"] and not result["counts_toward_patience"]
    assert result["baseline_initialized"] is False
    assert result["history_cursor"] == checkpoint["history_cursor"] == 0
    assert read_research_status(runtime.run.run_dir())["task_coverage_progress"] == checkpoint

    recovered = observe_task_progress(
        runtime, coverage={**complete_sample(), "quality_thresholds": THRESHOLDS},
    )
    assert recovered["baseline_initialized"] is True
    assert recovered["quality_thresholds"] == THRESHOLDS
    assert recovered["best_confirmed_failed_tasks"] == 2
    assert recovered["improved"] is False
    assert recovered["confirmed_task_gain"] == 0
    assert recovered["completed_attempts"] == recovered["experiments_since_improve"] == 1
    assert recovered["history_cursor"] == 1


def test_only_current_claim_final_skip_closes_pending_attempt_without_inventing_review(
    runtime, monkeypatch,
):
    runtime.library = SimpleNamespace(functions=[SimpleNamespace(name="parent")])
    receipt = {"claim_hash": "another-claim"}
    monkeypatch.setattr("autofa.session.task_progress.current_final_validation_skip",
                        lambda rt, fn: receipt)
    unreviewed = complete_sample(review=False)
    start_sampled_attempt(runtime, unreviewed)
    pending = observe_task_progress(runtime, coverage=unreviewed)
    assert pending["completed_attempts"] == 0
    assert pending["progress_blocked"] is True
    assert pending["history_cursor"] == 0
    receipt = {"claim_hash": "current"}
    result = observe_task_progress(runtime, coverage=unreviewed)
    assert result["history_cursor"] == 1
    assert result["completed_attempts"] == 1
    assert result["progress_blocked"] is False
    assert result["confirmed_task_gain"] == 0
    assert unreviewed["detectors"][0]["review_complete"] is False
    assert observe_task_progress(runtime, coverage=unreviewed)["completed_attempts"] == 1


@pytest.mark.parametrize("change", [{"claim": "changed"}, {"complete": False}])
def test_incomplete_or_stale_capped_attempt_cannot_establish_plateau(runtime, change):
    start_sampled_attempt(runtime)
    result = observe_task_progress(runtime, coverage=complete_sample(**change))
    assert result["counts_toward_patience"] is False
    assert result["progress_blocked"] is True
    assert result["experiments_since_improve"] == 0
    # A later judge completes the existing attempt without another evaluation.
    result = observe_task_progress(runtime, coverage=complete_sample())
    assert result["counts_toward_patience"] is True
    assert result["experiments_since_improve"] == 1


def test_pending_review_survives_resume_and_counts_once(runtime):
    start_sampled_attempt(runtime)
    result = observe_task_progress(runtime, coverage=complete_sample(review=False))
    assert result["completed_attempts"] == 0
    assert result["progress_blocked"] is True
    resumed = SimpleNamespace(run=runtime.run, scores_history=list(runtime.scores_history))
    observe_task_progress(resumed, coverage=complete_sample(review=False), initialize=True)
    result = observe_task_progress(resumed, coverage=complete_sample())
    assert result["counts_toward_patience"] is True
    assert result["completed_attempts"] == result["experiments_since_improve"] == 1
    assert result["progress_blocked"] is False
    for _ in range(2):
        result = observe_task_progress(resumed, coverage=complete_sample())
        assert result["completed_attempts"] == result["experiments_since_improve"] == 1
        assert result["counts_toward_patience"] is False


def test_later_attempt_finishes_while_earlier_review_is_still_pending(runtime):
    start_sampled_attempt(runtime)
    second = {**sample_evaluation(), "evaluated_names": ["other"],
              "evaluated_claims": {"other": "current"}}
    runtime.scores_history.append(second)
    snapshot = complete_sample(review=False)
    snapshot["detectors"].append({
        "name": "other", "current_claim_hash": "current",
        "selected_evaluation_complete": True, "review_complete": True,
    })
    # Rows since the cursor are one batch: the reviewed later row waits for the earlier one.
    result = observe_task_progress(runtime, coverage=snapshot)
    assert result["completed_attempts"] == result["experiments_since_improve"] == 0
    assert result["progress_blocked"] is True
    assert result["history_cursor"] == 0
    snapshot["detectors"][0]["review_complete"] = True
    result = observe_task_progress(runtime, coverage=snapshot)
    assert result["completed_attempts"] == result["experiments_since_improve"] == 1
    assert result["counts_toward_patience"] is True
    assert result["history_cursor"] == 2
    repeated = observe_task_progress(runtime, coverage=snapshot)
    assert repeated["completed_attempts"] == repeated["experiments_since_improve"] == 1
    assert repeated["counts_toward_patience"] is False


def test_stale_pending_attempts_from_older_checkpoints_are_dropped(runtime):
    observe_task_progress(runtime, coverage=complete_sample(), initialize=True)
    status = read_research_status(runtime.run.run_dir())
    status["task_coverage_progress"]["pending_attempts"] = [
        {"start": 0, "stop": 1, "counts_toward_patience": True}]
    write_research_status(runtime.run.run_dir(), status)
    runtime.scores_history.append(sample_evaluation())
    result = observe_task_progress(runtime, coverage=complete_sample())
    assert "pending_attempts" not in result
    assert result["completed_attempts"] == result["experiments_since_improve"] == 1
    saved = read_research_status(runtime.run.run_dir())["task_coverage_progress"]
    assert "pending_attempts" not in saved


def test_judge_only_gain_resets_without_consuming_another_attempt(runtime):
    observe_task_progress(runtime, coverage=coverage(), initialize=True)
    runtime.scores_history.append(full_evaluation())
    observe_task_progress(runtime, coverage=coverage())
    result = observe_task_progress(runtime, coverage=coverage("a", "b"))
    assert result["improved"] is True
    assert result["confirmed_task_gain"] == 2
    assert result["experiments_since_improve"] == 0
    assert result["completed_attempts"] == 1
    assert read_research_status(runtime.run.run_dir())["task_coverage_progress"][
        "best_confirmed_failed_tasks"
    ] == 2


def test_source_outage_preserves_checkpoint_pending_work_and_same_corpus_high_water(runtime):
    snapshot = complete_sample()
    start_sampled_attempt(runtime, snapshot)
    observe_task_progress(runtime, coverage=snapshot)
    runtime.scores_history.append(sample_evaluation())
    observe_task_progress(runtime, coverage=complete_sample(review=False))
    checkpoint = read_research_status(runtime.run.run_dir())["task_coverage_progress"]
    outage = {"available": False, "corpus_fingerprint": "unverified-other-corpus",
              "n_confirmed_failed_tasks": 0, "detectors": []}
    result = observe_task_progress(runtime, coverage=outage)
    assert result["progress_blocked"] and not result["counts_toward_patience"]
    assert result["baseline_initialized"] is False
    assert read_research_status(runtime.run.run_dir())["task_coverage_progress"] == checkpoint
    recovered = observe_task_progress(runtime, coverage=snapshot)
    assert recovered["improved"] is False
    assert recovered["best_confirmed_failed_tasks"] == 2
    assert recovered["completed_attempts"] == recovered["experiments_since_improve"] == 2
    changed = observe_task_progress(runtime, coverage=coverage("x", fingerprint="verified-new"))
    assert changed["baseline_initialized"] is True
    assert changed["best_confirmed_failed_tasks"] == 1
    assert changed["completed_attempts"] == 0


def test_resume_observes_new_confirmations_before_plateau_check(runtime):
    observe_task_progress(runtime, coverage=coverage("a", "b"), initialize=True)
    for _ in range(6):
        runtime.scores_history.append(full_evaluation())
        observe_task_progress(runtime, coverage=coverage("a", "b"))
    resumed = observe_task_progress(runtime, coverage=coverage("a", "b", "c"), initialize=True)
    assert resumed["improved"] is True
    assert resumed["experiments_since_improve"] == 0
    assert resumed["best_confirmed_failed_tasks"] == 3


def test_final_validation_handoff_for_supported_candidates_without_corpus_scan(runtime):
    snapshot = coverage("a", "b", detectors=[
        {"name": name, "eligible": True, "evaluation_sampled": True,
         "corpus_evaluation_complete": False}
        for name in ["parent", "other"]
    ])
    progress = {"task_coverage": snapshot, "progress_blocked": False}
    target = {"minimum_met": False, "min_failure_modes": 2}
    assert not final_validation_handoff(runtime, progress, target, ready=False)[
        "ready_for_final_validation"
    ]
    result = final_validation_handoff(runtime, progress, target, ready=True)
    assert result["ready_for_final_validation"] is True
    assert result["final_validation_candidates"] == ["parent", "other"]
    assert "mode='full'" in result["final_validation_guidance"]  # The agent runs the scans.
    # Pending requests remain visible until the full scans actually finish.
    assert final_validation_handoff(runtime, progress, target, ready=True)[
        "ready_for_final_validation"
    ]
    for row in snapshot["detectors"]:
        row["evaluation_sampled"] = False
        row["corpus_evaluation_complete"] = True
    assert not final_validation_handoff(runtime, progress, target, ready=True)[
        "ready_for_final_validation"
    ]


def test_round_boundary_handoff_can_ignore_unrelated_pending_attempts(runtime):
    candidates = [
        {"name": "reviewed", "eligible": True, "corpus_evaluation_complete": False},
        {"name": "complete", "eligible": True, "corpus_evaluation_complete": True},
    ]
    progress = {
        "task_coverage": coverage("a", "b", detectors=[
            *candidates,
            {"name": "unreviewed", "eligible": False, "corpus_evaluation_complete": False},
        ]),
        "progress_blocked": True,
    }
    target = {"minimum_met": False, "min_failure_modes": 2}
    assert not final_validation_handoff(runtime, progress, target, ready=True)[
        "ready_for_final_validation"
    ]
    result = final_validation_handoff(runtime, progress, target, ready=True, allow_pending=True)
    assert result["ready_for_final_validation"]
    assert result["final_validation_candidates"] == ["reviewed"]
    # Unreviewed sampled modes cannot fill the supported-candidate minimum.
    candidates[1]["eligible"] = False
    assert not final_validation_handoff(runtime, progress, target, ready=True, allow_pending=True)[
        "ready_for_final_validation"
    ]


def test_display_good_boundary_handoff_uses_projected_support_not_progress_gate(runtime):
    progress = {
        "task_coverage": coverage("one-observed-task", detectors=[
            {"name": "sampled_good", "eligible": False, "corpus_evaluation_complete": False},
            {"name": "full_good", "eligible": False, "corpus_evaluation_complete": True},
        ]),
        "progress_blocked": True,
    }
    strict_target = {"minimum_met": True, "min_failure_modes": 2}
    discovery_target = {
        "minimum_met": True, "min_failure_modes": 2,
        "failure_mode_eligibility": [
            {"name": "sampled_good", "eligible": True, "measured": {
                "sampled": True, "corpus_evaluation_complete": False,
                "n_confirmed_failed_datapoints": 1,
                "estimated_n_confirmed_failed_tasks": 7.5,
            }},
            {"name": "full_good", "eligible": True,
             "measured": {"corpus_evaluation_complete": True}},
            {"name": "weak", "eligible": False,
             "measured": {"corpus_evaluation_complete": False}},
        ],
    }
    before = deepcopy((progress, strict_target, discovery_target, vars(runtime)))

    result = final_validation_handoff(
        runtime, progress, strict_target, ready=True, allow_pending=True,
        discovery_target=discovery_target,
    )

    assert result["ready_for_final_validation"]
    assert result["final_validation_candidates"] == ["sampled_good"]
    # The driver, not the agent, runs the full scan for display-Good candidates.
    assert "evaluate_functions" not in result["final_validation_guidance"]
    assert (progress, strict_target, discovery_target, vars(runtime)) == before


def test_display_good_handoff_includes_every_unfinished_good_corpus_scan(runtime):
    progress = {"task_coverage": coverage(), "progress_blocked": False}
    strict_target = {"minimum_met": False, "min_failure_modes": 1}
    row = {"name": "good", "eligible": True,
           "measured": {"sampled": False, "corpus_evaluation_complete": False}}
    discovery_target = {"minimum_met": True, "failure_mode_eligibility": [row]}

    pending = final_validation_handoff(
        runtime, progress, strict_target, ready=True, discovery_target=discovery_target,
    )
    assert pending["final_validation_candidates"] == ["good"]
    row["measured"]["corpus_evaluation_complete"] = True
    completed = final_validation_handoff(
        runtime, progress, strict_target, ready=True, discovery_target=discovery_target,
    )
    assert not completed["ready_for_final_validation"]
    assert completed["final_validation_candidates"] == []
    assert completed["final_validation_guidance"] is None


@pytest.mark.parametrize("ready,minimum_met,blocked,allow_pending", [
    (False, True, False, True),
    (True, False, False, True),
    (True, True, True, False),
])
def test_display_good_handoff_still_requires_authorized_boundary_and_minimum(
    runtime, ready, minimum_met, blocked, allow_pending,
):
    progress = {"task_coverage": coverage(), "progress_blocked": blocked}
    target = {"minimum_met": False, "min_failure_modes": 1}
    discovery_target = {"minimum_met": minimum_met, "failure_mode_eligibility": [
        {"name": "good", "eligible": True,
         "measured": {"corpus_evaluation_complete": False}},
    ]}
    result = final_validation_handoff(
        runtime, progress, target, ready=ready, allow_pending=allow_pending,
        discovery_target=discovery_target,
    )
    assert not result["ready_for_final_validation"]
    assert result["final_validation_guidance"] is None


def test_research_handoff_keeps_existing_strict_target_and_progress_policy(runtime):
    progress = {"task_coverage": coverage("confirmed", detectors=[
        {"name": "supported", "eligible": True, "corpus_evaluation_complete": False},
    ]), "progress_blocked": False}
    target = {"minimum_met": False, "min_failure_modes": 1}
    assert final_validation_handoff(runtime, progress, target, ready=True)[
        "ready_for_final_validation"
    ]
    target["minimum_met"] = True
    assert not final_validation_handoff(runtime, progress, target, ready=True)[
        "ready_for_final_validation"
    ]
