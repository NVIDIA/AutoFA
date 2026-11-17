# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstentions, fixed-panel breadth and development evidence guide search, never keepers."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.pareto import rank_functions
from autofa.metrics.search_association import search_association
from autofa.metrics.search_panel import build_search_panel, current_search_panel
from autofa.metrics.search_ranking import search_ranking, search_score
from autofa.metrics.task_recall import build_task_recall

from support.coverage import candidate

# The evaluator saves this shape when the source task inventory is unavailable.
UNAVAILABLE_PANEL = {"version": 1, "available": False, "scope": "fixed_scoring_panel",
                     "task_recall": None, "reason": "Source task inventory unavailable."}


def test_current_development_evidence_ranks_separately_from_validated_keepers():
    config = Config(behavior_strictness="causal")
    fn = candidate(reviewed=("yes", "yes"), reused=2)
    before = deepcopy(fn.to_dict())
    score = search_score(fn, config, run_id="r1")
    assert score["eligible"]
    assert score["score"] == pytest.approx(.25 ** .5)
    assert score["provisional"]
    assert not rank_functions([fn.to_dict()], config=config)[0]["ranking"]["eligible"]
    assert fn.to_dict() == before


def test_fully_accounted_abstentions_use_conservative_precision_without_changing_saved_scores():
    fn = candidate(panel_ids=["a1", "a2", "b1", "p1"])
    fn.scores.update(a=7., b=0., c=274., d=428., n_items=709., n_scored=709,
                     n_insufficient=3, n_unresolved=3, evaluation_complete=False,
                     assessment_coverage=709 / 712,
                     unresolved_bounds={"recall": [7 / 283, 9 / 283], "fpr": [0, 1 / 429]},
                     # Even an old inflated aggregate cannot remove the current uncertain case.
                     behavior_purity=1.)
    before = deepcopy(fn.scores)
    result = search_score(fn, Config(behavior_strictness="causal"))
    assert result["eligible"]
    assert result["association_scope"] == "worst_case_abstentions"
    assert result["abstention_n"] == 3
    assert result["objective_values"]["ppv_at_pi0"] == pytest.approx(.81974522293)
    assert result["objective_values"]["behavior_purity"] == pytest.approx(2 / 3)
    assert result["evidence"]["uncertain"] == 1
    assert not rank_functions(
        [fn.to_dict()], config=Config(behavior_strictness="causal"),
    )[0]["ranking"]["eligible"]
    assert fn.scores == before


def test_completed_capped_selection_can_guide_search_on_the_common_panel():
    fn = candidate(panel_ids=["a1", "a2", "b1", "p1"])
    fn.scores.update(evaluation_scope_complete=False, eval_mode="capped", eval_truncated=True,
                     task_recall_scope="observed_lower_bound")
    score = search_score(fn, Config(behavior_strictness="causal"))
    assert score["eligible"]
    assert score["task_recall_scope"] == "fixed_scoring_panel"
    assert score["objective_values"]["task_recall"] == .5


def test_breadth_uses_distinct_tasks_on_the_same_panel():
    panel = ["a1", "a2", "b1", "p1"]
    repeated = candidate("repeated", reviewed=("yes", "yes"), matched=("a1", "a2", "d1"),
                         panel_ids=panel)
    broader = candidate("broader", reviewed=("yes", "yes"), matched=("a1", "b1"),
                        panel_ids=panel)
    broader.scores.update(sampled=True, task_recall_scope="observed_lower_bound")
    result = search_ranking([repeated, broader], Config(behavior_strictness="causal"), run_id="r1")
    assert [row["name"] for row in result["ranked"]] == ["broader", "repeated"]
    assert result["ranked"][0]["objective_values"]["task_recall"] == 1
    assert result["ranked"][1]["objective_values"]["task_recall"] == .5
    assert repeated.scores["task_recall"] == broader.scores["task_recall"] == .5


def test_unavailable_scoring_panel_is_not_ranked_against_a_census():
    census = candidate("census")
    sampled = candidate("sampled", matched=("a1", "b1"))
    sampled.scores.update(sampled=True, task_recall_scope="observed_lower_bound",
                          search_panel=UNAVAILABLE_PANEL)
    result = search_ranking([census, sampled], Config(behavior_strictness="causal"), run_id="r1")
    assert [row["name"] for row in result["ranked"]] == ["census"]
    assert result["unranked"][0]["score"] is None


@pytest.mark.parametrize("full_corpus", [False, True])
def test_different_panel_membership_or_full_corpus_is_not_silently_ranked(full_corpus):
    first = candidate("first", panel_ids=["a1", "b1", "p1"])
    second = candidate("second", panel_ids=None if full_corpus else ["a1", "c1", "p1"])
    result = search_ranking([first, second], Config(behavior_strictness="causal"), run_id="r1")
    assert result["ranked"] == []
    assert len(result["unranked"]) == 2
    assert all(row["score"] is None for row in result["unranked"])
    assert all(row["status"] == "incomparable_cohorts" for row in result["unranked"])


def test_corpus_frontier_waits_for_census_while_panel_search_and_attempt_metrics_remain_usable():
    from autofa.metrics.pareto import (
        function_ranking_eligibility,
        library_frontier_points,
        point_from_scores,
    )

    fn = candidate(panel_ids=["a1", "a2", "b1", "p1"])
    fn.scores.update(sampled=True, task_recall_scope="observed_lower_bound")
    cfg = Config(behavior_strictness="causal", objective_axes=["task_recall"],
                 objective_weights=[1])
    before = deepcopy(fn.scores)
    assert search_score(fn, Config(behavior_strictness="causal"), run_id="r1")["eligible"]
    status = function_ranking_eligibility(fn, ["task_recall"], config=cfg)
    assert status["eligible"] is False
    assert status["status"] == "task_coverage_awaiting_census"
    assert point_from_scores(fn.scores, ["task_recall"]) is None
    assert library_frontier_points([fn], ["task_recall"], config=cfg) == []
    assert not rank_functions([fn.to_dict()], config=cfg)[0]["ranking"]["eligible"]
    assert function_ranking_eligibility(fn, ["recall"])["eligible"]
    assert fn.scores == before
    fn.scores.update(sampled=False, task_recall_scope="full_selected_corpus")
    assert point_from_scores(fn.scores, ["task_recall"]) == (.25,)
    assert library_frontier_points([fn], ["task_recall"], config=cfg) == [(.25,)]
    assert rank_functions([fn.to_dict()], config=cfg)[0]["ranking"]["eligible"]


@pytest.mark.parametrize("change", [
    {"claim_hash": "old"}, {"judge_policy_hash": "old"}, {"judge_contract_version": 0},
    {"behavior_n_yes": 3},  # Well-typed counts that disagree with the verdict rows.
    {"errors": 1}, {"evaluation_scope_complete": False}, {"search_panel": UNAVAILABLE_PANEL},
    {"judge_prefilter": {"status": "skip"}}, {"behavior_sample": {"actual_n": 5}},
], ids=["old-claim", "old-judge-policy", "old-judge-contract", "inconsistent-review-counts",
        "processing-error", "incomplete-scope", "unavailable-panel", "prefilter-rejected",
        "inconsistent-review-sample"])
def test_unavailable_stale_malformed_or_processing_error_evidence_never_becomes_zero(change):
    fn = candidate()
    fn.scores.update(change)
    score = search_score(fn, Config(behavior_strictness="causal"), run_id="r1")
    assert not score["eligible"]
    assert score["score"] is None
    assert search_ranking([fn], Config(behavior_strictness="causal"))["ranked"] == []


def test_malformed_case_metadata_does_not_break_library_search_ranking():
    malformed = candidate("malformed")
    malformed.scores["behavior_verdicts"][0]["completion_status"] = ["complete"]
    healthy = candidate("healthy")
    ranking = search_ranking([malformed, healthy], Config(behavior_strictness="causal"),
                             run_id="r1")
    assert [row["name"] for row in ranking["ranked"]] == ["healthy"]
    assert ranking["unranked"][0]["score"] is None


@pytest.mark.parametrize("change,panel", [
    ("claim", False), ("dependency", True), ("foreign_run", True), ("positive_proof", False),
])
def test_current_provenance_includes_claim_dependencies_run_and_confirmation(change, panel):
    fn = candidate(panel_ids=["a1", "b1", "p1"] if panel else None)
    if change == "claim":
        fn.description += " revised"
    elif change == "dependency":
        fn.meta["dependency_bundle"] = {"version": 1, "nodes": {}, "helpers": {}, "detectors": {}}
    elif change == "foreign_run":
        fn.scores["evaluation_run_id"] = "another_run"
    else:
        fn.scores["behavior_verdicts"][0]["initial_review"]["snapshot_id"] = "another-evidence"
    assert search_score(fn, Config(behavior_strictness="causal"), run_id="r1")["score"] is None


def test_all_uncertain_is_unavailable_but_measured_rejection_can_be_zero():
    assert search_score(candidate(reviewed=("uncertain",)),
                        Config(behavior_strictness="causal"))["score"] is None
    rejected = search_score(candidate(reviewed=("no",)), Config(behavior_strictness="causal"))
    assert rejected["eligible"]
    assert rejected["score"] == 0


def test_custom_objectives_keep_attempt_recall():
    config = Config(behavior_strictness="causal",
                    objective_axes=["recall", "ppv_at_pi0", "behavior_purity"],
                    objective_weights=[])
    fn = candidate(config=config)
    result = search_score(fn, config)
    assert result["eligible"]
    assert list(result["objective_values"]) == ["recall", "ppv_at_pi0", "behavior_purity"]
    assert result["objective_values"]["recall"] == .4
    assert result["score"] == pytest.approx((.4 * 2 / 3) ** (1 / 3))


def test_library_inspection_exposes_provisional_order_and_score(tmp_path, monkeypatch):
    from autofa.mcp import tools_library

    fn = candidate()
    rt = SimpleNamespace(config=Config(behavior_strictness="causal"),
                         library=FunctionLibrary(functions=[fn]),
                         helpers=SimpleNamespace(aliases={}),
                         run=SimpleNamespace(run_id="r1", run_dir=lambda: tmp_path))
    monkeypatch.setattr(tools_library, "get_runtime", lambda: rt)
    monkeypatch.setattr(tools_library, "_detector_exploration", lambda: {})
    monkeypatch.setattr(tools_library, "_failure_modes_summary", lambda: {})
    assert tools_library.list_library_functions()["search_ranking"]["ranked"][0]["name"] == fn.name
    assert tools_library.get_library_function(fn.name)["search_score"]["eligible"]
    assert tools_library.get_function_feedback(fn.name)["search_score"]["eligible"]


# Conservative search association.

def _scores():
    # Seven known matches, two abstained failures, and one abstained passing case.
    return {
        **contingency(7, 0, 274, 428),
        "evaluated": True, "evaluation_scope_complete": True, "evaluation_complete": False,
        "sampled": False, "n_scored": 709, "n_insufficient": 3, "n_unresolved": 3,
        "assessment_coverage": 709 / 712,
        "unresolved_bounds": {"recall": [7 / 283, 9 / 283], "fpr": [0, 1 / 429]},
    }


def test_full_or_capped_abstentions_lower_search_precision_without_changing_saved_scores():
    scores = _scores()
    before = deepcopy(scores)
    result = search_association(scores)
    assert result is not None
    assert result["recall"] == pytest.approx(7 / 283)
    assert result["fpr"] == pytest.approx(1 / 429)
    assert result["ppv_at_pi0"] == pytest.approx(.3 * (7 / 283) / (
        .3 * (7 / 283) + .7 * (1 / 429)))
    assert result["raw_ppv"] == pytest.approx(7 / 8)
    assert result["search_association_scope"] == "worst_case_abstentions"
    assert result["search_abstention_n"] == 3
    assert scores == before and scores["ppv_at_pi0"] == 1
    assert result["evaluation_complete"] is False  # No promotion into validated ranking.
    # A completed capped scope can guide search with the same conservative bound.
    capped = search_association({**scores, "eval_mode": "capped",
                                 "evaluation_scope_complete": False, "eval_truncated": True})
    assert capped["ppv_at_pi0"] == pytest.approx(result["ppv_at_pi0"])
    assert capped["evaluation_scope_complete"] is False


def test_sampled_abstentions_keep_weighted_frame_and_actual_sample_counts_separate():
    # 100 sampled failed + 100 sampled passing attempts represent a 712-row frame.
    scores = {
        **_scores(), **contingency(3 * 2.83, 0, 95 * 2.83, 99 * 4.29),
        "sampled": True, "n_scored": 197, "sample_n": 200, "sample_frame_n": 712,
        "evaluation_item_ids": [f"case-{i}" for i in range(200)],
        "assessment_coverage": 197 / 200,
        "unresolved_bounds": {"recall": [.03, .05], "fpr": [0, .01]},
    }
    result = search_association(scores)
    assert result is not None
    assert result["ppv_at_pi0"] == pytest.approx(.5625)
    assert result["recall"] == pytest.approx(.03)
    assert result["n_items"] == pytest.approx(712)
    assert result["search_abstention_n"] == 3


@pytest.mark.parametrize("key,value", [
    ("evaluated", False), ("errors", 1), ("n_unresolved", 4),
    ("assessment_coverage", 1), ("unresolved_bounds", None),
    ("unresolved_bounds", {"recall": [7 / 283, 9 / 283], "fpr": [0, 0]}),
    ("unresolved_bounds", {"recall": [9 / 283, 7 / 283], "fpr": [0, 1 / 429]}),
], ids=["unevaluated", "errors", "inconsistent-abstentions",
        "inconsistent-coverage", "no-bounds", "bounds-miss-frame", "inverted-bounds"])
def test_errors_incomplete_scope_and_inconsistent_abstentions_remain_unranked(key, value):
    scores = _scores()
    scores[key] = value
    assert search_association(scores) is None


@pytest.mark.parametrize("defect", ["duplicate_panel", "sample_count"])
def test_sampled_abstentions_require_consistent_panel_accounting(defect):
    scores = {**_scores(), "sampled": True, "sample_n": 712, "sample_frame_n": 712,
              "evaluation_item_ids": [f"case-{i}" for i in range(712)]}
    if defect == "duplicate_panel":
        scores["evaluation_item_ids"][-1] = "case-0"
    else:
        scores["sample_n"] = 711
    assert search_association(scores) is None


def test_complete_evaluation_without_abstentions_preserves_current_association():
    scores = {**contingency(7, 0, 276, 429), "evaluated": True,
              "evaluation_scope_complete": True}
    assert search_association(scores) == scores
    scores["evaluation_complete"] = False
    assert search_association(scores) is None


# Fixed scoring panel breadth.

@pytest.fixture
def inventory():
    return {
        "inventory_complete": True, "corpus_fingerprint": "corpus", "n_items": 6,
        "n_failed_attempts": 5, "n_failed_tasks": 3, "n_unknown_outcomes": 0,
        "item_ids": ["a1", "a2", "a3", "b1", "c1", "pass"],
        "task_by_failed_item_id": {"a1": "a", "a2": "a", "a3": "a", "b1": "b", "c1": "c"},
    }


def detector(inventory, matches, **scores):
    fn = FeatureFunction(name="flag", description="Explicit field mismatch",
                         mechanism="An explicit value is replaced incorrectly.",
                         source="def flag(item): return True")
    fn.scores = {
        "evaluated": True, "evaluation_run_id": "run", "evaluation_scope_complete": True,
        "evaluation_claim_hash": claim_hash(source=fn.source, description=fn.description,
                                             mechanism=fn.mechanism),
        "sampled": False, "n_items": 6, "n_scored": 6,
        "n_flagged": len(matches), "n_fail_caught": len(matches),
        "bucket": "True", "bucket_mode": "exact", **scores,
    }
    fn.failure_localisations = {
        item: {"item_id": item, "assigned_label": "True", "source_sha256": text_hash(fn.source)}
        for item in matches
    }
    fn.scores.update(build_task_recall(fn, inventory, run_id="run"))
    return fn


def test_sample_and_census_compare_same_tasks_without_overwriting_corpus_metrics(inventory):
    full = detector(inventory, ["a1", "a2", "b1", "c1"])
    sample = detector(inventory, ["a1", "a2", "b1"], sampled=True)
    panel_ids = ["a1", "a2", "b1", "pass"]
    snapshots = []
    for fn in (full, sample):
        before = deepcopy(fn.to_dict())
        snapshots.append(build_search_panel(fn, inventory, item_ids=panel_ids, run_id="run"))
        assert fn.to_dict() == before
    assert snapshots[0] == snapshots[1]
    assert snapshots[0]["available"] is True
    assert snapshots[0]["task_recall"] == 1
    assert snapshots[0]["n_failed_tasks"] == 2
    assert full.scores["task_recall"] == 1
    assert sample.scores["task_recall"] == 2 / 3


def test_repeats_and_matches_outside_panel_cannot_add_breadth(inventory):
    fn = detector(inventory, ["a1", "a2", "c1"])
    result = build_search_panel(fn, inventory, item_ids=["a1", "a2", "b1", "pass"], run_id="run")
    assert result["task_recall"] == .5
    assert result["n_matched_failed_tasks"] == 1
    fn.scores["search_panel"] = result
    fn.failure_localisations["b1"] = {**fn.failure_localisations["a1"], "item_id": "b1"}
    assert current_search_panel(fn.scores)["task_recall"] == .5  # Later judge probe is not scoring.


def test_abstentions_remain_in_panel_task_denominator(inventory):
    fn = detector(inventory, ["a1"], n_insufficient=1, n_unresolved=1, n_scored=5)
    result = build_search_panel(fn, inventory, item_ids=["a1", "b1", "pass"], run_id="run")
    assert result["task_recall"] == .5
    assert result["n_failed_tasks"] == 2
    assert result["observed_lower_bound"] is True


@pytest.mark.parametrize("change", ["errors", "unknown_task", "unknown_outcome"])
def test_unavailable_or_error_evidence_never_becomes_negative_predictions(inventory, change):
    fn = detector(inventory, ["a1"])
    if change == "unknown_task":
        inventory["task_by_failed_item_id"]["b1"] = None
    elif change == "unknown_outcome":
        inventory.update(n_unknown_outcomes=1, unknown_outcome_item_ids=["b1"])
    else:
        fn.scores[change] = 1
    result = build_search_panel(fn, inventory, item_ids=["a1", "b1", "pass"], run_id="run")
    assert result["available"] is False
    assert result["task_recall"] is None


def test_missing_point_reads_do_not_shrink_the_requested_panel(inventory):
    fn = detector(inventory, ["a1"], sampled=True)
    result = build_search_panel(fn, inventory, item_ids=["a1", "b1", "pass"], run_id="run",
                                attempted_item_ids=["a1", "pass"])
    assert result["available"] is False
    assert result["task_recall"] is None
    assert result["n_items"] == 3


@pytest.mark.parametrize("key,value", [
    ("claim_hash", "other"), ("corpus_fingerprint", "other"),
    ("n_matched_failed_tasks", 20), ("task_recall", .9),
])
def test_current_panel_checks_provenance_and_counts(inventory, key, value):
    fn = detector(inventory, ["a1"])
    fn.scores["search_panel"] = build_search_panel(
        fn, inventory, item_ids=["a1", "b1", "pass"], run_id="run",
    )
    assert current_search_panel(fn.scores) is not None
    fn.scores["search_panel"][key] = value
    assert current_search_panel(fn.scores) is None


def test_stale_panel_and_legacy_sample_are_not_backfilled(inventory):
    fn = detector(inventory, ["a1"], sampled=True)
    assert current_search_panel(fn.scores) is None
    fn.scores["search_panel"] = build_search_panel(
        fn, inventory, item_ids=["a1", "b1", "pass"], run_id="run",
    )
    fn.scores["evaluation_stale"] = True
    assert current_search_panel(fn.scores) is None
