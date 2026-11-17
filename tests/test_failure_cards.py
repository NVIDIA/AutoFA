# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The failure-cards report renders saved scores, reviews and policies without inventing data."""

from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.behavior_policy import STRICTNESS_LABELS
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import detector_hash
from autofa.data.parquet_store import Dataset
from autofa.mcp.tools_library import mechanism_prose_warning
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.reports.failure_cards import (
    _extract_autofa_data,
    _spill_items_to_disk,
    build_data,
    build_report_assets_zip,
    build_report_bundle_zip,
    bundle_zip_filename,
    failure_mode_card_id,
    file_url,
    refresh_report_data,
    render_html,
    serve_failure_cards,
    write_failure_cards,
)

from support.task_support import (
    _function,
    _render_card,
    attach_sampled_task_support,
    attach_task_support,
    report_renderers,
    run_node,
)


def _sample_lib() -> FunctionLibrary:
    return FunctionLibrary(model_id="test-model", benchmark="bench-a", functions=[FeatureFunction(
        name="misses_edge_case",
        description="Solves the happy path, skips the documented edge case",
        mechanism="The model implements the common path and never tests the edge case.",
        source="def misses_edge_case(item, llm_client=None):\n    return True\n",
        kind="regex", example_fail_ids=["m|b|id1", "m|b|id2"],
        scores={
            "evaluated": True, "n_items": 100, "a": 20, "b": 10, "c": 30, "d": 40,
            "recall": 0.4, "precision": 0.667, "raw_ppv": 0.667, "ppv_at_pi0": 0.55,
            "pct_caught": 0.4, "behavior_purity": 0.8, "behavior_specificity": 0.8,
            "n_flagged": 30, "n_fail_caught": 20,
            "buckets": [{
                "bucket": "True", "n": 30, "n_fail": 20, "n_pass": 10, "purity": 0.667,
                "recall": 0.4, "ppv_at_pi0": 0.55, "odds_ratio": 2.67, "or_ci_low": 1.1,
                "or_ci_high": 6.4, "is_high_purity": False,
                "example_fail_ids": ["m|b|id1", "m|b|id2"],
            }],
        },
    )])


def _current_review(fn, **scores):
    """Attach a current-policy review receipt for ``fn``'s claim."""
    fn.scores.update(
        judge_contract_version=JUDGE_CONTRACT_VERSION, judge_policy_hash=judge_policy_hash(),
        claim_hash=claim_hash(source=fn.source, description=fn.description, mechanism=fn.mechanism),
        **scores,
    )
    return fn


def _tones(card):
    return {label.strip(): color for color, label in re.findall(
        r'class="tax-statcard (\w+)">\s*<div class="tax-statcard-label">([^<]+)', card,
    )}


def test_build_data_envelope():
    data = build_data(_sample_lib())
    assert data["meta"]["model_id"] == "test-model"
    assert data["meta"]["n_functions"] == 1
    assert data["meta"]["n_failures"] == 50
    assert data["items"] == {}
    fn = data["functions"][0]
    assert fn["name"] == "misses_edge_case"
    assert fn["metrics"]["behavior_purity"] == 0.8
    assert fn["verdict"]["code"] == "OK"
    assert fn["buckets"][0]["example_fail_ids"] == ["m|b|id1", "m|b|id2"]


def test_report_never_substitutes_attempt_recall_for_missing_or_stale_task_recall():
    library = _sample_lib()
    fn = library.functions[0]
    assert build_data(library)["functions"][0]["metrics"]["task_recall"] is None
    fingerprint = claim_hash(source=fn.source, description=fn.description, mechanism=fn.mechanism)
    fn.scores.update(task_recall=.2, task_recall_scope="full_selected_corpus",
                     evaluation_claim_hash=fingerprint, task_recall_claim_hash=fingerprint)
    original = deepcopy(fn.scores)
    fn.description = "Changed detector claim"
    metrics = build_data(library)["functions"][0]["metrics"]
    assert metrics["task_recall"] is None
    assert metrics["task_recall_scope"] == "unavailable"
    assert fn.scores == original


def test_refresh_saved_report_retains_cohort_examples_and_verified_passages():
    library = _sample_lib()
    fn = library.functions[0]
    record = {"detector_hash": detector_hash(fn.source), "passages": [{"text": "edge case"}]}
    fn.failure_localisations = {"m|b|id1": record}
    saved = build_data(library)
    saved["meta"].update(n_items=712, n_failures=391, items_base="saved-items")
    saved["items"] = {"m|b|id2": {"response_head": "Saved full trajectory"}}
    saved["item_index"] = {"m|b|id1": "example.json"}
    saved_fn = saved["functions"][0]
    saved_fn.pop("card_id")
    saved_fn.pop("behavior_validation")
    saved_fn["failure_localisations"] = deepcopy(fn.failure_localisations)
    saved_fn["failure_localisations"]["m|b|id1"]["passages"][0]["navigation"] = {
        "event_index": 4, "status": "verified",
    }
    before = deepcopy(saved)

    refreshed = refresh_report_data(saved, library)

    assert saved == before
    for key in ("meta", "items", "item_index"):
        assert refreshed[key] == before[key]
    refreshed_fn = refreshed["functions"][0]
    assert refreshed_fn["card_id"] == failure_mode_card_id(fn.name)
    assert refreshed_fn["metrics"] == before["functions"][0]["metrics"]
    assert refreshed_fn["buckets"] == before["functions"][0]["buckets"]
    assert refreshed_fn["failure_localisations"] == before["functions"][0]["failure_localisations"]
    assert "behavior_validation" in refreshed_fn
    # A passage tied to an earlier detector must not survive a source change.
    fn.source = "def misses_edge_case(item): return False"
    assert refresh_report_data(saved, library)["functions"][0]["failure_localisations"] == {}


def test_saved_and_new_reports_share_order_independent_of_optimizer_preferences(monkeypatch):
    library = _sample_lib()
    fn = _current_review(
        library.functions[0], behavior_n_judged=1, behavior_n_completed=1, behavior_n_yes=1,
        behavior_purity=1,
    )
    fn.scores["claim_contract"] = build_claim_contract(
        source=fn.source, description=fn.description, mechanism=fn.mechanism)
    wide = deepcopy(fn)
    wide.name = "wider_coverage"
    wide.scores.update(recall=.8, ppv_at_pi0=.3)
    library.functions.append(wide)
    candidate = deepcopy(fn)
    candidate.name = "unjudged_candidate"
    candidate.scores.update(recall=.99, behavior_n_judged=0, behavior_n_yes=0)
    library.functions.insert(0, candidate)
    data = build_data(library)
    # Better quality precedes raw breadth: OK (.4 recall) beats WEAK (.8),
    # and the unreviewed .99-recall candidate remains last with review unavailable.
    expected = [fn.name, wide.name, candidate.name]
    assert [f["name"] for f in data["functions"]] == expected
    saved = deepcopy(data)
    saved["functions"].reverse()
    assert refresh_report_data(saved, library) == data
    rendered = _extract_autofa_data(render_html(saved))
    assert rendered["functions"] == data["functions"]
    assert rendered["meta"]["n_pending_scores"] == 0
    assert rendered["functions"][-1]["verdict"]["code"] == "UNSCORED"
    # Old saved badges cannot dictate the current metric-based presentation.
    stale_badges = deepcopy(saved)
    for function in stale_badges["functions"]:
        function["verdict"] = {"code": "GOOD" if function["name"] == wide.name else "WEAK"}
    assert [f["name"] for f in refresh_report_data(stale_badges, library)["functions"]] == expected
    rerendered = _extract_autofa_data(render_html(stale_badges))
    assert [f["name"] for f in rerendered["functions"]] == expected
    # Viewing a published assessment on a newer install is not a re-evaluation.
    monkeypatch.setattr("autofa.metrics.claude_judge.judge_policy_hash",
                        lambda *args, **kwargs: "new-policy")
    historical = refresh_report_data(saved, library)
    assert [f["name"] for f in historical["functions"]] == expected
    for previous, current in zip(data["functions"][:2], historical["functions"][:2], strict=True):
        assert current["verdict"] == previous["verdict"]
        for metric in ("recall", "ppv_at_pi0", "behavior_purity"):
            assert current["metrics"][metric] == previous["metrics"][metric]
        assert current["behavior_validation"]["status"] == "historical"
        assert not current["ranking"]["eligible"]
    # A real detector change must still invalidate the old assessment.
    fn.source = "def misses_edge_case(item): return False"
    changed = next(card for card in refresh_report_data(saved, library)["functions"]
                   if card["name"] == fn.name)
    assert changed["behavior_validation"]["status"] == "stale"
    assert changed["metrics"]["behavior_purity"] == -1
    assert changed["verdict"]["code"] == "UNSCORED"


@pytest.mark.parametrize("rows,filters,expected", [
    pytest.param([
        {"model_id": "m", "benchmark": "b", "problem_id": "a", "is_correct": False},
        {"model_id": "m", "benchmark": "b", "problem_id": "b", "is_correct": True},
        {"model_id": "other", "benchmark": "b", "problem_id": "c", "is_correct": False},
        {"model_id": "m", "benchmark": "other", "problem_id": "d", "is_correct": False},
    ], {"model_id": "m", "benchmarks": ["b"]}, (2, 1, .5), id="filtered-cohort"),
    pytest.param([{"problem_id": "a", "is_correct": True},
                  {"problem_id": "b", "is_correct": None}], {}, (2, None, None),
                 id="unlabeled-rows-are-unknown"),
])
def test_empty_library_retains_exact_cohort_counts_without_inventing_failures(
    tmp_path, rows, filters, expected,
):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    dataset = Dataset([path], **filters)
    data = build_data(FunctionLibrary(model_id="m", benchmark="b"), dataset=dataset)
    meta = data["meta"]
    assert (meta["n_items"], meta["n_failures"], meta["base_fail_rate"]) == expected
    assert data["items"] == {}


def test_report_preserves_zero_failures_and_unknown_counts():
    lib = _sample_lib()
    lib.functions[0].scores.update(a=0, c=0, n_fail_caught=0, recall=0)
    data = build_data(lib)
    assert data["meta"]["n_failures"] == 0
    assert data["meta"]["base_fail_rate"] == 0
    empty = build_data(FunctionLibrary(model_id="m", benchmark="b"))
    assert empty["meta"]["n_items"] is None
    assert empty["meta"]["n_failures"] is None


def test_missing_population_stratum_never_reconstructs_rates_in_report_details():
    library = _sample_lib()
    library.functions[0].scores = {
        "evaluated": True, "sampled": True, "sample_n": 1, "sample_frame_n": 1000,
        "n_scored": 1, "evaluation_scope_complete": True, "evaluation_complete": False,
        "metrics_scope": "unrepresented_strata",
        "sampling_limitation": "Population rates unavailable: missing benchmark|fail.",
        "sample_unrepresented_strata": ["benchmark|fail"],
        "n_items": 900., "a": 0., "b": 900., "c": 0., "d": 0.,
        "n_flagged": 900., "n_rows_scored": 1,
        "recall": None, "precision": None, "raw_ppv": None, "ppv_at_pi0": None,
        "pct_caught": None, "purity": 0., "bucket": "True", "bucket_mode": "exact",
    }
    original = deepcopy(library.to_dict())
    data = build_data(library)
    metrics = data["functions"][0]["metrics"]
    assert metrics["precision"] is None
    assert metrics["pct_caught"] is None
    card = _render_card(data)
    assert "NOT EVALUATED" in card
    assert "100.0%" not in card
    assert library.to_dict() == original


@pytest.mark.parametrize("stale", [False, True])
def test_report_distinguishes_diagnostic_only_review_from_pending_adherence(stale):
    library = FunctionLibrary(functions=[_function("diagnostic_stale" if stale else
                                                   "diagnostic_empty")])
    original = deepcopy(library.to_dict())
    data = build_data(library)
    assert data["functions"][0]["metrics"]["behavior_purity"] == -1.
    assert data["functions"][0]["behavior_validation"]["sampled"] == 0
    assert "GOOD" not in _render_card(data)
    assert library.to_dict() == original


def test_final_review_receipt_displays_unavailable_reason_without_hiding_numeric_scores():
    library = _sample_lib()
    fn = library.functions[0]
    _current_review(
        fn, behavior_purity=1, behavior_n_judged=1, behavior_n_completed=1,
        behavior_n_yes=1, behavior_n_no=0, behavior_n_errors=0, behavior_n_uncertain=0,
    )
    fn.scores["final_review"] = {"status": "unavailable", "claim_hash": fn.scores["claim_hash"],
                                 "reason": "The remaining review needs <original evidence>."}
    original = deepcopy(library.to_dict())
    data = build_data(library)
    assert data["functions"][0]["final_review"] == fn.scores["final_review"]
    card = _render_card(data)
    assert 'tax-statcard-value">1.00</div>' in card
    assert "Final review unavailable: The remaining review needs &lt;original evidence&gt;." in card
    assert "<original evidence>" not in card
    assert library.to_dict() == original
    fn.source = "def misses_edge_case(item): return False"
    assert build_data(library)["functions"][0]["final_review"] is None


@pytest.mark.parametrize("overrides,label,tone", [
    ({}, "GOOD", "good"),
    ({"recall": .005}, "WEAK", "bad"),
    ({"ppv_at_pi0": .40}, "OK", "warn"),
    ({"a": 0, "b": 0, "n_fail_caught": 0, "n_flagged": 0,
      "precision": 0, "raw_ppv": 0, "recall": 0, "pct_caught": 0,
      "behavior_n_judged": 0}, "NO MATCHES", "neutral"),
    ({"errors": 3, "n_scored": 97, "evaluation_complete": False}, "GOOD", "good"),
    ({"evaluation_stale": True}, "NOT EVALUATED", "neutral"),
    ({"behavior_n_judged": 0}, "NOT EVALUATED", "neutral"),
])
def test_report_verdict_label_and_color_match_shared_classification(overrides, label, tone):
    library = _sample_lib()
    fn = _current_review(
        library.functions[0], ppv_at_pi0=.9, behavior_purity=1.0, behavior_n_judged=5,
        behavior_n_yes=5, behavior_n_no=0, behavior_n_uncertain=0, behavior_n_errors=0,
    )
    attach_task_support(fn)
    fn.scores.update(overrides)
    data = build_data(library)
    assert data["functions"][0]["verdict"]["label"] == label
    # Old frozen presentation must not override the current shared classifier.
    data["functions"][0]["verdict"] = {"code": "WRONG", "label": "OLD VERDICT"}
    card = _render_card(data)
    assert f'class="tax-card {tone}"' in card
    assert label in card and "OLD VERDICT" not in card
    if overrides == {"behavior_n_judged": 0}:
        # Waiting for the judge does not erase the completed detector evaluation.
        assert "40.0%" in card and 'tax-statcard-value">0.90</div>' in card


def test_report_refreshes_old_badge_from_observed_task_support():
    library = _sample_lib()
    fn = library.functions[0]
    fn.scores.update(ppv_at_pi0=.9, behavior_purity=1, behavior_n_judged=6,
                     behavior_n_yes=6, behavior_n_no=0,
                     behavior_n_uncertain=0, behavior_n_errors=0)
    attach_task_support(fn, task_n=3, repeat_n=2)
    before = deepcopy(library.to_dict())
    data = build_data(library)
    metrics = data["functions"][0]["metrics"]
    assert metrics["n_confirmed_failed_tasks"] == 3
    assert metrics["confirmed_task_support_status"] == "verified"
    assert data["functions"][0]["verdict"]["label"] == "OK"
    # Old HTML has neither the count nor the new support condition in its badge.
    old = deepcopy(data)
    for field in ("n_confirmed_failed_tasks", "confirmed_task_support_status"):
        old["functions"][0]["metrics"].pop(field)
    old["functions"][0]["verdict"] = {"code": "GOOD", "label": "GOOD"}
    old["functions"][0]["badge"] = "GOOD"
    refreshed = refresh_report_data(old, library)
    assert refreshed["functions"][0]["verdict"]["label"] == "OK"
    assert refreshed["functions"][0]["badge"] == "OK"
    assert refreshed["functions"][0]["metrics"] == metrics
    assert library.to_dict() == before


def test_report_preserves_scoring_cohort_task_count_separately_from_verified_breadth():
    library = _sample_lib()
    fn = library.functions[0]
    attach_task_support(fn, task_n=7)
    # Later supported matches can extend breadth without changing the already
    # recorded scoring-panel task recall and its observed numerator.
    fn.scores.update(task_recall=2 / 79, task_recall_scope="observed_lower_bound",
                     task_recall_claim_hash=fn.scores["evaluation_claim_hash"],
                     n_failed_tasks=79, n_matched_failed_tasks=2, n_matched_failed_attempts=2)
    before = deepcopy(library.to_dict())
    metrics = build_data(library)["functions"][0]["metrics"]
    assert metrics["recall"] == .4
    assert metrics["task_recall"] == 2 / 79
    assert metrics["task_recall_scope"] == "observed_lower_bound"
    assert metrics["n_evaluated_matched_failed_tasks"] == 2
    assert metrics["n_evaluated_matched_failed_attempts"] == 2
    assert metrics["n_matched_failed_tasks"] == metrics["n_matched_failed_attempts"] == 7
    assert library.to_dict() == before


@pytest.mark.parametrize("sampled,label", [(True, "GOOD"), (False, "OK")])
def test_report_compares_validated_projection_and_keeps_observed_confirmations(sampled, label):
    library = _sample_lib()
    fn = library.functions[0]
    fn.scores.update(ppv_at_pi0=.9, behavior_purity=1, behavior_n_judged=2,
                     behavior_n_yes=2, behavior_n_no=0,
                     behavior_n_uncertain=0, behavior_n_errors=0)
    attach_sampled_task_support(fn)
    # The previous projection cannot qualify a full-corpus assessment.
    fn.scores["sampled"] = sampled
    before = deepcopy(library.to_dict())
    data = build_data(library)
    metrics = data["functions"][0]["metrics"]
    assert metrics["n_confirmed_failed_tasks"] == 2
    assert metrics["estimated_n_confirmed_failed_tasks"] == (
        pytest.approx(8) if sampled else None)
    assert metrics["task_support_comparison_basis"] == ("estimated" if sampled else "observed")
    assert data["functions"][0]["verdict"]["label"] == label
    refreshed = refresh_report_data(deepcopy(data), library)
    assert refreshed["functions"][0]["metrics"] == metrics
    assert refreshed["functions"][0]["verdict"]["label"] == label
    assert library.to_dict() == before


def test_unavailable_current_evaluation_hides_saved_metrics_and_keeps_independent_review():
    library = _sample_lib()
    _current_review(
        library.functions[0], n_scored=100, recall=.1, ppv_at_pi0=.95,
        behavior_purity=.9, behavior_n_judged=10, behavior_n_yes=9, behavior_n_no=1,
        behavior_n_uncertain=0, behavior_n_errors=0, evaluation_stale=True,
    )
    card = _render_card(build_data(library))
    tones = _tones(card)
    assert "NOT EVALUATED" in card
    assert tones["Coverage"] == tones["Normalized precision"] == "neutral"
    assert tones["Detector adherence"] == "green"
    assert 'tax-statcard-value">0.90</div>' in card
    assert "10.0%" not in card and "0.95" not in card
    assert '<table class="bucket-table">' not in card


def test_report_no_matches_uses_displayed_parent_cohort_without_replacing_global_counts():
    library = _sample_lib()
    library.functions[0].scores.update(
        within_parent="parent", within_recall=0, within_n_flagged=0,
    )
    card = build_data(library)["functions"][0]
    assert card["metrics"]["n_flagged"] == 30
    assert card["metrics"]["badge_n_flagged"] == 0
    assert card["verdict"]["label"] == "NO MATCHES"


def test_partial_evaluation_preserves_scope_and_qualifies_resolved_only_coverage():
    errors, insufficient = 49, 50
    library = _sample_lib()
    library.functions[0].scores = {
        **contingency(1, 0, 0, 0), "evaluated": True, "n_scored": 1,
        "errors": errors, "n_insufficient": insufficient, "n_unresolved": 99,
        "metrics_scope": "resolved_items", "assessment_coverage": 0.01,
        "evaluation_complete": False,
        "unresolved_bounds": {"recall": [0.01, 1], "fpr": None},
        "within_errors": errors, "within_n_insufficient": insufficient,
        "within_n_unresolved": 99, "within_assessment_coverage": 0.01,
        "within_unresolved_bounds": {"recall": [0.01, 1], "fpr": None},
    }
    data = build_data(library)
    evaluation = data["functions"][0]["evaluation"]
    assert evaluation["metrics_scope"] == "resolved_items"
    assert evaluation["errors"] == errors
    assert evaluation["n_insufficient"] == insufficient
    assert evaluation["assessment_coverage"] == 0.01
    assert evaluation["within_unresolved_bounds"] == evaluation["unresolved_bounds"]
    # Resolved-only counts must not masquerade as the original dataset's size.
    assert data["meta"]["n_items"] is None
    assert data["meta"]["n_failures"] is None
    assert "Resolved rollouts only" in _render_card(data)


def test_all_unresolved_evaluation_displays_unavailable_in_real_details_and_buckets():
    library = _sample_lib()
    library.functions[0].scores = {
        **contingency(0, 0, 0, 0), "evaluated": True, "n_scored": 0,
        "errors": 40, "n_insufficient": 60, "n_unresolved": 100,
        "metrics_scope": "resolved_items", "assessment_coverage": 0,
        "evaluation_complete": False,
        "unresolved_bounds": {"recall": [0, 1], "fpr": [0, 1]},
        "bucket": "True", "buckets": [], "n_items": 100,
    }
    data = build_data(library)
    function = data["functions"][0]
    assert function["n_scored"] == 0
    assert function["buckets"] == [] and function["best_bucket"] is None
    # Previously generated payloads can contain the spurious numeric bucket.
    function["buckets"] = [{"bucket": "True", "purity": 0, "recall": 0,
                            "ppv_at_pi0": 0, "n_fail": 0, "n": 0}]
    function["best_bucket"] = function["buckets"][0]
    card = _render_card(data)
    assert "No resolved detector results; association metrics are unavailable" in card
    assert "0.0%" not in card
    assert '<table class="bucket-table">' not in card


def test_historical_single_bucket_fallback_still_renders_when_bucket_list_is_absent():
    library = _sample_lib()
    scores = library.functions[0].scores
    scores.pop("buckets")
    scores.update(bucket="True", n=30, n_fail=20, n_pass=10, purity=2 / 3)
    data = build_data(library)
    assert data["functions"][0]["buckets"][0]["bucket"] == "True"
    assert '<table class="bucket-table">' in _render_card(data)


def test_complete_legacy_report_keeps_coverage_without_incomplete_qualifiers():
    lib = _sample_lib()
    # Coverage shows failure recall, never the fraction of all items flagged.
    lib.functions[0].scores.update(pct_caught=0.03, recall=0.14)
    card = _render_card(build_data(lib))
    coverage = re.search(r'label">Coverage.*?statcard-value">([^<]+)<', card, re.S)
    assert coverage.group(1) == "14.0%"
    assert "Resolved rollouts only" not in card


def test_partial_detector_counts_fall_back_to_actual_dataset_cohort():
    library = _sample_lib()
    library.functions[0].scores.update(metrics_scope="resolved_items", n_items=1, a=1, c=0)
    rows = [SimpleNamespace(model_id="test-model", benchmark="bench-a", is_correct=i < 20)
            for i in range(100)]
    dataset = SimpleNamespace(model_id="test-model", allowed_benchmarks=lambda: {"bench-a"},
                              identity_index=lambda: SimpleNamespace(rows=rows))
    data = build_data(library, dataset=dataset, items={"example": {"source_only": True}})
    assert data["meta"]["n_items"] == 100
    assert data["meta"]["n_failures"] == 80
    assert data["meta"]["base_fail_rate"] == 0.8


def test_report_visibility_keeps_hidden_candidates_in_payload_and_counts_visible_only():
    def row(name, recall, precision, adherence):
        return {"name": name, "metrics": {
            "recall": recall, "precision": precision, "ppv_at_pi0": precision,
            "behavior_purity": adherence, "n_matched_failed_tasks": 5,
        }}

    rows = [row("weak", .9, .3, 1), row("ok", .8, .5, .8),
            row("good_narrow", .03, .9, 1), row("good_broad", .2, .9, 1),
            row("pending", None, None, None)]
    original = deepcopy(rows)
    saved = _extract_autofa_data(render_html({"functions": rows}))
    assert rows == original
    assert {fn["name"] for fn in saved["functions"]} == {fn["name"] for fn in rows}
    assert saved["meta"]["n_functions"] == 3
    assert saved["meta"]["n_hidden_functions"] == 2
    assert saved["meta"]["n_pending_scores"] == 1
    # Reopening an already rendered report must not accumulate hidden/pending counts.
    assert _extract_autofa_data(render_html(saved))["meta"] == saved["meta"]


def test_input_visibility_is_preserved_in_report_without_private_paths():
    scope = "89 original traces retrieved. Artifact file contents not inspected."
    data = build_data(_sample_lib(), reference_data=[{
        "run_id": "run", "export_url": "https://example.test/v1/runs/run/export/autofa",
        "source_harness": "hermes", "parquet_path": "/private/cache/dataset.parquet",
        "evidence_scope": scope,
    }])
    reference = data["meta"]["reference_data"][0]
    assert reference["evidence_scope"] == scope
    assert "parquet_path" not in reference
    assert scope in render_html(data)


def test_failure_mode_cards_have_stable_namespaced_deep_links():
    name = "misses_edge_case"
    card_id = failure_mode_card_id(name)
    assert card_id == failure_mode_card_id(name)
    assert card_id != failure_mode_card_id("misses_edge_case_v2")
    assert card_id.startswith("failure-mode-misses-edge-case-")
    data = build_data(_sample_lib())
    assert data["functions"][0]["card_id"] == card_id
    html = render_html(data)
    assert f'"card_id": "{card_id}"' in html
    # Embedding hosts rely on this message contract to request a failure-mode link.
    assert "namespace: 'autofa.failure-cards'" in html
    assert "type: 'failure-link-request'" in html


def test_mechanism_prose_warning_flags_detector_rules_but_accepts_failure_stories():
    rule = ("Fires when the trace shows many non-zero exit codes or the same error "
            "signature repeated many times.")
    assert mechanism_prose_warning(rule, rule) is not None
    story = ("The model hits a non-zero exit, then re-runs nearly the same command "
             "instead of diagnosing the root cause, so the grader never gets a patch.")
    assert mechanism_prose_warning(
        story, "Retries the same failing command until the budget dies") is None


def test_write_failure_cards_spills_full_text_sidecars(tmp_path: Path):
    long_prompt = "PROMPT_START " + ("x" * 20_000) + " PROMPT_END"
    long_reason = "REASON_START " + ("y" * 30_000) + " REASON_END"
    items = {"m|b|id1": {
        "item_id": "m|b|id1", "label": 0, "is_correct": False,
        "prompt_head": long_prompt, "reasoning_head": long_reason,
        "response_head": "full response body", "judge_head": "",
        "prompt_chars": len(long_prompt), "reasoning_chars": len(long_reason),
        "response_chars": 17,
    }}
    data = build_data(_sample_lib(), items=items)
    assert data["meta"]["n_embedded_items"] == 1
    out = tmp_path / "failure_cards.html"
    spilled = _spill_items_to_disk(data, out_path=out)
    out.write_text(render_html(spilled), encoding="utf-8")
    assert spilled["items"] == {}
    side = tmp_path / "failure_cards_data" / "items" / spilled["item_index"]["m|b|id1"]
    payload = json.loads(side.read_text(encoding="utf-8"))
    assert payload["prompt_head"].endswith("PROMPT_END")
    assert payload["reasoning_head"].endswith("REASON_END")
    # HTML itself stays small (no full text inline).
    assert out.stat().st_size < 500_000
    # Cloud assets keep the items_base-relative path the library server resolves lazily.
    with zipfile.ZipFile(io.BytesIO(build_report_assets_zip(tmp_path))) as archive:
        assert archive.namelist() == [f"failure_cards_data/items/{side.name}"]
        assert json.loads(archive.read(archive.namelist()[0])) == payload


def test_write_failure_cards(tmp_path: Path):
    out = write_failure_cards(_sample_lib(), out=tmp_path / "failure_cards.html")
    text = out.read_text(encoding="utf-8")
    assert "autofa-data" in text and "misses_edge_case" in text
    assert file_url(out).startswith("file://")


def test_build_report_bundle_zip_embeds_trajectories(tmp_path: Path):
    catalog = tmp_path / "library" / "nano-v3.5" / "swebench-verified"
    catalog.mkdir(parents=True)
    (catalog / "library.json").write_text('{"functions":[]}', encoding="utf-8")
    (catalog / "config.json").write_text('{"parquet_paths":["/private/local/dataset.parquet"]}',
                                         encoding="utf-8")
    write_failure_cards(_sample_lib(), out=catalog / "failure_cards.html")
    data_dir = catalog / "failure_cards_data" / "items"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Live/server mode: the HTML indexes a spilled sidecar it fetches over HTTP.
    (data_dir / "deadbeef.json").write_text(json.dumps({
        "item_id": "m|b|id1", "prompt_head": "PROMPT_BODY",
        "response_head": "RESPONSE_BODY", "label": 0,
    }), encoding="utf-8")
    live = _extract_autofa_data((catalog / "failure_cards.html").read_text(encoding="utf-8"))
    live["items"] = {}
    live["item_index"] = {"m|b|id1": "deadbeef.json"}
    live.setdefault("meta", {})["items_base"] = "failure_cards_data/items"
    (catalog / "failure_cards.html").write_text(render_html(live), encoding="utf-8")

    payload, name = build_report_bundle_zip(catalog)
    assert name == bundle_zip_filename(catalog) == "nano-v3.5_swebench-verified_autofa.zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        assert {"failure_cards.html", "library.json", "README.txt"} <= names
        assert "config.json" not in names  # Local paths stay private.
        assert not any(n.startswith("failure_cards_data/") for n in names)
        offline = _extract_autofa_data(zf.read("failure_cards.html").decode("utf-8"))
    assert offline["items"]["m|b|id1"]["prompt_head"] == "PROMPT_BODY"
    assert not offline.get("item_index")
    assert not offline["meta"].get("items_base")
    assert offline["meta"]["offline_export"] is True


def test_serve_failure_cards_bundle_endpoint(tmp_path: Path):
    catalog = tmp_path / "m" / "b"
    catalog.mkdir(parents=True)
    (catalog / "library.json").write_text('{"functions":[]}', encoding="utf-8")
    html = write_failure_cards(_sample_lib(), out=catalog / "failure_cards.html")
    base = serve_failure_cards(html).rsplit("/", 1)[0]
    with urllib.request.urlopen(f"{base}/autofa-bundle.zip", timeout=5) as resp:
        assert resp.status == 200
        assert "zip" in (resp.headers.get("Content-Type") or "")
        assert "filename=" in (resp.headers.get("Content-Disposition") or "")
        assert resp.read()[:2] == b"PK"


def _policy_function(profile, verdict="yes"):
    source = "def f(item): return True"
    return FeatureFunction(
        name=profile, source=source, description="Claim", mechanism="Claim", scores={
            "evaluated": True, "n_items": 100, "n_scored": 100,
            "recall": .4, "precision": .9, "ppv_at_pi0": .9, "pct_caught": .4,
            "n_flagged": 10, "n_fail_caught": 9,
            "judge_contract_version": JUDGE_CONTRACT_VERSION,
            "judge_behavior_strictness": profile,
            "judge_policy_hash": judge_policy_hash(behavior_strictness=profile),
            "claim_hash": claim_hash(source=source, description="Claim", mechanism="Claim"),
            "behavior_purity": float(verdict == "yes"), "behavior_n_judged": 1,
            "behavior_n_completed": int(verdict in {"yes", "no"}),
            "behavior_n_yes": int(verdict == "yes"), "behavior_n_no": int(verdict == "no"),
            "behavior_n_uncertain": int(verdict == "uncertain"), "behavior_n_errors": 0,
            "behavior_verdicts": [{"item_id": "example", "verdict": verdict,
                                    "rationale": "The trace shows <script>run()</script>."}],
        },
    )


def _render_policy_views(data):
    """Execute the card, item-validation and summary renderers for every function."""
    output = run_node(
        report_renderers(render_html(data))
        + "\nconsole.log(JSON.stringify(" + json.dumps(data["functions"])
        + ".map(fn => ({name: fn.name, card: renderCard(fn, 1, 1), "
          "item: renderItemValidation(fn, 'example'), "
          "summary: renderBehaviorSummary(fn.behavior_validation, fn.judge_rationale)}))));"
    )
    return {row["name"]: row for row in json.loads(output)}


def test_mixed_library_reports_saved_policy_independently_of_active_settings():
    functions = [_policy_function(profile) for profile in ("behavioral", "contextual", "causal")]
    before = deepcopy([fn.to_dict() for fn in functions])
    data = build_data(FunctionLibrary(functions=functions))
    rows = _render_policy_views(data)
    for fn in data["functions"]:
        assert fn["behavior_validation"]["judge_behavior_strictness"] == fn["name"]
        assert fn["behavior_validation"]["status"] == "confirmed"
    for profile in ("behavioral", "contextual", "causal"):
        row = rows[profile]
        label = f"Judge policy: {STRICTNESS_LABELS[profile]}."
        assert all(label in row[view] for view in ("summary", "item", "card"))
        # Judge rationales quote untrusted trace text; every view escapes it.
        assert "&lt;script&gt;run()" in row["summary"]
        assert not any("<script>" in row[view] for view in ("summary", "item", "card"))
        # Only a causal review may claim a contribution to the task failure.
        caused = "material contribution to the task failure" in row["card"]
        assert caused == (profile == "causal")
    assert [fn.to_dict() for fn in functions] == before


def test_historical_reviews_remain_historical_when_selected_policy_changes():
    data = build_data(FunctionLibrary(functions=[_policy_function("causal")]))
    validation = data["functions"][0]["behavior_validation"]
    validation["judge_contract_version"] = None
    validation.pop("judge_behavior_strictness")
    # Published historical reports can retain their old confirmed status.
    validation["status"] = "confirmed"
    row = _render_policy_views(data)["causal"]
    assert "historical review" in row["item"]
    assert "Judge policy:" not in row["summary"]
    assert "contributed to the task failure" not in row["summary"]


@pytest.mark.parametrize("policy,label", [
    # The current contract without the newer saved field still identifies causal review.
    (None, STRICTNESS_LABELS["causal"]),
    ("__proto__", "Unknown policy"),
])
def test_missing_or_unknown_saved_policy_never_invents_causal_confirmation(policy, label):
    data = build_data(FunctionLibrary(functions=[_policy_function("causal")]))
    validation = data["functions"][0]["behavior_validation"]
    if policy is None:
        validation.pop("judge_behavior_strictness")
    else:
        validation["judge_behavior_strictness"] = policy
    row = _render_policy_views(data)["causal"]
    assert label in row["summary"]
    known = policy is None
    assert ("material contribution to the task failure" in row["card"]) == known
    if not known:
        assert "historical review" in row["item"]
