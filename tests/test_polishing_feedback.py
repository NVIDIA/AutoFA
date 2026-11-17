# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closing repair feedback names evidence and pages every retained control."""

import json
from copy import deepcopy

import pytest

from autofa.mcp.tools_library import get_function_feedback
from autofa.session.polishing_evaluation import run_polishing_evaluation
from autofa.session.polishing_feedback import compact_polishing_feedback, polishing_claim_identity
from autofa.session.validation_controls import collect_full_validation_controls

from support.discovery import _function
from support.evaluation import PANEL_N, write_phase
from support.evaluation import polishing_runtime as polishing_runtime


def _start(polishing_runtime):
    rt, fn, before, _ = polishing_runtime
    before = {**before, "ppv_at_pi0": .9, "recall": .5}
    controls = collect_full_validation_controls(rt, fn, previous_scores=before)
    write_phase(rt, frozen_names=[fn.name], controls={fn.name: controls},
                baseline_grades={"functions": [{"name": fn.name, "code": "GOOD"}]})
    return rt, fn, controls


def test_feedback_tool_pages_entire_uncapped_control_union_with_current_outcomes(polishing_runtime):
    rt, fn, controls = _start(polishing_runtime)
    assert run_polishing_evaluation(rt, fn, controls)["complete"]
    original = deepcopy(fn.scores)
    rows, offset = [], 0
    while offset is not None:
        response = get_function_feedback(fn.name, offset=offset, limit=30)
        assert response["ok"], response
        feedback = response["polishing_feedback"]
        assert feedback["population_metrics"] is False
        assert feedback["diagnostic"]["status"] == "complete"
        assert len(feedback["cases"]) <= 30
        rows.extend(feedback["cases"])
        offset = feedback["pagination"]["next_offset"]
    assert {row["item_id"] for row in rows} == set(controls["item_ids"])
    assert len(rows) == len(controls["item_ids"])
    assert all(row["diagnostic"]["status"] == "value" for row in rows)
    assert fn.scores == original


def test_previews_bound_review_reasons_and_item_feedback_keeps_full_findings(polishing_runtime):
    from autofa.session.round_validation import validation_feedback

    rt, fn, controls = _start(polishing_runtime)
    row = controls["controls"][0]
    reason = "Detailed causal evidence. " * 1000 + "The later recovery breaks the claim."
    row["review_findings"] = [{"verdict": "no", "reason": reason}]
    write_phase(rt, frozen_names=[fn.name], controls={fn.name: controls})
    round_feedback = validation_feedback(rt)["detectors"][0]["polishing_feedback"]
    assert "cases" not in round_feedback and len(round_feedback["cases_preview"]) <= 5
    assert reason not in json.dumps(round_feedback)
    full = get_function_feedback(fn.name, item_id=row["item_id"])["polishing_feedback"]
    assert len(full["cases"]) == 1
    assert full["cases"][0]["review_findings"] == row["review_findings"]


def test_saved_diagnostic_is_not_reused_after_another_edit(polishing_runtime):
    rt, fn, controls = _start(polishing_runtime)
    assert run_polishing_evaluation(rt, fn, controls)["complete"]
    fn.source += "\n# another revision\n"
    feedback = get_function_feedback(fn.name)["polishing_feedback"]
    assert feedback["diagnostic"]["status"] == "not_run_for_current_version"
    assert all(row["diagnostic"]["status"] == "not_run_for_current_version"
               for row in feedback["cases"])
    assert feedback["revision_comparison"]["comparable"] is False
    assert feedback["revision_comparison"]["metric_changes"] == {}


def test_revision_comparison_needs_both_versions_resolved_on_the_same_panel(polishing_runtime):
    rt, fn, saved_controls = _start(polishing_runtime)
    fn.source = fn.source.replace("'BAD'", "'fine'")  # Flags exactly the other panel rows.
    saved_diagnostic = run_polishing_evaluation(rt, fn, saved_controls)
    claim = polishing_claim_identity(fn)
    comparison = compact_polishing_feedback(fn.name, saved_controls, saved_diagnostic,
                                            current_claim=claim)["revision_comparison"]
    assert comparison["comparable"] and comparison["population_metrics"] is False
    assert comparison["panel_n"] == PANEL_N < saved_diagnostic["n"]
    assert comparison["cohort_fingerprint"] == saved_controls["panel_fingerprint"]
    assert all(comparison[side]["resolved_n"] == PANEL_N for side in ("before", "after"))
    recall = comparison["metric_changes"]["recall"]
    assert recall["before"] + recall["after"] == pytest.approx(1)
    item_id = saved_controls["panel_item_ids"][0]
    for missing in ("abstention", "legacy_outcome", "baseline_prediction"):
        diagnostic, controls = deepcopy(saved_diagnostic), deepcopy(saved_controls)
        row = next(row for row in controls["controls"] if row["item_id"] == item_id)
        if missing == "abstention":
            next(row for row in diagnostic["results"]
                 if row["item_id"] == item_id)["status"] = "abstained"
        elif missing == "legacy_outcome":
            row.pop("benchmark_outcome")
        else:
            row.pop("baseline_label")
        comparison = compact_polishing_feedback(
            fn.name, controls, diagnostic, current_claim=claim,
        )["revision_comparison"]
        assert comparison["comparable"] is False, missing
        assert sum(comparison[side]["unresolved_n"] for side in ("before", "after")) >= 1
        assert all(row["delta"] is None for row in comparison["metric_changes"].values())


def test_compact_feedback_bounds_ids_and_outcomes():
    controls = {"available": True, "complete": True, "panel_item_ids": [],
                "item_ids": [str(i) for i in range(10000)], "controls": [
                    {"item_id": str(i), "benchmark": "bench",
                     "categories": ["flagged_success_contrast"], "baseline_label": True}
                    for i in range(10000)]}
    diagnostic = {"complete": True, "completed": 10000, "n": 10000, "results": [],
                  "groups": {"flagged_success_contrast": {
                      "n": 10000, "lost_matches": controls["item_ids"]}}}
    feedback = compact_polishing_feedback("example", controls, diagnostic)
    assert feedback["total"] == 10000
    assert len(feedback["cases_preview"]) == 5
    group = feedback["groups"]["flagged_success_contrast"]
    assert group["lost_matches"]["n"] == 10000
    assert len(group["lost_matches"]["item_ids_preview"]) == 3
    assert "item_ids" not in feedback and "results" not in feedback


def test_still_firing_counterexamples_do_not_transfer_old_verdicts_to_the_revision():
    current_claim = {"claim_hash": "new-revision", "source_sha256": "new-source",
                     "description": "Same claim", "mechanism": "Same mechanism"}
    finding = {"claim_hash": "old-revision", "source_sha256": "old-source", "verdict": "no",
               "claim_description": "Same claim", "claim_mechanism": "Same mechanism",
               "reason": "The later successful recovery contradicts the claimed omission."}
    rows = [{"item_id": name, "categories": ["reviewed_counterexample"],
             "review_findings": [deepcopy(finding)]}
            for name in ("still-matches", "excluded", "abstained", "not-checked")]
    controls = {"selection": ["positive", "legacy_boolean"], "controls": rows,
                "item_ids": [row["item_id"] for row in rows], "panel_item_ids": []}
    diagnostic = {"claim_hash": "new-revision", "complete": False, "results": [
        {"item_id": "still-matches", "status": "value", "prediction": True},
        {"item_id": "excluded", "status": "value", "prediction": False},
        {"item_id": "abstained", "status": "abstained", "prediction": None},
    ]}
    fn = _function("example", .08)
    before = deepcopy(fn.scores)
    result = compact_polishing_feedback(
        fn.name, controls, diagnostic, current_scores=fn.scores, current_claim=current_claim,
    )
    residuals = result["known_counterexamples"]
    assert residuals["still_matching_n"] == 1
    assert residuals["excluded_n"] == 1 and residuals["unresolved_n"] == 2
    assert residuals["status"] == "requires_reconciliation"
    assert residuals["cases_preview"][0]["item_id"] == "still-matches"
    assert result["summary"]["current_blockers"] == []  # Existing Good thresholds are unchanged.
    assert residuals["changes_adherence"] is False
    assert residuals["current_negative_verdicts_inferred"] is False
    assert fn.scores == before
    # Predictions from another revision are pending checks, not residual matches.
    diagnostic["claim_hash"] = "another-revision"
    stale = compact_polishing_feedback("example", controls, diagnostic,
                                       current_claim=current_claim)["known_counterexamples"]
    assert stale["status"] == "checks_pending"
    assert stale["still_matching_n"] == 0 and stale["unresolved_n"] == 4
