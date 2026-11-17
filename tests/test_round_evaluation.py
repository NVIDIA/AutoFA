# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round validation expands measured evidence without changing discovery sampling."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest

from autofa.core.library import FeatureFunction
from autofa.mcp import tools_eval
from autofa.metrics.sampling import ROUND_SAMPLE_VERSION, proportional_sample

from support.evaluation import (
    _LLM_SOURCE,
    _REGEX_SOURCE,
    _FakeEvalDataset,
    _llm_flag,
    _record_calls,
    _runtime,
)


@pytest.fixture
def round_runtime(tmp_path, monkeypatch):
    # 100 rows (20 failed) and a 20-row discovery panel: a 30% audit (30 rows)
    # exceeds the panel, while an 80% checkpoint and a census exceed both.
    fn = _llm_flag()
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=3, functions=[fn])
    rt.dataset = _FakeEvalDataset([
        {"model_id": "m", "benchmark": "bench", "problem_id": str(i), "run_idx": 0,
         "is_correct": i % 5 != 0, "response_text": "BAD" if i % 7 == 0 else "fine",
         "_item_id": f"id-{i}"} for i in range(100)
    ])
    return rt, fn, _record_calls(monkeypatch, lambda source, item: (source, item["_item_id"]))


def test_proportional_validation_preserves_outcome_prevalence_and_weights():
    frame = {f"f-{i}": "bench|fail" for i in range(562)}
    frame.update({f"p-{i}": "bench|pass" for i in range(3151)})
    sample = proportional_sample(frame, fraction=.3, seed=19)
    assert sample.strata["bench|fail"]["sampled"] == 169
    assert sample.strata["bench|pass"]["sampled"] == 946
    assert sample.sample_n == 1115
    assert sum(sample.weights.values()) == pytest.approx(3713)
    assert sample.item_ids == proportional_sample(
        dict(reversed(list(frame.items()))), fraction=.3, seed=19,
    ).item_ids
    assert sample.item_ids != proportional_sample(frame, fraction=.3, seed=20).item_ids


@pytest.mark.parametrize("fraction", [0, 1.01, True, "0.3"])
def test_invalid_round_fraction_does_not_access_runtime(fraction, monkeypatch):
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: pytest.fail("Invalid request"))
    assert tools_eval.evaluate_round_functions(fraction=fraction)["ok"] is False
    with pytest.raises(ValueError, match="fraction"):
        proportional_sample({}, fraction=fraction, seed=1)


def test_small_strata_and_discovery_scoring_budget_survive_proportional_checkpoint():
    frame = {"rare": "rare|fail", **{str(i): "main|pass" for i in range(100)}}
    sample = proportional_sample(frame, fraction=.3, seed=1)
    assert sample.strata["rare|fail"]["sampled"] == 1
    assert sample.strata["main|pass"]["sampled"] == 30
    assert not sample.is_census()
    assert proportional_sample(frame, fraction=1, seed=1).is_census()
    # A proportional checkpoint never shrinks below the discovery scoring budget.
    frame = {str(i): "b|fail" if i % 5 == 0 else "b|pass" for i in range(600)}
    sample = proportional_sample(frame, fraction=.3, seed=1, minimum_n=200)
    assert sample.sample_n == 200
    assert sample.strata["b|fail"]["sampled"] == 40
    assert sample.strata["b|pass"]["sampled"] == 160
    assert proportional_sample(frame, fraction=.3, seed=1, minimum_n=1000).is_census()


def test_round_validation_reuses_predictions_and_keeps_discovery_panel(round_runtime):
    rt, fn, calls = round_runtime
    original_config = deepcopy(vars(rt.config))
    assert tools_eval.evaluate_functions()["ok"]
    discovery_ids = set(fn.scores["evaluation_item_ids"])
    original_panel = deepcopy(fn.scores["search_panel"])
    assert len(discovery_ids) == len(calls) == 20

    result = tools_eval.evaluate_round_functions()
    assert result["ok"] and result["evaluated_names"] == [fn.name]
    scores = fn.scores
    audit_ids = set(scores["evaluation_item_ids"])
    assert len(audit_ids) == scores["sample_n"] == scores["n_scored"] == 30
    assert scores["sample_frame_n"] == 100
    assert scores["sample_allocation_version"] == ROUND_SAMPLE_VERSION
    assert scores["sample_stratum_counts"]["bench|fail"]["sampled"] == 6
    assert scores["sample_stratum_counts"]["bench|pass"]["sampled"] == 24
    assert scores["search_panel"] == original_panel
    assert scores["round_validation"]["complete"] is True
    assert scores["round_validation"]["corpus_fingerprint"] == scores[
        "task_recall_corpus_fingerprint"]
    assert len(calls) == len(discovery_ids | audit_ids)
    assert set(fn.failure_localisations) <= audit_ids
    assert vars(rt.config) == original_config

    expected = Counter()
    for item in rt.dataset._fake_items:
        if item["_item_id"] in audit_ids:
            key = (not item["is_correct"], "BAD" in item["response_text"])
            expected[key] += 1
    assert scores["recall"] == pytest.approx(expected[True, True] / 6)
    assert scores["fpr"] == pytest.approx(expected[False, True] / 24)
    assert scores["n_flagged"] == pytest.approx(
        (expected[True, True] + expected[False, True]) / .3)


def test_larger_checkpoint_and_full_census_cannot_be_replaced_by_small_panel(round_runtime):
    rt, fn, calls = round_runtime
    assert tools_eval.evaluate_functions()["ok"]
    assert tools_eval.evaluate_round_functions()["ok"]
    checkpoint = deepcopy(fn.scores)
    spent = len(calls)
    rt.run.round += 1
    for evaluate in (tools_eval.evaluate_functions, tools_eval.evaluate_round_functions):
        result = evaluate()
        assert result["ok"] and result["reused_names"] == [fn.name]
        assert not result["evaluated_names"]
        assert fn.scores == checkpoint and len(calls) == spent

    assert tools_eval.final_evaluate_functions()["ok"]
    assert len(calls) == 100
    census = deepcopy(fn.scores)
    assert not census["sampled"] and census["n_scored"] == 100
    assert "round_validation" not in census
    for evaluate in (tools_eval.evaluate_functions, tools_eval.evaluate_round_functions):
        result = evaluate()
        assert result["reused_names"] == [fn.name]
        assert fn.scores == census and len(calls) == 100


def test_revised_detector_does_not_reuse_old_authority(round_runtime):
    rt, fn, calls = round_runtime
    assert tools_eval.evaluate_round_functions()["ok"]
    prior = deepcopy(fn.scores["round_validation"])
    spent = len(calls)
    fn.source += "\n# revised version\n"
    assert tools_eval.evaluate_functions()["evaluated_names"] == [fn.name]
    assert fn.scores["sample_n"] == 20
    assert "round_validation" not in fn.scores
    assert len(calls) == spent + 20
    assert tools_eval.evaluate_round_functions()["evaluated_names"] == [fn.name]
    assert fn.scores["round_validation"]["panel_fingerprint"] == prior["panel_fingerprint"]


def test_round_validation_keeps_already_complete_python_scores(round_runtime):
    rt, fn, calls = round_runtime
    python = FeatureFunction(name="plain", description="local", source=_REGEX_SOURCE)
    rt.library.functions.append(python)
    assert tools_eval.evaluate_functions()["ok"]
    census = deepcopy(python.scores)
    result = tools_eval.evaluate_round_functions()
    assert result["evaluated_names"] == [fn.name]
    assert result["reused_names"] == [python.name]
    assert python.scores == census
    assert sum(source == python.source for source, _ in calls) == 100
    assert python.scores["search_panel"]["cohort_fingerprint"] == fn.scores[
        "search_panel"]["cohort_fingerprint"]


def test_round_validation_preserves_unresolved_status_and_receipts(round_runtime, monkeypatch):
    from autofa.core.sandbox import DetectorAbstained

    rt, fn, _ = round_runtime
    original = tools_eval.run_detector

    def abstain(source, item, **kwargs):
        if int(item["problem_id"]) % 3 == 0:
            raise DetectorAbstained("Missing retained evidence")
        return original(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", abstain)
    fn.scores["round_review"] = {"status": "pending", "operation_id": "saved"}
    assert tools_eval.evaluate_round_functions()["ok"]
    assert fn.scores["n_insufficient"] > 0 and fn.scores["n_unresolved"] > 0
    assert fn.scores["n_scored"] + fn.scores["n_unresolved"] == 30
    assert fn.scores["round_validation"]["complete"] is True
    assert fn.scores["evaluation_complete"] is False
    assert fn.scores["round_review"]["operation_id"] == "saved"


def test_changed_endpoint_or_source_content_invalidates_final_census(round_runtime):
    rt, fn, calls = round_runtime
    assert tools_eval.final_evaluate_functions()["censused"] == [fn.name]
    context, flagged = fn.scores["evaluation_context"], fn.scores["n_flagged"]
    assert tools_eval.final_evaluate_functions()["censused"] == []
    rt.config.feature_model = "changed-detector-model"
    assert tools_eval.final_evaluate_functions()["censused"] == [fn.name]
    assert fn.scores["evaluation_context"] != context
    assert len(calls) == 200
    rt.dataset._fake_items[0]["response_text"] = "fine"
    assert tools_eval.final_evaluate_functions()["censused"] == [fn.name]
    assert fn.scores["n_flagged"] == flagged - 1
    assert len(calls) == 201  # Only the changed input needs a fresh detector execution.


def test_unknown_source_identity_does_not_certify_authoritative_reuse(round_runtime, monkeypatch):
    rt, fn, _ = round_runtime
    monkeypatch.setattr(rt.dataset, "evaluation_fingerprint", None)
    assert tools_eval.final_evaluate_functions()["censused"] == [fn.name]
    assert fn.scores["evaluation_context"] is None
    assert tools_eval.final_evaluate_functions()["censused"] == [fn.name]


def test_missing_point_reads_do_not_certify_a_complete_checkpoint(round_runtime, monkeypatch):
    rt, fn, _ = round_runtime
    original = tools_eval._fetch_items_by_id
    frame = {row["_item_id"]: "bench|pass" if row["is_correct"] else "bench|fail"
             for row in rt.dataset._fake_items}
    omit = proportional_sample(frame, fraction=.3, seed=tools_eval._evaluation_seed(rt.run),
                               minimum_n=20).item_ids[0]

    def incomplete(dataset, ids, **kwargs):
        yield from (row for row in original(dataset, ids, **kwargs)
                    if row["_item_id"] != omit)

    monkeypatch.setattr(tools_eval, "_fetch_items_by_id", incomplete)
    assert tools_eval.evaluate_round_functions()["ok"]
    assert fn.scores["round_validation"]["complete"] is False
    assert fn.scores["sample_n"] < fn.scores["round_validation"]["sample_n"]


def test_capped_requests_keep_larger_scan_errors_without_hidden_rescans(round_runtime, monkeypatch):
    _, fn, _ = round_runtime
    attempts = []
    original = tools_eval.run_detector

    def broken(source, item, **kwargs):
        attempts.append(item["_item_id"])
        if int(item["problem_id"]) % 3 == 0:
            raise ValueError("Malformed evidence")
        return original(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", broken)
    assert tools_eval.evaluate_round_functions()["ok"]
    before = deepcopy(fn.scores)
    spent = len(attempts)
    assert before["errors"] > 0
    retained = tools_eval.evaluate_functions()
    assert retained["reused_names"] == [fn.name]
    assert fn.scores == before and len(attempts) == spent
    assert tools_eval.evaluate_round_functions()["evaluated_names"] == [fn.name]
    assert len(attempts) > spent
    assert fn.scores["evaluation_item_ids"] == before["evaluation_item_ids"]


def test_reusing_validated_scores_does_not_hide_other_screen_blocks(round_runtime, monkeypatch):
    rt, fn, calls = round_runtime
    assert tools_eval.evaluate_round_functions()["ok"]
    spent = len(calls)
    blocked = FeatureFunction(name="blocked", description="Awaiting review",
                              source=_LLM_SOURCE, kind="llm")
    rt.library.functions.append(blocked)
    monkeypatch.setattr("autofa.session.candidate_screen.ensure_candidate_screen",
                        lambda rt, function, **kwargs: {
                            "status": "not_required" if function.name == fn.name else "pending",
                            "evaluation_allowed": function.name == fn.name,
                        })
    result = tools_eval.evaluate_functions()
    assert result["ok"] is False and result["candidate_screen_blocked"]
    assert result["blocked_functions"] == [blocked.name]
    assert result["reused_names"] == [fn.name] and result["evaluated_names"] == []
    assert len(calls) == spent and not blocked.scores


def test_llm_parent_checkpoint_is_reused_until_effective_parent_changes(round_runtime):
    rt, parent, calls = round_runtime
    child = FeatureFunction(name="child", description="within semantic parent",
                            source=_REGEX_SOURCE, explains=parent.name)
    rt.library.functions.append(child)
    result = tools_eval.evaluate_round_functions(names=[child.name], fraction=.8)
    assert result["evaluated_names"] == [child.name]
    assert child.scores["sample_n"] == 80
    assert child.scores["round_validation"]["fraction"] == .8
    assert child.scores["evaluation_parent_identity"]
    checkpoint = deepcopy(child.scores)
    spent = len(calls)
    for fraction in (.8, .3):
        result = tools_eval.evaluate_round_functions(names=[child.name], fraction=fraction)
        assert result["reused_names"] == [child.name]
        assert child.scores == checkpoint and len(calls) == spent
    assert tools_eval.evaluate_functions(names=[child.name])["reused_names"] == [child.name]
    parent.source += "\n# changed parent interpretation"
    result = tools_eval.evaluate_round_functions(names=[child.name], fraction=.8)
    assert result["evaluated_names"] == [child.name]
    assert child.scores["evaluation_parent_identity"] != checkpoint["evaluation_parent_identity"]
    assert len(calls) > spent


def test_resolved_transport_changes_invalidate_scope_and_prediction_reuse(
    round_runtime, monkeypatch,
):
    rt, fn, calls = round_runtime

    class RoutedAdapter:
        available = True

        def __init__(self):
            self.decisions = None

        def routing_identity(self):
            return {"transport_contract_version": 2, "api": "auto",
                    "decisions_available": self.decisions}

    adapter = RoutedAdapter()
    original = tools_eval.run_detector

    def resolving(source, item, **kwargs):
        if adapter.decisions is None:
            adapter.decisions = False
        return original(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: adapter)
    monkeypatch.setattr(tools_eval, "run_detector", resolving)
    assert tools_eval.evaluate_round_functions(fraction=.8)["ok"]
    assert fn.scores["evaluation_context"] == tools_eval._evaluation_context(rt)
    with monkeypatch.context() as offline:
        offline.setattr(tools_eval, "_llm_adapter",
                        lambda: pytest.fail("An explicit route must not construct a client"))
        assert tools_eval._evaluation_context(
            rt, routing_identity=adapter.routing_identity(),
        ) == fn.scores["evaluation_context"]
    spent = len(calls)
    assert tools_eval.evaluate_round_functions(fraction=.8)["reused_names"] == [fn.name]
    assert len(calls) == spent
    # Both aggregate authority and item prediction keys follow the resolved API.
    adapter.decisions = True
    assert tools_eval.evaluate_round_functions(fraction=.8)["evaluated_names"] == [fn.name]
    assert len(calls) == spent * 2
    assert tools_eval.evaluate_round_functions(fraction=.8)["reused_names"] == [fn.name]
    assert len(calls) == spent * 2
