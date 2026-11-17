# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outcome rules and evaluator evidence stay separate from candidate traces."""

from __future__ import annotations

import json
from typing import Any

from autofa.data.outcomes import (
    discovery_is_correct,
    efficiency_outcome,
    is_discovery_success,
    is_failure_reward,
    is_healthy_scored,
    is_scored_failure,
    population_outcome,
)
from autofa.data.remote_api import DataApiClient
from autofa.data.source_errors import SourceDataError
from autofa.taxonomy.outcome_evidence import (
    MAX_OUTCOME_EVIDENCE_CHARACTERS,
    load_outcome_evidence_record,
    project_outcome_evidence,
)

HEALTHY = {"status": "ok", "ingest_status": "ok", "error": None, "is_scored": True}


def _output(text: str) -> dict[str, Any]:
    return {"output": [{"content": [{"text": text}]}]}


def _items(evidence: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    return [item for item in evidence["items"] if item["id"].startswith(prefix)]


# --- Workflow outcome rules ---

def test_workflow_outcome_rules_remain_explicit() -> None:
    row = {**HEALTHY, "reward": 0.5, "accuracy": 0.5}
    assert not is_failure_reward(row["reward"])
    assert is_healthy_scored(row)
    assert not is_scored_failure(row)
    assert discovery_is_correct(row)
    assert is_discovery_success(row)
    assert efficiency_outcome(row["reward"]) == "fail"
    assert population_outcome(row["reward"]) == "pass"
    # Without an accuracy field, discovery correctness falls back to the canonical reward.
    assert not is_discovery_success({**HEALTHY, "reward": 0.49})
    assert is_discovery_success({**HEALTHY, "reward": 0.5})


def test_invalid_rewards_are_unscored() -> None:
    for value in (None, True, "bad", float("nan")):
        assert efficiency_outcome(value) == "unscored"
        assert population_outcome(value) == "unscored"
        assert not is_failure_reward(value)
        row = {**HEALTHY, "reward": value}
        assert not is_healthy_scored(row)
        assert not is_discovery_success(row)


# --- Evaluator outcome evidence projection ---

def test_comparison_trials_are_oriented_to_the_candidate() -> None:
    evidence, receipt = project_outcome_evidence({
        "reward": 0.0,
        "rubric_pretty": "Prefer clinically safe and complete work.",
        "judge_response": {"per_ref_repeat": [{
            "ref_id": "reference", "trial_judges": ["j1", "j2", "j3"],
            "raw_responses": ["Reference is safer. BOXED[A]",
                              "The candidate is more complete. BOXED[A]",
                              "They are equivalent. BOXED[TIE]"],
        }]},
        "baseline_answer": "Reference response",
        "policy_answer": "Candidate response",
        "games": [
            {"verdict": "[[BB]]",
             "response": _output("Both answers contain language errors. [[BB]]")},
            {"verdict": "[[A>>B]]", "response": _output("The reference is better. [[A>>B]]")},
        ],
    }, source_ref="artifacts/evaluator_rollouts.jsonl:10")
    trials = _items(evidence, "comparison")
    assert evidence["semantics"] == "comparative"
    assert [item["value"]["candidate_label"] for item in trials] == ["B", "A", "B"]
    assert [item["value"]["candidate_outcome"] for item in trials] == ["loss", "win", "tie"]
    games = _items(evidence, "pairwise-game")
    assert [item["value"]["candidate_label"] for item in games] == ["A", "B"]
    assert [item["value"]["candidate_outcome"] for item in games] == ["both_bad", "loss"]
    # The judge's rationale is parsed out of the Responses-style game output.
    assert "language errors" in games[0]["text"]
    assert receipt["status"] == "included"
    assert receipt["items_omitted"] == 0


def test_pairwise_judge_prompt_is_typed_as_evaluator_rubric() -> None:
    system = {"role": "system",
              "content": "Compare both answers for correctness and helpfulness."}
    user = {"role": "user",
            "content": "Candidate and reference answers belong to the evaluator."}
    evidence, _receipt = project_outcome_evidence({
        "baseline_answer": "Reference",
        "policy_answer": "Candidate",
        "games": [
            {"responses_create_params": {"input": [system, user]}, "verdict": "[[B>>A]]",
             "response": _output("B is more accurate. [[B>>A]]")},
            {"responses_create_params": {"input": [system]}, "verdict": "[[A>>B]]",
             "response": _output("A is more accurate. [[A>>B]]")},
        ],
    }, source_ref="raw.jsonl:30")

    rubrics = [item for item in evidence["items"] if item["kind"] == "evaluator_rubric"]
    assert len(rubrics) == 1
    assert rubrics[0]["subject"] == "evaluator"
    assert rubrics[0]["source_path"] == "games[0].responses_create_params.input[0]"
    assert "correctness and helpfulness" in rubrics[0]["text"]
    assert "Candidate and reference answers belong" not in json.dumps(evidence)


def test_reference_and_metric_fields_are_projected_without_candidate_response() -> None:
    evidence, receipt = project_outcome_evidence({
        "reward": 0.0, "answer": "F", "candidates": ["A", "B", "C", "D", "E", "F"], "score": 0,
        "response": {"output": "candidate response stays in the trace",
                     "incomplete_details": {"reason": "max_output_tokens"}},
        "benchmark/component": 0.1,
    }, source_ref="raw.jsonl:20")
    assert evidence["semantics"] == "reference"
    assert {item["source_path"] for item in evidence["items"]} >= {
        "answer", "candidates", "benchmark/component", "response",
    }
    assert "candidate response stays in the trace" not in json.dumps(evidence)
    assert "response" in receipt["fields_found"]

    unknown, unknown_receipt = project_outcome_evidence({
        "answer": "candidate-owned field",
        "games": [{"verdict": "[[A>>B]]"}, {"verdict": "[[B>>A]]"}],
    }, source_ref="unknown.jsonl:1")
    assert unknown["semantics"] == "unknown"
    assert unknown["items"] == []
    assert "answer" not in unknown_receipt["fields_found"]
    assert "games" not in unknown_receipt["fields_found"]

    beside_response, _ = project_outcome_evidence(
        {"answer": "38", "response": {"status": "completed"}, "reward": 0.0},
        source_ref="reference.jsonl:1",
    )
    assert beside_response["semantics"] == "reference"
    assert next(
        item["value"] for item in beside_response["items"] if item["source_path"] == "answer"
    ) == "38"

    metric, _ = project_outcome_evidence(
        {"reward": 0.0, "step_results": [True, False], "num_steps_passed": 1},
        source_ref="scicode.jsonl:1",
    )
    assert metric["semantics"] == "metric"
    assert metric["observed_outcome"]["step_results"] == [True, False]


def test_tau2_reward_info_projects_action_and_database_failures() -> None:
    evidence, receipt = project_outcome_evidence({
        "reward": 0.0,
        "result": {"reward_info": {
            "reward": 0.0, "reward_basis": ["DB"], "reward_breakdown": {"DB": 0.0},
            "db_check": {"db_match": False, "db_reward": 0.0},
            "action_checks": [{
                "action": {"action_id": "002_0", "requestor": "user",
                           "name": "apply_for_credit_card",
                           "arguments": {"card_type": "Platinum Rewards Card"}},
                "action_match": False, "action_reward": 0.0, "tool_type": "write",
            }],
        }},
    }, source_ref="artifacts/evaluator_rollouts.jsonl:42")

    assert evidence["semantics"] == "judge"
    by_id = {item["id"]: item for item in evidence["items"]}
    assert by_id["tau2-reward-summary"]["value"]["db_check"]["db_match"] is False
    action = by_id["tau2-action-check-0"]["value"]
    assert action["action"]["name"] == "apply_for_credit_card"
    assert action["action_match"] is False
    assert "result.reward_info" in receipt["fields_found"]


def test_execution_outcome_projects_bounded_test_evidence() -> None:
    evidence, receipt = project_outcome_evidence({
        "reward": 0.0, "resolved": False, "evaluation_completed": True,
        "failure_kind": "tests_failed",
        # Failure lines in the middle of a long log survive the excerpt.
        "test_output": "\n".join(["setup" * 500, "test_widget FAILED",
                                  "AssertionError: expected 3", "teardown" * 500]),
        "test_results": {"passed": 2, "failed": 3},
        "golden_patch_output": "REFERENCE SOLUTION MUST NOT BE EXPOSED",
    }, source_ref="results/evaluator_rollouts.jsonl:42")

    assert evidence["semantics"] == "judge"
    assert evidence["observed_outcome"] == {"reward": 0.0}
    by_id = {item["id"]: item for item in evidence["items"]}
    assert by_id["execution-summary"]["value"]["failure_kind"] == "tests_failed"
    assert by_id["test-results"]["value"] == {"passed": 2, "failed": 3}
    assert "expected 3" in by_id["test-output"]["text"]
    assert by_id["test-output"]["value"]["failure_excerpt"] is True
    assert "REFERENCE SOLUTION" not in json.dumps(evidence)
    assert {"execution", "test_output", "test_results"} <= set(receipt["fields_found"])


def test_generic_evaluator_metadata_projects_output_mismatch() -> None:
    evidence, receipt = project_outcome_evidence({
        "reward": 0.0,
        "metadata": {
            "inputs": "large task input must not be copied",
            "output": "0 0 23 22", "expected": "22 23 0 0", "error_code": -2,
            "error_message": "Wrong answer at output_line_idx=0: 0 0 23 22 != 22 23 0 0",
        },
        "result": [True, True, True, True, -2],
    }, source_ref="results/evaluator_rollouts.jsonl:7")

    assert evidence["semantics"] == "judge"
    item = next(item for item in evidence["items"] if item["id"] == "evaluator-metadata")
    assert item["value"]["output"] == "0 0 23 22"
    assert item["value"]["expected"] == "22 23 0 0"
    assert item["value"]["error_code"] == -2
    assert item["value"]["result"][-1] == -2
    assert "inputs" not in item["value"]
    assert "metadata" in receipt["fields_found"]


def test_candidate_patch_is_bounded_without_exposing_golden_patch() -> None:
    evidence, receipt = project_outcome_evidence({
        "reward": 0.0,
        "model_patch": "diff --git a/a.py b/a.py\n" + ("+candidate change\n" * 300),
        "golden_patch": "SECRET REFERENCE PATCH",
        "reference_patch": "ANOTHER SECRET PATCH",
    }, source_ref="results/evaluator_rollouts.jsonl:9")

    item = next(item for item in evidence["items"] if item["id"] == "candidate-patch")
    assert item["kind"] == "candidate_patch"
    assert item["text"]["truncated"] is True
    assert item["value"]["full_characters"] > 1_200
    assert "SECRET REFERENCE" not in json.dumps(evidence)
    assert "model_patch" in receipt["fields_found"]


def test_four_comparison_rationales_fit_the_outcome_budget() -> None:
    evidence, receipt = project_outcome_evidence({
        "rubric_pretty": "rubric " * 600,
        "judge_response": {"per_ref_repeat": [{
            "ref_id": "reference", "trial_judges": ["j1", "j2", "j3", "j4"],
            "raw_responses": [("finding " * 400) + f"BOXED[{verdict}]"
                              for verdict in ("A", "B", "A", "B")],
        }]},
    }, source_ref="raw.jsonl:40")
    trials = _items(evidence, "comparison")
    assert len(trials) == 4
    assert all(item["text"]["truncated"] for item in trials)
    assert receipt["items_omitted"] == 0
    assert receipt["truncated"] is True
    assert (len(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")))
            <= MAX_OUTCOME_EVIDENCE_CHARACTERS)


def test_comparison_trials_without_judge_names_are_projected() -> None:
    evidence, receipt = project_outcome_evidence({
        "reward": 0.0,
        "judge_response": {"per_ref_repeat": [{
            "ref_repeat": "task-example",
            "raw_responses": ["Reference A is better. BOXED[A]",
                              "Reference B is better. BOXED[B]"],
        }]},
    }, source_ref="artifacts/evaluator_rollouts.jsonl:99")
    trials = _items(evidence, "comparison")
    assert evidence["semantics"] == "comparative"
    assert [item["value"]["candidate_outcome"] for item in trials] == ["loss"] * 2
    assert {item["value"]["reference_id"] for item in trials} == {"task-example"}
    assert all(item["value"]["judge"] is None for item in trials)
    assert "judge_response" in receipt["fields_found"]


def test_evaluator_results_take_priority_over_large_references() -> None:
    evidence, receipt = project_outcome_evidence({
        "expected_answer": "reference " * 2_000,
        "choices": ["choice " * 1_000 for _ in range(4)],
        "rubric": "rubric " * 1_000,
        "test_results": {"failed": ["test_decisive"]},
        "test_output": "test_decisive FAILED\nAssertionError: expected 3, actual 4",
    }, source_ref="results.jsonl:1")

    item_ids = [item["id"] for item in evidence["items"]]
    assert "test-results" in item_ids
    assert "test-output" in item_ids
    assert item_ids.index("test-results") < item_ids.index("reference-expected_answer")
    assert receipt["truncated"] is True


def test_raw_record_fetch_uses_source_offset_and_is_optional(monkeypatch) -> None:
    client = DataApiClient("https://example.test")
    raw = json.dumps({"reward": 0, "expected_answer": "A"}) + "\n"
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def fake_get_json(path: str, query: dict[str, Any] | None = None) -> Any:
        calls.append((path, query))
        return {"text": raw, "length": len(raw.encode()), "eof": True}

    monkeypatch.setattr(client, "_get_json", fake_get_json)
    evidence, receipt, _ = load_outcome_evidence_record(
        client, {"run_id": "run", "raw_record_ref": "records.jsonl:42"}, []
    )
    assert evidence["semantics"] == "reference"
    assert receipt["source_ref"] == "records.jsonl:42"
    assert calls[0][1]["offset"] == 42

    evidence, receipt, _ = load_outcome_evidence_record(
        client, {"run_id": "run"}, [{"name": "answer.pdf", "role": "output"}]
    )
    assert receipt["status"] == "absent"
    assert [item["id"] for item in evidence["items"]] == ["artifact-inventory"]


def test_unavailable_raw_record_uses_only_allowlisted_artifact_fallback() -> None:
    class MissingSource:
        def read_raw_record(self, run_id, source_ref):
            raise SourceDataError("source logs were removed")

    artifacts = [
        {
            "name": "tests.log",
            "kind": "log",
            "role": "verifier",
            "text": "FAILED test_decisive",
        },
        {
            "name": "model.patch",
            "kind": "patch",
            "role": "output",
            "text": "diff --git a/a.py b/a.py\n+candidate",
        },
        {
            "name": "other.patch",
            "kind": "patch",
            "role": "output",
            "text": "ARBITRARY OUTPUT PATCH",
        },
        {
            "name": "golden.patch",
            "kind": "patch",
            "role": "verifier",
            "text": "SECRET REFERENCE PATCH",
        },
    ]
    evidence, receipt, record = load_outcome_evidence_record(
        MissingSource(),
        {"run_id": "run", "raw_record_ref": "missing.jsonl:0"},
        artifacts,
    )

    assert record is None and receipt["status"] == "artifact_fallback"
    patch = next(item for item in evidence["items"] if item["id"] == "candidate-patch")
    assert patch["text"] == "diff --git a/a.py b/a.py\n+candidate"
    serialized = json.dumps(evidence)
    assert "ARBITRARY OUTPUT PATCH" not in serialized
    assert "SECRET REFERENCE PATCH" not in serialized


def test_hydrated_candidate_patch_is_bounded_with_the_full_size_recorded() -> None:
    class MissingSource:
        def read_raw_record(self, run_id, source_ref):
            raise SourceDataError("source logs were removed")

    text = "diff --git a/a.py b/a.py\n" + ("+candidate change\n" * 1_000)
    evidence, receipt, _record = load_outcome_evidence_record(
        MissingSource(),
        {"run_id": "run", "raw_record_ref": "missing.jsonl:0"},
        [{
            "name": "model.patch",
            "kind": "patch",
            "role": "output",
            "text": text,
        }],
    )

    patch = next(item for item in evidence["items"] if item["id"] == "candidate-patch")
    assert patch["text"]["truncated"] is True
    assert patch["text"]["characters"] == len(text)
    assert patch["value"]["full_characters"] == len(text)
    assert receipt["truncated"] is True
    assert len(json.dumps(evidence, ensure_ascii=False)) < MAX_OUTCOME_EVIDENCE_CHARACTERS


def test_hydrated_candidate_patch_complements_a_readable_raw_record() -> None:
    class Source:
        def read_raw_record(self, run_id, source_ref):
            return {"reward": 0, "test_output": "FAILED test_decisive"}

    evidence, receipt, record = load_outcome_evidence_record(
        Source(),
        {"run_id": "run", "raw_record_ref": "results.jsonl:0"},
        [{
            "name": "model.patch",
            "kind": "patch",
            "role": "output",
            "text": "diff --git a/a.py b/a.py\n+candidate",
        }],
    )

    assert record is not None and receipt["status"] == "included"
    patch = next(item for item in evidence["items"] if item["id"] == "candidate-patch")
    assert patch["text"].endswith("+candidate")
    assert patch["value"]["artifact_fallback"] is False
