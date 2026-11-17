# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy pipeline: rollout preparation, evidence quality, trace digests, windows and judging."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from autofa.data.evidence_refs import compact_trace
from autofa.taxonomy.classifier import TaxonomyClassifier
from autofa.taxonomy.models import ProviderResponse
from autofa.taxonomy.pipeline import (
    _exclude_auxiliary_evaluator_prompts,
    classify_prepared,
    evidence_quality_report,
    prepare_rollout,
)
from autofa.taxonomy.schema import Taxonomy
from autofa.taxonomy.trace_digest import build_trace_digest, build_trace_judgement_context
from autofa.taxonomy.trace_windows import build_trace_judgement_windows

from support.taxonomy import _diagnosis_answer


def _taxonomy() -> Taxonomy:
    return Taxonomy.from_dict({
        "id": "example.custom", "name": "example", "version": "0",
        "root": {"id": "failure", "label": "Failure", "description": "Failure", "children": [
            {"id": "data", "label": "Data", "description": "Data marker"},
            {"id": "model", "label": "Model", "description": "Model marker"},
        ]},
    })


def _incident(
    trace: list[dict[str, Any]],
    *,
    outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"evidence": [{"kind": "bounded_ordered_trace_digest", "observation": {
        "ordered_trace": trace,
        "outcome_evidence": outcome
        or {"semantics": "judge", "observed_outcome": {"reward": 0}, "items": [{}]},
        "coverage": {"verifier_artifact_count": 0},
    }}]}


def test_evidence_quality_requires_marked_setup_and_warns_on_reward_only() -> None:
    unmarked = evidence_quality_report(_incident([
        {"event_ref": {"event_idx": 0, "role": "assistant"}},
        {"event_ref": {"event_idx": 1, "role": "user"}},
    ]))
    assert unmarked["status"] == "fail"
    assert unmarked["issues"][0]["code"] == "pre_task_model_attribution"

    trace = [
        {"event_ref": {"event_idx": 0, "role": "assistant", "phase": "pre_task_context"}},
        {"event_ref": {"event_idx": 1, "role": "user"}},
    ]
    assert evidence_quality_report(_incident(trace))["status"] == "pass"

    warning = evidence_quality_report(_incident(
        trace, outcome={"semantics": "unknown", "observed_outcome": {"reward": 0}, "items": []},
    ))
    assert warning["status"] == "warn"
    assert warning["issues"][0]["code"] == "outcome_cause_unavailable"


def _judge_game(*prompts: tuple[str, str]) -> dict[str, Any]:
    return {"responses_create_params": {
        "input": [{"role": role, "content": content} for role, content in prompts],
    }}


def test_nested_pairwise_judge_prompt_is_excluded_from_candidate_trace() -> None:
    judge_prompt = "Compare assistant A and assistant B, then output a verdict."
    shared_prompt = "Act as a judge."
    events = [
        {"event_idx": 0, "source_event_id": "reminder:0:0", "role": "system",
         "parts": [{"text": shared_prompt}]},
        {"event_idx": 1, "source_event_id": "reminder:1:0", "role": "system",
         "parts": [{"text": judge_prompt}]},
        {"event_idx": 2, "source_event_id": "rs_candidate", "role": "output",
         "parts": [{"text": "I should explain registers."}]},
    ]
    # The candidate's own system prompt stays even when the nested judge game repeats it.
    raw_record = {**_judge_game(("system", shared_prompt)),
                  "games": [_judge_game(("system", shared_prompt), ("system", judge_prompt))]}

    filtered, excluded = _exclude_auxiliary_evaluator_prompts(events, raw_record)

    assert excluded == 1
    assert [event["event_idx"] for event in filtered] == [0, 2]


class _FirstChoice:
    def evaluate(self, state: Any, questions: dict[str, dict[str, Any]]) -> ProviderResponse:
        answers = {}
        for question_id, question in questions.items():
            if question["type"] != "choice":
                continue
            keys = list(question["criteria"])
            answers[question_id] = {
                "type": "choice",
                "choice": keys[0],
                "alternatives": keys[1:3],
                "rationale": "scripted",
            }
        if "diagnosis" in questions:
            answers["diagnosis"] = _diagnosis_answer()
        return ProviderResponse("scripted", answers, {})


class _Source:
    def __init__(
        self,
        artifacts: list[dict[str, Any]] | None = None,
        raw_record: dict[str, Any] | None = None,
    ) -> None:
        self._artifacts = artifacts or []
        self.raw_record = raw_record

    def artifacts(self, task_id: str, repeat: int) -> list[dict[str, Any]]:
        return [dict(artifact) for artifact in self._artifacts]

    def read_raw_record(self, run_id: str, source_ref: str) -> dict[str, Any]:
        assert self.raw_record is not None
        return self.raw_record


def _item(events: list[dict[str, Any]], *, raw_record_ref: str | None = None) -> dict[str, Any]:
    execution = {
        "rollout": {"n_events": len(events), "n_tool_calls": 0, "raw_record_ref": raw_record_ref},
        "events": events,
        "tool_calls": [],
        "availability": {
            "events": {"status": "available"},
            "tool_calls": {"status": "available"},
        },
    }
    return {
        "problem_id": "task",
        "run_idx": 0,
        "reward": 0.0,
        "lane": "",
        "response_text": json.dumps({"execution": execution}),
    }


def _user(text: str) -> list[dict[str, Any]]:
    return [{"event_idx": 0, "lane": "", "role": "user", "parts": [{"text": text}]}]


def _classify(item: dict[str, Any], source: _Source | None = None, **kwargs: Any) -> dict:
    analyzer = TaxonomyClassifier(_taxonomy(), _FirstChoice())
    prepared = prepare_rollout(item, source or _Source(), run_id="run", **kwargs)
    return classify_prepared(analyzer, prepared)


def test_classify_records_taxonomy_and_evidence_receipt() -> None:
    source = _Source(
        artifacts=[{"name": "answer.pdf", "kind": "document", "role": "output"}],
        raw_record={"reward": 0, "expected_answer": "A"},
    )
    row = _classify(_item(_user("do the task"), raw_record_ref="raw.jsonl:10"), source)

    assert row["status"] == "ok"
    assert row["rollout_id"] == "run:task:0"
    assert row["evidence_receipt"]["trace"]["event_count"] == 1
    assert row["evidence_receipt"]["raw_outcome"]["status"] == "included"
    assert row["evidence_receipt"]["artifacts"]["roles"] == ["output"]
    assert row["digest_budget"]["requires_windowing"] is False
    assert row["analysis"]["taxonomy_id"] == "example.custom"
    assert row["analysis"]["taxonomy_version"] == "0"
    assert row["analysis"]["selected_leaf_id"] == "data"


def test_prepare_keeps_only_the_rows_conversation_lane() -> None:
    events = [
        {"event_idx": 0, "lane": "conversation", "role": "user", "parts": [{"text": "task"}]},
        {"event_idx": 0, "lane": "grader", "role": "user", "parts": [{"text": "grade"}]},
    ]
    item = _item(events)
    item["lane"] = "conversation"
    prepared = prepare_rollout(item, _Source(), run_id="run")

    trace = prepared["incident"]["evidence"][0]["observation"]["ordered_trace"]
    assert [event["event_ref"]["lane"] for event in trace] == ["conversation"]
    assert prepared["lane"] == "conversation"
    row = classify_prepared(TaxonomyClassifier(_taxonomy(), _FirstChoice()), prepared)
    assert row["lane"] == "conversation"
    # Without a raw record reference the outcome is explicitly absent.
    assert row["evidence_receipt"]["raw_outcome"]["status"] == "absent"


def test_prepare_projects_non_evaluation_auxiliary_lanes_and_resources() -> None:
    events = [
        {"event_idx": 0, "lane": "conversation", "role": "user",
         "parts": [{"part_idx": 0, "text": "task"}]},
        {"event_idx": 0, "lane": "worker", "role": "assistant",
         "parts": [{"part_idx": 0, "text": "auxiliary evidence"}]},
        {"event_idx": 0, "lane": "grader", "role": "user", "evaluation": True,
         "parts": [{"part_idx": 0, "text": "hidden evaluation"}]},
    ]
    item = _item(events)
    item["lane"] = "conversation"
    envelope = json.loads(item["response_text"])
    envelope["execution"]["attempts"] = [{"attempt_idx": 0, "error": "retry"}]
    envelope["execution"]["prompt_messages"] = [
        {"msg_hash": "safe", "msg_json": "visible prompt"}
    ]
    item["response_text"] = json.dumps(envelope)

    prepared = prepare_rollout(item, _Source(), run_id="run")
    observation = prepared["incident"]["evidence"][0]["observation"]

    assert [event["event_ref"]["lane"] for event in observation["ordered_trace"]] == [
        "conversation"
    ]
    assert [
        (lane["lane"], lane["ordered_trace"][0]["event_ref"]["event_idx"])
        for lane in observation["auxiliary_lanes"]
    ] == [("worker", 0)]
    assert observation["trace_resources"]["attempts"][0]["value"]["error"] == "retry"
    assert (
        observation["trace_resources"]["prompt_messages"][0]["value"]["msg_json"]
        == "visible prompt"
    )
    assert "hidden evaluation" not in json.dumps(observation)


def test_prepare_retains_trusted_artifact_identity_and_normalized_trace() -> None:
    item = _item(_user("task"))
    envelope = json.loads(item["response_text"])
    execution = envelope["execution"]
    execution["source"] = {"kind": "trace_api", "base_url": "https://data.example"}
    execution["artifacts"] = [{"artifact_idx": 2, "name": "failure.pdf", "role": "verifier"}]
    execution["events"].append(
        {"event_idx": 4, "lane": "grader", "role": "evaluator", "evaluation": True},
    )
    envelope["steps"] = [
        {"step_id": 0, "source": "user", "message": "task",
         "metadata": {"event_idx": 0, "lane": "", "role": "user"}},
        {"step_id": 4, "source": "user", "message": "hidden grader prompt",
         "metadata": {"event_idx": 4, "lane": "grader"}},
    ]
    envelope["evidence_scope"] = {"source": "original API event parts and tool arguments",
                                  "event_lane": ""}
    compact = compact_trace(envelope)
    assert "_autofa_evidence_refs" in compact
    item["response_text"] = json.dumps(compact)

    prepared = prepare_rollout(item, _Source(), run_id="run")
    retained = prepared["artifact_source"]["trace"]["execution"]

    assert retained["rollout"]["run_id"] == "run"
    assert retained["source"]["kind"] == "trace_api"
    assert retained["artifacts"][0]["artifact_idx"] == 2
    assert [event["event_idx"] for event in retained["events"]] == [0]
    steps = prepared["artifact_source"]["trace"]["steps"]
    assert [step["message"] for step in steps] == ["task"]


def test_prepare_reports_missing_execution_trace() -> None:
    item = {"problem_id": "task", "run_idx": 0, "reward": 0.0, "response_text": "plain text"}
    prepared = prepare_rollout(item, _Source(), run_id="run")

    assert prepared["status"] == "error"
    assert "full traces" in prepared["error"]


def test_classify_stops_when_compaction_cannot_fit_the_budget() -> None:
    row = _classify(_item(_user("task " * 5000)), max_serialized_characters=1)

    assert row["status"] == "needs_windowing"
    assert "analysis" not in row
    assert row["digest_budget"]["requires_windowing"] is True


def test_prepare_compacts_when_combined_instruction_anchors_cannot_fit() -> None:
    roles = ("system", "developer", "user", "assistant")
    events = [{"event_idx": index, "lane": "", "role": role,
               "parts": [{"text": "terminal" if role == "assistant" else role[0] * 1_200}]}
              for index, role in enumerate(roles)]

    prepared = prepare_rollout(_item(events), _Source(), run_id="run",
                               max_serialized_characters=4_000)

    assert prepared["status"] == "prepared"
    assert prepared["digest_budget"]["compaction_level"] > 0
    assert prepared["digest_budget"]["fallback_compaction"] is True


def test_classify_windowable_trace_in_ordered_windows() -> None:
    events = [{"event_idx": index, "lane": "", "role": "user" if index == 0 else "assistant",
               "parts": [{"text": f"event-{index} " * 180}]} for index in range(10)]
    row = _classify(_item(events), max_serialized_characters=8_000)

    assert row["status"] == "ok"
    assert row["digest_budget"]["requires_windowing"] is True
    assert row["digest_budget"]["compaction_level"] == 0
    assert row["digest_budget"]["fallback_compaction"] is False
    assert row["analysis"]["windowed"] is True
    assert row["analysis"]["window_count"] > 1
    assert row["analysis"]["aggregation"] == "rolling_failure_ledger"


def test_prepare_chunks_oversized_event_before_compaction() -> None:
    events = _user("task") + [
        {"event_idx": index, "lane": "", "role": "assistant",
         "parts": [{"text": f"DECISIVE_END_{index} " + ("x" * 5_000)}]}
        for index in range(1, 41)
    ]
    events.insert(20, {"event_idx": 50, "lane": "", "role": "assistant",
                       "parts": [{"text": "OVERSIZED_END " + ("y" * 50_000)}]})

    prepared = prepare_rollout(_item(events), _Source(), run_id="run",
                               max_serialized_characters=32_000)

    assert prepared["status"] == "prepared"
    assert prepared["digest_budget"]["fallback_compaction"] is False
    assert prepared["digest_budget"]["compaction_level"] == 0
    assert prepared["digest_budget"]["requires_windowing"] is True
    assert len(prepared["windows"]) > 1
    visible = json.dumps(prepared["windows"], ensure_ascii=False)
    assert "DECISIVE_END_1" in visible
    assert "OVERSIZED_END" in visible


def test_classification_distinguishes_preparation_and_judge_failures() -> None:
    before_judge = classify_prepared(
        TaxonomyClassifier(_taxonomy(), _FirstChoice()),
        {"status": "error", "rollout_id": "run:task:0", "run_id": "run", "task_id": "task",
         "repeat": 0, "error": "preparation failed"},
    )

    class _FailingJudge:
        def evaluate(self, state: Any, questions: dict[str, Any]) -> ProviderResponse:
            raise RuntimeError("judge failed")

    prepared = prepare_rollout(_item(_user("task")), _Source(), run_id="run")
    during_judge = classify_prepared(TaxonomyClassifier(_taxonomy(), _FailingJudge()), prepared)

    class _FailingAgenticReview:
        def review(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("agentic review failed")

    during_review = classify_prepared(
        TaxonomyClassifier(_taxonomy(), _FirstChoice(), second_pass="agentic",
                           agentic_provider=_FailingAgenticReview()),
        prepared,
    )

    assert before_judge["judge_attempted"] is False
    assert during_judge["judge_attempted"] is True
    assert during_judge["error"] == "judge failed"
    assert during_review["status"] == "review_failed"
    assert during_review["analysis"]["second_pass"]["status"] == "failed"
    assert during_review["error"] == "agentic review failed"


# Trace digests and judgement windows keep ordered evidence within character budgets.

# --- digest ---


def raw_event(event_idx, role, text, **extra):
    return {"event_idx": event_idx, "role": role, "parts": [{"text": text}], **extra}


def tool_call(event_idx, tool_name, status, args, *, result=None, call_idx=0, **extra):
    call = {"event_idx": event_idx, "call_idx": call_idx, "tool_name": tool_name,
            "status": status, "args_full": args, **extra}
    if result is not None:
        call["result_event_idx"] = result
    return call


def digest_of(events, calls=(), **kwargs):
    return build_trace_digest(rollout={"n_events": len(events), "n_tool_calls": len(calls)},
                              tool_calls=calls, events=events, **kwargs)


def event_indexes(context):
    return [entry["event_ref"]["event_idx"] for entry in context["ordered_trace"]]


def test_under_budget_digest_preserves_every_value_in_full():
    full_error = "Error: expected integer, got string"
    digest = digest_of(
        [raw_event(0, "user", "do the task"), raw_event(1, "assistant", "try tool"),
         raw_event(2, "tool", full_error), raw_event(3, "assistant", "recover now"),
         raw_event(4, "tool", "success output")],
        [tool_call(1, "bad_tool", "error", {"amount": "2"}, result=2),
         tool_call(3, "good_tool", "ok", {"verbose": "a" * 20}, result=4, call_idx=1)],
    )
    assert event_indexes(digest) == [0, 1, 2, 3, 4]
    failed = digest["ordered_trace"][1]["tool_calls"][0]
    succeeded = digest["ordered_trace"][3]["tool_calls"][0]
    assert failed["args"]["full"] == '{"amount":"2"}'
    assert failed["result"]["full"] == full_error
    assert succeeded["args"]["full"] == '{"verbose":"' + ("a" * 20) + '"}'
    assert succeeded["result"]["full"] == "success output"
    assert digest["budget"]["compaction_level"] == 0
    assert digest["ordered_trace"][2]["text"]["omitted"] == "duplicate of the tool-call result"
    assert digest["ordered_trace"][1]["text"]["full"] == "try tool"
    assert digest["ordered_trace"][3]["text"]["full"] == "recover now"
    judgement = build_trace_judgement_context(digest)
    assert event_indexes(judgement) == [0, 1, 2, 3, 4]
    assert "sha256" not in str(judgement)
    assert "budget" not in judgement


def test_digest_marks_model_messages_before_first_user_as_pre_task_context() -> None:
    digest = digest_of([raw_event(0, "system", "policy"),
                        raw_event(1, "assistant", "canned greeting"),
                        raw_event(2, "user", "actual task"),
                        raw_event(3, "assistant", "task response")])
    refs = [item["event_ref"] for item in digest["ordered_trace"]]
    assert refs[1]["phase"] == "pre_task_context"
    assert all("phase" not in refs[index] for index in (0, 2, 3))


def test_final_compaction_keeps_task_boundary_tool_scope_and_agent_terminal():
    command = "fasttext supervised -input train.txt -output model\n" + ("x" * 400)
    digest = digest_of(
        [raw_event(0, "user", "A" * 3000 + "TASK_END"), raw_event(1, "assistant", "B" * 3000),
         raw_event(2, "agent", "trained " * 800)],
        [tool_call(1, "file_editor", "ok", '{"command":"replace","path":"/workspace/app.py"}'),
         tool_call(2, "bash_command", "ok", {"keystrokes": command, "duration": 0.1})],
        max_serialized_characters=100,
    )
    trace = digest["ordered_trace"]
    assert trace[0]["text"]["tail"].endswith("TASK_END")
    assert trace[1]["tool_calls"][0]["scope"] == {
        "operation": "replace", "path": "/workspace/app.py",
    }
    scope = trace[2]["tool_calls"][0]["scope"]
    assert scope["keystrokes"].startswith("fasttext supervised -input train.txt")
    assert len(scope["keystrokes"]) == 240
    assert scope["keystrokes_truncated"] is True
    assert trace[2]["text"]["tail"].endswith("trained ")
    assert digest["budget"]["requires_windowing"] is True


def test_digest_keeps_verifier_failure_evidence_with_compacted_trace():
    verifier_log = "\n".join(["ordinary setup" * 200, "expected: 1", "actual: 2", "FAIL"])
    digest = digest_of(
        [raw_event(0, "user", "task" * 1000)],
        verifier_artifacts=[{"artifact_idx": 1, "name": "test_stdout", "kind": "log",
                             "role": "verifier", "text": verifier_log}],
        max_serialized_characters=1_500,
    )
    evidence = digest["verifier_artifacts"][0]
    assert evidence["failure_extract"]["full"] == "expected: 1\nactual: 2\nFAIL"
    assert digest["coverage"]["verifier_artifact_count"] == 1
    judgement = build_trace_judgement_context(digest)
    assert judgement["verifier_artifacts"][0]["failure_extract"]["full"].endswith("FAIL")


def test_digest_includes_outcome_evidence_but_not_receipt_in_judge_context() -> None:
    # Source values may use the same key names as digest bookkeeping; they stay intact.
    outcome = {"semantics": "reference", "observed_outcome": {"reward": 0}, "items": [
        {"id": "reference-answer", "kind": "reference", "subject": "reference",
         "source_path": "expected_answer", "value": {
             "budget": 100, "sha256": "expected-answer-hash",
             "evidence_receipt": {"status": "source-value"}}},
    ]}
    receipt = {"trace": {"status": "included", "event_count": 1},
               "raw_outcome": {"status": "included", "item_ids_included": ["reference-answer"]}}
    digest = digest_of([raw_event(0, "user", "Choose A or B")],
                       outcome_evidence=outcome, evidence_receipt=receipt)
    assert digest["coverage"]["outcome_evidence_items"] == 1
    assert digest["evidence_receipt"] == receipt
    judgement = build_trace_judgement_context(digest)
    assert judgement["outcome_evidence"] == outcome
    assert "evidence_receipt" not in judgement


def test_digest_keeps_auxiliary_lane_with_only_a_tool_call() -> None:
    call = {
        **tool_call(3, "worker_tool", "error", {"value": 1}),
        "lane": "worker",
    }
    digest = digest_of(
        [raw_event(0, "user", "task")],
        auxiliary_events=[],
        auxiliary_tool_calls=[call],
    )

    assert [lane["lane"] for lane in digest["auxiliary_lanes"]] == ["worker"]
    [lane] = digest["auxiliary_lanes"]
    assert lane["coverage"]["observed_events"] == 0
    assert lane["coverage"]["observed_tool_calls"] == 1
    [unattached] = lane["unattached_tool_calls"]
    assert unattached["event_idx"] == 3
    assert unattached["tool_name"] == "worker_tool"
    assert unattached["status"] == "error"
    assert unattached["args"]["full"] == '{"value":1}'
    assert unattached["result"]["full"] == ""


def test_unchanged_error_retries_keep_the_first_and_last_body():
    error = "Error: expected integer, got string"
    essay = "the parameter requires a number " * 80
    calls = []
    events = [raw_event(0, "user", "write two items")]
    for event_idx in (1, 3, 5):
        calls.append(tool_call(event_idx, "write_items", "error", {"count": "2"},
                               result=event_idx + 1))
        events += [raw_event(event_idx, "assistant", essay),
                   raw_event(event_idx + 1, "tool", error)]
    digest = digest_of(events, calls, max_serialized_characters=4_500)
    assert event_indexes(digest) == list(range(7))
    assert digest["budget"]["compaction_level"] > 0
    by_event = {entry["event_ref"]["event_idx"]: entry for entry in digest["ordered_trace"]}
    assert digest["error_extract"][0]["repeat_count"] == 3
    assert by_event[1]["tool_calls"][0]["args"]["full"] == '{"count":"2"}'
    assert by_event[3]["tool_calls"][0]["same_failure_as_event_idx"] == 1
    assert "full" not in by_event[3]["tool_calls"][0]["args"]
    assert by_event[5]["tool_calls"][0]["result"]["full"] == error


def test_digest_keeps_same_index_events_and_calls_separate_by_lane():
    digest = digest_of(
        [raw_event(0, "assistant", "answer", lane=""),
         raw_event(0, "assistant", "grader note", lane="grader")],
        [tool_call(0, "lookup", "ok", {}, lane="")],
    )
    by_lane = {entry["event_ref"]["lane"]: entry for entry in digest["ordered_trace"]}
    assert len(by_lane[""]["tool_calls"]) == 1
    assert by_lane["grader"]["tool_calls"] == []


def test_compaction_caps_failed_calls_and_large_verifier_extracts() -> None:
    huge_error = "FAILED " * 40_000
    digest = digest_of(
        [raw_event(0, "user", "Fix the tests"), raw_event(1, "assistant", "running tests"),
         raw_event(2, "tool", huge_error)],
        [tool_call(1, "run_tests", "error", {"command": "pytest " + ("x" * 20_000)}, result=2)],
        verifier_artifacts=[
            {"artifact_idx": 3, "name": "tests.log", "role": "verifier", "text": huge_error},
        ],
        max_serialized_characters=10_000,
    )
    failed = digest["ordered_trace"][1]["tool_calls"][0]
    assert failed["result"]["truncated"] is True
    assert failed["result"]["characters"] == len(huge_error)
    assert digest["ordered_trace"][2]["text"]["omitted"] == "duplicate of the tool-call result"
    assert digest["verifier_artifacts"][0]["failure_extract"]["truncated"] is True
    assert "result" not in digest["error_extract"][0]
    assert digest["budget"]["judgement_serialized_characters"] < 10_000


# --- windows ---


def _event(event_idx, role, text, *, kind="message", result_event_idx=None) -> dict:
    """A digest ordered-trace entry, optionally with one call answered by result_event_idx."""
    calls = []
    if result_event_idx is not None:
        calls.append({
            "event_idx": event_idx, "call_idx": 0, "tool_name": "tool", "status": "ok",
            "args": {"value": "argument", "truncated": False, "characters": 8},
            "result_event_idx": result_event_idx,
            "result": {"event_ref": result_event_idx, "characters": len(text)}, "scope": None,
        })
    return {"event_ref": {"event_idx": event_idx, "kind": kind, "role": role},
            "text": {"full": text, "characters": len(text)}, "full_reason": "test",
            "tool_calls": calls}


def _digest(events: list[dict]) -> dict:
    return {"coverage": {"observed_events": len(events)}, "error_extract": [],
            "evidence_receipt": {"trace": {"status": "included"}}, "ordered_trace": events,
            "verifier_artifacts": [], "budget": {"target_serialized_characters": 32_000}}


def _size(window):
    return len(json.dumps(window, ensure_ascii=False, separators=(",", ":")))


def _source_sha256(event):
    source = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(source.encode()).hexdigest()


def _visible(window):
    return [event for section in ("task_anchor", "terminal_anchor", "ordered_trace")
            for event in window[section]]


def _chunks(windows):
    """Chunked events by source hash, then by chunk index (overlapping copies collapse)."""
    chunks = {}
    for window in windows:
        for event in _visible(window):
            if receipt := event.get("event_chunk"):
                chunks.setdefault(receipt["source_sha256"], {})[receipt["index"]] = event
    return chunks


def _rebuild(chunks):
    """Decode a source event from its ordered chunks, dropping repeated overlap."""
    return json.loads("".join(event["text"]["value"][event["event_chunk"]["overlap_before"]:]
                              for _, event in sorted(chunks.items())))


def _calls_resolve_locally(window):
    local = {event["event_ref"]["event_idx"] for event in window["ordered_trace"]}
    return all(call["result_event_idx"] in local
               for event in window["ordered_trace"] for call in event["tool_calls"])


def test_under_budget_context_passes_through_unchanged():
    digest = _digest([_event(0, "user", "task"), _event(1, "assistant", "answer")])
    windows = build_trace_judgement_windows(digest, max_serialized_characters=10_000)
    assert windows == [build_trace_judgement_context(digest)]


def test_oversized_context_splits_on_atomic_event_boundaries():
    events = [_event(0, "system", "system " * 20), _event(1, "developer", "developer " * 20),
              _event(2, "user", "task " * 20)]
    # Each call's result arrives after an interleaved progress message.
    for start in range(3, 15, 3):
        events += [_event(start, "assistant", f"call-{start} " * 55, result_event_idx=start + 2),
                   _event(start + 1, "assistant", f"progress-{start} " * 55),
                   _event(start + 2, "tool", f"result-{start} " * 55, kind="tool_result")]
    events.append(_event(15, "assistant", "terminal " * 40))
    digest = _digest(events)
    digest["outcome_evidence"] = {
        "semantics": "reference",
        "items": [{"id": "expected", "kind": "reference", "value": "A"}],
    }
    windows = build_trace_judgement_windows(digest, max_serialized_characters=6_500)
    assert len(windows) > 1
    assert all(_size(window) <= 6_500 for window in windows)
    # System, developer and user instructions anchor every window.
    assert all([event["event_ref"]["event_idx"] for event in window["task_anchor"]] == [0, 1, 2]
               for window in windows)
    assert all(window["terminal_anchor"][0]["event_ref"]["event_idx"] == 15 for window in windows)
    assert all(window["outcome_evidence"] == digest["outcome_evidence"] for window in windows)
    covered = {event["event_ref"]["event_idx"] for window in windows for event in _visible(window)}
    assert covered == set(range(16))
    assert all(_calls_resolve_locally(window) for window in windows)
    for previous, current in zip(windows, windows[1:], strict=False):
        previous_indexes = {event["event_ref"]["event_idx"] for event in previous["ordered_trace"]}
        current_indexes = {event["event_ref"]["event_idx"] for event in current["ordered_trace"]}
        assert previous_indexes & current_indexes


def test_windows_preserve_bounded_auxiliary_lane_and_resource_projection() -> None:
    events = [_event(0, "user", "task")] + [
        _event(index, "assistant", f"primary-{index} " * 80)
        for index in range(1, 8)
    ]
    digest = _digest(events)
    digest["auxiliary_lanes"] = [{
        "lane": "worker",
        "ordering": "independent",
        "coverage": {"observed_events": 1, "observed_tool_calls": 0, "error_call_count": 0},
        "error_extract": [],
        "ordered_trace": [{
            **_event(1, "assistant", "auxiliary fact"),
            "event_ref": {
                "lane": "worker", "event_idx": 1, "kind": "message", "role": "assistant",
            },
        }],
    }]
    digest["trace_resources"] = {
        "attempts": [{"resource_idx": 0, "value": {"error": "retry detail"}}],
    }

    windows = build_trace_judgement_windows(
        digest, max_serialized_characters=4_000
    )

    assert len(windows) > 1
    assert all(_size(window) <= 4_000 for window in windows)
    auxiliary_window = next(
        window for window in windows if window.get("auxiliary_lanes")
    )
    assert auxiliary_window["auxiliary_lanes"][0]["lane"] == "worker"
    assert auxiliary_window["window"]["scope"] == "auxiliary_lane"
    assert auxiliary_window["window"]["lane"] == "worker"
    assert auxiliary_window["window"]["start_event_idx"] == 1
    assert any(
        window.get("trace_resources", {}).get("attempts", [{}])[0].get("value", {}).get("error")
        == "retry detail"
        for window in windows
    )


def test_windows_chunk_oversized_resource_rows_without_dropping_content() -> None:
    digest = _digest([
        _event(0, "user", "task"),
        _event(1, "assistant", "terminal"),
    ])
    row = {"resource_idx": 0, "value": {"msg_json": "prompt-" * 2_000}}
    digest["trace_resources"] = {"prompt_messages": [row]}

    windows = build_trace_judgement_windows(
        digest, max_serialized_characters=2_000
    )

    assert all(_size(window) <= 2_000 for window in windows)
    chunks = [
        item
        for window in windows
        for item in window.get("trace_resources", {}).get("prompt_messages", [])
    ]
    assert chunks
    assert all(
        window["window"]["scope"] == "trace_resources"
        and window["window"]["start_event_idx"] is None
        for window in windows
        if window.get("trace_resources")
    )
    rebuilt = "".join(
        item["value"]
        for item in sorted(chunks, key=lambda item: item["resource_chunk"]["index"])
    )
    assert json.loads(rebuilt) == row


def test_overlap_is_dropped_when_each_unit_only_fits_alone():
    events = [_event(0, "user", "task " * 100), _event(1, "assistant", "first " * 300),
              _event(2, "assistant", "second " * 300), _event(3, "assistant", "terminal " * 100)]
    windows = build_trace_judgement_windows(
        _digest(events), max_serialized_characters=5_000
    )
    assert len(windows) == 2
    assert all(window["window"]["overlap_event_indexes"] == [] for window in windows)


def test_one_oversized_atomic_unit_is_exactly_chunked():
    events = [_event(0, "user", "task"), _event(1, "assistant", "call", result_event_idx=2),
              _event(2, "tool", "y" * 5_000, kind="tool_result"),
              _event(3, "assistant", "terminal")]
    windows = build_trace_judgement_windows(
        _digest(events), max_serialized_characters=2_000,
    )
    assert len(windows) > 1
    assert all(_size(window) <= 2_000 for window in windows)
    chunks = _chunks(windows)
    assert set(chunks) == {_source_sha256(events[2])}
    result_chunks = chunks[_source_sha256(events[2])]
    for event in result_chunks.values():
        assert event["event_ref"]["event_idx"] == 2
        [caller] = event["event_chunk"]["context"]["linked_callers"]
        assert caller["caller_event_ref"]["event_idx"] == 1
        assert caller["call"]["args"]["value"] == "argument"
        receipt = event["event_chunk"]
        assert hashlib.sha256(event["text"]["value"].encode()).hexdigest() == receipt["sha256"]
    assert _rebuild(result_chunks) == events[2]


def test_linked_unit_is_chunked_when_its_events_only_exceed_the_budget_together():
    events = [_event(0, "user", "task"), _event(1, "assistant", "x" * 5_000, result_event_idx=2),
              _event(2, "tool", "y" * 5_000, kind="tool_result"),
              _event(3, "assistant", "terminal")]
    windows = build_trace_judgement_windows(_digest(events), max_serialized_characters=10_000)
    assert all(_size(window) <= 10_000 for window in windows)
    chunks = _chunks(windows)
    for source_event in events[1:3]:
        assert _rebuild(chunks[_source_sha256(source_event)]) == source_event


def test_oversized_call_arguments_are_referenced_from_result_chunks() -> None:
    caller = _event(1, "assistant", "call", result_event_idx=2)
    caller["tool_calls"][0]["args"] = {"value": "argument" * 2_000, "truncated": False}
    windows = build_trace_judgement_windows(
        _digest([_event(0, "user", "task"), caller,
                 _event(2, "tool", "result" * 2_000, kind="tool_result"),
                 _event(3, "assistant", "terminal")]),
        max_serialized_characters=3_000,
    )
    assert all(_size(window) <= 3_000 for window in windows)
    caller_sha256 = _source_sha256(caller)
    ordered = [event for window in windows for event in window["ordered_trace"]]
    assert any(event.get("event_chunk", {}).get("source_sha256") == caller_sha256
               for event in ordered)
    result_chunks = [event for event in ordered
                     if event.get("event_chunk") and event["event_ref"]["event_idx"] == 2]
    assert result_chunks
    for event in result_chunks:
        [linked] = event["event_chunk"]["context"]["linked_callers"]
        assert "args" not in linked["call"]
        assert linked["call"]["args_reference"]["caller_source_sha256"] == caller_sha256


def test_anchor_only_window_uses_anchor_event_span() -> None:
    digest = _digest([_event(4, "user", "task " * 200), _event(9, "assistant", "terminal " * 200)])
    digest["error_extract"] = [{"tool_name": "failed_tool", "event_idx": 4, "event_idxs": [4],
                                "repeat_count": 1, "legacy_result": "x" * 10_000}]
    windows = build_trace_judgement_windows(digest, max_serialized_characters=5_000)
    assert len(windows) == 1
    assert windows[0]["window"]["start_event_idx"] == 4
    assert windows[0]["window"]["end_event_idx"] == 9
