# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detectors and judges read complete, scoped and lossless trace evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import trace
from autofa.config import Config, RunConfig
from autofa.core.detector_lint import evidence_budget_warnings, lint_detector_source
from autofa.core.library import FunctionLibrary
from autofa.core.sandbox import DetectorAbstained, NullLLM, run_detector
from autofa.data.evidence_refs import EVIDENCE_REFS_VERSION, compact_trace, expand_trace
from autofa.data.gym_adapter import materialize_gym
from autofa.data.parquet_store import Dataset
from autofa.data.source_completeness import (
    EVENT_CONTENT_LIMITATION,
    GENERATION_CUTOFF_NOTICE,
    TRUNCATED_EVENT_LIMITATION,
)
from autofa.data.source_normalization import normalize_events
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trajectory import parse_trajectory
from autofa.mcp import tools_eval, tools_helpers
from autofa.mcp.runtime import Runtime
from autofa.metrics import claude_judge
from autofa.metrics.claim_contract import build_claim_contract
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot
from autofa.trace import evidence_text

from support.judge import read_ranges
from support.traces import fixture

ROW = {"_item_id": "m|b|long-trace|0", "model_id": "m", "benchmark": "b",
       "problem_id": "long-trace", "run_idx": 0, "is_correct": False, "reasoning_text": ""}


def item(*texts, field="message", **row):
    """Return a dataset row whose trajectory has one assistant step per text."""
    return {**row, "response_text": json.dumps({"steps": [
        {"source": "assistant", field: text} for text in texts
    ]})}


# --- Long traces beyond the old 200k prefix cap ---

@pytest.mark.parametrize(("field", "step_field", "guess_at_end"), [
    ("reasoning_text", "reasoning_content", False), ("final_message", "message", True),
])
def test_detector_tail_uses_real_ending(field, step_field, guess_at_end):
    # An abandoned guess near 200k must not masquerade as the closing decision;
    # a genuine admission after 200k must not disappear either.
    middle = "An intermediate derivation." if guess_at_end else "This is a guess."
    text = "x" * 199_000 + middle + "y" * 30_000
    text += "This is a guess." if guess_at_end else "I derived the final solution."
    source = f'''
def detect(item, llm_client=None):
    return "this is a guess" in item.get("{field}", "").lower()[-9000:]
'''
    assert run_detector(source, item(text, field=step_field, **ROW)) is guess_at_end


def test_native_reasoning_is_preserved():
    text = "x" * 230_000 + "NATIVE_END"
    row = item("different trajectory reasoning", field="reasoning_content", **ROW)
    row["reasoning_text"] = text
    assert derive_trace_fields(row)["reasoning_text"] == text


def test_explicit_llm_tail_window_uses_full_reasoning():
    row = item("x" * 230_000 + "REAL_END", field="reasoning_content", **ROW)
    window = trace.evidence_text(row, surface="reasoning", focus="tail", max_chars=100)
    assert len(window) == 100
    assert window.endswith("REAL_END")


def test_judge_archives_full_evidence_but_bounds_prompt(tmp_path, monkeypatch, stub_claim_compiler):
    text = "x" * 230_000 + "REAL_END"
    row = item(text, field="reasoning_content", **ROW)

    async def judge_session(*, prompt, **kwargs):
        assert len(prompt) < 10_000
        return json.dumps({
            "verdicts": [{"item_id": row["_item_id"], "correct": False}],
            "behavior_specificity": 0.7,
        })

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", judge_session)
    claude_judge.judge_behavior_with_claude(
        name="tail_detector", mechanism="admitted guess at the end", description="",
        source="def detect(item, llm_client=None): return False", items=[row],
        assigned_labels=[True], cwd=tmp_path, model="test", max_chars=1000,
    )
    pack = json.loads((tmp_path / "behavior_judge/tail_detector.evidence.json").read_text())
    evidence = pack["evidence"][0]
    assert evidence["reasoning_text"].endswith(text)
    assert evidence["manifest"]["reasoning_text"]["total_chars"] == len(evidence["reasoning_text"])
    snapshot = json.loads(Path(evidence["evidence_path"]).read_text())
    assert snapshot["fields"]["reasoning_text"].endswith(text)


def test_gym_conversion_preserves_long_reasoning_and_response(tmp_path, monkeypatch):
    from autofa.data import gym_adapter

    monkeypatch.setattr(gym_adapter, "converted_root", lambda: tmp_path / "converted")
    reasoning = "x" * 230_000 + "REASONING_END"
    response = "y" * 2_010_000 + "RESPONSE_END"
    source = tmp_path / "parsed_trajectories.jsonl"
    source.write_text(json.dumps({
        "task_id": "long-trace", "is_resolved": False,
        "steps": [{"reasoning": reasoning, "message": response}],
    }) + "\n")
    path = materialize_gym(source)
    row = pq.read_table(path).to_pylist()[0]
    assert row["reasoning_text"] == reasoning
    assert json.loads(row["response_text"])["steps"][0]["message"] == response
    assert materialize_gym(source) == path


def test_smoke_detector_uses_complete_sampled_rows(tmp_path, monkeypatch):
    row = item("x" * 230_000 + "This is a guess.", field="reasoning_content", **ROW)
    path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    runtime = SimpleNamespace(
        dataset=Dataset(parquet_paths=[path]), config=SimpleNamespace(text_truncate_chars=100),
    )
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: runtime)
    monkeypatch.setattr(tools_eval, "_llm_adapter", NullLLM)
    result = tools_eval.smoke_test_feature('''
def detect(item, llm_client=None):
    return "this is a guess" in item.get("reasoning_text", "").lower()[-9000:]
''', n=1)
    assert result["errors"] == 0
    assert result["histogram"] == {"True": 1}


# --- trace.step_evidence: scoped selections without claiming full-history review ---

def test_search_reaches_late_evidence_and_later_repair_beyond_old_trace_limit():
    row = item("irrelevant " * 20_000, "target.py: broken implementation",
               "unrelated investigation", "target.py: repair applied and tests pass",
               "Submitted the repaired implementation.")
    result = trace.step_evidence(row, r"target\.py", context_steps=0, closing_steps=1)
    assert result["matched_step_indices"] == [2, 4]
    assert result["selected_step_indices"] == [2, 4, 5]
    assert result["omitted_step_indices"] == [1, 3]
    assert result["selection_complete"] is True
    assert result["full_trace"] is False
    assert result["text"].index("broken implementation") < result["text"].index("repair applied")
    assert "Submitted the repaired implementation." in result["text"]
    assert len(result["text"]) <= 16_000


def test_nonadjacent_call_result_links_survive_selection():
    events = [
        {"event_idx": 4, "role": "assistant", "kind": "tool_call", "parts": []},
        {"event_idx": 6, "role": "assistant", "kind": "message", "parts": [
            {"part_idx": 0, "part_type": "text", "text": "unrelated"}]},
        {"event_idx": 8, "role": "tool", "kind": "tool_result", "parts": [
            {"part_idx": 0, "part_type": "text", "text": "repair confirmed"}],
         "paired_event_idx": 4, "paired_call_idx": 0},
    ]
    calls = [{"event_idx": 4, "call_idx": 0, "tool_name": "terminal",
              "args_full": '{"command":"edit target.py"}', "args_chars": 28,
              "result_event_idx": 8}]
    for event in events:
        event["n_parts"] = len(event["parts"])
        for part in event["parts"]:
            part["chars"] = len(part["text"])
    row = {"response_text": json.dumps(normalize_events(
        {"n_events": 3, "n_tool_calls": 1}, events, calls,
    ))}
    for pattern in (r"target\.py", "repair confirmed"):
        result = trace.step_evidence(row, pattern, context_steps=0, closing_steps=0)
        assert result["selected_step_indices"] == [1, 3]
        assert "target.py" in result["text"] and "repair confirmed" in result["text"]
        assert result["selection_complete"] and not result["full_trace"]


def test_oversized_steps_keep_match_excerpt_real_ending_and_explicit_read_scope():
    original = "x" * 120_000 + "TARGET Ω wrong transform" + "y" * 120_000
    row = item(original, "Finished.")
    result = trace.step_evidence(row, "TARGET Ω", context_steps=0, closing_steps=1, max_chars=512)
    assert "TARGET Ω wrong transform" in result["text"]
    assert "Finished." in result["text"]
    assert "excerpt" in result["text"]
    assert result["truncated_step_indices"] == [1]
    assert result["omitted_step_indices"] == []
    assert not result["selection_complete"] and not result["full_trace"]
    assert len(result["text"]) <= 512
    assert trace.steps(row)[0].message == original
    row = item("TARGET issue", "x" * 120_000 + "REPAIR_AT_REAL_END")
    result = trace.step_evidence(row, "TARGET", context_steps=0, closing_steps=1, max_chars=512)
    assert "TARGET issue" in result["text"]
    assert "REPAIR_AT_REAL_END" in result["text"]
    assert result["truncated_step_indices"] == [2]
    assert not result["full_trace"]


def test_many_selected_steps_do_not_recreate_empty_length_rejection():
    row = item(*(f"TARGET {index}" for index in range(2000)))
    result = trace.step_evidence(row, "TARGET", max_chars=512)
    assert result["text"] and len(result["text"]) <= 512
    assert len(result["matched_step_indices"]) == 2000
    assert result["omitted_step_indices"]
    assert not result["selection_complete"] and not result["full_trace"]


def test_whole_short_trace_can_be_complete_and_no_match_is_only_regex_absence():
    result = trace.step_evidence(item("No target here.", "Done."), "not-found")
    assert result["matched_step_indices"] == []
    assert result["full_trace"] and result["selection_complete"]
    selected = trace.step_evidence(item("No target here."), "not-found", closing_steps=0)
    assert selected["status"] == "ok" and selected["text"] == ""
    assert selected["selection_complete"] and not selected["full_trace"]
    # Many short matches that fit the budget are not dropped.
    many = trace.step_evidence(item(*("target" for _ in range(100))), "target")
    assert many["full_trace"] and many["selection_complete"]
    assert many["omitted_step_indices"] == []


def test_missing_trace_is_unavailable_not_absent():
    result = trace.step_evidence({"response_text": "not a trajectory"}, "TARGET")
    assert result["status"] == "unavailable"
    assert result["text"] == ""
    assert not result["selection_complete"] and not result["full_trace"]


def test_helper_is_available_in_sandbox_and_incomplete_scope_can_abstain():
    source = '''
def detect(item):
    from autofa.trace import step_evidence
    evidence = step_evidence(item, "TARGET", context_steps=0, closing_steps=0)
    if not evidence["full_trace"]:
        return library.abstain("Global absence requires the omitted history")
    return False
'''
    with pytest.raises(DetectorAbstained, match="omitted history"):
        run_detector(source, item("unrelated " * 20_000, "TARGET"))
    assert run_detector(source, item("TARGET")) is False


LENGTH_REJECTION = '''
def extract(item, budget=100000):
    text = item.get("response_text", "")
    if len(text) > budget:
        return {"complete": False, "text": "", "reason": "Over budget"}
    return {"complete": True, "text": text, "reason": ""}
'''


def test_empty_evidence_length_guard_gets_actionable_nonfatal_warning():
    warnings = evidence_budget_warnings(LENGTH_REJECTION)
    assert len(warnings) == 1 and "step_evidence" in warnings[0]


def test_honest_scoped_fallback_does_not_warn():
    source = LENGTH_REJECTION.replace(
        '{"complete": False, "text": "", "reason": "Over budget"}',
        'step_evidence(item, "target.py")',
    )
    assert evidence_budget_warnings(source) == []


def test_helper_warnings_preserve_result(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path)
    rt = Runtime(
        dataset=Dataset(parquet_paths=[]), library=FunctionLibrary(),
        run=RunConfig(run_id="long-helper", parquet_paths=[], model_id="m"), config=Config(),
    )
    monkeypatch.setattr(tools_helpers, "get_runtime", lambda: rt)
    submitted = tools_helpers.submit_helper("old_style", LENGTH_REJECTION)
    assert submitted["ok"] and submitted["evidence_warnings"]
    assert tools_helpers.get_helper("old_style")["evidence_warnings"]
    tested = tools_helpers.test_helper(
        "old_style", inputs=[{"args": [{"response_text": "x" * 100_001}]}],
    )
    assert tested["evidence_warnings"]
    assert tested["results"][0]["value"] == {"complete": False, "text": "", "reason": "Over budget"}


# --- Deliberate evidence windows for LLM classify (agentic + non-agentic) ---

STEPS = ("early plan: rewrite the parser", "middle: still failing tests",
         "closing: tests still fail but I will finish anyway")


@pytest.mark.parametrize("row,options,present,absent", [
    (item(*STEPS, field="reasoning_content"),
     {"surface": "reasoning", "focus": "closing", "max_steps": 2},
     ("finish anyway", "still failing tests"), ("early plan",)),
    (item(*STEPS, field="reasoning_content"),
     {"surface": "reasoning", "focus": "opening", "max_steps": 1},
     ("early plan",), ("finish anyway",)),
    ({"response_text": "INTRO " + "x" * 8_000 + " ACKNOWLEDGE_BROKEN_AND_SHIP"},
     {"surface": "response", "focus": "closing", "max_chars": 500},
     ("ACKNOWLEDGE_BROKEN_AND_SHIP",), ("INTRO",)),
    ({"reasoning_text": "I think the bug is in auth", "response_text": "final answer 42"},
     {"surface": "auto", "focus": "full"}, ("auth",), ()),
], ids=["step-closing", "step-opening", "prose-closing", "auto-prefers-reasoning"])
def test_evidence_text_selects_the_requested_window(row, options, present, absent):
    text = evidence_text(row, **options)
    assert all(snippet in text for snippet in present)
    assert not any(snippet in text for snippet in absent)


@pytest.mark.parametrize("window,allowed", [
    ("text[-4000:]", False), ('text, focus="tail"', True),
], ids=["magic-slice", "focus-kwarg"])
def test_lint_requires_classify_focus_instead_of_magic_slice(window, allowed):
    src = f'''
def ships_broken(item, llm_client=None):
    text = item.get("reasoning_text") or ""
    if llm_client is None:
        return "unknown"
    return llm_client.classify("ships broken?", {window}) == "yes"
'''
    ok, reason = lint_detector_source(src)
    assert ok is allowed, reason


# --- Lossless source-backed compaction keeps legacy raw content and helper contracts ---

MARKER = "_autofa_evidence_refs"


def envelope():
    metadata, events, calls = fixture()
    for event in events:
        event["lane"] = ""
        event["provider_extension"] = {"latency_ms": 12, "details": ["retained"]}
        for part in event["parts"]:
            part["lane"] = ""
    for call in calls:
        call["lane"] = ""
    part = events[3]["parts"][0]
    part["text"] = '  λ\\quoted "tool result"\n' * 6000
    part["chars"] = len(part["text"])
    second = {**part, "part_idx": 2, "text": "tail evidence", "chars": 13}
    events[3]["parts"].append(second)
    events[3]["n_parts"] = 2
    calls[0]["args_full"] = json.dumps({"command": "long arguments " * 3000})
    calls[0]["args_chars"] = len(calls[0]["args_full"])
    return normalize_events(metadata, events, calls)


@pytest.mark.parametrize("include_rendered", [False, True])
def test_exact_roundtrip_and_meaningful_size_reduction(include_rendered):
    original = envelope()
    before = deepcopy(original)
    compact = compact_trace(original, include_rendered=include_rendered)
    assert original == before
    assert compact[MARKER]["version"] == EVIDENCE_REFS_VERSION
    assert compact["execution"] == original["execution"]
    assert expand_trace(compact) == original
    assert compact_trace(compact, include_rendered=include_rendered) == compact
    ratio = 0.44 if include_rendered else 0.73
    assert len(json.dumps(compact)) < len(json.dumps(original)) * ratio
    if not include_rendered:
        for old, new in zip(original["steps"], compact["steps"], strict=True):
            for field in ("message", "observation", "reasoning_content"):
                assert new.get(field) == old.get(field)
            for old_call, new_call in zip(old.get("tool_calls", []), new.get("tool_calls", []),
                                          strict=True):
                for field in ("function_name", "arguments", "source_call_index",
                              "result_event_idx"):
                    assert new_call[field] == old_call[field]
    else:
        assert "observation" not in compact["steps"][3]
        assert "arguments" not in compact["steps"][1]["tool_calls"][0]


def test_shared_parser_and_helpers_return_identical_evidence():
    # include_rendered=True also drops rendered step text, so it is the stricter mode.
    original = envelope()
    compact = compact_trace(original, include_rendered=True)
    full_item = {"response_text": json.dumps(original)}
    compact_item = {"response_text": json.dumps(compact)}
    assert parse_trajectory(compact_item["response_text"]) == parse_trajectory(
        full_item["response_text"]
    )
    for helper in (trace.steps, trace.events, trace.record, trace.tool_calls, trace.coverage):
        assert helper(compact_item) == helper(full_item)
    full_fields, compact_fields = derive_trace_fields(full_item), derive_trace_fields(compact_item)
    for field in ("actions_text", "observations_text", "reasoning_text", "final_message"):
        assert compact_fields.get(field) == full_fields.get(field)
    assert parse_trajectory([compact])._steps == parse_trajectory(original)._steps


def test_results_and_expansion_are_independent_copies():
    original = envelope()
    compact = compact_trace(original)
    before = deepcopy(compact)
    expanded = expand_trace(compact)
    assert compact == before
    expanded["steps"][0]["metadata"]["provider_extension"]["details"].append("change")
    expanded["steps"][0]["parts"][0]["text"] = "change"
    assert compact == before
    assert original["steps"][0]["metadata"]["provider_extension"]["details"] == ["retained"]
    assert expanded["execution"] == original["execution"]
    compact["execution"]["events"][0]["provider_extension"]["details"].append("change")
    assert original["execution"]["events"][0]["provider_extension"]["details"] == ["retained"]


def test_conflicting_values_and_unknown_extensions_stay_inline():
    original = envelope()
    original["custom"] = {"future": [1, 2]}
    step = original["steps"][0]
    step["future_step_field"] = {"text": "fix it"}
    step["metadata"]["custom"] = "not in original event"
    step["parts"][0]["text"] = "different representation"
    call = original["steps"][1]["tool_calls"][0]
    call["status"] = "custom status"
    call["custom"] = {"preserve": True}
    compact = compact_trace(original, include_rendered=True)
    assert compact["steps"][0]["metadata"] == step["metadata"]
    assert compact["steps"][0]["parts"] == step["parts"]
    assert compact["steps"][0]["future_step_field"] == step["future_step_field"]
    assert compact["steps"][1]["tool_calls"][0]["status"] == "custom status"
    assert compact["steps"][1]["tool_calls"][0]["custom"] == {"preserve": True}
    assert compact["custom"] == original["custom"]
    assert expand_trace(compact) == original


def test_equal_but_differently_typed_values_are_not_shared():
    original = envelope()
    original["execution"]["tool_calls"][0]["count"] = 1
    original["steps"][1]["tool_calls"][0]["count"] = True
    compact = compact_trace(original)
    assert compact["steps"][1]["tool_calls"][0]["count"] is True
    assert expand_trace(compact)["steps"][1]["tool_calls"][0]["count"] is True


def test_duplicate_event_and_call_identities_are_never_guessed():
    original = envelope()
    original["execution"]["events"].append(deepcopy(original["execution"]["events"][0]))
    original["execution"]["tool_calls"].append(deepcopy(original["execution"]["tool_calls"][0]))
    compact = compact_trace(original, include_rendered=True)
    assert compact["steps"][0] == original["steps"][0]
    assert compact["steps"][1]["tool_calls"] == original["steps"][1]["tool_calls"]
    assert expand_trace(compact) == original


def test_same_event_id_in_other_lane_is_a_distinct_source_occurrence():
    original = envelope()
    auxiliary = deepcopy(original["execution"]["events"][0])
    auxiliary["lane"] = "worker"
    auxiliary["parts"][0]["text"] = "different source occurrence"
    original["execution"]["events"].insert(0, auxiliary)
    compact = compact_trace(original, include_rendered=True)
    assert "message" not in compact["steps"][0]
    assert expand_trace(compact) == original
    original["execution"]["events"] = [auxiliary]
    assert compact_trace(original, include_rendered=True) == original


@pytest.mark.parametrize("mutation", [
    lambda value: value["steps"][3].pop("source_parts"),
    lambda value: value["steps"][3].update(observation="a different rendering"),
])
def test_unverified_joined_text_stays_inline(mutation):
    original = envelope()
    mutation(original)
    compact = compact_trace(original, include_rendered=True)
    assert compact["steps"][3]["observation"] == original["steps"][3]["observation"]
    assert expand_trace(compact) == original


@pytest.mark.parametrize("mutation", [
    lambda value: value["evidence_scope"].update(trajectory_source="recovered raw trace"),
    lambda value: value.pop("execution"),
])
def test_unknown_and_legacy_shapes_are_not_inferred(mutation):
    original = envelope()
    mutation(original)
    assert compact_trace(original, include_rendered=True) == original
    assert expand_trace(original) == original


# One mutation per validation branch of an explicit reference.
@pytest.mark.parametrize("mutation", [
    lambda value: value[MARKER].update(version=999),
    lambda value: value[MARKER].update(version=True),  # equal to version 1, but not an int
    lambda value: value[MARKER]["steps"][0].update(index=-1),
    lambda value: value[MARKER]["steps"][0].update(event_idx=999),
    lambda value: value[MARKER]["steps"][0].update(fields=["unknown"]),
    lambda value: value["steps"][0].update(metadata={}),
    lambda value: value["execution"]["events"].append(deepcopy(value["execution"]["events"][0])),
    lambda value: value[MARKER]["steps"][1]["calls"][0].update(call_idx=999),
    lambda value: value[MARKER]["steps"][1]["calls"][0].update(fields=["source_call_index"]),
    lambda value: value[MARKER]["steps"][1]["calls"][0].update(fields=["missing"]),
    lambda value: value["steps"][3]["source_parts"][1].update(start=1),
])
def test_malformed_explicit_references_fail_visibly(mutation):
    compact = compact_trace(envelope(), include_rendered=True)
    mutation(compact)
    before = deepcopy(compact)
    with pytest.raises(ValueError, match="evidence referenc"):
        expand_trace(compact)
    with pytest.raises(ValueError, match="evidence referenc"):
        compact_trace(compact)
    assert compact == before
    parsed = parse_trajectory(json.dumps(compact))
    assert not parsed.ok
    assert "evidence referenc" in parsed.warning


# --- A fully exported provider cutoff must not masquerade as missing source bytes ---

SOURCE = "def detect(item): return True"
FINAL = "Final commitment, then wait once more."


def _cutoff_trace():
    body = "Earlier grounded analysis.\n" * 1000 + "I commit to A. Wait.\n" * 400 + FINAL

    def event(index, kind, role, text, channel):
        return {
            "event_idx": index, "lane": "", "kind": kind, "role": role, "n_parts": 1,
            "parts": [{"event_idx": index, "lane": "", "part_idx": 0,
                       "part_type": "text", "channel": channel,
                       "text": text, "chars": len(text)}],
        }

    events = [event(0, "message", "user", "Read the chart and answer.", "content"),
              event(1, "reasoning", "output", body, "reasoning")]
    events[1].update(length_truncated=True, finish_reason="incomplete", completion_tokens=100000)
    return {
        "steps": [
            {"source": "user", "step_id": 0, "message": events[0]["parts"][0]["text"]},
            {"source": "assistant", "step_id": 1, "reasoning_content": body},
        ],
        "evidence_scope": {"source": "original API event parts and tool arguments",
                           "event_lane": "", "limitations": {},
                           "notices": {GENERATION_CUTOFF_NOTICE: 1}},
        "execution": {
            "rollout": {"n_events": 2, "n_assistant_turns": 0, "n_reasoning_segments": 1,
                        "n_tool_calls": 0, "completion_tokens": 100000},
            "events": events, "tool_calls": [],
            "availability": {"events": {
                "status": "available",
                "reason": "Exported table retrieved completely; evaluation records are excluded.",
            }},
        },
    }


def _cutoff_review(tmp_path, trace):
    description = "The response keeps promising an answer until the provider stops generation."
    clause = {
        "id": "ending", "source_field": "description", "quote": description,
        "predicate": description, "required_review_scope": "bounded_interval",
        "review_interval": {
            "start": {"kind": "event", "predicate": "The response starts", "position": "before"},
            "end": {"kind": "event", "predicate": "The provider ends this response",
                    "position": "after"},
        },
    }
    contract = build_claim_contract(source=SOURCE, description=description, mechanism="",
                                    defining_clauses=[clause])
    snapshot = make_snapshot({"response_text": trace}, item_id="case", source=SOURCE,
                             claim_contract=contract)
    return EvidenceReview(snapshot, tmp_path / "review")


def test_cutoff_event_can_be_read_without_claiming_a_completed_answer(tmp_path):
    trace = _cutoff_trace()
    trace["execution"]["events"][1]["finish_reason"] = "length"
    original = deepcopy(trace)
    review = _cutoff_review(tmp_path, trace)
    navigation = review.interval_navigation()
    assert navigation["available"] and navigation["generation_cutoff_events"] == [1]
    assert all(event["complete"] for event in navigation["events"])
    # These are review receipts, not generated verdicts. Reading the entire event
    # remains mandatory, and the retained source/snapshot stays byte-for-byte intact.
    span = navigation["events"][1]["ranges"][-1]
    review.read(span["field"], span["start"], 200)
    anchor = review.cite(span["field"], '"event_idx": 1', offset=span["start"])["citations"][0]
    receipt = review.review_interval("ending", 2, 2, start_citation_id=anchor["citation_id"],
                                     end_citation_id=anchor["citation_id"])
    assert not receipt["complete"]
    read_ranges(review, receipt["required_ranges"])
    assert review.status()["review_intervals"][0]["complete"]
    assert review.snapshot["fields"]["trace_structure"].find('"n_assistant_message_events": 0') >= 0
    assert trace == original


_UNRETRIEVED = "events were not completely retrieved"
_LIMITED = "recorded evidence limitations prevent complete interval coverage"


@pytest.mark.parametrize(("damage", "unavailable"), [
    (lambda trace, event: event.pop("finish_reason"), None),  # bare cutoff without a finish reason
    (lambda trace, event: event["parts"][0].update(chars=event["parts"][0]["chars"] + 1), None),
    (lambda trace, event: trace["execution"]["availability"].update(
        events={"status": "partial", "reason": "Page failed"}), _UNRETRIEVED),
    (lambda trace, event: trace["evidence_scope"]["limitations"].update(
        {"missing tool events": 1}), _LIMITED),
    (lambda trace, event: trace["evidence_scope"]["limitations"].update(
        {TRUNCATED_EVENT_LIMITATION: 1}), _LIMITED),
    (lambda trace, event: trace["execution"]["availability"].update(
        events={"status": "partial", "reason": EVENT_CONTENT_LIMITATION}), _UNRETRIEVED),
], ids=["bare_cutoff", "short_part", "unretrieved_events", "unknown_scope_loss",
        "truncated_event_label", "unavailable_event_content"])
def test_cutoff_does_not_waive_actual_source_loss(tmp_path, damage, unavailable):
    trace = _cutoff_trace()
    damage(trace, trace["execution"]["events"][1])
    review = _cutoff_review(tmp_path, trace)
    navigation = review.interval_navigation()
    if unavailable is None:
        # A damaged event stays incomplete while the rest of the trace remains navigable.
        assert navigation["available"] and "generation_cutoff_events" not in navigation
        assert [event["complete"] for event in navigation["events"]] == [True, False]
        reason = "interval contains incomplete or truncated source events"
    else:
        # Declared source loss makes the interval unavailable.
        assert navigation == {"available": False, "reason": unavailable}
        reason = unavailable
    with pytest.raises(ValueError, match=reason):
        review.review_interval("ending", 2, 2)
    assert not review.intervals


def test_fresh_normalization_keeps_cutoff_separate_from_source_availability():
    execution = _cutoff_trace()["execution"]
    execution["availability"]["events"] = {"status": "available", "reason": "All events retrieved"}
    execution["availability"]["tool_calls"] = {"status": "available",
                                               "reason": "All calls retrieved"}
    normalized = normalize_events(execution["rollout"], execution["events"], [],
                                  execution=execution)
    assert normalized["execution"]["availability"]["events"]["status"] == "available"
    assert TRUNCATED_EVENT_LIMITATION not in normalized["evidence_scope"]["limitations"]
    assert normalized["evidence_scope"]["notices"][GENERATION_CUTOFF_NOTICE] == 1
    assert normalized["execution"]["events"][1]["length_truncated"] is True
    execution["events"][1]["content_truncated"] = True
    damaged = normalize_events(execution["rollout"], execution["events"], [], execution=execution)
    assert damaged["execution"]["availability"]["events"]["status"] == "partial"
