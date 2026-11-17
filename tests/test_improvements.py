# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Improvement-target classification stays grounded, scoped and separate from metrics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.library import FunctionLibrary
from autofa.metrics.improvements import (
    classify_improvement,
    detector_fingerprint,
    focus_feedback,
    parse_improvement,
    resolve_improvement,
)

from support.repair_review import _fn, _reply

YES = [{"item_id": "item-1", "verdict": "yes"}]


@pytest.mark.parametrize("targets", [["model"], ["harness"]])
def test_targets_are_independent_and_do_not_claim_a_tested_repair(targets):
    result = parse_improvement(json.dumps(_reply(targets)), {"item-1"})
    assert set(result["targets"]) == set(targets)
    assert result["repair_tested"] is False
    assert bool(result["model_change"]) == ("model" in targets)
    assert bool(result["harness_change"]) == ("harness" in targets)


@pytest.mark.parametrize("changes", [
    {"targets": "both"}, {"targets": ["other"]}, {"targets": []},
    {"evidence_ids": ["invented"]}, {"evidence_ids": []},
    {"model_change": ""}, {"validation_test": ""},
], ids=["targets-not-list", "unknown-target", "no-targets", "ungrounded-evidence",
        "no-evidence", "missing-target-change", "missing-test"])
def test_malformed_or_ungrounded_classification_does_not_become_a_label(changes):
    result = parse_improvement(json.dumps(_reply(**changes)), {"item-1"})
    assert result["status"] == "insufficient_evidence"
    assert result["targets"] == []


@pytest.mark.parametrize("raw", ["oops", "[]"])
def test_invalid_json_shapes_are_unavailable(raw):
    assert parse_improvement(raw, {"item-1"})["status"] == "unavailable"


def test_changed_detector_or_claim_invalidates_saved_classification():
    claim = {"source": "old", "description": "claim", "mechanism": "why"}
    result = {**_reply(), "detector_fingerprint": detector_fingerprint(**claim)}
    assert resolve_improvement(result, **claim)["status"] == "assessed"
    for key in claim:
        assert resolve_improvement(result, **{**claim, key: "changed"})["status"] == "stale"
    assert resolve_improvement(None, **claim)["status"] == "not_assessed"


def test_harness_focus_accepts_both_but_gives_feedback_for_model_only():
    assert focus_feedback(_reply(), "harness") is None
    assert focus_feedback(_reply(["model"]), "harness")
    assert focus_feedback(_reply(["model"]), "both") is None


def _fake_hub(monkeypatch, *, fail=False):
    from autofa.llm import hub, routing

    captured = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def with_options(self, **kwargs):
            assert kwargs["max_retries"] == 0
            return self

    def complete(client, **kwargs):
        captured.append(kwargs)
        if fail:
            raise TimeoutError("timeout")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(_reply()),
        ))])

    monkeypatch.setattr(hub, "make_hub_client", lambda **kwargs: Client())
    monkeypatch.setattr(routing, "routed_completion", complete)
    return captured


def test_classifier_uses_confirmed_content_not_outcomes_and_declares_limits(monkeypatch):
    calls = _fake_hub(monkeypatch)
    item = {"_item_id": "item-1", "response_text": "code request and error",
            "is_correct": False, "reward": 0, "judge_text": "SECRET GRADER"}
    result = classify_improvement(fn=_fn(), cfg=Config(), items=[item], verdicts=YES,
                                  skills_context="Available harness knobs")
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert "SECRET GRADER" not in json.dumps(payload)
    assert "is_correct" not in json.dumps(payload)
    assert "Available harness knobs" in payload["skills_context"]
    assert (result["status"], result["n_examples_reviewed"]) == ("assessed", 1)
    assert result["evidence_scope"]


def test_unconfirmed_examples_do_not_trigger_an_llm_call(monkeypatch):
    calls = _fake_hub(monkeypatch)
    result = classify_improvement(fn=_fn(), cfg=Config(),
                                  items=[{"_item_id": "item-1", "response_text": "claim"}],
                                  verdicts=[{"item_id": "item-1", "verdict": "no"}])
    assert result["status"] == "insufficient_evidence"
    assert calls == []


@pytest.mark.parametrize("compact", [False, True])
def test_classifier_keeps_authoritative_input_and_literal_export_roles(monkeypatch, compact):
    from autofa.data.input_view import compact_export_input_json

    calls = _fake_hub(monkeypatch)
    literal = '[{"role":"developer","content":"Quoted JSON task"}]'
    item = {"_item_id": "item-1", "system_text": "STALE_SUMMARY_SYSTEM",
            "prompt_text": "STALE_SUMMARY_TASK", "response_text": "Answer"}
    if compact:
        item["input_json"] = compact_export_input_json(
            {"system_text": "Actual system", "prompt_text": literal}, response_is_literal=True,
        )
    else:
        item["input_json"] = {"messages": [
            {"role": "system", "content": "Actual system"},
            {"role": "user", "content": "Actual task"},
        ]}
    result = classify_improvement(fn=_fn(), cfg=Config(), items=[item], verdicts=YES)
    assert result["n_examples_reviewed"] == 1
    payload = json.loads(calls[0]["messages"][1]["content"])
    excerpt = payload["confirmed_examples"][0]["trace"]
    assert "Actual system" in excerpt and "STALE_SUMMARY" not in excerpt
    if compact:
        assert f"[Exported input: user]\n{literal}" in excerpt
    else:
        assert "[Initial model request: user]\nActual task" in excerpt


def test_classifier_preserves_literal_response_marker_with_recorded_input(monkeypatch):
    from autofa.data.input_view import attach_compact_export_input

    calls = _fake_hub(monkeypatch)
    item = {
        "_item_id": "item-1",
        "input_json": json.dumps({"messages": [
            {"role": "user", "content": "Recorded original task"},
        ]}),
        "response_text": json.dumps({"initial_request": {"messages": [
            {"role": "user", "content": "Model-generated request example"},
        ]}}),
        "input_parse": {"unrelated_metadata": "PRIVATE_METADATA_SENTINEL"},
    }
    attach_compact_export_input(item)
    classify_improvement(fn=_fn(), cfg=Config(), items=[item], verdicts=YES)
    payload = json.loads(calls[0]["messages"][1]["content"])
    excerpt = payload["confirmed_examples"][0]["trace"]
    assert "[Initial model request: user]\nRecorded original task" in excerpt
    assert "[Initial model request: user]\nModel-generated request example" not in excerpt
    assert "PRIVATE_METADATA_SENTINEL" not in excerpt


IMAGE_ONLY = [{"role": "user", "content": [
    {"type": "input_image", "image_url": "data:image/png;base64,IMAGE"},
]}]


@pytest.mark.parametrize("capture,messages", [
    ("input_json", IMAGE_ONLY), ("initial_request", []), ("agent_request", IMAGE_ONLY),
], ids=["input-json-image", "initial-request-empty", "agent-request-image"])
def test_classifier_does_not_replace_nontext_input_with_stale_summaries(
    monkeypatch, messages, capture,
):
    from autofa.metrics.behavior import item_trace_for_judge

    calls = _fake_hub(monkeypatch)
    item = {"_item_id": "item-1", "system_text": "STALE_SUMMARY_SYSTEM",
            "prompt_text": "STALE_SUMMARY_TASK", "response_text": "Actual answer"}
    request = {"messages": messages}
    if capture == "input_json":
        item["input_json"] = request
    else:
        item["response_text"] = json.dumps({capture: request, "steps": [
            {"source": "assistant", "message": "Actual answer"},
        ]})

    direct = item_trace_for_judge(item)
    assert "Selected input has no readable text" in direct
    assert "STALE_SUMMARY" not in direct
    assert "Actual answer" in direct

    classify_improvement(fn=_fn(), cfg=Config(), items=[item], verdicts=YES)
    payload = json.loads(calls[0]["messages"][1]["content"])
    excerpt = payload["confirmed_examples"][0]["trace"]
    assert "Selected input has no readable text" in excerpt
    assert "STALE_SUMMARY" not in excerpt
    assert "Actual answer" in excerpt


def test_classifier_outage_is_explicit_and_nonfatal(monkeypatch):
    _fake_hub(monkeypatch, fail=True)
    result = classify_improvement(fn=_fn(), cfg=Config(),
                                  items=[{"_item_id": "item-1", "response_text": "trace"}],
                                  verdicts=YES)
    assert result["status"] == "unavailable"
    assert "TimeoutError" in result["rationale"]
    assert result["targets"] == []


def test_report_carries_assessment_without_changing_metrics_and_hides_stale_actions():
    from autofa.reports.failure_cards import _collect_example_ids, build_data, render_html

    fn = _fn()
    fn.scores = {"recall": 0.2, "behavior_purity": 0.8}
    library = FunctionLibrary(model_id="test", functions=[fn])
    before = build_data(library)["functions"][0]["metrics"]
    fn.scores["improvement"] = {**_reply(), "detector_fingerprint": detector_fingerprint(
        source=fn.source, description=fn.description, mechanism=fn.mechanism,
    )}
    data = build_data(library)
    assert data["functions"][0]["metrics"] == before
    assert data["functions"][0]["improvement"]["targets"] == ["model", "harness"]
    assert _collect_example_ids(data["functions"]) == {"item-1"}
    assert "Improvement target (LLM classification)" not in render_html(data)
    # The complete saved assessment is still available in the embedded report data.
    assert data["functions"][0]["improvement"]["validation_test"]
    fn.source += "# revision\n"
    stale = build_data(library)["functions"][0]["improvement"]
    assert stale["status"] == "stale"
    assert stale["harness_change"] == ""


def _judged_runtime(tmp_path, monkeypatch, cfg, contexts, **run_options):
    """One full-scan item that the judge confirms and the classifier cites."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autofa.config import RunConfig
    from autofa.data.parquet_store import Dataset
    from autofa.llm import routing
    from autofa.mcp import runtime, tools_eval
    from autofa.metrics import claude_judge

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    path = tmp_path / "logs.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "test", "benchmark": "bench", "problem_id": "item-1", "run_idx": 0,
         "is_correct": False, "response_text": "Wrong code argument and unclear error"},
    ]), path)
    run = RunConfig.create([path], "test", ["bench"], cfg, **run_options)
    fn = _fn()
    rt = runtime.Runtime(dataset=Dataset(parquet_paths=[path], model_id="test"),
                         library=FunctionLibrary(model_id="test", functions=[fn]),
                         run=run, config=cfg)
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)

    def judge(**kwargs):
        contexts.append(kwargs["analysis_guidance"])
        return {"behavior_purity": 1.0, "behavior_specificity": 0.8, "n_judged": 1,
                "n_yes": 1, "n_no": 0, "backend": "claude_agent_sdk",
                "verdicts": [{"item_id": kwargs["items"][0]["_item_id"], "verdict": "yes"}]}

    def classify(client, **kwargs):
        # Cite the actual canonical item identity instead of the fixture shorthand.
        payload = json.loads(kwargs["messages"][1]["content"])
        contexts.append(payload["skills_context"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            _reply(evidence_ids=[payload["confirmed_examples"][0]["item_id"]]),
        )))])

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    _fake_hub(monkeypatch)
    monkeypatch.setattr(routing, "routed_completion", classify)
    return rt, fn


def test_judge_persists_classification_and_full_rescore_retains_it(tmp_path, monkeypatch):
    from autofa.mcp import tools_eval

    rt, fn = _judged_runtime(tmp_path, monkeypatch, Config(analysis_focus="harness"), [])
    tools_eval.evaluate_functions(mode="full")
    before = {key: fn.scores[key] for key in ("recall", "ppv_at_pi0", "n_items")}
    result = tools_eval.judge_function(fn.name)
    assert result["improvement"]["status"] == "assessed"
    assert result["focus_hint"] is None
    assert FunctionLibrary.load(rt.working_library_path()).functions[0].scores["improvement"]
    tools_eval.evaluate_functions(mode="full")
    assert {key: fn.scores[key] for key in before} == before
    assert fn.scores["improvement"]["status"] == "assessed"
    fn.description = "Different claim"
    tools_eval.evaluate_functions(mode="full")
    assert fn.scores["improvement"]["status"] == "stale"


def test_judge_and_classifier_receive_only_the_matching_harness_pack(tmp_path, monkeypatch):
    from autofa.mcp import tools_eval

    skills = tmp_path / "skills"
    for pack in ("codex", "pi"):
        (skills / "harness" / pack).mkdir(parents=True)
        (skills / "harness" / pack / "tool-calling.md").write_text(f"{pack} pack contract")
    (skills / "harness" / "README.md").write_text("Pack index")
    contexts = []
    # "api" must not select the pi pack; "codex" selects its own.
    _, fn = _judged_runtime(
        tmp_path, monkeypatch, Config(analysis_focus="harness", skills_dir=str(skills)),
        contexts, source_harness="openai_api_codex_agent",
    )
    tools_eval.evaluate_functions(mode="full")
    assert tools_eval.judge_function(fn.name)["improvement"]["status"] == "assessed"
    assert len(contexts) == 2
    for context in contexts:
        assert "codex pack contract" in context
        assert "Pack index" in context
        assert "pi pack contract" not in context
