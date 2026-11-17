# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Decisions contract over the real SDK's HTTP transport."""

import base64
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from autofa.config import Config, load_config
from autofa.core.detector_lint import lint_detector_source
from autofa.core.localisation import CapturingLLM
from autofa.core.sandbox import DetectorAbstained, run_detector
from autofa.defaults import default_value
from autofa.llm.endpoints import endpoint_for_config
from autofa.llm.hub import HubLLM
from autofa.mcp import tools_eval
from autofa.mcp.runtime import Runtime
from autofa.session.diversity import infer_strategy
from autofa.session.llm_readiness import llm_readiness_guidance

from support.config import save_config


def choice(value="no", *, values=("yes", "no"), name=None):
    return {"type": "choice", "name": name, "choice": value, "confidence": 0.9,
            "probabilities": [{"value": v, "probability": 0.9 if v == value else 0.1}
                              for v in values]}


def client_for(handler):
    return OpenAI(api_key="test-key", base_url="https://provider.example/prefix/v1",
                  max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_existing_explicit_models_and_decisions_api_are_preserved(tmp_path, monkeypatch):
    path = tmp_path / "autofa.toml"
    monkeypatch.setenv("AUTOFA_CONFIG", str(path))
    path.write_text('feature_model = "gcp/google/gemini-3.5-flash-lite"\n')
    assert load_config().feature_model == "gcp/google/gemini-3.5-flash-lite"
    cfg = Config(feature_api="decisions")
    save_config(cfg)
    assert load_config() == cfg
    assert endpoint_for_config(cfg, "judge").api == "decisions"


def test_runtime_routes_fixed_labels_to_decisions_with_the_selected_endpoint(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"answers": [choice()]})

    with client_for(handler) as client:
        monkeypatch.setattr("autofa.llm.hub.make_hub_client", lambda **kw: client)
        cfg = Config(feature_api="decisions", feature_request_options={"max_tokens": 12})
        rt = Runtime.__new__(Runtime)
        rt.config, rt.hub_llm = cfg, None
        llm = rt.ensure_hub_llm()
        src = '''
def detect(item, llm_client=None):
    return llm_client.classify_options(item['response_text'], 'Did the test pass?') == 'yes'
'''
        assert run_detector(src, {"response_text": "The test failed."}, llm=llm) is False
    assert len(requests) == 1
    assert str(requests[0].url) == "https://provider.example/prefix/v1/decisions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content) == {
        "model": cfg.feature_model, "input": "The test failed.",
        "questions": [{"type": "choice", "instructions": "Did the test pass?",
                       "choices": [{"value": "yes"}, {"value": "no"}]}],
    }


def test_probe_checks_decisions_and_responses_while_localisation_uses_responses():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/decisions"):
            return httpx.Response(200, json={"answers": [choice("OK", values=("OK", "FAILED"))]})
        return httpx.Response(200, json={"id": "r", "model": "m", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "OK", "annotations": []}]}]})

    with client_for(handler) as client:
        llm = HubLLM(client, api="decisions")
        assert llm.probe().ok
        assert llm.localise("Extract a quote", "evidence") == "OK"
    assert paths == ["/prefix/v1/decisions", "/prefix/v1/responses", "/prefix/v1/responses"]


def test_bulk_decision_scoring_generates_quotes_only_when_requested():
    from autofa.core.localisation import resolve_deferred_localisation

    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/decisions"):
            body = json.loads(request.content)
            return httpx.Response(200, json={"answers": [
                choice("yes" if "guessed" in body["input"] else "no")
            ]})
        assert request.url.path.endswith("/responses")
        return httpx.Response(200, json={"id": "highlight", "model": "m", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps({"quotes": ["I guessed."]}),
                 "annotations": []}]}]})

    source = '''
def detect(item, llm_client=None):
    return llm_client.classify_options(item['response_text'], 'Was the answer guessed?') == 'yes'
'''
    positive = {"_item_id": "positive", "response_text": "I guessed. There was no evidence."}
    with client_for(handler) as client:
        llm = HubLLM(client, api="auto", model="deferred-quote-test")
        pending = {}
        assert run_detector(source, positive, llm=llm, localisation=pending) is True
        negative = {}
        assert run_detector(source, {"_item_id": "negative", "response_text": "I checked."},
                            llm=llm, localisation=negative) is False
        assert paths == ["/prefix/v1/decisions"] * 2
        assert not negative
        completed = resolve_deferred_localisation(pending, positive, source, llm)
        assert paths == ["/prefix/v1/decisions"] * 2 + ["/prefix/v1/responses"]
        assert completed["passages"][0]["quote"] == "I guessed."
        assert resolve_deferred_localisation(completed, positive, source, llm) == completed
        assert len(paths) == 3


def test_multimodal_decisions_preserve_verified_bytes_and_evidence_window():
    bodies = []
    data = b"synthetic-image"
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    receipt = {"status": "available", "data_url": url, "sha256": hashlib.sha256(data).hexdigest()}

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        values = tuple(c["value"] for c in body["questions"][0]["choices"])
        return httpx.Response(200, json={"answers": [choice(values=values)]})

    with client_for(handler) as client:
        llm = HubLLM(client, api="decisions")
        text = "HEAD " + "context " * 100 + "TAIL"
        assert llm.classify_options(text, "Check", focus="tail", max_chars=64,
                                    images=[receipt]) == "no"
        with pytest.raises(ValueError, match="checksum"):
            llm.classify_options("evidence", "Check", images=[{**receipt, "sha256": "wrong"}])
    assert len(bodies) == 1
    content = bodies[0]["input"][0]["content"]
    assert bodies[0]["input"][0]["role"] == "user"
    assert content[0]["type"] == "input_text"
    assert "analyzer window metadata" in content[0]["text"]
    fragments = json.loads(content[0]["text"].split("--- exact source fragments ---\n")[1])
    assert sum(len(p["text"]) for p in fragments) <= 64
    assert fragments[-1]["text"].endswith("TAIL")
    assert all("HEAD" not in p["text"] for p in fragments)
    assert content[1] == {"type": "input_image", "image_url": url}


def test_mixed_questions_preserve_order_typed_choices_scores_and_refusals():
    questions = [
        {"type": "predicate", "name": "test", "instructions": "The test passed."},
        {"type": "choice", "instructions": "Select.",
         "choices": [{"value": True}, {"value": "true"}]},
        {"type": "score", "instructions": "Rate.",
         "levels": [{"label": "bad"}, {"label": "good", "description": "All correct."}]},
        {"type": "predicate", "instructions": "Check."},
    ]
    answers = [{"type": "predicate", "name": "test", "probability": 0.75},
               choice(True, values=(True, "true")),
               {"type": "score", "score": 0.8, "confidence": 0.8, "probabilities": [
                   {"label": "bad", "value": 0, "probability": 0.2},
                   {"label": "good", "value": 1, "probability": 0.8}]},
               {"type": "refusal", "name": None}]
    with client_for(lambda r: httpx.Response(200, json={"answers": answers})) as client:
        assert HubLLM(client).decide(questions, "evidence") == answers


@pytest.mark.parametrize("answer", [
    {"type": "refusal", "name": None},
    {**choice(), "choice": "maybe"},
    {**choice(), "name": "another question"}, {**choice(), "confidence": 1.1},
    {**choice(), "probabilities": []},
    {**choice(), "probabilities": [{"value": "yes", "probability": 1},
                                  {"value": "yes", "probability": 0}]},
    {"type": "predicate", "probability": 0.1},
])
def test_refused_or_malformed_decision_cannot_become_a_negative_label(answer):
    with client_for(lambda r: httpx.Response(200, json={"answers": [answer]})) as client:
        expected = DetectorAbstained if answer.get("type") == "refusal" else RuntimeError
        with pytest.raises(expected):
            HubLLM(client, api="decisions").classify_options("evidence", "Check")


# 404 is an unsupported route that auto mode would fall back from; 429 is an outage.
@pytest.mark.parametrize("status", [404, 429])
def test_explicit_decisions_unavailability_never_falls_back(status):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(status, json={"error": {"message": "Unavailable"}})

    with client_for(handler) as client:
        probe = HubLLM(client, api="decisions").probe()
        assert not probe.ok
        if status == 404:
            assert "Decisions API unavailable" in probe.error
    assert paths == ["/prefix/v1/decisions"]


@pytest.mark.parametrize("questions", [[], [{"type": "unknown", "instructions": "Check"}],
    [{"type": "choice", "instructions": "Check", "choices": [{"value": "yes"}, {"value": "yes"}]}],
    [{"type": "predicate", "instructions": "Check", "model": "override"}],
])
def test_invalid_questions_fail_before_http(questions):
    def unexpected(request):
        pytest.fail("Invalid questions should not be sent")

    with client_for(unexpected) as client, pytest.raises(ValueError):
        HubLLM(client).decide(questions, "evidence")


def test_decision_calls_reach_passage_capture():
    calls = []
    requests = []
    questions = [{"type": "predicate", "instructions": "The test failed."}]
    answers = [{"type": "predicate", "probability": 0.9}]
    image_data = b"synthetic-image"
    images = [{"status": "available",
               "data_url": "data:image/png;base64," + base64.b64encode(image_data).decode(),
               "sha256": hashlib.sha256(image_data).hexdigest()}]
    window = {"focus": "full", "max_chars": 500, "images": images}

    def decide(*args, **kwargs):
        requests.append((args, kwargs))
        return deepcopy(answers)

    client = SimpleNamespace(decide=decide)
    capture = SimpleNamespace(llm_call=lambda *a: calls.append(a))
    wrapped = CapturingLLM(client, capture)
    assert wrapped.decide(questions, "The test failed.", **window) == answers
    assert requests == [((questions, "The test failed."), window)]
    assert calls == [(json.dumps(questions), "The test failed.", json.dumps(answers),
                      window)]


def test_mcp_prompt_with_options_uses_selected_label_api(monkeypatch):
    calls = []
    llm = SimpleNamespace(classify_options=lambda *a, **k: calls.append((a, k)) or "no")
    monkeypatch.setattr(tools_eval, "get_runtime",
                        lambda: SimpleNamespace(ensure_hub_llm=lambda: llm))
    result = tools_eval.llm_classify("evidence", prompt="Check", options=["yes", "no"])
    assert result["ok"] and result["label"] == "no"
    assert calls == [(("evidence", "Check"), {"options": ["yes", "no"], "focus": "auto"})]


@pytest.mark.parametrize("call", [
    "llm.classify_options(text, 'Apply rubric', options=['yes', 'no']) == 'yes'",
    "llm.decide([{'type': 'predicate', 'instructions': 'Apply rubric'}], "
    "text)[0]['probability'] > 0.8",
])
def test_semantic_helpers_are_recognized_for_lint_sampling_and_strategy(call):
    source = f"def detect(item, llm=None):\n    text = item['response_text']\n    return {call}"
    fn = SimpleNamespace(source=source, kind="regex", metadata={})
    assert tools_eval.is_llm_detector(fn)
    assert infer_strategy(source, kind="regex") == "llm"
    assert lint_detector_source(source)[0]
    # The same evidence-window lint applies to both supported label APIs.
    sliced = source.replace("text = item['response_text']", "text = item['response_text'][-4000:]")
    assert not lint_detector_source(sliced)[0]


@pytest.mark.parametrize("api", ["chat_completions", "decisions"])
def test_agent_knows_active_api_and_semantic_detector_availability(api):
    guidance = llm_readiness_guidance({
        "api": api, "status": "ready", "model": default_value("feature_model"),
        "message": "Ready", "available": True,
    })
    assert f"label_api={api}" in guidance
    assert "Semantic LLM detectors are available" in guidance


def test_dataset_info_exposes_the_selected_label_api(monkeypatch):
    from autofa.mcp import tools_data

    # Use the real dataset_info coverage path with an empty dataset fixture.
    class Dataset:
        def info(self, **kwargs):
            return {}

        def estimate_matching_rows(self):
            return 0

        def iter_batches(self, **kwargs):
            return iter([])

    rt = SimpleNamespace(config=Config(feature_api="decisions"), dataset=Dataset(),
                         run=SimpleNamespace(eval_limit=100),
                         llm_readiness={"status": "ready", "configured": True, "available": True})
    monkeypatch.setattr(tools_data, "get_runtime", lambda: rt)
    result = tools_data.dataset_info()
    assert result["llm_detectors"]["api"] == "decisions"
    assert "decide" in result["llm_detectors"]["methods"]
