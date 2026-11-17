# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM endpoints route, fall back and isolate credentials; API consumers keep request policy."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest
from openai import APIStatusError, APITimeoutError, OpenAI

from autofa.config import Config, load_config
from autofa.core.localisation import CapturingLLM
from autofa.llm.capabilities import APIUnavailable, cached_capability
from autofa.llm.endpoints import (
    completion_options,
    endpoint_for_config,
    improvement_endpoint_for_config,
)
from autofa.llm.hub import HubLLM, chat_completions_safe, completion_text, make_hub_client
from autofa.llm.routing import routed_completion
from autofa.mcp.runtime import Runtime
from autofa.metrics.improvements import classify_improvement
from autofa.preflight.cheating import detector as cheating
from autofa.preflight.cheating import pipeline as cheating_pipeline
from autofa.taxonomy.providers import OpenAIProvider, ProviderError
from autofa.util.cancellation import Cancellation, cancellation_scope
from autofa.util.model_budget import model_call_budget, reserve_model_call

from support.config import save_config
from support.judge import cheating_review
from support.repair_review import _fn, _reply

try:
    import httpx
except ImportError:
    # OpenAI SDK 3.x uses httpx2 and also accepts a legacy httpx client.
    import httpx2 as httpx


def response(text="yes", *, finish_reason="stop"):
    return {"id": "test", "object": "chat.completion", "created": 0, "model": "any/model",
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": text}}]}


def reply(body):
    """Answer "yes" as a completion, or as an event stream when the request streams."""
    if not body.get("stream"):
        return httpx.Response(200, json=response())
    envelope = {"id": "synthetic", "created": 0, "model": "arbitrary"}
    chunks = [
        {**envelope, "object": "chat.completion.chunk", "choices": [{
            "index": 0, "delta": {"role": "assistant", "content": "yes"},
            "finish_reason": None,
        }]},
        {**envelope, "object": "chat.completion.chunk", "choices": [{
            "index": 0, "delta": {}, "finish_reason": "stop",
        }]},
    ]
    content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          content=content + "data: [DONE]\n\n")


def mock_openai(handler, **kwargs):
    kwargs.setdefault("api_key", "test")
    return OpenAI(max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                  **kwargs)


def client_for(handler, *, key="key-one", endpoint="https://provider.example/prefix/v1"):
    return mock_openai(handler, api_key=key, base_url=endpoint)


def generated(text="yes", **overrides):
    """A completed Responses API answer."""
    return {"id": "resp_test", "model": "model", "status": "completed", "created_at": 1,
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text,
                                     "annotations": []}]}], **overrides}


def decision(value="yes", values=("yes", "no")):
    """A Decisions API choice with its probabilities."""
    return {"answers": [{"type": "choice", "name": None, "choice": value,
                         "confidence": .9, "probabilities": [
                             {"value": item, "probability": .9 if item == value else .1}
                             for item in values]}]}


def recording(requests, answer=reply):
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return answer(body)

    return handler


def intercept_openai(monkeypatch):
    """Let make_hub_client build real SDK clients that send to a recording transport."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response())

    real_openai = OpenAI
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: real_openai(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ))
    return requests


def unsupported(param, alternative="max_completion_tokens"):
    return httpx.Response(400, json={"error": {
        "param": param, "code": "unsupported_parameter",
        "message": f"Unsupported {param}; use {alternative}.",
    }})


def adapting(body):
    """Reject temperature, then max_tokens, then answer."""
    for param in ("temperature", "max_tokens"):
        if param in body:
            return unsupported(param)
    return reply(body)


def oscillating(body):
    """Reject each token parameter in favour of the other."""
    param = "max_tokens" if "max_tokens" in body else "max_completion_tokens"
    return unsupported(param, "max_completion_tokens" if param == "max_tokens" else "max_tokens")


def capturing(llm, captures=None):
    record = (lambda *args: captures.append(args)) if captures is not None else lambda *args: None
    return CapturingLLM(llm, SimpleNamespace(llm_call=record))


SURFACES = [
    "chat.completions.parse",
    "chat.completions.with_raw_response.create",
    "chat.completions.with_raw_response.parse",
    "chat.with_raw_response.completions.create",
    "chat.with_raw_response.completions.parse",
    "chat.completions.with_streaming_response.create",
    "chat.completions.with_streaming_response.parse",
    "chat.with_streaming_response.completions.create",
    "chat.with_streaming_response.completions.parse",
    "chat.completions.stream",
]


# One immediate raw wrapper, one deferred streaming wrapper and the stream() context manager.
REPRESENTATIVE_SURFACES = [
    "chat.completions.with_raw_response.create",
    "chat.completions.with_streaming_response.parse",
    "chat.completions.stream",
]


def _method(client, surface):
    for part in surface.split("."):
        client = getattr(client, part)
    return client


def _deferred(surface):
    return "with_streaming_response" in surface or surface.endswith(".stream")


def _consume(result, surface):
    if surface.endswith(".stream"):
        with result as stream:
            return stream.get_final_completion()
    if "with_streaming_response" in surface:
        with result as raw:
            return raw.parse()
    return result.parse() if "with_raw_response" in surface else result


def test_legacy_defaults_and_independent_judge():
    cfg = Config()
    feature = endpoint_for_config(cfg)
    assert feature.base_url == cfg.hub_base_url
    assert feature.api_key_env == cfg.hub_api_key_env
    assert endpoint_for_config(cfg, "judge").base_url == feature.base_url
    cfg.feature_base_url = "https://arbitrary-gateway.example/custom/v1/"
    cfg.feature_api_key_env = "GATEWAY_KEY"
    cfg.feature_model = "any/model-ID"
    cfg.judge_model = "another-model"
    assert endpoint_for_config(cfg, "judge").api_key_env == "GATEWAY_KEY"
    cfg.judge_base_url = "https://second-gateway.example/api/v1"
    cfg.judge_api_key_env = "CLASSIFIER_KEY"
    assert endpoint_for_config(cfg).model == "any/model-ID"
    assert endpoint_for_config(cfg, "judge").model == "another-model"
    assert endpoint_for_config(cfg, "judge").api_key_env == "CLASSIFIER_KEY"


def test_explicit_judge_endpoint_is_independent_of_invalid_detector_settings():
    cfg = Config(feature_base_url="unfinished", feature_api_key_env="",
                 judge_base_url="https://classifier.example/v1", judge_api_key_env="KEY")
    assert endpoint_for_config(cfg, "judge").base_url == "https://classifier.example/v1"
    with pytest.raises(ValueError):
        endpoint_for_config(cfg)


def test_workflow_api_override_preserves_endpoint_model_key_and_options():
    cfg = Config(feature_api="decisions", feature_base_url="https://feature.example/v1",
                 feature_api_key_env="FEATURE_KEY", judge_base_url="https://judge.example/v1",
                 judge_api_key_env="JUDGE_KEY", feature_model="detector", judge_model="judge",
                 judge_request_options={"temperature": 0.5})
    inherited = endpoint_for_config(cfg, "judge")
    override = endpoint_for_config(cfg, "judge", api="chat_completions")
    assert override.api == "chat_completions"
    assert (override.base_url, override.api_key_env, override.model, override.request_options) == (
        inherited.base_url, inherited.api_key_env, inherited.model, inherited.request_options,
    )
    assert cfg.feature_api == "decisions"


def test_explicit_api_does_not_validate_an_unrelated_discovery_preference():
    cfg = Config(feature_api="invalid-discovery-setting")
    assert endpoint_for_config(cfg, "judge", api="chat_completions").api == "chat_completions"
    with pytest.raises(ValueError):
        endpoint_for_config(cfg)
    with pytest.raises(ValueError):
        endpoint_for_config(Config(), api="invalid-workflow-setting")


@pytest.mark.parametrize("options", [
    {"reasoning_effort": "high"}, {"extra_body": {"reasoning": {"enabled": True}}},
])
def test_explicit_shared_reasoning_controls_override_improvement_defaults(options):
    cfg = Config(judge_request_options=deepcopy(options))
    assert improvement_endpoint_for_config(cfg).request_options == options
    assert endpoint_for_config(cfg, "judge").request_options == options
    assert cfg.judge_request_options == options
    assert cfg.improvement_request_options == {"reasoning_effort": "none"}
    assert endpoint_for_config(cfg).request_options == {"reasoning_effort": "none"}


@pytest.mark.parametrize("factory", [dict, SimpleNamespace])
def test_sparse_configs_use_discovery_default_unless_explicit(factory):
    assert improvement_endpoint_for_config(factory(judge_request_options={})).request_options == {
        "reasoning_effort": "none",
    }
    explicit = factory(improvement_request_options={})
    assert improvement_endpoint_for_config(explicit).request_options == {}


def test_improvement_merge_retains_budget_and_sampling_controls_for_dispatch():
    cfg = Config(improvement_request_options={
        "reasoning_effort": "none", "max_tokens": 200, "temperature": 0.5,
        "extra_body": {"routing": {"order": ["route-a"]}},
    }, judge_request_options={"max_output_tokens": 400, "omit_parameters": ["temperature"]})
    endpoint = improvement_endpoint_for_config(cfg)
    assert endpoint.request_options == {
        "reasoning_effort": "none", "max_output_tokens": 400,
        "omit_parameters": ["temperature"], "extra_body": {"routing": {"order": ["route-a"]}},
    }
    actual = completion_options({"temperature": 0.2, "max_completion_tokens": 100},
                                endpoint.request_options)
    assert "temperature" not in actual and "max_completion_tokens" not in actual
    assert actual["max_output_tokens"] == 400
    endpoint.request_options["extra_body"]["routing"]["order"].append("route-b")
    assert cfg.improvement_request_options["extra_body"]["routing"]["order"] == ["route-a"]


def test_explicit_sampling_option_overrides_discovery_omission():
    cfg = Config(improvement_request_options={"omit_parameters": ["temperature", "top_p"]},
                 judge_request_options={"temperature": 0.5})
    endpoint = improvement_endpoint_for_config(cfg)
    actual = completion_options({"temperature": 0.2, "top_p": 0.9}, endpoint.request_options)
    assert actual == {"temperature": 0.5}


@pytest.mark.parametrize("options", [None, {"model": "override"}])
def test_invalid_improvement_options_are_rejected_before_client_construction(options):
    with pytest.raises(ValueError):
        improvement_endpoint_for_config(Config(improvement_request_options=options))


@pytest.mark.parametrize("role", ["feature", "judge"])
def test_changing_endpoint_requires_explicit_credential_selection(role):
    cfg = Config()
    setattr(cfg, f"{role}_base_url", "https://another.example/v1")
    with pytest.raises(ValueError, match=f"Set {role}_api_key_env explicitly"):
        endpoint_for_config(cfg, role)


def test_custom_key_never_falls_back_to_existing_legacy_key(monkeypatch):
    monkeypatch.setenv("INFERENCE_HUB_API_KEY", "secret-hub-key")
    monkeypatch.setenv("INFERENCE_API_KEY", "secret-alias-key")
    monkeypatch.delenv("CUSTOM_PROVIDER_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CUSTOM_PROVIDER_KEY not set") as error:
        make_hub_client(base_url="https://another.example/v1", api_key_env="CUSTOM_PROVIDER_KEY")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("url", ["file:///tmp/api", "https://user:secret@api.example/v1",
                                 "https://api.example/v1?key=secret",
                                 "https://api.example/v1/chat/completions"])
def test_invalid_endpoint_fails_before_loading_credentials(url):
    with pytest.raises(ValueError, match="API root"):
        make_hub_client(base_url=url)


def test_configuration_round_trip_preserves_nested_request_options(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "autofa.toml"))
    cfg = Config(feature_base_url="https://gateway.example/v1", feature_api_key_env="KEY",
                 judge_base_url="https://classifier.example/v1", judge_api_key_env="JUDGE_KEY",
                 feature_model="arbitrary/model", judge_model="different-model")
    cfg.feature_request_options = {
        "max_completion_tokens": 8192, "omit_parameters": ["temperature"],
        "extra_body": {"routing": {"order": ["route-a", "route-b"]}},
    }
    cfg.judge_request_options = {"temperature": 0.5, "max_tokens": 4096}
    save_config(cfg)
    loaded = load_config()
    assert endpoint_for_config(loaded) == endpoint_for_config(cfg)
    assert endpoint_for_config(loaded, "judge") == endpoint_for_config(cfg, "judge")


@pytest.mark.parametrize("options", [
    {"messages": []}, {"api_key": "secret"},
    {"max_tokens": 5, "max_completion_tokens": 10}, {"max_tokens": -1},
    {"omit_parameters": ["messages"]}, {"extra_body": {"messages": []}},
])
def test_options_cannot_replace_evidence_or_invalid_budgets(options):
    with pytest.raises(ValueError):
        completion_options({"max_tokens": 64}, options)


def test_chat_protocol_routes_arbitrary_model_and_parameters(monkeypatch):
    requests = intercept_openai(monkeypatch)
    monkeypatch.setenv("PROVIDER_KEY", "provider-secret")
    with make_hub_client(base_url="https://unregistered-provider.example/prefix/v1",
                         api_key_env="PROVIDER_KEY") as client:
        llm = HubLLM(client, api="chat_completions", model="any/model-or-deployment",
                     request_options={
            "max_completion_tokens": 8192, "omit_parameters": ["temperature"],
            "extra_body": {"custom_option": "passed-through"},
        })
        assert llm.classify("Classify", "evidence") == "yes"
    request = requests[0]
    assert str(request.url) == "https://unregistered-provider.example/prefix/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer provider-secret"
    payload = json.loads(request.content)
    assert payload["model"] == "any/model-or-deployment"
    assert payload["max_completion_tokens"] == 8192
    assert payload["custom_option"] == "passed-through"
    assert "temperature" not in payload and "max_tokens" not in payload


def test_conflicting_token_parameter_rejections_cannot_loop():
    calls = []
    with mock_openai(recording(calls, oscillating)) as client:
        with pytest.raises(APIStatusError):
            chat_completions_safe(client, model="m", messages=[], max_completion_tokens=4096)
    assert len(calls) == 3
    assert calls[1]["max_tokens"] == 4096


@pytest.mark.parametrize("status,param,code,message", [
    (401, "temperature", "unsupported_parameter", "Unsupported"),
    (400, "temperature", "unsupported_value", "Value out of range"),
    (400, "model", "model_not_found", "Model not found"),
])
def test_other_failures_are_not_retried_or_reinterpreted(status, param, code, message):
    calls = []
    with mock_openai(recording(calls, lambda body: httpx.Response(status, json={"error": {
        "param": param, "code": code, "message": message,
    }}))) as client:
        with pytest.raises(APIStatusError):
            chat_completions_safe(client, model="m", messages=[], temperature=0)
    assert len(calls) == 1


def test_cache_is_isolated_by_endpoint_client_and_request_parameters():
    calls = []

    def client(label, url):
        def complete(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=label),
            )])
        return SimpleNamespace(base_url=url, chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        ))

    first = HubLLM(client("yes", "https://one.example/v1"),
                   api="chat_completions", model="same-model")
    second = HubLLM(client("no", "https://two.example/v1"),
                    api="chat_completions", model="same-model")
    assert first.classify("question", "same evidence") == "yes"
    assert second.classify("question", "same evidence") == "no"
    assert first.classify("question", "same evidence") == "yes"
    assert len(calls) == 2
    first.request_options = {"temperature": 0.5}
    assert first.classify("question", "same evidence") == "yes"
    assert len(calls) == 3
    # Different clients/accounts on the same URL must not share previous results.
    third = HubLLM(client("no", "https://one.example/v1"),
                   api="chat_completions", model="same-model")
    assert third.classify("question", "same evidence") == "no"
    assert len(calls) == 4


@pytest.mark.parametrize("content,finish", [(None, "stop"), ("yes", "length")])
def test_missing_or_partial_completion_cannot_become_a_label(content, finish):
    with mock_openai(lambda request: httpx.Response(
        200, json=response(content, finish_reason=finish),
    )) as client:
        with pytest.raises(RuntimeError):
            HubLLM(client, api="chat_completions").classify_options("text", "question")


def test_detector_and_improvement_classifier_use_separate_endpoints(monkeypatch):
    routes, requests = [], []

    class Client:
        def __init__(self, base_url):
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def with_options(self, **kwargs):
            return self

        def complete(self, **kwargs):
            requests.append((self.base_url, kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"status":"insufficient_evidence", "rationale":"Test"}',
            ))])

    def make(**kwargs):
        routes.append(kwargs)
        return Client(kwargs["base_url"])

    monkeypatch.setattr("autofa.llm.hub.make_hub_client", make)
    cfg = Config(feature_base_url="https://first.example/v1", feature_api_key_env="FIRST_KEY",
                 judge_base_url="https://second.example/v1", judge_api_key_env="SECOND_KEY",
                 feature_model="model-a", judge_model="model-b",
                 judge_request_options={"max_completion_tokens": 6000})
    runtime = Runtime(dataset=None, library=None, run=None, config=cfg)
    llm = runtime.ensure_hub_llm()
    assert llm.model == "model-a"
    fn = SimpleNamespace(source="def check(item): return True", name="check",
                         description="claim", mechanism="mechanism")
    result = classify_improvement(fn=fn, cfg=cfg,
                                 items=[{"_item_id": "1", "response_text": "evidence"}],
                                 verdicts=[{"item_id": "1", "verdict": "yes"}])
    assert [r["api_key_env"] for r in routes] == ["FIRST_KEY", "SECOND_KEY"]
    assert requests[0][0] == "https://second.example/v1"
    assert requests[0][1]["model"] == "model-b"
    assert requests[0][1]["max_completion_tokens"] == 6000
    assert result["classifier_base_url"] == "https://second.example/v1"
    assert result["backend"] == "openai_compatible"


def test_local_endpoint_never_sends_environment_credentials_and_supports_both_roles(monkeypatch):
    requests = intercept_openai(monkeypatch)
    monkeypatch.setenv("INFERENCE_HUB_API_KEY", "legacy-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "another-secret")
    monkeypatch.setenv("none", "not-a-real-key")
    # An explicit feature endpoint does not require (or validate) legacy hub settings.
    cfg = Config(hub_base_url="unused", feature_base_url="http://localhost:8000/v1",
                 feature_api_key_env="none", feature_model="served-model",
                 judge_model="classifier-model")
    for role in ("feature", "judge"):
        endpoint = endpoint_for_config(cfg, role)
        with make_hub_client(base_url=endpoint.base_url,
                             api_key_env=endpoint.api_key_env) as client:
            llm = HubLLM(client, api="chat_completions", model=endpoint.model)
            assert llm.classify("question", "trace") == "yes"
    assert len(requests) == 2
    assert all(req.headers["authorization"] == "Bearer unused" for req in requests)
    assert all(str(req.url) == "http://localhost:8000/v1/chat/completions" for req in requests)
    assert [json.loads(req.content)["model"] for req in requests] == [
        "served-model", "classifier-model",
    ]


# The two spellings of "reasoning off" alternate across the detector operations.
EFFORT_OFF = {"reasoning_effort": "none"}
REASONING_DISABLED = {"extra_body": {"reasoning": {"enabled": False}}}


@pytest.mark.parametrize("operation,options", [
    ("probe", EFFORT_OFF), ("classify", REASONING_DISABLED), ("classify_options", EFFORT_OFF),
    ("localise", REASONING_DISABLED), ("raw", EFFORT_OFF), ("safe_raw", REASONING_DISABLED),
    ("captured_raw", EFFORT_OFF),
])
def test_explicit_reasoning_off_reaches_every_detector_chat_surface(operation, options):
    requests = []
    configured = {**options, "max_completion_tokens": 8192, "omit_parameters": ["temperature"]}
    with mock_openai(recording(requests), base_url="https://provider.example/v1") as client:
        llm = HubLLM(client, api="chat_completions", model="arbitrary-detector-model",
                     request_options=configured)
        if operation == "probe":
            assert llm.probe().ok
        elif operation in {"raw", "safe_raw", "captured_raw"}:
            messages = [{"role": "user", "content": "Full original evidence"}]
            raw_options = {
                "model": "generated-model", "messages": messages,
                "reasoning_effort": "high",
                "extra_body": {"reasoning_effort": "high",
                               "reasoning": {"enabled": True, "effort": "high"},
                               "routing": {"order": ["route-a"]}},
            }
            if operation == "safe_raw":
                answer = chat_completions_safe(llm, **raw_options)
            else:
                exposed = capturing(llm) if operation == "captured_raw" else llm
                answer = exposed.chat.completions.create(**raw_options)
            assert completion_text(answer) == "yes"
            assert requests[0]["messages"] == messages
            assert requests[0]["model"] == "generated-model"
            assert requests[0]["routing"] == {"order": ["route-a"]}
        else:
            assert getattr(llm, operation)("question", "evidence") == "yes"
    assert len(requests) == 1
    assert requests[0]["max_completion_tokens"] == 8192
    assert "temperature" not in requests[0] and "max_tokens" not in requests[0]
    if "reasoning_effort" in options:
        assert requests[0]["reasoning_effort"] == "none"
        assert "reasoning" not in requests[0]
    else:
        assert requests[0]["reasoning"] == {"enabled": False}
        assert "reasoning_effort" not in requests[0]


def test_raw_request_configuration_preserves_evidence_and_nested_unrelated_fields():
    supplied = {
        "model": "source-model", "messages": [{"role": "user", "content": "evidence"}],
        "max_tokens": 17, "temperature": .7, "reasoning_effort": "high",
        "extra_body": {
            "reasoning": {"enabled": True, "effort": "high"},
            "routing": {"order": ["a", "b"], "allow_fallbacks": True},
            "chat_template_kwargs": {"enable_thinking": True, "format": "plain"},
        },
    }
    configured = {
        "max_completion_tokens": 128, "reasoning_effort": "none",
        "omit_parameters": ["temperature"],
        "extra_body": {"routing": {"allow_fallbacks": False},
                       "chat_template_kwargs": {"enable_thinking": False}},
    }
    before = deepcopy((supplied, configured))
    merged = completion_options(supplied, configured)
    assert merged["model"] == "source-model"
    assert merged["messages"] is supplied["messages"]
    assert merged["max_completion_tokens"] == 128
    assert "max_tokens" not in merged and "temperature" not in merged
    assert merged["reasoning_effort"] == "none"
    assert merged["extra_body"] == {
        "routing": {"order": ["a", "b"], "allow_fallbacks": False},
        "chat_template_kwargs": {"enable_thinking": False, "format": "plain"},
    }
    merged["extra_body"]["routing"]["order"].append("c")
    assert (supplied, configured) == before


def test_raw_completion_wrapper_reads_current_options_and_preserves_explicit_empty_options():
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs),
    )))
    llm = HubLLM(client, api="chat_completions", request_options={"reasoning_effort": "none"})
    completions = llm.chat.completions
    completions.create(model="custom", messages=[], reasoning_effort="high")
    llm.request_options = {"reasoning_effort": "low"}
    completions.create(model="custom", messages=[], reasoning_effort="high")
    llm.request_options = {}
    completions.create(model="custom", messages=[], reasoning_effort="high")
    assert [call["reasoning_effort"] for call in calls] == ["none", "low", "high"]


def test_reasoning_off_rejection_is_not_retried_without_the_control():
    calls = []
    with mock_openai(recording(calls, lambda body: httpx.Response(400, json={"error": {
        "param": "reasoning_effort", "code": "unsupported_value",
        "message": "Reasoning effort none is not supported by this model.",
    }}))) as client:
        llm = HubLLM(client, api="chat_completions", request_options={"reasoning_effort": "none"})
        with pytest.raises(APIStatusError):
            chat_completions_safe(llm, model="reasoning-required", messages=[])
    assert len(calls) == 1 and calls[0]["reasoning_effort"] == "none"


def test_excluding_reasoning_output_does_not_disable_reasoning_generation():
    merged = completion_options(
        {"reasoning_effort": "high", "extra_body": {"reasoning": {"enabled": True}}},
        {"extra_body": {"reasoning": {"exclude": True}}},
    )
    assert merged["reasoning_effort"] == "high"
    assert merged["extra_body"]["reasoning"] == {"enabled": True, "exclude": True}


@pytest.mark.parametrize("surface", [
    "raw", "safe_raw", "captured_safe_raw", *REPRESENTATIVE_SURFACES,
])
def test_compatibility_adapts_configured_parameters_once_at_the_request_boundary(surface):
    calls, captures = [], []
    # Unknown provider and a model name supports_custom_temperature() flags: parameters
    # are kept until the endpoint rejects them, never stripped by provider or model name.
    with mock_openai(recording(calls, adapting), base_url="https://unknown.example/v1") as client:
        llm = HubLLM(client, api="chat_completions", request_options={
            "reasoning_effort": "none", "temperature": 0, "max_tokens": 128,
        })
        exposed = capturing(llm, captures) if surface == "captured_safe_raw" else llm
        arguments = {
            "model": "openai/openai/gpt-5.5", "messages": [{"role": "user", "content": "trace"}],
            "temperature": .7, "max_tokens": 64, "extra_body": {
                "temperature": 1.2, "max_completion_tokens": 8192, "routing": {"mode": "keep"},
            },
        }
        if surface == "raw":
            result = exposed.chat.completions.create(**arguments)
        elif surface in REPRESENTATIVE_SURFACES:
            result = _consume(_method(exposed, surface)(**arguments), surface)
        else:
            result = chat_completions_safe(exposed, **arguments)
    assert completion_text(result) == "yes"
    assert len(calls) == 3
    assert calls[0]["temperature"] == 0
    assert [call.get("max_tokens") for call in calls] == [128, 128, None]
    assert calls[-1]["max_completion_tokens"] == 128
    assert "temperature" not in calls[1] and "temperature" not in calls[2]
    assert all(call["reasoning_effort"] == "none" for call in calls)
    assert all((call["model"], call["messages"]) == (arguments["model"], arguments["messages"])
               for call in calls)
    assert len(captures) == (1 if surface == "captured_safe_raw" else 0)
    assert all(call["routing"] == {"mode": "keep"} for call in calls)


def test_nested_safe_call_does_not_restart_exhausted_compatibility_retries():
    calls = []
    with mock_openai(recording(calls, oscillating)) as client:
        llm = HubLLM(client, api="chat_completions",
                     request_options={"max_tokens": 128, "reasoning_effort": "none"})
        with pytest.raises(APIStatusError):
            chat_completions_safe(capturing(llm), model="arbitrary", messages=[])
    assert len(calls) == 3
    assert all(call["reasoning_effort"] == "none" for call in calls)


@pytest.mark.parametrize("surface", [
    "chat.completions.stream", "chat.completions.with_raw_response.create",
])
def test_auto_legacy_sdk_surfaces_keep_dispatch_budget(surface):
    calls, reservations = [], []

    def handler(request):
        calls.append(request.url.path)
        return reply(json.loads(request.content))

    with mock_openai(handler) as raw, model_call_budget(lambda: reservations.append(1)):
        llm = HubLLM(raw, api="auto")
        result = _consume(_method(llm, surface)(model="arbitrary", messages=[]), surface)
        assert completion_text(result) == "yes"
    assert calls == ["/v1/chat/completions"] and reservations == [1]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("api,captured,reasoning", [
    ("chat_completions", False, EFFORT_OFF), ("auto", True, REASONING_DISABLED),
], ids=["chat-effort", "auto-captured-reasoning"])
def test_alternate_chat_surfaces_enforce_options_on_actual_requests(
    surface, api, captured, reasoning,
):
    requests = []
    configured = {**reasoning, "max_completion_tokens": 128,
                  "omit_parameters": ["temperature", "top_p"]}
    supplied = {
        "model": "generated-model", "messages": [{"role": "user", "content": "Full evidence"}],
        "reasoning_effort": "high", "max_tokens": 8192, "temperature": .7,
        "extra_body": {"reasoning_effort": "high", "reasoning": {"enabled": True},
                       "max_tokens": 4096, "max_completion_tokens": 4096,
                       "temperature": .9, "top_p": .5, "routing": {"keep": True}},
    }
    original = deepcopy((configured, supplied))
    with mock_openai(recording(requests)) as raw:
        llm = HubLLM(raw, api=api, request_options=configured)
        if captured:
            llm = capturing(llm)
        pending = _method(llm, surface)(**supplied)
        if _deferred(surface):
            assert requests == []
        assert completion_text(_consume(pending, surface)) == "yes"
    assert len(requests) == 1
    wire = requests[0]
    assert wire["model"] == supplied["model"] and wire["messages"] == supplied["messages"]
    assert wire["max_completion_tokens"] == 128
    assert not {"max_tokens", "temperature", "top_p"}.intersection(wire)
    assert wire["routing"] == {"keep": True}
    if "reasoning_effort" in reasoning:
        assert wire["reasoning_effort"] == "none" and "reasoning" not in wire
    else:
        assert wire["reasoning"] == {"enabled": False} and "reasoning_effort" not in wire
    assert (configured, supplied) == original


@pytest.mark.parametrize("surface", REPRESENTATIVE_SURFACES)
@pytest.mark.parametrize("error_kind", ["reasoning", "token_oscillation"])
def test_alternate_chat_preserves_reasoning_and_bounds_compatibility_retries(surface, error_kind):
    requests = []
    reject = (oscillating if error_kind == "token_oscillation"
              else lambda body: unsupported("reasoning_effort", "max_tokens"))
    with mock_openai(recording(requests, reject)) as raw:
        llm = HubLLM(raw, api="chat_completions",
                     request_options={"reasoning_effort": "none", "max_tokens": 128})
        with pytest.raises(APIStatusError):
            _consume(_method(llm, surface)(model="arbitrary", messages=[]), surface)
    assert len(requests) == (1 if error_kind == "reasoning" else 3)
    assert all(body["reasoning_effort"] == "none" for body in requests)


@pytest.mark.parametrize("surface,api,when", [
    (surface, api, when) for surface in REPRESENTATIVE_SURFACES
    for api in ("chat_completions", "auto") for when in ("before", "during", "during_error")
    # Cancelling before dispatch takes one path for both APIs and both response wrappers.
    if when != "before" or (api == "chat_completions" and "with_streaming" not in surface)
])
def test_alternate_chat_cancellation_prevents_requests_retries_and_late_results(surface, api, when):
    token, requests, responses = Cancellation(), [], []

    def answer(body):
        token.cancel()
        responses.append(reply(body) if when != "during_error" else unsupported("temperature"))
        return responses[-1]

    with mock_openai(recording(requests, answer)) as raw:
        llm = capturing(HubLLM(raw, api=api, request_options={"temperature": 0}))
        if when == "before":
            token.cancel()
        with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
            _consume(_method(llm, surface)(model="arbitrary", messages=[]), surface)
    assert len(requests) == (0 if when == "before" else 1)
    if _deferred(surface):
        assert all(response.is_closed for response in responses)


@pytest.mark.parametrize("surface", [s for s in REPRESENTATIVE_SURFACES if _deferred(s)])
def test_alternate_chat_cancellation_after_context_creation_prevents_request(surface):
    requests = []

    with mock_openai(recording(requests)) as raw:
        llm = HubLLM(raw, api="chat_completions")
        token = Cancellation()
        with cancellation_scope(token):
            pending = _method(llm, surface)(model="arbitrary", messages=[])
            token.cancel()
            with pytest.raises(asyncio.CancelledError):
                _consume(pending, surface)
    assert requests == []


# The capture proxy intercepts only chat.completions.create; raw responses pass through it.
@pytest.mark.parametrize("surface,captured", [
    ("chat.completions.create", True), ("chat.with_raw_response.completions.create", False),
])
@pytest.mark.parametrize("cancel", [False, True])
def test_raw_chat_cancellation_closes_unconsumed_stream(surface, captured, cancel):
    token, responses = Cancellation(), []

    class Body(httpx.SyncByteStream):
        consumed = False
        closed = False

        def __iter__(self):
            self.consumed = True
            yield b"data: [DONE]\n\n"

        def close(self):
            self.closed = True

    body = Body()

    def handler(request):
        response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body,
        )
        responses.append(response)
        if cancel:
            token.cancel()
        return response

    with mock_openai(handler) as raw:
        llm = HubLLM(raw, api="chat_completions")
        if captured:
            llm = capturing(llm)
        with cancellation_scope(token):
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    _method(llm, surface)(model="arbitrary", messages=[], stream=True)
                # Assert before client teardown, without consuming the body.
                assert body.closed and responses[0].is_closed
            else:
                response = _method(llm, surface)(model="arbitrary", messages=[], stream=True)
                assert not body.closed and not responses[0].is_closed
                if "with_raw_response" in surface:
                    response.http_response.close()
                else:
                    response.close()
        assert not body.consumed


@pytest.mark.parametrize("surface,close_error", [("hub_return", True), ("captured_raw", False)])
def test_cancellation_at_response_handoff_closes_before_propagating(
    monkeypatch, surface, close_error,
):
    from autofa.llm import hub

    token, closed, captures = Cancellation(), [], []

    def close():
        closed.append(True)
        if close_error:
            raise RuntimeError("connection close failed")

    response = SimpleNamespace(close=close)

    def create(**kwargs):
        if surface == "captured_raw":
            token.cancel()
        return response

    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    if surface == "hub_return":
        complete = hub._compatible_completion

        def cancel_after_completion(*args, **kwargs):
            result = complete(*args, **kwargs)
            token.cancel()
            return result

        monkeypatch.setattr(hub, "_compatible_completion", cancel_after_completion)
        llm = HubLLM(raw, api="chat_completions")
    else:
        llm = capturing(raw, captures)
    with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
        llm.chat.completions.create(model="arbitrary", messages=[], stream=True)
    assert closed == [True]
    assert captures == []


# Decisions and Responses route discovery, fallback and capability caching.


def test_auto_probe_prefers_decisions_and_reports_both_routes():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/decisions"):
            return httpx.Response(200, json=decision("OK", ("OK", "FAILED")))
        return httpx.Response(200, json=generated("OK"))

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="auto", model="luna")
        assert llm.label_api is None  # Auto claims no label route before discovery.
        before = llm.routing_identity()
        probe = llm.probe()
        assert probe.ok and probe.label_api == "decisions" and probe.text_api == "responses"
        assert before != llm.routing_identity()
        assert llm.localise("Extract the exact quote", "OK") == "OK"
    assert paths == ["/prefix/v1/decisions", "/prefix/v1/responses", "/prefix/v1/responses"]


@pytest.mark.parametrize("api", ["auto", "responses"])
def test_probe_uses_valid_responses_minimum_without_changing_generation_budget(api):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/decisions"):
            return httpx.Response(200, json=decision("OK", ("OK", "FAILED")))
        assert request.url.path.endswith("/responses")
        bodies.append(body)
        if body["max_output_tokens"] < 16:
            return httpx.Response(400, json={"error": {
                "type": "invalid_request_error", "param": "max_output_tokens",
                "message": "max_output_tokens must be at least 16",
            }})
        return httpx.Response(200, json=generated("OK"))

    with client_for(handler) as raw:
        llm = HubLLM(raw, api=api, model="luna", max_tokens=64)
        probe = llm.probe()
        assert probe.ok and probe.text_api == "responses"
        assert cached_capability(raw, "luna", "responses") is True
        assert llm.max_tokens == 64
    assert len(bodies) == 1
    assert bodies[0]["max_output_tokens"] == 16


def test_probe_worker_inherits_dispatch_budget_and_cancellation():
    reservations, calls = [], []
    token = Cancellation()

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/decisions"):
            return httpx.Response(404, json={"error": {"code": "route_not_found"}})
        return httpx.Response(200, json=generated("OK"))

    with client_for(handler) as raw, model_call_budget(lambda: reservations.append(1)), \
            cancellation_scope(token):
        llm = HubLLM(raw, api="auto")
        assert llm.probe().ok
        assert len(reservations) == len(calls) == 2
        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            llm.probe()
        assert len(calls) == 2


def test_explicit_responses_bypasses_decisions_in_labels_and_extraction():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=generated("yes"))

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="responses")
        assert llm.classify_options("Evidence", "Check") == "yes"
        assert llm.localise("Extract", "Evidence") == "yes"
        with pytest.raises(ValueError, match="native Decisions probabilities"):
            llm.decide([{"type": "predicate", "instructions": "Check"}], "Evidence")
    assert paths == ["/prefix/v1/responses", "/prefix/v1/responses"]


@pytest.mark.parametrize("api, route", [("auto", "decisions"), ("responses", "responses")])
def test_cancelled_failed_request_neither_caches_unavailability_nor_falls_back(api, route):
    token = Cancellation()
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if len(paths) == 1:
            token.cancel()
            return httpx.Response(404, json={"error": {"code": "route_not_found"}})
        return httpx.Response(200, json=decision() if route == "decisions" else generated())

    with client_for(handler) as raw:
        llm = HubLLM(raw, api=api, model="test-model")
        with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
            llm.classify_options("Evidence", "Check")
        assert paths == [f"/prefix/v1/{route}"]
        assert cached_capability(raw, "test-model", route) is None
        # A cancelled operation must not poison route selection for the next tool.
        assert llm.classify_options("Evidence", "Check") == "yes"
        assert cached_capability(raw, "test-model", route) is True
    assert paths == [f"/prefix/v1/{route}"] * 2


@pytest.mark.parametrize("changed", ["nothing", "endpoint", "model", "credential"])
def test_capability_cache_is_scoped_to_endpoint_model_and_account(changed):
    probes = []

    def handler(request):
        if request.url.path.endswith("/decisions"):
            probes.append(request.url)
            return httpx.Response(404, json={"error": {"code": "route_not_found"}})
        return httpx.Response(200, json=generated())

    for index in range(2):
        endpoint = "https://other.example/v1" if index and changed == "endpoint" else None
        with client_for(handler, key="key-two" if index and changed == "credential" else "key-one",
                        **({"endpoint": endpoint} if endpoint else {})) as raw:
            HubLLM(raw, api="auto", model="new" if index and changed == "model" else "old"
                   ).classify_options("Evidence", "Check")
    # An unsupported route is cached across clients for the same endpoint, model and account.
    assert len(probes) == (1 if changed == "nothing" else 2)


def test_credential_rotation_changes_persistent_identity_and_cached_answers():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=generated("yes" if len(calls) == 1 else "no"))

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="responses")
        assert llm.classify("Question", "Evidence") == "yes"
        before = llm.routing_identity()
        raw.api_key = "rotated-secret"
        assert llm.classify("Question", "Evidence") == "no"
        after = llm.routing_identity()
    assert before != after
    assert before["responses_available"] is after["responses_available"] is True
    assert "key-one" not in json.dumps(before) and "rotated-secret" not in json.dumps(after)
    assert len(calls) == 2


@pytest.mark.parametrize("status", [401, 501, 503])
def test_outages_never_fall_back_or_negative_cache(status):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(status, json={"error": {"message": "Unavailable"}})

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="auto", model="luna")
        for _ in range(2):
            with pytest.raises(APIStatusError):
                llm.classify_options("Evidence", "Check")
            assert cached_capability(raw, "luna", "decisions") is None
    assert paths == ["/prefix/v1/decisions"] * 2


def test_missing_model_is_not_an_unsupported_api():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(404, json={"error": {"code": "model_not_found",
                                                 "message": "Model does not exist"}})

    with client_for(handler) as raw, pytest.raises(APIStatusError):
        HubLLM(raw, api="auto").classify_options("Evidence", "Check")
    assert paths == ["/prefix/v1/decisions"]


@pytest.mark.parametrize("failure", ["timeout", "malformed_answer"])
def test_timeout_or_malformed_decision_does_not_trigger_fallback(failure):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if failure == "timeout":
            raise httpx.ReadTimeout("Connection timed out", request=request)
        return httpx.Response(200, json={"answers": []})

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="auto", model="luna")
        for _ in range(2):
            with pytest.raises(APITimeoutError if failure == "timeout" else RuntimeError):
                llm.classify_options("Evidence", "Check")
        assert cached_capability(raw, "luna", "decisions") is None
    assert paths == ["/prefix/v1/decisions"] * 2


@pytest.mark.parametrize("status,code", [(400, "unsupported_model"),
                                         (405, "method_not_allowed"), (501, "not_implemented")])
def test_explicit_unsupported_endpoint_or_model_can_fall_back(status, code):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/decisions"):
            return httpx.Response(status, json={"error": {"code": code}})
        return httpx.Response(200, json=generated())

    with client_for(handler) as raw:
        assert HubLLM(raw, api="auto").classify_options("Evidence", "Check") == "yes"
    assert paths == ["/prefix/v1/decisions", "/prefix/v1/responses"]


def test_unsupported_responses_falls_back_to_chat_only_in_auto():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=response("OK"))
        return httpx.Response(404, json={"error": {"code": "route_not_found"}})

    with client_for(handler) as raw:
        probe = HubLLM(raw, api="auto").probe()
        assert probe.ok and probe.label_api == probe.text_api == "chat_completions"
        with pytest.raises(APIUnavailable):
            HubLLM(raw, api="responses").classify("Prompt", "Evidence")
    assert paths == ["/prefix/v1/decisions", "/prefix/v1/responses", "/prefix/v1/chat/completions"]


def test_responses_preserves_json_images_reasoning_budget_and_tools():
    bodies = []
    image_bytes = b"verified-image"
    image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    schema = {"type": "object", "properties": {"reason": {"type": "string"}},
              "required": ["reason"], "additionalProperties": False}

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=generated('{"reason":"exact evidence"}', usage={
            "input_tokens": 12, "output_tokens": 4, "total_tokens": 16,
        }))

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="responses", request_options={
            "max_completion_tokens": 512, "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "evidence", "schema": schema, "strict": True}},
        })
        result = llm.classify("Extract JSON", "Exact source", images=[{
            "status": "available", "data_url": image_url,
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        }])
        assert json.loads(result) == {"reason": "exact evidence"}
        response = llm.chat.completions.create(
            model="chosen-model", messages=[{"role": "user", "content": "Source"}],
            tools=[{"type": "function", "function": {"name": "report", "parameters": schema}}],
            tool_choice={"type": "function", "function": {"name": "report"}},
        )
        assert completion_text(response) == result
        assert response.usage.total_tokens == 16
    assert bodies[0]["input"][0]["content"][1] == {
        "type": "input_image", "image_url": image_url}
    assert bodies[0]["max_output_tokens"] == 512
    assert bodies[0]["reasoning"] == {"effort": "none"}
    assert bodies[0]["text"]["format"]["schema"] == schema
    assert bodies[1]["model"] == "chosen-model"
    assert bodies[1]["tools"][0] == {"type": "function", "name": "report", "parameters": schema}
    assert bodies[1]["tool_choice"] == {"type": "function", "name": "report"}


def test_responses_function_call_round_trip_keeps_ids_arguments_and_output():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=generated(output=[{
            "type": "function_call", "call_id": "call_1", "name": "report",
            "arguments": '{"reason":"exact source"}',
        }]))

    with client_for(handler) as raw:
        result = routed_completion(raw, api="responses", model="chosen", messages=[
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "prior", "type": "function",
                "function": {"name": "fetch", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "prior", "content": "source text"},
        ])
    assert bodies[0]["input"][1] == {
        "type": "function_call_output", "call_id": "prior", "output": "source text"}
    assert result.choices[0].finish_reason == "tool_calls"
    call = result.choices[0].message.tool_calls[0]
    assert call.id == "call_1" and json.loads(call.function.arguments)["reason"] == "exact source"


def test_native_responses_settings_override_generated_chat_aliases():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=generated())

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="responses", request_options={
            "max_output_tokens": 64, "reasoning_effort": "none",
        })
        llm.classify_options("Evidence", "Check")
        llm.chat.completions.create(model="chosen", messages=[{"role": "user", "content": "x"}],
                                    max_tokens=1000, reasoning_effort="high", extra_body={
                                        "reasoning": {"effort": "high"}, "max_tokens": 900})
    for body in bodies:
        assert body["max_output_tokens"] == 64
        assert body["reasoning"] == {"effort": "none"}
        assert not {"max_tokens", "max_completion_tokens", "reasoning_effort"}.intersection(body)


def test_auto_chat_only_request_preserves_contract_without_poisoning_responses():
    paths, bodies = [], []

    def handler(request):
        paths.append(request.url.path)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=(generated() if request.url.path.endswith("/responses")
                                        else response()))

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="auto", model="luna")
        for _ in range(2):
            result = llm.chat.completions.create(model="luna", messages=[
                {"role": "user", "content": "Evidence"}], stop=["END"])
            assert completion_text(result) == "yes"
        assert cached_capability(raw, "luna", "responses") is None
        assert llm.classify("Question", "Evidence") == "yes"
    assert paths == ["/prefix/v1/chat/completions"] * 2 + ["/prefix/v1/responses"]
    assert bodies[0]["stop"] == ["END"]


def test_auto_unsupported_responses_preserves_native_budget_and_reasoning_on_chat():
    bodies = []

    def handler(request):
        bodies.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/responses"):
            return httpx.Response(404, json={"error": {"code": "route_not_found"}})
        return httpx.Response(200, json=response())

    with client_for(handler) as raw:
        llm = HubLLM(raw, api="auto", request_options={
            "max_output_tokens": 64, "reasoning": {"effort": "none"},
        })
        assert llm.classify("Question", "Evidence") == "yes"
    assert len(bodies) == 2
    assert bodies[1][1]["max_completion_tokens"] == 64
    assert bodies[1][1]["reasoning_effort"] == "none"
    assert "max_output_tokens" not in bodies[1][1] and "reasoning" not in bodies[1][1]


@pytest.mark.parametrize("body", [
    generated("yes", status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
    generated(status="failed", error={"code": "server_error"}),
    generated(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "No"}]}]),
])
def test_incomplete_failed_or_refused_responses_are_never_labels(body):
    from autofa.core.sandbox import DetectorAbstained

    refused = any(part.get("type") == "refusal" for item in body.get("output", [])
                  for part in item.get("content", []))
    with client_for(lambda request: httpx.Response(200, json=body)) as raw:
        with pytest.raises(DetectorAbstained if refused else RuntimeError):
            HubLLM(raw, api="responses", max_tokens=128).classify_options("Evidence", "Check")


def test_negative_capability_discovery_is_single_flight():
    lock = threading.Lock()
    counts = {"decisions": 0, "responses": 0}

    def handler(request):
        api = request.url.path.rsplit("/", 1)[-1]
        with lock:
            counts[api] += 1
        return (httpx.Response(404, json={"error": {"code": "route_not_found"}})
                if api == "decisions" else httpx.Response(200, json=generated()))

    with client_for(handler) as raw, ThreadPoolExecutor(max_workers=4) as pool:
        llm = HubLLM(raw, api="auto")
        assert list(pool.map(lambda _: llm.classify_options("Evidence", "Check"), range(4))) == [
            "yes"] * 4
    assert counts == {"decisions": 1, "responses": 4}


def test_cancelled_discovery_waiter_does_not_reserve_or_dispatch(monkeypatch):
    from autofa.llm import capabilities

    first_dispatched = threading.Event()
    waiter_entered = threading.Event()
    release_first = threading.Event()
    waiter_token = Cancellation()
    calls, reservations = [], []

    class ObservedGate:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if self.lock.locked():
                waiter_entered.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    def handler(request):
        calls.append(request.url.path)
        first_dispatched.set()
        assert release_first.wait(5), "Test did not release the first request"
        return httpx.Response(200, json=decision())

    with client_for(handler) as raw, ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setitem(capabilities._DISCOVERY_LOCKS,
                            capabilities._key(raw, "luna", "decisions"), ObservedGate())
        llm = HubLLM(raw, api="auto", model="luna")

        def call(token):
            with cancellation_scope(token), model_call_budget(lambda: reservations.append(1)):
                return llm.classify_options("Evidence", "Check")

        first = pool.submit(call, Cancellation())
        try:
            assert first_dispatched.wait(5)
            waiter = pool.submit(call, waiter_token)
            assert waiter_entered.wait(5)
            waiter_token.cancel()
        finally:
            release_first.set()
        assert first.result(timeout=5) == "yes"
        with pytest.raises(asyncio.CancelledError):
            waiter.result(timeout=5)
    assert calls == ["/prefix/v1/decisions"] and reservations == [1]


@pytest.mark.parametrize("surface,extra", [
    ("create", {"stop": ["END"]}), ("parse", {}), ("with_raw_response.create", {}),
], ids=["chat_only_option", "parse", "raw_response"])
def test_unsupported_sdk_shapes_fail_explicitly_without_http(surface, extra):
    with client_for(lambda request: pytest.fail("Unsupported request must not dispatch")) as raw:
        method = _method(HubLLM(raw, api="responses").chat.completions, surface)
        with pytest.raises(ValueError, match="select Chat Completions"):
            method(model="chosen", messages=[{"role": "user", "content": "Source"}], **extra)


@pytest.mark.parametrize('details,expected', [
    (None, 'length'), ({'reason': 'content_filter'}, 'content_filter'),
])
def test_incomplete_responses_only_explicit_filter_is_refusal(details, expected):
    from autofa.llm.responses import _completion

    result = _completion({'status': 'incomplete', 'incomplete_details': details, 'output': []}, 'm')
    assert result.choices[0].finish_reason == expected


def _response(text=None, *, calls=None, status="completed"):
    output = calls or [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text}]}]
    return SimpleNamespace(
        status=status, error=None, model="model", output_text=text,
        incomplete_details=({"reason": "max_output_tokens"} if status == "incomplete" else None),
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        model_dump=lambda: {
            "status": status, "model": "model", "output": output,
            "incomplete_details": ({"reason": "max_output_tokens"}
                                   if status == "incomplete" else None),
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    )


def _client(monkeypatch, responses):
    calls = []

    class Client:
        api_key = "unit-test-key"
        base_url = "https://provider.invalid/v1"

        def __init__(self):
            self.responses = SimpleNamespace(create=self.create)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.forbid))

        def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

        def forbid(self, **kwargs):
            pytest.fail("This supported Responses request must not fall back to Chat")

        def post(self, *args, **kwargs):
            pytest.fail("Freeform evidence and tool conversations cannot use Decisions")

        def with_options(self, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    client = Client()
    monkeypatch.setattr("autofa.llm.hub.make_hub_client", lambda **kwargs: client)
    return client, calls


def test_improvement_explanations_use_generation_and_preserve_reasoning_policy(monkeypatch):
    # Even an explicit Decisions preference uses generation for freeform explanations.
    api = "decisions"
    _, calls = _client(monkeypatch, [_response(json.dumps(_reply()))])
    reservations = []
    with model_call_budget(lambda: reservations.append(1)):
        result = classify_improvement(
            fn=_fn(), items=[{"_item_id": "item-1", "response_text": "Wrong tool argument"}],
            verdicts=[{"item_id": "item-1", "verdict": "yes"}],
            cfg=Config(feature_api=api),
        )
    assert result["status"] == "assessed" and result["classifier_api"] == api
    assert result["rationale"] == _reply()["rationale"]
    assert len(calls) == len(reservations) == 1
    assert calls[0]["reasoning"] == {"effort": "none"}
    assert "confirmed_examples" in calls[0]["input"][1]["content"]


def test_taxonomy_explicit_responses_keeps_choice_rationale_and_structured_evidence(monkeypatch):
    payload = {"answers": {
        "kind": {"type": "choice", "choice": "a", "alternatives": ["b"],
                 "rationale": "Event 2 supports A instead of B."},
        "evidence": {"type": "structured", "value": {"quote": "specific evidence"}},
    }}
    _, calls = _client(monkeypatch, [_response(json.dumps(payload))])
    provider = OpenAIProvider(endpoint="https://judge.example/v1", api_key="unit-test-key",
                              model="model", api="responses",
                              max_retries=0, request_options={"reasoning_effort": "none"})
    reservations = []
    with model_call_budget(lambda: reservations.append(1)):
        result = provider.evaluate("Event 2: specific evidence", {
            "kind": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            "evidence": {"type": "structured", "schema": {"quote": ""}},
        })
    assert result.answers["kind"]["alternatives"] == ["b"]
    assert result.answers["kind"]["rationale"] == payload["answers"]["kind"]["rationale"]
    assert result.answers["evidence"]["value"] == {"quote": "specific evidence"}
    assert result.usage == {"input_tokens": 10, "output_tokens": 5}
    assert len(calls) == len(reservations) == 1
    assert calls[0]["reasoning"] == {"effort": "none"}


def test_taxonomy_rejects_truncated_json_even_when_it_is_parseable(monkeypatch):
    payload = {"answers": {"kind": {"type": "choice", "choice": "a", "rationale": "A"}}}
    _client(monkeypatch, [_response(json.dumps(payload), status="incomplete")])
    provider = OpenAIProvider(endpoint="https://judge.example/v1", api_key="unit-test-key",
                              model="model", api="responses",
                              max_retries=0)
    with pytest.raises(ProviderError, match="incomplete"):
        provider.evaluate("evidence", {"kind": {"type": "choice", "criteria": {"a": "A"}}})


def test_cancelled_taxonomy_never_dispatches_a_request(monkeypatch):
    _, calls = _client(monkeypatch, [])
    provider = OpenAIProvider(endpoint="https://judge.example/v1", api_key="unit-test-key",
                              model="model", api="responses",
                              max_retries=0)
    cancellation = Cancellation()
    cancellation.cancel()
    with cancellation_scope(cancellation), pytest.raises(asyncio.CancelledError):
        provider.evaluate("evidence", {"kind": {"type": "choice", "criteria": {"a": "A"}}})
    assert calls == []


def test_cheating_review_preserves_chat_tool_roundtrip_and_endpoint_defaults(
    monkeypatch, tmp_path,
):
    review = cheating_review(tmp_path)
    calls = []
    messages_to_return = [
        SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            id="read-1", function=SimpleNamespace(
                name="read_evidence", arguments=json.dumps({"field": "response_text"}),
            ),
        )]),
        SimpleNamespace(content="Final explanation with quoted evidence", tool_calls=[]),
    ]

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=messages_to_return.pop(0))],
            usage=SimpleNamespace(total_tokens=15),
        )

    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("autofa.llm.hub.make_hub_client", lambda **kwargs: raw)
    cfg = Config(feature_api="decisions")
    client = cheating_pipeline._client(cfg)
    reservations = []
    with model_call_budget(lambda: reservations.append(1)):
        text, messages, tokens = cheating._session(
            review, "Inspect the evidence", model=cfg.cheating_check_model, client=client,
            max_turns=2, resume=None,
        )
    assert text == "Final explanation with quoted evidence"
    assert len(calls) == len(reservations) == 2 and tokens == 30
    assert all("reasoning" not in call and "reasoning_effort" not in call for call in calls)
    assert calls[0]["tools"][0]["type"] == "function"
    outputs = [row for row in calls[1]["messages"] if row.get("role") == "tool"]
    assert len(outputs) == 1 and outputs[0]["tool_call_id"] == "read-1"
    assert messages[-2]["role"] == "tool"


def test_discovery_policy_changes_preserve_cheating_cache():
    baseline = Config(feature_api="chat_completions")
    assert cheating_pipeline.check_version(baseline) == cheating_pipeline.check_version(Config(
        feature_api="decisions", feature_request_options={"reasoning_effort": "low"},
        improvement_request_options={"reasoning_effort": "low"},
    ))


def test_cheating_workers_inherit_dispatch_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cheating_pipeline, "_client", lambda cfg: None)

    def confirm(*args, **kwargs):
        reserve_model_call()
        return {"status": "unjudged", "error": "test review"}

    monkeypatch.setattr(cheating_pipeline, "confirm", confirm)
    reservations = []
    with model_call_budget(lambda: reservations.append(1)):
        cheating_pipeline.judge_items(
            [{"_item_id": "one", "response_text": "git log --all"}], Config(), tmp_path)
    assert reservations == [1]
