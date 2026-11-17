# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Account actual API dispatches without changing lazy SDK or cache behavior."""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
from openai import LengthFinishReasonError, OpenAI

from autofa.config import RunConfig
from autofa.core.localisation import CapturingLLM
from autofa.core.sandbox import DetectorAbstained
from autofa.harness.usage import normalize_tokens
from autofa.llm.decisions import create_decision
from autofa.llm.hub import HubLLM, chat_completions_safe
from autofa.llm.responses import create_response_completion
from autofa.llm.usage import _endpoint as api_endpoint
from autofa.llm.usage import normalized_usage, observed_call
from autofa.mcp import runtime
from autofa.session import usage as session_usage
from autofa.session.usage import _endpoint as pricing_endpoint
from autofa.session.usage import set_usage_pricing, usage_scope, usage_summary
from autofa.util.cancellation import Cancellation, cancellation_scope
from autofa.util.model_budget import model_call_budget

try:
    import httpx
except ImportError:
    import httpx2 as httpx


USAGE = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
         "prompt_tokens_details": {"cached_tokens": 4},
         "completion_tokens_details": {"reasoning_tokens": 2}}


@pytest.mark.parametrize("bound", [False, True])
def test_shared_raw_chat_does_not_inherit_discovery_accounting(tmp_path, monkeypatch, bound):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig("previous-discovery", [], "model")
    run.save()
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(run=run))
    response = SimpleNamespace(usage=USAGE, id="other-workflow")
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    scope = usage_scope(run_dir=run.run_dir(), phase="discovery") if bound else nullcontext()
    with scope:
        # Taxonomy and cheating use this shared helper without an accounting role.
        assert chat_completions_safe(client, model="other-model", messages=[]) is response
    assert sent == [{"model": "other-model", "messages": []}]
    assert not (run.run_dir() / "usage.sqlite3").exists()
    assert not (run.run_dir() / "usage_gaps.jsonl").exists()


def test_shared_chat_explicit_discovery_role_records_once(tmp_path):
    response = SimpleNamespace(usage=USAGE, id="discovery-dispatch")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: response)))
    with usage_scope(run_dir=tmp_path, phase="discovery", exact_dir=True):
        assert chat_completions_safe(client, model="model", messages=[],
                                     usage_role="detector") is response
    summary = usage_summary(tmp_path)
    assert summary["totals"]["calls"] == 1
    assert summary["by_role"][0]["role"] == "detector"
    assert summary["totals"]["tokens"]["total_tokens"] == 13


@pytest.fixture
def accounting(monkeypatch):
    rows = {}
    context = {"run_dir": "/unused", "round_index": 2, "phase": "discovery"}
    monkeypatch.setattr(session_usage, "capture_context", lambda **kw: {**context, **kw})

    def record(captured, **row):
        rows[row["event_id"]] = deepcopy({**captured, **row})

    monkeypatch.setattr(session_usage, "record_usage", record)
    return rows, context


def _client(handler):
    return OpenAI(api_key="private-key", base_url="https://example.test/v1", max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _completion(*, finish="stop", refusal=None, usage=USAGE):
    return {"id": "completion-id", "created": 0, "model": "model", "usage": usage,
            "object": "chat.completion", "choices": [{"index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": "yes", "refusal": refusal}}]}


def test_cached_classification_does_not_charge_again(accounting):
    rows, _ = accounting
    with _client(lambda req: httpx.Response(200, json={
        **_completion(), "cost_usd": .125,
    })) as client:
        llm = HubLLM(client, model="model", api="chat_completions")
        assert llm.classify("private prompt", "private evidence") == "yes"
        assert llm.classify("private prompt", "private evidence") == "yes"
    assert len(rows) == 1
    row, = rows.values()
    assert row["usage"] == {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
                            "cache_read_input_tokens": 4, "reasoning_output_tokens": 2}
    assert row["cost_usd"] == .125 and row["cost_kind"] == "reported"
    assert row["role"] == "detector" and row["round_index"] == 2
    assert "private" not in json.dumps(row)


def test_truncation_rescue_records_every_consumed_completion(accounting):
    rows, _ = accounting
    caps = []

    def handler(request):
        cap = json.loads(request.content)["max_tokens"]
        caps.append(cap)
        return httpx.Response(200, json=_completion(finish="stop" if cap == 8192 else "length"))

    with _client(handler) as client:
        assert HubLLM(client, model="model", api="chat_completions").classify("p", "e") == "yes"
    assert caps == [64, 4096, 8192]
    assert len(rows) == 3
    assert sum(r["usage"]["input_tokens"] for r in rows.values()) == 30


def test_compatibility_retries_and_refusal_usage_are_retained(accounting):
    rows, _ = accounting

    def handler(request):
        if "temperature" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"param": "temperature",
                "code": "unsupported_parameter", "message": "Unsupported temperature"}})
        return httpx.Response(200, json=_completion(refusal="Cannot answer"))

    with _client(handler) as client, pytest.raises(DetectorAbstained):
        HubLLM(client, model="model", api="chat_completions").classify("p", "e")
    failed, refused = rows.values()
    assert failed["status"] == "failed" and failed["usage"] is None
    assert failed["cost_usd"] is None
    assert refused["usage"]["input_tokens"] == 10


@pytest.mark.parametrize("api", ["responses", "decisions"])
def test_usage_survives_response_validation_failure(accounting, api):
    rows, _ = accounting
    payload = {"id": "invalid", "usage": {"input_tokens": 12}, "status": "failed"}
    with _client(lambda request: httpx.Response(200, json=payload)) as client:
        with pytest.raises(RuntimeError):
            if api == "responses":
                create_response_completion(client, model="model", messages=[])
            else:
                create_decision(client, model="model", input="evidence", questions=[{
                    "type": "predicate", "instructions": "Is it true?",
                }])
    row, = rows.values()
    assert row["usage"] == {"input_tokens": 12}
    assert row["cost_usd"] is None


def _stream_response():
    chunks = [
        {"id": "stream", "created": 0, "model": "model", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "yes"},
                      "finish_reason": None}]},
        {"id": "stream", "created": 0, "model": "model", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "stream", "created": 0, "model": "model", "object": "chat.completion.chunk",
         "choices": [], "usage": USAGE},
    ]
    content = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


# Each SDK surface once. The capture proxy intercepts only create(), so it wraps the
# lazy stream; the other surfaces pass through it unchanged.
@pytest.mark.parametrize("surface,captured", [
    ("raw", False), ("deferred", False), ("stream", True), ("stream_raw", False),
    ("stream_sdk", False),
])
def test_lazy_sdk_reads_update_one_dispatch_in_original_round(accounting, surface, captured):
    rows, context = accounting
    calls = []

    def handler(request):
        calls.append(1)
        if json.loads(request.content).get("stream"):
            return _stream_response()
        return httpx.Response(200, json=_completion())

    with _client(handler) as client:
        llm = HubLLM(client, model="model", api="chat_completions")
        if captured:
            llm = CapturingLLM(llm, SimpleNamespace(llm_call=lambda *args: None))
        resource = llm.chat.completions
        if surface == "raw":
            response = resource.with_raw_response.create(model="model", messages=[])
        elif surface == "deferred":
            response = resource.with_streaming_response.create(model="model", messages=[])
        elif surface == "stream":
            response = resource.create(model="model", messages=[], stream=True)
        elif surface == "stream_raw":
            response = resource.with_raw_response.create(model="model", messages=[], stream=True)
        else:
            response = resource.stream(model="model", messages=[])
        if surface in {"deferred", "stream_sdk"}:
            assert not rows and not calls
            with response as entered:
                context["round_index"] = 3
                if surface == "deferred":
                    assert entered.parse().choices[0].message.content == "yes"
                else:
                    assert entered.get_final_completion().choices[0].message.content == "yes"
        else:
            assert len(rows) == 1
            assert next(iter(rows.values()))["usage"] is None
            context["round_index"] = 3
            if surface == "raw":
                assert response.parse().choices[0].message.content == "yes"
            elif surface == "stream_raw":
                assert len(list(response.parse())) == 3
            else:
                assert len(list(response)) == 3
    row, = rows.values()
    assert row["usage"]["input_tokens"] == 10
    assert row["round_index"] == 2 and len(calls) == 1


def test_cancelled_stream_remains_unconsumed_and_unknown(accounting):
    rows, _ = accounting
    token = Cancellation()

    class Body(httpx.SyncByteStream):
        consumed = closed = False

        def __iter__(self):
            self.consumed = True
            yield b"data: [DONE]\n\n"

        def close(self):
            self.closed = True

    body = Body()

    def handler(request):
        token.cancel()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    with (_client(handler) as client, cancellation_scope(token),
          pytest.raises(asyncio.CancelledError)):
        HubLLM(client, api="chat_completions").chat.completions.create(
            model="model", messages=[], stream=True)
    assert body.closed and not body.consumed
    row, = rows.values()
    assert row["usage"] is None and row["cost_usd"] is None


def test_localisation_has_independent_role(accounting):
    rows, _ = accounting
    with _client(lambda request: httpx.Response(200, json=_completion())) as client:
        HubLLM(client, model="model", api="chat_completions").localise("p", "e")
    assert next(iter(rows.values()))["role"] == "highlighting"


def test_unknown_cost_and_invalid_counts_are_not_zero(accounting):
    rows, _ = accounting
    result = {"usage": {"input_tokens": -1, "output_tokens": True, "cost": 99}}
    client = SimpleNamespace(base_url="https://secret:password@example.test/v1?key=private",
                             max_retries=2)
    assert observed_call(lambda: result, client=client, usage_model="m", api="responses") is result
    row, = rows.values()
    assert row["usage"] is None and row["cost_usd"] is None
    assert row["endpoint"] == "https://example.test/v1"
    assert normalized_usage({"usage": {"input_tokens": 0}}) == {"input_tokens": 0}


def test_budget_rejection_does_not_record_unsent_request(accounting):
    rows, _ = accounting

    def exhausted():
        raise RuntimeError("Budget exhausted")

    with (_client(lambda request: pytest.fail("Request must not be sent")) as client,
          model_call_budget(exhausted), pytest.raises(RuntimeError, match="Budget exhausted")):
        HubLLM(client, api="chat_completions").chat.completions.create(model="model", messages=[])
    assert not rows


def test_sdk_parse_error_preserves_consumed_usage(accounting):
    rows, _ = accounting
    with (_client(lambda request: httpx.Response(200, json=_completion(finish="length"))) as client,
          pytest.raises(LengthFinishReasonError)):
        HubLLM(client, api="chat_completions").chat.completions.parse(model="model", messages=[])
    row, = rows.values()
    assert row["usage"]["input_tokens"] == 10 and row["status"] == "failed"


def test_consumed_raw_response_updates_durable_ledger_once(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"run_id": "test", "round": 0}))
    with (_client(lambda request: httpx.Response(200, json=_completion())) as client,
          usage_scope(run_dir=tmp_path, round_index=0)):
        raw = HubLLM(client, api="chat_completions").chat.completions.with_raw_response.create(
            model="model", messages=[])
        pending = usage_summary(tmp_path)["totals"]
        assert pending["calls"] == 1 and pending["missing_usage_calls"] == 1
        raw.parse()
        raw.parse()  # SDK returns its cached parse; it must not become a second call.
    totals = usage_summary(tmp_path)["totals"]
    assert totals["calls"] == 1 and totals["missing_usage_calls"] == 0
    assert totals["tokens"]["input_tokens"] == 10
    assert totals["known_cost_usd"] is None and totals["missing_cost_calls"] == 1


def test_sdk_event_stream_explicit_next_preserves_nested_usage(accounting):
    rows, _ = accounting
    with _client(lambda request: _stream_response()) as client:
        llm = HubLLM(client, api="chat_completions")
        with llm.chat.completions.stream(model="model", messages=[]) as stream:
            while True:
                try:
                    next(stream)
                except StopIteration:
                    break
    row, = rows.values()
    assert row["usage"]["input_tokens"] == 10
    assert row["usage"]["output_tokens"] == 3
    assert row["status"] == "complete"


def test_real_chat_usage_uses_exact_endpoint_rate_card_without_subset_double_charge(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"run_id": "test", "round": 0}))
    set_usage_pricing(tmp_path, {"version": 1, "currency": "USD", "rates": [{
        "provider": "openai_compatible", "model": "model", "endpoint": "https://example.test/v1",
        "input_per_million": 2, "cached_input_per_million": 1, "output_per_million": 4,
        "source": "Synthetic test rates", "effective_at": "2026-11-03",
    }]})
    with (_client(lambda request: httpx.Response(200, json=_completion())) as client,
          usage_scope(run_dir=tmp_path)):
        HubLLM(client, model="model", api="chat_completions").classify("p", "e")
    totals = usage_summary(tmp_path)["totals"]
    # 6 fresh + 4 cached input; all 3 output tokens already include the 2 reasoning tokens.
    assert totals["estimated_cost_usd"] == pytest.approx((6 * 2 + 4 * 1 + 3 * 4) / 1_000_000)
    assert totals["reported_cost_usd"] is None
    assert totals["cost_complete"]


# Shared parsing must not make the same provider counts differ by dispatch path.
@pytest.mark.parametrize("value,expected", [(8.0, 8), (8.5, None), (True, None)])
def test_token_counts_are_consistent_across_harness_and_api(value, expected):
    native = normalize_tokens({"inputTokens": value})
    api = normalized_usage({"usage": {"prompt_tokens": value}})
    assert native.get("input_tokens") == api.get("input_tokens") == expected


@pytest.mark.parametrize("endpoint,expected", [
    ("https://user:secret@example.org:8443/v1/?token=secret#fragment",
     "https://example.org:8443/v1"),
    ("http://[::1]:8080/v1/", "http://[::1]:8080/v1"),
    ("http://example.org:bad/v1", None),
])
def test_endpoint_identity_is_shared_between_api_and_pricing(endpoint, expected):
    assert api_endpoint(SimpleNamespace(base_url=endpoint)) == expected
    assert pricing_endpoint(endpoint) == expected
