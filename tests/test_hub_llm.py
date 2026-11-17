# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HubLLM wiring, probes, sandboxed detectors, reasoning-safe labels and windowed evidence."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from autofa.core.detector_lint import evidence_budget_warnings, lint_detector_source
from autofa.core.sandbox import DetectorAbstained, compile_detector, run_detector
from autofa.llm.hub import (
    HubLLM,
    _classify_evidence,
    completion_text,
    get_api_key,
    humanize_hub_error,
    make_hub_client,
    resolve_api_key_env,
)
from autofa.llm.windowing import text_window, text_window_evidence, window_provenance

from support.judge import client_for, response


def delayed(seconds, reply):
    def wait():
        time.sleep(seconds)
        return reply

    return wait


def test_optional_highlight_timeout_disables_sdk_retries_without_changing_detector(monkeypatch):
    (client, _), (bounded, _) = client_for(), client_for()
    options = []
    client.with_options = lambda **kw: options.append(kw) or bounded
    seen = []
    monkeypatch.setattr(HubLLM, "classify", lambda self, *args, **kw:
                        seen.append((self._client, self.model, self.api)) or "quote")
    llm = HubLLM(client, model="test-model", api="decisions")
    assert llm.localise("Find the quote", "evidence", timeout=.25) == "quote"
    assert options == [{"timeout": .25, "max_retries": 0}]
    assert seen == [(bounded, "test-model", "decisions")]
    assert llm._client is client
    with pytest.raises(TimeoutError):
        llm.localise("Find the quote", "evidence", timeout=0)
    assert len(options) == len(seen) == 1




@pytest.mark.parametrize("api,probe_tokens", [("auto", 16), ("chat_completions", 8)])
def test_hub_probe_ok(api, probe_tokens):
    client, calls = client_for(response("OK"))
    hub = HubLLM(client, model="gcp/google/gemini-3.5-flash-lite", api=api)
    result = hub.probe(timeout_s=5.0, slow_s=8.0)
    assert result.ok is True and result.reply == "OK"
    assert calls[0]["max_tokens"] == probe_tokens


def test_hub_probe_error():
    exc = RuntimeError("Error code: 404 - model not found")
    exc.status_code = 404  # type: ignore[attr-defined]
    hub = HubLLM(client_for(exc)[0], model="missing-model")
    result = hub.probe(timeout_s=5.0)
    assert result.ok is False
    assert "404" in result.error or "model" in result.error.lower()


def test_hub_probe_timeout():
    hub = HubLLM(client_for(delayed(1.0, response("OK")))[0], model="slow-model")
    result = hub.probe(timeout_s=0.2, slow_s=8.0)
    assert result.ok is False
    assert result.error and "timed out" in result.error


def test_humanize_hub_error_auth():
    exc = RuntimeError("Incorrect API key")
    exc.status_code = 401  # type: ignore[attr-defined]
    assert "auth" in humanize_hub_error(exc).lower()


def test_sandbox_llm_detector_with_compat():
    src = '''
def detect(item, llm_client=None):
    from autofa.llm.compat import chat_completions_safe
    text = item.get("response_text") or ""
    if llm_client is None:
        return "unknown"
    r = chat_completions_safe(
        llm_client,
        model="gcp/google/gemini-3.5-flash-lite",
        messages=[{"role": "user", "content": text}],
        temperature=0.0,
        max_tokens=32,
    )
    return (r.choices[0].message.content or "").strip()
'''
    compile_detector(src)
    llm = HubLLM(client_for(response("abandoned_plan"))[0])
    out = run_detector(src, {"response_text": "hello"}, llm=llm)
    assert out == "abandoned_plan"


def test_configured_api_key_env_wins_over_inference_api_key_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INFERENCE_HUB_API_KEY", "nvapi-configured")
    monkeypatch.setenv("INFERENCE_API_KEY", "nvapi-fallback")
    assert resolve_api_key_env("INFERENCE_HUB_API_KEY") == "INFERENCE_HUB_API_KEY"
    assert get_api_key("INFERENCE_HUB_API_KEY") == "nvapi-configured"
    assert resolve_api_key_env("MY_OWN_KEY") is None
    # Agent sandboxes export the hub key as INFERENCE_API_KEY.
    monkeypatch.delenv("INFERENCE_HUB_API_KEY")
    assert resolve_api_key_env("INFERENCE_HUB_API_KEY") == "INFERENCE_API_KEY"
    assert get_api_key("INFERENCE_HUB_API_KEY") == "nvapi-fallback"


def test_missing_api_key_error_names_every_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    monkeypatch.delenv("INFERENCE_HUB_API_KEY", raising=False)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    assert resolve_api_key_env("INFERENCE_HUB_API_KEY") is None
    with pytest.raises(RuntimeError, match="INFERENCE_HUB_API_KEY / INFERENCE_API_KEY"):
        make_hub_client(base_url="https://hub.example/v1", api_key_env="INFERENCE_HUB_API_KEY")


def test_no_endpoint_is_built_in(monkeypatch: pytest.MonkeyPatch):
    from autofa.config import Config
    from autofa.llm.endpoints import NO_ENDPOINT_HELP, endpoint_for_config, endpoint_setup_help

    monkeypatch.setenv("INFERENCE_HUB_API_KEY", "set-but-unused")
    cfg = Config()
    endpoint = endpoint_for_config(cfg)
    assert (endpoint.base_url, endpoint.api_key_env) == ("", "")
    assert (endpoint_for_config(cfg, "judge").base_url,
            endpoint_for_config(cfg, "judge").api_key_env) == ("", "")
    assert endpoint_setup_help(cfg) == NO_ENDPOINT_HELP
    with pytest.raises(RuntimeError, match="No LLM endpoint is configured"):
        make_hub_client()


def test_hub_client_loads_shared_key_outside_cli(monkeypatch, tmp_path):
    import openai

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / ".env").write_text("INFERENCE_HUB_API_KEY=shared-test-key\n")
    worktree = tmp_path / "new-worktree"
    worktree.mkdir()
    (worktree / ".env").write_text("UNRELATED_SETTING=1\n")
    monkeypatch.chdir(worktree)
    monkeypatch.setenv("AUTOFA_HOME", str(profile))
    monkeypatch.delenv("INFERENCE_HUB_API_KEY", raising=False)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_SETTING", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: calls.append(kw) or object())
    make_hub_client(base_url="https://hub.example/v1", api_key_env="INFERENCE_HUB_API_KEY")
    assert calls == [{"api_key": "shared-test-key", "base_url": "https://hub.example/v1"}]


# Reasoning-model responses must not silently change detector labels.


@pytest.mark.parametrize("content", [
    "no", "<thinking>yes is incorrect</thinking><final>no</final>",
    [{"type": "reasoning", "text": "yes"}, {"type": "text", "text": "no"}],
])
def test_reasoning_is_excluded_before_sandbox_detector_compares_label(content):
    client, _ = client_for(response(content, reasoning_content="yes", reasoning="yes"))
    source = '''
def detect(item, llm_client=None):
    answer = llm_client.classify("Reply yes or no", item["response_text"])
    return answer == "yes"
'''
    llm = HubLLM(client, api="chat_completions", model="openai/openai/gpt-6-luna")
    assert run_detector(source, {"response_text": "The test failed."}, llm=llm) is False


@pytest.mark.parametrize("content", [
    "<think>yes", "<think>yes</think>", "<final>yes", [{"type": "thinking", "text": "yes"}],
])
def test_missing_final_answer_is_an_error(content):
    with pytest.raises(RuntimeError):
        completion_text(response(content))


@pytest.mark.parametrize("raw", ["NO", '"no".', "<think>yes?</think>no"])
def test_classify_options_matches_only_a_complete_label(raw):
    client, _ = client_for(response(raw))
    assert HubLLM(client, api="chat_completions").classify_options("evidence", "question") == "no"


@pytest.mark.parametrize("raw", ["yesterday", "Yes, but actually no."])
def test_classify_options_rejects_substring_matches_and_unmarked_explanations(raw):
    client, _ = client_for(response(raw), response(raw))
    with pytest.raises(RuntimeError, match="exactly one"):
        HubLLM(client, api="chat_completions").classify_options("evidence", "question")


# Overlapping options are not matched by substring; an exact option wins over punctuation.
@pytest.mark.parametrize("raw,options", [
    ("not applicable", ["no", "not applicable"]), ("no!", ["no", "no!"]),
])
def test_custom_options_match_the_complete_exact_label(raw, options):
    client, _ = client_for(response(raw))
    assert HubLLM(client, api="chat_completions").classify_options(
        "evidence", "question", options) == raw


def test_json_and_evidence_tags_survive_reasoning_cleanup_and_localisation():
    answer = json.dumps({"quote": "<think>literal evidence</think>", "label": "no"})
    client, calls = client_for(response("<think>plan</think>\n" + answer))
    llm = HubLLM(client, api="chat_completions", request_options={"reasoning_effort": "low"})
    assert json.loads(llm.localise("Extract JSON", "evidence")) == json.loads(answer)
    assert calls[0]["reasoning_effort"] == "low"


@pytest.mark.parametrize("operation,budget", [("probe", 8), ("classify", 64)])
def test_default_budget_retries_once_for_reasoning_models(operation, budget):
    # Includes a plausible partial label: it must not be accepted or cached.
    client, calls = client_for(response("yes", "length"), response("no"))
    llm = HubLLM(client, api="chat_completions", model="arbitrary-reasoning-model")
    if operation == "probe":
        assert llm.probe().ok
    else:
        assert getattr(llm, operation)("question", "evidence") == "no"
    assert [call["max_tokens"] for call in calls] == [budget, 4096]
    assert calls[0]["messages"] == calls[1]["messages"]
    assert calls[0]["model"] == calls[1]["model"] == "arbitrary-reasoning-model"
    if operation == "classify":
        assert llm.classify("question", "evidence") == "no"
        assert len(calls) == 2


def test_retry_is_bounded_and_incomplete_answers_are_not_cached():
    client, calls = client_for(response("yes", "length"), response("yes", "length"),
                               response("yes", "length"), response("no"))
    llm = HubLLM(client, api="chat_completions")
    with pytest.raises(RuntimeError, match="incomplete"):
        llm.classify("question", "evidence")
    assert llm.classify("question", "evidence") == "no"
    assert [call["max_tokens"] for call in calls] == [64, 4096, 8192, 4096]
    assert llm.classify("question", "evidence") == "no"
    assert len(calls) == 4


def test_learned_default_budget_is_shared_by_all_helpers():
    client, calls = client_for(response("yes", "length"), *[response("no") for _ in range(5)])
    llm = HubLLM(client, api="chat_completions")
    assert llm.localise("first question", "first evidence") == "no"
    assert llm.probe().ok
    assert llm.classify("second question", "evidence") == "no"
    assert llm.classify_options("evidence", "question") == "no"
    assert llm.localise("Extract quotes", "evidence") == "no"
    assert [call["max_tokens"] for call in calls] == [1200, 4096, 4096, 4096, 4096, 4096]


def test_learned_budget_does_not_invalidate_completed_answer_cache():
    client, calls = client_for(response("no"), response("", "length"), response("OK"))
    llm = HubLLM(client, api="chat_completions")
    assert llm.classify("question", "evidence") == "no"
    assert llm.probe().ok
    assert llm.classify("question", "evidence") == "no"
    assert [call["max_tokens"] for call in calls] == [64, 8, 4096]


def test_rejected_larger_budget_does_not_poison_later_requests():
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if kwargs["max_tokens"] > 2048:
            error = RuntimeError("max_tokens exceeds the endpoint maximum of 2048")
            error.status_code = 400
            raise error
        return response("", "length") if len(calls) == 1 else response("no")

    llm = HubLLM(NS(chat=NS(completions=NS(create=complete))))
    with pytest.raises(RuntimeError, match="endpoint maximum"):
        llm.classify("question", "evidence")
    assert llm.classify("question", "evidence") == "no"
    assert llm.classify("question", "evidence") == "no"
    assert [call["max_tokens"] for call in calls] == [64, 4096, 64]


def test_learned_budget_is_isolated_between_adapters_and_models():
    client, calls = client_for(response("", "length"), response("no"),
                               response("no"), response("no"))
    for model in ("first", "first", "second"):
        llm = HubLLM(client, api="chat_completions", model=model)
        assert llm.classify_options("evidence", "question") == "no"
    assert [call["max_tokens"] for call in calls] == [64, 4096, 64, 64]


def test_concurrent_truncations_share_learned_budget_without_serializing_requests():
    calls = []
    lock = threading.Lock()
    small_requests = threading.Barrier(2)

    def complete(**kwargs):
        with lock:
            calls.append(kwargs)
        if kwargs["max_tokens"] == 64:
            small_requests.wait(timeout=5)
            return response("yes", "length")
        return response("no")

    llm = HubLLM(NS(chat=NS(completions=NS(create=complete))))
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(lambda text: llm.classify("question", text), ["one", "two"]))
    assert answers == ["no", "no"]
    assert llm.localise("Extract quotes", "evidence") == "no"
    assert llm.classify_options("new evidence", "question") == "no"
    assert [call["max_tokens"] for call in calls] == [64, 64, 4096, 4096, 4096, 4096]


def test_learned_budget_preserves_per_call_limit_and_rejects_further_truncation():
    client, calls = client_for(response("", "length"), response("no"),
                               response("yes", "length"), response("yes", "length"),
                               response("yes", "length"))
    llm = HubLLM(client, api="chat_completions")
    assert llm.classify_options("first", "question") == "no"
    with pytest.raises(RuntimeError, match="incomplete"):
        llm.classify_options("second", "question", max_tokens=17)
    with pytest.raises(RuntimeError, match="incomplete"):
        llm.classify_options("third", "question")
    assert [call["max_tokens"] for call in calls] == [64, 4096, 17, 4096, 8192]


def test_explicit_request_limit_overrides_previously_learned_budget():
    client, calls = client_for(response("", "length"), response("no"),
                               response("yes", "length"))
    llm = HubLLM(client, api="chat_completions")
    assert llm.classify_options("first", "question") == "no"
    llm.request_options["max_completion_tokens"] = 17
    with pytest.raises(RuntimeError, match="incomplete"):
        llm.localise("Extract quotes", "evidence")
    assert len(calls) == 3
    assert calls[-1]["max_completion_tokens"] == 17
    assert "max_tokens" not in calls[-1]


@pytest.mark.parametrize("operation", ["probe", "classify"])
def test_explicit_budget_is_never_increased(operation):
    client, calls = client_for(response("", "length"))
    llm = HubLLM(client, api="chat_completions", request_options={"max_tokens": 17})
    if operation == "probe":
        assert not llm.probe().ok
    else:
        with pytest.raises(RuntimeError, match="incomplete"):
            getattr(llm, operation)("question", "evidence")
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 17


@pytest.mark.parametrize("operation,limit", [("probe", 4), ("probe", 17), ("localise", 17)])
def test_constructor_limit_is_a_hard_cap_for_every_helper(operation, limit):
    client, calls = client_for(response("", "length"))
    llm = HubLLM(client, api="chat_completions", max_tokens=limit)
    if operation == "probe":
        assert not llm.probe().ok
    else:
        with pytest.raises(RuntimeError, match="incomplete"):
            getattr(llm, operation)("question", "evidence")
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == (min(8, limit) if operation == "probe" else limit)


# Classifier windows and crops keep omission provenance, so analyzer omissions are never
# mistaken for source defects.

OMISSION = "…[middle omitted]…"

# Same extraction/JSON/classifier shape as the observed detector at transcript
# line 925. The fixture substitutes the response field for the selected final step.
WINDOW_DETECTOR = '''
import json
from autofa.trace import text_window

def detect(item, llm_client=None):
    text = json.dumps({
        'reasoning_end': text_window(item['reasoning_text'], focus='tail', max_chars=5000),
        'final_reply': text_window(item['response_text'], focus='ends', max_chars=10000),
    }, ensure_ascii=ASCII)
    return llm_client.classify_options(
        text, 'Does the final reply contradict the plan?',
        options=['contradicts_plan', 'consistent'], focus='full', max_chars=len(text),
        require_complete=REQUIRE_COMPLETE,
    ) == 'contradicts_plan'
'''

# The attempted repair at transcript line 948 returned a resolved negative when
# the complete final reply exceeded its chosen evidence budget.
BUDGET_DETECTOR = '''
import json
from autofa.trace import steps, text_window

def detect(item, llm_client=None):
    finals = [s for s in steps(item) if s.source == 'assistant' and s.message.strip()]
    fm = finals[-1].message
    if len(fm) > 15000:
        return False
    text = json.dumps({'final_reply': fm}, ensure_ascii=False)
    return llm_client.classify_options(
        text, 'Does the complete reply contradict the plan?',
        options=['contradicts_plan', 'consistent'], focus='full', max_chars=len(text),
    ) == 'contradicts_plan'
'''


def fake_llm(monkeypatch, reply="consistent", **kwargs):
    llm = HubLLM(NS(chat=NS(completions=NS())), **kwargs)
    calls = []

    def complete(messages, options, **kw):
        calls.append(messages)
        return reply

    monkeypatch.setattr(llm, "_complete_text", complete)
    return llm, calls


def fragments(prepared):
    return json.loads(prepared.split("--- exact source fragments ---\n", 1)[1])


@pytest.mark.parametrize("focus", ["tail", "ends"])
def test_structured_window_separates_omissions_from_exact_source(focus):
    source = "START α\\n'quoted' " + "valid code " * 200 + " END"
    evidence = text_window_evidence(source, focus=focus, max_chars=200)
    assert not evidence["complete"]
    metadata = evidence["analyzer_window_metadata"]
    assert len(metadata) == 1 and metadata[0]["original_chars"] == len(source)
    omitted = metadata[0]["omitted_ranges"]
    spans = [(p["start"], p["end"]) for p in evidence["source_fragments"]]
    assert sorted([*omitted, *[list(s) for s in spans]])[0][0] == 0
    assert sum(b - a for a, b in [*omitted, *spans]) == len(source)
    for piece in evidence["source_fragments"]:
        assert piece["text"] == source[piece["start"]:piece["end"]]
        assert OMISSION not in piece["text"]


@pytest.mark.parametrize("ascii_only", [True, False])
def test_legacy_json_windows_keep_provenance_in_classifier(monkeypatch, ascii_only):
    llm, calls = fake_llm(monkeypatch)
    source = WINDOW_DETECTOR.replace("ASCII", str(ascii_only)).replace("REQUIRE_COMPLETE", "False")
    reply = "const text = 'α🙂\\\\n';\n" * 1500
    result = run_detector(source, {"response_text": reply, "reasoning_text": "Write complete code"},
                          llm=llm)
    assert result is False  # Excerpts can still support an ordinary decision.
    prompt = calls[0][0]["content"]
    assert OMISSION not in prompt
    assert "__autofa_insufficient_evidence__" in prompt
    actual = [p for p in fragments(prompt) if p["text"].startswith("const")
              or p["text"].endswith("';\n")]
    assert len(actual) == 2
    for piece in actual:
        assert piece["text"] == reply[piece["start"]:piece["end"]]
    assert f'"original_chars": {len(reply)}' in prompt


def test_require_complete_abstains_before_dispatch_for_json_window(monkeypatch):
    llm, calls = fake_llm(monkeypatch)
    source = WINDOW_DETECTOR.replace("ASCII", "True").replace("REQUIRE_COMPLETE", "True")
    with pytest.raises(DetectorAbstained, match="selected text window"):
        run_detector(source, {"response_text": "valid code " * 2000,
                              "reasoning_text": "Write complete code"}, llm=llm)
    assert calls == []


@pytest.mark.parametrize("method", ["classify", "classify_options"])
def test_incomplete_evidence_has_an_abstention_answer_not_forced_negative(monkeypatch, method):
    llm, calls = fake_llm(monkeypatch, "__autofa_insufficient_evidence__")
    with pytest.raises(DetectorAbstained):
        if method == "classify":
            llm.classify("Is this code complete?", "valid code " * 200, max_chars=200)
        else:
            llm.classify_options("valid code " * 200, "Is this code complete?", max_chars=200)
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["classify", "decide"])
def test_require_complete_catches_hub_crop_before_dispatch(monkeypatch, method):
    llm, calls = fake_llm(monkeypatch)
    with pytest.raises(DetectorAbstained):
        if method == "classify":
            llm.classify("Question", "x" * 1000, max_chars=200, require_complete=True)
        else:
            llm.decide([{"type": "predicate", "instructions": "Question"}], "x" * 1000,
                       max_chars=200, require_complete=True)
    assert calls == []


def test_decisions_classifier_maps_insufficient_evidence_to_abstention(monkeypatch):
    llm, _ = fake_llm(monkeypatch, api="decisions")
    calls = []

    def decide(client, **kwargs):
        calls.append(kwargs)
        return [{"type": "choice", "choice": "__autofa_insufficient_evidence__"}]

    monkeypatch.setattr("autofa.llm.decisions.create_decision", decide)
    with pytest.raises(DetectorAbstained):
        llm.classify_options("x" * 1000, "Is the original code incomplete?", max_chars=200)
    assert len(calls) == 1
    assert OMISSION not in calls[0]["input"]
    assert "__autofa_insufficient_evidence__" in {
        c["value"] for c in calls[0]["questions"][0]["choices"]
    }


def test_copying_window_strings_preserves_their_receipts_without_detector_scope():
    window = deepcopy(text_window("x" * 2000, max_chars=200))
    assert isinstance(window, str) and len(window) == 200
    with pytest.raises(DetectorAbstained):
        _classify_evidence(window, require_complete=True)


def test_receipt_capacity_fails_explicitly_and_resets_after_item(monkeypatch):
    import autofa.llm.windowing as windowing

    monkeypatch.setattr(windowing, "_MAX_RECEIPTS", 2)
    with pytest.raises(ValueError, match="provenance budget exceeded"), window_provenance():
        for char in "abc":
            text_window(char * 2000, max_chars=200)
    with window_provenance():
        # An exhausted call cannot contaminate the next detector's receipt scope.
        assert len(text_window("d" * 2000, max_chars=200)) == 200


@pytest.mark.parametrize("focus", ["head", "tail", "ends"])
def test_second_hub_crop_preserves_inner_source_spans(focus):
    original = "const α = '🙂\\n';\n" * 200
    with window_provenance():
        selected = text_window(original, focus="ends", max_chars=600)
        packed = json.dumps({"reply": selected}, ensure_ascii=True)
        prepared = _classify_evidence(packed, focus=focus, max_chars=250)
    assert prepared.incomplete
    assert OMISSION not in prepared.text
    evidence = fragments(prepared.text)
    # Some wrapper characters are exact spans of the supplied serialized JSON;
    # the remaining pieces must still be exact spans of the original reply.
    for piece in evidence:
        assert piece["text"] in (
            original[piece["start"]:piece["end"]],
            packed[piece["start"]:piece["end"]],
        )


def test_window_scope_does_not_leak_to_the_next_item(monkeypatch):
    llm, calls = fake_llm(monkeypatch)
    source = '''
from autofa.trace import text_window
def detect(item, llm_client=None):
    return str(text_window(item['response_text'], focus='ends', max_chars=200))
'''
    selected = run_detector(source, {"response_text": "x" * 2000}, llm=llm)
    assert OMISSION in selected
    prepared = _classify_evidence(selected, focus="full", max_chars=len(selected),
                                  require_complete=True)
    assert not prepared.incomplete and prepared.text == selected
    assert calls == []


def test_literal_marker_inside_known_window_remains_exact_source():
    original = OMISSION + " original " * 100 + "DONE"
    prepared = _classify_evidence(text_window(original, max_chars=200))
    pieces = fragments(prepared.text)
    assert sum(p["text"].count(OMISSION) for p in pieces) == 1


def test_nested_json_packing_retains_omission_provenance():
    original = "start\nα🙂" + "MIDDLE" * 80 + "\nend"
    with window_provenance():
        text = text_window(original, focus="ends", max_chars=100)
        for ascii_only in (True, False, True):
            text = json.dumps({"payload": text}, ensure_ascii=ascii_only)
        prepared = _classify_evidence(text, focus="full", max_chars=len(text))
        with pytest.raises(DetectorAbstained):
            _classify_evidence(text, focus="full", max_chars=len(text), require_complete=True)
    assert prepared.incomplete
    assert all(OMISSION not in piece["text"] for piece in fragments(prepared.text))


@pytest.mark.parametrize("condition", ["15000 < len(fm)", "len(fm) >= budget"])
def test_observed_budget_false_branch_gets_actionable_advisory(condition):
    source = BUDGET_DETECTOR.replace("len(fm) > 15000", condition)
    assert lint_detector_source(source)[0]
    warnings = evidence_budget_warnings(source)
    assert len(warnings) == 1
    assert "library.abstain" in warnings[0] and "expected_abstentions" in warnings[0]


def test_budget_abstention_and_semantic_guards_remain_usable():
    assert not evidence_budget_warnings(
        BUDGET_DETECTOR.replace("return False", "library.abstain('reply exceeds budget')")
    )
    # Content conjunction, word-count semantics, and unrelated collections are
    # deliberately not interpreted as blanket evidence-budget rejection.
    for condition in ("len(fm) > 15000 and 'short reply required' in fm",
                      "len(fm.split()) > 12", "len(finals) > 5"):
        source = BUDGET_DETECTOR.replace("len(fm) > 15000", condition)
        assert lint_detector_source(source)[0]
        assert not evidence_budget_warnings(source)
    # A small semantic length predicate is advisory, never rejected.
    assert lint_detector_source(BUDGET_DETECTOR.replace("len(fm) > 15000", "len(fm) > 12"))[0]
