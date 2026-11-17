# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passage provenance, label compatibility, persistence, and judge separation."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.dependencies import build_bundle
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import (
    CONTEXT_CHARS,
    MAX_PASSAGES,
    MAX_QUOTE_CHARS,
    resolve_deferred_localisation,
    text_hash,
    validate_localisation,
)
from autofa.core.sandbox import run_detector
from autofa.data.parquet_store import Dataset
from autofa.data.trace_fields import derive_trace_fields
from autofa.mcp import tools_eval
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics import claude_judge
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot
from autofa.reports.failure_cards import build_data

REGEX_SOURCE = """
import re
def detect(item, llm_client=None):
    return bool(re.search(r"I guessed", item.get("response_text", "")))
"""
LLM_SOURCE = """
def detect(item, llm_client=None):
    return llm_client.classify("Is the answer guessed?", item["response_text"],
                               focus="tail", max_chars=9000) == "yes"
"""


def _run(source, item, llm=None, **kwargs):
    record = {"stale": True}
    result = run_detector(source, item, llm, localisation=record, **kwargs)
    return result, record


def _assert_coordinates(item, record):
    fields = derive_trace_fields(item)
    for p in record["passages"]:
        value = fields[p["field"]]
        assert value[p["start"] : p["end"]] == p["quote"]
        assert p["field_sha256"] == text_hash(value)
        assert p["start_line"] == value[: p["start"]].count("\n") + 1
        assert p["start_column"] == p["start"] - value.rfind("\n", 0, p["start"])
        assert value[p["context_start"] : p["context_end"]] == p["context"]
        assert p["context_start"] <= p["start"] < p["end"] <= p["context_end"]
        assert p["start"] - p["context_start"] <= CONTEXT_CHARS
        assert p["context_end"] - p["end"] <= CONTEXT_CHARS
        assert p["field_chars"] == len(value)


@pytest.mark.parametrize("kind", ["regex", "llm"])
def test_context_for_both_detector_types_keeps_highlight_and_repeated_unicode_offsets(kind):
    text = ("before " * 30 + "🙂é\nI guessed <answer>. " + "after " * 30
            + "🙂 I guessed <answer>. End.")
    item = {"response_text": text}
    source = LLM_SOURCE if kind == "llm" else REGEX_SOURCE.replace(
        're.search(r"I guessed"', 're.findall(r"I guessed"'
    )
    client = DetectorLLM()
    _, record = _run(source, item, client)
    record = resolve_deferred_localisation(record, item, source, client)
    assert len(record["passages"]) == 2
    _assert_coordinates(item, record)
    assert record["passages"][0]["context_start"] > 0
    for p in record["passages"]:
        assert "🙂" in p["context"] and "<answer>" in p["context"]
        assert p["context"] != p["quote"]
        assert len(text[p["context_start"] : p["start"]].split()) <= 12
        assert len(text[p["end"] : p["context_end"]].split()) <= 12


def test_prompt_parsing_and_envelope_copies_do_not_displace_the_final_answer():
    source = '''
import re
def detect(item):
    lang = re.search(r"into (\\w+)", item["prompt_text"]).group(1).lower()
    if "japanese" not in lang:
        return False
    return bool(re.search("People Swimming in the Swimming Pool", item["final_message"]))
'''
    quote = "People Swimming in the Swimming Pool"
    final = "ここに " + quote + " を翻訳します。"
    item = {
        "prompt_text": "Translate into Japanese",
        "response_text": json.dumps({"steps": [{"source": "assistant", "message": final}],
                                     "copy": final}),
        "final_message": final,
        "reasoning_text": "The target is Japanese.",
    }
    label, record = _run(source, item)
    assert label is True
    assert [(p["field"], p["quote"]) for p in record["passages"]] == [("final_message", quote)]
    _assert_coordinates(item, record)


def test_direct_case_transformed_and_prompt_derived_surfaces_keep_their_field():
    source = '''
import re
def detect(item):
    src = re.split("SOURCE: ", item["prompt_text"])[1]
    re.findall(r"\\(.*?\\)", src)
    text = item["final_message"][2:].lower()
    return bool(re.search("bad phrase", text))
'''
    item = {"prompt_text": "SOURCE: quoted (10 January)",
            "reasoning_text": "quoted (10 January)", "final_message": "🙂 BAD PHRASE. End."}
    _, record = _run(source, item)
    assert [(p["field"], p["quote"]) for p in record["passages"]] == [
        ("final_message", "BAD PHRASE")
    ]
    _assert_coordinates(item, record)


@pytest.mark.parametrize("mode", ["sentences", "windows", "count"])
def test_repetition_early_returns_have_real_passages_without_regex_hits(mode):
    repeated = "これは何度も繰り返される長い文章です"
    text = (repeated + "。") * 4
    body = {
        "sentences": '''
    seen = {}
    for part in re.split(r"[。.!?\\n]+", text):
        s = part.strip()
        if len(s) >= 12:
            seen[s] = seen.get(s, 0) + 1
            if seen[s] >= 3:
                return True
''',
        "windows": '''
    counts = {}
    for i in range(0, len(text) - 12):
        s = text[i:i+12]
        counts[s] = counts.get(s, 0) + 1
        if counts[s] >= 3:
            return True
''',
        "count": f"\n    if text.count({repeated!r}) >= 3:\n        return True\n",
    }[mode]
    source = ('import re\ndef detect(item):\n    text = item["final_message"]\n' + body
              + '    return bool(re.search("missing", text))\n')
    item = {"final_message": text}
    label, record = _run(source, item)
    assert label is True and label == run_detector(source, item)
    assert record["status"] == "located"
    assert len(record["passages"]) >= 3
    assert all(p["method"] == "repetition" and p["field"] == "final_message"
               for p in record["passages"])
    _assert_coordinates(item, record)
    assert _run(source, {"final_message": repeated}) == (False, {})


def test_aggregate_counts_do_not_invent_repeated_evidence_and_subscripts_keep_semantics():
    source = '''
def detect(item):
    counts = {"absent": 3}
    return counts["absent"] == 3 and [1, 2, 1].count(1) == 2 and "abc"[1:] == "bc"
'''
    label, record = _run(source, {"response_text": "not repeated"})
    assert label is True and record["status"] == "unavailable"


@pytest.mark.parametrize(
    "expression",
    [
        're.compile("I guessed").search(text)',
        're.findall("(I) (guessed)", text)',
        're.match("I guessed", text[3:])',
        '"i guessed" in text.lower()',
    ],
)
def test_regex_and_literal_matches_keep_verified_unicode_coordinates(expression):
    item = {"_item_id": "one", "response_text": "🙂é\nI guessed.\nThen I guessed again."}
    source = (
        f"import re\ndef detect(item):\n text = item['response_text']\n return bool({expression})"
    )
    label, record = _run(source, item)
    assert label is True
    assert record["status"] == "located"
    _assert_coordinates(item, record)
    first = record["passages"][0]
    assert (first["start"], first["end"]) == (3, 12)
    assert (first["start_line"], first["start_column"], first["end_column"]) == (2, 1, 10)


@pytest.mark.parametrize("compiled,pattern", [
    (False, "(a)(z)?"), (False, "(?=a)"), (True, "(z)?a"),
])
def test_findall_keeps_group_and_zero_width_semantics(compiled, pattern):
    call = (
        f"re.compile({pattern!r}).findall(text)" if compiled else f"re.findall({pattern!r}, text)"
    )
    source = f"import re\ndef detect(item):\n text = item['response_text']\n return {call}"
    item = {"response_text": "aa az"}
    assert _run(source, item)[0] == run_detector(source, item)


@pytest.mark.parametrize("compiled,positional,concurrent", [
    (False, False, True), (True, True, False),
])
def test_findall_explicit_concurrency_keeps_capture_coordinates(compiled, positional, concurrent):
    if compiled:
        call = 're.compile("(I) (guessed)").findall(text'
        optional = f", None, None, False, {concurrent!r}"
    else:
        call = 're.findall("(I) (guessed)", text'
        optional = f", 0, None, None, False, {concurrent!r}"
    call += (optional if positional else f", concurrent={concurrent!r}") + ")"
    source = f"import re\ndef detect(item):\n text = item['response_text']\n return {call}"
    item = {"response_text": "🙂é\nI guessed.\nThen I guessed again."}
    label, record = _run(source, item)
    assert label == [("I", "guessed"), ("I", "guessed")]
    assert record["status"] == "located"
    _assert_coordinates(item, record)
    assert [(p["start"], p["end"]) for p in record["passages"]] == [(3, 12), (19, 28)]


def test_match_occurrences_are_not_collapsed_and_negative_results_drop_evidence():
    source = (
        "import re\ndef detect(item):\n"
        ' return bool(list(re.finditer("guess", item["response_text"])))'
    )
    item = {"response_text": "guess then guess"}
    _, record = _run(source, item)
    assert [(p["start"], p["end"]) for p in record["passages"]] == [(0, 5), (11, 16)]
    negative = source.replace("return bool(", "return not bool(")
    assert _run(negative, item) == (False, {})
    assert _run(REGEX_SOURCE, {"response_text": "clean"}) == (False, {})


def test_absence_rules_report_unavailable_instead_of_fabricating_a_passage():
    label, record = _run(
        'def detect(item): return "proof" not in item["response_text"]', {"response_text": "42"}
    )
    assert label is True and record["status"] == "unavailable"
    assert record["passages"] == [] and record["notes"]


@pytest.mark.parametrize("expression", [
    're.search("email", observation)',
    '"email" in observation',
])
def test_rich_trace_capture_keeps_the_selected_padded_observation(expression):
    observation = "  email: synthetic@example.test\n"
    item = {"response_text": json.dumps({
        "steps": [
            {"source": "tool", "observation": observation},
            {"source": "tool", "observation": observation},
        ],
        "execution": {"events": [{
            "event_idx": 99, "lane": "auxiliary", "parts": [
                {"part_idx": 0, "text": observation},
            ],
        }]},
    })}
    source = (
        "import re\nfrom autofa.trace import steps\ndef detect(item):\n"
        "    observation = steps(item)[1].observation\n"
        f"    return bool({expression})\n"
    )
    label, record = _run(source, item)
    assert label is True and label == run_detector(source, item)
    assert record["status"] == "located"
    evidence, = record["passages"]
    location, = evidence["trace_locations"]
    assert evidence["quote"] == "email"
    assert location["step_index"] == 2
    assert "source_event" not in location
    _assert_coordinates(item, record)


def test_source_only_spans_do_not_disable_capture_with_native_text_surfaces():
    item = {
        "actions_text": "Native action summary",
        "observations_text": "Native observation summary",
        "response_text": json.dumps({
            "steps": [{"source": "tool", "observation": "Account lookup: email available"}],
            "execution": {"events": [{
                "event_idx": 99, "lane": "auxiliary", "parts": [
                    {"part_idx": 0, "text": "Auxiliary source content"},
                ],
            }]},
        }),
    }
    source = '''
import re
from autofa.trace import steps
def detect(item):
    return bool(re.search("email", steps(item)[0].observation))
'''
    label, record = _run(source, item)
    assert label is True and record["status"] == "located"
    evidence, = record["passages"]
    assert evidence["trace_locations"][0]["step_index"] == 1
    _assert_coordinates(item, record)


@pytest.mark.parametrize("stage", ["match", "navigation"])
def test_evidence_processing_failure_preserves_label_with_unavailable_evidence(monkeypatch, stage):
    from autofa.core.localisation import PassageCapture
    from autofa.data.trace_navigation import TraceNavigation

    def unavailable(*args, **kwargs):
        raise ValueError("bad evidence mapping")

    if stage == "match":
        monkeypatch.setattr(PassageCapture, "regex_match", unavailable)
    else:
        monkeypatch.setattr(TraceNavigation, "attach", unavailable)
    item = {"_item_id": "synthetic", "response_text": "I guessed"}
    label, record = _run(REGEX_SOURCE, item)
    assert label is True and label == run_detector(REGEX_SOURCE, item)
    assert record["status"] == "unavailable"
    assert record["passages"] == [] and record["truncated"]
    assert record["item_id"] == "synthetic"
    assert any("ValueError" in note for note in record["notes"])
    assert _run(REGEX_SOURCE, {"response_text": "Verified"}) == (False, {})


def test_capture_does_not_suppress_detector_errors_or_interrupts(monkeypatch):
    from autofa.core.localisation import PassageCapture

    with pytest.raises(ValueError, match="detector error"):
        _run('def detect(item): raise ValueError("detector error")', {})

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(PassageCapture, "regex_match", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(REGEX_SOURCE, {"response_text": "I guessed"})


def test_passage_limits_and_derived_coordinates_past_old_trace_caps():
    source = (
        'import re\ndef detect(item): return bool(re.search("I guessed", item["reasoning_text"]))'
    )
    item = {"response_text": json.dumps({"steps": [{
        "source": "assistant", "reasoning_content": "x" * 230_000 + "I guessed",
        "message": "Answer: 42",
    }]})}
    _, record = _run(source, item)
    _assert_coordinates(item, record)
    assert any(p["field"] == "reasoning_text" and p["start"] > 230_000 for p in record["passages"])
    _, many = _run(
        'import re\ndef detect(item): return bool(re.findall("x", item["response_text"]))',
        {"response_text": "x" * 20},
    )
    assert len(many["passages"]) == MAX_PASSAGES and many["truncated"]
    _, wide = _run(
        'import re\ndef detect(item): return bool(re.search("x+", item["response_text"]))',
        {"response_text": "x" * 2000},
    )
    assert len(wide["passages"][0]["quote"]) == MAX_QUOTE_CHARS and wide["truncated"]


def test_dependency_matches_and_concurrent_captures_are_isolated():
    child = 'def detect(item, llm_client=None): return library.fire("parent", item)'
    parent = (
        'import re\ndef detect(item): return bool(re.search(r"BAD\\d+", item["response_text"]))'
    )
    library = FunctionLibrary(functions=[
        FeatureFunction(name="parent", description="parent", source=parent),
    ])
    bundle = build_bundle(library, depends_on=["parent"])

    def call(i):
        return _run(child, {"_item_id": str(i), "response_text": f"BAD{i}"}, bundle=bundle)

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(call, range(40)))
    for i, (label, record) in enumerate(records):
        assert label is True and record["item_id"] == str(i)
        assert [p["quote"] for p in record["passages"]] == [f"BAD{i}"]


class DetectorLLM:
    def __init__(self, quote="I guessed", error=None):
        self.quote, self.error = quote, error
        self.classifications = []
        self.extractions = []

    def classify(self, prompt, text, **kwargs):
        self.classifications.append((prompt, text, kwargs))
        return "yes" if "I guessed" in text else "no"

    def localise(self, prompt, text, **kwargs):
        self.extractions.append((prompt, text, kwargs))
        if self.error:
            raise self.error
        return json.dumps({"quotes": [self.quote]})


def test_option_classifier_images_are_not_stored_or_replayed_for_navigation():
    class VisualLLM(DetectorLLM):
        def classify_options(self, text, question, **kwargs):
            self.classifications.append((question, text, kwargs))
            return "yes"

    source = '''
def detect(item, llm_client=None):
    return llm_client.classify_options(
        item["response_text"], "Does the image support the guess?", options=["yes", "no"],
        focus="full", max_chars=500,
        images=[{"status": "available", "data_url": "data:image/png;base64,c3ludGhldGlj"}],
    ) == "yes"
'''
    client = VisualLLM()
    item = {"response_text": "I guessed"}
    label, record = _run(source, item, client)
    assert label is True and not client.extractions
    assert "base64" not in json.dumps(record)
    resolved = resolve_deferred_localisation(record, item, source, client)
    assert resolved["status"] == "located"
    assert "images" not in client.extractions[0][2]


def test_llm_scoring_defers_navigation_without_changing_classification_or_window():
    client = DetectorLLM()
    item = {"_item_id": "one", "response_text": "Unclear. I guessed. Done."}
    label, record = _run(LLM_SOURCE, item, client)
    assert label is True and len(client.classifications) == 1 and not client.extractions
    assert client.classifications[0][2] == {"focus": "tail", "max_chars": 9000}
    assert record["deferred_localisation"]["status"] == "pending"
    resolved = resolve_deferred_localisation(record, item, LLM_SOURCE, client)
    _assert_coordinates(item, resolved)
    assert resolved["passages"][0]["method"] == "post_scoring_navigation"
    assert resolved["origin"] == "post_scoring_navigation"
    assert resolved["assigned_label"] == record["assigned_label"]
    assert (resolved["deferred_localisation"]["request_hash"]
            == record["deferred_localisation"]["request_hash"])
    assert resolve_deferred_localisation(resolved, item, LLM_SOURCE, client) == resolved
    assert len(client.classifications) == len(client.extractions) == 1
    assert _run(LLM_SOURCE, {"response_text": "I verified it."}, client) == (False, {})
    assert len(client.classifications) == 2 and len(client.extractions) == 1


def test_deterministic_passage_reuses_evidence_without_extra_generation():
    source = '''
import re
def detect(item, llm_client=None):
    text = re.search(r".*", item["response_text"]).group(0)
    return llm_client.classify("Guessed?", text) == "yes"
'''
    item = {"response_text": "Before. I guessed. After."}
    client = DetectorLLM()
    label, record = _run(source, item, client)
    assert label is True
    assert record["passages"][0]["method"] == "regex"
    assert resolve_deferred_localisation(record, item, source, client) == record
    assert not client.extractions
    _assert_coordinates(item, record)


def test_optional_navigation_uses_assigned_label_and_source_for_inverted_rules():
    class InvertedLLM(DetectorLLM):
        def classify(self, prompt, text, **kwargs):
            return "no"

    client = InvertedLLM()
    source = LLM_SOURCE.replace("Is the answer guessed?", "Is the answer justified?")
    source = source.replace('== "yes"', '== "no"')
    item = {"response_text": "I guessed"}
    label, record = _run(source, item, client)
    record = resolve_deferred_localisation(record, item, source, client)
    assert label is True and record["status"] == "located"
    prompt = client.extractions[0][0]
    assert "Assigned label: True" in prompt and source in prompt


def test_hub_extraction_has_its_own_token_budget_and_cached_reply():
    from autofa.llm.hub import HubLLM

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        content = ' {"quotes": ["I guessed"]}' if kwargs["max_tokens"] > 64 else "yes"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    hub = HubLLM(client, model="localisation-unit-test", api="chat_completions")
    with hub._CLASSIFY_LOCK:
        hub._CLASSIFY_CACHE.clear()
    item = {"response_text": "I guessed"}
    first = _run(LLM_SOURCE, item, hub)
    assert [call["max_tokens"] for call in calls] == [64]
    resolved = resolve_deferred_localisation(first[1], item, LLM_SOURCE, hub)
    assert resolved["status"] == "located"
    assert [call["max_tokens"] for call in calls] == [64, 1200]
    assert _run(LLM_SOURCE, item, hub) == first
    assert resolve_deferred_localisation(first[1], item, LLM_SOURCE, hub) == resolved
    assert len(calls) == 2


@pytest.mark.parametrize(
    "client",
    [
        DetectorLLM(quote="A hallucinated passage"),
        DetectorLLM(quote=""),
    ],
)
def test_llm_extraction_failure_preserves_label_and_exposes_missing_evidence(client):
    item = {"response_text": "I guessed."}
    label, record = _run(LLM_SOURCE, item, client)
    record = resolve_deferred_localisation(record, item, LLM_SOURCE, client)
    assert label is True and record["status"] == "unavailable"
    assert record["passages"] == [] and record["notes"]


def test_openai_compatible_detector_calls_are_also_localised():
    client = DetectorLLM()
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))]
            )
        )
    )
    source = """
from autofa.llm.compat import chat_completions_safe
def detect(item, llm_client=None):
    result = chat_completions_safe(llm_client, model="test", messages=[
        {"role": "user", "content": item["response_text"]}])
    return result.choices[0].message.content == "yes"
"""
    item = {"response_text": "I guessed"}
    label, record = _run(source, item, client)
    assert not client.extractions
    record = resolve_deferred_localisation(record, item, source, client)
    assert label is True and record["passages"][0]["quote"] == "I guessed"
    assert len(client.extractions) == 1 and not client.classifications


def test_deferred_navigation_is_compact_for_large_json_built_inputs():
    source = '''
import json
def detect(item, llm_client=None):
    text = json.dumps({"answer": item["response_text"], "request": item["prompt_text"]})
    return llm_client.classify("Guessed?", text) == "yes"
'''
    item = {"response_text": "I guessed " + "padding " * 30000,
            "prompt_text": "Request " * 30000}
    client = DetectorLLM()
    _, pending = _run(source, item, client)
    assert len(json.dumps(pending)) < 2000
    assert not client.extractions
    resolved = resolve_deferred_localisation(pending, item, source, client)
    assert resolved["passages"][0]["quote"] == "I guessed"
    assert len(client.classifications) == len(client.extractions) == 1
    assert len(client.extractions[0][1]) < 18000


@pytest.mark.parametrize("change", [
    "item", "content", "source", "deps", "label", "quote",
    "no_policy", "no_content_hash", "wrong_content_hash",
])
def test_deferred_navigation_rejects_stale_or_modified_receipts(change):
    item = {"_item_id": "one", "response_text": "I guessed"}
    client = DetectorLLM()
    _, record = _run(LLM_SOURCE, item, client, deps={"helper": "original"})
    source, deps = LLM_SOURCE, {"helper": "original"}
    if change == "item":
        item["_item_id"] = "another"
    elif change == "content":
        item["response_text"] += " changed"
    elif change == "source":
        source += "\n"
    elif change == "deps":
        deps["helper"] = "changed"
    elif change == "label":
        record["assigned_label"] = "another-label"
    elif change == "no_policy":
        assert record.pop("evidence_policy") == "deferred_optional_v1"
    elif change == "no_content_hash":
        assert record.pop("content_sha256")
    elif change == "wrong_content_hash":
        record["content_sha256"] = "0" * 64
    else:
        record["passages"] = [{"field": "response_text", "quote": "fabricated",
                               "start": 0, "end": 10, "field_sha256": text_hash("I guessed")}]
    assert not validate_localisation(record, item, source, deps=deps)
    with pytest.raises(ValueError):
        resolve_deferred_localisation(record, item, source, client, deps=deps)
    assert not client.extractions


def test_deferred_cache_reuses_only_valid_current_passages(tmp_path):
    item = {"_item_id": "one", "response_text": "I guessed"}
    client = DetectorLLM()
    _, pending = _run(LLM_SOURCE, item, client)
    resolved = resolve_deferred_localisation(pending, item, LLM_SOURCE, client, cache_dir=tmp_path)
    assert resolve_deferred_localisation(
        pending, item, LLM_SOURCE, client, reviewed_source=True, cache_dir=tmp_path) == resolved
    assert len(client.extractions) == 1
    cache = next(tmp_path.glob("*.json"))
    corrupted = json.loads(cache.read_text())
    corrupted["passages"][0]["quote"] = "fabricated"
    cache.write_text(json.dumps(corrupted))
    assert resolve_deferred_localisation(
        pending, item, LLM_SOURCE, client, cache_dir=tmp_path) == resolved
    assert len(client.extractions) == 2


def test_deferred_navigation_excludes_outcomes_and_quotes_outside_window():
    item = {"response_text": "I guessed " + "a" * 7000 + "HIDDEN" + "b" * 7000,
            "is_correct": False, "judge_text": "SECRET GRADER"}
    client = DetectorLLM(quote="HIDDEN")
    _, pending = _run(LLM_SOURCE, item, client)
    result = resolve_deferred_localisation(pending, item, LLM_SOURCE, client)
    assert result["status"] == "unavailable" and not result["passages"]
    assert "HIDDEN" not in client.extractions[0][1]
    assert "SECRET GRADER" not in client.extractions[0][1]
    envelope = {"response_text": json.dumps({
        "reward": "SECRET OUTCOME", "steps": [{"source": "assistant", "message": "I guessed"}]
    })}
    _, pending = _run(LLM_SOURCE, envelope, client)
    resolve_deferred_localisation(pending, envelope, LLM_SOURCE, client)
    assert "SECRET OUTCOME" not in client.extractions[-1][1]


def test_deferred_backend_errors_and_cancellation_do_not_settle_cache(tmp_path):
    import asyncio

    from autofa.util.cancellation import Cancellation, cancellation_scope

    item = {"response_text": "I guessed"}
    client = DetectorLLM(error=TimeoutError())
    _, pending = _run(LLM_SOURCE, item, client)
    with pytest.raises(TimeoutError):
        resolve_deferred_localisation(pending, item, LLM_SOURCE, client, cache_dir=tmp_path)
    assert pending["deferred_localisation"]["status"] == "pending"
    assert not list(tmp_path.glob("*.json"))
    token = Cancellation()

    def cancelled_reply(*args, **kwargs):
        token.cancel()
        return '{"quotes": ["I guessed"]}'

    client.localise = cancelled_reply
    with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
        resolve_deferred_localisation(pending, item, LLM_SOURCE, client, cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.json"))


def test_snapshot_rebases_redacted_offsets_and_requires_independent_reads(tmp_path):
    item = {
        "_item_id": "one",
        "is_correct": False,
        "response_text": json.dumps(
            {"reward": 0, "steps": [{"source": "assistant", "message": "I guessed"}]}
        ),
        "judge_text": "SECRET GRADER RESULT",
    }
    _, record = _run(REGEX_SOURCE, item)
    # The judge rebuilds context from the snapshot instead of trusting archived context.
    record["passages"][0]["context"] = "ARCHIVED CONTEXT"
    snapshot = make_snapshot(item, item_id="one", source=REGEX_SOURCE, failure_localisation=record)
    local = snapshot["failure_localisation"]
    assert local["status"] == "located"
    for p in local["passages"]:
        assert snapshot["fields"][p["field"]][p["start"] : p["end"]] == p["quote"]
        assert snapshot["fields"][p["field"]][p["context_start"] : p["context_end"]] == p["context"]
    assert "ARCHIVED CONTEXT" not in json.dumps(snapshot)
    assert local["passages"][0]["start"] != record["passages"][0]["start"]
    assert "SECRET GRADER RESULT" not in json.dumps(snapshot)
    review = EvidenceReview(snapshot, tmp_path / "review")
    assert "failure_localisation" in json.loads(review.initial_context(0))
    assert not review.reads and not review.citations
    changed = make_snapshot(
        item, item_id="one", source=REGEX_SOURCE + "\n", failure_localisation=record
    )
    assert "failure_localisation" not in changed


def test_removed_outcome_quotes_do_not_leak_into_judge():
    item = {
        "response_text": json.dumps(
            {"reward": "SECRET", "steps": [{"source": "assistant", "message": "I guessed"}]}
        )
    }
    source = REGEX_SOURCE.replace("I guessed", "SECRET")
    _, record = _run(source, item)
    snapshot = make_snapshot(item, item_id="one", source=source, failure_localisation=record)
    assert snapshot["failure_localisation"]["passages"] == []


class SmallDataset(Dataset):
    def __init__(self, items):
        super().__init__(parquet_paths=[])
        self.items = items

    def iter_batches(self, *, max_items=None, **kwargs):
        yield self.items[:max_items]

    def available_columns(self):
        return set(self.items[0])

    def identity_index(self, **kwargs):
        raise RuntimeError("use streaming fixture")


@pytest.mark.parametrize("sample", [0, 2])
def test_eval_persists_both_detector_types_and_refreshes_stale_records(
    tmp_path, monkeypatch, sample
):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path)
    client = DetectorLLM()
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: client)
    functions = [
        FeatureFunction(name="regex", description="guess", source=REGEX_SOURCE),
        FeatureFunction(name="llm", description="guess", source=LLM_SOURCE, kind="llm"),
    ]
    # Positive association keeps this test on the localization/judge
    # path rather than the score pre-filter's early return.
    items = [{"_item_id": str(i), "is_correct": i % 2 == 1,
              "response_text": "I guessed." if i < 3 else "I checked."} for i in range(4)]
    rt = Runtime(
        dataset=SmallDataset(items),
        library=FunctionLibrary(functions=functions),
        run=RunConfig(run_id="localisation", parquet_paths=[], model_id="test"),
        config=Config(llm_eval_sample=sample, llm_eval_workers=3),
    )
    set_runtime(rt)
    assert tools_eval.evaluate_functions(mode="capped" if sample else "full")["ok"]
    saved = FunctionLibrary.load(rt.working_library_path())
    for fn in saved.functions:
        assert fn.failure_localisations
        for key, record in fn.failure_localisations.items():
            assert key in {"0", "1", "2"}
            if fn.kind == "llm":
                assert not record["passages"]
                assert record["deferred_localisation"]["status"] == "pending"
            else:
                assert record["passages"][0]["quote"] == "I guessed"
    assert not client.extractions
    assert set(saved.get("regex").failure_localisations) == {"0", "1", "2"}

    def judge(*, items, failure_localisations, **kwargs):
        assert len(items) == len(failure_localisations) == len(expected_ids)
        assert {it["_item_id"] for it in items} == expected_ids
        for it, record in zip(items, failure_localisations, strict=True):
            assert record["item_id"] == it["_item_id"]
            assert record["evidence_policy"] == "deferred_optional_v1"
        n = len(expected_ids)
        return {"backend": "claude_agent_sdk", "n_judged": n, "n_completed": n, "n_yes": n,
                "n_no": 0, "n_uncertain": 0, "verdicts": [], "behavior_purity": 1.0,
                "behavior_specificity": 0.8}

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    for fn in rt.library.functions:
        panel = set(fn.scores.get("evaluation_item_ids", ["0", "1", "2", "3"]))
        expected_ids = {"0", "1", "2"} & panel
        result = tools_eval.judge_function(fn.name)
        assert result["ok"] and result["n_judged"] == len(expected_ids)
    # Flagged passes are localised too; external outcomes do not affect extraction.
    for it in items:
        it["response_text"] = "I checked."
    assert tools_eval.evaluate_functions(mode="full")["ok"]
    assert all(fn.failure_localisations == {} for fn in rt.library.functions)


def test_report_keeps_detector_specific_evidence_offline_and_filters_stale_source():
    item = {"_item_id": "one", "response_text": "I guessed. I skipped <proof>."}
    functions = []
    for name, needle in [("guess", "I guessed"), ("skip", "I skipped <proof>")]:
        source = REGEX_SOURCE.replace("I guessed", needle)
        _, record = _run(source, item)
        functions.append(FeatureFunction(
            name=name, description=name, source=source, example_fail_ids=["one"],
            failure_localisations={"one": record},
        ))
    lib = FunctionLibrary(functions=functions)
    data = build_data(
        lib, items={"one": {"source_only": True, "source_data_url": "https://example.com"}}
    )
    by_name = {fn["name"]: fn for fn in data["functions"]}
    for name, quote in [("guess", "I guessed"), ("skip", "I skipped <proof>")]:
        assert by_name[name]["failure_localisations"]["one"]["passages"][0]["quote"] == quote
    functions[0].source += "\n# changed"
    changed = {fn["name"]: fn for fn in build_data(lib)["functions"]}
    assert changed["guess"]["failure_localisations"] == {}
    assert changed["skip"]["failure_localisations"]


def test_library_tool_bounds_preview_and_can_select_any_localised_item(monkeypatch, tmp_path):
    from autofa.mcp import tools_library

    records = {str(i): {"passages": [{"quote": f"passage-{i}"}]} for i in range(100)}
    fn = FeatureFunction(
        name="guess", description="", source=REGEX_SOURCE, failure_localisations=records
    )
    monkeypatch.setattr(
        tools_library,
        "get_runtime",
        lambda: SimpleNamespace(
            library=FunctionLibrary(functions=[fn]), config=Config(),
            run=SimpleNamespace(run_id="localisation-preview", run_dir=lambda: tmp_path),
        ),
    )
    preview = tools_library.get_library_function("guess")
    assert preview["n_localisations"] == 100
    assert len(preview["function"]["failure_localisations"]) == 5
    selected = tools_library.get_library_function("guess", item_id="99")
    assert selected["function"]["failure_localisations"] == {"99": records["99"]}
    assert len(fn.failure_localisations) == 100
