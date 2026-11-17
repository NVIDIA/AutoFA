# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered exploration, occurrence identity and source event/part provenance."""

import json
from urllib.parse import parse_qs, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import trace
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import detector_hash, passage
from autofa.core.sandbox import run_detector
from autofa.data.parquet_store import Dataset
from autofa.data.reference_data import event_url
from autofa.data.source_export import normalize_events
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trace_navigation import NAVIGATION_VERSION, TraceNavigation, trace_events
from autofa.mcp import tools_data
from autofa.mcp.server import TOOL_CALLABLES
from autofa.reports.failure_cards import _compact_item, build_data


def navigation_item():
    def event(index, role, kind, texts):
        return {"event_idx": index, "role": role, "kind": kind, "lane": "worker/a",
                "n_parts": len(texts), "parts": [
                    {"part_idx": part, "part_type": "text", "text": text,
                     "chars": len(text), "lane": "worker/a"} for part, text in texts
                ]}

    events = [
        event(0, "user", "message", [(0, "Fix the failing test.")]),
        event(25, "assistant", "reasoning", [(0, "🙂 I guessed <answer>. Checking now.")]),
        event(26, "assistant", "tool_call", []),
        event(27, "tool", "tool_result", [(0, "Tests failed.")]),
        event(40, "assistant", "reasoning", [(0, "Earlier work. " * 17000),
                                              (7, "🙂 I guessed <answer>. I will fix it.")]),
        event(41, "assistant", "tool_call", []),
        event(42, "tool", "tool_result", [(0, "Tests passed after the correction.")]),
        event(50, "assistant", "message", [(0, "The fix is ready.")]),
    ]
    calls = [{"event_idx": index, "call_idx": 0, "tool_name": "test",
              "args_full": '{"command": "pytest"}', "args_chars": 21,
              "result_event_idx": result, "lane": "worker/a"}
             for index, result in [(26, 27), (41, 42)]]
    for i, parent in [(3, 26), (6, 41)]:
        events[i].update(paired_event_idx=parent, paired_call_idx=0)
    trajectory = normalize_events({"n_events": len(events), "n_tool_calls": 2},
                                  events, calls, conversation_lane="worker/a")
    return {"_item_id": "example", "problem_id": "suite/task", "run_idx": 0,
            "is_correct": False, "response_text": json.dumps(trajectory), "reasoning_text": ""}


def later_passage(item):
    fields = derive_trace_fields(item)
    text = fields["reasoning_text"]
    start = text.rindex("I guessed <answer>")
    return passage("reasoning_text", text, start, start + len("I guessed <answer>"),
                   method="regex")


CHECK = "def check(item): return True"


def check_library(localisations, **fields):
    return FunctionLibrary(functions=[FeatureFunction(
        name="check", description="Example", source=CHECK, failure_localisations=localisations,
        **fields)])


def locate_span(item, start, end, method="literal"):
    raw = item["response_text"]
    return TraceNavigation(item).locate(passage("response_text", raw, start, end, method=method))


def source_literal(item, literal):
    start = item["response_text"].index(literal)
    return locate_span(item, start, start + len(literal))


def with_part(text, *, event_idx=60, lane="worker/α", part_idx=9):
    """The navigation item plus one auxiliary execution event carrying a single part."""
    item = navigation_item()
    value = json.loads(item["response_text"])
    value["execution"]["events"].append(
        {"event_idx": event_idx, "lane": lane, "parts": [{"part_idx": part_idx, "text": text}]})
    item["response_text"] = json.dumps(value)
    return item


def test_repeated_quote_maps_later_occurrence_to_source_part_with_unicode():
    item = navigation_item()
    nav = TraceNavigation(item)
    location, = nav.locate(later_passage(item))
    assert location["step_index"] == 5
    assert location["source_event"] == {"event_idx": 40, "lane": "worker/a", "part_idx": 7}
    part = nav.events[4]["sections"][0]
    assert part["text"][location["start"]:location["end"]] == "I guessed <answer>"
    assert nav.events[3]["paired_step_index"] == 3
    assert nav.events[2]["result_step_indices"] == [4]
    assert trace.steps(item)[4].source_event == {"event_idx": 40, "lane": "worker/a"}
    assert trace.steps(item)[5].result_step_indices == (7,)


@pytest.mark.parametrize("mutation", ["quote", "hash"])
def test_stale_passages_never_navigate_to_an_unverified_event(mutation):
    item = navigation_item()
    p = later_passage(item)
    if mutation == "quote":
        p["quote"] = "invented"
    else:
        p["field_sha256"] = "stale"
    assert TraceNavigation(item).locate(p) == []


@pytest.mark.parametrize("raw_fallback", [False, True])
def test_local_or_raw_recovery_steps_do_not_invent_api_event_ids(raw_fallback):
    item = navigation_item()
    value = json.loads(item["response_text"])
    if raw_fallback:
        value["evidence_scope"]["trajectory_source"] = "raw ng_trajectory.model_calls"
    else:  # A local import declares no source identity at all.
        value.pop("evidence_scope")
    item["response_text"] = json.dumps(value)
    assert all("source_event" not in event for event in trace_events(item))
    location, = TraceNavigation(item).locate(later_passage(item))
    assert location["step_index"] == 5 and "source_event" not in location


def test_v8_cache_still_links_events_without_guessing_part_ids():
    item = navigation_item()
    value = json.loads(item["response_text"])
    for step in value["steps"]:
        step.pop("source_parts", None)
    item["response_text"] = json.dumps(value)
    location, = TraceNavigation(item).locate(later_passage(item))
    assert location["source_event"] == {"event_idx": 40, "lane": "worker/a"}


def test_raw_json_offsets_map_only_execution_content_and_decode_unicode():
    item = navigation_item()
    value = json.loads(item["response_text"])
    value["model_call_evidence"] = {"copy": "🙂 I guessed <answer>. I will fix it."}
    item["response_text"] = json.dumps(value, ensure_ascii=True, indent=2)
    raw = item["response_text"]
    quote = json.dumps("🙂 I guessed <answer>", ensure_ascii=True)[1:-1]
    # Select the later step's rendered reasoning explicitly: preserved source
    # parts legitimately add earlier duplicates of the same text.
    later_step = raw.index('"step_id": 40')
    reasoning = raw.index('"reasoning_content":', later_step)
    later = raw.index(quote, reasoning)
    loc, = locate_span(item, later, later + len(quote), method="regex")
    assert loc["step_index"] == 5
    assert loc["quote"] == "🙂 I guessed <answer>"
    assert loc["source_event"] == {"event_idx": 40, "lane": "worker/a", "part_idx": 7}
    part = TraceNavigation(item).events[4]["sections"][0]
    assert part["text"][loc["start"]:loc["end"]] == "🙂 I guessed <answer>"
    # The step's and execution's preserved parts map to the same step, source identity
    # and exact occurrence, not to the earlier duplicate in event 25.
    step_part = raw.index(quote, later_step)
    assert step_part < reasoning
    executed = raw.index(quote, raw.index('"event_idx": 40', raw.index('"execution":')))
    for preserved in (step_part, executed):
        preserved_loc, = locate_span(item, preserved, preserved + len(quote))
        assert preserved_loc["step_index"] == 5
        assert preserved_loc["source_event"] == loc["source_event"]
        assert part["text"][preserved_loc["start"]:preserved_loc["end"]] == loc["quote"]
    duplicate = raw.rindex(quote)
    # A request pool is not an execution event.
    assert locate_span(item, duplicate, duplicate + len(quote), method="regex") == []


def test_raw_tool_arguments_map_to_call_event_without_guessing_a_part():
    item = navigation_item()
    raw = item["response_text"]
    start = raw.rindex("pytest")
    loc, = locate_span(item, start, start + 6)
    assert loc["step_index"] == 6 and loc["section"] == "tool:0"
    assert loc["source_event"] == {"event_idx": 41, "lane": "worker/a"}


def test_source_urls_keep_event_zero_part_zero_and_lane_identity():
    rollout = "https://runs.example.test/prefix/#/rollout/run/suite%252Ftask/0"
    assert event_url(rollout, {"event_idx": 0, "part_idx": 0, "lane": "worker/a"}) == (
        rollout + "/e0-p0?lane=worker%2Fa")
    assert event_url(rollout, {"event_idx": 25, "lane": ""}) == rollout + "/e25?lane="
    for bad in [{"event_idx": -1, "lane": ""}, {"event_idx": 0}]:
        assert not event_url(rollout, bad)
    assert not event_url("javascript:alert(1)", {"event_idx": 0, "lane": ""})


def test_agent_search_pages_full_trace_and_context_preserves_recovery(monkeypatch):
    item = navigation_item()
    monkeypatch.setattr(tools_data, "_load_item_raw", lambda _: item)
    first = tools_data.search_trajectory("example", "I guessed", limit=1)
    assert first["total_matches"] == 2 and first["next_offset"] == 1
    later = tools_data.search_trajectory("example", "I guessed", offset=1, limit=1)
    assert later["next_offset"] is None
    hit, = later["matches"]
    assert hit["step_index"] == 5 and hit["start"] > 200000
    assert hit["source_event"]["event_idx"] == 40
    assert hit["context"][hit["start"] - hit["context_start"]:
                           hit["end"] - hit["context_start"]] == hit["quote"]
    context = tools_data.get_trajectory_context("example", 5, before=2, after=2)
    assert [e["index"] for e in context["events"]] == [3, 4, 5, 6, 7]
    assert "passed after the correction" in context["events"][-1]["sections"][0]["text"]
    assert context["events"][2]["sections"][0]["truncated"]
    step = tools_data.get_trajectory_step("example", 5, field="reasoning", offset=hit["start"])
    assert step["text"].startswith("I guessed") and step["source_event"]["event_idx"] == 40
    assert tools_data.get_trajectory("example", limit=3)["next_offset"] == 3
    assert {"search_trajectory", "get_trajectory_context"} <= TOOL_CALLABLES.keys()


def test_structured_detector_retains_selected_repeated_step_and_labels():
    text = "Identical long reasoning: I guessed <answer>."
    item = {"response_text": json.dumps({"steps": [
        {"source": "assistant", "reasoning_content": text},
        {"source": "assistant", "reasoning_content": text},
    ]})}
    source = '''
from autofa.trace import steps
import re
def detect(item):
    return bool(re.search("I guessed", steps(item)[1].reasoning))
'''
    record = {}
    assert run_detector(source, item, localisation=record) is True
    p, = record["passages"]
    loc, = p["trace_locations"]
    assert loc["step_index"] == 2


def test_report_upgrades_old_evidence_without_mutating_library():
    item = navigation_item()
    old = {"passages": [later_passage(item)], "detector_hash": detector_hash(CHECK)}
    reference = [{"run_id": "run", "export_url": "https://runs.example.test/v1/runs/run/export/autofa"}]
    compact = _compact_item(item, reference_data=reference, localisations={"check": old})
    data = build_data(check_library({"example": old}), items={"example": compact})
    saved = data["functions"][0]["failure_localisations"]["example"]
    loc, = saved["passages"][0]["trace_locations"]
    assert loc["source_url"].endswith(
        "/e40-p7?lane=worker%2Fa&text=I%20guessed%20%3Canswer%3E")
    assert "trace_locations" not in old["passages"][0]
    assert len(data["items"]["example"]["trace_events"]) == 8


def test_link_report_uses_saved_event_links_without_reading_trace(tmp_path, monkeypatch):
    item = navigation_item()
    item.update(model_id="model", benchmark="benchmark")
    item.pop("_item_id")
    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([item]), path)
    dataset = Dataset([path])
    row = next(dataset.iter_batches())[0]
    iid = row["_item_id"]
    record = TraceNavigation(row).attach(
        {"passages": [later_passage(row)], "detector_hash": detector_hash(CHECK)})
    library = check_library({iid: record}, example_fail_ids=[iid])
    reference = [{"run_id": "run", "parquet_path": str(path),
                  "export_url": "https://runs.example.test/v1/runs/run/export/autofa"}]
    # Saved records already have verified identities: no full-text re-read.
    monkeypatch.setattr(dataset, "read_locations", lambda *a, **k: pytest.fail("unexpected read"))
    data = build_data(library, dataset=dataset, reference_data=reference, prefer_source_links=True)
    saved = data["functions"][0]["failure_localisations"][iid]
    loc, = saved["passages"][0]["trace_locations"]
    assert loc["source_url"].endswith(
        "/e40-p7?lane=worker%2Fa&text=I%20guessed%20%3Canswer%3E")
    assert data["items"][iid]["source_only"]
    assert "Earlier work." * 10 not in json.dumps(data)
    assert "response_raw" not in data["items"][iid]
    assert saved["navigation_version"] == NAVIGATION_VERSION
    # Unverified coordinates keep only the saved quote and whole-rollout link.
    library.functions[0].failure_localisations[iid] = {**record, "navigation_version": 2}
    stale = build_data(library, dataset=dataset, reference_data=reference,
                       prefer_source_links=True)
    unverified, = stale["functions"][0]["failure_localisations"][iid]["passages"]
    assert unverified["quote"] == record["passages"][0]["quote"]
    assert unverified["trace_locations"] == []


def test_auxiliary_execution_lane_has_verified_source_link_without_invented_step():
    item = with_part("AUXILIARY", event_idx=40, lane="worker/b", part_idx=3)
    location, = source_literal(item, "AUXILIARY")
    assert location == {"section": "part", "source_event": {
        "event_idx": 40, "lane": "worker/b", "part_idx": 3,
    }}


@pytest.mark.parametrize("auxiliary,cloud", [(False, True), (True, False)])
def test_report_source_link_decodes_json_unicode_before_highlighting(auxiliary, cloud):
    selected = "🙂 quoted \"word\" and\nnext line"
    if auxiliary:
        item = with_part(selected)
    else:
        item = navigation_item()
        value = json.loads(item["response_text"])
        # The normalized step and retained source part both carry the same source text.
        step = value["steps"][1]
        step["reasoning_content"] = selected
        step["parts"][0]["text"] = selected
        step["source_parts"][0].update(start=0, end=len(selected))
        value["execution"]["events"][1]["parts"][0]["text"] = selected
        item["response_text"] = json.dumps(value)
    encoded = json.dumps(selected, ensure_ascii=True)[1:-1]
    start = item["response_text"].index(encoded)
    evidence = passage("response_text", item["response_text"], start, start + len(encoded),
                       method="literal")
    record = TraceNavigation(item).attach({
        "passages": [evidence], "detector_hash": detector_hash(CHECK),
    })
    location, = record["passages"][0]["trace_locations"]
    assert location["quote"] == selected
    rollout = ("https://runs.example.test/" if cloud else "/traces/" + "a" * 64 + "/")
    rollout += "#/rollout/r/t/0"
    report = build_data(check_library({"example": record}),
                        items={"example": {"source_data_url": rollout}})
    saved = report["functions"][0]["failure_localisations"]["example"]["passages"][0]
    url = saved["trace_locations"][0]["source_url"]
    query = parse_qs(urlsplit(url).fragment.split("?", 1)[1])
    assert query["text"] == [" ".join(selected.split())]
    assert query["lane"] == (["worker/α"] if auxiliary else ["worker/a"])


def test_identical_long_context_keeps_event_link_without_wrong_occurrence_highlight():
    item = with_part("/" * 2000)
    start = item["response_text"].index("/" * 2000) + 1000
    location, = locate_span(item, start, start + 1)
    assert location["source_context"] == {"unavailable": True}
    assert event_url("https://runs.example.test/#/rollout/r/t/0", location["source_event"], "/",
                     context=location["source_context"]).endswith("/e60-p9?lane=worker%2F%CE%B1")


def test_highlight_context_counts_overlapping_prefix_matches():
    item = with_part("a" * 40)
    start = item["response_text"].index("a" * 40) + 16
    location, = locate_span(item, start, start + 1)
    # Sixteen characters on either side would also identify offsets17..23.
    # The chosen suffix must reach the end to identify only offset16.
    assert location["source_context"] == {"prefix": "a" * 16, "suffix": "a" * 23}


def test_partial_json_escape_links_event_without_highlighting_escape_text():
    # A selection of half a surrogate pair has no source text to highlight.
    item = with_part("🙂")
    start = item["response_text"].rindex("\\ud83d")
    location, = locate_span(item, start, start + 6)
    assert location["quote"] == "" and location["source_event"]["part_idx"] == 9


@pytest.mark.parametrize("ambiguous", [False, True])
def test_artifact_text_links_only_when_its_recorded_event_identity_is_unambiguous(ambiguous):
    item = navigation_item()
    value = json.loads(item["response_text"])
    value["execution"]["artifacts"] = [{
        "artifact_idx": 5, "event_idx": 40, "part_idx": 7, "text": "ARTIFACT_CONTENT",
    }]
    if ambiguous:
        value["execution"]["events"].append({"event_idx": 40, "lane": "worker/b", "parts": []})
    item["response_text"] = json.dumps(value)
    locations = source_literal(item, "ARTIFACT_CONTENT")
    if ambiguous:
        assert locations == []
    else:
        assert locations == [{"section": "artifact", "source_event": {
            "event_idx": 40, "lane": "worker/a", "part_idx": 7,
        }}]


@pytest.mark.parametrize("capture", ["event_text", "snapshot", "copied"])
def test_request_capture_text_never_masquerades_as_executed_step_evidence(capture):
    item = navigation_item()
    value = json.loads(item["response_text"])
    event = {"event_idx": 70, "lane": "worker/a", "parts": []}
    if capture == "event_text":
        event["event_text"] = {"request_messages": [{"role": "user", "content": "CAPTURE_ONLY"}]}
    elif capture == "snapshot":
        event.update(kind="other", role="harness", parts=[{"part_idx": 0, "text": json.dumps({
            "message_ids": ["one"],
            "messages_first_seen_here": {"one": {"role": "user", "content": "CAPTURE_ONLY"}},
        })}])
    else:
        event.update(is_copied_context=True, parts=[{"part_idx": 0, "text": "CAPTURE_ONLY"}])
    value["execution"]["events"].append(event)
    item["response_text"] = json.dumps(value)
    assert source_literal(item, "CAPTURE_ONLY") == []


def test_rewritten_source_part_has_event_link_without_source_character_coordinates():
    item = {"response_text": json.dumps({"steps": [],
        "evidence_scope": {"identity": "evalrdl_tables_v1"}, "execution": {"events": [{
        "event_idx": 0, "lane": "", "parts": [{"part_idx": 0, "text": "FILTERED_CONTENT",
                                                  "metadata_filtered": True, "source_chars": 99}],
    }]}})}
    location, = source_literal(item, "FILTERED_CONTENT")
    assert location == {"section": "part", "source_event": {
        "event_idx": 0, "lane": "", "part_idx": 0,
    }}


def test_artifact_link_needs_explicit_part_artifact_reference_not_a_bare_part_number():
    item = navigation_item()
    value = json.loads(item["response_text"])
    event = next(row for row in value["execution"]["events"] if row["event_idx"] == 40)
    event["parts"][1]["artifact_idx"] = 12
    value["execution"]["artifacts"] = [{"artifact_idx": 12, "text": "LINKED_ARTIFACT"},
                                       {"part_idx": 7, "text": "UNLINKED_ARTIFACT"}]
    item["response_text"] = json.dumps(value)
    assert source_literal(item, "LINKED_ARTIFACT") == [{"section": "artifact", "source_event": {
        "event_idx": 40, "lane": "worker/a", "part_idx": 7,
    }}]
    assert source_literal(item, "UNLINKED_ARTIFACT") == []


def test_event_url_names_the_passage_for_the_viewer_to_highlight():
    rollout = "/traces/" + "a" * 64 + "/#/rollout/r/t/0"
    short = event_url(rollout, {"event_idx": 2, "lane": ""}, "5-7 business, days & more")
    assert short == rollout + "/e2?lane=&text=5%2D7%20business%2C%20days%20%26%20more"
    words = [f"word{i:03d}" for i in range(60)]
    long = event_url(rollout, {"event_idx": 2, "lane": ""}, "  ".join(words))
    start, end = long.split("&text=")[1].split(",")
    assert start == "%20".join(words[:8]) and end == "%20".join(words[-8:])
    assert event_url(rollout, {"event_idx": 2, "lane": ""}, "  ") == rollout + "/e2?lane="


INPUT_INSTRUCTION = '🙂 Follow the instructions.\nWhen finished, generate "###STOP###".'


def input_navigation_item(*, content=INPUT_INSTRUCTION, request=None, extra_events=()):
    texts = ([content] if isinstance(content, str) else [block['text'] for block in content])
    def event(index, role, texts, lane=''):
        return {'event_idx': index, 'role': role, 'kind': 'message', 'lane': lane,
                'n_parts': len(texts), 'parts': [
                    {'part_idx': i, 'part_type': 'text', 'text': text,
                     'chars': len(text), 'lane': lane} for i, text in enumerate(texts)
                ]}
    events = [event(0, 'system', texts), event(1, 'user', ['Do the task.']),
              event(13, 'assistant', ['Please confirm.']), event(14, 'user', ['###STOP###']),
              *extra_events]
    value = normalize_events({'n_events': len(events), 'n_tool_calls': 0}, events, [])
    request = request if request is not None else {'messages': [
        {'role': 'system', 'content': content}, {'role': 'user', 'content': 'Do the task.'},
    ]}
    value['initial_request'] = request
    return {'_item_id': 'input-example', 'problem_id': 'task', 'run_idx': 0,
            'response_text': json.dumps(value), 'input_json': json.dumps(request)}


def input_passage(item, quote='###STOP###', *, field='input_json', last=False):
    text = derive_trace_fields(item)[field]
    start = text.rindex(quote) if last else text.index(quote)
    return passage(field, text, start, start + len(quote), method='literal')


@pytest.mark.parametrize('container', [
    None,
    [{'role': 'system', 'content': INPUT_INSTRUCTION}],
    {'system': INPUT_INSTRUCTION, 'input': 'Do the task.'},
    {'instructions': INPUT_INSTRUCTION, 'input': [{'role': 'user', 'content': 'Do the task.'}]},
])
def test_input_passage_maps_complete_message_not_later_stop_quote(container):
    item = input_navigation_item(request=container)
    location, = TraceNavigation(item).locate(input_passage(item))
    assert location['source_event'] == {'event_idx': 0, 'lane': '', 'part_idx': 0}
    assert location['step_index'] == 1
    assert (location['start'], location['end']) == (
        INPUT_INSTRUCTION.index('###STOP###'), INPUT_INSTRUCTION.index('###STOP###') + 10,
    )


@pytest.mark.parametrize('field', ['input_json', 'input_text'])
def test_input_text_blocks_use_offsets_for_repeated_quotes(field):
    content = [{'type': 'text', 'text': '🙂 First ###STOP###.'},
               {'type': 'text', 'text': 'Second ###STOP###.'}]
    item = input_navigation_item(content=content)
    location, = TraceNavigation(item).locate(input_passage(item, field=field, last=True))
    assert location['source_event']['part_idx'] == 1
    assert location['start'] == len(content[0]['text']) + 2 + len('Second ')


def test_input_json_selection_decodes_escapes_for_native_highlight():
    item = input_navigation_item()
    decoded = '"###STOP###"'
    location, = TraceNavigation(item).locate(input_passage(item, '\\"###STOP###\\"'))
    assert location['quote'] == decoded
    assert INPUT_INSTRUCTION[location['start']:location['end']] == decoded
    assert location['source_event']['part_idx'] == 0


@pytest.mark.parametrize('mutation', [
    'duplicate', 'wrong_role', 'tool_input', 'duplicate_key', 'multimodal',
])
def test_input_mapping_refuses_ambiguous_or_non_message_evidence(mutation):
    item = input_navigation_item()
    value = json.loads(item['response_text'])
    if mutation == 'duplicate':
        duplicate = {**value['steps'][0], 'step_id': 99}
        value['steps'].insert(1, duplicate)
    elif mutation == 'wrong_role':
        item['input_json'] = json.dumps({'messages': [
            {'role': 'user', 'content': INPUT_INSTRUCTION},
        ]})
    elif mutation == 'tool_input':  # a tool call's input is not a prompt message
        item['input_json'] = json.dumps({'messages': [
            {'type': 'tool_use', 'input': 'Do the task.'},
        ]})
    elif mutation == 'duplicate_key':
        item['input_json'] = item['input_json'].replace('"role": "system"',
                                                       '"role": "user", "role": "system"')
    else:
        request = {'messages': [{'role': 'system', 'content': [
            {'type': 'text', 'text': INPUT_INSTRUCTION},
            {'type': 'image_url', 'image_url': 'image'},
        ]}]}
        item['input_json'] = json.dumps(request)
        value['initial_request'] = request
    item['response_text'] = json.dumps(value)
    quote = 'Do the task.' if mutation == 'tool_input' else '###STOP###'
    assert TraceNavigation(item).locate(input_passage(item, quote)) == []
    if mutation == 'multimodal':
        assert TraceNavigation(item).locate(input_passage(item, field='input_text')) == []


def test_input_literal_prompt_still_maps_without_a_rendered_header():
    item = navigation_item()
    item['prompt_text'] = 'Fix the failing test.'
    location, = TraceNavigation(item).locate(input_passage(item, 'failing', field='input_text'))
    assert location['source_event'] == {'event_idx': 0, 'lane': 'worker/a', 'part_idx': 0}


def test_input_raw_fallback_never_invents_native_event_identity():
    item = input_navigation_item()
    value = json.loads(item['response_text'])
    value['evidence_scope'] = {'trajectory_source': 'raw', 'identity': 'evalrdl_tables_v1'}
    item['response_text'] = json.dumps(value)
    locations = TraceNavigation(item).locate(input_passage(item))
    assert all('source_event' not in loc for loc in locations)


def test_input_localisation_gets_native_highlight_in_cloud_report():
    item = input_navigation_item()
    record = {'navigation_version': NAVIGATION_VERSION, 'detector_hash': detector_hash(CHECK),
              'passages': [{**input_passage(item), 'trace_locations': []}]}
    compact = _compact_item(item, localisations={'check': record})
    compact['source_data_url'] = 'https://runs.example.test/#/rollout/r/task/0'
    report = build_data(check_library({'input-example': record}),
                        items={'input-example': compact}, prefer_source_links=True)
    saved = report['functions'][0]['failure_localisations']['input-example']
    assert saved['navigation_version'] == NAVIGATION_VERSION
    location, = saved['passages'][0]['trace_locations']
    assert '/e0-p0?lane=&text=%23%23%23STOP%23%23%23' in location['source_url']
    assert record['passages'][0]['trace_locations'] == []
