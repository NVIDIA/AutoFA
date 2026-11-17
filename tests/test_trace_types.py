# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public autofa.trace contracts, execution records and command prefixes keep real evidence."""

import copy
import json
import types
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    Required,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import pytest

from autofa import trace
from autofa.core.sandbox import DetectorAbstained, run_detector
from autofa.trace_reference import render_trace_api_reference

from support.traces import execution_item


def assert_contract(value, annotation):
    """Check normalized outputs against their public structural typing contract.

    Open dictionary extensions and explicit raw alternatives are intentional;
    this checks only guarantees made by the declarations, not provider schemas.
    """
    if annotation is Any:
        return
    if isinstance(annotation, TypeAliasType):
        return assert_contract(value, annotation.__value__)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Annotated, Required, NotRequired):
        return assert_contract(value, args[0])
    if origin in (Union, types.UnionType):
        for choice in args:
            try:
                assert_contract(value, choice)
                return
            except AssertionError:
                pass
        raise AssertionError(f"{value!r} does not match {annotation!r}")
    if origin is Literal:
        assert any(type(value) is type(choice) and value == choice for choice in args)
        return
    if is_typeddict(annotation):
        assert isinstance(value, dict)
        assert annotation.__required_keys__ <= value.keys()
        for key, hint in get_type_hints(annotation, include_extras=True).items():
            if key in value:
                assert_contract(value[key], hint)
        return
    if origin in (list, tuple):
        assert isinstance(value, origin)
        for element in value:
            assert_contract(element, args[0])
        return
    if origin is dict:
        assert isinstance(value, dict)
        for key, element in value.items():
            assert_contract(key, args[0])
            assert_contract(element, args[1])
        return
    assert isinstance(value, annotation)


def checked(fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    assert_contract(result, get_type_hints(fn, include_extras=True)["return"])
    return result


def item_with_trajectory():
    return {"response_text": json.dumps({
        "steps": [
            {"step_id": 15, "source": "user", "message": "Test the fix."},
            {"step_id": 20, "source": "assistant", "reasoning_content": "Check first.",
             "tool_calls": [{"id": "c1", "function": {"name": "test", "arguments":
                 '{"command": "pytest", "code": "{literal}"}'}, "extension": {"x": 1}}]},
            {"step_id": 21, "source": "tool", "call_id": "c1",
             "paired_event_idx": 20, "observation": '{"passed": true}'},
            {"step_id": 30, "source": "assistant", "message": "Done."},
        ],
        "evidence_scope": {"source": "original API event parts and tool arguments",
                           "event_lane": ""},
    })}


def test_step_and_flat_calls_keep_identity_structure_and_original_arguments():
    item = item_with_trajectory()
    before = copy.deepcopy(item)
    events = checked(trace.steps, item)
    for step in events:
        for field in fields(trace.Step):
            assert_contract(getattr(step, field.name),
                            get_type_hints(trace.Step, include_extras=True)[field.name])
    call = checked(trace.tool_calls, item)[0]
    assert call["step"] == 2 and events[1].source_event == {"event_idx": 20, "lane": ""}
    assert events[1].step_id == 20
    assert "part_idx" not in events[1].source_event
    assert call["call_id"] == "c1" and call["arguments"]["code"] == "{literal}"
    assert call["arguments_raw"].startswith('{"command"')
    assert call["extension"] == {"x": 1}
    assert events[1].result_step_indices == (3,) and events[2].paired_step_index == 2
    assert events[2].observation_data == {"passed": True}
    assert events[1].has_reasoning and events[2].has_observation
    with pytest.raises(FrozenInstanceError):
        events[1].index = 5
    assert checked(trace.parse, item)["n_tool_calls"] == 1
    assert before == item


def test_absent_optional_fields_are_absent_not_fabricated():
    item = {"response_text": {"steps": [
        {"source": "assistant", "tool_calls": [{"name": "opaque", "arguments": "not-json"}]},
    ]}}
    call = checked(trace.tool_calls, item)[0]
    assert call == {"name": "opaque", "arguments": "not-json", "step": 1}
    step, = checked(trace.steps, item)
    assert step.source_event is None and step.step_id is None
    assert step.paired_step_index is None and step.result_step_indices == ()


@pytest.mark.parametrize("arguments", [17, "{broken"])
def test_arbitrary_call_arguments_and_identities_remain_source_evidence(arguments):
    item = {"response_text": {"steps": [{"source": "assistant", "tool_calls": [
        {"name": "custom", "arguments": arguments, "call_id": {"opaque": 1}},
    ]}]}}
    call, = checked(trace.tool_calls, item)
    assert call["arguments"] == arguments and call["call_id"] == {"opaque": 1}
    if arguments == "{broken":
        assert "invalid" in call["arguments_warning"]


def test_messages_and_definitions_expose_shapes_without_discarding_opaque_content():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Read"},
         {"type": "image", "source": {"data": "IMAGE"}, "provider_extension": [1]},
         {"type": "future_block", "payload": {"x": 2}}]},
        {"type": "function_call", "name": "custom", "call_id": "c", "arguments": "[1,2]"},
        {"role": "assistant", "content": 42},
    ]
    definitions = [{"function": {"name": "custom", "parameters": '{"type":"array"}'},
                    "provider_extension": {"x": True}}, {"type": "native_search"}]
    item = {"input_json": {"messages": messages, "tools": definitions}}
    before = copy.deepcopy(item)
    output = checked(trace.input_messages, item)
    tools = checked(trace.input_tools, item)
    assert output[0]["content"] == messages[0]["content"]
    assert "content" not in output[1] and output[1]["arguments"] == [1, 2]
    assert output[2]["content"] == 42
    assert tools[0]["function"]["parameters"] == {"type": "array"}
    assert tools[0]["function"]["parameters_raw"] == '{"type":"array"}'
    assert tools[0]["provider_extension"] == {"x": True}
    assert tools[1] == {"type": "native_search"}
    assert "IMAGE" not in checked(trace.input_text, item)
    assert item == before


def test_model_call_expansion_is_separate_from_timeline_and_preserves_extensions():
    item = {"response_text": {"model_call_evidence": {
        "messages": [{"role": "user", "content": "Task"},
                     {"role": "assistant", "content": "Previous answer"}],
        "tools": [[{"type": "native_search"}]],
        "calls": [{"request": [0, 1], "response": [], "tools": 0,
                   "provider_meta": {"x": [1, 2]}}, {}],
    }}}
    before = copy.deepcopy(item)
    calls = checked(trace.model_calls, item)
    assert calls[0]["request"][1]["content"] == "Previous answer"
    assert calls[0]["provider_meta"] == {"x": [1, 2]}
    assert "request_recorded" not in calls[0]
    assert calls[1] == {"request": [], "response": [], "tools": []}
    assert checked(trace.model_calls, item, offset=1, limit=1) == [calls[1]]
    assert checked(trace.model_calls, item, limit=0) == []
    calls[0]["provider_meta"]["x"].append(3)
    assert item == before
    item["response_text"]["model_call_evidence"]["calls"][0]["request"] = [99]
    with pytest.raises(ValueError, match="invalid model-call pool references"):
        trace.model_calls(item)


def test_parse_failure_is_discriminated_and_empty_helpers_remain_compatible():
    assert checked(trace.parse, {}) == {"ok": False}
    bad = {"response_text": "This is plain prose, not a trajectory."}
    assert checked(trace.parse, bad)["ok"] is False
    assert "warning" in trace.parse(bad)
    for helper in (trace.steps, trace.tool_calls, trace.input_messages, trace.input_tools,
                   trace.model_calls):
        assert checked(helper, {}) == []
    for helper in (trace.reasoning, trace.actions, trace.observations, trace.final_message,
                   trace.input_text, trace.evidence_text):
        assert checked(helper, {}) == ""


def test_passage_contract_distinguishes_no_match_missing_field_and_capped_evidence():
    text = "🙂 prefix\nmatch one\nmatch two\nend"
    item = {"reasoning_text": text}
    result = checked(trace.regex_passages, item, "reasoning_text", "match", max_passages=1)
    record, = result["value"]
    assert result["status"] == "ok" and not result["complete"]
    assert text[record["start"]:record["end"]] == record["quote"] == "match"
    assert record["start_line"] == 2 and record["start_column"] == 1
    assert record["context"] == text[record["context_start"]:record["context_end"]]
    assert not checked(trace.render_passages, result)["complete"]
    empty = checked(trace.regex_passages, item, "reasoning_text", "absent")
    assert empty == {"status": "ok", "complete": True, "value": [], "reason": ""}
    assert checked(trace.render_passages, empty)["complete"]
    unavailable = checked(trace.regex_passages, {}, "reasoning_text", "match")
    assert unavailable["status"] == "unavailable" and not unavailable["complete"]
    assert not checked(trace.render_passages, unavailable)["complete"]
    long = checked(trace.regex_passages, {"reasoning_text": "a" * 100}, "reasoning_text", "a+")
    assert checked(trace.render_passages, long, max_chars=64) == {
        "text": "", "complete": False, "budget_chars": 64,
    }


def test_public_contracts_can_be_imported_in_a_real_detector():
    assert run_detector('''
from autofa.trace import LocatedToolCall, ParseSummary, Step, parse, steps, tool_calls
def check(item):
    summary: ParseSummary = parse(item)
    event: Step = steps(item)[1]
    call: LocatedToolCall = tool_calls(item)[0]
    return summary['ok'] and event.has_reasoning and call['step'] == event.index
''', item_with_trajectory())


# --- execution records ---


def test_record_is_full_structured_evidence_with_no_budget_or_unknown_field_loss():
    item = execution_item()
    result = checked(trace.record, item)
    expected = item["response_text"]["execution"]
    for key, value in expected.items():
        if key != "availability":
            assert result[key] == value
    assert len(result["events"][0]["parts"][0]["text"]) > 100_000
    assert result["availability"]["artifact_texts"]["status"] == "partial"
    for helper, key in [(trace.attempts, "attempts"), (trace.tool_definitions, "tool_definitions"),
                        (trace.provenance, "provenance"), (trace.relations, "relations")]:
        assert checked(helper, item) == expected[key]


def test_event_lane_selection_is_explicit_and_preserves_source_order():
    item = execution_item()
    assert [e["event_idx"] for e in checked(trace.events, item)] == [41, 42]
    assert [e["event_idx"] for e in checked(trace.events, item, lane="")] == [41]
    assert [e["event_idx"] for e in checked(trace.events, item, lane="worker/2")] == [42]
    assert checked(trace.events, item, lane="absent") == []
    assert trace.coverage(item)["events"]["lanes"] == ["", "worker/2"]


def test_helpers_return_independent_copies_that_cannot_mutate_parser_or_source():
    item = execution_item()
    before = copy.deepcopy(item)
    event = checked(trace.events, item)[0]
    event["provider_extension"]["cache"].append("changed")
    event["parts"][0]["text"] = "changed"
    trace.metadata(item)["provider_extension"]["worker"].append(7)
    trace.artifacts(item)[0]["extension"] = {"new": "change"}
    trace.record(item)["events"][0]["provider_extension"]["cache"].append("changed")
    trace.coverage(item)["events"]["lanes"].append("invented")
    step = trace.steps(item)[0]
    step.metadata["provider_extension"]["cache"].append("changed")
    step.parts[0]["text"] = "changed"
    trace.steps(item)[1].observation_data["exit_code"] = 0
    again = trace.steps(item)
    assert again[0].metadata["provider_extension"]["cache"] == ["cold"]
    assert len(again[0].parts[0]["text"]) > 100_000
    assert again[1].observation_data["exit_code"] == 1
    assert item == before


def test_recorded_empty_is_distinct_from_missing_unavailable_and_partial_resources():
    item = {"response_text": {"execution": {
        "events": [], "attempts": [], "artifacts": [],
        "availability": {"artifacts": {"status": "unavailable", "reason": "Table missing"},
                         "attempts": {"status": "partial", "reason": "Only one shard imported"}},
    }}}
    result = checked(trace.record, item)
    assert result["events"] == result["attempts"] == result["artifacts"] == []
    assert result["availability"]["events"]["status"] == "available"
    assert result["availability"]["attempts"]["status"] == "partial"
    assert result["availability"]["artifacts"] == {
        "status": "unavailable", "reason": "Table missing",
    }
    assert result["availability"]["relations"]["status"] == "unavailable"
    absent = trace.record({})
    assert all(v["status"] == "unavailable" for v in absent["availability"].values())
    assert absent["rollout"] == {} and absent["events"] == []


def test_legacy_parsed_projection_is_partial_without_fabricated_source_identities():
    envelope = {"steps": [{"step_id": 42, "source": "assistant", "message": "hello",
                           "tool_calls": [{"name": "test", "arguments": {"x": [1]}}]}]}
    item = {"response_text": json.dumps(envelope)}
    result = checked(trace.record, item)
    event, = result["events"]
    assert event["step_index"] == 1 and event["step_id"] == 42
    assert "event_idx" not in event and "lane" not in event
    assert result["tool_calls"][0]["step"] == 1
    assert result["availability"]["events"]["status"] == "partial"
    assert result["availability"]["tool_calls"]["status"] == "partial"
    assert result["availability"]["rollout"]["status"] == "unavailable"
    assert trace.events(item, lane="") == []
    assert trace.metadata(item) == {} and trace.artifacts(item) == []


def test_malformed_resource_values_are_retained_with_explicit_unavailable_scope():
    item = {"response_text": {"execution": {
        "rollout": None, "events": [{"event_idx": 1, "extension": [None]}, "opaque"],
        "attempts": 17,
    }}}
    result = checked(trace.record, item)
    assert result["rollout"] == {} and result["attempts"] == []
    assert result["events"] == [{"event_idx": 1, "extension": [None]}]
    assert result["unparsed_resources"] == item["response_text"]["execution"]
    assert result["availability"]["events"]["status"] == "partial"
    assert result["availability"]["rollout"]["status"] == "unavailable"


# --- command prefixes ---


def assistant(*commands, **extra):
    return {"source": "assistant", "message": json.dumps({
        "commands": [{"keystrokes": command} for command in commands],
    }), **extra}


def trace_item(*events, **extra):
    return {"response_text": {"steps": list(events), **extra}}


def prefix(value, step=1, quote="mutate original"):
    return trace.command_prefix(value, step_index=step, command_quote=quote)


def test_same_keystrokes_prefix_includes_backup_before_mutation():
    value = trace_item(assistant("cp original backup && mutate original; cleanup"))
    before = copy.deepcopy(value)
    result = prefix(value)
    assert result["complete"]
    assert result["commands"] == [{"step_index": 1,
        "path": "message.commands[0].keystrokes", "text": "cp original backup && "}]
    assert result["anchor"]["start"] == len("cp original backup && ")
    assert result["anchor"]["quote"] == "mutate original"
    assert value == before


def test_prior_steps_and_prior_commands_in_same_event_are_preserved_in_order():
    result = prefix(trace_item(
        assistant("inspect original"), {"source": "tool", "message": "observed"},
        assistant("cp original backup", "prepare; mutate original", "later command"),
    ), step=3)
    assert result["complete"]
    assert [c["text"] for c in result["commands"]] == [
        "inspect original", "cp original backup", "prepare; ",
    ]
    assert result["anchor"]["path"] == "message.commands[1].keystrokes"
    assert [c["step_index"] for c in result["commands"]] == [1, 3, 3]


def test_native_calls_are_ordered_and_identical_message_mirrors_are_not_duplicate_anchors():
    calls = [{"name": "terminal", "arguments": {"command": text}}
             for text in ["cp original backup", "mutate original"]]
    for event in ({"source": "assistant", "tool_calls": calls},
                  assistant("cp original backup", "mutate original", tool_calls=calls)):
        result = prefix(trace_item(event))
        assert result["complete"]
        assert [c["text"] for c in result["commands"]] == ["cp original backup"]


def test_encoded_native_batch_preserves_unicode_offsets_and_all_prior_segments():
    command = "cp café backup; mutate café"
    value = trace_item({"source": "agent", "tool_calls": [{"function": {
        "name": "terminal", "arguments": json.dumps({"commands": [
            {"keystrokes": "prepare"}, {"keystrokes": command},
        ]}),
    }}]})
    result = prefix(value, quote="mutate café")
    assert result["complete"]
    assert [c["text"] for c in result["commands"]] == ["prepare", "cp café backup; "]
    assert command[result["anchor"]["start"]:result["anchor"]["end"]] == "mutate café"


def test_nonexecution_text_and_request_history_do_not_enter_the_command_prefix():
    value = trace_item({"source": "user", "message": "cp original backup"},
                 {"source": "assistant", "message": "I could copy original into backup."},
                 assistant("mutate original"),
                 model_call_evidence={"messages": [{"role": "assistant", "content":
                                                      "cp original backup"}]})
    assert prefix(value, step=3)["commands"] == []
    assert prefix(value, step=3)["complete"]


def test_normal_reasoning_events_and_role_alias_do_not_block_known_commands():
    value = trace_item({"source": "assistant", "reasoning_content": "Consider a protected copy.",
                  "parts": [{"part_type": "reasoning", "text": "Think", "chars": 5}]},
                 {"role": "assistant", "message": assistant("mutate original")["message"]})
    assert prefix(value, step=2)["complete"]
    assert prefix(value, step=2)["commands"] == []


@pytest.mark.parametrize("event,quote", [
    (assistant("aaaa"), "aaa"),
    (assistant("something else"), "mutate original"),
    (assistant("mutate original", tool_calls=[{"name": "terminal", "arguments": {
        "command": "cp original backup; mutate original"}}]), "mutate original"),
])
def test_missing_repeated_or_conflicting_anchor_is_unavailable(event, quote):
    result = prefix(trace_item(event), quote=quote)
    assert not result["complete"]
    assert result["commands"] == [] and result["anchor"] is None and result["reason"]


@pytest.mark.parametrize("earlier", [
    assistant("cp original backup", length_truncated=True),
    assistant("cp original backup", metadata={"finish_reason": "length"}),
    assistant("cp original backup", parts=[{"part_type": "text", "text": "cp", "chars": 9}]),
    {"source": "assistant", "message": '{"commands":['},
    {"source": "assistant", "tool_calls": [{"name": "terminal", "arguments": "{invalid"}]},
    {"source": "assistant", "tool_calls": [{"name": "copy_file", "arguments": {"src": "a"}}]},
    {"source": "assistant", "tool_calls": [{"name": "terminal", "arguments": {"cmd": "cp"},
        "args_full": '{"cmd":"cp"}', "args_chars": 80}]},
    {"source": "unknown", "message": "cp original backup"},
    {"message": assistant("cp original backup")["message"]},  # no recorded actor
])
def test_incomplete_or_unsupported_earlier_commands_cannot_prove_absence(earlier):
    assert not prefix(trace_item(earlier, assistant("mutate original")), step=2)["complete"]


def test_incomplete_or_undercounted_events_after_anchor_do_not_invalidate_known_prefix():
    value = trace_item(assistant("cp original backup; mutate original"),
                       assistant("later", length_truncated=True),
                       assistant("later", n_parts=1, metadata={"n_tool_calls": 1}))
    # JSON-encoded steps keep their original actor and command data.
    value["response_text"]["steps"] = json.dumps(value["response_text"]["steps"])
    assert prefix(value)["complete"]


@pytest.mark.parametrize("collection,count_key,location,retained", [
    ("tool_calls", "n_tool_calls", "event", None), ("parts", "n_parts", "metadata", 0),
])
def test_declared_missing_calls_and_parts_make_prefix_unavailable(
    collection, count_key, location, retained,
):
    earlier = {"source": "assistant", "message": "Checking the original."}
    if retained is not None:
        records = [{"name": "terminal", "arguments": {"cmd": "cp original backup"}}]
        if collection == "parts":
            records = [{"part_type": "text", "text": "Checking the original."}]
        earlier[collection] = records[:retained]
    declaration = earlier if location == "event" else earlier.setdefault("metadata", {})
    declaration[count_key] = (retained or 0) + 1
    result = prefix(trace_item(earlier, assistant("mutate original")), step=2)
    assert result["status"] == "unavailable" and not result["complete"]
    assert result["commands"] == [] and result["anchor"] is None
    assert count_key in result["reason"]


@pytest.mark.parametrize("count_key,location,count", [
    ("n_tool_calls", "event", -1), ("n_parts", "metadata", True),
])
def test_malformed_declared_counts_cannot_certify_an_empty_prefix(count_key, location, count):
    event = assistant("mutate original")
    declaration = event if location == "event" else event.setdefault("metadata", {})
    declaration[count_key] = count
    result = prefix(trace_item(event))
    assert result["status"] == "unavailable" and not result["complete"]
    assert result["commands"] == [] and count_key in result["reason"]


@pytest.mark.parametrize("location,count", [("event", 0), ("metadata", None)])
def test_absent_collections_allow_zero_or_unspecified_counts(location, count):
    event = assistant("cp original backup; mutate original")
    declaration = event if location == "event" else event.setdefault("metadata", {})
    declaration.update(n_tool_calls=count, n_parts=count)
    result = prefix(trace_item(event))
    assert result["complete"]
    assert [command["text"] for command in result["commands"]] == ["cp original backup; "]


def test_complete_declared_counts_survive_json_call_normalization():
    event = {"source": "assistant", "tool_calls": json.dumps([{"function": {
        "name": "terminal", "arguments": json.dumps({"cmd": "cp original backup"}),
    }}, {"function": {
        "name": "terminal", "arguments": json.dumps({"cmd": "mutate original"}),
    }}]), "parts": [{"part_type": "reasoning", "text": "Inspect the original."}]}
    event.update(n_tool_calls=2, n_parts=1, metadata={"n_tool_calls": 2, "n_parts": 1})
    result = prefix(trace_item(event))
    assert result["complete"]
    assert [command["text"] for command in result["commands"]] == ["cp original backup"]


def test_retained_records_cannot_exceed_declared_counts():
    event = assistant("mutate original", n_tool_calls=0, tool_calls=[{
        "name": "terminal", "arguments": {"cmd": "mutate original"},
    }])
    result = prefix(trace_item(event))
    assert result["status"] == "unavailable" and "n_tool_calls" in result["reason"]


def test_message_format_keeps_missing_call_declarations_for_prefix_validation():
    value = {"response_text": {"messages": [{
        "role": "assistant", "content": assistant("mutate original")["message"],
        "n_tool_calls": 1,
    }]}}
    result = prefix(value)
    assert result["status"] == "unavailable" and "n_tool_calls" in result["reason"]


@pytest.mark.parametrize("value,step", [
    ({"response_text": '{"steps": ['}, 1),
    (trace_item(assistant("mutate original"), truncated=True), 1),
    (trace_item({"source": "tool", "message": "mutate original"}), 1),
    (trace_item(assistant("mutate original")), 2),
])
def test_missing_trace_scope_or_step_is_unavailable(value, step):
    assert not prefix(value, step=step)["complete"]


@pytest.mark.parametrize("limitations", [
    {"missing events": 1}, ["missing events"], "missing events",
    # Importer labels from source normalization for dropped events.
    {"excluded event kind/actor: action/assistant": 1}, ["unsupported event role: unknown"],
])
def test_recorded_event_gaps_prevent_command_prefix_certification(limitations):
    value = trace_item(assistant("mutate original"), evidence_scope={"limitations": limitations})
    result = prefix(value)
    assert result["status"] == "unavailable" and not result["complete"]
    assert result["commands"] == [] and result["anchor"] is None


@pytest.mark.parametrize("limitations", [
    {"missing events": "0"}, {"missing events": float("nan")}, [False],
])
def test_malformed_event_limitations_cannot_certify_absence(limitations):
    value = trace_item(assistant("mutate original"), evidence_scope={"limitations": limitations})
    result = prefix(value)
    assert result["status"] == "unavailable" and not result["complete"]
    assert "Malformed recorded evidence limitation" in result["reason"]


@pytest.mark.parametrize("collection,declaration", [
    ("messages", {"truncated": True}),
    ("conversation", {"evidence_scope": {"limitations": {"command events unavailable": 1}}}),
    ("output", {"_parse_limitations": ["command event omitted"]}),
])
def test_messages_envelope_retains_completeness_declarations(collection, declaration):
    value = {"response_text": {collection: [{"role": "assistant", "content":
                                            assistant("mutate original")["message"]}],
                               **declaration}}
    before = copy.deepcopy(value)
    result = prefix(value)
    assert result["status"] == "unavailable" and not result["complete"]
    assert result["commands"] == [] and result["anchor"] is None
    assert value == before


def test_complete_messages_envelope_preserves_ordered_command_prefix():
    value = {"response_text": {"conversation": [{"role": "assistant", "content":
                assistant("cp original backup; mutate original")["message"]}],
                "truncated": False, "evidence_scope": {"limitations": {
                    "missing events": 0, "encrypted reasoning unavailable": 1}}}}
    result = prefix(value)
    assert result["complete"]
    assert [command["text"] for command in result["commands"]] == ["cp original backup; "]


def test_unrelated_scope_limits_and_command_words_are_not_missing_event_flags():
    value = trace_item(assistant("echo 'missing events'; mutate original"), evidence_scope={
        "limitations": {"encrypted reasoning unavailable": 1, "missing artifacts": 1,
                        "missing events": 0, "raw model response not recorded": 1},
    })
    result = prefix(value)
    assert result["complete"]
    assert result["commands"][0]["text"] == "echo 'missing events'; "


@pytest.mark.parametrize("step,quote", [(0, "x"), (1, " ")])
def test_bad_anchor_arguments_are_programming_errors(step, quote):
    with pytest.raises(ValueError):
        prefix(trace_item(assistant("mutate original")), step=step, quote=quote)


def test_helper_is_available_inside_real_detector_sandbox_and_generated_reference():
    source = '''
from autofa.trace import command_prefix
def detect(item):
    result = command_prefix(item, step_index=1, command_quote="mutate original")
    if not result['complete']:
        return library.abstain(result['reason'])
    return any('cp original backup' in command['text'] for command in result['commands'])
'''
    assert run_detector(source, trace_item(assistant("cp original backup; mutate original")))
    reference = render_trace_api_reference("command_prefix")
    assert "-> CommandPrefix" in reference
    assert "### `CommandFragment`" in reference and "### `CommandAnchor`" in reference


SOURCE_LOSS = Path(__file__).parent / "fixtures/detectors/source_loss_before_backup.py"


class SourceLossLLM:
    available = True

    def classify(self, prompt, evidence, **kwargs):
        # Deliberately miss the backup, just like the failing generated classifier.
        return json.dumps({"label": "causal_loss", "mutation_step": 2,
                           "command_quote": "mutate original", "source_path": "original",
                           "prior_backup": "absent"})


@pytest.mark.parametrize("commands,expected", [
    (("cp original backup; mutate original",), "abstain"),
    (("mutate original; cp original backup",), True),
])
def test_corrected_candidate_catches_same_batch_gap_without_assuming_backup_success(
    commands, expected,
):
    source = SOURCE_LOSS.read_text()
    value = trace_item({"source": "user", "message": "Recover data from original."},
                 assistant(*commands),
                 {"source": "tool", "message": "Original data lost. Recovery failed."},
                 {"source": "assistant", "message": "Finished with required data missing."})
    if expected == "abstain":
        with pytest.raises(DetectorAbstained, match="backup candidate"):
            run_detector(source, value, llm=SourceLossLLM())
    else:
        assert run_detector(source, value, llm=SourceLossLLM()) is expected
