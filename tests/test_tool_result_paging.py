# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tool calls validate arguments, cancel cleanly, and save large results for exact paging."""

import asyncio
import hashlib
import inspect
import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import jsonschema
import pytest

from autofa.llm.hub import HubLLM
from autofa.mcp import runtime, server
from autofa.mcp.server import (
    ENVELOPE_HEAD_CHARS,
    MCP_HARD_CEILING_CHARS,
    MCP_SOFT_WARN_CHARS,
    clamp_mcp_payload,
)
from autofa.mcp.tool_results import (
    DESCRIPTION_CHILDREN,
    PAGE_CHARS,
    describe_result_value,
    read_tool_result,
    save_tool_result,
)
from autofa.metrics import claim_compiler, claude_judge
from autofa.util.cancellation import Cancellation, cancellation_scope
from autofa.util.work_queue import ordered_bounded_map


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A minimal live runtime whose run directory holds saved results and operation records."""
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(
        run=SimpleNamespace(run_id="interrupted-run", run_dir=lambda: tmp_path),
    ))
    return tmp_path


def _operation(directory):
    return json.loads(next((directory / "operations").glob("*.json")).read_text())


def _read_all(result_id, pointer="", limit=7):
    chunks, offset = [], 0
    while offset is not None:
        page = read_tool_result(result_id, pointer=pointer, offset=offset, limit=limit)
        assert page["ok"]
        assert page["scope"] == ("selection" if pointer else "whole_result")
        assert page["view"] == "json" and page["offset_unit"] == "characters"
        assert page["full_result_complete"] == (not pointer and page["complete"])
        assert page["complete"] == (offset == 0 and page["next_offset"] is None)
        assert page["pointer"] == pointer and page["result_id"] == result_id
        chunks.append(page["text"])
        offset = page["next_offset"]
    text = "".join(chunks)
    assert len(text) == page["total_chars"]
    return text


# Native schemas and argument repair.

def test_every_claude_tool_has_native_schema_and_full_documentation(monkeypatch):
    import claude_agent_sdk

    tools = []
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server",
                        lambda **kwargs: tools.extend(kwargs["tools"]))
    server.build_mcp_server()
    native = {spec["name"]: spec for spec in server.codex_tool_specs()}
    assert {tool.name for tool in tools} == set(native)
    for tool in tools:
        assert tool.input_schema == native[tool.name]["inputSchema"]
        assert tool.description == native[tool.name]["description"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"made_up_parameter": 1}, tool.input_schema)

    evaluate = next(tool for tool in tools if tool.name == "evaluate_functions")
    args = {"names": ["candidate"], "mode": "capped", "limit": 200}
    jsonschema.validate(args, evaluate.input_schema)
    for value in ({"args": args}, {**args, "llm_eval_sample": 800}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(value, evaluate.input_schema)


@pytest.mark.parametrize("wrapped", [True, False])
def test_invalid_evaluation_arguments_return_schema_before_any_work(monkeypatch, wrapped):
    from autofa.session import recovery

    def never_execute(*args, **kwargs):
        pytest.fail("Invalid arguments must not reserve work or make model calls")

    monkeypatch.setattr(recovery, "execute_operation", never_execute)
    args = {"names": ["candidate"], "llm_eval_sample": 800}
    if wrapped:
        # The pre-GA Claude {"args": ...} envelope is rejected even around valid arguments.
        args = {"args": {"names": ["candidate"], "mode": "capped"}}
    response = asyncio.run(server.invoke_tool("evaluate_functions", args))
    result = json.loads(response["content"][0]["text"])
    assert result["ok"] is False
    assert ("'args'" if wrapped else "'llm_eval_sample'") in result["error"]
    assert result["allowed_arguments"] == ["names", "limit", "mode", "chunk_size", "within"]
    assert result["required_arguments"] == []
    assert "mode" in result["argument_schema"]["properties"]


def test_missing_required_argument_returns_an_executable_signature():
    response = asyncio.run(server.invoke_tool("get_item", {}))
    result = json.loads(response["content"][0]["text"])
    assert result["required_arguments"] == ["item_id"]
    assert result["allowed_arguments"] == ["item_id", "include_trace", "max_chars"]
    assert result["argument_schema"]["properties"]["item_id"]["type"] == "string"


# Cancellation stops the worker pipeline and child judge sessions.

@pytest.mark.parametrize("succeeded", [True, False])
def test_cancel_after_tool_returns_retains_committed_result(run_dir, monkeypatch, succeeded):
    from autofa.session.recovery import save_turn
    from autofa.util.cancellation import _CURRENT, cancellation_guard

    committed = threading.Event()
    saved = run_dir / "library.json"
    result = {"ok": succeeded, "removed": "old-detector", "detail": "Committed result"}
    save_turn(run_dir, prompt="Continue discovery", session_id="saved", state="running")

    def mutate():
        with cancellation_guard():
            saved.write_text('{"functions": []}')
        committed.set()
        # Cancellation arrives after the mutation, before its caller can record
        # the return value. Both successful and reported-failed outcomes survive.
        assert _CURRENT.get().event.wait(3)
        return result

    monkeypatch.setitem(server.TOOL_CALLABLES, "remove_function", mutate)

    async def scenario():
        task = asyncio.create_task(server.invoke_tool("remove_function", {}))
        assert await asyncio.to_thread(committed.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)

    asyncio.run(scenario())
    assert json.loads(saved.read_text()) == {"functions": []}
    operation = _operation(run_dir)
    assert operation["state"] == ("complete" if succeeded else "failed")
    assert operation["result"] == result
    assert operation["cancelled_after_completion"] is True


@pytest.mark.parametrize("surface", ["tool", "final_scoring"])
def test_cancelling_during_checkpoint_keeps_event_loop_responsive(run_dir, monkeypatch, surface):
    from io import StringIO

    from rich.console import Console

    from autofa.session import finalization
    from autofa.util.cancellation import cancellation_guard

    checkpoint_started, cancel_requested, release = (
        threading.Event(), threading.Event(), threading.Event(),
    )
    released_by_event_loop = []

    class ObservedCancellation(Cancellation):
        def cancel(self):
            cancel_requested.set()
            super().cancel()

    def checkpoint(*args):
        with cancellation_guard():
            checkpoint_started.set()
            # A timeout bounds the old deadlock: with synchronous cancellation,
            # the event loop cannot release this simulated checkpoint I/O.
            released_by_event_loop.append(release.wait(3))
        return {"ok": True}

    if surface == "tool":
        monkeypatch.setattr(server, "Cancellation", ObservedCancellation)
        monkeypatch.setitem(server.TOOL_CALLABLES, "library_save", checkpoint)
    else:
        monkeypatch.setattr(finalization, "Cancellation", ObservedCancellation)
        monkeypatch.setattr(finalization, "evaluate_final_library", checkpoint)

    async def scenario():
        coroutine = (server.invoke_tool("library_save", {}) if surface == "tool"
                     else finalization.finalize_discovery(Console(file=StringIO())))
        task = asyncio.create_task(coroutine)
        try:
            assert await asyncio.to_thread(checkpoint_started.wait, 3)
            task.cancel()
            assert await asyncio.to_thread(cancel_requested.wait, 3)
            if surface == "tool":
                # A second cancellation while acquiring the token lock must not
                # release the runtime before the first worker finishes.
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        finally:
            release.set()

    asyncio.run(scenario())
    assert released_by_event_loop == [True]


def test_cancel_stops_detector_queue_and_late_scores(run_dir, monkeypatch):
    started, finish, cancelled = threading.Event(), threading.Event(), threading.Event()
    calls, stages = [], []
    saved = run_dir / "library.json"
    saved.write_text('{"previous": "completed checkpoint"}')
    previous = saved.read_bytes()

    class ObservedCancellation(Cancellation):
        def cancel(self):
            super().cancel()
            cancelled.set()

    monkeypatch.setattr(server, "Cancellation", ObservedCancellation)

    def create(**kwargs):
        calls.append(kwargs)
        started.set()
        assert finish.wait(5)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content="yes"),
        )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    hub = HubLLM(client, model="fake")

    def submit_function():
        list(ordered_bounded_map(
            lambda i: hub.classify("Test", f"row {i}"), range(200),
            workers=2, max_pending=4, max_bytes=100, size_of=lambda _: 1,
        ))
        stages.append("judge")
        saved.write_text('{"late": "result"}')
        return {"ok": True}

    monkeypatch.setitem(server.TOOL_CALLABLES, "submit_function", submit_function)

    async def scenario():
        task = asyncio.create_task(server.invoke_tool("submit_function", {}))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            # The cancellation lock is acquired off the event loop. Wait for
            # that acknowledgement before completing the simulated HTTP call.
            assert await asyncio.to_thread(cancelled.wait, 3)
            assert not task.done()  # the already sent HTTP request must finish safely
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
    assert 1 <= len(calls) <= 2
    assert stages == []
    assert saved.read_bytes() == previous
    assert _operation(run_dir)["state"] == "cancelled"
    assert "result" not in _operation(run_dir)


def test_cancellation_closes_active_async_child(run_dir, monkeypatch):
    started, closed = threading.Event(), threading.Event()

    async def child(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(claim_compiler, "_compiler_session", child)
    monkeypatch.setitem(server.TOOL_CALLABLES, "screen_function", claim_compiler._run_compiler)

    async def scenario():
        task = asyncio.create_task(server.invoke_tool("screen_function", {}))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert closed.is_set()

    asyncio.run(scenario())
    assert _operation(run_dir)["state"] == "cancelled"


def test_cancellation_is_isolated_and_blocks_retry():
    from autofa.llm.hub import chat_completions_safe

    token = Cancellation()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        token.cancel()
        error = RuntimeError("temperature unsupported")
        error.status_code = 400
        error.body = {"error": {"param": "temperature", "code": "unsupported_parameter"}}
        raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
        chat_completions_safe(client, model="fake", messages=[], temperature=0)
    assert len(calls) == 1

    client.chat.completions.create = lambda **kwargs: "unrelated session result"
    assert chat_completions_safe(client, model="fake", messages=[]) == "unrelated session result"


@pytest.mark.parametrize("surface,when", [
    ("hub", "before"), ("hub", "during_error"), ("captured_hub", "during"),
    ("captured_raw", "during"),
])
def test_raw_chat_completion_stops_before_next_call_or_capture(surface, when):
    from autofa.core.localisation import CapturingLLM

    token = Cancellation()
    calls, captures = [], []

    def create(**kwargs):
        calls.append(kwargs)
        token.cancel()
        if when == "during_error":
            raise RuntimeError("request failed after cancellation")
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="yes"),
        )])

    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = raw if surface == "captured_raw" else HubLLM(raw, model="fake")
    if surface.startswith("captured"):
        client = CapturingLLM(client, SimpleNamespace(llm_call=lambda *args: captures.append(args)))
    if when == "before":
        token.cancel()
    with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
        # Detectors sometimes issue several raw requests for one example.
        # Cancellation must abort the first response, not merely the next row.
        client.chat.completions.create(model="fake", messages=[])
        client.chat.completions.create(model="fake", messages=[])
    assert len(calls) == (0 if when == "before" else 1)
    assert captures == []

    # The proxy reads the current tool token; it cannot poison other sessions.
    raw.chat.completions.create = lambda **kwargs: "another tool"
    assert client.chat.completions.create(model="fake", messages=[]) == "another tool"


def test_cancel_propagates_into_parallel_judges(run_dir, monkeypatch, stub_claim_compiler):
    started = threading.Event()
    lock = threading.Lock()
    launched, closed = [], []

    async def child(*, review, **kwargs):
        item_id = review.snapshot["item_id"]
        with lock:
            launched.append(item_id)
            if len(launched) == 2:
                started.set()
        try:
            await asyncio.Event().wait()
        finally:
            with lock:
                closed.append(item_id)

    def judge_function():
        return claude_judge.judge_behavior_with_claude(
            name="test", source="def detect(item): return True",
            description="The assistant bypasses a computation.",
            mechanism="The output is fabricated.", cwd=run_dir, model="fake",
            items=[{"_item_id": str(i), "response_text": "evidence"} for i in range(8)],
            assigned_labels=[True] * 8, workers=2,
        )

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", child)
    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", judge_function)

    async def scenario():
        task = asyncio.create_task(server.invoke_tool("judge_function", {}))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
    assert sorted(launched) == sorted(closed) == ["0", "1"]
    assert _operation(run_dir)["state"] == "cancelled"
    progress = json.loads(next((run_dir / "behavior_judge").glob("*/progress.json")).read_text())
    assert progress["state"] == "interrupted"
    assert progress["completed"] == 0
    assert progress["active_items"] == []


def test_cancelled_runtime_cannot_overwrite_saved_library_or_scores(tmp_path):
    from autofa.core.library import FunctionLibrary

    run = SimpleNamespace(
        run_id="cancelled-run", exploration_base_seed=42, exploration_sequence_index=0,
        run_dir=lambda: tmp_path, library_model_id=lambda: "model", benchmark_label=lambda: "bench",
    )
    rt = runtime.Runtime(dataset=None, library=FunctionLibrary(), run=run, config=None)
    library_path = rt.persist_library()
    history_path = rt.persist_scores_history()
    previous_library, previous_history = library_path.read_bytes(), history_path.read_bytes()
    rt.scores_history.append({"uncommitted": "result"})
    # Save preparation would normalize these fields and advance the library cursor.
    # Cancellation must prevent those in-memory mutations as well as disk writes.
    rt.library.model_id = "uncommitted-model"
    run.exploration_sequence_index += 1
    live_library = deepcopy(rt.library)
    live_run = vars(run).copy()
    token = Cancellation()
    token.cancel()
    with cancellation_scope(token):
        with pytest.raises(asyncio.CancelledError):
            rt.persist_library()
        with pytest.raises(asyncio.CancelledError):
            rt.persist_scores_history()
    assert library_path.read_bytes() == previous_library
    assert history_path.read_bytes() == previous_history
    assert rt.library == live_library
    assert vars(run) == live_run


# Exact saved-result selection never hides changes to source evidence or scope.

def test_select_exact_values_and_reconstruct_original_snapshot(run_dir):
    value = {"verdict": "é雪🧪\\\"\n", "evidence": ["snow 雪", None, False, 0]}
    original = {"large_unneeded_trace": "before" * 4000, "feedback": {"chosen": value}}
    saved = save_tool_result(original)
    path = run_dir / "tool_results" / f"{saved['result_id']}.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    stored_text = json.loads(path.read_text())["text"]
    assert saved["result_id"] == hashlib.sha256(stored_text.encode()).hexdigest()

    selected = _read_all(saved["result_id"], "/feedback/chosen")

    assert json.loads(selected) == value
    # A present but falsy scalar is selected exactly, never treated as missing.
    assert _read_all(saved["result_id"], "/feedback/chosen/evidence/1") == "null"
    assert _read_all(saved["result_id"], "/feedback/chosen/evidence/3") == "0"
    whole =read_tool_result(saved["result_id"], pointer="/feedback/chosen")["whole_result_chars"]
    assert whole == saved["total_chars"]
    assert _read_all(saved["result_id"], limit=PAGE_CHARS) == stored_text
    assert json.loads(stored_text) == original
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert save_tool_result(original) == saved


@pytest.mark.parametrize("pointer,expected", [
    ("/a~1b/~0name/1/", "escaped / ~ key"), ("/~01", "literal tilde-one"),
])
def test_rfc6901_escaped_keys_array_indices_and_empty_keys(run_dir, pointer, expected):
    saved = save_tool_result({"a/b": {"~name": [None, {"": "escaped / ~ key"}]},
                              "": True, "~1": "literal tilde-one"})
    assert json.loads(_read_all(saved["result_id"], pointer)) == expected


@pytest.mark.parametrize("pointer", [
    "#/a", "/bad~escape", "/missing", "/a/0/x", "/a/01", "/a/2", "/a/" + "9" * 5000,
], ids=["no-slash", "bad-escape", "missing-key", "through-scalar", "leading-zero",
        "out-of-range", "huge-index"])
def test_invalid_selection_returns_explicit_error(run_dir, pointer):
    saved = save_tool_result({"a": [None, 1]})
    result = read_tool_result(saved["result_id"], pointer=pointer)
    assert result["ok"] is False and "pointer" in result["error"].lower()
    assert "text" not in result


@pytest.mark.parametrize("arguments", [
    {"result_id": "../config"}, {"offset": -1}, {"offset": 1, "describe": True},
    {"describe": "yes"},
], ids=["path-traversal", "negative-offset", "describe-offset-past-children", "describe-type"])
def test_invalid_saved_result_requests_fail(run_dir, arguments):
    saved = save_tool_result(None)
    assert not read_tool_result(**{"result_id": saved["result_id"], **arguments})["ok"]


def test_describe_reveals_nested_paths_without_reading_large_evidence(run_dir):
    original = {"large": "SECRET_EVIDENCE" * 10000,
                "feedback": {"a/b": [{"~reason": None}], "allowed_actions": ["retain"]}}
    saved = save_tool_result(original)
    root = read_tool_result(saved["result_id"], describe=True)
    assert root["ok"] and root["view"] == "structure"
    assert root["complete"] and not root["full_result_complete"]
    assert "SECRET_EVIDENCE" not in json.dumps(root) and "text" not in root
    assert root["total_chars"] == root["whole_result_chars"] == saved["total_chars"]
    assert {child["pointer"] for child in root["children"]} == {"/large", "/feedback"}
    nested = read_tool_result(saved["result_id"], pointer="/feedback", describe=True)
    assert nested["scope"] == "selection" and nested["value_type"] == "object"
    assert nested["children"] == [
        {"pointer": "/feedback/a~1b", "value_type": "array", "total_chars": 19,
         "total_children": 1},
        {"pointer": "/feedback/allowed_actions", "value_type": "array", "total_chars": 10,
         "total_children": 1},
    ]
    array = read_tool_result(saved["result_id"], pointer="/feedback/a~1b", describe=True)
    child = array["children"][0]
    assert child["pointer"] == "/feedback/a~1b/0"
    assert json.loads(_read_all(saved["result_id"], child["pointer"])) == {"~reason": None}


def test_structural_pagination_is_bounded_and_all_children_remain_accessible(run_dir):
    original = {f"key-{index}": index for index in range(DESCRIPTION_CHILDREN * 3)}
    saved = save_tool_result(original)
    children, offset = [], 0
    while offset is not None:
        page = read_tool_result(saved["result_id"], describe=True, offset=offset, limit=10000)
        assert page["ok"] and page["offset_unit"] == "children"
        assert not page["full_result_complete"]
        assert len(page["children"]) <= DESCRIPTION_CHILDREN
        assert len(json.dumps(page)) < PAGE_CHARS + 1000
        assert page["total_children"] == len(original)
        children.extend(page["children"])
        offset = page["next_offset"]
    assert [child["pointer"] for child in children] == ["/" + key for key in original]


def test_description_budget_never_truncates_a_pointer():
    value = {"a" * 500: 1, "b" * 500: 2, "c" * 500: 3}
    first = describe_result_value(value, max_chars=1100)
    assert len(first["children"]) == 1 and first["next_offset"] == 1
    assert first["children"][0]["pointer"] == "/" + "a" * 500
    with pytest.raises(ValueError, match="raw JSON paging"):
        describe_result_value({"x" * 10000: None}, max_chars=1000)


def test_producer_description_matches_stored_json_types_and_key_conversion(run_dir):
    value = {None: (True, None), False: 1}
    saved = save_tool_result(value)
    description = describe_result_value(value)
    assert [child["pointer"] for child in description["children"]] == ["/null", "/false"]
    assert description["children"][0]["value_type"] == "array"
    assert description["children"][0]["total_children"] == 2
    assert json.loads(_read_all(saved["result_id"], "/null")) == [True, None]


def test_corrupt_saved_text_is_reported_without_claiming_a_complete_selection(run_dir):
    saved = save_tool_result({"status": True})
    path = run_dir / "tool_results" / f"{saved['result_id']}.json"
    path.write_text(json.dumps({"text": "not JSON"}))
    for request in ({"pointer": "/status"}, {"describe": True}):
        result = read_tool_result(saved["result_id"], **request)
        assert result["ok"] is False and "Cannot select" in result["error"]


# Oversized tool results are saved behind a bounded envelope.

def test_large_mutation_result_is_preserved_and_can_be_read_losslessly(run_dir):
    original = {"ok": True, "verdicts": [{"evidence": "é\\\"" * 50000}], "error": None}
    before = deepcopy(original)
    payload = json.loads(clamp_mcp_payload(original, "judge_function"))
    assert payload["ok"] is True
    assert payload["result_saved"] and len(json.dumps(payload)) < 1500
    assert not payload["full_result_complete"]
    assert {child["pointer"] for child in payload["structure"]["children"]} == {
        "/ok", "/verdicts", "/error",
    }
    # Mutating the producer cannot change the evidence behind its saved ID.
    original["verdicts"].clear()
    pages = []
    offset = 0
    while offset is not None:
        page = read_tool_result(payload["result_id"], offset=offset, limit=1000000)
        assert page["ok"] and len(page["text"]) <= PAGE_CHARS
        assert len(clamp_mcp_payload(page, "read_tool_result")) < MCP_HARD_CEILING_CHARS
        pages.append(page["text"])
        offset = page["next_offset"]
    assert json.loads("".join(pages)) == before


def test_result_near_claude_code_limit_is_saved_with_its_decisive_fields(run_dir):
    message = "ROUND 4 CLOSING REFINEMENT: recover the blocked review before scoring. " * 80
    original = {
        "ok": True, "advanced": False, "round": 3, "max_rounds": 8, "goal_met": False,
        "refinement_pending": True, "stop_kind": None, "status": "validation_blocked",
        "n_qualifying_detectors": 0,
        "verdict": {"code": "OK", "label": "OK", "reason": "Coverage above 0.5%"},
        "round_validation": {"grades": "g" * 40_000},
        "message": message,
        "next_action": {"tool": "judge_function", "arguments": {"name": "detector"}},
    }
    # Within the margin kept below Claude Code's 50,000-character file diversion.
    assert MCP_HARD_CEILING_CHARS < len(json.dumps(original)) < 50_000
    text = clamp_mcp_payload(original, "advance_round")
    envelope = json.loads(text)
    assert envelope["result_saved"] is True
    assert len(text) < MCP_SOFT_WARN_CHARS
    for key in ("ok", "advanced", "round", "max_rounds", "goal_met", "refinement_pending",
                "stop_kind", "status", "n_qualifying_detectors", "verdict", "next_action"):
        assert envelope[key] == original[key]
    assert "round_validation" not in envelope
    head = envelope["message"]
    assert message.startswith(head)
    assert ENVELOPE_HEAD_CHARS - 10 <= len(json.dumps(head)) <= ENVELOPE_HEAD_CHARS
    assert set(envelope["truncated_fields"]) == {"message"}
    rest = envelope["truncated_fields"]["message"]
    assert "pointer='/message'" in rest["read_with"]
    page = read_tool_result(envelope["result_id"], pointer="/message")
    assert page["next_offset"] is None and json.loads(page["text"]) == message
    assert rest["total_chars"] == page["total_chars"]


def test_saved_envelope_stays_bounded_for_many_fields_and_escaped_heads(run_dir):
    original = {f"n_{index}": index for index in range(5000)}
    original.update({"ok": False, "error": 'é"\n' * 20_000, "message": "λ" * 20_000,
                     "next_action": {"reason": "ü" * 20_000}})
    text = clamp_mcp_payload(original, "pareto_status")
    envelope = json.loads(text)
    assert envelope["result_saved"] is True and envelope["ok"] is False
    assert len(text) < MCP_SOFT_WARN_CHARS
    for key in ("error", "message", "next_action"):
        assert len(json.dumps(envelope[key])) <= ENVELOPE_HEAD_CHARS
        assert key in envelope["truncated_fields"]
    assert envelope["message"] == "λ" * len(envelope["message"])
    assert envelope["n_0"] == 0 and "n_4999" not in envelope
    assert read_tool_result(envelope["result_id"], pointer="/n_4999")["text"] == "4999"


def test_recovery_and_exposure_messages_cannot_bypass_envelope_ceiling(run_dir):
    long_field, small_field = "discovery_recovery", "_exposure_warning"
    original = {"ok": False, "advanced": False, "round": 4, "goal_met": False,
                long_field: {"prompt": 'Inspect the source: é"\n' * 10_000},
                small_field: {"message": "Inspect the saved evidence."}}
    rendered = clamp_mcp_payload(original, "submit_function")
    envelope = json.loads(rendered)
    assert len(rendered) < MCP_HARD_CEILING_CHARS
    assert envelope["ok"] is False and envelope["advanced"] is False
    assert envelope["round"] == 4 and envelope["goal_met"] is False
    assert envelope[small_field] == original[small_field]
    assert len(json.dumps(envelope[long_field])) <= ENVELOPE_HEAD_CHARS
    assert long_field in envelope["truncated_fields"]
    selected = _read_all(envelope["result_id"], f"/{long_field}", limit=PAGE_CHARS)
    assert json.loads(selected) == original[long_field]


def test_final_envelope_size_guard_preserves_outcomes_and_paging(run_dir, monkeypatch):
    # An accidental expansion in a preview producer must not defeat the last
    # transport boundary or force a repeat of the completed operation.
    monkeypatch.setattr("autofa.mcp.server.describe_result_value",
                        lambda *args, **kwargs: {"preview": "x" * 60_000})
    original = {"ok": True, "advanced": False, "round": 4, "result": "y" * 60_000}
    rendered = clamp_mcp_payload(original, "advance_round")
    envelope = json.loads(rendered)
    assert len(rendered) < MCP_HARD_CEILING_CHARS
    assert envelope["ok"] is True and envelope["advanced"] is False and envelope["round"] == 4
    assert envelope["result_saved"] and "read_tool_result" in envelope["read_with"]
    assert not envelope["full_result_complete"]
    assert read_tool_result(envelope["result_id"], pointer="/advanced")["text"] == "false"


def test_judge_transport_keeps_original_reviews_and_full_evidence(run_dir):
    citation = {"item_id": "task-a", "quote": "Full evidence λ " * 10_000,
                "start": 0, "complete": True}
    original = {"ok": True, "verdicts": [{
        "item_id": "task-a", "citations": [citation],
        "claim_assessments": [{"citations": [deepcopy(citation)]}],
    }]}
    before = deepcopy(original)
    envelope = json.loads(clamp_mcp_payload(original, "judge_function"))
    assert original == before
    compact_id = envelope["result_id"]
    reference = json.loads(read_tool_result(
        compact_id, pointer="/verdicts/0/claim_assessments/0/citations/0",
    )["text"])
    assert reference == {"$ref": "#/verdicts/0/citations/0"}
    full_result = json.loads(read_tool_result(compact_id, pointer="/full_result")["text"])
    assert full_result["result_id"] != compact_id
    # Both the canonical citation and full unmodified response remain readable.
    selected = _read_all(compact_id, reference["$ref"][1:], limit=PAGE_CHARS)
    assert json.loads(selected) == citation
    assert json.loads(_read_all(full_result["result_id"], limit=PAGE_CHARS)) == before


def test_default_field_chunk_window_of_trace_json_arrives_inline():
    # Escaping and response metadata must still fit below the ceiling.
    default = inspect.signature(
        server.TOOL_CALLABLES["read_field_chunk"]).parameters["limit"].default
    trace = json.dumps([{"role": "assistant", "content": 'Ran "pytest -q".\nAll passed.'}] * 5000)
    window = trace[:default]
    payload = json.loads(clamp_mcp_payload({
        "ok": True, "text": window, "offset": 0, "returned_chars": len(window),
        "total_chars": len(trace), "next_offset": len(window), "is_final": False,
    }, "read_field_chunk"))
    assert "result_saved" not in payload and payload["text"] == window
