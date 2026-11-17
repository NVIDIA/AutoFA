# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codex harness lifecycle: dynamic tools, recovery, capacity model fallback and conversation."""

import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.harness import codex, codex_app_server, model_catalog
from autofa.harness.model_catalog import HarnessModel, ModelCatalog
from autofa.harness.model_fallback import effort_for_model, lower_codex_models
from autofa.mcp import runtime, server
from autofa.metrics import claude_judge
from autofa.session.recovery import RunInterruptedError
from autofa.session.status_log import latest_turn_status
from autofa.ui.activity import AgentActivity
from autofa.util.cancellation import Cancellation, cancellation_scope


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(AgentActivity, "start", lambda self: None)
    monkeypatch.setattr(AgentActivity, "stop", lambda self, **kwargs: None)
    h = codex.CodexHarness.__new__(codex.CodexHarness)
    h.model, h.cli_path, h.effort = "gpt-6-luna", "/test/codex", "high"
    return h


@pytest.fixture
def sleeps(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(codex.asyncio, "sleep", sleep)
    return delays


def answer(**kwargs):
    return SimpleNamespace(**{
        "thread_id": "thread-1", "turn_id": "turn-1", "status": "completed",
        "text": "Finished", "error": None, "error_code": None, "retryable": False, **kwargs,
    })


def overloaded(**overrides):
    return answer(**{
        "thread_id": "retained-thread", "status": "failed",
        "error": "Selected model is at capacity", "error_code": "serverOverloaded",
        "retryable": True, **overrides,
    })


def model(value, **kwargs):
    return HarnessModel(value=value, **({
        "display_name": value, "supports_effort": True,
        "effort_levels": ("low", "medium", "high"), "default_effort": "medium",
    } | kwargs))


def fake_transport(monkeypatch, fn):
    monkeypatch.setattr(codex_app_server, "run_codex_turn", fn)


def test_constructor_uses_codex_auth_and_effort_only(monkeypatch):
    monkeypatch.setattr(codex, "_find_codex_cli", lambda path: path or "/bin/codex")
    monkeypatch.setattr("autofa.harness.claude.ensure_claude_auth", lambda **kw: pytest.fail())
    h = codex.CodexHarness("any-model", effort="xhigh")
    assert h.model == "any-model" and h.cli_path == "/bin/codex" and h.effort == "xhigh"
    assert codex.CodexHarness("any-model").effort is None
    with pytest.raises(ValueError, match="Effort"):
        codex.CodexHarness("any-model", thinking="adaptive")


def test_turn_persists_thread_before_tools_and_runs_same_runtime(tmp_path, monkeypatch, harness):
    invoked = []

    async def invoke(name, args, **kwargs):
        invoked.append((name, args))
        return {"content": [{"type": "text", "text": '{"ok":true,"n":5}'}]}

    monkeypatch.setattr(codex, "invoke_tool", invoke)

    async def run(**kwargs):
        assert kwargs["model"] == "gpt-6-luna"
        assert kwargs["effort"] == "high" and kwargs["cli_path"] == "/test/codex"
        assert kwargs["system_prompt"] == "system instructions"
        assert latest_turn_status(tmp_path)["status"] == "started"
        await kwargs["on_thread"]("thread-1")
        assert json.loads((tmp_path / "session.json").read_text()) == {
            "session_id": "thread-1", "harness": "codex",
        }
        assert [tool["name"] for tool in kwargs["tools"]] == ["dataset_info"]
        result = await kwargs["call_tool"]("dataset_info", {})
        assert result["success"] is True
        assert result["contentItems"][0]["type"] == "inputText"
        denied = await kwargs["call_tool"]("clear_library", {"confirm": True})
        assert denied["success"] is False
        return answer()

    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn(
        "/discover", cwd=str(tmp_path), system_prompt="system instructions",
        mcp_servers={"autofa": True}, allowed_tools=["mcp__autofa__dataset_info"],
        show_result=False,
    ))
    assert invoked == [("dataset_info", {})]
    assert result.text == "Finished" and result.error is None
    assert latest_turn_status(tmp_path)["status"] == "finished"
    assert json.loads((tmp_path / "turn.json").read_text())["state"] == "finished"


def test_without_autofa_server_no_tools_are_exposed(tmp_path, monkeypatch, harness):
    async def run(**kwargs):
        assert kwargs["tools"] == []
        result = await kwargs["call_tool"]("dataset_info", {})
        assert result["success"] is False
        return answer()

    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn(
        "hello", cwd=str(tmp_path), system_prompt="test", show_result=False,
    ))
    assert result.error is None


def test_retry_resumes_thread_and_preserves_original_work(tmp_path, monkeypatch, harness, sleeps):
    calls = []
    library = tmp_path / "library.json"

    async def run(**kwargs):
        calls.append(kwargs)
        await kwargs["on_thread"]("thread-1")
        if len(calls) == 1:
            library.write_text('{"functions":["saved"]}')
            return answer(status="failed", error="Connection closed mid-response", retryable=True)
        assert library.read_text() == '{"functions":["saved"]}'
        assert kwargs["resume"] == "thread-1"
        assert "do not repeat completed work" in kwargs["prompt"]
        assert "Original request for this turn:\nDiscover 8 rounds" in kwargs["prompt"]
        return answer(text="Recovered")

    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn(
        "Discover 8 rounds", cwd=str(tmp_path), system_prompt="system", show_result=False,
    ))
    assert result.text == "Recovered" and len(calls) == 2 and sleeps == [5]


def test_permanent_error_never_retries(tmp_path, monkeypatch, harness):
    calls = []

    async def run(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("HTTP 403 forbidden")

    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn("Discover", cwd=str(tmp_path), system_prompt="test"))
    assert len(calls) == 1 and result.error == "HTTP 403 forbidden"


def test_model_denial_keeps_saved_thread_and_never_retries(tmp_path, monkeypatch, harness):
    from autofa.harness.codex_app_server import CodexTurnResult, _error_details

    calls = []
    code, message, retryable = _error_details({
        "codexErrorInfo": "other", "message": "You do not have access to this model",
    }, model=harness.model)

    async def run(**kwargs):
        calls.append(kwargs)
        await kwargs["on_thread"]("saved-thread")
        return CodexTurnResult(
            thread_id="saved-thread", error=message, error_code=code, retryable=retryable,
        )

    async def sleep(_):
        pytest.fail("Model access failures must not be retried")

    fake_transport(monkeypatch, run)
    monkeypatch.setattr(codex.asyncio, "sleep", sleep)
    result = asyncio.run(harness.run_turn("Discover", cwd=str(tmp_path), system_prompt="test"))
    assert len(calls) == 1
    assert result.error == message and not result.retryable
    assert result.session_id == "saved-thread"
    assert latest_turn_status(tmp_path)["status"] == "failed"
    assert json.loads((tmp_path / "session.json").read_text())["session_id"] == "saved-thread"


def test_cancellation_keeps_thread_and_partial_text(tmp_path, monkeypatch, harness):
    async def run(**kwargs):
        await kwargs["on_thread"]("saved-thread")
        await kwargs["on_event"]("item/agentMessage/delta", {"itemId": "msg", "delta": "Partial"})
        raise asyncio.CancelledError

    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn("Discover", cwd=str(tmp_path), system_prompt="test"))
    assert result.stopped is True and result.text == "Partial"
    assert result.session_id == "saved-thread"
    assert latest_turn_status(tmp_path)["status"] == "interrupted"
    assert json.loads((tmp_path / "turn.json").read_text())["state"] == "interrupted"


def test_unconfirmed_completion_is_error(tmp_path, monkeypatch, harness):
    async def run(**kwargs):
        return answer(status="inProgress")

    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn("Discover", cwd=str(tmp_path), system_prompt="test"))
    assert result.error == "Codex turn ended without confirmed normal completion"


def test_context_limit_compacts_and_retries_same_thread_without_reset(
    tmp_path, monkeypatch, harness,
):
    calls, compactions = [], []
    library = tmp_path / "library.json"
    library.write_text('{"functions":["saved"]}')

    async def compact(**kwargs):
        compactions.append(kwargs)
        return answer(thread_id="retained")

    async def run(**kwargs):
        calls.append(kwargs)
        await kwargs["on_thread"]("retained")
        if len(calls) == 1:
            return answer(thread_id="retained", status="failed", error="Codex context is full")
        assert kwargs["resume"] == "retained"
        assert "do not repeat completed" in kwargs["prompt"]
        original_request = "Original request for this turn:\nContinue within the original scope"
        assert original_request in kwargs["prompt"]
        assert library.read_text() == '{"functions":["saved"]}'
        return answer(thread_id="retained")

    monkeypatch.setattr(codex_app_server, "compact_codex_thread", compact)
    monkeypatch.setattr("autofa.session.recovery.context_recovery_prompt",
                        lambda *args, **kwargs: pytest.fail("Must not replace the conversation"))
    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn(
        "Continue within the original scope", cwd=str(tmp_path), system_prompt="test",
    ))
    assert result.error is None and result.session_id == "retained"
    assert compactions == [{"thread_id": "retained", "cwd": str(tmp_path),
                            "cli_path": harness.cli_path}]
    assert len(calls) == 2
    assert not (tmp_path / "context_recovery").exists()
    assert json.loads((tmp_path / "turn.json").read_text())["prompt"] == (
        "Continue within the original scope")


@pytest.mark.parametrize("outcome", ["failed", "inProgress", "interrupted", "cancel"])
def test_failed_compaction_preserves_intent_and_recovers_unless_stopped(
    tmp_path, monkeypatch, harness, outcome,
):
    calls = []

    async def compact(**kwargs):
        if outcome == "cancel":
            raise asyncio.CancelledError
        return answer(thread_id="retained", status=outcome,
                      error="service unavailable" if outcome == "failed" else None)

    async def run(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            assert kwargs["resume"] is None
            assert "Continue" in kwargs["prompt"] and "operations/" in kwargs["prompt"]
            await kwargs["on_thread"]("replacement")
            return answer(thread_id="replacement")
        await kwargs["on_thread"]("retained")
        return answer(thread_id="retained", status="failed", error="Codex context is full")

    monkeypatch.setattr(codex_app_server, "compact_codex_thread", compact)
    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn("Continue", cwd=str(tmp_path), system_prompt="test"))
    turn = json.loads((tmp_path / "turn.json").read_text())
    assert turn["prompt"] == "Continue"
    if outcome in {"cancel", "interrupted"}:
        assert len(calls) == 1 and result.session_id == "retained"
        assert result.stopped and turn["state"] == "interrupted"
        assert latest_turn_status(tmp_path)["status"] == "interrupted"
        assert not (tmp_path / "context_recovery").exists()
    else:
        assert len(calls) == 2 and result.session_id == "replacement"
        assert not result.error and turn["state"] == "finished"
        assert (tmp_path / "context_recovery").exists()
    assert json.loads((tmp_path / "session.json").read_text())["session_id"] == result.session_id


def test_compaction_does_not_loop_if_context_is_still_full(tmp_path, monkeypatch, harness):
    calls, compactions = [], []

    async def compact(**kwargs):
        compactions.append(kwargs)
        return answer(thread_id="retained")

    async def run(**kwargs):
        calls.append(kwargs)
        await kwargs["on_thread"]("retained")
        return answer(thread_id="retained", status="failed", error="Codex context is full")

    monkeypatch.setattr(codex_app_server, "compact_codex_thread", compact)
    fake_transport(monkeypatch, run)
    result = asyncio.run(harness.run_turn("Continue", cwd=str(tmp_path), system_prompt="test"))
    assert result.session_id == "retained" and "context is full" in result.error
    assert len(calls) == 3 and len(compactions) == 1
    assert calls[-1]["resume"] is None
    saved = json.loads((tmp_path / "turn.json").read_text())
    assert saved["state"] == "interrupted" and saved["resumable"]
    assert saved["recovery_reason"] == "context_recovery_exhausted"


def test_explicit_compact_preserves_interrupted_turn_and_uses_native_api(
    tmp_path, monkeypatch, harness,
):
    turn = tmp_path / "turn.json"
    original = '{"prompt":"Preserve this interrupted request", "state":"failed"}'
    turn.write_text(original)
    calls = []

    async def compact(**kwargs):
        calls.append(kwargs)
        return answer(thread_id="retained")

    async def forbidden(**kwargs):
        pytest.fail("/compact must not be submitted as an ordinary model turn")

    monkeypatch.setattr(codex_app_server, "compact_codex_thread", compact)
    fake_transport(monkeypatch, forbidden)
    result = asyncio.run(harness.run_turn(
        "/compact", cwd=str(tmp_path), system_prompt="test", resume="retained", show_result=False,
    ))
    assert result.error is None and result.session_id == "retained"
    assert turn.read_text() == original
    assert calls == [{"thread_id": "retained", "cwd": str(tmp_path), "cli_path": harness.cli_path}]


@pytest.mark.parametrize("resume,command", [(None, "/compact"), ("retained", "/compact focus")])
def test_invalid_compact_request_never_starts_inference(
    tmp_path, monkeypatch, harness, resume, command,
):
    async def forbidden(**kwargs):
        pytest.fail("Compaction must not start another conversation")

    fake_transport(monkeypatch, forbidden)
    result = asyncio.run(harness.run_turn(
        command, cwd=str(tmp_path), system_prompt="test", resume=resume,
    ))
    assert result.error and result.session_id == resume
    assert not (tmp_path / "turn.json").exists()


# Dynamic tools share AutoFA's live runtime, receipts, durable operations and library.


def live_run(monkeypatch, tmp_path, run_id="saved-run", **extra):
    live = SimpleNamespace(run=SimpleNamespace(run_id=run_id, run_dir=lambda: tmp_path), **extra)
    monkeypatch.setattr(runtime, "_RUNTIME", live)
    return live


def operation(tmp_path):
    return json.loads(next((tmp_path / "operations").glob("*.json")).read_text())


def test_native_tool_schemas_have_real_parameters_and_defaults():
    specs = {spec["name"]: spec for spec in server.codex_tool_specs()}
    assert len(specs) == len(server.TOOL_CALLABLES)
    spec = specs["submit_function"]
    assert spec["type"] == "function"
    schema = spec["inputSchema"]
    assert schema["required"] == ["name", "source"]
    assert schema["properties"]["source"]["type"] == "string"
    assert schema["properties"]["auto_evaluate"] == {"type": "boolean", "default": True}
    assert schema["properties"]["depends_on"]["anyOf"][0] == {
        "type": "array", "items": {"type": "string"},
    }
    assert schema["additionalProperties"] is False
    assert specs["dataset_info"]["inputSchema"]["properties"] == {}
    assert not any(name.startswith("mcp__") for name in specs)
    json.dumps(list(specs.values()))


def test_allowed_tools_are_exact_and_empty_is_not_all():
    assert server.codex_tool_specs([]) == []
    assert server.codex_tool_specs(["submit_function", "Bash", "mcp__other__dataset_info"]) == []
    specs = server.codex_tool_specs(["mcp__autofa__dataset_info", "Read"])
    assert [spec["name"] for spec in specs] == ["dataset_info"]


def test_shared_tool_executes_live_runtime_and_checkpoints(tmp_path, monkeypatch):
    live = live_run(monkeypatch, tmp_path, saved=[])

    def submit_function(name: str):
        current = runtime.get_runtime()
        current.saved.append(name)
        operations = list((tmp_path / "operations").glob("*.json"))
        assert json.loads(operations[0].read_text())["state"] == "running"
        return {"ok": True, "saved": name}

    monkeypatch.setitem(server.TOOL_CALLABLES, "submit_function", submit_function)
    response = asyncio.run(server.invoke_tool("submit_function", {"name": "semantic_mode"}))
    payload = json.loads(response["content"][0]["text"])
    assert payload == {"ok": True, "saved": "semantic_mode"}
    assert live.saved == ["semantic_mode"]
    saved = operation(tmp_path)
    assert saved["state"] == "complete"
    assert saved["args"] == {"name": "semantic_mode"}


def test_shared_errors_are_returned_as_failed_codex_calls(monkeypatch):
    monkeypatch.setitem(server.TOOL_CALLABLES, "test", lambda required: {"ok": True})
    for name, args in [("unknown", {}), ("test", {}), ("test", [])]:
        response = asyncio.run(server.invoke_tool(name, args))
        converted = server.codex_tool_response(response)
        assert converted["success"] is False
        assert json.loads(converted["contentItems"][0]["text"])["ok"] is False


def test_interrupted_operation_retains_resume_command(tmp_path, monkeypatch):
    live_run(monkeypatch, tmp_path)

    def submit_function():
        raise RunInterruptedError("service unavailable")

    monkeypatch.setitem(server.TOOL_CALLABLES, "submit_function", submit_function)
    result = server.codex_tool_response(asyncio.run(server.invoke_tool("submit_function", {})))
    payload = json.loads(result["contentItems"][0]["text"])
    assert result["success"] is False
    assert payload["resumable"] is True
    assert payload["resume_command"] == "autofa resume --run saved-run"
    saved = operation(tmp_path)
    assert saved["state"] == "interrupted"


def test_verified_image_bytes_reach_codex_outside_text_budget():
    response = server.tool_response({
        "ok": True, "artifact": {"locator": "page:2"},
        "_artifact_image": {"status": "available", "data_url": "data:image/png;base64,eA=="},
    }, "view_artifact")
    result = server.codex_tool_response(response)
    assert result["success"] is True
    assert result["contentItems"][1] == {
        "type": "inputImage", "imageUrl": "data:image/png;base64,eA==",
    }
    assert "base64" not in result["contentItems"][0]["text"]
    assert "_artifact_image" not in result["contentItems"][0]["text"]


def test_oversized_result_can_be_paged_without_repeating_operation(tmp_path, monkeypatch):
    live_run(monkeypatch, tmp_path)
    monkeypatch.setitem(server.TOOL_CALLABLES, "test", lambda: {
        "ok": True, "evidence": "x" * server.MCP_HARD_CEILING_CHARS,
    })
    result = server.codex_tool_response(asyncio.run(server.invoke_tool("test", {})))
    payload = json.loads(result["contentItems"][0]["text"])
    assert payload["result_saved"] is True
    assert payload["read_with"].startswith("read_tool_result(")
    assert result["success"] is True


def test_cancel_waits_for_active_worker_before_releasing_runtime(tmp_path, monkeypatch):
    live_run(monkeypatch, tmp_path)
    started, finish = threading.Event(), threading.Event()

    def submit_function():
        started.set()
        assert finish.wait(timeout=5)
        return {"ok": True, "saved": True}

    monkeypatch.setitem(server.TOOL_CALLABLES, "submit_function", submit_function)

    async def scenario():
        task = asyncio.create_task(server.invoke_tool("submit_function", {}))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        saved = operation(tmp_path)
        assert saved["state"] == "complete"
        assert saved["result"] == {"ok": True, "saved": True}

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["regex", "llm"])
def test_codex_tools_submit_evaluate_judge_save_and_resume_live_library(
    tmp_path, monkeypatch, kind,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "autofa"))
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(AgentActivity, "start", lambda self: None)
    monkeypatch.setattr(AgentActivity, "stop", lambda self, **kwargs: None)
    rows = [{
        "model_id": "evaluated-model", "benchmark": "suite", "problem_id": str(index),
        "run_idx": 0, "is_correct": index % 2 == 1,
        "response_text": ("All tests pass, but the test output reports FAILED."
                          if index % 2 == 0 else "The tests passed after fixing the error."),
    } for index in range(20)]
    parquet = tmp_path / "traces.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    cfg = Config(harness="codex", model="gpt-6-luna", behavior_judge_model="gpt-6-luna",
                 library_root=str(tmp_path / "catalog"), llm_eval_workers=1)
    run = RunConfig(run_id=f"codex-{kind}", parquet_paths=[str(parquet)],
                    model_id="evaluated-model", benchmarks=["suite"])
    classifier_calls = []

    class Classifier:
        available = True

        def classify(self, prompt, text, **kwargs):
            classifier_calls.append(text)
            return "yes" if "FAILED" in text else "no"

    live = runtime.Runtime(dataset=Dataset(parquet_paths=[parquet]),
                           library=FunctionLibrary(), run=run, config=cfg,
                           hub_llm=Classifier(), llm_readiness={"available": True})
    monkeypatch.setattr(runtime, "_RUNTIME", live)
    judge_calls = []

    def judge(**kwargs):
        judge_calls.append(kwargs)
        assert kwargs["harness"] == "codex"
        assert kwargs["model"] == "gpt-6-luna"
        assert len(kwargs["items"]) == 10
        assert all("FAILED" in item["response_text"] for item in kwargs["items"])
        return {"behavior_purity": 1.0, "behavior_specificity": .8,
                "n_judged": 10, "n_completed": 10, "n_yes": 10, "n_no": 0,
                "n_uncertain": 0, "n_errors": 0, "verdicts": [],
                "backend": "codex_app_server", "judge_model": "gpt-6-luna"}

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", judge)
    if kind == "regex":
        source = 'def detect(item):\n    return "FAILED" in item.get("response_text", "")\n'
    else:
        source = (
            'def detect(item, llm_client):\n'
            '    return llm_client.classify("Does the response claim tests pass despite an '
            'explicit FAILED observation? Reply yes or no.", '
            'item.get("response_text", "")) == "yes"\n'
        )
    turns = []

    async def transport(**kwargs):
        turns.append(kwargs)
        await kwargs["on_thread"]("codex-thread")

        async def call(name, args):
            response = await kwargs["call_tool"](name, args)
            payload = json.loads(response["contentItems"][0]["text"])
            assert response["success"] is True, payload
            return payload

        if len(turns) == 1:
            info = await call("dataset_info", {})
            assert info["estimated_corpus_rows"] == 20
            submitted = await call("submit_function", {
                "name": "contradicted_success", "source": source, "kind": kind,
                "description": "Claims tests passed despite a failed test observation",
                "mechanism": "The response claims success despite an explicit failed test. "
                             "The contradictory observation remains unresolved in its answer.",
                "auto_evaluate": False,
            })
            assert submitted["ok"]
            assert live.library.get("contradicted_success") is not None
            await call("evaluate_functions", {"names": ["contradicted_success"], "mode": "full"})
            scores = live.library.get("contradicted_success").scores
            assert scores["n_scored"] == 20 and scores["a"] == 10 and scores["d"] == 10
            await call("judge_function", {"name": "contradicted_success"})
            await call("library_save", {"to_catalog": False})
        else:
            assert kwargs["resume"] == "codex-thread"
            detail = await call("get_library_function", {"name": "contradicted_success"})
            assert detail["function"]["scores"]["judge_backend"] == "codex_app_server"
        return SimpleNamespace(thread_id="codex-thread", status="completed", text="Done",
                               error=None, retryable=False)

    monkeypatch.setattr(codex_app_server, "run_codex_turn", transport)
    harness = codex.CodexHarness.__new__(codex.CodexHarness)
    harness.model, harness.cli_path, harness.effort = "gpt-6-luna", "/test/codex", "medium"

    async def scenario():
        first = await harness.run_turn(
            "/discover", cwd=str(run.run_dir()), system_prompt="Discover and validate failures.",
            mcp_servers={"autofa": True},
            allowed_tools=[f"mcp__autofa__{name}" for name in server.tool_names()],
            show_result=False,
        )
        assert first.error is None
        second = await harness.run_turn(
            "Inspect the saved detector", cwd=str(run.run_dir()), system_prompt="Inspect failures.",
            mcp_servers={"autofa": True}, allowed_tools=["mcp__autofa__get_library_function"],
            resume=first.session_id, show_result=False,
        )
        assert second.error is None

    asyncio.run(scenario())
    saved = FunctionLibrary.load(run.run_dir() / "library.json")
    assert saved.get("contradicted_success").scores["judge_backend"] == "codex_app_server"
    assert len(judge_calls) == 1
    assert bool(classifier_calls) is (kind == "llm")
    operations = [json.loads(path.read_text()) for path in (run.run_dir() / "operations").glob("*")]
    assert {op["tool"] for op in operations} >= {
        "submit_function", "evaluate_functions", "judge_function", "library_save",
    }
    assert all(op["state"] == "complete" for op in operations)
    assert json.loads((run.run_dir() / "session.json").read_text()) == {
        "session_id": "codex-thread", "harness": "codex",
    }


# Capacity fallback policy (pure catalog decisions).


def policy_catalog():
    # Matches installed model/list order, deliberately different from fallback order.
    return ModelCatalog(models=tuple(map(model, (
        "gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol",
        "gpt-daybreak-blue-latest",
    ))))


def lower(source, current):
    return tuple(m.value for m in lower_codex_models(source, current))


def test_known_chain_uses_explicit_policy_of_available_known_models_and_never_escalates():
    expected = ("gpt-6.1-sol", "gpt-6-sol", "gpt-5.6-sol", "gpt-6-luna")
    source = policy_catalog()
    assert lower(source, "gpt-6-astra") == expected
    reversed_catalog = replace(source, models=tuple(reversed(source.models)))
    assert lower(reversed_catalog, "gpt-6-astra") == expected
    assert lower(source, "gpt-6-sol") == ("gpt-5.6-sol", "gpt-6-luna")
    assert lower(source, "gpt-6-luna") == ()
    # Candidates must be advertised and known; a reassuring display name is not enough.
    advertised = ModelCatalog(models=(
        model("gpt-6-astra"), model("gpt-6-sol"), model("gpt-6-luna-preview"),
        model("gpt-6-sol-unknown", display_name="Small safe fallback"),
    ))
    assert lower(advertised, "gpt-6-astra") == ("gpt-6-sol",)


def catalog_of(*models):
    return ModelCatalog(models=models)


@pytest.mark.parametrize("source,current", [
    pytest.param(catalog_of(*policy_catalog().models, model("gpt-6-astra-preview")),
                 "gpt-6-astra-preview", id="unknown_variant"),
    pytest.param(catalog_of(*policy_catalog().models, model("gpt-6-astra-extra/gpt-99-sol")),
                 "gpt-6-astra-extra/gpt-99-sol", id="specialist"),
    pytest.param(ModelCatalog(), "gpt-6-astra", id="empty_catalog"),
    pytest.param(replace(policy_catalog(), error="unavailable"), "gpt-6-astra",
                 id="failed_catalog"),
    pytest.param(catalog_of(model("gpt-6-sol")), "gpt-6-astra", id="current_not_advertised"),
    # A known name resolving to an unknown concrete model does not inherit its quality.
    pytest.param(catalog_of(model("gpt-6-astra", resolved_model="experimental-mystery"),
                            model("gpt-6-sol")), "gpt-6-astra", id="known_name_unknown_model"),
])
def test_unknown_model_quality_or_missing_catalog_produces_no_guesses(source, current):
    assert lower(source, current) == ()


@pytest.mark.parametrize("source,current,expected", [
    pytest.param(catalog_of(*map(model, (
        "gateway/gpt-6-astra", "other/gpt-6-sol", "gpt-6.1-sol", "gateway/gpt-6-sol",
        "gateway/gpt-6-luna", "gateway/more/gpt-6.1-sol", "gateway:gpt-6.1-sol",
    ))), "gateway/gpt-6-astra", ("gateway/gpt-6-sol", "gateway/gpt-6-luna"),
        id="namespace_kept_and_qualifiers_not_stripped"),
    pytest.param(catalog_of(
        model("default", resolved_model="gpt-6-astra"),
        model("fast", resolved_model="gpt-6-sol"), model("gpt-6-sol"),
        model("small", resolved_model="gpt-6-luna"),
    ), "default", ("gpt-6-sol", "small"), id="aliases_resolve_and_deduplicate"),
    pytest.param(catalog_of(
        model("gateway/default", resolved_model="gpt-6-astra"),
        model("gpt-6.1-sol"), model("other/fast", resolved_model="gpt-6-sol"),
        model("gateway/fast", resolved_model="gpt-6-sol"),
        model("gateway/ambiguous", resolved_model="other/gpt-6.1-sol"),
    ), "gateway/default", ("gateway/fast",), id="alias_cannot_switch_provider"),
])
def test_namespaces_and_aliases_never_switch_provider(source, current, expected):
    assert lower(source, current) == expected


def test_unknown_or_disabled_effort_support_does_not_guess():
    assert effort_for_model(model("gpt-6-sol", supports_effort=False), "high") is None
    assert effort_for_model(model("gpt-6-sol", effort_levels=()), "high") is None
    assert effort_for_model(model("gpt-6-sol", default_effort="unsupported"), "default") is None


# Offline capacity fallback keeps the conversation, saved work and requested settings.


@pytest.fixture
def scenario(tmp_path, monkeypatch, harness, sleeps):
    monkeypatch.setattr("autofa.config._write_config_atomically",
                        lambda *_: pytest.fail("No global config edit"))
    harness.model, harness.effort, harness.cli_path = "gpt-6-astra", "xhigh", "/fixture/codex"
    state = SimpleNamespace(
        harness=harness, directory=tmp_path, calls=[], sleeps=sleeps, catalog_calls=[], answers=[],
        catalog=ModelCatalog(models=(
            model("gpt-6-astra", effort_levels=("high", "xhigh"), default_effort="high"),
            model("gpt-6.1-sol"), model("gpt-6-sol"), model("gpt-5.6-sol"),
            model("gpt-6-luna"),
        )),
    )
    # A real work/checkpoint file must survive retry and switching byte for byte.
    state.library = tmp_path / "library.json"
    state.library.write_text('{"functions":[{"name":"already_saved"}]}')
    state.initial_library = state.library.read_bytes()
    state.checkpoint = tmp_path / "review_checkpoint.json"
    state.checkpoint.write_text('{"completed_review":"retain me"}')

    async def transport(**kwargs):
        state.calls.append(kwargs)
        assert state.library.read_bytes() == state.initial_library
        assert state.checkpoint.read_text() == '{"completed_review":"retain me"}'
        await kwargs["on_thread"]("retained-thread")
        result = state.answers.pop(0) if state.answers else answer(thread_id="retained-thread")
        if isinstance(result, BaseException):
            raise result
        return result

    async def catalog(cli_path, **kwargs):
        state.catalog_calls.append((cli_path, kwargs))
        return state.catalog

    fake_transport(monkeypatch, transport)
    monkeypatch.setattr(model_catalog, "load_model_catalog_async", catalog)
    return state


def run(state, *, resume=None):
    return asyncio.run(state.harness.run_turn(
        "Discover within this round", cwd=str(state.directory), system_prompt="Original policy",
        resume=resume, show_result=False,
    ))


def new_harness(state):
    harness = codex.CodexHarness.__new__(codex.CodexHarness)
    harness.model, harness.effort, harness.cli_path = (
        state.harness.model, state.harness.effort, state.harness.cli_path,
    )
    state.harness = harness
    return harness


def fallback_once(state):
    state.answers = [overloaded(), overloaded(), answer(thread_id="retained-thread")]
    result = run(state)
    assert result.error is None
    return json.loads((state.directory / "codex_model.json").read_text())


def assert_interrupted(directory):
    assert latest_turn_status(directory)["status"] == "interrupted"
    assert json.loads((directory / "turn.json").read_text())["state"] == "interrupted"


@pytest.mark.parametrize("effort,expected_effort", [("xhigh", "medium"), ("high", "high")])
def test_capacity_retries_once_then_switches_preserving_thread_and_saved_work(
    scenario, effort, expected_effort,
):
    scenario.harness.effort = effort
    saved = fallback_once(scenario)
    assert [call["model"] for call in scenario.calls] == [
        "gpt-6-astra", "gpt-6-astra", "gpt-6.1-sol",
    ]
    assert scenario.sleeps == [5]
    assert scenario.calls[0]["resume"] is None
    assert all(call["resume"] == "retained-thread" for call in scenario.calls[1:])
    assert all(call["system_prompt"] == "Original policy" for call in scenario.calls)
    for call in scenario.calls[1:]:
        assert "do not repeat completed work" in call["prompt"]
        assert "Original request for this turn:\nDiscover within this round" in call["prompt"]
    assert scenario.calls[-1]["effort"] == expected_effort
    assert scenario.harness.model == "gpt-6-astra" and scenario.harness.effort == effort
    assert (saved["requested_model"], saved["requested_effort"]) == ("gpt-6-astra", effort)
    assert (saved["active_model"], saved["active_effort"]) == ("gpt-6.1-sol", expected_effort)
    assert saved["session_id"] == "retained-thread"
    # The first overload is an ordinary retry; only the second one consults the catalog.
    assert scenario.catalog_calls == [("/fixture/codex", {"harness": "codex"})]
    assert latest_turn_status(scenario.directory)["status"] == "finished"


def test_every_model_overloaded_is_bounded_to_two_switches(scenario):
    scenario.answers = [overloaded() for _ in range(20)]
    result = run(scenario)
    assert result.error_code == "serverOverloaded" and result.error
    assert [call["model"] for call in scenario.calls] == [
        "gpt-6-astra", "gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-sol",
        "gpt-6-sol", "gpt-6-sol", "gpt-6-sol",
    ]
    assert len(scenario.catalog_calls) == 2
    assert all(call["resume"] == "retained-thread" for call in scenario.calls[1:])
    assert_interrupted(scenario.directory)


def test_unavailable_catalog_retains_primary_and_existing_retry_bound(scenario):
    scenario.catalog = ModelCatalog(error="unavailable")
    scenario.answers = [overloaded() for _ in range(3)]
    result = run(scenario)
    assert result.error_code == "serverOverloaded"
    assert [call["model"] for call in scenario.calls] == ["gpt-6-astra"] * 3
    assert scenario.sleeps == [5, 10]
    assert not (scenario.directory / "codex_model.json").exists()


def test_catalog_transport_failure_does_not_replace_capacity_error(scenario, monkeypatch):
    async def unavailable(*_args, **_kwargs):
        raise OSError("catalog process unavailable")

    monkeypatch.setattr(model_catalog, "load_model_catalog_async", unavailable)
    scenario.answers = [overloaded() for _ in range(3)]
    result = run(scenario)
    assert result.error == "Selected model is at capacity"
    assert len(scenario.calls) == 3
    assert not (scenario.directory / "codex_model.json").exists()


@pytest.mark.parametrize("code,message", [
    (None, "serverOverloaded: model is at capacity"),
    ("rateLimitExceeded", "Too many requests"),
])
def test_other_retryable_errors_are_bounded_and_never_change_model(scenario, code, message):
    scenario.answers = [overloaded(error=message, error_code=code) for _ in range(3)]
    result = run(scenario)
    assert result.error == message and len(scenario.calls) == 3
    assert scenario.catalog_calls == []
    assert all(call["model"] == "gpt-6-astra" for call in scenario.calls)
    assert not (scenario.directory / "codex_model.json").exists()
    assert latest_turn_status(scenario.directory)["status"] == "interrupted"


def test_nonretryable_capacity_error_does_not_switch_or_retry(scenario):
    scenario.answers = [overloaded(retryable=False)]
    result = run(scenario)
    assert result.error_code == "serverOverloaded" and not result.retryable
    assert len(scenario.calls) == 1
    assert not scenario.sleeps and not scenario.catalog_calls


@pytest.mark.parametrize("when", ["backoff", "catalog", "catalog_token"])
def test_cancellation_never_starts_another_attempt_or_checkpoints_a_switch(
    scenario, monkeypatch, when,
):
    scenario.answers = [overloaded() for _ in range(4)]
    token = Cancellation()

    async def cancelled_sleep(_delay):
        raise asyncio.CancelledError

    async def cancelled_catalog(*_args, **_kwargs):
        if when == "catalog_token":
            token.cancel()
            return scenario.catalog
        raise asyncio.CancelledError

    if when == "backoff":
        monkeypatch.setattr(codex.asyncio, "sleep", cancelled_sleep)
    else:
        monkeypatch.setattr(model_catalog, "load_model_catalog_async", cancelled_catalog)
    with cancellation_scope(token):
        result = run(scenario)
    assert result.stopped is True
    assert len(scenario.calls) == (1 if when == "backoff" else 2)
    assert result.session_id == "retained-thread"
    assert not (scenario.directory / "codex_model.json").exists()
    assert_interrupted(scenario.directory)


def test_resumed_run_restores_fallback_effort_without_trying_primary_again(scenario):
    fallback_once(scenario)
    new_harness(scenario)
    before = len(scenario.calls)
    result = run(scenario, resume="retained-thread")
    assert result.error is None
    assert len(scenario.calls) == before + 1
    assert scenario.calls[-1]["model"] == "gpt-6.1-sol"
    assert scenario.calls[-1]["effort"] == "medium"
    assert scenario.harness.model == "gpt-6-astra" and scenario.harness.effort == "xhigh"
    assert len(scenario.catalog_calls) == 1


def test_alias_fallback_preserves_selector_and_resolved_identity_on_resume(scenario):
    scenario.harness.model = "gateway/frontier"
    scenario.catalog = ModelCatalog(models=(
        model("gateway/frontier", resolved_model="gpt-6-astra"),
        model("gateway/standard", resolved_model="gpt-6.1-sol"),
        model("other/gpt-6-sol"),
    ))
    saved = fallback_once(scenario)
    assert scenario.calls[-1]["model"] == "gateway/standard"
    assert saved["requested_resolved_model"] == "gpt-6-astra"
    assert saved["active_resolved_model"] == "gpt-6.1-sol"
    new_harness(scenario)
    assert run(scenario, resume="retained-thread").error is None
    assert scenario.calls[-1]["model"] == "gateway/standard"


@pytest.mark.parametrize("change", ["session", "model", "effort"])
def test_fallback_is_not_restored_for_another_request_or_conversation(scenario, change):
    fallback_once(scenario)
    new_harness(scenario)
    resume = "different-thread" if change == "session" else "retained-thread"
    if change == "model":
        scenario.harness.model = "gpt-6-sol"
    elif change == "effort":
        scenario.harness.effort = "low"
    assert run(scenario, resume=resume).error is None
    assert scenario.calls[-1]["model"] == scenario.harness.model
    assert scenario.calls[-1]["effort"] == scenario.harness.effort


def test_corrupt_saved_fallback_is_ignored(scenario):
    (scenario.directory / "codex_model.json").write_text("{")
    assert run(scenario, resume="retained-thread").error is None
    assert scenario.calls[-1]["model"] == "gpt-6-astra"
    assert scenario.calls[-1]["effort"] == "xhigh"


@pytest.mark.parametrize("patch", [
    {"version": 999}, {"active_model": ""}, {"active_effort": {}},
    {"active_model": "unrelated-model", "active_resolved_model": "unrelated-model"},
    {"active_model": "other/gpt-6.1-sol", "active_resolved_model": "other/gpt-6.1-sol"},
], ids=["version", "empty_model", "bad_effort", "unrelated_model", "other_provider"])
def test_invalid_or_unrelated_saved_choices_do_not_change_primary(scenario, patch):
    saved = fallback_once(scenario)
    saved.update(patch)
    (scenario.directory / "codex_model.json").write_text(json.dumps(saved))
    new_harness(scenario)
    assert run(scenario, resume="retained-thread").error is None
    assert scenario.calls[-1]["model"] == "gpt-6-astra"
    assert scenario.calls[-1]["effort"] == "xhigh"


def test_failed_switch_checkpoint_leaves_active_model_unchanged(scenario, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("checkpoint disk unavailable")

    monkeypatch.setattr(scenario.harness, "_save_discovery_model", unavailable)
    scenario.answers = [overloaded() for _ in range(3)]
    result = run(scenario)
    assert result.error_code == "serverOverloaded"
    assert [call["model"] for call in scenario.calls] == ["gpt-6-astra"] * 3
    assert scenario.harness._active_model == "gpt-6-astra"
    assert scenario.harness._active_effort == "xhigh"
    assert not (scenario.directory / "codex_model.json").exists()
