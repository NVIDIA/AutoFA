# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol tests use a real child process, without a model or login."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.harness import codex_app_server
from autofa.harness.codex_app_server import (
    CodexAppServer,
    CodexAppServerError,
    CodexTurnConnection,
    _error_details,
    compact_codex_thread,
    run_codex_turn,
)
from autofa.harness.usage import CodexUsage
from autofa.mcp import runtime, server
from autofa.metrics import claim_compiler
from autofa.session.usage import usage_summary

_SERVER = r"""
import json, os, sys
def send(obj):
    print(json.dumps(obj), flush=True)
def event(method, params):
    send({'method': method, 'params': params})
def reply(request, result):
    send({'id': request['id'], 'result': result})
mode = ''
calls = 0
model_error = {'message': "The 'gpt-6-astra' model is not supported when using Codex "
                         "with a ChatGPT account. Authorization: SECRET_TOKEN",
               'codexErrorInfo': 'other'}
def tool(name='evidence', args=None, namespace=None):
    global calls
    calls += 1
    send({'id': 'tool-' + str(calls), 'method': 'item/tool/call', 'params': {
        'threadId': 'thread-1', 'turnId': 'turn-1', 'callId': str(calls),
        'tool': name, 'arguments': {} if args is None else args, 'namespace': namespace}})
def finish(text='done'):
    event('item/completed', {'threadId': 'thread-1', 'turnId': 'turn-1', 'item': {
        'id': 'commentary', 'type': 'agentMessage', 'phase': 'commentary', 'text': 'Checking.'}})
    for total, last in [(200, 20), (210, 10)]:
        event('thread/tokenUsage/updated', {'threadId':'thread-1', 'turnId':'turn-1',
            'tokenUsage': {'total': {'inputTokens': total, 'outputTokens': total//10},
                           'last': {'inputTokens': last, 'outputTokens': last//10}}})
    event('item/completed', {'threadId': 'thread-1', 'turnId': 'turn-1', 'item': {
        'id': 'answer', 'type': 'agentMessage', 'phase': 'final_answer', 'text': text}})
    event('turn/completed', {'threadId': 'thread-1', 'turn': {
        'id': 'turn-1', 'status': 'completed', 'items': []}})
with open(os.environ['AUTOFA_TEST_CODEX_LOG'], 'a') as out:
    out.write(json.dumps({'argv': sys.argv[1:]}) + '\n')
for line in sys.stdin:
    request = json.loads(line)
    with open(os.environ['AUTOFA_TEST_CODEX_LOG'], 'a') as out:
        out.write(line)
    method = request.get('method')
    params = request.get('params', {})
    if method == 'initialize':
        event('configWarning', {'message': 'Test warning'})
        reply(request, {})
    elif method == 'config/read':
        reply(request, {'config': {'mcp_servers': {'external.server': {
            'enabled': True, 'command': '/test/mcp', 'tool_timeout_sec': None}}}})
    elif method == 'account/usage/read':
        reply(request, {'threadUsage': {'threadId': 'thread-1',
              'nativeUsageUsdMicros': (5_000_000 + calls * 500_000) if mode == 'rounds'
                                     else 5_500_000 if mode else 5_000_000,
              'estimatedUsageUsdMicros': None, 'estimatedUsageCreditsMicros': 0}})
    elif method in ('thread/start', 'thread/resume'):
        if params.get('model') == 'denied':
            send({'id': request['id'], 'error': {'code': -32600, **model_error}})
        elif params.get('model') == 'invalid':
            send({'id': request['id'], 'error': {'code': -32602, 'message': 'SECRET_TOKEN'}})
        elif params.get('model') == 'crash':
            sys.exit(1)
        elif params.get('model') == 'state-crash':
            print('failed to initialize sqlite state runtime', file=sys.stderr, flush=True)
            sys.exit(1)
        else:
            environments = None if params.get('model') == 'old-cli' else []
            busy = os.environ.get('AUTOFA_TEST_COMPACT_MODE') == 'busy'
            thread_status = 'active' if busy else 'idle'
            reply(request, {'thread': {'id':'thread-1', 'environments': environments,
                                      'status': {'type': thread_status}},
                            'sandbox': {'type':'readOnly'}})
    elif method == 'thread/compact/start':
        compact_mode = os.environ.get('AUTOFA_TEST_COMPACT_MODE', 'success')
        if compact_mode == 'unsupported':
            send({'id':request['id'],'error':{'code':-32601,'message':'unsupported'}})
            continue
        reply(request, {})
        event('turn/started', {'threadId':'thread-1','turn': {'id':'compact-1'}})
        if compact_mode == 'pending':
            continue
        if compact_mode == 'legacy':
            event('thread/compacted', {'threadId':'thread-1','turnId':'compact-1'})
        elif compact_mode != 'unconfirmed':
            event('item/completed', {'threadId':'thread-1','turnId':'compact-1',
                                    'item': {'id':'summary','type':'contextCompaction'}})
        error = {'message':'SECRET_TOKEN','codexErrorInfo':'contextWindowExceeded'}
        event('turn/completed', {'threadId':'thread-1','turn': {'id':'compact-1',
              'status':'failed' if compact_mode == 'failed' else 'completed', 'items':[],
              'error': error if compact_mode == 'failed' else None}})
    elif method == 'turn/start':
        mode = params['input'][0]['text']
        if mode == 'pending_ack':
            event('turn/started', {'threadId':'thread-1', 'turn': {'id':'turn-1'}})
            continue
        if mode == 'tool':
            # Exercise a server request arriving before the start response.
            tool()
        reply(request, {'turn': {'id': 'turn-1', 'status': 'inProgress', 'items': []}})
        event('turn/started', {'threadId':'thread-1', 'turn': {'id':'turn-1'}})
        if mode == 'pending':
            continue
        elif mode == 'rounds':
            tool('advance_round')
        elif mode == 'limit':
            tool()
        elif mode == 'compaction':
            event('turn/completed', {'threadId':'thread-1', 'turn': {
                'id':'older-turn', 'status':'completed', 'items': [
                    {'id':'older-summary','type':'contextCompaction'}]}})
            event('thread/compacted', {'threadId':'thread-1', 'turnId':'turn-1'})
            tool()
        elif mode == 'unknown':
            tool('not_registered')
        elif mode == 'namespace':
            tool(namespace='unrelated')
        elif mode == 'bad_arguments':
            tool(args='not an object')
        elif mode == 'approval':
            send({'id':'approval','method':'item/commandExecution/requestApproval','params':{}})
        elif mode == 'failure':
            event('error', {'threadId':'thread-1', 'turnId':'turn-1', 'willRetry':True,
                  'error':{'message':'temporary'}})
            event('turn/completed', {'threadId':'thread-1','turn':{'id':'turn-1',
                  'status':'failed','items':[], 'error':{'message':'SECRET_TOKEN',
                  'codexErrorInfo': {'httpConnectionFailed': {'httpStatusCode':401}}}}})
        elif mode.startswith('model_error'):
            error = model_error
            if mode != 'model_error':
                event('error', {'threadId':'thread-1', 'turnId':'turn-1', 'willRetry':False,
                               'error':model_error})
                error = {'message':'Turn failed', 'codexErrorInfo':'other'}
            if mode == 'model_error_recovered':
                finish()
            else:
                event('turn/completed', {'threadId':'thread-1', 'turn':{
                      'id':'turn-1', 'status':'failed', 'items':[], 'error':error}})
        elif mode == 'complete_exit':
            finish()
            sys.exit(0)
        elif mode != 'tool':
            finish()
    elif method == 'turn/interrupt':
        reply(request, {})
    elif method == 'no/reply':
        pass
    elif 'method' not in request and str(request.get('id','')).startswith('tool-'):
        if mode == 'limit' and calls == 1:
            tool()
        elif mode == 'rounds' and calls == 1:
            tool('advance_round')
        else:
            finish(json.dumps(request.get('result')))
    elif request.get('id') == 'approval' and 'method' not in request:
        finish(request['result']['decision'])
"""


@pytest.fixture
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    executable = tmp_path / "codex"
    executable.write_text(f"#!{sys.executable}\n{_SERVER}")
    executable.chmod(0o755)
    log = tmp_path / "messages.jsonl"
    monkeypatch.setenv("AUTOFA_TEST_CODEX_LOG", str(log))
    return executable, log


@pytest.fixture
def observed_clients(monkeypatch):
    """Expose each live CodexAppServer the transport opens."""
    clients = []

    class ObservedClient(CodexAppServer):
        async def __aenter__(self):
            clients.append(self)
            return await super().__aenter__()

    monkeypatch.setattr(codex_app_server, "CodexAppServer", ObservedClient)
    return clients


def _messages(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def _sent(fake_codex, method: str) -> list[dict]:
    return [x["params"] for x in _messages(fake_codex[1]) if x.get("method") == method]


def _record_usage(fake_codex) -> Path:
    directory = fake_codex[0].parent
    (directory / "config.json").write_text(json.dumps({"run_id": "run", "round": 0}))
    return directory


def _args(fake_codex, **overrides):
    executable, _ = fake_codex

    async def call_tool(name, arguments):
        return {
            "success": True,
            "contentItems": [
                {"type": "inputText", "text": f"{name}: {arguments}"},
                {"type": "inputImage", "imageUrl": "data:image/png;base64,YQ=="},
            ],
        }

    return {
        "prompt": "tool",
        "system_prompt": "AutoFA instructions",
        "model": "test-model",
        "cwd": str(executable.parent),
        "cli_path": str(executable),
        "tools": [{"name": "evidence", "description": "Read evidence", "inputSchema": {}}],
        "call_tool": call_tool,
        "restricted": True,
        **overrides,
    }


def _compact_args(fake_codex, **overrides):
    return {
        "thread_id": "thread-1",
        "cwd": str(fake_codex[0].parent),
        "cli_path": str(fake_codex[0]),
        **overrides,
    }


def _pending(operation, fake_codex, monkeypatch, **overrides):
    """Start a turn or compaction that the fake server never completes."""
    if operation == "turn":
        return run_codex_turn(**_args(fake_codex, prompt="pending", **overrides))
    monkeypatch.setenv("AUTOFA_TEST_COMPACT_MODE", "pending")
    return compact_codex_thread(**_compact_args(fake_codex, **overrides))


@pytest.mark.parametrize("operation,resume", [("turn", "thread-1"), ("compaction", None)])
def test_accounting_delay_cannot_consume_model_deadline(
    fake_codex, monkeypatch, operation, resume,
):
    calls = []

    async def slow_account(self, client, **kwargs):
        # Advance the loop clock past the whole model deadline instead of sleeping:
        # a deadline still active around accounting expires at the next await.
        calls.append(kwargs)
        loop, clock = asyncio.get_running_loop(), asyncio.get_running_loop().time
        loop.time = lambda: clock() + 60
        await asyncio.sleep(0)
        return False

    monkeypatch.setattr(CodexUsage, "account", slow_account)
    if operation == "turn":
        result = asyncio.run(run_codex_turn(**_args(
            fake_codex, prompt="plain", resume=resume, timeout_s=30,
        )))
        assert result.text == "done"
        assert calls[-1] == {"status": "completed"}
    else:
        result = asyncio.run(compact_codex_thread(**_compact_args(fake_codex, timeout_s=30)))
    assert result.status == "completed"


def test_real_stdio_multiplexes_dynamic_text_and_image_tools(fake_codex):
    threads = []

    async def on_thread(thread_id):
        # ID must be persisted before turn/start is written to the child.
        assert not _sent(fake_codex, "turn/start")
        threads.append(thread_id)

    result = asyncio.run(run_codex_turn(**_args(fake_codex, effort="high", on_thread=on_thread)))
    assert (result.status, result.thread_id, threads) == ("completed", "thread-1", ["thread-1"])
    output = json.loads(result.text)
    assert output["success"]
    assert output["contentItems"][1] == {
        "type": "inputImage", "imageUrl": "data:image/png;base64,YQ==",
    }
    assert result.usage["input_tokens"] == 210
    started = _sent(fake_codex, "thread/start")[0]
    assert started["sandbox"] == "read-only"
    assert started["environments"] == []
    assert started["dynamicTools"][0]["type"] == "function"
    external = started["config"]["mcp_servers"]["external.server"]
    assert external["enabled"] is False and external["command"] == "/test/mcp"
    assert "tool_timeout_sec" not in external
    assert started["config"]["features.shell_tool"] is False
    assert started["config"]["features.unified_exec"] is False
    turn = _sent(fake_codex, "turn/start")[0]
    assert turn["effort"] == "high"
    assert turn["environments"] == []


def test_resume_uses_persisted_tools_and_checkpoints_new_currency_before_round_changes(
    fake_codex,
):
    directory = _record_usage(fake_codex)
    path = directory / "config.json"
    seen = []

    async def advance(name, arguments):
        state = json.loads(path.read_text())
        rows = usage_summary(directory)["by_round"]
        seen.append((state["round"], rows[-1]["known_cost_usd"]))
        state["round"] += 1
        path.write_text(json.dumps(state))
        return {"success": True, "contentItems": []}

    result = asyncio.run(run_codex_turn(**_args(
        fake_codex, resume="thread-1", prompt="rounds", call_tool=advance,
        tools=[{"name": "advance_round", "description": "advance", "inputSchema": {}}],
    )))
    assert result.status == "completed"
    resumed = _sent(fake_codex, "thread/resume")[0]
    assert resumed["threadId"] == "thread-1"
    assert "dynamicTools" not in resumed
    assert not _sent(fake_codex, "thread/start")
    [argv] = [message["argv"] for message in _messages(fake_codex[1]) if "argv" in message]
    assert not any(
        argument.startswith("sqlite_home=")
        for index, argument in enumerate(argv)
        if index and argv[index - 1] == "-c"
    )
    # Only usage after the resumed thread's baseline is recorded.
    assert result.cost_usd == 1 and result.cost_kind == "reported"
    assert seen == [(0, 0.5), (1, 0.5)]
    summary = usage_summary(directory)
    assert summary["totals"]["tokens"]["input_tokens"] == 30
    rounds = {row["round_index"]: row["known_cost_usd"] for row in summary["by_round"]}
    assert rounds[0] == rounds[1] == 0.5
    assert None not in rounds


def test_discovery_uses_workspace_write_and_harness_default_effort(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, restricted=False, effort="default")))
    assert result.status == "completed"
    started = _sent(fake_codex, "thread/start")[0]
    assert started["sandbox"] == "workspace-write"
    assert started["approvalPolicy"] == "never"
    assert "environments" not in started
    assert "effort" not in _sent(fake_codex, "turn/start")[0]


@pytest.mark.parametrize("prompt", ["unknown", "namespace", "bad_arguments"])
def test_invalid_tool_calls_do_not_reach_runtime(fake_codex, prompt):
    async def forbidden(*args):
        pytest.fail("Invalid calls must not reach the runtime")

    result = asyncio.run(run_codex_turn(**_args(fake_codex, prompt=prompt, call_tool=forbidden)))
    assert result.status == "completed"
    assert json.loads(result.text)["success"] is False


def test_command_approval_is_declined(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, prompt="approval")))
    assert result.text == "decline"


def test_tool_budget_interrupts_turn(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, prompt="limit", max_tool_calls=1)))
    assert result.status == "failed"
    assert result.error_code == "toolCallLimitExceeded"
    assert not result.retryable
    assert _sent(fake_codex, "turn/interrupt")


@pytest.mark.parametrize("operation,turn_id", [("turn", "turn-1"), ("compaction", "compact-1")])
def test_timeout_interrupts_active_work(fake_codex, monkeypatch, operation, turn_id):
    result = asyncio.run(_pending(operation, fake_codex, monkeypatch, timeout_s=0.2))
    assert result.error_code == "requestTimeout"
    assert _sent(fake_codex, "turn/interrupt") == [{"threadId": "thread-1", "turnId": turn_id}]


@pytest.mark.parametrize("operation", ["turn", "compaction"])
def test_cancellation_interrupts_and_propagates(fake_codex, monkeypatch, operation):
    async def scenario():
        started = asyncio.Event()

        async def on_event(method, params):
            if method == "turn/started":
                started.set()

        task = asyncio.create_task(_pending(operation, fake_codex, monkeypatch, on_event=on_event))
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert _sent(fake_codex, "turn/interrupt")


def test_cancelled_turn_stops_in_process_judge_before_closing(fake_codex, monkeypatch):
    directory = fake_codex[0].parent
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(
        run=SimpleNamespace(run_id="interrupted-run", run_dir=lambda: directory),
    ))
    started, closed = threading.Event(), threading.Event()

    async def compiler(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(claim_compiler, "_compiler_session", compiler)
    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", claim_compiler._run_compiler)

    async def call_tool(name, arguments):
        return server.codex_tool_response(await server.invoke_tool("judge_function", {}))

    async def scenario():
        task = asyncio.create_task(run_codex_turn(**_args(fake_codex, call_tool=call_tool)))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert closed.is_set()

    asyncio.run(scenario())
    assert _sent(fake_codex, "turn/interrupt")
    operation = json.loads(next((directory / "operations").glob("*.json")).read_text())
    assert operation["state"] == "cancelled"


def test_tool_failure_interrupts_and_propagates(fake_codex):
    async def call_tool(*args):
        raise RuntimeError("A local test failure")

    with pytest.raises(RuntimeError, match="local test failure"):
        asyncio.run(run_codex_turn(**_args(fake_codex, call_tool=call_tool)))
    assert _sent(fake_codex, "turn/interrupt")


def test_failure_event_is_not_confused_with_retriable_error_notification(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, prompt="failure")))
    assert result.status == "failed"
    assert result.error_code == "unauthorized"
    assert "codex login" in result.error
    assert "SECRET_TOKEN" not in result.error
    assert not result.retryable


@pytest.mark.parametrize(
    "model,code,retryable",
    [
        ("invalid", "-32602", False),
        ("crash", "transportClosed", True),
    ],
)
def test_request_error_and_eof_are_reported(fake_codex, model, code, retryable):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, model=model)))
    assert result.status == "failed"
    assert result.error_code == code
    assert result.retryable is retryable
    assert "SECRET_TOKEN" not in result.error


def test_restricted_fresh_turn_isolates_codex_state(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, prompt="plain")))

    assert result.status == "completed"
    [argv] = [message["argv"] for message in _messages(fake_codex[1]) if "argv" in message]
    sqlite_override = next(
        argv[index + 1]
        for index, argument in enumerate(argv)
        if argument == "-c" and argv[index + 1].startswith("sqlite_home=")
    )
    state_directory = Path(json.loads(sqlite_override.removeprefix("sqlite_home=")))
    assert not state_directory.exists()


def test_persistent_restricted_connection_isolates_codex_state(fake_codex):
    async def scenario():
        async with CodexTurnConnection() as connection:
            result = await run_codex_turn(
                **_args(fake_codex, prompt="plain", connection=connection)
            )
            [argv] = [
                message["argv"]
                for message in _messages(fake_codex[1])
                if "argv" in message
            ]
            sqlite_override = next(
                argv[index + 1]
                for index, argument in enumerate(argv)
                if argument == "-c" and argv[index + 1].startswith("sqlite_home=")
            )
            state_directory = Path(
                json.loads(sqlite_override.removeprefix("sqlite_home="))
            )
            assert state_directory.exists()
            assert result.status == "completed"
        return state_directory

    assert not asyncio.run(scenario()).exists()


def test_state_initialization_failure_is_safe_and_retryable(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, model="state-crash")))

    assert result.error_code == "stateInitializationFailed"
    assert result.error == "Codex could not initialize its local state (exit code 1)."
    assert result.retryable
    [argv] = [message["argv"] for message in _messages(fake_codex[1]) if "argv" in message]
    sqlite_override = next(
        argv[index + 1]
        for index, argument in enumerate(argv)
        if argument == "-c" and argv[index + 1].startswith("sqlite_home=")
    )
    assert not Path(json.loads(sqlite_override.removeprefix("sqlite_home="))).exists()


def test_state_initialization_json_rpc_error_is_retryable() -> None:
    code, _message, retryable = _error_details({
        "code": -32603,
        "message": "failed to initialize sqlite state runtime",
    })

    assert code == "stateInitializationFailed"
    assert retryable


def test_request_timeout_closes_process(fake_codex):
    async def scenario():
        async with CodexAppServer(str(fake_codex[0])) as client:
            client.request_timeout_s = 0.05
            with pytest.raises(CodexAppServerError) as caught:
                await client.request("no/reply", {})
            assert caught.value.code == "requestTimeout"
        assert client.process.returncode is not None

    asyncio.run(scenario())


def test_restricted_mode_fails_closed_when_cli_ignores_environment_override(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, model="old-cli")))
    assert result.error_code == "restrictedToolsUnsupported"
    assert not _sent(fake_codex, "turn/start")


def test_cancellation_before_start_ack_still_interrupts_known_active_turn(
    fake_codex, observed_clients,
):
    async def scenario():
        task = asyncio.create_task(run_codex_turn(**_args(fake_codex, prompt="pending_ack")))
        async with asyncio.timeout(2):
            while not observed_clients or not observed_clients[0].active_turns:
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observed_clients[0].process.returncode is not None

    asyncio.run(scenario())
    assert _sent(fake_codex, "turn/interrupt")


def test_compaction_is_observed_before_following_evidence_read(fake_codex, observed_clients):
    observed = []

    async def on_event(method, params):
        if method == "thread/compacted":
            # Notification callbacks may await other RPCs. The reader must keep
            # resolving responses while later tool requests wait for this work.
            await observed_clients[0].request("config/read", {"includeLayers": False})
            await asyncio.sleep(0.01)
            observed.append("compacted")

    async def call_tool(name, args):
        assert observed == ["compacted"]
        observed.append("read")
        return {"success": True, "contentItems": []}

    result = asyncio.run(run_codex_turn(**_args(
        fake_codex, prompt="compaction", on_event=on_event, call_tool=call_tool,
    )))
    assert result.status == "completed"
    assert observed == ["compacted", "read"]


def test_automatic_compaction_records_gap_without_changing_turn_tokens_or_cost(fake_codex):
    directory = _record_usage(fake_codex)
    result = asyncio.run(run_codex_turn(**_args(fake_codex, prompt="compaction")))
    assert result.status == "completed"
    assert result.usage["input_tokens"] == 210
    assert result.cost_usd == 5.5
    summary = usage_summary(directory)
    assert summary["totals"]["tokens"]["input_tokens"] == 210
    assert summary["totals"]["known_cost_usd"] == 5.5
    assert summary["totals"]["compaction_usage_gaps"] == 1


def test_completion_before_process_exit_is_preserved(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, prompt="complete_exit")))
    assert result.status == "completed"
    assert result.text == "done"


@pytest.mark.parametrize("mode", ["success", "legacy"])
def test_native_compaction_preserves_thread_and_records_explicit_missing_usage(
    fake_codex, monkeypatch, mode,
):
    directory = _record_usage(fake_codex)
    monkeypatch.setenv("AUTOFA_TEST_COMPACT_MODE", mode)
    result = asyncio.run(compact_codex_thread(**_compact_args(fake_codex)))
    assert (result.status, result.thread_id) == ("completed", "thread-1")
    assert result.turn_id == "compact-1"
    assert result.error is None
    methods = [x.get("method") for x in _messages(fake_codex[1])]
    assert methods.index("thread/resume") < methods.index("thread/compact/start")
    assert "thread/start" not in methods
    assert "turn/start" not in methods
    resumed = _sent(fake_codex, "thread/resume")[0]
    assert resumed["threadId"] == "thread-1"
    assert resumed["sandbox"] == "read-only"
    for field in ("model", "baseInstructions", "developerInstructions", "dynamicTools"):
        assert field not in resumed
    summary = usage_summary(directory)
    assert summary["totals"]["tokens"]["total_tokens"] is None
    assert summary["totals"]["compaction_usage_gaps"] == 1
    assert summary["by_role"][0]["role"] == "compaction"


@pytest.mark.parametrize(
    "mode,error",
    [
        ("unconfirmed", "compactionUnconfirmed"),
        ("failed", "contextWindowExceeded"),
        ("unsupported", "-32601"),
        ("busy", "threadBusy"),
    ],
)
def test_native_compaction_requires_both_success_signals(fake_codex, monkeypatch, mode, error):
    monkeypatch.setenv("AUTOFA_TEST_COMPACT_MODE", mode)
    result = asyncio.run(compact_codex_thread(**_compact_args(fake_codex)))
    assert result.status == "failed"
    assert result.error_code == error
    assert "SECRET_TOKEN" not in result.error
    if mode == "busy":
        assert not _sent(fake_codex, "thread/compact/start")


@pytest.mark.parametrize(
    "code,retryable",
    [
        ("rateLimitExceeded", True),
        ("contextWindowExceeded", False),
        ({"httpConnectionFailed": {"httpStatusCode": 503}}, True),
        ({"httpConnectionFailed": {"httpStatusCode": 400}}, False),
    ],
)
def test_retry_policy_uses_protocol_error_codes(code, retryable):
    assert _error_details({"codexErrorInfo": code})[2] is retryable


@pytest.mark.parametrize("overrides", [
    {"prompt": "model_error", "model": "gpt-6-astra"},
    {"prompt": "model_error_event", "model": "gpt-6-astra"},
    {"model": "denied"},  # Rejected while starting the thread.
])
def test_model_denial_explains_selected_model_and_recovery_without_secrets(
    fake_codex, overrides,
):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, **overrides)))
    assert result.status == "failed"
    assert result.error_code == "modelAccessDenied"
    assert result.retryable is False
    assert f"model '{overrides['model']}'" in result.error
    assert "Settings > LLM endpoints > AutoFA harness" in result.error
    assert "SECRET_TOKEN" not in result.error
    if "prompt" in overrides:
        assert result.thread_id == "thread-1"


def test_success_does_not_inherit_an_earlier_model_error(fake_codex):
    result = asyncio.run(run_codex_turn(**_args(fake_codex, prompt="model_error_recovered")))
    assert result.status == "completed"
    assert result.error is None


@pytest.mark.parametrize("detail,code", [
    ("Model does not exist or you do not have access to it. Bearer SECRET_TOKEN",
     "modelAccessDenied"),
    ("SECRET_TOKEN", "forbidden"),
])
def test_http_403_is_model_denial_or_permission_error_never_bad_login(detail, code):
    actual, message, retryable = _error_details({
        "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 403}},
        "additionalDetails": detail,
    }, model="gpt-6-astra")
    assert actual == code
    assert ("gpt-6-astra" in message) is (code == "modelAccessDenied")
    assert "codex login" not in message
    assert "SECRET_TOKEN" not in message
    assert retryable is False
