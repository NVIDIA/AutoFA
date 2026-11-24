# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codex app-server transport for in-process AutoFA tools.

The protocol comes from ``codex app-server generate-json-schema --experimental``
(Codex 0.166.0). No API key or separate MCP process is needed: the installed
harness owns authentication, model routing, and persisted conversation history.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autofa.harness.errors import is_model_access_error, model_access_message
from autofa.harness.model_catalog import _find_codex_cli
from autofa.harness.usage import CodexUsage
from autofa.session.admission import paid_async_dispatch
from autofa.util.usage_values import normalize_tokens

ToolCallback = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]
ThreadCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class CodexTurnResult:
    thread_id: str = ""
    turn_id: str = ""
    status: str = "failed"
    text: str = ""
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False
    cost_usd: float | None = None
    cost_kind: str = "unknown"


_TRANSIENT_ERRORS = {
    "rateLimitExceeded",
    "flexUnavailable",
    "serverOverloaded",
    "internalServerError",
    "httpConnectionFailed",
    "responseStreamConnectionFailed",
    "responseStreamDisconnected",
    "responseTooManyFailedAttempts",
    "transportClosed",
    "requestTimeout",
    "remoteCompactionFailed",
    "stateInitializationFailed",
}

_STATE_INITIALIZATION_ERROR = "failed to initialize sqlite state runtime"
_STDERR_LIMIT_BYTES = 64 * 1024


def _error_details(error: Any, *, model: str | None = None) -> tuple[str, str, bool]:
    """Keep authentication helpers and potentially secret upstream errors out of logs."""
    error = error if isinstance(error, dict) else {}
    info = error.get("codexErrorInfo", error.get("code", "other"))
    detail: dict[str, Any] = {}
    if isinstance(info, dict) and info:
        code = next(iter(info))
        detail = info[code] if isinstance(info[code], dict) else {}
    else:
        code = str(info)
    status = detail.get("httpStatusCode")
    if status == 401 or code == "unauthorized":
        return "unauthorized", "Codex authentication failed. Run `codex login` and retry.", False
    # Codes such as `other` also wrap provider model/entitlement errors. Inspect
    # their explanation for classification, but never echo raw auth/helper output.
    explanation = " ".join(str(error.get(key) or "") for key in (
        "code", "message", "additionalDetails", "data",
    ))
    if _STATE_INITIALIZATION_ERROR in explanation.lower():
        code = "stateInitializationFailed"
    if is_model_access_error(explanation):
        return "modelAccessDenied", model_access_message("Codex", model), False
    if status == 403:
        return "forbidden", (
            "Codex permission denied. Check that your account or workspace allows the "
            "selected model, or choose another model in Settings > LLM endpoints."
        ), False
    from autofa.session.recovery import discovery_failure_kind

    kind = discovery_failure_kind(explanation, code)
    if kind == "terminal" and (code == "other" or code in _TRANSIENT_ERRORS):
        # Generic provider errors can include misleading retryable transport
        # codes. Do not retry account, policy or invalid-input rejections.
        return "nonretryableError", (
            f"Codex rejected the request ({code}). Check account and request settings."
        ), False
    if code == "other" and status not in (400, 404, 422):
        if kind == "compaction":
            code = "remoteCompactionFailed"
        elif kind == "context":
            code = "contextWindowExceeded"
        elif kind == "transport":
            code = "responseStreamDisconnected"
    messages = {
        "serverOverloaded": "The selected Codex model is temporarily at capacity.",
        "contextWindowExceeded": "Codex context is full. Start a new session.",
        "usageLimitExceeded": "Codex usage limit reached. Check your account limits.",
        "sessionBudgetExceeded": "Codex session budget reached.",
        "badRequest": "Codex rejected the request. Check the model and reasoning effort.",
        "-32600": "Codex rejected the request. Check your Codex configuration and CLI version.",
        "-32601": "Codex does not support this operation. Update the Codex CLI.",
        "-32602": "Codex rejected an option. Update the Codex CLI and check model settings.",
        "-32603": "Codex app-server reported an internal error.",
        "sandboxError": "Codex could not run in its sandbox. Check your Codex installation.",
        "transportClosed": "Codex app-server exited before completing the turn.",
        "requestTimeout": "Codex app-server did not respond in time.",
        "remoteCompactionFailed": "Codex remote compaction failed before completion.",
        "responseStreamDisconnected": "Codex response stream disconnected before completion.",
        "stateInitializationFailed": "Codex could not initialize its local state.",
        "toolCallLimitExceeded": "Codex exceeded this session's tool-call limit.",
        "restrictedToolsUnsupported": (
            "This Codex CLI cannot isolate judge tools. Update Codex before running judging."
        ),
        "compactionUnconfirmed": "Codex did not confirm that context compaction completed.",
        "threadBusy": "Codex is running this thread. Wait for it to finish before compacting.",
    }
    retryable = code in _TRANSIENT_ERRORS and status not in (400, 404, 422)
    message = messages.get(code, f"Codex turn failed ({code}).")
    diagnostics = []
    if code in {"stateInitializationFailed", "transportClosed"}:
        if type(error.get("returnCode")) is int:
            diagnostics.append(f"exit code {error['returnCode']}")
        if isinstance(error.get("readerError"), str):
            diagnostics.append(f"reader error {error['readerError']}")
    if diagnostics:
        message = f"{message.removesuffix('.')} ({'; '.join(diagnostics)})."
    return code, message, retryable


class CodexAppServerError(RuntimeError):
    def __init__(self, error: dict[str, Any]) -> None:
        self.error = error
        self.code, message, self.retryable = _error_details(error)
        self.usage: dict[str, int] = {}
        super().__init__(message)


class _ToolCallLimit(CodexAppServerError):
    def __init__(self) -> None:
        super().__init__({"code": "toolCallLimitExceeded"})


# AutoFA owns the tools and instructions for these sessions. Preserve the user's
# provider/auth settings, but do not import unrelated integrations or automation.
_BASE_CONFIG: dict[str, Any] = {
    "features.hooks": False,
    "features.plugins": False,
    "features.apps": False,
    "features.multi_agent": False,
    "features.memories": False,
    "features.skip_host_skill_discovery": True,
    "features.code_mode": False,
    "features.code_mode_only": False,
    "features.unified_exec": False,
    "features.tool_suggest": False,
    "features.image_generation": False,
    "features.browser_use": False,
    "features.computer_use": False,
    "features.default_mode_request_user_input": False,
    "features.send_message_to_user_async": False,
    "web_search": "disabled",
    "project_doc_max_bytes": 0,
    "skills.bundled.enabled": False,
    "skills.include_instructions": False,
}

_RESTRICTED_CONFIG: dict[str, Any] = {
    "features.shell_tool": False,
    "features.view_image": False,
    "features.context_management": False,
    "features.token_budget": False,
    "features.current_time_reminder": False,
    "features.sleep_tool": False,
    "features.request_permissions_tool": False,
    "tools.update_plan.enabled": False,
}


class CodexAppServer:
    """A multiplexed JSON-lines connection, including server-initiated tool calls."""

    def __init__(
        self,
        cli_path: str | None = None,
        *,
        cwd: str | None = None,
        on_request: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        on_notification: EventCallback | None = None,
        request_timeout_s: float = 30.0,
        isolated_state: bool = False,
    ) -> None:
        self.cli_path = cli_path
        self.cwd = cwd
        self.on_request = on_request
        self.on_notification = on_notification
        self.request_timeout_s = request_timeout_s
        self.isolated_state = isolated_state
        self.process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[None] | None = None
        self._stderr = bytearray()
        self._notifier: asyncio.Task[None] | None = None
        self._requests: set[asyncio.Task[None]] = set()
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._events: asyncio.Queue[tuple[str, dict[str, Any]] | Exception] = asyncio.Queue()
        self._notifications: asyncio.Queue[
            tuple[str, dict[str, Any], asyncio.Future[None]] | Exception
        ] = asyncio.Queue()
        self._last_notification: asyncio.Future[None] | None = None
        self._notification_error: Exception | None = None
        self._write_lock = asyncio.Lock()
        self._sequence = 0
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._state_directory: tempfile.TemporaryDirectory[str] | None = None
        self.active_turns: dict[str, str] = {}

    async def __aenter__(self) -> CodexAppServer:
        binary = _find_codex_cli(self.cli_path)
        args = [binary, "app-server", "--listen", "stdio://"]
        for key, value in _BASE_CONFIG.items():
            args.extend(["-c", f"{key}={json.dumps(value)}"])
        if self.isolated_state:
            self._state_directory = tempfile.TemporaryDirectory(
                prefix=".codex-state-",
                dir=self.cwd,
            )
            args.extend(["-c", f"sqlite_home={json.dumps(self._state_directory.name)}"])
        try:
            self.process = await asyncio.create_subprocess_exec(
                *args,
                cwd=self.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Evidence images and complete assistant items can be large.
                limit=16 * 1024 * 1024,
            )
        except BaseException:
            self._cleanup_state_directory()
            raise
        self._reader = asyncio.create_task(self._read_messages())
        self._stderr_reader = asyncio.create_task(self._read_stderr())
        self._notifier = asyncio.create_task(self._notify())
        try:
            await self.request(
                "initialize",
                {
                    "clientInfo": {"name": "autofa", "title": "AutoFA", "version": "1.0"},
                    "capabilities": {"experimentalApi": True, "explicitGatewayOauth": True},
                },
            )
            await self._send({"method": "initialized", "params": {}})
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _send(self, message: dict[str, Any]) -> None:
        process = self.process
        if self._closed or process is None or process.stdin is None:
            raise CodexAppServerError({"code": "transportClosed"})
        async with self._write_lock:
            try:
                process.stdin.write((json.dumps(message) + "\n").encode())
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise CodexAppServerError({"code": "transportClosed"}) from exc

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._sequence += 1
        request_id = self._sequence
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method, "params": params})
            try:
                async with asyncio.timeout(self.request_timeout_s):
                    return await future
            except TimeoutError as exc:
                raise CodexAppServerError({"code": "requestTimeout"}) from exc
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def next_event(self) -> tuple[str, dict[str, Any]]:
        event = await self._events.get()
        if isinstance(event, Exception):
            raise event
        return event

    async def _read_messages(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        failure: Exception = CodexAppServerError({"code": "transportClosed"})
        reader_error: Exception | None = None
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("Invalid Codex app-server message")
                method = message.get("method")
                if isinstance(method, str):
                    params = message.get("params") or {}
                    if not isinstance(params, dict):
                        raise ValueError("Invalid Codex app-server parameters")
                    if "id" in message:
                        task = asyncio.create_task(
                            self._answer_request(
                                message["id"],
                                method,
                                params,
                                self._last_notification,
                            )
                        )
                        self._requests.add(task)
                        task.add_done_callback(self._requests.discard)
                    else:
                        if method == "turn/started":
                            thread_id = params.get("threadId")
                            turn_id = params.get("turn", {}).get("id")
                            if isinstance(thread_id, str) and isinstance(turn_id, str):
                                self.active_turns[thread_id] = turn_id
                        barrier = asyncio.get_running_loop().create_future()
                        self._last_notification = barrier
                        await self._notifications.put((method, params, barrier))
                elif "id" in message:
                    future = self._pending.get(message["id"])
                    if future is None or future.done():
                        continue
                    if "error" in message:
                        future.set_exception(CodexAppServerError(message["error"]))
                    elif isinstance(message.get("result"), dict):
                        future.set_result(message["result"])
                    else:
                        future.set_exception(CodexAppServerError({"code": "invalidResponse"}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reader_error = exc
        finally:
            if reader_error is not None or not self._closed:
                failure = await self._transport_failure(reader_error)
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(failure)
            if not self._closed:
                # Preserve the ordering of a final turn/completed before EOF.
                await self._notifications.put(failure)

    async def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while chunk := await self.process.stderr.read(8192):
            self._stderr.extend(chunk)
            if len(self._stderr) > _STDERR_LIMIT_BYTES:
                del self._stderr[:-_STDERR_LIMIT_BYTES]

    async def _transport_failure(self, reader_error: Exception | None) -> CodexAppServerError:
        process = self.process
        if process is not None and process.returncode is None:
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=0.2)
        if self._stderr_reader is not None:
            await asyncio.wait({self._stderr_reader}, timeout=0.2)
        stderr = self._stderr.decode(errors="replace")
        error = {
            "code": (
                "stateInitializationFailed"
                if _STATE_INITIALIZATION_ERROR in stderr.lower()
                else "transportClosed"
            ),
        }
        if process is not None and process.returncode is not None:
            error["returnCode"] = process.returncode
        if reader_error is not None:
            error["readerError"] = type(reader_error).__name__
        return CodexAppServerError(error)

    async def _notify(self) -> None:
        """Preserve notification/tool ordering without blocking the RPC reader."""
        while True:
            notification = await self._notifications.get()
            if isinstance(notification, Exception):
                await self._events.put(notification)
                return
            method, params, barrier = notification
            try:
                if self.on_notification is not None and self._notification_error is None:
                    await self.on_notification(method, params)
                await self._events.put((method, params))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._notification_error = exc
                await self._events.put(exc)
            finally:
                if not barrier.done():
                    barrier.set_result(None)

    async def _answer_request(
        self,
        request_id: Any,
        method: str,
        params: dict[str, Any],
        after_notification: asyncio.Future[None] | None,
    ) -> None:
        try:
            if method in (
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
            ):
                response = {"decision": "decline"}
            elif method == "item/tool/requestUserInput":
                response = {"answers": {}}
            elif method == "item/tool/call" and self.on_request is not None:
                # A context-compaction notification must reach the evidence
                # tracker before a following evidence read can mutate its epoch.
                if after_notification is not None:
                    await asyncio.shield(after_notification)
                if self._notification_error is not None:
                    raise self._notification_error
                response = await self.on_request(method, params)
            else:
                await self._send(
                    {
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": "AutoFA does not support this server request.",
                        },
                    }
                )
                return
            await self._send({"id": request_id, "result": response})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A broken tool callback must not leave Codex waiting indefinitely.
            with suppress(Exception):
                await self._send(
                    {
                        "id": request_id,
                        "result": {
                            "success": False,
                            "contentItems": [{"type": "inputText", "text": "AutoFA tool failed."}],
                        },
                    }
                )
            await self._events.put(exc)

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_resources())
        interrupted = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                # A second Ctrl+C must not release the runtime while a tool can
                # still mutate it; propagate the stop after draining safely.
                interrupted = True
        self._close_task.result()
        if interrupted:
            raise asyncio.CancelledError

    async def _close_resources(self) -> None:
        # Stop admission before draining callbacks: a server may send more tool
        # calls while a cancelled worker finishes its checkpoint or HTTP call.
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        await self.drain_requests()
        if self._notifier is not None:
            self._notifier.cancel()
            await asyncio.gather(self._notifier, return_exceptions=True)
        process = self.process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        if self._stderr_reader is not None:
            self._stderr_reader.cancel()
            await asyncio.gather(self._stderr_reader, return_exceptions=True)
        self._cleanup_state_directory()

    def _cleanup_state_directory(self) -> None:
        state_directory, self._state_directory = self._state_directory, None
        if state_directory is not None:
            state_directory.cleanup()

    async def drain_requests(self) -> None:
        """Finish cancelled tool cleanup before another turn can use its runtime."""
        requests = tuple(self._requests)
        for task in requests:
            task.cancel()
        if requests:
            await asyncio.gather(*requests, return_exceptions=True)


class CodexTurnConnection:
    """Lazy transport retained only across successful turns of one isolated review.

    The CLI cannot override environments when resuming a persisted thread. Keep
    its acknowledged empty environment in the live server for bounded repairs;
    every turn still explicitly disables environments. Any failure discards the
    connection, leaving the caller's existing reconnect policy in charge.
    """

    def __init__(self) -> None:
        self.client: CodexAppServer | None = None
        self.thread_id = ""
        self.closed_turns: set[str] = set()
        self._signature: str | None = None
        self._busy = False
        self._keep = False

    async def __aenter__(self) -> CodexTurnConnection:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        client, self.client = self.client, None
        self.thread_id = ""
        self._signature = None
        self.closed_turns.clear()
        if client is not None:
            await client.close()

    @asynccontextmanager
    async def use(
        self, *, cli_path: str | None, cwd: str, on_request: Callable,
        on_notification: EventCallback, signature: str,
    ):
        if self._busy:
            raise RuntimeError("A persistent Codex review cannot run concurrent turns")
        if self.client is not None and self._signature != signature:
            raise ValueError("A persistent Codex review cannot change its isolated configuration")
        self._busy, self._keep = True, False
        try:
            if self.client is None:
                self.client = CodexAppServer(
                    cli_path, cwd=cwd, on_request=on_request, on_notification=on_notification,
                    isolated_state=True,
                )
                self._signature = signature
                await self.client.__aenter__()
            else:
                self.client.on_request = on_request
                self.client.on_notification = on_notification
            yield self.client
            if self._keep:
                await self.client.drain_requests()
        except BaseException:
            await self.close()
            raise
        finally:
            if not self._keep:
                await self.close()
            self._busy = False


def _tool_failure(message: str) -> dict[str, Any]:
    return {"success": False, "contentItems": [{"type": "inputText", "text": message}]}


def _without_nulls(value: Any) -> Any:
    """Config reads contain null defaults; TOML overrides cannot express them."""
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    names: set[str] = set()
    for tool in tools:
        spec = {"type": "function", **tool}
        name = spec.get("name")
        if spec["type"] != "function" or not isinstance(name, str) or not name or name in names:
            raise ValueError("AutoFA Codex tools must have unique function names")
        names.add(name)
        result.append(spec)
    return result


async def _session_config(
    client: CodexAppServer,
    cwd: str,
    *,
    restricted: bool = False,
) -> dict[str, Any]:
    """Disable ambient automation while retaining the installed harness's auth."""
    effective = await client.request("config/read", {"cwd": cwd, "includeLayers": False})
    config = dict(_BASE_CONFIG)
    if restricted:
        config.update(_RESTRICTED_CONFIG)
    # Empty tables merge with user config. Explicitly disable each configured
    # server, preserving its transport and omitting null TOML defaults.
    configured = effective.get("config", {}).get("mcp_servers", {})
    if isinstance(configured, dict):
        config["mcp_servers"] = {
            name: {**_without_nulls(server), "enabled": False}
            for name, server in configured.items()
            if isinstance(server, dict)
        }
    return config


def _usage_delta(total: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    current, previous = normalize_tokens(total), normalize_tokens(baseline)
    return {key: max(0, value - previous.get(key, 0)) for key, value in current.items()}



async def run_codex_turn(  # noqa: C901
    *,
    prompt: str,
    system_prompt: str,
    model: str,
    cwd: str,
    tools: list[dict[str, Any]],
    call_tool: ToolCallback,
    effort: str | None = None,
    cli_path: str | None = None,
    resume: str | None = None,
    on_event: EventCallback | None = None,
    on_thread: ThreadCallback | None = None,
    restricted: bool = False,
    timeout_s: float | None = None,
    max_tool_calls: int | None = None,
    connection: CodexTurnConnection | None = None,
    usage_role: str = "discovery",
) -> CodexTurnResult:
    """Run or resume a Codex turn, forwarding AutoFA tools without an MCP subprocess.

    ``restricted`` removes environments (therefore shell, file editing, and local
    file viewing), for evidence-only judging. Discovery uses workspace-write.
    An optional ``connection`` retains that verified live thread for bounded
    judge repairs; its context manager closes the server after the review.
    Callbacks run inside AutoFA and see its live Runtime. Cancellation interrupts
    the active turn, cancels pending callbacks, closes the CLI, and propagates.
    """
    if connection is not None and not restricted:
        raise ValueError("Persistent Codex connections are limited to isolated evidence reviews")
    specs = _normalize_tools(tools)
    names = {tool["name"] for tool in specs}
    thread_id = resume or ""
    turn_id = ""
    messages: dict[str, dict[str, Any]] = {}
    deltas: dict[str, str] = {}
    usage: dict[str, int] = {}
    usage_baseline: dict[str, Any] | None = None
    tool_calls = 0
    last_error: dict[str, Any] | None = None
    finished = False
    accounting = CodexUsage(run_dir=cwd, role=usage_role, model=model, effort=effort)
    accounting.resumed = bool(resume)

    def result(status: str, error: dict[str, Any] | None = None) -> CodexTurnResult:
        nonlocal finished
        finished = True
        if connection is not None:
            connection._keep = status == "completed" and not error
            if turn_id:
                connection.closed_turns.add(turn_id)
        final = [item["text"] for item in messages.values() if item.get("phase") == "final_answer"]
        ordinary = [item["text"] for item in messages.values() if item.get("phase") != "commentary"]
        text = "\n".join(final or ordinary or deltas.values())
        code, message, retryable = (
            _error_details(error, model=model) if error else (None, None, False)
        )
        return CodexTurnResult(thread_id, turn_id, status, text, message, usage, code, retryable,
                               accounting.cost_usd, accounting.cost_kind)

    async def dispatch(_method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal tool_calls
        if finished or connection is not None and params.get("turnId") in connection.closed_turns:
            return _tool_failure("Tool call belongs to an earlier AutoFA turn.")
        name = params.get("tool")
        if params.get("namespace") not in (None, "functions") or name not in names:
            return _tool_failure("This tool is not available in this AutoFA session.")
        if params.get("threadId") != thread_id:
            return _tool_failure("Tool call belongs to another AutoFA session.")
        if turn_id and params.get("turnId") != turn_id:
            return _tool_failure("Tool call belongs to an earlier AutoFA turn.")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            return _tool_failure("Tool arguments must be a JSON object.")
        tool_calls += 1
        if max_tool_calls is not None and tool_calls > max_tool_calls:
            raise _ToolCallLimit()
        checkpoint = False
        if name == "advance_round":
            checkpoint = await accounting.account(client, status="observed")
        response = await call_tool(name, arguments)
        if checkpoint:
            # The account snapshot precedes the round-changing tool. Subsequent
            # native work belongs to the new round; a failed snapshot retains
            # the span so later dollars are not assigned to an arbitrary round.
            accounting.context = accounting.current_context()
        return response

    async def forward_event(method: str, params: dict[str, Any]) -> None:
        event_turn = params.get("turnId") or params.get("turn", {}).get("id")
        if connection is not None and event_turn in connection.closed_turns:
            return
        if thread_id and params.get("threadId", thread_id) != thread_id:
            return
        if turn_id and event_turn is not None and event_turn != turn_id:
            return
        accounting.observe(method, params)
        if on_event is not None:
            await on_event(method, params)

    directory = str(Path(cwd).resolve())
    transport = (
        connection.use(
            cli_path=cli_path,
            cwd=directory,
            on_request=dispatch,
            on_notification=forward_event,
            signature=json.dumps(
                [cli_path, directory, model, system_prompt, specs, effort],
                sort_keys=True,
            ),
        )
        if connection is not None
        else CodexAppServer(
            cli_path,
            cwd=directory,
            on_request=dispatch,
            on_notification=forward_event,
            isolated_state=restricted and resume is None,
        )
    )
    async with paid_async_dispatch(accounting.context), transport as client:

        async def interrupt() -> None:
            active_turn = turn_id or client.active_turns.get(thread_id)
            if thread_id and active_turn:
                with suppress(Exception):
                    async with asyncio.timeout(2.0):
                        await client.request(
                            "turn/interrupt", {"threadId": thread_id, "turnId": active_turn}
                        )

        try:
            async with asyncio.timeout(timeout_s) as deadline:
                if connection is not None and resume and resume == connection.thread_id:
                    # This live process already acknowledged isolated thread
                    # creation. Persisted resume cannot express environments.
                    thread_id = resume
                else:
                    config = await _session_config(client, cwd, restricted=restricted)
                    params: dict[str, Any] = {
                        "model": model,
                        "cwd": directory,
                        "baseInstructions": system_prompt,
                        "developerInstructions": "",
                        "approvalPolicy": "never",
                        "config": config,
                        "sandbox": "read-only" if restricted else "workspace-write",
                    }
                    if resume:
                        params.update(threadId=resume, excludeTurns=True)
                        started = await client.request("thread/resume", params)
                    else:
                        params.update(dynamicTools=specs, experimentalRawEvents=False)
                        if restricted:
                            params["environments"] = []
                        started = await client.request("thread/start", params)
                    thread_id = started.get("thread", {}).get("id", "")
                    if not isinstance(thread_id, str) or not thread_id:
                        raise CodexAppServerError({"code": "invalidResponse"})
                    if restricted and started["thread"].get("environments") != []:
                        # Never run evidence judging unless environment removal
                        # was acknowledged by this live server.
                        raise CodexAppServerError({"code": "restrictedToolsUnsupported"})
                    if connection is not None:
                        connection.thread_id = thread_id
                if on_thread is not None:
                    await on_thread(thread_id)
                accounting.thread_id = thread_id
                remaining = (None if deadline.when() is None else
                             max(0.0, deadline.when() - asyncio.get_running_loop().time()))
            # Accounting has its own bounded RPC timeout and must not consume the
            # model's deadline, including a resumed thread's initial baseline.
            if resume:
                await accounting.account(client, baseline=True)
            async with asyncio.timeout(remaining):
                turn_params: dict[str, Any] = {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}],
                }
                if effort and effort != "default":
                    turn_params["effort"] = effort
                if restricted:
                    turn_params["environments"] = []
                started_turn = await client.request("turn/start", turn_params)
                turn_id = started_turn.get("turn", {}).get("id", "")
                if not isinstance(turn_id, str) or not turn_id:
                    raise CodexAppServerError({"code": "invalidResponse"})
                while True:
                    method, params = await client.next_event()
                    if params.get("threadId", thread_id) != thread_id:
                        continue
                    if params.get("turnId", turn_id) != turn_id:
                        continue
                    if method == "item/agentMessage/delta":
                        item_id = str(params.get("itemId", ""))
                        deltas[item_id] = deltas.get(item_id, "") + params.get("delta", "")
                    elif method == "item/completed":
                        item = params.get("item", {})
                        if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                            messages[str(item.get("id", ""))] = item
                    elif method == "thread/tokenUsage/updated":
                        token_usage = params.get("tokenUsage", {})
                        total = token_usage.get("total", {})
                        if usage_baseline is None:
                            last = token_usage.get("last", {})
                            usage_baseline = {
                                key: value - last.get(key, 0) if resume else 0
                                for key, value in total.items()
                            }
                        usage = _usage_delta(total, usage_baseline)
                    elif method == "error" and params.get("willRetry") is not True:
                        last_error = params.get("error")
                    elif method == "turn/completed":
                        turn = params.get("turn", {})
                        if turn.get("id") != turn_id:
                            continue
                        for item in turn.get("items", []):
                            if item.get("type") == "agentMessage" and isinstance(
                                item.get("text"), str
                            ):
                                messages[str(item.get("id", ""))] = item
                        status = turn.get("status", "failed")
                        error = turn.get("error")
                        if status == "failed":
                            if not error or (
                                _error_details(error)[0] == "other" and last_error
                            ):
                                error = last_error or error or {"code": "other"}
                        break
            await accounting.account(client, status=status)
            return result(status, error)
        except asyncio.CancelledError:
            await interrupt()
            await accounting.account(client, status="interrupted")
            raise
        except TimeoutError:
            await interrupt()
            await accounting.account(client, status="failed")
            return result("failed", {"code": "requestTimeout"})
        except CodexAppServerError as exc:
            await interrupt()
            await accounting.account(client, status="failed")
            return result("failed", exc.error)
        except BaseException:
            await interrupt()
            await accounting.account(client, status="failed")
            raise
        finally:
            accounting.finish("complete" if finished else "incomplete")


async def compact_codex_thread(  # noqa: C901
    *,
    thread_id: str,
    cwd: str,
    cli_path: str | None = None,
    on_event: EventCallback | None = None,
    timeout_s: float = 180.0,
) -> CodexTurnResult:
    """Ask Codex to compact its existing thread and wait for confirmed success.

    A fresh app-server must first load the persisted thread via ``thread/resume``.
    ``thread/compact/start`` then launches Codex's native compaction task, which
    has no tool execution or user prompt. Its acknowledgement only means queued;
    success requires both a completed contextCompaction item and its completed
    turn. Stored model, instructions, dynamic tools, and the thread ID survive.
    """
    if not thread_id:
        raise ValueError("A Codex thread ID is required for compaction")
    turn_id = ""
    compacted = False
    submitted = False
    last_error: dict[str, Any] | None = None
    accounting = CodexUsage(run_dir=cwd, role="compaction")
    accounting.thread_id = thread_id
    accounting.resumed = True

    def result(status: str, error: dict[str, Any] | None = None) -> CodexTurnResult:
        code, message, retryable = _error_details(error) if error else (None, None, False)
        return CodexTurnResult(
            thread_id=thread_id,
            turn_id=turn_id,
            status=status,
            error=message,
            error_code=code,
            retryable=retryable,
            cost_usd=accounting.cost_usd,
            cost_kind=accounting.cost_kind,
        )

    async def forward_event(method: str, params: dict[str, Any]) -> None:
        if submitted and params.get("threadId") == thread_id:
            accounting.observe(method, params)
        if submitted and params.get("threadId") == thread_id and on_event is not None:
            await on_event(method, params)

    async with paid_async_dispatch(accounting.context), CodexAppServer(
        cli_path,
        cwd=str(Path(cwd).resolve()),
        on_notification=forward_event,
    ) as client:

        async def interrupt() -> None:
            active_turn = turn_id or client.active_turns.get(thread_id)
            if submitted and active_turn:
                with suppress(Exception):
                    async with asyncio.timeout(2.0):
                        await client.request(
                            "turn/interrupt",
                            {"threadId": thread_id, "turnId": active_turn},
                        )

        try:
            async with asyncio.timeout(timeout_s) as deadline:
                config = await _session_config(client, cwd)
                resumed = await client.request(
                    "thread/resume",
                    {
                        "threadId": thread_id,
                        "excludeTurns": True,
                        "cwd": str(Path(cwd).resolve()),
                        "config": config,
                        "approvalPolicy": "never",
                        "sandbox": "read-only",
                    },
                )
                thread = resumed.get("thread", {})
                if thread.get("id") != thread_id:
                    raise CodexAppServerError({"code": "invalidResponse"})
                if thread.get("status", {}).get("type") == "active":
                    return result("failed", {"code": "threadBusy"})
                remaining = max(0.0, deadline.when() - asyncio.get_running_loop().time())
            await accounting.account(client, baseline=True)
            async with asyncio.timeout(remaining):
                submitted = True
                await client.request("thread/compact/start", {"threadId": thread_id})
                while True:
                    method, params = await client.next_event()
                    if params.get("threadId") != thread_id:
                        continue
                    if method == "turn/started" and not turn_id:
                        turn_id = params.get("turn", {}).get("id", "")
                    if params.get("turnId", turn_id) != turn_id:
                        continue
                    if method == "item/completed":
                        if params.get("item", {}).get("type") == "contextCompaction":
                            compacted = True
                    elif method == "thread/compacted":
                        # Compatibility with older v2 implementations.
                        compacted = True
                    elif method == "error" and params.get("willRetry") is not True:
                        last_error = params.get("error")
                    elif method == "turn/completed":
                        turn = params.get("turn", {})
                        if not turn_id or turn.get("id") != turn_id:
                            continue
                        status = turn.get("status", "failed")
                        error = turn.get("error") or last_error
                        compacted = compacted or any(
                            item.get("type") == "contextCompaction"
                            for item in turn.get("items", [])
                        )
                        break
            await accounting.account(client, status=status)
            if status == "completed" and not error and compacted:
                return result("completed")
            if status == "interrupted" and not error:
                return result("interrupted")
            return result("failed", error or {"code": "compactionUnconfirmed"})
        except asyncio.CancelledError:
            await interrupt()
            await accounting.account(client, status="interrupted")
            raise
        except TimeoutError:
            await interrupt()
            await accounting.account(client, status="failed")
            return result("failed", {"code": "requestTimeout"})
        except CodexAppServerError as exc:
            await interrupt()
            await accounting.account(client, status="failed")
            return result("failed", exc.error)
        except BaseException:
            await interrupt()
            await accounting.account(client, status="failed")
            raise
        finally:
            if submitted:
                accounting.finish("complete" if compacted else "incomplete")
