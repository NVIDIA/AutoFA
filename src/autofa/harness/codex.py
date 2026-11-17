# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoFA discovery through the installed Codex app server and its login."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from autofa.harness.base import BatchSessionCancelled
from autofa.harness.claude import TurnResult
from autofa.harness.commands import handle_local_command
from autofa.harness.model_catalog import _find_codex_cli
from autofa.mcp.server import codex_tool_response, codex_tool_specs, invoke_tool
from autofa.session.recovery import drain_discovery_tools
from autofa.ui.activity import AgentActivity
from autofa.ui.errors import show_action_error, show_action_notice
from autofa.ui.theme import GREEN

log = logging.getLogger(__name__)
console = Console()


class CodexHarness:
    """Keep AutoFA tools in process while Codex owns its persistent conversation."""

    name = "codex"

    def __init__(
        self, model: str | None = None, cli_path: str | None = None, *,
        thinking: str | None = None, effort: str | None = None,
        thinking_budget_tokens: int | None = None,
    ) -> None:
        from autofa.defaults import default_value, harness_default

        if model is None:
            model = harness_default("codex", "model")
        if thinking is None:
            thinking = harness_default("codex", "thinking")
        if effort is None:
            effort = harness_default("codex", "effort")
        if cli_path is None:
            cli_path = default_value("codex_cli") or None
        if thinking != "default":
            raise ValueError("Codex controls thinking with Effort; set thinking='default'.")
        if not isinstance(effort, str) or not effort.strip():
            raise ValueError("Codex effort must be a nonempty string or 'default'.")
        self.model = model
        self.cli_path = _find_codex_cli(cli_path)
        self.effort = None if effort == "default" else effort

    def _restore_discovery_model(self, directory: Path, session_id: str | None) -> None:
        """Keep a run's capacity fallback without changing its requested settings."""
        self._active_model, self._active_effort = self.model, self.effort
        self._requested_resolved_model = self._active_resolved_model = self.model
        if not session_id:
            return
        from autofa.harness.model_catalog import HarnessModel, ModelCatalog
        from autofa.harness.model_fallback import lower_codex_models

        try:
            saved = json.loads((directory / "codex_model.json").read_text())
        except (OSError, ValueError):
            return
        if (not isinstance(saved, dict) or saved.get("version") != 1
                or saved.get("requested_model") != self.model
                or saved.get("requested_effort") != self.effort
                or saved.get("session_id") != session_id):
            return
        model, effort = saved.get("active_model"), saved.get("active_effort")
        if (not isinstance(model, str) or not model
                or (effort is not None and (not isinstance(effort, str) or not effort))):
            return
        # A saved file cannot introduce an unrelated model or upgrade the request.
        requested_id = saved.get("requested_resolved_model", self.model)
        active_id = saved.get("active_resolved_model", model)
        if not isinstance(requested_id, str) or not isinstance(active_id, str):
            return
        choices = ModelCatalog(models=(
            HarnessModel(self.model, self.model, resolved_model=requested_id),
            HarnessModel(model, model, resolved_model=active_id),
        ))
        if not lower_codex_models(choices, self.model):
            return
        self._active_model, self._active_effort = model, effort
        self._requested_resolved_model, self._active_resolved_model = requested_id, active_id
        from autofa.estimates import mark_model_fallback

        mark_model_fallback(directory)

    def _save_discovery_model(self, directory: Path, session_id: str | None) -> None:
        from datetime import UTC, datetime

        from autofa.util.atomic import atomic_write_json

        atomic_write_json(directory / "codex_model.json", {
            "version": 1, "requested_model": self.model, "requested_effort": self.effort,
            "active_model": self._active_model, "active_effort": self._active_effort,
            "requested_resolved_model": self._requested_resolved_model,
            "active_resolved_model": self._active_resolved_model,
            "session_id": session_id, "reason": "serverOverloaded",
            "updated_at": datetime.now(UTC).isoformat(),
        })
        from autofa.estimates import mark_model_fallback

        mark_model_fallback(directory)

    async def _fallback_discovery_model(self, directory: Path, session_id: str | None) -> bool:
        from autofa.harness.model_catalog import load_model_catalog_async
        from autofa.harness.model_fallback import effort_for_model, lower_codex_models
        from autofa.session.status_log import append_status

        try:
            catalog = await load_model_catalog_async(self.cli_path, harness="codex")
        except (OSError, ValueError, RuntimeError):
            log.warning("Could not load Codex capacity fallback models", exc_info=True)
            return False
        choices = lower_codex_models(catalog, self._active_model)
        if not choices:
            message = "No supported lower Codex model is available for automatic fallback."
            if catalog.error:
                message += " Could not load the model catalog."
            console.print(message, style="yellow", markup=False)
            return False
        previous, previous_effort = self._active_model, self._active_effort
        previous_resolved = self._active_resolved_model
        candidate = choices[0]
        if self._active_model == self.model:
            original = catalog.find(self.model)
            self._requested_resolved_model = original.resolved_model or original.value
        self._active_model = candidate.value
        self._active_effort = effort_for_model(candidate, self.effort)
        self._active_resolved_model = candidate.resolved_model or candidate.value
        try:
            self._save_discovery_model(directory, session_id)
        except OSError:
            self._active_model, self._active_effort = previous, previous_effort
            self._active_resolved_model = previous_resolved
            log.warning("Could not checkpoint Codex model fallback", exc_info=True)
            console.print("Could not save the Codex model switch; retaining the current model.",
                          style="yellow")
            return False
        message = (f"Codex model at capacity: switching discovery from {previous} to "
                   f"{self._active_model}. Continuing the same conversation and saved work.")
        if self._active_effort != previous_effort:
            message += f" Reasoning effort: {self._active_effort or 'model default'}."
        console.print(message, style="yellow", markup=False)
        try:
            append_status(directory, message, source="harness", context={
                "recovery": "model_fallback", "error_code": "serverOverloaded",
                "requested_model": self.model, "previous_model": previous,
                "active_model": self._active_model, "active_effort": self._active_effort,
                "session_id": session_id,
            })
        except OSError:
            log.warning("Could not append Codex model fallback status", exc_info=True)
        return True

    async def run_turn(
        self, prompt: str, *, cwd: str, system_prompt: str,
        mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None, resume: str | None = None,
        show_result: bool = True, activity_hint: str | None = None,
    ) -> TurnResult:
        """Resume transient failures and recover exhausted context from saved work."""
        from autofa.session.recovery import (
            context_recovery_prompt,
            discovery_failure_kind,
            read_json,
            save_turn,
            save_turn_outcome,
            transport_recovery_prompt,
        )
        from autofa.session.status_log import append_status

        command = prompt.strip().split(maxsplit=1)
        if command and command[0].lower() == "/compact":
            return await self.compact_session(
                cwd=cwd, system_prompt=system_prompt, mcp_servers=mcp_servers,
                allowed_tools=allowed_tools, resume=resume,
                instructions=command[1] if len(command) > 1 else "", show_result=show_result,
            )
        current_prompt, session_id = prompt, resume
        directory = Path(cwd)
        self._restore_discovery_model(directory, session_id)
        if self._active_model != self.model:
            console.print(f"Continuing Codex discovery with {self._active_model} "
                          f"(capacity fallback from {self.model}).", style="yellow", markup=False)
        save_turn(directory, prompt=prompt, session_id=session_id, state="running")
        attempt, compacted, replaced, switches, fallback_unavailable = 0, False, False, 0, False
        while True:
            try:
                result = await self._run_turn_once(
                    current_prompt, cwd=cwd, system_prompt=system_prompt, mcp_servers=mcp_servers,
                    allowed_tools=allowed_tools, resume=session_id, show_result=show_result,
                    activity_hint=activity_hint,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                saved_session = read_json(directory / "session.json").get("session_id")
                result = TurnResult(session_id=saved_session or session_id, stopped=True)
            session_id = result.session_id or session_id
            result.session_id = session_id
            kind = discovery_failure_kind(result.error, result.error_code)
            if kind == "terminal":
                result.retryable = False
            if not result.stopped and kind in {"context", "compaction"}:
                if kind == "context" and session_id and not compacted and not replaced:
                    compacted = True
                    try:
                        append_status(
                            directory, "Codex context is full; compacting the conversation",
                            source="harness", context={"recovery": "native_compaction"},
                        )
                    except OSError:
                        log.warning("Could not persist compaction status", exc_info=True)
                    try:
                        compact = await self.compact_session(
                            cwd=cwd, system_prompt=system_prompt, mcp_servers=mcp_servers,
                            allowed_tools=allowed_tools, resume=session_id, show_result=show_result,
                        )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        compact = TurnResult(session_id=session_id, stopped=True)
                    if not compact.error and not compact.stopped:
                        current_prompt = transport_recovery_prompt(prompt)
                        continue
                    result = compact
                    kind = discovery_failure_kind(result.error, result.error_code)
                    if result.stopped or kind not in {"context", "compaction", "transport"}:
                        save_turn_outcome(directory, prompt=prompt, result=result)
                        return result
                if not replaced:
                    # The transport has drained cancelled tools before replacement.
                    reason = "compaction_failed" if kind != "context" else "context_exhausted"
                    current_prompt = context_recovery_prompt(
                        directory, prompt=prompt, session_id=session_id, harness=self.name,
                        reason=reason,
                    )
                    append_status(directory, "Recovering saved work in a fresh Codex session",
                                  source="harness", context={"recovery": "fresh_session",
                                  "reason": reason, "previous_session_id": session_id})
                    session_id, replaced = None, True
                    save_turn(directory, prompt=prompt, session_id=None, state="running",
                              resumable=True, recovery_reason=reason)
                    continue
                save_turn_outcome(directory, prompt=prompt, result=result,
                                  reason="context_recovery_exhausted")
                return result
            switched = False
            if (not result.stopped and result.retryable
                    and result.error_code == "serverOverloaded" and attempt >= 1
                    and switches < 2 and not fallback_unavailable):
                try:
                    switched = await self._fallback_discovery_model(directory, session_id)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    stopped = TurnResult(session_id=session_id, text=result.text, stopped=True)
                    save_turn_outcome(directory, prompt=prompt, result=stopped)
                    return stopped
                if switched:
                    attempt, compacted, switches = 0, False, switches + 1
                else:
                    fallback_unavailable = True
            if result.stopped or not result.retryable or (attempt == 2 and not switched):
                save_turn_outcome(directory, prompt=prompt, result=result)
                return result
            if not switched:
                delay = 5 * (attempt + 1)
                message = f"Temporary Codex failure; retry {attempt + 1}/2 in {delay}s"
                console.print(f"[yellow]{message}[/yellow]")
                try:
                    append_status(directory, message, source="harness", context={
                        "retry_attempt": attempt + 1, "error": result.error,
                        "active_model": self._active_model,
                    })
                except OSError:
                    log.warning("Could not persist retry status", exc_info=True)
                try:
                    await asyncio.sleep(delay)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    stopped = TurnResult(session_id=session_id, text=result.text, stopped=True)
                    save_turn_outcome(directory, prompt=prompt, result=stopped)
                    return stopped
            if session_id:
                current_prompt = transport_recovery_prompt(prompt)
            if not switched:
                attempt += 1

    @drain_discovery_tools
    async def compact_session(
        self, *, cwd: str, system_prompt: str,
        mcp_servers: dict[str, Any] | None = None, allowed_tools: list[str] | None = None,
        resume: str | None = None, instructions: str = "", show_result: bool = True,
    ) -> TurnResult:
        """Compact natively without resetting the conversation or saved turn intent."""
        if not resume:
            return TurnResult(session_id=None,
                              error="No Codex conversation to compact yet. Start a turn first.")
        if instructions.strip():
            return TurnResult(session_id=resume,
                              error="Codex supports /compact without additional instructions.")
        from autofa.harness.codex_app_server import compact_codex_thread
        from autofa.session.recovery import discovery_failure_kind
        from autofa.session.status_log import append_status

        if show_result:
            console.print("[dim]Compacting Codex conversation…[/dim]")
        result = None
        try:
            result = await compact_codex_thread(thread_id=resume, cwd=cwd, cli_path=self.cli_path)
            if result.status == "interrupted":
                return TurnResult(session_id=resume, stopped=True)
            if result.error or result.status != "completed":
                raise RuntimeError(result.error or "Codex compaction did not complete.")
        except (KeyboardInterrupt, asyncio.CancelledError):
            return TurnResult(session_id=resume, stopped=True)
        except Exception as exc:
            code = getattr(result, "error_code", None) or getattr(exc, "code", None)
            if result is not None and not result.error and result.status != "completed":
                code = "compactionUnconfirmed"
            kind = discovery_failure_kind(str(exc), code)
            return TurnResult(
                session_id=resume,
                error=f"Codex compaction failed: {exc}. The existing conversation and saved "
                      "work are retained. Retry /compact before continuing.",
                error_code=code,
                retryable=kind != "terminal" and (
                    getattr(result, "retryable", False) or kind in {"transport", "compaction"}
                ),
            )
        _save_session_id(Path(cwd), resume)
        try:
            append_status(Path(cwd), "Codex compacted the existing conversation", source="harness",
                          context={"recovery": "native_compaction", "session_id": resume})
        except OSError:
            log.warning("Could not persist compaction status", exc_info=True)
        if show_result:
            console.print("[dim]Codex conversation compacted; the same session is retained.[/dim]")
        return TurnResult(session_id=resume, text="Codex conversation compacted.")

    @drain_discovery_tools
    async def _run_turn_once(
        self, prompt: str, *, cwd: str, system_prompt: str,
        mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None, resume: str | None = None,
        show_result: bool = True, activity_hint: str | None = None,
    ) -> TurnResult:
        from autofa.harness.codex_app_server import run_codex_turn
        from autofa.session.status_log import (
            bind_activity_listener,
            clear_status_listener,
            record_turn_status,
        )
        from autofa.util.transient import transient_message

        directory = Path(cwd)
        session_id = resume
        # The SDK server object is not passed across process boundaries. Its same
        # registered functions execute here, with the active dataset/library runtime.
        specs = codex_tool_specs(allowed_tools) if (mcp_servers or {}).get("autofa") else []
        permitted = {spec["name"] for spec in specs}
        model = getattr(self, "_active_model", self.model)
        effort = getattr(self, "_active_effort", self.effort)
        activity = AgentActivity(console, label=f"Codex running · {model}")
        from autofa.estimates import LiveRuntimeEstimate

        activity.runtime_hint = LiveRuntimeEstimate(Path(cwd))
        activity.runtime_elapsed = activity.runtime_hint.elapsed_seconds
        if activity_hint:
            activity.hint = activity_hint
        activity.start()
        bind_activity_listener(activity)
        record_turn_status(directory, "started")
        status = "incomplete"
        display_text: str | None = None
        display_error: str | None = None
        text_parts: dict[str, str] = {}
        work_seen = False

        async def on_thread(thread_id: str) -> None:
            nonlocal session_id
            session_id = thread_id
            _save_session_id(directory, thread_id)
            if model != self.model:
                self._save_discovery_model(directory, thread_id)
            activity.set_phase("connected to Codex")

        async def on_event(method: str, params: dict[str, Any]) -> None:
            nonlocal work_seen
            activity.on_codex_event(method, params)
            if method.startswith("item/"):
                work_seen = True
            if method == "item/agentMessage/delta":
                item_id = str(params.get("itemId") or "message")
                text_parts[item_id] = text_parts.get(item_id, "") + str(params.get("delta") or "")
            elif method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage":
                    text_parts[str(item.get("id") or "message")] = str(item.get("text") or "")

        async def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            if name not in permitted:
                return {"success": False, "contentItems": [{
                    "type": "inputText", "text": f"AutoFA tool is not allowed: {name}",
                }]}
            activity.update_tool_arguments(name, args)
            return codex_tool_response(await invoke_tool(name, args))

        try:
            try:
                result = await run_codex_turn(
                    prompt=prompt, system_prompt=system_prompt, model=model, cwd=cwd,
                    tools=specs, call_tool=call_tool, effort=effort, cli_path=self.cli_path,
                    resume=resume, on_event=on_event, on_thread=on_thread,
                )
                session_id = result.thread_id or session_id
                if session_id:
                    _save_session_id(directory, session_id)
                text = result.text or "\n\n".join(text_parts.values()) or None
                display_text = text
                if result.status == "interrupted":
                    status = "interrupted"
                    return TurnResult(session_id=session_id, text=text, stopped=True)
                error = result.error
                if result.status != "completed" and not error:
                    error = "Codex turn ended without confirmed normal completion"
                if error:
                    status = "failed"
                    display_error = error
                    return TurnResult(
                        session_id=session_id, text=text, error=error,
                        retryable=result.retryable and bool(session_id or not work_seen),
                        error_code=getattr(result, "error_code", None),
                    )
                status = "finished"
                return TurnResult(session_id=session_id, text=text)
            except (KeyboardInterrupt, asyncio.CancelledError):
                status = "interrupted"
                return TurnResult(
                    session_id=session_id, text="\n\n".join(text_parts.values()) or None,
                    stopped=True,
                )
            except Exception as exc:
                status = "failed"
                message = str(exc).strip() or type(exc).__name__
                display_error = message
                log.exception("Codex session error")
                return TurnResult(
                    session_id=session_id, text="\n\n".join(text_parts.values()) or None,
                    error=message,
                    retryable=(getattr(exc, "retryable", False) or transient_message(message))
                    and bool(session_id or not work_seen),
                    error_code=getattr(exc, "code", None),
                )
        finally:
            activity.stop(final="stopped" if status == "interrupted" else
                          "done" if status == "finished" else "error")
            record_turn_status(directory, status)
            clear_status_listener()
            if display_error:
                console.print(f"[yellow]Codex turn ended with error:[/yellow] {display_error}")
            if show_result and display_text and status != "interrupted":
                console.print(Panel(Markdown(display_text), title="agent",
                                    border_style=GREEN if status == "finished" else "yellow"))

    async def run_session(
        self, *, cwd: str, system_prompt: str, initial_prompt: str | None = None,
        mcp_servers: dict[str, Any] | None = None, allowed_tools: list[str] | None = None,
        resume: str | None = None,
        discovery_continuation: Callable[[], str | None] | None = None,
        discovery_finalization: Callable[[], Awaitable[str | None]] | None = None,
        turn_validation: Callable[[], Awaitable[str | None]] | None = None,
        exit_after_discovery: bool = False,
    ) -> str | None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.shortcuts import CompleteStyle

        from autofa.config import autofa_home
        from autofa.harness.commands import SlashCommandCompleter
        from autofa.ui.menus import prepare_step

        session = None
        if not exit_after_discovery:
            session = PromptSession(
                history=FileHistory(str(autofa_home() / "prompt_history")),
                completer=SlashCommandCompleter(), complete_while_typing=True,
                complete_style=CompleteStyle.MULTI_COLUMN,
            )
        session_id, first = resume, initial_prompt
        discovery_active = bool(initial_prompt)
        self._restore_discovery_model(Path(cwd), resume)
        prepare_step()
        console.print(Panel(
            f"[bold {GREEN}]AutoFA[/] session  ·  harness=codex\n"
            f"[dim]auth=Codex login  model={self._active_model}[/dim]\n"
            "[dim]Commands: /help /library /goal /dashboard /clean[/dim]\n"
            "[dim]/save saves + publishes the report; "
            "/exit or /quit also ends the session[/dim]\n"
            "[dim]Agent: /discover /validate  ·  "
            "[bold]Ctrl+C[/bold] stops a running turn[/dim]",
            border_style=GREEN,
        ))
        while True:
            if first:
                user_text, first = first, None
                console.print(f"[dim]auto-starting:[/dim] {user_text.splitlines()[0][:80]}…")
            elif exit_after_discovery:
                break
            else:
                try:
                    user_text = await _async_prompt(session)
                    discovery_active = user_text.strip().split(maxsplit=1)[:1] == ["/discover"]
                except EOFError:
                    show_action_notice(
                        "Codex session ended because terminal input closed (EOF/Ctrl+D). "
                        "Resume the saved run to continue.", console=console,
                    )
                    break
                except KeyboardInterrupt:
                    show_action_notice(
                        "Codex session interrupted by Ctrl+C. Resume the saved run to continue.",
                        console=console,
                    )
                    break
            if not user_text.strip():
                continue
            local = handle_local_command(user_text, console)
            if local == "exit":
                break
            if local == "handled":
                continue
            from autofa.session.supervision import check_lifecycle, steering_prompt

            check_lifecycle(Path(cwd))
            user_text += steering_prompt(Path(cwd), user_text=user_text)
            result = await self.run_turn(
                user_text, cwd=cwd, system_prompt=system_prompt, mcp_servers=mcp_servers,
                allowed_tools=allowed_tools, resume=session_id, show_result=True,
            )
            session_id = result.session_id
            finalize = discovery_active and result.stopped and not result.error
            if result.error:
                discovery_active = False
                show_action_error(RuntimeError(result.error), console=console, wait=False,
                                  context="Codex discovery turn")
                if exit_after_discovery:
                    raise RuntimeError(result.error)
            if result.stopped:
                discovery_active = False
                console.print(
                    f"\n[{GREEN}]Stopped.[/{GREEN}] "
                    "[dim]Ask another question, /clean, /dashboard, or /discover again.[/dim]"
                )
                if exit_after_discovery:
                    raise BatchSessionCancelled("Batch discovery was cancelled")
            automatic_finalization = False
            if not result.error and not result.stopped and turn_validation is not None:
                feedback = await turn_validation()
                if feedback:
                    # Give changed measurements back to the same conversation
                    # before asking the runtime whether discovery can finish.
                    first = feedback
                    continue
            if discovery_active and discovery_continuation is not None:
                first = discovery_continuation()
                discovery_active = bool(first)
                finalize = not discovery_active
                automatic_finalization = finalize
            if finalize and discovery_finalization is not None:
                continuation = await discovery_finalization()
                # Validation can disqualify provisional successes. Resume in the
                # same conversation only after an automatic completion decision;
                # final scoring after a user stop must never restart discovery.
                if automatic_finalization and continuation and continuation.strip():
                    first = continuation
                    discovery_active = True
            if exit_after_discovery and not first:
                break
        return session_id


def _save_session_id(run_dir: Path, session_id: str | None) -> None:
    from autofa.util.atomic import atomic_write_json

    try:
        atomic_write_json(run_dir / "session.json", {
            "session_id": session_id, "harness": "codex",
        })
    except OSError:
        log.warning("Could not persist Codex thread ID", exc_info=True)


async def _async_prompt(session: Any) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: session.prompt("AutoFA › "))
