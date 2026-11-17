# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude harness — uses your installed Claude Code CLI (login/subscription)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from autofa.defaults import default_value, harness_default
from autofa.harness.base import BatchSessionCancelled, Harness
from autofa.harness.claude_cli import (
    describe_auth_mode,
    ensure_claude_auth,
)
from autofa.harness.claude_transport import CLAUDE_MAX_BUFFER_SIZE
from autofa.harness.commands import handle_local_command
from autofa.harness.errors import (
    claude_auth_message,
    is_claude_auth_error,
    is_model_access_error,
    model_access_message,
)
from autofa.session.recovery import drain_discovery_tools
from autofa.ui.activity import AgentActivity
from autofa.ui.errors import show_action_error, show_action_notice
from autofa.ui.theme import GREEN

log = logging.getLogger(__name__)
console = Console()


@dataclass
class TurnResult:
    """Outcome of a single agent turn (native discovery)."""

    session_id: str | None
    text: str | None = None
    error: str | None = None
    stopped: bool = False
    retryable: bool = False
    error_code: str | None = None


def resolve_model(model: str) -> str:
    """Use the gateway's own alias when the shipped default model was kept.

    Gateways (NVIDIA agent sandboxes) scope their key to an alias set and publish the
    mapping as ANTHROPIC_DEFAULT_<TIER>_MODEL; a literal model id is rejected with 403.
    """
    if model == harness_default("claude", "model"):
        for tier in ("opus", "sonnet", "haiku"):
            alias_env = f"ANTHROPIC_DEFAULT_{tier.upper()}_MODEL"
            if model.startswith(f"claude-{tier}-") and os.environ.get(alias_env, "").strip():
                return tier
    return model


class ClaudeHarness:
    """Agent via Claude Code CLI login, or ANTHROPIC_API_KEY + bundled SDK CLI."""

    name = "claude"

    def __init__(
        self, model: str | None = None, cli_path: str | None = None, *,
        thinking: str | None = None, effort: str | None = None,
        thinking_budget_tokens: int | None = None,
    ) -> None:
        from autofa.harness.thinking import reasoning_options

        self.reasoning_options = reasoning_options(
            thinking=(harness_default(self.name, "thinking") if thinking is None else thinking),
            effort=harness_default(self.name, "effort") if effort is None else effort,
            thinking_budget_tokens=(harness_default(self.name, "thinking_budget_tokens")
                                    if thinking_budget_tokens is None else thinking_budget_tokens),
        )
        self.model = resolve_model(harness_default(self.name, "model") if model is None else model)
        self.cli_path = ensure_claude_auth(
            cli_path=(default_value("claude_cli") or None) if cli_path is None else cli_path,
        )
        console.print(f"[dim]Claude auth:[/dim] {describe_auth_mode(self.cli_path)}")

    async def run_turn(
        self,
        prompt: str,
        *,
        cwd: str,
        system_prompt: str,
        mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        resume: str | None = None,
        show_result: bool = True,
        activity_hint: str | None = None,
    ) -> TurnResult:
        """Recover transport/context failures without resetting saved discovery work."""
        command = prompt.strip().split(maxsplit=1)
        if command and command[0] == "/compact":
            return await self.compact_session(
                cwd=cwd, system_prompt=system_prompt, mcp_servers=mcp_servers,
                allowed_tools=allowed_tools, resume=resume,
                instructions=command[1] if len(command) > 1 else "", show_result=show_result,
            )
        current_prompt, session_id = prompt, resume
        directory = Path(cwd)
        from autofa.session.recovery import (
            context_recovery_prompt,
            discovery_failure_kind,
            read_json,
            save_turn,
            save_turn_outcome,
            transport_recovery_prompt,
        )
        from autofa.session.status_log import append_status

        save_turn(directory, prompt=prompt, session_id=session_id, state="running")
        attempt, compacted, replaced = 0, False, False
        while True:
            try:
                result = await self._run_turn_once(
                    current_prompt, cwd=cwd, system_prompt=system_prompt,
                    mcp_servers=mcp_servers, allowed_tools=allowed_tools,
                    resume=session_id, show_result=show_result, activity_hint=activity_hint,
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
                    message = "Claude context is full; compacting the existing conversation"
                    console.print(f"[yellow]{message}[/yellow]")
                    append_status(directory, message, source="harness",
                                  context={"session_id": session_id,
                                           "recovery": "native_compaction"})
                    try:
                        compact = await self.compact_session(
                            cwd=cwd, system_prompt=system_prompt, mcp_servers=mcp_servers,
                            allowed_tools=allowed_tools, resume=session_id, show_result=False,
                        )
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        compact = TurnResult(session_id=session_id, stopped=True)
                    if not compact.error and not compact.stopped:
                        current_prompt = transport_recovery_prompt(prompt)
                        save_turn(directory, prompt=prompt, session_id=session_id, state="running")
                        continue
                    result = compact
                    kind = discovery_failure_kind(result.error, result.error_code)
                    if result.stopped or kind not in {"context", "compaction", "transport"}:
                        save_turn_outcome(directory, prompt=prompt, result=result)
                        return result
                if not replaced:
                    # SDK query cleanup drains its in-process tools before replacement.
                    reason = "compaction_failed" if kind != "context" else "context_exhausted"
                    current_prompt = context_recovery_prompt(
                        directory, prompt=prompt, session_id=session_id, harness=self.name,
                        reason=reason,
                    )
                    append_status(directory, "Recovering saved work in a fresh Claude session",
                                  source="harness", context={"recovery": "fresh_session",
                                  "reason": reason, "previous_session_id": session_id})
                    session_id, replaced = None, True
                    save_turn(directory, prompt=prompt, session_id=None, state="running",
                              resumable=True, recovery_reason=reason)
                    continue
                save_turn_outcome(directory, prompt=prompt, result=result,
                                  reason="context_recovery_exhausted")
                return result
            if result.stopped or not result.retryable or attempt == 2:
                save_turn_outcome(directory, prompt=prompt, result=result)
                return result
            delay = 5 * (attempt + 1)
            message = f"Temporary Claude failure; retry {attempt + 1}/2 in {delay}s"
            console.print(f"[yellow]{message}[/yellow]")
            try:
                append_status(directory, message, source="harness",
                              context={"retry_attempt": attempt + 1, "error": result.error})
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
            attempt += 1

    def _agent_options(
        self, *, cwd: str, system_prompt: str, mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None, resume: str | None = None,
    ):
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            model=self.model, cwd=cwd, system_prompt=system_prompt,
            mcp_servers=mcp_servers or {}, allowed_tools=allowed_tools,
            permission_mode="acceptEdits", setting_sources=["project"], resume=resume,
            max_buffer_size=CLAUDE_MAX_BUFFER_SIZE,
            cli_path=self.cli_path, **self.reasoning_options,
        )

    @drain_discovery_tools
    async def compact_session(
        self, *, cwd: str, system_prompt: str, resume: str | None,
        mcp_servers: dict[str, Any] | None = None, allowed_tools: list[str] | None = None,
        instructions: str = "", show_result: bool = True,
    ) -> TurnResult:
        """Run Claude Code's native /compact without replacing the conversation."""
        from autofa.harness.claude_compaction import compact_claude_session
        from autofa.session.recovery import discovery_failure_kind
        from autofa.session.status_log import append_status, record_turn_status

        if not resume:
            return TurnResult(session_id=None,
                              error="Start or resume a Claude session before using /compact.")
        activity = AgentActivity(console, label="compacting Claude conversation")
        activity.start()
        record_turn_status(Path(cwd), "started")

        async def on_message(message):
            activity.on_message(message)

        try:
            async with asyncio.timeout(180):
                await compact_claude_session(
                    options=self._agent_options(
                        cwd=cwd, system_prompt=system_prompt, mcp_servers=mcp_servers,
                        allowed_tools=allowed_tools, resume=resume,
                    ),
                    session_id=resume, instructions=instructions, on_message=on_message,
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            activity.stop(final="stopped")
            record_turn_status(Path(cwd), "interrupted")
            return TurnResult(session_id=resume, stopped=True)
        except Exception as exc:
            activity.stop(final="error")
            record_turn_status(Path(cwd), "failed")
            message = (
                f"Claude native compaction failed: {str(exc) or type(exc).__name__}. "
                "The existing session and saved work were retained. Retry /compact, "
                "or reduce the request or evidence page size before continuing."
            )
            return TurnResult(session_id=resume, error=message,
                              error_code=getattr(exc, "code", None),
                              retryable=discovery_failure_kind(str(exc)) == "transport")
        activity.stop(final="compacted")
        _save_session_id(Path(cwd), resume)
        append_status(Path(cwd), "Claude conversation compacted", source="harness",
                      context={"session_id": resume, "compaction": "native"})
        record_turn_status(Path(cwd), "finished")
        text = "Claude conversation compacted; the same session is ready to continue."
        if show_result:
            console.print(f"[dim]{text}[/dim]")
        return TurnResult(session_id=resume, text=text)

    @drain_discovery_tools
    async def _run_turn_once(  # noqa: C901
        self,
        prompt: str,
        *,
        cwd: str,
        system_prompt: str,
        mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        resume: str | None = None,
        show_result: bool = True,
        activity_hint: str | None = None,
    ) -> TurnResult:
        """One SDK attempt, shared by interactive and batch discovery."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            query,
        )

        session_id = resume
        from autofa.harness.usage import track_claude_messages

        options = self._agent_options(
            cwd=cwd, system_prompt=system_prompt, mcp_servers=mcp_servers,
            allowed_tools=allowed_tools, resume=session_id,
        )

        activity = AgentActivity(console, label="Claude Code running")
        from autofa.estimates import LiveRuntimeEstimate

        activity.runtime_hint = LiveRuntimeEstimate(Path(cwd))
        activity.runtime_elapsed = activity.runtime_hint.elapsed_seconds
        # Prompts describe possible future actions as well as the present task.
        # Only explicit caller hints and live tool events establish actual activity.
        activity.hint = activity_hint or "Working… (tool calls shown below)"
        activity.start()
        result_text: str | None = None
        result_err: str | None = None
        assistant_error: str | None = None
        from autofa.session.status_log import (
            bind_activity_listener,
            clear_status_listener,
            record_turn_status,
        )

        bind_activity_listener(activity)
        record_turn_status(Path(cwd), "started")
        turn_status = "incomplete"
        work_seen = False
        try:
            try:
                async for message in track_claude_messages(
                    query(prompt=prompt, options=options), run_dir=cwd,
                    model=self.model, effort=self.reasoning_options.get("effort"), resume=resume,
                ):
                    activity.on_message(message)
                    if (isinstance(message, SystemMessage)
                            and getattr(message, "subtype", None) == "init"):
                        data = getattr(message, "data", None) or {}
                        if isinstance(data, dict) and data.get("session_id"):
                            session_id = data["session_id"]
                            _save_session_id(Path(cwd), session_id)
                        activity.set_phase("connected to Claude Code")
                    elif not isinstance(message, ResultMessage):
                        work_seen = True
                    if isinstance(message, AssistantMessage) and message.error:
                        assistant_error = str(message.error) + " " + " ".join(
                            block.text for block in message.content if isinstance(block, TextBlock)
                        )
                    if isinstance(message, ResultMessage):
                        if getattr(message, "session_id", None):
                            session_id = message.session_id
                            _save_session_id(Path(cwd), session_id)
                        terminal_reason = getattr(message, "terminal_reason", None)
                        stop_reason = getattr(message, "stop_reason", None)
                        turn_status = "finished" if (
                            getattr(message, "subtype", None) == "success"
                            and terminal_reason in {None, "completed"}
                            and stop_reason in {
                                None, "end_turn", "stop_sequence",
                            }
                        ) else "incomplete"
                        result = getattr(message, "result", None)
                        if result:
                            result_text = str(result)
                            activity.set_phase("composing reply")
                        context_exceeded = "model_context_window_exceeded" in {
                            terminal_reason, stop_reason,
                        }
                        if getattr(message, "is_error", False) or context_exceeded:
                            result_err = _format_result_error(message)
                            provider_error = " ".join(filter(None, (result_err, assistant_error)))
                            if is_claude_auth_error(provider_error):
                                result_err = claude_auth_message(self.cli_path)
                                result_text = None
                            elif is_model_access_error(provider_error):
                                result_err = model_access_message("Claude Code", self.model)
                                # Provider errors are not an assistant answer.
                                result_text = None
                            if context_exceeded:
                                result_err = f"model_context_window_exceeded: {result_err}"
                            turn_status = "failed"
                        if terminal_reason in {"aborted_streaming", "aborted_tools"}:
                            turn_status = "interrupted"
            except (KeyboardInterrupt, asyncio.CancelledError):
                turn_status = "interrupted"
                activity.stop(final="stopped")
                return TurnResult(session_id=session_id, text=result_text, stopped=True)
            except Exception as e:  # noqa: BLE001
                if turn_status == "interrupted":
                    activity.stop(final="stopped")
                    return TurnResult(session_id=session_id, text=result_text, stopped=True)
                if turn_status == "finished":
                    # A CLI teardown failure cannot undo a confirmed completed
                    # turn. Retrying here could repeat already-finished work.
                    activity.stop(final="done")
                    log.warning("Claude transport failed after confirmed completion", exc_info=True)
                    if show_result and result_text:
                        console.print(
                            Panel(Markdown(result_text), title="agent", border_style=GREEN)
                        )
                    return TurnResult(session_id=session_id, text=result_text)
                turn_status = "failed"
                activity.stop(final="error")
                msg = _humanize_agent_error(e, result_err=result_err, result_text=result_text)
                provider_error = " ".join(filter(None, (msg, assistant_error)))
                if is_claude_auth_error(provider_error):
                    msg = claude_auth_message(self.cli_path)
                    result_text = None
                elif is_model_access_error(provider_error):
                    msg = model_access_message("Claude Code", self.model)
                    result_text = None
                console.print(f"[red]Agent error:[/red] {msg}")
                if result_text and result_text.strip() and result_text.strip() not in msg:
                    console.print(
                        Panel(Markdown(result_text), title="agent (partial)", border_style="yellow")
                    )
                log.exception("session error")
                return TurnResult(
                    session_id=session_id, text=result_text, error=msg,
                    retryable=_is_transient_agent_error(msg) and bool(session_id or not work_seen),
                )

            if (
                turn_status not in {"finished", "interrupted"}
                and not result_err and assistant_error
            ):
                if is_claude_auth_error(assistant_error):
                    result_err = claude_auth_message(self.cli_path)
                elif is_model_access_error(assistant_error):
                    result_err = model_access_message("Claude Code", self.model)
                if result_err:
                    result_text = None
                    turn_status = "failed"
            activity.stop(final="stopped" if turn_status == "interrupted" else
                          "error" if result_err else "done")
            if turn_status == "interrupted":
                return TurnResult(session_id=session_id, text=result_text, stopped=True)
            if result_err:
                console.print(f"[yellow]Turn ended with error:[/yellow] {result_err}")
                if show_result and result_text and result_text.strip():
                    console.print(
                        Panel(Markdown(result_text), title="agent", border_style="yellow")
                    )
                return TurnResult(
                    session_id=session_id, text=result_text, error=result_err,
                    retryable=_is_transient_agent_error(result_err)
                    and bool(session_id or not work_seen),
                )

            if turn_status != "finished":
                return TurnResult(
                    session_id=session_id, text=result_text,
                    error="Claude turn ended without confirmed normal completion",
                )

            if show_result and result_text:
                console.print(Panel(Markdown(result_text), title="agent", border_style=GREEN))
            return TurnResult(session_id=session_id, text=result_text)
        finally:
            # Campaign admission signals intentionally bypass Exception handlers.
            # Always release Rich Live even on those non-error control boundaries.
            activity.stop(final="stopped" if turn_status != "finished" else "done")
            record_turn_status(Path(cwd), turn_status)
            clear_status_listener()

    async def run_session(
        self,
        *,
        cwd: str,
        system_prompt: str,
        initial_prompt: str | None = None,
        mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
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

        history_file = autofa_home() / "prompt_history"
        session = None
        if not exit_after_discovery:
            session = PromptSession(
                history=FileHistory(str(history_file)),
                completer=SlashCommandCompleter(),
                complete_while_typing=True,
                complete_style=CompleteStyle.MULTI_COLUMN,
            )
        session_id: str | None = resume

        auth = describe_auth_mode(self.cli_path)
        # Leave wizard / dataset-setup chrome in scrollback before the REPL.
        # AgentActivity.start() clears again immediately before Live so the
        # panel is never painted under a tall stack (Rich overflow dump).
        try:
            from autofa.ui.menus import prepare_step

            prepare_step()
        except Exception:  # noqa: BLE001
            pass
        console.print(
            Panel(
                f"[bold {GREEN}]AutoFA[/] session  ·  harness=claude\n"
                f"[dim]auth={auth}  model={self.model}[/dim]\n"
                "[dim]Commands: /help /library /goal /dashboard /clean[/dim]\n"
                "[dim]/save saves + publishes the report; "
                "/exit or /quit also ends the session[/dim]\n"
                "[dim]Agent: /discover /validate  ·  "
                "[bold]Ctrl+C[/bold] stops a running turn[/dim]",
                border_style=GREEN,
            )
        )

        first = initial_prompt
        discovery_active = bool(initial_prompt)
        while True:
            if first:
                user_text = first
                first = None
                console.print(f"[dim]auto-starting:[/dim] {user_text.splitlines()[0][:80]}…")
            elif exit_after_discovery:
                break
            else:
                try:
                    user_text = await _async_prompt(session)
                    # Explicit discovery opts into automatic follow-up turns.
                    # Ordinary questions and local commands never start a quota loop.
                    discovery_active = user_text.strip().split(maxsplit=1)[:1] == ["/discover"]
                except EOFError:
                    show_action_notice("Claude session ended because terminal input closed "
                                       "(EOF/Ctrl+D). Resume the saved run to continue.",
                                       console=console)
                    break
                except KeyboardInterrupt:
                    show_action_notice("Claude session interrupted by Ctrl+C. "
                                       "Resume the saved run to continue.", console=console)
                    break
            if not user_text.strip():
                continue
            if user_text.strip().split(maxsplit=1)[0] == "/login":
                discovery_active = False
                show_action_notice(claude_auth_message(self.cli_path), console=console)
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
                user_text,
                cwd=cwd,
                system_prompt=system_prompt,
                mcp_servers=mcp_servers,
                allowed_tools=allowed_tools,
                resume=session_id,
                show_result=True,
            )
            session_id = result.session_id
            finalize = discovery_active and result.stopped and not result.error
            if result.error:
                discovery_active = False
                show_action_error(RuntimeError(result.error), console=console, wait=False,
                                  context="Claude discovery turn")
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
    """Keep Resume usable even if a process dies before the interactive session ends."""
    target = run_dir / "session.json"
    try:
        from autofa.util.atomic import atomic_write_json

        atomic_write_json(target, {"session_id": session_id})
    except OSError:
        log.warning("Could not persist Claude session ID", exc_info=True)


def _is_transient_agent_error(message: str) -> bool:
    from autofa.session.recovery import discovery_failure_kind

    return discovery_failure_kind(message) in {"transport", "compaction"}


def _format_result_error(message: Any) -> str:
    """Build a readable error from a ResultMessage (SDK often hides this)."""
    parts: list[str] = []
    status = getattr(message, "api_error_status", None)
    if status is not None:
        parts.append(f"HTTP {status}")
    errors = getattr(message, "errors", None) or []
    if errors:
        parts.append("; ".join(str(e) for e in errors))
    result = getattr(message, "result", None)
    if result:
        parts.append(str(result).strip())
    subtype = getattr(message, "subtype", None)
    if subtype and subtype not in {"success", "error"} and not parts:
        parts.append(str(subtype))
    return " — ".join(parts) if parts else "Claude Code reported is_error without details"


def _humanize_agent_error(
    exc: BaseException,
    *,
    result_err: str | None = None,
    result_text: str | None = None,
) -> str:
    """Replace known SDK nonsense like 'error result: success' with the real cause."""
    raw = str(exc).strip()
    if result_err:
        return result_err
    # Known SDK bug: empty errors[] → falls back to subtype "success".
    if "error result: success" in raw.lower():
        if result_text and result_text.strip():
            return result_text.strip()
        return (
            "Claude Code ended the turn with is_error=True (often an API refusal, "
            "rate limit, or Usage Policy block). Check the last agent message / "
            "try a different benchmark subset (e.g. avoid CBRNE-style content)."
        )
    if result_text and "usage policy" in result_text.lower():
        return result_text.strip()
    return raw


async def _async_prompt(session) -> str:
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: session.prompt("AutoFA › "))


def get_harness(
    name: str, model: str, cli_path: str | None = None, *, thinking: str | None = None,
    effort: str | None = None, thinking_budget_tokens: int | None = None, cfg: Any = None,
) -> Harness:
    from autofa.harness.settings import HARNESS_NAMES, harness_cli, harness_model

    if name not in HARNESS_NAMES:
        raise ValueError(f"unknown harness: {name}")
    model = harness_model(name, model, cfg=cfg)
    cli_path = harness_cli(cfg, name) if cli_path is None else cli_path
    thinking = harness_default(name, "thinking", cfg) if thinking is None else thinking
    effort = harness_default(name, "effort", cfg) if effort is None else effort
    thinking_budget_tokens = (harness_default(name, "thinking_budget_tokens", cfg)
                              if thinking_budget_tokens is None else thinking_budget_tokens)
    if name == "claude":
        return ClaudeHarness(
            model=model, cli_path=cli_path, thinking=thinking, effort=effort,
            thinking_budget_tokens=thinking_budget_tokens,
        )
    if name == "codex":
        from autofa.harness.codex import CodexHarness

        return CodexHarness(model=model, cli_path=cli_path, effort=effort)
    raise ValueError(f"unknown harness: {name}")
