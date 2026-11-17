# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep action failures visible and debuggable before returning to a menu."""

from __future__ import annotations

import json
import sys
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rich.console import Console

from autofa.config import autofa_home

_action: ContextVar[dict | None] = ContextVar("ui_action", default=None)
_notice: tuple[str, str] | None = None


def notice_fragments() -> list[tuple[str, str]]:
    """Keep the last outcome in the next menu, even if acknowledgement was skipped."""
    if _notice is None:
        return []
    style, message = _notice
    return [(style, f"  Last action: {message}\n")]


def _record(event: str, *, console: Console, **values) -> None:
    context = _action.get()
    if context is None:
        return
    try:
        with (autofa_home() / "cli-actions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        **context,
                        **values,
                        "event": event,
                        "time": datetime.now(UTC).isoformat(),
                    },
                    default=str,
                )
                + "\n"
            )
    except OSError as exc:
        console.print(f"Could not save action history: {exc}", style="yellow", markup=False)


def begin_action(name: str, *, console: Console):
    global _notice
    _notice = None
    token = _action.set(
        {
            "action_id": uuid4().hex,
            "action": name,
            "stage": "starting",
            "source_path": str(Path(__file__).resolve().parents[1]),
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
        }
    )
    _record("started", console=console)
    return token


def end_action(token, *, console: Console, state: str) -> None:
    _record("finished", console=console, state=state)
    _action.reset(token)


def set_action_stage(stage: str, **details) -> None:
    """Persist progress before an operation, including setup before a run exists."""
    context = _action.get()
    if context is not None:
        _action.set({**context, **details, "stage": stage})
        _record("stage", console=Console())


def show_action_notice(message: str, *, console: Console, warning: bool = True) -> None:
    global _notice
    # A subsequent "saved"/"closed" message must not replace the failure cause.
    if _notice is None or (_notice[0] != "class:bad" and (warning or _notice[0] == "class:dim")):
        _notice = ("class:ok" if warning else "class:dim", message)
    console.print(message, style="yellow" if warning else "dim", markup=False)
    _record("notice", console=console, message=message)


def cancel_action(message: str = "Cancelled by the user; no analysis was started.") -> None:
    context = _action.get()
    if context is not None:
        _action.set({**context, "outcome": "cancelled"})
    show_action_notice(message, console=Console(), warning=False)


def action_outcome() -> str:
    if _notice and _notice[0] == "class:bad":
        return "failed"
    return (_action.get() or {}).get("outcome", "completed")


def _error_text(error: BaseException) -> str:
    if isinstance(error, SystemExit) and isinstance(error.code, int):
        if error.__cause__:
            return str(error.__cause__) or type(error.__cause__).__name__
        return f"Action exited with status {error.code}."
    return str(error) or type(error).__name__


def show_action_error(
    error: BaseException,
    *,
    console: Console,
    wait: bool = True,
    context: str = "",
) -> None:
    """Save the exception chain even when no analysis run has been created yet."""
    global _notice
    action = _action.get() or {}
    if action:
        _action.set({**action, "outcome": "failed"})
    location = context or " / ".join(str(action[k]) for k in ("action", "stage") if action.get(k))
    message = f"{location}: {_error_text(error)}" if location else _error_text(error)
    _notice = ("class:bad", message)
    console.print(f"Error: {message}", style="red", markup=False)
    _record("error", console=console, error=message)
    try:
        log_path = autofa_home() / "cli-errors.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{datetime.now(UTC).isoformat()}\n")
            handle.write(json.dumps(action, default=str) + "\n")
            traceback.print_exception(error, file=handle)
    except OSError as exc:
        console.print(f"Could not save error details: {exc}", style="yellow", markup=False)
        console.print("".join(traceback.format_exception(error)), markup=False)
    else:
        console.print(f"Error details: {log_path}", style="dim", markup=False)
        _notice = ("class:bad", f"{message}\nError details: {log_path}")

    next_step = "Correct the issue above and retry the action. Full details are in the error log."
    if any(word in message.lower() for word in ("network", "resolve", "nodename", "timed out")):
        next_step = (
            "Check VPN/network access to the API, then retry. Cached downloads are reusable."
        )
    console.print(f"Next step: {next_step}", markup=False)

    if wait and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            input("Press Enter to return to the menu...")
        except (EOFError, KeyboardInterrupt):
            pass
