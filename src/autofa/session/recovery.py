# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable turn and tool intent for resuming every phase of an analysis run."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

from autofa.util.atomic import atomic_write_json

# Read-only browsing is already recorded in the harness transcript. Persist work
# that can be expensive or change state before dispatching it to a worker.
CHECKPOINT_TOOLS = {
    "evaluate_functions", "judge_function", "screen_function", "submit_function", "remove_function",
    "clear_library", "restore_failure_mode", "advance_round", "library_load",
    "library_save", "llm_classify", "plan_function_repair", "defer_function_repair",
    "restore_function_version",
    "submit_helper", "retain_as_helper", "deprecate_helper", "smoke_test_feature",
    "classifier_profiles",
    "library_import_candidates", "record_external_inspection", "record_source_audit",
}


class RunInterruptedError(RuntimeError):
    """A temporary service failure interrupted work, not a detector judgment."""


_DISCOVERY_WORKERS: ContextVar[set[tuple[asyncio.Task, Any]] | None] = ContextVar(
    "autofa_discovery_workers", default=None,
)


def track_discovery_worker(worker: asyncio.Task, cancellation: Any) -> None:
    """Retain the real worker even if an SDK abandons its cancelled MCP callback."""
    workers = _DISCOVERY_WORKERS.get()
    if workers is not None:
        entry = (worker, cancellation)
        workers.add(entry)
        worker.add_done_callback(lambda _: workers.discard(entry))


def drain_discovery_tools(fn: Callable) -> Callable:
    """End an attempt only after its detached SDK workers can no longer mutate state."""
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        workers: set[tuple[asyncio.Task, Any]] = set()
        token = _DISCOVERY_WORKERS.set(workers)

        async def drain() -> None:
            while workers:
                pending = tuple(workers)
                # Never cancel the to_thread task itself: that loses the handle
                # while its synchronous operation continues in the background.
                await asyncio.gather(*(asyncio.to_thread(c.cancel) for _, c in pending))
                await asyncio.gather(*(w for w, _ in pending), return_exceptions=True)
                workers.difference_update(pending)

        try:
            return await fn(*args, **kwargs)
        finally:
            _DISCOVERY_WORKERS.reset(token)
            closing = asyncio.create_task(drain())
            interrupted = False
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    interrupted = True
            closing.result()
            if interrupted:
                raise asyncio.CancelledError

    return wrapped


@contextmanager
def run_lock(run_dir: Path):
    """Prevent two sessions overwriting one run; OS releases the lock after a crash."""
    path = Path(run_dir) / ".run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"Run {Path(run_dir).name} is already active in another session. "
                "Use that session or stop it before resuming this run."
            ) from exc
        try:
            if (Path(run_dir) / "config.json").is_file():
                from autofa.session.supervision import prune_dead_operations

                try:
                    prune_dead_operations(Path(run_dir))
                except OSError:
                    # Observation cleanup must not prevent recovery of saved work.
                    pass
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def exclusive_run(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped(run, *args, **kwargs):
        from autofa.session.supervision import check_lifecycle

        check_lifecycle(run.run_dir())
        with run_lock(run.run_dir()):
            return fn(run, *args, **kwargs)

    return wrapped


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_turn(
    run_dir: Path, *, prompt: str, session_id: str | None, state: str,
    resumable: bool | None = None, recovery_reason: str | None = None,
) -> None:
    previous = read_json(Path(run_dir) / "turn.json")
    if prompt.startswith("Resume the interrupted request from the saved conversation"):
        prompt = previous.get("prompt") or prompt
    atomic_write_json(Path(run_dir) / "turn.json", {
        "prompt": prompt, "session_id": session_id, "state": state,
        "resumable": resumable if resumable is not None else state == "interrupted",
        "recovery_reason": recovery_reason,
        "updated_at": datetime.now(UTC).isoformat(),
    })


def discovery_failure_kind(message: str | None, code: str | None = None) -> str:
    """Shared recovery policy; terminal provider failures override transport wording."""
    from autofa.harness.errors import is_claude_auth_error, is_model_access_error
    from autofa.util.transient import transient_message

    text = (message or "").lower()
    if code in {
        "unauthorized", "forbidden", "modelAccessDenied", "contentPolicyViolation",
        "nonretryableError",
        "usageLimitExceeded", "sessionBudgetExceeded", "badRequest", "sandboxError",
        "toolCallLimitExceeded", "restrictedToolsUnsupported", "-32600", "-32601", "-32602",
    } or is_claude_auth_error(text) or is_model_access_error(text) or any(
        marker in text for marker in (
            "content policy", "content_policy", "contentpolicyviolation", "usage policy",
            "safety policy", "authentication", "login required",
            "unauthorized", "forbidden", "permission denied", "insufficient_quota",
            "credit balance", "usage limit", "quota exceeded", "invalid request",
            "invalid_request", "bad request", "http 400", "http 401", "http 403",
            "http 404", "http 422", "unknown model",
        )
    ):
        return "terminal"
    if code in {"remoteCompactionFailed", "compactionUnconfirmed"} or any(
        marker in text for marker in (
            "error running remote compact task", "remote compaction failed",
        )
    ):
        return "compaction"
    if context_limit_error(message) or code == "contextWindowExceeded":
        return "context"
    if transient_message(message or "") or any(marker in text for marker in (
        "transport error", "network error", "error decoding response body", "service unavailable",
    )):
        return "transport"
    return "other"


def save_turn_outcome(
    run_dir: Path, *, prompt: str, result: Any, reason: str | None = None,
) -> None:
    """Persist a resumable interruption distinctly from a permanent provider failure."""
    from autofa.session.status_log import record_turn_status

    kind = discovery_failure_kind(result.error, result.error_code)
    resumable = bool(result.stopped or (result.error and (
        reason == "context_recovery_exhausted" or result.retryable
    ) and kind != "terminal"))
    state = "interrupted" if resumable else "failed" if result.error else "finished"
    if reason is None:
        reason = ("user_stopped" if result.stopped else
                  "transport_retry_exhausted" if resumable else
                  "nonretryable_error" if result.error else None)
    save_turn(run_dir, prompt=prompt, session_id=result.session_id, state=state,
              resumable=resumable, recovery_reason=reason)
    status = ("incomplete" if result.error and "without confirmed normal completion" in result.error
              and not resumable else state)
    record_turn_status(run_dir, status)


def transport_recovery_prompt(prompt: str) -> str:
    return (
        "The previous response was interrupted by a temporary connection or service failure. "
        "Continue the same request from the saved conversation and current working state. "
        "Check the latest tool results, operations/, round_validation.json, existing library "
        "and failure-mode ledger first; do not repeat completed work or duplicate "
        "writes/evaluations. Resume interrupted validation from saved predictions and judge "
        "checkpoints. Preserve the original scope, round target and counters; a retry is not "
        "a new discovery round. Do not advance the round just for recovery.\n\n"
        f"Original request for this turn:\n{prompt}"
    )


def context_recovery_prompt(
    run_dir: Path, *, prompt: str, session_id: str | None, harness: str = "claude",
    reason: str = "context_exhausted",
) -> str:
    """Checkpoint a full conversation, but hand a fresh session bounded working context."""
    from autofa.mcp.tool_results import save_tool_result

    directory = Path(run_dir) / "context_recovery" / uuid4().hex
    config = read_json(Path(run_dir) / "config.json")
    handoff = {
        "previous_session_id": session_id,
        "round": config.get("round"),
        "max_rounds": config.get("max_rounds"),
        "min_failure_modes": config.get("min_failure_modes"),
        "reason": reason,
        "saved_state": {name: str(Path(run_dir) / name) for name in (
            "config.json", "library.json", "operations", "round_validation.json", "behavior_judge",
        )},
        "request": prompt,
        "prior_requests": [],
        "last_assistant_text": "",
    }
    if session_id and harness == "claude":
        try:
            from claude_agent_sdk import get_session_messages

            messages = get_session_messages(session_id, directory=str(run_dir))
            # Keep the whole history accessible, including tool calls/results, without
            # sending it all back to the model that just rejected its size.
            if messages:
                handoff["conversation"] = save_tool_result(
                    [{"type": m.type, "message": m.message} for m in messages],
                    run_dir=Path(run_dir),
                )
            for message in messages:
                content = message.message.get("content", [])
                text = content if isinstance(content, str) else "\n".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
                if not text.strip():
                    continue
                if message.type == "user":
                    handoff["prior_requests"].append(text)
                elif message.type == "assistant" and not context_limit_error(text):
                    handoff["last_assistant_text"] = text
        except (ImportError, OSError, ValueError, AttributeError) as exc:
            handoff["transcript_read_error"] = str(exc)
    elif session_id:
        handoff["harness"] = harness
        handoff["transcript_note"] = (
            "The prior conversation is retained by Codex. Use the saved request, tool "
            "operation records and current failure-mode ledger to recover working state."
        )
    # Full requests and conversation are retained, even if their previews exceed
    # the handoff budget. Nothing in the library, scores or repair state is reset.
    atomic_write_json(directory / "handoff.json", handoff)
    atomic_write_json(Path(run_dir) / "context_recovery.json", {
        "handoff_path": str(directory / "handoff.json"),
        "previous_session_id": session_id,
        "reason": reason,
    })
    preview = json.dumps(handoff, ensure_ascii=True)
    if len(preview) > 16_000:
        preview = (
            f"Handoff exceeds the inline budget. Read {directory / 'handoff.json'} "
            "in bounded pages to recover the complete requests and conversation reference."
        )
    return (
        "Continue the same AutoFA run in a fresh conversation because the previous "
        "conversation exhausted its context or its compaction was interrupted. "
        "This is recovery, not a new discovery "
        "round or experiment. Preserve the original scope, constraints and budgets.\n"
        "Call list_library_functions, list_helpers, list_failure_modes and pareto_status "
        "first. Use get_function_repair for unresolved candidates. Inspect operations/ "
        "and behavior_judge/*/progress.json for interrupted work before repeating a write "
        "or expensive evaluation; completed checkpoints remain valid. Do not count missing "
        "evidence or processing errors as negative labels. Do not repeat completed work.\n"
        "Inspect round_validation.json; resume interrupted validation using saved predictions "
        "and judgment checkpoints. Preserve the stored round, round target and counters; "
        "do not advance the round just for recovery. A cancelled operation can have completed "
        "writes: its receipt and current library establish which work remains.\n"
        "The complete recovery record is saved at " + str(directory / "handoff.json") + ". "
        "If needed, page its conversation reference with read_tool_result; do not load "
        "the entire history or library JSON into context.\n\n"
        + preview
    )


def context_limit_error(message: str | None) -> bool:
    """Only provider context exhaustion, not quota/auth/general request failures."""
    text = (message or "").lower()
    return any(marker in text for marker in (
        "prompt is too long", "prompt too long", "context length exceeded",
        "context_length_exceeded", "model_context_window_exceeded",
        "exceeds the context window", "maximum context length",
        "codex context is full",
    ))


def execute_operation(fn: Callable, name: str, args: dict[str, Any]) -> Any:
    from autofa.session.operation_admission import admit_operation

    # Busy calls never start work or create unfinished recovery records. The
    # reservation covers nested evaluation/judging performed by submit_function.
    with admit_operation(name, args) as busy:
        if busy is not None:
            return busy
        return _execute_admitted_operation(fn, name, args)


def _execute_admitted_operation(fn: Callable, name: str, args: dict[str, Any]) -> Any:
    from autofa.mcp.runtime import get_runtime
    from autofa.util.cancellation import check_cancelled

    if (name not in CHECKPOINT_TOOLS
            or (name == "classifier_profiles" and args.get("action", "status") == "status")):
        check_cancelled()
        result = fn(**args)
        return result
    rt = get_runtime()
    path = rt.run.run_dir() / "operations" / f"{uuid4().hex}.json"
    operation = {
        "tool": name, "args": args, "state": "running",
        "started_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(path, operation)
    from autofa.session.supervision import observe_operation

    observe_operation(rt.run.run_dir(), path.stem, operation)
    try:
        check_cancelled()
        result = fn(**args)
    except BaseException as exc:
        operation.update(
            state="cancelled" if isinstance(exc, asyncio.CancelledError) else "interrupted",
            error=str(exc) or type(exc).__name__,
        )
        atomic_write_json(path, operation)
        observe_operation(rt.run.run_dir(), path.stem, operation)
        raise
    # Returning from a tool establishes its outcome, even when cancellation
    # arrives after a library mutation was committed. This is an audit receipt,
    # not a new mutation: retain it so recovery cannot mistake completed work
    # for an operation that must be repeated.
    operation.update(state="complete" if result.get("ok", True) else "failed", result=result)
    try:
        check_cancelled()
    except asyncio.CancelledError:
        operation["cancelled_after_completion"] = True
    finally:
        atomic_write_json(path, operation)
        observe_operation(rt.run.run_dir(), path.stem, operation)
    return result


def resume_prompt(run_dir: Path, functions: list[Any], *, fallback: str, round_idx: int) -> str:
    from autofa.metrics.judge_checkpoints import pending_judgments

    turn = read_json(Path(run_dir) / "turn.json")
    recovery = read_json(Path(run_dir) / "context_recovery.json")
    pending = pending_judgments(run_dir, functions)
    latest = {}
    for path in (Path(run_dir) / "operations").glob("*.json"):
        operation = read_json(path)
        if not operation.get("tool"):
            continue
        key = json.dumps([operation["tool"], operation.get("args")], sort_keys=True)
        if key not in latest or operation["started_at"] > latest[key][1]["started_at"]:
            latest[key] = (path, operation)
    unfinished = [str(p) for p, row in latest.values()
                  if row["state"] != "complete" and row["tool"] != "respond_to_steering"]
    original = turn.get("prompt") if (
        turn.get("state") != "finished" or pending or unfinished
    ) else None
    if not original and not pending and not unfinished:
        return fallback
    # A recovery prompt itself is not persisted as the user's original request.
    return (
        "Resume the interrupted request from the saved conversation and working library. "
        f"The stored round is {round_idx}; do not reset or advance it just for recovery. "
        "For regular discovery, check pareto_status first: if goal_met is true, return the summary "
        "and let the driver run final scoring without restarting pending discovery work. "
        "Honor saved supervisor lifecycle intent. "
        "Otherwise read list_library_functions and list_failure_modes. Inspect saved tool results "
        "before repeating operations: a submitted detector may already be saved even when "
        "its evaluation was interrupted. Do not repeat completed mutations or publish twice. "
        "Finish interrupted evaluation/judging before new discovery; unchanged completed "
        "detector evaluations and judge stages are reused automatically. Connection failures "
        "are not negative labels or evidence against a detector.\n"
        f"Unfinished judge batches: {json.dumps(pending)}\n"
        f"Unfinished tool records (read to recover arguments): {json.dumps(unfinished)}\n\n"
        f"Prior context recovery (read the handoff if conversation was reset): "
        f"{json.dumps(recovery)}\n\n"
        f"Original request:\n{original or fallback}"
    )
