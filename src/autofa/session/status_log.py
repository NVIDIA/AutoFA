# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-scoped natural-language status log for live UI + human/agent consult."""

from __future__ import annotations

import hashlib
import json
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATUS_JSONL = "status_log.jsonl"
STATUS_MD = "status.md"
STATUS_MD_TAIL = 40
CONTEXT_INLINE_BYTES = 16 * 1024
STATUS_READ_CHUNK_BYTES = 64 * 1024
MAX_STATUS_RECORD_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _OversizedRecord:
    start: int
    end: int
    newline_terminated: bool


def _reverse_lines(path: Path):
    """Scan backwards with bounded chunks and record buffers, retaining byte references."""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        record_end = position
        terminated = False
        fragments: list[bytes] = []
        buffered = 0
        oversized = False
        while position:
            size = min(position, STATUS_READ_CHUNK_BYTES)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            segment_end = len(chunk)
            while True:
                separator = chunk.rfind(b"\n", 0, segment_end)
                segment = chunk[separator + 1:segment_end]
                if not oversized:
                    buffered += len(segment)
                    if buffered > MAX_STATUS_RECORD_BYTES:
                        fragments.clear()
                        oversized = True
                    elif segment:
                        fragments.append(segment)
                if separator < 0:
                    break
                if oversized:
                    yield _OversizedRecord(position + separator + 1, record_end, terminated)
                elif fragments:
                    yield b"".join(reversed(fragments))
                fragments.clear()
                buffered = 0
                oversized = False
                record_end = position + separator
                terminated = True
                segment_end = separator
        if oversized:
            yield _OversizedRecord(0, record_end, terminated)
        elif fragments:
            yield b"".join(reversed(fragments))


def _oversized_observation(path: Path, record: _OversizedRecord) -> dict[str, Any]:
    return {
        "source": "status_reader",
        "message": ("Status record unavailable: exceeds the bounded reader limit; "
                    f"preserved at {path} bytes [{record.start}, {record.end})."),
        "context": {
            "status": "unavailable", "reason": "oversized_record", "artifact": str(path),
            "byte_start": record.start, "byte_end": record.end,
            "bytes": record.end - record.start, "limit_bytes": MAX_STATUS_RECORD_BYTES,
            "newline_terminated": record.newline_terminated,
        },
    }

# Optional Live-panel listener (harness registers AgentActivity.set_status).
_STATUS_LISTENER: Callable[[str], None] | None = None
_ACTIVITY_LISTENER: Callable[[str, dict[str, Any]], None] | None = None


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_status_listener(callback: Callable[[str], None] | None) -> None:
    """Register (or clear) the live UI callback. Prefer a bound method / weak wrapper."""
    global _STATUS_LISTENER, _ACTIVITY_LISTENER
    _STATUS_LISTENER = callback
    _ACTIVITY_LISTENER = None


def clear_status_listener() -> None:
    set_status_listener(None)


def bind_activity_listener(activity: Any) -> None:
    """Register ``activity.set_status`` without keeping a strong cycle if possible."""
    global _ACTIVITY_LISTENER
    try:
        ref = weakref.ref(activity)

        def _forward(msg: str) -> None:
            obj = ref()
            if obj is not None:
                obj.set_status(msg)

        set_status_listener(_forward)

        def _forward_activity(source: str, context: dict[str, Any]) -> None:
            obj = ref()
            if obj is not None and hasattr(obj, "set_runtime_activity"):
                obj.set_runtime_activity(source, context)

        _ACTIVITY_LISTENER = _forward_activity
    except TypeError:
        # Some objects are not weakref-able; keep a strong ref for the turn.
        set_status_listener(activity.set_status)
        _ACTIVITY_LISTENER = getattr(activity, "set_runtime_activity", None)


def _notify_status(message: str, source: str, context: dict[str, Any] | None) -> None:
    if _STATUS_LISTENER is not None:
        try:
            _STATUS_LISTENER(message)
        except Exception:  # noqa: BLE001
            log.debug("status listener failed", exc_info=True)
    if context and _ACTIVITY_LISTENER is not None:
        try:
            _ACTIVITY_LISTENER(source, context)
        except Exception:  # noqa: BLE001
            log.debug("activity listener failed", exc_info=True)


def append_status(
    run_dir: Path | None,
    message: str,
    *,
    source: str = "agent",
    context: dict[str, Any] | None = None,
    notify: bool = True,
) -> dict[str, Any]:
    """Append one status line to disk, refresh markdown, and optionally push Live UI."""
    msg = " ".join((message or "").strip().split())
    if not msg:
        raise ValueError("status message required")
    if len(msg) > 400:
        msg = msg[:397] + "…"
    row: dict[str, Any] = {
        "ts": _now_iso(),
        "source": (source or "agent").strip() or "agent",
        "message": msg,
    }
    if context:
        row["context"] = context

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if context:
            encoded = json.dumps(context, ensure_ascii=False, default=str).encode("utf-8")
            if len(encoded) > CONTEXT_INLINE_BYTES:
                from autofa.util.atomic import atomic_write_bytes

                digest = hashlib.sha256(encoded).hexdigest()
                context_path = run_dir / "status_context" / f"{digest}.json"
                context_path.parent.mkdir(exist_ok=True)
                if not context_path.exists():
                    atomic_write_bytes(context_path, encoded)
                row["context"] = {"artifact": str(context_path.relative_to(run_dir)),
                                  "sha256": digest, "bytes": len(encoded)}
        path = run_dir / STATUS_JSONL
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        write_status_markdown(run_dir)

    if notify:
        _notify_status(msg, source, context)
    return row


def read_status(run_dir: Path, *, limit: int = 30) -> list[dict[str, Any]]:
    """Return the newest ``limit`` status rows (oldest→newest within the window)."""
    path = Path(run_dir) / STATUS_JSONL
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _reverse_lines(path):
        if isinstance(line, _OversizedRecord):
            rows.append(_oversized_observation(path, line))
        else:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        if limit > 0 and len(rows) >= limit:
            break
    return list(reversed(rows))


TURN_STATUS_MESSAGES = {
    "started": "Agent turn started; no completion recorded yet.",
    "finished": "Agent turn finished normally; this does not establish exhaustive analysis.",
    "interrupted": "Agent turn interrupted; saved results may be incomplete.",
    "failed": "Agent turn failed; saved results may be incomplete.",
    "incomplete": ("Agent stream ended without confirmed normal completion; "
                   "saved results may be incomplete."),
}


def record_turn_status(run_dir: Path, status: str) -> None:
    """Observe execution without making logging a condition for running discovery."""
    try:
        context: dict[str, Any] = {"turn_status": status}
        # The durable outcome is written before this observation. Copy only
        # matching, allowlisted metadata; never expose provider exception text.
        from autofa.session.recovery import read_json

        turn = read_json(Path(run_dir) / "turn.json")
        if turn.get("state") == status and status in {"interrupted", "failed"}:
            if isinstance(turn.get("resumable"), bool):
                context["resumable"] = turn["resumable"]
            reason = turn.get("recovery_reason")
            if reason in {"user_stopped", "transport_retry_exhausted",
                          "context_recovery_exhausted", "nonretryable_error"}:
                context["recovery_reason"] = reason
        append_status(run_dir, TURN_STATUS_MESSAGES[status], source="harness",
                      context=context, notify=False)
    except Exception:  # noqa: BLE001
        log.warning("Could not persist agent turn status", exc_info=True)


def latest_turn_status(run_dir: Path) -> dict[str, Any]:
    """Report only harness-recorded outcomes, never infer completion from modes."""
    unavailable: dict[str, Any] = {}
    try:
        path = Path(run_dir) / STATUS_JSONL
        if not path.exists():
            return {"status": "unknown"}
        for line in _reverse_lines(path):
            if isinstance(line, _OversizedRecord):
                # A newer unreadable record could contain another harness outcome.
                # Keep scanning for a last-known receipt without calling it current.
                if not unavailable:
                    unavailable["latest_unavailable_record"] = _oversized_observation(path, line)
                    unavailable["unavailable_records"] = 0
                unavailable["unavailable_records"] += 1
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(row, dict) or row.get("source") != "harness":
                continue
            context = row.get("context")
            status = context.get("turn_status") if isinstance(context, dict) else None
            if isinstance(status, str) and status in TURN_STATUS_MESSAGES:
                result = {"status": status, "recorded_at": str(row.get("ts") or "")}
                if status in {"interrupted", "failed"}:
                    for key in ("resumable", "recovery_reason"):
                        if key in context:
                            result[key] = context[key]
                if unavailable:
                    return {"status": "unknown", "last_known_turn": result, **unavailable}
                return result
    except (OSError, ValueError):
        pass
    return {"status": "unknown", **unavailable}


def write_status_markdown(run_dir: Path, *, tail: int = STATUS_MD_TAIL) -> Path:
    run_dir = Path(run_dir)
    rows = read_status(run_dir, limit=tail)
    lines = [
        "# Discovery status log",
        "",
        "Natural-language progress lines from the agent and high-signal tools.",
        "",
    ]
    if not rows:
        lines.append("_No status updates yet._")
    else:
        for r in rows:
            ts = r.get("ts") or ""
            src = r.get("source") or ""
            msg = r.get("message") or ""
            lines.append(f"- `{ts}` · **{src}** — {msg}")
    path = run_dir / STATUS_MD
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def emit_runtime_status(
    message: str,
    *,
    source: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append status for the current MCP runtime run_dir (no-op if runtime missing)."""
    try:
        from autofa.mcp.runtime import get_runtime

        rt = get_runtime()
        return append_status(
            rt.run.run_dir(),
            message,
            source=source,
            context=context,
        )
    except Exception:  # noqa: BLE001
        # Still try to notify Live UI even without a run dir.
        msg = " ".join((message or "").strip().split())
        if msg:
            _notify_status(msg[:400], source, context)
        return None
