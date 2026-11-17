# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-mode ledger — track every discovered mode and detector status.

Modes live here even after their detector is removed from the working library,
so the agent can always see what was explored (active / open / rejected / redundant).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_STATUSES = frozenset({"active", "open", "rejected", "redundant"})

LEDGER_JSONL = "modes_ledger.jsonl"
LEDGER_MD = "modes_ledger.md"

CLAUDE_BEGIN = "<!-- BEGIN_FAILURE_MODES_LEDGER -->"
CLAUDE_END = "<!-- END_FAILURE_MODES_LEDGER -->"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ledger_path(run_dir: Path) -> Path:
    return Path(run_dir) / LEDGER_JSONL


def read_modes(run_dir: Path) -> list[dict[str, Any]]:
    """Return latest-wins ledger rows (one per mode name, chronological order of last write)."""
    path = ledger_path(run_dir)
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        if name not in latest:
            order.append(name)
        latest[name] = row
    return [latest[n] for n in order if n in latest]


def get_mode(run_dir: Path, name: str) -> dict[str, Any] | None:
    want = (name or "").strip()
    if not want:
        return None
    for row in read_modes(run_dir):
        if row.get("name") == want:
            return row
    return None


def upsert_mode(
    run_dir: Path,
    *,
    name: str,
    description: str = "",
    mechanism: str = "",
    status: str = "active",
    reason: str = "",
    detector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a ledger row (latest-wins by name) and refresh markdown + CLAUDE.md block."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    st = (status or "active").strip().lower()
    if st not in VALID_STATUSES:
        st = "open"
    row: dict[str, Any] = {
        "ts": _now_iso(),
        "name": (name or "").strip(),
        "description": (description or "").strip(),
        "mechanism": (mechanism or "").strip(),
        "status": st,
        "reason": (reason or "").strip(),
        "detector": detector if isinstance(detector, dict) else None,
    }
    if not row["name"]:
        raise ValueError("mode name required")
    with ledger_path(run_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    write_modes_markdown(run_dir)
    refresh_claude_modes_block(run_dir)
    return row


def summarize_modes(run_dir: Path) -> dict[str, Any]:
    """Compact summary for tool responses / prompts."""
    rows = read_modes(run_dir)
    counts = {s: 0 for s in ("active", "open", "rejected", "redundant")}
    open_modes: list[dict[str, str]] = []
    for row in rows:
        st = str(row.get("status") or "")
        if st in counts:
            counts[st] += 1
        if st == "open":
            open_modes.append(
                {
                    "name": str(row.get("name") or ""),
                    "description": str(row.get("description") or "")[:160],
                    "reason": str(row.get("reason") or "")[:160],
                }
            )
    return {
        "counts": counts,
        "n_modes": len(rows),
        "open": open_modes,
        "modes": [
            {
                "name": str(r.get("name") or ""),
                "status": str(r.get("status") or ""),
                "description": str(r.get("description") or "")[:160],
                "reason": str(r.get("reason") or "")[:120],
            }
            for r in rows
        ],
    }


def format_modes_prompt_block(
    run_dir: Path | None = None, *, summary: dict[str, Any] | None = None,
) -> str:
    """Markdown block injected into CLAUDE.md / kickoffs."""
    if summary is None:
        if run_dir is None:
            summary = {
                "counts": {"active": 0, "open": 0, "rejected": 0, "redundant": 0},
                "n_modes": 0,
                "open": [],
                "modes": [],
            }
        else:
            summary = summarize_modes(run_dir)
    counts = summary.get("counts") or {}
    lines = [
        "## Failure modes ledger (always check)",
        "Before proposing or submitting any new failure mode, call `list_failure_modes` "
        "then `list_library_functions`. Open modes are unfinished work — improve/restore "
        "them before inventing a nearby duplicate. Do not re-propose rejected/redundant claims.",
        (
            f"Counts: active={counts.get('active', 0)} · open={counts.get('open', 0)} · "
            f"rejected={counts.get('rejected', 0)} · redundant={counts.get('redundant', 0)}"
        ),
    ]
    open_modes = summary.get("open") or []
    if open_modes:
        lines.append("Open (improve these first):")
        for m in open_modes[:12]:
            desc = m.get("description") or "(no description)"
            reason = m.get("reason") or ""
            extra = f" — {reason}" if reason else ""
            lines.append(f"- `{m.get('name')}`: {desc}{extra}")
    elif not summary.get("n_modes"):
        lines.append("Ledger empty — no modes recorded yet this run.")
    else:
        lines.append("No open modes. Still call `list_failure_modes` before proposing new ones.")
    return "\n".join(lines) + "\n"


def write_modes_markdown(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    rows = read_modes(run_dir)
    lines = [
        "# Failure modes ledger",
        "",
        "Status: **active** = detector in library · **open** = mode believed real, "
        "detector weak/dropped · **rejected** = claim wrong · "
        "**redundant** = folded into another mode.",
        "",
    ]
    if not rows:
        lines.append("_No modes recorded yet._")
    else:
        lines.append("| status | name | description | reason |")
        lines.append("|--------|------|-------------|--------|")
        for r in rows:
            desc = (r.get("description") or "").replace("|", "/").replace("\n", " ")[:80]
            reason = (r.get("reason") or "").replace("|", "/").replace("\n", " ")[:60]
            lines.append(
                f"| {r.get('status')} | `{r.get('name')}` | {desc} | {reason} |"
            )
    path = run_dir / LEDGER_MD
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def refresh_claude_modes_block(run_dir: Path) -> None:
    """Refresh the live ledger in either harness's workspace instructions."""
    for path in (Path(run_dir) / "CLAUDE.md", Path(run_dir) / "AGENTS.md"):
        block = format_modes_prompt_block(run_dir)
        section = f"{CLAUDE_BEGIN}\n{block}{CLAUDE_END}\n"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if CLAUDE_BEGIN in text and CLAUDE_END in text:
            pre = text.split(CLAUDE_BEGIN, 1)[0]
            post = text.split(CLAUDE_END, 1)[1]
            path.write_text(pre + section + post.lstrip("\n"), encoding="utf-8")
            continue
        # Insert after Live config / SSoT area: before "## Protocol"
        marker = "## Protocol"
        if marker in text:
            pre, post = text.split(marker, 1)
            path.write_text(
                pre.rstrip() + "\n\n" + section + "\n" + marker + post, encoding="utf-8",
            )
        else:
            path.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")


def ledger_peers_for_novelty(run_dir: Path, *, exclude_name: str = "") -> list[dict[str, Any]]:
    """Peers for novelty_warning — all ledger modes except exclude_name."""
    out: list[dict[str, Any]] = []
    skip = (exclude_name or "").strip()
    for row in read_modes(run_dir):
        name = str(row.get("name") or "")
        if not name or name == skip:
            continue
        out.append(
            {
                "name": name,
                "description": row.get("description") or "",
                "mechanism": row.get("mechanism") or "",
                "status": row.get("status") or "",
            }
        )
    return out


def empty_claude_ledger_section() -> str:
    """Placeholder markers for a fresh CLAUDE.md."""
    return f"{CLAUDE_BEGIN}\n{format_modes_prompt_block(summary=None)}{CLAUDE_END}\n"
