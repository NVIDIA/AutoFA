# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live status UI while a discovery agent is working."""

from __future__ import annotations

import json
import re
import time
from collections import deque
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from autofa.ui.theme import GREEN

# Light Claude-Code-ish idle verbs when no NL status yet.
_IDLE_VERBS = (
    "Thinking",
    "Reviewing results",
    "Planning next step",
    "Sifting traces",
    "Cross-checking evidence",
    "Choosing next action",
)

_STATUS_MAX = 400  # match status_log; Live wraps instead of hard-clipping at 120
_STATUS_SPINNER_MAX = 100  # spinner line stays short
_RECENT_N = 3
_STATUS_WRAP = 96  # wrap width for the status row(s)
_DOTS_INTERVAL_S = 0.45
# Constant display width so Live line count does not flicker as dots animate.
_DOTS_FRAMES = (".  ", ".. ", "...")
_TRAILING_DOTS_RE = re.compile(r"(?:\s*(?:\.{1,3}|…))+$")

# Spinner copy for MCP / builtin tools — keep short, present progressive.
_TOOL_STATUS: dict[str, str] = {
    "get_trace_api": "Reading trace helper contracts…",
    "get_trace_record": "Reading execution records and metadata…",
    "download_artifact": "Downloading and verifying an artifact…",
    "read_artifact": "Reading artifact contents…",
    "view_artifact": "Inspecting an artifact page…",
    "prepare_artifact_support": "Preparing artifact readers…",
    "transcribe_artifact": "Transcribing an audio window locally…",
    "dataset_info": "Reading dataset overview…",
    "family_discovery": "Comparing causal families and search coverage…",
    "describe_item_fields": "Inspecting field sizes…",
    "dataset_chunks": "Listing corpus chunks…",
    "peek_chunk": "Peeking a data chunk…",
    "sample_items": "Sampling traces…",
    "search_items": "Searching the corpus…",
    "get_item": "Opening a rollout…",
    "get_field_meta": "Checking field metadata…",
    "read_field_chunk": "Reading a large field in chunks…",
    "corpus_trace_stats": "Surveying fail-vs-pass corpus stats…",
    "get_trajectory": "Loading an agent trajectory…",
    "list_trajectory_tools": "Listing tools used in a trajectory…",
    "get_trajectory_step": "Reading a trajectory step…",
    "search_trajectory": "Searching the full trajectory…",
    "get_trajectory_context": "Reading events around a trace passage…",
    "smoke_test_feature": "Smoke-testing a detector…",
    "classifier_profiles": "Checking classifier profiles…",
    "evaluate_functions": "Scoring detectors on the corpus…",
    "judge_function": "Preparing detector review…",
    "llm_classify": "Prototyping an LLM classify rubric…",
    "submit_function": "Submitting a detector…",
    "remove_function": "Dropping a detector from the library…",
    "clear_library": "Clearing the working library…",
    "list_library_functions": "Listing library detectors…",
    "get_library_function": "Inspecting a library detector…",
    "get_function_feedback": "Reading detector feedback…",
    "list_function_versions": "Inspecting saved detector versions…",
    "restore_function_version": "Restoring a saved detector version…",
    "list_failure_modes": "Checking the failure-mode ledger…",
    "get_failure_mode": "Inspecting a ledger failure mode…",
    "restore_failure_mode": "Restoring a mode from the ledger…",
    "report_status": "Updating status…",
    "list_status_updates": "Reading status history…",
    "library_save": "Saving the library…",
    "library_load": "Loading a library…",
    "pareto_status": "Checking Pareto / frontier progress…",
    "advance_round": "Advancing to the next discovery round…",
    # Non-MCP Claude Code tools the agent sometimes uses
    "Read": "Reading a file…",
    "Write": "Writing a file…",
    "Edit": "Editing a file…",
    "Glob": "Finding files…",
    "Grep": "Searching files…",
    "Bash": "Running a shell command…",
    "AskUserQuestion": "Asking you a question…",
}

_ACTIVITY_OPERATIONS = {
    "polish": "Polishing {name}",
    "evaluate": "Scoring {name} detector",
    "review_adherence": "Reviewing adherence for {name}",
    "diagnostic": "Reviewing diagnostic examples for {name}",
    "prepare_review": "Preparing adherence review for {name}",
    "prepare_diagnostic": "Preparing diagnostic review for {name}",
    "submit": "Submitting {name} detector",
    "repair": "Updating {name} detector",
    "smoke_test": "Testing {name} detector",
    "plan_repair": "Planning a repair for {name}",
    "gather_evidence": "Inspecting evidence for {name}",
    "inspect_versions": "Inspecting saved versions of {name}",
    "restore_version": "Restoring saved version of {name}",
}
_ACTIVITY_TOOL_OPERATIONS = {
    "evaluate_functions": "evaluate",
    "judge_function": "prepare_review",
    "submit_function": "submit",
    "smoke_test_feature": "smoke_test",
    "plan_function_repair": "plan_repair",
    "get_function_feedback": "gather_evidence",
    "get_function_repair": "gather_evidence",
    "list_function_versions": "inspect_versions",
    "restore_function_version": "restore_version",
}
_GOAL_LABELS = {
    "ppv_at_pi0": "precision", "behavior_purity": "adherence",
    "recall": "coverage", "matched_failed_tasks": "task support",
}


def _safe_goal(value: Any) -> str:
    """Show known dimensions, never an arbitrary rationale or model prompt."""
    if not isinstance(value, str):
        return ""
    aliases = {
        "precision": r"\bprecision\b",
        "adherence": r"\b(?:adherence|purity)\b",
        "coverage": r"\b(?:coverage|recall|breadth)\b",
        "task support": r"\b(?:task support|distinct.tasks)\b",
        "implementation": r"\b(?:implementation|execution|parsing)\b",
    }
    return " + ".join(label for label, pattern in aliases.items()
                      if re.search(pattern, value, re.IGNORECASE))[:60]


def _rationale_goal(value: Any) -> str:
    """Infer only an explicit improvement target, not every mentioned trade-off."""
    if not isinstance(value, str):
        return ""
    dimensions = re.findall(
        r"\b(?:improve|increase|raise|repair|fix|expand|target)\s+(?:the\s+)?"
        r"(?:(?:detector|normalized|observed)\s+){0,2}"
        r"(precision|adherence|purity|coverage|recall|breadth|task support|"
        r"distinct tasks|implementation)\b",
        value, re.IGNORECASE,
    )
    return _safe_goal(" + ".join(dimensions))


def detector_activity_context(
    name: str, operation: str, *, scores: dict | None = None, goal: str | None = None,
    target_grade: str | None = None, finished: bool = False,
) -> dict[str, Any]:
    """Attach a truthful operation and current measured goal to a runtime status event."""
    activity: dict[str, Any] = {"operation": operation, "finished": finished}
    if not finished:
        if isinstance(scores, dict) and (goal is None or target_grade is not None):
            from autofa.metrics.discovery_grades import grade_gap_for_scores

            try:
                gap = grade_gap_for_scores(scores, target_grade=target_grade)
                if target_grade is not None and gap.get("target_met"):
                    goal = None
                elif goal is None:
                    goal = " + ".join(_GOAL_LABELS[key] for key in gap.get("bottlenecks", [])
                                      if key in _GOAL_LABELS)
                if target_grade is not None and goal:
                    activity["target_grade"] = gap.get("target_grade")
            except (TypeError, ValueError, KeyError, OverflowError):
                # Display advice must never stop the actual detector operation.
                goal = None
        if safe_goal := _safe_goal(goal):
            activity["goal"] = safe_goal
    return {"name": name, "activity": activity}


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    if not isinstance(value, dict):
        return {}
    # Never retain source code, classifier prompts, or evidence in live labels.
    arguments = {key: value[key] for key in (
        "name", "names", "review_mode", "repair_plan_id", "action", "rationale",
    ) if key in value}
    if "rationale" in arguments:
        arguments["rationale"] = _rationale_goal(arguments["rationale"])
    return arguments


def _detector_names(arguments: dict[str, Any]) -> list[str]:
    raw = arguments.get("names") or [arguments.get("name")]
    if not isinstance(raw, list):
        return []
    return [value for value in raw if isinstance(value, str) and value.strip()]


def _detector_label(names: list[str]) -> str:
    labels = [_truncate_status(re.sub(r"[^\w .-]", " ", name), 42) for name in names[:2]]
    return ", ".join(labels) + (f" +{len(names) - 2}" if len(names) > 2 else "")


def _tool_short_name(name: str) -> str:
    """Strip mcp__autofa__ / mcp__ prefix to the bare tool id."""
    n = (name or "").strip()
    if n.startswith("mcp__autofa__"):
        return n[len("mcp__autofa__") :]
    if n.startswith("mcp__"):
        parts = n.split("__", 2)
        return parts[-1] if parts else n
    return n


def friendly_tool_status(name: str, arguments: dict[str, Any] | None = None) -> str:
    """Natural-language spinner line for a tool call."""
    short = _tool_short_name(name)
    args = arguments or {}
    names = _detector_names(args)
    if names and (operation := _ACTIVITY_TOOL_OPERATIONS.get(short)):
        if short == "judge_function" and args.get("review_mode") == "diagnostic":
            operation = "prepare_diagnostic"
        elif short == "submit_function" and args.get("repair_plan_id"):
            operation = "repair"
        status = _ACTIVITY_OPERATIONS[operation].format(name=_detector_label(names))
        if goal := _safe_goal(args.get("rationale")):
            status += f" · goal: {goal}"
        return status + "…"
    if short in _TOOL_STATUS:
        return _TOOL_STATUS[short]
    # Fallback: humanize snake_case without the mcp noise.
    label = short.replace("_", " ").strip() or "tool"
    return f"Running {label}…"


def _strip_trailing_dots(msg: str) -> str:
    """Remove static trailing dots/ellipsis so we can animate them."""
    return _TRAILING_DOTS_RE.sub("", (msg or "").rstrip()).rstrip()


def _animated_dots(now: float, *, t0: float) -> str:
    """Cycle `.` → `..` → `...` while the agent is running."""
    frame = int(max(0.0, now - t0) / _DOTS_INTERVAL_S) % len(_DOTS_FRAMES)
    return _DOTS_FRAMES[frame]


def _with_animated_dots(msg: str, *, now: float, t0: float) -> str:
    base = _strip_trailing_dots(msg)
    if not base:
        base = "Working"
    return f"{base}{_animated_dots(now, t0=t0)}"


def _truncate_status(msg: str, n: int = _STATUS_MAX) -> str:
    s = " ".join((msg or "").strip().split())
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _wrap_status_lines(msg: str, *, width: int = _STATUS_WRAP, max_lines: int = 4) -> list[str]:
    """Word-wrap a status message for the Live panel (no silent mid-word clip)."""
    import textwrap

    s = " ".join((msg or "").strip().split())
    if not s:
        return []
    lines = textwrap.wrap(s, width=max(32, width), break_long_words=True, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        # mark continuation on last line
        last = lines[-1]
        if len(last) >= width:
            lines[-1] = last[: max(0, width - 1)] + "…"
        else:
            lines[-1] = last + "…"
    return lines


def _first_short_sentence(text: str) -> str | None:
    """Pull a one-liner from assistant prose for soft Live status (no disk log)."""
    raw = " ".join((text or "").strip().split())
    if not raw or len(raw) < 12:
        return None
    # Prefer first sentence / line; skip markdown headings and huge blobs.
    chunk = raw.split("\n", 1)[0].strip()
    if chunk.startswith("#") or chunk.startswith("```"):
        return None
    m = re.match(r"^(.+?[.!?])(?:\s|$)", chunk)
    line = (m.group(1) if m else chunk).strip()
    if len(line) < 12 or len(line) > 160:
        return None
    # Skip pure tool-looking noise.
    if line.lower().startswith(("mcp__", "tool ")):
        return None
    return line


class AgentActivity:
    """Tracks and renders Claude Code–like running state."""

    def __init__(self, console: Console, *, label: str = "Working") -> None:
        self.console = console
        self.label = label
        self.t0 = time.monotonic()
        self.phase = "starting"
        self.last_tool = ""
        self.active_tools: dict[str, str] = {}
        self._active_tool_arguments: dict[str, dict[str, Any]] = {}
        self._active_tool_actions: dict[str, dict[str, Any]] = {}
        self.n_tools = 0
        self.n_messages = 0
        self.hint = "Ctrl+C stops this turn — then you can ask something else"
        self.runtime_hint = None
        self.runtime_elapsed = None
        self.status_line = ""
        self.status_recent: deque[str] = deque(maxlen=_RECENT_N)
        self._idle_idx = 0
        self._idle_at = 0.0
        self._anonymous_tool_seq = 0
        self._spinner = Spinner("dots", text="", style=GREEN)
        self._live: Live | None = None
        self._live_done = False
        self._live_final: str | None = None
        self._snap_cache: dict[str, Any] | None = None
        self._snap_at = 0.0
        self._snap_ttl = 2.0  # seconds — avoid hammering library.json / rich

    def start(self) -> None:
        self.t0 = time.monotonic()
        self._idle_at = self.t0
        self._live_done = False
        self._live_final = None
        # Drop wizard / setup chrome before Live. If the panel is painted under a
        # tall scroll stack, Rich's in-place refresh overflows the terminal and
        # reprints / corrupts the previous UI into the viewport.
        try:
            from autofa.ui.menus import prepare_step

            prepare_step()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._live = Live(
                console=self.console,
                refresh_per_second=4,
                transient=True,
                # Keep Live from scrolling the buffer (which desyncs cursor-up clears).
                vertical_overflow="ellipsis",
                # Don't capture SDK / subprocess stdout into the Live proxy.
                redirect_stdout=False,
                redirect_stderr=False,
                get_renderable=self._live_renderable,
            )
            self._live.start()
        except Exception:  # noqa: BLE001
            self._live = None

    def stop(self, *, final: str | None = None) -> None:
        if self._live is not None:
            try:
                self._live_done = True
                self._live_final = final
                self._live.refresh()
                self._live.stop()
            except Exception:  # noqa: BLE001
                try:
                    self._live.stop()
                except Exception:  # noqa: BLE001
                    pass
            self._live = None
        elapsed = time.monotonic() - self.t0
        total, cumulative = self._elapsed()
        found = self._library_snapshot(force=True)
        extra = f" · {found['n']} features" if found["n"] else ""
        timing = (f"Turn finished in {elapsed:.0f}s · "
                  f"discovery elapsed {self._duration(total)} active"
                  if cumulative else f"Finished in {elapsed:.0f}s")
        self.console.print(
            f"[dim]{timing} · {self.n_tools} tool calls · "
            f"{self.n_messages} events{extra}[/dim]"
        )
        if self.status_line:
            self.console.print(f"[dim]last status · {self.status_line}[/dim]")
        if found["lines"]:
            self.console.print(f"[{GREEN}]Features so far:[/{GREEN}]")
            for line in found["lines"]:
                self.console.print(f"  {line}")

    def tick(self) -> None:
        if self._live is not None:
            try:
                self._live.refresh()
            except Exception:  # noqa: BLE001
                pass

    def _live_renderable(self) -> Panel:
        """Build each Live frame so elapsed time and library state keep moving."""
        return self._render(done=self._live_done, final=self._live_final)

    def _start_tool(
        self, raw_name: str, tool_id: str | None, arguments: Any = None,
    ) -> None:
        short = _tool_short_name(raw_name)
        if not tool_id:
            self._anonymous_tool_seq += 1
            tool_id = f"anonymous-{self._anonymous_tool_seq}"
        if tool_id not in self.active_tools:
            self.n_tools += 1
        self.active_tools[tool_id] = short
        self._active_tool_arguments[tool_id] = _tool_arguments(arguments)
        self._active_tool_actions.pop(tool_id, None)
        self.last_tool = short

    def _finish_tool(self, tool_id: str | None) -> None:
        if tool_id:
            self.active_tools.pop(tool_id, None)
            self._active_tool_arguments.pop(tool_id, None)
            self._active_tool_actions.pop(tool_id, None)

    def _active_tool_names(self) -> str:
        counts: dict[str, int] = {}
        for name in list(self.active_tools.values()):
            counts[name] = counts.get(name, 0) + 1
        parts = [f"{name} ×{count}" if count > 1 else name for name, count in counts.items()]
        return ", ".join(parts)

    def _active_tool_status(self) -> str:
        statuses = []
        for tool_id, name in self.active_tools.items():
            action = self._active_tool_actions.get(tool_id) or {}
            if action.get("finished"):
                continue
            operation = _ACTIVITY_OPERATIONS.get(action.get("operation"))
            if operation and action.get("name"):
                status = operation.format(name=_detector_label([action["name"]]))
                if action.get("operation") == "polish":
                    completed, total = action.get("completed"), action.get("total")
                    if (type(completed) is int and type(total) is int
                            and 0 <= completed <= total):
                        status += f" · {completed}/{total} regression checks"
                    status += " · goal: Good"
                elif goal := _safe_goal(action.get("goal")):
                    status += f" · goal: {goal}"
                    if action.get("target_grade") in {"OK", "GOOD", "EXCELLENT", "OUTSTANDING"}:
                        status += f" ({action['target_grade'].title()})"
            else:
                status = friendly_tool_status(name, self._active_tool_arguments.get(tool_id))
            statuses.append(_strip_trailing_dots(status))
        if len(statuses) == 1:
            return statuses[0]
        if statuses:
            return f"{len(statuses)} tools active · " + "; ".join(statuses)
        return ""

    def set_runtime_activity(self, source: str, context: dict[str, Any]) -> None:
        """Use explicit operation events for nested submission/evaluation/review work."""
        activity = context.get("activity")
        name = context.get("name")
        if source == "polishing_evaluation":
            name = context.get("detector")
            activity = {
                "operation": "polish", "completed": context.get("completed"),
                "total": context.get("total"), "finished": bool(context.get("finished")),
            }
        if (not isinstance(activity, dict) or not isinstance(name, str)
                or activity.get("operation") not in _ACTIVITY_OPERATIONS):
            return
        matches = [
            tool_id for tool_id in self.active_tools
            if name in _detector_names(self._active_tool_arguments.get(tool_id, {}))
        ]
        if not matches:
            matches = [tool_id for tool_id, tool in self.active_tools.items()
                       if tool == _tool_short_name(source)
                       and not _detector_names(self._active_tool_arguments.get(tool_id, {}))]
        if not matches and source == "polishing_evaluation":
            matches = [tool_id for tool_id, tool in self.active_tools.items()
                       if tool in {"advance_round", "evaluate_functions", "submit_function"}
                       and not _detector_names(self._active_tool_arguments.get(tool_id, {}))]
        # An unnamed concurrent call cannot safely be attributed to one detector.
        if len(matches) != 1:
            return
        self._active_tool_actions[matches[0]] = {**activity, "name": name}
        self.phase = self._active_tool_status() or "Thinking…"
        self.tick()

    def update_tool_arguments(self, name: str, arguments: dict[str, Any]) -> None:
        """Fill arguments omitted from a Codex item-start event before dispatch."""
        short = _tool_short_name(name)
        matches = [tool_id for tool_id, tool in self.active_tools.items()
                   if tool == short and not self._active_tool_arguments.get(tool_id)]
        if len(matches) == 1:
            self._active_tool_arguments[matches[0]] = _tool_arguments(arguments)

    def _clear_tools(self) -> None:
        self.active_tools.clear()
        self._active_tool_arguments.clear()
        self._active_tool_actions.clear()

    def _start_idle_phase(self) -> None:
        self.phase = "Thinking…"
        self._idle_idx = 0
        self._idle_at = time.monotonic()

    def set_status(self, message: str) -> None:
        """Push a natural-language status line into the Live panel."""
        msg = _truncate_status(message)
        if not msg:
            return
        if self.status_line and self.status_line != msg:
            self.status_recent.append(self.status_line)
        self.status_line = msg
        # Prefer NL status as the spinner phase when present.
        self.phase = msg
        self.tick()

    def on_message(self, message: Any) -> None:
        self.n_messages += 1
        subtype = getattr(message, "subtype", None)
        mtype = getattr(message, "type", None) or type(message).__name__

        content = getattr(message, "content", None)
        saw_tool = False
        saw_tool_result = False
        if isinstance(content, list):
            text_bits: list[str] = []
            for block in content:
                btype = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                name = getattr(block, "name", None) or (
                    block.get("name") if isinstance(block, dict) else None
                )
                tool_use_id = getattr(block, "tool_use_id", None) or (
                    block.get("tool_use_id") if isinstance(block, dict) else None
                )
                if btype in {"tool_result", "server_tool_result"} or tool_use_id:
                    saw_tool_result = True
                    self._finish_tool(str(tool_use_id) if tool_use_id else None)
                    continue
                if btype in {"tool_use", "tool_call"} or (
                    name and btype not in {"text", "input_text", "thinking"}
                ):
                    saw_tool = True
                    raw_name = str(name or btype or "tool")
                    block_id = getattr(block, "id", None) or (
                        block.get("id") if isinstance(block, dict) else None
                    )
                    arguments = getattr(block, "input", None) or (
                        block.get("input", block.get("arguments"))
                        if isinstance(block, dict) else None)
                    self._start_tool(raw_name, str(block_id) if block_id else None, arguments)
                elif btype in {"text", "input_text"} or (
                    isinstance(block, dict) and "text" in block and not name
                ):
                    t = getattr(block, "text", None) or (
                        block.get("text") if isinstance(block, dict) else None
                    )
                    if t:
                        text_bits.append(str(t))
                elif getattr(block, "text", None):
                    text_bits.append(str(block.text))
            if saw_tool or saw_tool_result:
                active_status = self._active_tool_status()
                if active_status:
                    self.phase = active_status
                else:
                    self._start_idle_phase()
            if text_bits and not saw_tool:
                soft = _first_short_sentence(" ".join(text_bits))
                if soft and not self.status_line:
                    # Soft Live-only hint — agent should use report_status for durable log.
                    self.phase = _truncate_status(soft)
        elif subtype == "init":
            self.phase = "connected"
        elif "tool" in str(mtype).lower() or "tool" in str(subtype or "").lower():
            if not self.status_line:
                self.phase = str(subtype or mtype)
        elif "Result" in type(message).__name__:
            self._clear_tools()
            self.phase = "finalizing"
        else:
            if self.phase == "starting" and not self.status_line:
                self.phase = "Thinking…"
        # Throttle UI updates — every message was redrawing + re-reading the library.
        if (
            self.n_messages == 1
            or self.n_messages % 3 == 0
            or "tool" in self.phase
            or self.status_line
        ):
            self.tick()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.tick()

    def on_codex_event(self, method: str, params: dict[str, Any]) -> None:
        """Track Codex item IDs without translating them into Claude SDK classes."""
        self.n_messages += 1
        item = params.get("item") or {}
        kind, item_id = item.get("type"), item.get("id")
        builtin = {"commandExecution": "Bash", "fileChange": "Edit",
                   "webSearch": "web search", "imageView": "view image"}
        if method in {"item/started", "item/completed"}:
            if kind in {"dynamicToolCall", "mcpToolCall", "collabAgentToolCall", *builtin}:
                if method == "item/started":
                    self._start_tool(
                        str(item.get("tool") or builtin.get(kind) or kind), item_id,
                        item.get("arguments", item.get("input")),
                    )
                else:
                    self._finish_tool(item_id)
                self.phase = self._active_tool_status() or "Thinking…"
            elif kind == "agentMessage" and method == "item/completed":
                soft = _first_short_sentence(str(item.get("text") or ""))
                if soft and not self.status_line:
                    self.phase = _truncate_status(soft)
            elif kind == "contextCompaction":
                self.phase = "Compacting conversation…"
            elif kind == "reasoning" and not self.active_tools:
                self.phase = "Thinking…"
        elif method == "turn/completed":
            self._clear_tools()
            self.phase = "finalizing"
        if self.n_messages == 1 or self.n_messages % 3 == 0 or method in {
            "item/started", "item/completed", "turn/completed",
        }:
            self.tick()

    def _spinner_text(self, *, done: bool) -> str:
        if done:
            return "complete"
        now = time.monotonic()
        active = self._active_tool_status()
        if active:
            base = _truncate_status(active, _STATUS_SPINNER_MAX)
            return _with_animated_dots(base, now=now, t0=self.t0)
        # Current tool/action phase wins for "what is happening now".
        # Durable NL status (report_status / auto-mirror) shows wrapped below.
        if self.phase and self.phase not in {"starting", "thinking", "Thinking…"}:
            # Keep spinner readable even if phase was set from a long status.
            base = _truncate_status(self.phase, _STATUS_SPINNER_MAX)
            return _with_animated_dots(base, now=now, t0=self.t0)
        # Rotate idle verbs while waiting for first NL status / tool.
        if now - self._idle_at >= 2.5:
            self._idle_idx = (self._idle_idx + 1) % len(_IDLE_VERBS)
            self._idle_at = now
        return _with_animated_dots(_IDLE_VERBS[self._idle_idx], now=now, t0=self.t0)

    def _library_snapshot(self, *, force: bool = False) -> dict[str, Any]:
        """Read working library: failures/features found so far (throttled)."""
        now = time.monotonic()
        if (
            not force
            and self._snap_cache is not None
            and (now - self._snap_at) < self._snap_ttl
        ):
            return self._snap_cache
        try:
            from autofa.mcp.runtime import get_runtime

            rt = get_runtime()
            fns = list(rt.library.functions)
            history = list(rt.scores_history or [])
        except Exception:  # noqa: BLE001
            out = {
                "n": 0,
                "lines": [],
                "summary": "features  0",
                "progress_lines": [],
            }
            self._snap_cache = out
            self._snap_at = now
            return out

        from autofa.behavior_policy import STRICTNESS_LABELS, strictness_for_config
        from autofa.metrics.verdicts import (
            extract_metrics,
            format_coverage_percent,
            format_coverage_support,
            verdict_for_scores,
        )
        from autofa.reports.ordering import rank_display_functions
        from autofa.ui.theme import verdict_color

        # Share the report's display policy, independent of discovery objectives.
        # The active validation policy still determines which reviews are current.
        # Keep references rather than deep-copying potentially large trace evidence.
        strictness = strictness_for_config(rt.config)
        rows = rank_display_functions(
            [{**vars(fn), "function": fn} for fn in fns],
            behavior_strictness=strictness,
            review_history=rt.library.review_history,
        )
        fns = [row["function"] for row in rows]

        lines: list[str] = []
        grade_counts: dict[str, int] = {}
        for row in rows:
            fn, s = row["function"], row["scores"]
            v = verdict_for_scores(s)
            grade_counts[v["code"]] = grade_counts.get(v["code"], 0) + 1
            m = extract_metrics(s)
            no_resolved = s.get("n_scored") == 0
            color = verdict_color(v["code"])
            kind = f"[{fn.kind}] " if fn.kind else ""
            if m["evaluation_status"] == "unscored" and not no_resolved:
                mline = v["reason"]
            else:
                from autofa.metrics.completeness import incomplete_evidence
                from autofa.metrics.verdicts import format_behavior_purity, is_sampled

                scope = ""
                if is_sampled(s):
                    scope = (f"estimate; stratified sample {s.get('sample_n') or '?'}/"
                             f"{s.get('sample_frame_n') or '?'} rollouts")
                    if s.get("eval_truncated") or s.get("evaluation_scope_complete") is False:
                        scope += "; partial corpus frame"
                elif s.get("eval_truncated") or s.get("evaluation_scope_complete") is False:
                    scope = "partial scan"
                elif s.get("evaluation_scope_complete") is True:
                    scope = "full scan"
                if no_resolved or incomplete_evidence(s):
                    detail = ("no resolved rollouts; metrics unavailable" if no_resolved
                              else "resolved rollouts only"
                              if s.get("metrics_scope") == "resolved_items"
                              else "evaluation incomplete")
                    scope = f"{scope}; {detail}" if scope else detail
                scope = f" ({scope})" if scope else ""
                coverage = ("Coverage within parent" if s.get("within_parent") or s.get("explains")
                            else "Coverage")
                adherence = format_behavior_purity(s).replace("beh=", "adherence=", 1)

                # ASCII-ish metric line — avoid exotic glyphs in Live redraws
                # The scoring estimate applies to association metrics; adherence
                # comes from a separate review of sampled detector matches.
                coverage_value = (format_coverage_percent(m["recall"])
                                  if m["coverage_available"] else "—")
                ppv_value = (f"{m['ppv_at_pi0']:.2f}"
                             if m["normalized_precision_available"] else "—")
                precision_value = f"{m['precision']:.2f}" if m["precision"] is not None else "—"
                association = "flagged=—  precision=—  Coverage=—  ppv@pi0=—" if no_resolved else (
                    f"flagged={m['flagged_detail']}  "
                    f"precision={precision_value}  "
                    f"{coverage}={coverage_value} ({format_coverage_support(m)})  "
                    f"ppv@pi0={ppv_value}"
                )
                mline = (f"{association}{scope}  |  "
                         f"Judge policy: {STRICTNESS_LABELS[strictness]} · {adherence}")
                if v["code"] == "UNSCORED":
                    mline += f" · {m['scoring_detail']} · {v['reason']}"
                elif m["diagnostic_review_detail"]:
                    mline += f" · {m['scoring_detail']} · {m['diagnostic_review_detail']}"
                elif s.get("final_validation_skip_detail"):
                    mline += f" · {s['final_validation_skip_detail']}"
            badge = f"[{color}]{v['symbol']} {v['label']}[/{color}]  "
            lines.append(f"{badge}{kind}{fn.name}  [dim]{mline}[/dim]")

        summary = f"features  {len(fns)}"
        if fns:
            names = ", ".join(f.name for f in fns[:5])
            if len(fns) > 5:
                names += f" +{len(fns) - 5} more"
            summary = f"features  {len(fns)}  ·  {names}"

        progress_lines: list[str] = []
        try:
            from autofa.metrics.live_progress import runtime_display_progress_snapshot
            from autofa.metrics.progress import display_progress_rows

            current = runtime_display_progress_snapshot(rt)
            progress_lines = [f"{label}  {value}" for label, value in
                              display_progress_rows(
                                  history, current=current, grade_counts=grade_counts,
                                  task_breadth=(getattr(rt, "_previous_discovery_grades", None)
                                                or {}).get("task_breadth"),
                              )]
        except Exception:  # noqa: BLE001
            progress_lines = []

        try:
            from autofa.objectives import discovery_strategy, preset_name
            from autofa.session.family_discovery import family_progress_summary
            from autofa.ui.family_progress import family_progress_lines

            search_lines = [f"optimization  {preset_name(rt.config)} · "
                            f"discovery strategy: {discovery_strategy(rt.config)}"]
            search_lines.extend(family_progress_lines(family_progress_summary(rt.run.run_dir())))
            if getattr(rt.config, "discovery_terminate_if_stalled", True):
                from autofa.session.round_stagnation import round_stagnation_status

                stagnation = round_stagnation_status(rt)
                if rt.run.round >= rt.run.max_rounds:
                    search_lines.append(
                        f"recovery patience  {stagnation['rounds_without_improve']}/"
                        f"{stagnation['patience']} completed extra rounds without "
                        "validated library improvement"
                    )
                else:
                    search_lines.append(
                        "recovery patience  planned rounds protected; "
                        f"{stagnation['patience']} extra rounds without validated improvement "
                        "before stall termination"
                    )
            headlines = [line for line in progress_lines
                         if line.startswith(("Library coverage  ", "Task coverage  "))]
            progress_lines = headlines + search_lines + [
                line for line in progress_lines if line not in headlines
            ]
        except (OSError, ValueError, AttributeError):
            # Optional saved progress must never prevent rendering discovery.
            pass

        out = {
            "n": len(fns),
            "lines": lines,
            "summary": summary,
            "progress_lines": progress_lines,
        }
        self._snap_cache = out
        self._snap_at = now
        return out

    def _elapsed(self) -> tuple[float, bool]:
        if self.runtime_elapsed is not None:
            try:
                seconds = self.runtime_elapsed()
                if seconds is not None:
                    return max(0, seconds), True
            except Exception:  # Display failures must never interrupt discovery.
                pass
        return max(0, time.monotonic() - self.t0), False

    @staticmethod
    def _duration(seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f"{hours}h {mins:02d}m {secs:02d}s" if hours else f"{mins}m {secs:02d}s"

    def _render(self, *, done: bool = False, final: str | None = None) -> Panel:
        elapsed, cumulative = self._elapsed()
        timing = self._duration(elapsed)
        timing += " active discovery · pauses excluded" if cumulative else " this turn"
        title = Text()
        if done:
            title.append("✓ ", style=f"bold {GREEN}")
            title.append(final or "Done", style=f"bold {GREEN}")
        else:
            title.append("AutoFA", style=f"bold {GREEN}")
            title.append(f" {self.label}", style="bold")

        found = self._library_snapshot()
        spin = self._spinner_text(done=done)
        if not done:
            self._spinner.update(text=spin, style=GREEN)
        tool_line = (
            f"active    {len(self.active_tools)} · {self._active_tool_names()}"
            if self.active_tools
            else f"last      {self.last_tool or '—'}"
        )
        rows: list[Any] = [
            self._spinner if not done else Text(spin, style=GREEN),
            Text(f"elapsed   {timing}", style="dim"),
            Text(f"tools     {self.n_tools}   events {self.n_messages}", style="dim"),
            Text(tool_line, style="dim"),
        ]
        if self.runtime_hint is not None:
            try:
                timing = self.runtime_hint()
                if timing:
                    rows.insert(2, Text(timing, style=GREEN))
            except Exception:  # Timing is optional; never interrupt discovery.
                pass
        # Durable status (report_status / auto-mirror) under the spinner action.
        # Recent + current both wrap (same width); recent keeps a slightly shorter budget.
        wrap_w = _STATUS_WRAP
        try:
            wrap_w = max(48, min(120, (self.console.size.width or _STATUS_WRAP) - 14))
        except Exception:  # noqa: BLE001
            wrap_w = _STATUS_WRAP
        for prev in list(self.status_recent)[-2:]:
            wrapped = _wrap_status_lines(prev, width=wrap_w, max_lines=4)
            for j, wl in enumerate(wrapped):
                prefix = "  · " if j == 0 else "    "
                rows.append(Text(f"{prefix}{wl}", style="dim"))
        if self.status_line:
            wrapped = _wrap_status_lines(self.status_line, width=wrap_w, max_lines=5)
            for j, wl in enumerate(wrapped):
                prefix = "status    " if j == 0 else "          "
                rows.append(Text(f"{prefix}{wl}", style=GREEN))
        rows.append(Text(found["summary"], style=f"bold {GREEN}" if found["n"] else "dim"))
        for i, pl in enumerate(found.get("progress_lines") or []):
            style = (f"bold {GREEN}" if pl.startswith(("Library coverage  ", "Task coverage  "))
                     else GREEN if i == 0 else "dim")
            rows.append(Text(pl, style=style))
        for line in found["lines"][:5]:
            try:
                rows.append(Text.from_markup(f"  {line}"))
            except Exception:  # noqa: BLE001
                rows.append(Text(f"  {line}", style="dim"))
        if found["n"] > 5:
            rows.append(Text(f"  • … +{found['n'] - 5} more", style="dim"))
        rows.append(
            Text.from_markup(
                self.hint.replace("Ctrl+C", "[bold]Ctrl+C[/bold]"),
                style="dim",
            )
        )
        rows.append(
            Text.from_markup(
                "[bold]Ctrl+C[/bold] = stop this turn · then ask /clean /dashboard /…",
                style="dim",
            )
        )

        return Panel(Group(*rows), title=title, border_style=GREEN)
