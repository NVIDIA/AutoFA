# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse agent trajectories embedded in ``response_text`` / ``generation``.

Many agent-run corpora (OpenHands / Harbor ATIF-v1.5 and similar) dump an entire
session as one JSON string. Detectors still see that raw string; this module
turns it into structured steps/tools for MCP exploration and the HTML inspector.

Dialects
--------
- ``atif_openhands`` — list-of-one envelope with ``schema_version`` / ``agent`` / ATIF
- ``steps_envelope`` — generic list-of-one or dict with a ``steps`` list
- ``None`` — unparsed; may still set ``warning`` when the blob *looks* structured
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from autofa.data.evidence_refs import expand_trace
from autofa.data.structured import (
    MESSAGE_TYPES,
    decode_container,
    message_steps,
    messages,
    normalize_step,
    tool_call,
)

DEFAULT_ARGS_EXCERPT = 600
DEFAULT_REASONING_EXCERPT = 2_000

_AGENT_SOURCES = frozenset({"agent", "assistant"})
_PARSE_CACHE = threading.local()


@dataclass
class ToolStat:
    name: str
    count: int = 0
    example_args_excerpt: str = ""


@dataclass
class StepSummary:
    index: int  # 1-based
    source: str = ""
    step_id: int | None = None
    has_reasoning: bool = False
    has_message: bool = False
    tool_names: list[str] = field(default_factory=list)
    n_tool_calls: int = 0
    observation_chars: int = 0
    model_name: str = ""


@dataclass
class ParseResult:
    ok: bool
    dialect: str | None = None
    warning: str | None = None
    n_steps: int = 0
    n_agent_steps: int = 0
    n_tool_calls: int = 0
    tools: list[ToolStat] = field(default_factory=list)
    steps_summary: list[StepSummary] = field(default_factory=list)
    reasoning_excerpts: list[str] = field(default_factory=list)
    agent_name: str | None = None
    schema_version: str | None = None
    model_name: str | None = None
    final_metrics: dict[str, Any] | None = None
    workspace_diff_preview: str | None = None
    tool_definitions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    # Kept for consumers that need the raw step dicts (MCP step fetch / HTML).
    _steps: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self, *, include_steps: bool = False) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_steps", None)
        if include_steps:
            d["steps"] = self._steps
        return d


def looks_like_agent_json(text: Any) -> bool:
    """Heuristic: blob looks like a structured agent trace, not plain prose."""
    if not text:
        return False
    decoded = decode_container(text)
    if isinstance(decoded, (dict, list)):
        objects = decoded if isinstance(decoded, list) else [decoded]
        return any(isinstance(obj, dict) and any(k in obj for k in (
            "steps", "tool_calls", "tool_definitions", "reasoning_content", "messages",
            "role", "model_calls", "ng_trajectory", "choices", "output",
        )) for obj in objects)
    s = str(text).lstrip()
    if not s or s[0] not in "[{":
        return False
    markers = ('"steps"', '"tool_calls"', '"tool_definitions"', '"reasoning_content"',
               '"messages"', '"role"', '"model_calls"', '"function_call"')
    return any(m in s for m in markers)


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return str(v)
    except Exception:  # noqa: BLE001
        return ""


def _truncate(s: str, n: int) -> str:
    if not s or n <= 0 or len(s) <= n:
        return s
    return s[:n] + f"…[+{len(s) - n} chars]"


def _args_to_str(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return _safe_str(args)


def _obs_to_str(obs: Any) -> str:
    if obs is None or obs == "":
        return ""
    if isinstance(obs, str):
        return obs
    try:
        return json.dumps(obs, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return _safe_str(obs)


def _obs_len(obs: Any) -> int:
    return len(_obs_to_str(obs))


def extract_tool_call(tc: dict[str, Any]) -> tuple[str, Any]:
    """Return ``(name, arguments)`` from Harbor or OpenAI-nested tool_call shapes."""
    fn_obj = tc.get("function") if isinstance(tc.get("function"), dict) else None
    name = (
        _safe_str(tc.get("function_name"))
        or _safe_str(tc.get("name"))
        or (_safe_str(fn_obj.get("name")) if fn_obj else "")
        or "tool"
    )
    args = tc.get("arguments")
    if args is None and tc.get("args") is not None:
        args = tc.get("args")
    if args is None and fn_obj is not None:
        args = fn_obj.get("arguments")
    return name, decode_container(args)


def extract_steps_list(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Pull ``steps`` and optional envelope dict from parsed JSON.

    Returns ``(steps, envelope_or_None)``.
    """
    data = decode_container(data)
    if isinstance(data, list):
        if data and all(isinstance(s, dict) and "source" in s for s in data):
            return [normalize_step(s) for s in data], {"steps": data}
        if data and all(isinstance(m, dict) and (
            "role" in m or m.get("type") in MESSAGE_TYPES
        ) for m in data):
            return _message_trajectory(data)
        if len(data) != 1:
            raise ValueError("expected exactly one trajectory envelope; refusing partial parse")
        data = decode_container(data[0])
    if isinstance(data, dict):
        if "_autofa_evidence_refs" in data:
            data = expand_trace(data)
        captured = decode_container(data.get("ng_trajectory", data))
        if isinstance(captured, dict) and "model_calls" in captured:
            from autofa.data.remote_api import DataApiError
            from autofa.data.source_model_calls import recover_calls

            calls = decode_container(captured["model_calls"])
            if not isinstance(calls, list):
                raise ValueError("model_calls must be an array; refusing partial parse")
            try:
                recovered = recover_calls(calls)
            except DataApiError as exc:
                raise ValueError(str(exc)) from exc
            steps = [normalize_step(s) for s in recovered.pop("steps")]
            limits = recovered.pop("limitations")
            recovered.pop("excluded_call_ids")  # No exported copies of them exist here.
            return steps, {"steps": steps, "model_call_evidence": recovered,
                           "_message_format": True, "_parse_limitations": list(limits)}
        steps = decode_container(data.get("steps"))
        if "steps" in data:
            if not isinstance(steps, list) or any(not isinstance(s, dict) for s in steps):
                raise ValueError("steps must be an array of objects; refusing partial parse")
            return [normalize_step(s) for s in steps], data
        for key in ("messages", "conversation", "output"):
            if key in data:
                return _message_trajectory(data[key], envelope=data)
        if data.get("role") or data.get("type") in MESSAGE_TYPES:
            return _message_trajectory([data])
        if "choices" in data:
            choices = decode_container(data["choices"])
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("expected one response choice; refusing partial parse")
            return _message_trajectory([choices[0].get("message")])
    return [], None


def _message_trajectory(value: Any, *, envelope: dict | None = None) -> tuple[list[dict], dict]:
    parsed, warnings = messages(value, role="assistant")
    if warnings:
        raise ValueError("; ".join(warnings) + "; refusing partial parse")
    steps = []
    for message in parsed:
        converted, issues = message_steps(message)
        steps.extend(normalize_step(s) for s in converted)
        warnings.extend(issues)
    # Completeness declarations belong to the enclosing trace, not just individual
    # messages. Keep them when replacing the input collection with normalized steps.
    inherited_limits = (envelope or {}).get("_parse_limitations", [])
    if not isinstance(inherited_limits, list):
        raise ValueError("trajectory parse limitations must be an array")
    return steps, {**(envelope or {}), "steps": steps, "_message_format": True,
                   "_parse_limitations": inherited_limits + warnings}


def detect_dialect(envelope: dict[str, Any] | None, raw_head: str) -> str | None:
    if envelope is None:
        return None
    if envelope.get("_message_format"):
        return "messages"
    schema = _safe_str(envelope.get("schema_version"))
    agent = envelope.get("agent") if isinstance(envelope.get("agent"), dict) else {}
    agent_name = _safe_str(agent.get("name")).lower()
    if (
        "ATIF" in schema.upper()
        or "atif" in raw_head[:400].lower()
        or "openhands" in agent_name
        or agent_name.endswith("-sdk")
    ):
        return "atif_openhands"
    if isinstance(envelope.get("steps"), list) or "steps" in envelope:
        return "steps_envelope"
    return None


def _step_reasoning(step: dict[str, Any]) -> str:
    for key in ("reasoning_content", "thought", "reasoning"):
        v = step.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _build_from_steps(
    steps: list[dict[str, Any]],
    *,
    dialect: str,
    envelope: dict[str, Any] | None,
    args_excerpt: int = DEFAULT_ARGS_EXCERPT,
    reasoning_excerpt: int = DEFAULT_REASONING_EXCERPT,
) -> ParseResult:
    tool_counts: Counter[str] = Counter()
    tool_examples: dict[str, str] = {}
    summaries: list[StepSummary] = []
    reasoning_excerpts: list[str] = []
    n_agent = 0
    n_tools = 0
    model_name = ""

    for i, step in enumerate(steps, 1):
        src = _safe_str(step.get("source"))
        rc = _step_reasoning(step)
        msg = step.get("message")
        has_msg = isinstance(msg, str) and bool(msg.strip())
        tcs = step.get("tool_calls") if isinstance(step.get("tool_calls"), list) else []
        names: list[str] = []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            n_tools += 1
            name, args = extract_tool_call(tc)
            names.append(name)
            tool_counts[name] += 1
            if name not in tool_examples:
                tool_examples[name] = _truncate(_args_to_str(args), args_excerpt)
        if src in _AGENT_SOURCES:
            n_agent += 1
            if not model_name:
                model_name = _safe_str(step.get("model_name"))
            if rc:
                reasoning_excerpts.append(
                    _truncate(f"[step {i}]\n{rc}", reasoning_excerpt + 32)
                )
        sid = step.get("step_id")
        summaries.append(
            StepSummary(
                index=i,
                source=src,
                step_id=int(sid) if isinstance(sid, int) else None,
                has_reasoning=bool(rc),
                has_message=has_msg,
                tool_names=names,
                n_tool_calls=len(names),
                observation_chars=_obs_len(step.get("observation")),
                model_name=_safe_str(step.get("model_name")),
            )
        )

    tools = [
        ToolStat(name=n, count=c, example_args_excerpt=tool_examples.get(n, ""))
        for n, c in tool_counts.most_common()
    ]

    agent_name = None
    schema_version = None
    final_metrics = None
    workspace_diff = None
    tool_defs: list[str] = []
    if envelope:
        schema_version = _safe_str(envelope.get("schema_version")) or None
        agent = envelope.get("agent") if isinstance(envelope.get("agent"), dict) else {}
        agent_name = _safe_str(agent.get("name")) or None
        tds = agent.get("tool_definitions") if isinstance(agent, dict) else None
        if isinstance(tds, list):
            for td in tds:
                if isinstance(td, dict):
                    # Harbor: {function: {name, …}} or {name: …}
                    fn = td.get("function") if isinstance(td.get("function"), dict) else None
                    nm = _safe_str(td.get("name")) or (
                        _safe_str(fn.get("name")) if fn else ""
                    )
                    if nm:
                        tool_defs.append(nm)
        fm = envelope.get("final_metrics")
        if isinstance(fm, dict):
            final_metrics = {
                k: fm.get(k)
                for k in (
                    "total_prompt_tokens",
                    "total_completion_tokens",
                    "total_cost_usd",
                )
                if k in fm
            }
            workspace_diff = _safe_str(fm.get("workspace_diff_preview")) or None

    return ParseResult(
        ok=True,
        dialect=dialect,
        warning=None,
        n_steps=len(steps),
        n_agent_steps=n_agent,
        n_tool_calls=n_tools,
        tools=tools,
        steps_summary=summaries,
        reasoning_excerpts=reasoning_excerpts[:40],
        agent_name=agent_name,
        schema_version=schema_version,
        model_name=model_name or None,
        final_metrics=final_metrics,
        workspace_diff_preview=_truncate(workspace_diff, 4_000) if workspace_diff else None,
        tool_definitions=tool_defs,
        _steps=steps,
    )


def parse_trajectory(text: Any) -> ParseResult:
    """Parse a complete trajectory; cache one result per thread and isolate callers."""
    if isinstance(text, str) and getattr(_PARSE_CACHE, "text", None) == text:
        cached = getattr(_PARSE_CACHE, "result", None)
        if cached is not None:
            return deepcopy(cached)
    result = _parse_trajectory(text)
    if isinstance(text, str):
        _PARSE_CACHE.text = text
        _PARSE_CACHE.result = result
    return deepcopy(result)


def _parse_trajectory(text: Any) -> ParseResult:
    if not text or not str(text).strip():
        return ParseResult(ok=False, warning=None)

    data = decode_container(text)
    raw = text.strip() if isinstance(text, str) else ""
    if not isinstance(data, (dict, list)) and not raw.startswith(("{", "[", '"')):
        return ParseResult(ok=False)
    looks = looks_like_agent_json(data)

    if not isinstance(data, (dict, list)):
        if looks:
            return ParseResult(
                ok=False,
                dialect=None,
                warning=(
                    "Response looks like a structured agent trace (JSON with steps/tools) "
                    "but JSON parse failed. Only OpenHands/ATIF and generic steps envelopes "
                    "are supported — inspect with get_item and consider adding a dialect."
                ),
            )
        return ParseResult(ok=False)

    try:
        steps, envelope = extract_steps_list(data)
    except ValueError as exc:
        return ParseResult(ok=False, warning=str(exc))
    if not steps:
        if looks:
            return ParseResult(
                ok=False,
                dialect=None,
                warning=(
                    "Many responses look like structured agent traces but no dialect "
                    "matched (expected a list/dict with a steps array). Only OpenHands/ATIF "
                    "+ generic steps envelopes are supported — inspect with get_item and "
                    "consider adding a dialect."
                ),
            )
        return ParseResult(ok=False)

    dialect = detect_dialect(envelope, raw[:800]) or "steps_envelope"
    result = _build_from_steps(steps, dialect=dialect, envelope=envelope)
    result.limitations = list(dict.fromkeys((envelope or {}).get("_parse_limitations", [])))
    for step in steps:
        for call in step.get("tool_calls") or []:
            warning = call.get("arguments_warning")
            if warning and warning not in result.limitations:
                result.limitations.append(warning)
    return result


def format_reasoning(steps: list[dict[str, Any]]) -> str:
    """Full concatenated reasoning stream (HTML / long views)."""
    parts: list[str] = []
    for i, step in enumerate(steps, 1):
        rc = _step_reasoning(step)
        if rc:
            parts.append(f"[step {i}]\n{rc}")
    return "\n\n".join(parts)


def format_actions(steps: list[dict[str, Any]], *, start_index: int = 1) -> str:
    """Human-readable tool/action transcript (HTML inspector)."""
    lines: list[str] = []
    for i, step in enumerate(steps, start_index):
        src = _safe_str(step.get("source"))
        msg = step.get("message")
        if isinstance(msg, str) and msg.strip() and src in _AGENT_SOURCES:
            lines.append(f"[step {i} · {src}]\n{msg.strip()}")
        tcs = step.get("tool_calls") if isinstance(step.get("tool_calls"), list) else []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            name, args = extract_tool_call(tc)
            arg_s = _args_to_str(args)
            if not isinstance(args, str) and args is not None:
                try:
                    arg_s = json.dumps(args, ensure_ascii=False, indent=2)
                except Exception:  # noqa: BLE001
                    pass
            lines.append(f"[step {i} · tool] {name}\n{arg_s}")
        obs = step.get("observation")
        obs_s = _obs_to_str(obs)
        if obs_s.strip():
            if isinstance(obs, (dict, list)):
                try:
                    obs_s = json.dumps(obs, ensure_ascii=False, indent=2)
                except Exception:  # noqa: BLE001
                    pass
            lines.append(f"[step {i} · observation]\n{obs_s}")
    return "\n\n".join(lines)


def get_step_detail(
    steps: list[dict[str, Any]],
    step_index: int,
    *,
    max_chars: int = 50_000,
) -> dict[str, Any] | None:
    """1-based step detail for MCP ``get_trajectory_step``."""
    if step_index < 1 or step_index > len(steps):
        return None
    step = steps[step_index - 1]
    tcs_out: list[dict[str, Any]] = []
    raw_tcs = step.get("tool_calls") if isinstance(step.get("tool_calls"), list) else []
    for tc in raw_tcs:
        if not isinstance(tc, dict):
            continue
        tcs_out.append(tool_call(tc))
    obs_s = _obs_to_str(step.get("observation"))
    reasoning = _step_reasoning(step)
    message = _safe_str(step.get("message"))
    return {
        "index": step_index,
        "source": _safe_str(step.get("source")),
        "step_id": step.get("step_id"),
        "model_name": _safe_str(step.get("model_name")) or None,
        "reasoning": _truncate(reasoning, max_chars),
        "message": _truncate(message, max_chars),
        "tool_calls": tcs_out,
        "observation": _truncate(obs_s, max_chars),
        "observation_data": decode_container(step.get("observation")),
        "reasoning_chars": len(reasoning),
        "message_chars": len(message),
        "observation_chars": len(obs_s),
    }
