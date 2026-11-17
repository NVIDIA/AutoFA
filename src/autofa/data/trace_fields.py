# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive detector-friendly text surfaces from agent trajectories.

On agent corpora ``response_text`` is a multi-MB JSON blob. Detectors and the
behavior judge need readable surfaces: reasoning, actions, observations, and
the final message. This module builds those once per item (size-1 cache) from
``autofa.data.trajectory``, preserving the full text by default. Prompt and
display budgets belong at the consuming boundary, after evidence selection.
"""

from __future__ import annotations

import threading
from typing import Any

from autofa.data.input_view import derive_input_fields, response_value
from autofa.data.trajectory import (
    _obs_to_str,
    format_actions,
    format_reasoning,
    parse_trajectory,
)

TRACE_FIELDS: tuple[str, ...] = (
    "reasoning_text",
    "actions_text",
    "observations_text",
    "final_message",
)

# Surfaces a detector may read; used by diversity / lint.
SURFACE_FIELDS: dict[str, frozenset[str]] = {
    "prompt": frozenset({"system_text", "prompt_text", "question", "problem", "input", "prompt",
                         "instruction", "user_message", "input_text"}),
    "reasoning": frozenset({"reasoning_text", "reasoning_content", "chain_of_thought", "cot",
                            "thinking", "thoughts", "scratchpad"}),
    "actions": frozenset({"actions_text"}),
    "observations": frozenset({"observations_text"}),
    "response": frozenset({"response_text", "generation", "answer", "final_answer", "completion",
                           "output", "model_answer", "final_message"}),
}

# Size-1 parse cache: eval is item-major (for it in items: for fn in targets).
# Thread-local because LLM detectors are scored on a worker pool, where a shared
# size-1 cache would both thrash and race between threads on different items.
_CACHE = threading.local()


def _response_blob(item: dict[str, Any]) -> Any:
    return response_value(item)


def _has_native_reasoning(item: dict[str, Any]) -> bool:
    for key in ("reasoning_text", "reasoning_content", "chain_of_thought", "cot"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


def _observations_text(steps: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, step in enumerate(steps, 1):
        obs = step.get("observation")
        obs_s = _obs_to_str(obs).strip()
        if obs_s:
            parts.append(f"[step {i} · observation]\n{obs_s}")
    return "\n\n".join(parts)


def _final_message(steps: list[dict[str, Any]]) -> str:
    for step in reversed(steps):
        src = str(step.get("source") or "").lower()
        if src not in {"agent", "assistant"}:
            continue
        msg = step.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    return ""


def derive_trace_fields(
    item: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Return a shallow copy of ``item`` with derived TRACE_FIELDS filled in.

    Does not mutate ``item``. Skips ``reasoning_text`` when a native CoT column
    is already present (unless ``overwrite``). Non-trajectory rows are returned
    unchanged. Parsing is shared with inspection and the structured helpers.
    """
    out = derive_input_fields(item)
    blob = _response_blob(item)
    if not blob:
        return out

    cached = getattr(_CACHE, "derived", None)
    cache_hit = getattr(_CACHE, "key", None) == blob and cached is not None
    if cache_hit:
        derived = dict(cached or {})
    else:
        parsed = parse_trajectory(blob)
        _CACHE.key = blob
        _CACHE.parsed = parsed
        if not parsed.ok or not parsed._steps:
            _CACHE.derived = {}
            return out
        steps = parsed._steps
        derived = {
            "actions_text": format_actions(steps),
            "observations_text": _observations_text(steps),
            "final_message": _final_message(steps),
        }
        reasoning = format_reasoning(steps)
        if reasoning:
            derived["reasoning_text"] = reasoning
        # Keep a few extras for consumers that want step counts without re-parse.
        derived["_trace_n_steps"] = str(parsed.n_steps)
        derived["_trace_n_tool_calls"] = str(parsed.n_tool_calls)
        derived["_trace_dialect"] = parsed.dialect or ""
        _CACHE.derived = derived

    for key in TRACE_FIELDS:
        val = derived.get(key)
        if not val:
            continue
        if key == "reasoning_text" and _has_native_reasoning(out) and not overwrite:
            continue
        if key in out and isinstance(out[key], str) and out[key].strip() and not overwrite:
            if key != "reasoning_text":
                # Prefer existing non-empty values unless overwrite.
                continue
        out[key] = str(val)
    return out


def surfaces_present(item: dict[str, Any]) -> set[str]:
    """Which named surfaces have non-empty content on this item."""
    enriched = derive_trace_fields(item)
    found: set[str] = set()
    for surface, fields in SURFACE_FIELDS.items():
        for f in fields:
            val = enriched.get(f)
            if isinstance(val, str) and val.strip():
                found.add(surface)
                break
    return found


def surface_byte_shares(item: dict[str, Any]) -> dict[str, int]:
    """Byte lengths per surface for one item (0 if absent)."""
    enriched = derive_trace_fields(item)
    shares: dict[str, int] = {}
    for surface, fields in SURFACE_FIELDS.items():
        n = 0
        for f in fields:
            val = enriched.get(f)
            if isinstance(val, str):
                n = max(n, len(val))
        shares[surface] = n
    return shares
