# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge-facing trace excerpts.

``item_trace_for_judge`` builds an outcome-free view of one datapoint's trace
for behavioral judges, dividing a character budget fairly across the available
evidence fields.
"""

from __future__ import annotations

from typing import Any


def item_trace_for_judge(item: dict[str, Any], *, max_chars: int = 6000) -> str:
    """Build a judge-facing trace without outcome labels.

    Prefers derived trajectory surfaces (reasoning / actions / observations /
    final message) over the raw multi-MB JSON blob so the budget is spent on
    behavior rather than ATIF envelope boilerplate.
    Each distinct field receives a budget, short fields are kept whole, and
    omissions are marked. Budgets too small to cover all fields fail explicitly.
    """
    from autofa.data.trace_fields import derive_trace_fields
    from autofa.data.trajectory import parse_trajectory
    from autofa.llm.windowing import text_window

    enriched = derive_trace_fields(item)
    fields: list[tuple[str, str]] = []
    input_text = enriched.get("input_text")
    origin = enriched.get("input_parse", {}).get("origin")
    if isinstance(input_text, str) and input_text.strip():
        # Captured requests outrank summaries. Use the shared input view so
        # literal exported JSON retains its user role and system context.
        input_field = origin if origin in {"prompt_text", "question"} and (
            input_text == enriched.get(origin)
        ) else "input_text"
        fields.append((input_field, input_text.strip()))
        # Local columns may carry distinct input evidence without a capture.
        selected = str(origin).removeprefix("system_text+")
        if selected in {"prompt_text", "question"}:
            for alternate in ("prompt_text", "question"):
                value = enriched.get(alternate)
                if (alternate != selected and isinstance(value, str) and value.strip()
                        and value != enriched.get(selected)):
                    fields.append((alternate, value.strip()))
    elif origin in {
        "captured_initial_request", "recorded_agent_request", "input_json",
        "model_call_evidence", "frontier_compact_export",
    }:
        # An empty text projection can represent non-text content or an invalid
        # capture. It does not justify replacing the selected input with summaries.
        fields.append(("input_text", "[Selected input has no readable text. "
                       "Refer to the stored input evidence.]"))
    preferred = (
        "reasoning_text", "reasoning_content",
        "chain_of_thought", "actions_text", "observations_text", "final_message",
        "response_text", "generation",
    )
    if not fields:
        preferred = ("system_text", "prompt_text", "question", *preferred)
    aliases = {
        "question": "prompt_text", "reasoning_content": "reasoning_text",
        "chain_of_thought": "reasoning_text", "generation": "response_text",
        "final_message": "response_text",
    }
    seen: set[tuple[str, str]] = set()
    for key in preferred:
        val = enriched.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        val = val.strip()
        identity = (aliases.get(key, key), val)
        if identity in seen:
            continue
        # A parsed envelope duplicates the derived fields. A flat response is
        # independent evidence and must retain its own share of the budget.
        if key in {"response_text", "generation"} and parse_trajectory(val).ok:
            continue
        seen.add(identity)
        fields.append((key, val))
    if not fields:
        return "[No supported trace fields available]"[:max_chars]

    note = (
        "Evidence excerpts: omitted text is not evidence of absence. Read full fields "
        "before deciding claims requiring omitted context; otherwise evidence is insufficient."
    )
    # Reserve headers first, then divide the remaining budget fairly. Short
    # fields are kept whole and donate unused space to the remaining fields.
    headers = [f"## {key} [total_chars={len(val)}]\n" for key, val in fields]
    overhead = len(note) + 2 + sum(map(len, headers)) + 2 * (len(fields) - 1)
    remaining = int(max_chars) - overhead
    if remaining < sum(min(64, len(val)) for _, val in fields):
        raise ValueError("judge excerpt budget too small to represent every evidence field")
    allocations = [0] * len(fields)
    pending = list(range(len(fields)))
    while pending:
        share = remaining // len(pending)
        short = [i for i in pending if len(fields[i][1]) <= share]
        if short:
            for i in short:
                allocations[i] = len(fields[i][1])
                remaining -= allocations[i]
                pending.remove(i)
        else:
            for j, i in enumerate(pending):
                allocations[i] = share + (j < remaining % len(pending))
            break
    parts = []
    for (_, val), header, budget in zip(fields, headers, allocations, strict=True):
        excerpt = val if len(val) <= budget else text_window(val, focus="ends", max_chars=budget)
        parts.append(header + excerpt)
    return note + "\n\n" + "\n\n".join(parts)
