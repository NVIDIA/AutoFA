# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render complete judge trajectories without the detector parser's actor guesses."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from autofa.data.structured import MESSAGE_TYPES, decode_container
from autofa.data.trajectory import parse_trajectory


def _actor(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _message(value: Any, default_actor: str) -> Any:
    value = deepcopy(decode_container(value))
    if isinstance(value, str):
        return {"type": "message", "role": default_actor, "content": value}
    if not isinstance(value, dict):
        return value
    if (not value.get("role") and value.get("type") not in MESSAGE_TYPES
            and any(key in value for key in ("messages", "input", "system", "instructions"))):
        for key in ("messages", "input"):
            if key in value:
                value[key] = _messages(value[key], default_actor)
        return value
    # A declared role or an explicit step source is evidence. The mere presence
    # of a tool call is not evidence that its caller was the assistant.
    value["role"] = _actor(value.get("role")) or _actor(value.get("source")) or default_actor
    if not value.get("type"):
        value["type"] = "message"
    return value


def _messages(value: Any, default_actor: str = "unknown") -> Any:
    value = decode_container(value)
    if isinstance(value, list):
        return [_message(message, default_actor) for message in value]
    return _message(value, default_actor)


def _prepare(value: Any) -> Any:  # noqa: C901
    """Visit schema boundaries only; never interpret JSON in tool arguments/text."""
    value = deepcopy(decode_container(value))
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict) and any(
            key in value[0] for key in ("steps", "messages", "conversation", "model_calls",
                                       "ng_trajectory", "choices", "output")
        ):
            return [_prepare(value[0])]
        if value and all(isinstance(step, dict) for step in value) and any(
            any(key in step for key in ("source", "message", "observation", "step_id"))
            for step in value
        ):
            return _prepare({"steps": value})
        return _messages(value)
    if not isinstance(value, dict):
        return value
    if "ng_trajectory" in value:
        value["ng_trajectory"] = _prepare(value["ng_trajectory"])
    if "model_calls" in value:
        calls = deepcopy(decode_container(value["model_calls"]))
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                request = deepcopy(decode_container(call.get("request")))
                if isinstance(request, dict):
                    # Request direction alone does not identify actors in a
                    # captured history containing several participants.
                    call["request"] = _messages(request)
                response = deepcopy(decode_container(call.get("response")))
                if isinstance(response, dict):
                    # A captured model response proves its default actor. Keep
                    # any explicitly recorded role instead of overwriting it.
                    if "output" in response:
                        response["output"] = _messages(response["output"], "assistant")
                    elif "choices" not in response:
                        response = _message(response, "assistant")
                    call["response"] = response
            value["model_calls"] = calls
    if "steps" in value:
        steps = deepcopy(decode_container(value["steps"]))
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    step["source"] = (
                        _actor(step.get("source")) or _actor(step.get("role")) or "unknown"
                    )
            value["steps"] = steps
        return value
    for key in ("messages", "conversation", "output"):
        if key in value:
            value[key] = _messages(value[key])
            return value
    if "choices" in value:
        choices = deepcopy(decode_container(value["choices"]))
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict) and "message" in choice:
                    choice["message"] = _message(choice["message"], "assistant")
            value["choices"] = choices
        return value
    if any(key in value for key in ("source", "message", "observation", "step_id")):
        return _prepare({"steps": [value]})
    if ("role" in value or value.get("type") in MESSAGE_TYPES
            or "content" in value):
        return _message(value, "unknown")
    return value


def judge_trajectory_text(value: Any) -> str | None:
    """Return all parsed events with explicit actors; unavailable actors stay unknown.

    Original evidence remains in its stored field. This auxiliary view uses the
    shared complete parser after neutralizing its defaults at known boundaries.
    Unsupported or incomplete parsing adds no potentially misleading derived view.
    """
    prepared = _prepare(value)
    trajectory = parse_trajectory(prepared)
    if not trajectory.ok or not trajectory._steps:
        return None
    steps = trajectory._steps
    envelope = prepared[0] if isinstance(prepared, list) and len(prepared) == 1 else prepared
    references = None
    if isinstance(envelope, dict):
        from autofa.data.evidence_refs import compact_trace

        try:
            compact = compact_trace(envelope)
        except ValueError:
            return None  # Original unsupported evidence remains in its stored field.
        if compact.get("_autofa_evidence_refs"):
            # Use the readable normalized fields once. Re-rendering hydrated
            # parser steps would copy original parts and metadata back in.
            steps = compact["steps"]
            references = compact["_autofa_evidence_refs"]
    preamble = (
        "Normalized conversation view. Source references point into the original "
        "stored trace's execution records; metadata, full source parts, and other "
        "execution lanes remain there. This view does not replace a whole-trace read.\n"
        + json.dumps({"source_references": references}, ensure_ascii=False, sort_keys=True)
        + "\n\n"
    ) if references else ""
    return preamble + "\n\n".join(
        f"[event {index} | source={step.get('source') or 'unknown'}]\n"
        + json.dumps(step, ensure_ascii=False, sort_keys=True, default=str)
        for index, step in enumerate(steps, 1)
    )
