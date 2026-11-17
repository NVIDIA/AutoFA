# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-aware decoding shared by imports, detectors and inspection tools.

Decode JSON at structural boundaries, never recursively inside arbitrary model
text or tool parameters. A parameter containing JSON source code is still text.
Raw columns remain the archive; changed structural values also retain raw values.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from autofa.data.execution_policy import is_evaluation_record, sanitize_metadata

MAX_JSON_LAYERS = 8
TEXT_TYPES = {"text", "input_text", "output_text", "summary_text", "reasoning_text"}
ROLES = {"system", "developer", "user", "assistant", "agent", "tool", "function"}
MESSAGE_TYPES = {
    "message",
    "function_call",
    "function_call_output",
    "reasoning",
    "tool_use",
    "tool_result",
}


def decode_container(value: Any) -> Any:
    """Unwrap JSON-encoded objects/arrays, including repeated JSON encoding.

    Scalars and non-JSON text are unchanged. The bound prevents pathological
    wrappers; callers report unsupported structures instead of guessing.
    """
    original = value
    for _ in range(MAX_JSON_LAYERS):
        if not isinstance(value, str):
            break
        if not value.lstrip().startswith(("{", "[", '"')):
            break
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return original
    return value if isinstance(value, (dict, list)) else original


def tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """Canonical tool name, parsed arguments and identity; keep original evidence."""
    out = deepcopy(call)
    function = decode_container(call.get("function"))
    if isinstance(function, dict):
        function = deepcopy(function)
        if function != call["function"]:
            out["function_raw"] = call["function"]
        if "arguments" in function:
            decoded = decode_container(function["arguments"])
            if decoded != function["arguments"]:
                function.setdefault("arguments_raw", function["arguments"])
                function["arguments"] = decoded
        out["function"] = function
    function = function if isinstance(function, dict) else call
    name = call.get("function_name") or function.get("name") or call.get("name")
    args = call.get("arguments")
    if args is None:
        args = call.get("args")
    if args is None:
        args = function.get("arguments")
    if args is None:
        args = function.get("input")
    parsed = decode_container(args)
    out.update(name=str(name or "unknown"), arguments=parsed)
    if parsed != args:
        out.setdefault("arguments_raw", args)
    elif isinstance(function, dict) and "arguments_raw" in function:
        out.setdefault("arguments_raw", function["arguments_raw"])
    elif isinstance(args, str) and args.lstrip().startswith(("{", "[")):
        out["arguments_warning"] = "invalid or excessively encoded JSON arguments; raw retained"
    identity = call.get("call_id", call.get("id", call.get("tool_call_id")))
    if identity is not None:
        out["call_id"] = identity
    return out


def text_content(value: Any) -> tuple[str, list[str]]:
    """Render recognized content blocks while preserving opaque blocks elsewhere."""
    decoded = decode_container(value)
    # Only interpret encoded text as content blocks when its shape says so.
    if isinstance(value, str) and not (
        isinstance(decoded, list)
        and decoded
        and all(isinstance(p, dict) and p.get("type") in TEXT_TYPES for p in decoded)
    ):
        return value, []
    if decoded is None:
        return "", []
    if isinstance(decoded, dict):
        decoded = [decoded]
    if not isinstance(decoded, list):
        return "", ["unsupported content retained without text rendering"]
    texts, warnings = [], []
    for part in decoded:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") in TEXT_TYPES:
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
            else:
                warnings.append("text block has no text string")
        elif isinstance(part, dict) and part.get("type") in {
            "tool_use",
            "tool_result",
            "thinking",
            "redacted_thinking",
        }:
            continue  # Handled by message_steps; not ordinary message text.
        else:
            warnings.append("non-text content retained but not inspected")
    return "\n\n".join(texts), warnings


def messages(value: Any, *, role: str = "user") -> tuple[list[dict], list[str]]:  # noqa: C901
    """Parse Chat Completions, Responses and Anthropic message containers."""
    decoded = decode_container(value)
    if isinstance(value, str) and isinstance(decoded, (dict, list)):
        candidates = decoded if isinstance(decoded, list) else [decoded]
        if not any(
            isinstance(m, dict)
            and (
                "role" in m
                or m.get("type") in MESSAGE_TYPES
                or any(k in m for k in ("messages", "input", "instructions", "system"))
            )
            for m in candidates
        ):
            return [{"role": role, "content": value}], []
    if isinstance(decoded, str):
        # A serialized message array is decoded above; a literal prompt stays literal.
        return [{"role": role, "content": decoded}], []
    if decoded is None:
        return [], []
    if isinstance(decoded, dict):
        if any(key in decoded for key in ("messages", "input", "instructions", "system")) and (
            "role" not in decoded and decoded.get("type") not in MESSAGE_TYPES
        ):
            result, warnings = messages(
                decoded.get("messages")
                if decoded.get("messages") is not None
                else decoded.get("input"),
                role=role,
            )
            prefix = []
            for key in ("system", "instructions"):
                if decoded.get(key) is not None:
                    prefix.append({"role": "system", "content": decoded[key]})
            # Prefix content uses the same structural decoding as explicit messages.
            # Otherwise encoded Anthropic system blocks remain strings in the input
            # view even though later model-call expansion parses those same blocks.
            prefix, prefix_warnings = messages(prefix, role=role)
            return prefix + result, prefix_warnings + warnings
        decoded = [decoded]
    if not isinstance(decoded, list):
        return [], ["unsupported message container"]
    result, warnings = [], []
    for entry in decoded:
        entry = decode_container(entry)
        if isinstance(entry, str):
            result.append({"role": role, "content": entry})
            continue
        if not isinstance(entry, dict) or not (
            entry.get("role") in ROLES
            or entry.get("type") in MESSAGE_TYPES
            or ("content" in entry and "role" not in entry)
        ):
            warnings.append("unsupported message retained in raw source")
            continue
        if is_evaluation_record(entry):
            warnings.append("evaluation message excluded from execution evidence")
            continue
        # Message boundaries are already established. Preserve provider extensions
        # rather than projecting onto a fixed field list; external labels and transport
        # credentials remain excluded without rewriting opaque message content.
        message = sanitize_metadata(entry)
        kind = message.get("type")
        if (
            not kind
            and message.get("role") in {"assistant", "agent"}
            and all(key in message for key in ("name", "call_id", "arguments"))
        ):
            kind = message["type"] = "function_call"
        inferred = "assistant" if kind in {"function_call", "reasoning", "tool_use"} else role
        if kind in {"function_call_output", "tool_result"}:
            inferred = "tool"
        message["role"] = message.get("role") or inferred
        if message["role"] == "function":
            message["role"] = "tool"
        content = message.get("content")
        blocks = decode_container(content)
        if (
            isinstance(content, str)
            and isinstance(blocks, list)
            and blocks
            and all(isinstance(b, dict) and isinstance(b.get("type"), str) for b in blocks)
        ):
            message["content_raw"] = content
            message["content"] = blocks
        if kind in {"function_call", "tool_use"}:
            normalized = tool_call(message)
            for key in ("arguments", "arguments_raw", "arguments_warning"):
                if key in normalized:
                    message[key] = normalized[key]
        if kind == "function_call_output":
            output = message.get("output")
            decoded_output = decode_container(output)
            if decoded_output != output:
                message["output_raw"] = output
                message["output"] = decoded_output
        raw_calls = decode_container(message.get("tool_calls"))
        if raw_calls is not None:
            if not isinstance(raw_calls, list) or any(not isinstance(c, dict) for c in raw_calls):
                warnings.append("tool_calls must be an array of objects")
            else:
                message["tool_calls"] = [tool_call(c) for c in raw_calls]
        function = decode_container(message.get("function_call"))
        if isinstance(function, dict):
            message["function_call"] = tool_call(function)
        result.append(message)
    return result, warnings


def message_steps(message: dict[str, Any]) -> tuple[list[dict], list[str]]:  # noqa: C901
    """Convert a message to steps without dropping structured blocks or call IDs."""
    source = message.get("role") or "assistant"
    kind = message.get("type")
    content = message.get("content", message.get("text"))
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    if kind == "reasoning":
        reasoning = message.get("summary", content)
        content = None
    if kind in {"function_call_output", "tool_result"}:
        source = "tool"
        content = message.get("output", content)
    if isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    ):
        # Tool results and new user text can share one Anthropic message. Keep
        # their original order instead of moving all results after the user text.
        steps, warnings, pending = [], [], []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                if pending:
                    converted, issues = message_steps({**message, "content": pending})
                    steps.extend(converted)
                    warnings.extend(issues)
                    pending = []
                steps.append(
                    {
                        "source": "tool",
                        "call_id": block.get("tool_use_id"),
                        "observation": deepcopy(block.get("content")),
                        "is_error": block.get("is_error", False),
                        "metadata": sanitize_metadata({
                            key: value for key, value in block.items() if key != "content"
                        }),
                    }
                )
            else:
                pending.append(block)
        if pending:
            converted, issues = message_steps({**message, "content": pending})
            steps.extend(converted)
            warnings.extend(issues)
        return steps, warnings
    text, warnings = text_content(content)
    reason, reason_warnings = text_content(reasoning)
    warnings.extend(reason_warnings)
    step: dict[str, Any] = {"source": source}
    metadata = sanitize_metadata({
        key: value for key, value in message.items()
        if key not in {
            "source", "role", "from", "content", "content_raw", "text", "reasoning_content",
            "reasoning", "summary", "tool_calls", "function_call", "arguments", "arguments_raw",
            "arguments_warning", "input", "output", "output_raw",
        }
    })
    if metadata:
        step["metadata"] = metadata
    if text:
        step["observation" if source == "tool" else "message"] = text
    if content is not None and not isinstance(content, str):
        step["content"] = deepcopy(content)
        if source == "tool":
            step["observation"] = deepcopy(content)
    if reason:
        step["reasoning_content"] = reason
    if message.get("refusal"):
        step["refusal"] = message["refusal"]
    if message.get("encrypted_content"):
        warnings.append("encrypted reasoning unavailable")
    calls = decode_container(message.get("tool_calls")) or []
    if not isinstance(calls, list) or any(not isinstance(c, dict) for c in calls):
        raise ValueError("tool_calls must be an array of objects")
    calls = list(calls)
    function = decode_container(message.get("function_call"))
    if function is not None:
        if not isinstance(function, dict):
            raise ValueError("function_call must be an object")
        calls.append({"function": function})
    if kind in {"function_call", "tool_use"}:
        calls.append(message)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                calls.append(block)
            elif block.get("type") == "thinking":
                step["reasoning_content"] = "\n\n".join(
                    filter(
                        None,
                        [
                            step.get("reasoning_content"),
                            block.get("thinking"),
                        ],
                    )
                )
            elif block.get("type") == "redacted_thinking":
                warnings.append("encrypted reasoning unavailable")
    if calls:
        step["tool_calls"] = [tool_call(c) for c in calls]
    if source == "tool":
        step["call_id"] = message.get(
            "tool_call_id",
            message.get("call_id", message.get("tool_use_id")),
        )
    return [step], warnings


def observation_data(value: Any) -> Any:
    """Decode tool output and ATIF result wrappers, keeping original result content."""
    parsed = deepcopy(decode_container(value))
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        for result in parsed["results"]:
            if not isinstance(result, dict) or "source_call_id" not in result:
                continue
            content = result.get("content")
            decoded = decode_container(content)
            if decoded != content:
                result["content_raw"] = content
                result["content"] = decoded
    return parsed


def normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    """Normalize known step fields; preserve extensions and original values."""
    out = deepcopy(step)
    out["source"] = str(step.get("source") or step.get("role") or "assistant").lower()
    if out["source"] in {"environment", "function"}:
        out["source"] = "tool"
    calls = decode_container(step.get("tool_calls"))
    if calls is not None:
        if not isinstance(calls, list) or any(not isinstance(c, dict) for c in calls):
            raise ValueError("tool_calls must be an array of objects; refusing partial parse")
        out["tool_calls"] = [tool_call(c) for c in calls]
    for key in ("message", "reasoning_content", "reasoning", "thought"):
        if key in step:
            text, warnings = text_content(step[key])
            if not warnings and text != step[key]:
                out[key + "_raw"] = deepcopy(step[key])
                out[key] = text
    obs = step.get("observation")
    if obs is None and out["source"] == "tool":
        obs = step.get("message", step.get("content"))
    if obs is not None:
        out["observation"] = observation_data(obs)
        if out["observation"] != obs:
            out["observation_raw"] = obs
    return out


def input_view(
    value: Any,
    *,
    tools: Any = None,
    label: str = "Initial model request",
) -> dict[str, Any]:
    """Structured input plus a readable projection, without interpreting task text."""
    parsed, warnings = messages(value)
    blocks = []
    for message in parsed:
        text, issues = text_content(message.get("content", message.get("text")))
        warnings.extend(issues)
        if text:
            blocks.append(f"[{label}: {message['role']}]\n{text}")
    definitions = decode_container(tools)
    if definitions is None:
        definitions = []
    if not isinstance(definitions, list) or any(not isinstance(t, dict) for t in definitions):
        warnings.append("unsupported tool definitions retained in raw source")
        definitions = []
    definitions = deepcopy(definitions)
    for definition in definitions:
        function = decode_container(definition.get("function"))
        if isinstance(function, dict):
            if function != definition["function"]:
                definition["function_raw"] = definition["function"]
            definition["function"] = function
        schema = function if isinstance(function, dict) else definition
        for key in ("parameters", "input_schema"):
            if key in schema:
                decoded = decode_container(schema[key])
                if decoded != schema[key]:
                    schema[key + "_raw"] = schema[key]
                    schema[key] = decoded
    return {
        "messages": parsed,
        "text": "\n\n".join(blocks),
        "tools": definitions,
        "limitations": list(dict.fromkeys(warnings)),
    }
