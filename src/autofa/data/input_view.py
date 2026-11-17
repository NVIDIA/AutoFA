# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover a consistent model-input view from stored rows and captured requests."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

from autofa.data.structured import decode_container, input_view

PROMPT_FIELDS = ("prompt_text", "question", "instruction", "prompt", "input", "user_message")
RESPONSE_FIELDS = ("response_text", "generation", "trace", "messages", "conversation")
_CACHE = threading.local()
COMPACT_INPUT_ORIGIN = "frontier_compact_export"


def compact_export_input_json(
    item: dict[str, Any], *, response_is_literal: bool = False,
) -> str:
    """Record the export's literal input fields without guessing message structure."""
    fields = {}
    for key in ("system_text", "prompt_text"):
        value = item.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Compact export {key} must be text")
        fields[key] = value or ""
    return json.dumps({
        "autofa_input_origin": COMPACT_INPUT_ORIGIN,
        "response_is_literal": response_is_literal,
        **fields,
    }, ensure_ascii=False)


def has_compact_export_provenance(path: Path) -> bool:
    """Require an explicit export sidecar; prepared traces carry their own input view."""
    path = Path(path)
    try:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if isinstance(manifest, dict) and "adapter_version" in manifest:
            return False
    except (OSError, ValueError):
        pass
    for sidecar in (path.with_suffix(".source.json"), path.parent / "source.json"):
        try:
            source = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(source, dict):
            continue
        run_id = source.get("run_id")
        if isinstance(run_id, str) and run_id and source.get("base_url"):
            if source.get("export_path") == f"/v1/runs/{quote(run_id, safe='')}/export/autofa":
                return True
    return False


def attach_compact_export_input(item: dict[str, Any]) -> None:
    """Attach declared export input to a loaded row, preserving stored captures."""
    if item.get("input_json"):
        parse = item.get("input_parse")
        item["input_parse"] = {
            **(parse if isinstance(parse, dict) else {}),
            "response_origin": COMPACT_INPUT_ORIGIN,
            "response_is_literal": True,
        }
    elif any(k in item for k in ("system_text", "prompt_text")):
        item["input_json"] = compact_export_input_json(item, response_is_literal=True)


def has_literal_export_response(item: dict[str, Any]) -> bool:
    """Only an explicit loader declaration changes capture precedence."""
    parse = item.get("input_parse")
    return (
        isinstance(parse, dict)
        and parse.get("response_origin") == COMPACT_INPUT_ORIGIN
        and parse.get("response_is_literal") is True
    )


def _compact_input_view(request: dict[str, Any], *, tools: Any = None) -> dict[str, Any]:
    # Do not run literal content through messages(): a user may be asking about
    # JSON that itself contains role/messages/type keys.
    view = input_view(None, tools=tools)
    messages = []
    for role, key in (("system", "system_text"), ("user", "prompt_text")):
        value = request.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Compact export {key} must be text")
        if value:
            messages.append({"role": role, "content": value})
    view["messages"] = messages
    view["text"] = "\n\n".join(
        f"[Exported input: {message['role']}]\n{message['content']}" for message in messages
    )
    view["limitations"].append("exact initial model request not recorded")
    if not request.get("prompt_text"):
        view["limitations"].append("task input unavailable")
    return {**view, "origin": COMPACT_INPUT_ORIGIN}


def response_value(item: dict[str, Any]) -> Any:
    for key in RESPONSE_FIELDS:
        value = item.get(key)
        if isinstance(value, (str, dict, list)) and value:
            return value
    return ""


def _leading_trace_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    from autofa.data.trajectory import parse_trajectory

    leading = []
    for step in parse_trajectory(response_value(item))._steps:
        role = step.get("source")
        if role not in {"system", "developer", "user"}:
            break
        leading.append({"role": role, "content": step.get("message_raw", step.get("message", ""))})
    return leading


def _is_agent_input_call(call: dict[str, Any], trace: dict[str, Any]) -> bool:
    """Do not mistake an auxiliary model's first request for the agent's input."""
    metadata = call.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    scope = trace.get("evidence_scope")
    scope = scope if isinstance(scope, dict) else {}
    lane = call.get("source_lane", metadata.get("lane"))
    if lane is not None and lane != scope.get("event_lane", ""):
        # Normalized traces already promote the selected lane's captures to
        # initial_request. Other lanes may include user simulators or workers.
        return False
    actor = metadata.get("actor")
    agent = trace.get("agent")
    agent = agent if isinstance(agent, dict) else {}
    return not actor or (isinstance(actor, str) and actor in (
        "main", "agent", "assistant", "harness", agent.get("name"),
    ))


def parsed_input(item: dict[str, Any]) -> dict[str, Any]:
    """Prefer exact first requests; identify fallbacks instead of inventing context."""
    key = (
        response_value(item),
        item.get("input_json"),
        "literal export response" if has_literal_export_response(item) else None,
        item.get("system_text"),
        item.get("tools"),
        *(item.get(k) for k in PROMPT_FIELDS),
    )
    cacheable = all(v is None or isinstance(v, str) for v in key)
    if cacheable and getattr(_CACHE, "key", None) == key:
        return deepcopy(_CACHE.view)
    view = _parsed_input(item)
    if cacheable:
        _CACHE.key, _CACHE.view = key, deepcopy(view)
    return view


def _parsed_input(item: dict[str, Any]) -> dict[str, Any]:
    stored_input = decode_container(item.get("input_json"))
    if (isinstance(stored_input, dict)
            and stored_input.get("autofa_input_origin") == COMPACT_INPUT_ORIGIN
            and stored_input.get("response_is_literal") is True):
        # The compact response is model output, even if it resembles a capture.
        return _compact_input_view(stored_input, tools=item.get("tools"))
    if has_literal_export_response(item):
        if isinstance(stored_input, dict):
            return {
                **input_view(stored_input, tools=stored_input.get("tools")),
                "origin": "input_json",
            }
        return _compact_input_view(item, tools=item.get("tools"))
    blob = decode_container(response_value(item))
    if isinstance(blob, list) and len(blob) == 1 and isinstance(blob[0], dict):
        blob = blob[0]
    tools = item.get("tools")
    if isinstance(blob, dict):
        agent = blob.get("agent")
        if isinstance(agent, dict):
            tools = agent.get("tool_definitions", tools)
        request = blob.get("initial_request")
        if isinstance(request, dict):
            view = input_view(request, tools=request.get("tools", tools))
            view["limitations"].extend(request.get("limitations", []))
            return {**view, "origin": "captured_initial_request"}
        evidence = blob.get("model_call_evidence")
        if isinstance(evidence, dict):
            calls = evidence.get("calls") or []
            if (
                calls
                and calls[0].get("index") == 0
                and calls[0].get("request_recorded")
                and calls[0].get("initial_request_recorded", True)
                and _is_agent_input_call(calls[0], blob)
            ):
                call = calls[0]
                pool = evidence.get("messages", [])
                definitions = evidence.get("tools", [])
                try:
                    indices = call.get("request", [])
                    if any(type(i) is not int or i < 0 for i in indices):
                        raise IndexError
                    request = [pool[i] for i in indices]
                    tool_index = call.get("tools")
                    if tool_index is not None:
                        if type(tool_index) is not int or tool_index < 0:
                            raise IndexError
                        tools = definitions[tool_index]
                    view = input_view(request, tools=tools)
                    return {**view, "origin": "captured_initial_request"}
                except (IndexError, KeyError, TypeError):
                    return {
                        **input_view(None),
                        "origin": "model_call_evidence",
                        "limitations": ["invalid initial request pool references"],
                    }
        agent_request = blob.get("agent_request")
        if isinstance(agent_request, dict):
            view = input_view(
                agent_request, tools=agent_request.get("tools"), label="Recorded agent request",
            )
            view["limitations"].extend(agent_request.get("limitations", []))
            return {**view, "origin": "recorded_agent_request"}
    if isinstance(stored_input, dict) and stored_input:
        if stored_input.get("autofa_input_origin") == COMPACT_INPUT_ORIGIN:
            # A source-export fallback without a task must not hide leading
            # input turns recovered from the raw trajectory.
            # Literal summary-only exports already returned above.
            if not stored_input.get("prompt_text"):
                leading = _leading_trace_messages(item)
                if leading:
                    system = stored_input.get("system_text")
                    if system and not any(
                        m["role"] == "system" and m["content"] == system for m in leading
                    ):
                        leading.insert(0, {"role": "system", "content": system})
                    view = input_view(leading, tools=tools)
                    view["limitations"].append("exact initial model request not recorded")
                    return {**view, "origin": "leading_trace_messages"}
            return _compact_input_view(stored_input, tools=tools)
        return {**input_view(stored_input, tools=stored_input.get("tools")), "origin": "input_json"}
    system = item.get("system_text")
    system = system if isinstance(system, str) and system.strip() else ""
    for field in PROMPT_FIELDS:
        value = item.get(field)
        if isinstance(value, (str, dict, list)) and value:
            view = input_view(value, tools=tools)
            # Preserve distinct system instructions beside structured context.
            # Only an identical message in the same role is a duplicate.
            if system and not any(
                m.get("role") == "system" and m.get("content") == system
                for m in view["messages"]
            ):
                combined = input_view(
                    [{"role": "system", "content": system}, *view["messages"]],
                    tools=view["tools"], label="Exported input",
                )
                combined["limitations"].extend(view["limitations"])
                combined["limitations"].append("exact initial model request not recorded")
                return {**combined, "origin": f"system_text+{field}"}
            # Ordinary task text is already readable; preserve it byte-for-byte.
            if view["messages"] == [{"role": "user", "content": value}]:
                view["text"] = value if isinstance(value, str) else view["text"]
            return {**view, "origin": field}
    # Leading input turns are evidence, but are not necessarily a full request capture.
    leading = _leading_trace_messages(item)
    if not leading and system:
        view = input_view(
            [{"role": "system", "content": system}], tools=tools, label="Exported input",
        )
        return {
            **view, "origin": "system_text",
            "limitations": view["limitations"] + [
                "exact initial model request not recorded", "task input unavailable",
            ],
        }
    view = input_view(leading, tools=tools)
    return {
        **view,
        "origin": "leading_trace_messages" if leading else "unavailable",
        "limitations": view["limitations"]
        + (["exact initial model request not recorded"] if leading else ["task input unavailable"]),
    }


def derive_input_fields(item: dict[str, Any]) -> dict[str, Any]:
    view = parsed_input(item)
    parse = {"origin": view["origin"], "limitations": view["limitations"]}
    if has_literal_export_response(item):
        parse.update(response_origin=COMPACT_INPUT_ORIGIN, response_is_literal=True)
    return {
        **item,
        "input_text": view["text"],
        "input_messages": view["messages"],
        "input_tools": view["tools"],
        "input_parse": parse,
    }
