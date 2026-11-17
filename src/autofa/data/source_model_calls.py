# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover model-visible evidence from streamed NeMo Gym model calls.

Request snapshots are retained separately from the event sequence: a repeated
conversation in a later request is not a second execution of those actions.
Preserve execution parameters/metadata, excluding external targets and transport credentials.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

import ijson

from autofa.data.execution_policy import (
    is_evaluation_record,
    mentions_evaluation_model,
    sanitize_metadata,
)
from autofa.data.json_stream import parse_json
from autofa.data.source_errors import SourceDataError as DataApiError
from autofa.data.source_execution import withhold_model_calls
from autofa.data.structured import decode_container, input_view, message_steps
from autofa.data.structured import messages as parse_messages

EXCLUDED_CALL = "evaluation model call excluded from execution evidence"
UNCAPTURED_RESPONSES = "provider response envelopes not recorded in event request captures"
_SCAN_BYTES = 1024**2


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON argument key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-JSON argument constant: {value}")


def _context_arguments(value: dict) -> dict:
    """Ignore only a verified redundant serialization, without changing the archive."""
    raw, parsed = value.get("arguments_raw"), value.get("arguments")
    if (isinstance(raw, str) and isinstance(parsed, (dict, list))
            and not value.get("arguments_warning")):
        try:
            decoded = json.loads(raw, object_pairs_hook=_unique_object,
                                 parse_constant=_invalid_constant)
            # Serialization equality preserves distinctions such as True versus 1;
            # never decode strings inside arguments or discard ambiguous raw JSON.
            if (json.dumps(decoded, sort_keys=True, allow_nan=False)
                    == json.dumps(parsed, sort_keys=True, allow_nan=False)):
                return {key: item for key, item in value.items() if key != "arguments_raw"}
        except (ValueError, RecursionError):
            pass
    return value


def _context_tool_call(call: dict) -> dict:
    compared = _context_arguments(call)
    if isinstance(call.get("function"), dict):
        compared = {**compared, "function": _context_arguments(call["function"])}
    return compared


def _context_identity(message: dict) -> str:
    """Compare normalized messages while retaining exact pooled snapshots separately.

    Tool argument JSON formatting is not a new conversation. Other raw fields,
    provider metadata, message text and tool identities still participate; arbitrary
    payload dictionaries with an ``arguments_raw`` key are not tool-call envelopes.
    """
    compared = dict(message)
    if isinstance(message.get("tool_calls"), list):
        compared["tool_calls"] = [
            _context_tool_call(call) if isinstance(call, dict) else call
            for call in message["tool_calls"]
        ]
    if isinstance(message.get("function_call"), dict):
        compared["function_call"] = _context_tool_call(message["function_call"])
    if message.get("type") in {"function_call", "tool_use"}:
        compared = _context_tool_call(compared)
    return hashlib.sha256(_json(compared).encode()).hexdigest()


def _messages(value: Any, limits: Counter, *, role: str) -> list[dict]:
    parsed, warnings = parse_messages(value, role=role)
    limits.update(warnings)
    return parsed


def _steps(message: dict, limits: Counter, *, role: str) -> list[dict]:
    try:
        steps, warnings = message_steps({"role": role, **message})
    except ValueError as exc:
        raise DataApiError(f"Raw model {exc}") from exc
    limits.update("raw " + w if w == "non-text content retained but not inspected" else w
                  for w in warnings)
    for step in steps:
        content = step.get("message", "")
        if step["source"] in {"assistant", "agent"} and isinstance(content, str):
            tagged = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
            if tagged and not step.get("reasoning_content"):
                step["reasoning_content"] = "\n\n".join(tagged)
        for call in step.get("tool_calls", []):
            call["function_name"] = call["name"]
            if call.get("arguments") is None:
                limits["raw tool arguments not recorded"] += 1
    return steps


def _captured_calls(events):
    """Yield one record's raw ``ng_trajectory.model_calls``, one call at a time."""
    from autofa.data.source_inputs import _project_value

    for prefix, event, value in events:
        if prefix in {"ng_trajectory", "ng_trajectory.model_calls"} and event == "string":
            decoded = decode_container(value)
            if prefix == "ng_trajectory" and isinstance(decoded, dict):
                decoded = decode_container(decoded.get("model_calls"))
            if not isinstance(decoded, list):
                raise DataApiError("Raw model_calls must be an array")
            yield from decoded
        elif prefix == "ng_trajectory.model_calls.item":
            yield _project_value(((e, v) for _, e, v in events), event, value, True)


def recover_model_calls(source: Path, offset: int) -> dict[str, Any]:
    """Read one validated record incrementally; deduplicate repeated contexts.

    The caller must validate the record identity with ``_read_input_record``
    before attaching this evidence. Memory includes the largest individual call
    or JSON scalar plus unique model-visible content, not all repeated histories.
    """
    from autofa.data.source_inputs import _JsonlRecordReader

    try:
        with source.open("rb") as handle:
            handle.seek(offset)
            records = _captured_calls(parse_json(
                _JsonlRecordReader(handle), with_prefix=True, use_float=True,
            ))
            return recover_calls(records)
    except (ijson.JSONError, UnicodeDecodeError, OverflowError) as exc:
        raise DataApiError("Raw model calls could not be read completely") from exc


def grader_call_ids(source: Path, offset: int) -> set[str]:
    """Model-call ids of the graders' calls captured in one raw JSONL record.

    Only the raw record names the model server behind each call. A record whose
    bytes name no grader's model reference is scanned but never parsed.
    """
    from autofa.data.source_inputs import _JsonlRecordReader

    try:
        with source.open("rb") as handle:
            handle.seek(offset)
            if not _names_grader(_JsonlRecordReader(handle)):
                return set()
            handle.seek(offset)
            calls = _captured_calls(parse_json(
                _JsonlRecordReader(handle), with_prefix=True, use_float=True,
            ))
            return {call["model_call_id"] for call in calls if isinstance(call, dict)
                    and isinstance(call.get("model_call_id"), str) and is_evaluation_record(call)}
    except (ijson.JSONError, UnicodeDecodeError, OverflowError) as exc:
        raise DataApiError("Raw model calls could not be read completely") from exc


def _names_grader(stream) -> bool:
    """Whether streamed JSON bytes can name a grader's model reference."""
    tail = b""
    while chunk := stream.read(_SCAN_BYTES):
        if mentions_evaluation_model(tail + chunk):
            return True
        tail = chunk[-1024:]  # A reference can span two reads.
    return False


def recover_calls(records) -> dict[str, Any]:  # noqa: C901
    """Normalize an iterable of captured calls, keeping repeated context separate.

    ``excluded_call_ids`` names the evaluation calls left out, whose exported
    copies callers withhold too.
    """
    limits: Counter[str] = Counter()
    notices: Counter[str] = Counter()
    messages: list[dict] = []
    tools: list[list] = []
    message_index: dict[str, int] = {}
    tool_index: dict[str, int] = {}
    calls, steps = [], []
    excluded: list[str] = []
    history: list[str] = []
    emitted_messages: set[str] = set()
    context_identities: dict[int, str] = {}

    def intern(value: Any, pool: list, index: dict) -> int:
        key = hashlib.sha256(_json(value).encode()).hexdigest()
        if key not in index:
            index[key] = len(pool)
            pool.append(value)
        return index[key]

    def context_identity(index: int) -> str:
        if index not in context_identities:
            context_identities[index] = _context_identity(messages[index])
        return context_identities[index]

    def append(indices: list[int], call_index: int, origin: str, role: str) -> None:
        for index in indices:
            for step in _steps(messages[index], limits, role=role):
                step.update(step_id=len(steps), model_call_index=call_index,
                            evidence_origin=origin, message_index=index)
                steps.append(step)
            emitted_messages.add(context_identity(index))

    for call_index, call in enumerate(records):
        if not isinstance(call, dict):
            limits["raw model call is not an object"] += 1
            continue
        if is_evaluation_record(call):
            limits[EXCLUDED_CALL] += 1
            if isinstance(call.get("model_call_id"), str):
                excluded.append(call["model_call_id"])
            continue
        request, response = (decode_container(call.get(k)) for k in ("request", "response"))
        request = request if isinstance(request, dict) else None
        response = response if isinstance(response, dict) else None
        evidence: dict[str, Any] = {
            "index": call_index, "request_recorded": request is not None,
            "response_recorded": response is not None,
        }
        evidence["metadata"] = sanitize_metadata({
            key: value for key, value in call.items() if key not in {"request", "response"}
        })
        if request is not None:
            evidence["request_metadata"] = sanitize_metadata({
                key: value for key, value in request.items()
                if key not in {"messages", "input", "instructions", "system", "tools"}
            })
            inputs = _messages(request, limits, role="user")
            ids = [intern(m, messages, message_index) for m in inputs]
            context = [context_identity(index) for index in ids]
            evidence["request"] = ids
            if "tools" in request:
                definitions = input_view(None, tools=request["tools"])
                limits.update(definitions["limitations"])
                if isinstance(decode_container(request["tools"]), list):
                    evidence["tools"] = intern(definitions["tools"], tools, tool_index)
            # Only a strict continuation is safe to turn into events.
            if context[:len(history)] == history:
                append(ids[len(history):], call_index, "request", "user")
            else:
                notices["model requests with changed context retained separately"] += 1
                # Recover new input messages, never replay old assistant actions.
                unseen = [i for i in ids if context_identity(i) not in emitted_messages and
                          messages[i].get("role") not in {"assistant", "agent"} and
                          messages[i].get("type") not in {"reasoning", "function_call"}]
                append(unseen, call_index, "request_context", "user")
            history = context
        else:
            limits["raw model request not recorded"] += 1
        output: list[dict] = []
        if response is not None:
            response_metadata = {
                key: value for key, value in response.items()
                if key not in {"output", "content", "choices"}
            }
            if isinstance(response.get("choices"), list):
                response_metadata["choices"] = [
                    {key: value for key, value in choice.items() if key != "message"}
                    if isinstance(choice, dict) else choice
                    for choice in response["choices"]
                ]
            evidence["response_metadata"] = sanitize_metadata(response_metadata)
            if "output" in response:
                output = _messages(response["output"], limits, role="assistant")
            elif isinstance(response.get("choices"), list):
                choices = response["choices"]
                if len(choices) > 1:
                    raise DataApiError(
                        "Multiple raw model response choices cannot be reconstructed "
                        "as one execution sequence"
                    )
                for choice in choices:
                    if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                        output.extend(_messages([choice["message"]], limits,
                                                role="assistant"))
            elif "content" in response or response.get("role") == "assistant":
                output = _messages(response, limits, role="assistant")
            else:
                limits["unsupported raw model response format"] += 1
            if not output:
                limits["raw model response has no captured message output"] += 1
        else:
            limits["raw model response not recorded"] += 1
        ids = [intern(m, messages, message_index) for m in output]
        evidence["response"] = ids
        append(ids, call_index, "response", "assistant")
        history.extend(context_identity(index) for index in ids)
        calls.append(evidence)
    pending: dict[str, list[tuple[dict, int, dict]]] = {}
    for step in steps:
        for index, call in enumerate(step.get("tool_calls", [])):
            identity = call.get("call_id")
            if isinstance(identity, str) and identity:
                pending.setdefault(identity, []).append((step, index, call))
            else:
                limits["raw tool result pairing unavailable"] += 1
        identity = step.get("call_id")
        if step["source"] == "tool" and isinstance(identity, str):
            candidates = pending.get(identity, [])
            if len(candidates) == 1:
                parent, call_index, call = candidates.pop()
                step["paired_event_idx"] = parent["step_id"]
                step["paired_call_idx"] = call_index
                call["result_event_idx"] = step["step_id"]
            else:
                limits["raw tool result pairing unavailable"] += 1
    missing = sum(len(candidates) for candidates in pending.values())
    if missing:
        limits["raw tool results not recorded"] = missing
    return {"steps": steps, "messages": messages, "tools": tools, "calls": calls,
            "limitations": dict(limits), "notices": dict(notices),
            "excluded_call_ids": excluded}


def _refs(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _call_id(call: Any) -> str | None:
    """The capture's model_call_id: raw calls keep it, event captures export it."""
    metadata = call.get("metadata") if isinstance(call, dict) else None
    if not isinstance(metadata, dict):
        return None
    if "source_event_index" in call:
        # Request snapshots also come from events, but not from model_call events.
        exported = "model_call" in (metadata.get("kind_raw"), metadata.get("lane_kind"))
        value = metadata.get("source_event_id") if exported else None
    else:
        value = metadata.get("model_call_id")
    return value if isinstance(value, str) else None


def drop_grader_calls(trace: dict[str, Any], call_ids: Collection[str] = ()) -> int:
    """Withhold graders' model calls from converted evidence; return how many calls.

    Raw calls are recognized by model reference and exported copies by model-call
    id; ``call_ids`` adds ids read from the raw record. Pool entries and steps that
    only those calls produced go too, as do their execution events and the request
    messages and tools only they sent. The policy model's calls keep their messages
    and tools, renumbered in order. The count includes every call ``call_ids``
    names, though an evaluation lane may already have excluded it.
    """
    ids = {value for value in call_ids if isinstance(value, str)}
    named = set(ids)
    withheld: set[Any] = set()
    evidence = trace.get("model_call_evidence")
    calls = evidence.get("calls") if isinstance(evidence, dict) else None
    if isinstance(calls, list):
        # A converted call keeps its captured provenance under "metadata".
        raw = [isinstance(call, dict) and (
            is_evaluation_record(call) or is_evaluation_record(call.get("metadata"))
        ) for call in calls]
        ids.update(_call_id(call) for call, grader in zip(calls, raw, strict=True)
                   if grader and _call_id(call))
        graders = [grader or _call_id(call) in ids
                   for call, grader in zip(calls, raw, strict=True)]
        if any(graders):
            withheld.update(_call_id(call) or position for position, (call, grader)
                            in enumerate(zip(calls, graders, strict=True)) if grader)
            _drop_calls(trace, evidence, graders)
    if isinstance(trace.get("execution"), dict):
        withheld.update(withhold_model_calls(trace["execution"], ids))
    return len(withheld | named)


def _drop_calls(trace: dict[str, Any], evidence: dict[str, Any], graders: list[bool]) -> None:
    calls = evidence["calls"]
    kept = [call for call, grader in zip(calls, graders, strict=True) if not grader]
    dropped = [call for call, grader in zip(calls, graders, strict=True)
               if grader and isinstance(call, dict)]
    # Only raw calls replay steps; event captures are never replayed.
    removed = {call["index"] for call in dropped
               if "source_event_index" not in call and type(call.get("index")) is int}
    referencing = [call for call in kept if isinstance(call, dict)]
    renumbered: dict[int, int] = {}
    for pool, sides in (("messages", ("request", "response")), ("tools", ("tools",))):
        values = evidence.get(pool)
        if not isinstance(values, list):
            continue
        # Invalid references stay invalid: the pool only becomes shorter.
        used = sorted({ref for call in referencing for side in sides
                       for ref in _refs(call.get(side))
                       if type(ref) is int and 0 <= ref < len(values)})
        index = {old: new for new, old in enumerate(used)}
        evidence[pool] = [values[old] for old in used]
        for call in referencing:
            for side in sides:
                value = call.get(side)
                if isinstance(value, list):
                    call[side] = [index.get(ref, ref) if type(ref) is int else ref
                                  for ref in value]
                elif type(value) is int:
                    call[side] = index.get(value, value)
        if pool == "messages":
            renumbered = index
    evidence["calls"] = kept
    if isinstance(trace.get("steps"), list):
        retained = []
        for step in trace["steps"]:
            origin = step.get("model_call_index") if isinstance(step, dict) else None
            if type(origin) is int:
                if origin in removed:
                    continue
                if type(step.get("message_index")) is int:
                    step["message_index"] = renumbered.get(
                        step["message_index"], step["message_index"],
                    )
            retained.append(step)
        trace["steps"] = retained
    scope = trace.get("evidence_scope")
    notices = scope.get("notices") if isinstance(scope, dict) else None
    captured = sum("source_event_index" in call for call in dropped)
    if captured and isinstance(notices, dict) and type(notices.get(UNCAPTURED_RESPONSES)) is int:
        # Counted per event capture; withheld captures no longer lack a response.
        notices[UNCAPTURED_RESPONSES] -= captured
        if notices[UNCAPTURED_RESPONSES] <= 0:
            del notices[UNCAPTURED_RESPONSES]
    if not kept:
        del trace["model_call_evidence"]
        if isinstance(scope, dict):
            scope.pop("model_calls", None)


def count_excluded_calls(trace: dict[str, Any], count: int) -> None:
    """Record withheld evaluation calls, so their absence is not read as missing data."""
    scope = trace.get("evidence_scope")
    limits = scope.get("limitations") if isinstance(scope, dict) else None
    if count and isinstance(limits, dict) and type(limits.get(EXCLUDED_CALL, 0)) is int:
        limits[EXCLUDED_CALL] = limits.get(EXCLUDED_CALL, 0) + count


def recover_event_requests(
    events: list[dict], *, lane: str | None = None,
    prompt_messages: list[dict] | None = None, tool_definitions: list[dict] | None = None,
) -> dict[str, Any]:
    """Expose EventText request captures without replaying their conversation history.

    EventText does not capture provider response envelopes. Responses remain in
    the execution event sequence; no synthetic response message is invented.
    Lane/event identity is retained because event indices can repeat across lanes.
    """
    from autofa.data.source_request_references import RequestReferenceResolver

    resolver = RequestReferenceResolver(prompt_messages or [], tool_definitions or [])
    limits: Counter[str] = Counter()
    notices: Counter[str] = Counter()
    messages: list[dict] = []
    tools: list[list] = []
    message_index: dict[str, int] = {}
    tool_index: dict[str, int] = {}
    calls: list[dict] = []
    output_seen: set[str] = set()
    identities: set[tuple[str, int]] = set()

    def intern(value: Any, pool: list, index: dict) -> int:
        key = hashlib.sha256(_json(value).encode()).hexdigest()
        if key not in index:
            index[key] = len(pool)
            pool.append(value)
        return index[key]

    ordered = sorted(events, key=lambda row: (row.get("lane") or "", row["event_idx"]))
    for event in ordered:
        event_lane = event.get("lane") or ""
        if (lane is not None and event_lane != lane) or is_evaluation_record(event):
            continue
        identity = (event_lane, event["event_idx"])
        if identity in identities:
            raise DataApiError("Duplicate lane/event identity in recorded model requests")
        identities.add(identity)
        captured = event.get("event_text")
        if isinstance(captured, dict) and any(
            captured.get(key) is not None for key in
            ("request_messages", "request_tools", "request_params")
        ):
            # Resolve only this request while interning it. The execution events
            # keep their compact references rather than expanded repeated histories.
            captured = resolver.resolve(captured)
            recorded = captured.get("request_messages") is not None
            parsed = _messages(captured.get("request_messages"), limits, role="user")
            definitions = input_view(None, tools=captured.get("request_tools"))
            limits.update(definitions["limitations"])
            params = decode_container(captured.get("request_params"))
            if params is not None and not isinstance(params, dict):
                raise DataApiError("Recorded model request parameters must be an object")
            metadata = sanitize_metadata({
                key: value for key, value in event.items() if key not in {"parts", "event_text"}
            })
            if captured.get("request_dialect") is not None:
                metadata["request_dialect"] = captured["request_dialect"]
            if isinstance(captured.get("request_params"), str):
                metadata["request_params_raw"] = sanitize_metadata({
                    "request_params": captured["request_params"],
                })["request_params"]
            prior_history = any(
                message.get("role") in {"assistant", "agent", "tool", "function"}
                for message in parsed
            )
            calls.append({
                "index": len(calls), "source_event_index": event["event_idx"],
                "source_lane": event_lane, "request_recorded": recorded,
                "response_recorded": False,
                "initial_request_recorded": (
                    recorded and event_lane not in output_seen and not prior_history
                ),
                "request": [intern(message, messages, message_index) for message in parsed],
                "response": [], "tools": intern(definitions["tools"], tools, tool_index),
                "request_metadata": sanitize_metadata(params or {}), "metadata": metadata,
            })
            if not recorded:
                limits["event model request messages not recorded"] += 1
        if event.get("role") in {"assistant", "agent", "output"}:
            output_seen.add(event_lane)
    if calls:
        notices[UNCAPTURED_RESPONSES] += len(calls)
    return {"messages": messages, "tools": tools, "calls": calls,
            "limitations": dict(limits), "notices": dict(notices)}
