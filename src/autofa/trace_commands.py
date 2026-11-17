# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative command-prefix extraction; no shell interpretation or execution."""

from __future__ import annotations

import json
import math
from typing import Any

from autofa.data.input_view import response_value
from autofa.data.structured import decode_container
from autofa.data.trajectory import extract_steps_list
from autofa.trace_types import CommandAnchor, CommandFragment, CommandPrefix


class _Unavailable(ValueError):
    pass


def _check_metadata(value: Any) -> None:
    if not isinstance(value, dict):
        return
    if (any(value.get(key) for key in ("truncated", "length_truncated", "content_truncated",
                                       "missing", "incomplete"))
            or value.get("complete") is False or value.get("prefix_complete") is False):
        raise _Unavailable("Recorded command-prefix content is incomplete or truncated")
    if value.get("finish_reason") == "length":
        raise _Unavailable("An assistant command event ended at its output limit")


def _check_count(record: Any, key: str, actual: int) -> None:
    if not isinstance(record, dict):
        return
    expected = record.get(key)
    # Optional source counts can be null; a declared count must describe every
    # retained record, including records omitted from an otherwise present list.
    if expected is not None and (type(expected) is not int or expected != actual):
        raise _Unavailable(f"An assistant event's recorded {key} does not match retained records")


def _check_scope(scope: Any) -> None:
    if scope is None:
        return
    if not isinstance(scope, dict):
        raise _Unavailable("Malformed recorded evidence scope")
    _check_metadata(scope)
    limitations = scope.get("limitations", {})
    if isinstance(limitations, dict):
        if any(not isinstance(label, str) or not label.strip()
               or not isinstance(count, (int, float)) or count < 0
               or (isinstance(count, float)
                   and (not math.isfinite(count) or not count.is_integer()))
               for label, count in limitations.items()):
            raise _Unavailable("Malformed recorded evidence limitations")
        labels = [label for label, count in limitations.items() if count]
    elif isinstance(limitations, list):
        labels = limitations
    else:
        raise _Unavailable("Malformed recorded evidence limitations")
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise _Unavailable("Malformed recorded evidence limitation")
        label = label.lower()
        # These are importer-declared limitations, never words in command text.
        # Missing requests/reasoning/artifacts do not imply missing command events.
        if (any(noun in label for noun in ("event", "command", "trajectory", "trace", "step",
                                           "tool call", "tool_call"))
                and any(word in label for word in (
                    "missing", "omitted", "gap", "truncat", "unavailable", "excluded",
                    "unsupported", "incomplete",
                ))):
            raise _Unavailable("Recorded evidence limitations leave a command-prefix gap")


def _command_fields(value: Any, path: str) -> list[tuple[str, str]]:
    """Only explicit command fields count; never mine prose or reason about shell syntax."""
    if not isinstance(value, dict):
        raise _Unavailable(f"Unsupported command structure at {path}")
    _check_metadata(value)
    keys = [key for key in ("keystrokes", "command", "cmd") if key in value]
    if "commands" in value:
        if keys or not isinstance(value["commands"], list):
            raise _Unavailable(f"Ambiguous or malformed command batch at {path}")
        return [entry for index, command in enumerate(value["commands"])
                for entry in _command_fields(command, f"{path}.commands[{index}]")]
    if len(keys) != 1 or not isinstance(value[keys[0]], str):
        raise _Unavailable(f"Missing or ambiguous command text at {path}")
    key = keys[0]
    return [(f"{path}.{key}", value[key])]


def _event_commands(step: dict[str, Any]) -> list[tuple[str, str]]:
    if not isinstance(step, dict):
        raise _Unavailable("Malformed assistant command event")
    _check_metadata(step)
    _check_metadata(step.get("metadata"))
    parts = step.get("parts", [])
    if not isinstance(parts, list) or any(not isinstance(part, dict) for part in parts):
        raise _Unavailable("Malformed assistant event parts")
    for record in (step, step.get("metadata")):
        _check_count(record, "n_parts", len(parts))
    for part in parts:
        _check_metadata(part)
        if part.get("part_type", part.get("type")) in {
            "text", "output_text",
        }:
            text, chars = part.get("text"), part.get("chars")
            if not isinstance(text, str) or (type(chars) is int and chars > len(text)):
                raise _Unavailable("An assistant text part is unavailable or truncated")
    message = step.get("message", "")
    decoded = decode_container(message)
    message_commands = None
    if isinstance(decoded, dict) and any(
        key in decoded for key in ("commands", "keystrokes", "command", "cmd")
    ):
        message_commands = _command_fields(decoded, "message")
    elif isinstance(decoded, str):
        if decoded.lstrip().startswith(("{", "[", "```")) or '"commands"' in decoded:
            # A complete non-command JSON object is harmless; invalid JSON or
            # unrecognized structured command text cannot establish an empty prefix.
            try:
                parsed = json.loads(decoded)
            except (ValueError, RecursionError) as exc:
                raise _Unavailable("Unparseable structured assistant message") from exc
            if not isinstance(parsed, dict):
                raise _Unavailable("Unsupported structured assistant message")
    elif decoded not in (None, "") and not isinstance(decoded, dict):
        raise _Unavailable("Unsupported assistant message content")

    calls = step.get("tool_calls", [])
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise _Unavailable("Malformed assistant tool calls")
    for record in (step, step.get("metadata")):
        _check_count(record, "n_tool_calls", len(calls))
    call_commands = []
    for index, call in enumerate(calls):
        _check_metadata(call)
        if call.get("arguments_warning"):
            raise _Unavailable("A tool call has incomplete or unparsed arguments")
        full, chars = call.get("args_full"), call.get("args_chars")
        if isinstance(full, str) and type(chars) is int and len(full) != chars:
            raise _Unavailable("A tool call's recorded argument length does not match")
        call_commands.extend(_command_fields(call.get("arguments"),
                                             f"tool_calls[{index}].arguments"))
    if message_commands is not None and calls:
        if [text for _, text in message_commands] != [text for _, text in call_commands]:
            raise _Unavailable("Assistant message and tool-call command records conflict")
        # Identical mirrored exports are one occurrence, not an ambiguous anchor.
        return message_commands
    return message_commands if message_commands is not None else call_commands


def command_prefix(
    item: dict[str, Any], *, step_index: int, command_quote: str,
) -> CommandPrefix:
    """Recorded assistant commands strictly before a unique exact quote in one step.

    step_index is 1-based, as in steps(item). Recognizes JSON assistant messages
    with commands[] and tool-call argument objects with commands[], keystrokes,
    command or cmd. Includes earlier commands in the SAME event and the text before
    the quote inside its command field (e.g. the copy in 'cp db backup; mutate db').
    Identical message/tool-call mirrors are counted once. No whitespace normalization,
    text cap, shell parsing, or model-request-history replay is performed.

    Check complete before using commands: missing/repeated anchors, unsupported calls,
    malformed commands, conflicting mirrors and recorded gaps return unavailable with
    an empty command list. Invalid argument types raise ValueError. Completeness refers
    to this recorded assistant-command prefix, not unrecorded actions or other lanes.
    List order is recorded order, not proof of execution order (parallel calls, shell
    conditionals, scripts, aliases and quoted text need separate interpretation).
    A backup command in the prefix does NOT prove it ran or copied the required data;
    inspect its result and the relevant state before accepting recovery or protection.
    """
    if type(step_index) is not int or step_index < 1:
        raise ValueError("step_index must be a positive 1-based integer")
    if not isinstance(command_quote, str) or not command_quote.strip():
        raise ValueError("command_quote must be a nonempty exact command quote")
    try:
        blob = response_value(item) if isinstance(item, dict) else None
        events, envelope = extract_steps_list(blob)
        if not events or step_index > len(events):
            raise _Unavailable("The requested step is unavailable")
        envelope = envelope or {}
        _check_metadata(envelope)
        _check_scope(envelope.get("evidence_scope"))
        if envelope.get("_parse_limitations"):
            raise _Unavailable("Trajectory parsing recorded omissions")
        target = events[step_index - 1]
        if target.get("source") not in {"assistant", "agent"}:
            raise _Unavailable("The requested step is not an assistant command event")
        raw_steps = decode_container(envelope.get("steps", events))
        before: list[CommandFragment] = []
        for index, step in enumerate(events[:step_index], 1):
            raw_step = raw_steps[index - 1]
            if not raw_step.get("source") and not raw_step.get("role"):
                raise _Unavailable("An earlier event has no recorded actor")
            if step.get("source") not in {
                "assistant", "agent", "system", "developer", "user", "tool",
            }:
                raise _Unavailable("An earlier event has an unsupported actor")
            if step.get("source") not in {"assistant", "agent"}:
                continue
            commands = _event_commands(step)
            if index < step_index:
                before.extend({"step_index": index, "path": path, "text": text}
                              for path, text in commands)
                continue
            matches = []
            for ci, (_, text) in enumerate(commands):
                offset = text.find(command_quote)
                if offset >= 0:
                    if matches or text.find(command_quote, offset + 1) >= 0:
                        raise _Unavailable("Command quote is repeated in the requested step")
                    matches.append((ci, offset))
            if len(matches) != 1:
                raise _Unavailable("Command quote must occur exactly once in the requested step")
            ci, offset = matches[0]
            before.extend({"step_index": index, "path": path, "text": text}
                          for path, text in commands[:ci])
            path, text = commands[ci]
            if offset:
                before.append({"step_index": index, "path": path, "text": text[:offset]})
            anchor: CommandAnchor = {"step_index": index, "path": path, "start": offset,
                                     "end": offset + len(command_quote), "quote": command_quote}
            return {"status": "ok", "complete": True, "commands": before,
                    "anchor": anchor, "reason": ""}
        raise _Unavailable("The requested command is unavailable")
    except (ValueError, TypeError, KeyError) as exc:
        return {"status": "unavailable", "complete": False, "commands": [],
                "anchor": None, "reason": str(exc)}
