# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered, lossless trace views and verified passage-to-event navigation.

Display positions are one-based. Source event/part IDs are independent identities,
never inferred from display positions. Combined detector fields remain compatible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from autofa.data.input_view import RESPONSE_FIELDS, response_value
from autofa.data.structured import decode_container, text_content
from autofa.data.trajectory import (
    _args_to_str,
    _obs_to_str,
    _step_reasoning,
    extract_tool_call,
    format_actions,
    parse_trajectory,
)

NAVIGATION_VERSION = 3


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_highlight_context(text, start, end, others=()):
    """Native text-fragment context that identifies this occurrence, or an explicit refusal."""
    quote = text[start:end]
    selected = quote.strip()
    if not selected:
        return {"unavailable": True}
    expected = (start + len(quote) - len(quote.lstrip()), end - len(quote) + len(quote.rstrip()))
    pattern = lambda value: r"\s+".join(re.escape(w) for w in value.strip().split())  # noqa: E731
    for width in (0, 16, 32, 64, 128, 256):
        prefix, suffix = text[max(0, start - width):start], text[end:end + width]
        body = (
            ("(" + pattern(prefix) + r"\s*)" if prefix.strip() else "()")
            + "(" + pattern(selected) + ")"
            + (r"(?=\s*" + pattern(suffix) + ")" if suffix.strip() else "")
        )
        # Count overlapping candidates too: consuming the prefix could hide a
        # second eligible occurrence inside that prefix in highly repetitive text.
        expression = re.compile("(?=" + body + ")", re.IGNORECASE)
        found = expression.finditer(text)
        first, second = next(found, None), next(found, None)
        if (first is not None and first.span(2) == expected and second is None
                and not any(expression.search(other) for other in others)):
            if width:
                return {"prefix": prefix, "suffix": suffix}
            # Keep long verified selections whole; a start/end abbreviation could
            # finish on an earlier repeated suffix inside the selected passage.
            return {} if len(selected) > 300 and len(selected.split()) > 16 else None
    # Repeated identical context cannot name a single occurrence in this API.
    # Preserve the event link and the report's exact offsets, without a wrong highlight.
    return {"unavailable": True}


def _verified_source_identity(scope: Any) -> bool:
    """Accept declared schema identities and legacy cloud caches, never raw fallback IDs."""
    return (isinstance(scope, dict) and not scope.get("trajectory_source")
            and (scope.get("identity") == "evalrdl_tables_v1"
                 or scope.get("source") == "original API event parts and tool arguments"))


def _json_children(text: str, start: int):
    """Yield immediate JSON members with original character spans.

    Only structural boundaries are traversed; strings containing JSON stay strings.
    The standard decoder validates values and handles escaped keys/Unicode.
    """
    decoder = json.JSONDecoder()
    while text[start].isspace():
        start += 1
    kind = text[start]
    if kind not in "[{":
        return
    pos, index = start + 1, 0
    while True:
        while text[pos].isspace():
            pos += 1
        if text[pos] in "]}":
            return
        key = index
        if kind == "{":
            key, pos = decoder.raw_decode(text, pos)
            while text[pos].isspace():
                pos += 1
            if text[pos] != ":":
                raise ValueError("invalid JSON member")
            pos += 1
        while text[pos].isspace():
            pos += 1
        value, end = decoder.raw_decode(text, pos)
        yield key, value, pos, end
        pos = end
        while text[pos].isspace():
            pos += 1
        if text[pos] == ",":
            pos += 1
        index += 1


def _input_json_messages(text: str):
    """Yield message roles, complete text and its serialized content spans.

    Traverse only request/message boundaries. Task text, tool definitions and
    arbitrary metadata never become messages just because they contain JSON.
    """
    def unique_object(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("ambiguous JSON member")
        return obj

    root = json.loads(text, object_pairs_hook=unique_object)

    def content(value, lo, hi):
        if isinstance(value, str):
            return [(value, lo + 1, hi - 1)]
        if isinstance(value, dict):
            blocks = [(None, value, lo, hi)]
        elif isinstance(value, list):
            blocks = _json_children(text, lo)
        else:
            return []
        spans = []
        for _, block, start, end in blocks:
            if isinstance(block, str):
                spans.append((block, start + 1, end - 1))
            elif (isinstance(block, dict) and block.get("type") in
                  {"text", "input_text", "output_text"} and isinstance(block.get("text"), str)):
                spans.extend((value, a + 1, b - 1) for key, value, a, b in
                             _json_children(text, start) if key == "text")
            else:
                return []  # Partial/non-text content cannot establish whole-message identity.
        return spans

    def message(role, value, lo, hi):
        spans = content(value, lo, hi)
        if spans and role in {"system", "developer", "user", "assistant", "agent"}:
            yield role, "\n\n".join(s[0] for s in spans), spans

    def messages(value, lo, hi, *, request=True):
        if isinstance(value, str):
            yield from message("user", value, lo, hi)
        elif isinstance(value, list):
            for _, entry, a, b in _json_children(text, lo):
                yield from messages(entry, a, b, request=False)
        elif isinstance(value, dict):
            if value.get("type") not in {None, "message"}:
                return
            members = {key: (v, a, b) for key, v, a, b in _json_children(text, lo)}
            role = value.get("role")
            if role is not None or value.get("type") == "message" or "content" in value:
                field = "content" if "content" in members else "text"
                if field in members:
                    yield from message(role or "user", *members[field])
            elif request:
                for field in ("system", "instructions"):
                    if field in members:
                        yield from message("system", *members[field])
                field = "messages" if value.get("messages") is not None else "input"
                if field in members:
                    yield from messages(*members[field], request=False)

    yield from messages(root, len(text) - len(text.lstrip()), len(text.rstrip()))


def trace_events(item: dict[str, Any], *, include_content: bool = True) -> list[dict[str, Any]]:
    """Content and relationships in execution order; no outcome/grader metadata."""
    blob = response_value(item)
    parsed = parse_trajectory(blob)
    if not parsed.ok:
        return []
    envelope = decode_container(blob)
    if isinstance(envelope, list) and len(envelope) == 1:
        envelope = envelope[0]
    scope = envelope.get("evidence_scope", {}) if isinstance(envelope, dict) else {}
    scope = scope if isinstance(scope, dict) else {}
    # v8 caches already preserve API IDs in step_id. Raw fallback steps use a
    # different numbering system and must never inherit API event links.
    api_events = _verified_source_identity(scope)
    events = []
    for index, step in enumerate(parsed._steps, 1):
        event: dict[str, Any] = {
            "index": index, "step_id": step.get("step_id"),
            "source": step.get("source", ""), "sections": [],
            "tool_calls": step.get("tool_calls") or [],
        }
        if api_events and type(step.get("step_id")) is int and step["step_id"] >= 0:
            event["source_event"] = {
                "event_idx": step["step_id"], "lane": scope.get("event_lane", ""),
            }
        for key in ("paired_event_idx", "paired_call_idx", "call_id", "model_call_index"):
            if step.get(key) is not None:
                event[key] = step[key]

        def section(key: str, label: str, text: str, event=event) -> None:
            if text:
                event["sections"].append({"key": key, "label": label, "text": text})

        if include_content:
            section("reasoning", "Reasoning", _step_reasoning(step))
            message = step.get("message")
            section("message", "Message", message if isinstance(message, str) else "")
            for call_index, call in enumerate(event["tool_calls"]):
                name, args = extract_tool_call(call)
                text = (json.dumps(args, ensure_ascii=False, indent=2)
                        if isinstance(args, (dict, list)) else _args_to_str(args))
                section(f"tool:{call_index}", f"Tool call · {name}", text)
            section("observation", "Tool result", _obs_to_str(step.get("observation")))
        if api_events:
            for part in event["sections"]:
                raw_key = {"reasoning": "reasoning_content", "message": "message",
                           "observation": "observation"}.get(part["key"])
                raw = step.get(raw_key + "_raw", step.get(raw_key)) if raw_key else None
                # Part offsets refer to original joined API text, before JSON
                # normalization or whitespace stripping. Keep only exact maps.
                if not isinstance(raw, str) or raw.strip() != part["text"].strip():
                    continue
                base = raw.find(part["text"])
                if base < 0:
                    continue
                part["source_parts"] = [
                    {"part_idx": p["part_idx"], "start": p["start"] - base,
                     "end": p["end"] - base}
                    for p in step.get("source_parts", [])
                    if p.get("field") == raw_key
                ]
        events.append(event)
    # Pair tools only by explicit IDs (or source event IDs), never adjacency.
    counts = Counter(e["step_id"] for e in events if type(e["step_id"]) in {int, str})
    by_id = {e["step_id"]: e["index"] for e in events
             if type(e["step_id"]) in {int, str} and counts[e["step_id"]] == 1}
    def key(value: Any) -> int | str | None:
        return value if type(value) in {int, str} else None

    calls_by_id: dict[int | str, list[tuple[int, int]]] = {}
    for event in events:
        for i, call in enumerate(event["tool_calls"]):
            identity = key(call.get("call_id"))
            if identity is not None and identity != "":
                calls_by_id.setdefault(identity, []).append((event["index"], i))
    for event in events:
        parent = by_id.get(key(event.get("paired_event_idx")))
        if parent is not None:
            event["paired_step_index"] = parent
        elif event["source"] == "tool":
            matches = calls_by_id.get(key(event.get("call_id")), [])
            if len(matches) == 1:
                event["paired_step_index"], event["paired_call_idx"] = matches[0]
        event["result_step_indices"] = [
            by_id[key(c.get("result_event_idx"))] for c in event["tool_calls"]
            if key(c.get("result_event_idx")) in by_id
        ]
    for event in events:
        parent = event.get("paired_step_index")
        if parent and event["index"] not in events[parent - 1]["result_step_indices"]:
            events[parent - 1]["result_step_indices"].append(event["index"])
    return events


class TraceNavigation:
    """Map exact detector-field coordinates to the ordered event view once per item."""

    def __init__(self, item: dict[str, Any]):
        from autofa.data.trace_fields import derive_trace_fields

        self.fields = derive_trace_fields(item)
        self.events = trace_events(item)
        self.regions: dict[str, list[dict[str, Any]]] = {}
        steps = parse_trajectory(response_value(item))._steps

        def bind(field: str, chunks: list[tuple[str, list[dict]]]) -> None:
            text = "\n\n".join(chunk for chunk, _ in chunks)
            if not text or self.fields.get(field) != text:
                return
            regions, offset = [], 0
            for chunk, spans in chunks:
                regions.extend({**s, "start": s["start"] + offset,
                                "end": s["end"] + offset} for s in spans)
                offset += len(chunk) + 2
            self.regions[field] = regions

        reasoning, plain_reasoning, actions, observations = [], [], [], []
        final = []
        for event, step in zip(self.events, steps, strict=True):
            i = event["index"]
            for part in event["sections"]:
                key, text = part["key"], part["text"]
                span = {"step_index": i, "section": key, "start": 0,
                        "end": len(text), "text": text}
                if key == "reasoning":
                    prefix = f"[step {i}]\n"
                    reasoning.append((prefix + text, [{**span, "start": len(prefix),
                                                       "end": len(prefix) + len(text)}]))
                    plain_reasoning.append((text, [span]))
                if key == "observation" and text.strip():
                    prefix, content = f"[step {i} · observation]\n", text.strip()
                    observations.append((prefix + content, [{**span, "text": content,
                        "start": len(prefix), "end": len(prefix) + len(content)}]))
                if key == "message" and event["source"] in {"assistant", "agent"} and text.strip():
                    final = [(text.strip(), [{**span, "text": text.strip(),
                                             "end": len(text.strip())}])]
            chunk = format_actions([step], start_index=i)
            spans, cursor = [], 0
            for part in event["sections"]:
                key = part["key"]
                if key == "reasoning" or (key == "message" and
                                           event["source"] not in {"assistant", "agent"}):
                    continue
                text = part["text"].strip() if key in {"message", "observation"} else part["text"]
                if key == "observation" and isinstance(step.get("observation"), (dict, list)):
                    text = json.dumps(step["observation"], ensure_ascii=False, indent=2)
                start = chunk.find(text, cursor) if text else -1
                if start >= 0:
                    spans.append({"step_index": i, "section": key, "start": start,
                                  "end": start + len(text), "text": text})
                    cursor = start + len(text)
            if chunk:
                actions.append((chunk, spans))
        bind("reasoning_text", reasoning)
        if "reasoning_text" not in self.regions:
            bind("reasoning_text", plain_reasoning)
        bind("actions_text", actions)
        bind("observations_text", observations)
        bind("final_message", final)
        self._bind_raw(response_value(item))
        self._bind_input(response_value(item))

    def _bind_input(self, blob: Any) -> None:
        """Join input messages to unique complete event messages, never short quotes."""
        by_message, leading, in_prefix = {}, set(), True
        for event in self.events:
            in_prefix = in_prefix and event["source"] in {"system", "developer", "user"}
            if in_prefix:
                leading.add(event["index"])
            for section in event["sections"]:
                if section["key"] == "message":
                    by_message.setdefault((event["source"], section["text"]), []).append({
                        "step_index": event["index"], "section": "message",
                        "text": section["text"],
                    })

        # Input provenance need not name a lane. An identical real message in
        # another lane (or duplicate source identity) makes the join ambiguous.
        envelope = decode_container(blob)
        if isinstance(envelope, list) and len(envelope) == 1:
            envelope = envelope[0]
        execution = envelope.get("execution", {}) if isinstance(envelope, dict) else {}
        rows = execution.get("events", []) if isinstance(execution, dict) else []
        rows = rows if isinstance(rows, list) else []
        identities = Counter((row.get("lane"), row.get("event_idx")) for row in rows
                             if isinstance(row, dict) and isinstance(row.get("lane"), str)
                             and type(row.get("event_idx")) is int)
        for row in rows:
            if (not isinstance(row, dict) or row.get("is_copied_context")
                    or row.get("kind") not in {"system", "message"}
                    or not isinstance(row.get("lane"), str)
                    or type(row.get("event_idx")) is not int
                    or not isinstance(row.get("role", ""), str)):
                continue
            parts = row.get("parts", [])
            if not isinstance(parts, list) or not all(
                isinstance(p, dict) and type(p.get("part_idx")) is int for p in parts
            ):
                continue
            full = "\n\n".join(p["text"] for p in sorted(parts, key=lambda p: p["part_idx"])
                               if p.get("part_type") == "text" and isinstance(p.get("text"), str)
                               and p.get("channel") not in {"reasoning", "analysis"})
            key = (row.get("role") or "system", full)
            identity = {"lane": row.get("lane"), "event_idx": row.get("event_idx")}
            matches = by_message.get(key, [])
            if matches and (identities.get((row.get("lane"), row.get("event_idx")), 0) != 1
                            or any(self.events[m["step_index"] - 1].get("source_event")
                                   != identity for m in matches)):
                by_message[key] = []

        def target(role, full):
            matches = by_message.get((role, full), [])
            if len(matches) == 1 and matches[0]["step_index"] in leading:
                return matches[0]
            return None

        raw = self.fields.get("input_json")
        if isinstance(raw, str) and raw.strip():
            self.regions["input_json"] = []
            try:
                spans = []
                for role, full, chunks in _input_json_messages(raw):
                    match = target(role, full)
                    if match is None:
                        continue
                    base = 0
                    for value, start, end in chunks:
                        spans.append({**match, "text": value, "start": start, "end": end,
                                      "json_value": True, "target_base": base})
                        base += len(value) + 2
                self.regions["input_json"] = spans
            except (ValueError, IndexError, TypeError, RecursionError):
                pass  # Invalid/unknown containers retain report coordinates only.

        # Rebuild the readable projection in order, then require whole-field
        # equality. Headers and repeated content cannot shift message offsets.
        messages = self.fields.get("input_messages", [])
        def readable_message(message):
            content = message.get("content", message.get("text"))
            value, issues = text_content(content)
            blocks = [content] if isinstance(content, dict) else content
            plain = isinstance(content, str) or (isinstance(blocks, list) and all(
                isinstance(b, str) or (isinstance(b, dict)
                    and b.get("type") in {"text", "input_text", "output_text"}
                    and isinstance(b.get("text"), str)) for b in blocks
            ))
            match = target(message.get("role"), value) if plain and not issues else None
            return value, match

        self.regions["input_text"] = []
        for label in ("Initial model request", "Recorded agent request", "Exported input"):
            blocks, spans, offset = [], [], 0
            for message in messages:
                role = message.get("role")
                value, match = readable_message(message)
                if not value:
                    continue
                prefix = f"[{label}: {role}]\n"
                blocks.append(prefix + value)
                if match is not None:
                    spans.append({**match, "start": offset + len(prefix),
                                  "end": offset + len(prefix) + len(value)})
                offset += len(prefix) + len(value) + 2
            if "\n\n".join(blocks) == self.fields.get("input_text"):
                self.regions["input_text"] = spans
                break
        if len(messages) == 1:
            value, match = readable_message(messages[0])
            if match is not None and value == self.fields.get("input_text"):
                self.regions["input_text"] = [{**match, "start": 0, "end": len(value)}]

    def _bind_raw(self, blob: Any) -> None:  # noqa: C901
        """Bind known step content inside raw JSON; never match duplicated request pools."""
        if not isinstance(blob, str):
            return
        start = len(blob) - len(blob.lstrip())
        try:
            if blob[start] == "[":
                children = list(_json_children(blob, start))
                if len(children) != 1 or not isinstance(children[0][1], dict):
                    return
                start = children[0][2]
            root = {key: (value, lo) for key, value, lo, _ in _json_children(blob, start)}
            raw_steps, array_start = root.get("steps", (None, 0))
            if not isinstance(raw_steps, list) or len(raw_steps) != len(self.events):
                return
            spans = []

            def add(index, section, value, lo, hi, *, target_base=None, source_event=None):
                if not any(p["key"] == section for p in self.events[index]["sections"]):
                    return
                spans.append({"step_index": index + 1, "section": section,
                              "start": lo + (1 if isinstance(value, str) else 0),
                              "end": hi - (1 if isinstance(value, str) else 0),
                              "text": value, "json_value": True,
                              **({"target_base": target_base} if target_base is not None else {}),
                              **({"source_event": source_event} if source_event else {})})

            fields = {"reasoning_content": "reasoning", "thought": "reasoning",
                      "reasoning": "reasoning", "message": "message", "observation": "observation"}
            for index, _, step_start, _ in _json_children(blob, array_start):
                for key, value, lo, hi in _json_children(blob, step_start):
                    if key in fields:
                        add(index, fields[key], value, lo, hi)
                    elif key == "tool_calls" and isinstance(value, list):
                        for call_index, _, call_start, _ in _json_children(blob, lo):
                            for name, args, a, b in _json_children(blob, call_start):
                                if name in {"arguments", "args", "input", "args_full"}:
                                    add(index, f"tool:{call_index}", args, a, b)
                                elif name == "function" and isinstance(args, dict):
                                    for name2, args2, a2, b2 in _json_children(blob, a):
                                        if name2 in {"arguments", "input"}:
                                            add(index, f"tool:{call_index}", args2, a2, b2)
            def source_identity(row):
                event_idx, lane = row.get("event_idx"), row.get("lane")
                if (_verified_source_identity(root.get("evidence_scope", (None, 0))[0])
                        and type(event_idx) is int and event_idx >= 0 and isinstance(lane, str)):
                    return {"event_idx": event_idx, "lane": lane}
                return None

            def source_only(identity, section, value, lo, hi):
                if not identity or not isinstance(value, str):
                    return
                spans.append({"source_event": identity, "section": section,
                              "start": lo + 1, "end": hi - 1,
                              "text": value, "json_value": True})

            by_source = {}
            for index, event in enumerate(self.events):
                source = event.get("source_event")
                if source:
                    by_source.setdefault((source["lane"], source["event_idx"]), []).append(index)

            def bind_part(index, identity, part, lo):
                part_idx = part.get("part_idx")
                if type(part_idx) is not int or part_idx < 0:
                    return
                source = {**identity, "part_idx": part_idx} if identity else None
                for key, value, a, b in _json_children(blob, lo):
                    if key != "text" or not isinstance(value, str):
                        continue
                    matches = []
                    if index is not None and not part.get("metadata_filtered"):
                        for section in self.events[index]["sections"]:
                            for mapping in section.get("source_parts", []):
                                start, end = mapping["start"], mapping["end"]
                                if (mapping["part_idx"] == part_idx
                                        and section["text"][start:end] == value):
                                    matches.append((section["key"], start))
                    if len(matches) == 1:
                        section, base = matches[0]
                        add(index, section, value, a, b, target_base=base, source_event=source)
                    else:
                        source_only(source, "part", value, a, b)

            # Rich step parts duplicate rendered content. Match the explicit part
            # identity and complete part text, never the first occurrence of a quote.
            for index, _, step_start, _ in _json_children(blob, array_start):
                identity = self.events[index].get("source_event")
                for key, value, lo, _ in _json_children(blob, step_start):
                    if key == "parts" and isinstance(value, list):
                        for _, part, a, _ in _json_children(blob, lo):
                            if isinstance(part, dict):
                                bind_part(index, identity, part, a)

            execution, execution_start = root.get("execution", (None, 0))
            if isinstance(execution, dict):
                resources = {key: (value, lo) for key, value, lo, _ in
                             _json_children(blob, execution_start)}
                raw_events, event_start = resources.get("events", (None, 0))
                source_events = {}
                if isinstance(raw_events, list):
                    for event in raw_events:
                        if isinstance(event, dict) and (identity := source_identity(event)):
                            source_events.setdefault(
                                (identity["lane"], identity["event_idx"]), [],
                            ).append(event)
                    for _, event, lo, _ in _json_children(blob, event_start):
                        identity = source_identity(event) if isinstance(event, dict) else None
                        if not identity:
                            continue
                        key = (identity["lane"], identity["event_idx"])
                        if len(source_events[key]) != 1:
                            continue
                        local = by_source.get(key, [])
                        index = local[0] if len(local) == 1 else None
                        if event.get("is_copied_context"):
                            continue
                        for field, parts, a, _ in _json_children(blob, lo):
                            if field == "parts" and isinstance(parts, list):
                                for _, part, b, _ in _json_children(blob, a):
                                    if isinstance(part, dict):
                                        capture = (decode_container(part.get("text"))
                                                   if event.get("kind") == "other"
                                                   and event.get("role") == "harness" else None)
                                        if (isinstance(capture, dict)
                                                and {"message_ids", "messages_first_seen_here"}
                                                <= capture.keys()):
                                            continue
                                        bind_part(index, identity, part, b)
                            # event_text contains model request snapshots, not executed content.
                raw_calls, calls_start = resources.get("tool_calls", (None, 0))
                if isinstance(raw_calls, list):
                    for _, call, lo, _ in _json_children(blob, calls_start):
                        identity = source_identity(call) if isinstance(call, dict) else None
                        if not identity:
                            continue
                        source_key = (identity["lane"], identity["event_idx"])
                        if len(source_events.get(source_key, [])) != 1:
                            continue
                        local = by_source.get(source_key, [])
                        index = local[0] if len(local) == 1 else None
                        matches = [] if index is None else [
                            i for i, candidate in enumerate(self.events[index]["tool_calls"])
                            if type(call.get("call_idx")) is int
                            and candidate.get("source_call_index") == call["call_idx"]
                        ]
                        for field, value, a, b in _json_children(blob, lo):
                            if field == "args_full" and isinstance(value, str):
                                if len(matches) == 1:
                                    add(index, f"tool:{matches[0]}", value, a, b,
                                        source_event=identity)
                                else:
                                    source_only(identity, "tool", value, a, b)
                artifacts, artifact_start = resources.get("artifacts", (None, 0))
                if isinstance(artifacts, list):
                    for _, artifact, lo, _ in _json_children(blob, artifact_start):
                        if not isinstance(artifact, dict):
                            continue
                        # Artifact rows often omit lane. Join only an unambiguous
                        # explicit event ID or a part's exact artifact ID reference.
                        identities = []
                        for key, candidates in source_events.items():
                            if len(candidates) != 1:
                                continue
                            if artifact.get("lane") is not None and artifact["lane"] != key[0]:
                                continue
                            event = candidates[0]
                            event_idx = artifact.get("event_idx")
                            part_idx = artifact.get("part_idx")
                            if type(event_idx) is int and event_idx != key[1]:
                                continue
                            parts = event.get("parts", [])
                            links = [part for part in parts if type(part.get("part_idx")) is int
                                     and part["part_idx"] >= 0 and (
                                         (type(event_idx) is int and type(part_idx) is int
                                          and part["part_idx"] == part_idx)
                                         or (part_idx is None
                                             and type(artifact.get("artifact_idx")) is int
                                             and part.get("artifact_idx")
                                             == artifact["artifact_idx"])
                                     )]
                            if links:
                                identities.extend({"lane": key[0], "event_idx": key[1],
                                                   "part_idx": part["part_idx"]} for part in links)
                            elif type(event_idx) is int:
                                identities.append({"lane": key[0], "event_idx": key[1]})
                        if len(identities) == 1:
                            for field, value, a, b in _json_children(blob, lo):
                                if field == "text":
                                    source_only(identities[0], "artifact", value, a, b)
            for field in RESPONSE_FIELDS:
                if self.fields.get(field) == blob:
                    self.regions[field] = spans
        except (ValueError, IndexError, TypeError):
            # Other supported dialects retain their structured/combined maps.
            # Unknown raw serialization does not justify an invented location.
            return

    def locate(self, passage: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: C901
        """Only verified offsets produce navigation; stale/ambiguous quotes do not."""
        field = passage.get("field")
        text = self.fields.get(field)
        start, end, quote = (passage.get(k) for k in ("start", "end", "quote"))
        if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text) or text[start:end] != quote
                or passage.get("field_sha256") != _hash(text)):
            return []
        candidates = []
        for span in self.regions.get(field, []):
            if span["start"] <= start and end <= span["end"]:
                candidates.append((span, start - span["start"], end - span["start"]))
        # Native single-section fields can be mapped by whole-field equality.
        # Never guess from a short quote repeated in requests or JSON envelopes.
        if field not in self.regions:
            for event in self.events:
                for part in event["sections"]:
                    if part["text"] == text:
                        candidates.append(({"step_index": event["index"],
                                            "section": part["key"], "text": text}, start, end))
            if len(candidates) != 1:
                return []
        locations = []
        for span, lo, hi in candidates:
            if "step_index" not in span:
                # Source-only execution lanes and artifacts have no local conversation
                # step. Their verified IDs still support a precise external event link.
                location = {"section": span["section"],
                            "source_event": dict(span["source_event"])}
                if span.get("json_value"):
                    try:
                        prefix = json.loads('"' + text[span["start"]:start] + '"')
                        through = json.loads('"' + text[span["start"]:end] + '"')
                        prefix.encode("utf-8")
                        through.encode("utf-8")
                        decoded = through[len(prefix):]
                        context = _source_highlight_context(span["text"], len(prefix), len(through))
                        if context is not None:
                            location["source_context"] = context
                    except ValueError:
                        decoded = ""  # A partial escape identifies an event, not source text.
                    if decoded != quote:
                        location["quote"] = decoded
                locations.append(location)
                continue
            event = self.events[span["step_index"] - 1]
            part = next(p for p in event["sections"] if p["key"] == span["section"])
            location = {"step_index": event["index"], "section": part["key"]}
            target_quote = quote
            if span.get("json_value"):
                if isinstance(span["text"], str):
                    try:
                        prefix = json.loads('"' + text[span["start"]:start] + '"')
                        through = json.loads('"' + text[span["start"]:end] + '"')
                        prefix.encode("utf-8")  # Refuse half of a surrogate-pair escape.
                        through.encode("utf-8")
                        target_quote = through[len(prefix):]
                        base = span.get("target_base", part["text"].find(span["text"]))
                        if base < 0 and span["text"].strip() == part["text"]:
                            base = -(len(span["text"]) - len(span["text"].lstrip()))
                            if base + len(prefix) < 0:
                                raise ValueError("match includes stripped whitespace")
                        elif base < 0:
                            raise ValueError("different rendered content")
                        if target_quote:
                            location.update(start=base + len(prefix), end=base + len(through))
                    except ValueError:
                        target_quote = ""  # Match crosses a JSON escape: event link only.
                if "start" not in location:
                    pos = part["text"].find(target_quote)
                    if pos >= 0 and part["text"].find(target_quote, pos + 1) < 0:
                        location.update(start=pos, end=pos + len(target_quote))
                if target_quote != quote:
                    location["quote"] = target_quote
            elif span["text"] == part["text"]:
                location.update(start=lo, end=hi)
            else:
                # A pretty-printed JSON field may differ from the detector's
                # compact field. Open the right event; highlight only exact text.
                pos = part["text"].find(quote)
                if pos >= 0 and part["text"].find(quote, pos + 1) < 0:
                    location.update(start=pos, end=pos + len(quote))
            source = span.get("source_event", event.get("source_event"))
            if source:
                location["source_event"] = dict(source)
                source_text, source_start, source_end = part["text"], None, None
                if "start" in location:
                    source_start, source_end = location["start"], location["end"]
                for source_part in part.get("source_parts", []):
                    if ("start" in location and source_part["start"] <= location["start"]
                            and location["end"] <= source_part["end"]):
                        location["source_event"]["part_idx"] = source_part["part_idx"]
                        source_text = part["text"][source_part["start"]:source_part["end"]]
                        source_start -= source_part["start"]
                        source_end -= source_part["start"]
                        break
                if source_start is not None:
                    others = [] if "part_idx" in location["source_event"] else [
                        section["text"] for section in event["sections"] if section is not part
                    ]
                    context = _source_highlight_context(
                        source_text, source_start, source_end, others,
                    )
                    if context is not None:
                        location["source_context"] = context
                else:
                    location["quote"] = ""  # The event is known, but the source span is not.
            locations.append(location)
        return locations

    def attach(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return a report/capture copy, replacing any stale saved navigation."""
        return {**record, "navigation_version": NAVIGATION_VERSION, "passages": [
            {**p, "trace_locations": self.locate(p)} for p in record.get("passages", [])
        ]}
