# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary-aware review text, with exact links back to retained source characters.

Only supported image payload slots at captured request/message boundaries are
projected. Prose, tool arguments, metadata and unavailable image data stay intact.
Original fields remain the source for artifact inspection; references are never
evidence that the image has been viewed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from autofa.data import inline_images
from autofa.data.execution_policy import is_evaluation_record
from autofa.data.structured import decode_container
from autofa.data.trace_navigation import _json_children

PROJECTION_VERSION = "binary-review-text-v1"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _string_offsets(token: str, positions: list[int]) -> dict[int, int]:
    """Map selected decoded boundaries without allocating an array for image bytes."""
    offsets, pos, decoded = {}, 1, 0
    for target in sorted(set(positions)):
        while decoded < target:
            if token[pos] != "\\":
                stop = token.find("\\", pos, len(token) - 1)
                if stop == -1:
                    stop = len(token) - 1
                count = min(target - decoded, stop - pos)
                pos, decoded = pos + count, decoded + count
                continue
            if token[pos + 1] != "u":
                pos += 2
            else:
                code = int(token[pos + 2:pos + 6], 16)
                pos += 6
                if (0xD800 <= code <= 0xDBFF and token[pos:pos + 2] == "\\u"
                        and 0xDC00 <= int(token[pos + 2:pos + 6], 16) <= 0xDFFF):
                    pos += 6
            decoded += 1
        offsets[target] = pos
    return offsets


def _payload_path(block: dict) -> tuple[str, ...]:
    if block.get("type") in {"image_url", "input_image"}:
        return (("image_url", "url") if isinstance(block.get("image_url"), dict)
                else ("image_url",))
    return ("source", "data")


def _payload_span(text: str, start: int, path: tuple[str, ...]) -> tuple[int, int]:
    for token in path:
        _, _, start, end = next(child for child in _json_children(text, start)
                                if child[0] == token)
    return start, end


def _replacements(text: str, catalog: dict[str, dict]) -> list[dict]:
    replacements, nodes = [], 0

    def visit(raw: str, start: int, value: Any, depth: int, path: list) -> list[dict]:
        nonlocal nodes
        nodes += 1
        if nodes > inline_images.MAX_NODES or depth > inline_images.MAX_DEPTH:
            return []
        if isinstance(value, str):
            decoded = decode_container(value)
            if not isinstance(decoded, (dict, list)):
                return []
            # Nested JSON is decoded only where the inventory also permits a
            # structural string (e.g. agent_request.messages), never prose.
            inner_start = len(value) - len(value.lstrip())
            found = visit(value, inner_start, decoded, depth + 1, path)
            if not found:
                return []
            _, end = json.JSONDecoder().raw_decode(raw, start)
            offsets = _string_offsets(raw[start:end], [offset for row in found
                                                       for offset in (row["source_start"],
                                                                      row["source_end"])])
            return [{**row, "source_start": start + offsets[row["source_start"]],
                     "source_end": start + offsets[row["source_end"]],
                     "text": json.dumps(row["text"], ensure_ascii=False)[1:-1]}
                    for row in found]
        if not isinstance(value, (list, dict)):
            return []
        if isinstance(value, dict):
            if (is_evaluation_record(value)
                    or value.get("role") in {"assistant", "agent", "tool", "function"}):
                return []
            if value.get("type") in {"input_image", "image_url", "image"}:
                try:
                    data, _ = inline_images._payload(value)
                except ValueError:
                    return []
                digest = hashlib.sha256(data).hexdigest()
                artifact = catalog.get(digest)
                if artifact is None:
                    return []
                lo, hi = _payload_span(raw, start, _payload_path(value))
                reference = f"artifact://{artifact['artifact_idx']}/sha256/{digest}"
                return [{"source_start": lo, "source_end": hi,
                         "artifact_idx": artifact["artifact_idx"], "sha256": digest,
                         "source_path": path + list(_payload_path(value)),
                         "text": json.dumps(reference)}]
        found = []
        for key, child, lo, _ in _json_children(raw, start):
            if isinstance(value, dict) and key not in inline_images._CHILDREN:
                continue
            if isinstance(child, str) and (key == "content" or path[-1:] == ["content"]):
                continue
            found.extend(visit(raw, lo, child, depth + 1, [*path, key]))
        return found

    try:
        start = len(text) - len(text.lstrip())
        value, end = json.JSONDecoder().raw_decode(text, start)
        if text[end:].strip():
            return []
        replacements = visit(text, start, value, 0, [])
    except (ValueError, TypeError, IndexError, StopIteration):
        # An unrecognized source is retained in full, never a partial projection.
        return []
    return replacements


def project_fields(fields: dict[str, str], catalog: list[dict]) -> tuple[dict, dict]:
    """Return projected text and replacement maps; never mutate retained fields."""
    accepted = {row["sha256"]: row for row in catalog
                if row.get("source_kind") == "retained_input_image"
                and row.get("status") == "available" and row.get("sha256")}
    if not accepted:
        return fields, {}
    from autofa.data.input_view import COMPACT_INPUT_ORIGIN, RESPONSE_FIELDS

    literal = False
    for field, origin in (("input_json", "autofa_input_origin"),
                          ("input_parse", "response_origin")):
        raw = fields.get(field, "")
        if "response_is_literal" not in raw:
            continue
        if len(raw) > inline_images.MAX_SOURCE_CHARS:
            literal = True
            continue
        value = decode_container(raw)
        literal |= (isinstance(value, dict) and value.get(origin) == COMPACT_INPUT_ORIGIN
                    and value.get("response_is_literal") is True)
    projected, maps = dict(fields), {}
    for field, text in fields.items():
        if (field not in inline_images._FIELDS
                or field in {"input", "prompt", "question", "problem", "user_message"}
                or literal and field in RESPONSE_FIELDS
                or len(text) > inline_images.MAX_SOURCE_CHARS):
            continue
        rows = _replacements(text, accepted)
        if not rows:
            continue
        chunks, previous, length, mappings = [], 0, 0, []
        for row in sorted(rows, key=lambda row: row["source_start"]):
            prefix = text[previous:row["source_start"]]
            chunks.extend((prefix, row["text"]))
            length += len(prefix)
            mappings.append({key: value for key, value in row.items() if key != "text"} | {
                "start": length, "end": length + len(row["text"]),
            })
            length += len(row["text"])
            previous = row["source_end"]
        chunks.append(text[previous:])
        projected[field] = "".join(chunks)
        maps[field] = {"source_sha256": _hash(text), "source_chars": len(text),
                       "replacements": mappings}
    return projected, maps


def source_span(mapping: dict, start: int, end: int) -> tuple[int, int] | None:
    """Map unchanged review text to its retained source; references are not quotes."""
    delta = 0
    for row in mapping.get("replacements", []):
        if row["start"] < end and start < row["end"]:
            return None
        if row["end"] <= start:
            delta += row["source_end"] - row["source_start"] - (row["end"] - row["start"])
    return start + delta, end + delta


def projected_span(mapping: dict, start: int, end: int) -> tuple[int, int] | None:
    """Map an exact original text hint without selecting a different occurrence."""
    delta = 0
    for row in mapping.get("replacements", []):
        if row["source_start"] < end and start < row["source_end"]:
            return None
        if row["source_end"] <= start:
            delta += row["end"] - row["start"] - (row["source_end"] - row["source_start"])
    return start + delta, end + delta
