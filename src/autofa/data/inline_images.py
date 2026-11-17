# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Index retained input-image bytes without fetching URLs or reading supplied paths.

Descriptors join the ordinary artifact catalog. Their original structural locations
and hashes bind the shared cache, detector dependencies and judge image receipts.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from autofa.data.execution_policy import is_evaluation_record
from autofa.data.structured import decode_container

INLINE_IMAGE_VERSION = "retained-input-images-v1"
MAX_SOURCE_CHARS = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGES = 64
MAX_NODES = 100_000
MAX_DEPTH = 32
_FIELDS = (
    "input_messages", "messages", "input_json", "input", "prompt", "question", "problem",
    "user_message", "conversation", "dialog", "response_text", "generation", "trace",
)
# Decode only structural request/message boundaries, never arbitrary model text,
# source code, tool arguments or evaluator metadata.
_CHILDREN = frozenset({
    "agent_request", "initial_request", "model_call_evidence", "calls", "request",
    "request_json", "messages", "input_messages", "input", "content", "steps",
})


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str,
    )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _payload(block: dict) -> tuple[bytes, str]:
    kind = block.get("type")
    if kind in {"image_url", "input_image"}:
        url = block.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if not isinstance(url, str) or not url.startswith("data:image/"):
            raise ValueError(
                "Image bytes are not retained inline; external URLs/paths are not fetched.",
            )
        header, sep, encoded = url.partition(";base64,")
        if not sep:
            raise ValueError("Retained image is not a base64 data URL.")
        mime = header.removeprefix("data:")
    else:
        source = block.get("source")
        if not isinstance(source, dict) or source.get("type") != "base64":
            raise ValueError("Image bytes are not retained as a supported base64 source.")
        mime, encoded = source.get("media_type"), source.get("data")
    if not isinstance(mime, str) or not mime.startswith("image/"):
        raise ValueError("Retained image has no image media type.")
    if not isinstance(encoded, str):
        raise ValueError("Retained image has no base64 data.")
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError(f"Retained image exceeds the {MAX_IMAGE_BYTES}-byte inspection limit.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Retained image has invalid base64 data.") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Retained image bytes are empty or exceed the inspection limit.")
    return data, mime


def image_inventory(item: dict, *, first_index: int = 0) -> tuple[list[dict], list[str]]:  # noqa: C901
    """Return bounded, deduplicated descriptors plus explicit inventory limitations."""
    from autofa.data.input_view import COMPACT_INPUT_ORIGIN, RESPONSE_FIELDS

    candidates: list[tuple[str, dict]] = []
    limitations: list[str] = []
    seen: set[str] = set()
    nodes = 0
    provenance_limited = False

    def provenance(field: str) -> Any:
        nonlocal provenance_limited
        value = item.get(field)
        if isinstance(value, str):
            # Normal input JSON often contains a large request; do not parse it
            # merely to check a provenance marker it does not contain.
            if "response_is_literal" not in value:
                return None
            if len(value) > MAX_SOURCE_CHARS:
                limitations.append(f"Retained-image provenance {field} exceeds decoding limit.")
                provenance_limited = True
                return None
        return decode_container(value)

    stored_input = provenance("input_json")
    parse = provenance("input_parse")
    literal_response = provenance_limited or (
        isinstance(stored_input, dict)
        and stored_input.get("autofa_input_origin") == COMPACT_INPUT_ORIGIN
        and stored_input.get("response_is_literal") is True
    ) or (
        isinstance(parse, dict)
        and parse.get("response_origin") == COMPACT_INPUT_ORIGIN
        and parse.get("response_is_literal") is True
    )

    def visit(value: Any, field: str, path: list, source_hash: str, depth: int = 0):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            reason = "Retained-image inventory exceeded its structural inspection limit."
            if reason not in limitations:
                limitations.append(reason)
            return
        value = decode_container(value)
        if isinstance(value, list):
            for index, child in enumerate(value):
                if path and path[-1] == "content" and isinstance(child, str):
                    continue  # Plain message text is not a captured image block.
                visit(child, field, [*path, index], source_hash, depth + 1)
                if nodes > MAX_NODES:
                    break
        elif isinstance(value, dict):
            if (is_evaluation_record(value)
                    or value.get("role") in {"assistant", "agent", "tool", "function"}):
                return
            if value.get("type") in {"input_image", "image_url", "image"}:
                descriptor: dict = {"source_kind": "retained_input_image", "source_field": field,
                                    "source_path": path, "source_field_sha256": source_hash,
                                    "requires_visual_review": True}
                try:
                    data, mime = _payload(value)
                    digest = hashlib.sha256(data).hexdigest()
                    descriptor.update(sha256=digest, size_bytes=len(data), mime_type=mime,
                                      status="available")
                    key = digest
                except ValueError as exc:
                    key = _hash(_text(value))
                    descriptor.update(status="unavailable", limitations=[str(exc)])
                if key in seen:
                    return
                seen.add(key)
                if len(candidates) >= MAX_IMAGES:
                    reason = "Retained-image inventory exceeds the image count limit."
                    if reason not in limitations:
                        limitations.append(reason)
                    # Choose the same bounded set even if derived input messages
                    # change encounter order in the judge's frozen representation.
                    largest = max(candidates, key=lambda pair: pair[0]) if candidates else None
                    if largest is None or key >= largest[0]:
                        return
                    candidates.remove(largest)
                candidates.append((key, descriptor))
                return
            for key, child in value.items():
                if key in _CHILDREN:
                    if key == "content" and isinstance(child, str):
                        continue
                    visit(child, field, [*path, key], source_hash, depth + 1)

    for field in _FIELDS:
        if field not in item:
            continue
        if literal_response and field in RESPONSE_FIELDS:
            continue
        if field in {"input", "prompt", "question", "problem", "user_message"} and isinstance(
            item[field], str,
        ):
            continue  # Literal task text can quote image-shaped JSON.
        source = _text(item[field])
        # Ordinary strings cannot contain structural image blocks. Avoid parsing
        # large traces unnecessarily when they contain no supported image marker.
        markers = ("input_image", "image_url", '"image"', '\\"image\\"')
        if not any(marker in source for marker in markers):
            continue
        if len(source) > MAX_SOURCE_CHARS:
            limitations.append(
                f"Retained-image source {field} exceeds the structural decoding limit.",
            )
            continue
        visit(item[field], field, [], _hash(source))
        if nodes > MAX_NODES:
            break
    catalog = []
    ordered = sorted(candidates, key=lambda pair: pair[0])
    for index, (_, descriptor) in enumerate(ordered, first_index):
        descriptor.update(artifact_idx=index, name=f"retained-input-image-{index}")
        catalog.append(descriptor)
    return catalog, limitations


def retained_image_bytes(item: dict, descriptor: dict) -> tuple[bytes, dict]:
    """Resolve only a verified structural pointer into the retained source value."""
    field = descriptor.get("source_field")
    path = descriptor.get("source_path")
    if field not in _FIELDS or not isinstance(path, list) or len(path) > MAX_DEPTH:
        raise ValueError("Invalid retained-image source locator.")
    source = item.get(field)
    if _hash(_text(source)) != descriptor.get("source_field_sha256"):
        raise ValueError("Retained-image source changed since its catalog was created.")
    value = source
    for token in path:
        value = decode_container(value)
        if isinstance(value, list) and type(token) is int and token >= 0:
            value = value[token]
        elif isinstance(value, dict) and token in _CHILDREN:
            value = value[token]
        else:
            raise ValueError("Invalid retained-image structural path.")
    value = decode_container(value)
    if not isinstance(value, dict):
        raise ValueError("Retained image is no longer a structured content block.")
    data, _ = _payload(value)
    digest = hashlib.sha256(data).hexdigest()
    if digest != descriptor.get("sha256") or len(data) != descriptor.get("size_bytes"):
        raise ValueError("Retained-image bytes differ from their catalog identity.")
    identity = {key: descriptor[key] for key in
                ("artifact_idx", "source_field", "source_path", "source_field_sha256", "sha256")}
    return data, {"kind": "retained_input_image", "version": INLINE_IMAGE_VERSION, **identity}
