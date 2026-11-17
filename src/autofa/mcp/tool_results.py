# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless, paged access to tool responses too large for discovery context."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from autofa.mcp.runtime import err, get_runtime, ok
from autofa.util.atomic import atomic_write_json

PAGE_CHARS = 12_000
DESCRIPTION_CHILDREN = 50


def save_tool_result(result: Any, *, run_dir: Path | None = None) -> dict:
    directory = run_dir if run_dir is not None else get_runtime().run.run_dir()
    payload = json.dumps(result, ensure_ascii=True, default=str)
    result_id = hashlib.sha256(payload.encode()).hexdigest()
    path = directory / "tool_results" / f"{result_id}.json"
    if not path.exists():
        atomic_write_json(path, {"text": payload})
    return {"result_id": result_id, "total_chars": len(payload)}


def _select_json_pointer(value: Any, pointer: str) -> Any:
    for token in pointer.split("/")[1:]:
        if re.search(r"~(?![01])", token):
            raise ValueError("JSON pointer escapes must be ~0 or ~1")
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if key not in value:
                raise ValueError(f"JSON pointer key not found: {key!r}")
            value = value[key]
        elif isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", key):
                raise ValueError("JSON pointer array index must be a nonnegative integer")
            if len(key) > len(str(len(value))) or int(key) >= len(value):
                raise ValueError("JSON pointer array index is out of range")
            value = value[int(key)]
        else:
            raise ValueError("JSON pointer cannot traverse a scalar value")
    return value


def _value_metadata(value: Any) -> dict:
    value_type = (
        "null" if value is None else "boolean" if isinstance(value, bool)
        else "object" if isinstance(value, dict) else "array" if isinstance(value, (list, tuple))
        else "number" if isinstance(value, (int, float)) else "string"
    )
    return {
        "value_type": value_type,
        "total_chars": len(json.dumps(value, ensure_ascii=True, default=str)),
        "total_children": len(value) if isinstance(value, (dict, list, tuple)) else 0,
    }


def describe_result_value(
    value: Any, *, pointer: str = "", offset: int = 0, limit: int = DESCRIPTION_CHILDREN,
    max_chars: int = PAGE_CHARS,
) -> dict:
    """Describe exact child paths without returning their evidence values."""
    metadata = _value_metadata(value)
    total = metadata["total_children"]
    if offset > total:
        raise ValueError("offset exceeds child count")
    keys = list(value) if isinstance(value, dict) else range(total)
    result = {
        "view": "structure", "scope": "selection" if pointer else "whole_result",
        "pointer": pointer, **metadata, "children": [], "offset": offset,
        "offset_unit": "children", "next_offset": None, "complete": False,
        "full_result_complete": False,
    }
    end = offset
    for key in keys[offset:offset + min(limit, DESCRIPTION_CHILDREN)]:
        # Match JSON's object-key conversion for producer dictionaries too; the
        # overflow manifest is generated before the already-saved JSON is parsed.
        token = (next(iter(json.loads(json.dumps({key: None}))))
                 if isinstance(value, dict) and not isinstance(key, str) else str(key))
        child_pointer = pointer + "/" + token.replace("~", "~0").replace("/", "~1")
        child = {"pointer": child_pointer, **_value_metadata(value[key])}
        result["children"].append(child)
        if len(json.dumps(result, ensure_ascii=True)) > max_chars:
            result["children"].pop()
            break
        end += 1
    if (end == offset and offset < total) or len(json.dumps(result)) > max_chars:
        raise ValueError("Structural metadata exceeds the page budget; use raw JSON paging")
    result["next_offset"] = end if end < total else None
    result["complete"] = offset == 0 and end == total
    return result


def read_tool_result(
    result_id: str, offset: int = 0, limit: int = PAGE_CHARS,
    pointer: str = "", describe: bool = False,
) -> dict:
    """Read saved results without repeating the original operation.

    Empty pointer pages the original JSON text. An RFC6901 pointer (e.g.
    /feedback/findings/0) pages only that exact value as JSON text; follow
    next_offset with the same pointer and join pages before JSON decoding.
    Stop when next_offset is null, NOT when complete is true: the final page of
    a multi-page result has complete=false and next_offset=null. Never pass null
    as offset or restart at zero after an error. Check ok before using text or
    next_offset; require each non-null cursor to be an integer greater than offset.
    Bound automated reads and report cap exhaustion as incomplete, not success.
    describe=True lists child pointers, types and sizes without evidence values;
    offset/limit then count children (at most 50), not text characters.
    complete is true only when ONE response contains the entire selected value
    or child listing; it is an evidence-scope flag, not an end-of-pages flag.
    full_result_complete is true only for an entire original JSON result.
    Any #/... references remain relative to the original saved result's root;
    selecting a reference returns it unchanged, without automatic dereferencing.
    """
    if not isinstance(result_id, str) or not re.fullmatch(r"[0-9a-f]{64}", result_id):
        return err("Invalid result_id")
    if type(offset) is not int or offset < 0 or type(limit) is not int or limit <= 0:
        return err("offset must be nonnegative and limit must be positive integers")
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        return err("pointer must be an RFC6901 string: empty or starting with /")
    if type(describe) is not bool:
        return err("describe must be a boolean")
    path = get_runtime().run.run_dir() / "tool_results" / f"{result_id}.json"
    try:
        payload = json.loads(path.read_text())["text"]
        if not isinstance(payload, str):
            raise ValueError("saved text must be a string")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return err(f"Cannot read saved tool result: {exc}")
    whole_result_chars = len(payload)
    if pointer or describe:
        try:
            value = _select_json_pointer(json.loads(payload), pointer)
            if describe:
                description = describe_result_value(
                    value, pointer=pointer, offset=offset, limit=limit,
                )
                return ok(result_id=result_id, whole_result_chars=whole_result_chars,
                          **description)
            payload = json.dumps(value, ensure_ascii=True, default=str)
        except ValueError as exc:
            return err(f"Cannot select saved tool result: {exc}", pointer=pointer)
    if offset > len(payload):
        return err("offset exceeds result length", total_chars=len(payload))
    end = min(len(payload), offset + min(limit, PAGE_CHARS))
    complete = offset == 0 and end == len(payload)
    return ok(result_id=result_id, text=payload[offset:end], offset=offset,
              next_offset=end if end < len(payload) else None,
              total_chars=len(payload), complete=complete, view="json",
              scope="selection" if pointer else "whole_result", pointer=pointer,
              offset_unit="characters", whole_result_chars=whole_result_chars,
              full_result_complete=not pointer and complete)
