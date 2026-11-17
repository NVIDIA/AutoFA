# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, lossless packing of memberships and review evidence in v2 libraries."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import zlib
from typing import Any

PACKED_KEY = "packed_bucket_examples"
MAX_UNPACKED_BYTES = 128 * 1024 * 1024
EXAMPLE_KEYS = {"example_fail_ids", "example_pass_ids"}
REVIEW_PACKED_KEY = "packed_review_evidence"
REVIEW_PACK_THRESHOLD = 32 * 1024 * 1024
REVIEW_SCORE_KEYS = frozenset({
    "behavior_review_assessments", "diagnostic_review", "repair_validation",
    "behavior_verdict_history", "behavior_review_localisations",
})
_HISTORY_PLACEHOLDER = {
    "schema_version": 1, "history_complete": False, "detectors": {},
    "discovery_item_ids": [], "discovery_task_groups": {},
}


def _compact_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _encode_records(records: list, *, other_bytes: int = 0,
                    schema_version: int | None = None) -> dict[str, Any]:
    raw = _compact_bytes(records)
    if len(raw) + other_bytes > MAX_UNPACKED_BYTES:
        raise ValueError("Evidence exceeds the portable decompression limit")
    result = {
        "encoding": "gzip+base64", "uncompressed_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii"),
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    return result


def _decode_records(packed: Any, *, schema_version: int | None = None) -> list:
    if not isinstance(packed, dict):
        raise ValueError("Packed evidence must be an object")
    size = packed.get("uncompressed_bytes")
    if (packed.get("encoding") != "gzip+base64" or type(size) is not int
            or not 0 <= size <= MAX_UNPACKED_BYTES
            or (schema_version is not None and
                (type(packed.get("schema_version")) is not int
                 or packed["schema_version"] != schema_version))):
        raise ValueError("Invalid evidence encoding or size")
    compressed = base64.b64decode(packed["data"], validate=True)
    decoder = zlib.decompressobj(wbits=31)
    raw = decoder.decompress(compressed, size + 1)
    if (len(raw) != size or not decoder.eof or decoder.unused_data
            or hashlib.sha256(raw).hexdigest() != packed["sha256"]):
        raise ValueError("Evidence size or integrity mismatch")
    records = json.loads(raw)
    if not isinstance(records, list):
        raise ValueError("Packed evidence must be an array")
    return records


def _pack_review_evidence(
    data: dict[str, Any], *, max_bytes: int = REVIEW_PACK_THRESHOLD, owned: bool = False,
) -> dict[str, Any]:
    """Compress verbose evidence once when ordinary packing exceeds the cap.

    Detector code, active verdicts, aggregate scores and pinned dependencies stay
    directly readable. Older readers see explicitly incomplete exposure history;
    this reader restores the exact history before importing or evaluating it.
    """
    if REVIEW_PACKED_KEY in data or len(_compact_bytes(data)) <= max_bytes:
        return data
    packed = data if owned else copy.deepcopy(data)
    records = []
    if "review_history" in packed:
        records.append(["root", None, "review_history", packed["review_history"]])
        packed["review_history"] = copy.deepcopy(_HISTORY_PLACEHOLDER)
    for fn in packed.get("functions", []):
        if "failure_localisations" in fn:
            records.append(["function", fn["name"], "failure_localisations",
                            fn.pop("failure_localisations")])
        scores = fn.get("scores") or {}
        for key in sorted(REVIEW_SCORE_KEYS):
            if key in scores:
                records.append(["scores", fn["name"], key, scores.pop(key)])
    if not records:
        return data
    packed[REVIEW_PACKED_KEY] = _encode_records(
        records, other_bytes=(packed.get(PACKED_KEY) or {}).get("uncompressed_bytes", 0),
        schema_version=1,
    )
    packed["version"] = 2
    return packed


def _unpack_review_evidence(data: dict[str, Any]) -> dict[str, Any]:
    if REVIEW_PACKED_KEY not in data:
        return data
    packed = data[REVIEW_PACKED_KEY]
    try:
        records = _decode_records(packed, schema_version=1)
        restored = copy.deepcopy(data)
        functions = {fn["name"]: fn for fn in restored.get("functions", [])}
        if len(functions) != len(restored.get("functions", [])):
            raise ValueError("Duplicate function names in packed review targets")
        seen = set()
        for record in records:
            if not isinstance(record, list) or len(record) != 4:
                raise ValueError("Invalid packed review record")
            scope, name, key, value = record
            if not isinstance(scope, str) or not isinstance(key, str):
                raise ValueError("Invalid packed review target")
            if scope == "root" and name is None and key == "review_history":
                if restored.get(key) != _HISTORY_PLACEHOLDER or not isinstance(value, dict):
                    raise ValueError("Invalid review history placeholder or value")
                target = restored
            elif isinstance(name, str) and name in functions:
                fn = functions[name]
                if scope == "scores" and key in REVIEW_SCORE_KEYS:
                    target = fn.setdefault("scores", {})
                elif scope == "function" and key == "failure_localisations":
                    target = fn
                else:
                    raise ValueError("Unsupported packed review target")
                if key in target:
                    raise ValueError("Packed review target already exists")
            else:
                raise ValueError("Unknown packed review function or scope")
            identity = (scope, name, key)
            if identity in seen:
                raise ValueError("Duplicate packed review target")
            seen.add(identity)
            target[key] = value
        del restored[REVIEW_PACKED_KEY]
        return restored
    except (KeyError, TypeError, ValueError, zlib.error) as exc:
        raise ValueError(f"Invalid packed review evidence: {exc}") from exc


def pack_bucket_examples(
    data: dict[str, Any], *, max_bytes: int = REVIEW_PACK_THRESHOLD,
) -> dict[str, Any]:
    """Pack memberships, then large review evidence; keep code and metrics readable."""
    if PACKED_KEY in data:
        return _pack_review_evidence(data, max_bytes=max_bytes)
    if not any(EXAMPLE_KEYS.intersection(bucket)
               for fn in data.get("functions", [])
               for bucket in (fn.get("scores") or {}).get("buckets", [])):
        return _pack_review_evidence(data, max_bytes=max_bytes)
    packed = copy.deepcopy(data)
    records = []
    for fi, fn in enumerate(packed.get("functions", [])):
        for bi, bucket in enumerate((fn.get("scores") or {}).get("buckets", [])):
            for key in sorted(EXAMPLE_KEYS):
                if key in bucket:
                    records.append([fi, bi, key, bucket.pop(key)])
    packed["version"] = 2
    packed[PACKED_KEY] = _encode_records(records)
    return _pack_review_evidence(packed, max_bytes=max_bytes, owned=True)


def unpack_bucket_examples(data: dict[str, Any]) -> dict[str, Any]:
    """Restore exact membership arrays, with bounded decompression and strict targets."""
    sizes = [data[key].get("uncompressed_bytes") for key in (PACKED_KEY, REVIEW_PACKED_KEY)
             if isinstance(data.get(key), dict)]
    if (all(type(size) is int and size >= 0 for size in sizes)
            and sum(sizes) > MAX_UNPACKED_BYTES):
        raise ValueError("Invalid packed bucket examples: total decompression limit exceeded")
    data = _unpack_review_evidence(data)
    if PACKED_KEY not in data:
        return data
    packed = data[PACKED_KEY]
    try:
        records = _decode_records(packed)
        restored = copy.deepcopy(data)
        for fi, bi, key, ids in records:
            if (type(fi) is not int or type(bi) is not int or fi < 0 or bi < 0
                    or key not in EXAMPLE_KEYS or not isinstance(ids, list)
                    or any(not isinstance(item_id, str) for item_id in ids)):
                raise ValueError("Invalid packed bucket example record")
            bucket = restored["functions"][fi]["scores"]["buckets"][bi]
            if key in bucket:
                raise ValueError("Duplicate packed bucket example target")
            bucket[key] = ids
        del restored[PACKED_KEY]
        return restored
    except (KeyError, TypeError, IndexError, ValueError, zlib.error) as exc:
        raise ValueError(f"Invalid packed bucket examples: {exc}") from exc
