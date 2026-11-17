# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An unscored catalog of ordinary reusable functions, independent of detectors."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autofa.core.dependencies import digest, validate_bundle
from autofa.util.atomic import atomic_write_json

MAX_RESULT_BYTES = 256_000
_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def validate_schema(schema: dict) -> None:
    supported = {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "additionalProperties",
        "description",
        "maxItems",
        "maxLength",
    }
    if not isinstance(schema, dict) or set(schema) - supported:
        raise ValueError(f"unsupported schema; supported keys: {sorted(supported)}")
    if "type" in schema and schema["type"] not in _TYPES:
        raise ValueError(f"unsupported schema type {schema['type']!r}")
    if not isinstance(schema.get("properties", {}), dict):
        raise ValueError("schema properties must be an object")
    for child in schema.get("properties", {}).values():
        validate_schema(child)
    if "items" in schema:
        validate_schema(schema["items"])
    if not isinstance(schema.get("required", []), list) or not all(
        isinstance(key, str) for key in schema.get("required", [])
    ):
        raise ValueError("schema required must be an array of property names")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError("schema enum must be an array")
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        raise ValueError("schema additionalProperties must be a boolean")
    for key in ("maxItems", "maxLength"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"schema {key} must be a nonnegative integer")


def check_schema(value: Any, schema: dict, path: str = "result") -> None:
    """Small explicit JSON contract subset; unsupported schema keywords are rejected."""
    supported = {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "additionalProperties",
        "description",
        "maxItems",
        "maxLength",
    }
    if not isinstance(schema, dict) or set(schema) - supported:
        raise ValueError(f"{path}: unsupported schema; supported keys: {sorted(supported)}")
    kind = schema.get("type")
    if kind is not None:
        if kind not in _TYPES:
            raise ValueError(f"{path}: unsupported type {kind!r}")
        if not isinstance(value, _TYPES[kind]) or (
            kind in {"integer", "number"} and isinstance(value, bool)
        ):
            raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value outside enum")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ValueError(f"{path}: undeclared keys")
        for key, child in props.items():
            if key in value:
                check_schema(value[key], child, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) > schema.get("maxItems", MAX_RESULT_BYTES):
            raise ValueError(f"{path}: too many items")
        for i, item in enumerate(value):
            check_schema(item, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(value, str) and len(value) > schema.get("maxLength", MAX_RESULT_BYTES):
        raise ValueError(f"{path}: string too long")


def bounded_result(value: Any, schema: dict) -> Any:
    check_schema(value, schema)
    # Reject custom objects / generators and nonfinite numbers, and detach mutable values.
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > MAX_RESULT_BYTES:
        raise ValueError(f"helper result exceeds {MAX_RESULT_BYTES} bytes")
    return json.loads(encoded)


@dataclass
class HelperFunction:
    name: str
    source: str
    entrypoint: str
    description: str = ""
    purpose: str = "transform"
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    required_surfaces: list[str] = field(default_factory=list)
    origin: dict = field(default_factory=dict)
    status: str = "active"
    bundle: dict = field(default_factory=dict)

    def node(self) -> dict:
        return {
            "role": "helper",
            "source": self.source,
            "entrypoint": self.entrypoint,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "helpers": self.bundle.get("helpers", {}),
            "detectors": {},
        }

    @property
    def revision(self) -> str:
        return digest(self.node())

    def to_dict(self) -> dict:
        return {**asdict(self), "revision": self.revision}

    @classmethod
    def from_dict(cls, data: dict) -> HelperFunction:
        helper = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        functions = {
            n.name for n in ast.parse(helper.source).body if isinstance(n, ast.FunctionDef)
        }
        if helper.entrypoint not in functions:
            raise ValueError("helper entrypoint must name a top-level function")
        validate_schema(helper.input_schema)
        validate_schema(helper.output_schema)
        if data.get("revision", helper.revision) != helper.revision:
            raise ValueError(f"corrupt helper revision: {helper.name}")
        if helper.bundle:
            validate_bundle(helper.bundle, source=helper.source)
        return helper


@dataclass
class HelperLibrary:
    aliases: dict[str, str] = field(default_factory=dict)
    revisions: dict[str, HelperFunction] = field(default_factory=dict)

    def get(self, name_or_revision: str) -> HelperFunction | None:
        return self.revisions.get(self.aliases.get(name_or_revision, name_or_revision))

    def upsert(self, helper: HelperFunction) -> None:
        # Save a detached immutable executable revision; aliases select the latest one.
        self.revisions[helper.revision] = HelperFunction.from_dict(helper.to_dict())
        self.aliases[helper.name] = helper.revision

    def merge(self, other: HelperLibrary) -> None:
        for ref, helper in other.revisions.items():
            self.revisions.setdefault(ref, HelperFunction.from_dict(helper.to_dict()))
        for alias, ref in other.aliases.items():
            if alias in self.aliases and self.aliases[alias] != ref:
                alias = f"{alias}@{ref[:12]}"
            self.aliases[alias] = ref

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "aliases": dict(self.aliases),
            "revisions": {ref: helper.to_dict() for ref, helper in self.revisions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> HelperLibrary:
        if data.get("version", 1) != 1:
            raise ValueError("unsupported helper library version")
        library = cls(aliases=dict(data.get("aliases", {})))
        for ref, value in data.get("revisions", {}).items():
            helper = HelperFunction.from_dict(value)
            if ref != helper.revision:
                raise ValueError("helper revision key does not match content")
            library.revisions[ref] = helper
        if set(library.aliases.values()) - set(library.revisions):
            raise ValueError("helper alias refers to a missing revision")
        return library

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())
