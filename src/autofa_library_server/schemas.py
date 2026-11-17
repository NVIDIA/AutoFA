# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .data_identity import normalize_data_identity

PUBLICATION_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"


class Identity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int = Field(gt=0)
    username: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    avatar_url: str = ""
    web_url: str = ""
    email: str = ""


class ReferenceDataRecord(BaseModel):
    """Reserved compatibility slot; local reports carry their input evidence."""

    @model_validator(mode="before")
    @classmethod
    def reject_external_reference(cls, value: Any) -> Any:
        raise ValueError("Use local trajectory evidence instead of external references")


class UploadMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_identity: dict[str, Any] | None = None
    source_run_id: str | None = None
    model_id: str = Field(min_length=1, max_length=500)
    benchmarks: list[str] = Field(default_factory=list, max_length=100)
    harness: str = Field(default="unknown", min_length=1, max_length=100)
    source_harness: str = Field(default="unknown", min_length=1, max_length=100)
    reference_data: list[ReferenceDataRecord] = Field(
        default_factory=list,
        max_length=100,
    )
    publish_trigger: Literal[
        "manual_cli",
        "browse_menu",
        "session_end",
        "local_catalog",
        "unknown",
    ] = "unknown"

    @field_validator("data_identity", mode="before")
    @classmethod
    def validate_data_identity(cls, value: Any) -> dict | None:
        return normalize_data_identity(value)

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_id cannot be empty")
        return value

    @field_validator("harness", "source_harness")
    @classmethod
    def validate_harness(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("harness cannot be empty")
        return value

    @field_validator("benchmarks")
    @classmethod
    def validate_benchmarks(cls, value: list[str]) -> list[str]:
        cleaned: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("benchmarks must contain non-empty strings")
            benchmark = item.strip()
            if len(benchmark) > 255:
                raise ValueError("benchmark names must be at most 255 characters")
            cleaned.add(benchmark)
        return sorted(cleaned)


class PublicationMetadata(UploadMetadata):
    source_run_id: str = Field(pattern=PUBLICATION_KEY_PATTERN)
    branch_key: str = Field(pattern=PUBLICATION_KEY_PATTERN)
    benchmarks: list[str] = Field(max_length=100)
    library_id: str | None = Field(default=None, min_length=1, max_length=100)
    branch_id: str | None = Field(default=None, min_length=1, max_length=100)


class LibraryRecord(BaseModel):
    data_identity: dict[str, Any] | None = None
    data_identities: list[dict[str, Any]] = Field(default_factory=list)
    data_group_count: int = 0
    is_merged: bool = False
    data_identity_status: Literal["known", "unknown", "legacy_mixed"] = "unknown"
    id: str
    owner_id: int
    owner_username: str
    owner_name: str
    owner_avatar: str
    source_run_id: str
    model_id: str
    benchmarks: list[str]
    function_count: int
    created_at: datetime
    updated_at: datetime
    revision: int
    library_sha256: str
    report_sha256: str
    latest_branch_id: str
    latest_version_id: str
    branch_count: int
    version_count: int
    harness: str
    source_harness: str
    reference_data: list[ReferenceDataRecord]
    publish_trigger: str
    is_public: bool


class LibraryPage(BaseModel):
    items: list[LibraryRecord]
    total: int
    limit: int
    offset: int


class BranchRecord(BaseModel):
    id: str
    library_id: str
    branch_key: str
    created_at: datetime
    updated_at: datetime
    latest_version_id: str
    latest_version_number: int
    latest_source_run_id: str
    version_count: int


class BranchPage(BaseModel):
    items: list[BranchRecord]
    total: int
    limit: int
    offset: int


class VersionRecord(BaseModel):
    data_identity: dict[str, Any] | None = None
    id: str
    library_id: str
    branch_id: str
    branch_key: str
    version_number: int
    source_run_id: str
    model_id: str
    benchmarks: list[str]
    function_count: int
    created_at: datetime
    library_sha256: str
    report_sha256: str
    library_url: str
    failure_card_url: str
    harness: str
    source_harness: str
    reference_data: list[ReferenceDataRecord]
    publish_trigger: str
    is_public: bool


class VersionPage(BaseModel):
    items: list[VersionRecord]
    total: int
    limit: int
    offset: int


class PublicationRecord(BaseModel):
    data_identity: dict[str, Any] | None = None
    library_id: str
    branch_id: str
    version_id: str
    branch_key: str
    version_number: int
    source_run_id: str
    model_id: str
    benchmarks: list[str]
    function_count: int
    created_at: datetime
    library_sha256: str
    report_sha256: str
    library_url: str
    failure_card_url: str
    harness: str
    source_harness: str
    reference_data: list[ReferenceDataRecord]
    publish_trigger: str
    is_public: bool


class MeResponse(Identity):
    auth_method: str


def validate_library_payload(data: Any) -> tuple[str, list[str], int]:
    if not isinstance(data, dict):
        raise ValueError("library JSON root must be an object")
    if type(data.get("version")) is not int or data.get("version") not in {1, 2}:
        raise ValueError("library version must be 1 or 2")
    if data["version"] == 2 or "contributions" in data:
        _validate_library_extensions(data)

    model_id = data.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("library model_id must be a non-empty string")
    model_id = model_id.strip()
    if len(model_id) > 500:
        raise ValueError("library model_id must be at most 500 characters")

    benchmark = data.get("benchmark", "")
    if not isinstance(benchmark, str):
        raise ValueError("library benchmark must be a string")
    benchmarks = sorted({item.strip() for item in benchmark.split("+") if item.strip()})

    functions = data.get("functions")
    if not isinstance(functions, list):
        raise ValueError("library functions must be a list")
    names: set[str] = set()
    relationships: list[tuple[int, list[str], str | None]] = []
    for index, function in enumerate(functions):
        if not isinstance(function, dict):
            raise ValueError(f"functions[{index}] must be an object")
        for field in ("name", "description", "source"):
            value = function.get(field)
            if not isinstance(value, str):
                raise ValueError(f"functions[{index}].{field} must be a string")
        name = function["name"].strip()
        if not name:
            raise ValueError(f"functions[{index}].name cannot be empty")
        if len(name) > 255:
            raise ValueError(f"functions[{index}].name is too long")
        if name in names:
            raise ValueError(f"duplicate function name: {name}")
        names.add(name)
        if not function["source"].strip():
            raise ValueError(f"functions[{index}].source cannot be empty")
        kind = function.get("kind", "regex")
        if not isinstance(kind, str) or kind not in {"regex", "llm", "python"}:
            raise ValueError(
                f"Invalid function kind {kind!r} for detector {function['name']!r}; expected "
                "'regex', 'llm' or 'python'. Retired strategy labels ran as Python: set kind "
                "to 'python' or recreate the detector."
            )
        depends_on = function.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(dependency, str) for dependency in depends_on
        ):
            raise ValueError(f"functions[{index}].depends_on must be a string list")
        explains = function.get("explains")
        if explains is not None and not isinstance(explains, str):
            raise ValueError(f"functions[{index}].explains must be a string or null")
        relationships.append((index, depends_on, explains))

    for index, depends_on, explains in relationships:
        # V2 pins executable snapshots, including parents archived from the active
        # catalog. Clients validate hashes/graphs before executing downloaded code.
        pinned = {}
        if data["version"] == 2:
            bundle = (functions[index].get("meta") or {}).get("dependency_bundle") or {}
            pinned = bundle.get("detectors", {})
        unknown = [dependency for dependency in depends_on
                   if dependency not in names and dependency not in pinned]
        if unknown:
            raise ValueError(
                f"functions[{index}].depends_on references unknown function: {unknown[0]}"
            )
        if explains is not None and explains not in names and explains not in pinned:
            raise ValueError(f"functions[{index}].explains references unknown function: {explains}")

    return model_id, benchmarks, len(functions)


def _validate_library_extensions(data: dict) -> None:
    """Validate bounded restored evidence and metadata without executing detector code."""
    from autofa.core.library_transport import unpack_bucket_examples
    from autofa.session.contributions import merge_records

    # Validate restored evidence before accepting an immutable upload. Merely
    # checking a base64 envelope would accept corrupt or conflicting history.
    data = unpack_bucket_examples(data)
    merge_records(data.get("contributions", []))
    helpers = data.get("helper_library", {})
    if not isinstance(helpers, dict) or helpers.get("version", 1) != 1:
        raise ValueError("helper_library must be a version 1 object")
    aliases, revisions = helpers.get("aliases", {}), helpers.get("revisions", {})
    if not isinstance(aliases, dict) or not isinstance(revisions, dict):
        raise ValueError("helper aliases and revisions must be objects")
    for ref, helper in revisions.items():
        if not re.fullmatch(r"[0-9a-f]{64}", ref) or not isinstance(helper, dict):
            raise ValueError("invalid helper revision")
        for key in ("name", "source", "entrypoint"):
            if not isinstance(helper.get(key), str) or not helper[key].strip():
                raise ValueError(f"helper {key} must be a non-empty string")
    if any(not isinstance(ref, str) or ref not in revisions for ref in aliases.values()):
        raise ValueError("helper alias references a missing revision")
    for fn in data.get("functions", []) if isinstance(data.get("functions"), list) else []:
        if not isinstance(fn, dict):
            continue  # The common function validation reports the precise error.
        uses = fn.get("uses_helpers", {})
        if not isinstance(uses, dict) or not all(isinstance(ref, str) for ref in uses.values()):
            raise ValueError("uses_helpers must be an alias-to-revision object")
        meta = fn.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError("function meta must be an object")
        bundle = meta.get("dependency_bundle")
        if bundle is None:
            if uses:
                raise ValueError("helper consumers require a pinned dependency_bundle")
            continue
        if not isinstance(bundle, dict) or bundle.get("version") != 1:
            raise ValueError("dependency_bundle must be a version 1 object")
        nodes = bundle.get("nodes")
        if not isinstance(nodes, dict) or len(nodes) > 128:
            raise ValueError("dependency_bundle requires at most 128 nodes")
        for key in ("helpers", "detectors"):
            edges = bundle.get(key, {})
            if not isinstance(edges, dict) or any(
                not isinstance(ref, str) or ref not in nodes for ref in edges.values()
            ):
                raise ValueError("dependency_bundle references a missing pinned node")
        if not set(uses) <= set(bundle.get("helpers", {})):
            raise ValueError("uses_helpers references an unpinned helper")
