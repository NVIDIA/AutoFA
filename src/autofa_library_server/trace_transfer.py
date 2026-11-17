# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated, bounded transfer of immutable managed trace stores.

The wire archive contains regular USTAR files only. No uploaded path, source
sidecar, or report descriptor is authority to read the server's filesystem.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

from fastapi import HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from autofa.core.library import FunctionLibrary
from autofa.data.local_trace_store import (
    _verify_file,
    file_digest,
    resolve_store,
    source_store_root,
)
from autofa.data.reference_data import make_local_reference_data

from .auth import AuthContext
from .lazy_report import build_lazy_report

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TABLE = re.compile(r"tables/[a-z][a-z0-9_]*\.parquet\Z")
_ARTIFACT = re.compile(r"artifacts/[0-9a-f]{64}\Z")
_BLOCK = 512
_CHUNK = 1024 * 1024


def _check_id(store_id: str) -> None:
    if not _DIGEST.fullmatch(store_id):
        raise HTTPException(status_code=422, detail="Invalid trace store ID")


def _inventory(store: Path, settings) -> dict:
    """Validate a complete managed store, including every captured artifact."""
    manifest_path = store / "manifest.json"
    if manifest_path.stat().st_size > settings.max_trace_manifest_bytes:
        raise ValueError("Trace manifest exceeds the configured limit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Invalid trace manifest")
    records = manifest.get("artifact_files", [])
    if not isinstance(records, list) or len(records) > settings.max_trace_files:
        raise ValueError("Invalid trace artifact inventory")
    expected = {"manifest.json", "dataset.parquet"}
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("Invalid trace table inventory")
    expected.update(f"tables/{name}" for name in tables)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid trace artifact record")
        digest, name = record.get("sha256"), record.get("path")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError("Invalid trace artifact checksum")
        if name != f"artifacts/{digest}":
            raise ValueError("Invalid trace artifact path")
        _verify_file(store / name, digest)
        expected.add(name)
    if len(expected) > settings.max_trace_files:
        raise ValueError("Trace store contains too many files")
    resolve_store(store.name, store.parent)
    return {"manifest": manifest, "files": expected}


def _receipt(manifest: dict) -> dict:
    return {"store_id": manifest["store_id"], "run_id": manifest["run_id"],
            "source_harness": manifest.get("source_harness") or "unknown",
            "manifest_sha256": file_digest(
                source_store_root() / manifest["store_id"] / "manifest.json"),
            "status": "present"}


def _existing(store_id: str, settings) -> dict:
    _check_id(store_id)
    root = source_store_root()
    if not (root / store_id).exists() and not (root / store_id).is_symlink():
        raise HTTPException(status_code=404, detail="Trace store not found")
    try:
        store = resolve_store(store_id)
        return _inventory(store, settings)["manifest"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=409, detail="Existing trace store is invalid") from exc


def _unpack(archive: Path, destination: Path, settings) -> set[str]:
    """Parse one plain tar without following links or expanding archive extensions."""
    names: set[str] = set()
    total = 0
    with archive.open("rb") as stream:
        while True:
            header = stream.read(_BLOCK)
            if len(header) != _BLOCK:
                raise ValueError("Truncated trace archive")
            if header == bytes(_BLOCK):
                if stream.read(_BLOCK) != bytes(_BLOCK):
                    raise ValueError("Invalid trace archive terminator")
                while chunk := stream.read(_CHUNK):
                    if any(chunk):
                        raise ValueError("Unexpected data after trace archive")
                return names
            try:
                member = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
            except (tarfile.HeaderError, UnicodeError, ValueError) as exc:
                raise ValueError("Invalid trace archive header") from exc
            name = member.name
            if (member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                    or member.linkname or member.size < 0
                    or not (name in {"manifest.json", "dataset.parquet"}
                            or _TABLE.fullmatch(name) or _ARTIFACT.fullmatch(name))):
                raise ValueError("Trace archive contains an unsupported path or entry")
            if name in names:
                raise ValueError("Trace archive contains duplicate paths")
            names.add(name)
            total += member.size
            if len(names) > settings.max_trace_files or total > settings.max_trace_unpacked_bytes:
                raise HTTPException(status_code=413, detail="Trace archive exceeds unpacked limits")
            if name == "manifest.json" and member.size > settings.max_trace_manifest_bytes:
                raise HTTPException(status_code=413, detail="Trace manifest exceeds the limit")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = stream.read(min(remaining, _CHUNK))
                    if not chunk:
                        raise ValueError("Truncated trace archive file")
                    output.write(chunk)
                    remaining -= len(chunk)
            padding = (-member.size) % _BLOCK
            if stream.read(padding) != bytes(padding):
                raise ValueError("Invalid trace archive padding")


def _install(archive: Path, staging: Path, store_id: str, settings) -> tuple[dict, bool]:
    directory = staging / store_id
    directory.mkdir()
    try:
        names = _unpack(archive, directory, settings)
        checked = _inventory(directory, settings)
        if names != checked["files"]:
            raise ValueError("Trace archive does not match the manifest inventory")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid trace store: {exc}") from exc
    manifest = checked["manifest"]
    # Only the destination profile owns filesystem paths in local references.
    destination = source_store_root() / store_id
    reference = make_local_reference_data(
        store_id=store_id, run_id=manifest["run_id"],
        source_harness=manifest.get("source_harness") or "unknown",
        parquet_path=destination / "dataset.parquet",
    )
    (directory / "source.json").write_text(json.dumps(reference), encoding="utf-8")

    def reuse_existing():
        existing = _existing(store_id, settings)
        if file_digest(directory / "manifest.json") != file_digest(destination / "manifest.json"):
            raise HTTPException(status_code=409, detail="Trace store identity already exists")
        return _receipt(existing), False

    # rename can replace an empty directory on Unix. Even an incomplete existing
    # identity belongs to the profile and must fail closed instead of being repaired.
    if destination.exists() or destination.is_symlink():
        return reuse_existing()
    try:
        os.rename(directory, destination)
        return _receipt(manifest), True
    except OSError:
        if not destination.exists() and not destination.is_symlink():
            raise
    return reuse_existing()


def report_for_transferred_traces(settings, raw: str, library_payload: dict, harness: str):
    """Build trusted descriptors only after validating explicit installed identities."""
    if len(raw.encode("utf-8")) > settings.max_metadata_bytes:
        raise HTTPException(status_code=413, detail="Trace references exceed the limit")
    try:
        references = json.loads(raw)
        if not isinstance(references, list) or not 1 <= len(references) <= 100:
            raise ValueError("Expected between one and 100 trace references")
        local = []
        seen = set()
        allowed = {"kind", "store_id", "run_id", "source_harness"}
        for reference in references:
            if (not isinstance(reference, dict) or set(reference) - allowed
                    or reference.get("kind") != "local_evalrdl"
                    or not isinstance(reference.get("store_id"), str)):
                raise ValueError("Expected managed trace references without filesystem paths")
            store_id = reference["store_id"]
            manifest = _existing(store_id, settings)
            if reference.get("run_id") != manifest["run_id"] or store_id in seen:
                raise ValueError("Trace reference identity mismatch or duplicate")
            seen.add(store_id)
            local.append(make_local_reference_data(
                store_id=store_id, run_id=manifest["run_id"],
                source_harness=manifest.get("source_harness") or "unknown",
                parquet_path=source_store_root() / store_id / "dataset.parquet",
            ))
        report = build_lazy_report(FunctionLibrary.from_dict(library_payload), local,
                                   source_harness=harness)
        if report is None:
            raise ValueError("Transferred traces cannot be resolved")
        from autofa.reports.failure_cards import _collect_example_ids, _extract_autofa_data

        data = _extract_autofa_data(report[0].decode("utf-8"))
        if not _collect_example_ids(data["functions"]) <= data["item_index"].keys():
            raise ValueError("Transferred traces do not cover every report example")
        return report
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid trace references: {exc}") from exc


def mount_transfer_routes(application, settings, auth_dependency) -> None:
    def require_token(context):
        if context.method != "local_token":
            raise HTTPException(status_code=401, detail="A server bearer token is required")

    @application.get("/api/v2/capabilities", tags=["system"])
    def capabilities(context: AuthContext = auth_dependency):
        require_token(context)
        return {"managed_trace_upload": {
            "version": 1, "format": "tar", "max_upload_bytes": settings.max_trace_upload_bytes,
            "max_unpacked_bytes": settings.max_trace_unpacked_bytes,
            "max_files": settings.max_trace_files,
            "max_manifest_bytes": settings.max_trace_manifest_bytes,
        }}

    @application.get("/api/v2/trace-stores/{store_id}", tags=["traces"])
    def get_store(store_id: str, context: AuthContext = auth_dependency):
        require_token(context)
        return _receipt(_existing(store_id, settings))

    @application.put("/api/v2/trace-stores/{store_id}", tags=["traces"])
    async def put_store(store_id: str, request: Request, response: Response,
                        context: AuthContext = auth_dependency):
        require_token(context)
        _check_id(store_id)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type not in {"application/x-tar", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="A plain tar archive is required")
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid content length") from exc
        if length < 0 or length > settings.max_trace_upload_bytes:
            raise HTTPException(status_code=413, detail="Trace upload exceeds the configured limit")
        root = source_store_root()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".upload-", dir=root) as scratch:
            staging = Path(scratch)
            archive = staging / "store.tar"
            size = 0
            with archive.open("wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > settings.max_trace_upload_bytes:
                        raise HTTPException(status_code=413,
                                            detail="Trace upload exceeds the limit")
                    await run_in_threadpool(output.write, chunk)
            result, created = await run_in_threadpool(
                _install, archive, staging, store_id, settings,
            )
        response.status_code = 201 if created else 200
        return result
