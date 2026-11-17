# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream complete managed trace snapshots to an explicitly selected server."""

from __future__ import annotations

import json
import re
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from autofa.data.local_trace_store import file_digest, resolve_store
from autofa.data.remote_library_api import RemoteLibraryClient, RemoteLibraryError

_DIGEST = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class TraceArchive:
    """A repeatable, bounded-memory tar stream over an immutable source store."""

    files: tuple[tuple[str, Path, int], ...]

    @property
    def size(self) -> int:
        return 1024 + sum(512 + ((size + 511) // 512) * 512 for _, _, size in self.files)

    def __iter__(self) -> Iterator[bytes]:
        for name, path, size in self.files:
            if path.is_symlink() or path.stat().st_size != size:
                raise RemoteLibraryError("A local trace file changed before upload")
            header = tarfile.TarInfo(name)
            header.size, header.mode = size, 0o600
            # GNU numeric fields represent individual files beyond USTAR's 8 GiB
            # size ceiling without adding any extension/long-name records.
            yield header.tobuf(format=tarfile.GNU_FORMAT if size >= 8 ** 11
                              else tarfile.USTAR_FORMAT)
            remaining = size
            with path.open("rb") as stream:
                while remaining:
                    chunk = stream.read(min(remaining, _CHUNK))
                    if not chunk:
                        raise RemoteLibraryError("A local trace file changed during upload")
                    remaining -= len(chunk)
                    yield chunk
                if stream.read(1):
                    raise RemoteLibraryError("A local trace file changed during upload")
            if size % 512:
                yield b"\0" * (512 - size % 512)
        yield b"\0" * 1024


def trace_archive(store_id: str) -> tuple[TraceArchive, dict, str]:
    """Select only validated managed data, never arbitrary paths from metadata."""
    store = resolve_store(store_id)
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = {"manifest.json": manifest_path, "dataset.parquet": store / "dataset.parquet"}
    for name in manifest["tables"]:
        files[f"tables/{name}"] = store / "tables" / name
    for artifact in manifest.get("artifact_files", []):
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise RemoteLibraryError("Invalid local trace artifact digest")
        name = f"artifacts/{digest}"
        if artifact.get("path") != name:
            raise RemoteLibraryError("Invalid local trace artifact path")
        path = store / name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(store)
                or file_digest(path) != digest):
            raise RemoteLibraryError("A local trace artifact is unavailable or changed")
        files[name] = path
    # The receiver reads the manifest first to establish the exact permitted inventory.
    ordered = ["manifest.json", *sorted(set(files) - {"manifest.json"})]
    archive = TraceArchive(tuple((name, files[name], files[name].stat().st_size)
                                 for name in ordered))
    return archive, manifest, file_digest(manifest_path)


def server_capabilities(client: RemoteLibraryClient) -> dict:
    """Require an authenticated, compatible receiver before sending full traces."""
    try:
        response = client._request_json("GET", "/api/v2/capabilities")
    except RemoteLibraryError as exc:
        if exc.status in {404, 405}:
            raise RemoteLibraryError(
                "This server does not support complete trace uploads. Upgrade the receiving "
                "AutoFA server, or select lightweight report publication."
            ) from exc
        raise
    capability = response.get("managed_trace_upload") if isinstance(response, Mapping) else None
    if (not isinstance(capability, dict) or capability.get("version") != 1
            or capability.get("format") != "tar"):
        raise RemoteLibraryError("The selected server does not support complete trace uploads")
    for key in ("max_upload_bytes", "max_unpacked_bytes", "max_files", "max_manifest_bytes"):
        if type(capability.get(key)) is not int or capability[key] <= 0:
            raise RemoteLibraryError("The selected server returned invalid trace upload limits")
    return capability


def upload_trace_stores(
    client: RemoteLibraryClient, references: list[dict], *, capabilities: dict,
) -> list[dict[str, str]]:
    """Transfer each missing snapshot, then return path-free publication bindings."""
    uploaded, seen = [], {}
    for reference in references:
        store_id = reference.get("store_id", "")
        if (reference.get("kind") != "local_evalrdl" or not isinstance(store_id, str)
                or not _DIGEST.fullmatch(store_id)):
            raise RemoteLibraryError("Complete trace publication requires managed local sources")
        if store_id in seen:
            if seen[store_id] != reference.get("run_id"):
                raise RemoteLibraryError("Conflicting run identities for one local trace snapshot")
            continue
        seen[store_id] = reference.get("run_id")
        archive, manifest, digest = trace_archive(store_id)
        if manifest["run_id"] != reference["run_id"]:
            raise RemoteLibraryError("Local trace reference does not match its stored run")
        path = f"/api/v2/trace-stores/{store_id}"
        try:
            remote = client._request_json("GET", path)
        except RemoteLibraryError as exc:
            if exc.status != 404:
                raise
            if (archive.size > capabilities["max_upload_bytes"]
                    or sum(size for _, _, size in archive.files)
                    > capabilities["max_unpacked_bytes"]
                    or len(archive.files) > capabilities["max_files"]
                    or archive.files[0][2] > capabilities["max_manifest_bytes"]):
                raise RemoteLibraryError(
                    "Local trace snapshot exceeds the receiving server limits"
                ) from None
            remote = client._request_json(
                "PUT", path, body=archive,
                headers={"Content-Type": "application/x-tar", "Content-Length": str(archive.size)},
            )
        if (not isinstance(remote, Mapping) or remote.get("store_id") != store_id
                or remote.get("run_id") != manifest["run_id"]
                or remote.get("manifest_sha256") != digest or remote.get("status") != "present"):
            raise RemoteLibraryError("The server trace snapshot does not match the local source")
        uploaded.append({"kind": "local_evalrdl", "store_id": store_id,
                         "run_id": reference["run_id"],
                         "source_harness": reference.get("source_harness") or "unknown"})
    return uploaded
