# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified artifact bytes shared by exploration, detectors, and evidence review.

Source adapters are optional. The generic store accepts an injected provider and
never follows URLs or filesystem paths supplied by artifact metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from autofa.data.execution_policy import is_evaluation_record, sanitize_metadata
from autofa.defaults import setting_value
from autofa.util.cancellation import check_cancelled

_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


@contextmanager
def _cache_lock(path: Path):
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(str(path), threading.RLock())
    while not lock.acquire(timeout=.25):
        check_cancelled()
    try:
        check_cancelled()
        if path.is_symlink():
            raise ValueError("Artifact lock must not be a symbolic link")
        with path.open("a+b") as stream:
            try:
                import fcntl
            except ImportError:
                fcntl = None
            if fcntl is not None:
                while True:
                    check_cancelled()
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(.25)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        lock.release()


class ArtifactProvider(Protocol):
    """Trusted source adapter configured by the application, never by detector input."""

    base_url: str

    def download_artifact(
        self, run_id: str, task_id: str, repeat: int, artifact_idx: int, dest: Path,
    ) -> Path: ...


def _base_url(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Artifact source API base URL is unavailable")
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Artifact source API base URL is invalid")
    return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(),
                                   parsed.path.rstrip("/"), "", ""))


def _digest(path: Path) -> tuple[str, int]:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest(), path.stat().st_size


def _catalog(item: dict, artifact_idx: int) -> tuple[dict, dict]:
    from autofa.trace import record

    if type(artifact_idx) is not int or artifact_idx < 0:
        raise ValueError("artifact_idx must be a nonnegative integer")
    execution = record(item)
    matches = [row for row in execution.get("artifacts", [])
               if type(row.get("artifact_idx")) is int and row["artifact_idx"] == artifact_idx
               and not is_evaluation_record(row)]
    if len(matches) != 1:
        raise ValueError("Artifact is absent, withheld, or ambiguous in the retained catalog")
    metadata = sanitize_metadata(matches[0])
    # Defense in depth for manually supplied envelopes: ambiguous evaluation
    # parentage is not permission to download a potentially grader-produced file.
    for event in execution.get("events", []):
        if (is_evaluation_record(event) and type(metadata.get("event_idx")) is int
                and event.get("event_idx") == metadata["event_idx"]
                and (metadata.get("lane") is None or metadata["lane"] == event.get("lane", ""))):
            raise ValueError("Artifact is associated with an evaluation event")
    return execution, metadata


def _resolve(execution: dict, metadata: dict, artifact_idx: int) -> tuple[dict, dict, dict]:
    rollout, source = execution.get("rollout"), execution.get("source")
    if not isinstance(rollout, dict) or not isinstance(source, dict):
        raise ValueError("Artifact download source was not recorded; reimport the source trace")
    if source.get("kind") not in {"trace_api", "local_evalrdl"}:
        raise ValueError("This artifact's source does not support verified byte access")
    identity = {}
    for key in ("run_id", "task_id", "repeat"):
        value = rollout.get(key)
        valid = type(value) is int and value >= 0 if key == "repeat" else (
            isinstance(value, str) and bool(value) and value not in {".", ".."}
        )
        if not valid or (metadata.get(key) is not None and metadata[key] != value):
            raise ValueError("Artifact identity is incomplete or differs from its rollout")
        identity[key] = value
    if (metadata.get("agent_ref") is not None and rollout.get("agent_ref") is not None
            and metadata["agent_ref"] != rollout["agent_ref"]):
        raise ValueError("Artifact agent identity differs from its rollout")
    if source["kind"] == "local_evalrdl":
        store_id = source.get("store_id")
        if (not isinstance(store_id, str) or not re.fullmatch(r"[0-9a-f]{64}", store_id)
                or source.get("run_id") != identity["run_id"]):
            raise ValueError("Local artifact source identity is invalid")
        identity.update(kind="local_evalrdl", store_id=store_id, artifact_idx=artifact_idx,
                        agent_ref=rollout.get("agent_ref"))
        return metadata, identity, source
    revision = source.get("trace_export_sha256")
    if revision is not None and (not isinstance(revision, str) or not _SHA256.fullmatch(revision)):
        raise ValueError("Artifact source revision is invalid")
    generation = source.get("export_generation")
    if generation is not None and not isinstance(generation, (str, int)):
        raise ValueError("Artifact source generation is invalid")
    identity.update(artifact_idx=artifact_idx, api_base_url=_base_url(source.get("base_url")),
                    trace_export_sha256=revision, export_generation=generation)
    return metadata, identity, source


class _LocalArtifactMissing(ValueError):
    pass


class _LocalArtifactProvider:
    """Copy captured bytes only from the registered store's exact content address."""

    def __init__(self, identity: dict):
        from autofa.data.local_trace_store import resolve_store

        directory = resolve_store(identity["store_id"])
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["run_id"] != identity["run_id"]:
            raise ValueError("Local artifact run differs from its registered source")
        records = manifest.get("artifact_files", [])
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise ValueError("Invalid local artifact inventory")
        matches = [row for row in records if all(row.get(key) == identity[key]
                   for key in ("task_id", "repeat", "artifact_idx"))]
        if not matches:
            raise _LocalArtifactMissing(
                "Original artifact bytes were not captured in this local store",
            )
        if len(matches) != 1 or matches[0].get("agent_ref") != identity["agent_ref"]:
            raise ValueError("Local artifact identity is ambiguous or differs from its rollout")
        record = matches[0]
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid managed artifact digest")
        if record.get("path") != f"artifacts/{digest}":
            raise ValueError("Invalid managed artifact path")
        self.path = directory / "artifacts" / digest
        if (self.path.parent.is_symlink() or self.path.is_symlink()
                or not self.path.is_file() or not self.path.resolve().is_relative_to(directory)):
            raise ValueError("Managed artifact is missing or is not a regular captured file")
        self.sha256, self.size_bytes = _digest(self.path)
        if self.sha256 != digest:
            raise ValueError("Managed artifact checksum changed")
        recorded_size = record.get("size_bytes")
        if recorded_size is None and manifest.get("adapter_version", 0) >= 6:
            raise ValueError("Managed artifact size is missing")
        if recorded_size is not None and (
            type(recorded_size) is not int or recorded_size < 0
        ):
            raise ValueError("Invalid managed artifact size")
        if recorded_size is not None and self.size_bytes != recorded_size:
            raise ValueError("Managed artifact size changed")

    def download_artifact(self, run_id, task_id, repeat, artifact_idx, dest):
        # _ensure_bytes checks the copied digest again, including changes during copying.
        shutil.copyfile(self.path, dest)
        return dest


class _RetainedImageProvider:
    """Copy verified inline bytes; never interpret an input URL or path."""

    def __init__(self, data: bytes):
        self.data = data

    def download_artifact(self, run_id, task_id, repeat, artifact_idx, dest):
        dest.write_bytes(self.data)
        return dest


class _ArchiveMemberProvider:
    """Read one exact member from a pinned archive, without unpacking its tree."""

    def __init__(self, saved: dict, member_path: str):
        self.saved, self.member_path = saved, member_path

    def download_artifact(self, run_id, task_id, repeat, artifact_idx, dest):
        from autofa.data.artifact_archive import extract_member

        extract_member(Path(self.saved["path"]), self.member_path, dest,
                       expected_sha256=self.saved["sha256"])
        return dest


class ArtifactStore:
    """Atomic content-addressed bytes and immutable per-source identity receipts."""

    def __init__(self, root: Path, provider: ArtifactProvider | None = None):
        self.root = Path(root).expanduser().resolve()
        self.provider = provider

    def _provider(self) -> ArtifactProvider | None:
        if self.provider is not None:
            return self.provider
        try:
            from autofa.config import load_config
            from autofa.data.remote_api import DataApiClient
        except ImportError:
            return None
        base = setting_value(load_config(), "data_api_base_url")
        return DataApiClient(_base_url(base)) if base else None

    def ensure(
        self, item: dict, artifact_idx: int, *, member_path: str | None = None,
    ) -> dict[str, Any]:
        if member_path is not None:
            return self._ensure_member(item, artifact_idx, member_path)
        metadata = None
        inline_bytes = None
        try:
            execution, metadata = _catalog(item, artifact_idx)
            if metadata.get("source_kind") == "retained_input_image":
                from autofa.data.inline_images import image_inventory, retained_image_bytes

                # The source's artifact metadata cannot grant access to an image
                # excluded by the input/evaluator policy. Recompute the descriptor.
                inventory, _ = image_inventory(item)
                comparison = {key: value for key, value in metadata.items()
                              if key not in {"artifact_idx", "name"}}
                if not any(comparison == {key: value for key, value in row.items()
                                          if key not in {"artifact_idx", "name"}}
                           for row in inventory):
                    raise ValueError("Retained-image descriptor is not in the input inventory.")
                inline_bytes, identity = retained_image_bytes(item, metadata)
                source = {"kind": "retained_input_image"}
            else:
                metadata, identity, source = _resolve(execution, metadata, artifact_idx)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            return {"status": "unavailable", "verified": False, "reason": str(exc),
                    **({"metadata": metadata} if metadata is not None else {})}
        result = {"metadata": deepcopy(metadata), "source_identity": identity, "verified": False}
        try:
            expected_size = metadata.get("size_bytes")
            if expected_size is not None and (type(expected_size) is not int or expected_size < 0):
                raise ValueError("Artifact declares an invalid byte size")
            expected_sha = metadata.get("sha256")
            if expected_sha is not None:
                if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
                    raise ValueError("Artifact declares an invalid SHA-256 digest")
                expected_sha = expected_sha.lower()
            if inline_bytes is not None:
                provider = _RetainedImageProvider(inline_bytes)
            elif source["kind"] == "local_evalrdl":
                provider = _LocalArtifactProvider(identity)
                if expected_sha is not None and expected_sha != provider.sha256:
                    raise ValueError("Local artifact digest differs from recorded metadata")
                if expected_size is not None and expected_size != provider.size_bytes:
                    raise ValueError("Local artifact size differs from recorded metadata")
                expected_sha, expected_size = provider.sha256, provider.size_bytes
            else:
                provider = self._provider()
                if provider is None:
                    return {**result, "status": "unsupported",
                            "reason": "No artifact source adapter"}
                if _base_url(provider.base_url) != identity["api_base_url"]:
                    raise ValueError("Artifact source differs from the configured trusted API")
            identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            key = hashlib.sha256(identity_json.encode()).hexdigest()
            result["cache_key"] = key
            blobs, receipts = self.root / "sha256", self.root / "receipts"
            for directory in (blobs, receipts):
                if directory.is_symlink():
                    raise ValueError("Artifact cache directory must not be a symbolic link")
                directory.mkdir(parents=True, exist_ok=True)
            with _cache_lock(receipts / f"{key}.lock"):
                return self._ensure_bytes(provider, result, expected_sha, expected_size, blobs,
                                          receipts / f"{key}.json")
        except _LocalArtifactMissing as exc:
            return {**result, "status": "unavailable", "reason": str(exc)}
        except (ValueError, OSError) as exc:
            return {**result, "status": "error", "reason": str(exc)}
        except Exception as exc:
            # Providers are optional; avoid importing private error types into public code.
            status = getattr(exc, "status", None)
            availability = ("unavailable" if status in {404, 410} else
                            "unsupported" if status in {405, 501} else "error")
            return {**result, "status": availability, "reason": str(exc)}

    def _ensure_member(self, item: dict, artifact_idx: int, member_path: str) -> dict:
        from autofa.data.artifact_archive import (
            ARCHIVE_POLICY,
            archive_format,
            member_metadata,
            validate_member_path,
        )

        result = {"member_path": member_path, "verified": False}
        try:
            validate_member_path(member_path)
            parent = self.ensure(item, artifact_idx)
            if parent.get("status") != "available":
                # Parent exported text is not evidence of the selected member.
                return {**result, "status": parent["status"],
                        "reason": parent.get("reason", "Original archive is unavailable")}
            result["archive_sha256"] = parent["sha256"]
            if not archive_format(parent["metadata"]):
                raise ValueError("member_path selection requires a ZIP artifact")
            metadata = member_metadata(Path(parent["path"]), member_path,
                                       expected_sha256=parent["sha256"])
            identity = {"kind": "archive_member", "archive_policy": ARCHIVE_POLICY,
                        "archive_source_identity": parent["source_identity"],
                        "archive_sha256": parent["sha256"], "member_path": member_path,
                        "artifact_idx": artifact_idx}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
            result.update(metadata=metadata, source_identity=identity, cache_key=key)
            # Original ensure has already validated/created these cache directories.
            blobs, receipts = self.root / "sha256", self.root / "receipts"
            with _cache_lock(receipts / f"{key}.lock"):
                return self._ensure_bytes(_ArchiveMemberProvider(parent, member_path), result,
                                          None, metadata["size_bytes"], blobs,
                                          receipts / f"{key}.json")
        except (ValueError, OSError, TypeError) as exc:
            return {**result, "status": "error", "reason": str(exc)}

    def _ensure_bytes(self, provider, result, expected_sha, expected_size, blobs, receipt_path):
        pinned = None
        if receipt_path.is_symlink():
            raise ValueError("Artifact receipt must not be a symbolic link")
        if receipt_path.exists():
            try:
                pinned = json.loads(receipt_path.read_text())
            except (OSError, ValueError) as exc:
                raise ValueError("Corrupt receipt; refusing to replace artifact pin") from exc
            if (not isinstance(pinned, dict) or pinned.get("version") != 1
                    or pinned.get("source_identity") != result["source_identity"]
                    or not isinstance(pinned.get("sha256"), str)
                    or not _SHA256.fullmatch(pinned["sha256"])
                    or type(pinned.get("size_bytes")) is not int or pinned["size_bytes"] < 0):
                raise ValueError("Artifact receipt is invalid; refusing to replace its pin")
            if expected_sha is not None and expected_sha != pinned["sha256"]:
                raise ValueError("Artifact declared digest conflicts with its pinned receipt")
            if expected_size is not None and expected_size != pinned["size_bytes"]:
                raise ValueError("Artifact declared size conflicts with its pinned receipt")
            expected_sha, expected_size = pinned["sha256"], pinned["size_bytes"]
        if expected_sha:
            cached = blobs / expected_sha
            if cached.is_file() and not cached.is_symlink():
                digest, size = _digest(cached)
                if digest == expected_sha and (expected_size is None or size == expected_size):
                    return self._finish(result, cached, digest, size, receipt_path, pinned)
        with tempfile.TemporaryDirectory(prefix="download-", dir=self.root) as temp:
            download = Path(temp) / "bytes"
            identity = result["source_identity"]
            provider.download_artifact(
                identity.get("run_id"), identity.get("task_id"), identity.get("repeat"),
                identity["artifact_idx"], download,
            )
            if not download.is_file() or download.is_symlink():
                raise ValueError("Artifact provider did not return a regular byte file")
            digest, size = _digest(download)
            if expected_size is not None and size != expected_size:
                raise ValueError("Artifact byte size differs from recorded metadata or pin")
            if expected_sha is not None and digest != expected_sha:
                raise ValueError("Artifact digest differs from recorded metadata or pin")
            destination = blobs / digest
            # The filename is computed from verified bytes, never a model-provided path.
            download.replace(destination)
            return self._finish(result, destination, digest, size, receipt_path, pinned)

    @staticmethod
    def _finish(result, path, digest, size, receipt_path, pinned):
        receipt = {"version": 1, "source_identity": result["source_identity"],
                   "sha256": digest, "size_bytes": size}
        if pinned is None:
            _pin_receipt(receipt_path, receipt)
        return {**result, "status": "available", "verified": True, "path": str(path),
                "sha256": digest, "size_bytes": size}


def _pin_receipt(receipt_path: Path, receipt: dict) -> None:
    # An atomic hard-link publication makes first-writer pinning exclusive;
    # interrupted writes never become visible as half-written receipts.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=receipt_path.parent,
                                         prefix="receipt-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(receipt, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, receipt_path)
        except FileExistsError:
            existing = json.loads(receipt_path.read_text())
            if existing != receipt:
                raise ValueError("Concurrent fetch differs from pinned bytes") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_artifact(
    item: dict, artifact_idx: int, *, store: ArtifactStore | None = None,
    member_path: str | None = None,
) -> dict:
    """Resolve a retained artifact and return verified immutable bytes, without extraction.

    No model-provided URL is used. Without declared hashes, the first verified
    download is pinned; subsequent cache repairs must reproduce those exact bytes.
    """
    if store is None:
        from autofa.config import autofa_home

        store = ArtifactStore(autofa_home() / "cache" / "artifacts")
    return store.ensure(item, artifact_idx, member_path=member_path)
