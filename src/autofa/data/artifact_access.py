# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, auditable artifact evidence for discovery, detectors, and judges."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_READS: ContextVar[list[dict] | None] = ContextVar("artifact_reads", default=None)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


@contextmanager
def capture_artifact_reads():
    """Track actual artifact dependencies, including unavailable evidence."""
    reads: list[dict] = []
    token = _READS.set(reads)
    try:
        yield reads
    finally:
        _READS.reset(token)


def _record(
    value: dict, artifact_idx: int, *, page: int | None = None,
    member_path: str | None = None, transcription: dict | None = None,
) -> dict:
    reads = _READS.get()
    if reads is not None:
        # File locations are installation details; content and transform identities matter.
        evidence = {key: child for key, child in value.items()
                    if key not in {"path", "download_path", "data_url"}}
        receipt = {"artifact_idx": artifact_idx, "page": page,
                   "fingerprint": _fingerprint(evidence)}
        if member_path is not None:
            receipt["member_path"] = member_path
        if transcription is not None:
            receipt["transcription"] = transcription
        if receipt not in reads:
            reads.append(receipt)
    return value


def validate_artifact_reads(item: dict, receipts: list[dict]) -> bool:
    """Revalidate used evidence before reusing a detector prediction."""
    if not isinstance(receipts, list):
        return False
    for receipt in receipts:
        if (not isinstance(receipt, dict) or type(receipt.get("artifact_idx")) is not int
                or not isinstance(receipt.get("fingerprint"), str)):
            return False
        selection = {}
        if "member_path" in receipt:
            if not isinstance(receipt["member_path"], str) or not receipt["member_path"]:
                return False
            selection["member_path"] = receipt["member_path"]
        try:
            with capture_artifact_reads() as current:
                if "transcription" in receipt:
                    from autofa.data.artifact_speech_access import artifact_transcript

                    transcription = receipt["transcription"]
                    if (not isinstance(transcription, dict)
                            or set(transcription) != {"start_seconds", "end_seconds", "model"}
                            or receipt.get("page") is not None):
                        return False
                    artifact_transcript(item, receipt["artifact_idx"], **transcription,
                                        **selection)
                elif receipt.get("page") is None:
                    artifact_evidence(item, receipt["artifact_idx"], **selection)
                else:
                    artifact_image(item, receipt["artifact_idx"], page=receipt["page"],
                                   **selection)
        except (ValueError, KeyError, TypeError, OSError):
            return False
        if current != [receipt]:
            return False
    return True


@contextmanager
def _verified_snapshot(saved: dict):
    """Inspect a private copy of the pinned bytes, immune to concurrent cache repair."""
    source = Path(saved["path"])
    with tempfile.TemporaryDirectory(prefix="artifact-inspect-") as directory:
        snapshot = Path(directory) / "bytes"
        shutil.copyfile(source, snapshot)
        with snapshot.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != saved["sha256"] or snapshot.stat().st_size != saved["size_bytes"]:
            raise ValueError("Artifact bytes changed after cache verification")
        yield snapshot


def _output_directory(saved: dict, version: str) -> Path:
    root = Path(saved["path"]).parent
    output = root
    for component in ("derived", saved["sha256"], version):
        output = output / component
        if output.is_symlink():
            raise ValueError("Artifact derived cache must not contain symbolic links")
        output.mkdir(exist_ok=True)
    return output


def _unavailable(result: dict, exc: Exception) -> dict:
    return {**result, "status": "error", "verified": False, "complete": False,
            "segments": [], "reason": str(exc), "limitations": [str(exc)]}


def artifact_evidence(
    item: dict, artifact_idx: int, *, store=None, member_path: str | None = None,
) -> dict:
    """Fetch once and inspect a retained artifact; no arbitrary paths or URLs accepted."""
    from autofa.data.artifact_content import ARTIFACT_TRANSFORM_VERSION, inspect_artifact
    from autofa.data.artifact_store import ensure_artifact

    selection = {"member_path": member_path} if member_path is not None else {}
    saved = ensure_artifact(item, artifact_idx, store=store, **selection)
    result = {**saved, "artifact_idx": artifact_idx,
              "transform_version": ARTIFACT_TRANSFORM_VERSION,
              "segments": [], "complete": False, "limitations": []}
    if saved.get("status") != "available":
        result["limitations"] = [saved.get("reason", "Artifact bytes unavailable")]
        metadata = saved.get("metadata") or {}
        if isinstance(metadata.get("text"), str):
            result["segments"] = [{"locator": "exported_text", "text": metadata["text"]}]
            result["status"] = "partial"
            result["limitations"].append("Exported text only; original bytes were not verified.")
        return _record(result, artifact_idx, **selection)
    path = Path(saved["path"])
    try:
        output = _output_directory(saved, ARTIFACT_TRANSFORM_VERSION)
        with _verified_snapshot(saved) as snapshot:
            parsed = inspect_artifact(snapshot, saved["metadata"], output_dir=output)
        if parsed.get("sha256") not in {None, saved["sha256"]}:
            raise ValueError("Artifact extraction differs from the pinned source bytes")
    except (ValueError, OSError) as exc:
        return _record(_unavailable(result, exc), artifact_idx, **selection)
    result.update(parsed)
    result.update(artifact_idx=artifact_idx, source_identity=saved["source_identity"],
                  metadata=saved["metadata"], sha256=saved["sha256"], download_path=str(path))
    return _record(result, artifact_idx, **selection)


def _render_pinned(saved: dict, output: Path, *, page: int) -> dict:
    """Persist the first successful rendering; repairs must reproduce its exact evidence."""
    from autofa.data.artifact_content import ARTIFACT_TRANSFORM_VERSION, render_artifact
    from autofa.data.artifact_store import _cache_lock, _pin_receipt

    identity = {
        "artifact_sha256": saved["sha256"], "transform_version": ARTIFACT_TRANSFORM_VERSION,
        "page": page,
        "format_metadata": {key: saved["metadata"].get(key)
                            for key in ("name", "mime", "mime_type")},
    }
    key = _fingerprint(identity)
    manifest = output / f"render-{key}.json"
    with _cache_lock(output / f"render-{key}.lock"):
        pinned = None
        if manifest.is_symlink():
            raise ValueError("Artifact render receipt must not be a symbolic link")
        if manifest.exists():
            pinned = json.loads(manifest.read_text())
            if (not isinstance(pinned, dict) or pinned.get("identity") != identity
                    or pinned.get("version") != 1 or not isinstance(pinned.get("render"), dict)):
                raise ValueError("Invalid pinned artifact render receipt")
            rendered = pinned["render"]
            digest = rendered.get("sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or rendered.get("status") != "available"):
                raise ValueError("Invalid pinned artifact image digest")
            target = output / f"{digest}.png"
            if target.is_file() and not target.is_symlink():
                with target.open("rb") as handle:
                    current = hashlib.file_digest(handle, "sha256").hexdigest()
                if current == digest:
                    return {**rendered, "path": str(target)}
        with _verified_snapshot(saved) as snapshot:
            rendered = render_artifact(snapshot, saved["metadata"], output_dir=output, page=page)
        if rendered.get("status") != "available":
            return rendered  # Installing an optional renderer may make a later attempt succeed.
        if rendered.get("original_sha256") != saved["sha256"]:
            raise ValueError("Artifact render differs from the pinned source bytes")
        target = Path(rendered.get("path", ""))
        if target.is_symlink() or not target.resolve().is_relative_to(output.resolve()):
            raise ValueError("Artifact render is outside its verified derived cache")
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != rendered.get("sha256"):
            raise ValueError("Rendered artifact bytes differ from their reported digest")
        evidence = {key: value for key, value in rendered.items() if key != "path"}
        receipt = {"version": 1, "identity": identity, "render": evidence}
        if pinned is not None and receipt != pinned:
            raise ValueError("Artifact render repair differs from the pinned image")
        destination = output / f"{digest}.png"
        if target != destination:
            target.replace(destination)
        if pinned is None:
            _pin_receipt(manifest, receipt)
        return {**evidence, "path": str(destination)}


def artifact_image(
    item: dict, artifact_idx: int, *, page: int = 1, store=None,
    member_path: str | None = None,
) -> dict:
    """Fetch and render one artifact page; its actual bytes can be supplied to a vision model."""
    from autofa.data.artifact_content import ARTIFACT_TRANSFORM_VERSION
    from autofa.data.artifact_store import ensure_artifact

    if type(page) is not int or page < 1:
        raise ValueError("page must be a positive integer")
    selection = {"member_path": member_path} if member_path is not None else {}
    saved = ensure_artifact(item, artifact_idx, store=store, **selection)
    result = {**saved, "artifact_idx": artifact_idx, "page": page,
              "artifact_sha256": saved.get("sha256"),
              "transform_version": ARTIFACT_TRANSFORM_VERSION}
    if saved.get("status") != "available":
        return _record(result, artifact_idx, page=page, **selection)
    try:
        output = _output_directory(saved, ARTIFACT_TRANSFORM_VERSION)
        rendered = _render_pinned(saved, output, page=page)
    except (ValueError, OSError) as exc:
        return _record(_unavailable(result, exc), artifact_idx, page=page, **selection)
    result.update(rendered)
    result.update(artifact_sha256=saved["sha256"], source_identity=saved["source_identity"])
    if rendered.get("status") == "available" and rendered.get("path"):
        image_path = Path(rendered["path"])
        if image_path.is_symlink() or not image_path.resolve().is_relative_to(output.resolve()):
            raise ValueError("Artifact render is outside its verified derived cache")
        if image_path.stat().st_size > 12 * 1024 * 1024:
            result.update(status="unavailable",
                          reason="Rendered image exceeds 12 MiB delivery limit")
        else:
            data = image_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != rendered.get("sha256"):
                raise ValueError("Artifact render changed after validation")
            mime = rendered["mime_type"]
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValueError("Unsupported rendered image media type")
            result["data_url"] = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return _record(result, artifact_idx, page=page, **selection)


def image_content(value: dict) -> dict:
    """Convert an available image receipt to an MCP image, never a remote URL."""
    if value.get("status") != "available" or not isinstance(value.get("data_url"), str):
        raise ValueError(value.get("reason") or "Artifact image is unavailable")
    header, encoded = value["data_url"].split(",", 1)
    mime = header.removeprefix("data:").removesuffix(";base64")
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError("Unsupported image media type")
    return {"type": "image", "mimeType": mime, "data": encoded}
