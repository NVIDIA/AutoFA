# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact files and shared located/visual evidence for the discovery agent."""

from __future__ import annotations

from typing import Any

from autofa.mcp.runtime import err, ok
from autofa.mcp.tools_data import _load_item_raw, _parsed_page


def download_artifact(
    item_id: str, artifact_idx: int, member_path: str | None = None,
) -> dict[str, Any]:
    """Download a retained artifact into the verified shared cache; returns its local path/hash.

    Find artifact_idx in get_trace_record(section='artifacts'). Only source artifact
    identities are accepted. No caller-provided URL or filesystem destination is used.
    """
    from autofa.data.artifact_store import ensure_artifact

    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    options = {"member_path": member_path} if member_path is not None else {}
    result = ensure_artifact(item, artifact_idx, **options)
    return {"ok": result.get("status") == "available", **result}


def read_artifact(
    item_id: str, artifact_idx: int, section: str = "manifest", pointer: str = "", *,
    offset: int = 0, limit: int = 20, max_chars: int = 16_000,
    member_path: str | None = None,
) -> dict[str, Any]:
    """Inspect an artifact with page/sheet/cell/slide/paragraph locations and full text paging.

    Start with manifest, then section='segments'; use pointer='/0/text' and next_offset
    for long segments. Downloads only the chosen retained artifact. Layout claims need
    view_artifact; extracted text is not proof that a chart or image was inspected.
    """
    from autofa import trace

    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    options = {"member_path": member_path} if member_path is not None else {}
    result = trace.artifact(item, artifact_idx, **options)
    if section == "manifest":
        value = {key: child for key, child in result.items() if key != "segments"}
        value["segment_count"] = len(result.get("segments", []))
    elif section in result:
        value = result[section]
    else:
        return err("unknown artifact section", available_sections=["manifest", *result])
    if pointer and not pointer.startswith("/"):
        return err("pointer must be empty or start with /")
    try:
        for token in pointer.split("/")[1:]:
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                index = int(token)
                if index < 0:
                    raise IndexError
                value = value[index]
            else:
                value = value[token]
    except (KeyError, IndexError, TypeError, ValueError):
        return err("invalid artifact pointer", pointer=pointer)
    return {**_parsed_page(value, pointer=pointer, offset=offset, limit=limit, max_chars=max_chars),
            "item_id": item_id, "artifact_idx": artifact_idx, "section": section,
            "member_path": member_path,
            "status": result.get("status"), "complete": result.get("complete", False),
            "sha256": result.get("sha256"), "source_identity": result.get("source_identity"),
            "limitations": result.get("limitations", []),
            "hint": "Follow next_offset/child_pointers. Missing content is not negative evidence."}


def view_artifact(
    item_id: str, artifact_idx: int, page: int = 1, member_path: str | None = None,
) -> dict[str, Any]:
    """View one 1-based artifact page/image/video frame with its source and hash receipt.

    PDFs/images use installed PDFium/Pillow; Office visual rendering requires LibreOffice.
    Videos use PyAV: page selects the decoded frame number. Read the artifact manifest
    for duration/frame metadata. A viewed frame does not inspect other frames or audio.
    Audio pages show 30-second waveform/spectrogram windows, not listening. Audio/video
    without the user-built PyAV or the file's codec is unavailable: an evidence gap, not
    absence. For ZIPs, choose member_path from read_artifact's archive inventory; only that
    file is extracted.
    Reports missing renderers, unsupported media and page bounds explicitly.
    """
    from autofa import trace

    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    options = {"member_path": member_path} if member_path is not None else {}
    result = trace.artifact_image(item, artifact_idx, page=page, **options)
    metadata = {key: value for key, value in result.items() if key != "data_url"}
    if result.get("status") != "available" or "data_url" not in result:
        return err(result.get("reason") or "Artifact image unavailable", artifact=metadata)
    return ok(artifact=metadata, _artifact_image=result)


def prepare_artifact_support(
    capability: str, install: bool = False, model: str = "base",
) -> dict[str, Any]:
    """Check/prepare supported artifact readers on demand: office, audio, or speech.

    Office and audio only check for LibreOffice and the user-built PyAV; AutoFA installs
    neither. Speech checks for PyAV and faster-whisper and, with install=True, downloads the
    selected tiny/base/small local model; it never installs packages.
    Does not accept arbitrary packages, URLs or shell commands.
    Files are processed locally; no artifact content is sent to a transcription API.
    """
    from autofa.data.artifact_dependencies import prepare_artifact_support as prepare

    return prepare(capability, install=install, model=model)


def transcribe_artifact(
    item_id: str, artifact_idx: int, start_seconds: float = 0,
    end_seconds: float | None = None, model: str = "base", member_path: str | None = None,
) -> dict[str, Any]:
    """Transcribe at most 120 seconds locally, retaining timestamps and source hashes.

    First prepare_artifact_support('speech', install=True, model=...). A transcript is
    an automatic speech estimate, not proof of audio quality, music or absence of speech.
    ZIP members can be selected by member_path. Original bytes remain downloadable.
    """
    from autofa import trace

    item = _load_item_raw(item_id)
    if item is None:
        return err(f"item not found: {item_id}")
    result = trace.artifact_transcript(
        item, artifact_idx, start_seconds=start_seconds, end_seconds=end_seconds,
        model=model, member_path=member_path,
    )
    return {"ok": result.get("status") == "available", "item_id": item_id, **result}
