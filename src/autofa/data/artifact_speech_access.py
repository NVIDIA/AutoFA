# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned local speech estimates shared by detectors, discovery and causal review."""

from __future__ import annotations

import json

from autofa.data.artifact_access import _fingerprint, _output_directory, _record, _verified_snapshot
from autofa.data.artifact_content import ARTIFACT_TRANSFORM_VERSION
from autofa.data.artifact_store import _cache_lock, _pin_receipt, ensure_artifact
from autofa.util.cancellation import check_cancelled

SPEECH_TRANSFORM_VERSION = "artifact-speech-v1"


def artifact_transcript(
    item: dict, artifact_idx: int, *, start_seconds: float = 0,
    end_seconds: float | None = None, model: str = "base", member_path: str | None = None,
    store=None,
) -> dict:
    """Read a bounded, cached speech estimate; never download models during evidence reads."""
    from autofa.data.artifact_audio import _window, speech_model_identity, transcribe_audio

    start_seconds, end_seconds = _window(start_seconds, end_seconds)
    request = {"start_seconds": start_seconds, "end_seconds": end_seconds, "model": model}
    selection = {"member_path": member_path} if member_path is not None else {}
    saved = ensure_artifact(item, artifact_idx, store=store, **selection)
    result = {**saved, "artifact_idx": artifact_idx, "segments": [], "complete": False,
              "audio_complete": False, "limitations": [],
              "transform_version": ARTIFACT_TRANSFORM_VERSION,
              "speech_transform_version": SPEECH_TRANSFORM_VERSION}
    # Only the verified original path is public. Private model/cache locations do
    # not become evidence identities and cannot vary between review consumers.
    if saved.get("status") != "available":
        return _record(result, artifact_idx, transcription=request, **selection)
    try:
        check_cancelled()
        model_identity = speech_model_identity(model)
        public_model = {key: value for key, value in model_identity.items() if key != "path"}
        identity = {"source_sha256": saved["sha256"], "model": public_model,
                    "request": request, "version": SPEECH_TRANSFORM_VERSION}
        key = _fingerprint(identity)
        output = _output_directory(saved, SPEECH_TRANSFORM_VERSION)
        manifest = output / f"transcript-{key}.json"
        with _cache_lock(output / f"transcript-{key}.lock"):
            check_cancelled()
            if manifest.is_symlink():
                raise ValueError("Speech cache receipt must not be a symbolic link")
            if manifest.exists():
                pinned = json.loads(manifest.read_text())
                if not isinstance(pinned, dict):
                    raise ValueError("Pinned speech transcript is invalid or changed")
                parsed = pinned.get("result")
                if (pinned.get("identity") != identity or not isinstance(parsed, dict)
                        or pinned.get("result_sha256") != _fingerprint(parsed)):
                    raise ValueError("Pinned speech transcript is invalid or changed")
            else:
                with _verified_snapshot(saved) as snapshot:
                    parsed = transcribe_audio(
                        snapshot, saved["metadata"], **request, model_dir=model_identity["path"],
                    )
                if (not isinstance(parsed, dict) or parsed.get("status") != "available"
                        or not isinstance(parsed.get("model"), dict)
                        or not isinstance(parsed.get("segments"), list)):
                    raise ValueError("Speech recognition did not produce a valid transcript")
                if parsed.get("source_sha256") != saved["sha256"]:
                    raise ValueError("Speech transcript differs from pinned source bytes")
                actual_model = {key: value for key, value in parsed.get("model", {}).items()
                                if key != "path"}
                if actual_model != public_model:
                    raise ValueError("Speech model changed while transcribing the artifact")
                parsed = {**parsed, "model": public_model}
                for index, segment in enumerate(parsed.get("segments", [])):
                    if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                        raise ValueError("Speech recognition returned an invalid text segment")
                    segment["locator"] = f"speech:{key}/segment:{index}"
                check_cancelled()
                # Freeze the first successful estimate for these exact source,
                # model and window bytes; reading it never repeats inference.
                _pin_receipt(manifest, {"identity": identity, "result": parsed,
                                        "result_sha256": _fingerprint(parsed)})
        if parsed.get("status") != "available":
            raise ValueError("Speech recognition did not produce an available transcript")
        result.update(parsed)
        result.update(sha256=saved["sha256"], source_identity=saved["source_identity"],
                      transform_version=ARTIFACT_TRANSFORM_VERSION,
                      speech_transform_version=SPEECH_TRANSFORM_VERSION)
    except (ImportError, OSError, ValueError) as exc:
        result.update(status="unavailable", segments=[], limitations=[str(exc)], reason=str(exc))
    return _record(result, artifact_idx, transcription=request, **selection)
