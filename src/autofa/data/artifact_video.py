# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded local video metadata and exact frame extraction using an optional, user-built PyAV.

Run native decoding in a short-lived worker so a corrupt file cannot leave an
unbounded parser inside discovery. Only already-verified local bytes are passed
here; demuxing cannot open external network resources or QuickTime references.
AutoFA installs no FFmpeg: the demuxers and decoders are those of the host's build.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from autofa.data.artifact_dependencies import AUDIO_VIDEO_SETUP

VIDEO_TIMEOUT = 30
MAX_DECODE_FRAMES = 18_000
MAX_DECODE_PIXELS = 16_000_000
VIDEO_LIMITATION = (
    "One selected video frame only: motion, intervening frames, other video streams and "
    "audio have not been reviewed. Frame viewing never establishes whole-video coverage."
)
_VIDEO_FORMATS = {
    ".mp4": "mov", ".m4v": "mov", ".mov": "mov", ".webm": "matroska", ".mkv": "matroska",
    "video/mp4": "mov", "video/quicktime": "mov", "video/x-m4v": "mov",
    "video/webm": "matroska", "video/x-matroska": "matroska",
}


class CodecUnavailableError(ImportError):
    """The host's FFmpeg lacks the artifact's demuxer or decoder: a missing reader."""


def _rebuild_hint() -> str:
    return (
        "AutoFA's documented FFmpeg build reads only AV1, VP8, VP9 and Theora video and Opus, "
        f"Vorbis, FLAC and linear PCM audio; to add codecs, see {AUDIO_VIDEO_SETUP}"
    )


def open_media(av, path: Path, demuxer: str):
    """Open verified local bytes with local-file-only demuxer options."""
    try:
        # No remote URLs, playlists, or QuickTime external data references.
        return av.open(str(path), mode="r", format=demuxer, options={
            "protocol_whitelist": "file", "enable_drefs": "0", "use_absolute_path": "0",
        })
    except ValueError as exc:
        if not str(exc).startswith("no container format"):
            raise
        raise CodecUnavailableError(
            f"This host's FFmpeg has no {demuxer!r} demuxer. {_rebuild_hint()}"
        ) from exc


def require_decoder(stream, kind: str) -> None:
    """PyAV leaves codec_context unset when FFmpeg has no decoder for the stream."""
    if stream.codec_context is None:
        raise CodecUnavailableError(
            f"This host's FFmpeg cannot decode the artifact's {kind} codec. {_rebuild_hint()}"
        )


def run_worker_main(worker) -> None:
    """Print a worker's JSON result; a missing reader is reported apart from other failures."""
    try:
        response = worker(json.load(sys.stdin))
    except ImportError as exc:
        # A PyAV whose FFmpeg libraries cannot load (a moved prefix, a missing run path)
        # is a missing reader too, not a fault in the artifact.
        message = str(exc)[:600] if isinstance(exc, CodecUnavailableError) else (
            f"PyAV could not be loaded ({type(exc).__name__}: {str(exc)[:300]}). "
            f"See {AUDIO_VIDEO_SETUP}"
        )
        print(json.dumps({"error": message, "unavailable": True}))
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:400]}"}))
        sys.exit(1)
    print(json.dumps(response))


def worker_failure(result: dict, fallback: str) -> Exception:
    """Rebuild a worker's failure: an unavailable reader keeps its ImportError status."""
    if result.get("unavailable"):
        return CodecUnavailableError(result.get("error") or fallback)
    return ValueError(result.get("error") or fallback)


def video_format(metadata: dict) -> str | None:
    """Support MP4/QuickTime and WebM/Matroska without enabling playlist demuxers."""
    name = str(metadata.get("name") or "")
    mime = str(metadata.get("mime_type") or metadata.get("mime") or "").split(";", 1)[0]
    return _VIDEO_FORMATS.get(Path(name).suffix.lower()) or _VIDEO_FORMATS.get(mime)


def _worker(path: Path, demuxer: str, page: int | None, output: Path) -> dict:
    import av  # pyright: ignore[reportMissingImports]

    if demuxer not in {"mov", "matroska"}:
        raise ValueError("Unsupported video container")
    with open_media(av, path, demuxer) as container:
        videos = list(container.streams.video)
        if not videos:
            raise ValueError("Artifact contains no video stream")
        stream = videos[0]
        require_decoder(stream, "video")
        width, height = stream.codec_context.width, stream.codec_context.height
        if width <= 0 or height <= 0 or width * height > MAX_DECODE_PIXELS:
            raise ValueError("Video frame dimensions exceed the supported decode limit")
        duration = (float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None else
                    float(container.duration / av.time_base) if container.duration else None)
        rate = float(stream.average_rate) if stream.average_rate else None
        result = {
            "stream_index": stream.index, "video_streams": len(videos),
            "audio_streams": len(container.streams.audio), "duration_seconds": duration,
            "frame_rate": rate, "declared_frame_count": stream.frames or None,
            "width": width, "height": height,
        }
        if page is None:
            return result
        if type(page) is not int or not 1 <= page <= MAX_DECODE_FRAMES:
            raise ValueError(f"Video frame must be between 1 and {MAX_DECODE_FRAMES}")
        stream.codec_context.thread_count = 1
        for number, frame in enumerate(container.decode(stream), 1):
            if frame.width * frame.height > MAX_DECODE_PIXELS:
                raise ValueError("Video frame dimensions exceed the supported decode limit")
            if number != page:
                continue
            image = frame.to_image()
            try:
                image.save(output, format="PNG")
            finally:
                image.close()
            timestamp = (float(frame.pts * frame.time_base)
                         if frame.pts is not None and frame.time_base is not None else None)
            if timestamp is not None and not math.isfinite(timestamp):
                timestamp = None
            return {**result, "frame": number, "timestamp_seconds": timestamp,
                    "locator": f"video:stream:{stream.index}/frame:{number}"}
        raise ValueError(f"Video ended before requested frame {page}")


def video_operation(path: Path, metadata: dict, *, page: int | None = None) -> tuple[dict, object]:
    """Return stream metadata and optionally a detached PIL image for one exact frame."""
    if importlib.util.find_spec("av") is None:
        raise ImportError(
            "Video frame inspection needs PyAV built against an FFmpeg on this host; AutoFA "
            f"does not install it. See {AUDIO_VIDEO_SETUP}"
        )
    demuxer = video_format(metadata)
    if not demuxer:
        raise ValueError("Unsupported video container")
    if page is not None and (type(page) is not int or not 1 <= page <= MAX_DECODE_FRAMES):
        raise ValueError(f"Video frame must be between 1 and {MAX_DECODE_FRAMES}")
    with tempfile.TemporaryDirectory(prefix="autofa-video-") as directory:
        output = Path(directory) / "frame.png"
        request = {"path": str(path), "demuxer": demuxer, "page": page, "output": str(output)}
        try:
            worker = subprocess.run(
                [sys.executable, "-m", "autofa.data.artifact_video"],
                input=json.dumps(request), capture_output=True, text=True,
                timeout=VIDEO_TIMEOUT, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Video inspection exceeded its {VIDEO_TIMEOUT}-second limit") from exc
        try:
            result = json.loads(worker.stdout)
        except ValueError as exc:
            raise ValueError("Video worker did not return a valid inspection result") from exc
        if worker.returncode or result.get("error"):
            raise worker_failure(result, "Video worker failed")
        if page is None:
            return result, None
        from PIL import Image

        with Image.open(output) as image:
            return result, image.copy()


if __name__ == "__main__":
    run_worker_main(lambda request: _worker(
        Path(request["path"]), request["demuxer"], request["page"], Path(request["output"]),
    ))
