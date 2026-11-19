# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded audio inspection and optional local speech recognition of retained bytes.

Waveforms are measurements, not listening. Speech recognition is an attributed
machine transcript, never proof of speaker identity, music, or sound quality.
Dependency/model installation is explicit; inspection and replay stay offline.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from autofa.data.artifact_dependencies import AUDIO_VIDEO_SETUP
from autofa.util.cancellation import cancellation_guard, check_cancelled

AUDIO_TIMEOUT = 30
TRANSCRIPTION_TIMEOUT = 300
MODEL_DOWNLOAD_TIMEOUT = 600
AUDIO_PAGE_SECONDS = 30
MAX_WINDOW_SECONDS = 120
MAX_AUDIO_SECONDS = 3600
SAMPLE_RATE = 16000
MAX_TRANSCRIPT_CHARS = 16000
MAX_WORKER_OUTPUT = 2 * 1024**2
SPEECH_MODELS = ("tiny", "base", "small")
_MODEL_INSTALL_LOCK = threading.RLock()
AUDIO_LIMITATION = (
    "A mono-downmixed waveform and spectrogram of the selected time window only; "
    "these measurements are not listening or speech recognition. Other time windows, "
    "separate channels, and other audio streams remain uninspected."
)
SPEECH_LIMITATION = (
    "Machine-generated speech transcript of one mono-downmixed time window; words and "
    "timestamps may be wrong. Music, other sounds, speaker identity, separate channels, "
    "and perceptual audio quality have not been reviewed. This is not native listening."
)
_FORMATS = {
    ".wav": "wav", ".wave": "wav", ".mp3": "mp3", ".flac": "flac",
    ".ogg": "ogg", ".oga": "ogg", ".opus": "ogg", ".m4a": "mov", ".aac": "aac",
    ".aif": "aiff", ".aiff": "aiff", "audio/wav": "wav", "audio/x-wav": "wav",
    "audio/wave": "wav", "audio/mpeg": "mp3", "audio/flac": "flac",
    "audio/x-flac": "flac", "audio/ogg": "ogg", "audio/opus": "ogg",
    "audio/mp4": "mov", "audio/aac": "aac", "audio/aiff": "aiff",
    "audio/x-aiff": "aiff",
}


def audio_format(metadata: dict) -> str | None:
    name = str(metadata.get("name") or "")
    mime = str(metadata.get("mime_type") or metadata.get("mime") or "").split(";", 1)[0]
    return _FORMATS.get(Path(name).suffix.lower()) or _FORMATS.get(mime)


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _window(start: float, end: float | None) -> tuple[float, float]:
    if type(start) not in {int, float} or not math.isfinite(start) or start < 0:
        raise ValueError("start_seconds must be a finite nonnegative number")
    end = start + MAX_WINDOW_SECONDS if end is None else end
    if type(end) not in {int, float} or not math.isfinite(end):
        raise ValueError("end_seconds must be a finite number")
    if not start < end <= MAX_AUDIO_SECONDS or end - start > MAX_WINDOW_SECONDS:
        raise ValueError(
            f"Audio windows must be positive, at most {MAX_WINDOW_SECONDS} seconds long, "
            f"and within the first {MAX_AUDIO_SECONDS} seconds"
        )
    return float(start), float(end)


def _decode(path: Path, demuxer: str, start: float | None, end: float | None) -> tuple:
    import av  # pyright: ignore[reportMissingImports]
    import numpy as np

    from autofa.data.artifact_video import open_media, require_decoder

    if demuxer not in set(_FORMATS.values()) | {"matroska"}:
        raise ValueError("Unsupported audio container")
    with open_media(av, path, demuxer) as container:
        streams = list(container.streams.audio)
        if not streams:
            raise ValueError("Artifact contains no audio stream")
        stream = streams[0]
        require_decoder(stream, "audio")
        duration = (float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None else
                    float(container.duration / av.time_base) if container.duration else None)
        if duration is not None and (not math.isfinite(duration) or duration < 0):
            duration = None
        result = {
            "source_sha256": _hash(path), "stream_index": stream.index,
            "audio_streams": len(streams), "duration_seconds": duration,
            "sample_rate": stream.codec_context.sample_rate,
            "channels": stream.codec_context.channels, "codec": stream.codec_context.name,
            "window_seconds": AUDIO_PAGE_SECONDS,
            "declared_window_count": math.ceil(duration / AUDIO_PAGE_SECONDS) if duration else None,
        }
        if start is None:
            return result, None
        assert end is not None
        first, last = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
        if last <= first:
            raise ValueError("Audio window is shorter than one decoded sample")
        stream.codec_context.thread_count = 1
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        parts, cursor, finished = [], 0, False
        origin = (float(stream.start_time * stream.time_base)
                  if stream.start_time is not None else None)

        def retain(frame) -> None:
            nonlocal cursor, finished, origin
            if frame.pts is not None and frame.time_base is not None:
                timestamp = float(frame.pts * frame.time_base)
                origin = timestamp if origin is None else origin
                position = round((timestamp - origin) * SAMPLE_RATE)
                if abs(position - cursor) > 2:
                    raise ValueError("Audio timestamps have gaps; exact window is unavailable")
            samples = frame.to_ndarray().reshape(-1).astype("<f4", copy=False)
            if not np.isfinite(samples).all():
                raise ValueError("Decoded audio contains nonfinite samples")
            lo, hi = max(0, first - cursor), min(samples.size, last - cursor)
            if hi > lo:
                parts.append(samples[lo:hi].copy())
            cursor += samples.size
            finished = cursor >= last

        for frame_number, frame in enumerate(container.decode(stream), 1):
            if frame_number > 500000 or frame.samples > 1000000:
                raise ValueError("Audio decoding exceeded its sample/frame bound")
            for converted in resampler.resample(frame):
                retain(converted)
            if finished:
                break
        if not finished:
            for converted in resampler.resample(None):
                retain(converted)
        if not parts:
            raise ValueError("Audio ended before the requested time window")
        samples = np.concatenate(parts)
        actual_end = start + samples.size / SAMPLE_RATE
        result.update(
            start_seconds=start, end_seconds=actual_end, decoded_sample_rate=SAMPLE_RATE,
            decoded_samples=int(samples.size),
            decoded_pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
            peak_amplitude=float(np.abs(samples).max()),
            rms_amplitude=float(np.sqrt(np.mean(samples.astype("float64") ** 2))),
            locator=f"audio:stream:{stream.index}/seconds:{start:g}-{actual_end:g}",
            audio_complete=False,
        )
        return result, samples


def _waveform(samples, metadata: dict, output: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    width, height, left, right = 1200, 640, 70, 1180
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 12), "Mono audio measurements (not listening)", fill="black")
    draw.text((left, 32),
              f"{metadata['start_seconds']:.3f} - {metadata['end_seconds']:.3f} seconds; "
              f"peak {metadata['peak_amplitude']:.4f}; RMS {metadata['rms_amplitude']:.4f}",
              fill="black")
    draw.line((left, 150, right, 150), fill="#aaaaaa")
    for x in range(right - left):
        lo = x * samples.size // (right - left)
        hi = max(lo + 1, (x + 1) * samples.size // (right - left))
        values = samples[lo:hi]
        if values.size:
            draw.line((left + x, 150 - int(min(1., values.max()) * 90),
                       left + x, 150 - int(max(-1., values.min()) * 90)), fill="#2266aa")
    draw.text((left, 260), "Spectrogram, 0-8 kHz; relative energy (dB)", fill="black")
    fft_size = 512
    padded = np.pad(samples, (0, max(0, fft_size - samples.size)))
    positions = np.linspace(0, max(0, padded.size - fft_size), right - left, dtype=int)
    windows = padded[positions[:, None] + np.arange(fft_size)] * np.hanning(fft_size)
    power = np.abs(np.fft.rfft(windows, axis=1))
    colors = np.clip((20 * np.log10(np.maximum(power, 1e-8)) + 80) / 100, 0, 1)
    gray = (colors.T[::-1] * 255).astype("uint8")
    with Image.fromarray(gray) as spectrum:
        with spectrum.convert("RGB").resize((right - left, 280)) as resized:
            image.paste(resized, (left, 290))
    draw.text((left, 590), "One window; mono downmix can hide separate-channel differences.",
              fill="black")
    try:
        image.save(output, "PNG")
    finally:
        image.close()


def _model_path(model: str) -> Path:
    from autofa.config import autofa_home

    if model not in SPEECH_MODELS:
        raise ValueError(f"Speech model must be one of {', '.join(SPEECH_MODELS)}")
    return autofa_home() / "artifact-models" / model


def speech_model_identity(model: str = "base", *, model_dir: str | Path | None = None) -> dict:
    """Hash exact local model assets. Never download during evidence revalidation."""
    default = _model_path(model)
    path = Path(model_dir) if model_dir is not None else default
    required = ("model.bin", "config.json", "tokenizer.json")
    if not path.is_dir() or not all((path / name).is_file() for name in required):
        raise ImportError(
            f"Speech model {model!r} is not installed. "
            f"Use prepare_artifact_support(capability='speech', install=True, model='{model}')."
        )
    files = sorted(child for child in path.iterdir() if child.is_file())
    if len(files) > 64:
        raise ValueError("Speech model directory contains too many files")
    manifest = {child.name: _hash(child) for child in files}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    try:
        version = importlib.metadata.version("faster-whisper")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    return {"status": "available", "name": model, "path": str(path), "sha256": fingerprint,
            "engine": "faster-whisper", "engine_version": version,
            "device": "cpu", "compute_type": "int8"}


def _transcribe(samples, audio: dict, model: dict) -> dict:
    from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]

    recognizer = WhisperModel(model["path"], device="cpu", compute_type="int8",
                              cpu_threads=2, num_workers=1, local_files_only=True)
    segments, info = recognizer.transcribe(
        samples, beam_size=5, temperature=0., condition_on_previous_text=False,
        vad_filter=False,
    )
    retained, chars = [], 0
    for number, segment in enumerate(segments, 1):
        text = str(segment.text).strip()
        chars += len(text)
        if chars > MAX_TRANSCRIPT_CHARS or number > 1000:
            raise ValueError("Transcript exceeds output limit; request a shorter audio window")
        start, end = float(segment.start), float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("Speech recognizer returned invalid timestamps")
        if end > audio["end_seconds"] - audio["start_seconds"] + .1:
            raise ValueError("Speech recognizer returned a timestamp beyond the reviewed window")
        start += audio["start_seconds"]
        end = min(end + audio["start_seconds"], audio["end_seconds"])
        retained.append({
            "locator": f"audio:stream:{audio['stream_index']}/speech:{number}",
            "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "start_seconds": start, "end_seconds": end,
        })
    return {
        "status": "available", "complete": False, "audio_complete": False,
        "transform_version": "artifact-speech-v1", "audio": audio, "model": model,
        "source_sha256": audio["source_sha256"], "segments": retained,
        "language": info.language, "limitations": [SPEECH_LIMITATION],
        "scope": "Machine speech recognition of the selected time window only.",
    }


def _stop_worker(worker: subprocess.Popen) -> None:
    if worker.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(worker.pid, signal.SIGTERM)
        else:
            worker.terminate()
    except ProcessLookupError:
        return
    try:
        worker.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(worker.pid, signal.SIGKILL)
            else:
                worker.kill()
        except ProcessLookupError:
            pass
        worker.wait(timeout=2)


def _run(request: dict, timeout: int) -> dict:
    check_cancelled()
    with (tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout,
          tempfile.TemporaryFile() as stderr):
        stdin.write(json.dumps(request).encode())
        stdin.seek(0)
        worker = subprocess.Popen(
            [sys.executable, "-m", "autofa.data.artifact_audio"],
            stdin=stdin, stdout=stdout, stderr=stderr, start_new_session=os.name == "posix",
        )
        deadline = time.monotonic() + timeout
        try:
            while worker.poll() is None:
                check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError(f"Audio operation exceeded its {timeout}-second limit")
                if any(os.fstat(stream.fileno()).st_size > MAX_WORKER_OUTPUT
                       for stream in (stdout, stderr)):
                    raise ValueError("Audio worker exceeded its output limit")
                try:
                    worker.wait(timeout=min(.25, remaining))
                except subprocess.TimeoutExpired:
                    pass
            check_cancelled()
            stdout.seek(0)
            raw = stdout.read(MAX_WORKER_OUTPUT + 1)
            if len(raw) > MAX_WORKER_OUTPUT:
                raise ValueError("Audio worker exceeded its output limit")
            try:
                result = json.loads(raw)
            except ValueError as exc:
                raise ValueError("Audio worker did not return a valid result") from exc
            if worker.returncode or result.get("error"):
                from autofa.data.artifact_video import worker_failure

                raise worker_failure(result, "Audio worker failed")
            return result
        finally:
            _stop_worker(worker)


def audio_operation(path: Path, metadata: dict, *, page: int | None = None) -> tuple[dict, object]:
    """Metadata or one 30-second waveform/spectrogram, using isolated native decoding."""
    demuxer = audio_format(metadata)
    if demuxer is None:
        raise ValueError("Unsupported audio container")
    start = end = None
    if page is not None:
        if type(page) is not int or page < 1:
            raise ValueError("Audio page must be a positive integer")
        start, end = _window((page - 1) * AUDIO_PAGE_SECONDS, page * AUDIO_PAGE_SECONDS)
    if importlib.util.find_spec("av") is None:
        raise ImportError(
            "Audio inspection needs PyAV built against an FFmpeg on this host; AutoFA does not "
            f"install it. See {AUDIO_VIDEO_SETUP}"
        )
    with tempfile.TemporaryDirectory(prefix="autofa-audio-") as directory:
        output = Path(directory) / "waveform.png"
        result = _run({"action": "inspect", "path": str(path), "demuxer": demuxer,
                       "start": start, "end": end, "output": str(output)}, AUDIO_TIMEOUT)
        if page is None:
            return result, None
        from PIL import Image

        with Image.open(output) as image:
            return result, image.copy()


def transcribe_audio(path: Path, metadata: dict, *, start_seconds: float = 0,
                     end_seconds: float | None = None, model: str = "base",
                     model_dir: str | Path | None = None) -> dict:
    """Transcribe at most 120 seconds locally, without installing or downloading anything."""
    from autofa.data.artifact_video import video_format

    if importlib.util.find_spec("av") is None or importlib.util.find_spec("faster_whisper") is None:
        raise ImportError(
            "Speech recognition needs PyAV and faster-whisper, which AutoFA does not install "
            f"(see {AUDIO_VIDEO_SETUP}). Then use "
            "prepare_artifact_support(capability='speech', install=True) to download a model."
        )
    start, end = _window(start_seconds, end_seconds)
    identity = speech_model_identity(model, model_dir=model_dir)
    demuxer = audio_format(metadata) or video_format(metadata)
    if demuxer is None:
        raise ValueError("Unsupported audio/video container")
    return _run({"action": "transcribe", "path": str(path), "demuxer": demuxer,
                 "start": start, "end": end, "model": identity}, TRANSCRIPTION_TIMEOUT)


def prepare_speech_model(model: str = "base") -> dict:
    """Explicitly download one allowlisted local speech model into the AutoFA cache."""
    check_cancelled()
    destination = _model_path(model)
    if importlib.util.find_spec("av") is None or importlib.util.find_spec("faster_whisper") is None:
        raise ImportError(
            "Install PyAV and faster-whisper before downloading a local speech model; "
            f"see {AUDIO_VIDEO_SETUP}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _model_lock(destination.parent / f".{model}.lock"):
        try:
            return speech_model_identity(model)
        except ImportError:
            pass
        with tempfile.TemporaryDirectory(prefix=f".{model}-", dir=destination.parent) as temp:
            staging = Path(temp) / "model"
            _run({"action": "download", "model_name": model, "destination": str(staging)},
                 MODEL_DOWNLOAD_TIMEOUT)
            # An interrupted download never becomes an apparently installed model.
            speech_model_identity(model, model_dir=staging)
            with cancellation_guard():
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(staging, destination)
        return speech_model_identity(model)


@contextmanager
def _model_lock(path: Path):
    """Serialize model publication without hiding cancellation behind another download."""
    while not _MODEL_INSTALL_LOCK.acquire(timeout=.25):
        check_cancelled()
    try:
        check_cancelled()
        if path.is_symlink():
            raise ValueError("Speech model lock must not be a symbolic link")
        with path.open("a+b") as handle:
            try:
                import fcntl
            except ImportError:
                fcntl = None
            if fcntl is not None:
                while True:
                    check_cancelled()
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(.25)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        _MODEL_INSTALL_LOCK.release()


def _worker(request: dict) -> dict:
    if request["action"] == "download":
        from faster_whisper.utils import download_model  # pyright: ignore[reportMissingImports]

        if request["model_name"] not in SPEECH_MODELS:
            raise ValueError("Unsupported speech model")
        download_model(request["model_name"], output_dir=request["destination"])
        return {"status": "available"}
    audio, samples = _decode(Path(request["path"]), request["demuxer"],
                             request.get("start"), request.get("end"))
    if request["action"] == "transcribe":
        return _transcribe(samples, audio, request["model"])
    if samples is not None:
        _waveform(samples, audio, Path(request["output"]))
    return audio


if __name__ == "__main__":
    from autofa.data.artifact_video import run_worker_main

    run_worker_main(_worker)
