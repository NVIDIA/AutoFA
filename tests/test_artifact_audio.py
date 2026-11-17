# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real audio decoding and offline speech estimates that stay cached and source-bound."""

import asyncio
import hashlib
import json
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import get_type_hints

import av
import numpy as np
import pytest

from autofa import trace
from autofa.core.sandbox import run_detector
from autofa.data import artifact_audio as audio
from autofa.data import artifact_speech_access as speech_access
from autofa.data.artifact_access import capture_artifact_reads, validate_artifact_reads
from autofa.data.artifact_speech_access import artifact_transcript
from autofa.data.artifact_store import _cache_lock as cache_lock
from autofa.mcp import tools_artifacts
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot
from autofa.util.cancellation import Cancellation, cancellation_scope

from support.traces import item, serve, zip_bytes


def wav(path: Path, *, seconds=2, rate=16000, channels=1, silent=False) -> Path:
    frames = np.arange(round(seconds * rate))
    samples = (np.sin(2 * np.pi * 440 * frames / rate) * 16000).astype("<i2")
    if silent:
        samples[:] = 0
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.repeat(samples, channels).tobytes())
    return path


@pytest.fixture
def tone(tmp_path):
    return wav(tmp_path / "tone.wav")


def test_metadata_has_provenance_without_decoding_or_claiming_listening(tone):
    result, image = audio.audio_operation(tone, {"name": "tone.wav"})
    assert image is None
    assert result["duration_seconds"] == 2
    assert result["channels"] == 1
    assert result["sample_rate"] == 16000
    assert result["source_sha256"] == hashlib.sha256(tone.read_bytes()).hexdigest()
    assert result["declared_window_count"] == 1
    assert "decoded_pcm_sha256" not in result


def test_waveform_is_exact_window_and_hash_bound_measurement(tone):
    result, image = audio.audio_operation(tone, {"name": "tone.wav"}, page=1)
    try:
        assert image.size == (1200, 640)
        assert result["locator"] == "audio:stream:0/seconds:0-2"
        assert result["decoded_samples"] == 32000
        assert not result["audio_complete"]
        assert .48 < result["peak_amplitude"] < .49
        assert .34 < result["rms_amplitude"] < .35
        expected = (np.sin(2 * np.pi * 440 * np.arange(32000) / 16000) * 16000)
        expected = expected.astype("<i2").astype("<f4") / 32768
        assert result["decoded_pcm_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    finally:
        image.close()


def test_later_window_and_stereo_resampling_do_not_duplicate_first_window(tmp_path):
    source = wav(tmp_path / "stereo.wav", seconds=31.5, rate=44100, channels=2)
    result, image = audio.audio_operation(source, {"name": "stereo.wav"}, page=2)
    try:
        assert result["channels"] == 2
        assert result["start_seconds"] == 30
        assert result["end_seconds"] == pytest.approx(31.5, abs=1 / 16000)
        assert result["decoded_sample_rate"] == 16000
        assert result["declared_window_count"] == 2
    finally:
        image.close()


def test_silence_measurements_do_not_produce_invalid_spectrogram(tmp_path):
    source = wav(tmp_path / "silence.wav", silent=True)
    result, image = audio.audio_operation(source, {"name": "silence.wav"}, page=1)
    try:
        assert result["peak_amplitude"] == result["rms_amplitude"] == 0
    finally:
        image.close()


@pytest.mark.parametrize("page", [0, 121])
def test_bad_audio_windows_rejected_before_worker(tone, monkeypatch, page):
    monkeypatch.setattr(audio, "_run", lambda *_: pytest.fail("must reject before decoding"))
    with pytest.raises(ValueError):
        audio.audio_operation(tone, {"name": "tone.wav"}, page=page)


def test_window_after_eof_fails(tone):
    with pytest.raises(ValueError, match="ended before"):
        audio.audio_operation(tone, {"name": "tone.wav"}, page=2)


def test_missing_audio_dependency_is_actionable(tone, monkeypatch):
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(ImportError, match="PyAV"):
        audio.audio_operation(tone, {"name": "tone.wav"})
    with pytest.raises(ImportError, match="prepare_artifact_support"):
        audio.transcribe_audio(tone, {"name": "tone.wav"})


def test_worker_timeout_is_bounded(tone, monkeypatch):
    monkeypatch.setattr(audio, "AUDIO_TIMEOUT", 0)
    with pytest.raises(ValueError, match="0-second limit"):
        audio.audio_operation(tone, {"name": "tone.wav"})


def test_cancelling_audio_operation_terminates_worker_promptly(tone, monkeypatch):
    original = subprocess.Popen
    workers = []

    def sleeping_worker(_command, **kwargs):
        worker = original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(audio.subprocess, "Popen", sleeping_worker)
    cancelled = Cancellation()
    cancelled.cancel()
    with cancellation_scope(cancelled), pytest.raises(asyncio.CancelledError):
        audio.audio_operation(tone, {"name": "tone.wav"})
    assert workers == []  # An already-cancelled operation never starts a worker.
    token = Cancellation()
    timer = threading.Timer(.15, token.cancel)
    started = time.monotonic()
    timer.start()
    try:
        with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
            audio.audio_operation(tone, {"name": "tone.wav"})
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3
    assert len(workers) == 1 and workers[0].poll() is not None


def test_audio_decoder_never_opens_external_protocols(tmp_path, monkeypatch):
    seen = []

    def reject(path, **kwargs):
        seen.append(kwargs)
        raise ValueError("fixture finished")

    monkeypatch.setattr(av, "open", reject)
    with pytest.raises(ValueError, match="fixture finished"):
        audio._decode(tmp_path / "artifact", "wav", None, None)
    assert seen[0]["options"] == {
        "protocol_whitelist": "file", "enable_drefs": "0", "use_absolute_path": "0",
    }
    assert audio.audio_format({"name": "playlist.m3u8"}) is None


def test_invalid_models_do_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(audio, "_run", lambda *_: pytest.fail("must not download"))
    for model in ("../../weights", "https://example.invalid/model", "large-v3"):
        with pytest.raises(ValueError, match="Speech model must"):
            audio.prepare_speech_model(model)


def test_explicit_model_download_commits_only_complete_offline_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda _: True)
    root = tmp_path / "artifact-models" / "base"
    calls = []

    def download(request, timeout):
        assert timeout == audio.MODEL_DOWNLOAD_TIMEOUT
        calls.append(request)
        staging = Path(request["destination"])
        staging.mkdir()
        if len(calls) == 1:
            (staging / "model.bin").write_bytes(b"partial weights")
            raise ValueError("interrupted")
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (staging / name).write_text(name)
        return {"status": "available"}

    monkeypatch.setattr(audio, "_run", download)
    with pytest.raises(ValueError, match="interrupted"):
        audio.prepare_speech_model()
    assert not root.exists()  # Partial assets are never published.
    assert audio.prepare_speech_model()["status"] == "available"
    assert audio.prepare_speech_model()["status"] == "available"
    assert len(calls) == 2
    identity = audio.speech_model_identity()
    assert identity["status"] == "available" and identity["path"] == str(root)
    # Identity is computed offline from the installed files and tracks changed weights.
    (root / "model.bin").write_text("new weights")
    assert audio.speech_model_identity()["sha256"] != identity["sha256"]
    (root / "tokenizer.json").unlink()
    with pytest.raises(ImportError, match="prepare_artifact_support"):
        audio.speech_model_identity()


def whisper(monkeypatch, text, start=0., end=1.):
    """Install an offline faster-whisper double and return its recorded calls."""
    calls = []

    class Recognizer:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)

        def transcribe(self, samples, **kwargs):
            calls.append((samples.size, kwargs))
            segment = SimpleNamespace(text=text, start=start, end=end)
            return iter([segment]), SimpleNamespace(language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Recognizer))
    return calls


def test_transcription_is_offline_timestamped_and_explicitly_not_listening(monkeypatch):
    calls = whisper(monkeypatch, " The meeting starts now. ", .2, 1.8)
    metadata = {"start_seconds": 30., "end_seconds": 32., "stream_index": 0,
                "source_sha256": "abc", "decoded_pcm_sha256": "def"}
    identity = {"path": "/installed/model", "sha256": "weights", "name": "base"}
    result = audio._transcribe(np.zeros(32000), metadata, identity)
    assert calls[0]["local_files_only"] is True
    assert calls[1][0] == 32000 and calls[1][1]["temperature"] == 0
    assert result["model"] == identity and result["source_sha256"] == "abc"
    assert result["segments"][0]["text"] == "The meeting starts now."
    assert result["segments"][0]["start_seconds"] == pytest.approx(30.2)
    assert result["segments"][0]["end_seconds"] == pytest.approx(31.8)
    assert not result["complete"] and not result["audio_complete"]


def test_transcript_output_limit_fails_explicitly_instead_of_truncating(monkeypatch):
    whisper(monkeypatch, "a" * 16001)
    with pytest.raises(ValueError, match="shorter audio window"):
        audio._transcribe(np.zeros(32000), {}, {"path": "fixture"})


@pytest.mark.parametrize("start,end", [(0, 121), (float("nan"), 1), (0, float("inf")), (2, 1)])
def test_invalid_transcription_windows(start, end):
    with pytest.raises(ValueError):
        audio._window(start, end)


# Shared speech estimates are cached, source-bound, and honest about review scope.

@pytest.fixture
def speech(tmp_path, monkeypatch):
    body = wav(tmp_path / "speech.wav", seconds=1, silent=True).read_bytes()
    provider = serve(monkeypatch, tmp_path / "home", body)
    source = item(body, name="recording.wav", mime="audio/wav")
    state = SimpleNamespace(provider=provider, item=source, calls=[], available=True,
                            tmp_path=tmp_path, body=body, model_mutation=None,
                            source_mutation=None)

    def identity(model="base", *, model_dir=None):
        if not state.available:
            raise ImportError("Speech model unavailable; prepare_artifact_support first")
        return {
            "status": "available", "name": model,
            "path": str(model_dir or tmp_path / "model-assets" / model),
            "sha256": hashlib.sha256(model.encode()).hexdigest(),
            "engine": "faster-whisper", "engine_version": "offline-fixture-v1",
            "device": "cpu", "compute_type": "int8",
        }

    def recognize(path, metadata, *, start_seconds=0, end_seconds=None,
                  model="base", model_dir=None):
        state.calls.append((start_seconds, end_seconds, model))
        used = identity(model, model_dir=model_dir)
        used.update(state.model_mutation or {})
        start, end = float(start_seconds), min(float(end_seconds or 1), 1)
        text = f"Delivery confirmed by {model} in window {start_seconds}."
        return {
            "status": "available", "complete": False, "audio_complete": False,
            "source_sha256": state.source_mutation or hashlib.sha256(path.read_bytes()).hexdigest(),
            "model": used, "transform_version": "artifact-speech-v1",
            "scope": "Machine speech estimate of the selected time window only.",
            "segments": [{"text": text, "locator": "unscoped-worker-locator",
                          "start_seconds": start, "end_seconds": end,
                          "sha256": hashlib.sha256(text.encode()).hexdigest()}],
            "audio": {"stream_index": 0, "start_seconds": start, "end_seconds": end},
            "limitations": ["Automatic speech estimate; not proof of audio quality or absence."],
        }

    monkeypatch.setattr(audio, "speech_model_identity", identity)
    monkeypatch.setattr(audio, "transcribe_audio", recognize)
    monkeypatch.setattr(tools_artifacts, "_load_item_raw", lambda _: state.item)
    return state


def archived(speech):
    """Serve the recording as one member of a zip artifact."""
    speech.provider.body = zip_bytes([("audio/session.wav", speech.body)])
    speech.item = item(speech.provider.body, name="recordings.zip", mime="application/zip")


def _review(speech, name="review"):
    return EvidenceReview(
        make_snapshot(speech.item, item_id="case", source="def detect(item): return True"),
        speech.tmp_path / name,
    )


def _verdict(review, citation, *, scope="local"):
    review.read("detector_source")
    return {"item_id": "case", "snapshot_id": review.snapshot["snapshot_id"],
            "scope": scope, "citations": [citation]}


def test_all_consumers_share_one_inference_and_replay_dependency_receipts(speech):
    with capture_artifact_reads() as receipts:
        direct = trace.artifact_transcript(speech.item, 2, end_seconds=1)
    assert direct["status"] == "available" and len(speech.calls) == 1
    assert direct["sha256"] == hashlib.sha256(speech.body).hexdigest()
    assert "path" not in direct["model"]
    assert receipts[0]["transcription"] == {
        "start_seconds": 0, "end_seconds": 1, "model": "base",
    }
    discovery = tools_artifacts.transcribe_artifact("case", 2, end_seconds=1)
    assert discovery["ok"] and discovery["segments"] == direct["segments"]
    review = _review(speech)
    judged = review.transcribe_artifact(2, end_seconds=1)
    assert judged["segments"] == direct["segments"]
    localisation = {}
    detector = (
        "from autofa.trace import artifact_transcript\n"
        "def detect(item):\n"
        "    result = artifact_transcript(item, 2, end_seconds=1)\n"
        "    return any('Delivery confirmed' in row['text'] for row in result['segments'])\n"
    )
    assert run_detector(detector, speech.item, localisation=localisation) is True
    assert localisation["artifact_reads"] == receipts
    assert validate_artifact_reads(speech.item, receipts)
    assert len(speech.calls) == 1 and speech.provider.calls == 1


@pytest.mark.parametrize("case", ["available", "missing_model", "missing_artifact", "archive"])
def test_public_transcript_contract_matches_returned_source_and_speech_fields(speech, case):
    selection = {}
    artifact_idx = 2
    if case == "missing_model":
        speech.available = False
    elif case == "missing_artifact":
        artifact_idx = 99
    elif case == "archive":
        archived(speech)
        selection["member_path"] = "audio/session.wav"

    # Only recognition is mocked: exercise the real store, public helper and
    # receipt recording so the schema follows what consumers actually receive.
    with capture_artifact_reads() as receipts:
        result = trace.artifact_transcript(speech.item, artifact_idx, end_seconds=1, **selection)
    contract = get_type_hints(trace.artifact_transcript)["return"]
    fields = get_type_hints(contract)
    assert result.keys() <= fields.keys()
    assert "download_path" not in fields and "download_path" not in result
    assert fields["path"] is str and fields["verified"] is bool
    assert result["verified"] is (case != "missing_artifact")
    assert len(receipts) == 1
    if case == "missing_artifact":
        assert "path" not in result
    else:
        assert isinstance(result["path"], str)
        assert result["cache_key"]
    assert result["status"] == ("unavailable" if case.startswith("missing_") else "available")
    if case.startswith("missing_"):
        assert result["segments"] == [] and result["reason"]
        assert "model" not in result
    else:
        assert "path" not in result["model"]  # Private model paths remain hidden.
    if case == "archive":
        assert result["member_path"] == selection["member_path"]
        assert result["archive_sha256"] == hashlib.sha256(speech.provider.body).hexdigest()


def test_window_and_model_have_separate_estimate_locators_and_cache_entries(speech):
    first = artifact_transcript(speech.item, 2, start_seconds=0, end_seconds=.5)
    later = artifact_transcript(speech.item, 2, start_seconds=.5, end_seconds=1)
    alternate = artifact_transcript(speech.item, 2, start_seconds=0, end_seconds=.5, model="small")
    assert len({row["segments"][0]["locator"] for row in (first, later, alternate)}) == 3
    assert all(row["segments"][0]["locator"].startswith("speech:")
               for row in (first, later, alternate))
    assert later["segments"][0]["start_seconds"] == .5
    assert len(speech.calls) == 3
    assert artifact_transcript(speech.item, 2, start_seconds=.5, end_seconds=1) == later
    assert len(speech.calls) == 3


def test_equivalent_numeric_window_inputs_reuse_one_estimate_within_review(speech):
    review = _review(speech)
    first = review.transcribe_artifact(2, start_seconds=0, end_seconds=1)
    repeated = review.transcribe_artifact(2, start_seconds=0.0, end_seconds=1.0)
    assert first["segments"] == repeated["segments"]
    assert len(speech.calls) == 1
    with capture_artifact_reads() as receipts:
        artifact_transcript(speech.item, 2, start_seconds=0.0, end_seconds=1.0)
    assert validate_artifact_reads(speech.item, receipts)
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.artifact_evidence_identity() == review.artifact_evidence_identity()


@pytest.mark.parametrize("tamper", ["text", "malformed_json", "wrong_shape"])
def test_corrupt_transcript_cache_fails_without_retranscribing(speech, tamper):
    with capture_artifact_reads() as receipts:
        artifact_transcript(speech.item, 2, end_seconds=1)
    path = next((speech.tmp_path / "home").rglob("transcript-*.json"))
    if tamper == "text":
        value = json.loads(path.read_text())
        value["result"]["segments"][0]["text"] = "Corrupted transcript"
        path.write_text(json.dumps(value))
    else:
        path.write_text("{" if tamper == "malformed_json" else "[]")
    result = artifact_transcript(speech.item, 2, end_seconds=1)
    assert result["status"] == "unavailable" and result["segments"] == []
    assert not validate_artifact_reads(speech.item, receipts)
    assert len(speech.calls) == 1


def test_missing_model_cannot_validate_previous_estimate_or_trigger_download(speech):
    with capture_artifact_reads() as receipts:
        artifact_transcript(speech.item, 2, end_seconds=1)
    speech.available = False
    result = artifact_transcript(speech.item, 2, end_seconds=1)
    assert result["status"] == "unavailable" and not result["complete"]
    assert "prepare_artifact_support" in result["reason"]
    assert not validate_artifact_reads(speech.item, receipts)
    assert len(speech.calls) == 1


def test_unsuccessful_worker_result_never_freezes_a_failed_cache(speech, monkeypatch):
    original = audio.transcribe_audio
    failed = {"status": "available", "model": [], "segments": []}
    monkeypatch.setattr(audio, "transcribe_audio", lambda *args, **kwargs: failed)
    result = artifact_transcript(speech.item, 2, end_seconds=1)
    assert result["status"] == "unavailable"
    assert not list((speech.tmp_path / "home").rglob("transcript-*.json"))
    monkeypatch.setattr(audio, "transcribe_audio", original)
    assert artifact_transcript(speech.item, 2, end_seconds=1)["status"] == "available"


@pytest.mark.parametrize("mismatch", ["source", "model"])
def test_worker_source_or_model_mismatch_never_pins_an_estimate(speech, mismatch):
    if mismatch == "source":
        speech.source_mutation = "f" * 64
    else:
        speech.model_mutation = {"sha256": "e" * 64}
    result = artifact_transcript(speech.item, 2, end_seconds=1)
    assert result["status"] == "unavailable" and result["segments"] == []
    assert not list((speech.tmp_path / "home").rglob("transcript-*.json"))


def test_speech_quotes_restore_and_need_rereading_after_compaction(speech):
    review = _review(speech)
    result = review.transcribe_artifact(2, end_seconds=1)
    segment = result["segments"][0]
    citation = review.cite_artifact(2, segment["locator"], "Delivery confirmed")
    assert citation["transcription"]["end_seconds"] == 1
    assert review.validate(_verdict(review, citation)) is None
    read = review.read_artifact(2, segment["locator"], offset=9, limit=9)
    assert read["text"] == "confirmed"
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.validate(_verdict(restored, citation)) is None
    restored.compacted({"trigger": "test"})
    assert "latest compaction" in restored.validate(_verdict(restored, citation))
    with pytest.raises(ValueError, match="latest compaction"):
        restored.cite_artifact(2, segment["locator"], "Delivery confirmed")
    restored.read_artifact(2, segment["locator"])
    assert restored.validate(_verdict(restored, citation)) is None
    assert len(speech.calls) == 1
    speech.available = False
    with pytest.raises(ValueError, match="stale"):
        EvidenceReview.restore(review.snapshot, review.directory)


def test_transcript_cannot_certify_whole_audio_or_other_windows(speech):
    review = _review(speech)
    first = review.transcribe_artifact(2, end_seconds=.5)
    citation = review.cite_artifact(2, first["segments"][0]["locator"], "Delivery confirmed")
    assert not first["complete"] and not first["audio_complete"]
    for field in review.fields:
        review.read(field)
    assert "unread or unavailable artifact" in review.validate(
        _verdict(review, citation, scope="whole_trace"),
    )
    # Another model/window generated outside this review cannot supply an issued locator.
    second = artifact_transcript(speech.item, 2, start_seconds=.5, end_seconds=1)
    with pytest.raises(ValueError, match="transcribe_artifact first"):
        review.cite_artifact(2, second["segments"][0]["locator"], "Delivery confirmed")


def test_selected_archive_wav_keeps_archive_member_and_speech_provenance(speech):
    archived(speech)
    with capture_artifact_reads() as receipts:
        result = trace.artifact_transcript(
            speech.item, 2, end_seconds=1, member_path="audio/session.wav",
        )
    assert result["status"] == "available"
    assert result["sha256"] == hashlib.sha256(speech.body).hexdigest()
    assert result["archive_sha256"] == hashlib.sha256(speech.provider.body).hexdigest()
    assert receipts[0]["member_path"] == "audio/session.wav"
    assert validate_artifact_reads(speech.item, receipts)
    altered = deepcopy(receipts)
    altered[0]["member_path"] = "audio/another.wav"
    assert not validate_artifact_reads(speech.item, altered)
    review = _review(speech)
    reviewed = review.transcribe_artifact(2, end_seconds=1, member_path="audio/session.wav")
    citation = review.cite_artifact(
        2, reviewed["segments"][0]["locator"], "Delivery confirmed",
        member_path="audio/session.wav",
    )
    assert citation["member_path"] == "audio/session.wav"
    assert review.validate(_verdict(review, citation)) is None
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.validate(_verdict(restored, citation)) is None
    assert len(speech.calls) == 1


@pytest.mark.parametrize("mutation", [
    {"transcription": {"model": "base"}},
    {"transcription": {"model": "base", "start_seconds": 0, "end_seconds": 1, "path": "/tmp"}},
    {"page": 1},
])
def test_invalid_speech_receipt_selectors_are_rejected(speech, mutation):
    with capture_artifact_reads() as receipts:
        artifact_transcript(speech.item, 2, end_seconds=1)
    assert not validate_artifact_reads(speech.item, [{**receipts[0], **mutation}])


@pytest.mark.parametrize("held_lock", ["thread", "file"])
def test_cancellation_interrupts_waiting_for_transcript_cache_lock(speech, monkeypatch, held_lock):
    artifact_transcript(speech.item, 2, end_seconds=1)
    lock_path = next((speech.tmp_path / "home").rglob("transcript-*.lock"))
    waiting, cancellation = Event(), Cancellation()

    @contextmanager
    def observed_lock(path):
        waiting.set()
        with cache_lock(path):
            yield

    @contextmanager
    def file_lock(path):
        fcntl = pytest.importorskip("fcntl")
        with path.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def read_in_worker():
        with cancellation_scope(cancellation):
            return artifact_transcript(speech.item, 2, end_seconds=1)

    monkeypatch.setattr(speech_access, "_cache_lock", observed_lock)
    hold = cache_lock if held_lock == "thread" else file_lock
    with ThreadPoolExecutor(max_workers=1) as workers:
        with hold(lock_path):
            pending = workers.submit(read_in_worker)
            assert waiting.wait(2), "Worker did not reach the held transcript lock"
            cancellation.cancel()
            with pytest.raises(asyncio.CancelledError):
                pending.result(timeout=2)
    assert len(speech.calls) == 1
