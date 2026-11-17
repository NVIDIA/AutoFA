# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact retrieval, cache identity and archive members pin the bytes every consumer sees."""

import base64
import builtins
import hashlib
import io
import json
import stat
import struct
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image

from autofa import trace
from autofa.core.dependencies import digest
from autofa.core.sandbox import run_detector
from autofa.data import artifact_archive as archive
from autofa.data import artifact_content as content
from autofa.data import artifact_store
from autofa.data.artifact_access import (
    artifact_evidence,
    artifact_image,
    capture_artifact_reads,
    validate_artifact_reads,
)
from autofa.data.artifact_delivery import source_image_delivery
from autofa.data.artifact_store import ArtifactStore, ensure_artifact
from autofa.data.remote_api import DataApiClient, DataApiError, _ArtifactRedirectHandler
from autofa.llm.hub import HubLLM
from autofa.mcp import tools_artifacts
from autofa.mcp.server import TOOL_CALLABLES, tool_response
from autofa.metrics.claude_judge import _evidence_tool_result
from autofa.metrics.codex_judge import _codex_tool_result

from support.traces import Api, Provider, compact, item, record, serve, zip_bytes


def mutate(item, change):
    """Apply ``change`` to the item's execution envelope in place."""
    value = json.loads(item["response_text"])
    change(value["execution"])
    item["response_text"] = json.dumps(value)
    return item


@pytest.fixture
def configured(tmp_path, monkeypatch):
    return serve(monkeypatch, tmp_path)


def test_discovery_helper_and_detector_reuse_one_download_and_record_identical_evidence(configured):
    evidence = trace.artifact(item(), 2)
    assert evidence["status"] == "available" and evidence["complete"] is True
    assert [segment["locator"] for segment in evidence["segments"]] == ["line:1", "line:2"]
    assert "".join(segment["text"] for segment in evidence["segments"]) == configured.body.decode()
    helper = {
        "role": "helper", "entrypoint": "document", "input_schema": {}, "output_schema": {},
        "source": "from autofa.trace import artifact\ndef document(item):\n"
                  "    result = artifact(item, 2)\n"
                  "    return ''.join(segment['text'] for segment in result['segments'])\n",
        "helpers": {}, "detectors": {},
    }
    revision = digest(helper)
    bundle = {"version": 1, "helpers": {"doc": revision}, "detectors": {},
              "nodes": {revision: helper}}
    source = "def detect(item):\n    return 'IMPORTANT' in helpers.call('doc', item)\n"
    localisation = {}
    assert run_detector(source, item(), bundle=bundle, localisation=localisation) is True
    assert configured.calls == 1
    assert len(localisation["artifact_reads"]) == 1
    assert validate_artifact_reads(item(), localisation["artifact_reads"])
    assert configured.calls == 1


def test_cache_repair_preserves_receipt_and_rejects_changed_remote_bytes(configured):
    with capture_artifact_reads() as receipts:
        evidence = trace.artifact(item(), 2)
    assert validate_artifact_reads(item(), receipts)
    Path(evidence["download_path"]).write_bytes(b"corrupted")
    repaired = trace.artifact(item(), 2)
    assert repaired["status"] == "available" and configured.calls == 2
    assert validate_artifact_reads(item(), receipts)
    Path(evidence["download_path"]).write_bytes(b"corrupted again")
    configured.body = b"different upstream artifact"
    assert not validate_artifact_reads(item(), receipts)
    rejected = trace.artifact(item(), 2)
    assert rejected["status"] == "error" and not rejected["complete"]
    assert rejected["segments"] == []


def test_mutation_between_verification_and_extraction_never_changes_pinned_evidence(
    configured, monkeypatch,
):
    saved = ensure_artifact(item(), 2)
    inspect = content.inspect_artifact
    def mutate_original(snapshot, metadata, *, output_dir):
        Path(saved["path"]).write_bytes(b"CHANGED ORIGINAL CACHE")
        return inspect(snapshot, metadata, output_dir=output_dir)
    monkeypatch.setattr(content, "inspect_artifact", mutate_original)
    evidence = artifact_evidence(item(), 2)
    assert evidence["status"] == "available"
    assert "".join(s["text"] for s in evidence["segments"]) == configured.body.decode()
    assert evidence["sha256"] == saved["sha256"]


def test_mutation_before_snapshot_is_explicitly_unavailable(configured, monkeypatch):
    actual = artifact_store.ensure_artifact
    def changed(*args, **kwargs):
        saved = actual(*args, **kwargs)
        Path(saved["path"]).write_bytes(b"tampered between reads")
        return saved
    monkeypatch.setattr(artifact_store, "ensure_artifact", changed)
    evidence = artifact_evidence(item(), 2)
    assert evidence["status"] == "error" and not evidence["verified"]
    assert evidence["segments"] == []
    assert "changed after cache verification" in evidence["reason"]


def test_legacy_catalog_text_remains_partial_without_authorizing_any_download(configured):
    original = mutate(item(text="Previously exported text"),
                      lambda execution: execution.pop("source"))
    result = trace.artifact(original, 2)
    assert result["status"] == "partial" and not result["complete"] and not result["verified"]
    assert result["segments"] == [{"locator": "exported_text", "text": "Previously exported text"}]
    assert configured.calls == 0
    assert "download_path" not in result


def test_concurrent_stores_download_same_artifact_once(tmp_path):
    started, release = Event(), Event()
    class Blocking(Provider):
        def download_artifact(self, *args):
            started.set()
            assert release.wait(5)
            return super().download_artifact(*args)
    provider = Blocking()
    first, second = ArtifactStore(tmp_path, provider), ArtifactStore(tmp_path, provider)
    with ThreadPoolExecutor(max_workers=2) as workers:
        a = workers.submit(ensure_artifact, item(), 2, store=first)
        assert started.wait(5)
        b = workers.submit(ensure_artifact, item(), 2, store=second)
        release.set()
        results = [a.result(5), b.result(5)]
    assert all(result["status"] == "available" for result in results)
    assert results[0]["sha256"] == results[1]["sha256"] and provider.calls == 1
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 1


def test_derived_cache_symlink_is_not_followed(configured, tmp_path):
    saved = ensure_artifact(item(), 2)
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(saved["path"]).parent / "derived").symlink_to(outside, target_is_directory=True)
    result = trace.artifact(item(), 2)
    assert result["status"] == "error"
    assert "symbolic" in result["reason"] and not list(outside.iterdir())


def test_image_page_receipt_contains_verified_delivered_pixels(tmp_path, monkeypatch):
    with io.BytesIO() as buffer:
        Image.new("RGB", (8, 6), "red").save(buffer, format="PNG")
        body = buffer.getvalue()
    provider = serve(monkeypatch, tmp_path, body)
    original = item(body, name="image.png", mime="image/png")
    with capture_artifact_reads() as receipts:
        image = trace.artifact_image(original, 2)
    assert image["status"] == "available" and image["data_url"].startswith("data:image/png;base64,")
    assert image["artifact_sha256"] == hashlib.sha256(body).hexdigest()
    assert validate_artifact_reads(original, receipts)
    Path(image["path"]).write_bytes(b"corrupt rendered cache")
    assert validate_artifact_reads(original, receipts)  # Re-render from pinned originals.
    assert provider.calls == 1


def test_optional_private_source_adapter_absence_is_explicit_unsupported(tmp_path, monkeypatch):
    original_import = builtins.__import__
    def without_private(name, *args, **kwargs):
        if name == "autofa.data.remote_api":
            raise ImportError("source adapter excluded from public build")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_private)
    result = ensure_artifact(item(), 2, store=ArtifactStore(tmp_path))
    assert result["status"] == "unsupported" and not result["verified"]
    assert "adapter" in result["reason"]


def test_source11_import_marker_and_default_trusted_adapter_work_together(tmp_path, monkeypatch):
    module = pytest.importorskip("autofa.data.source_export")
    remote = pytest.importorskip("autofa.data.remote_api")
    import pyarrow.parquet as pq

    source = record(task="task")
    source["tables"] = {"artifacts": [{"artifact_idx": 2, "name": "evidence.txt",
                                     "mime": "text/plain", "size_bytes": 5,
                                     "sha256": hashlib.sha256(b"hello").hexdigest()}]}
    prepared = module.materialize_source_traces(
        compact(tmp_path / "compact.parquet", task="task"), api=Api([source]),
        run_id="run", catalog={"run_id": "run"},
    )
    imported = pq.read_table(prepared).to_pylist()[0]
    execution = json.loads(imported["response_text"])["execution"]
    assert execution["source"]["kind"] == "trace_api"
    assert len(execution["source"]["trace_export_sha256"]) == 64
    calls = []
    def download(self, run, task, repeat, artifact_idx, dest):
        calls.append((self.base_url, run, task, repeat, artifact_idx))
        dest.write_bytes(b"hello")
        return dest
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    monkeypatch.setattr("autofa.config.load_config",
                        lambda: SimpleNamespace(data_api_base_url="https://api.test"))
    monkeypatch.setattr(remote.DataApiClient, "download_artifact", download)
    result = trace.artifact(imported, 2)
    assert result["status"] == "available" and result["segments"][0]["text"] == "hello"
    assert calls == [("https://api.test", "run", "task", 0, 2)]


def rendered_page(path, output_dir, data, page=1):
    output = output_dir / "renderer-output.png"
    output.write_bytes(data)
    return {"status": "available", "complete": True, "path": str(output),
            "mime_type": "image/png", "sha256": hashlib.sha256(data).hexdigest(),
            "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "locator": f"page:{page}", "page_count": 1, "limitations": []}


def test_derived_render_is_pinned_and_changed_repair_is_rejected(configured, monkeypatch):
    calls = []
    def changing_renderer(path, metadata, *, output_dir, page):
        # A renderer can change output with the environment even for identical
        # inputs. Only the first successful rendering is accepted for this version.
        data = b"first rendered image" if not calls else b"different rendered image"
        calls.append(page)
        return rendered_page(path, output_dir, data, page)
    monkeypatch.setattr(content, "render_artifact", changing_renderer)
    with capture_artifact_reads() as receipts:
        first = trace.artifact_image(item(), 2)
    assert first["status"] == "available"
    assert trace.artifact_image(item(), 2)["data_url"] == first["data_url"]
    assert validate_artifact_reads(item(), receipts) and calls == [1]
    Path(first["path"]).write_bytes(b"corrupt image cache")
    changed = trace.artifact_image(item(), 2)
    assert changed["status"] == "error" and "pinned image" in changed["reason"]
    assert "data_url" not in changed and calls == [1, 1]
    assert not validate_artifact_reads(item(), receipts)


def test_unavailable_renderer_is_retried_when_it_becomes_available(configured, monkeypatch):
    calls = []
    def installable(path, metadata, *, output_dir, page):
        calls.append(page)
        if len(calls) == 1:
            return {"status": "unavailable", "complete": False,
                    "limitations": ["renderer not installed"]}
        return rendered_page(path, output_dir, b"rendered", page)
    monkeypatch.setattr(content, "render_artifact", installable)
    assert trace.artifact_image(item(), 2)["status"] == "unavailable"
    assert trace.artifact_image(item(), 2)["status"] == "available"
    assert trace.artifact_image(item(), 2)["status"] == "available"
    assert calls == [1, 1] and configured.calls == 1


# Discovery tools, classifiers and sandboxed detectors receive the same evidence.

def image_receipt(data=b"test image bytes"):
    return {"status": "available", "sha256": hashlib.sha256(data).hexdigest(),
            "artifact_sha256": "original", "locator": "page:2", "artifact_idx": 4,
            "data_url": "data:image/png;base64," + base64.b64encode(data).decode()}


def test_discovery_returns_actual_image_blocks_without_base64_in_text(monkeypatch):
    image = image_receipt()
    monkeypatch.setattr(tools_artifacts, "_load_item_raw", lambda item_id: {})
    monkeypatch.setattr(trace, "artifact_image", lambda *args, **kwargs: image)
    assert TOOL_CALLABLES["view_artifact"] is tools_artifacts.view_artifact
    result = tools_artifacts.view_artifact("item", 4, page=2)
    blocks = tool_response(result, "view_artifact")["content"]
    assert [block["type"] for block in blocks] == ["text", "image"]
    assert "data_url" not in blocks[0]["text"]
    assert blocks[1]["data"] == image["data_url"].split(",")[1]
    assert json.loads(blocks[0]["text"])["artifact"]["locator"] == "page:2"


def test_artifact_reader_pages_long_segments_without_losing_locations(monkeypatch):
    text = "é evidence " * 5000
    monkeypatch.setattr(tools_artifacts, "_load_item_raw", lambda item_id: {})
    monkeypatch.setattr(trace, "artifact", lambda *args: {
        "status": "available", "complete": True, "sha256": "source",
        "segments": [{"locator": "sheet:Budget!A1", "text": text}],
    })
    overview = tools_artifacts.read_artifact("a", 4)
    assert overview["value"]["segment_count"] == 1
    offset, parts = 0, []
    while True:
        result = tools_artifacts.read_artifact(
            "a", 4, section="segments", pointer="/0/text", offset=offset, max_chars=1000,
        )
        assert result["ok"] and result["sha256"] == "source"
        parts.append(result["value"])
        if result["is_final"]:
            break
        offset = result["next_offset"]
    assert "".join(parts) == text


def test_unavailable_artifact_view_is_explicit_not_a_blank_image(monkeypatch):
    monkeypatch.setattr(tools_artifacts, "_load_item_raw", lambda item_id: {})
    monkeypatch.setattr(trace, "artifact_image", lambda *a, **k: {
        "status": "unavailable", "reason": "Office renderer missing",
    })
    result = tools_artifacts.view_artifact("a", 0)
    assert not result["ok"] and "Office renderer" in result["error"]
    assert [c["type"] for c in tool_response(result, "view_artifact")["content"]] == ["text"]


def hub_client():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_image_bytes_reach_classifier_and_cache_keys_distinguish_visual_inputs():
    client, calls = hub_client()
    hub = HubLLM(client, model="artifact-test-model")
    HubLLM._CLASSIFY_CACHE.clear()
    a, b = image_receipt(b"first"), image_receipt(b"second")
    assert hub.classify("Compare table", "context", images=[a]) == "yes"
    assert hub.classify("Compare table", "context", images=[a]) == "yes"
    assert hub.classify("Compare table", "context", images=[b]) == "yes"
    assert len(calls) == 2
    message = calls[0]["messages"][0]["content"]
    assert message[0]["type"] == "text"
    assert message[1]["image_url"]["url"] == a["data_url"]
    assert hub.classify_options("context", "Check", images=[a]) == "yes"
    assert calls[-1]["messages"][0]["content"][1]["type"] == "image_url"


@pytest.mark.parametrize("image", [
    {"status": "unavailable"}, {"status": "available", "data_url": "https://file.test/img"},
    {**image_receipt(), "sha256": "wrong"},
])
def test_missing_or_invalid_images_never_silently_classify_only_text(image):
    client, calls = hub_client()
    with pytest.raises(ValueError):
        HubLLM(client, model="image-test").classify("Check", "text", images=[image])
    assert calls == []


# Smaller judge delivery is allowed only when source bytes reproduce pinned image pixels.

def encoded(image, **options):
    stream = io.BytesIO()
    image.save(stream, format="PNG",
               **({"exif": image.info["exif"]} if "exif" in image.info else {}), **options)
    return stream.getvalue()


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    image = Image.new("RGB", (600, 600), (12, 45, 78))
    original, rendered = encoded(image), encoded(image, compress_level=0)
    path = tmp_path / "source.png"
    path.write_bytes(original)
    source = {"status": "available", "verified": True, "path": str(path),
              "sha256": hashlib.sha256(original).hexdigest()}
    monkeypatch.setattr(artifact_store, "ensure_artifact", lambda *a, **kw: source)
    receipt = {"status": "available", "format": "image", "page": 1, "artifact_idx": 2,
               "sha256": hashlib.sha256(rendered).hexdigest(),
               "artifact_sha256": source["sha256"],
               "data_url": "data:image/png;base64," + base64.b64encode(rendered).decode()}
    return SimpleNamespace(receipt=receipt, source=source, image=image, original=original,
                           rendered=rendered, path=path)


@pytest.mark.parametrize("change", ["pixels", "orientation", "unverified"])
def test_nonidentical_or_unverified_original_keeps_pinned_render(delivery, change):
    image = delivery.image.copy()
    if change == "pixels":
        image.putpixel((0, 0), (255, 0, 0))
    elif change == "orientation":
        exif = image.getexif()
        exif[274] = 3
        image.info["exif"] = exif.tobytes()
    else:
        delivery.source["verified"] = False
    if change != "unverified":
        delivery.path.write_bytes(encoded(image))
        changed = hashlib.sha256(delivery.path.read_bytes()).hexdigest()
        delivery.source["sha256"] = delivery.receipt["artifact_sha256"] = changed
    payload, receipt = source_image_delivery(delivery.receipt, {"response_text": "source"})
    assert base64.b64decode(payload["data"]) == delivery.rendered
    assert receipt is None


def test_cache_change_is_rejected_before_delivery(delivery):
    delivery.path.write_bytes(b"changed cached bytes")
    with pytest.raises(ValueError, match="changed before delivery"):
        source_image_delivery(delivery.receipt, {"response_text": "source"})


def test_both_harness_payloads_deliver_actual_unchanged_image_bytes(delivery):
    receipt, recorded = delivery.receipt, []

    class Review:
        artifact_source_fields = {"response_text": "source"}

        def view_artifact(self, **kwargs):
            return delivery.receipt

        def record(self, event, **kwargs):
            assert event == "artifact_delivery"
            sent = kwargs["delivery"]
            assert sent["decoded_rgb_identical"] and sent["width"] == sent["height"] == 600
            assert sent["rendered_sha256"] == receipt["sha256"]
            assert sent["source_bytes"] < sent["rendered_bytes"]
            recorded.append(sent)

    result = _evidence_tool_result("view_artifact", Review().view_artifact, {"artifact_idx": 2})
    assert not result.get("isError") and len(recorded) == 1
    assert json.loads(result["content"][0]["text"])["image_delivery"]["decoded_rgb_identical"]
    assert base64.b64decode(result["content"][1]["data"]) == delivery.original
    codex = _codex_tool_result(result)
    assert codex["contentItems"][1]["type"] == "inputImage"
    assert base64.b64decode(codex["contentItems"][1]["imageUrl"].split(",", 1)[1]) == (
        delivery.original)


# Artifact store retrieval, cache identity and safe HTTP transport.

def artifact_item(body=b"artifact bytes", *, declared=True, **metadata):
    identity = {"run_id": "run", "task_id": "task/one", "repeat": 0, "agent_ref": "agent"}
    artifact = {**identity, "artifact_idx": 2, "name": "../../outside.txt",
                "uri": "https://untrusted.example/secret?credential=never-send", **metadata}
    if declared:
        artifact.update(size_bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    return {"response_text": json.dumps({"steps": [], "execution": {
        "rollout": identity, "artifacts": [artifact],
        "source": {"kind": "trace_api", "base_url": "https://api.example/base",
                   "trace_export_sha256": "a" * 64, "export_generation": "generation-1"},
    }})}


class RecordingProvider:
    base_url = "https://api.example/base"

    def __init__(self, body=b"artifact bytes", error=None):
        self.body, self.error, self.calls = body, error, []

    def download_artifact(self, run_id, task_id, repeat, artifact_idx, dest):
        self.calls.append((run_id, task_id, repeat, artifact_idx))
        if self.error:
            raise self.error
        dest.write_bytes(self.body)
        return dest


def test_retained_identity_downloads_verified_bytes_and_reuses_shared_cache(tmp_path):
    provider = RecordingProvider()
    store = ArtifactStore(tmp_path, provider)
    item = artifact_item()
    first = ensure_artifact(item, 2, store=store)
    assert first["status"] == "available" and first["verified"] is True
    assert Path(first["path"]).parent == tmp_path / "sha256"
    assert Path(first["path"]).name == hashlib.sha256(provider.body).hexdigest()
    assert Path(first["path"]).read_bytes() == provider.body
    assert first["source_identity"]["trace_export_sha256"] == "a" * 64
    assert first["source_identity"]["export_generation"] == "generation-1"
    assert provider.calls == [("run", "task/one", 0, 2)]
    second = ensure_artifact(item, 2, store=ArtifactStore(tmp_path, provider))
    assert first == second
    assert len(provider.calls) == 1
    assert not (tmp_path.parent / "outside.txt").exists()
    assert not list(tmp_path.glob("download-*"))


def test_corrupt_cache_repair_must_reproduce_original_pinned_bytes(tmp_path):
    # Without declared integrity metadata only the receipt pin protects the repair.
    provider = RecordingProvider()
    store = ArtifactStore(tmp_path, provider)
    item = artifact_item(declared=False)
    original = ensure_artifact(item, 2, store=store)
    Path(original["path"]).write_bytes(b"corrupt")
    provider.body = b"changed remote bytes"
    rejected = ensure_artifact(item, 2, store=store)
    assert rejected["status"] == "error" and rejected["verified"] is False
    assert "pin" in rejected["reason"]
    receipt = json.loads(next((tmp_path / "receipts").glob("*.json")).read_text())
    assert receipt["sha256"] == original["sha256"]
    provider.body = b"artifact bytes"
    repaired = ensure_artifact(item, 2, store=store)
    assert repaired["sha256"] == original["sha256"] and repaired["verified"] is True


def test_new_source_revision_has_distinct_pin_without_overwriting_previous_blob(tmp_path):
    provider = RecordingProvider()
    store = ArtifactStore(tmp_path, provider)
    first = ensure_artifact(artifact_item(declared=False), 2, store=store)
    revised = mutate(artifact_item(declared=False),
                     lambda execution: execution["source"].update(trace_export_sha256="b" * 64))
    provider.body = b"revision two"
    second = ensure_artifact(revised, 2, store=store)
    assert second["verified"] is True and first["sha256"] != second["sha256"]
    assert first["cache_key"] != second["cache_key"]
    assert Path(first["path"]).read_bytes() == b"artifact bytes"
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 2


@pytest.mark.parametrize("declared", [{"size_bytes": 100}, {"sha256": "0" * 64}],
                         ids=["size", "digest"])
def test_declared_integrity_mismatch_never_becomes_a_cache_receipt(tmp_path, declared):
    item = mutate(artifact_item(), lambda execution: execution["artifacts"][0].update(declared))
    result = ensure_artifact(item, 2, store=ArtifactStore(tmp_path, RecordingProvider()))
    assert result["status"] == "error" and not result["verified"]
    assert not list((tmp_path / "receipts").glob("*.json"))
    assert not list((tmp_path / "sha256").iterdir())


@pytest.mark.parametrize("case", ["missing", "ambiguous", "grader_parent", "identity", "origin",
                                  "bad_hash", "bad_size"])
def test_unretained_untrusted_or_invalid_artifact_never_invokes_provider(tmp_path, case):
    item, provider = artifact_item(), RecordingProvider()
    def change(execution):
        artifact = execution["artifacts"][0]
        if case == "missing":
            execution["artifacts"] = []
        elif case == "grader_parent":
            artifact["event_idx"] = 4
            execution["events"] = [{"event_idx": 4, "lane": "grader", "actor": "grader"}]
        elif case == "ambiguous":
            execution["artifacts"].append(artifact.copy())
        elif case == "identity":
            artifact["task_id"] = "other task"
        elif case == "origin":
            execution["source"]["base_url"] = "https://untrusted.example"
        elif case == "bad_hash":
            artifact["sha256"] = "../../escape"
        elif case == "bad_size":
            artifact["size_bytes"] = True  # bool is an int subtype, not a byte count
    mutate(item, change)
    result = ensure_artifact(item, 2, store=ArtifactStore(tmp_path, provider))
    assert result["status"] in {"unavailable", "error"}
    assert not result["verified"] and provider.calls == []


@pytest.mark.parametrize("status,expected", [(404, "unavailable"), (405, "unsupported"),
                                             (503, "error")])
def test_source_availability_is_explicit(tmp_path, status, expected):
    provider = RecordingProvider(error=DataApiError("source error", status=status))
    result = ensure_artifact(artifact_item(), 2, store=ArtifactStore(tmp_path, provider))
    assert result["status"] == expected and result["verified"] is False
    assert not list((tmp_path / "receipts").glob("*.json"))


def test_corrupt_receipt_is_not_silently_repinned(tmp_path):
    provider = RecordingProvider()
    store = ArtifactStore(tmp_path, provider)
    result = ensure_artifact(artifact_item(declared=False), 2, store=store)
    receipt = tmp_path / "receipts" / (result["cache_key"] + ".json")
    receipt.write_text("partial")
    rejected = ensure_artifact(artifact_item(declared=False), 2, store=store)
    assert rejected["status"] == "error" and "pin" in rejected["reason"]
    assert len(provider.calls) == 1


def test_empty_artifact_is_valid_when_declared_empty(tmp_path):
    store = ArtifactStore(tmp_path, RecordingProvider(b""))
    result = ensure_artifact(artifact_item(b""), 2, store=store)
    assert result["verified"] is True and result["size_bytes"] == 0


def test_symlink_cache_directory_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "cache"
    root.mkdir()
    (root / "sha256").symlink_to(outside, target_is_directory=True)
    provider = RecordingProvider()
    result = ensure_artifact(artifact_item(), 2, store=ArtifactStore(root, provider))
    assert result["status"] == "error" and provider.calls == []
    assert not list(outside.iterdir())


class Response(io.BytesIO):
    def __init__(self, body, declared=None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if declared is None else declared)}

    def read(self, size=-1):
        assert 0 < size <= 256 * 1024
        return super().read(size)


def test_transport_streams_identity_endpoint_and_retries_incomplete_body(tmp_path, monkeypatch):
    responses = iter([Response(b"cut", declared=8), Response(b"complete")])
    requests, delays = [], []
    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return next(responses)
    monkeypatch.setattr(urllib.request, "build_opener", lambda handler: Opener())
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", delays.append)
    destination = tmp_path / "artifact"
    destination.write_bytes(b"prior verified bytes")
    DataApiClient("https://api.example/base").download_artifact(
        "run", "task/one", 0, 2, destination,
    )
    assert destination.read_bytes() == b"complete" and len(requests) == 2
    assert requests[0].full_url == (
        "https://api.example/base/v1/rollouts/run/task%2Fone/0/artifacts/2/download"
    )
    assert delays and not destination.with_suffix(".partial").exists()


def test_redirect_drops_credentials_and_preserves_signed_storage_query():
    request = urllib.request.Request("https://api.example/artifact", headers={
        "Authorization": "Bearer never-forward", "Cookie": "session=never-forward",
        "X-Api-Key": "never-forward", "Proxy-Authorization": "never-forward",
    })
    redirected = _ArtifactRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, "https://storage.example/object?signature=source-issued",
    )
    assert redirected.full_url.endswith("?signature=source-issued")
    assert redirected.header_items() == [("Accept", "application/octet-stream")]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://storage.example/object",
                                  "https://user:password@storage.example/object"])
def test_unsafe_artifact_redirect_is_rejected(url):
    with pytest.raises(DataApiError, match="redirect"):
        _ArtifactRedirectHandler().redirect_request(
            urllib.request.Request("https://api.example/artifact"), None, 302, "Found", {}, url,
        )


# Archive member selection preserves provenance without extracting arbitrary file trees.

@pytest.fixture
def zipped(tmp_path, monkeypatch):
    return serve(monkeypatch, tmp_path / "home", zip_bytes([
        ("documents/", b""), ("documents/report.txt", b"Evidence inside archive\n"),
        ("other.txt", b"Other evidence\n"),
    ]))


def archive_record(provider, **metadata):
    return item(provider.body, name="deliverables.zip", mime="application/zip", **metadata)


def test_manifest_lists_members_without_decompression_or_complete_content_claim(
    tmp_path, monkeypatch,
):
    path = tmp_path / "files.zip"
    path.write_bytes(zip_bytes([("report.txt", b"Stored text"), ("folder/", b"")]))

    def no_extract(*args, **kwargs):
        raise AssertionError("Inventory must not decompress members")

    monkeypatch.setattr(zipfile.ZipFile, "open", no_extract)
    result = archive.inspect_archive(path)
    assert result["complete"] is False
    assert result["archive"]["inventory_complete"] is True
    assert [row["member_path"] for row in result["archive"]["members"]] == [
        "report.txt", "folder/",
    ]


def test_member_text_download_and_receipts_keep_both_archive_and_member_identity(zipped):
    source = archive_record(zipped)
    with capture_artifact_reads() as receipts:
        evidence = artifact_evidence(source, 2, member_path="documents/report.txt")
    assert evidence["status"] == "available" and evidence["complete"]
    assert evidence["segments"][0]["locator"] == "line:1"
    assert evidence["segments"][0]["text"] == "Evidence inside archive\n"
    assert evidence["archive_sha256"] == hashlib.sha256(zipped.body).hexdigest()
    assert evidence["sha256"] == hashlib.sha256(b"Evidence inside archive\n").hexdigest()
    assert evidence["source_identity"]["member_path"] == "documents/report.txt"
    assert evidence["source_identity"]["archive_source_identity"]["task_id"] == "task"
    assert Path(evidence["download_path"]).name == evidence["sha256"]
    assert receipts[0]["member_path"] == "documents/report.txt"
    assert validate_artifact_reads(source, receipts)
    assert zipped.calls == 1
    original = ensure_artifact(source, 2)
    assert Path(original["path"]).read_bytes() == zipped.body
    assert "member_path" not in original
    changed_selection = [{**receipts[0], "member_path": "other.txt"}]
    assert not validate_artifact_reads(source, changed_selection)


def test_two_members_with_identical_bytes_have_distinct_provenance(zipped):
    zipped.body = zip_bytes([("first.txt", b"same"), ("second.txt", b"same")])
    source = archive_record(zipped)
    first = ensure_artifact(source, 2, member_path="first.txt")
    second = ensure_artifact(source, 2, member_path="second.txt")
    assert first["sha256"] == second["sha256"]
    assert first["cache_key"] != second["cache_key"]
    assert first["source_identity"] != second["source_identity"]


def test_member_cache_repair_recovers_exact_pinned_bytes(zipped):
    source = archive_record(zipped)
    with capture_artifact_reads() as receipts:
        first = artifact_evidence(source, 2, member_path="documents/report.txt")
    Path(first["download_path"]).write_bytes(b"corrupted member")
    assert validate_artifact_reads(source, receipts)
    assert Path(first["download_path"]).read_bytes() == b"Evidence inside archive\n"
    assert zipped.calls == 1
    parent = ensure_artifact(source, 2)
    Path(parent["path"]).write_bytes(b"corrupted archive")
    zipped.body = zip_bytes([("documents/report.txt", b"new remote evidence")])
    assert not validate_artifact_reads(source, receipts)


def test_selected_png_has_verified_visual_evidence_and_member_receipt(zipped):
    output = io.BytesIO()
    with Image.new("RGB", (8, 5), "red") as picture:
        picture.save(output, "PNG")
    zipped.body = zip_bytes([("charts/result.png", output.getvalue())])
    source = archive_record(zipped)
    with capture_artifact_reads() as receipts:
        viewed = artifact_image(source, 2, member_path="charts/result.png")
    assert viewed["status"] == "available" and viewed["data_url"].startswith("data:image/png;")
    assert viewed["artifact_sha256"] == hashlib.sha256(output.getvalue()).hexdigest()
    assert viewed["archive_sha256"] == hashlib.sha256(zipped.body).hexdigest()
    assert receipts[0]["member_path"] == "charts/result.png" and receipts[0]["page"] == 1
    assert validate_artifact_reads(source, receipts)


@pytest.mark.parametrize("member_path", ["../report.txt", "/report.txt", "C:/report.txt",
                                         "folder\\report.txt", "./report.txt", "", "a\x00b"])
def test_unsafe_selection_fails_before_download(zipped, member_path):
    result = ensure_artifact(archive_record(zipped), 2, member_path=member_path)
    assert result["status"] == "error" and not result["verified"]
    assert zipped.calls == 0


@pytest.mark.parametrize("entries", [
    [("../escape.txt", b"bad"), ("safe.txt", b"safe")],
    [("same.txt", b"one"), ("same.txt", b"two")],
    [("node", b"file"), ("node/child", b"ambiguous")],
])
def test_invalid_inventory_cannot_be_bypassed_by_selecting_safe_member(zipped, entries):
    zipped.body = zip_bytes(entries)
    saved = ensure_artifact(archive_record(zipped), 2)
    with pytest.raises(ValueError):
        archive.inspect_archive(Path(saved["path"]))
    selected = ensure_artifact(archive_record(zipped), 2, member_path="safe.txt")
    assert selected["status"] == "error" and not selected["verified"]


def test_symlinks_and_devices_are_rejected(zipped):
    for mode in (stat.S_IFLNK, stat.S_IFCHR, stat.S_IFIFO):
        info = zipfile.ZipInfo("special")
        info.create_system = 3
        info.external_attr = (mode | 0o777) << 16
        zipped.body = zip_bytes([(info, b"target")])
        # Give each archive an independent source revision rather than replacing its pin.
        source = mutate(archive_record(zipped), lambda execution, generation=str(mode): (
            execution["source"].update(export_generation=generation)))
        result = ensure_artifact(source, 2, member_path="special")
        assert result["status"] == "error" and "special file" in result["reason"]


def test_encrypted_members_are_explicitly_unavailable(zipped):
    data = bytearray(zip_bytes([("safe.txt", b"secret")]))
    struct.pack_into("<H", data, data.index(b"PK\x03\x04") + 6, 1)
    struct.pack_into("<H", data, data.index(b"PK\x01\x02") + 8, 1)
    zipped.body = bytes(data)
    result = ensure_artifact(archive_record(zipped), 2, member_path="safe.txt")
    assert result["status"] == "error" and "Encrypted" in result["reason"]


def test_central_count_mismatch_fails_before_zipfile_allocation(tmp_path, monkeypatch):
    data = bytearray(zip_bytes([("one", b"1"), ("two", b"2")]))
    struct.pack_into("<HH", data, data.rfind(b"PK\x05\x06") + 8, 1, 1)
    path = tmp_path / "dishonest.zip"
    path.write_bytes(data)

    def not_opened(*args, **kwargs):
        raise AssertionError("Preflight must reject dishonest count before ZipFile")

    monkeypatch.setattr(zipfile, "ZipFile", not_opened)
    with pytest.raises(ValueError, match="declared member count"):
        archive.inspect_archive(path)


@pytest.mark.parametrize("limit", ["MAX_MEMBERS", "MAX_EXPANDED_BYTES", "MAX_COMPRESSION_RATIO"])
def test_archive_resource_boundaries_are_explicit(tmp_path, monkeypatch, limit):
    path = tmp_path / "bounded.zip"
    path.write_bytes(zip_bytes([("first.txt", b"a" * 1024), ("second.txt", b"b" * 1024)]))
    monkeypatch.setattr(archive, limit, 1)
    with pytest.raises(ValueError, match="limit"):
        archive.inspect_archive(path)


def test_crc_failure_never_pins_extracted_member(zipped):
    data = bytearray(zip_bytes([("data.txt", b"correct original")], compression=zipfile.ZIP_STORED))
    index = data.index(b"correct original")
    data[index:index + 7] = b"changed"
    zipped.body = bytes(data)
    result = ensure_artifact(archive_record(zipped), 2, member_path="data.txt")
    assert result["status"] == "error" and not result["verified"]
    assert "CRC" in result["reason"]
    cache_root = Path(ensure_artifact(archive_record(zipped), 2)["path"]).parent.parent
    receipts = list((cache_root / "receipts").glob("*.json"))
    assert len(receipts) == 1  # The outer archive alone is retained.


def test_selected_member_does_not_fall_back_to_parent_exported_text(zipped):
    source = mutate(archive_record(zipped, text="Unrelated exported archive text"),
                    lambda execution: execution.pop("source"))
    evidence = artifact_evidence(source, 2, member_path="documents/report.txt")
    assert evidence["status"] == "unavailable" and evidence["segments"] == []
    assert not evidence["complete"]


def test_nested_archives_can_be_downloaded_but_not_recursively_selected(zipped):
    inner = zip_bytes([("inside.txt", b"inner content")])
    zipped.body = zip_bytes([("nested.zip", inner)])
    source = archive_record(zipped)
    downloaded = ensure_artifact(source, 2, member_path="nested.zip")
    assert downloaded["status"] == "available" and Path(downloaded["path"]).read_bytes() == inner
    failed = ensure_artifact(source, 2, member_path="nested.zip/inside.txt")
    assert failed["status"] == "error" and "absent" in failed["reason"]
