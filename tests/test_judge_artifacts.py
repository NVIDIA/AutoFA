# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judges inspect artifacts on demand, never stale copies, blocked transports or racing receipts."""

import asyncio
import base64
import hashlib
import io
import json
import random
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa import trace
from autofa.core.localisation import passage, text_hash
from autofa.data import artifact_content, artifact_store, inline_images
from autofa.data.artifact_access import capture_artifact_reads, validate_artifact_reads
from autofa.data.artifact_store import ArtifactStore, ensure_artifact
from autofa.mcp import tools_artifacts
from autofa.mcp.server import tool_response
from autofa.metrics import claude_judge, codex_judge
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot
from autofa.metrics.judge_intervals import interval_navigation
from autofa.metrics.judge_projection import source_span
from autofa.util.cancellation import Cancellation, cancellation_scope, check_cancelled

from support.traces import Provider as ArchiveProvider
from support.traces import item as archive_item
from support.traces import zip_bytes

SOURCE = "def detect(item): return True"
ANSWER = "The tallest bar is blue."


@pytest.fixture(autouse=True)
def no_configured_artifact_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "autofa"))
    explicit = ArtifactStore._provider

    def provider(self):
        if self.provider is None:
            pytest.fail("Artifacts must not consult a configured remote provider")
        return explicit(self)

    monkeypatch.setattr(ArtifactStore, "_provider", provider)


def _verdict(review, *citations, scope="local"):
    review.read("detector_source")
    return {"item_id": review.snapshot["item_id"], "snapshot_id": review.snapshot["snapshot_id"],
            "scope": scope, "citations": list(citations)}


def _png(color="red"):
    image = pytest.importorskip("PIL.Image")
    with io.BytesIO() as output:
        image.new("RGB", (8, 6), color).save(output, format="PNG")
        return output.getvalue()


# Remote trace artifacts fetched through a provider on first use.


class IndexedProvider:
    base_url = "https://trace.example"

    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def download_artifact(self, run_id, task_id, repeat, artifact_idx, dest):
        self.calls.append(artifact_idx)
        dest.write_bytes(self.bodies[artifact_idx])
        return dest


def _trace_item(catalog):
    return {"response_text": json.dumps({"steps": [], "execution": {
        "rollout": {"run_id": "run", "task_id": "task", "repeat": 0},
        "artifacts": catalog,
        "source": {"kind": "trace_api", "base_url": IndexedProvider.base_url,
                   "trace_export_sha256": "a" * 64},
    }})}


@pytest.fixture
def files(tmp_path, monkeypatch):
    provider = IndexedProvider({2: "résumé confirmed\nsecond line\n".encode()})
    store = artifact_store.ArtifactStore(tmp_path / "artifact-cache", provider)
    original = artifact_store.ensure_artifact
    monkeypatch.setattr(artifact_store, "ensure_artifact",
                        lambda item, idx, **kw: original(item, idx, store=store))
    item = _trace_item([{"artifact_idx": 2, "name": "result.txt", "mime_type": "text/plain"}])
    snapshot = make_snapshot(item, item_id="case", source=SOURCE)
    review = EvidenceReview(snapshot, tmp_path / "review")
    return SimpleNamespace(review=review, item=item, provider=provider, store=store,
                           monkeypatch=monkeypatch)


def _with_image(files, tmp_path):
    # A fixed valid PNG isolates judge delivery/identity from optional native renderers.
    pixels = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    )
    files.provider.bodies[3] = pixels
    inspect = artifact_content.inspect_artifact

    def inspect_image(path, metadata, **kwargs):
        if metadata.get("name") != "plot.png":
            return inspect(path, metadata, **kwargs)
        return {"status": "available", "complete": True, "segments": [], "limitations": [],
                "requires_visual_review": True, "visual_pages": 1, "visual_complete": False,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "transform_version": artifact_content.ARTIFACT_TRANSFORM_VERSION}

    def render(path, metadata, *, output_dir, page=1):
        assert page == 1
        output = Path(output_dir) / "fixture.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(pixels)
        return {"status": "available", "complete": True, "limitations": [], "page_count": 1,
                "sha256": hashlib.sha256(pixels).hexdigest(), "locator": "image:1",
                "original_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "path": str(output), "mime_type": "image/png",
                "transform_version": artifact_content.ARTIFACT_TRANSFORM_VERSION}

    files.monkeypatch.setattr(artifact_content, "inspect_artifact", inspect_image)
    files.monkeypatch.setattr(artifact_content, "render_artifact", render)
    envelope = json.loads(files.item["response_text"])
    envelope["execution"]["artifacts"].append(
        {"artifact_idx": 3, "name": "plot.png", "mime_type": "image/png"},
    )
    files.item = {"response_text": json.dumps(envelope)}
    files.review = EvidenceReview(
        make_snapshot(files.item, item_id="case", source=SOURCE), tmp_path / "image-review",
    )
    return files.review


def test_catalog_snapshot_is_lazy_and_binds_source_and_transform(files, monkeypatch):
    review = files.review
    catalog = review.list_artifacts()
    assert files.provider.calls == []
    assert catalog["artifacts"][0]["name"] == "result.txt"
    snapshot = review.snapshot
    alternate = json.loads(files.item["response_text"])
    alternate["execution"]["source"]["trace_export_sha256"] = "b" * 64
    assert make_snapshot({"response_text": json.dumps(alternate)}, item_id="case",
                         source=SOURCE)["snapshot_id"] != snapshot["snapshot_id"]
    monkeypatch.setattr(artifact_content, "ARTIFACT_TRANSFORM_VERSION", "next-transform")
    assert make_snapshot(files.item, item_id="case", source=SOURCE)["snapshot_id"] != (
        snapshot["snapshot_id"]
    )
    with pytest.raises(ValueError, match="transform version changed"):
        review.inspect_artifact(2)


def test_text_inspection_is_not_read_and_citations_bind_exact_segment(files):
    review = files.review
    identity_before = review.artifact_evidence_identity()
    inspected = review.inspect_artifact(2)
    assert "text" not in inspected["segments"][0]
    assert inspected["segments"][0]["locator"] == "line:1"
    assert review.artifact_reads == []
    with pytest.raises(ValueError, match="not read"):
        review.cite_artifact(2, "line:1", "confirmed")
    first = review.read_artifact(2, "line:1", limit=7)
    second = review.read_artifact(2, "line:1", offset=first["next_offset"])
    assert first["text"] + second["text"] == "résumé confirmed\n"
    assert first["artifact_sha256"] == hashlib.sha256(files.provider.bodies[2]).hexdigest()
    assert first["sha256"] == hashlib.sha256("résumé confirmed\n".encode()).hexdigest()
    citation = review.cite_artifact(2, "line:1", "confirmed")
    assert citation["start"] == 7 and citation["end"] == 16
    assert citation["source_link"] == (
        "https://trace.example/v1/rollouts/run/task/0/artifacts/2/download"
    )
    assert review.validate(_verdict(review, citation)) is None
    assert review.artifact_evidence_identity() != identity_before
    assert files.provider.calls == [2]
    assert review.validate(_verdict(review, {**citation, "locator": "line:2"})) == (
        "artifact citation was not issued by cite_artifact"
    )


def test_unread_artifacts_cannot_support_whole_trace_absence(files):
    review = files.review
    for field in review.fields:
        review.read(field)
    review.read_artifact(2, "line:1")
    citation = review.cite_artifact(2, "line:1", "confirmed")
    verdict = _verdict(review, citation, scope="whole_trace")
    assert "unread artifact 2:line:2" in review.validate(verdict)
    review.read_artifact(2, "line:2")
    assert review.validate(verdict) is None


def test_partial_verified_text_remains_readable_but_is_not_exhaustive(files, monkeypatch):
    inspect = trace.artifact

    def partial(*args):
        return {**inspect(*args), "status": "error", "complete": False,
                "limitations": ["Extraction stopped after its resource limit."]}

    monkeypatch.setattr(trace, "artifact", partial)
    review = files.review
    read = review.read_artifact(2, "line:1")
    assert read["text"] == "résumé confirmed\n" and read["complete"] is False
    citation = review.cite_artifact(2, "line:1", "confirmed")
    assert review.validate(_verdict(review, citation)) is None
    for field in review.fields:
        review.read(field)
    assert "unread or unavailable artifact 2" in review.validate(
        _verdict(review, citation, scope="whole_trace"),
    )


def test_visual_citations_need_pixels_and_mcp_delivers_actual_image(files, tmp_path, monkeypatch):
    import claude_agent_sdk as sdk

    review = _with_image(files, tmp_path)
    assert review.inspect_artifact(3)["requires_visual_review"] is True
    with pytest.raises(ValueError, match="not viewed"):
        review.cite_artifact(3, "image:1")
    handlers = {}
    create = sdk.create_sdk_mcp_server

    def capture(**kwargs):
        handlers.update({tool.name: tool.handler for tool in kwargs["tools"]})
        return create(**kwargs)

    monkeypatch.setattr(sdk, "create_sdk_mcp_server", capture)
    claude_judge._evidence_server(review)
    response = asyncio.run(handlers["view_artifact"]({"artifact_idx": 3}))
    assert [block["type"] for block in response["content"]] == ["text", "image"]
    metadata = json.loads(response["content"][0]["text"])
    assert "data_url" not in metadata
    pixels = base64.b64decode(response["content"][1]["data"])
    assert pixels.startswith(b"\x89PNG")
    assert hashlib.sha256(pixels).hexdigest() == metadata["sha256"]
    citation = review.cite_artifact(3, "image:1")
    assert citation["artifact_sha256"] == hashlib.sha256(files.provider.bodies[3]).hexdigest()
    assert review.validate(_verdict(review, citation)) is None
    review.compacted({"trigger": "test"})
    assert "latest compaction" in review.validate(_verdict(review, citation))
    review.view_artifact(3)
    assert review.validate(_verdict(review, citation)) is None


def test_visual_whole_trace_requires_all_pixels_and_honors_render_limits(
    files, tmp_path, monkeypatch,
):
    review = _with_image(files, tmp_path)
    for field in review.fields:
        review.read(field)
    for locator in ("line:1", "line:2"):
        review.read_artifact(2, locator)
    citation = review.cite_artifact(2, "line:1", "confirmed")
    review.inspect_artifact(3)
    verdict = _verdict(review, citation, scope="whole_trace")
    assert "unread or unavailable artifact visuals 3" in review.validate(verdict)
    original = trace.artifact_image

    def limited(*args, **kwargs):
        return {**original(*args, **kwargs), "limitations": ["Image resized"]}

    monkeypatch.setattr(trace, "artifact_image", limited)
    review.view_artifact(3)
    assert "incomplete artifact visual coverage 3" in review.validate(verdict)


def test_restore_revalidates_used_hashes_without_fetching_unused_files(files, tmp_path):
    review = _with_image(files, tmp_path)
    review.read_artifact(2, "line:1")
    citation = review.cite_artifact(2, "line:1", "confirmed")
    verdict = _verdict(review, citation)
    before = (review.directory / "audit.jsonl").read_bytes()
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.validate(verdict) is None
    assert restored.artifact_evidence_identity() == review.artifact_evidence_identity()
    assert (review.directory / "audit.jsonl").read_bytes() == before
    assert files.provider.calls == [2]
    # A broken cache must not silently rebind the source identity to changed upstream bytes.
    digest = review.artifact_pins[2]["sha256"]
    (files.store.root / "sha256" / digest).write_bytes(b"corrupt cache")
    files.provider.bodies[2] = b"changed upstream content"
    with pytest.raises(ValueError, match="stale"):
        EvidenceReview.restore(review.snapshot, review.directory)


@pytest.mark.parametrize("change", ["text", "source", "image"])
def test_restore_rejects_changed_text_source_or_image(files, tmp_path, monkeypatch, change):
    review = _with_image(files, tmp_path)
    review.read_artifact(2, "line:1")
    review.view_artifact(3)
    original = trace.artifact_image if change == "image" else trace.artifact

    def altered(*args, **kwargs):
        result = deepcopy(original(*args, **kwargs))
        if change == "text":
            result["segments"][0]["text"] = "different extracted text"
        elif change == "source":  # Same bytes and text from a different trace export.
            result["source_identity"]["trace_export_sha256"] = "c" * 64
        else:
            result["sha256"] = "d" * 64
        return result

    monkeypatch.setattr(trace, "artifact_image" if change == "image" else "artifact", altered)
    with pytest.raises(ValueError, match="stale"):
        EvidenceReview.restore(review.snapshot, review.directory)


def test_unavailable_reads_stay_unknown_and_changed_availability_invalidates_restore(
    files, monkeypatch,
):
    review = files.review
    monkeypatch.setattr(trace, "artifact", lambda *args: {
        "status": "unavailable", "reason": "source is offline", "segments": [],
    })
    assert review.inspect_artifact(2)["status"] == "unavailable"
    with pytest.raises(ValueError, match="unavailable"):
        review.read_artifact(2, "line:1")
    assert not review.artifact_pins and review.artifact_accesses
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.artifact_evidence_identity() == review.artifact_evidence_identity()
    monkeypatch.setattr(trace, "artifact", lambda *args: {
        "status": "unsupported", "reason": "new parser boundary", "segments": [],
    })
    with pytest.raises(ValueError, match="stale"):
        EvidenceReview.restore(review.snapshot, review.directory)


def test_catalog_evaluation_records_and_targets_are_not_judge_evidence(tmp_path):
    item = _trace_item([
        {"artifact_idx": 2, "name": "output.txt", "score": "hidden grade",
         "text": "The user-facing file legitimately mentions score=5."},
        {"artifact_idx": 3, "role": "grader", "name": "secret-grader-report"},
    ])
    snapshot = make_snapshot(item, item_id="case", source=SOURCE)
    serialized = json.dumps(snapshot)
    assert "secret-grader-report" not in serialized and "hidden grade" not in serialized
    assert "score=5" in serialized
    review = EvidenceReview(snapshot, tmp_path / "review")
    with pytest.raises(ValueError, match="exactly one artifact"):
        review.inspect_artifact(3)


def test_judge_contract_and_archived_verdict_keep_image_citations(
    files, tmp_path, monkeypatch, stub_claim_compiler,
):
    _with_image(files, tmp_path)

    def supported_review(*, review, **kwargs):
        review.read("detector_source")
        review.view_artifact(3)
        citation = review.cite_artifact(3, "image:1")
        contract = review.snapshot["claim_contract"]
        return json.dumps({"verdicts": [{
            "item_id": "case", "snapshot_id": review.snapshot["snapshot_id"],
            "claim_contract_id": contract["contract_id"], "scope": "local",
            "claim_assessments": [{
                "requirement_id": row["id"], "status": "supported",
                "rationale": "Observed the original output image.",
                "citations": [{"citation_id": citation["citation_id"]}],
            } for row in contract["requirements"]],
        }]})

    monkeypatch.setattr(claude_judge, "_run_review", supported_review)
    result = claude_judge.judge_behavior_with_claude(
        name="image", description="The output image has the wrong color.",
        mechanism="Produces an image with the wrong color", source=SOURCE,
        items=[{"_item_id": "case", **files.item}], assigned_labels=[True],
        cwd=tmp_path, model="test",
    )
    assert result["n_yes"] == 1
    verdict = result["verdicts"][0]
    citation = verdict["citations"][0]
    assert citation["kind"] == "image" and citation["locator"] == "image:1"
    assert citation["source_link"].endswith("/artifacts/3/download")
    assert citation["artifact_sha256"] == hashlib.sha256(files.provider.bodies[3]).hexdigest()
    archived = json.loads((Path(verdict["review_path"]) / "review-verdict.json").read_text())
    assert archived["citations"] == verdict["citations"]
    assert verdict["artifact_evidence_id"]


def test_image_content_hash_must_match_the_delivered_pixels(files, tmp_path, monkeypatch):
    review = _with_image(files, tmp_path)
    render = trace.artifact_image

    def tampered(*args, **kwargs):
        return {**render(*args, **kwargs), "sha256": "f" * 64}

    monkeypatch.setattr(trace, "artifact_image", tampered)
    with pytest.raises(ValueError, match="does not match its content hash"):
        review.view_artifact(3)
    assert not review.artifact_reads
    with pytest.raises(ValueError, match="not viewed"):
        review.cite_artifact(3, "image:1")


def test_changed_artifact_prevents_completed_judge_checkpoint_reuse(
    files, tmp_path, monkeypatch, stub_claim_compiler,
):
    calls = []

    def unknown_review(*, review, **kwargs):
        calls.append(review)
        review.read("detector_source")
        review.inspect_artifact(2)
        contract = review.snapshot["claim_contract"]
        return json.dumps({"verdicts": [{
            "item_id": "case", "snapshot_id": review.snapshot["snapshot_id"],
            "claim_contract_id": contract["contract_id"], "scope": "local",
            "claim_assessments": [{"requirement_id": row["id"], "status": "unknown",
                **({"defining_requirement_ids": [contract["defining_clauses"][0]["id"]]}
                   if row["kind"] in {"claim", "review_check"} else {}),
                                   "rationale": "Review is inconclusive.", "citations": []}
                                  for row in contract["requirements"]],
        }]})

    monkeypatch.setattr(claude_judge, "_run_review", unknown_review)
    kwargs = {"name": "test", "description": "The final file is incorrect.",
              "mechanism": "Writes an incorrect output file", "source": SOURCE,
              "items": [{"_item_id": "case", **files.item}], "assigned_labels": [True],
              "cwd": tmp_path, "model": "test"}
    first = claude_judge.judge_behavior_with_claude(**kwargs)
    assert first["n_uncertain"] == 1 and len(calls) == 1
    assert first["verdicts"][0]["artifact_evidence_id"] == calls[0].artifact_evidence_identity()
    second = claude_judge.judge_behavior_with_claude(**kwargs)
    assert second["n_uncertain"] == 1 and len(calls) == 1
    original = trace.artifact

    def new_extraction(*args, **kw):
        result = original(*args, **kw)
        result["segments"][0]["text"] += "changed extraction"
        return result

    monkeypatch.setattr(trace, "artifact", new_extraction)
    third = claude_judge.judge_behavior_with_claude(**kwargs)
    assert third["n_uncertain"] == 1 and len(calls) == 2
    assert third["verdicts"][0]["artifact_evidence_id"] != first["verdicts"][0][
        "artifact_evidence_id"
    ]
    coverage = json.loads((Path(third["verdicts"][0]["review_path"]) / "coverage.json").read_text())
    assert coverage["artifact_evidence_id"] == third["verdicts"][0]["artifact_evidence_id"]


# ZIP evidence stays bound to both the selected member and its retained archive.


@pytest.fixture
def archive(tmp_path, monkeypatch):
    entries = [
        ("first/report.txt", b"Shared evidence: first report\n"),
        ("second/report.txt", b"Shared evidence: second report\n"),
        ("first/plot.png", _png("red")),
        ("second/plot.png", _png("blue")),
    ]
    provider = ArchiveProvider(zip_bytes(entries))
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(ArtifactStore, "_provider", lambda self: provider)
    source = archive_item(provider.body, name="deliverables.zip", mime="application/zip")
    review = EvidenceReview(make_snapshot(source, item_id="case", source=SOURCE),
                            tmp_path / "review")
    return SimpleNamespace(review=review, provider=provider, item=source,
                           entries=dict(entries), tmp_path=tmp_path)


def test_archive_manifest_does_not_mark_members_as_reviewed(archive):
    review = archive.review
    manifest = review.inspect_artifact(2)
    assert manifest["format"] == "zip" and not manifest["complete"]
    assert manifest["archive"]["member_count"] == 4
    assert not review.artifact_reads
    selected = review.inspect_artifact(2, member_path="first/report.txt")
    assert selected["segments"][0]["locator"] == "line:1"
    assert "text" not in selected["segments"][0]
    assert not review.artifact_reads
    with pytest.raises(ValueError, match="not read"):
        review.cite_artifact(2, "line:1", "Shared evidence", member_path="first/report.txt")
    assert "unread or unavailable artifact" in review._whole_artifact_error()


def test_same_basenames_and_locators_do_not_share_read_or_citation_authority(archive):
    review = archive.review
    first = review.read_artifact(2, "line:1", member_path="first/report.txt")
    assert first["text"] == archive.entries["first/report.txt"].decode()
    with pytest.raises(ValueError, match="not read"):
        review.cite_artifact(2, "line:1", "Shared evidence", member_path="second/report.txt")
    cite_first = review.cite_artifact(
        2, "line:1", "Shared evidence", member_path="first/report.txt",
    )
    review.read_artifact(2, "line:1", member_path="second/report.txt")
    cite_second = review.cite_artifact(
        2, "line:1", "Shared evidence", member_path="second/report.txt",
    )
    assert cite_first["source_identity"] != cite_second["source_identity"]
    assert cite_first["artifact_sha256"] != cite_second["artifact_sha256"]
    assert cite_first["source_link"] == cite_second["source_link"] == (
        "https://api.example/v1/rollouts/run/task/0/artifacts/2/download"
    )
    assert review.validate(_verdict(review, cite_first, cite_second)) is None
    swapped = {**cite_first, "member_path": "second/report.txt"}
    assert "not issued" in review.validate(_verdict(review, swapped))
    assert review.artifact_pins[2]["sha256"] == hashlib.sha256(archive.provider.body).hexdigest()
    first_pin = review.artifact_pins[(2, "first/report.txt")]
    assert first_pin["sha256"] == hashlib.sha256(archive.entries["first/report.txt"]).hexdigest()
    assert first_pin["source_identity"]["archive_sha256"] == review.artifact_pins[2]["sha256"]
    assert archive.provider.calls == 1


def test_restore_repairs_member_cache_preserves_pins_and_requires_fresh_reads_after_compaction(
    archive,
):
    review = archive.review
    read = review.read_artifact(2, "line:1", member_path="first/report.txt")
    text = review.cite_artifact(2, "line:1", "first report", member_path="first/report.txt")
    review.view_artifact(2, member_path="second/plot.png")
    with pytest.raises(ValueError, match="not viewed"):
        review.cite_artifact(2, "image:1", member_path="first/plot.png")
    image = review.cite_artifact(2, "image:1", member_path="second/plot.png")
    verdict = _verdict(review, text, image)
    audit_before = (review.directory / "audit.jsonl").read_bytes()
    member = archive.tmp_path / "home" / "cache" / "artifacts" / "sha256" / read["artifact_sha256"]
    member.write_bytes(b"corrupt cached member")
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert member.read_bytes() == archive.entries["first/report.txt"]
    assert restored.validate(verdict) is None
    assert restored.artifact_pins == review.artifact_pins
    assert restored.artifact_evidence_identity() == review.artifact_evidence_identity()
    assert (review.directory / "audit.jsonl").read_bytes() == audit_before
    restored.compacted({"trigger": "test"})
    assert "latest compaction" in restored.validate(_verdict(restored, text, image))
    with pytest.raises(ValueError, match="latest compaction"):
        restored.cite_artifact(2, "line:1", "first report", member_path="first/report.txt")
    restored.read_artifact(2, "line:1", member_path="first/report.txt")
    restored.view_artifact(2, member_path="second/plot.png")
    assert restored.validate(_verdict(restored, text, image)) is None
    twice = EvidenceReview.restore(restored.snapshot, restored.directory)
    assert twice.validate(_verdict(twice, text, image)) is None
    assert archive.provider.calls == 1


def test_changed_outer_archive_invalidates_review_even_if_selected_member_is_unchanged(archive):
    review = archive.review
    review.read_artifact(2, "line:1", member_path="first/report.txt")
    root_digest = review.artifact_pins[2]["sha256"]
    cache = archive.tmp_path / "home" / "cache" / "artifacts" / "sha256" / root_digest
    cache.write_bytes(b"corrupt cached archive")
    archive.provider.body = zip_bytes([
        ("first/report.txt", archive.entries["first/report.txt"]),
        ("second/report.txt", b"Changed other member"),
    ])
    with pytest.raises(ValueError, match="stale|verified"):
        EvidenceReview.restore(review.snapshot, review.directory)


def test_discovery_tools_forward_selection_and_deliver_actual_member_pixels(archive, monkeypatch):
    monkeypatch.setattr(tools_artifacts, "_load_item_raw", lambda _: archive.item)
    text = tools_artifacts.read_artifact(
        "case", 2, section="segments", pointer="/0/text", member_path="first/report.txt",
    )
    assert text["ok"] and text["value"] == archive.entries["first/report.txt"].decode()
    downloaded = tools_artifacts.download_artifact("case", 2, member_path="first/report.txt")
    assert downloaded["ok"] and Path(downloaded["path"]).read_bytes() == (
        archive.entries["first/report.txt"]
    )
    result = tools_artifacts.view_artifact("case", 2, member_path="second/plot.png")
    content = tool_response(result, "view_artifact")["content"]
    assert [block["type"] for block in content] == ["text", "image"]
    metadata = json.loads(content[0]["text"])["artifact"]
    assert metadata["member_path"] == "second/plot.png"
    assert hashlib.sha256(base64.b64decode(content[1]["data"])).hexdigest() == metadata["sha256"]


# Retained input images use the same artifact evidence in discovery and review, while the
# review text replaces their encoded payloads with exact, compact references.


def _block(color="red"):
    return {"type": "input_image", "image_url": "data:image/png;base64," +
            base64.b64encode(_png(color)).decode()}


@pytest.fixture
def chart():
    """A real-sized chart whose encoded payload dwarfs its compact reference."""
    image = pytest.importorskip("PIL.Image")
    with io.BytesIO() as stream:
        image.frombytes("RGB", (320, 256), random.Random(7).randbytes(320 * 256 * 3)).save(
            stream, format="PNG",
        )
        data = stream.getvalue()
    return {"type": "input_image", "image_url": "data:image/png;base64," +
            base64.b64encode(data).decode()}


def _chart_item(block=None):
    messages = [{"role": "user", "content": [
        {"type": "input_text", "text": "🙂 Which bar is highest?"}, block or _block(),
        {"type": "text", "text": "Explain the chart carefully."},
    ]}]
    # CharXiv retains the request inside a JSON response envelope; its message
    # content can itself be encoded. Some imports also preserve duplicate pools.
    return {"response_text": json.dumps({
        "agent_request": {"messages": json.dumps(messages)},
        "initial_request": {"messages": messages},
        "model_call_evidence": {"messages": messages, "calls": []},
        "steps": [{"source": "assistant", "message": ANSWER}],
    })}


def _chart_review(tmp_path, item=None):
    return EvidenceReview(make_snapshot(item or _chart_item(), item_id="chart", source=SOURCE),
                          tmp_path / "review")


def test_charxiv_nested_images_are_deduplicated_in_all_catalog_surfaces():
    item = _chart_item()
    catalog = trace.artifacts(item)
    assert len(catalog) == 1
    assert catalog == trace.record(item)["artifacts"]
    row = catalog[0]
    assert row["source_path"] == ["agent_request", "messages", 0, "content", 1]
    assert row["sha256"] == hashlib.sha256(_png()).hexdigest()
    assert "base64" not in json.dumps(row)
    coverage = trace.coverage(item)["artifacts"]
    assert coverage["retained_input_images"] == 1
    assert coverage["input_image_inventory_complete"] is True
    # Retained input content does not establish that external files were imported.
    assert coverage["status"] == "partial"


@pytest.mark.parametrize("limit", [64, 1])
def test_image_indices_do_not_change_when_snapshot_adds_derived_input_messages(monkeypatch, limit):
    monkeypatch.setattr(inline_images, "MAX_IMAGES", limit)
    item = {"response_text": json.dumps({
        "agent_request": {"messages": [{"role": "user", "content": [_block("red")]}]},
        "initial_request": {"messages": [{"role": "user", "content": [_block("blue")]}]},
        "execution": {"artifacts": [{"artifact_idx": 7, "name": "existing.txt"}]},
        "steps": [],
    })}
    raw = trace.artifacts(item)
    frozen = make_snapshot(item, item_id="chart", source=SOURCE)["artifact_manifest"]["catalog"]
    raw_identities = [(row["artifact_idx"], row.get("sha256")) for row in raw]
    frozen_identities = [(row["artifact_idx"], row.get("sha256")) for row in frozen]
    assert raw_identities == frozen_identities
    assert [index for index, _ in raw_identities] == list(range(7, 8 + limit))[:len(raw)]


def test_oversized_provenance_is_explicitly_unknown_and_never_decoded(monkeypatch):
    original = inline_images.decode_container

    def bounded_decoding(value):
        assert not isinstance(value, str) or len(value) <= 128
        return original(value)

    monkeypatch.setattr(inline_images, "MAX_SOURCE_CHARS", 128)
    monkeypatch.setattr(inline_images, "decode_container", bounded_decoding)
    item = _chart_item()
    item["input_json"] = json.dumps({"response_is_literal": True, "padding": "x" * 4096})
    catalog, limitations = inline_images.image_inventory(item)
    assert catalog == []
    assert any("provenance input_json exceeds" in message for message in limitations)


@pytest.mark.parametrize("surface,provider", [
    ("input_messages", "openai"), ("initial_request", "anthropic"),
])
def test_provider_and_normalized_input_surfaces(surface, provider):
    block = _block()
    if provider == "anthropic":
        block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                 "data": block["image_url"].split(",", 1)[1]}}
    messages = [{"role": "user", "content": [block]}]
    item = ({"input_messages": json.dumps(messages)} if surface == "input_messages" else
            {"response_text": json.dumps({surface: {"messages": messages}})})
    assert len(trace.artifacts(item)) == 1
    assert trace.artifact_image(item, 0)["status"] == "available"


def test_evaluator_model_and_quoted_images_are_not_input_artifacts():
    block = _block()
    item = {"response_text": json.dumps({
        "grader": {"messages": [{"role": "user", "content": [block]}]},
        "steps": [{"role": "assistant", "content": [block]},
                  {"role": "tool", "content": [block]}],
        "agent_request": {"messages": [{"role": "user", "content": [
            {"type": "text", "text": json.dumps(block)},
        ]}]},
    }), "grader_text": json.dumps(block)}
    assert trace.artifacts(item) == []
    text = json.dumps(block)
    assert trace.artifacts({"prompt": text}) == []
    assert trace.artifacts({"input_messages": [{"role": "user", "content": text}]}) == []
    assert trace.artifacts({"input_messages": [{"role": "user", "content": [text]}]}) == []


@pytest.mark.parametrize("marker", ["input_json", "input_parse"])
def test_declared_literal_responses_do_not_become_input_images(tmp_path, marker):
    from autofa.data.input_view import COMPACT_INPUT_ORIGIN

    item = _chart_item()
    item[marker] = json.dumps({
        "autofa_input_origin" if marker == "input_json" else "response_origin":
            COMPACT_INPUT_ORIGIN,
        "response_is_literal": True,
        "prompt_text": "Explain this JSON structure",
    })
    assert trace.artifacts(item) == []
    assert _chart_review(tmp_path, item).list_artifacts()["artifacts"] == []


def test_source_artifact_metadata_cannot_authorize_excluded_input_images():
    item = _chart_item()
    descriptor = trace.artifacts(item)[0]
    envelope = json.loads(item["response_text"])
    envelope["agent_request"]["messages"] = [{"role": "assistant", "content": [_block()]}]
    envelope.pop("initial_request")
    envelope.pop("model_call_evidence")
    # A fabricated source descriptor points at the assistant's own generated image.
    descriptor["source_path"] = ["agent_request", "messages", 0, "content", 0]
    envelope["execution"] = {"artifacts": [descriptor]}
    item["response_text"] = json.dumps(envelope)
    result = ensure_artifact(item, 0)
    assert result["status"] == "unavailable"
    assert "not in the input inventory" in result["reason"]


def test_original_bytes_and_rendered_pixels_have_distinct_verified_hashes():
    item = _chart_item()
    saved = ensure_artifact(item, 0)
    assert saved["verified"] is True
    assert Path(saved["path"]).read_bytes() == _png()
    with capture_artifact_reads() as receipts:
        image = trace.artifact_image(item, 0)
    assert image["status"] == "available"
    assert image["artifact_sha256"] == hashlib.sha256(_png()).hexdigest()
    pixels = base64.b64decode(image["data_url"].split(",", 1)[1])
    assert image["sha256"] == hashlib.sha256(pixels).hexdigest()
    assert image["source_identity"]["kind"] == "retained_input_image"
    assert validate_artifact_reads(item, receipts)
    assert not validate_artifact_reads(_chart_item(_block("blue")), receipts)
    Path(saved["path"]).write_bytes(b"corrupt source cache")
    assert validate_artifact_reads(item, receipts)  # Restore only from retained, verified bytes.


def test_text_reads_never_count_as_visual_review_and_citations_survive_restoration(tmp_path):
    review = _chart_review(tmp_path)
    for field in review.fields:
        assert review.read(field)["is_final"]
    verdict = _verdict(review, review.cite("response_text", ANSWER)["citations"][0],
                       scope="whole_trace")
    assert "unread or unavailable artifact" in review.validate(verdict)
    row = review.list_artifacts()["artifacts"][0]
    index = row["artifact_idx"]
    inspection = review.inspect_artifact(index)
    assert inspection["requires_visual_review"] and not inspection["visual_complete"]
    assert not review.artifact_reads
    with pytest.raises(ValueError, match="not viewed"):
        review.cite_artifact(index, "image:1")
    assert "artifact visuals" in review.validate(verdict)
    image = review.view_artifact(index)
    assert image["status"] == "available"
    citation = review.cite_artifact(index, "image:1")
    assert citation["artifact_sha256"] == hashlib.sha256(_png()).hexdigest()
    assert image["receipt"]["source_identity"]["source_field"] == row["source_field"]
    assert image["receipt"]["source_identity"]["source_path"] == row["source_path"]
    verdict["citations"].append(citation)
    assert review.validate(verdict) is None
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.validate(verdict) is None
    assert restored.citations[citation["citation_id"]] == citation
    review.compacted({"trigger": "test"})
    assert "latest compaction" in review.validate(verdict)


@pytest.mark.parametrize("url,reason", [
    ("https://example.invalid/chart.png", "not fetched"),
    ("file:///etc/passwd", "not fetched"),
    ("data:image/png;base64,!!!", "invalid base64"),
    ("data:image/png,raw", "not a base64"),
])
def test_unavailable_images_remain_visible_without_network_files_or_review_receipts(
    tmp_path, url, reason,
):
    review = _chart_review(tmp_path, _chart_item({"type": "input_image", "image_url": url}))
    descriptor = review.list_artifacts()["artifacts"][0]
    assert descriptor["status"] == "unavailable"
    result = review.view_artifact(descriptor["artifact_idx"])
    assert result["status"] == "unavailable" and reason in result["reason"]
    assert not review.artifact_reads
    assert "unavailable artifact" in review._whole_artifact_error()


def test_invalid_image_content_and_out_of_range_pages_do_not_issue_receipts(tmp_path):
    review = _chart_review(tmp_path)
    assert review.view_artifact(0, page=2)["status"] != "available"
    assert not review.artifact_reads
    invalid = _chart_item({"type": "input_image", "image_url": "data:image/png;base64," +
                           base64.b64encode(b"not an image").decode()})
    review = EvidenceReview(make_snapshot(invalid, item_id="chart", source=SOURCE),
                            tmp_path / "invalid-review")
    assert review.view_artifact(0)["status"] != "available"
    assert not review.artifact_reads


def test_bounded_inventory_reports_gaps_and_blocks_whole_trace_receipts(tmp_path, monkeypatch):
    item = {"input_messages": [{"role": "user", "content": [_block(), _block("blue")]}]}
    monkeypatch.setattr(inline_images, "MAX_IMAGES", 1)
    assert len(trace.artifacts(item)) == 1
    assert trace.coverage(item)["artifacts"]["input_image_inventory_complete"] is False
    review = _chart_review(tmp_path, item)
    review.inspect_artifact(0)
    review.view_artifact(0)
    assert review._whole_artifact_error() == (
        "whole-trace verdict has an incomplete retained-image inventory"
    )


@pytest.mark.parametrize("limit,value", [("MAX_IMAGE_BYTES", 8), ("MAX_SOURCE_CHARS", 10),
                                        ("MAX_NODES", 1), ("MAX_DEPTH", 1)])
def test_extraction_limits_are_explicit(limit, value, monkeypatch):
    monkeypatch.setattr(inline_images, limit, value)
    catalog, limitations = inline_images.image_inventory(_chart_item())
    assert limitations or any(row.get("status") == "unavailable" for row in catalog)


def test_image_snapshot_changes_without_invalidating_text_only_projection_policy(tmp_path):
    plain = make_snapshot({"response_text": "plain evidence"}, item_id="chart", source=SOURCE)
    assert "artifact_manifest" not in plain
    first = make_snapshot(_chart_item(), item_id="chart", source=SOURCE)
    second = make_snapshot(_chart_item(_block("blue")), item_id="chart", source=SOURCE)
    assert first["snapshot_id"] != second["snapshot_id"]
    assert first["projection_hash"] == plain["projection_hash"]
    review = EvidenceReview(first, tmp_path / "review")
    context = json.loads(review.initial_context(0))
    assert "retains input images" in context["instructions"]
    modified = deepcopy(first)
    modified["artifact_manifest"]["catalog"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="saved evidence does not match"):
        EvidenceReview.restore(modified, review.directory)


def test_real_shaped_nested_and_duplicate_image_payloads_are_short_references(chart, tmp_path):
    item = _chart_item(chart)
    original = deepcopy(item)
    review = _chart_review(tmp_path, item)
    assert item == original
    encoded = chart["image_url"].split(",", 1)[1]
    assert len(encoded) > 300_000
    assert all(encoded not in value for value in review.fields.values())
    assert all(len(value) < 16_000 for value in review.fields.values())
    assert review.fields["response_text"].count("artifact://0/sha256/") == 3
    assert "artifact://0/sha256/" in review.fields["input_messages"]
    assert ANSWER in review.fields["response_text"]
    assert "Explain the chart carefully." in review.fields["input_messages"]
    manifest = review.manifest()
    assert manifest["manifest"]["response_text"]["origin"] == "stored"
    assert "artifact_source_fields" not in manifest
    assert review.artifact_source_fields["response_text"] == original["response_text"]
    assert len(manifest["review_projection"]["fields"]["response_text"]["replacements"]) == 3
    context = review.initial_context(0)
    assert encoded not in context
    assert "artifact://" not in json.dumps(manifest["artifact_manifest"])


def test_direct_image_view_completes_visual_coverage_without_separate_inventory_call(tmp_path):
    review = _chart_review(tmp_path)
    for field in review.fields:
        review.read(field)
    assert not review.artifact_inspections
    assert review.view_artifact(0)["status"] == "available"
    verdict = _verdict(review, review.cite_artifact(0, "image:1"), scope="whole_trace")
    assert review.validate(verdict) is None
    assert review.artifact_inspections[0]["requires_visual_review"]
    assert len(review.artifact_reads) == 1 and review.artifact_reads[0]["kind"] == "image"


def test_implicit_visual_inventory_does_not_grant_document_text_read_credit(
    tmp_path, monkeypatch,
):
    inspect = trace.artifact

    def with_document_segment(*args, **kwargs):
        return {**inspect(*args, **kwargs), "segments": [
            {"locator": "document:1", "text": "Retained text outside the visual page."},
        ]}

    monkeypatch.setattr(trace, "artifact", with_document_segment)
    review = _chart_review(tmp_path)
    review.view_artifact(0)
    assert review._whole_artifact_error() == "whole-trace verdict has unread artifact 0:document:1"
    assert all(row["kind"] == "image" for row in review.artifact_reads)
    review.read_artifact(0, "document:1")
    assert review._whole_artifact_error() is None


def test_projection_maps_and_localisation_keep_selected_original_occurrence(chart, tmp_path):
    item = _chart_item(chart)
    original = item["response_text"]
    start = original.rindex(ANSWER)
    localised = {"source_sha256": text_hash(SOURCE), "passages": [
        passage("response_text", original, start, start + len(ANSWER), method="literal"),
    ]}
    snapshot = make_snapshot(item, item_id="chart", source=SOURCE,
                             failure_localisation=localised)
    selected, = snapshot["failure_localisation"]["passages"]
    projected = snapshot["fields"]["response_text"]
    assert selected["start"] == projected.rindex(ANSWER) < start
    mapping = snapshot["review_projection"]["fields"]["response_text"]
    assert source_span(mapping, selected["start"], selected["end"]) == (
        start, start + len(ANSWER),
    )
    # Every non-binary character survives exactly, including escaped nested JSON.
    old_offset = new_offset = 0
    for row in mapping["replacements"]:
        assert original[old_offset:row["source_start"]] == projected[new_offset:row["start"]]
        assert source_span(mapping, row["start"], row["end"]) is None
        old_offset, new_offset = row["source_end"], row["end"]
    assert original[old_offset:] == projected[new_offset:]
    review = EvidenceReview(snapshot, tmp_path / "review")
    review.read("response_text")
    citation, = review.cite("response_text", ANSWER)["citations"]
    assert citation["source_span"] == {"field": "response_text", "start": start,
                                        "end": start + len(ANSWER),
                                        "field_sha256": text_hash(original)}
    navigation = interval_navigation(snapshot)
    assert navigation["available"] and navigation["event_count"] == 1
    span, = navigation["events"][0]["ranges"]
    assert span["start"] <= citation["start"] < citation["end"] <= span["end"]
    assert ANSWER in projected[span["start"]:span["end"]]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_only_real_payload_slots_change_not_plain_prose_or_tool_arguments(provider):
    data = base64.b64encode(b"retained image bytes").decode()
    block = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}
    if provider == "anthropic":
        block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                               "data": data}}
    quoted = json.dumps(block)
    item = {"input_messages": [{"role": "user", "content": [
        {"type": "text", "text": quoted}, quoted, block,
    ]}], "response_text": json.dumps({"steps": [
        {"role": "assistant", "content": [block], "arguments": quoted},
    ]}), "prompt": quoted, "reasoning_text": quoted}
    snapshot = make_snapshot(item, item_id="chart", source=SOURCE)
    messages = json.loads(snapshot["fields"]["input_messages"])
    assert messages[0]["content"][:2] == item["input_messages"][0]["content"][:2]
    assert data not in json.dumps(messages[0]["content"][2])
    for field in ("response_text", "prompt", "reasoning_text"):
        assert snapshot["fields"][field] == item[field]


@pytest.mark.parametrize("value", ["plain words base64 image_url data:image/png;base64,abcd",
                                    '{"type":"input_image","image_url":"INVALID"}'])
def test_normal_text_and_unavailable_payloads_are_unchanged(value):
    item = {"response_text": value, "input_messages": [{"role": "user", "content": value}]}
    snapshot = make_snapshot(item, item_id="chart", source=SOURCE)
    assert snapshot["fields"]["response_text"] == value
    assert "review_projection" not in snapshot


def test_replacement_context_requires_fresh_text_and_pixels_and_restores_notes(tmp_path):
    review = _chart_review(tmp_path)
    for field in review.fields:
        review.read(field)
    review.inspect_artifact(0)
    review.view_artifact(0)
    citation = review.cite("response_text", ANSWER)["citations"][0]
    review.save_notes("Original image suggested blue; verify it again.")
    review.start_replacement_context({"reason": "context exhausted"})
    assert not review.reads and not review.artifact_reads
    assert review.citations[citation["citation_id"]] == citation
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.notes == review.notes
    assert not restored.reads and not restored.artifact_reads
    assert restored.epoch == 1
    for field in restored.fields:
        restored.read(field)
    verdict = _verdict(restored, citation, scope="whole_trace")
    assert "artifact visuals" in restored.validate(verdict)
    restored.view_artifact(0)
    assert restored.validate(verdict) is None


def test_artifact_references_are_navigation_not_textual_image_citations(tmp_path):
    review = _chart_review(tmp_path)
    review.read("response_text")
    with pytest.raises(ValueError, match="references are navigation"):
        review.cite("response_text", "artifact://0/")
    match = review.search("response_text", "artifact://0/")["matches"][0]
    assert "citation" not in match and "navigation" in match["citation_unavailable"]
    verdict = _verdict(review, {"field": "response_text", "start": match["match_start"],
                                "end": match["match_end"], "quote": "artifact://0/"})
    assert "references are navigation" in review.validate(verdict)


def test_literal_response_is_preserved_even_when_same_image_has_an_input_artifact():
    from autofa.data.input_view import COMPACT_INPUT_ORIGIN

    block = _block()
    item = {"response_text": json.dumps(block),
            "input_messages": [{"role": "user", "content": [block]}],
            "input_parse": {"response_origin": COMPACT_INPUT_ORIGIN,
                            "response_is_literal": True}}
    snapshot = make_snapshot(item, item_id="chart", source=SOURCE)
    assert snapshot["fields"]["response_text"] == item["response_text"]
    assert "artifact://0/" in snapshot["fields"]["input_messages"]


# Long artifact preparation must not block judge transports or race receipts.


async def _wait_for(event: threading.Event):
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    pytest.fail("tool worker did not reach expected checkpoint")


def test_async_evidence_installer_leaves_event_loop_responsive_and_cancels():
    started = threading.Event()
    finished = threading.Event()

    def install():
        # A long setup step, such as a speech model download, polls for cancellation.
        started.set()
        try:
            for _ in range(6000):
                check_cancelled()
                time.sleep(0.01)
        finally:
            finished.set()

    async def exercise():
        task = asyncio.create_task(claude_judge._async_evidence_tool_result(
            "prepare_artifact_support", install, {}, lock=asyncio.Lock(),
        ))
        await _wait_for(started)
        # This coroutine must run while the synchronous installer is still busy.
        assert not finished.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        assert finished.is_set()

    asyncio.run(exercise())


def test_evidence_cancellation_drains_before_next_review_mutation():
    started, cleaning, release, second_started = (threading.Event() for _ in range(4))
    parent = Cancellation()

    def first():
        started.set()
        try:
            while True:
                check_cancelled()
                threading.Event().wait(0.01)
        finally:
            cleaning.set()
            assert release.wait(timeout=3)

    def second():
        second_started.set()
        return {"ok": True}

    async def exercise():
        lock = asyncio.Lock()
        with cancellation_scope(parent):
            task = asyncio.create_task(claude_judge._async_evidence_tool_result(
                "transcribe_artifact", first, {}, lock=lock,
            ))
        await _wait_for(started)
        task.cancel()
        await _wait_for(cleaning)
        following = asyncio.create_task(claude_judge._async_evidence_tool_result(
            "cite_artifact", second, {}, lock=lock,
        ))
        task.cancel()  # Repeated cancellation must not bypass worker draining.
        await asyncio.sleep(0.05)
        assert not second_started.is_set()
        assert not parent.event.is_set()  # Only the interrupted tool gets cancelled.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await following
        assert json.loads(result["content"][0]["text"]) == {"ok": True}

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_parent_review_cancellation_reaches_artifact_worker():
    started, stopped = threading.Event(), threading.Event()
    parent = Cancellation()

    def work():
        started.set()
        try:
            while True:
                check_cancelled()
                threading.Event().wait(0.01)
        finally:
            stopped.set()

    async def exercise():
        with cancellation_scope(parent):
            task = asyncio.create_task(claude_judge._async_evidence_tool_result(
                "transcribe_artifact", work, {}, lock=asyncio.Lock(),
            ))
        await _wait_for(started)
        await asyncio.to_thread(parent.cancel)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        assert stopped.is_set()

    asyncio.run(exercise())


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_harness_serializes_evidence_tools_and_preserves_images(
    harness, monkeypatch, tmp_path,
):
    started, release, image_started = (threading.Event() for _ in range(3))

    def prepare():
        started.set()
        assert release.wait(timeout=3)
        return {"status": "available"}

    def view():
        image_started.set()
        return {"status": "available", "data_url": "data:image/png;base64,aW1hZ2U="}

    def invalid():
        raise ValueError("artifact is not available")

    specs = [
        ("prepare_artifact_support", "prepare", {}, prepare),
        ("view_artifact", "view", {}, view),
        ("invalid", "invalid", {}, invalid),
    ]
    monkeypatch.setattr(claude_judge, "_evidence_tool_specs", lambda _: specs)

    async def exercise(call):
        first = asyncio.create_task(call("prepare_artifact_support", {}))
        await _wait_for(started)
        second = asyncio.create_task(call("view_artifact", {}))
        await asyncio.sleep(0.05)
        assert not image_started.is_set()
        release.set()
        await first
        result = await second
        failed = await call("invalid", {})
        if harness == "claude":
            assert [block["type"] for block in result["content"]] == ["text", "image"]
            assert result["content"][1] == {
                "type": "image", "data": "aW1hZ2U=", "mimeType": "image/png",
            }
            assert failed["isError"]
        else:
            assert result["success"]
            assert result["contentItems"][1] == {
                "type": "inputImage", "imageUrl": "data:image/png;base64,aW1hZ2U=",
            }
            assert failed["success"] is False
        assert "artifact is not available" in json.dumps(failed)

    try:
        if harness == "claude":
            import claude_agent_sdk as sdk

            handlers = {}

            def server(**kwargs):
                handlers.update({tool.name: tool.handler for tool in kwargs["tools"]})
                return {}

            monkeypatch.setattr(sdk, "create_sdk_mcp_server", server)
            claude_judge._evidence_server(SimpleNamespace())

            async def call(name, args):
                return await handlers[name](args)

            asyncio.run(exercise(call))
        else:
            async def run(**kwargs):
                await exercise(kwargs["call_tool"])
                raise RuntimeError("tool transport verified")

            monkeypatch.setattr(codex_judge, "_run_turn", run)
            evidence = EvidenceReview(make_snapshot(
                {"response_text": "Retained evidence"}, item_id="case",
                source="def detect(item): return True",
            ), tmp_path / "review")
            with pytest.raises(RuntimeError, match="tool transport verified"):
                asyncio.run(codex_judge.run_codex_judge_session(
                    prompt="Review", cwd=tmp_path, model="test", cli_path=None,
                    review=evidence, system_prompt="Test review",
                ))
    finally:
        release.set()
