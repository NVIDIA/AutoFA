# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real documents, images and videos exercise extraction, rendering, review and dependency setup."""

import base64
import hashlib
import io
import json
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from autofa import trace
from autofa.data import artifact_audio
from autofa.data import artifact_content as content
from autofa.data import artifact_dependencies as dependencies
from autofa.data import artifact_video as video
from autofa.data.artifact_access import capture_artifact_reads, validate_artifact_reads
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot

from support.traces import item, serve

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def package(path, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Default Extension="xml" '
            'ContentType="application/xml"/></Types>',
        )
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def document(path, text="Document tail"):
    return package(
        path,
        {
            "word/document.xml": f'<w:document xmlns:w="{W}"><w:body>'
            f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
            "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell A</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>Cell B</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
            "</w:body></w:document>",
            "word/header1.xml": f'<w:hdr xmlns:w="{W}"><w:p><w:r><w:t>Header</w:t>'
            "</w:r></w:p></w:hdr>",
        },
    )


def pdf(path):
    stream = b"BT /F1 12 Tf 20 60 Td (Fixture PDF text) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data, offsets = b"%PDF-1.4\n", [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(data)
    data += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    data += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    path.write_bytes(data)
    return path


def workbook(path):
    return package(
        path,
        {
            "xl/workbook.xml": f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets>'
            '<sheet name="Budget 2026" sheetId="7" state="hidden" r:id="sheet1"/>'
            "</sheets></workbook>",
            "xl/_rels/workbook.xml.rels": f'<Relationships xmlns="{PKG}">'
            f'<Relationship Id="sheet1" Type="{R}/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
            "xl/sharedStrings.xml": f'<sst xmlns="{S}"><si><t>Revenue λ</t></si></sst>',
            "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{S}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>10</v></c></row>'
            '<row r="2"><c r="B2"><f>SUM(B1,5)</f><v>15</v></c>'
            '<c r="C2"><f>NOW()</f></c><c r="D2" t="inlineStr"><is><t>Final</t></is></c></row>'
            "</sheetData></worksheet>",
        },
    )


def presentation(path):
    return package(
        path,
        {
            "ppt/presentation.xml": f'<p:presentation xmlns:p="{P}" xmlns:r="{R}">'
            '<p:sldIdLst><p:sldId id="256" r:id="slide1"/></p:sldIdLst></p:presentation>',
            "ppt/_rels/presentation.xml.rels": f'<Relationships xmlns="{PKG}">'
            f'<Relationship Id="slide1" Type="{R}/slide" Target="slides/slide1.xml"/>'
            "</Relationships>",
            "ppt/slides/slide1.xml": f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld>'
            "<p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Quarterly revenue</a:t></a:r></a:p>"
            "<a:p><a:r><a:t>Second paragraph</a:t></a:r></a:p></p:txBody></p:sp></p:spTree>"
            "</p:cSld></p:sld>",
            "ppt/slides/_rels/slide1.xml.rels": f'<Relationships xmlns="{PKG}">'
            f'<Relationship Id="notes1" Type="{R}/notesSlide" '
            'Target="../notesSlides/notesSlide1.xml"/>'
            "</Relationships>",
            "ppt/notesSlides/notesSlide1.xml": f'<p:notes xmlns:p="{P}" xmlns:a="{A}">'
            "<a:p><a:r><a:t>Presenter note</a:t></a:r></a:p></p:notes>",
        },
    )


def png(path):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(b"\0\xff\0\0\0\xff\0"))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)
    return path


def inspect(path, tmp_path, **metadata):
    return content.inspect_artifact(path, metadata, output_dir=tmp_path / "derived")


def render(path, tmp_path, metadata=None, **options):
    return content.render_artifact(path, metadata, output_dir=tmp_path / "render", **options)


def test_document_text_is_untruncated_and_missing_office_renderer_is_explicit(
    tmp_path, monkeypatch,
):
    text = "λ" * 80_000 + " COMPLETE_TAIL"
    path = document(tmp_path / "report.docx", text)
    original = path.read_bytes()
    result = inspect(path, tmp_path)
    assert result["status"] == "available" and result["complete"]
    segments = {segment["locator"]: segment for segment in result["segments"]}
    paragraph = segments["word/document.xml#paragraph:1"]
    assert paragraph["text"] == text
    assert paragraph["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert segments["word/document.xml#table:1/row:1/cell:2"]["text"] == "Cell B"
    assert segments["word/header1.xml#paragraph:1"]["text"] == "Header"
    assert result["requires_visual_review"] and not result["visual_complete"]
    assert result["visual_pages"] is None
    assert path.read_bytes() == original
    monkeypatch.setattr(content, "office_executable", lambda: None)
    rendered = render(path, tmp_path)
    assert rendered["status"] == "unavailable"
    limitations = " ".join(rendered["limitations"])
    assert "LibreOffice" in limitations and dependencies.OFFICE_SETUP in limitations
    assert "install=True" not in limitations


def test_spreadsheet_coordinates_hidden_sheet_formula_and_cached_values(tmp_path):
    path = workbook(tmp_path / "budget.xlsx")
    result = inspect(path, tmp_path)
    assert result["status"] == "available" and result["complete"]
    cells = {segment["locator"]: segment for segment in result["segments"]}
    assert cells["sheet:Budget%202026!A1"]["text"] == "Revenue λ"
    formula = cells["sheet:Budget%202026!B2"]
    assert formula["formula"] == "SUM(B1,5)" and formula["cached_value"] == "15"
    assert formula["location"]["sheet_state"] == "hidden"
    assert cells["sheet:Budget%202026!C2"]["cached_value"] is None
    assert cells["sheet:Budget%202026!C2"]["formula"] == "NOW()"


def test_presentation_keeps_slide_ids_text_and_speaker_notes(tmp_path):
    result = inspect(presentation(tmp_path / "deck.pptx"), tmp_path)
    assert result["status"] == "available" and result["complete"]
    assert result["visual_pages"] == 1
    slide, notes = result["segments"]
    assert slide["locator"] == "slide:1"
    assert slide["location"]["slide_id"] == "256"
    assert slide["text"] == "Quarterly revenue\nSecond paragraph"
    assert notes["locator"] == "slide:1/notes" and notes["text"] == "Presenter note"


def test_text_and_csv_keep_all_content_and_stable_coordinates(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("first\nsecond λ\n", encoding="utf-8")
    result = inspect(path, tmp_path)
    assert "".join(segment["text"] for segment in result["segments"]) == path.read_text()
    assert result["segments"][1]["locator"] == "line:2"
    assert not result["requires_visual_review"]
    csv = tmp_path / "data.csv"
    large = "λ" * 200_000
    csv.write_text('name,value\n"multiline\nname","' + large + '"\n')
    result = inspect(csv, tmp_path)
    assert result["status"] == "available" and result["complete"]
    assert result["segments"][2]["text"] == "multiline\nname"
    assert result["segments"][3]["text"] == large
    assert result["segments"][3]["locator"] == "row:2/column:2"


def test_limits_are_explicit_and_retained_segments_are_not_claimed_complete(tmp_path, monkeypatch):
    path = tmp_path / "lines.txt"
    path.write_text("one\ntwo\n")
    monkeypatch.setattr(content, "MAX_SEGMENTS", 1)
    result = inspect(path, tmp_path)
    assert result["status"] == "error" and not result["complete"]
    assert len(result["segments"]) == 1
    # Decompressed Office members are bounded separately from the file size (zip bombs).
    monkeypatch.setattr(content, "MAX_ARCHIVE_BYTES", 1)
    office = inspect(document(tmp_path / "large.docx"), tmp_path)
    assert office["status"] == "error" and not office["complete"]
    monkeypatch.setattr(content, "MAX_FILE_BYTES", 1)
    assert not inspect(path, tmp_path)["complete"]


@pytest.mark.parametrize("mode", ["duplicate", "traversal", "entity", "encrypted"])
def test_malformed_or_hostile_office_packages_fail_explicitly(tmp_path, mode):
    path = tmp_path / "bad.docx"
    if mode == "duplicate":
        # Both copies are well-formed, so only the duplicate-member guard can reject it.
        document(path)
        with zipfile.ZipFile(path) as archive:
            body = archive.read("word/document.xml")
        with pytest.warns(UserWarning), zipfile.ZipFile(path, "a") as archive:
            archive.writestr("word/document.xml", body)
    elif mode == "traversal":
        package(path, {"../outside": "data"})
    elif mode == "entity":
        package(path, {"word/document.xml": '<!DOCTYPE d [<!ENTITY x "expanded">]><d>&x;</d>'})
    else:
        path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"encrypted")
    result = inspect(path, tmp_path)
    assert result["status"] == "error" and not result["complete"]
    assert result["limitations"]
    assert not (tmp_path / "outside").exists()


def test_missing_renderer_dependency_is_distinct_from_absent_content(tmp_path, monkeypatch):
    path = png(tmp_path / "image.png")
    original_import = content.importlib.import_module

    def missing(name):
        if name.startswith("PIL") or name == "pypdfium2":
            raise ImportError("Optional rendering dependency missing")
        return original_import(name)

    monkeypatch.setattr(content.importlib, "import_module", missing)
    assert inspect(path, tmp_path)["status"] == "unavailable"
    rendered = render(path, tmp_path)
    assert rendered["status"] == "unavailable" and not rendered["complete"]
    assert "path" not in rendered
    assert rendered["original_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_office_renderer_refuses_macros_and_external_content_before_subprocess(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(content, "office_executable", lambda: "/mock/libreoffice")
    monkeypatch.setattr(content.subprocess, "run", lambda *a, **k: pytest.fail("must not execute"))
    for name, extra in [
        ("macro", {"word/vbaProject.bin": b"macro bytes"}),
        (
            "external",
            {
                "word/_rels/document.xml.rels": f'<Relationships xmlns="{PKG}">'
                f'<Relationship Id="image" Type="{R}/image" TargetMode="External" '
                'Target="https://external.test/image.png"/></Relationships>'
            },
        ),
    ]:
        path = document(tmp_path / f"{name}.docx")
        with zipfile.ZipFile(path, "a") as archive:
            for part, value in extra.items():
                archive.writestr(part, value)
        result = render(path, tmp_path)
        assert result["status"] == "error" and not result["complete"]


def test_office_renderer_refuses_active_external_formulas(tmp_path, monkeypatch):
    monkeypatch.setattr(content, "office_executable", lambda: "/mock/libreoffice")
    monkeypatch.setattr(content.subprocess, "run", lambda *a, **k: pytest.fail("must not execute"))
    path = package(tmp_path / "external.xlsx", {
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{S}"><c r="A1">'
        '<f>WEBSERVICE("https://external.test")</f><v>cached</v></c></worksheet>',
    })
    result = render(path, tmp_path)
    assert result["status"] == "error" and not result["complete"]


def test_unsupported_and_invalid_encoding_are_explicit(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"\xff\x80")
    assert inspect(path, tmp_path)["status"] == "unsupported"
    assert inspect(path, tmp_path, name="data.txt")["status"] == "error"


def test_real_png_render_has_verified_bytes_and_complete_frame(tmp_path):
    path = png(tmp_path / "image.png")
    result = inspect(path, tmp_path)
    assert result["image"] == {"width": 2, "height": 1, "mode": "RGB"}
    rendered = render(path, tmp_path)
    assert rendered["status"] == "available" and rendered["complete"]
    assert rendered["locator"] == "image:1" and rendered["page_count"] == 1
    assert hashlib.sha256(Path(rendered["path"]).read_bytes()).hexdigest() == rendered["sha256"]
    assert render(path, tmp_path, page=2)["status"] == "error"


def test_image_render_respects_recorded_exif_orientation(tmp_path):
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (3, 2), "red")
    orientation = Image.Exif()
    orientation[274] = 6
    image.save(path, exif=orientation)
    image.close()
    rendered = render(path, tmp_path)
    assert rendered["status"] == "available"
    assert (rendered["width"], rendered["height"]) == (2, 3)


def test_real_pdf_text_and_page_render_have_matching_locators(tmp_path):
    path = pdf(tmp_path / "report.pdf")
    result = inspect(path, tmp_path)
    assert result["status"] == "available" and result["complete"]
    assert "Fixture PDF text" in result["segments"][0]["text"]
    assert result["requires_visual_review"] and not result["visual_complete"]
    rendered = render(path, tmp_path)
    assert rendered["status"] == "available" and rendered["complete"]
    assert rendered["locator"] == result["segments"][0]["locator"] == "page:1"
    assert rendered["original_sha256"] == result["sha256"]
    assert render(path, tmp_path, page=0)["status"] == "error"


def test_reader_installed_mid_review_preserves_native_text_and_restores_receipts(
    tmp_path, monkeypatch,
):
    # The user can install a missing renderer mid-review without losing prior reads.
    body = document(tmp_path / "original.docx").read_bytes()
    serve(monkeypatch, tmp_path / "home", body)
    source = item(body, name="original.docx", mime="application/octet-stream")
    installed = False

    def office_pdf(path, kind, output):
        if not installed:
            raise ImportError("LibreOffice unavailable")
        return pdf(output / "converted.pdf")

    def prepare(capability, *, install=False, model="base"):
        assert capability == "office" and not install
        return {"status": "available" if installed else "unavailable"}

    monkeypatch.setattr(content, "_office_pdf", office_pdf)
    monkeypatch.setattr(dependencies, "prepare_artifact_support", prepare)
    snapshot = make_snapshot(source, item_id="case", source="def detect(x): return True")
    review = EvidenceReview(snapshot, tmp_path / "review")
    inspection = review.inspect_artifact(2)
    locator = inspection["segments"][0]["locator"]
    read = review.read_artifact(2, locator)
    text_citation = review.cite_artifact(2, locator, read["text"])
    assert review.view_artifact(2)["status"] == "unavailable"
    assert review.prepare_artifact_support("office")["status"] == "unavailable"
    installed = True  # The user installs LibreOffice outside AutoFA.
    assert review.prepare_artifact_support("office")["status"] == "available"
    image = review.view_artifact(2)
    assert image["status"] == "available"
    image_citation = review.cite_artifact(2, image["locator"])
    assert review.cite_artifact(2, locator, read["text"])["quote"] == text_citation["quote"]
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.artifact_evidence_identity() == review.artifact_evidence_identity()
    assert restored.cite_artifact(2, image["locator"])["sha256"] == image_citation["sha256"]
    assert len(restored.artifact_accesses) == 2  # One native read and the recovered page view.


# Optional readers are only detected; a speech model is downloaded only on request.


@pytest.fixture
def no_dependency_install(monkeypatch):
    monkeypatch.delenv("AUTOFA_LIBREOFFICE", raising=False)
    monkeypatch.setattr(dependencies.shutil, "which", lambda _: None)


# Applied to these tests only: it unsets AUTOFA_LIBREOFFICE and hides every PATH executable.
NO_DEPENDENCY_INSTALL = pytest.mark.usefixtures("no_dependency_install")


def on_path(monkeypatch, *names):
    monkeypatch.setattr(dependencies.shutil, "which",
                        lambda name: f"/bin/{name}" if name in names else None)


def forbid(monkeypatch, module, name):
    monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail(f"{name} must not run"))


@NO_DEPENDENCY_INSTALL
def test_office_uses_explicit_executable_not_shell_command(tmp_path, monkeypatch):
    path = tmp_path / "Office With Spaces" / "soffice"
    path.parent.mkdir()
    path.write_text("executable")
    path.chmod(0o755)
    monkeypatch.setenv("AUTOFA_LIBREOFFICE", str(path))
    assert dependencies.office_executable() == str(path)
    monkeypatch.setenv("AUTOFA_LIBREOFFICE", f"{path} --headless")
    assert dependencies.office_executable() is None


@NO_DEPENDENCY_INSTALL
def test_office_uses_path_before_standard_installs(monkeypatch):
    on_path(monkeypatch, "soffice")
    assert dependencies.office_executable() == "/bin/soffice"


@NO_DEPENDENCY_INSTALL
def test_office_detects_standard_non_path_install(monkeypatch):
    expected = str(Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice")
    monkeypatch.setattr(dependencies.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == expected)
    monkeypatch.setattr(dependencies.os, "access", lambda path, _: str(path) == expected)
    assert dependencies.office_executable() == expected


@NO_DEPENDENCY_INSTALL
@pytest.mark.parametrize("install", [False, True])
def test_office_setup_reports_but_never_installs_libreoffice(monkeypatch, install):
    # Some distributions' LibreOffice packages also install audio and video codecs.
    on_path(monkeypatch, "brew", "winget", "sudo", "apt-get", "dnf")
    monkeypatch.setattr(dependencies, "office_executable", lambda: None)
    forbid(monkeypatch, subprocess, "Popen")
    result = dependencies.prepare_artifact_support("office", install=install)
    assert result["status"] == "unavailable" and result["executable"] is None
    assert result["setup"] == dependencies.OFFICE_SETUP
    assert "AUTOFA_LIBREOFFICE" in result["reason"]
    assert "next_action" not in result and "install_command" not in result
    assert "install=True" not in json.dumps(result)


@NO_DEPENDENCY_INSTALL
def test_office_setup_reports_an_existing_installation(monkeypatch):
    monkeypatch.setattr(dependencies, "office_executable", lambda: "/app/soffice")
    result = dependencies.prepare_artifact_support("office", install=True)
    assert result["status"] == "available" and result["executable"] == "/app/soffice"
    assert "setup" not in result and "reason" not in result


@NO_DEPENDENCY_INSTALL
def test_invalid_office_override_is_reported_not_replaced(monkeypatch):
    monkeypatch.setenv("AUTOFA_LIBREOFFICE", "/missing/soffice")
    forbid(monkeypatch, subprocess, "Popen")
    result = dependencies.prepare_artifact_support("office", install=True)
    assert result["status"] == "unavailable"
    assert "does not name an executable" in result["reason"]


@NO_DEPENDENCY_INSTALL
@pytest.mark.parametrize("capability", ["audio", "speech"])
def test_media_setup_never_installs_pyav_or_speech_packages(monkeypatch, tmp_path, capability):
    # PyAV's PyPI wheels bundle FFmpeg codecs; the user builds PyAV and installs faster-whisper.
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    on_path(monkeypatch, "uv", "pip", "brew", "apt-get")
    monkeypatch.setattr(dependencies.importlib.util, "find_spec", lambda _: None)
    forbid(monkeypatch, subprocess, "Popen")
    forbid(monkeypatch, artifact_audio, "prepare_speech_model")
    result = dependencies.prepare_artifact_support(capability, install=True)
    assert result["status"] == "unavailable"
    assert result["setup"] == dependencies.AUDIO_VIDEO_SETUP
    assert result["missing"] == (["PyAV"] if capability == "audio"
                                 else ["PyAV", "faster-whisper"])
    assert "next_action" not in result and "install_command" not in result
    # Nothing in the result invites repeating a call that cannot install the packages.
    assert "install=True" not in json.dumps(result)


@NO_DEPENDENCY_INSTALL
@pytest.mark.parametrize("arguments", [
    {"capability": "arbitrary-package", "install": True},
    {"capability": "speech", "install": True, "model": "arbitrary/repo"},
    {"capability": "office", "install": "yes"},  # a truthy string is not consent to install
])
def test_setup_rejects_unrecognized_install_requests(monkeypatch, arguments):
    forbid(monkeypatch, subprocess, "Popen")
    with pytest.raises(ValueError):
        dependencies.prepare_artifact_support(**arguments)


@NO_DEPENDENCY_INSTALL
def test_speech_status_does_not_download_missing_model(monkeypatch):
    monkeypatch.setattr(dependencies, "_media_support", lambda *a, **k: {"status": "available"})

    def missing(model):
        raise ImportError("model files missing")

    monkeypatch.setattr(artifact_audio, "speech_model_identity", missing)
    forbid(monkeypatch, artifact_audio, "prepare_speech_model")
    result = dependencies.prepare_artifact_support("speech", model="small")
    assert result["status"] == "unavailable"
    assert result["dependency_status"] == "available"
    assert result["next_action"]["arguments"] == {
        "capability": "speech", "install": True, "model": "small",
    }


@NO_DEPENDENCY_INSTALL
def test_explicit_speech_setup_prepares_selected_model(monkeypatch):
    monkeypatch.setattr(dependencies, "_media_support", lambda *a, **k: {"status": "available"})
    seen = []

    def prepare(model):
        seen.append(model)
        return {"status": "available", "name": model, "sha256": "saved-identity"}

    monkeypatch.setattr(artifact_audio, "prepare_speech_model", prepare)
    result = dependencies.prepare_artifact_support("speech", install=True, model="tiny")
    assert result["status"] == "available"
    assert seen == ["tiny"]
    assert result["model"]["sha256"] == "saved-identity"


@NO_DEPENDENCY_INSTALL
def test_failed_speech_dependency_does_not_download_model(monkeypatch):
    monkeypatch.setattr(dependencies, "_media_support", lambda *a, **k: {
        "status": "unavailable", "setup": dependencies.AUDIO_VIDEO_SETUP})
    forbid(monkeypatch, artifact_audio, "prepare_speech_model")
    monkeypatch.setattr(artifact_audio, "speech_model_identity",
                        lambda *a: {"status": "unavailable"})
    result = dependencies.prepare_artifact_support("speech", install=True)
    assert result["status"] == result["dependency_status"] == "unavailable"


# Reading page/notebook source does not establish inspection of its rendered output.

HTML = '<html><body><img src="chart.png"><script>renderChart()</script></body></html>\n'
NOTEBOOK = json.dumps({"nbformat": 4, "cells": [{
    "cell_type": "code", "source": ["plot(data)"],
    "outputs": [{"output_type": "display_data", "data": {"image/png": "encoded pixels"}}],
}]})


@pytest.mark.parametrize("metadata,source,kind", [
    ({"name": "page", "mime": "text/html"}, HTML, "html"),
    ({"name": "analysis", "mime": "application/x-ipynb+json"}, NOTEBOOK, "notebook"),
])
def test_complete_source_retains_visual_gap(tmp_path, metadata, source, kind):
    path = tmp_path / "bytes"
    path.write_text(source)
    inspection = inspect(path, tmp_path, **metadata)
    assert inspection["status"] == "available"
    assert inspection["format"] == kind
    assert inspection["complete"] is True
    assert "".join(row["text"] for row in inspection["segments"]) == source
    assert inspection["requires_visual_review"] is True
    assert inspection["visual_complete"] is False
    assert inspection["visual_pages"] is None
    assert inspection["limitations"]
    rendered = render(path, tmp_path, metadata)
    assert rendered["status"] == "unsupported" and rendered["complete"] is False


def test_source_citations_do_not_certify_whole_trace_visual_review(tmp_path, monkeypatch):
    serve(monkeypatch, tmp_path / "cache", HTML.encode())
    record = item(HTML.encode(), name="page.html", mime=None)
    snapshot = make_snapshot(record, item_id="case", source="def detect(item): return True")
    review = EvidenceReview(snapshot, tmp_path / "review")
    for field in review.fields:
        review.read(field)
    inspected = review.inspect_artifact(2)
    assert inspected["format"] == "html"
    for segment in inspected["segments"]:
        review.read_artifact(2, segment["locator"])
    citation = review.cite_artifact(2, "line:1", HTML[:20])
    verdict = {"item_id": "case", "snapshot_id": snapshot["snapshot_id"],
               "scope": "local", "citations": [citation]}
    assert review.validate(verdict) is None  # Original source remains usable evidence.
    assert "artifact visuals 2" in review.validate({**verdict, "scope": "whole_trace"})
    assert review.view_artifact(2)["status"] == "unsupported"
    assert "artifact visuals 2" in review.validate({**verdict, "scope": "whole_trace"})


@pytest.mark.parametrize("name", ["page.html", "notes.txt"])
def test_inspection_policy_changes_only_affected_item_snapshots(monkeypatch, name):
    record = {"response_text": json.dumps({"steps": [], "execution": {
        "artifacts": [{"artifact_idx": 0, "name": name}],
    }})}
    current = make_snapshot(record, item_id="case", source="def detect(item): return True")
    with monkeypatch.context() as legacy:
        legacy.setattr(content, "artifact_inspection_policy", lambda metadata: None)
        previous = make_snapshot(record, item_id="case", source="def detect(item): return True")
    assert (current["snapshot_id"] != previous["snapshot_id"]) is (name != "notes.txt")


# Real encoded frames exercise the shared artifact reader and judge receipts offline. They
# need a PyAV built against FFmpeg (tools/build_ffmpeg.sh); AutoFA itself installs none.

@pytest.fixture
def movie(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "movie.webm"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libvpx-vp9", rate=2)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for color in ("red", "lime", "blue"):
            picture = Image.new("RGB", (16, 16), color)
            try:
                frame = av.VideoFrame.from_image(picture)
            finally:
                picture.close()
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def test_video_manifest_reports_metadata_without_claiming_content_was_reviewed(movie, tmp_path):
    result = inspect(movie, tmp_path)
    assert result["status"] == "available" and result["format"] == "video"
    assert result["requires_visual_review"] and not result["visual_complete"]
    assert not result["complete"] and result["segments"] == []
    assert result["video"]["declared_frame_count"] is None  # WebM declares no frame count.
    assert result["video"]["audio_streams"] == 0
    assert result["video"]["duration_seconds"] == pytest.approx(1.5)
    assert result["sha256"] == hashlib.sha256(movie.read_bytes()).hexdigest()


def test_exact_frame_pixels_timestamp_and_original_hash(movie, tmp_path):
    result = render(movie, tmp_path, page=2)
    assert result["status"] == "available"
    assert result["original_sha256"] == hashlib.sha256(movie.read_bytes()).hexdigest()
    assert result["locator"] == "video:stream:0/frame:2"
    assert result["page_count"] is None
    assert result["video"]["timestamp_seconds"] == pytest.approx(.5)
    assert result["temporal_complete"] is False and result["audio_complete"] is False
    with Image.open(result["path"]) as image:
        r, g, b = image.getpixel((0, 0))
        assert g > 240 and r < 15 and b < 15
    assert hashlib.sha256(Path(result["path"]).read_bytes()).hexdigest() == result["sha256"]


def test_video_evidence_reuses_cache_and_citations_without_whole_video_confirmation(
    movie, tmp_path, monkeypatch,
):
    provider = serve(monkeypatch, tmp_path / "cache", movie.read_bytes())
    record = item(provider.body, name="movie.webm", mime="video/webm")
    with capture_artifact_reads() as receipts:
        rendered = trace.artifact_image(record, 2, page=2)
    assert rendered["status"] == "available"
    assert hashlib.sha256(base64.b64decode(rendered["data_url"].split(",", 1)[1])).hexdigest() == (
        rendered["sha256"]
    )
    assert validate_artifact_reads(record, receipts)
    snapshot = make_snapshot(record, item_id="clip", source="def detect(x): return True")
    review = EvidenceReview(snapshot, tmp_path / "review")
    assert review.inspect_artifact(2)["complete"] is False
    review.view_artifact(2, page=2)
    citation = review.cite_artifact(2, rendered["locator"])
    assert citation["page"] == 2
    assert "unread or unavailable artifact" in review._whole_artifact_error()
    restored = EvidenceReview.restore(review.snapshot, review.directory)
    assert restored.cite_artifact(2, rendered["locator"])["sha256"] == citation["sha256"]
    assert provider.calls == 1


@pytest.mark.parametrize("page", [4, video.MAX_DECODE_FRAMES + 1])
def test_video_frame_bounds_fail_without_image_receipt(movie, tmp_path, page):
    result = render(movie, tmp_path, page=page)
    assert result["status"] != "available" and "path" not in result


@pytest.mark.parametrize("name", ["movie.mp4", "speech.ogg"])
def test_missing_pyav_is_an_explicit_evidence_gap(tmp_path, monkeypatch, name):
    path = tmp_path / name
    path.write_bytes(b"retained media bytes")
    monkeypatch.setattr(video.importlib.util, "find_spec", lambda _: None)
    inspected, viewed = inspect(path, tmp_path), render(path, tmp_path)
    assert inspected["status"] == viewed["status"] == "unavailable"
    assert not inspected["complete"] and "path" not in viewed
    for result in (inspected, viewed):
        limitations = " ".join(result["limitations"])
        assert "PyAV" in limitations and dependencies.AUDIO_VIDEO_SETUP in limitations
        assert "reinstall AutoFA" not in limitations


def test_codec_missing_from_host_ffmpeg_is_unavailable_not_corrupt(movie, tmp_path):
    # Renaming the WebM codec ID leaves PyAV without a decoder, as H.264 would in the
    # documented FFmpeg build: a missing reader, not a corrupt or empty video.
    data = movie.read_bytes()
    assert data.count(b"V_VP9") == 1
    unknown = tmp_path / "unknown-codec.webm"
    unknown.write_bytes(data.replace(b"V_VP9", b"V_ZZ9"))
    inspected, viewed = inspect(unknown, tmp_path), render(unknown, tmp_path, page=1)
    assert inspected["status"] == viewed["status"] == "unavailable"
    assert "path" not in viewed
    for result in (inspected, viewed):
        limitations = " ".join(result["limitations"])
        assert "cannot decode the artifact's video codec" in limitations
        assert dependencies.AUDIO_VIDEO_SETUP in limitations


def test_pyav_that_cannot_load_is_reported_as_a_missing_reader(monkeypatch, capsys):
    def broken_pyav(request):
        raise ImportError("libavformat.so.63: cannot open shared object file")

    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    with pytest.raises(SystemExit):
        video.run_worker_main(broken_pyav)
    result = json.loads(capsys.readouterr().out)
    assert result["unavailable"] is True
    assert "PyAV could not be loaded" in result["error"]
    assert dependencies.AUDIO_VIDEO_SETUP in result["error"]


def test_worker_reports_missing_readers_apart_from_failures():
    unavailable = video.worker_failure({"error": "no decoder", "unavailable": True}, "fallback")
    assert isinstance(unavailable, ImportError) and str(unavailable) == "no decoder"
    failed = video.worker_failure({}, "Video worker failed")
    assert type(failed) is ValueError and str(failed) == "Video worker failed"


def test_native_video_worker_timeout_is_bounded(movie, tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == video.VIDEO_TIMEOUT
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(video.subprocess, "run", timeout)
    result = render(movie, tmp_path)
    assert result["status"] != "available" and "path" not in result


def test_large_frames_are_rejected_before_decoding(movie, tmp_path, monkeypatch):
    monkeypatch.setattr(video, "MAX_DECODE_PIXELS", 10)
    with pytest.raises(ValueError, match="decode limit"):
        video._worker(movie, "matroska", 1, tmp_path / "not-created.png")
    assert not (tmp_path / "not-created.png").exists()


def test_corrupt_video_and_unsupported_playlists_are_not_successful_empty_inspections(tmp_path):
    pytest.importorskip("av")
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"#EXTM3U\nhttps://example.invalid/untrusted-stream\n")
    inspected, viewed = inspect(path, tmp_path), render(path, tmp_path)
    assert inspected["status"] == viewed["status"] == "error"
    assert not inspected["complete"] and "path" not in viewed
    assert video.video_format({"name": "playlist.m3u8", "mime": "application/x-mpegURL"}) is None


def test_video_decoder_cannot_open_external_protocols(monkeypatch, tmp_path):
    seen = []

    def fail_open(path, **kwargs):
        seen.append(kwargs)
        raise ValueError("fixture complete")

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=fail_open))
    with pytest.raises(ValueError, match="fixture complete"):
        video._worker(tmp_path / "verified-bytes", "mov", None, tmp_path / "unused")
    assert seen[0]["format"] == "mov"
    assert seen[0]["options"] == {
        "protocol_whitelist": "file", "enable_drefs": "0", "use_absolute_path": "0",
    }
