# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract recorded artifact content and render pages without executing document code.

Text extraction is distinct from visual review. Native OOXML locators describe
stored paragraphs/cells/slides; office PDF pages describe the application's print
layout and cannot be treated as a lossless view of hidden or non-printing content.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import mimetypes
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from autofa.data.artifact_dependencies import office_executable

ARTIFACT_TRANSFORM_VERSION = "artifact-content-v1"
MAX_FILE_BYTES = 256 * 1024**2
MAX_ARCHIVE_BYTES = 256 * 1024**2
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024**2
MAX_ARCHIVE_MEMBERS = 10_000
MAX_SEGMENTS = 250_000
MAX_RENDER_PIXELS = 16_000_000
MAX_RENDER_SIDE = 8_000
OFFICE_RENDER_TIMEOUT = 90
_PDFIUM_LOCK = threading.RLock()  # PDFium operations, including close(), are not thread safe.
_CSV_LOCK = threading.RLock()
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


class ArtifactContentError(ValueError):
    """Malformed, unsupported, encrypted, or bounded artifact content."""


class ArtifactContentLimit(ArtifactContentError):
    """An explicit resource boundary was reached; partial evidence is not complete."""


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _format(path: Path, metadata: dict) -> tuple[str, str]:
    name = str(metadata.get("name") or path.name)
    suffix = Path(name).suffix.lower()
    mime = str(
        metadata.get("mime_type")
        or metadata.get("mime")
        or mimetypes.guess_type(name)[0]
        or "application/octet-stream"
    ).split(";", 1)[0]
    formats = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".pptx": "pptx",
        ".csv": "csv",
        ".tsv": "tsv",
        ".html": "html",
        ".htm": "html",
        ".ipynb": "notebook",
    }
    if suffix in formats:
        return formats[suffix], mime
    from autofa.data.artifact_archive import archive_format
    from autofa.data.artifact_audio import audio_format
    from autofa.data.artifact_video import video_format

    if video_format({"name": name, "mime_type": mime}):
        return "video", mime
    if audio_format({"name": name, "mime_type": mime}):
        return "audio", mime
    if archive_format({"name": name, "mime_type": mime}):
        return "zip", mime
    if mime in {"text/html", "application/xhtml+xml"}:
        return "html", mime
    if mime == "application/x-ipynb+json":
        return "notebook", mime
    if mime in {"text/csv", "text/tab-separated-values"}:
        return ("csv" if mime == "text/csv" else "tsv"), mime
    if mime.startswith("image/") or suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".tiff",
        ".tif",
        ".bmp",
    }:
        return "image", mime
    if mime == "application/pdf":
        return "pdf", mime
    office_mimes = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    }
    if mime in office_mimes:
        return office_mimes[mime], mime
    if (
        mime.startswith("text/")
        or mime in {"application/json", "application/xml"}
        or suffix
        in {
            ".txt",
            ".md",
            ".json",
            ".jsonl",
            ".log",
            ".xml",
            ".yaml",
            ".yml",
            ".py",
        }
    ):
        return "text", mime
    return "unsupported", mime


def artifact_inspection_policy(metadata: dict) -> str | None:
    """Version newly supported evidence surfaces without invalidating other formats."""
    kind, _ = _format(Path("artifact"), metadata)
    return {
        "html": "artifact-inspection-html-source-v1",
        "notebook": "artifact-inspection-notebook-source-v1",
        "video": "artifact-inspection-video-frame-v1",
        "audio": "artifact-inspection-audio-window-v1",
        "zip": "artifact-inspection-archive-member-v1",
    }.get(kind)


def _base(path: Path, metadata: dict) -> dict:
    if not path.is_file():
        raise ArtifactContentError("Artifact bytes are not available as a local file.")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ArtifactContentLimit(f"Artifact exceeds the {MAX_FILE_BYTES}-byte inspection limit.")
    format_name, mime = _format(path, metadata)
    visual = format_name in {
        "image", "pdf", "docx", "xlsx", "pptx", "video", "audio", "html", "notebook",
    }
    return {
        "status": "available",
        "format": format_name,
        "mime_type": mime,
        "size_bytes": size,
        "sha256": _sha256(path),
        "transform_version": ARTIFACT_TRANSFORM_VERSION,
        "segments": [],
        "complete": True,
        "limitations": [],
        "requires_visual_review": visual,
        "visual_complete": not visual,
        "visual_pages": None if visual else 0,
        "scope": "Stored textual content; visual appearance is a separate evidence surface.",
    }


def _segment(result: dict, locator: str, text: str, location: dict, **extra: Any) -> None:
    if len(result["segments"]) >= MAX_SEGMENTS:
        raise ArtifactContentLimit(f"Extraction stopped at the {MAX_SEGMENTS}-segment limit.")
    result["segments"].append(
        {
            "locator": locator,
            "location": location,
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            **extra,
        }
    )


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


class _Package:
    def __init__(self, path: Path):
        with path.open("rb") as handle:
            if handle.read(8) == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
                raise ArtifactContentError(
                    "Encrypted or legacy compound office files are unsupported."
                )
        self.archive = zipfile.ZipFile(path)
        members = self.archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            self.archive.close()
            raise ArtifactContentLimit("Office archive contains too many members.")
        total = 0
        names = set()
        for member in members:
            total += member.file_size
            if (
                member.filename in names
                or member.flag_bits & 1
                or member.filename.startswith("/")
                or ".." in member.filename.split("/")
            ):
                self.archive.close()
                raise ArtifactContentError(
                    "Office archive has encrypted, duplicate, or unsafe members."
                )
            names.add(member.filename)
            if member.file_size > MAX_ARCHIVE_MEMBER_BYTES or total > MAX_ARCHIVE_BYTES:
                self.archive.close()
                raise ArtifactContentLimit("Office archive exceeds the decompressed-content limit.")
        self.names = names

    def close(self) -> None:
        self.archive.close()

    def xml(self, name: str) -> ET.Element:
        data = self.archive.read(name)
        if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
            raise ArtifactContentError("XML declarations with DTD/entities are not supported.")
        return ET.fromstring(data)

    def relationships(self, part: str) -> dict[str, tuple[str, str]]:
        directory, basename = posixpath.split(part)
        name = posixpath.join(directory, "_rels", basename + ".rels")
        if name not in self.names:
            return {}
        result = {}
        for relationship in self.xml(name):
            if relationship.get("TargetMode") == "External":
                continue
            target = relationship.get("Target", "")
            target = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath(posixpath.join(directory, target))
            )
            if target.startswith("../") or target not in self.names:
                raise ArtifactContentError(
                    "Office relationship points outside the available package."
                )
            result[relationship.get("Id", "")] = (target, relationship.get("Type", ""))
        return result


def _xml_text(element: ET.Element) -> str:
    pieces = []
    for child in element.iter():
        tag = _tag(child)
        if tag in {"t", "delText"}:
            pieces.append(child.text or "")
        elif tag == "tab":
            pieces.append("\t")
        elif tag in {"br", "cr"}:
            pieces.append("\n")
    return "".join(pieces)


def _docx(package: _Package, result: dict) -> None:
    parts = [
        "word/document.xml",
        *sorted(
            name
            for name in package.names
            if name.startswith(("word/header", "word/footer")) and name.endswith(".xml")
        ),
        *[
            name
            for name in ("word/footnotes.xml", "word/endnotes.xml", "word/comments.xml")
            if name in package.names
        ],
    ]
    for part in parts:
        root = package.xml(part)
        container = next((child for child in root if _tag(child) == "body"), root)
        paragraph, table = 0, 0
        for child in container:
            if _tag(child) == "p":
                paragraph += 1
                _segment(
                    result,
                    f"{part}#paragraph:{paragraph}",
                    _xml_text(child),
                    {"kind": "paragraph", "part": part, "paragraph": paragraph},
                )
            elif _tag(child) == "tbl":
                table += 1
                rows = [row for row in child if _tag(row) == "tr"]
                for row_number, row in enumerate(rows, 1):
                    for cell_number, cell in enumerate(
                        (cell for cell in row if _tag(cell) == "tc"),
                        1,
                    ):
                        text = "\n".join(_xml_text(p) for p in cell.iter() if _tag(p) == "p")
                        _segment(
                            result,
                            f"{part}#table:{table}/row:{row_number}/cell:{cell_number}",
                            text,
                            {
                                "kind": "table_cell",
                                "part": part,
                                "table": table,
                                "row": row_number,
                                "cell": cell_number,
                            },
                        )
            else:
                # Notes/comments/content controls wrap ordinary paragraphs.
                for nested in child.iter():
                    if _tag(nested) == "p":
                        paragraph += 1
                        _segment(
                            result,
                            f"{part}#paragraph:{paragraph}",
                            _xml_text(nested),
                            {"kind": "paragraph", "part": part, "paragraph": paragraph},
                        )
        if any(_tag(child) in {"del", "ins"} for child in root.iter()):
            result["limitations"].append(
                f"{part}: tracked changes retained as stored text; "
                "revision display is not resolved."
            )
    result["limitations"].append(
        "DOCX extraction covers stored paragraphs/tables/notes, not pagination or embedded visuals."
    )


def _xlsx(package: _Package, result: dict) -> None:
    workbook = "xl/workbook.xml"
    root, relationships = package.xml(workbook), package.relationships(workbook)
    strings = []
    if "xl/sharedStrings.xml" in package.names:
        strings = [_xml_text(entry) for entry in package.xml("xl/sharedStrings.xml")]
    result["sheets"] = []
    for sheet in (entry for entry in root.iter() if _tag(entry) == "sheet"):
        name = sheet.get("name", "")
        reference = relationships.get(sheet.get(_R, ""))
        if reference is None:
            raise ArtifactContentError("Workbook sheet relationship is missing.")
        part = reference[0]
        result["sheets"].append(
            {
                "name": name,
                "sheet_id": sheet.get("sheetId"),
                "state": sheet.get("state", "visible"),
                "part": part,
            }
        )
        for cell in (entry for entry in package.xml(part).iter() if _tag(entry) == "c"):
            coordinate = cell.get("r")
            if not coordinate:
                raise ArtifactContentError("Spreadsheet cell coordinate is missing.")
            value = next((child.text or "" for child in cell if _tag(child) == "v"), None)
            formula_node = next((child for child in cell if _tag(child) == "f"), None)
            cell_type = cell.get("t", "n")
            if cell_type == "s":
                try:
                    if value is None or int(value) < 0:
                        raise ValueError
                    text = strings[int(value)]
                except (ValueError, IndexError) as exc:
                    raise ArtifactContentError(
                        "Invalid spreadsheet shared-string reference."
                    ) from exc
            elif cell_type == "inlineStr":
                text = _xml_text(cell)
            else:
                text = value or ""
            extras = {"cell_type": cell_type, "style_id": cell.get("s"), "cached_value": value}
            if formula_node is not None:
                extras["formula"] = formula_node.text or ""
                extras["formula_attributes"] = dict(formula_node.attrib)
                text = (
                    "="
                    + extras["formula"]
                    + ("\nCached value: " + text if value is not None else "")
                )
            _segment(
                result,
                f"sheet:{urllib.parse.quote(name, safe='')}!{coordinate}",
                text,
                {
                    "kind": "cell",
                    "sheet": name,
                    "cell": coordinate,
                    "part": part,
                    "sheet_state": sheet.get("state", "visible"),
                },
                **extras,
            )
    result["limitations"].append(
        "Spreadsheet cells retain formulas and cached XML values; formulas are not evaluated, "
        "and number formatting, charts, drawings and print layout require separate review."
    )


def _pptx(package: _Package, result: dict) -> None:
    presentation = "ppt/presentation.xml"
    root, relationships = package.xml(presentation), package.relationships(presentation)
    slides = [entry for entry in root.iter() if _tag(entry) == "sldId"]
    result["visual_pages"] = len(slides)
    for number, slide in enumerate(slides, 1):
        reference = relationships.get(slide.get(_R, ""))
        if reference is None:
            raise ArtifactContentError("Presentation slide relationship is missing.")
        part = reference[0]
        xml = package.xml(part)
        text = "\n".join(_xml_text(p) for p in xml.iter() if _tag(p) == "p")
        _segment(
            result,
            f"slide:{number}",
            text,
            {"kind": "slide", "slide": number, "slide_id": slide.get("id"), "part": part},
        )
        for notes, relation_type in package.relationships(part).values():
            if relation_type.endswith("/notesSlide"):
                text = "\n".join(_xml_text(p) for p in package.xml(notes).iter() if _tag(p) == "p")
                _segment(
                    result,
                    f"slide:{number}/notes",
                    text,
                    {"kind": "slide_notes", "slide": number, "part": notes},
                )
    result["limitations"].append(
        "Slide extraction retains stored text and notes; layout, images and charts need rendering."
    )


def _text(path: Path, format_name: str, result: dict) -> None:
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    text = raw.decode(encoding)
    result["encoding"] = encoding
    if format_name in {"csv", "tsv"}:
        with _CSV_LOCK:
            previous = csv.field_size_limit(MAX_FILE_BYTES)
            try:
                reader = csv.reader(
                    io.StringIO(text), delimiter="\t" if format_name == "tsv" else ","
                )
                for row_number, row in enumerate(reader, 1):
                    for column, value in enumerate(row, 1):
                        _segment(
                            result,
                            f"row:{row_number}/column:{column}",
                            value,
                            {"kind": "csv_cell", "row": row_number, "column": column},
                        )
            finally:
                csv.field_size_limit(previous)
    else:
        for number, line in enumerate(text.splitlines(keepends=True), 1):
            _segment(result, f"line:{number}", line, {"kind": "line", "line": number})


def _pdf(path: Path, result: dict) -> None:
    pdfium = importlib.import_module("pypdfium2")
    with _PDFIUM_LOCK:
        document = pdfium.PdfDocument(str(path))
        try:
            result["visual_pages"] = len(document)
            for number in range(len(document)):
                page = document[number]
                try:
                    textpage = page.get_textpage()
                    try:
                        text = textpage.get_text_bounded()
                    finally:
                        textpage.close()
                    _segment(
                        result, f"page:{number + 1}", text, {"kind": "page", "page": number + 1}
                    )
                finally:
                    page.close()
        finally:
            document.close()
    result["limitations"].append(
        "PDF text extraction does not inspect images, vector graphics or scanned page content; "
        "render pages for visual evidence."
    )


def _image(path: Path, result: dict) -> None:
    image_module = importlib.import_module("PIL.Image")
    with image_module.open(path) as source:
        result["visual_pages"] = getattr(source, "n_frames", 1)
        result["image"] = {"width": source.width, "height": source.height, "mode": source.mode}
    result["limitations"].append(
        "Image pixels require render_artifact; no OCR or visual review performed."
    )


def inspect_artifact(
    path: str | Path, metadata: dict | None = None, *, output_dir: str | Path
) -> dict:
    """Return all supported text segments with stable locators and explicit coverage.

    No network access, macro execution, formula evaluation, or implicit OCR occurs.
    ``output_dir`` reserves the shared service interface; extraction remains in memory.
    """
    del output_dir
    path = Path(path)
    result: dict = {
        "status": "error",
        "segments": [],
        "complete": False,
        "limitations": [],
        "transform_version": ARTIFACT_TRANSFORM_VERSION,
        "visual_complete": False,
    }
    try:
        result = _base(path, metadata or {})
        kind = result["format"]
        if kind in {"text", "csv", "tsv", "html", "notebook"}:
            _text(path, kind, result)
            if kind in {"html", "notebook"}:
                result["scope"] = (
                    "Complete stored source text only; rendered output is uninspected."
                )
                result["limitations"].append(
                    "HTML source was read without rendering layout, executing scripts, or "
                    "retrieving embedded/linked resources. Visual and interactive behavior "
                    "remains uninspected; no HTML renderer is available."
                    if kind == "html" else
                    "Notebook JSON source was read without executing cells or displaying "
                    "embedded outputs. Plots, images, and rendered output remain uninspected; "
                    "no notebook renderer is available."
                )
        elif kind in {"docx", "xlsx", "pptx"}:
            package = _Package(path)
            try:
                {"docx": _docx, "xlsx": _xlsx, "pptx": _pptx}[kind](package, result)
            finally:
                package.close()
        elif kind == "pdf":
            _pdf(path, result)
        elif kind == "image":
            _image(path, result)
        elif kind == "zip":
            from autofa.data.artifact_archive import inspect_archive

            result.update(inspect_archive(path))
        elif kind == "audio":
            from autofa.data.artifact_audio import audio_operation

            audio, _ = audio_operation(path, metadata or {"name": path.name})
            result.update(audio=audio, complete=False, audio_complete=False,
                          visual_pages=audio.get("declared_window_count"))
            result["scope"] = "Audio stream metadata only; no sound has been listened to."
            result["limitations"].append(
                "Use view_artifact for waveform/spectrogram windows and transcribe_artifact "
                "for local speech estimates. Neither substitutes for perceptual audio review."
            )
        elif kind == "video":
            from autofa.data.artifact_video import video_operation

            video, _ = video_operation(path, metadata or {"name": path.name})
            result.update(video=video, complete=False,
                          visual_pages=video["declared_frame_count"])
            result["scope"] = "Video stream metadata only; use frame views for visual evidence."
            result["limitations"].append(
                "Video stream metadata only; no frames or audio were inspected. "
                "Use view_artifact to inspect selected frames with their recorded timestamps."
            )
        else:
            result.update(status="unsupported", complete=False)
            result["limitations"].append(
                "No content inspector is available for this artifact format."
            )
    except ImportError as exc:
        result.update(status="unavailable", complete=False, visual_complete=False)
        # Audio/video readers explain their own setup; AutoFA does not install PyAV.
        result["limitations"].append(
            str(exc) if result.get("format") in {"audio", "video"} else
            "PDF/image inspection needs Pillow and pypdfium2, which AutoFA installs; "
            "reinstall AutoFA. " + str(exc)
        )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError, csv.Error) as exc:
        result.update(status="error", complete=False, visual_complete=False)
        result["limitations"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
    except Exception as exc:  # optional native parsers use provider-specific exception types
        result.update(status="error", complete=False, visual_complete=False)
        result["limitations"].append(
            f"Artifact could not be decoded: {type(exc).__name__}: {str(exc)[:500]}"
        )
    return result


def _render_size(width: float, height: float, scale: float = 1.5) -> tuple[float, bool]:
    if width <= 0 or height <= 0:
        raise ArtifactContentError("Page/image dimensions must be positive.")
    bounded = min(
        scale,
        MAX_RENDER_SIDE / width,
        MAX_RENDER_SIDE / height,
        (MAX_RENDER_PIXELS / (width * height)) ** 0.5,
    )
    return bounded, bounded < scale


def _render_pdf(path: Path, page: int) -> tuple[Any, int, list[str]]:
    pdfium = importlib.import_module("pypdfium2")
    with _PDFIUM_LOCK:
        document = pdfium.PdfDocument(str(path))
        try:
            count = len(document)
            if not 1 <= page <= count:
                raise ArtifactContentError(f"Page must be between 1 and {count}.")
            source = document[page - 1]
            try:
                scale, reduced = _render_size(*source.get_size())
                bitmap = source.render(scale=scale)
                try:
                    image = bitmap.to_pil().copy()
                finally:
                    bitmap.close()
            finally:
                source.close()
        finally:
            document.close()
    return (
        image,
        count,
        (
            ["Page rasterization reduced to the configured pixel/dimension limit."]
            if reduced
            else []
        ),
    )


def _render_image(path: Path, page: int) -> tuple[Any, int, list[str]]:
    module = importlib.import_module("PIL.Image")
    image_ops = importlib.import_module("PIL.ImageOps")
    with module.open(path) as source:
        count = getattr(source, "n_frames", 1)
        if not 1 <= page <= count:
            raise ArtifactContentError(f"Image frame must be between 1 and {count}.")
        source.seek(page - 1)
        if source.width * source.height > MAX_RENDER_PIXELS * 16:
            raise ArtifactContentLimit("Image decode exceeds the supported pixel limit.")
        oriented = image_ops.exif_transpose(source)
        try:
            scale, reduced = _render_size(oriented.width, oriented.height, 1.0)
            image = oriented.convert("RGBA")
            if reduced:
                resized = image.resize(
                    (max(1, int(oriented.width * scale)), max(1, int(oriented.height * scale)))
                )
                image.close()
                image = resized
        finally:
            oriented.close()
    return (
        image,
        count,
        (["Image resized to the configured pixel/dimension limit."] if reduced else []),
    )


def _office_pdf(path: Path, kind: str, output_dir: Path) -> Path:
    executable = office_executable()
    if not executable:
        raise ImportError(
            "LibreOffice is required for office document visual rendering. Use "
            "prepare_artifact_support(capability='office', install=True), or install "
            "LibreOffice and set AUTOFA_LIBREOFFICE to its soffice executable."
        )
    # Reject encrypted/oversized archives before an external renderer sees them.
    package = _Package(path)
    try:
        if any(name.lower().endswith("vbaproject.bin") for name in package.names):
            raise ArtifactContentError("Macro-bearing office documents are not rendered.")
        for name in package.names:
            if name.endswith(".rels"):
                if any(
                    relation.get("TargetMode") == "External"
                    and not relation.get("Type", "").endswith("/hyperlink")
                    for relation in package.xml(name)
                ):
                    raise ArtifactContentError(
                        "Office rendering with external linked content is unsupported; "
                        "the renderer must not fetch untracked evidence."
                    )
            if name.endswith(".xml") and name.startswith(("word/", "xl/worksheets/")):
                for node in package.xml(name).iter():
                    if _tag(node) == "f" and re.search(
                        r"\b(?:DDE|WEBSERVICE|RTD|IMAGE)\s*\(",
                        node.text or "",
                        re.IGNORECASE,
                    ):
                        raise ArtifactContentError(
                            "Office rendering with active external-data formulas is unsupported."
                        )
                    instructions = [node.text or ""] if _tag(node) == "instrText" else []
                    instructions.extend(
                        value
                        for key, value in node.attrib.items()
                        if key.rsplit("}", 1)[-1] == "instr"
                    )
                    if any(
                        re.match(r"\s*DDE(?:AUTO)?(?:\s|$)", value, re.IGNORECASE)
                        for value in instructions
                    ):
                        raise ArtifactContentError(
                            "Office rendering with active DDE fields is unsupported."
                        )
    finally:
        package.close()
    directory = Path(tempfile.mkdtemp(prefix="office-render-", dir=output_dir))
    source = directory / f"source.{kind}"
    shutil.copyfile(path, source)
    profile = directory / "profile"
    user = profile / "user"
    user.mkdir(parents=True)
    # Isolated profile prevents macros, link refresh, and inherited user extensions.
    (user / "registrymodifications.xcu").write_text(
        '<?xml version="1.0"?><oor:items xmlns:oor="http://openoffice.org/2001/registry">'
        '<item oor:path="/org.openoffice.Office.Common/Security/Scripting">'
        '<prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop></item>'
        '<item oor:path="/org.openoffice.Office.Calc/Content/Update">'
        '<prop oor:name="Link" oor:op="fuse"><value>0</value></prop></item></oor:items>',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            executable,
            "-env:UserInstallation=" + profile.as_uri(),
            "--headless",
            "--nologo",
            "--nodefault",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            str(directory),
            str(source),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=OFFICE_RENDER_TIMEOUT,
        check=False,
    )
    target = directory / "source.pdf"
    if completed.returncode or not target.is_file():
        raise ArtifactContentError("LibreOffice did not produce a PDF representation.")
    return target


def render_artifact(
    path: str | Path,
    metadata: dict | None = None,
    *,
    output_dir: str | Path,
    page: int = 1,
) -> dict:
    """Render one complete page/frame to PNG; return hashes, locator, and rendering limits."""
    result: dict = {
        "status": "error",
        "complete": False,
        "limitations": [],
        "transform_version": ARTIFACT_TRANSFORM_VERSION,
    }
    try:
        path, output_dir = Path(path), Path(output_dir)
        base = _base(path, metadata or {})
        result.update(original_sha256=base["sha256"], format=base["format"])
        if type(page) is not int or page < 1:
            raise ArtifactContentError("page must be a positive 1-based integer.")
        output_dir.mkdir(parents=True, exist_ok=True)
        kind = base["format"]
        video = None
        audio = None
        if kind == "image":
            image, count, limitations = _render_image(path, page)
            locator = f"image:{page}"
        elif kind == "video":
            from autofa.data.artifact_video import VIDEO_LIMITATION, video_operation

            video, image = video_operation(path, metadata or {"name": path.name}, page=page)
            count, locator = video["declared_frame_count"], video["locator"]
            limitations = [VIDEO_LIMITATION]
        elif kind == "audio":
            from autofa.data.artifact_audio import AUDIO_LIMITATION, audio_operation

            audio, image = audio_operation(path, metadata or {"name": path.name}, page=page)
            count, locator = audio.get("declared_window_count"), audio["locator"]
            limitations = [AUDIO_LIMITATION]
        elif kind == "pdf":
            image, count, limitations = _render_pdf(path, page)
            locator = f"page:{page}"
        elif kind in {"docx", "xlsx", "pptx"}:
            converted = _office_pdf(path, kind, output_dir)
            image, count, limitations = _render_pdf(converted, page)
            locator = f"page:{page}"
            limitations.append(
                "Office rendering is a print-layout conversion, not a native text locator; "
                "hidden/non-printing content may be omitted and formulas may be recalculated. "
                "Inspect native formula/cached-value segments for the original recorded values."
            )
        else:
            result.update(status="unsupported")
            result["limitations"].append("This format has no visual page renderer.")
            return result
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".png", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            image.save(temporary, format="PNG")
            rendered_hash = _sha256(temporary)
            destination = output_dir / f"{rendered_hash}.png"
            os.replace(temporary, destination)
            result.update(
                status="available",
                complete=True,
                path=str(destination),
                mime_type="image/png",
                sha256=rendered_hash,
                locator=locator,
                page_count=count,
                width=image.width,
                height=image.height,
                limitations=limitations,
            )
            if video is not None:
                result.update(video=video, temporal_complete=False, audio_complete=False)
            if audio is not None:
                result.update(audio=audio, temporal_complete=False, audio_complete=False)
        finally:
            image.close()
            temporary.unlink(missing_ok=True)
    except ImportError as exc:
        result.update(status="unavailable")
        result["limitations"].append(
            str(exc) or "PDF/image rendering needs Pillow and pypdfium2; reinstall AutoFA."
        )
    except Exception as exc:
        result["limitations"].append(
            f"Rendering unavailable: {type(exc).__name__}: {str(exc)[:500]}"
        )
    return result
