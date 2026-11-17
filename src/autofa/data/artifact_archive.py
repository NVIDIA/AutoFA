# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded ZIP inventories and explicit member reads from verified local bytes.

No member name is used as an output path. Archives are never extracted as a tree,
and listing members does not establish that their contents have been inspected.
"""

from __future__ import annotations

import hashlib
import mimetypes
import stat
import struct
import zipfile
from contextlib import contextmanager
from pathlib import Path

ARCHIVE_POLICY = "artifact-inspection-zip-member-v1"
MAX_ARCHIVE_BYTES = 256 * 1024**2
MAX_MEMBER_BYTES = 64 * 1024**2
MAX_EXPANDED_BYTES = 256 * 1024**2
MAX_MEMBERS = 10_000
MAX_DIRECTORY_BYTES = 16 * 1024**2
MAX_COMPRESSION_RATIO = 1_000


def archive_format(metadata: dict) -> bool:
    name = str(metadata.get("name") or "")
    mime = str(metadata.get("mime_type") or metadata.get("mime") or "").split(";", 1)[0]
    return Path(name).suffix.lower() == ".zip" or mime in {
        "application/zip", "application/x-zip-compressed",
    }


def validate_member_path(value: str) -> str:
    if (not isinstance(value, str) or not value or len(value) > 4096
            or "\\" in value or ":" in value or "\x00" in value
            or value.startswith("/") or any(ord(char) < 32 for char in value)):
        raise ValueError("ZIP member must be an exact, relative archive path")
    parts = value.removesuffix("/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("ZIP member contains unsafe or ambiguous path components")
    return value


def _preflight(handle) -> int:
    """Bound the central directory before ZipFile allocates one object per entry."""
    handle.seek(0, 2)
    size = handle.tell()
    if size > MAX_ARCHIVE_BYTES:
        raise ValueError("ZIP exceeds the 256 MiB archive inspection limit")
    handle.seek(max(0, size - 65_557))
    tail = handle.read(65_557)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or len(tail) - offset < 22:
        raise ValueError("ZIP end-of-directory record is unavailable")
    _, disk, start_disk, disk_count, count, directory_size, directory_start, comment_size = (
        struct.unpack("<4s4H2LH", tail[offset:offset + 22])
    )
    if offset + 22 + comment_size != len(tail):
        raise ValueError("ZIP has ambiguous trailing records")
    if (disk or start_disk or disk_count != count or count == 0xffff
            or directory_size == 0xffffffff or directory_start == 0xffffffff):
        raise ValueError("Split and ZIP64 archives are not supported")
    if count > MAX_MEMBERS or directory_size > MAX_DIRECTORY_BYTES:
        raise ValueError("ZIP exceeds bounded member inventory limits")
    end_position = size - len(tail) + offset
    if directory_start + directory_size != end_position:
        raise ValueError("ZIP central directory offsets are inconsistent")
    # A dishonest EOCD count must not bypass the object-allocation limit. Parse
    # only the fixed-width entry headers and skip bounded metadata before ZipFile.
    handle.seek(directory_start)
    observed = 0
    while handle.tell() < end_position:
        header = handle.read(46)
        if len(header) != 46 or header[:4] != b"PK\x01\x02":
            raise ValueError("ZIP central directory entry is malformed")
        name_size, extra_size, note_size = struct.unpack_from("<3H", header, 28)
        handle.seek(name_size + extra_size + note_size, 1)
        observed += 1
        if observed > MAX_MEMBERS or handle.tell() > end_position:
            raise ValueError("ZIP exceeds bounded member inventory limits")
    if observed != count:
        raise ValueError("ZIP declared member count differs from its directory")
    handle.seek(0)
    return count


def _members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    expanded = 0
    canonical: set[str] = set()
    for info in archive.infolist():
        name = validate_member_path(info.orig_filename)
        normalized = name.removesuffix("/")
        if normalized in canonical or name != info.filename:
            raise ValueError("ZIP contains duplicate or ambiguous member paths")
        canonical.add(normalized)
        mode = stat.S_IFMT(info.external_attr >> 16)
        if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError("ZIP contains symbolic links or special file members")
        if info.flag_bits & 1:
            raise ValueError("Encrypted ZIP members are not supported")
        if info.compress_type not in {
            zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2,
        }:
            raise ValueError("ZIP member compression method is unsupported")
        if (info.file_size > MAX_MEMBER_BYTES
                or info.file_size > max(1, info.compress_size) * MAX_COMPRESSION_RATIO):
            raise ValueError("ZIP member exceeds extraction size or compression-ratio limits")
        expanded += info.file_size
        if expanded > MAX_EXPANDED_BYTES:
            raise ValueError("ZIP expanded members exceed the 256 MiB inspection limit")
        members[name] = info
    # File 'a' plus 'a/b' is not a coherent file tree, even though we never unpack it.
    files = {name for name, info in members.items() if not info.is_dir()}
    for name in members:
        parts = name.removesuffix("/").split("/")
        if any("/".join(parts[:index]) in files for index in range(1, len(parts))):
            raise ValueError("ZIP contains conflicting file and directory paths")
    return members


@contextmanager
def _open_archive(path: Path, *, expected_sha256: str | None = None):
    try:
        with path.open("rb") as handle:
            if expected_sha256 is not None:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected_sha256:
                    raise ValueError("Archive bytes differ from the pinned source")
                handle.seek(0)
            _preflight(handle)
            with zipfile.ZipFile(handle) as archive:
                yield archive, _members(archive)
            if expected_sha256 is not None:
                handle.seek(0)
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected_sha256:
                    raise ValueError("Archive bytes changed during member inspection")
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
        raise ValueError(f"ZIP could not be inspected: {exc}") from exc


def _metadata(info: zipfile.ZipInfo) -> dict:
    return {
        "member_path": info.filename, "name": info.filename,
        "mime_type": mimetypes.guess_type(info.filename)[0] or "application/octet-stream",
        "size_bytes": info.file_size, "compressed_bytes": info.compress_size,
        "directory": info.is_dir(), "crc32": f"{info.CRC:08x}",
    }


def inspect_archive(path: Path) -> dict:
    """List bounded member metadata; never decompress or claim member review."""
    with _open_archive(Path(path)) as (_, members):
        rows = [_metadata(info) for info in members.values()]
    return {
        "archive": {"members": rows, "member_count": len(rows), "inventory_complete": True},
        "segments": [{"locator": f"archive-member:{row['member_path']}",
                      "text": f"{row['member_path']} ({row['size_bytes']} bytes)"}
                     for row in rows],
        "complete": False,
        "scope": "ZIP member inventory only; member contents remain uninspected.",
        "limitations": [
            "Archive listing does not inspect member contents. Select an exact member_path "
            "to read, render, or download that file. Nested archive selection is unsupported.",
        ],
    }


def member_metadata(path: Path, member_path: str, *, expected_sha256: str) -> dict:
    validate_member_path(member_path)
    with _open_archive(Path(path), expected_sha256=expected_sha256) as (_, members):
        info = members.get(member_path)
        if info is None or info.is_dir():
            raise ValueError("Selected ZIP member is absent or is a directory")
        return _metadata(info)


def extract_member(path: Path, member_path: str, dest: Path, *, expected_sha256: str) -> None:
    """Stream one member to the caller's private path; verify CRC and byte count."""
    validate_member_path(member_path)
    with _open_archive(Path(path), expected_sha256=expected_sha256) as (archive, members):
        info = members.get(member_path)
        if info is None or info.is_dir():
            raise ValueError("Selected ZIP member is absent or is a directory")
        count = 0
        with archive.open(info) as source, dest.open("xb") as target:
            while chunk := source.read(256 * 1024):
                count += len(chunk)
                if count > min(info.file_size, MAX_MEMBER_BYTES):
                    raise ValueError(
                        "ZIP member exceeds its declared byte count or extraction limit",
                    )
                target.write(chunk)
        if count != info.file_size:
            raise ValueError("ZIP member size differs from its declared byte count")
