# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Protocol

from .config import Settings


class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


def _safe_key(key: str) -> tuple[str, ...]:
    if not key or "\\" in key:
        raise ValueError("artifact key is invalid")
    path = PurePosixPath(key)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact key is invalid")
    return path.parts


def _archive_member(key: str) -> tuple[str, str]:
    archive_key, separator, member = key.removeprefix("zip:").partition("!/")
    if not separator or not archive_key.endswith(".zip"):
        raise ValueError("archive artifact key is invalid")
    _safe_key(archive_key)
    _safe_key(member)
    return archive_key, member


class FilesystemArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        destination = self.root.joinpath(*_safe_key(key)).resolve()
        try:
            destination.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact key escapes storage root") from exc
        return destination

    def put(self, key: str, data: bytes) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temp_name = temporary.name
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temp_name, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)

    def get(self, key: str) -> bytes:
        from .lazy_report import KEY_PREFIX, resolve_lazy_asset

        if key.startswith(KEY_PREFIX):
            try:
                return resolve_lazy_asset(self.get(key.removeprefix(KEY_PREFIX)))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise FileNotFoundError("Local report example is unavailable") from exc
        if key.startswith("zip:"):
            archive_key, member = _archive_member(key)
            try:
                with zipfile.ZipFile(self._path(archive_key)) as archive:
                    return archive.read(member)
            except KeyError as exc:
                raise FileNotFoundError(key) from exc
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def put_archive(self, key: str, members: list[tuple[str, bytes]]) -> None:
        """Store one immutable snapshot of many report sidecars atomically.

        A local run can have thousands of trajectories. One archive avoids making
        thousands of independently flushed files just to display that saved run.
        """
        payload = io.BytesIO()
        with zipfile.ZipFile(
            payload, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for name, content in members:
                _safe_key(name)
                archive.writestr(name, content)
        self.put(key, payload.getvalue())

    def delete(self, key: str) -> None:
        from .lazy_report import KEY_PREFIX

        key = key.removeprefix(KEY_PREFIX)
        # Archive members belong to a single immutable version. Removing that
        # version/library removes its archive once; subsequent removals are no-ops.
        if key.startswith("zip:"):
            key, _ = _archive_member(key)
        path = self._path(key)
        path.unlink(missing_ok=True)
        parent = path.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def create_artifact_store(settings: Settings) -> ArtifactStore:
    return FilesystemArtifactStore(settings.data_dir / "artifacts")
