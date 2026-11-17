# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed, immutable table stores shared by local imports and the trace viewer."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from autofa.config import autofa_home

STORE_VERSION = 1
_DIGEST = re.compile(r"[0-9a-f]{64}")
_TABLE = re.compile(r"[a-z][a-z0-9_]*\.parquet")


def source_store_root() -> Path:
    return autofa_home() / "sources"


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@lru_cache(maxsize=512)
def _checked_digest(path: str, signature: tuple[int, ...]) -> str:
    return file_digest(Path(path))


def _verify_file(path: Path, expected: str) -> None:
    if (not isinstance(expected, str) or not _DIGEST.fullmatch(expected)
            or path.is_symlink() or not path.is_file()):
        raise ValueError("Local trace store has an invalid or missing file")
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if _checked_digest(str(path), signature) != expected:
        raise ValueError("Local trace store checksum mismatch")


def resolve_store(store_id: str, root: Path | None = None) -> Path:
    """Resolve only a complete managed store; validate its immutable table contents."""
    if not isinstance(store_id, str) or not _DIGEST.fullmatch(store_id):
        raise ValueError("Invalid local trace store ID")
    base = Path(root if root is not None else source_store_root()).expanduser().resolve()
    store = base / store_id
    if store.is_symlink() or not store.is_dir():
        raise ValueError("Local trace store is unavailable")
    manifest_path = store / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Invalid local trace store manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Local trace store has no complete manifest") from exc
    if (not isinstance(manifest, dict) or manifest.get("version") != STORE_VERSION
            or manifest.get("status") != "complete" or manifest.get("store_id") != store_id
            or not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]):
        raise ValueError("Invalid local trace store manifest")
    tables = manifest.get("tables")
    directory = store / "tables"
    if (not isinstance(tables, dict) or not {"runs.parquet", "rollouts.parquet"} <= tables.keys()
            or directory.is_symlink() or not directory.is_dir()):
        raise ValueError("Local trace store is missing its run tables")
    if {p.name for p in directory.iterdir()} != set(tables):
        raise ValueError("Local trace store table inventory changed")
    for name, digest in tables.items():
        if not isinstance(name, str) or not _TABLE.fullmatch(name):
            raise ValueError("Invalid local trace table name")
        _verify_file(directory / name, digest)
    _verify_file(store / "dataset.parquet", manifest.get("dataset_sha256"))
    return store
