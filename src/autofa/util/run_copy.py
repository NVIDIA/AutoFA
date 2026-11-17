# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy saved run artifacts with one shared credential, staging and SQLite policy."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

_PRIVATE_FILES = {".env", "auth.json", "credentials.json", ".credentials.json"}


def copy_run_artifacts(source: Path, destination: Path) -> list[Path]:
    """Copy independent files; include committed WAL through SQLite backup.

    Callers own the source run lock. Symlinks require an explicit source decision,
    rather than silently copying data outside the saved run.
    """
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Run copy destination must be outside the source run")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A rejected symlink, interrupted copy or failed backup must leave the same
    # --output available for retry, without deleting any pre-existing destination.
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    copied = []
    try:
        for root, directories, files in os.walk(source, followlinks=False):
            root = Path(root)
            if root == source:
                directories[:] = [name for name in directories if name != "publication-artifacts"]
            for name in directories:
                if (root / name).is_symlink():
                    raise ValueError("Run contains a symlink; resolve it explicitly before copying")
            target = staging / root.relative_to(source)
            target.mkdir(exist_ok=True)
            for name in files:
                path = root / name
                if name.endswith(("-wal", "-shm", ".lock")) or name in _PRIVATE_FILES:
                    continue
                if path.is_symlink():
                    raise ValueError("Run contains a symlink; resolve it explicitly before copying")
                with path.open("rb") as stream:
                    sqlite = stream.read(16) == b"SQLite format 3\x00"
                output = target / name
                if sqlite:
                    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as original:
                        with closing(sqlite3.connect(output)) as copy:
                            original.backup(copy)
                            copy.execute("PRAGMA journal_mode=DELETE")
                else:
                    shutil.copy2(path, output)
                copied.append(output.relative_to(staging))
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return [destination / path for path in copied]
