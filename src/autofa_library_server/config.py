# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem settings for the local library dashboard."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from autofa.local_service import service_root, service_token


def _library_dir() -> Path:
    from autofa.config import load_config

    return load_config().resolved_library_root()


def _runs_dir() -> Path:
    from autofa.config import runs_root

    return runs_root()


def _data_api_base_url() -> str:
    from autofa.config import load_config

    return str(getattr(load_config(), "data_api_base_url", "") or "")


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path = field(default_factory=service_root)
    library_dir: Path = field(default_factory=_library_dir)
    runs_dir: Path = field(default_factory=_runs_dir)
    data_api_base_url: str = field(default_factory=_data_api_base_url)
    public_url: str = "http://127.0.0.1:8765"
    # Served beyond loopback (`autofa serve --host`): any Host on the port, same-origin only.
    remote_hosts: bool = False
    local_token: str = field(default_factory=lambda: service_token(create=True), repr=False)
    session_secret: str = ""
    max_metadata_bytes: int = 256 * 1024
    max_library_bytes: int = 32 * 1024 * 1024
    max_local_library_bytes: int = 128 * 1024 * 1024
    max_report_bytes: int = 128 * 1024 * 1024
    max_report_assets_bytes: int = 64 * 1024 * 1024
    max_report_assets_unpacked_bytes: int = 256 * 1024 * 1024
    # Separate from the small report upload: immutable trace stores can be large.
    max_trace_upload_bytes: int = field(default_factory=lambda: int(
        os.environ.get("AUTOFA_SERVER_MAX_TRACE_BYTES", 64 * 1024**3)))
    max_trace_unpacked_bytes: int = field(default_factory=lambda: int(
        os.environ.get("AUTOFA_SERVER_MAX_TRACE_UNPACKED_BYTES", 64 * 1024**3)))
    max_trace_files: int = 100_000
    max_trace_manifest_bytes: int = 16 * 1024 * 1024

    @property
    def database_path(self) -> Path:
        return self.data_dir / "autofa-library.db"
