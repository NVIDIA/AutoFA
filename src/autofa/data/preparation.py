# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared result of preparing data for an AutoFA workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreparedData:
    """Analysis-ready data plus the source selection that produced it."""

    parquet_paths: list[Path]
    model_id: str
    benchmarks: list[str]
    source_paths: list[str] = field(default_factory=list)
    source_kind: str = "parquet"
    source_harness: str = "unknown"
    reference_data: list[dict[str, str]] = field(default_factory=list)
    selection_id: str | None = None
