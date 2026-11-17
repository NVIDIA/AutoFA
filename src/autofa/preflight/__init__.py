# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional preflight checks run before analysis."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreflightResult:
    """Detector artifacts and execution status (findings alone are not failures)."""

    report: Path
    data: Path
    status: int = 0
