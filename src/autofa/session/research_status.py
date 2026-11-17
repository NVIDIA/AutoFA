# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved task-coverage progress, without an execution loop."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from autofa.util.atomic import atomic_write_json


def write_research_status(run_dir: Path, status: dict) -> None:
    status["updated_at"] = datetime.now(UTC).isoformat()
    path = run_dir / "research_status.json"
    atomic_write_json(path, status)


def read_research_status(run_dir: Path) -> dict:
    try:
        data = json.loads((run_dir / "research_status.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
