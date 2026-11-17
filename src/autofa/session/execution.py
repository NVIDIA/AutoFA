# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the installed code that actually starts a session, independently of cwd."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def record_execution(run_dir: Path) -> dict:
    source = Path(__file__).resolve().parents[1]
    checkout = source.parent.parent
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "source_path": str(source),
        "git_revision": None,
        "git_dirty": None,
    }
    # A wheel installation need not have a Git checkout. Never infer the code
    # revision from the user's working directory.
    if (checkout / ".git").exists():
        try:
            metadata["git_revision"] = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip()
            metadata["git_dirty"] = bool(subprocess.check_output(
                ["git", "-C", str(checkout), "status", "--porcelain"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip())
        except (OSError, subprocess.SubprocessError):
            pass
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "execution_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata) + "\n")
    return metadata
