# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check source notices and release-license metadata without changing files."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".js", ".css", ".html", ".sh", ".toml", ".yml", ".yaml"}
SOURCE_NAMES = {".gitignore", "MANIFEST.in"}
NVIDIA_COPYRIGHT = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) 20\d{2}(?:-20\d{2})? "
    r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)
APACHE_NOTICE = "Licensed under the Apache License, Version 2.0"


def needs_header(path: str) -> bool:
    if path.startswith("eval_data/") and Path(path).suffix in {".yml", ".yaml"}:
        return False
    if path.startswith("third_party/licenses/"):
        return False
    return Path(path).suffix in SOURCE_SUFFIXES or Path(path).name in SOURCE_NAMES


def header_errors(path: str, content: str) -> list[str]:
    if not needs_header(path):
        return []
    header = "\n".join(content.splitlines()[:12])
    errors = []
    if "NVIDIA" not in header:
        # OSRB permits SPDX-only notices for NVIDIA-authored files only.
        # Keep external authors' full Apache notices and their ownership intact.
        if not re.search(r"Copyright\s+\S", header, re.I) or APACHE_NOTICE not in header:
            errors.append(f"{path}: external source needs its full copyright/license notice")
        return errors
    if not re.search(r"SPDX-FileCopyrightText: \S", header):
        errors.append(f"{path}: missing copyright notice")
    if not re.search(r"SPDX-License-Identifier: Apache-2\.0(?:\s|$)", header):
        errors.append(f"{path}: missing Apache-2.0 source notice")
    if not NVIDIA_COPYRIGHT.search(header):
        errors.append(f"{path}: NVIDIA copyright must use the approved header")
    return errors


def working_tree_paths(root: Path) -> list[str]:
    """Inspect additions and modifications, excluding files removed in this checkout."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    ).decode().split("\0")
    deleted = subprocess.check_output(
        ["git", "ls-files", "-z", "--deleted"], cwd=root,
    ).decode().split("\0")
    return sorted(set(tracked) - set(deleted) - {""})


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = []
    checked = 0
    for name in working_tree_paths(root):
        if needs_header(name):
            checked += 1
            errors.extend(header_errors(name, (root / name).read_text(encoding="utf-8")))
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if project.get("license") != "CC-BY-4.0 AND Apache-2.0":
        errors.append("pyproject.toml: expected CC-BY-4.0 AND Apache-2.0")
    required = {
        "LICENSE", "NOTICE", "CONTRIBUTING.md", "THIRD_PARTY_NOTICES.md",
        "third_party/evalrdl/README.md", "third_party/dependency-notices.json",
        "third_party/NATIVE_BINARY_REVIEW.md",
    }
    for name in sorted(required):
        if not (root / name).is_file() or not (root / name).stat().st_size:
            errors.append(f"{name}: missing or empty release notice")
        if name not in project.get("license-files", []):
            errors.append(f"pyproject.toml: {name} must be packaged")
    if "third_party/licenses/**/*" not in project.get("license-files", []):
        errors.append("pyproject.toml: third-party license texts must be packaged")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Checked {checked} source files and release-license metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
