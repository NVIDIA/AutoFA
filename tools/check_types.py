#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type-check `src` with pyright and hold its exemption list to files that still need it.

`[tool.pyright]` in pyproject.toml checks every source file in basic mode, except the files in
`ignore` that had type errors when the check was added. This script runs pyright once with
`ignore` removed. It fails when a file outside the list has an error, and when a listed file
has none, so the list can only shrink.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def pyright_settings(pyproject: Path) -> dict[str, Any]:
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["pyright"]


def errors_by_file(report: dict[str, Any], root: Path) -> dict[str, list[dict[str, Any]]]:
    errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for diagnostic in report.get("generalDiagnostics", []):
        if diagnostic.get("severity") == "error":
            path = Path(diagnostic["file"]).resolve().relative_to(root.resolve()).as_posix()
            errors[path].append(diagnostic)
    return dict(errors)


def compare(
    errors: dict[str, list[dict[str, Any]]], ignored: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Return errors in files outside the exemption list, and listed files without errors."""
    listed = set(ignored)
    new = {path: found for path, found in sorted(errors.items()) if path not in listed}
    stale = sorted(listed - set(errors))
    return new, stale


def run_pyright(root: Path, settings: dict[str, Any]) -> dict[str, Any]:
    # Pyright resolves config paths relative to the config file, so it must live in the root.
    config = {key: value for key, value in settings.items() if key != "ignore"}
    with tempfile.NamedTemporaryFile(
        "w", dir=root, prefix=".pyright-ratchet-", suffix=".json", delete=False,
    ) as handle:
        json.dump(config, handle)
    try:
        result = subprocess.run(
            # Resolve imports against this interpreter's environment, not whatever is on PATH.
            [sys.executable, "-m", "pyright", "--outputjson", "--project", handle.name,
             "--pythonpath", sys.executable],
            cwd=root, capture_output=True, text=True, check=False,
        )
    finally:
        os.unlink(handle.name)
    # Exit status 1 means "errors found"; anything else means pyright itself failed.
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pyright failed ({result.returncode}):\n{result.stderr}{result.stdout}")
    return json.loads(result.stdout)


def _describe(path: str, diagnostic: dict[str, Any]) -> str:
    start = diagnostic.get("range", {}).get("start", {})
    line, column = start.get("line", 0) + 1, start.get("character", 0) + 1
    rule = f" ({diagnostic['rule']})" if diagnostic.get("rule") else ""
    message = diagnostic.get("message", "").splitlines()[0]
    return f"{path}:{line}:{column}: {message}{rule}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    settings = pyright_settings(args.root / "pyproject.toml")
    report = run_pyright(args.root, settings)
    new, stale = compare(errors_by_file(report, args.root), settings.get("ignore", []))
    for path, found in new.items():
        for diagnostic in found:
            print(_describe(path, diagnostic), file=sys.stderr)
    if new:
        print(
            f"\nPyright found type errors in {len(new)} checked file(s). Fix them; do not add "
            "files to [tool.pyright] ignore.",
            file=sys.stderr,
        )
    if stale:
        print(
            "\nThese files now type-check. Remove them from [tool.pyright] ignore in "
            "pyproject.toml:",
            file=sys.stderr,
        )
        for path in stale:
            print(f'  "{path}",', file=sys.stderr)
    if new or stale:
        return 1
    summary = report.get("summary", {})
    print(
        f"Pyright: {summary.get('filesAnalyzed', '?')} files analyzed; no errors outside the "
        f"{len(settings.get('ignore', []))} files listed in [tool.pyright] ignore."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
