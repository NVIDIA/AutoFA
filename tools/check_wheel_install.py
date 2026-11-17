# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check an installed AutoFA wheel: modules, package data, imports and console scripts.

The test suite runs against the source checkout with the dev extra, so it cannot see a
module or data file the wheel leaves out, or a module that imports a dev-only package.
Run this with the interpreter of a fresh environment that has only the wheel installed,
from the repository root, in isolated mode so the checkout's `src` is not importable:

    uv build --wheel --out-dir dist
    uv venv /tmp/autofa-wheel && uv pip install --python /tmp/autofa-wheel/bin/python dist/*.whl
    /tmp/autofa-wheel/bin/python -I tools/check_wheel_install.py
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def source_packages(src: Path) -> list[str]:
    """Top-level packages that setuptools discovers under `src`."""
    return sorted(path.name for path in src.iterdir() if (path / "__init__.py").is_file())


def missing_modules(source: Path, installed: Path) -> list[str]:
    """Python files under a source package that the installed package lacks."""
    return [
        f"{source.name}/{path.relative_to(source).as_posix()} is missing from the wheel"
        for path in sorted(source.rglob("*.py"))
        if "__pycache__" not in path.parts and not (installed / path.relative_to(source)).is_file()
    ]


def missing_package_data(source: Path, installed: Path, patterns: list[str]) -> list[str]:
    """Problems with one package's `package-data` patterns in an installed copy."""
    problems = []
    for pattern in patterns:
        expected = sorted(path.relative_to(source) for path in source.glob(pattern)
                          if path.is_file())
        if not expected:
            problems.append(f"{source.name}: package-data pattern {pattern!r} matches no file")
        problems.extend(f"{source.name}/{relative.as_posix()} is missing from the wheel"
                        for relative in expected if not (installed / relative).is_file())
    return problems


def import_failures(packages: list[str]) -> tuple[int, list[str]]:
    """Import every module of `packages`; return the count and the failures."""
    failures: list[str] = []

    def record(name: str) -> None:
        """Record a package that `walk_packages` failed to import while listing modules."""
        failures.append(f"{name}: {sys.exc_info()[1]!r}")

    count = 0
    for package in packages:
        module = importlib.import_module(package)
        for info in pkgutil.walk_packages(module.__path__, f"{package}.", onerror=record):
            count += 1
            try:
                importlib.import_module(info.name)
            except Exception as exc:
                failures.append(f"{info.name}: {exc!r}")
    return count, failures


def console_script_failures(scripts: list[str]) -> list[str]:
    """Run each console script's `--help` from the environment's bin directory."""
    bin_dir = Path(sys.executable).parent
    failures = []
    with tempfile.TemporaryDirectory() as home:
        env = {**os.environ, "AUTOFA_HOME": home, "NO_COLOR": "1"}
        for name in scripts:
            result = subprocess.run([str(bin_dir / name), "--help"], cwd=home, env=env,
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                failures.append(f"{name} --help exited {result.returncode}: "
                                f"{result.stderr.strip()[-500:]}")
    return failures


def main() -> int:
    """Return 0 if the wheel passes, 1 if it is incomplete, 2 if `src` was imported instead."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = source_packages(SRC)
    installed: dict[str, Path] = {}
    for package in packages:
        location = Path(importlib.import_module(package).__file__).resolve().parent
        if location.is_relative_to(SRC):
            print(f"{package} was imported from the checkout ({location}). Run this with the "
                  "wheel environment's interpreter and `python -I`.", file=sys.stderr)
            return 2
        installed[package] = location

    problems = []
    for package in packages:
        problems += missing_modules(SRC / package, installed[package])
    for package, patterns in project["tool"]["setuptools"]["package-data"].items():
        problems += missing_package_data(SRC / package, installed[package], patterns)
    count, failures = import_failures(packages)
    problems += failures
    scripts = list(project["project"]["scripts"])
    problems += console_script_failures(scripts)

    if problems:
        print("The installed wheel is incomplete:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"Wheel OK on Python {sys.version.split()[0]}: {count} modules imported, package data "
          f"present, {', '.join(scripts)} --help ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
