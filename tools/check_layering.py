#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail when a lower AutoFA layer imports an orchestration or presentation layer.

Every import counts, including imports deferred inside functions and imports guarded by
`TYPE_CHECKING`: a deferred import still couples the layers and still forms an import cycle.
Today's violations are listed in `tools/layering_exceptions.txt`. The check fails on a new
violation and on an exception that no longer matches an import, so the list can only shrink.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
EXCEPTIONS_FILE = ROOT / "tools" / "layering_exceptions.txt"


@dataclass(frozen=True)
class Contract:
    name: str
    layers: tuple[str, ...]
    forbidden: tuple[str, ...]


CONTRACTS = (
    Contract(
        name="core, data and metrics stay below the runtime, session and interfaces",
        layers=("autofa.core", "autofa.data", "autofa.metrics"),
        forbidden=("autofa.mcp", "autofa.ui", "autofa.cli", "autofa.session"),
    ),
)


@dataclass(frozen=True, order=True)
class Import:
    importer: str
    imported: str
    path: str
    line: int

    @property
    def edge(self) -> str:
        return f"{self.importer} -> {self.imported}"


def module_name(path: Path, source_root: Path) -> str:
    parts = list(path.relative_to(source_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _within(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def _resolve_relative(importer: str, is_package: bool, level: int, module: str | None) -> str:
    base = importer.split(".")
    # A module's level-1 anchor is its package; a package's is the package itself.
    drop = level - 1 if is_package else level
    if drop >= len(base):
        raise ValueError(f"relative import beyond top-level package in {importer}")
    return ".".join([*base[: len(base) - drop], *([module] if module else [])])


def module_imports(path: Path, source_root: Path, modules: set[str]) -> Iterator[Import]:
    """Yield every first-party import in `path`, wherever it appears in the module."""
    importer = module_name(path, source_root)
    is_package = path.name == "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_path = path.relative_to(source_root.parent).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = (
                _resolve_relative(importer, is_package, node.level, node.module)
                if node.level
                else node.module or ""
            )
            # `from package import submodule` imports the submodule, not just the package.
            targets = [
                f"{base}.{alias.name}" if f"{base}.{alias.name}" in modules else base
                for alias in node.names
            ]
        else:
            continue
        for target in dict.fromkeys(targets):
            if target in modules or any(_within(m, target) for m in modules):
                yield Import(importer, target, relative_path, node.lineno)


def violations(source_root: Path, contracts: Iterable[Contract] = CONTRACTS) -> list[Import]:
    paths = sorted(source_root.rglob("*.py"))
    modules = {module_name(path, source_root) for path in paths}
    contracts = tuple(contracts)
    found = set()
    for path in paths:
        importer = module_name(path, source_root)
        applicable = [c for c in contracts if any(_within(importer, p) for p in c.layers)]
        if not applicable:
            continue
        for item in module_imports(path, source_root, modules):
            if any(_within(item.imported, f) for c in applicable for f in c.forbidden):
                found.add(item)
    return sorted(found)


def read_exceptions(path: Path) -> list[str]:
    if not path.exists():
        return []
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.append(" ".join(line.split()))
    return entries


def check(source_root: Path, exceptions_file: Path) -> tuple[list[Import], list[str]]:
    """Return imports that break a contract without an exception, and stale exceptions."""
    found = violations(source_root)
    allowed = set(read_exceptions(exceptions_file))
    new = [item for item in found if item.edge not in allowed]
    stale = sorted(allowed - {item.edge for item in found})
    return new, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--exceptions", type=Path, default=EXCEPTIONS_FILE)
    parser.add_argument(
        "--list", action="store_true",
        help="print every current violation as an exception line and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for edge in sorted({item.edge for item in violations(args.source_root)}):
            print(edge)
        return 0

    new, stale = check(args.source_root, args.exceptions)
    for item in new:
        print(f"{item.path}:{item.line}: {item.edge}", file=sys.stderr)
    if new:
        print(
            "\nThese imports cross a layer boundary. Move the shared code into a lower layer, "
            "pass the dependency in, or depend on a Protocol defined in the lower layer.",
            file=sys.stderr,
        )
        for contract in CONTRACTS:
            print(
                f"Contract: {contract.name}: {', '.join(contract.layers)} must not import "
                f"{', '.join(contract.forbidden)}.",
                file=sys.stderr,
            )
    if stale:
        print(
            f"\n{args.exceptions.name} lists imports that no longer exist. Remove these lines:",
            file=sys.stderr,
        )
        for edge in stale:
            print(f"  {edge}", file=sys.stderr)
    if new or stale:
        return 1
    total = len(read_exceptions(args.exceptions))
    print(f"Layering contracts hold ({total} listed exceptions remain).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
