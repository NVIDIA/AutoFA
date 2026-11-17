# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality ratchets: layering contracts and lint exceptions can shrink but never grow."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from tools import check_layering, check_types

ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relative: str, text: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _synthetic_tree(root: Path) -> Path:
    src = root / "src"
    for package in ("autofa", "autofa/core", "autofa/data", "autofa/metrics", "autofa/mcp",
                    "autofa/session", "autofa/ui", "autofa/util"):
        _write(src, f"{package}/__init__.py")
    _write(src, "autofa/cli.py")
    _write(src, "autofa/mcp/runtime.py")
    _write(src, "autofa/session/runner.py")
    _write(src, "autofa/ui/verdicts.py")
    _write(src, "autofa/util/atomic.py")
    _write(src, "autofa/core/allowed.py", "import os\nfrom autofa.util.atomic import x\n")
    _write(src, "autofa/core/deferred.py",
           "def load():\n    from autofa.mcp.runtime import get_runtime\n")
    _write(src, "autofa/data/relative.py", "from ..session import runner\n")
    _write(src, "autofa/data/package.py", "from autofa import cli, util\n")
    _write(src, "autofa/metrics/__init__.py", "from ..ui.verdicts import VERDICT\n")
    _write(src, "autofa/metrics/typing_only.py",
           "from typing import TYPE_CHECKING\n"
           "if TYPE_CHECKING:\n    import autofa.session.runner\n")
    # Upper layers may import each other; only the lower layers are constrained.
    _write(src, "autofa/session/uses_ui.py", "from autofa.ui import verdicts\n")
    return src


def test_layering_counts_deferred_relative_package_and_type_only_imports(tmp_path):
    src = _synthetic_tree(tmp_path)

    found = [(item.edge, item.path, item.line) for item in check_layering.violations(src)]

    assert found == [
        ("autofa.core.deferred -> autofa.mcp.runtime", "src/autofa/core/deferred.py", 2),
        ("autofa.data.package -> autofa.cli", "src/autofa/data/package.py", 1),
        ("autofa.data.relative -> autofa.session.runner", "src/autofa/data/relative.py", 1),
        ("autofa.metrics -> autofa.ui.verdicts", "src/autofa/metrics/__init__.py", 1),
        ("autofa.metrics.typing_only -> autofa.session.runner",
         "src/autofa/metrics/typing_only.py", 3),
    ]


def test_layering_exceptions_only_shrink(tmp_path, capsys):
    src = _synthetic_tree(tmp_path)
    exceptions = tmp_path / "exceptions.txt"
    exceptions.write_text(
        "# comment\n"
        "autofa.core.deferred -> autofa.mcp.runtime\n"
        "autofa.data.package  ->  autofa.cli   # spacing and comments are ignored\n"
        "autofa.data.relative -> autofa.session.runner\n"
        "autofa.metrics -> autofa.ui.verdicts\n"
        "autofa.metrics.typing_only -> autofa.session.runner\n"
    )
    assert check_layering.main(["--source-root", str(src), "--exceptions", str(exceptions)]) == 0

    # A new crossing fails and names the file and line.
    _write(src, "autofa/core/new.py", "\n\nimport autofa.session.runner as runner\n")
    assert check_layering.main(["--source-root", str(src), "--exceptions", str(exceptions)]) == 1
    assert "src/autofa/core/new.py:3: autofa.core.new -> autofa.session.runner" in (
        capsys.readouterr().err)

    # Removing an import without removing its exception also fails, so the list shrinks.
    (src / "autofa/core/new.py").unlink()
    _write(src, "autofa/core/deferred.py", "def load():\n    return None\n")
    assert check_layering.main(["--source-root", str(src), "--exceptions", str(exceptions)]) == 1
    err = capsys.readouterr().err
    assert "Remove these lines" in err
    assert "autofa.core.deferred -> autofa.mcp.runtime" in err


def test_repository_layering_matches_its_exception_list():
    new, stale = check_layering.check(check_layering.SOURCE_ROOT, check_layering.EXCEPTIONS_FILE)

    assert [f"{item.path}:{item.line}: {item.edge}" for item in new] == []
    assert stale == []


def test_ruff_per_file_ignores_name_existing_files_that_still_need_them():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]
    assert all((ROOT / path).is_file() for path in ignores), "entries must be single files"

    # Lint the listed files with the exemptions removed: every exempted code must still occur.
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--exit-zero", "--output-format", "json",
         "--config", "lint.per-file-ignores = {}", *ignores],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    found = {
        (Path(item["filename"]).resolve().relative_to(ROOT).as_posix(), item["code"])
        for item in json.loads(result.stdout)
    }
    stale = [f"{path}: {code}" for path, codes in ignores.items() for code in codes
             if (path, code) not in found]
    assert stale == [], "remove these per-file-ignores from pyproject.toml"


def test_type_check_exemptions_only_shrink(tmp_path):
    def error(path, rule="reportArgumentType"):
        return {"file": str(tmp_path / path), "severity": "error", "rule": rule,
                "message": "bad", "range": {"start": {"line": 0, "character": 0}}}

    report = {"generalDiagnostics": [
        error("src/listed.py"), error("src/new.py"),
        {**error("src/warned.py"), "severity": "warning"},
    ]}
    errors = check_types.errors_by_file(report, tmp_path)
    assert sorted(errors) == ["src/listed.py", "src/new.py"]

    new, stale = check_types.compare(errors, ["src/listed.py", "src/fixed.py"])
    assert list(new) == ["src/new.py"]
    assert stale == ["src/fixed.py"]


def test_type_check_exemptions_are_files_under_src():
    ignored = check_types.pyright_settings(ROOT / "pyproject.toml")["ignore"]
    assert ignored == sorted(set(ignored))
    assert all(path.startswith("src/") and (ROOT / path).is_file() for path in ignored)
