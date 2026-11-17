# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository checks that CI runs outside pytest: the wheel install and the secrets scan."""

import re
import secrets
import string
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tools import check_wheel_install

ROOT = Path(__file__).resolve().parents[1]
# Synthetic credentials and hosts, assembled so this file holds no literal one.
KEY = "pass" + "word"
ASSIGNMENT = f'{KEY} = "Zx81QpLm73kWv"\n'
OTHER_ASSIGNMENT = f'{KEY} = "Rt52NbWe94cXq"\n'
DOMAIN = "nvidia" + ".com"


def _write(root: Path, relative: str, text: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_wheel_check_reports_modules_and_data_the_wheel_leaves_out(tmp_path):
    source, installed = tmp_path / "src" / "pkg", tmp_path / "site" / "pkg"
    for root in (source, installed):
        _write(root, "__init__.py")
        _write(root, "templates/kept.html")
    _write(source, "nested/module.py")
    _write(source, "templates/dropped.html")

    assert check_wheel_install.missing_modules(source, installed) == [
        "pkg/nested/module.py is missing from the wheel"]
    assert check_wheel_install.missing_package_data(
        source, installed, ["templates/*.html", "static/*.js"]) == [
        "pkg/templates/dropped.html is missing from the wheel",
        "pkg: package-data pattern 'static/*.js' matches no file",
    ]


def test_wheel_check_reports_codec_bundling_distributions():
    assert check_wheel_install.codec_distributions(("pytest", "autofa-no-such-dist")) == [
        f"pytest {pytest.__version__} is installed; AutoFA must not depend on it"]


def test_no_requirement_installs_codec_bundling_packages():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = project["dependencies"] + [
        requirement for extra in project["optional-dependencies"].values()
        for requirement in extra]
    names = {re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower().replace("_", "-")
             for requirement in requirements}
    assert names.isdisjoint(check_wheel_install.CODEC_DISTRIBUTIONS)


def test_every_package_data_pattern_matches_a_source_file():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]
    src = ROOT / "src"
    assert set(package_data) <= set(check_wheel_install.source_packages(src))
    for package, patterns in package_data.items():
        assert check_wheel_install.missing_package_data(
            src / package, src / package, patterns) == []


def _check_secrets(cwd: Path, *files: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ROOT / "tools" / "check_secrets.py"), *files],
                          cwd=cwd, capture_output=True, text=True, check=False)


def _baseline(cwd: Path, *files: str) -> None:
    scan = subprocess.run([sys.executable, "-m", "detect_secrets", "scan", *files], cwd=cwd,
                          capture_output=True, text=True, check=True)
    (cwd / ".secrets.baseline").write_text(scan.stdout)


def test_secrets_check_accepts_listed_secrets_that_moved_without_rewriting_baseline(tmp_path):
    _write(tmp_path, "settings.py", ASSIGNMENT)
    _baseline(tmp_path, "settings.py")
    baseline = (tmp_path / ".secrets.baseline").read_bytes()
    _write(tmp_path, "settings.py", "# moved down\n\n" + ASSIGNMENT)

    result = _check_secrets(tmp_path, "settings.py", ".secrets.baseline")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".secrets.baseline").read_bytes() == baseline


def test_secrets_check_reports_unlisted_secrets(tmp_path):
    _write(tmp_path, "settings.py", ASSIGNMENT)
    _baseline(tmp_path, "settings.py")
    _write(tmp_path, "settings.py", ASSIGNMENT + OTHER_ASSIGNMENT)
    _write(tmp_path, "clean.py", "value = 1\n")

    result = _check_secrets(tmp_path, "settings.py", "clean.py")

    assert result.returncode == 1
    assert "settings.py:2: Secret Keyword" in result.stderr
    assert "settings.py:1" not in result.stderr


def test_repository_baseline_flags_nvidia_keys_and_non_public_nvidia_hosts(tmp_path):
    settings = tmp_path / "settings.toml"
    settings.write_text(
        f'key = "nvapi-{secrets.token_urlsafe(48)}"\n'
        f'data_api_base_url = "https://runs.example-internal.{DOMAIN}"\n'
        f'docs_url = "https://docs.{DOMAIN}/autofa"\n'
        f"contact = maintainer@{DOMAIN}\n"
        f'docs = "https://example-internal.{DOMAIN}/x"  # pragma: allowlist secret\n'
    )

    findings = _repository_findings(settings)

    # The random key may also trip the built-in entropy detector.
    assert {("1", "NVIDIA API Key"), ("2", "Non-public NVIDIA Hostname")} <= findings
    assert {line for line, _ in findings} == {"1", "2"}


def test_repository_baseline_flags_secrets_passed_as_call_arguments(tmp_path):
    # detect-secrets' default is_indirect_reference filter drops every finding on a line
    # shaped like `x = call("...")`, whatever the detector; the baseline turns it off.
    token = "ghp_" + "".join(secrets.choice(string.ascii_letters + string.digits)
                             for _ in range(36))
    nvidia_key = "nvapi-" + secrets.token_urlsafe(48)
    module = tmp_path / "clients.py"
    module.write_text(
        f'client = Github("{token}")\n'
        f'llm = OpenAI(api_key="{nvidia_key}")\n'
        f'api = DataApiClient("https://runs.example-internal.{DOMAIN}")\n'
    )

    findings = _repository_findings(module)

    assert {("1", "GitHub Token"), ("2", "NVIDIA API Key"),
            ("3", "Non-public NVIDIA Hostname")} <= findings


def _repository_findings(path: Path) -> set[tuple[str, str]]:
    """Check `path` against the repository baseline; return (line, type) pairs reported."""
    # Run from the root so the repository baseline loads tools/secret_detectors.py.
    result = subprocess.run([sys.executable, "tools/check_secrets.py", str(path)], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr
    return {tuple(line.strip().removeprefix(f"{path}:").split(": ", 1))
            for line in result.stderr.splitlines() if str(path) in line}
