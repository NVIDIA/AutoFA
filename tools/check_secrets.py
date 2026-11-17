# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail when a file contains a potential secret that `.secrets.baseline` does not list.

detect-secrets matches baseline entries by file, secret type and secret hash, not by line,
so moving a listed line needs no baseline change. Unlike `detect-secrets-hook`, this check
never rewrites the baseline to refresh its line numbers; that rewrite would fail every
commit that shifts a line in one of the many test files with synthetic credentials.

With no file arguments, every tracked file is checked. Refresh the baseline after a
reviewed change with `detect-secrets scan --baseline .secrets.baseline`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from detect_secrets import SecretsCollection
from detect_secrets.core import baseline

BASELINE = ".secrets.baseline"


def _tracked_files() -> list[str]:
    """Paths Git tracks in the current repository, including intent-to-add files."""
    result = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True)
    return [name.decode() for name in result.stdout.split(b"\0") if name]


def new_secrets(filenames: list[str], baseline_path: str = BASELINE) -> SecretsCollection:
    """Scan `filenames` with the baseline's plugins and filters; return unlisted findings."""
    # Loading the baseline also applies its plugin and filter settings to the scan below;
    # scan_file then skips files matching the baseline's --exclude-files pattern.
    known = baseline.load(baseline.load_from_file(baseline_path), baseline_path)
    found = SecretsCollection()
    skip = Path(baseline_path).resolve()
    for filename in filenames:
        path = Path(filename)
        if path.is_file() and not path.is_symlink() and path.resolve() != skip:
            found.scan_file(filename)
    return found - known


def main(argv: list[str] | None = None) -> int:
    """Return 0 if every finding is in the baseline; otherwise print the others and return 1."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("filenames", nargs="*", help="files to check (default: tracked files)")
    args = parser.parse_args(argv)

    unlisted = new_secrets(args.filenames or _tracked_files(), args.baseline)
    findings = sorted((filename, secret.line_number, secret.type)
                      for filename, secret in unlisted)
    if not findings:
        return 0
    print("Potential secrets not listed in .secrets.baseline:", file=sys.stderr)
    for filename, line, kind in findings:
        print(f"  {filename}:{line}: {kind}", file=sys.stderr)
    print(
        "\nRemove real credentials and rotate them. For a synthetic value, add an\n"
        "`# pragma: allowlist secret` comment to the line, or review it and refresh the\n"
        "baseline; see 'Secrets scan' in CONTRIBUTING.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
