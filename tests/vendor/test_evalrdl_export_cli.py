# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`evalrdl-export`: one run's tables in a public exporter's format."""

from __future__ import annotations

import sys
import zipfile

import pytest

from evalrdl_export.cli import main
from evalrdl_export.conformance import sample_rows, write_partition
from evalrdl_schema import CURRENT


def test_a_run_s_tables_export_to_a_trace_zip(tmp_path, capsys):
    tables = write_partition(tmp_path / "tables", CURRENT, sample_rows())
    assert main(["trace", str(tables), str(tmp_path / "trace.zip")]) == 0
    with zipfile.ZipFile(tmp_path / "trace.zip") as z:
        assert z.testzip() is None and "manifest.json" in z.namelist()
    assert "3 rollouts" in capsys.readouterr().out


def test_an_unknown_exporter_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        main(["no-such-format", str(tmp_path), str(tmp_path / "x")])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
