# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`evalrdl-export <exporter> <tables dir> <out>`: one run's tables, a `<table>.parquet` per
declared table, written in an exporter's format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from evalrdl_schema.tables import RunTables

from . import PUBLIC


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalrdl-export")
    ap.add_argument("exporter", choices=sorted(PUBLIC))
    ap.add_argument("tables", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)

    exporter = PUBLIC[args.exporter]
    runs = args.tables / "runs.parquet"
    run = pl.read_parquet(runs).row(0, named=True) if runs.is_file() else {}
    n = exporter.to_file(RunTables(args.tables, exporter.reads), run, None, args.out)
    print(f"{args.tables}: {n} rollouts -> {args.out} ({exporter.title})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
