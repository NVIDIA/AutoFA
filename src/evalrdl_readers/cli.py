# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`evalrdl-readers read <run folder> <tables dir>`: one run folder, read by the public readers and
written as one run's tables, a `<table>.parquet` per declared table (the `tables` extra)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalrdl-readers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    read = sub.add_parser("read", help="read a run folder into one run's tables")
    read.add_argument("folder", type=Path)
    read.add_argument("tables", type=Path)
    args = ap.parse_args(argv)

    from .convert import convert, write_tables
    from .readers import RUNDIR

    folder = args.folder.resolve()
    result = RUNDIR.read(str(folder))
    if not result.claimed:
        print(f"{folder}: no public reader claims this folder", file=sys.stderr)
        return 1
    conv = convert(result, path=str(folder), label=folder.name, readers=RUNDIR.readers)
    args.tables.mkdir(parents=True, exist_ok=True)
    write_tables(conv, args.tables)
    n = len(conv.tables.get("rollouts", ()))
    print(
        f"{folder.name}: read by {result.reader}, run {conv.run_id}, {n} rollouts -> {args.tables}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
