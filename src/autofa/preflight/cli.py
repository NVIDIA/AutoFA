# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch ``autofa preflight <detector>`` without loading unused detectors."""

import argparse
import sys
from collections.abc import Sequence

from autofa.preflight.registry import detectors


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "classifiers":
        from autofa.preflight.classifiers import main as classifiers_main

        return classifiers_main(args[1:])
    available = {detector.name: detector for detector in detectors()}
    if args and args[0] in available:
        return available[args[0]].load().main(args[1:])
    parser = argparse.ArgumentParser(
        prog="autofa preflight", description="Run an optional preflight check.",
    )
    commands = parser.add_subparsers(dest="detector", required=True)
    commands.add_parser("classifiers", help="Inspect/configure permitted detector profiles")
    for detector in available.values():
        commands.add_parser(detector.name, help=detector.description)
    parser.parse_args(args)
    return 0
