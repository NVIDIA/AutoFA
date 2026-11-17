# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One registry for detector settings, standalone commands and analysis runs.

Each entrypoint exposes ``main(argv)`` and
``run_dataset(dataset, cfg, output, *, sources) -> PreflightResult``.
Implementations are imported only when selected.
"""

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class PreflightDetector:
    name: str
    label: str
    description: str
    entrypoint: str

    def load(self) -> ModuleType:
        return import_module(self.entrypoint)


DETECTORS = (
    PreflightDetector(
        "cheating", "Cheating", "Screen and review rollouts for cheating",
        "autofa.preflight.cheating.cli",
    ),
)


def detectors() -> tuple[PreflightDetector, ...]:
    return DETECTORS


def validate_detectors(value: list[str]) -> list[str]:
    names = {detector.name for detector in detectors()}
    if not isinstance(value, list) or any(
        not isinstance(name, str) or name not in names for name in value
    ):
        raise ValueError(
            "preflight_detectors must be a list containing only: " + ", ".join(sorted(names))
        )
    return list(dict.fromkeys(value))
