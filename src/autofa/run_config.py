# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autofa.config import Config, validate_min_failure_modes
from autofa.data.preparation import PreparedData


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _optional_mapping(value: Any, name: str) -> dict[str, Any]:
    return {} if value is None else _mapping(value, name)


def _keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} has unknown keys: {', '.join(unknown)}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_positive_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _optional_choice(value: Any, choices: set[str], name: str) -> str | None:
    if value is None:
        return None
    result = _text(value, name)
    if result not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return result


@dataclass(frozen=True)
class SourceSpec:
    evalrdl_url: str | None = None
    local_folder: Path | None = None
    refresh: bool = False

    def __post_init__(self) -> None:
        if (self.evalrdl_url is None) == (self.local_folder is None):
            raise ValueError("source must define exactly one of evalrdl_url or local_folder")


@dataclass(frozen=True)
class AutoFAExecution:
    mode: str
    auto_discover: bool = True
    exit_after_discovery: bool = False
    publish: bool = False
    require_llm_detectors: bool = False
    offer_efficiency_publication: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"session", "batch"}:
            raise ValueError("execution mode must be session or batch; "
                             "use the driver for campaigns")


@dataclass(frozen=True)
class AutoFAConfig:
    settings: Config
    source: SourceSpec | PreparedData
    execution: AutoFAExecution

    def __post_init__(self) -> None:
        if not isinstance(self.source, (SourceSpec, PreparedData)):
            raise TypeError("source must be a SourceSpec or PreparedData")
        if (
            isinstance(self.settings.max_rounds, bool)
            or not isinstance(self.settings.max_rounds, int)
            or self.settings.max_rounds <= 0
        ):
            raise ValueError("max_rounds must be greater than zero")
        if (
            isinstance(self.settings.top_x, bool)
            or not isinstance(self.settings.top_x, int)
            or self.settings.top_x <= 0
        ):
            raise ValueError("top_x must be greater than zero")
        validate_min_failure_modes(self.settings.min_failure_modes)
        if self.execution.mode == "batch" and (
            "agent" not in self.settings.analysis_outputs
        ):
            raise ValueError(f"{self.execution.mode} execution requires agent analysis")
