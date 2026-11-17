# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge response and classification result."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ProviderResponse:
    """Validated judge answers."""

    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)


Decision = Literal["accept", "review"]
FirstPassMethod = Literal["llm", "agentic"]
SecondPassMethod = Literal["off", "llm", "agentic"]


@dataclass(frozen=True)
class AnalysisResult:
    """One classification with its evidence trail."""

    incident_id: str
    incident_sha256: str
    taxonomy_id: str
    taxonomy_name: str
    taxonomy_version: str
    taxonomy_sha256: str
    protocol_version: str
    model: str
    decision: Decision
    decision_reasons: tuple[str, ...]
    selected_leaf_id: str | None
    selected_code: str | None
    selected_label: str
    selected_path: tuple[str, ...]
    selected_path_ids: tuple[str, ...]
    selected_path_codes: tuple[str | None, ...]
    alternatives: tuple[dict[str, Any], ...]
    owner: str | None
    remediation: tuple[str, ...]
    related: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]
    diagnosis: dict[str, Any]
    abstention_reason: str | None
    provider_calls: int
    usage: dict[str, int]
    first_pass: dict[str, Any] = field(default_factory=dict)
    second_pass: dict[str, Any] = field(default_factory=dict)
    windowed: bool = False
    window_count: int = 1
    window_sha256s: tuple[str, ...] = ()
    window_receipts: tuple[dict[str, Any], ...] = ()
    aggregation: str | None = None
    episodes: tuple[dict[str, Any], ...] = ()
    selected_episode_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
