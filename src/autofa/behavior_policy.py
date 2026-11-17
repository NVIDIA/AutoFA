# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared identities for behavioral validation policies.

New analyses use the configured default policy, and selecting a new policy never
relabels old evidence. Scores without a recorded policy (never-judged functions) display
and age as causal; review reuse, processing retries and repair provenance require the
recorded policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autofa.defaults import setting_value

BEHAVIOR_STRICTNESSES = ("behavioral", "contextual", "causal")
STRICTNESS_LABELS = {
    "behavioral": "Behavior match",
    "contextual": "Failure relevance",
    "causal": "Causal contribution",
}


def validate_behavior_strictness(value: Any) -> str:
    if not isinstance(value, str) or value not in BEHAVIOR_STRICTNESSES:
        raise ValueError("behavior_strictness must be behavioral, contextual, or causal")
    return value


def strictness_for_config(config: Any = None) -> str:
    return validate_behavior_strictness(setting_value(config, "behavior_strictness"))


def strictness_for_scores(scores: Mapping[str, Any] | None) -> str:
    return validate_behavior_strictness((scores or {}).get("judge_behavior_strictness", "causal"))


def acceptance_requirement_id(strictness: str) -> str | None:
    return {"behavioral": None, "contextual": "failure_relevance",
            "causal": "failure_contribution"}[validate_behavior_strictness(strictness)]
