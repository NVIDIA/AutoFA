# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact trust policy shared by taxonomy outcome workflows."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_REFERENCE_ROLES = {
    "answer",
    "golden",
    "ground_truth",
    "reference",
    "solution",
    "target",
}
_REFERENCE_NAMES = {
    "golden_patch",
    "reference_patch",
}
_REFERENCE_FIELDS = _REFERENCE_NAMES | {
    "expected_answer",
    "golden",
    "ground_truth",
    "reference",
    "reference_answer",
    "solution",
    "target",
}


def _reference_token(value: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or "").strip())
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def without_reference_fields(value: Any) -> Any:
    """Recursively remove evaluator-only reference fields from JSON-like values."""
    if isinstance(value, dict):
        return {
            key: without_reference_fields(item)
            for key, item in value.items()
            if _reference_token(key) not in _REFERENCE_FIELDS
        }
    if isinstance(value, list):
        return [without_reference_fields(item) for item in value]
    return value


def is_reference_artifact(artifact: Mapping[str, Any]) -> bool:
    """Whether an artifact contains evaluator-only reference material."""
    role = _reference_token(artifact.get("role"))
    name = _reference_token(artifact.get("name"))
    return role in _REFERENCE_ROLES or name in _REFERENCE_NAMES


def is_candidate_patch_artifact(artifact: Mapping[str, Any]) -> bool:
    """Whether an artifact is the exact portable candidate patch."""
    return (
        artifact.get("role") == "output"
        and artifact.get("kind") == "patch"
        and artifact.get("name") == "model.patch"
    )


def is_outcome_text_artifact(artifact: Mapping[str, Any]) -> bool:
    """Whether artifact text may be injected automatically as outcome evidence."""
    if is_reference_artifact(artifact):
        return False
    return artifact.get("role") == "verifier" or is_candidate_patch_artifact(artifact)
