# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared outcome rules for AutoFA workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

Outcome = Literal["pass", "fail", "unscored"]


def is_failure_reward(reward: Any) -> bool:
    """Return whether a recorded reward is a numeric failure."""
    return (
        not isinstance(reward, bool)
        and isinstance(reward, (int, float))
        and not (isinstance(reward, float) and math.isnan(reward))
        and reward <= 0
    )


def is_healthy_scored(row: Mapping[str, Any]) -> bool:
    """Return whether a rollout has a usable numeric score."""
    reward = row.get("reward")
    return (
        row.get("status") == "ok"
        and row.get("ingest_status") == "ok"
        and row.get("error") is None
        and row.get("is_scored") is True
        and not isinstance(reward, bool)
        and isinstance(reward, (int, float))
        and not (isinstance(reward, float) and math.isnan(reward))
    )


def is_scored_failure(row: Mapping[str, Any]) -> bool:
    """Return the failure rule used by cohort selection and taxonomy."""
    reward = row.get("reward")
    return is_healthy_scored(row) and is_failure_reward(reward)


def discovery_is_correct(row: Mapping[str, Any]) -> bool:
    """Return the correctness rule used by discovery."""
    if row.get("is_correct") is not None:
        return bool(row["is_correct"])
    accuracy = row.get("accuracy")
    if accuracy is not None:
        try:
            return float(accuracy) >= 0.5
        except (TypeError, ValueError):
            return bool(accuracy)
    reward = row.get("reward")
    return (
        not isinstance(reward, bool)
        and isinstance(reward, (int, float))
        and not (isinstance(reward, float) and math.isnan(reward))
        and reward >= 0.5
    )


def is_discovery_success(row: Mapping[str, Any]) -> bool:
    """Return whether a canonical rollout is a usable discovery negative."""
    return is_healthy_scored(row) and discovery_is_correct(row)


def efficiency_outcome(reward: Any) -> Outcome:
    """Return the reward grouping used by efficiency reports."""
    if reward is None or isinstance(reward, bool):
        return "unscored"
    try:
        value = float(reward)
    except (TypeError, ValueError):
        return "unscored"
    if math.isnan(value):
        return "unscored"
    return "pass" if value >= 1 else "fail"


def population_outcome(reward: Any) -> Outcome:
    """Group source-population rewards using a zero threshold."""
    if reward is None or isinstance(reward, bool):
        return "unscored"
    try:
        value = float(reward)
    except (TypeError, ValueError):
        return "unscored"
    if math.isnan(value):
        return "unscored"
    return "pass" if value > 0 else "fail"
