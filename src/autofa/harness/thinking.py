# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate saved reasoning controls into Claude SDK options and audit metadata."""

from __future__ import annotations

from typing import Any

from autofa.defaults import default_value

THINKING_MODES = ("default", "adaptive", "enabled", "disabled")
EFFORT_LEVELS = ("default", "low", "medium", "high", "xhigh", "max")


def reasoning_options(
    *, thinking: str | None = None, effort: str | None = None,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    """None retains the legacy effort-implies-adaptive API; default delegates to Claude.

    Claude Code may normalize these requested controls for the selected model
    (for example, use adaptive thinking instead of a fixed token budget).
    """
    if thinking_budget_tokens is None:
        thinking_budget_tokens = default_value("thinking_budget_tokens")
    if thinking is not None and thinking not in THINKING_MODES:
        raise ValueError("thinking must be default, adaptive, enabled or disabled")
    if effort is not None and effort not in EFFORT_LEVELS:
        raise ValueError("effort must be default, low, medium, high, xhigh or max")
    if type(thinking_budget_tokens) is not int or thinking_budget_tokens < 1024:
        raise ValueError("thinking_budget_tokens must be an integer of at least 1024")
    explicit_effort = None if effort in (None, "default") else effort
    mode = thinking or ("adaptive" if explicit_effort else "default")
    options: dict[str, Any] = {}
    if mode == "enabled":
        options["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
    elif mode != "default":
        options["thinking"] = {"type": mode}
    if explicit_effort and mode in ("default", "adaptive"):
        options["effort"] = explicit_effort
    return options


def reasoning_metadata(**kwargs: Any) -> dict[str, Any]:
    """Record requested SDK settings, retaining legacy None for harness defaults."""
    options = reasoning_options(**kwargs)
    thinking = options.get("thinking", {})
    metadata = {"judge_effort": options.get("effort"), "judge_thinking": thinking.get("type")}
    if thinking.get("type") == "enabled":
        metadata["judge_thinking_budget_tokens"] = thinking["budget_tokens"]
    return metadata


def reasoning_arguments(
    *, thinking: str | None = None, effort: str | None = None,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    """Forward explicitly configured controls without changing legacy default calls."""
    if thinking_budget_tokens is None:
        thinking_budget_tokens = default_value("thinking_budget_tokens")
    arguments: dict[str, Any] = {}
    if effort is not None:
        arguments["effort"] = effort
    if thinking is not None:
        arguments["thinking"] = thinking
    if thinking == "enabled":
        arguments["thinking_budget_tokens"] = thinking_budget_tokens
    return arguments
