# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend identity and reasoning settings shared by all judging stages."""

from __future__ import annotations

from typing import Any

from autofa.defaults import default_value
from autofa.harness.thinking import reasoning_arguments, reasoning_metadata


def backend_name(harness: str) -> str:
    if harness == "claude":
        return "claude_agent_sdk"
    if harness == "codex":
        return "codex_app_server"
    raise ValueError(f"Unsupported judge harness: {harness}")


def backend_arguments(harness: str) -> dict[str, str]:
    """Keep existing Claude call signatures compatible with integrations."""
    backend_name(harness)
    return {"harness": harness} if harness != "claude" else {}


def judge_reasoning_metadata(
    *, harness: str = "claude", thinking: str | None = None, effort: str | None = None,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    if thinking_budget_tokens is None:
        thinking_budget_tokens = default_value("behavior_judge_thinking_budget_tokens")
    backend_name(harness)
    if harness == "claude":
        settings = reasoning_metadata(
            thinking=thinking, effort=effort, thinking_budget_tokens=thinking_budget_tokens,
        )
    else:
        if effort is not None and (not isinstance(effort, str) or not effort.strip()):
            raise ValueError("Codex judge effort must be a nonempty string")
        settings = {"judge_effort": None if effort in (None, "default") else effort,
                    "judge_thinking": None}
    return {"judge_harness": harness, **settings}


def judge_reasoning_arguments(
    *, harness: str = "claude", thinking: str | None = None, effort: str | None = None,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    if thinking_budget_tokens is None:
        thinking_budget_tokens = default_value("behavior_judge_thinking_budget_tokens")
    if harness == "codex":
        metadata = judge_reasoning_metadata(harness=harness, effort=effort)
        return {"effort": metadata["judge_effort"]} if effort is not None else {}
    backend_name(harness)
    return reasoning_arguments(thinking=thinking, effort=effort,
                               thinking_budget_tokens=thinking_budget_tokens)
