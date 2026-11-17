# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose generation APIs without changing the caller's completion contract."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from autofa.llm.capabilities import APIUnavailable


def chat_options(options: dict) -> dict:
    """Preserve configured Responses controls when a Chat route is necessary."""
    result = deepcopy(options)
    if "max_output_tokens" in result:
        if {"max_tokens", "max_completion_tokens"}.intersection(result):
            raise ValueError("Specify only one output token limit")
        result["max_completion_tokens"] = result.pop("max_output_tokens")
    if "reasoning" in result:
        reasoning = result.pop("reasoning")
        if not isinstance(reasoning, dict) or set(reasoning) - {"effort"}:
            raise ValueError("Chat fallback cannot preserve these native Responses reasoning "
                             "options; select a Responses-capable endpoint")
        if "effort" in reasoning:
            result["reasoning_effort"] = reasoning["effort"]
    if "text" in result:
        text = result.pop("text")
        if not isinstance(text, dict) or set(text) - {"format"}:
            raise ValueError("Chat fallback cannot preserve these native Responses text options")
        if format_ := text.get("format"):
            result["response_format"] = (
                {"type": "json_schema", "json_schema": {
                    key: value for key, value in format_.items() if key != "type"}}
                if format_.get("type") == "json_schema" else format_)
    return result


def routed_completion(client: Any, *, api: str, model: str, messages: list[dict],
                      on_compatibility: Callable[[str], None] | None = None,
                      usage_role: str = "detector",
                      **options: Any) -> Any:
    from autofa.llm.hub import chat_completions_safe
    from autofa.llm.responses import ResponsesUnsupportedRequest, create_response_completion

    if api not in {"auto", "decisions", "responses", "chat_completions"}:
        raise ValueError("Unknown LLM API policy")
    if api != "chat_completions":
        try:
            return create_response_completion(client, model=model, messages=messages,
                                              usage_role=usage_role, **options)
        except APIUnavailable:
            if api == "responses":
                raise
        except ResponsesUnsupportedRequest as exc:
            if api == "responses":
                raise
            if on_compatibility is not None:
                on_compatibility(str(exc))
    return chat_completions_safe(client, model=model, messages=messages,
                                 usage_role=usage_role, **chat_options(options))
