# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared provider usage primitives; missing measurements remain unknown."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def nonnegative_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) and value >= 0 else None
    return None


def token_count(value: Any) -> int | None:
    number = nonnegative_number(value)
    return int(number) if number is not None and number.is_integer() else None


def field(value: Any, name: str, default=None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def normalize_tokens(value: Any, *, anthropic: bool = False) -> dict[str, int]:
    """Canonical input includes cache; output already includes reasoning."""
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "outputTokens", "completion_tokens"),
        "cache_read_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens",
                                    "cachedInputTokens"),
        "cache_creation_input_tokens": ("cache_creation_input_tokens",
                                        "cacheCreationInputTokens", "cacheWriteInputTokens"),
        "reasoning_output_tokens": ("reasoning_output_tokens", "reasoningOutputTokens"),
        "total_tokens": ("total_tokens", "totalTokens"),
    }
    result = {}
    for target, names in aliases.items():
        for name in names:
            count = token_count(field(value, name))
            if count is not None:
                result[target] = count
                break
    for names, source, target in [
        (("input_tokens_details", "prompt_tokens_details"), "cached_tokens",
         "cache_read_input_tokens"),
        (("output_tokens_details", "completion_tokens_details"), "reasoning_tokens",
         "reasoning_output_tokens"),
    ]:
        for name in names:
            count = token_count(field(field(value, name), source))
            if count is not None:
                result[target] = count
                break
    if anthropic and "input_tokens" in result:
        result["input_tokens"] += (result.get("cache_read_input_tokens", 0)
                                   + result.get("cache_creation_input_tokens", 0))
    if "input_tokens" in result and "output_tokens" in result:
        result.setdefault("total_tokens", result["input_tokens"] + result["output_tokens"])
    return result


def usage_endpoint(value: Any) -> str | None:
    """Canonical endpoint identity, without credentials, query or trailing slash."""
    if value is None:
        return None
    try:
        parsed = urlsplit(str(value))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        if parsed.port is not None:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    except (TypeError, ValueError):
        return None
