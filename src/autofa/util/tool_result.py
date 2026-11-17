# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read native result semantics without depending on a command-line adapter."""

from __future__ import annotations

import json
from typing import Any


def result_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the ordinary MCP text envelope; retain unknown envelopes as-is."""
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                payload = json.loads(block["text"])
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                return payload
    return result


def result_failed(result: dict[str, Any]) -> bool:
    """Preserve native failures, including failures inside an MCP envelope."""
    if result.get("ok") is False or result.get("isError"):
        return True
    return any(result_payload({"content": [block]}).get("ok") is False
               for block in result.get("content", []))
