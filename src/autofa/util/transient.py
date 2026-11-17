# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize transport failures without retrying authentication or invalid input."""

from __future__ import annotations

import re


def transient_message(message: str) -> bool:
    text = message.lower()
    if any(marker in text for marker in (
        "usage policy", "authentication", "unauthorized", "forbidden", "permission denied",
        "insufficient_quota", "credit balance", "usage limit", "quota exceeded",
        "context length", "prompt is too long",
    )):
        return False
    return bool(re.search(
        r"\b(?:http|api error:?|error code:?)\s*(?:408|429|500|502|503|504|529)\b", text,
    )) or any(marker in text for marker in (
        "connection closed mid-response", "connection reset", "connection refused",
        "connection aborted", "connection timed out", "connection error", "read timed out",
        "read operation timed out", "request timed out", "stream disconnected",
        "overloaded_error", "rate_limit_error", "network is unreachable", "network is down",
        "getaddrinfo", "fetch failed", "temporary failure in name resolution",
        "broken pipe", "server disconnected", "network connection lost",
    ))


def is_transient_error(exc: Exception) -> bool:
    return (
        isinstance(exc, (ConnectionError, TimeoutError))
        or transient_message(str(exc))
        or (isinstance(exc, OSError)
            and exc.errno in {8, 50, 51, 54, 60, 61, 65, 101, 104, 110, 111, 113})
    )
