# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, sanitized service diagnostics without request bodies or credentials."""

from __future__ import annotations

import re


def safe_error_detail(value: object, *, limit: int = 400) -> str:
    text = " ".join(str(value).split())
    text = re.sub(r"(?i)\b(?:(?:bearer|basic)\s+|sk-|nvapi-)[\w.\-+/=]+", "[redacted]", text)
    text = re.sub(
        r"(?i)(api[_-]?key|authorization|access[_-]?token|password|secret)"
        r"['\"]?\s*[:=]\s*['\"]?[^\s,'\"}]+", r"\1=[redacted]", text,
    )
    text = re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", text)
    text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", text)
    return text[:limit]


def model_error_diagnostic(exc: Exception) -> dict:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {}
    result: dict[str, str | int] = {"exception": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        result["status_code"] = status
    for key in ("code", "type", "param"):
        value = error.get(key) or getattr(exc, key, None)
        if isinstance(value, (str, int)):
            result[key] = safe_error_detail(value, limit=100)
    result["message"] = safe_error_detail(error.get("message") or str(exc)
                                         or type(exc).__name__)
    return result
