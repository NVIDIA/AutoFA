# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare saved detector content independently of transport and display order."""

from __future__ import annotations

from typing import Any

from autofa.core.library_transport import unpack_bucket_examples

from .schemas import validate_library_payload


def _comparable_library(payload: Any) -> dict | None:
    if not isinstance(payload, dict):
        return None
    version = payload.get("version", 1)
    if type(version) is not int or version not in {1, 2}:
        return None
    try:
        # Packing changes the supported envelope version, not detector content.
        comparable = {**unpack_bucket_examples(payload), "version": 2}
        functions = comparable.get("functions")
        if not isinstance(functions, list) or any(not isinstance(fn, dict) for fn in functions):
            return None
        validate_library_payload(comparable)
        comparable["functions"] = sorted(comparable["functions"], key=lambda fn: fn["name"])
        return comparable
    except (KeyError, TypeError, ValueError):
        return None


def same_library_content(left: Any, right: Any) -> bool:
    """Ignore only packing, supported versions and function order.

    Scores, source, dependencies, evidence and unknown saved fields remain exact.
    Neither input is modified, and malformed envelopes never establish identity.
    """
    comparable = _comparable_library(left)
    return comparable is not None and comparable == _comparable_library(right)
