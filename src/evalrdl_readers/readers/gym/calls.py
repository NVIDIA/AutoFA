# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym's two accounts of a rollout record's model calls: `ng_trajectory.model_calls`, which
keeps each request whole, and `ng_model_call_capture.calls`, the call as captured, stating its
dialect and endpoint. Both are keyed by `model_call_id`."""

from __future__ import annotations

from typing import Any

from ..common import dig
from ..prompts import dialect_of, request_items


def trajectory_calls(rec: dict) -> list:
    """`ng_trajectory.model_calls` as written, entries of any type, so a position stays the
    producer's; [] where it is not a list."""
    calls = dig(rec, "ng_trajectory", "model_calls")
    return calls if isinstance(calls, list) else []


def captured_calls(rec: dict) -> list[dict]:
    """The calls of `ng_model_call_capture.calls` that carry an id."""
    return [
        c
        for c in dig(rec, "ng_model_call_capture", "calls") or ()
        if isinstance(c, dict) and c.get("model_call_id")
    ]


def endpoint(call: dict) -> str:
    """The model endpoint a captured call went to; "" where it names none."""
    return str(dig(call, "model_ref", "name") or "")


def stated_dialects(rec: dict) -> dict[str, Any]:
    """Each captured call's stated dialect, by `model_call_id`."""
    return {c["model_call_id"]: c.get("dialect") for c in captured_calls(rec)}


def _stated(call: dict, stated: dict[str, Any]) -> Any:
    """The dialect the capture states for a trajectory call, else its own response's word."""
    cid = call.get("model_call_id")
    return (stated.get(cid) if cid is not None else None) or dig(
        call, "response_metadata", "dialect"
    )


def call_dialect(call: dict, stated: dict[str, Any]) -> str | None:
    """A trajectory call's request dialect; None for one no reader knows."""
    request = call.get("request")
    return dialect_of(_stated(call, stated), request) if isinstance(request, dict) else None


def call_items(call: Any, stated: dict[str, Any]) -> list:
    """A trajectory call's request message items, in its dialect; [] where it has none or its
    dialect is unknown. A request with no captured call to land on is read all the same."""
    if not isinstance(call, dict):
        return []
    return request_items(call.get("request"), _stated(call, stated))
