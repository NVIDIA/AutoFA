# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted taxonomy judge answers."""

from typing import Any


def _diagnosis_answer() -> dict[str, Any]:
    """A minimal valid answer to the structured diagnosis question the classifier always asks."""
    return {
        "type": "structured",
        "value": {
            "observed_system_failure": "scripted failure",
            "causal_chain": [],
            "critical_event": {
                "event_ref": None,
                "failure": "scripted",
                "why_earliest_unrecovered": "scripted",
            },
            "counterfactual_edit": {
                "target_event_ref": None,
                "minimal_edit": "scripted",
                "expected_effect": "scripted",
                "support": "unknown",
            },
            "evidence_support": "complete",
            "abstention_reason": None,
        },
    }
