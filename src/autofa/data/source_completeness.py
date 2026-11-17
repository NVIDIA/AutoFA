# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distinguish provider generation termination from missing retained source bytes."""

from __future__ import annotations

from typing import Any

# Imports declare these for genuine content loss; a recognized provider generation
# cutoff is reported with GENERATION_CUTOFF_NOTICE instead.
EVENT_CONTENT_LIMITATION = (
    "Exported records retained, but the source reports truncated or encrypted "
    "event content; the missing content cannot be recovered from this export."
)
TRUNCATED_EVENT_LIMITATION = "source event marked truncated"
GENERATION_CUTOFF_NOTICE = "provider generation ended before normal completion"
_PROVIDER_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens", "incomplete"}


def generation_cutoff(record: Any) -> bool:
    """Recognize a provider finish state, never infer it from incomplete text.

    EvalRDL's Responses importer records response.status='incomplete' as
    length_truncated. That says generation ended, not that the export lost bytes.
    Part lengths, inventory completeness and the cause of termination still need
    their own verification. A bare truncation flag remains ambiguous.
    """
    return (isinstance(record, dict) and record.get("length_truncated") is True
            and isinstance(record.get("finish_reason"), str)
            and record.get("finish_reason") in _PROVIDER_FINISH_REASONS)


def event_content_unavailable(event: dict) -> bool:
    """Content-loss declarations, excluding a recognized provider finish state."""
    metadata = event.get("metadata", {})
    if not isinstance(metadata, dict):
        return True
    for record in (event, metadata):
        if (record.get("has_encrypted_reasoning")
                or any(record.get(key) for key in (
                    "truncated", "content_truncated", "missing", "incomplete",
                )) or record.get("complete") is False or record.get("prefix_complete") is False
                or (record.get("length_truncated") and not generation_cutoff(record))):
            return True
    return False
