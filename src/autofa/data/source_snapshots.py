# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read indexed request snapshots carried by normalized harness events."""

from __future__ import annotations

from typing import Any

from autofa.data.execution_policy import is_evaluation_record, sanitize_metadata
from autofa.data.source_errors import SourceDataError as DataApiError
from autofa.data.structured import decode_container, messages


def recover_request_snapshots(events: list[dict], *, lane: str = "") -> dict[str, Any]:
    """Recover indexed requests and their execution metadata without replaying messages."""
    pool, calls = [], []
    indices: dict[str, int] = {}
    event_indices = set()
    model_output_seen = False
    for event in events:
        if event.get("role") in {"assistant", "agent", "output"}:
            model_output_seen = True
        if (
            event.get("kind") != "other"
            or event.get("role") != "harness"
            or is_evaluation_record(event)
        ):
            continue
        for part in event.get("parts", []):
            value = decode_container(part.get("text"))
            if not isinstance(value, dict) or not {
                "message_ids",
                "messages_first_seen_here",
            }.issubset(value):
                continue
            if (
                len(event["parts"]) != event.get("n_parts")
                or (part.get("lane") or "") != lane
                or len(part["text"]) != part.get("chars")
                or event.get("length_truncated")
            ):
                raise DataApiError("Incomplete recorded request snapshot")
            references = decode_container(value["message_ids"])
            new = decode_container(value["messages_first_seen_here"])
            if (
                not isinstance(references, list)
                or any(not isinstance(i, str) for i in references)
                or not isinstance(new, dict)
            ):
                raise DataApiError("Invalid recorded request snapshot references")
            for identity, message in new.items():
                parsed, warnings = messages([message])
                if warnings or len(parsed) != 1:
                    raise DataApiError("Unsupported message in recorded request snapshot")
                if identity in indices:
                    if pool[indices[identity]] != parsed[0]:
                        raise DataApiError("Conflicting recorded request message identity")
                else:
                    indices[identity] = len(pool)
                    pool.append(parsed[0])
            if any(i not in indices for i in references):
                raise DataApiError("Recorded request snapshot references missing messages")
            calls.append(
                {
                    "index": len(calls),
                    "source_event_index": event["event_idx"],
                    "request_recorded": True,
                    "response_recorded": False,
                    "initial_request_recorded": not model_output_seen,
                    "request": [indices[i] for i in references],
                    "response": [],
                    "request_metadata": sanitize_metadata({
                        key: child for key, child in value.items()
                        if key not in {"message_ids", "messages_first_seen_here"}
                    }),
                    "metadata": sanitize_metadata({
                        key: child for key, child in event.items() if key != "parts"
                    }),
                }
            )
            event_indices.add(event["event_idx"])
    return {"messages": pool, "tools": [], "calls": calls, "event_indices": event_indices}
