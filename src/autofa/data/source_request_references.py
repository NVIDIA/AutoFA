# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve content-addressed EventText requests without expanding the archive.

Request histories grow on every call. Keep their original references on events
and resolve one call at a time for the deduplicated model-call evidence view.
"""

from __future__ import annotations

import json
import re
from typing import Any

from autofa.data.source_errors import SourceDataError
from autofa.data.structured import decode_container

_ROLLOUT = ("run_id", "task_id", "repeat", "agent_ref")
_AMBIGUOUS = object()


def _identity(row: dict) -> tuple:
    return tuple(row.get(key) for key in _ROLLOUT)


def _decoded(value: Any, field: str) -> Any:
    if not isinstance(value, str):
        raise SourceDataError(f"Model request references missing {field} JSON")
    try:
        return json.loads(value)
    except (ValueError, RecursionError) as exc:
        raise SourceDataError(f"Model request references invalid {field} JSON") from exc


class RequestReferenceResolver:
    """Resolve messages/tools only within their original rollout and agent."""

    def __init__(self, prompt_messages: list[dict], tool_definitions: list[dict]):
        self.pools: dict[str, dict[tuple, Any]] = {}
        self.decoded: dict[tuple, Any] = {}
        self.patterns = {
            "request_messages": re.compile(r"[0-9a-fA-F]{16}"),
            "request_tools": re.compile(r"[0-9a-fA-F]{64}"),
        }
        for field, rows, reference, payload in (
            ("request_messages", prompt_messages, "msg_hash", "msg_json"),
            ("request_tools", tool_definitions, "sha256", "def_json"),
        ):
            pool: dict[tuple, Any] = {}
            for row in rows:
                if not isinstance(row.get(reference), str):
                    continue
                key = (*_identity(row), row[reference])
                value = row.get(payload)
                if key in pool and pool[key] != value:
                    # Equal JSON with different whitespace is not an ambiguity.
                    try:
                        same = json.dumps(_decoded(pool[key], field), sort_keys=True) == (
                            json.dumps(_decoded(value, field), sort_keys=True)
                        )
                    except SourceDataError:
                        same = False
                    if not same:
                        pool[key] = _AMBIGUOUS
                else:
                    pool[key] = value
            self.pools[field] = pool

    def _references(self, captured: dict, field: str, values: list):
        pool = self.pools[field]
        for index, value in enumerate(values):
            key = (*_identity(captured), value) if isinstance(value, str) else None
            if key is not None and (key in pool or self.patterns[field].fullmatch(value)):
                if key not in pool or pool[key] is None:
                    raise SourceDataError(f"Model request references missing {field}: {value}")
                if pool[key] is _AMBIGUOUS:
                    raise SourceDataError(f"Model request references ambiguous {field}: {value}")
                yield index, key

    def _value(self, field: str, key: tuple) -> Any:
        cache_key = (field, *key)
        if cache_key not in self.decoded:
            self.decoded[cache_key] = _decoded(self.pools[field][key], field)
        return self.decoded[cache_key]

    def resolve(self, captured: dict) -> dict:
        """Return one resolved request; preserve raw fields and never mutate input."""
        result = dict(captured)
        for field in self.pools:
            values = decode_container(captured.get(field))
            if not isinstance(values, list):
                continue
            references = list(self._references(captured, field, values))
            values = list(values)
            for index, key in references:
                values[index] = self._value(field, key)
            if references or isinstance(captured[field], str):
                result.setdefault(field + "_raw", captured[field])
            # The EventText schema establishes this structural boundary, including
            # empty lists and legacy arrays of literal strings. Generic message
            # parsing would otherwise treat their serialized JSON as literal text.
            result[field] = values
        return result


def attach_event_requests(
    requests: list[dict], events: dict[tuple, dict], keys: tuple[str, ...],
) -> None:
    """Attach exact source fields by full event identity, without expanding history."""
    seen: set[tuple] = set()
    for raw in requests:
        try:
            identity = tuple(raw[key] for key in keys)
        except KeyError as exc:
            raise SourceDataError("Model request lacks its event identity") from exc
        if identity not in events:
            raise SourceDataError("Orphan model request")
        if identity in seen:
            raise SourceDataError("Duplicate model request event identity")
        seen.add(identity)
        events[identity]["event_text"] = dict(raw)
