# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restricted Claude/Codex taxonomy classification sessions."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from autofa.config import autofa_home
from autofa.harness.settings import (
    HarnessNotFoundError,
    _authentication_status,
    behavior_judge_settings,
)
from autofa.harness.thinking import reasoning_options

from .classifier import (
    ABSTENTION_REASONS,
    NO_FAILURE_CODE,
    _validate_diagnosis_references,
)
from .judge_guidance import (
    AGENTIC_CLASSIFICATION_INSTRUCTIONS,
    AGENTIC_DIRECT_CLASSIFICATION_INSTRUCTIONS,
)
from .providers import ProviderError
from .schema import Taxonomy

AGENTIC_CLASSIFICATION_VERSION = "taxonomy-agentic-classification-v1"
AGENTIC_SESSION_TIMEOUT_SECONDS = 600
CODEX_TRANSPORT_ATTEMPTS = 3
TRACE_RESOURCE_NAMES = (
    "attempts",
    "relations",
    "provenance",
    "tool_definitions",
    "prompt_messages",
    "prompt_summaries",
)


def _default_checklist() -> dict[str, Any]:
    return {
        "task_context_checked": False,
        "outcome_checked": False,
        "recovery_checked": False,
        "counterevidence_checked": False,
        "candidate_leaves": [],
        "unresolved_questions": [],
    }


def _schema(
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass
class TaxonomyEvidenceReview:
    """In-memory, receipt-producing evidence capability for one rollout."""

    taxonomy: Taxonomy
    incident: dict[str, Any]
    first_pass: dict[str, Any] | None
    windows: list[dict[str, Any]]
    episodes: list[dict[str, Any]]
    artifact_source: dict[str, Any] | None = None
    fields: dict[str, str] = field(init=False)
    artifact_catalog: list[dict[str, Any]] = field(init=False)
    reads: list[dict[str, Any]] = field(default_factory=list)
    trace_reads: list[dict[str, Any]] = field(default_factory=list)
    resource_reads: list[dict[str, Any]] = field(default_factory=list)
    citations: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    artifact_pins: dict[int, dict[str, Any]] = field(default_factory=dict)
    artifact_accesses: list[dict[str, Any]] = field(default_factory=list)
    artifact_reads: list[dict[str, Any]] = field(default_factory=list)
    submission: dict[str, Any] | None = None
    checklist: dict[str, Any] = field(default_factory=_default_checklist)

    def __post_init__(self) -> None:
        self.artifact_catalog = []
        if self.artifact_source:
            from .assets import artifact_descriptors

            self.artifact_catalog = artifact_descriptors(self.artifact_source)
        self.fields = {
            "incident": _canonical(self.incident),
            **(
                {"first_pass": _canonical(self.first_pass)}
                if self.first_pass is not None
                else {}
            ),
            "episodes": _canonical(self.episodes),
            **{
                f"window:{index}": _canonical(window)
                for index, window in enumerate(self.windows)
            },
        }

    def reset_runtime_state(self) -> None:
        """Discard partial receipts before restarting a failed judge transport."""
        self.reads.clear()
        self.trace_reads.clear()
        self.resource_reads.clear()
        self.citations.clear()
        self.tool_calls.clear()
        self.artifact_pins.clear()
        self.artifact_accesses.clear()
        self.artifact_reads.clear()
        self.submission = None
        self.checklist = _default_checklist()

    def overview(self) -> dict[str, Any]:
        return self._record(
            "evidence_overview",
            {},
            {
                "fields": {
                    name: {
                        "characters": len(value),
                        "sha256": _sha256(value),
                    }
                    for name, value in self.fields.items()
                },
                "window_count": len(self.windows),
                "artifact_count": len(self.artifact_catalog),
                **(
                    {
                        "phase_one": {
                            "proposal": deepcopy(self.first_pass),
                            "episodes": deepcopy(self.episodes),
                        }
                    }
                    if self.first_pass is not None
                    else {"classification_mode": "direct"}
                ),
                "taxonomy": {
                    "id": self.taxonomy.taxonomy_id,
                    "version": self.taxonomy.version,
                    "sha256": self.taxonomy.content_sha256,
                    "leaf_count": len(tuple(self.taxonomy.leaves())),
                },
            },
        )

    def read(self, field: str, offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        value = self._field(field)
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 20_000:
            raise ValueError("limit must be an integer between 1 and 20000")
        end = min(len(value), offset + limit)
        self.reads.append({"field": field, "start": offset, "end": end})
        return self._record(
            "read_evidence",
            {"field": field, "offset": offset, "limit": limit},
            {
                "field": field,
                "offset": offset,
                "end": end,
                "next_offset": end if end < len(value) else None,
                "total_characters": len(value),
                "sha256": _sha256(value),
                "content": value[offset:end],
            },
        )

    def search(
        self,
        field: str,
        query: str,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        value = self._field(field)
        if not isinstance(query, str) or not query:
            raise ValueError("query must be nonempty text")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        matches = []
        cursor = offset
        while len(matches) < limit:
            index = value.find(query, cursor)
            if index < 0:
                break
            matches.append(
                {
                    "start": index,
                    "end": index + len(query),
                    "preview": value[max(0, index - 120) : index + len(query) + 120],
                }
            )
            cursor = index + max(1, len(query))
        next_offset = (
            cursor
            if len(matches) == limit and value.find(query, cursor) >= 0
            else None
        )
        return self._record(
            "search_evidence",
            {"field": field, "query": query, "offset": offset, "limit": limit},
            {"field": field, "matches": matches, "next_offset": next_offset},
        )

    def cite(self, field: str, quote: str) -> dict[str, Any]:
        value = self._field(field)
        if not isinstance(quote, str) or not quote:
            raise ValueError("quote must be nonempty text")
        indexes = []
        cursor = 0
        while (index := value.find(quote, cursor)) >= 0:
            indexes.append(index)
            cursor = index + max(1, len(quote))
        if len(indexes) != 1:
            raise ValueError(
                f"quote must identify exactly one occurrence in {field!r}; found {len(indexes)}"
            )
        start = indexes[0]
        if not any(
            receipt["field"] == field
            and receipt["start"] <= start
            and receipt["end"] >= start + len(quote)
            for receipt in self.reads
        ):
            raise ValueError("cite_evidence requires reading the quoted range first")
        citation_id = f"cite-{len(self.citations) + 1}"
        receipt = {
            "citation_id": citation_id,
            "field": field,
            "start": start,
            "end": start + len(quote),
            "quote": quote,
            "field_sha256": _sha256(value),
        }
        self.citations[citation_id] = receipt
        return self._record("cite_evidence", {"field": field, "quote": quote}, receipt)

    def _trace_source(self) -> dict[str, Any]:
        if self.artifact_source is None:
            raise ValueError("normalized trace source is unavailable")
        return self.artifact_source

    def trace_overview(self) -> dict[str, Any]:
        """Describe normalized EvalRDL resources without returning their full content."""
        from autofa import trace

        source = self._trace_source()
        resources = {
            "events": len(trace.events(source)),
            "steps": len(trace.steps(source)),
            "tool_calls": len(trace.tool_calls(source)),
            "attempts": len(trace.attempts(source)),
            "artifacts": len(trace.artifacts(source)),
            "tool_definitions": len(trace.tool_definitions(source)),
            "prompt_messages": len(trace.prompt_messages(source)),
            "prompt_summaries": len(trace.prompt_summaries(source)),
            "provenance": len(trace.provenance(source)),
            "relations": len(trace.relations(source)),
        }
        return self._record(
            "trace_overview",
            {},
            {
                "coverage": _portable(trace.coverage(source)),
                "resources": resources,
                "hint": (
                    "List event identities or auxiliary resources, inspect context, then "
                    "read exact fields before citing them."
                ),
            },
        )

    def list_trace_events(
        self,
        offset: int = 0,
        limit: int = 50,
        lane: str | None = None,
    ) -> dict[str, Any]:
        """Page native EvalRDL event identities and part metadata."""
        from autofa import trace

        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if lane is not None and not isinstance(lane, str):
            raise ValueError("lane must be text or null")
        events = trace.events(self._trace_source(), lane=lane)
        page = []
        for event in events[offset : offset + limit]:
            parts = event.get("parts") if isinstance(event.get("parts"), list) else []
            page.append(
                {
                    key: _portable(value)
                    for key, value in event.items()
                    if key != "parts"
                }
                | {
                    "parts": [
                        {
                            key: _portable(value)
                            for key, value in part.items()
                            if key not in {"text", "value", "content"}
                        }
                        | _content_summary(part)
                        for part in parts
                        if isinstance(part, dict)
                    ]
                }
            )
        end = offset + len(page)
        return self._record(
            "list_trace_events",
            {"offset": offset, "limit": limit, "lane": lane},
            {
                "offset": offset,
                "total": len(events),
                "next_offset": end if end < len(events) else None,
                "events": page,
            },
        )

    def _trace_resource_rows(self, resource: str) -> list[dict[str, Any]]:
        from autofa import trace

        readers = {
            "attempts": trace.attempts,
            "relations": trace.relations,
            "provenance": trace.provenance,
            "tool_definitions": trace.tool_definitions,
            "prompt_messages": trace.prompt_messages,
            "prompt_summaries": trace.prompt_summaries,
        }
        if resource not in readers:
            raise ValueError(
                "resource must be one of " + ", ".join(TRACE_RESOURCE_NAMES)
            )
        return [_portable(row) for row in readers[resource](self._trace_source())]

    def list_trace_resource(
        self,
        resource: str,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Page a closed set of standardized auxiliary EvalRDL resources."""
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        rows = self._trace_resource_rows(resource)
        page = [_resource_summary(row) for row in rows[offset : offset + limit]]
        end = offset + len(page)
        return self._record(
            "list_trace_resource",
            {"resource": resource, "offset": offset, "limit": limit},
            {
                "resource": resource,
                "offset": offset,
                "total": len(rows),
                "next_offset": end if end < len(rows) else None,
                "rows": page,
            },
        )

    def read_trace_resource_field(
        self,
        resource: str,
        row_index: int,
        field: str,
        offset: int = 0,
        limit: int = 12_000,
    ) -> dict[str, Any]:
        """Read one exact field from a standardized auxiliary resource row."""
        if type(row_index) is not int or row_index < 0:
            raise ValueError("row_index must be a non-negative integer")
        if not isinstance(field, str) or not field:
            raise ValueError("field must be nonempty text")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 20_000:
            raise ValueError("limit must be an integer between 1 and 20000")
        rows = self._trace_resource_rows(resource)
        if row_index >= len(rows):
            raise ValueError(f"row_index exceeds the {len(rows)} available rows")
        row = rows[row_index]
        if field not in row:
            raise ValueError(f"field {field!r} is absent from this {resource} row")
        value = row[field]
        text = value if isinstance(value, str) else _canonical(value)
        if offset > len(text):
            raise ValueError("offset exceeds resource field length")
        end = min(len(text), offset + limit)
        receipt = {
            "resource": resource,
            "row_index": row_index,
            "field": field,
            "start": offset,
            "end": end,
            "sha256": _sha256(text),
        }
        self.resource_reads.append(receipt)
        return self._record(
            "read_trace_resource_field",
            {
                "resource": resource,
                "row_index": row_index,
                "field": field,
                "offset": offset,
                "limit": limit,
            },
            {
                **receipt,
                "text": text[offset:end],
                "total_characters": len(text),
                "next_offset": end if end < len(text) else None,
            },
        )

    def cite_trace_resource(
        self,
        resource: str,
        row_index: int,
        field: str,
        quote: str,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Cite exact standardized resource text already read."""
        if not isinstance(quote, str) or not quote or len(quote) > 2_000:
            raise ValueError("quote must contain between 1 and 2000 characters")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        rows = self._trace_resource_rows(resource)
        if type(row_index) is not int or not 0 <= row_index < len(rows):
            raise ValueError("row_index is outside the resource")
        row = rows[row_index]
        if field not in row:
            raise ValueError(f"field {field!r} is absent from this {resource} row")
        value = row[field]
        text = value if isinstance(value, str) else _canonical(value)
        start = text.find(quote, offset)
        reads = [
            item
            for item in self.resource_reads
            if item["resource"] == resource
            and item["row_index"] == row_index
            and item["field"] == field
        ]
        if start < 0 or not _resource_ranges_cover(reads, start, start + len(quote)):
            raise ValueError("exact resource quote was not read")
        citation_id = f"cite-{len(self.citations) + 1}"
        citation = {
            "citation_id": citation_id,
            "kind": "trace_resource",
            "resource": resource,
            "row_index": row_index,
            "field": field,
            "start": start,
            "end": start + len(quote),
            "quote": quote,
            "sha256": _sha256(text),
        }
        self.citations[citation_id] = citation
        return self._record(
            "cite_trace_resource",
            {
                "resource": resource,
                "row_index": row_index,
                "field": field,
                "quote": quote,
                "offset": offset,
            },
            citation,
        )

    def _trace_event_part(
        self,
        lane: str,
        event_idx: int,
        part_idx: int,
        field: str,
    ) -> tuple[dict[str, Any], str]:
        from autofa import trace

        if not isinstance(lane, str):
            raise ValueError("lane must be text")
        if type(event_idx) is not int or event_idx < 0:
            raise ValueError("event_idx must be a non-negative integer")
        if type(part_idx) is not int or part_idx < 0:
            raise ValueError("part_idx must be a non-negative integer")
        events = [
            event
            for event in trace.events(self._trace_source(), lane=lane)
            if event.get("event_idx") == event_idx
        ]
        if len(events) != 1:
            raise ValueError("lane and event_idx must identify exactly one event")
        parts = events[0].get("parts")
        if not isinstance(parts, list):
            raise ValueError("event has no standardized parts")
        matches = [
            part
            for index, part in enumerate(parts)
            if isinstance(part, dict) and part.get("part_idx", index) == part_idx
        ]
        if len(matches) != 1 or field not in matches[0]:
            raise ValueError("part_idx and field must identify exactly one event part field")
        value = matches[0][field]
        return events[0], value if isinstance(value, str) else _canonical(value)

    def read_trace_event_part(
        self,
        lane: str,
        event_idx: int,
        part_idx: int,
        field: str = "text",
        offset: int = 0,
        limit: int = 12_000,
    ) -> dict[str, Any]:
        """Read exact part content from any standardized execution lane."""
        if not isinstance(field, str) or not field:
            raise ValueError("field must be nonempty text")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 20_000:
            raise ValueError("limit must be an integer between 1 and 20000")
        event, text = self._trace_event_part(lane, event_idx, part_idx, field)
        if offset > len(text):
            raise ValueError("offset exceeds event part field length")
        end = min(len(text), offset + limit)
        receipt = {
            "kind": "event_part",
            "lane": lane,
            "event_idx": event_idx,
            "part_idx": part_idx,
            "field": field,
            "start": offset,
            "end": end,
            "sha256": _sha256(text),
        }
        self.trace_reads.append(receipt)
        return self._record(
            "read_trace_event_part",
            {
                "lane": lane,
                "event_idx": event_idx,
                "part_idx": part_idx,
                "field": field,
                "offset": offset,
                "limit": limit,
            },
            {
                **receipt,
                "role": event.get("role"),
                "event_kind": event.get("kind"),
                "text": text[offset:end],
                "total_characters": len(text),
                "next_offset": end if end < len(text) else None,
            },
        )

    def cite_trace_event_part(
        self,
        lane: str,
        event_idx: int,
        part_idx: int,
        field: str,
        quote: str,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Cite exact event-part text already read."""
        if not isinstance(quote, str) or not quote or len(quote) > 2_000:
            raise ValueError("quote must contain between 1 and 2000 characters")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        _, text = self._trace_event_part(lane, event_idx, part_idx, field)
        start = text.find(quote, offset)
        reads = [
            item
            for item in self.trace_reads
            if item.get("kind") == "event_part"
            and item["lane"] == lane
            and item["event_idx"] == event_idx
            and item["part_idx"] == part_idx
            and item["field"] == field
        ]
        if start < 0 or not _resource_ranges_cover(reads, start, start + len(quote)):
            raise ValueError("exact event part quote was not read")
        citation_id = f"cite-{len(self.citations) + 1}"
        citation = {
            "citation_id": citation_id,
            "kind": "trace_event_part",
            "lane": lane,
            "event_idx": event_idx,
            "part_idx": part_idx,
            "field": field,
            "start": start,
            "end": start + len(quote),
            "quote": quote,
            "sha256": _sha256(text),
        }
        self.citations[citation_id] = citation
        return self._record(
            "cite_trace_event_part",
            {
                "lane": lane,
                "event_idx": event_idx,
                "part_idx": part_idx,
                "field": field,
                "quote": quote,
                "offset": offset,
            },
            citation,
        )

    def _trace_steps(self) -> list[dict[str, Any]]:
        from autofa import trace

        return [_portable(step) for step in trace.steps(self._trace_source())]

    @staticmethod
    def _step_field(step: dict[str, Any], field_name: str) -> str:
        if field_name in {"reasoning", "message", "observation"}:
            return str(step.get(field_name) or "")
        if field_name in {"tool_calls", "metadata", "parts"}:
            return _canonical(step.get(field_name) or ([] if field_name != "metadata" else {}))
        raise ValueError(
            "field must be reasoning, message, observation, tool_calls, metadata, or parts"
        )

    def read_trace_step(
        self,
        step_index: int,
        field: str,
        offset: int = 0,
        limit: int = 12_000,
    ) -> dict[str, Any]:
        """Read one exact normalized step field with pairing and source-event identity."""
        if type(step_index) is not int or step_index < 1:
            raise ValueError("step_index must be a positive integer")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 20_000:
            raise ValueError("limit must be an integer between 1 and 20000")
        steps = self._trace_steps()
        if step_index > len(steps):
            raise ValueError(f"step_index exceeds the {len(steps)} available steps")
        step = steps[step_index - 1]
        text = self._step_field(step, field)
        if offset > len(text):
            raise ValueError("offset exceeds trace step field length")
        end = min(len(text), offset + limit)
        receipt = {
            "kind": "text",
            "step_index": step_index,
            "field": field,
            "start": offset,
            "end": end,
            "sha256": _sha256(text),
            "source_event": step.get("source_event"),
        }
        self.trace_reads.append(receipt)
        return self._record(
            "read_trace_step",
            {
                "step_index": step_index,
                "field": field,
                "offset": offset,
                "limit": limit,
            },
            {
                **receipt,
                "source": step.get("source"),
                "paired_step_index": step.get("paired_step_index"),
                "result_step_indices": step.get("result_step_indices") or [],
                "text": text[offset:end],
                "total_characters": len(text),
                "next_offset": end if end < len(text) else None,
            },
        )

    def search_trace(
        self,
        query: str,
        field: str = "all",
        offset: int = 0,
        limit: int = 20,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """Search literal text across normalized ordered steps."""
        from autofa.data.trace_navigation import trace_events

        if not isinstance(query, str) or not 1 <= len(query) <= 2_000:
            raise ValueError("query must contain between 1 and 2000 characters")
        if field not in {"all", "reasoning", "message", "observation", "tool"}:
            raise ValueError("field must be all, reasoning, message, observation, or tool")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        pattern = re.compile(re.escape(query), 0 if case_sensitive else re.IGNORECASE)
        matches: list[dict[str, Any]] = []
        total = 0
        events = trace_events(self._trace_source())
        for event in events:
            for part in event["sections"]:
                section = part["key"].split(":")[0]
                if field != "all" and section != field:
                    continue
                for match in pattern.finditer(part["text"]):
                    if offset <= total < offset + limit:
                        start, end = match.span()
                        lo = max(0, start - 160)
                        hi = min(len(part["text"]), end + 160)
                        matches.append(
                            {
                                "step_index": event["index"],
                                "step_id": event.get("step_id"),
                                "source": event.get("source"),
                                "source_event": event.get("source_event"),
                                "section": part["key"],
                                "start": start,
                                "end": end,
                                "quote": match.group(),
                                "context": part["text"][lo:hi],
                                "context_start": lo,
                                "context_end": hi,
                                "total_characters": len(part["text"]),
                            }
                        )
                    total += 1
        return self._record(
            "search_trace",
            {
                "query": query,
                "field": field,
                "offset": offset,
                "limit": limit,
                "case_sensitive": case_sensitive,
            },
            {
                "matches": matches,
                "total_matches": total,
                "offset": offset,
                "next_offset": offset + len(matches)
                if offset + len(matches) < total
                else None,
                "steps_total": len(events),
            },
        )

    def trace_context(
        self,
        step_index: int,
        before: int = 2,
        after: int = 2,
        max_characters: int = 2_000,
    ) -> dict[str, Any]:
        """Read bounded ordered context around one normalized step."""
        from autofa.data.trace_navigation import trace_events

        events = trace_events(self._trace_source())
        if type(step_index) is not int or not 1 <= step_index <= len(events):
            raise ValueError("step_index is outside the normalized trace")
        if any(type(value) is not int for value in (before, after, max_characters)):
            raise ValueError("before, after, and max_characters must be integers")
        before = max(0, min(before, 10))
        after = max(0, min(after, 10))
        max_characters = max(100, min(max_characters, 5_000))
        start = max(0, step_index - 1 - before)
        end = min(len(events), step_index + after)
        page = []
        for event in events[start:end]:
            page.append(
                {
                    key: _portable(value)
                    for key, value in event.items()
                    if key not in {"sections", "tool_calls"}
                }
                | {
                    "tool_call_count": len(event.get("tool_calls") or []),
                    "sections": [
                        {
                            "key": part["key"],
                            "label": part["label"],
                            "text": part["text"][:max_characters],
                            "total_characters": len(part["text"]),
                            "truncated": len(part["text"]) > max_characters,
                        }
                        for part in event["sections"]
                    ],
                }
            )
        return self._record(
            "trace_context",
            {
                "step_index": step_index,
                "before": before,
                "after": after,
                "max_characters": max_characters,
            },
            {
                "steps_total": len(events),
                "steps_offset": start,
                "events": page,
            },
        )

    def cite_trace(
        self,
        step_index: int,
        field: str,
        quote: str,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Cite exact normalized trace text already read."""
        if not isinstance(quote, str) or not quote or len(quote) > 2_000:
            raise ValueError("quote must contain between 1 and 2000 characters")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        steps = self._trace_steps()
        if type(step_index) is not int or not 1 <= step_index <= len(steps):
            raise ValueError("step_index is outside the normalized trace")
        text = self._step_field(steps[step_index - 1], field)
        start = text.find(quote, offset)
        reads = [
            item
            for item in self.trace_reads
            if item.get("step_index") == step_index and item.get("field") == field
        ]
        if start < 0 or not _ranges_cover(reads, start, start + len(quote)):
            raise ValueError("exact trace quote was not read")
        source_event = steps[step_index - 1].get("source_event")
        citation_id = f"cite-{len(self.citations) + 1}"
        citation = {
            "citation_id": citation_id,
            "kind": "trace",
            "step_index": step_index,
            "field": field,
            "start": start,
            "end": start + len(quote),
            "quote": quote,
            "sha256": _sha256(text),
            "source_event": source_event,
        }
        self.citations[citation_id] = citation
        return self._record(
            "cite_trace",
            {
                "step_index": step_index,
                "field": field,
                "quote": quote,
                "offset": offset,
            },
            citation,
        )

    def list_artifacts(self) -> dict[str, Any]:
        """List trusted artifact descriptors without claiming to inspect their contents."""
        return self._record(
            "list_artifacts",
            {},
            {
                "artifacts": deepcopy(self.artifact_catalog),
                "pins": deepcopy(list(self.artifact_pins.values())),
                "note": "Unavailable or unread artifact content remains unknown.",
            },
        )

    def _artifact_allowed(self, artifact_idx: int) -> None:
        if type(artifact_idx) is not int or sum(
            item.get("artifact_idx") == artifact_idx for item in self.artifact_catalog
        ) != 1:
            raise ValueError("artifact_idx must identify exactly one retained artifact")
        if self.artifact_source is None:
            raise ValueError("artifact source identity is unavailable")

    @staticmethod
    def _artifact_fingerprint(evidence: dict[str, Any]) -> str:
        return _sha256(
            _canonical(
                {
                    key: value
                    for key, value in evidence.items()
                    if key not in {"path", "download_path", "data_url"}
                }
            )
        )

    def _track_artifact_access(
        self,
        artifact_idx: int,
        evidence: dict[str, Any],
        *,
        page: int | None = None,
    ) -> None:
        receipt = {
            "artifact_idx": artifact_idx,
            "page": page,
            "fingerprint": self._artifact_fingerprint(evidence),
        }
        previous = next(
            (
                item
                for item in self.artifact_accesses
                if item["artifact_idx"] == artifact_idx and item["page"] == page
            ),
            None,
        )
        if previous is not None and previous != receipt:
            raise ValueError("artifact evidence changed during this review")
        if previous is None:
            self.artifact_accesses.append(receipt)

    def _pin_artifact(
        self,
        artifact_idx: int,
        evidence: dict[str, Any],
        *,
        image: bool = False,
    ) -> None:
        if evidence.get("verified") is not True or not isinstance(
            evidence.get("source_identity"), dict
        ):
            raise ValueError("artifact evidence has no verified source identity")
        digest = (
            evidence.get("artifact_sha256", evidence.get("original_sha256"))
            if image
            else evidence.get("sha256")
        )
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("artifact evidence has no verified content hash")
        pin = {
            "artifact_idx": artifact_idx,
            "sha256": digest,
            "source_identity": evidence["source_identity"],
            "transform_version": evidence.get("transform_version"),
        }
        prior = self.artifact_pins.get(artifact_idx)
        if prior is not None and prior != pin:
            raise ValueError("artifact bytes or source identity changed during this review")
        self.artifact_pins.setdefault(artifact_idx, pin)

    def _inspect_artifact(self, artifact_idx: int) -> dict[str, Any]:
        from autofa import trace

        self._artifact_allowed(artifact_idx)
        assert self.artifact_source is not None
        result = trace.artifact(self.artifact_source, artifact_idx)
        self._track_artifact_access(artifact_idx, result)
        if result.get("status") == "available":
            self._pin_artifact(artifact_idx, result)
        return result

    def inspect_artifact(self, artifact_idx: int) -> dict[str, Any]:
        """Inspect artifact availability and text locators without reading their text."""
        result = self._inspect_artifact(artifact_idx)
        summary = {
            key: deepcopy(value)
            for key, value in result.items()
            if key not in {"segments", "path", "download_path", "data_url"}
        }
        summary["segments"] = [
            {
                key: deepcopy(value)
                for key, value in segment.items()
                if key != "text"
            }
            | {
                "total_characters": len(segment.get("text", "")),
                "text_sha256": _sha256(segment.get("text", "")),
            }
            for segment in result.get("segments") or []
            if isinstance(segment, dict)
        ]
        return self._record("inspect_artifact", {"artifact_idx": artifact_idx}, summary)

    def _artifact_segment(
        self,
        artifact_idx: int,
        locator: str,
    ) -> tuple[dict[str, Any], str]:
        result = self._inspect_artifact(artifact_idx)
        if result.get("verified") is not True or not result.get("sha256"):
            raise ValueError("verified artifact text is unavailable")
        matches = [
            segment
            for segment in result.get("segments") or []
            if isinstance(segment, dict) and segment.get("locator") == locator
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("text"), str):
            raise ValueError("artifact locator is unavailable or ambiguous")
        return result, matches[0]["text"]

    def read_artifact(
        self,
        artifact_idx: int,
        locator: str,
        offset: int = 0,
        limit: int = 12_000,
    ) -> dict[str, Any]:
        """Read a bounded exact artifact-text range with immutable receipts."""
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 20_000:
            raise ValueError("limit must be an integer between 1 and 20000")
        result, text = self._artifact_segment(artifact_idx, locator)
        if offset > len(text):
            raise ValueError("offset exceeds artifact segment length")
        end = min(len(text), offset + limit)
        receipt = {
            "kind": "text",
            "artifact_idx": artifact_idx,
            "locator": locator,
            "artifact_sha256": result["sha256"],
            "sha256": _sha256(text),
            "transform_version": result.get("transform_version"),
            "source_identity": result.get("source_identity"),
            "start": offset,
            "end": end,
        }
        self.artifact_reads.append(receipt)
        response = {
            **receipt,
            "text": text[offset:end],
            "total_characters": len(text),
            "next_offset": end if end < len(text) else None,
            "is_final": end == len(text),
            "limitations": result.get("limitations") or [],
            "status": result.get("status"),
            "complete": result.get("complete", False),
        }
        return self._record(
            "read_artifact",
            {
                "artifact_idx": artifact_idx,
                "locator": locator,
                "offset": offset,
                "limit": limit,
            },
            response,
        )

    def view_artifact(self, artifact_idx: int, page: int = 1) -> dict[str, Any]:
        """Render and return one verified artifact page to a vision-capable judge."""
        from autofa import trace

        self._artifact_allowed(artifact_idx)
        if type(page) is not int or page < 1:
            raise ValueError("page must be a positive integer")
        assert self.artifact_source is not None
        result = trace.artifact_image(self.artifact_source, artifact_idx, page=page)
        self._track_artifact_access(artifact_idx, result, page=page)
        if result.get("status") != "available":
            return self._record(
                "view_artifact",
                {"artifact_idx": artifact_idx, "page": page},
                result,
            )
        self._pin_artifact(artifact_idx, result, image=True)
        data_url = result.get("data_url")
        if not isinstance(data_url, str) or ";base64," not in data_url:
            raise ValueError("artifact renderer returned no image content")
        encoded = data_url.split(";base64,", 1)[1]
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("invalid rendered image encoding") from exc
        if hashlib.sha256(image_bytes).hexdigest() != result.get("sha256"):
            raise ValueError("rendered artifact image does not match its content hash")
        locator = result.get("locator")
        if not isinstance(locator, str) or not locator:
            raise ValueError("rendered artifact has no source locator")
        receipt = {
            "kind": "image",
            "artifact_idx": artifact_idx,
            "locator": locator,
            "page": page,
            "artifact_sha256": self.artifact_pins[artifact_idx]["sha256"],
            "sha256": result["sha256"],
            "transform_version": result.get("transform_version"),
            "source_identity": result.get("source_identity"),
        }
        self.artifact_reads.append(receipt)
        response = {
            key: value
            for key, value in result.items()
            if key not in {"path", "download_path"}
        }
        response["receipt"] = receipt
        return self._record(
            "view_artifact",
            {"artifact_idx": artifact_idx, "page": page},
            response,
        )

    def cite_artifact(
        self,
        artifact_idx: int,
        locator: str,
        quote: str = "",
        offset: int = 0,
    ) -> dict[str, Any]:
        """Cite exact artifact text already read or an image already viewed."""
        self._artifact_allowed(artifact_idx)
        if not isinstance(locator, str) or not locator or not isinstance(quote, str):
            raise ValueError("locator and quote must be strings")
        reads = [
            item
            for item in self.artifact_reads
            if item["artifact_idx"] == artifact_idx and item["locator"] == locator
        ]
        if quote:
            if len(quote) > 2_000 or type(offset) is not int or offset < 0:
                raise ValueError("invalid artifact quote or offset")
            _, text = self._artifact_segment(artifact_idx, locator)
            start = text.find(quote, offset)
            if start < 0 or not _ranges_cover(reads, start, start + len(quote)):
                raise ValueError("exact artifact quote was not read")
            citation = {
                **next(item for item in reads if item["kind"] == "text"),
                "start": start,
                "end": start + len(quote),
                "quote": quote,
            }
        else:
            images = [item for item in reads if item["kind"] == "image"]
            if not images:
                raise ValueError("artifact image was not viewed")
            citation = dict(images[-1])
        citation_id = f"cite-{len(self.citations) + 1}"
        citation["citation_id"] = citation_id
        self.citations[citation_id] = citation
        return self._record(
            "cite_artifact",
            {
                "artifact_idx": artifact_idx,
                "locator": locator,
                "quote": quote,
                "offset": offset,
            },
            citation,
        )

    def search_taxonomy(self, query: str = "", limit: int = 50) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        needle = query.strip().casefold()
        matches = []
        for node in self.taxonomy.leaves():
            text = " ".join(
                (
                    node.id,
                    node.code or "",
                    node.label,
                    node.description,
                    " ".join(node.include),
                    " ".join(node.exclude),
                )
            )
            if needle and needle not in text.casefold():
                continue
            matches.append(
                {
                    "id": node.id,
                    "code": node.code,
                    "label": node.label,
                    "fault_side": node.fault_side,
                    "criterion": node.criterion(),
                }
            )
        return self._record(
            "search_taxonomy",
            {"query": query, "limit": limit},
            {
                "matches": matches[:limit],
                "match_count": len(matches),
                "routing_choice": NO_FAILURE_CODE,
            },
        )

    def status(self) -> dict[str, Any]:
        return self._record(
            "review_status",
            {},
            {
                "read_ranges": list(self.reads),
                "trace_reads": deepcopy(self.trace_reads),
                "resource_reads": deepcopy(self.resource_reads),
                "artifact_pins": deepcopy(list(self.artifact_pins.values())),
                "artifact_reads": deepcopy(self.artifact_reads),
                "citations": list(self.citations.values()),
                "checklist": dict(self.checklist),
                "ready_to_submit": self._ready_to_submit(),
                "submitted": self.submission is not None,
            },
        )

    def update_status(
        self,
        task_context_checked: bool,
        outcome_checked: bool,
        recovery_checked: bool,
        counterevidence_checked: bool,
        candidate_leaves: list[str],
        unresolved_questions: list[str] | None = None,
    ) -> dict[str, Any]:
        leaves = {node.code or node.id for node in self.taxonomy.leaves()}
        valid = {*leaves, NO_FAILURE_CODE}
        if not isinstance(candidate_leaves, list) or not candidate_leaves:
            raise ValueError("candidate_leaves must be a nonempty list")
        unknown = set(candidate_leaves) - valid
        if unknown:
            raise ValueError(f"unknown candidate taxonomy codes: {sorted(unknown)}")
        questions = unresolved_questions or []
        if not isinstance(questions, list) or any(
            not isinstance(question, str) or not question.strip()
            for question in questions
        ):
            raise ValueError("unresolved_questions must contain nonempty text")
        self.checklist = {
            "task_context_checked": bool(task_context_checked),
            "outcome_checked": bool(outcome_checked),
            "recovery_checked": bool(recovery_checked),
            "counterevidence_checked": bool(counterevidence_checked),
            "candidate_leaves": list(dict.fromkeys(candidate_leaves)),
            "unresolved_questions": questions,
        }
        return self._record(
            "update_review_status",
            dict(self.checklist),
            {
                "checklist": dict(self.checklist),
                "ready_to_submit": self._ready_to_submit(),
            },
        )

    def submit(
        self,
        selected_code: str,
        rationale: str,
        diagnosis: dict[str, Any],
        citation_ids: list[str],
        alternatives: list[str] | None = None,
        abstention_reason: str | None = None,
        selected_episode_id: str | None = None,
    ) -> dict[str, Any]:
        if self.submission is not None:
            raise ValueError("classification was already submitted")
        leaves = {node.code or node.id for node in self.taxonomy.leaves()}
        if selected_code != NO_FAILURE_CODE and selected_code not in leaves:
            raise ValueError(f"unknown taxonomy leaf code: {selected_code!r}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be nonempty text")
        if not isinstance(diagnosis, dict):
            raise ValueError("diagnosis must be an object")
        _validate_diagnosis_shape(diagnosis)
        try:
            _validate_diagnosis_references(
                diagnosis,
                {
                    "incident": self.incident,
                    "windows": self.windows,
                    "verifier_artifacts": self.artifact_catalog,
                },
            )
        except ProviderError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(citation_ids, list) or any(
            citation_id not in self.citations for citation_id in citation_ids
        ):
            raise ValueError("citation_ids must reference citations issued by cite_evidence")
        if selected_code != NO_FAILURE_CODE and not citation_ids:
            raise ValueError("a selected failure requires at least one citation")
        if selected_code == NO_FAILURE_CODE:
            if abstention_reason not in ABSTENTION_REASONS:
                raise ValueError(
                    "no_supported_failure requires abstention_reason: "
                    + ", ".join(ABSTENTION_REASONS)
                )
            if not (
                citation_ids
                or self.reads
                or self.trace_reads
                or self.resource_reads
                or self.artifact_accesses
            ):
                raise ValueError(
                    "no_supported_failure requires inspected evidence or an unavailable "
                    "artifact receipt"
                )
        elif abstention_reason is not None:
            raise ValueError("abstention_reason is only valid for no_supported_failure")
        if not self._ready_to_submit():
            raise ValueError(
                "complete task context, outcome, recovery, and counterevidence checks "
                "before submission"
            )
        if selected_code not in self.checklist["candidate_leaves"]:
            raise ValueError("selected_code must be listed in candidate_leaves")
        episode_ids = {
            str(episode.get("episode_id"))
            for episode in self.episodes
            if episode.get("episode_id") is not None
        }
        if selected_episode_id is not None and selected_episode_id not in episode_ids:
            raise ValueError(f"unknown selected_episode_id: {selected_episode_id!r}")
        alternatives = list(dict.fromkeys(alternatives or []))
        unknown = set(alternatives) - leaves
        if unknown:
            raise ValueError(f"unknown alternative taxonomy codes: {sorted(unknown)}")
        self.submission = {
            "selected_code": selected_code,
            "alternatives": alternatives,
            "rationale": rationale.strip(),
            "diagnosis": diagnosis,
            "abstention_reason": abstention_reason,
            "selected_episode_id": selected_episode_id,
            "citations": [self.citations[citation_id] for citation_id in citation_ids],
        }
        return self._record(
            "submit_classification",
            {
                "selected_code": selected_code,
                "citation_ids": citation_ids,
                "selected_episode_id": selected_episode_id,
            },
            {"accepted": True, "selected_code": selected_code},
        )

    def specs(self) -> list[tuple[str, str, dict[str, Any], Callable[..., dict[str, Any]]]]:
        string = {"type": "string"}
        integer = {"type": "integer"}
        string_list = {"type": "array", "items": string}
        nullable_integer = {"type": ["integer", "null"]}
        nullable_string = {"type": ["string", "null"]}
        event_reference = {
            "anyOf": [
                integer,
                _schema({"lane": string, "event_idx": integer}, ["lane", "event_idx"]),
                {"type": "null"},
            ]
        }
        artifact_reference = _schema(
            {
                "artifact_idx": nullable_integer,
                "name": {"type": ["string", "null"]},
            },
            ["artifact_idx", "name"],
        )
        diagnosis_schema = _schema(
            {
                "observed_system_failure": string,
                "causal_chain": {
                    "type": "array",
                    "minItems": 1,
                    "items": _schema(
                        {
                            "source_type": {
                                "type": "string",
                                "enum": [
                                    "trace_event",
                                    "verifier_artifact",
                                    "rollout_artifact",
                                    "other",
                                ],
                            },
                            "event_ref": event_reference,
                            "artifact_ref": {
                                "anyOf": [artifact_reference, {"type": "null"}]
                            },
                            "claim": string,
                            "evidence": string,
                        },
                        [
                            "source_type",
                            "event_ref",
                            "artifact_ref",
                            "claim",
                            "evidence",
                        ],
                    ),
                },
                "critical_event": _schema(
                    {
                        "event_ref": event_reference,
                        "failure": string,
                        "why_earliest_unrecovered": string,
                    },
                    ["event_ref", "failure", "why_earliest_unrecovered"],
                ),
                "counterfactual_edit": _schema(
                    {
                        "target_event_ref": event_reference,
                        "minimal_edit": string,
                        "expected_effect": string,
                        "support": {
                            "type": "string",
                            "enum": ["supported", "weak", "unknown"],
                        },
                    },
                    [
                        "target_event_ref",
                        "minimal_edit",
                        "expected_effect",
                        "support",
                    ],
                ),
                "evidence_support": {
                    "type": "string",
                    "enum": ["complete", "partial", "insufficient"],
                },
            },
            [
                "observed_system_failure",
                "causal_chain",
                "critical_event",
                "counterfactual_edit",
                "evidence_support",
            ],
        )
        specs = [
            (
                "evidence_overview",
                "List available evidence fields, sizes, hashes, windows, and first pass.",
                _schema({}, []),
                self.overview,
            ),
            (
                "read_evidence",
                "Read one evidence field by exact character range; page with next_offset.",
                _schema(
                    {"field": string, "offset": integer, "limit": integer},
                    ["field"],
                ),
                self.read,
            ),
            (
                "search_evidence",
                "Locate literal text in one evidence field; search cannot prove absence.",
                _schema(
                    {
                        "field": string,
                        "query": string,
                        "offset": integer,
                        "limit": integer,
                    },
                    ["field", "query"],
                ),
                self.search,
            ),
            (
                "cite_evidence",
                "Issue a hash-bound citation for one exact quote already read.",
                _schema({"field": string, "quote": string}, ["field", "quote"]),
                self.cite,
            ),
            (
                "trace_overview",
                "Show normalized EvalRDL resource coverage and counts.",
                _schema({}, []),
                self.trace_overview,
            ),
            (
                "list_trace_events",
                "Page native EvalRDL event identities and bounded part metadata.",
                _schema(
                    {
                        "offset": integer,
                        "limit": integer,
                        "lane": nullable_string,
                    },
                    [],
                ),
                self.list_trace_events,
            ),
            (
                "read_trace_event_part",
                "Read exact content from a standardized event part in any execution lane.",
                _schema(
                    {
                        "lane": string,
                        "event_idx": integer,
                        "part_idx": integer,
                        "field": string,
                        "offset": integer,
                        "limit": integer,
                    },
                    ["lane", "event_idx", "part_idx"],
                ),
                self.read_trace_event_part,
            ),
            (
                "cite_trace_event_part",
                "Cite exact event-part text already read.",
                _schema(
                    {
                        "lane": string,
                        "event_idx": integer,
                        "part_idx": integer,
                        "field": string,
                        "quote": string,
                        "offset": integer,
                    },
                    ["lane", "event_idx", "part_idx", "field", "quote"],
                ),
                self.cite_trace_event_part,
            ),
            (
                "list_trace_resource",
                "Page standardized attempts, relations, provenance, tools, or prompts.",
                _schema(
                    {
                        "resource": {"type": "string", "enum": list(TRACE_RESOURCE_NAMES)},
                        "offset": integer,
                        "limit": integer,
                    },
                    ["resource"],
                ),
                self.list_trace_resource,
            ),
            (
                "read_trace_resource_field",
                "Read one exact field from a standardized auxiliary resource row.",
                _schema(
                    {
                        "resource": {"type": "string", "enum": list(TRACE_RESOURCE_NAMES)},
                        "row_index": integer,
                        "field": string,
                        "offset": integer,
                        "limit": integer,
                    },
                    ["resource", "row_index", "field"],
                ),
                self.read_trace_resource_field,
            ),
            (
                "cite_trace_resource",
                "Cite exact standardized resource text already read.",
                _schema(
                    {
                        "resource": {"type": "string", "enum": list(TRACE_RESOURCE_NAMES)},
                        "row_index": integer,
                        "field": string,
                        "quote": string,
                        "offset": integer,
                    },
                    ["resource", "row_index", "field", "quote"],
                ),
                self.cite_trace_resource,
            ),
            (
                "read_trace_step",
                "Read one exact normalized step field with tool-result relationships.",
                _schema(
                    {
                        "step_index": integer,
                        "field": string,
                        "offset": integer,
                        "limit": integer,
                    },
                    ["step_index", "field"],
                ),
                self.read_trace_step,
            ),
            (
                "search_trace",
                "Search literal text across normalized ordered trace sections.",
                _schema(
                    {
                        "query": string,
                        "field": string,
                        "offset": integer,
                        "limit": integer,
                        "case_sensitive": {"type": "boolean"},
                    },
                    ["query"],
                ),
                self.search_trace,
            ),
            (
                "trace_context",
                "Read bounded normalized context before and after one step.",
                _schema(
                    {
                        "step_index": integer,
                        "before": integer,
                        "after": integer,
                        "max_characters": integer,
                    },
                    ["step_index"],
                ),
                self.trace_context,
            ),
            (
                "cite_trace",
                "Cite exact normalized trace text already read with read_trace_step.",
                _schema(
                    {
                        "step_index": integer,
                        "field": string,
                        "quote": string,
                        "offset": integer,
                    },
                    ["step_index", "field", "quote"],
                ),
                self.cite_trace,
            ),
            (
                "list_artifacts",
                "List retained artifact descriptors and pinned hashes.",
                _schema({}, []),
                self.list_artifacts,
            ),
            (
                "inspect_artifact",
                "Inspect artifact availability and text locators without reading all text.",
                _schema({"artifact_idx": integer}, ["artifact_idx"]),
                self.inspect_artifact,
            ),
            (
                "read_artifact",
                "Read exact artifact text by locator with hash-bound paginated receipts.",
                _schema(
                    {
                        "artifact_idx": integer,
                        "locator": string,
                        "offset": integer,
                        "limit": integer,
                    },
                    ["artifact_idx", "locator"],
                ),
                self.read_artifact,
            ),
            (
                "view_artifact",
                "View one rendered artifact image or page with a verified hash receipt.",
                _schema({"artifact_idx": integer, "page": integer}, ["artifact_idx"]),
                self.view_artifact,
            ),
            (
                "cite_artifact",
                "Cite artifact text already read or an image already viewed.",
                _schema(
                    {
                        "artifact_idx": integer,
                        "locator": string,
                        "quote": string,
                        "offset": integer,
                    },
                    ["artifact_idx", "locator"],
                ),
                self.cite_artifact,
            ),
            (
                "search_taxonomy",
                "Search selectable taxonomy leaves and their exact criteria.",
                _schema({"query": string, "limit": integer}, []),
                self.search_taxonomy,
            ),
            (
                "update_review_status",
                "Record semantic task, outcome, recovery, counterevidence, and candidate checks.",
                _schema(
                    {
                        "task_context_checked": {"type": "boolean"},
                        "outcome_checked": {"type": "boolean"},
                        "recovery_checked": {"type": "boolean"},
                        "counterevidence_checked": {"type": "boolean"},
                        "candidate_leaves": string_list,
                        "unresolved_questions": string_list,
                    },
                    [
                        "task_context_checked",
                        "outcome_checked",
                        "recovery_checked",
                        "counterevidence_checked",
                        "candidate_leaves",
                    ],
                ),
                self.update_status,
            ),
            (
                "review_status",
                "Show read coverage, citations, and submission status.",
                _schema({}, []),
                self.status,
            ),
            (
                "submit_classification",
                "Submit exactly one validated primary taxonomy classification.",
                _schema(
                    {
                        "selected_code": string,
                        "alternatives": string_list,
                        "rationale": string,
                        "diagnosis": diagnosis_schema,
                        "citation_ids": string_list,
                        "abstention_reason": {"type": ["string", "null"]},
                        "selected_episode_id": {"type": ["string", "null"]},
                    },
                    [
                        "selected_code",
                        "rationale",
                        "diagnosis",
                        "citation_ids",
                    ],
                ),
                self.submit,
            ),
        ]
        return [
            (name, description, schema, self._tracked(name, callback))
            for name, description, schema, callback in specs
        ]

    def _tracked(
        self,
        name: str,
        callback: Callable[..., dict[str, Any]],
    ) -> Callable[..., dict[str, Any]]:
        def invoke(**arguments: Any) -> dict[str, Any]:
            try:
                return callback(**arguments)
            except Exception as exc:
                self.tool_calls.append(
                    {
                        "name": name,
                        "arguments_sha256": _sha256(_canonical(arguments)),
                        "is_error": True,
                        "error_type": type(exc).__name__,
                    }
                )
                raise

        return invoke

    def _field(self, name: str) -> str:
        try:
            return self.fields[name]
        except KeyError:
            raise ValueError(
                f"unknown evidence field {name!r}; available: {', '.join(self.fields)}"
            ) from None

    def _ready_to_submit(self) -> bool:
        return all(
            self.checklist[field]
            for field in (
                "task_context_checked",
                "outcome_checked",
                "recovery_checked",
                "counterevidence_checked",
            )
        ) and bool(self.checklist["candidate_leaves"])

    def _record(
        self,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        self.tool_calls.append(
            {
                "name": name,
                "arguments_sha256": _sha256(_canonical(arguments)),
                "result_sha256": _sha256(_canonical(result)),
                "is_error": False,
            }
        )
        return result


def _validate_diagnosis_shape(diagnosis: dict[str, Any]) -> None:
    if not isinstance(diagnosis.get("observed_system_failure"), str) or not diagnosis[
        "observed_system_failure"
    ].strip():
        raise ValueError("diagnosis.observed_system_failure must be nonempty text")
    chain = diagnosis.get("causal_chain")
    if not isinstance(chain, list) or not chain:
        raise ValueError("diagnosis.causal_chain must be a nonempty list")
    for index, item in enumerate(chain):
        if not isinstance(item, dict):
            raise ValueError(f"diagnosis.causal_chain[{index}] must be an object")
        if item.get("source_type") not in {
            "trace_event",
            "verifier_artifact",
            "rollout_artifact",
            "other",
        }:
            raise ValueError(
                f"diagnosis.causal_chain[{index}].source_type must identify its evidence source"
            )
        for field_name in ("claim", "evidence"):
            if not isinstance(item.get(field_name), str) or not item[field_name].strip():
                raise ValueError(
                    f"diagnosis.causal_chain[{index}].{field_name} must be nonempty text"
                )
        event_ref = item.get("event_ref")
        valid_event_ref = (
            type(event_ref) is int
            or (
                isinstance(event_ref, dict)
                and isinstance(event_ref.get("lane"), str)
                and type(event_ref.get("event_idx")) is int
            )
        )
        if item["source_type"] == "trace_event" and not valid_event_ref:
            raise ValueError(
                f"diagnosis.causal_chain[{index}].event_ref must identify a trace event"
            )
        if item["source_type"] in {"verifier_artifact", "rollout_artifact"} and not isinstance(
            item.get("artifact_ref"), dict
        ):
            raise ValueError(
                f"diagnosis.causal_chain[{index}].artifact_ref must identify an artifact"
            )
    critical = diagnosis.get("critical_event")
    if not isinstance(critical, dict):
        raise ValueError("diagnosis.critical_event must be an object")
    for field_name in ("failure", "why_earliest_unrecovered"):
        if not isinstance(critical.get(field_name), str) or not critical[field_name].strip():
            raise ValueError(f"diagnosis.critical_event.{field_name} must be nonempty text")
    counterfactual = diagnosis.get("counterfactual_edit")
    if not isinstance(counterfactual, dict):
        raise ValueError("diagnosis.counterfactual_edit must be an object")
    for field_name in ("minimal_edit", "expected_effect"):
        if (
            not isinstance(counterfactual.get(field_name), str)
            or not counterfactual[field_name].strip()
        ):
            raise ValueError(
                f"diagnosis.counterfactual_edit.{field_name} must be nonempty text"
            )
    if counterfactual.get("support") not in {"supported", "weak", "unknown"}:
        raise ValueError(
            "diagnosis.counterfactual_edit.support must be supported, weak, or unknown"
        )
    if diagnosis.get("evidence_support") not in {"complete", "partial", "insufficient"}:
        raise ValueError(
            "diagnosis.evidence_support must be complete, partial, or insufficient"
        )


@dataclass(frozen=True)
class AgenticClassificationProvider:
    harness: str
    model: str
    cli_path: str | None
    effort: str | None
    thinking: str | None
    thinking_budget_tokens: int | None
    max_turns: int = 32
    runner: Callable[..., dict[str, Any]] | None = field(default=None, repr=False)

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        max_turns: int,
        preflight: bool = True,
    ) -> AgenticClassificationProvider:
        settings = behavior_judge_settings(cfg)
        cli_path = settings["cli_path"]
        if preflight:
            cli_path = _preflight_harness(settings["harness"], cli_path)
        return cls(
            harness=settings["harness"],
            model=settings["model"],
            cli_path=cli_path,
            effort=settings["effort"],
            thinking=settings["thinking"],
            thinking_budget_tokens=settings["thinking_budget_tokens"],
            max_turns=max_turns,
        )

    def classify(
        self,
        *,
        taxonomy: Taxonomy,
        incident: dict[str, Any],
        windows: list[dict[str, Any]],
        artifact_source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Classify evidence directly without a prior proposal."""
        return self._run(
            taxonomy=taxonomy,
            incident=incident,
            first_pass=None,
            windows=windows,
            episodes=[],
            artifact_source=artifact_source,
        )

    def review(
        self,
        *,
        taxonomy: Taxonomy,
        incident: dict[str, Any],
        first_pass: dict[str, Any],
        windows: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        artifact_source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Review a stored proposal against the complete evidence."""
        return self._run(
            taxonomy=taxonomy,
            incident=incident,
            first_pass=first_pass,
            windows=windows,
            episodes=episodes,
            artifact_source=artifact_source,
        )

    def _run(
        self,
        *,
        taxonomy: Taxonomy,
        incident: dict[str, Any],
        first_pass: dict[str, Any] | None,
        windows: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        artifact_source: dict[str, Any] | None,
    ) -> dict[str, Any]:
        review = TaxonomyEvidenceReview(
            taxonomy=taxonomy,
            incident=incident,
            first_pass=first_pass,
            windows=windows,
            episodes=episodes,
            artifact_source=artifact_source,
        )
        mode = "review" if first_pass is not None else "classification"
        if first_pass is not None:
            prompt = (
                "Review the existing first-pass proposal and confirm it, revise it, or abstain. "
                "The first pass is not ground truth. Start with evidence_overview, which returns "
                "the complete phase-one proposal and episode ledger. Inspect its selected critical "
                "event and nearby action/result context first. Check its recovery and "
                "counterevidence anchors next. Expand to other trace regions only when those "
                "findings cannot confirm "
                "or contradict the proposal. The proposal and all evidence are available only "
                "through the registered tools."
            )
        else:
            prompt = (
                "Classify this rollout directly from the recorded evidence. There is no prior "
                "classification. Start with evidence_overview, then inspect "
                "the task, outcome, trace, recovery evidence, and counterevidence with the "
                "registered tools. Submit one evidence-backed taxonomy classification or abstain "
                "when the "
                "record cannot support a causal attribution."
            )
        runner = self.runner or _run_restricted_session
        session = runner(
            harness=self.harness,
            prompt=prompt,
            system_prompt=(
                AGENTIC_CLASSIFICATION_INSTRUCTIONS
                if first_pass is not None
                else AGENTIC_DIRECT_CLASSIFICATION_INSTRUCTIONS
            ),
            review=review,
            mode=mode,
            model=self.model,
            cli_path=self.cli_path,
            effort=self.effort,
            thinking=self.thinking,
            thinking_budget_tokens=self.thinking_budget_tokens,
            max_turns=self.max_turns,
        )
        if review.submission is None:
            raise RuntimeError(f"agentic taxonomy {mode} ended without a submission")
        return {
            **review.submission,
            "model": self.model,
            "artifact_catalog": review.artifact_catalog,
            "usage": session.get("usage") or {},
            "receipts": {
                "protocol": AGENTIC_CLASSIFICATION_VERSION,
                "mode": mode,
                "harness": self.harness,
                "model": self.model,
                "max_turns": self.max_turns if self.harness == "claude" else None,
                "configured_max_turns": self.max_turns,
                "tool_call_limit": self.max_turns if self.harness == "codex" else None,
                "session": session,
                "tool_calls": review.tool_calls,
                "citations": list(review.citations.values()),
                "trace_reads": review.trace_reads,
                "resource_reads": review.resource_reads,
                "artifact_pins": list(review.artifact_pins.values()),
                "artifact_accesses": review.artifact_accesses,
                "artifact_reads": review.artifact_reads,
            },
        }

    def configuration(self) -> dict[str, Any]:
        """Behavior-affecting settings without credentials or local paths."""
        return {
            "protocol": AGENTIC_CLASSIFICATION_VERSION,
            "harness": self.harness,
            "model": self.model,
            "effort": self.effort,
            "thinking": self.thinking,
            "thinking_budget_tokens": self.thinking_budget_tokens,
            "configured_max_turns": self.max_turns,
            "tool_call_limit": self.max_turns if self.harness == "codex" else None,
        }

    def configuration_sha256(self) -> str:
        return _sha256(json.dumps(self.configuration(), sort_keys=True, separators=(",", ":")))


def _preflight_harness(harness: str, cli_path: str | None) -> str | None:
    if harness == "claude":
        from autofa.harness.claude_cli import ensure_claude_auth

        resolved = ensure_claude_auth(cli_path=cli_path)
        if resolved and (problem := _authentication_status(resolved, "claude")):
            raise HarnessNotFoundError(f"Claude Code is unavailable: {problem}")
        return resolved
    if harness == "codex":
        from autofa.harness.model_catalog import _find_codex_cli

        resolved = _find_codex_cli(cli_path)
        if problem := _authentication_status(resolved, "codex"):
            raise HarnessNotFoundError(f"Codex is unavailable: {problem}")
        return resolved
    raise HarnessNotFoundError(f"unsupported Judge endpoint harness: {harness!r}")


def _run_restricted_session(**kwargs: Any) -> dict[str, Any]:
    from autofa.util.cancellation import cancellable

    return asyncio.run(cancellable(_run_restricted_session_async(**kwargs)))


async def _run_restricted_session_async(
    *,
    harness: str,
    prompt: str,
    system_prompt: str,
    review: TaxonomyEvidenceReview,
    model: str,
    cli_path: str | None,
    effort: str | None,
    thinking: str | None,
    thinking_budget_tokens: int | None,
    max_turns: int,
    mode: str = "review",
) -> dict[str, Any]:
    directory = autofa_home() / "cache" / "taxonomy-agentic"
    directory.mkdir(parents=True, exist_ok=True)
    if harness == "codex":
        from autofa.harness.codex_app_server import CodexAppServerError

        remaining_turns = max_turns
        cumulative_usage: Counter[str] = Counter()
        for attempt in range(CODEX_TRANSPORT_ATTEMPTS):
            try:
                session = await _run_codex(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    review=review,
                    model=model,
                    cli_path=cli_path,
                    effort=effort,
                    max_turns=remaining_turns,
                    directory=directory,
                    mode=mode,
                )
                session["transport_attempts"] = attempt + 1
                session["transport_restarts"] = attempt
                cumulative_usage.update(session.get("usage") or {})
                session["usage"] = dict(cumulative_usage)
                return session
            except CodexAppServerError as exc:
                cumulative_usage.update(exc.usage)
                remaining_turns -= len(review.tool_calls)
                if not exc.retryable or attempt + 1 == CODEX_TRANSPORT_ATTEMPTS:
                    raise
                if remaining_turns <= 0:
                    raise RuntimeError(
                        "Codex transport failed after exhausting the taxonomy tool budget"
                    ) from exc
                review.reset_runtime_state()
                await asyncio.sleep(_codex_transport_retry_delay(review, attempt))
        raise AssertionError("Codex transport retry loop exhausted unexpectedly")
    return await _run_claude(
        prompt=prompt,
        system_prompt=system_prompt,
        review=review,
        model=model,
        cli_path=cli_path,
        effort=effort,
        thinking=thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        max_turns=max_turns,
        directory=directory,
        mode=mode,
    )


def _codex_transport_retry_delay(
    review: TaxonomyEvidenceReview,
    attempt: int,
) -> float:
    """Spread worker restarts so a startup burst does not fail in lockstep."""
    identity = _canonical(
        {
            "id": review.incident.get("id"),
            "source": review.incident.get("source"),
        }
    )
    jitter = int(hashlib.sha256(identity.encode()).hexdigest()[:4], 16) / 65535
    return min(4.0, 0.5 * (2**attempt) + jitter)


async def _run_codex(
    *,
    prompt: str,
    system_prompt: str,
    review: TaxonomyEvidenceReview,
    model: str,
    cli_path: str | None,
    effort: str | None,
    max_turns: int,
    directory: Path,
    mode: str = "review",
) -> dict[str, Any]:
    from autofa.harness.codex_app_server import run_codex_turn
    from autofa.metrics.claude_judge import _evidence_tool_result
    from autofa.metrics.codex_judge import _codex_tool_result

    specs = review.specs()
    callbacks = {name: fn for name, _description, _schema_value, fn in specs}
    tool_lock = asyncio.Lock()
    tools = [
        {"name": name, "description": description, "inputSchema": schema_value}
        for name, description, schema_value, _fn in specs
    ]

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in callbacks:
            return {
                "success": False,
                "contentItems": [
                    {"type": "inputText", "text": "Only taxonomy evidence tools are available."}
                ],
            }
        async with tool_lock:
            result = await asyncio.to_thread(
                _evidence_tool_result,
                name,
                callbacks[name],
                arguments,
            )
        return _codex_tool_result(result)

    result = await run_codex_turn(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        cwd=str(directory),
        cli_path=cli_path,
        effort=None if effort in (None, "default") else effort,
        tools=tools,
        call_tool=call_tool,
        restricted=True,
        timeout_s=AGENTIC_SESSION_TIMEOUT_SECONDS,
        max_tool_calls=max_turns,
    )
    if (result.status != "completed" or result.error) and review.submission is None:
        if getattr(result, "retryable", False):
            from autofa.harness.codex_app_server import CodexAppServerError

            error = CodexAppServerError(
                {"code": getattr(result, "error_code", None) or "transportClosed"}
            )
            error.usage = dict(result.usage)
            raise error
        raise RuntimeError(
            f"incomplete agentic taxonomy {mode}: {result.error or result.status}"
        )
    return {
        "session_id": result.thread_id,
        "turn_id": result.turn_id,
        "status": result.status,
        "usage": result.usage,
    }


def _claude_result_incomplete(
    *, subtype: str, is_error: bool, submitted: bool
) -> bool:
    """A valid taxonomy submission outranks Claude's generic error flag."""
    return not submitted and (subtype != "success" or is_error)


async def _run_claude(
    *,
    prompt: str,
    system_prompt: str,
    review: TaxonomyEvidenceReview,
    model: str,
    cli_path: str | None,
    effort: str | None,
    thinking: str | None,
    thinking_budget_tokens: int | None,
    max_turns: int,
    directory: Path,
    mode: str = "review",
) -> dict[str, Any]:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ResultMessage,
        SystemMessage,
        create_sdk_mcp_server,
        query,
        tool,
    )

    from autofa.harness.claude_transport import CLAUDE_MAX_BUFFER_SIZE
    from autofa.metrics.claude_judge import _evidence_tool_result

    tool_lock = asyncio.Lock()

    def expose(
        name: str,
        description: str,
        schema_value: dict[str, Any],
        fn: Callable[..., dict[str, Any]],
    ) -> Any:
        @tool(name, description, schema_value)
        async def call(arguments: dict[str, Any]) -> dict[str, Any]:
            async with tool_lock:
                return await asyncio.to_thread(
                    _evidence_tool_result,
                    name,
                    fn,
                    arguments,
                )

        return call

    specs = review.specs()
    server = create_sdk_mcp_server(
        name="taxonomy_evidence",
        version="1.0.0",
        tools=[expose(*spec) for spec in specs],
    )
    tool_names = [f"mcp__taxonomy_evidence__{spec[0]}" for spec in specs]
    options = ClaudeAgentOptions(
        model=model,
        **reasoning_options(
            thinking=thinking,
            effort=effort,
            thinking_budget_tokens=thinking_budget_tokens,
        ),
        cwd=str(directory),
        system_prompt=system_prompt,
        tools=[],
        allowed_tools=tool_names,
        mcp_servers={"taxonomy_evidence": server},
        strict_mcp_config=True,
        permission_mode="bypassPermissions",
        setting_sources=[],
        max_turns=max_turns,
        max_buffer_size=CLAUDE_MAX_BUFFER_SIZE,
        cli_path=cli_path,
    )

    async def inputs() -> Any:
        yield {"type": "user", "message": {"role": "user", "content": prompt}}

    session: dict[str, Any] = {}
    failure: str | None = None
    try:
        async with asyncio.timeout(AGENTIC_SESSION_TIMEOUT_SECONDS):
            async for message in query(prompt=inputs(), options=options):
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    session["session_id"] = message.data.get("session_id")
                elif isinstance(message, ResultMessage):
                    if _claude_result_incomplete(
                        subtype=message.subtype,
                        is_error=message.is_error,
                        submitted=review.submission is not None,
                    ):
                        from autofa.harness.claude import _format_result_error

                        failure = (
                            f"incomplete agentic taxonomy {mode}: "
                            f"{message.subtype}/{message.stop_reason}; "
                            f"turns={message.num_turns}; {_format_result_error(message)}"
                        )
                        continue
                    session.update(
                        {
                            "session_id": message.session_id,
                            "status": message.subtype,
                            "stop_reason": message.stop_reason,
                            "num_turns": message.num_turns,
                            "usage": message.usage,
                            "cost_usd": message.total_cost_usd,
                            "sdk_is_error": message.is_error,
                            "sdk_errors": list(message.errors or ()),
                            "terminal_reason": message.terminal_reason,
                        }
                    )
    except TimeoutError as exc:
        if review.submission is None:
            raise RuntimeError(
                f"incomplete agentic taxonomy {mode}: "
                f"session timed out after {AGENTIC_SESSION_TIMEOUT_SECONDS} seconds"
            ) from exc
        session.setdefault("status", "timeout")
    if failure is not None:
        raise RuntimeError(failure)
    if not session:
        raise RuntimeError(f"agentic taxonomy {mode} returned no session result")
    return session


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _portable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.loads(_canonical(value))


def _content_summary(part: dict[str, Any]) -> dict[str, Any]:
    for key in ("text", "value", "content"):
        value = part.get(key)
        if value is None:
            continue
        text = value if isinstance(value, str) else _canonical(value)
        return {
            "content_field": key,
            "characters": len(text),
            "sha256": _sha256(text),
            "preview": text[:500],
            "truncated": len(text) > 500,
        }
    return {}


def _resource_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in row.items():
        rendered = value if isinstance(value, str) else _canonical(value)
        if len(rendered) > 500 or key.endswith("_json"):
            summary[key] = {
                "characters": len(rendered),
                "sha256": _sha256(rendered),
                "preview": rendered[:500],
                "truncated": len(rendered) > 500,
            }
        else:
            summary[key] = _portable(value)
    return summary


def _resource_ranges_cover(
    reads: list[dict[str, Any]], start: int, end: int
) -> bool:
    current = start
    for item in sorted(reads, key=lambda item: item["start"]):
        if item["start"] <= current:
            current = max(current, item["end"])
    return current >= end


def _ranges_cover(reads: list[dict[str, Any]], start: int, end: int) -> bool:
    current = start
    for item in sorted(
        (item for item in reads if item["kind"] == "text"),
        key=lambda item: item["start"],
    ):
        if item["start"] <= current:
            current = max(current, item["end"])
    return current >= end


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
