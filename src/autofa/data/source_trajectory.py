# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover conversation and execution metadata when the normalized API has no trace.

Raw export schema labels are not sufficient to identify the record structure.
Recovery requires a recognized trajectory and matching explicit task/repeat IDs.
External evaluation targets and transport credentials never enter the conversation.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import ijson

from autofa.data.execution_policy import is_evaluation_record, sanitize_metadata
from autofa.data.json_stream import parse_json
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.structured import ROLES, decode_container
from autofa.data.trajectory import parse_trajectory

_TRAJECTORY = object()
_OBSERVATION = object()
_STEP_FIELDS = dict.fromkeys((
    "step_id", "source", "role", "actor", "message", "content", "reasoning_content",
    "reasoning", "thought", "observation", "call_id", "tool_call_id", "timestamp",
    "model_name", "paired_event_idx", "paired_call_idx",
), True)
_STEP_FIELDS["tool_calls"] = (True,)
_STEP_FIELDS["observation"] = _OBSERVATION
_ENVELOPE_FIELDS = {
    "schema_version": True,
    "agent": True,
    "steps": (_STEP_FIELDS,),
}
_TRAJECTORY_ITEM_FIELDS = {**_ENVELOPE_FIELDS, **_STEP_FIELDS}
_RAW_FIELDS = {
    **_ENVELOPE_FIELDS,
    **dict.fromkeys(("problem_idx", "problem_id", "task_id", "repeat", "run_idx"), True),
    "trajectory": _TRAJECTORY,
}


def _project(
    events: Iterator[tuple[str, Any]], event: str, value: Any, fields: Any,
) -> Any:
    """Project every selected array item while skipping unrelated JSON subtrees."""
    from autofa.data.source_inputs import _project_value

    if fields is True or fields is None:
        return _project_value(events, event, value, fields)
    if fields is _OBSERVATION:
        observation = _project_value(events, event, value, True)
        decoded = decode_container(observation)
        if not isinstance(decoded, dict) or "results" not in decoded:
            return observation  # Arbitrary tool-visible output remains content.
        results = decode_container(decoded["results"])
        if not isinstance(results, list):
            raise DataApiError("Raw trajectory observation results must be an array")
        clean = []
        for result in results:
            result = decode_container(result)
            if not isinstance(result, dict) or "content" not in result:
                raise DataApiError("Raw trajectory observation result has no content")
            clean.append(sanitize_metadata(result))
        return sanitize_metadata({**decoded, "results": clean})
    if event == "null":
        return None
    if event == "string":
        decoded = decode_container(value)
        if not isinstance(decoded, (dict, list)):
            raise DataApiError("Raw trajectory structural field contains invalid JSON")
        decoded_events = parse_json(io.BytesIO(json.dumps(decoded).encode()))
        first_event, first_value = next(decoded_events)
        return _project(decoded_events, first_event, first_value, fields)
    if fields is _TRAJECTORY:
        fields = ((_TRAJECTORY_ITEM_FIELDS,) if event == "start_array"
                  else _ENVELOPE_FIELDS)
    if isinstance(fields, dict) and event == "start_map":
        result = {}
        while True:
            event, key = next(events)
            if event == "end_map":
                if fields is _RAW_FIELDS:
                    return result
                if "steps" not in result and is_evaluation_record(result):
                    raise DataApiError(
                        "Raw trajectory contains an unsupported conversation role/actor"
                    )
                return sanitize_metadata(result, rollout="steps" in result)
            # Recognized trajectory boundaries keep unknown execution extensions.
            # Unrelated raw-record fields are still streamed past without loading.
            child_fields = fields.get(key, None if fields is _RAW_FIELDS else True)
            event, value = next(events)
            child = _project(events, event, value, child_fields)
            if child_fields is not None:
                result[key] = child
    if isinstance(fields, tuple) and event == "start_array":
        result = []
        while True:
            event, value = next(events)
            if event == "end_array":
                return result
            result.append(_project(events, event, value, fields[0]))
    raise DataApiError("Raw trajectory structural field has an unsupported shape")


def _read_record(source: Path, offset: int) -> dict:
    from autofa.data.source_inputs import _JSON_CHUNK_BYTES, _JsonlRecordReader

    try:
        with source.open("rb") as handle:
            if offset < 0:
                raise DataApiError("Raw trajectory byte offset must be nonnegative")
            if offset:
                handle.seek(offset - 1)
                if handle.read(1) != b"\n":
                    raise DataApiError("Raw trajectory byte offset is not a JSONL record boundary")
            events = parse_json(_JsonlRecordReader(handle), buf_size=_JSON_CHUNK_BYTES)
            event, value = next(events)
            result = _project(events, event, value, _RAW_FIELDS)
            for _ in events:
                raise DataApiError("Raw trajectory record contains trailing JSON values")
            return result
    except (ijson.JSONError, UnicodeDecodeError, StopIteration) as exc:
        raise DataApiError("Raw trajectory offset did not identify a complete JSON record") from exc


def read_raw_trajectory(source: Path, offset: int, *, task: str, repeat: int) -> dict | None:
    """Return sanitized parsed steps, or None when no conversation is recorded.

    Record identities are authoritative. Missing or contradictory identities
    fail closed; selecting an offset alone cannot prove which rollout it belongs to.
    """
    raw = _read_record(source, offset)
    candidate = raw.get("trajectory", raw if "steps" in raw else None)
    if candidate is None or candidate == []:
        return None
    task_ids = [raw[k] for k in ("problem_idx", "problem_id", "task_id") if k in raw]
    repeats = [raw[k] for k in ("repeat", "run_idx") if k in raw]
    if (not task_ids or any(type(v) not in (str, int) or str(v) != task for v in task_ids)
            or not repeats or any(type(v) is not int or v != repeat for v in repeats)):
        raise DataApiError("Raw trajectory task/repeat identity does not match rollout")
    parsed = parse_trajectory(candidate)
    if not parsed.ok:
        # Empty envelopes are valid evidence of an absent trace. Recognizable
        # but malformed nonempty structures must not silently become emptiness.
        envelope = candidate
        if isinstance(envelope, list) and len(envelope) == 1:
            envelope = envelope[0]
        if isinstance(envelope, dict) and envelope.get("steps") == []:
            return None
        raise DataApiError(f"Raw trajectory could not be parsed completely: {parsed.warning}")
    for step in parsed._steps:
        if step.get("source") not in ROLES or is_evaluation_record(step):
            raise DataApiError("Raw trajectory contains an unsupported conversation role/actor")
    envelope = candidate[0] if isinstance(candidate, list) and len(candidate) == 1 else candidate
    agent = sanitize_metadata(envelope.get("agent", {})) if isinstance(envelope, dict) else {}
    metadata = sanitize_metadata({
        key: value for key, value in envelope.items() if key not in {"steps", "agent"}
    }, rollout=True) if isinstance(envelope, dict) else {}
    return {"steps": parsed._steps, "agent": agent or {}, "metadata": metadata,
            "limitations": parsed.limitations}


def recover_raw_trajectory(
    record: dict, *, api: DataApiClient, run_id: str, root: Path,
    source_files: dict[str, Path] | None = None,
    download_progress: Callable[[int, int], None] | None = None,
    activity: Callable[[str], None] | None = None,
) -> bool:
    """Fill an empty normalized trace from recognized raw conversation fields."""
    if record["trace"].get("steps"):
        return False
    from autofa.data.source_inputs import _source_record_path

    metadata = record["metadata"]
    reference = str(metadata.get("raw_record_ref") or "")
    name, _, offset = reference.rpartition(":")
    limits = record["trace"]["evidence_scope"]["limitations"]
    if not name.endswith(".jsonl") or not offset.isdigit():
        return False
    files = source_files if source_files is not None else {}
    source, local_offset = _source_record_path(
        api, run_id, name, int(offset), root, files,
        progress=download_progress, activity=activity,
    )
    try:
        recovered = read_raw_trajectory(
            source, local_offset, task=metadata["task_id"], repeat=metadata["repeat"],
        )
    except DataApiError as exc:
        raise DataApiError(
            f"{exc} (run={run_id}, task={metadata['task_id']}, "
            f"repeat={metadata['repeat']}, source={reference})"
        ) from exc
    if recovered is None:
        limits["raw source contains no recognized conversation trajectory"] = 1
        return False
    trace = record["trace"]
    trace["steps"] = recovered["steps"]
    execution = trace.setdefault("execution", {})
    execution["raw_trajectory_metadata"] = recovered["metadata"]
    execution["recovered_steps_source"] = "steps"
    execution.setdefault("availability", {})["recovered_steps"] = {
        "status": "partial",
        "reason": "Recovered raw trajectory steps and execution metadata; original API event "
                  "identities and exact model request envelopes are not inferred.",
    }
    if recovered["agent"]:
        trace["agent"] = {**trace.get("agent", {}), **{
            key: value for key, value in recovered["agent"].items() if value
        }}
    trace["evidence_scope"]["trajectory_source"] = "raw record trajectory"
    trace["evidence_scope"]["raw_trajectory_reference"] = reference
    limits.pop("source rollout contains no events", None)
    limits["raw trajectory recovered; exact model requests not recorded"] = 1
    for limitation in recovered["limitations"]:
        limits[limitation] = limits.get(limitation, 0) + 1
    record["raw_trajectory_recovered"] = True
    return True
