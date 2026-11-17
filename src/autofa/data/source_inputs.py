# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover model requests and trajectories omitted by the normalized event stream."""

from __future__ import annotations

import errno
import hashlib
import http.client
import io
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO

import ijson

from autofa.data.execution_policy import is_evaluation_record
from autofa.data.json_stream import parse_json
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.structured import decode_container, input_view
from autofa.session.managed_launch import preparation_event

_JSON_CHUNK_BYTES = 64 * 1024
_REQUEST_FIELDS = {
    "messages": True, "input": True, "tools": True, "instructions": True, "system": True,
}
# Dicts select object fields; a one-element list selects only the first array
# entry; True retains a whole value. Grader fields and later requests are skipped.
# The first call's model reference identifies a grader's request.
_INPUT_FIELDS = {
    "instance_config": {"problem_info": {
        "instance_id": True, "task_name": True, "problem_statement": True,
    }},
    "_ng_task_index": True,
    "_ng_rollout_index": True,
    "ng_trajectory": {"model_calls": [{
        "request": _REQUEST_FIELDS, "model_ref": True, "response_metadata": True,
    }]},
    "responses_create_params": _REQUEST_FIELDS,
}


def _download_diagnostic(exc: BaseException) -> dict[str, Any]:
    """Keep actionable transport fields while discarding signed URLs and messages."""
    cause = exc
    seen = set()
    while (isinstance(cause, DataApiError) and isinstance(cause.__cause__, BaseException)
           and id(cause) not in seen):
        seen.add(id(cause))
        cause = cause.__cause__
    status = (getattr(exc, "code", getattr(exc, "status", None))
              or getattr(cause, "code", getattr(cause, "status", None)))
    reason = getattr(cause, "reason", None)
    if isinstance(reason, BaseException):
        cause = reason
    return {"error_type": type(cause).__name__,
            "errno": getattr(cause, "errno", None), "http_status": status,
            "retryable": (status in {408, 429, 500, 502, 503, 504} if status is not None
                          else not isinstance(cause, ssl.SSLCertVerificationError) and (
                              isinstance(cause, (TimeoutError, ConnectionError,
                                                 http.client.IncompleteRead,
                                                 http.client.RemoteDisconnected))
                              or getattr(cause, "errno", None) in {
                                  errno.ECONNRESET, errno.ECONNABORTED, errno.ETIMEDOUT,
                                  errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPIPE,
                              }))}


def _source_file(
    api: DataApiClient, run_id: str, name: str, root: Path, *,
    progress: Callable[[int, int], None] | None = None,
    activity: Callable[[str], None] | None = None,
) -> Path:
    """Fetch once as bytes: byte-offset JSONL reads must not corrupt UTF-8 boundaries."""
    endpoint = f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/raw"
    preparation_event("checking_raw_source")
    if activity:
        activity("Checking raw log size…")
    try:
        stat = api._get_json(endpoint + "/slice", {"path": name, "offset": 0, "length": 0})
    except Exception as exc:
        preparation_event("source_preparation_failed", diagnostic=_download_diagnostic(exc))
        raise
    size = stat.get("size") if isinstance(stat, dict) else None
    if type(size) is not int or size <= 0:
        raise DataApiError("Raw source file has an unknown or invalid size")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / (hashlib.sha256(name.encode()).hexdigest() + ".jsonl")
    checksum = dest.with_suffix(".sha256")
    if dest.is_file() and dest.stat().st_size == size and checksum.is_file():
        if activity:
            activity("Verifying cached raw log…")
        with dest.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() == checksum.read_text():
                preparation_event("source_cache_verified")
                if activity:
                    activity("Preparing rollout traces…")
                return dest
    temp = dest.with_suffix(".partial")
    last_diagnostic: dict[str, Any] = {}

    def download() -> str:
        nonlocal last_diagnostic
        # EvalRDL can stream raw archives without supporting byte ranges. A new
        # request must start over, including its checksum and progress counter.
        digest = hashlib.sha256()
        read = 0
        if progress:
            progress(0, size)
        preparation_event("downloading_raw_source")
        try:
            # Reopen the API endpoint on every attempt to obtain a fresh URL if
            # redirected. Signed attachment URLs must never appear in errors.
            with urllib.request.urlopen(
                api._url(endpoint + "/file", {"path": name}), timeout=api.timeout,
            ) as response, temp.open("wb") as handle:
                while chunk := response.read(1024**2):
                    read += len(chunk)
                    if read > size:
                        raise DataApiError("Raw source download exceeded its declared size")
                    digest.update(chunk)
                    handle.write(chunk)
                    if progress:
                        progress(read, size)
            if read != size:
                raise http.client.IncompleteRead(b"", size - read)
            return digest.hexdigest()
        except BaseException as exc:
            last_diagnostic = _download_diagnostic(exc)
            preparation_event("raw_source_attempt_failed", diagnostic=last_diagnostic)
            temp.unlink(missing_ok=True)
            raise

    def retrying(attempt: int, delay: float) -> None:
        preparation_event("retrying_raw_source", diagnostic={
            **last_diagnostic, "attempt": attempt, "max_attempts": api.max_attempts,
        })
        if activity:
            activity(
                f"Raw log download interrupted; retrying from start in {delay:g}s "
                f"(attempt {attempt}/{api.max_attempts})…"
            )

    try:
        checksum_value = api._request_with_retries(
            download, label="downloading raw source file", on_retry=retrying,
            redact_errors=True,
        )
        temp.replace(dest)
        checksum.write_text(checksum_value, encoding="ascii")
        preparation_event("preparing_rollout_traces")
        if activity:
            activity("Preparing rollout traces…")
    except (DataApiError, OSError, urllib.error.URLError) as exc:
        diagnostic = (last_diagnostic if isinstance(exc, DataApiError) and last_diagnostic
                      else _download_diagnostic(exc))
        preparation_event("source_preparation_failed", diagnostic=diagnostic)
        if isinstance(exc, DataApiError):
            exc.diagnostic = diagnostic
            raise
        detail = ", ".join(f"{key}={value}" for key, value in diagnostic.items()
                           if value is not None)
        failure = DataApiError(f"Raw source download failed ({detail}); retry preparation")
        failure.diagnostic = diagnostic
        raise failure from None
    finally:
        temp.unlink(missing_ok=True)
    return dest


def _source_record(
    api: DataApiClient, run_id: str, name: str, offset: int, root: Path, *,
    progress: Callable[[int, int], None] | None = None,
    activity: Callable[[str], None] | None = None,
) -> Path:
    """Fetch and cache only the JSONL record beginning at ``offset``."""
    preparation_event("checking_raw_source")
    try:
        if offset < 0:
            raise DataApiError("Raw source byte offset must be nonnegative")
        reference = f"{name}:{offset}"
        root.mkdir(parents=True, exist_ok=True)
        dest = root / (hashlib.sha256(reference.encode()).hexdigest() + ".jsonl")
        checksum = dest.with_suffix(".sha256")
        if dest.is_file() and checksum.is_file():
            with dest.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() == checksum.read_text():
                    preparation_event("source_cache_verified")
                    return dest

        temp = dest.with_suffix(".partial")
        digest = hashlib.sha256()
        written = 0
        endpoint = f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/raw/slice"
        cursor = offset

        def retrying(attempt: int, delay: float) -> None:
            preparation_event("retrying_raw_source", diagnostic={
                "attempt": attempt, "max_attempts": api.max_attempts, "retryable": True,
            })

        if activity:
            activity("Fetching selected raw record…")
        preparation_event("downloading_raw_source")
        try:
            with temp.open("wb") as handle:
                while True:
                    data = api._get_json(
                        endpoint, {"path": name, "offset": cursor, "length": _JSON_CHUNK_BYTES},
                        on_retry=retrying,
                    )
                    text = data.get("text") if isinstance(data, dict) else None
                    consumed = data.get("length") if isinstance(data, dict) else None
                    if not isinstance(text, str) or type(consumed) is not int or consumed < 0:
                        raise DataApiError("Raw source slice has invalid text or byte length")
                    raw = text.encode("utf-8")
                    if consumed != len(raw):
                        raise DataApiError("Raw source slice byte length does not match its text")
                    line, newline, _ = raw.partition(b"\n")
                    selected = line + newline
                    handle.write(selected)
                    digest.update(selected)
                    written += len(selected)
                    size = data.get("size")
                    if progress and type(size) is int and size > offset:
                        progress(written, max(written, size - offset))
                    if newline or data.get("eof"):
                        break
                    if not consumed:
                        raise DataApiError("Raw source slice ended before the selected record")
                    cursor += consumed
                    if type(size) is int and cursor >= size:
                        raise DataApiError("Raw source record exceeds its declared file size")
            if progress:
                progress(written, written)
            checksum_value = digest.hexdigest()
            temp.replace(dest)
            checksum.write_text(checksum_value, encoding="ascii")
            preparation_event("preparing_rollout_traces")
            if activity:
                activity("Preparing rollout traces…")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return dest
    except BaseException as exc:
        diagnostic = _download_diagnostic(exc)
        preparation_event("source_preparation_failed", diagnostic=diagnostic)
        if isinstance(exc, DataApiError):
            exc.diagnostic = diagnostic
        raise


def cache_raw_source_file(
    record: dict, *, api: DataApiClient, run_id: str, root: Path,
    source_files: dict[str, Path],
    download_progress: Callable[[int, int], None] | None = None,
    activity: Callable[[str], None] | None = None,
) -> None:
    """Download one complete raw log when the prepared dataset contains every rollout."""
    metadata = record.get("metadata") or {}
    if metadata.get("record_schema") != "responses-api" and (record.get("trace") or {}).get(
        "steps"
    ):
        return
    reference = str(metadata.get("raw_record_ref") or "")
    name, _, offset = reference.rpartition(":")
    if name in source_files or not name.endswith(".jsonl") or not offset.isdigit():
        return
    source_files[name] = _source_file(
        api, run_id, name, root, progress=download_progress, activity=activity,
    )


def _source_record_path(
    api: DataApiClient, run_id: str, name: str, offset: int, root: Path,
    files: dict[str, Path],
    progress: Callable[[int, int], None] | None = None,
    activity: Callable[[str], None] | None = None,
) -> tuple[Path, int]:
    """Read from a cached raw log, or fetch and cache the referenced record."""
    reference = f"{name}:{offset}"
    if reference in files:
        return files[reference], 0
    if name in files:
        return files[name], offset
    files[reference] = _source_record(
        api, run_id, name, offset, root, progress=progress, activity=activity,
    )
    return files[reference], 0


class _JsonlRecordReader:
    """Expose one JSONL record as bounded reads, stopping at its first newline."""

    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.finished = False

    def read(self, size: int) -> bytes:
        if size < 0:
            raise ValueError("Raw JSON records require bounded reads")
        if self.finished or size == 0:
            return b""
        chunk = self.handle.readline(size)
        self.finished = not chunk or chunk.endswith(b"\n")
        return chunk


def _project_value(
    events: Iterator[tuple[str, Any]], event: str, value: Any, fields: Any,
) -> Any:
    """Build only selected subtrees while consuming/validating every JSON token."""
    if fields is None:
        # Skip irrelevant subtrees without constructing nested lists/dicts.
        depth = int(event in {"start_map", "start_array"})
        while depth:
            event, _ = next(events)
            depth += int(event in {"start_map", "start_array"})
            depth -= int(event in {"end_map", "end_array"})
        return None
    if event == "null":
        return None
    if event == "string" and isinstance(fields, (dict, list)):
        decoded = decode_container(value)
        if not isinstance(decoded, (dict, list)):
            raise DataApiError("Raw source structural field contains invalid JSON")
        decoded_events = parse_json(io.BytesIO(json.dumps(decoded).encode()))
        first_event, first_value = next(decoded_events)
        return _project_value(decoded_events, first_event, first_value, fields)
    if isinstance(fields, dict) and event != "start_map":
        raise DataApiError("Raw source input field must be an object")
    if isinstance(fields, list) and event != "start_array":
        raise DataApiError("Raw source model_calls field must be an array")
    if event == "start_map":
        result = {}
        while True:
            event, key = next(events)
            if event == "end_map":
                return result
            child_fields = True if fields is True else fields.get(key)
            event, value = next(events)
            child = _project_value(events, event, value, child_fields)
            if child_fields is not None:
                result[key] = child
    if event == "start_array":
        result = []
        index = 0
        while True:
            event, value = next(events)
            if event == "end_array":
                return result
            child_fields = True if fields is True else (fields[0] if index == 0 else None)
            child = _project_value(events, event, value, child_fields)
            if child_fields is not None:
                result.append(child)
            index += 1
    # ijson preserves arbitrary-size integers; convert decimal JSON numbers to
    # the floats json.loads used to produce (tool schemas must stay serializable).
    return float(value) if isinstance(value, Decimal) else value


def _read_input_record(source: Path, offset: int) -> dict[str, Any]:
    """Read input/identity fields without materializing the complete rollout."""
    try:
        with source.open("rb") as handle:
            if offset:
                handle.seek(offset - 1)
                if handle.read(1) != b"\n":
                    raise DataApiError("Raw source byte offset is not at a JSONL record boundary")
            events = parse_json(_JsonlRecordReader(handle), buf_size=_JSON_CHUNK_BYTES)
            event, value = next(events)
            if event != "start_map":
                raise DataApiError("Raw source record is not an object")
            result = _project_value(events, event, value, _INPUT_FIELDS)
            # Exhaust the parser: malformed tails/trailing data must fail even
            # when all requested fields appeared near the beginning of the line.
            for _ in events:
                raise DataApiError("Raw source record contains trailing JSON values")
            return result
    except (ijson.JSONError, UnicodeDecodeError, StopIteration) as exc:
        raise DataApiError(
            "Raw source byte offset did not identify a complete JSON record"
        ) from exc


def initial_request(record: dict, *, task: str, repeat: int) -> dict[str, Any]:
    """Select only model-visible input, never problem solutions or grader fields."""
    config = record.get("instance_config") or {}
    problem = config.get("problem_info") or {}
    identity = problem.get("instance_id")
    # Terminal Bench raw records namespace instance_id; the API uses task_name.
    # Accept only this known alias, with no contradictory task_name when present.
    if identity == f"terminal_bench::{task}" and problem.get("task_name", task) == task:
        identity = task
    # NeMo Gym exports can identify tasks by their collection index while
    # problem_info retains the benchmark's instance ID. Verify the explicit
    # index, never infer it from row order or merely accept any numeric task ID.
    task_index = record.get("_ng_task_index")
    if type(task_index) is int and task_index >= 0 and str(task_index) == task:
        identity = task
    elif identity is None and "_ng_task_index" in record:
        raise DataApiError("Raw source task identity does not match rollout")
    if identity is not None and identity != task:
        raise DataApiError(
            f"Raw source task identity does not match rollout: expected {task!r}, "
            f"got {identity!r} (task_name={problem.get('task_name')!r})"
        )
    rollout_index = record.get("_ng_rollout_index")
    if "_ng_rollout_index" in record and (
        type(rollout_index) is not int or rollout_index < 0 or rollout_index != repeat
    ):
        raise DataApiError("Raw source repeat identity does not match rollout")
    trajectory = decode_container(record.get("ng_trajectory")) or {}
    calls = decode_container(trajectory.get("model_calls")) or []
    if calls and isinstance(calls[0], dict):
        calls = [{**calls[0], "request": decode_container(calls[0].get("request"))}, *calls[1:]]
    if (not calls or not isinstance(calls[0], dict)
            or not isinstance(calls[0].get("request"), dict)
            or is_evaluation_record(calls[0])):
        # No first agent model request was captured (a grader's call is never
        # the agent's input). This request was sent to the agent. The agent may
        # add instructions or transform the conversation before its first model
        # call, so retain it with distinct provenance rather than claiming an
        # exact model capture.
        agent_request = decode_container(record.get("responses_create_params"))
        if isinstance(agent_request, dict):
            view = input_view(
                agent_request, tools=agent_request.get("tools"), label="Recorded agent request",
            )
            if view["messages"] or view["tools"]:
                return {
                    "prompt": view["text"], "messages": view["messages"], "tools": view["tools"],
                    "origin": "recorded_agent_request",
                    "limitations": view["limitations"] + [
                        "exact initial model request not recorded; recorded agent request retained",
                    ],
                }
        statement = problem.get("problem_statement")
        return {"prompt": "", "tools": [],
                "task_statement": (
                    statement if identity == task and isinstance(statement, str) else ""
                ),
                "limitations": ["initial model request not recorded"]}
    request = calls[0]["request"]
    view = input_view(request, tools=request.get("tools"))
    limitations = [s.replace("non-text content retained but not inspected",
                            "non-text initial request content not inspected")
                   for s in view["limitations"]]
    return {"prompt": view["text"], "messages": view["messages"],
            "tools": view["tools"], "limitations": limitations}


def _reconcile_request_capture_evidence(trace: dict, recovered: dict) -> None:
    """Clear only superseded capture gaps, retaining true partial-recovery gaps.

    Event requests and raw model calls are different views of the same validated
    rollout. Do not infer complete recovery from one recovered response, or clear
    limitations from the execution events themselves.
    """
    scope = trace["evidence_scope"]
    previous = trace.get("model_call_evidence", {}).get("calls", [])
    calls = recovered["calls"]
    covers_captures = bool(calls) and len(calls) >= len(previous)
    if not covers_captures:
        return
    capture_lanes = {call.get("source_lane") for call in previous}
    raw_lanes = {
        call["metadata"][key] for call in calls for key in ("lane", "source_lane")
        if isinstance(call.get("metadata", {}).get(key), str)
    }
    # Raw captures need not record an event/lane mapping. A total call count
    # cannot certify that every lane's missing evidence was recovered.
    if len(capture_lanes) > 1 or (raw_lanes and not raw_lanes.issubset(capture_lanes)):
        return
    limits = scope["limitations"]
    capture_limits = scope.get("request_capture_limitations", {})
    requests_recorded = all(call.get("request_recorded") for call in calls)
    tools_recorded = all(type(call.get("tools")) is int for call in calls)
    resolved = {}
    for reason, count in list(capture_limits.items()):
        if not requests_recorded:
            continue
        if reason == "unsupported tool definitions retained in raw source":
            if not tools_recorded:
                continue
        elif reason == "event model request messages not recorded":
            # A request envelope can exist while its messages/input are absent,
            # null or empty. That does not resolve a prior missing-message gap.
            if not all(call.get("request") for call in calls):
                continue
        else:
            continue
        remaining = limits.get(reason, 0) - count
        if remaining > 0:
            limits[reason] = remaining
        else:
            limits.pop(reason, None)
        capture_limits.pop(reason)
        resolved[reason] = count
    if resolved:
        scope["superseded_request_capture_limitations"] = resolved
    if not capture_limits:
        scope.pop("request_capture_limitations", None)
    if all(call.get("response_recorded") and call.get("response") for call in calls):
        notice = "provider response envelopes not recorded in event request captures"
        scope.get("notices", {}).pop(notice, None)


def recover_initial_inputs(
    records: list[dict], *, api: DataApiClient, run_id: str, root: Path,
    source_files: dict[str, Path] | None = None,
    download_progress: Callable[[int, int], None] | None = None,
    activity: Callable[[str], None] | None = None,
) -> None:
    """Attach input, model-call evidence and missing trajectories to source records."""
    from autofa.data.source_model_calls import (
        EXCLUDED_CALL,
        drop_grader_calls,
        recover_model_calls,
    )

    files: dict[str, Path] = source_files if source_files is not None else {}
    for record in records:
        metadata = record["metadata"]
        if metadata.get("record_schema") != "responses-api":
            continue
        reference = str(metadata.get("raw_record_ref") or "")
        name, _, offset = reference.rpartition(":")
        limits = record["trace"]["evidence_scope"]["limitations"]
        if not name.endswith(".jsonl") or not offset.isdigit():
            limits["initial model request source reference unsupported"] = 1
            continue
        source, local_offset = _source_record_path(
            api, run_id, name, int(offset), root, files,
            progress=download_progress, activity=activity,
        )
        try:
            raw = _read_input_record(source, local_offset)
            recovered = initial_request(raw, task=metadata["task_id"], repeat=metadata["repeat"])
            model_calls = recover_model_calls(source, local_offset)
        except DataApiError as exc:
            raise DataApiError(
                f"{exc} (run={run_id}, task={metadata['task_id']}, "
                f"repeat={metadata['repeat']}, source={reference})"
            ) from exc
        if excluded_calls := model_calls.pop("excluded_call_ids"):
            # The event export repeats these calls under the same ids (already counted).
            drop_grader_calls(record["trace"], excluded_calls)
        agent_input = recovered.get("origin") == "recorded_agent_request"
        record["initial_prompt"] = "" if agent_input else recovered["prompt"]
        if agent_input:
            record["agent_prompt"] = recovered["prompt"]
            record["trace"]["agent_request"] = {
                "messages": recovered["messages"], "tools": recovered["tools"],
                "limitations": recovered["limitations"],
            }
            record["trace"]["evidence_scope"]["agent_request"] = (
                "Original request to the agent; exact first model request not recorded"
            )
        elif recovered.get("messages"):
            record["trace"]["initial_request"] = {
                "messages": recovered["messages"], "tools": recovered["tools"],
            }
        if recovered.get("task_statement"):
            record["task_prompt"] = (
                "[Benchmark task statement; exact initial model request not recorded]\n"
                + recovered["task_statement"]
            )
            record["trace"]["evidence_scope"]["task_input"] = (
                "Original benchmark problem_statement, not an exact model-request capture"
            )
        record["trace"]["agent"]["tool_definitions"] = recovered["tools"]
        if recovered["prompt"] and not agent_input:
            record["trace"]["evidence_scope"]["initial_request"] = "original first model request"
        for limitation in recovered["limitations"]:
            limits[limitation] = limits.get(limitation, 0) + 1
        if model_calls["calls"]:
            _reconcile_request_capture_evidence(record["trace"], model_calls)
            steps = model_calls.pop("steps")
            raw_limits = model_calls.pop("limitations")
            raw_notices = model_calls.pop("notices", {})
            record["trace"]["model_call_evidence"] = model_calls
            record["n_model_calls_recovered"] = len(model_calls["calls"])
            scope = record["trace"]["evidence_scope"]
            scope["model_calls"] = (
                "Recorded requests and responses; message/tool pools are indexed by each call. "
                "Repeated context is retained separately, not counted as repeated actions."
            )
            if (not record["trace"].get("steps")
                    and any(s["evidence_origin"] == "response" for s in steps)):
                record["trace"]["steps"] = steps
                execution = record["trace"].setdefault("execution", {})
                execution["recovered_steps_source"] = "steps"
                execution.setdefault("availability", {})["recovered_steps"] = {
                    "status": "partial",
                    "reason": "Steps reconstructed from captured model requests and responses; "
                              "repeated context is deduplicated; API event IDs are not inferred.",
                }
                record["raw_trajectory_recovered"] = True
                scope["trajectory_source"] = "raw ng_trajectory.model_calls"
                limits.pop("source rollout contains no events", None)
            for limitation, count in raw_limits.items():
                limits[limitation] = limits.get(limitation, 0) + count
            if raw_notices:
                notices = scope.setdefault("notices", {})
                for notice, count in raw_notices.items():
                    notices[notice] = notices.get(notice, 0) + count
        elif excluded := model_calls["limitations"].get(EXCLUDED_CALL):
            # Every captured call was an evaluation record; it is withheld, not absent.
            limits[EXCLUDED_CALL] = limits.get(EXCLUDED_CALL, 0) + excluded
