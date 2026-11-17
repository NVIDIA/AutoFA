# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover source observability resources only on explicit agent request.

Archive tables remain authoritative. Optional API reads fill unavailable resources
and are cached independently, so subsequent reads need not repeat them.
Prompt summaries describe history changes; they are never used as full messages.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

from autofa.data.execution_policy import is_evaluation_record, sanitize_metadata
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.util.atomic import atomic_write_json

_RESOURCES = {"provenance": "rows", "relations": "edges", "prompt_summaries": "calls"}


def _rows(payload, resource: str, *, run_id: str, task_id: str, repeat: int) -> list[dict]:
    values = payload if isinstance(payload, list) else (
        payload.get(_RESOURCES[resource]) if isinstance(payload, dict) else None
    )
    if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
        raise DataApiError(f"Invalid source {resource}: expected complete record list")
    if isinstance(payload, dict) and (
        payload.get("run_id") not in (None, run_id)
        or payload.get("task_id") not in (None, task_id)
        or payload.get("repeat") not in (None, repeat)
    ):
        raise DataApiError(f"Source {resource} response belongs to another rollout")
    if resource in {"relations", "provenance"} and isinstance(payload, dict) and "n" in payload:
        if type(payload["n"]) is not int or payload["n"] != len(values):
            raise DataApiError(f"Source {resource} count does not match returned records")
    result = []
    for row in values:
        if row.get("run_id") not in (None, run_id):
            raise DataApiError(f"Source {resource} belongs to another run")
        if resource != "relations" and (
            row.get("task_id") not in (None, task_id)
            or row.get("repeat") not in (None, repeat)
        ):
            raise DataApiError(f"Source {resource} belongs to another rollout")
        if resource == "relations":
            endpoints = [(row.get(f"{side}_task_id"), row.get(f"{side}_repeat"))
                         for side in ("frm", "to")]
            if any(task is not None for task, _ in endpoints) and not any(
                task == task_id and attempt in (None, repeat) for task, attempt in endpoints
            ):
                raise DataApiError("Source relation does not touch the requested rollout")
        if not is_evaluation_record(row):
            result.append(sanitize_metadata(row))
    return result


def recover_execution_resources(
    record: dict, *, api, run_id: str, root: Path,
    resources: tuple[str, ...] = tuple(_RESOURCES),
) -> None:
    """Fill missing metadata resources without pretending previews are complete requests."""
    if any(resource not in _RESOURCES for resource in resources):
        raise ValueError("Only provenance, relations and prompt_summaries can be requested")
    execution = record["trace"].setdefault("execution", {})
    availability = execution.setdefault("availability", {})
    metadata = record["metadata"]
    task_id, repeat = metadata["task_id"], metadata["repeat"]
    reader = getattr(api, "get_trace_resource", None)
    for resource in resources:
        if availability.get(resource, {}).get("status") == "available":
            continue
        execution.setdefault(resource, [])
        if not callable(reader):
            availability.setdefault(resource, {
                "status": "unavailable", "reason": "not exported; optional API reader unavailable",
            })
            continue
        identity = [getattr(api, "base_url", ""), run_id, task_id, repeat, resource]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        cached = root / f"{key}.json"
        entry = None
        if cached.is_file():
            try:
                entry = json.loads(cached.read_text())
                if (not isinstance(entry, dict) or entry.get("identity") != identity
                        or ("payload" in entry) == ("unavailable" in entry)):
                    entry = None
                elif "unavailable" in entry:
                    if not isinstance(entry["unavailable"], str):
                        entry = None
                else:
                    _rows(entry["payload"], resource, run_id=run_id,
                          task_id=task_id, repeat=repeat)
            except (OSError, ValueError, DataApiError):
                entry = None
        if entry is None:
            try:
                payload = reader(run_id, task_id, repeat, resource)
            except DataApiError as exc:
                if exc.status not in {404, 405, 501}:
                    raise
                entry = {"identity": identity, "unavailable": f"source API returned {exc.status}"}
            else:
                # Validate before certifying/caching the response. Raw cache is not detector input.
                _rows(payload, resource, run_id=run_id, task_id=task_id, repeat=repeat)
                entry = {"identity": identity, "payload": payload}
            atomic_write_json(cached, entry)
        if "unavailable" in entry:
            existing = execution.get(resource)
            availability[resource] = {
                "status": "partial" if existing else "unavailable",
                "reason": entry["unavailable"]
                + ("; previously imported records retained" if existing else ""),
            }
            continue
        values = _rows(entry["payload"], resource, run_id=run_id,
                       task_id=task_id, repeat=repeat)
        execution[resource] = values
        payload = entry["payload"]
        if isinstance(payload, dict):
            execution.setdefault("resource_metadata", {})[resource] = sanitize_metadata({
                key: value for key, value in payload.items() if key != _RESOURCES[resource]
            })
        partial = isinstance(payload, dict) and any(
            payload.get(key) is True for key in ("truncated", "has_more", "aggregate")
        )
        availability[resource] = {
            "status": "partial" if partial else "available",
            "reason": "source API records; evaluation records excluded"
            + ("; source reports incomplete or aggregated records" if partial else "")
            + ("; summaries only, use model_calls/events for captured request contents"
               if resource == "prompt_summaries" else ""),
        }


def fetch_execution_resource(
    execution: dict, resource: str, *, root: Path, api=None, base_url: str = "",
) -> None:
    """Enrich a caller-owned copy, keeping the imported dataset and detector inputs stable."""
    if resource not in _RESOURCES:
        raise ValueError("Only provenance, relations and prompt_summaries can be requested")
    if execution.get("availability", {}).get(resource, {}).get("status") == "available":
        return
    source, rollout = execution.get("source"), execution.get("rollout")
    if not isinstance(source, dict) or source.get("kind") != "trace_api":
        raise ValueError("No trace API source recorded for this item")
    if not isinstance(rollout, dict):
        raise ValueError("No source rollout identity recorded for this item")
    api = api if api is not None else DataApiClient(base_url)

    def origin(value):
        if not isinstance(value, str):
            raise ValueError("Source API base URL is unavailable")
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Invalid source API base URL")
        return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(),
                                       parsed.path.rstrip("/"), "", ""))

    base_url = origin(source.get("base_url"))
    if base_url != origin(api.base_url):
        raise ValueError("Recorded source differs from the configured trusted API")
    for key in ("run_id", "task_id", "repeat"):
        value = rollout.get(key)
        valid = type(value) is int and value >= 0 if key == "repeat" else (
            isinstance(value, str) and bool(value) and value not in {".", ".."}
        )
        if not valid:
            raise ValueError("Incomplete source rollout identity")
    revision = source.get("trace_export_sha256")
    if (not isinstance(revision, str) or len(revision) != 64
            or any(char not in "0123456789abcdef" for char in revision)):
        raise ValueError("Missing or invalid trace export revision; reimport this source")
    generation = source.get("export_generation")
    if generation is not None and type(generation) not in (str, int):
        raise ValueError("Invalid source export generation")
    namespace = hashlib.sha256(json.dumps([base_url, revision, generation]).encode()).hexdigest()
    recover_execution_resources(
        {"trace": {"execution": execution}, "metadata": rollout},
        api=api, run_id=rollout["run_id"], root=root / namespace, resources=(resource,),
    )
