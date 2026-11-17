# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provenance and viewer addresses for cloud exports and managed local runs."""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

LOCAL_REFERENCE_KIND = "local_evalrdl"
_STORE_ID = re.compile(r"[0-9a-f]{64}\Z")
_LOCAL_VIEWER_PATH = re.compile(r"/traces/[0-9a-f]{64}/\Z")


def local_viewer_base(reference: Mapping[str, Any]) -> str:
    """A port-independent viewer path for a validated managed-store reference."""
    store_id = reference.get("store_id")
    if (reference.get("kind") != LOCAL_REFERENCE_KIND
            or not isinstance(store_id, str) or not _STORE_ID.fullmatch(store_id)
            or not str(reference.get("run_id") or "").strip()):
        return ""
    return f"/traces/{store_id}/"


def make_local_reference_data(
    *, store_id: str, run_id: str, source_harness: str = "unknown",
    parquet_path: str | Path | None = None,
) -> dict[str, str]:
    """Build a managed local reference without inventing a cloud export URL."""
    record = {
        "kind": LOCAL_REFERENCE_KIND, "store_id": store_id,
        "run_id": str(run_id).strip(), "source_harness": _clean_harness(source_harness),
    }
    if not local_viewer_base(record):
        raise ValueError("Local reference requires a lowercase 64-hex store_id and run_id")
    if parquet_path:
        record["parquet_path"] = str(Path(parquet_path).expanduser().resolve())
    return record


def canonical_export_url(base_url: str, run_id: str) -> str:
    """Return the exact AutoFA export endpoint for one Data API run."""
    rid = str(run_id).strip()
    if not rid:
        raise ValueError("Data API run_id is required")
    try:
        parsed = urllib.parse.urlsplit(str(base_url).strip())
    except ValueError as exc:
        raise ValueError("Data API base URL is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Data API base URL must be an HTTP(S) origin or path")
    path = (
        f"{parsed.path.rstrip('/')}/v1/runs/"
        f"{urllib.parse.quote(rid, safe='')}/export/autofa"
    )
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def parse_evalrdl_run_url(value: str) -> tuple[str, str, str]:
    text = str(value or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError as exc:
        raise ValueError("EvalRDL URL is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
    ):
        raise ValueError("EvalRDL URL must be an HTTP(S) URL without credentials or query")
    run_id = ""
    base_path = ""
    if parsed.fragment:
        pieces = parsed.fragment.strip("/").split("/")
        if len(pieces) != 2 or pieces[0] != "run" or not pieces[1]:
            raise ValueError("EvalRDL inspector URL must end with #/run/<run_id>")
        run_id = urllib.parse.unquote(pieces[1]).strip()
        base_path = parsed.path.rstrip("/")
    else:
        match = re.fullmatch(
            r"(?P<base>.*)/v1/runs/(?P<run>[^/]+)/export/autofa",
            parsed.path,
        )
        if match is None:
            raise ValueError(
                "EvalRDL URL must be a run inspector or canonical /export/autofa URL"
            )
        run_id = urllib.parse.unquote(match.group("run")).strip()
        base_path = match.group("base").rstrip("/")
    if not run_id or "/" in run_id:
        raise ValueError("EvalRDL URL has an invalid run_id")
    base_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))
    return base_url, run_id, canonical_export_url(base_url, run_id)


def make_reference_data(
    *,
    run_id: str,
    export_url: str,
    source_harness: str = "unknown",
    parquet_path: str | Path | None = None,
    evidence_scope: str = "",
) -> dict[str, str]:
    """Build one normalized local reference record."""
    record = {
        "run_id": str(run_id).strip(),
        "export_url": str(export_url).strip(),
        "source_harness": _clean_harness(source_harness),
    }
    if parquet_path:
        record["parquet_path"] = str(Path(parquet_path).expanduser().resolve())
    if evidence_scope:
        record["evidence_scope"] = evidence_scope
    return record


def remote_evidence_scope(
    *,
    efficiency_only: bool = False,
    trace_manifest: Mapping[str, Any] | None = None,
) -> str:
    """Describe evidence visibility for a prepared remote EvalRDL export."""
    if efficiency_only:
        return "Efficiency tables only; raw rollout evidence was not downloaded."
    if trace_manifest is None:
        return (
            "Summary-only API export: intermediate messages, tool observations, images "
            "and artifact contents may be absent. "
            "Harness inspection is not validated by this input."
        )
    limitations = trace_manifest.get("limitations") or {}
    notices = trace_manifest.get("notices") or {}
    return (
        f"Original text traces: {trace_manifest['n_rows']} rollouts, "
        f"{trace_manifest['n_events']} events, "
        f"{trace_manifest['n_tool_calls']} tool calls retrieved. "
        f"{trace_manifest.get('n_initial_requests_recovered', 0)} first model requests "
        "recovered with recorded input text/tool definitions. "
        f"{trace_manifest.get('n_model_calls_recovered', 0)} raw model calls retained, "
        "including available later request snapshots. "
        f"{trace_manifest.get('n_raw_trajectories_recovered', 0)} missing trajectories "
        "recovered. Retrieval does not mean every event was reviewed. Images and artifact "
        "file contents are not inspected. "
        + "; ".join(
            f"Evidence limitation: {reason}: {count} occurrences"
            for reason, count in limitations.items()
        )
        + (
            ". Source notes (capture format and context): "
            + "; ".join(
                f"{reason}: {count} model requests"
                for reason, count in notices.items()
            )
            if notices
            else ""
        )
    )


def remote_reference_details(
    export: Path,
    catalog: Mapping[str, Any] | None,
    *,
    full_traces: bool = False,
    efficiency_only: bool = False,
) -> tuple[str, str, dict[str, Any] | None]:
    """Read trace provenance once and derive reference harness and evidence scope."""
    trace_manifest = None
    if full_traces:
        trace_manifest = json.loads(
            (Path(export).parent / "manifest.json").read_text(encoding="utf-8")
        )
    source_harness = str((catalog or {}).get("harness") or "unknown").strip()
    if trace_manifest and trace_manifest.get("agent_refs"):
        source_harness = ", ".join(trace_manifest["agent_refs"])
    return (
        source_harness or "unknown",
        remote_evidence_scope(
            efficiency_only=efficiency_only,
            trace_manifest=trace_manifest,
        ),
        trace_manifest,
    )


def normalize_reference_data(
    value: Any,
    *,
    include_local_paths: bool = True,
) -> list[dict[str, str]]:
    """Normalize reference records and remove duplicates."""
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        run_id = str(raw.get("run_id") or "").strip()
        export_url = str(raw.get("export_url") or "").strip()
        if raw.get("kind") == LOCAL_REFERENCE_KIND:
            if not local_viewer_base(raw):
                continue
            item = {
                "kind": LOCAL_REFERENCE_KIND, "store_id": raw["store_id"],
                "run_id": run_id, "source_harness": _clean_harness(raw.get("source_harness")),
            }
            key = (LOCAL_REFERENCE_KIND, raw["store_id"], run_id)
        else:
            if not run_id or not export_url:
                continue
            item = {
                "run_id": run_id, "export_url": export_url,
                "source_harness": _clean_harness(raw.get("source_harness")),
            }
            key = ("cloud", run_id, export_url)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(raw.get("evidence_scope"), str) and raw["evidence_scope"].strip():
            item["evidence_scope"] = raw["evidence_scope"].strip()
        parquet_path = str(raw.get("parquet_path") or "").strip()
        if include_local_paths and parquet_path:
            item["parquet_path"] = str(Path(parquet_path).expanduser().resolve())
        normalized.append(item)
    return normalized


def require_canonical_reference_data(
    value: Any,
    parquet_paths: Iterable[str | Path],
) -> list[dict[str, str]]:
    """Require every analysis parquet to be owned by one canonical EvalRDL reference."""
    references = normalize_reference_data(value)
    selected = [str(Path(path).expanduser().resolve()) for path in parquet_paths]
    if not references or not selected:
        raise ValueError(
            "Efficiency analysis requires data imported through EvalRDL. "
            "Select a supported run source and retry."
        )
    if len(selected) != len(set(selected)):
        raise ValueError("The efficiency selection contains duplicate dataset paths.")

    owned: list[str] = []
    for reference in references:
        if reference.get("kind") == LOCAL_REFERENCE_KIND:
            from autofa.data.local_trace_store import resolve_store

            expected = str((resolve_store(reference["store_id"]) / "dataset.parquet").resolve())
            recorded = str(reference.get("parquet_path") or "").strip()
            if recorded:
                recorded = str(Path(recorded).expanduser().resolve())
                if recorded != expected:
                    expected = _local_cohort_dataset(reference, recorded)
        else:
            if data_api_location(reference) is None:
                raise ValueError("A remote EvalRDL reference is invalid.")
            recorded = str(reference.get("parquet_path") or "").strip()
            if not recorded:
                raise ValueError("A remote EvalRDL reference is missing its materialized dataset.")
            expected = str(Path(recorded).expanduser().resolve())
        owned.append(expected)

    if len(owned) != len(set(owned)) or set(owned) != set(selected):
        raise ValueError(
            "Every efficiency dataset must belong to exactly one selected EvalRDL run. "
            "Remove unreferenced or mismatched files and retry."
        )
    return references


def _local_cohort_dataset(reference: Mapping[str, Any], recorded: str) -> str:
    from autofa.data.cohort import cohorts_root, load_cohort

    path = Path(recorded)
    root = cohorts_root().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError(
            "A local EvalRDL reference does not own its recorded dataset."
        ) from None
    if len(relative.parts) != 3 or relative.parts[1] != "datasets":
        raise ValueError("A local EvalRDL reference does not own its recorded dataset.")
    try:
        cohort = load_cohort(root / relative.parts[0])
    except (OSError, ValueError, TypeError):
        raise ValueError(
            "A local EvalRDL reference does not own its recorded dataset."
        ) from None
    owned = any(
        dataset.get("kind") == LOCAL_REFERENCE_KIND
        and dataset.get("store_id") == reference.get("store_id")
        and dataset.get("run_id") == reference.get("run_id")
        and str(Path(dataset.get("parquet_path") or "").expanduser().resolve()) == recorded
        for dataset in cohort.datasets
    )
    if not owned or not path.is_file():
        raise ValueError("A local EvalRDL reference does not own its recorded dataset.")
    return recorded


def public_reference_data(value: Any) -> list[dict[str, str]]:
    """Cloud publication excludes machine-local registries and cache paths."""
    return [reference for reference in normalize_reference_data(value, include_local_paths=False)
            if reference.get("kind") != LOCAL_REFERENCE_KIND]


def summarize_source_harness(
    values: Iterable[str | None],
    *,
    default: str = "unknown",
) -> str:
    """Collapse one or more source harnesses to a dashboard-friendly value."""
    harnesses = {
        cleaned
        for value in values
        if (cleaned := _clean_harness(value, default="")) and cleaned != "unknown"
    }
    if not harnesses:
        return default
    if len(harnesses) == 1:
        return next(iter(harnesses))
    return "mixed"


def inspect_url(reference: Mapping[str, Any]) -> str:
    """Derive the Data API run inspector from a canonical export URL."""
    if base := local_viewer_base(reference):
        return f"{base}#/run/{urllib.parse.quote(str(reference['run_id']).strip(), safe='')}"
    parsed = _parsed_reference(reference)
    if parsed is None:
        return ""
    scheme, netloc, base_path, run_id = parsed
    return (
        f"{scheme}://{netloc}{base_path}/#/run/"
        f"{urllib.parse.quote(run_id, safe='')}"
    )


def data_api_location(
    reference: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Return ``(base_url, run_id)`` for a validated export reference."""
    parsed = _parsed_reference(reference)
    if parsed is None:
        return None
    scheme, netloc, base_path, run_id = parsed
    return f"{scheme}://{netloc}{base_path}", run_id


def rollout_url(
    reference: Mapping[str, Any],
    *,
    problem_id: Any,
    run_idx: Any,
) -> str:
    """Derive the exact Data API rollout inspector URL for an AutoFA row."""
    parsed = _parsed_reference(reference)
    local_base = local_viewer_base(reference)
    task_id = str(problem_id) if problem_id is not None else ""
    if not local_base:
        task_id = task_id.strip()
    repeat = str(run_idx).strip() if run_idx is not None else ""
    if (parsed is None and not local_base) or not task_id or not repeat:
        return ""
    if local_base:
        base, run_id = local_base, str(reference["run_id"]).strip()
    else:
        scheme, netloc, base_path, run_id = parsed
        base = f"{scheme}://{netloc}{base_path}/"
    # The Data API SPA double-escapes slash characters inside task IDs.
    encoded_task = urllib.parse.quote(task_id, safe="").replace("%2F", "%252F")
    return (
        f"{base}#/rollout/"
        f"{urllib.parse.quote(run_id, safe='')}/{encoded_task}/"
        f"{urllib.parse.quote(repeat, safe='')}"
    )


def event_url(
    rollout: str, source_event: Any, text: str | None = None,
    *, context: Mapping[str, Any] | None = None,
) -> str:
    """EvalRDL event/part deep link; lane is part of the event's identity.

    The viewer accepts #/rollout/run/task/repeat/e25-p0?lane=agent, and `text`, which it
    highlights in the event: URL text-fragment syntax, a long passage as its first and last words.
    """
    if not isinstance(source_event, dict):
        return ""
    index, part = source_event.get("event_idx"), source_event.get("part_idx")
    lane = source_event.get("lane")
    if (type(index) is not int or index < 0 or not isinstance(lane, str)
            or (part is not None and (type(part) is not int or part < 0))):
        return ""
    try:
        parsed = urllib.parse.urlsplit(rollout)
    except ValueError:
        return ""
    local = (not parsed.scheme and not parsed.netloc
             and bool(_LOCAL_VIEWER_PATH.fullmatch(parsed.path)))
    remote = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    if (not (local or remote)
            or parsed.username or parsed.password or parsed.query
            or not parsed.fragment.startswith("/rollout/")
            or len(parsed.fragment.split("/")) != 5 or "?" in parsed.fragment):
        return ""
    anchor = f"/e{index}" + (f"-p{part}" if part is not None else "")
    query = urllib.parse.urlencode({"lane": lane})
    words = str(text or "").split() if not (context or {}).get("unavailable") else []
    if words:
        piece = lambda w: urllib.parse.quote(" ".join(w), safe="").replace("-", "%2D")  # noqa: E731
        whole = " ".join(words)
        value = (piece(words) if context is not None or len(whole) <= 300 or len(words) <= 16
                 else piece(words[:8]) + "," + piece(words[-8:]))
        if context:
            before, after = str(context.get("prefix") or "").split(), str(
                context.get("suffix") or "",
            ).split()
            if before:
                value = piece(before) + "-," + value
            if after:
                value += ",-" + piece(after)
        query += "&text=" + value
    return rollout + anchor + "?" + query


def reference_for_parquet(
    references: Any,
    parquet_path: str | Path | None,
) -> dict[str, str] | None:
    """Resolve an item source parquet to its Data API provenance record."""
    normalized = normalize_reference_data(references)
    if not normalized:
        return None
    candidate = str(parquet_path or "").strip()
    if candidate:
        try:
            candidate = str(Path(candidate).expanduser().resolve())
        except OSError:
            pass
        for reference in normalized:
            if reference.get("parquet_path") == candidate:
                return reference
        if any(reference.get("parquet_path") for reference in normalized):
            return None
    return normalized[0] if len(normalized) == 1 else None


def _parsed_reference(
    reference: Mapping[str, Any],
) -> tuple[str, str, str, str] | None:
    if reference.get("kind") == LOCAL_REFERENCE_KIND:
        return None
    run_id = str(reference.get("run_id") or "").strip()
    export_url = str(reference.get("export_url") or "").strip()
    if not run_id or not export_url:
        return None
    try:
        parsed = urllib.parse.urlsplit(export_url)
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return None
    suffix = (
        f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/export/autofa"
    )
    if not parsed.path.endswith(suffix):
        return None
    base_path = parsed.path[: -len(suffix)]
    if base_path.endswith("/"):
        return None
    return parsed.scheme, parsed.netloc, base_path, run_id


def _clean_harness(value: Any, *, default: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default
