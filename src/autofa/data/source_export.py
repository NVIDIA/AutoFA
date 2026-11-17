# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize original API events into AutoFA's existing steps format.

The compact AutoFA export contains only the final assistant response, not the
tool observations. Keep that export intact and build a separate, checked cache.
Execution metadata is retained; the shared policy separates external targets and credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.parse
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from autofa.data.evidence_refs import compact_trace
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.source_normalization import _validate_identities, normalize_events

ADAPTER_VERSION = 15
GRADER_CALLS_EXCLUDED_VERSION = 15  # earlier prepared caches can retain graders' calls


def require_current_source_cache(path: Path) -> None:
    """Refuse rows from a source cache prepared before graders' calls were excluded.

    Such caches can retain graders' model calls, so their rows are never loaded.
    Local import stores version their own manifests, which hold no ``output_sha256``.
    """
    root = Path(path).parent
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError):
        return
    version = manifest.get("adapter_version") if isinstance(manifest, dict) else None
    if (
        type(version) is int
        and version < GRADER_CALLS_EXCLUDED_VERSION
        and "output_sha256" in manifest
    ):
        raise DataApiError(
            f"Source cache {root.name} was prepared by an earlier AutoFA version and can "
            "expose graders' model calls; re-prepare the source by selecting its run in a "
            "new analysis"
        )


def _write_json(path: Path, value: Any) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Serialize builders of one cache directory across threads and processes."""
    with path.open("a+b") as stream:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _get(api: DataApiClient, path: str, query: dict | None = None) -> Any:
    for attempt in range(3):
        try:
            return api._get_json(path, query)
        except (DataApiError, TimeoutError, ConnectionError) as exc:
            transient = not isinstance(exc, DataApiError) or (
                exc.status in {429, 500, 502, 503, 504} or str(exc).startswith("Network error")
            )
            if not transient or attempt == 2:
                raise
            time.sleep(2 ** attempt)


def materialize_source_traces(
    export_path: Path, *, api: DataApiClient, run_id: str,
    progress: Callable[[int, int], None] | None = None, workers: int = 4,
    catalog: dict[str, Any] | None = None,
    download_progress: Callable[[int, int | None], None] | None = None,
    raw_download_progress: Callable[[int, int], None] | None = None,
    activity: Callable[[str], None] | None = None,
) -> Path:
    """Preserve all rows; fail visibly rather than substitute shortened evidence.

    A checked bulk ZIP and disk index make interrupted imports resumable. A changed
    compact export, explicit refresh or adapter version gets a distinct directory. Outcomes stay in
    the original columns and are never inserted into the detector-visible trace.
    """
    export_path = Path(export_path)
    if activity:
        activity("Verifying source cache…")
    # The API accepts legacy run aliases but returns canonical IDs in trace rows.
    run_meta = catalog if catalog is not None else _get(
        api, f"/v1/runs/{urllib.parse.quote(run_id, safe='')}",
    )
    canonical_id = run_meta.get("run_id") if isinstance(run_meta, dict) else None
    if not isinstance(canonical_id, str) or not canonical_id.strip():
        raise DataApiError("Source API run lookup did not resolve a canonical run ID")
    canonical_id = canonical_id.strip()
    with export_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    source = export_path.parent / "source.json"
    try:
        source_meta = json.loads(source.read_text()) if source.is_file() else {}
    except ValueError as exc:
        raise DataApiError("Invalid source cache metadata") from exc
    if not isinstance(source_meta, dict):
        raise DataApiError("Invalid source cache metadata")
    generation = source_meta.get("export_generation")
    identity = json.dumps([
        api.base_url.rstrip("/"), canonical_id, digest, ADAPTER_VERSION, generation,
    ])
    cache_key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    root = export_path.parent / f"source-v{ADAPTER_VERSION}-{cache_key}"
    root.mkdir(parents=True, exist_ok=True)
    # A concurrent builder of the same cache waits, then reuses the verified result.
    with _exclusive(root / ".lock"):
        return _materialize(
            export_path, root, api=api, run_id=run_id, canonical_id=canonical_id,
            digest=digest, generation=generation, run_meta=run_meta, progress=progress,
            download_progress=download_progress, raw_download_progress=raw_download_progress,
            activity=activity,
        )


def materialize_efficiency_tables(
    export_path: Path,
    *,
    api: DataApiClient,
    run_id: str,
    catalog: dict[str, Any] | None = None,
    download_progress: Callable[[int, int | None], None] | None = None,
    activity: Callable[[str], None] | None = None,
) -> Path:
    """Cache the EvalRDL tables needed by efficiency without building trace text."""
    export_path = Path(export_path)
    run_meta = catalog if catalog is not None else _get(
        api, f"/v1/runs/{urllib.parse.quote(run_id, safe='')}",
    )
    canonical_id = run_meta.get("run_id") if isinstance(run_meta, dict) else None
    if not isinstance(canonical_id, str) or not canonical_id.strip():
        raise DataApiError("Source API run lookup did not resolve a canonical run ID")
    canonical_id = canonical_id.strip()
    with export_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    source = export_path.parent / "source.json"
    try:
        source_meta = json.loads(source.read_text()) if source.is_file() else {}
    except ValueError as exc:
        raise DataApiError("Invalid source cache metadata") from exc
    generation = source_meta.get("export_generation") if isinstance(source_meta, dict) else None

    compact = pq.ParquetFile(export_path)
    identity_columns = ["problem_id", "run_idx"]
    if "lane" in compact.schema_arrow.names:
        identity_columns.append("lane")
    identities: list[tuple[str, int, str]] = []
    for batch in compact.iter_batches(batch_size=1024, columns=identity_columns):
        for row in batch.to_pylist():
            task, repeat, lane = row["problem_id"], row["run_idx"], row.get("lane", "")
            if (
                not isinstance(task, str)
                or not task
                or type(repeat) is not int
                or repeat < 0
                or not isinstance(lane, str)
            ):
                raise DataApiError("AutoFA export has invalid rollout identities or lane")
            identities.append((task, repeat, lane))
    if not identities:
        raise DataApiError("AutoFA export has no rollouts")
    if len({(task, repeat) for task, repeat, _lane in identities}) != len(identities):
        raise DataApiError("AutoFA export has duplicate task/repeat identities")

    from autofa.data.bulk_trace import TraceArchives

    total_rollouts = run_meta.get("n_rollouts")
    whole_run = type(total_rollouts) is int and total_rollouts == len(identities)
    selection = None if whole_run else [(task, repeat) for task, repeat, _lane in identities]
    bulk_identity = json.dumps(
        [api.base_url.rstrip("/"), canonical_id, digest, generation, selection]
    )
    bulk_root = export_path.parent / (
        "source-bulk-" + hashlib.sha256(bulk_identity.encode()).hexdigest()[:20]
    )
    lanes = {(task, repeat): lane for task, repeat, lane in identities}
    with TraceArchives(
        bulk_root,
        api=api,
        run_id=canonical_id,
        identities=[(task, repeat) for task, repeat, _lane in identities],
        whole_run=whole_run,
        progress=download_progress,
        activity=activity,
    ) as bulk:
        for selected in bulk.selections:
            task, repeat = (
                (identities[0][0], identities[0][1])
                if selected is None
                else selected[0]
            )
            bulk.rollout(task, repeat, lane=lanes[(task, repeat)])
        exports = list(bulk.exports)
    marker = (
        export_path.parent / "efficiency-tables.json"
        if export_path.name == "dataset.parquet"
        else export_path.with_name(f"{export_path.name}.efficiency-tables.json")
    )
    _write_json(
        marker,
        {
            "run_id": canonical_id,
            "export_sha256": digest,
            "trace_exports": exports,
        },
    )
    return export_path


def _materialize(
    export_path: Path, root: Path, *, api: DataApiClient, run_id: str, canonical_id: str,
    digest: str, generation: Any, run_meta: dict[str, Any],
    progress: Callable[[int, int], None] | None,
    download_progress: Callable[[int, int | None], None] | None,
    raw_download_progress: Callable[[int, int], None] | None,
    activity: Callable[[str], None] | None,
) -> Path:
    out = root / "dataset.parquet"
    if out.is_file() and (root / "manifest.json").is_file():
        try:
            manifest = json.loads((root / "manifest.json").read_text())
        except ValueError:
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        with out.open("rb") as handle:
            if manifest.get("output_sha256") == hashlib.file_digest(handle, "sha256").hexdigest():
                return out
    compact = pq.ParquetFile(export_path)
    required = {"problem_id", "run_idx", "response_text", "reasoning_text"}
    if not required.issubset(compact.schema_arrow.names) or not compact.metadata.num_rows:
        raise DataApiError("AutoFA export lacks rollout identity/content columns or has no rows")
    identity_columns = ["problem_id", "run_idx"]
    if "lane" in compact.schema_arrow.names:
        identity_columns.append("lane")
    identities = []
    for batch in compact.iter_batches(batch_size=1024, columns=identity_columns):
        for row in batch.to_pylist():
            task, repeat, lane = row["problem_id"], row["run_idx"], row.get("lane", "")
            if (not isinstance(task, str) or not task or type(repeat) is not int
                    or repeat < 0 or not isinstance(lane, str)):
                raise DataApiError("AutoFA export has invalid rollout identities or lane")
            identities.append((task, repeat, lane))
    if len({(t, r) for t, r, _ in identities}) != len(identities):
        raise DataApiError("AutoFA export has duplicate task/repeat identities")
    from autofa.data.bulk_trace import SELECTION_BATCH_SIZE, TraceArchives

    total_rollouts = run_meta.get("n_rollouts")
    whole_run = type(total_rollouts) is int and total_rollouts == len(identities)
    selection = None if whole_run else [(task, repeat) for task, repeat, _ in identities]
    from autofa.data.source_inputs import cache_raw_source_file, recover_initial_inputs
    from autofa.data.source_trajectory import recover_raw_trajectory

    input_root = root / "initial-inputs"
    _reuse_raw_downloads(
        input_root, export_path.parent, api.base_url.rstrip("/"), canonical_id, digest, generation,
    )
    source_files: dict[str, Path] = {}
    counts: Counter[str] = Counter()
    limitations: Counter[str] = Counter()
    notices: Counter[str] = Counter()
    agent_refs: set[str] = set()
    schema = compact.schema_arrow
    for column in ("prompt_text", "response_text", "reasoning_text"):
        index = schema.get_field_index(column)
        field = pa.field(column, pa.string())
        schema = schema.set(index, field) if index >= 0 else schema.append(field)
    if "input_json" not in schema.names:
        schema = schema.append(pa.field("input_json", pa.string()))
    bulk_identity = json.dumps([
        api.base_url.rstrip("/"), canonical_id, digest, generation, selection,
    ])
    bulk_key = hashlib.sha256(bulk_identity.encode()).hexdigest()[:20]
    bulk_root = export_path.parent / f"source-bulk-{bulk_key}"
    if whole_run or len(identities) <= SELECTION_BATCH_SIZE:
        _reuse_bulk_download(
            bulk_root, export_path.parent, api.base_url.rstrip("/"), canonical_id, digest,
            generation, selection,
        )
    temp = out.with_name(f".{out.name}.{uuid.uuid4().hex}.partial")
    try:
        with TraceArchives(bulk_root, api=api, run_id=canonical_id,
                           identities=[(task, repeat) for task, repeat, _ in identities],
                           whole_run=whole_run, progress=download_progress,
                           activity=activity) as bulk, \
                pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            for index, batch in enumerate(compact.iter_batches(batch_size=1)):
                task, repeat, lane = identities[index]
                raw = bulk.rollout(task, repeat, lane=lane)
                _validate_identities(raw["metadata"], raw["events"], raw["calls"])
                trace = normalize_events(**raw, conversation_lane=lane)
                trace["execution"]["source"] = {
                    "kind": "trace_api", "base_url": api.base_url.rstrip("/"),
                    "trace_export_sha256": bulk.digest, "export_generation": generation,
                }
                record = {"trace": trace, "metadata": raw["metadata"]}
                counts["n_events"] += len(raw["events"])
                counts["n_tool_calls"] += len(raw["calls"])
                del raw
                if whole_run:
                    cache_raw_source_file(
                        record, api=api, run_id=canonical_id, root=input_root,
                        source_files=source_files, download_progress=raw_download_progress,
                        activity=activity,
                    )
                recover_initial_inputs(
                    [record], api=api, run_id=canonical_id, root=input_root,
                    source_files=source_files, download_progress=raw_download_progress,
                    activity=activity,
                )
                recover_raw_trajectory(
                    record, api=api, run_id=canonical_id, root=input_root,
                    source_files=source_files, download_progress=raw_download_progress,
                    activity=activity,
                )
                initial_prompt = record.get("initial_prompt")
                if not initial_prompt and trace.get("initial_request"):
                    from autofa.data.structured import input_view

                    initial_prompt = input_view(trace["initial_request"])["text"]
                row = batch.to_pylist()[0]
                if (not row.get("input_json") and not initial_prompt
                        and not record.get("agent_prompt") and not record.get("task_prompt")
                        and not trace.get("initial_request") and not trace.get("agent_request")
                        and pa.types.is_string(schema.field("input_json").type)):
                    from autofa.data.input_view import compact_export_input_json

                    row["input_json"] = compact_export_input_json(row)
                row["prompt_text"] = (initial_prompt or record.get("agent_prompt")
                                      or record.get("task_prompt")
                                      or row.get("prompt_text") or "")
                if not row["prompt_text"] and not any(
                    s.get("source") in {"system", "developer", "user"} and s.get("message")
                    for s in trace["steps"]
                ):
                    trace["evidence_scope"]["limitations"]["task/user input unavailable"] = 1
                row["response_text"] = json.dumps(compact_trace(trace), ensure_ascii=False)
                # Never let capped compact reasoning shadow the recovered steps.
                row["reasoning_text"] = ""
                writer.write_table(pa.Table.from_pylist([row], schema=schema))
                counts["n_initial_requests_recovered"] += bool(initial_prompt)
                counts["n_agent_requests_recovered"] += bool(record.get("agent_prompt"))
                counts["n_model_calls_recovered"] += record.get("n_model_calls_recovered", 0)
                counts["n_request_snapshots_recovered"] += sum(
                    "source_event_index" in call
                    for call in trace.get("model_call_evidence", {}).get("calls", [])
                )
                counts["n_raw_trajectories_recovered"] += bool(
                    record.get("raw_trajectory_recovered")
                )
                counts["n_benchmark_task_statements_recovered"] += bool(record.get("task_prompt"))
                limitations.update(trace["evidence_scope"]["limitations"])
                notices.update(trace["evidence_scope"].get("notices", {}))
                agent_refs.add(trace["agent"]["name"])
                if progress:
                    progress(index + 1, len(identities))
        temp.replace(out)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    if activity:
        activity("Verifying prepared dataset…")
    with out.open("rb") as handle:
        output_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    _write_json(root / "manifest.json", {
        "adapter_version": ADAPTER_VERSION, "run_id": run_id,
        "canonical_run_id": canonical_id,
        "base_url": api.base_url, "export_sha256": digest, "output_sha256": output_digest,
        "n_rows": compact.metadata.num_rows,
        "trace_export_sha256": bulk.exports[0]["sha256"] if len(bulk.exports) == 1 else None,
        "trace_exports": bulk.exports, **counts,
        "agent_refs": sorted(agent_refs),
        "limitations": dict(limitations),
        "notices": dict(notices),
        "outcomes": "Unchanged API export labels; benchmark-specific semantics must be checked.",
        "scope": "Exported execution records, full event text/tool arguments, artifact text, "
                 "and recorded model calls retained without overview truncation. "
                 "Per-resource availability describes missing data. Retrieval is not review.",
    })
    source = export_path.parent / "source.json"
    if source.is_file():
        _write_json(root / "source.json", json.loads(source.read_text()))
    return out


def _reuse_raw_downloads(
    destination: Path, parent: Path, base_url: str, run_id: str, digest: str, generation: Any,
) -> None:
    """Reuse raw bytes across selections; normal source reads verify them.

    Raw files are immutable: a replacement download is written to a temporary
    file then renamed. Copy the tiny checksum receipt rather than linking it,
    because the downloader updates that receipt in place.
    """
    for candidate in sorted(parent.glob("source-v*-*")):
        if candidate / "initial-inputs" == destination:
            continue
        try:
            manifest = json.loads((candidate / "manifest.json").read_text())
            source = candidate / "source.json"
            previous_generation = (json.loads(source.read_text()).get("export_generation")
                                   if source.is_file() else None)
            if (manifest.get("base_url", "").rstrip("/") != base_url
                    or manifest.get("canonical_run_id") != run_id
                    or previous_generation != generation
                    or (
                        generation is None
                        and manifest.get("export_sha256") != digest
                    )):
                continue
            for receipt in (candidate / "initial-inputs").glob("*.sha256"):
                raw = receipt.with_suffix(".jsonl")
                target = destination / raw.name
                if not raw.is_file() or target.exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(raw, target)
                    shutil.copyfile(receipt, target.with_suffix(".sha256"))
                except OSError:
                    # Cache reuse is optional; normal retrieval remains the fallback.
                    continue
        except (OSError, ValueError, AttributeError):
            continue


def _reuse_bulk_download(
    destination: Path, parent: Path, base_url: str, run_id: str, digest: str, generation: Any,
    selection: list[tuple[str, int]] | None,
) -> None:
    """Reuse checked raw ZIPs across adapter upgrades, never normalized evidence."""
    if (destination / "trace.zip").is_file():
        return
    for candidate in sorted(parent.glob("source-v*-*")):
        try:
            manifest = json.loads((candidate / "manifest.json").read_text())
            source = candidate / "source.json"
            previous_generation = (json.loads(source.read_text()).get("export_generation")
                                   if source.is_file() else None)
            if (manifest.get("base_url", "").rstrip("/") != base_url
                    or manifest.get("canonical_run_id") != run_id
                    or manifest.get("export_sha256") != digest
                    or previous_generation != generation):
                continue
            downloaded = candidate / "bulk" / "download.json"
            receipt = json.loads(downloaded.read_text())
            expected_selection = ([list(identity) for identity in selection]
                                  if selection is not None else None)
            if receipt.get("selection") != expected_selection:
                continue
            archive = candidate / "bulk" / "trace.zip"
            with archive.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != receipt.get("archive_sha256"):
                continue
            destination.mkdir(parents=True, exist_ok=True)
            partial = destination / "trace.partial"
            shutil.copyfile(archive, partial)
            partial.replace(destination / "trace.zip")
            _write_json(destination / "download.json", receipt)
            return
        except (OSError, ValueError, AttributeError):
            continue
