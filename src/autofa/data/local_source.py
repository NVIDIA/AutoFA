# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply original-source recovery to locally selected compact API exports."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path

import pyarrow.parquet as pq

from autofa.config import load_config
from autofa.data.gym_adapter import CANONICAL_COLUMNS, converted_root
from autofa.data.reference_data import canonical_export_url
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.source_export import ADAPTER_VERSION, materialize_source_traces

_EXPORT_NAME = re.compile(
    r"(?P<run>[0-9a-f]{16,64})(?:_(?P<task>.+)_(?P<repeat>\d+))?_autofa\.parquet"
)


def materialize_local_parquet(path: Path) -> Path:
    """Recover known compact exports; never infer provenance from row position.

    Standard download filenames identify a run (and optionally task/repeat).
    Renamed exports need their source sidecar. Generic local Parquets and
    already-prepared current source caches remain local.
    """
    path = Path(path).expanduser().resolve()
    from autofa.data.local_import import reference_for_import
    from autofa.data.local_trace_store import resolve_store

    if reference := reference_for_import(path):
        return resolve_store(reference["store_id"]) / "dataset.parquet"
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if "adapter_version" in manifest:
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            if (manifest["adapter_version"] == ADAPTER_VERSION
                    and manifest.get("output_sha256") == checksum):
                return path
            compact = path.parent.parent / "dataset.parquet"
            if not compact.is_file():
                raise DataApiError("Prepared source cache is stale; select its original API export")
            return materialize_local_parquet(compact)

    match = _EXPORT_NAME.fullmatch(path.name)
    provenance = {}
    for sidecar in (path.with_suffix(".source.json"), path.parent / "source.json"):
        if sidecar.is_file():
            data = json.loads(sidecar.read_text())
            if isinstance(data, dict) and data.get("run_id") and data.get("base_url"):
                provenance = data
                break
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    known_export = bool(match or provenance)
    if not known_export:
        # These are exactly the compact exporter columns. Structured Gym traces
        # have the same columns, but do not need remote recovery.
        compact_columns = set(CANONICAL_COLUMNS) - {"input_json"}
        if schema not in (compact_columns, compact_columns | {"system_text", "lane"}):
            return path
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["response_text"]):
            for value in batch.column(0).to_pylist():
                try:
                    trace = json.loads(value)
                except (ValueError, TypeError):
                    trace = None
                if not isinstance(trace, dict) or not isinstance(trace.get("steps"), list):
                    known_export = True
                    break
            if known_export:
                break
        if not known_export:
            return path
    cfg = load_config()
    if not cfg.remote_full_traces:
        warnings.warn(
            f"Summary-only local export: {path.name}; original traces were not recovered.",
            stacklevel=2,
        )
        return path
    run_id = str(provenance.get("run_id") or (match["run"] if match else ""))
    if not run_id:
        raise DataApiError(
            "Local Parquet looks like a compact API export but has no source run identity. "
            "Select the remote run, restore its original *_autofa.parquet filename, or add "
            "a .source.json sidecar with run_id and base_url. "
            "remote_full_traces=false explicitly selects summary-only analysis."
        )
    base_url = str(provenance.get("base_url") or cfg.data_api_base_url).rstrip("/")
    canonical_export_url(base_url, run_id)  # Validate before constructing API requests.
    if base_url != cfg.data_api_base_url.rstrip("/"):
        raise DataApiError(
            "Local source uses a different Data API; set data_api_base_url explicitly"
        )
    if match and provenance and match["run"] != run_id:
        raise DataApiError("Local export filename and source sidecar disagree on run identity")
    if match and match["task"] is not None:
        identities = pq.read_table(path, columns=["problem_id", "run_idx"]).to_pylist()
        if identities != [{"problem_id": match["task"], "run_idx": int(match["repeat"])}]:
            raise DataApiError("Local rollout export filename does not match its task/repeat")
    cache_root = converted_root()
    if provenance and path.is_relative_to(cache_root.resolve()):
        # Reuse the existing remote event/raw-file caches when a user browses
        # into converted/; do not redownload multi-GB source JSONL files.
        return materialize_source_traces(path, api=DataApiClient(base_url), run_id=run_id)
    # Keep downloads and source caches out of the user's Downloads/input folder.
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    key = hashlib.sha256(json.dumps([base_url, run_id, digest]).encode()).hexdigest()[:20]
    root = cache_root / "local-remote" / key
    root.mkdir(parents=True, exist_ok=True)
    compact = root / "dataset.parquet"
    temp = compact.with_suffix(".partial")
    shutil.copyfile(path, temp)
    temp.replace(compact)
    (root / "source.json").write_text(json.dumps({
        "run_id": run_id, "base_url": base_url,
        "export_path": canonical_export_url(base_url, run_id)[len(base_url):],
        "original_local_path": str(path),
    }) + "\n")
    return materialize_source_traces(compact, api=DataApiClient(base_url), run_id=run_id)
