# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable names for dataset mixtures, independent of local cache locations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from autofa.data.reference_data import (
    canonical_export_url,
    data_api_location,
    local_viewer_base,
    normalize_reference_data,
)


def dataset_mixture_id(
    parquet_paths: list[str],
    *,
    model_id: str,
    benchmarks: list[str],
    reference_data: list[dict[str, Any]],
) -> str:
    """Name a source set plus filters; retain model names for single-source runs.

    Remote sources use their Data API origin and run ID. Local sources use a
    full content digest. Selection order, duplicate references, timestamps,
    cache paths, and trace-adapter versions do not affect a remote mixture ID.
    The caller persists the result so local files need hashing only once.
    """
    paths = list(dict.fromkeys(Path(p).expanduser().resolve() for p in parquet_paths))
    references = normalize_reference_data(reference_data)
    all_models = model_id in {"*", "ALL", ""}
    if len(paths) <= 1 and len(references) <= 1 and not all_models:
        return ""

    def source_identity(reference):
        if local_viewer_base(reference):
            return f"local-evalrdl:{reference['store_id']}:{reference['run_id']}"
        location = data_api_location(reference)
        return canonical_export_url(*location) if location is not None else None

    by_path = {
        reference["parquet_path"]: identity
        for reference in references
        if reference.get("parquet_path")
        and (identity := source_identity(reference)) is not None
    }
    reference_identities = [
        identity for reference in references
        if (identity := source_identity(reference)) is not None
    ]
    # Public provenance intentionally omits machine-local paths. A complete
    # remote binding still identifies the source set when caches are absent.
    if not by_path and len(reference_identities) == len(paths) and reference_identities:
        paths = []
    sources: set[str] = set()
    for path in paths:
        identity = by_path.get(str(path))
        location = _remote_sidecar_location(path) if identity is None else None
        if identity is not None:
            sources.add(identity)
        elif location is not None:
            sources.add(canonical_export_url(*location))
        else:
            with path.open("rb") as handle:
                sources.add("sha256:" + hashlib.file_digest(handle, "sha256").hexdigest())
    if not paths:
        sources.update(reference_identities)
    if not sources:
        return ""
    payload = {
        "version": 1,
        "sources": sorted(sources),
        "model_filter": "*" if all_models else model_id,
        "benchmark_filter": sorted(set(benchmarks)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "mixture-" + hashlib.sha256(encoded).hexdigest()[:16]


def _remote_sidecar_location(path: Path) -> tuple[str, str] | None:
    try:
        data = json.loads((path.parent / "source.json").read_text(encoding="utf-8"))
        url = canonical_export_url(data["base_url"], data["run_id"])
        return data_api_location({"run_id": data["run_id"], "export_url": url})
    except (OSError, ValueError, KeyError, TypeError):
        return None


_DATA_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DATA_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


def _data_origin(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Data identity origin must be an HTTP(S) origin or API base path")
    try:
        value = value.strip()
        if any(character.isspace() for character in value):
            raise ValueError("Invalid data origin")
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Invalid data origin")
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        default_port = 443 if parsed.scheme == "https" else 80
        if parsed.port not in {None, default_port}:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    except ValueError:
        raise ValueError(
            "Data identity origin must be an HTTP(S) origin or API base path"
        ) from None


def normalize_data_identity(value: Any) -> dict[str, Any] | None:
    """Validate and canonicalize the privacy-safe source identity wire format.

    A missing identity means unknown; it must never imply model/benchmark equality.
    Local paths, source harnesses, sampling limits and discovery state do not belong
    in this identity. The server derives the grouping key from these source records.
    """
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"version", "sources"}
            or type(value["version"]) is not int or value["version"] != 1
            or not isinstance(value["sources"], list) or not 1 <= len(value["sources"]) <= 100):
        raise ValueError("Data identity requires version 1 and between 1 and 100 sources")
    sources = {}
    for raw in value["sources"]:
        if not isinstance(raw, dict):
            raise ValueError("Invalid data identity source")
        if raw.get("kind") == "evalrdl" and set(raw) == {"kind", "origin", "run_id"}:
            run_id = raw["run_id"]
            if not isinstance(run_id, str) or not _DATA_RUN_ID.fullmatch(run_id.strip()):
                raise ValueError("Invalid EvalRDL data run ID")
            source = {"kind": "evalrdl", "origin": _data_origin(raw["origin"]),
                      "run_id": run_id.strip()}
        elif raw.get("kind") == "local" and set(raw) == {"kind", "sha256"}:
            digest = raw["sha256"]
            if not isinstance(digest, str) or not _DATA_DIGEST.fullmatch(digest):
                raise ValueError("Local data identity requires a lowercase SHA256 digest")
            source = {"kind": "local", "sha256": digest}
        else:
            raise ValueError("Invalid data identity source fields")
        sources[json.dumps(source, sort_keys=True, separators=(",", ":"))] = source
    return {"version": 1, "sources": [sources[key] for key in sorted(sources)]}


@lru_cache(maxsize=128)
def _publication_file_digest(path: str, signature: tuple[int, ...]) -> str:
    # Catalog saves are frequent. File identity/timestamps only invalidate this
    # process-local cache; grouping always uses the actual complete content hash.
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file_signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def publication_data_identity(
    parquet_paths: Iterable[str | Path], *, reference_data: Any,
) -> dict[str, Any] | None:
    """Identify all selected inputs, independent of analysis and cache location.

    EvalRDL sources use validated references or export sidecars without trajectory
    reads. Inputs without remote provenance and managed local inputs use full
    content SHA256. An incomplete binding returns None, never an identity for
    only the available subset. Callers send that explicit null so a server cannot
    mistake partial references for a complete legacy identity. No source paths
    are included in the result.
    """
    try:
        paths = set(Path(path).expanduser().resolve() for path in parquet_paths)
        raw_references = reference_data if isinstance(reference_data, (list, tuple)) else []
        # Preserve each path binding: multiple caches may name the same source.
        references = [ref for raw in raw_references for ref in normalize_reference_data([raw])]
        by_path: dict[Path, list[dict[str, str]]] = {}
        for reference in references:
            if reference.get("parquet_path"):
                by_path.setdefault(Path(reference["parquet_path"]), []).append(reference)
            elif paths:
                # Do not assign a pathless reference to an arbitrary selected file.
                return None
        if not paths:
            sources = []
            for raw in raw_references:
                normalized = normalize_reference_data([raw])
                location = data_api_location(normalized[0]) if normalized else None
                if location is None:
                    return None
                origin, run_id = location
                sources.append({"kind": "evalrdl", "origin": origin, "run_id": run_id})
        else:
            sources = []
            for path in paths:
                remotes = []
                for reference in by_path.get(path, []):
                    location = data_api_location(reference)
                    if location is not None:
                        origin, run_id = location
                        remotes.append({"kind": "evalrdl", "origin": origin, "run_id": run_id})
                if path not in by_path:
                    # Cached exports can retain source.json without a bound reference.
                    # A sidecar must never override an explicit remote or local binding.
                    location = _remote_sidecar_location(path)
                    if location is not None:
                        origin, run_id = location
                        remotes.append({"kind": "evalrdl", "origin": origin, "run_id": run_id})
                if remotes:
                    remote = normalize_data_identity({"version": 1, "sources": remotes})
                    if len(remote["sources"]) != 1:
                        return None  # Contradictory source ownership is not a data identity.
                    sources.extend(remote["sources"])
                else:
                    before = _file_signature(path)
                    digest = _publication_file_digest(str(path), before)
                    if _file_signature(path) != before:
                        return None  # The dataset changed while it was being hashed.
                    sources.append({"kind": "local", "sha256": digest})
        return normalize_data_identity({"version": 1, "sources": sources}) if sources else None
    except (OSError, TypeError, ValueError):
        return None
