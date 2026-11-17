# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable provenance sidecars for local catalog libraries."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autofa.data.dataset_identity import normalize_data_identity, publication_data_identity
from autofa.data.reference_data import normalize_reference_data

if TYPE_CHECKING:
    from autofa.config import RunConfig
    from autofa.core.library import FunctionLibrary

CATALOG_PROVENANCE_VERSION = 2
CATALOG_PROVENANCE_SUFFIX = ".provenance.json"


@dataclass(frozen=True)
class CatalogProvenance:
    """Local source run and cloud lineage for one durable catalog snapshot."""

    source_run_id: str
    remote_library_id: str | None = None
    remote_branch_id: str | None = None
    remote_branch_key: str | None = None
    remote_version_id: str | None = None
    remote_base_version_id: str | None = None
    source_harness: str = "unknown"
    reference_data: tuple[dict[str, str], ...] = ()
    data_identity: dict[str, Any] | None = None
    selection_id: str | None = None
    schema_version: int = CATALOG_PROVENANCE_VERSION

    @classmethod
    def from_run(cls, run: RunConfig) -> CatalogProvenance:
        branch_key = (
            _optional_text(getattr(run, "remote_branch_key", None))
            or str(run.run_id)
        )
        return cls(
            source_run_id=str(run.run_id),
            remote_library_id=_optional_text(getattr(run, "remote_library_id", None)),
            remote_branch_id=_optional_text(getattr(run, "remote_branch_id", None)),
            remote_branch_key=branch_key,
            remote_version_id=_optional_text(getattr(run, "remote_version_id", None)),
            remote_base_version_id=_optional_text(
                getattr(run, "remote_base_version_id", None)
            ),
            source_harness=_optional_text(
                getattr(run, "source_harness", None)
            )
            or "unknown",
            reference_data=tuple(
                normalize_reference_data(getattr(run, "reference_data", []))
            ),
            selection_id=_optional_text(getattr(run, "selection_id", None)),
            data_identity=publication_data_identity(
                getattr(run, "parquet_paths", []),
                reference_data=getattr(run, "reference_data", []),
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CatalogProvenance:
        data = dict(value)
        source_run_id = _optional_text(data.get("source_run_id"))
        if not source_run_id:
            raise ValueError("catalog provenance is missing source_run_id")
        schema_version = data.get("schema_version")
        if schema_version != CATALOG_PROVENANCE_VERSION:
            raise ValueError(
                f"unsupported catalog provenance version: {schema_version}"
            )
        return cls(
            source_run_id=source_run_id,
            remote_library_id=_optional_text(data.get("remote_library_id")),
            remote_branch_id=_optional_text(data.get("remote_branch_id")),
            remote_branch_key=_optional_text(data.get("remote_branch_key")),
            remote_version_id=_optional_text(data.get("remote_version_id")),
            remote_base_version_id=_optional_text(data.get("remote_base_version_id")),
            source_harness=_optional_text(data.get("source_harness")) or "unknown",
            reference_data=tuple(normalize_reference_data(data.get("reference_data"))),
            selection_id=_optional_text(data.get("selection_id")),
            data_identity=normalize_data_identity(data.get("data_identity")),
            schema_version=CATALOG_PROVENANCE_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_run_id": self.source_run_id,
            "remote_library_id": self.remote_library_id,
            "remote_branch_id": self.remote_branch_id,
            "remote_branch_key": self.remote_branch_key,
            "remote_version_id": self.remote_version_id,
            "remote_base_version_id": self.remote_base_version_id,
            "source_harness": self.source_harness,
            "reference_data": list(self.reference_data),
            "data_identity": self.data_identity,
            "selection_id": self.selection_id,
        }


def catalog_provenance_path(library_path: Path) -> Path:
    """Return the sidecar path adjacent to a catalog ``library.json``."""
    path = Path(library_path)
    return path.with_name(f"{path.stem}{CATALOG_PROVENANCE_SUFFIX}")


def read_catalog_provenance(library_path: Path) -> CatalogProvenance | None:
    """Read a sidecar, returning ``None`` when it is missing or invalid."""
    path = catalog_provenance_path(library_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            return None
        return CatalogProvenance.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def write_catalog_provenance(
    library_path: Path,
    provenance: CatalogProvenance,
) -> Path:
    """Atomically replace a catalog's provenance sidecar."""
    path = catalog_provenance_path(library_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    try:
        temporary.write_text(
            json.dumps(provenance.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_catalog_library(
    library: FunctionLibrary,
    path: Path,
    run: RunConfig,
) -> tuple[Path, Path]:
    """Save a durable catalog and atomically update its provenance sidecar."""
    destination = Path(path)
    library.save(destination)
    sidecar = write_catalog_provenance(destination, CatalogProvenance.from_run(run))
    return destination, sidecar


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
