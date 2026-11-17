# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grader-side outcome evidence for prepared remote and local rollouts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from autofa.data.artifact_evidence import is_outcome_text_artifact
from autofa.data.remote_api import DataApiClient
from autofa.data.source_errors import SourceDataError


class RemoteOutcomeSource:
    """Point reads from the EvalRDL API for one remote run."""

    def __init__(self, api: DataApiClient, run_id: str):
        self.api, self.run_id = api, run_id

    def artifacts(self, task_id: str, repeat: int) -> list[dict[str, Any]]:
        artifacts = self.api.get_rollout_artifacts(self.run_id, task_id, repeat)
        selected = [artifact for artifact in artifacts if is_outcome_text_artifact(artifact)]
        generic = getattr(self.api, "get_rollout_artifact_texts", None)
        if callable(generic):
            hydrated = iter(
                cast(
                    Iterable[dict[str, Any]],
                    generic(
                        selected,
                        run_id=self.run_id,
                        task_id=task_id,
                        repeat=repeat,
                    ),
                )
            )
        else:
            # A verifier-only source cannot hydrate candidate patches.
            verifiers = [artifact for artifact in selected if artifact.get("role") == "verifier"]
            hydrated_verifiers = iter(
                self.api.get_rollout_verifier_artifacts(
                    verifiers,
                    run_id=self.run_id,
                    task_id=task_id,
                    repeat=repeat,
                )
            )
            hydrated = iter(
                [
                    next(hydrated_verifiers)
                    if artifact.get("role") == "verifier"
                    else artifact
                    for artifact in selected
                ]
            )
        return [
            next(hydrated) if is_outcome_text_artifact(artifact) else artifact
            for artifact in artifacts
        ]

    def read_raw_record(self, run_id: str, source_ref: str) -> dict[str, Any]:
        return self.api.read_raw_record(run_id or self.run_id, source_ref)


class LocalOutcomeSource:
    """Read outcome evidence from an imported EvalRDL store."""

    def __init__(self, store: Path):
        self.store = Path(store)
        manifest_path = self.store / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceDataError(
                f"cannot read local source manifest {manifest_path}: {exc}"
            ) from exc
        source_path = manifest.get("source_path") if isinstance(manifest, Mapping) else None
        if not isinstance(source_path, str) or not source_path.strip():
            raise SourceDataError(f"local source manifest {manifest_path} is missing source_path")
        source = Path(source_path)
        self.root = source.parent if source.suffix == ".jsonl" else source
        self._artifact_index: dict[tuple[str, str], list[dict[str, Any]]] | None = None

    def artifacts(self, task_id: str, repeat: int) -> list[dict[str, Any]]:
        artifacts = self._artifact_rows(task_id, repeat)
        texts = (
            {row["artifact_idx"]: row.get("text") for row in self._text_rows(task_id, repeat)}
            if any(is_outcome_text_artifact(artifact) for artifact in artifacts)
            else {}
        )
        for artifact in artifacts:
            if not is_outcome_text_artifact(artifact):
                continue
            text = texts.get(artifact.get("artifact_idx"))
            if isinstance(text, str):
                artifact["text"] = text
            else:
                artifact["text_error"] = "artifact text was not imported"
        return artifacts

    def read_raw_record(self, run_id: str, source_ref: str) -> dict[str, Any]:
        path, _, offset = source_ref.rpartition(":")
        file = (self.root / path).resolve()
        if not file.is_relative_to(self.root.resolve()):
            raise SourceDataError(f"raw record ref escapes the run folder: {source_ref!r}")
        try:
            with file.open("rb") as handle:
                handle.seek(int(offset))
                record = json.loads(handle.readline())
        except (OSError, ValueError) as exc:
            raise SourceDataError(f"raw record {source_ref} is unavailable: {exc}") from exc
        if not isinstance(record, dict):
            raise SourceDataError(f"raw record at {source_ref} is not an object")
        return record

    def _artifact_rows(self, task_id: str, repeat: int) -> list[dict[str, Any]]:
        if self._artifact_index is None:
            index: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in self._read("artifacts"):
                index[(str(row.get("task_id")), str(row.get("repeat")))].append(row)
            self._artifact_index = dict(index)
        return [dict(row) for row in self._artifact_index.get((str(task_id), str(repeat)), ())]

    def _text_rows(self, task_id: str, repeat: int) -> list[dict[str, Any]]:
        rows = self._read("artifact_texts", filters=[("task_id", "=", task_id)])
        return [row for row in rows if str(row.get("repeat")) == str(repeat)]

    def _read(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq

        path = self.store / "tables" / f"{table}.parquet"
        return pq.read_table(path, **kwargs).to_pylist() if path.is_file() else []
