# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import harness logs through the vendored readers, schema, importer and exporter."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from autofa.data.artifact_evidence import is_reference_artifact
from autofa.data.execution_policy import is_evaluation_record, is_evaluation_table
from autofa.data.local_trace_store import (
    STORE_VERSION,
    file_digest,
    resolve_store,
    source_store_root,
)
from autofa.data.source_errors import SourceDataError
from autofa.data.source_normalization import _validate_identities, normalize_events
from autofa.data.source_request_references import attach_event_requests

# Version 6 snapshots trusted inline/source-field artifacts into immutable local stores.
IMPORT_VERSION = 6
LIBRARY_REVISION = "f88df243c3d19a01b01305df0c3a512f71cb91d9"
_METADATA = {"eval_metadata.json", "metadata.yaml", "run_config.yml", "results.yml"}


@dataclass(frozen=True)
class LocalImport:
    parquet_path: Path
    store_id: str
    run_id: str
    source_harness: str
    dataset_kind: str = "rich"

    def reference_data(self) -> dict[str, str]:
        return {
            "kind": "local_evalrdl", "store_id": self.store_id, "run_id": self.run_id,
            "source_harness": self.source_harness, "parquet_path": str(self.parquet_path),
        }


def reader_dispatch():
    from evalrdl_readers.readers import RUNDIR

    return RUNDIR


def _files(path: Path) -> tuple[Path, list[Path]]:
    if path.is_file() and path.suffix == ".jsonl":
        root = path.parent
        # A selected stream includes its own sidecars, never another rollout stream.
        files = {path}
        files.update(p for p in root.iterdir() if p.is_file() and (
            p.name in _METADATA or p.name.startswith(path.stem + "_")
        ))
    elif path.is_dir():
        root = path
        files = {p for p in root.rglob("*") if p.is_file()}
    else:
        raise ValueError(f"Expected a run folder or JSONL file: {path}")
    for file in files:
        target = file.resolve()
        # NeMo Gym can link model-call captures from a shared store outside the run
        # folder. Such a link is followed only when both it and its target are named as
        # captures: the target's name is not chosen by the run folder's author, and the
        # link's name makes the reader treat the bytes as a capture, not as run records.
        if not target.is_relative_to(root) and not all(
            name.endswith(".capture.jsonl") for name in (file.name, target.name)
        ):
            raise ValueError(f"Source file escapes the selected run folder: {file.name}")
    return root, sorted(files, key=lambda p: p.relative_to(root).as_posix())


class _InputTree:
    """Give the reader exactly the selected files without copying large input logs."""

    def __init__(self, path: Path):
        from evalrdl_readers.core.tree import Entry

        self.root, files = _files(path)
        self.name = path.name
        self._files = {p.relative_to(self.root).as_posix(): p for p in files}
        self.linked = {name for name, p in self._files.items()
                       if not p.resolve().is_relative_to(self.root)}
        self._entries = [Entry(name, p.stat().st_size) for name, p in self._files.items()]

    def paths(self):
        return self._entries

    def read(self, path: str, limit: int | None = None) -> bytes:
        with self.open(path) as handle:
            return handle.read() if limit is None else handle.read(limit)

    def open(self, path: str):
        return self._files[path].open("rb", buffering=1024 * 1024)

    def fingerprint(self) -> str:
        files = [(name, file_digest(path)) for name, path in self._files.items()]
        return _hash(files)


def _refuse_linked_records(result, linked: set[str]) -> None:
    """Refuse a read that kept records from a capture linked from outside the run folder.

    The Gym reader joins a linked capture's request parameters to the run's own calls,
    and those rows name the run's records. A row naming the linked file means another
    input, such as Harbor's artifact listing, kept that file's contents.
    """
    for bucket in (result.rows, result.losers):
        for rows in bucket.values():
            for row in rows:
                if row.at_path in linked:
                    raise ValueError(
                        f"Linked capture outside the selected run folder was read as run "
                        f"data: {row.at_path}"
                    )


def detect_reader(path: Path | str) -> str | None:
    """Identify a declared harness before considering generic Parquet/JSONL inputs."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir() and not (path.is_file() and path.suffix == ".jsonl"):
        return None
    reader = reader_dispatch().pick(_InputTree(path))
    return reader.name if reader is not None else None


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _import_result(store: Path) -> LocalImport:
    try:
        store = resolve_store(store.name, store.parent)
    except ValueError as exc:
        raise ValueError(
            f"Cannot reuse local snapshot {store.name}: {exc}. Restore this snapshot "
            "from a backup or re-import in a different AUTOFA_HOME profile. "
            "force=True verifies source logs; it never replaces an existing snapshot."
        ) from exc
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    if "dataset_kind" not in manifest:
        raise SourceDataError(
            "Local snapshot predates the current store format; re-import the run."
        )
    result = LocalImport(
        store / "dataset.parquet",
        manifest["store_id"],
        manifest["run_id"],
        manifest["source_harness"],
        manifest["dataset_kind"],
    )
    _ensure_source_reference(result)
    return result


def _ensure_source_reference(result: LocalImport) -> None:
    """Restore a missing adjunct atomically; never overwrite an existing reference."""
    sidecar = result.parquet_path.parent / "source.json"
    if sidecar.is_symlink():
        raise ValueError("Invalid local trace reference: source.json is a symlink")
    try:
        text = sidecar.read_text(encoding="utf-8")
    except FileNotFoundError:
        fd, name = tempfile.mkstemp(prefix=".source-", suffix=".json", dir=sidecar.parent)
        pending = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(result.reference_data(), handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            try:
                os.link(pending, sidecar)
            except FileExistsError:
                pass  # A concurrent reader installed it; validate that reference below.
        finally:
            pending.unlink(missing_ok=True)
        if sidecar.is_symlink():
            raise ValueError("Invalid local trace reference: source.json is a symlink") from None
        text = sidecar.read_text(encoding="utf-8")
    try:
        reference = json.loads(text)
    except ValueError as exc:
        raise ValueError("Invalid local trace reference in source.json") from exc
    expected = result.reference_data()
    # Paths can refer to an old profile location; manifest identity remains stable.
    if not isinstance(reference, dict) or any(
        reference.get(key) != expected[key]
        for key in ("kind", "store_id", "run_id", "source_harness")
    ):
        raise ValueError("Local trace reference and manifest disagree on source identity")


def reference_for_import(path: Path) -> dict[str, str] | None:
    """Return managed local provenance for a materialized dataset, if present."""
    selected = Path(path).expanduser().resolve()
    if (selected.name == "dataset.parquet"
            and selected.parent.parent == source_store_root().expanduser().resolve()):
        # The validated manifest owns this identity even if its adjunct was removed.
        return _import_result(selected.parent).reference_data()
    try:
        ref = json.loads((Path(path).parent / "source.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(ref, dict) and ref.get("kind") == "local_evalrdl":
        store = resolve_store(ref.get("store_id"))
        if selected != store / "dataset.parquet":
            raise ValueError("Local trace reference does not belong to this managed dataset")
        result = _import_result(store)
        if ref.get("run_id") != result.run_id:
            raise ValueError("Local trace reference and manifest disagree on run identity")
        return result.reference_data()
    return None


def import_local_run(
    path: Path | str, *, model_id: str | None = None, benchmark: str | None = None,
    force: bool = False, store_root: Path | None = None,
    efficiency_only: bool = False,
) -> LocalImport:
    """Materialize a local run; force verifies conversion without replacing snapshots."""
    from evalrdl_export import AUTOFA
    from evalrdl_readers.convert import DuplicateRollout, convert, render_loss, write_tables
    from evalrdl_schema import CURRENT
    from evalrdl_schema.tables import RunTables

    path = Path(path).expanduser().resolve()
    tree = _InputTree(path)
    fingerprint = tree.fingerprint()
    identity = [
        IMPORT_VERSION,
        LIBRARY_REVISION,
        CURRENT.version,
        fingerprint,
        model_id,
        benchmark,
    ]
    if efficiency_only:
        identity.append("efficiency")
    store_id = _hash(identity)
    if store_root is not None and not force:
        cached = source_store_root() / store_id
        if cached.exists():
            return _import_result(cached)
    root = (
        Path(store_root).expanduser().resolve()
        if store_root is not None
        else source_store_root()
    )
    root.mkdir(parents=True, exist_ok=True)
    store = root / store_id
    if store.exists() and not force:
        return _import_result(store)
    dispatch = reader_dispatch()
    result = dispatch.read(tree)
    _refuse_linked_records(result, tree.linked)
    if not result.claimed:
        raise SourceDataError(f"No declared reader recognizes {path}")
    # Source run identity follows bytes, independently of adapter version and cache location.
    try:
        conversion = convert(result, path=str(tree.root), label=path.name, readers=dispatch.readers,
                             run_id=fingerprint, consume=True,
                             harness_of={reader.name: reader.name for reader in dispatch.readers})
    except DuplicateRollout as exc:
        raise SourceDataError(f"Local run has ambiguous task/repeat identities: {exc}") from exc
    rollouts = conversion.tables.get("rollouts", [])
    quarantined = [r for r in rollouts if r.get("ingest_status") == "quarantined"]
    diagnostics = {
        "reader": result.reader, "reader_ok": result.ok,
        "findings": [finding.line() for finding in result.findings],
        "conversion": conversion.record, "loss": render_loss(conversion.loss),
        "loss_counts": {
            "rows_dropped": conversion.loss.rows_dropped_total,
            "cells_dropped": sum(conversion.loss.cells_dropped.values()),
        },
        "quarantined_rollouts": [
            {key: row.get(key) for key in ("task_id", "repeat", "ingest_violation")}
            for row in quarantined
        ],
    }
    if not conversion.admitted or not rollouts or quarantined:
        quarantine = root / "quarantine"
        quarantine.mkdir(exist_ok=True)
        record = quarantine / f"{store_id}.json"
        _json(record, diagnostics)
        raise SourceDataError(
            f"Local import has {len(quarantined)} quarantined rollouts or no usable rows; "
            f"no analysis started. Diagnostics: {record}"
        )
    pairs = [(r["task_id"], r["repeat"]) for r in rollouts]
    if len(set(pairs)) != len(pairs):
        raise SourceDataError("Local run has ambiguous task/repeat identities across agents")
    run = conversion.tables["runs"][0]
    _metadata_fallback(run, tree.root, model_id=model_id, benchmark=benchmark)
    staged = Path(tempfile.mkdtemp(prefix=f".{store_id}-", dir=root))
    try:
        tables = staged / "tables"
        tables.mkdir()
        artifact_files = _capture_artifacts(conversion, tree, staged)
        write_tables(conversion, tables)
        grader_calls = _grader_index(rollouts, tree._files)
        # Text-heavy reader results are released before exporter/trace hydration begins.
        del conversion, rollouts
        compact = staged / ("dataset.parquet" if efficiency_only else "compact.parquet")
        AUTOFA.to_file(RunTables(tables, AUTOFA.reads), run, None, compact)
        if efficiency_only:
            counts = {"n_rows": pq.ParquetFile(compact).metadata.num_rows}
        else:
            counts = _write_rich_dataset(
                tables,
                compact,
                staged / "dataset.parquet",
                store_id,
                grader_calls=grader_calls,
            )
            compact.unlink()
        if _InputTree(path).fingerprint() != fingerprint:
            raise SourceDataError(
                "Source logs changed during import; retry after writing completes"
            )
        imported = LocalImport(store / "dataset.parquet", store_id, run["run_id"], result.reader)
        _json(staged / "source.json", imported.reference_data())
        _json(staged / "diagnostics.json", diagnostics)
        _json(staged / "manifest.json", {
            "version": STORE_VERSION, "status": "complete", "store_id": store_id,
            "run_id": run["run_id"], "source_harness": result.reader,
            "reader": result.reader, "schema_version": CURRENT.version,
            "adapter_version": IMPORT_VERSION, "library_revision": LIBRARY_REVISION,
            "dataset_kind": "compact" if efficiency_only else "rich",
            "grader_call_ids_by_record": grader_calls,
            "input_fingerprint": fingerprint, "source_path": str(path),
            "tables": {p.name: file_digest(p) for p in sorted(tables.iterdir())},
            "dataset_sha256": file_digest(staged / "dataset.parquet"),
            "diagnostics": diagnostics, "artifact_files": artifact_files, **counts,
        })
        if store.exists():
            # Existing reports hold immutable stores. A forced read validates the new
            # conversion, then keeps the completed original instead of replacing it.
            _import_result(store)
        else:
            try:
                os.rename(staged, store)
            except OSError:
                if not store.exists():
                    raise
                _import_result(store)  # Another process completed this same import.
        if diagnostics["findings"] or any(diagnostics["loss_counts"].values()):
            warnings.warn(
                f"Local import retained {len(diagnostics['findings'])} reader findings; "
                f"{diagnostics['loss_counts']['cells_dropped']} reader values were not written; "
                f"see {store / 'diagnostics.json'}", stacklevel=2,
            )
        return _import_result(store)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def _metadata_fallback(run, root, *, model_id, benchmark):
    try:
        meta = json.loads((root / "eval_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    # The schema's model is the tested model, never the agent's router reference.
    # Keep the harness's benchmark spelling; explicit consumer labels live here.
    run["model"] = (model_id or run.get("model") or meta.get("model_id") or meta.get("model")
                    or "unknown")
    run["benchmark"] = (benchmark or run.get("benchmark") or meta.get("benchmark")
                        or meta.get("dataset") or "unknown")
    # Export fields are strings even where a producer wrote a nested metadata object.
    run["model"], run["benchmark"] = str(run["model"]), str(run["benchmark"])


def _capturable_artifact_keys(conversion, keys: tuple[str, ...]) -> set[tuple[Any, ...]]:
    """Apply the shared evaluation-parent policy before any artifact bytes are persisted."""
    from autofa.data.source_execution import execution_record
    from evalrdl_schema import CURRENT

    rollout_keys = ("run_id", "task_id", "repeat")
    grouped: dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]] = {}
    for table_name in ("parts", "events", "tool_calls", "artifacts"):
        by_rollout: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in conversion.tables.get(table_name, []):
            identity = tuple(row.get(key) for key in rollout_keys)
            by_rollout.setdefault(identity, []).append(row)
        grouped[table_name] = by_rollout
    identities = set(grouped["artifacts"])
    event_keys = CURRENT.table("events").key
    allowed: set[tuple[Any, ...]] = set()
    for identity in identities:
        parts_by_event: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for part in grouped["parts"].get(identity, []):
            parts_by_event.setdefault(
                tuple(part.get(key) for key in event_keys), []
            ).append(part)
        events = [
            {
                **event,
                "parts": parts_by_event.get(
                    tuple(event.get(key) for key in event_keys), []
                ),
            }
            for event in grouped["events"].get(identity, [])
        ]
        filtered = execution_record(
            dict(zip(rollout_keys, identity, strict=True)),
            [],
            [],
            execution={
                "rollout": dict(zip(rollout_keys, identity, strict=True)),
                "events": events,
                "tool_calls": grouped["tool_calls"].get(identity, []),
                "artifacts": grouped["artifacts"][identity],
                "availability": {},
            },
        )
        allowed.update(
            tuple(row.get(key) for key in keys) for row in filtered["artifacts"]
        )
    return allowed


def _inline_artifact_bytes(row: dict[str, Any], text: str) -> bytes | None:
    """Decode only the exact inline encodings emitted by the bundled readers."""
    origin = row.get("origin")
    if origin == "source_field":
        return text.encode("utf-8")
    if origin != "inline":
        return None
    head, separator, payload = text.partition(",")
    if (
        not separator
        or not head.startswith("data:")
        or "base64" not in {part.lower() for part in head.split(";")[1:]}
    ):
        raise SourceDataError("Inline artifact is not a base64 data URL")
    declared_mime = head[len("data:"):].split(";", 1)[0] or None
    if row.get("mime") is not None and row["mime"] != declared_mime:
        raise SourceDataError("Inline artifact media type differs from reader metadata")
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        raise SourceDataError("Inline artifact has invalid base64 data") from exc


def _capture_artifacts(conversion, tree, staged):
    """Capture verified external and reader-retained inline artifact bytes."""
    from evalrdl_schema import CURRENT

    keys = CURRENT.table("artifacts").key
    texts = {tuple(r[k] for k in keys): r for r in conversion.tables["artifact_texts"]}
    allowed = _capturable_artifact_keys(conversion, keys)
    captured = []
    for row in conversion.tables["artifacts"]:
        key = tuple(row[k] for k in keys)
        if is_evaluation_record(row) or is_reference_artifact(row) or key not in allowed:
            continue
        uri = row.get("uri")
        original = None
        data = None
        if isinstance(uri, str) and uri.startswith("file://"):
            relative = uri[len("file://"):]
            original = tree._files.get(relative)
            if original is None:
                continue
        elif row.get("storage") == "inline":
            text = texts.get(key, {}).get("text")
            if not isinstance(text, str):
                raise SourceDataError("Inline artifact text is absent from reader tables")
            data = _inline_artifact_bytes(row, text)
            if data is None:
                continue
        else:
            continue
        digest = hashlib.sha256(data).hexdigest() if data is not None else file_digest(original)
        size = len(data) if data is not None else original.stat().st_size
        if row.get("sha256") and row["sha256"] != digest:
            raise SourceDataError("Local artifact checksum differs from reader metadata")
        if row.get("size_bytes") is not None and row["size_bytes"] != size:
            raise SourceDataError("Local artifact size differs from reader metadata")
        destination = staged / "artifacts" / digest
        destination.parent.mkdir(exist_ok=True)
        if data is not None:
            destination.write_bytes(data)
        else:
            shutil.copyfile(original, destination)
        if file_digest(destination) != digest:
            raise SourceDataError("Local artifact changed while copying")
        captured.append({
            **{k: row[k] for k in ("task_id", "repeat", "agent_ref", "artifact_idx")},
            "name": row.get("name"),
            "path": f"artifacts/{digest}",
            "sha256": digest,
            "size_bytes": size,
            "media_type": row.get("mime"),
        })
        # Readers keep a preview of external text files. The selected original bytes
        # are available here, so preserve their complete text in the analysis tables.
        if original is not None and key in texts:
            try:
                full_text = destination.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            texts[key]["text"] = full_text
            if "chars" in row:
                row["chars"] = len(full_text)
    return captured


def _write_rich_dataset(
    tables: Path, compact: Path, output: Path, store_id: str, *,
    grader_calls: dict[str, list[str]],
) -> dict:
    from autofa.data.local_table_batches import rollout_batches
    from autofa.data.source_model_calls import count_excluded_calls, drop_grader_calls
    from evalrdl_schema import CURRENT
    from evalrdl_schema.tables import RunTables

    source = RunTables(tables, {t.name: tuple(t.column_names) for t in CURRENT.tables})
    parquet = pq.ParquetFile(compact)
    schema = parquet.schema_arrow.with_metadata({
        **(parquet.schema_arrow.metadata or {}),
        b"autofa.local_trace_revision": str(IMPORT_VERSION).encode(),
        b"autofa.local_store_id": store_id.encode(),
        b"autofa.compact_sha256": file_digest(compact).encode(),
    })
    for column in ("prompt_text", "response_text", "reasoning_text", "input_json"):
        index = schema.get_field_index(column)
        field = pa.field(column, pa.string())
        schema = schema.set(index, field) if index >= 0 else schema.append(field)
    counts: Counter = Counter()
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for rows, batch_source in rollout_batches(source, parquet):
            for row in rows:
                raw = _read_rollout(batch_source, row["problem_id"], row["run_idx"], row["lane"])
                _validate_identities(raw["metadata"], raw["events"], raw["calls"])
                trace = normalize_events(**raw, conversation_lane=row["lane"])
                if graders := grader_calls.get(raw["metadata"].get("raw_record_ref"), []):
                    count_excluded_calls(trace, drop_grader_calls(trace, graders))
                trace["evidence_scope"].update({
                    "identity": "evalrdl_tables_v1",
                    "source": "local EvalRDL event parts and tool arguments",
                })
                trace["execution"]["source"] = {
                    "kind": "local_evalrdl", "store_id": store_id,
                    "run_id": raw["metadata"]["run_id"],
                }
                initial = trace.get("initial_request")
                if not initial:
                    leading = _leading_request(raw["events"])
                    # An assistant greeting can precede the first user turn.
                    # Its system-only prefix is not the task input, whereas the
                    # exporter still records the first user message correctly.
                    if leading and (not row.get("prompt_text") or any(
                        message["role"] == "user" for message in leading["messages"]
                    )):
                        initial = leading
                if initial:
                    trace["initial_request"] = initial
                    row["input_json"] = json.dumps(initial, ensure_ascii=False)
                else:
                    from autofa.data.input_view import compact_export_input_json

                    row["input_json"] = compact_export_input_json(row)
                if initial:
                    from autofa.data.structured import input_view

                    row["prompt_text"] = input_view(initial)["text"] or row["prompt_text"]
                row["response_text"] = json.dumps(trace, ensure_ascii=False)
                row["reasoning_text"] = ""
                counts["n_rows"] += 1
                counts["n_events"] += len(raw["execution"]["events"])
                counts["n_tool_calls"] += len(raw["execution"]["tool_calls"])
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return dict(counts)


def materialize_local_traces(store: Path, compact: Path, output: Path) -> Path:
    """Build rich traces for one selected compact dataset."""
    store = resolve_store(store.name, store.parent)
    compact = compact.expanduser().resolve()
    output = output.expanduser().resolve()

    def identities(path: Path) -> list[tuple[str, int]]:
        rows = pq.read_table(path, columns=["problem_id", "run_idx"]).to_pylist()
        return [(str(row["problem_id"]), int(row["run_idx"])) for row in rows]

    selected = identities(compact)
    try:
        metadata = pq.read_schema(output).metadata or {}
        if (metadata.get(b"autofa.local_trace_revision") == str(IMPORT_VERSION).encode()
                and metadata.get(b"autofa.local_store_id") == store.name.encode()
                and metadata.get(b"autofa.compact_sha256") == file_digest(compact).encode()
                and identities(output) == selected):
            return output
    except (OSError, ValueError):
        pass

    output.parent.mkdir(parents=True, exist_ok=True)
    grader_calls = _saved_grader_index(store)
    temporary = output.with_name(f".{output.name}.{os.urandom(8).hex()}.partial")
    try:
        _write_rich_dataset(
            store / "tables",
            compact,
            temporary,
            store.name,
            grader_calls=grader_calls,
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _grader_index(rollouts: list[dict], files: dict[str, Path]) -> dict[str, list[str]]:
    """Save attribution while original logs are available, without copying grader text."""
    return {row["raw_record_ref"]: sorted(_grader_calls(row, files))
            for row in rollouts if row.get("record_schema") == "responses-api"}


def _saved_grader_index(store: Path) -> dict[str, list[str]]:
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    rollouts = pq.read_table(
        store / "tables" / "rollouts.parquet", columns=["raw_record_ref", "record_schema"],
    ).to_pylist()
    required = {row["raw_record_ref"] for row in rollouts
                if row.get("record_schema") == "responses-api"}
    index = manifest.get("grader_call_ids_by_record")
    if index is None:
        raise SourceDataError(
            "Local snapshot has no grader attribution index; re-import the run before discovery."
        )
    if (not isinstance(index, dict) or not required <= index.keys()
            or any(not isinstance(ids, list)
                   or any(not isinstance(value, str) for value in ids)
                   for ids in index.values())):
        raise SourceDataError("Local snapshot has an invalid grader attribution index")
    return index


def _grader_calls(metadata: dict, files: dict[str, Path]) -> set[str]:
    """Graders' model-call ids in a NeMo Gym rollout; reader tables omit attribution.

    The reader keeps each capture's id but not the model reference naming its server,
    so read that from the rollout's raw record in the selected input files.
    """
    from autofa.data.source_model_calls import grader_call_ids

    name, _, offset = str(metadata.get("raw_record_ref") or "").rpartition(":")
    source = files.get(name)
    if metadata.get("record_schema") != "responses-api":
        return set()
    if source is None or not offset.isdigit():
        raise SourceDataError("Cannot attribute grader calls: local raw record is unavailable")
    return grader_call_ids(source, int(offset))


def _leading_request(events):
    messages = []
    for event in events:
        if is_evaluation_record(event) or event.get("is_copied_context"):
            continue
        role = event.get("role")
        if role in {"assistant", "agent", "output", "tool"}:
            break
        if role not in {"system", "developer", "user"}:
            continue
        text = "\n\n".join(p["text"] for p in event.get("parts", [])
                           if p.get("part_type") == "text" and isinstance(p.get("text"), str))
        if text:
            messages.append({"role": role, "content": text})
    if messages:
        return {"messages": messages,
                "limitations": ["Initial context reconstructed from recorded leading messages; "
                                "initial tool definitions were not recorded."]}
    return None


def _read_rollout(source, task: str, repeat: int, lane: str) -> dict:  # noqa: C901
    """Hydrate full original tables for one rollout, independently of viewer caps."""
    rows_cache: dict[str, list[dict]] = {}

    def rows(table):
        if table not in rows_cache:
            columns = source.reads[table]
            query = source.scan(table)
            if "task_id" in columns:
                task_match = pl.col("task_id") == task
                if table == "provenance":
                    task_match = pl.col("task_id").is_null() | task_match
                query = query.filter(task_match)
            if "repeat" in columns:
                repeat_match = pl.col("repeat") == repeat
                if table == "provenance":
                    repeat_match = pl.col("repeat").is_null() | repeat_match
                query = query.filter(repeat_match)
            keys = list(source.schema.table(table).key)
            rows_cache[table] = query.sort(keys).collect().to_dicts()
        return rows_cache[table]

    metadata_rows = rows("rollouts")
    if len(metadata_rows) != 1:
        raise SourceDataError("Local run has missing or ambiguous task/repeat identity")
    metadata = metadata_rows[0]

    def hydrated(table, text_table):
        keys = source.schema.table(table).key
        by_key = {tuple(r[k] for k in keys): r for r in rows(text_table)}
        if len(by_key) != len(rows(text_table)):
            raise SourceDataError(f"Duplicate local {text_table} identity")
        result = [{**r, **by_key.pop(tuple(r[k] for k in keys), {})} for r in rows(table)]
        if by_key:
            raise SourceDataError(f"Orphan local {text_table} record")
        return result

    events = rows("events")
    event_key = source.schema.table("events").key
    by_event = {tuple(e[k] for k in event_key): e for e in events}
    if len(by_event) != len(events):
        raise SourceDataError("Duplicate local event identity")
    for event in events:
        event["parts"] = []
    for part in hydrated("parts", "part_texts"):
        parent = by_event.get(tuple(part[k] for k in event_key))
        if parent is None:
            raise SourceDataError("Orphan local event part")
        parent["parts"].append(part)
    definitions = hydrated("tool_defs", "tool_def_texts")
    attach_event_requests(rows("event_texts"), by_event, event_key)
    calls = hydrated("tool_calls", "tool_call_texts")
    calls_by_event: Counter = Counter()
    for call in calls:
        parent_key = tuple(call[k] for k in event_key)
        if parent_key not in by_event:
            raise SourceDataError("Orphan local tool call")
        calls_by_event[parent_key] += 1
    # Validate every lane, not only the conversation selected by the exporter.
    for event in events:
        if event.get("n_parts") != len(event["parts"]):
            raise SourceDataError("Local event part count mismatch")
        if (event.get("n_tool_calls") is not None
                and event["n_tool_calls"] != calls_by_event[tuple(event[k] for k in event_key)]):
            raise SourceDataError("Local event tool-call count mismatch")
        for part in event["parts"]:
            if part.get("part_type") in {"text", "reasoning"}:
                if not isinstance(part.get("text"), str) or len(part["text"]) != part.get("chars"):
                    raise SourceDataError("Local event has missing or shortened full text")
    for call in calls:
        if (not isinstance(call.get("args_full"), str)
                or len(call["args_full"]) != call.get("args_chars")):
            raise SourceDataError("Local tool call has missing or shortened arguments")
    execution = {
        "rollout": metadata, "events": events, "tool_calls": calls,
        "attempts": rows("attempts"), "artifacts": hydrated("artifacts", "artifact_texts"),
        "tool_definitions": definitions, "prompt_messages": rows("prompt_messages"),
        "provenance": rows("provenance"), "relations": _relations(source, task, repeat),
        "availability": {}, "extra_resources": {},
    }
    for name, table in (("rollout", "rollouts"), ("events", "events"),
                        ("tool_calls", "tool_calls"), ("attempts", "attempts"),
                        ("artifacts", "artifacts"), ("tool_definitions", "tool_defs"),
                        ("prompt_messages", "prompt_messages"), ("provenance", "provenance"),
                        ("relations", "relations")):
        present = (source.directory / f"{table}.parquet").is_file()
        execution["availability"][name] = {
            "status": "available" if present else "unavailable",
            "reason": ("Imported local table read in full." if present
                       else "Not recorded by reader."),
        }
    for resource, text, length in (("artifacts", "text", "chars"),
                                    ("tool_definitions", "def_json", "def_chars")):
        if any(not isinstance(r.get(text), str) for r in execution[resource]):
            execution["availability"][resource] = {
                "status": "partial",
                "reason": "Metadata retained; some full text was not recorded.",
            }
        for record in execution[resource]:
            if record.get(text) is not None and record.get(length) is not None:
                if not isinstance(record[text], str) or len(record[text]) != record[length]:
                    raise SourceDataError(f"Local {resource} text is shortened")
    known = {"runs", "rollouts", "events", "parts", "part_texts", "event_texts", "tool_calls",
             "tool_call_texts", "attempts", "artifacts", "artifact_texts", "tool_defs",
             "tool_def_texts", "prompt_messages", "provenance", "relations"}
    for table, columns in source.reads.items():
        if table not in known and "task_id" in columns and not is_evaluation_table(table):
            execution["extra_resources"][table] = rows(table)
    execution["availability"]["extra_resources"] = {
        "status": "available", "reason": "All declared non-grading rollout tables were inspected.",
    }
    selected_events = [e for e in events if (e.get("lane") or "") == lane]
    selected_calls = [c for c in calls if (c.get("lane") or "") == lane]
    if events and not selected_events:
        raise SourceDataError(f"Local conversation lane {lane!r} is absent")
    return {
        "metadata": {**metadata, "n_events": len(selected_events),
                     "n_tool_calls": len(selected_calls)},
        "events": selected_events, "calls": selected_calls, "execution": execution,
    }


def _relations(source, task, repeat):
    def end(side):
        return ((pl.col(f"{side}_task_id") == task)
                & (pl.col(f"{side}_repeat").is_null() | (pl.col(f"{side}_repeat") == repeat)))
    return source.scan("relations").filter(end("frm") | end("to")).collect().to_dicts()
