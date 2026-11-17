# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and index a run's trace ZIP on disk, then hydrate one rollout at a time."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from autofa.data.execution_policy import is_evaluation_table
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.source_request_references import attach_event_requests

INDEX_VERSION = 2
DOWNLOAD_VERSION = 1
SELECTION_BATCH_SIZE = 500
_ROLLOUT = ("run_id", "task_id", "repeat", "agent_ref")
_EVENT = (*_ROLLOUT, "lane", "event_idx")
_KEYS = {
    "rollouts": _ROLLOUT,
    "events": _EVENT,
    "parts": (*_EVENT, "part_idx"),
    "part_texts": (*_EVENT, "part_idx"),
    "event_texts": _EVENT,
    "prompt_messages": (*_ROLLOUT, "msg_idx"),
    "tool_calls": (*_EVENT, "call_idx"),
    "tool_call_texts": (*_EVENT, "call_idx"),
    "attempts": (*_ROLLOUT, "attempt_idx"),
    "artifacts": (*_ROLLOUT, "artifact_idx"),
    "artifact_texts": (*_ROLLOUT, "artifact_idx"),
    "tool_defs": (*_ROLLOUT, "def_idx"),
    "tool_def_texts": (*_ROLLOUT, "def_idx"),
}
_PARENTS = {
    "events": "rollouts",
    "parts": "events",
    "part_texts": "parts",
    "event_texts": "events",
    "prompt_messages": "rollouts",
    "tool_calls": "events",
    "tool_call_texts": "tool_calls",
    "attempts": "rollouts",
    "artifacts": "rollouts",
    "artifact_texts": "artifacts",
    "tool_defs": "rollouts",
    "tool_def_texts": "tool_defs",
}
_REQUIRED = {"rollouts", "events", "parts", "part_texts", "tool_calls", "tool_call_texts"}


class InvalidTraceArchive(DataApiError):
    """The downloaded bytes violate the trace contract and must be fetched again."""


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _receipt(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_receipt(path: Path, value: dict) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value) + "\n")
    partial.replace(path)


def _open_parquet(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(path)
    except OSError as exc:
        if exc.errno is not None:
            raise
        # Arrow also uses plain OSError for malformed Parquet headers/pages.
        raise InvalidTraceArchive(f"Invalid trace Parquet: {exc}") from exc


def _parquet_batches(parquet: pq.ParquetFile) -> Iterator[pa.RecordBatch]:
    try:
        yield from parquet.iter_batches(batch_size=32)
    except OSError as exc:
        if exc.errno is not None:
            raise
        raise InvalidTraceArchive(f"Invalid trace Parquet: {exc}") from exc


def _count(value, label: str) -> int:
    if type(value) is not int or value < 0:
        raise InvalidTraceArchive(f"Invalid trace export {label}")
    return value


def _key(row: dict, fields: tuple[str, ...]) -> str:
    values = []
    for field in fields:
        value = row.get(field)
        if field in {
            "repeat",
            "event_idx",
            "part_idx",
            "call_idx",
            "attempt_idx",
            "artifact_idx",
            "def_idx", "msg_idx",
        }:
            _count(value, field)
        elif not isinstance(value, str) or (field in {"run_id", "task_id"} and not value):
            raise InvalidTraceArchive(f"Invalid trace export {field}")
        values.append(value)
    return json.dumps(values, ensure_ascii=False)


def _belongs_to_rollout(row: dict, identity: dict, table: str) -> bool:
    def matches(prefix=""):
        return all(row.get(prefix + key) is None or row[prefix + key] == identity[key]
                   for key in ("task_id", "repeat"))

    if table == "relations" and any(row.get(prefix + "task_id") is not None
                                    for prefix in ("frm_", "to_")):
        return any(row.get(prefix + "task_id") is not None and matches(prefix)
                   for prefix in ("frm_", "to_"))
    return matches() and (row.get("agent_ref") is None
                          or row["agent_ref"] == identity["agent_ref"])


class TraceArchive:
    """A checksum-checked disk cache; no run-wide text table is loaded into memory.

    The immutable ZIP is retained to rebuild an interrupted/corrupt index without
    another HTTP request. Only a fully validated index receives a cache receipt.
    """

    def __init__(self, root: Path, *, api, run_id: str, progress=None, rollouts=None,
                 activity: Callable[[str], None] | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive = self.root / "trace.zip"
        self.index = self.root / "trace.sqlite"
        receipt_path = self.root / "index.json"
        download_path = self.root / "download.json"
        receipt = _receipt(receipt_path)
        downloaded = _receipt(download_path)
        selection = [list(identity) for identity in rollouts] if rollouts is not None else None
        self.selection = selection
        if activity:
            activity("Verifying trace tables…")
        expected = {"version": DOWNLOAD_VERSION, "run_id": run_id, "selection": selection}
        same = all(downloaded.get(k) == v for k, v in expected.items())
        archive_ok = (
            same
            and self.archive.is_file()
            and downloaded.get("archive_sha256") == _digest(self.archive)
        )
        if not archive_ok:
            kwargs = {"rollouts": rollouts} if rollouts is not None else {}
            api.download_trace_export(run_id, self.archive, progress=progress, **kwargs)
            if activity:
                activity("Verifying trace tables…")
            downloaded = {**expected, "archive_sha256": _digest(self.archive)}
            _write_receipt(download_path, downloaded)
        index_expected = {**downloaded, "index_version": INDEX_VERSION}
        index_ok = (
            all(receipt.get(k) == v for k, v in index_expected.items())
            and self.index.is_file()
            and receipt.get("index_sha256") == _digest(self.index)
        )
        if not index_ok:
            if activity:
                activity("Indexing trace tables…")
            try:
                self._build(run_id)
            except InvalidTraceArchive:
                # A completed transfer can still contain invalid source data.
                # Retrying must fetch repaired bytes instead of the same bad ZIP.
                for path in (download_path, receipt_path, self.index, self.archive):
                    path.unlink(missing_ok=True)
                raise
            receipt = {**index_expected, "index_sha256": _digest(self.index)}
            _write_receipt(receipt_path, receipt)
        self.digest = receipt["archive_sha256"]
        self.db = sqlite3.connect(f"{self.index.as_uri()}?mode=ro", uri=True)
        self.db.execute("PRAGMA cache_size=-8192")
        if activity:
            activity("Preparing rollout traces…")

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _build(self, run_id: str) -> None:
        temp_index = self.index.with_suffix(".partial")
        temp_index.unlink(missing_ok=True)
        try:
            with closing(sqlite3.connect(temp_index)) as db, db:
                db.execute("PRAGMA cache_size=-8192")
                db.execute("PRAGMA temp_store=FILE")
                db.execute(
                    "CREATE TABLE records (tbl TEXT, key TEXT, rollout TEXT, lane TEXT, "
                    "event_idx INTEGER, part_idx INTEGER, call_idx INTEGER, parent TEXT, "
                    "payload TEXT, PRIMARY KEY(tbl,key)) WITHOUT ROWID"
                )
                db.execute(
                    "CREATE TABLE expected (tbl TEXT, rollout TEXT, n INTEGER, "
                    "PRIMARY KEY(tbl,rollout)) WITHOUT ROWID"
                )
                with (
                    zipfile.ZipFile(self.archive) as archive,
                    tempfile.TemporaryDirectory(
                        dir=self.root,
                        prefix="tables-",
                    ) as scratch,
                ):
                    self._index_archive(db, archive, Path(scratch), run_id)
                db.execute(
                    "CREATE INDEX by_rollout ON records"
                    "(tbl,rollout,lane,event_idx,part_idx,call_idx)"
                )
                db.execute("CREATE INDEX by_parent ON records(tbl,parent)")
                db.execute(
                    "CREATE INDEX rollout_identity ON records("
                    "json_extract(payload,'$.task_id'),json_extract(payload,'$.repeat')) "
                    "WHERE tbl='rollouts'"
                )
                self._validate_relations(db)
            temp_index.replace(self.index)
        except (
            zipfile.BadZipFile,
            zlib.error,
            EOFError,
            sqlite3.IntegrityError,
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            ValueError,
            KeyError,
            TypeError,
            UnicodeError,
        ) as exc:
            raise InvalidTraceArchive(f"Invalid bulk trace export: {exc}") from exc
        except (OSError, sqlite3.OperationalError) as exc:
            raise DataApiError(f"Could not build the local trace index: {exc}") from exc
        finally:
            temp_index.unlink(missing_ok=True)

    def _index_archive(self, db, archive, scratch: Path, run_id: str) -> None:  # noqa: C901
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise InvalidTraceArchive("Trace ZIP has duplicate entries or lacks manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("exporter") != "trace"
            or type(manifest.get("exporter_version")) is not int
            or manifest.get("exporter_version") != 1
            or manifest.get("run_id") != run_id
            or not isinstance(manifest.get("schema"), str)
        ):
            raise InvalidTraceArchive(
                "Trace export manifest has an unsupported format or run identity"
            )
        tables, rollouts = manifest.get("tables"), manifest.get("rollouts")
        if not isinstance(tables, dict) or not _REQUIRED <= tables.keys():
            raise InvalidTraceArchive("Trace export is missing required tables")
        if not isinstance(rollouts, list) or len(rollouts) != _count(
            manifest.get("n_rollouts"), "n_rollouts"
        ):
            raise InvalidTraceArchive("Trace export rollout count does not match its manifest")
        if not isinstance(tables["rollouts"], dict) or tables["rollouts"].get("rows") != len(
            rollouts
        ):
            raise InvalidTraceArchive("Trace manifest rollout table count mismatch")
        if self.selection is not None and {
            (row.get("task_id"), row.get("repeat")) for row in rollouts if isinstance(row, dict)
        } != {tuple(identity) for identity in self.selection}:
            raise InvalidTraceArchive("Trace manifest rollouts differ from requested selection")
        wanted = {"manifest.json"}
        for table, info in tables.items():
            if (
                not re.fullmatch(r"[a-z][a-z0-9_]*", table)
                or not isinstance(info, dict)
                or info.get("file") != f"{table}.parquet"
            ):
                raise InvalidTraceArchive("Trace export has an invalid table filename")
            wanted.add(info["file"])
            _count(info.get("rows"), f"{table} row count")
        if set(names) != wanted:
            raise InvalidTraceArchive("Trace ZIP entries do not match its manifest tables")
        seen = set()
        for row in rollouts:
            if not isinstance(row, dict):
                raise InvalidTraceArchive("Trace manifest has an invalid rollout")
            key = _key({"run_id": run_id, **row}, _ROLLOUT)
            if key in seen:
                raise InvalidTraceArchive("Trace manifest has duplicate rollout identities")
            seen.add(key)
            _count(row.get("n_events"), "n_events")
            _count(row.get("n_tool_calls"), "n_tool_calls")
            if not isinstance(row.get("rows"), dict) or row["rows"].keys() != tables.keys():
                raise InvalidTraceArchive("Trace manifest lacks per-rollout table counts")
            if row["rows"]["rollouts"] != 1:
                raise InvalidTraceArchive("Trace manifest must name one row per rollout identity")
            if row["n_events"] > _count(row["rows"]["events"], "event rows") or row[
                "n_tool_calls"
            ] > _count(row["rows"]["tool_calls"], "tool-call rows"):
                raise InvalidTraceArchive("Trace manifest rollup exceeds its table row count")
            for table, n in row["rows"].items():
                db.execute("INSERT INTO expected VALUES (?,?,?)", (table, key, _count(n, table)))
            # Rollup counts need not equal all-lane table counts: copied context
            # and auxiliary lanes are excluded by the producer's rollup rules.
            db.execute(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    "_manifest",
                    key,
                    key,
                    "",
                    None,
                    None,
                    None,
                    None,
                    json.dumps(row),
                ),
            )
        for table, info in tables.items():
            path = scratch / info["file"]
            # Reading every member to EOF verifies its ZIP CRC, including tables
            # that are not currently used as model-visible evidence.
            with archive.open(info["file"]) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            parquet = _open_parquet(path)
            identity_fields = {"run_id"} if table in {"provenance", "relations"} else set(_ROLLOUT)
            if (parquet.metadata.num_rows != info["rows"]
                    or not identity_fields <= set(parquet.schema_arrow.names)):
                raise InvalidTraceArchive(
                    f"Trace table {table} has invalid row counts or identity columns"
                )
            fields = _KEYS.get(table)
            ordinal = 0
            for batch in _parquet_batches(parquet):
                for row in batch.to_pylist():
                    if row.get("run_id") != run_id:
                        raise InvalidTraceArchive(f"Trace table {table} contains another run")
                    if table in {"provenance", "relations"}:
                        # Viewer provenance may be run-level, and relation rows identify
                        # endpoints rather than carrying ordinary rollout identity columns.
                        targets = [identity for identity in rollouts
                                   if _belongs_to_rollout(row, identity, table)]
                        if not targets:
                            raise InvalidTraceArchive(f"Trace {table} row has no exported rollout")
                        identities = [_key({"run_id": run_id, **identity}, _ROLLOUT)
                                      for identity in targets]
                    else:
                        identities = [_key(row, _ROLLOUT)]
                    for rollout in identities:
                        key = _key(row, fields) if fields else f"{rollout}:{ordinal}"
                        parent_table = _PARENTS.get(table, "rollouts")
                        parent = (rollout if table in {"provenance", "relations"} else
                                  _key(row, _KEYS[parent_table]) if table != "rollouts" else None)
                        db.execute(
                            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                            (table, key, rollout, row.get("lane", ""), row.get("event_idx"),
                             row.get("part_idx"), row.get("call_idx"), parent,
                             json.dumps(row, ensure_ascii=False, default=str)),
                        )
                    ordinal += 1
            path.unlink()
        bad = db.execute("""
            WITH actual AS (SELECT tbl,rollout,count(*) n FROM records
                            WHERE tbl != '_manifest' GROUP BY tbl,rollout)
            SELECT e.tbl FROM expected e LEFT JOIN actual a USING(tbl,rollout)
            WHERE e.n != coalesce(a.n,0)
            UNION ALL SELECT a.tbl FROM actual a LEFT JOIN expected e USING(tbl,rollout)
            WHERE e.tbl IS NULL LIMIT 1
        """).fetchone()
        if bad:
            raise InvalidTraceArchive(
                f"Trace table {bad[0]} counts/rollout identities differ from manifest"
            )
        for payload, expected in db.execute(
            "SELECT r.payload,m.payload FROM records r JOIN records m "
            "ON m.tbl='_manifest' AND r.key=m.key WHERE r.tbl='rollouts'"
        ):
            row, listed = json.loads(payload), json.loads(expected)
            if any(_count(row.get(k), k) != listed[k] for k in ("n_events", "n_tool_calls")):
                raise InvalidTraceArchive("Trace rollout rollups differ from manifest")

    @staticmethod
    def _validate_relations(db) -> None:
        for table, parent in _PARENTS.items():
            bad = db.execute(
                "SELECT 1 FROM records c LEFT JOIN records p ON p.tbl=? AND p.key=c.parent "
                "WHERE c.tbl=? AND p.key IS NULL LIMIT 1",
                (parent, table),
            ).fetchone()
            if bad:
                raise InvalidTraceArchive(f"Trace export has orphan {table} rows")
        # Validate text and argument lengths for every lane, not just the chosen
        # conversation. The index must never certify a partially joined export.
        for table, text_table, value, length in (
            ("parts", "part_texts", "text", "chars"),
            ("tool_calls", "tool_call_texts", "args_full", "args_chars"),
        ):
            rows = db.execute(
                "SELECT p.payload,t.payload FROM records p LEFT JOIN records t "
                "ON t.tbl=? AND t.key=p.key WHERE p.tbl=?",
                (text_table, table),
            )
            for payload, hydrated in rows:
                row = json.loads(payload)
                if table == "parts" and row.get("part_type") not in {"text", "reasoning"}:
                    continue
                text = json.loads(hydrated).get(value) if hydrated else None
                if not isinstance(text, str) or len(text) != _count(row.get(length), length):
                    raise InvalidTraceArchive(f"Trace export is missing or has shortened {value}")
        # Optional full payloads must not silently contain shortened strings when
        # their source metadata supplies a length. Missing payloads remain unavailable.
        for table, text_table, value, length in (
            ("tool_defs", "tool_def_texts", "def_json", "def_chars"),
            ("artifacts", "artifact_texts", "text", "chars"),
        ):
            for payload, hydrated in db.execute(
                "SELECT p.payload,t.payload FROM records p JOIN records t "
                "ON t.tbl=? AND t.key=p.key WHERE p.tbl=?", (text_table, table),
            ):
                row, text = json.loads(payload), json.loads(hydrated).get(value)
                if text is not None and not isinstance(text, str):
                    raise InvalidTraceArchive(f"Trace export has invalid {value}")
                if row.get(length) is not None and text is not None:
                    if len(text) != _count(row[length], length):
                        raise InvalidTraceArchive(f"Trace export has shortened {value}")
        for payload, n_parts, n_calls in db.execute("""
            SELECT e.payload,
              (SELECT count(*) FROM records p WHERE p.tbl='parts' AND p.parent=e.key),
              (SELECT count(*) FROM records c WHERE c.tbl='tool_calls' AND c.parent=e.key)
            FROM records e WHERE e.tbl='events'
        """):
            event = json.loads(payload)
            if n_parts != _count(event.get("n_parts"), "n_parts"):
                raise InvalidTraceArchive("Trace export event part count mismatch")
            if event.get("n_tool_calls") is not None and n_calls != _count(
                event["n_tool_calls"], "n_tool_calls"
            ):
                raise InvalidTraceArchive("Trace export event tool-call count mismatch")

    def rollout(self, task: str, repeat: int, *, lane: str) -> dict:
        candidates = self.db.execute(
            "SELECT rollout,payload FROM records WHERE tbl='rollouts' "
            "AND json_extract(payload,'$.task_id')=? "
            "AND json_extract(payload,'$.repeat')=?",
            (task, repeat),
        ).fetchall()
        if len(candidates) != 1:
            raise DataApiError("Trace export has missing or ambiguous task/repeat identity")
        key, payload = candidates[0]
        metadata = json.loads(payload)
        tables = {
            row[0] for row in self.db.execute(
                "SELECT tbl FROM expected WHERE rollout=?", (key,)
            )
        }

        def rows(table):
            return [json.loads(row[0]) for row in self.db.execute(
                "SELECT payload FROM records WHERE tbl=? AND rollout=? "
                "ORDER BY lane,event_idx,part_idx,call_idx,key", (table, key),
            )]

        def hydrated(table, text_table):
            result = []
            for payload, text in self.db.execute(
                "SELECT p.payload,t.payload FROM records p LEFT JOIN records t "
                "ON t.tbl=? AND t.key=p.key WHERE p.tbl=? AND p.rollout=? "
                "ORDER BY p.lane,p.event_idx,p.part_idx,p.call_idx,p.key",
                (text_table, table, key),
            ):
                row = json.loads(payload)
                if text:
                    row.update(json.loads(text))
                result.append(row)
            return result

        all_events = rows("events")
        event_map = {(event.get("lane") or "", event["event_idx"]): event
                     for event in all_events}
        for event in all_events:
            event["parts"] = []
        for part in hydrated("parts", "part_texts"):
            event_map[(part.get("lane") or "", part["event_idx"])]["parts"].append(part)
        attach_event_requests(
            rows("event_texts"),
            {tuple(event[k] for k in _EVENT): event for event in all_events},
            _EVENT,
        )
        all_calls = hydrated("tool_calls", "tool_call_texts")
        events = [event for event in all_events if (event.get("lane") or "") == lane]
        calls = [call for call in all_calls if (call.get("lane") or "") == lane]
        if not events and all_events:
            raise DataApiError(f"Exported conversation lane {lane!r} is absent from trace tables")
        execution = {
            "rollout": metadata,
            "events": all_events,
            "tool_calls": all_calls,
            "attempts": rows("attempts"),
            "prompt_messages": rows("prompt_messages"),
            "prompt_summaries": rows("prompt_summaries"),
            "artifacts": hydrated("artifacts", "artifact_texts"),
            "tool_definitions": hydrated("tool_defs", "tool_def_texts"),
            "provenance": rows("provenance"),
            "relations": rows("relations"),
            "availability": {},
        }
        resources = {
            "rollout": "rollouts", "events": "events", "tool_calls": "tool_calls",
            "attempts": "attempts", "artifacts": "artifacts",
            "prompt_messages": "prompt_messages",
            "prompt_summaries": "prompt_summaries",
            "tool_definitions": "tool_defs", "provenance": "provenance",
            "relations": "relations", "event_texts": "event_texts",
            "artifact_texts": "artifact_texts", "tool_definition_texts": "tool_def_texts",
        }
        known_tables = {*resources.values(), "parts", "part_texts", "tool_call_texts"}
        execution["extra_resources"] = {
            table: rows(table) for table in sorted(tables - known_tables)
            if not is_evaluation_table(table)
        }
        execution["availability"]["extra_resources"] = {
            "status": "available",
            "reason": "Additional execution tables retained; evaluation tables excluded.",
        }
        for table in execution["extra_resources"]:
            execution["availability"][f"extra_resources.{table}"] = {
                "status": "available",
                "reason": "The exported extension table was read completely.",
            }
        for resource, table in resources.items():
            present = table in tables
            execution["availability"][resource] = {
                "status": "available" if present else "unavailable",
                "reason": ("Exported table retrieved completely; evaluation records are excluded."
                           if present else f"The trace export has no {table} table."),
            }
        for resource, text_key in (("artifacts", "text"), ("tool_definitions", "def_json")):
            missing = sum(not isinstance(row.get(text_key), str)
                          for row in execution[resource])
            if missing:
                execution["availability"][resource] = {
                    "status": "partial",
                    "reason": f"All exported metadata retained; {missing} records lack exported "
                              f"{text_key}. Binary bytes are not included in text exports.",
                }
        expected_artifacts = metadata.get("n_artifacts")
        if (type(expected_artifacts) is int and expected_artifacts > len(execution["artifacts"])
                and "artifacts" in tables):
            execution["availability"]["artifacts"] = {
                "status": "partial",
                "reason": "The rollout metadata counts more artifacts than the exported table; "
                          "only the exported records and text are available.",
            }
        # Rollup counts can differ from all-lane table counts. Preserve the original
        # rollout metadata above; normalize the selected lane using its actual counts.
        selected_metadata = {**metadata, "n_events": len(events), "n_tool_calls": len(calls)}
        return {"metadata": selected_metadata, "events": events, "calls": calls,
                "execution": execution}


class TraceArchives:
    """Cache bounded selections independently and keep only one index open."""

    def __init__(self, root: Path, *, api: DataApiClient, run_id: str,
                 identities: list[tuple[str, int]], whole_run: bool,
                 progress: Callable[[int, int | None], None] | None = None,
                 activity: Callable[[str], None] | None = None) -> None:
        self.root, self.api, self.run_id, self.progress = root, api, run_id, progress
        self.activity = activity
        self.selections = ([None] if whole_run else [
            identities[i:i + SELECTION_BATCH_SIZE]
            for i in range(0, len(identities), SELECTION_BATCH_SIZE)
        ])
        self.locations = ({identity: 0 for identity in identities} if whole_run else {
            identity: index
            for index, selection in enumerate(self.selections) for identity in selection
        })
        self.exports: list[dict] = []
        self._exported: set[int] = set()
        self.current: TraceArchive | None = None
        self.current_index: int | None = None
        self.digest = ""

    def __enter__(self) -> TraceArchives:
        return self

    def __exit__(self, *exc) -> None:
        if self.current is not None:
            self.current.close()

    def rollout(self, task: str, repeat: int, *, lane: str) -> dict:
        index = self.locations[(task, repeat)]
        if index != self.current_index:
            if self.current is not None:
                self.current.close()
                self.current = None
                self.current_index = None
            selection = self.selections[index]
            suffix = hashlib.sha256(json.dumps(selection).encode()).hexdigest()[:16]
            path = self.root if len(self.selections) == 1 else self.root / f"batch-{suffix}"
            self.current = TraceArchive(path, api=self.api, run_id=self.run_id,
                                        progress=self.progress, rollouts=selection,
                                        activity=self.activity)
            self.current_index = index
            self.digest = self.current.digest
            if index not in self._exported:
                self.exports.append({"sha256": self.digest, "rollouts": selection})
                self._exported.add(index)
        return self.current.rollout(task, repeat, lane=lane)
