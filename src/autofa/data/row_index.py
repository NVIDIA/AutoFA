# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-cached identity indexes (metadata / ID cols only)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# Columns used to build item ids + lightweight sampling filters.
_ID_CORE = ("model_id", "benchmark", "problem_id", "run_idx", "hb_problem_idx")
_ID_ALIASES = (
    "task_id",
    "instance_id",
    "item_id",
    "question_id",
    "repeat",
)
_LABEL_COLS = (
    "is_correct",
    "resolved",
    "tb_is_resolved",
    "is_resolved",
    "correct",
    "accuracy",
)


class DuplicateItemIdError(ValueError):
    """A lookup would ambiguously select one of multiple input rows."""


@dataclass(frozen=True)
class RowLocation:
    path: Path
    row_group: int
    local_offset: int


@dataclass
class IdentityRow:
    """Lightweight per-row record for lookup and filterable sampling."""

    global_idx: int
    item_id: str
    path: str
    row_group: int
    local_offset: int
    model_id: str
    benchmark: str
    is_correct: bool | None


@dataclass
class IdentityIndex:
    fingerprint: str
    paths: list[str]
    by_id: dict[str, RowLocation]
    rows: list[IdentityRow]


def files_fingerprint(paths: list[Path]) -> str:
    """Stable hash of ``(path, mtime, size)`` per file — rebuild indexes when exports change."""
    # Rebuild old indexes that could silently overwrite duplicate identities.
    parts: list[str] = ["identity-v3-unique"]
    for p in paths:
        try:
            st = p.stat()
            parts.append(f"{p.resolve()}\0{st.st_mtime_ns}\0{st.st_size}")
        except OSError:
            parts.append(f"{p}\0missing")
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()


def identity_index_cache_path(paths: list[Path]) -> Path:
    """Cache file beside the first parquet: ``{stem}.autofa_id_index.json``."""
    first = paths[0]
    return first.parent / f"{first.stem}.autofa_id_index.json"


def _identity_columns(schema_names: set[str]) -> list[str]:
    wanted = list(_ID_CORE) + list(_ID_ALIASES) + list(_LABEL_COLS)
    seen: set[str] = set()
    out: list[str] = []
    for c in wanted:
        if c in schema_names and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _row_is_correct(row: dict[str, Any]) -> bool | None:
    if "is_correct" in row and row["is_correct"] is not None:
        return bool(row["is_correct"])
    acc = row.get("accuracy")
    if acc is not None:
        try:
            return float(acc) >= 0.5
        except (TypeError, ValueError):
            return bool(acc)
    return None


def build_identity_index(
    paths: list[Path],
    *,
    make_item_id,
    apply_column_aliases,
    cache: bool = True,
) -> IdentityIndex:
    """Scan identity columns only; cache ``item_id -> (path, row_group, local_offset)``."""
    if not paths:
        return IdentityIndex(fingerprint="", paths=[], by_id={}, rows=[])

    fp = files_fingerprint(paths)
    cache_path = identity_index_cache_path(paths)
    if cache and cache_path.is_file():
        try:
            loaded = _load_identity_index(cache_path, expected_fp=fp)
            if loaded is not None:
                return loaded
        except Exception:  # noqa: BLE001
            pass

    by_id: dict[str, RowLocation] = {}
    rows: list[IdentityRow] = []
    global_idx = 0

    for path in paths:
        try:
            pf = pq.ParquetFile(path)
            md = pf.metadata
        except Exception:  # noqa: BLE001
            continue
        schema_names = {f.name for f in pf.schema_arrow}
        cols = _identity_columns(schema_names)
        if md is None:
            continue
        path_str = str(path)
        for rg_idx in range(md.num_row_groups):
            n_in_group = int(md.row_group(rg_idx).num_rows)
            if not cols:
                for local in range(n_in_group):
                    item_id = f"{path.stem}:row:{global_idx}"
                    loc = RowLocation(path=path, row_group=rg_idx, local_offset=local)
                    by_id[item_id] = loc
                    rows.append(
                        IdentityRow(
                            global_idx=global_idx,
                            item_id=item_id,
                            path=path_str,
                            row_group=rg_idx,
                            local_offset=local,
                            model_id="",
                            benchmark="",
                            is_correct=None,
                        )
                    )
                    global_idx += 1
                continue

            table = pf.read_row_group(rg_idx, columns=cols)
            names = table.column_names
            cols_data = {name: table.column(name).to_pylist() for name in names}
            for local in range(table.num_rows):
                row = {name: cols_data[name][local] for name in names}
                apply_column_aliases(row)
                item_id = make_item_id(row, path)
                loc = RowLocation(path=path, row_group=rg_idx, local_offset=local)
                if item_id in by_id:
                    raise DuplicateItemIdError(
                        f"duplicate item ID {item_id!r}: {by_id[item_id]} and {loc}; "
                        "provide distinct problem_id/run_idx values"
                    )
                by_id[item_id] = loc
                rows.append(
                    IdentityRow(
                        global_idx=global_idx,
                        item_id=item_id,
                        path=path_str,
                        row_group=rg_idx,
                        local_offset=local,
                        model_id=str(row.get("model_id") or ""),
                        benchmark=str(row.get("benchmark") or ""),
                        is_correct=_row_is_correct(row),
                    )
                )
                global_idx += 1

    index = IdentityIndex(
        fingerprint=fp,
        paths=[str(p) for p in paths],
        by_id=by_id,
        rows=rows,
    )
    if cache:
        try:
            _save_identity_index(cache_path, index)
        except Exception:  # noqa: BLE001
            pass
    return index


def _save_identity_index(path: Path, index: IdentityIndex) -> None:
    payload = {
        "fingerprint": index.fingerprint,
        "paths": index.paths,
        "by_id": {
            iid: {
                "path": loc.path if isinstance(loc.path, str) else str(loc.path),
                "row_group": loc.row_group,
                "local_offset": loc.local_offset,
            }
            for iid, loc in index.by_id.items()
        },
        "rows": [
            {
                "g": r.global_idx,
                "id": r.item_id,
                "p": r.path,
                "rg": r.row_group,
                "lo": r.local_offset,
                "m": r.model_id,
                "b": r.benchmark,
                "c": r.is_correct,
            }
            for r in index.rows
        ],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _load_identity_index(path: Path, *, expected_fp: str) -> IdentityIndex | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != expected_fp:
        return None
    by_id: dict[str, RowLocation] = {}
    for iid, loc in (data.get("by_id") or {}).items():
        by_id[str(iid)] = RowLocation(
            path=Path(loc["path"]),
            row_group=int(loc["row_group"]),
            local_offset=int(loc["local_offset"]),
        )
    rows: list[IdentityRow] = []
    for r in data.get("rows") or []:
        rows.append(
            IdentityRow(
                global_idx=int(r["g"]),
                item_id=str(r["id"]),
                path=str(r["p"]),
                row_group=int(r["rg"]),
                local_offset=int(r["lo"]),
                model_id=str(r.get("m") or ""),
                benchmark=str(r.get("b") or ""),
                is_correct=r.get("c", None),
            )
        )
    if len({r.item_id for r in rows}) != len(rows) or len(by_id) != len(rows):
        return None
    return IdentityIndex(
        fingerprint=expected_fp,
        paths=list(data.get("paths") or []),
        by_id=by_id,
        rows=rows,
    )


def column_byte_stats(paths: list[Path]) -> dict[str, dict[str, int]]:
    """Sum ``ColumnChunkMetaData.total_uncompressed_size`` (and compressed) per column.

    Returns ``{column: {uncompressed_bytes, compressed_bytes, num_values}}``.
    """
    stats: dict[str, dict[str, int]] = {}
    for path in paths:
        try:
            pf = pq.ParquetFile(path)
            md = pf.metadata
        except Exception:  # noqa: BLE001
            continue
        if md is None:
            continue
        for rg_idx in range(md.num_row_groups):
            rg = md.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                chunk = rg.column(col_idx)
                name = chunk.path_in_schema
                slot = stats.setdefault(
                    name,
                    {"uncompressed_bytes": 0, "compressed_bytes": 0, "num_values": 0},
                )
                slot["uncompressed_bytes"] += int(chunk.total_uncompressed_size or 0)
                slot["compressed_bytes"] += int(chunk.total_compressed_size or 0)
                slot["num_values"] += int(chunk.num_values or 0)
    return stats


def _iter_indexed_rows_at_locations(
    locations: Iterable[tuple[int, RowLocation]],
    *,
    make_item_id,
    apply_column_aliases,
    resolve_columns: Callable[[set[str]], list[str]],
    group_batch_limits: dict[tuple[Path, int], int] | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read ordered requests with one open file and one active Arrow batch.

    Forward requests reuse the row-group decoder. A backwards seek restarts
    that group without retaining previously returned traces. File and decoder
    lifetimes belong to this iterator, including when its consumer stops early.
    """
    from autofa.data.input_view import attach_compact_export_input, has_compact_export_provenance
    from autofa.data.parquet_store import DEFAULT_BATCH_SIZE, MAX_BATCH_BYTES
    from autofa.data.source_export import require_current_source_cache
    from autofa.util.memory import row_group_bytes_per_row

    pf = batches = batch = None
    active_path = None
    active_group = None
    metadata = group = None
    read_cols = None
    start = 0
    compact_export = False

    def close_batches():
        nonlocal batches, batch
        batch = None
        if batches is not None:
            close = getattr(batches, "close", None)
            if close is not None:
                close()
            batches = None

    try:
        for req_i, loc in locations:
            path = Path(loc.path)
            if path != active_path:
                close_batches()
                if pf is not None:
                    pf.close()
                pf = None
                metadata = group = None
                active_group = None
                active_path = path
                try:
                    pf = pq.ParquetFile(path)
                except Exception:  # noqa: BLE001
                    continue
                metadata = pf.metadata
                schema_names = set(pf.schema_arrow.names)
                read_cols = resolve_columns(schema_names)
                compact_export = has_compact_export_provenance(path)
                require_current_source_cache(path)
            if pf is None or metadata is None:
                continue

            rg_idx = int(loc.row_group)
            local = int(loc.local_offset)
            if rg_idx != active_group:
                close_batches()
                active_group = rg_idx
                group = None
                try:
                    group = metadata.row_group(rg_idx)
                except Exception:  # noqa: BLE001
                    continue
            if group is None or not 0 <= local < group.num_rows:
                continue
            if batches is None or local < start:
                close_batches()
                row_bytes = row_group_bytes_per_row(group, read_cols)
                batch_size = max(1, min(DEFAULT_BATCH_SIZE, group.num_rows,
                                        MAX_BATCH_BYTES // max(1, row_bytes)))
                if group_batch_limits is not None:
                    batch_size = min(batch_size, group_batch_limits[(path, rg_idx)])
                start = 0
                batches = pf.iter_batches(
                    batch_size=batch_size, columns=read_cols, row_groups=[rg_idx],
                )
            while batch is None or local >= start + batch.num_rows:
                if batch is not None:
                    start += batch.num_rows
                    batch = None
                try:
                    batch = next(batches)
                except Exception:  # noqa: BLE001
                    close_batches()
                    # Match point-read behavior: unreadable groups are skipped.
                    group = None
                    break
            if batch is None:
                continue
            # Convert only the requested row, including a fresh dict for each
            # duplicate request. Arrow decoding is byte-sized from metadata;
            # a single oversized row is the irreducible working-set minimum.
            row = batch.slice(local - start, 1).to_pylist()[0]
            apply_column_aliases(row)
            if compact_export:
                attach_compact_export_input(row)
            row["_item_id"] = make_item_id(row, path)
            row["_source_parquet"] = str(path)
            yield req_i, row
            del row
    finally:
        close_batches()
        if pf is not None:
            pf.close()


def iter_rows_at_locations(
    locations: Iterable[RowLocation],
    *,
    make_item_id,
    apply_column_aliases,
    resolve_columns: Callable[[set[str]], list[str]],
) -> Iterator[dict[str, Any]]:
    """Yield requested rows in caller order, skipping invalid locations.

    Duplicate locations yield independent rows. Requests grouped by file and
    increasing row-group/offset reuse both the open file and forward decoding;
    arbitrary order is supported with bounded memory by restarting as needed.
    """
    rows = _iter_indexed_rows_at_locations(
        enumerate(locations), make_item_id=make_item_id,
        apply_column_aliases=apply_column_aliases, resolve_columns=resolve_columns,
    )
    try:
        for _, row in rows:
            yield row
    finally:
        rows.close()


def read_rows_at_locations(
    locations: list[RowLocation],
    *,
    make_item_id,
    apply_column_aliases,
    resolve_columns: Callable[[set[str]], list[str]],
) -> list[dict[str, Any]]:
    """Materialize selected rows, restoring caller order after grouped reads."""
    # The list API already owns all returned rows, so reorder lightweight
    # locations for efficient decoding and restore the request order afterward.
    files: dict[Path, dict[int, list[tuple[int, RowLocation]]]] = {}
    group_batch_limits: dict[tuple[Path, int], int] = {}
    for req_i, loc in enumerate(locations):
        loc = RowLocation(Path(loc.path), int(loc.row_group), int(loc.local_offset))
        groups = files.setdefault(loc.path, {})
        groups.setdefault(loc.row_group, []).append((req_i, loc))
        key = (loc.path, loc.row_group)
        group_batch_limits[key] = max(group_batch_limits.get(key, 1), loc.local_offset + 1)
    requests = (
        request for groups in files.values() for group in groups.values()
        for request in sorted(group, key=lambda request: request[1].local_offset)
    )
    results: list[dict[str, Any] | None] = [None] * len(locations)
    rows = _iter_indexed_rows_at_locations(
        requests, make_item_id=make_item_id,
        apply_column_aliases=apply_column_aliases, resolve_columns=resolve_columns,
        group_batch_limits=group_batch_limits,
    )
    try:
        for req_i, row in rows:
            results[req_i] = row
    finally:
        rows.close()
    return [row for row in results if row is not None]
