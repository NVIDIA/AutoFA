# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet dataset access for AutoFA."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from autofa.data.outcomes import discovery_is_correct
from autofa.data.row_index import (
    DuplicateItemIdError,
    IdentityIndex,
    RowLocation,
    build_identity_index,
    iter_rows_at_locations,
    read_rows_at_locations,
)
from autofa.data.row_index import (
    column_byte_stats as _column_byte_stats,
)
from autofa.data.trajectory import looks_like_agent_json, parse_trajectory
from autofa.defaults import default_value

TEXT_FIELDS = (
    "system_text",
    "prompt_text",
    "response_text",
    "reasoning_text",
    "reasoning_content",
    "generation",
    "judge_text",
    "question",
    "expected_answer",
    "predicted_answer",
    "hb_scoring_details_json",
    "hb_failure_category",
)

IDENTITY_FIELDS = (
    "model_id",
    "benchmark",
    "problem_id",
    "run_idx",
    "is_correct",
    "accuracy",
    "strata_difficulty",
    "strata_subject",
    "strata_subset",
    "domain",
    "topic",
    "finish_reason",
    "vpr_log_formats",
)

# Fill missing canonical fields from common alternate column names.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "prompt_text": ("instruction", "question", "question_content", "prompt"),
    "response_text": ("generation", "answer", "completion"),
    "reasoning_text": ("reasoning_content", "chain_of_thought", "cot"),
    "is_correct": ("resolved", "tb_is_resolved", "is_resolved", "correct"),
    "problem_id": ("task_id", "instance_id", "item_id", "question_id"),
    "run_idx": ("repeat",),
}

PLACEHOLDER_RESPONSES = {
    "",
    "[workspace modified]",
    "[workspace_modified]",
    "none",
    "null",
}

# Hard ceiling when caller passes limit=None — loading a full multi-GB parquet
# as Python dicts has segfaulted / OOM'd processes.
ABSOLUTE_MAX_LOAD = 50_000
DEFAULT_BATCH_SIZE = 2_048
# Bound streaming working sets by bytes as well as rows. A single oversized
# trace is yielded intact: this is a batch budget, not a per-trace truncation.
MAX_BATCH_BYTES = 64 * 1024**2


def iter_arrow_rows(batch: Any):
    """Convert byte-limited slices, even when dictionary metadata undersizes a read.

    The native Arrow batch is already decoded and may exceed the budget. Check
    each zero-copy slice's actual bytes before expanding it into Python objects;
    a single oversized row remains intact.
    """
    start = 0
    slice_rows = max(1, min(batch.num_rows,
                            MAX_BATCH_BYTES * batch.num_rows // max(1, batch.nbytes * 4)))
    while start < batch.num_rows:
        part = batch.slice(start, min(slice_rows, batch.num_rows - start))
        while part.num_rows > 1 and part.nbytes * 4 > MAX_BATCH_BYTES:
            part = part.slice(0, max(1, part.num_rows // 2))
        rows = part.to_pylist()
        yield from rows
        start += part.num_rows
        del rows, part


def item_memory_bytes(item: Any) -> int:
    """Estimate owned Python bytes without counting aliased fields twice."""
    total = 0
    seen: set[int] = set()
    pending = [item]
    while pending:
        value = pending.pop()
        ident = id(value)
        if ident in seen:
            continue
        seen.add(ident)
        total += sys.getsizeof(value)
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
    return total

# Re-export for describe_item_fields / schema tooling.
column_byte_stats = _column_byte_stats


@dataclass
class Dataset:
    parquet_paths: list[Path]
    model_id: str | None = None
    benchmarks: list[str] = field(default_factory=list)  # empty = all
    _identity_index: IdentityIndex | None = field(default=None, repr=False)

    def allowed_benchmarks(self) -> set[str] | None:
        """None means no filter (all benches)."""
        if self.benchmarks:
            return set(self.benchmarks)
        return None

    def _bench_ok(self, row: dict[str, Any]) -> bool:
        allowed = self.allowed_benchmarks()
        if not allowed:
            return True
        return str(row.get("benchmark") or "") in allowed

    def available_columns(self) -> set[str]:
        cols: set[str] = set()
        for path in self.parquet_paths:
            pf = pq.ParquetFile(path)
            cols.update(f.name for f in pf.schema_arrow)
        return cols

    @staticmethod
    def _resolve_schema_columns(columns: list[str] | None, available: set[str]) -> list[str]:
        if columns is None:
            wanted = list(IDENTITY_FIELDS) + list(TEXT_FIELDS) + [
                "trace_json", "raw_row_json", "input", "messages", "conversation", "trace",
                "input_json", "tools",
            ]
            for canonical, aliases in COLUMN_ALIASES.items():
                if canonical in wanted or canonical in available:
                    wanted.extend(aliases)
        else:
            wanted = list(columns)
            for canonical, aliases in COLUMN_ALIASES.items():
                if canonical in wanted or any(alias in wanted for alias in aliases):
                    wanted.append(canonical)
                    wanted.extend(aliases)
        return list(dict.fromkeys(col for col in wanted if col in available))

    def resolve_columns(self, columns: list[str] | None) -> list[str]:
        """Intersect requested columns with available schema (plus aliases for known fields)."""
        return self._resolve_schema_columns(columns, self.available_columns())

    def estimate_matching_rows(self) -> int | None:
        """Fast metadata estimate of rows after model/benchmark filters (may overestimate)."""
        try:
            total = 0
            for path in self.parquet_paths:
                total += int(pq.ParquetFile(path).metadata.num_rows)
            return total
        except Exception:  # noqa: BLE001
            return None

    def identity_index(self, *, refresh: bool = False) -> IdentityIndex:
        """Disk-cached ``item_id -> (path, row_group, local_offset)`` from identity columns only."""
        if self._identity_index is not None and not refresh:
            return self._identity_index
        if not self.parquet_paths:
            empty = IdentityIndex(fingerprint="", paths=[], by_id={}, rows=[])
            self._identity_index = empty
            return empty
        idx = build_identity_index(
            self.parquet_paths,
            make_item_id=self._make_item_id,
            apply_column_aliases=apply_column_aliases,
            cache=True,
        )
        self._identity_index = idx
        return idx

    def _location_matches(self, row: dict[str, Any]) -> bool:
        # The identity index and point reader contain unfiltered dataset rows.
        if (
            self.model_id
            and self.model_id != "*"
            and row.get("model_id")
            and row["model_id"] != self.model_id
        ):
            return False
        return self._bench_ok(row)

    def read_locations(self, locations: list[RowLocation]) -> list[dict[str, Any]]:
        """Point-read rows at indexed locations (only those row groups)."""
        items = read_rows_at_locations(
            locations,
            make_item_id=self._make_item_id,
            apply_column_aliases=apply_column_aliases,
            resolve_columns=lambda schema: self._resolve_schema_columns(None, schema),
        )
        return [row for row in items if self._location_matches(row)]

    def iter_locations(self, locations: Iterable[RowLocation]) -> Iterator[dict[str, Any]]:
        """Stream indexed rows in caller order with the same filters as ``read_locations``.

        File-ordered locations reuse an open reader and forward row-group
        decoding. Unsorted and duplicate requests are supported without caching
        traces. Close the iterator if stopping early to release its active file.
        """
        rows = iter_rows_at_locations(
            locations,
            make_item_id=self._make_item_id,
            apply_column_aliases=apply_column_aliases,
            resolve_columns=lambda schema: self._resolve_schema_columns(None, schema),
        )
        try:
            for row in rows:
                if self._location_matches(row):
                    yield row
        finally:
            rows.close()

    def iter_batches(
        self,
        *,
        batch_size: int = 512,
        max_items: int | None = None,
        columns: list[str] | None = None,
    ):
        """Yield lists of matching item dicts (streaming; does not cache the full corpus).

        Use this to process the entire parquet safely. ``max_items`` stops early
        (capped / probe mode). ``None`` streams until EOF.

        ``columns`` optionally projects to a subset (default: all resolved schema columns).
        ``batch_size`` is an upper bound; large traces produce smaller batches.
        """
        from autofa.data.input_view import (
            attach_compact_export_input,
            has_compact_export_provenance,
        )
        from autofa.data.source_export import require_current_source_cache
        from autofa.util.memory import row_group_bytes_per_row

        batch_size = max(1, int(batch_size))
        cols = self.resolve_columns(columns)
        yielded = 0
        buf: list[dict[str, Any]] = []
        buf_bytes = 0
        seen_ids: set[str] = set()

        def flush():
            nonlocal buf, buf_bytes, yielded
            if not buf:
                return
            out = buf
            buf = []
            buf_bytes = 0
            yielded += len(out)
            return out

        for path in self.parquet_paths:
            if max_items is not None and yielded >= max_items:
                break
            try:
                pf = pq.ParquetFile(path)
            except Exception:  # noqa: BLE001
                continue
            compact_export = has_compact_export_provenance(path)
            # Saved runs stay bound to their cache; one from an earlier adapter is refused.
            require_current_source_cache(path)
            schema_names = set(pf.schema_arrow.names)
            # An explicit projection can be absent from a heterogeneous shard.
            # Arrow's [] reads zero columns; None would load every trace field.
            read_cols = [col for col in cols if col in schema_names]
            if columns is None and not cols:
                read_cols = None
            for rg_idx in range(pf.metadata.num_row_groups):
                row_bytes = row_group_bytes_per_row(pf.metadata.row_group(rg_idx), read_cols)
                arrow_bs = max(1, min(batch_size, DEFAULT_BATCH_SIZE,
                                      MAX_BATCH_BYTES // max(1, row_bytes)))
                if max_items is not None:
                    arrow_bs = min(arrow_bs, max(1, max_items - yielded - len(buf)))
                for arrow_batch in pf.iter_batches(
                    batch_size=arrow_bs, columns=read_cols, row_groups=[rg_idx],
                ):
                    for row in iter_arrow_rows(arrow_batch):
                        if max_items is not None and yielded + len(buf) >= max_items:
                            chunk = flush()
                            if chunk:
                                yield chunk
                            return
                        apply_column_aliases(row)
                        if compact_export:
                            attach_compact_export_input(row)
                        if (
                            self.model_id
                            and self.model_id != "*"
                            and row.get("model_id")
                            and row["model_id"] != self.model_id
                        ):
                            continue
                        if not self._bench_ok(row):
                            continue
                        row["_item_id"] = self._make_item_id(row, path)
                        if columns is None:
                            if row["_item_id"] in seen_ids:
                                raise DuplicateItemIdError(
                                    f"duplicate item ID {row['_item_id']!r} in {path}; "
                                    "provide distinct problem_id/run_idx values"
                                )
                            seen_ids.add(row["_item_id"])
                        row["_source_parquet"] = str(path)
                        row_size = item_memory_bytes(row)
                        if buf and buf_bytes + row_size > MAX_BATCH_BYTES:
                            chunk = flush()
                            yield chunk
                            del chunk
                        buf.append(row)
                        buf_bytes += row_size
                        if len(buf) >= batch_size or buf_bytes >= MAX_BATCH_BYTES:
                            chunk = flush()
                            yield chunk
                            del chunk
                            if max_items is not None and yielded >= max_items:
                                return
                    del arrow_batch
        chunk = flush()
        if chunk:
            yield chunk

    def _make_item_id(
        self,
        row: dict[str, Any],
        path: Path | None = None,
    ) -> str:
        return make_item_id(
            row,
            path,
            include_source=len(self.parquet_paths) > 1,
        )

    def peek_chunk(
        self,
        chunk_index: int,
        *,
        chunk_size: int = 500,
        max_chars: int = 1500,
    ) -> dict[str, Any]:
        """Return one 0-based chunk of truncated items for agent inspection."""
        chunk_index = max(0, int(chunk_index))
        chunk_size = max(1, int(chunk_size))
        target_start = chunk_index * chunk_size
        seen = 0
        items: list[dict[str, Any]] = []
        # Stop after the requested window — no need to scan the rest of the file.
        for batch in self.iter_batches(
            batch_size=chunk_size,
            max_items=target_start + chunk_size,
        ):
            if seen + len(batch) <= target_start:
                seen += len(batch)
                continue
            offset = max(0, target_start - seen)
            window = batch[offset:offset + chunk_size - len(items)]
            items.extend(truncate_item(it, max_chars=max_chars) for it in window)
            seen += len(batch)
            if len(items) >= chunk_size:
                break
        return {
            "ok": True,
            "chunk_index": chunk_index,
            "chunk_size": chunk_size,
            "n": len(items),
            "items": items,
            "exhausted": len(items) < chunk_size,
        }

    def load_items(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Load up to ``limit`` matching rows without retaining them on the dataset.

        .. deprecated::
            Prefer ``iter_batches`` for full scans, ``sample_items`` for uniform
            sampling, and ``get_item`` (identity index) for point reads. This
            materializes the **first** ``limit`` rows in file order — not a
            random sample — and is unsafe as a default for large corpora.
        """
        effective = ABSOLUTE_MAX_LOAD if limit is None else max(1, int(limit))
        items: list[dict[str, Any]] = []
        for batch in self.iter_batches(
            batch_size=min(DEFAULT_BATCH_SIZE, effective), max_items=effective,
        ):
            items.extend(batch)
            if len(items) >= effective:
                break
        return items[:effective]

    def info(self, *, scan_limit: int | None = 50_000) -> dict[str, Any]:
        cap = min(scan_limit or ABSOLUTE_MAX_LOAD, ABSOLUTE_MAX_LOAD)
        n = fails = 0
        models: set[str] = set()
        lengths: dict[str, list[int]] = {col: [] for col in TEXT_FIELDS}
        by_bench: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "fail": 0})
        # Exact percentiles only need integer lengths. Do not retain the trace
        # strings (or call the deprecated full-corpus materialization API).
        for batch in self.iter_batches(batch_size=DEFAULT_BATCH_SIZE, max_items=max(1, cap)):
            for it in batch:
                n += 1
                failed = not _is_correct(it)
                fails += failed
                b = str(it.get("benchmark") or "unknown")
                by_bench[b]["n"] += 1
                by_bench[b]["fail"] += failed
                if it.get("model_id"):
                    models.add(str(it["model_id"]))
                for col in TEXT_FIELDS:
                    value = it.get(col)
                    if isinstance(value, str) and value:
                        lengths[col].append(len(value))

        field_stats: dict[str, dict[str, Any]] = {}
        for col in TEXT_FIELDS:
            lens = sorted(lengths[col])
            nonempty = len(lens)
            field_stats[col] = {
                "nonempty_pct": round(100.0 * nonempty / n, 1) if n else 0.0,
                "median_len": lens[len(lens) // 2] if lens else 0,
                "p95_len": lens[int(0.95 * (len(lens) - 1))] if lens else 0,
            }

        return {
            "n_items": n,
            "n_fail": fails,
            "n_pass": n - fails,
            "fail_rate": round(fails / n, 4) if n else 0.0,
            "models": sorted(models),
            "benchmarks": {
                b: {
                    "n": v["n"],
                    "fail": v["fail"],
                    "fail_rate": round(v["fail"] / v["n"], 4) if v["n"] else 0.0,
                }
                for b, v in sorted(by_bench.items(), key=lambda x: -x[1]["n"])
            },
            "columns": sorted(self.available_columns()),
            "text_field_stats": field_stats,
            "parquet_paths": [str(p) for p in self.parquet_paths],
            "filter_model_id": self.model_id,
            "filter_benchmarks": list(self.benchmarks),
            "scanned_limit": cap,
            "estimated_file_rows": self.estimate_matching_rows(),
        }


def apply_column_aliases(row: dict[str, Any]) -> dict[str, Any]:
    """Fill missing canonical fields from alternate column names (in-place)."""
    for canonical, aliases in COLUMN_ALIASES.items():
        cur = row.get(canonical)
        if cur is not None and not (isinstance(cur, str) and not cur.strip()):
            continue
        for alt in aliases:
            if alt not in row:
                continue
            val = row[alt]
            if val is None:
                continue
            if isinstance(val, str) and not val.strip():
                continue
            row[canonical] = val
            break
    return row


def make_item_id(
    row: dict[str, Any],
    path: Path | None = None,
    *,
    include_source: bool = False,
) -> str:
    parts = [
        str(row.get("model_id") or ""),
        str(row.get("benchmark") or ""),
        str(row.get("problem_id") or row.get("hb_problem_idx") or ""),
        str(row.get("run_idx") if row.get("run_idx") is not None else ""),
    ]
    if include_source and path is not None:
        parts.append(_portable_source_identity(Path(path)))
    raw = "|".join(parts)
    if raw.strip("|"):
        return raw
    blob = repr(sorted((k, str(v)[:80]) for k, v in row.items() if not str(k).startswith("_")))
    digest = hashlib.sha1(blob.encode()).hexdigest()[:12]
    prefix = path.stem if path else "row"
    return f"{prefix}:{digest}"


@lru_cache(maxsize=1024)
def _portable_source_identity(path: Path) -> str:
    """Stable source discriminator without exposing or depending on local paths."""
    resolved = Path(path).expanduser().resolve()
    sidecar = resolved.parent / "source.json"
    from autofa.data.local_trace_store import source_store_root

    if (resolved.name == "dataset.parquet"
            and resolved.parent.parent == source_store_root().expanduser().resolve()):
        from autofa.data.local_import import reference_for_import

        metadata = reference_for_import(resolved)
    else:
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    if isinstance(metadata, dict):
        from autofa.data.reference_data import local_viewer_base

        if local_viewer_base(metadata):
            raw = f"local-evalrdl:{metadata['store_id']}:{metadata['run_id']}".encode()
            return hashlib.sha256(raw).hexdigest()[:16]
        run_id = str(metadata.get("run_id") or "").strip()
        base_url = str(metadata.get("base_url") or "").strip().rstrip("/")
        if run_id and base_url:
            raw = f"data-api:{base_url}:{run_id}".encode()
            return hashlib.sha256(raw).hexdigest()[:16]

    digest = hashlib.sha256()
    digest.update(resolved.name.encode())
    try:
        size = resolved.stat().st_size
        digest.update(str(size).encode())
        with resolved.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if size > 1024 * 1024:
                handle.seek(max(0, size - 1024 * 1024))
                digest.update(handle.read(1024 * 1024))
    except OSError:
        digest.update(resolved.name.encode())
    return digest.hexdigest()[:16]


def _is_correct(item: dict[str, Any]) -> bool:
    return discovery_is_correct(item)


def is_informative(item: dict[str, Any], *, min_response_chars: int = 40) -> bool:
    resp = item.get("response_text") or item.get("generation") or ""
    if not isinstance(resp, str):
        return False
    s = resp.strip().lower()
    if s in PLACEHOLDER_RESPONSES or len(s) < min_response_chars:
        return False
    return True


def annotate_completeness(
    item: dict[str, Any],
    field: str,
    returned: int,
    total: int,
    *,
    truncated: bool | None = None,
) -> None:
    """Attach ``{field}__meta`` describing truncation / how to read the rest."""
    item_id = str(item.get("_item_id") or "")
    is_trunc = bool(truncated) if truncated is not None else returned < total
    item[f"{field}__meta"] = {
        "truncated": is_trunc,
        "returned_chars": int(returned),
        "total_chars": int(total),
        "full_read": f"read_field_chunk(item_id={item_id!r}, field={field!r}, offset=0)",
    }


def _head_tail(val: str, max_chars: int) -> str:
    if max_chars <= 0 or len(val) <= max_chars:
        return val
    head_n = max_chars // 2
    tail_n = max_chars - head_n
    omitted = len(val) - head_n - tail_n
    return (
        f"{val[:head_n]}\n"
        f"…[truncated {omitted} chars]…\n"
        f"{val[-tail_n:]}"
    )


def _trajectory_digest(val: str, *, max_chars: int) -> str | None:
    """Build a labelled digest for agent-trace JSON; None if parse fails."""
    parsed = parse_trajectory(val)
    if not parsed.ok:
        return None
    tool_bits = [f"{t.name}×{t.count}" for t in parsed.tools[:16]]
    tool_hist = ", ".join(tool_bits) if tool_bits else "(none)"
    n_tools = sum(t.count for t in parsed.tools) if parsed.tools else parsed.n_tool_calls
    lines = [
        f"[trajectory digest · {parsed.n_steps} steps · {n_tools} tools]",
    ]
    if parsed.dialect:
        lines.append(f"dialect: {parsed.dialect}")
    if parsed.agent_name:
        lines.append(f"agent: {parsed.agent_name}")
    lines.append(f"tools: {tool_hist}")
    excerpts = [e.strip() for e in (parsed.reasoning_excerpts or []) if e and str(e).strip()]
    if excerpts:
        lines.append("reasoning excerpts:")
        for ex in excerpts[:4]:
            snippet = str(ex).replace("\n", " ").strip()
            if len(snippet) > 180:
                snippet = snippet[:177] + "…"
            lines.append(f"  - {snippet}")
    # Actions tail: last steps with tool names.
    summaries = list(parsed.steps_summary or [])
    tail = summaries[-6:] if summaries else []
    if tail:
        lines.append("actions tail:")
        for s in tail:
            tools = ",".join(getattr(s, "tool_names", None) or []) or "-"
            idx = getattr(s, "index", "?")
            lines.append(f"  step {idx}: tools=[{tools}]")
    digest = "\n".join(lines)
    if len(digest) > max_chars:
        digest = digest[: max(0, max_chars - 20)] + "\n…[digest truncated]"
    return digest


def truncate_item(
    item: dict[str, Any], *, max_chars: int | None = default_value("text_truncate_chars"),
) -> dict[str, Any]:
    """Return a copy with long text fields shortened.

    Agent-trace JSON (``looks_like_agent_json``) becomes a labelled trajectory
    digest instead of a prefix slice. Other long strings use head+tail.
    Each truncated field gets a companion ``{field}__meta`` dict. ``None``
    preserves canonical values for callers that execute detectors on the rows.
    """
    out = dict(item)
    if max_chars is None:
        return out
    for key in TEXT_FIELDS:
        val = out.get(key)
        if not isinstance(val, str) or len(val) <= max_chars:
            continue
        total = len(val)
        if looks_like_agent_json(val):
            digest = _trajectory_digest(val, max_chars=max_chars)
            if digest is not None:
                out[key] = digest
                annotate_completeness(out, key, len(digest), total, truncated=True)
                continue
        shortened = _head_tail(val, max_chars)
        out[key] = shortened
        annotate_completeness(out, key, len(shortened), total, truncated=True)
    return out


def _benchmark_allows(
    item: dict[str, Any],
    *,
    benchmark: str | None = None,
    benchmarks: list[str] | None = None,
    dataset: Dataset | None = None,
) -> bool:
    """True if item's benchmark passes optional single/multi filters + dataset policy."""
    b = str(item.get("benchmark") or "")
    if benchmarks:
        if b not in set(benchmarks):
            return False
    elif benchmark:
        if b != benchmark:
            return False
    if dataset is not None and not dataset._bench_ok(item):
        return False
    return True


def _identity_row_allows(
    row,
    dataset: Dataset,
    *,
    benchmark: str | None = None,
    benchmarks: list[str] | None = None,
) -> bool:
    if (
        dataset.model_id
        and dataset.model_id != "*"
        and row.model_id
        and row.model_id != dataset.model_id
    ):
        return False
    fake = {"benchmark": row.benchmark}
    return _benchmark_allows(
        fake, benchmark=benchmark, benchmarks=benchmarks, dataset=dataset
    )


def _round_robin_groups(items: list, key, *, separate_unknowns: bool = True) -> list:
    """Interleave groups; unknown tasks stay distinct, unlabeled benchmarks pool."""
    groups: dict[Any, list] = defaultdict(list)
    for index, item in enumerate(items):
        value = key(item)
        group = (("known", value) if value not in (None, "") else
                 ("unknown", index if separate_unknowns else None))
        groups[group].append(item)
    ordered = []
    active = [iter(group) for group in groups.values()]
    while active:
        remaining = []
        for group in active:
            item = next(group, None)
            if item is not None:
                ordered.append(item)
                remaining.append(group)
        active = remaining
    return ordered


def _pick_distinct_tasks(
    items: list[dict[str, Any]], n: int, task_map: dict[str, str | None],
    rng: random.Random, prefer_informative: bool, by_benchmark: bool,
) -> list[dict[str, Any]]:
    ordered = list(items)
    rng.shuffle(ordered)
    if prefer_informative:
        ordered.sort(key=lambda item: (
            0 if is_informative(item) else 1,
            -len(str(item.get("response_text") or item.get("generation") or "")),
        ))
    if by_benchmark:
        ordered = _round_robin_groups(ordered, lambda item: item.get("benchmark"),
                                      separate_unknowns=False)
    # Rank a broad pool of distinct tasks before sampling. Taking only the top
    # n would let trace length erase the seed and repeat the same tasks each round.
    pool_size = max(n * 4, n) if prefer_informative else n
    seen, selected = set(), []
    for item in ordered:
        task = task_map.get(str(item.get("_item_id") or ""))
        if task is not None and task in seen:
            continue
        if task is not None:
            seen.add(task)
        selected.append(item)
        if len(selected) >= pool_size:
            break
    if prefer_informative:
        rng.shuffle(selected)
        if by_benchmark:
            selected = _round_robin_groups(selected, lambda item: item.get("benchmark"),
                                           separate_unknowns=False)
    return selected[:n]


def _sample_via_index(  # noqa: C901
    dataset: Dataset,
    *,
    n_fail: int,
    n_pass: int,
    benchmark: str | None,
    benchmarks: list[str] | None,
    prefer_informative: bool,
    by_benchmark: bool,
    min_response_chars: int,
    require_fields: list[str] | None,
    seed: int,
    max_chars: int | None,
    exclude_fail_item_ids: set[str] | None = None,
    task_by_item_id: dict[str, str | None] | None = None,
    distinct_tasks: bool = False,
) -> dict[str, Any] | None:
    """Uniform sample over the full corpus via identity index + row-group reads."""
    if not dataset.parquet_paths:
        return None
    try:
        index = dataset.identity_index()
    except DuplicateItemIdError:
        raise
    except Exception:  # noqa: BLE001
        return None
    if not index.rows:
        return None

    fail_locs: list[RowLocation] = []
    pass_locs: list[RowLocation] = []
    task_by_location: dict[RowLocation, str | None] = {}
    benchmark_by_location: dict[RowLocation, str | None] = {}
    for row in index.rows:
        if not _identity_row_allows(
            row, dataset, benchmark=benchmark, benchmarks=benchmarks
        ):
            continue
        loc = RowLocation(
            path=Path(row.path),
            row_group=row.row_group,
            local_offset=row.local_offset,
        )
        if by_benchmark:
            benchmark_by_location[loc] = row.benchmark
        if distinct_tasks:
            task_by_location[loc] = (task_by_item_id or {}).get(row.item_id)
        # Unknown label treated as fail for sampling balance (matches _is_correct=False).
        if row.is_correct is True:
            pass_locs.append(loc)
        elif not exclude_fail_item_ids or row.item_id not in exclude_fail_item_ids:
            fail_locs.append(loc)

    rng = random.Random(seed)

    def draw(locs: list[RowLocation], n: int) -> list[RowLocation]:
        if not locs or n <= 0:
            return []
        # Oversample when post-filters / informative ranking need a larger pool.
        want = n
        if prefer_informative or min_response_chars or require_fields or by_benchmark:
            want = min(len(locs), max(n * 8, n))
        if distinct_tasks or by_benchmark:
            ordered = list(locs)
            rng.shuffle(ordered)
            # Visit tasks before repeat attempts. The larger pool leaves room
            # for response/field filters without letting common tasks crowd out
            # rarer tasks before trajectories are loaded.
            if distinct_tasks:
                ordered = _round_robin_groups(ordered, task_by_location.get)
            # Balance before the bounded row read, not only after it: a rare
            # benchmark absent from the candidate pool cannot be recovered later.
            if by_benchmark:
                ordered = _round_robin_groups(ordered, benchmark_by_location.get,
                                              separate_unknowns=False)
            return ordered[:want]
        if want >= len(locs):
            chosen = list(locs)
            rng.shuffle(chosen)
            return chosen
        return rng.sample(locs, want)

    cand_locs = draw(fail_locs, n_fail) + draw(pass_locs, n_pass)
    if not cand_locs:
        return {"n_fail": 0, "n_pass": 0, "prefer_informative": prefer_informative, "items": []}

    items = dataset.read_locations(cand_locs)

    def ok(it: dict[str, Any]) -> bool:
        if not _benchmark_allows(
            it, benchmark=benchmark, benchmarks=benchmarks, dataset=dataset
        ):
            return False
        resp = it.get("response_text") or it.get("generation") or ""
        if min_response_chars and (not isinstance(resp, str) or len(resp) < min_response_chars):
            return False
        if require_fields:
            for f in require_fields:
                v = it.get(f)
                if v is None or (isinstance(v, str) and not v.strip()):
                    return False
        return True

    pool = [it for it in items if ok(it)]
    fails = [it for it in pool if not _is_correct(it)]
    passes = [it for it in pool if _is_correct(it)]

    def pick(group: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        if not group or n <= 0:
            return []
        if distinct_tasks:
            return _pick_distinct_tasks(group, n, task_by_item_id or {}, rng,
                                        prefer_informative, by_benchmark)
        if prefer_informative:
            ranked = sorted(
                group,
                key=lambda it: (
                    0 if is_informative(it) else 1,
                    -len(str(it.get("response_text") or it.get("generation") or "")),
                ),
            )
            top = ranked[: max(n * 4, n)]
            rng.shuffle(top)
            out = top[:n]
            if by_benchmark:
                by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for it in ranked:
                    by_b[str(it.get("benchmark") or "unknown")].append(it)
                out = []
                keys = list(by_b.keys())
                rng.shuffle(keys)
                i = 0
                while len(out) < n and keys:
                    k = keys[i % len(keys)]
                    if by_b[k]:
                        out.append(by_b[k].pop(0))
                    else:
                        keys.remove(k)
                        if not keys:
                            break
                        i = 0
                        continue
                    i += 1
                return out[:n]
            return out
        ordered = list(group)
        rng.shuffle(ordered)
        if by_benchmark:
            by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for it in ordered:
                by_b[str(it.get("benchmark") or "unknown")].append(it)
            out = []
            keys = list(by_b.keys())
            i = 0
            while len(out) < n and keys:
                k = keys[i % len(keys)]
                if by_b[k]:
                    out.append(by_b[k].pop(0))
                else:
                    keys.remove(k)
                    if not keys:
                        break
                    i = 0
                    continue
                i += 1
            return out[:n]
        return ordered[:n]

    chosen_fail = pick(fails, n_fail)
    chosen_pass = pick(passes, n_pass)
    chosen = chosen_fail + chosen_pass
    return {
        "n_fail": len(chosen_fail),
        "n_pass": len(chosen_pass),
        "prefer_informative": prefer_informative,
        "items": [truncate_item(it, max_chars=max_chars) for it in chosen],
    }


def sample_items(  # noqa: C901
    dataset: Dataset,
    *,
    n_fail: int | None = None,
    n_pass: int | None = None,
    benchmark: str | None = None,
    benchmarks: list[str] | None = None,
    prefer_informative: bool | None = None,
    by_benchmark: bool = False,
    min_response_chars: int = 0,
    require_fields: list[str] | None = None,
    seed: int = 0,
    max_chars: int | None = default_value("text_truncate_chars"),
    max_total: int | None = None,
    exclude_fail_item_ids: set[str] | None = None,
    task_by_item_id: dict[str, str | None] | None = None,
    distinct_tasks: bool = False,
) -> dict[str, Any]:
    if n_fail is None:
        n_fail = default_value("sample_fail_default")
    if n_pass is None:
        n_pass = default_value("sample_pass_default")
    if prefer_informative is None:
        prefer_informative = default_value("prefer_informative")
    if max_total is None:
        max_total = default_value("max_sample_per_call")
    n_fail = min(n_fail, max_total)
    n_pass = min(n_pass, max_total - n_fail)

    indexed = _sample_via_index(
        dataset,
        n_fail=n_fail,
        n_pass=n_pass,
        benchmark=benchmark,
        benchmarks=benchmarks,
        prefer_informative=prefer_informative,
        by_benchmark=by_benchmark,
        min_response_chars=min_response_chars,
        require_fields=require_fields,
        seed=seed,
        max_chars=max_chars,
        exclude_fail_item_ids=exclude_fail_item_ids,
        task_by_item_id=task_by_item_id,
        distinct_tasks=distinct_tasks,
    )
    if indexed is not None:
        return indexed

    # Fallback for FakeDataset / empty paths: first-N pool via load_items.
    pool_limit = min(ABSOLUTE_MAX_LOAD, max(2_000, max_total * 200))
    items = dataset.load_items(limit=pool_limit)
    rng = random.Random(seed)

    def ok(it: dict[str, Any]) -> bool:
        if not _benchmark_allows(
            it, benchmark=benchmark, benchmarks=benchmarks, dataset=dataset
        ):
            return False
        resp = it.get("response_text") or it.get("generation") or ""
        if min_response_chars and (not isinstance(resp, str) or len(resp) < min_response_chars):
            return False
        if require_fields:
            for f in require_fields:
                v = it.get(f)
                if v is None or (isinstance(v, str) and not v.strip()):
                    return False
        return True

    pool = [it for it in items if ok(it)]
    fails = [it for it in pool if not _is_correct(it)
             and (not exclude_fail_item_ids
                  or str(it.get("_item_id") or "") not in exclude_fail_item_ids)]
    passes = [it for it in pool if _is_correct(it)]

    def pick(group: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        if not group or n <= 0:
            return []
        if distinct_tasks:
            return _pick_distinct_tasks(group, n, task_by_item_id or {}, rng,
                                        prefer_informative, by_benchmark)
        if prefer_informative:
            ranked = sorted(
                group,
                key=lambda it: (
                    0 if is_informative(it) else 1,
                    -len(str(it.get("response_text") or it.get("generation") or "")),
                ),
            )
            top = ranked[: max(n * 4, n)]
            rng.shuffle(top)
            out = top[:n]
            if by_benchmark:
                by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for it in ranked:
                    by_b[str(it.get("benchmark") or "unknown")].append(it)
                out = []
                keys = list(by_b.keys())
                rng.shuffle(keys)
                i = 0
                while len(out) < n and keys:
                    k = keys[i % len(keys)]
                    if by_b[k]:
                        out.append(by_b[k].pop(0))
                    else:
                        keys.remove(k)
                        if not keys:
                            break
                        i = 0
                        continue
                    i += 1
                return out[:n]
            return out
        ordered = list(group)
        rng.shuffle(ordered)
        if by_benchmark:
            by_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for it in ordered:
                by_b[str(it.get("benchmark") or "unknown")].append(it)
            out = []
            keys = list(by_b.keys())
            i = 0
            while len(out) < n and keys:
                k = keys[i % len(keys)]
                if by_b[k]:
                    out.append(by_b[k].pop(0))
                else:
                    keys.remove(k)
                    if not keys:
                        break
                    i = 0
                    continue
                i += 1
            return out[:n]
        return ordered[:n]

    chosen_fail = pick(fails, n_fail)
    chosen_pass = pick(passes, n_pass)
    chosen = chosen_fail + chosen_pass
    return {
        "n_fail": len(chosen_fail),
        "n_pass": len(chosen_pass),
        "prefer_informative": prefer_informative,
        "items": [truncate_item(it, max_chars=max_chars) for it in chosen],
    }


def search_items(
    dataset: Dataset,
    *,
    is_fail: bool | None = None,
    benchmark: str | None = None,
    benchmarks: list[str] | None = None,
    query: str | None = None,
    strata_difficulty: str | None = None,
    limit: int = 20,
    offset: int = 0,
    max_chars: int = default_value("text_truncate_chars"),
) -> dict[str, Any]:
    """Stream the corpus for matches; supports ``offset`` pagination + ``limit``.

    Does not materialize a fixed prefix into memory.
    """
    out: list[dict[str, Any]] = []
    q = (query or "").lower()
    skip = max(0, int(offset))
    limit = max(0, int(limit))
    matched = 0
    search_cols = list(IDENTITY_FIELDS) + [
        "system_text",
        "prompt_text",
        "response_text",
        "reasoning_text",
        "judge_text",
        "question",
        "generation",
    ]
    for aliases in COLUMN_ALIASES.values():
        search_cols.extend(aliases)

    for batch in dataset.iter_batches(batch_size=DEFAULT_BATCH_SIZE, columns=search_cols):
        for it in batch:
            if is_fail is not None and (not _is_correct(it)) != is_fail:
                continue
            if not _benchmark_allows(
                it, benchmark=benchmark, benchmarks=benchmarks, dataset=dataset
            ):
                continue
            if strata_difficulty and it.get("strata_difficulty") != strata_difficulty:
                continue
            if q:
                blob = " ".join(
                    str(it.get(f) or "")
                    for f in (
                        "system_text",
                        "prompt_text",
                        "response_text",
                        "reasoning_text",
                        "judge_text",
                        "question",
                    )
                ).lower()
                if q not in blob:
                    continue
            if matched < skip:
                matched += 1
                continue
            matched += 1
            out.append(truncate_item(it, max_chars=max_chars))
            if len(out) >= limit:
                return {"n": len(out), "offset": skip, "items": out}
    return {"n": len(out), "offset": skip, "items": out}


def get_item(
    dataset: Dataset,
    item_id: str,
    *,
    max_chars: int | None = 8000,
    include_trace: bool = False,
) -> dict[str, Any]:
    """Fetch one item; max_chars=None preserves canonical values for bounded readers."""
    target = str(item_id)

    def prepare(item: dict[str, Any]) -> dict[str, Any]:
        out = dict(item) if max_chars is None else truncate_item(item, max_chars=max_chars)
        if not include_trace:
            out.pop("trace_json", None)
            out.pop("raw_row_json", None)
        return out

    if dataset.parquet_paths:
        try:
            index = dataset.identity_index()
            loc = index.by_id.get(target)
            if loc is not None:
                items = dataset.read_locations([loc])
                if items:
                    return {"ok": True, "item": prepare(items[0])}
        except DuplicateItemIdError:
            raise
        except Exception:  # noqa: BLE001
            pass

        # Index miss / stale: stream full corpus (no ABSOLUTE_MAX_LOAD cap).
        for batch in dataset.iter_batches(batch_size=DEFAULT_BATCH_SIZE):
            for it in batch:
                if str(it.get("_item_id")) == target:
                    return {"ok": True, "item": prepare(it)}
        return {"ok": False, "error": f"item not found: {item_id}"}

    # FakeDataset / no parquet: legacy load_items path.
    for it in dataset.load_items(limit=ABSOLUTE_MAX_LOAD):
        if str(it.get("_item_id")) == target:
            return {"ok": True, "item": prepare(it)}
    return {"ok": False, "error": f"item not found: {item_id}"}


def infer_models_benchmarks(paths: list[Path], *, max_rows: int | None = None) -> dict[str, Any]:
    """Scan parquet files for distinct model_id / benchmark values.

    Uses column projection only (not full text) so large files stay workable —
    benchmarks that appear late in the file are not missed when max_rows is None.
    """
    from collections import Counter

    models: Counter[str] = Counter()
    benches: Counter[str] = Counter()
    n_rows = 0
    n_fail = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        schema_names = [f.name for f in pf.schema_arrow]
        # Include alias columns so remapped labels still count.
        label_cols = [
            c
            for c in (
                "is_correct",
                "resolved",
                "tb_is_resolved",
                "is_resolved",
                "accuracy",
            )
            if c in schema_names
        ]
        cols = [c for c in ("model_id", "benchmark", *label_cols) if c in schema_names]
        if not cols:
            # Still count rows for fail-rate warning when identity cols absent
            try:
                n_rows += int(pf.metadata.num_rows)
            except Exception:  # noqa: BLE001
                pass
            continue
        seen = 0
        for batch in pf.iter_batches(batch_size=65_536, columns=cols):
            names = batch.schema.names
            cols_data = {name: batch.column(i).to_pylist() for i, name in enumerate(names)}
            for i in range(batch.num_rows):
                row = {name: cols_data[name][i] for name in names}
                apply_column_aliases(row)
                mid = row.get("model_id")
                if mid is not None and str(mid):
                    models[str(mid)] += 1
                b = row.get("benchmark")
                if b is not None and str(b):
                    benches[str(b)] += 1
                n_rows += 1
                if not _is_correct(row):
                    n_fail += 1
            seen += batch.num_rows
            if max_rows is not None and seen >= max_rows:
                break
    return {
        "models": models.most_common(),
        "benchmarks": benches.most_common(),
        "n_rows": n_rows,
        "n_fail": n_fail,
    }
