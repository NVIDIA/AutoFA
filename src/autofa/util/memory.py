# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host memory probe → streaming chunk-size sweet spot."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autofa.defaults import setting_value

# Nice steps so logs / config stay readable.
_CHUNK_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)

# Conservative default when we cannot size from parquet (Python dict + text fields).
_DEFAULT_BYTES_PER_ROW = 96 * 1024

# Leave headroom for Claude Code / SDK / OS; never spend more than this fraction
# of *available* RAM on one streaming window.
_USABLE_FRACTION = 0.18
_RESERVE_BYTES = 2 * 1024**3  # 2 GiB absolute reserve when possible
# Multiple windows/processes can coexist; abundant host RAM is not a reason to
# materialize gigabytes of traces in a single batch.
_MAX_WINDOW_BYTES = 256 * 1024**2
_UNKNOWN_WINDOW_BYTES = 32 * 1024**2
_ROW_EXPANSION = 4  # Arrow buffers plus Python values/normalization.
_MIN_ROW_BYTES = 16 * 1024

_AUTO_ATTR = "_eval_chunk_auto"


@dataclass(frozen=True)
class MemoryInfo:
    available_bytes: int | None
    total_bytes: int | None
    source: str  # proc | vm_stat | sysconf | unknown


@dataclass(frozen=True)
class ChunkTune:
    eval_chunk_size: int
    available_bytes: int | None
    total_bytes: int | None
    bytes_per_row_est: int
    source: str  # auto | config | fallback
    note: str

    def summary(self) -> str:
        mem = (
            f"{_fmt_gib(self.available_bytes)} available"
            if self.available_bytes is not None
            else "unknown RAM"
        )
        return (
            f"mem: {mem} → eval_chunk_size={self.eval_chunk_size} "
            f"({self.source}; ~{_fmt_kib(self.bytes_per_row_est)}/row est)"
        )


def read_memory() -> MemoryInfo:
    """Best-effort available/total RAM (Linux /proc first)."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        vals: dict[str, int] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if not parts:
                continue
            try:
                kb = int(parts[0])
            except ValueError:
                continue
            vals[key] = kb * 1024
        avail = vals.get("MemAvailable", vals.get("MemFree"))
        total = vals.get("MemTotal")
        if avail is not None or total is not None:
            return MemoryInfo(avail, total, "proc")
    except OSError:
        pass

    if sys.platform == "darwin":
        return _read_darwin_memory()

    try:
        import os

        page = int(os.sysconf("SC_PAGE_SIZE"))
        phys = int(os.sysconf("SC_PHYS_PAGES"))
        total = page * phys
        try:
            free_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            available = page * free_pages if free_pages >= 0 else None
        except (ValueError, OSError, AttributeError):
            available = None
        return MemoryInfo(available, total if total > 0 else None, "sysconf")
    except (ValueError, OSError, AttributeError):
        return MemoryInfo(None, None, "unknown")


def _read_darwin_memory() -> MemoryInfo:
    """Conservative reclaimable RAM; never infer availability from total RAM.

    Inactive anonymous memory can require compression or swapping. Only count
    inactive pages up to the file-backed count, alongside free pages. Purgeable
    and speculative counters overlap other categories and are not added again.
    This is an estimate, not macOS's memory-pressure calculation.
    """
    total = None
    available = None
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        value = int(result.stdout.strip())
        total = value if value > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        result = subprocess.run(
            ["/usr/bin/vm_stat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        page_match = re.search(r"page size of (\d+) bytes", result.stdout)
        counts = {
            key.strip(): int(count)
            for key, count in re.findall(r"^([^:\n]+):\s+(\d+)\.?\s*$", result.stdout, re.M)
        }
        if page_match and "Pages free" in counts:
            page_size = int(page_match.group(1))
            if page_size > 0:
                reclaimable = min(
                    counts.get("Pages inactive", 0), counts.get("File-backed pages", 0)
                )
                available = page_size * (counts["Pages free"] + reclaimable)
                if total is not None:
                    available = min(available, total)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return MemoryInfo(available, total, "vm_stat" if available is not None else "unknown")


def row_group_bytes_per_row(row_group: Any, columns: list[str] | None = None) -> int:
    """Estimate materialized row cost from uncompressed selected-column metadata.

    This is a sizing heuristic, not an upper bound: dictionary encoding and
    unusually large individual rows can exceed the average. Readers must also
    bound their buffers by actual bytes. No trace data is decoded here.
    """
    rows = int(row_group.num_rows)
    if rows <= 0:
        return _MIN_ROW_BYTES
    wanted = set(columns) if columns is not None else None
    uncompressed = 0
    for index in range(int(row_group.num_columns)):
        column = row_group.column(index)
        column_path = column.path_in_schema
        if (
            wanted is not None
            and column_path not in wanted
            and column_path.split(".", 1)[0] not in wanted
        ):
            continue
        uncompressed += max(0, int(column.total_uncompressed_size))
    return max(_MIN_ROW_BYTES, math.ceil(uncompressed * _ROW_EXPANSION / rows))


def estimate_bytes_per_row(parquet_paths: list[Path | str] | None) -> int:
    """Size for the heaviest row-group mean, using metadata instead of compression.

    A corpus-wide mean hides large traces in a later shard or row group. Inspect
    every file's footer and keep the maximum; do not cap large estimates.
    """
    if not parquet_paths:
        return _DEFAULT_BYTES_PER_ROW
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return _DEFAULT_BYTES_PER_ROW

    estimate = 0
    for raw in parquet_paths:
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            with pq.ParquetFile(path) as parquet:
                metadata = parquet.metadata
                for index in range(metadata.num_row_groups):
                    group = metadata.row_group(index)
                    if group.num_rows > 0:
                        estimate = max(estimate, row_group_bytes_per_row(group))
        except Exception:  # noqa: BLE001
            continue
    return estimate or _DEFAULT_BYTES_PER_ROW


def suggest_chunk_size(
    *,
    available_bytes: int | None,
    bytes_per_row: int = _DEFAULT_BYTES_PER_ROW,
) -> int:
    """Pick a streaming chunk size from free RAM and per-row cost."""
    bpr = max(8 * 1024, int(bytes_per_row))
    if available_bytes is None:
        budget = _UNKNOWN_WINDOW_BYTES
    else:
        available = max(0, int(available_bytes))
        reserve = min(_RESERVE_BYTES, available // 4)
        budget = min(_MAX_WINDOW_BYTES, int((available - reserve) * _USABLE_FRACTION))
    # A single row is the irreducible minimum even when it exceeds the budget.
    raw = max(1, budget // bpr)
    return _snap(raw)


def apply_memory_tuning(
    cfg: Any,
    *,
    parquet_paths: list[Path | str] | None = None,
) -> ChunkTune:
    """Set ``cfg.eval_chunk_size`` from free RAM unless the user pinned a value.

    Config convention: ``eval_chunk_size <= 0`` means auto (default). Once auto-tuned,
    later calls may recompute with a better per-row estimate from parquet paths.
    """
    mem = read_memory()
    bpr = estimate_bytes_per_row(parquet_paths)
    configured = int(setting_value(cfg, "eval_chunk_size") or 0)

    if not hasattr(cfg, _AUTO_ATTR):
        # First touch: toml/pin >0 stays pinned; 0 stays auto forever on this object.
        setattr(cfg, _AUTO_ATTR, configured <= 0)

    auto = bool(getattr(cfg, _AUTO_ATTR))
    if not auto:
        size = max(1, min(configured if configured > 0 else 512, 8192))
        cfg.eval_chunk_size = size
        return ChunkTune(
            eval_chunk_size=size,
            available_bytes=mem.available_bytes,
            total_bytes=mem.total_bytes,
            bytes_per_row_est=bpr,
            source="config",
            note="pinned in config.toml",
        )

    size = suggest_chunk_size(available_bytes=mem.available_bytes, bytes_per_row=bpr)
    cfg.eval_chunk_size = size
    source = "auto" if mem.available_bytes is not None else "fallback"
    note = (
        f"auto from {mem.source} available-memory estimate"
        if source == "auto"
        else "RAM unknown — conservative default"
    )
    if parquet_paths:
        note += " · sized with parquet row estimate"
    return ChunkTune(
        eval_chunk_size=size,
        available_bytes=mem.available_bytes,
        total_bytes=mem.total_bytes,
        bytes_per_row_est=bpr,
        source=source,
        note=note,
    )


def _snap(n: int) -> int:
    """Round down so the chosen step does not exceed the computed budget."""
    n = max(1, min(int(n), 4096))
    chosen = 1
    for step in _CHUNK_STEPS:
        if step > n:
            break
        chosen = step
    return chosen


def _fmt_gib(n: int | None) -> str:
    if n is None:
        return "?"
    return f"{n / (1024**3):.1f} GiB"


def _fmt_kib(n: int) -> str:
    return f"{n / 1024:.0f} KiB"
