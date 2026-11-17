# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound and checkpoint diagnostic work independently of scientific scores.

Small smoke controls may use 30 LLM rows or 500 Python rows. Extra discovery
rounds each receive a fresh diagnostic allowance; exhausting it directs a pivot
or round completion, never discovery termination. These are work limits, not evidence.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from autofa.metrics.detector_checkpoints import DETECTOR_EXECUTION_VERSION
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import cancellation_guard, check_cancelled

LLM_ROWS = 30
PYTHON_ROWS = 500
CALLS_PER_PANEL = LLM_ROWS * 2
_LOCK = threading.RLock()
_ITEM_LOCKS: dict[str, threading.Lock] = {}


class SmokeBudgetExceeded(BaseException):
    """Do not let detector Exception handlers turn an exhausted budget into a label."""


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, default=_json_default, ensure_ascii=False,
    ).encode()).hexdigest()


def dataset_benchmark_slot(dataset: Any) -> str | None:
    """The sole selected benchmark, else None.

    Saved evaluation, diagnostic and smoke identities include this slot, so it
    keeps its value for an unchanged run selection.
    """
    benchmarks = getattr(dataset, "benchmarks", None)
    return benchmarks[0] if benchmarks and len(benchmarks) == 1 else None


def _directory(rt: Any) -> Path | None:
    run = getattr(rt, "run", None)
    return Path(run.run_dir()) if run is not None else None


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid diagnostic work checkpoint: {path}")
    return value


def _epoch(directory: Path | None, *, round_idx: int | None = None) -> str:
    state = _read(directory / "discovery_stall.json") if directory else {}
    generation = state.get("quality_generation", 0)
    return digest(generation if round_idx is None else [generation, "recovery_round", round_idx])


def _state(rt: Any) -> tuple[Path | None, dict]:
    directory = _directory(rt)
    path = directory / "smoke_work.json" if directory else None
    run = getattr(rt, "run", None)
    round_idx = getattr(run, "round", None)
    planned = getattr(run, "max_rounds", None)
    recovery_round = (round_idx if isinstance(round_idx, int) and isinstance(planned, int)
                      and round_idx >= planned else None)
    epoch = _epoch(directory, round_idx=recovery_round)
    state = _read(path) if path else getattr(rt, "_smoke_work", {})
    if state.get("epoch") != epoch:
        state = {"version": 1, "epoch": epoch, "rows": 0, "model_calls": 0}
    return path, state


def _save(rt: Any, path: Path | None, state: dict) -> None:
    if path:
        atomic_write_json(path, state)
    else:
        rt._smoke_work = state


def diagnostic_budget(rt: Any) -> dict:
    """Read work usage, renewing ordinary discovery at each extra round."""
    from autofa.session.discovery_stall import stall_termination_allowed

    with _LOCK:
        _, state = _state(rt)
        patience = max(1, int(getattr(rt.config, "discovery_plateau_patience", 6)))
        rows_limit, calls_limit = PYTHON_ROWS * patience, CALLS_PER_PANEL * patience
        allowance_exhausted = state["rows"] >= rows_limit or state["model_calls"] >= calls_limit
        enforced = stall_termination_allowed(rt)
        return {
            "rows": state["rows"], "row_limit": rows_limit,
            "model_calls": state["model_calls"], "model_call_limit": calls_limit,
            "allowance_exhausted": allowance_exhausted,
            "enforced": enforced,
            "exhausted": allowance_exhausted and enforced,
            "scope": "discovery_round",
            "accounting": "SDK dispatches, including AutoFA retries; SDK transport retries "
                          "are managed inside the configured client.",
        }


def _model_identity(client: Any) -> dict:
    underlying = getattr(client, "_client", None)
    return {
        "adapter": f"{type(client).__module__}.{type(client).__qualname__}",
        "model": getattr(client, "model", None), "api": getattr(client, "api", None),
        "endpoint": str(getattr(underlying, "base_url", "")),
        # Do not persist credentials. A different account must not reuse its results.
        "account": digest(str(getattr(underlying, "api_key", ""))),
        "options": getattr(client, "request_options", {}),
        "temperature": getattr(client, "temperature", None),
        "max_tokens": getattr(client, "max_tokens", None),
    }


class SmokeWork:
    def __init__(self, rt: Any, *, source: str, bundle: dict | None,
                 policy: Any, client: Any, uses_llm: bool, deps: dict | None = None) -> None:
        self.rt = rt
        self.directory = _directory(rt)
        self.uses_llm = uses_llm
        self.calls = 0
        self.reused = 0
        dataset = rt.dataset
        files = []
        for raw in getattr(dataset, "parquet_paths", []):
            path = Path(raw)
            try:
                stat = path.stat()
                files.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
            except FileNotFoundError:
                files.append([str(path), None])
        self.identity = digest({
            "version": 1, "source": source, "bundle": bundle, "policy": policy,
            **({"deps": deps} if deps else {}),
            "detector_execution_version": DETECTOR_EXECUTION_VERSION,
            "model": _model_identity(client),
            "dataset": [files, getattr(dataset, "model_id", None),
                        dataset_benchmark_slot(dataset),
                        getattr(dataset, "benchmarks", None)],
        })
        self._memory = getattr(rt, "_smoke_results", {})
        rt._smoke_results = self._memory

    def known_item_ids(self) -> set[str]:
        """Discover completed controls without trusting author expectations."""
        path = (self.directory / "smoke_results" / f"index-{self.identity}.json"
                if self.directory else None)
        with _LOCK:
            index = _read(path) if path else self._memory.get(f"index-{self.identity}", {})
        return set(index)

    def cached_result(self, item: dict) -> dict | None:
        """Read only compatible completed predictions; never spend diagnostic budget."""
        key = digest([self.identity, item])
        location = self.directory / "smoke_results" / f"{key}.json" if self.directory else None
        saved = _read(location) if location else self._memory.get(key, {})
        cached, receipts = saved.get("result") or {}, saved.get("artifact_reads")
        if cached.get("status") not in {"value", "abstained"} or not isinstance(receipts, list):
            return None
        from autofa.data.artifact_access import validate_artifact_reads
        from autofa.data.trace_fields import derive_trace_fields

        if receipts and not validate_artifact_reads(derive_trace_fields(item), receipts):
            return None
        return cached

    def artifact_read_receipts(self, item: dict) -> list[dict]:
        """Carry actual pinned artifact reads into a completed diagnostic receipt."""
        key = digest([self.identity, item])
        location = (self.directory / "smoke_results" / f"{key}.json"
                    if self.directory else None)
        saved = _read(location) if location else self._memory.get(key, {})
        return saved.get("artifact_reads") or []

    def _index_result(self, item: dict, key: str) -> None:
        # Index identifiers only. Retrieval still checks current execution,
        # source item, and artifact receipts; expectations never enter the index.
        with _LOCK, cancellation_guard():
            index_key = f"index-{self.identity}"
            path = (self.directory / "smoke_results" / f"{index_key}.json"
                    if self.directory else None)
            index = _read(path) if path else self._memory.get(index_key, {})
            index = {**index, str(item["_item_id"]): key}
            if path:
                atomic_write_json(path, index)
            else:
                self._memory[index_key] = index

    def reserve_model_call(self) -> None:
        with _LOCK, cancellation_guard():
            path, state = _state(self.rt)
            budget = diagnostic_budget(self.rt)
            if self.calls >= CALLS_PER_PANEL:
                raise SmokeBudgetExceeded(
                    f"A diagnostic panel allows at most {CALLS_PER_PANEL} model calls. "
                    "Simplify the control or submit the detector for capped evaluation."
                )
            # This item's row was already admitted. Using the final available
            # row must not forbid the model calls needed to finish that row.
            if budget["enforced"] and budget["model_calls"] >= budget["model_call_limit"]:
                raise SmokeBudgetExceeded(
                    "Diagnostic work allowance is exhausted for this recovery round. "
                    "Use saved evidence to complete or defer the current family, then finish "
                    "the round's validation and refinement. This does not terminate discovery; "
                    "ordinary discovery receives a fresh allowance in its next round. "
                )
            # A request that fails or is interrupted still consumed work. Persist first.
            self.calls += 1
            state["model_calls"] += 1
            _save(self.rt, path, state)

    @contextmanager
    def item(self, item: dict):
        # Full source evidence, rather than just its ID, invalidates changed rows.
        key = digest([self.identity, item])
        location = (self.directory / "smoke_results" / f"{key}.json"
                    if self.directory else None)
        lock_key = str(location) if location else f"{id(self.rt)}:{key}"
        with _LOCK:
            lock = _ITEM_LOCKS.setdefault(lock_key, threading.Lock())
        while not lock.acquire(timeout=.1):
            check_cancelled()
        try:
            check_cancelled()
            saved = _read(location) if location else self._memory.get(key, {})
            cached = saved.get("result") or {}
            receipts = saved.get("artifact_reads")
            fresh = False
            if cached.get("status") in {"value", "abstained"} and isinstance(receipts, list):
                from autofa.data.artifact_access import validate_artifact_reads
                from autofa.data.trace_fields import derive_trace_fields

                fresh = not receipts or validate_artifact_reads(derive_trace_fields(item), receipts)
            if fresh:
                with _LOCK:
                    self.reused += 1
                self._index_result(item, key)
                yield cached, lambda result, artifacts: None
                return
            with _LOCK, cancellation_guard():
                path, state = _state(self.rt)
                if diagnostic_budget(self.rt)["exhausted"]:
                    raise SmokeBudgetExceeded(
                        "Diagnostic work allowance is exhausted for this recovery round. "
                        "Use saved evidence to complete or defer the current family, then finish "
                        "the round's validation and refinement. This does not terminate discovery; "
                        "ordinary discovery receives a fresh allowance in its next round. "
                    )
                state["rows"] += 1
                _save(self.rt, path, state)

            def complete(result: dict, artifact_reads: list[dict]) -> None:
                # Errors remain retryable and cannot become cached negative labels.
                if result["status"] not in {"value", "abstained"}:
                    return
                try:
                    restored = json.loads(json.dumps(result, allow_nan=False))
                except (ValueError, TypeError):
                    return
                if restored != result:
                    # Arbitrary diagnostic values need not have lossless JSON
                    # representations (e.g. tuple labels). Preserve their result,
                    # but do not reuse a value whose type/content would change.
                    return
                # Reservations and indexing acquire _LOCK before the shared
                # cancellation token. Keep the complete checkpoint transaction
                # in that order too: another worker may be reserving its row or
                # model call while this one finishes.
                with _LOCK, cancellation_guard():
                    saved = {"result": restored, "artifact_reads": artifact_reads}
                    if location:
                        atomic_write_json(location, saved)
                    else:
                        self._memory[key] = saved
                    self._index_result(item, key)

            yield None, complete
        finally:
            lock.release()
