# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a cohort of rollouts and prepare its analysis-ready datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from autofa.config import autofa_home, load_config
from autofa.data.outcomes import is_healthy_scored, is_scored_failure, population_outcome
from autofa.data.reference_data import (
    canonical_export_url,
    make_local_reference_data,
    make_reference_data,
    remote_reference_details,
)
from autofa.data.remote_api import (
    DataApiClient,
    DataApiError,
    MissingSelectedRolloutsError,
    materialize_remote_run,
    stage_export_subset,
)
from autofa.data.source_export import materialize_source_traces
from autofa.util.atomic import atomic_write_json
from evalrdl_schema import CURRENT

COHORT_VERSION = 1
PAGE_SIZE = 1000
MAX_RUNS = 100
SAMPLES = ("first", "random")
_COLUMNS = tuple(column.name for column in CURRENT.table("rollouts").columns)
_SETTINGS = ("runs_where", "where", "latest", "per_run", "sample", "seed")
_HEALTHY_SCORED = (
    "status = 'ok' AND ingest_status = 'ok' AND error IS NULL AND is_scored = true"
)
_FAILURES = (
    f"{_HEALTHY_SCORED} AND reward <= 0"
)
# ClickHouse rejecting the column itself: "Unknown expression identifier `run_uuid`",
# "Missing columns: 'run_uuid'". Other errors may quote the SQL, which names run_uuid too.
_NO_RUN_UUID_COLUMN = re.compile(r"(?:identifier|columns?)\W{1,4}run_uuid\b", re.IGNORECASE)
_REWARD = re.compile(
    r"^\s*(<=|>=|==|=|<|>)\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


@dataclass
class Cohort:
    runs_where: str
    where: str
    sources: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    latest: int = 0
    per_run: int = 0
    sample: str = "first"
    seed: int = 0
    datasets: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    parent_selection: str = ""
    taxonomy_filter: dict[str, Any] = field(default_factory=dict)
    limit: int = 0

    @property
    def id(self) -> str:
        sources = []
        for source in self.sources:
            identity = {
                key: source.get(key)
                for key in ("kind", "base_url", "store_id", "run_id")
            }
            if source.get("selection"):
                identity["selection"] = source["selection"]
            sources.append(identity)
        identities = [[row["run_id"], row["task_id"], row["repeat"]] for row in self.rows]
        settings = [getattr(self, name) for name in _SETTINGS]
        payload = [COHORT_VERSION, *settings, sources, identities]
        if self.parent_selection or self.taxonomy_filter:
            payload.append([self.parent_selection, self.taxonomy_filter])
        if self.limit:
            payload.append(["limit", self.limit])
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:20]

    def identities(self, run_id: str | None = None) -> list[tuple[str, int]]:
        return [(row["task_id"], row["repeat"]) for row in self.rows
                if run_id is None or row["run_id"] == run_id]

    def count(self, run_id: str | None = None) -> int:
        rows = sum(1 for row in self.rows if run_id is None or row["run_id"] == run_id)
        whole = sum(
            int(source.get("n_rollouts") or 0)
            for source in self.sources
            if source.get("selection") == "all"
            and (run_id is None or source["run_id"] == run_id)
        )
        return rows + whole

    @property
    def empty(self) -> bool:
        return not self.rows and not any(
            source.get("selection") == "all" for source in self.sources
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": COHORT_VERSION, "id": self.id, "created_at": self.created_at,
            **{name: getattr(self, name) for name in _SETTINGS},
            "sources": self.sources, "n_rollouts": self.count(), "rows": self.rows,
            "datasets": self.datasets,
            "parent_selection": self.parent_selection,
            "taxonomy_filter": self.taxonomy_filter,
            "limit": self.limit,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Cohort:
        if data.get("version") != COHORT_VERSION:
            raise ValueError(f"Unsupported cohort version {data.get('version')!r}")
        cohort = cls(
            **{name: data[name] for name in _SETTINGS},
            sources=list(data["sources"]), rows=list(data["rows"]),
            datasets=list(data.get("datasets") or []), created_at=data.get("created_at") or "",
            parent_selection=str(data.get("parent_selection") or ""),
            taxonomy_filter=dict(data.get("taxonomy_filter") or {}),
            limit=int(data.get("limit") or 0),
        )
        if data.get("id") != cohort.id:
            raise ValueError("Cohort file does not match its id")
        return cohort


def cohorts_root() -> Path:
    return autofa_home() / "cohorts"


def select_cohort(
    *, where: str = "", runs_where: str = "", api_runs: Iterable[str] = (),
    folders: Iterable[Path | str] = (), api: DataApiClient | None = None,
    models: Iterable[str] = (), benchmarks: Iterable[str] = (),
    failures: bool = False, tasks: Iterable[str] = (),
    rewards: Iterable[str] = (), healthy_scored: bool = False,
    latest: int = 0, per_run: int = 0, sample: str = "first", seed: int = 0,
    limit: int = 0,
    local_store_root: Path | None = None,
    refresh: bool = False,
    efficiency_only: bool = False,
) -> Cohort:
    """Select matching rollouts from remote runs or local EvalRDL tables."""
    if sample not in SAMPLES:
        raise ValueError(f"sample must be one of {', '.join(SAMPLES)}")
    if per_run < 0:
        raise ValueError("per_run must not be negative")
    if latest < 0:
        raise ValueError("latest must not be negative")
    if limit < 0:
        raise ValueError("limit must not be negative")
    if sample == "random" and not per_run:
        raise ValueError("random sampling requires per_run")
    if seed and sample != "random":
        raise ValueError("seed requires random sampling")
    api_runs, folders = list(api_runs), list(folders)
    models, benchmarks, tasks = list(models), list(benchmarks), list(tasks)
    reward_filters = parse_reward_filters(rewards)
    if api_runs and latest:
        raise ValueError("latest cannot be combined with explicit API runs")
    if folders and latest:
        raise ValueError("latest requires a remote catalog query")
    if folders and (runs_where.strip() or where.strip()):
        raise ValueError("--runs-where and --where require remote EvalRDL runs")
    run_filter = _all_of(_one_of("model", models), _one_of("benchmark", benchmarks),
                         runs_where)
    outcome_filter = (
        _FAILURES
        if failures
        else _all_of(
            _HEALTHY_SCORED if healthy_scored or reward_filters else "",
            *(_reward_sql(operator, value) for operator, value in reward_filters),
        )
    )
    rollout_filter = _all_of(outcome_filter, _one_of("task_id", tasks), where)
    if not api_runs and not folders and not run_filter and not latest:
        raise ValueError("Select at least one remote run, local folder or run filter")
    if not api_runs and not folders and run_filter and not latest:
        raise ValueError("Catalog filters require latest")
    order = _Order(sample, seed)
    sources: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if api_runs or not folders:
        whole_runs = not rollout_filter and not per_run and not limit
        api = api or DataApiClient()
        for run_id, run_uuid in _remote_runs(api, run_filter, api_runs, latest):
            source, selected = (
                (_select_whole_remote(api, run_id), [])
                if whole_runs
                else _select_remote(api, run_id, run_uuid, rollout_filter, per_run, order)
            )
            sources.append(source)
            rows.extend(selected)
    for path in folders:
        selected_run = _select_local(
            Path(path),
            models,
            benchmarks,
            failures,
            reward_filters,
            healthy_scored,
            tasks,
            order,
            local_store_root,
            refresh,
            efficiency_only,
        )
        if selected_run:
            source, selected = selected_run
            sources.append(source)
            rows.extend(selected[:per_run] if per_run else selected)
    if limit:
        rows = rows[:limit]
        selected_runs = {str(row["run_id"]) for row in rows}
        sources = [
            source for source in sources
            if str(source["run_id"]) in selected_runs
        ]
    identities = [(row["run_id"], row["task_id"], row["repeat"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise DataApiError("The same run was selected more than once")
    return Cohort(
        runs_where=run_filter,
        where=rollout_filter,
        latest=latest, per_run=per_run, sample=sample, seed=seed,
        limit=limit,
        sources=sources, rows=rows,
        created_at=datetime.now(UTC).isoformat(),
    )


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    raise TypeError(f"unsupported SQL value {value!r}")


def _one_of(name: str, values: Iterable[Any]) -> str:
    values = list(values)
    return f"{name} IN ({', '.join(_sql_literal(value) for value in values)})" if values else ""


def _all_of(*conditions: str) -> str:
    parts = [condition.strip() for condition in conditions if condition and condition.strip()]
    return parts[0] if len(parts) == 1 else " AND ".join(f"({part})" for part in parts)


def parse_reward_filters(values: Iterable[str]) -> list[tuple[str, float]]:
    filters = []
    for raw in values:
        match = _REWARD.fullmatch(str(raw))
        if not match:
            raise ValueError(
                f"Invalid reward comparison {raw!r}; use forms such as '<1', '<=0.5', or '=0'"
            )
        operator, number = match.groups()
        filters.append(("==" if operator == "=" else operator, float(number)))
    return filters


def _reward_sql(operator: str, value: float) -> str:
    return f"reward {'=' if operator == '==' else operator} {_sql_literal(value)}"


def matches_reward_filters(value: Any, filters: list[tuple[str, float]]) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return all(
        {
            "<": value < target,
            "<=": value <= target,
            "==": value == target,
            ">=": value >= target,
            ">": value > target,
        }[operator]
        for operator, target in filters
    )


@dataclass(frozen=True)
class _Order:
    """Rollout order within a run, identical for the analytics API and local tables."""

    sample: str
    seed: int

    def _hash_sql(self) -> str:
        return f"SHA256(concat({_sql_literal(f'{self.seed}/')}, task_id, '/', toString(repeat)))"

    def _hash(self, row: Mapping[str, Any]) -> bytes:
        return hashlib.sha256(f"{self.seed}/{row['task_id']}/{row['repeat']}".encode()).digest()

    def sql(self) -> str:
        if self.sample == "random":
            return f"{self._hash_sql()}, agent_ref"
        return "task_id, repeat, agent_ref"

    def after(self, row: Mapping[str, Any]) -> str:
        agent = _sql_literal(row["agent_ref"])
        if self.sample == "random":
            return f"({self.sql()}) > (unhex('{self._hash(row).hex()}'), {agent})"
        return f"({self.sql()}) > ({_sql_literal(row['task_id'])}, {row['repeat']}, {agent})"

    def key(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        if self.sample == "random":
            return self._hash(row), row.get("agent_ref") or ""
        return row["task_id"], row["repeat"]


def query_remote_run(
    api: DataApiClient, run_id: str, run_uuid: str,
    select: str, conditions: Iterable[str] = (), tail: str = "",
) -> tuple[dict[str, Any], str]:
    """Run ``select WHERE <this run> AND conditions tail`` on EvalRDL's trace tables.

    The trace tables (``di.rollouts``, ``di.events``, ...) are partitioned and sorted by
    ``run_uuid``, where a ``run_id`` filter reads every partition. A server whose trace tables
    predate that layout rejects the unknown column with a 400, and the query runs again on
    ``run_id``. Returns the result and the ``run_uuid`` to keep using: empty after a fallback.
    """
    conditions = list(conditions)

    def sql(run: str) -> str:
        return f"{select} WHERE {' AND '.join([run, *conditions])} {tail}".rstrip()

    if run_uuid:
        try:
            return api.query_analytics(sql(f"run_uuid = {_sql_literal(run_uuid)}")), run_uuid
        except DataApiError as exc:
            if exc.status != 400 or not _NO_RUN_UUID_COLUMN.search(str(exc)):
                raise
    return api.query_analytics(sql(f"run_id = {_sql_literal(run_id)}")), ""


def remote_run_uuid(api: DataApiClient, source: Mapping[str, Any]) -> str:
    """The ``run_uuid`` of a remote cohort source; a source saved without one asks the catalog."""
    recorded = str(source.get("run_uuid") or "").strip()
    if recorded:
        return recorded
    return str(api.get_run(str(source["run_id"])).get("run_uuid") or "").strip()


def _catalog_identity(catalog: Mapping[str, Any], requested: str) -> tuple[str, str]:
    return (
        str(catalog.get("run_id") or requested).strip(),
        str(catalog.get("run_uuid") or "").strip(),
    )


def _remote_runs(
    api: DataApiClient, run_filter: str, runs: list[str], latest: int,
) -> list[tuple[str, str]]:
    """``(run_id, run_uuid)`` of each selected run; ``run_uuid`` is empty if the server has none.

    The catalog accepts a run uuid or run_id; the rows keep the canonical run_id.
    """
    explicit = [_catalog_identity(api.get_run(run), run) for run in runs]
    if explicit and not run_filter:
        return explicit
    clauses = ["run_id IS NOT NULL"]
    if run_filter:
        clauses.append(f"({run_filter})")
    if explicit:
        clauses.append(f"run_id IN ({', '.join(_sql_literal(run) for run, _ in explicit)})")
    result = api.query_analytics(
        f"SELECT run_id, any(run_uuid) AS run_uuid, max(run_ts) AS selected_run_ts FROM di.runs "
        f"WHERE {' AND '.join(clauses)} GROUP BY run_id "
        f"ORDER BY selected_run_ts DESC NULLS LAST, run_id LIMIT {latest or MAX_RUNS + 1}"
    )
    matched = {row["run_id"]: str(row.get("run_uuid") or "") for row in result["rows"]}
    if not latest and len(matched) > MAX_RUNS:
        raise ValueError(f"More than {MAX_RUNS} runs match; narrow the run filter")
    if explicit:
        return [identity for identity in explicit if identity[0] in matched]
    return list(matched.items())


def _select_whole_remote(api: DataApiClient, run_id: str) -> dict[str, Any]:
    catalog = api.get_run(run_id)
    canonical, run_uuid = _catalog_identity(catalog, run_id)
    return {
        "kind": "remote",
        "base_url": api.base_url,
        "run_id": canonical,
        **({"run_uuid": run_uuid} if run_uuid else {}),
        "selection": "all",
        "n_rollouts": catalog.get("n_rollouts"),
        "dataset_contract": catalog.get("dataset_contract"),
        "metadata_watermark": catalog.get("metadata_watermark"),
    }


def _select_remote(
    api: DataApiClient, run_id: str, run_uuid: str, rollout_filter: str, limit: int,
    order: _Order,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clauses = [f"({rollout_filter})"] if rollout_filter else []
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    after = ""
    address = run_uuid
    while True:
        size = min(PAGE_SIZE, limit - len(rows)) if limit else PAGE_SIZE
        result, address = query_remote_run(
            api, run_id, address, f"SELECT {', '.join(_COLUMNS)} FROM di.rollouts",
            clauses + ([after] if after else []), f"ORDER BY {order.sql()} LIMIT {size}",
        )
        page = result["rows"]
        rows.extend(page)
        if page:
            after = order.after(page[-1])
        if (limit and len(rows) >= limit) or not page or (
                len(page) < size and not result.get("possibly_truncated")):
            break
    pairs = {(row["task_id"], row["repeat"]) for row in rows}
    if len(pairs) != len(rows):
        raise DataApiError(f"Run {run_id} has ambiguous task/repeat identities across agents")
    source = {
        "kind": "remote", "base_url": api.base_url, "run_id": run_id,
        **({"run_uuid": run_uuid} if run_uuid else {}),
        "dataset_contract": result.get("dataset_contract"),
        "metadata_watermark": result.get("metadata_watermark"),
    }
    return source, rows


def _select_local(
    path: Path, models: list[str], benchmarks: list[str], failures: bool,
    reward_filters: list[tuple[str, float]], healthy_scored: bool,
    tasks: list[str], order: _Order, local_store_root: Path | None,
    refresh: bool, efficiency_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    from autofa.data.local_import import import_local_run, reference_for_import
    from autofa.data.local_trace_store import resolve_store

    reference = reference_for_import(path)
    if reference:
        store = resolve_store(reference["store_id"])
        tables = store / "tables"
        store_id, run_id = reference["store_id"], reference["run_id"]
        source_harness = reference.get("source_harness") or "unknown"
    else:
        imported = import_local_run(
            path,
            store_root=local_store_root,
            force=refresh,
            efficiency_only=efficiency_only,
        )
        tables = imported.parquet_path.parent / "tables"
        store_id, run_id = imported.store_id, imported.run_id
        source_harness = imported.source_harness
    if models or benchmarks:
        run_rows = pq.read_table(tables / "runs.parquet").to_pylist()
        if not any(
            (not models or row.get("model") in models)
            and (not benchmarks or row.get("benchmark") in benchmarks)
            for row in run_rows
        ):
            return None
    table = pq.ParquetFile(tables / "rollouts.parquet")
    columns = [name for name in _COLUMNS if name in table.schema_arrow.names]
    task_set = set(tasks)

    def matches(row: Mapping[str, Any]) -> bool:
        if task_set and row.get("task_id") not in task_set:
            return False
        if failures:
            return is_scored_failure(row)
        if healthy_scored or reward_filters:
            if not is_healthy_scored(row):
                return False
        return not reward_filters or matches_reward_filters(row.get("reward"), reward_filters)

    rows = sorted(
        (row for row in table.read(columns=columns).to_pylist() if matches(row)),
        key=order.key,
    )
    source = {
        "kind": "local", "path": str(path.expanduser().resolve()),
        "store_id": store_id, "run_id": run_id, "source_harness": source_harness,
    }
    return source, rows


def save_cohort(cohort: Cohort) -> Path:
    """Write ``cohort.json``; an existing cohort keeps its prepared datasets."""
    path = cohorts_root() / cohort.id / "cohort.json"
    if path.is_file():
        previous = load_cohort(path)
        current_runs = {
            str(dataset.get("run_id"))
            for dataset in cohort.datasets
            if dataset.get("run_id") is not None
        }
        cohort.datasets = [
            *cohort.datasets,
            *(
                dataset
                for dataset in previous.datasets
                if str(dataset.get("run_id")) not in current_runs
            ),
        ]
        previous_sources = {
            str(source["run_id"]): source
            for source in previous.sources
            if source.get("run_id") is not None
        }
        for source in cohort.sources:
            old = previous_sources.get(str(source.get("run_id")))
            if old and source.get("population") is None and old.get("population") is not None:
                source["population"] = old["population"]
        cohort.created_at = previous.created_at
    atomic_write_json(path, cohort.to_json())
    return path


def load_cohort(value: Path | str) -> Cohort:
    """Load a cohort by id or by the path of its ``cohort.json``."""
    path = Path(value).expanduser()
    if not path.exists():
        path = cohorts_root() / str(value) / "cohort.json"
    elif path.is_dir():
        path = path / "cohort.json"
    return Cohort.from_json(json.loads(path.read_text(encoding="utf-8")))


def resolve_inputs(values: Iterable[str | Path]) -> tuple[str, list[Path | str]]:
    """Classify explicit CLI inputs without guessing between incompatible source kinds."""
    resolved: list[tuple[str, Path | str]] = []
    for raw in values:
        value = str(raw)
        path = Path(value).expanduser()
        cohort_file = (
            path / "cohort.json"
            if path.is_dir()
            else path
            if path.is_file() and path.name == "cohort.json"
            else cohorts_root() / value / "cohort.json"
        )
        if cohort_file.is_file():
            resolved.append(("selection", path if path.exists() else value))
            continue
        if not path.exists():
            raise ValueError(
                f"Input does not exist and is not a saved selection: {value}"
            )
        from autofa.taxonomy.manifest import manifest_path

        taxonomy = (
            path.is_file() and manifest_path(path).is_file()
        ) or (
            path.is_dir()
            and any(
                manifest_path(candidate).is_file()
                for candidate in path.rglob("*.jsonl")
            )
        )
        resolved.append(("taxonomy" if taxonomy else "local", path))
    kinds = {kind for kind, _value in resolved}
    if len(kinds) > 1:
        raise ValueError("Do not mix local data, taxonomy results, and saved selections")
    kind = next(iter(kinds), "")
    selected = [value for _kind, value in resolved]
    if kind == "selection" and len(selected) != 1:
        raise ValueError("Use one saved selection at a time")
    if kind == "local":
        selected = _expand_local_run_inputs(selected)
    return kind, selected


def _expand_local_run_inputs(values: Iterable[Path | str]) -> list[Path]:
    """Expand parent folders into declared Gym/Harbor runs, preserving explicit inputs."""
    from autofa.data.sources import discover_eval_sources

    expanded: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        declared = [
            source.path.expanduser().resolve()
            for source in discover_eval_sources(path)
            if source.kind == "evalrdl"
        ]
        expanded.extend(declared or [path])
    return list(dict.fromkeys(expanded))


def refine_cohort(
    cohort: Cohort,
    *,
    tasks: Iterable[str] = (),
    rewards: Iterable[str] = (),
    per_run: int = 0,
    sample: str = "first",
    seed: int = 0,
    limit: int = 0,
    api: DataApiClient | None = None,
) -> Cohort:
    """Narrow a materialized selection without returning to its source."""
    tasks, rewards = list(tasks), list(rewards)
    parent_selection = cohort.id
    narrowing = any((tasks, rewards, per_run, limit, sample != "first", seed))
    if any(source.get("selection") == "all" for source in cohort.sources):
        if not narrowing:
            return cohort
        cohort = _materialize_whole_run_rows(cohort, api=api)
    task_set = set(tasks)
    reward_filters = parse_reward_filters(rewards)
    order = _Order(sample, seed)
    rows = [
        row
        for row in cohort.rows
        if (not task_set or str(row.get("task_id")) in task_set)
        and (
            not reward_filters
            or (
                is_healthy_scored(row)
                and matches_reward_filters(row.get("reward"), reward_filters)
            )
        )
    ]
    selected_rows = []
    for source in cohort.sources:
        run_rows = sorted(
            (row for row in rows if row["run_id"] == source["run_id"]),
            key=order.key,
        )
        selected_rows.extend(run_rows[:per_run] if per_run else run_rows)
    if limit:
        selected_rows = selected_rows[:limit]
    if selected_rows == cohort.rows:
        return cohort
    selected_runs = {str(row["run_id"]) for row in selected_rows}
    return Cohort(
        runs_where=cohort.runs_where,
        where=cohort.where,
        sources=[
            dict(source)
            for source in cohort.sources
            if str(source["run_id"]) in selected_runs
        ],
        rows=selected_rows,
        latest=cohort.latest,
        per_run=per_run,
        sample=sample,
        seed=seed,
        limit=limit,
        created_at=datetime.now(UTC).isoformat(),
        parent_selection=parent_selection,
        taxonomy_filter=dict(cohort.taxonomy_filter),
    )


def _materialize_whole_run_rows(
    cohort: Cohort,
    *,
    api: DataApiClient | None = None,
) -> Cohort:
    """Resolve rollout identities for a saved whole-run selection before refinement."""
    sources: list[dict[str, Any]] = []
    rows = [dict(row) for row in cohort.rows]
    for source in cohort.sources:
        if source.get("selection") != "all":
            sources.append(dict(source))
            continue
        if source.get("kind") != "remote":
            raise ValueError("Whole-run refinement is supported only for remote EvalRDL runs")
        source_api = api
        if source_api is None or source_api.base_url != source.get("base_url"):
            source_api = DataApiClient(str(source.get("base_url") or ""))
        selected_source, selected_rows = _select_remote(
            source_api,
            str(source["run_id"]),
            remote_run_uuid(source_api, source),
            "",
            0,
            _Order("first", 0),
        )
        sources.append(selected_source)
        rows.extend(selected_rows)
    return Cohort(
        runs_where=cohort.runs_where,
        where=cohort.where,
        sources=sources,
        rows=rows,
        latest=cohort.latest,
        per_run=0,
        sample="first",
        seed=0,
        created_at=datetime.now(UTC).isoformat(),
        parent_selection=cohort.id,
        taxonomy_filter=dict(cohort.taxonomy_filter),
    )


def prepare_cohort(
    cohort: Cohort, *, api: DataApiClient | None = None,
    activity: Callable[[str], None] | None = None,
    efficiency_only: bool = False,
    refresh: bool = False,
) -> list[dict[str, str]]:
    """Materialize exact selected datasets and save their references."""
    from autofa.data.efficiency_support import check_cohort

    check_cohort(cohort)
    datasets = []
    for source in cohort.sources:
        run_id = source["run_id"]
        identities = cohort.identities(run_id)
        whole_run = source.get("selection") == "all"
        if not identities and not whole_run:
            continue
        if activity:
            count = source.get("n_rollouts") if whole_run else len(identities)
            activity(f"Preparing {count or 'all'} rollout(s) from {run_id}…")
        if source["kind"] == "remote":
            if api is None or api.base_url != source["base_url"]:
                api = DataApiClient(source["base_url"])
            datasets.append(
                _prepare_remote(
                    api,
                    source,
                    None if whole_run else identities,
                    activity,
                    efficiency_only=efficiency_only,
                    refresh=refresh,
                )
            )
        else:
            from autofa.data.local_trace_store import resolve_store

            store = resolve_store(source["store_id"])
            source["population"] = _parquet_population(
                store / "tables" / "rollouts.parquet"
            )
            destination_dir = cohorts_root() / cohort.id / "datasets"
            subset = stage_export_subset(
                store / "dataset.parquet",
                identities,
                destination_dir=destination_dir,
            )
            manifest = json.loads(
                (store / "manifest.json").read_text(encoding="utf-8")
            )
            if not efficiency_only and manifest.get("dataset_kind") == "compact":
                from autofa.data.local_import import materialize_local_traces

                subset = materialize_local_traces(
                    store,
                    subset,
                    destination_dir / f"{subset.stem}.traces{subset.suffix}",
                )
            datasets.append(make_local_reference_data(
                store_id=source["store_id"],
                run_id=run_id,
                source_harness=source.get("source_harness") or "unknown",
                parquet_path=subset,
            ))
    cohort.datasets = datasets
    save_cohort(cohort)
    return datasets


def prepare_data(
    cohort: Cohort, *, api: DataApiClient | None = None,
    activity: Callable[[str], None] | None = None,
    efficiency_only: bool = False,
    refresh: bool = False,
):
    """Prepare a saved rollout selection for an AutoFA workflow."""
    from autofa.data.parquet_store import infer_models_benchmarks
    from autofa.data.preparation import PreparedData
    from autofa.data.reference_data import summarize_source_harness

    datasets = prepare_cohort(
        cohort,
        api=api,
        activity=activity,
        efficiency_only=efficiency_only,
        refresh=refresh,
    )
    paths = [Path(dataset["parquet_path"]) for dataset in datasets]
    if not paths:
        raise ValueError("The selection contains no prepared datasets")
    metadata = infer_models_benchmarks(paths)
    models = [str(model) for model, _count in metadata["models"]]
    benchmarks = [str(benchmark) for benchmark, _count in metadata["benchmarks"]]
    by_run = {dataset["run_id"]: dataset for dataset in datasets}
    source_paths = [
        (
            canonical_export_url(source["base_url"], source["run_id"])
            if source["kind"] == "remote"
            and source.get("base_url")
            else str(
                by_run.get(source["run_id"], {}).get("export_url")
                or source.get("path")
                or by_run.get(source["run_id"], {}).get("parquet_path")
                or ""
            )
        )
        for source in cohort.sources
        if source.get("selection") == "all" or cohort.identities(source["run_id"])
    ]
    kinds = {source["kind"] for source in cohort.sources}
    return PreparedData(
        parquet_paths=paths,
        model_id=models[0] if len(models) == 1 else "*",
        benchmarks=benchmarks,
        source_paths=source_paths,
        source_kind="remote_run" if kinds == {"remote"} else "evalrdl",
        source_harness=summarize_source_harness(
            dataset.get("source_harness") for dataset in datasets
        ),
        reference_data=datasets,
        selection_id=cohort.id,
    )


def _prepare_remote(
    api: DataApiClient,
    source: dict[str, Any],
    identities: list[tuple[str, int]] | None,
    activity: Callable[[str], None] | None,
    *,
    efficiency_only: bool = False,
    refresh: bool = False,
) -> dict[str, str]:
    run_id = source["run_id"]
    catalog = api.get_run(run_id)
    export = materialize_remote_run(
        run_id,
        base_url=api.base_url,
        client=api,
        catalog=catalog,
        force=refresh,
    )
    source["population"] = _parquet_population(export)
    if identities is not None:
        try:
            export = stage_export_subset(export, identities)
        except MissingSelectedRolloutsError:
            if activity:
                activity(f"Refreshing incomplete cached export for {run_id}…")
            export = materialize_remote_run(
                run_id, base_url=api.base_url, client=api, catalog=catalog, force=True,
            )
            source["population"] = _parquet_population(export)
            export = stage_export_subset(export, identities)
    full_traces = False
    if efficiency_only:
        from autofa.data.source_export import materialize_efficiency_tables

        export = materialize_efficiency_tables(
            export,
            api=api,
            run_id=run_id,
            catalog=catalog,
            activity=activity,
        )
    elif load_config().remote_full_traces:
        export = materialize_source_traces(
            export, api=api, run_id=run_id, catalog=catalog, activity=activity,
        )
        full_traces = True
    source_harness, evidence_scope, _trace_manifest = remote_reference_details(
        export,
        catalog,
        full_traces=full_traces,
        efficiency_only=efficiency_only,
    )
    return make_reference_data(
        run_id=run_id, export_url=canonical_export_url(api.base_url, run_id),
        source_harness=source_harness,
        parquet_path=export,
        evidence_scope=evidence_scope,
        run_uuid=catalog.get("run_uuid"),
    )


def _parquet_population(path: Path) -> dict[str, int]:
    """Count source outcomes from a complete rollout Parquet."""
    rewards = pq.read_table(path, columns=["reward"])["reward"].to_pylist()
    outcomes = [population_outcome(reward) for reward in rewards]
    scored = len(outcomes) - outcomes.count("unscored")
    zero_reward = outcomes.count("fail")
    positive_reward = outcomes.count("pass")
    rows = len(outcomes)
    return {
        "rows": rows,
        "scored": scored,
        "zero_reward": zero_reward,
        "positive_reward": positive_reward,
        "unscored": rows - scored,
    }


def _positive_count(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def add_cohort_arguments(
    parser: argparse.ArgumentParser,
    *,
    taxonomy_inputs: bool = False,
    limit_help: str | None = None,
) -> None:
    """Add the rollout selection flags shared by commands that build cohorts."""
    runs = parser.add_argument_group("which runs")
    runs.add_argument("--api-run", action="append", default=[], metavar="ID",
                      help="EvalRDL run uuid, or its 16-hex run_id (repeatable)")
    runs.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="PATH_OR_SELECTION",
        help=(
            "Local evaluation data, taxonomy results, or saved selection (repeatable)"
            if taxonomy_inputs
            else "Local evaluation data or saved selection (repeatable)"
        ),
    )
    runs.add_argument("--model", action="append", default=[],
                      help="Only runs of this model (repeatable)")
    runs.add_argument("--benchmark", action="append", default=[],
                      help="Only runs of this benchmark (repeatable)")
    runs.add_argument("--runs-where", default="", metavar="FILTER",
                      help="EvalRDL filter for di.runs (remote only)")
    runs.add_argument("--latest", type=_positive_count, default=0, metavar="N",
                      help="Use the N newest runs from a catalog query")
    rollouts = parser.add_argument_group("which rollouts")
    rollouts.add_argument("--task", action="append", default=[], metavar="ID",
                          help="Only this task id (repeatable)")
    rollouts.add_argument(
        "--reward",
        action="append",
        default=[],
        metavar="COMPARISON",
        help="Reward comparison such as '<1', '<=0.5', or '=0' (repeatable)",
    )
    rollouts.add_argument("--where", default="", metavar="FILTER",
                          help="EvalRDL filter for di.rollouts (remote only)")
    count = parser.add_argument_group("how many")
    count.add_argument(
        "--limit",
        type=_positive_count,
        default=0,
        metavar="N",
        help=limit_help or (
            "Keep at most N rollouts across evaluation inputs; for taxonomy input, "
            "limit failures before controls are added"
        ),
    )
    count.add_argument("--per-run", type=_positive_count, default=0, metavar="N",
                       help="Keep at most N rollouts per run")
    count.add_argument("--sample", choices=SAMPLES, default="first",
                       help="Select first or random rollouts within each run")
    count.add_argument("--seed", type=int, default=0, help="Random sample seed")


def add_taxonomy_filter_arguments(parser: argparse.ArgumentParser) -> None:
    taxonomy = parser.add_argument_group("taxonomy results")
    taxonomy.add_argument(
        "--judge-view",
        metavar="MODEL",
        help="Use one taxonomy judge instead of consensus",
    )
    taxonomy.add_argument(
        "--group",
        choices=("all", "accepted", "review", "no-supported", "judge-disagreement"),
        default="all",
        help="Taxonomy review group (default: all)",
    )
    taxonomy.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="LABEL_OR_CODE",
        help="Taxonomy label or code (repeatable)",
    )
    taxonomy.add_argument(
        "--evidence",
        action="append",
        default=[],
        choices=("complete", "partial", "insufficient"),
        help="Evidence support level (repeatable)",
    )
    taxonomy.add_argument(
        "--list",
        action="store_true",
        dest="list_taxonomy",
        help="List taxonomy filters available in the input",
    )


def cohort_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "api_runs": args.api_run, "folders": [], "models": args.model,
        "benchmarks": args.benchmark, "rewards": args.reward, "tasks": args.task,
        "runs_where": args.runs_where, "where": args.where,
        "latest": args.latest, "per_run": args.per_run, "limit": args.limit,
        "sample": args.sample, "seed": args.seed,
    }


_EPILOG = """\
Named filters work for remote and local runs. Advanced filters are passed
unchanged to EvalRDL and require remote runs.

examples:
  autofa prepare
  autofa prepare --api-run RUN_UUID
  autofa prepare --model MODEL --benchmark BENCH --latest 5
  autofa prepare --input RESULTS_DIR --label OWN-IGM

"""


def _pick_interactive_sources(
    api: DataApiClient, *, api_name: str = "EvalRDL remote server",
) -> tuple[list[str], list[str]] | None:
    from autofa.ui.menus import MenuItem, select_option

    source = select_option(
        "Data source",
        [
            MenuItem("Local eval logs (NeMo Gym / Harbor)", "local"),
            *([MenuItem(api_name, "remote")] if api.base_url else []),
        ],
        allow_cancel=True,
    )
    if source is None:
        return None
    if source.value == "remote":
        from autofa.wizard.remote_runs import pick_remote_runs

        selected = pick_remote_runs(api)
        if selected is None:
            return None
        return [str(run["run_id"]) for run in selected], []

    from autofa.wizard.dataset import pick_local_sources

    selected = pick_local_sources(raw_only=True)
    if selected is None:
        return None
    _, sources = selected
    return [], [str(source.path) for source in sources]


def _pick_taxonomy_cohort() -> Cohort | None:
    from autofa.taxonomy.groups import (
        classification_facets,
        cohort_from_classifications,
    )
    from autofa.ui.menus import (
        MenuItem,
        browse_filesystem,
        select_items_multi,
        select_option,
    )

    selected = browse_filesystem(
        Path.home(),
        title="Select taxonomy results file or folder",
        select_files=True,
        select_dirs=True,
        file_glob="*.jsonl",
    )
    if selected is None:
        return None
    facets = classification_facets([selected])
    judge_choice = select_option(
        "Taxonomy judge view",
        [MenuItem("Consensus across judges", None)]
        + [MenuItem(judge, judge) for judge in facets["judges"]],
        allow_cancel=True,
    )
    if judge_choice is None:
        return None
    judge = judge_choice.value
    groups = [
        MenuItem("All classified failures", "all"),
        MenuItem("Accepted classifications", "accepted"),
        MenuItem("Needs review", "review"),
        MenuItem("No supported failure", "no-supported"),
    ]
    if judge is None:
        groups.append(MenuItem("Judge disagreement", "judge-disagreement"))
    group_choice = select_option("Taxonomy group", groups, allow_cancel=True)
    if group_choice is None:
        return None
    labels = select_items_multi(
        [
            MenuItem(label, code, detail=code)
            for code, label in sorted(facets["labels"].items())
        ],
        title="Failure labels (leave empty for all)",
        empty_means_all=True,
        confirm=True,
        confirm_noun="failure labels",
    )
    if labels is None:
        return None
    evidence = select_items_multi(
        [
            MenuItem(level.replace("-", " ").capitalize(), level)
            for level in facets["evidence"]
        ],
        title="Evidence support (leave empty for all)",
        empty_means_all=True,
        confirm=True,
        confirm_noun="evidence levels",
    )
    if evidence is None:
        return None
    return cohort_from_classifications(
        facets["paths"],
        judge=judge,
        group=group_choice.value,
        labels=labels,
        evidence=evidence,
    )


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        prog="autofa prepare",
        description="Prepare a reusable rollout selection.",
        epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_cohort_arguments(parser, taxonomy_inputs=True)
    add_taxonomy_filter_arguments(parser)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show match counts without saving or downloading")
    parser.add_argument("--api-base", default=None, help="Data API base URL")
    args = parser.parse_args(argv)
    options = cohort_options(args)
    if args.sample == "random" and not args.per_run:
        parser.error("--sample random requires --per-run")
    if args.seed and args.sample != "random":
        parser.error("--seed requires --sample random")
    taxonomy_filters = bool(
        args.judge_view
        or args.group != "all"
        or args.label
        or args.evidence
        or args.list_taxonomy
    )

    cfg = load_config()
    api = DataApiClient(args.api_base or cfg.data_api_base_url)
    try:
        input_kind, inputs = resolve_inputs(args.input)
        if taxonomy_filters and input_kind != "taxonomy":
            parser.error("taxonomy filters require taxonomy results passed with --input")
        if input_kind in {"selection", "taxonomy"} and any((
            options["api_runs"],
            options["models"],
            options["benchmarks"],
            options["runs_where"],
            options["latest"],
            options["where"],
        )):
            parser.error(
                f"{input_kind} input cannot be combined with run or advanced WHERE filters"
            )
        if input_kind == "local":
            options["folders"] = inputs
        has_source = any((
            inputs, options["api_runs"], options["folders"], options["models"],
            options["benchmarks"], options["runs_where"], options["latest"],
        ))
        cohort = None
        if not has_source:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                parser.error(
                    "pass --input, --api-run, --model, --benchmark, "
                    "--runs-where or --latest"
                )
            from autofa.ui.menus import MenuItem, select_option

            source = select_option(
                "Prepare data from",
                [
                    MenuItem("Evaluation runs", "runs"),
                    MenuItem("Taxonomy classification results", "taxonomy"),
                ],
                allow_cancel=True,
            )
            if source is None:
                return 0
            if source.value == "taxonomy":
                cohort = _pick_taxonomy_cohort()
                if cohort is None:
                    return 0
            else:
                selected = _pick_interactive_sources(api, api_name=cfg.data_api_name)
                if selected is None:
                    return 0
                options["api_runs"], options["folders"] = selected
        if cohort is None:
            if input_kind == "taxonomy":
                from autofa.taxonomy.groups import (
                    classification_facets,
                    cohort_from_classifications,
                )

                if args.list_taxonomy:
                    facets = classification_facets([Path(value) for value in inputs])
                    print("Judges: " + ", ".join(facets["judges"]))
                    print("Labels:")
                    for code, label in sorted(facets["labels"].items()):
                        print(f"  {code}: {label}")
                    print("Evidence: " + ", ".join(facets["evidence"]))
                    return 0
                cohort = cohort_from_classifications(
                    [Path(value) for value in inputs],
                    judge=args.judge_view,
                    group=args.group,
                    labels=args.label,
                    evidence=args.evidence,
                )
                cohort = refine_cohort(
                    cohort,
                    tasks=options["tasks"],
                    rewards=options["rewards"],
                    per_run=options["per_run"],
                    sample=options["sample"],
                    seed=options["seed"],
                    limit=options["limit"],
                    api=api,
                )
            elif input_kind == "selection":
                cohort = refine_cohort(
                    load_cohort(inputs[0]),
                    tasks=options["tasks"],
                    rewards=options["rewards"],
                    per_run=options["per_run"],
                    sample=options["sample"],
                    seed=options["seed"],
                    limit=options["limit"],
                    api=api,
                )
            elif args.dry_run and options["folders"]:
                with tempfile.TemporaryDirectory(prefix="autofa-prepare-") as root:
                    cohort = select_cohort(
                        **options, api=api, local_store_root=Path(root),
                    )
            else:
                cohort = select_cohort(**options, api=api)
        if cohort.taxonomy_filter:
            from autofa.taxonomy.groups import add_success_controls

            cohort = add_success_controls(cohort, api=api, activity=print)
        for source in cohort.sources:
            count = cohort.count(source["run_id"])
            print(f"{source['kind']} run {source['run_id']}: {count} rollout(s) selected")
        if cohort.empty:
            if args.dry_run:
                print("No rollouts matched")
                return 0
            raise ValueError("No rollouts matched")
        if args.dry_run:
            return 0
        path = save_cohort(cohort)
        print(f"Selection {cohort.id}: {cohort.count()} rollout(s), saved to {path}")
        for dataset in prepare_cohort(cohort, api=api, activity=print):
            print(f"Ready {dataset['run_id']}: {dataset['parquet_path']}")
        if cohort.taxonomy_filter:
            print(f"Start discovery: autofa new --input {cohort.id}")
    except (DataApiError, ValueError, OSError) as exc:
        print(f"autofa prepare: {exc}", file=sys.stderr)
        return 1
    return 0
