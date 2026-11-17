# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Efficiency metrics computed using EvalRDL tables.

Input and completion tokens, assistant turns, and tool calls come directly from
the rollout summary. Total tokens are input plus completion. A null reward is
unscored.

Metrics available only on detailed rows use EvalRDL's rollup rules: counted
lanes only, one value per ``(lane, request_idx)`` where an index exists, one
value per row otherwise, and the larger of a rollout's alternate accounts
(``history`` and ``session``). Alternate accounts are never added together.
Missing rollout token totals fall back directly to requests recorded for the
selected model. Auxiliary models are excluded, and ambiguous model identities
are not guessed.

A missing column, or a column that is entirely null, is reported as a warning
and left out. Recorded zeros stay in the metric. Nothing is estimated from text
length. A remote
download keeps the same rows in its trace index when it has no parquet tables.

Cost is simulated from the recorded token counts:
uncached input × input rate + cached input × cached rate + completion × output rate,
per million tokens. Cached tokens EvalRDL did not record count as zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from autofa.data.outcomes import efficiency_outcome
from autofa.data.reference_data import require_canonical_reference_data
from evalrdl_readers.convert import ROLLUP_LANES_ALTERNATE, ROLLUP_LANES_NOT_COUNTED
from evalrdl_schema.tables import RunTables

_ROLL_KEYS = ("run_id", "task_id", "repeat", "agent_ref")

# (name, label, unit, table, column).
_METRICS = (
    ("input_tokens", "Input tokens", "tokens", "rollouts", "prompt_tokens"),
    ("completion_tokens", "Completion tokens", "tokens", "rollouts", "completion_tokens"),
    ("cached_tokens", "Cached tokens", "tokens", "events", "cached_tokens"),
    ("reasoning_tokens", "Reasoning tokens", "tokens", "events", "reasoning_tokens"),
    ("n_turns", "Assistant turns", "turns", "rollouts", "n_assistant_turns"),
    ("n_tool_calls", "Tool calls", "calls", "rollouts", "n_tool_calls"),
)

_TOKEN_EVENT_COLUMN = {
    "input_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "cached_tokens": "cached_tokens",
    "reasoning_tokens": "reasoning_tokens",
}
_READS = {
    "rollouts": (
        *_ROLL_KEYS,
        "reward",
        "is_scored",
        "prompt_tokens",
        "completion_tokens",
        "n_assistant_turns",
        "n_tool_calls",
    ),
    "events": (
        *_ROLL_KEYS,
        "lane",
        "event_idx",
        "request_idx",
        "model_name",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "reasoning_tokens",
    ),
    "tool_calls": (*_ROLL_KEYS, "lane", "tool_name", "args_chars"),
    "parts": (*_ROLL_KEYS, "lane", "channel", "chars"),
}

_SIMULATED_COST = ("simulated_cost_usd", "Simulated cost", "USD")
CARD_METRICS = ("total_tokens", "n_turns", "n_tool_calls", _SIMULATED_COST[0])

_DEFAULT_PRICING = {
    "source": "simulated_fallback",
    "input_usd_per_million": 0.15,
    "cached_usd_per_million": 0.03,
    "output_usd_per_million": 0.50,
}

_NOT_COUNTED = tuple(ROLLUP_LANES_NOT_COUNTED)
_ALTERNATE = tuple(ROLLUP_LANES_ALTERNATE)


class UnsupportedEfficiencySelection(ValueError):
    """Efficiency analysis requires exactly one selected evaluation run."""


def validate_efficiency_run_selection(selection: Any) -> str:
    """Return the selected run identity, rejecting multi-run or empty selections."""
    references = require_canonical_reference_data(
        getattr(selection, "reference_data", None),
        getattr(selection, "parquet_paths", None) or [],
    )
    identities: set[str] = set()
    for reference in references:
        run_id = str(reference.get("run_id") or "").strip()
        store_id = str(reference.get("store_id") or "").strip()
        export_url = str(reference.get("export_url") or "").strip()
        if store_id:
            identities.add(f"store:{store_id}:run:{run_id}")
        elif export_url:
            identities.add(f"export:{export_url}:run:{run_id}")
        else:
            identities.add(f"run:{run_id}")
    if len(identities) != 1:
        raise UnsupportedEfficiencySelection(
            "Efficiency analysis supports exactly one evaluation run. "
            "Select a single run and retry."
        )
    tables = locate_evalrdl_tables(selection)
    indexes = [] if tables else locate_bulk_indexes(selection)
    if not tables and not indexes:
        raise UnsupportedEfficiencySelection(
            "Efficiency analysis requires complete EvalRDL tables for the selected run. "
            "Import or refresh the run data and retry."
        )
    return next(iter(identities))


def locate_evalrdl_tables(run: Any) -> list[Path]:
    """Table directories for this run's EvalRDL stores."""
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if resolved in seen or not (resolved / "rollouts.parquet").is_file():
            return
        seen.add(resolved)
        found.append(resolved)

    for ref in getattr(run, "reference_data", None) or []:
        if not isinstance(ref, dict):
            continue
        store_id = ref.get("store_id")
        if ref.get("kind") == "local_evalrdl" and isinstance(store_id, str):
            from autofa.data.local_trace_store import resolve_store

            try:
                add(resolve_store(store_id) / "tables")
            except (OSError, ValueError):
                continue
    return found


def locate_bulk_indexes(run: Any) -> list[Path]:
    """Trace indexes for the selected export, including all of its batches."""
    found: list[Path] = []
    seen: set[Path] = set()
    visited_exports: set[Path] = set()
    references = require_canonical_reference_data(
        getattr(run, "reference_data", None),
        getattr(run, "parquet_paths", None) or [],
    )
    for reference in references:
        if reference.get("kind") == "local_evalrdl":
            continue
        path = Path(reference["parquet_path"])
        export = path if path.is_file() else path / "dataset.parquet"
        adapter_root = export.parent
        try:
            adapter_identity = adapter_root.resolve()
        except OSError:
            continue
        if adapter_identity in visited_exports:
            continue
        visited_exports.add(adapter_identity)
        cache_root = adapter_root.parent
        expected_hashes = _trace_export_hashes(adapter_root / "manifest.json")
        if not expected_hashes:
            try:
                with export.open("rb") as handle:
                    export_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            except OSError:
                export_sha256 = ""
            markers = [
                export.with_name(f"{export.name}.efficiency-tables.json"),
                adapter_root / "efficiency-tables.json",
            ]
            for marker in markers:
                efficiency_tables = _json_file(marker)
                if (
                    efficiency_tables.get("run_id") == reference.get("run_id")
                    and efficiency_tables.get("export_sha256") == export_sha256
                ):
                    expected_hashes = {
                        str(item["sha256"])
                        for item in efficiency_tables.get("trace_exports") or []
                        if isinstance(item, dict) and item.get("sha256")
                    }
                    break
        if not expected_hashes:
            raise UnsupportedEfficiencySelection(
                "The selected remote EvalRDL export has no trace-batch manifest. "
                "Refresh the run data and retry."
            )
        candidates: list[Path] = []
        for pattern_root, pattern in (
            (adapter_root, "bulk/**/trace.sqlite"),
            (adapter_root, "source-bulk-*/**/trace.sqlite"),
            (cache_root, "source-bulk-*/**/trace.sqlite"),
        ):
            try:
                candidates.extend(
                    sorted(pattern_root.glob(pattern)) if pattern_root.is_dir() else []
                )
            except OSError:
                continue
        matched: dict[str, Path] = {}
        for index in candidates:
            try:
                resolved = index.resolve()
            except OSError:
                continue
            if resolved in seen or not resolved.is_file():
                continue
            receipt = _json_file(index.parent / "download.json")
            archive_hash = receipt.get("archive_sha256")
            if archive_hash not in expected_hashes:
                continue
            matched.setdefault(archive_hash, resolved)
        if set(matched) != expected_hashes:
            raise UnsupportedEfficiencySelection(
                "Efficiency analysis could not resolve every trace batch for the selected run. "
                "Refresh the run data and retry."
            )
        for resolved in matched.values():
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
    return found


def _trace_export_hashes(manifest_path: Path) -> set[str]:
    exports = _json_file(manifest_path).get("trace_exports")
    if not isinstance(exports, list):
        return set()
    return {
        value
        for item in exports
        if isinstance(item, dict)
        and isinstance((value := item.get("sha256")), str)
        and value
    }


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_evalrdl_efficiency(run: Any, pricing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize recorded EvalRDL usage. Missing fields become warnings."""
    validate_efficiency_run_selection(run)
    tables = locate_evalrdl_tables(run)
    indexes = [] if tables else locate_bulk_indexes(run)
    if not tables and not indexes:
        payload = _empty_payload()
        payload["warnings"] = [
            "EvalRDL tables were not found for this run, so no efficiency metrics were computed."
        ]
        return payload
    frames = [_read_store(path, bulk=not tables) for path in (tables or indexes)]
    combined = {
        name: pl.concat([frame[name] for frame in frames], how="diagonal_relaxed")
        for name in _READS
    }
    selected = _require_selected_rollouts(run, combined["rollouts"])
    if selected is not None:
        combined = {
            name: (
                frame.join(selected, on=["task_id", "repeat"], how="semi")
                if frame.height
                else frame
            )
            for name, frame in combined.items()
        }
    return _summarize(
        combined["rollouts"],
        combined["events"],
        combined["tool_calls"],
        combined["parts"],
        _pricing(pricing),
    )


@dataclass(frozen=True)
class EfficiencyReadiness:
    """What the efficiency report will be able to show for the selected attempts."""

    n_attempts: int
    n_pass: int
    n_fail: int
    n_with_tokens: int
    """Attempts with both input and completion tokens, as the report counts them."""


def verify_efficiency_data(run: Any) -> EfficiencyReadiness:
    """Check that prepared data can produce an efficiency report and summarize its contents.

    Reads rollout summaries, and request records only where rollout token totals are
    missing, so the counts match the report without computing it.
    """
    validate_efficiency_run_selection(run)
    tables = locate_evalrdl_tables(run)
    paths = tables or locate_bulk_indexes(run)

    def read(table: str, columns: tuple[str, ...]) -> pl.DataFrame:
        return pl.concat(
            [
                _read_store(path, bulk=not tables, reads={table: columns})[table]
                for path in paths
            ],
            how="diagonal_relaxed",
        )

    rollouts = read("rollouts", (*_ROLL_KEYS, "reward", "prompt_tokens", "completion_tokens"))
    selected = _require_selected_rollouts(run, rollouts)
    if selected is not None:
        rollouts = rollouts.join(selected, on=["task_id", "repeat"], how="semi")
    if rollouts.height == 0:
        raise UnsupportedEfficiencySelection(
            "The selected run has no rollouts, so efficiency cannot be computed."
        )
    tokens = {
        "input_tokens": _aligned(rollouts, "prompt_tokens"),
        "completion_tokens": _aligned(rollouts, "completion_tokens"),
    }
    if any(series is None or series.null_count() for series in tokens.values()):
        events = read("events", _READS["events"])
        selected_model = str(getattr(run, "model_id", "") or "").strip() or None
        for name, series in tokens.items():
            tokens[name], _ = _with_token_fallback(
                name, series, events, rollouts, selected_model=selected_model,
            )
    prompt, completion = tokens.values()
    groups = _scored(rollouts)["counts"]
    return EfficiencyReadiness(
        n_attempts=rollouts.height,
        n_pass=int(groups["n_pass"]),
        n_fail=int(groups["n_fail"]),
        n_with_tokens=(
            0 if prompt is None or completion is None
            else int((prompt.is_not_null() & completion.is_not_null()).sum())
        ),
    )


def _require_selected_rollouts(run: Any, rollouts: pl.DataFrame) -> pl.DataFrame | None:
    """Reject rows from another run or missing selected attempts; return the selection."""
    loaded_run_ids = {
        str(value)
        for value in rollouts["run_id"].drop_nulls().unique().to_list()
        if str(value).strip()
    }
    selected_run_ids = {
        str(reference.get("run_id")).strip()
        for reference in getattr(run, "reference_data", None) or []
        if isinstance(reference, dict) and str(reference.get("run_id") or "").strip()
    }
    if len(loaded_run_ids) > 1 or (
        selected_run_ids and loaded_run_ids and loaded_run_ids != selected_run_ids
    ):
        raise UnsupportedEfficiencySelection(
            "Efficiency analysis found rows from a different evaluation run. "
            "Select a single run and retry."
        )
    selected = _selected_rollouts(run)
    if selected is not None:
        available = rollouts.select("task_id", "repeat").unique()
        missing = selected.join(available, on=["task_id", "repeat"], how="anti")
        if missing.height:
            raise UnsupportedEfficiencySelection(
                "Efficiency data is missing selected rollouts. Refresh the run data and retry."
            )
    return selected


def _selected_rollouts(run: Any) -> pl.DataFrame | None:
    frames = []
    for value in getattr(run, "parquet_paths", None) or []:
        path = Path(value)
        if path.is_dir():
            path = path / "dataset.parquet"
        try:
            schema = pl.read_parquet_schema(path)
        except (OSError, ValueError):
            continue
        if not {"problem_id", "run_idx"} <= set(schema):
            continue
        frames.append(
            pl.read_parquet(path, columns=["problem_id", "run_idx"]).select(
                pl.col("problem_id").cast(pl.String).alias("task_id"),
                pl.col("run_idx").cast(pl.Int64).alias("repeat"),
            )
        )
    if not frames:
        return None
    return pl.concat(frames).unique(maintain_order=True)


def _empty_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "evalrdl",
        "n_items": 0,
        "n_pass": 0,
        "n_fail": 0,
        "n_unscored": 0,
        "n_problems": 0,
        "accuracy": None,
        "best_of_n": None,
        "best_of_label": "best-of-n",
        "metrics": [],
        "warnings": [],
        "alerts": [],
        "provenance": [],
        "has_agent_activity": False,
        "tool_types": [],
        "pricing": {},
        "_samples": {},
    }


def _read_store(
    path: Path, *, bulk: bool, reads: dict[str, tuple[str, ...]] = _READS,
) -> dict[str, pl.DataFrame]:
    if not bulk:
        tables = RunTables(path, reads)
        return {name: tables.scan(name).collect() for name in reads}
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        return {name: _json_rows(db, name, columns) for name, columns in reads.items()}


def _json_rows(db: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> pl.DataFrame:
    selected = ", ".join(f"json_extract(payload, '$.{name}')" for name in columns)
    rows = db.execute(
        f"SELECT {selected} FROM records WHERE tbl = ?",
        (table,),
    ).fetchall()
    if not rows:
        return pl.DataFrame(schema=dict.fromkeys(columns, pl.Null))
    return pl.DataFrame(
        rows, schema=list(columns), orient="row", infer_schema_length=None,
    )


def _summarize(
    rollouts: pl.DataFrame,
    events: pl.DataFrame,
    tool_calls: pl.DataFrame,
    parts: pl.DataFrame,
    pricing: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    if rollouts.height == 0:
        payload = _empty_payload()
        payload["warnings"] = ["EvalRDL rollouts are empty, so no efficiency metrics were computed."]
        return payload

    scored = _scored(rollouts)
    has_agent_activity = _has_agent_activity(rollouts, tool_calls)
    agent_metrics = {"n_turns", "n_tool_calls"}
    catalog = [
        metric for metric in _METRICS
        if has_agent_activity or metric[0] not in agent_metrics
    ]
    values: dict[str, pl.Series] = {}
    origins: dict[str, tuple[str, str]] = {}
    selected_model = str(pricing.get("model_id") or "").strip() or None
    for name, label, _unit, table, column in catalog:
        series, reason = _metric_series(table, column, label, rollouts, events)
        if name in _TOKEN_EVENT_COLUMN:
            series, request_fallback = _with_token_fallback(
                name,
                series,
                events,
                rollouts,
                selected_model=selected_model,
            )
            if request_fallback:
                origins[name] = (
                    "recorded",
                    (
                        "Used the EvalRDL summary or counted transcript where available; "
                        "rollouts without them use recorded requests for the selected model."
                        if _present(series) > request_fallback
                        else "Summed recorded requests for the selected model once per request."
                    ),
                )
        if series is None:
            warnings.append(
                reason
                or f"EvalRDL did not record {label.lower()}, so that metric was not computed."
            )
            continue
        if name not in origins:
            if table == "events":
                origins[name] = (
                    "recorded",
                    "Summed recorded values once per model request.",
                )
            else:
                origins[name] = ("recorded", "Read from the EvalRDL rollout summary.")
        _note_partial(warnings, label, series, rollouts.height)
        values[name] = series
    _derived(
        values,
        catalog,
        events,
        rollouts,
        tool_calls,
        parts,
        warnings,
        origins,
        include_agent_activity=has_agent_activity,
        selected_model=selected_model,
    )
    total = _sum_total(values)
    if total is None:
        warnings.append(
            "Input or classified output tokens were unavailable, "
            "so total tokens were not computed."
        )
    else:
        values["total_tokens"] = total
        catalog.append(("total_tokens", "Total tokens", "tokens", "", ""))
        origins["total_tokens"] = ("derived", "Input tokens plus classified output tokens.")
        _note_partial(warnings, "Total tokens", total, rollouts.height)
    uncached = _uncached_input(values)
    if uncached is not None:
        values["uncached_tokens"] = uncached
        catalog.append(("uncached_tokens", "Uncached input tokens", "tokens", "", ""))
        origins["uncached_tokens"] = ("derived", "Input tokens minus cached input tokens.")
    cost = _simulated_cost(values, pricing, rollouts.height)
    if cost is None:
        warnings.append(
            "Input or classified output tokens were unavailable, "
            "so simulated cost was not computed."
        )
    else:
        present = int(cost.is_not_null().sum())
        if present < rollouts.height:
            warnings.append(
                f"Simulated cost covers {present} of {rollouts.height} rollouts; "
                "the others lack recorded input or completion tokens."
            )
        values[_SIMULATED_COST[0]] = cost
        catalog.append((*_SIMULATED_COST, "", ""))

    payload = _empty_payload()
    payload["n_items"] = rollouts.height
    payload["pricing"] = pricing
    payload.update(scored["counts"])
    scored_attempts = payload["n_pass"] + payload["n_fail"]
    payload["accuracy"] = (
        payload["n_pass"] / scored_attempts if scored_attempts else None
    )
    if payload["best_of_n"] is None:
        warnings.append("EvalRDL did not record rollout reward, so pass rate was not computed.")
    payload["warnings"] = warnings
    payload["has_agent_activity"] = has_agent_activity
    metrics = []
    samples: dict[str, dict[str, list[float]]] = {}
    for name, label, unit, _table, _column in catalog:
        series = values.get(name)
        if series is None:
            continue
        source = pricing["source"] if name == _SIMULATED_COST[0] else "evalrdl"
        metric, sample = _metric_summary(
            name, label, unit, series, scored["groups"], rollouts, source=source
        )
        metrics.append(metric)
        samples[name] = sample
    payload["metrics"] = metrics
    if not metrics and scored_attempts == 0:
        payload["alerts"] = [
            "This EvalRDL import contains rollout identities, but no scored results "
            "or efficiency measurements. The selected export does not include the "
            "full rollout records needed to build this report."
        ]
    payload["_samples"] = samples
    payload["_attempts"] = {
        "task_id": [str(task) for task in rollouts["task_id"].to_list()],
        "repeat": [int(repeat or 0) for repeat in rollouts["repeat"].to_list()],
        "groups": scored["groups"],
        "values": {
            name: values[name].to_list() for name in CARD_METRICS if name in values
        },
    }
    payload["tool_types"] = (
        _tool_types(tool_calls, scored["groups"], rollouts)
        if has_agent_activity else []
    )
    payload["provenance"] = _provenance(
        rollouts=rollouts,
        scored=scored,
        values=values,
        origins=origins,
        pricing=pricing,
        tool_types=payload["tool_types"],
        has_agent_activity=has_agent_activity,
    )
    return payload


def _has_agent_activity(
    rollouts: pl.DataFrame,
    tool_calls: pl.DataFrame,
) -> bool:
    """A run has agent activity only when at least one tool call was recorded."""
    if "n_tool_calls" in rollouts.columns:
        counts = rollouts["n_tool_calls"].cast(pl.Float64, strict=False).fill_null(0)
        if float(counts.sum()) > 0:
            return True
    if tool_calls.height == 0:
        return False
    if "lane" not in tool_calls.columns:
        return True
    kind = pl.col("lane").fill_null("").str.split("/").list.first().fill_null("")
    return tool_calls.filter(~kind.is_in(list(_NOT_COUNTED))).height > 0


def _present(series: pl.Series | None) -> int:
    return 0 if series is None else int(series.is_not_null().sum())


def _provenance(
    *,
    rollouts: pl.DataFrame,
    scored: dict[str, Any],
    values: dict[str, pl.Series],
    origins: dict[str, tuple[str, str]],
    pricing: dict[str, Any],
    tool_types: list[dict[str, Any]],
    has_agent_activity: bool,
) -> list[dict[str, Any]]:
    """Describe actual coverage and provenance without exposing storage internals."""
    height = rollouts.height
    scored_n = int(scored["counts"]["n_pass"] + scored["counts"]["n_fail"])
    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        label: str,
        *,
        present: int,
        status: str,
        detail: str,
        total: int = height,
    ) -> None:
        rows.append({
            "name": name,
            "label": label,
            "status": status if present else "unavailable",
            "present": present,
            "total": total,
            "detail": detail if present else "No usable value was recorded; this metric is omitted.",
        })

    add(
        "accuracy",
        "Overall accuracy",
        present=scored_n,
        status="derived",
        detail=(
            "Passing scored rollouts divided by all scored rollouts; "
            "reward ≥ 1 is pass and a null reward is unscored."
        ),
    )
    scored_problems = len(_problem_outcomes(rollouts, scored["groups"]))
    total_problems = _n_problems(rollouts)
    add(
        "best_of_n",
        "Best-of-N",
        present=scored_problems,
        total=total_problems,
        status="derived",
        detail="Share of tasks with at least one passing scored rollout.",
    )

    labels = {name: label for name, label, *_ in _METRICS}
    labels.update({
        "answer_tokens": "Answer tokens",
        "output_tokens": "Output tokens",
        "total_tokens": "Total tokens",
        "uncached_tokens": "Uncached input tokens",
        "avg_tool_calls_per_turn": "Tool calls per turn",
        "tool_arg_chars": "Tool argument characters",
        "tool_output_chars": "Tool output characters",
        "prompt_tokens": "Prompt size",
        "final_context_tokens": "Final context length",
        "simulated_cost_usd": "Cost",
    })
    expected = [
        "input_tokens", "cached_tokens", "uncached_tokens", "reasoning_tokens",
        "answer_tokens", "output_tokens", "total_tokens", "n_turns", "n_tool_calls",
        "avg_tool_calls_per_turn", "tool_arg_chars", "tool_output_chars",
        "prompt_tokens", "final_context_tokens", "simulated_cost_usd",
    ]
    if values.get("cached_tokens") is None:
        expected.remove("uncached_tokens")
    if not has_agent_activity:
        expected = [
            name for name in expected
            if name not in {
                "n_turns", "n_tool_calls", "avg_tool_calls_per_turn",
                "tool_arg_chars", "tool_output_chars", "final_context_tokens",
            }
        ]
    derived = {
        "answer_tokens": ("derived", "Answer tokens after classifying completion semantics."),
        "output_tokens": ("derived", "Total model output after classifying completion semantics."),
        "uncached_tokens": ("derived", "Input tokens minus cached input tokens."),
        "avg_tool_calls_per_turn": ("derived", "Tool calls divided by assistant turns."),
        "tool_arg_chars": ("recorded", "Summed recorded tool-argument characters."),
        "tool_output_chars": ("recorded", "Summed recorded tool-result characters."),
        "prompt_tokens": ("derived", "Input tokens of the first model request."),
        "final_context_tokens": ("derived", "Input tokens of the last model request."),
    }
    for name in expected:
        series = values.get(name)
        status, detail = origins.get(name, derived.get(
            name, ("recorded", "Read from recorded EvalRDL values.")
        ))
        if name == "simulated_cost_usd":
            source = str(pricing.get("source") or "simulated_fallback")
            if source == "model_config":
                rate_detail = "user-configured model rates"
            else:
                rates = {**_DEFAULT_PRICING, **pricing}
                rate_detail = (
                    f"stand-in rates (${rates['input_usd_per_million']:g} input, "
                    f"${rates['cached_usd_per_million']:g} cached, "
                    f"${rates['output_usd_per_million']:g} output per million)"
                )
            cached = values.get("cached_tokens")
            cached_note = (
                " Missing cached-token values count as zero."
                if cached is None or _present(cached) < height else ""
            )
            status = "derived"
            detail = (
                f"Calculated from input, cached, and classified output tokens using {rate_detail}."
                f"{cached_note}"
            )
        add(
            name,
            labels[name],
            present=_present(series),
            status=status,
            detail=detail,
        )

    if not has_agent_activity:
        rows.append({
            "name": "agent_activity",
            "label": "Agent activity",
            "status": "not_applicable",
            "present": 0,
            "total": 0,
            "detail": "No agent tool calls were recorded, so the Agent activity section is omitted.",
        })
        return rows

    tool_call_series = values.get("n_tool_calls")
    recorded_calls = (
        float(tool_call_series.fill_null(0).sum()) if tool_call_series is not None else 0.0
    )
    if not tool_types and recorded_calls > 0:
        rows.append({
            "name": "tool_types",
            "label": "Calls by tool type",
            "status": "unavailable",
            "present": 0,
            "total": height,
            "detail": "Tool-call counts were recorded, but tool names were unavailable; this breakdown is omitted.",
        })
    else:
        add(
            "tool_types",
            "Calls by tool type",
            present=_present(tool_call_series),
            status="recorded",
            detail=(
                "Grouped recorded tool calls by tool name."
                if tool_types else "Tool-call counts were recorded and no tool calls occurred."
            ),
        )
    return rows


def _pricing(pricing: dict[str, Any] | None) -> dict[str, Any]:
    rates = dict(_DEFAULT_PRICING)
    if not pricing:
        return rates
    for key in ("input_usd_per_million", "cached_usd_per_million", "output_usd_per_million"):
        value = pricing.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rates[key] = float(value)
    source = pricing.get("source")
    if isinstance(source, str) and source:
        rates["source"] = source
    model_id = pricing.get("model_id")
    if model_id is not None:
        rates["model_id"] = model_id
    return rates


def _sum_total(values: dict[str, pl.Series]) -> pl.Series | None:
    """Add canonical input to the classified total model output."""
    prompt = values.get("input_tokens")
    output = values.get("output_tokens")
    if prompt is None or output is None:
        return None
    prompt_f = prompt.cast(pl.Float64, strict=False)
    output_f = output.cast(pl.Float64, strict=False)
    summed = pl.select(
        pl.when(prompt_f.is_not_null() & output_f.is_not_null())
        .then(prompt_f + output_f)
        .otherwise(None)
    ).to_series()
    return summed if int(summed.is_not_null().sum()) else None


def _uncached_input(values: dict[str, pl.Series]) -> pl.Series | None:
    """Input tokens not served from cache, only where cached tokens were recorded."""
    prompt = values.get("input_tokens")
    cached = values.get("cached_tokens")
    if prompt is None or cached is None:
        return None
    prompt_f = prompt.cast(pl.Float64, strict=False)
    cached_f = cached.cast(pl.Float64, strict=False)
    uncached = pl.select(
        pl.when(prompt_f.is_not_null() & cached_f.is_not_null())
        .then((prompt_f - cached_f).clip(lower_bound=0.0))
        .otherwise(None)
    ).to_series()
    return uncached if int(uncached.is_not_null().sum()) else None


def _note_partial(warnings: list[str], label: str, series: pl.Series, height: int) -> None:
    present = int(series.is_not_null().sum())
    if present < height:
        warnings.append(
            f"EvalRDL recorded {label.lower()} for {present} of {height} "
            "rollouts; the others are omitted from that metric."
        )


def _simulated_cost(
    values: dict[str, pl.Series],
    pricing: dict[str, Any],
    height: int,
) -> pl.Series | None:
    """Price recorded tokens. Missing cached tokens count as zero."""
    prompt = values.get("input_tokens")
    output = values.get("output_tokens")
    if prompt is None or output is None:
        return None
    cached = values.get("cached_tokens")
    if cached is None:
        cached = pl.Series("cached_tokens", [0.0] * height, dtype=pl.Float64)
    else:
        cached = cached.fill_null(0.0)
    prompt_f = prompt.cast(pl.Float64, strict=False)
    output_f = output.cast(pl.Float64, strict=False)
    cached_f = cached.cast(pl.Float64, strict=False)
    uncached = (prompt_f.fill_null(0.0) - cached_f).clip(lower_bound=0.0)
    cost = (
        uncached * float(pricing["input_usd_per_million"])
        + cached_f * float(pricing["cached_usd_per_million"])
        + output_f.fill_null(0.0) * float(pricing["output_usd_per_million"])
    ) / 1_000_000
    series = pl.select(
        pl.when(prompt_f.is_not_null() & output_f.is_not_null())
        .then(cost)
        .otherwise(None)
    ).to_series()
    if int(series.is_not_null().sum()) == 0:
        return None
    return series


def _metric_series(
    table: str,
    column: str,
    label: str,
    rollouts: pl.DataFrame,
    events: pl.DataFrame,
) -> tuple[pl.Series | None, str | None]:
    if table == "rollouts":
        return _aligned(rollouts, column), None
    return _call_totals(events, column, label, rollouts)


def _aligned(frame: pl.DataFrame, column: str) -> pl.Series | None:
    if column not in frame.columns:
        return None
    series = frame[column]
    if series.null_count() == frame.height:
        return None
    return series.cast(pl.Float64, strict=False)


def _with_token_fallback(
    name: str,
    series: pl.Series | None,
    events: pl.DataFrame,
    rollouts: pl.DataFrame,
    *,
    selected_model: str | None,
) -> tuple[pl.Series | None, int]:
    """Fill missing canonical values from selected-model requests."""
    column = _TOKEN_EVENT_COLUMN[name]
    requests, _reason = _call_totals(
        events,
        column,
        name,
        rollouts,
        only_kind="model_call",
        selected_model=selected_model,
    )
    series, request_added = _fill_missing(series, requests)
    return series, request_added


def _fill_missing(
    primary: pl.Series | None,
    fallback: pl.Series | None,
) -> tuple[pl.Series | None, int]:
    if fallback is None:
        return primary, 0
    if primary is None:
        return fallback, _present(fallback)
    before = _present(primary)
    primary_f = primary.cast(pl.Float64, strict=False)
    fallback_f = fallback.cast(pl.Float64, strict=False)
    merged = pl.select(
        pl.when(primary_f.is_not_null()).then(primary_f).otherwise(fallback_f)
    ).to_series()
    return merged, _present(merged) - before


def _derived(
    values: dict[str, pl.Series],
    catalog: list[tuple],
    events: pl.DataFrame,
    rollouts: pl.DataFrame,
    tool_calls: pl.DataFrame,
    parts: pl.DataFrame,
    warnings: list[str],
    origins: dict[str, tuple[str, str]],
    *,
    include_agent_activity: bool,
    selected_model: str | None,
) -> None:
    completion = values.get("completion_tokens")
    reasoning = values.get("reasoning_tokens")
    if completion is not None:
        rollout_completion = _aligned(rollouts, "completion_tokens")
        model_completion, _reason = _call_totals(
            events,
            "completion_tokens",
            "completion tokens",
            rollouts,
            only_kind="model_call",
            selected_model=selected_model,
        )
        answer, output, classes = _classify_completion_tokens(
            completion,
            reasoning,
            rollout_completion,
            model_completion,
        )
        detail = (
            "Compared rollout completion, reasoning, and selected-model completion per rollout: "
            f"{classes['answer_only']} answer-only, "
            f"{classes['inclusive']} reasoning-inclusive, "
            f"{classes['zero_reasoning']} zero-reasoning, "
            f"{classes['reasoning_unrecorded']} without recorded reasoning, and "
            f"{classes['unresolved']} unresolved."
        )
        if answer is not None:
            values["answer_tokens"] = answer
            catalog.append(("answer_tokens", "Answer tokens", "tokens", "", ""))
            origins["answer_tokens"] = ("derived", detail)
        if output is not None:
            values["output_tokens"] = output
            catalog.append(("output_tokens", "Output tokens", "tokens", "", ""))
            origins["output_tokens"] = ("derived", detail)
        if classes["unresolved"]:
            warnings.append(
                f"Completion-token semantics were unresolved for {classes['unresolved']} "
                "rollouts; their answer split was omitted. Selected-model completion "
                "was retained as total output where available."
            )
    if completion is not None and reasoning is None:
        warnings.append(
            "EvalRDL did not record reasoning tokens beside completion, so answer tokens were not split out."
        )
    if include_agent_activity:
        turns = values.get("n_turns")
        calls = values.get("n_tool_calls")
        if turns is not None and calls is not None:
            turns_f = turns.cast(pl.Float64, strict=False)
            calls_f = calls.cast(pl.Float64, strict=False)
            average = pl.select(
                pl.when(turns_f.is_not_null() & calls_f.is_not_null() & (turns_f > 0))
                .then(calls_f / turns_f)
                .otherwise(None)
            ).to_series()
            if int(average.is_not_null().sum()):
                values["avg_tool_calls_per_turn"] = average
                catalog.append((
                    "avg_tool_calls_per_turn",
                    "Tool calls per turn",
                    "calls/turn",
                    "",
                    "",
                ))
        arguments = _sum_column(
            tool_calls, "args_chars", rollouts, counted_only=True,
        )
        if arguments is not None:
            values["tool_arg_chars"] = arguments
            catalog.append((
                "tool_arg_chars", "Tool argument characters", "characters", "", "",
            ))
        else:
            warnings.append(
                "EvalRDL did not record tool-argument size, so tool argument characters "
                "were not computed."
            )
        output_chars = _sum_column(
            parts.filter(pl.col("channel") == "tool_result")
            if "channel" in parts.columns else parts.clear(),
            "chars",
            rollouts,
            counted_only=True,
        )
        if output_chars is not None:
            values["tool_output_chars"] = output_chars
            catalog.append((
                "tool_output_chars", "Tool output characters", "characters", "", "",
            ))
        else:
            warnings.append(
                "EvalRDL did not record tool-result size, so tool output characters were not computed."
            )
    prompt_size, final_context = _context_sizes(
        events, rollouts, selected_model=selected_model,
    )
    if prompt_size is not None:
        values["prompt_tokens"] = prompt_size
        catalog.append(("prompt_tokens", "Prompt size", "tokens", "", ""))
    if include_agent_activity and final_context is not None:
        values["final_context_tokens"] = final_context
        catalog.append(("final_context_tokens", "Final context length", "tokens", "", ""))
    if prompt_size is None and final_context is None:
        warnings.append(
            "EvalRDL did not record one unambiguous parent request sequence, "
            "so prompt size and final context were not computed."
        )


def _classify_completion_tokens(
    completion: pl.Series,
    reasoning: pl.Series | None,
    rollout_completion: pl.Series | None,
    model_completion: pl.Series | None,
) -> tuple[pl.Series | None, pl.Series | None, dict[str, int]]:
    """Split answer and total output only where recorded token identities establish it."""
    size = len(completion)
    reasoning_values = reasoning.to_list() if reasoning is not None else [None] * size
    rollout_values = (
        rollout_completion.to_list() if rollout_completion is not None else [None] * size
    )
    model_values = model_completion.to_list() if model_completion is not None else [None] * size
    answers: list[float | None] = []
    outputs: list[float | None] = []
    classes = {
        "answer_only": 0,
        "inclusive": 0,
        "zero_reasoning": 0,
        "reasoning_unrecorded": 0,
        "unresolved": 0,
    }
    for completion_value, reasoning_value, rollout_value, model_value in zip(
        completion.to_list(),
        reasoning_values,
        rollout_values,
        model_values,
        strict=True,
    ):
        if completion_value is None:
            answers.append(None)
            outputs.append(None)
            classes["unresolved"] += 1
            continue
        completion_number = float(completion_value)
        model_number = None if model_value is None else float(model_value)
        if reasoning_value is None:
            answers.append(None)
            outputs.append(model_number if model_number is not None else completion_number)
            classes["reasoning_unrecorded"] += 1
            continue
        reasoning_number = float(reasoning_value)
        if reasoning_number == 0:
            if model_number is None or model_number == completion_number:
                answers.append(completion_number)
                outputs.append(completion_number)
                classes["zero_reasoning"] += 1
            else:
                answers.append(None)
                outputs.append(model_number)
                classes["unresolved"] += 1
            continue
        rollout_number = None if rollout_value is None else float(rollout_value)
        if (
            rollout_number is not None
            and model_number is not None
            and model_number == rollout_number + reasoning_number
        ):
            answers.append(rollout_number)
            outputs.append(model_number)
            classes["answer_only"] += 1
        elif (
            model_number is not None
            and model_number == completion_number
            and completion_number >= reasoning_number
        ):
            answers.append(completion_number - reasoning_number)
            outputs.append(completion_number)
            classes["inclusive"] += 1
        else:
            answers.append(None)
            outputs.append(model_number)
            classes["unresolved"] += 1
    answer = pl.Series("answer_tokens", answers, dtype=pl.Float64)
    output = pl.Series("output_tokens", outputs, dtype=pl.Float64)
    return (
        answer if _present(answer) else None,
        output if _present(output) else None,
        classes,
    )


def _context_sizes(
    events: pl.DataFrame,
    rollouts: pl.DataFrame,
    *,
    selected_model: str | None,
) -> tuple[pl.Series | None, pl.Series | None]:
    """First and last request sizes from one identifiable parent request sequence."""
    request_prompt, request_final = _model_request_context_sizes(
        events, rollouts, selected_model=selected_model,
    )
    transcript_prompt, transcript_final = _parent_context_sizes(events, rollouts)
    prompt, _added = _fill_missing(request_prompt, transcript_prompt)
    final, _added = _fill_missing(request_final, transcript_final)
    return prompt, final


def _parent_context_sizes(
    events: pl.DataFrame,
    rollouts: pl.DataFrame,
) -> tuple[pl.Series | None, pl.Series | None]:
    if events.height == 0 or "prompt_tokens" not in events.columns or "request_idx" not in events.columns:
        return None, None
    kind = pl.col("lane").fill_null("").str.split("/").list.first().fill_null("")
    recorded = events.filter(
        pl.col("prompt_tokens").is_not_null() & pl.col("request_idx").is_not_null()
    ).with_columns(kind.alias("_kind"))
    recorded = recorded.filter(
        ~pl.col("_kind").is_in([*_NOT_COUNTED, "subagent"])
    )
    if recorded.height == 0:
        return None, None
    keys = [key for key in _ROLL_KEYS if key in recorded.columns and key in rollouts.columns]
    accounted = _with_account(recorded)
    order = ["event_idx"] if "event_idx" in accounted.columns else []
    calls = accounted.sort([*keys, "lane", *order], nulls_last=True).group_by(
        [*keys, "_account", "lane", "request_idx"], maintain_order=True,
    ).agg(
        pl.col("prompt_tokens").first(),
        pl.col("event_idx").min().alias("_first_event")
        if "event_idx" in accounted.columns else pl.lit(0).alias("_first_event"),
    )
    per_account = calls.sort(
        [*keys, "_account", "lane", "request_idx", "_first_event"], nulls_last=True,
    ).group_by([*keys, "_account"], maintain_order=True).agg(
        pl.col("lane").n_unique().alias("_n_lanes"),
        pl.len().alias("_n_calls"),
        pl.col("prompt_tokens").first().alias("prompt_tokens"),
        pl.col("prompt_tokens").last().alias("final_context_tokens"),
    ).filter(pl.col("_n_lanes") == 1)
    if per_account.height == 0:
        return None, None
    edges = _choose_larger_account(
        per_account, keys, "_n_calls",
    ).select(
        [*keys, "prompt_tokens", "final_context_tokens"]
    )
    return (
        _join_metric(edges, "prompt_tokens", keys, rollouts),
        _join_metric(edges, "final_context_tokens", keys, rollouts),
    )


def _model_request_context_sizes(
    events: pl.DataFrame,
    rollouts: pl.DataFrame,
    *,
    selected_model: str | None,
) -> tuple[pl.Series | None, pl.Series | None]:
    """Context endpoints from one ordered selected-model request lane."""
    if events.height == 0 or "prompt_tokens" not in events.columns or "lane" not in events.columns:
        return None, None
    kind = pl.col("lane").fill_null("").str.split("/").list.first().fill_null("")
    recorded = events.with_columns(
        kind.alias("_kind")
    ).filter(pl.col("_kind") == "model_call")
    recorded = _selected_model_requests(recorded, selected_model).filter(
        pl.col("prompt_tokens").is_not_null()
    )
    if recorded.height == 0 or "event_idx" not in recorded.columns:
        return None, None
    keys = [key for key in _ROLL_KEYS if key in recorded.columns and key in rollouts.columns]
    per_rollout = recorded.sort([*keys, "lane", "event_idx"], nulls_last=True).group_by(
        keys, maintain_order=True,
    ).agg(
        pl.col("lane").n_unique().alias("_n_lanes"),
        pl.col("prompt_tokens").first().alias("prompt_tokens"),
        pl.col("prompt_tokens").last().alias("final_context_tokens"),
    ).filter(pl.col("_n_lanes") == 1)
    if per_rollout.height == 0:
        return None, None
    return (
        _join_metric(per_rollout, "prompt_tokens", keys, rollouts),
        _join_metric(per_rollout, "final_context_tokens", keys, rollouts),
    )


def _sum_column(
    frame: pl.DataFrame,
    column: str,
    rollouts: pl.DataFrame,
    *,
    counted_only: bool,
) -> pl.Series | None:
    if frame.height == 0 or column not in frame.columns:
        return None
    recorded = frame.filter(pl.col(column).is_not_null())
    if recorded.height == 0 or "lane" not in recorded.columns:
        return None
    if counted_only:
        kind = pl.col("lane").fill_null("").str.split("/").list.first().fill_null("")
        recorded = recorded.with_columns(kind.alias("_kind")).filter(
            ~pl.col("_kind").is_in(list(_NOT_COUNTED))
        )
    if recorded.height == 0:
        return None
    keys = [key for key in _ROLL_KEYS if key in recorded.columns and key in rollouts.columns]
    per_account = _with_account(recorded).group_by([*keys, "_account"]).agg(
        pl.col(column).sum().alias(column)
    )
    totals = per_account.group_by(keys).agg(pl.col(column).max().alias(column))
    return _join_metric(totals, column, keys, rollouts)


def _with_account(frame: pl.DataFrame) -> pl.DataFrame:
    """Label main and alternate transcript accounts using EvalRDL's lane contract."""
    kind = pl.col("lane").fill_null("").str.split("/").list.first().fill_null("")
    return frame.with_columns(kind.alias("_kind")).with_columns(
        pl.when(pl.col("_kind").is_in(list(_ALTERNATE)))
        .then(pl.col("_kind"))
        .otherwise(pl.lit(""))
        .alias("_account")
    )


def _choose_larger_account(
    per_account: pl.DataFrame,
    keys: list[str],
    size_column: str,
) -> pl.DataFrame:
    """Choose one complete account per rollout, preferring main on equal coverage."""
    ranked = per_account.with_columns(
        (pl.col("_account") != "").cast(pl.Int8).alias("_alternate")
    ).sort(
        [*keys, size_column, "_alternate", "_account"],
        descending=[False] * len(keys) + [True, False, False],
    )
    return ranked.group_by(keys, maintain_order=True).first()


def _join_metric(
    totals: pl.DataFrame, column: str, keys: list[str], rollouts: pl.DataFrame,
) -> pl.Series | None:
    indexed = rollouts.with_row_index("_row")
    joined = indexed.select(["_row", *keys]).join(totals, on=keys, how="left").sort("_row")
    series = joined[column].cast(pl.Float64, strict=False)
    if int(series.is_not_null().sum()) == 0:
        return None
    return series


def _tool_types(
    tool_calls: pl.DataFrame, groups: list[str], rollouts: pl.DataFrame,
) -> list[dict[str, Any]]:
    if tool_calls.height == 0 or "tool_name" not in tool_calls.columns:
        return []
    kind = (
        pl.col("lane").fill_null("").str.split("/").list.first().fill_null("")
        if "lane" in tool_calls.columns else pl.lit("")
    )
    recorded = tool_calls.with_columns(kind.alias("_kind")).filter(
        ~pl.col("_kind").is_in(list(_NOT_COUNTED))
    )
    if recorded.height == 0:
        return []
    keys = [key for key in _ROLL_KEYS if key in recorded.columns and key in rollouts.columns]
    names = _with_account(recorded).with_columns(
        pl.col("tool_name").fill_null("tool").alias("tool_name")
    )
    account_sizes = names.group_by([*keys, "_account"]).agg(pl.len().alias("_n_calls"))
    selected = _choose_larger_account(account_sizes, keys, "_n_calls")
    names = names.join(
        selected.select([*keys, "_account"]),
        on=[*keys, "_account"],
        how="inner",
    )
    indexed = rollouts.with_row_index("_row").with_columns(pl.Series("group", groups))
    joined = names.join(indexed.select(["_row", *keys, "group"]), on=keys, how="inner")
    counts: dict[str, dict[str, int]] = {}
    for name, group in zip(joined["tool_name"].to_list(), joined["group"].to_list(), strict=True):
        bucket = counts.setdefault(str(name), {"overall": 0, "pass": 0, "fail": 0})
        bucket["overall"] += 1
        if group in {"pass", "fail"}:
            bucket[group] += 1
    return [
        {"name": name, "overall": bucket["overall"], "pass": bucket["pass"], "fail": bucket["fail"]}
        for name, bucket in sorted(counts.items(), key=lambda item: item[1]["overall"], reverse=True)
    ]


def _call_totals(
    events: pl.DataFrame,
    column: str,
    label: str,
    rollouts: pl.DataFrame,
    *,
    only_kind: str | None = None,
    selected_model: str | None = None,
) -> tuple[pl.Series | None, str | None]:
    """One value per indexed request or unindexed row, then the larger account."""
    if events.height == 0 or column not in events.columns or "request_idx" not in events.columns:
        return None, None
    if events[column].null_count() == events.height:
        return None, None
    # Resolve model identity before dropping requests with missing usage.
    recorded = events
    keys = [key for key in _ROLL_KEYS if key in recorded.columns and key in rollouts.columns]
    if not keys or "lane" not in recorded.columns:
        return None, None
    kind = pl.col("lane").fill_null("").str.split("/").list.first().fill_null("")
    recorded = recorded.with_columns(kind.alias("_kind"))
    if only_kind is None:
        recorded = recorded.filter(~pl.col("_kind").is_in(list(_NOT_COUNTED)))
    else:
        recorded = recorded.filter(pl.col("_kind") == only_kind)
        if only_kind == "model_call":
            recorded = _selected_model_requests(recorded, selected_model)
    recorded = recorded.filter(pl.col(column).is_not_null())
    if recorded.height == 0:
        return None, (
            f"EvalRDL has no unambiguous {label.lower()} source for the selected model, "
            "so that metric was not computed."
            if only_kind == "model_call"
            else (
                f"EvalRDL recorded {label.lower()} only on lanes that copy the rollout, "
                "so that metric was not computed."
            )
        )
    counted = recorded.with_row_index("_usage_row").with_columns(
        pl.when(pl.col("_kind").is_in(list(_ALTERNATE)))
        .then(pl.col("_kind"))
        .otherwise(pl.lit(""))
        .alias("_account")
    )
    event_identity = (
        pl.when(pl.col("event_idx").is_not_null())
        .then(pl.col("event_idx").cast(pl.String))
        .otherwise(pl.col("_usage_row").cast(pl.String))
        if "event_idx" in counted.columns
        else pl.col("_usage_row").cast(pl.String)
    )
    counted = counted.with_columns(
        pl.when(pl.col("request_idx").is_not_null())
        .then(pl.lit("request:") + pl.col("request_idx").cast(pl.String))
        .otherwise(pl.lit("event:") + event_identity)
        .alias("_usage_key")
    )
    order = ["event_idx"] if "event_idx" in counted.columns else []
    counted = counted.sort([*keys, *order], nulls_last=True)
    calls = counted.group_by(
        [*keys, "lane", "_usage_key"], maintain_order=True
    ).agg(pl.col(column).first(), pl.col("_account").first())
    per_account = calls.group_by([*keys, "_account"]).agg(pl.col(column).sum())
    totals = per_account.group_by(keys).agg(pl.col(column).max().alias(column))
    indexed = rollouts.with_row_index("_row")
    joined = indexed.select(["_row", *keys]).join(totals, on=keys, how="left").sort("_row")
    series = joined[column].cast(pl.Float64, strict=False)
    if int(series.is_not_null().sum()) == 0:
        return None, None
    return series, None


def _selected_model_requests(
    frame: pl.DataFrame,
    selected_model: str | None,
) -> pl.DataFrame:
    """Select one model's request rows, refusing an ambiguous multi-model capture."""
    if frame.height == 0 or "model_name" not in frame.columns:
        return frame
    named = frame.filter(pl.col("model_name").is_not_null())
    if selected_model and named.height:
        exact = named.filter(
            pl.col("model_name").cast(pl.String).str.to_lowercase()
            == selected_model.lower()
        )
        return exact
    names = named["model_name"].cast(pl.String).unique().to_list() if named.height else []
    has_unnamed = frame["model_name"].null_count() > 0
    if not names:
        return frame
    if len(names) == 1 and not has_unnamed:
        return named
    return frame.clear()


def _scored(rollouts: pl.DataFrame) -> dict[str, Any]:
    height = rollouts.height
    if "reward" not in rollouts.columns or rollouts["reward"].null_count() == height:
        return {
            "groups": ["unscored"] * height,
            "counts": {
                "n_pass": 0,
                "n_fail": 0,
                "n_unscored": height,
                "n_problems": _n_problems(rollouts),
                "best_of_n": None,
                "best_of_label": "best-of-n",
            },
        }
    reward = rollouts["reward"].cast(pl.Float64, strict=False)
    groups = [efficiency_outcome(value) for value in reward.to_list()]
    n_pass = groups.count("pass")
    n_fail = groups.count("fail")
    problems = _problem_outcomes(rollouts, groups)
    best = [any(ok for _repeat, ok in rows) for rows in problems.values() if rows]
    repeats = {repeat for rows in problems.values() for repeat, _ok in rows}
    width = max(repeats) + 1 if repeats else 1
    return {
        "groups": groups,
        "counts": {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_unscored": groups.count("unscored"),
            "n_problems": len(problems),
            "best_of_n": (sum(best) / len(best)) if best else None,
            "best_of_label": f"best-of-{width}",
        },
    }


def _problem_outcomes(rollouts: pl.DataFrame, groups: list[str]) -> dict[str, list[tuple[int, bool]]]:
    task_column = "task_id" if "task_id" in rollouts.columns else None
    repeat_column = "repeat" if "repeat" in rollouts.columns else None
    tasks = rollouts[task_column].to_list() if task_column else list(range(rollouts.height))
    repeats = rollouts[repeat_column].to_list() if repeat_column else [0] * rollouts.height
    problems: dict[str, list[tuple[int, bool]]] = {}
    for task, repeat, group in zip(tasks, repeats, groups, strict=True):
        if group == "unscored":
            continue
        try:
            repeat_idx = int(repeat or 0)
        except (TypeError, ValueError):
            repeat_idx = 0
        problems.setdefault(str(task), []).append((repeat_idx, group == "pass"))
    return problems


def _n_problems(rollouts: pl.DataFrame) -> int:
    if "task_id" not in rollouts.columns:
        return rollouts.height
    return rollouts["task_id"].n_unique()


def _metric_summary(
    name: str,
    label: str,
    unit: str,
    series: pl.Series,
    groups: list[str],
    rollouts: pl.DataFrame,
    *,
    source: str = "evalrdl",
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    from collections import defaultdict

    from autofa.reports.efficiency_card import _problem_cluster_permutation_test

    n_items = rollouts.height
    tasks = rollouts["task_id"].to_list() if "task_id" in rollouts.columns else list(range(n_items))
    repeats = rollouts["repeat"].to_list() if "repeat" in rollouts.columns else [0] * n_items
    buckets = {"overall": [], "pass": [], "fail": []}
    clusters: dict[tuple[str, str, str], list[tuple[int, float, bool]]] = defaultdict(list)
    for task, repeat, value, group in zip(tasks, repeats, series.to_list(), groups, strict=True):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        number = float(value)
        buckets["overall"].append(number)
        if group not in {"pass", "fail"}:
            continue
        buckets[group].append(number)
        try:
            repeat_idx = int(repeat or 0)
        except (TypeError, ValueError):
            repeat_idx = 0
        clusters[("evalrdl", "", str(task))].append((repeat_idx, number, group == "pass"))
    metric = {
        "name": name,
        "label": label,
        "unit": unit,
        "coverage": len(buckets["overall"]) / n_items if n_items else 0.0,
        "sources": {source: len(buckets["overall"])},
        "overall": _stats(buckets["overall"]),
        "pass": _stats(buckets["pass"]),
        "fail": _stats(buckets["fail"]),
    }
    test = _problem_cluster_permutation_test(clusters, metric=name)
    if test is not None:
        metric["permutation_test"] = test
    return metric, buckets


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    def quantile(q: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[index]

    total = sum(values)
    return {
        "n": len(values),
        "total": total,
        "mean": total / len(values),
        "min": ordered[0],
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "p90": quantile(0.9),
        "max": ordered[-1],
    }
