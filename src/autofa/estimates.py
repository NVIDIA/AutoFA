# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime planning from measured, comparable discovery rounds (no price guesses)."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from autofa.config import Config, runs_root
from autofa.harness.settings import behavior_judge_settings, harness_model

log = logging.getLogger(__name__)
SCHEMA_VERSION = 2
MIN_COMPARABLE_RUNS = 3


@dataclass(frozen=True)
class Workload:
    rows: int = 0
    mean_chars: float = 0
    mean_steps: float = 0
    sampled_rows: int = 0


def profile_workload(paths: list[Path], model_id: str = "*",
                     benchmarks: list[str] | None = None) -> Workload:
    """Count selected rows; bound trace reads to 48 samples spread over row groups.

    Row groups/files receive proportional sample slots, so the beginning of one
    dataset does not stand in for every selected dataset. These are sample sizes,
    not a census or a fitted runtime model.
    """
    import pyarrow.dataset as ds

    from autofa.data.trajectory import parse_trajectory

    groups = []
    rows = 0
    for path in dict.fromkeys(paths):
        data = ds.dataset(path, format="parquet")
        predicate = None
        if model_id not in {"*", "ALL", ""} and "model_id" in data.schema.names:
            predicate = ds.field("model_id") == model_id
        if benchmarks and "benchmark" in data.schema.names:
            bench = ds.field("benchmark").isin(benchmarks)
            predicate = bench if predicate is None else predicate & bench
        for fragment in data.get_fragments(filter=predicate):
            for group in fragment.split_by_row_group(filter=predicate):
                count = group.count_rows(filter=predicate)
                rows += count
                if count:
                    groups.append((group, predicate, count))
    # Evenly spaced slots across selected rows allocate samples by group size.
    n = min(rows, 48)
    slots = [int((i + 0.5) * rows / n) for i in range(n)]
    chars, steps = [], []
    offset = 0
    for group, predicate, count in groups:
        quota = sum(offset <= slot < offset + count for slot in slots)
        offset += count
        if not quota:
            continue
        names = group.physical_schema.names
        columns = [name for name in ("prompt_text", "response_text", "reasoning_text",
                                    "generation") if name in names]
        if not columns:
            continue
        sample = group.scanner(columns=columns, filter=predicate, batch_size=8,
                               batch_readahead=0, fragment_readahead=0).head(quota)
        for row in sample.to_pylist():
            text = row.get("response_text") or row.get("generation") or ""
            chars.append(sum(len(str(value or "")) for value in row.values()))
            parsed = parse_trajectory(text)
            steps.append(parsed.n_steps if parsed.ok else 1)
    return Workload(rows, mean(chars) if chars else 0, mean(steps) if steps else 0, len(chars))


def model_signature(cfg: Config) -> dict[str, Any]:
    judge = behavior_judge_settings(cfg)
    return {
        "harness": cfg.harness,
        "model": harness_model(cfg.harness, cfg.model, cfg=cfg),
        "effort": cfg.effort, "thinking": cfg.thinking,
        "thinking_budget": cfg.thinking_budget_tokens,
        "judge_model": judge["model"], "judge_effort": judge["effort"],
        "judge_thinking": judge["thinking"], "judge_budget": judge["thinking_budget_tokens"],
        "judge_harness": judge["harness"], "judge_workers": cfg.judge_workers,
        "feature_model": cfg.feature_model, "llm_eval_workers": cfg.llm_eval_workers,
        "eval_limit": cfg.default_eval_limit, "llm_eval_sample": cfg.llm_eval_sample,
        "min_failure_modes": cfg.min_failure_modes,
    }


def _minutes(seconds: float) -> int:
    return max(1, math.ceil(seconds / 60))


def _duration(minutes: int, *, rounding: str = "nearest") -> str:
    if minutes < 60:
        return f"{minutes} min"
    hours = (math.floor(minutes / 60) if rounding == "down" else
             math.ceil(minutes / 60) if rounding == "up" else math.floor(minutes / 60 + 0.5))
    return f"{hours}h"


def _time_range(low_minutes: int, high_minutes: int) -> str:
    low = _duration(low_minutes, rounding="down")
    high = _duration(high_minutes, rounding="up")
    return low if low == high else f"{low}–{high}"


@dataclass(frozen=True)
class Estimate:
    low_minutes: int
    high_minutes: int
    basis: str
    median_minutes: int = 0
    rough: bool = False

    @property
    def time_label(self) -> str:
        if not self.high_minutes:
            return "not enough comparable timing data"
        if self.rough:
            return f"rough ~{_time_range(self.low_minutes, self.high_minutes)}"
        return (f"~{_duration(self.median_minutes)} "
                f"(observed {_time_range(self.low_minutes, self.high_minutes)})")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_rows(path: Path, *, offset: int = 0) -> list[dict]:
    try:
        rows = []
        with path.open() as stream:
            stream.seek(offset)
            for line in stream:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
                except ValueError:
                    continue
        return rows
    except OSError:
        return []


@dataclass
class RoundProgress:
    completed: dict[int, float] = field(default_factory=dict)
    round_index: int = 0
    current_seconds: float = 0
    current_valid: bool = True


def round_progress(events: list[dict], *, start_round: int = 0,
                   now: float | None = None) -> RoundProgress:
    """Only advance_round closes a round; turn completion alone never does.

    Sum active harness intervals, excluding time between turns. Failed or
    interrupted attempts invalidate their round. Overlaps, missing starts,
    nonsequential boundaries and backward timestamps invalidate the log.
    """
    progress = RoundProgress(round_index=start_round)
    active = None
    last_ts = None
    for row in events:
        ctx = row.get("context") or {}
        state = ctx.get("turn_status") if row.get("source") == "harness" else None
        boundary = row.get("source") == "advance_round"
        if not state and not boundary:
            continue
        try:
            ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00")).timestamp()
        except (KeyError, TypeError, ValueError):
            return RoundProgress(current_valid=False)
        if last_ts is not None and ts < last_ts:
            return RoundProgress(current_valid=False)
        last_ts = ts
        if state == "started":
            if active is not None:
                return RoundProgress(current_valid=False)
            active = ts
        elif boundary:
            if active is None or ctx.get("round") != progress.round_index + 1:
                return RoundProgress(current_valid=False)
            progress.current_seconds += ts - active
            if progress.current_valid and progress.current_seconds > 0:
                progress.completed[progress.round_index] = progress.current_seconds
            progress.round_index += 1
            progress.current_seconds = 0
            progress.current_valid = True
            active = ts
        elif state:
            if active is None:
                return RoundProgress(current_valid=False)
            progress.current_seconds += ts - active
            active = None
            if state != "finished":
                progress.current_valid = False
    if active is not None and now is not None:
        progress.current_seconds += max(0, now - active)
    return progress


def active_discovery_seconds(events: list[dict], *, now: float) -> float | None:
    """Sum retained harness intervals across turns, excluding time spent paused.

    Interrupted/failed turns still consumed time. An unmatched older start is
    discarded when a new start arrives: a crashed process must not turn the
    overnight gap before Resume into active discovery time.
    """
    total = 0.0
    active = None
    last_ts = None
    observed = False
    for row in events:
        if row.get("source") != "harness":
            continue
        context = row.get("context")
        state = context.get("turn_status") if isinstance(context, dict) else None
        if state not in {"started", "finished", "interrupted", "failed", "incomplete"}:
            continue
        try:
            ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00")).timestamp()
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if last_ts is not None and ts < last_ts:
            continue
        last_ts = ts
        if state == "started":
            active = ts
            observed = True
        elif active is not None:
            total += max(0, ts - active)
            active = None
    if active is not None:
        total += max(0, now - active)
    return total if observed else None


def historical_samples(*, exclude_directory: Path | None = None) -> list[dict]:
    """Keep round measurements, not inferred round counts or token-priced guesses."""
    baseline = _read_json(Path(__file__).with_name("estimate_baselines.json"))
    samples = (baseline.get("samples", [])
               if baseline.get("schema_version") == SCHEMA_VERSION else [])
    samples = list(samples)
    for path in sorted(runs_root().glob("*/estimate_context.json"), reverse=True)[:100]:
        if exclude_directory is not None and path.parent.resolve() == exclude_directory.resolve():
            continue  # This run supplies live calibration, never its own baseline.
        context = _read_json(path)
        if context.get("schema_version") != SCHEMA_VERSION or context.get("mixed_settings"):
            continue
        progress = round_progress(_read_rows(path.with_name("status_log.jsonl"),
                                            offset=context.get("status_offset", 0)),
                                  start_round=context.get("start_round", 0))
        if progress.completed:
            samples.append({**context, "round_seconds": {
                str(k): v for k, v in progress.completed.items()}})
    return samples


def _nearby(work: Workload, other: dict) -> bool:
    # Eligibility gates only: never multiply durations by an invented workload law.
    # Require every dimension within 25%; short and deep traces cannot cancel out.
    for key in ("rows", "mean_chars", "mean_steps"):
        a, b = getattr(work, key), other.get(key, 0)
        if (not isinstance(b, (int, float)) or not math.isfinite(b) or b <= 0
                or not 0.8 <= a / b <= 1.25):
            return False
    return work.sampled_rows > 0 and other.get("sampled_rows", 0) > 0


def _valid_duration(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


class Estimator:
    def __init__(self, cfg: Config, workload: Workload, samples: list[dict] | None = None,
                 *, signature: dict | None = None, exclude_directory: Path | None = None):
        self.workload = workload
        self.signature = model_signature(cfg) if signature is None else signature
        self.samples = (historical_samples(exclude_directory=exclude_directory)
                        if samples is None else samples)

    def estimate(self, rounds: int, top_x: int = 5) -> Estimate:
        if rounds < 1:
            return Estimate(0, 0, "Choose at least one round")
        if not self.workload.rows or not self.workload.sampled_rows:
            return self._rough_estimate(rounds, top_x)
        candidates = [s for s in self.samples if s.get("signature") == self.signature
                      and s.get("top_x") == top_x and _nearby(self.workload, s.get("workload", {}))]
        totals = []
        for sample in candidates:
            observed = sample.get("round_seconds", {})
            if rounds < 1 or len(observed) < rounds:
                continue
            durations = [observed.get(str(i)) for i in range(rounds)]
            if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in durations):
                totals.append(sum(durations))
        if len(totals) < MIN_COMPARABLE_RUNS:
            return self._rough_estimate(rounds, top_x)
        return Estimate(_minutes(min(totals)), _minutes(max(totals)),
                        f"Median and observed range of {len(totals)} runs with matching models/"
                        "settings and similar log size/depth; extra rounds excluded",
                        _minutes(median(totals)))

    def _rough_estimate(self, rounds: int, top_x: int) -> Estimate:
        """A labeled historical planning range, never an asserted model speed.

        Prefer matching settings, then discovery model, among similar workloads.
        If nothing matches, expose the mismatch and use the full reference range.
        Linear round scaling is a rough assumption, not a fitted performance law.
        """
        usable = [s for s in self.samples if any(_valid_duration(v)
                  for v in s.get("round_seconds", {}).values())]
        if not usable:
            return Estimate(0, 0, "No recorded round timings; live timing will calibrate this run")
        nearby = [s for s in usable if s.get("top_x") == top_x
                  and _nearby(self.workload, s.get("workload", {}))]
        exact = [s for s in nearby if s.get("signature") == self.signature]
        same_model = [s for s in nearby if all(
            s.get("signature", {}).get(k) == self.signature.get(k) for k in ("harness", "model")
        )]
        candidates = exact or same_model or nearby or usable
        values = [v * rounds for s in candidates for v in s["round_seconds"].values()
                  if _valid_duration(v)]
        if len(values) < 2:
            return Estimate(0, 0, "Only one comparable round; live timing needs more observations")
        # Round outwards so sparse history does not imply minute-level precision.
        step = 5 if max(values) < 3600 else 15
        low = max(1, math.floor(min(values) / (60 * step)) * step)
        high = max(low, math.ceil(max(values) / (60 * step)) * step)
        typical = max(1, round(median(values) / (60 * step)) * step)
        if exact:
            basis = "Limited matching history"
        elif same_model:
            basis = "Same discovery model; other settings uncalibrated"
        elif nearby:
            basis = "Similar logs; selected models uncalibrated"
        else:
            basis = "Reference logs; selected workload/models uncalibrated"
        basis += (f" · rough range from {len(candidates)} run(s), {len(values)} measured rounds. "
                  "Assumes similar pace per round; live timing refines it.")
        return Estimate(low, high, basis, typical, rough=True)


def prepare_estimator(cfg: Config, paths: list[Path], model_id: str = "*",
                      benchmarks: list[str] | None = None) -> Estimator:
    try:
        workload = profile_workload(paths, model_id, benchmarks)
    except (OSError, ValueError):
        log.warning("Could not profile workload for planning", exc_info=True)
        workload = Workload()
    return Estimator(cfg, workload)


def save_estimate_context(run, cfg: Config) -> None:
    """Save measured settings and workload, never prompts or credentials."""
    from autofa.util.atomic import atomic_write_json

    path = run.run_dir() / "estimate_context.json"
    signature = model_signature(cfg)
    signature.update(eval_limit=run.eval_limit, min_failure_modes=run.min_failure_modes)
    previous = _read_json(path)
    if previous.get("schema_version") == SCHEMA_VERSION:
        if previous.get("signature") != signature or previous.get("top_x") != run.top_x:
            previous["mixed_settings"] = True
            atomic_write_json(path, previous)
        return
    try:
        workload = profile_workload([Path(p) for p in run.parquet_paths],
                                    run.model_id, run.resolved_benchmarks())
        status = path.with_name("status_log.jsonl")
        atomic_write_json(path, {"schema_version": SCHEMA_VERSION, "signature": signature,
                                 "workload": asdict(workload), "top_x": run.top_x,
                                 "start_round": run.round,
                                 "status_offset": status.stat().st_size if status.exists() else 0})
    except Exception:
        log.warning("Could not record estimate context", exc_info=True)


def mark_model_fallback(run_dir: Path) -> None:
    """Invalidate mixed-model timing without rewriting the requested settings."""
    from autofa.util.atomic import atomic_write_json

    path = run_dir / "estimate_context.json"
    context = _read_json(path)
    if context.get("schema_version") != SCHEMA_VERSION or context.get("mixed_settings"):
        return
    context["mixed_settings"] = True
    try:
        atomic_write_json(path, context)
    except OSError:
        log.warning("Could not invalidate estimate after model fallback", exc_info=True)


class LiveRuntimeEstimate:
    """Recalculate pace from this run, so an unseen model can calibrate itself."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.updated = -math.inf
        self.text = ""
        self.estimator: Estimator | None = None
        self._files: dict[tuple[Path, int], tuple[tuple[int, int, int] | None, Any]] = {}

    def _cached_read(self, name: str, *, offset: int = 0) -> Any:
        path = self.directory / name
        try:
            stat = path.stat()
            version = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            version = None
        key = (path, offset)
        previous = self._files.get(key)
        if previous is not None and previous[0] == version:
            return previous[1]
        value = (_read_rows(path, offset=offset) if name.endswith('.jsonl') else _read_json(path))
        self._files[key] = (version, value)
        return value

    def elapsed_seconds(self) -> float | None:
        """Full run's active time, independent of ETA validity or round changes."""
        return active_discovery_seconds(self._cached_read("status_log.jsonl"), now=time.time())

    def __call__(self) -> str:
        if time.monotonic() - self.updated < 1:
            return self.text
        self.updated = time.monotonic()
        self.text = ""
        config = self._cached_read("config.json")
        if not config:
            return ""
        context = self._cached_read("estimate_context.json")
        recorded = context.get("schema_version") == SCHEMA_VERSION
        now = time.time()
        progress = round_progress(self._cached_read("status_log.jsonl",
                                                   offset=context.get("status_offset", 0)),
                                  start_round=context.get("start_round", 0), now=now)
        if context.get("mixed_settings") or (
            "round" in config and config["round"] != progress.round_index
        ):
            progress = RoundProgress(round_index=config.get("round", 0), current_valid=False)
        if not recorded:
            progress = RoundProgress(round_index=config.get("round", 0))
        if self.estimator is None:
            self.estimator = Estimator(Config(), Workload(**context.get("workload", {})),
                                       signature=context.get("signature", {}),
                                       exclude_directory=self.directory)
        max_rounds = config.get("max_rounds", 3)
        start_round = context.get("start_round", 0) if recorded else progress.round_index
        initial = self.estimator.estimate(max_rounds - start_round, config.get("top_x", 5))
        self.text = live_timing_label(progress, max_rounds, initial=initial, now=now)
        return self.text


def _live_eta(typical: float, low: float, high: float, *, now: float | None,
              basis: str, rough: bool = False) -> str:
    """Live estimates use minutes, independently of the coarser setup labels."""
    typical = max(0, typical)
    text = f"{'rough ' if rough else ''}~{_minutes(typical)} min remaining"
    if now is not None and typical < 7 * 86400:
        finish = datetime.fromtimestamp(now + typical)
        today = datetime.fromtimestamp(now).date()
        clock = finish.strftime('%H:%M' if finish.date() == today else '%a %H:%M')
        text += f" · finish around {clock}"
    low_min, high_min = _minutes(low), _minutes(high)
    bounds = f"{low_min}–{high_min}" if low_min != high_min else str(high_min)
    return text + f"\nRange: {bounds} min · {basis}"


def live_timing_label(progress: RoundProgress, max_rounds: int, *,
                      initial: Estimate | None = None, now: float | None = None) -> str:
    round_number = progress.round_index + 1
    if progress.round_index >= max_rounds:
        return (f"Round {round_number} (beyond {max_rounds} planned) · "
                "ETA: unavailable for extra validation rounds")
    prefix = f"Round {round_number} of {max_rounds} · ETA: "
    n = len(progress.completed)
    if not progress.current_valid:
        return prefix + "recalibrating after interrupted work or changed settings"
    if n < 2:
        if initial and initial.high_minutes:
            elapsed = sum(progress.completed.values()) + progress.current_seconds
            if elapsed >= initial.high_minutes * 60:
                return prefix + "recalculating; elapsed time exceeds the initial range"
            low = max(0, initial.low_minutes * 60 - elapsed)
            high = initial.high_minutes * 60 - elapsed
            typical = (initial.median_minutes or
                       (initial.low_minutes + initial.high_minutes) / 2) * 60 - elapsed
            return prefix + _live_eta(
                typical, low, high, now=now, rough=initial.rough,
                basis="calibrating; extra rounds excluded")
        return prefix + "calibrating (need 2 completed rounds)"
    # Recent rounds track changes in pace instead of anchoring the whole run to
    # initial setup overhead. Keep all observed durations as uncertainty bounds.
    values = [progress.completed[i] for i in sorted(progress.completed)]
    remaining = max_rounds - progress.round_index
    if progress.current_seconds >= max(values):
        return prefix + "recalculating; this round is taking longer than earlier rounds"
    low = max(0, min(values) - progress.current_seconds) + min(values) * (remaining - 1)
    high = max(values) * remaining - progress.current_seconds
    pace = median(values[-3:])
    typical = max(0, pace - progress.current_seconds) + pace * (remaining - 1)
    return prefix + _live_eta(
        typical, low, high, now=now,
        basis=f"{n} completed rounds; extra rounds excluded")
