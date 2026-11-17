# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Efficiency metrics generated independently of detector discovery."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPORT_NAME = "efficiency.html"
DATA_NAME = "efficiency.json"
PERMUTATIONS = 5_000
SIGNIFICANCE_ALPHA = 0.05


def _seed(name: str, group: str) -> int:
    digest = hashlib.sha256(f"{name}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return math.nan
    values = np.sort(np.concatenate((left, right)))
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    left_cdf = np.searchsorted(left_sorted, values, side="right") / len(left_sorted)
    right_cdf = np.searchsorted(right_sorted, values, side="right") / len(right_sorted)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _problem_cluster_permutation_test(
    rows: dict[tuple[str, str, str], list[tuple[int, float, bool]]],
    *,
    metric: str,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any] | None:
    clusters = [
        sorted(values, key=lambda value: value[0])
        for values in rows.values()
        if values
    ]
    if len(clusters) < 2:
        return None
    observed_pass = np.asarray(
        [value for cluster in clusters for _, value, correct in cluster if correct],
        dtype=float,
    )
    observed_fail = np.asarray(
        [value for cluster in clusters for _, value, correct in cluster if not correct],
        dtype=float,
    )
    if not len(observed_pass) or not len(observed_fail):
        return None
    observed = _ks_statistic(observed_pass, observed_fail)
    strata: dict[
        tuple[int, ...],
        list[tuple[np.ndarray, np.ndarray]],
    ] = defaultdict(list)
    for cluster in clusters:
        repeat_pattern = tuple(repeat for repeat, _, _ in cluster)
        metric_values = np.asarray([value for _, value, _ in cluster], dtype=float)
        outcome_profile = np.asarray(
            [correct for _, _, correct in cluster],
            dtype=bool,
        )
        strata[repeat_pattern].append((metric_values, outcome_profile))

    rng = np.random.default_rng(_seed(metric, "permutation"))
    exceedances = 0
    for _ in range(permutations):
        shuffled_pass: list[float] = []
        shuffled_fail: list[float] = []
        for stratum in strata.values():
            profile_order = rng.permutation(len(stratum))
            for target, profile_index in zip(stratum, profile_order, strict=True):
                metric_values = target[0]
                outcome_profile = stratum[int(profile_index)][1]
                shuffled_pass.extend(metric_values[outcome_profile])
                shuffled_fail.extend(metric_values[~outcome_profile])
        statistic = _ks_statistic(
            np.asarray(shuffled_pass, dtype=float),
            np.asarray(shuffled_fail, dtype=float),
        )
        exceedances += statistic >= observed
    p_value = (exceedances + 1) / (permutations + 1)
    return {
        "method": "problem_cluster_permutation_ks",
        "statistic": observed,
        "p_value": p_value,
        "significant": p_value < SIGNIFICANCE_ALPHA,
        "alpha": SIGNIFICANCE_ALPHA,
        "permutations": permutations,
        "n_problems": len(clusters),
        "n_strata": len(strata),
        "n_pass": len(observed_pass),
        "n_fail": len(observed_fail),
        "corrected": False,
    }


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def resolve_pricing(cfg: Any, model_id: str) -> dict[str, Any]:
    """Resolve model rates, falling back to the configured simulation proxy."""
    configured = getattr(cfg, "efficiency_model_prices", {}) or {}
    short_name = str(model_id).rsplit("/", 1)[-1]
    raw = configured.get(model_id) or configured.get(short_name)
    if isinstance(raw, dict):
        input_rate = _number(raw.get("input"))
        cached_rate = _number(raw.get("cached"))
        output_rate = _number(raw.get("output"))
        if input_rate is not None and output_rate is not None:
            return {
                "source": "model_config",
                "model_id": model_id,
                "input_usd_per_million": input_rate,
                "cached_usd_per_million": (
                    cached_rate if cached_rate is not None else input_rate
                ),
                "output_usd_per_million": output_rate,
            }
    return {
        "source": "simulated_fallback",
        "model_id": model_id,
        "input_usd_per_million": float(
            getattr(cfg, "efficiency_simulated_input_rate", 0.15)
        ),
        "cached_usd_per_million": float(
            getattr(cfg, "efficiency_simulated_cached_rate", 0.03)
        ),
        "output_usd_per_million": float(
            getattr(cfg, "efficiency_simulated_output_rate", 0.50)
        ),
    }


_PLOT_LABELS = {
    "simulated_cost_usd": "Simulated cost",
    "input_tokens": "Input tokens",
    "cached_tokens": "Cached input tokens",
    "uncached_tokens": "Uncached input tokens",
    "reasoning_tokens": "Reasoning tokens",
    "answer_tokens": "Answer tokens",
    "completion_tokens": "Completion tokens",
    "n_turns": "Assistant turns",
    "n_tool_calls": "Tool calls",
    "avg_tool_calls_per_turn": "Tool calls per turn",
    "output_tokens": "Output tokens",
    "total_tokens": "Total tokens",
    "prompt_tokens": "Prompt size",
    "final_context_tokens": "Final context length",
    "problem_tokens_est": "Problem input tokens (estimated)",
    "tool_arg_chars": "Tool argument characters",
    "tool_output_chars": "Tool output characters",
    "final_context_tokens_est": "Final context tokens (simulated)",
}

_MAX_CHART_POINTS = 700
_DENSITY_POINTS = 64


def _compact(value: float) -> float:
    return float(f"{value:.6g}")


def _density(values: np.ndarray) -> list[list[float]]:
    """Gaussian KDE with Scott's bandwidth, scaled so the widest point is 1."""
    if len(values) < 2:
        return []
    spread = float(np.std(values, ddof=1))
    if not math.isfinite(spread) or spread == 0:
        return []
    bandwidth = spread * len(values) ** (-1 / 5)
    grid = np.linspace(float(values.min()), float(values.max()), _DENSITY_POINTS)
    density = np.exp(
        -0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2
    ).sum(axis=1)
    density /= density.max()
    return [[_compact(y), round(float(d), 4)] for y, d in zip(grid, density, strict=True)]


def _chart_group(
    name: str, group: str, raw: list[float], *, max_points: int = _MAX_CHART_POINTS,
) -> dict[str, Any] | None:
    values = np.asarray(raw or [], dtype=float)
    if not len(values):
        return None
    shown = values
    if len(shown) > max_points:
        rng = np.random.default_rng(_seed(name, group))
        shown = rng.choice(shown, size=max_points, replace=False)
    p25, median, p75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "p25": _compact(float(p25)),
        "median": _compact(float(median)),
        "p75": _compact(float(p75)),
        "min": _compact(float(values.min())),
        "max": _compact(float(values.max())),
        "density": _density(values),
        "points": [_compact(float(value)) for value in shown],
    }


def _chart_label(name: str, metric: dict[str, Any]) -> str:
    provenance = set(metric.get("sources") or {})
    label = _PLOT_LABELS.get(
        name, str(metric.get("label") or name).replace("_", " ").capitalize(),
    )
    if name not in _PLOT_LABELS and provenance == {"estimated"}:
        label += " (estimated)"
    elif provenance == {"simulated"}:
        label += " (simulated)"
    return label


def _chart_section(
    payload: dict[str, Any],
    samples: dict[str, dict[str, list[float]]],
    requested: tuple[str, ...],
    *,
    columns: int,
) -> dict[str, Any]:
    metric_data = {metric["name"]: metric for metric in payload.get("metrics") or []}
    names = [name for name in requested if name in metric_data]
    charts = {}
    for name in names:
        metric = metric_data[name]
        correct = _chart_group(name, "pass", samples.get(name, {}).get("pass") or [])
        incorrect = _chart_group(name, "fail", samples.get(name, {}).get("fail") or [])
        charts[name] = {
            "label": _chart_label(name, metric),
            "unit": str(metric.get("unit") or ""),
            "correct": correct,
            "incorrect": incorrect,
            "delta": (
                incorrect["mean"] - correct["mean"]
                if correct and incorrect else None
            ),
            "significant": bool((metric.get("permutation_test") or {}).get("significant")),
        }
    return {"metrics": names, "columns": min(columns, len(names)), "charts": charts}


_DATA_RE = re.compile(
    r'(<script id="autofa-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)

_OVERVIEW_MARKER = "/* —— AutoFA efficiency metrics —— */"

_OVERVIEW_CSS = "\n" + _OVERVIEW_MARKER + """
/* —— Efficiency metrics —— */
.eff-kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
@media (max-width: 720px) { .eff-kpis { grid-template-columns: 1fr 1fr; } }
.eff-section {
  background: var(--bg-card); border: 1px solid var(--line-1);
  border-left: 4px solid var(--gf-green); border-radius: var(--r2);
  padding: 20px 24px; margin-bottom: 16px;
}
.eff-section h2 { margin: 0 0 4px; font-size: 21px; }
.eff-note { color: var(--text-3); font-size: 12px; margin: 0 0 14px; }
.eff-formula {
  font-family: var(--mono); color: var(--text-2); background: var(--bg-card-2);
  border: 1px solid var(--line-1); border-radius: 6px; padding: 9px 12px;
  margin: 0 0 14px; overflow-wrap: anywhere;
}
.eff-defs .eff-formula { margin: 12px 14px; }
.eff-plot {
  display: block; width: 100%; max-width: 920px; border-radius: 6px;
  margin: 0 0 14px;
}
.eff-plot.single {
  max-width: 680px; margin-left: auto; margin-right: auto;
}
.eff-plot.light { display: none; }
body.theme-light .eff-plot.light { display: block; }
body.theme-light .eff-plot.dark { display: none; }
.eff-charts {
  display: grid; grid-template-columns: repeat(var(--eff-cols, 2), minmax(0, 1fr));
  gap: 16px; margin: 14px 0 4px;
}
.eff-charts.single { max-width: 760px; margin-left: auto; margin-right: auto; }
@media (max-width: 1000px) { .eff-charts { --eff-cols: 2 !important; } }
@media (max-width: 640px) { .eff-charts { --eff-cols: 1 !important; } }
.eff-chart {
  margin: 0; min-width: 0; padding: 14px 14px 12px;
  border: 1px solid var(--line-1); border-radius: 12px; background: var(--bg-card);
  box-shadow: 0 1px 2px rgba(0, 0, 0, .04);
}
.eff-chart-head {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  min-height: 22px; margin-bottom: 6px;
}
.eff-chart-title { font-size: 15px; font-weight: 700; color: var(--text-1); }
.eff-chart-body { position: relative; height: 250px; }
.eff-charts.single .eff-chart-body { height: 300px; }
.eff-chart-svg { position: absolute; inset: 0; }
.eff-chart-svg svg { display: block; width: 100%; height: 100%; overflow: visible; }
.eff-sig {
  flex: none; padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 700;
  color: var(--gf-green); background: color-mix(in srgb, var(--gf-green) 16%, transparent);
}
.eff-chart-foot {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px;
  margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line-1);
}
.eff-chart-foot div {
  display: flex; flex-direction: column; align-items: center; gap: 1px; min-width: 0;
}
.eff-chart-foot span { font-size: 11px; color: var(--text-3); }
.eff-chart-foot b { font-size: 18px; font-weight: 700; color: var(--text-1); white-space: nowrap; }
.eff-chart-foot .correct b { color: var(--gf-green); }
.eff-chart-foot .incorrect b { color: var(--gf-bad); }
.eff-grid { stroke: var(--line-1); stroke-dasharray: 3 4; }
.eff-tick { fill: var(--text-3); font-size: 11px; }
.eff-unit { fill: var(--text-3); font-size: 11px; }
.eff-xlabel { fill: var(--text-2); font-size: 12px; font-weight: 600; }
.eff-xn { fill: var(--text-3); font-size: 11px; }
.eff-violin { fill-opacity: .26; transition: fill-opacity .15s ease; cursor: default; }
.eff-violin:hover { fill-opacity: .4; }
.eff-dot { fill-opacity: .55; }
.eff-dot:hover { fill-opacity: 1; r: 4.5px; }
.eff-chart-svg [data-tip] { cursor: default; }
.eff-violin.correct, .eff-dot.correct { fill: var(--gf-green); }
.eff-violin.incorrect, .eff-dot.incorrect,
.eff-violin.card, .eff-dot.card { fill: var(--gf-bad); }
.eff-violin.other, .eff-dot.other { fill: var(--text-3); }
.eff-card { margin: 14px 0; }
.eff-card-body { padding: 4px 14px 10px; }
.eff-card-note { margin: 12px 0; font-size: 12px; color: var(--text-3); }
.eff-iqr { stroke: var(--text-1); stroke-width: 2.2; stroke-linecap: round; pointer-events: none; }
.eff-chart-notes { margin: 14px 0 6px; font-size: 12px; color: var(--text-3); }
.eff-chart-notes p { margin: 2px 0; }
.eff-chart-notes .sig { color: var(--gf-green); }
.eff-tip {
  position: fixed; z-index: 1000; pointer-events: none; display: none;
  padding: 8px 10px; border-radius: 8px; font-size: 12px; line-height: 1.45;
  color: var(--text-1); background: var(--bg-card); border: 1px solid var(--line-1);
  box-shadow: 0 6px 20px rgba(0, 0, 0, .18);
}
.eff-tip b { display: block; margin-bottom: 2px; }
.eff-tip span { color: var(--text-3); }
.eff-defs {
  margin: 0 0 16px; border: 1px solid var(--line-1); border-radius: 10px;
  background: var(--bg-card);
}
.eff-defs > summary {
  display: flex; align-items: center; gap: 10px; cursor: pointer;
  padding: 10px 14px; font-weight: 600; list-style: none; user-select: none;
}
.eff-defs > summary::-webkit-details-marker { display: none; }
.eff-defs > summary:hover { background: var(--bg-card-2); border-radius: 10px; }
.eff-defs-chevron {
  display: inline-block; color: var(--gf-green); font-size: 16px; line-height: 1;
  transition: transform .15s ease;
}
.eff-defs[open] .eff-defs-chevron { transform: rotate(90deg); }
.eff-defs-hint {
  margin-left: auto; color: var(--text-3); font-size: 11px; font-weight: 500;
  border: 1px solid var(--line-1); border-radius: 999px; padding: 2px 10px;
}
.eff-defs-hint::after { content: attr(data-closed); }
.eff-defs[open] .eff-defs-hint::after { content: attr(data-open); }
.eff-defs[open] > summary { border-bottom: 1px solid var(--line-1); }
.eff-defs .eff-table { margin: 6px 14px 14px; width: calc(100% - 28px); }
.eff-defs .eff-table td:last-child,
.eff-defs .eff-table th:last-child { text-align: left; }
.eff-defs .eff-table td:first-child { white-space: nowrap; padding-right: 24px; }
.eff-section .eff-exact { margin: 18px 0 6px; }
.eff-exact .eff-table td { vertical-align: top; }
.eff-exact .eff-table td:first-child { white-space: normal; width: 58%; }
.eff-exact .eff-table td:last-child,
.eff-exact .eff-table th:last-child { text-align: right; }
.eff-exact-name { font-weight: 600; color: var(--text-1); }
.eff-exact-def { margin-top: 2px; font-size: 11px; line-height: 1.45; color: var(--text-3); }
.eff-exact-note { margin: 0 14px 14px; font-size: 11px; line-height: 1.45; color: var(--text-3); }
.eff-exact-note b { color: var(--text-2); }
.eff-summary {
  border: 1px solid var(--gf-green);
  background: color-mix(in srgb, var(--gf-green) 3%, var(--bg-card));
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--gf-green) 14%, transparent),
    0 10px 30px color-mix(in srgb, var(--gf-green) 12%, transparent);
}
.eff-summary h2 { margin: 0 0 14px; }
.eff-summary-list {
  list-style: none; margin: 0; padding: 0; display: grid; gap: 12px;
  font-size: 14px; line-height: 1.55; color: var(--text-1);
}
.eff-summary-list > li { position: relative; padding-left: 18px; }
.eff-summary-list > li::before {
  content: ""; position: absolute; left: 0; top: .5em; width: 7px; height: 7px;
  border-radius: 2px; background: var(--gf-green);
}
.eff-summary-lead { font-weight: 600; }
.eff-summary-share { font-weight: 700; color: var(--gf-bad); }
.eff-summary-sizes {
  display: grid; grid-template-columns: max-content 1fr; gap: 8px 12px; margin-top: 8px;
}
.eff-summary-size {
  justify-self: start; align-self: start; padding: 1px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 700; line-height: 20px; color: var(--gf-green);
  background: color-mix(in srgb, var(--gf-green) 16%, transparent);
}
.eff-summary-size.moderate { background: color-mix(in srgb, var(--gf-green) 9%, transparent); }
.eff-summary-size.small { color: var(--text-2); background: var(--bg-card-2); }
.eff-summary-size.negligible {
  color: var(--text-3); background: transparent; box-shadow: inset 0 0 0 1px var(--line-1);
}
.eff-summary-metric { white-space: nowrap; }
.eff-summary-arrow { font-weight: 700; }
.eff-summary-arrow.up { color: var(--gf-bad); }
.eff-summary-arrow.down { color: var(--gf-green); }
.eff-summary-why { display: block; font-size: 12px; color: var(--text-3); }
.eff-summary-key {
  margin: 16px 0 0; padding-top: 10px; border-top: 1px solid var(--line-1);
  font-size: 12px; color: var(--text-3);
}
.eff-provenance-note {
  margin: 0 14px 12px; color: var(--text-3); font-size: 11px;
}
.eff-provenance .eff-table td:nth-child(2),
.eff-provenance .eff-table td:nth-child(3) { white-space: nowrap; padding-left: 16px; }
.eff-status {
  display: inline-block; border: 1px solid var(--line-1); border-radius: 999px;
  padding: 2px 8px; font-size: 10px; font-weight: 700; text-transform: uppercase;
}
.eff-status.recorded { color: var(--gf-green); }
.eff-status.derived { color: #4f8fd8; }
.eff-status.estimated { color: #b7791f; }
.eff-status.not_applicable { color: var(--text-3); }
.eff-status.partial,.eff-status.unavailable { color: var(--gf-bad); }
.eff-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.eff-table th,.eff-table td {
  text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--line-1);
}
.eff-table th:first-child,.eff-table td:first-child { text-align: left; }
.eff-table th {
  color: var(--text-3); font-size: 10px; letter-spacing: .06em;
  text-transform: uppercase;
}
.tool-rates { margin-top: 4px; }
.tool-rates-row {
  display: grid; grid-template-columns: minmax(120px, 180px) minmax(200px, 1fr) 128px;
  gap: 18px; align-items: center; padding: 12px 0;
  border-bottom: 1px solid var(--line-1);
}
.tool-rates-row:last-child { border-bottom: 0; }
.tool-rates-head { padding: 0 0 10px; color: var(--text-3); font-size: 12px; }
.tool-rates-legend { display: flex; gap: 18px; grid-column: 1 / 3; }
.tool-rates-legend span { display: inline-flex; align-items: center; gap: 7px; }
.tool-rates-legend i { width: 9px; height: 9px; border-radius: 50%; }
.tool-rates-legend i.correct { background: var(--gf-green); }
.tool-rates-legend i.incorrect { background: var(--gf-bad); }
.tool-rates-ratio-head { text-align: center; white-space: nowrap; }
.tool-rates-name {
  font-family: var(--mono); font-size: 13px; color: var(--text-1);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tool-rates-bars { display: grid; gap: 6px; }
.tool-bar-row {
  display: grid; grid-template-columns: minmax(0, 1fr) 44px;
  gap: 10px; align-items: center;
}
.tool-bar-track {
  display: block; height: 10px; background: var(--bg-card-3);
  border-radius: 999px; overflow: hidden;
}
.tool-bar { display: block; height: 100%; min-width: 2px; border-radius: inherit; }
.tool-bar.correct { background: var(--gf-green); }
.tool-bar.incorrect { background: var(--gf-bad); }
.tool-bar-value {
  color: var(--text-2); font-family: var(--mono); font-size: 12px; text-align: right;
}
.tool-ratio {
  justify-self: center; padding: 4px 12px; border-radius: 999px;
  font-family: var(--mono); font-size: 13px; font-weight: 700;
  color: var(--gf-bad); background: color-mix(in srgb, var(--gf-bad) 14%, transparent);
}
.report-tabs {
  display: flex; gap: 4px; margin: 0 0 20px; padding: 4px;
  width: fit-content; max-width: 100%; overflow-x: auto;
  border: 1px solid var(--gf-green); border-radius: 8px; background: var(--bg-card-2);
  box-shadow: 0 0 0 1px rgba(118, 185, 0, .16);
}
.report-tab {
  border: 1px solid transparent; border-radius: 5px; padding: 8px 14px; white-space: nowrap;
  color: var(--text-3); background: transparent; cursor: pointer;
  font-size: 12px; font-weight: 700;
}
.report-tab:hover { color: var(--text-1); background: var(--bg-card-hover); }
.report-tab:focus-visible {
  outline: 2px solid var(--gf-green-bright); outline-offset: 1px;
}
.report-tab[aria-selected="true"] {
  color: var(--bg-page); background: var(--gf-green); border-color: var(--gf-green);
  box-shadow: 0 2px 8px rgba(118, 185, 0, .28);
}
.report-tab-panel:focus { outline: none; }
.eff-toggle {
  margin: 0 0 22px; border: 1px solid var(--line-1); border-radius: 12px;
  background: var(--bg-card); padding: 0 16px 8px;
}
.eff-toggle > summary {
  cursor: pointer; font-weight: 700; padding: 14px 4px; list-style: none;
}
.eff-toggle > summary::-webkit-details-marker { display: none; }
.eff-toggle[open] > summary {
  border-bottom: 1px solid var(--line-1); margin-bottom: 14px;
}
.eff-warning {
  margin: 0 0 10px; padding: 10px 12px; border: 1px solid var(--gf-bad);
  border-radius: 8px; color: var(--text-1); background: rgba(239, 111, 108, .12);
}
"""

_OVERVIEW_JS = "\n" + _OVERVIEW_MARKER + r"""
/* —— Efficiency metrics —— */
(function () {
  const additive = new Set([
    'simulated_cost_usd', 'input_tokens', 'cached_tokens', 'uncached_tokens',
    'reasoning_tokens', 'answer_tokens', 'completion_tokens', 'output_tokens', 'total_tokens',
    'n_turns', 'n_tool_calls', 'tool_arg_chars', 'tool_output_chars'
  ]);

  const EVALRDL_DEFINITIONS = [
    ['simulated_cost_usd', 'Cost', 'Requires input and classified output tokens. Formula: '
      + 'uncached input × input rate + cached input × cached rate + output × output rate, '
      + 'per million. Missing cached tokens count as zero. Rates come from user '
      + 'configuration, otherwise disclosed stand-in rates.'],
    ['input_tokens', 'Input tokens', 'Use the EvalRDL rollout prompt-token total. When that total is missing, use recorded requests for the selected model. Never add these sources together.'],
    ['reasoning_tokens', 'Reasoning tokens', 'Sum counted transcript requests once each. When unavailable, use recorded requests for the selected model. Omit rollouts without an unambiguous source.'],
    ['answer_tokens', 'Answer tokens', 'Compare rollout completion with reasoning and selected-model completion. If rollout completion excludes reasoning, use it directly; if it includes reasoning, subtract reasoning. Omit unresolved rollouts.'],
    ['output_tokens', 'Output tokens', 'Use selected-model completion as total output when available. Otherwise classify whether rollout completion includes reasoning; omit rollouts whose total output cannot be established.'],
    ['cached_tokens', 'Cached tokens', 'Sum counted transcript requests once each. When unavailable, use recorded requests for the selected model. Omit rollouts without an unambiguous source.'],
    ['uncached_tokens', 'Uncached input tokens',
      'Input tokens minus cached input tokens. Omit rollouts without recorded cached tokens.'],
    ['total_tokens', 'Total tokens', 'Input tokens plus classified output tokens. Omit rollouts missing either component.'],
    ['n_turns', 'Assistant turns', 'Use the EvalRDL rollout assistant-turn total. Omit rollouts without that total.'],
    ['n_tool_calls', 'Tool calls', 'Use the recorded number of tool calls. Parallel calls count separately. Omit when no count is recorded.'],
    ['avg_tool_calls_per_turn', 'Tool calls per turn', 'Tool calls divided by assistant turns. Omit when either value is missing or turns is zero.'],
    ['tool_arg_chars', 'Tool argument characters', 'Sum exact recorded tool-argument characters using EvalRDL transcript-account rules. Omit when argument characters are unavailable.'],
    ['tool_output_chars', 'Tool output characters', 'Sum exact recorded characters returned by tools using EvalRDL transcript-account rules. Omit when tool-result characters are unavailable.'],
    ['', 'Calls by tool type', 'Group calls from the selected EvalRDL transcript account by tool name and report overall, pass, and fail counts. Omit when tool names are unavailable.'],
    ['prompt_tokens', 'Prompt size', 'Input tokens of the first request in one ordered lane for the selected model. If that sequence is unavailable, use the parent agent’s first transcript request. Child-agent lanes are excluded.'],
    ['final_context_tokens', 'Final context length', 'Input tokens of the last request in one ordered lane for the selected model. If that sequence is unavailable, use the parent agent’s last transcript request. Child-agent lanes are excluded.'],
  ];

  function number(value, unit) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    const n = Number(value);
    if (unit === 'USD') {
      return '$' + (Math.abs(n) >= 1
        ? n.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})
        : n.toLocaleString(undefined, {maximumFractionDigits: 4}));
    }
    return n.toLocaleString(undefined, {maximumFractionDigits: 1});
  }

  const METRIC_DEFINITIONS = Object.fromEntries(EVALRDL_DEFINITIONS
    .filter(([name]) => name).map(([name, , definition]) => [name, definition]));
  const TOOL_TYPE_DEFINITION = EVALRDL_DEFINITIONS
    .find(([, label]) => label === 'Calls by tool type')[2];

  function costFormula(pricing) {
    return `<div class="eff-formula">cost = uncached input ×
      $${pricing.input_usd_per_million}/M + cached input ×
      $${pricing.cached_usd_per_million}/M + output ×
      $${pricing.output_usd_per_million}/M</div>`;
  }

  function exactValues(report, key, names) {
    const fromEval = report.source === 'evalrdl';
    const byName = Object.fromEntries((report.metrics || []).map(metric => [metric.name, metric]));
    const rows = names.map(name => byName[name]).filter(Boolean).map(metric => {
      const definition = fromEval ? METRIC_DEFINITIONS[metric.name] : '';
      return `<tr>
        <td><div class="eff-exact-name">${escapeHtml(metric.label)}</div>
          ${definition ? `<div class="eff-exact-def">${escapeHtml(definition)}</div>` : ''}</td>
        <td>${additive.has(metric.name) ? number(metric.overall.total, metric.unit) : '—'}</td>
        <td>${number(metric.pass.median, metric.unit)}</td>
        <td>${number(metric.fail.median, metric.unit)}</td>
      </tr>`;
    }).join('');
    if (!rows) return '';
    const formula = key === 'cost' && report.pricing ? costFormula(report.pricing) : '';
    const toolNote = key === 'activity' && fromEval && (report.tool_types || []).length
      ? `<p class="eff-exact-note"><b>Tool calls per rollout:</b>
          ${escapeHtml(TOOL_TYPE_DEFINITION)}</p>`
      : '';
    return `<details class="eff-defs eff-exact">
      <summary><span class="eff-defs-chevron" aria-hidden="true">▸</span>
        <span>Metric rules &amp; exact values</span>
        <span class="eff-defs-hint" data-closed="Show" data-open="Hide"></span></summary>
      ${formula}
      <table class="eff-table"><thead><tr><th>metric</th><th>total</th>
        <th>correct median</th><th>incorrect median</th></tr></thead>
      <tbody>${rows}</tbody></table>
      ${toolNote}
    </details>`;
  }

  function toolTable(report) {
    const tools = (report.tool_types || []).map(tool => ({
      name: tool.name,
      correct: report.n_pass ? tool.pass / report.n_pass : 0,
      incorrect: report.n_fail ? tool.fail / report.n_fail : 0,
    })).sort((a, b) => b.incorrect - a.incorrect).slice(0, 12);
    if (!tools.length) return '';
    const scale = Math.max(
      ...tools.flatMap(tool => [tool.correct, tool.incorrect]), Number.EPSILON);
    const bar = (kind, value) => `<div class="tool-bar-row">
      <span class="tool-bar-track"><span class="tool-bar ${kind}"
        style="width:${100 * value / scale}%"></span></span>
      <span class="tool-bar-value">${value.toFixed(1)}</span></div>`;
    const rows = tools.map(tool => `<div class="tool-rates-row">
      <span class="tool-rates-name" title="${escapeHtml(tool.name)}">${escapeHtml(tool.name)}</span>
      <div class="tool-rates-bars">
        ${bar('correct', tool.correct)}${bar('incorrect', tool.incorrect)}</div>
      <span class="tool-ratio">
        ${tool.correct ? (tool.incorrect / tool.correct).toFixed(1) + '×' : '—'}</span>
    </div>`).join('');
    return `<h3>Tool calls per rollout</h3><div class="tool-rates">
      <div class="tool-rates-row tool-rates-head">
        <span class="tool-rates-legend"><span><i class="correct"></i>Correct rollouts</span>
          <span><i class="incorrect"></i>Incorrect rollouts</span></span>
        <span class="tool-rates-ratio-head">Incorrect ÷ correct</span>
      </div>${rows}</div>`;
  }

  function compact(value, unit) {
    const n = Number(value);
    if (value == null || !Number.isFinite(n)) return '—';
    const trim = (x, digits) => String(Number(x.toFixed(digits)));
    const size = Math.abs(n);
    if (unit === 'USD') return '$' + trim(n, size && size < 0.01 ? 3 : 2);
    if (size >= 1e6) return trim(n / 1e6, 2) + 'M';
    if (size >= 1e3) return trim(n / 1e3, 1) + 'k';
    return trim(n, 1);
  }

  function signed(value, unit) {
    if (value == null) return '—';
    const sign = value > 0 ? '+' : value < 0 ? '−' : '±';
    return sign + compact(Math.abs(value), unit);
  }

  function meanPair(chart) {
    const a = (chart.correct || {}).mean;
    const b = (chart.incorrect || {}).mean;
    const short = [compact(a, chart.unit), compact(b, chart.unit)];
    if (short[0] !== short[1] || a == null || a === b) return short;
    const exact = x => chart.unit === 'USD'
      ? '$' + String(Number(Number(x).toFixed(4)))
      : Number(Number(x).toFixed(1)).toLocaleString();
    return [exact(a), exact(b)];
  }

  function niceTicks(lo, hi) {
    if (!(hi > lo)) { hi = lo + 1; }
    const rough = (hi - lo) / 5;
    const power = Math.pow(10, Math.floor(Math.log10(rough)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * power).find(s => s >= rough);
    const ticks = [];
    for (let t = Math.floor(lo / step) * step; t <= hi + step * 1e-9; t += step) {
      ticks.push(Number(t.toPrecision(12)));
    }
    const last = ticks[ticks.length - 1];
    if (last < hi) ticks.push(Number((last + step).toPrecision(12)));
    return ticks;
  }

  function seeded(text) {
    let state = 2166136261;
    for (const char of text) state = Math.imul(state ^ char.charCodeAt(0), 16777619);
    return () => {
      state = (state + 0x6D2B79F5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const GROUPS = [
    ['correct', 'Correct'], ['incorrect', 'Incorrect'],
    ['card', 'This failure'], ['other', 'Other failures'],
  ];

  function drawChart(target, name, chart) {
    const width = target.clientWidth;
    const height = target.clientHeight;
    if (!width || !height) return;
    const left = 58, right = 8, top = 30, bottom = 44;
    const x0 = left, x1 = width - right, y0 = top, y1 = height - bottom;
    const groups = GROUPS.map(([key, label]) => [key, label, chart[key]]).filter(g => g[2]);
    const lo = Math.min(...groups.map(g => g[2].min));
    const hi = Math.max(...groups.map(g => g[2].max));
    const ticks = niceTicks(lo, hi);
    const domainLo = ticks[0], domainHi = ticks[ticks.length - 1];
    const y = v => y1 - (v - domainLo) / (domainHi - domainLo || 1) * (y1 - y0);
    const spacing = (x1 - x0) * 0.92 / groups.length;
    const centers = Object.fromEntries(groups.map(([key], index) =>
      [key, x0 + (x1 - x0) * (0.04 + 0.92 * (index + 0.5) / groups.length)]));
    const halfWidth = spacing * 0.41;
    const unitWord = chart.unit === 'USD' ? '' : ' ' + chart.unit;
    const parts = [];
    ticks.forEach(t => {
      parts.push(`<line class="eff-grid" x1="${x0}" x2="${x1}" y1="${y(t)}" y2="${y(t)}"/>`);
      parts.push(`<text class="eff-tick" x="${x0 - 10}" y="${y(t)}" text-anchor="end"
        dominant-baseline="middle">${compact(t, chart.unit)}</text>`);
    });
    const unitLabel = chart.unit === 'USD' ? 'Cost (USD)' : chart.unit;
    if (unitLabel) {
      parts.push(`<text class="eff-unit" transform="translate(12 ${(y0 + y1) / 2}) rotate(-90)"
        text-anchor="middle">${escapeHtml(unitLabel)}</text>`);
    }
    groups.forEach(([key, label, group]) => {
      const cx = centers[key];
      if (group.density.length) {
        const rightSide = group.density.map(([v, d]) => `${cx + d * halfWidth},${y(v)}`);
        const leftSide = group.density.slice().reverse()
          .map(([v, d]) => `${cx - d * halfWidth},${y(v)}`);
        const fmt = value => compact(value, chart.unit);
        const tip = `<b>${label} rollouts</b><span>Median</span> ${fmt(group.median)}<br>`
          + `<span>Middle 50%</span> ${fmt(group.p25)} – ${fmt(group.p75)}<br>`
          + `<span>Mean</span> ${fmt(group.mean)}<br>`
          + `<span>Rollouts</span> ${group.n.toLocaleString()}`;
        parts.push(`<path class="eff-violin ${key}"
          d="M${rightSide.join('L')}L${leftSide.join('L')}Z" data-tip="${escapeHtml(tip)}"/>`);
      }
      const random = seeded(name + ':' + key);
      group.points.forEach(value => {
        const u = Math.max(random(), 1e-9);
        const gaussian = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
        const cxPoint = cx + Math.max(-3, Math.min(3, gaussian)) * spacing * 0.035;
        const tip = `<b>${label} rollout</b>${compact(value, chart.unit)}${escapeHtml(unitWord)}`;
        parts.push(`<circle class="eff-dot ${key}" cx="${cxPoint}" cy="${y(value)}" r="2.3"
          data-tip="${escapeHtml(tip)}"/>`);
      });
      parts.push(`<line class="eff-iqr" x1="${cx}" x2="${cx}"
        y1="${y(group.p25)}" y2="${y(group.p75)}"/>`);
      parts.push(`<line class="eff-iqr" x1="${cx - spacing * 0.09}" x2="${cx + spacing * 0.09}"
        y1="${y(group.median)}" y2="${y(group.median)}"/>`);
      parts.push(`<text class="eff-xlabel" x="${cx}" y="${y1 + 22}"
        text-anchor="middle">${label}</text>`);
      parts.push(`<text class="eff-xn" x="${cx}" y="${y1 + 38}"
        text-anchor="middle">n=${group.n.toLocaleString()}</text>`);
    });
    target.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img"
      aria-label="${escapeHtml(chart.label)}">${parts.join('')}</svg>`;
  }

  function chartGrid(key, section) {
    const names = section.metrics.filter(name => section.charts[name]);
    const figures = names.map(name => {
      const chart = section.charts[name];
      const [correct, incorrect] = meanPair(chart);
      return `<figure class="eff-chart">
        <figcaption class="eff-chart-head">
          <div class="eff-chart-title">${escapeHtml(chart.label)}</div>
          ${chart.significant ? '<span class="eff-sig">p &lt; 0.05</span>' : ''}
        </figcaption>
        <div class="eff-chart-body">
          <div class="eff-chart-svg" data-section="${key}" data-metric="${escapeHtml(name)}"></div>
        </div>
        <div class="eff-chart-foot">
          <div class="correct"><span>Mean correct</span><b>${correct}</b></div>
          <div class="incorrect"><span>Mean incorrect</span><b>${incorrect}</b></div>
          <div><span>Difference</span><b>${signed(chart.delta, chart.unit)}</b></div>
        </div>
      </figure>`;
    }).join('');
    return `<div class="eff-charts${names.length === 1 ? ' single' : ''}"
        style="--eff-cols:${section.columns || 2}">${figures}</div>
      <div class="eff-chart-notes">
        <p>Each point is one rollout. Means average across rollouts;
          difference is incorrect − correct.</p>
        <p class="sig">Significance uses a task-cluster permutation KS test (p &lt; 0.05).</p>
      </div>`;
  }

  function cardDropdown(body) {
    return `<details class="eff-defs eff-card">
      <summary><span class="eff-defs-chevron" aria-hidden="true">▸</span>
        <span>Efficiency</span>
        <span class="eff-defs-hint" data-closed="Show" data-open="Hide"></span>
      </summary>
      <div class="eff-card-body">${body}</div>
    </details>`;
  }

  function cardEfficiency(name, card) {
    if (card.unavailable) {
      return cardDropdown(`<p class="eff-card-note">Efficiency cannot be computed because
        this detector's matched rollouts are not recorded.</p>`);
    }
    if (card.n < card.min_attempts) {
      return cardDropdown(`<p class="eff-card-note">Efficiency cannot be computed: this
        failure mode matched ${card.n} failed rollouts, and at least ${card.min_attempts}
        are needed.</p>`);
    }
    if (!(card.metrics || []).length) {
      return cardDropdown(`<p class="eff-card-note">Efficiency cannot be computed because
        no efficiency metrics were recorded for these rollouts.</p>`);
    }
    const mean = (chart, key) => compact((chart[key] || {}).mean, chart.unit);
    const figures = card.metrics.map(metric => {
      const chart = card.charts[metric];
      return `<figure class="eff-chart">
        <figcaption class="eff-chart-head">
          <div class="eff-chart-title">${escapeHtml(chart.label)}</div>
          ${chart.significant ? '<span class="eff-sig">p &lt; 0.05</span>' : ''}
        </figcaption>
        <div class="eff-chart-body">
          <div class="eff-chart-svg" data-card="${escapeHtml(name)}"
            data-metric="${escapeHtml(metric)}"></div>
        </div>
        <div class="eff-chart-foot">
          <div class="correct"><span>Mean correct</span><b>${mean(chart, 'correct')}</b></div>
          <div class="incorrect"><span>Mean this failure</span><b>${mean(chart, 'card')}</b></div>
          <div><span>Mean other failures</span><b>${mean(chart, 'other')}</b></div>
        </div>
      </figure>`;
    }).join('');
    const scope = card.sampled ? ' in the evaluated sample' : '';
    return cardDropdown(`
      <p class="eff-note">${card.n} failed rollouts match this failure mode${scope}. They are
        compared with the run's correct rollouts and its other failed rollouts.</p>
      <div class="eff-charts" style="--eff-cols:2">${figures}</div>
      <div class="eff-chart-notes">
        <p>Each point is one rollout.</p>
        <p class="sig">Significance compares this failure's rollouts with correct rollouts
          using a task-cluster permutation KS test (p &lt; 0.05).</p>
        ${card.sampled ? `<p>The detector was evaluated on a sample, so other failures
          include rollouts it did not check.</p>` : ''}
      </div>`);
  }

  function addCardEfficiency(root, report) {
    const cards = report.cards || {};
    root.querySelectorAll('.tax-card[data-function-name]').forEach(article => {
      const name = article.getAttribute('data-function-name');
      const detector = article.querySelector('.tax-detector');
      if (name in cards && detector) {
        detector.insertAdjacentHTML('afterend', cardEfficiency(name, cards[name]));
      }
    });
  }

  function mountCharts(report) {
    const targets = document.querySelectorAll('.eff-chart-svg');
    if (!targets.length) return;
    const observer = new ResizeObserver(entries => entries.forEach(entry => {
      const target = entry.target;
      const owner = target.dataset.card != null
        ? (report.cards || {})[target.dataset.card] : report.sections[target.dataset.section];
      const charts = (owner || {}).charts || {};
      const chart = charts[target.dataset.metric];
      if (chart) drawChart(target, target.dataset.metric, chart);
    }));
    targets.forEach(target => observer.observe(target));
    let tip = document.querySelector('.eff-tip');
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'eff-tip';
      document.body.appendChild(tip);
    }
    document.addEventListener('mousemove', event => {
      const source = event.target.closest && event.target.closest('.eff-chart-svg [data-tip]');
      if (!source) { tip.style.display = 'none'; return; }
      tip.innerHTML = source.getAttribute('data-tip');
      tip.style.display = 'block';
      const box = tip.getBoundingClientRect();
      const x = Math.min(event.clientX + 14, window.innerWidth - box.width - 8);
      const yPos = event.clientY + 14 + box.height > window.innerHeight
        ? event.clientY - box.height - 10 : event.clientY + 14;
      tip.style.left = x + 'px';
      tip.style.top = yPos + 'px';
    });
  }

  function section(report, key, title, note, extra) {
    const section = report.sections[key] || {};
    const names = section.metrics || [];
    if (!names.length) return '';
    if (section.charts) {
      const noteMarkup = note ? `<p class="eff-note">${note}</p>` : '';
      return `<section class="eff-section"><h2>${escapeHtml(title)}</h2>
        ${noteMarkup}${extra || ''}${chartGrid(key, section)}
        ${exactValues(report, key, names)}
        ${key === 'activity' ? toolTable(report) : ''}</section>`;
    }
    const sizeClass = names.length === 1 ? ' single' : '';
    const image = (src, theme) => src
      ? `<img class="eff-plot ${theme}${sizeClass}" alt="${escapeHtml(title)}" src="${src}"/>` : '';
    const plot = image(section.plot, 'dark') + image(section.plot_light, 'light');
    const noteMarkup = note ? `<p class="eff-note">${note}</p>` : '';
    return `<section class="eff-section"><h2>${escapeHtml(title)}</h2>
      ${noteMarkup}${extra || ''}${plot}
      ${exactValues(report, key, names)}
      ${key === 'activity' ? toolTable(report) : ''}</section>`;
  }

  function summaryBox(report) {
    const summary = report.summary;
    if (!summary) return '';
    const share = value => value > 0 && value < 0.01 ? '<1%' : `${Math.round(value * 100)}%`;
    const partial = summary.partial || {};
    const covered = value => value < 0.01 ? '<1%' : `${Math.floor(value * 100)}%`;
    const based = label => label in partial ? `${covered(partial[label])} of rollouts` : '';
    const lower = label => escapeHtml(label.charAt(0).toLowerCase() + label.slice(1));
    const name = label => based(label) ? `${lower(label)} (${based(label)})` : lower(label);
    const names = labels => labels.map(name).join(', ');
    const pricing = report.pricing || {};
    const realPrices = pricing.source === 'model_config';
    const bullets = [];
    if (summary.cost) {
      const costBasis = summary.cost.coverage
        ? ` Based on ${covered(summary.cost.coverage)} of rollouts.` : '';
      bullets.push(`<span class="eff-summary-lead">Incorrect</span> rollouts make up
        <b>${share(summary.cost.attempt_share)}</b> of rollouts
        and <span class="eff-summary-share">${share(summary.cost.cost_share)}</span>
        of ${realPrices ? 'cost' : 'simulated cost'}.${costBasis}`);
    }
    const significant = summary.differ.length || summary.negligible.length;
    if (significant) {
      const rows = ['large', 'moderate', 'small'].map(size => {
        const items = summary.differ.filter(item => item.size === size);
        if (!items.length) return '';
        const metrics = items.map(item => {
          const arrow = item.direction === 'up' ? '↑' : '↓';
          const basis = based(item.label) ? `, ${based(item.label)}` : '';
          return `<span class="eff-summary-metric">${lower(item.label)}
            (<span class="eff-summary-arrow ${item.direction}">${arrow}</span>${basis})</span>`;
        });
        const title = size.charAt(0).toUpperCase() + size.slice(1);
        return `<span class="eff-summary-size ${size}">${title}</span>
          <span>${metrics.join(', ')}</span>`;
      }).join('');
      const negligible = summary.negligible.length
        ? `<span class="eff-summary-size negligible">Negligible</span>
          <span>${names(summary.negligible)}</span>` : '';
      bullets.push(`Correct and incorrect rollouts
        <span class="eff-summary-lead">differ significantly</span> in:
        <div class="eff-summary-sizes">${rows}${negligible}</div>`);
    }
    const lists = summary.partial ? [
      ['No significant difference', summary.not_significant, ''],
      ['Too few rollouts', summary.insufficient, 'Needs 20 correct and 20 incorrect rollouts.'],
    ] : [
      ['No significant difference', summary.not_significant, ''],
      ['Not enough data', summary.insufficient, `Needs 90% coverage and 20 correct and
        20 incorrect rollouts.`],
    ];
    for (const [title, labels, why] of lists) {
      if (labels.length) {
        const note = why ? `<span class="eff-summary-why">${why}</span>` : '';
        bullets.push(`<span class="eff-summary-lead">${title}:</span> ${names(labels)}.${note}`);
      }
    }
    if (!bullets.length) return '';
    const key = significant ? `<p class="eff-summary-key">
      <span class="eff-summary-arrow up">↑</span> incorrect higher than correct,
      <span class="eff-summary-arrow down">↓</span> incorrect lower than correct. Difference
      size comes from how far apart the two distributions are: small from 0.1, moderate from
      0.2, large from 0.4. Negligible: below 0.1, or medians within 5%.</p>` : '';
    return `<section class="eff-section eff-summary"><h2>Summary</h2>
      <ul class="eff-summary-list">${bullets.map(bullet => `<li>${bullet}</li>`).join('')}</ul>
      ${key}</section>`;
  }

  function rate(value) {
    return value == null ? '—' : pct(value, 1);
  }

  function incompleteNotice(report) {
    const groups = new Map();
    for (const metric of report.metrics || []) {
      const n = Number(metric.overall?.n || 0);
      if (!metric.label || !n || n >= Number(report.n_items || 0)) continue;
      const counts = [n, metric.pass?.n || 0, metric.fail?.n || 0];
      const key = counts.join('/');
      if (!groups.has(key)) groups.set(key, {counts, labels: []});
      groups.get(key).labels.push(metric.label);
    }
    if (!groups.size) return '';
    const count = value => Number(value).toLocaleString();
    const rows = [...groups.values()].map(({counts: [n, correct, incorrect], labels}) =>
      `<li>${escapeHtml(labels.join(', '))}: ${count(n)} of ${count(report.n_items)} rollouts
        (${count(correct)} of ${count(report.n_pass)} correct,
        ${count(incorrect)} of ${count(report.n_fail)} incorrect)</li>`).join('');
    return `<div class="eff-warning" role="alert"><strong>Incomplete data.</strong>
      These metrics were recorded for only some rollouts. Their charts and summary results
      use only those rollouts, and the summary shows what share of rollouts each is based on.
      <ul>${rows}</ul></div>`;
  }

  function reportMarkup(report) {
    const pricing = report.pricing || {};
    const fromEval = report.source === 'evalrdl';
    const alerts = (report.alerts || []).map(alert =>
      `<p class="eff-warning" role="alert">${escapeHtml(alert)}</p>`).join('');
    const warnings = (report.warnings || []).map(warning =>
      `<p class="eff-warning">${escapeHtml(warning)}</p>`).join('');
    const provenanceRows = (report.provenance || []).map(item => {
      const present = Number(item.present || 0);
      const total = Number(item.total || 0);
      const partial = present > 0 && present < total;
      const status = partial ? 'partial' : item.status;
      const label = status === 'not_applicable' ? 'not applicable' : status;
      const coverage = status === 'not_applicable'
        ? '—' : `${present.toLocaleString()} / ${total.toLocaleString()}`;
      return `<tr><td>${escapeHtml(item.label)}</td>
        <td><span class="eff-status ${escapeHtml(status)}">${escapeHtml(label)}</span></td>
        <td>${coverage}</td>
        <td>${escapeHtml(item.detail)}</td></tr>`;
    }).join('');
    const provenanceIssues = (report.provenance || []).filter(item =>
      item.status !== 'not_applicable' &&
      (!item.present || Number(item.present) < Number(item.total))).length;
    const provenanceHint = provenanceIssues
      ? `${provenanceIssues} coverage ${provenanceIssues === 1 ? 'issue' : 'issues'}`
      : 'Show';
    const provenance = provenanceRows ? `<details class="eff-defs eff-provenance">
      <summary><span class="eff-defs-chevron" aria-hidden="true">▸</span>
        <span>Data coverage &amp; provenance</span>
        <span class="eff-defs-hint" data-closed="${provenanceHint}" data-open="Hide"></span>
      </summary>
      <p class="eff-provenance-note">EvalRDL supplies the recorded values. Derived and estimated
        rows state the exact transformation. Unavailable metrics are omitted rather than invented.</p>
      <table class="eff-table"><thead><tr><th>metric</th><th>status</th>
        <th>coverage</th><th>source or fallback</th></tr></thead>
      <tbody>${provenanceRows}</tbody></table></details>` : '';
    const priceNote = pricing.source === 'model_config'
      ? `Configured model prices: $${pricing.input_usd_per_million}/M input ·
         $${pricing.cached_usd_per_million}/M cached input ·
         $${pricing.output_usd_per_million}/M completion.`
      : `Simulated prices: $${pricing.input_usd_per_million}/M input ·
         $${pricing.cached_usd_per_million}/M cached input ·
         $${pricing.output_usd_per_million}/M completion.`;
    const realPrices = pricing.source === 'model_config';
    const costTitle = realPrices ? 'Cost' : 'Simulated cost';
    const costNote = realPrices
      ? priceNote
      : priceNote + ' Real rates replace these when efficiency_model_prices is set for the model.';
    const tokenNote = fromEval
      ? ''
      : 'Reported values are used when available; otherwise values are simulated from visible text.';
    const activityNote = fromEval
      ? ''
      : 'Turns group consecutive assistant events from one model response. Tool-output characters are exact.';
    const contextNote = fromEval
      ? ''
      : 'Problem input and final visible context use the stated four-characters-per-token estimate.';
    return `
      ${alerts}
      ${incompleteNotice(report)}
      ${provenanceRows ? '' : warnings}
      <div class="eff-kpis">
        <div class="kpi-card"><div class="kpi-eyebrow">Overall accuracy</div>
          <div class="kpi-value green">${rate(report.accuracy)}</div></div>
        <div class="kpi-card"><div class="kpi-eyebrow">${escapeHtml(report.best_of_label || '')}</div>
          <div class="kpi-value green">${rate(report.best_of_n)}</div></div>
        <div class="kpi-card"><div class="kpi-eyebrow">Tasks</div>
          <div class="kpi-value">${report.n_problems}</div></div>
        <div class="kpi-card"><div class="kpi-eyebrow">Rollouts</div>
          <div class="kpi-value">${report.n_items}</div></div>
      </div>
      ${provenance}
      ${summaryBox(report)}
      ${section(report, 'cost', costTitle, costNote)}
      ${section(report, 'tokens', 'Token usage', tokenNote)}
      ${section(report, 'activity', 'Agent activity', activityNote)}
      ${section(report, 'context', 'Context', contextNote)}
    `;
  }

  const originalBoot = boot;
  boot = function () {
    const report = (RAW.meta && RAW.meta.efficiency_metrics) || null;
    const layout = (RAW.meta && RAW.meta.efficiency_layout) || 'replace';
    originalBoot();
    const root = document.getElementById('cards');
    if (!root || !report) return;
    const markup = reportMarkup(report);
    queueMicrotask(() => mountCharts(report));
    if (layout === 'efficiency-tab') {
      const shell = root.closest('main') || root.parentNode;
      const tabs = document.createElement('nav');
      tabs.className = 'report-tabs';
      tabs.setAttribute('role', 'tablist');
      tabs.setAttribute('aria-label', 'Report sections');
      tabs.innerHTML = `
        <button type="button" class="report-tab" id="report-tab-efficiency"
          role="tab" aria-controls="report-panel-efficiency"
          aria-selected="true">Efficiency metrics</button>`;
      shell.insertBefore(tabs, shell.firstChild);
      document.getElementById('library-coverage')?.remove();
      document.getElementById('kpi-band')?.remove();
      document.getElementById('execution-status')?.remove();
      document.getElementById('input-scope')?.remove();
      root.classList.add('report-tab-panel');
      root.id = 'report-panel-efficiency';
      root.setAttribute('role', 'tabpanel');
      root.setAttribute('aria-labelledby', 'report-tab-efficiency');
      root.innerHTML = markup;
      return;
    }
    if (layout === 'tabs') {
      if (report.cards) addCardEfficiency(root, report);
      const shell = root.closest('main') || root.parentNode;
      const tabs = document.createElement('nav');
      tabs.className = 'report-tabs';
      tabs.setAttribute('role', 'tablist');
      tabs.setAttribute('aria-label', 'Report sections');
      tabs.innerHTML = `
        <button type="button" class="report-tab" id="report-tab-failures"
          role="tab" aria-controls="report-panel-failures">Failure modes</button>
        <button type="button" class="report-tab" id="report-tab-efficiency"
          role="tab" aria-controls="report-panel-efficiency">Efficiency metrics</button>`;

      const failures = document.createElement('section');
      failures.className = 'report-tab-panel';
      failures.id = 'report-panel-failures';
      failures.setAttribute('role', 'tabpanel');
      failures.setAttribute('aria-labelledby', 'report-tab-failures');
      [
        document.getElementById('library-coverage'),
        document.getElementById('kpi-band'),
        document.getElementById('failure-index'),
        root,
      ].filter(Boolean).forEach(node => failures.appendChild(node));

      const efficiency = document.createElement('section');
      efficiency.className = 'report-tab-panel';
      efficiency.id = 'report-panel-efficiency';
      efficiency.setAttribute('role', 'tabpanel');
      efficiency.setAttribute('aria-labelledby', 'report-tab-efficiency');
      efficiency.innerHTML = markup;

      shell.insertBefore(tabs, shell.firstChild);
      shell.appendChild(failures);
      shell.appendChild(efficiency);
      const buttons = Array.from(tabs.querySelectorAll('[role="tab"]'));
      const selectTab = name => {
        const showEfficiency = name === 'efficiency';
        failures.hidden = showEfficiency;
        efficiency.hidden = !showEfficiency;
        buttons[0].setAttribute('aria-selected', String(!showEfficiency));
        buttons[0].tabIndex = showEfficiency ? -1 : 0;
        buttons[1].setAttribute('aria-selected', String(showEfficiency));
        buttons[1].tabIndex = showEfficiency ? 0 : -1;
      };
      buttons[0].addEventListener('click', () => selectTab('failures'));
      buttons[1].addEventListener('click', () => selectTab('efficiency'));
      selectTab(location.hash === '#efficiency' ? 'efficiency' : 'failures');
      return;
    }
    if (layout === 'toggle') {
      const panel = document.createElement('details');
      panel.className = 'eff-toggle';
      panel.open = true;
      panel.innerHTML = '<summary>Efficiency metrics</summary>' + markup;
      const shell = root.closest('main') || root.parentNode;
      shell.insertBefore(panel, shell.firstChild);
      return;
    }
    document.getElementById('library-coverage')?.remove();
    document.getElementById('kpi-band')?.remove();
    document.getElementById('execution-status')?.remove();
    document.getElementById('input-scope')?.remove();
    root.innerHTML = markup;
  };
})();
"""


def _inject_overview(
    report_html: str,
    payload: dict[str, Any],
    *,
    sections: dict[str, dict[str, Any]],
    layout: str = "replace",
) -> str:
    match = _DATA_RE.search(report_html)
    if match is None:
        raise ValueError("failure-cards HTML is missing its data payload")
    data = json.loads(match.group(2))
    meta = data.setdefault("meta", {})
    if layout not in {"toggle", "tabs"}:
        meta.update({
            "n_items": int(payload.get("n_items") or 0),
            "n_failures": int(payload.get("n_fail") or 0),
            "base_fail_rate": (
                float(payload.get("n_fail") or 0) / int(payload.get("n_items") or 1)
            ),
        })
    meta.update(
        {
            "efficiency": True,
            "efficiency_layout": layout,
            "efficiency_metrics": {**payload, "sections": sections},
        }
    )
    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    report_html = report_html[: match.start(2)] + encoded + report_html[match.end(2) :]
    return _render_overview(report_html, layout=layout)


def _render_overview(report_html: str, *, layout: str) -> str:
    """Render saved efficiency metadata without recomputing metrics or cohort counts."""
    style_end = report_html.find("</style>")
    style_marker = report_html.find(_OVERVIEW_MARKER, 0, style_end)
    if style_marker >= 0:
        report_html = report_html[:style_marker] + report_html[style_end:]
    boot_marker = "\nboot();\n"
    boot_at = report_html.find(boot_marker)
    script_marker = report_html.find(_OVERVIEW_MARKER, report_html.find("</style>"))
    if script_marker >= 0 and boot_at >= script_marker:
        report_html = report_html[:script_marker] + report_html[boot_at:]
    report_html = report_html.replace("</style>", _OVERVIEW_CSS + "\n</style>", 1)
    report_html = report_html.replace("\nboot();\n", "\n" + _OVERVIEW_JS + "\nboot();\n", 1)
    if layout not in {"toggle", "tabs"}:
        report_html = report_html.replace("Model Failure Cards", "Efficiency metrics")
        report_html = report_html.replace(">Failure modes<", ">Efficiency metrics<", 1)
    return report_html


_EVALRDL_SECTIONS = {
    "cost": ("simulated_cost_usd",),
    "tokens": (
        "input_tokens",
        "cached_tokens",
        "uncached_tokens",
        "reasoning_tokens",
        "answer_tokens",
        "output_tokens",
        "total_tokens",
    ),
    "activity": (
        "n_turns",
        "n_tool_calls",
        "avg_tool_calls_per_turn",
        "tool_arg_chars",
        "tool_output_chars",
    ),
    "context": ("prompt_tokens", "final_context_tokens"),
}


def _evalrdl_sections(
    run: Any,
    pricing: dict[str, Any] | None = None,
    *,
    detectors: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from autofa.data.evalrdl_efficiency import collect_evalrdl_efficiency

    payload = collect_evalrdl_efficiency(run, pricing=pricing)
    samples = payload.pop("_samples")
    attempts = payload.pop("_attempts", None)
    if detectors and attempts:
        payload["cards"] = _card_sections(payload, attempts, detectors, _item_attempts(run))
    present = {metric["name"] for metric in payload.get("metrics") or []}
    sections = {}
    for key, requested in _EVALRDL_SECTIONS.items():
        if key == "activity" and not payload.get("has_agent_activity"):
            continue
        if not any(name in present for name in requested):
            continue
        section = _chart_section(
            payload,
            samples,
            requested,
            columns=3 if key in {"tokens", "activity"} else 2,
        )
        if section["metrics"]:
            sections[key] = section
    payload["summary"] = _summary(payload, sections)
    return payload, sections


_SUMMARY_MIN_ATTEMPTS = 20
_SUMMARY_MIN_EFFECT = 0.1
_SUMMARY_MIN_RELATIVE = 0.05
_SUMMARY_EFFECT_SIZES = (("large", 0.4), ("moderate", 0.2), ("small", _SUMMARY_MIN_EFFECT))
_SUMMARY_SECTION_TITLES = {
    "tokens": "Token usage", "activity": "Agent activity", "context": "Context",
}


def _summary(payload: dict[str, Any], sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify each charted metric with fixed rules for the report summary.

    A metric is eligible when it has enough recorded attempts in both groups. An eligible
    significant metric is negligible when its KS statistic is below the minimum effect
    or its medians differ by less than the minimum relative amount; otherwise its KS
    statistic sets the effect size. A section is named instead of its metrics only when
    all of them differ in the same direction with the same effect size. Cost is reported
    as shares of attempts and spend rather than listed. Metrics recorded for only some
    attempts keep their coverage so the summary can say what they are based on.
    """
    metric_data = {metric["name"]: metric for metric in payload.get("metrics") or []}
    differ, negligible, not_significant, insufficient = [], [], [], []
    partial: dict[str, float] = {}
    cost = None
    for key, section in sections.items():
        section_differ = []
        section_coverage = []
        names = section["metrics"]
        for name in names:
            metric = metric_data[name]
            label = section["charts"][name]["label"]
            correct, incorrect = metric.get("pass") or {}, metric.get("fail") or {}
            test = metric.get("permutation_test") or {}
            coverage = float(metric.get("coverage") or 0)
            section_coverage.append(coverage)
            if coverage < 1:
                partial[label] = coverage
            if (
                min(int(correct.get("n") or 0), int(incorrect.get("n") or 0))
                < _SUMMARY_MIN_ATTEMPTS
                or test.get("statistic") is None
            ):
                insufficient.append(label)
                continue
            if name == "simulated_cost_usd":
                spent = float(correct.get("total") or 0) + float(incorrect.get("total") or 0)
                if spent > 0:
                    attempts = int(correct["n"]) + int(incorrect["n"])
                    cost = {
                        "attempt_share": int(incorrect["n"]) / attempts,
                        "cost_share": float(incorrect.get("total") or 0) / spent,
                        **({"coverage": coverage} if coverage < 1 else {}),
                    }
                continue
            if not test.get("significant"):
                not_significant.append(label)
                continue
            low, high = float(correct.get("median") or 0), float(incorrect.get("median") or 0)
            scale = max(abs(low), abs(high))
            relative = abs(high - low) / scale if scale else 0.0
            if float(test["statistic"]) < _SUMMARY_MIN_EFFECT or relative < _SUMMARY_MIN_RELATIVE:
                negligible.append(label)
                continue
            effect = float(test["statistic"])
            section_differ.append({
                "label": label,
                "direction": "up" if high > low else "down",
                "effect": effect,
                "size": next(size for size, floor in _SUMMARY_EFFECT_SIZES if effect >= floor),
            })
        agree = {(item["direction"], item["size"]) for item in section_differ}
        if len(names) > 1 and len(section_differ) == len(names) and len(agree) == 1:
            title = _SUMMARY_SECTION_TITLES.get(key, key)
            differ.append({
                **max(section_differ, key=lambda item: item["effect"]),
                "label": title,
                "section": True,
            })
            if min(section_coverage) < 1:
                partial[title] = min(section_coverage)
        else:
            differ.extend(section_differ)
    differ.sort(key=lambda item: -item["effect"])
    return {
        "differ": differ,
        "cost": cost,
        "negligible": negligible,
        "not_significant": not_significant,
        "insufficient": insufficient,
        "partial": partial,
    }


_CARD_CHART_POINTS = 200


def _item_attempts(run: Any) -> dict[str, tuple[str, int]]:
    """Map the run's item IDs to the attempts they identify."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autofa.data.parquet_store import COLUMN_ALIASES, apply_column_aliases, make_item_id

    identity = ("model_id", "benchmark", "problem_id", "run_idx")
    wanted = {*identity, *(alias for name in identity for alias in COLUMN_ALIASES.get(name, ()))}
    paths = [
        path / "dataset.parquet" if path.is_dir() else path
        for path in map(Path, getattr(run, "parquet_paths", None) or [])
    ]
    attempts: dict[str, tuple[str, int]] = {}
    for path in paths:
        try:
            names = pq.read_schema(path).names
            rows = pq.read_table(
                path, columns=[name for name in names if name in wanted],
            ).to_pylist()
        except (OSError, ValueError, pa.ArrowException):
            continue
        for row in rows:
            apply_column_aliases(row)
            if row.get("problem_id") is None or row.get("run_idx") is None:
                continue
            item_id = make_item_id(row, path, include_source=len(paths) > 1)
            attempts[item_id] = (str(row["problem_id"]), int(row["run_idx"]))
    return attempts


def _card_sections(
    payload: dict[str, Any],
    attempts: dict[str, Any],
    detectors: dict[str, dict[str, Any]],
    item_attempts: dict[str, tuple[str, int]],
) -> dict[str, dict[str, Any]]:
    """Compare each failure mode's matched attempts with correct and other failed attempts.

    A card needs at least the summary's minimum of matched failed attempts; its significance
    test compares the matched attempts with the correct ones.
    """
    from autofa.data.evalrdl_efficiency import CARD_METRICS

    metric_data = {metric["name"]: metric for metric in payload.get("metrics") or []}
    names = [name for name in CARD_METRICS if name in metric_data and name in attempts["values"]]
    groups = attempts["groups"]
    tasks, repeats = attempts["task_id"], attempts["repeat"]
    row_of = {key: row for row, key in enumerate(zip(tasks, repeats, strict=True))}
    cards = {}
    for card, detector in detectors.items():
        matches = detector.get("matches")
        rows = {
            row for row in (row_of.get(item_attempts.get(item_id)) for item_id in matches or ())
            if row is not None and groups[row] == "fail"
        }
        if matches is None or (matches and not rows):
            cards[card] = {"unavailable": True}
            continue
        entry: dict[str, Any] = {
            "n": len(rows),
            "min_attempts": _SUMMARY_MIN_ATTEMPTS,
            "sampled": bool(detector.get("sampled")),
        }
        cards[card] = entry
        if len(rows) < _SUMMARY_MIN_ATTEMPTS:
            continue
        charts = {}
        for name in names:
            values = attempts["values"][name]
            split: dict[str, list[float]] = {"correct": [], "card": [], "other": []}
            clusters: dict[tuple[str, str, str], list[tuple[int, float, bool]]] = defaultdict(list)
            for row, value in enumerate(values):
                number = _number(value)
                if number is None or groups[row] not in {"pass", "fail"}:
                    continue
                group = (
                    "correct" if groups[row] == "pass" else "card" if row in rows else "other"
                )
                split[group].append(number)
                if group != "other":
                    clusters[("evalrdl", "", tasks[row])].append(
                        (repeats[row], number, group == "correct"),
                    )
            shown = {
                group: _chart_group(
                    f"{card}:{name}", group, raw, max_points=_CARD_CHART_POINTS,
                )
                for group, raw in split.items()
            }
            if not shown["card"] or not shown["correct"]:
                continue
            test = _problem_cluster_permutation_test(clusters, metric=f"{card}:{name}")
            charts[name] = {
                "label": _chart_label(name, metric_data[name]),
                "unit": str(metric_data[name].get("unit") or ""),
                **shown,
                "significant": bool((test or {}).get("significant")),
            }
        entry["metrics"] = list(charts)
        entry["charts"] = charts
    return cards


def write_efficiency_card(
    run: Any,
    output_dir: Path,
    *,
    source_harness: str = "unknown",
    reference_data: list[dict[str, str]] | None = None,
    pricing: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write a standalone efficiency report from an EvalRDL run."""
    from autofa.core.library import FunctionLibrary
    from autofa.reports.failure_cards import write_failure_cards

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, sections = _evalrdl_sections(run, pricing=pricing)
    data_path = output_dir / DATA_NAME
    html_path = output_dir / REPORT_NAME
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_library = FunctionLibrary(
        model_id=run.library_model_id(),
        benchmark=run.benchmark_label(),
    )
    write_failure_cards(
        report_library,
        out=html_path,
        dataset=None,
        source_harness=source_harness,
        reference_data=reference_data,
    )
    rendered = _inject_overview(
        html_path.read_text(encoding="utf-8"),
        payload,
        sections=sections,
        layout="efficiency-tab",
    )
    html_path.write_text(rendered, encoding="utf-8")
    return html_path, data_path, payload


def _report_detectors(report_html: str, library: Any) -> dict[str, dict[str, Any]] | None:
    """Matched failed attempts of the detectors that have cards in this report."""
    match = _DATA_RE.search(report_html)
    if library is None or match is None:
        return None
    from autofa.metrics.task_support import matched_failed_attempts

    names = {str(fn.get("name")) for fn in json.loads(match.group(2)).get("functions") or []}
    return {
        fn.name: {
            "matches": matched_failed_attempts(fn.scores),
            "sampled": bool((fn.scores or {}).get("sampled")),
        }
        for fn in library.functions
        if fn.name in names
    }


def attach_saved_efficiency(html_path: Path, library: Any = None) -> bool:
    """Embed EvalRDL efficiency when this run asked for both analyses.

    With the report's ``library``, failure cards also compare their matched attempts.
    """
    html_path = Path(html_path)
    if not (html_path.parent / "config.json").is_file():
        return False
    from autofa.config import RunConfig

    try:
        run = RunConfig.load(html_path.parent.name)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    selected = set(getattr(run, "analysis_outputs", None) or [])
    if "efficiency" not in selected or "agent" not in selected:
        return False
    from autofa.config import load_config

    try:
        cfg = load_config()
    except (OSError, ValueError, TypeError):
        cfg = None
    report_html = html_path.read_text(encoding="utf-8")
    payload, sections = _evalrdl_sections(
        run,
        pricing=resolve_pricing(cfg, getattr(run, "model_id", "")),
        detectors=_report_detectors(report_html, library),
    )
    cards = payload.pop("cards", None)
    data_path = html_path.parent / DATA_NAME
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rendered = _inject_overview(
        report_html,
        {**payload, **({"cards": cards} if cards else {})},
        sections=sections,
        layout="tabs",
    )
    html_path.write_text(rendered, encoding="utf-8")
    return True
