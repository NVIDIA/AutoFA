# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live terminal progress chart for frontier evolution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autofa.ui.theme import GREEN

_SPARK = "▁▂▃▄▅▆▇█"


def coverage_grade_label(coverage: dict[str, Any] | None) -> str:
    """Describe the saved union's cutoff."""
    minimum_grade = (coverage or {}).get("minimum_grade")
    return {"OK": "OK-or-better", "GOOD": "Good-or-better",
            "EXCELLENT": "Excellent-or-better", "OUTSTANDING": "Outstanding"}.get(
                minimum_grade, str(minimum_grade))


def library_coverage_rows(coverage: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Headline match unions, separate from the much smaller reviewed subset."""
    data = coverage or {}
    grade_label = coverage_grade_label(data)
    count = data.get("n_eligible_detectors", 0)
    lower_bound = data.get("status") == "observed_lower_bound"
    scope = "observed lower bound; sampled/partial evidence" if lower_bound else "full scan"
    if data.get("status") == "exact" and count == 0:
        scope = f"no {grade_label} detectors yet"
    def display(key: str, matched_key: str, total_key: str, units: str) -> str:
        value = data.get(key)
        if not data.get("available") or value is None:
            reason = data.get("task_reason") if key == "task_coverage" else None
            return "— · " + str(reason or data.get("reason") or "source evidence unavailable")
        matched, total = data.get(matched_key), data.get(total_key)
        counts = (f"{matched}/{total} {units}" if matched is not None and total is not None
                  else units)
        percentage = f"{100 * value:.3g}%" if 0 < value < .0005 else f"{value:.1%}"
        if lower_bound:
            percentage = f"≥{percentage}"
        detectors = f"{grade_label} detector" + ("" if count == 1 else "s")
        return f"{percentage} · {counts} · {count} {detectors} · {scope}"

    return [
        ("Library coverage", display("coverage", "n_matched_failed_attempts",
                                     "n_failed_attempts", "failed rollouts")),
        ("Task coverage", display("task_coverage", "n_matched_failed_tasks",
                                  "n_failed_tasks", "distinct failed tasks")),
    ]


def sparkline(values: Sequence[float], *, width: int = 24) -> str:
    """Unicode sparkline for a numeric series (empty → '—')."""
    if not values:
        return "—"
    xs = [float(v) for v in values]
    if width > 0 and len(xs) > width:
        # Downsample by taking evenly spaced points including ends.
        idxs = [round(i * (len(xs) - 1) / (width - 1)) for i in range(width)]
        xs = [xs[i] for i in idxs]
    lo = min(xs)
    hi = max(xs)
    if hi <= lo:
        return _SPARK[0] * len(xs)
    out = []
    for v in xs:
        t = (v - lo) / (hi - lo)
        out.append(_SPARK[min(len(_SPARK) - 1, int(t * (len(_SPARK) - 1)))])
    return "".join(out)


def history_series(scores_history: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Extract time series from scores_history snapshots."""
    frontier: list[float] = []
    n_fns: list[float] = []
    hv: list[float] = []
    for snap in scores_history or []:
        frontier.append(float(snap.get("frontier_value") or 0))
        n_fns.append(float(snap.get("n_functions") or 0))
        hv.append(float(snap.get("hypervolume") or 0))
    return {"frontier_value": frontier, "n_functions": n_fns, "hypervolume": hv}


def latest_frontier_stats(scores_history: list[dict[str, Any]]) -> dict[str, Any]:
    series = history_series(scores_history)
    fv = series["frontier_value"]
    nf = series["n_functions"]
    cur_fv = fv[-1] if fv else 0.0
    prev_fv = fv[-2] if len(fv) >= 2 else None
    best_fv = max(fv) if fv else 0.0
    delta = None if prev_fv is None else cur_fv - prev_fv
    return {
        "n_experiments": len(fv),
        "frontier_value": cur_fv,
        "frontier_delta": delta,
        "best_frontier_value": best_fv,
        "n_functions": int(nf[-1]) if nf else 0,
        "hypervolume": series["hypervolume"][-1] if series["hypervolume"] else 0.0,
        "series": series,
    }


def _delta_arrow(delta: float | None, *, eps: float = 1e-4) -> str:
    if delta is None:
        return ""
    if delta > eps:
        return f"↑{delta:+.3f}"
    if delta < -eps:
        return f"↓{delta:+.3f}"
    return f"→{delta:+.3f}"


def display_progress_rows(
    scores_history: list[dict[str, Any]], *, current: dict[str, Any] | None = None,
    grade_counts: dict[str, int] | None = None,
    task_breadth: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Shared CLI presentation; sampled search is distinct from validation."""
    from autofa.metrics.live_progress import display_progress_stats

    stats = display_progress_stats(scores_history, current=current)
    snapshot = stats["current"] or {}
    search = snapshot.get("search") or {}
    validated = snapshot.get("validated") or {}
    if search.get("status") == "available" and search.get("value") is not None:
        value = search["value"]
        delta = _delta_arrow(stats["search_delta"])
        scope = {"fixed_scoring_panel": "scoring panel",
                 "full_selected_corpus": "full corpus"}.get(search.get("scope"), "observed scores")
        best = stats["best_search_value"]
        best_text = f"  (series peak {best:.3f})" if best is not None else ""
        search_text = (
            f"{value:.3f}{'  ' + delta if delta else ''}{best_text} · provisional, {scope}"
        )
        if not stats["search_values"]:
            search_text += " · " + (search.get("comparison_reason") or "Trend unavailable.")
    else:
        reason = search.get("reason") or "No saved display history; awaiting current scores."
        search_text = f"{search.get('status', 'unavailable')} · {reason}"
    validated_text = (
        f"{validated['value']:.3f} · {validated['n_ranked']} eligible detectors"
        if validated.get("status") == "available" and validated.get("value") is not None
        else "pending · no eligible detectors" if validated
        else "unavailable · awaiting current scores"
    )
    rows = library_coverage_rows(snapshot["library_quality_coverage"]) if (
        "library_quality_coverage" in snapshot
    ) else []
    rows += [("best search score", search_text), ("validated frontier", validated_text)]
    if stats["search_values"]:
        rows.append(("search trend", sparkline(stats["search_values"])))
    if grade_counts is not None:
        rows.append(("library quality", " · ".join(
            f"{grade_counts.get(code, 0)} {label}" for code, label in (
                ("GOOD", "Good"), ("OK", "OK"), ("WEAK", "Weak"),
                ("EMPTY", "No matches"), ("UNSCORED", "Not evaluated"),
            )
        )))
    if task_breadth:
        # This is the last runtime assessment, never a sum of per-detector
        # supports (which would double-count overlaps) or a new corpus scan.
        confirmed = task_breadth.get("n_confirmed_failed_tasks")
        total = task_breadth.get("n_failed_tasks")
        if (isinstance(confirmed, int) and not isinstance(confirmed, bool)
                and isinstance(total, int) and not isinstance(total, bool)
                and 0 <= confirmed <= total):
            rows.append(("confirmed tasks", f"{confirmed}/{total} failed tasks · "
                         "observed union, last round assessment"))
    rows.append(("progress", f"n_fns={snapshot.get('n_functions', '—')}  "
                 f"evals={stats['n_evaluations']}"))
    return rows


def render_terminal_progress(
    scores_history: list[dict[str, Any]],
    console: Console | None = None,
    *,
    title: str = "frontier progress",
    current: dict[str, Any] | None = None,
) -> None:
    """Print the same provisional/validated status as the live discovery panel."""
    c = console or Console()
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=18)
    table.add_column()
    for label, value in display_progress_rows(scores_history, current=current):
        if label in {"Library coverage", "Task coverage"}:
            from rich.text import Text

            table.add_row(Text(label, style="bold"), Text(value, style=f"bold {GREEN}"))
        else:
            table.add_row(label, value)
    c.print(Panel(table, title=f"[bold {GREEN}]{title}[/]", border_style=GREEN))
