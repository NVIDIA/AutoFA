# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pareto frontier over configurable objective axes (default 3D)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.defaults import default_value, setting_value


def _axes_from_config(config=None) -> list[str]:
    try:
        from autofa.config import load_config, resolve_objective_axes

        return resolve_objective_axes(config or load_config())
    except Exception:  # noqa: BLE001
        return default_value("objective_axes")


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if a dominates b (all ≥ and at least one >)."""
    if len(a) != len(b):
        raise ValueError("point dimension mismatch")
    return all(x >= y for x, y in zip(a, b, strict=True)) and any(
        x > y for x, y in zip(a, b, strict=True)
    )


def point_from_scores(
    scores: dict[str, Any],
    axes: Sequence[str] | None = None,
) -> tuple[float, ...] | None:
    ax = list(axes) if axes is not None else _axes_from_config()
    from autofa.metrics.verdicts import behavior_ranking_eligibility

    if not behavior_ranking_eligibility(scores, ax)["eligible"]:
        return None
    out: list[float] = []
    for name in ax:
        if name in scores and scores[name] is not None:
            out.append(float(scores[name]))
        elif name == "recall":
            out.append(float(scores.get("sensitivity") or 0.0))
        elif name in {"precision", "raw_ppv"}:
            out.append(float(scores.get("raw_ppv") or scores.get("precision") or 0.0))
        elif name == "pct_caught":
            from autofa.metrics.association import pct_caught_from_scores

            out.append(float(pct_caught_from_scores(scores)))
        else:
            out.append(0.0)
    return tuple(out)


def _function_scores(
    fn: Any, config=None, *, behavior_strictness: str | None = None,
) -> dict[str, Any]:
    """Check a function's claim fingerprint before using its saved objectives."""
    data = fn if isinstance(fn, dict) else vars(fn)
    scores = dict(data.get("scores") or {})
    if "source" in data:
        scores["behavior_requires_validation"] = True
    if "source" in data:
        from autofa.metrics.verdicts import current_behavior_scores
        from autofa.objectives import discovery_strategy

        return current_behavior_scores(
            scores, source=review_source(data), description=data.get("description") or "",
            mechanism=data.get("mechanism") or data.get("description") or data.get("name") or "",
            objective_strategy=discovery_strategy(config) if config is not None else None,
            behavior_strictness=(behavior_strictness if behavior_strictness is not None else
                                 strictness_for_config(config) if config is not None else None),
        )
    return scores


def pareto_front(
    points: list[tuple[float, ...]] | list[tuple[float, float, float]],
) -> list[tuple[float, ...]]:
    front: list[tuple[float, ...]] = []
    typed = [tuple(float(x) for x in p) for p in points]
    for p in typed:
        if any(dominates(q, p) for q in typed if q is not p):
            continue
        front.append(p)
    return sorted(set(front), reverse=True)


def frontier_value(
    points: list[tuple[float, ...]] | list[Sequence[float]] | None,
    *, config, axes: Sequence[str] | None = None,
) -> float:
    """Mean quality across frontier points, using the same scalar as detector ranking."""
    if not points:
        return 0.0
    front = pareto_front([tuple(float(x) for x in p) for p in points])
    if not front:
        return 0.0
    from autofa.objectives import weights_by_axis

    ax = list(axes) if axes is not None else _axes_from_config(config)
    weights = weights_by_axis(config)
    if any(axis not in weights for axis in ax):
        raise ValueError("Frontier axes must have configured weights")
    if any(len(point) != len(ax) for point in front):
        raise ValueError("Frontier point dimension does not match objective axes")
    values = [_scalar_score(point, weights=[weights[axis] for axis in ax]) for point in front]
    return float(sum(values) / len(values))


def hypervolume(
    points: list[tuple[float, ...]] | list[Sequence[float]] | None,
) -> float:
    """Hypervolume of the Pareto front relative to the origin.

    Assumes higher-is-better axes in [0, 1]. Uses inclusion of orthants from
    each non-dominated point down to the origin (exact for small fronts).
    """
    if not points:
        return 0.0
    front = pareto_front([tuple(float(x) for x in p) for p in points])
    if not front:
        return 0.0
    dim = len(front[0])
    # Clip negative contributions.
    shifted = [tuple(max(0.0, v) for v in p) for p in front]
    shifted = [p for p in shifted if any(v > 0 for v in p)]
    if not shifted:
        return 0.0
    return float(_hypervolume_recursive(shifted, dim))


def _hypervolume_recursive(points: list[tuple[float, ...]], dim: int) -> float:
    """Integrate slices of maximization orthants from the origin."""
    if not points or dim == 0:
        return 0.0
    if dim == 1:
        return max(p[0] for p in points)
    # Sweep upward; every point reaching the slice's upper height contributes.
    pts = sorted(points, key=lambda p: p[dim - 1])
    volume = 0.0
    prev_h = 0.0
    for i, p in enumerate(pts):
        h = p[dim - 1]
        if h <= prev_h:
            continue
        # The suffix includes all points tied at h on the first visit to that height.
        slice_pts = [q[: dim - 1] for q in pts[i:]]
        # Non-dominated in the remaining dimensions.
        nd: list[tuple[float, ...]] = []
        for q in slice_pts:
            if any(all(a >= b for a, b in zip(r, q, strict=True)) and any(
                a > b for a, b in zip(r, q, strict=True)
            ) for r in slice_pts if r is not q):
                continue
            nd.append(q)
        volume += (h - prev_h) * _hypervolume_recursive(nd, dim - 1)
        prev_h = h
    return volume


def snapshot_frontier_metrics(
    points: list[tuple[float, ...]] | list[Sequence[float]] | None,
    *, config, axes: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute frontier_value + hypervolume for a scores_history snapshot."""
    typed = [tuple(float(x) for x in p) for p in (points or [])]
    return {
        "frontier_value": frontier_value(typed, config=config, axes=axes),
        "hypervolume": hypervolume(typed),
    }


def _scores_look_evaluated(scores: dict[str, Any] | None) -> bool:
    """True when ``scores`` look like a real eval (not an empty placeholder)."""
    if not scores:
        return False
    from autofa.metrics.completeness import incomplete_evidence

    if incomplete_evidence(scores):
        return False
    if scores.get("evaluated"):
        return True
    if scores.get("n_items") is not None:
        return True
    return any(
        k in scores and scores[k] is not None
        for k in ("ppv_at_pi0", "task_recall", "recall", "a", "n_fail_caught")
    )


def library_frontier_points(
    functions: Sequence[Any],
    axes: Sequence[str] | None = None,
    *, config=None,
) -> list[tuple[float, ...]]:
    """Objective points for every library member that already has eval scores.

    Used so partial evals (e.g. ``submit_function`` auto-eval of one new detector)
    still report frontier quality over the **full working library**, with the
    newly scored function merged in via its updated ``fn.scores``.
    """
    if axes is None and config is not None:
        axes = _axes_from_config(config)
    points: list[tuple[float, ...]] = []
    for fn in functions or []:
        scores = _function_scores(fn, config)
        if not _scores_look_evaluated(scores):
            continue
        point = point_from_scores(scores, axes=axes)
        if point is not None:
            points.append(point)
    return points


def library_frontier_metrics(
    functions: Sequence[Any],
    axes: Sequence[str] | None = None,
    *, config=None,
) -> dict[str, Any]:
    """Frontier metrics + points for the full scored library."""
    from autofa.config import load_config

    cfg = config or load_config()
    ax = list(axes) if axes is not None else _axes_from_config(cfg)
    points = library_frontier_points(functions, axes=ax, config=config)
    candidates = sum(
        not _ranking_eligibility(_function_scores(fn, config), ax)["eligible"]
        for fn in functions
    )
    metrics = snapshot_frontier_metrics(points, config=cfg, axes=ax)
    return {
        **metrics,
        "points": points,
        "n_scored": len(points),
        "n_rank_eligible": len(points),
        "n_research_candidates": candidates,
        "ranking_eligible": candidates == 0,
        "objective_axes": ax,
    }


def _ranking_eligibility(scores: dict[str, Any], axes: Sequence[str]) -> dict[str, Any]:
    from autofa.metrics.completeness import incomplete_evidence
    from autofa.metrics.verdicts import behavior_ranking_eligibility

    ranking = behavior_ranking_eligibility(scores, axes)
    if (incomplete_evidence(scores)
            or (scores.get("behavior_requires_validation") and not _scores_look_evaluated(scores))):
        return {**ranking, "eligible": False, "status": "evaluation_incomplete",
                "behavior_status": ranking["status"],
                "reason": "Dataset evaluation is incomplete or unavailable; retain as a "
                          "research candidate until it completes."}
    return ranking


def function_ranking_eligibility(
    function: Any, axes: Sequence[str], *, config=None, behavior_strictness: str | None = None,
) -> dict[str, Any]:
    """Resolve claim provenance and eligibility without imposing optimizer order.

    With no active config, assess the saved judge strategy. Reports use this to
    retain their own display policy regardless of the current search preference.
    Live displays may supply only strictness to retain that objective independence.
    """
    return _ranking_eligibility(
        _function_scores(function, config, behavior_strictness=behavior_strictness), axes,
    )


def _scalar_score(values: Sequence[float], *, weights: Sequence[float]) -> float:
    """Weighted geometric mean; a zero on an active objective gives zero."""
    if not values:
        return 0.0
    active_weights = list(weights) if weights else [1.0] * len(values)
    if len(active_weights) != len(values):
        raise ValueError("Each objective needs one weight")
    if any(not math.isfinite(w) or w < 0 for w in active_weights):
        raise ValueError("Objective weights must be finite and non-negative")
    total = sum(active_weights)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("At least one objective must have a positive weight")
    active = [(float(value), weight / total) for value, weight in
              zip(values, active_weights, strict=True) if weight > 0]
    if any(not math.isfinite(value) or value < 0 for value, _ in active):
        raise ValueError("Geometric objectives require finite, non-negative values")
    if any(value == 0 for value, _ in active):
        return 0.0
    # Logs avoid intermediate underflow for small positive recall.
    return float(math.exp(sum(weight * math.log(value) for value, weight in active)))


def rank_functions(
    functions: list[dict[str, Any]],
    top_x: int | None = None,
    *,
    config=None,
) -> list[dict[str, Any]]:
    """Rank eligible functions, then retain unranked research candidates for inspection.

    Candidate order is stable, not a fabricated numeric quality ranking.
    """
    from autofa.config import load_config
    from autofa.objectives import weights_by_axis

    cfg = config or load_config()
    if top_x is None:
        top_x = setting_value(cfg, "top_x")
    ax = _axes_from_config(cfg)
    eligible, candidates = [], []
    for fn in functions:
        ranking = _ranking_eligibility(_function_scores(fn, config), ax)
        row = {**fn, "ranking": ranking}
        (eligible if ranking["eligible"] else candidates).append(row)
    weights = weights_by_axis(cfg)

    def weighted_key(fn: dict[str, Any]) -> float:
        point = point_from_scores(_function_scores(fn, config), ax)
        assert point is not None
        return _scalar_score(point, weights=[weights[axis] for axis in ax])

    return (sorted(eligible, key=weighted_key, reverse=True) + candidates)[:top_x]


def objective_summary(config=None) -> str:
    from autofa.config import load_config
    from autofa.objectives import objective_guidance

    return objective_guidance(config or load_config())
