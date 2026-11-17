# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative association scores for discovery when detectors honestly abstain."""

from __future__ import annotations

from math import isclose, isfinite
from typing import Any

from autofa.metrics.association import contingency
from autofa.metrics.completeness import incomplete_evidence


def _number(value: Any, *, count: bool = False) -> float | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0 or (count and int(value) != value)):
        return None
    return float(value)


def _outcome_total(interval: Any, positive: float, resolved: float) -> float | None:
    """Recover weighted outcome totals without treating abstentions as negatives."""
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return None
    low, high = (_number(value) for value in interval)
    if low is None or high is None or not 0 <= low <= high <= 1 or high - low >= 1:
        return None
    total = resolved / (1 - high + low)
    if total <= 0 or not isclose(low, positive / total, abs_tol=1e-12):
        return None
    return total


def search_association(scores: dict) -> dict | None:
    """Return a search-only copy; unresolved executions never become useful scores.

    For fully accounted abstentions, assume every abstained failed attempt is
    missed and every abstained passing attempt is flagged. This lowers recall
    and precision together. Saved evaluation metrics and keeper gates stay intact.
    """
    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or (scores.get("evaluation_scope_complete") is not True
                and scores.get("eval_mode") != "capped")
            or any(_number(scores.get(key, 0), count=True) != 0 for key in (
                "errors", "n_errors", "parent_errors",
            ))):
        return None
    abstained = _number(scores.get("n_insufficient", 0), count=True)
    if (abstained is None
            or _number(scores.get("n_unresolved", abstained), count=True) != abstained):
        return None
    if not abstained:
        if incomplete_evidence(scores) or scores.get("dataset_evaluation_complete") is False:
            return None
        return dict(scores)

    resolved = _number(scores.get("n_scored"), count=True)
    cells = [_number(scores.get(key)) for key in ("a", "b", "c", "d")]
    if resolved is None or not resolved or any(value is None for value in cells):
        return None
    a, b, c, d = cells
    n_items = _number(scores.get("n_items"))
    coverage = _number(scores.get("assessment_coverage"))
    attempted = resolved + abstained
    if (n_items is None or not isclose(n_items, sum(cells))
            or coverage is None or not isclose(coverage, resolved / attempted)
            or not isinstance(scores.get("sampled"), bool)):
        return None
    if scores["sampled"]:
        selected = scores.get("evaluation_item_ids")
        frame_n = _number(scores.get("sample_frame_n"), count=True)
        if (not isinstance(selected, list)
                or any(not isinstance(item_id, str) or not item_id for item_id in selected)
                or len(selected) != attempted or len(set(selected)) != len(selected)
                or _number(scores.get("sample_n"), count=True) != attempted
                or frame_n is None or frame_n < attempted):
            return None
    else:
        frame_n = attempted
        if not isclose(resolved, sum(cells)):
            return None

    bounds = scores.get("unresolved_bounds") or scores.get("abstention_bounds")
    if not isinstance(bounds, dict):
        return None
    n_fail = _outcome_total(bounds.get("recall"), a, a + c)
    n_pass = _outcome_total(bounds.get("fpr"), b, b + d)
    if (n_fail is None or n_pass is None or n_fail + 1e-8 < a + c
            or n_pass + 1e-8 < b + d or not isclose(n_fail + n_pass, frame_n)):
        return None
    missing_pass = max(0.0, n_pass - b - d)
    conservative = contingency(a, b + missing_pass, n_fail - a, d)
    return {
        **scores, **conservative,
        "search_association_scope": "worst_case_abstentions",
        "search_abstention_n": int(abstained),
    }
