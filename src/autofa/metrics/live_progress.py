# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display-only discovery progress, separate from validated stopping history."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from time import time
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.metrics.pareto import library_frontier_metrics
from autofa.metrics.search_ranking import search_ranking
from autofa.objectives import weights_by_axis
from autofa.util.cancellation import cancellation_guard

_LOG = logging.getLogger(__name__)
_VERSION = 3


def _count(value: Any) -> int | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        return None
    return int(value)


def _evaluation_cohort(scores: Mapping[str, Any]) -> list[Any] | None:
    """Identify association evidence, independently of the task-breadth panel.

    A panel fingerprint describes projected task breadth, not necessarily the
    rows used for precision or attempt recall. Counts alone cannot identify a
    sample or capped scan; older records without provenance break the trend.
    """
    corpus = scores.get("task_recall_corpus_fingerprint")
    run_id = scores.get("evaluation_run_id")
    if not isinstance(corpus, str) or not corpus or not run_id:
        return None
    if (scores.get("sampled") is False
            and scores.get("task_recall_scope") == "full_selected_corpus"
            and scores.get("evaluation_scope_complete") is True
            and not scores.get("eval_truncated")):
        return [run_id, corpus, "full_selected_corpus"]
    ids = scores.get("evaluation_item_ids")
    resolved = _count(scores.get("n_scored"))
    abstained = _count(scores.get("n_insufficient", 0))
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(item_id, str) or not item_id for item_id in ids)
            or len(set(ids)) != len(ids) or resolved is None or abstained is None
            or len(ids) != resolved + abstained):
        return None
    if scores.get("sampled") is True:
        frame_n = _count(scores.get("sample_frame_n"))
        if (_count(scores.get("sample_n")) != len(ids)
                or frame_n is None or frame_n < len(ids)):
            return None
        return [run_id, corpus, "sample", sorted(ids), frame_n, scores.get("sample_strata"),
                scores.get("sample_allocation_version")]
    if scores.get("sampled") is False:
        return [run_id, corpus, "selected_items", sorted(ids)]
    return None


def display_progress_snapshot(
    functions: Sequence[Any], config: Any, *, run_id: str | None = None,
) -> dict[str, Any]:
    """Read current evidence without evaluating, judging or promoting candidates."""
    ranking = search_ranking(functions, config, run_id=run_id)
    axes = ranking["objective_axes"]
    weights = weights_by_axis(config)
    objective = {"axes": axes, "rank": config.objective_rank,
                 "weights": [weights.get(axis, 1.0) for axis in axes],
                 "behavior_strictness": strictness_for_config(config)}
    ranked, unranked = ranking["ranked"], ranking["unranked"]
    search = {
        "status": "pending", "value": None, "comparison_key": None,
        "scope": None, "n_ranked": len(ranked), "n_unranked": len(unranked),
        "reason": "Awaiting comparable evaluation and causal review.",
    }
    if ranked:
        cohort = ranked[0].get("breadth_cohort") or {}
        scope = cohort.get("scope", "observed_scores")
        records = {record.get("name"): record for fn in functions
                   for record in [fn if isinstance(fn, Mapping) else vars(fn)]}
        cohorts = [_evaluation_cohort(records[row["name"]].get("scores") or {})
                   for row in ranked]
        comparison_key = None
        candidate_cohorts = {}
        if all(value is not None for value in cohorts):
            # Adding another candidate on the same evidence remains comparable.
            # Any changed association sample resets even historical objectives,
            # and even when a task-breadth panel stays unchanged.
            candidate_cohorts = {
                row["name"]: hashlib.sha256(json.dumps(
                    value, separators=(",", ":"),
                ).encode()).hexdigest()
                for row, value in zip(ranked, cohorts, strict=True)
            }
            identities = sorted(set(candidate_cohorts.values()))
            identity = [_VERSION, run_id, objective, scope,
                        cohort.get("cohort_fingerprint"), identities]
            comparison_key = hashlib.sha256(json.dumps(
                identity, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
        # A useful alternative must not lower the chart merely by joining the
        # frontier. Use the best CURRENT score, not a historical high-water mark:
        # a weakened or removed leading detector must still show a decline.
        search.update(
            status="available", scope=scope, reason=None,
            value=max(row["score"] for row in ranked),
            aggregation="best_current_search_score",
            comparison_key=comparison_key,
            evaluation_cohorts=candidate_cohorts,
            comparison_reason=(None if comparison_key else
                               "Evaluation cohort unknown; trend unavailable."),
        )
    elif any(row.get("status") == "incomparable_cohorts" for row in unranked):
        search.update(status="incomparable", reason="Candidates use different scoring panels.")
    validated = library_frontier_metrics(functions, config=config)
    n_validated = validated["n_rank_eligible"]
    return {
        "version": _VERSION, "n_functions": len(functions), "objective": objective,
        "search": search,
        "validated": {"status": "available" if n_validated else "pending",
                      "value": validated["frontier_value"] if n_validated else None,
                      "n_ranked": n_validated},
        "updated_at": time(),
    }


def display_progress_stats(
    history: Sequence[Mapping[str, Any]], *, current: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use only a trailing comparable display series; never infer missing history.

    Current live state replaces the latest evaluation's display slot, so judgment,
    mutation or removal changes the display without adding a fictitious evaluation.
    Legacy evaluations remain gaps, and completeness/progress fields stay untouched.
    """
    snapshots = [row.get("display_progress") for row in history]
    if current is not None:
        if snapshots:
            snapshots[-1] = current
        else:
            snapshots = [current]
    latest = snapshots[-1] if snapshots else None
    search = latest.get("search", {}) if isinstance(latest, Mapping) else {}
    key = search.get("comparison_key")
    newer_cohorts = search.get("evaluation_cohorts") or {}
    values = []
    if key and search.get("status") == "available":
        for snapshot in reversed(snapshots):
            if not isinstance(snapshot, Mapping) or snapshot.get("version") != _VERSION:
                break
            point = snapshot.get("search", {})
            value = point.get("value")
            cohorts = point.get("evaluation_cohorts") or {}
            if (point.get("status") != "available" or point.get("comparison_key") != key
                    or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or any(cohorts[name] != newer_cohorts[name]
                           for name in cohorts.keys() & newer_cohorts.keys())):
                break
            values.append(float(value))
            # A candidate may temporarily be unranked while its causal review is
            # pending. Keep its identity across that gap, without mutating history.
            newer_cohorts = {**newer_cohorts, **cohorts}
    values.reverse()
    return {
        "current": latest, "search_values": values,
        "search_delta": values[-1] - values[-2] if len(values) > 1 else None,
        "best_search_value": max(values) if values else None,
        "n_evaluations": len(history),
    }


def runtime_display_progress_snapshot(rt: Any) -> dict[str, Any]:
    """Read live scores and the cached corpus inventory; never scan from the UI."""
    from autofa.metrics.library_quality_coverage import library_quality_coverage_for_runtime

    return {
        **display_progress_snapshot(
            rt.library.functions, rt.config, run_id=getattr(rt.run, "run_id", None),
        ),
        "library_quality_coverage": library_quality_coverage_for_runtime(rt, cached_only=True),
    }


def refresh_display_progress(rt: Any) -> None:
    """Refresh display telemetry after review, without adding progress attempts."""
    # Selection and mutation must serialize with evaluation appends and the
    # JSON writer, not only with other chart refreshes.
    with cancellation_guard(), rt._scores_history_lock:
        if not rt.scores_history:
            return
        evaluation = rt.scores_history[-1]
        snapshot = runtime_display_progress_snapshot(rt)
        evaluation["display_progress"] = snapshot
        try:
            rt.persist_scores_history()
        except OSError:
            # The detector's result is already durable. A chart write failure must
            # not turn a successful review into a retry that consumes more work.
            _LOG.warning("Could not persist updated display progress", exc_info=True)
