# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stratified subsampling and interval estimation for sampled detector scoring.

LLM detectors cost one HTTP call per item, so scoring them on the full eval slice
is impractical. They are instead scored on a stratified subsample and reported as
estimates with confidence intervals. Everything here exists to keep those estimates
honest: correct weights so the point estimate is unbiased, and effective sample
sizes so the interval does not pretend the subsample is the whole corpus.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from autofa.metrics.association import wilson_interval

__all__ = [
    "wilson_interval",
    "effective_n",
    "stratum_key",
    "StratifiedSample",
    "StratifiedReservoir",
    "interval_fields",
    "SCORING_SAMPLE_VERSION",
    "ROUND_SAMPLE_VERSION",
    "proportional_sample",
]


SCORING_SAMPLE_VERSION = "outcome_balanced_v2"
ROUND_SAMPLE_VERSION = "proportional_round_v1"


def effective_n(weights: Iterable[float]) -> float:
    """Kish effective sample size ``(Σw)² / Σw²``.

    Unequal stratum weights carry less information than the raw count suggests;
    intervals are computed against this rather than the number of rows scored.
    """
    total = 0.0
    total_sq = 0.0
    for w in weights:
        w = float(w)
        total += w
        total_sq += w * w
    if total_sq <= 0:
        return 0.0
    return (total * total) / total_sq


def stratum_key(item: dict[str, Any], *, is_fail: bool) -> str:
    """Stratify on benchmark × outcome — the two axes that move recall/precision."""
    bench = str(item.get("benchmark") or "unknown")
    return f"{bench}|{'fail' if is_fail else 'pass'}"


@dataclass
class _Stratum:
    key: str
    population: int = 0
    reservoir: list[str] = field(default_factory=list)


@dataclass
class StratifiedSample:
    """Result of an allocation: which ids to score and what each one represents."""

    item_ids: list[str]
    weights: dict[str, float]
    strata: dict[str, dict[str, float]]
    frame_n: int
    sample_n: int
    population: dict[str, float] = field(default_factory=dict)
    stratum_of: dict[str, str] = field(default_factory=dict)

    @property
    def unrepresented_strata(self) -> list[str]:
        """Populated strata with no selected/read rows cannot yield population estimates."""
        return sorted(key for key, meta in self.strata.items()
                      if meta["population"] > 0 and meta["sampled"] == 0)

    def weight_for(self, item_id: str) -> float:
        return float(self.weights.get(item_id, 1.0))

    def is_census(self) -> bool:
        """True when every frame row was selected — estimates are then exact."""
        return self.sample_n >= self.frame_n and all(
            abs(w - 1.0) < 1e-9 for w in self.weights.values()
        )

    def reweight(self, scored_ids: Iterable[str]) -> None:
        """Recompute weights from the rows that were *actually* scored.

        Allocation weights assume every selected id gets scored, but a row can go
        missing between allocation and scoring (dataset filters applied on read, an
        id absent from the index). Rescaling against the realized per-stratum count
        keeps the estimator unbiased instead of silently under-representing a
        stratum by the rows it lost.
        """
        realized: dict[str, int] = {}
        kept: list[str] = []
        for iid in scored_ids:
            key = self.stratum_of.get(iid)
            if key is None:
                continue
            realized[key] = realized.get(key, 0) + 1
            kept.append(iid)
        self.item_ids = sorted(set(kept))
        self.sample_n = len(kept)
        new_weights: dict[str, float] = {}
        for key, n_s in realized.items():
            pop = float(self.population.get(key, 0.0))
            w = (pop / n_s) if n_s else 0.0
            meta = self.strata.setdefault(key, {"population": pop})
            meta["sampled"] = float(n_s)
            meta["weight"] = w
            for iid in kept:
                if self.stratum_of.get(iid) == key:
                    new_weights[iid] = w
        for key in list(self.strata):
            if key not in realized:
                self.strata[key]["sampled"] = 0.0
                self.strata[key]["weight"] = 0.0
        self.weights = new_weights


class StratifiedReservoir:
    """Single-pass stratified reservoir over a stream of items.

    Holds only item ids, so the memory cost is independent of trace size. Exact
    per-stratum population counts are tracked alongside, which is what makes the
    post-hoc weights correct without a second counting pass.
    """

    def __init__(self, *, per_stratum_cap: int = 200, seed: int = 0) -> None:
        self.per_stratum_cap = max(1, int(per_stratum_cap))
        self._rng = random.Random(int(seed) & 0xFFFFFFFF)
        self._strata: dict[str, _Stratum] = {}
        self.frame_n = 0

    def observe(self, item_id: str, *, key: str) -> None:
        if not item_id:
            return
        self.frame_n += 1
        st = self._strata.get(key)
        if st is None:
            st = _Stratum(key=key)
            self._strata[key] = st
        st.population += 1
        if len(st.reservoir) < self.per_stratum_cap:
            st.reservoir.append(item_id)
        else:
            j = self._rng.randrange(st.population)
            if j < self.per_stratum_cap:
                st.reservoir[j] = item_id
        # A repeated item_id would be scored once but counted twice in the
        # allocation denominator, inflating that stratum's weight.
        if len(st.reservoir) != len(set(st.reservoir)):
            st.reservoir = list(dict.fromkeys(st.reservoir))

    def allocate(self, *, target_n: int, min_per_stratum: int = 5) -> StratifiedSample:
        """Balance failed/passing outcomes, then preserve benchmark strata within each.

        An odd budget gives failures the extra row. A scarce outcome spills its
        unused allocation into the other outcome; weights remain population / n
        within each benchmark × outcome stratum. If even one row per stratum
        cannot fit, the hard budget wins and ``unrepresented_strata`` records the
        missing population evidence rather than certifying complete estimates.
        """
        target_n = max(1, int(target_n))
        min_per_stratum = max(1, int(min_per_stratum))
        strata = sorted((s for s in self._strata.values() if s.population > 0),
                        key=lambda st: st.key)
        if not strata or self.frame_n <= 0:
            return StratifiedSample([], {}, {}, self.frame_n, 0)

        populations = {st.key: float(st.population) for st in strata}
        available = {st.key: min(st.population, len(st.reservoir)) for st in strata}
        budget = min(target_n, sum(available.values()))
        outcomes: dict[str, list[_Stratum]] = {}
        for st in strata:
            outcome = st.key.rsplit("|", 1)[-1]
            outcomes.setdefault(outcome, []).append(st)
        if set(outcomes) <= {"fail", "pass"} and len(outcomes) == 2:
            capacity = {outcome: sum(available[st.key] for st in group)
                        for outcome, group in outcomes.items()}
            failed_n = min(capacity["fail"], (budget + 1) // 2)
            if budget >= len(strata):
                failed_n = max(len(outcomes["fail"]),
                               min(failed_n, budget - len(outcomes["pass"])))
            passed_n = min(capacity["pass"], budget - failed_n)
            failed_n += min(capacity["fail"] - failed_n, budget - failed_n - passed_n)
            outcome_budgets = {"fail": failed_n, "pass": passed_n}
        else:
            # A one-outcome corpus needs no artificial empty half. Unknown custom
            # keys retain ordinary proportional allocation as one group.
            outcomes = {"all": strata}
            outcome_budgets = {"all": budget}

        alloc: dict[str, int] = {}
        for outcome, group in outcomes.items():
            group_budget = outcome_budgets[outcome]
            floor = min(min_per_stratum, group_budget // len(group))
            total_population = sum(st.population for st in group)
            alloc.update({st.key: min(available[st.key], floor) for st in group})
            # Largest deficit from proportional quotas, with fixed rare-stratum
            # floors and capacity bounds. Filling one slot at a time is bounded
            # by the configured sample size (normally 200), not corpus size.
            queue = [(alloc[st.key] - group_budget * st.population / total_population,
                      st.key) for st in group if alloc[st.key] < available[st.key]]
            heapq.heapify(queue)
            remaining = group_budget - sum(alloc[st.key] for st in group)
            while remaining and queue:
                deficit, key = heapq.heappop(queue)
                alloc[key] += 1
                remaining -= 1
                if alloc[key] < available[key]:
                    heapq.heappush(queue, (deficit + 1, key))

        ids = []
        weights: dict[str, float] = {}
        strata_meta: dict[str, dict[str, float]] = {}
        stratum_of = {}
        for st in strata:
            n_s = alloc.get(st.key, 0)
            if n_s <= 0:
                strata_meta[st.key] = {
                    "population": float(st.population),
                    "sampled": 0.0,
                    "weight": 0.0,
                }
                continue
            pool = st.reservoir
            picked = pool if n_s >= len(pool) else self._rng.sample(pool, n_s)
            # Each sampled row stands in for population/sampled rows of its stratum.
            w = st.population / float(len(picked))
            for iid in picked:
                weights[iid] = w
                stratum_of[iid] = st.key
            ids.extend(picked)
            strata_meta[st.key] = {
                "population": float(st.population),
                "sampled": float(len(picked)),
                "weight": float(w),
            }

        return StratifiedSample(
            item_ids=sorted(set(ids)),
            weights=weights,
            strata=strata_meta,
            frame_n=self.frame_n,
            sample_n=len(ids),
            population=populations,
            stratum_of=stratum_of,
        )


def proportional_sample(
    stratum_by_item_id: dict[str, str], *, fraction: float, seed: int, minimum_n: int = 0,
) -> StratifiedSample:
    """Select a stable fraction of every benchmark/outcome stratum.

    This validation cohort is independent of the small discovery panel. Retain
    IDs only; callers reuse any overlapping prediction checkpoints. Rounding up
    per stratum represents rare benchmarks without balancing outcomes to 50/50.
    ``minimum_n`` raises the shared fraction before allocation, so a checkpoint
    never uses fewer examples than discovery already scored.
    """
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("Validation fraction must be greater than 0 and at most 1")
    if type(minimum_n) is not int or minimum_n < 0:
        raise ValueError("Validation minimum_n must be a nonnegative integer")
    if stratum_by_item_id:
        fraction = max(fraction, min(minimum_n / len(stratum_by_item_id), 1.0))
    groups: dict[str, list[str]] = {}
    for item_id, key in stratum_by_item_id.items():
        if not isinstance(item_id, str) or not item_id or not isinstance(key, str) or not key:
            raise ValueError("Validation sampling requires nonempty item and stratum IDs")
        groups.setdefault(key, []).append(item_id)
    ids, weights, strata, populations, selected_strata = [], {}, {}, {}, {}
    for key, group in sorted(groups.items()):
        population = len(group)
        count = math.ceil(fraction * population)
        # A separate stream for each stratum avoids order dependence between
        # benchmarks or correlations with discovery's reservoir random draws.
        rng = random.Random(hashlib.sha256(
            f"{ROUND_SAMPLE_VERSION}:{int(seed)}:{key}".encode()).digest())
        picked = rng.sample(sorted(group), count)
        weight = population / count
        ids.extend(picked)
        weights.update({item_id: weight for item_id in picked})
        selected_strata.update({item_id: key for item_id in picked})
        populations[key] = float(population)
        strata[key] = {"population": float(population), "sampled": float(count),
                       "weight": weight}
    return StratifiedSample(sorted(ids), weights, strata, len(stratum_by_item_id), len(ids),
                            populations, selected_strata)


def interval_fields(
    *,
    recall: float,
    raw_ppv: float,
    n_eff_fail: float,
    n_eff_flagged: float,
    fpr: float | None = None,
    n_eff_pass: float | None = None,
) -> dict[str, float]:
    """Wilson intervals for the proportions the loop actually optimizes."""
    r_lo, r_hi = wilson_interval(recall * n_eff_fail, n_eff_fail)
    p_lo, p_hi = wilson_interval(raw_ppv * n_eff_flagged, n_eff_flagged)
    out = {
        "recall_ci_low": float(r_lo),
        "recall_ci_high": float(r_hi),
        "raw_ppv_ci_low": float(p_lo),
        "raw_ppv_ci_high": float(p_hi),
    }
    if fpr is not None and n_eff_pass is not None:
        f_lo, f_hi = wilson_interval(fpr * n_eff_pass, n_eff_pass)
        out["fpr_ci_low"] = float(f_lo)
        out["fpr_ci_high"] = float(f_hi)
    return out
