# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Association metrics: recall, ppv@π₀, contingency tables."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

REFERENCE_PI0 = 0.30
Z_95 = 1.959963984540054


def wilson_interval(successes: float, n: float, *, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    successes = min(max(0.0, float(successes)), float(n))
    p = successes / n
    # Exact extremes: the Wilson formula can leave a ~1e-16 overshoot/undershoot.
    if p <= 0.0:
        z2 = z * z
        hi = z2 / (n + z2)
        return (0.0, float(min(1.0, hi)))
    if p >= 1.0:
        z2 = z * z
        lo = n / (n + z2)
        return (float(max(0.0, lo)), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_pvalue(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p-value for a 2×2 table (pure Python).

    Association test for binary detectors. Prefer this over asking whether a
    Wald odds-ratio CI excludes 1 — that CI is unreliable with a zero cell.
    """
    a, b, c, d = (max(0, int(a)), max(0, int(b)), max(0, int(c)), max(0, int(d)))
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    col1 = a + c

    def log_p(k: int) -> float:
        # P(X=k) under hypergeometric given margins.
        return (
            _log_comb(col1, k)
            + _log_comb(n - col1, row1 - k)
            - _log_comb(n, row1)
        )

    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)
    if lo > hi:
        return 1.0

    log_obs = log_p(a)
    # Two-sided: sum mass of tables at least as extreme as observed.
    total = 0.0
    for k in range(lo, hi + 1):
        lp = log_p(k)
        if lp <= log_obs + 1e-12:
            total += math.exp(lp)
    return float(min(1.0, max(0.0, total)))


def _nearly_integral(*vals: float, eps: float = 1e-6) -> bool:
    return all(abs(float(v) - round(float(v))) <= eps for v in vals)


def contingency(a: float, b: float, c: float, d: float) -> dict[str, Any]:
    """2×2 cells: a=flagged fail, b=flagged pass, c=unflagged fail, d=unflagged pass.

    Primary association report for detectors is recall / FPR with Wilson CIs and
    Fisher's exact p — not a Wald odds-ratio interval. OR is kept as a secondary
    point estimate (Haldane-Anscombe / Firth when a cell is empty); its Wald CI
    is omitted under separation because it is known to be misleading.
    """
    a_f, b_f, c_f, d_f = float(a), float(b), float(c), float(d)
    n = a_f + b_f + c_f + d_f
    n_fail = a_f + c_f
    n_pass = b_f + d_f
    n_flagged = a_f + b_f
    sens = a_f / n_fail if n_fail else 0.0
    spec = d_f / n_pass if n_pass else 0.0
    fpr = b_f / n_pass if n_pass else 0.0
    denom = sens * REFERENCE_PI0 + (1.0 - spec) * (1.0 - REFERENCE_PI0)
    ppv_at_pi0 = (sens * REFERENCE_PI0 / denom) if denom > 0 else 0.0
    raw_ppv = a_f / n_flagged if n_flagged else 0.0

    # Haldane-Anscombe / Firth point estimate for OR (principled under separation).
    zero_cell = min(a_f, b_f, c_f, d_f) <= 0.0
    if zero_cell:
        aa, bb, cc, dd = a_f + 0.5, b_f + 0.5, c_f + 0.5, d_f + 0.5
    else:
        aa, bb, cc, dd = a_f, b_f, c_f, d_f
    log_or = math.log((aa * dd) / (bb * cc))
    odds_ratio = float(math.exp(log_or))

    # Wald OR CI only when every cell is positive — under a zero cell the
    # corrected-table SE is dominated by the fabricated ½ and can straddle 1
    # while Fisher rejects hard.
    if zero_cell:
        or_ci_low: float | None = None
        or_ci_high: float | None = None
    else:
        se = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
        or_ci_low = float(math.exp(log_or - 1.96 * se))
        or_ci_high = float(math.exp(log_or + 1.96 * se))

    rec_lo, rec_hi = wilson_interval(a_f, n_fail)
    fpr_lo, fpr_hi = wilson_interval(b_f, n_pass)

    fisher_p: float | None = None
    if _nearly_integral(a_f, b_f, c_f, d_f):
        fisher_p = fisher_exact_pvalue(
            int(round(a_f)), int(round(b_f)), int(round(c_f)), int(round(d_f))
        )

    return {
        "a": a_f,
        "b": b_f,
        "c": c_f,
        "d": d_f,
        "n_items": float(n),
        "n_fail_caught": float(a_f),
        "n_flagged": float(n_flagged),
        # Share of ALL examples the function fires on (not just failures).
        "pct_caught": float(n_flagged / n) if n else 0.0,
        "recall": float(sens),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "fpr": float(fpr),
        # Standardized precision at π₀=0.30 — NOT raw corpus PPV.
        "ppv_at_pi0": float(ppv_at_pi0),
        "raw_ppv": float(raw_ppv),
        "youden_j": float(sens + spec - 1.0),
        "odds_ratio": odds_ratio,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
        "recall_ci_low": float(rec_lo),
        "recall_ci_high": float(rec_hi),
        "fpr_ci_low": float(fpr_lo),
        "fpr_ci_high": float(fpr_hi),
        "fisher_p": fisher_p,
        "f1_at_pi0": f1_at_pi0(sens, ppv_at_pi0),
    }


def f1_at_pi0(recall: float | None, ppv: float | None) -> float:
    r = float(recall or 0.0)
    p = float(ppv or 0.0)
    if r + p <= 0:
        return 0.0
    return 2.0 * r * p / (r + p)


_NEGATIVE_BUCKET_LABELS = frozenset(
    {"", "0", "false", "no", "none", "null", "pass", "ok", "neg", "negative"}
)


def is_negative_bucket_label(value: Any) -> bool:
    """True for non-fire / negative polarity labels (False, pass, ok, …)."""
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return not bool(value)
    return str(value).strip().lower() in _NEGATIVE_BUCKET_LABELS


def truthy_bucket(value: Any) -> bool:
    """Map detector output to 'in fail-bucket' boolean (fires = True)."""
    return not is_negative_bucket_label(value)


def selected_bucket(scores: Mapping[str, Any]) -> tuple[Any, str]:
    """Resolve the selected operating point without treating its name as a wildcard.

    New scores distinguish an exact output value from the synthetic truthy
    aggregate. Historical bucket tables also establish exact matching. An old
    synthetic ``positive`` without that provenance only supports boolean fires,
    never an arbitrary category whose name happens to be truthy.
    """
    bucket = scores.get("bucket")
    rows = [row for row in scores.get("buckets") or [] if isinstance(row, Mapping)]
    if bucket is not None and is_negative_bucket_label(bucket):
        # Older primary selection could incorrectly pick the non-fire side.
        bucket = next((row.get("bucket") for row in rows
                       if not is_negative_bucket_label(row.get("bucket"))), None)
        return (bucket, "exact") if bucket is not None else (None, "legacy_boolean")
    mode = scores.get("bucket_mode")
    if mode is not None:
        return bucket, mode
    if bucket not in (None, "positive") or any(row.get("bucket") == bucket for row in rows):
        return bucket, "exact"
    return bucket, "legacy_boolean"


def matches_selected_bucket(value: Any, selection: tuple[Any, str]) -> bool:
    """Match detector/localisation values to a resolved positive operating point."""
    bucket, mode = selection
    if not truthy_bucket(value):
        return False
    if mode == "exact":
        return bucket is not None and str(value) == str(bucket)
    if mode == "truthy":
        return True
    if mode == "legacy_boolean":
        return value is True or (isinstance(value, str) and value == "True")
    return False


_ASSOC_ROW_KEYS = (
    "recall",
    "sensitivity",
    "specificity",
    "fpr",
    "ppv_at_pi0",
    "f1_at_pi0",
    "odds_ratio",
    "or_ci_low",
    "or_ci_high",
    "youden_j",
    "a",
    "b",
    "c",
    "d",
    "pct_caught",
    "n_items",
    "n_fail_caught",
    "n_flagged",
    "raw_ppv",
    "recall_ci_low",
    "recall_ci_high",
    "fpr_ci_low",
    "fpr_ci_high",
    "fisher_p",
)


def _assoc_fields(scores: dict[str, Any]) -> dict[str, Any]:
    return {k: scores.get(k) for k in _ASSOC_ROW_KEYS}


def pick_primary_bucket(buckets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the fire / detection operating point for headline metrics + judge.

    ``buckets`` is sorted by f1_at_pi0 desc. Prefer a non-negative (fire-side) label
    with fail enrichment (a >= b). Never promote False/ok/pass just because the
    majority non-fire side has higher recall on a high-fail corpus — that would
    make precision/recall/ppv and behavior_purity describe the wrong items.
    """
    if not buckets:
        return None
    fire_side = [r for r in buckets if not is_negative_bucket_label(r.get("bucket"))]
    for row in fire_side:
        if float(row.get("a") or 0) >= float(row.get("b") or 0):
            return row
    # No fail-enriched row: still prefer best fire-side f1 over a negative label.
    return fire_side[0] if fire_side else None


class StreamingBucketAccumulator:
    """Accumulate per-bucket fail/pass counts while streaming the corpus.

    Counters stay O(#buckets). Example ids are kept for dashboard inspection
    (fails always; passes up to ``max_pass_examples_per_bucket``).

    Supports weighted updates for stratified subsample scoring: pass ``weight``
    to have a row stand in for that many corpus rows. When any weight differs
    from 1 the finalized rows are estimates and carry ``sampled=True`` plus
    Wilson confidence intervals computed against the effective sample size.
    """

    def __init__(
        self,
        *,
        max_examples_per_bucket: int | None = None,
        max_pass_examples_per_bucket: int | None = None,
    ) -> None:
        self._counts: dict[str, list[float]] = {}  # bucket -> [w_fail, w_pass]
        self.n_fail_total = 0.0
        self.n_pass_total = 0.0
        self.errors = 0
        self.error_examples: list[dict[str, str]] = []
        self.error_fail = 0.0
        self.error_pass = 0.0
        self.n_abstained = 0
        self.abstained_fail = 0.0
        self.abstained_pass = 0.0
        self.n_items = 0  # rows actually scored (not the weighted estimate)
        self._examples: dict[str, list[str]] = {}
        self._pass_examples: dict[str, list[str]] = {}
        # None = keep every fail id (dashboard wants full bucket membership).
        self._max_ex = max_examples_per_bucket
        self._max_pass_ex = max_pass_examples_per_bucket
        # Per-bucket weight lists drive the Kish effective n for intervals.
        self._weights: dict[str, list[float]] = {}
        self._fail_weights: list[float] = []
        self._pass_weights: list[float] = []
        self._weighted = False

    def update(
        self,
        pred: Any,
        *,
        is_correct: bool,
        item_id: str | None = None,
        weight: float = 1.0,
    ) -> None:
        w = float(weight)
        if w <= 0:
            return
        if abs(w - 1.0) > 1e-9:
            self._weighted = True
        self.n_items += 1
        fail = not is_correct
        if fail:
            self.n_fail_total += w
            self._fail_weights.append(w)
        else:
            self.n_pass_total += w
            self._pass_weights.append(w)
        key = str(pred)
        cell = self._counts.setdefault(key, [0.0, 0.0])
        self._weights.setdefault(key, []).append(w)
        if fail:
            cell[0] += w
            if item_id:
                ex = self._examples.setdefault(key, [])
                if self._max_ex is None or len(ex) < self._max_ex:
                    ex.append(str(item_id))
        else:
            cell[1] += w
            if item_id:
                ex = self._pass_examples.setdefault(key, [])
                if self._max_pass_ex is None or len(ex) < self._max_pass_ex:
                    ex.append(str(item_id))

    def note_error(
        self, *, is_correct: bool, weight: float = 1.0,
        error: str | None = None, item_id: str | None = None,
    ) -> None:
        """Record an unresolved execution, never a negative detector prediction."""
        self.errors += 1
        if is_correct:
            self.error_pass += weight
        else:
            self.error_fail += weight
        if error and len(self.error_examples) < 3:
            self.error_examples.append({"item_id": item_id or "", "error": error[:300]})

    def note_abstention(self, *, is_correct: bool, weight: float = 1.0) -> None:
        self.n_abstained += 1
        if is_correct:
            self.abstained_pass += weight
        else:
            self.abstained_fail += weight

    def abstention_bounds(self, scores: dict) -> dict:
        return self._missing_bounds(scores, self.abstained_fail, self.abstained_pass)

    def unresolved_bounds(self, scores: dict) -> dict:
        """Conservative rates across resolved predictions, abstentions and errors."""
        return self._missing_bounds(
            scores, self.abstained_fail + self.error_fail,
            self.abstained_pass + self.error_pass,
        )

    def _missing_bounds(self, scores: dict, missing_fail: float, missing_pass: float) -> dict:
        bounds = {}
        for label, positive, known, missing in (
            ("recall", scores.get("a", 0), self.n_fail_total, missing_fail),
            ("fpr", scores.get("b", 0), self.n_pass_total, missing_pass),
        ):
            total = known + missing
            bounds[label] = [positive / total, (positive + missing) / total] if total else None
        return bounds

    def finalize(self, *, top_n: int = 20, sampled: bool | None = None) -> list[dict[str, Any]]:
        from autofa.metrics.sampling import effective_n, interval_fields

        is_sampled = self._weighted if sampled is None else bool(sampled)
        rows: list[dict[str, Any]] = []
        for bucket, (a, b) in self._counts.items():
            c = self.n_fail_total - a
            d = self.n_pass_total - b
            scores = contingency(a, b, c, d)
            n = a + b
            purity = (a / n) if n else 0.0
            fail_ids = list(self._examples.get(bucket, []))
            if self._max_ex is not None:
                fail_ids = fail_ids[: self._max_ex]
            pass_ids = list(self._pass_examples.get(bucket, []))
            if self._max_pass_ex is not None:
                pass_ids = pass_ids[: self._max_pass_ex]
            row = {
                "bucket": bucket,
                "bucket_mode": "exact",
                "n": n,
                "n_fail": a,
                "n_pass": b,
                "purity": float(purity),
                "precision": float(purity),
                "is_high_purity": purity >= 0.70 and n >= 3,
                **_assoc_fields(scores),
                "example_fail_ids": fail_ids,
                "example_pass_ids": pass_ids,
            }
            if is_sampled:
                n_eff_fail = effective_n(self._fail_weights)
                n_eff_pass = effective_n(self._pass_weights)
                n_eff_flagged = effective_n(self._weights.get(bucket, []))
                row["sampled"] = True
                # Raw rows landing in this bucket; ``n`` above is the weighted
                # population estimate, which is a different number when sampled.
                row["n_rows_scored"] = len(self._weights.get(bucket, []))
                row["n_eff_fail"] = float(n_eff_fail)
                row["n_eff_pass"] = float(n_eff_pass)
                row["n_eff_flagged"] = float(n_eff_flagged)
                # Fisher exact does not apply to weighted population estimates.
                row["fisher_p"] = None
                row.update(
                    interval_fields(
                        recall=float(scores["recall"]),
                        raw_ppv=float(scores["raw_ppv"]),
                        n_eff_fail=n_eff_fail,
                        n_eff_flagged=n_eff_flagged,
                        fpr=float(scores["fpr"]),
                        n_eff_pass=n_eff_pass,
                    )
                )
            rows.append(row)
        rows.sort(key=lambda r: (-float(r["f1_at_pi0"]), -float(r["purity"]), -float(r["n_fail"])))
        return rows[:top_n]


def pct_caught_from_scores(scores: dict[str, Any] | None) -> float:
    """Share of all examples the function fires on.

    Prefers stored ``pct_caught``, then the judge feedback's rate (sampled
    detectors withhold the top-level rate); otherwise derives it from
    contingency cells (``(a+b)/n_items``).
    """
    if not scores:
        return 0.0
    if scores.get("pct_caught") is not None:
        return float(scores["pct_caught"])
    fb = scores.get("feedback")
    if isinstance(fb, dict) and fb.get("pct_caught") is not None:
        return float(fb["pct_caught"])
    a = float(scores.get("a") or 0)
    b = float(scores.get("b") or 0)
    n = float(scores.get("n_items") or 0)
    if n <= 0:
        n = a + b + float(scores.get("c") or 0) + float(scores.get("d") or 0)
    if n <= 0:
        flagged = scores.get("n_flagged")
        if flagged is not None and scores.get("n_items"):
            return float(flagged) / float(scores["n_items"])
        return 0.0
    return (a + b) / n
