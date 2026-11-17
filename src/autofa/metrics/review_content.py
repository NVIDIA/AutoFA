# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recheck retained judge evidence against the current source during evaluation."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from autofa.core.dependencies import review_source
from autofa.metrics.judge_evidence import make_snapshot


def _review_rows(scores: dict) -> list[dict]:
    return list(scores.get("behavior_verdicts") or []) + list(
        (scores.get("diagnostic_review") or {}).get("verdicts") or [])


def saved_review_localisations(fn: Any) -> dict[str, Any]:
    """Freeze original review locators before later detector scans replace them."""
    scores = fn.scores or {}
    retained = {
        **((scores.get("diagnostic_review") or {}).get("review_localisations") or {}),
        **(scores.get("behavior_review_localisations") or {}),
    }
    return {
        row["item_id"]: deepcopy(retained[row["item_id"]] if row["item_id"] in retained
                                 else fn.failure_localisations.get(row["item_id"]))
        for row in _review_rows(scores)
        if isinstance(row, dict) and isinstance(row.get("item_id"), str)
    }


class RetainedReviewContent:
    """Hash only previously reviewed rows; never retain streamed rollout bodies."""

    def __init__(self, functions: list[Any]):
        self.functions = {fn.name: fn for fn in functions}
        self.by_item: dict[str, set[str]] = defaultdict(set)
        self.snapshots: dict[str, dict[str, str | None]] = defaultdict(dict)
        self.diagnostic_snapshots: dict[str, dict[str, str | None]] = defaultdict(dict)
        self.localisations: dict[str, dict[str, Any]] = defaultdict(dict)
        self.diagnostic_localisations: dict[str, dict[str, Any]] = defaultdict(dict)
        for fn in functions:
            scores = fn.scores or {}
            self.localisations[fn.name] = saved_review_localisations(fn)
            diagnostic = scores.get("diagnostic_review") or {}
            saved = diagnostic.get("review_localisations") or {}
            self.diagnostic_localisations[fn.name] = {
                row["item_id"]: deepcopy(saved.get(row["item_id"],
                                                  fn.failure_localisations.get(row["item_id"])))
                for row in diagnostic.get("verdicts") or []
                if isinstance(row, dict) and isinstance(row.get("item_id"), str)
            }
            for row in _review_rows(scores):
                if isinstance(row, dict) and isinstance(row.get("item_id"), str):
                    if row.get("verdict") not in {"yes", "no"} and not row.get("snapshot_id"):
                        continue
                    self.by_item[row["item_id"]].add(fn.name)

    def observe(self, item: dict[str, Any]) -> None:
        item_id = str(item.get("_item_id") or "")
        for name in self.by_item.get(item_id, ()):
            fn = self.functions[name]
            scores = fn.scores or {}
            for snapshots, localisations, contract in (
                (self.snapshots, self.localisations, scores.get("claim_contract")),
                (self.diagnostic_snapshots, self.diagnostic_localisations,
                 (scores.get("diagnostic_review") or {}).get("claim_contract")),
            ):
                if (snapshots is self.diagnostic_snapshots
                        and item_id not in self.diagnostic_localisations[name]):
                    continue
                try:
                    snapshot = make_snapshot(
                        item, item_id=item_id, source=review_source(fn),
                        failure_localisation=localisations[name].get(item_id),
                        claim_contract=contract,
                    )["snapshot_id"]
                except (KeyError, TypeError, ValueError, OSError):
                    snapshot = None
                previous = snapshots[name].get(item_id, snapshot)
                snapshots[name][item_id] = snapshot if previous == snapshot else None

    def missing_item_ids(self) -> set[str]:
        return {item_id for item_id, names in self.by_item.items()
                if any(item_id not in self.snapshots[name] for name in names)}

    def apply(self, name: str, scores: dict[str, Any]) -> dict[str, Any]:
        result = scores
        rows = scores.get("behavior_verdicts")
        if isinstance(rows, list) and rows:
            rows, stale = self._validate_rows(name, rows)
            result = {**result, "behavior_verdicts": rows, "behavior_evidence_stale": stale,
                      "behavior_review_localisations": self.localisations[name]}
        diagnostic = scores.get("diagnostic_review")
        if isinstance(diagnostic, dict) and isinstance(diagnostic.get("verdicts"), list):
            rows, stale = self._validate_rows(name, diagnostic["verdicts"], diagnostic=True)
            result = {
                **result, "diagnostic_evidence_stale": stale,
                "diagnostic_review": {
                    **diagnostic, "verdicts": rows, "evidence_stale": stale,
                    "review_localisations": {
                        row["item_id"]: self.diagnostic_localisations[name].get(row["item_id"])
                        for row in rows if isinstance(row, dict) and "item_id" in row
                    },
                },
            }
        return result

    def _validate_rows(self, name: str, rows: list, *, diagnostic=False) -> tuple[list, bool]:
        rows = deepcopy(rows)
        stale = False
        snapshots = self.diagnostic_snapshots if diagnostic else self.snapshots
        for row in rows:
            if not isinstance(row, dict):
                stale = True
                continue
            if row.get("verdict") not in {"yes", "no"} and not row.get("snapshot_id"):
                continue
            historical = row.get("historical_review") or row
            item_id = row.get("item_id")
            current = snapshots[name].get(item_id)
            if current and current == historical.get("snapshot_id"):
                original = deepcopy(historical)
                row.clear()
                row.update(original)
                continue
            stale = True
            original = deepcopy(historical)
            row["historical_review"] = original
            row["verification_status"] = "historical"
            row["content_validation"] = {
                "status": "changed" if current else "unavailable",
                "current_snapshot_id": current,
                "reason": "Retained review does not establish the current rollout's behavior.",
            }
        return rows, stale
