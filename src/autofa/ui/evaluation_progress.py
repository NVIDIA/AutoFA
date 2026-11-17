# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional final-scoring progress, separate from the agent's live turn display."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


class EvaluationProgress:
    """Count completed detector/example pairs without retaining traces or results."""

    def __init__(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callback = callback
        self.completed = 0
        self.total: int | None = None
        self.rows: int | None = None
        self.estimated = False
        self.by_detector: Counter[str] = Counter()
        self.target_rows: dict[str, int] = {}

    def start(self, names: list[str], rows: int | None, *, estimated: bool = False) -> None:
        self.completed = 0
        self.rows = rows
        self.total = rows * len(names) if rows is not None else None
        self.estimated = estimated
        self.by_detector.clear()
        self.target_rows.clear()
        self._emit(names[0] if names else "", "scoring")

    def set_targets(self, rows_by_detector: dict[str, int], *, estimated: bool = False) -> None:
        """Refine allocated work without losing predictions completed during the scan."""
        self.target_rows = dict(rows_by_detector)
        self.total = sum(self.target_rows.values())
        self.estimated = estimated
        self._emit(next(reversed(self.by_detector), next(iter(rows_by_detector), "")), "scoring")

    def advance(self, name: str) -> None:
        self.completed += 1
        self.by_detector[name] += 1
        # An upper/lower metadata estimate must never claim completion while
        # unvisited rows or in-flight evaluations still remain.
        if self.estimated and self.total is not None and self.completed >= self.total:
            self.total = self.completed + 1
        self._emit(name, "scoring")

    def finish_scoring(self) -> None:
        self.total = self.completed
        self.rows = max(self.by_detector.values(), default=0)
        self.estimated = False
        self._emit(next(reversed(self.by_detector), ""), "saving")

    def _emit(self, name: str, phase: str) -> None:
        self.callback({"completed": self.completed, "total": self.total,
                       "estimated": self.estimated, "detector": name,
                       "detector_completed": self.by_detector[name] if name else 0,
                       "rows": self.target_rows.get(name, self.rows), "phase": phase})


_CURRENT: ContextVar[EvaluationProgress | None] = ContextVar("evaluation_progress", default=None)


def current_evaluation_progress() -> EvaluationProgress | None:
    return _CURRENT.get()


@contextmanager
def evaluation_progress(callback: Callable[[dict[str, Any]], None]) -> Iterator[EvaluationProgress]:
    reporter = EvaluationProgress(callback)
    token = _CURRENT.set(reporter)
    try:
        yield reporter
    finally:
        _CURRENT.reset(token)
