# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-early checks for whether selected data can produce an efficiency report.

Each check runs just before the step it protects: source checks before a local
import, selection checks before a download, and data checks before analysis. The
data check also previews what the report will contain.
Checks apply only inside :func:`efficiency_gate`, which the caller opens when
efficiency metrics were requested.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from autofa.data.evalrdl_efficiency import (
    EfficiencyReadiness,
    UnsupportedEfficiencySelection,
    verify_efficiency_data,
)

ONE_RUN = "Efficiency analysis supports exactly one evaluation run; this selection has {n}."
NOT_EVALRDL = (
    "Efficiency analysis needs EvalRDL run data; {name} is not an EvalRDL run or a "
    "supported harness log."
)
NO_ATTEMPTS = "The selected run has no rollouts, so efficiency cannot be computed."
FAILURE_ANALYSIS_ONLY = "Turn off efficiency metrics to run failure discovery only."
NO_REWARD = (
    "No rollout has a recorded reward, so the efficiency report cannot compare correct "
    "and incorrect rollouts."
)
ONE_OUTCOME = (
    "Every scored rollout is {outcome}, so the efficiency report has no {missing} "
    "rollouts to compare."
)
NO_TOKENS = (
    "Token usage is not recorded, so the efficiency report will have no token or cost "
    "charts."
)
PARTIAL_TOKENS = (
    "Token usage is recorded for {n} of {total} rollouts; token and cost charts cover "
    "only those."
)
SMALL_GROUPS = (
    "The efficiency summary needs at least {minimum} correct and {minimum} incorrect "
    "rollouts; this run has {n_pass} correct and {n_fail} incorrect."
)

BOTH, WITHOUT, CONTINUE, STOP = "both", "without", "continue", "stop"


@dataclass
class EfficiencyGate:
    """What happens when the selected data cannot produce a full efficiency report.

    When the report is impossible, ``continue_without`` receives the reason and
    returns whether to continue without efficiency; without it, the run stops. When
    the report would have no correct-versus-incorrect comparison, ``choose``
    receives the allowed choices; without it, the run continues. ``notify`` shows
    the data preview and warnings.
    """

    combined: bool = False
    continue_without: Callable[[str], bool] | None = None
    choose: Callable[[tuple[str, ...]], str] | None = None
    notify: Callable[[str, bool], None] | None = None
    enabled: bool = True

    def reject(self, reason: str) -> None:
        if self.continue_without is not None and self.continue_without(reason):
            self.enabled = False
            return
        hint = f" {FAILURE_ANALYSIS_ONLY}" if self.combined else ""
        raise UnsupportedEfficiencySelection(reason + hint)

    def caution(self, reason: str) -> None:
        self.note(reason, warning=True)
        choices = (BOTH, WITHOUT, STOP) if self.combined else (CONTINUE, STOP)
        choice = self.choose(choices) if self.choose is not None else choices[0]
        if choice == WITHOUT:
            self.enabled = False
        elif choice not in {BOTH, CONTINUE}:
            raise UnsupportedEfficiencySelection(reason)

    def note(self, message: str, *, warning: bool = False) -> None:
        if self.notify is not None:
            self.notify(message, warning)


_GATE: ContextVar[EfficiencyGate | None] = ContextVar("autofa_efficiency_gate", default=None)


@contextmanager
def efficiency_gate(gate: EfficiencyGate | None) -> Iterator[EfficiencyGate | None]:
    token = _GATE.set(gate)
    try:
        yield gate
    finally:
        _GATE.reset(token)


def _active() -> EfficiencyGate | None:
    gate = _GATE.get()
    return gate if gate is not None and gate.enabled else None


def check_local_sources(sources: Iterable[Any]) -> None:
    """Before importing local logs: one run, readable as EvalRDL data."""
    gate = _active()
    if gate is None:
        return
    sources = list(sources)
    if len(sources) != 1:
        gate.reject(ONE_RUN.format(n=len(sources)))
    elif not _is_evalrdl_source(sources[0]):
        gate.reject(NOT_EVALRDL.format(name=sources[0].path.name))


def _is_evalrdl_source(source: Any) -> bool:
    from autofa.data.local_import import detect_reader, reference_for_import

    if source.kind == "evalrdl":
        return True
    if source.kind in {"gym_dir", "jsonl"}:
        return detect_reader(source.path) is not None
    return source.path.is_file() and reference_for_import(source.path) is not None


def check_cohort(cohort: Any) -> None:
    """Before downloading a rollout selection: one run with at least one attempt."""
    gate = _active()
    if gate is None:
        return
    sources = [
        source for source in cohort.sources
        if source.get("selection") == "all" or cohort.identities(source["run_id"])
    ]
    runs = {str(source["run_id"]) for source in sources}
    if len(runs) > 1:
        gate.reject(ONE_RUN.format(n=len(runs)))
    elif not cohort.rows and all(source.get("n_rollouts") == 0 for source in sources):
        gate.reject(NO_ATTEMPTS)


def check_references(references: Iterable[dict[str, Any]]) -> None:
    """Before rematerializing saved references: they name one run."""
    gate = _active()
    if gate is None:
        return
    runs = {str(reference.get("run_id") or "") for reference in references}
    if len(runs) > 1:
        gate.reject(ONE_RUN.format(n=len(runs)))


def check_prepared(selection: Any) -> None:
    """Before analysis: the prepared data covers exactly the selected run.

    Shows what the report will contain, warns about partial content, and asks before
    producing a report without a correct-versus-incorrect comparison.
    """
    gate = _active()
    if gate is None:
        return
    try:
        readiness = verify_efficiency_data(selection)
    except ValueError as exc:
        gate.reject(str(exc))
        return
    gate.note(preview(readiness))
    for warning in content_warnings(readiness):
        gate.note(warning, warning=True)
    problem = comparison_problem(readiness)
    if problem:
        gate.caution(problem)


def preview(readiness: EfficiencyReadiness) -> str:
    n = readiness.n_attempts
    unscored = n - readiness.n_pass - readiness.n_fail
    parts = [
        f"{n:,} rollouts",
        f"{readiness.n_pass:,} correct / {readiness.n_fail:,} incorrect",
        *([f"{unscored:,} unscored"] if unscored else []),
        f"token usage recorded for {readiness.n_with_tokens / n:.0%}",
    ]
    return "Efficiency data: " + " · ".join(parts)


def comparison_problem(readiness: EfficiencyReadiness) -> str | None:
    if not readiness.n_pass and not readiness.n_fail:
        return NO_REWARD
    if not readiness.n_fail:
        return ONE_OUTCOME.format(outcome="correct", missing="incorrect")
    if not readiness.n_pass:
        return ONE_OUTCOME.format(outcome="incorrect", missing="correct")
    return None


def content_warnings(readiness: EfficiencyReadiness) -> list[str]:
    from autofa.reports.efficiency_card import _SUMMARY_MIN_ATTEMPTS

    warnings = []
    if not readiness.n_with_tokens:
        warnings.append(NO_TOKENS)
    elif readiness.n_with_tokens < readiness.n_attempts:
        warnings.append(PARTIAL_TOKENS.format(
            n=f"{readiness.n_with_tokens:,}", total=f"{readiness.n_attempts:,}",
        ))
    if (
        comparison_problem(readiness) is None
        and min(readiness.n_pass, readiness.n_fail) < _SUMMARY_MIN_ATTEMPTS
    ):
        warnings.append(SMALL_GROUPS.format(
            minimum=_SUMMARY_MIN_ATTEMPTS, n_pass=readiness.n_pass, n_fail=readiness.n_fail,
        ))
    return warnings
