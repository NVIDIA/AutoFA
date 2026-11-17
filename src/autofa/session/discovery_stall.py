# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop unproductive discovery without mistaking activity for quality progress.

This is an optional cost guard for extra rounds, not a validation or minimum-detector gate. A
finished inconclusive review is a concluded investigation; an interrupted judge
or detector execution is not. No network requests or new scoring happen here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.config import resolve_objective_axes
from autofa.core.dependencies import review_source
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.completeness import completed_association_rejection
from autofa.metrics.live_progress import _evaluation_cohort
from autofa.metrics.search_ranking import search_score
from autofa.metrics.verdicts import QUALITY_THRESHOLDS, current_behavior_scores, verdict_for_scores
from autofa.objectives import discovery_strategy, weights_by_axis
from autofa.session.discovery_target import _detector_identities
from autofa.session.round_stagnation import (
    complete_discovery_round as complete_discovery_round,
)
from autofa.session.round_stagnation import (
    round_stagnation_status,
)
from autofa.util.atomic import atomic_write_json

_LOCK = threading.RLock()
_LOG = logging.getLogger(__name__)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _count(value: Any) -> int | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or value != int(value)):
        return None
    return int(value)


def stall_termination_allowed(rt: Any) -> bool:
    """Protect every planned discovery round before permitting a plateau stop.

    Round indices are zero-based, so ``max_rounds - 1`` is still a planned round.
    Missing budget metadata cannot establish that ordinary discovery is over its
    allowance. This gate never resets accumulated investigations or smoke usage.
    """
    run = getattr(rt, "run", None)
    round_idx = _count(getattr(run, "round", None))
    planned = _count(getattr(run, "max_rounds", None))
    return round_idx is not None and planned is not None and round_idx >= planned


def discovery_stall_recovery(assessment: dict) -> dict | None:
    """Direct stalled exploration toward recovery until termination is permitted.

    This is a view of the existing completed-investigation counter, not another
    round or attempt allowance. Repeated tool reads and resumed sessions return
    the same phase; only the quality observer can reset the underlying counter.
    Exhausting patience during planned rounds keeps recovery guidance active.
    """
    patience = _count(assessment.get("patience"))
    since_improve = _count(assessment.get("since_improve"))
    if (assessment.get("enabled") is not True or assessment.get("should_stop")
            or not patience or since_improve is None):
        return None
    starts_after = max(1, (patience + 1) // 2)
    if (since_improve < starts_after
            or (since_improve >= patience
                and not assessment.get("stop_deferred_by_round_budget"))):
        return None
    message = "Directed recovery: resolve the tracked evidence check before choosing another lead."
    assignment = assessment.get("recovery_assignment")
    detail = ("\nCurrent investigation (saved across calls and Resume):\n"
              + json.dumps(assignment, ensure_ascii=False, indent=2) if assignment else "")
    prompt = f"""{message}{detail}
Follow any current runtime stopping or final-validation direction first. While discovery
continues, finish this same recovery pass within the assigned discovery rounds:
1. Use saved per-case feedback to identify counterexamples and evidence gaps. Separate
implementation errors, missing evidence and a contradicted mechanism. The assigned
next_action is a starting read; inspect full assessments with details_tool when provided.
2. Check the specific distinction against the relevant requirement, actual actions or
changes, observed result and subsequent recovery. Locate the final assistant response
by role and event position, not a fixed step index. Short excerpts and the traced agent's
admissions or reassurances are clues, not proof. Retrieve the relevant
full context and artifacts before dismissing an alternative mechanism; paginate until the
decisive result is visible. Do not infer hidden tests. A complete-looking answer is not
proof of correctness, and a passing case with the same surface behavior does not refute
a distinct harmful operation.
3. Record the outcome of this evidence check in the existing family assessment: supported,
contradicted, or unresolved, citing what establishes the distinction. For a recorded family
with unavailable evidence, defer_family with the specific
gap and next retrieval step. Do not report an unavailable source as a counterexample.
A prompt-only quote, a sample refresh, status update or round advance does not complete
this check. Follow existing family prerequisites; the assignment does not bypass them.
4. Before another scored submission, state the new observable distinction that addresses
the existing counterexamples and check relevant passing contrasts. Otherwise investigate
a different mechanism in distinct failed tasks. A deep dive into a useful subtype can
add explanation, diagnostic use or a remedy without additional coverage; renaming or
changing implementation alone does not. Use Python, hybrid or LLM according to evidence.
Preserve useful existing detectors. Good and Excellent remain useful at their own
requirements; the active grade is not a retention threshold. Apply the configured Judge Policy.
The runtime decides when discovery ends. Complete meaningful investigations promptly;
withholding conclusions does not preserve an investigation allowance. Only measured quality
improvement resets the cost guard. Repeated guidance refers to this same saved check,
not a request to restart or pay for the same work again.
"""
    return {"prompt": prompt, "message": message, "assignment": assignment}


def _finished(scores: dict, fingerprint: str) -> bool:
    """Concluded investigations include unknown verdicts, never processing failures."""
    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or scores.get("evaluation_claim_hash") != fingerprint
            or (scores.get("evaluation_scope_complete") is not True
                and scores.get("eval_mode") != "capped")
            or any(_count(scores.get(key, 0)) != 0 for key in (
                "errors", "n_errors", "parent_errors"))
            or not _count(scores.get("n_scored"))):
        return False
    if completed_association_rejection(scores):
        return True
    from autofa.metrics.review_scheduling import current_review_schedule

    if current_review_schedule(scores) is not None:
        # A completed score-based triage is an investigation, not a pending judge.
        # It supplies neither behavioral support nor a quality improvement.
        return True
    rows = scores.get("behavior_verdicts")
    if (scores.get("claim_hash") != fingerprint or scores.get("behavior_judgment_stale")
            or scores.get("behavior_evidence_stale")
            or not isinstance(rows, list) or not rows
            or _count(scores.get("behavior_n_judged")) != len(rows)
            or _count(scores.get("behavior_n_errors", 0)) != 0):
        return False
    # needs_evidence_review describes an inconclusive answer, not an SDK session
    # still running. Such outcomes cannot keep discovery alive indefinitely.
    return all(isinstance(row, dict) and row.get("verdict") in {"yes", "no", "uncertain"}
               and row.get("processing_status") == "completed"
               and not row.get("validation_error")
               and row.get("completion_status") in {"complete", "needs_evidence_review"}
               for row in rows)


def _records(rt: Any) -> list[dict]:
    """Order recoverable completed claims by their original evaluation history."""
    functions = {fn.name: fn for fn in rt.library.functions}
    observations = {}
    for index, row in enumerate(rt.scores_history):
        claims = row.get("evaluated_claims") or {}
        for result in row.get("results", []):
            name = result.get("name")
            fingerprint = claims.get(name)
            fn = functions.get(name)
            if not fn or not fingerprint:
                continue
            source = review_source(fn)
            if fingerprint != claim_hash(source=source, description=fn.description or "",
                                         mechanism=fn.mechanism or fn.description or name):
                # Historical rows without complete old evidence are not invented
                # failed investigations, including renamed/replaced candidates.
                continue
            scores = current_behavior_scores(
                fn.scores, source=source, description=fn.description or "",
                mechanism=fn.mechanism or fn.description or name,
                objective_strategy=discovery_strategy(rt.config),
                behavior_strictness=strictness_for_config(rt.config),
            )
            if (scores.get("evaluation_run_id") != rt.run.run_id
                    or not _finished(scores, fingerprint)):
                continue
            claim_identity, predicate_identity = _detector_identities(fn)
            # Renaming/comment changes don't constitute another investigation.
            attempt = _digest([claim_identity, predicate_identity])
            if attempt in observations:
                continue
            cohort = _evaluation_cohort(scores)
            search = search_score(fn, rt.config, run_id=rt.run.run_id)
            breadth = search.get("breadth_cohort") or {}
            # Match counts and abstentions are outcomes, not cohort identity.
            # Including them would hide genuine coverage gains on the same panel.
            breadth_identity = [breadth.get(key) for key in (
                "scope", "cohort_fingerprint", "corpus_fingerprint",
            )]
            cohort_key = (_digest([cohort, breadth_identity])
                          if cohort is not None and search.get("eligible") else None)
            observations[attempt] = {
                "attempt": attempt, "name": name, "history_index": index,
                "claim": _digest(claim_identity), "predicate": _digest(predicate_identity),
                "grade": verdict_for_scores(scores)["label"],
                "cohort": cohort_key, "score": search.get("score"),
            }
    for record in _negative_family_records(rt):
        observations.setdefault(record["attempt"], record)
    return sorted(observations.values(), key=lambda record: record["history_index"])


def _negative_family_records(rt: Any) -> list[dict]:
    """Only accepted, evidence-checked negative comparisons are investigations."""
    try:
        state = json.loads((rt.run.run_dir() / "family_discovery.json").read_text())
    except (OSError, ValueError):
        return []
    if state.get("run_id") != rt.run.run_id:
        return []
    from autofa.session.family_coverage import cached_task_inventory

    try:
        corpus = cached_task_inventory(rt).get("corpus_fingerprint")
    except (AttributeError, OSError, ValueError):
        return []
    records = []
    for batch in [*state.get("history", []), state]:
        if batch.get("behavior_strictness", "causal") != strictness_for_config(rt.config):
            continue
        for comparison in batch.get("comparisons", []):
            if (not corpus or comparison.get("stall_corpus_fingerprint") != corpus
                    or not isinstance(comparison.get("stall_history_index"), (int, float))):
                # Historical negative findings without a source receipt remain
                # useful notes; they cannot retroactively consume this corpus's
                # allowance after a dataset change or an imported family file.
                continue
            assessment = comparison.get("assessment") or {}
            survey = comparison.get("survey") or {}
            evidence = survey.get("evidence")
            if (assessment.get("outcome") != "no_coherent_family"
                    or survey.get("outcome") != "no_coherent_family"
                    or not isinstance(evidence, list) or not evidence
                    or not all(isinstance(row, dict) and row.get("item_id")
                               and isinstance(row.get("quotes"), list) and row["quotes"]
                               and all(isinstance(quote, dict)
                                       and isinstance(quote.get("field"), str)
                                       and isinstance(quote.get("quote"), str)
                                       and quote["field"] and quote["quote"]
                                       for quote in row["quotes"])
                               and row.get("causal_chain")
                               for row in evidence)):
                continue
            # The family tool checked these source quotes before committing.
            # Names, status reads, repeated rounds and rewritten summaries do
            # not make the same inspected evidence another investigation.
            identity = sorted([
                {"item_id": row["item_id"],
                 "quotes": sorted({(quote["field"], quote["quote"])
                                   for quote in row["quotes"]})}
                for row in evidence
            ], key=lambda row: row["item_id"])
            attempt = _digest([
                "negative_family", corpus, strictness_for_config(rt.config), identity,
            ])
            records.append({
                "attempt": attempt, "name": f"negative_family:{attempt}",
                "kind": "negative_family",
                "history_index": comparison["stall_history_index"],
                "claim": attempt, "predicate": attempt, "grade": "NOT EVALUATED",
                "cohort": None, "score": None,
            })
    return records


def _quality(records: dict[str, dict]) -> dict:
    claims, predicates = set(), set()
    good = useful = 0
    for row in sorted(records.values(), key=lambda row: row["grade"] != "GOOD"):
        if row["grade"] not in {"GOOD", "OK"}:
            continue
        if row["claim"] in claims or row["predicate"] in predicates:
            continue
        claims.add(row["claim"])
        predicates.add(row["predicate"])
        useful += 1
        good += row["grade"] == "GOOD"
    return {"good": good, "good_or_ok": useful}


def observe_discovery_stall(rt: Any, *, initialize: bool = False) -> dict:
    """Persist investigation guidance, separately from completed-round stopping.

    A newly useful distinct detector, increased Good count, or better best search
    score on the same verified cohort resets patience. Weak additions, renamed
    detectors, repeated evaluations, status writes and round changes do not.
    Ordinary discovery stops only through validated extra-round accounting; these
    investigation counters trigger directed recovery.
    ``initialize`` also reconciles completed retained history for older runs.
    """
    del initialize  # Reconciliation is idempotent at start and after every turn.
    enabled = bool(getattr(rt.config, "discovery_terminate_if_stalled", True))
    patience = max(1, int(getattr(rt.config, "discovery_plateau_patience", 6)))
    termination_allowed = stall_termination_allowed(rt)
    policy = _digest([strictness_for_config(rt.config), resolve_objective_axes(rt.config),
                      rt.config.objective_rank, weights_by_axis(rt.config), QUALITY_THRESHOLDS])
    path = rt.run.run_dir() / "discovery_stall.json"
    with _LOCK:
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            state = {}
        baseline = state.get("version") != 1 or state.get("policy") != policy
        if baseline:
            generation = state.get("quality_generation", 0)
            state = {"version": 1, "policy": policy, "processed": [], "records": {},
                     "completed_attempts": 0, "since_improve": 0,
                     "best_quality": {"good": 0, "good_or_ok": 0}, "best_search": {},
                     "quality_generation": generation}
        names = {fn.name for fn in rt.library.functions}
        state["records"] = {
            name: row for name, row in state["records"].items()
            if name in names or row.get("kind") == "negative_family"
        }
        processed = set(state["processed"])
        improved, completed_now, signals = False, 0, []
        for record in _records(rt):
            previous = state["records"].get(record["name"])
            if previous == record:
                continue
            state["records"][record["name"]] = record
            quality = _quality(state["records"])
            gains = [key for key, value in quality.items() if value > state["best_quality"][key]]
            for key, value in quality.items():
                state["best_quality"][key] = max(value, state["best_quality"][key])
            cohort, value = record["cohort"], record["score"]
            if (record["grade"] in {"GOOD", "OK"} and cohort
                    and isinstance(value, (int, float)) and math.isfinite(value)):
                best = state["best_search"].get(cohort)
                # An unknown/new panel establishes a baseline, never progress.
                if best is not None and value > best + 1e-6:
                    gains.append("comparable_search_score")
                state["best_search"][cohort] = max(value, best if best is not None else value)
            new_attempt = record["attempt"] not in processed
            if new_attempt:
                processed.add(record["attempt"])
                state["completed_attempts"] += 1
                completed_now += 1
            if gains:
                state["since_improve"] = 0
                improved = True
                signals.extend(gains)
            elif new_attempt:
                state["since_improve"] += 1
        state["processed"] = sorted(processed)
        if improved:
            state["quality_generation"] = state.get("quality_generation", 0) + 1
        from autofa.session.discovery_recovery import update_recovery_assignment
        from autofa.session.smoke_work import diagnostic_budget

        recovery_assignment = update_recovery_assignment(
            rt, state, active=(enabled and max(1, (patience + 1) // 2)
                               <= state["since_improve"]),
        )
        atomic_write_json(path, state)
        # The diagnostic budget reads the persisted quality generation.
        diagnostics = diagnostic_budget(rt)
        diagnostic_exhausted = diagnostics.get("allowance_exhausted", diagnostics["exhausted"])
        stalled = state["since_improve"] >= patience or diagnostic_exhausted
        round_status = round_stagnation_status(rt)
        # A capacity receipt is written only after full polishing verification.
        # Honor it even if a crash preceded committing that round's new cursor.
        should_stop = bool(round_status.get("review_capacity_block")) or (
            round_status["should_stop"] and termination_allowed)
        reason = round_status["reason"]
        return {
            "enabled": enabled, "should_stop": should_stop,
            "stop_reason": round_status.get("stop_reason") if should_stop else None,
            "review_capacity_block": round_status.get("review_capacity_block"),
            "round_stagnation": round_status,
            "investigations_trigger_recovery_only": True,
            "termination_allowed": termination_allowed,
            "stop_deferred_by_round_budget": enabled and stalled and not should_stop,
            "since_improve": state["since_improve"], "patience": patience,
            "completed_attempts": state["completed_attempts"], "completed_now": completed_now,
            "improved": improved, "progress_signals": sorted(set(signals)),
            "best_quality": state["best_quality"], "baseline_initialized": baseline,
            "diagnostic_budget": diagnostics,
            "recovery_assignment": recovery_assignment,
            "reason": reason,
        }


def checkpoint_discovery_stall(rt: Any) -> None:
    """Record committed investigations before a later tool replaces the detector.

    An accounting write must never turn completed evaluation/review work into a
    failed tool response. The driver retries reconciliation at its next boundary.
    """
    try:
        observe_discovery_stall(rt)
    except Exception:
        _LOG.warning("Could not checkpoint discovery quality progress", exc_info=True)
