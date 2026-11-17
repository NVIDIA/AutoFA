# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist evidence for top-down families and bottom-up detector synthesis.

This is an authoring workflow, not another causal judge. Evidence citations are
checked against the selected dataset; whether they establish the proposed cause
is still determined by the existing detector evaluation and claim review.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Literal

from autofa.behavior_policy import strictness_for_config
from autofa.objectives import discovery_strategy
from autofa.session.synthesis_decisions import (
    FAMILY_DECISION_SCHEMA as FAMILY_DECISION_SCHEMA,
)
from autofa.session.synthesis_decisions import (
    UNRESOLVED_OUTCOMES,
    checked_decision,
    continue_decision,
    finish_decision,
    pending_decision,
    record_decision,
)
from autofa.util.atomic import atomic_write_json

STATE_FILE = "family_discovery.json"
SURVEY_TARGETS = {"precision": 3, "balanced": 6, "recall": 8}
COMPARISON_BUDGETS = {"precision": 1, "balanced": 3, "recall": 5}
MIN_BROAD_FAMILY_TASKS = 2
_MAX_EVIDENCE = 24
_RUN_LOCKS: dict[Path, Any] = {}
_RUN_LOCKS_GUARD = threading.Lock()

FAMILY_ACTION_ARGUMENTS = {
    "status": [],
    "record_family": ["family_name", "description", "mechanism", "evidence", "boundary_notes"],
    "no_coherent_family": ["evidence", "boundary_notes", "assessment", "decision"],
    "assess_family": ["family_name", "assessment"],
    "defer_family": ["family_name", "blocker", "next_step"],
}
FAMILY_EVIDENCE_SCHEMA = {
    "type": "array", "minItems": 1, "maxItems": _MAX_EVIDENCE,
    "items": {
        "type": "object",
        "required": ["item_id", "role", "causal_chain", "quotes"],
        "properties": {
            "item_id": {"type": "string"},
            "role": {"type": "string", "enum": ["support", "boundary", "unrelated", "uncertain"]},
            "causal_chain": {"type": "string", "minLength": 1, "maxLength": 4000},
            "quotes": {
                "type": "array", "minItems": 1, "maxItems": 6,
                "items": {"type": "object", "required": ["field", "quote"],
                          "properties": {"field": {"type": "string"},
                                         "quote": {"type": "string", "minLength": 1,
                                                   "maxLength": 2000}}},
            },
        },
    },
}


def family_argument_guidance(action: Any) -> dict[str, Any]:
    """Expose the whole action contract when a malformed call needs correction."""
    return {
        "required_arguments": FAMILY_ACTION_ARGUMENTS.get(str(action), []),
        "action_arguments": FAMILY_ACTION_ARGUMENTS,
        "evidence_schema": FAMILY_EVIDENCE_SCHEMA,
        "decision_schema": FAMILY_DECISION_SCHEMA,
        "detector_link": "Use detector_name to reference an existing differently named detector. "
                         "Keep family_name as the recorded hypothesis name; do not duplicate "
                         "a detector just to match that name. Description and mechanism must "
                         "match the recorded claim before assessment.",
    }


def family_detector_name(survey: dict[str, Any]) -> str | None:
    """Keep historical same-name surveys compatible with explicit detector links."""
    return survey.get("detector_name") or survey.get("family_name")


def _claim_differences(fn: Any, survey: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {key: {"recorded_family": survey[key], "detector": getattr(fn, key) or ""}
            for key in ("description", "mechanism")
            if (getattr(fn, key) or "").strip() != survey[key].strip()}

FAMILY_CONTRIBUTION_GUIDANCE = (
    "Before retaining another detector from the same family, use boundary_notes or "
    "assessment to explain its contribution: a different mechanism, a useful subtype, "
    "a different remedy, or a distinction that resolves an observed counterexample. "
    "A subtype can be useful even with no additional task coverage; establish its own "
    "claim and support across tasks. A confidence filter, wording change or alternate "
    "implementation of the same claim is normally a revision, not another discovery. "
    "Preserve supported versions while comparing; overlap alone is not a reason to "
    "remove a detector. Semantic parentage does not require execution gating."
)
FAMILY_EVIDENCE_GUIDANCE = (
    "Before closing an alternative mechanism as unsupported, inspect the evidence "
    "needed to distinguish it from competing explanations. Short excerpts, summaries "
    "and the traced agent's own explanation are leads, not sufficient evidence to rule "
    "it out. Retrieve the relevant requirements, actions, results, later recovery and "
    "source artifacts when needed; inspect the full trace when the claim requires it. "
    "Check the decisive distinction within the existing investigation budget, not an "
    "exhaustive catalog. If decisive evidence is unavailable, keep the lead open: "
    "use defer_family with blocker and next_step only for a recorded family in "
    "assess_family; otherwise save the lead, evidence gap and next step with "
    "report_status(message=...). Do not invent support to record a family. "
    "Unavailable evidence does not refute the mechanism."
)


def family_transaction(function: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize a run's complete read/check/write transition across MCP workers."""
    parameters = signature(function)

    @wraps(function)
    def locked(*args: Any, **kwargs: Any) -> Any:
        from autofa.mcp.runtime import get_runtime
        from autofa.util.cancellation import check_cancelled

        rt = parameters.bind_partial(*args, **kwargs).arguments.get("rt")
        if rt is None:
            rt = get_runtime()
        directory = Path(rt.run.run_dir()).resolve()
        with _RUN_LOCKS_GUARD:
            lock = _RUN_LOCKS.setdefault(directory, threading.RLock())
        # Reentrant: family tools initialize state and consuming guards can
        # refresh synthesis inside this same transaction.
        with lock:
            check_cancelled()
            return function(*args, **kwargs)
    return locked


def _load(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = Path(run_dir) / STATE_FILE
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("version") != 3:
        raise ValueError("Unsupported family discovery state; preserve it before restarting")
    return state


def _inventory(rt: Any) -> dict[str, Any]:
    # The inventory contains projected identity/label fields only, never trajectories.
    # A runtime selects a fixed dataset, so status/tool checks can reuse this small map.
    from autofa.session.family_coverage import cached_task_inventory

    return cached_task_inventory(rt)


def _inspected_failures(state: dict[str, Any]) -> tuple[set[str], set[str]]:
    tasks, items = set(), set()
    for comparison in state.get("comparisons") or []:
        for row in (comparison.get("survey") or {}).get("evidence") or []:
            if row.get("outcome") is False:
                items.add(row["item_id"])
                if row.get("task_id"):
                    tasks.add(row["task_id"])
    return tasks, items


def _exploration_remaining(rt: Any, state: dict[str, Any]) -> bool:
    if len(state["comparisons"]) >= state["comparison_budget"]:
        state["exploration_exhausted"] = "comparison_budget_completed"
        return False
    inventory = _inventory(rt)
    available = inventory["task_by_failed_item_id"]
    known = {task for task in available.values() if task}
    unknown = {item for item, task in available.items() if not task}
    inspected_tasks, inspected_items = _inspected_failures(state)
    if (inventory.get("inventory_complete")
            and not (known - inspected_tasks or unknown - inspected_items)):
        state["exploration_exhausted"] = "available_failed_task_evidence_exhausted"
        return False
    state["exploration_exhausted"] = None
    return True


def _complete_comparison(rt: Any, state: dict[str, Any]) -> None:
    state["comparisons"].append({"round": state["round"], "survey": state["survey"],
                                 "assessment": state["assessment"],
                                 "stall_history_index": len(getattr(rt, "scores_history", [])) - .5,
                                 "stall_corpus_fingerprint": _inventory(rt).get(
                                     "corpus_fingerprint")})
    state["phase"] = "survey" if _exploration_remaining(rt, state) else "specialize"
    # Keep the last candidate/assessment visible for summaries; the phase determines
    # whether it remains active. A new record replaces it without erasing comparison.


@family_transaction
def initialize_family_discovery(rt: Any) -> dict[str, Any]:
    """Preserve bounded family comparisons and pending work across tool calls/resume."""
    from autofa.session.round_feedback import round_snapshot

    state = _load(rt.run.run_dir())
    new_round_baseline = state is None
    previous = json.dumps(state, sort_keys=True)
    round_idx = int(getattr(rt.run, "round", 0) or 0)
    strategy = discovery_strategy(rt.config)
    if state is None:
        state = {"version": 3, "run_id": rt.run.run_id, "round": round_idx,
                 "phase": "survey", "objective_strategy": strategy,
                 "behavior_strictness": strictness_for_config(rt.config),
                 "survey": None, "assessment": None, "history": [],
                 "comparisons": [], "mutation_attempts": {},
                 "comparison_budget": COMPARISON_BUDGETS[strategy],
                 "exploration_exhausted": None}
    elif state.get("run_id") != rt.run.run_id:
        raise ValueError("Family discovery state belongs to a different run")
    strictness = strictness_for_config(rt.config)
    if state.get("behavior_strictness", "causal") != strictness:
        # Preserve prior evidence but never carry an assessed family or exhausted search
        # into another acceptance policy as if its judgments were current.
        state.setdefault("history", []).append({
            **{key: state.get(key) for key in (
                "round", "survey", "assessment", "comparisons", "mutation_attempts",
                "synthesis_required", "deferred_synthesis", "synthesis_checks",
                "round_baseline",
            )},
            "behavior_strictness": state.get("behavior_strictness", "causal"),
            "reason": "behavior_strictness_changed",
        })
        state.update(round=round_idx, phase="survey", survey=None, assessment=None,
                     comparisons=[], mutation_attempts={}, exploration_exhausted=None,
                     comparison_budget=COMPARISON_BUDGETS[strategy])
        for key in ("synthesis_required", "deferred_synthesis", "synthesis_checks"):
            state.pop(key, None)
        new_round_baseline = True
    state["behavior_strictness"] = strictness
    if state["phase"] == "specialize" and round_idx > state["round"]:
        outcome = round_snapshot(rt)
        state["history"].append({**{key: state.get(key) for key in (
            "round", "survey", "assessment", "comparisons", "behavior_strictness",
            "round_baseline",
        )}, "round_outcome": outcome})
        state.update(round=round_idx, phase="survey", survey=None, assessment=None,
                     comparisons=[], mutation_attempts={}, exploration_exhausted=None,
                     comparison_budget=COMPARISON_BUDGETS[strategy])
        state["round_baseline"] = outcome
        new_round_baseline = False
        if deferred := state.pop("deferred_synthesis", None):
            # The evidence inspection cost no extra comparison in the exhausted
            # round. Its recorded hypothesis gets the next ordinary comparison.
            state.update(survey=deferred, phase="assess_family")
    state["objective_strategy"] = strategy
    state["suggested_distinct_tasks"] = SURVEY_TARGETS[strategy]
    if state["phase"] == "survey" and not _exploration_remaining(rt, state):
        state["phase"] = "specialize"
    _refresh_synthesis_requirement(rt, state)
    if new_round_baseline:
        state["round_baseline"] = round_snapshot(rt)
    if json.dumps(state, sort_keys=True) != previous:
        atomic_write_json(rt.run.run_dir() / STATE_FILE, state)
    return state


def _checked_evidence(rt: Any, evidence: Any) -> list[dict[str, Any]]:
    from autofa.data.parquet_store import get_item
    from autofa.data.row_index import _row_is_correct
    from autofa.session.family_coverage import verified_identity_projection
    from autofa.session.review_evidence import record_discovery_exposure

    if not isinstance(evidence, list) or not 1 <= len(evidence) <= _MAX_EVIDENCE:
        raise ValueError(f"Provide 1–{_MAX_EVIDENCE} inspected evidence records")
    result, seen, exposed = [], set(), []
    for record in evidence:
        if not isinstance(record, dict):
            raise ValueError("Each evidence record must be an object")
        item_id = str(record.get("item_id") or "").strip()
        if not item_id or item_id in seen:
            raise ValueError("Evidence must have distinct, nonempty item_id values")
        seen.add(item_id)
        role = record.get("role")
        if (not isinstance(role, str)
                or role not in {"support", "boundary", "unrelated", "uncertain"}):
            raise ValueError("Evidence role must be support, boundary, unrelated, or uncertain")
        chain = str(record.get("causal_chain") or "").strip()
        if not chain or len(chain) > 4000:
            raise ValueError("Each item needs a causal_chain of at most 4000 characters")
        loaded = get_item(rt.dataset, item_id, max_chars=None, include_trace=True)
        if not loaded.get("ok"):
            raise ValueError(f"Evidence item is not in the selected dataset: {item_id}")
        item = loaded["item"]
        outcome = _row_is_correct(item)
        if role == "support" and outcome is not False:
            raise ValueError(f"Family support must have a recorded failed outcome: {item_id}")
        quotes = record.get("quotes")
        if not isinstance(quotes, list) or not 1 <= len(quotes) <= 6:
            raise ValueError("Each item needs 1–6 quotes with field and quote")
        checked = []
        for citation in quotes:
            if not isinstance(citation, dict):
                raise ValueError("Each quote must be an object with field and quote")
            field = str(citation.get("field") or "")
            quote = str(citation.get("quote") or "")
            if not quote.strip() or len(quote) > 2000:
                raise ValueError("Each quote must be nonempty and at most 2000 characters")
            value = item.get(field)
            # Support exact parsed text as well as canonical serialized values. The
            # corpus readers often display decoded message strings from JSON fields.
            if not _contains_quote(value, quote):
                raise ValueError(f"Quote not found in {item_id} field {field!r}")
            checked.append({"field": field, "quote": quote})
        task_id = _inventory(rt)["task_by_item_id"].get(item_id)
        result.append({"item_id": item_id, "task_id": task_id,
                       "outcome": outcome, "role": role,
                       "causal_chain": chain, "quotes": checked})
        exposed.append(verified_identity_projection(rt, item_id))
    record_discovery_exposure(rt.run.run_dir(), exposed)
    return result


def _contains_quote(value: Any, quote: str) -> bool:
    if isinstance(value, str):
        if quote in value:
            return True
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return False
        return _contains_quote(parsed, quote) if not isinstance(parsed, str) else quote in parsed
    if isinstance(value, dict):
        return any(_contains_quote(child, quote) for child in value.values())
    if isinstance(value, list):
        return any(_contains_quote(child, quote) for child in value)
    return False


def _seed_candidates(rt: Any) -> list[dict[str, Any]]:
    """Current per-case support can seed a hypothesis, even before review completes."""
    from autofa.core.dependencies import review_source
    from autofa.metrics.claim_contract import claim_hash
    from autofa.metrics.library_coverage import library_failure_coverage
    from autofa.metrics.verdicts import behavior_validation, current_behavior_scores

    inventory = _inventory(rt)
    coverage = library_failure_coverage(
        rt.library, inventory["task_by_failed_item_id"], run_id=rt.run.run_id,
        objective_strategy=discovery_strategy(rt.config), include_item_ids=True,
        behavior_strictness=strictness_for_config(rt.config),
        task_by_item_id=inventory["task_by_item_id"],
    )
    candidates = []
    for row in coverage["detectors"]:
        if not row["n_review_confirmed_failed"]:
            continue
        fn = rt.library.get(row["name"])
        source = review_source(fn)
        reviewed = current_behavior_scores(
            fn.scores, source=source, description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
            objective_strategy=discovery_strategy(rt.config),
            behavior_strictness=strictness_for_config(rt.config),
        )
        validation = behavior_validation(reviewed, include_details=False)
        candidates.append({
            "name": fn.name, "description": fn.description, "mechanism": fn.mechanism,
            "claim_hash": claim_hash(source=source, description=fn.description or "",
                                     mechanism=fn.mechanism or fn.description or fn.name),
            "precision": fn.scores.get("precision"), "recall": fn.scores.get("recall"),
            "review": {key: validation[key] for key in (
                "status", "sampled", "confirmed", "rejected", "uncertain",
                "processing_errors", "ranking_eligible",
            )},
            **{key: row[key] for key in (
                "n_known_matched_failed", "n_matched_failed_tasks",
                "n_review_confirmed_failed", "n_review_confirmed_failed_tasks",
                "generalization_status", "known_matched_failed_item_ids",
                "review_confirmed_failed_item_ids",
            )},
        })
    return candidates


def _checked_seeds(rt: Any, names: Any, family_name: str,
                   rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (not isinstance(names, list) or len(names) < 2
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(names) != len(set(names))):
        raise ValueError("seed_functions needs at least two distinct existing detector names")
    if family_name in names:
        raise ValueError("A synthesized family cannot be its own seed; preserve its sources")
    candidates = {row["name"]: row for row in _seed_candidates(rt)}
    support_ids = {row["item_id"] for row in rows if row["role"] == "support"}
    seeds = []
    for name in names:
        candidate = candidates.get(name)
        if candidate is None:
            raise ValueError(f"Seed {name!r} needs a current evaluation and individually "
                             "confirmed failed support under its current claim and judge policy")
        cited = support_ids & set(candidate["review_confirmed_failed_item_ids"])
        if not cited:
            raise ValueError(f"Cite checked support evidence from seed {name!r}'s current "
                             "review-confirmed failed cases")
        seeds.append({"name": name, "claim_hash": candidate["claim_hash"],
                      "support_item_ids": sorted(cited)})
    if len({seed["claim_hash"] for seed in seeds}) != len(seeds):
        raise ValueError("Duplicate seed claims do not establish distinct source detectors")
    return seeds


def _seed_provenance(rt: Any, seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep historical lineage truthful when a source is removed, edited or reevaluated."""
    candidates = {row["name"]: row for row in _seed_candidates(rt)}
    stale = []
    for seed in seeds:
        current = candidates.get(seed["name"])
        if (current is None or current["claim_hash"] != seed["claim_hash"]
                or not set(seed.get("support_item_ids") or []) <= set(
                    current["review_confirmed_failed_item_ids"])):
            stale.append(seed["name"])
    return {"current": not stale, "stale_seed_functions": stale}


def _refresh_synthesis_requirement(rt: Any, state: dict[str, Any]) -> None:
    """Schedule one evidence inspection after stalled top-down work, without a judge.

    The checkpoint survives rounds and resume. Changing a seed's actual claim or
    adding confirmed task support can warrant another inspection; repeated rollouts,
    repeated status reads and changes to numeric scores cannot.
    """
    from autofa.session.research_status import read_research_status

    if (state.get("deferred_synthesis")
            or (state["phase"] == "assess_family"
                and (state.get("survey") or {}).get("approach") == "bottom_up")):
        # Finish the checked parent before scheduling another inspection. Changed
        # seed evidence must not block a deferred parent's next-round activation.
        state.pop("synthesis_required", None)
        return
    task_map = _inventory(rt)["task_by_failed_item_id"]
    seeds = []
    claims = set()
    for candidate in sorted(_seed_candidates(rt), key=lambda row: row["name"]):
        tasks = sorted({task_map[item_id]
                        for item_id in candidate["review_confirmed_failed_item_ids"]
                        if task_map.get(item_id)})
        if not tasks or candidate["claim_hash"] in claims:
            continue
        claims.add(candidate["claim_hash"])
        seeds.append({"name": candidate["name"], "claim_hash": candidate["claim_hash"],
                      "confirmed_task_ids": tasks})
    if len(seeds) < 2 or len({task for seed in seeds
                              for task in seed["confirmed_task_ids"]}) < 2:
        state.pop("synthesis_required", None)
        return
    signature = hashlib.sha256(json.dumps(seeds, sort_keys=True).encode()).hexdigest()
    if any(row.get("signature") == signature for row in state.get("synthesis_checks", [])):
        state.pop("synthesis_required", None)
        return
    existing = state.get("synthesis_required") or {}
    if existing.get("signature") == signature:
        return
    survey = state.get("survey") or {}
    top_down = survey.get("approach", "top_down") == "top_down"
    progress = read_research_status(rt.run.run_dir()).get("task_coverage_progress") or {}
    mutations = (state.get("mutation_attempts", {}).get(family_detector_name(survey), 0)
                 if state["phase"] == "assess_family"
                 else max(state.get("mutation_attempts", {}).values(), default=0))
    comparisons = [row for previous in state.get("history", [])
                   if previous.get("behavior_strictness", "causal") == state["behavior_strictness"]
                   for row in previous.get("comparisons", [])]
    comparisons.extend(state.get("comparisons") or [])
    # A previous bottom-up comparison resets the top-down streak; it must not
    # exempt all future rounds or newly supported seeds from another inspection.
    top_down_comparisons = 0
    for row in comparisons:
        if (row.get("survey") or {}).get("approach", "top_down") == "bottom_up":
            top_down_comparisons = 0
        else:
            top_down_comparisons += 1
    if top_down and mutations >= 2:
        reason = "repeated_top_down_mutations"
    elif progress.get("experiments_since_improve", 0) > 0:
        reason = "confirmed_task_coverage_stalled"
    elif (top_down_comparisons
          and (state["phase"] == "specialize"
               or top_down_comparisons >= max(1, state["comparison_budget"] - 1))):
        reason = "top_down_comparison_budget"
    else:
        state.pop("synthesis_required", None)
        return
    state["synthesis_required"] = {
        "signature": signature, "reason": reason, "seeds": seeds,
        "seed_functions": [seed["name"] for seed in seeds],
    }


def _resolve_synthesis_requirement(state: dict[str, Any], *, outcome: str,
                                   seeds: list[dict[str, Any]]) -> None:
    requirement = state.pop("synthesis_required", None)
    if requirement:
        state.setdefault("synthesis_checks", []).append({
            **requirement, "round": state["round"], "outcome": outcome,
            "inspected_seed_functions": [seed["name"] for seed in seeds],
            "scope": "One inspected seed subset; other combinations remain unexplored.",
        })


def _synthesis_status(rt: Any, state: dict[str, Any]) -> dict[str, Any]:
    candidates = _seed_candidates(rt)
    task_map = _inventory(rt)["task_by_failed_item_id"]
    tasks = {task_map[item_id] for row in candidates
             for item_id in row["known_matched_failed_item_ids"] if task_map.get(item_id)}
    best_tasks = max((row["n_matched_failed_tasks"] for row in candidates), default=0)
    promising = len(candidates) >= 2 and len(tasks) > best_tasks
    pending = state["phase"] == "assess_family"
    deferred = state.get("deferred_synthesis")
    required = state.get("synthesis_required")
    direction = ("assess_current" if pending else "next_round" if deferred else "bottom_up"
                 if required or promising else "top_down")
    cards = []
    for row in candidates:
        distinct, other, seen = [], [], set()
        for item_id in row["review_confirmed_failed_item_ids"]:
            task = task_map.get(item_id)
            if task is not None and task not in seen:
                distinct.append(item_id)
                seen.add(task)
            else:
                other.append(item_id)
        cards.append({
            key: value for key, value in row.items()
            if key not in {"known_matched_failed_item_ids", "review_confirmed_failed_item_ids"}
        } | {"support_item_ids": (distinct + other)[:3]})
    seeds = (state.get("survey") or {}).get("seed_functions") or []
    return {
        "recommended_direction": direction,
        "synthesis_candidates": cards,
        "seed_provenance": _seed_provenance(rt, seeds) if seeds else None,
        "synthesis_guidance": (
            "A coherent synthesis hypothesis is saved for the next round. Call advance_round "
            "to activate it within that round's ordinary comparison budget; then implement, "
            "evaluate and assess that recorded family before another new hypothesis. "
            if deferred else
            "A bottom-up evidence inspection is due before the next new family comparison. "
            "Finish or defer a pending family, then inspect current seed support across tasks. "
            "Record a shared mechanism with seed_functions, or use no_coherent_family with "
            "seed_functions, checked evidence and an explanation of why synthesis is unsound. "
            if required else
            "Finish or defer the recorded hypothesis before changing direction. " if pending else
            "Reviewed detectors cover complementary tasks; try bottom-up synthesis if their "
            "evidence supports one shared mechanism under the selected review standard. "
            if promising else
            "Inspect uncovered failures for a top-down hypothesis, or synthesize related "
            "review-supported detectors when their evidence warrants it. "
        ) + "Direction scheduling is not semantic grouping or proof of a broader parent. Current "
            "per-item confirmations from limited/development review can seed a hypothesis. "
            "Preserve the sources, inspect their evidence and contrasts, and independently "
            "evaluate/review the new claim; repeated attempts do not establish task breadth.",
    }


def _status(state: dict[str, Any]) -> dict[str, Any]:
    survey = state.get("survey") or {}
    evidence = survey.get("evidence") or []
    support_tasks = {e["task_id"] for e in evidence
                     if e["role"] == "support" and e["task_id"] is not None}
    tasks = {e["task_id"] for e in evidence if e["task_id"] is not None}
    phase = state["phase"]
    completed = len(state.get("comparisons") or [])
    budget = state.get("comparison_budget", 1)
    next_action = {
        "survey": "Choose a top-down hypothesis from new failed tasks or synthesize a shared "
                  "mechanism from reviewed narrow seeds; record_family with checked evidence. "
                  "Earlier rare candidates remain available. If the top-down survey finds no "
                  "coherent family, document no_coherent_family and retain useful diagnostics.",
        "assess_family": "Implement and evaluate the recorded claim if needed. Inspect the "
                         "detector's saved evaluation and feedback, then "
                         "assess_family. Pass detector_name if the detector has a different "
                         "name; do not submit a duplicate to match family_name. Assess or defer, "
                         "then inspect "
                         "uncovered failures for breadth or parent matches and contrasts "
                         "for a useful subtype; family effort counts do not block useful edits.",
        "specialize": "The bounded comparison phase is complete. Compare candidates' recall, "
                      "distinct tasks and additional coverage; synthesize a broader shared "
                      "mechanism from precise narrow seeds or develop informative children "
                      "with semantic parentage. Use explains only when intentionally restricting "
                      "execution to the parent detector's matches. Retain useful rare findings.",
    }[phase]
    if phase == "specialize" and completed >= budget:
        next_action += " Call advance_round before starting another family comparison."
    comparisons = [{"family_name": (row.get("survey") or {}).get("family_name"),
                    "detector_name": family_detector_name(row.get("survey") or {}),
                    "approach": (row.get("survey") or {}).get("approach", "top_down"),
                    "seed_functions": (row.get("survey") or {}).get("seed_functions", []),
                    "outcome": (row.get("assessment") or {}).get("outcome"),
                    "scores": (row.get("assessment") or {}).get("scores"),
                    "coverage": (row.get("assessment") or {}).get("coverage"),
                    "seed_comparison": (row.get("assessment") or {}).get("seed_comparison"),
                    "seed_provenance": (row.get("assessment") or {}).get("seed_provenance"),
                    "decision": (row.get("assessment") or {}).get("decision")}
                   for row in state.get("comparisons") or []]
    inspected_tasks, inspected_items = _inspected_failures(state)
    assessment = state.get("assessment")
    requirement = state.get("synthesis_required")
    if requirement:
        next_action = (
            "Finish or defer the current family first. " if phase == "assess_family" else ""
        ) + ("Before another top-down comparison, inspect supported seeds across distinct tasks. "
             "Use record_family(seed_functions=[...]) for one shared mechanism, or "
             "no_coherent_family(seed_functions=[...]) with checked evidence and an explanation "
             "of incompatible mechanisms. Preserve seed detectors and their supported cases; "
             "additional task coverage is desirable, not a prerequisite for consolidation.")
    deferred = state.get("deferred_synthesis")
    if deferred:
        next_action = (
            f"Call advance_round to activate the saved synthesis {deferred['family_name']!r} "
            "as the next ordinary family comparison. The current round's comparison "
            "budget is exhausted; no extra evaluation or review is required now."
        )
    pending = pending_decision(state)
    if pending and phase == "survey" and not deferred:
        next_action = pending["instruction"] + " " + next_action
    return {"phase": phase, "round": state["round"],
            "objective_strategy": state["objective_strategy"],
            "behavior_strictness": state["behavior_strictness"],
            "suggested_distinct_tasks": state["suggested_distinct_tasks"],
            "comparison_budget": budget, "completed_comparisons": completed,
            "remaining_comparisons": max(0, budget - completed), "comparisons": comparisons,
            "exploration_exhausted": state.get("exploration_exhausted"),
            "n_previously_inspected_failed_tasks": len(inspected_tasks),
            "n_previously_inspected_failed_items": len(inspected_items),
            "mutation_attempts": state.get("mutation_attempts", {}),
            "family_name": survey.get("family_name"),
            "detector_name": family_detector_name(survey),
            "recorded_claim": {key: survey.get(key) for key in ("description", "mechanism")}
            if survey else None,
            "approach": survey.get("approach", "top_down"),
            "seed_functions": survey.get("seed_functions", []),
            "n_inspected_items": len(evidence), "n_inspected_tasks": len(tasks),
            "n_supporting_tasks": len(support_tasks),
            "minimum_broad_family_tasks": MIN_BROAD_FAMILY_TASKS,
            "cross_task_support_observed": len(support_tasks) >= MIN_BROAD_FAMILY_TASKS,
            "scope_limit": survey.get("scope_limit"),
            "assessment": assessment,
            "completed_surveys": completed,
            "synthesis_required": bool(requirement),
            "synthesis_pending": bool(requirement or deferred or (
                phase == "assess_family" and survey.get("approach") == "bottom_up"
            )),
            "synthesis_reason": requirement.get("reason") if requirement else None,
            "synthesis_seed_functions": (requirement or {}).get("seed_functions", []),
            "completed_synthesis_checks": len(state.get("synthesis_checks") or []),
            "deferred_synthesis": deferred,
            "pending_decision": pending,
            "decision_record_count": len(state.get("decision_records", {})),
            "recent_decisions": [
                {"id": row["id"], "status": row["status"],
                 "outcome": row["decision"]["outcome"],
                 "shared_relation": row["decision"]["shared_relation"][:400],
                 "reason": row["decision"]["reason"][:600],
                 "next_action": row["decision"]["next_action"][:400],
                 "experiment_id": row["decision"].get("experiment_id"),
                 "comparison_id": row["decision"].get("comparison_id"),
                 "requests": row["requests"], "last_round": row["last_round"],
                 "reference": f"family_discovery.json#/decision_records/{row['id']}"}
                for row in list(state.get("decision_records", {}).values())[-5:]
            ],
            "next_action": next_action,
            "candidate_guidance": FAMILY_CONTRIBUTION_GUIDANCE,
            "evidence_guidance": FAMILY_EVIDENCE_GUIDANCE,
            "interpretation": "This is discovery progress, not causal validation. Survey "
                              "quotes and task identities are checked; the proposed behavioral "
                              "interpretation remains subject to normal evaluation and review."}


def family_progress_summary(run_dir: Path | None) -> dict[str, Any] | None:
    """Read persisted progress without initializing a phase, dataset scan or judge call."""
    state = _load(run_dir)
    return _status(state) if state else None


def family_comparison_pending(run_dir: Path | None) -> bool:
    """Whether family comparisons remain before reserved confirmation."""
    state = _load(run_dir)
    return bool(state and state["phase"] != "specialize")


def _check_survey_breadth(rt: Any, state: dict[str, Any], rows: list[dict[str, Any]],
                         *, revising: bool) -> None:
    available = _inventory(rt)["task_by_failed_item_id"]
    known = {task for task in available.values() if task}
    unknown = {item for item, task in available.items() if not task}
    failed = [row for row in rows if row["outcome"] is False]
    failed_tasks = {row["task_id"] for row in failed if row["task_id"]}
    failed_items = {row["item_id"] for row in failed}
    required = min(state["suggested_distinct_tasks"], len(known))
    if len(failed_tasks) < required:
        raise ValueError(f"Inspect at least {required} distinct failed tasks for this survey "
                         f"({len(failed_tasks)} supplied). scope_limit cannot replace available "
                         "task diversity; unrelated failures can be included as contrasts.")
    if revising:
        return
    inspected_tasks, inspected_items = _inspected_failures(state)
    remaining = known - inspected_tasks
    needed_new = min(2, len(remaining))
    if len(failed_tasks - inspected_tasks) < needed_new:
        raise ValueError(f"Inspect {needed_new} new failed tasks not used in earlier family "
                         "comparisons before recording another search attempt.")
    if not remaining:
        remaining_unknown = unknown - inspected_items
        needed_items = min(2, len(remaining_unknown))
        if len((failed_items - inspected_items) & remaining_unknown) < needed_items:
            raise ValueError(f"Task identities are unavailable for remaining failures. Inspect "
                             f"{needed_items} new failed items; these do not establish "
                             "task diversity.")
        if state["comparisons"] and not remaining_unknown:
            raise ValueError("Available failed evidence is exhausted; finish bounded exploration.")


@family_transaction
def family_discovery(  # noqa: C901
    action: Literal["status", "record_family", "no_coherent_family",
                    "assess_family", "defer_family"] = "status", *,
    family_name: str = "", detector_name: str = "", description: str = "",
    mechanism: str = "", evidence: list[dict[str, Any]] | None = None,
    boundary_notes: str = "", scope_limit: str = "", assessment: str = "",
    blocker: str = "", next_step: str = "",
    seed_functions: list[str] | None = None,
    decision: dict[str, Any] | None = None,
    prior_decision_id: str = "",
) -> dict[str, Any]:
    """Manage top-down families or bottom-up synthesis through the same assessment.

    record_family: family_name, description, mechanism, boundary_notes, and evidence.
    detector_name optionally links the hypothesis to a differently named detector.
    Omit it to use family_name (or preserve an existing link when revising). Keep the
    exact detector description and mechanism when recording an existing claim; family
    names are bookkeeping and never require a duplicate detector or another review.
    Each evidence record has item_id, role (support|boundary|unrelated|uncertain),
    causal_chain (evidence map for the selected behavioral review standard), and
    quotes=[{field: 'response_text', quote: 'verbatim source text'}]. Task identities
    and outcomes come from the dataset. Inspect different tasks, not just repeats.
    scope_limit explains unavailable cross-task support or missing natural negatives.
    prior_decision_id explicitly links record_family to a saved unresolved decision;
    retain that decision's checked supporting cases. Other families using similar
    examples do not silently close the unresolved hypothesis.
    In boundary_notes or assessment, explain what a same-family candidate adds beyond
    existing detectors: a different mechanism, useful subtype, remedy or resolved
    counterexample. Overlap is allowed; wording/confidence-only variants normally revise
    the existing claim. A useful subtype does not require additional task coverage.
    seed_functions: optional >=2 existing detector names for bottom-up synthesis.
    Cite checked support from each seed's current per-item confirmed failures. Limited
    development reviews can support a hypothesis; they do not validate the parent.
    Synthesis can reuse inspected evidence. A scheduled inspection after the comparison
    budget is exhausted saves a proposal for the next round without an extra comparison;
    single-task seeds need explicit scope_limit. It creates semantic provenance only,
    never execution dependencies, and never removes or changes the source detectors.
    Revising the same family without seed_functions preserves historical seed claims;
    provide seed_functions explicitly only to refresh the checked seed baseline.
    no_coherent_family: evidence, boundary_notes, assessment and structured decision
    (outcome, reason, next_action; proposed shared_relation, observable_conditions,
    support_item_ids and negative_control_ids when investigating a supported relation).
    A supported but insufficient_definition remains unresolved until operationalized
    and tested. implementation_failed needs a saved smoke experiment_id/source_sha256;
    classifier_limited needs a completed classifier comparison_id. Author controls are
    diagnostic only, never causal validation. Incompatible causes or contradicted source
    evidence can justify rejection without forcing an implementation. Missing evidence
    remains unresolved; use defer_family for an already recorded family. Historical
    unstructured decisions remain readable, but new no_coherent_family calls require
    the structured decision contract. In every case,
    inspect the decisive trace or artifact evidence before ruling out a mechanism.
    Short excerpts and the traced agent's explanation are leads, not grounds for rejection.
    If decisive evidence is unavailable, use defer_family only for a recorded family
    awaiting assessment; otherwise report_status saves the lead, gap and next step.
    Do not invent support to record a family or treat missing evidence as refutation.
    records one bounded search attempt. With seed_functions, cite current seed support
    and explain why the inspected causes cannot coherently share a mechanism; this
    satisfies a scheduled synthesis inspection without inventing a parent. Without
    seeds, inspect new failed tasks while comparison budget remains.
    assess_family: family_name and assessment of saved evaluation/feedback, including
    false positives, missing evidence and retained coverage. No score threshold is
    required. Compare up to 1/3/5 families in Precision/Balanced/Recall each round;
    optionally pass detector_name to select an existing differently named detector.
    Its description/mechanism and current evaluated source must match the recorded claim.
    Neither assessment nor status runs a detector or judge.
    defer_family: family_name, blocker, next_step. Preserve a blocked family as open
    and investigate other causes without claiming the family was assessed or disproven.
    """
    from autofa.mcp.runtime import err, get_runtime, ok
    from autofa.session.polishing_guard import active_polishing, polishing_tool_guard

    rt = get_runtime()
    if blocked := polishing_tool_guard(rt, "family_discovery", {"action": action}):
        return blocked
    if polishing := active_polishing(rt):
        return ok(
            phase="polishing", target_grade="GOOD", frozen_names=polishing["frozen_names"],
            discovery_history=family_progress_summary(rt.run.run_dir()),
            next_action=("Family exploration is closed. Inspect get_function_feedback for "
                         "the admitted detectors; repair, restore, retain or defer them, then "
                         "submit the polishing outcome through "
                         "advance_round(validation_response=...)."),
        )
    try:
        state = initialize_family_discovery(rt)
        required = FAMILY_ACTION_ARGUMENTS.get(action, [])
        supplied = {"family_name": family_name, "description": description, "mechanism": mechanism,
                    "evidence": evidence, "boundary_notes": boundary_notes,
                    "assessment": assessment, "blocker": blocker, "next_step": next_step,
                    "decision": decision}
        missing = [key for key in required
                   if not supplied[key] or (isinstance(supplied[key], str)
                                            and not supplied[key].strip())]
        if missing:
            return err(f"{action} requires: {', '.join(missing)}.",
                       missing_arguments=missing, **family_argument_guidance(action))
        if detector_name and action not in {"record_family", "assess_family"}:
            raise ValueError("detector_name is only used by record_family or assess_family")
        if detector_name and not re.fullmatch(r"[a-z][a-z0-9_]*", detector_name):
            raise ValueError("detector_name must name a snake_case detector")
        if decision is not None and action not in {"no_coherent_family", "defer_family"}:
            raise ValueError("decision is used by no_coherent_family or defer_family")
        if prior_decision_id and action != "record_family":
            raise ValueError("prior_decision_id is only used by record_family")
        if action == "status":
            return ok(**_status(state), **_synthesis_status(rt, state))
        if action in {"record_family", "no_coherent_family"}:
            revising = action == "record_family" and state["phase"] == "assess_family"
            historical_seeds = ((state["survey"].get("seed_functions") or [])
                                if revising and seed_functions is None else [])
            linked_name = (detector_name or
                           (family_detector_name(state["survey"]) if revising else None)
                           or family_name)
            bottom_up = seed_functions is not None or bool(historical_seeds)
            if state.get("synthesis_required") and not bottom_up and not revising:
                raise ValueError("A bottom-up evidence inspection is due before another "
                                 "top-down comparison. Inspect current synthesis_candidates; "
                                 "record_family with seed_functions for a shared mechanism, "
                                 "or no_coherent_family with seed_functions, checked evidence "
                                 "and reasons the mechanisms cannot coherently combine.")
            if (state["phase"] != "survey" and not revising
                    and not (bottom_up and state["phase"] == "specialize")):
                raise ValueError("This round already has a survey; finish its family assessment")
            exhausted = len(state["comparisons"]) >= state["comparison_budget"]
            if exhausted and not revising and not (bottom_up and state.get("synthesis_required")):
                raise ValueError("The comparison budget is exhausted; advance_round before "
                                 "recording another family. A scheduled synthesis inspection "
                                 "may save a proposal for that next round.")
            rows = _checked_evidence(rt, evidence)
            checked = checked_decision(rt, decision, rows, required=action == "no_coherent_family")
            if not any(row["outcome"] is False for row in rows):
                raise ValueError("A failure survey must inspect at least one recorded failed item")
            if not boundary_notes.strip():
                raise ValueError("boundary_notes must describe contrasts and recovery checks")
            supports = [row for row in rows if row["role"] == "support"]
            supporting_tasks = {row["task_id"] for row in supports if row["task_id"]}
            # A repaired parent can retain its historical lineage even if a source
            # was edited or removed. Explicit seed_functions requests a fresh,
            # checked baseline; omission during revision never silently refreshes it.
            seeds = (historical_seeds or _checked_seeds(rt, seed_functions, linked_name, rows)
                     if bottom_up else [])
            if any(seed["name"] == linked_name for seed in seeds):
                raise ValueError("A synthesized detector cannot be its own seed")
            cited_seed_ids = {item_id for seed in seeds for item_id in seed["support_item_ids"]}
            cited_seed_tasks = {row["task_id"] for row in rows
                                if row["item_id"] in cited_seed_ids and row["task_id"]}
            if (bottom_up and state.get("synthesis_required")
                    and len(cited_seed_tasks) < MIN_BROAD_FAMILY_TASKS):
                raise ValueError("The scheduled synthesis inspection needs checked seed "
                                 "support from at least two distinct failed tasks; repeated "
                                 "rollouts of one task do not satisfy it.")
            if not bottom_up:
                _check_survey_breadth(rt, state, rows, revising=revising)
            if action == "record_family":
                if revising and family_name != state["survey"]["family_name"]:
                    raise ValueError("Keep the recorded family_name when revising, or defer the "
                                     "current family before comparing a different hypothesis.")
                if not revising and any((row.get("survey") or {}).get("family_name") == family_name
                                        for row in state["comparisons"]):
                    raise ValueError("This family was already compared; investigate a different "
                                     "hypothesis before further refinement.")
                if not revising and any(
                    (str((row.get("survey") or {}).get("description") or "").strip(),
                     str((row.get("survey") or {}).get("mechanism") or "").strip())
                    == (description.strip(), mechanism.strip())
                    for row in state["comparisons"]
                ):
                    raise ValueError("This claim was already compared under another name; "
                                     "investigate a different hypothesis.")
                if not re.fullmatch(r"[a-z][a-z0-9_]*", family_name):
                    raise ValueError("family_name must be a snake_case hypothesis name")
                if not description.strip() or not mechanism.strip() or not supports:
                    raise ValueError("A family needs description, mechanism and failed support")
                if (len(supporting_tasks) < 2 or not any(
                        row["role"] == "boundary" for row in rows)) and not scope_limit.strip():
                    raise ValueError("Provide cross-task support and a boundary case, or explain "
                                     "the scope_limit; do not invent corpus evidence")
            elif not assessment.strip():
                raise ValueError("Explain why the inspected failures have no coherent family")
            defer_decision = False
            if checked is not None and checked["outcome"] in UNRESOLVED_OUTCOMES:
                from autofa.session.smoke_work import diagnostic_budget

                budget_exhausted = exhausted or diagnostic_budget(rt)["exhausted"]
                defer_decision = checked["outcome"] != "insufficient_definition" or budget_exhausted
                if checked["outcome"] == "insufficient_definition" and budget_exhausted:
                    checked["deferral_basis"] = ("comparison_budget_exhausted" if exhausted
                                                 else "diagnostic_budget_exhausted")
            if (checked is not None and checked["outcome"] in UNRESOLVED_OUTCOMES
                    and not defer_decision):
                record_decision(state, checked, rows, unresolved=True)
                atomic_write_json(rt.run.run_dir() / STATE_FILE, state)
                return err(
                    "This decision leaves the relation unresolved; it does not close a "
                    "family comparison or the scheduled synthesis investigation.",
                    decision_recorded=True,
                    **_status(state), **family_argument_guidance(action),
                )
            if revising:
                state["history"].append({"round": state["round"], "survey": state["survey"],
                                         "outcome": "revised_family_hypothesis"})
            survey = {
                "outcome": action,
                "family_name": family_name if action == "record_family" else None,
                "detector_name": linked_name if action == "record_family" else None,
                "description": description, "mechanism": mechanism, "evidence": rows,
                "boundary_notes": boundary_notes, "scope_limit": scope_limit,
                "approach": "bottom_up" if bottom_up else "top_down",
                "seed_functions": seeds,
            }
            if action == "record_family":
                if decision_id := continue_decision(state, rows, family_name, prior_decision_id):
                    survey["decision_id"] = decision_id
            elif checked is not None:
                survey["decision_id"] = record_decision(
                    state, checked, rows, unresolved=False,
                )["id"]
                survey["decision"] = checked
                if defer_decision:
                    finish_decision(state, survey, "deferred")
            if exhausted and bottom_up and not revising:
                # An evidence checkpoint is not a bonus paid family comparison.
                # Retain its checked proposal and activate it only on a new round.
                _resolve_synthesis_requirement(
                    state, outcome="deferred" if defer_decision else action, seeds=seeds,
                )
                state["history"].append({
                    "round": state["round"], "outcome": "synthesis_inspection_only",
                    "survey": survey, "assessment": assessment,
                    **({"decision": checked, "disposition": "deferred"} if defer_decision else {}),
                })
                if action == "record_family":
                    state["deferred_synthesis"] = survey
                atomic_write_json(rt.run.run_dir() / STATE_FILE, state)
                return ok(**_status(state), **_synthesis_status(rt, state))
            state["survey"] = survey
            state["assessment"] = None
            state["phase"] = "assess_family" if action == "record_family" else "survey"
            if bottom_up:
                _resolve_synthesis_requirement(
                    state, outcome="deferred" if defer_decision else action, seeds=seeds,
                )
            if action == "no_coherent_family":
                state["assessment"] = {"outcome": "deferred" if defer_decision else action,
                                       "notes": assessment,
                                       "decision": checked,
                                       "independent_family_credit": False}
                if defer_decision:
                    state["assessment"].update(blocker=checked["reason"],
                                               next_step=checked["next_action"])
                _complete_comparison(rt, state)
        elif action == "assess_family":
            if state["phase"] != "assess_family":
                raise ValueError("Record a family before assessing it")
            name = (state.get("survey") or {}).get("family_name")
            if family_name != name or not assessment.strip():
                raise ValueError(
                    f"Use family_name={name!r} and an assessment of the saved evaluation. "
                    f"Received family_name={family_name!r}. If that is the detector's name, "
                    "pass it as detector_name instead."
                )
            survey = state["survey"]
            selected_name = detector_name or family_detector_name(survey)
            fn = rt.library.get(selected_name)
            if fn is None:
                candidates = [candidate.name for candidate in rt.library.functions
                              if not _claim_differences(candidate, survey)]
                return err(
                    f"Family {name!r} refers to detector {selected_name!r}, which is not in the "
                    "library. Pass detector_name to assess an existing detector; do not submit "
                    "a duplicate just to match the family name.",
                    matching_detectors=candidates,
                    next_action={"tool": "get_library_function",
                                 "arguments": {"name": candidates[0]}} if candidates else {
                                     "tool": "list_library_functions", "arguments": {}},
                    **family_argument_guidance(action),
                )
            seeds = survey.get("seed_functions") or []
            if any(seed["name"] == fn.name for seed in seeds):
                raise ValueError("A synthesized detector cannot be its own seed")
            provenance = _seed_provenance(rt, seeds) if seeds else None
            if differences := _claim_differences(fn, survey):
                return err(
                    f"Detector {fn.name!r}'s evaluated claim differs from the recorded family "
                    f"{name!r}: {', '.join(differences)}. If this is the intended revised "
                    "hypothesis, inspect its evidence and record_family again with the same "
                    "family_name, this detector_name, and its exact description/mechanism. "
                    "Then assess_family using the existing evaluation; another detector or "
                    "review does not repair the recorded claim.",
                    claim_differences=differences,
                    next_action={"tool": "get_library_function", "arguments": {"name": fn.name}},
                    **family_argument_guidance("record_family"),
                )
            from autofa.core.dependencies import review_source
            from autofa.metrics.claim_contract import claim_hash
            from autofa.metrics.verdicts import behavior_validation, current_behavior_scores

            scores = fn.scores or {}
            fingerprint = claim_hash(source=review_source(fn), description=fn.description or "",
                                     mechanism=fn.mechanism or fn.description or fn.name)
            if (not scores.get("evaluated") or scores.get("evaluation_stale")
                    or scores.get("evaluation_claim_hash") != fingerprint
                    or scores.get("evaluation_run_id") != rt.run.run_id):
                return err(
                    f"Family {name!r}'s detector {fn.name!r} needs a current evaluation in this "
                    "run first.",
                    next_action={"tool": "evaluate_functions", "arguments": {"names": [fn.name]}},
                )
            # Persist the selected link only after every freshness/claim check succeeds.
            survey["detector_name"] = fn.name
            state["assessment"] = {
                "outcome": "assessed",
                "family_name": name, "detector_name": fn.name,
                "claim_hash": fingerprint, "notes": assessment,
                "scores": {key: scores.get(key) for key in (
                    "precision", "ppv_at_pi0", "recall", "behavior_purity", "n_fail_caught",
                    "n_unresolved", "evaluation_scope_complete", "sampled", "judge_prefilter",
                )},
                "review": behavior_validation(current_behavior_scores(
                    scores, source=review_source(fn), description=fn.description or "",
                    mechanism=fn.mechanism or fn.description or fn.name,
                    objective_strategy=discovery_strategy(rt.config),
                    behavior_strictness=strictness_for_config(rt.config),
                ), include_details=False),
            }
            from autofa.session.family_coverage import candidate_coverage

            state["assessment"]["coverage"] = candidate_coverage(
                rt, fn, [(row.get("assessment") or {}).get("coverage")
                         for row in state["comparisons"]
                         if (row.get("assessment") or {}).get("coverage")],
            )
            from autofa.metrics.library_coverage import library_failure_coverage

            inventory = _inventory(rt)
            saved_coverage = library_failure_coverage(
                rt.library, inventory["task_by_failed_item_id"],
                run_id=rt.run.run_id, objective_strategy=discovery_strategy(rt.config),
                behavior_strictness=strictness_for_config(rt.config),
                task_by_item_id=inventory["task_by_item_id"],
            )
            reviewed = next(row for row in saved_coverage["detectors"] if row["name"] == fn.name)
            state["assessment"]["coverage"].update(
                n_review_confirmed_failed_tasks=reviewed["n_review_confirmed_failed_tasks"],
                generalization_status=reviewed["generalization_status"],
            )
            if seeds:
                from autofa.session.family_coverage import seed_union_coverage

                state["assessment"]["seed_provenance"] = provenance
                state["assessment"]["seed_comparison"] = seed_union_coverage(rt, fn, seeds)
                if not provenance["current"]:
                    state["assessment"]["seed_provenance_warning"] = (
                        "Recorded seed support is no longer current for: "
                        + ", ".join(provenance["stale_seed_functions"])
                        + ". Historical seed fingerprints are preserved. The parent's "
                        "independent evaluation remains usable; re-record checked seed "
                        "evidence only if refreshing the synthesis comparison is useful."
                    )
            finish_decision(state, survey, "evaluated")
            _complete_comparison(rt, state)
        elif action == "defer_family":
            if state["phase"] != "assess_family":
                raise ValueError("Only a recorded family awaiting assessment can be deferred")
            survey = state["survey"]
            if family_name != survey["family_name"] or not blocker.strip() or not next_step.strip():
                raise ValueError("Use the recorded family_name and document blocker and next_step")
            from autofa.session.modes_ledger import upsert_mode

            fn = rt.library.get(family_detector_name(survey))
            checked = checked_decision(rt, decision, survey.get("evidence", []))
            if checked is not None and checked["outcome"] not in UNRESOLVED_OUTCOMES:
                raise ValueError("A deferral must preserve an unresolved decision outcome")
            if checked is not None:
                finish_decision(state, survey, "deferred")
                deferred_decision = record_decision(
                    state, checked, survey.get("evidence", []), unresolved=False,
                )
                deferred_decision["family_name"] = family_name
                survey["decision_id"] = deferred_decision["id"]
            upsert_mode(rt.run.run_dir(), name=family_name, description=survey["description"],
                        mechanism=survey["mechanism"], status="open",
                        reason=f"Family assessment deferred: {blocker}. Next: {next_step}",
                        detector=fn.to_dict() if fn else None)
            state["assessment"] = {"outcome": "deferred", "family_name": family_name,
                                   "detector_name": family_detector_name(survey),
                                   "blocker": blocker, "next_step": next_step,
                                   "decision": checked,
                                   "independent_family_credit": False,
                                   "notes": "Family remains open; no completed assessment or "
                                            "causal validation is claimed."}
            finish_decision(state, survey, "deferred")
            _complete_comparison(rt, state)
        else:
            raise ValueError("action must be status, record_family, no_coherent_family, "
                             "assess_family, defer_family")
        _refresh_synthesis_requirement(rt, state)
        atomic_write_json(rt.run.run_dir() / STATE_FILE, state)
        if action == "no_coherent_family":
            from autofa.session.discovery_stall import checkpoint_discovery_stall

            checkpoint_discovery_stall(rt)
        return ok(**_status(state), **_synthesis_status(rt, state))
    except (ValueError, OSError) as exc:
        return err(str(exc), **family_argument_guidance(action))


@family_transaction
def family_submission_guard(
    rt: Any, name: str, *, explains: str | None = None,
    description: str | None = None, mechanism: str | None = None,
    operation: str = "submit", consume_attempt: bool = False,
) -> str | None:
    """Record authoring effort without making family sequencing an edit permission.

    The survey/compare/refine workflow remains in prompts and family_discovery.
    Candidate compilation, claim freshness and ordinary evaluation still apply.
    """
    from autofa.session.polishing_guard import active_polishing, polishing_tool_guard

    if active_polishing(rt) is not None:
        blocked = polishing_tool_guard(rt, "submit_function", {"name": name})
        return blocked["error"] if blocked else None
    if _load(rt.run.run_dir()) is None:
        return None
    state = initialize_family_discovery(rt)
    if operation not in {"submit", "repair"}:
        raise ValueError("Family guard operation must be submit or repair")
    if consume_attempt:
        state["mutation_attempts"][name] = state["mutation_attempts"].get(name, 0) + 1
        _refresh_synthesis_requirement(rt, state)
        atomic_write_json(rt.run.run_dir() / STATE_FILE, state)
    return None


def family_discovery_prompt(run_dir: Path | None = None, *, objective_config=None) -> str:
    """Use either discovery direction without another judge or execution dependency."""
    state = _load(run_dir)
    if objective_config is None:
        from autofa.config import load_config
        from autofa.objectives import config_from_run_dir

        objective_config = (config_from_run_dir(run_dir) if run_dir is not None else None)
        objective_config = objective_config or load_config()
    strategy = discovery_strategy(objective_config)
    target = SURVEY_TARGETS[strategy]
    budget = (state or {}).get("comparison_budget", COMPARISON_BUDGETS[strategy])
    completed = len((state or {}).get("comparisons") or [])
    phase = (state or {}).get("phase", "survey")
    required = (state or {}).get("synthesis_required")
    scheduled = (
        "BOTTOM-UP INSPECTION DUE: finish or defer the current family, then inspect "
        f"supported seeds {required['seed_functions']!r}. Before another top-down family "
        "comparison, use record_family(seed_functions=[...]) for a shared mechanism, or "
        "no_coherent_family(seed_functions=[...]) with checked support and a concrete "
        "explanation of why no coherent abstraction is supported. This consumes existing "
        "search effort; it does not require another model review or a new parent. "
        if required else ""
    )
    if deferred := (state or {}).get("deferred_synthesis"):
        scheduled += (
            f"Saved synthesis {deferred['family_name']!r} awaits the next round's budget. "
            "Call advance_round; it will activate this recorded hypothesis for evaluation "
            "and assessment before another family comparison. "
        )
    elif phase == "assess_family":
        survey = (state or {}).get("survey") or {}
        scheduled += (
            f"Recorded family {survey.get('family_name')!r}, detector "
            f"{family_detector_name(survey)!r}, awaits evaluation and assessment. Finish or "
            "explicitly defer it before a new family. "
        )
    return (
        f"Bidirectional discovery phase: {phase}. "
        f"Family comparisons: {completed}/{budget}. "
        "These next actions apply while discovery continues. In regular discovery, if "
        "pareto_status.goal_met is true, return the summary without starting another family "
        "or round. Honor saved supervisor lifecycle intent. "
        + scheduled +
        "Choose the direction supported by your findings: top-down cross-task surveys can "
        "produce broad families and informative children; bottom-up synthesis can abstract "
        "a shared behavioral mechanism from precise narrow detectors. If broad attempts remain "
        "low-recall while narrow detectors have reviewed support, try synthesis instead of "
        "only tightening or collecting more narrow cases. Call family_discovery(action='status') "
        "for progress, synthesis_candidates and any scheduled synthesis inspection. "
        "After stalled coverage, repeated top-down edits or the bounded top-down search, "
        "current reviewed seeds across tasks require one evidence inspection before another "
        "top-down comparison. The same unchanged seed support is not repeatedly scheduled. "
        "record_family stores inspected "
        "evidence; it does not call another "
        "relationship judge. No separate causal screen is required before evaluation. "
        "Link an existing differently named detector with detector_name in record_family or "
        "assess_family; use its exact description and mechanism. Do not duplicate or re-review "
        "a detector to match the bookkeeping family_name. "
        f"Aim for {target} distinct tasks per top-down survey, capped by available failed tasks. "
        "For bottom-up record_family, pass seed_functions with at least two current "
        "review-supported detectors and cite checked support from each. Reuse inspected "
        "seed evidence; the fresh-task survey requirement does not apply to synthesis. "
        "A scheduled synthesis inspection after the comparison budget is exhausted saves "
        "a checked hypothesis for the next round; it grants no extra comparison or review. "
        "One-task seed "
        "support needs explicit scope_limit; it is a hypothesis, not broad validation. "
        "Implement and evaluate one shared abstract mechanism with concrete evidence for every "
        "claim condition, then follow review_schedule for review under the selected behavioral "
        "standard or actionable deferral. Missing adherence alone does not require standard "
        "review. "
        "A union of unrelated detectors or task-name cases is not that abstraction. "
        "Compare the new parent with the union of its seed matches, inspect lost supported "
        "cases, and seek new failed tasks beyond that union while preserving purity. "
        "Keep the narrow source detectors and do not gate the parent with depends_on or "
        "explains: it must be able to find new manifestations beyond those seeds. "
        "Task identity comes from the source dataset: multiple repeats of one datapoint "
        "do not establish breadth. Keep useful single-task diagnostics with that scope labeled; "
        "cross-task reviewed support requires current ordinary judge confirmations on at least "
        "two distinct tasks. Compare library_coverage's distinct matched and confirmed tasks "
        "and marginal additional coverage; do not infer confirmation from matches alone. "
        "Record one coherent hypothesis, submit/evaluate it, read review_schedule and any "
        "judge feedback, "
        "then assess_family and compare another top-down hypothesis on new failed tasks while the "
        "bounded comparison budget remains. Family quotas guide search; they never block "
        "saving a detector or making a useful repair. Preserve supported broad parents when "
        "adding children, and inspect lost supported matches before tightening. Semantic "
        "parentage does not require explains/depends_on execution gating. "
        + FAMILY_CONTRIBUTION_GUIDANCE + " " + FAMILY_EVIDENCE_GUIDANCE + " " +
        "No coherent family: record no_coherent_family with exact evidence and reasoning. "
        "A blocker: defer_family with a concrete next_step and retain the candidate as open. "
        "Do not invent families or repeat attempts solely to reach a count. "
        "Broad recall is a search priority, never permission to include cases that contradict "
        "the claim or selected review standard. Standard detector adherence reviews use 10% of "
        "flagged examples, rounded up, with a minimum of three and capped by available "
        "eligible matches. During iterative evaluation this uses observed flagged examples "
        "on the scoring panel; final review uses full-dataset matches. Family comparison "
        "does not impose a separate standard-review cap. An explicit "
        "judge_function(name, review_mode='diagnostic') reviews at most three examples when "
        "claim evidence will guide a repair; it does not produce an adherence score or confirmed "
        "support. Follow review_schedule and grade_gap for the actual limiting dimension. "
        "Coherent subtype specialization may trade coverage for precision when supported; revise "
        "the claim consistently. "
        "Freeze the selected claim before requesting reserved confirmation. Previously "
        "inspected tasks do not become independent validation after a repair or resume.\n"
    )
