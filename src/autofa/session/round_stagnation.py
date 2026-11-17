# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated extra-round patience, separate from investigation/pivot accounting.

The closing refinement exchange owns this checkpoint. Tool traffic and provisional
scores cannot consume it. Historical useful versions remain known after deletion,
so restoring or renaming them cannot continually buy more discovery rounds.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
from autofa.metrics.final_validation_selection import current_final_validation_skip
from autofa.metrics.library_quality_coverage import library_quality_coverage_for_runtime
from autofa.metrics.search_association import search_association
from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS
from autofa.metrics.verdicts import QUALITY_THRESHOLDS
from autofa.objectives import discovery_strategy
from autofa.reports.ordering import rank_display_functions
from autofa.session.discovery_target import _detector_identities
from autofa.util.atomic import atomic_write_json

_LOCK = threading.RLock()
_USEFUL = {"GOOD", "EXCELLENT", "OUTSTANDING", "OK"}
_GOOD = _USEFUL - {"OK"}
# Sensitivity for progress accounting only: these do not change detector grades.
_RATE_GAIN = .01
_EPSILON = 1e-9
_PROGRESS_ACCOUNTING_VERSION = 3


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _number(value: Any) -> float | None:
    return (float(value) if isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 else None)


def _load(rt: Any) -> dict:
    try:
        state = json.loads((rt.run.run_dir() / "discovery_round_stagnation.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict) or state.get("version") != 1 \
            or state.get("run_id") != rt.run.run_id:
        return {}
    return state


def _policy(rt: Any, *, max_rounds: int | None = None) -> str:
    return _hash([strictness_for_config(rt.config), discovery_strategy(rt.config),
                  QUALITY_THRESHOLDS, JUDGE_CONTRACT_VERSION,
                  rt.run.max_rounds if max_rounds is None else max_rounds,
                  "substantive_polishing_recovery_v2"])


def migrate_round_policy(rt: Any, *, old_max_rounds: int,
                         supersede_capacity: bool = False) -> None:
    """An authorized live goal edit preserves measured progress and consumed patience."""
    with _LOCK:
        state = _load(rt)
        if not state:
            return
        previous_policy = _policy(rt, max_rounds=old_max_rounds)
        if state.get("policy") != previous_policy:
            return  # Never bless an unrelated stale scientific policy.
        state["policy"] = _policy(rt)
        if old_max_rounds != rt.run.max_rounds:
            from autofa.session.family_coverage import cached_task_inventory

            corpus = cached_task_inventory(rt).get("corpus_fingerprint")
            if corpus and state.get("comparison_key") == _hash([corpus, previous_policy]):
                state["comparison_key"] = _hash([corpus, state["policy"]])
            state.setdefault("goal_revisions", []).append({
                "old_max_rounds": old_max_rounds, "max_rounds": rt.run.max_rounds,
                "round": rt.run.round, "consumed_patience": state.get("rounds_without_improve", 0),
            })
        if supersede_capacity and state.get("review_capacity_block"):
            state["review_capacity_block"]["superseded_by"] = "authorized_goal_update"
        atomic_write_json(rt.run.run_dir() / "discovery_round_stagnation.json", state)


def _review_capacity_fingerprint(rt: Any, block: dict) -> str:
    from autofa.metrics.review_failures import review_failures
    from autofa.session.family_coverage import cached_task_inventory
    from autofa.session.turn_validation import _review_context

    inventory = cached_task_inventory(rt)
    blocked = {row["name"]: row for row in block["detectors"]}
    fields = {"full_corpus": "final_review", "selection_checkpoint": "selection_review",
              "round_checkpoint": "round_review"}
    reviews = sorted((fn.name, {
        "claim": claim_hash(source=review_source(fn), description=fn.description or "",
                            mechanism=fn.mechanism or fn.description or fn.name),
        "evaluation": {key: (fn.scores or {}).get(key) for key in (
            "evaluation_claim_hash", "evaluation_context", "evaluation_stale",
            "task_recall_corpus_fingerprint", "behavior_judgment_stale", "behavior_evidence_stale",
        )},
        "failures": [(row.get("item_id"), row.get("kind")) for row in review_failures(
            (fn.scores or {}).get("behavior_verdicts") or [])],
        "receipt": {key: ((fn.scores or {}).get(fields[blocked[fn.name]["review_scope"]])
                          or {}).get(key)
                    for key in ("claim_hash", "scope", "status", "identity", "item_ids")},
    }) for fn in rt.library.functions if fn.name in blocked)
    # Repair-plan chatter alone neither resolves a failed review nor grants calls;
    # do not load the much larger repair ledger for this operational stop receipt.
    return _hash([getattr(rt.run, "min_failure_modes", None), inventory.get("corpus_fingerprint"),
                  sorted(fn.name for fn in rt.library.functions), _review_context(rt), reviews])


def _review_capacity_block(rt: Any, validation: dict, snapshot: dict) -> dict | None:
    """Stop an impossible frozen target, without turning unfinished reviews into grades."""
    refinement = validation.get("refinement") or {}
    if refinement.get("mode") != "polishing":
        return None
    frozen = set(refinement.get("frozen_names") or [])
    pending = set(snapshot.get("pending_detector_names") or [])
    if not frozen or not pending:
        return None
    from autofa.mcp.tools_eval import _review_settings, _review_settings_match
    from autofa.metrics.review_failures import review_failures

    _, _, settings = _review_settings(rt.config)
    blocked = []
    for fn in rt.library.functions:
        if fn.name not in frozen & pending:
            continue
        scores = fn.scores or {}
        current_claim = claim_hash(
            source=review_source(fn), description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
        )
        if (scores.get("evaluation_claim_hash") != current_claim
                or scores.get("evaluation_stale") or scores.get("behavior_judgment_stale")
                or scores.get("behavior_evidence_stale")
                or not _review_settings_match(scores, settings)):
            continue
        # A blocked selection/round review can prevent reaching the final review
        # altogether. Keep its actual scope; it does not certify a full-corpus result.
        operation = scores.get("behavior_review_operation") or {}
        receipts = [receipt for field, scope in (
            ("final_review", "full_corpus"), ("selection_review", "selection_checkpoint"),
            ("round_review", "round_checkpoint"),
        ) if (receipt := scores.get(field) or {}).get("claim_hash") == current_claim
            and receipt.get("scope") == scope and receipt.get("identity")
            and receipt.get("status") == "unavailable"
            and (not operation.get("operation_id")
                 or receipt.get("operation_id") == operation["operation_id"])]
        failures = review_failures(scores.get("behavior_verdicts") or [])
        for receipt in receipts:
            exhausted = [row for row in failures if row.get("kind") == "review_budget_exhausted"
                         and row.get("item_id") in (receipt.get("item_ids") or [])]
            if exhausted:
                blocked.append({"name": fn.name, "kind": "review_budget_exhausted",
                                "review_scope": receipt["scope"],
                                "review_identity": receipt["identity"],
                                "item_ids": sorted({str(row["item_id"]) for row in exhausted})})
                break
    if not blocked:
        return None
    minimum = getattr(rt.run, "min_failure_modes", 0)
    possible = len(frozen & {fn.name for fn in rt.library.functions}) - len(blocked)
    if possible >= minimum:
        return None  # Other frozen candidates may still satisfy the requested target.
    reason = (
        "Polishing is blocked by exhausted review capacity for "
        + ", ".join(row["name"] for row in blocked)
        + ". Required review remains incomplete within its saved allowance; "
        "these detectors are incomplete, not rejected. "
        "The remaining frozen candidates cannot meet the requested minimum. "
        "Preserve the claims, cohort, completed judgments and "
        "consumed allowances. Inspect the saved review workload, budget and evidence "
        "before arranging a supported recovery; unchanged Resume cannot replenish this budget."
    )
    block = {"status": "blocked", "stop_reason": "review_capacity_exhausted",
             "round": rt.run.round, "detectors": blocked, "min_failure_modes": minimum,
             "maximum_unblocked_candidates": possible, "reason": reason}
    block["evidence_fingerprint"] = _review_capacity_fingerprint(rt, block)
    return block


def round_stagnation_status(rt: Any) -> dict:
    """Read a saved round decision without treating an observation as a new round."""
    state = _load(rt)
    from autofa.session.detector_service import infrastructure_block

    blocked = infrastructure_block(rt)
    if state and state.get("policy") != _policy(rt):
        state = {}
    patience = max(1, int(getattr(rt.config, "discovery_stall_rounds", 3)))
    count = state.get("rounds_without_improve", 0)
    enabled = bool(getattr(rt.config, "discovery_terminate_if_stalled", True))
    reason = (f"{count} completed extra recovery rounds without validated library improvement"
              if count >= patience else None)
    from autofa.session.polishing_progress import recovery_state_fingerprint

    exhaustion_current = (not state.get("recovery_exhausted")
                          or state.get("recovery_evidence_fingerprint")
                          == recovery_state_fingerprint(rt))
    exhausted = bool(state.get("recovery_exhausted") and exhaustion_current)
    if state.get("recovery_exhausted") and not exhaustion_current:
        count, reason = 0, None
    if exhausted:
        reason = ((state.get("last_assessment", {}).get("recovery_attempt", {}).get(
            "limitations", {}).get("library_capacity") or {}).get("reason")
                  or "Every remaining below-Good detector has a current evidence-backed deferral "
                  "with a tested recovery investigation or a recorded unavailable-source finding. "
                  "The recorded polishing actions are exhausted; the requested minimum "
                  "may remain unmet.")
    capacity = state.get("review_capacity_block")
    if capacity and capacity.get("superseded_by"):
        capacity = None
    if (capacity and capacity.get("evidence_fingerprint")
            != _review_capacity_fingerprint(rt, capacity)):
        capacity = None
    if capacity:
        reason = capacity["reason"]
    ordinary_stop = enabled and (count >= patience or exhausted)
    campaign = bool(getattr(rt.run, "campaign_policy", None))
    should_stop = bool(capacity) or ordinary_stop and not campaign
    return {
        "infrastructure_blocked": bool(blocked), "service": blocked,
        "unit": "completed_extra_rounds", "patience": patience,
        "rounds_without_improve": count,
        "completed_extra_rounds": state.get("completed_extra_rounds", 0),
        "initialized": bool(state.get("comparison_key")),
        "last_completed_round": state.get("last_completed_round"),
        "should_stop": should_stop,
        "stop_reason": ("review_capacity_exhausted" if capacity else
                        "stalled" if should_stop else None),
        "review_capacity_block": capacity,
        "termination_deferred_by_campaign": bool(campaign and ordinary_stop and not capacity),
        "recovery_exhausted": exhausted,
        "reason": reason,
        "last_assessment": state.get("last_assessment"),
    }


def _closed_review(scores: dict, fingerprint: str, *, unresolved: bool = False) -> bool:
    """Unavailable evidence conclusions are finished; unavailable operations are not."""
    receipt = scores.get("final_review") or {}
    if (receipt.get("claim_hash") != fingerprint or receipt.get("scope") != "full_corpus"
            or receipt.get("evidence_unavailable")):
        return False
    if not scores.get("n_flagged"):
        # The finalizer cannot compute adherence when a complete scan honestly
        # abstained on every item. No positive match needs a judge in this case;
        # the outer census/error checks still require every attempt accounted for.
        all_abstained = (scores.get("n_scored") == 0
                         and (_number(scores.get("n_insufficient")) or 0) > 0)
        return (receipt.get("status") == "not_required"
                or (all_abstained and receipt.get("status") == "unavailable"
                    and not receipt.get("identity")))
    if receipt.get("status") not in {"complete", "unavailable"} or not receipt.get("identity"):
        return False
    # A busy/failed finalizer can leave a previous complete receipt untouched.
    if unresolved and receipt.get("status") == "complete":
        return False
    ids, rows = receipt.get("item_ids"), scores.get("behavior_verdicts")
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
            or not isinstance(rows, list) or len(rows) != len(ids)
            or {row.get("item_id") for row in rows if isinstance(row, dict)} != set(ids)):
        return False
    contract = scores.get("claim_contract") or {}
    contract_id = contract.get("contract_id")
    if not contract_id or contract.get("claim_hash", contract_id) != fingerprint:
        return False
    return all(isinstance(row, dict) and row.get("claim_contract_id") == contract_id
               and row.get("verdict") in {"yes", "no", "uncertain"}
               and row.get("processing_status") == "completed"
               and row.get("completion_status") in {"complete", "needs_evidence_review"}
               and not row.get("validation_error") for row in rows)


def _snapshot(rt: Any, validation: dict) -> dict:
    """Compare fully checked keepers after every candidate has a current disposition."""
    # Display coverage includes useful OK detectors; quality patience continues
    # to compare the Good-or-better union used by the discovery target.
    union = library_quality_coverage_for_runtime(rt, minimum_grade="GOOD")
    corpus = union.get("corpus_fingerprint")
    if not corpus:
        return {"available": False, "reason": "Source corpus identity is unavailable."}
    from autofa.session.family_coverage import cached_task_inventory

    inventory = cached_task_inventory(rt)
    if inventory.get("n_unknown_outcomes"):
        return {"available": False, "reason": "Source outcome inventory is incomplete."}
    unresolved = {row.get("name") for row in validation.get("review_unresolved", [])
                  if isinstance(row, dict)}
    selection_unresolved = {
        row.get("name")
        for row in (validation.get("final_validation_selection") or {}).get("unresolved", [])
        if isinstance(row, dict)
    }
    rows = rank_display_functions(
        [vars(fn) for fn in rt.library.functions],
        behavior_strictness=strictness_for_config(rt.config),
        objective_strategy=discovery_strategy(rt.config),
        review_history=rt.library.review_history,
    )
    views = {row["name"]: row for row in rows}
    useful, pending = [], []
    from autofa.metrics.discovery_grades import grade_for_metrics

    for fn in rt.library.functions:
        if fn.name in selection_unresolved:
            # An unavailable selection prerequisite is unfinished work even if
            # an earlier review concluded that its evidence was inconclusive.
            pending.append(fn.name)
            continue
        if current_final_validation_skip(rt, fn) is not None:
            # A bounded evaluation can explicitly decline further work on a
            # non-keeper. That finishes the operation, not its corpus evidence:
            # it contributes neither a useful detector nor measured progress.
            continue
        scores, metrics = views[fn.name]["scores"], views[fn.name]["metrics"]
        fingerprint = claim_hash(
            source=review_source(fn), description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
        )
        resolved, abstained = (_number(scores.get(key, 0))
                                for key in ("n_scored", "n_insufficient"))
        full_attempts = (resolved is not None and abstained is not None
                         and resolved + abstained == inventory["n_items"]
                         and _number(scores.get("n_items")) == resolved
                         and _number(scores.get("n_unresolved", abstained)) == abstained)
        if full_attempts and inventory["n_items"]:
            expected = resolved / inventory["n_items"]
            measured = _number(scores.get("assessment_coverage", expected))
            full_attempts = measured is not None and math.isclose(measured, expected)
        if (scores.get("evaluated") is not True or not full_attempts
                or scores.get("evaluation_run_id") != rt.run.run_id
                or scores.get("evaluation_claim_hash") != fingerprint
                or scores.get("evaluation_stale") or scores.get("sampled") is not False
                or scores.get("evaluation_scope_complete") is not True
                or scores.get("eval_truncated")
                or scores.get("task_recall_corpus_fingerprint") != corpus
                or (not abstained and (scores.get("evaluation_complete") is False
                                      or scores.get("dataset_evaluation_complete") is False))
                or any(scores.get(key) for key in (
                    "errors", "n_errors", "parent_errors", "behavior_n_errors",
                )) or not _closed_review(scores, fingerprint, unresolved=fn.name in unresolved)):
            pending.append(fn.name)
            continue
        grade = grade_for_metrics(metrics)["code"]
        if grade not in _USEFUL:
            continue
        vector = {key: _number(metrics.get(key)) for key in (
            "recall", "ppv_at_pi0", "behavior_purity", "n_matched_failed_tasks",
        )}
        # Honest inconclusive reviews can complete a round, but cannot manufacture
        # a supported improvement. Missing task identity similarly gives no breadth gain.
        if any(vector[key] is None for key in ("recall", "ppv_at_pi0", "behavior_purity")):
            continue
        target_metrics = dict(vector)
        if abstained:
            conservative = search_association(scores)
            # Within-parent association has a different population. Do not mix
            # its rates with corpus bounds, or guess missing abstention bounds.
            if conservative is None or scores.get("within_parent") or scores.get("explains"):
                target_metrics = None
            else:
                target_metrics.update({key: _number(conservative.get(key))
                                       for key in ("recall", "ppv_at_pi0")})
        claim, predicate = _detector_identities(fn)
        useful.append({"name": fn.name, "claim": _hash(claim), "predicate": _hash(predicate),
                       "grade": grade, "metrics": vector, "comparable_rates": not abstained,
                       "target_metrics": target_metrics,
                       "target_metric_basis": ("worst_case_abstentions" if abstained
                                               else "full_corpus")})
    if pending:
        return {"available": False,
                "reason": "Final validation has unfinished evaluations or reviews.",
                "pending_detector_names": pending}
    # Exclude identical claims or predicates using the same rule as target accounting.
    distinct, claims, predicates = [], set(), set()
    for row in sorted(useful, key=lambda row: row["grade"] not in _GOOD):
        if row["claim"] in claims or row["predicate"] in predicates:
            continue
        claims.add(row["claim"])
        predicates.add(row["predicate"])
        distinct.append(row)
    return {
        "available": True, "comparison_key": _hash([corpus, _policy(rt)]),
        "detectors": distinct,
        "quality": {"good": sum(row["grade"] in _GOOD for row in distinct),
                    "good_or_ok": len(distinct)},
        "union": ({key: union.get(key) for key in (
            "comparison_key", "n_matched_failed_attempts", "n_matched_failed_tasks",
            "coverage", "task_coverage",
        )} if union.get("status") == "exact" else None),
    }


def _related(left: dict, right: dict) -> bool:
    return any(left[key] == right[key] for key in ("name", "claim", "predicate"))


def _dominates(current: dict, previous: dict, *, meaningful: bool = False) -> bool:
    """Require retained qualifying dimensions, not a scalar tradeoff."""
    before, after = previous["metrics"], current["metrics"]
    if not current.get("comparable_rates", True) or not previous.get("comparable_rates", True):
        # A fully attempted scan can honestly abstain. It completes a round, but
        # rates on a changing resolved subset are not comparable measurements.
        return before == after if not meaningful else False
    if any(before[key] is not None and (after[key] is None or after[key] < before[key] - _EPSILON)
           for key in before):
        return False
    if not meaningful:
        return True
    return any(
        before[key] is not None and after[key] is not None
        and after[key] - before[key] >= (
            1 if key == "n_matched_failed_tasks" else _RATE_GAIN) - _EPSILON
        for key in before
    )


def _toward_good(current: dict, previous: dict, *, meaningful: bool = False) -> bool:
    """Compare remaining Good requirements without rewarding surplus tradeoffs.

    Already qualified dimensions may narrow but must stay qualified. Blocking
    dimensions cannot regress. This records validated repair, not a population
    rate gain; abstaining scans use worst-case bounds on the full attempted corpus.
    """
    before, after = previous.get("target_metrics"), current.get("target_metrics")
    if not before or not after:
        return False
    requirements = {key: (value["good"], value["strict"])
                    for key, value in QUALITY_THRESHOLDS.items()}
    requirements["n_matched_failed_tasks"] = (MIN_GOOD_FAILED_TASKS, False)
    gained = False
    for key, (threshold, strict) in requirements.items():
        old, new = before.get(key), after.get(key)
        if old is None or new is None:
            return False
        old_met = old > threshold if strict else old >= threshold
        new_met = new > threshold if strict else new >= threshold
        if old_met:
            if not new_met:
                return False
        else:
            if new < old - _EPSILON:
                return False
            gain = 1 if key == "n_matched_failed_tasks" else _RATE_GAIN
            gained |= new_met or new - old >= gain - _EPSILON
    return gained if meaningful else True


def _target_repair(row: dict, history: list[dict]) -> dict | None:
    if not history or row["grade"] in _GOOD:
        return None
    # Missing historical bounds cannot establish a positive gain. Compare raw
    # rates only with raw rates to veto known restorations; mixing old resolved
    # rates with new conservative bounds would prevent later supported repairs.
    legacy = [old for old in history if not old.get("target_metrics")]
    comparable = [old for old in history if old.get("target_metrics")]
    if (any(old["grade"] in _GOOD for old in history)
            or any(_toward_good({"target_metrics": old["metrics"]},
                               {"target_metrics": row["metrics"]}) for old in legacy)
            or not any(_toward_good(row, old, meaningful=True) for old in history)):
        return None
    # Compare with each dimension's historical high-water mark. Comparing only
    # with the oldest version lets tiny gains repeatedly reuse a prior big repair.
    high_water = {
        key: max((old["target_metrics"].get(key) for old in comparable
                  if old["target_metrics"].get(key) is not None), default=None)
        for key in row["target_metrics"]
    }
    if not _toward_good(row, {"target_metrics": high_water}, meaningful=True):
        return None
    dimensions = []
    for key, old in high_water.items():
        threshold, strict = ((MIN_GOOD_FAILED_TASKS, False) if key == "n_matched_failed_tasks"
                             else (QUALITY_THRESHOLDS[key]["good"],
                                   QUALITY_THRESHOLDS[key]["strict"]))
        old_met = old > threshold if strict else old >= threshold
        if old_met:
            continue
        new = row["target_metrics"][key]
        reached = new > threshold if strict else new >= threshold
        gain = 1 if key == "n_matched_failed_tasks" else _RATE_GAIN
        if reached or new - old >= gain - _EPSILON:
            dimensions.append(key)
    return {"name": row["name"], "target": "GOOD", "dimensions": dimensions,
            "basis": row["target_metric_basis"], "previous_high_water": high_water,
            "current": row["target_metrics"]}


def _progress(state: dict, snapshot: dict) -> list[str]:
    known = state["known_detectors"]
    signals = []
    # Dropping useful functions cannot hide a concurrent library-quality regression.
    preserved = all(snapshot["quality"][key] >= state["previous_quality"][key]
                    for key in ("good", "good_or_ok"))
    for row in snapshot["detectors"]:
        history = [previous for previous in known if _related(previous, row)]
        duplicate = any(previous["claim"] == row["claim"]
                        or previous["predicate"] == row["predicate"] for previous in known)
        if not history and not duplicate and preserved:
            signals.append(f"new_useful_detector:{row['name']}")
        elif (preserved and history and row["grade"] in _GOOD
              and not any(old["grade"] in _GOOD for old in history)):
            # Fully validated qualification is a categorical gain. Abstentions
            # still prohibit numeric rate comparisons, but cannot erase the
            # first Good result; a restored/renamed historical Good is excluded.
            signals.append(f"retained_detector_improvement:{row['name']}")
        elif (preserved and row["grade"] in _GOOD and history
              and not any(_dominates(old, row) for old in history)
              and any(_dominates(row, old, meaningful=True)
                      or (row["grade"] in _GOOD and old["grade"] not in _GOOD
                          and _dominates(row, old)) for old in history)):
            signals.append(f"retained_detector_improvement:{row['name']}")
        elif preserved and (repair := _target_repair(row, history)):
            signals.append(f"validated_good_target_repair:{row['name']}")
            snapshot.setdefault("target_repairs", []).append(repair)
        # Keep incomparable historical versions; an old restored version is known
        # even after a failed revision or removal. Identical renames add no history.
        if not any(previous["claim"] == row["claim"]
                   and previous["predicate"] == row["predicate"]
                   and previous["grade"] == row["grade"]
                   and previous["metrics"] == row["metrics"]
                   and previous.get("comparable_rates") == row.get("comparable_rates")
                   and previous.get("target_metrics") == row.get("target_metrics")
                   for previous in known):
            known.append(row)
    union, best = snapshot["union"], state.get("best_union")
    if union:
        if (preserved and best and union["comparison_key"] == best["comparison_key"]
                and union["n_matched_failed_attempts"] > best["n_matched_failed_attempts"]):
            signals.append("good_combined_coverage")
        if (not best or union["comparison_key"] != best["comparison_key"]
                or union["n_matched_failed_attempts"] > best["n_matched_failed_attempts"]):
            state["best_union"] = union
    state["previous_quality"] = snapshot["quality"]
    return signals


def complete_discovery_round(rt: Any, validation: dict) -> dict:
    """Checkpoint only a fully validated closing exchange, before incrementing round.

    The last planned round establishes a baseline. Legacy runs first establish one
    at their next completed full-validation round; historical attempts are never
    converted to exhausted round patience. Repeated calls recover the same receipt.
    """
    with _LOCK:
        round_idx, planned = rt.run.round, rt.run.max_rounds
        status = round_stagnation_status(rt)
        if status.get("infrastructure_blocked"):
            return {**status, "counted": False, "status": "infrastructure_blocked",
                    "available": False,
                    "reason": "Unavailable semantic experiments are unfinished; "
                              "existing patience and validated history are preserved."}
        if (validation.get("ok") is not True or validation.get("fraction") != 1.0
                or round_idx < planned - 1):
            return {**status, "counted": False, "status": "not_full_validation_boundary"}
        state = _load(rt)
        policy = _policy(rt)
        if state.get("policy") != policy:
            state = {}
        from autofa.session.polishing_progress import recovery_state_fingerprint

        if (state.get("recovery_exhausted") and state.get("recovery_evidence_fingerprint")
                != recovery_state_fingerprint(rt)):
            state["rounds_without_improve"] = 0
            state["recovery_exhausted"] = False
        snapshot = _snapshot(rt, validation)
        if not snapshot["available"]:
            capacity = _review_capacity_block(rt, validation, snapshot)
            if capacity:
                # Operational exhaustion is not a completed no-improvement round.
                # Keep the existing quality history and every review untouched.
                state.update(version=1, run_id=rt.run.run_id, policy=policy,
                             review_capacity_block=capacity)
                atomic_write_json(rt.run.run_dir() / "discovery_round_stagnation.json", state)
                return {**round_stagnation_status(rt), "counted": False,
                        "status": "review_capacity_exhausted", "available": False,
                        "pending_detector_names": snapshot.get("pending_detector_names", [])}
            # An outage must not erase the previous durable progress checkpoint.
            return {**status, "counted": False, "status": "validation_incomplete", **snapshot}
        state.pop("review_capacity_block", None)
        baseline = state.get("comparison_key") != snapshot["comparison_key"]
        if not baseline and str(round_idx) in state.get("rounds", {}):
            return {**status, **state["rounds"][str(round_idx)],
                    "rounds_without_improve": state["rounds_without_improve"], "reused": True}
        if baseline:
            state = {"version": 1, "run_id": rt.run.run_id, "policy": policy,
                     "progress_accounting_version": _PROGRESS_ACCOUNTING_VERSION,
                     "comparison_key": snapshot["comparison_key"], "rounds": {},
                     "rounds_without_improve": 0, "completed_extra_rounds": 0,
                     "known_detectors": snapshot["detectors"], "best_union": snapshot["union"],
                     "previous_quality": snapshot["quality"]}
            signals = []
        else:
            signals = _progress(state, snapshot)
        refinement = validation.get("refinement") or {}
        attempt = refinement.get("recovery_attempt") or {}
        consumed = set(state.get("completed_recovery_work", []))
        work_ids = set(attempt.get("work_ids") or [])
        fresh_work = work_ids - consumed
        substantive = (refinement.get("mode") != "polishing"
                       or (attempt.get("substantive") is True and bool(fresh_work))
                       or bool(signals))
        counted = not baseline and round_idx >= planned and substantive
        if counted:
            state["completed_extra_rounds"] += 1
            state["rounds_without_improve"] = (
                0 if signals else state["rounds_without_improve"] + 1)
        state["recovery_exhausted"] = bool(
            round_idx >= planned and refinement.get("mode") == "polishing"
            and attempt.get("exhausted") and attempt.get("limitations")
        )
        state["recovery_evidence_fingerprint"] = recovery_state_fingerprint(rt)
        state["completed_recovery_work"] = sorted(consumed | work_ids)
        receipt = {"round": round_idx, "counted": counted,
                   "status": ("baseline" if baseline else
                              "improved" if signals else
                              "no_improvement" if substantive else "no_new_recovery_work"),
                   "improved": bool(signals), "progress_signals": signals,
                   "quality": snapshot["quality"],
                   "detectors": snapshot["detectors"],
                   "target_repairs": snapshot.get("target_repairs", []),
                   "rounds_without_improve": state["rounds_without_improve"],
                   "recovery_attempt": attempt}
        state["last_completed_round"] = round_idx
        state["last_assessment"] = receipt
        state["rounds"][str(round_idx)] = receipt
        atomic_write_json(rt.run.run_dir() / "discovery_round_stagnation.json", state)
        return {**round_stagnation_status(rt), **receipt}
