# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover the final repair exchange from interrupted final validations and stale contexts."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.session.polishing import current_claims, prepare_polishing
from autofa.session.recovery import read_json
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import cancellation_guard


def _timestamp(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _interrupted_boundary(rt: Any, receipt: dict) -> dict | None:
    """Require an automatic full scan interrupted before its polishing checkpoint.

    A cheap detector's ordinary full scan is not a final-validation transition.
    A later unfinished advance_round also has no authority without its runtime
    full-validation event. Read these local receipts; never infer from labels.
    """
    if receipt.get("checkpoint_kind") == "campaign_intermediate":
        return None
    if not receipt and (rt.run.run_dir() / "round_validation.json").exists():
        return None
    pending = []
    for path in (rt.run.run_dir() / "operations").glob("*.json"):
        operation = read_json(path)
        started = _timestamp(operation.get("started_at"))
        if (operation.get("tool") == "advance_round"
                and operation.get("state") in {"running", "interrupted", "cancelled"}
                and started is not None):
            pending.append((started, path.name))
    if not pending:
        return None
    started, operation = max(pending)
    try:
        with (rt.run.run_dir() / "status_log.jsonl").open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                context = event.get("context") or {}
                when = _timestamp(event.get("ts"))
                if (event.get("source") == "round_validation"
                        and context.get("fraction") == 1.0
                        and context.get("phase") == "scoring"
                        and when is not None and when >= started):
                    return {"kind": "interrupted_full_checkpoint", "operation": operation,
                            "started_at": started, "round": rt.run.round}
    except OSError:
        pass
    return None


def _historical_baseline(rt: Any, boundary: dict, corpus: str) -> tuple[dict, dict, list[str]]:
    """Recover measurements, never fabricate a missing grade or cohort identity."""
    from autofa.metrics.claude_judge import judge_policy_hash

    claims = current_claims(rt)
    policy = judge_policy_hash(behavior_strictness=strictness_for_config(rt.config))
    before, grades, admitted = {}, [], []
    with rt._scores_history_lock:
        history = deepcopy(rt.scores_history)
    for name, claim in claims.items():
        records = []
        for entry in history:
            if not isinstance(entry, dict) or entry.get("evaluated_claims", {}).get(name) != claim:
                continue
            if ("round" in boundary and (type(entry.get("round")) is not int
                                         or entry["round"] > boundary["round"])):
                continue
            if "claims" in boundary and boundary["claims"].get(name) != claim:
                continue
            for row in entry.get("results") or []:
                if row.get("name") != name:
                    continue
                scores = row.get("scores") or {}
                panel = scores.get("search_panel") or {}
                if (panel.get("run_id") != rt.run.run_id or panel.get("claim_hash") != claim
                        or panel.get("corpus_fingerprint") != corpus):
                    continue
                records.append((entry, row))
        # Recover the checkpoint immediately before this revision's first census,
        # not its best-ever score and not measurements from a different revision.
        first_full = next((i for i, (entry, _) in enumerate(records)
                           if entry.get("eval_mode") == "full"), len(records))
        candidates = [row for entry, row in records[:first_full]
                      if entry.get("eval_mode") in {"capped", "round_validation"}
                      and (boundary.get("started_at") is None or
                           isinstance(entry.get("wall_clock"), (int, float))
                           and entry["wall_clock"] <= boundary["started_at"])]
        if not candidates:
            grades.append({"name": name, "code": None,
                           "reason": "No verified pre-validation scores were saved."})
            continue
        row = candidates[-1]
        scores = deepcopy(row["scores"])
        before[name] = scores
        grade = (row.get("discovery_grade") or row.get("verdict") or {}).get("code")
        review = scores.get("behavior_validation") or {}
        policy_matches = (review.get("judge_policy_hash") == policy
                          and review.get("judge_behavior_strictness") == strictness_for_config(
                              rt.config))
        grades.append({"name": name, "code": grade, "historical": True,
                       "policy_matches": policy_matches,
                       "reason": ("Saved pre-validation grade and policy match."
                                  if policy_matches else
                                  "Saved measurements do not establish admission under the "
                                  "current judge policy; no earlier qualification is assumed.")})
        if policy_matches and grade in {"GOOD", "EXCELLENT", "OUTSTANDING", "OK"}:
            admitted.append(name)
    return before, {"functions": grades}, admitted


def recover_interrupted_validation(rt: Any) -> dict | None:
    """Recover once under the runner's exclusive run lock, without model calls.

    Live scores, the round cursor, review allocations and prediction caches remain
    untouched. Recovery is persisted before any expensive work so interruption
    cannot reopen discovery or lose the original downgrade comparison.
    """
    path = rt.run.run_dir() / "round_validation.json"
    receipt = read_json(path)
    state = receipt.get("refinement") or {}
    if isinstance(state, dict) and state.get("mode") == "polishing":
        return _recover_polishing_context(rt, receipt)
    if getattr(rt.run, "campaign_policy", None):
        # Campaign full checkpoints are intermediate; an interrupted one is never final
        # validation. The ordinary max_rounds hint says nothing about their terminal phase.
        return None
    if (not isinstance(state, dict) or state.get("mode")
            or not rt.library.functions or rt.run.round < max(0, rt.run.max_rounds - 1)):
        return None
    boundary = _interrupted_boundary(rt, receipt)
    if boundary is None:
        return None
    from autofa.mcp.tools_eval import _evaluation_context
    from autofa.metrics.claude_judge import judge_policy_hash
    from autofa.session.family_coverage import cached_task_inventory

    inventory = cached_task_inventory(rt)
    context = _evaluation_context(rt)
    if not context or not inventory.get("inventory_complete") or not inventory.get(
        "corpus_fingerprint"
    ):
        raise RuntimeError("Cannot recover final validation until the saved corpus is available.")
    previous_scores, baseline, admitted = _historical_baseline(
        rt, boundary, inventory["corpus_fingerprint"])
    claims = current_claims(rt)
    # Legacy evaluation hashes included process-random set ordering. Verify the
    # corpus independently here; finalization refreshes mismatched aggregates
    # through the evaluator and its per-item cache before restoring admission.
    # Never relabel an old opaque hash as a verified current context.
    matching = [fn.name for fn in rt.library.functions
                if (fn.scores or {}).get("task_recall_corpus_fingerprint")
                == inventory["corpus_fingerprint"]]
    if not matching:
        raise RuntimeError("Cannot recover final validation: saved scores belong to a "
                           "different or unverified corpus. Restore its source first.")
    state = prepare_polishing(rt, before=baseline, previous_scores=previous_scores)
    state["legacy_recovery"] = {
        "version": 1, "validation_pending": True, "boundary": boundary,
        "previous_scores": previous_scores, "admitted_names": admitted,
        "claims": claims, "run_id": rt.run.run_id, "source_signature": context,
        "policy_hash": judge_policy_hash(behavior_strictness=strictness_for_config(rt.config)),
    }
    migrated = {"ok": True, "round": rt.run.round, "fraction": 1.0,
                "refinement": state,
                "message": "Recovered interrupted final validation. Finish pending validation "
                           "using saved predictions, then polish the existing detectors."}
    with cancellation_guard():
        backup = path.with_name("round_validation.before_polishing_migration.json")
        if not backup.exists():
            atomic_write_json(backup, receipt)
        atomic_write_json(path, migrated)
    return state


def _recover_polishing_context(rt: Any, receipt: dict) -> dict | None:
    """Finish a saved modern checkpoint before its first resumed repair turn."""
    from autofa.mcp.tools_eval import _evaluation_context
    from autofa.metrics.claude_judge import judge_policy_hash
    from autofa.session.family_coverage import cached_task_inventory
    from autofa.session.polishing_guard import active_polishing

    phase = active_polishing(rt)
    if phase is None or (phase.get("legacy_recovery") or {}).get("validation_pending"):
        return None
    context = _evaluation_context(rt)
    candidates = [fn for fn in rt.library.functions
                  if fn.name in phase["frozen_names"] and (fn.scores or {}).get("evaluated")
                  and (fn.scores or {}).get("evaluation_context")
                  and (fn.scores or {}).get("evaluation_context") != context]
    if not candidates:
        return None
    inventory = cached_task_inventory(rt)
    corpus = inventory.get("corpus_fingerprint")
    if (not context or not inventory.get("inventory_complete") or not corpus
            or any((fn.scores or {}).get("task_recall_corpus_fingerprint") != corpus
                   for fn in candidates)):
        raise RuntimeError("Cannot recover polishing: saved scores belong to a different "
                           "or unverified corpus. Restore its source first.")
    boundary = {"kind": "polishing_context_refresh", "round": phase["round"],
                "claims": dict(phase.get("evaluated_claims") or {})}
    previous_scores, baseline, admitted = _historical_baseline(rt, boundary, corpus)
    claims = current_claims(rt)
    strictness = strictness_for_config(rt.config)
    policy = judge_policy_hash(behavior_strictness=strictness)
    completed = set()
    for fn in candidates:
        selection = (fn.scores or {}).get("final_validation_selection") or {}
        if (selection.get("version") != 1 or selection.get("run_id") != rt.run.run_id
                or selection.get("claim_hash") != claims[fn.name]
                or selection.get("source_signature") != fn.scores.get("evaluation_context")
                or selection.get("policy_hash") != policy
                or selection.get("judge_behavior_strictness") != strictness):
            continue
        if selection.get("status") == "selected":
            admitted.append(fn.name)
        elif selection.get("status") == "complete":
            completed.add(fn.name)
    admitted = list(dict.fromkeys(name for name in admitted
                                 if name in phase["frozen_names"] and name not in completed))
    # Retain historical controls separately. Their old signature is never
    # rewritten into a current executable control receipt.
    state = deepcopy(receipt["refinement"])
    state.setdefault("baseline_grades", baseline)
    state["legacy_recovery"] = {
        "version": 1, "validation_pending": True, "boundary": boundary,
        "previous_scores": previous_scores, "admitted_names": admitted,
        "claims": claims, "run_id": rt.run.run_id, "source_signature": context,
        "policy_hash": policy, "historical_controls": deepcopy(state.get("controls") or {}),
    }
    path = rt.run.run_dir() / "round_validation.json"
    with cancellation_guard():
        backup = path.with_name("round_validation.before_context_recovery.json")
        if not backup.exists():
            atomic_write_json(backup, receipt)
        atomic_write_json(path, {**receipt, "refinement": state})
    return state


def pending_recovered_validation(rt: Any) -> dict | None:
    """Return the saved recovery receipt while its final validation is still pending."""
    from autofa.session.polishing_guard import active_polishing

    phase = active_polishing(rt)
    recovery = (phase or {}).get("legacy_recovery") or {}
    return recovery if recovery.get("validation_pending") else None


def prepare_recovered_validation(rt: Any) -> None:
    """Restore admission once; normal final-review receipts govern completion."""
    from autofa.mcp.tools_eval import _evaluation_context
    from autofa.metrics.claude_judge import judge_policy_hash
    from autofa.metrics.final_validation_selection import recover_final_validation_selection

    recovery = pending_recovered_validation(rt)
    if recovery:
        if (recovery["run_id"] != rt.run.run_id
                or recovery["source_signature"] != _evaluation_context(rt)
                or recovery["policy_hash"] != judge_policy_hash(
                    behavior_strictness=strictness_for_config(rt.config))
                or recovery["claims"] != current_claims(rt)):
            raise RuntimeError("Saved final-validation recovery no longer matches the run, "
                               "corpus/settings, judge policy or detector revisions. "
                               "Restore the saved inputs before resuming this checkpoint.")
        result = recover_final_validation_selection(
            rt, run_id=recovery["run_id"], names=recovery["admitted_names"],
            claims=recovery["claims"], source_signature=recovery["source_signature"],
            policy_hash=recovery["policy_hash"],
        )
        if result["unresolved"]:
            raise RuntimeError("Cannot resume saved validation: " + "; ".join(
                f"{row['name']}: {row['reason']}" for row in result["unresolved"]))


def complete_recovered_validation(rt: Any) -> None:
    """Rebuild all regression controls before handing the resumed run to the agent."""
    from autofa.session.round_validation import refresh_validated_target, validation_feedback

    path = rt.run.run_dir() / "round_validation.json"
    receipt = read_json(path)
    phase = receipt.get("refinement") or {}
    recovery = pending_recovered_validation(rt)
    if recovery is None:
        return
    unfinished = []
    for fn in rt.library.functions:
        scores = fn.scores or {}
        selection = scores.get("final_validation_selection") or {}
        if selection.get("status") not in {"selected", "reviewing"}:
            continue
        review = scores.get("final_review" if selection["status"] == "selected"
                            else "selection_review") or {}
        if review.get("status") == "deferred":
            # A deliberate scheduling deferral is repair feedback, not a failed
            # provider call. Keep its receipt and let polishing address the gap.
            continue
        unfinished.append(f"{fn.name}: {review.get('reason') or 'review has not completed'}")
    if unfinished:
        raise RuntimeError("Final validation recovery is still pending. Saved predictions and "
                           "review checkpoints are retained; resume to finish: "
                           + "; ".join(unfinished))
    previous = deepcopy(phase)
    historical = recovery.get("historical_controls") or {}
    for name, controls in historical.items():
        if controls.get("source_signature") != recovery["source_signature"]:
            previous.get("controls", {}).pop(name, None)
    state = prepare_polishing(
        rt, before=phase["baseline_grades"], previous_scores=recovery["previous_scores"],
        previous=previous,
    )
    if historical:
        from autofa.session.validation_controls import collect_full_validation_controls

        for fn in rt.library.functions:
            old = historical.get(fn.name) or {}
            if not old.get("available") or fn.name not in state["frozen_names"]:
                continue
            collected = collect_full_validation_controls(
                rt, fn, previous_scores=recovery["previous_scores"].get(fn.name),
                historical_controls=old,
            )
            if collected.get("available"):
                state["controls"][fn.name] = collected
    if recovery.get("boundary", {}).get("kind") == "polishing_context_refresh":
        # Context repair does not consume or reset the saved polishing exchange.
        for key in ("status", "polishing_id", "round", "evaluated_claims"):
            if key in phase:
                state[key] = phase[key]
    state["legacy_recovery"] = {**recovery, "validation_pending": False}
    with cancellation_guard():
        atomic_write_json(path, {**receipt, "refinement": state})
        refresh_validated_target(rt)
        atomic_write_json(path, {**receipt, **validation_feedback(rt, state["baseline_grades"]),
                                 "refinement": state})
