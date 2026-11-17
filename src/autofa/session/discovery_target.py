# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count evidence-backed failure modes for the discovery stopping condition."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Sequence
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
from autofa.metrics.judge_prefilter import judge_prefilter
from autofa.metrics.library_coverage import library_failure_coverage
from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS
from autofa.metrics.verdicts import (
    QUALITY_THRESHOLDS,
    behavior_validation,
    current_behavior_scores,
    judge_prefilter_skipped,
    metric_tier,
)
from autofa.objectives import discovery_strategy

if TYPE_CHECKING:
    from autofa.mcp.runtime import Runtime


class DiscoveryProgress:
    """Detect repeated idle turns for recovery without requiring successful candidates."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.previous = self._snapshot()
        self.idle_turns = 0

    def _snapshot(self) -> tuple:
        rt = self.runtime
        library = hashlib.sha256(json.dumps(
            rt.library.to_dict(), sort_keys=True, default=str,
        ).encode()).hexdigest()
        # These checkpoints also capture rejected candidates and repair work that
        # does not change the surviving library. Avoid rereading large ledgers.
        checkpoints = []
        for name in ("modes_ledger.jsonl", "detector_repairs.json", "scores_history.json",
                     "family_discovery.json"):
            try:
                stat = (rt.run.run_dir() / name).stat()
                checkpoints.append((stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                checkpoints.append(None)
        # Status logs, transcripts and library-save timestamps are not progress.
        return (library, rt.run.round, len(rt.scores_history), *checkpoints)

    def stalled(self) -> bool:
        current = self._snapshot()
        self.idle_turns = self.idle_turns + 1 if current == self.previous else 0
        self.previous = current
        if self.idle_turns < 2:
            return False
        # Give recovery guidance once per pair of unchanged turns.
        self.idle_turns = 0
        return True


def _predicate_identity(source: str) -> str:
    """Ignore comments/formatting and a renamed entry point when counting modes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source.strip()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.name = "detector"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


MIN_STANDARDIZED_PRECISION = QUALITY_THRESHOLDS["ppv_at_pi0"]["good"]
MIN_BEHAVIOR_PURITY = QUALITY_THRESHOLDS["behavior_purity"]["good"]
MIN_MATCHED_DATAPOINTS = MIN_GOOD_FAILED_TASKS


def _detector_identities(fn: FeatureFunction) -> tuple[tuple[str, str], tuple[str, str, str]]:
    """Share duplicate handling between discovery grades and final validation."""
    description = fn.description or ""
    mechanism = fn.mechanism or description or fn.name
    claim = (" ".join(description.casefold().split()), " ".join(mechanism.casefold().split()))
    meta = fn.meta or {}
    predicate = (_predicate_identity(review_source(fn)),
                 json.dumps(meta.get("deps") or {}, sort_keys=True),
                 json.dumps(meta.get("dependency_bundle") or {}, sort_keys=True))
    return claim, predicate


def _number(value: Any, *, count: bool = False) -> float | int | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value < 0):
        return None
    if count:
        return int(value) if int(value) == value else None
    return float(value) if value <= 1 else None


def _corpus_association(scores: dict, inventory: dict) -> dict | None:
    """Use whole-corpus denominators; abstentions are never inferred matches.

    A complete scan can have honest abstentions. Its positive findings remain
    measurable, while execution errors, unvisited rows and stale/sampled results
    cannot qualify. This does not change evaluation or ranking completeness.
    """
    cells = [_number(scores.get(key), count=True) for key in ("a", "b", "c", "d")]
    if any(value is None for value in cells):
        return None
    a, b, c, d = cells
    n = inventory["n_items"]
    n_fail, n_pass = inventory["n_failed_attempts"], inventory["n_passed_attempts"]
    abstained = _number(scores.get("n_insufficient", 0), count=True)
    unresolved = _number(scores.get("n_unresolved", abstained), count=True)
    if (abstained is None or unresolved != abstained or not n_fail or not n_pass
            or n_fail < a + c or n_pass < b + d
            or sum(cells) + abstained != n
            or _number(scores.get("n_items"), count=True) != sum(cells)
            or _number(scores.get("n_scored", scores.get("n_items")), count=True) != sum(cells)
            or _number(scores.get("n_fail_caught", a), count=True) != a
            or _number(scores.get("n_flagged"), count=True) != a + b
            or scores.get("evaluation_scope_complete") is not True
            or scores.get("sampled") is not False or scores.get("eval_truncated")
            or scores.get("dataset_evaluation_complete") is False
            or any(_number(scores.get(key, 0), count=True) != 0 for key in (
                "errors", "n_errors", "parent_errors",
            ))
            or (not abstained and scores.get("evaluation_complete") is False)):
        return None
    assessment = scores.get("assessment_coverage")
    if assessment is not None and (_number(assessment) is None
                                  or not isclose(assessment, sum(cells) / n)):
        return None
    return contingency(a, b, n_fail - a, n_pass - b)


def _completed_review(scores: dict, fingerprint: str) -> bool:
    """A decisive current review can support discovery even on development data."""
    contract = scores.get("claim_contract") or {}
    rows = scores.get("behavior_verdicts") or []
    keys = ("behavior_n_judged", "behavior_n_completed", "behavior_n_yes", "behavior_n_no",
            "behavior_n_uncertain", "behavior_n_errors")
    counts = [_number(scores.get(key), count=True) for key in keys]
    if any(value is None for value in counts):
        return False
    total, completed, yes, no, uncertain, errors = counts
    if (judge_prefilter_skipped(scores) or not total or completed != total
            or yes + no != total or uncertain or errors
            or _number(scores.get("behavior_n_review_incomplete", 0), count=True) != 0
            or scores.get("judge_contract_version") != JUDGE_CONTRACT_VERSION
            or scores.get("claim_hash") != fingerprint or scores.get("behavior_judgment_stale")
            or not isinstance(contract, dict) or not contract.get("contract_id")
            or contract.get("claim_hash", contract["contract_id"]) != fingerprint
            or not isinstance(rows, list) or len(rows) != total):
        return False
    sample = scores.get("behavior_sample") or {}
    if (not isinstance(sample, dict) or ("actual_n" in sample
            and _number(sample["actual_n"], count=True) != total)):
        return False
    ids, observed_yes = set(), 0
    for row in rows:
        if not isinstance(row, dict):
            return False
        item_id = row.get("item_id")
        if (not isinstance(item_id, str) or not item_id or item_id in ids
                or row.get("verdict") not in {"yes", "no"}
                or row.get("completion_status") != "complete" or row.get("validation_error")
                or row.get("claim_contract_id") != contract["contract_id"]
                or not row.get("snapshot_id")):
            return False
        ids.add(item_id)
        if row["verdict"] == "yes":
            observed_yes += 1
            initial = row.get("initial_review") or {}
            if (row.get("verification_status") != "confirmed" or not isinstance(initial, dict)
                    or initial.get("verdict") != "yes" or initial.get("item_id") != item_id
                    or initial.get("claim_contract_id") != contract["contract_id"]
                    or initial.get("snapshot_id") != row["snapshot_id"]):
                return False
    purity = _number(scores.get("behavior_purity"))
    return observed_yes == yes and purity is not None and isclose(purity, yes / total)


def failure_mode_target(
    functions: Sequence[Any], *, minimum: int, config: Any, run_id: str | None = None,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Count distinct Good detectors supported by a full scan and task evidence.

    Source metadata establishes different datapoints. Previously inspected,
    reused or development examples may establish causal support; independent
    held-out validation remains separate and its ranking rules are unchanged.
    """
    if type(minimum) is not int or minimum < 0:
        raise ValueError("min_failure_modes must be a nonnegative integer")
    recall_floor = QUALITY_THRESHOLDS["recall"]["good"]
    thresholds = {
        "min_failure_recall": recall_floor,
        "min_failure_recall_exclusive": QUALITY_THRESHOLDS["recall"]["strict"],
        "min_standardized_precision": MIN_STANDARDIZED_PRECISION,
        "min_behavior_purity": MIN_BEHAVIOR_PURITY,
        "min_matched_failed_datapoints": MIN_MATCHED_DATAPOINTS,
    }
    functions = [FeatureFunction.from_dict(fn) if isinstance(fn, dict) else fn for fn in functions]
    inventory_ok = bool(
        isinstance(inventory, dict) and inventory.get("inventory_complete") is True
        and _number(inventory.get("n_unknown_outcomes"), count=True) == 0
        and isinstance(inventory.get("task_by_failed_item_id"), dict)
        and isinstance(inventory.get("task_by_item_id"), dict)
        and _number(inventory.get("n_items"), count=True) == len(inventory["task_by_item_id"])
        and inventory["n_items"] > 0
        and _number(inventory.get("n_failed_attempts"), count=True)
        == len(inventory["task_by_failed_item_id"])
        and _number(inventory.get("n_passed_attempts"), count=True)
        == inventory["n_items"] - len(inventory["task_by_failed_item_id"])
        and inventory["task_by_failed_item_id"].keys() <= inventory["task_by_item_id"].keys()
    )
    coverage = {}
    if inventory_ok:
        failed = inventory["task_by_failed_item_id"]
        passed = set(inventory["task_by_item_id"]) - failed.keys()
        for fn in functions:
            try:
                # Include passes here to verify both positive contingency cells.
                # Causal breadth below still intersects failures only.
                details = library_failure_coverage(
                    FunctionLibrary(functions=[fn]), inventory["task_by_item_id"],
                    run_id=run_id, objective_strategy=discovery_strategy(config),
                    behavior_strictness=strictness_for_config(config),
                    include_item_ids=True, task_by_item_id=failed,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            row = details["detectors"][0]
            matches = set(row["known_matched_failed_item_ids"])
            confirmed = set(row["review_confirmed_failed_item_ids"]) & failed.keys()
            coverage[fn.name] = {
                "known_positive_item_ids": matches,
                "n_known_matched_failed": len(matches & failed.keys()),
                "n_known_matched_failed_tasks": len({failed[item_id]
                                                    for item_id in matches & failed.keys()
                                                    if failed[item_id] is not None}),
                "n_known_matched_passed": len(matches & passed),
                "n_review_confirmed_failed": len(confirmed),
                "n_review_confirmed_failed_tasks": len({failed[item_id] for item_id in confirmed
                                                       if failed[item_id] is not None}),
            }
    names, excluded, eligibility = [], [], []
    seen_claims: set[tuple[str, str]] = set()
    seen_predicates: set[tuple[str, str, str]] = set()
    for fn in functions:
        data, name = vars(fn), fn.name
        source = review_source(data)
        description = data.get("description") or ""
        mechanism = data.get("mechanism") or description or name
        scores = current_behavior_scores(
            data.get("scores"), source=source, description=description, mechanism=mechanism,
            objective_strategy=discovery_strategy(config),
            behavior_strictness=strictness_for_config(config),
        )
        fingerprint = claim_hash(source=source, description=description, mechanism=mechanism)
        matched = coverage.get(name, {})
        association = _corpus_association(scores, inventory) if inventory_ok else None
        current = (scores.get("evaluated") is True and not scores.get("evaluation_stale")
                   and scores.get("evaluation_claim_hash") == fingerprint
                   and (run_id is None or scores.get("evaluation_run_id") == run_id))
        reviewed = _completed_review(scores, fingerprint)
        if reviewed:
            reviewed = {row["item_id"] for row in scores["behavior_verdicts"]} <= matched.get(
                "known_positive_item_ids", set(),
            )
        measured = {
            "failure_recall": association["recall"] if association and current else None,
            "standardized_precision": (association["ppv_at_pi0"]
                                       if association and current else None),
            "behavior_purity": scores.get("behavior_purity") if reviewed else None,
            "n_reviewed": scores.get("behavior_n_judged") if reviewed else None,
            "n_confirmed_failed_datapoints": matched.get("n_review_confirmed_failed_tasks", 0),
            "n_matched_failed_datapoints": matched.get("n_known_matched_failed_tasks", 0),
            "n_confirmed_failed_attempts": matched.get("n_review_confirmed_failed", 0),
            "n_matched_failed": matched.get("n_known_matched_failed", 0),
            "n_failed_in_corpus": inventory["n_failed_attempts"] if inventory_ok else None,
            "n_abstained": scores.get("n_insufficient", 0),
            "recall_scope": "whole_corpus_observed_matches",
            "independent_validation": behavior_validation(scores, include_details=False)["status"]
            if reviewed else "unavailable",
        }
        reason = None
        if not inventory_ok:
            reason = "Needs source-verified corpus outcomes and datapoint identities."
        elif not current or association is None:
            reason = ("Needs a current complete corpus scan without execution errors "
                      "or unvisited rows.")
        elif (matched.get("n_known_matched_failed") != association["a"]
              or matched.get("n_known_matched_passed") != association["b"]):
            reason = ("Saved positive evidence does not account for the evaluated "
                      "failed and passed matches.")
        elif metric_tier("recall", association["recall"]) != 2:
            reason = (f"Whole-corpus Coverage is {association['recall']:.2%}; Good requires "
                      f"Coverage >{recall_floor:.2%}.")
        elif metric_tier("ppv_at_pi0", association["ppv_at_pi0"]) != 2:
            reason = ("Normalized precision is below the Good threshold of "
                      f"{MIN_STANDARDIZED_PRECISION:.0%}.")
        else:
            # Run the existing rejection policy on whole-corpus observed rates.
            # These local copies do not overwrite unresolved or ranking metadata.
            prefilter_scores = {**scores, **association, "n_scored": inventory["n_items"],
                                "evaluation_complete": True,
                                "n_insufficient": 0, "assessment_coverage": 1}
            explains = data.get("explains") or (data.get("meta") or {}).get("explains")
            if explains:
                prefilter_scores["explains"] = explains
            decision = judge_prefilter(prefilter_scores)
            if decision["status"] != "pass":
                reason = decision["reason"]
            elif not reviewed:
                reason = ("Needs a current completed claim review with consistent "
                          "per-example verdicts.")
            elif metric_tier("behavior_purity", scores["behavior_purity"]) != 2:
                reason = ("Detector adherence is below the Good threshold of "
                          f"{MIN_BEHAVIOR_PURITY:.0%}.")
            elif matched["n_known_matched_failed_tasks"] < MIN_MATCHED_DATAPOINTS:
                reason = (f"Needs detector matches on at least {MIN_GOOD_FAILED_TASKS} "
                          "different failed datapoints.")

        claim, predicate = _detector_identities(fn)
        if reason is None and (claim in seen_claims or predicate in seen_predicates):
            reason = "Duplicate of an already counted claim or detector."
        eligibility.append({"name": name, "eligible": reason is None, "reason": reason,
                            "measured": measured})
        if reason:
            excluded.append({"name": name, "reason": reason, "measured": measured})
            continue
        names.append(name)
        seen_claims.add(claim)
        seen_predicates.add(predicate)
    return {
        "min_failure_modes": minimum,
        "n_qualifying_failure_modes": len(names), "qualifying_failure_modes": names,
        # Preserve stored/tool consumers while labeling the new semantics explicitly.
        "n_validated_failure_modes": len(names), "validated_failure_modes": names,
        "minimum_met": len(names) >= minimum,
        "remaining_failure_modes": max(0, minimum - len(names)),
        "pending_failure_modes": excluded, "failure_mode_eligibility": eligibility,
        "failure_mode_thresholds": thresholds,
    }


def discovery_good_target(
    functions: Sequence[Any], *, minimum: int, config: Any,
    review_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Count current public Good grades for discovery, including sampled estimates.

    The same read view and classifier drive CLI/report labels. Full-corpus
    validation rechecks these same grade requirements before discovery finishes.
    This helper neither scans source data nor mutates saved evaluation evidence.
    """
    from autofa.metrics.verdicts import verdict_for_metrics
    from autofa.reports.ordering import rank_display_functions

    if type(minimum) is not int or minimum < 0:
        raise ValueError("min_failure_modes must be a nonnegative integer")
    functions = [FeatureFunction.from_dict(fn) if isinstance(fn, dict) else fn for fn in functions]
    rows = rank_display_functions(
        [vars(fn) for fn in functions],
        behavior_strictness=strictness_for_config(config),
        review_history=review_history,
    )
    seen_claims, seen_predicates = set(), set()
    names, eligibility = [], []
    for row in rows:
        metrics, scores = row["metrics"], row["scores"]
        verdict = verdict_for_metrics(metrics)
        claim, predicate = _detector_identities(FeatureFunction.from_dict(row))
        reason = None if verdict["code"] == "GOOD" else verdict["reason"]
        if reason is None and (claim in seen_claims or predicate in seen_predicates):
            reason = "Duplicate of an already counted claim or detector."
        if reason is None:
            names.append(row["name"])
            seen_claims.add(claim)
            seen_predicates.add(predicate)
        measured = {
            "failure_recall": metrics.get("recall"),
            "standardized_precision": metrics.get("ppv_at_pi0"),
            "behavior_purity": metrics.get("behavior_purity"),
            "n_reviewed": scores.get("behavior_n_judged"),
            "n_confirmed_failed_datapoints": metrics.get("n_confirmed_failed_tasks"),
            "n_matched_failed_datapoints": metrics.get("n_matched_failed_tasks"),
            "n_matched_failed": metrics.get("n_matched_failed_attempts"),
            "n_failed_in_corpus": scores.get("n_failed_attempts"),
            "n_abstained": scores.get("n_insufficient", 0),
            "sampled": bool(scores.get("sampled")),
            "recall_scope": "sampled_estimate" if scores.get("sampled") else "observed",
            "corpus_evaluation_complete": bool(
                scores.get("evaluated") is True and not scores.get("evaluation_stale")
                and scores.get("evaluation_scope_complete") is True
                and scores.get("sampled") is False and not scores.get("eval_truncated")
                and not any(scores.get(key, 0) for key in (
                    "errors", "n_errors", "parent_errors",
                ))
            ),
            **{key: metrics.get(key) for key in (
                "estimated_n_confirmed_failed_tasks", "estimated_n_matched_failed_tasks",
                "task_support_comparison",
                "task_support_comparison_basis", "task_support_estimate_status",
            )},
        }
        eligibility.append({"name": row["name"], "eligible": reason is None,
                            "reason": reason, "measured": measured})
    return {
        "min_failure_modes": minimum,
        "n_qualifying_failure_modes": len(names), "qualifying_failure_modes": names,
        "n_validated_failure_modes": len(names), "validated_failure_modes": names,
        "minimum_met": len(names) >= minimum,
        "remaining_failure_modes": max(0, minimum - len(names)),
        "pending_failure_modes": [row for row in eligibility if not row["eligible"]],
        "failure_mode_eligibility": eligibility,
        "failure_mode_thresholds": {
            "min_failure_recall": QUALITY_THRESHOLDS["recall"]["good"],
            "min_failure_recall_exclusive": QUALITY_THRESHOLDS["recall"]["strict"],
            "min_standardized_precision": MIN_STANDARDIZED_PRECISION,
            "min_behavior_purity": MIN_BEHAVIOR_PURITY,
            "min_matched_failed_datapoints": MIN_MATCHED_DATAPOINTS,
            "task_support_scope": "estimated_when_sampled_else_observed",
        },
    }


def discovery_good_target_for_runtime(rt: Runtime, *, minimum: int | None = None) -> dict[str, Any]:
    """Read the visible Good minimum without source scans or runtime writes."""
    return discovery_good_target(
        rt.library.functions, minimum=rt.run.min_failure_modes if minimum is None else minimum,
        config=rt.config, review_history=rt.library.review_history,
    )


def failure_mode_target_for_runtime(rt: Runtime, *, minimum: int | None = None) -> dict[str, Any]:
    """Use the runtime's cached identity-only inventory; never load trajectory text."""
    from autofa.metrics.final_validation_selection import current_final_validation_skip
    from autofa.session.family_coverage import cached_task_inventory

    minimum = rt.run.min_failure_modes if minimum is None else minimum
    inventory, error = None, None
    if rt.library.functions:
        try:
            inventory = cached_task_inventory(rt)
        except Exception as exc:
            error = f"Source metadata unavailable: {type(exc).__name__}: {exc}"
    result = failure_mode_target(rt.library.functions, minimum=minimum, config=rt.config,
                                 run_id=rt.run.run_id, inventory=inventory)
    skipped = {
        fn.name: receipt
        for fn in rt.library.functions
        if (receipt := current_final_validation_skip(rt, fn)) is not None
    }
    for row in result["failure_mode_eligibility"] + result["pending_failure_modes"]:
        if row.get("eligible"):
            continue
        receipt = skipped.get(row["name"])
        if receipt is not None:
            row["reason"] = (
                receipt["reason"]
                + " Retain the current evidence; improve or resolve the candidate "
                "before requesting another full scan. This does not qualify as Good."
            )
    if error:
        result["inventory_error"] = error
    return result


def agent_target_grade_for_runtime(rt: Runtime) -> str | None:
    """Read the current agent aspiration without scans, state changes or stop decisions."""
    try:
        from autofa.session.polishing_guard import active_polishing

        if active_polishing(rt) is not None:
            return "GOOD"
        # Reusing visible grades avoids loading or populating a source inventory.
        target = discovery_good_target_for_runtime(rt)
        return agent_failure_mode_target(
            target, round_idx=rt.run.round, max_rounds=rt.run.max_rounds,
        )["target_grade"]
    except Exception:  # noqa: BLE001 - display advice must not interrupt actual work
        return None


def discovery_grade_feedback(
    rt: Any, *, target: dict[str, Any] | None = None,
    task_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic turn assessment, using current grades without another model call."""
    from autofa.metrics.discovery_grades import grade_summary
    from autofa.session.discovery_budget import discovery_budget_feedback
    from autofa.session.polishing_guard import active_polishing
    from autofa.session.round_feedback import round_bulletin

    summary = grade_summary(rt.library.functions,
                            behavior_strictness=strictness_for_config(rt.config),
                            review_history=rt.library.review_history)
    summary["task_breadth"] = _task_breadth_summary(task_coverage)
    previous = getattr(rt, "_previous_discovery_grades", None)
    rt._previous_discovery_grades = summary
    polishing = active_polishing(rt)
    if target is None:
        target = discovery_good_target_for_runtime(rt)
    agent_target = agent_failure_mode_target(
        target, round_idx=rt.run.round, max_rounds=rt.run.max_rounds,
        target_grade="GOOD" if polishing is not None else None,
    )
    result = discovery_grade_assessment(
        summary, previous, minimum=rt.run.min_failure_modes,
        budget_feedback=discovery_budget_feedback(
            rt.run.round, rt.run.max_rounds,
            campaign_policy=getattr(rt.run, "campaign_policy", None),
        ),
        target=agent_target, pending=agent_target["pending_failure_modes"],
        task_coverage=task_coverage,
        round_feedback=(None if polishing is not None else
                        round_bulletin(rt, grades=summary, task_coverage=task_coverage)),
    )
    if polishing is not None:
        counts = summary["counts"]
        count_text = ", ".join(f"{number} {code.title()}" for code, number in counts.items())
        pending = "; ".join(
            f"{row['name']}: {row.get('reason') or 'inspect get_function_feedback'}"
            for row in agent_target["pending_failure_modes"]
        )
        result["phase"] = "polishing"
        result["message"] = (
            f"- Final polishing: library currently has {count_text}.\n"
            f"- Target: {agent_target['n_qualifying_failure_modes']}/"
            f"{rt.run.min_failure_modes} Good or better detectors. Preserve supported "
            "Good, Excellent and Outstanding versions.\n"
            "- Work only on admitted detectors: inspect their current findings and controls, "
            "repair the blocking dimensions, restore a supported saved version, or explain "
            "why the current version should be retained/deferred. The scoped repair budget "
            "is explicit in get_function_repair; history and prior evidence remain preserved.\n"
            + (f"- Remaining qualification requirements: {pending}.\n" if pending else "")
            + "- Submit the tested outcome through advance_round(validation_response=...). "
            "The runtime verifies selected revisions before deciding completion or assigning "
            "another polishing round under the configured stopping policy. User cancellation "
            "and configured stall limits still apply."
        )
    return result


def _task_breadth_summary(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reuse the existing reviewed-task union; never sum detector coverage or run a scan."""
    if not snapshot or snapshot.get("available") is not True:
        return None
    counts = [snapshot.get(key) for key in ("n_confirmed_failed_tasks", "n_failed_tasks")]
    if any(isinstance(n, bool) or not isinstance(n, (int, float))
           or not isfinite(n) or n < 0 or int(n) != n for n in counts):
        return None
    if counts[0] > counts[1]:
        return None
    return {key: snapshot.get(key) for key in (
        "corpus_fingerprint", "behavior_strictness", "quality_thresholds",
        "n_confirmed_failed_tasks", "n_failed_tasks",
    )}


def _round_target_instruction(minimum: int, label: str) -> str:
    """Give the current assignment without forecasting later targets or changing stops."""
    if not minimum:
        return (
            "The minimum detector count is disabled. Improve supported coverage and quality "
            "within the runtime's budget and stopping decision."
        )
    return (
        "Build a library of different supported failure mechanisms and informative subtypes "
        f"across tasks. Aim for {minimum} {label} or better detectors, while "
        "treating Good and Excellent detectors as successful, valuable discoveries. "
        "The active grade is an aspiration, not a retention threshold; assess each "
        "candidate against its own grade's quality and task-support requirements. "
        "Keep these useful findings as you pursue the active target. If the target remains "
        "unmet after completing the investigation batch, pivot and request the next round; "
        "missing the target alone does not complete a round. If the planned round limit is "
        f"reached before you have {minimum} {label} or better detectors, the runtime will "
        "keep assigning discovery work while its stopping policy authorizes continuation, "
        "with no fixed extra-round limit. Running out of planned rounds does not end the run. "
        "When Terminate if stalled is enabled, it can end discovery below the target only "
        "after the configured number of complete, validated recovery rounds without meaningful "
        "library improvement. This counter starts at zero after the planned budget; failed "
        "probes prompt a pivot and cannot terminate planned rounds. "
        "The runtime's stopping decision remains authoritative: follow "
        "ready_for_final_validation when requested and stop when "
        "goal_met is true, even if further grade improvements remain possible. Explicit "
        "cancellation still stops discovery."
    )


def discovery_grade_assessment(
    summary: dict[str, Any], previous: dict[str, Any] | None, *, minimum: int,
    budget_feedback: dict[str, Any] | None = None,
    pending: list[dict[str, Any]] | None = None,
    target: dict[str, Any] | None = None,
    task_coverage: dict[str, Any] | None = None,
    round_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format grade changes and next action without changing runtime or stopping state."""
    from autofa.metrics.discovery_grades import GRADE_CODES

    old = {row["name"]: row for row in (previous or {}).get("functions", [])}
    current = {row["name"]: row for row in summary["functions"]}
    promoted, slipped, added = [], [], []
    for name, row in current.items():
        if name not in old:
            if previous is not None:
                added.append(f"{name}: {row['symbol']} {row['label']}")
            continue
        before = old[name]
        if row["code"] == before["code"]:
            continue
        change = f"{name}: {before['label']} → {row['symbol']} {row['label']}"
        (promoted if GRADE_CODES.index(row["code"]) < GRADE_CODES.index(before["code"])
         else slipped).append(change)
    removed = sorted(set(old) - set(current))
    counts = summary["counts"]
    provisional = [row for row in summary["functions"]
                   if row.get("task_support_comparison_basis") == "estimated"]
    count_text = ", ".join(f"{counts[code]} {code.title()}" for code in GRADE_CODES
                           if counts[code] or code in {"OUTSTANDING", "EXCELLENT", "GOOD"})
    target_label = (target or {}).get("target_grade", "OUTSTANDING").title()
    target_n = (target or {}).get("n_qualifying_failure_modes", summary["n_outstanding"])
    remaining = max(0, minimum - target_n)
    new_useful = sum(current[name]["code"] in {"GOOD", "EXCELLENT", "OUTSTANDING"}
                     for name in current.keys() - old.keys()) if previous is not None else 0
    encouraging = bool(previous is not None and (
        new_useful
        or summary["n_outstanding"] > previous["n_outstanding"]
        or summary["n_good_or_better"] > previous["n_good_or_better"]
        or any(current[name]["code"] in {"GOOD", "EXCELLENT", "OUTSTANDING"}
               and GRADE_CODES.index(current[name]["code"]) < GRADE_CODES.index(old[name]["code"])
               for name in old.keys() & current.keys())
    ))
    assessment = (
        "Baseline assessment; no progress comparison yet." if previous is None else
        "Useful progress: the library gained Good-or-better discoveries or stronger grades."
        if encouraging else "No new Good-or-better discovery or stronger grade this turn."
    )
    next_action = (
        "Preserve useful Good, Excellent and Outstanding detectors and finish requested "
        "evaluation/review."
        if not remaining else
        "Retain useful Good and Excellent findings. Choose the most promising evidence-supported "
        "next step: investigate a distinct mechanism or informative subtype, repair a promising "
        "candidate, or broaden supported detectors (including synthesis of related seeds). "
        "Preserve the supported "
        "version while testing a broader candidate. Preserve precision, adherence and "
        "distinct-task support. Inspect get_function_feedback for the next review or "
        "implementation prerequisite."
    )
    if budget_feedback and budget_feedback["phase"] in {"consolidate", "boundary", "overrun"}:
        next_action = (
            "Prioritize the best current candidates and their remaining requirements below. "
            "Inspect get_function_feedback(name=...) before each repair or review; preserve "
            "useful Good and Excellent findings and valid results. Avoid unchanged retries. "
            "Check pareto_status for the existing "
            "final-validation handoff or stopping decision."
        )
    bullets = [f"- Library currently has: {count_text}.",
               f"- Active aspiration this round: {target_n}/{minimum} "
               f"{target_label} or better; {remaining} still needed.",
               "- " + _round_target_instruction(
                   minimum, target_label,
               )]
    if budget_feedback:
        bullets.insert(0, "- " + budget_feedback["message"])
    if pending:
        pending_by_name = {row["name"]: row for row in pending}
        # Reuse the shared quality/coverage ordering and existing eligibility reasons.
        # These are final qualification gaps, not a demotion of provisional grades.
        priorities = [row for row in summary["functions"] if row["name"] in pending_by_name][:5]
        if priorities:
            details = "; ".join(
                f"{row['name']} [{row['label']}]: "
                + (pending_by_name[row['name']].get("reason") or "Inspect get_function_feedback")
                for row in priorities
            )
            bullets.append("- Remaining requirements for final target (priority order): " + details)
    if provisional:
        details = "; ".join(
            f"{row['name']}: {row['label']}, "
            f"{row['estimated_n_matched_failed_tasks']:.1f} estimated matched failed tasks "
            f"({row.get('n_matched_failed_tasks') or 0} observed)"
            for row in provisional
        )
        bullets.append(
            "- Provisional sampled support: " + details + ". These grades use extrapolated "
            "support; final qualification requires five actually matched distinct failed "
            "tasks after full evaluation.")
    if promoted:
        bullets.append("- Grades increased: " + "; ".join(promoted) + ".")
    if slipped:
        bullets.append("- Grades decreased or evidence became unavailable: "
                       + "; ".join(slipped) + ".")
    if added:
        bullets.append("- New detectors: " + "; ".join(added) + ".")
    if previous is not None:
        bullets.append(f"- New useful detectors since the previous assessment: {new_useful}.")
    if removed:
        bullets.append("- Removed detectors: " + ", ".join(removed) + ".")
    breadth = _task_breadth_summary(task_coverage)
    prior_breadth = (previous or {}).get("task_breadth")
    breadth_delta = None
    if breadth is None:
        breadth_message = (
            "No current confirmed-task union supplied; inspect pareto_status.task_coverage "
            "or library_coverage before claiming additional task coverage."
        )
    else:
        breadth_message = (
            f"{breadth['n_confirmed_failed_tasks']:.0f}/{breadth['n_failed_tasks']:.0f} "
            "distinct failed tasks with quality-supported review confirmations "
            "(observed union, not extrapolated)."
        )
        if (prior_breadth and breadth.get("corpus_fingerprint")
                and breadth.get("behavior_strictness") and breadth.get("quality_thresholds")
                and all(breadth[key] == prior_breadth.get(key) for key in (
                    "corpus_fingerprint", "behavior_strictness", "quality_thresholds",
                    "n_failed_tasks",
                ))):
            breadth_delta = int(breadth["n_confirmed_failed_tasks"]
                                - prior_breadth["n_confirmed_failed_tasks"])
            breadth_message += (
                f" Net change since the previous assessment: {breadth_delta:+d} tasks."
            )
        else:
            breadth_message += " No compatible prior union for a change comparison."
    bullets += [
        f"- Discovery/quality assessment: {assessment}",
        f"- Breadth assessment: {breadth_message}",
        "- Library assessment: group current detectors by claimed mechanism and state what "
        "each new candidate contributes: a different mechanism, informative subtype, "
        "diagnostic use or remedy. For same-family candidates, name the observable "
        "distinction or counterexample addressed; a stricter implementation of the same "
        "claim normally belongs in a revision. Do not claim distinct mechanisms from "
        "different names, grades or match sets alone.",
        "- Overlapping detectors and useful subsets can add supported explanations "
        "without increasing this union. Assess their own quality and task support; no "
        "coverage gain is required to justify an informative subtype. Report additional "
        "confirmed tasks separately from useful discoveries.",
        f"- Next action: {next_action}",
    ]
    if round_feedback is not None:
        # Use one library-level bulletin across status, round transitions and
        # harness turns. Candidate repairs already have their own tool feedback.
        goal_reminder = (
            _round_target_instruction(minimum, target_label)
            if not minimum else
            f"Aim for {minimum} {target_label} or better detectors. Keep useful Good and "
            "Excellent findings while pursuing the active aspiration. Complete the existing "
            "investigation batch before advancing; follow the runtime's goal_met, "
            "ready_for_final_validation and stopping decision."
        )
        bullets = [
            f"- Library currently has: {count_text}.",
            f"- Active aspiration this round: {target_n}/{minimum} "
            f"{target_label} or better; {remaining} still needed.",
            round_feedback["message"],
            "- Sampled grades remain provisional; observed matches, extrapolated support "
            "and reviewed confirmations are different measurements.",
            "- " + goal_reminder,
        ]
        if budget_feedback:
            bullets.insert(0, "- " + budget_feedback["message"])
        encouraging = round_feedback["encouraging"]
    return {"summary": summary, "encouraging": encouraging,
            "task_breadth": breadth, "confirmed_task_net_change": breadth_delta,
            **({"budget_feedback": budget_feedback} if budget_feedback else {}),
            "target_grade": target_label.upper(), "remaining_target": remaining,
            "remaining_outstanding": max(0, minimum - summary["n_outstanding"]),
            **({"round_bulletin": round_feedback} if round_feedback is not None else {}),
            "message": "\n".join(bullets)}


def agent_failure_mode_target(
    target: dict[str, Any], *, round_idx: int = 0, max_rounds: int | None = None,
    target_grade: str | None = None,
) -> dict[str, Any]:
    """Relax round guidance only while the actual Good stopping minimum is unmet."""
    from autofa.metrics.discovery_grades import (
        EXCELLENT_COVERAGE,
        OUTSTANDING_COVERAGE,
        grade_for_metrics,
    )

    graded = []
    for row in target.get("failure_mode_eligibility", []):
        measured = row.get("measured") or {}
        grade = grade_for_metrics({
            "recall": measured.get("failure_recall"),
            "ppv_at_pi0": measured.get("standardized_precision"),
            "behavior_purity": measured.get("behavior_purity"),
            "n_confirmed_failed_tasks": measured.get("n_confirmed_failed_datapoints"),
            "n_matched_failed_tasks": measured.get("n_matched_failed_datapoints"),
            **{key: measured.get(key) for key in (
                "estimated_n_confirmed_failed_tasks", "estimated_n_matched_failed_tasks",
                "task_support_comparison",
                "task_support_comparison_basis", "task_support_estimate_status",
            )},
        })
        graded.append((row, grade))
    minimum = target["min_failure_modes"]
    tiers = (("OUTSTANDING", OUTSTANDING_COVERAGE), ("EXCELLENT", EXCELLENT_COVERAGE),
             ("GOOD", QUALITY_THRESHOLDS["recall"]["good"]))
    target_index = 0
    # The incoming target is the driver's Good eligibility decision. Higher
    # aspirational tiers must never cause a downgrade after this minimum is met.
    if (max_rounds is not None and minimum
            and not target["minimum_met"]):
        current_round = round_idx + 1  # Stored rounds are zero-based, like the CLI's index.
        target_index = 2 if current_round > max_rounds else int(current_round == max_rounds)
    if target_grade is not None:
        if target_grade != "GOOD":
            raise ValueError("The closing polishing target must be GOOD")
        target_index = 2
    target_code, coverage_floor = tiers[target_index]
    target_label = target_code.title()
    accepted = {code for code, _ in tiers[:target_index + 1]}
    rows, names = [], []
    for row, grade in graded:
        eligible = bool(row.get("eligible") and grade["code"] in accepted)
        reason = None if eligible else row.get("reason") or (
            f"Current grade is {grade['label']}; {target_label} requires Coverage "
            f">{coverage_floor:.0%} while preserving precision, adherence and task support."
        )
        rows.append({"name": row["name"], "eligible": eligible, "reason": reason,
                     "grade": grade, "measured": row.get("measured") or {}})
        if eligible:
            names.append(row["name"])
    thresholds = {**target.get("failure_mode_thresholds", {}),
                  "min_failure_recall": coverage_floor,
                  "min_failure_recall_exclusive": True, "target_grade": target_code}
    return {
        "min_failure_modes": minimum, "target_grade": target_code,
        "n_qualifying_failure_modes": len(names), "qualifying_failure_modes": names,
        "n_validated_failure_modes": len(names), "validated_failure_modes": names,
        "minimum_met": len(names) >= minimum,
        "remaining_failure_modes": max(0, minimum - len(names)),
        "pending_failure_modes": [row for row in rows if not row["eligible"]],
        "failure_mode_eligibility": rows, "failure_mode_thresholds": thresholds,
    }


def inactivity_recovery_prompt() -> str:
    """Turn inactivity into an actionable next turn, never a stopping permission."""
    return (
        "Automatic discovery remains active. The last two turns made no recorded discovery "
        "progress. A final reply, a request to change the setup, or exhaustion of the planned "
        "rounds does not end discovery. The runtime has not authorized stopping. "
        "Execute the next supported discovery action under the current Judge Policy now, "
        "rather than repeating the blocker or waiting for a policy change.\n"
        "Start with pareto_status and family_discovery(action='status'); follow their current "
        "next actions. Complete any ready final-validation handoff or recover unfinished "
        "evaluations, reviews, or implementation repairs using their saved checkpoints. "
        "Before declaring evidence unavailable, inspect the retained input, trace, tool "
        "results and available artifacts, including images through view_artifact; distinguish "
        "missing evidence from evidence you have not retrieved or inspected. "
        "When a hypothesis cannot be supported, record its evidence-backed deferral, then "
        "use sample_items(coverage='unconfirmed_tasks', seed=...) with a new explicit seed "
        "to investigate a different mechanism across distinct failed tasks. An empty sample "
        "does not prove corpus exhaustion. Follow family_discovery.next_action and complete "
        "the comparison batch before advance_round. Preserve useful detectors and document "
        "real evidence limitations; do not repeat unchanged reviews, lower the policy, "
        "invent failures, or create progress records without investigation. "
        "The runtime's configured stall stop, user cancellation, service errors and saved "
        "supervisor lifecycle intent remain authoritative."
    )


def minimum_target_prompt(
    target: dict[str, Any], *, round_idx: int, max_rounds: int,
) -> str:
    """Ask for the current round's target while the driver authorizes exploration."""
    from autofa.metrics.discovery_grades import grading_table

    target = agent_failure_mode_target(
        target, round_idx=round_idx, max_rounds=max_rounds,
    )
    label = target["target_grade"].title()
    coverage = target["failure_mode_thresholds"]["min_failure_recall"]
    round_goal = _round_target_instruction(
        target["min_failure_modes"], label,
    )
    return (
        f"Continue discovery toward this round's {label} failure-detector target.\n"
        f"{label} failure detectors: {target['n_validated_failure_modes']}/"
        f"{target['min_failure_modes']}; still need {target['remaining_failure_modes']}. "
        f"Current round={round_idx + 1}, planned rounds={max_rounds}.\n"
        f"{round_goal}\n"
        f"Each target detector must be {label} or better: Coverage "
        f">{coverage:.0%}, normalized precision "
        f">={MIN_STANDARDIZED_PRECISION:.0%}, and detector adherence "
        f">={MIN_BEHAVIOR_PURITY:.0%}. Use current evaluation and claim-review evidence, "
        f"with detector matches on at least {MIN_MATCHED_DATAPOINTS} different failed "
        "datapoints: valid full-corpus support estimates count during sampled discovery. "
        "Keep observed confirmations separate; the runtime validates 30% at earlier round "
        "boundaries and validates OK-or-better detectors on the full dataset at the last "
        "planned round and extra rounds, "
        "before deciding completion. Validation opens one closing refinement pass within "
        "the same round; report its outcome through advance_round(validation_response=...) "
        "before moving to new library-wide exploration. Repeated attempts "
        "of one task do not establish breadth; previously inspected tasks can supply support "
        "without becoming held-out validation. "
        "Follow the runtime's round and continuation guidance. Preserve qualifying keepers, "
        "inspect list_failure_modes and list_library_functions, then repair promising candidates "
        "or explore a different failure family within this round. Call advance_round only "
        "after completing the family-comparison batch shown by family_discovery(action='status'), "
        "including assessment or explicit deferral of the current hypothesis. This advances "
        "the deterministic sampling sequence from the fixed main seed. Within a round, use "
        "sample_items with coverage filters or an explicit seed to inspect different examples. "
        "Iterate with capped evaluation and judge survivors with missing/stale reviews. "
        "Task breadth counts all distinct failed tasks in the prediction bucket, not just "
        "the reviewed subset. Judge reviews establish adherence separately. "
        "If a full scan matches fewer than five distinct failed tasks and the runtime "
        "requests further exploration, consider evidence-supported broadening or synthesis, "
        "or investigate a distinct mechanism. For sampled evaluations, "
        "compare the estimated full-corpus matched task support with five for provisional "
        "grading; fewer observed matches alone do not establish narrowness. Preserve "
        "promising survivors for planned final evaluation; do not repeatedly review the "
        "same small bucket. "
        "Respect judge_prefilter skips, recorded "
        "repair plans and their attempt limits; do not repeat unchanged judgments. "
        "Check pareto_status: when ready_for_final_validation is true, follow "
        "final_validation_guidance. Freeze its listed keepers and either run the requested "
        "mode='full' scan or return the summary for driver-managed scoring, as directed. "
        "Use the updated grades and findings returned by validation; the runtime "
        "rechecks completion after full scoring and review. "
        "A sampled candidate is provisional, not a reason to keep running "
        "full scans while editing. "
        "Do not weaken evidence standards, rename duplicates or invent failures to fill "
        "the quota. Explicit cancellation and saved supervisor lifecycle intent still apply.\n"
        f"Current candidates needing work: {json.dumps(target['pending_failure_modes'][:12])}\n"
        "Call pareto_status for current grades and next actions. At the end of the turn, "
        "briefly group detectors by mechanism, assess grades and explain what each new "
        "candidate adds to the library. Separate useful discoveries, including informative "
        "subtypes, from additional confirmed task coverage, then name the next concrete "
        "action. Overlapping or subset detectors can add supported explanations without "
        "adding tasks.\n"
        + grading_table()
    )
