# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable, per-case support for quality across distinct failed tasks.

Task identities and failed outcomes come from source metadata, never item-ID
syntax or weighted association counts. The receipt records detector matches;
reviewed confirmations remain a separate subset of that observed breadth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from autofa.core.dependencies import review_source
from autofa.core.localisation import text_hash
from autofa.metrics.association import matches_selected_bucket, selected_bucket
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash

MIN_GOOD_FAILED_TASKS = 5


def _field(fn: Any, name: str, default: Any = None) -> Any:
    return fn.get(name, default) if isinstance(fn, Mapping) else getattr(fn, name, default)


def _matched_failed_task_map(scores: Mapping[str, Any] | None) -> Mapping[str, str] | None:
    """Read source-verified failed matches without requiring their judge reviews."""
    s = scores or {}
    receipt = s.get("task_support")
    if (not isinstance(receipt, Mapping) or receipt.get("version") != 1
            or s.get("evaluated") is not True or s.get("evaluation_stale")
            or not s.get("evaluation_run_id")
            or receipt.get("evaluation_run_id") != s.get("evaluation_run_id")
            or not receipt.get("claim_hash")
            or receipt["claim_hash"] != s.get("evaluation_claim_hash")
            or receipt.get("selection") != list(selected_bucket(s))
            or not receipt.get("corpus_fingerprint")
            or receipt["corpus_fingerprint"] != s.get("task_recall_corpus_fingerprint")):
        return None
    matches = receipt.get("failed_matched_task_by_item_id")
    if (not isinstance(matches, Mapping)
            or any(not isinstance(item_id, str) or not item_id
                   or not isinstance(task, str) or not task.strip()
                   for item_id, task in matches.items())):
        return None
    return matches


def matched_failed_task_support(scores: Mapping[str, Any] | None) -> int | None:
    """Distinct observed failed-task matches, independent of the review sample."""
    matches = _matched_failed_task_map(scores)
    return len(set(matches.values())) if matches is not None else None


def matched_failed_attempts(scores: Mapping[str, Any] | None) -> Mapping[str, str] | None:
    """Failed attempts the current evaluation matched, as item ID to task."""
    return _matched_failed_task_map(scores)


def _confirmed_failed_tasks(
    scores: Mapping[str, Any] | None, *, item_ids: set[str] | None = None,
) -> set[str] | None:
    """Count current verified yes cases, deduplicating source-identified tasks.

    A missing receipt stays unavailable. In particular, a bare saved count or a
    list of opaque item IDs cannot promote a detector to Good. Reused reviews
    count: this is support, not an assertion of fresh independent confirmation.
    """
    from autofa.behavior_policy import strictness_for_scores
    from autofa.metrics.claim_contract import _validate_contract
    from autofa.metrics.claude_judge import compatible_judge_policy

    s = scores or {}
    receipt = s.get("task_support")
    if (not isinstance(receipt, Mapping) or receipt.get("version") != 1
            or s.get("evaluated") is not True or s.get("evaluation_stale")
            or s.get("behavior_judgment_stale")
            or s.get("judge_contract_version") != JUDGE_CONTRACT_VERSION
            or not s.get("evaluation_run_id")
            or receipt.get("evaluation_run_id") != s.get("evaluation_run_id")
            or not receipt.get("claim_hash")
            or receipt["claim_hash"] != s.get("evaluation_claim_hash")
            or receipt["claim_hash"] != s.get("claim_hash")
            or receipt.get("selection") != list(selected_bucket(s))):
        return None
    if (receipt.get("corpus_fingerprint") is not None
            and receipt["corpus_fingerprint"] != s.get("task_recall_corpus_fingerprint")):
        return None
    contract = s.get("claim_contract")
    try:
        _validate_contract(contract)
        if contract.get("claim_hash", contract["contract_id"]) != receipt["claim_hash"]:
            return None
        strictness = strictness_for_scores(s)
        strategy = s.get("judge_objective_strategy", "balanced")
        if (strategy not in {"precision", "balanced", "recall"}
                or not compatible_judge_policy(
                    s.get("judge_policy_hash"), behavior_strictness=strictness)):
            return None
        from autofa.metrics.claim_contract import contract_behavior_strictness

        if contract_behavior_strictness(contract) != strictness:
            return None
    except (TypeError, ValueError, KeyError):
        return None
    task_by_item = receipt.get("failed_matched_task_by_item_id")
    if not isinstance(task_by_item, Mapping):
        return None
    rows = s.get("behavior_verdicts")
    if not isinstance(rows, list) or not rows:
        return None
    # Ambiguous duplicate outcomes cannot provide a yes by selecting only the
    # favorable row. Normal judge aggregates have exactly one row per item.
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("item_id"), str):
            grouped.setdefault(row["item_id"], []).append(row)
    tasks = set()
    for item_id, reviews in grouped.items():
        if item_ids is not None and item_id not in item_ids:
            continue
        task = task_by_item.get(item_id)
        if len(reviews) != 1 or not isinstance(task, str) or not task.strip():
            continue
        row = reviews[0]
        initial = row.get("initial_review")
        if (row.get("verdict") != "yes" or row.get("verification_status") != "confirmed"
                or row.get("completion_status") != "complete" or row.get("validation_error")
                or row.get("processing_status") == "error"
                or row.get("claim_contract_id") != contract["contract_id"]
                or not isinstance(initial, Mapping) or initial.get("verdict") != "yes"
                or initial.get("completion_status") != "complete"
                or initial.get("validation_error") or initial.get("processing_status") == "error"
                or initial.get("item_id") != item_id
                or initial.get("claim_contract_id") != contract["contract_id"]
                or not isinstance(row.get("snapshot_id"), str) or not row["snapshot_id"]
                or initial.get("snapshot_id") != row["snapshot_id"]
                or ("assigned_label" in row and not matches_selected_bucket(
                    row["assigned_label"], selected_bucket(s)))):
            continue
        tasks.add(task)
    return tasks


def confirmed_failed_task_support(scores: Mapping[str, Any] | None) -> int | None:
    """Actual confirmed distinct failed tasks, never a sample extrapolation."""
    tasks = _confirmed_failed_tasks(scores)
    return len(tasks) if tasks is not None else None


def _count(value: Any) -> int | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        return None
    return int(value)


def _sampling_panel(scores: Mapping[str, Any]) -> list[str] | None:
    """Retain abstentions in the probability sample; never condition on resolving."""
    panel = scores.get("evaluation_item_ids")
    if (scores.get("sampled") is not True or scores.get("sample_reason") != "llm_detector"
            or scores.get("evaluation_scope_complete") is not True
            or scores.get("eval_truncated")
            or not isinstance(panel, list) or not panel
            or any(not isinstance(item_id, str) or not item_id for item_id in panel)
            or len(set(panel)) != len(panel)
            or _count(scores.get("sample_n")) != len(panel)
            or _count(scores.get("sample_requested_n")) != len(panel)):
        return None
    resolved = _count(scores.get("n_scored"))
    unresolved = _count(scores.get("n_unresolved", 0))
    frame = _count(scores.get("sample_frame_n"))
    if (resolved is None or unresolved is None or resolved + unresolved != len(panel)
            or frame is None or frame <= len(panel)):
        return None
    return panel


def _panel_hash(panel: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(panel), separators=(",", ":")).encode()).hexdigest()


def _build_sampling_design(
    scores: Mapping[str, Any], inventory: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Record only source metadata needed to project tasks in the sampled panel."""
    panel = _sampling_panel(scores)
    failed = inventory.get("task_by_failed_item_id")
    identities = inventory.get("identity_by_item_id")
    all_ids = inventory.get("item_ids")
    if (panel is None or inventory.get("inventory_complete") is not True
            or inventory.get("corpus_fingerprint") != scores.get("task_recall_corpus_fingerprint")
            or not isinstance(failed, Mapping) or not isinstance(identities, Mapping)
            or not isinstance(all_ids, list)
            or any(not isinstance(item_id, str) or not item_id for item_id in all_ids)
            or len(set(all_ids)) != len(all_ids)
            or _count(inventory.get("n_items")) != len(all_ids)
            or _count(scores.get("sample_frame_n")) != len(all_ids)
            or _count(inventory.get("n_unknown_outcomes")) != 0
            or _count(inventory.get("n_failed_attempts")) != len(failed)
            or _count(inventory.get("n_passed_attempts")) != len(all_ids) - len(failed)
            or not set(failed) <= set(all_ids) or not set(panel) <= set(all_ids)
            or any(not isinstance(task, str) or not task.strip() for task in failed.values())):
        return None
    task_population: dict[str, dict[str, Any]] = {}
    strata: dict[str, dict[str, int]] = {}
    for item_id, task in failed.items():
        identity = identities.get(item_id)
        if (not isinstance(identity, Mapping) or identity.get("_item_id") != item_id
                or not isinstance(identity.get("benchmark", ""), str)):
            return None
        benchmark = identity.get("benchmark") or "unknown"
        task_record = task_population.setdefault(
            task, {"stratum": benchmark, "failed_attempt_n": 0},
        )
        if task_record["stratum"] != benchmark:
            return None
        task_record["failed_attempt_n"] += 1
        stratum = strata.setdefault(benchmark, {"population_n": 0, "task_n": 0, "sample_n": 0})
        stratum["population_n"] += 1
    if _count(inventory.get("n_failed_tasks")) != len(task_population):
        return None
    for task_record in task_population.values():
        strata[task_record["stratum"]]["task_n"] += 1
    panel_failed = {item_id: failed[item_id] for item_id in panel if item_id in failed}
    for task in panel_failed.values():
        strata[task_population[task]["stratum"]]["sample_n"] += 1
    # There is no full-corpus estimate if an entire failed benchmark is absent.
    if not strata or any(stratum["sample_n"] == 0 for stratum in strata.values()):
        return None
    return {
        "version": 1, "method": "stratified_task_inclusion",
        "panel_sha256": _panel_hash(panel), "frame_n": len(all_ids), "sample_n": len(panel),
        "failed_population_n": len(failed), "failed_task_n": len(task_population),
        "strata": strata, "sample_failed_task_by_item_id": panel_failed,
        "task_population": {task: task_population[task] for task in set(panel_failed.values())},
    }


def _read_sampling_design(scores: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate portable design against the current evaluation, corpus and panel."""
    panel = _sampling_panel(scores)
    receipt = scores.get("task_support")
    if (panel is None or not isinstance(receipt, Mapping) or receipt.get("version") != 1
            or scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or not scores.get("evaluation_run_id")
            or receipt.get("evaluation_run_id") != scores["evaluation_run_id"]
            or not receipt.get("claim_hash")
            or receipt["claim_hash"] != scores.get("evaluation_claim_hash")
            or receipt.get("selection") != list(selected_bucket(scores))
            or not receipt.get("corpus_fingerprint")
            or receipt["corpus_fingerprint"] != scores.get("task_recall_corpus_fingerprint")
            or not isinstance(receipt.get("failed_matched_task_by_item_id"), Mapping)):
        return None
    design = receipt.get("sampling_design")
    if (not isinstance(design, Mapping) or design.get("version") != 1
            or design.get("method") != "stratified_task_inclusion"
            or design.get("panel_sha256") != _panel_hash(panel)
            or _count(design.get("frame_n")) != _count(scores.get("sample_frame_n"))
            or _count(design.get("sample_n")) != len(panel)):
        return None
    strata, tasks = design.get("strata"), design.get("task_population")
    failed_panel = design.get("sample_failed_task_by_item_id")
    if (not isinstance(strata, Mapping) or not strata or not isinstance(tasks, Mapping)
            or any(not isinstance(key, str) or not key for key in strata)
            or not isinstance(failed_panel, Mapping) or not set(failed_panel) <= set(panel)
            or any(not isinstance(task, str) or not task for task in failed_panel.values())
            or set(tasks) != set(failed_panel.values())):
        return None
    population_n = task_n = 0
    for stratum in strata.values():
        if not isinstance(stratum, Mapping):
            return None
        population, count, sampled = (_count(stratum.get(key)) for key in
                                       ("population_n", "task_n", "sample_n"))
        if (population is None or count is None or sampled is None
                or not 0 < count <= population or not 0 < sampled <= population):
            return None
        population_n += population
        task_n += count
    if (population_n != _count(design.get("failed_population_n"))
            or task_n != _count(design.get("failed_task_n"))
            or population_n > design["frame_n"]):
        return None
    multiplicities = Counter(failed_panel.values())
    sampled_by_stratum: Counter[str] = Counter()
    tasks_by_stratum: Counter[str] = Counter()
    attempts_by_stratum: Counter[str] = Counter()
    for task, record in tasks.items():
        if (not isinstance(record, Mapping) or not isinstance(record.get("stratum"), str)
                or record["stratum"] not in strata):
            return None
        benchmark = record["stratum"]
        multiplicity = _count(record.get("failed_attempt_n"))
        if (multiplicity is None or not multiplicities[task] <= multiplicity
                <= strata[benchmark]["population_n"]):
            return None
        sampled_by_stratum[benchmark] += multiplicities[task]
        tasks_by_stratum[benchmark] += 1
        attempts_by_stratum[benchmark] += multiplicity
    if any(sampled_by_stratum[benchmark] != record["sample_n"]
           or tasks_by_stratum[benchmark] > record["task_n"]
           or attempts_by_stratum[benchmark] > record["population_n"]
           for benchmark, record in strata.items()):
        return None
    return dict(design)


def _task_inclusion_probability(population: int, sampled: int, multiplicity: int) -> float:
    """P(at least one repeat selected), without huge combinations or cancellation."""
    if sampled + multiplicity > population:
        return 1.0
    # C(N-m,n)/C(N,n) can equivalently use n/m swapped; use the shorter product.
    terms, numerator = min(sampled, multiplicity), max(sampled, multiplicity)
    return -math.expm1(sum(math.log1p(-numerator / (population - i)) for i in range(terms)))


def task_support_assessment(scores: Mapping[str, Any] | None) -> dict[str, Any]:
    """Observed support and a provisional corpus projection from the scoring panel.

    Breadth counts source-verified failed matches; confirmed reviews are a
    separate diagnostic. Projection weights correct benchmark allocation and
    repeated attempts. Only in-panel cases contribute to either projection.
    Differing outcomes across repeats can make these conservative projections;
    neither is a census or independently validated count. No design-based
    confidence interval is claimed for adaptively reviewed cases.
    """
    s = scores or {}
    confirmed_actual = confirmed_failed_task_support(s)
    matched_actual = matched_failed_task_support(s)
    result = {
        "n_confirmed_failed_tasks": confirmed_actual,
        "confirmed_task_support_status": "verified" if confirmed_actual is not None
        else "unavailable",
        "n_matched_failed_tasks": matched_actual,
        "matched_task_support_status": "verified" if matched_actual is not None else "unavailable",
        "estimated_n_confirmed_failed_tasks": None, "estimated_n_matched_failed_tasks": None,
        "task_support_estimate_status": "unavailable", "n_sampled_failed_tasks": None,
        "n_sampled_confirmed_failed_tasks": None, "n_sampled_matched_failed_tasks": None,
        "task_support_comparison": matched_actual,
        "task_support_comparison_basis": "observed" if matched_actual is not None
        else "unavailable",
        "task_support_estimate_reason": (
            "A full source-verified probability-sample frame is required for extrapolation."
            if s.get("sampled") is True
            else "Task breadth uses source-verified distinct failed-task matches; "
            "reviewed confirmations and detector adherence are separate."
        ),
    }
    design = _read_sampling_design(s)
    if design is None or matched_actual is None:
        return result
    panel_failed = design["sample_failed_task_by_item_id"]
    observed_matches = s["task_support"]["failed_matched_task_by_item_id"]
    matched = {task for item_id, task in panel_failed.items()
               if observed_matches.get(item_id) == task}
    confirmed = _confirmed_failed_tasks(s, item_ids=set(panel_failed))

    def project(tasks: set[str]) -> float:
        by_stratum: Counter[str] = Counter()
        for task in sorted(tasks):
            record = design["task_population"][task]
            benchmark = record["stratum"]
            stratum = design["strata"][benchmark]
            probability = _task_inclusion_probability(
                int(stratum["population_n"]), int(stratum["sample_n"]),
                int(record["failed_attempt_n"]),
            )
            by_stratum[benchmark] += 1 / probability
        return sum(min(value, design["strata"][benchmark]["task_n"])
                   for benchmark, value in by_stratum.items())

    confirmed_estimate = project(confirmed & matched) if confirmed is not None else None
    matched_estimate = project(matched)
    result.update({
        "estimated_n_confirmed_failed_tasks": confirmed_estimate,
        "estimated_n_matched_failed_tasks": matched_estimate,
        "task_support_estimate_status": "provisional",
        "n_sampled_failed_tasks": len(design["task_population"]),
        "n_sampled_confirmed_failed_tasks": len(confirmed & matched)
        if confirmed is not None else None,
        "n_sampled_matched_failed_tasks": len(matched),
        "task_support_estimate_reason": (
            "Provisional full-corpus projection from the scoring sample, weighted by each task's "
            "chance of inclusion under benchmark/outcome stratification and repeated attempts. "
            "Task breadth uses matching failed tasks, independently of how many matches were "
            "reviewed. Only reviewed, confirmed in-panel matches contribute to the separate "
            "confirmed-support diagnostic. Rollout-specific behavior can undercount breadth; "
            "this is not a verified census or independent confirmation. Full scans use actual "
            "distinct failed-task matches."
        ),
    })
    if matched_actual < MIN_GOOD_FAILED_TASKS and matched_estimate > matched_actual:
        result["task_support_comparison"] = matched_estimate
        result["task_support_comparison_basis"] = "estimated"
    return result


def _historical_failed_tasks(fn: Any, history: Mapping[str, Any]) -> dict[str, str]:
    """Recover saved source identities and failed matches without new reviews.

    Only the selected evaluated bucket's explicit failed examples qualify;
    sampled counts, arbitrary examples and identifier spelling do not.
    """
    if history.get("schema_version") != 1:
        return {}
    states = history.get("detectors")
    state = states.get(_field(fn, "name"), {}) if isinstance(states, Mapping) else {}
    groups = state.get("item_task_groups", {}) if isinstance(state, Mapping) else {}
    if not isinstance(groups, Mapping):
        return {}
    scores = _field(fn, "scores", {}) or {}
    selection = selected_bucket(scores)
    failed_ids = {
        item_id for bucket in scores.get("buckets") or []
        if isinstance(bucket, Mapping) and matches_selected_bucket(bucket.get("bucket"), selection)
        for item_id in bucket.get("example_fail_ids") or [] if isinstance(item_id, str)
    }
    result = {}
    for item_id in failed_ids:
        group = groups.get(item_id)
        try:
            parts = json.loads(group) if isinstance(group, str) else None
        except (TypeError, ValueError):
            continue
        if (isinstance(parts, list) and len(parts) == 2
                and all(isinstance(part, str) for part in parts) and parts[1].strip()):
            result[item_id] = json.dumps(parts, separators=(",", ":"))
    return result


def build_task_support(
    fn: Any, inventory: Mapping[str, Any] | None = None, *,
    review_history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a small support receipt from verified matches; never edit ``fn``.

    Evaluation and judging call this with their cached identity inventory. Report
    read views may recover legacy cases from portable exposure history and saved
    failed examples, preserving existing valid reviews without additional cost.
    """
    scores = _field(fn, "scores", {}) or {}
    source = review_source(fn)
    fingerprint = claim_hash(
        source=source, description=_field(fn, "description", "") or "",
        mechanism=(_field(fn, "mechanism", "") or _field(fn, "description", "")
                   or _field(fn, "name", "")),
    )
    receipt = None
    evaluation_current = (scores.get("evaluated") is True and not scores.get("evaluation_stale")
                          and scores.get("evaluation_claim_hash") == fingerprint
                          and bool(scores.get("evaluation_run_id")))
    corpus = None
    failed = None
    if evaluation_current and inventory is not None:
        if (inventory.get("inventory_complete") is True and inventory.get("corpus_fingerprint")
                and isinstance(inventory.get("task_by_failed_item_id"), Mapping)):
            failed = inventory["task_by_failed_item_id"]
            corpus = inventory["corpus_fingerprint"]
    elif (evaluation_current and review_history is not None
          and not isinstance(scores.get("task_support"), Mapping)):
        failed = _historical_failed_tasks(fn, review_history) or None
        corpus = scores.get("task_recall_corpus_fingerprint")
    if failed is not None:
        records = _field(fn, "failure_localisations", {}) or {}
        selection = selected_bucket(scores)
        source_hash = text_hash(source)
        matched = {
            item_id: task for item_id, task in failed.items()
            if isinstance(task, str) and task.strip()
            and isinstance(record := records.get(item_id), Mapping)
            and record.get("item_id") == item_id
            and record.get("source_sha256") == source_hash
            and "assigned_label" in record
            and matches_selected_bucket(record["assigned_label"], selection)
        }
        previous = scores.get("task_support")
        panel = scores.get("evaluation_item_ids")
        # A reviewed case outside a newly scored panel remains evidence for the
        # same frozen detector. Rechecked negatives/errors are excluded, while
        # full scans always replace the receipt with their current matches.
        if (scores.get("sampled") is True and isinstance(panel, list)
                and all(isinstance(item_id, str) for item_id in panel)
                and isinstance(previous, Mapping) and previous.get("version") == 1
                and previous.get("claim_hash") == fingerprint
                and corpus and previous.get("corpus_fingerprint") == corpus
                and previous.get("selection") == list(selection)
                and isinstance(previous.get("failed_matched_task_by_item_id"), Mapping)):
            evaluated_ids = set(panel)
            old_matches = previous["failed_matched_task_by_item_id"]
            matched.update({
                item_id: task for item_id, task in old_matches.items()
                if item_id not in evaluated_ids and isinstance(task, str) and task.strip()
                and failed.get(item_id) == task
            })
        receipt = {
            "version": 1, "claim_hash": fingerprint,
            "evaluation_run_id": scores["evaluation_run_id"],
            "corpus_fingerprint": corpus, "selection": list(selection),
            "failed_matched_task_by_item_id": matched,
        }
    elif evaluation_current:
        receipt = scores.get("task_support")
    if isinstance(receipt, Mapping) and inventory is not None:
        receipt = dict(receipt)
        design = _build_sampling_design(scores, inventory)
        if design is not None:
            receipt["sampling_design"] = design
        else:
            receipt.pop("sampling_design", None)
    view = {**scores, "task_support": receipt}
    assessment = task_support_assessment(view)
    # The evaluator's persisted count belongs to its frozen recall cohort.
    # Support receipts may additionally retain current matches outside that
    # cohort; expose their breadth through assessment, without changing recall.
    assessment.pop("n_matched_failed_tasks")
    return {"task_support": receipt, **assessment}
