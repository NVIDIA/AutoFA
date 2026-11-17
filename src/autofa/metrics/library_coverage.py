# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only distinct failure coverage for discovery, never inferred from examples.

Saved positive detector records establish *observed matches*. Current independent
judge confirmations establish *reviewed causal cases*. Neither a sample's purity
nor an estimated contingency count can turn unreviewed cases into confirmations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from autofa.core.dependencies import review_source
from autofa.core.library import FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.metrics.association import matches_selected_bucket, selected_bucket, truthy_bucket
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
from autofa.metrics.verdicts import current_behavior_scores


def failure_cohort(dataset: Any) -> dict[str, Any]:
    """Identify the selected dataset's failures using identity columns only.

    Unknown benchmark outcomes are kept separate; they are not failed tasks.
    The existing identity index avoids loading multi-megabyte rollout text.
    """
    from autofa.data.parquet_store import IDENTITY_FIELDS, _identity_row_allows
    from autofa.data.row_index import _row_is_correct

    failed, passed, unknown = set(), set(), set()
    if getattr(dataset, "parquet_paths", None):
        rows = dataset.identity_index().rows
        identities = ((row.item_id, row.is_correct) for row in rows
                      if _identity_row_allows(row, dataset))
    else:
        batches = dataset.iter_batches(columns=list(IDENTITY_FIELDS))
        identities = ((str(row.get("_item_id") or ""), _row_is_correct(row))
                      for batch in batches for row in batch)
    for item_id, outcome in identities:
        if not item_id:
            raise ValueError("Failure coverage requires stable item IDs")
        target = passed if outcome is True else failed if outcome is False else unknown
        target.add(item_id)
    if failed & passed or failed & unknown or passed & unknown:
        raise ValueError("Failure coverage found conflicting outcomes for an item ID")
    return {"failed_item_ids": failed, "n_items": len(failed | passed | unknown),
            "n_passed": len(passed), "n_unknown_outcomes": len(unknown)}


def library_failure_coverage(
    library: FunctionLibrary,
    failed_item_ids: Iterable[str],
    *,
    run_id: str | None = None,
    objective_strategy: str | None = None,
    behavior_strictness: str | None = None,
    include_item_ids: bool = False,
    task_by_item_id: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Union saved, current matches and confirmations; expose overlap explicitly.

    ``exclusive`` means additional cases beyond *all other* library detectors.
    Parent/child unions therefore never count a task twice. All counts are
    observed lower bounds: unseen, abstained and unreviewed items stay unknown.
    ``include_item_ids`` is for internal sampling; normal tool replies are small.
    """
    failures = set(failed_item_ids)
    task_map = task_by_item_id or {}

    def task_ids(ids: Iterable[str]) -> set[str]:
        return {task_map[item_id] for item_id in ids if task_map.get(item_id) is not None}

    matches: dict[str, set[str]] = {}
    confirmations: dict[str, set[str]] = {}
    details: dict[str, dict[str, Any]] = {}
    for fn in library.functions:
        scores = fn.scores or {}
        source = review_source(fn)
        source_hash = text_hash(source)
        fingerprint = claim_hash(source=source, description=fn.description or "",
                                 mechanism=fn.mechanism or fn.description or fn.name)
        current_evaluation = (
            bool(scores.get("evaluated"))
            and scores.get("evaluation_claim_hash") == fingerprint
            and not scores.get("evaluation_stale")
            and (run_id is None or scores.get("evaluation_run_id") == run_id)
        )
        matched: set[str] = set()
        ignored_records = 0
        selection = selected_bucket(scores)
        if current_evaluation:
            for item_id, record in (fn.failure_localisations or {}).items():
                if item_id not in failures:
                    continue
                if (isinstance(record, dict)
                        and record.get("item_id") == item_id
                        and record.get("source_sha256") == source_hash
                        and "assigned_label" in record
                        and truthy_bucket(record["assigned_label"])):
                    if matches_selected_bucket(record["assigned_label"], selection):
                        matched.add(item_id)
                else:
                    ignored_records += 1
        judged = current_behavior_scores(
            scores, source=source, description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
            objective_strategy=objective_strategy,
            behavior_strictness=behavior_strictness,
        )
        current_judgment = (
            current_evaluation
            and judged.get("judge_contract_version") == JUDGE_CONTRACT_VERSION
            and judged.get("claim_hash") == fingerprint
            and not judged.get("behavior_judgment_stale")
        )
        confirmed: set[str] = set()
        contract = judged.get("claim_contract") or {}
        contract_id = contract.get("contract_id") if isinstance(contract, dict) else None
        if current_judgment and contract_id:
            for row in judged.get("behavior_verdicts") or []:
                if not isinstance(row, dict):
                    continue
                item_id = row.get("item_id")
                initial = row.get("initial_review") or {}
                if (item_id in matched and row.get("verdict") == "yes"
                        and row.get("verification_status") == "confirmed"
                        and row.get("claim_contract_id") == contract_id
                        and isinstance(initial, dict) and initial.get("verdict") == "yes"
                        and initial.get("item_id") == item_id
                        and initial.get("claim_contract_id") == contract_id
                        and row.get("snapshot_id")
                        and initial.get("snapshot_id") == row["snapshot_id"]):
                    confirmed.add(item_id)
        matches[fn.name], confirmations[fn.name] = matched, confirmed
        details[fn.name] = {
            "name": fn.name, "explains": fn.explains, "depends_on": list(fn.depends_on),
            "bucket": selection[0],
            "evaluation_current": current_evaluation,
            "judgment_current": current_judgment,
            "evaluation_sampled": bool(scores.get("sampled")),
            "evaluation_scope_complete": scores.get("evaluation_scope_complete") is True,
            "n_detector_unresolved": int(scores.get("n_unresolved") or 0),
            "n_ignored_match_records": ignored_records,
            "review_reused_n": int((scores.get("behavior_sample") or {}).get("reused_n") or 0),
            "n_known_matched_failed": len(matched),
            "n_review_confirmed_failed": len(confirmed),
        }
    match_counts = Counter(item_id for ids in matches.values() for item_id in ids)
    confirm_counts = Counter(item_id for ids in confirmations.values() for item_id in ids)
    union_matches, union_confirmed = set(match_counts), set(confirm_counts)
    matched_tasks = {name: task_ids(ids) for name, ids in matches.items()}
    confirmed_tasks = {name: task_ids(ids) for name, ids in confirmations.items()}
    match_task_counts = Counter(task for ids in matched_tasks.values() for task in ids)
    confirm_task_counts = Counter(task for ids in confirmed_tasks.values() for task in ids)
    for name, row in details.items():
        row["n_exclusive_matched_failed"] = sum(match_counts[item_id] == 1
                                                for item_id in matches[name])
        row["n_exclusive_review_confirmed_failed"] = sum(confirm_counts[item_id] == 1
                                                         for item_id in confirmations[name])
        unknown_n = sum(task_map.get(item_id) is None for item_id in matches[name])
        known_n = len(matches[name]) - unknown_n
        task_counts = Counter(task_map[item_id] for item_id in matches[name]
                              if task_map.get(item_id) is not None)
        row.update({
            "n_matched_failed_tasks": len(matched_tasks[name]),
            "n_review_confirmed_failed_tasks": len(confirmed_tasks[name]),
            "n_exclusive_matched_failed_tasks": sum(
                match_task_counts[task] == 1 for task in matched_tasks[name]),
            "n_exclusive_review_confirmed_failed_tasks": sum(
                confirm_task_counts[task] == 1 for task in confirmed_tasks[name]),
            "n_matched_failed_attempts_without_task_identity": unknown_n,
            "n_repeated_matched_failed_attempts": known_n - len(matched_tasks[name]),
            "largest_task_share_of_known_matches": (
                max(task_counts.values()) / known_n if known_n else None),
            "generalization_status": (
                "cross_task_causal_evidence" if len(confirmed_tasks[name]) >= 2 else
                "cross_task_matches_only" if len(matched_tasks[name]) >= 2 else
                "task_identity_unknown" if unknown_n else
                "single_task_diagnostic" if matched_tasks[name] else "no_observed_failed_matches"
            ),
        })
        parent = row["explains"]
        if parent:
            row["parent_present"] = parent in matches
            row["n_additional_matched_beyond_parent"] = (
                len(matches[name] - matches[parent]) if parent in matches else None)
            row["n_additional_confirmed_beyond_parent"] = (
                len(confirmations[name] - confirmations[parent])
                if parent in confirmations else None)
            row["n_additional_matched_tasks_beyond_parent"] = (
                len(matched_tasks[name] - matched_tasks[parent]) if parent in matches else None)
        if include_item_ids:
            row["known_matched_failed_item_ids"] = sorted(matches[name])
            row["review_confirmed_failed_item_ids"] = sorted(confirmations[name])
    all_tasks = task_ids(failures)
    result = {
        "coverage_scope": "saved_current_positive_records",
        "denominator_scope": "selected_corpus_failed_attempts",
        "n_failed": len(failures),
        "n_failed_tasks": len(all_tasks),
        "n_failed_attempts_without_task_identity": sum(
            task_map.get(item_id) is None for item_id in failures),
        "n_known_matched_failed_tasks": len(match_task_counts),
        "n_review_confirmed_failed_tasks": len(confirm_task_counts),
        "n_tasks_without_known_match": len(all_tasks - match_task_counts.keys()),
        "observed_task_recall_lower_bound": (
            len(match_task_counts) / len(all_tasks) if all_tasks else None),
        "reviewed_task_confirmation_recall_lower_bound": (
            len(confirm_task_counts) / len(all_tasks) if all_tasks else None),
        "n_known_matched_failed": len(union_matches),
        "n_review_confirmed_failed": len(union_confirmed),
        "n_without_known_match": len(failures - union_matches),
        "n_without_review_confirmation": len(failures - union_confirmed),
        "observed_match_recall_lower_bound": (
            len(union_matches) / len(failures) if failures else None),
        "reviewed_confirmation_recall_lower_bound": (
            len(union_confirmed) / len(failures) if failures else None),
        "detectors": list(details.values()),
        "interpretation": (
            "Failed attempts and metadata-identified tasks are counted separately, once across "
            "all parents and children; repeated attempts never establish distinct datapoints. "
            "Task recall uses only tasks with known source identity; missing identities stay "
            "unknown. Matches are "
            "observed positives in the scored bucket, not proof of a cause. Reviewed counts "
            "include individually confirmed cases under the current claim and judge policy. They "
            "are not an extrapolation from sample purity or fresh held-out validation. No known "
            "match means absent from saved current positives, including untested or unresolved "
            "cases; it does not prove every detector returned negative. Exclusive counts are "
            "additional observed cases beyond all other detectors, not additive coverage scores. "
            "Single-task diagnostics are not cross-datapoint validation."
        ),
    }
    if include_item_ids:
        result["known_matched_failed_item_ids"] = sorted(union_matches)
        result["review_confirmed_failed_item_ids"] = sorted(union_confirmed)
    return result
