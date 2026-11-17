# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only breadth signals for family search, separate from objective metrics.

Only identity columns and saved detector results are read. Observed matches do
not establish causal validity, and sampled positives never become a census by
being compared with a second detector's saved positives.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.core.localisation import text_hash
from autofa.metrics.association import matches_selected_bucket, selected_bucket
from autofa.metrics.claim_contract import claim_hash

_IDENTITY_COLUMNS = [
    "model_id", "benchmark", "problem_id", "task_id", "instance_id", "question_id",
    "item_id", "run_idx", "repeat", "hb_problem_idx", "is_correct", "accuracy",
    "resolved", "tb_is_resolved", "is_resolved", "correct",
]
_TASK_COLUMNS = ("problem_id", "task_id", "instance_id", "question_id")


def _explicit_task(row: Mapping[str, Any], columns: Iterable[str] = _TASK_COLUMNS) -> str | None:
    task = next((row[key] for key in columns
                 if row.get(key) is not None and str(row[key]).strip()), None)
    return f"{row.get('benchmark') or 'unknown'}:{task}" if task is not None else None


def _source_task(
    dataset: Any, row: Mapping[str, Any], item_id: str, source_cache: dict,
) -> str | None:
    """Recover genuine task metadata before item_id-to-problem_id aliasing.

    A schema check handles ordinary shards. When a source contains item_id plus
    real task columns, read those small identity columns before aliasing: a NULL
    problem_id may otherwise already have been filled with the attempt ID.
    """
    source = row.get("_source_parquet")
    if not source:
        return _explicit_task(row)
    if source not in source_cache:
        import pyarrow.parquet as pq

        from autofa.data.parquet_store import MAX_BATCH_BYTES, apply_column_aliases, iter_arrow_rows
        from autofa.util.memory import row_group_bytes_per_row

        with pq.ParquetFile(source) as parquet:
            schema = set(parquet.schema_arrow.names)
            genuine = tuple(key for key in _TASK_COLUMNS if key in schema)
            before_aliases = None
            if genuine and "item_id" in schema:
                before_aliases = {}
                columns = [key for key in _IDENTITY_COLUMNS if key in schema]
                for group in range(parquet.metadata.num_row_groups):
                    row_bytes = row_group_bytes_per_row(parquet.metadata.row_group(group), columns)
                    batch_size = max(1, min(512, MAX_BATCH_BYTES // max(1, row_bytes)))
                    for batch in parquet.iter_batches(
                        batch_size=batch_size, columns=columns, row_groups=[group],
                    ):
                        for raw in iter_arrow_rows(batch):
                            task = _explicit_task(raw, genuine)
                            apply_column_aliases(raw)
                            raw_id = dataset._make_item_id(raw, Path(source))
                            before_aliases[raw_id] = task
            source_cache[source] = (genuine, before_aliases)
    genuine, before_aliases = source_cache[source]
    if before_aliases is not None:
        if item_id not in before_aliases:
            raise ValueError(f"Task metadata missing for indexed item: {item_id}")
        return before_aliases[item_id]
    return _explicit_task(row, genuine)


def failed_task_inventory(dataset: Any) -> dict[str, Any]:
    """Count attempts and real task identities in the selected, filtered corpus.

    The identity index intentionally omits problem IDs, so stream an identity-only
    projection for the task map. Unknown outcomes are never counted as failures;
    an opaque attempt ID never establishes a distinct task.
    """
    from autofa.data.parquet_store import apply_column_aliases
    from autofa.data.row_index import _row_is_correct

    if callable(getattr(dataset, "iter_batches", None)):
        batches = dataset.iter_batches(columns=_IDENTITY_COLUMNS, max_items=None)
    elif not getattr(dataset, "parquet_paths", None) and callable(
        getattr(dataset, "load_items", None)
    ):
        # Compatibility for small in-memory adapters; never fall back to eagerly
        # loading a real Parquet corpus after a streaming failure.
        batches = [dataset.load_items()]
    else:
        raise ValueError("Family coverage needs an identity-column dataset reader")
    identities: dict[str, tuple[bool | None, str | None]] = {}
    review_identities: dict[str, dict[str, str]] = {}
    source_cache: dict[str, tuple] = {}
    try:
        for batch in batches:
            for original in batch:
                row = {key: original[key] for key in _IDENTITY_COLUMNS if key in original}
                apply_column_aliases(row)
                item_id = str(original.get("_item_id") or "")
                if not item_id:
                    raise ValueError("Family coverage requires stable item IDs")
                if item_id in identities:
                    raise ValueError(f"Family coverage found duplicate item ID: {item_id}")
                task_id = _source_task(dataset, original, item_id, source_cache)
                identities[item_id] = (_row_is_correct(row), task_id)
                benchmark = str(original.get("benchmark") or "")
                projection = {"_item_id": item_id, "benchmark": benchmark}
                if task_id is not None:
                    prefix = f"{benchmark or 'unknown'}:"
                    if not task_id.startswith(prefix):
                        raise ValueError(f"Task metadata benchmark differs for item {item_id}")
                    projection["problem_id"] = task_id[len(prefix):]
                review_identities[item_id] = projection
    finally:
        if callable(getattr(batches, "close", None)):
            batches.close()
    failed = {item_id: task for item_id, (outcome, task) in identities.items()
              if outcome is False}
    fingerprint = hashlib.sha256(json.dumps(
        sorted(identities.items()), separators=(",", ":"),
    ).encode()).hexdigest()
    return {
        "inventory_complete": True, "corpus_fingerprint": fingerprint,
        "item_ids": sorted(identities),
        "n_items": len(identities), "n_failed_attempts": len(failed),
        "n_passed_attempts": sum(outcome is True for outcome, _ in identities.values()),
        "n_unknown_outcomes": sum(outcome is None for outcome, _ in identities.values()),
        "n_failed_tasks": len({task for task in failed.values() if task is not None}),
        "n_failed_attempts_without_task_identity": sum(task is None for task in failed.values()),
        "task_by_failed_item_id": failed,
        "task_by_item_id": {item_id: task for item_id, (_, task) in identities.items()},
        "identity_by_item_id": review_identities,
        "passed_item_ids": sorted(item_id for item_id, (outcome, _) in identities.items()
                                  if outcome is True),
        "unknown_outcome_item_ids": sorted(item_id for item_id, (outcome, _) in identities.items()
                                           if outcome is None),
    }


def cached_task_inventory(rt: Any) -> dict[str, Any]:
    """Cache lightweight source identities once for the runtime's fixed dataset."""
    inventory = getattr(rt, "_family_failed_task_inventory", None)
    if inventory is None or "identity_by_item_id" not in inventory:
        inventory = failed_task_inventory(rt.dataset)
        rt._family_failed_task_inventory = inventory
    return inventory


def verified_identity_projection(rt: Any, item_or_id: Mapping[str, Any] | str) -> dict[str, str]:
    """Return source-verified task metadata for exposure and review allocation.

    Display rows and normalized problem_id aliases are not authoritative. An
    unknown item or opaque attempt ID stays unknown, and cannot establish task
    independence. The returned copy contains no trajectory or outcome fields.
    """
    item_id = (item_or_id if isinstance(item_or_id, str)
               else item_or_id.get("_item_id", item_or_id.get("item_id")))
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("Verified task identity requires a nonempty item ID")
    identities = cached_task_inventory(rt)["identity_by_item_id"]
    return dict(identities.get(item_id, {"_item_id": item_id}))


def _count(value: Any) -> int | None:
    # Association scores serialize census counts as floats, too. Weighted sample
    # estimates are excluded separately; integer-looking samples are not censuses.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        return None
    return int(value)


def _rate(value: Any) -> float | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        return None
    return float(value)


def _snapshot(rt: Any, fn: Any, inventory: dict[str, Any]) -> dict[str, Any]:
    scores = fn.scores or {}
    source = review_source(fn)
    fingerprint = claim_hash(source=source, description=fn.description or "",
                             mechanism=fn.mechanism or fn.description or fn.name)
    current = bool(scores.get("evaluated") and not scores.get("evaluation_stale")
                   and scores.get("evaluation_run_id") == rt.run.run_id
                   and scores.get("evaluation_claim_hash") == fingerprint)
    selection = selected_bucket(scores)
    positives = set()
    if current:
        source_hash = text_hash(source)
        for item_id, record in (fn.failure_localisations or {}).items():
            if (isinstance(record, dict) and record.get("item_id") == item_id
                    and record.get("source_sha256") == source_hash
                    and "assigned_label" in record
                    and matches_selected_bucket(record["assigned_label"], selection)):
                positives.add(item_id)
    task_map = inventory["task_by_failed_item_id"]
    failed = positives & task_map.keys()
    sampled = bool(scores.get("sampled"))
    unresolved = any(scores.get(key) for key in (
        "n_unresolved", "n_insufficient", "errors", "n_errors", "parent_errors",
    )) or (scores.get("evaluation_complete") is False
           or scores.get("dataset_evaluation_complete") is False
           or scores.get("assessment_coverage", 1) != 1)
    census = bool(current and inventory["inventory_complete"] and not sampled
                  and scores.get("sampled") is False
                  and scores.get("evaluation_scope_complete") is True
                  and not scores.get("eval_truncated") and not unresolved
                  and not inventory["n_unknown_outcomes"]
                  and _count(scores.get("n_items")) == inventory["n_items"])
    n_flagged = _count(scores.get("n_flagged"))
    n_failed = _count(scores.get("n_fail_caught", scores.get("a")))
    matches_complete = bool(census and positives <= set(inventory["item_ids"])
                            and n_flagged == len(positives)
                            and n_failed == len(failed))
    if not current:
        recall_scope = "unavailable_stale_or_missing_evaluation"
    elif sampled:
        recall_scope = "sample_estimate"
    elif unresolved:
        recall_scope = "resolved_evaluated_attempts"
    elif not census:
        recall_scope = "evaluated_subset"
    else:
        recall_scope = "full_selected_corpus"
    tasks = {task_map[item_id] for item_id in failed if task_map[item_id] is not None}
    unknown_tasks = sum(task_map[item_id] is None for item_id in failed)
    return {
        "name": fn.name, "claim_hash": fingerprint, "evaluation_run_id": rt.run.run_id,
        "corpus_fingerprint": inventory["corpus_fingerprint"],
        "evaluation_current": current, "failure_recall": _rate(scores.get("recall"))
        if current else None, "recall_scope": recall_scope,
        "evaluation_sampled": sampled, "census_complete": census,
        "match_inventory_complete": matches_complete,
        "n_failed_attempts_in_corpus": inventory["n_failed_attempts"],
        "n_known_matched_failed": len(failed),
        "n_matched_failed_tasks": len(tasks),
        "n_repeated_matched_failed_attempts": len(failed) - unknown_tasks - len(tasks),
        "generalization_status": (
            "cross_task_matches_observed" if len(tasks) >= 2 else
            "task_identity_unknown" if unknown_tasks else
            "single_task_diagnostic" if tasks else "no_observed_failed_matches"
        ),
        "matched_task_inventory_complete": matches_complete and all(
            task_map[item_id] is not None for item_id in failed),
        "n_matched_failed_attempts_without_task_identity": unknown_tasks,
        "known_matched_failed_item_ids": sorted(failed),
        "observed_failure_recall_lower_bound": len(failed) / len(task_map) if task_map else None,
    }


def candidate_coverage(
    rt: Any, fn: Any, previous_candidates: Iterable[Any] = (),
) -> dict[str, Any]:
    """Compare a candidate with assessed family snapshots, using no model calls.

    Previous candidates may be persisted snapshots returned by this function,
    library names, or FeatureFunctions. Snapshots retain the assessed revision's
    observations even if its detector is later repaired. Exact additional coverage
    requires complete current-run census records for every compared candidate.
    """
    inventory = cached_task_inventory(rt)
    result = _snapshot(rt, fn, inventory)
    others, compared, unavailable = set(), [], []
    comparisons_complete = True
    task_comparisons_complete = True
    for previous in previous_candidates:
        if isinstance(previous, str):
            previous_name = previous
            previous = rt.library.get(previous)
            if previous is None:
                unavailable.append(previous_name)
                comparisons_complete = False
                continue
        snapshot = (dict(previous) if isinstance(previous, Mapping)
                    else _snapshot(rt, previous, inventory))
        name = snapshot.get("name")
        if name == fn.name:
            continue
        compatible = (snapshot.get("evaluation_current") is True
                      and snapshot.get("evaluation_run_id") == rt.run.run_id
                      and snapshot.get("corpus_fingerprint") == inventory["corpus_fingerprint"])
        if not compatible:
            unavailable.append(str(name or "unknown"))
            comparisons_complete = False
            continue
        compared.append(name)
        others.update(snapshot.get("known_matched_failed_item_ids") or [])
        comparisons_complete &= snapshot.get("match_inventory_complete") is True
        task_comparisons_complete &= snapshot.get("matched_task_inventory_complete") is True
    additional = set(result["known_matched_failed_item_ids"]) - others
    complete = result["match_inventory_complete"] and comparisons_complete
    task_map = inventory["task_by_failed_item_id"]
    own_tasks = {task_map[item_id] for item_id in result["known_matched_failed_item_ids"]
                 if task_map.get(item_id) is not None}
    other_tasks = {task_map[item_id] for item_id in others if task_map.get(item_id) is not None}
    additional_tasks = own_tasks - other_tasks
    task_complete = (complete and result["matched_task_inventory_complete"]
                     and task_comparisons_complete)
    result.update({
        "compared_families": compared, "unavailable_comparisons": unavailable,
        "comparison_complete": complete,
        "n_additional_failed_attempts": len(additional) if complete else None,
        "n_observed_additional_failed_attempts": len(additional)
        if result["evaluation_current"] else None,
        "task_comparison_complete": task_complete,
        "n_additional_failed_tasks": len(additional_tasks) if task_complete else None,
        "n_observed_additional_failed_tasks": len(additional_tasks)
        if result["evaluation_current"] else None,
        "interpretation": (
            "Search signals, not causal validation or extra optimization metrics. Recall uses "
            "the stated evaluation scope; match/task counts use saved positive records and "
            "are lower bounds unless their inventories are complete. Unknown task identities "
            "are counted separately, never as distinct tasks. Observed additional attempts "
            "are absent from the other saved matches, not necessarily new coverage: missing "
            "or sampled comparison records can overstate that difference. Exact additional "
            "coverage is reported only for complete, comparable census inventories. "
            "Repeated attempts of one task do not establish cross-datapoint generalization. "
            "Single-task findings remain useful diagnostics; seek the same mechanism on "
            "another metadata-identified task before presenting a general detector."
        ),
    })
    return result


def seed_union_coverage(rt: Any, fn: Any, seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare an independently evaluated parent with its recorded seed revisions.

    The seed union is a coverage baseline, never an executable dependency or an
    inherited causal judgment. Partial inventories show observed differences only.
    """
    from autofa.metrics.library_coverage import library_failure_coverage
    from autofa.objectives import discovery_strategy

    inventory = cached_task_inventory(rt)
    parent = _snapshot(rt, fn, inventory)
    snapshots, stale = [], []
    for seed in seeds:
        source = rt.library.get(seed["name"])
        snapshot = _snapshot(rt, source, inventory) if source is not None else None
        if (snapshot is None or not snapshot["evaluation_current"]
                or snapshot["claim_hash"] != seed["claim_hash"]):
            stale.append(seed["name"])
        else:
            snapshots.append(snapshot)
    available = bool(seeds and not stale and parent["evaluation_current"])
    complete = bool(available and parent["match_inventory_complete"]
                    and all(s["match_inventory_complete"] for s in snapshots))
    task_complete = bool(complete and parent["matched_task_inventory_complete"]
                         and all(s["matched_task_inventory_complete"] for s in snapshots))
    seed_ids = set().union(*(set(s["known_matched_failed_item_ids"]) for s in snapshots))
    parent_ids = set(parent["known_matched_failed_item_ids"])
    task_map = inventory["task_by_failed_item_id"]

    def tasks(ids):
        return {task_map[item_id] for item_id in ids if task_map.get(item_id) is not None}

    seed_tasks, parent_tasks = tasks(seed_ids), tasks(parent_ids)
    added, missing = parent_ids - seed_ids, seed_ids - parent_ids
    added_tasks, missing_tasks = parent_tasks - seed_tasks, seed_tasks - parent_tasks
    coverage = library_failure_coverage(
        rt.library, task_map, run_id=rt.run.run_id,
        objective_strategy=(discovery_strategy(rt.config) if hasattr(rt, "config") else None),
        behavior_strictness=(strictness_for_config(rt.config) if hasattr(rt, "config") else None),
        include_item_ids=True, task_by_item_id=task_map,
    )
    reviewed = next((row for row in coverage["detectors"] if row["name"] == fn.name), {})
    confirmed = set(reviewed.get("review_confirmed_failed_item_ids") or [])
    return {
        "seed_names": [seed["name"] for seed in seeds],
        "seed_claims_current": bool(seeds and not stale), "stale_seed_names": stale,
        "parent_evaluation_current": parent["evaluation_current"],
        "comparison_complete": complete, "task_comparison_complete": task_complete,
        "comparison_scope": ("unavailable_seed_revision" if stale else
                             "unavailable_evaluation" if not available else
                             "full_resolved_census" if complete else "partial_observed_matches"),
        "seed_union_matched_failed": len(seed_ids) if available else None,
        "seed_union_matched_failed_tasks": len(seed_tasks) if available else None,
        "parent_matched_failed": len(parent_ids) if parent["evaluation_current"] else None,
        "parent_matched_failed_tasks": len(parent_tasks) if parent["evaluation_current"] else None,
        "preserved_seed_matches": len(parent_ids & seed_ids) if available else None,
        "seed_matches_not_observed_in_parent": len(missing) if available else None,
        "additional_matched_failed": len(added) if available else None,
        "additional_matched_failed_tasks": len(added_tasks) if available else None,
        "n_additional_failed_attempts": len(added) if complete else None,
        "n_additional_failed_tasks": len(added_tasks) if task_complete else None,
        "n_lost_failed_attempts": len(missing) if complete else None,
        "n_lost_failed_tasks": len(missing_tasks) if task_complete else None,
        "n_review_confirmed_additional_failed": len(confirmed & added) if available else None,
        "n_review_confirmed_additional_tasks": len(tasks(confirmed) - seed_tasks)
        if available else None,
        "interpretation": (
            "Compare the parent with the union of the recorded seed revisions, counting "
            "overlapping matches once. Observed differences are not established gains or "
            "losses when either side is sampled, unresolved or missing positive records. "
            "Exact differences require complete comparable censuses; exact task differences "
            "also require known task identities. A missing or changed seed makes the baseline "
            "unavailable until its provenance is recorded again. Extra repeats of an existing "
            "task do not establish broader task coverage. Seed judgments never validate the "
            "parent: reviewed counts use only its own current per-item claim confirmations, "
            "not necessarily untouched confirmation evidence. Inspect new parent matches, "
            "recovered incidents and competing causes; retain useful seeds while testing "
            "whether the shared mechanism generalizes beyond their observed matches."
        ),
    }
