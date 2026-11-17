# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable review exposure history and task-group confirmation reservations.

Reservations isolate future reviews from recorded discovery and judge exposure.
External file reads cannot be observed and no stronger independence is claimed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from autofa.util.atomic import atomic_write_json

SCHEMA_VERSION = 1
_LOCK = threading.RLock()
_INCOMPLETE_RUNS: set[Path] = set()
_LOG = logging.getLogger(__name__)


def empty_history(*, complete: bool = True) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "history_complete": complete, "detectors": {},
            "discovery_item_ids": [], "discovery_task_groups": {},}


def _ids(values: Any) -> set[str]:
    return {v for v in values or [] if isinstance(v, str) and v}


def _state(history: dict, name: str) -> dict:
    return history.setdefault("detectors", {}).setdefault(name, {
        "reviewed_item_ids": [], "item_task_groups": {}, "confirmation_task_groups": [],
        "allocations": {},
    })


def task_group(item: dict[str, Any]) -> str | None:
    """Use actual task metadata; opaque item IDs never establish task independence."""
    problem = next((item.get(key) for key in
                    ("problem_id", "task_id", "instance_id", "question_id")
                    if item.get(key) is not None and str(item.get(key)).strip()), None)
    if problem is None:
        return None
    return json.dumps([str(item.get("benchmark") or ""), str(problem)], separators=(",", ":"))


def merge_histories(*histories: dict | None) -> dict:
    """Conservative union, including removed detectors and consumed reservations."""
    result = empty_history()
    for history in histories:
        if not history:
            continue
        if history.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported review evidence history schema")
        result["history_complete"] &= bool(history.get("history_complete", False))
        result["discovery_item_ids"] = sorted(_ids(result["discovery_item_ids"])
                                               | _ids(history.get("discovery_item_ids")))
        for item_id, group in history.get("discovery_task_groups", {}).items():
            if (item_id in result["discovery_task_groups"]
                    and result["discovery_task_groups"][item_id] != group):
                raise ValueError("Conflicting task identities in discovery exposure history")
            result["discovery_task_groups"][item_id] = group
        for name, incoming in history.get("detectors", {}).items():
            target = _state(result, name)
            for key in ("reviewed_item_ids", "confirmation_task_groups", "completed_allocations"):
                target[key] = sorted(_ids(target.get(key)) | _ids(incoming.get(key)))
            for item_id, group in incoming.get("item_task_groups", {}).items():
                if not isinstance(item_id, str) or not isinstance(group, str):
                    raise ValueError("Invalid review evidence task identity")
                if (item_id in target["item_task_groups"]
                        and target["item_task_groups"][item_id] != group):
                    raise ValueError("Conflicting task identities in review evidence history")
                target["item_task_groups"][item_id] = group
            for key, allocation in incoming.get("allocations", {}).items():
                if key in target["allocations"] and target["allocations"][key] != allocation:
                    raise ValueError("Conflicting frozen review evidence allocations")
                target["allocations"][key] = deepcopy(allocation)
    return result


@contextmanager
def _ledger(run_dir: Path, *, write: bool = False):
    directory = Path(run_dir)
    path = directory / "review_evidence.json"
    directory.mkdir(parents=True, exist_ok=True)
    with _LOCK, (directory / ".review_evidence.lock").open("a") as lock:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text()) if path.exists() else empty_history()
            if data.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported review evidence history schema")
            if directory.resolve() in _INCOMPLETE_RUNS:
                data["history_complete"] = False
            yield data
            if write:
                atomic_write_json(path, data)
                _INCOMPLETE_RUNS.discard(directory.resolve())
        finally:
            if fcntl:
                fcntl.flock(lock, fcntl.LOCK_UN)


def function_review_history(functions: list[Any], *, complete: bool = True) -> dict:
    """Harvest current scores without forgetting older versions or dropped functions."""
    history = empty_history(complete=complete)
    for function in functions:
        fn = function if isinstance(function, dict) else function.to_dict()
        state = _state(history, fn["name"])
        scores = fn.get("scores") or {}
        state["reviewed_item_ids"] = sorted({
            str(row["item_id"]) for row in scores.get("behavior_verdicts", [])
            if isinstance(row, dict) and row.get("item_id")
        } | _ids((scores.get("repair_validation") or {}).get("excluded_item_ids")))
    return history



def repair_review_history(repairs: list[dict]) -> dict:
    """Preserve the full exclusion union, including archived split children."""
    history = empty_history()
    for repair in repairs:
        names = {repair["name"]} | {
            name for plan in repair.get("history", []) for name in plan.get("candidates", {})
        }
        for name in names:
            state = _state(history, name)
            state["reviewed_item_ids"] = sorted(_ids(state["reviewed_item_ids"])
                                                 | _ids(repair.get("reviewed_item_ids")))
    return history


def import_seed_review_history(
    library: Any, seed_path: Path, *, runs_root: Path | None = None,
) -> None:
    """Read available local seed ledgers without writing to a possibly active source run.

    Portable history is authoritative for remote Continue. For older local libraries,
    recover any available exclusion ledger; absent historic exposure remains unknown.
    """
    candidates = {Path(seed_path).parent}
    if runs_root is not None:
        root = Path(runs_root).resolve()
        for fn in library.functions:
            run_id = (fn.scores or {}).get("evaluation_run_id")
            if isinstance(run_id, str) and run_id and Path(run_id).name == run_id:
                candidate = (root / run_id).resolve()
                if candidate.parent == root:
                    candidates.add(candidate)
    for directory in sorted(candidates):
        review_path = directory / "review_evidence.json"
        if review_path.is_file():
            library.review_history = merge_histories(
                library.review_history, json.loads(review_path.read_text()),
            )
        repair_path = directory / "detector_repairs.json"
        if repair_path.is_file():
            data = json.loads(repair_path.read_text())
            if data.get("schema_version") != "detector-repair-v1":
                raise ValueError("Unsupported seed repair workflow schema")
            library.review_history = merge_histories(
                library.review_history,
                repair_review_history(list(data.get("repairs", {}).values())),
            )

def sync_review_history(
    library: Any, run_dir: Path, *, repairs: list[dict] | None = None,
    persist: Callable[[], None] | None = None,
) -> dict:
    """Validate the latest ledger before optionally persisting the library under its lock."""
    from autofa.session.repair_workflow import list_repairs

    # Version selection already holds the repair ledger's interprocess lock.
    # Its supplied snapshot avoids reacquiring that non-reentrant file lock.
    repair_history = repair_review_history(list_repairs(run_dir) if repairs is None else repairs)
    with _ledger(run_dir) as data:
        merged = merge_histories(data, library.review_history, repair_history,
                                 function_review_history(library.functions))
        # Exposure history is conservative and may survive a failed library
        # save. Write it first so no ledger I/O can reject an operation after
        # the candidate library has already replaced the durable checkpoint.
        atomic_write_json(Path(run_dir) / "review_evidence.json", merged)
        _INCOMPLETE_RUNS.discard(Path(run_dir).resolve())
        previous = library.review_history
        library.review_history = deepcopy(merged)
        if persist is not None:
            try:
                persist()
            except BaseException:
                library.review_history = previous
                # A rejected import/submission must not replace portable
                # history either. Keep the ledger lock through rollback.
                try:
                    atomic_write_json(Path(run_dir) / "review_evidence.json", data)
                except Exception:
                    # If storage also prevents rollback, the durable union is
                    # conservative; retain the original save exception.
                    _LOG.warning("Could not restore exposure history after a failed library save",
                                 exc_info=True)
                raise
        return deepcopy(merged)


def review_context(
    run_dir: Path, name: str, *, related_names: list[str] | tuple[str, ...] = (),
) -> dict:
    with _ledger(run_dir) as history:
        states = list(history.get("detectors", {}).values())
        ids = (_ids(history.get("discovery_item_ids"))
               | set().union(*(_ids(state.get("reviewed_item_ids")) for state in states)))
        groups = (set(history.get("discovery_task_groups", {}).values())
                  | {group for state in states for item_id, group in
                     state.get("item_task_groups", {}).items() if item_id in ids})
        return {
            "reviewed_item_ids": sorted(ids), "reviewed_task_groups": sorted(groups),
            "reserved_confirmation_task_groups": sorted(set().union(*(
                _ids(state.get("confirmation_task_groups")) for state in states))),
            "history_complete": bool(history.get("history_complete", False)),
        }


def record_discovery_exposure(run_dir: Path, items_or_ids: list[dict[str, Any] | str]) -> None:
    """Record structured tool exposure, without guessing task keys from opaque IDs."""
    with _ledger(run_dir, write=True) as history:
        ids = _ids(history.get("discovery_item_ids"))
        groups = history.setdefault("discovery_task_groups", {})
        for item in items_or_ids:
            item_id = item if isinstance(item, str) else item.get("_item_id", item.get("item_id"))
            if not isinstance(item_id, str) or not item_id:
                continue
            ids.add(item_id)
            if isinstance(item, dict) and (group := task_group(item)):
                if item_id in groups and groups[item_id] != group:
                    raise ValueError("Task identity changed for exposed discovery evidence")
                groups[item_id] = group
        history["discovery_item_ids"] = sorted(ids)


def mark_exposure_history_incomplete(run_dir: Path) -> None:
    """Keep subsequent confirmations limited even if the ledger cannot be written yet."""
    with _LOCK:
        _INCOMPLETE_RUNS.add(Path(run_dir).resolve())
        with _ledger(run_dir, write=True) as history:
            history["history_complete"] = False


def _order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _diverse_ids(
    items: dict[str, dict], eligible: set[str], *, target_n: int, seed: int,
    selected: list[str] | None = None,
    reviewed: set[str] | None = None,
) -> list[str]:
    groups: dict[str, list[str]] = {}
    reviewed = reviewed or set()
    for item_id in eligible:
        group = task_group(items[item_id]) or "unknown-task"
        groups.setdefault(group, []).append(item_id)
    for ids in groups.values():
        ids.sort(key=lambda value: (value in reviewed, _order(value, seed)))
    represented = {task_group(items[item_id]) or "unknown-task" for item_id in selected or []}
    ordered_groups = sorted(groups, key=lambda value: (
        value in represented, all(item_id in reviewed for item_id in groups[value]),
        _order(value, seed),
    ))
    selected = []
    while ordered_groups and len(selected) < target_n:
        for group in ordered_groups:
            if groups[group] and len(selected) < target_n:
                selected.append(groups[group].pop(0))
        ordered_groups = [group for group in ordered_groups if groups[group]]
    return selected


def _stratified_ids(
    items: dict[str, dict], eligible: set[str], strata: dict[str, set[str]], *,
    reviewed: set[str], target_n: int, seed: int,
) -> list[str]:
    """Represent new tasks, retained support and passing matches before filling."""
    selected: list[str] = []
    for ids in strata.values():
        if len(selected) >= target_n:
            break
        pool = (ids & eligible) - set(selected)
        if pool:
            selected += _diverse_ids(
                items, pool, target_n=1, seed=seed, selected=selected, reviewed=reviewed,
            )
    selected += _diverse_ids(items, eligible - set(selected), seed=seed, selected=selected,
                             reviewed=reviewed, target_n=target_n - len(selected))
    return selected


def _outcome_diagnostics(
    items: dict[str, dict], selected: list[str], *, outcome_ids: dict[str, list[str]],
    reserved_ids: set[str], reviewed: set[str], seed: int,
) -> tuple[list[str], dict]:
    """Inspect an absent outcome without changing the cohort used for adherence."""
    outcomes = {key: _ids(outcome_ids.get(key)) & items.keys() for key in ("failed", "passed")}
    if outcomes["failed"] & outcomes["passed"]:
        raise ValueError("Review outcome identities must not classify an item both ways")
    represented = {task_group(items[item_id]) for item_id in selected}
    diagnostic_ids = []
    strata = {}
    for key, ids in outcomes.items():
        sampled = ids & set(selected)
        eligible = {
            item_id for item_id in ids - set(selected) - reserved_ids
            if task_group(items[item_id]) is not None
            and task_group(items[item_id]) not in represented
        }
        added = []
        if ids and not sampled and eligible and not diagnostic_ids:
            added = _diverse_ids(items, eligible, target_n=1, seed=seed, reviewed=reviewed)
            diagnostic_ids += added
            represented.update(task_group(items[item_id]) for item_id in added)
        strata[key] = {
            "matched_n": len(ids), "scored_review_n": len(sampled),
            "diagnostic_item_ids": added,
            "status": ("represented" if sampled else "supplemental" if added else
                       "no_matches" if not ids else
                       "budget_limited" if eligible and diagnostic_ids else "unavailable"),
        }
    return diagnostic_ids, {
        "purpose": "outcome_coverage", "item_ids": diagnostic_ids,
        "excluded_from_adherence": True, "excluded_from_confirmed_support": True,
        "strata": strata,
        "reason": (
            "Outcome-selected controls diagnose missed failure/pass contexts separately. "
            "They do not change the original adherence sample or estimate. Reservations and "
            "distinct-task boundaries are preserved; unavailable outcomes remain unreviewed."
        ),
    }


def _allocation_key(run_dir: Path, phase: str, review_key: str) -> str:
    # Continue preserves exposure, but only the original run owns its operation.
    return f"{phase}:{Path(run_dir).name}:{review_key}"


def find_review_allocation(run_dir: Path, name: str, *, review_key: str, phase: str) -> dict | None:
    """Find a same-run operation before scanning can change its automatic target."""
    with _ledger(run_dir) as history:
        state = history.get("detectors", {}).get(name, {})
        key = _allocation_key(run_dir, phase, review_key)
        allocation = (state.get("allocations", {}).get(key + ":judge-expansion")
                      or state.get("allocations", {}).get(key))
        if (allocation and not allocation.get("item_ids")
                and not allocation.get("diagnostic_item_ids")
                and key not in state.get("completed_allocations", [])):
            # No judge could have started. Later known matches may create the
            # first actual cohort; an empty availability result is not frozen.
            return None
        if not allocation or allocation.get("origin_run_id") != Path(run_dir).name:
            return None
        if phase == "confirmation" and not allocation.get("sample_info", {}).get(
            "independent_task_n",
        ):
            return None
        return {**deepcopy(allocation),
                "operation_completed": key in state.get("completed_allocations", [])}


def complete_review_allocation(run_dir: Path, name: str, *, review_key: str, phase: str) -> None:
    """Persist completion separately from the immutable reservation and exposure."""
    with _ledger(run_dir, write=True) as history:
        state = _state(history, name)
        key = _allocation_key(run_dir, phase, review_key)
        if key not in state["allocations"]:
            raise ValueError("Review allocation is missing its original operation")
        state["completed_allocations"] = sorted(_ids(state.get("completed_allocations")) | {key})


def allocate_review(  # noqa: C901
    run_dir: Path, name: str, fire_items: list[dict[str, Any]], *, target_n: int,
    seed: int, review_key: str, phase: str = "development",
    related_names: list[str] | tuple[str, ...] = (),
    coverage_review: dict | None = None,
    support_review: dict | None = None,
    outcome_ids: dict[str, list[str]] | None = None,
    diagnostic_items: list[dict] | None = None,
) -> dict:
    """Reserve feedback-independent tasks, then freeze exposure before running a judge.

    A complete fire pool gives the strongest allocation. Callers using a reservoir
    must disclose that scope separately. A frozen allocation is reused on resume;
    a changed claim/policy must supply a different review_key.
    """
    if phase not in {"development", "confirmation"}:
        raise ValueError("Unknown review evidence phase")
    if support_review is not None and phase != "development":
        raise ValueError("Support expansion is diagnostic development evidence only")
    if diagnostic_items and (phase != "development" or support_review is not None):
        raise ValueError("Off-panel controls are ordinary development diagnostics only")
    if type(target_n) is not int or target_n < 0 or not review_key:
        raise ValueError("Review allocation needs a nonnegative target and a stable review key")
    items = {item.get("_item_id"): item for item in fire_items}
    if (len(items) != len(fire_items)
            or any(not isinstance(item_id, str) or not item_id for item_id in items)):
        raise ValueError("Review allocation requires unique nonempty item IDs")
    diagnostics = {item.get("_item_id"): item for item in diagnostic_items or []}
    if (len(diagnostics) != len(diagnostic_items or []) or len(diagnostics) > 3
            or any(not isinstance(key, str) or not key for key in diagnostics)
            or set(diagnostics) & items.keys()):
        raise ValueError("Off-panel diagnostics require up to three distinct external items")
    retained = _ids((support_review or {}).get("retained_item_ids"))
    if not retained <= items.keys():
        raise ValueError("Current review evidence is missing from the fire bucket; "
                         "restore it before expanding support")
    strata = {key: _ids(values) & items.keys()
              for key, values in (coverage_review or {}).get("strata", {}).items()}
    with _ledger(run_dir, write=True) as history:
        state = _state(history, name)
        allocation_key = _allocation_key(run_dir, phase, review_key)
        if allocation_key in state["allocations"]:
            previous = (state["allocations"].get(allocation_key + ":judge-expansion")
                        or state["allocations"][allocation_key])
            if (not previous.get("item_ids") and not previous.get("diagnostic_item_ids")
                    and allocation_key not in state.get("completed_allocations", [])):
                del state["allocations"][allocation_key]
        if allocation_key in state["allocations"]:
            previous = (state["allocations"].get(allocation_key + ":judge-expansion")
                        or state["allocations"][allocation_key])
            frozen_ids = set(previous["item_ids"]) | set(previous.get("diagnostic_item_ids", []))
            all_items = {**items, **diagnostics}
            if not frozen_ids <= set(all_items):
                raise ValueError("Frozen review evidence is missing; restore its original dataset")
            for item_id in frozen_ids:
                if task_group(all_items[item_id]) != state["item_task_groups"].get(item_id):
                    raise ValueError("Task identity changed for frozen review evidence")
            return deepcopy(previous)
        for key in related_names:
            _state(history, key)
        # Feedback from any detector can inform later hypotheses; renaming must
        # not make a reviewed case fresh. Reserve confirmation groups globally too.
        related = list(history["detectors"].values())
        reviewed = (_ids(history.get("discovery_item_ids"))
                    | set().union(*(_ids(value.get("reviewed_item_ids")) for value in related)))
        for item_id, item in items.items():
            if group := task_group(item):
                old = state["item_task_groups"].get(item_id)
                if old is not None and old != group:
                    raise ValueError("Task identity changed for previously seen evidence")
                if item_id in reviewed:
                    state["item_task_groups"][item_id] = group
        known_groups = {item_id: task_group(item) for item_id, item in items.items()}
        reviewed_groups = (set(history.get("discovery_task_groups", {}).values())
                           | {group for value in related
                              for item_id, group in value.get("item_task_groups", {}).items()
                              if item_id in reviewed})
        known_prior_ids = (set(history.get("discovery_task_groups", {}))
                           | {item_id for value in related
                              for item_id in value.get("item_task_groups", {})})
        unknown_prior_n = len(reviewed - known_prior_ids)
        untouched = {item_id for item_id, group in known_groups.items()
                     if item_id not in reviewed and group is not None
                     and group not in reviewed_groups}
        available_groups = {known_groups[item_id] for item_id in untouched}
        reserved_groups = set().union(*(_ids(value.get("confirmation_task_groups"))
                                        for value in related))
        # Keep enough examples for a useful initial review. Tiny single-task
        # detectors remain reviewable, with their lack of independent tasks explicit.
        if phase == "development" and support_review is None and len(available_groups) >= 2:
            n_groups = max(1, min(target_n, len(available_groups) // 5))
            # Another family's reservation may have no overlap with this candidate.
            # Hold enough item capacity for the eventual confirmation when possible,
            # while keeping tasks available for diagnosis (one for a two-task cohort).
            development_groups = min(2, len(available_groups) - 1)
            group_sizes = {group: 0 for group in available_groups}
            for item_id in untouched:
                group_sizes[known_groups[item_id]] += 1
            relevant_reserved = available_groups & reserved_groups
            reserved_capacity = sum(group_sizes[group] for group in relevant_reserved)
            # Preserve a development example of each priority where possible;
            # existing reservations remain untouched. Seed support is often
            # already exposed, so prioritize a new task and a passing control.
            development_keep = set()
            for key in ("new_task_matches", "passing_matches", "retained_seed_matches"):
                groups = {known_groups[item_id] for item_id in strata.get(key, set())
                          if item_id in untouched} - reserved_groups - development_keep
                if groups and len(development_keep) < development_groups:
                    development_keep.add(min(groups, key=lambda value: _order(value, seed)))
            for group in sorted(available_groups - reserved_groups,
                                key=lambda value: (value in development_keep, _order(value, seed))):
                if (len(relevant_reserved) >= len(available_groups) - development_groups
                        or (len(relevant_reserved) >= n_groups and reserved_capacity >= target_n)):
                    break
                reserved_groups.add(group)
                relevant_reserved.add(group)
                reserved_capacity += group_sizes[group]
        state["confirmation_task_groups"] = sorted(reserved_groups)
        reserved_ids = {item_id for item_id, group in known_groups.items()
                        if group in reserved_groups}
        if phase == "development":
            # Exposed evidence remains useful for diagnosis. Prefer unseen tasks,
            # but fill a short sample with reused items and disclose that below.
            eligible = set(items) - reserved_ids
        else:
            eligible = untouched
        preferred = eligible & reserved_ids if phase == "confirmation" else eligible
        if support_review is not None:
            # An explicit survivor review preserves all current findings, then
            # seeks different failed tasks. Previously reserved tasks may supply
            # diagnostic support; their exposure prevents later freshness claims.
            failed = _ids(support_review.get("failed_item_ids")) & items.keys()
            selected = sorted(retained, key=lambda value: _order(value, seed))
            represented = {known_groups[item_id] for item_id in retained & failed
                           if known_groups[item_id] is not None}
            required = support_review["min_failed_tasks"]
            needed = max(0, required - len(represented))
            target_n = max(target_n, len(selected) + needed)
            for pool in (failed - reserved_ids, failed):
                pool = {item_id for item_id in pool - set(selected)
                        if known_groups[item_id] is not None
                        and known_groups[item_id] not in represented}
                added = _diverse_ids(
                    items, pool, target_n=min(needed, len({known_groups[key] for key in pool})),
                    seed=seed, selected=selected, reviewed=reviewed,
                )
                selected += added
                represented.update(known_groups[item_id] for item_id in added)
                needed = max(0, required - len(represented))
            # Keep synthesis controls represented within the remaining budget,
            # including passing matches, rather than only reviewing failures.
            for ids in strata.values():
                if len(selected) >= target_n:
                    break
                if not ids & set(selected):
                    selected += _diverse_ids(
                        items, (ids - reserved_ids) - set(selected), target_n=1,
                        seed=seed, selected=selected, reviewed=reviewed,
                    )
            selected += _diverse_ids(
                items, (set(items) - reserved_ids) - set(selected),
                target_n=max(0, target_n - len(selected)), seed=seed,
                selected=selected, reviewed=reviewed,
            )
            eligible = set(items)
        elif strata:
            # A synthesized claim must be reviewed on retained support too.
            # Reuse is permitted for development, with truthful exposure flags;
            # confirmation never promotes exposed tasks into independent ones.
            pool = set(items) - reserved_ids if phase == "development" else eligible
            selected = _stratified_ids(items, pool, strata, reviewed=reviewed,
                                      target_n=target_n, seed=seed)
        else:
            selected = _diverse_ids(items, preferred, target_n=target_n, seed=seed,
                                    reviewed=reviewed)
            selected += _diverse_ids(items, eligible - set(selected),
                                     target_n=target_n - len(selected), seed=seed,
                                     selected=selected, reviewed=reviewed)
        reused_n = 0
        if not selected and items:
            # Exhaustion never blocks diagnosis or becomes fresh validation.
            fallback = (set(items) - reserved_ids if phase == "development" else set(items))
            previously_exposed = fallback & reviewed
            if previously_exposed:
                fallback = previously_exposed
            selected = (_stratified_ids(items, fallback, strata, reviewed=reviewed,
                                       target_n=target_n, seed=seed) if strata else
                        _diverse_ids(items, fallback, target_n=target_n, seed=seed))
        reused_n = sum(item_id in reviewed for item_id in selected)
        independent = {known_groups[item_id] for item_id in selected
                       if item_id in untouched and known_groups[item_id] is not None}
        unknown = sum(known_groups[item_id] is None for item_id in selected)
        fresh_n = sum(item_id not in reviewed for item_id in selected)
        group_limited = phase == "confirmation" and (
            bool(reused_n or unknown or unknown_prior_n)
            or any(item_id not in untouched for item_id in selected)
        )
        limited = len(selected) < target_n or bool(reused_n) or group_limited
        reason = None
        if limited:
            reason = (f"Reviewed {len(selected)}/{target_n} requested examples; {fresh_n} unseen "
                      f"items across {len(independent)} tasks without recorded discovery "
                      "or review exposure. "
                      "This is limited evidence, not a completed independent confirmation.")
        if phase == "confirmation" and unknown_prior_n:
            reason = ((reason + " ") if reason else "") + (
                f"Task identity is unavailable for {unknown_prior_n} previously exposed items; "
                "different-task independence cannot be established for them."
            )
        if not history.get("history_complete", False):
            reason = ((reason + " ") if reason else "") + (
                "Earlier review history is incomplete; freshness cannot be established."
            )
            limited = phase == "confirmation" or limited
        allocation = {
            "origin_run_id": Path(run_dir).name,
            "item_ids": selected,
            **({"candidate_item_ids": sorted(items)} if phase == "development" else {}),
            "reserved_confirmation_item_ids": sorted(reserved_ids - set(selected)),
            "sample_info": {
                "target_n": target_n, "actual_n": len(selected), "eligible_n": len(eligible),
                "reused_n": reused_n, "fresh_item_n": fresh_n,
                "independent_task_n": len(independent), "unknown_task_n": unknown,
                "unknown_exposure_item_n": unknown_prior_n,
                "limited": limited, "reason": reason, "phase": phase,
                "history_complete": bool(history.get("history_complete", False)),
                "independence_scope": "recorded_discovery_and_judge_exposure",
                "task_group_limited": group_limited,
            },
        }
        diagnostic_ids = []
        if (phase == "development" and selected and support_review is None
                and outcome_ids is not None):
            diagnostic_ids, diagnostic_sample = _outcome_diagnostics(
                items, selected, outcome_ids=outcome_ids, reserved_ids=reserved_ids,
                reviewed=reviewed, seed=seed,
            )
            allocation["diagnostic_item_ids"] = diagnostic_ids
            allocation["sample_info"]["diagnostic_sample"] = diagnostic_sample
        if diagnostics:
            # These author-selected positives must never enter items, candidate_item_ids,
            # representative adherence, or the adaptive expansion's matched pool.
            represented = {known_groups[item_id] for item_id in selected + diagnostic_ids}
            available = {key for key, item in diagnostics.items()
                         if task_group(item) is not None
                         and task_group(item) not in represented | reserved_groups}
            target = min(3, len({task_group(diagnostics[key]) for key in available}))
            added = _diverse_ids(diagnostics, available, target_n=target, seed=seed,
                                 reviewed=reviewed)
            for item_id in added:
                old = state["item_task_groups"].get(item_id)
                if old is not None and old != task_group(diagnostics[item_id]):
                    raise ValueError("Task identity changed for diagnostic evidence")
            diagnostic_ids += added
            known_groups.update({key: task_group(diagnostics[key]) for key in added})
            allocation["diagnostic_item_ids"] = diagnostic_ids
            allocation["sample_info"]["diagnostic_sample"] = {
                "purpose": "known_smoke_matches", "item_ids": diagnostic_ids,
                "off_panel_item_ids": added,
                "excluded_from_adherence": True, "excluded_from_confirmed_support": True,
                "selected_development_evidence": True, "reused_n": len(added),
                "reason": "Up to three verified failed-task smoke positives outside the "
                          "scoring panel. Author-selected controls diagnose the mechanism; "
                          "they do not estimate population coverage or adherence.",
            }
        if support_review is not None:
            allocation["sample_info"]["support_review"] = {
                "min_failed_tasks": required,
                "retained_item_n": len(retained),
                "selected_failed_task_n": len({known_groups[item_id]
                    for item_id in set(selected) & failed if known_groups[item_id] is not None}),
            }
        if coverage_review is not None:
            allocation["sample_info"]["coverage_review"] = {
                **{key: value for key, value in coverage_review.items() if key != "strata"},
                "strata": {key: {
                    "available_n": len(ids), "selected_n": len(ids & set(selected)),
                    "selected_task_n": len({known_groups[item_id]
                                            for item_id in ids & set(selected)
                                            if known_groups[item_id] is not None}),
                } for key, ids in strata.items()},
            }
        if phase == "confirmation" and not independent:
            # No judge will run. This is an availability observation, not an
            # operation or new exposure; a larger corpus may later supply tasks.
            return allocation
        if phase == "development" and not selected and not diagnostic_ids:
            return allocation
        state["allocations"][allocation_key] = deepcopy(allocation)
        exposed = selected + diagnostic_ids
        state["item_task_groups"].update({item_id: known_groups[item_id] for item_id in exposed
                                           if known_groups[item_id] is not None})
        state["reviewed_item_ids"] = sorted(_ids(state["reviewed_item_ids"]) | set(exposed))
        return allocation


def allocate_judge_expansion(
    run_dir: Path, name: str, items: list[dict], *, review_key: str, seed: int, request: dict,
) -> dict:
    """Freeze one diagnostic batch within the original matched pool.

    The child allocation retains the original cohort. Only different tasks are
    added; confirmation reservations and the original operation stay immutable.
    """
    from autofa.metrics.adaptive_review import normalize_sample_request

    request = normalize_sample_request(request)
    if request is None:
        raise ValueError("Additional review requires a specific cross-task evidence question")
    by_id = {item["_item_id"]: item for item in items}
    with _ledger(run_dir, write=True) as history:
        state = _state(history, name)
        key = _allocation_key(run_dir, "development", review_key)
        child_key = key + ":judge-expansion"
        if child_key in state["allocations"]:
            return deepcopy(state["allocations"][child_key])
        original = state["allocations"].get(key)
        if not original or original.get("origin_run_id") != Path(run_dir).name:
            raise ValueError("Expansion requires the original same-run development allocation")
        if not original.get("candidate_item_ids"):
            raise ValueError("Historical review has no frozen matched pool; no automatic expansion")
        candidates = set(original["candidate_item_ids"])
        if set(by_id) != candidates:
            raise ValueError("Original matched pool is missing; restore it before expansion")
        selected = list(original["item_ids"])
        groups = {item_id: task_group(item) for item_id, item in by_id.items()}
        for item_id in selected:
            if groups[item_id] != state["item_task_groups"].get(item_id):
                raise ValueError("Task identity changed for frozen review evidence")
        diagnostic_ids = set(original.get("diagnostic_item_ids") or [])
        represented = {groups[item_id] for item_id in selected}
        represented.update(state["item_task_groups"].get(item_id) for item_id in diagnostic_ids)
        related = list(history["detectors"].values())
        reserved = set().union(*(_ids(value.get("confirmation_task_groups"))
                                 for value in related))
        reviewed = (_ids(history.get("discovery_item_ids"))
                    | set().union(*(_ids(value.get("reviewed_item_ids")) for value in related)))
        reviewed_groups = (set(history.get("discovery_task_groups", {}).values())
                           | {group for value in related for item_id, group in
                              value.get("item_task_groups", {}).items() if item_id in reviewed})
        eligible = {item_id for item_id in candidates - set(selected) - diagnostic_ids
                    if groups[item_id] is not None and groups[item_id] not in represented
                    and groups[item_id] not in reserved}
        initial_target = original["sample_info"]["target_n"]
        extra_n = min(initial_target, len({groups[item_id] for item_id in eligible}))
        added = _diverse_ids(by_id, eligible, target_n=extra_n, seed=seed,
                             selected=selected, reviewed=reviewed)
        sample = deepcopy(original["sample_info"])
        extra_reused = sum(item_id in reviewed for item_id in added)
        sample.update(
            target_n=max(initial_target, len(selected) + len(added)),
            actual_n=len(selected) + len(added),
            reused_n=sample["reused_n"] + extra_reused,
            fresh_item_n=sample["fresh_item_n"] + len(added) - extra_reused,
            independent_task_n=sample["independent_task_n"] + len({groups[item_id]
                for item_id in added if item_id not in reviewed
                and groups[item_id] not in reviewed_groups}),
            limited=True,
            reason="Diagnostic sample expanded once; original findings retained. "
                   "This is not independent confirmation.",
            judge_expansion={
                "request": request, "initial_target_n": initial_target,
                "initial_item_ids": selected, "added_item_ids": added,
                "max_total_n": 2 * initial_target,
                "status": "allocated" if added else "unavailable",
                "reason": "One additional batch on distinct tasks." if added else
                "No additional nonreserved distinct tasks in the original matched pool.",
            },
        )
        allocation = {**deepcopy(original), "item_ids": selected + added, "sample_info": sample}
        state["allocations"][child_key] = deepcopy(allocation)
        state["reviewed_item_ids"] = sorted(_ids(state["reviewed_item_ids"]) | set(added))
        state["item_task_groups"].update({item_id: groups[item_id] for item_id in added})
        return allocation
