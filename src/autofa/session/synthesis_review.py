# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose existing review evidence for a synthesized detector without new scans."""

from __future__ import annotations

import json
from typing import Any

from autofa.session.family_coverage import cached_task_inventory, candidate_coverage


def synthesis_review_strata(rt: Any, name: str, fire_ids: set[str]) -> dict | None:
    """Use recorded seed revisions and source identities, never item-ID heuristics.

    These are sampling priorities, not inherited judgments. A task absent from
    partial seed records is only an observed difference, not exact new coverage.
    """
    try:
        return _synthesis_review_strata(rt, name, fire_ids)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Sampling priorities are optional. Damaged/stale workflow metadata must
        # not prevent ordinary review or silently claim a complete baseline.
        return {"baseline_scope": "unavailable_synthesis_metadata", "strata": {},
                "reason": f"Synthesis review priorities unavailable: {exc}"}


def _synthesis_review_strata(rt: Any, name: str, fire_ids: set[str]) -> dict | None:
    from autofa.session.family_discovery import _seed_provenance, family_detector_name

    path = rt.run.run_dir() / "family_discovery.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    surveys: list[dict] = []

    def collect(node):
        if not isinstance(node, dict):
            raise ValueError("Family history must contain objects")
        for old in node.get("history") or []:
            collect(old)
        for old in node.get("comparisons") or []:
            collect(old)
        if survey := node.get("survey"):
            if not isinstance(survey, dict):
                raise ValueError("Family survey must be an object")
            if family_detector_name(survey) == name:
                surveys.append(survey)

    collect(state)
    if not surveys or not (seeds := surveys[-1].get("seed_functions")):
        return None
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(not isinstance(seed, dict) or not isinstance(seed.get("name"), str)
                   or not isinstance(seed.get("claim_hash"), str)
                   or not isinstance(seed.get("support_item_ids"), list) for seed in seeds)):
        raise ValueError("Synthesized family needs recorded seed revisions and support IDs")
    inventory = cached_task_inventory(rt)
    task_map = inventory["task_by_failed_item_id"]
    snapshots = []
    stale = set(_seed_provenance(rt, seeds)["stale_seed_functions"])
    for seed in seeds:
        fn = rt.library.get(seed["name"])
        snapshot = candidate_coverage(rt, fn) if fn is not None else None
        if (snapshot is None or not snapshot["evaluation_current"]
                or snapshot["claim_hash"] != seed["claim_hash"]
                or not set(seed.get("support_item_ids") or []) <= set(
                    snapshot["known_matched_failed_item_ids"])):
            stale.add(seed["name"])
        else:
            snapshots.append(snapshot)
    context = {
        "seed_names": [seed["name"] for seed in seeds],
        "stale_seed_names": sorted(stale),
        "baseline_scope": "unavailable_seed_revision",
        "strata": {},
        "interpretation": (
            "New-task matches are relative to the seed failure matches, not necessarily "
            "untouched evidence. Partial seed inventories establish observed differences "
            "only. The parent needs its own causal review of every sampled match."
        ),
    }
    if stale:
        return context
    seed_ids = set().union(*(set(row["known_matched_failed_item_ids"]) for row in snapshots))
    seed_tasks = {task_map[item_id] for item_id in seed_ids if task_map.get(item_id) is not None}
    exact = all(row["matched_task_inventory_complete"] for row in snapshots)
    context.update(
        baseline_scope="complete_seed_task_inventory" if exact else "observed_seed_matches",
        strata={
            "new_task_matches": sorted(item_id for item_id in fire_ids
                                       if task_map.get(item_id) is not None
                                       and task_map[item_id] not in seed_tasks),
            "retained_seed_matches": sorted(fire_ids & seed_ids),
            "passing_matches": sorted(fire_ids & set(inventory["passed_item_ids"])),
        },
    )
    return context
