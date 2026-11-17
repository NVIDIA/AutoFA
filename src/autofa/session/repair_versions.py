# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain exact scored detector versions without creating repair attempts."""

from __future__ import annotations

from typing import Any

from autofa.session import repair_workflow


def collect_versions(
    data: dict, name: str, current_function: dict | None = None,
) -> dict[str, dict]:
    """Read scored snapshots, including histories written before version retention.

    A version identifies the literal claim, source and frozen dependencies, not
    its mutable evaluation. Later snapshots of the same version replace earlier
    scores; the current library entry wins when it has been evaluated.
    """
    versions: dict[str, dict] = {}

    def add(function: Any, expected_version: str | None = None) -> None:
        if not isinstance(function, dict) or function.get("name") != name:
            return
        scores = function.get("scores")
        if not isinstance(scores, dict) or scores.get("evaluated") is not True:
            return
        fingerprint = repair_workflow._version(function)
        if expected_version is not None and expected_version != fingerprint:
            raise ValueError("Saved detector version does not match its snapshot")
        versions[fingerprint] = repair_workflow._copy(function)

    # Search parent histories as well: a split child need not own a workflow.
    for state in (data.get("repairs") or {}).values():
        add(state.get("baseline"), state.get("baseline_version_hash"))
        for plan in state.get("history") or []:
            add(plan.get("before"), plan.get("before_version_hash"))
            add(plan.get("after"), plan.get("after_version_hash"))
            for candidate in (plan.get("candidates") or {}).values():
                fingerprint = candidate.get("after_version_hash")
                add(candidate.get("after"), fingerprint)
                add(candidate.get("after_validation"), fingerprint)
    for fingerprint, function in ((data.get("versions") or {}).get(name) or {}).items():
        add(function, fingerprint)
    add(current_function)
    return versions


def archive_scored_version(run_dir, function: dict) -> None:
    """Archive before replacement; never change history, budgets or exposure IDs."""
    name = function.get("name")
    scores = function.get("scores")
    if not isinstance(name, str) or not name:
        raise ValueError("Function name is required")
    if not isinstance(scores, dict) or scores.get("evaluated") is not True:
        return
    snapshot = repair_workflow._copy(function)
    fingerprint = repair_workflow._version(snapshot)
    with repair_workflow._ledger(run_dir, write=True) as data:
        data.setdefault("versions", {}).setdefault(name, {})[fingerprint] = snapshot
