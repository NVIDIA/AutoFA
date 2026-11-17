# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist the existing-detector repair exchange after full validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from autofa.core.dependencies import review_source
from autofa.metrics.claim_contract import claim_hash
from autofa.util.atomic import atomic_write_json


def current_claims(rt: Any) -> dict[str, str]:
    return {fn.name: claim_hash(
        source=review_source(fn), description=fn.description or "",
        mechanism=fn.mechanism or fn.description or fn.name,
    ) for fn in rt.library.functions}


def prepare_polishing(
    rt: Any, *, before: dict, previous_scores: dict, previous: dict | None = None,
) -> dict:
    """Freeze identities and retain all validated controls across repair revisions."""
    from autofa.session.validation_controls import collect_full_validation_controls

    previous = previous if (previous or {}).get("mode") == "polishing" else {}
    controls = deepcopy(previous.get("controls") or {})
    names = previous.get("frozen_names", [fn.name for fn in rt.library.functions])
    for fn in rt.library.functions:
        if fn.name not in names:
            continue
        collected = collect_full_validation_controls(
            rt, fn, previous_scores=previous_scores.get(fn.name),
            previous_controls=controls.get(fn.name) if controls.get(fn.name, {}).get("available")
            else None,
        )
        if collected.get("available") or not controls.get(fn.name, {}).get("available"):
            controls[fn.name] = collected
        else:
            # A skipped candidate or interrupted review cannot erase the previous
            # full-validation regression set. Its executor verifies corpus identity.
            controls[fn.name] = {**controls[fn.name], "refresh_limitation": collected.get("reason")}
    from autofa.session.polishing_progress import recovery_work

    work = previous.get("recovery_work") or recovery_work(rt, names, controls)
    return {
        **previous, "version": 1, "mode": "polishing", "target_grade": "GOOD",
        "run_id": rt.run.run_id, "round": rt.run.round, "status": "awaiting_response",
        "polishing_id": f"{rt.run.run_id}:{rt.run.round}", "frozen_names": names,
        "evaluated_claims": current_claims(rt), "controls": controls,
        "baseline_grades": previous.get("baseline_grades", before),
        "recovery_work": work,
        "recovery_initial_work": previous.get("recovery_initial_work", sorted(work["work"])),
    }


def public_validation_receipt(receipt: dict) -> dict:
    """Keep feedback compact; the paged feedback tool exposes every retained case."""
    result = {**receipt}
    if state := receipt.get("refinement"):
        hidden = {"controls", "recovery_work", "recovery_initial_work"}
        result["refinement"] = {key: value for key, value in state.items() if key not in hidden}
        if recovery := state.get("legacy_recovery"):
            result["refinement"]["legacy_recovery"] = {
                key: recovery[key] for key in ("version", "validation_pending", "boundary")
                if key in recovery
            }
        result["refinement"]["regression_sets"] = {
            name: {"available": controls.get("available", False),
                   "complete": controls.get("complete", False),
                   "n": len(controls.get("item_ids") or []),
                   "panel_n": len(controls.get("panel_item_ids") or []),
                   "evidence_action": f"get_function_feedback(name={name!r})"}
            for name, controls in (state.get("controls") or {}).items()
        }
    return result


def continue_polishing(rt: Any, receipt: dict) -> dict:
    """Carry the frozen library and tests into a runtime-authorized extra round."""
    from autofa.session.polishing_progress import recovery_work, remember_recovery_work

    old = receipt["refinement"]
    state = {**old, "status": "awaiting_response", "round": rt.run.round,
             "polishing_id": f"{rt.run.run_id}:{rt.run.round}",
             "evaluated_claims": current_claims(rt),
             "recovery_work": remember_recovery_work(
                 recovery_work(rt, old["frozen_names"], old.get("controls")),
                 old.get("recovery_work"))}
    for key in ("response", "changed_detectors", "removed_detectors", "retained_detectors",
                "transition_pending", "recovery_attempt"):
        state.pop(key, None)
    result = {**receipt, "refinement": state}
    atomic_write_json(rt.run.run_dir() / "round_validation.json", result)
    return result


def unchecked_revisions(rt: Any, state: dict) -> list[str]:
    """Don't jump straight to a full scan when a saved edit has not been tested."""
    from autofa.session.polishing_evaluation import current_polishing_evaluation

    claims = current_claims(rt)
    pending = []
    for fn in rt.library.functions:
        if claims[fn.name] == state.get("evaluated_claims", {}).get(fn.name):
            continue
        controls = state.get("controls", {}).get(fn.name) or {}
        if not controls.get("available"):
            continue
        result = current_polishing_evaluation(rt, fn, controls)
        if not result or not result.get("complete"):
            pending.append(fn.name)
    return pending
