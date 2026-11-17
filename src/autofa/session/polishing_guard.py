# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep final polishing focused on the detectors admitted to the closing phase."""

from __future__ import annotations

import json
from typing import Any


def active_polishing(rt: Any) -> dict | None:
    """Read the current closing phase without creating or advancing workflow state."""
    from autofa.session.round_validation import active_round_refinement

    state = active_round_refinement(rt)
    if state is None:
        # Saving the new round cursor and its continuation receipt spans two
        # files. A process can exit between them; never reopen discovery merely
        # because the previous closing exchange still carries the old cursor.
        try:
            receipt = json.loads((rt.run.run_dir() / "round_validation.json").read_text())
            previous = receipt.get("refinement") if isinstance(receipt, dict) else None
        except (OSError, TypeError, ValueError):
            previous = None
        if (isinstance(previous, dict) and previous.get("version") == 1
                and previous.get("mode") == "polishing"
                and previous.get("run_id") == rt.run.run_id
                and previous.get("round") == rt.run.round - 1
                and previous.get("status") in {
                    "awaiting_response", "validating_response", "complete",
                }):
            state = {
                **previous, "round": rt.run.round, "status": "awaiting_response",
                "polishing_id": f"{rt.run.run_id}:{rt.run.round}", "transition_pending": True,
            }
    if not state or state.get("mode") != "polishing":
        return None
    names = state.get("frozen_names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        return None
    return state


def polishing_repair_id(rt: Any, name: str) -> str | None:
    """A persisted phase identity selects a bounded allowance; it never resets history."""
    phase = active_polishing(rt)
    if phase and name in phase["frozen_names"]:
        identifier = phase.get("polishing_id")
        if isinstance(identifier, str) and identifier:
            return identifier
    return None


def polishing_tool_guard(rt: Any, name: str, args: dict[str, Any]) -> dict | None:
    """Reject fresh discovery mutations while retaining evidence and repair tools."""
    phase = active_polishing(rt)
    if phase is None:
        return None
    frozen = set(phase["frozen_names"])
    reason = None
    if name in {"library_load", "clear_library", "retain_as_helper"}:
        reason = "Final polishing preserves the admitted library and its detector identities."
    elif name == "family_discovery" and args.get("action", "status") != "status":
        reason = "Final polishing repairs the existing library; new family exploration is closed."
    elif name == "plan_function_repair" and args.get("action") == "split_detector":
        reason = "Final polishing revises existing detectors and does not create split children."
    elif name in {
        "submit_function", "restore_function_version", "restore_failure_mode",
        "plan_function_repair", "defer_function_repair", "judge_function", "screen_function",
        "remove_function",
    } and args.get("name") not in frozen:
        reason = "This detector was not part of the library admitted to final polishing."
    elif name == "evaluate_functions":
        names = args.get("names")
        requested = set(names if names is not None else (fn.name for fn in rt.library.functions))
        if requested - frozen:
            reason = "Final polishing evaluates only the admitted detector versions."
    if reason is None:
        return None
    return {
        "ok": False, "error": reason, "phase": "polishing", "target_grade": "GOOD",
        "frozen_names": sorted(frozen),
        "next_action": (
            "Inspect get_function_feedback for an admitted detector. Repair it toward Good, "
            "restore a supported saved version, or record why it should be retained/deferred. "
            "Use advance_round(validation_response=...) to submit the polishing outcome; "
            "the runtime owns validation and the completion decision."
        ),
    }
