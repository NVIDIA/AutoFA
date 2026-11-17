# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-facing round awareness without changing discovery stopping decisions."""

from typing import Any

DISCOVERY_ROUND_GUIDANCE = """A discovery round is a bounded batch of family comparisons:
Precision 1, Balanced 3, Recall 5, using the existing family-comparison budget across both
top-down and bottom-up work. Assessed families, evidence-backed rejections/no_coherent_family,
and evidence-backed deferrals count; a successful detector is not required.
Check family_discovery(action="status") for completed_comparisons, comparison_budget,
remaining_comparisons, phase, exploration_exhausted and next_action. Call advance_round only
after the batch is complete or the tool verifies that the available failed-task source
evidence is exhausted.
Resolve or explicitly defer the current hypothesis and complete any required synthesis
inspection first. A stalled frontier, weak scores, an empty sample or a new angle alone
does not complete a round. Pivot within the same round and refresh evidence with
sample_items(coverage="unconfirmed_tasks", seed=...) or another peek_chunk as needed.
Keep dependent mutations and advance_round sequential: inspect the assessment/mutation
result and recheck pareto_status before requesting the next round. Never batch advance_round
with a mutation it depends on. If advance_round is refused, follow the returned
family_discovery.next_action within the current round. These rules do not override goal_met,
the runtime's configured stall stop, explicit cancellation, or an outer supervisor's
saved lifecycle intent.
"""


def discovery_budget_feedback(
    round_idx: int,
    max_rounds: int,
    *, campaign_policy: dict | None = None,
) -> dict[str, Any]:
    """Describe the current budget and priorities; the driver decides when to stop."""
    if campaign_policy:
        limit = campaign_policy.get("round_limit")
        return {
            "round": round_idx, "max_rounds": max_rounds,
            "rounds_remaining": max(0, limit - round_idx) if limit is not None else None,
            "phase": "explore", "campaign_mode": campaign_policy.get("mode", "improve_until"),
            "message": (
                f"Campaign checkpoint round {round_idx}. The legacy planned round count "
                f"({max_rounds}) is not a consolidation deadline. Continue useful discovery "
                "and refinement while pareto_status permits it; use the agreed time/spend "
                "horizon and any explicit round cap for the final-validation handoff. "
                "Full intermediate checkpoints do not freeze discovery. Finish the current "
                "comparison batch and retain its evidence before advancing."
            ),
        }
    remaining = max(0, max_rounds - round_idx)
    result = {
        "round": round_idx,
        "max_rounds": max_rounds,
        "rounds_remaining": remaining,
    }

    if round_idx > max_rounds:
        phase = "overrun"
        priority = (
            "The planned round boundary has passed. Continue only as authorized by the "
            "existing driver and its unmet-target or plateau policy; prioritize "
            "consolidating viable candidates."
        )
    elif round_idx == max_rounds:
        phase = "boundary"
        priority = (
            "The planned round boundary is reached. Consolidate viable candidates and "
            "check pareto_status for the existing final-validation handoff."
        )
    elif round_idx >= max_rounds - 1:
        phase = "consolidate"
        priority = (
            "You are in the final two planned rounds. Consolidate now: prioritize "
            "promising candidates rather than starting unrelated work when viable "
            "survivors already exist."
        )
    else:
        phase = "explore"
        priority = (
            "Use the remaining exploration rounds to seek broader supported coverage, "
            "compare promising failure mechanisms, and improve useful candidates."
        )

    if phase != "explore":
        priority += (
            " Repair execution faults, complete needed reviews, and preserve supported "
            "seeds. If existing candidates cannot qualify, explore another supported "
            "direction within the driver's remaining authorization."
        )
    message = (
        f"Round budget: current={round_idx}, planned={max_rounds}; "
        f"{remaining} round advances before the planned boundary. "
        "These count discovery rounds, not tool calls or agent turns. "
        "Finish the current comparison batch or verify source-evidence exhaustion before "
        "advancing; pivots and sample refreshes can stay within the same round. "
        + priority
        + " Keep evaluation capped while iterating. At ready_for_final_validation, follow "
        "final_validation_guidance: freeze candidates and run a full scan only if it "
        "requests one; otherwise return the summary for driver-managed final scoring. "
        "Keep the same evidence and quality requirements; do not weaken a claim or "
        "its review to fill the target. Zero remaining round advances is not itself "
        "a stop instruction: honor the existing goal_met and driver decision, including "
        "authorized soft-limit extensions."
    )
    return {**result, "phase": phase, "message": message}
