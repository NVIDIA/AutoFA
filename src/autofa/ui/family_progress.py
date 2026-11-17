# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display saved family-search effort and explicitly scoped breadth observations."""

from __future__ import annotations

from typing import Any


def family_progress_lines(state: dict[str, Any] | None) -> list[str]:
    if not state:
        return []
    phase = str(state.get("phase") or "survey").replace("_", " ")
    completed = state.get("completed_comparisons", 0)
    budget = state.get("comparison_budget", 0)
    attempts = sum((state.get("mutation_attempts") or {}).values())
    lines = [f"family search  {completed}/{budget} comparisons · {phase} · "
             f"{attempts} submissions/retries"]
    for row in state.get("comparisons") or []:
        name = row.get("family_name") or "survey without a supported family"
        if row.get("approach") == "bottom_up":
            name += " (bottom-up)"
        coverage = row.get("coverage") or {}
        if not coverage:
            lines.append(f"  {name}: {row.get('outcome', 'pending')} · coverage unavailable")
            continue
        recall = coverage.get("failure_recall")
        recall_text = f"{recall:.1%}" if isinstance(recall, (float, int)) else "—"
        scope = coverage.get("recall_scope")
        if scope != "full_selected_corpus":
            recall_text += " (sample)" if scope == "sample_estimate" else " (partial/unknown)"
        tasks = coverage.get("n_matched_failed_tasks", 0)
        task_text = (str(tasks) if coverage.get("matched_task_inventory_complete")
                     else f"≥{tasks} observed")
        added = coverage.get("n_additional_failed_attempts")
        if added is None:
            observed = coverage.get("n_observed_additional_failed_attempts")
            added_text = "unknown" if observed is None else f"{observed} observed; total unknown"
        else:
            added_text = str(added)
        added_tasks = coverage.get("n_additional_failed_tasks")
        if added_tasks is None:
            observed_tasks = coverage.get("n_observed_additional_failed_tasks")
            task_added_text = ("unknown" if observed_tasks is None
                               else f"{observed_tasks} observed; total unknown")
        else:
            task_added_text = str(added_tasks)
        status = coverage.get("generalization_status")
        generalization = (" · single-task diagnostic" if status == "single_task_diagnostic"
                          else " · task identity unknown" if status == "task_identity_unknown"
                          else "")
        confirmed = coverage.get("n_review_confirmed_failed_tasks")
        confirmed_text = f" · confirmed failed tasks={confirmed}" if confirmed is not None else ""
        lines.append(f"  assessed {name}: recall={recall_text} · "
                     f"failed tasks={task_text} · "
                     f"added vs compared families={added_text} · "
                     f"additional failed tasks={task_added_text}{confirmed_text}{generalization}")
        seed_comparison = row.get("seed_comparison") or {}
        if seed_comparison:
            complete = seed_comparison.get("comparison_complete", False)
            scope = "" if complete else " observed"
            task_scope = "" if seed_comparison.get("task_comparison_complete") else " observed"
            missing_label = ("lost seed matches" if complete
                             else "seed matches not observed in parent")
            counts = {key: "unknown" if value is None else value
                      for key, value in seed_comparison.items()}
            lines.append(
                "    vs seed union: "
                f"preserved={counts.get('preserved_seed_matches', 'unknown')}{scope} · "
                f"{missing_label}="
                f"{counts.get('seed_matches_not_observed_in_parent', 'unknown')} · "
                f"additional failed tasks="
                f"{counts.get('additional_matched_failed_tasks', 'unknown')}{task_scope} · "
                "confirmed among observed additions="
                f"{counts.get('n_review_confirmed_additional_tasks', 'unknown')} tasks"
            )
            if seed_comparison.get("stale_seed_names"):
                lines.append("    seed claims changed: "
                             + ", ".join(seed_comparison["stale_seed_names"])
                             + "; comparison unavailable until provenance is refreshed")
    if reason := state.get("exploration_exhausted"):
        lines.append(f"  search stopped: {reason}")
    if pending := state.get("pending_decision"):
        lines.append("  unresolved synthesis: " + str(pending.get("shared_relation") or
                                                     pending.get("outcome") or "see decision"))
        if instruction := pending.get("instruction"):
            lines.append("    next: " + str(instruction))
    for decision in state.get("recent_decisions") or []:
        lines.append(f"  synthesis decision: {decision['outcome']} · {decision['status']} · "
                     f"{decision['reason']}")
        if receipt := decision.get("experiment_id") or decision.get("comparison_id"):
            lines.append(f"    diagnostic receipt: {receipt} (author controls)")
        if next_action := decision.get("next_action"):
            lines.append("    next: " + next_action)
    return lines
