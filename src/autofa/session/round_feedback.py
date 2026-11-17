# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library exploration bulletins from saved round boundaries and existing evidence.

These assessments are advice, not another progress counter or stopping policy.
The family transaction owns persistence; rendering a bulletin is read-only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.metrics.discovery_grades import GRADE_CODES, grade_summary
from autofa.metrics.live_progress import runtime_display_progress_snapshot
from autofa.metrics.progress import coverage_grade_label
from autofa.session.discovery_target import _task_breadth_summary

_USEFUL = {"GOOD", "EXCELLENT", "OUTSTANDING"}


def round_snapshot(
    rt: Any, *, grades: dict | None = None, task_coverage: dict | None = None,
) -> dict[str, Any]:
    """Capture compact facts without evaluating detectors or reviewing evidence."""
    from autofa.session.task_coverage import task_coverage_snapshot

    if grades is None:
        grades = grade_summary(
            rt.library.functions, behavior_strictness=strictness_for_config(rt.config),
            review_history=rt.library.review_history,
        )
    if task_coverage is None:
        task_coverage = task_coverage_snapshot(rt)
    # Share the CLI's cohort/objective compatibility rules, including association
    # samples. A task-panel fingerprint alone cannot establish comparability.
    progress = runtime_display_progress_snapshot(rt)
    return deepcopy({
        "grades": grades,
        "behavior_strictness": strictness_for_config(rt.config),
        "task_breadth": _task_breadth_summary(task_coverage),
        "library_quality_coverage": progress["library_quality_coverage"],
        "search": {key: progress["search"].get(key) for key in (
            "value", "comparison_key", "scope", "comparison_reason",
        )},
    })


def round_change(start: dict | None, end: dict) -> dict[str, Any]:
    """Compare actual round boundaries; unknown evidence never becomes zero."""
    result: dict[str, Any] = {
        "baseline_available": start is not None, "new_useful_candidates": [],
        "grade_count_changes": None, "search_score_change": None,
        "confirmed_task_net_change": None, "encouraging": False,
        "matched_failure_net_change": None, "matched_task_net_change": None,
    }
    if start is None or start.get("behavior_strictness") != end.get("behavior_strictness"):
        result["baseline_available"] = False
        result["message"] = (
            "Round-start baseline unavailable under the current review policy; "
            "current grades are not evidence of improvement."
        )
        return result
    before, after = start["grades"], end["grades"]
    old = {row["name"]: row for row in before["functions"]}
    new = [row["name"] for row in after["functions"]
           if row["code"] in _USEFUL and row["name"] not in old]
    result["new_useful_candidates"] = new
    deltas = {code: after["counts"].get(code, 0) - before["counts"].get(code, 0)
              for code in GRADE_CODES}
    result["grade_count_changes"] = deltas
    grade_changes = ", ".join(f"{n:+d} {code.title()}" for code, n in deltas.items() if n)
    parts = [f"Grade-count changes: {grade_changes or 'none'}."]
    if new:
        parts.append(
            f"{len(new)} new Good-or-better candidate(s): {', '.join(new[:5])}"
            + (" (first five)" if len(new) > 5 else "")
            + ". Compare their claims before counting distinct mechanisms."
        )
    a, b = start.get("search") or {}, end.get("search") or {}
    if (a.get("comparison_key") and a["comparison_key"] == b.get("comparison_key")
            and a.get("value") is not None and b.get("value") is not None):
        delta = b["value"] - a["value"]
        result["search_score_change"] = delta
        parts.append(
            f"Best provisional search score {a['value']:.3f} → {b['value']:.3f} "
            f"({delta:+.3f}, same objective and evaluation cohorts)."
        )
    else:
        parts.append("Search-score change unavailable: no compatible evaluation-cohort baseline.")
    a, b = start.get("task_breadth"), end.get("task_breadth")
    if (a and b and b.get("corpus_fingerprint") and b.get("behavior_strictness")
            and b.get("quality_thresholds") and all(a.get(key) == b.get(key) for key in (
                "corpus_fingerprint", "behavior_strictness", "quality_thresholds", "n_failed_tasks",
            ))):
        delta = int(b["n_confirmed_failed_tasks"] - a["n_confirmed_failed_tasks"])
        result["confirmed_task_net_change"] = delta
        parts.append(
            f"Quality-supported confirmed-task union {a['n_confirmed_failed_tasks']:.0f}"
            f" → {b['n_confirmed_failed_tasks']:.0f}/{b['n_failed_tasks']:.0f} "
            f"({delta:+d}; observed, not extrapolated)."
        )
    else:
        parts.append("Confirmed-task change unavailable: no compatible corpus/policy baseline.")
    a, b = start.get("library_quality_coverage") or {}, end.get("library_quality_coverage") or {}
    if (a.get("status") == b.get("status") == "exact" and b.get("comparison_key")
            and a.get("comparison_key") == b["comparison_key"]):
        failures = b["n_matched_failed_attempts"] - a["n_matched_failed_attempts"]
        tasks = (b["n_matched_failed_tasks"] - a["n_matched_failed_tasks"]
                 if a.get("task_status") == b.get("task_status") == "exact" else None)
        result.update(matched_failure_net_change=failures, matched_task_net_change=tasks)
        task_change = f"and {tasks:+d} distinct failed tasks" if tasks is not None else (
            "with task change unavailable"
        )
        parts.append(
            f"{coverage_grade_label(b)} combined Coverage changed by {failures:+d} "
            "matched failed attempts "
            f"{task_change} (same corpus, full scans; overlaps count once)."
        )
    elif b:
        parts.append(
            "Combined Coverage change unavailable: sampled/partial evidence or an incompatible "
            "baseline. Extra observed matches after additional scoring "
            "are not proof of improvement."
        )
    result["encouraging"] = bool(
        new or after["n_good_or_better"] > before["n_good_or_better"]
        or any(row["name"] in old and row["code"] in _USEFUL
               and GRADE_CODES.index(row["code"]) < GRADE_CODES.index(old[row["name"]]["code"])
               for row in after["functions"])
        or (result["search_score_change"] or 0) > 1e-9
        or (result["confirmed_task_net_change"] or 0) > 0
        or (result["matched_failure_net_change"] or 0) > 0
        or (result["matched_task_net_change"] or 0) > 0
    )
    result["message"] = " ".join(parts)
    return result


def round_bulletin(
    rt: Any, *, grades: dict, task_coverage: dict | None = None,
) -> dict[str, Any]:
    """Render current-round and last completed-round facts plus exploration options."""
    from autofa.metrics.progress import library_coverage_rows
    from autofa.session.discovery_recovery import recovery_round_context
    from autofa.session.family_discovery import _load
    from autofa.session.round_exploration import build_round_exploration

    state = _load(rt.run.run_dir()) or {}
    current = round_snapshot(rt, grades=grades, task_coverage=task_coverage)
    baseline = state.get("round_baseline") if state.get("round") == rt.run.round else None
    change = round_change(baseline, current)
    completed = next((entry for entry in reversed(state.get("history") or [])
                      if entry.get("round_outcome") is not None
                      and entry.get("behavior_strictness") == current["behavior_strictness"]), None)
    last = None
    quality_coverage = current["library_quality_coverage"]
    lines = [
        f"- {label} ({coverage_grade_label(quality_coverage)} matches): {value}."
        for label, value in library_coverage_rows(quality_coverage)
    ]
    lines.append(
        "- Combined Coverage counts all detected matches of qualifying detectors, not just "
        "judge-reviewed cases. Overlap counts once; task coverage also deduplicates rollouts. "
        "Sampled/partial unions are observed lower bounds, not extrapolated estimates."
    )
    if completed is not None:
        last = {"round": completed["round"], **round_change(
            completed.get("round_baseline"), completed["round_outcome"],
        )}
        lines.append(f"- Completed round {last['round'] + 1}: {last['message']}")
    lines.append(f"- Current round {rt.run.round + 1}, since its start: {change['message']}")
    context = build_round_exploration(rt, grades=grades, family_state=state)
    lines.append(context["message"])
    recovery = recovery_round_context(rt)
    if recovery is not None:
        lines.append("Recovery exploration lessons (historical; inspect the saved evidence):")
        for lesson in recovery["completed_investigations"]:
            name = lesson.get("family_name") or lesson.get("detector_name") or "task evidence"
            result = lesson.get("result") or {}
            note = " ".join(str(result.get("notes") or result.get("grade") or "").split())[:300]
            lines.append(
                f"- {name}: {result.get('outcome', 'unknown')}. {note} "
                f"Evidence: {', '.join(lesson.get('item_ids') or []) or 'see saved result'}. "
                f"Record: {lesson['reference']}."
            )
        if recovery["pending"]:
            lines.append(
                "- A tracked recovery investigation remains open; finish its evidence check "
                "or record its specific unavailable-evidence blocker through the family tools."
            )
        lines.append(recovery["guidance"])
    lines.append(
        "- At the round transition, briefly summarize what the library explains, the round's "
        "progress and lessons, then choose among these exploration directions and state the "
        "first evidence check. Form a testable mechanism hypothesis from inspected evidence. "
        "Different names or match sets do not establish different mechanisms. Useful subtypes "
        "may overlap completely while adding an explanation, diagnostic use or remedy. "
        "Do not turn the next-round plan into a list of individual detector repairs; handle "
        "those within investigations. Reflect within the existing turn; status checks during "
        "an investigation do not require repeating the reflection."
    )
    return {
        "round": rt.run.round, "current_round": change, "last_completed_round": last,
        "exploration": context, "encouraging": change["encouraging"],
        "recovery_exploration": recovery,
        "library_quality_coverage": quality_coverage,
        "message": "\n".join(lines),
    }
