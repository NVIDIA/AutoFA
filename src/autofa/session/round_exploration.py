# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only, library-level exploration context from existing discovery records.

This is a bounded index of claims and investigation lessons, not a generated
failure taxonomy. The agent must inspect the cited evidence before proposing a
mechanism; neither grades nor match overlap establish semantic relationships.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autofa.session.family_discovery import _load
from autofa.session.modes_ledger import read_modes

_MAX_CLAIMS = 6
_MAX_LESSONS = 5
_SUPPORTED = {"GOOD", "EXCELLENT", "OUTSTANDING"}


def _text(value: Any, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)


def _comparisons(state: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    rows = []
    for index, old in enumerate(state.get("history") or []):
        if not isinstance(old, dict):
            continue
        path = f"family_discovery.json#/history/{index}"
        comparisons = old.get("comparisons") or []
        if comparisons:
            rows.extend(
                (row, f"{path}/comparisons/{i}")
                for i, row in enumerate(comparisons)
                if isinstance(row, dict)
            )
        elif old.get("survey"):
            rows.append((old, path))
    rows.extend(
        (row, f"family_discovery.json#/comparisons/{i}")
        for i, row in enumerate(state.get("comparisons") or [])
        if isinstance(row, dict)
    )
    return rows


def _lessons(state: dict[str, Any], modes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lessons, seen = [], set()
    for row, reference in reversed(_comparisons(state)):
        survey = row.get("survey") or {}
        assessment = row.get("assessment") or {}
        if (isinstance(survey, dict) and isinstance(assessment, str)
                and row.get("outcome") == "synthesis_inspection_only"):
            # Budget-exhausted synthesis inspections save the author's assessment
            # directly, without the completed-comparison assessment wrapper.
            assessment = {"outcome": survey.get("outcome", "recorded"), "notes": assessment}
        if not isinstance(survey, dict) or not isinstance(assessment, dict):
            continue
        outcome = assessment.get("outcome", "recorded")
        note = assessment.get("blocker") or assessment.get("notes")
        if not note:
            continue
        name = survey.get("family_name") or assessment.get("family_name")
        identity = (name, outcome, str(note))
        if identity in seen:
            continue
        seen.add(identity)
        if outcome == "deferred":
            revisit = assessment.get("next_step") or "Obtain the missing decisive evidence."
        elif outcome == "no_coherent_family":
            revisit = (
                "New cross-task evidence must support a coherent distinction beyond "
                "the inspected cases."
            )
        else:
            revisit = (
                "Check the saved assessment and current claim before treating this "
                "historical lesson as current."
            )
        evidence = survey.get("evidence") or []
        lessons.append(
            {
                "name": _text(name),
                "outcome": _text(outcome),
                "mechanism": _text(survey.get("mechanism")),
                "lesson": _text(note, 400),
                "revisit_condition": _text(revisit, 400),
                "evidence_item_ids": [
                    _text(item.get("item_id"), 120)
                    for item in evidence
                    if isinstance(item, dict) and item.get("item_id")
                ][:3],
                "historical": True,
                "reference": reference,
            }
        )
        if len(lessons) >= _MAX_LESSONS:
            break
    # Removed modes remain useful exploration knowledge even without a family assessment.
    family_names = {row["name"] for row in lessons}
    for mode in reversed(modes):
        status = mode.get("status")
        if status not in {"open", "rejected", "redundant"} or not mode.get("reason"):
            continue
        if _text(mode.get("name")) in family_names or len(lessons) >= _MAX_LESSONS:
            continue
        revisit = {
            "open": (
                "Resolve the recorded evidence gap or supply a new supported distinction; "
                "open does not mean refuted."
            ),
            "rejected": (
                "Address the recorded counterevidence with new evidence or a "
                "materially different claim."
            ),
            "redundant": (
                "Establish a distinct explanation, diagnostic use or remedy, "
                "not another wording of the same claim."
            ),
        }[status]
        lessons.append(
            {
                "name": _text(mode.get("name")),
                "outcome": status,
                "mechanism": _text(mode.get("mechanism")),
                "lesson": _text(mode["reason"], 400),
                "revisit_condition": revisit,
                "historical": True,
                "reference": {"tool": "list_failure_modes", "arguments": {}},
            }
        )
    return lessons


def build_round_exploration(
    rt: Any,
    *,
    grades: dict[str, Any] | None = None,
    family_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index saved claims and lessons without changing state or executing detectors.

    Output is bounded and points to existing tools/records for full evidence.
    Calling this for status must not create a round, consume work, read a corpus,
    or turn historical assessments into current causal support.
    """
    if grades is None:
        from autofa.behavior_policy import strictness_for_config
        from autofa.metrics.discovery_grades import grade_summary

        grades = grade_summary(
            rt.library.functions,
            behavior_strictness=strictness_for_config(rt.config),
            review_history=rt.library.review_history,
        )
    unavailable = []
    state = family_state
    if state is None:
        try:
            state = _load(rt.run.run_dir()) or {}
        except (OSError, ValueError):
            state = {}
            unavailable.append("family history unavailable")
    if state.get("run_id") not in {None, rt.run.run_id}:
        state = {}
        unavailable.append("family history belongs to another run")
    try:
        modes = read_modes(rt.run.run_dir())
    except (OSError, ValueError, TypeError, AttributeError):
        modes = []
        unavailable.append("failure-mode history unavailable")
    functions = {_field(fn, "name"): fn for fn in rt.library.functions}
    cards = []
    all_grades = grades.get("functions") or []
    for row in all_grades[:_MAX_CLAIMS]:
        fn = functions.get(row.get("name"))
        if fn is None:
            continue
        cards.append(
            {
                "name": _text(row["name"]),
                "grade": row.get("code", "UNSCORED"),
                "description": _text(_field(fn, "description")),
                "mechanism": _text(_field(fn, "mechanism")),
                "reference": {"tool": "get_library_function", "arguments": {"name": row["name"]}},
            }
        )
    lessons = _lessons(state, modes)
    supported = [row for row in all_grades if row.get("code") in _SUPPORTED]
    directions = [
        {
            "kind": "different_mechanism",
            "title": "Find a different failure mechanism",
            "basis": (
                f"The library retains {len(functions)} detector claims; "
                "compare new evidence with those claims."
                if functions
                else "No detector claims are retained yet; establish a first cross-task mechanism."
            ),
            "first_checks": [
                {"tool": "library_coverage", "arguments": {}},
                {
                    "tool": "sample_items",
                    "arguments": {
                        "coverage": "unmatched" if functions else "all",
                        "distinct_tasks": True,
                    },
                },
            ],
            "evidence_question": (
                "Which concrete damaging decision is not explained by the retained claims? "
                "Inspect requirements, actions, results and later recovery across different tasks."
            ),
        }
    ]
    scheduled = state.get("deferred_synthesis") or state.get("synthesis_required")
    if scheduled:
        directions.append(
            {
                "kind": "recorded_synthesis",
                "title": "Inspect the saved shared-family lead",
                "basis": (
                    "The family workflow already recorded a synthesis lead or due inspection; "
                    "this is not inferred from overlapping matches."
                ),
                "first_checks": [{"tool": "family_discovery", "arguments": {"action": "status"}}],
                "evidence_question": (
                    "Do the cited seeds share one supported mechanism? Check a boundary and "
                    "new manifestations while preserving useful originals; a union of "
                    "unrelated failures is not a family."
                ),
            }
        )
    if supported:
        directions.append(
            {
                "kind": "informative_subtype",
                "title": "Explore a useful distinction within covered failures",
                "basis": (
                    f"{len(supported)} retained detector(s) meet Good-or-better requirements; "
                    "their covered families may contain informative subtypes."
                ),
                "first_checks": [
                    {"tool": "library_coverage", "arguments": {}},
                    {"tool": "list_library_functions", "arguments": {}},
                ],
                "evidence_question": (
                    "Across matched tasks, is there a distinct decision, competing explanation "
                    "or remedy worth diagnosing separately? Inspect full cases and boundaries; "
                    "a useful subtype need not add matched tasks. Choose the family from "
                    "this evidence, then retrieve get_library_function(name=...) for its claim."
                ),
            }
        )
    if len(directions) < 3:
        deferred = next((lesson for lesson in lessons if lesson["outcome"] == "deferred"), None)
        if deferred:
            directions.append(
                {
                    "kind": "conditional_revisit",
                    "title": "Check whether a blocked exploration direction has become testable",
                    "basis": deferred["lesson"],
                    "first_checks": [
                        {"tool": "family_discovery", "arguments": {"action": "status"}},
                        {"tool": "list_failure_modes", "arguments": {}},
                    ],
                    "evidence_question": (
                        "Revisit only if this evidence check can now be completed: "
                        + deferred["revisit_condition"]
                    ),
                }
            )
    if len(directions) < 2:
        directions.append(
            {
                "kind": "contrast",
                "title": "Investigate an alternative explanation using contrasting tasks",
                "basis": (
                    "The saved records do not establish another supported family "
                    "or a ready synthesis lead."
                ),
                "first_checks": [
                    {
                        "tool": "sample_items",
                        "arguments": {"coverage": "all", "distinct_tasks": True},
                    }
                ],
                "evidence_question": (
                    "Compare failed and successful tasks without treating outcome as a "
                    "mechanism label. What action or later recovery distinguishes competing "
                    "explanations? Same-task rollouts can supplement, not establish, cross-task "
                    "support."
                ),
            }
        )
    directions = directions[:3]
    lines = ["Mechanism map (retained claims; no inferred grouping):"]
    lines.extend(
        f"- {card['grade']} {card['name']}: "
        f"{_text(card['mechanism'] or card['description'] or 'claim detail unavailable', 140)}"
        for card in cards[:3]
    )
    if not cards:
        lines.append("- No retained claim details available.")
    if len(functions) > min(3, len(cards)):
        lines.append(
            f"- {len(functions) - min(3, len(cards))} further claims: list_library_functions."
        )
    if lessons:
        lines.append("Investigation lessons (historical; inspect full records before reuse):")
        lines.extend(
            f"- {row['outcome']} {row['name'] or 'inspected evidence'}: "
            f"{_text(row['lesson'], 160)} Revisit: {_text(row['revisit_condition'], 140)}"
            for row in lessons[:2]
        )
    exhausted = state.get("exploration_exhausted")
    if exhausted:
        lines.append(
            f"Current search allowance/evidence state: {_text(exhausted)}. "
            "This does not refute unexplored mechanisms."
        )
    lines.append("Next exploration options; choose from the evidence:")
    questions = {
        "different_mechanism": (
            "Seek a damaging decision beyond retained claims; check later recovery."
        ),
        "recorded_synthesis": "Check one shared mechanism and boundaries; preserve useful seeds.",
        "informative_subtype": (
            "Seek distinct decisions or remedies inside covered failures; "
            "a useful subtype need not add matched tasks."
        ),
        "conditional_revisit": "Revisit only if the recorded missing evidence is now retrievable.",
        "contrast": "Use cross-task contrasts to distinguish explanations, not outcome labels.",
    }
    for direction in directions:
        checks = []
        for step in direction["first_checks"]:
            arguments = ", ".join(f"{key}={value!r}" for key, value in step["arguments"].items())
            checks.append(f"{step['tool']}({arguments})")
        lines.append(
            f"- {direction['title']}: {questions[direction['kind']]} "
            f"First: {' → '.join(checks)}."
        )
    lines.append(
        "No known match means untested, unresolved or not matched, not a tested negative. "
        "Do not sum detector coverage or infer distinct mechanisms from names, grades or overlap. "
        "Individual detector repairs belong within the investigation."
    )
    if unavailable:
        lines.append("Unavailable context: " + "; ".join(unavailable) + ".")
    return {
        "claim_cards": cards,
        "n_retained_claims": len(functions),
        "n_claim_cards_omitted": max(0, len(functions) - len(cards)),
        "lessons": lessons,
        "directions": directions,
        "scope": "saved_claims_and_investigations_not_a_validated_taxonomy",
        "unavailable": unavailable,
        "message": "\n".join(lines),
    }
