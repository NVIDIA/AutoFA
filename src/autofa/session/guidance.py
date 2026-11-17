# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build analysis-focus and skill instructions for discovery."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from autofa.defaults import setting_value
from autofa.session.behavior_guidance import behavior_policy_prompt

ANALYSIS_FOCI = ("model", "harness", "both")
ANALYSIS_FOCUS_LABELS = {
    "model": "Model failures", "harness": "Harness failures", "both": "Model + harness",
}


def snapshot_guidance(cfg: Any, *, source_harnesses: Iterable[str] = ()) -> dict[str, Any]:
    """Capture analysis focus and skills matching the analyzed logs' harnesses."""
    focus = setting_value(cfg, "analysis_focus")
    if focus not in ANALYSIS_FOCI:
        raise ValueError(f"analysis_focus must be one of {ANALYSIS_FOCI}; got {focus!r}")
    from autofa.session.analysis_skills import snapshot_configured_skills

    return {
        "analysis_focus": focus,
        **snapshot_configured_skills(cfg, source_harnesses),
    }


def guidance_prompt(run: Any, *, audience: str = "judge") -> str:
    """Build the discovery prompt for the selected analysis focus and skills."""
    focus = getattr(run, "analysis_focus", "model")
    from autofa.session.analysis_skills import skills_prompt

    if audience not in {"discovery", "judge"}:
        raise ValueError("Guidance audience must be discovery or judge")
    skills = skills_prompt(run, audience=audience)
    scopes = [r["evidence_scope"] for r in (getattr(run, "reference_data", None) or [])
              if isinstance(r, dict) and r.get("evidence_scope")]
    if focus == "model" and not skills and not scopes:
        return ""
    lines = ["\n\n## User-selected analysis scope (applies throughout this run)"]
    lines.append(f"analysis_focus={focus}")
    if scopes:
        lines.extend(["### Input visibility (not evidence of a failure)", *scopes])
    if focus == "harness":
        lines.append(
            "Focus discovery on failures addressable through concrete harness changes: "
            "tool contracts, "
            "environment/setup, orchestration, budgets, context handling, or feedback delivery. "
            "This includes model-originated mistakes that the harness can help prevent or recover "
            "from; do not exclude a mode just because model improvements could also help. "
            "Model mistakes without a supported harness intervention are out of scope. "
            "Actively investigate tool requests and responses, ambiguous errors, missing feedback, "
            "and recovery opportunities. If a judged mode is model-only, pivot toward a "
            "harness-addressable mechanism instead of relabeling it to satisfy the focus. "
            "Model-only wording elsewhere in the "
            "protocol should be read as the selected harness scope, not a requirement to "
            "blame the model. A mechanism should explain the observed harness behavior, "
            "its effect on the agent, and a concrete harness change worth testing."
        )
    elif focus == "both":
        lines.append(
            "Investigate both model behavior and harness-addressable failures, including "
            "interactions; actively explore both directions rather than assuming model fault. "
            "For each mechanism distinguish what went wrong from where an improvement can "
            "be made: model, harness, both, or insufficient evidence. Model mistakes can have "
            "harness remedies. Use judge_function's improvement assessment to refine these "
            "proposals; do not force every mode into both categories."
        )
    else:
        lines.append("Prioritize the model's reasoning, decisions, verification, and tool use.")
    lines.append(
        "Preserve the existing discovery/evaluation/judging loop and anti-leakage field "
        "policy. Focus does not grant access to new data or relax evidence requirements. "
        "Do not invent missing tool observations, test results, or hidden-grader causality. "
        "Use the selected behavioral review standard in every analysis focus; changing "
        "focus does not strengthen or weaken that standard. Ground the claim in observed "
        "behavior, compare alternative explanations and inspect relevant recovery. "
        "Test the predicate on inspected positive and contrast item IDs before corpus "
        "evaluation; author-selected controls do not validate the mode. "
        "Distinguish observed behavior from a suspected cause. State visibility limits "
        "in the mechanism when they affect attribution. A suggested repair is a hypothesis "
        "to test, not a model/harness intervention AutoFA has executed."
    )
    lines.append(behavior_policy_prompt(run, include_discovery_guidance=audience == "discovery"))
    return "\n".join(lines) + skills
