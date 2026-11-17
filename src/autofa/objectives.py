# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Objective preferences shared by settings, discovery, ranking, and saved runs."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autofa.config import Config, RunConfig

OBJECTIVE_LABELS = {
    "ppv_at_pi0": "Normalized precision",
    "task_recall": "Failure task recall",
    "behavior_purity": "Detector adherence",
}
OBJECTIVE_DISPLAY_LABELS = {**OBJECTIVE_LABELS, "recall": "Failure attempt recall"}
# New discovery optimizes these three outcomes.
PRESETS = {
    "balanced": [100.0 / 3] * 3,
    "precision": [50.0, 20.0, 30.0],
    "recall": [20.0, 50.0, 30.0],
}
OBJECTIVE_FIELDS = ("objective_axes", "objective_weights", "objective_rank")


def weights_by_axis(cfg: Config) -> dict[str, float]:
    from autofa.config import normalize_objective_name

    axes = [normalize_objective_name(a) for a in cfg.objective_axes]
    weights = cfg.objective_weights or [1.0] * len(axes)
    if len(weights) != len(axes):
        raise ValueError(f"Each objective needs one weight: objective_axes has {len(axes)} "
                         "entries but objective_weights has "
                         f"{len(weights)}; edit both together, or use objective_weights = [] "
                         "for equal weights")
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("Each objective needs one weight and a unique name")
    if any(isinstance(w, bool) or not isinstance(w, (int, float))
           or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Objective weights must be finite, non-negative numbers")
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("At least one objective must have a positive weight")
    return {a: w / total for a, w in zip(axes, weights, strict=True)}


def apply_preset(cfg: Config, preset: str) -> Config:
    return replace(cfg, objective_axes=list(OBJECTIVE_LABELS),
                   objective_weights=list(PRESETS[preset]), objective_rank="geometric_mean")


def config_for_new_discovery(cfg: Config) -> Config:
    """Use current objectives for fresh discovery, leaving saved runs untouched.

    Starting a new run (including Continue from a library), or editing defaults
    for future runs, with axes other than the three current objectives uses
    Recall instead. Current presets and custom weights survive, including
    reordered axes and inactive weights.
    """
    from autofa.config import normalize_objective_name

    axes = [normalize_objective_name(axis) for axis in cfg.objective_axes]
    if len(axes) != len(OBJECTIVE_LABELS) or set(axes) != set(OBJECTIVE_LABELS):
        return apply_preset(cfg, "recall")
    weights_by_axis(cfg)
    return cfg if axes == cfg.objective_axes else replace(cfg, objective_axes=axes)


def preset_name(cfg: Config) -> str:
    actual = weights_by_axis(cfg)
    for name, values in PRESETS.items():
        expected = dict(zip(OBJECTIVE_LABELS, [v / sum(values) for v in values], strict=True))
        if actual.keys() == expected.keys() and all(
            math.isclose(actual[key], value) for key, value in expected.items()
        ):
            return name.capitalize()
    return "Custom"


def objective_summary(cfg: Config) -> str:
    """Describe the actual scoring preference."""
    weights = ", ".join(
        f"{OBJECTIVE_DISPLAY_LABELS.get(axis, axis)}={weight:.1%}"
        for axis, weight in weights_by_axis(cfg).items()
    )
    return f"{preset_name(cfg)} · {cfg.objective_rank} · {weights}"


def snapshot_objectives(cfg: Config) -> dict[str, Any]:
    weights_by_axis(cfg)
    return {name: list(value) if isinstance(value := getattr(cfg, name), list) else value
            for name in OBJECTIVE_FIELDS}


def _saved_objectives(saved: dict[str, Any]) -> dict[str, Any]:
    """Apply only the objective fields this version defines from a run snapshot."""
    return {name: deepcopy(saved[name]) for name in OBJECTIVE_FIELDS if name in saved}


def config_for_run(cfg: Config, run: RunConfig) -> Config:
    from autofa.behavior_policy import validate_behavior_strictness

    strictness = validate_behavior_strictness(run.behavior_strictness)
    return replace(
        cfg, **_saved_objectives(run.objective_settings), behavior_strictness=strictness,
        discovery_wait_for_plateau=run.discovery_wait_for_plateau,
        discovery_terminate_if_stalled=run.discovery_terminate_if_stalled,
        discovery_stall_rounds=run.discovery_stall_rounds,
        campaign_policy=run.campaign_policy,
        usage_pricing=run.usage_pricing,
    )


def config_from_run_dir(directory: Path | None) -> Config | None:
    from autofa.config import Config

    if directory is None:
        return None
    try:
        data = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return Config(**_saved_objectives(data.get("objective_settings", {})),
                  behavior_strictness=data.get("behavior_strictness", "causal"),
                  discovery_wait_for_plateau=data.get("discovery_wait_for_plateau", False),
                  discovery_terminate_if_stalled=data.get("discovery_terminate_if_stalled", True),
                  discovery_stall_rounds=data.get("discovery_stall_rounds", 3),
                  campaign_policy=data.get("campaign_policy", {}),
                  usage_pricing=data.get("usage_pricing", {}))


def discovery_strategy(cfg: Config) -> str:
    """Map presets and custom preferences to a reproducible search strategy.

    Custom weights choose precision/recall only when that metric outweighs its
    counterpart. A tie, or neither axis, gives balanced discovery.
    """
    weights = weights_by_axis(cfg)
    precision = weights.get("ppv_at_pi0", 0.0) + weights.get("raw_ppv", 0.0)
    recall = weights.get("recall", 0.0) + weights.get("task_recall", 0.0)
    if math.isclose(precision, recall):
        return "balanced"
    return "precision" if precision > recall else "recall"


def judge_objective_guidance(
    cfg: Config | str, *, behavior_strictness: str = "causal",
) -> str:
    """Judge a frozen claim under its policy, independently of search preferences."""
    from autofa.behavior_policy import validate_behavior_strictness

    mode = validate_behavior_strictness(behavior_strictness)
    if mode != "causal":
        standard = (
            "Judge the behavior exactly as claimed. Causality and permanent harm are required "
            "only when asserted; do not add them to a descriptive claim."
            if mode == "behavioral" else
            "Require a trace-supported mistake or shortcut plausibly relevant to failure. "
            "Inspect later recovery and reject fully recovered or unrelated incidents. "
            "Do not require reconstruction of an exact final defect or a complete causal chain."
        )
        return ("Accept coherent broad families "
                "as well as specific mechanisms. Do not demand the same technical subtype "
                "across tasks or weaken any stated claim. " + standard)
    return ("Accept coherent broad causal families as well as specific mechanisms. "
            "Judge the shared causal relation actually claimed; different technical "
            "subtypes may support that relation. Require every stated distinguishing "
            "detail, without adding child-level detail to a parent claim or silently "
            "broadening a specific claim. Require evidence that the described behavior contributed "
            "to this task's failure. Exclude recovered incidents without a remaining "
            "causal effect. Reject vacuous failure restatements, generic symptoms, and "
            "mere correlations. Unsupported causal attribution stays unknown or "
            "refuted. Search preferences do not change the evidence standard.")


def objective_guidance(cfg: Config) -> str:
    weights = weights_by_axis(cfg)
    summary = ", ".join(f"{name}={weight:.1%}" for name, weight in weights.items())
    objective = (f"{preset_name(cfg)} weighted preference: {summary}. Configured "
                 "score: weighted geometric mean; preserve supported Pareto alternatives. "
                 "A zero on an active metric gives zero combined score; high "
                 "precision and purity cannot erase negligible recall. ")
    objective += ("Follow pareto_status.target_grade and the live round assessment for the "
                  "active discovery target. Scores diagnose trade-offs; grades use the shared "
                  "quality requirements, and runtime goal_met determines completion. ")
    strategy = discovery_strategy(cfg)
    from autofa.behavior_policy import strictness_for_config

    mode = strictness_for_config(cfg)
    search = {
        "precision": (
            "Inspect flagged passes and competing explanations to find supported "
            "boundaries. Retain precise broad families and useful specific mechanisms."
        ),
        "balanced": (
            "Search for useful families across tasks and informative child mechanisms. "
            "Choose boundary checks, supported extensions or another mechanism from "
            "the evidence. Generalize incidental implementation details."
        ),
        "recall": (
            "Investigate unexplained failures and test whether supported mechanisms "
            "transfer across tasks. Preserve precision and adherence while developing "
            "broader families; retain informative children and overlapping explanations."
        ),
    }[strategy]
    return (objective + f"Discovery strategy: {strategy}. {search} "
            "Optimization preferences guide search and ranking, not claim validity. "
            + judge_objective_guidance(strategy, behavior_strictness=mode) + " "
            "During discovery, search_score uses distinct-task recall on the common fixed "
            "scoring panel: matched failed tasks / failed tasks in that panel, with repeated "
            "rollouts counted once per task. Compare only matching panel fingerprints. "
            "Stored corpus task_recall instead divides observed matched tasks by all failed "
            "tasks in the selected corpus; sampled or partial scans give a lower bound. "
            "CLI/report Coverage is failed-attempt recall, estimated with sampling weights "
            "when sampled. None of these match counts alone establishes judge confirmation. "
            "Useful discovery progress includes distinct explanations, remedies and quality "
            "improvements from overlapping or subset detectors. Report those gains separately "
            "from additional quality-confirmed distinct-task coverage, counting task overlap "
            "once. Equivalent renamed detectors add no discovery value. Preserve supported "
            "seeds and review new matches under the selected Judge Policy. Specificity is "
            "descriptive detail, not a standard objective or minimum breadth requirement. "
            "Zero-weight objectives are inactive.")
