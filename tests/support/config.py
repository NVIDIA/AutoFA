# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a whole validated Config to the active config.toml for test setup."""

from dataclasses import asdict

from autofa.behavior_policy import validate_behavior_strictness
from autofa.config import (
    Config,
    _toml_value,
    config_path,
    validate_analysis_outputs,
    validate_discovery_stall_rounds,
    validate_discovery_terminate_if_stalled,
    validate_discovery_wait_for_plateau,
    validate_min_failure_modes,
    validate_taxonomy_agentic_max_turns,
    validate_taxonomy_first_pass,
    validate_taxonomy_second_pass,
)


def save_config(cfg: Config) -> None:
    from autofa.session.campaign_policy import validate_campaign_policy
    from autofa.session.classifier_profiles import validate_profile_config
    from autofa.util.capacity import validate_config_capacity

    validate_profile_config(cfg)
    cfg.campaign_policy = validate_campaign_policy(cfg.campaign_policy)
    validate_config_capacity(cfg)
    validate_analysis_outputs(cfg.analysis_outputs, cfg.preflight_detectors)
    validate_min_failure_modes(cfg.min_failure_modes)
    validate_discovery_wait_for_plateau(cfg.discovery_wait_for_plateau)
    validate_discovery_terminate_if_stalled(cfg.discovery_terminate_if_stalled)
    validate_discovery_stall_rounds(cfg.discovery_stall_rounds)
    validate_behavior_strictness(cfg.behavior_strictness)
    validate_taxonomy_first_pass(cfg.taxonomy_first_pass)
    validate_taxonomy_second_pass(cfg.taxonomy_second_pass)
    validate_taxonomy_agentic_max_turns(cfg.taxonomy_agentic_max_turns)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for k, v in asdict(cfg).items():
        if k == "library_root" and not v:
            continue
        if k == "exploration_seed" and v is None:
            lines.append('exploration_seed = "auto"')
            continue
        lines.append(f"{k} = {_toml_value(v)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
