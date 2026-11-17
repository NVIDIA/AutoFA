# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared execution-capacity bounds and effective worker limits."""

from typing import Any

CAPACITY_BOUNDS = {
    "llm_eval_workers": 256,
    "llm_eval_max_workers": 256,
    "llm_eval_queue_mib": 4096,
    "judge_workers": 8,
}


def validate_capacity(settings: dict[str, Any]) -> dict[str, int]:
    unknown = set(settings) - CAPACITY_BOUNDS.keys()
    if unknown:
        raise ValueError("Unsupported capacity setting(s): " + ", ".join(sorted(map(str, unknown))))
    for name, value in settings.items():
        if type(value) is not int or not 1 <= value <= CAPACITY_BOUNDS[name]:
            raise ValueError(f"{name} must be an integer between 1 and {CAPACITY_BOUNDS[name]}")
    return dict(settings)


def validate_config_capacity(cfg: Any) -> dict[str, int]:
    from autofa.defaults import setting_value

    return validate_capacity({name: setting_value(cfg, name) for name in CAPACITY_BOUNDS})


def effective_llm_workers(cfg: Any) -> int:
    from autofa.defaults import setting_value

    settings = validate_capacity({name: setting_value(cfg, name)
                                  for name in ("llm_eval_workers", "llm_eval_max_workers")})
    return min(settings.values())
