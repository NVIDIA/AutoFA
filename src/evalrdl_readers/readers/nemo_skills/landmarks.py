# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Landmarks of nemo-skills rundirs: nested subtasks, flat multi-seed, single-seed.

Same nemo-evaluator wrapper output as nemo-gym; `framework_name` and `eval-results/` tell
them apart. `dispatch.RUNDIR` tries this reader first because it needs more.
"""

from __future__ import annotations

from evalrdl_readers.core import AnyOf, Card, Landmark, Says

from ..nemo_evaluator.landmarks import RUN_CONFIG

RUN = Landmark(
    name="ns_run",
    marker=Says(
        RUN_CONFIG,
        contains=AnyOf("framework_name: nemo_skills", '"framework_name": "nemo_skills"'),
        because=(
            "`run_config.yml` is the wrapper's file and says only that nemo-evaluator ran; "
            "`framework_name` is what tells a nemo-skills run from a nemo-gym one"
        ),
    ),
    where=("", "artifacts"),
    expect=Card.ONE(because="nemo-evaluator writes exactly one per evaluation, gym or skills"),
)

TASK = Landmark(
    name="ns_task", marker=AnyOf("metrics.json", "output-rs*.jsonl"),
    within=RUN, where=("eval-results/*",),
    expect=Card.AT_LEAST_ONE(
        because=(
            "`eval-results/<benchmark>/` is the enumeration for nemo-skills: a scored task "
            "carries `metrics.json`, an unscored one only its output streams. `framework_name` "
            "on RUN identifies the producer, so this marker enumerates rather than identifies"
        ),
    ),
)

SUBTASK = Landmark(
    name="ns_subtask", marker="metrics.json", within=TASK, where=("*",),
    expect=Card.ANY(
        because=(
            "some benchmarks split into subtask directories and most into none; "
            "nearest-wins would swallow the children into the parent if they were one "
            "landmark, so they are two"
        ),
    ),
)
