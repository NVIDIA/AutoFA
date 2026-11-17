# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from evalrdl_readers.core import AnyOf, Card, Landmark, Says

from ..common import AGGREGATE_METRICS
from ..nemo_evaluator.landmarks import METADATA, RUN_CONFIG

#: Gym's rollout streams. The caller names them, so any `.jsonl` of the run is one.
STREAMS = "*.jsonl"

# --------------------------------------------------------------------- landmarks

RUN = Landmark(
    name="gym_run",
    marker=Says(
        AnyOf(STREAMS, f"*{AGGREGATE_METRICS}.json", RUN_CONFIG, METADATA),
        contains=AnyOf(
            '"_ng_task_index"',
            '{"responses_create_params":',
            "framework_name: nemo_gym",
            '"framework_name": "nemo_gym"',
            '"nemo_gym": {',
        ),
        because=(
            "two layers, either sufficient: gym's own bytes, since the stream's name is the "
            "caller's (`_ng_task_index` on every row it writes, a rollout row opening with "
            "`responses_create_params`); and the envelope, nemo-evaluator's `run_config.yml` "
            "and `metadata.yaml` naming the harness in `framework_name`, key and value since "
            "`nemo_gym` alone also occurs in params. The envelope is the only witness of a run "
            "whose streams are empty"
        ),
    ),
    # `*/artifacts`: launcher layout, one `<job>.<n>/` per job. `results`: the EFB direct
    # runner's mount of Gym's output, beside its own config record.
    where=("", "artifacts", "*/artifacts", "results"),
    expect=Card.ONE(
        because=(
            "nemo-evaluator writes one `run_config.yml` and one `metadata.yaml` per "
            "evaluation, and one evaluation is one site. A launcher directory holds a rundir "
            "per job, and this count refuses it rather than letting one job stand for the launcher"
        ),
    ),
)

DELIVERABLES = Landmark(
    name="gdpval_repeat",
    # Any file: the Hermes gdpval agent writes only the deliverables, no metadata or history.
    marker="*",
    within=RUN,
    where=("gdpval/deliverables_cache/*/repeat_*",),
    expect=Card.ANY(
        because=("a gdpval agent writes this tree only when `PERSIST_DELIVERABLES_DIR` is set"),
    ),
)

SHARDS = Landmark(
    name="gym_shard_dir",
    marker="combined.jsonl",
    within=RUN,
    where=("*",),
    expect=Card.ANY(
        because=("only a sharded run writes a per-shard directory"),
    ),
)
