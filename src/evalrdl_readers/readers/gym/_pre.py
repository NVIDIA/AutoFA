# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Evaluator 0.2.x wrapping NeMo Gym: the envelope (evaluator) and the rollouts (harness).

    artifacts/
      run_config.yml results.yml metadata.yaml eval_factory_metrics.json  <- envelope
      evaluator_rollouts.jsonl, evaluator_rollouts_{failures,...}.*          <- harness

run.producer/adapter_version/model/started_at come from the envelope; run.benchmark,
run.score and rollout.* from the harness (the envelope only flattens the score).
Plain Gym has no envelope and no fixed stream name, hence `common.primary_stream`.
nemo-evaluator 0.4.0 writes a different folder; `run.adapter_version` tells them apart.
"""

from __future__ import annotations

# Run state, not deliverables; filtered in a translator
# because `At` has no exclusion and `Ignore` on a matched file is CONTESTED.
GYM_IGNORE_FILES = frozenset(
    {
        "finish_params.json",
        "history.json",
        "history.pkl",
        "inprogress_history.json",
        "metadata.json",
        "log.txt",
    }
)
