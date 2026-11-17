# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the evaluator writes a benchmark's run: one directory per benchmark.

<bench>/eval-*.json            one or two run manifests
<bench>/results.jsonl          one record per attempted problem and repeat
<bench>/trajectories.jsonl     the same records with their trajectories
<bench>/model_traffic.jsonl    the adapter proxy's record of every model call
<bench>/agent_logs/<trial>/    an agent's own logs, copied out of the sandbox; the agent's
                               reader is mounted there (`readers.mounts`)
<bench>/inference_log.jsonl, verified_log.jsonl, runtime_stats.json,
        failure_analysis.json, trajectories_enriched.jsonl     its logs and rollups
"""

from __future__ import annotations

from evalrdl_readers.core import Card, Landmark

RUN = Landmark(
    name="bench_dir",
    marker="results.jsonl",
    where=("", "*"),
    expect=Card.ONE(
        because=(
            "the evaluator writes one directory per benchmark run, named after the benchmark "
            "in more than one spelling for the same registry benchmark, so the directory is "
            "found by the stream inside it"
        ),
    ),
)
