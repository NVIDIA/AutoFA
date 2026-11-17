# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a gym run holds that this reader deliberately does not read."""

from __future__ import annotations

from evalrdl_readers.core import ROOT, Ignore

from .landmarks import RUN, SHARDS

GYM_IGNORES = (
    Ignore(
        ROOT,
        "verification.log",
        because="the certification pass's own log: the two documents beside it, `completion.json` and `result.json`, concatenated; `gym_verification` reads the first of them",
    ),
    Ignore(
        ROOT,
        "verification/result.json",
        because="the certification pass's score card: `score_percentage` is a percentage where "
        "`rollouts.reward` is a fraction, and dividing by 100 is arithmetic no declaration "
        "authorises",
    ),
    Ignore(
        ROOT,
        "verification/generation-samples.md",
        because="a few rollouts rendered to markdown for a human to skim, each a record of the stream this reader already reads",
    ),
    Ignore(
        RUN,
        "model_calls*/*.capture.incomplete",
        because=(
            "the producer's empty marker that a capture was cut off before it closed; "
            "that the capture beside it is short is a completeness fact about the run, "
            "not a model call"
        ),
    ),
    Ignore(
        RUN,
        "opencode/*/runner.log",
        because=(
            "the opencode runner's completion marker, one per task directory: "
            "`RUNNER_DONE`, no log and no outcome"
        ),
    ),
    Ignore(
        RUN,
        "opencode/*/failure.json",
        because=(
            "written instead of a `generation.json` when the agent never ran; the "
            "failures sidecar `gym_failures` reads holds the same failure with strictly "
            "more, and a second account of it under a second key is a collision, not a fact"
        ),
    ),
    *(
        Ignore(
            RUN,
            f"*/{capture_artifact}",
            because="the producer's training-token capture store with restrictive permissions",
        )
        for capture_artifact in (
            "*.tokens.jsonl",
            "*.tokens.state.json",
            "*.tokens.intents",
            "*.tokens.incomplete",
            "*.tokens.lock",
            ".tokens-state-*",
        )
    ),
    Ignore(RUN, "progress", because="a one-byte progress marker the harness polls"),
    Ignore(
        RUN,
        "vllm_pd_node_ips.txt",
        because="the IP addresses of the prefill/decode nodes — infrastructure, gone the moment the job ended",
    ),
    Ignore(
        RUN,
        "pd-runtime/runtime.env",
        because="the environment the disaggregated serving stack was started with",
    ),
    Ignore(
        RUN,
        "quality_summary.json",
        because="the TB4 harness's own quality rollup, derived from the rollout stream",
    ),
    Ignore(
        RUN,
        "nemo_gym_logs/*.log",
        because="per-component gym logs, written only when `nemo_gym_log_dir` is set: harness chatter",
    ),
    Ignore(
        RUN,
        "pd-role-logs/*.log",
        because="per-role serving logs from the disaggregated vLLM deployment — infrastructure",
    ),
    Ignore(RUN, "pd-role-status/*.status", because="the serving roles' readiness files"),
    Ignore(
        SHARDS,
        "box_*.log",
        because="one log per sandbox VM, describing the machine, not the rollout",
    ),
    Ignore(
        SHARDS,
        "tasks.jsonl",
        because="the shard runner's copy of the task list, i.e. the dataset again",
    ),
    Ignore(
        SHARDS,
        "shard_*.tar.gz",
        because="per-sandbox filesystem captures, gigabytes each, of the machine rather than the rollout",
    ),
)
