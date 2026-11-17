# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The envelope read on its own; a harness reader mounts it with `wraps=True`."""

from __future__ import annotations

from evalrdl_readers.core import (
    At,
    Card,
    Ignore,
    Input,
    Landmark,
    Order,
    Population,
    Reader,
)

from .. import questions as Q
from ..store_schema import READER_SCHEMA
from .landmarks import (
    CONFIG,
    ENVELOPE,
    EVAL_METADATA,
    EXPORT_CONFIG,
    FACTORY_METRICS,
    METADATA,
    RESULTS,
    RUN_CONFIG,
)
from .translate import (
    ne_eval_factory_metrics,
    ne_eval_metadata,
    ne_export_config,
    ne_metadata,
    ne_results_yml,
    ne_run_config,
)

ABSENT = "each envelope file is optional, and a translator that finds none says nothing"

NE_RUN_CONFIG = Input(
    name="ne_run_config",
    via=ne_run_config,
    reads=[At(ENVELOPE, RUN_CONFIG)],
    answers={
        "run.producer": Card.ANY(
            because=f"`framework_name` names the harness the wrapper ran; {ABSENT}"
        ),
        "run.benchmark": Card.ANY(
            because="`config.params.task` is the task the wrapper was given; null where the harness takes its benchmark in its own block"
        ),
        "run.model": Card.ANY(
            because="`target.api_endpoint.model_id` is the endpoint the wrapper was pointed at"
        ),
    },
)

NE_RESULTS_YML = Input(
    name="ne_results_yml",
    via=ne_results_yml,
    reads=[At(ENVELOPE, RESULTS)],
    answers={
        "run.benchmark": Card.ANY(
            because="the small shape writes `benchmark` at the top; the launcher shape does not, and a None counts as an answer"
        ),
        "run.model": Card.ANY(because=ABSENT),
        "run.n_expected": Card.ANY(
            because="`num_rollouts` is null where the wrapper made no claim, and a null is silence"
        ),
        "run.adapter_version": Card.ANY(
            because="`metadata.versioning.nemo_evaluator` stamps the layout the wrapper chose; the small shape has none"
        ),
    },
)

NE_METADATA = Input(
    name="ne_metadata",
    via=ne_metadata,
    reads=[At(ENVELOPE, METADATA)],
    answers={
        "run.output_dir_name": Card.ANY(because=ABSENT),
        "run.cluster": Card.ANY(
            because="`launcher_resolved_config.cluster.name`, where the launcher resolved one"
        ),
        "run.source_user": Card.ANY(
            because="`launcher_resolved_config.execution.username`, the account the launcher submitted as"
        ),
    },
)

NE_EXPORT_CONFIG = Input(
    name="ne_export_config",
    via=ne_export_config,
    reads=[At(ENVELOPE, EXPORT_CONFIG)],
    answers={
        "run.benchmark": Card.ANY(because="written only when the launcher exports the run"),
        "run.model": Card.ANY(because=ABSENT),
        "run.cluster": Card.ANY(because="an operator's own tag; a run filed without one is legal"),
    },
)

NE_EVAL_METADATA = Input(
    name="ne_eval_metadata",
    via=ne_eval_metadata,
    reads=[At(ENVELOPE, EVAL_METADATA)],
    answers={
        "run.suite": Card.ANY(because="written only by one launch path"),
        "run.started_at": Card.ANY(because="`date` is that launch path's own start stamp"),
        "run.agent_harness": Card.ANY(
            because="`harness.name` is the agent harness the benchmark drove, where it names one"
        ),
        "run.model": Card.ANY(because=ABSENT),
    },
)

NE_EFM = Input(
    name="ne_efm",
    via=ne_eval_factory_metrics,
    reads=[At(ENVELOPE, FACTORY_METRICS)],
    answers={
        "run.started_at": Card.ANY(
            because=(
                "written whenever the adapter ran. `evaluation.start_time` carries an "
                "unformatted `%f`, so the well-formed `run_start` beside it is published and "
                "the malformed string carried as a note"
            ),
        )
    },
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,
    name="nemo-evaluator",
    landmarks=(ENVELOPE,),
    inputs=(
        ("nemo-evaluator-launcher", (NE_EXPORT_CONFIG,)),
        ("nemo-evaluator", (NE_RUN_CONFIG, NE_RESULTS_YML, NE_METADATA, NE_EVAL_METADATA, NE_EFM)),
    ),
    questions=(
        Q.RUN_PRODUCER,
        Q.RUN_BENCHMARK,
        Q.RUN_MODEL,
        Q.RUN_EXPECTED,
        Q.RUN_ADAPTER,
        Q.RUN_OUTDIR,
        Q.RUN_CLUSTER,
        Q.RUN_SOURCE_USER,
        Q.RUN_SUITE,
        Q.RUN_AGENT_HARNESS,
        Q.RUN_STARTED,
    ),
    orders=(
        Order(
            question="run.benchmark",
            order=("ne_run_config", "ne_export_config", "ne_results_yml"),
            keep_loser=True,
            because=(
                "the task the wrapper was given, then the name a human tagged the export "
                "with, then the small shape's summary; the loser is kept, so a tag that "
                "disagrees is contested"
            ),
        ),
        Order(
            question="run.cluster",
            order=("ne_metadata", "ne_export_config"),
            keep_loser=True,
            because="the cluster the launcher resolved and ran on before the tag an operator typed",
        ),
        Order(
            question="run.started_at",
            order=("ne_eval_metadata", "ne_efm"),
            keep_loser=True,
            because=(
                "eval_metadata's `date` is the launch path's own stamp; eval_factory_metrics' "
                "`run_start` is in the node's local zone"
            ),
        ),
        Order(
            question="run.model",
            order=("ne_results_yml", "ne_eval_metadata", "ne_run_config", "ne_export_config"),
            keep_loser=True,
            because=(
                "what the endpoint served first, then the `model_id` the wrapper was asked "
                "for, and the tag a human chose last"
            ),
        ),
    ),
    ignored=(
        Ignore(
            ENVELOPE,
            CONFIG,
            because="the wrapper's merged configuration, the run's plan; what is stored from it is also in `run_config.yml`",
        ),
        Ignore(
            ENVELOPE,
            "launcher_unresolved_config.yaml",
            because="the launcher's config before resolution",
        ),
        Ignore(
            ENVELOPE,
            "report.json",
            because="the adapter's post-run report, a rendering of the stream",
        ),
        Ignore(ENVELOPE, "report.html", because="the same report as HTML"),
        Ignore(
            ENVELOPE,
            "cache/*/cache.db*",
            because="the adapter's HTTP cache and its SQLite sidecars: the same requests the stream records",
        ),
        Ignore(
            ENVELOPE,
            "reasoning_stats_cache/cache.db*",
            because="a statistics cache the adapter rebuilds from the responses",
        ),
        Ignore(
            ENVELOPE, "response_stats_cache/cache.db*", because="the same, for response statistics"
        ),
        Ignore(
            ENVELOPE, "run_times/*.json", because="the launcher's wall-clock accounting per attempt"
        ),
    ),
)


def launcher_ignores(root: Landmark) -> tuple[Ignore, ...]:
    """What the Slurm launcher writes around the rundir, at the tree root a host names."""
    return (
        Ignore(
            root,
            "export.sbatch",
            because="the Slurm script that uploaded the rundir after the run finished",
        ),
        Ignore(
            root,
            "run.sub",
            because="the Slurm submission script: how the run was launched, not what it produced",
        ),
        Ignore(
            root,
            ".accumulated_walltime",
            because="the launcher's running total of wall-clock time across requeues",
        ),
        Ignore(root, ".job_start_time", because="a Slurm bookkeeping stamp"),
        Ignore(root, ".slurm_job_id.list", because="the Slurm job ids this rundir was produced by"),
        Ignore(
            root,
            "logs/*.log",
            because="the launcher's client, server and export logs; the store has no place for a serving log",
        ),
        Ignore(
            root,
            "proxy.cfg",
            because="the load-balancer configuration generated for the served model, naming addresses that ended with the job",
        ),
        Ignore(
            root,
            "proxy.cfg.template",
            because="the template that configuration was interpolated from",
        ),
    )
