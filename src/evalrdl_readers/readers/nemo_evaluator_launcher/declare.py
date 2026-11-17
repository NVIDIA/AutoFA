# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evaluator's run read on its own: its manifest, each record's envelope, its proxy's
call log and its rollups. Harbor's documents inside its records, an agent's logs in
`agent_logs/` and a converter's bundle around it are those producers' readers, mounted on this
one (`readers.mounts`).
"""

from __future__ import annotations

from evalrdl_readers.core import (
    ROOT,
    At,
    Card,
    Compare,
    Contribute,
    Decoders,
    Fold,
    Grain,
    Ignore,
    Input,
    Key,
    Lane,
    Merge,
    Op,
    Order,
    Population,
    Question,
    Reader,
    Rows,
)

from .. import questions as Q
from ..dialects.atif import ATIF_CANDIDATES, TRAJECTORY_BECAUSE
from ..store_schema import READER_SCHEMA
from .landmarks import RUN
from .traffic import NEL_TRAFFIC
from .translate import (
    SINGLE_TURN,
    nel_manifest,
    nel_own_events,
    nel_results,
    nel_scorer_logs,
    nel_traj_meta,
)

RUN_N_MANIFESTS = type(Q.RUN_N_TRIALS)(
    name="run.n_manifests",
    grain=Q.RUN_N_TRIALS.grain,
    merge=Q.RUN_N_TRIALS.merge,
    lands="runs.n_manifests",
    means=(
        "how many run manifests the bench directory holds; more than one is common and "
        "otherwise invisible"
    ),
)
RESULTS_RECORD = Question(
    name="rollout.results_record",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.results_record",
    means="the `results.jsonl` record a rollout is, by file and line (`record_ref`)",
)
TRAJECTORY_RECORD = Question(
    name="rollout.trajectory_record",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.trajectory_record",
    means="the `trajectories.jsonl` record a rollout is, by file and line (`record_ref`)",
)

NEL_MANIFEST = Input(
    name="nel_manifest",
    via=nel_manifest,
    reads=[At(RUN, "eval-*.json")],
    answers={
        "run.benchmark": Card.ONE(because="config.benchmark is written before the run starts"),
        "run.source_run_id": Card.ONE(
            because=(
                "the manifest names the run the evaluator thinks it ran, written before the run "
                "starts alongside the benchmark"
            )
        ),
        "run.model": Card.ONE(because="config.model is written before the run starts"),
        "run.started_at": Card.ONE(
            because="every manifest carries a timestamp, even where the catalog's run_ts is null"
        ),
        "run.n_expected": Card.ONE(
            because=(
                "`n_results` is the manifest's claim about how many rollouts exist and can "
                "disagree with the streams, so it is stored beside the enumeration rather than "
                "instead of it"
            )
        ),
        "run.n_manifests": Card.ONE(
            because="counting the manifests is the only way this reader can say there was more than one"
        ),
        "run.score": Card.ONE(
            because="benchmark.scores.pass@1 is written by the scorer on every completed manifest"
        ),
    },
)
NEL_RESULTS = Input(
    name="nel_results",
    via=nel_results,
    reads=[At(RUN, "results.jsonl")],
    answers={
        "rollout.results_record": Card.ONE(because="every line is a record"),
        "rollout.score": Card.ONE(because="a line in results.jsonl exists because it has a reward"),
        "rollout.task": Card.ANY(
            because=(
                "a record spells the task as `metadata.task_id` or `seed_metadata.task_id`, never "
                "both, and may carry neither"
            )
        ),
        "rollout.repeat": Card.ONE(because="repeat is on every line"),
        "rollout.n_tokens": Card.ANY(
            because=(
                "the swebench layouts write `tokens` and the terminal-bench ones do not; where it "
                "is absent the translator stays silent rather than answering a null that would win "
                "the settle against `nel_traj_meta`"
            )
        ),
        "rollout.status": Card.ANY(
            because=(
                "only some layouts write `status`; the others say nothing about how the rollout "
                "ended and this reader will not invent it"
            )
        ),
        "rollout.stop_reason": Card.ANY(
            because="the same records, keeping the producer's own word beside the store's verdict of it"
        ),
    },
    mints=(
        Key(
            "results_record",
            because="a record of `results.jsonl`, by file and line, the name a co-owner of the record joins on",
        ),
    ),
)
NEL_TRAJ_META = Input(
    name="nel_traj_meta",
    via=nel_traj_meta,
    reads=[At(RUN, "trajectories.jsonl")],
    answers={
        "rollout.trajectory_record": Card.ANY(
            because=(
                "every line is a record; a run whose trajectories.jsonl is absent is reported "
                "once, by this input's missing file"
            )
        ),
        "rollout.score": Card.ONE(
            because="every trajectory record repeats the reward its results.jsonl twin carries"
        ),
        "rollout.task": Card.ONE(
            because="`seed_metadata.task_id` is the same task name results.jsonl carries under `metadata.task_id`"
        ),
        "rollout.agent_model": Card.ONE(
            because="`model.model` names the model that produced this record"
        ),
        "rollout.n_tokens": Card.ANY(
            because="`model.tokens` is a mapping carrying `total`, a scalar total, or absent"
        ),
        "rollout.n_prompt_tokens": Card.ANY(
            because=(
                "`model.tokens.prompt` is often written as null, which is the sampler not counting "
                "the prompt rather than a prompt of no tokens"
            )
        ),
        "rollout.n_completion_tokens": Card.ANY(
            because="`model.tokens.completion`, present wherever the mapping is"
        ),
        "rollout.error": Card.ANY(
            because=(
                "`failure_category` is on every record and null where it succeeded, so a rollout "
                "that worked has no error to state"
            )
        ),
        "rollout.started_at": Card.ONE(because="every trajectory record carries a unix timestamp"),
    },
    mints=(
        Key(
            "trajectory_record",
            because="a record of `trajectories.jsonl`, by file and line, the name a co-owner of the record joins on",
            optional=True,
        ),
    ),
)
NEL_OWN_EVENTS = Input(
    name="nel_own_events",
    via=nel_own_events,
    # Same content in both; results.jsonl is sometimes the only copy.
    reads=[At(RUN, "trajectories.jsonl"), At(RUN, "results.jsonl")],
    decoders=Decoders(
        candidates=ATIF_CANDIDATES + (SINGLE_TURN,),
        because=(
            "ATIF version is a property of the document, and one stream can interleave "
            "versions, so sniffing the first record would misread the rest"
        ),
    ),
    answers={
        "rollout.n_prompt_tokens": Card.ANY(
            because="only a document the evaluator wrote itself is read here, and few have totals"
        ),
        "rollout.n_completion_tokens": Card.ANY(
            because="the same block, absent from the same documents"
        ),
        "rollout.events": Card.ANY(
            because=(
                "a single-turn record, or an ATIF document the evaluator wrote around an agent "
                "that writes none; every other trajectory is the agent harness's"
            )
        ),
        "rollout.artifacts": Card.ANY(
            because="the workspace diff of a document the evaluator wrote, where the agent changed files"
        ),
    },
)
NEL_SCORER_LOGS = Input(
    name="nel_scorer_logs",
    via=nel_scorer_logs,
    reads=[At(RUN, "results.jsonl"), At(RUN, "trajectories.jsonl")],
    answers={
        "rollout.artifacts": Card.ANY(
            because="the scorer's `scoring_details.test_stdout` and `test_stderr`; a single-turn record carries neither"
        ),
    },
)

TOKENS_BECAUSE = (
    ". Below the ATIF decoders sits `nel_traj_meta`, which reads the same record's "
    "`model.tokens` block and can meet them; the trajectory's `final_metrics` wins as the rollup "
    "over the steps stored, and a disagreement is contested rather than overwritten"
)
OWN_TOKEN_SOURCES = tuple(f"nel_own_events/{d.name}" for d in ATIF_CANDIDATES)[::-1] + (
    "nel_traj_meta",
    f"nel_own_events/{SINGLE_TURN.name}",
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,  # a results.jsonl line per attempted problem/repeat
    name="nemo-evaluator-launcher",
    landmarks=(RUN,),
    inputs=(
        (
            "nemo-evaluator-launcher",
            (
                NEL_MANIFEST,
                NEL_RESULTS,
                NEL_TRAJ_META,
                NEL_OWN_EVENTS,
                NEL_SCORER_LOGS,
                NEL_TRAFFIC,
            ),
        ),
    ),
    questions=(
        Q.RUN_BENCHMARK,
        Q.RUN_MODEL,
        Q.RUN_STARTED,
        Q.RUN_EXPECTED,
        Q.RUN_SCORE,
        Q.RUN_N_SCORED,
        RUN_N_MANIFESTS,
        Q.RUN_SOURCE_RUN_ID,
        Q.ROLLOUT_SCORE,
        Q.ROLLOUT_TASK,
        Q.ROLLOUT_REPEAT,
        Q.ROLLOUT_AGENT_MODEL,
        Q.ROLLOUT_ERROR,
        Q.ROLLOUT_STARTED,
        Q.ROLLOUT_EVENTS,
        Q.ROLLOUT_STATUS,
        Q.ROLLOUT_STOP,
        Q.ROLLOUT_N_TOKENS,
        Q.ROLLOUT_N_PROMPT_TOKENS,
        Q.ROLLOUT_N_COMPLETION_TOKENS,
        Q.ROLLOUT_ARTIFACTS,
        Q.ROLLOUT_PROMPT_MESSAGES,
        Q.ROLLOUT_TOOL_DEFS,
        RESULTS_RECORD,
        TRAJECTORY_RECORD,
    ),  # fmt: skip
    lanes=(
        Lane(
            kind="model_call",
            order_by="started_at",
            identity="source_event_id",
            because=(
                "the proxy's account of the rollout's model calls, written by a different "
                "process than the agent's trajectory: the same minutes, not the same sequence. "
                "`actor` is unset because the author is the proxy, not an agent"
            ),
        ),
    ),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=("nel_results", "nel_traj_meta"),
            because=(
                "results.jsonl and trajectories.jsonl each carry one record per rollout and are "
                "the only two enumerations this layout has; every other input reads into a "
                "rollout they name and may fill it in, never name one"
            ),
        ),
        Rows(
            question="rollout.score",
            may_create=("nel_results", "nel_traj_meta"),
            because="results.jsonl and the trajectory file each carry one record per rollout",
        ),
        Rows(
            question="rollout.task",
            may_create=("nel_results", "nel_traj_meta"),
            because="the same two per-rollout records",
        ),
    ),
    orders=(
        Order(
            question="rollout.n_tokens",
            order=("nel_results", "nel_traj_meta"),
            keep_loser=True,
            because=(
                "results.jsonl writes a scalar `tokens` and the trajectory record writes "
                "`model.tokens`, the same total split into prompt, completion and reasoning; "
                "results.jsonl ranks first as the stream holding every rollout, and a "
                "disagreement is contested, not overwritten"
            ),
        ),
        Order(
            question="rollout.n_prompt_tokens",
            order=OWN_TOKEN_SOURCES,
            keep_loser=True,
            because=TRAJECTORY_BECAUSE + TOKENS_BECAUSE,
        ),
        Order(
            question="rollout.n_completion_tokens",
            order=OWN_TOKEN_SOURCES,
            keep_loser=True,
            because=TRAJECTORY_BECAUSE + TOKENS_BECAUSE,
        ),
        Order(
            question="rollout.task",
            order=("nel_results", "nel_traj_meta"),
            keep_loser=True,
            because=(
                "`metadata.task_id` and `seed_metadata.task_id` are the same string where both "
                "exist; results.jsonl ranks first as the stream that holds every rollout"
            ),
        ),
        Order(
            question="rollout.score",
            order=("nel_results", "nel_traj_meta"),
            keep_loser=True,
            because=(
                "results.jsonl is the scorer's own record and the trajectory record copies it, so "
                "the copy loses; the two streams need not even agree on how many rollouts exist"
            ),
        ),
    ),
    folds=(
        Fold(
            into="run.n_scored",
            over="rollout.score",
            op=Op.COUNT,
            expect=Card.ANY(
                because="a rollout present in one stream and not the other still answers, so there is nothing to withhold",
            ),
            because=(
                "the manifest's `n_results` is a claim and this is the enumeration; the Compare "
                "below says when they disagree"
            ),
        ),
    ),
    compare=(
        Compare(
            left="run.n_scored",
            right="run.n_expected",
            must="equal",
            because=(
                "the evaluator writes one results.jsonl line per attempted problem/repeat, so the "
                "manifest's `n_results` and the count of scored rollouts measure one number; where "
                "they differ the folder does not hold what its producer says it ran"
            ),
        ),
    ),
    ignored=(
        Ignore(
            RUN,
            "runtime_stats.json",
            because="the launcher's totals over the run: tokens, latency percentiles, model errors",
        ),
        Ignore(
            RUN,
            "failure_analysis.json",
            because=(
                "the failed rollouts again, grouped by category, with the error each stream "
                "record carries"
            ),
        ),
        Ignore(
            RUN,
            "verified_log.jsonl",
            because=(
                "the scorer's per-rollout log. MEASURED on 12 trials of 4 runs: reward, scoring "
                "details and response equal `results.jsonl`'s on every one"
            ),
        ),
        Ignore(
            RUN,
            "inference_log.jsonl",
            because=(
                "the solver's per-rollout log and a config hash. MEASURED on 12 trials of 4 "
                "runs: response, tokens, prompt and trajectory equal `results.jsonl`'s on every "
                "one; its usage totals are the sums of `model_traffic.jsonl`"
            ),
        ),
        Ignore(
            RUN,
            "trajectories_enriched.jsonl",
            because=(
                "the trajectory stream with the proxy's records copied in "
                "(`extra.captured_model_calls`). MEASURED: 66 of 66 copied calls equal their "
                "`model_traffic.jsonl` record, which `nel_traffic` reads"
            ),
        ),
        Ignore(
            ROOT,
            "nel_run.json",
            because=(
                "the launcher's own run id, start time and output path; `runs.source_run_id` is "
                "the manifest's id by definition and no column holds a second one"
            ),
        ),
        Ignore(
            ROOT,
            "checkpoint.json",
            because="the launcher's resume record: which benchmarks finished, and where their files are",
        ),
        Ignore(
            ROOT,
            "full_config.yaml",
            because=(
                "the resolved run configuration (endpoint, proxy interceptors, sampling, "
                "sandbox): the model and benchmark are the manifest's, the sampling each request "
                "actually sent is its `request_params`, the rest has no column"
            ),
        ),
        Ignore(
            ROOT,
            "report.md",
            because="the launcher's rendering of the manifest's scores as a table",
        ),
        Ignore(
            ROOT,
            "trajectories_report.json",
            because="a rollup of the trajectory stream: rewards, failure categories, empty responses",
        ),
    ),
    attach=(
        Contribute(
            to="rollout.events",
            input="nel_own_events",
            order_by="started_at",
            identity="source_event_id",
            because="ATIF numbers its own steps within a document and stamps its own timestamps",
        ),
        Contribute(
            to="rollout.artifacts",
            input="nel_scorer_logs",
            order_by="name",
            identity="sha256",
            because="the logs are named by their field; the digest identifies one, as it does every other artifact",
        ),
        Contribute(
            to="rollout.artifacts",
            input="nel_own_events",
            order_by="name",
            identity="sha256",
            because="a patch has no name; its digest identifies it",
        ),
        Contribute(
            to="rollout.events",
            input="nel_traffic",
            order_by="started_at",
            identity="source_event_id",
            because="the proxy stamps each record with the time the call started and its own request id",
        ),
        Contribute(
            to="rollout.prompt_messages",
            input="nel_traffic",
            order_by="msg_hash",
            identity="msg_hash",
            because=(
                "a message is named by the hash of its JSON, so a message re-sent by any call is "
                "one row and its row number depends only on which messages the rollout sent"
            ),
        ),
        Contribute(
            to="rollout.tool_defs",
            input="nel_traffic",
            order_by="def_idx",
            identity="def_idx",
            because=(
                "one row per distinct definition JSON, numbered in the order the rollout's "
                "requests first offered it; that number is the only order and, per rollout, the "
                "identity"
            ),
        ),
    ),
)
