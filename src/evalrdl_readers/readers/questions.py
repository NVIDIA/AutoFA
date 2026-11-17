# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The questions every reader in this package shares, by reference.

`lands` injectivity is checked per reader. `run.benchmark` and `run.suite` are separate
because both are right and an `Order` would delete one.
"""

from __future__ import annotations

from evalrdl_readers.core import Grain, Lane, Merge, Question

# -------------------------------------------------------------------------- run grain

RUN_PRODUCER = Question(
    name="run.producer", grain=Grain.RUN, merge=Merge.ONE, lands="runs.producer",
    means="which harness wrote this tree, as the tree itself says so",
)
RUN_BENCHMARK = Question(
    name="run.benchmark", grain=Grain.RUN, merge=Merge.ONE, lands="runs.benchmark",
    means="the registry benchmark this run is filed under",
)
RUN_SUITE = Question(
    name="run.suite", grain=Grain.RUN, merge=Merge.ONE, lands="runs.suite",
    means=(
        "the internal suite name the producing harness used, which is not the registry "
        "benchmark (`anyswe` for the registry's `swebench-verified`); both are right"
    ),
)
RUN_MODEL = Question(
    name="run.model", grain=Grain.RUN, merge=Merge.ONE, lands="runs.model",
    means="the model under test",
)
RUN_SOURCE_USER = Question(
    name="run.source_user", grain=Grain.RUN, merge=Merge.ONE, lands="runs.source_user",
    means=(
        "who ran it, as the run's own record names them; a converted bundle names the user "
        "in the `conversion-provenance.json` beside the converted stream"
    ),
)
RUN_CLUSTER = Question(
    name="run.cluster", grain=Grain.RUN, merge=Merge.ONE, lands="runs.cluster",
    means=(
        "which cluster the run ran on, as its own export configuration names it in the "
        "mlflow `cluster` tag of `export_config.yml`"
    ),
)
RUN_STARTED = Question(
    name="run.started_at", grain=Grain.RUN, merge=Merge.ONE, lands="runs.run_ts",
    means="when the run began, by whichever clock the tree carries",
)
RUN_EXPECTED = Question(
    name="run.n_expected", grain=Grain.RUN, merge=Merge.ONE, lands="runs.expected_n_rollouts",
    means="how many rollouts the run's own record says there should be",
)
RUN_COST = Question(
    name="run.cost", grain=Grain.RUN, merge=Merge.ONE, lands="runs.cost",
    means="what the run cost in USD",
)
RUN_ADAPTER = Question(
    name="run.adapter_version", grain=Grain.RUN, merge=Merge.ONE,
    lands="runs.adapter_version",
    means="the version of the wrapper that produced this layout, where it records one",
)
RUN_SCORE = Question(
    name="run.score", grain=Grain.RUN, merge=Merge.ONE, lands="runs.score",
    means="the run-level score the producer published",
)
RUN_N_SCORED = Question(
    name="run.n_scored", grain=Grain.RUN, merge=Merge.ONE, lands="runs.n_scored",
    means="how many rollouts this read actually found a score for",
)

# ---------------------------------------------------------------------- rollout grain

ROLLOUT_TASK = Question(
    name="rollout.task", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.task_name",
    means="which task this rollout attempted",
)
ROLLOUT_SCORE = Question(
    name="rollout.score", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.reward",
    means="what this rollout scored",
)
ROLLOUT_ERROR = Question(
    name="rollout.error", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.error",
    means="how this rollout failed, where it did",
)
ROLLOUT_COST = Question(
    name="rollout.cost", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.cost",
    means="what this rollout cost in USD",
)
ROLLOUT_STARTED = Question(
    name="rollout.started_at", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.started_at", means="when this rollout began",
)
ROLLOUT_AGENT_MODEL = Question(
    name="rollout.agent_model", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.agent_model",
    means=(
        "the model the agent believed it was talking to, which under a router is the "
        "router's own name"
    ),
)
ROLLOUT_AGENT_NAME = Question(
    name="rollout.agent_name", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.agent_name",
    means=(
        "the agent harness that ran this rollout, as the trial's own record names it; it "
        "is at rollout grain because harbor's run-level `config.json` does not reliably "
        "write it"
    ),
)
ROLLOUT_VERIFIER = Question(
    name="rollout.verifier", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.verifier_report",
    means="the verifier's own structured report for this rollout",
)
ROLLOUT_STOP = Question(
    name="rollout.stop_reason", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.stop_reason", means="why this rollout stopped",
)
ROLLOUT_EVENTS = Question(
    name="rollout.events", grain=Grain.ROLLOUT, merge=Merge.UNION,
    lands="events",
    means=(
        "the transcript, which several files each hold a part of; it lands in the whole "
        "table because each answer is a mapping, one key per column of one events row"
    ),
)
ROLLOUT_ARTIFACTS = Question(
    name="rollout.artifacts", grain=Grain.ROLLOUT, merge=Merge.UNION,
    lands="artifacts",
    means=(
        "one file found under this rollout, whoever wrote it; it lands in the whole "
        "table for the same reason `rollout.events` does: the answer is a row, not a path"
    ),
)
ROLLOUT_ATTEMPTS = Question(
    name="rollout.attempts", grain=Grain.ROLLOUT, merge=Merge.UNION,
    lands="attempts",
    means=(
        "one try at this rollout that did not finish, with the harness's own verdict on "
        "why. A rollout that succeeded first time has no attempt, so attempts are a "
        "table of their own and not a column on `rollouts`"
    ),
)
ROLLOUT_TOOL_DEFS = Question(
    name="rollout.tool_defs", grain=Grain.ROLLOUT, merge=Merge.UNION,
    lands="tool_defs",
    means=(
        "one distinct tool definition the model was offered in this rollout, its JSON in "
        "`tool_def_texts`; the whole table, because the answer is a row"
    ),
)
ROLLOUT_PROMPT_MESSAGES = Question(
    name="rollout.prompt_messages", grain=Grain.ROLLOUT, merge=Merge.UNION,
    lands="prompt_messages",
    means=(
        "one distinct message this rollout's captured requests sent, named by the hash of its "
        "JSON, once however many requests re-sent it. Whole table: the answer is a row"
    ),
)
ROLLOUT_SUBAGENTS = Question(
    name="rollout.subagents", grain=Grain.ROLLOUT, merge=Merge.UNION,
    lands="rollouts.subagents",
    means="the sub-agents this rollout delegated to, each with its own transcript file",
)

# ------------------------------------------------------------------------- call grain

ROLLOUT_CRITERIA = Question(
    name="rollout.criteria", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.criteria",
    means=(
        "every per-criterion score for this rollout, flattened into one mapping because "
        "a scoring dimension has no rung on the closed Grain ladder"
    ),
)
RUN_OUTDIR = Question(
    name="run.output_dir_name", grain=Grain.RUN, merge=Merge.ONE,
    lands="runs.output_dir_name",
    means=(
        "the last segment of the output directory the operator chose, one of several "
        "signals of what the run actually evaluated"
    ),
)
RUN_AGENT_HARNESS = Question(
    name="run.agent_harness", grain=Grain.RUN, merge=Merge.ONE, lands="runs.agent_harness",
    means=(
        "the agent harness inside the producer, such as `hermes` under nemo-gym or "
        "`claude-code` under harbor; not `runs.harness`, which is the producer family"
    ),
)
RUN_AGENT_MODEL_REF = Question(
    name="run.agent_model_ref", grain=Grain.RUN, merge=Merge.ONE, lands="runs.agent_model_ref",
    means=(
        "the model reference the agent was configured with; under a router it names the "
        "router, not a model, so it is kept apart from `run.model`"
    ),
)
RUN_N_TRIALS = Question(
    name="run.n_trials", grain=Grain.RUN, merge=Merge.ONE, lands="runs.n_trials",
    means="how many trials the orchestrator says it ran",
)
ROLLOUT_N_CALLS = Question(
    name="rollout.n_model_calls", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.n_model_calls",
    means="how many model calls the agent's own metadata counted for this rollout",
)
RUN_N_REPEATS = Question(
    name="run.n_repeats", grain=Grain.RUN, merge=Merge.ONE, lands="runs.n_repeats",
    means="the repeat count the run was configured with, a multiplier over its tasks",
)
RUN_TASK_NAMESPACE = Question(
    name="run.task_namespace", grain=Grain.RUN, merge=Merge.ONE, lands="runs.task_namespace",
    means="a `<namespace>__` prefix every task name carries, which the rollout key drops",
)
ROLLOUT_STAGE = Question(
    name="rollout.stage", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.stage",
    means="which stage of a multistage driver produced this rollout",
)
ROLLOUT_PROMPT = Question(
    name="rollout.prompt", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.prompt",
    means="the prompt this rollout was given, reconstructed where the producer stored it in deltas",
)
ROLLOUT_REPEAT = Question(
    name="rollout.repeat", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.repeat",
    means="which repeat of its task this rollout is, where the producer records one",
)
ROLLOUT_STATUS = Question(
    name="rollout.status", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.status",
    means=(
        "the store's two-word verdict on this rollout, `ok` or `failed`, which is what "
        "the column holds in production and what the page colours on. The producer's own "
        "word for it — `passed`, `runner_error`, a termination reason — is "
        "`rollout.stop_reason`, and that is a different fact with no column"
    ),
)
ROLLOUT_HEALTH = Question(
    name="rollout.health", grain=Grain.ROLLOUT, merge=Merge.ONE, lands="rollouts.health",
    means=(
        "the harness's own health report on this rollout — its verdict, the checks that "
        "failed and the checks it could not observe. It is not `rollout.status`, which "
        "is the store's two-word verdict, nor `ingest_violation`, which is this "
        "bridge's account of its own losses"
    ),
)
ROLLOUT_FINISHED = Question(
    name="rollout.finished_at", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.finished_at", means="when this rollout ended",
)
ROLLOUT_N_TOKENS = Question(
    name="rollout.n_tokens", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.total_tokens",
    means=(
        "how many tokens this rollout spent in total, where the producer counts them; "
        "records are exact where aggregates round"
    ),
)
RUN_SOURCE_RUN_ID = Question(
    name="run.source_run_id", grain=Grain.RUN, merge=Merge.ONE, lands="runs.source_run_id",
    means=(
        "the producer's own identity for this run, as the producer wrote it. `run_id` is "
        "the store's identity, minted here; this is the one a person quotes back to the "
        "team that ran it"
    ),
)
ROLLOUT_N_COMPLETION_TOKENS = Question(
    name="rollout.n_completion_tokens", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.completion_tokens",
    means=(
        "how many tokens this rollout's completions cost, where the producer counts "
        "them. It sits beside `rollout.n_prompt_tokens` because the store keeps the two "
        "halves in two columns and sums them into `runs.total_completion_tokens`"
    ),
)
ROLLOUT_N_PROMPT_TOKENS = Question(
    name="rollout.n_prompt_tokens", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.prompt_tokens",
    means="how many tokens this rollout's prompt cost, where the producer counts them",
)


# ------------------------------------------------------------------- an agent's own logs

#: The lanes an agent's own logs land in, shared so readers mounted side by side agree on them.
AGENT_SESSION_LANE = Lane(
    kind="session", order_by="started_at", identity="source_event_id", actor="main",
    because=(
        "the agent's own transcript: a different author's account of the same minutes as the "
        "harness's trajectory, so the two are not one sequence"
    ),
)
AGENT_BOOKKEEPING_LANE = Lane(
    kind="bookkeeping", order_by="source_event_id", identity="source_event_id", actor="harness",
    because=(
        "what the harness did around the conversation: model and effort switches, compaction "
        "and retry notices, errors and stray text. Ordered by file position, because stream "
        "records carry no clock"
    ),
)
CODEX_UI_LANE = Lane(
    kind="codex-ui", order_by="started_at", identity="source_event_id", actor="main",
    because=(
        "codex writes two accounts of every turn into one rollout log, the responses-API items "
        "and the items its own UI completed"
    ),
)
