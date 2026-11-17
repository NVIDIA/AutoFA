# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The store columns these readers may land in.

`STORE_SCHEMA` is rendered from `evalrdl_schema`, which writer and server both read.
`NO_HOME` lists facts readers produce that the store has no column for, each with why; a
landing in neither is refused rather than dropped.
"""

from __future__ import annotations

from evalrdl_schema import CURRENT

SCHEMA_VERSION: str = CURRENT.version


STORE_SCHEMA: dict[str, tuple[str, ...]] = CURRENT.columns()


NO_HOME: dict[str, str] = {
    "rollouts.health": "the harness's own health verdict on one rollout and the checks behind it (`rollout_verdicts.jsonl`); the store holds `status` for its own two-word verdict and `ingest_violation` for the bridge's, and neither is the producer's",
    # -- run grain ------------------------------------------------------------------
    "runs.telemetry": "lossless OTLP observations and correlation evidence at run grain; normalized analytics storage is a separate schema contract",
    "runs.producer": "which harness wrote the tree; the store infers it into source_format and keeps no field for what the tree itself said",
    "runs.suite": "the harness's internal suite name, which is not the registry benchmark name",
    "runs.score": "a run-level score is a metric over a population, not a property of the run row",
    "runs.n_scored": "how many rollouts a read found a score for; `n_rollouts` is the whole set and scored is a subset, which is the entire point of the number",
    "runs.n_trials": "the orchestrator's own trial count, a claim to hold beside the enumeration rather than instead of it",
    "runs.n_tasks": "the manifest's task count, which may exceed the task directories present",
    "runs.n_stages": "how many stages a multistage driver declared",
    "runs.n_agents": "how many agents the run configured",
    "runs.n_dataset_rows": "how many rows the input dataset held",
    "runs.task_namespace": "a prefix every task name carries; the rollout key drops it and `rollouts.task_name` keeps the producer's whole name",
    "runs.n_planned": "how many rollouts the run prepared; the store holds it only as `expected_n_rollouts` with `expected_source = task_list`, where the run states no count",
    "runs.n_repeats": "the configured repeat multiplier; the store uses it only to scale a registry row's `expected_tasks`",
    "runs.n_manifests": "how many manifests a harbor upload wrote; an upload may write more than one",
    "runs.n_uploaded": "how many trials the uploader says it sent",
    "runs.cost": "what the run cost in USD",
    "runs.stop_reason": "why the run stopped",
    "runs.output_dir_name": "the last segment of the operator's output directory, sometimes the only benchmark signal in the tree",
    "runs.serving": "the inference server's own ordered telemetry, which belongs to the run because one server serves every rollout; `events` is keyed by task, which a run-grain sequence cannot supply",
    "runs.task_metrics": "every metric nemo-skills published, at both levels; there is no metrics table or metric grain, and `part_texts` is keyed by event, which a run-grain answer cannot supply",
    "runs.model_calls": "one HTTP call to a model; `tool_calls` holds the agent's calls, keyed by event, and a capture file knows no event",
    "runs.artifacts": "a file that belongs to the run rather than to a rollout; `artifacts` is keyed by rollout",
    "runs.adapter_version": "the version of the wrapper that produced this layout",
    "runs.agent_model_ref": "the model reference the agent was configured with, which under a router is not the model under test",
    "runs.agent_harness": "the agent harness inside the producer; `harness` is the producer family, filled from the reader's identity",
    # -- rollout grain --------------------------------------------------------------
    "rollouts.task_name": "the task's name; `task_id` is a key column and carries the identity, not the name",
    "rollouts.results_record": "which `results.jsonl` record a rollout is; a co-owner of the record joins on it",
    "rollouts.trajectory_record": "which `trajectories.jsonl` record a rollout is; a co-owner of the record joins on it",
    "rollouts.atif_results_record": "the `results.jsonl` record an ATIF document was read from; the other end of that join",
    "rollouts.atif_trajectory_record": "the `trajectories.jsonl` record an ATIF document was read from; the other end of that join",
    "rollouts.pi_session": "the pi session a trial's trajectory names; it joins pi's logs to the trial",
    "rollouts.pi_log": "the session id pi's session file names itself by; the other end of that join",
    "rollouts.codex_session": "the codex session a trial's ATIF document names; it joins codex's logs to the trial",
    "rollouts.codex_log": "the session id codex's rollout file names itself by; the other end of that join",
    "rollouts.nooa_session": "the NOOA session a host's rollout record names; it joins NOOA's session stores to the rollout",
    "rollouts.nooa_log": "the root session id NOOA's session store names itself by; the other end of that join",
    "rollouts.stop_reason": "why this rollout stopped. `status` is the store's own classification of the record, not the producer's report",
    "rollouts.cost": "what this rollout cost in USD",
    "rollouts.stage": "which stage of a multistage driver produced this rollout",
    "rollouts.prompt": "the prompt this rollout was given",
    "rollouts.criteria": "every per-criterion score, flattened, because a scoring dimension has no rung on the grain ladder",
    "rollouts.verifier_report": "the verifier's own structured report",
    "rollouts.agent_name": "the agent harness that ran this rollout; it lands as the key column `agent_ref`, which the bridge keys the rollout by",
    "rollouts.agent_model": "the model the agent believed it was talking to, which under a router is the router's own name",
    "rollouts.deliverables_dir": "the gdpval repeat directory a rollout's files were read from; its files land as `artifacts`, and the directory is only the join to the stream record",
    "rollouts.n_model_calls": "how many model calls this rollout made; `n_tool_calls` differs because one model call can hold many tool calls",
    "rollouts.subagents": "the sub-agents this rollout delegated to, each with its own transcript",
    # -- event grain ----------------------------------------------------------------
    # Keys of the `rollout.events` mapping, resolved against the landing table at read time.
    # `content` (home: `part_texts.text`) and `agent` (`events.actor` unruled) are omitted.
    "events.tokens_total": "the call's total token count, the sum of `tokens_in` and `tokens_out`, which both land",
    "events.status_code": "the HTTP status, constant across captured calls; failed calls carry no status and are named by `error_category`",
    "events.cache_hit": "whether the call was served from cache, derived from `cached_tokens`, which lands and carries more",
    "events.cache_creation_tokens": "tokens written into the cache; the producer writes the key with no value",
    "events.read_seq": "the position of this item in the producer's own arrays, zero-padded so it compares as a string; the order it gives is kept as `events.event_idx`",
    "events.n_tokens_estimated": "a token count the producer marked `approximate_char_split`; the store's token columns mean a provider-reported count",
    "events.answers_call_ids": "the calls this event is the result of, in the producer's own id space; the store's columns for this edge hold positions, resolved from these ids after the lane is ordered",
    "events.parent": "the event this one answers, in the producer's own id space; the store has no self-edge column",
    "events.agent_role": "what role the agent that produced this event was playing in its delegation tree",
    "events.agent_depth": "how deep in the delegation tree this event was produced",
    "events.attempt": "which attempt of this event's step produced it; the store's `attempts` table counts rollout attempts, which is a different population",
    "events.retry_of": "the event this one retries, in the producer's own id space; `events.parent_event_id` holds a different edge",
    "events.thinking_complete": "whether the producer believes the reasoning block it emitted was finished",
    "events.n_prompt_messages": "how many prompt messages this model request carried; `prompt_tokens` counts tokens, not messages",
    "events.session": "which session file of a multi-file transcript this event came from; the store flattens the forest and keeps no column for its trunk",
    "events.side": "which half of a responses-api record this event came from (`input` or `output`); `request_idx` numbers requests, it does not name a side",
    "artifacts.path": "the path an artifact was found at; `uri` is where the store put it, a different fact",
}

#: A table whose long strings the write step moves to a sibling keyed like it; a row of the
#: table carries them by the sibling's column name. The write step reads this same map.
TEXT_SIBLING: dict[str, str] = {
    "events": "event_texts", "parts": "part_texts", "tool_calls": "tool_call_texts",
    "artifacts": "artifact_texts", "tool_defs": "tool_def_texts",
}

#: What every `Reader` here registers: store columns, their text siblings' columns, and
#: `NO_HOME`.
READER_SCHEMA: dict[str, tuple[str, ...]] = {
    table: cols + tuple(
        c for c in STORE_SCHEMA.get(TEXT_SIBLING.get(table, ""), ()) if c not in cols
    ) + tuple(
        c for slot, _why in NO_HOME.items()
        for t, _, c in (slot.partition("."),)
        if t == table
    )
    for table, cols in STORE_SCHEMA.items()
}
