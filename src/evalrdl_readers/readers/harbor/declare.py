# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from evalrdl_readers.core import (
    ROOT,
    Answer,
    At,
    Card,
    Contribute,
    Decoders,
    Fold,
    Grain,
    Ignore,
    Input,
    Join,
    Key,
    Lane,
    Op,
    Order,
    Population,
    Reader,
    Relate,
    Rows,
    Sources,
    Unreadable,
)

from .. import questions as Q
from ..common import INTAKE_RECEIPT, event_kind, site_key, text_parts
from ..dialects.atif import ATIF_CANDIDATES, TRAJECTORY_BECAUSE
from ..dialects.sessions import SESSION_CANDIDATES
from ..store_schema import READER_SCHEMA
from .decoders import CTRF, RCA_REPORT, REWARD_V46, REWARD_V47, SCALEAI_OUT
from .landmarks import JOB, OWN_CONFIG, TRIAL
from .translate import (
    job_config,
    job_result,
    trial_agent_result,
    trial_agent_store,
    trial_artifacts,
    trial_criteria,
    trial_decision,
    trial_exception,
    trial_result,
    trial_reward,
    trial_reward_json,
    trial_reward_txt,
    trial_session,
    trial_subagent,
    trial_subagent_meta,
    trial_tool_results,
    trial_traces,
    trial_trajectory,
    trial_verifier,
    trial_verifier_logs,
    trial_verifier_reports,
    trial_verifier_result,
)

# --------------------------------------------------------------------------- inputs

TRIAL_RESULT = Input(
    name="harbor_trial", via=trial_result, reads=[At(TRIAL, "result.json")],
    answers={
        "rollout.task": Card.ONE(because="harbor writes task_name on every result.json it writes"),
        "rollout.started_at": Card.ONE(because="harbor stamps started_at on every trial it starts"),
        "rollout.finished_at": Card.ANY(because="a trial harbor started and never finished has no `finished_at`, which is the fact this column holds"),
        "rollout.cost": Card.ANY(because="the key is always present and null where the agent did not price its run; a null is the producer's statement that it has no cost, not a missing answer"),
        "rollout.agent_model": Card.ONE(because="agent_info is written when the agent is constructed, before it can fail"),
        "rollout.error": Card.ANY(because="`exception_info` is always present and null on success, so a null is a trial with no error rather than an error nobody recorded"),
        "rollout.agent_name": Card.ONE(because="`agent_info.name` is written when the agent is constructed"),
    },
)

TRIAL_REWARD = Input(
    name="harbor_reward", via=trial_reward, reads=[At(TRIAL, "result.json")],
    decoders=Decoders(
        candidates=(REWARD_V46, REWARD_V47),
        because=(
            "the rollup key inside `verifier_result.rewards` was renamed between adapter "
            "versions, from `composite_score` to `reward`"
        ),
    ),
    answers={
        "rollout.score": Card.ANY(
            because=(
                "a trial that crashed before the verifier ran has `verifier_result` null; "
                "a trial that threw after it ran is still scored"
            ),
        )
    },
)

TRIAL_VERIFIER_RESULT = Input(
    name="harbor_verifier_result",
    via=trial_verifier_result,
    reads=[At(TRIAL, "result.json")],
    answers={
        "rollout.artifacts": Card.ANY(
            because="a completed verifier_result is retained whole as outcome evidence",
        ),
    },
)

TRIAL_REWARD_TXT = Input(
    name="harbor_reward_txt", via=trial_reward_txt,
    reads=[At(TRIAL, "verifier/reward.txt")],
    answers={
        "rollout.score": Card.ANY(because="the verifier writes reward.txt only after it has run to completion"
        )
    },
)

TRIAL_REWARD_JSON = Input(
    name="harbor_reward_json", via=trial_reward_json,
    reads=[At(TRIAL, "verifier/reward.json")],
    answers={
        "rollout.score": Card.ANY(
            because="only the v46 criteria verifier writes `verifier/reward.json`",
        )
    },
)

TRIAL_CRITERIA = Input(
    name="harbor_criteria", via=trial_criteria,
    reads=[At(TRIAL, "verifier/criteria/*/reward.json")],
    answers={
        "rollout.criteria": Card.ANY(because="only the v46 criteria verifier writes a criteria tree"
        ),
        "rollout.artifacts": Card.ANY(
            because=(
                "each criterion's reward.json kept whole, as production keeps it: the "
                "mapping above holds only the score and drops the rest of the document"
            ),
        ),
    },
)

TRIAL_EXCEPTION = Input(
    name="harbor_exception", via=trial_exception,
    reads=[At(TRIAL, "exception.txt")],
    answers={
        "rollout.error": Card.ANY(
            because="`exception.txt` exists only where the trial threw",
        )
    },
)

TRIAL_DECISION = Input(
    name="harbor_decision", via=trial_decision,
    reads=[At(TRIAL, "agent/decision.json")],
    answers={
        "rollout.stop_reason": Card.ANY(because="only the v46 nooa-bench agent writes a decision"
        )
    },
)

TRIAL_VERIFIER = Input(
    name="harbor_verifier", via=trial_verifier,
    reads=[
        At(TRIAL, "verifier/ctrf.json"),
        At(TRIAL, "verifier/output.json"),
        At(TRIAL, "verifier/report.json"),
    ],
    decoders=Decoders(
        candidates=(CTRF, SCALEAI_OUT, RCA_REPORT),
        because=(
            "the verifier's report shape is per-benchmark, not per-producer: one producer "
            "writes `ctrf.json` for terminal-bench and `output.json` for swebench-pro"
        ),
    ),
    answers={
        "rollout.artifacts": Card.ANY(
            because=(
                "a CTRF report is kept whole as the verifier's artifact, as production keeps "
                "it, because `rollout.verifier` has no column; not every trial writes one"
            ),
        ),
        "rollout.verifier": Card.ANY(because="a trial whose verifier never ran writes no report at all"
        )
    },
)

TRIAL_VERIFIER_REPORTS = Input(
    name="harbor_verifier_reports",
    via=trial_verifier_reports,
    reads=[
        At(TRIAL, "verifier/output.json"),
        At(TRIAL, "verifier/report.json"),
    ],
    answers={
        "rollout.artifacts": Card.ANY(
            because=(
                "the other recognized reports are kept whole as outcome evidence; an input of "
                "their own, so an artifact never decides the rollout's record schema"
            ),
        ),
    },
)

TRIAL_VERIFIER_LOGS = Input(
    name="harbor_verifier_logs",
    via=trial_verifier_logs,
    reads=[
        At(TRIAL, "verifier/test-stdout.txt"),
        At(TRIAL, "verifier/run-script-std*.txt"),
    ],
    answers={
        "rollout.artifacts": Card.ANY(
            because="raw verifier logs can contain decisive diagnostics omitted by reports",
        ),
    },
)

TRIAL_TRAJECTORY = Input(
    name="harbor_trajectory", via=trial_trajectory,
    reads=[At(TRIAL, "agent/trajectory.json")],
    decoders=Decoders(
        candidates=ATIF_CANDIDATES,
        because=(
            "ATIF version is a property of the record, and one trajectories file can "
            "interleave records of more than one version"
        ),
    ),
    answers={
        "rollout.events": Card.ANY(because="a trial that crashed during setup wrote no trajectory"),
        "rollout.n_prompt_tokens": Card.ANY(
            because=(
                "`final_metrics.total_prompt_tokens` is missing from some ATIF "
                "trajectories, and the file itself is absent from trials that crashed "
                "during setup"
            ),
        ),
        "rollout.n_completion_tokens": Card.ANY(
            because="the same block, absent from the same trajectories",
        ),
    },
    mints=(Key(
            "atif_step",
            optional=True,  # a rollout without a trajectory file emits no step id
            because="the step_id the trajectory carries, prefixed by its trial"),),
)

def trial_mini_swe(src: Sources):
    """mini-swe-agent's own transcript, sitting inside a harbor trial tree."""
    if not src.has(0):
        return
    doc = src.doc(0).json()
    if not isinstance(doc, dict):
        yield Unreadable("rollout.events", where=str(src.site),
                         detail="mini-swe-agent.trajectory.json is not an object; "
                                "nothing here can say what it is instead")
        return
    key = (site_key(src.site),)
    for i, msg in enumerate(doc.get("messages") or ()):
        if not isinstance(msg, dict):
            continue
        yield Answer(
            "rollout.events",
            {
                "source_event_id": f"mini-swe:{i}",
                "read_seq": f"{i:09d}",
                "role": msg.get("role"),
                **event_kind(msg.get("role")),
                "parts": text_parts(msg.get("content")),
                "tool_calls": [],
            },
            key=key, input_key=str(i),
        )


TRIAL_MINI_SWE = Input(
    name="harbor_mini_swe", via=trial_mini_swe,
    reads=[At(TRIAL, "agent/mini-swe-agent.trajectory.json")],
    answers={
        "rollout.events": Card.ANY(
            because="only the mini-swe-agent harness writes this file; every other harbor "
                    "trial writes `agent/trajectory.json`, which `harbor_trajectory` reads"),
    },
)

TRIAL_SESSION = Input(
    name="harbor_session", via=trial_session,
    reads=[
        At(TRIAL, "agent/sessions/projects/*/*.jsonl"),
        At(TRIAL, "agent/sessions/*/*/*/rollout-*.jsonl"),
    ],
    decoders=Decoders(
        candidates=SESSION_CANDIDATES,
        because=(
            "which agent wrote the log decides how a record reads, and one file interleaves "
            "several record shapes; picking per record makes an unrecognised record a "
            "finding instead of an event with no content"
        ),
    ),
    answers={
        "rollout.events": Card.ANY(
            because="the raw agent session is archived for some trials only",
        )
    },
)

TRIAL_SUBAGENT = Input(
    name="harbor_subagent", via=trial_subagent,
    reads=[At(TRIAL, "agent/sessions/projects/*/*/subagents/agent-*.jsonl")],
    answers={
        "rollout.events": Card.ANY(because="only a trial that delegated has a `subagents/` directory"
        )
    },
    mints=(Key(
            "subagent",
            optional=True,  # a run with no sub-agent writes no .meta.json
            because="the sub-agent file's own stem, prefixed by its trial"),),
)

TRIAL_SUBAGENT_META = Input(
    name="harbor_subagent_meta", via=trial_subagent_meta,
    reads=[At(TRIAL, "agent/sessions/projects/*/*/subagents/agent-*.meta.json")],
    answers={
        "rollout.subagents": Card.ANY(because="a .meta.json sits beside every sub-agent transcript and nowhere else"
        )
    },
)

TRIAL_TOOL_RESULTS = Input(
    name="harbor_tool_results", via=trial_tool_results,
    reads=[At(TRIAL, "agent/sessions/projects/*/*/tool-results/*.txt")],
    answers={
        "rollout.events": Card.ANY(
            because="a tool result is spilled to its own file only when it is large",
        ),
        "rollout.artifacts": Card.ANY(
            because="the event is a preview; the complete side-file remains pageable evidence",
        ),
    },
)

TRIAL_TRACES = Input(
    name="harbor_traces", via=trial_traces,
    reads=[
        At(TRIAL, "artifacts/traces/*.jsonl"),
        At(TRIAL, "agent/artifacts/traces/*.jsonl"),
    ],
    answers={
        "rollout.events": Card.ANY(
            because="trace availability is per-agent, not per-adapter-version",
        ),
        "rollout.n_prompt_tokens": Card.ANY(
            because=(
                "`llm.token_count.prompt` of the last LLM span, which is cumulative: every "
                "call re-sends the history. A trace with no LLM span has no count to give"
            ),
        ),
        "rollout.n_completion_tokens": Card.ANY(
            because="the sum of `llm.token_count.completion` over the LLM spans, the same spans",
        ),
    },
)

TRIAL_AGENT_RESULT = Input(
    name="harbor_agent_result", via=trial_agent_result,
    reads=[At(TRIAL, "agent/result.json")],
    answers={
        "rollout.events": Card.ANY(
            because=(
                "only some agents write this file, and only the LangGraph agent's holds a "
                "transcript under `messages`; the others write a self-report with no such key"
            ),
        )
    },
)

TRIAL_ARTIFACTS = Input(
    name="harbor_artifacts", via=trial_artifacts,
    reads=[At(TRIAL, "artifacts/**")],
    answers={
        "rollout.artifacts": Card.ANY(because="an agent that wrote no file into the collected directory is legal"
        )
    },
)

TRIAL_AGENT_STORE = Input(
    name="harbor_agent_store", via=trial_agent_store,
    reads=[At(TRIAL, "agent/opencode/xdg-data/opencode/opencode.db*")],
    answers={
        "rollout.artifacts": Card.ANY(because="only the opencode agent keeps a SQLite store"
        )
    },
)

TRIAL_INPUTS = (
    TRIAL_RESULT,
    TRIAL_REWARD,
    TRIAL_VERIFIER_RESULT,
    TRIAL_REWARD_TXT,
    TRIAL_REWARD_JSON,
    TRIAL_CRITERIA,
    TRIAL_EXCEPTION,
    TRIAL_DECISION,
    TRIAL_VERIFIER,
    TRIAL_VERIFIER_REPORTS,
    TRIAL_VERIFIER_LOGS,
    TRIAL_TRAJECTORY,
    TRIAL_MINI_SWE,
    TRIAL_SESSION,
    TRIAL_SUBAGENT,
    TRIAL_SUBAGENT_META,
    TRIAL_TOOL_RESULTS,
    TRIAL_TRACES,
    TRIAL_AGENT_RESULT,
    TRIAL_ARTIFACTS,
    TRIAL_AGENT_STORE,
)

JOB_RESULT = Input(
    name="harbor_job", via=job_result, reads=[At(JOB, "result.json")],
    answers={
        "run.source_run_id": Card.ANY(because="harbor's job id, the only id harbor gives the run"),
        "run.n_trials": Card.ONE(because="harbor writes n_total_trials into every job result"),
        "run.started_at": Card.ONE(because="harbor stamps started_at when the job begins"),
        "run.cost": Card.ANY(because="stats.cost_usd is null where no agent priced its run, and nothing is yielded for a null"),
        "run.score": Card.ANY(because="an eval block with no trials carries a mean that is not a score, so none is published"),
    },
)

JOB_CONFIG = Input(
    name="harbor_job_config", via=job_config, reads=[At(JOB, "config.json")],
    answers={
        "run.agent_harness": Card.ANY(because="agents[0].name is optional in the job config"),
        "run.agent_model_ref": Card.ANY(because="agents[0].model_name is optional in the job config"),
    },
)

TRAJECTORY_SOURCES = ("harbor_traces",) + tuple(
    f"harbor_trajectory/{d.name}" for d in ATIF_CANDIDATES)

TRIAL_SCORE_BECAUSE = (
    "`result.json` carries the reward the harness recorded, `verifier/reward.json` the same "
    "number, and `reward.txt` that number rendered to 4dp, so the lossy one ranks last. The "
    "two decoder names cannot both appear on one record and are ranked only because the "
    "order must be total"
)

TRIAL_ERROR_BECAUSE = (
    "`exception_info` in `result.json` is the harness's typed record and `exception.txt` a "
    "rendered traceback written only on failure; where both exist they describe one "
    "failure, so the typed one wins and the text is kept"
)



# ---------------------------------------------------------------------------- attach

TRIAL_ATTACH = (
    Relate(
        frm="rollout.subagents", to="rollout.events", on=Join(space="subagent"),
        cardinality=Card.AT_LEAST_ONE,
        to_card=Card.ONE,
        because=(
            "a sub-agent that a `.meta.json` declares must have the transcript file whose "
            "stem it is named after, so the edge catches a delegation archived without its "
            "conversation"
        ),
    ),
    Contribute(
        to="rollout.events", input="harbor_trajectory", order_by="started_at", identity="source_event_id",
        because="ATIF numbers its own steps and stamps its own timestamps",
    ),
    Contribute(
        to="rollout.events", input="harbor_mini_swe", order_by="read_seq", identity="source_event_id",
        because=("mini-swe-agent writes no clock into a message, so the file's own order is "
                 "the only order and `read_seq` carries it, zero-padded; the identity is a "
                 "separate key because an index that is also an identity sorts 0, 1, 10, 2"),
    ),
    Contribute(
        to="rollout.events", input="harbor_session", order_by="started_at", identity="source_event_id",
        because="the agent's session log carries its own uuid per line and its own clock",
    ),
    Contribute(
        to="rollout.events", input="harbor_subagent", order_by="started_at", identity="source_event_id",
        because="a sub-agent writes its own file with its own uuids and its own clock",
    ),
    Contribute(
        to="rollout.events", input="harbor_tool_results", order_by="source_event_id", identity="source_event_id",
        because=(
            "a spilled tool result has no clock at all — its file name is the key the "
            "transcript refers to it by, so it is both its identity and, for want of "
            "anything better, its order"
        ),
    ),
    Contribute(
        to="rollout.events", input="harbor_agent_result", order_by="read_seq", identity="source_event_id",
        because=(
            "a LangGraph message list has no clock, so its list position is the order and "
            "`read_seq` states it, zero-padded. The identity cannot order: it is a uuid where "
            "the producer wrote one and a bare index where it did not"
        ),
    ),
    Contribute(
        to="rollout.events", input="harbor_traces", order_by="read_seq", identity="source_event_id",
        because=(
            "an OpenInference conversation is the last LLM span's message list, and only its "
            "assistant turns carry a span's clock, so list position is the order and "
            "`read_seq` states it. A trace with no LLM span is one row per OTLP batch, "
            "identified by its first span id and ordered by that span's start time, "
            "zero-padded into the same column"
        ),
    ),
    Contribute(
        to="rollout.artifacts", input="harbor_criteria", order_by="name", identity="name",
        because="one file per criterion, named after its directory",
    ),
    Contribute(
        to="rollout.artifacts", input="harbor_verifier_result", order_by="name", identity="name",
        because="the structured verifier_result is one explicitly selected report per trial",
    ),
    Contribute(
        to="rollout.artifacts", input="harbor_verifier", order_by="name", identity="name",
        because="one CTRF report per trial",
    ),
    Contribute(
        to="rollout.artifacts", input="harbor_verifier_reports", order_by="name", identity="name",
        because="one report file per name and trial",
    ),
    Contribute(
        to="rollout.artifacts", input="harbor_verifier_logs", order_by="path", identity="sha256",
        because="raw verifier logs are content-addressed retained outcome evidence",
    ),
    Contribute(
        to="rollout.artifacts", input="harbor_tool_results", order_by="path", identity="sha256",
        because="large tool-result side-files are content-addressed trace evidence",
    ),
    Contribute(
        to="rollout.artifacts", input="harbor_artifacts", order_by="path", identity="sha256",
        because=(
            "identity is the content digest, not the path, because one run can hold the "
            "same bytes at two paths"
        ),
    ),
    Contribute(
        to="rollout.artifacts", input="harbor_agent_store", order_by="path", identity="path",
        because=(
            "identity is the path here and not the digest: .db, .db-shm and .db-wal are "
            "three different files of one store and none of them is a duplicate of another"
        ),
    ),
    Contribute(
        to="rollout.subagents", input="harbor_subagent_meta", order_by="agent", identity="agent",
        because="a sub-agent is named by its file stem, which is also the key its transcript is filed under",
    ),
)


# --------------------------------------------------------------------------- ignores

TRIAL_IGNORES = (
    Ignore(TRIAL, "verifier/original-repo-ctrf.json",
           because="the verifier's run against the repo before the agent touched it: a "
                   "baseline no question asks for; `verifier/ctrf.json` beside it is the "
                   "scored outcome"),
    Ignore(TRIAL, "trial.log",
           because="the harness's own stdout; every fact in it is also in `result.json`"),
    Ignore(TRIAL, "config.json",
           because="the trial's input configuration, the run's plan rather than its output; the parts this reader stores come from `result.json`, which repeats them"),
    Ignore(TRIAL, "lock.json",
           because="the scheduler's lease on this trial directory, a fact about the orchestrator rather than the rollout"),
    Ignore(TRIAL, "verifier/criteria/*/reward-details.json",
           because="the judge's per-criterion prose; the number beside it is read by `harbor_criteria` and the prose has no column"),
    Ignore(TRIAL, "verifier/criteria/*/status.json",
           because="restates the criterion's value and adds a returncode; the value is already read from `reward.json` in the same directory"),
    Ignore(TRIAL, "verifier/criteria/*/stdout",
           because="one criterion's raw stdout, a few bytes long"),
    Ignore(TRIAL, "verifier/criteria/*/stderr",
           because="one criterion's raw stderr, the same boilerplate across criteria"),
    Ignore(TRIAL, "verifier/criteria/*/exit_code",
           because="an exit code per criterion; `status.json` already carries the returncode and neither has a column"),
    Ignore(TRIAL, "agent/*.txt",
           because="the agent's own console transcript and the shell command line the harness built to launch it, inputs and renderings; the structured transcript is `agent/trajectory.json`, which `harbor_trajectory` reads"),
    Ignore(TRIAL, "agent/*.log",
           because="the agent wrapper's log: harness chatter, not the model's turns"),
    Ignore(TRIAL, "agent/decision.report.md",
           because="the v46 agent's prose rendering of `agent/decision.json`, which `harbor_decision` reads"),
    Ignore(TRIAL, "agent/summary.json",
           because="the LangGraph agent's own summary of `agent/result.json`, which sits beside it"),
    Ignore(TRIAL, "agent/langgraph_project/**",
           because="the agent's own source code, shipped into the run directory: the program, not its output"),
    Ignore(TRIAL, "agent/langgraph_runner.py",
           because="the entrypoint of that same agent program, an input to the run"),
    Ignore(TRIAL, "agent/sessions/.claude.json",
           because="the agent's global settings file, captured with the session directory"),
    Ignore(TRIAL, "agent/sessions/backups/.claude.json.backup.*",
           because="numbered backups of that same settings file"),
    Ignore(TRIAL, "agent/sessions/shell-snapshots/snapshot-bash-*.sh",
           because="a snapshot of the container's bash environment taken at agent start"),
    Ignore(TRIAL, "agent/sessions/sessions/*.json",
           because="the agent's session index, which names the transcript files harbor_session already reads"),
    Ignore(TRIAL, "agent/opencode/xdg-data/opencode/opencode.db-journal",
           because="a rollback journal, present only while a write is in flight"),
)


VERIFIER_BECAUSE = (
    "no trial writes two of these three reports, so this order can never fire; it states "
    "which would be believed if one did: the criteria report, the only one carrying a "
    "rollup score as well as the per-test detail"
)


JOB_IGNORES = (
    Ignore(JOB, "job.log",
           because="harbor's own job log; in a single-trial job the trial's `trial.log` holds the same bytes, and concatenating the two would double it"),
    Ignore(JOB, "lock.json",
           because="the orchestrator's lease on the job directory"),
    Ignore(ROOT, ".intake-manifest.json", because=INTAKE_RECEIPT),
)


TRIAL_LANES = (
    Lane(kind="trace", order_by="started_at", identity="source_event_id",
         actor="harness",
         because="the harness's own ATIF record of the trial: one sequence, its own "
                 "clock, written by the runner rather than by the agent"),
    Lane(kind="session", order_by="started_at", identity="source_event_id",
         actor="main",
         because="the agent's own session log: a different author's account of the same "
                 "minutes as the trace, so the two are not one sequence"),
    Lane(kind="codex-ui", order_by="started_at", identity="source_event_id",
         actor="main",
         because="codex writes two accounts of every turn into one rollout log, the "
                 "responses-API items and the items its own UI completed; ordering them "
                 "together would interleave two renderings of one turn"),
    Lane(kind="bookkeeping", order_by="source_event_id", identity="source_event_id",
         actor="harness",
         because="what the harness did around the conversation rather than in it, such as "
                 "a queued prompt or a resume marker. It is ordered by file position because "
                 "some of these records carry no timestamp, and each row keeps its whole "
                 "record as JSON because no column holds its parameters"),
    Lane(kind="subagent", order_by="started_at", identity="source_event_id",
         because="one file per sub-agent, each its own turns; `actor` is left unset "
                 "because which agent it is, is the lane's instance, not a constant"),
)


# --------------------------------------------------------------------------- reader

_DECLARED = dict(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,  # a trial directory exists before the trial runs
    name="harbor",
    inputs=(("harbor", TRIAL_INPUTS + (JOB_RESULT, JOB_CONFIG)),),
    lanes=TRIAL_LANES,
    questions=(
        Q.RUN_N_TRIALS, Q.RUN_SOURCE_RUN_ID, Q.RUN_STARTED, Q.RUN_COST, Q.RUN_SCORE, Q.RUN_N_SCORED,
        Q.RUN_AGENT_HARNESS, Q.RUN_AGENT_MODEL_REF,
        Q.ROLLOUT_TASK, Q.ROLLOUT_SCORE, Q.ROLLOUT_ERROR, Q.ROLLOUT_COST,
        Q.ROLLOUT_STARTED, Q.ROLLOUT_FINISHED, Q.ROLLOUT_AGENT_MODEL, Q.ROLLOUT_AGENT_NAME,
        Q.ROLLOUT_VERIFIER, Q.ROLLOUT_STOP, Q.ROLLOUT_CRITERIA, Q.ROLLOUT_EVENTS,
        Q.ROLLOUT_ARTIFACTS, Q.ROLLOUT_SUBAGENTS,
        Q.ROLLOUT_N_PROMPT_TOKENS, Q.ROLLOUT_N_COMPLETION_TOKENS,
    ),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=tuple(p.name for p in TRIAL_INPUTS),
            because="every rollout-grain input reads a file inside one trial directory, and the directory is the enumeration",
        ),
        Rows(
            question="rollout.score",
            may_create=("harbor_reward", "harbor_reward_json", "harbor_reward_txt"),
            because="three spellings of one per-trial reward file, each inside a trial directory",
        ),
        Rows(
            question="rollout.error",
            may_create=("harbor_exception", "harbor_trial"),
            because="both read files inside one trial directory",
        ),
    ),
    orders=(
        Order(question="rollout.n_prompt_tokens", order=TRAJECTORY_SOURCES[::-1],
              keep_loser=True, because=TRAJECTORY_BECAUSE),
        Order(question="rollout.n_completion_tokens", order=TRAJECTORY_SOURCES[::-1],
              keep_loser=True, because=TRAJECTORY_BECAUSE),
        Order(question="rollout.score",
              order=("harbor_reward/rewards-composite", "harbor_reward/rewards-reward",
                     "harbor_reward_json", "harbor_reward_txt"),
              keep_loser=True, because=TRIAL_SCORE_BECAUSE),
        Order(question="rollout.error", order=("harbor_trial", "harbor_exception"),
              keep_loser=True, because=TRIAL_ERROR_BECAUSE),
        Order(question="rollout.verifier",
              order=("harbor_verifier/criteria-report", "harbor_verifier/ctrf",
                     "harbor_verifier/runscript-output"),
              keep_loser=True, because=VERIFIER_BECAUSE),
    ),
    folds=(
        Fold(
            into="run.n_scored", over="rollout.score", op=Op.COUNT,
            expect=Card.ANY(because="a trial that threw before its verifier ran has no score, and the shortfall is the fact"),
            because="how many rollouts this read actually scored, which no file in the job states",
        ),
    ),
    ignored=TRIAL_IGNORES + JOB_IGNORES,
    attach=TRIAL_ATTACH,
)

READER = Reader(landmarks=(JOB, TRIAL), **_DECLARED)

#: Harbor as a run of its own, as dispatch offers it: also needs the job's own config, so a folder
#: that only holds a `job.log` or another producer's `config.json` is not claimed. `READER` is the
#: one a host mounts, where the host has already said the folder is a harbor job.
STANDALONE = Reader(landmarks=(JOB, TRIAL, OWN_CONFIG), **_DECLARED)
