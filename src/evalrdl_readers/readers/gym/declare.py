# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from evalrdl_readers.core import (
    ROOT,
    Card,
    Compare,
    Contribute,
    Fold,
    Grain,
    Ignore,
    Join,
    Lane,
    Link,
    Merge,
    Op,
    Order,
    Population,
    Question,
    Reader,
    Relate,
    Rows,
)

from .. import questions as Q
from ..common import INTAKE_RECEIPT
from ..store_schema import READER_SCHEMA
from .agents.tau2 import GYM_TAU2
from .agents.trial_agent import GYM_TRIAL_AGENT
from .gdpval import (
    GDPVAL_DELIVERABLES,
    GDPVAL_FINISH,
    GDPVAL_HISTORY,
    GDPVAL_INPUTS,
    GDPVAL_METADATA,
    GDPVAL_REPEAT,
)
from .ignores import GYM_IGNORES
from .invocations import GYM_INVOCATIONS
from .landmarks import DELIVERABLES, RUN, SHARDS
from .record import GYM_COMBINED, GYM_FAILURES, GYM_OUTCOMES, GYM_ROLLOUTS, GYM_SHARDS
from .run_files import (
    GYM_AGGREGATE,
    GYM_DATASET,
    GYM_DATASET_NAME,
    GYM_MULTISTAGE,
    GYM_PLAN,
    GYM_PLANNED,
    GYM_SHARD_METRICS,
    GYM_SHARD_RUN,
    GYM_VERDICTS,
    GYM_VERIFICATION,
)
from .telemetry import GYM_TELEMETRY, RUN_TELEMETRY, TELEMETRY_ATTACH, TELEMETRY_LANES
from .trajectory import GYM_INLINE_CALLS, GYM_TRANSCRIPT

# ---------------------------------------------------------------------- questions

RUN_N_AGENTS = Question(
    name="run.n_agents", grain=Grain.RUN, merge=Merge.ONE, lands="runs.n_agents",
    means="how many agent groups the aggregate metrics file reports",
)
RUN_N_STAGES = Question(
    name="run.n_stages", grain=Grain.RUN, merge=Merge.ONE, lands="runs.n_stages",
    means="how many stages a multistage driver recorded — gdpval only",
)
RUN_N_PLANNED = Question(
    name="run.n_planned", grain=Grain.RUN, merge=Merge.ONE, lands="runs.n_planned",
    means="how many rollouts the run prepared: the lines of its materialized inputs",
)
RUN_N_DATASET = Question(
    name="run.n_dataset_rows", grain=Grain.RUN, merge=Merge.ONE, lands="runs.n_dataset_rows",
    means="how many task rows the preprocessed dataset holds — the denominator the run was given",
)
ROLLOUT_DELIV_DIR = Question(
    name="rollout.deliverables_dir", grain=Grain.ROLLOUT, merge=Merge.ONE,
    lands="rollouts.deliverables_dir",
    means="the gdpval repeat directory holding this rollout's files, `task_<uuid>/repeat_<n>`",
)


# ------------------------------------------------------------------------ reader

SAME_KEY = (
    "the merged stream is the one the harness writes and the shard and combined files are "
    "its records again; they meet where a record names its task, the copies are kept as "
    "losers, and a disagreement is CONTESTED"
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,  # finished rollouts plus the _failures sidecar
    name="nemo-gym",
    landmarks=(RUN, DELIVERABLES, SHARDS),
    inputs=(
        ("nemo-gym", (GYM_ROLLOUTS, GYM_OUTCOMES, GYM_TRANSCRIPT, GYM_TAU2, GYM_FAILURES, GYM_PLAN,
                      GYM_AGGREGATE, GYM_MULTISTAGE, GYM_DATASET, GYM_PLANNED, GYM_DATASET_NAME,
                      GYM_INLINE_CALLS, GYM_TRIAL_AGENT, GYM_SHARDS, GYM_COMBINED,
                      GYM_VERIFICATION, GYM_SHARD_RUN,
                      GYM_SHARD_METRICS, GYM_VERDICTS,
                      GDPVAL_REPEAT, GDPVAL_HISTORY, GDPVAL_METADATA,
                      GDPVAL_FINISH, GDPVAL_DELIVERABLES, GDPVAL_INPUTS, GYM_TELEMETRY,
                      GYM_INVOCATIONS)),
    ),
    lanes=TELEMETRY_LANES + (
        Lane(kind="verdict", order_by="read_seq", identity="source_event_id",
             actor="harness",
             because=(
                 "the quality pass's own account of a rollout it checked: its verdict, the "
                 "checks that failed and what it could not observe. Written after the "
                 "rollout, by the harness, about it -- not a turn of it"
             )),
        Lane(kind="attempt", order_by="read_seq", identity="source_event_id",
             actor="main",
             because=(
                 "a try the harness gave up on and ran again: the failures sidecar holds its "
                 "transcript and the stream the try that finished. One lane per try, numbered "
                 "as the sidecar orders them, so the finished transcript stays whole"
             )),
        Lane(kind="history", order_by="read_seq", identity="source_event_id",
             actor="main",
             because=(
                 "gdpval's history.json is the agent's own multi-turn transcript of the "
                 "rollout; the stream record carries the final exchange. Two accounts of one "
                 "rollout by different writers, so two sequences, as a harbor trial's trace "
                 "and session are"
             )),
        Lane(kind="subagent", order_by="read_seq", identity="source_event_id",
             because=(
                 "one lane per agent invocation `ng_trajectory` names a parent for: its own "
                 "conversation, which the parent's transcript does not hold. Its items carry no "
                 "clock, so their position is the order"
             )),
        Lane(kind="model_call", order_by="started_at", identity="source_event_id",
             because=(
                 "the model server's own account of the same rollout, written by a different "
                 "process on its own clock: the same minutes as the agent's transcript, not the "
                 "same sequence. The lane sets no `actor` because the author is the serving "
                 "layer; a call states whose it is where the record says so. A record whose "
                 "calls name more than one endpoint, or an interactive episode, puts each "
                 "endpoint in its own lane instance, since each keeps its own prompt history"
             )),
    ),
    questions=(
        RUN_TELEMETRY, Q.RUN_SOURCE_RUN_ID,
        Q.RUN_BENCHMARK, Q.RUN_EXPECTED, Q.RUN_SCORE, Q.RUN_N_SCORED,
        RUN_N_AGENTS, RUN_N_STAGES, RUN_N_DATASET, RUN_N_PLANNED,
        Q.ROLLOUT_SCORE, Q.ROLLOUT_TASK, Q.ROLLOUT_REPEAT, Q.ROLLOUT_STAGE,
        Q.ROLLOUT_ERROR, Q.ROLLOUT_PROMPT, Q.ROLLOUT_STOP, Q.ROLLOUT_EVENTS,
        Q.ROLLOUT_ARTIFACTS, Q.ROLLOUT_N_CALLS, ROLLOUT_DELIV_DIR,
        Q.ROLLOUT_AGENT_MODEL,
        Q.ROLLOUT_AGENT_NAME, Q.ROLLOUT_ATTEMPTS, Q.ROLLOUT_HEALTH, Q.ROLLOUT_TOOL_DEFS,
        Q.ROLLOUT_PROMPT_MESSAGES,
        Q.ROLLOUT_N_PROMPT_TOKENS, Q.ROLLOUT_N_COMPLETION_TOKENS, Q.ROLLOUT_N_TOKENS,
    ),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=(
                "gym_rollouts", "gym_transcript", "gym_failures", "gym_verdicts",
                "gym_trial_agent", "gdpval_history",
            ),
            because=(
                "the reader counts attempted rollouts, so the enumerations are what the harness "
                "wrote after trying: the stream, its failures sidecar and a gdpval repeat's "
                "history. The shard and combined files repeat the stream's records and `gym_plan` "
                "is the prepared set, so neither creates a row"
            ),
        ),
        Rows(
            question="rollout.task",
            may_create=("gym_rollouts", "gym_failures", "gdpval_history"),
            because="the stream enumerates; the failures sidecar names the tries the stream never finished",
        ),
        Rows(
            question="rollout.error",
            may_create=("gym_failures", "gym_rollouts"),
            because="a failure is a rollout, and each reads one per-rollout record; the shard and combined files only describe the stream's rollouts",
        ),
        Rows(
            question="rollout.stop_reason",
            may_create=("gdpval_finish", "gym_rollouts"),
            because="each reads one record or file per rollout; the shard and combined files only describe the stream's rollouts",
        ),
        Rows(
            question="rollout.agent_model",
            may_create=("gym_rollouts",),
            because="the shard and combined files hold the stream's records and only describe its rollouts",
        ),
        Rows(
            question="rollout.score",
            may_create=("gym_rollouts", "gym_failures"),
            because=(
                "the stream is the enumeration and the failures sidecar the second: it can name "
                "rollouts the stream does not hold, and its reward is their only score"
            ),
        ),
    ),
    orders=(
        Order(
            question="rollout.score",
            order=("gym_rollouts", "gym_combined", "gym_shards", "gym_failures"),
            keep_loser=True,
            because=SAME_KEY + "; the failures sidecar answers only rollouts the stream does not hold",
        ),
        Order(
            question="rollout.error",
            order=("gym_failures", "gym_rollouts", "gym_shards", "gym_combined"),
            keep_loser=True,
            because=(
                "the failures sidecar is the harness's typed record; the stream's `error_kind` "
                "is one untyped word, read so that a run whose sidecar is empty still says why "
                "its rollouts are blank"
            ),
        ),
        Order(question="rollout.task",
              order=("gym_rollouts", "gym_failures", "gdpval_history"),
              keep_loser=True,
              because=("the failures sidecar answers only keys the stream has no record for; a "
                       "gdpval directory the stream names states the same uuid the stream does")),
        Order(question="rollout.repeat",
              order=("gym_rollouts", "gym_combined", "gym_shards", "gym_failures",
                     "gdpval_history", "gym_verdicts"),
              keep_loser=True, because=SAME_KEY),
        Order(question="rollout.stage",
              order=("gym_rollouts", "gym_combined", "gym_shards"), keep_loser=True,
              because=SAME_KEY),
        Order(question="rollout.agent_name",
              order=("gym_rollouts", "gym_combined", "gym_shards"), keep_loser=True,
              because=SAME_KEY),
        Order(question="rollout.agent_model",
              order=("gym_rollouts", "gym_combined", "gym_shards"), keep_loser=True,
              because=SAME_KEY),
        Order(
            question="rollout.stop_reason",
            order=("gdpval_finish", "gym_rollouts", "gym_combined", "gym_shards"),
            keep_loser=True,
            because=(
                "the agent's own finish reason is its statement of why it stopped, which is what "
                "the column means; the stream's `agent_error_kind` is a class, not the agent's words"
            ),
        ),
        Order(
            question="run.score",
            order=("gym_aggregate", "gym_shard_metrics"),
            keep_loser=True,
            because="the harness's aggregate is over the merged stream; the shard runner's metrics are over its own shards",
        ),
        Order(
            question="run.n_expected",
            order=("gym_verification", "gym_shard_run"),
            keep_loser=True,
            because=(
                "two statements of the count of rollouts the run attempted, each silent where "
                "null: the certification pass's `expected_rollouts`, then the sharded runner's "
                "`tasks.rollout_count`"
            ),
        ),
    ),
    folds=(
        Fold(
            into="run.n_scored", over="rollout.score", op=Op.COUNT,
            expect=Card.ANY(
                because=(
                    "a gdpval deliverables tree can mint rollouts the stream has no score "
                    "for, so the count may cover fewer subjects than exist and the shortfall "
                    "is the fact worth having"
                ),
            ),
            because="the enumerated total, kept beside run.n_expected, which the Compare below holds it to",
        ),
    ),
    compare=(
        Compare(
            left="run.n_scored", right="run.n_expected", must="equal",
            because=(
                "counts=Population.ATTEMPTED, and a count the run states (`expected_rollouts`, "
                "`tasks.rollout_count`) is of the rollouts it attempted, so the two measure one "
                "number: fewer is a run cut short, more is a rollout counted twice"
            ),
        ),
    ),
    ignored=GYM_IGNORES + (Ignore(ROOT, ".intake-manifest.json", because=INTAKE_RECEIPT),),
    attach=TELEMETRY_ATTACH + (
        Relate(
            frm="rollout.deliverables_dir", to="rollout.task", on=Join(space="deliv_dir"),
            cardinality=Card.ONE,
            to_card=None,
            same_rollout=True,
            because=(
                "a deliverables directory is `task_<task_id>/repeat_<_ng_rollout_index>` of the "
                "stream record it belongs to. Not the record's `deliverables_dir` path: that is "
                "where the agent wrote, and a wrapper may move and renumber it. A directory no "
                "record names is a finding, not silence"
            ),
        ),
        Contribute(to="rollout.events", input="gym_transcript", order_by="read_seq", identity="source_event_id",
                   because="a responses-api record has no clock on its items, so input comes before output and the array position is the only order there is; `read_seq` states it"),
        Contribute(to="rollout.tool_defs", input="gym_transcript", order_by="def_idx", identity="def_idx",
                   because="the translator keeps one row per distinct definition JSON and numbers them in the order first offered, `responses_create_params` before the requests; that number is the only order and, per rollout, the identity"),
        Contribute(to="rollout.events", input="gym_tau2", order_by="read_seq", identity="source_event_id",
                   because="tau2's messages carry a timestamp each, but the system prompt beside them carries none, so the record's own order is the only one both have: the system prompt as the input side, then the messages as the record wrote them. The identity is the message's position, since only tool results carry an id and it is the call's"),
        Contribute(to="rollout.events", input="gym_trial_agent", order_by="read_seq", identity="source_event_id",
                   because="a trial's gym-agent.json is the same responses-api record the stream carries, so it is ordered and deduped the same way: the array position its items were written in, and the producer's own item id"),
        Contribute(to="rollout.events", input="gym_failures", order_by="read_seq", identity="source_event_id",
                   because="the same responses-api records, in the failures sidecar; same absent clock, same array position, same `read_seq`"),
        Contribute(to="rollout.events", input="gym_verdicts", order_by="read_seq", identity="source_event_id",
                   because="one event per verdict record, in a lane of its own; its only position is the one it has"),
        Contribute(to="rollout.events", input="gym_invocations", order_by="read_seq", identity="source_event_id",
                   because="a sub-agent's conversation is a responses-api item list with no clock, ordered and named as the transcript's are, its ids scoped by the invocation since they restart in each"),
        Link(word="spawned", to="rollout.events", frm_input="gym_transcript", to_input="gym_invocations",
             on=Join("agent_call"), cardinality=Card.ONE, to_card=None,
             because="an invocation names the tool call that started it, and the parent's tool call names it back; one call starts one agent. A sub-agent the integration could not link names no call, and one whose parent is itself a sub-agent is named from that sub-agent's lane"),
        Link(word="spawned", to="rollout.events", frm_input="gym_invocations", to_input="gym_invocations",
             on=Join("agent_call"), cardinality=Card.ONE, to_card=None,
             because="a sub-agent that starts another names it from its own conversation"),
        Contribute(to="rollout.events", input="gdpval_history", order_by="read_seq", identity="source_event_id",
                   because="`history.json` is an ordered message list whose only clock is the process's monotonic one, on some entries, so list position is the order and `read_seq` states it zero-padded. The identity is the same number unpadded and cannot state an order"),
        Contribute(
            to="rollout.attempts", input="gym_failures",
            order_by="attempt_idx", identity="attempt_idx",
            because=(
                "the sidecar's own order is the attempt order: the harness appends a record "
                "when a try ends badly and writes nothing else naming which try it was, so "
                "`attempt_idx` states the producer's order rather than a rank this reader invented"
            ),
        ),
        Contribute(to="rollout.artifacts", input="gdpval_deliverables", order_by="path", identity="sha256",
                   because="a deliverable is identified by its bytes: the same document is written as both .pdf and .docx and those are two files, while a byte-identical copy is one"),
        Contribute(to="rollout.artifacts", input="gdpval_inputs", order_by="path", identity="sha256",
                   because="reference files are stored under an md5 of their content, so the digest is already the producer's own identity for them"),
        Contribute(
            to="rollout.artifacts",
            input="gym_outcomes",
            order_by="name",
            identity="name",
            because=(
                "each explicitly allowlisted evaluator field has one stable semantic name per "
                "rollout; reference and golden fields are never read"
            ),
        ),
        Contribute(to="rollout.prompt_messages", input="gym_inline_calls", order_by="msg_hash", identity="msg_hash",
                   because="a message is named by the hash of its JSON, so the same message from any input or any request is one row; the hash is also the order, so a message's row number depends only on which messages the rollout sent"),
        Contribute(to="rollout.events", input="gym_inline_calls", order_by="request_idx", identity="source_event_id",
                   because="the inline copy carries `call_index`, the capture file's own append position and the order the producer wrote, so it is read rather than the clock re-derived"),
        Link(word="produced", to="rollout.events", frm_input="gym_inline_calls", to_input="gym_transcript",
             on=Join("item_id"), cardinality=Card.ANY, to_card=None,
             because="a call's response lists the output items it produced, by the producer's item id, or a chat response's tool calls by their call id: the capture sidecar keeps it, and `ng_trajectory.model_calls` does where an upload carries no sidecar. MEASURED over the corpus: 17,242 of 17,242 calls both keep list the same ids. A tau2 transcript holds no turn to name, and a response's tool call the transcript never kept (an aborted call, a nested agent's own log) matches nothing. A call of another actor (a judge) names items this transcript does not hold, and an item the harness wrote itself (a prefilled call) has no producing call"),
        Link(word="sent_in", to="rollout.events", frm_input="gym_transcript", to_input="gym_inline_calls",
             on=Join("model_call_id"), cardinality=Card.ONE, to_card=None,
             because="a turn lifted from a request names the one captured call whose request first carried it; a call is sent any number of such turns"),
        Link(word="sent_in", to="rollout.events", frm_input="gym_transcript", to_input="gym_inline_calls",
             on=Join("item_id"), cardinality=Card.ANY, to_card=Card.ANY,
             because="a call's request replays the exchange so far by the items' own ids, and a tool result by its call id; each item is first sent in one call, the last answer in none. A request may carry an item the exchange does not keep, which matches nothing"),
        Link(word="produced", to="rollout.events", frm_input="gym_inline_calls", to_input="gym_tau2",
             on=Join("response_id"), cardinality=Card.ANY, to_card=Card.ONE,
             because="a tau2 message the model wrote keeps its completion, whose id is one captured call's `response_id`. MEASURED over the corpus' tau2 records: 373 of 373 such messages name exactly one call, and 2 of 109 simulator calls produced no message tau2 kept"),
        Link(word="produced_by_text", to="rollout.events", frm_input="gym_inline_calls", to_input="gym_transcript",
             on=Join("text_key"), cardinality=Card.ANY, to_card=None, inferred=True,
             because="where no id is stated, a call whose response says a text equal to a turn of the exchange (text-v1) is linked to it as an inference: a user simulator's tool passes on the message the agent is next given, and a chat answer carries no item id. MEASURED on prod 3c1323ac71d43379: 219 of 229 simulator turns match one call, 10 match two; over 26 corpus Gym runs a text reached at most 7 rows below the bound of 8, and one model's `<think>Run.</think>`, written 88 times in a rollout, is past it and links nothing"),
        Link(word="sent_in_by_text", to="rollout.events", frm_input="gym_transcript", to_input="gym_inline_calls",
             on=Join("text_key"), cardinality=Card.ANY, to_card=None, inferred=True,
             because="where no id is stated, a turn whose text equals a message a claimed call's request first carries (text-v1) is linked to that call as an inference: a delivered user turn has no id to replay. MEASURED over 26 corpus Gym runs: most are a rollout's task message reaching the first call; tau2 and NOOA (whose requests wrap the task) gain none"),
    ),
)
