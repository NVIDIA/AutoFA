# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""inspect_ai's logs read on their own: the questions they answer and how a run folds them."""

from __future__ import annotations

from evalrdl_readers.core import (
    At,
    Card,
    Contribute,
    Fold,
    Grain,
    Input,
    Merge,
    Op,
    Population,
    Question,
    Reader,
    Rows,
)

from .. import questions as Q
from ..store_schema import READER_SCHEMA
from .landmarks import LOGS
from .translate import in_logs

RUN_N_TASKS = Question(
    name="run.n_tasks_declared",
    grain=Grain.RUN,
    merge=Merge.ONE,
    lands="runs.n_tasks",
    means="how many task log files this run directory holds",
)

IN_LOGS = Input(
    name="in_logs",
    via=in_logs,
    reads=(At(LOGS, "*.json"),),
    answers={
        "run.model": Card.ONE(because="one model per inspect run"),
        "run.n_tasks_declared": Card.ONE(because="counting the log files is the enumeration"),
        "rollout.score": Card.AT_LEAST_ONE(because="every sample in this fixture carries a score"),
        "rollout.task": Card.AT_LEAST_ONE(
            because=(
                "`<eval.task>:<sample.id>` names one sample; `eval.task` alone is shared by "
                "every sample in the log"
            )
        ),
        "rollout.repeat": Card.AT_LEAST_ONE(
            because="`epoch` is inspect's repeat index and is on every sample"
        ),
        "rollout.prompt": Card.AT_LEAST_ONE(
            because="`sample.input` is the prompt, on every sample"
        ),
        "rollout.n_prompt_tokens": Card.ANY(
            because=(
                "`output.usage.input_tokens`; inspect omits usage for a sample the model never "
                "answered"
            )
        ),
        "rollout.n_completion_tokens": Card.ANY(
            because="`output.usage.output_tokens`, omitted with the rest of usage"
        ),
        "rollout.started_at": Card.ANY(
            because=(
                "`eval.created` is when the log was written, the only clock an inspect sample "
                "has, and a log may omit it"
            )
        ),
        "rollout.events": Card.AT_LEAST_ONE(
            because="`sample.messages` is the conversation, on every sample"
        ),
    },
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.COMPLETED,  # a sample appears in the log once it has been scored
    name="inspect",
    landmarks=(LOGS,),
    inputs=(  # outermost producer first
        ("inspect", (IN_LOGS,)),
    ),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=("in_logs",),
            because=(
                "a sample exists because it is in a log file, and `in_logs` is the only "
                "input that opens one"
            ),
        ),
    ),
    questions=(
        Q.RUN_MODEL,
        Q.RUN_SCORE,
        Q.RUN_N_SCORED,
        RUN_N_TASKS,
        Q.ROLLOUT_SCORE,
        Q.ROLLOUT_TASK,
        Q.ROLLOUT_REPEAT,
        Q.ROLLOUT_PROMPT,
        Q.ROLLOUT_EVENTS,
        Q.ROLLOUT_N_PROMPT_TOKENS,
        Q.ROLLOUT_N_COMPLETION_TOKENS,
        Q.ROLLOUT_STARTED,
    ),
    attach=(
        Contribute(
            to="rollout.events",
            input="in_logs",
            order_by="read_seq",
            identity="source_event_id",
            because=(
                "`sample.messages` is an ordered list with no clock on its turns, so list "
                "position, carried by `read_seq`, is the only order"
            ),
        ),
    ),
    folds=(
        Fold(
            into="run.score",
            over="rollout.score",
            op=Op.MEAN,
            expect=Card.ANY(because="every sample here is scored, so the mean covers all of them"),
            because=(
                "the fixture publishes no run-level result block at all, unlike real inspect_ai"
            ),
        ),
        Fold(
            into="run.n_scored",
            over="rollout.score",
            op=Op.COUNT,
            expect=Card.ANY(because="same population, same completeness"),
            because="the enumerated count of scored samples",
        ),
    ),
)
