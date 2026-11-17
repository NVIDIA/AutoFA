# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi's logs read on their own; a host that runs pi mounts this where pi's logs sit."""

from __future__ import annotations

from evalrdl_readers.core import (
    At,
    Card,
    Contribute,
    Grain,
    Ignore,
    Input,
    Merge,
    Population,
    Question,
    Reader,
    Rows,
)

from .. import questions as Q
from ..store_schema import READER_SCHEMA
from .landmarks import LOGS
from .translate import pi_session, pi_stdout

ROLLOUT_PI_LOG = Question(
    name="rollout.pi_log",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.pi_log",
    means="the session id pi's session file names itself by: the name a host joins to its trial",
)

PI_SESSION = Input(
    name="pi_session",
    via=pi_session,
    reads=[At(LOGS, "pi/sessions/*.jsonl")],
    answers={
        "rollout.pi_log": Card.ANY(
            because=(
                "pi opens every session file with its `session` record, and a file that does "
                "not is reported unreadable, so a count here would say it twice"
            )
        ),
        "rollout.events": Card.ANY(because="an empty session file holds no event"),
    },
)
PI_STDOUT = Input(
    name="pi_stdout",
    via=pi_stdout,
    reads=[At(LOGS, "pi.txt")],
    answers={
        "rollout.events": Card.ANY(
            because=(
                "most of pi's stdout restates its session file and is not landed; a short run "
                "compacts nothing, retries nothing and crashes nowhere"
            )
        )
    },
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,
    name="pi",
    landmarks=(LOGS,),
    inputs=(("pi", (PI_SESSION, PI_STDOUT)),),
    questions=(Q.ROLLOUT_EVENTS, ROLLOUT_PI_LOG),
    lanes=(Q.AGENT_SESSION_LANE, Q.AGENT_BOOKKEEPING_LANE),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=("pi_session", "pi_stdout"),
            because=(
                "read alone, pi's logs name their session and nothing else; a host that "
                "enumerates its trials declares its own"
            ),
        ),
    ),
    ignored=(
        Ignore(
            LOGS,
            "pi.full.txt",
            because=(
                "pi's stdout before the host cut it to `pi.txt`: the same records plus every "
                "token delta (`message_update`) and tool progress update. MEASURED on four "
                "trials: 2,533 records, 2,349 of them those two types, the other 184 exactly "
                "`pi.txt`'s"
            ),
        ),
    ),
    attach=(
        Contribute(
            to="rollout.events",
            input="pi_session",
            order_by="started_at",
            identity="source_event_id",
            because="pi gives every record of its session file an id and a clock",
        ),
        Contribute(
            to="rollout.events",
            input="pi_stdout",
            order_by="source_event_id",
            identity="source_event_id",
            because="pi's stdout records carry no clock, so the line is the only order and the identity",
        ),
    ),
)
