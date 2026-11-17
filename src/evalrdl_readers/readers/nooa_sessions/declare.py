# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA's session stores read on their own: a folder of sessions is a run, a root session with its
children a rollout."""

from __future__ import annotations

from evalrdl_readers.core import (
    At,
    Card,
    Contribute,
    Grain,
    Ignore,
    Input,
    Lane,
    Merge,
    Population,
    Question,
    Reader,
    Rows,
)

from .. import questions as Q
from ..store_schema import READER_SCHEMA
from .landmarks import SESSIONS
from .translate import nooa_session_lanes, nooa_sessions

STORE = "????????-????-????-????-????????????.db"

BOOKKEEPING_LANE = Lane(
    kind="bookkeeping",
    order_by="read_seq",
    identity="source_event_id",
    actor="harness",
    because=(
        "what NOOA recorded around the conversation: the session's start, title and settings, "
        "each turn's admitted items, start and end with its usage, a call's receipt, a child's "
        "attributed usage and close, and the agent's saved state"
    ),
)
SUBAGENT_LANE = Lane(
    kind="subagent",
    order_by="read_seq",
    identity="source_event_id",
    because=(
        "a child session is a store of its own naming its parent session, its events in the "
        "order it stored them; nothing in either store names the call that started it"
    ),
)
CONTEXT_LANE = Lane(
    kind="context",
    order_by="read_seq",
    identity="source_event_id",
    actor="harness",
    because=(
        "what NOOA rendered into the prompt beside the conversation: each turn's task text, the "
        "cells it ran before the model's turn and what they printed, and the dynamic context a "
        "response was generated against"
    ),
)

NOOA_SESSION = Input(
    name="nooa_session",
    via=nooa_sessions,
    reads=(At(SESSIONS, STORE),),
    answers={
        "rollout.events": Card.ANY(
            because="a session opened and closed at once stores only its start"
        ),
        "rollout.started_at": Card.ANY(because="a store without a session start is not a rollout"),
        "rollout.finished_at": Card.ANY(because="a store without a session start is not a rollout"),
        "run.started_at": Card.ANY(because="a folder may hold no readable session"),
        "run.model": Card.ANY(because="a session that never reached the model names none"),
    },
)

IGNORED = (
    Ignore(SESSIONS, "*.lock", because="a live session's claim, empty"),
    Ignore(SESSIONS, f"{STORE}-shm", because="SQLite's shared-memory index of the -wal"),
    Ignore(SESSIONS, f"{STORE}-journal", because="SQLite's rollback journal of a write in flight"),
    Ignore(
        SESSIONS,
        "*-memory.db*",
        because="nooa-memory's long-term store beside a session, not the session",
    ),
)
EVENTS_IN_ORDER = (
    "NOOA numbers every event it stores in insertion order and names it by a uuid; its clock is "
    "the writer's local time, and a response is stamped when it was built, so the stored order "
    "is the order"
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,
    name="nooa-sessions",
    landmarks=(SESSIONS,),
    inputs=(("nooa-sessions", (NOOA_SESSION,)),),
    questions=(
        Q.ROLLOUT_EVENTS,
        Q.ROLLOUT_STARTED,
        Q.ROLLOUT_FINISHED,
        Q.RUN_STARTED,
        Q.RUN_MODEL,
    ),
    lanes=(BOOKKEEPING_LANE, SUBAGENT_LANE, CONTEXT_LANE),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=("nooa_session",),
            because="each root session is one rollout, named by its session id",
        ),
    ),
    ignored=IGNORED,
    attach=(
        Contribute(
            to="rollout.events",
            input="nooa_session",
            order_by="read_seq",
            identity="source_event_id",
            because=EVENTS_IN_ORDER,
        ),
    ),
)

# ------------------------------------------------------------- mounted where a host runs NOOA

SESSION_LANE = Lane(
    kind="session",
    order_by="read_seq",
    identity="source_event_id",
    actor="main",
    because=(
        "NOOA's own account of the root session's turns, beside the host's: a different "
        "author's account of the same minutes, in the order NOOA stored it"
    ),
)

ROLLOUT_NOOA_LOG = Question(
    name="rollout.nooa_log",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.nooa_log",
    means="the root session id NOOA's store names itself by: the name a host joins to its rollout",
)

NOOA_SESSION_LANES = Input(
    name="nooa_session_lanes",
    via=nooa_session_lanes,
    reads=(At(SESSIONS, STORE),),
    answers={
        "rollout.events": Card.ANY(
            because="a session opened and closed at once stores only its start"
        ),
        "rollout.nooa_log": Card.ANY(because="a store without a session start is not a session"),
    },
)

#: NOOA's stores read inside a host that keeps its own account of the turns: the same rows as
#: `READER`, with the root session as an alternate account and the run left to the host.
MOUNTED = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,
    name="nooa",
    landmarks=(SESSIONS,),
    inputs=(("nooa", (NOOA_SESSION_LANES,)),),
    questions=(Q.ROLLOUT_EVENTS, ROLLOUT_NOOA_LOG),
    lanes=(SESSION_LANE, BOOKKEEPING_LANE, SUBAGENT_LANE, CONTEXT_LANE),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=("nooa_session_lanes",),
            because=(
                "read alone, a store names its root session and nothing else; a host that "
                "enumerates its rollouts declares its own"
            ),
        ),
    ),
    ignored=IGNORED,
    attach=(
        Contribute(
            to="rollout.events",
            input="nooa_session_lanes",
            order_by="read_seq",
            identity="source_event_id",
            because=EVENTS_IN_ORDER,
        ),
    ),
)
