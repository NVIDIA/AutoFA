# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""codex's logs read on their own; a host that runs codex mounts this where codex's logs sit."""

from __future__ import annotations

from evalrdl_readers.core import (
    At,
    Card,
    Contribute,
    Decoders,
    Grain,
    Input,
    Merge,
    Population,
    Question,
    Reader,
    Rows,
)

from .. import questions as Q
from ..dialects.sessions import SESSION_CANDIDATES
from ..store_schema import READER_SCHEMA
from .landmarks import LOGS
from .translate import codex_log, codex_session, codex_stdout

ROLLOUT_CODEX_LOG = Question(
    name="rollout.codex_log",
    grain=Grain.ROLLOUT,
    merge=Merge.ONE,
    lands="rollouts.codex_log",
    means="the session id codex's rollout file names itself by: the name a host joins to its trial",
)

ROLLOUT_FILES = "sessions/*/*/*/rollout-*.jsonl"

CODEX_LOG = Input(
    name="codex_log",
    via=codex_log,
    reads=[At(LOGS, ROLLOUT_FILES)],
    answers={
        "rollout.codex_log": Card.ANY(
            because="a failed run can leave its error files and no rollout file"
        )
    },
)
CODEX_SESSION = Input(
    name="codex_session",
    via=codex_session,
    reads=[At(LOGS, ROLLOUT_FILES)],
    decoders=Decoders(
        candidates=SESSION_CANDIDATES,
        because="codex's rollout records name their own shape in `type`, picked per record",
    ),
    answers={
        "rollout.events": Card.ANY(
            because="a failed run can leave its error files and no rollout file"
        )
    },
)
CODEX_STDOUT = Input(
    name="codex_stdout",
    via=codex_stdout,
    reads=[At(LOGS, "codex.txt")],
    answers={
        "rollout.events": Card.ANY(
            because=(
                "items, turns and the thread restate the rollout file; a run with no error and "
                "no stray text leaves nothing else"
            )
        )
    },
)

READER = Reader(
    schema=READER_SCHEMA,
    counts=Population.ATTEMPTED,
    name="codex",
    landmarks=(LOGS,),
    inputs=(("codex", (CODEX_LOG, CODEX_SESSION, CODEX_STDOUT)),),
    questions=(Q.ROLLOUT_EVENTS, ROLLOUT_CODEX_LOG),
    lanes=(Q.AGENT_SESSION_LANE, Q.CODEX_UI_LANE, Q.AGENT_BOOKKEEPING_LANE),
    rows=(
        Rows(
            question=Grain.ROLLOUT,
            may_create=("codex_log", "codex_session", "codex_stdout"),
            because=(
                "read alone, codex's logs name their session and nothing else; a host that "
                "enumerates its trials declares its own"
            ),
        ),
    ),
    attach=(
        Contribute(
            to="rollout.events",
            input="codex_session",
            order_by="started_at",
            identity="source_event_id",
            because="codex stamps every rollout record; an item without an id is named by its line",
        ),
        Contribute(
            to="rollout.events",
            input="codex_stdout",
            order_by="source_event_id",
            identity="source_event_id",
            because="codex's stdout carries no clock, so the line is the only order and the identity",
        ),
    ),
)
