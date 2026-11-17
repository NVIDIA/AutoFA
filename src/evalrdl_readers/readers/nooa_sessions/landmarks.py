# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA's interactive sessions, as its session store keeps them in one directory.

    <session id>.db          one session: a SQLite store of the agent's events, in order
    <session id>.lock        the live claim on a session, empty
    <session id>-memory.db   nooa-memory's store beside a session, not a session

A child session (a sub-agent) is a store of its own in the same directory, naming its parent's
session id. A folder holding any number of sessions is one run; each root session is a rollout.
"""

from __future__ import annotations

from evalrdl_readers.core import Card, Landmark, Says

SESSIONS = Landmark(
    name="nooa_sessions",
    marker=Says(
        "????????-????-????-????-????????????.db",
        contains="active_tags",
        because=(
            "NOOA's SQLite store creates an `active_tags` table, which SQLite writes into the "
            "schema on the file's first page; a session store is named by its uuid"
        ),
    ),
    where=("",),
    expect=Card.ONE(
        because="NOOA keeps every session of one workspace, children included, in one directory"
    ),
)
