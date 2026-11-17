# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A nested harbor trial's `gym-agent.json`: the responses-api exchange of the gym agent."""

from __future__ import annotations

from evalrdl_readers.core import At, Card, Input, Sources, Unreadable

from ...common import records
from ..landmarks import RUN, STREAMS
from ..record import _responses_events

# ------------------------------------------------------------------- translators


def gym_trial_agent(src: Sources):
    """A responses-api exchange held by a nested trial; skipped if the stream has its `response.id`."""
    seen: set[str] = set()
    for doc in src.docs(0):
        for item in records(doc, "rollout.events"):
            if isinstance(item, Unreadable):
                continue
            _line, rec = item
            rid = ((rec.get("response") or {}) or {}).get("id")
            if rid:
                seen.add(rid)
    for doc in src.docs(1):
        rec = doc.json()
        if not isinstance(rec, dict):
            yield Unreadable(
                "rollout.events",
                where=doc.name,
                detail="a trial's gym-agent.json is not a JSON object",
            )
            continue
        rid = ((rec.get("response") or {}) or {}).get("id")
        if rid and rid in seen:
            continue
        sid = rec.get("session_id")
        if not sid:
            yield Unreadable(
                "rollout.events",
                where=doc.name,
                detail="a trial agent record with no session_id: the "
                "directory name is the only key it has and this "
                "file does not repeat it",
            )
            continue
        yield from _responses_events(rec, (str(sid),), 0)


# ------------------------------------------------------------------------- inputs

GYM_TRIAL_AGENT = Input(
    name="gym_trial_agent",
    via=gym_trial_agent,
    # One input: telling a duplicate trial from the only copy needs the stream too.
    reads=[At(RUN, STREAMS), At(RUN, "resources/*/gym-agent.json")],
    answers={
        "rollout.events": Card.ANY(
            because=(
                "a nested trial tree appears where gym hosted harbor. A `gym-agent.json` "
                "duplicating a stream response id is skipped; the rest are the only record of "
                "their items, because that export kept every request and lost every response"
            ),
        )
    },
)
