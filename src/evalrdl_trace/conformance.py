# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every `TraceSource` must hold to, checked over one rollout it has.

`check(source, key)` asserts that the source's answers agree with each other: lanes count the
events, a window or a lane is a slice of the whole, every tool call sits on an event, a point
read finds each event, a part's whole text is the text the events carry, and the trace map and timeline count what `events` returns.
"""

from __future__ import annotations

from .assemble import timeline
from .source import RolloutKey, TraceSource

#: Every row of a trace table is ordered by these, in this order.
EVENT_ORDER = ("agent_ref", "lane", "event_idx")


def _order(row: dict) -> tuple:
    return tuple(row.get(k) for k in EVENT_ORDER)


def check(source: TraceSource, key: RolloutKey) -> None:
    rollout = source.rollout(key)
    assert rollout is not None, f"{key}: no rollout"
    assert (rollout["task_id"], rollout["repeat"]) == (key.task_id, key.repeat), rollout

    events = source.events(key, parts=False)
    assert [_order(e) for e in events] == sorted(_order(e) for e in events), "events unordered"

    lanes = source.lanes(key)
    assert sum(ln["n_events"] for ln in lanes) == len(events), (lanes, len(events))
    for ln in lanes:
        mine = [e for e in events if (e["agent_ref"], e["lane"]) == (ln["agent_ref"], ln["lane"])]
        assert len(mine) == ln["n_events"], ln
        assert mine[0]["event_idx"] == ln["first_event_idx"], ln
        assert mine[-1]["event_idx"] == ln["last_event_idx"], ln
        assert source.events(key, parts=False, lane=ln["lane"]) == [
            e for e in events if e["lane"] == ln["lane"]
        ], ln["lane"]

    assert source.events(key, parts=False, offset=1, limit=2) == events[1:3], "window"
    hydrated = source.events(key, parts=True)
    assert [_order(e) for e in hydrated] == [_order(e) for e in events], "parts change the rows"
    assert all(isinstance(e["parts"], list) for e in hydrated)

    on_events = {_order(e) for e in events}
    for call in source.tool_calls(key, args=False):
        assert _order(call) in on_events, f"tool call on no event: {_order(call)}"

    for ev in events[:20]:
        point = source.event_parts(key, ev["event_idx"], ev["lane"])
        assert point is not None, _order(ev)
        assert [_order(e) for e in point["events"]] == [_order(ev)], _order(ev)

    for ev in hydrated[:20]:
        for p in ev["parts"]:
            got = source.part_text(key, ev["event_idx"], p["part_idx"], ev["lane"])
            assert [r["text"] for r in got] == [p.get("text")], (_order(ev), p["part_idx"])
    some = next((e for e in hydrated if e["parts"]), None)
    if some is not None:
        idx = some["parts"][0]["part_idx"]
        assert source.part_text(key, some["event_idx"], idx, (some["lane"] or "") + "#absent") == []

    tmap = source.trace_map(key)
    assert tmap is not None and tmap["total_events"] == len(events), "trace map count"
    assert len(tmap["events"]) == min(len(events), source.event_cap)

    rows = source.timeline_rows(key)
    assert rows is not None and rows["total_events"] == len(events), "timeline count"
    assert timeline(rows, source.event_cap)["n_timed_events"] <= len(events)

    missing = RolloutKey(key.run_id, key.task_id + "#absent", key.repeat)
    assert source.rollout(missing) is None
    assert source.events(missing, parts=False) == [] and source.lanes(missing) == []
    assert source.trace_map(missing) is None and source.timeline_rows(missing) is None
