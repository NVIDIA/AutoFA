# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The routes, the timeline and the conformance checks, over a source held in memory."""

from __future__ import annotations

import sys

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from evalrdl_trace import ArtifactBytes, RolloutKey
from evalrdl_trace.assemble import timeline
from evalrdl_trace.conformance import check
from evalrdl_trace.routes import PART_TEXT_WINDOW, artifact_routes, routes

KEY = RolloutKey("r1", "t1", 0)
LONG = "x" * (PART_TEXT_WINDOW + 7)
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16


def _ev(lane, idx, kind, role, start, end=None, calls=0):
    return dict(
        task_id="t1", repeat=0, agent_ref="a", lane=lane, event_idx=idx, kind=kind, role=role,
        started_at=start, ended_at=end, n_tool_calls=calls, completion_tokens=None,
    )  # fmt: skip


class MemorySource:
    event_cap = 3

    def __init__(self, record_schema="atif-v1.6"):
        self.ro = dict(task_id="t1", repeat=0, record_schema=record_schema, status="ok",
                       started_at="10", finished_at="20")  # fmt: skip
        self.ev = [
            _ev("", 0, "message", "user", "10"),
            _ev("", 1, "message", "assistant", "11", calls=1),
            _ev("", 2, "message", "assistant", "14"),
            _ev("sub", 0, "message", "assistant", "12", "13"),
        ]
        self.calls = [dict(agent_ref="a", lane="", event_idx=1, call_idx=0, tool_name="sh",
                           status="ok", started_at="11", ended_at="12.5", result_event_idx=2)]  # fmt: skip
        # Part indices with a gap, as a producer may number them, and one text over a window.
        self.parts = [
            dict(agent_ref="a", lane="", event_idx=0, part_idx=0, part_type="text", text="hi"),
            dict(agent_ref="a", lane="", event_idx=0, part_idx=2, part_type="text", text=LONG),
            dict(agent_ref="a", lane="sub", event_idx=0, part_idx=0, part_type="text", text="s"),
        ]
        self.arts = {
            0: (dict(artifact_idx=0, name="shot.png", kind="image", storage="inline"), PNG),
            1: (dict(artifact_idx=1, name="page.html", kind="image", storage="inline"),
                b"<script>alert(1)</script>"),
            2: (dict(artifact_idx=2, name="gone", kind="file", storage="external", uri=None), None),
        }  # fmt: skip

    def _has(self, key):
        return key == KEY

    def rollout(self, key):
        return dict(self.ro) if self._has(key) else None

    def lanes(self, key):
        out = {}
        for e in self.events(key, parts=False):
            ln = out.setdefault((e["agent_ref"], e["lane"]), dict(agent_ref=e["agent_ref"],
                                lane=e["lane"], n_events=0, first_event_idx=e["event_idx"]))  # fmt: skip
            ln["n_events"] += 1
            ln["last_event_idx"] = e["event_idx"]
        return list(out.values())

    def events(self, key, *, parts, offset=None, limit=None, lane=None):
        if not self._has(key):
            return []
        rows = sorted(
            (dict(e) for e in self.ev if lane is None or e["lane"] == lane),
            key=lambda e: (e["agent_ref"], e["lane"], e["event_idx"]),
        )
        rows = rows[offset or 0 :][:limit] if limit is not None else rows[offset or 0 :]
        if parts:
            for e in rows:
                e["parts"] = [
                    dict(p) for p in self.parts
                    if (p["agent_ref"], p["lane"], p["event_idx"])
                    == (e["agent_ref"], e["lane"], e["event_idx"])
                ]  # fmt: skip
        return rows

    def part_text(self, key, event_idx, part_idx, lane=None):
        if not self._has(key):
            return []
        return [
            {"lane": p["lane"], "text": p["text"]} for p in self.parts
            if (p["event_idx"], p["part_idx"]) == (event_idx, part_idx)
            and (lane is None or p["lane"] == lane)
        ]  # fmt: skip

    def artifacts(self, key):
        return [dict(row) for row, _ in self.arts.values()] if self._has(key) else []

    def artifact_bytes(self, key, artifact_idx):
        row, data = self.arts.get(artifact_idx, (None, None))
        return ArtifactBytes(data, None) if self._has(key) and data is not None else None

    def event_parts(self, key, event_idx, lane=None):
        evs = [e for e in self.events(key, parts=True, lane=lane) if e["event_idx"] == event_idx]
        return {"events": evs, "parts": [], "tool_calls": []} if evs else None

    def tool_calls(self, key, *, args, lane=None):
        return (
            [dict(c) for c in self.calls if lane is None or c["lane"] == lane]
            if self._has(key)
            else []
        )

    def attempts(self, key):
        return []

    def provenance(self, key):
        return [{"question": "rollout.score", "n_values": 1}] if self._has(key) else []

    def edges(self, key):
        return (
            [{"resolved": True, "asserted_by": "reader"}, {"resolved": False}]
            if self._has(key)
            else []
        )

    def trace_map(self, key):
        evs = self.events(key, parts=False)
        return (
            {"total_events": len(evs), "events": evs[: self.event_cap]} if self._has(key) else None
        )

    def timeline_rows(self, key):
        if not self._has(key):
            return None
        return {"rollout": dict(self.ro), "total_events": len(self.ev),
                "events": self.events(key, parts=False), "tool_calls": [dict(c) for c in self.calls]}  # fmt: skip


def _client(source=None):
    def scope(request):
        if request.path_params["run_id"] != "r1":
            return JSONResponse({"error": "run not found"}, status_code=404)
        p = request.path_params
        return RolloutKey(p["run_id"], p["task_id"], int(p["repeat"]))

    source = source or MemorySource()
    return TestClient(Starlette(routes=[*artifact_routes(source, scope), *routes(source, scope)]))


def test_the_memory_source_conforms():
    check(MemorySource(), KEY)


def test_a_source_whose_lanes_miscount_does_not_conform():
    class Miscounting(MemorySource):
        def lanes(self, key):
            return [dict(ln, n_events=ln["n_events"] + 1) for ln in super().lanes(key)]

    with pytest.raises(AssertionError):
        check(Miscounting(), KEY)


def test_the_host_s_refusal_is_the_answer():
    assert _client().get("/v1/rollouts/nope/t1/0/events").status_code == 404


def test_an_unknown_include_or_lane_is_a_400_naming_what_there_is():
    c = _client()
    assert c.get("/v1/rollouts/r1/t1/0/events?include=x").json()["accepts"] == ["parts"]
    r = c.get("/v1/rollouts/r1/t1/0/tool_calls?lane=zz")
    assert r.status_code == 400 and r.json()["lanes"] == ["", "sub"]


def test_an_event_two_lanes_number_is_a_409_until_the_lane_is_named():
    c = _client()
    r = c.get("/v1/rollouts/r1/t1/0/events/0/parts")
    assert r.status_code == 409 and [x["lane"] for x in r.json()["lanes"]] == ["", "sub"]
    assert c.get("/v1/rollouts/r1/t1/0/events/0/parts?lane=sub").json()["lane"] == "sub"


def test_a_part_s_whole_text_is_read_window_by_window():
    c = _client()
    url = "/v1/rollouts/r1/t1/0/events/0/parts/2/text"
    first = c.get(url).json()
    assert (first["chars"], first["start"], first["end"], first["eof"]) == (
        len(LONG), 0, PART_TEXT_WINDOW, False)  # fmt: skip
    rest = c.get(url, params={"start": first["end"]}).json()
    assert rest["eof"] and first["text"] + rest["text"] == LONG
    assert c.get(url, params={"start": 3, "end": 5}).json()["text"] == "xx"


def test_a_part_text_read_names_its_part_by_index_not_position():
    c = _client()
    assert c.get("/v1/rollouts/r1/t1/0/events/0/parts/1/text?lane=").status_code == 404
    assert c.get("/v1/rollouts/r1/t1/0/events/0/parts/0/text?lane=").json()["text"] == "hi"


def test_a_part_text_read_refuses_what_it_cannot_answer():
    c = _client()
    r = c.get("/v1/rollouts/r1/t1/0/events/0/parts/0/text")
    assert r.status_code == 409 and r.json()["lanes"] == ["", "sub"]
    assert c.get("/v1/rollouts/r1/t1/0/events/0/parts/0/text?start=-1").status_code == 400
    assert c.get("/v1/rollouts/r1/t1/0/events/0/parts/0/text?start=4&end=2").status_code == 400
    assert c.get("/v1/rollouts/nope/t1/0/events/0/parts/0/text").status_code == 404


def test_artifacts_list_the_urls_they_can_be_fetched_at():
    rows = {r["artifact_idx"]: r for r in _client().get("/v1/rollouts/r1/t1/0/artifacts").json()}
    base = "/v1/rollouts/r1/t1/0/artifacts"
    assert rows[0]["view_url"] == f"{base}/0/view"
    assert rows[0]["download_url"] == f"{base}/0/download"
    assert rows[2]["view_url"] is None and rows[2]["download_url"] is None


def test_view_renders_only_bytes_that_are_an_image():
    c = _client()
    r = c.get("/v1/rollouts/r1/t1/0/artifacts/0/view")
    assert r.status_code == 200 and r.content == PNG
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert c.get("/v1/rollouts/r1/t1/0/artifacts/1/view").status_code == 415
    assert c.get("/v1/rollouts/r1/t1/0/artifacts/2/view").status_code == 404


def test_download_is_always_an_attachment():
    r = _client().get("/v1/rollouts/r1/t1/0/artifacts/1/download")
    assert r.status_code == 200 and r.content == b"<script>alert(1)</script>"
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["content-disposition"] == "attachment; filename*=UTF-8''page.html"


def test_edges_count_what_is_dangling():
    got = _client().get("/v1/rollouts/r1/t1/0/edges").json()
    assert (got["n"], got["resolved"], got["dangling"]) == (2, 1, 1)


def test_start_only_steps_last_until_the_next_one_in_their_own_lane():
    got = timeline(MemorySource().timeline_rows(KEY), 3)
    by = {(e["lane"], e["event_idx"]): e for e in got["events"]}
    assert by[("", 0)]["derivation"] == "to_next" and by[("", 0)]["duration_s"] == 1.0
    assert by[("", 2)]["derivation"] == "open"
    assert by[("sub", 0)]["derivation"] == "span"
    assert got["rollup"]["tool_s"] == 1.5 and got["span_basis"] == "rollout"


def test_without_atif_a_lone_stamp_is_a_point():
    got = timeline(MemorySource(record_schema="v11").timeline_rows(KEY), 3)
    assert {e["derivation"] for e in got["events"] if e["lane"] == ""} == {"point"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
