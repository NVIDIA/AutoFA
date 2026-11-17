# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each model call's prompt, read back by hash, and what each call did to the history before it."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from evalrdl_trace import RolloutKey
from evalrdl_trace.prompts import calls, message_text, usage_calls
from evalrdl_trace.routes import routes

KEY = RolloutKey("r1", "t1", 0)


def _msg(role, content):
    text = json.dumps({"role": role, "content": content})
    return hashlib.sha256(text.encode()).hexdigest()[:16], text


S, U, A1, T1, A2 = (_msg("system", "be terse"), _msg("user", "fix the bug"),
                    _msg("assistant", "looking"), _msg("tool", "ls: a b"), _msg("assistant", "done?"))  # fmt: skip
Q, H, A3, H2 = (_msg("user", "summarise the work so far"), _msg("user", "summary: fixed a"),
                _msg("assistant", "continuing"), _msg("user", "summary: fixed a (2 turns left)"))  # fmt: skip
# (messages, prompt tokens) per call, and what each call is
CALLS = [
    ([S, U], 100, "first"),
    ([S, U, A1, T1], 200, "append"),
    ([S, U, A1, T1], 200, "repeat"),
    ([Q], 20, "new_context"),
    ([S, U, A1, T1, A2], 220, "resume"),
    ([S, H], 40, "compaction"),
    ([S, H, A3], 60, "append"),
    ([S, H2, A3], 70, "rewrite"),
]


class PromptSource:
    event_cap = 100

    def __init__(self):
        self.ev = [dict(task_id="t1", repeat=0, agent_ref="a", lane="model_call", event_idx=i,
                        lane_kind="model_call", prompt_tokens=tok, source_event_id=f"c{i}")
                   for i, (_, tok, _) in enumerate(CALLS)]  # fmt: skip
        self.ev.append(dict(task_id="t1", repeat=0, agent_ref="a", lane="", event_idx=0,
                            lane_kind=None, prompt_tokens=None, source_event_id="m0"))  # fmt: skip
        self.req = [dict(agent_ref="a", lane="model_call", event_idx=i, request_dialect="chat",
                         request_messages=json.dumps([h for h, _ in msgs]),
                         request_tools=json.dumps(["t" * 64]) if i == 0 else None,
                         request_params=json.dumps({"model": "m", "temperature": 0}))
                    for i, (msgs, _, _) in enumerate(CALLS)]  # fmt: skip
        self.msgs = {h: t for msgs, _, _ in CALLS for h, t in msgs}

    def rollout(self, key):
        return {"task_id": "t1", "repeat": 0} if key == KEY else None

    def events(self, key, *, parts, offset=None, limit=None, lane=None):
        return [dict(e) for e in self.ev] if key == KEY else []

    def requests(self, key):
        return [dict(r) for r in self.req] if key == KEY else []

    def prompt_messages(self, key, hashes=None, *, text=False):
        out = []
        for h, t in self.msgs.items():
            if hashes is not None and h not in hashes:
                continue
            row = {"msg_hash": h, "chars": len(t), "role": json.loads(t)["role"], "item_type": None}
            if text:
                row["msg_json"] = t
            out.append(row)
        return out

    def tool_defs(self, key):
        return [{"sha256": "t" * 64, "name": "bash"}]


def _client():
    def scope(request):
        p = request.path_params
        return RolloutKey(p["run_id"], p["task_id"], int(p["repeat"]))

    return TestClient(Starlette(routes=routes(PromptSource(), scope)))


def test_each_call_is_named_by_what_it_did_to_the_history_before_it():
    src = PromptSource()
    events = [e for e in src.events(KEY, parts=False) if e["lane_kind"] == "model_call"]
    sizes = {m["msg_hash"]: m for m in src.prompt_messages(KEY)}
    rows = calls(events, src.requests(KEY), sizes)
    assert [r["class"] for r in rows] == [c for _, _, c in CALLS]
    repeat, resume, compaction = rows[2], rows[4], rows[5]
    assert (repeat["base_event_idx"], repeat["dropped"], repeat["added"]) == (1, 0, 0)
    assert (resume["base_event_idx"], resume["shared"], resume["added"]) == (2, 4, 1)
    assert (compaction["base_event_idx"], compaction["shared"], compaction["dropped"]) == (4, 1, 4)


def test_a_summary_in_place_of_turns_is_a_compaction_though_the_system_prompt_keeps_it_large():
    events = [dict(agent_ref="a", lane="model_call", event_idx=i, prompt_tokens=tok)
              for i, tok in enumerate((9000, 9500, 8000))]  # fmt: skip
    sent = [[S, U, A1, T1], [S, U, A1, T1, A2], [S, H]]
    requests = [dict(agent_ref="a", lane="model_call", event_idx=i, request_dialect="chat",
                     request_messages=json.dumps([h for h, _ in msgs]), request_tools=None,
                     request_params="{}")
                for i, msgs in enumerate(sent)]  # fmt: skip
    rows = calls(events, requests, {h: {"chars": len(t)} for msgs in sent for h, t in msgs})
    assert [r["class"] for r in rows] == ["first", "append", "compaction"]
    assert (rows[2]["dropped"], rows[2]["added"]) == (4, 1)


def test_the_summary_counts_calls_and_ships_no_text():
    got = _client().get("/v1/rollouts/r1/t1/0/prompts").json()
    assert (got["n_model_calls"], got["n_with_request"]) == (8, 8)
    assert got["classes"] == {"first": 1, "append": 2, "repeat": 1, "resume": 1, "new_context": 1,
                              "rewrite": 1, "compaction": 1}  # fmt: skip
    assert "fix the bug" not in json.dumps(got)


def test_a_compaction_shows_what_it_kept_sent_and_dropped():
    got = _client().get("/v1/rollouts/r1/t1/0/prompts/5?lane=model_call").json()
    assert [m["state"] for m in got["messages"]] == ["kept", "new"]
    assert got["messages"][1]["preview"] == "summary: fixed a"
    assert "preview" not in got["messages"][0]
    assert got["dropped"] == [U[0], A1[0], T1[0], A2[0]] and got["n_dropped"] == 4
    assert got["params"] == {"model": "m", "temperature": 0}


def test_a_message_sent_before_is_resent_not_new():
    got = _client().get("/v1/rollouts/r1/t1/0/prompts/4?lane=model_call").json()
    assert [m["state"] for m in got["messages"]] == ["kept"] * 4 + ["new"]
    first = _client().get("/v1/rollouts/r1/t1/0/prompts/0?lane=model_call").json()
    assert first["tools"] == [{"sha256": "t" * 64, "name": "bash"}]


def test_messages_are_read_by_hash_and_a_call_without_a_request_is_a_404():
    c = _client()
    got = c.get("/v1/rollouts/r1/t1/0/prompt_messages", params={"hash": H[0]}).json()
    assert [(m["text"], m["chars"]) for m in got["messages"]] == [("summary: fixed a", len(H[1]))]
    assert c.get("/v1/rollouts/r1/t1/0/prompt_messages").status_code == 400
    assert c.get("/v1/rollouts/r1/t1/0/prompts/99?lane=model_call").status_code == 404


def _usage_events():
    def ev(idx, tokens, rid, lane="", kind_raw=None, out=1):
        return dict(agent_ref="a", lane=lane, event_idx=idx, lane_kind=None, kind_raw=kind_raw,
                    prompt_tokens=tokens, completion_tokens=out, request_idx=rid,
                    source_event_id=f"e{idx}")  # fmt: skip

    return [ev(0, 100, 0), ev(1, 100, 0, out=9), ev(2, 150, 1), ev(3, None, None), ev(4, 140, 2),
            ev(5, None, None, kind_raw="compact_boundary"), ev(6, 130, 3), ev(7, 60, 4),
            ev(8, 500, 0, lane="subagent/x")]  # fmt: skip


def test_a_rollout_without_requests_is_sized_by_its_reported_input_tokens():
    rows = usage_calls(_usage_events())
    assert [(r["event_idx"], r["prompt_tokens"], r["class"]) for r in rows] == [
        (0, 100, "first"), (2, 150, "append"), (4, 140, "rewrite"), (6, 130, "compaction"),
        (7, 60, "compaction"), (8, 500, "first")]  # fmt: skip
    assert (rows[0]["completion_tokens"], rows[2]["base_event_idx"]) == (9, 2)
    assert all(r["has_request"] is False for r in rows)


def test_a_compaction_marker_with_usage_is_a_call_and_marks_the_next():
    evs = [dict(agent_ref="a", lane="", event_idx=i, kind_raw=k, prompt_tokens=t,
                completion_tokens=1, request_idx=i, source_event_id=f"e{i}")
           for i, (t, k) in enumerate([(100, None), (80, "compacted"), (120, None)])]  # fmt: skip
    assert [r["class"] for r in usage_calls(evs)] == ["first", "rewrite", "compaction"]


class UsageOnlySource(PromptSource):
    def __init__(self):
        super().__init__()
        self.ev = _usage_events()
        self.req = []

    def requests(self, key):
        return []


def test_the_summary_of_a_rollout_without_requests_comes_from_usage():
    def scope(request):
        p = request.path_params
        return RolloutKey(p["run_id"], p["task_id"], int(p["repeat"]))

    c = TestClient(Starlette(routes=routes(UsageOnlySource(), scope)))
    got = c.get("/v1/rollouts/r1/t1/0/prompts").json()
    assert (got["source"], got["n_model_calls"], got["n_with_request"]) == ("usage", 6, 0)
    assert got["classes"]["compaction"] == 2 and len(got["calls"]) == 6


def test_a_responses_item_reads_as_its_text_or_its_call():
    item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    assert message_text(json.dumps(item)) == "hi"
    call = {"type": "function_call", "name": "bash", "arguments": '{"cmd":"ls"}'}
    assert message_text(json.dumps(call)) == '→ bash({"cmd":"ls"})'


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
