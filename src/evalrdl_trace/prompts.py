# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each model call's prompt, as the store keeps it, and where an agent rewrote its history.

A call's request lists its messages by hash (`event_texts.request_messages`), each message stored
once (`prompt_messages`). A call usually re-sends an earlier call's messages and appends a few.
Its *base* is the earlier call of the same lane sharing the longest leading run of messages with
it (the most recent such call on a tie), and how the two relate says what the agent did:

    first        the lane's first call with a request
    append       continues the previous call: all of it, then more
    repeat       sends the previous call's request again, unchanged: a retry
    resume       continues an earlier call that is not the previous one (the previous one was a
                 side call, as Terminus-2's handoff questions are)
    new_context  shares no leading message with any earlier call (a summariser's own prompt, a
                 fresh sub-context)
    rewrite      drops or edits part of its base's history, and its prompt does not shrink by
                 half (a per-call reminder edited into an early message, a trimmed tail)
    compaction   drops part of its base's history, and its prompt is at most half its base's or
                 holds fewer messages than its base (a summary in place of the turns it replaced,
                 beside a system prompt too large to halve)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

#: A rewrite whose prompt is at most this share of its base's is a compaction.
COMPACTION_SHARE = 0.5
#: A producer's own mark that it compacted the history: claude-code writes a boundary with no
#: usage; codex writes the summary, with the usage of the request that wrote it.
COMPACTION_MARKERS = frozenset({"compact_boundary", "compacted"})
#: Characters of a message's text a call's message list carries; the whole message is a point read.
PREVIEW_CHARS = 240
#: Hashes of dropped messages a call's detail lists.
DROPPED_LISTED = 200

CLASSES = ("first", "append", "repeat", "resume", "new_context", "rewrite", "compaction")


def _strings(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for v in value:
            _strings(v, out)
    elif isinstance(value, dict):
        for k in ("text", "content", "output", "summary", "arguments", "refusal"):
            if k in value:
                _strings(value[k], out)


def message_text(msg_json: str | None) -> str:
    """The readable text of one chat message or Responses input item; its JSON if it has none."""
    if not msg_json:
        return ""
    try:
        msg = json.loads(msg_json)
    except ValueError:
        return msg_json
    out: list[str] = []
    if isinstance(msg, dict):
        _strings(msg, out)
        for call in msg.get("tool_calls") or ():
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict):
                out.append(f"→ {fn.get('name')}({fn.get('arguments') or ''})")
        if msg.get("type") == "function_call":
            out = [f"→ {msg.get('name')}({msg.get('arguments') or ''})"]
    text = "\n".join(s for s in out if s)
    return text or msg_json


class _Trie:
    """Leading runs of message hashes, each node naming the latest call that passed through it."""

    def __init__(self) -> None:
        self.root: dict = {}

    def longest(self, names: list[str]) -> tuple[int, int | None]:
        node, depth, last = self.root, 0, None
        for n in names:
            nxt = node.get(n)
            if nxt is None:
                break
            node, depth, last = nxt, depth + 1, nxt[None]
        return depth, last

    def add(self, names: list[str], call: int) -> None:
        node = self.root
        for n in names:
            node = node.setdefault(n, {})
            node[None] = call


def _size(call: dict) -> float | None:
    return call["prompt_tokens"] if call["prompt_tokens"] is not None else call["chars"]


def calls(events: list[dict], requests: list[dict], sizes: dict[str, dict]) -> list[dict]:
    """One row per `model_call` event that carries a request, in lane order: its size, its base
    and what it did to the base's history. `events` are the rollout's model-call events,
    `requests` its `event_texts` rows, `sizes` each message's `chars` by hash."""
    req = {
        (r["agent_ref"], r["lane"], r["event_idx"]): r for r in requests if r.get("request_dialect")
    }
    out: list[dict] = []
    state: dict[tuple, dict] = {}
    for ev in events:
        k = (ev["agent_ref"], ev["lane"], ev["event_idx"])
        r = req.get(k)
        if r is None:
            continue
        names = json.loads(r["request_messages"]) if r.get("request_messages") else []
        tools = json.loads(r["request_tools"]) if r.get("request_tools") else []
        lane = state.setdefault((ev["agent_ref"], ev["lane"]), {"trie": _Trie(), "seen": set(),
                                                               "calls": []})  # fmt: skip
        shared, base_i = lane["trie"].longest(names)
        prev = lane["calls"][-1] if lane["calls"] else None
        base = lane["calls"][base_i] if base_i is not None else None
        row = {
            "agent_ref": ev["agent_ref"],
            "lane": ev["lane"],
            "event_idx": ev["event_idx"],
            "source_event_id": ev.get("source_event_id"),
            "started_at": ev.get("started_at"),
            "prompt_tokens": ev.get("prompt_tokens"),
            "completion_tokens": ev.get("completion_tokens"),
            "cached_tokens": ev.get("cached_tokens"),
            "dialect": r["request_dialect"],
            "n_messages": len(names),
            "n_tools": len(tools),
            "chars": sum((sizes.get(n) or {}).get("chars") or 0 for n in names),
            "first_sent": sum(1 for n in set(names) if n not in lane["seen"]),
            "base_event_idx": None if base is None else base["event_idx"],
            "shared": shared,
            "dropped": 0 if base is None else base["n_messages"] - shared,
            "added": len(names) - shared,
        }
        sent = (names, r.get("request_tools"), r.get("request_params"))
        if prev is None:
            row["class"] = "first"
        elif sent == prev["_sent"]:
            row["class"] = "repeat"
        elif base is not None and shared == base["n_messages"]:
            row["class"] = "append" if base is prev else "resume"
        elif shared == 0:
            row["class"] = "new_context"
        else:
            ref = base or prev
            small, big = _size(row), _size(ref)
            shrank = small is not None and big and small <= COMPACTION_SHARE * big
            fewer = len(names) < ref["n_messages"]
            row["class"] = "compaction" if shrank or fewer else "rewrite"
        lane["seen"].update(names)
        lane["trie"].add(names, len(lane["calls"]))
        lane["calls"].append(row)
        row["_names"] = names
        row["_sent"] = sent
        out.append(row)
    return out


def summary(events: list[dict], requests: list[dict], sizes: dict[str, dict]) -> dict:
    """The rollout's prompt summary: every call row, and how many model calls carry a request."""
    rows = calls(events, requests, sizes)
    for r in rows:
        r.pop("_names")
        r.pop("_sent")
    counts = {c: 0 for c in CLASSES}
    for r in rows:
        counts[r["class"]] += 1
    return {
        "n_model_calls": len(events),
        "n_with_request": len(rows),
        "classes": counts,
        "compaction_share": COMPACTION_SHARE,
        "calls": rows,
    }


def usage_calls(events: list[dict]) -> list[dict]:
    """One row per model response of a rollout whose producer kept no request, from the input
    tokens each response reports: the size is what the API processed, so nothing is rebuilt. A
    response written as several events (one per content block, each with the whole usage) is one
    row, at its first event. Calls are classed by size alone: `compaction` after a producer's
    compaction marker (itself a call where it carries usage) or a drop to at most
    `COMPACTION_SHARE` of the call before, `rewrite` for a smaller drop (history was dropped),
    else `first` or `append`."""
    out: list[dict] = []
    last: dict[tuple, dict] = {}
    by_response: dict[tuple, dict] = {}
    marked: set[tuple] = set()
    for ev in sorted(events, key=lambda e: (e["agent_ref"], e["lane"] or "", e["event_idx"])):
        lane = (ev["agent_ref"], ev["lane"])
        marker = ev.get("kind_raw") in COMPACTION_MARKERS
        if not ev.get("prompt_tokens"):
            if marker:
                marked.add(lane)
            continue
        rid = ev.get("request_idx")
        row = by_response.get((*lane, rid)) if rid is not None else None
        if row is not None:
            row["completion_tokens"] = max(
                row["completion_tokens"] or 0, ev.get("completion_tokens") or 0
            )
            continue
        prev = last.get(lane)
        size = ev["prompt_tokens"]
        if prev is None:
            cls = "first"
        elif lane in marked or size <= COMPACTION_SHARE * prev["prompt_tokens"]:
            cls = "compaction"
        elif size < prev["prompt_tokens"]:
            cls = "rewrite"
        else:
            cls = "append"
        marked.discard(lane)
        row = {
            "agent_ref": ev["agent_ref"],
            "lane": ev["lane"],
            "event_idx": ev["event_idx"],
            "source_event_id": ev.get("source_event_id"),
            "started_at": ev.get("started_at"),
            "prompt_tokens": size,
            "completion_tokens": ev.get("completion_tokens"),
            "cached_tokens": ev.get("cached_tokens"),
            "has_request": False,
            "class": cls,
            "base_event_idx": None if prev is None else prev["event_idx"],
        }
        last[lane] = row
        if marker:
            marked.add(lane)
        if rid is not None:
            by_response[(*lane, rid)] = row
        out.append(row)
    return out


def usage_summary(events: list[dict]) -> dict:
    """The rollout's prompt summary from usage alone: `source` is `usage`, and no call has a
    request, a message count or a base's history to compare."""
    rows = usage_calls(events)
    counts = {c: 0 for c in CLASSES}
    for r in rows:
        counts[r["class"]] += 1
    return {
        "source": "usage",
        "n_model_calls": len(rows),
        "n_with_request": 0,
        "classes": counts,
        "compaction_share": COMPACTION_SHARE,
        "calls": rows,
    }


def detail(
    events: list[dict],
    requests: list[dict],
    sizes: dict[str, dict],
    lane: str,
    event_idx: int,
    fetch: Callable[[list[str]], dict[str, str]],
    tool_names: dict[str, str | None],
) -> dict | None:
    """One call's prompt: its row, its parameters and tools, and each message in order, marked
    `kept` (the leading run it shares with its base), `new` (first sent in this lane here) or
    `resent`. `fetch(hashes)` answers the JSON of the messages that are not kept."""
    rows = calls(events, requests, sizes)
    row = next((r for r in rows if r["lane"] == lane and r["event_idx"] == event_idx), None)
    if row is None:
        return None
    names = row.pop("_names")
    row.pop("_sent")
    seen: set[str] = set()
    base_names: list[str] = []
    for r in rows:
        if r is row:
            break
        if r["lane"] == lane and r["agent_ref"] == row["agent_ref"]:
            seen.update(r["_names"])
            if r["event_idx"] == row["base_event_idx"]:
                base_names = r["_names"]
    texts = fetch(sorted(set(names[row["shared"] :])))
    messages = []
    for pos, n in enumerate(names):
        s = sizes.get(n) or {}
        m = {"pos": pos, "msg_hash": n, "role": s.get("role"), "item_type": s.get("item_type"),
             "chars": s.get("chars")}  # fmt: skip
        if pos < row["shared"]:
            m["state"] = "kept"
        else:
            m["state"] = "resent" if n in seen else "new"
            text = message_text(texts.get(n))
            m["preview"] = text[:PREVIEW_CHARS]
            m["preview_chars"] = len(text)
        messages.append(m)
    req = next(r for r in requests if r["lane"] == lane and r["event_idx"] == event_idx
               and r["agent_ref"] == row["agent_ref"])  # fmt: skip
    tools = json.loads(req["request_tools"]) if req.get("request_tools") else []
    dropped = base_names[row["shared"] :]
    return {
        "call": row,
        "params": json.loads(req["request_params"]) if req.get("request_params") else None,
        "tools": [{"sha256": t, "name": tool_names.get(t)} for t in tools],
        "messages": messages,
        "dropped": dropped[:DROPPED_LISTED],
        "n_dropped": len(dropped),
        "preview_chars": PREVIEW_CHARS,
    }
