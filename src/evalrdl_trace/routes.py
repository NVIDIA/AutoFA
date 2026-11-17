# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The HTTP routes of one rollout's trace, over any `TraceSource`, and of its artifacts, over any
`ArtifactSource`.

A host supplies `scope`: it turns a request into the rollout's key, or into the response that
refuses it (an unknown run, a malformed repeat, a caller who may not read it).
"""

from __future__ import annotations

import urllib.parse
from collections import Counter
from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from evalrdl_schema.vocab import edge_kinds

from . import prompts
from .assemble import timeline
from .source import ArtifactSource, RolloutKey, TraceSource

#: Every route's path below the rollout's own.
ROLLOUT = "/{run_id}/{task_id}/{repeat}"

#: Characters of one part's text one `parts/{part_idx}/text` read returns at most.
PART_TEXT_WINDOW = 200_000
#: Bytes an artifact `view` renders at most; the download is uncapped.
VIEW_MAX = 25 << 20
#: Image types `view` renders, by the bytes' own signature. SVG is never one: it runs script.
IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

#: Hits one `search` read returns at most.
SEARCH_MAX = 500

#: Hashes one `prompt_messages` read may name.
PROMPT_MESSAGES_MAX = 200
#: Characters of one message's JSON a `prompt_messages` read returns; the export has the rest.
MESSAGE_CAP = 200_000

Scope = Callable[[Request], "RolloutKey | Response"]


def routes(source: TraceSource, scope: Scope, prefix: str = "/v1/rollouts") -> list[Route]:  # noqa: C901
    """The rollout, its events, lanes, parts, tool calls, attempts, provenance, edges, trace map
    and timeline, under `prefix`."""

    def _unknown_lane(key: RolloutKey, lane: str) -> JSONResponse | None:
        # An unknown lane is a 400 naming the lanes there are, not 200 with zero rows.
        known = [ln["lane"] for ln in source.lanes(key)]
        if lane in known:
            return None
        return JSONResponse(
            {
                "error": "unknown lane",
                "have": lane,
                "lanes": known,
                "how": "GET .../lanes for this rollout's sequences; the unnamed lane is the "
                "empty string",
            },
            status_code=400,
        )

    def get_rollout(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        row = source.rollout(key)
        if row is None:
            return JSONResponse({"error": "rollout not found"}, status_code=404)
        return JSONResponse(row)

    def rollout_events(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        qp = request.query_params
        if "include" in qp and qp["include"] != "parts":
            return JSONResponse(
                {
                    "error": "unknown include",
                    "have": qp["include"],
                    "accepts": ["parts"],
                    "how": "omit `include` for events alone, or `include=parts` to hydrate them",
                },
                status_code=400,
            )
        include = qp.get("include") == "parts"
        try:
            offset = int(qp["offset"]) if "offset" in qp else None
            limit = int(qp["limit"]) if "limit" in qp else None
        except ValueError:
            return JSONResponse({"error": "offset/limit must be integers"}, status_code=400)
        if (offset is not None and offset < 0) or (limit is not None and limit < 0):
            return JSONResponse({"error": "offset/limit must be non-negative"}, status_code=400)
        lane = qp.get("lane")
        if lane is not None and (bad := _unknown_lane(key, lane)):
            return bad
        return JSONResponse(
            source.events(key, parts=include, offset=offset, limit=limit, lane=lane)
        )

    def rollout_lanes(request: Request) -> Response:
        """The rollout's lanes, each with its event range: what turns an `event_idx` into an
        address. A rollout stored before lanes answers one row, lane `""`."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        lanes = source.lanes(key)
        return JSONResponse(
            {
                "lanes": lanes,
                "n_lanes": len(lanes),
                "multi_lane": len({ln["lane"] for ln in lanes}) > 1,
            }
        )

    def rollout_provenance(request: Request) -> Response:
        """The run's provenance narrowed to one rollout, plus the run-grain rows that settle its
        values. An aggregate: a row names a single cell's source only where `n_values` is 1."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        rows = source.provenance(key)
        return JSONResponse(
            {"rows": rows, "n": len(rows), "grain": "question,input,site", "aggregate": True}
        )

    def rollout_edges(request: Request) -> Response:
        """Every stated edge touching this rollout. An edge whose far end reached no stored row
        is returned and counted in `dangling`, never filtered away."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        rows = source.edges(key)
        resolved = sum(1 for r in rows if r.get("resolved"))
        vocab = edge_kinds()
        return JSONResponse(
            {
                "edges": rows,
                "n": len(rows),
                "resolved": resolved,
                "dangling": len(rows) - resolved,
                "by_author": dict(Counter(r.get("asserted_by") or "unknown" for r in rows)),
                # What each kind present means, and which end it implies came first.
                "kinds": {
                    s: {
                        "means": vocab[s].means,
                        "earlier": vocab[s].earlier,
                        "because": vocab[s].because,
                        "inferred_from": vocab[s].inferred_from,
                    }
                    for s in sorted({r.get("space") for r in rows} & set(vocab))
                },
            }
        )

    def rollout_event_parts(request: Request) -> Response:
        """Point read: one event's row, its parts (truncated server-side) and its tool calls.

        `event_idx` numbers per lane, so the lane goes in `?lane=`. Without it the read resolves
        when one lane numbers that event and 409s with the candidates when several do; it never
        guesses."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        try:
            event_idx = int(request.path_params["event_idx"])
        except ValueError:
            return JSONResponse({"error": "bad event index"}, status_code=404)
        lane = request.query_params.get("lane")
        data = source.event_parts(key, event_idx, lane)
        if data is None:
            return JSONResponse({"error": "event not found"}, status_code=404)
        cands = sorted({e["lane"] for e in data["events"]})
        if lane is None and len(cands) > 1:
            base = request.url.path
            return JSONResponse(
                {
                    "error": "ambiguous event address",
                    "detail": f"event {event_idx} exists in {len(cands)} lanes of this rollout; "
                    "name the lane with ?lane=",
                    "event_idx": event_idx,
                    "lanes": [
                        {"lane": ln, "url": f"{base}?lane={urllib.parse.quote(ln, safe='')}"}
                        for ln in cands
                    ],
                },
                status_code=409,
            )
        data["lane"] = lane if lane is not None else (cands[0] if cands else None)
        return JSONResponse(data)

    def rollout_part_text(request: Request) -> Response:
        """One part's whole text, or the `[start, end)` window of it, in characters.

        The event and part reads cap texts; this is how the rest is read. Lane rules are the
        event read's: without `?lane=` it 409s when several lanes have the part."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        try:
            event_idx = int(request.path_params["event_idx"])
            part_idx = int(request.path_params["part_idx"])
            start = int(request.query_params.get("start", 0))
            end = request.query_params.get("end")
            end = None if end is None else int(end)
        except ValueError:
            return JSONResponse({"error": "bad index or range"}, status_code=400)
        if start < 0 or (end is not None and end < start):
            return JSONResponse({"error": "bad index or range"}, status_code=400)
        lane = request.query_params.get("lane")
        rows = source.part_text(key, event_idx, part_idx, lane)
        if not rows:
            return JSONResponse({"error": "part not found"}, status_code=404)
        if len(rows) > 1:
            return JSONResponse(
                {
                    "error": "ambiguous part address",
                    "detail": "name the lane with ?lane=",
                    "lanes": sorted(r["lane"] for r in rows),
                },
                status_code=409,
            )
        text = rows[0]["text"] or ""
        stop = min(len(text), start + PART_TEXT_WINDOW, len(text) if end is None else end)
        return JSONResponse(
            {
                "lane": rows[0]["lane"],
                "event_idx": event_idx,
                "part_idx": part_idx,
                "chars": len(text),
                "start": min(start, len(text)),
                "end": stop,
                "eof": stop >= len(text),
                "text": text[start:stop],
            }
        )

    def rollout_search(request: Request) -> Response:
        """The parts and tool call arguments holding every word of `?q=`, in trace order.

        A source without text search answers 404; `truncated` says more hits exist."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        found = getattr(source, "search", None)
        if found is None:
            return JSONResponse({"error": "this store has no text search"}, status_code=404)
        q = (request.query_params.get("q") or "").strip()
        if not q:
            return JSONResponse({"error": "q takes words to find"}, status_code=400)
        try:
            limit = min(SEARCH_MAX, max(1, int(request.query_params.get("limit", SEARCH_MAX))))
        except ValueError:
            return JSONResponse({"error": "bad limit"}, status_code=400)
        hits = found(key, q, limit + 1)
        return JSONResponse({"hits": hits[:limit], "truncated": len(hits) > limit})

    def rollout_trace_map(request: Request) -> Response:
        """Per-rollout aggregate for a trace map; never ships texts."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        data = source.trace_map(key)
        if data is None:
            return JSONResponse({"error": "rollout not found"}, status_code=404)
        return JSONResponse(data)

    def rollout_timeline(request: Request) -> Response:
        """Timing for a timeline: per-event start + duration, tool-call spans, rollup."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        data = source.timeline_rows(key)
        if data is None:
            return JSONResponse({"error": "rollout not found"}, status_code=404)
        return JSONResponse(timeline(data, source.event_cap))

    def rollout_tool_calls(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        include = request.query_params.get("include") == "args"
        lane = request.query_params.get("lane")
        if lane is not None and (bad := _unknown_lane(key, lane)):
            return bad
        return JSONResponse(source.tool_calls(key, args=include, lane=lane))

    def _prompt_rows(key: RolloutKey) -> tuple[list[dict], list[dict], dict[str, dict]]:
        reqs = source.requests(key)
        if not reqs:
            return [], [], {}
        events = [e for e in source.events(key, parts=False) if e.get("lane_kind") == "model_call"]
        sizes = {m["msg_hash"]: m for m in source.prompt_messages(key)}
        return events, reqs, sizes

    def rollout_prompts(request: Request) -> Response:
        """Each model call that carries its request: its size, the earlier call it continues
        (its base), how many of the base's messages it kept, dropped and added, and what that
        makes it (`append`, `repeat`, `resume`, `new_context`, `rewrite`, `compaction`). Never
        ships texts."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        if source.rollout(key) is None:
            return JSONResponse({"error": "rollout not found"}, status_code=404)
        events, reqs, sizes = _prompt_rows(key)
        if not reqs:
            every = source.events(key, parts=False)
            events = [e for e in every if e.get("lane_kind") == "model_call"]
            if not events:
                return JSONResponse(prompts.usage_summary(every))
        return JSONResponse(prompts.summary(events, reqs, sizes))

    def rollout_prompt(request: Request) -> Response:
        """One call's prompt: its parameters and tools, and every message in order, each marked
        `kept` (shared with its base), `new` or `resent`, with a preview of those not kept.
        `event_idx` numbers per lane, so the lane goes in `?lane=` (default `model_call`)."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        try:
            event_idx = int(request.path_params["event_idx"])
        except ValueError:
            return JSONResponse({"error": "bad event index"}, status_code=404)
        lane = request.query_params.get("lane", "model_call")
        events, reqs, sizes = _prompt_rows(key)
        mine = next((r for r in reqs if r["lane"] == lane and r["event_idx"] == event_idx), None)
        if mine is None:
            return JSONResponse({"error": "no request on this event", "lane": lane,
                                 "event_idx": event_idx}, status_code=404)  # fmt: skip

        def fetch(hashes: list[str]) -> dict[str, str]:
            if not hashes:
                return {}
            return {m["msg_hash"]: m.get("msg_json")
                    for m in source.prompt_messages(key, hashes, text=True)}  # fmt: skip

        tool_names = {t["sha256"]: t.get("name") for t in source.tool_defs(key)}
        data = prompts.detail(events, reqs, sizes, lane, event_idx, fetch, tool_names)
        if data is None:
            return JSONResponse({"error": "no request on this event"}, status_code=404)
        return JSONResponse(data)

    def rollout_prompt_messages(request: Request) -> Response:
        """Messages by hash (`?hash=` repeated, at most 200): each one's JSON, capped, and its
        readable text."""
        key = scope(request)
        if isinstance(key, Response):
            return key
        hashes = request.query_params.getlist("hash")
        if not hashes or len(hashes) > PROMPT_MESSAGES_MAX:
            return JSONResponse({"error": f"name 1 to {PROMPT_MESSAGES_MAX} messages with ?hash="},
                                status_code=400)  # fmt: skip
        rows = source.prompt_messages(key, hashes, text=True)
        out = []
        for m in rows:
            j = m.get("msg_json") or ""
            out.append({
                "msg_hash": m["msg_hash"], "role": m.get("role"), "item_type": m.get("item_type"),
                "chars": len(j), "truncated": len(j) > MESSAGE_CAP,
                "msg_json": j[:MESSAGE_CAP], "text": prompts.message_text(j)[:MESSAGE_CAP],
            })  # fmt: skip
        return JSONResponse({"messages": out, "cap_chars": MESSAGE_CAP})

    def rollout_attempts(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        return JSONResponse(source.attempts(key))

    return [
        Route(prefix + ROLLOUT + path, endpoint)
        for path, endpoint in (
            ("", get_rollout),
            ("/events", rollout_events),
            ("/lanes", rollout_lanes),
            ("/provenance", rollout_provenance),
            ("/edges", rollout_edges),
            ("/events/{event_idx}/parts", rollout_event_parts),
            ("/events/{event_idx}/parts/{part_idx}/text", rollout_part_text),
            ("/search", rollout_search),
            ("/trace_map", rollout_trace_map),
            ("/timeline", rollout_timeline),
            ("/tool_calls", rollout_tool_calls),
            ("/attempts", rollout_attempts),
            ("/prompts", rollout_prompts),
            ("/prompts/{event_idx}", rollout_prompt),
            ("/prompt_messages", rollout_prompt_messages),
        )
    ]


def image_type(data: bytes) -> str | None:
    """The image type the bytes' signature names, else None."""
    for magic, media in IMAGE_MAGIC:
        if data.startswith(magic):
            return media
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def artifact_routes(
    source: ArtifactSource, scope: Scope, prefix: str = "/v1/rollouts"
) -> list[Route]:
    """`/artifacts`, `/artifacts/{idx}/view` and `/artifacts/{idx}/download` of a rollout.

    `view` renders an image only when its bytes say it is one, and never as anything that can
    run script; `download` is always an attachment. A row is listed with `view_url` and
    `download_url` relative to `prefix`, null where there is nothing to fetch."""

    def _artifact(request: Request):
        key = scope(request)
        if isinstance(key, Response):
            return key, None, None
        try:
            idx = int(request.path_params["idx"])
        except ValueError:
            return JSONResponse({"error": "bad artifact index"}, status_code=404), None, None
        got = source.artifact_bytes(key, idx)
        if got is None:
            return JSONResponse({"error": "artifact has no readable bytes"}, 404), None, None
        return key, idx, got

    def listing(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        task = urllib.parse.quote(key.task_id, safe="").replace("%2F", "%252F")
        base = f"{prefix}/{urllib.parse.quote(key.run_id, safe='')}/{task}/{key.repeat}"
        rows = source.artifacts(key)
        for row in rows:
            url = f"{base}/artifacts/{row['artifact_idx']}"
            has_bytes = row.get("storage") == "inline" or bool(row.get("uri"))
            row["download_url"] = url + "/download" if has_bytes else None
            row["view_url"] = url + "/view" if has_bytes and row.get("kind") == "image" else None
        return JSONResponse(rows)

    def view(request: Request) -> Response:
        key, idx, got = _artifact(request)
        if idx is None:
            return key
        media = image_type(got.data)
        if media is None or len(got.data) > VIEW_MAX:
            return JSONResponse({"error": "not an image this view renders"}, status_code=415)
        return Response(
            got.data,
            media_type=media,
            headers={
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; sandbox",
            },
        )

    def download(request: Request) -> Response:
        key, idx, got = _artifact(request)
        if idx is None:
            return key
        row = next((r for r in source.artifacts(key) if r["artifact_idx"] == idx), {})
        name = (row.get("name") or f"artifact-{idx}").rsplit("/", 1)[-1]
        return Response(
            got.data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''"
                + urllib.parse.quote(name, safe=""),
                "X-Content-Type-Options": "nosniff",
            },
        )

    return [
        Route(prefix + ROLLOUT + "/artifacts", listing),
        Route(prefix + ROLLOUT + "/artifacts/{idx}/view", view),
        Route(prefix + ROLLOUT + "/artifacts/{idx}/download", download),
    ]
