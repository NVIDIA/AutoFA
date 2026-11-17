# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host the vendored trace viewer over immutable stores in the local profile."""

from __future__ import annotations

import hashlib
import html
import json
import re
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import polars as pl
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from autofa.data.local_trace_store import resolve_store
from evalrdl_trace import STATIC, RolloutKey
from evalrdl_trace.routes import ROLLOUT, routes
from evalrdl_trace.tables import TablesSource

from .theme import ui_theme

_PACKAGE = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_PACKAGE / "templates"))
_TEMPLATES.env.globals["ui_theme"] = ui_theme
_SAFE_DIGEST = re.compile(r"[0-9a-f]{64}")
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
_TRACE_ASSETS = (
    *(STATIC / name for name in ("style.css", "kit.js", "graph.js", "trace.js")),
    _PACKAGE / "static" / "trace_theme.css",
    _PACKAGE / "static" / "trace_host.js",
)


def _trace_asset_version() -> str:
    """Reload the viewer and its host together whenever either bundle changes."""
    digest = hashlib.sha256()
    for path in _TRACE_ASSETS:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


class LocalTablesSource(TablesSource):
    """Hydrate only one event window's previews; full text is an explicit point read."""

    def event_previews(self, key, *, offset=None, limit=None, lane=None):
        rows = super().events(key, parts=False, offset=offset, limit=limit, lane=lane)
        if not rows:
            return rows
        event_keys = ["agent_ref", "lane", "event_idx"]
        part_keys = [*event_keys, "part_idx"]
        schema = self.tables.scan("events").collect_schema()
        window = pl.DataFrame(
            [{key: row[key] for key in event_keys} for row in rows],
            schema={key: schema[key] for key in event_keys},
        ).lazy()
        selected = self._of("parts", key, lane).join(
            window, on=event_keys, how="semi", nulls_equal=True,
        )
        texts = self._of("part_texts", key, lane).join(
            window, on=event_keys, how="semi", nulls_equal=True,
        ).select(
            *part_keys, pl.col("text").str.slice(0, self.text_cap),
            pl.col("text").str.len_chars().alias("text_chars"),
            (pl.col("text").str.len_chars() > self.text_cap).alias("text_truncated"),
        )
        parts = selected.join(texts, on=part_keys, how="left", nulls_equal=True)
        grouped = {}
        for part in parts.sort(part_keys, nulls_last=True).collect().to_dicts():
            grouped.setdefault(tuple(part[key] for key in event_keys), []).append(part)
        for event in rows:
            event["parts"] = grouped.get(tuple(event[key] for key in event_keys), [])
        return rows


def _missing(message: str = "Local trace is unavailable") -> JSONResponse:
    return JSONResponse({"error": message}, status_code=404, headers=_HEADERS)


def _store_app(directory: Path, store_id: str) -> Starlette:  # noqa: C901
    """One source instance belongs to one generation, even when run IDs repeat."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source = LocalTablesSource(directory / "tables")
    run_id = manifest["run_id"]
    rows = source.tables.scan("runs").filter(pl.col("run_id") == run_id).collect().to_dicts()
    run = rows[0] if len(rows) == 1 else {"run_id": run_id}
    rollouts = source.tables.scan("rollouts").filter(pl.col("run_id") == run_id).collect()
    identities: dict[tuple[str, int], list[str]] = {}
    for row in rollouts.select("task_id", "repeat").to_dicts():
        # Native viewer URLs escape slashes twice, everything else once. Match the
        # ASGI-decoded spelling exactly; unquote() would corrupt literal %20 etc.
        # A literal %2F and / have the same native address: refuse collisions.
        key = (row["task_id"].replace("/", "%2F"), row["repeat"])
        identities.setdefault(key, []).append(row["task_id"])

    def scope(request: Request) -> RolloutKey | Response:
        p = request.path_params
        if p["run_id"] != run_id:
            return _missing("Run not found in this local store")
        try:
            repeat = int(p["repeat"])
        except ValueError:
            return JSONResponse({"error": "Invalid repeat"}, status_code=400)
        tasks = identities.get((p["task_id"], repeat), [])
        if not tasks:
            return _missing("Rollout not found")
        if len(tasks) != 1:
            return JSONResponse({"error": "Ambiguous rollout or encoded task identity"},
                                status_code=409)
        return RolloutKey(run_id, tasks[0], repeat)

    def page(request: Request) -> Response:
        identity = getattr(request.state, "autofa_identity", None)
        return _TEMPLATES.TemplateResponse(
            request, "trace_viewer.html",
            {"store_id": store_id, "run": run, "me": identity(request) if identity else None,
             "trace_asset_version": _trace_asset_version()},
            headers=_HEADERS,
        )

    def list_runs(request: Request) -> Response:
        return JSONResponse([run], headers=_HEADERS)

    def get_run(request: Request) -> Response:
        return (JSONResponse(run, headers=_HEADERS)
                if request.path_params["run_id"] == run_id else _missing("Run not found"))

    def list_rollouts(request: Request) -> Response:
        if request.path_params["run_id"] != run_id:
            return _missing("Run not found")
        return JSONResponse(rollouts.sort("task_id", "repeat").to_dicts(), headers=_HEADERS)

    def artifacts(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        # Original paths/URLs describe provenance, not permission to fetch their bytes.
        rows = source.artifacts(key)
        for row in rows:
            row.pop("view_url", None)
            row.pop("download_url", None)
        return JSONResponse(rows, headers=_HEADERS)

    def event_previews(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        query = request.query_params
        try:
            offset = int(query.get("offset", 0))
            limit = int(query.get("limit", 50))
            if offset < 0 or not 0 <= limit <= 200:
                raise ValueError
        except ValueError:
            return JSONResponse({"error": "Invalid event preview window"}, status_code=400)
        return JSONResponse(source.event_previews(
            key, offset=offset, limit=limit, lane=query.get("lane"),
        ), headers=_HEADERS)

    def artifact_file(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        try:
            index = int(request.path_params["artifact_idx"])
        except ValueError:
            return _missing("Artifact not found")
        matches = [r for r in manifest.get("artifact_files", [])
                   if r.get("task_id") == key.task_id and r.get("repeat") == key.repeat
                   and r.get("artifact_idx") == index]
        if len(matches) != 1:
            return _missing("Original artifact bytes were not captured in this local store")
        record = matches[0]
        digest = record.get("sha256", "")
        if not isinstance(digest, str) or not _SAFE_DIGEST.fullmatch(digest):
            return _missing("Invalid managed artifact")
        expected = f"artifacts/{digest}"
        if record.get("path") != expected:
            return _missing("Invalid managed artifact")
        path = directory / expected
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
            return _missing("Managed artifact unavailable")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                return _missing("Managed artifact checksum changed")
            stream.seek(0)
            head = stream.read(16)
        if request.path_params["action"] == "download":
            return FileResponse(path, media_type="application/octet-stream", filename=digest,
                                headers=_HEADERS)
        media_type = (
            "image/png" if head.startswith(b"\x89PNG\r\n\x1a\n") else
            "image/jpeg" if head.startswith(b"\xff\xd8\xff") else
            "image/gif" if head.startswith((b"GIF87a", b"GIF89a")) else
            "image/webp" if head.startswith(b"RIFF") and head[8:12] == b"WEBP" else None
        )
        if request.path_params["action"] != "view" or media_type is None:
            return _missing("This artifact has no inline image preview")
        return FileResponse(path, media_type=media_type, headers={
            **_HEADERS, "Content-Security-Policy": "default-src 'none'; sandbox",
        })

    def full_text(request: Request, *, as_json=False) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        try:
            event_idx = int(request.query_params["event_idx"])
            part_idx = (int(request.query_params["part_idx"])
                        if "part_idx" in request.query_params else None)
        except (KeyError, ValueError):
            return JSONResponse({"error": "An event index is required"}, status_code=400)
        if as_json and part_idx is None:
            return JSONResponse({"error": "A part index is required"}, status_code=400)
        lane = request.query_params.get("lane")
        events = [event for event in source.events(key, parts=False, lane=lane)
                  if event["event_idx"] == event_idx]
        if not events:
            return _missing("Event not found")
        if len(events) != 1:
            return JSONResponse({"error": "Specify the event's lane"}, status_code=409)
        at = ((pl.col("run_id") == key.run_id) & (pl.col("task_id") == key.task_id)
              & (pl.col("repeat") == key.repeat) & (pl.col("event_idx") == event_idx)
              & pl.col("agent_ref").eq_missing(events[0]["agent_ref"])
              & pl.col("lane").eq_missing(events[0]["lane"]))
        part_at = at & (pl.lit(True) if part_idx is None else pl.col("part_idx") == part_idx)
        parts = source.tables.scan("parts").filter(part_at).sort("part_idx").collect().to_dicts()
        text_rows = source.tables.scan("part_texts").filter(part_at).collect().to_dicts()
        texts = {row["part_idx"]: row["text"] for row in text_rows}
        if part_idx is not None and not parts:
            return _missing("Part not found")
        if as_json:
            if len(parts) != 1 or len(text_rows) > 1:
                return JSONResponse({"error": "Ambiguous part identity"}, status_code=409)
            text = texts.get(part_idx)
            if not isinstance(text, str):
                return _missing("No text was recorded for this part")
            identity = {key: parts[0][key] for key in (
                "run_id", "task_id", "repeat", "agent_ref", "lane", "event_idx", "part_idx",
            )}
            return JSONResponse({**identity, "text": text, "chars": len(text)}, headers=_HEADERS)
        blocks = []
        for part in parts:
            text = texts.get(part["part_idx"])
            if isinstance(text, str):
                blocks.append(f"<h2>Part {part['part_idx']}</h2><pre>{html.escape(text)}</pre>")
        if part_idx is None:
            calls = (source.tables.scan("tool_call_texts").filter(at).sort("call_idx")
                     .collect().to_dicts())
            for call in calls:
                text = call.get("args_full")
                if isinstance(text, str):
                    blocks.append(f"<h2>Tool arguments {call['call_idx']}</h2>"
                                  f"<pre>{html.escape(text)}</pre>")
        return HTMLResponse(
            '<!doctype html><html><head><meta charset="utf-8"><title>Full trace text</title>'
            '<style>body{margin:2rem;font-family:system-ui}pre{white-space:pre-wrap;'
            'overflow-wrap:anywhere}h1,h2{font-size:1rem}</style></head><body>'
            f"<h1>Event {event_idx} · {html.escape(events[0]['lane'] or 'conversation')}</h1>"
            + ("".join(blocks) or "<p>No text was recorded for this part.</p>") + "</body></html>",
            headers={**_HEADERS, "Content-Security-Policy":
                     "sandbox; default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'"},
        )

    def part_text(request: Request) -> Response:
        return full_text(request, as_json=True)

    def export(request: Request) -> Response:
        key = scope(request)
        if isinstance(key, Response):
            return key
        from evalrdl_export import TRACE
        from evalrdl_schema.tables import RunTables

        scratch = tempfile.TemporaryDirectory(prefix="autofa-trace-export-")
        output = Path(scratch.name) / "trace.zip"
        try:
            TRACE.to_file(RunTables(directory / "tables", TRACE.reads), run,
                          (key.task_id, key.repeat), output)
        except BaseException:
            scratch.cleanup()
            raise
        return FileResponse(output, filename="trace.zip", media_type="application/zip",
                            headers=_HEADERS, background=BackgroundTask(scratch.cleanup))

    prefix = "/api/v1/rollouts"
    app = Starlette(routes=[
        Route("/", page),
        Route("/api/v1/runs", list_runs),
        Route("/api/v1/runs/{run_id}", get_run),
        Route("/api/v1/runs/{run_id}/rollouts", list_rollouts),
        Route(prefix + ROLLOUT + "/artifacts", artifacts),
        Route(prefix + ROLLOUT + "/artifacts/{artifact_idx}/{action}", artifact_file),
        Route(prefix + ROLLOUT + "/text", full_text),
        Route(prefix + ROLLOUT + "/part-text", part_text),
        Route(prefix + ROLLOUT + "/events/preview", event_previews),
        Route(prefix + ROLLOUT + "/export", export),
        *routes(source, scope, prefix=prefix),
    ])
    return app


class LocalTraces:
    """Bounded source cache behind the parent application's local-origin boundary."""

    def __init__(self, identity=None):
        self._cache: OrderedDict[tuple, Starlette] = OrderedDict()
        self._lock = threading.Lock()
        self._identity = identity

    def _app(self, store_id: str) -> Starlette:
        directory = resolve_store(store_id)
        stat = (directory / "manifest.json").stat()
        key = (str(directory), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = _store_app(directory, store_id)
                while len(self._cache) > 16:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            return self._cache[key]

    async def __call__(self, scope, receive, send):
        try:
            app = await run_in_threadpool(self._app, scope["path_params"]["store_id"])
        except (OSError, ValueError, KeyError):
            await _missing()(scope, receive, send)
            return
        if self._identity is not None:
            scope = {**scope, "state": {
                **scope.get("state", {}), "autofa_identity": self._identity,
            }}
        await app(scope, receive, send)


def mount_traces(application, identity=None) -> None:
    """Mount within AutoFA so the existing host/origin and profile lifecycle apply.

    `identity(request)`: the signed-in user for the page's top bar, or None."""
    application.mount("/trace-static", StaticFiles(directory=STATIC), name="trace-static")
    application.router.routes.append(Mount("/traces/{store_id}", app=LocalTraces(identity)))
