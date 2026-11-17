# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`evalrdl-trace serve <path>`: one run's trace in the browser, on this machine.

`path` is one run's tables (a directory of `<table>.parquet`), a `trace` export ZIP, or a run
folder a producer wrote, which the public readers read first (the `folders` extra).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.parse
import zipfile
from pathlib import Path

import polars as pl
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from . import STATIC, RolloutKey
from .routes import artifact_routes, routes
from .source import ArtifactResolver
from .tables import TablesSource, folder_resolver


def tables_of(path: Path, scratch: Path) -> Path:
    """The directory of `<table>.parquet` files `path` is, holds, or reads as."""
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(scratch)
        return scratch
    if (path / "rollouts.parquet").is_file():
        return path
    from evalrdl_readers.convert import convert, write_tables
    from evalrdl_readers.readers import RUNDIR

    result = RUNDIR.read(str(path))
    if not result.claimed:
        raise SystemExit(f"{path}: no public reader claims this folder")
    conv = convert(result, path=str(path), label=path.name, readers=RUNDIR.readers)
    write_tables(conv, scratch)
    return scratch


def app(directory: Path, resolver: ArtifactResolver | None = None) -> Starlette:
    """`resolver` reads the artifacts not stored inline; without one only inline ones have bytes."""
    source = TablesSource(directory, resolver=resolver)
    runs = (
        pl.read_parquet(directory / "runs.parquet").to_dicts()
        if (directory / "runs.parquet").is_file()
        else [
            {"run_id": r}
            for r in pl.read_parquet(directory / "rollouts.parquet")["run_id"].unique()
        ]
    )
    run_ids = {r["run_id"] for r in runs}

    def scope(request: Request) -> RolloutKey | Response:
        p = request.path_params
        if p["run_id"] not in run_ids:
            return JSONResponse({"error": "run not found"}, status_code=404)
        try:
            repeat = int(p["repeat"])
        except ValueError:
            return JSONResponse({"error": "bad repeat"}, status_code=404)
        task = urllib.parse.unquote(p["task_id"]) if "%2f" in p["task_id"].lower() else p["task_id"]
        return RolloutKey(p["run_id"], task, repeat)

    def page(request: Request) -> Response:
        return FileResponse(STATIC / "viewer.html")

    def static(request: Request) -> Response:
        f = STATIC / request.path_params["name"]
        return FileResponse(f) if f.is_file() else JSONResponse({"error": "not found"}, 404)

    def list_runs(request: Request) -> Response:
        return JSONResponse(runs)

    def get_run(request: Request) -> Response:
        run = next((r for r in runs if r["run_id"] == request.path_params["run_id"]), None)
        return JSONResponse(run) if run else JSONResponse({"error": "run not found"}, 404)

    def run_rollouts(request: Request) -> Response:
        rows = source.tables.scan("rollouts").filter(
            pl.col("run_id") == request.path_params["run_id"]
        )
        return JSONResponse(rows.sort("task_id", "repeat", "agent_ref").collect().to_dicts())

    return Starlette(
        routes=[
            Route("/", page),
            Route("/static/{name}", static),
            Route("/v1/runs", list_runs),
            Route("/v1/runs/{run_id}", get_run),
            Route("/v1/runs/{run_id}/rollouts", run_rollouts),
            *artifact_routes(source, scope),
            *routes(source, scope),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalrdl-trace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="serve one run's trace on this machine")
    serve.add_argument("path", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    import uvicorn

    with tempfile.TemporaryDirectory(prefix="evalrdl-trace-") as scratch:
        path = args.path.resolve()
        directory = tables_of(path, Path(scratch))
        resolver = folder_resolver(path) if path.is_dir() and directory != path else None
        print(f"{args.path}: http://{args.host}:{args.port}/", flush=True)
        uvicorn.run(app(directory, resolver), host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
