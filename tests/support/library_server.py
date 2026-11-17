# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local library server apps, saved source stores and lazy report runs."""

import base64
import hashlib
import json
from urllib.parse import quote, unquote

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import detector_hash, passage
from autofa.data.parquet_store import Dataset
from autofa.data.reference_data import make_local_reference_data
from autofa.data.source_normalization import normalize_events
from autofa.data.trace_fields import derive_trace_fields
from autofa_library_server.catalog import LocalCatalog
from autofa_library_server.config import Settings
from autofa_library_server.main import create_app
from evalrdl_schema import CURRENT, polars_schemas

# Local trace viewer and browser.
RUN = "source-run"
TASK = "suite/task%20 π"
ENCODED_TASK = quote(TASK, safe="").replace("%2F", "%252F")
STORE = "a" * 64


TEXT = ("🙂 / earlier / " + "context " * 1200
        + "🙂 TAIL EVIDENCE /testbed/astropy/coordinates/file.py <script>unsafe()</script>")


def _write_store(profile, store_id=STORE, *, text=TEXT, collision=False):
    directory = profile / "sources" / store_id
    tables = directory / "tables"
    tables.mkdir(parents=True)
    key = dict(run_id=RUN, task_id=TASK, repeat=2, agent_ref="")
    event = {**key, "lane": "conversation", "event_idx": 55,
             "kind": "tool_result", "role": "tool", "n_parts": 1, "n_tool_calls": 0}
    part = {**key, "lane": "conversation", "event_idx": 55, "part_idx": 7,
            "part_type": "text", "channel": "tool_result", "chars": len(text)}
    image_key = {**key, "lane": "conversation", "event_idx": 55, "part_idx": 8}
    rows = {
        "runs": [{"run_id": RUN, "model": "local-model", "benchmark": "local-bench"}],
        "rollouts": [{**key, "n_events": 2, "n_tool_calls": 0, "n_artifacts": 1,
                      "status": "ok", "reward": 0.0}],
        "events": [{**event, "n_parts": 2},
                   {**event, "lane": "model_call", "role": "assistant", "kind": "message"}],
        "parts": [part, {**part, "lane": "model_call"},
                  {**image_key, "part_type": "image", "artifact_idx": 0,
                   "media_type": "image/png"}],
        "part_texts": [{**part, "text": text}, {**part, "lane": "model_call", "text": "aux"}],
        "artifacts": [{**key, "artifact_idx": 0, "event_idx": 55, "part_idx": 8,
                       "name": "tiny.png", "media_type": "image/png"}],
    }
    if collision:
        rows["rollouts"].append({**rows["rollouts"][0], "task_id": TASK.replace("/", "%2F")})
    hashes = {}
    for name, content in rows.items():
        schema = polars_schemas(CURRENT)[name]
        defaults = {column.name: column.backfill for column in CURRENT.table(name).columns}
        frame = pl.DataFrame([{**defaults, **row} for row in content],
                             schema=schema, orient="row", strict=False)
        path = tables / f"{name}.parquet"
        frame.write_parquet(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = directory / "dataset.parquet"
    pl.DataFrame({"problem_id": [TASK]}).write_parquet(dataset)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZQkAAAAASUVORK5CYII="
    )
    digest = hashlib.sha256(png).hexdigest()
    (directory / "artifacts").mkdir()
    (directory / "artifacts" / digest).write_bytes(png)
    manifest = {
        "version": 1, "status": "complete", "store_id": store_id, "run_id": RUN,
        "dataset_kind": "rich", "tables": hashes,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "artifact_files": [{"task_id": TASK, "repeat": 2, "artifact_idx": 0,
                            "sha256": digest, "path": f"artifacts/{digest}"}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def _client(app, base_url):
    async def wire_paths(scope, receive, send):
        # TestClient + httpx2 decodes URL.path twice. Uvicorn decodes raw_path once;
        # preserve that wire contract when testing the viewer's escaped task IDs.
        if scope["type"] == "http":
            scope = {**scope, "path": unquote(scope["raw_path"].decode("ascii"))}
        await app(scope, receive, send)

    return TestClient(wire_paths, base_url=base_url)


def _base(store=STORE):
    return f"/traces/{store}/api/v1/rollouts/{RUN}/{ENCODED_TASK}/2"


# Local lazy report.
@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(LocalCatalog, "refresh_interval", 0)
    store = _write_store(tmp_path)
    value = normalize_events({"n_events": 1, "n_tool_calls": 0}, [{
        "event_idx": 55, "role": "tool", "kind": "tool_result", "lane": "conversation",
        "n_parts": 1, "n_tool_calls": 0,
        "parts": [{"part_idx": 7, "part_type": "text", "text": TEXT,
                   "lane": "conversation", "chars": len(TEXT)}],
    }], [], conversation_lane="conversation")
    value["evidence_scope"]["identity"] = "evalrdl_tables_v1"
    row = {"model_id": "local-model", "benchmark": "local-bench", "problem_id": TASK,
           "run_idx": 2, "is_correct": False, "response_text": json.dumps(value)}
    path = store / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    manifest = json.loads((store / "manifest.json").read_text())
    manifest["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (store / "manifest.json").write_text(json.dumps(manifest))
    reference = make_local_reference_data(store_id=STORE, run_id=RUN, parquet_path=path)
    (store / "source.json").write_text(json.dumps(reference))
    item = next(Dataset([path]).iter_batches())[0]
    iid = item["_item_id"]
    field = derive_trace_fields(item)["observations_text"]
    start = field.index("🙂 TAIL EVIDENCE")
    source = "def detector(item): return True"
    # Older saved evidence deliberately lacks native navigation; the first item
    # request must upgrade it against the full text before the inspector renders.
    evidence = {"detector_hash": detector_hash(source), "passages": [
        passage("observations_text", field, start, start + len("🙂 TAIL EVIDENCE"),
                method="literal"),
    ]}
    library = FunctionLibrary(model_id="local-model", benchmark="local-bench", functions=[
        FeatureFunction(name="detector", source=source, description="Failure",
                        example_fail_ids=[iid], failure_localisations={iid: evidence},
                        scores={"evaluated": True, "n_items": 1,
                                "a": 1, "b": 0, "c": 0, "d": 0,
                                "recall": 1, "precision": 1, "ppv_at_pi0": 1,
                                "behavior_purity": .8, "behavior_specificity": .8,
                                "buckets": [{"bucket": "True", "n": 1, "n_fail": 1,
                                             "n_pass": 0, "example_fail_ids": [iid]}]}),
    ])
    run = tmp_path / "runs" / "saved"
    run.mkdir(parents=True)
    library.save(run / "library.json")
    (run / "config.json").write_text(json.dumps({"run_id": "saved",
                                                "reference_data": [reference]}))
    return Settings(data_dir=tmp_path / "server"), store, library, iid


# Local library.
LIBRARY = {
    "version": 1,
    "model_id": "example-model",
    "benchmark": "example-benchmark",
    "functions": [
        {
            "name": "repeated_error",
            "description": "Repeated error",
            "kind": "regex",
            "source": "def detect(item):\n    return False\n",
        }
    ],
}


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    settings = Settings(
        data_dir=tmp_path / "server", data_api_base_url="https://traces.example.test",
    )
    app = create_app(settings)
    with TestClient(app, base_url=settings.public_url) as browser:
        yield app, browser, settings
