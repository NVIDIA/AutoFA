# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local viewer and managed trace stores keep durable, port-independent provenance."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import detector_hash, passage
from autofa.data.dataset_identity import dataset_mixture_id
from autofa.data.local_import import import_local_run, reference_for_import
from autofa.data.parquet_store import Dataset, _portable_source_identity, make_item_id
from autofa.data.reference_data import (
    event_url,
    inspect_url,
    make_local_reference_data,
    normalize_reference_data,
    rollout_url,
)
from autofa.data.source_normalization import normalize_events
from autofa.data.sources import discover_eval_sources
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trace_navigation import TraceNavigation
from autofa.reports.failure_cards import _extract_autofa_data, build_data, write_failure_cards
from autofa_library_server.config import Settings
from autofa_library_server.main import create_app
from autofa_library_server.trace import LocalTablesSource
from evalrdl_trace import RolloutKey
from evalrdl_trace.conformance import check
from evalrdl_trace.tables import TablesSource

from support.library_server import RUN, STORE, TASK, TEXT, _base, _client, _write_store

pytestmark = pytest.mark.filterwarnings("ignore:Local import retained:UserWarning")

CLOUD = "https://runs.example.test"


@pytest.fixture
def trace_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    directory = _write_store(tmp_path)
    settings = Settings(data_dir=tmp_path / "server", data_api_base_url=CLOUD)
    app = create_app(settings)
    with _client(app, settings.public_url) as client:
        yield app, client, settings, directory


def test_managed_tables_source_conforms_and_full_text_survives_restart(trace_app):
    _, client, settings, directory = trace_app
    check(TablesSource(directory / "tables"), RolloutKey(RUN, TASK, 2))
    check(LocalTablesSource(directory / "tables"), RolloutKey(RUN, TASK, 2))
    assert client.get(f"/traces/{STORE}/").status_code == 200
    assert client.get("/trace-static/graph.js").status_code == 200
    assert client.get(_base() + "/edges").json()["kinds"] == {}
    assert client.get(f"/traces/{STORE}/api/v1/runs").json()[0]["run_id"] == RUN
    complete = client.get(_base() + "/events", params={"include": "parts"}).json()
    assert complete[0]["parts"][0]["text"] == TEXT
    events = client.get(_base() + "/events/preview").json()
    part = events[0]["parts"][0]
    assert part["text"] == TEXT[:4000] and part["text_truncated"] is True
    assert part["text_chars"] == len(TEXT)
    window = client.get(_base() + "/events/preview", params={
        "include": "parts", "offset": 1, "limit": 1,
    }).json()
    assert [event["lane"] for event in window] == ["model_call"]
    assert window[0]["parts"][0]["text"] == "aux"
    assert client.get(_base() + "/events/55/parts").status_code == 409
    preview = client.get(_base() + "/events/55/parts", params={"lane": "conversation"}).json()
    assert "TAIL EVIDENCE" not in preview["parts"][0]["text"]
    query = {"event_idx": 55, "part_idx": 7, "lane": "conversation"}
    full = client.get(_base() + "/text", params=query)
    assert "TAIL EVIDENCE" in full.text and "&lt;script&gt;unsafe()&lt;/script&gt;" in full.text
    assert "<script>unsafe()" not in full.text
    original = client.get(_base() + "/part-text", params=query)
    assert original.json() == {
        "run_id": RUN, "task_id": TASK, "repeat": 2, "agent_ref": "",
        "lane": "conversation", "event_idx": 55, "part_idx": 7,
        "text": TEXT, "chars": len(TEXT),
    }
    assert client.get(_base() + "/part-text", params={
        **query, "lane": "model_call",
    }).json()["text"] == "aux"
    with _client(create_app(settings), settings.public_url) as restarted:
        assert restarted.get(_base() + "/text", params=query).text == full.text
        assert restarted.get(_base() + "/part-text", params=query).json() == original.json()


def test_trace_route_boundaries_and_artifacts(trace_app):
    _, client, _, directory = trace_app
    assert client.get("/traces/not-a-store/").status_code == 404
    assert client.get(f"/traces/{'f' * 64}/").status_code == 404
    assert client.get(_base().replace(RUN, "wrong-run") + "/events").status_code == 404
    assert client.get(_base().replace("/2", "/bad") + "/events").status_code == 400
    assert client.get(_base() + "/events?lane=wrong").status_code == 400
    assert client.get(_base() + "/events/preview?limit=201").status_code == 400
    assert client.get(_base() + "/events", headers={
        "Origin": "https://foreign.test",
    }).status_code == 403
    assert client.get(_base() + "/events", headers={"Host": "foreign.test"}).status_code == 403
    image = client.get(_base() + "/artifacts/0/view")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
    download = client.get(_base() + "/artifacts/0/download")
    assert "attachment" in download.headers["content-disposition"]
    assert download.content == image.content
    assert client.get(_base() + "/artifacts/90/view").status_code == 404
    assert client.get(_base() + "/text?event_idx=55").status_code == 409
    assert client.get(_base() + "/part-text?event_idx=55").status_code == 400
    assert client.get(_base() + "/part-text?event_idx=55&part_idx=7").status_code == 409
    assert client.get(_base() + "/part-text", params={
        "event_idx": 55, "part_idx": 8, "lane": "conversation",
    }).status_code == 404
    assert client.get(_base() + "/text", params={
        "event_idx": 55, "lane": "conversation", "part_idx": 0,
    }).status_code == 404
    (directory / "tables" / "events.parquet").write_bytes(b"corrupt")
    assert client.get(_base() + "/events").status_code == 404


def test_same_run_id_in_two_snapshots_and_rollout_export(trace_app, tmp_path):
    _, client, _, _ = trace_app
    _write_store(tmp_path, "b" * 64, text="second immutable generation")
    response = client.get(_base("b" * 64) + "/text?event_idx=55&lane=conversation&part_idx=7")
    assert "second immutable generation" in response.text and "TAIL EVIDENCE" not in response.text
    response = client.get(_base() + "/export")
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "part_texts.parquet" in archive.namelist()
        frame = pl.read_parquet(io.BytesIO(archive.read("part_texts.parquet")))
        assert TEXT in frame["text"].to_list()


def test_native_encoded_task_collision_is_refused(trace_app, tmp_path):
    _, client, _, _ = trace_app
    _write_store(tmp_path, "c" * 64, collision=True)
    response = client.get(_base("c" * 64) + "/events")
    assert response.status_code == 409
    assert "Ambiguous" in response.json()["error"]


# Managed local provenance survives report storage without becoming a cloud URL.


def local_item():
    events = [
        {"event_idx": 4, "role": "user", "kind": "message", "lane": "conversation",
         "n_parts": 1, "parts": [{"part_idx": 0, "part_type": "text", "text": "Fix this"}]},
        {"event_idx": 55, "role": "tool", "kind": "tool_result", "lane": "conversation",
         "n_parts": 1, "parts": [{"part_idx": 7, "part_type": "text",
                                   "text": "🙂 Tests failed in the local run."}]},
    ]
    for event in events:
        for part in event["parts"]:
            part.update(lane=event["lane"], chars=len(part["text"]))
    value = normalize_events({"n_events": 2, "n_tool_calls": 0}, events, [],
                             conversation_lane="conversation")
    value["evidence_scope"].update(identity="evalrdl_tables_v1",
                                   source="local EvalRDL event parts and tool arguments")
    return {"model_id": "m", "benchmark": "b", "problem_id": "suite/task", "run_idx": 2,
            "is_correct": False, "prompt_text": "Fix this", "response_text": json.dumps(value)}


def test_local_references_reject_unmanaged_paths():
    assert normalize_reference_data([{"kind": "local_evalrdl", "store_id": "../secret",
                                       "run_id": "run"}]) == []
    with pytest.raises(ValueError):
        make_local_reference_data(store_id="../secret", run_id="run")


def test_local_urls_are_port_independent_and_keep_source_identity():
    ref = make_local_reference_data(store_id=STORE, run_id="run")
    base = rollout_url(ref, problem_id="suite/🙂 50%", run_idx=0)
    assert base == f"/traces/{STORE}/#/rollout/run/suite%252F%F0%9F%99%82%2050%25/0"
    assert inspect_url(ref) == f"/traces/{STORE}/#/run/run"
    assert event_url(base, {"event_idx": 55, "part_idx": 7, "lane": "worker/a"}).endswith(
        "/e55-p7?lane=worker%2Fa")
    for path in ("//host/traces/" + STORE + "/", "/other/", "/traces/../"):
        assert event_url(path + "#/rollout/r/t/0", {"event_idx": 0, "lane": ""}) == ""


def test_local_identity_marker_maps_passage_without_fabricating_fallback_ids():
    item = local_item()
    text = derive_trace_fields(item)["observations_text"]
    start = text.index("Tests failed")
    evidence = passage("observations_text", text, start, start + len("Tests failed"),
                       method="literal")
    location, = TraceNavigation(item).locate(evidence)
    assert location["source_event"] == {"event_idx": 55, "part_idx": 7, "lane": "conversation"}
    value = json.loads(item["response_text"])
    value["evidence_scope"]["trajectory_source"] = "raw fallback"
    item["response_text"] = json.dumps(value)
    location, = TraceNavigation(item).locate(evidence)
    assert "source_event" not in location
    # Auxiliary execution text must obey the same identity gate as conversation steps.
    raw = item["response_text"]
    start = raw.rindex("Tests failed")
    raw_evidence = passage("response_text", raw, start, start + len("Tests failed"),
                           method="literal")
    assert not any(loc.get("source_event") for loc in TraceNavigation(item).locate(raw_evidence))


def test_local_reports_keep_full_evidence_and_relative_links_in_spilled_assets(tmp_path):
    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([local_item()]), path)
    dataset = Dataset([path])
    item = next(dataset.iter_batches())[0]
    iid = item["_item_id"]
    text = derive_trace_fields(item)["observations_text"]
    start = text.index("Tests failed")
    saved = {"detector_hash": detector_hash("def check(item): return True"),
             "passages": [passage("observations_text", text, start, start + 12, method="literal")]}
    library = FunctionLibrary(model_id="m", benchmark="b", functions=[FeatureFunction(
        name="check", source="def check(item): return True", description="failure",
        example_fail_ids=[iid], failure_localisations={iid: saved},
    )])
    refs = [make_local_reference_data(store_id=STORE, run_id="run", parquet_path=path)]
    data = build_data(library, dataset=dataset, reference_data=refs, prefer_source_links=True)
    assert not data["items"][iid].get("source_only")
    assert "Tests failed" in data["items"][iid]["response_raw"]
    location, = data["functions"][0]["failure_localisations"][iid]["passages"][0]["trace_locations"]
    assert location["source_url"] == (
        f"/traces/{STORE}/#/rollout/run/suite%252Ftask/2/e55-p7?lane=conversation"
        "&text=Tests%20failed"
    )
    assert "parquet_path" not in data["meta"]["reference_data"][0]
    report = write_failure_cards(library, out=tmp_path / "failure_cards.html", dataset=dataset,
                                 reference_data=refs, prefer_source_links=True)
    payload = _extract_autofa_data(report.read_text())
    asset = tmp_path / payload["meta"]["items_base"] / payload["item_index"][iid]
    assert json.loads(asset.read_text())["source_data_url"].startswith(f"/traces/{STORE}/")


def test_local_source_identity_survives_profile_moves_and_distinguishes_runs(tmp_path):
    item = local_item()
    paths = [tmp_path / "first" / "dataset.parquet", tmp_path / "moved" / "dataset.parquet"]
    refs = []
    for path in paths:
        path.parent.mkdir()
        reference = make_local_reference_data(store_id=STORE, run_id="run", parquet_path=path)
        (path.parent / "source.json").write_text(json.dumps(reference))
        refs.append(reference)
    assert make_item_id(item, paths[0], include_source=True) == make_item_id(
        item, paths[1], include_source=True)
    names = [dataset_mixture_id([str(path)], model_id="*", benchmarks=[], reference_data=[ref])
             for path, ref in zip(paths, refs, strict=True)]
    assert names[0] == names[1]
    changed = {**refs[0], "store_id": "b" * 64}
    assert dataset_mixture_id([str(paths[0])], model_id="*", benchmarks=[],
                              reference_data=[changed]) != names[0]


@pytest.mark.parametrize("bound", [False, True])
def test_mixed_sources_never_inherit_an_unbound_local_store(tmp_path, bound):
    from autofa.reports.failure_cards import _load_example_items

    paths = [tmp_path / "first.parquet", tmp_path / "unrelated.parquet"]
    for index, path in enumerate(paths):
        row = {**local_item(), "problem_id": f"task-{index}"}
        pq.write_table(pa.Table.from_pylist([row]), path)
    dataset = Dataset(paths)
    items = [row for batch in dataset.iter_batches() for row in batch]
    reference = make_local_reference_data(
        store_id=STORE, run_id="source-run", parquet_path=paths[0] if bound else None,
    )
    loaded = _load_example_items(
        dataset, [row["_item_id"] for row in items], reference_data=[reference],
        prefer_source_links=True,
    )
    assert loaded[items[1]["_item_id"]]["source_data_url"] == ""
    assert bool(loaded[items[0]["_item_id"]]["source_data_url"]) is bound


def test_parent_popup_bridge_uses_current_origin_and_rejects_untrusted_targets():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to exercise the browser message handler")
    script = Path(__file__).parents[1] / "src/autofa_library_server/static/app.js"
    harness = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const opened = [], listeners = {};
const frame = {contentWindow: {}, dataset: {localTraces: 'true',
  dataApiBase: 'https://cloud.example/data'}, addEventListener() {}};
const document = {querySelector() {return null},
  querySelectorAll(s) {return s === 'iframe[data-failure-bridge]' ? [frame] : []},
  createElement() {return {click() {opened.push(this.href)}}}};
const window = {location: {origin: 'http://127.0.0.1:9123', href: 'http://127.0.0.1:9123/report'},
  addEventListener(type, fn) {listeners[type] = fn}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {window, document, URL});
const store = 'a'.repeat(64);
const local = `/traces/${store}/#/rollout/run/suite%252Ftask/2/e55-p7?lane=worker%2Fa`;
const send = (url, source = frame.contentWindow, extra = {}) => listeners.message({source,
  data: {namespace: 'autofa.source-data', type: 'open', url, ...extra}});
send(local);
assert.deepEqual(opened, ['http://127.0.0.1:9123' + local]);
send('https://cloud.example/data/#/rollout/run/task/2');
assert.equal(opened.length, 2);
for (const url of [
 'https://evil.example' + local, '//evil.example' + local,
 'http://127.0.0.1:9000' + local, '/api/token#/rollout/run/task/0',
 `/traces/${store}/?bad=1#/rollout/run/task/0`,
 `/traces/${store}/#/run/run`, `/traces/../secret/#/rollout/run/task/0`,
 `/traces/${store}/#/rollout/run/task/0/e1?lane=x&redirect=evil`,
 'javascript:alert(1)',
]) send(url);
send(local, {});
send(local, frame.contentWindow, {extra: 'ignored'});
assert.equal(opened.length, 2);
"""
    subprocess.run([node, "-e", harness, str(script)], check=True, capture_output=True, text=True)


# Managed provenance survives adjunct loss without changing immutable snapshots.


@pytest.fixture
def imported_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    run = tmp_path / "gym"
    run.mkdir()
    (run / "rollouts.jsonl").write_text(json.dumps({
        "_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 0.0,
        "responses_create_params": {
            "model": "model", "input": [{"role": "user", "content": "Read the file"}],
        },
        "response": {"status": "completed", "output": [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Unable to finish"}],
        }]},
    }) + "\n")
    return run, import_local_run(run)


def _immutable_contents(store):
    paths = [store / "manifest.json", store / "dataset.parquet", *(store / "tables").iterdir()]
    return {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize("operation", ["reference", "forced", "directory"])
def test_missing_sidecar_recovers_without_replacing_snapshot(imported_run, operation):
    run, result = imported_run
    store = result.parquet_path.parent
    before = _immutable_contents(store)
    (store / "source.json").unlink()

    if operation == "reference":
        assert reference_for_import(result.parquet_path) == result.reference_data()
    elif operation == "directory":
        # Selecting the store folder must find its snapshot, not the tables' parquet files.
        assert [source.path for source in discover_eval_sources(store)] == [result.parquet_path]
    else:
        assert import_local_run(run, force=True) == result

    assert json.loads((store / "source.json").read_text()) == result.reference_data()
    assert _immutable_contents(store) == before
    assert not list(store.glob(".source-*"))


def test_saved_reference_reopens_without_sidecar_or_original_logs(imported_run, monkeypatch):
    from autofa.wizard import dataset

    run, result = imported_run
    row = {"model_id": "model", "benchmark": "bench", "problem_id": "task", "run_idx": 0}
    item_id = make_item_id(row, result.parquet_path, include_source=True)
    (result.parquet_path.parent / "source.json").unlink()
    shutil.rmtree(run)
    _portable_source_identity.cache_clear()  # Exercise reopening in a new process.
    assert make_item_id(row, result.parquet_path, include_source=True) == item_id
    monkeypatch.setattr(dataset, "_finish_selection", lambda paths, **kwargs: (paths, kwargs))
    paths, options = dataset.reuse_reference_data([result.reference_data()])
    assert paths == [result.parquet_path]
    assert options["reference_data"] == [result.reference_data()]


def test_concurrent_sidecar_recovery_keeps_complete_reference(imported_run):
    _, result = imported_run
    store = result.parquet_path.parent
    (store / "source.json").unlink()
    with ThreadPoolExecutor(max_workers=4) as pool:
        references = list(pool.map(reference_for_import, [result.parquet_path] * 8))
    assert references == [result.reference_data()] * 8
    assert json.loads((store / "source.json").read_text()) == result.reference_data()
    assert not list(store.glob(".source-*"))


@pytest.mark.parametrize("damage", ["malformed", "run_id", "symlink"])
def test_existing_bad_sidecar_is_never_repaired_or_ignored(imported_run, damage):
    run, result = imported_run
    sidecar = result.parquet_path.parent / "source.json"
    if damage == "malformed":
        sidecar.write_text("{broken")
    elif damage == "symlink":
        target = sidecar.with_name("other-reference.json")
        sidecar.rename(target)
        sidecar.symlink_to(target)
    else:
        reference = result.reference_data()
        reference[damage] = "wrong"
        sidecar.write_text(json.dumps(reference))
    before = sidecar.read_bytes()
    for operation in (
        lambda: reference_for_import(result.parquet_path),
        lambda: import_local_run(run),
        lambda: import_local_run(run, force=True),
    ):
        with pytest.raises(ValueError, match="reference"):
            operation()
        assert sidecar.read_bytes() == before
    assert sidecar.is_symlink() == (damage == "symlink")


def test_moved_profile_sidecar_path_is_not_treated_as_source_identity(imported_run):
    _, result = imported_run
    sidecar = result.parquet_path.parent / "source.json"
    reference = result.reference_data()
    reference["parquet_path"] = "/previous/profile/dataset.parquet"
    sidecar.write_text(json.dumps(reference))
    assert reference_for_import(result.parquet_path) == result.reference_data()
    assert json.loads(sidecar.read_text()) == reference


@pytest.mark.parametrize("force", [False, True])
def test_corrupt_snapshot_requires_explicit_recovery_and_is_not_replaced(imported_run, force):
    run, result = imported_run
    store = result.parquet_path.parent
    (store / "source.json").unlink()
    damaged = store / "tables" / "part_texts.parquet"
    damaged.write_bytes(b"damaged")
    before = _immutable_contents(store)
    with pytest.raises(ValueError, match="checksum.*backup.*AUTOFA_HOME.*force=True"):
        import_local_run(run, force=force)
    with pytest.raises(ValueError, match="checksum"):
        reference_for_import(result.parquet_path)
    assert _immutable_contents(store) == before
    assert not (store / "source.json").exists()
    assert not list(store.parent.glob(".*"))


def test_missing_sidecar_does_not_bind_unrelated_parquet(imported_run, tmp_path):
    _, result = imported_run
    (result.parquet_path.parent / "source.json").unlink()
    sibling = result.parquet_path.with_name("unrelated.parquet")
    shutil.copyfile(result.parquet_path, sibling)
    external = tmp_path / "dataset.parquet"
    shutil.copyfile(result.parquet_path, external)
    assert reference_for_import(sibling) is None
    assert reference_for_import(external) is None
    assert not (result.parquet_path.parent / "source.json").exists()
