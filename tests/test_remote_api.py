# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Data API client, export cache and run picker, and local exports' source recovery."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import ssl
import urllib.error
import urllib.parse
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from autofa.config import Config
from autofa.data import local_source
from autofa.data.gym_adapter import materialize_eval_source
from autofa.data.reference_data import public_reference_data
from autofa.data.remote_api import (
    REMOTE_EXPORT_CACHE_VERSION,
    DataApiClient,
    DataApiError,
    _retry_delay,
    current_remote_export,
    materialize_remote_run,
    remote_cache_dir,
    remote_export_is_current,
)
from autofa.data.sources import EvalSource
from autofa.session.guidance import guidance_prompt
from autofa.ui import rollout_table
from autofa.ui.rollout_table import (
    _RolloutTableState,
    filter_rollout_runs,
    index_rollout_runs,
    select_rollout_runs,
)
from autofa.wizard import remote_runs
from autofa.wizard.dataset import _materialize_sources
from autofa.wizard.remote_runs import RemoteRunDownload

from support.traces import UrlResponse

API = "https://example.test"
RUN = {"base_url": API, "catalog": {"run_id": "run"}}


def _parquet_bytes(rows: list[dict] | None = None) -> bytes:
    rows = rows or [{"problem_id": "t1", "run_idx": 0, "response_text": "r"}]
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def _trace_bytes() -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("manifest.json", '{"exporter":"trace"}')
    return out.getvalue()


def _urlopen(monkeypatch, respond):
    """Route urllib through ``respond(req, n)``; record requests and retry sleeps."""
    calls, sleeps = [], []

    def urlopen(req, timeout=None):
        calls.append(req)
        return respond(req, len(calls))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", sleeps.append)
    return calls, sleeps


def _raises(exc):
    def respond(*args, **kwargs):
        raise exc

    return respond


class _Interrupted(UrlResponse):
    """Serve the body, then end it early (``error=None``) or fail the next read."""

    def __init__(self, body, error, **kwargs):
        super().__init__(body, **kwargs)
        self._error = error

    def read(self, n=-1):
        if self._body or self._error is None:
            return super().read(n)
        raise self._error


def test_catalog_requests_keep_lane_and_drop_empty_filters(monkeypatch):
    client = DataApiClient(API + "/")
    # An empty lane is meaningful (every lane); other empty filters are omitted.
    assert client._url("/v1/events", {"lane": "", "include": "parts"}) == (
        f"{API}/v1/events?lane=&include=parts"
    )
    payloads = {
        f"{API}/v1/meta": {"n_runs": 1},
        f"{API}/v1/runs?benchmark=swe": [{"run_id": "abc"}, "not-a-run"],
    }
    _urlopen(monkeypatch, lambda req, n: UrlResponse(json.dumps(payloads[req.full_url]).encode()))
    assert client.meta() == {"n_runs": 1}
    assert client.list_runs(benchmark="swe", harness="") == [{"run_id": "abc"}]


def test_artifact_text_hydration_is_bound_to_rollout_identity_and_bytes(monkeypatch):
    text = "verified failure"
    encoded = text.encode()
    expected_url = "/v1/rollouts/run/task/0/artifacts/2/text"
    artifact = {
        "artifact_idx": 2,
        "text_url": expected_url,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }
    client = DataApiClient(API)
    monkeypatch.setattr(client, "_get_json", lambda path: {"text": text})

    [hydrated] = client.get_rollout_artifact_texts(
        [artifact], run_id="run", task_id="task", repeat=0,
    )
    assert hydrated["text"] == text and "text_error" not in hydrated

    for changed in (
        {
            **artifact,
            "text": "unverified metadata text",
            "text_url": "/v1/rollouts/other/task/0/artifacts/2/text",
        },
        {**artifact, "text": "unverified metadata text", "sha256": "0" * 64},
        {
            **artifact,
            "text": "unverified metadata text",
            "size_bytes": len(encoded) + 1,
        },
    ):
        [rejected] = client.get_rollout_artifact_texts(
            [changed], run_id="run", task_id="task", repeat=0,
        )
        assert "text" not in rejected and "text_error" in rejected


def test_export_retries_timeout_and_busy_service(tmp_path, monkeypatch):
    """Reproduce the overnight timeout followed by the service's busy-export 503."""
    payload = _parquet_bytes()

    def respond(req, n):
        if n == 1:
            raise TimeoutError("The read operation timed out")
        if n == 2:
            raise urllib.error.HTTPError(
                req.full_url, 503, "busy", {"Retry-After": "17"}, io.BytesIO(b"busy"),
            )
        return UrlResponse(payload, headers={"Content-Length": str(len(payload))})

    calls, sleeps = _urlopen(monkeypatch, respond)
    dest = tmp_path / "dataset.parquet"
    DataApiClient(API).download_autofa_export("run", dest)
    assert len(calls) == 3 and len({req.full_url for req in calls}) == 1
    assert sleeps == [5, 17]
    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".parquet.partial").exists()


def test_export_restarts_after_body_shorter_than_content_length(tmp_path, monkeypatch):
    payload = _parquet_bytes()
    dest = tmp_path / "dataset.parquet"
    headers = {"Content-Length": str(len(payload))}

    def respond(req, n):
        assert not dest.with_suffix(".parquet.partial").exists()
        if n == 1:
            return _Interrupted(payload[:10], None, headers=headers)
        return UrlResponse(payload, headers=headers)

    calls, _ = _urlopen(monkeypatch, respond)
    DataApiClient(API).download_autofa_export("run", dest)
    assert dest.read_bytes() == payload
    assert len(calls) == 2


def test_export_retry_exhaustion_preserves_existing_cache(tmp_path, monkeypatch):
    dest = tmp_path / "dataset.parquet"
    dest.write_bytes(b"existing cache")
    calls, sleeps = _urlopen(
        monkeypatch, lambda req, n: _Interrupted(b"partial", TimeoutError("read timed out")),
    )
    with pytest.raises(DataApiError, match="read timed out") as error:
        DataApiClient(API, max_attempts=3).download_autofa_export("run", dest)
    assert isinstance(error.value.__cause__, TimeoutError)
    assert len(calls) == 3 and sleeps == [5, 10]
    assert dest.read_bytes() == b"existing cache"
    assert not dest.with_suffix(".parquet.partial").exists()


@pytest.mark.parametrize("respond,error", [
    (_raises(urllib.error.HTTPError(API, 404, "missing", {}, io.BytesIO(b"x"))), DataApiError),
    (lambda req, n: _Interrupted(b"partial", KeyboardInterrupt()), KeyboardInterrupt),
], ids=["permanent_http_error", "cancelled"])
def test_download_stops_without_retry_or_partial(tmp_path, monkeypatch, respond, error):
    calls, sleeps = _urlopen(monkeypatch, respond)
    dest = tmp_path / "dataset.parquet"
    with pytest.raises(error):
        DataApiClient(API).download_autofa_export("run", dest)
    assert len(calls) == 1 and sleeps == []
    assert not dest.exists()
    assert not dest.with_suffix(".parquet.partial").exists()


@pytest.mark.parametrize("respond,match", [
    (lambda req, n: UrlResponse(b"{broken JSON"), "Invalid JSON"),
    (_raises(urllib.error.URLError(ssl.SSLCertVerificationError("bad cert"))), "bad cert"),
], ids=["invalid_json", "bad_certificate"])
def test_unrecoverable_json_failures_are_not_retried(monkeypatch, respond, match):
    calls, sleeps = _urlopen(monkeypatch, respond)
    with pytest.raises(DataApiError, match=match):
        DataApiClient(API).get_run("abc")
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("retry_after,delay", [("999", 60), ("bad", 5)])
def test_retry_after_is_bounded(retry_after, delay):
    assert _retry_delay(0, retry_after) == delay


def test_trace_export_streams_without_content_length_and_retries(tmp_path, monkeypatch):
    body = _trace_bytes()

    def respond(req, n):
        if n == 1:
            return _Interrupted(body[:20], http.client.IncompleteRead(b"", 1))
        return UrlResponse(body)

    calls, _ = _urlopen(monkeypatch, respond)
    path = tmp_path / "trace.zip"
    DataApiClient(API).download_trace_export("run/alias", path)
    assert len(calls) == 2
    assert calls[0].full_url == f"{API}/v1/runs/run%2Falias/export/trace"
    assert calls[0].get_header("Accept") == "application/zip,*/*"
    assert path.read_bytes() == body
    assert not path.with_suffix(".zip.partial").exists()


@pytest.mark.parametrize("name,body,match", [
    ("autofa", _parquet_bytes()[:-10], "Invalid"),
    ("trace", b'{"error":"failed"}', "Invalid"),
    ("autofa", b"", "Empty"),
], ids=["truncated_parquet", "trace_not_zip", "empty"])
def test_invalid_export_never_replaces_cache(tmp_path, monkeypatch, name, body, match):
    path = tmp_path / "export"
    path.write_bytes(b"previous complete download")
    _urlopen(monkeypatch, lambda req, n: UrlResponse(body))
    with pytest.raises(DataApiError, match=match):
        getattr(DataApiClient(API), f"download_{name}_export")("run", path)
    assert path.read_bytes() == b"previous complete download"
    assert not path.with_suffix(".partial").exists()


def test_trace_subset_encodes_repeated_rollout_selectors(tmp_path, monkeypatch):
    calls, _ = _urlopen(monkeypatch, lambda req, n: UrlResponse(_trace_bytes()))
    client = DataApiClient(API)
    client.download_trace_export(
        "run", tmp_path / "trace.zip", rollouts=[("a/b:c", 0), ("space task", 2)],
    )
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0].full_url).query) == {
        "rollout": ["a/b:c:0", "space task:2"],
    }
    with pytest.raises(ValueError, match="1 and 500"):
        client.download_trace_export("run", tmp_path / "trace.zip", rollouts=[])


@pytest.mark.parametrize("sidecar", [None, {"export_cache_version": 1}],
                         ids=["no_sidecar", "old_version"])
def test_legacy_exports_redownload_instead_of_relabeling_cache(tmp_path, monkeypatch, sidecar):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    root = remote_cache_dir("run", base_url=API)
    legacy, legacy_meta = root / "dataset.parquet", root / "source.json"
    original = _parquet_bytes()
    legacy.write_bytes(original)
    if sidecar is not None:
        legacy_meta.write_text(json.dumps(sidecar))
    fresh = _parquet_bytes([{"prompt_text": "fresh"}])
    calls, _ = _urlopen(monkeypatch, lambda req, n: UrlResponse(fresh))
    path = materialize_remote_run("run", **RUN)
    assert [req.full_url for req in calls] == [f"{API}/v1/runs/run/export/autofa"]
    assert path != legacy and path.read_bytes() == fresh
    assert legacy.read_bytes() == original
    assert (json.loads(legacy_meta.read_text()) if legacy_meta.exists() else None) == sidecar
    assert remote_export_is_current(path, run_id="run", base_url=API)
    metadata = json.loads((path.parent / "source.json").read_text())
    assert metadata["export_cache_version"] == REMOTE_EXPORT_CACHE_VERSION
    assert materialize_remote_run("run", **RUN) == path
    assert len(calls) == 1


def test_refresh_preserves_history_and_failed_refresh_preserves_current(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    bodies = [_parquet_bytes([{"prompt_text": "original"}]),
              _parquet_bytes([{"prompt_text": "updated"}]), b"broken"]
    _urlopen(monkeypatch, lambda req, n: UrlResponse(bodies[n - 1]))
    saved = materialize_remote_run("run", **RUN)
    saved_meta = (saved.parent / "source.json").read_bytes()
    refreshed = materialize_remote_run("run", force=True, **RUN)
    assert pq.read_table(saved).to_pylist() == [{"prompt_text": "original"}]
    assert (saved.parent / "source.json").read_bytes() == saved_meta
    assert pq.read_table(refreshed).to_pylist() == [{"prompt_text": "updated"}]
    with pytest.raises(DataApiError, match="Invalid autofa export"):
        materialize_remote_run("run", force=True, **RUN)
    assert current_remote_export("run", base_url=API) == refreshed
    assert materialize_remote_run("run", **RUN) == refreshed
    assert sorted(saved.parent.parent.iterdir()) == sorted([saved.parent, refreshed.parent])
    # A damaged current generation requires a new download, not a fallback.
    refreshed.write_bytes(b"damaged")
    assert current_remote_export("run", base_url=API) is None


def test_uuid_and_canonical_run_share_export_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))

    def respond(req, n):
        if req.full_url.endswith("/export/autofa"):
            return UrlResponse(_parquet_bytes())
        return UrlResponse(b'{"run_id":"canonical","run_uuid":"uuid"}')

    calls, _ = _urlopen(monkeypatch, respond)
    path = materialize_remote_run("uuid", base_url=API)
    assert materialize_remote_run("canonical", base_url=API) == path
    assert materialize_remote_run("uuid", base_url=API) == path
    downloads = [req.full_url for req in calls if req.full_url.endswith("/export/autofa")]
    assert downloads == [f"{API}/v1/runs/canonical/export/autofa"]
    assert json.loads((path.parent / "source.json").read_text())["catalog"]["run_uuid"] == "uuid"


def test_known_canonical_identity_takes_precedence_over_an_old_alias_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.data.remote_api.current_remote_export", lambda rid, **kw: (
        tmp_path / ("current.parquet" if rid == "canonical" else "old-alias.parquet")
    ))
    result = materialize_remote_run(
        "uuid", base_url=API, catalog={"run_id": "canonical", "run_uuid": "uuid"},
    )
    assert result == tmp_path / "current.parquet"


def test_failed_generation_publication_keeps_saved_export(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    _urlopen(monkeypatch, lambda req, n: UrlResponse(_parquet_bytes()))
    saved = materialize_remote_run("run", **RUN)
    saved_bytes = saved.read_bytes()
    saved_source = (saved.parent / "source.json").read_bytes()
    cache = remote_cache_dir("run", base_url=API)
    pointer = (cache / "current.json").read_bytes()
    replace = Path.replace

    # The download and provenance completed; only moving the current pointer fails.
    def fail_publication(path, target):
        if Path(target) == cache / "current.json":
            raise OSError("simulated disk failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    with pytest.raises(OSError, match="simulated disk failure"):
        materialize_remote_run("run", force=True, **RUN)
    assert current_remote_export("run", base_url=API) == saved
    assert saved.read_bytes() == saved_bytes
    assert (saved.parent / "source.json").read_bytes() == saved_source
    assert (cache / "current.json").read_bytes() == pointer
    assert list((cache / "exports").iterdir()) == [saved.parent]
    assert not list(cache.glob("*.partial"))


def test_interruption_after_publication_keeps_new_generation_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    bodies = [_parquet_bytes([{"prompt_text": "original"}]),
              _parquet_bytes([{"prompt_text": "refreshed"}])]
    _urlopen(monkeypatch, lambda req, n: UrlResponse(bodies[n - 1]))
    saved = materialize_remote_run("run", **RUN)
    cache = remote_cache_dir("run", base_url=API)
    replace = Path.replace

    def interrupted_publication(path, target):
        result = replace(path, target)
        if Path(target) == cache / "current.json":
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(Path, "replace", interrupted_publication)
    with pytest.raises(KeyboardInterrupt):
        materialize_remote_run("run", force=True, **RUN)
    current = current_remote_export("run", base_url=API)
    assert current is not None and current != saved
    assert pq.read_table(current).to_pylist() == [{"prompt_text": "refreshed"}]
    assert pq.read_table(saved).to_pylist() == [{"prompt_text": "original"}]
    assert materialize_remote_run("run", **RUN) == current
    assert not list(cache.glob("*.partial"))


# The interactive remote rollout-set picker.


def _runs() -> list[dict]:
    return [
        {"run_id": "older-id", "run_ts": "2026-08-01T10:00:00", "model": "Nano 3.5",
         "benchmark": "hle", "harness": "nemo-skills", "n_rollouts": 1234,
         "source_user": "alice", "label": "baseline"},
        {"run_id": "newer-id", "run_ts": "2026-09-02T11:00:00", "model": "Nemotron 3.5 Super",
         "benchmark": "swebench-verified", "harness": "nemo-gym", "n_rollouts": 2500,
         "source_user": "bob", "label": "Super Candidate"},
        {"run_id": "ingested-date-id", "run_ts": None,
         "ingested_at": "2026-08-15T09:30:00+00:00", "model": None,
         "benchmark": "terminal-bench", "harness": "harbor", "n_rollouts": None,
         "source_user": None, "label": None},
    ]


def _chosen(state) -> set[str]:
    return {run["run_id"] for run in state.chosen()}


@pytest.mark.parametrize("query,expected,error", [
    ("ALICE", {"older-id"}, ""),
    ("model:nemotron bench:swe harness:gym", {"newer-id"}, ""),
    ("model:nemotron user:alice", set(), ""),
    ("model:nano owner:bob", {"older-id"}, "Unknown field: owner"),
], ids=["plain_text_any_column", "field_aliases", "fields_and", "unknown_field_kept_terms"])
def test_rollout_filter_queries(query, expected, error):
    result = filter_rollout_runs(index_rollout_runs(_runs()), query)
    assert {row.run_id for row in result.rows} == expected
    assert result.error == error


def test_only_checked_rows_are_chosen_across_filters():
    state = _RolloutTableState(index_rollout_runs(_runs()))
    state.move(1)
    assert state.chosen() == []
    state.apply_query("bench:swebench")
    state.toggle_visible()
    state.apply_query("user:alice")
    state.toggle_visible()
    assert _chosen(state) == {"newer-id", "older-id"}
    state.toggle_visible()
    assert _chosen(state) == {"newer-id"}
    state.apply_query("no-matches")
    state.toggle_visible()
    assert _chosen(state) == {"newer-id"}


@pytest.mark.parametrize("keys,expected", [
    # Enter without checks must stay in the picker. Then check two rows.
    ("\t\r \x1b[B \r", {"newer-id", "ingested-date-id"}),
    ("\ta\r", {"older-id", "newer-id", "ingested-date-id"}),
], ids=["enter_without_checks_stays", "select_all"])
def test_keyboard_selection_returns_only_checked_runs(monkeypatch, keys, expected):
    with create_pipe_input() as pipe:
        monkeypatch.setattr(
            rollout_table, "Application",
            lambda **kwargs: Application(input=pipe, output=DummyOutput(), **kwargs),
        )
        pipe.send_text(keys)
        selected = select_rollout_runs(_runs(), clear_screen=False)
    assert {run["run_id"] for run in selected} == expected


class _FakeApi:
    base_url = API

    def __init__(self):
        self.list_calls = 0

    def list_runs(self):
        self.list_calls += 1
        return [{"run_id": "run-a", "model": "super-a"}, {"run_id": "run-b", "model": "super-b"}]


def _stub_download(monkeypatch, tmp_path, *, calls=None, **config):
    """No current cached export; the compact download is recorded and returned."""
    compact = tmp_path / "dataset.parquet"
    compact.write_bytes(b"placeholder")
    monkeypatch.setattr(remote_runs, "current_remote_export", lambda *a, **kw: None)
    monkeypatch.setattr(remote_runs, "load_config", lambda: Config(**config))
    monkeypatch.setattr(remote_runs, "materialize_remote_run", lambda rid, **kw: (
        calls.append((rid, kw)) if calls is not None else None) or compact)
    return compact


@pytest.mark.parametrize("pick,expected", [
    (lambda runs: [runs[1], runs[0]], ["run-b", "run-a"]),
    (lambda runs: None, None),
], ids=["selected_rows_in_order", "cancel"])
def test_picker_lists_catalog_once_and_downloads_only_selected_rows(monkeypatch, pick, expected):
    api = _FakeApi()
    downloaded: list[str] = []
    monkeypatch.setattr(remote_runs, "prepare_step", lambda: None)
    monkeypatch.setattr(remote_runs, "select_rollout_runs", pick)

    def fake_download(api, run_id, catalog, *, force=False, efficiency_only=False):
        assert not force and not efficiency_only
        downloaded.append(run_id)
        return RemoteRunDownload(parquet_path=Path(run_id), export_url="", run_id=run_id)

    monkeypatch.setattr(remote_runs, "_download_one", fake_download)
    result = remote_runs._interactive_pick_and_download(api)
    assert api.list_calls == 1
    assert downloaded == (expected or [])
    if expected is None:
        assert result is None
    else:
        assert [item.run_id for item in result] == expected


@pytest.mark.parametrize("full", [True, False], ids=["full_traces", "summary_only"])
def test_download_visibility_survives_run_provenance(tmp_path, monkeypatch, full):
    compact = _stub_download(monkeypatch, tmp_path, remote_full_traces=full)
    source = tmp_path / "source" / "dataset.parquet"
    source.parent.mkdir()
    (source.parent / "manifest.json").write_text(json.dumps({
        "n_rows": 89, "n_events": 7000, "n_tool_calls": 2500,
        "limitations": {"encrypted reasoning unavailable": 3}, "agent_refs": ["hermes"],
        "notices": {"changed context retained separately": 17},
    }))
    calls = []
    monkeypatch.setattr(remote_runs, "materialize_source_traces",
                        lambda *args, **kwargs: calls.append(True) or source)
    result = remote_runs._download_one(_FakeApi(), "run-a", {"harness": "nemo-gym"})
    assert calls == ([True] if full else [])
    assert result.parquet_path == (source if full else compact)
    references = public_reference_data([result.reference_data()])
    assert "parquet_path" not in references[0]
    scope = references[0]["evidence_scope"]
    assert ("2500 tool calls" if full else "Summary-only") in scope
    if full:
        assert "Evidence limitation: encrypted reasoning unavailable: 3 occurrences" in scope
        assert "changed context retained separately: 17 model requests" in scope
    prompt = guidance_prompt(SimpleNamespace(analysis_focus="model", reference_data=references))
    assert scope in prompt


def test_full_trace_failure_never_falls_back_silently(tmp_path, monkeypatch):
    _stub_download(monkeypatch, tmp_path)
    monkeypatch.setattr(remote_runs, "materialize_source_traces",
                        _raises(DataApiError("Missing source parts")))
    with pytest.raises(SystemExit, match="No analysis was started"):
        remote_runs._download_one(_FakeApi(), "run-a", {})


def test_efficiency_download_skips_raw_source_recovery(tmp_path, monkeypatch):
    compact = _stub_download(monkeypatch, tmp_path, remote_full_traces=True)
    calls = []
    monkeypatch.setattr(remote_runs, "materialize_efficiency_tables",
                        lambda path, **kwargs: calls.append(kwargs) or path)
    monkeypatch.setattr(remote_runs, "materialize_source_traces",
                        lambda *a, **kw: pytest.fail("raw source recovery was called"))
    result = remote_runs._download_one(
        _FakeApi(), "run-a", {"harness": "nemo-gym"}, efficiency_only=True,
    )
    assert result.parquet_path == compact
    assert len(calls) == 1
    assert "Efficiency tables only" in result.evidence_scope


def test_download_uses_canonical_run_for_provenance_and_passes_explicit_refresh(
    tmp_path, monkeypatch,
):
    calls = []
    _stub_download(monkeypatch, tmp_path, calls=calls, remote_full_traces=False)
    result = remote_runs._download_one(
        _FakeApi(), "uuid", {"run_id": "canonical", "run_uuid": "uuid"}, force=True,
    )
    (run_id, options), = calls
    assert run_id == "canonical" and options["force"] is True
    assert result.run_id == "canonical"
    assert result.export_url == f"{API}/v1/runs/canonical/export/autofa"


# Local rollout exports enter Data API source recovery only with a known identity.


@pytest.fixture
def source_env(tmp_path, monkeypatch):
    cfg = SimpleNamespace(remote_full_traces=True, data_api_base_url="https://api.test")
    monkeypatch.setattr(local_source, "load_config", lambda: cfg)
    monkeypatch.setattr(local_source, "converted_root", lambda: tmp_path / "cache")
    monkeypatch.setattr("urllib.request.urlopen", _raises(AssertionError("network contacted")))
    return cfg


def _compact(path, **columns):
    row = {
        "model_id": "model", "benchmark": "bench", "problem_id": "170", "run_idx": 2,
        "is_correct": False, "reward": 0.0, "prompt_text": "Question",
        "response_text": "Final answer", "reasoning_text": "capped",
    }
    pq.write_table(pa.Table.from_pylist([{**row, **columns}]), path)
    return path


def test_local_rollout_export_enters_source_recovery(tmp_path, monkeypatch, source_env):
    path = _compact(tmp_path / "b067150ff320809c_170_2_autofa.parquet")
    original = path.read_bytes()
    calls = []

    def recover(export, *, api, run_id):
        calls.append((export, api.base_url, run_id))
        assert export.read_bytes() == original
        assert json.loads((export.parent / "source.json").read_text())["run_id"] == run_id
        return export.parent / "recovered.parquet"

    monkeypatch.setattr(local_source, "materialize_source_traces", recover)
    result = materialize_eval_source(EvalSource(kind="parquet", path=path))
    assert result.name == "recovered.parquet"
    assert calls[0][1:] == ("https://api.test", "b067150ff320809c")
    assert calls[0][0].parent != path.parent
    assert path.read_bytes() == original
    assert _materialize_sources([EvalSource(kind="parquet", path=path)]) == [result]


@pytest.mark.parametrize("name,sidecar,columns,match", [
    ("renamed.parquet", None, {"system_text": "instructions", "lane": "conversation"},
     "no source run identity"),
    ("b067150ff320809c_wrong_2_autofa.parquet", None, {}, "task/repeat"),
    ("data.parquet", {"run_id": "run", "base_url": "https://other.test"}, {},
     "different Data API"),
], ids=["newer_columns_without_identity", "filename_task_mismatch", "sidecar_other_api"])
def test_untrusted_local_identity_fails_before_network(
    tmp_path, source_env, name, sidecar, columns, match,
):
    path = _compact(tmp_path / name, **columns)
    if sidecar:
        path.with_suffix(".source.json").write_text(json.dumps(sidecar))
    with pytest.raises(DataApiError, match=match):
        local_source.materialize_local_parquet(path)


def test_renamed_export_uses_per_file_sidecar(tmp_path, monkeypatch, source_env):
    path = _compact(tmp_path / "renamed.parquet")
    path.with_suffix(".source.json").write_text(json.dumps({
        "run_id": "source-run", "base_url": "https://api.test",
    }))
    monkeypatch.setattr(local_source, "materialize_source_traces", lambda p, **kw: (
        p if kw["run_id"] == "source-run" else None
    ))
    assert local_source.materialize_local_parquet(path) != path


def test_summary_only_requires_explicit_setting(tmp_path, source_env):
    path = _compact(tmp_path / "renamed.parquet")
    source_env.remote_full_traces = False
    with pytest.warns(UserWarning, match="Summary-only"):
        assert local_source.materialize_local_parquet(path) == path


@pytest.mark.parametrize("write", [
    lambda path: _compact(path, response_text=json.dumps(
        {"steps": [{"source": "user", "message": "Hi"}]})),
    lambda path: pq.write_table(pa.table({"question": ["Q"], "answer": ["A"]}), path),
], ids=["structured_gym_trace", "plain_table"])
def test_non_export_parquet_stays_local(tmp_path, source_env, write):
    path = tmp_path / "example_autofa.parquet"
    write(path)
    assert materialize_eval_source(EvalSource(kind="parquet", path=path)) == path


def test_selecting_cached_export_reuses_existing_downloads(tmp_path, monkeypatch, source_env):
    root = tmp_path / "cache" / "remote"
    root.mkdir(parents=True)
    path = _compact(root / "dataset.parquet")
    (root / "source.json").write_text(json.dumps({"run_id": "run", "base_url": "https://api.test"}))
    called = []
    monkeypatch.setattr(local_source, "materialize_source_traces",
                        lambda p, **kwargs: called.append(p) or p)
    assert local_source.materialize_local_parquet(path) == path
    assert called == [path]
