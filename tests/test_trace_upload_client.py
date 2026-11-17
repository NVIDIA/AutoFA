# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace stores upload as bounded archives and install atomically on a receiving server."""

import copy
import hashlib
import io
import json
import shutil
import tarfile
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.data.local_trace_store import resolve_store
from autofa.data.remote_library_api import (
    RemoteLibraryClient,
    RemoteLibraryError,
    client_from_config,
)
from autofa.data.trace_upload import (
    TraceArchive,
    server_capabilities,
    trace_archive,
    upload_trace_stores,
)
from autofa.reports.failure_cards import _extract_autofa_data
from autofa_library_server import lazy_report
from autofa_library_server.catalog import LocalCatalog
from autofa_library_server.config import Settings
from autofa_library_server.main import create_app

from support.library_server import RUN, STORE, TEXT, _base, _client, _write_store
from support.library_server import saved_run as saved_run

CAPABILITIES = {"version": 1, "format": "tar", "max_upload_bytes": 64 * 1024**3,
                "max_unpacked_bytes": 64 * 1024**3, "max_files": 100_000,
                "max_manifest_bytes": 16 * 1024**2}
LOCAL_REFERENCE = {"kind": "local_evalrdl", "store_id": STORE, "run_id": RUN}
REFERENCE = {**LOCAL_REFERENCE, "source_harness": "gym"}


# --- sender: archive and upload client ---


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    return _write_store(tmp_path)


def _receipt(source):
    return {"store_id": STORE, "run_id": RUN, "status": "present",
            "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()}


def test_archive_contains_only_snapshot_files_in_bounded_repeatable_chunks(source):
    blob = b"captured artifact\n" * 140_000
    digest = hashlib.sha256(blob).hexdigest()
    (source / "artifacts" / digest).write_bytes(blob)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifact_files"].append({"path": f"artifacts/{digest}", "sha256": digest,
                                       "task_id": "other", "repeat": 0, "artifact_idx": 1})
    manifest_path.write_text(json.dumps(manifest))
    (source / "source.json").write_text('{"parquet_path":"/private/sender/path"}')
    (source / "diagnostics.json").write_text("private sender details")
    (source / "cache").mkdir()
    (source / "cache" / "index.json").write_text("not transferable")
    archive, recorded, manifest_digest = trace_archive(STORE)
    assert recorded == manifest
    assert manifest_digest == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    chunks = list(archive)
    assert max(map(len, chunks)) <= 1024 * 1024
    encoded = b"".join(chunks)
    assert len(encoded) == archive.size and b"".join(archive) == encoded
    with tarfile.open(fileobj=io.BytesIO(encoded), mode="r:") as decoded:
        entries = decoded.getmembers()
        assert entries[0].name == "manifest.json"
        assert {member.name for member in entries} == set(_files(source))
        assert all(member.isreg() and not member.linkname for member in entries)
        for member in entries:
            assert decoded.extractfile(member).read() == (source / member.name).read_bytes()


def test_gnu_large_file_header_is_regular_and_does_not_materialize_the_file():
    huge = 9 * 1024**3

    class NeverRead:
        def is_symlink(self):
            return False

        def stat(self):
            return SimpleNamespace(st_size=huge)

        def open(self, *args):
            pytest.fail("Generating the header must not open or allocate the file")

    archive = TraceArchive((("dataset.parquet", NeverRead(), huge),))
    header = next(iter(archive))
    assert len(header) == 512
    member = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
    assert member.isreg() and member.size == huge and member.name == "dataset.parquet"
    assert archive.size == huge + 1536


@pytest.mark.parametrize("corruption", ["artifact_path", "artifact_hash", "symlink"])
def test_archive_rejects_unsafe_or_changed_artifacts(source, tmp_path, corruption):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    artifact = manifest["artifact_files"][0]
    path = source / artifact["path"]
    if corruption == "artifact_path":
        artifact["path"] = "../../secret"
        manifest_path.write_text(json.dumps(manifest))
    elif corruption == "artifact_hash":
        path.write_bytes(b"different bytes")
    else:
        original = path.read_bytes()
        path.unlink()
        other = tmp_path / "outside"
        other.write_bytes(original)
        path.symlink_to(other)
    with pytest.raises(RemoteLibraryError):
        trace_archive(STORE)


def test_archive_detects_files_that_change_after_inventory(source):
    archive, _, _ = trace_archive(STORE)
    (source / "dataset.parquet").write_bytes(b"changed")
    with pytest.raises(RemoteLibraryError, match="changed"):
        list(archive)


@pytest.mark.parametrize("response", [
    {}, {"managed_trace_upload": {**CAPABILITIES, "version": 2}},
    {"managed_trace_upload": {**CAPABILITIES, "max_files": False}},
], ids=["missing", "incompatible", "invalid_limit"])
def test_capability_negotiation_rejects_incompatible_or_invalid_receivers(response):
    client = SimpleNamespace(_request_json=lambda *args: response)
    with pytest.raises(RemoteLibraryError):
        server_capabilities(client)


@pytest.mark.parametrize("status", [404, 500])
def test_capability_http_failure_never_silently_enables_uploads(status):
    def request(*args):
        raise RemoteLibraryError("unavailable", status=status)

    with pytest.raises(RemoteLibraryError) as error:
        server_capabilities(SimpleNamespace(_request_json=request))
    if status == 404:
        assert "Upgrade" in str(error.value)
    else:
        assert error.value.status == status


def test_existing_identical_stores_are_reused_and_references_drop_machine_paths(source):
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path))
        assert not kwargs and method == "GET"
        return _receipt(source)

    result = upload_trace_stores(SimpleNamespace(_request_json=request),
        [{**REFERENCE, "parquet_path": "/private/sender/dataset.parquet"}, REFERENCE],
        capabilities=CAPABILITIES)
    assert result == [REFERENCE]
    assert calls == [("GET", f"/api/v2/trace-stores/{STORE}")]


def test_existing_same_id_with_different_manifest_is_not_reused_or_overwritten(source):
    calls = []

    def request(method, path, **kwargs):
        calls.append(method)
        return {**_receipt(source), "manifest_sha256": "b" * 64}

    with pytest.raises(RemoteLibraryError, match="does not match"):
        upload_trace_stores(SimpleNamespace(_request_json=request), [REFERENCE],
                            capabilities=CAPABILITIES)
    assert calls == ["GET"]


def test_receiver_limits_fail_before_any_put(source):
    calls = []

    def request(method, path, **kwargs):
        calls.append(method)
        raise RemoteLibraryError("missing", status=404)

    with pytest.raises(RemoteLibraryError, match="limits"):
        upload_trace_stores(SimpleNamespace(_request_json=request), [REFERENCE],
                            capabilities={**CAPABILITIES, "max_upload_bytes": 1})
    assert calls == ["GET"]


def test_missing_store_upload_preserves_stream_and_declares_exact_content_length(source):
    methods = []

    def request(method, path, **kwargs):
        methods.append(method)
        if method == "GET":
            raise RemoteLibraryError("missing", status=404)
        stream = kwargs["body"]
        assert isinstance(stream, TraceArchive)
        assert kwargs["headers"] == {"Content-Type": "application/x-tar",
                                      "Content-Length": str(stream.size)}
        assert sum(len(chunk) for chunk in stream) == stream.size
        return _receipt(source)

    assert upload_trace_stores(SimpleNamespace(_request_json=request), [REFERENCE],
                               capabilities=CAPABILITIES) == [REFERENCE]
    assert methods == ["GET", "PUT"]


def test_conflicting_duplicate_source_bindings_are_rejected(source):
    client = SimpleNamespace(_request_json=lambda *args: _receipt(source))
    with pytest.raises(RemoteLibraryError, match="(?i)identit|match|conflict"):
        upload_trace_stores(client, [REFERENCE, {**REFERENCE, "run_id": "different"}],
                            capabilities=CAPABILITIES)


@pytest.mark.parametrize("url,token_env", [
    ("http://127.0.0.1:8765", "GITLAB_TOKEN"),
    ("https://receiver.example.test", "CUSTOM_SERVER_TOKEN"),
])
def test_server_factory_uses_only_the_selected_server_token(monkeypatch, url, token_env):
    def forbidden(*args, **kwargs):
        pytest.fail("Server mode must not touch profile or hosted credentials")

    monkeypatch.setattr("autofa.local_service.service_token", forbidden)
    monkeypatch.setattr("autofa.data.remote_library_credentials.resolve_credential", forbidden)
    monkeypatch.setenv("GITLAB_TOKEN", "private-gitlab-token")
    monkeypatch.setenv("AUTOFA_LIBRARY_TOKEN", "receiver-default-token")
    monkeypatch.setenv("CUSTOM_SERVER_TOKEN", "receiver-custom-token")
    config = Config(remote_library_mode="server", remote_library_base_url=url,
                    remote_library_token_env=token_env)
    client = client_from_config(config, credential_store=SimpleNamespace(get_token=forbidden))
    expected = "receiver-custom-token" if token_env == "CUSTOM_SERVER_TOKEN" else (
        "receiver-default-token")
    assert client.token == expected and client.credential_source == "server"
    monkeypatch.delenv("CUSTOM_SERVER_TOKEN" if token_env == "CUSTOM_SERVER_TOKEN"
                       else "AUTOFA_LIBRARY_TOKEN")
    with pytest.raises(RemoteLibraryError, match="receiving AutoFA server"):
        client_from_config(config, credential_store=SimpleNamespace(get_token=forbidden))


@pytest.mark.parametrize("url", ["ftp://receiver", "https://user:secret@receiver",
                                 "https://receiver?token=secret", "https://recei\nver"])
def test_server_address_rejects_nonorigins_before_transport(url):
    with pytest.raises(ValueError):
        RemoteLibraryClient(url, token="test-token", credential_source="server")


def test_server_tokens_reject_characters_that_cannot_be_bearer_headers():
    with pytest.raises(ValueError, match="token"):
        RemoteLibraryClient("http://receiver:8765", token="header\r\ninjection",
                            credential_source="server")


def test_server_transport_refuses_redirects_without_forwarding_credentials():
    requests = []

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            self.send_response(int(self.path.removeprefix("/")))
            self.send_header("Location", f"http://localhost:{self.server.server_port}/stolen")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RemoteLibraryClient(f"http://127.0.0.1:{server.server_port}",
                                     token="server-secret", credential_source="server")
        for code in (302, 307):
            with pytest.raises(RemoteLibraryError, match="redirect"):
                client._request_json("GET", f"/{code}")
        assert requests == [("/302", "Bearer server-secret"), ("/307", "Bearer server-secret")]
        with pytest.raises(RemoteLibraryError, match="different host"):
            client._request_json("GET", "http://another.example.test/stolen")
        assert len(requests) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- receiver: managed trace-store transfer ---


def _files(store):
    manifest = json.loads((store / "manifest.json").read_text())
    names = {"manifest.json", "dataset.parquet"}
    names.update(f"tables/{name}" for name in manifest["tables"])
    names.update(record["path"] for record in manifest["artifact_files"])
    return {name: (store / name).read_bytes() for name in sorted(names)}


def _archive(files, *, extra=None):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
        if extra is not None:
            archive.addfile(extra)
    return output.getvalue()


# Building the sender's saved run dominates setup, and its bytes are identical for every
# test, so each worker builds it once and every test gets a fresh receiver profile.
_SENT = {}


@pytest.fixture
def destination(request, tmp_path, monkeypatch):
    if not _SENT:
        _, store, library, iid = request.getfixturevalue("saved_run")
        _SENT.update(files=_files(store), library=library, iid=iid)
        shutil.rmtree(store)
    monkeypatch.setattr(LocalCatalog, "refresh_interval", 0)
    profile = tmp_path / "remote"
    monkeypatch.setenv("AUTOFA_HOME", str(profile))
    settings = Settings(data_dir=profile / "server", library_dir=profile / "libraries",
                        runs_dir=profile / "runs", local_token="remote-test-secret")
    return settings, dict(_SENT["files"]), copy.deepcopy(_SENT["library"]), _SENT["iid"]


def _headers(settings):
    return {"Authorization": f"Bearer {settings.local_token}", "Content-Type": "application/x-tar"}


def _put(client, settings, content, headers=None):
    return client.put(f"/api/v2/trace-stores/{STORE}", content=content,
                      headers=_headers(settings) if headers is None else headers)


def _publish(client, settings, library, references):
    return client.post("/api/v2/publications",
        headers={"Authorization": f"Bearer {settings.local_token}"},
        data={"metadata": json.dumps({"branch_key": "remote", "source_run_id": "sender-run",
                                      "model_id": "local-model", "benchmarks": ["local-bench"]}),
              "trace_references": json.dumps(references)},
        files={"library": ("library.json", json.dumps(library.to_dict()), "application/json"),
               "failure_card": ("report.html", "<html>Lightweight</html>", "text/html")})


def test_remote_store_publication_retains_full_trace_artifacts_and_highlights(destination):
    settings, files, library, iid = destination
    settings = replace(settings, max_report_assets_bytes=1, max_report_assets_unpacked_bytes=1)
    app = create_app(settings)
    with _client(app, settings.public_url) as client:
        capabilities = client.get("/api/v2/capabilities", headers=_headers(settings))
        assert capabilities.json()["managed_trace_upload"]["version"] == 1
        path = f"/api/v2/trace-stores/{STORE}"
        assert client.get(path, headers=_headers(settings)).status_code == 404
        sent = _put(client, settings, _archive(files))
        assert sent.status_code == 201, sent.text
        assert sent.json()["manifest_sha256"] == hashlib.sha256(files["manifest.json"]).hexdigest()
        assert client.get(path, headers=_headers(settings)).json() == sent.json()
        assert _put(client, settings, _archive(files)).status_code == 200
        destination_store = resolve_store(STORE)
        reference = json.loads((destination_store / "source.json").read_text())
        assert reference["parquet_path"] == str(destination_store / "dataset.parquet")
        assert not settings.runs_dir.exists()
        publication = _publish(client, settings, library, [LOCAL_REFERENCE])
        assert publication.status_code == 201, publication.text
        version = publication.json()
        report = client.get(f"/libraries/{version['library_id']}/report")
        data = _extract_autofa_data(report.text)
        assert data["meta"]["local_lazy_items"] and not data["items"]
        item = client.get(data["meta"]["items_base"] + "/" + data["item_index"][iid]).json()
        assert "TAIL EVIDENCE" in item["response_raw"]
        location = item["localisations"]["detector"]["passages"][0]["trace_locations"][0]
        assert location["source_url"].startswith(f"/traces/{STORE}/#/")
        assert "text=%F0%9F%99%82%20TAIL%20EVIDENCE" in location["source_url"]
        full = client.get(_base() + "/part-text?event_idx=55&part_idx=7&lane=conversation")
        assert full.status_code == 200 and full.json()["text"] == TEXT
        image = client.get(_base() + "/artifacts/0/download")
        assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
    # All runtime state can disappear; the persisted store still drives the native viewer.
    with _client(create_app(settings), settings.public_url) as client:
        assert client.get(_base() + "/part-text?event_idx=55&part_idx=7&lane=conversation").json()[
            "text"] == TEXT


def test_renderer_update_refreshes_pushed_lazy_assets_without_republication(
    destination, monkeypatch,
):
    """A receiver upgrade must invalidate cached item assets even with a stable session key."""
    settings, files, library, iid = destination
    settings = replace(settings, session_secret="stable-across-server-restarts")
    monkeypatch.setattr(lazy_report, "renderer_fingerprint", lambda: "old-renderer")
    with _client(create_app(settings), settings.public_url) as client:
        assert _put(client, settings, _archive(files)).status_code == 201
        version = _publish(client, settings, library, [LOCAL_REFERENCE]).json()

    path = f"/libraries/{version['library_id']}/report"
    selection = {"branch_id": version["branch_id"], "version_id": version["version_id"]}

    def report(client, etag=None):
        headers = {"If-None-Match": etag} if etag else None
        response = client.get(path, params=selection, headers=headers)
        if response.status_code == 304:
            return response, None
        data = _extract_autofa_data(response.text)
        return response, data["meta"]["items_base"] + "/" + data["item_index"][iid]

    with _client(create_app(settings), settings.public_url) as client:
        old, old_url = report(client)

    # Only the receiver's code changes; the immutable pushed version is not republished.
    monkeypatch.setattr(lazy_report, "renderer_fingerprint", lambda: "new-renderer")
    with _client(create_app(settings), settings.public_url) as client:
        new, new_url = report(client, old.headers["etag"])
        assert new.status_code == 200 and new.headers["etag"] != old.headers["etag"]
        assert new_url != old_url and client.get(old_url).status_code == 404
        assert "TAIL EVIDENCE" in client.get(new_url).json()["response_raw"]
        assert report(client, new.headers["etag"])[0].status_code == 304


@pytest.mark.parametrize("case", ["traversal", "symlink", "duplicate", "table_hash", "missing",
                                  "manifest_id", "truncated"])
def test_invalid_transfer_is_atomic_and_never_installs(destination, case):
    settings, files, _, _ = destination
    extra = None
    if case == "traversal":
        files["../outside"] = b"invalid"
    elif case == "symlink":
        extra = tarfile.TarInfo("tables/evil.parquet")
        extra.type = tarfile.SYMTYPE
        extra.linkname = "/etc/passwd"
    elif case == "duplicate":
        extra = tarfile.TarInfo("dataset.parquet")
    elif case == "table_hash":
        files[next(name for name in files if name.startswith("tables/"))] += b"corruption"
    elif case == "missing":
        del files["dataset.parquet"]
    elif case == "manifest_id":
        manifest = json.loads(files["manifest.json"])
        manifest["store_id"] = "b" * 64
        files["manifest.json"] = json.dumps(manifest).encode()
    archive = _archive(files, extra=extra)
    if case == "truncated":
        archive = archive[:600]
    with _client(create_app(settings), settings.public_url) as client:
        response = _put(client, settings, archive)
        assert response.status_code == 422, response.text
    root = settings.data_dir.parent / "sources"
    assert not (root / STORE).exists()
    assert not list(root.glob(".upload-*"))


def _gnu_header(files):
    entry = tarfile.TarInfo("dataset.parquet")
    entry.size = 9 * 1024**3
    return entry.tobuf(format=tarfile.GNU_FORMAT)


def _chunked(files):
    archive = _archive(files)
    return (archive[offset:offset + 257] for offset in range(0, len(archive), 257))


@pytest.mark.parametrize("field,value,body", [
    ("max_trace_upload_bytes", 10, _archive), ("max_trace_files", 1, _archive),
    ("max_trace_manifest_bytes", 10, _archive),
    # Streaming without Content-Length is still bounded.
    ("max_trace_upload_bytes", 1024, _chunked),
    # A GNU numeric member size is checked before allocating disk.
    ("max_trace_unpacked_bytes", 1024, _gnu_header),
])
def test_transfer_limits_are_separate_from_small_report_limits(destination, field, value, body):
    settings, files, _, _ = destination
    settings = replace(settings, **{field: value}, max_report_assets_bytes=1)
    with _client(create_app(settings), settings.public_url) as client:
        response = _put(client, settings, body(files))
        assert response.status_code == 413, response.text
    root = settings.data_dir.parent / "sources"
    assert not (root / STORE).exists() and not list(root.glob(".upload-*"))


def test_transfer_requires_server_token_before_reading_archive(destination):
    settings, _, _, _ = destination
    with _client(create_app(settings), settings.public_url) as client:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            assert _put(client, settings, b"invalid", headers).status_code in {401, 403}
            assert client.get("/api/v2/capabilities", headers=headers).status_code in {401, 403}
    assert not (settings.data_dir.parent / "sources").exists()


def test_existing_store_cannot_be_replaced(destination):
    settings, files, _, _ = destination
    with _client(create_app(settings), settings.public_url) as client:
        path = f"/api/v2/trace-stores/{STORE}"
        assert _put(client, settings, _archive(files)).status_code == 201
        before = client.get(path, headers=_headers(settings)).json()
        manifest = json.loads(files["manifest.json"])
        manifest["source_harness"] = "different"
        files["manifest.json"] = json.dumps(manifest).encode()
        assert _put(client, settings, _archive(files)).status_code == 409
        assert client.get(path, headers=_headers(settings)).json() == before


def test_existing_incomplete_store_is_not_replaced(destination):
    settings, files, _, _ = destination
    existing = settings.data_dir.parent / "sources" / STORE
    existing.mkdir(parents=True)
    with _client(create_app(settings), settings.public_url) as client:
        assert _put(client, settings, _archive(files)).status_code == 409
    assert existing.is_dir() and not list(existing.iterdir())


@pytest.mark.parametrize("case", ["path", "missing", "duplicate", "uncovered"])
def test_report_transfer_references_must_resolve_all_examples(destination, case):
    settings, files, library, _ = destination
    with _client(create_app(settings), settings.public_url) as client:
        assert _put(client, settings, _archive(files)).status_code == 201
        reference = dict(LOCAL_REFERENCE)
        if case == "path":
            reference["parquet_path"] = "/etc/passwd"
        elif case == "missing":
            reference["store_id"] = "b" * 64
        elif case == "uncovered":
            library.functions[0].example_fail_ids.append("missing-source-item")
        response = _publish(client, settings, library,
                            [reference, reference] if case == "duplicate" else [reference])
        assert response.status_code in {404, 422}, response.text
        assert client.get("/api/v2/libraries").json()["items"] == []
