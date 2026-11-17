# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AutoFA library v2 client: immutable cache, portable transport, credential resolution and
a local dashboard selection that stays isolated from cloud credentials."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import io
import json
import threading
import urllib.error
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.library_transport import (
    MAX_UNPACKED_BYTES,
    PACKED_KEY,
    REVIEW_PACKED_KEY,
    pack_bucket_examples,
    unpack_bucket_examples,
)
from autofa.data.remote_library_api import (
    RemoteBranch,
    RemoteCliToken,
    RemoteCliTokenPending,
    RemoteLibrary,
    RemoteLibraryClient,
    RemoteLibraryError,
    RemoteLibraryRef,
    RemoteVersion,
    client_from_config,
    generate_pkce_pair,
    materialize_remote_library,
    remote_library_cache_dir,
)
from autofa.data.remote_library_credentials import (
    CredentialError,
    KeyringCredentialStore,
    normalize_service_url,
)
from autofa.local_service import service_token

from support.traces import UrlResponse


def _serve(monkeypatch, *responses):
    """Answer successive urllib requests with the given responses, recording each request."""
    calls, pending = [], list(responses)

    def fake_urlopen(request, timeout=None):
        calls.append((request, timeout))
        return pending.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def _http_error(status, body):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "Error", hdrs=None,
                                     fp=io.BytesIO(body))

    return fake_urlopen


def _library_bytes() -> bytes:
    return json.dumps(
        {"version": 1, "model_id": "model-a", "benchmark": "bench-a", "functions": []}
    ).encode()


def _reference(run_id: str) -> dict:
    return {"run_id": run_id, "source_harness": "nemo-gym",
            "export_url": f"https://eval.example.org/v1/runs/{run_id}/export/autofa"}


def _logical(library_id: str, *, latest: str) -> dict:
    return {
        "id": library_id, "owner_id": 7, "owner_username": "owner", "owner_name": "Owner",
        "owner_avatar": "", "source_run_id": "latest-run", "model_id": "model-a",
        "harness": "claude", "source_harness": "nemo-gym",
        "reference_data": [_reference("data-run")], "publish_trigger": "session_end",
        "benchmarks": ["bench-a"], "function_count": 3, "created_at": "2026-09-01T10:00:00Z",
        "updated_at": "2026-09-02T10:00:00Z", "revision": 2, "library_sha256": "abc",
        "report_sha256": "def", "latest_branch_id": "branch-1", "latest_version_id": latest,
        "branch_count": 1, "version_count": 2,
    }


def _branch(*, latest: str = "version-2") -> dict:
    return {
        "id": "branch-1", "library_id": "library-1", "branch_key": "origin-run",
        "created_at": "2026-09-01T10:00:00Z", "updated_at": "2026-09-02T10:00:00Z",
        "latest_version_id": latest, "latest_version_number": 2,
        "latest_source_run_id": "continued-run", "version_count": 2,
    }


def _version(version_id: str, number: int) -> dict:
    base = f"/api/v2/libraries/library-1/branches/branch-1/versions/{version_id}"
    return {
        "id": version_id, "library_id": "library-1", "branch_id": "branch-1",
        "branch_key": "origin-run", "version_number": number, "source_run_id": f"run-{number}",
        "model_id": "model-a", "harness": "claude", "source_harness": "nemo-gym",
        "reference_data": [_reference(f"data-run-{number}")], "publish_trigger": "manual_cli",
        "benchmarks": ["bench-a"], "function_count": number,
        "created_at": f"2026-09-0{number}T10:00:00Z",
        "library_sha256": hashlib.sha256(_library_bytes()).hexdigest(),
        "report_sha256": "report", "library_url": base + "/library.json",
        "failure_card_url": base + "/failure-card",
    }


def _publication(version_id: str, number: int, **extra) -> dict:
    return {**_version(version_id, number), "library_id": "library-1", "branch_id": "branch-1",
            "version_id": version_id, **extra}


def test_list_v2_follows_pagination_and_parses_logical_records(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request.full_url, request.get_header("Authorization"), timeout))
        offset = 1 if "offset=1" in request.full_url else 0
        item = _logical(f"library-{offset + 1}", latest=f"version-{offset + 1}")
        return UrlResponse(
            json.dumps({"items": [item], "total": 2, "limit": 1, "offset": offset}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = RemoteLibraryClient("https://example.test/", token="secret", timeout=7)
    libraries = client.list_libraries()

    assert [item.id for item in libraries] == ["library-1", "library-2"]
    assert (libraries[0].latest_branch_id, libraries[0].publish_trigger) == (
        "branch-1", "session_end")
    assert calls == [
        ("https://example.test/api/v2/libraries", "Bearer secret", 7),
        ("https://example.test/api/v2/libraries?limit=1&offset=1", "Bearer secret", 7),
    ]


def test_branch_version_endpoints_and_latest_resolution(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        calls.append(url)
        if url.endswith("/api/v2/libraries"):
            payload = {"items": [_logical("library-1", latest="version-2")], "total": 1}
        elif url.endswith("/libraries/library-1/branches"):
            payload = {"items": [_branch()], "total": 1}
        elif url.endswith("/branches/branch-1/versions"):
            # Deliberately return oldest first; the client guarantees newest first.
            payload = {"items": [_version("version-1", 1), _version("version-2", 2)], "total": 2}
        elif url.endswith("/branches/branch-1/versions/version-2"):
            payload = _version("version-2", 2)
        else:
            raise AssertionError(url)
        return UrlResponse(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = RemoteLibraryClient("https://example.test", token="secret")

    versions = client.list_versions("library-1", "branch-1")
    assert [version.id for version in versions] == ["version-2", "version-1"]
    assert (versions[0].harness, versions[0].source_harness, versions[0].publish_trigger) == (
        "claude", "nemo-gym", "manual_cli")
    assert versions[0].reference_data[0].run_id == "data-run-2"
    selected = client.resolve_library_ref("library-1")
    assert (selected.library_id, selected.branch_id, selected.version_id,
            selected.branch_key) == ("library-1", "branch-1", "version-2", "origin-run")
    assert calls[-1].endswith("/branches/branch-1/versions/version-2")


def test_get_library_not_found_uses_v2_collection(monkeypatch):
    calls = _serve(monkeypatch, UrlResponse(b'{"items":[],"total":0,"limit":50,"offset":0}'))
    client = RemoteLibraryClient("https://example.test", token="secret")
    with pytest.raises(RemoteLibraryError, match="not found") as error:
        client.get_library("folder/id")
    assert error.value.status == 404
    assert [request.full_url for request, _ in calls] == ["https://example.test/api/v2/libraries"]


@pytest.mark.parametrize("payload", [
    [_logical("library-1", latest="version-1")],
    {"libraries": [_logical("library-1", latest="version-1")], "total": 1},
])
def test_listing_requires_the_v2_items_envelope(monkeypatch, payload):
    _serve(monkeypatch, UrlResponse(json.dumps(payload).encode()))
    client = RemoteLibraryClient("https://example.test", token="secret")
    with pytest.raises(RemoteLibraryError, match="paginated item array"):
        client.list_libraries()


def test_publication_posts_v2_multipart_with_full_metadata(monkeypatch):
    calls = _serve(monkeypatch, UrlResponse(json.dumps(_publication(
        "version-3", 3, branch_key="origin-run", source_run_id="current-run",
    )).encode()))
    client = RemoteLibraryClient("https://example.test", token="secret")
    result = client.upload_library(
        metadata={
            "branch_key": "origin-run", "source_run_id": "current-run", "model_id": "model-a",
            "harness": "claude", "source_harness": "nemo-gym",
            "reference_data": [_reference("data-run")], "publish_trigger": "manual_cli",
            "benchmarks": ["bench-a"], "library_id": "library-1", "branch_id": "branch-1",
        },
        library_json=_library_bytes(),
        failure_cards_html="<html>lean report</html>",
        report_assets_zip=b"PK-report-assets",
    )

    (request, _), = calls
    body = request.data
    assert request.method == "POST"
    assert request.full_url == "https://example.test/api/v2/publications"
    for expected in (
        b'name="metadata"', b'"branch_key":"origin-run"', b'"source_run_id":"current-run"',
        b'"harness":"claude"', b'"source_harness":"nemo-gym"', b'"run_id":"data-run"',
        b'"publish_trigger":"manual_cli"', b'"library_id":"library-1"',
        b'name="library"; filename="library.json"',
        b'name="failure_card"; filename="failure_cards.html"',
        b'name="report_assets"; filename="failure_card_assets.zip"', b"PK-report-assets",
    ):
        assert expected in body
    assert b"config.json" not in body
    assert result.version_id == "version-3"
    assert result.dashboard_url.endswith(
        "/libraries/library-1?branch_id=branch-1&version_id=version-3"
    )


def test_oversized_library_upload_packs_examples_and_keeps_all_values(monkeypatch):
    from email import policy
    from email.parser import BytesParser

    library = FunctionLibrary(version=2, model_id="model-a", functions=[FeatureFunction(
        "f", "claim", "def detect(item):\n    return False", scores={"buckets": [{
            "bucket": "False", "n": 5000, "example_fail_ids": [
                f"model|benchmark|problem-{i}|repeat-0" for i in range(5000)
            ], "example_pass_ids": [],
        }]},
    )]).to_dict()
    original = json.dumps(library)
    captured = {}

    def request(method, path, *, body, headers):
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body,
        )
        uploaded = next(part.get_payload(decode=True) for part in message.iter_parts()
                        if part.get_param("name", header="content-disposition") == "library")
        assert len(uploaded) < 50_000
        captured.update(json.loads(uploaded))
        return _publication("version-3", 3, branch_key="run")

    client = RemoteLibraryClient("https://example.test", token="secret", max_library_bytes=50_000)
    monkeypatch.setattr(client, "_request_json", request)
    client.upload_library(metadata={"branch_key": "run", "source_run_id": "run",
                                    "model_id": "model-a"},
                          library_json=library, failure_cards_html="<html></html>")
    assert PACKED_KEY in captured
    assert unpack_bucket_examples(captured) == library
    assert json.dumps(library) == original


def test_download_concrete_version_is_atomic_hash_validated_and_limited(
    tmp_path: Path, monkeypatch,
):
    destination = tmp_path / "library.json"
    destination.write_text('{"old":true}', encoding="utf-8")
    calls = _serve(monkeypatch, UrlResponse(_library_bytes()), UrlResponse(b"not-json"),
                   UrlResponse(b"x" * 257))
    client = RemoteLibraryClient("https://example.test", token="secret", max_library_bytes=256)

    def download(**kwargs):
        client.download_library("library-1", destination, branch_id="branch-1",
                                version_id="version-1", **kwargs)

    download(expected_sha256=hashlib.sha256(_library_bytes()).hexdigest())
    assert json.loads(destination.read_text(encoding="utf-8"))["version"] == 1
    assert calls[0][0].full_url.endswith(
        "/api/v2/libraries/library-1/branches/branch-1/versions/version-1/library.json"
    )
    with pytest.raises(RemoteLibraryError, match="invalid"):
        download()
    assert json.loads(destination.read_text(encoding="utf-8"))["version"] == 1
    with pytest.raises(RemoteLibraryError, match="limit"):
        download()
    assert list(tmp_path.glob(".*.partial")) == []


def test_hash_mismatch_keeps_previous_file(tmp_path: Path, monkeypatch):
    destination = tmp_path / "library.json"
    destination.write_text('{"version":1,"functions":[]}', encoding="utf-8")
    original = destination.read_bytes()
    _serve(monkeypatch, UrlResponse(_library_bytes()))
    client = RemoteLibraryClient("https://example.test", token="secret")
    with pytest.raises(RemoteLibraryError, match="SHA-256 mismatch"):
        client.download_library("library-1", destination, branch_id="branch-1",
                                version_id="version-1", expected_sha256="0" * 64)
    assert destination.read_bytes() == original


def test_materialize_cache_is_service_and_version_scoped_not_timestamp_scoped(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    payload = _library_bytes()
    version = RemoteVersion(
        id="version-1", library_id="library-1", branch_id="branch-1", branch_key="origin-run",
        created_at="2026-09-01T10:00:00Z", library_sha256=hashlib.sha256(payload).hexdigest(),
    )
    ref = RemoteLibraryRef(
        library=RemoteLibrary(id="library-1", latest_branch_id="branch-1",
                              latest_version_id="version-1"),
        branch=RemoteBranch(id="branch-1", library_id="library-1", branch_key="origin-run",
                            latest_version_id="version-1"),
        version=version,
    )
    client = RemoteLibraryClient("https://example.test", token="secret")
    calls = {"count": 0}

    def fake_download(selected, destination, **kwargs):
        calls["count"] += 1
        destination.write_bytes(payload)
        return destination

    monkeypatch.setattr(client, "download_version", fake_download)
    path = materialize_remote_library(ref, client=client)
    expected = remote_library_cache_dir("version-1", base_url=client.base_url)
    assert path == expected / "library.json"
    metadata = json.loads((expected / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["version"]["id"] == "version-1"
    assert metadata["selection"]["branch"]["branch_key"] == "origin-run"
    assert "secret" not in json.dumps(metadata)

    changed_timestamp = replace(ref, version=replace(version, created_at="2026-09-02T10:00:00Z"))
    assert materialize_remote_library(changed_timestamp, client=client) == path
    assert calls["count"] == 1
    other_service = remote_library_cache_dir("version-1", base_url="https://other.example.test")
    assert other_service != expected


def test_materialize_rejects_logical_library_without_latest_version():
    client = RemoteLibraryClient("https://example.test", token="secret")
    with pytest.raises(RemoteLibraryError, match="no latest branch or version"):
        materialize_remote_library(RemoteLibrary(id="library-1"), client=client)


def test_pkce_pair_is_high_entropy_rfc7636_s256():
    pair = generate_pkce_pair(lambda size: bytes(range(size)))
    assert len(pair.verifier) == 86
    assert "=" not in pair.verifier
    assert len(pair.challenge) == 43
    assert "=" not in pair.challenge
    expected = hashlib.sha256(pair.verifier.encode("ascii")).digest()
    assert pair.challenge == base64.urlsafe_b64encode(expected).rstrip(b"=").decode("ascii")


def test_cli_auth_start_pending_success_and_request_secrecy(monkeypatch):
    pair = generate_pkce_pair(lambda size: b"x" * size)
    flow_id = "f" * 43
    calls = _serve(
        monkeypatch,
        UrlResponse(json.dumps({
            "flow_id": flow_id, "login_url": f"https://example.test/auth/cli/{flow_id}",
            "expires_in": 600, "interval": 2,
        }).encode()),
        UrlResponse(b'{"status":"pending","interval":2}', status=202),
        UrlResponse(json.dumps({
            "token": "autofa_cli_abcdefghijklmnopqrstuvwxyz0123456789", "token_type": "bearer",
            "expires_at": "2026-09-03T12:00:00Z", "expires_in": 3600,
            "identity": {"id": 7, "username": "alice", "name": "Alice"},
        }).encode()),
    )
    client = RemoteLibraryClient("https://example.test", timeout=9)
    started = client.start_cli_login(pair.challenge)
    pending = client.poll_cli_token(started.flow_id, pair.verifier)
    completed = client.poll_cli_token(started.flow_id, pair.verifier)

    assert started.interval == 2
    assert isinstance(pending, RemoteCliTokenPending)
    assert isinstance(completed, RemoteCliToken)
    assert completed.identity is not None
    assert completed.identity.username == "alice"
    assert completed.token not in repr(completed)
    requests = [request for request, _ in calls]
    assert requests[0].full_url.endswith("/api/v2/auth/cli/start")
    assert requests[0].get_header("Authorization") is None
    assert json.loads(requests[0].data)["code_challenge"] == pair.challenge
    assert all(call.full_url.endswith("/api/v2/auth/cli/token") for call in requests[1:])
    assert pair.verifier not in requests[1].full_url
    assert json.loads(requests[1].data) == {"flow_id": flow_id, "verifier": pair.verifier}
    assert [timeout for _, timeout in calls] == [9, 9, 9]


@pytest.mark.parametrize("host,login_id", [("evil.test", "f" * 43), ("example.test", "g" * 43)],
                         ids=["foreign-host", "foreign-flow"])
def test_cli_auth_rejects_mismatched_login_url(monkeypatch, host, login_id):
    _serve(monkeypatch, UrlResponse(json.dumps({
        "flow_id": "f" * 43, "login_url": f"https://{host}/auth/cli/{login_id}",
        "expires_in": 600, "interval": 2,
    }).encode()))
    client = RemoteLibraryClient("https://example.test")
    with pytest.raises(RemoteLibraryError, match="configured service"):
        client.start_cli_login("A" * 43)


def test_cli_auth_rejects_invalid_token_response(monkeypatch):
    pair = generate_pkce_pair(lambda size: b"x" * size)
    _serve(monkeypatch, UrlResponse(b'{"token":"not-a-cli-token","expires_in":3600}'))
    with pytest.raises(RemoteLibraryError, match="Invalid CLI token response"):
        RemoteLibraryClient("https://example.test").poll_cli_token("f" * 43, pair.verifier)


def test_cli_auth_response_size_is_bounded(monkeypatch):
    _serve(monkeypatch, UrlResponse(b"{}", headers={"Content-Length": str(65 * 1024)}))
    with pytest.raises(RemoteLibraryError, match="exceeds"):
        RemoteLibraryClient("https://example.test").start_cli_login("A" * 43)


def test_me_and_revoke_use_authenticated_endpoints(monkeypatch):
    calls = _serve(
        monkeypatch,
        UrlResponse(b'{"id":7,"username":"alice","name":"Alice","auth_method":"cli_token"}'),
        UrlResponse(b"", status=204),
    )
    client = RemoteLibraryClient("https://example.test", token="autofa_cli_secret",
                                 credential_source="keychain")
    identity = client.me()
    client.revoke_cli_token()

    assert identity.username == "alice"
    assert identity.auth_method == "cli_token"
    assert client.credential_source == "keychain"
    requests = [request for request, _ in calls]
    assert requests[0].full_url.endswith("/api/v1/me")
    assert requests[1].method == "DELETE"
    assert requests[1].full_url.endswith("/api/v2/auth/cli/token")
    assert all(call.get_header("Authorization") == "Bearer autofa_cli_secret"
               for call in requests)


def test_cli_auth_errors_redact_token_and_verifier(monkeypatch):
    token = "autofa_cli_super-secret-token"
    pair = generate_pkce_pair(lambda size: b"z" * size)
    monkeypatch.setattr("urllib.request.urlopen",
                        _http_error(400, f'{{"detail":"{token} {pair.verifier}"}}'.encode()))
    client = RemoteLibraryClient("https://example.test", token=token)
    with pytest.raises(RemoteLibraryError) as error:
        client.poll_cli_token("f" * 43, pair.verifier)
    message = str(error.value)
    assert token not in message
    assert pair.verifier not in message
    assert "[REDACTED]" in message


# Portable bucket-example and review-history transport.


def library_data():
    fn = FeatureFunction("f", "Claim", "def detect(item):\n    return False",
                         scores={"recall": 0.25, "buckets": [
                             {"bucket": "True", "n": 3000, "example_fail_ids": [
                                 f"model|benchmark|problem-{n}|0" for n in range(3000)
                             ], "example_pass_ids": []},
                         ]})
    return FunctionLibrary(version=2, functions=[fn]).to_dict()


CORRUPTIONS = {
    "oversize": {"uncompressed_bytes": MAX_UNPACKED_BYTES + 1}, "checksum": {"sha256": "0" * 64},
    "length": {"uncompressed_bytes": 1}, "schema": {"schema_version": 2},
}


def _repack(member, records):
    raw = json.dumps(records).encode()
    member.update(data=base64.b64encode(gzip.compress(raw)).decode(),
                  uncompressed_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("change", ["oversize", "checksum"])
def test_corrupt_packed_examples_fail_before_library_loading(change):
    packed = pack_bucket_examples(library_data())
    packed[PACKED_KEY].update(CORRUPTIONS[change])
    with pytest.raises(ValueError, match="Invalid packed bucket examples"):
        FunctionLibrary.from_dict(packed)


@pytest.mark.parametrize("record", [
    [0, 0, "recall", []], [9, 0, "example_fail_ids", []],
])
def test_packed_records_cannot_target_other_fields(record):
    packed = pack_bucket_examples(library_data())
    _repack(packed[PACKED_KEY], [record])
    with pytest.raises(ValueError, match="Invalid packed bucket examples"):
        unpack_bucket_examples(packed)


def test_large_review_history_restores_exactly_without_changing_scores_or_sources():
    from autofa_library_server.library_identity import same_library_content

    data = library_data()
    data.update(model_id="model", benchmark="benchmark")
    evidence = [{"item_id": f"item-{i}", "reason": "saved causal evidence " * 50}
                for i in range(100)]
    data["functions"][0]["scores"].update(
        behavior_review_assessments=evidence, behavior_purity=.93,
        behavior_verdict_history=[{"grade": "GOOD", "evidence": evidence}],
    )
    before = copy.deepcopy(data)
    packed = pack_bucket_examples(data, max_bytes=1024)
    assert REVIEW_PACKED_KEY in packed
    assert packed["review_history"]["history_complete"] is False
    assert packed["functions"][0]["source"] == data["functions"][0]["source"]
    assert packed["functions"][0]["scores"]["behavior_purity"] == .93
    assert len(json.dumps(packed)) < len(json.dumps(data)) / 10
    assert unpack_bucket_examples(packed) == before
    assert same_library_content(packed, before)
    assert FunctionLibrary.from_dict(packed).to_dict() == before
    assert data == before
    assert pack_bucket_examples(packed, max_bytes=1024) == packed


@pytest.mark.parametrize("change", ["length", "schema", "duplicate"])
def test_corrupt_review_envelopes_are_rejected(change):
    data = library_data()
    data["functions"][0]["scores"]["diagnostic_review"] = {"reason": "evidence" * 100}
    packed = pack_bucket_examples(data, max_bytes=0)
    member = packed[REVIEW_PACKED_KEY]
    if change == "duplicate":
        records = json.loads(gzip.decompress(base64.b64decode(member["data"])))
        _repack(member, records + records[:1])
    else:
        member.update(CORRUPTIONS[change])
    with pytest.raises(ValueError, match="Invalid packed review evidence"):
        FunctionLibrary.from_dict(packed)


def test_combined_decompression_limit_applies_before_decoding():
    packed = pack_bucket_examples(library_data(), max_bytes=0)
    packed[PACKED_KEY]["uncompressed_bytes"] = MAX_UNPACKED_BYTES
    with pytest.raises(ValueError, match="total decompression limit"):
        unpack_bucket_examples(packed)


# Credential storage and precedence without touching the OS keychain.


class FakeBackend:
    priority = 5

    def __init__(self):
        self.values = {}
        self.calls = []
        self.error = None

    def get_password(self, service, account):
        self.calls.append(("get", service, account))
        if self.error:
            raise self.error
        return self.values.get((service, account))

    def set_password(self, service, account, password):
        self.calls.append(("set", service, account, password))
        if self.error:
            raise self.error
        self.values[(service, account)] = password

    def delete_password(self, service, account):
        self.calls.append(("delete", service, account))
        if self.error:
            raise self.error
        self.values.pop((service, account), None)


def test_normalize_service_url_canonicalizes_keychain_identity():
    assert (
        normalize_service_url("HTTPS://Example.TEST:443/")
        == "https://example.test"
    )
    assert (
        normalize_service_url("http://Example.TEST:8080/base/")
        == "http://example.test:8080/base"
    )
    with pytest.raises(CredentialError, match="query"):
        normalize_service_url("https://example.test?token=bad")


def test_keyring_save_read_delete_is_url_scoped_and_idempotent():
    backend = FakeBackend()
    store = KeyringCredentialStore(backend)
    token = "autofa_cli_secret"

    store.set_token("HTTPS://Example.TEST:443/", token)
    assert store.get_token("https://example.test") == token
    assert store.get_token("https://other.test") is None
    assert store.delete_token("https://example.test") is True
    assert store.delete_token("https://example.test") is False
    assert all("secret" not in str(call[:3]) for call in backend.calls)


def test_keyring_rejects_invalid_tokens_and_unusable_backend():
    backend = FakeBackend()
    store = KeyringCredentialStore(backend)
    with pytest.raises(CredentialError, match="invalid"):
        store.set_token("https://example.test", "gitlab-pat")

    class FailBackend(FakeBackend):
        priority = 0

    with pytest.raises(CredentialError, match="No usable OS keychain"):
        KeyringCredentialStore(FailBackend()).get_token("https://example.test")

    class PlaintextBackend(FakeBackend):
        pass

    class ChainerBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.backends = [FakeBackend(), PlaintextBackend()]

    with pytest.raises(CredentialError, match="No usable OS keychain"):
        KeyringCredentialStore(ChainerBackend()).get_token("https://example.test")


def test_keyring_errors_are_actionable_and_redacted():
    backend = FakeBackend()
    backend.error = RuntimeError("backend exploded with autofa_cli_secret")
    store = KeyringCredentialStore(backend)
    with pytest.raises(CredentialError) as error:
        store.get_token("https://example.test")
    assert "autofa_cli_secret" not in str(error.value)
    assert "OS keychain" in str(error.value)


def test_client_from_config_requires_named_environment_token(monkeypatch):
    cfg = Config(remote_library_base_url="https://library.example.org",
                 remote_library_token_env="CUSTOM_GITLAB_TOKEN")
    monkeypatch.delenv("CUSTOM_GITLAB_TOKEN", raising=False)
    with pytest.raises(RemoteLibraryError, match="CUSTOM_GITLAB_TOKEN") as error:
        client_from_config(cfg, credential_store=SimpleNamespace(get_token=lambda url: None))
    assert "autofa login" in str(error.value)


# A local dashboard selection stays isolated from cloud credentials and transport.


@contextmanager
def _http_service(*, redirect: str = ""):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if redirect:
                self.send_response(302)
                self.send_header("Location", redirect)
                self.end_headers()
                return
            body = json.dumps({"id": 1, "username": "local", "name": "Researcher"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                              daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _no_cloud_credentials(*args, **kwargs):
    pytest.fail("Local library operations must not access cloud credentials")


def test_local_client_uses_only_profile_token(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setenv("GITLAB_TOKEN", "cloud-environment-secret")
    monkeypatch.setenv("AUTOFA_LIBRARY_TOKEN", "other-environment-secret")
    token = service_token(create=True)
    store = SimpleNamespace(get_token=_no_cloud_credentials)
    cfg = Config(remote_library_base_url="http://127.0.0.1:8765/")

    client = client_from_config(cfg, credential_store=store)

    assert client.token == token
    assert client.credential_source == "local"
    assert client.base_url == "http://127.0.0.1:8765"
    assert (tmp_path / "server" / "token").is_file()


def test_missing_local_token_never_falls_back_to_cloud(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setenv("GITLAB_TOKEN", "cloud-environment-secret")
    cfg = Config(remote_library_base_url="http://127.0.0.1:8765")
    with pytest.raises(RemoteLibraryError, match="autofa serve"):
        client_from_config(cfg, credential_store=SimpleNamespace(get_token=_no_cloud_credentials))


# The two loopback validations share one parser; each malformed shape is checked once.
@pytest.mark.parametrize("url", ["https://127.0.0.1:8765", "http://user:password@localhost:8765"])
def test_malformed_local_config_rejects_before_reading_credentials(monkeypatch, url):
    monkeypatch.setattr("autofa.local_service.service_token", _no_cloud_credentials)
    monkeypatch.setattr(
        "autofa.data.remote_library_credentials.resolve_credential",
        _no_cloud_credentials,
    )
    with pytest.raises(RemoteLibraryError, match="HTTP loopback origin"):
        client_from_config(Config(remote_library_base_url=url))


@pytest.mark.parametrize("url", ["http://127.0.0.1:8765/path", "http://localhost.:8765"])
def test_explicit_token_client_rejects_malformed_loopback_origin(url):
    with pytest.raises(ValueError, match="HTTP loopback origin"):
        RemoteLibraryClient(url, token="explicit-secret")


@pytest.mark.parametrize("source", ["env", "keychain"])
def test_cloud_client_preserves_credential_resolution(monkeypatch, source):
    monkeypatch.setattr("autofa.local_service.service_token", _no_cloud_credentials)
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    calls = []
    store = SimpleNamespace(get_token=lambda url: calls.append(url) or "cloud-keychain-token")
    if source == "env":
        monkeypatch.setenv("GITLAB_TOKEN", "cloud-environment-token")
    client = client_from_config(
        Config(remote_library_base_url="HTTPS://Example.TEST:443/",
               remote_library_token_env="GITLAB_TOKEN"),
        credential_store=store,
    )
    assert client.credential_source == source
    assert client.base_url == "https://example.test"
    assert client.token == (
        "cloud-environment-token" if source == "env" else "cloud-keychain-token"
    )
    assert calls == ([] if source == "env" else ["https://example.test"])


def test_loopback_requests_ignore_environment_proxy(monkeypatch):
    with _http_service() as (proxy_url, proxy_requests), _http_service() as (url, requests):
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.setenv("NO_PROXY", "")
        identity = RemoteLibraryClient(url, token="local-secret", credential_source="local").me()
        assert identity.username == "local"
        assert requests == [("/api/v1/me", "Bearer local-secret")]
        assert proxy_requests == []


@pytest.mark.parametrize("download,explicit_local", [(False, True), (True, False)])
def test_local_api_and_download_reject_redirects(tmp_path, download, explicit_local):
    with _http_service() as (target_url, target_requests):
        with _http_service(redirect=f"{target_url}/secret") as (url, requests):
            client = RemoteLibraryClient(
                url,
                token="local-secret",
                credential_source="local" if explicit_local else "explicit",
            )
            with pytest.raises(RemoteLibraryError, match="cannot redirect"):
                if download:
                    client.download_library(
                        "library", tmp_path / "library.json", download_url=f"{url}/download"
                    )
                else:
                    client.me()
            assert len(requests) == 1
            assert target_requests == []
            assert list(tmp_path.iterdir()) == []


def test_local_client_rejects_foreign_download_origin(tmp_path):
    client = RemoteLibraryClient(
        "http://127.0.0.1:8765", token="local-secret", credential_source="local"
    )
    with pytest.raises(RemoteLibraryError, match="different host"):
        client.download_library("library", tmp_path / "library.json",
                                download_url="https://example.test/library.json")
