# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local library server: publication, dashboard, request boundaries, shared-server sign-in
and a dashboard lifetime that follows the CLI that starts it."""

from __future__ import annotations

import asyncio
import io
import json
import re
import socket
import threading
import time
import zipfile
from contextlib import asynccontextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from urllib.error import URLError
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rich.console import Console
from starlette.requests import Request

from autofa.config import Config
from autofa.data.remote_library_api import (
    RemoteLibraryClient,
    RemoteLibraryError,
    materialize_remote_library,
)
from autofa.local_service import (
    dashboard_url_from_config,
    is_local_service_url,
    service_root,
    service_token,
    valid_local_request,
    validate_service_url,
)
from autofa_library_server import local
from autofa_library_server.auth import _COOKIE, _SESSION_SECONDS
from autofa_library_server.config import Settings
from autofa_library_server.main import create_app

from support.library_server import LIBRARY, STORE, TEXT, _base, _client, _write_store
from support.library_server import local_app as local_app


def publish(browser, settings, *, branch="branch-a", library=None,
            report="<html><body>failure card details</body></html>", assets=None, expect=201):
    files = {
        "library": ("library.json", json.dumps(LIBRARY if library is None else library),
                    "application/json"),
        "failure_card": ("failure.html", report, "text/html"),
    }
    if assets is not None:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as stream:
            for name, content in assets.items():
                stream.writestr(name, content)
        files["report_assets"] = ("assets.zip", archive.getvalue(), "application/zip")
    response = browser.post(
        "/api/v2/publications",
        data={"metadata": json.dumps({
            "branch_key": branch, "source_run_id": "run-1", "model_id": "example-model",
            "benchmarks": ["example-benchmark"], "publish_trigger": "manual_cli",
            "data_identity": {"version": 1, "sources": [{"kind": "local", "sha256": "a" * 64}]},
        })},
        files=files,
        headers={"Authorization": f"Bearer {settings.local_token}"},
    )
    assert response.status_code == expect, response.text
    return response.json()


def unused_url(*, ipv6: bool = False) -> str:
    host = "::1" if ipv6 else "127.0.0.1"
    family = socket.AF_INET6 if ipv6 else socket.AF_INET
    with socket.socket(family) as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]
    return f"http://[{host}]:{port}" if ipv6 else f"http://{host}:{port}"


def assert_closed(url: str, original_threads=None) -> None:
    """The port is released and, when given the earlier threads, no dashboard worker remains."""
    parsed = urlsplit(url)
    with pytest.raises(OSError):
        with socket.create_connection((parsed.hostname, parsed.port), timeout=0.3):
            pass
    if original_threads is not None:
        assert not [thread for thread in threading.enumerate() if thread not in original_threads
                    and thread.name == "autofa-local-dashboard"]


def client_for(url: str) -> RemoteLibraryClient:
    return RemoteLibraryClient(url, token=service_token(), credential_source="local", timeout=1)


@pytest.fixture
def dashboard_url(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    return unused_url()


def test_publication_preserves_supported_detector_kinds(local_app):
    _, browser, settings = local_app
    library = deepcopy(LIBRARY)
    library["version"] = 2
    library["functions"][0]["kind"] = "llm"
    published = publish(browser, settings, library=library)
    downloaded = browser.get(published["library_url"])
    assert downloaded.status_code == 200
    assert downloaded.content == json.dumps(library).encode()


@pytest.mark.parametrize("version,kind", [(1, "unknown"), (2, None)])
def test_publication_rejects_invalid_detector_kinds(local_app, version, kind):
    _, browser, settings = local_app
    library = deepcopy(LIBRARY)
    library["version"] = version
    library["functions"][0]["kind"] = kind
    response = publish(browser, settings, library=library, expect=422)
    assert "Invalid function kind" in response["detail"]


def test_versions_dashboard_and_restart(local_app):
    app, browser, settings = local_app
    first = publish(browser, settings)
    second = publish(browser, settings)
    branch = publish(browser, settings, branch="branch-b")
    assert first["library_id"] == second["library_id"] == branch["library_id"]
    assert first["branch_id"] == second["branch_id"] != branch["branch_id"]
    assert first["version_id"] != second["version_id"]
    assert second["version_number"] == 2
    assert "example-model" in browser.get("/").text
    path = (
        f"/api/v2/libraries/{first['library_id']}/branches/{first['branch_id']}"
        f"/versions/{first['version_id']}"
    )
    assert browser.get(path + "/library.json").json() == LIBRARY
    detail = browser.get(f"/libraries/{first['library_id']}")
    assert detail.status_code == 200 and 'sandbox="allow-scripts"' in detail.text
    with TestClient(create_app(settings), base_url=settings.public_url) as restarted:
        assert restarted.get(path + "/library.json").json() == LIBRARY
        assert restarted.get(path + "/failure-card").text == (
            "<html><body>failure card details</body></html>"
        )
    assert settings.database_path.is_file()
    assert list((settings.data_dir / "artifacts").rglob("*"))


def test_existing_comment_data_does_not_prevent_library_restart(local_app):
    app, browser, settings = local_app
    saved = publish(browser, settings)
    # Simulate the retired table in an existing profile. Starting the new server
    # should leave it untouched while continuing to serve the library versions.
    with app.state.database.transaction() as connection:
        connection.execute(
            """
            CREATE TABLE report_comments (
                id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL
                    REFERENCES library_versions(id) ON DELETE CASCADE,
                author_id INTEGER
                    REFERENCES owners(id) ON UPDATE CASCADE ON DELETE SET NULL,
                author_username TEXT NOT NULL,
                body TEXT NOT NULL,
                target_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0 CHECK(resolved IN (0, 1))
            )
            """
        )
        connection.execute(
            """
            INSERT INTO report_comments
                (id, version_id, author_username, body, target_json, created_at, updated_at)
            VALUES ('old-comment', ?, 'local', 'Saved note', '{}', ?, ?)
            """,
            (saved["version_id"], saved["created_at"], saved["created_at"]),
        )
        original = dict(connection.execute("SELECT * FROM report_comments").fetchone())
    path = (
        f"/api/v2/libraries/{saved['library_id']}/branches/{saved['branch_id']}"
        f"/versions/{saved['version_id']}"
    )
    with TestClient(create_app(settings), base_url=settings.public_url) as restarted:
        assert restarted.get(path + "/library.json").json() == LIBRARY
        assert publish(restarted, settings)["version_number"] == 2
    with app.state.database.read() as connection:
        assert dict(connection.execute("SELECT * FROM report_comments").fetchone()) == original


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example:8765"}, {"Origin": "http://127.0.0.1:9876"},
    {"Sec-Fetch-Site": "cross-site"},
], ids=["host", "origin", "cross_site"])
def test_foreign_browser_requests_rejected(local_app, headers):
    _, browser, _ = local_app
    assert browser.get("/api/v1/me", headers=headers).status_code == 403


def test_mutation_needs_local_token_or_csrf(local_app):
    _, browser, _ = local_app
    for headers in ({}, {"X-CSRF-Token": "wrong"}, {"Authorization": "Bearer wrong"}):
        assert browser.delete("/api/v1/libraries/unknown", headers=headers).status_code in {
            401,
            403,
        }


@pytest.mark.parametrize("url", ["https://example.com", "http://127.0.0.1.evil.example:8765"])
def test_client_rejects_nonlocal_destinations(url):
    with pytest.raises(ValueError):
        RemoteLibraryClient(url, credential_source="local")
    assert not is_local_service_url(url)


def test_local_profile_token_and_config(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    first = service_token(create=True)
    assert first == service_token(create=True) == service_token()
    assert (tmp_path / "server/token").stat().st_mode & 0o777 == 0o600
    cfg = Config(remote_library_base_url="http://127.0.0.1:8765")
    assert validate_service_url(cfg.remote_library_base_url) == "http://127.0.0.1:8765"
    assert is_local_service_url(cfg.remote_library_base_url)
    assert service_root() == tmp_path / "server"


def test_real_http_client_publish_download_and_continue(dashboard_url):
    from autofa.cli import _continue_seed_from_remote

    with local.managed_dashboard(url=dashboard_url):
        client = RemoteLibraryClient(dashboard_url, token=service_token(),
                                     credential_source="local")
        publication = client.upload_library(
            metadata={"branch_key": "run-a", "source_run_id": "run-a",
                      "model_id": "example-model", "benchmarks": ["example-benchmark"]},
            library_json=LIBRARY,
            failure_cards_html="<html><body>evidence</body></html>",
        )
        assert len(client.list_libraries()) == 1
        selected = client.resolve_library_ref(publication.library_id)
        path = materialize_remote_library(selected, client=client)
        assert json.loads(path.read_text()) == LIBRARY
        seed = _continue_seed_from_remote(path, selected)
        assert seed.remote_branch_key == "run-a"
        assert seed.remote_version_id == publication.version_id
        assert "example-model" in path.read_text()
    assert_closed(dashboard_url)


def test_sandboxed_report_loads_versioned_trajectory_asset(local_app):
    _, browser, settings = local_app
    payload = {"meta": {"items_base": "items"}, "item_index": {"row-1": "row-1.json"}}
    publication = publish(
        browser, settings,
        report='<script id="autofa-data" type="application/json">' + json.dumps(payload)
        + "</script>",
        assets={"items/row-1.json": json.dumps({"steps": [{"text": "evidence"}]})},
    )
    report = browser.get(f"/libraries/{publication['library_id']}/report")
    assert report.status_code == 200, report.text
    payload = json.loads(
        re.search(
            r'<script id="autofa-data" type="application/json">(.*?)</script>',
            report.text,
        ).group(1)
    )
    asset_url = payload["meta"]["items_base"] + "/row-1.json"
    response = browser.get(asset_url, headers={"Origin": "null", "Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 200, response.text
    assert response.json()["steps"][0]["text"] == "evidence"
    invalid = browser.get("/report-assets/invalid/invalid/items/row-1.json",
                          headers={"Origin": "null", "Sec-Fetch-Site": "cross-site"})
    assert invalid.status_code == 404


@pytest.fixture
def legacy_report(local_app):
    from autofa.core.library import FunctionLibrary
    from autofa.core.localisation import detector_hash
    from autofa.data.trace_navigation import NAVIGATION_VERSION
    from autofa.reports.failure_cards import build_data

    _, client, settings = local_app
    library = deepcopy(LIBRARY)
    fn = library["functions"][0]
    fn["scores"] = {
        "evaluated": True, "n_items": 10, "a": 2, "b": 0, "c": 3, "d": 5,
        "recall": .4, "precision": 1, "ppv_at_pi0": 1,
        "behavior_purity": .8, "behavior_specificity": .8,
        "judge_backend": "claude_agent_sdk", "behavior_n_judged": 5, "behavior_n_yes": 4,
        "judge_rationale": "The detector catches repeated errors but misses recovery.",
        "buckets": [{"bucket": "True", "n": 2, "n_fail": 2, "n_pass": 0,
                     "purity": 1, "example_fail_ids": ["row-1"]}],
    }
    fn["failure_localisations"] = {"row-1": {
        "detector_hash": detector_hash(fn["source"]), "navigation_version": NAVIGATION_VERSION,
        "passages": [{
            "field": "response_text", "quote": "Saved", "start": 0, "end": 5,
            "context": "Saved full example trajectory", "context_start": 0, "context_end": 29,
            "field_chars": 29, "start_line": 1, "start_column": 1,
            "end_line": 1, "end_column": 6, "method": "literal",
            "trace_locations": [{"step_index": 56, "section": "observation",
                                 "source_event": {"event_idx": 55, "part_idx": 0, "lane": ""}}],
        }],
    }}
    item = {
        "item_id": "row-1", "response_head": "Saved full example trajectory",
        "source_data_url": "https://traces.example.test/#/rollout/source-run/task/0",
    }
    data = build_data(FunctionLibrary.from_dict(library), items={"row-1": item})
    # The saved report already contains its verified link; displaying it must
    # work without regenerating URLs through the original input adapter.
    location = data["functions"][0]["failure_localisations"]["row-1"]["passages"][0]
    location["trace_locations"][0]["source_url"] = (
        "https://traces.example.test/#/rollout/source-run/task/0/e55-p0?lane="
    )
    data["items"] = {}
    data["meta"].update(items_base="items", n_embedded_items=1)
    data["item_index"] = {"row-1": "row-1.json"}
    data["functions"][0].pop("card_id")
    data["functions"][0].pop("behavior_validation")
    # A persisted page from an older generator, with its own obsolete UI and script.
    report = (
        '<html><body><p>GATED RECALL · Example fail ids · fisher p</p>'
        '<script>document.body.className = "theme-dark";</script>'
        '<script id="autofa-data" type="application/json">'
        + json.dumps(data) + '</script></body></html>'
    ).encode()
    saved = publish(client, settings, branch="old", library=library, report=report,
                    assets={"items/row-1.json": json.dumps(item)})
    return client, saved, report, data


def test_saved_report_uses_current_ui_preserving_artifacts_assets_and_cache(legacy_report):
    from autofa.reports.failure_cards import _extract_autofa_data

    client, saved, original, before = legacy_report
    page_url = f"/libraries/{saved['library_id']}"
    selectors = {"branch_id": saved["branch_id"], "version_id": saved["version_id"]}
    detail = client.get(page_url, params=selectors)
    from autofa_library_server.main import REPORT_VIEW_VERSION

    assert f"view={REPORT_VIEW_VERSION}" in detail.text
    response = client.get(page_url + "/report", params=selectors)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert 'id="failure-index"' in response.text
    assert "GATED RECALL" not in response.text and "Example fail ids" not in response.text
    data = _extract_autofa_data(response.text)
    assert data["meta"]["n_items"] == before["meta"]["n_items"]
    assert data["item_index"] == before["item_index"]
    assert data["functions"][0]["metrics"]["recall"] == .4
    assert data["functions"][0]["behavior_validation"]["status"] == "legacy"
    asset_url = data["meta"]["items_base"] + "/row-1.json"
    assert client.get(asset_url).json()["response_head"] == "Saved full example trajectory"
    unchanged = client.get(page_url + "/report", params=selectors,
                           headers={"If-None-Match": response.headers["ETag"]})
    assert unchanged.status_code == 304
    assert unchanged.headers["Cache-Control"] == "no-store"
    artifact_url = (f"/api/v2/libraries/{saved['library_id']}/branches/{saved['branch_id']}"
                    f"/versions/{saved['version_id']}/failure-card")
    assert client.get(artifact_url).content == original


# `autofa serve --host`: beyond loopback, any Host on the port is the server's own,
# and an Origin must be that same host.

PUBLIC = "http://127.0.0.1:8000"


def _from_host(host: str, *, origin: str | None = None, method: str = "GET", site: str = ""):
    headers = [(b"host", host.encode())]
    if origin:
        headers.append((b"origin", origin.encode()))
    if site:
        headers.append((b"sec-fetch-site", site.encode()))
    name, _, port = host.rpartition(":")
    return Request({"type": "http", "method": method, "path": "/", "query_string": b"",
                    "headers": headers, "scheme": "http", "server": (name, int(port or 80))})


@pytest.mark.parametrize("host,origin,method,site,remote_hosts,expected", [
    ("127.0.0.1:8000", None, "GET", "", False, True),
    ("10.0.0.5:8000", None, "GET", "", False, False),
    ("10.0.0.5:8000", None, "GET", "", True, True),
    ("10.0.0.5:8000", "http://10.0.0.5:8000", "POST", "", True, True),
    ("10.0.0.5:9000", None, "GET", "", True, False),
    ("10.0.0.5:8000", "http://evil.example:8000", "POST", "", True, False),
    ("10.0.0.5:8000", None, "GET", "cross-site", True, False),
    ("user:pw@10.0.0.5:8000", None, "GET", "", True, False),
], ids=["loopback", "remote_by_default", "remote_host", "same_origin_post", "other_port",
        "foreign_origin", "cross_site", "userinfo"])
def test_remote_hosts_accept_only_the_machine_s_address_on_the_port(
    host, origin, method, site, remote_hosts, expected,
):
    request = _from_host(host, origin=origin, method=method, site=site)
    assert valid_local_request(request, PUBLIC, remote_hosts) is expected


# A remotely bound shared server requires a token before exposing saved data.

SHARED_TOKEN = "shared-server-token-" + "a" * 48
SHARED_BASE = "http://reports.example.test:8765"


@pytest.fixture
def shared_server(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    _write_store(tmp_path)
    settings = Settings(data_dir=tmp_path / "server", local_token=SHARED_TOKEN, remote_hosts=True)
    app = create_app(settings)
    with _client(app, SHARED_BASE) as client:
        yield app, client, settings


def _login(client, *, token=SHARED_TOKEN, target="/"):
    return client.post("/login", data={"token": token, "next": target}, follow_redirects=False)


def test_anonymous_remote_browser_and_api_cannot_read_reports_or_traces(shared_server):
    _, client, _ = shared_server
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"].startswith("/login?")
    login = client.get(response.headers["location"])
    assert login.status_code == 200 and SHARED_TOKEN not in login.text
    assert "frame-ancestors 'none'" in login.headers["content-security-policy"]
    for path in ("/api/v1/me", "/api/v2/libraries", _base() + "/events",
                 _base() + "/artifacts/0/view", "/trace-static/trace.js"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 401, path
        assert SHARED_TOKEN not in response.text and "TAIL EVIDENCE" not in response.text
    assert client.get("/healthz").status_code == 200


def test_login_uses_signed_cookie_and_separate_csrf_then_serves_full_trace(shared_server):
    _, client, _ = shared_server
    response = _login(client, target=f"/traces/{STORE}/")
    assert response.status_code == 303
    assert response.headers["location"] == f"/traces/{STORE}/"
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and SHARED_TOKEN not in cookie
    assert client.get("/api/v1/me").json()["auth_method"] == "session"
    assert client.get(f"/traces/{STORE}/").status_code == 200
    full = client.get(_base() + "/part-text", params={
        "event_idx": 55, "part_idx": 7, "lane": "conversation",
    })
    assert full.json()["text"] == TEXT
    assert client.get(_base() + "/artifacts/0/view").content.startswith(b"\x89PNG")
    dashboard = client.get("/")
    assert SHARED_TOKEN not in dashboard.text
    assert client.post("/logout").status_code == 403
    assert client.post("/libraries/missing/delete", data={"csrf_token": "bad"}).status_code == 403
    csrf = client.get("/api/v1/me").json().get("csrf_token")
    # The API intentionally need not reveal CSRF; the signed cookie context owns it.
    context = shared_server[0].state.sessions._cookie_context(client.cookies.get(_COOKIE))
    assert context is not None and context.csrf_token and context.csrf_token != SHARED_TOKEN
    assert not csrf or csrf != SHARED_TOKEN
    response = client.post("/logout", headers={"X-CSRF-Token": context.csrf_token},
                           follow_redirects=False)
    assert response.status_code == 303 and not client.cookies.get(_COOKIE)
    assert client.get("/api/v1/me").status_code == 401


def test_bearer_remains_valid_and_bad_login_does_not_reflect_password(shared_server):
    _, client, _ = shared_server
    response = client.get("/api/v1/me", headers={"Authorization": f"Bearer {SHARED_TOKEN}"})
    assert response.status_code == 200 and response.json()["auth_method"] == "local_token"
    response = _login(client, token="incorrect-secret-🙂")
    assert response.status_code == 401 and "incorrect-secret" not in response.text
    assert not client.cookies
    assert client.get("/api/v1/me", headers={"Authorization": "Bearer bad"}).status_code == 401
    assert _login(client, target="https://foreign.test").headers["location"] == "/"


def test_login_rejects_foreign_origin_and_oversized_input(shared_server):
    _, client, _ = shared_server
    response = client.post("/login", data={"token": SHARED_TOKEN},
                           headers={"Origin": "http://foreign.test:8765"})
    assert response.status_code == 403 and not client.cookies
    assert client.post("/login", content=b"token=" + b"a" * 8192).status_code == 413
    assert client.post("/login", content=b"token=\xff").status_code == 400


def test_cookies_cannot_be_forged_and_expire_or_rotate_with_token(shared_server, monkeypatch):
    app, client, settings = shared_server
    _login(client)
    original = client.cookies.get(_COOKIE)
    assert original and SHARED_TOKEN not in original
    client.cookies.clear()
    client.cookies.set(_COOKIE, original + "altered")
    assert client.get("/api/v1/me").status_code == 401
    rotated_app = create_app(replace(settings, local_token="new-" + SHARED_TOKEN))
    with _client(rotated_app, SHARED_BASE) as rotated:
        rotated.cookies.set(_COOKIE, original)
        assert rotated.get("/api/v1/me").status_code == 401
    later = time.time() + _SESSION_SECONDS + 1
    monkeypatch.setattr("autofa_library_server.auth.time.time", lambda: later)
    assert app.state.sessions._cookie_context(original) is None


def test_report_sidecar_capabilities_still_validate_without_browser_cookie(shared_server):
    _, client, _ = shared_server
    # Sandboxed report fetches carry Origin:null and no ambient browser cookie;
    # their version token is independently required by the asset route.
    path = "/report-assets/not-a-version/not-a-token/failure_cards_data/items/one.json"
    response = client.get(path, headers={"Origin": "null"})
    assert response.status_code == 404


def test_loopback_profile_still_opens_without_login(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    settings = Settings(data_dir=tmp_path / "server", local_token=SHARED_TOKEN)
    with _client(create_app(settings), settings.public_url) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/v1/me").json()["auth_method"] == "session"


# Dashboard lifetime follows the CLI that starts it.


@pytest.mark.parametrize("error", [
    TimeoutError("first response is slow"),
    RemoteLibraryError("Network error"),
], ids=["read_timeout", "wrapped_network"])
def test_owned_dashboard_retries_transient_readiness_failures(dashboard_url, monkeypatch, error):
    original_me = RemoteLibraryClient.me
    calls = []

    def interrupted_probe(client):
        calls.append(client.timeout)
        if len(calls) == 1:
            raise error from URLError("listener not ready")
        return original_me(client)

    monkeypatch.setattr(RemoteLibraryClient, "me", interrupted_probe)
    with local.managed_dashboard(url=dashboard_url, startup_timeout=2) as owned:
        assert owned is True
        assert len(calls) == 2
    assert_closed(dashboard_url)


@pytest.mark.parametrize("failure", ["http", "identity"])
def test_owned_dashboard_fails_fast_on_invalid_authentication(
    dashboard_url, monkeypatch, failure
):
    from fastapi.responses import JSONResponse

    from autofa_library_server import main

    app = FastAPI()
    requests = []

    @app.get("/api/v1/me")
    def identity():
        requests.append("identity")
        if failure == "http":
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return {"id": 1, "username": "local", "name": "Local", "auth_method": "session"}

    monkeypatch.setattr(main, "create_app", lambda settings: app)
    with pytest.raises(RuntimeError) as caught:
        with local.managed_dashboard(url=dashboard_url, startup_timeout=2):
            pytest.fail("An unauthenticated dashboard must not be ready")
    assert "timed out" not in str(caught.value)
    assert requests == ["identity"]
    assert_closed(dashboard_url)


def test_worker_death_during_http_readiness_releases_port(dashboard_url, monkeypatch):
    failure = RuntimeError("Dashboard worker failed during readiness")
    probes = []

    async def failed_server(server, sockets):
        server.started = True
        raise failure

    def failed_probe(client):
        probes.append(client.timeout)
        # If the main thread reaches the first probe before the worker has
        # finished unwinding, leave it enough time to report its failure.
        for thread in threading.enumerate():
            if thread.name == "autofa-local-dashboard":
                thread.join(timeout=1)
        raise ConnectionRefusedError("worker exited")

    monkeypatch.setattr(local._CatalogServer, "serve", failed_server)
    monkeypatch.setattr(RemoteLibraryClient, "me", failed_probe)
    with pytest.raises(RuntimeError, match="startup failed") as caught:
        with local.managed_dashboard(url=dashboard_url, startup_timeout=2):
            pytest.fail("A stopped dashboard worker cannot be ready")
    assert caught.value.__cause__ is failure
    assert len(probes) <= 1
    assert_closed(dashboard_url)


def test_owned_dashboard_stays_responsive_and_closes_during_report_indexing(
    dashboard_url, tmp_path, monkeypatch, caplog, capsys
):
    import uvicorn

    from autofa_library_server import catalog, main

    run = tmp_path / "runs" / "saved-run"
    run.mkdir(parents=True)
    (run / "library.json").write_text(
        json.dumps(
            {"version": 1, "model_id": "saved-model", "benchmark": "test", "functions": []}
        )
    )
    asset = run / "items" / "example.json"
    asset.parent.mkdir()
    asset.write_text('{"steps": [{"text": "saved evidence"}]}')
    # Defer the large report so the initial saved-library snapshot can be listed
    # while the background worker is busy preparing its richer report.
    (run / "failure_cards.html").write_text(
        '<html><script id="autofa-data" type="application/json">'
        + json.dumps(
            {"meta": {"items_base": "items"}, "item_index": {"example": "example.json"}}
        )
        + "</script><!--"
        + "x" * (1024 * 1024)
        + "--></html>"
    )
    entered_asset_read = threading.Event()
    release_asset_read = threading.Event()
    original_threads = set(threading.enumerate())
    applications = []
    stopped_before_drain = []
    original_create_app = main.create_app
    original_read = catalog._bounded_read
    original_shutdown = uvicorn.Server.shutdown

    def capture_app(settings):
        app = original_create_app(settings)
        applications.append(app)
        return app

    def slow_asset_read(path, maximum):
        if path == asset:
            entered_asset_read.set()
            if not release_asset_read.wait(timeout=15):
                raise TimeoutError("test did not release the paused asset read")
        return original_read(path, maximum)

    async def inspect_shutdown(server, *args, **kwargs):
        stopped_before_drain.append(applications[0].state.local_catalog._stopped.is_set())
        await original_shutdown(server, *args, **kwargs)

    monkeypatch.setattr(main, "create_app", capture_app)
    monkeypatch.setattr(catalog, "_bounded_read", slow_asset_read)
    monkeypatch.setattr(uvicorn.Server, "shutdown", inspect_shutdown)
    try:
        with local.managed_dashboard(url=dashboard_url):
            assert entered_asset_read.wait(timeout=5)
            saved_catalog = applications[0].state.local_catalog
            # The pause is inside the real _report_assets call while it owns the
            # catalog lock, reproducing the old request-path refresh deadlock.
            acquired = saved_catalog._scan_lock.acquire(blocking=False)
            if acquired:
                saved_catalog._scan_lock.release()
            assert not acquired
            with httpx.Client(base_url=dashboard_url, trust_env=False, timeout=1) as browser:
                started = time.monotonic()
                for path in ("/", "/api/v2/libraries"):
                    response = browser.get(path)
                    assert response.status_code == 200, response.text
                    assert "saved-model" in response.text
                assert time.monotonic() - started < 2
            # Exit with report processing still paused. Shutdown must neither
            # wait for its scan lock nor cancel otherwise healthy HTTP requests.
            started_shutdown = time.monotonic()
        assert time.monotonic() - started_shutdown < 3
        assert stopped_before_drain == [True]
        assert_closed(dashboard_url, original_threads)
    finally:
        release_asset_read.set()
        if applications:
            applications[0].state.local_catalog.close()
    captured = capsys.readouterr()
    logs = caplog.text + captured.out + captured.err
    assert "timeout graceful shutdown exceeded" not in logs
    assert "Exception in ASGI application" not in logs
    assert "CancelledError" not in logs


def test_wrong_profile_cannot_reuse_or_stop_dashboard(dashboard_url, tmp_path, monkeypatch):
    with local.managed_dashboard(url=dashboard_url):
        owner_client = client_for(dashboard_url)
        with monkeypatch.context() as other_profile:
            other_profile.setenv("AUTOFA_HOME", str(tmp_path / "other-profile"))
            service_token(create=True)
            with pytest.raises(RuntimeError):
                with local.managed_dashboard(url=dashboard_url, startup_timeout=0.5):
                    pytest.fail("A different profile must not reuse the dashboard")
        assert owner_client.me().auth_method == "local_token"
    assert_closed(dashboard_url)


def test_unrelated_http_listener_is_left_running(dashboard_url):
    service_token(create=True)

    class UnrelatedHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            # A shaped identity alone does not establish token authentication.
            payload = json.dumps(
                {"id": 1, "username": "local", "name": "Local researcher", "auth_method": "session"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    parsed = urlsplit(dashboard_url)
    with HTTPServer((parsed.hostname, parsed.port), UnrelatedHandler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                                  daemon=True)
        thread.start()
        try:
            start = time.monotonic()
            with pytest.raises(RuntimeError):
                with local.managed_dashboard(url=dashboard_url, startup_timeout=0.5):
                    pytest.fail("An unrelated service must not be reused")
            assert time.monotonic() - start < 5
            assert thread.is_alive()
            assert client_for(dashboard_url).me().auth_method == "session"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_timed_out_startup_cancels_worker_and_releases_port(dashboard_url, monkeypatch):
    from autofa_library_server import main

    entered = threading.Event()
    cancelled = threading.Event()
    original_threads = set(threading.enumerate())

    @asynccontextmanager
    async def suspended_lifespan(app):
        entered.set()
        try:
            await asyncio.Event().wait()
            yield  # pragma: no cover
        finally:
            cancelled.set()

    monkeypatch.setattr(main, "create_app", lambda settings: FastAPI(lifespan=suspended_lifespan))
    with pytest.raises(RuntimeError, match="timed out"):
        with local.managed_dashboard(url=dashboard_url, startup_timeout=0.1):
            pytest.fail("A suspended application cannot be ready")
    assert entered.is_set()
    assert cancelled.is_set()
    assert_closed(dashboard_url, original_threads)


@pytest.fixture
def configured_cli(dashboard_url, tmp_path, monkeypatch):
    from autofa import cli
    from autofa.util import memory

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "ensure_default_config", lambda **kwargs: tmp_path / "config.toml")
    monkeypatch.setattr(cli, "load_config", lambda: Config(remote_library_base_url=dashboard_url))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(
        memory, "apply_memory_tuning", lambda cfg: SimpleNamespace(summary=lambda: "test")
    )
    return cli


def test_cli_keeps_dashboard_through_action_finalization(
    configured_cli, dashboard_url, monkeypatch,
):
    events = []

    def action(*args, **kwargs):
        client = client_for(dashboard_url)
        try:
            assert client.me().auth_method == "local_token"
            events.append("action")
        finally:
            assert client.me().auth_method == "local_token"
            events.append("finalization")

    monkeypatch.setattr(configured_cli, "_dispatch", action)
    configured_cli.main(["publish", "--once"])
    assert events == ["action", "finalization"]
    assert_closed(dashboard_url)


@pytest.mark.parametrize(
    "error,exit_code", [(RuntimeError("failed"), 1), (KeyboardInterrupt(), 130)]
)
def test_cli_failed_action_stops_dashboard(
    configured_cli, dashboard_url, monkeypatch, error, exit_code
):
    def action(*args):
        assert client_for(dashboard_url).me().auth_method == "local_token"
        raise error

    monkeypatch.setattr(configured_cli, "_dispatch", action)
    with pytest.raises(SystemExit) as caught:
        configured_cli.main(["new", "--once"])
    assert caught.value.code == exit_code
    assert_closed(dashboard_url)


def test_cli_warns_and_continues_if_dashboard_is_unavailable(
    configured_cli, dashboard_url, monkeypatch,
):
    calls = []

    def unavailable(**kwargs):
        raise RuntimeError("The configured port is occupied")

    monkeypatch.setattr(local, "managed_dashboard", unavailable)
    monkeypatch.setattr(configured_cli, "console", Console(file=io.StringIO()))
    monkeypatch.setattr(configured_cli, "_home", lambda *args, **kwargs: calls.append("home"))
    configured_cli.main(["home", "--once"])
    assert calls == ["home"]
    assert_closed(dashboard_url)


@pytest.mark.parametrize("argv,exit_code", [(["--help"], 0), (["unknown"], 2)])
def test_cli_help_and_unknown_command_do_not_start_dashboard(
    configured_cli, monkeypatch, argv, exit_code
):
    def unexpected_start(**kwargs):
        pytest.fail("Argument handling must not start the dashboard")

    monkeypatch.setattr(local, "managed_dashboard", unexpected_start)
    with pytest.raises(SystemExit) as caught:
        configured_cli.main(argv)
    assert caught.value.code == exit_code


def test_cli_serve_remains_independent(configured_cli, monkeypatch):
    calls = []

    def unexpected_start(**kwargs):
        pytest.fail("The standalone serve command must not use automatic startup")

    monkeypatch.setattr(local, "managed_dashboard", unexpected_start)
    monkeypatch.setattr(local, "serve", lambda **kwargs: calls.append(kwargs))
    configured_cli.main(["serve", "--port", "9876"])
    assert calls == [{"port": 9876, "host": "127.0.0.1"}]


@pytest.mark.parametrize("standalone", [False, True])
def test_cli_starts_or_reuses_local_browser_with_separate_publication_service(
    configured_cli, dashboard_url, monkeypatch, standalone,
):
    config = Config(remote_library_base_url="https://publication.example.test",
                    local_library_base_url=dashboard_url)
    monkeypatch.setattr(configured_cli, "load_config", lambda: config)

    def action(args, *, status_lines):
        assert client_for(dashboard_url).me().auth_method == "local_token"
        assert args._local_library_url == dashboard_url
        assert config.remote_library_base_url == "https://publication.example.test"

    monkeypatch.setattr(configured_cli, "_home", action)
    with local.managed_dashboard(url=dashboard_url) if standalone else nullcontext():
        configured_cli.main(["home", "--once"])
        if standalone:
            assert client_for(dashboard_url).me().auth_method == "local_token"
    assert_closed(dashboard_url)


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({"remote_library_base_url": "https://publication.example.test"},
         "http://127.0.0.1:8765"),
        ({"remote_library_base_url": "http://localhost:9988/"}, "http://localhost:9988"),
        ({"remote_library_base_url": "https://publication.example.test",
          "local_library_base_url": "http://127.0.0.1:8766/"}, "http://127.0.0.1:8766"),
        # A tunnel to a remote shared server does not select the local dashboard address.
        ({"remote_library_mode": "server", "remote_library_base_url": "http://127.0.0.1:9876"},
         "http://127.0.0.1:8765"),
    ],
)
def test_local_browser_address_resolution(settings, expected):
    assert dashboard_url_from_config(Config(**settings)) == expected


def test_local_browser_rejects_nonloopback_address():
    with pytest.raises(ValueError, match="HTTP loopback"):
        dashboard_url_from_config(Config(local_library_base_url="https://remote.example.test"))
