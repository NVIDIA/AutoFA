# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated client and immutable cache for the AutoFA library v2 API."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from autofa.config import remote_library_cache_root
from autofa.defaults import default_value, setting_value

DEFAULT_TIMEOUT = 60.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_LIBRARY_BYTES = 32 * 1024 * 1024
MAX_REPORT_BYTES = 128 * 1024 * 1024
MAX_REPORT_ASSETS_BYTES = 64 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 256 * 1024
MAX_AUTH_RESPONSE_BYTES = 64 * 1024
_CLI_FLOW_ID_RE = re.compile(r"^[A-Za-z0-9_-]{40,128}$")
_CLI_VERIFIER_RE = re.compile(r"^[A-Za-z0-9._~-]{43,128}$")
_CLI_TOKEN_RE = re.compile(r"^autofa_cli_[A-Za-z0-9_-]{32,}$")


class RemoteLibraryError(RuntimeError):
    """HTTP, network, or protocol error from the remote library service."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class PkcePair:
    verifier: str = field(repr=False)
    challenge: str


def generate_pkce_pair(
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
) -> PkcePair:
    """Generate a high-entropy RFC7636 S256 verifier/challenge pair."""
    verifier = base64.urlsafe_b64encode(random_bytes(64)).rstrip(b"=").decode("ascii")
    if _CLI_VERIFIER_RE.fullmatch(verifier) is None:
        raise RemoteLibraryError("Failed to generate a valid PKCE verifier")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return PkcePair(verifier=verifier, challenge=challenge)


@dataclass(frozen=True)
class RemoteIdentity:
    """Authenticated AutoFA library identity."""

    id: int
    username: str
    name: str
    avatar_url: str = ""
    web_url: str = ""
    email: str = ""
    auth_method: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemoteIdentity:
        data = dict(value)
        identity_id = _positive_int(data.get("id"))
        username = _text(data.get("username"))
        name = _text(data.get("name"))
        if identity_id is None or not username or not name:
            raise RemoteLibraryError("Identity response is missing id, username, or name")
        return cls(
            id=identity_id,
            username=username,
            name=name,
            avatar_url=_text(data.get("avatar_url")),
            web_url=_text(data.get("web_url")),
            email=_text(data.get("email")),
            auth_method=_text(data.get("auth_method")),
        )


@dataclass(frozen=True)
class RemoteCliAuthStart:
    flow_id: str
    login_url: str
    expires_in: int
    interval: int


@dataclass(frozen=True)
class RemoteCliTokenPending:
    interval: int


@dataclass(frozen=True)
class RemoteCliToken:
    token: str = field(repr=False)
    identity: RemoteIdentity
    expires_at: str = ""
    expires_in: int = 0
    token_type: str = "bearer"


@dataclass(frozen=True)
class RemoteDataReference:
    """One safe Data API input attached to an immutable report version."""

    run_id: str
    export_url: str
    source_harness: str = "unknown"
    inspect_url: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemoteDataReference:
        data = dict(value)
        run_id = _text(data.get("run_id"))
        export_url = _text(data.get("export_url"))
        if not run_id or not export_url:
            raise RemoteLibraryError(
                "Reference data is missing run_id or export_url"
            )
        return cls(
            run_id=run_id,
            export_url=export_url,
            source_harness=_text(data.get("source_harness")) or "unknown",
            inspect_url=_text(data.get("inspect_url")),
        )

    def to_dict(self) -> dict[str, str]:
        result = {
            "run_id": self.run_id,
            "export_url": self.export_url,
            "source_harness": self.source_harness,
        }
        if self.inspect_url:
            result["inspect_url"] = self.inspect_url
        return result


@dataclass(frozen=True)
class RemoteLibrary:
    """One logical library, carrying metadata from its latest version."""

    id: str
    owner_id: int = 0
    owner_username: str = ""
    owner_name: str = ""
    owner_avatar: str = ""
    source_run_id: str = ""
    model_id: str = ""
    harness: str = ""
    source_harness: str = "unknown"
    reference_data: tuple[RemoteDataReference, ...] = ()
    publish_trigger: str = ""
    benchmarks: tuple[str, ...] = ()
    function_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    revision: int = 0
    library_sha256: str = ""
    report_sha256: str = ""
    latest_branch_id: str = ""
    latest_version_id: str = ""
    branch_count: int = 0
    version_count: int = 0
    dashboard_url: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemoteLibrary:
        data = dict(value)
        library_id = _text(data.get("id"))
        if not library_id:
            raise RemoteLibraryError("Library metadata is missing an id")
        return cls(
            id=library_id,
            owner_id=_nonnegative_int(data.get("owner_id")),
            owner_username=_text(data.get("owner_username")),
            owner_name=_text(data.get("owner_name")),
            owner_avatar=_text(data.get("owner_avatar")),
            source_run_id=_text(data.get("source_run_id")),
            model_id=_text(data.get("model_id")),
            harness=_text(data.get("harness")),
            source_harness=_text(data.get("source_harness")) or "unknown",
            reference_data=_reference_data_tuple(data.get("reference_data")),
            publish_trigger=_text(data.get("publish_trigger")),
            benchmarks=_string_tuple(data.get("benchmarks")),
            function_count=_nonnegative_int(data.get("function_count")),
            created_at=_text(data.get("created_at")),
            updated_at=_text(data.get("updated_at")),
            revision=_nonnegative_int(data.get("revision")),
            library_sha256=_text(data.get("library_sha256")),
            report_sha256=_text(data.get("report_sha256")),
            latest_branch_id=_text(data.get("latest_branch_id")),
            latest_version_id=_text(data.get("latest_version_id")),
            branch_count=_nonnegative_int(data.get("branch_count")),
            version_count=_nonnegative_int(data.get("version_count")),
            dashboard_url=_text(data.get("dashboard_url")),
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "owner_username": self.owner_username,
            "owner_name": self.owner_name,
            "owner_avatar": self.owner_avatar,
            "source_run_id": self.source_run_id,
            "model_id": self.model_id,
            "harness": self.harness,
            "source_harness": self.source_harness,
            "reference_data": [item.to_dict() for item in self.reference_data],
            "publish_trigger": self.publish_trigger,
            "benchmarks": list(self.benchmarks),
            "function_count": self.function_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
            "library_sha256": self.library_sha256,
            "report_sha256": self.report_sha256,
            "latest_branch_id": self.latest_branch_id,
            "latest_version_id": self.latest_version_id,
            "branch_count": self.branch_count,
            "version_count": self.version_count,
            "dashboard_url": self.dashboard_url,
        }


@dataclass(frozen=True)
class RemoteBranch:
    """One stable publication branch within a logical library."""

    id: str
    library_id: str
    branch_key: str
    created_at: str = ""
    updated_at: str = ""
    latest_version_id: str = ""
    latest_version_number: int = 0
    latest_source_run_id: str = ""
    version_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemoteBranch:
        data = dict(value)
        branch_id = _text(data.get("id"))
        library_id = _text(data.get("library_id"))
        branch_key = _text(data.get("branch_key"))
        if not branch_id or not library_id or not branch_key:
            raise RemoteLibraryError("Branch metadata is missing id, library_id, or branch_key")
        return cls(
            id=branch_id,
            library_id=library_id,
            branch_key=branch_key,
            created_at=_text(data.get("created_at")),
            updated_at=_text(data.get("updated_at")),
            latest_version_id=_text(data.get("latest_version_id")),
            latest_version_number=_nonnegative_int(data.get("latest_version_number")),
            latest_source_run_id=_text(data.get("latest_source_run_id")),
            version_count=_nonnegative_int(data.get("version_count")),
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "library_id": self.library_id,
            "branch_key": self.branch_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "latest_version_id": self.latest_version_id,
            "latest_version_number": self.latest_version_number,
            "latest_source_run_id": self.latest_source_run_id,
            "version_count": self.version_count,
        }


@dataclass(frozen=True)
class RemoteVersion:
    """One immutable concrete library version."""

    id: str
    library_id: str
    branch_id: str
    branch_key: str
    version_number: int = 0
    source_run_id: str = ""
    model_id: str = ""
    harness: str = ""
    source_harness: str = "unknown"
    reference_data: tuple[RemoteDataReference, ...] = ()
    publish_trigger: str = ""
    benchmarks: tuple[str, ...] = ()
    function_count: int = 0
    created_at: str = ""
    library_sha256: str = ""
    report_sha256: str = ""
    library_url: str = ""
    failure_card_url: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemoteVersion:
        data = dict(value)
        version_id = _text(data.get("id"))
        library_id = _text(data.get("library_id"))
        branch_id = _text(data.get("branch_id"))
        branch_key = _text(data.get("branch_key"))
        if not version_id or not library_id or not branch_id:
            raise RemoteLibraryError(
                "Version metadata is missing id, library_id, or branch_id"
            )
        return cls(
            id=version_id,
            library_id=library_id,
            branch_id=branch_id,
            branch_key=branch_key,
            version_number=_nonnegative_int(data.get("version_number")),
            source_run_id=_text(data.get("source_run_id")),
            model_id=_text(data.get("model_id")),
            harness=_text(data.get("harness")),
            source_harness=_text(data.get("source_harness")) or "unknown",
            reference_data=_reference_data_tuple(data.get("reference_data")),
            publish_trigger=_text(data.get("publish_trigger")),
            benchmarks=_string_tuple(data.get("benchmarks")),
            function_count=_nonnegative_int(data.get("function_count")),
            created_at=_text(data.get("created_at")),
            library_sha256=_text(data.get("library_sha256")),
            report_sha256=_text(data.get("report_sha256")),
            library_url=_text(data.get("library_url")),
            failure_card_url=_text(data.get("failure_card_url")),
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "library_id": self.library_id,
            "branch_id": self.branch_id,
            "branch_key": self.branch_key,
            "version_number": self.version_number,
            "source_run_id": self.source_run_id,
            "model_id": self.model_id,
            "harness": self.harness,
            "source_harness": self.source_harness,
            "reference_data": [item.to_dict() for item in self.reference_data],
            "publish_trigger": self.publish_trigger,
            "benchmarks": list(self.benchmarks),
            "function_count": self.function_count,
            "created_at": self.created_at,
            "library_sha256": self.library_sha256,
            "report_sha256": self.report_sha256,
            "library_url": self.library_url,
            "failure_card_url": self.failure_card_url,
        }


@dataclass(frozen=True)
class RemoteLibraryRef:
    """A fully resolved logical-library, branch, and immutable-version selection."""

    library: RemoteLibrary
    branch: RemoteBranch
    version: RemoteVersion

    @property
    def library_id(self) -> str:
        return self.library.id

    @property
    def branch_id(self) -> str:
        return self.branch.id

    @property
    def branch_key(self) -> str:
        return self.branch.branch_key

    @property
    def version_id(self) -> str:
        return self.version.id

    @property
    def source_run_id(self) -> str:
        return self.version.source_run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "library": self.library.to_dict(),
            "branch": self.branch.to_dict(),
            "version": self.version.to_dict(),
        }


@dataclass(frozen=True)
class RemotePublication:
    """Response from appending one immutable version."""

    library_id: str
    branch_id: str
    version_id: str
    branch_key: str
    version_number: int = 0
    source_run_id: str = ""
    model_id: str = ""
    harness: str = ""
    source_harness: str = "unknown"
    reference_data: tuple[RemoteDataReference, ...] = ()
    publish_trigger: str = ""
    benchmarks: tuple[str, ...] = ()
    function_count: int = 0
    created_at: str = ""
    library_sha256: str = ""
    report_sha256: str = ""
    library_url: str = ""
    failure_card_url: str = ""
    dashboard_url: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RemotePublication:
        data = dict(value)
        library_id = _text(data.get("library_id"))
        branch_id = _text(data.get("branch_id"))
        version_id = _text(data.get("version_id"))
        branch_key = _text(data.get("branch_key"))
        if not library_id or not branch_id or not version_id or not branch_key:
            raise RemoteLibraryError(
                "Publication response is missing library, branch, version, or branch key"
            )
        return cls(
            library_id=library_id,
            branch_id=branch_id,
            version_id=version_id,
            branch_key=branch_key,
            version_number=_nonnegative_int(data.get("version_number")),
            source_run_id=_text(data.get("source_run_id")),
            model_id=_text(data.get("model_id")),
            harness=_text(data.get("harness")),
            source_harness=_text(data.get("source_harness")) or "unknown",
            reference_data=_reference_data_tuple(data.get("reference_data")),
            publish_trigger=_text(data.get("publish_trigger")),
            benchmarks=_string_tuple(data.get("benchmarks")),
            function_count=_nonnegative_int(data.get("function_count")),
            created_at=_text(data.get("created_at")),
            library_sha256=_text(data.get("library_sha256")),
            report_sha256=_text(data.get("report_sha256")),
            library_url=_text(data.get("library_url")),
            failure_card_url=_text(data.get("failure_card_url")),
            dashboard_url=_text(data.get("dashboard_url")),
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "library_id": self.library_id,
            "branch_id": self.branch_id,
            "version_id": self.version_id,
            "branch_key": self.branch_key,
            "version_number": self.version_number,
            "source_run_id": self.source_run_id,
            "model_id": self.model_id,
            "harness": self.harness,
            "source_harness": self.source_harness,
            "reference_data": [item.to_dict() for item in self.reference_data],
            "data_identity": self.raw.get("data_identity"),
            "publish_trigger": self.publish_trigger,
            "benchmarks": list(self.benchmarks),
            "function_count": self.function_count,
            "created_at": self.created_at,
            "library_sha256": self.library_sha256,
            "report_sha256": self.report_sha256,
            "library_url": self.library_url,
            "failure_card_url": self.failure_card_url,
            "dashboard_url": self.dashboard_url,
        }


_Record = TypeVar("_Record", RemoteLibrary, RemoteBranch, RemoteVersion)


class RemoteLibraryClient:
    """Bearer-authenticated client for the versioned library API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str = "",
        credential_source: str = "explicit",
        timeout: float = DEFAULT_TIMEOUT,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_library_bytes: int = MAX_LIBRARY_BYTES,
    ):
        if base_url is None:
            base_url = default_value("remote_library_base_url")
        from autofa.local_service import validate_service_url

        token = str(token).strip()
        configured_url = str(base_url).strip()
        self.base_url = configured_url.rstrip("/")
        if not self.base_url:
            raise ValueError("A remote library base URL is required")
        self.token = token
        self.credential_source = str(credential_source or "explicit")
        self._server_transport = self.credential_source == "server"
        self._local_transport = not self._server_transport and (
            self.credential_source == "local" or _targets_loopback(self.base_url)
        )
        if self._server_transport:
            parsed = urllib.parse.urlsplit(configured_url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.port == 0 or parsed.username or parsed.password
                    or parsed.query or parsed.fragment or parsed.path not in {"", "/"}
                    or any(character.isspace() for character in configured_url)):
                raise ValueError("The AutoFA server address must be an HTTP(S) origin")
            if any(ord(character) < 33 or ord(character) >= 127 for character in token):
                raise ValueError("The AutoFA server token contains invalid characters")
        if self._local_transport:
            self.base_url = validate_service_url(configured_url)
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)
        self.max_library_bytes = int(max_library_bytes)

    def _open(self, request: urllib.request.Request):
        if self._local_transport:
            return _local_urlopen(request, timeout=self.timeout)
        if self._server_transport:
            handlers = [_NoRedirect()]
            if _targets_loopback(self.base_url):
                handlers.append(urllib.request.ProxyHandler({}))
            return urllib.request.build_opener(*handlers).open(request, timeout=self.timeout)
        return urllib.request.urlopen(request, timeout=self.timeout)

    def _url(self, path_or_url: str) -> str:
        if urllib.parse.urlsplit(path_or_url).scheme:
            base = urllib.parse.urlsplit(self.base_url)
            target = urllib.parse.urlsplit(path_or_url)
            if (target.scheme, target.netloc) != (base.scheme, base.netloc):
                raise RemoteLibraryError(
                    "Refusing to send the library token to a different host"
                )
            return path_or_url
        path = path_or_url if path_or_url.startswith("/") else f"/{path_or_url}"
        return f"{self.base_url}{path}"

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": "autofa/remote-library-v2",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request_with_status(
        self,
        method: str,
        path_or_url: str,
        *,
        body: bytes | Iterable[bytes] | None = None,
        headers: Mapping[str, str] | None = None,
        max_bytes: int | None = None,
        allow_empty: bool = False,
        sensitive_values: tuple[str, ...] = (),
    ) -> tuple[int, bytes]:
        url = self._url(path_or_url)
        request_headers = self._headers()
        request_headers.update(headers or {})
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with self._open(request) as response:
                status = int(
                    getattr(response, "status", None)
                    or (
                        response.getcode()
                        if callable(getattr(response, "getcode", None))
                        else 200
                    )
                    or 200
                )
                payload = _read_limited(
                    response,
                    max_bytes=max_bytes or self.max_response_bytes,
                    label=f"response from {url}",
                )
        except urllib.error.HTTPError as exc:
            redactions = (self.token, *sensitive_values)
            detail = _redact_text(_http_error_detail(exc), *redactions)
            reason = _redact_text(str(exc.reason), *redactions)
            raise RemoteLibraryError(
                f"Remote library API {exc.code} for {method} {url}: "
                f"{detail or reason}",
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            reason = _redact_text(
                str(exc.reason),
                self.token,
                *sensitive_values,
            )
            raise RemoteLibraryError(
                f"Network error contacting {url}: {reason}"
            ) from exc
        if not payload and not allow_empty:
            raise RemoteLibraryError(f"Empty response from {url}")
        return status, payload

    def _request(
        self,
        method: str,
        path_or_url: str,
        *,
        body: bytes | Iterable[bytes] | None = None,
        headers: Mapping[str, str] | None = None,
        max_bytes: int | None = None,
        allow_empty: bool = False,
        sensitive_values: tuple[str, ...] = (),
    ) -> bytes:
        _, payload = self._request_with_status(
            method,
            path_or_url,
            body=body,
            headers=headers,
            max_bytes=max_bytes,
            allow_empty=allow_empty,
            sensitive_values=sensitive_values,
        )
        return payload

    def _request_json(
        self,
        method: str,
        path_or_url: str,
        *,
        body: bytes | Iterable[bytes] | None = None,
        headers: Mapping[str, str] | None = None,
        max_bytes: int | None = None,
        sensitive_values: tuple[str, ...] = (),
    ) -> Any:
        payload = self._request(
            method,
            path_or_url,
            body=body,
            headers=headers,
            max_bytes=max_bytes,
            sensitive_values=sensitive_values,
        )
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteLibraryError(
                f"Invalid JSON from {self._url(path_or_url)}: {exc}"
            ) from exc

    def _request_json_with_status(
        self,
        method: str,
        path_or_url: str,
        *,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        max_bytes: int | None = None,
        sensitive_values: tuple[str, ...] = (),
    ) -> tuple[int, Any]:
        status, payload = self._request_with_status(
            method,
            path_or_url,
            body=body,
            headers=headers,
            max_bytes=max_bytes,
            sensitive_values=sensitive_values,
        )
        try:
            return status, json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteLibraryError(
                f"Invalid JSON from {self._url(path_or_url)}: {exc}"
            ) from exc

    def start_cli_login(self, code_challenge: str) -> RemoteCliAuthStart:
        """Start an unauthenticated browser login flow."""
        challenge = str(code_challenge).strip()
        if not _valid_pkce_challenge(challenge):
            raise RemoteLibraryError("Invalid PKCE code challenge")
        data = self._request_json(
            "POST",
            "/api/v2/auth/cli/start",
            body=_json_body({"code_challenge": challenge}),
            headers={"Content-Type": "application/json"},
            max_bytes=MAX_AUTH_RESPONSE_BYTES,
        )
        if not isinstance(data, Mapping):
            raise RemoteLibraryError("Expected a CLI login start object from the service")
        flow_id = _text(data.get("flow_id"))
        login_url = _text(data.get("login_url"))
        expires_in = _positive_int(data.get("expires_in"))
        interval = _positive_int(data.get("interval"))
        if (
            _CLI_FLOW_ID_RE.fullmatch(flow_id) is None
            or expires_in is None
            or interval is None
            or interval > expires_in
        ):
            raise RemoteLibraryError("Invalid CLI login start response from the service")
        parsed_login = urllib.parse.urlsplit(login_url)
        base_path = urllib.parse.urlsplit(self.base_url).path.rstrip("/")
        expected_path = f"{base_path}/auth/cli/{flow_id}"
        if (
            not _same_origin(self.base_url, login_url)
            or parsed_login.username
            or parsed_login.password
            or parsed_login.query
            or parsed_login.fragment
            or parsed_login.path != expected_path
        ):
            raise RemoteLibraryError(
                "CLI login URL does not match the configured service"
            )
        return RemoteCliAuthStart(
            flow_id=flow_id,
            login_url=login_url,
            expires_in=expires_in,
            interval=interval,
        )

    def poll_cli_token(
        self,
        flow_id: str,
        verifier: str,
    ) -> RemoteCliTokenPending | RemoteCliToken:
        """Poll a CLI flow; 202 is pending and 200 returns the service token."""
        flow = str(flow_id).strip()
        verifier_value = str(verifier).strip()
        if _CLI_FLOW_ID_RE.fullmatch(flow) is None:
            raise RemoteLibraryError("Invalid CLI login flow id")
        if _CLI_VERIFIER_RE.fullmatch(verifier_value) is None:
            raise RemoteLibraryError("Invalid PKCE verifier")
        status, data = self._request_json_with_status(
            "POST",
            "/api/v2/auth/cli/token",
            body=_json_body({"flow_id": flow, "verifier": verifier_value}),
            headers={"Content-Type": "application/json"},
            max_bytes=MAX_AUTH_RESPONSE_BYTES,
            sensitive_values=(verifier_value,),
        )
        if not isinstance(data, Mapping):
            raise RemoteLibraryError("Expected a CLI token response object")
        if status == 202:
            interval = _positive_int(data.get("interval"))
            if _text(data.get("status") or "pending") != "pending" or interval is None:
                raise RemoteLibraryError("Invalid pending CLI token response")
            return RemoteCliTokenPending(interval=interval)
        if status != 200:
            raise RemoteLibraryError(
                f"Unexpected CLI token response status: {status}",
                status=status,
            )
        token = _text(data.get("token"))
        token_type = _text(data.get("token_type") or "bearer").lower()
        expires_at = _text(data.get("expires_at"))
        expires_in = _positive_int(data.get("expires_in"))
        identity_data = data.get("identity")
        if (
            _CLI_TOKEN_RE.fullmatch(token) is None
            or token_type != "bearer"
            or not _valid_iso_datetime(expires_at)
            or expires_in is None
            or not isinstance(identity_data, Mapping)
        ):
            raise RemoteLibraryError("Invalid CLI token response from the service")
        identity = RemoteIdentity.from_dict(identity_data)
        return RemoteCliToken(
            token=token,
            token_type=token_type,
            expires_at=expires_at,
            expires_in=expires_in,
            identity=identity,
        )

    def me(self) -> RemoteIdentity:
        """Return the identity associated with this client's bearer credential."""
        data = self._request_json(
            "GET",
            "/api/v1/me",
            max_bytes=MAX_AUTH_RESPONSE_BYTES,
        )
        if not isinstance(data, Mapping):
            raise RemoteLibraryError("Expected an identity object from the service")
        return RemoteIdentity.from_dict(data)

    def revoke_cli_token(self) -> None:
        """Revoke this client's AutoFA CLI service token."""
        self._request(
            "DELETE",
            "/api/v2/auth/cli/token",
            max_bytes=MAX_AUTH_RESPONSE_BYTES,
            allow_empty=True,
        )

    def _paginated(
        self,
        path: str,
        parser: Callable[[Mapping[str, Any]], _Record],
    ) -> list[_Record]:
        records: list[_Record] = []
        next_path = path
        while next_path:
            data = self._request_json("GET", next_path)
            if not isinstance(data, Mapping) or not isinstance(data.get("items"), list):
                raise RemoteLibraryError("Expected a paginated item array from the service")
            items = data["items"]
            total = _nonnegative_int(data.get("total", len(items)))
            limit = max(1, _nonnegative_int(data.get("limit", len(items))) or 1)
            offset = _nonnegative_int(data.get("offset"))
            for item in items:
                if not isinstance(item, Mapping):
                    raise RemoteLibraryError("Expected service records to be JSON objects")
                records.append(parser(item))
            consumed = offset + len(items)
            if not items or consumed >= total:
                break
            next_offset = offset + limit
            separator = "&" if "?" in path else "?"
            next_path = (
                f"{path}{separator}"
                + urllib.parse.urlencode({"limit": limit, "offset": next_offset})
            )
        return records

    def list_libraries(self) -> list[RemoteLibrary]:
        """Return all visible logical libraries, following v2 pagination."""
        records = self._paginated("/api/v2/libraries", RemoteLibrary.from_dict)
        return [self._remote_library(item) for item in records]

    def get_library(self, library_id: str) -> RemoteLibrary:
        """Resolve a logical library from the paginated v2 collection."""
        wanted = str(library_id).strip()
        for library in self.list_libraries():
            if library.id == wanted:
                return library
        raise RemoteLibraryError(f"Remote library not found: {wanted}", status=404)

    def list_branches(self, library_id: str) -> list[RemoteBranch]:
        path = f"/api/v2/libraries/{_quote_id(library_id)}/branches"
        return self._paginated(path, RemoteBranch.from_dict)

    def get_branch(self, library_id: str, branch_id: str) -> RemoteBranch:
        wanted = str(branch_id).strip()
        for branch in self.list_branches(library_id):
            if branch.id == wanted:
                return branch
        raise RemoteLibraryError(f"Remote branch not found: {wanted}", status=404)

    def list_versions(self, library_id: str, branch_id: str) -> list[RemoteVersion]:
        path = (
            f"/api/v2/libraries/{_quote_id(library_id)}/branches/"
            f"{_quote_id(branch_id)}/versions"
        )
        versions = self._paginated(path, self._remote_version)
        return sorted(
            versions,
            key=lambda item: (item.version_number, item.created_at, item.id),
            reverse=True,
        )

    def get_version(
        self,
        library_id: str,
        branch_id: str,
        version_id: str,
    ) -> RemoteVersion:
        path = (
            f"/api/v2/libraries/{_quote_id(library_id)}/branches/"
            f"{_quote_id(branch_id)}/versions/{_quote_id(version_id)}"
        )
        data = self._request_json("GET", path)
        if not isinstance(data, Mapping):
            raise RemoteLibraryError("Expected a version object from the service")
        return self._remote_version(data)

    def resolve_library_ref(
        self,
        library_id: str,
        *,
        branch_id: str | None = None,
        version_id: str | None = None,
    ) -> RemoteLibraryRef:
        """Resolve omitted selectors to latest, returning an immutable concrete ref."""
        library = self.get_library(library_id)
        selected_branch_id = str(branch_id or library.latest_branch_id).strip()
        if not selected_branch_id:
            raise RemoteLibraryError("Logical library metadata has no latest branch")
        branch = self.get_branch(library.id, selected_branch_id)
        selected_version_id = str(version_id or branch.latest_version_id).strip()
        if not selected_version_id:
            raise RemoteLibraryError("Branch metadata has no latest version")
        version = self.get_version(library.id, branch.id, selected_version_id)
        return RemoteLibraryRef(library=library, branch=branch, version=version)

    def upload_library(
        self,
        *,
        metadata: Mapping[str, Any],
        library_json: bytes | str | Path | Mapping[str, Any],
        failure_cards_html: bytes | str | Path,
        report_assets_zip: bytes | str | Path | None = None,
        trace_references: list[dict[str, str]] | None = None,
        _prepared_library: bool = False,
    ) -> RemotePublication:
        """Append a version through ``POST /api/v2/publications``."""
        publication_metadata = dict(metadata)
        for key in ("branch_key", "source_run_id", "model_id"):
            if not _text(publication_metadata.get(key)):
                raise RemoteLibraryError(f"Publication metadata requires {key}")
        publication_metadata.setdefault("benchmarks", [])

        # Callers may reuse bytes already validated and normalized by
        # prepare_library_payload, avoiding another pass over large evidence.
        if _prepared_library:
            if not isinstance(library_json, bytes):
                raise TypeError("Prepared library payload must be bytes")
            library_payload = library_json
        else:
            library_payload = prepare_library_payload(library_json,
                                                     max_bytes=self.max_library_bytes)
        report_payload = _bytes_payload(failure_cards_html)
        assets_payload = (
            _bytes_payload(report_assets_zip)
            if report_assets_zip is not None
            else None
        )
        if len(library_payload) > self.max_library_bytes:
            raise RemoteLibraryError(
                f"library.json exceeds the {self.max_library_bytes}-byte upload limit"
            )
        if len(report_payload) > MAX_REPORT_BYTES:
            raise RemoteLibraryError(
                f"failure_cards.html exceeds the {MAX_REPORT_BYTES}-byte upload limit"
            )
        if assets_payload is not None and len(assets_payload) > MAX_REPORT_ASSETS_BYTES:
            raise RemoteLibraryError(
                "failure-card assets exceed the "
                f"{MAX_REPORT_ASSETS_BYTES}-byte upload limit"
            )
        boundary = f"autofa-{uuid.uuid4().hex}"
        body = _multipart_body(
            boundary,
            metadata=publication_metadata,
            library_json=library_payload,
            failure_cards_html=report_payload,
            report_assets_zip=assets_payload,
            trace_references=trace_references,
        )
        data = self._request_json(
            "POST", "/api/v2/publications", body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        if not isinstance(data, Mapping):
            raise RemoteLibraryError("Expected a publication object from the service")
        publication = RemotePublication.from_dict(data)
        return self._remote_publication(publication)

    def delete_library(self, library_id: str) -> None:
        """Delete remains on v1 because the paired service exposes no v2 delete."""
        self._request(
            "DELETE",
            f"/api/v1/libraries/{_quote_id(library_id)}",
            allow_empty=True,
        )

    def download_library(
        self,
        library_id: str,
        destination: Path,
        *,
        branch_id: str | None = None,
        version_id: str | None = None,
        download_url: str | None = None,
        expected_sha256: str | None = None,
        progress: Callable[[int, int | None], None] | None = None,
    ) -> Path:
        """Atomically download and validate one concrete version-1 library."""
        if not download_url and (not branch_id or not version_id):
            selected = self.resolve_library_ref(
                library_id,
                branch_id=branch_id,
                version_id=version_id,
            )
            branch_id = selected.branch_id
            version_id = selected.version_id
            download_url = selected.version.library_url
            expected_sha256 = expected_sha256 or selected.version.library_sha256
        path_or_url = download_url or (
            f"/api/v2/libraries/{_quote_id(library_id)}/branches/"
            f"{_quote_id(branch_id or '')}/versions/{_quote_id(version_id or '')}/library.json"
        )
        url = self._url(path_or_url)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
        request = urllib.request.Request(
            url,
            headers=self._headers(accept="application/json"),
            method="GET",
        )
        try:
            with self._open(request) as response:
                total = _content_length(response)
                if total is not None and total > self.max_library_bytes:
                    raise RemoteLibraryError(
                        f"Remote library exceeds the {self.max_library_bytes}-byte limit"
                    )
                downloaded = 0
                if progress is not None:
                    progress(downloaded, total)
                with partial.open("wb") as output:
                    while True:
                        chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > self.max_library_bytes:
                            raise RemoteLibraryError(
                                f"Remote library exceeds the "
                                f"{self.max_library_bytes}-byte limit"
                            )
                        output.write(chunk)
                        if progress is not None:
                            progress(downloaded, total)
            _validate_library_file(partial, expected_sha256=expected_sha256)
            os.replace(partial, destination)
        except urllib.error.HTTPError as exc:
            detail = _redact_text(_http_error_detail(exc), self.token)
            reason = _redact_text(str(exc.reason), self.token)
            raise RemoteLibraryError(
                f"Remote library API {exc.code} downloading {version_id or library_id}: "
                f"{detail or reason}",
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            reason = _redact_text(str(exc.reason), self.token)
            raise RemoteLibraryError(
                f"Network error downloading {url}: {reason}"
            ) from exc
        finally:
            partial.unlink(missing_ok=True)
        return destination

    def download_version(
        self,
        version: RemoteVersion,
        destination: Path,
        *,
        progress: Callable[[int, int | None], None] | None = None,
    ) -> Path:
        return self.download_library(
            version.library_id,
            destination,
            branch_id=version.branch_id,
            version_id=version.id,
            download_url=version.library_url or None,
            expected_sha256=version.library_sha256 or None,
            progress=progress,
        )

    def _remote_library(self, library: RemoteLibrary) -> RemoteLibrary:
        dashboard = library.dashboard_url or self.dashboard_url(library.id)
        return RemoteLibrary(
            **{
                **library.__dict__,
                "dashboard_url": self._public_url(dashboard),
            }
        )

    def _remote_version(self, value: Mapping[str, Any]) -> RemoteVersion:
        version = RemoteVersion.from_dict(value)
        return RemoteVersion(
            **{
                **version.__dict__,
                "library_url": self._public_url(version.library_url),
                "failure_card_url": self._public_url(version.failure_card_url),
            }
        )

    def _remote_publication(self, publication: RemotePublication) -> RemotePublication:
        dashboard = publication.dashboard_url or self.dashboard_url(
            publication.library_id,
            branch_id=publication.branch_id,
            version_id=publication.version_id,
        )
        return RemotePublication(
            **{
                **publication.__dict__,
                "library_url": self._public_url(publication.library_url),
                "failure_card_url": self._public_url(publication.failure_card_url),
                "dashboard_url": self._public_url(dashboard),
            }
        )

    def _public_url(self, value: str) -> str:
        if not value:
            return ""
        return urllib.parse.urljoin(f"{self.base_url}/", value)

    def dashboard_url(
        self,
        library_id: str,
        *,
        branch_id: str | None = None,
        version_id: str | None = None,
    ) -> str:
        url = f"{self.base_url}/libraries/{_quote_id(library_id)}"
        query = {
            key: value
            for key, value in (
                ("branch_id", branch_id),
                ("version_id", version_id),
            )
            if value
        }
        return f"{url}?{urllib.parse.urlencode(query)}" if query else url


def client_from_config(
    config: Any,
    *,
    credential_store: Any | None = None,
) -> RemoteLibraryClient:
    """Use the profile token locally, or environment/keychain credentials remotely."""
    from autofa.local_service import service_token, validate_service_url

    configured_url = str(setting_value(config, "remote_library_base_url") or "").strip()
    if not configured_url:
        from autofa.services import LIBRARY_NOT_CONFIGURED

        raise RemoteLibraryError(LIBRARY_NOT_CONFIGURED)
    if setting_value(config, "remote_library_mode") == "server":
        token_env = str(setting_value(config, "remote_library_token_env") or "").strip()
        if not token_env or token_env == "GITLAB_TOKEN":
            token_env = "AUTOFA_LIBRARY_TOKEN"
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise RemoteLibraryError(
                f"Set {token_env} to the receiving AutoFA server's token before publishing."
            )
        return RemoteLibraryClient(configured_url, token=token, credential_source="server",
                                   timeout=600.0)
    if _targets_loopback(configured_url):
        try:
            configured_url = validate_service_url(configured_url)
            token = service_token()
        except (OSError, RuntimeError, ValueError) as exc:
            raise RemoteLibraryError(str(exc)) from exc
        return RemoteLibraryClient(configured_url, token=token, credential_source="local")

    from autofa.data.remote_library_credentials import (
        CredentialError,
        normalize_service_url,
        resolve_credential,
    )

    token_env = str(
        setting_value(config, "remote_library_token_env")
        or default_value("remote_library_token_env")
    ).strip()
    try:
        base_url = normalize_service_url(configured_url)
        credential = resolve_credential(
            base_url,
            env_name=token_env,
            store=credential_store,
        )
    except CredentialError as exc:
        raise RemoteLibraryError(str(exc)) from exc
    return RemoteLibraryClient(
        base_url,
        token=credential.token,
        credential_source=credential.source,
    )


def library_service_name(base_url: str, *, mode: str = "cloud") -> str:
    """Name the configured dashboard without confusing it with on-disk catalogs."""
    from autofa.local_service import is_local_service_url

    if mode == "server":
        return "AutoFA Server"
    return "AutoFA Local Library" if is_local_service_url(base_url) else "AutoFA Cloud Library"


def _targets_loopback(value: str) -> bool:
    """Recognize local intent before reading any cloud credentials."""
    try:
        hostname = urllib.parse.urlsplit(value).hostname or ""
    except ValueError:
        return False  # The cloud URL validator will report a malformed URL.
    if hostname.rstrip(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname.rstrip(".")).is_loopback
    except ValueError:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RemoteLibraryError("Local library requests cannot redirect")


def _local_urlopen(request: urllib.request.Request, *, timeout: float):
    from autofa.local_service import validate_service_url

    parsed = urllib.parse.urlsplit(request.full_url)
    validate_service_url(
        urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    )
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect(),
    ).open(request, timeout=timeout)


def remote_library_cache_dir(
    version_id: str,
    *,
    base_url: str | None = None,
) -> Path:
    """Collision-safe immutable cache keyed by service URL and full version id."""
    if base_url is None:
        base_url = default_value("remote_library_base_url")
    identity = f"{base_url.rstrip('/')}\0{version_id}".encode()
    cache_key = hashlib.sha256(identity).hexdigest()
    path = remote_library_cache_root() / cache_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def materialize_remote_library(
    selection: RemoteLibraryRef | RemoteVersion | RemoteLibrary | str,
    *,
    client: RemoteLibraryClient,
    force: bool = False,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download or reuse the exact immutable version selected by the caller."""
    reference, version = _resolve_materialization(selection, client)
    cache = remote_library_cache_dir(version.id, base_url=client.base_url)
    destination = cache / "library.json"
    sidecar = cache / "metadata.json"

    if destination.is_file() and not force:
        try:
            _validate_library_file(
                destination,
                expected_sha256=version.library_sha256 or None,
            )
            if not sidecar.is_file():
                _write_cache_sidecar(
                    sidecar,
                    version=version,
                    reference=reference,
                    base_url=client.base_url,
                )
            return destination
        except RemoteLibraryError:
            destination.unlink(missing_ok=True)

    client.download_version(version, destination, progress=progress)
    _write_cache_sidecar(
        sidecar,
        version=version,
        reference=reference,
        base_url=client.base_url,
    )
    return destination


def _resolve_materialization(
    selection: RemoteLibraryRef | RemoteVersion | RemoteLibrary | str,
    client: RemoteLibraryClient,
) -> tuple[RemoteLibraryRef | None, RemoteVersion]:
    if isinstance(selection, RemoteLibraryRef):
        return selection, selection.version
    if isinstance(selection, RemoteVersion):
        return None, selection
    if isinstance(selection, str):
        reference = client.resolve_library_ref(selection)
        return reference, reference.version
    if not selection.latest_branch_id or not selection.latest_version_id:
        raise RemoteLibraryError("Logical library metadata has no latest branch or version")
    reference = client.resolve_library_ref(selection.id)
    return reference, reference.version


def _write_cache_sidecar(
    path: Path,
    *,
    version: RemoteVersion,
    reference: RemoteLibraryRef | None,
    base_url: str,
) -> None:
    payload = {
        "cached_at": datetime.now(UTC).isoformat(),
        "service_base_url": base_url.rstrip("/"),
        "version": version.to_dict(),
    }
    if reference is not None:
        payload["selection"] = reference.to_dict()
    _atomic_json_write(path, payload)


def _multipart_body(
    boundary: str,
    *,
    metadata: Mapping[str, Any],
    library_json: bytes,
    failure_cards_html: bytes,
    report_assets_zip: bytes | None = None,
    trace_references: list[dict[str, str]] | None = None,
) -> bytes:
    metadata_json = json.dumps(
        dict(metadata),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    chunks: list[bytes] = []

    def add_part(
        name: str,
        payload: bytes,
        content_type: str,
        *,
        filename: str | None = None,
    ) -> None:
        chunks.append(f"--{boundary}\r\n".encode())
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        chunks.append(f"{disposition}\r\n".encode())
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        chunks.append(payload)
        chunks.append(b"\r\n")

    add_part("metadata", metadata_json, "application/json")
    if trace_references is not None:
        add_part("trace_references", json.dumps(
            trace_references, ensure_ascii=False, allow_nan=False,
        ).encode(), "application/json")
    add_part(
        "library",
        library_json,
        "application/json",
        filename="library.json",
    )
    add_part(
        "failure_card",
        failure_cards_html,
        "text/html; charset=utf-8",
        filename="failure_cards.html",
    )
    if report_assets_zip is not None:
        add_part(
            "report_assets",
            report_assets_zip,
            "application/zip",
            filename="failure_card_assets.zip",
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)


def prepare_library_payload(
    value: bytes | str | Path | Mapping[str, Any], *, max_bytes: int = MAX_LIBRARY_BYTES,
) -> bytes:
    """Return exact upload bytes, compacting large evidence without dropping fields."""
    payload = _json_payload(value)
    if len(payload) > max_bytes:
        from autofa.core.library_transport import pack_bucket_examples

        try:
            packed = pack_bucket_examples(json.loads(payload), max_bytes=max_bytes)
            payload = json.dumps(packed, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8")
            _validate_library_data(packed)
        except (TypeError, ValueError) as exc:
            raise RemoteLibraryError(f"Cannot pack library for upload: {exc}") from exc
    if len(payload) > max_bytes:
        raise RemoteLibraryError(
            f"library.json is {len(payload)} bytes after lossless packing; "
            f"the upload limit is {max_bytes} bytes"
        )
    return payload


def _json_payload(value: bytes | str | Path | Mapping[str, Any]) -> bytes:
    from autofa.core.library_transport import (
        PACKED_KEY,
        REVIEW_PACKED_KEY,
        pack_bucket_examples,
        unpack_bucket_examples,
    )
    from autofa.reports.ordering import order_library_functions

    if isinstance(value, Mapping):
        payload = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ).encode("utf-8")
    else:
        payload = _bytes_payload(value)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteLibraryError(f"Invalid library.json upload: {exc}") from exc
    changed = False
    _validate_library_data(decoded)
    # Packed membership records address functions by position. Restore those
    # arrays before sorting, then rebuild their positions in the outgoing order.
    was_packed = PACKED_KEY in decoded or REVIEW_PACKED_KEY in decoded
    had_review_packing = REVIEW_PACKED_KEY in decoded
    if was_packed:
        decoded = unpack_bucket_examples(decoded)
    functions = decoded.get("functions", [])
    ordered = order_library_functions(functions, review_history=decoded.get("review_history"))
    if any(before is not after for before, after in zip(functions, ordered, strict=True)):
        decoded["functions"] = ordered
        changed = True
    if changed:
        if was_packed:
            decoded = pack_bucket_examples(decoded, max_bytes=0 if had_review_packing
                                           else MAX_LIBRARY_BYTES)
        payload = json.dumps(decoded, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    return payload


def _bytes_payload(value: bytes | str | Path) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, Path):
        return value.read_bytes()
    return str(value).encode("utf-8")


def _validate_library_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RemoteLibraryError("Downloaded library.json is empty")
    try:
        payload = path.read_bytes()
        data = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteLibraryError(f"Downloaded library.json is invalid: {exc}") from exc
    _validate_library_data(data)
    expected = _text(expected_sha256).lower()
    if expected:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RemoteLibraryError(
                f"Downloaded library.json SHA-256 mismatch: expected {expected}, got {actual}"
            )


def _validate_library_data(data: Any) -> None:
    from autofa.core.library import FUNCTION_KINDS

    if not isinstance(data, dict):
        raise RemoteLibraryError("library.json must contain a JSON object")
    version = data.get("version", 1)
    if type(version) is not int or version not in {1, 2}:
        raise RemoteLibraryError(f"Unsupported FunctionLibrary version: {version!r}")
    if not isinstance(data.get("functions", []), list):
        raise RemoteLibraryError("library.json functions must be an array")
    for index, function in enumerate(data.get("functions", [])):
        if not isinstance(function, dict):
            raise RemoteLibraryError(f"library.json functions[{index}] must be an object")
        if function.get("kind", "regex") not in FUNCTION_KINDS:
            raise RemoteLibraryError(
                f"Invalid function kind {function['kind']!r} for detector "
                f"{function.get('name')!r}; expected 'regex', 'llm' or 'python'. Retired "
                "strategy labels ran as Python: set kind to 'python' or recreate the detector."
            )
    if version == 2 or "contributions" in data:
        from autofa.core.library import FunctionLibrary

        try:
            FunctionLibrary.from_dict(data)
        except (TypeError, ValueError, KeyError, SyntaxError) as exc:
            raise RemoteLibraryError(f"Invalid composed library: {exc}") from exc


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_limited(response: Any, *, max_bytes: int, label: str) -> bytes:
    total = _content_length(response)
    if total is not None and total > max_bytes:
        raise RemoteLibraryError(f"{label} exceeds the {max_bytes}-byte limit")
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = response.read(min(64 * 1024, max_bytes - size + 1))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            raise RemoteLibraryError(f"{label} exceeds the {max_bytes}-byte limit")
    return b"".join(chunks)


def _content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length") or headers.get("content-length")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read(1024).decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        return ""


def _json_body(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _valid_pkce_challenge(value: str) -> bool:
    if len(value) != 43 or re.fullmatch(r"[A-Za-z0-9_-]{43}", value) is None:
        return False
    try:
        decoded = base64.b64decode(value + "=", altchars=b"-_", validate=True)
    except ValueError:
        return False
    encoded = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    return len(decoded) == 32 and encoded == value


def _valid_iso_datetime(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _same_origin(left: str, right: str) -> bool:
    try:
        left_parsed = urllib.parse.urlsplit(left)
        right_parsed = urllib.parse.urlsplit(right)
        left_port = left_parsed.port or (
            443 if left_parsed.scheme.lower() == "https" else 80
        )
        right_port = right_parsed.port or (
            443 if right_parsed.scheme.lower() == "https" else 80
        )
    except ValueError:
        return False
    return (
        left_parsed.scheme.lower(),
        (left_parsed.hostname or "").lower().rstrip("."),
        left_port,
    ) == (
        right_parsed.scheme.lower(),
        (right_parsed.hostname or "").lower().rstrip("."),
        right_port,
    )


def _redact_text(value: str, *secrets_to_hide: str) -> str:
    result = str(value)
    for secret in secrets_to_hide:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    # Defensive response scrubbing if a remote error echoes an unknown CLI token.
    result = re.sub(r"autofa_cli_[A-Za-z0-9_-]+", "[REDACTED]", result)
    return result


def _quote_id(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("A library, branch, version, or run id is required")
    return urllib.parse.quote(text, safe="")


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _reference_data_tuple(value: Any) -> tuple[RemoteDataReference, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    records: list[RemoteDataReference] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            records.append(RemoteDataReference.from_dict(item))
        except RemoteLibraryError:
            continue
    return tuple(records)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _positive_int(value: Any) -> int | None:
    if type(value) is not int:
        return None
    return value if value > 0 else None
