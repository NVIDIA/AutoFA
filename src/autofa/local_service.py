# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local dashboard paths, credentials and loopback boundary."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from autofa.config import Config

DEFAULT_LOCAL_LIBRARY_URL = "http://127.0.0.1:8765"


def dashboard_url_from_config(config: Config) -> str:
    """The local browser runs independently of the publication service.

    An existing loopback publication URL remains the default for compatibility.
    An explicit local URL allows the local and shared services to coexist.
    """
    configured = config.local_library_base_url.strip()
    if not configured:
        publication_url = config.remote_library_base_url.strip()
        configured = (
            publication_url if is_local_service_url(publication_url)
            and getattr(config, "remote_library_mode", "cloud") != "server"
            else DEFAULT_LOCAL_LIBRARY_URL
        )
    return validate_service_url(configured)


def service_root() -> Path:
    from autofa.config import autofa_home

    root = autofa_home() / "server"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root.resolve()


def service_token(*, create: bool = False) -> str:
    path = service_root() / "token"
    if create:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(48) + "\n")
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError("Start the local library dashboard with `autofa serve` first") from exc
    if len(token) < 32:
        raise ValueError("The local library token is invalid")
    return token


def validate_service_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.port == 0
    ):
        raise ValueError("The library dashboard URL must be an HTTP loopback origin")
    return value.rstrip("/")


def is_local_service_url(value: str) -> bool:
    """Whether a configured URL selects the loopback-only local dashboard."""
    try:
        validate_service_url(value)
    except ValueError:
        return False
    return True


def valid_local_request(request, public_url: str, remote_hosts: bool = False) -> bool:
    """Reject foreign Host/Origin headers and cross-site browser requests.

    `remote_hosts`: the server listens beyond loopback, so any Host on its port is its own, and
    an Origin must be that same host."""
    try:
        host = request.headers.get("host", "")
        if remote_hosts:
            parsed_host = urlsplit(f"http://{host}")
            if not parsed_host.hostname or parsed_host.username or parsed_host.password:
                return False
        else:
            validate_service_url(f"http://{host}")
        expected_port = urlsplit(public_url).port or 80
        if (urlsplit(str(request.url)).port or 80) != expected_port:
            return False
        origin = request.headers.get("origin")
        # Sandboxed reports have an opaque origin. Their trajectory URLs carry
        # a version-bound token verified by the server before serving an asset.
        opaque_asset = (
            origin == "null"
            and request.method in {"GET", "HEAD"}
            and request.url.path.startswith("/report-assets/")
            and len(request.url.path.split("/")) >= 5
        )
        if origin and not opaque_asset:
            if remote_hosts:
                if urlsplit(origin).hostname != urlsplit(f"http://{host}").hostname:
                    return False
            else:
                validate_service_url(origin)
            if (urlsplit(origin).port or 80) != expected_port:
                return False
        if request.headers.get("sec-fetch-site") == "cross-site" and not opaque_asset:
            return False
        return True
    except ValueError:
        return False
