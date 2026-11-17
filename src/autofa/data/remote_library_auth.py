# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit AutoFA Cloud Library login and logout workflows."""

from __future__ import annotations

import os
import time
import webbrowser
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from typing import Any

from rich.console import Console
from rich.status import Status

from autofa.data.remote_library_api import (
    RemoteCliToken,
    RemoteCliTokenPending,
    RemoteIdentity,
    RemoteLibraryClient,
    RemoteLibraryError,
    _targets_loopback,
    client_from_config,
    generate_pkce_pair,
)
from autofa.data.remote_library_credentials import (
    CLI_TOKEN_PREFIX,
    CredentialError,
    CredentialStore,
    KeyringCredentialStore,
    normalize_service_url,
)
from autofa.defaults import default_value, setting_value
from autofa.local_service import is_local_service_url, validate_service_url

CLI_AUTH_REQUEST_TIMEOUT = 15.0
CLI_AUTH_MAX_WAIT_SECONDS = 15 * 60


def login_to_remote_library(
    config: Any,
    *,
    credential_store: CredentialStore | None = None,
    client: RemoteLibraryClient | Any | None = None,
    authenticated_client_factory: Callable[[str], Any] | None = None,
    console: Console | Any | None = None,
    browser_open: Callable[[str], bool] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    status_factory: Callable[..., AbstractContextManager[Any]] = Status,
) -> RemoteIdentity:
    """Complete browser PKCE login, validate, then persist the CLI token."""
    con = console or Console()
    if setting_value(config, "remote_library_mode") == "server":
        authenticated = client or client_from_config(config, credential_store=credential_store)
        identity = authenticated.me()
        if identity.auth_method != "local_token":
            raise RemoteLibraryError("The server did not authenticate the supplied AutoFA token")
        con.print("[green]Connected to the AutoFA server with the configured token.[/green]")
        return identity
    base_url = _configured_base_url(config)
    if is_local_service_url(base_url):
        con.print("[dim]AutoFA Local Library uses this profile; no sign-in is required.[/dim]")
        return RemoteIdentity(id=1, username="local", name="Local researcher")
    store = credential_store or KeyringCredentialStore()
    # Fail before opening a browser when secure persistence is unavailable.
    store.get_token(base_url)
    unauthenticated = client or RemoteLibraryClient(
        base_url,
        timeout=CLI_AUTH_REQUEST_TIMEOUT,
        credential_source="none",
    )
    pair = generate_pkce_pair()
    started = unauthenticated.start_cli_login(pair.challenge)

    con.print("[bold]Open this URL to sign in to AutoFA Cloud Library:[/bold]")
    con.print(f"[link={started.login_url}]{started.login_url}[/link]")
    opener = browser_open or webbrowser.open
    try:
        opened = bool(opener(started.login_url))
    except Exception:  # noqa: BLE001
        opened = False
    if not opened:
        con.print("[yellow]The browser did not open; use the clickable URL above.[/yellow]")

    deadline = monotonic() + min(
        started.expires_in,
        CLI_AUTH_MAX_WAIT_SECONDS,
    )
    token_response: RemoteCliToken | None = None
    with status_factory(
        "Waiting for browser sign-in…",
        console=con,
        spinner="dots",
    ):
        while monotonic() < deadline:
            polled = unauthenticated.poll_cli_token(started.flow_id, pair.verifier)
            if isinstance(polled, RemoteCliToken):
                token_response = polled
                break
            if not isinstance(polled, RemoteCliTokenPending):
                raise RemoteLibraryError("Invalid CLI login polling response")
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            sleeper(min(float(polled.interval or started.interval), remaining))
    if token_response is None or token_response.identity is None:
        raise RemoteLibraryError(
            "AutoFA Cloud Library login timed out or expired; run `autofa login` again"
        )

    auth_factory = authenticated_client_factory or (
        lambda token: RemoteLibraryClient(
            base_url,
            token=token,
            credential_source="keychain",
            timeout=CLI_AUTH_REQUEST_TIMEOUT,
        )
    )
    authenticated = auth_factory(token_response.token)
    validated = authenticated.me()
    issued_identity = token_response.identity
    if (
        validated.id != issued_identity.id
        or validated.username != issued_identity.username
        or (
            validated.auth_method
            and validated.auth_method != "cli_token"
        )
    ):
        with suppress(Exception):
            authenticated.revoke_cli_token()
        raise RemoteLibraryError(
            "CLI token identity did not match the completed browser login"
        )

    try:
        store.set_token(base_url, token_response.token)
    except CredentialError:
        with suppress(Exception):
            authenticated.revoke_cli_token()
        raise
    con.print(
        f"[green]Logged in to AutoFA Cloud Library as @{validated.username}[/green]"
    )
    return validated


def logout_from_remote_library(
    config: Any,
    *,
    credential_store: CredentialStore | None = None,
    authenticated_client_factory: Callable[[str], Any] | None = None,
    console: Console | Any | None = None,
) -> bool:
    """Best-effort revoke a keychain token, then remove it locally."""
    con = console or Console()
    if setting_value(config, "remote_library_mode") == "server":
        env_name = str(setting_value(config, "remote_library_token_env"))
        if not env_name or env_name == "GITLAB_TOKEN":
            env_name = "AUTOFA_LIBRARY_TOKEN"
        con.print(
            f"[dim]The AutoFA server uses {env_name}. Remove it from your shell and "
            "profile .env to disconnect; no browser login is stored.[/dim]"
        )
        return False
    base_url = _configured_base_url(config)
    if is_local_service_url(base_url):
        con.print("[dim]AutoFA Local Library uses this profile; no sign-out is required.[/dim]")
        return False
    env_name = str(
        setting_value(config, "remote_library_token_env")
        or default_value("remote_library_token_env")
    ).strip()
    environment_pat_present = bool(os.environ.get(env_name, "").strip())
    store = credential_store or KeyringCredentialStore()
    token = store.get_token(base_url)
    removed = False
    revoke_failed = False

    if token:
        if token.startswith(CLI_TOKEN_PREFIX):
            auth_factory = authenticated_client_factory or (
                lambda value: RemoteLibraryClient(
                    base_url,
                    token=value,
                    credential_source="keychain",
                    timeout=CLI_AUTH_REQUEST_TIMEOUT,
                )
            )
            try:
                auth_factory(token).revoke_cli_token()
            except Exception:  # noqa: BLE001
                revoke_failed = True
        removed = store.delete_token(base_url)
        if revoke_failed:
            con.print(
                "[yellow]Removed the local AutoFA login; remote revocation "
                "could not be confirmed.[/yellow]"
            )
        else:
            con.print("[green]Logged out of AutoFA Cloud Library.[/green]")
    else:
        con.print("[dim]Already logged out of AutoFA Cloud Library.[/dim]")

    if environment_pat_present:
        con.print(
            f"[yellow]{env_name} is an environment GitLab PAT. `autofa logout` "
            "does not unset or revoke it; remove it from your shell and revoke it "
            "in GitLab if needed.[/yellow]"
        )
    return removed


def _configured_base_url(config: Any) -> str:
    configured_url = str(setting_value(config, "remote_library_base_url") or "").strip()
    if not configured_url:
        from autofa.services import LIBRARY_NOT_CONFIGURED

        raise RemoteLibraryError(LIBRARY_NOT_CONFIGURED)
    if _targets_loopback(configured_url):
        try:
            return validate_service_url(configured_url)
        except ValueError as exc:
            raise RemoteLibraryError(
                "Invalid local library configuration: use an HTTP loopback origin"
            ) from exc
    return normalize_service_url(configured_url)
