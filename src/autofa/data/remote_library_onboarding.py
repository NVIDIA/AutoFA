# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-run cloud sign-in prompt and nonfatal home identity status."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from autofa.config import autofa_home
from autofa.data.remote_library_api import (
    RemoteIdentity,
    RemoteLibraryClient,
    RemoteLibraryError,
    _targets_loopback,
    client_from_config,
)
from autofa.data.remote_library_auth import login_to_remote_library
from autofa.data.remote_library_credentials import (
    CredentialError,
    CredentialStore,
    KeyringCredentialStore,
    find_credential,
    normalize_service_url,
)
from autofa.defaults import default_value, setting_value
from autofa.local_service import service_token, validate_service_url

FIRST_PROMPT_MARKER = "cloud-library-onboarding.json"
HOME_IDENTITY_TIMEOUT = 5.0
LOGIN_PROMPT = "Sign in to save libraries to AutoFA Cloud Library?"


def home_cloud_status(
    config: Any,
    *,
    credential_store: CredentialStore | None = None,
    marker_root: Path | None = None,
    prompt_fn: Callable[..., bool],
    login_fn: Callable[..., RemoteIdentity] = login_to_remote_library,
    client_factory: Callable[[str, str, str], Any] | None = None,
    console: Console | Any | None = None,
) -> str:
    """Resolve one home-entry status and optionally run first-use onboarding."""
    from autofa.services import remote_library_enabled

    if not remote_library_enabled(config):
        return ""
    con = console or Console()
    if setting_value(config, "remote_library_mode") == "server":
        return _server_identity_status(config, client_factory=client_factory)
    configured_url = _configured_base_url(config)
    if _targets_loopback(configured_url):
        try:
            base_url = validate_service_url(configured_url)
        except ValueError:
            return "AutoFA Local Library: invalid configuration · use an HTTP loopback origin"
        return _local_identity_status(base_url, client_factory=client_factory)

    try:
        base_url = normalize_service_url(configured_url)
    except CredentialError:
        return _negative_status("unavailable")

    store = credential_store or KeyringCredentialStore()
    env_name = str(
        setting_value(config, "remote_library_token_env")
        or default_value("remote_library_token_env")
    ).strip()
    try:
        credential = find_credential(
            base_url,
            env_name=env_name,
            store=store,
        )
    except CredentialError:
        return _negative_status("unavailable")

    if credential is not None:
        return _identity_status(
            base_url,
            credential.token,
            credential.source,
            client_factory=client_factory,
        )

    try:
        marker = first_prompt_marker_path(marker_root)
    except OSError:
        return _negative_status("unavailable")
    if first_prompt_was_shown(marker):
        return _not_logged_in_status()

    # Mark before displaying the prompt so Ctrl-C does not create a prompt loop.
    try:
        mark_first_prompt_shown(marker)
    except OSError:
        con.print(
            "[yellow]Cloud sign-in setup is unavailable; continuing locally. "
            "`autofa login` remains available.[/yellow]"
        )
        return _negative_status("unavailable")
    try:
        should_login = prompt_fn(LOGIN_PROMPT, default_yes=True)
    except (KeyboardInterrupt, EOFError):
        con.print(
            "[yellow]Cloud sign-in was interrupted; "
            "`autofa login` remains available.[/yellow]"
        )
        return _not_logged_in_status()
    except Exception:  # noqa: BLE001
        con.print(
            "[yellow]Cloud sign-in prompt is unavailable; continuing locally. "
            "`autofa login` remains available.[/yellow]"
        )
        return _negative_status("unavailable")
    if not should_login:
        con.print("[dim]Skipped cloud sign-in; `autofa login` remains available.[/dim]")
        return _not_logged_in_status()

    try:
        identity = login_fn(
            config,
            credential_store=store,
            console=con,
        )
    except (KeyboardInterrupt, EOFError):
        con.print(
            "[yellow]Cloud sign-in was interrupted; continuing locally. "
            "`autofa login` remains available.[/yellow]"
        )
        return _not_logged_in_status()
    except Exception:  # noqa: BLE001
        con.print(
            "[yellow]Cloud sign-in is unavailable; continuing locally. "
            "Run `autofa login` to retry.[/yellow]"
        )
        return _negative_status("unavailable")
    # login_to_remote_library has already validated this identity with one /me.
    return f"AutoFA Cloud Library: logged in as @{identity.username}"


def _server_identity_status(config: Any, *, client_factory) -> str:
    """A configured server uses a shared token, without GitLab onboarding."""
    try:
        client = client_from_config(config)
        if client_factory is not None:
            client = client_factory(client.base_url, client.token, client.credential_source)
        client.timeout = HOME_IDENTITY_TIMEOUT
        identity = client.me()
        if identity.auth_method != "local_token":
            raise RemoteLibraryError("Server token was not authenticated", status=401)
    except RemoteLibraryError as exc:
        state = "invalid token" if exc.status in {401, 403} else "unavailable"
        return f"AutoFA server: {state} · check Library server in Settings"
    except Exception:  # noqa: BLE001
        return "AutoFA server: unavailable · check Library server in Settings"
    return f"AutoFA server: ready · {client.base_url}"


def _local_identity_status(
    base_url: str,
    *,
    client_factory: Callable[[str, str, str], Any] | None,
) -> str:
    factory = client_factory or (
        lambda url, token, source: RemoteLibraryClient(
            url,
            token=token,
            credential_source=source,
            timeout=HOME_IDENTITY_TIMEOUT,
        )
    )
    try:
        factory(base_url, service_token(), "local").me()
    except Exception:  # noqa: BLE001
        return "AutoFA Local Library: unavailable · run autofa serve to retry"
    return f"AutoFA Local Library: ready · {base_url}"


def first_prompt_marker_path(root: Path | None = None) -> Path:
    marker_root = Path(root) if root is not None else autofa_home()
    return marker_root / FIRST_PROMPT_MARKER


def first_prompt_was_shown(path: Path) -> bool:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and data.get("version") == 1 and data.get("prompted") is True


def mark_first_prompt_shown(path: Path) -> Path:
    """Atomically persist the non-sensitive one-time prompt marker."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.partial"
    try:
        temporary.write_text(
            json.dumps({"version": 1, "prompted": True}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _identity_status(
    base_url: str,
    token: str,
    source: str,
    *,
    client_factory: Callable[[str, str, str], Any] | None,
) -> str:
    factory = client_factory or (
        lambda url, value, credential_source: RemoteLibraryClient(
            url,
            token=value,
            credential_source=credential_source,
            timeout=HOME_IDENTITY_TIMEOUT,
        )
    )
    try:
        identity = factory(base_url, token, source).me()
    except RemoteLibraryError as exc:
        if exc.status in {401, 403}:
            return _negative_status("credential expired/invalid")
        return _negative_status("unavailable")
    except Exception:  # noqa: BLE001
        return _negative_status("unavailable")
    return f"AutoFA Cloud Library: logged in as @{identity.username}"


def _not_logged_in_status() -> str:
    return _negative_status("not logged in")


def _negative_status(state: str) -> str:
    return (
        f"AutoFA Cloud Library: {state} "
        "· launch autofa login first to log in"
    )


def _configured_base_url(config: Any) -> str:
    return str(setting_value(config, "remote_library_base_url") or "").strip()
