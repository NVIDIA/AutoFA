# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OS-keychain storage and resolution for AutoFA Cloud Library credentials."""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import Any, Protocol

import keyring
from keyring.errors import KeyringError, NoKeyringError, PasswordDeleteError

KEYRING_SERVICE_PREFIX = "autofa-cloud-library:"
KEYRING_ACCOUNT = "cli-token"
CLI_TOKEN_PREFIX = "autofa_cli_"


class CredentialError(RuntimeError):
    """A safe, actionable credential-store failure."""


class CredentialStore(Protocol):
    """Injectable token store used by login, logout, and client construction."""

    def get_token(self, base_url: str) -> str | None: ...

    def set_token(self, base_url: str, token: str) -> None: ...

    def delete_token(self, base_url: str) -> bool: ...


@dataclass(frozen=True)
class ResolvedCredential:
    token: str
    source: str  # env | keychain


class KeyringCredentialStore:
    """Store one CLI service token per normalized service URL."""

    def __init__(self, backend: Any | None = None):
        self._injected_backend = backend

    def get_token(self, base_url: str) -> str | None:
        backend = self._backend()
        try:
            value = backend.get_password(_service_name(base_url), KEYRING_ACCOUNT)
        except (KeyringError, NoKeyringError, RuntimeError) as exc:
            raise CredentialError(_keyring_error_message()) from exc
        token = str(value).strip() if value is not None else ""
        return token or None

    def set_token(self, base_url: str, token: str) -> None:
        value = str(token).strip()
        if not value.startswith(CLI_TOKEN_PREFIX):
            raise CredentialError("Refusing to store an invalid AutoFA CLI token")
        backend = self._backend()
        try:
            backend.set_password(_service_name(base_url), KEYRING_ACCOUNT, value)
        except (KeyringError, NoKeyringError, RuntimeError) as exc:
            raise CredentialError(_keyring_error_message()) from exc

    def delete_token(self, base_url: str) -> bool:
        backend = self._backend()
        service = _service_name(base_url)
        try:
            existing = backend.get_password(service, KEYRING_ACCOUNT)
            if existing is None:
                return False
            backend.delete_password(service, KEYRING_ACCOUNT)
            return True
        except PasswordDeleteError:
            return False
        except (KeyringError, NoKeyringError, RuntimeError) as exc:
            raise CredentialError(_keyring_error_message()) from exc

    def _backend(self) -> Any:
        try:
            backend = (
                self._injected_backend
                if self._injected_backend is not None
                else keyring.get_keyring()
            )
            priority = float(backend.priority)
        except Exception as exc:  # noqa: BLE001
            raise CredentialError(_keyring_error_message()) from exc
        if priority <= 0 or not _backend_is_secure(backend):
            raise CredentialError(_keyring_error_message())
        return backend


def resolve_credential(
    base_url: str,
    *,
    env_name: str,
    store: CredentialStore | None = None,
) -> ResolvedCredential:
    """Resolve explicit environment credentials before the OS keychain."""
    credential = find_credential(base_url, env_name=env_name, store=store)
    if credential is not None:
        return credential
    raise CredentialError(
        f"No AutoFA Cloud Library credential is available. "
        f"Run `autofa login` or set {env_name}."
    )


def find_credential(
    base_url: str,
    *,
    env_name: str,
    store: CredentialStore | None = None,
) -> ResolvedCredential | None:
    """Return the preferred credential, or ``None`` when neither source exists."""
    environment_token = os.environ.get(env_name, "").strip()
    if environment_token:
        return ResolvedCredential(token=environment_token, source="env")
    credential_store = store or KeyringCredentialStore()
    keychain_token = credential_store.get_token(base_url)
    if keychain_token:
        return ResolvedCredential(token=keychain_token, source="keychain")
    return None


def normalize_service_url(value: str) -> str:
    """Canonicalize a configured service URL for keychain identity."""
    raw = str(value).strip()
    parsed = urllib.parse.urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise CredentialError("AutoFA Cloud Library URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CredentialError(
            "AutoFA Cloud Library URL cannot contain credentials, query, or fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise CredentialError("AutoFA Cloud Library URL has an invalid port") from exc
    hostname = parsed.hostname.lower().rstrip(".")
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = (scheme == "https" and port == 443) or (
        scheme == "http" and port == 80
    )
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((scheme, netloc, path, "", ""))


def _service_name(base_url: str) -> str:
    return f"{KEYRING_SERVICE_PREFIX}{normalize_service_url(base_url)}"


def _keyring_error_message() -> str:
    return (
        "No usable OS keychain backend is available. Configure macOS Keychain, "
        "Windows Credential Locker, or Secret Service; alternatively set GITLAB_TOKEN."
    )


def _backend_is_secure(backend: Any) -> bool:
    backend_name = f"{type(backend).__module__}.{type(backend).__qualname__}".lower()
    if any(marker in backend_name for marker in ("fail", "null", "plaintext")):
        return False
    children = getattr(backend, "backends", None)
    if children:
        # A chainer that contains a plaintext backend may silently fall through
        # to it when the secure backend is unavailable.
        return all(_backend_is_secure(child) for child in children)
    return True
