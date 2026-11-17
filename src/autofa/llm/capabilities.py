# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Account-scoped API discovery; outages never become unsupported-route decisions."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from typing import Any

TRANSPORT_CONTRACT_VERSION = 2
_LOCK = threading.Lock()
_CACHE: dict[tuple, bool] = {}
_DISCOVERY_LOCKS: dict[tuple, threading.Lock] = {}


class APIUnavailable(RuntimeError):
    """The selected endpoint/account/model explicitly lacks this API."""

    def __init__(self, api: str, status_code: int | None = None) -> None:
        self.api, self.status_code = api, status_code
        super().__init__(f"{api.title()} API unavailable for the selected endpoint and model"
                         + (f" (HTTP {status_code})" if status_code else ""))


def _key(client: Any, model: str, api: str) -> tuple:
    credential = getattr(client, "api_key", None)
    # Keep credentials out of cache keys, diagnostics and persisted fingerprints.
    account = hashlib.sha256(json.dumps([
        credential if isinstance(credential, str) else id(client),
        getattr(client, "organization", None), getattr(client, "project", None),
    ], default=str).encode()).hexdigest()
    return (str(getattr(client, "base_url", "")).rstrip("/"), model, account, api)


def unsupported_api(exc: Exception) -> bool:
    if isinstance(exc, APIUnavailable):
        return True
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {}
    code = error.get("code") or getattr(exc, "code", None)
    message = str(error.get("message") or "").lower()
    explicit = code in {"unsupported_endpoint", "endpoint_not_found", "route_not_found",
                        "unsupported_api", "model_not_supported", "unsupported_model",
                        "not_implemented"}
    if status in {401, 403, 429} or (status and status >= 500 and status != 501):
        return False
    if status == 501:
        return explicit
    if status in {400, 422}:
        return explicit or (error.get("param") == "model" and any(
            phrase in message for phrase in ("not supported", "does not support", "unsupported")
        ))
    if status in {404, 405}:
        # A bad model ID is not proof that a different API would work.
        return not (code == "model_not_found" or ("model" in message and any(
            phrase in message for phrase in ("not found", "does not exist", "no access")
        )))
    return False


def cached_capability(client: Any, model: str, api: str) -> bool | None:
    with _LOCK:
        return _CACHE.get(_key(client, model, api))


def invoke_api[T](client: Any, model: str, api: str, request: Callable[[], T]) -> T:
    """Use one real request for discovery, with single-flight negative caching."""
    from autofa.util.cancellation import check_cancelled

    key = _key(client, model, api)

    def invoke() -> T:
        # Cancellation may arrive while another request owns the discovery gate.
        check_cancelled()
        try:
            result = request()
        except Exception as exc:
            # A failed in-flight request is not a capability finding when its
            # enclosing operation has been cancelled. In particular, do not
            # cache an unavailable route or fall back to another paid request.
            check_cancelled()
            if not unsupported_api(exc):
                raise
            with _LOCK:
                _CACHE[key] = False
            raise APIUnavailable(api, getattr(exc, "status_code", None)) from exc
        with _LOCK:
            _CACHE[key] = True
        return result

    with _LOCK:
        supported = _CACHE.get(key)
        gate = _DISCOVERY_LOCKS.setdefault(key, threading.Lock())
    if supported is False:
        raise APIUnavailable(api)
    if supported is True:
        return invoke()
    with gate:
        with _LOCK:
            supported = _CACHE.get(key)
        if supported is False:
            raise APIUnavailable(api)
        if supported is None:
            return invoke()
    return invoke()


def routing_identity(client: Any, model: str, api: str) -> dict:
    """No-network fingerprint of the selected and observed transport contract."""
    namespace = hashlib.sha256(json.dumps(_key(client, model, api)).encode()).hexdigest()
    return {"transport_contract_version": TRANSPORT_CONTRACT_VERSION, "api": api,
            "endpoint_account_namespace": namespace,
            "decisions_available": cached_capability(client, model, "decisions"),
            "responses_available": cached_capability(client, model, "responses")}
