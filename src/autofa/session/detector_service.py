# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact discovery service observations; outages are not detector measurements."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from autofa.session.recovery import read_json
from autofa.util.atomic import atomic_write_json
from autofa.util.model_errors import model_error_diagnostic
from autofa.util.usage_values import usage_endpoint

STATE_FILE = "detector_service_health.json"
_LOCK = threading.RLock()
_PHASES = {"discovery", "round_validation", "finalization"}


def failure_kind(error: BaseException) -> str:
    """Classify transport/configuration failures separately from local detector faults."""
    status = getattr(error, "status_code", None)
    diagnostic = model_error_diagnostic(error)
    text = (str(error) + " " + diagnostic["message"]).lower()
    code = str(diagnostic.get("code", "")).lower()
    param = str(diagnostic.get("param", "")).lower()
    if status in {401, 403} or status is None and any(s in text for s in (
            "http 401", "http 403", "api key", "not set")):
        return "authentication"
    if status == 429 or status is None and ("http 429" in text or "rate limit" in text):
        return "rate_limit"
    if (status == 404 or status is None and "http 404" in text or code in {
            "model_not_found", "deployment_not_found", "unsupported_model", "invalid_model",
            "model_not_supported", "invalid_api_version", "invalid_endpoint",
    } or param in {"model", "deployment", "api_version", "base_url", "endpoint"}
            or any(s in text for s in ("model not found", "unknown model",
                                      "unsupported model", "invalid model", "model does not exist",
                                      "deployment not found"))):
        return "configuration"
    if status in {400, 422} or "http 400" in text or "http 422" in text:
        # Invalid images, context limits and other item payload rejections do not
        # establish a service outage. Startup readiness can still reject this
        # route explicitly; otherwise retain the item as unresolved evidence.
        return "request_rejected"
    if (isinstance(status, int) and status >= 500 or isinstance(error, (TimeoutError, OSError))
            or any(s in text for s in ("http 50", "timed out", "timeout", "connection",
                                      "service unavailable"))):
        return "transport"
    if isinstance(error, (TypeError, ValueError, AttributeError)):
        return "detector_code"
    return "unclassified"


def observe_dispatch(context: dict | None, *, model: str, endpoint: str | None,
                     api: str, settings: dict | None = None,
                     error: BaseException | None = None, readiness_failure: bool = False) -> None:
    """Observe actual discovery calls only; never infer failure labels or change outputs."""
    if (not context or context.get("phase") not in _PHASES
            or context.get("role") not in {"detector", "classifier_probe"}):
        return
    directory = Path(context["run_dir"])
    config = read_json(directory / "config.json")
    if (not config.get("run_id")
            or (config.get("analysis_outputs") and "agent" not in config["analysis_outputs"])):
        return  # Setup probes and unrelated workflows do not create discovery incidents.
    identity = {"model": model, "endpoint": usage_endpoint(endpoint), "api": api,
                "role": context["role"], "settings": settings or {}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    now = datetime.now(UTC)
    with _LOCK:
        state = read_json(directory / STATE_FILE) or {"version": 1, "routes": {}}
        routes = state.setdefault("routes", {})
        row = routes.get(key, {**identity, "failure_streak": 0, "successes": 0, "failures": 0})
        row["observed_at"] = now.isoformat()
        if error is None:
            row.update(last_success=now.isoformat(), failure_streak=0, next_retry_at=None)
            row["successes"] += 1
        else:
            row["failure_streak"] += 1
            row["failures"] += 1
            row.update(last_failure=now.isoformat(), category=failure_kind(error),
                       diagnostic=model_error_diagnostic(error),
                       next_retry_at=(now + timedelta(seconds=min(
                           300, 30 * 2 ** min(row["failure_streak"] - 1, 4)))).isoformat(),
                       affected_work={"phase": context["phase"],
                                      "round": context.get("round_index"),
                                      "usage_event_id": context.get("usage_event_id")})
        routes[key] = row
        # Probes for inactive alternatives cannot block a working selected service.
        service_failure = error is not None and (readiness_failure or row.get("category") in {
            "authentication", "configuration", "transport", "rate_limit",
        })
        if context["role"] == "detector" and (error is None or service_failure):
            state["active_route"] = key
            state["blocked"] = service_failure
        if len(routes) > 32:
            for old in sorted(routes, key=lambda k: routes[k].get("observed_at", "")):
                if old != state.get("active_route"):
                    routes.pop(old)
                    if len(routes) <= 32:
                        break
        state["updated_at"] = now.isoformat()
        atomic_write_json(directory / STATE_FILE, state)


def record_readiness(rt: Any, readiness: dict, *, blocking: bool = True) -> None:
    """A rejected startup probe blocks required semantic work, including client setup errors."""
    if not getattr(rt, "run", None):
        return
    from autofa.llm.endpoints import endpoint_for_config

    endpoint = endpoint_for_config(rt.config)
    error = (None if readiness.get("available") is True else
             RuntimeError(readiness.get("error") or "Detector service readiness unverified"))
    observe_dispatch({"run_dir": str(rt.run.run_dir()),
                      "role": "detector" if blocking else "classifier_probe",
                      "phase": "discovery", "round_index": rt.run.round},
                     model=endpoint.model, endpoint=endpoint.base_url, api=endpoint.api,
                     settings=endpoint.request_options, error=error,
                     readiness_failure=error is not None)


def detector_service_status(run_dir: Path) -> dict:
    """Read a bounded service summary without modifying state or probing the provider."""
    state = read_json(Path(run_dir) / STATE_FILE)
    if not state:
        return {"status": "unobserved", "infrastructure_blocked": False}
    active = state.get("routes", {}).get(state.get("active_route"), {})
    blocked = bool(state.get("blocked"))
    return {"status": "infrastructure_blocked" if blocked else "ready",
            "infrastructure_blocked": blocked, "incident_id": state.get("active_route"),
            "updated_at": state.get("updated_at"), "active": active,
            "route_count": len(state.get("routes", {})),
            "next_action": ("Use classifier_profiles(action='recover') with unchanged positive/"
                            "negative source controls, or request a bounded wait/configuration "
                            "change. Unavailable calls are unresolved work, not negative matches."
                            if blocked else None)}


def infrastructure_block(rt: Any) -> dict | None:
    status = detector_service_status(rt.run.run_dir())
    return status if status["infrastructure_blocked"] else None


def recovery_turn_guidance(rt: Any) -> str | None:
    """Bound paid recovery prompting across Resume; waiting needs an outer owner decision."""
    with _LOCK:
        path = rt.run.run_dir() / STATE_FILE
        state = read_json(path)
        if not state.get("blocked"):
            return None
        key = state.get("active_route")
        turns = state.setdefault("recovery_turns", {})
        if turns.get(key, 0) >= 2:
            return None
        turns[key] = turns.get(key, 0) + 1
        atomic_write_json(path, state)
        status = detector_service_status(rt.run.run_dir())
    return ("DETECTOR SERVICE RECOVERY: semantic work is infrastructure_blocked. "
            + str(status["next_action"]) + " Preserve all source evidence, unfinished candidate "
            "work and current grades. Do not advance rounds or call this scientific exhaustion. "
            "The native recovery allowance is finite; if recovery fails, return the concrete "
            "blocker for a resumable partial report instead of repeating paid turns.")
