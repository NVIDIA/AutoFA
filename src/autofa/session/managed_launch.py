# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured bindings between a native run and its supervising host campaign.

These hooks do nothing outside an explicitly managed launch. The portable host
controller owns the receipt; native code writes bounded preparation/binding
events. Continued discovery is independent of monitoring unless the user explicitly
selects a pause-on-supervision-loss policy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from autofa.util.atomic import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Managed launch record exceeds its size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Managed launch record must be an object")
    return value


def launch_context() -> tuple[Path, dict[str, Any]] | None:
    """Return the exact launch receipt inherited from the managed worker."""
    named = os.environ.get("AUTOFA_DRIVER_LAUNCH")
    if not named:
        return None
    path = Path(named).expanduser().resolve(strict=True)
    record = _read(path)
    if record.get("launch_id") != os.environ.get("AUTOFA_DRIVER_LAUNCH_ID"):
        raise ValueError("Managed launch identifier does not match its receipt")
    return path, record


def preparation_event(stage: str, *, diagnostic: dict[str, Any] | None = None) -> None:
    """Save one small stage event without URLs, paths, credentials or exception text."""
    context = launch_context()
    if context is None:
        return
    path, record = context
    allowed = {"error_type", "errno", "http_status", "retryable", "attempt", "max_attempts"}
    safe = {
        key: value
        for key, value in (diagnostic or {}).items()
        if key in allowed and isinstance(value, (str, int, float, bool, type(None)))
    }
    atomic_write_json(
        path.parent / "preparation.json",
        {
            "launch_id": record["launch_id"],
            "stage": stage[:100],
            "updated_at_epoch": time.time(),
            "diagnostic": safe,
        },
    )


def bind_run(run_dir: Path, run_id: str | None = None) -> None:
    """Bind a created/resumed run by inherited launch ID, never by directory ordering."""
    context = launch_context()
    if context is None:
        return
    path, receipt = context
    campaign = _read(Path(receipt["campaign_path"]) / "campaign.json")
    if campaign["id"] != receipt["campaign_id"] or campaign["state"] != "active":
        raise ValueError("Managed launch campaign is stopped or no longer matches")
    directory = Path(run_dir).resolve(strict=True)
    name = run_id or directory.name
    if not name or Path(name).name != name:
        raise ValueError("Invalid managed native run ID")
    binding = {
        "launch_id": receipt["launch_id"],
        "campaign_id": receipt["campaign_id"],
        "campaign_path": receipt["campaign_path"],
        "receipt_path": str(path),
        "run_id": name,
        "run_dir": str(directory),
    }
    (directory / "supervision").mkdir(exist_ok=True)
    existing = directory / "supervision" / "driver_launch.json"
    if existing.exists():
        saved = _read(existing)
        if saved.get("campaign_id") != binding["campaign_id"]:
            raise ValueError("Run belongs to a different managed campaign")
    atomic_write_json(existing, binding)
    folder = path.parent / "bindings"
    folder.mkdir(exist_ok=True)
    atomic_write_json(folder / (name + ".json"), binding)
    preparation_event("run_bound")
    enforce_supervision(directory, phase="discovery")


def monitoring_settings(binding: dict[str, Any]) -> dict[str, Any]:
    """Read live monitoring preferences without changing the native run's policy."""
    root = Path(binding["campaign_path"]) / "supervision"
    from autofa.session.recovery import read_json

    config = read_json(root / "config.json")
    saved = read_json(root / "monitoring.json")
    return {
        "enabled": saved.get("enabled", config.get("monitoring_enabled", True)),
        "review_seconds": saved.get("review_seconds", config.get("review_seconds", 600)),
        "on_host_loss": saved.get("on_host_loss", config.get("on_host_loss", "continue")),
        "revision": saved.get("revision", 0),
        "host_session_id": saved.get("host_session_id", config.get("host", {}).get("session_id")),
    }


def supervision_available(binding: dict[str, Any], *, now: float | None = None) -> bool:
    """Report actual monitoring liveness; availability does not grant permission to run."""
    clock = time.time() if now is None else now
    root = Path(binding["campaign_path"])
    try:
        campaign = _read(root / "campaign.json")
        owner = _read(root / "supervision" / "owner.json")
        config = _read(root / "supervision" / "config.json")
        monitoring = monitoring_settings(binding)
        if not monitoring["enabled"]:
            return False
        interval = monitoring["review_seconds"]
        grace = config.get("host_grace_seconds", 420)
        guard_grace = max(45, config.get("poll_seconds", 15) * 3)
        guard_live = (
            campaign["id"] == binding["campaign_id"] and campaign["state"] == "active"
            and owner.get("state") == "running"
            and clock <= owner["heartbeat_epoch"] + guard_grace
        )
        if config.get("supervision_mode", "reviewer") != "host":
            state = _read(root / "supervision" / "state.json")
            return bool(guard_live and state.get("generation") == owner.get("generation")
                        and state.get("last_completed_review_at")
                        and clock <= state["last_completed_review_at"]
                        + interval + grace)
        host = _read(root / "supervision" / "host-ack.json")
        last_review = host.get("last_review_at_epoch", 0)
        return bool(
            guard_live
            and host.get("generation") == owner.get("generation")
            and host.get("session_id") == monitoring["host_session_id"]
            and host.get("revision") == campaign["revision"]
            and host.get("monitoring_revision", 0) == monitoring["revision"]
            and clock <= last_review + interval + grace
            and clock <= host["acknowledged_at_epoch"] + interval + grace
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def enforce_supervision(run_dir: Path, *, phase: str = "discovery") -> None:
    """Honor explicit stops and opt-in supervision leases; normally run independently.

    Existing completed/in-flight results still commit. A reconnect does not
    automatically release a pause; the host must inspect it and explicitly resume.
    """
    path = Path(run_dir) / "supervision" / "driver_launch.json"
    if not path.is_file():
        return
    binding = _read(path)
    from autofa.session.recovery import read_json

    campaign = read_json(Path(binding["campaign_path"]) / "campaign.json")
    campaign_stopped = (campaign.get("id") == binding.get("campaign_id")
                        and campaign.get("state") == "stopped")
    if not campaign_stopped and phase not in {"discovery", "round_validation"}:
        return
    monitoring = monitoring_settings(binding)
    if not campaign_stopped and (
        not monitoring["enabled"] or monitoring["on_host_loss"] != "pause"
        or supervision_available(binding)
    ):
        return
    from autofa.session.supervision import (
        SupervisorInterrupted,
        lifecycle_action,
        lifecycle_request,
    )

    action = lifecycle_action(Path(run_dir))
    if campaign_stopped:
        if action != "stop":
            action = lifecycle_request(Path(run_dir), "stop")["action"]
    elif action not in {"pause", "stop"}:
        atomic_write_json(
            Path(run_dir) / "supervision" / "host_loss.json",
            {
                "reason": "host_supervision_lost",
                "at_epoch": time.time(),
                "launch_id": binding["launch_id"],
            },
        )
        receipt = lifecycle_request(Path(run_dir), "pause", preserve_stop=True)
        action = receipt["action"]
    raise SupervisorInterrupted(Path(run_dir), action)
