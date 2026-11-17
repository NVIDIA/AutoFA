#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded host observations and managed launches; no model calls or shell commands.

Run through supervisor.py's observe/watch/ack/launch entrypoints. The same guard
owns native fencing in host and detached-review modes; only the current user
session performs quality reviews in host mode.
"""

from __future__ import annotations

import hashlib
import importlib.util
import ipaddress
import os
import socket
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

_SPEC = importlib.util.spec_from_file_location(
    "autofa_host_records", Path(__file__).with_name("driver.py")
)
records = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(records)


def document(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Host supervision record exceeds 2 MiB")
        return records.object_file(path)
    except FileNotFoundError:
        if default is not None:
            return default.copy()
        raise


def process_identity(pid: int) -> dict[str, Any]:
    """Record numeric birth time with fixed locale/timezone, never PID alone."""
    marker = None
    if os.name == "posix":
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart="],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
                env={**os.environ, "TZ": "UTC", "LC_ALL": "C"},
            )
            if result.stdout.strip():
                value = datetime.strptime(result.stdout.strip(), "%a %b %d %H:%M:%S %Y")
                marker = int(value.replace(tzinfo=timezone.utc).timestamp())  # noqa: UP017
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return {"pid": pid, "hostname": socket.gethostname(), "started_epoch": marker}


def process_alive(identity: dict[str, Any]) -> bool | None:
    # Signal 0 is a liveness probe only on POSIX; on Windows it terminates the process.
    if os.name != "posix":
        return None
    if identity.get("hostname") != socket.gethostname() or not identity.get("pid"):
        return None
    try:
        os.kill(identity["pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    marker = identity.get("started_epoch")
    if marker is None:
        return None
    current = process_identity(identity["pid"])["started_epoch"]
    return current == marker if current is not None else None


def monitoring_policy(root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read live review preferences independently of native execution and its limits."""
    config = config if config is not None else document(root / "supervision" / "config.json")
    saved = document(root / "supervision" / "monitoring.json", {})
    policy = {
        "version": 1,
        "revision": saved.get("revision", 0),
        "enabled": saved.get("enabled", config.get("monitoring_enabled", True)),
        "review_seconds": saved.get("review_seconds", config.get("review_seconds", 600)),
        "on_host_loss": saved.get("on_host_loss", config.get("on_host_loss", "continue")),
        "host_session_id": saved.get("host_session_id", config.get("host", {}).get("session_id")),
    }
    if type(policy["enabled"]) is not bool:
        raise ValueError("Monitoring enabled must be true or false")
    if type(policy["revision"]) is not int or policy["revision"] < 0:
        raise ValueError("Monitoring revision must be a nonnegative integer")
    seconds = policy["review_seconds"]
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not 1 <= seconds <= 86400):
        raise ValueError("review_seconds must be between 1 and 86400")
    if policy["on_host_loss"] not in {"continue", "pause"}:
        raise ValueError("on_host_loss must be continue or pause")
    return policy


def configure_monitoring(
    root: Path, action: str, *, generation: str | None = None,
    review_seconds: float | None = None, on_host_loss: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Pause/reconnect review work without changing native goals, processes or lifecycle."""
    if action not in {"status", "pause", "resume", "configure"}:
        raise ValueError("Unknown monitoring action")
    with records.state_lock(root):
        config = document(root / "supervision" / "config.json")
        current = monitoring_policy(root, config)
        if action == "status":
            if any(value is not None for value in (review_seconds, on_host_loss, session_id)):
                raise ValueError("Monitoring status does not change settings")
            return current
        owner = document(root / "supervision" / "owner.json")
        if not generation or generation != owner.get("generation"):
            raise ValueError("Current supervisor generation required; inspect status and reattach")
        if session_id is not None and (action != "resume" or not session_id.strip()
                                       or config.get("supervision_mode") != "host"):
            raise ValueError("A host session ID is supported only by monitoring resume")
        updated = dict(current)
        if action in {"pause", "resume"}:
            updated["enabled"] = action == "resume"
        if session_id is not None:
            updated["host_session_id"] = session_id
        if review_seconds is not None:
            updated["review_seconds"] = review_seconds
        if on_host_loss is not None:
            updated["on_host_loss"] = on_host_loss
        # Changing the observer invalidates delayed actions without changing the goal.
        if any(updated[key] != current[key] for key in ("enabled", "host_session_id")):
            updated["revision"] += 1
        seconds = updated["review_seconds"]
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not 1 <= seconds <= 86400):
            raise ValueError("review_seconds must be between 1 and 86400")
        if updated["on_host_loss"] not in {"continue", "pause"}:
            raise ValueError("on_host_loss must be continue or pause")
        updated["updated_at_epoch"] = time.time()
        records.atomic(root / "supervision" / "monitoring.json", updated)
        return {**updated, "native_execution_changed": False,
                "acknowledgment_required": updated["enabled"]
                and updated["revision"] != current["revision"]}


def owner_state(
    root: Path, generation: str, *, now: float | None = None, allow_delivery: bool = False
) -> tuple[dict, dict, dict]:
    clock = time.time() if now is None else now
    campaign = document(root / "campaign.json")
    config = document(root / "supervision" / "config.json")
    owner = document(root / "supervision" / "owner.json")
    if config.get("supervision_mode") != "host":
        raise ValueError("This command requires host-owned supervision")
    if owner.get("generation") != generation:
        raise ValueError("Host supervision owner changed; reattach before acting")
    if allow_delivery and campaign["state"] == "stopped":
        if (campaign["goal"].get("delivery") or {}).get("publish_on_stop") is not True:
            raise ValueError("Saved delivery after stop is not authorized by the campaign")
        if owner.get("state") not in {"running", "stopped"}:
            raise ValueError("Saved-delivery owner is unavailable")
        return campaign, config, owner
    if owner.get("state") != "running":
        raise ValueError("Host supervision owner changed; reattach before acting")
    if clock > owner.get("heartbeat_epoch", 0) + max(45, config.get("poll_seconds", 15) * 3):
        raise ValueError("Supervision guard heartbeat is stale; restart and reattach")
    return campaign, config, owner


def host_available(root: Path, generation: str, *, now: float | None = None) -> bool:
    clock = time.time() if now is None else now
    try:
        campaign, config, _ = owner_state(root, generation, now=clock)
        monitoring = monitoring_policy(root, config)
        ack = document(root / "supervision" / "host-ack.json", {})
        grace = config.get("host_grace_seconds", 420)
        interval = monitoring["review_seconds"]
        return bool(
            campaign["state"] == "active"
            and monitoring["enabled"]
            and ack.get("generation") == generation
            and ack.get("revision") == campaign["revision"]
            and ack.get("monitoring_revision", 0) == monitoring["revision"]
            and ack.get("session_id") == monitoring["host_session_id"]
            and clock <= ack.get("acknowledged_at_epoch", 0) + interval + grace
            and clock <= ack.get("last_review_at_epoch", 0) + interval + grace
        )
    except (OSError, KeyError, ValueError, TypeError):
        return False


def snapshot(root: Path, generation: str, *, now: float | None = None) -> dict[str, Any]:
    """Only bounded native projections/events; never inline native logs or trace records."""
    clock = time.time() if now is None else now
    campaign, config, owner = owner_state(root, generation, now=clock, allow_delivery=True)
    delivery_only = campaign["state"] == "stopped"
    state = document(root / "supervision" / "state.json", {})
    ack = document(root / "supervision" / "host-ack.json", {})
    monitoring = monitoring_policy(root, config)
    next_review = ack.get("last_review_at_epoch", 0) + monitoring["review_seconds"]
    launches = []
    for path in sorted((root / "supervision" / "launches").glob("*/receipt.json"))[-16:]:
        receipt = document(path)
        launches.append(
            {
                key: receipt.get(key)
                for key in (
                    "launch_id",
                    "state",
                    "exit_code",
                    "phase",
                    "worker",
                    "process",
                    "log_path",
                    "run_ids",
                    "failure",
                    "updated_at_epoch",
                )
            }
        )
    native = state.get("observations", {})
    observations = {
        run_id: {
            key: value.get(key)
            for key in (
                "round",
                "grade_counts",
                "phase",
                "turn",
                "campaign",
                "lifecycle",
                "control",
                "usage",
                "snapshots",
                "library_sha256",
                "stale",
                "observation_error",
                "observation_path",
                "detector_service",
            )
        }
        for run_id, value in list(native.items())[:32]
    }
    result = {
        "campaign_id": campaign["id"],
        "revision": campaign["revision"],
        "campaign_state": campaign["state"],
        "delivery_only": delivery_only,
        "generation": generation,
        "host": {"kind": config["host"]["kind"], "session_id": monitoring["host_session_id"]},
        "monitoring": monitoring,
        "guard_heartbeat_epoch": owner["heartbeat_epoch"],
        "supervision_available": host_available(root, generation, now=clock),
        "review_due": not delivery_only and monitoring["enabled"] and (
            clock >= next_review or bool(state.get("pending_events"))),
        "next_review_at_epoch": None if delivery_only or not monitoring["enabled"] else next_review,
        "events": state.get("pending_events", [])[-32:],
        "event_sequence": state.get("event_sequence", 0),
        "launches": launches,
        "runs": observations,
    }
    # A native extension may grow its status. Preserve paths and core results at
    # the boundary rather than accidentally flooding the host conversation.
    if len(records.canonical(result)) > 64 * 1024:
        result["runs"] = {
            run_id: {
                key: row.get(key)
                for key in (
                    "round",
                    "grade_counts",
                    "phase",
                    "observation_error",
                    "observation_path",
                )
            }
            for run_id, row in observations.items()
        }
        result["detail_notice"] = "Read the referenced bounded native observation for details"
    return result


def notification_fingerprint(root: Path, result: dict[str, Any]) -> str:
    """Wake on important events or review deadlines, not refreshed status counters."""
    path = root / "supervision" / "host-notifications.json"
    with records.state_lock(root):
        saved = document(path, {})
        signal = (dict(saved) if saved.get("generation") == result["generation"] else
                  {"generation": result["generation"], "timer_sequence": 0, "loss_sequence": 0})
        monitoring = result["monitoring"]
        due = result["next_review_at_epoch"]
        if monitoring["enabled"] and not result["delivery_only"]:
            if due is not None and time.time() >= due and signal.get("timer_due") != due:
                signal["timer_sequence"] += 1
                signal["timer_due"] = due
            if signal.get("was_available") and not result["supervision_available"]:
                signal["loss_sequence"] += 1
        signal["was_available"] = result["supervision_available"]
        if saved != signal:
            records.atomic(path, signal)
    comparable = {
        key: result[key] for key in (
            "campaign_id", "revision", "campaign_state", "delivery_only", "generation",
            "host", "monitoring", "event_sequence",
        )
    }
    comparable.update(
        timer_sequence=signal["timer_sequence"], loss_sequence=signal["loss_sequence"],
        run_ids=sorted(result["runs"]),
        launches=[{key: row.get(key) for key in (
            "launch_id", "state", "exit_code", "phase", "worker", "process", "run_ids", "failure",
        )} for row in result["launches"]],
    )
    return hashlib.sha256(records.canonical(comparable)).hexdigest()


def observe(
    root: Path, generation: str, *, after: str | None = None, timeout: float = 0
) -> dict[str, Any]:
    """Wait at most one minute; return a receipt the current host must acknowledge."""
    if not 0 <= timeout <= 60:
        raise ValueError("watch timeout must be between zero and 60 seconds")
    deadline = time.monotonic() + timeout
    while True:
        result = snapshot(root, generation)
        fingerprint = notification_fingerprint(root, result)
        if fingerprint != after or time.monotonic() >= deadline:
            break
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    receipt_id = uuid4().hex
    result.update(observation_id=receipt_id, fingerprint=fingerprint, observed_at_epoch=time.time())
    folder = root / "supervision" / "host-observations"
    folder.mkdir(exist_ok=True)
    with records.state_lock(root):
        # Revalidate after the bounded wait, before creating a usable acknowledgment.
        owner_state(root, generation, allow_delivery=True)
        records.atomic(folder / (receipt_id + ".json"), result)
        existing = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
        for path in existing[:-64]:
            path.unlink(missing_ok=True)
    return result


def watch_stream(root: Path, generation: str, *, after: str | None = None,
                 duration: float = 600) -> Iterator[dict[str, Any]]:
    """Emit bounded JSON events for a host Monitor/background task, then expire explicitly.

    Each individual wait remains at most one minute. The host still acknowledges
    consumed events and performs reviews; the stream cannot renew its own lease.
    """
    if not 1 <= duration <= 1800:
        raise ValueError("stream duration must be between one and 1800 seconds")
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        result = observe(root, generation, after=after,
                         timeout=min(60, max(0, deadline - time.monotonic())))
        changed = result["fingerprint"] != after
        after = result["fingerprint"]
        if changed:
            yield result
    yield {"kind": "watch_expired", "generation": generation, "renew_required": True,
           "after": after}


def acknowledge(
    root: Path,
    generation: str,
    observation_id: str,
    *,
    session_id: str,
    task_id: str,
    assessment: str | None = None,
    summary_sha256: str | None = None,
) -> dict[str, Any]:
    """A host consuming a watch result renews the lease; only review records reset cadence."""
    if not records.re.fullmatch(r"[a-f0-9]{32}", observation_id) or not task_id.strip():
        raise ValueError("A current observation ID and host task/monitor ID are required")
    with records.state_lock(root):
        now = time.time()
        campaign, config, _ = owner_state(root, generation, now=now, allow_delivery=True)
        monitoring = monitoring_policy(root, config)
        delivery_only = campaign["state"] == "stopped"
        observation = document(
            root / "supervision" / "host-observations" / (observation_id + ".json")
        )
        if (
            observation["generation"] != generation
            or observation["revision"] != campaign["revision"]
            or observation["campaign_id"] != campaign["id"]
            or observation.get("monitoring", {}).get("revision", 0) != monitoring["revision"]
            or session_id != monitoring["host_session_id"]
            or now - observation["observed_at_epoch"] > 60
        ):
            raise ValueError("Stale observation or different host; observe and acknowledge again")
        if not delivery_only and not monitoring["enabled"]:
            raise ValueError("Monitoring is paused; resume monitoring before acknowledging")
        if assessment is not None and (not assessment.strip() or len(assessment) > 8000):
            raise ValueError("Review assessment must contain between one and 8000 characters")
        if summary_sha256 is not None and not records.re.fullmatch(r"[a-f0-9]{64}", summary_sha256):
            raise ValueError("summary_sha256 must identify the summary actually sent to the user")
        if delivery_only:
            summary = document(root / "delivery" / "summary.json", {})
            if (not summary_sha256 or summary.get("sha256") != summary_sha256
                    or summary.get("campaign_id") != campaign["id"]
                    or summary.get("revision") != campaign["revision"]):
                raise ValueError(
                    "Stopped delivery acknowledgment requires the current prepared summary"
                )
        previous = document(root / "supervision" / "host-ack.json", {})
        reviewed_at = (now if assessment is not None or previous.get("generation") != generation
                       or previous.get("monitoring_revision", 0) != monitoring["revision"]
                       else previous.get("last_review_at_epoch", 0))
        next_review = reviewed_at + monitoring["review_seconds"]
        receipt = {
            "id": uuid4().hex,
            "observation_id": observation_id,
            "generation": generation,
            "revision": campaign["revision"],
            "monitoring_revision": monitoring["revision"],
            "session_id": session_id,
            "task_id": task_id,
            "acknowledged_at_epoch": now,
            "last_review_at_epoch": reviewed_at,
            "next_review_at_epoch": next_review,
            "assessment": assessment,
            "channel": "foreground",
            "summary_sha256": summary_sha256,
            "scope": "saved_delivery" if delivery_only else "supervision",
        }
        archive = root / "supervision" / "host-acks"
        archive.mkdir(exist_ok=True)
        if delivery_only:
            # Delivery evidence cannot renew supervision, mint a mutation fence,
            # change the owner's lifecycle or erase the user's stop.
            receipt["next_review_at_epoch"] = None
            records.atomic(archive / (receipt["id"] + ".json"), receipt)
            return receipt
        fences = root / "supervision" / "host-fences"
        fences.mkdir(exist_ok=True)
        fence_path = fences / (receipt["id"] + ".json")
        records.atomic(
            fence_path,
            {
                "campaign_path": str(root),
                "revision": campaign["revision"],
                "owner_generation": generation,
                "monitoring_revision": monitoring["revision"],
            },
        )
        receipt["fence_file"] = str(fence_path)
        records.atomic(root / "supervision" / "host-ack.json", receipt)
        if summary_sha256 is not None:
            records.atomic(archive / (receipt["id"] + ".json"), receipt)
        if assessment is not None:
            folder = root / "supervision" / "host-reviews"
            folder.mkdir(exist_ok=True)
            records.atomic(folder / (receipt["id"] + ".json"), receipt)
            state_path = root / "supervision" / "state.json"
            state = document(state_path, {})
            # Do not acknowledge events first observed after this review's snapshot.
            seen = {records.canonical(event) for event in observation.get("events", [])}
            state["pending_events"] = [
                event
                for event in state.get("pending_events", [])
                if records.canonical(event) not in seen
            ]
            state["last_completed_review_at"] = now
            records.atomic(state_path, state)
        return receipt


def _validate_spec(spec: dict[str, Any], config: dict[str, Any]) -> None:
    command = spec.get("argv")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(x, str) or not x for x in command)
    ):
        raise ValueError("Managed launch requires a nonempty argument array")
    if command[: len(config["native"]["argv"])] != config["native"]["argv"]:
        raise ValueError("Managed launch must use the configured native executable")
    if Path(spec.get("cwd", "")).resolve() != Path(config["native"]["cwd"]).resolve():
        raise ValueError("Managed launch cwd must match the configured checkout")
    env = spec.get("env", {})
    allowed = {"AUTOFA_CONFIG", "AUTOFA_HOME", "AUTOFA_SETUP_CONTRACT"}
    if (
        not isinstance(env, dict)
        or set(env) - allowed
        or any(not isinstance(x, str) for x in env.values())
    ):
        raise ValueError(
            "Launch env supports only scoped config/home/contract paths, never credentials"
        )
    files = spec.get("config_files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("Pin the approved config/contract files before launch")
    for name, expected in files.items():
        if not Path(name).is_absolute() or records.digest(name) != expected:
            raise ValueError("Approved launch configuration changed; prepare and agree it again")
    if not env.get("AUTOFA_SETUP_CONTRACT") or env["AUTOFA_SETUP_CONTRACT"] not in files:
        raise ValueError("Launch must pin AUTOFA_SETUP_CONTRACT")
    if not env.get("AUTOFA_CONFIG") or env["AUTOFA_CONFIG"] not in files:
        raise ValueError("Launch must pin its explicit AUTOFA_CONFIG")


def validate_delivery(root: Path, spec: dict[str, Any]) -> None:
    """The agreed destination determines launch publication, not hand-written YAML."""
    module_spec = importlib.util.spec_from_file_location(
        "autofa_host_delivery", Path(__file__).with_name("delivery.py")
    )
    delivery = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(delivery)
    campaign = document(root / "campaign.json")
    required = delivery.execution_settings(campaign["goal"])["publish"]
    contract = document(Path(spec["env"]["AUTOFA_SETUP_CONTRACT"]))
    if contract.get("execution", {}).get("publish") is not required:
        raise ValueError("Launch publication differs from the agreed delivery destination")
    if required:
        destination = contract.get("settings", {}).get("remote_library_base_url", "")
        parsed = urllib.parse.urlsplit(destination)
        name = (parsed.hostname or "").lower().rstrip(".")
        local = name in {"localhost", "localhost.localdomain"} or name.endswith(".localhost")
        try:
            address = ipaddress.ip_address(name)
            local = local or address.is_loopback or address.is_unspecified
        except ValueError:
            pass
        if not name or parsed.scheme not in {"http", "https"} or local:
            raise ValueError("Cloud delivery requires an explicit non-loopback Library URL")


def launch(root: Path, generation: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Write a receipt before Popen; a worker owns exit status even before run creation."""
    with records.state_lock(root):
        campaign, config, _ = owner_state(root, generation)
        if campaign["state"] != "active" or not host_available(root, generation):
            raise ValueError("A current host acknowledgment is required before native launch")
        _validate_spec(spec, config)
        validate_delivery(root, spec)
        # A caller-provided id makes a retried request inspect the same launch.
        launch_id = spec.get("launch_id", uuid4().hex)
        if not isinstance(launch_id, str) or not records.re.fullmatch(r"[a-f0-9]{32}", launch_id):
            raise ValueError("launch_id must be a 32-character hex identifier")
        directory = root / "supervision" / "launches" / launch_id
        if directory.exists():
            saved = document(directory / "receipt.json")
            if saved["spec"] != spec:
                raise ValueError("Conflicting launch ID")
            return saved
        directory.mkdir(parents=True)
        native_env = {**config["native"].get("env", {}), **spec.get("env", {})}
        native_env["AUTOFA_HOME"] = str(Path(
            native_env.get("AUTOFA_HOME") or os.environ.get("AUTOFA_HOME") or "~/.autofa"
        ).expanduser().resolve())
        receipt = {
            "launch_id": launch_id,
            "campaign_id": campaign["id"],
            "campaign_path": str(root),
            "revision": campaign["revision"],
            "owner_generation": generation,
            "state": "prepared",
            "phase": "preparing",
            "spec": spec,
            "native": {**config["native"], "env": native_env},
            "created_at_epoch": time.time(),
            "run_ids": [],
            "log_path": str(directory / "native.log"),
        }
        path = directory / "receipt.json"
        records.atomic(path, receipt)
        try:
            with (directory / "worker.log").open("ab") as output:
                worker = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name("managed_launch.py")), str(path)],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=output,
                    start_new_session=os.name != "nt",
                )
            # The child takes this same lock before updating the receipt.
            receipt.update(
                state="starting", worker=process_identity(worker.pid), updated_at_epoch=time.time()
            )
            records.atomic(path, receipt)
        except BaseException as exc:
            receipt.update(
                state="launch_failed",
                failure={"error_type": type(exc).__name__},
                updated_at_epoch=time.time(),
            )
            records.atomic(path, receipt)
            raise
        return receipt


def launch_observations(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconcile only structured bindings and worker receipts, never guess a run directory."""
    events, runs = [], []
    launches = [(path, document(path)) for path in
                (root / "supervision" / "launches").glob("*/receipt.json")]
    for path, receipt in sorted(launches, key=lambda row: row[1].get("created_at_epoch", 0)):
        preparation = document(path.parent / "preparation.json", {})
        if preparation.get("launch_id") == receipt["launch_id"]:
            events.append({"kind": "preparation", "launch_id": receipt["launch_id"], **preparation})
        for binding_path in sorted((path.parent / "bindings").glob("*.json")):
            binding = document(binding_path)
            if (
                binding.get("campaign_id") != receipt["campaign_id"]
                or binding.get("launch_id") != receipt["launch_id"]
            ):
                raise ValueError("Native binding does not match managed launch")
            runs.append(
                {
                    "run_id": binding["run_id"],
                    "label": receipt["spec"].get("label", ""),
                    "launch_id": receipt["launch_id"],
                    "native": receipt["native"],
                }
            )
        if (
            receipt["state"] in {"starting", "running"}
            and process_alive(receipt.get("worker", {})) is False
        ):
            with records.state_lock(root):
                current = document(path)
                if current["state"] in {"starting", "running"}:
                    current.update(
                        state="interrupted",
                        failure={"error_type": "WorkerExited"},
                        updated_at_epoch=time.time(),
                    )
                    records.atomic(path, current)
                    receipt = current
        events.append(
            {
                "kind": "launch",
                "launch_id": receipt["launch_id"],
                "state": receipt["state"],
                "exit_code": receipt.get("exit_code"),
                "failure": receipt.get("failure"),
            }
        )
    return events, runs
