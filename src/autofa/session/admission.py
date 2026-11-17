# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery admission with durable, route-specific in-flight cost reservations."""

from __future__ import annotations

import logging
import math
import os
import re
import socket
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from autofa.session.recovery import read_json
from autofa.util.atomic import atomic_write_json

_LOCK = threading.RLock()
_LOCKS: dict[Path, threading.RLock] = {}
_OWNER = uuid4().hex
_OWNER_START: int | None = None
_INFLIGHT: dict[Path, dict[str, float | None]] = {}
_DELIVERY_PHASES = {"finalization", "publication", "highlighting"}


def _lock(directory: Path) -> threading.RLock:
    with _LOCK:
        return _LOCKS.setdefault(directory, threading.RLock())


def _records(directory: Path) -> list[dict]:
    records = [read_json(path) for path in
               (directory / "supervision" / "reservations").glob("*.json")]
    return [record for record in records if record.get("state") == "inflight"]


def _owner_alive(record: dict) -> bool:
    if record.get("pid") == os.getpid() and record.get("owner") == _OWNER:
        return True
    if os.name != "posix":
        return False  # Unverified ownership retains exposure; signal 0 is unsafe on Windows.
    if record.get("hostname") != socket.gethostname():
        return False  # Unverified remote ownership cannot erase a reservation.
    pid = record.get("pid")
    if type(pid) is not int or pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if record.get("process_started_at") is not None:
        from autofa.session.supervision import _process_started

        if _process_started(pid) != record["process_started_at"]:
            return False
    return True


def reservation_summary(directory: Path) -> dict:
    """Read-only outstanding exposure; interrupted owners never make it disappear."""
    active = orphaned = unknown = 0
    active_amount = orphaned_amount = 0.0
    pending = []
    owners = {}
    for record in _records(directory):
        amount = record.get("estimated_usd")
        identity = tuple(record.get(key) for key in (
            "hostname", "pid", "owner", "process_started_at"))
        if identity not in owners:
            owners[identity] = _owner_alive(record)
        if owners[identity]:
            active += 1
            active_amount += amount or 0
        else:
            orphaned += 1
            if amount is None:
                unknown += 1
            else:
                orphaned_amount += amount
            pending.append({key: record.get(key) for key in
                            ("id", "estimated_usd", "dispatch_identity", "observed_at")})
    return {"inflight_dispatches": active, "estimated_reserved_usd": active_amount,
            "unresolved_dispatches": orphaned,
            "unresolved_estimated_dispatches": orphaned - unknown,
            "unresolved_unpriced_dispatches": unknown,
            "unresolved_reserved_usd": orphaned_amount, "pending": pending[:64]}


def _save_snapshot(directory: Path) -> None:
    try:
        atomic_write_json(directory / "supervision/admission.json", {
            **reservation_summary(directory), "durable_reservations": True,
            "enforcement": "estimated_guard", "pid": os.getpid(), "hostname": socket.gethostname(),
            "observed_at": datetime.now(UTC).isoformat(),
            "note": "Interrupted reservations remain separate from reported charges. "
                    "Reconcile actual provider costs, or use an explicitly approved "
                    "estimate policy.",
        })
    except OSError:
        # Per-dispatch records are authoritative and already durable. A failed
        # summary refresh cannot leak a reservation or discard completed work.
        logging.getLogger(__name__).warning("Admission summary refresh failed")


@contextmanager
def paid_dispatch(context: dict | None):
    """Check the actual call route and save its reservation before dispatching.

    A result already dispatched is preserved. A kill leaves a durable reservation;
    another owner cannot recover the spent allowance by starting a new process.
    """
    from autofa.session.campaign_policy import CampaignBoundaryReached, directory_status
    from autofa.session.usage import admission_usage
    from autofa.session.usage_estimates import dispatch_cost_estimate

    directory = (Path(context["run_dir"]).resolve()
                 if context and context.get("run_dir") else None)
    if directory is None or (context or {}).get("phase") in {"taxonomy", "efficiency"}:
        yield
        return
    from autofa.session.managed_launch import enforce_supervision
    from autofa.session.supervision import SupervisorInterrupted

    try:
        enforce_supervision(directory, phase=context.get("phase", "discovery"))
    except SupervisorInterrupted as exc:
        # This may be a tool worker, where KeyboardInterrupt can tear down the
        # whole event loop. Use the existing normal campaign-boundary result path.
        raise CampaignBoundaryReached(
            {"stop_reason": "user_" + exc.action, "continue_discovery": False,
             "finalize_due": False, "supervision_lost": exc.action == "pause",
             "intent": exc.action},
            phase=context.get("phase", "discovery"),
        ) from None
    previous = {key: context.get(key) for key in ("admission_id", "dispatch_started_at")}
    context["dispatch_started_at"] = datetime.now(UTC).isoformat()
    phase = context.get("phase", "discovery")
    reservation_id = uuid4().hex
    with _lock(directory):
        status = directory_status(directory, usage={"estimated_dispatch_cost_usd": None})
        inactive = not status.get("active")
        policy = status.get("policy", {})
        estimate = dispatch_cost_estimate(directory, context) if not inactive else None
        if not inactive:
            usage = admission_usage(directory)
            status = directory_status(directory, usage={**usage,
                                      "estimated_dispatch_cost_usd": estimate})
        outstanding = reservation_summary(directory)["estimated_reserved_usd"]
        remaining = status.get("remaining_cost_usd")
        reserve = 0 if phase in _DELIVERY_PHASES else policy.get("delivery_reserve_usd", 0)
        exceeds = (remaining is not None
                   and outstanding + (estimate or 0) > remaining - reserve + 1e-9)
        blocked = not inactive and (
            status.get("resources_exhausted")
            or status.get("cost_admission_allowed") is False
            or status.get("stop_reason") in {"cost_unavailable", "user_stop", "user_pause",
                                              "budget_transferred"}
            or phase not in _DELIVERY_PHASES and not status.get("continue_discovery")
        )
        if blocked or exceeds:
            if exceeds and not blocked:
                status = {**status, "continue_discovery": False, "finalize_due": True,
                          "stop_reason": "inflight_spend_reserve"}
            raise CampaignBoundaryReached(status, phase=phase)
        if not inactive:
            global _OWNER_START
            if _OWNER_START is None:
                from autofa.session.supervision import _process_started

                _OWNER_START = _process_started(os.getpid())
            context["admission_id"] = reservation_id
            record = {"id": reservation_id, "state": "inflight", "owner": _OWNER,
                      "pid": os.getpid(), "hostname": socket.gethostname(),
                      "process_started_at": _OWNER_START,
                      "estimated_usd": estimate, "phase": phase,
                      "dispatch_identity": context.get("dispatch_identity"),
                      "role": context.get("role"), "round_index": context.get("round_index"),
                      "observed_at": context["dispatch_started_at"]}
            atomic_write_json(directory / "supervision/reservations" / (reservation_id + ".json"),
                              record)
            _INFLIGHT.setdefault(directory, {})[reservation_id] = estimate
            _save_snapshot(directory)
    try:
        yield
    finally:
        if not inactive:
            with _lock(directory):
                (directory / "supervision/reservations" / (reservation_id + ".json")).unlink(
                    missing_ok=True)
                _INFLIGHT[directory].pop(reservation_id, None)
                if not _INFLIGHT[directory]:
                    del _INFLIGHT[directory]
                _save_snapshot(directory)
        for key, value in previous.items():
            if value is None:
                context.pop(key, None)
            else:
                context[key] = value


def admission_snapshot(directory: Path) -> dict[str, Any]:
    return {**read_json(directory / "supervision/admission.json"),
            **reservation_summary(directory)}


def reconcile_reservation(run_dir: Path, reservation_id: str, cost_usd: float,
                          source: str) -> dict:
    """Record an explicitly supplied total charge for one interrupted dispatch.

    Source identifies the provider receipt or billing evidence. Previously
    measured charges remain unchanged; only the missing monetary delta is added.
    Token gaps remain token gaps. The caller must not invent an amount.
    """
    from autofa.session.recovery import run_lock
    from autofa.session.usage import _connect, record_usage

    if (not isinstance(reservation_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", reservation_id)):
        raise ValueError("Reservation ID must be a saved filename-safe ID")
    if (isinstance(cost_usd, bool) or not isinstance(cost_usd, (int, float))
            or not math.isfinite(cost_usd) or cost_usd < 0):
        raise ValueError("cost_usd must be a finite nonnegative total for this dispatch")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("A provider receipt or billing source is required")
    directory = run_dir.resolve()
    path = directory / "supervision/reservations" / (reservation_id + ".json")
    with run_lock(directory), _lock(directory):
        record = read_json(path)
        if not record:
            raise ValueError("Saved reservation not found")
        if record.get("state") == "reconciled":
            if record.get("cost_usd") != cost_usd or record.get("source") != source:
                raise ValueError("Reservation already has a different reconciliation")
            return {"ok": True, "reservation": record, "idempotent": True}
        if _owner_alive(record):
            raise ValueError("The reservation owner is still active; wait for its result")
        event_id = "reservation-reconciliation:" + reservation_id
        db = _connect(directory)
        try:
            known = db.execute("SELECT COALESCE(SUM(cost_usd),0) FROM events "
                               "WHERE json_extract(payload,'$.admission_id')=? AND id!=?",
                               (reservation_id, event_id)).fetchone()[0]
            if cost_usd + 1e-9 < known:
                raise ValueError("Reconciled total cannot erase already measured charges")
        finally:
            db.close()
        identity = record.get("dispatch_identity") or {}
        context = {"run_dir": str(directory), "phase": record.get("phase", "discovery"),
                   "role": record.get("role") or "discovery",
                   "round_index": record.get("round_index"), "admission_id": reservation_id,
                   "dispatch_identity": identity}
        record_usage(context, event_id=event_id, provider=identity.get("provider", "reconciled"),
                     model=identity.get("model"), measurement="cost",
                     cost_usd=max(0, cost_usd - known), cost_kind="reported",
                     source_id=source, note="Explicit interrupted-dispatch billing reconciliation")
        db = _connect(directory)
        try:
            with db:
                if not db.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone():
                    raise RuntimeError("Reconciliation charge could not be saved")
                db.execute("CREATE TABLE IF NOT EXISTS cost_reconciliations "
                           "(id TEXT PRIMARY KEY, cost_usd REAL NOT NULL, source TEXT NOT NULL)")
                db.execute("INSERT OR REPLACE INTO cost_reconciliations VALUES (?,?,?)",
                           (reservation_id, cost_usd, source))
                db.execute("UPDATE admission_totals SET revision=revision+1 WHERE id=1")
        finally:
            db.close()
        record.update(state="reconciled", cost_usd=cost_usd, source=source,
                      reconciled_at=datetime.now(UTC).isoformat())
        atomic_write_json(path, record)
        _save_snapshot(directory)
        return {"ok": True, "reservation": record}


@asynccontextmanager
async def paid_async_dispatch(context: dict | None):
    with paid_dispatch(context):
        yield
