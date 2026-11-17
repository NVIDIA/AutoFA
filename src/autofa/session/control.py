# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable supervisor requests, applied by the owning runtime at safe boundaries."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from autofa.session.recovery import read_json
from autofa.session.supervision import (
    _now,
    _receipt_id,
    lifecycle_action,
    supervision_lock,
)
from autofa.util.atomic import atomic_write_bytes, atomic_write_json
from autofa.util.tool_result import result_failed, result_payload

_RECORD_LOCK = threading.Lock()
_RECORDS: dict[Path, tuple[tuple, dict]] = {}

CONTRIBUTION_TOOLS = frozenset({
    "submit_function", "submit_helper", "library_import_candidates",
    "record_external_inspection", "record_source_audit", "plan_function_repair",
    "defer_function_repair", "restore_function_version", "classifier_profiles",
})


class FenceBusy(RuntimeError):
    """A short campaign edit is active; retain queued work for the next boundary."""


def verify_fence(fence: dict | None, *, protective: bool = False) -> None:
    """Fence delayed work from superseded driver owners and goal revisions."""
    if fence is None:
        return
    required = {"campaign_path", "revision", "owner_generation"}
    if (not isinstance(fence, dict) or not required.issubset(fence)
            or set(fence) - required - {"monitoring_revision", "scope"}):
        raise ValueError("fence requires campaign_path, revision and owner_generation")
    if fence.get("scope") not in {None, "guard"}:
        raise ValueError("Unknown campaign fence scope")
    root = Path(fence["campaign_path"])
    if not root.is_absolute():
        raise ValueError("fence campaign_path must be absolute")
    campaign = read_json(root / "campaign.json")
    owner = read_json(root / "supervision" / "owner.json")
    if (campaign.get("revision") != fence["revision"]
            or owner.get("generation") != fence["owner_generation"]
            or owner.get("state") != "running"
            or campaign.get("state") not in ({"active", "stopped"} if protective else {"active"})):
        raise ValueError("Superseded or stopped campaign owner/revision; request not applied")
    if fence.get("scope") != "guard":
        monitoring = read_json(root / "supervision" / "monitoring.json")
        config = read_json(root / "supervision" / "config.json")
        if (not monitoring.get("enabled", config.get("monitoring_enabled", True))
                or fence.get("monitoring_revision", 0) != monitoring.get("revision", 0)):
            raise ValueError("Monitoring is paused or its session changed; request not applied")


@contextmanager
def guarded_fence(fence: dict | None, *, protective: bool = False, timeout: float = 2.0):
    """Serialize a short native commit with the driver's owner/revision updates.

    Use the portable driver's existing lock, never hold it during inference or
    a subprocess wait. Contention rejects/defer the request rather than risking
    a run-lock/campaign-lock deadlock.
    """
    verify_fence(fence, protective=protective)
    if fence is None:
        yield
        return
    with (Path(fence["campaign_path"]) / ".driver.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise FenceBusy("Campaign state is busy; retry the fenced request") from exc
                time.sleep(.02)
        try:
            verify_fence(fence, protective=protective)
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _paths(directory: Path) -> list[Path]:
    return sorted((directory / "supervision" / "controls").glob("*.json"))


def _record(path: Path) -> dict:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    with _RECORD_LOCK:
        previous = _RECORDS.get(path)
        if previous is None or previous[0] != signature:
            previous = (signature, read_json(path))
            if len(_RECORDS) >= 2048:
                _RECORDS.pop(next(iter(_RECORDS)))
            _RECORDS[path] = previous
        return deepcopy(previous[1])


def _records(directory: Path) -> list[tuple[Path, dict]]:
    """Read each atomic receipt once per version; never reserve work for an empty queue."""
    records = [(path, _record(path)) for path in _paths(directory)]
    return sorted(records, key=lambda pair: pair[1].get("created_at", ""))


def control_status(directory: Path) -> dict:
    """Small read-only receipts; source and imported library bytes stay out of status."""
    config = read_json(directory / "config.json")
    records = [record for _, record in _records(directory)]
    keys = ("id", "kind", "state", "expected_revision", "effective_revision", "created_at",
            "updated_at", "reason", "tool", "transition")
    return {"revision": config.get("control_revision", 0),
            "pending": sum(row.get("state") in {"queued", "deferred", "applying"}
                           for row in records),
            "receipts": [{key: row[key] for key in keys if key in row}
                         for row in records[-16:]]}


def _enqueue(directory: Path, payload: dict, *, request_id: str | None,
             expected_revision: int | None, fence: dict | None) -> dict:
    request_id = _receipt_id(request_id)
    if expected_revision is not None and (
        isinstance(expected_revision, bool) or not isinstance(expected_revision, int)
        or expected_revision < 0
    ):
        raise ValueError("expected_revision must be a nonnegative integer")
    verify_fence(fence)
    # Validate JSON and bound source submission before reserving a request.
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > 2 * 1024 * 1024:
        raise ValueError("Control request exceeds 2 MiB; use a native candidate bundle")
    identity = hashlib.sha256(json.dumps(
        {"payload": payload, "expected_revision": expected_revision, "fence": fence},
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    path = directory / "supervision" / "controls" / f"{request_id}.json"
    with supervision_lock(directory), guarded_fence(fence):
        if path.exists():
            prior = read_json(path)
            if prior.get("identity") != identity:
                raise ValueError("Request ID already identifies a different control request")
            return prior
        config = read_json(directory / "config.json")
        revision = config.get("control_revision", 0)
        dependency = payload.get("after_update")
        if dependency:
            parent = read_json(path.parent / f"{_receipt_id(dependency)}.json")
            if parent.get("kind") != "update" or parent.get("state") not in {
                "queued", "deferred", "applying", "applied",
            }:
                raise ValueError("after_update must name an accepted update receipt")
            if parent.get("fence") != fence:
                raise ValueError("Dependent contribution must use the update's campaign fence")
            effective = parent.get("effective_revision", parent["expected_revision"] + 1)
            if expected_revision is not None and expected_revision not in {
                parent["expected_revision"], effective,
            }:
                raise ValueError("Contribution revision conflicts with its update dependency")
            expected_revision = effective
        elif expected_revision is not None and revision != expected_revision:
            raise ValueError(
                f"Stale control revision: expected {expected_revision}, current {revision}")
        if lifecycle_action(directory) in {"pause", "stop"}:
            raise ValueError("Saved supervisor stop/pause blocks changes; release explicitly")
        record = {**payload, "id": request_id, "identity": identity, "fence": fence,
                  "expected_revision": expected_revision if dependency else revision,
                  "created_at": _now(), "state": "queued"}
        if payload["kind"] == "update":
            settings = dict(payload["settings"])
            additional = settings.pop("additional_rounds", None)
            if additional is not None:
                if (isinstance(additional, bool) or not isinstance(additional, int)
                        or additional <= 0 or "max_rounds" in settings):
                    raise ValueError("Use positive additional_rounds or max_rounds, not both")
                settings["max_rounds"] = max(config.get("max_rounds", 0),
                                             config.get("round", 0)) + additional
            record["settings"] = settings
            from autofa.session.campaign_policy import normalize_policy_update

            try:
                normalize_policy_update(config, settings)
            except (TypeError, OverflowError) as exc:
                raise ValueError(f"Invalid live goal settings: {exc}") from exc
        atomic_write_json(path, record)
    return record


def submit_update(directory: Path, settings: dict, *, request_id: str | None = None,
                  expected_revision: int | None = None, fence: dict | None = None) -> dict:
    if not isinstance(settings, dict) or not settings or set(settings) - {
        "max_rounds", "additional_rounds", "min_failure_modes", "campaign_policy",
    }:
        raise ValueError("Use max_rounds/additional_rounds, min_failure_modes or campaign_policy")
    return _enqueue(directory, {"kind": "update", "settings": settings},
                    request_id=request_id, expected_revision=expected_revision, fence=fence)


def submit_contribution(directory: Path, tool: str, arguments: dict, *,
                        request_id: str | None = None, expected_revision: int | None = None,
                        library_sha256: str | None = None, fence: dict | None = None,
                        after_update: str | None = None) -> dict:
    from autofa.mcp.server import TOOL_CALLABLES

    if tool not in CONTRIBUTION_TOOLS or tool not in TOOL_CALLABLES:
        raise ValueError("Tool is not supported for queued contributions; inspect native tools")
    if not isinstance(arguments, dict):
        raise ValueError("Contribution arguments must be an object")
    if after_update is not None and not isinstance(after_update, str):
        raise ValueError("after_update must name an accepted update receipt")
    if library_sha256 is not None and (
        not isinstance(library_sha256, str) or len(library_sha256) != 64
        or any(char not in "0123456789abcdef" for char in library_sha256)
    ):
        raise ValueError("library_sha256 must be a lowercase SHA-256 digest")
    try:
        inspect.signature(TOOL_CALLABLES[tool]).bind(**arguments)
    except TypeError as exc:
        raise ValueError(f"Invalid arguments for {tool}: {exc}") from exc
    artifacts = {}
    if tool == "library_import_candidates":
        source = Path(arguments["path"]).expanduser().resolve()
        artifacts[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        arguments = {**arguments, "path": str(source)}
    return _enqueue(directory, {"kind": "contribution", "tool": tool, "arguments": arguments,
                               "library_sha256": library_sha256, "artifacts": artifacts,
                               **({"after_update": after_update} if after_update else {})},
                    request_id=request_id, expected_revision=expected_revision, fence=fence)


def _check_request(rt: Any, record: dict) -> None:
    verify_fence(record.get("fence"))
    if lifecycle_action(rt.run.run_dir()) in {"pause", "stop"}:
        raise ValueError("Saved supervisor stop/pause blocks this request")
    if dependency := record.get("after_update"):
        parent = read_json(rt.run.run_dir() / "supervision" / "controls" /
                           f"{_receipt_id(dependency)}.json")
        if parent.get("state") in {"queued", "deferred", "applying"}:
            raise FenceBusy("Dependent goal update has not been applied yet")
        if (parent.get("state") != "applied"
                or parent.get("effective_revision") != record["expected_revision"]):
            raise ValueError("Dependent goal update was not applied at the expected revision")
    if getattr(rt.run, "control_revision", 0) != record["expected_revision"]:
        raise ValueError("Control revision changed before application; inspect and resubmit")
    if expected := record.get("library_sha256"):
        actual = hashlib.sha256(rt.working_library_path().read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Library changed before contribution; inspect and resubmit")
    for artifact, expected in record.get("artifacts", {}).items():
        if hashlib.sha256(Path(artifact).read_bytes()).hexdigest() != expected:
            raise ValueError("Contribution artifact changed after submission")


def _finish(path: Path, record: dict, state: str, **updates: Any) -> dict:
    record.update(state=state, updated_at=_now(), **updates)
    atomic_write_json(path, record)
    return record


_UPDATE_FILES = ("config.json", "round_validation.json", "discovery_round_stagnation.json")


def _transaction_path(directory: Path, request_id: str) -> Path:
    return directory / "supervision" / "control-transactions" / _receipt_id(request_id)


def _restore_update(directory: Path, transaction: Path) -> None:
    manifest = read_json(transaction / "manifest.json")
    for name in _UPDATE_FILES:
        if name not in manifest:
            continue
        if manifest[name]:
            atomic_write_bytes(directory / name, (transaction / name).read_bytes())
        else:
            (directory / name).unlink(missing_ok=True)


def _restore_is_current(directory: Path, transaction: Path) -> bool:
    """Round/resume writes can change state without changing the control revision."""
    before = transaction / "config.json"
    current = directory / "config.json"
    return before.is_file() and current.is_file() and before.read_bytes() == current.read_bytes()


@contextmanager
def _update_transaction(rt: Any, request_id: str):
    """Config+revision is the commit point; recover earlier phase writes after a crash."""
    directory = rt.run.run_dir()
    transaction = _transaction_path(directory, request_id)
    if (transaction / "manifest.json").exists():
        if not _restore_is_current(directory, transaction):
            raise ValueError("Rollback journal conflicts with newer saved run state")
        _restore_update(directory, transaction)
    before_run, before_config = deepcopy(vars(rt.run)), deepcopy(vars(rt.config))
    manifest = {}
    for name in _UPDATE_FILES:
        source = directory / name
        manifest[name] = source.exists()
        if source.exists():
            atomic_write_bytes(transaction / name, source.read_bytes())
    atomic_write_json(transaction / "manifest.json", manifest)
    try:
        yield
    except BaseException:
        if read_json(directory / "config.json").get("control_last_request") != request_id:
            vars(rt.run).clear()
            vars(rt.run).update(before_run)
            vars(rt.config).clear()
            vars(rt.config).update(before_config)
            _restore_update(directory, transaction)
        raise


def apply_pending_updates(rt: Any, *, records: list[tuple[Path, dict]] | None = None) -> list[dict]:
    """Apply only at an operation boundary; busy operations retain their fixed settings."""
    from autofa.session.campaign_policy import apply_policy_update
    from autofa.session.operation_admission import admit_operation
    from autofa.util.cancellation import cancellation_guard

    directory = rt.run.run_dir()
    pending = [(path, row) for path, row in (_records(directory) if records is None else records)
               if row.get("kind") == "update" and row.get("state") in {
                   "queued", "deferred", "applying"}]
    if not pending:
        return []
    completed = []
    with admit_operation("driver_control", {}) as busy:
        if busy:
            return []
        with cancellation_guard(), supervision_lock(directory):
            for path, record in pending:
                record = _record(path)
                if record.get("kind") != "update" or record.get("state") not in {
                    "queued", "deferred", "applying",
                }:
                    continue
                if getattr(rt.run, "control_last_request", "") == record["id"]:
                    completed.append(_finish(path, record, "applied",
                                             effective_revision=rt.run.control_revision,
                                             reason="Reconciled committed update"))
                    shutil.rmtree(_transaction_path(directory, record["id"]), ignore_errors=True)
                    continue
                transaction = _transaction_path(directory, record["id"])
                if (transaction / "manifest.json").exists():
                    # Recover partial phase writes even if this driver's fence
                    # was superseded during the crash. Authorization to replay
                    # the request is separate from restoring its uncommitted state.
                    current = read_json(directory / "config.json")
                    if (current.get("control_revision", 0) != record["expected_revision"]
                            or not _restore_is_current(directory, transaction)):
                        completed.append(_finish(path, record, "uncertain", reason=
                            "Rollback journal conflicts with newer saved run state; "
                            "reconcile saved state before retrying"))
                        continue
                    _restore_update(directory, transaction)
                try:
                    with guarded_fence(record.get("fence")):
                        _check_request(rt, record)
                        with _update_transaction(rt, record["id"]):
                            result = apply_policy_update(rt, record["settings"], persist=False)
                            if result.get("ok") is False:
                                completed.append(_finish(
                                    path, record, "deferred", result=result,
                                    reason=result.get("reason", "Native deferral")))
                                continue
                            rt.run.control_revision = record["expected_revision"] + 1
                            rt.run.control_last_request = record["id"]
                            rt.run.save()
                except FenceBusy as exc:
                    completed.append(_finish(path, record, "deferred", reason=str(exc)))
                    continue
                except (ValueError, TypeError, OverflowError) as exc:
                    completed.append(_finish(path, record, "rejected", reason=str(exc)))
                else:
                    completed.append(_finish(path, record, "applied", result=result,
                                             transition=result.get("transition"),
                                             effective_revision=rt.run.control_revision))
                shutil.rmtree(_transaction_path(directory, record["id"]), ignore_errors=True)
    return completed


async def consume_control_requests(rt: Any) -> list[dict]:
    """Use normal native tool admission/scoring with explicit driver attribution."""
    import asyncio

    from autofa.mcp.server import invoke_tool
    from autofa.session.operation_admission import admit_operation
    directory = rt.run.run_dir()
    records = _records(directory)
    completed = apply_pending_updates(rt, records=records)
    pending = [(path, row) for path, row in records if row.get("kind") == "contribution"
               and row.get("state") in {"queued", "deferred", "applying"}]
    if not pending:
        return completed
    with admit_operation("driver_control", {}) as busy:
        if busy:
            return completed
        for path, record in pending:
            with supervision_lock(directory):
                record = _record(path)
                if record.get("kind") != "contribution":
                    continue
                if record.get("state") == "applying":
                    completed.append(_finish(
                        path, record, "uncertain",
                        reason="Interrupted contribution; reconcile the native operation first"))
                    continue
                if record.get("state") not in {"queued", "deferred"}:
                    continue
                try:
                    with guarded_fence(record.get("fence")):
                        _check_request(rt, record)
                        _finish(path, record, "applying")
                except FenceBusy as exc:
                    completed.append(_finish(path, record, "deferred", reason=str(exc)))
                    continue
                except (ValueError, OSError) as exc:
                    completed.append(_finish(path, record, "rejected", reason=str(exc)))
                    continue
            try:
                result = await invoke_tool(record["tool"], record["arguments"], caller="driver")
            except BaseException as exc:
                with supervision_lock(directory):
                    _finish(path, record, "uncertain", reason=type(exc).__name__)
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                raise
            with supervision_lock(directory):
                payload = result_payload(result)
                uncertain = payload.get("status") in {
                    "stopped_by_supervisor", "campaign_boundary", "interrupted",
                } or payload.get("resumable") is True or (
                    "tool crashed" in str(payload.get("error", "")).lower())
                state = ("uncertain" if uncertain else "deferred" if payload.get("busy")
                         or payload.get("retryable") else "rejected" if result_failed(result)
                         else "applied")
                completed.append(_finish(path, record,
                    state, result=result,
                    effective_revision=getattr(rt.run, "control_revision", 0)))
    return completed
