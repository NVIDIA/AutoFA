# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native supervision receipts, separate from frozen analysis and judge context."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import socket
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from autofa.session.recovery import read_json, run_lock
from autofa.util.atomic import atomic_write_json

_LIBRARY_DIGEST_LOCK = threading.Lock()
_LIBRARY_DIGESTS: dict[Path, tuple[tuple, str]] = {}


class SupervisionBusy(RuntimeError):
    """A transient receipt lock conflict; the same request can safely be retried."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_run_dir(run_id: str) -> Path:
    """Resolve an existing run without creating a profile or initializing a runtime."""
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run must be a single saved run ID")
    root = Path(os.environ.get("AUTOFA_HOME", "~/.autofa")).expanduser() / "runs"
    root = root.resolve()
    directory = (root / run_id).resolve()
    # Native resume permits runs/<id> to point at another disk. The lexical ID
    # check above prevents traversal without rejecting a deliberate run symlink.
    if not (directory / "config.json").is_file():
        raise ValueError(f"Saved run not found: {run_id}")
    return directory



@contextmanager
def supervision_lock(run_dir: Path, *, timeout: float = 2.0):
    """Serialize short receipt updates without racing the discovery delivery hook."""
    deadline = time.monotonic() + timeout
    with ExitStack() as stack:
        while True:
            try:
                stack.enter_context(run_lock(run_dir / "supervision"))
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    message = "Supervision receipt update is busy; retry shortly"
                    raise SupervisionBusy(message) from None
                time.sleep(.02)
        yield


class SupervisorInterrupted(KeyboardInterrupt):
    """Durable controller intent, distinct from a terminal interrupt."""

    def __init__(self, run_dir: Path, action: str):
        self.run_dir, self.action = Path(run_dir), action
        self.release_command = f"autofa supervise request --run {self.run_dir.name} release"
        super().__init__(f"Supervisor requested {action}; saved intent blocks new work. "
                         f"To permit work again: {self.release_command}")

    def as_result(self) -> dict:
        return {"ok": False, "status": "stopped_by_supervisor", "intent": self.action,
                "run_id": self.run_dir.name, "error": str(self),
                "release_command": self.release_command}


def stop_requested(run_dir: Path) -> bool:
    """Pause/stop cancel in-flight work; finalize stops only at discovery boundaries."""
    return lifecycle_action(run_dir) in {"pause", "stop"}


def stop_action(run_dir: Path) -> str | None:
    """Return the observed cancellable intent for a tool-local cancellation token."""
    action = lifecycle_action(run_dir)
    return action if action in {"pause", "stop"} else None


def supervisor_cancelled(cancellation: Any) -> bool:
    """Never turn actual asyncio task cancellation into a successful protocol reply."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return bool(cancellation.supervisor_requested and not (task and task.cancelling()))


def supervisor_interruption(run_dir: Path, cancellation: Any) -> SupervisorInterrupted:
    """Describe the intent that cancelled work, not a later controller update."""
    return SupervisorInterrupted(run_dir, cancellation.supervisor_action or "stop")


@contextmanager
def lifecycle_cancellation(run_dir: Path):
    from autofa.util.cancellation import Cancellation, cancellation_scope, current_cancellation

    cancellation = current_cancellation() or Cancellation()
    previous = cancellation.stop_requested
    cancellation.stop_requested = lambda: (previous and previous()) or stop_action(run_dir)
    try:
        with cancellation_scope(cancellation):
            yield cancellation
    finally:
        cancellation.stop_requested = previous


def _small(value: Any, *, depth: int = 0) -> Any:
    """Bound status projections independently of library/report size."""
    if isinstance(value, dict):
        if depth >= 2:
            return {"key_count": len(value)}
        return {str(key): _small(item, depth=depth + 1)
                for key, item in list(value.items())[:24]}
    if isinstance(value, list):
        return {"count": len(value)}
    return value[:1024] + "…" if isinstance(value, str) and len(value) > 1024 else value


def _snapshot(path: Path, *, full: bool = False) -> dict:
    try:
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        value = read_json(path)
    except OSError:
        return {"path": str(path), "size_bytes": None, "mtime_ns": None,
                "sha256": None, "record": {}}
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest, "key_count": len(value),
            "record": value if full else _small(value)}


def _process_started(pid: int) -> int | None:
    """Return an epoch birth marker independent of the caller's timezone and locale."""
    try:
        if os.name == "posix":
            result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                                    capture_output=True, text=True, timeout=1, check=False,
                                    env={**os.environ, "TZ": "UTC", "LC_ALL": "C"})
            if result.stdout.strip():
                _, month, day, clock, year = result.stdout.split()
                months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
                month_number = months.index(month) + 1
                hour, minute, second = map(int, clock.split(":"))
                return int(datetime(int(year), month_number, int(day), hour, minute, second,
                                    tzinfo=UTC).timestamp())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def _observed_operation(record: dict) -> dict:
    """Project dead owners as stale without writes during read-only status."""
    if record.get("state") != "running":
        return record
    if record.get("hostname") != socket.gethostname():
        return {**record, "state": "unknown", "liveness": "remote_owner"}
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return {**record, "state": "unknown", "liveness": "owner_not_recorded"}
    alive = True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        alive = False
    except PermissionError:
        pass
    expected = record.get("process_started_at")
    actual = _process_started(pid) if alive and expected is not None else None
    if not alive or (expected and actual and expected != actual):
        return {**record, "state": "interrupted", "liveness": "owner_exited"}
    return {**record, "liveness": "owner_present"}


def prune_dead_operations(run_dir: Path) -> None:
    """Prune dead local owners after taking the native run lock; status stays read-only."""
    directory = run_dir / "supervision"
    for path in (directory / "active").glob("*.json"):
        record = _observed_operation(read_json(path))
        if record.get("liveness") == "owner_exited" or record.get("state") in {
            "completed", "failed", "interrupted",
        }:
            path.unlink(missing_ok=True)
    latest = directory / "operation.json"
    record = read_json(latest)
    observed = _observed_operation(record)
    if observed != record and observed.get("liveness") == "owner_exited":
        atomic_write_json(latest, observed)

def _status_projection(value: Any, *, diagnostic: bool = False) -> Any:
    """Sanitize legacy provider errors in observations without changing saved evidence."""
    from autofa.util.model_errors import safe_error_detail

    if isinstance(value, dict):
        return {key: _status_projection(item, diagnostic=diagnostic or key in {
            "error", "finalization_error", "error_details", "finalization_error_details",
        }) for key, item in value.items()}
    if isinstance(value, list):
        return [_status_projection(item, diagnostic=diagnostic) for item in value]
    if diagnostic and isinstance(value, str):
        return safe_error_detail(value)
    return value


def run_status(run_dir: Path, *, full_snapshots: bool = False) -> dict[str, Any]:
    """Read saved projections only; missing or stale measurements remain explicit."""
    from autofa.session.admission import admission_snapshot
    from autofa.session.campaign_policy import directory_status
    from autofa.session.control import control_status
    from autofa.session.detector_service import detector_service_status
    from autofa.session.status_log import latest_turn_status, read_status
    from autofa.session.usage import snapshot_usage

    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "validation_summary.json")
    turn = read_json(run_dir / "turn.json")
    research = read_json(run_dir / "research_status.json")
    control = read_json(run_dir / "supervision" / "lifecycle.json")
    operation = _observed_operation(read_json(run_dir / "supervision" / "operation.json"))
    active = [_observed_operation(read_json(path)) for path in
              sorted((run_dir / "supervision" / "active").glob("*.json"))]
    stale = [row for row in active if row.get("state") != "running"]
    active = [row for row in active if row.get("state") == "running"]
    snapshots = {}
    for name in ("validation_summary", "research_status", "round_validation", "publication",
                 "finalization", "finalization_outcome", "scoring_capacity", "coverage_summary",
                 "scoring_settings"):
        path = run_dir / f"{name}.json"
        snapshots[name] = _snapshot(path, full=full_snapshots)
    logs = read_status(run_dir, limit=8)
    usage = snapshot_usage(run_dir)
    reference = usage.get("token_cost_estimate")
    if not full_snapshots and isinstance(reference, dict):
        # Per-role/round groups repeat full price provenance. Keep monitoring
        # bounded; the cost command retains those detailed accounting records.
        usage = {**usage, "token_cost_estimate": {
            **{key: value for key, value in reference.items()
               if key not in {"groups", "unpriced_groups"}},
            "priced_group_count": len(reference.get("groups", [])),
            "unpriced_group_count": len(reference.get("unpriced_groups", [])),
            "details_command": f"autofa supervise cost --run {config.get('run_id', run_dir.name)}",
        }}
    return _status_projection({
        "version": 1, "run_id": config.get("run_id", run_dir.name), "observed_at": _now(),
        "round": config.get("round"), "planned_rounds": config.get("max_rounds"),
        "completed_rounds": config.get("round"),
        "round_semantics": "Stored zero-based next round; increments on completed native "
                           "advance_round, never on recovery attempts.",
        "active_operation": operation,
        "active_operations": active, "stale_operations": stale,
        "phase": operation.get("tool") if operation.get("state") == "running" else
                 turn.get("state", "unknown"),
        "turn": {k: turn.get(k) for k in
                 ("state", "updated_at", "resumable", "recovery_reason")},
        "latest_turn": latest_turn_status(run_dir), "lifecycle": control,
        "grade_counts": summary.get("grade_counts"),
        "library_sha256": saved_library_sha256(run_dir),
        "final_validation": _small(summary.get("final_validation")),
        "research": _small(research), "snapshots": snapshots,
        "recent_progress": [_small(row) for row in logs],
        "usage": usage,
        "detector_service": detector_service_status(run_dir),
        "control": control_status(run_dir),
        "campaign": directory_status(run_dir, usage=usage),
        "admission": admission_snapshot(run_dir),
        "steering": [_small(row) for row in steering_records(run_dir)[-32:]],
        "notice": "Saved observations, not a liveness guarantee. Missing metrics are unknown; "
                  "each snapshot has its own age. Status never scores or scans source data.",
    })


def saved_library_sha256(run_dir: Path) -> str | None:
    """Expose a contribution precondition without parsing or repeatedly hashing a library."""
    path = run_dir / "library.json"
    try:
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
            with _LIBRARY_DIGEST_LOCK:
                previous = _LIBRARY_DIGESTS.get(path)
                if previous and previous[0] == signature:
                    return previous[1]
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None
    with _LIBRARY_DIGEST_LOCK:
        if len(_LIBRARY_DIGESTS) >= 128:
            _LIBRARY_DIGESTS.pop(next(iter(_LIBRARY_DIGESTS)))
        _LIBRARY_DIGESTS[path] = (signature, digest)
    return digest


def observe_operation(run_dir: Path, receipt_id: str, operation: dict) -> None:
    """Best-effort small observations; an observer failure cannot fail committed work."""
    record = {"id": receipt_id, "tool": operation["tool"], "state": operation["state"],
              "started_at": operation["started_at"], "updated_at": _now(),
              "pid": os.getpid(), "process_started_at": _process_started(os.getpid()),
              "hostname": socket.gethostname()}
    try:
        directory = run_dir / "supervision"
        active = directory / "active" / f"{receipt_id}.json"
        if operation["state"] == "running":
            atomic_write_json(active, record)
        else:
            active.unlink(missing_ok=True)
        atomic_write_json(directory / "operation.json", record)
    except OSError:
        logging.getLogger(__name__).warning("Unable to save supervision observation")


def save_quality_snapshot(rt: Any, *, force: bool = False) -> None:
    """Refresh saved quality after native work, using only the existing corpus inventory."""
    from autofa.metrics.library_quality_coverage import library_quality_coverage_for_runtime
    from autofa.session.validation_completion import validated_good_target_for_runtime
    from autofa.session.validation_summary import save_validation_summary

    now = time.monotonic()
    if not force and now - getattr(rt, "_quality_snapshot_at", float("-inf")) < 30:
        return
    rt._quality_snapshot_at = now
    save_validation_summary(rt, completion=validated_good_target_for_runtime(rt))
    atomic_write_json(rt.run.run_dir() / "coverage_summary.json", {
        "updated_at": _now(),
        "good": library_quality_coverage_for_runtime(rt, cached_only=True, minimum_grade="GOOD"),
        "good_or_ok": library_quality_coverage_for_runtime(
            rt, cached_only=True, minimum_grade="OK"),
    })


def _receipt_id(value: str | None) -> str:
    value = value or uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError("Receipt ID must contain 1–100 letters, digits, underscores or hyphens")
    return value


def steering_records(run_dir: Path) -> list[dict]:
    records = [read_json(path) for path in (run_dir / "supervision" / "steering").glob("*.json")]
    return sorted(records, key=lambda row: (row.get("created_at", ""), row.get("id", "")))


def submit_steering(run_dir: Path, message: str, *, receipt_id: str | None = None,
                    fence: dict | None = None) -> dict:
    """Queue an idempotent discovery instruction without changing a judge rubric."""
    if not message.strip() or len(message.encode()) > 32 * 1024:
        raise ValueError("Steering must be nonempty and at most 32 KiB")
    receipt_id = _receipt_id(receipt_id)
    control = run_dir / "supervision"
    with supervision_lock(run_dir):
        from autofa.session.control import guarded_fence

        with guarded_fence(fence):
            path = control / "steering" / f"{receipt_id}.json"
            previous = read_json(path)
            if previous:
                if previous.get("message") != message or previous.get("fence") != fence:
                    raise ValueError("Steering ID already names a different instruction")
                return previous
            receipt = {"id": receipt_id, "audience": "discovery", "message": message,
                       "created_at": _now(), "state": "queued"}
            if fence is not None:
                receipt["fence"] = fence
            path.parent.mkdir(exist_ok=True)
            atomic_write_json(path, receipt)
            return receipt


def discovery_steering(
    run_dir: Path, *, deliver: bool = False, max_chars: int = 8000,
) -> list[dict]:
    """Deliver queued notes once; outstanding acknowledgments never force new turns."""
    if not (run_dir / "supervision" / "steering").is_dir():
        return []
    records = []
    with supervision_lock(run_dir):
        for receipt in steering_records(run_dir):
            if receipt.get("state") not in ({"queued"} if deliver else {"queued", "delivered"}):
                continue
            if deliver:
                import json

                from autofa.session.control import FenceBusy, guarded_fence

                try:
                    with guarded_fence(receipt.get("fence")):
                        # The complete note remains at this path; bounded excerpts
                        # cannot force a readable result into a file envelope.
                        offered = {"id": receipt["id"], "message": receipt["message"][:1500],
                                   "truncated": len(receipt["message"]) > 1500,
                                   "path": str(run_dir / "supervision" / "steering" /
                                               f"{receipt['id']}.json")}
                        size = len(json.dumps(offered, ensure_ascii=False)) + 2
                        if size > max_chars:
                            continue
                        max_chars -= size
                        receipt.update(state="delivered", delivered_at=_now())
                        atomic_write_json(run_dir / "supervision" / "steering" /
                                          f"{receipt['id']}.json", receipt)
                except FenceBusy:
                    continue
                except ValueError as exc:
                    receipt.update(state="superseded", reason=str(exc), updated_at=_now())
                    atomic_write_json(run_dir / "supervision" / "steering" /
                                      f"{receipt['id']}.json", receipt)
                    continue
            records.append(offered if deliver else receipt)
    return records


def respond_to_steering(receipt_id: str, response: str) -> dict:
    """Record discovery's response; ordinary input/contention errors remain tool results."""
    from autofa.mcp.runtime import err

    try:
        return _respond_to_steering(receipt_id, response)
    except (OSError, RuntimeError) as exc:
        return err(str(exc), retryable=True)


def _respond_to_steering(receipt_id: str, response: str) -> dict:
    """Record discovery's response to a supervisor instruction, without changing grades."""
    from autofa.mcp.runtime import err, get_runtime

    directory = get_runtime().run.run_dir()
    try:
        receipt_id = _receipt_id(receipt_id)
    except ValueError as exc:
        return err(str(exc))
    if not response.strip():
        return err("A substantive response or deferral reason is required")
    with supervision_lock(directory):
        path = directory / "supervision" / "steering" / f"{receipt_id}.json"
        receipt = read_json(path)
        if not receipt:
            return err("Unknown steering receipt")
        if receipt.get("response") and receipt["response"] != response:
            return err("Response already recorded; submit a new steering instruction")
        if receipt.get("response") == response:
            return {"ok": True, "receipt": receipt}
        receipt.update(state="addressed", response=response, addressed_at=_now())
        atomic_write_json(path, receipt)
        return {"ok": True, "receipt": receipt}


def steering_prompt(run_dir: Path, *, user_text: str = "") -> str:
    """Offer pending notes at a turn boundary, preserving native results on contention."""
    if user_text.lstrip().startswith("/"):
        return ""
    try:
        notes = discovery_steering(run_dir, deliver=True)
    except (OSError, RuntimeError):
        return ""
    if not notes:
        return ""
    return ("\n\n## Supervisor discovery instructions (not judge policy)\n"
            + "\n".join(
                f"[{note['id']}] {note['message']}\nFull note: {note['path']}"
                + (" (excerpt only; read this file before acting)" if note["truncated"] else "")
                for note in notes)
            + "\nUse respond_to_steering to record your response or deferral.\n")


def lifecycle_request(
    run_dir: Path, action: str, *, fence: dict | None = None, preserve_stop: bool = False,
) -> dict:
    """Persist cooperative intent. A request is not a claim the process has exited."""
    if action not in {"pause", "stop", "finalize", "release"}:
        raise ValueError("Unknown lifecycle request")
    from autofa.session.control import guarded_fence

    control = run_dir / "supervision"
    with supervision_lock(run_dir), guarded_fence(fence, protective=action in {"pause", "stop"}):
        previous = read_json(control / "lifecycle.json")
        if (action == "pause" and previous.get("action") in {"pause", "stop"}
                and (preserve_stop or fence is not None)):
            return {**previous, "preserved": True, "requested_action": action}
        if fence is not None and action in {"finalize", "release"}:
            if previous.get("action") in {"pause", "stop"}:
                return {"ok": False, "status": "stopped_by_supervisor",
                        "action": previous["action"], "requested_action": action,
                        "error": "A saved pause/stop requires an explicit unfenced release; "
                                 "a delayed driver request cannot replace it."}
            if action == "finalize":
                from autofa.session.campaign_policy import directory_status

                status = directory_status(run_dir)
                pending = status.get("overtime_pending")
                if pending and status.get("deadline"):
                    policy = status.get("policy") or {}
                    extended = datetime.fromisoformat(status["deadline"]) + timedelta(
                        seconds=(policy.get("overtime") or {}).get("seconds", 0)
                        - policy.get("delivery_reserve_seconds", 0))
                    pending = datetime.now(UTC) < extended
                if pending:
                    return {"ok": False, "status": "deferred", "retryable": True,
                            "reason": "overtime_pending", "requested_action": action,
                            "error": "Native owner must consume its approved overtime first."}
        receipt = {"action": action, "requested_at": _now(), "state": "requested",
                   "previous_action": previous.get("action"),
                   "semantics": "Cooperative boundary request; in-flight calls may finish. "
                                "Verify owned processes before claiming stopped."}
        atomic_write_json(control / "lifecycle.json", receipt)
        return receipt


def lifecycle_action(run_dir: Path) -> str | None:
    return read_json(run_dir / "supervision" / "lifecycle.json").get("action")


def check_lifecycle(run_dir: Path, *, phase: str = "discovery") -> None:
    action = lifecycle_action(run_dir)
    if action in {"pause", "stop"} or (action == "finalize" and phase == "discovery"):
        from autofa.util.cancellation import check_cancelled, current_cancellation

        if cancellation := current_cancellation():
            cancellation.request_supervisor_stop(action)
            check_cancelled()
            return
        raise SupervisorInterrupted(run_dir, action)


def checkpoint_run(run_dir: Path, destination: Path) -> dict:
    """Copy a quiescent run under its native lock, backing up SQLite including WAL."""
    run_dir, destination = run_dir.resolve(), destination.expanduser().resolve()
    if destination == run_dir or run_dir in destination.parents:
        raise ValueError("Checkpoint destination must be outside the run")
    with run_lock(run_dir):
        from autofa.util.run_copy import copy_run_artifacts

        files = {}
        for target in copy_run_artifacts(run_dir, destination):
            with target.open("rb") as stream:
                files[str(target.relative_to(destination))] = hashlib.file_digest(
                    stream, "sha256").hexdigest()
        receipt = {"version": 1, "source": str(run_dir), "created_at": _now(),
                   "files": files, "sqlite": "consistent backup including committed WAL"}
        atomic_write_json(destination / "checkpoint.json", receipt)
        return receipt


def finalize_run(run_id: str) -> dict:
    """Use the native finalizer under the native lock; never start discovery."""
    from rich.console import Console

    from autofa.config import RunConfig, load_config, restore_harness_settings
    from autofa.mcp.runtime import load_runtime_from_run
    from autofa.mcp.tools_library import library_save
    from autofa.session.finalization import evaluate_final_library

    directory = resolve_run_dir(run_id)
    check_lifecycle(directory, phase="validation")
    with run_lock(directory):
        run = RunConfig.load(run_id)
        initial_round = run.round
        runtime = load_runtime_from_run(run, restore_harness_settings(load_config(), run))
        receipt = {"state": "running", "started_at": _now(), "round": initial_round}
        atomic_write_json(directory / "finalization.json", receipt)
        try:
            outcome = evaluate_final_library(Console(stderr=True))
            library_save(to_catalog=False)
            # Final validation can change matches after the last discovery turn.
            # Flush committed coverage even when a recent tool snapshot was throttled.
            try:
                save_quality_snapshot(runtime, force=True)
            except Exception as exc:
                from autofa.util.model_errors import safe_error_detail

                receipt["snapshot_warning"] = safe_error_detail(str(exc))
                logging.getLogger(__name__).warning(
                    "Final validation saved; quality snapshot failed")
            summary = read_json(directory / "validation_summary.json")
            completion = summary.get("final_validation") or {}
            receipt.update(state="returned", validation=completion, outcome=outcome,
                           report_ready=False,
                           note="Native validation returned; inspect retained detector blockers. "
                                "Publish saved results separately.")
            if runtime.run.round != initial_round:
                raise RuntimeError("Validation unexpectedly changed the discovery round")
        except BaseException as exc:
            interrupted = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            receipt.update(state="interrupted" if interrupted else "failed",
                           error_type=type(exc).__name__)
            raise
        finally:
            receipt["finished_at"] = _now()
            atomic_write_json(directory / "finalization.json", receipt)
        return receipt
