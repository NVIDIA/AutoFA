# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare optional report navigation only at an explicit report boundary."""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextvars import copy_context
from copy import deepcopy
from threading import Thread
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from autofa.core.dependencies import execution_bundle
from autofa.core.localisation import detector_hash
from autofa.metrics.association import truthy_bucket
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import (
    Cancellation,
    cancellation_guard,
    cancellation_scope,
    check_cancelled,
)
from autofa.util.model_errors import model_error_diagnostic

HIGHLIGHT_REQUEST_SECONDS = 30.0
HIGHLIGHT_PASS_SECONDS = 60.0


def reuse_report_highlights(record, previous, item, source, *, deps=None, bundle=None):
    """Keep exact, current report evidence when scoring reuses a deferred result."""
    pending = (record or {}).get("deferred_localisation") or {}
    saved = (previous or {}).get("deferred_localisation") or {}
    if (pending.get("status") != "pending" or not pending.get("request_hash") or not previous
            or saved.get("status") not in {"complete", "unavailable"}
            or pending["request_hash"] != saved.get("request_hash")):
        return record
    from autofa.core.localisation import validate_localisation

    if not validate_localisation(previous, item, source, deps=deps, bundle=bundle):
        return record
    return deepcopy(previous)


def _receipt(rt):
    try:
        saved = json.loads((rt.run.run_dir() / "report_highlights.json").read_text())
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) and saved.get("version") == 1 else {}


def _unavailable(record, reason):
    result = deepcopy(record)
    result["deferred_localisation"]["status"] = "unavailable"
    result["notes"] = [*(result.get("notes") or []), reason]
    return result


def prepare_report_highlights(
    rt: Any, *, console: Console | None = None, pending_only: bool = False,
    request_timeout: float = HIGHLIGHT_REQUEST_SECONDS,
    overall_timeout: float = HIGHLIGHT_PASS_SECONDS,
) -> dict:
    """Checkpoint optional quotes without changing predictions, scores, or reviews.

    No caller in scoring, judging, dashboard rendering, or the report renderer
    invokes this function. A cancelled final report resumes its saved selection.
    """
    from autofa.session.campaign_policy import CampaignBoundaryReached
    from autofa.session.supervision import (
        lifecycle_cancellation,
        stop_requested,
        supervisor_cancelled,
    )
    from autofa.session.usage import usage_scope

    if stop_requested(rt.run.run_dir()):
        return {"ok": False, "status": "pending", "reason": "supervisor_stop"}
    try:
        with (usage_scope(run_dir=rt.run.run_dir(), phase="highlighting"),
              lifecycle_cancellation(rt.run.run_dir()) as cancellation):
            return _prepare_report_highlights(
                rt, console=console, pending_only=pending_only,
                request_timeout=request_timeout, overall_timeout=overall_timeout)
    except CampaignBoundaryReached as exc:
        return {"ok": False, "status": "pending", "reason": exc.reason}
    except asyncio.CancelledError:
        if supervisor_cancelled(cancellation):
            return {"ok": False, "status": "pending", "reason": "supervisor_stop"}
        raise
    except Exception as exc:
        # Optional navigation must not prevent saving already committed scores.
        # Keep unfinished work pending for an explicit save or Resume retry.
        if stop_requested(rt.run.run_dir()):
            return {"ok": False, "status": "pending", "reason": "supervisor_stop"}
        receipt = _receipt(rt)
        receipt["error_details"] = model_error_diagnostic(exc)
        receipt["error"] = ("Optional report highlighting unavailable: "
                            + receipt["error_details"]["message"])
        try:
            with cancellation_guard():
                atomic_write_json(rt.run.run_dir() / "report_highlights.json", receipt)
        except asyncio.CancelledError:
            if supervisor_cancelled(cancellation):
                return {"ok": False, "status": "pending", "reason": "supervisor_stop"}
            raise
        if console is not None:
            console.print(f"[yellow]{receipt['error']} Saved results remain available; "
                          "unfinished highlighting can be retried on Resume.[/yellow]")
        return {"ok": False, "status": "pending", "error": receipt["error"]}


def _prepare_report_highlights(rt, *, console, pending_only, request_timeout, overall_timeout):  # noqa: C901
    from autofa.core.localisation import resolve_deferred_localisation, validate_localisation
    from autofa.data.parquet_store import MAX_BATCH_BYTES, item_memory_bytes
    from autofa.mcp.tools_eval import (
        _fetch_items_by_id,
        _fn_deps,
        _llm_adapter,
        _llm_eval_settings,
    )
    from autofa.session.operation_admission import admit_operation

    if request_timeout <= 0 or overall_timeout <= 0:
        raise ValueError("Highlighting deadlines must be positive")
    started = time.monotonic()
    deadline = started + overall_timeout
    saved = _receipt(rt)
    if pending_only and saved.get("status") != "pending":
        return {"ok": True, "status": "not_required", "completed": 0}
    with admit_operation("library_save", {}) as busy:
        if busy:
            raise RuntimeError("Report highlighting is waiting for active detector work: "
                               + busy["error"])
        check_cancelled()
        selected = []
        resumed = {(row["name"], row["item_id"], row["request_hash"])
                   for row in saved.get("remaining", [])} if pending_only else None
        for fn in rt.library.functions:
            fingerprint = detector_hash(fn.source, _fn_deps(fn), bundle=execution_bundle(fn))
            for item_id, record in fn.failure_localisations.items():
                deferred = record.get("deferred_localisation") or {}
                if (deferred.get("status") != "pending" or not deferred.get("request_hash")
                        or record.get("detector_hash") != fingerprint
                        or not truthy_bucket(record.get("assigned_label"))):
                    continue
                row = {"name": fn.name, "item_id": item_id,
                       "request_hash": deferred["request_hash"]}
                if resumed is None or (fn.name, item_id, deferred["request_hash"]) in resumed:
                    selected.append(row)
        if not selected:
            if saved.get("status") == "pending":
                with cancellation_guard():
                    atomic_write_json(rt.run.run_dir() / "report_highlights.json",
                                      {**saved, "status": "complete", "remaining": []})
            return {"ok": True, "status": "not_required", "completed": 0}
        receipt = {"version": 1, "status": "pending", "remaining": selected,
                   "total": len(selected), "completed": 0,
                   "request_timeout_seconds": request_timeout,
                   "overall_timeout_seconds": overall_timeout}
        path = rt.run.run_dir() / "report_highlights.json"
        with cancellation_guard():
            atomic_write_json(path, receipt)
        by_item = {}
        for row in selected:
            by_item.setdefault(row["item_id"], []).append(row)

        client = _llm_adapter()
        failures = []
        stop_submitting = False

        def jobs():
            for item in _fetch_items_by_id(rt.dataset, set(by_item), max_items=None):
                for row in by_item.pop(str(item["_item_id"]), []):
                    yield row, item

        def resolve(job, cancellation, expires):
            row, item = job
            fn = rt.library.get(row["name"])
            record = deepcopy(fn.failure_localisations[row["item_id"]])

            class BoundedLocaliser:
                def localise(self, *args, **kwargs):
                    check_cancelled()
                    remaining = expires - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Optional highlighting request deadline expired")
                    response = client.localise(*args, timeout=remaining, **kwargs)
                    check_cancelled()
                    if time.monotonic() >= expires:
                        raise TimeoutError("Optional highlighting request deadline expired")
                    return response

            with cancellation_scope(cancellation):
                check_cancelled()
                try:
                    resolved = resolve_deferred_localisation(
                        record, item, fn.source, BoundedLocaliser(),
                        deps=_fn_deps(fn), bundle=execution_bundle(fn),
                        cache_dir=rt.run.run_dir() / "localisation_cache",
                    )
                except ValueError:
                    if validate_localisation(record, item, fn.source,
                                             deps=_fn_deps(fn), bundle=execution_bundle(fn)):
                        raise
                    resolved = _unavailable(record, "Report highlighting source is stale.")
                check_cancelled()
                if (resolved.get("deferred_localisation") or {}).get("status") == "pending":
                    raise RuntimeError("Optional report highlighting did not complete")
                return row, resolved

        def save(row, record):
            with cancellation_guard():
                rt.library.get(row["name"]).failure_localisations[row["item_id"]] = record
                rt.persist_library()
                receipt["remaining"] = [entry for entry in receipt["remaining"] if entry != row]
                receipt["completed"] += 1
                atomic_write_json(path, receipt)

        with Progress(
            SpinnerColumn(), TextColumn("[cyan]{task.description}"), BarColumn(),
            MofNCompleteColumn(), TimeElapsedColumn(), TextColumn("{task.fields[detector]}"),
            console=console, refresh_per_second=4,
        ) as progress:
            task = progress.add_task("Preparing report highlights", total=len(selected),
                                     detector="")
            workers = _llm_eval_settings(rt.config)[2]
            pending = {}
            source = iter(jobs())
            lookahead = None
            exhausted = False
            try:
                while pending or not exhausted:
                    check_cancelled()
                    now = time.monotonic()
                    if now >= deadline:
                        failures.append(model_error_diagnostic(
                            TimeoutError("Optional highlighting overall time budget expired")))
                        break
                    # Keep at most one job per worker and bound retained input bytes.
                    while not stop_submitting and not exhausted and len(pending) < workers:
                        if lookahead is None:
                            try:
                                lookahead = next(source)
                            except StopIteration:
                                exhausted = True
                                break
                        size = item_memory_bytes(lookahead[1])
                        if (pending and sum(entry[3] for entry in pending.values())
                                + size > MAX_BATCH_BYTES):
                            break
                        from autofa.session.supervision import stop_action

                        cancellation = Cancellation(
                            lambda: stop_action(rt.run.run_dir()))
                        expires = min(deadline, time.monotonic() + request_timeout)
                        future = Future()

                        def run(future, job, cancellation, expires):
                            if future.set_running_or_notify_cancel():
                                try:
                                    future.set_result(resolve(job, cancellation, expires))
                                except BaseException as exc:
                                    future.set_exception(exc)

                        # Optional transport work must not keep process exit waiting
                        # after its deadline if an SDK ignores cancellation/timeout.
                        Thread(target=copy_context().run,
                               args=(run, future, lookahead, cancellation, expires),
                               daemon=True, name="report-highlights").start()
                        pending[future] = (lookahead[0], cancellation, expires, size)
                        lookahead = None
                    if not pending:
                        break
                    done, _ = wait(pending, timeout=min(.1, max(0, deadline - time.monotonic())),
                                   return_when=FIRST_COMPLETED)
                    # Persist every completed success before handling failures, regardless
                    # of submission order. A slow first item cannot hide later successes.
                    for future in done:
                        row, cancellation, expires, size = pending.pop(future)
                        try:
                            saved_row, resolved = future.result()
                        except asyncio.CancelledError:
                            from autofa.util.cancellation import current_cancellation

                            if cancellation.supervisor_requested:
                                parent = current_cancellation()
                                if parent is not None:
                                    parent.request_supervisor_stop(
                                        cancellation.supervisor_action or "stop")
                            raise
                        except Exception as exc:
                            failures.append({"name": row["name"], "item_id": row["item_id"],
                                             **model_error_diagnostic(exc)})
                            stop_submitting = True
                        else:
                            save(saved_row, resolved)
                            progress.update(task, completed=receipt["completed"],
                                            detector=saved_row["name"])
                    for future, (row, cancellation, expires, _size) in list(pending.items()):
                        if time.monotonic() >= expires:
                            cancellation.cancel()
                            future.cancel()
                            pending.pop(future)
                            reason = ("Optional highlighting overall time budget expired"
                                      if expires == deadline else
                                      "Optional highlighting request deadline expired")
                            failures.append({"name": row["name"], "item_id": row["item_id"],
                                             **model_error_diagnostic(TimeoutError(reason))})
                            stop_submitting = True
                    if stop_submitting and not pending:
                        break
            finally:
                # Do not wait for an uncooperative transport after the optional deadline.
                # Cancelled workers cannot publish late cache writes or library changes.
                for future, (_, cancellation, _, _) in pending.items():
                    cancellation.cancel()
                    future.cancel()
                source.close()
            if exhausted and not failures:
                for rows in by_item.values():
                    for row in rows:
                        record = rt.library.get(row["name"]).failure_localisations[row["item_id"]]
                        save(row, _unavailable(record,
                                               "Report highlighting source item is unavailable."))
                        progress.update(task, completed=receipt["completed"])
        with cancellation_guard():
            receipt["status"] = "pending" if receipt["remaining"] else "complete"
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if failures:
                receipt["errors"] = failures
                receipt["error"] = "Optional highlighting paused: " + failures[0]["message"]
            atomic_write_json(path, receipt)
        if failures and console is not None:
            console.print(
                "[yellow]Optional report highlighting paused; completed passages are saved. "
                "Publishing available results; Resume can finish pending highlights.[/yellow]")
        return {"ok": not failures, "status": receipt["status"],
                "completed": receipt["completed"], "remaining": len(receipt["remaining"]),
                **({"errors": failures} if failures else {})}



async def prepare_report_highlights_async(
    rt: Any, *, console: Console | None = None, pending_only: bool = False,
) -> dict:
    """Drain cancelled background extraction before returning to the session."""
    cancellation = Cancellation()
    with cancellation_scope(cancellation):
        worker = asyncio.create_task(asyncio.to_thread(
            prepare_report_highlights, rt, console=console, pending_only=pending_only))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancelling = asyncio.create_task(asyncio.to_thread(cancellation.cancel))
        draining = asyncio.gather(cancelling, worker, return_exceptions=True)
        while not draining.done():
            try:
                await asyncio.shield(draining)
            except asyncio.CancelledError:
                continue
        draining.result()
        raise
