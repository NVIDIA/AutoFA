#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed discovery delivery contract; stdlib-only orchestration of native verification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_SPEC = importlib.util.spec_from_file_location(
    "autofa_delivery_records", Path(__file__).with_name("driver.py")
)
records = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(records)

DEFAULT_TIMEOUT_SECONDS = 900


def _delivery_timeout(native: dict[str, Any]) -> float:
    value = native.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Delivery native.timeout_seconds must be a positive finite number")
    try:
        timeout = float(value)
    except OverflowError as exc:
        raise ValueError(
            "Delivery native.timeout_seconds must be a positive finite number"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Delivery native.timeout_seconds must be a positive finite number")
    return timeout


def delivery_contract(goal: dict[str, Any]) -> dict[str, Any]:
    """Require an agreed destination rather than silently choosing local-only."""
    value = goal.get("delivery")
    if not isinstance(value, dict) or value.get("destination") not in {"cloud", "local"}:
        raise ValueError("Agree goal.delivery.destination as cloud or local before launching")
    for key in ("publish_on_stop", "summary_required"):
        if key in value and type(value[key]) is not bool:
            raise ValueError("delivery." + key + " must be boolean")
    return {
        "destination": value["destination"],
        "publish_on_stop": value.get("publish_on_stop", False),
        "summary_required": value.get("summary_required", True),
    }


def execution_settings(goal: dict[str, Any]) -> dict[str, bool]:
    return {"publish": delivery_contract(goal)["destination"] == "cloud"}


def prepare_run_config(
    goal: dict[str, Any], run_spec: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Write native JSON (a YAML subset), mapping the agreed destination deterministically."""
    if not isinstance(run_spec, dict) or not isinstance(run_spec.get("execution", {}), dict):
        raise ValueError("Run specification and execution must be objects")
    value = {**run_spec, "execution": {**run_spec.get("execution", {}), **execution_settings(goal)}}
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records.atomic(output, value)
    return {"path": str(output), "execution": value["execution"], "sha256": records.digest(output)}


def _document(path: Path) -> dict[str, Any]:
    return records.object_file(path) if path.exists() else {}


def _bindings(
    root: Path, native: dict | None = None, *, run_id: str | None = None
) -> tuple[dict, list[str]]:
    configured = _document(root / "delivery" / "config.json")
    supervisor = _document(root / "supervision" / "config.json")
    managed = next(
        (
            row.get("native")
            for row in supervisor.get("runs", [])
            if row.get("run_id") == run_id and row.get("native")
        ),
        None,
    )
    configured_run = next(
        (
            row.get("native")
            for row in configured.get("runs", [])
            if row.get("run_id") == run_id and row.get("native")
        ),
        None,
    )
    binding = (
        managed
        or configured_run
        or native
        or configured.get("native")
        or supervisor.get("native")
        or {}
    )
    runs = [row["run_id"] for row in supervisor.get("runs", [])] or configured.get("run_ids", [])
    argv = binding.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or not v for v in argv):
        raise ValueError("Delivery requires the bound native executable argument array")
    _delivery_timeout(binding)
    return binding, runs


def configure(root: Path, native: dict[str, Any], run_ids: list[str]) -> dict[str, Any]:
    root = Path(root)
    campaign = records.object_file(root / "campaign.json")
    delivery_contract(campaign["goal"])
    if not run_ids or any(not isinstance(run, str) or not run or "/" in run for run in run_ids):
        raise ValueError("Bind nonempty native run IDs for delivery")
    _bindings(root, native)
    for run_id in run_ids:
        _bindings(root, native, run_id=run_id)
    output = {"version": 1, "native": native, "run_ids": list(dict.fromkeys(run_ids))}
    (root / "delivery").mkdir(exist_ok=True)
    records.atomic(root / "delivery" / "config.json", output)
    return output


def _invoke(
    native: dict[str, Any], run_id: str, *, publish: bool, local: bool = False
) -> dict[str, Any]:
    timeout = _delivery_timeout(native)
    env = {**os.environ, **native.get("env", {})}
    if native.get("config") and "AUTOFA_CONFIG" not in native.get("env", {}):
        env["AUTOFA_CONFIG"] = str(native["config"])
    if native.get("home") and "AUTOFA_HOME" not in native.get("env", {}):
        env["AUTOFA_HOME"] = str(native["home"])
    argv = [*native["argv"], "publish", "--run", run_id, "--saved-only", "--json"]
    if not publish:
        argv.append("--verify-only")
    if local:
        argv.append("--local-only")
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            subprocess.run(
                argv,
                cwd=native.get("cwd"),
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
            stdout.seek(0)
            output = stdout.read(2 * 1024 * 1024 + 1)
    except subprocess.TimeoutExpired:
        return {
            "status": "publication_uncertain" if publish and not local else "blocked",
            "run_id": run_id,
            "reason": "Native delivery timed out; reconcile before retry",
        }
    if len(output) > 2 * 1024 * 1024:
        return {"status": "blocked", "run_id": run_id, "reason": "Delivery response exceeded bound"}
    try:
        value = json.loads(output)
        if not isinstance(value, dict) or value.get("run_id") != run_id:
            raise ValueError("Mismatched run identity")
    except ValueError:
        return {
            "status": "blocked",
            "run_id": run_id,
            "reason": "Native delivery returned invalid structured evidence",
        }
    return value


def reconcile_delivery(
    campaign_dir: Path, run_id: str, *, native: dict | None = None, allow_publish: bool = False
) -> dict[str, Any]:
    """Bounded saved-only native operation; callers run this outside the watchdog thread."""
    root = Path(campaign_dir)
    campaign = records.object_file(root / "campaign.json")
    contract = delivery_contract(campaign["goal"])
    binding, bound = _bindings(root, native, run_id=run_id)
    if bound and run_id not in bound:
        raise ValueError("Delivery run is not bound to this campaign")
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("Invalid native run ID")
    directory = root / "delivery"
    directory.mkdir(exist_ok=True)
    path = directory / (run_id + ".json")
    previous = _document(path)
    local = contract["destination"] == "local"
    result = _invoke(binding, run_id, publish=False, local=local)
    can_publish = (
        allow_publish
        and result.get("status") == "pending"
        and (previous.get("status") != "publication_uncertain" or result.get("retry_safe") is True)
        and (campaign.get("state") != "stopped" or contract["publish_on_stop"])
    )
    if can_publish:
        # Durable outer intent precedes spawning, including process creation/crash failures.
        records.atomic(
            path,
            {
                "status": "publication_uncertain",
                "run_id": run_id,
                "campaign_id": campaign["id"],
                "revision": campaign["revision"],
            },
        )
        result = _invoke(binding, run_id, publish=True, local=local)
    elif (
        previous.get("status") == "publication_uncertain"
        and result.get("status") == "pending"
        and result.get("retry_safe") is not True
    ):
        result = {**result, "status": "publication_uncertain", "retry_safe": False}
    receipt = {
        **result,
        "campaign_id": campaign["id"],
        "revision": campaign["revision"],
        "destination": contract["destination"],
        "observed_at": records.now(),
    }
    records.atomic(path, receipt)
    return receipt


def prepare_summary(campaign_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Pin the findings message to exact immutable evidence before sending it to the user."""
    root = Path(campaign_dir)
    campaign = records.object_file(root / "campaign.json")
    _, bound = _bindings(root)
    if summary.get("outcome") not in {"complete", "partial"}:
        raise ValueError("Summary outcome must be complete or partial")
    records.required_text(summary, "text")
    findings = summary.get("findings")
    actions = summary.get("next_actions")
    if not isinstance(findings, list) or not findings or not isinstance(actions, list):
        raise ValueError("Summary needs supported findings and next_actions")
    cost = summary.get("cost")
    if not isinstance(cost, dict) or not cost:
        raise ValueError("Summary needs final cost or an explicit unknown-cost explanation")
    if cost.get("status") == "unknown":
        records.required_text(cost, "reason")
    evidence = {}
    for run_id in bound:
        receipt = _document(root / "delivery" / (run_id + ".json"))
        if receipt.get("status") != "verified":
            raise ValueError("Verify every deliverable before preparing the final summary")
        evidence[run_id] = {
            key: receipt.get(key) for key in ("url", "snapshot", "metrics", "finalization")
        }
        if receipt.get("url") and receipt["url"] not in summary["text"]:
            raise ValueError("Summary text must contain every exact verified report URL")
    if not evidence:
        raise ValueError("No bound deliverables")
    output = {
        "version": 1,
        "campaign_id": campaign["id"],
        "revision": campaign["revision"],
        "summary": summary,
        "runs": evidence,
    }
    output["sha256"] = hashlib.sha256(records.canonical(output)).hexdigest()
    records.atomic(root / "delivery" / "summary.json", output)
    return output


def _acknowledged(root: Path, summary_hash: str) -> bool:
    owner = _document(root / "supervision" / "owner.json")
    config = _document(root / "supervision" / "config.json")
    monitoring = _document(root / "supervision" / "monitoring.json")
    session_id = monitoring.get("host_session_id", config.get("host", {}).get("session_id"))
    for path in (root / "supervision" / "host-acks").glob("*.json"):
        ack = _document(path)
        if (
            ack.get("summary_sha256") == summary_hash
            and ack.get("channel") == "foreground"
            and ack.get("generation") == owner.get("generation")
            and ack.get("session_id") == session_id
            and ack.get("monitoring_revision", 0) == monitoring.get("revision", 0)
            and ack.get("id")
            and ack.get("observation_id")
        ):
            return True
    # Detached adapters must have acknowledged this exact prepared message.
    summary = _document(root / "delivery" / "summary.json")
    for path in (root / "supervision" / "outbox").glob("*.json"):
        message = _document(path)
        ack = message.get("delivery_receipt", {})
        if (
            message.get("id") == "final-summary-" + summary_hash
            and message.get("message") == summary.get("summary", {}).get("text")
            and message.get("state") == "delivered"
            and ack.get("delivered") is True
            and ack.get("id") == message["id"]
            and ack.get("remote_id")
            and ack.get("channel") == config.get("delivery", {}).get("channel")
        ):
            return True
    return False


def completion_evidence(
    campaign_dir: Path, run_ids: list[str] | None = None, *, native: dict | None = None
) -> dict[str, Any]:
    """Require fresh native readback and acknowledged findings before completion."""
    root = Path(campaign_dir)
    campaign = records.object_file(root / "campaign.json")
    contract = delivery_contract(campaign["goal"])
    _, bound = _bindings(root, native)
    selected = run_ids or bound
    reasons, runs = [], {}
    if not selected or (bound and set(selected) != set(bound)):
        reasons.append("Every bound run requires verified delivery")
    for run_id in selected:
        result = reconcile_delivery(root, run_id, native=native)
        runs[run_id] = result
        if result.get("status") != "verified":
            reasons.append(run_id + ": publication is " + str(result.get("status")))
        outcome = result.get("finalization", {})
        partial_blocked = (
            outcome.get("state") == "blocked"
            and outcome.get("report_generated")
            and outcome.get("resumable")
        )
        stopped_saved = (
            outcome.get("state") == "stopped"
            and outcome.get("stop_reason") == "user_stop"
            and outcome.get("report_generated")
            and contract["publish_on_stop"]
        )
        if (
            outcome.get("state") not in {"complete", "completed", "completed_with_gaps"}
            and not partial_blocked
            and not stopped_saved
        ):
            reasons.append(run_id + ": terminal native validation/outcome is missing")
    summary = _document(root / "delivery" / "summary.json")
    goal_sha256 = hashlib.sha256(records.canonical(campaign["goal"])).hexdigest()
    committed = campaign.get("completion", {})
    committed = committed.get("delivery", committed)
    accepted_contract = (
        campaign.get("state") == "complete"
        and committed.get("complete") is True
        and committed.get("campaign_id") == campaign["id"]
        and committed.get("completed_revision") == campaign["revision"]
        and committed.get("contract_revision") == summary.get("revision")
        and committed.get("summary_sha256") == summary.get("sha256")
        and committed.get("goal_sha256") == goal_sha256
    )
    expected_summary_revision = (
        committed["contract_revision"] if accepted_contract else campaign["revision"]
    )
    if contract["summary_required"]:
        unsigned = {key: value for key, value in summary.items() if key != "sha256"}
        if hashlib.sha256(records.canonical(unsigned)).hexdigest() != summary.get("sha256"):
            reasons.append("Prepared findings summary hash is missing or changed")
        if (
            summary.get("campaign_id") != campaign["id"]
            or summary.get("revision") != expected_summary_revision
        ):
            reasons.append("Current findings summary has not been prepared")
        else:
            if (
                any(
                    row.get("finalization", {}).get("state")
                    in {"blocked", "stopped", "completed_with_gaps"}
                    or row.get("finalization", {}).get("retained_results_complete") is False
                    or row.get("finalization", {}).get("target", {}).get("minimum_met") is False
                    for row in runs.values()
                )
                and summary.get("summary", {}).get("outcome") != "partial"
            ):
                reasons.append(
                    "Unmet native targets or validation gaps must be delivered as a partial result"
                )
            for run_id, result in runs.items():
                saved_run = summary.get("runs", {}).get(run_id, {})
                if saved_run.get("snapshot") != result.get("snapshot") or saved_run.get(
                    "url"
                ) != result.get("url"):
                    reasons.append(run_id + ": findings summary refers to a different snapshot")
            # Completion already committed this exact acknowledged message. A
            # guard restart cannot make past user delivery un-happen. New or
            # reopened contracts still require the current owner's acknowledgment.
            if not accepted_contract and not _acknowledged(root, summary.get("sha256", "")):
                reasons.append("Final findings message has no host/delivery acknowledgment")
    return {
        "complete": not reasons,
        "reasons": reasons,
        "runs": runs,
        "summary_sha256": summary.get("sha256"),
        "campaign_id": campaign["id"],
        "contract_revision": expected_summary_revision,
        "checked_revision": campaign["revision"],
        "goal_sha256": goal_sha256,
    }
