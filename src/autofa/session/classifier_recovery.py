# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite availability recovery, distinct from paired classifier quality experiments."""

from __future__ import annotations

import base64
import hashlib
import struct
import threading
import zlib
from copy import copy, deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from autofa.session.recovery import read_json
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import cancellation_guard, check_cancelled
from autofa.util.model_errors import model_error_diagnostic

STATE_FILE = "classifier_availability_recovery.json"
_LOCK = threading.RLock()
_ATTEMPTS_PER_PROFILE = 2
_PROBE_CALLS = 12


class _RecoveryEvidenceChanged(ValueError):
    """A source mutation invalidates every pending profile's frozen experiment."""


def recovery_status(rt: Any) -> dict:
    state = read_json(rt.run.run_dir() / STATE_FILE)
    attempts = state.get("attempts", [])
    return {"attempts_used": len(attempts), "attempts_per_profile": _ATTEMPTS_PER_PROFILE,
            "model_calls": state.get("model_calls", 0),
            "attempts": [{key: row.get(key) for key in (
                "id", "profile", "status", "outcome", "scope", "model_calls", "next_retry_at",
            )} for row in attempts],
            "scope": "availability_only_not_quality_improvement"}


def _check_recovery_boundary(rt: Any) -> dict:
    """Use tool cancellation for stop/pause and a normal boundary reply for finalize."""
    from autofa.session.campaign_policy import CampaignBoundaryReached, check_campaign_boundary
    from autofa.session.supervision import check_lifecycle, lifecycle_action

    check_lifecycle(rt.run.run_dir(), phase="validation")
    if lifecycle_action(rt.run.run_dir()) == "finalize":
        raise CampaignBoundaryReached({"active": bool(getattr(rt.run, "campaign_policy", None)),
                                       "continue_discovery": False, "finalize_due": True,
                                       "stop_reason": "user_finalize"})
    campaign = check_campaign_boundary(rt)
    if campaign.get("finalize_due"):
        raise CampaignBoundaryReached(campaign)
    return campaign


def _vision_control() -> dict:
    """A locally generated protocol fixture; no evaluation case or model-generated label."""
    def chunk(name: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + name + data
                + struct.pack(">I", zlib.crc32(name + data)))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", 16, 16, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 16) * 16))
            + chunk(b"IEND", b""))
    return {"status": "available", "sha256": hashlib.sha256(data).hexdigest(),
            "data_url": "data:image/png;base64," + base64.b64encode(data).decode()}


def startup_controls(client: Any, *, capabilities: list[str]) -> dict:
    """Check fixed positive/negative protocol controls, and explicit vision requirements."""
    rows = []
    for text, expected in (("The lamp is on.", "yes"), ("The lamp is off.", "no")):
        actual = client.classify_options(text, "Is the lamp on?", options=["yes", "no"])
        rows.append({"control": text, "expected": expected, "actual": actual})
    if "vision" in capabilities:
        actual = client.classify_options("Inspect the attached image.",
                                         "What is its solid color?", options=["red", "blue"],
                                         images=[_vision_control()])
        rows.append({"control": "local_solid_red_image", "expected": "red", "actual": actual})
    return {"ok": True, "checks_passed": all(r["actual"] == r["expected"] for r in rows),
            "results": rows, "scope": "startup_capability_only_not_detector_validation"}


def _reopen_authorized_discovery(rt: Any, recovery_id: str, campaign: dict) -> None:
    """Preserve closing evidence when recovery resumes already-authorized discovery."""
    from autofa.session.smoke_work import digest

    authorized = (campaign.get("continue_discovery") if campaign.get("active") else
                  rt.run.round + 1 < rt.run.max_rounds)
    path = rt.run.run_dir() / "round_validation.json"
    receipt = read_json(path)
    refinement = receipt.get("refinement") or {}
    if not authorized or refinement.get("mode") != "polishing":
        return
    with cancellation_guard():
        archive = (rt.run.run_dir() / "campaign" / "closing-checkpoints"
                   / (digest(receipt) + ".json"))
        atomic_write_json(archive, receipt)
        transition = {"reason": "detector_service_recovered", "recovery_id": recovery_id,
                      "checkpoint": str(archive), "pending_validation_preserved": True}
        atomic_write_json(path, {**receipt, "refinement": {
            **refinement, "mode": "round_checkpoint", "superseded_by": transition}})


def recover_classifier(rt: Any, *, request: dict | None = None,
                       diagnosis: str = "", startup: bool = False) -> dict:
    """Try only approved routes, with durable attempts/calls and unchanged source controls.

    A replacement requires passing controls, but makes no comparison with the down
    baseline. All normal scoring/review remains required under the selected context.
    """
    from autofa.session import classifier_profiles as profiles
    from autofa.session.detector_service import infrastructure_block, record_readiness
    from autofa.session.smoke_work import CALLS_PER_PANEL, LLM_ROWS, SmokeBudgetExceeded, digest
    from autofa.session.usage import usage_scope
    from autofa.util.model_budget import model_call_budget

    incident = infrastructure_block(rt)
    if not incident:
        return {"recovered": False, "status": "service_not_blocked",
                "next_action": "Use compare/adopt for available classifiers that miss controls."}
    _check_recovery_boundary(rt)
    source = items = None
    if not startup:
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            raise ValueError("Explain the service failure and unchanged source-supported controls")
        request = deepcopy(request or {})
        ids, expected = request.get("item_ids"), request.get("expected_outputs")
        abstentions = request.get("expected_abstentions") or []
        if (not isinstance(ids, list) or not 2 <= len(ids) <= LLM_ROWS
                or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids)
                or not isinstance(expected, dict) or set(expected.values()) != {True, False}
                or any(type(v) is not bool for v in expected.values())
                or not isinstance(abstentions, list)
                or any(not isinstance(v, str) for v in abstentions)
                or set(expected) | set(abstentions) != set(ids)
                or set(expected) & set(abstentions)):
            raise ValueError("Recovery requires 2–30 unchanged positive/negative source controls")
        request.update(item_ids=sorted(ids), expected_abstentions=sorted(abstentions))
        source = profiles._source(rt, request)
        from autofa.llm.detection import LLM_CALL_RE

        if not LLM_CALL_RE.search(str(source)):
            raise ValueError("Recovery controls must exercise the detector's LLM evidence path")
        items = deepcopy(profiles._items(rt, request["item_ids"]))
    state = profiles._read_state(rt)
    index = profiles._active_index(rt, state)
    if index < -1:
        raise ValueError("Active classifier does not match this run's approved profile policy")
    active_name = (state["profiles"][index]["name"] if index >= 0 else "baseline")
    choices = [{"name": active_name, "settings": profiles._settings(rt.config)},
               *state["profiles"][index + 1:]]
    # Permission is the saved run policy; zero comparison allowance explicitly
    # disables trying alternatives, but still permits a bounded same-route retry.
    if (not state["policy"]["comparison_limit"]
            or incident.get("active", {}).get("category") == "provider_budget"):
        choices = choices[:1]
    path = rt.run.run_dir() / STATE_FILE
    policy = digest(state["policy"])
    source_identity = digest({"source": source, "items": items, "request": request})
    for target in choices:
        check_cancelled()
        _check_recovery_boundary(rt)
        with _LOCK, cancellation_guard():
            saved = read_json(path) or {"version": 1, "policy": policy,
                                        "attempts": [], "model_calls": 0}
            if saved["policy"] != policy:
                raise ValueError("Recovery policy changed; saved authorization must be preserved")
            prior = [row for row in saved["attempts"] if row["profile"] == target["name"]]
            if len(prior) >= _ATTEMPTS_PER_PROFILE:
                continue
            now = datetime.now(UTC)
            if prior and prior[-1].get("next_retry_at", "") > now.isoformat():
                continue
            attempt_id = digest([incident["incident_id"], target, source_identity, len(prior)])
            attempt = {"id": attempt_id, "profile": target["name"], "settings": target["settings"],
                       "status": "interrupted", "outcome": "unresolved", "model_calls": 0,
                       "incident_id": incident["incident_id"], "source_identity": source_identity,
                       "scope": "startup_capability_only" if startup else "author_source_controls",
                       "request": request, "started_at": now.isoformat(),
                       "next_retry_at": (now + timedelta(seconds=30)).isoformat()}
            saved["attempts"].append(attempt)
            atomic_write_json(path, saved)

        def reserve(attempt_id: str = attempt_id) -> None:
            with _LOCK, cancellation_guard():
                ledger = read_json(path)
                entry = next(row for row in ledger["attempts"] if row["id"] == attempt_id)
                if entry["model_calls"] >= (_PROBE_CALLS if startup else CALLS_PER_PANEL):
                    raise SmokeBudgetExceeded("Availability recovery call allowance exhausted")
                entry["model_calls"] += 1
                ledger["model_calls"] += 1
                atomic_write_json(path, ledger)

        cfg = deepcopy(rt.config)
        for key, value in target["settings"].items():
            setattr(cfg, key, deepcopy(value))
        client = None
        result = None
        try:
            client = profiles._client(cfg)
            with usage_scope(run_dir=rt.run.run_dir(), role="classifier_probe", phase="discovery",
                             round_index=rt.run.round):
                if startup:
                    from autofa.setup_contract import required_capabilities

                    with model_call_budget(reserve):
                        result = startup_controls(client, capabilities=required_capabilities(
                            rt.run.run_dir()))
                else:
                    from autofa.mcp.tools_eval import _smoke_test_feature

                    scoped = copy(rt)
                    scoped.config, scoped.hub_llm = cfg, client
                    result = _smoke_test_feature(scoped, **request, client=client,
                                                frozen_items=items, reserve_model_call=reserve)
                    if (digest(profiles._source(rt, request)) != digest(source)
                            or digest(profiles._items(rt, request["item_ids"])) != digest(items)):
                        raise _RecoveryEvidenceChanged(
                            "Source or control evidence changed during recovery")
            passed = bool(result.get("ok") and result.get("checks_passed")
                          and not result.get("errors"))
            with _LOCK, cancellation_guard():
                saved = read_json(path)
                attempt = next(row for row in saved["attempts"] if row["id"] == attempt_id)
                attempt.update(status="complete", result=result,
                               outcome="controls_passed" if passed else "controls_failed")
                atomic_write_json(path, saved)
            if not passed:
                if (not startup and target["name"] == active_name and result.get("ok")
                        and not result.get("errors")
                        and all(row.get("status") in {"value", "abstained"}
                                for row in result.get("results", []))):
                    record_readiness(rt, {"available": True})
                    return {"recovered": False, "status": "quality_comparison_required",
                            "controls": result, "quality_improvement": False,
                            "next_action": "The selected service answered but missed valid "
                                           "controls. Use ordinary compare/adopt on these same "
                                           "controls; availability recovery cannot authorize "
                                           "a quality replacement."}
                continue
            check_cancelled()
            campaign = _check_recovery_boundary(rt)
            replaced = target["name"] != active_name
            with _LOCK, cancellation_guard():
                # Once selection commits, finish its health/provenance receipt
                # before observing a later stop. Nested saves share this guard.
                if replaced:
                    profiles.apply_profile_selection(rt, target, attempt_id,
                                                     kind="availability_replacement")
                else:
                    rt.hub_llm = rt.llm_readiness = None
                record_readiness(rt, {"available": True})
                _reopen_authorized_discovery(rt, attempt_id, campaign)
                saved = read_json(path)
                attempt = next(row for row in saved["attempts"] if row["id"] == attempt_id)
                attempt.update(outcome=("availability_replacement" if replaced
                                        else "service_recovered"),
                               selection_committed=True)
                atomic_write_json(path, saved)
            return {"recovered": True, "status": attempt["outcome"], "profile": target["name"],
                    "recovery_id": attempt_id, "controls": result, "quality_improvement": False,
                    "next_action": "Re-evaluate affected detectors normally under the selected "
                                   "context; recovery controls establish no grade or recall."}
        except (Exception, SmokeBudgetExceeded) as exc:
            with _LOCK, cancellation_guard():
                saved = read_json(path)
                attempt = next(row for row in saved["attempts"] if row["id"] == attempt_id)
                attempt.update(status="failed", diagnostic=model_error_diagnostic(exc))
                atomic_write_json(path, saved)
            if isinstance(exc, _RecoveryEvidenceChanged):
                raise
        finally:
            close = getattr(getattr(client, "_client", None), "close", None)
            if callable(close):
                close()
    return {"recovered": False, "status": "infrastructure_blocked",
            "recovery": recovery_status(rt), "incident": incident,
            "next_action": "No approved replacement passed within this finite allowance. "
                           "Request a bounded wait/configuration decision or deliver a partial "
                           "result; do not claim that semantic failure mechanisms are exhausted."}
