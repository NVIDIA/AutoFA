# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery horizons and owner-applied goal changes, independent of review allowances.

An empty policy preserves the original round/quality stopping rules. Observations are
read-only; only a native boundary may consume the explicitly approved overtime once.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import cancellation_guard

_FIELDS = {
    "version", "mode", "deadline", "spend_limit_usd", "checkpoint_every_rounds",
    "delivery_reserve_seconds", "delivery_reserve_usd", "coverage_target", "coverage_unit",
    "round_limit", "overtime", "unknown_cost_policy",
}
_LIVE_FIELDS = {"max_rounds", "min_failure_modes", "campaign_policy"}
_PENDING_LOCK = threading.RLock()
_PENDING_CACHE: dict[Path, tuple[tuple | None, list[dict]]] = {}


class CampaignBoundaryReached(BaseException):
    """Expected admission boundary; never retry it as a model/backend failure."""

    def __init__(self, status: dict, *, phase: str = "discovery"):
        self.status = status
        self.phase = phase
        self.reason = status.get("stop_reason") or "campaign_boundary"
        super().__init__(f"Campaign boundary reached ({self.reason}) during {phase}")

    def as_result(self) -> dict:
        return {"ok": False, "status": "campaign_boundary", "stop_reason": self.reason,
                "phase": self.phase, "campaign": self.status,
                "message": "Return the saved findings for native finalization; "
                           "do not repeat the blocked paid request."}


def _read(path: Path, *, strict: bool = False) -> dict:
    """Missing optional state is empty; unreadable admission controls must fail closed."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        if strict:
            raise ValueError(f"Cannot read campaign control state {path}: {exc}") from exc
        return {}
    if strict and not isinstance(value, dict):
        raise ValueError(f"Campaign control state {path} must contain a JSON object")
    return value if isinstance(value, dict) else {}


def _date(value: Any) -> datetime:
    try:
        if isinstance(value, datetime):
            parsed = value  # TOML/YAML parsers preserve unquoted aware timestamps.
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise ValueError("not a timestamp")
        if parsed.tzinfo is None:
            raise ValueError("campaign deadline must include a timezone offset")
        return parsed.astimezone(UTC)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and "timezone offset" in str(exc):
            raise
        raise ValueError("campaign deadline must be a timezone-aware ISO8601 timestamp") from exc


def _number(value: Any, field: str, *, integer: bool = False, positive: bool = False) -> Any:
    try:
        finite = isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if (isinstance(value, bool) or not finite or value < 0 or positive and value == 0
            or integer and type(value) is not int):
        kind = "positive integer" if integer else "finite nonnegative number"
        raise ValueError(f"{field} must be a {kind}")
    return value


def validate_campaign_policy(value: Any) -> dict:  # noqa: C901
    """Normalize a portable policy; never infer a deadline from a local timezone."""
    if value is None or value == {}:
        return {}
    if not isinstance(value, dict):
        raise ValueError("campaign_policy must be an object")
    unknown = set(value) - _FIELDS
    if unknown:
        raise ValueError("Unknown campaign policy fields: " + ", ".join(sorted(unknown)))
    policy = copy.deepcopy(value)
    if type(policy.get("version", 1)) is not int or policy.get("version", 1) != 1:
        raise ValueError("Unsupported campaign policy version")
    policy["version"] = 1
    policy.setdefault("mode", "improve_until")
    if policy["mode"] not in ("improve_until", "finish_on_target"):
        raise ValueError("campaign mode must be improve_until or finish_on_target")
    if policy.get("deadline") is not None:
        policy["deadline"] = _date(policy["deadline"]).isoformat()
    else:
        policy.pop("deadline", None)
    for field in ("spend_limit_usd", "delivery_reserve_seconds", "delivery_reserve_usd",
                  "coverage_target"):
        if policy.get(field) is None:
            policy.pop(field, None)
        elif field in policy:
            _number(policy[field], field)
    if policy.get("coverage_target", 0) > 1:
        raise ValueError("coverage_target must be between zero and one")
    for field in ("round_limit", "checkpoint_every_rounds"):
        if policy.get(field) is None:
            policy.pop(field, None)
        elif field in policy:
            _number(policy[field], field, integer=True, positive=True)
    policy.setdefault("checkpoint_every_rounds", 3)
    policy.setdefault("delivery_reserve_seconds", 0)
    policy.setdefault("delivery_reserve_usd", 0)
    policy.setdefault("coverage_unit", "failed_attempts")
    if policy["coverage_unit"] not in ("failed_attempts", "failed_tasks"):
        raise ValueError("coverage_unit must be failed_attempts or failed_tasks")
    policy.setdefault("unknown_cost_policy", "stop")
    if policy["unknown_cost_policy"] not in ("stop", "allow_estimate"):
        raise ValueError("unknown_cost_policy must be stop or allow_estimate")
    if not any(field in policy for field in ("deadline", "spend_limit_usd", "round_limit")):
        raise ValueError("A campaign requires a deadline, spend_limit_usd or explicit round_limit")
    overtime = policy.get("overtime")
    overtime = {} if overtime is None else overtime
    if not isinstance(overtime, dict) or set(overtime) - {"seconds", "usd", "when"}:
        raise ValueError("overtime supports only seconds, usd and when")
    if overtime:
        if overtime.get("when", "minimum_unmet") != "minimum_unmet":
            raise ValueError("overtime.when must be minimum_unmet")
        for field in ("seconds", "usd"):
            _number(overtime.get(field, 0), "overtime." + field)
        if overtime.get("seconds", 0) and "deadline" not in policy:
            raise ValueError("Time overtime requires a deadline")
        if overtime.get("usd", 0) and "spend_limit_usd" not in policy:
            raise ValueError("Spend overtime requires a spend_limit_usd")
        policy["overtime"] = {"when": "minimum_unmet", "seconds": overtime.get("seconds", 0),
                              "usd": overtime.get("usd", 0)}
        if policy.get("deadline"):
            try:
                _date(policy["deadline"]) + timedelta(seconds=overtime.get("seconds", 0))
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    "overtime.seconds extends beyond the supported date range") from exc
        if "spend_limit_usd" in policy:
            _number(policy["spend_limit_usd"] + overtime.get("usd", 0),
                    "spend_limit_usd plus overtime.usd")
    else:
        policy.pop("overtime", None)
    return policy


def _effective_limits(policy: dict, state: dict) -> tuple[str | None, float | None]:
    deadline, cap = policy.get("deadline"), policy.get("spend_limit_usd")
    extension = state.get("overtime") or {}
    if extension.get("used"):
        # A live base-limit edit must not revoke the already consumed finite
        # allowance or consume the configured extension for a second time.
        if deadline and extension.get("deadline") and extension.get("base_deadline"):
            granted = _date(extension["deadline"]) - _date(extension["base_deadline"])
            try:
                deadline = (_date(deadline) + granted).isoformat()
            except OverflowError as exc:
                raise ValueError("Updated deadline plus consumed overtime is out of range") from exc
        if cap is not None and extension.get("base_spend_limit_usd") is not None:
            cap += max(0, extension.get("spend_limit_usd", 0)
                       - extension["base_spend_limit_usd"])
            _number(cap, "Updated spend limit plus consumed overtime")
    return deadline, cap


def _pending_update_records(directory: Path) -> list[dict]:
    """Atomic receipt replacement changes the directory stamp; scan only on changes."""
    root = directory / "supervision" / "controls"
    try:
        stat = root.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns)
    except FileNotFoundError:
        signature = None
    with _PENDING_LOCK:
        cached = _PENDING_CACHE.get(root)
        if cached is not None and cached[0] == signature:
            return copy.deepcopy(cached[1])
        records = [_read(path) for path in sorted(root.glob("*.json"))] if signature else []
        records = [row for row in records if row.get("kind") == "update" and row.get("state")
                   in {"queued", "deferred", "applying"}]
        records.sort(key=lambda row: row.get("created_at", ""))
        if len(_PENDING_CACHE) >= 128:
            _PENDING_CACHE.pop(next(iter(_PENDING_CACHE)))
        _PENDING_CACHE[root] = (signature, records)
        return copy.deepcopy(records)


def restrictive_admission_config(directory: Path, config: dict) -> tuple[dict, dict | None]:
    """Enforce queued tighter limits while the current operation safely drains.

    Only the first valid request at the current revision can commit. Extensions,
    goals, overtime, scoring settings and phase changes wait for that commit. A
    protective projection cannot authorize work that the saved policy forbids.
    """
    from autofa.session.control import verify_fence

    revision = config.get("control_revision", 0)
    saved = validate_campaign_policy(config.get("campaign_policy"))
    state = config.get("campaign_state") or {}
    for record in _pending_update_records(directory):
        if record.get("expected_revision") != revision:
            continue
        try:
            verify_fence(record.get("fence"))
            proposed = normalize_policy_update(config, record.get("settings"))
            next_deadline, next_cap = _effective_limits(proposed["campaign_policy"], state)
        except (OSError, TypeError, ValueError, OverflowError):
            continue  # Native application will reject the same invalid request.
        requested = proposed["campaign_policy"]
        if not requested:
            return config, None  # Disabling/expanding policy waits for native commit.
        deadline, cap = _effective_limits(saved, state)
        projected = copy.deepcopy(saved)
        changed = []
        if next_deadline and (not deadline or _date(next_deadline) < _date(deadline)):
            projected["deadline"] = requested["deadline"]
            changed.append("deadline")
        if next_cap is not None and (cap is None or next_cap < cap):
            projected["spend_limit_usd"] = requested["spend_limit_usd"]
            changed.append("spend_limit_usd")
        for field in ("round_limit",):
            if field in requested and (field not in saved or requested[field] < saved[field]):
                projected[field] = requested[field]
                changed.append(field)
        reserves = {
            "delivery_reserve_seconds": bool(next_deadline and (
                not deadline or _date(next_deadline) <= _date(deadline))),
            "delivery_reserve_usd": next_cap is not None and (cap is None or next_cap <= cap),
        }
        for field, bounded in reserves.items():
            # A larger reserve coupled to a larger total budget is not an early
            # instruction to shorten the old horizon. Wait for full application.
            if bounded and requested.get(field, 0) > saved.get(field, 0):
                projected[field] = requested[field]
                changed.append(field)
        if not changed:
            return config, None  # Later requests at this revision will be stale.
        # A queued update may introduce the first resource horizon, but cannot
        # grant its new quality rule/overtime before the owner applies it.
        projected.setdefault("mode", "improve_until")
        projected = validate_campaign_policy(projected)
        overlay = {"request_id": record.get("id"), "expected_revision": revision,
                   "fields": changed, "full_policy_applied": False,
                   "reason": "Queued tighter limits apply only to new paid admission "
                             "while the current operation drains."}
        return {**config, "campaign_policy": projected}, overlay
    return config, None


def _decision(config: dict, directory: Path, *, now: datetime | None = None,
              usage: dict | None = None, quality: dict | None = None) -> dict:
    policy = validate_campaign_policy(config.get("campaign_policy"))
    if not policy:
        return {"active": False, "continue_discovery": None, "finalize_due": False,
                "stop_reason": None}
    now = (now or datetime.now(UTC)).astimezone(UTC)
    state = config.get("campaign_state") or {}
    target = quality if quality is not None else _read(directory / "discovery_target.json")
    minimum_met = bool(target.get("minimum_met"))
    if quality is None:
        minimum = config.get("min_failure_modes", target.get("min_failure_modes", 1))
        # A saved provisional public grade, or a target from before a live goal
        # edit, cannot close paid discovery on an unverified quality milestone.
        minimum_met = minimum == 0 or (
            target.get("validation_scope") == "current_full_corpus_and_representative_review"
            and target.get("n_validated_failure_modes", 0) >= minimum)
    coverage = target.get("library_quality_coverage") or _read(directory / "coverage_summary.json")
    if "good" in coverage:
        coverage = coverage["good"] or {}
    if "library_quality_coverage" in coverage:
        coverage = coverage["library_quality_coverage"] or {}
    field = "task_coverage" if policy["coverage_unit"] == "failed_tasks" else "coverage"
    measured = coverage.get(field)
    coverage_met = ("coverage_target" not in policy or
                    (coverage.get("status") == "exact" and isinstance(measured, (int, float))
                     and measured >= policy["coverage_target"]))
    deadline, cap = _effective_limits(policy, state)
    extension = state.get("overtime") or {}
    seconds = (_date(deadline) - now).total_seconds() if deadline else None
    totals: dict = {}
    if cap is not None:
        if usage is None:
            from autofa.session.usage import admission_usage

            usage = admission_usage(directory)
        totals = usage.get("totals", usage)
    money = (usage or {}).get("monetary_view") or {}
    spent = money.get("admission_accounted_usd", money.get(
        "known_and_derived_subtotal_usd", totals.get("known_cost_usd")))
    calls = totals.get("calls", totals.get("total_calls", 0)) or 0
    missing = money.get("missing_cost_observations", totals.get("missing_cost_calls", 0))
    read_failed = any("could not be read" in str(note).lower()
                      for note in (usage or {}).get("notices", []))
    initialization = _read(directory / "usage_initialization.json")
    pristine = (initialization.get("pristine_start") is True
                and not initialization.get("observations_recorded")
                and not (directory / "usage_gaps.jsonl").exists()
                and not calls and not config.get("round") and not read_failed
                and not totals.get("missing_usage_calls") and not missing)
    if spent is None and pristine:
        spent = 0.0
    complete = money.get("cost_complete", not missing)
    estimated_complete = money.get("estimated_cost_complete", complete)
    permitted_estimate = policy["unknown_cost_policy"] == "allow_estimate"
    unknown_cost = bool(missing or read_failed
                        or (calls and spent is None)
                        or (cap is not None and spent is None and not pristine)
                        or ((usage or {}).get("available") is False and not pristine)
                        or (not (estimated_complete if permitted_estimate else complete)
                            and not pristine))
    transferred = bool(state.get("budget_transferred"))
    received = state.get("budget_received") or {}
    transfer_pending = False
    if received:
        source_dir = received.get("source_run_dir")
        source = _read(Path(source_dir) / "config.json", strict=True) if source_dir else {}
        handoff = (source.get("campaign_state") or {}).get("budget_transferred") or {}
        transfer_pending = (not received.get("id") or handoff.get("id") != received["id"]
                            or handoff.get("run_id") != config.get("run_id")
                            or source.get("run_id") != received.get("source_run_id"))
    cost_admission_allowed = not (transferred or transfer_pending) and (
        cap is None or not unknown_cost)
    remaining = cap - (spent or 0) if cap is not None else None
    estimate = (usage or {}).get("estimated_dispatch_cost_usd")
    if cap is not None and "estimated_dispatch_cost_usd" not in (usage or {}):
        # Full report/status snapshots do not include the admission estimator.
        # Reuse its cheap ledger-keyed view without loading a library or forecast.
        from autofa.session.usage import admission_usage

        estimate = admission_usage(directory).get("estimated_dispatch_cost_usd")
    if (isinstance(estimate, bool) or not isinstance(estimate, (int, float))
            or not math.isfinite(estimate) or estimate <= 0):
        estimate = None
    # Use the same observed-cost estimator as paid admission. Without this native
    # boundary, a tool can repeatedly reject an unaffordable request while the
    # turn loop sees a positive remaining balance and keeps buying turns.
    discovery_fits = delivery_fits = None
    if remaining is not None and estimate is not None:
        discovery_fits = estimate <= remaining - policy["delivery_reserve_usd"]
        delivery_fits = estimate <= remaining
    reason = None
    lifecycle = _read(directory / "supervision/lifecycle.json", strict=True).get("action")
    stopped = lifecycle in {"stop", "pause"}
    if stopped:
        reason = "user_" + lifecycle
    elif transferred:
        reason = "budget_transferred"
    elif transfer_pending:
        reason = "budget_transfer_pending"
    elif lifecycle == "finalize":
        reason = "user_finalize"
    elif cap is not None and unknown_cost:
        reason = "cost_unavailable"
    elif seconds is not None and seconds <= policy["delivery_reserve_seconds"]:
        reason = "deadline_exhausted" if seconds <= 0 else "deadline_reserve"
    elif remaining is not None and remaining <= policy["delivery_reserve_usd"]:
        reason = "spend_exhausted" if remaining <= 0 else "spend_reserve"
    elif discovery_fits is False:
        reason = "estimated_spend_reserve" if delivery_fits else "estimated_spend_exhausted"
    elif policy.get("round_limit") is not None and config.get("round", 0) >= policy["round_limit"]:
        reason = "round_limit"
    elif policy["mode"] == "finish_on_target" and minimum_met and coverage_met:
        reason = "quality_target_met"
    overtime = policy.get("overtime") or {}
    overtime_pending = bool(reason in {"deadline_reserve", "deadline_exhausted", "spend_reserve",
                                       "spend_exhausted", "estimated_spend_reserve",
                                       "estimated_spend_exhausted"}
                            and not (minimum_met and coverage_met) and not extension.get("used")
                            and (overtime.get("seconds", 0) or overtime.get("usd", 0)))
    return {"active": True, "mode": policy["mode"], "policy": policy,
            "continue_discovery": reason is None, "finalize_due": bool(reason and not stopped),
            "stop_reason": reason, "minimum_met": minimum_met, "coverage_target_met": coverage_met,
            "quality_target_met": minimum_met and coverage_met, "coverage": measured,
            "seconds_remaining": seconds, "remaining_cost_usd": remaining,
            "known_cost_usd": spent, "cost_complete": bool(complete or pristine),
            "cost_admission_allowed": cost_admission_allowed,
            "cost_estimate_accepted": bool(permitted_estimate and not complete
                                            and estimated_complete),
            "estimated_next_dispatch_usd": estimate,
            "discovery_dispatch_fits": discovery_fits, "delivery_dispatch_fits": delivery_fits,
            "estimated_spend_guard": bool(discovery_fits is False),
            "dispatch_estimate_note": ("Observed dispatch cost estimate, not a provider billing "
                                       "cap; actual remaining dollars are reported separately."),
            "deadline": deadline, "spend_limit_usd": cap,
            "overtime_pending": overtime_pending, "overtime_used": bool(extension.get("used")),
            "resources_exhausted": bool(seconds is not None and seconds <= 0
                                         or remaining is not None and remaining <= 0),
            "control_revision": config.get("control_revision", 0)}


def directory_status(run_dir: str | Path, *, now: datetime | None = None,
                     usage: dict | None = None) -> dict:
    """Saved-policy admission view: no library load, detector call or mutation."""
    directory = Path(run_dir)
    config, overlay = restrictive_admission_config(
        directory, _read(directory / "config.json", strict=True))
    status = _decision(config, directory, now=now, usage=usage)
    if overlay:
        status["pending_admission_restriction"] = overlay
    return status


def campaign_status(rt: Any, *, now: datetime | None = None, usage: dict | None = None,
                    quality: dict | None = None) -> dict:
    if not getattr(rt.run, "campaign_policy", None):
        return {"active": False, "continue_discovery": None, "finalize_due": False,
                "stop_reason": None}
    if quality is None:
        from autofa.session.validation_completion import validated_good_target_for_runtime

        quality = validated_good_target_for_runtime(rt)
        if "coverage_target" in rt.run.campaign_policy:
            from autofa.metrics.library_quality_coverage import library_quality_coverage_for_runtime

            quality["library_quality_coverage"] = library_quality_coverage_for_runtime(
                rt, minimum_grade="GOOD")
    return _decision(vars(rt.run), rt.run.run_dir(), now=now, usage=usage, quality=quality)


def check_campaign_boundary(rt: Any, *, now: datetime | None = None,
                            usage: dict | None = None) -> dict:
    """Consume an agreed finite extension once, at an owning-runtime safe boundary."""
    policy = getattr(rt.run, "campaign_policy", None) or {}
    if policy.get("mode", "improve_until") == "improve_until":
        # Current grades cannot end this mode before its resource boundary.
        # Avoid re-ranking the library at every guard and prompt in a healthy
        # turn; label checkpoint quality explicitly. At the boundary, fresh
        # quality below decides whether conditional overtime is authorized.
        preliminary = _decision(vars(rt.run), rt.run.run_dir(), now=now, usage=usage)
        if preliminary.get("continue_discovery") or not preliminary.get("active"):
            return {**preliminary, "quality_basis": "saved_checkpoint"}
    status = campaign_status(rt, now=now, usage=usage)
    if not status.get("overtime_pending"):
        return status
    policy = status["policy"]
    extension = policy["overtime"]
    state = copy.deepcopy(getattr(rt.run, "campaign_state", {}) or {})
    receipt = {"used": True, "activated_at": (now or datetime.now(UTC)).isoformat(),
               "base_deadline": policy.get("deadline"),
               "base_spend_limit_usd": policy.get("spend_limit_usd")}
    if policy.get("deadline"):
        receipt["deadline"] = (_date(policy["deadline"]) + timedelta(
            seconds=extension.get("seconds", 0))).isoformat()
    if policy.get("spend_limit_usd") is not None:
        receipt["spend_limit_usd"] = policy["spend_limit_usd"] + extension.get("usd", 0)
    with cancellation_guard():
        state["overtime"] = receipt
        previous = rt.run.campaign_state
        rt.run.campaign_state = state
        try:
            rt.run.save()
        except BaseException:
            rt.run.campaign_state = previous
            raise
    return campaign_status(rt, now=now, usage=usage)


def checkpoint_fraction(rt: Any, *, status: dict | None = None) -> float | None:
    """Horizon checkpoints can be full scans without becoming terminal polishing."""
    policy = getattr(rt.run, "campaign_policy", None)
    if not policy:
        return None
    status = check_campaign_boundary(rt) if status is None else status
    if status["finalize_due"]:
        return 1.0
    return 1.0 if (rt.run.round + 1) % policy.get("checkpoint_every_rounds", 3) == 0 else .3


def campaign_guidance(rt: Any, *, status: dict | None = None) -> str:
    status = campaign_status(rt) if status is None else status
    if not status["active"]:
        if getattr(rt.run, "control_revision", 0):
            return ("ACTIVE DISCOVERY GOAL UPDATE (supersedes earlier goal instructions): "
                    f"planned rounds={rt.run.max_rounds}; minimum Good detectors="
                    f"{rt.run.min_failure_modes}; revision={rt.run.control_revision}. "
                    "Follow the current native phase and stopping decision.")
        return ""
    return (
        "ACTIVE CAMPAIGN POLICY (supersedes earlier round/goal instructions): "
        + json.dumps({key: status.get(key) for key in (
            "mode", "continue_discovery", "finalize_due", "stop_reason", "minimum_met",
            "coverage_target_met", "deadline", "spend_limit_usd", "control_revision",
        )}, ensure_ascii=False)
        + ". Rounds are checkpoint units; a full intermediate checkpoint or detector-count "
        "milestone does not end improve_until exploration. Preserve completed reviews and "
        "all consumed allowances. Follow the current native phase and stopping decision."
    )


def normalize_policy_update(current: dict, settings: dict) -> dict:
    """Validate a queued goal edit without changing runtime, cohorts or allowance ledgers."""
    from autofa.config import validate_min_failure_modes

    if not isinstance(settings, dict) or set(settings) - _LIVE_FIELDS:
        raise ValueError("Live goal settings support max_rounds, min_failure_modes "
                         "and campaign_policy")
    old = {key: copy.deepcopy(current.get(key, {})) for key in _LIVE_FIELDS}
    new = copy.deepcopy(old)
    if "max_rounds" in settings:
        new["max_rounds"] = _number(settings["max_rounds"], "max_rounds",
                                    integer=True, positive=True)
    if "min_failure_modes" in settings:
        new["min_failure_modes"] = validate_min_failure_modes(settings["min_failure_modes"])
    if "campaign_policy" in settings:
        patch = settings["campaign_policy"]
        if patch is None or patch == {}:
            new["campaign_policy"] = {}
        else:
            if not isinstance(patch, dict):
                raise ValueError("campaign_policy must be an object")
            merged = {**(old["campaign_policy"] or {}), **copy.deepcopy(patch)}
            if isinstance(patch.get("overtime"), dict):
                merged["overtime"] = {**(old["campaign_policy"] or {}).get("overtime", {}),
                                      **patch["overtime"]}
            new["campaign_policy"] = validate_campaign_policy(merged)
    return new


def apply_policy_update(rt: Any, settings: dict, *, now: datetime | None = None,
                        persist: bool = True) -> dict:
    """Apply an already authorized queue request through its owning runtime.

    The queue owns revision checks and must serialize this with other native work.
    With persist=False it commits the updated revision together with run.save().
    """
    lifecycle = _read(rt.run.run_dir() / "supervision/lifecycle.json", strict=True).get("action")
    if lifecycle in {"stop", "pause"}:
        return {"ok": False, "status": "deferred", "reason": "user_" + lifecycle}
    old = {key: copy.deepcopy(getattr(rt.run, key, {})) for key in _LIVE_FIELDS}
    new = normalize_policy_update(old, settings)
    if new == old:
        return {"ok": True, "status": "unchanged", "before": old, "after": new}
    directory = rt.run.run_dir()
    transition = None
    with cancellation_guard():
        for key, value in new.items():
            setattr(rt.run, key, copy.deepcopy(value))
            setattr(rt.config, key, copy.deepcopy(value))
        status = campaign_status(rt, now=now)
        can_continue = (status.get("continue_discovery") if status["active"] else
                        new["max_rounds"] > rt.run.round + 1
                        or new["min_failure_modes"] > old["min_failure_modes"])
        receipt_path = directory / "round_validation.json"
        receipt = _read(receipt_path)
        refinement = receipt.get("refinement") or {}
        if can_continue and refinement.get("mode") == "polishing":
            encoded = json.dumps(receipt, sort_keys=True, default=str).encode()
            key = hashlib.sha256(encoded).hexdigest()
            archive = directory / "campaign" / "closing-checkpoints" / (key + ".json")
            atomic_write_json(archive, receipt)
            transition = {"reason": "authorized_goal_update", "checkpoint": str(archive),
                          "pending_validation_preserved": refinement.get("status") != "complete"}
            # Reopen family discovery while retaining any unfinished closing
            # response. Do not turn an acknowledged response into a new question.
            atomic_write_json(receipt_path, {**receipt, "refinement": {
                **refinement, "mode": "round_checkpoint", "superseded_by": transition}})
            rt.run.campaign_state = {**(getattr(rt.run, "campaign_state", {}) or {}),
                                     "last_transition": transition}
        from autofa.session.round_stagnation import migrate_round_policy

        migrate_round_policy(rt, old_max_rounds=old["max_rounds"],
                             supersede_capacity=transition is not None)
        if persist:
            rt.run.save()
    return {"ok": True, "status": "applied", "before": old, "after": new,
            "transition": transition, "campaign": status}
