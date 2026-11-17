# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derived monetary estimates; original execution observations are never rewritten."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

_ADMISSION_LOCK = threading.Lock()
_ADMISSION_CACHE: dict[Path, tuple[tuple, dict]] = {}


def _ledger_signature(directory: Path) -> tuple:
    values = []
    for name in ("usage.sqlite3", "usage.sqlite3-wal", "usage_pricing.json",
                 "usage_gaps.jsonl", "usage_initialization.json", "config.json"):
        try:
            stat = (directory / name).stat()
            # SQLite may create an empty WAL on a read-only connection. It has
            # no committed frames and does not invalidate the same DB snapshot.
            values.append(None if name.endswith("-wal") and stat.st_size == 0 else
                          (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            values.append(None)
    from autofa.session.usage import _read_ledger

    rows = _read_ledger(directory / "usage.sqlite3",
                        "SELECT revision FROM admission_totals WHERE id=1")
    if rows is not None:
        values.append(tuple(rows[0]) if rows else None)
    return tuple(values)


def monetary_history_complete(directory: Path, config: dict) -> bool:
    """Detect missing completed-round history without requiring a single-round split."""
    from autofa.session.usage import _read_ledger

    completed = config.get("round", 0)
    if type(completed) is not int or completed <= 0:
        return True
    rows = _read_ledger(directory / "usage.sqlite3",
                        "SELECT DISTINCT round_index,round_span FROM events")
    if rows is None:
        return False
    recorded = set()
    for row in rows:
        if row["round_index"] is not None:
            recorded.add(row["round_index"])
        span = json.loads(row["round_span"]) if row["round_span"] else None
        if (isinstance(span, list) and len(span) == 2
                and all(type(value) is int and value >= 0 for value in span)
                and span[0] <= span[1]):
            recorded.update(range(span[0], min(span[1] + 1, completed)))
    return not (set(range(completed)) - recorded)


def dispatch_cost_estimate(directory: Path, context: dict) -> float | None:
    """Select a comparable paid operation; a whole turn is never an API-call estimate."""
    from autofa.session.usage import _read_ledger

    path = directory / "usage.sqlite3"
    if not path.exists():
        return None
    identity = context.get("dispatch_identity") or {"role": context.get("role", "discovery")}
    conditions, args = ["role=?", "json_extract(payload,'$.baseline') != 1"], [identity["role"]]
    for key in ("provider", "model"):
        if identity.get(key):
            if key == "model" and identity.get("kind") == "harness":
                conditions.append("(model=? OR "
                                  "json_extract(payload,'$.dispatch_identity.model')=?)")
                args.extend([identity[key], identity[key]])
            else:
                conditions.append(key + "=?")
                args.append(identity[key])
    if identity.get("kind") == "api":
        conditions.append("measurement='combined'")
        conditions.append("json_extract(payload,'$.endpoint') IS ?")
        args.append(identity.get("endpoint"))
    elif identity.get("kind") == "harness":
        conditions.append("measurement='cost'")
    else:
        conditions.append("measurement IN ('combined','cost')")
    rows = _read_ledger(path, "SELECT MAX(cost_usd) AS amount FROM events WHERE "
                        + " AND ".join(conditions), tuple(args))
    amount = rows[0]["amount"] if rows else None
    return amount if isinstance(amount, (int, float)) and amount >= 0 else None


def budget_exposure(directory: Path, result: dict) -> dict:
    """Add changing reservations and carried campaign spend outside the ledger cache."""
    from autofa.session.admission import reservation_summary
    from autofa.session.usage import _accounting_json

    result = deepcopy(result)
    exposure = reservation_summary(directory)
    config = _accounting_json(directory / "config.json")
    prior = (config.get("campaign_state") or {}).get("prior_spend_usd", 0)
    money = result["monetary_view"]
    subtotal = money.get("known_and_derived_subtotal_usd")
    complete = money["cost_complete"]
    money.update(campaign_prior_spend_usd=prior,
                 admission_accounted_usd=(subtotal + prior + exposure["unresolved_reserved_usd"]
                                          if subtotal is not None else None),
                 unresolved_estimated_dispatches=exposure["unresolved_estimated_dispatches"],
                 unresolved_unpriced_dispatches=exposure["unresolved_unpriced_dispatches"],
                 cost_complete=complete and not exposure["unresolved_dispatches"],
                 estimated_cost_complete=(complete
                                          and not exposure["unresolved_unpriced_dispatches"]))
    result["reservations"] = exposure
    return result


def resolved_cost_gaps(directory: Path) -> int:
    """Explicit provider reconciliation can complete money without inventing tokens."""
    from autofa.session.usage import _read_rows

    if not (directory / "usage.sqlite3").exists():
        return 0
    try:
        rows = _read_rows(directory / "usage.sqlite3", """
            SELECT COUNT(*) AS count FROM events e JOIN cost_reconciliations r
            ON json_extract(e.payload,'$.admission_id')=r.id WHERE e.missing_cost=1
        """)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return 0
        raise
    return rows[0]["count"]


def admission_usage(directory: Path) -> dict:
    """Cache only unchanged ledgers; no forecasts, round/model grouping or report reads."""
    from autofa.session.usage import _accounting_json, _aggregate, _now, _read_ledger

    directory = directory.resolve()
    signature = _ledger_signature(directory)
    with _ADMISSION_LOCK:
        cached = _ADMISSION_CACHE.get(directory)
        if cached is not None and cached[0] == signature:
            return budget_exposure(directory, cached[1])
    rows, notices, dispatch_cost = [], [], None
    path = directory / "usage.sqlite3"
    try:
        if path.exists():
            rows = [dict(row) for row in _read_ledger(path,
                "SELECT * FROM admission_totals WHERE id=1") or []]
            if rows and not rows[0].pop("cost_count"):
                rows[0]["cost_usd"] = None
            if not rows or not rows[0]["calls"]:
                rows = []
    except sqlite3.Error:
        notices.append("Saved usage could not be read; admission cost is unavailable.")
    totals = _aggregate(rows)
    derived = {"derived_cost_usd": None, "estimated_observations": 0}
    if totals["missing_cost_calls"] and (directory / "usage_pricing.json").exists():
        derived = derived_usage_estimate(directory)
    derived["reconciled_observations"] = (resolved_cost_gaps(directory)
                                          if totals["missing_cost_calls"] else 0)
    money = monetary_view(totals, derived)
    receipt = _accounting_json(directory / "usage_initialization.json")
    config = _accounting_json(directory / "config.json")
    pristine = (receipt.get("version") == 1 and receipt.get("pristine_start") is True
                and not receipt.get("observations_recorded")
                and not config.get("round")
                and not (directory / "usage_gaps.jsonl").exists() and not notices)
    if not rows and pristine:
        totals.update(known_cost_usd=0.0, cost_complete=True)
        money.update(recorded_subtotal_usd=0.0, known_and_derived_subtotal_usd=0.0,
                     cost_complete=True)
    if not monetary_history_complete(directory, config):
        money["cost_complete"] = totals["cost_complete"] = False
        notices.append("Some completed rounds have no recorded monetary history.")
    if (directory / "usage_gaps.jsonl").exists():
        money["cost_complete"] = totals["cost_complete"] = False
        notices.append("Accounting write gaps remain; the recorded subtotal is incomplete.")
    if config.get("harness"):
        dispatch_cost = dispatch_cost_estimate(directory, {"dispatch_identity": {
            "kind": "harness", "role": "discovery", "provider": config["harness"],
            "model": (config.get("harness_settings") or {}).get("model"),
        }})
    if derived.get("pricing_error"):
        notices.append(derived["note"])
    result = {"available": bool(rows or pristine), "totals": totals, "monetary_view": money,
              "estimated_dispatch_cost_usd": dispatch_cost,
              "reservation_basis": "Estimates are selected per actual dispatch identity; "
              "API calls never borrow a whole harness-turn estimate.",
              "observed_at": _now(), "notices": notices}
    # A concurrent writer changes the key: never cache a read under the old signature.
    if not notices and signature == _ledger_signature(directory):
        with _ADMISSION_LOCK:
            if len(_ADMISSION_CACHE) >= 128:
                _ADMISSION_CACHE.pop(next(iter(_ADMISSION_CACHE)))
            _ADMISSION_CACHE[directory] = (signature, deepcopy(result))
    return budget_exposure(directory, result)


def derived_usage_estimate(directory: Path, pricing: dict | None = None) -> dict:
    """Price only unpriced combined observations, avoiding separate harness counters."""
    from autofa.session.usage import (
        TOKEN_FIELDS,
        _accounting_json,
        _now,
        _price,
        _read_ledger,
        _read_rows,
        select_usage_rate,
        validate_pricing,
    )

    card = pricing if pricing is not None else _accounting_json(directory / "usage_pricing.json")
    result = {"available": False, "derived_cost_usd": None, "estimated_observations": 0,
              "unpriced_observations": 0, "rates_sha256": None, "groups": [],
              "observed_at": _now(), "ledger_modified": False,
              "note": "Derived historical estimate using the stated rate assumptions; "
                      "original usage and provider charges are unchanged."}
    if not card or not (directory / "usage.sqlite3").exists():
        return result
    try:
        card = validate_pricing(card)
    except (ValueError, TypeError, KeyError):
        result["note"] = "Saved pricing is invalid; recorded costs remain available."
        result["pricing_error"] = True
        return result
    result["rates_sha256"] = hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()
    # Group by counter presence as well as identity. Missing cache counters must
    # not be disguised by SUM ignoring nulls; invalid inputs remain unpriced.
    counters = ", ".join(f"SUM({key}) AS {key}" for key in TOKEN_FIELDS)
    presence = ", ".join(f"({key} IS NULL)" for key in TOKEN_FIELDS)
    invalid = ("(input_tokens < COALESCE(cache_read_input_tokens,0) + "
               "COALESCE(cache_creation_input_tokens,0))")
    try:
        reconciliations = _read_rows(directory / "usage.sqlite3",
            "SELECT 1 FROM sqlite_master WHERE name='cost_reconciliations'")
        reconciled_filter = ("AND NOT EXISTS (SELECT 1 FROM cost_reconciliations r "
                             "WHERE r.id=json_extract(events.payload,'$.admission_id'))"
                             if reconciliations else "")
        rows = _read_ledger(directory / "usage.sqlite3", f"""
            SELECT provider, model, role, phase, round_index, round_span,
                   json_extract(payload,'$.endpoint') AS endpoint,
                   json_extract(payload,'$.baseline_unknown') AS baseline_unknown,
                   {invalid} AS invalid_cache, COUNT(*) AS observations, {counters}
            FROM events WHERE cost_usd IS NULL AND measurement='combined'
              AND json_extract(payload,'$.baseline') != 1
              {reconciled_filter}
            GROUP BY provider, model, role, phase, round_index, round_span,
                     endpoint, baseline_unknown, {presence}, {invalid}
        """)
    except sqlite3.Error:
        result["note"] = "Saved usage could not be read; historical estimates are unavailable."
        return result
    if rows is None:
        return result
    total = 0.0
    for row in rows:
        rate = select_usage_rate(directory, row["provider"], row["model"], row["endpoint"],
                                 pricing=card)
        if rate is None or row["baseline_unknown"] or row["invalid_cache"]:
            result["unpriced_observations"] += row["observations"]
            continue
        tokens = {key: row[key] for key in TOKEN_FIELDS if row[key] is not None}
        amount, _ = _price(directory, row["provider"], row["model"], row["endpoint"],
                           tokens, rate=rate)
        if amount is None:
            result["unpriced_observations"] += row["observations"]
            continue
        total += amount
        result["estimated_observations"] += row["observations"]
        result["groups"].append({**{key: row[key] for key in (
            "provider", "model", "role", "phase", "round_index", "round_span", "endpoint")},
            "observations": row["observations"], "derived_cost_usd": amount,
            "rate": rate})
    if result["estimated_observations"]:
        result.update(available=True, derived_cost_usd=total)
    return result


def observed_token_cost_estimate(directory: Path, pricing: dict | None = None) -> dict:
    """Price observed token deltas as a reference, independently of provider billing.

    Harness token and dollar counters describe overlapping work. This view uses
    only token observations and never joins, replaces or adds their account
    dollars/credits. A rate edit affects this reference, not spending admission.
    """
    from autofa.session.usage import (
        CODEX_COMPACTION_USAGE_GAP,
        TOKEN_FIELDS,
        _accounting_json,
        _now,
        _price,
        _read_ledger,
        select_usage_rate,
        validate_pricing,
    )

    directory = Path(directory)
    result = {
        "available": False, "currency": "USD", "estimated_cost_usd": None,
        "complete": False, "token_observations": 0, "priced_token_observations": 0,
        "unpriced_token_observations": 0, "missing_token_observations": 0,
        "ledger_write_gaps": (directory / "usage_gaps.jsonl").exists(),
        "pricing_source": None, "rates_sha256": None, "pricing_error": False,
        "groups": [], "unpriced_groups": [], "by_round": [], "reasons": [],
        "observed_at": _now(), "ledger_modified": False, "affects_spend_admission": False,
        "note": "Reference value of observed tokens at the supplied rates, not a provider "
                "bill or a conversion of credits. Do not add this entire view to reported "
                "costs. Missing tokens remain unestimated. Grouped deltas do not establish "
                "individual request lengths or pricing tiers; the stated rate assumptions apply.",
    }
    card, pricing_reason = None, "pricing_unavailable"
    try:
        if pricing is not None:
            result["pricing_source"] = "explicit_override"
            card = pricing
        else:
            for filename in ("usage_estimate_pricing.json", "usage_pricing.json"):
                if (directory / filename).exists():
                    result["pricing_source"] = filename
                    card = _accounting_json(directory / filename)
                    break
        if card is not None:
            card = validate_pricing(card)
            result["rates_sha256"] = hashlib.sha256(
                json.dumps(card, sort_keys=True, allow_nan=False).encode()).hexdigest()
            pricing_reason = "missing_rate"
    except (OSError, ValueError, TypeError, KeyError):
        card, pricing_reason = None, "pricing_invalid"
        result["pricing_error"] = True
    path = directory / "usage.sqlite3"
    if not path.exists():
        result["reasons"] = [{"reason": "no_observed_tokens", "observations": 0}]
        return result
    counters = ", ".join(f"SUM({key}) AS {key}" for key in TOKEN_FIELDS)
    presence = ", ".join(f"({key} IS NULL)" for key in TOKEN_FIELDS)
    invalid = ("(input_tokens < COALESCE(cache_read_input_tokens,0) + "
               "COALESCE(cache_creation_input_tokens,0))")
    try:
        rows = _read_ledger(path, f"""
            SELECT provider, model, role, phase, round_index, round_span,
                   json_extract(payload,'$.endpoint') AS endpoint,
                   COALESCE(json_extract(payload,'$.baseline_unknown'),
                            json_extract(payload,'$.cumulative'),0) AS unknown_baseline,
                   status=? AS compaction_gap, {invalid} AS invalid_cache,
                   COUNT(*) AS observations, {counters}
            FROM events WHERE measurement IN ('combined','usage')
              AND COALESCE(json_extract(payload,'$.baseline'),0) != 1
            GROUP BY provider, model, role, phase, round_index, round_span,
                     endpoint, unknown_baseline, compaction_gap, {presence}, {invalid}
        """, (CODEX_COMPACTION_USAGE_GAP,))
    except (OSError, sqlite3.Error):
        result["reasons"] = [{"reason": "ledger_unavailable", "observations": None}]
        return result
    if rows is None:
        result["reasons"] = [{"reason": "no_observed_tokens", "observations": 0}]
        return result
    total, reasons, rounds = 0.0, Counter(), {}
    for row in rows:
        count = row["observations"]
        result["token_observations"] += count
        tokens = {key: row[key] for key in TOKEN_FIELDS if row[key] is not None}
        missing = any(key not in tokens for key in ("input_tokens", "output_tokens"))
        result["missing_token_observations"] += count if missing else 0
        group = {key: row[key] for key in (
            "provider", "model", "role", "phase", "round_index", "round_span", "endpoint")}
        group.update(observations=count, tokens=tokens)
        round_row = rounds.setdefault(row["round_index"], {
            "round_index": row["round_index"], "estimated_cost_usd": None,
            "priced_token_observations": 0, "unpriced_token_observations": 0,
            "missing_token_observations": 0, "complete": False,
        })
        round_row["missing_token_observations"] += count if missing else 0
        rate = (select_usage_rate(directory, row["provider"], row["model"], row["endpoint"],
                                  pricing=card) if card is not None else None)
        reason = ("compaction_usage_gap" if row["compaction_gap"] else
                  "unknown_token_baseline" if row["unknown_baseline"] else
                  "missing_token_counters" if missing else
                  "invalid_cache_counters" if row["invalid_cache"] else
                  "missing_model_identity" if not row["model"] else
                  pricing_reason if rate is None else None)
        amount = None
        if reason is None:
            amount, _ = _price(directory, row["provider"], row["model"], row["endpoint"],
                               tokens, rate=rate)
            if amount is None:
                reason = "missing_cache_counters"
        if reason is not None:
            result["unpriced_token_observations"] += count
            round_row["unpriced_token_observations"] += count
            reasons[reason] += count
            result["unpriced_groups"].append({**group, "reason": reason})
            continue
        total += amount
        result["priced_token_observations"] += count
        round_row["priced_token_observations"] += count
        round_row["estimated_cost_usd"] = (round_row["estimated_cost_usd"] or 0) + amount
        result["groups"].append({**group, "estimated_cost_usd": amount, "rate": rate})
    if result["priced_token_observations"]:
        result.update(available=True, estimated_cost_usd=total)
    for index in sorted(rounds, key=lambda value: (value is None, value or 0)):
        row = rounds[index]
        row["complete"] = bool(row["priced_token_observations"]
                               and not row["unpriced_token_observations"]
                               and not result["ledger_write_gaps"])
        result["by_round"].append(row)
    result["reasons"] = [{"reason": reason, "observations": count}
                         for reason, count in sorted(reasons.items())]
    if not rows:
        result["reasons"].append({"reason": "no_observed_tokens", "observations": 0})
    if result["ledger_write_gaps"]:
        result["reasons"].append({"reason": "ledger_write_gaps", "observations": None})
    result["complete"] = bool(result["available"] and not result["unpriced_token_observations"]
                              and not result["ledger_write_gaps"])
    return result


def monetary_view(totals: dict, derived: dict) -> dict:
    """Keep immutable ledger totals separate from newly derived assumptions."""
    recorded, estimated = totals["known_cost_usd"], derived["derived_cost_usd"]
    subtotal = ((recorded or 0) + (estimated or 0)
                if recorded is not None or estimated is not None else None)
    missing = max(0, totals["missing_cost_calls"] - derived["estimated_observations"]
                  - derived.get("reconciled_observations", 0))
    return {"recorded_subtotal_usd": recorded, "derived_estimate_usd": estimated,
            "known_and_derived_subtotal_usd": subtotal, "missing_cost_observations": missing,
            "estimated_guard_only": True,
            "cost_complete": bool(subtotal is not None and not missing),
            "reported_cost_complete": bool(subtotal is not None and not missing
                                           and not totals.get("estimated_cost_usd")
                                           and not estimated),
            "note": "An estimated spending guard, not a provider billing cap. "
                    "Credits and unobservable outer-driver charges are not dollars."}


def horizon_projection(directory: Path, config: dict, *, view: dict, totals: dict,
                       derived: dict, basis_rounds: list[int], settings_consistent: bool) -> dict:
    """Forecast the remaining agreed time using observed compatible discovery spend."""
    from autofa.session.campaign_policy import _effective_limits
    from autofa.session.usage import _read_rows

    policy = config["campaign_policy"]
    reserve = policy.get("delivery_reserve_usd", 0)
    result = {"available": False,
              "method": "observed discovery/validation USD per paid-active second",
              "basis_rounds": basis_rounds, "projected_remaining_cost_usd": None,
              "projected_total_cost_usd": None, "includes_finalization": True,
              "finalization_delivery_allowance_usd": reserve,
              "finalization_is_allowance_not_prediction": True,
              "reason": "Two compatible completed rounds and complete monetary data are required",
              "estimated_guard_only": True}
    if not settings_consistent:
        result["reason"] = "Models, effort, pricing or analysis scope changed; collect a new basis"
        return result
    if (len(basis_rounds) < 2 or not view["cost_complete"]
            or (directory / "usage_gaps.jsonl").exists()):
        return result
    try:
        effective_deadline, effective_cap = _effective_limits(
            policy, config.get("campaign_state") or {})
        deadline = datetime.fromisoformat(effective_deadline)
        if deadline.tzinfo is None:
            raise ValueError("Deadline needs a timezone")
    except (KeyError, TypeError, ValueError):
        result["reason"] = "No agreed time horizon; spend-only guards do not predict workload"
        return result
    now = datetime.now(UTC)
    remaining = max(0, (deadline - now).total_seconds()
                    - policy.get("delivery_reserve_seconds", 0))
    result["remaining_discovery_seconds"] = remaining
    placeholders = ",".join("?" for _ in basis_rounds)
    try:
        rows = _read_rows(directory / "usage.sqlite3", f"""
            SELECT observed_at, json_extract(payload,'$.dispatch_started_at') AS started_at,
                   cost_usd FROM events
            WHERE phase IN ('discovery','validation','round_validation')
              AND round_index IN ({placeholders}) AND measurement IN ('combined','cost')
              AND json_extract(payload,'$.baseline') != 1
        """, tuple(basis_rounds))
        intervals = []
        for row in rows:
            # Wall time between sparse observations includes pauses and offline
            # time. Never silently treat that as an active spending interval.
            if not row["started_at"]:
                raise ValueError("Missing measured dispatch duration")
            start, end = (datetime.fromisoformat(row[key]) for key in ("started_at", "observed_at"))
            if start.tzinfo is None or end.tzinfo is None or end < start:
                raise ValueError("Invalid measured dispatch duration")
            intervals.append((start, end))
        elapsed = 0.0
        previous_end = None
        for start, end in sorted(intervals):
            if previous_end is not None:
                start = max(start, previous_end)
            elapsed += max(0, (end - start).total_seconds())
            previous_end = max(previous_end, end) if previous_end else end
    except (sqlite3.Error, TypeError, ValueError, IndexError):
        result["reason"] = "Paid-active discovery/validation durations are unavailable"
        return result
    if elapsed < 1:
        result["reason"] = "A positive measured paid-active interval is required"
        return result
    amount = sum(row["cost_usd"] or 0 for row in rows) + sum(
        group["derived_cost_usd"] for group in derived["groups"]
        if group["phase"] in {"discovery", "validation", "round_validation"}
        and group["round_index"] in basis_rounds)
    burn = amount / elapsed
    future = burn * remaining
    cap = effective_cap
    accounted = view.get("admission_accounted_usd", view["known_and_derived_subtotal_usd"]) or 0
    available_reserve = reserve
    if cap is not None:
        future = min(future, max(0, cap - accounted - reserve))
        available_reserve = min(reserve, max(0, cap - accounted))
    result.update(available=True, reason=None, observed_interval_seconds=elapsed,
                  observed_burn_usd_per_second=burn, projected_remaining_cost_usd=future,
                  projected_total_cost_usd=accounted + future + available_reserve,
                  available_finalization_allowance_usd=available_reserve,
                  assumptions="Same actual model/effort/workload mix over the remaining authorized "
                  "paid-active time; the observed basis excludes idle and paused intervals. "
                  "Finalization/delivery is the agreed allowance, not a measured forecast; "
                  "in-flight/hidden retries and unobserved driver costs may exceed estimates.")
    return result
