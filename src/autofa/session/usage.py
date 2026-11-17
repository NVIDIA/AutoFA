# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable execution usage, independent of detector quality and source-run efficiency."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from autofa.session.recovery import read_json, run_lock
from autofa.util.atomic import atomic_write_json
from autofa.util.usage_values import nonnegative_number as _number
from autofa.util.usage_values import usage_endpoint as _endpoint

log = logging.getLogger(__name__)
_CONTEXT: ContextVar[dict | None] = ContextVar("autofa_usage", default=None)
CODEX_COMPACTION_USAGE_GAP = "codex_compaction_usage_unavailable"
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
_NON_ROUND_PHASES = {"finalization", "publication", "highlighting", "taxonomy", "preflight"}
_DATABASE_SETUP_LOCK = threading.Lock()
_INITIALIZED_DATABASES: dict[Path, tuple[int, int]] = {}


class _UsageConnections:
    """Short-lived writer cache shared by the workers of one native operation."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pid = os.getpid()
        self.closed = False
        self.connections: dict[Path, tuple[sqlite3.Connection, tuple[int, int]]] = {}

    @contextmanager
    def lease(self, directory: Path):
        if self.pid != os.getpid():
            with closing(_connect(directory)) as db:
                yield db
            return
        with self.lock:
            if self.closed:
                # A cancelled tool can finish before a late worker reaches its
                # accounting call. It still gets an independent durable write.
                with closing(_connect(directory)) as db:
                    yield db
                return
            path = (directory / "usage.sqlite3").resolve()
            saved = self.connections.pop(path, None)
            stat = path.stat() if path.exists() else None
            identity = (stat.st_dev, stat.st_ino) if stat is not None else None
            if saved is not None and saved[1] != identity:
                saved[0].close()
                saved = None
            if saved is None:
                if len(self.connections) >= 4:
                    self.connections.pop(next(iter(self.connections)))[0].close()
                db = _connect(directory)
                stat = path.stat()
                saved = (db, (stat.st_dev, stat.st_ino))
            self.connections[path] = saved
            # SQLite serializes writers itself. This lock also makes a reused
            # connection safe across copied worker contexts, without holding it
            # during inference or delaying the durability of any usage event.
            yield saved[0]

    def close(self) -> None:
        with self.lock:
            self.closed = True
            for db, _ in self.connections.values():
                db.close()
            self.connections.clear()


_CONNECTIONS: ContextVar[_UsageConnections | None] = ContextVar("autofa_usage_connections",
                                                              default=None)


@contextmanager
def usage_connection_scope():
    """Reuse bounded writer connections only until the current operation finishes."""
    if _CONNECTIONS.get() is not None:
        yield
        return
    connections = _UsageConnections()
    token = _CONNECTIONS.set(connections)
    try:
        yield
    finally:
        _CONNECTIONS.reset(token)
        connections.close()


@contextmanager
def _writing_database(directory: Path):
    connections = _CONNECTIONS.get()
    if connections is None:
        with closing(_connect(directory)) as db:
            yield db
    else:
        with connections.lease(directory) as db:
            yield db


@lru_cache(maxsize=128)
def _cached_json(path: Path, signature: tuple) -> dict:
    return read_json(path)


def _accounting_json(path: Path) -> dict:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    return _cached_json(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))


@contextmanager
def _read_database(path: Path):
    """Use normal SQLite locking for live ledgers, with a read-only archive fallback.

    An empty WAL is not evidence that a ledger is immutable: a writer may commit
    immediately after the check. Only an unwritable directory with no pending
    WAL can use immutable mode. Existing WAL contents must always be replayed.
    """
    path = path.resolve()
    wal = path.with_name(path.name + "-wal")
    try:
        active_wal = wal.stat().st_size > 0
    except FileNotFoundError:
        active_wal = False
    writable = os.access(path.parent, os.W_OK)
    if not writable and active_wal and not path.with_name(path.name + "-shm").is_file():
        raise sqlite3.OperationalError("Active usage WAL has no readable shared-memory index")
    query = "?mode=ro&immutable=1" if not writable and not active_wal else "?mode=ro"
    with closing(sqlite3.connect(path.resolve().as_uri() + query, uri=True, timeout=1)) as db:
        yield db


def _read_rows(path: Path, query: str, parameters: tuple = ()) -> list[sqlite3.Row]:
    """Retry a transient read failure once using a fresh SQLite snapshot."""
    for attempt in range(2):
        try:
            with _read_database(path) as db:
                db.row_factory = sqlite3.Row
                return list(db.execute(query, parameters))
        except sqlite3.DatabaseError:
            if attempt:
                raise
    raise AssertionError("unreachable")


def _read_ledger(path: Path, query: str, parameters: tuple = ()) -> list[sqlite3.Row] | None:
    """Read the ledger, or return None when it is missing or has no committed schema.

    The schema is created in one transaction, so a file without tables holds no
    observations: a first writer is still creating it, or stopped before its
    commit. Readers treat it like a missing ledger. Other failures propagate.
    """
    if not path.is_file():
        return None
    try:
        return _read_rows(path, query, parameters)
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            raise
    if not _read_rows(path, "SELECT 1 FROM sqlite_master LIMIT 1"):
        return None
    # The first writer may have committed in between; its schema is now complete.
    return _read_rows(path, query, parameters)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_ancestor(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if read_json(candidate / "config.json").get("run_id"):
            return candidate
    return None


def _record_gap(directory: Path, error: BaseException, *, stage: str) -> None:
    log.warning("AutoFA usage %s failed (%s); accounting may be incomplete",
                stage, type(error).__name__)
    try:
        with (directory / "usage_gaps.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": _now(), "error_type": type(error).__name__,
                                     "stage": stage}) + "\n")
    except OSError:
        pass


def capture_context(
    run_dir: str | Path | None = None,
    *,
    role: str = "discovery",
    phase: str | None = None,
    refresh_round: bool = False,
) -> dict | None:
    """Capture attribution without creating profiles; mark failures in an existing run."""
    bound = _CONTEXT.get()
    directory = _run_ancestor(Path(run_dir)) if run_dir is not None else None
    explicit = Path(run_dir) if run_dir is not None else None
    matches_bound = (
        (
            explicit is None
            or directory == Path(bound["run_dir"])
            or (explicit == Path(bound["run_dir"]))
            or Path(bound["run_dir"]) in explicit.parents
        )
        if bound
        else False
    )
    if bound and matches_bound:
        result = dict(bound)
    else:
        if explicit is not None and directory is None:
            return None
        live_round = None
        if directory is None:
            run = None
            try:
                from autofa.config import RunConfig
                from autofa.mcp.runtime import get_runtime
                from autofa.session.supervision import resolve_run_dir

                run = get_runtime().run
                # RunConfig.run_dir() creates directories. An accounting observer
                # must not create a stale runtime's run under a different profile.
                candidate = (resolve_run_dir(run.run_id) if isinstance(run, RunConfig)
                             else Path(run.run_dir()))
                directory = (
                    candidate if read_json(candidate / "config.json").get("run_id") else None
                )
                live_round = run.round
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                run_id = getattr(run, "run_id", None)
                if isinstance(run_id, str):
                    # Do not call autofa_home()/run_dir(): those create a profile
                    # when a stale runtime belongs to a different environment.
                    root = Path(os.environ.get("AUTOFA_HOME", "~/.autofa")).expanduser() / "runs"
                    candidate = root / run_id
                    if (Path(run_id).name == run_id and run_id not in {".", ".."}
                            and read_json(candidate / "config.json").get("run_id") == run_id):
                        _record_gap(candidate, exc, stage="context attribution")
                return None
        if directory is None:
            return None
        config = _accounting_json(directory / "config.json")
        result = {
            "run_dir": str(directory),
            "round_index": (live_round if live_round is not None else config.get("round")),
            "phase": "discovery",
        }
    if refresh_round and result.get("phase") == "discovery":
        result["round_index"] = read_json(Path(result["run_dir"]) / "config.json").get("round")
    result["role"] = role
    if phase is not None:
        result["phase"] = phase
    if result.get("phase") in _NON_ROUND_PHASES:
        result["round_index"] = None
    return result


@contextmanager
def usage_scope(
    *,
    run_dir: str | Path | None = None,
    role: str | None = None,
    phase: str | None = None,
    round_index: int | None = None,
    exact_dir: bool = False,
):
    """Bind attribution for native worker contexts, including non-discovery workflows."""
    context = (None if exact_dir else
               capture_context(run_dir, role=role or "discovery", phase=phase))
    if context is None and run_dir is not None:
        context = {
            "run_dir": str(run_dir),
            "role": role or "discovery",
            "phase": phase or "discovery",
            "round_index": round_index,
        }
    if context is not None:
        if role is not None:
            context["role"] = role
        if phase is not None:
            context["phase"] = phase
        if round_index is not None:
            context["round_index"] = round_index
        if context["phase"] in _NON_ROUND_PHASES:
            context["round_index"] = None
    token = _CONTEXT.set(context)
    try:
        with usage_connection_scope():
            yield context
    finally:
        _CONTEXT.reset(token)


def _tokens(usage: dict | None) -> dict:
    return {
        key: value
        for key, value in (usage or {}).items()
        if key in TOKEN_FIELDS and type(value) is int and value >= 0
    }


def validate_pricing(value: Any) -> dict:
    """Accept an explicit USD rate card, never a bundled guess about provider pricing."""
    if not isinstance(value, dict) or set(value) != {"version", "currency", "rates"}:
        raise ValueError("Pricing requires version, currency and rates")
    if type(value["version"]) is not int or value["version"] != 1 or value["currency"] != "USD":
        raise ValueError("Pricing supports version 1 and USD only")
    if not isinstance(value["rates"], list):
        raise ValueError("rates must be a list")
    rates, identities = [], set()
    required = {
        "provider",
        "model",
        "input_per_million",
        "output_per_million",
        "source",
        "effective_at",
    }
    allowed = required | {"endpoint", "cached_input_per_million", "cache_creation_per_million",
                          "basis", "retrieved_at", "billing_provider", "pricing_model",
                          "assumptions"}
    for row in value["rates"]:
        if not isinstance(row, dict) or not required <= row.keys() or row.keys() - allowed:
            raise ValueError(
                "Invalid pricing fields; specify exact provider/model and rate provenance"
            )
        clean = dict(row)
        for key in ("provider", "model", "source", "effective_at"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Pricing {key} must be a nonempty string")
        for key in ("retrieved_at", "billing_provider", "pricing_model", "assumptions"):
            if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                raise ValueError(f"Pricing {key} must be a nonempty string")
        if row.get("basis", "configured") not in {"configured", "public", "proxy"}:
            raise ValueError("Pricing basis must be configured, public or proxy")
        if row.get("basis") in {"public", "proxy"}:
            if (not row.get("retrieved_at")
                    or not row.get("endpoint") and row["provider"] not in {"codex", "claude"}):
                raise ValueError("Public/proxy pricing requires exact endpoint and retrieval date")
            if not _endpoint(row["source"]):
                raise ValueError("Public/proxy pricing requires a source HTTP(S) URL")
            if row["basis"] == "proxy" and not row.get("assumptions"):
                raise ValueError("Proxy pricing requires explicit assumptions")
        for key in allowed:
            if key.endswith("per_million") and key in row and _number(row[key]) is None:
                raise ValueError(f"Pricing {key} must be finite and nonnegative")
        if "endpoint" in row:
            clean["endpoint"] = _endpoint(row["endpoint"])
            if not clean["endpoint"] or clean["endpoint"] != row["endpoint"].rstrip("/"):
                raise ValueError(
                    "Pricing endpoint must be an HTTP(S) URL without credentials/query"
                )
        identity = (clean["provider"], clean["model"], clean.get("endpoint"))
        if identity in identities:
            raise ValueError("Duplicate pricing identity")
        identities.add(identity)
        rates.append(clean)
    return {"version": 1, "currency": "USD", "rates": rates}


def set_usage_pricing(run_dir: Path, value: Any, *, estimate_only: bool = False) -> dict:
    """Set accounting rates or a live-safe, independent token-reference rate card."""
    pricing = validate_pricing(value)
    if not run_dir.is_dir():
        raise ValueError("Pricing requires an existing run or standalone usage directory")
    if estimate_only:
        from autofa.session.supervision import supervision_lock

        with supervision_lock(run_dir):
            atomic_write_json(run_dir / "usage_estimate_pricing.json", pricing)
        return {
            "ok": True,
            "pricing": pricing,
            "scope": "observed-token reference estimate only",
            "ledger_modified": False,
            "affects_spend_admission": False,
            "note": "Safe during an active run. Reported billing, native limits and "
                    "accounting rates remain unchanged; no inference is performed.",
        }
    with run_lock(run_dir):
        atomic_write_json(run_dir / "usage_pricing.json", pricing)
    return {
        "ok": True,
        "pricing": pricing,
        "scope": "future observations in this usage directory",
        "note": "Existing recorded costs retain their original pricing provenance.",
    }


def initialize_run_accounting(run_dir: Path, pricing: dict | None = None, *,
                              before_first_dispatch: bool = False) -> dict:
    """Install agreed startup rates before dispatch; caller holds the native run lock.

    Resume preserves its existing snapshot. An explicit pricing edit uses
    set_usage_pricing instead, so changing a global default cannot reprice a run.
    """
    directory = Path(run_dir)
    if not directory.is_dir():
        raise ValueError("Accounting startup requires an existing run directory")
    receipt = directory / "usage_initialization.json"
    if before_first_dispatch and not receipt.exists():
        # Only the creator can assert this boundary. Resume must never infer
        # zero historical spend merely from a missing/empty old ledger.
        pristine = not any((directory / name).exists()
                           for name in ("usage.sqlite3", "usage_gaps.jsonl"))
        atomic_write_json(receipt, {"version": 1, "initialized_at": _now(),
                                   "pristine_start": pristine})
    path = directory / "usage_pricing.json"
    if path.exists():
        return {"ok": True, "initialized": False,
                "pricing": validate_pricing(_accounting_json(path)), "source": "saved_run"}
    if not pricing:
        return {"ok": True, "initialized": False, "pricing": None, "source": "unavailable"}
    validated = validate_pricing(pricing)
    atomic_write_json(path, validated)
    return {"ok": True, "initialized": True, "pricing": validated, "source": "startup"}


def derived_usage_estimate(run_dir: Path, pricing: dict | None = None) -> dict:
    """Estimate saved unpriced API events without altering their original measurements."""
    from autofa.session.usage_estimates import derived_usage_estimate as estimate

    return estimate(Path(run_dir), pricing)


def admission_usage(run_dir: Path) -> dict:
    """Return a bounded monetary view for paid-call admission, without full status scans."""
    from autofa.session.usage_estimates import admission_usage as snapshot

    try:
        return snapshot(Path(run_dir))
    except (OSError, ValueError, TypeError, sqlite3.Error):
        return {"available": False, "totals": _aggregate([]),
                "monetary_view": {"known_and_derived_subtotal_usd": None,
                                  "cost_complete": False, "estimated_guard_only": True},
                "estimated_dispatch_cost_usd": None,
                "notices": ["Saved usage or pricing could not be read; admission cost "
                            "is unavailable, not zero."]}


def select_usage_rate(directory: Path, provider: str, model: str | None,
                      endpoint: str | None, *, pricing: dict | None = None) -> dict | None:
    """Resolve one explicit rate, preferring the exact endpoint over a generic rate."""
    pricing = pricing if pricing is not None else _accounting_json(directory / "usage_pricing.json")
    rows = [row for row in pricing.get("rates", [])
            if row.get("provider") == provider and row.get("model") == model
            and row.get("endpoint") in {None, endpoint}]
    return max(rows, key=lambda item: "endpoint" in item) if rows else None


def _price(
    directory: Path,
    provider: str,
    model: str | None,
    endpoint: str | None,
    tokens: dict,
    *,
    rate: dict | None = None,
) -> tuple[float | None, dict | None]:
    if rate is None:
        rate = select_usage_rate(directory, provider, model, endpoint)
        if rate is None:
            return None, None
    row = rate
    if any(key not in tokens for key in ("input_tokens", "output_tokens")):
        return None, None
    # The input price applies to all input unless a distinct cache rate is declared.
    # A missing category count is safe only when that category has the same price.
    if (
        sum(
            tokens.get(key, 0) for key in ("cache_read_input_tokens", "cache_creation_input_tokens")
        )
        > tokens["input_tokens"]
    ):
        return None, None
    amount = (
        tokens["input_tokens"] * row["input_per_million"]
        + tokens["output_tokens"] * row["output_per_million"]
    )
    for key, price in (
        ("cache_read_input_tokens", "cached_input_per_million"),
        ("cache_creation_input_tokens", "cache_creation_per_million"),
    ):
        adjustment = row.get(price, row["input_per_million"]) - row["input_per_million"]
        if adjustment and key not in tokens:
            return None, None
        amount += tokens.get(key, 0) * adjustment
    return amount / 1_000_000, row


def _scope_fingerprint(directory: Path) -> str:
    """Hash cost-relevant saved settings; never store configuration secrets in the ledger."""
    config = _accounting_json(directory / "config.json")
    fields = (
        "harness",
        "harness_settings",
        "classifier_profile_settings",
        "top_x",
        "eval_limit",
        "parquet_paths",
        "model_id",
        "benchmarks",
        "benchmark",
        "objective_settings",
        "behavior_strictness",
        "analysis_outputs",
        "analysis_focus",
    )
    settings = {key: config.get(key) for key in fields}
    settings["pricing"] = _accounting_json(directory / "usage_pricing.json")
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def _connect(directory: Path) -> sqlite3.Connection:
    # Changing journal mode races on the first concurrent worker connection.
    # Retry just this read/write setup; never retry the inference that produced usage.
    for attempt in range(5):
        try:
            return _open_database(directory)
        except sqlite3.OperationalError as exc:
            if attempt == 4 or getattr(exc, "sqlite_errorcode", None) not in {
                sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED,
            }:
                raise
            time.sleep(.025 * 2 ** attempt)
    raise AssertionError("unreachable")


def _open_database(directory: Path) -> sqlite3.Connection:
    path = (directory / "usage.sqlite3").resolve()
    # A scoped operation may lease this connection from different workers. The
    # scope serializes complete transactions before permitting the next lease.
    db = sqlite3.connect(path, timeout=5, check_same_thread=False)
    try:
        with _DATABASE_SETUP_LOCK:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if _INITIALIZED_DATABASES.get(path) != identity:
                db.execute("PRAGMA journal_mode=WAL")
                # One transaction: readers see either no tables or the whole schema,
                # and concurrent first writers cannot both seed the admission row.
                db.execute("BEGIN IMMEDIATE")
                db.execute("""CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT, role TEXT NOT NULL,
                phase TEXT NOT NULL, round_index INTEGER, round_span TEXT, source_id TEXT,
                measurement TEXT NOT NULL, status TEXT NOT NULL, observed_at TEXT NOT NULL,
                input_tokens INTEGER, output_tokens INTEGER, cache_read_input_tokens INTEGER,
                cache_creation_input_tokens INTEGER, reasoning_output_tokens INTEGER,
                total_tokens INTEGER, cost_usd REAL, cost_kind TEXT NOT NULL, credits REAL,
                missing_usage INTEGER NOT NULL, missing_cost INTEGER NOT NULL,
                unattributed INTEGER NOT NULL, payload TEXT NOT NULL)""")
                db.execute("CREATE TABLE IF NOT EXISTS snapshots "
                           "(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                db.execute("CREATE INDEX IF NOT EXISTS events_source ON events(source_id)")
                db.execute("CREATE INDEX IF NOT EXISTS events_dispatch_cost ON events"
                           "(role, provider, model, measurement, cost_usd)")
                db.execute("CREATE INDEX IF NOT EXISTS events_round_span ON events"
                           "(round_index, round_span)")
                db.execute("CREATE TABLE IF NOT EXISTS cost_reconciliations "
                           "(id TEXT PRIMARY KEY, cost_usd REAL NOT NULL, source TEXT NOT NULL)")
                _initialize_admission_totals(db)
                db.commit()
                _INITIALIZED_DATABASES[path] = identity
    except BaseException:
        db.close()
        raise
    return db


def _initialize_admission_totals(db: sqlite3.Connection) -> None:
    """Maintain the admission subtotal in the same transaction as each observation.

    Active admission reads a single row, rather than rescanning a growing ledger
    under the dispatch lock. Its revision is a SQLite commit value, independent
    of filesystem timestamp resolution.
    """
    measures = {
        "calls": "1", "cost_count": "cost_usd IS NOT NULL", "cost_usd": "COALESCE(cost_usd,0)",
        "reported_cost_usd": "CASE WHEN cost_kind='reported' THEN COALESCE(cost_usd,0) ELSE 0 END",
        "estimated_cost_usd": ("CASE WHEN cost_kind='estimated' "
                               "THEN COALESCE(cost_usd,0) ELSE 0 END"),
        "credits": "COALESCE(credits,0)", "missing_usage": "missing_usage",
        "missing_cost": "missing_cost", "unattributed": "unattributed",
        "compaction_usage_gaps": f"status='{CODEX_COMPACTION_USAGE_GAP}'",
    }
    existed = db.execute("SELECT 1 FROM sqlite_master WHERE name='admission_totals'").fetchone()
    db.execute("CREATE TABLE IF NOT EXISTS admission_totals (id INTEGER PRIMARY KEY, "
               "revision INTEGER NOT NULL, " + ", ".join(f"{key} REAL NOT NULL" for key in measures)
               + ")")
    if not existed:
        db.execute("INSERT INTO admission_totals VALUES (1, 0" + ", 0" * len(measures) + ")")
    # REPLACE uses explicit DELETE below, so both sides update these totals.
    for action, prefix, sign in (("INSERT", "NEW", "+"), ("DELETE", "OLD", "-")):
        assignments = []
        for key, expr in measures.items():
            # Qualify identifiers in these fixed internal expressions.
            qualified = re.sub(r"\b(cost_usd|cost_kind|credits|missing_usage|missing_cost|"
                               r"unattributed|status)\b",
                               lambda m, qualifier=prefix: qualifier + "." + m[0], expr)
            assignments.append(f"{key}={key}{sign}({qualified})")
        db.execute(f"CREATE TRIGGER IF NOT EXISTS admission_{action.lower()} AFTER {action} "
                   f"ON events WHEN json_extract({prefix}.payload,'$.baseline') != 1 BEGIN "
                   "UPDATE admission_totals SET revision=revision+1, "
                   + ", ".join(assignments) + " WHERE id=1; END")


def _cumulative_delta(
    db: sqlite3.Connection,
    key: str,
    incoming: dict,
    *,
    baseline: bool,
    baseline_unknown: bool,
    reset: bool,
) -> tuple[dict, bool]:
    saved = db.execute("SELECT payload FROM snapshots WHERE id=?", (key,)).fetchone()
    previous = json.loads(saved[0]) if saved and not reset else {}
    unknown = False
    if baseline and previous and not reset:
        # Repeated start snapshots must not discard spend left by an interrupted turn.
        if any(
            previous.get(name) is not None and amount is not None and amount > previous[name]
            for name, amount in incoming.items()
        ):
            previous["_unattributed"] = True
        for name, amount in incoming.items():
            if amount is not None and name not in previous:
                previous[name] = amount
        db.execute("UPDATE snapshots SET payload=? WHERE id=?", (json.dumps(previous), key))
        return {}, False
    values, deltas, backwards = dict(previous), {}, False
    for name, amount in incoming.items():
        if amount is None:
            continue
        old = previous.get(name)
        field_unknown = baseline_unknown and old is None and not reset
        unknown |= field_unknown
        if old is not None and amount < old and not reset:
            # Failure responses can zero counters. Never erase observed spend or assume reset.
            backwards = True
            continue
        values[name] = amount
        if not baseline and not field_unknown:
            deltas[name] = amount - (old or 0)
    pending_gap = values.pop("_unattributed", False)
    db.execute(
        "INSERT OR REPLACE INTO snapshots VALUES (?, ?)", (key, json.dumps(values, sort_keys=True))
    )
    return deltas, unknown or backwards or pending_gap


def source_cost_kind(context: dict | None, source_id_prefix: str) -> str | None:
    """Read the first selected dollar counter for a source without creating files.

    Returning to an old native counter after temporarily counting a credit-derived
    estimate would charge the same interval twice. Callers must retain the first
    observed counter even when a later response temporarily omits it. Unknown
    sources return None; read failures propagate so callers can mark a gap rather
    than accidentally select a different counter.
    """
    if context is None:
        return None
    path = Path(context["run_dir"]) / "usage.sqlite3"
    if not path.is_file():
        return None
    choices = {f"{source_id_prefix}:reported": "reported",
               f"{source_id_prefix}:estimated": "estimated"}
    rows = _read_ledger(
        path, "SELECT source_id FROM events WHERE measurement='cost' AND source_id IN (?, ?) "
        "ORDER BY rowid LIMIT 1", tuple(choices),
    )
    return choices.get(rows[0][0]) if rows else None


def record_usage(
    context: dict | None,
    *,
    event_id: str,
    provider: str,
    model: str | None,
    usage: dict | None = None,
    cost_usd: float | None = None,
    cost_kind: str = "unknown",
    credits: float | None = None,
    source_id: str | None = None,
    credit_source_id: str | None = None,
    cumulative: bool = False,
    baseline: bool = False,
    baseline_unknown: bool = False,
    reset: bool = False,
    status: str = "complete",
    effort: str | None = None,
    endpoint: str | None = None,
    round_span: list[int] | None = None,
    note: str | None = None,
    measurement: str = "combined",
) -> None:
    """Record only usage metadata; observer failures must not repeat committed model work.

    Additive event IDs can be enriched after lazy response parsing. Cumulative observation
    IDs are immutable and idempotent; source IDs identify stable session/model counters.
    Input tokens include cache read/create; reasoning tokens are a subset of output tokens.
    """
    if context is None:
        return
    try:
        _record_usage(
            context,
            event_id=event_id,
            provider=provider,
            model=model,
            usage=usage,
            cost_usd=cost_usd,
            cost_kind=cost_kind,
            credits=credits,
            source_id=source_id,
            credit_source_id=credit_source_id,
            cumulative=cumulative,
            baseline=baseline,
            baseline_unknown=baseline_unknown,
            reset=reset,
            status=status,
            effort=effort,
            endpoint=endpoint,
            round_span=round_span,
            note=note,
            measurement=measurement,
        )
    except Exception as exc:
        try:
            _record_gap(Path(context["run_dir"]), exc, stage="ledger write")
        except (OSError, KeyError, TypeError):
            pass


def _record_usage(context: dict, **event) -> None:
    from autofa.util.model_errors import safe_error_detail

    directory = Path(context["run_dir"])
    if not directory.is_dir():
        raise ValueError("Usage destination must already exist")
    if not isinstance(event["event_id"], str) or not event["event_id"]:
        raise ValueError("Usage requires a stable observation ID")
    if event["measurement"] not in {"combined", "usage", "cost"}:
        raise ValueError("Unknown usage measurement channel")
    tokens = _tokens(event["usage"])
    amount, credits = _number(event["cost_usd"]), _number(event["credits"])
    kind = event["cost_kind"] if amount is not None else "unknown"
    if kind not in {"reported", "estimated", "unknown"}:
        raise ValueError("Unknown cost kind")
    if amount is not None and kind == "unknown":
        amount = None
    endpoint = _endpoint(event["endpoint"])
    round_index = context.get("round_index")
    if type(round_index) is not int or round_index < 0 or event["round_span"]:
        round_index = None
    payload = {
        "cumulative": event["cumulative"],
        "measurement": event["measurement"],
        "endpoint": endpoint,
        "effort": event["effort"],
        "pricing_sha256": None,
        "scope_fingerprint": _scope_fingerprint(directory),
        "note": safe_error_detail(event["note"]) if event["note"] else None,
        "admission_id": context.get("admission_id"),
        "dispatch_identity": context.get("dispatch_identity"),
        "dispatch_started_at": context.get("dispatch_started_at"),
    }
    with _writing_database(directory) as db, db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            "SELECT payload FROM events WHERE id=?", (event["event_id"],)
        ).fetchone()
        if event["status"] == CODEX_COMPACTION_USAGE_GAP:
            # A modern compaction item has a stable identity. Preserve its first
            # observed round when start/completion/Resume repeat the same item.
            # Legacy notifications lack item IDs; one turn-level gap covers
            # them without double-counting a modern and legacy notification.
            legacy_id = f'{event["source_id"]}:legacy'
            gaps = db.execute(
                "SELECT id FROM events WHERE source_id=? AND status=?",
                (event["source_id"], CODEX_COMPACTION_USAGE_GAP),
            ).fetchall()
            if existing or any(row[0] == legacy_id for row in gaps):
                return
            if event["event_id"] == legacy_id and gaps:
                return
        if existing and event["cumulative"]:
            return
        if existing:
            old = json.loads(existing[0])
            if old.get("cumulative"):
                raise ValueError("Cannot rewrite a cumulative observation")
            # Incomplete late stream events cannot erase previously parsed measurements.
            tokens = {**old.get("tokens", {}), **tokens}
            if amount is None:
                amount, kind = old.get("cost_usd"), old.get("cost_kind", "unknown")
            if credits is None:
                credits = old.get("credits")
            payload["pricing_sha256"] = old.get("pricing_sha256")
            payload["pricing_rate"] = old.get("pricing_rate")
            payload["scope_fingerprint"] = old.get("scope_fingerprint")
            if _number(event["cost_usd"]) is not None and event["cost_kind"] == "reported":
                payload["pricing_rate"] = payload["pricing_sha256"] = None
        unattributed = False
        if event["cumulative"]:
            if not event["source_id"]:
                raise ValueError("Cumulative usage requires a source ID")
            key = json.dumps([event["provider"], event["source_id"], event["measurement"]])
            delta, unattributed = _cumulative_delta(
                db,
                key,
                {**tokens, "cost_usd": amount,
                 "credits": None if event["credit_source_id"] else credits},
                baseline=event["baseline"],
                baseline_unknown=event["baseline_unknown"],
                reset=event["reset"],
            )
            tokens = {key: value for key, value in delta.items() if key in TOKEN_FIELDS}
            amount = delta.get("cost_usd")
            if event["credit_source_id"]:
                credit_key = json.dumps([event["provider"], event["credit_source_id"], "credits"])
                credit_delta, credit_unknown = _cumulative_delta(
                    db, credit_key, {"credits": credits}, baseline=event["baseline"],
                    baseline_unknown=event["baseline_unknown"], reset=event["reset"],
                )
                credits = credit_delta.get("credits")
                unattributed |= credit_unknown
            else:
                credits = delta.get("credits")
        if (
            event["measurement"] == "combined"
            and not event["baseline"]
            and (amount is None or payload.get("pricing_rate"))
        ):
            amount, rate = _price(
                directory,
                event["provider"],
                event["model"],
                endpoint,
                tokens,
                rate=payload.get("pricing_rate"),
            )
            if amount is not None:
                kind, payload["pricing_rate"] = "estimated", rate
                payload["pricing_sha256"] = hashlib.sha256(
                    json.dumps(rate, sort_keys=True).encode()
                ).hexdigest()
        if amount is None:
            kind = "unknown"
        payload.update(
            tokens=tokens,
            cost_usd=amount,
            cost_kind=kind,
            credits=credits,
            baseline=event["baseline"],
            baseline_unknown=unattributed,
        )
        missing_usage = event["measurement"] != "cost" and not all(
            key in tokens for key in ("input_tokens", "output_tokens")
        )
        missing_cost = event["measurement"] != "usage" and amount is None
        if event["baseline"]:
            missing_usage = missing_cost = False
        if unattributed:
            missing_usage |= event["measurement"] != "cost"
            missing_cost |= event["measurement"] != "usage"
        round_unattributed = bool(event["round_span"]) or (
            context.get("phase") == "discovery" and round_index is None
        )
        fields = [
            event["event_id"],
            event["provider"],
            event["model"],
            context.get("role", "unknown"),
            context.get("phase", "unknown"),
            round_index,
            json.dumps(event["round_span"]),
            event["source_id"],
            event["measurement"],
            event["status"],
            _now(),
            *(tokens.get(key) for key in TOKEN_FIELDS),
            amount,
            kind,
            credits,
            int(missing_usage),
            int(missing_cost),
            int(unattributed or round_unattributed),
            json.dumps(payload),
        ]
        db.execute("DELETE FROM events WHERE id=?", (event["event_id"],))
        db.execute(f"INSERT INTO events VALUES ({','.join('?' for _ in fields)})", fields)
    receipt_path = directory / "usage_initialization.json"
    receipt = _accounting_json(receipt_path)
    if receipt and not event["baseline"] and not receipt.get("observations_recorded"):
        # The usage transaction must be visible before invalidating pristine zero;
        # a concurrent admission must not see an empty ledger plus a spent receipt.
        atomic_write_json(receipt_path, {**receipt, "observations_recorded": True})


def _aggregate(rows: list[dict]) -> dict:
    def total(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return sum(values) if values else None

    known = total("cost_usd")
    missing_cost = sum(row["missing_cost"] for row in rows)
    calls = sum(row["calls"] for row in rows)
    return {
        "calls": calls,
        "observations": calls,
        "missing_usage_calls": sum(row["missing_usage"] for row in rows),
        "missing_cost_calls": missing_cost,
        "compaction_usage_gaps": sum(row.get("compaction_usage_gaps", 0) for row in rows),
        "unattributed_observations": sum(row["unattributed"] for row in rows),
        "tokens": {key: total(key) for key in TOKEN_FIELDS},
        "known_cost_usd": known,
        "reported_cost_usd": total("reported_cost_usd"),
        "estimated_cost_usd": total("estimated_cost_usd"),
        "credits": total("credits"),
        "cost_complete": bool(calls and known is not None and not missing_cost),
    }


def _projection(
    rounds: list[dict],
    totals: dict,
    completed: int | None,
    planned: int | None,
    *,
    settings_consistent: bool = True,
    basis_rounds: list[int] | None = None,
) -> dict:
    result = {
        "available": False,
        "method": "mean observed completed round cost",
        "reason": "At least two completely accounted completed rounds are required",
        "completed_rounds": completed,
        "remaining_rounds": None,
        "basis_rounds": [],
        "projected_total_cost_usd": None,
        "projected_remaining_cost_usd": None,
        "includes_finalization": False,
    }
    if type(completed) is not int or type(planned) is not int or min(completed, planned) < 0:
        result["reason"] = "No declared native round plan"
        return result
    remaining = max(0, planned - completed)
    result["remaining_rounds"] = remaining
    if remaining == 0:
        result["reason"] = "No declared discovery rounds remain; finalization costs are separate"
        return result
    if not settings_consistent:
        result["reason"] = "Models, effort, pricing or analysis scope changed; collect a new basis"
        return result
    eligible = set(range(completed) if basis_rounds is None else basis_rounds)
    basis = sorted(
        (row for row in rounds if row["round_index"] in eligible),
        key=lambda row: row["round_index"],
    )
    if (
        len(basis) < 2
        or [row["round_index"] for row in basis]
        != list(range(completed - len(basis), completed))
        or not totals["cost_complete"]
        or any(not row["cost_complete"] for row in basis)
        or any(row["unattributed_observations"] for row in rounds)
    ):
        return result
    costs = [row["known_cost_usd"] for row in basis]
    active = sum(row["known_cost_usd"] or 0 for row in rounds if row["round_index"] == completed)
    future = max(0, sum(costs) / len(costs) - active) + (remaining - 1) * sum(costs) / len(costs)
    result.update(
        available=True,
        reason=None,
        basis_rounds=[row["round_index"] for row in basis],
        projected_remaining_cost_usd=future,
        projected_total_cost_usd=(totals["known_cost_usd"] or 0) + future,
        assumptions="Same models, effort, cohort and work per round; future finalization "
        "and delivery are excluded. Recompute after settings or scope changes.",
    )
    return result


def usage_summary(  # noqa: C901
    run_dir: str | Path, *, planned_rounds: int | None = None, completed_rounds: int | None = None
) -> dict:
    """Read committed measurements without inference, profile creation or database writes."""
    directory = Path(run_dir)
    config = _accounting_json(directory / "config.json")
    planned = config.get("max_rounds") if planned_rounds is None else planned_rounds
    completed = config.get("round") if completed_rounds is None else completed_rounds
    rows, notices = [], []
    path = directory / "usage.sqlite3"
    if path.is_file():
        columns = ", ".join(f"SUM({key}) AS {key}" for key in TOKEN_FIELDS)
        try:
            rows = [
                dict(row)
                for row in _read_ledger(path, f"""SELECT provider, model, role, phase,
                    json_extract(payload,'$.scope_fingerprint') AS scope_fingerprint,
                    json_extract(payload,'$.effort') AS effort,
                    json_extract(payload,'$.endpoint') AS endpoint,
                    round_index, round_span, COUNT(*) AS calls, {columns},
                    SUM(cost_usd) AS cost_usd,
                    SUM(CASE WHEN cost_kind='reported' THEN cost_usd END) AS reported_cost_usd,
                    SUM(CASE WHEN cost_kind='estimated' THEN cost_usd END) AS estimated_cost_usd,
                    SUM(credits) AS credits, SUM(missing_usage) AS missing_usage,
                    SUM(missing_cost) AS missing_cost, SUM(unattributed) AS unattributed,
                    SUM(CASE WHEN status=? THEN 1 ELSE 0 END) AS compaction_usage_gaps
                    FROM events WHERE json_extract(payload,'$.baseline') != 1
                    GROUP BY provider, model, role, phase, round_index, round_span,
                             scope_fingerprint, effort, endpoint""",
                    (CODEX_COMPACTION_USAGE_GAP,)) or []
            ]
        except sqlite3.Error:
            notices.append("Saved usage could not be read; totals are unavailable, not zero.")
    totals = _aggregate(rows)
    if not rows:
        notices.append("No execution usage recorded. Older or uninstrumented work is unknown.")
    if totals["missing_cost_calls"] or totals["missing_usage_calls"]:
        notices.append("Some observations lack usage or cost; known totals are partial.")
    if totals["compaction_usage_gaps"]:
        notices.append(
            "Codex context compaction did not expose separately attributable token usage. "
            "Known token totals may omit compaction work; later ordinary token snapshots "
            "do not resolve that gap. Cumulative dollar and credit measurements are retained "
            "independently, without inferred compaction charges.")
    if totals["missing_cost_calls"] and totals["known_cost_usd"] == 0:
        notices.append(
            "A zero known USD subtotal does not mean the run was free: some costs are "
            "unavailable. Credits, when present, are separate from dollars.")
    if any(row["provider"] == "openai_compatible" for row in rows):
        notices.append(
            "API totals cover exposed usage. Provider/SDK internal retries can hide "
            "failed-attempt usage, so fully accounted observations may still omit spend.")
    if totals["unattributed_observations"]:
        notices.append(
            "Some counters lack a trusted baseline or span multiple rounds. "
            "Known run totals are retained; exact round attribution is unavailable."
        )
    recorded_rounds = {row["round_index"] for row in rows if row["round_index"] is not None}
    for row in rows:
        span = json.loads(row["round_span"]) if row.get("round_span") else None
        if (isinstance(span, list) and len(span) == 2
                and all(type(value) is int and value >= 0 for value in span)
                and span[0] <= span[1]):
            # Shared counters account for run spend without inventing a per-round split.
            recorded_rounds.update(range(span[0], span[1] + 1))
    if type(completed) is int and completed > 0 and set(range(completed)) - recorded_rounds:
        totals["cost_complete"] = False
        notices.append(
            "Some completed rounds have no attributed observations; historical "
            "usage or per-round costs are incomplete."
        )
    if (directory / "usage_gaps.jsonl").exists():
        totals["cost_complete"] = False
        notices.append("Accounting write gaps were recorded; totals may be incomplete.")
    notices.append(
        "Dollar figures are reported usage or estimates, not a billing statement. "
        "Missing amounts are unknown, not zero; token cache/reasoning subsets "
        "must not be added to their parent totals."
    )

    def grouped(keys):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        order = (lambda item: (item[0][0] is None, item[0][0] or 0)) if keys == [
            "round_index"] else (lambda item: str(item[0]))
        return [
            {**dict(zip(keys, key, strict=True)), **_aggregate(value)}
            for key, value in sorted(groups.items(), key=order)
        ]

    rounds = grouped(["round_index"])
    signatures = defaultdict(set)
    current_scope = _scope_fingerprint(directory)
    for row in rows:
        if row["round_index"] is not None:
            signatures[row["round_index"]].add(
                tuple(
                    row.get(key)
                    for key in (
                        "provider",
                        "model",
                        "role",
                        "effort",
                        "endpoint",
                        "scope_fingerprint",
                    )
                )
            )
    # A settings change invalidates the old basis, not all future projections.
    # Use a contiguous suffix ending at the most recent completed round, with
    # the same actual model/effort mix and the current saved configuration.
    reference = signatures.get(completed - 1, set()) if type(completed) is int else set()
    reference_is_current = bool(reference) and all(
        signature[-1] == current_scope for signature in reference
    )
    basis_rounds = []
    if reference_is_current:
        for index in range(completed - 1, -1, -1):
            if signatures.get(index) != reference:
                break
            basis_rounds.append(index)
        basis_rounds.reverse()
    # An active round can have dispatched only some of its usual roles so far.
    # A newly observed model/effort, however, must invalidate the projection now,
    # including runtime fallbacks that do not change the persisted config.
    active_signature = signatures.get(completed, set()) if type(completed) is int else set()
    settings_consistent = not reference or (
        reference_is_current and active_signature <= reference
    )
    for row in rounds:
        row["round_label"] = (
            f"Round {row['round_index'] + 1}"
            if row["round_index"] is not None
            else "Outside a round / attribution unavailable"
        )
    from autofa.session.usage_estimates import (
        budget_exposure,
        horizon_projection,
        monetary_view,
        observed_token_cost_estimate,
        resolved_cost_gaps,
    )

    derived = derived_usage_estimate(directory)
    derived["reconciled_observations"] = (resolved_cost_gaps(directory)
                                          if totals["missing_cost_calls"] else 0)
    if derived.get("pricing_error"):
        notices.append(derived["note"])
    money = monetary_view(totals, derived)
    if ((directory / "usage_gaps.jsonl").exists()
            or (type(completed) is int and completed > 0
                and set(range(completed)) - recorded_rounds)):
        money["cost_complete"] = False
    if not rows and completed in {None, 0}:
        initial = admission_usage(directory)
        if initial.get("available") and initial["monetary_view"]["cost_complete"]:
            totals.update(known_cost_usd=0.0, cost_complete=True)
            money.update(recorded_subtotal_usd=0.0, known_and_derived_subtotal_usd=0.0,
                         cost_complete=True)
            notices = [note for note in notices if not note.startswith("No execution usage")]
    money = budget_exposure(directory, {"monetary_view": money})["monetary_view"]
    projection = _projection(rounds, totals, completed, planned,
                             settings_consistent=settings_consistent, basis_rounds=basis_rounds)
    if config.get("campaign_policy"):
        projection = horizon_projection(directory, config, view=money, totals=totals,
                                        derived=derived, basis_rounds=basis_rounds,
                                        settings_consistent=settings_consistent)
    return {
        "schema_version": 1,
        "available": bool(rows) or money["cost_complete"],
        "observed_at": _now(),
        "totals": totals,
        "by_round": rounds,
        "by_role": grouped(["role"]),
        "by_model": grouped(["provider", "model"]),
        "by_phase": grouped(["phase"]),
        "derived_estimate": derived,
        "token_cost_estimate": observed_token_cost_estimate(directory),
        "monetary_view": money,
        "projection": projection,
        "notices": notices,
    }


def snapshot_usage(run_dir: Path) -> dict[str, Any]:
    """Best-effort read of saved accounting; missing records never imply free work."""
    try:
        return usage_summary(run_dir)
    except Exception:  # noqa: BLE001 — saved status should survive missing accounting
        return {
            "schema_version": 1,
            "available": False,
            "notices": ["Saved AutoFA usage could not be read; cost is unknown."],
        }
