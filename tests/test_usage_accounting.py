# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable usage accounting, pricing, projections and lifecycle attribution for analysis runs.

Observed-token reference prices never rewrite bills or grant spending authority.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from contextvars import copy_context
from copy import deepcopy
from datetime import UTC, datetime
from io import StringIO
from types import SimpleNamespace

import pytest
from openai import OpenAI
from rich.console import Console

from autofa import supervise_cli
from autofa.config import Config, RunConfig
from autofa.llm.hub import HubLLM, chat_completions_safe
from autofa.mcp import runtime
from autofa.preflight.cheating import pipeline
from autofa.preflight.cheating.report import write_report
from autofa.session import finalization, round_validation, supervision, usage, usage_estimates
from autofa.session.recovery import run_lock
from autofa.session.usage_estimates import observed_token_cost_estimate
from autofa.taxonomy import report as taxonomy_report
from autofa.taxonomy.record import RECORD_KIND
from autofa.util.run_copy import copy_run_artifacts
from autofa.util.work_queue import ordered_bounded_map

try:
    import httpx
except ImportError:
    import httpx2 as httpx

GATEWAY = "https://gateway.invalid/v1"


@pytest.fixture
def run(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"run_id": "test", "round": 0,
                                                     "max_rounds": 4}))
    return tmp_path


def configure(run, **changes):
    path = run / "config.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


def record(run, event="one", *, round_index=0, phase="discovery", role="discovery",
           dispatch_started_at=None, provider="test", model="model", **kwargs):
    usage.record_usage({"run_dir": str(run), "round_index": round_index, "phase": phase,
                        "role": role, "dispatch_started_at": dispatch_started_at},
                       event_id=event, provider=provider, model=model, **kwargs)


def gateway_record(run, event="one", **overrides):
    """A detector call through an explicit deployment, observing 1M input/100k output tokens."""
    record(run, event, role="detector", provider="openai_compatible", model="deployment",
           endpoint=GATEWAY, **{"usage": {"input_tokens": 1_000_000, "output_tokens": 100_000},
                                **overrides})


def round_cost(run, index, cost=2, *, model="model", effort="medium", role="discovery"):
    record(run, f"{index}-{model}-{effort}-{role}", round_index=index, role=role, model=model,
           effort=effort, usage={"input_tokens": 10, "output_tokens": 2}, cost_usd=cost,
           cost_kind="reported")


def counters(value):
    return {"input_tokens": value, "output_tokens": value // 10,
            "total_tokens": value + value // 10}


def rate_card(**rates):
    return {"version": 1, "currency": "USD", "rates": [{
        "provider": "test", "model": "model", "input_per_million": 2.0,
        "output_per_million": 4.0, "source": "test rate card", "effective_at": "2026-11-03",
        **rates,
    }]}


def gateway_card(**rates):
    return rate_card(**{"provider": "openai_compatible", "model": "deployment",
                        "endpoint": GATEWAY, **rates})


def totals(run, **kwargs):
    return usage.usage_summary(run, **kwargs)["totals"]


# Run attribution and read-only access


def test_readonly_unknown_run_does_not_create_profile_or_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "absent"))
    before = list(tmp_path.iterdir())
    result = usage.usage_summary(tmp_path / "missing")
    assert not result["available"]
    assert result["totals"]["known_cost_usd"] is None
    assert result["totals"]["tokens"]["input_tokens"] is None
    assert list(tmp_path.iterdir()) == before


def test_stale_runtime_cannot_create_run_in_another_profile(tmp_path, monkeypatch):
    profile = tmp_path / "absent_profile"
    monkeypatch.setenv("AUTOFA_HOME", str(profile))
    run = RunConfig(run_id="previous", model_id="m", parquet_paths=[])
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(run=run))
    assert usage.capture_context() is None
    assert not profile.exists()


def test_context_attribution_failure_marks_existing_run_gap(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    directory = tmp_path / "runs" / "run"
    directory.parent.mkdir()
    target = tmp_path / "external_run"
    target.mkdir()
    directory.symlink_to(target, target_is_directory=True)
    (directory / "config.json").write_text(json.dumps({"run_id": "run", "round": 0}))
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(
        run=RunConfig(run_id="run", model_id="m", parquet_paths=[])))

    def unavailable(_):
        raise ValueError("Synthetic attribution failure")

    monkeypatch.setattr(supervision, "resolve_run_dir", unavailable)
    assert usage.capture_context() is None
    gap = json.loads((directory / "usage_gaps.jsonl").read_text())
    assert gap["stage"] == "context attribution"
    assert not totals(directory)["cost_complete"]


@pytest.mark.parametrize("single_run", [False, True])
def test_usage_observer_follows_symlinked_runs_without_losing_api_usage(
    tmp_path, monkeypatch, single_run,
):
    profile, target = tmp_path / "profile", tmp_path / "storage"
    profile.mkdir()
    target.mkdir()
    directory = target / "run"
    directory.mkdir()
    if single_run:
        # Native resume permits runs/<id> itself to point at another disk.
        (profile / "runs").mkdir()
        (profile / "runs" / "run").symlink_to(directory, target_is_directory=True)
    else:
        (profile / "runs").symlink_to(target, target_is_directory=True)
    (directory / "config.json").write_text(json.dumps({"run_id": "run", "round": 0}))
    monkeypatch.setenv("AUTOFA_HOME", str(profile))
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(
        run=RunConfig(run_id="run", model_id="m", parquet_paths=[])))
    context = usage.capture_context(role="detector")
    assert context is not None
    usage.record_usage(context, event_id="api", provider="test", model="m", cost_usd=1,
                       cost_kind="reported", usage=counters(10))
    assert totals(directory)["known_cost_usd"] == 1


def test_readonly_closed_wal_ledger_does_not_create_sidecars(run):
    record(run, source_id="route:reported", cost_usd=1, cost_kind="reported",
           measurement="cost")
    names = {path.name for path in run.iterdir()}
    assert "usage.sqlite3-wal" not in names
    run.chmod(0o555)
    try:
        summary = usage.usage_summary(run)
        kind = usage.source_cost_kind({"run_dir": str(run)}, "route")
        assert summary["totals"]["known_cost_usd"] == 1
        assert kind == "reported"
        assert {path.name for path in run.iterdir()} == names
    finally:
        run.chmod(0o755)


def test_readonly_usage_includes_uncheckpointed_wal(run):
    with usage._connect(run) as keeper:
        keeper.execute("PRAGMA wal_autocheckpoint=0")
        record(run, cost_usd=2, cost_kind="reported", usage=counters(20))
        before = {path.name for path in run.iterdir()}
        assert (run / "usage.sqlite3-wal").stat().st_size > 0
        assert totals(run)["known_cost_usd"] == 2
        assert {path.name for path in run.iterdir()} == before
    keeper.close()


def test_writable_closed_ledger_uses_sqlite_locks(run, monkeypatch):
    record(run, cost_usd=2, cost_kind="reported")
    assert not (run / "usage.sqlite3-wal").exists()
    connect = sqlite3.connect
    opened = []

    def observed_connect(database, *args, **kwargs):
        opened.append(str(database))
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    assert totals(run)["known_cost_usd"] == 2
    assert opened and all("immutable=1" not in uri for uri in opened)


def test_usage_read_retries_transient_database_error_once(run, monkeypatch):
    record(run, source_id="route:reported", cost_usd=1, cost_kind="reported",
           measurement="cost")
    connect = sqlite3.connect
    attempts = []

    def interrupted_read(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise sqlite3.DatabaseError("Synthetic concurrent ledger replacement")
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", interrupted_read)
    # Every ledger reader shares this single fresh-snapshot retry.
    assert usage.source_cost_kind({"run_dir": str(run)}, "route") == "reported"
    assert len(attempts) == 2


# Ledger writes


def test_live_usage_reads_survive_wal_open_close_cycles(run):
    record(run, event="initial", cost_usd=1, cost_kind="reported")
    start = threading.Barrier(2)

    def writer():
        start.wait()
        for index in range(40):
            record(run, event=f"live-{index}", cost_usd=1, cost_kind="reported")

    with ThreadPoolExecutor(max_workers=1) as executor:
        writing = executor.submit(writer)
        start.wait()
        seen = []
        for _ in range(40):
            summary = usage.usage_summary(run)
            assert not any("could not be read" in notice for notice in summary["notices"])
            seen.append(summary["totals"]["known_cost_usd"])
        writing.result()
    assert seen == sorted(seen)
    assert totals(run)["known_cost_usd"] == 41


def test_usage_scope_commits_each_event_and_closes_its_writer_on_error(run, monkeypatch):
    connect = usage._connect
    opened = []

    def counted_connect(directory):
        opened.append(connect(directory))
        return opened[-1]

    monkeypatch.setattr(usage, "_connect", counted_connect)
    with pytest.raises(RuntimeError, match="Synthetic interrupted tool"):
        with usage.usage_connection_scope():
            for index in range(3):
                record(run, event=f"batch-{index}", cost_usd=1, cost_kind="reported")
                # An independent reader sees every commit before the scope closes.
                assert totals(run)["known_cost_usd"] == index + 1
            assert (run / "usage.sqlite3-wal").exists()
            raise RuntimeError("Synthetic interrupted tool")
    assert not (run / "usage.sqlite3-wal").exists()
    assert totals(run)["known_cost_usd"] == 3
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_usage_scope_serializes_worker_writes_and_supports_late_context(run):
    with usage.usage_connection_scope():
        late_context = copy_context()
        with ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [executor.submit(copy_context().run, record, run, f"worker-{index}",
                                     cost_usd=1, cost_kind="reported")
                     for index in range(48)]
            for task in tasks:
                task.result()
        assert totals(run)["known_cost_usd"] == 48
    # A copied context that outlives cancellation must not reuse the closed pool.
    late_context.run(record, run, "late", cost_usd=1, cost_kind="reported")
    assert not (run / "usage.sqlite3-wal").exists()
    assert totals(run)["known_cost_usd"] == 49
    assert not (run / "usage_gaps.jsonl").exists()


def test_checkpoint_copies_committed_usage_from_active_connection_scope(run, tmp_path):
    checkpoint = tmp_path.parent / f"{tmp_path.name}-checkpoint"
    with usage.usage_connection_scope():
        record(run, cost_usd=1, cost_kind="reported")
        copy_run_artifacts(run, checkpoint)
        record(run, "later", cost_usd=2, cost_kind="reported")
    assert totals(checkpoint)["known_cost_usd"] == 1
    assert totals(run)["known_cost_usd"] == 3


def test_parallel_writers_and_replayed_requests_are_idempotent(run):
    def write(index):
        record(run, f"call{index % 40}", usage=counters(10), cost_usd=.25, cost_kind="reported")

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(write, range(120)))
    result = totals(run)
    assert result["calls"] == 40 and result["known_cost_usd"] == 10
    assert result["tokens"]["input_tokens"] == 400
    assert not (run / "usage_gaps.jsonl").exists()


def test_database_initialization_retries_busy_without_repeating_inference(run, monkeypatch):
    original = usage._open_database
    attempts = []

    def contested(directory):
        attempts.append(directory)
        if len(attempts) == 1:
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        return original(directory)

    monkeypatch.setattr(usage, "_open_database", contested)
    record(run, cost_usd=1, cost_kind="reported")
    assert len(attempts) == 2
    assert totals(run)["known_cost_usd"] == 1
    assert not (run / "usage_gaps.jsonl").exists()


def test_ledger_failure_does_not_fail_model_work_and_marks_partial(run, monkeypatch):
    def fail(*args):
        raise sqlite3.OperationalError("private detail")

    monkeypatch.setattr(usage, "_connect", fail)
    record(run, cost_usd=3, cost_kind="reported")
    assert not totals(run)["cost_complete"]
    assert "private detail" not in (run / "usage_gaps.jsonl").read_text()


# Counter semantics


def test_zero_is_measured_but_missing_is_unknown(run):
    record(run, cost_usd=0, cost_kind="reported", usage=counters(0))
    assert totals(run)["cost_complete"] and totals(run)["known_cost_usd"] == 0
    record(run, "missing")
    result = totals(run)
    assert not result["cost_complete"]
    assert result["known_cost_usd"] == 0 and result["missing_cost_calls"] == 1


def test_cumulative_resume_idempotency_and_failed_zero_cannot_erase_spend(run):
    options = {"source_id": "session", "cumulative": True, "cost_kind": "estimated"}
    record(run, "start", usage=counters(100), cost_usd=1, baseline=True, **options)
    record(run, "done", usage=counters(200), cost_usd=2, **options)
    record(run, "done", usage=counters(200), cost_usd=2, **options)
    record(run, "failed", usage=counters(0), cost_usd=0, status="failed", **options)
    record(run, "resumed", usage=counters(350), cost_usd=3.5, baseline_unknown=True, **options)
    result = totals(run)
    assert result["known_cost_usd"] == 2.5
    assert result["tokens"]["input_tokens"] == 250
    assert result["calls"] == 3


LEGACY = {"source_id": "legacy", "cumulative": True, "cost_kind": "estimated",
          "baseline_unknown": True}


def test_unseen_resume_excludes_unknown_history_then_counts_new_work(run):
    record(run, "import", usage=counters(100), cost_usd=20, **LEGACY)
    assert totals(run)["known_cost_usd"] is None
    record(run, "next", usage=counters(150), cost_usd=21, **LEGACY)
    result = totals(run)
    assert result["known_cost_usd"] == 1
    assert result["tokens"]["input_tokens"] == 50
    assert not result["cost_complete"]
    assert result["unattributed_observations"] == 1


def test_resumed_counters_baseline_each_newly_available_field(run):
    # A missing resume snapshot cannot make the next snapshot charge history.
    record(run, "missing", **LEGACY)
    record(run, "tokens_only", usage=counters(150), **LEGACY)
    record(run, "first_dollars", usage=counters(200), cost_usd=21, **LEGACY)
    result = totals(run)
    assert result["known_cost_usd"] is None
    assert result["tokens"]["input_tokens"] == 50
    record(run, "next_dollars", usage=counters(250), cost_usd=22, **LEGACY)
    result = totals(run)
    assert result["known_cost_usd"] == 1
    assert result["tokens"]["input_tokens"] == 100


def test_explicit_reset_starts_new_counter_epoch(run):
    options = {"source_id": "session", "cumulative": True, "cost_kind": "reported"}
    record(run, "old", usage=counters(200), cost_usd=2, **options)
    record(run, "reset", usage=counters(100), cost_usd=1, reset=True, **options)
    assert totals(run)["known_cost_usd"] == 3


def test_duplicate_baseline_preserves_unreported_interrupted_work(run):
    options = {"source_id": "session", "cumulative": True, "cost_kind": "reported"}
    record(run, "old", usage=counters(100), cost_usd=1, **options)
    record(run, "resume_start", usage=counters(200), cost_usd=2, baseline=True, **options)
    record(run, "next", usage=counters(300), cost_usd=3, **options)
    result = totals(run)
    assert result["known_cost_usd"] == 3
    assert result["unattributed_observations"] == 1


def test_additive_stream_enrichment_does_not_duplicate_or_erase(run):
    record(run, status="incomplete")
    record(run, usage=counters(100), cost_usd=1, cost_kind="reported")
    record(run, status="interrupted")
    result = totals(run)
    assert result["calls"] == 1 and result["cost_complete"]
    assert result["known_cost_usd"] == 1 and result["tokens"]["input_tokens"] == 100


# Pricing


def test_explicit_rates_account_for_cache_and_preserve_provenance(run):
    card = rate_card(cached_input_per_million=.5)
    usage.set_usage_pricing(run, card)
    tokens = {**counters(1_000_000), "cache_read_input_tokens": 400_000}
    record(run, usage=tokens)
    expected = .6 * 2 + .4 * .5 + .1 * 4
    assert totals(run)["estimated_cost_usd"] == pytest.approx(expected)
    usage.set_usage_pricing(run, rate_card(input_per_million=9))
    record(run, usage={**tokens, "output_tokens": 200_000})
    assert totals(run)["estimated_cost_usd"] == pytest.approx(expected + .4)
    with sqlite3.connect(run / "usage.sqlite3") as db:
        payload = json.loads(db.execute("SELECT payload FROM events").fetchone()[0])
    assert payload["pricing_rate"] == card["rates"][0]
    assert payload["pricing_sha256"]


def test_unknown_discounted_cache_cannot_be_assumed_zero(run):
    usage.set_usage_pricing(run, rate_card(cached_input_per_million=.5))
    record(run, usage=counters(100))
    assert totals(run)["known_cost_usd"] is None


def test_separate_native_tokens_and_costs_cannot_double_price(run):
    usage.set_usage_pricing(run, rate_card())
    record(run, "tokens", usage=counters(1_000_000), measurement="usage")
    record(run, "dollars", cost_usd=3, cost_kind="estimated", measurement="cost")
    assert totals(run)["known_cost_usd"] == 3


def test_reported_amount_replaces_late_stream_rate_estimate(run):
    usage.set_usage_pricing(run, rate_card())
    record(run, usage=counters(1_000_000))
    record(run, usage=counters(1_000_000), cost_usd=5, cost_kind="reported")
    result = totals(run)
    assert result["known_cost_usd"] == 5 and result["estimated_cost_usd"] is None


@pytest.mark.parametrize("updates", [
    {"input_per_million": float("nan")}, {"endpoint": "https://user:secret@example.com/api"},
    {"secret": "hidden"},
])
def test_invalid_rate_cards_do_not_write(run, updates):
    with pytest.raises(ValueError):
        usage.set_usage_pricing(run, rate_card(**updates))
    assert not (run / "usage_pricing.json").exists()


def test_duplicate_pricing_identity_rejected():
    card = rate_card()
    card["rates"].append(deepcopy(card["rates"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        usage.validate_pricing(card)


def test_researched_harness_reference_rates_do_not_require_a_fictitious_endpoint():
    card = rate_card(provider="codex", basis="proxy", source="https://provider.invalid/prices",
                     retrieved_at="2026-11-05T14:00:00Z", assumptions="Token reference, not a bill")
    assert usage.validate_pricing(card) == card
    card["rates"][0]["provider"] = "openai_compatible"
    with pytest.raises(ValueError, match="endpoint"):
        usage.validate_pricing(card)


@pytest.mark.parametrize("updates", [
    {"basis": "public"},
    {"basis": "public", "retrieved_at": "2026-11-04"},
    {"basis": "proxy", "retrieved_at": "2026-11-04", "source": "https://provider.invalid/pricing"},
])
def test_researched_rates_require_endpoint_provenance_and_proxy_assumptions(updates):
    with pytest.raises(ValueError):
        usage.validate_pricing(gateway_card(**updates))


def test_public_rate_snapshot_retains_qualified_source(run):
    value = gateway_card(basis="proxy", source="https://provider.invalid/pricing",
                         retrieved_at="2026-11-04T10:00:00Z",
                         assumptions="Public list price equivalent",
                         billing_provider="provider", pricing_model="public-model")
    usage.initialize_run_accounting(run, value)
    gateway_record(run)
    rows = usage._read_rows(run / "usage.sqlite3", "SELECT payload FROM events")
    payload = json.loads(rows[0]["payload"])
    assert payload["pricing_rate"] == value["rates"][0]
    assert payload["cost_kind"] == "estimated"
    with pytest.raises(ValueError, match="USD"):
        usage.validate_pricing({**value, "currency": "EUR"})


def test_live_run_pricing_edit_respects_native_lock_but_reference_pricing_does_not(run):
    record(run, usage=counters(1_000_000))
    before = (run / "usage.sqlite3").read_bytes()
    admission = usage.admission_usage(run)
    with run_lock(run):
        with pytest.raises(RuntimeError, match="already active"):
            usage.set_usage_pricing(run, rate_card())
        result = usage.set_usage_pricing(run, rate_card(), estimate_only=True)
    assert result["scope"] == "observed-token reference estimate only"
    assert result["affects_spend_admission"] is False
    assert json.loads((run / "usage_estimate_pricing.json").read_text()) == rate_card()
    assert not (run / "usage_pricing.json").exists()
    assert (run / "usage.sqlite3").read_bytes() == before
    assert usage.admission_usage(run) == admission


def test_startup_prices_first_dispatch_and_resume_preserves_snapshot(run):
    assert usage.initialize_run_accounting(run, gateway_card())["initialized"]
    gateway_record(run)
    summary = usage.usage_summary(run)
    assert summary["totals"]["estimated_cost_usd"] == pytest.approx(2.4)
    resumed = usage.initialize_run_accounting(run, gateway_card(input_per_million=100))
    assert resumed["source"] == "saved_run"
    assert resumed["pricing"] == gateway_card()
    assert not summary["derived_estimate"]["available"]


def test_historical_estimate_does_not_rewrite_ledger_or_reported_cost(run):
    gateway_record(run, "unpriced")
    gateway_record(run, "reported", cost_usd=7, cost_kind="reported")
    gateway_record(run, "credits_only", measurement="cost", usage=None, credits=5)
    before = (run / "usage.sqlite3").read_bytes()
    estimate = usage.derived_usage_estimate(run, gateway_card())
    assert estimate["derived_cost_usd"] == pytest.approx(2.4)
    assert estimate["estimated_observations"] == 1
    assert estimate["ledger_modified"] is False
    assert (run / "usage.sqlite3").read_bytes() == before
    usage.initialize_run_accounting(run, gateway_card())
    summary = usage.usage_summary(run)
    assert summary["totals"]["known_cost_usd"] == 7
    assert summary["monetary_view"]["known_and_derived_subtotal_usd"] == pytest.approx(9.4)
    assert summary["monetary_view"]["missing_cost_observations"] == 1
    assert not summary["monetary_view"]["cost_complete"]
    assert summary["totals"]["credits"] == 5


def test_derived_cache_pricing_does_not_hide_missing_counters(run):
    gateway_record(run, "measured", usage={"input_tokens": 1_000_000, "output_tokens": 100_000,
                                           "cache_read_input_tokens": 500_000,
                                           "reasoning_output_tokens": 50_000})
    gateway_record(run, "cache_unknown")
    estimate = usage.derived_usage_estimate(run, gateway_card(cached_input_per_million=.5))
    assert estimate["estimated_observations"] == estimate["unpriced_observations"] == 1
    assert estimate["derived_cost_usd"] == pytest.approx(1.65)
    other_endpoint = gateway_card(endpoint="https://different.invalid/v1")
    assert not usage.derived_usage_estimate(run, other_endpoint)["available"]


# Observed-token reference estimates


TOKENS = {"input_tokens": 1_000_000, "cache_read_input_tokens": 800_000,
          "output_tokens": 1_000, "reasoning_output_tokens": 500}


def token_card(provider="codex", model="test-model", **updates):
    return {"version": 1, "currency": "USD", "rates": [{
        "provider": provider, "model": model, "input_per_million": 2,
        "cached_input_per_million": .2, "output_per_million": 8,
        "source": "synthetic test rates", "effective_at": "2026-11-05", **updates,
    }]}


def write_token_card(run, value=None, name="usage_estimate_pricing.json"):
    (run / name).write_text(json.dumps(token_card() if value is None else value))


def token_record(run, event, *, role="judge", phase="validation", **updates):
    """A Codex token observation priced at $0.568 by ``token_card``."""
    record(run, event, role=role, phase=phase, **{
        "provider": "codex", "model": "test-model", "measurement": "usage",
        "source_id": "codex:thread:tokens", "usage": TOKENS, **updates,
    })


def test_codex_reference_estimate_ignores_native_dollars_and_never_changes_admission(run):
    token_record(run, "tokens", role="discovery", phase="discovery")
    during_turn = observed_token_cost_estimate(run, token_card())
    assert during_turn["complete"] and during_turn["groups"][0]["role"] == "discovery"
    assert usage.admission_usage(run)["totals"]["known_cost_usd"] is None
    # Codex reports zero account dollars and positive credits for the same work.
    token_record(run, "account", model=None, measurement="cost",
                 source_id="codex:thread:account", usage=None, cost_usd=0, cost_kind="reported",
                 credits=2)
    before = usage.admission_usage(run)
    ledger_before = (run / "usage.sqlite3").read_bytes()
    write_token_card(run)
    result = observed_token_cost_estimate(run)
    assert result["available"] and result["complete"]
    assert result["estimated_cost_usd"] == during_turn["estimated_cost_usd"]
    assert result["estimated_cost_usd"] == pytest.approx(.568)
    assert result["token_observations"] == result["priced_token_observations"] == 1
    assert result["groups"][0]["tokens"]["reasoning_output_tokens"] == 500
    assert result["ledger_modified"] is result["affects_spend_admission"] is False
    assert (run / "usage.sqlite3").read_bytes() == ledger_before
    after = usage.admission_usage(run)
    assert before["totals"] == after["totals"]
    assert before["monetary_view"] == after["monetary_view"]
    assert after["totals"]["known_cost_usd"] == 0
    assert after["totals"]["credits"] == 2


def test_reference_card_priority_does_not_change_execution_rates(run):
    execution = token_card(provider="openai_compatible", cached_input_per_million=2)
    usage.initialize_run_accounting(run, execution)
    token_record(run, "api", provider="openai_compatible", measurement="combined")
    original = (run / "usage_pricing.json").read_bytes()
    assert observed_token_cost_estimate(run)["estimated_cost_usd"] == pytest.approx(2.008)
    write_token_card(run, token_card(provider="openai_compatible"))
    result = observed_token_cost_estimate(run)
    assert result["estimated_cost_usd"] == pytest.approx(.568)
    assert result["pricing_source"] == "usage_estimate_pricing.json"
    override = token_card(provider="openai_compatible", cached_input_per_million=1)
    result = observed_token_cost_estimate(run, override)
    assert result["estimated_cost_usd"] == pytest.approx(1.208)
    assert result["pricing_source"] == "explicit_override"
    assert (run / "usage_pricing.json").read_bytes() == original
    assert usage.admission_usage(run)["totals"]["known_cost_usd"] == pytest.approx(2.008)


def test_mixed_cache_presence_and_invalid_counts_remain_separate(run):
    token_record(run, "valid")
    token_record(run, "cache_unknown", usage={"input_tokens": 1_000_000, "output_tokens": 1_000})
    token_record(run, "cache_invalid", usage={"input_tokens": 100, "output_tokens": 1,
                                              "cache_read_input_tokens": 101})
    result = observed_token_cost_estimate(run, token_card())
    assert result["estimated_cost_usd"] == pytest.approx(.568)
    assert result["priced_token_observations"] == 1
    assert result["unpriced_token_observations"] == 2
    assert not result["complete"]
    assert {row["reason"] for row in result["unpriced_groups"]} == {
        "missing_cache_counters", "invalid_cache_counters"}


def test_cumulative_resume_uses_saved_deltas_and_excludes_baseline(run):
    # Every Codex session start records a baseline; it must not make the estimate partial.
    token_record(run, "baseline", baseline=True, cumulative=True)
    token_record(run, "resumed", cumulative=True, usage={
        "input_tokens": 2_000_000, "cache_read_input_tokens": 1_600_000,
        "output_tokens": 2_000, "reasoning_output_tokens": 1_000,
    })
    result = observed_token_cost_estimate(run, token_card())
    assert result["complete"] and result["token_observations"] == 1
    assert result["estimated_cost_usd"] == pytest.approx(.568)
    assert result["groups"][0]["tokens"]["input_tokens"] == 1_000_000


def test_untrusted_resume_deltas_and_compaction_keep_reference_partial(run):
    token_record(run, "valid")
    token_record(run, "unknown", cumulative=True, baseline_unknown=True,
                 source_id="codex:resumed:tokens")
    token_record(run, "compaction", status=usage.CODEX_COMPACTION_USAGE_GAP, usage=None,
                 source_id="codex:thread:compaction")
    result = observed_token_cost_estimate(run, token_card())
    assert result["available"] and not result["complete"]
    assert result["estimated_cost_usd"] == pytest.approx(.568)
    assert result["unpriced_token_observations"] == result["missing_token_observations"] == 2
    assert {row["reason"] for row in result["reasons"]} == {
        "unknown_token_baseline", "compaction_usage_gap"}


@pytest.mark.parametrize("value,reason", [(None, "pricing_unavailable"),
                                         ({}, "pricing_invalid"),
                                         (token_card(model="different"), "missing_rate")])
def test_missing_pricing_is_explained_instead_of_zero_unpriced_observations(run, value, reason):
    token_record(run, "tokens")
    result = observed_token_cost_estimate(run, value)
    assert result["estimated_cost_usd"] is None and not result["complete"]
    assert result["unpriced_token_observations"] == 1
    assert result["reasons"] == [{"reason": reason, "observations": 1}]


def test_invalid_reference_card_never_silently_uses_execution_card(run):
    write_token_card(run, name="usage_pricing.json")
    write_token_card(run, {})
    token_record(run, "tokens")
    result = observed_token_cost_estimate(run)
    assert result["pricing_error"] and not result["available"]
    assert result["pricing_source"] == "usage_estimate_pricing.json"


def test_ledger_write_gaps_preserve_partial_estimate(run):
    token_record(run, "tokens")
    (run / "usage_gaps.jsonl").write_text('{"reason":"synthetic lost observation"}\n')
    result = observed_token_cost_estimate(run, token_card())
    assert result["available"] and result["ledger_write_gaps"] and not result["complete"]
    assert result["estimated_cost_usd"] == pytest.approx(.568)
    assert not result["by_round"][0]["complete"]


def test_status_keeps_multi_round_reference_compact_without_losing_cost_details(run):
    for round_index in range(14):
        token_record(run, f"round-{round_index}", round_index=round_index)
    write_token_card(run, token_card(assumptions="Synthetic public-rate context. " * 200))
    detailed = usage.usage_summary(run)["token_cost_estimate"]
    status = supervision.run_status(run)
    reference = status["usage"]["token_cost_estimate"]
    assert len(json.dumps(status)) < 32 * 1024
    assert reference["estimated_cost_usd"] == detailed["estimated_cost_usd"]
    assert reference["by_round"] == detailed["by_round"]
    assert reference["rates_sha256"] == detailed["rates_sha256"]
    assert reference["priced_group_count"] == 14
    assert reference["unpriced_group_count"] == 0
    assert "groups" not in reference and "unpriced_groups" not in reference
    assert reference["details_command"] == "autofa supervise cost --run test"
    assert len(detailed["groups"]) == 14
    assert supervision.run_status(run, full_snapshots=True)["usage"]["token_cost_estimate"][
        "groups"] == detailed["groups"]


def test_no_usage_is_unknown_not_a_zero_dollar_estimate(run):
    before = set(run.iterdir())
    result = observed_token_cost_estimate(run, token_card())
    assert not result["available"] and result["estimated_cost_usd"] is None
    assert result["reasons"] == [{"reason": "no_observed_tokens", "observations": 0}]
    assert set(run.iterdir()) == before


# Spend admission


def test_admission_distinguishes_new_zero_from_missing_historical_ledger(run):
    assert not usage.admission_usage(run)["available"]
    usage.initialize_run_accounting(run, before_first_dispatch=True)
    fresh = usage.admission_usage(run)
    assert fresh["available"] and fresh["monetary_view"]["cost_complete"]
    assert fresh["totals"]["known_cost_usd"] == 0
    gateway_record(run, "account_baseline", cost_usd=10, cost_kind="reported",
                   measurement="cost", cumulative=True, source_id="previous-account",
                   baseline=True)
    assert usage.admission_usage(run)["totals"]["known_cost_usd"] == 0
    assert usage.admission_usage(run)["available"]
    gateway_record(run, cost_usd=3, cost_kind="reported")
    assert usage.admission_usage(run)["totals"]["known_cost_usd"] == 3
    (run / "usage.sqlite3").unlink()
    lost = usage.admission_usage(run)
    assert not lost["available"]
    assert lost["totals"]["known_cost_usd"] is None
    assert not lost["monetary_view"]["cost_complete"]


def test_admission_cache_invalidates_rates_and_writes_and_excludes_token_snapshots(run):
    gateway_record(run)
    assert usage.admission_usage(run)["monetary_view"]["missing_cost_observations"] == 1
    usage.initialize_run_accounting(run, gateway_card())
    assert usage.admission_usage(run)["monetary_view"][
        "known_and_derived_subtotal_usd"] == pytest.approx(2.4)
    gateway_record(run, "paid", cost_usd=5, cost_kind="reported")
    gateway_record(run, "tokens", measurement="usage")
    expected = usage.admission_usage(run)
    assert expected["monetary_view"]["known_and_derived_subtotal_usd"] == pytest.approx(7.4)
    assert expected["estimated_dispatch_cost_usd"] is None  # no configured discovery harness
    assert usage_estimates.dispatch_cost_estimate(run, {"role": "detector"}) == 5


def test_admission_reads_a_ledger_another_worker_is_creating_as_empty(run, monkeypatch):
    # The first writer commits the whole schema at once. A concurrent admission read
    # sees no observations instead of unavailable cost, which would stop paid work.
    usage.initialize_run_accounting(run, before_first_dispatch=True)
    read_database, initialize_totals = usage._read_database, usage._initialize_admission_totals
    connect = usage._connect
    attempted, visible, readers, result = threading.Event(), [], [], {}

    @contextmanager
    def observed(path):
        with read_database(path) as db:
            visible.append(sorted(row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")))
            try:
                yield db
            finally:
                attempted.set()

    def initialize(db):
        readers.append(threading.Thread(target=lambda: result.update(
            usage.admission_usage(run))))
        readers[0].start()
        assert attempted.wait(5)
        initialize_totals(db)

    def connected(directory):
        db = connect(directory)
        readers[0].join(5)  # schema committed; the first observation is not yet written
        return db

    monkeypatch.setattr(usage, "_read_database", observed)
    monkeypatch.setattr(usage, "_initialize_admission_totals", initialize)
    monkeypatch.setattr(usage, "_connect", connected)
    record(run, cost_usd=1, cost_kind="reported")
    assert not readers[0].is_alive()
    assert visible[0] == []  # the first read ran before the schema commit
    assert result["available"] and result["notices"] == []
    assert result["totals"]["known_cost_usd"] == 0
    assert usage.admission_usage(run)["totals"]["known_cost_usd"] == 1


def test_ledger_left_without_schema_reads_as_missing(run):
    # A first writer stopped before its schema commit leaves no observations.
    usage.initialize_run_accounting(run, before_first_dispatch=True)
    with closing(sqlite3.connect(run / "usage.sqlite3")) as db:
        db.execute("PRAGMA journal_mode=WAL")
    admission = usage.admission_usage(run)
    assert admission["available"] and admission["notices"] == []
    assert admission["totals"]["known_cost_usd"] == 0
    assert usage.usage_summary(run)["totals"]["calls"] == 0
    record(run, cost_usd=1, cost_kind="reported")
    assert usage.admission_usage(run)["totals"]["known_cost_usd"] == 1


def test_admission_does_not_cache_transient_read_failure(run, monkeypatch):
    gateway_record(run, cost_usd=3, cost_kind="reported")
    original = usage._read_rows
    monkeypatch.setattr(usage, "_read_rows", lambda *args, **kwargs:
                        (_ for _ in ()).throw(sqlite3.DatabaseError("transient read")))
    assert not usage.admission_usage(run)["available"]
    monkeypatch.setattr(usage, "_read_rows", original)
    assert usage.admission_usage(run)["totals"]["known_cost_usd"] == 3


def test_admission_corrupted_pricing_is_unknown_instead_of_failing_dispatch(run):
    gateway_record(run)
    (run / "usage_pricing.json").write_text(json.dumps({"version": 1}))
    snapshot = usage.admission_usage(run)
    assert snapshot["available"] and not snapshot["monetary_view"]["cost_complete"]
    assert snapshot["totals"]["known_cost_usd"] is None


# Projections


def test_projection_uses_completed_rounds_and_subtracts_active_work(run):
    for index, dollars in enumerate([2, 4]):
        round_cost(run, index, dollars)
        round_cost(run, index, 1, role="judge")
    # The active round may have only some of the expected roles so far.
    round_cost(run, 2, 1)
    projection = usage.usage_summary(run, completed_rounds=2, planned_rounds=4)["projection"]
    assert projection["available"]
    assert projection["projected_remaining_cost_usd"] == 7
    assert projection["projected_total_cost_usd"] == 16


@pytest.mark.parametrize("reason", ["span", "missing", "model_change", "config_change"])
def test_projection_refuses_misleading_basis(run, reason):
    for index in range(2):
        round_cost(run, index, effort="high" if index == 1 and reason == "model_change"
                   else "medium")
    if reason == "span":
        record(run, "shared", round_span=[0, 1], cost_usd=10, cost_kind="estimated")
    elif reason == "missing":
        record(run, "missing", round_index=1)
    elif reason == "config_change":
        configure(run, top_x=200)
    projection = usage.usage_summary(run, completed_rounds=2, planned_rounds=4)["projection"]
    assert not projection["available"]


def test_projection_recovers_after_two_fresh_consistent_rounds(run):
    """Cost projections follow the recent workload after driver tuning or fallback."""
    for index in (0, 1):
        round_cost(run, index, cost=1)
    configure(run, top_x=200)
    round_cost(run, 2, cost=4)
    assert not usage.usage_summary(run, completed_rounds=3, planned_rounds=6)["projection"][
        "available"]
    round_cost(run, 3, cost=6)
    projection = usage.usage_summary(run, completed_rounds=4, planned_rounds=6)["projection"]
    assert projection["available"]
    assert projection["basis_rounds"] == [2, 3]
    assert projection["projected_remaining_cost_usd"] == 10
    assert projection["projected_total_cost_usd"] == 22


def test_missing_old_cost_still_blocks_a_fresh_basis(run):
    round_cost(run, 0, cost=None)
    configure(run, top_x=200)
    for index in (1, 2):
        round_cost(run, index)
    assert not usage.usage_summary(run, completed_rounds=3, planned_rounds=5)["projection"][
        "available"]


def test_shared_round_spend_is_complete_without_inventing_per_round_cost(run):
    configure(run, round=3, max_rounds=5)
    record(run, "shared", round_index=None, provider="claude",
           usage={"input_tokens": 100, "output_tokens": 20},
           cost_usd=5, cost_kind="estimated", round_span=[0, 2])
    summary = usage.usage_summary(run)
    assert summary["totals"]["cost_complete"]
    assert summary["totals"]["known_cost_usd"] == 5
    assert not summary["projection"]["available"]
    assert summary["by_round"][0]["round_index"] is None


class Clock(datetime):
    """Freeze the forecast clock at ``Clock.at``."""

    at = None

    @classmethod
    def now(cls, tz=None):
        return cls.at


def test_time_horizon_uses_time_and_reserve_not_planned_round_denominator(run, monkeypatch):
    policy = {"version": 1, "mode": "improve_until", "deadline": "2026-11-04T00:01:40+00:00",
              "delivery_reserve_seconds": 10, "delivery_reserve_usd": 5, "spend_limit_usd": 100}
    configure(run, round=2, max_rounds=2, campaign_policy=policy)
    monkeypatch.setattr(usage, "_now", lambda: "2026-11-04T00:00:00+00:00")
    gateway_record(run, "round0", round_index=0, cost_usd=2, cost_kind="reported",
                   dispatch_started_at="2026-11-03T23:59:55+00:00")
    monkeypatch.setattr(usage, "_now", lambda: "2026-11-04T00:00:10+00:00")
    gateway_record(run, "round1", round_index=1, cost_usd=4, cost_kind="reported",
                   dispatch_started_at="2026-11-04T00:00:05+00:00")
    monkeypatch.setattr(Clock, "at", datetime(2026, 11, 4, 0, 0, 20, tzinfo=UTC))
    monkeypatch.setattr(usage_estimates, "datetime", Clock)
    forecast = usage.usage_summary(run)["projection"]
    assert forecast["available"], forecast
    assert forecast["remaining_discovery_seconds"] == 70
    assert forecast["projected_remaining_cost_usd"] == pytest.approx(42)
    assert forecast["finalization_delivery_allowance_usd"] == 5
    assert forecast["projected_total_cost_usd"] == pytest.approx(53)
    # Final work is observed outside rounds and contributes to the recorded total.
    gateway_record(run, "final", phase="finalization", round_index=None,
                   cost_usd=3, cost_kind="reported")
    assert totals(run)["known_cost_usd"] == 9


def test_unpriced_or_compaction_work_prevents_confident_horizon_forecast(run):
    configure(run, max_rounds=2, campaign_policy={"deadline": "2099-01-01T00:00:00Z"})
    gateway_record(run)
    report = usage.usage_summary(run)
    assert not report["projection"]["available"]
    assert not report["monetary_view"]["cost_complete"]


def test_scope_isolation_and_discovery_refresh_preserve_dispatch_context(run, tmp_path):
    standalone = tmp_path / "taxonomy.usage"
    standalone.mkdir()
    with usage.usage_scope(run_dir=run, round_index=0):
        dispatched = usage.capture_context(role="detector")
        configure(run, round=1)
        assert dispatched["round_index"] == 0
        assert usage.capture_context(refresh_round=True)["round_index"] == 1
        with usage.usage_scope(run_dir=standalone, phase="taxonomy", exact_dir=True):
            assert usage.capture_context()["run_dir"] == str(standalone)
            assert usage.capture_context()["round_index"] is None


# Native lifecycle attribution of real API calls


@pytest.fixture
def api():
    def respond(request):
        return httpx.Response(200, json={
            "id": "synthetic", "created": 0, "model": "model", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "yes"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })

    with OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(
        transport=httpx.MockTransport(respond),
    )) as client:
        yield client


@pytest.fixture
def current(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"run_id": "current", "round": 2}))
    rt = SimpleNamespace(
        run=SimpleNamespace(run_dir=lambda: directory, round=2, max_rounds=8),
        config=Config(), library=SimpleNamespace(functions=[], review_history={}),
    )
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    monkeypatch.setattr("autofa.session.operation_admission.admit_operation",
                        lambda *args: nullcontext(None))
    return rt


def test_finalization_api_workers_are_outside_discovery_round(current, api, monkeypatch):
    llm = HubLLM(api, model="model", api="chat_completions")

    def evaluate(console):
        assert usage.capture_context()["phase"] == "finalization"
        return {"answers": list(ordered_bounded_map(
            lambda item: llm.classify("predicate", str(item)), range(2), workers=2,
            max_pending=2, max_bytes=100, size_of=lambda item: 1,
        ))}

    monkeypatch.setattr(finalization, "_evaluate_final_library", evaluate)
    with usage.usage_scope(run_dir=current.run.run_dir(), phase="discovery", round_index=2):
        result = finalization.evaluate_final_library(Console(file=StringIO()))
        assert usage.capture_context()["phase"] == "discovery"
    assert result == {"answers": ["yes", "yes"]}
    summary = usage.usage_summary(current.run.run_dir())
    assert summary["totals"]["calls"] == 2
    assert summary["by_phase"][0]["phase"] == "finalization"
    assert summary["by_round"][0]["round_index"] is None


def test_round_validation_uses_live_round_and_restores_outer_scope(current, api, monkeypatch):
    current.run.round = 3  # Persisted config still identifies the previous cursor.
    current.library.functions = [SimpleNamespace(name="predicate", scores={})]
    monkeypatch.setattr(round_validation, "grade_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(round_validation, "active_round_refinement", lambda rt: None)
    monkeypatch.setattr(round_validation, "emit_runtime_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(round_validation, "validation_feedback", lambda *args: {"message": "ok"})
    monkeypatch.setattr(round_validation, "refresh_validated_target", lambda *args, **kwargs: {})

    def score(**kwargs):
        chat_completions_safe(api, model="model", messages=[], usage_role="detector")
        return {"ok": True}

    def review(**kwargs):
        chat_completions_safe(api, model="model", messages=[], usage_role="judge")
        return {"ok": True, "unresolved": []}

    monkeypatch.setattr("autofa.mcp.tools_eval.evaluate_round_functions", score)
    monkeypatch.setattr("autofa.metrics.final_review.final_review_functions", review)
    with usage.usage_scope(run_dir=current.run.run_dir(), phase="discovery", round_index=2):
        assert round_validation.validate_round(current)["ok"]
        assert usage.capture_context()["phase"] == "discovery"
        assert usage.capture_context()["round_index"] == 2
    summary = usage.usage_summary(current.run.run_dir())
    assert summary["totals"]["calls"] == 2
    assert summary["by_phase"][0]["phase"] == "round_validation"
    assert summary["by_round"][0]["round_index"] == 3
    assert {r["role"] for r in summary["by_role"]} == {"detector", "judge"}


def test_mcp_call_keeps_dispatch_run_when_runtime_changes_before_parse(
    current, api, monkeypatch, tmp_path,
):
    current.run.round = 3
    llm = HubLLM(api, model="model", api="chat_completions")
    response = llm.chat.completions.with_raw_response.create(model="model", messages=[])
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"run_id": "other", "round": 7}))
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(run=SimpleNamespace(
        run_dir=lambda: other, round=7,
    )))
    response.parse()
    assert not usage.usage_summary(other)["available"]
    summary = usage.usage_summary(current.run.run_dir())
    assert summary["totals"]["calls"] == 1
    assert summary["totals"]["tokens"]["input_tokens"] == 10
    assert summary["by_round"][0]["round_index"] == 3


@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "within_run"])
def test_cheating_preflight_keeps_native_cache_without_discovery_ledger(
    current, api, monkeypatch, tmp_path, nested,
):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    if not nested:
        monkeypatch.setattr(runtime, "_RUNTIME", None)
    monkeypatch.setattr(pipeline, "_client", lambda cfg: api)
    monkeypatch.setattr(pipeline, "screen", lambda item: {"escalate": True})

    def confirm(*args, **kwargs):
        chat_completions_safe(api, model="model", messages=[])
        return {"status": "labeled", "tokens": 12, "cheated": False}

    monkeypatch.setattr(pipeline, "confirm", confirm)
    workdir = ((current.run.run_dir() / "preflight" / "cheating") if nested else
               tmp_path / "standalone")
    items = [{"_item_id": str(i), "response_text": "trace"} for i in range(2)]
    cfg = Config(judge_workers=2)
    scope = (usage.usage_scope(run_dir=current.run.run_dir(), phase="discovery", round_index=2)
             if nested else nullcontext())
    with scope:
        assert len(pipeline.judge_items(items, cfg, workdir)) == 2
        reused = pipeline.judge_items(items, cfg, workdir)
    assert all(r["cached"] for r in reused.values())
    assert not list(tmp_path.rglob("usage.sqlite3"))


# Usage stays operational: status and CLI, never report sections


def test_cheating_report_keeps_existing_schema_and_rendering(current, tmp_path, monkeypatch):
    workdir = current.run.run_dir() / "preflight"
    workdir.mkdir(parents=True)
    monkeypatch.setattr(usage, "snapshot_usage",
                        lambda *args: pytest.fail("cheating report must not read discovery usage"))
    summary = pipeline.summarize({}, Config())
    data, page = write_report(workdir, summary=summary, rows=[], sources=[])
    saved = json.loads(data.read_text())
    assert set(saved) == {"created", "summary", "sources", "rollouts"}
    assert "AutoFA analysis usage" not in page.read_text()
    assert not list(tmp_path.rglob("usage.sqlite3"))


def test_taxonomy_report_does_not_read_or_embed_discovery_usage(tmp_path, monkeypatch):
    source = tmp_path / "classifications.jsonl"
    source.write_text(json.dumps({"kind": RECORD_KIND, "status": "ok", "rollout_id": "r:t:0",
                                 "selected_code": "example", "benchmark": "b"}) + "\n")
    monkeypatch.setattr(usage, "snapshot_usage",
                        lambda *args: pytest.fail("taxonomy must not read discovery usage"))
    html = taxonomy_report.write_report([source, source], tmp_path / "taxonomy.html").read_text()
    assert "AutoFA analysis usage" not in html
    assert "autofa_usage_sources" not in taxonomy_report.build_report([source])["meta"]
    assert "__AUTOFA_USAGE__" not in html


def test_taxonomy_data_markers_are_not_replaced_in_embedded_data(tmp_path, monkeypatch):
    payload = {"meta": {}, "warnings": [], "case": "literal __AUTOFA_USAGE__ and __DATA__"}
    monkeypatch.setattr(taxonomy_report, "build_report", lambda *args, **kwargs: payload)
    output = taxonomy_report.write_report([], output=tmp_path / "report.html")
    assert "literal __AUTOFA_USAGE__ and __DATA__" in output.read_text()


def test_status_and_snapshot_accounting_failures_preserve_delivery(run, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("sensitive provider details must not be copied")

    monkeypatch.setattr(usage, "usage_summary", unavailable)
    assert usage.snapshot_usage(run)["available"] is False
    status = supervision.run_status(run)
    assert status["run_id"] == "test" and status["usage"]["available"] is False
    assert "sensitive provider" not in json.dumps(status)


def test_cli_cost_is_readonly_and_pricing_is_native(run, monkeypatch, capsys):
    monkeypatch.setattr(supervision, "resolve_run_dir", lambda _: run)
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    assert supervise_cli.main(["cost", "--run", "test"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["known_cost_usd"] is None
    assert before == {path.name: path.read_bytes() for path in run.iterdir()}
    settings = run / "rates.json"
    settings.write_text(json.dumps(rate_card()))
    pricing = ["pricing", "--run", "test", "--settings-file", str(settings)]
    # Reference rates install while the native run is locked, without touching its ledger.
    with run_lock(run):
        assert supervise_cli.main([*pricing, "--estimate-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ledger_modified"] is False
    assert result["affects_spend_admission"] is False
    assert not (run / "usage_pricing.json").exists()
    assert supervise_cli.main(pricing) == 0
    assert json.loads(capsys.readouterr().out)["ok"]


def test_standalone_cli_pricing_then_accounting(tmp_path, capsys):
    directory = tmp_path / "results.usage"
    directory.mkdir()
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(rate_card()))
    assert supervise_cli.main(["pricing", "--usage-dir", str(directory),
                               "--settings-file", str(rates)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    with usage.usage_scope(run_dir=directory, phase="taxonomy", exact_dir=True):
        usage.record_usage(usage.capture_context(role="taxonomy"), event_id="taxonomy1",
                           provider="test", model="model", usage=counters(1_000_000))
    assert supervise_cli.main(["cost", "--usage-dir", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["known_cost_usd"] == pytest.approx(2.4)
