# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native accounting contracts, using synthetic provider data only."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from autofa.harness import usage
from autofa.session.usage import source_cost_kind, usage_summary


def run_round(run_dir, round_index=0):
    (run_dir / "config.json").write_text(json.dumps({"run_id": "run", "round": round_index}))
    return run_dir / "config.json"


class UsageClient:
    """Answer Codex account usage reads from mutable micro-unit counters."""

    def __init__(self, native=None, estimate=None, credits=None):
        self.native, self.estimate, self.credits = native, estimate, credits

    async def request(self, method, params):
        assert method == "account/usage/read"
        assert params == {"threadId": "thread"}
        return {"threadUsage": {"threadId": "thread", "nativeUsageUsdMicros": self.native,
                                "estimatedUsageUsdMicros": self.estimate,
                                "estimatedUsageCreditsMicros": self.credits}}


@pytest.fixture
def observations(monkeypatch, tmp_path):
    state = {"round_index": 0}
    rows = []

    def capture(run_dir=None, *, role="discovery", phase=None, **_):
        return {"run_dir": str(tmp_path), "round_index": state["round_index"],
                "phase": phase or "discovery", "role": role}

    monkeypatch.setattr(usage, "capture_context", capture)
    monkeypatch.setattr(usage, "record_usage", lambda context, **kwargs:
                        rows.append({**context, **kwargs}))
    return rows, state


def test_claude_uses_subagent_aware_totals_and_estimated_cost(observations):
    rows, _ = observations
    tracker = usage.ClaudeUsage(model="parent", resume="session")
    tracker.observe(SimpleNamespace(model="parent", usage={"output_tokens": 1}))
    tracker.observe(SimpleNamespace(
        num_turns=2, session_id="session", is_error=False, total_cost_usd=1.25,
        usage={"input_tokens": 999, "output_tokens": 999},
        model_usage={
            "parent": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 20,
                       "cacheCreationInputTokens": 3, "costUSD": 1},
            "child": {"inputTokens": 7, "outputTokens": 2, "costUSD": 0.25},
        },
    ))
    tracker.finish()
    assert len(rows) == 3
    assert rows[0]["usage"]["input_tokens"] == 33
    assert rows[1]["model"] == "child"
    assert all(row["cumulative"] and row["baseline_unknown"] for row in rows)
    assert sum(row.get("cost_usd") or 0 for row in rows) == 1.25
    assert rows[-1]["cost_kind"] == "estimated"


def test_claude_cross_round_aggregate_is_not_assigned_to_final_round(observations):
    rows, state = observations
    tracker = usage.ClaudeUsage(model="model")
    state["round_index"] = 2
    tracker.observe({"num_turns": 4, "session_id": "session", "total_cost_usd": 2,
                     "usage": {"input_tokens": 50, "output_tokens": 10}})
    assert all(row["round_index"] is None for row in rows)
    assert all(row["round_span"] == [0, 2] for row in rows)


def test_claude_interrupted_stream_records_gap_not_placeholder_tokens(observations):
    rows, _ = observations

    async def messages():
        yield SimpleNamespace(usage={"input_tokens": 100, "output_tokens": 1})
        raise asyncio.CancelledError

    async def run():
        async for _ in usage.track_claude_messages(messages(), role="judge", model="model"):
            pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert len(rows) == 1
    assert rows[0]["status"] == "interrupted"
    assert "usage" not in rows[0]
    assert rows[0]["role"] == "judge"


def test_claude_reset_is_explicit(observations):
    rows, _ = observations
    tracker = usage.ClaudeUsage(resume="old")
    tracker.observe(SimpleNamespace(new_conversation_id="new", session_id="old"))
    tracker.observe({"num_turns": 1, "session_id": "new", "total_cost_usd": 0.2,
                     "model_usage": {"model": {"inputTokens": 5, "outputTokens": 2}}})
    assert all(row["reset"] and not row["baseline_unknown"] for row in rows)


def test_codex_token_baseline_and_round_boundary(observations):
    rows, state = observations
    tracker = usage.CodexUsage(model="model")
    tracker.resumed = True
    tracker.tokens({"threadId": "thread", "tokenUsage": {
        "total": {"inputTokens": 120, "cachedInputTokens": 50, "outputTokens": 10},
        "last": {"inputTokens": 20, "cachedInputTokens": 10, "outputTokens": 2},
    }})
    state["round_index"] = 1
    tracker.tokens({"threadId": "thread", "tokenUsage": {
        "total": {"inputTokens": 140, "cachedInputTokens": 55, "outputTokens": 12},
        "last": {"inputTokens": 20, "cachedInputTokens": 5, "outputTokens": 2},
    }})
    assert rows[0]["baseline"]
    assert rows[0]["usage"]["input_tokens"] == 100
    assert rows[1]["usage"]["input_tokens"] == 120  # cache already included
    assert [row["round_index"] for row in rows] == [0, 0, 1]
    assert len({row["source_id"] for row in rows}) == 1


def test_codex_compaction_gap_survives_duplicate_events_resume_and_later_tokens(tmp_path):
    run_round(tmp_path)
    tracker = usage.CodexUsage(run_dir=tmp_path, model="model")
    tracker.thread_id = "thread"
    notification = {"threadId": "thread", "turnId": "turn", "item": {
        "id": "summary", "type": "contextCompaction"}}
    tracker.observe("item/started", notification)
    tracker.observe("item/completed", notification)
    tracker.observe("thread/compacted", {"threadId": "thread", "turnId": "turn"})
    tracker.observe("thread/tokenUsage/updated", {"threadId": "thread", "tokenUsage": {
        "total": {"inputTokens": 120, "outputTokens": 12},
        "last": {"inputTokens": 20, "outputTokens": 2},
    }})
    resumed = usage.CodexUsage(run_dir=tmp_path, model="model")
    resumed.thread_id, resumed.resumed = "thread", True
    resumed.observe("item/completed", notification)
    resumed.observe("thread/tokenUsage/updated", {"threadId": "thread", "tokenUsage": {
        "total": {"inputTokens": 140, "outputTokens": 14},
        "last": {"inputTokens": 20, "outputTokens": 2},
    }})
    summary = usage_summary(tmp_path)
    assert summary["totals"]["missing_usage_calls"] == 1
    assert summary["totals"]["compaction_usage_gaps"] == 1
    assert summary["by_role"][0]["compaction_usage_gaps"] == 1
    assert summary["totals"]["tokens"]["total_tokens"] == 154
    assert summary["totals"]["known_cost_usd"] is None


def test_codex_compaction_gaps_keep_observed_rounds_and_ignore_completion_replay(tmp_path):
    path = run_round(tmp_path)
    tracker = usage.CodexUsage(run_dir=tmp_path, model="model")
    tracker.thread_id = "thread"
    first = {"threadId": "thread", "turnId": "turn", "item": {
        "id": "first-summary", "type": "contextCompaction"}}
    tracker.observe("item/started", first)
    path.write_text(json.dumps({"run_id": "run", "round": 1}))
    tracker.observe("item/completed", first)
    tracker.observe("thread/compacted", {"threadId": "thread", "turnId": "turn"})
    tracker.observe("item/completed", {"threadId": "thread", "turnId": "turn", "item": {
        "id": "next-summary", "type": "contextCompaction"}})
    path.write_text(json.dumps({"run_id": "run", "round": 2}))
    tracker.observe("turn/completed", {"threadId": "thread", "turn": {
        "id": "turn", "items": [{"id": "next-summary", "type": "contextCompaction"}]}})
    summary = usage_summary(tmp_path)
    assert summary["totals"]["missing_usage_calls"] == 2
    assert summary["totals"]["compaction_usage_gaps"] == 2
    assert [(r["round_index"], r["compaction_usage_gaps"]) for r in summary["by_round"]] == [
        (0, 1), (1, 1)]


def test_codex_legacy_compaction_gap_is_unattributed_and_coalesces_modern_resume(tmp_path):
    path = run_round(tmp_path)
    tracker = usage.CodexUsage(run_dir=tmp_path, model="model")
    tracker.observe("thread/compacted", {"threadId": "thread", "turnId": "turn"})
    path.write_text(json.dumps({"run_id": "run", "round": 2}))
    resumed = usage.CodexUsage(run_dir=tmp_path, model="model")
    resumed.resumed = True
    resumed.observe("item/completed", {"threadId": "thread", "turnId": "turn", "item": {
        "id": "summary", "type": "contextCompaction"}})
    resumed.observe("thread/compacted", {"threadId": "thread", "turnId": "turn"})
    summary = usage_summary(tmp_path)
    assert summary["totals"]["missing_usage_calls"] == 1
    assert summary["totals"]["compaction_usage_gaps"] == 1
    assert summary["totals"]["unattributed_observations"] == 1
    assert summary["by_round"][0]["round_index"] is None


def test_codex_compaction_seen_only_at_completion_keeps_unknown_round_span(tmp_path):
    import sqlite3

    path = run_round(tmp_path)
    tracker = usage.CodexUsage(run_dir=tmp_path, model="model")
    tracker.thread_id = "thread"
    path.write_text(json.dumps({"run_id": "run", "round": 2}))
    tracker.context = tracker.current_context()  # A successful round cost checkpoint.
    tracker.observe("turn/completed", {"threadId": "thread", "turn": {
        "id": "turn", "items": [{"id": "summary", "type": "contextCompaction"}]}})
    tracker.finish("complete")
    summary = usage_summary(tmp_path)
    assert summary["totals"]["missing_usage_calls"] == 1
    assert summary["totals"]["compaction_usage_gaps"] == 1
    assert summary["totals"]["unattributed_observations"] == 1
    assert summary["by_round"][0]["round_index"] is None
    assert not summary["projection"]["available"]
    with sqlite3.connect(tmp_path / "usage.sqlite3") as db:
        row = db.execute("SELECT round_span FROM events WHERE status=?",
                         (usage.CODEX_COMPACTION_USAGE_GAP,)).fetchone()
    assert json.loads(row[0]) == [0, 2]


def test_codex_compaction_accounting_failure_does_not_fail_observer(observations, monkeypatch):
    tracker = usage.CodexUsage(model="model")
    captured = usage.record_usage

    def fail(*args, **kwargs):
        raise OSError("Accounting unavailable")

    monkeypatch.setattr(usage, "record_usage", fail)
    tracker.observe("thread/compacted", {"threadId": "thread", "turnId": "turn"})
    assert observations[0] == []
    monkeypatch.setattr(usage, "record_usage", captured)
    tracker.observe("thread/compacted", {"threadId": "thread", "turnId": "turn"})
    assert len(observations[0]) == 1


def test_codex_unavailable_dollars_are_never_zero(observations):
    rows, _ = observations
    tracker = usage.CodexUsage(model="model")
    tracker.thread_id = "thread"
    asyncio.run(tracker.account(UsageClient()))
    assert tracker.cost_usd is None
    assert rows[-1].get("cost_usd") is None


def test_codex_does_not_switch_incomparable_currency_counters_mid_turn(observations):
    rows, _ = observations
    tracker = usage.CodexUsage(model="model")
    tracker.thread_id = "thread"
    tracker.resumed = True

    async def run():
        client = UsageClient(estimate=5_000_000)
        await tracker.account(client, baseline=True)
        client.native, client.estimate = 1_000_000, 5_500_000
        await tracker.account(client)

    asyncio.run(run())
    assert tracker.cost_usd == 0.5
    assert tracker.cost_kind == "estimated"
    assert rows[-1]["source_id"].endswith(":estimated")


def test_account_query_failure_does_not_fail_model(observations):
    rows, _ = observations
    tracker = usage.CodexUsage(model="model")
    tracker.thread_id = "thread"

    class Client:
        async def request(self, method, params):
            raise RuntimeError("unsupported")

    asyncio.run(tracker.account(Client(), status="failed"))
    assert rows[-1]["status"] == "failed"
    # Without threadUsage the recorded dollars stay unknown, never zero.
    assert tracker.cost_usd is None and rows[-1].get("cost_usd") is None


def test_claude_durable_totals_survive_resume_and_zeroed_failure(tmp_path):
    run_round(tmp_path)

    def result(cost, tokens, *, failed=False):
        return {"num_turns": 1, "session_id": "session", "is_error": failed,
                "total_cost_usd": cost, "model_usage": {
                    "model": {"inputTokens": tokens, "outputTokens": tokens // 10}}}

    usage.ClaudeUsage(run_dir=tmp_path).observe(result(2, 100))
    usage.ClaudeUsage(run_dir=tmp_path, resume="session").observe(result(3, 150))
    usage.ClaudeUsage(run_dir=tmp_path, resume="session").observe(result(0, 0, failed=True))
    usage.ClaudeUsage(run_dir=tmp_path, resume="session").observe(result(4, 200))
    totals = usage_summary(tmp_path)["totals"]
    assert totals["known_cost_usd"] == 4
    assert totals["tokens"]["input_tokens"] == 200
    assert totals["tokens"]["output_tokens"] == 20
    assert not totals["cost_complete"]  # zeroed failed counters are not a confirmed reset


def test_claude_old_resumed_history_is_not_charged_as_new_work(tmp_path):
    run_round(tmp_path)
    for dollars in (100, 101):
        usage.ClaudeUsage(run_dir=tmp_path, resume="legacy").observe({
            "session_id": "legacy", "num_turns": 1, "total_cost_usd": dollars,
            "model_usage": {"model": {"inputTokens": dollars * 100, "outputTokens": dollars}},
        })
    totals = usage_summary(tmp_path)["totals"]
    assert totals["known_cost_usd"] == 1
    assert totals["tokens"]["input_tokens"] == 100
    assert not totals["cost_complete"]


def test_native_worker_nested_directory_keeps_dispatch_round(tmp_path):
    path = run_round(tmp_path)
    directory = tmp_path / "behavior_judge" / "batch" / "judge"
    directory.mkdir(parents=True)
    tracker = usage.ClaudeUsage(run_dir=directory, role="judge", model="model")
    path.write_text(json.dumps({"run_id": "run", "round": 1}))
    tracker.observe({"num_turns": 1, "session_id": "worker", "total_cost_usd": 0.5,
                     "usage": {"input_tokens": 10, "output_tokens": 2}})
    summary = usage_summary(tmp_path)
    assert summary["by_round"][0]["round_index"] == 0
    assert summary["by_role"][0]["role"] == "judge"


def test_codex_dollar_route_stays_frozen_across_restarts_and_gaps(tmp_path):
    run_round(tmp_path)

    def tracker(resumed):
        value = usage.CodexUsage(run_dir=tmp_path)
        value.thread_id, value.resumed = "thread", resumed
        return value

    async def run():
        client = UsageClient(1_000_000, 10_000_000)
        await tracker(False).account(client)
        assert source_cost_kind({"run_dir": str(tmp_path)}, "codex:thread:account") == "reported"
        client.native = None
        second = tracker(True)
        await second.account(client, baseline=True)
        client.estimate = 20_000_000
        await second.account(client)
        assert second.cost_usd is None
        client.native = 3_000_000
        third = tracker(True)
        await third.account(client, baseline=True)
        client.native = 4_000_000
        await third.account(client)

    asyncio.run(run())
    totals = usage_summary(tmp_path)["totals"]
    assert totals["known_cost_usd"] == 4
    assert totals["estimated_cost_usd"] is None
    assert not totals["cost_complete"]


def test_source_cost_kind_lookup_uses_exact_prefix_and_creates_no_files(tmp_path):
    context = {"run_dir": str(tmp_path)}
    assert source_cost_kind(context, "codex:thread:account") is None
    assert list(tmp_path.iterdir()) == []
    run_round(tmp_path)
    usage.record_usage({**context, "round_index": 0, "phase": "discovery", "role": "discovery"},
                       event_id="route", provider="codex", model=None, cost_usd=1,
                       cost_kind="reported", source_id="codex:thread:account:reported",
                       measurement="cost")
    assert source_cost_kind(context, "codex:%:account") is None
    assert source_cost_kind(context, "codex:thread:account") == "reported"


def test_codex_credit_counter_survives_late_dollar_route(tmp_path):
    run_round(tmp_path)
    tracker = usage.CodexUsage(run_dir=tmp_path)
    tracker.thread_id = "thread"

    async def run():
        client = UsageClient(credits=2_000_000)
        await tracker.account(client)
        client.credits, client.estimate = 3_000_000, 1_500_000
        await tracker.account(client)
        restarted = usage.CodexUsage(run_dir=tmp_path)
        restarted.thread_id, restarted.resumed = "thread", True
        await restarted.account(client, baseline=True)
        client.credits, client.estimate = 4_000_000, 2_000_000
        await restarted.account(client)

    asyncio.run(run())
    totals = usage_summary(tmp_path)["totals"]
    assert totals["credits"] == 4
    assert totals["known_cost_usd"] == 2
