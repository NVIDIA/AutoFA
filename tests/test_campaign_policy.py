# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Campaign horizons, paid admission, spend accounting and live goal changes stay honest."""

import asyncio
import json
import socket
import subprocess
import sys
from contextlib import ExitStack
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.harness.claude import TurnResult
from autofa.llm import usage as api_usage
from autofa.mcp import runtime as runtime_module
from autofa.mcp import server, tools_library
from autofa.mcp.runtime import Runtime
from autofa.metrics.library_quality_coverage import library_quality_coverage_for_runtime
from autofa.metrics.task_support import build_task_support
from autofa.objectives import config_for_run
from autofa.session import admission, control, round_validation, runner, validation_recovery
from autofa.session import campaign_policy as policy
from autofa.session import usage as session_usage
from autofa.session.admission import (
    admission_snapshot,
    paid_dispatch,
    reconcile_reservation,
    reservation_summary,
)
from autofa.session.campaign_policy import (
    CampaignBoundaryReached,
    directory_status,
    validate_campaign_policy,
)
from autofa.session.control import submit_update
from autofa.session.discovery_budget import discovery_budget_feedback
from autofa.session.polishing_guard import active_polishing
from autofa.session.recovery import read_json
from autofa.session.round_stagnation import migrate_round_policy
from autofa.session.round_validation import active_round_refinement, refresh_validated_target
from autofa.session.usage_estimates import dispatch_cost_estimate
from autofa.session.validation_recovery import recover_interrupted_validation
from autofa.util.atomic import atomic_write_json

from support.coverage import corpus, coverage_detector
from support.discovery import close, detector, keeper, next_round
from support.discovery import interactive as interactive
from support.discovery import interactive_runner as interactive_runner
from support.discovery import rt as stagnation_runtime  # noqa: F401
from support.families import batch_runtime as batch_runtime
from support.families import polishing_round as polishing_round

NOW = datetime(2026, 11, 4, 12, tzinfo=UTC)
PAST = "2000-01-01T00:00:00+00:00"


def decision(tmp_path, value, *, usage=None, quality=None, **config):
    return policy._decision({"campaign_policy": value, **config}, tmp_path, now=NOW,
                            quality=quality or {"minimum_met": False}, usage=usage)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig(run_id="admission", parquet_paths=[], model_id="fixture")
    run.campaign_policy = {"mode": "improve_until", "deadline": PAST}
    run.save()
    return run


def context(run, phase="discovery"):
    return {"run_dir": str(run.run_dir()), "phase": phase, "role": "detector", "round_index": 0}


def refused(dispatch, match=None):
    """Assert that ``dispatch`` cannot admit paid work."""
    with pytest.raises(CampaignBoundaryReached, match=match):
        with paid_dispatch(dispatch):
            pytest.fail("Paid work was admitted past a campaign boundary")


def record_spend(run):
    for i in range(10):
        session_usage.record_usage(context(run), event_id=str(i), provider="fixture",
                                   model="fixture", cost_usd=.1, cost_kind="reported",
                                   usage={"input_tokens": 1, "output_tokens": 1})


# Policy validation and boundary decisions


def test_optional_horizon_and_portable_deadline(tmp_path):
    assert not decision(tmp_path, {})["active"]
    state = decision(tmp_path, {"deadline": "2026-11-04T15:00:00+02:00"},
                     max_rounds=3, round=17)
    assert state["continue_discovery"] and state["seconds_remaining"] == 3600
    # TOML and YAML parsers hand over aware timestamps rather than strings.
    parsed = datetime(2030, 1, 1, 9, tzinfo=UTC)
    assert validate_campaign_policy({"deadline": parsed})["deadline"] == parsed.isoformat()


@pytest.mark.parametrize("change", [
    {"deadline": "2026-11-04T13:00:00"}, {"deadline": "9999-12-31T23:59:59-05:00"},
    {"deadline": "9999-12-31T23:59:59Z", "overtime": {"seconds": 10}},
    {"overtime": {"usd": 5}}, {"round_limit": True}, {"spend_limit_usd": 10 ** 400},
    {"mode": {}}, {"unknown_cost_policy": []},
    {"deadline": None},  # an unbounded campaign: no deadline, spend limit or round limit
])
def test_malformed_policy_is_an_ordinary_validation_error(change):
    with pytest.raises(ValueError):
        validate_campaign_policy({"deadline": "2030-01-01T09:00:00Z", **change})


def test_live_base_increase_keeps_consumed_overtime():
    state = {"overtime": {"used": True, "base_deadline": "2030-01-01T09:00:00Z",
                          "deadline": "2030-01-01T10:00:00Z",
                          "base_spend_limit_usd": 50, "spend_limit_usd": 60}}
    deadline, cap = policy._effective_limits(
        {"deadline": "2030-01-01T09:30:00Z", "spend_limit_usd": 55}, state)
    assert deadline == "2030-01-01T10:30:00+00:00"
    assert cap == 65


def test_delivery_reserve_and_derived_cost_control_admission(tmp_path):
    state = decision(tmp_path, {"spend_limit_usd": 10, "delivery_reserve_usd": 2}, usage={
        "available": True, "totals": {"calls": 2, "known_cost_usd": 1,
                                      "missing_cost_calls": 1},
        "monetary_view": {"known_and_derived_subtotal_usd": 8.5,
                          "missing_cost_observations": 0, "cost_complete": True},
    })
    assert state["stop_reason"] == "spend_reserve"
    assert state["remaining_cost_usd"] == 1.5 and not state["resources_exhausted"]
    state = decision(tmp_path, {"deadline": "2026-11-04T12:01:00Z",
                                "delivery_reserve_seconds": 90})
    assert state["stop_reason"] == "deadline_reserve"


@pytest.mark.parametrize("estimate,reason,delivery_fits", [
    (0, None, None), (1, None, True), (2, "estimated_spend_reserve", True),
    (4, "estimated_spend_exhausted", False),
])
def test_next_dispatch_guard_preserves_delivery_reserve(tmp_path, estimate, reason, delivery_fits):
    usage = {"available": True, "totals": {"calls": 2, "known_cost_usd": 7},
             "estimated_dispatch_cost_usd": estimate}
    value = {"spend_limit_usd": 10, "delivery_reserve_usd": 2}
    first = decision(tmp_path, value, usage=usage)
    assert first["stop_reason"] == reason
    assert first["delivery_dispatch_fits"] is delivery_fits
    assert first["remaining_cost_usd"] == 3 and not first["resources_exhausted"]
    assert first["estimated_spend_guard"] is (reason is not None)
    assert decision(tmp_path, value, usage=usage) == first


def test_only_new_accounted_runs_start_from_zero(tmp_path):
    unknown = {"available": False, "totals": {"calls": 0}}
    assert decision(tmp_path, {"spend_limit_usd": 10}, usage=unknown)["stop_reason"] == (
        "cost_unavailable")
    path = tmp_path / "usage_initialization.json"
    atomic_write_json(path, {"version": 1, "pristine_start": True})
    assert decision(tmp_path, {"spend_limit_usd": 10}, usage=unknown)["continue_discovery"]
    atomic_write_json(path, {"version": 1, "pristine_start": True,
                             "observations_recorded": True})
    assert decision(tmp_path, {"spend_limit_usd": 10}, usage=unknown)["stop_reason"] == (
        "cost_unavailable")


def test_saved_coverage_uses_good_exact_union(tmp_path):
    atomic_write_json(tmp_path / "coverage_summary.json", {
        "good": {"status": "exact", "coverage": .7, "task_coverage": .85},
        "good_or_ok": {"status": "exact", "coverage": .95},
    })
    value = {"mode": "finish_on_target", "round_limit": 20, "coverage_target": .8,
             "coverage_unit": "failed_tasks"}
    status = decision(tmp_path, value, quality={"minimum_met": True})
    assert status["stop_reason"] == "quality_target_met" and status["coverage"] == .85
    value["coverage_unit"] = "failed_attempts"
    assert decision(tmp_path, value, quality={"minimum_met": True})["continue_discovery"]


@pytest.mark.parametrize("target,coverage,pending", [(.8, .6, True), (.8, .8, False),
                                                    (None, .6, False)])
def test_agreed_overtime_accounts_for_explicit_coverage_goal(tmp_path, target, coverage, pending):
    value = {"deadline": NOW.isoformat(), "overtime": {"seconds": 3600}}
    if target is not None:
        value["coverage_target"] = target
    status = decision(tmp_path, value, quality={
        "minimum_met": True,
        "library_quality_coverage": {"status": "exact", "coverage": coverage},
    })
    assert status["minimum_met"]
    assert status["overtime_pending"] is pending


def test_user_stop_wins_over_expired_horizon_and_overtime(tmp_path):
    atomic_write_json(tmp_path / "supervision/lifecycle.json", {"action": "stop"})
    status = decision(tmp_path, {"deadline": "2026-11-04T11:00:00Z",
                                "overtime": {"seconds": 7200}})
    assert status["stop_reason"] == "user_stop"
    assert not status["finalize_due"] and not status["overtime_pending"]


def test_missing_optional_state_and_corrupt_snapshots_remain_conservative(tmp_path):
    assert not policy.directory_status(tmp_path, now=NOW)["active"]
    atomic_write_json(tmp_path / "config.json", {
        "campaign_policy": {"mode": "finish_on_target", "round_limit": 20},
        "min_failure_modes": 1,
    })
    assert policy.directory_status(tmp_path, now=NOW)["continue_discovery"]
    for filename in ("discovery_target.json", "coverage_summary.json", "usage_initialization.json"):
        (tmp_path / filename).write_text("{incomplete snapshot")
    status = policy.directory_status(tmp_path, now=NOW)
    assert status["continue_discovery"] and not status["quality_target_met"]
    atomic_write_json(tmp_path / "config.json", {"campaign_policy": {"spend_limit_usd": 10}})
    status = policy.directory_status(tmp_path, now=NOW,
                                     usage={"available": False, "totals": {"calls": 0}})
    assert status["stop_reason"] == "cost_unavailable"


def test_saved_old_target_cannot_close_live_raised_minimum(tmp_path):
    value = validate_campaign_policy({"mode": "finish_on_target", "round_limit": 30})
    atomic_write_json(tmp_path / "config.json", {"campaign_policy": value, "min_failure_modes": 8})
    path = tmp_path / "discovery_target.json"
    atomic_write_json(path, {"minimum_met": True, "min_failure_modes": 5,
                             "n_validated_failure_modes": 5,
                             "validation_scope": "current_full_corpus_and_representative_review"})
    assert policy.directory_status(tmp_path, now=NOW)["continue_discovery"]
    atomic_write_json(path, {"minimum_met": True, "min_failure_modes": 8,
                             "n_validated_failure_modes": 8})
    assert policy.directory_status(tmp_path, now=NOW)["continue_discovery"]


def test_horizon_budget_guidance_does_not_consolidate_at_default_round_limit():
    result = discovery_budget_feedback(10, 3, campaign_policy={"round_limit": 30})
    assert result["phase"] == "explore" and result["rounds_remaining"] == 20


def test_new_run_snapshots_policy_pricing_and_strict_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    cfg = Config(campaign_policy={"round_limit": 10}, require_llm_detectors=True)
    run = RunConfig.create([], "fixture", [], cfg)
    assert run.campaign_policy["round_limit"] == 10 and run.require_llm_detectors
    assert run.classifier_policy
    initialization = json.loads((run.run_dir() / "usage_initialization.json").read_text())
    assert initialization["pristine_start"]
    cfg.campaign_policy = validate_campaign_policy({"round_limit": 99})
    assert config_for_run(cfg, run).campaign_policy["round_limit"] == 10
    legacy = RunConfig(run_id="legacy", parquet_paths=[], model_id="fixture")
    assert not config_for_run(cfg, legacy).campaign_policy


# Paid dispatch admission


@pytest.mark.parametrize("observer_fails", [False, True])
def test_deadline_checked_before_actual_api_dispatch(run, monkeypatch, observer_fails):
    monkeypatch.setattr(session_usage, "capture_context", lambda **kw: context(run))
    if observer_fails:
        monkeypatch.setattr(api_usage, "_Observation", lambda *args, **kwargs:
                            (_ for _ in ()).throw(ValueError("observer metadata unavailable")))
    with pytest.raises(CampaignBoundaryReached):
        api_usage.observed_call(lambda **kw: pytest.fail("paid dispatch after deadline"),
                                client=SimpleNamespace(), usage_model="fixture", api="responses")
    assert not (run.run_dir() / "usage.sqlite3").exists()


@pytest.mark.parametrize("filename,contents", [
    ("config.json", "{broken"), ("supervision/lifecycle.json", "[]"),
    pytest.param("supervision/lifecycle.json", None, id="io_error"),
])
def test_unreadable_admission_controls_never_admit_paid_work(run, monkeypatch, filename, contents):
    path = run.run_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if contents is not None:
        path.write_text(contents)
    else:
        # An I/O error must not disable the guards either.
        path.write_text('{"action": "stop"}')
        original = Path.read_text

        def unreadable(self, *args, **kwargs):
            if self == path:
                raise PermissionError("synthetic unreadable control")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(ValueError, match=path.name):
        with paid_dispatch(context(run)):
            pytest.fail("Unusable saved control state admitted paid work")


def test_non_posix_owner_probe_preserves_unverified_reservations(run, monkeypatch):
    pid = admission.os.getpid()
    monkeypatch.setattr(admission, "os", SimpleNamespace(
        name="nt", getpid=lambda: pid,
        kill=lambda *args: pytest.fail("A non-POSIX liveness check must never signal the owner")))
    root = run.run_dir() / "supervision/reservations"
    atomic_write_json(root / "external.json", {
        "id": "external", "state": "inflight", "pid": pid + 1,
        "hostname": socket.gethostname(), "estimated_usd": 2.5, "owner": "other-process",
    })
    atomic_write_json(root / "own.json", {
        "id": "own", "state": "inflight", "pid": pid,
        "hostname": socket.gethostname(), "estimated_usd": .5, "owner": admission._OWNER,
    })
    summary = admission.reservation_summary(run.run_dir())
    assert summary["inflight_dispatches"] == 1
    assert summary["estimated_reserved_usd"] == .5
    assert summary["unresolved_dispatches"] == 1
    assert summary["unresolved_reserved_usd"] == 2.5
    assert summary["pending"][0]["id"] == "external"


def test_finalization_uses_reserve_but_cannot_cross_hard_deadline(run):
    run.campaign_policy.update(deadline=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                               delivery_reserve_seconds=600)
    run.save()
    refused(context(run), "deadline_reserve")
    with paid_dispatch(context(run, "finalization")):
        assert admission_snapshot(run.run_dir())["inflight_dispatches"] == 1
    run.campaign_policy["deadline"] = PAST
    run.save()
    refused(context(run, "finalization"))


def test_completed_dispatch_retained_if_deadline_changes(run):
    run.campaign_policy["deadline"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    run.save()
    with paid_dispatch(context(run)):
        run.campaign_policy["deadline"] = PAST
        run.save()
    assert admission_snapshot(run.run_dir())["inflight_dispatches"] == 0


def test_concurrent_mean_reservations_stop_new_work(run, tmp_path):
    run.campaign_policy = {"spend_limit_usd": 1.15, "unknown_cost_policy": "stop"}
    run.save()
    record_spend(run)
    outer = context(run)
    # A path alias must not split reservations for the same physical run.
    link = tmp_path / "run-alias"
    link.symlink_to(run.run_dir(), target_is_directory=True)
    outer["run_dir"] = str(link)
    with paid_dispatch(outer):
        refused(context(run), "inflight_spend_reserve")
    assert admission_snapshot(run.run_dir())["estimated_reserved_usd"] == 0


def test_other_workflows_and_unconfigured_runs_unchanged(run):
    with paid_dispatch(context(run, "taxonomy")):
        pass
    run.campaign_policy = {}
    run.save()
    with paid_dispatch(context(run)):
        pass
    with paid_dispatch(None):
        pass


def test_boundary_is_normal_native_tool_result(run, monkeypatch):
    rt = Runtime(SimpleNamespace(), FunctionLibrary(), run, Config())
    monkeypatch.setattr(runtime_module, "get_runtime", lambda: rt)

    def tool():
        with paid_dispatch(context(run)):
            pytest.fail("paid work started")

    monkeypatch.setitem(server.TOOL_CALLABLES, "test_campaign_boundary", tool)
    result = asyncio.run(server.invoke_tool("test_campaign_boundary", {}))
    payload = json.loads(result["content"][0]["text"])
    assert payload["status"] == "campaign_boundary"
    assert payload["stop_reason"] == "deadline_exhausted"


@pytest.mark.parametrize("finish", ["exhaust", "close", "parse", "error"])
def test_lazy_response_keeps_reservation_until_finished(run, monkeypatch, finish):
    run.campaign_policy["deadline"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    run.save()
    monkeypatch.setattr(session_usage, "capture_context", lambda **kw: context(run))

    class Response:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if finish == "error":
                raise RuntimeError("stream failed")
            raise StopIteration

        def parse(self):
            return {"usage": {"input_tokens": 2, "output_tokens": 1}, "cost_usd": .1}

        def close(self):
            self.closed = True

    response = Response()
    wrapped = api_usage.observed_call(lambda: response, client=SimpleNamespace(),
                            usage_model="m", api="responses")
    assert admission_snapshot(run.run_dir())["inflight_dispatches"] == 1
    if finish == "close":
        wrapped.close()
        assert response.closed
    elif finish == "parse":
        assert wrapped.parse()["cost_usd"] == .1
    elif finish == "error":
        with pytest.raises(RuntimeError, match="stream failed"):
            next(wrapped)
    else:
        assert list(wrapped) == []
    assert admission_snapshot(run.run_dir())["inflight_dispatches"] == 0


# Spend accounting, interrupted charges and continued grants


@pytest.fixture
def spend_run(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOFA_HOME', str(tmp_path))
    return RunConfig.create([], 'fixture', [], Config(campaign_policy={'spend_limit_usd': 50}))


def spend_context(run, role='detector', provider='openai_compatible', model='fixture', kind='api'):
    return {'run_dir': str(run.run_dir()), 'phase': 'discovery', 'round_index': run.round,
            'role': role, 'dispatch_identity': {'kind': kind, 'role': role,
                                               'provider': provider, 'model': model,
                                               'endpoint': None}}


def charge(run, value, event='one', ctx=None, **extra):
    ctx = ctx or spend_context(run)
    identity = ctx['dispatch_identity']
    session_usage.record_usage(ctx, event_id=event, provider=identity['provider'],
                               model=identity['model'],
                               usage={'input_tokens': 10, 'output_tokens': 5},
                               cost_usd=value, cost_kind='reported', **extra)


def seeded_library(run):
    """Charge $40 to ``run`` and save the library a continuation is seeded from."""
    charge(run, 40)
    path = run.run_dir() / 'library.json'
    FunctionLibrary().save(path)
    return path


def test_claude_spanning_round_keeps_complete_monetary_admission(spend_run):
    from autofa.harness.usage import ClaudeUsage

    tracker = ClaudeUsage(run_dir=spend_run.run_dir(), model='fixture')
    spend_run.round = 1
    spend_run.save()
    tracker.observe({'num_turns': 1, 'session_id': 's', 'total_cost_usd': 2,
                     'usage': {'input_tokens': 10, 'output_tokens': 5}})
    report = session_usage.usage_summary(spend_run.run_dir())
    assert report['totals']['unattributed_observations'] > 0
    assert report['monetary_view']['cost_complete']
    assert directory_status(spend_run.run_dir())['continue_discovery']


def test_codex_token_only_compaction_gap_preserves_known_dollars(spend_run):
    from autofa.harness.usage import CodexUsage

    ctx = spend_context(spend_run, 'discovery', 'codex', kind='harness')
    charge(spend_run, .5, ctx=ctx, measurement='cost')
    CodexUsage(run_dir=spend_run.run_dir(), model='fixture').compaction_gap(
        {'threadId': 't', 'turnId': 'turn', 'item': {'id': 'compact'}})
    report = session_usage.usage_summary(spend_run.run_dir())
    assert report['totals']['missing_usage_calls'] == 1
    assert report['monetary_view']['cost_complete']
    assert directory_status(spend_run.run_dir())['continue_discovery']


def test_claude_missing_result_is_recorded_before_reservation_release(spend_run, monkeypatch):
    from autofa.harness.usage import ClaudeUsage, track_claude_messages

    original = ClaudeUsage.finish
    seen = []

    def finish(tracker, status='incomplete'):
        assert reservation_summary(spend_run.run_dir())['inflight_dispatches'] == 1
        seen.append(tracker.context['admission_id'])
        return original(tracker, status)

    async def stream():
        yield {'type': 'assistant'}
        raise RuntimeError('connection closed')

    async def consume():
        async for _ in track_claude_messages(stream(), run_dir=spend_run.run_dir(),
                                             model='fixture'):
            pass

    monkeypatch.setattr(ClaudeUsage, 'finish', finish)
    with pytest.raises(RuntimeError, match='connection closed'):
        asyncio.run(consume())
    rows = session_usage._read_rows(spend_run.run_dir() / 'usage.sqlite3',
                                    'SELECT payload FROM events')
    assert seen and json.loads(rows[0]['payload'])['admission_id'] == seen[0]
    assert directory_status(spend_run.run_dir())['stop_reason'] == 'cost_unavailable'


def test_orphan_reconciliation_does_not_reprice_the_original_unpriced_observation(spend_run):
    ctx ={**spend_context(spend_run), 'admission_id': 'partial-stream'}
    session_usage.record_usage(ctx, event_id='partial', provider='openai_compatible',
                               model='fixture',
                               usage={'input_tokens': 1_000_000, 'output_tokens': 0})
    session_usage.initialize_run_accounting(spend_run.run_dir(), {
        'version': 1, 'currency': 'USD', 'rates': [{
            'provider': 'openai_compatible', 'model': 'fixture', 'input_per_million': 2,
            'output_per_million': 4, 'source': 'test', 'effective_at': '2026-11-04'}]})
    atomic_write_json(spend_run.run_dir() / 'supervision/reservations/partial-stream.json', {
        'id': 'partial-stream', 'state': 'inflight', 'estimated_usd': 2,
        'pid': 99999999, 'hostname': socket.gethostname(), 'role': 'detector',
        'round_index': 0, 'phase': 'discovery', 'dispatch_identity': ctx['dispatch_identity']})
    estimate = session_usage.usage_summary(spend_run.run_dir())['derived_estimate']
    assert estimate['derived_cost_usd'] == 2
    reconcile_reservation(spend_run.run_dir(), 'partial-stream', 3, 'actual provider receipt')
    report = session_usage.usage_summary(spend_run.run_dir())
    assert report['monetary_view']['known_and_derived_subtotal_usd'] == 3
    assert report['monetary_view']['cost_complete']
    assert report['derived_estimate']['derived_cost_usd'] is None


def test_api_reservations_do_not_borrow_whole_turn_costs(spend_run):
    harness = spend_context(spend_run, 'discovery', 'claude', kind='harness')
    api = spend_context(spend_run)
    for index in range(3):
        charge(spend_run, 4, 'turn-' + str(index), harness, measurement='cost')
        charge(spend_run, .001, 'api-' + str(index), api)
    assert dispatch_cost_estimate(spend_run.run_dir(), api) == .001
    assert dispatch_cost_estimate(spend_run.run_dir(), harness) == 4
    with ExitStack() as stack:
        stack.enter_context(paid_dispatch(harness))
        for _ in range(8):
            stack.enter_context(paid_dispatch(deepcopy(api)))
        reserved = reservation_summary(spend_run.run_dir())['estimated_reserved_usd']
        assert reserved == pytest.approx(4.008)


def test_actual_api_can_finish_when_another_whole_turn_is_unaffordable(spend_run):
    spend_run.harness = 'claude'
    spend_run.harness_settings['model'] = 'fixture'
    spend_run.campaign_policy = {'spend_limit_usd': 10, 'delivery_reserve_usd': 2}
    spend_run.save()
    harness = spend_context(spend_run, 'discovery', 'claude', kind='harness')
    api = spend_context(spend_run)
    charge(spend_run, 4, 'turn', harness, measurement='cost')
    charge(spend_run, .001, 'api', api)
    assert not directory_status(spend_run.run_dir())['continue_discovery']
    with paid_dispatch(api):
        pass
    refused(harness, 'estimated_spend_reserve')


def test_round_goal_edit_does_not_change_pricing_scope(spend_run):
    before = session_usage._scope_fingerprint(spend_run.run_dir())
    spend_run.min_failure_modes += 5
    spend_run.save()
    assert session_usage._scope_fingerprint(spend_run.run_dir()) == before


def test_round_advance_with_no_history_is_not_pristine_zero(spend_run):
    spend_run.round = 4
    spend_run.save()
    summary = session_usage.usage_summary(spend_run.run_dir())
    admission = session_usage.admission_usage(spend_run.run_dir())
    assert not summary['monetary_view']['cost_complete']
    assert not admission['monetary_view']['cost_complete']
    assert directory_status(spend_run.run_dir())['stop_reason'] == 'cost_unavailable'


def test_first_charge_commits_before_pristine_receipt_changes(spend_run, monkeypatch):
    original = session_usage.atomic_write_json
    observed = []

    def during_receipt(path, value):
        if path.name == 'usage_initialization.json' and value.get('observations_recorded'):
            concurrent = session_usage.admission_usage(spend_run.run_dir())
            observed.append(concurrent['monetary_view']['cost_complete'])
            assert concurrent['totals']['known_cost_usd'] == 2
        return original(path, value)

    monkeypatch.setattr(session_usage, 'atomic_write_json', during_receipt)
    charge(spend_run, 2)
    assert observed == [True]


def test_bad_saved_pricing_does_not_discard_reported_money(spend_run):
    charge(spend_run, 7)
    (spend_run.run_dir() / 'usage_pricing.json').write_text('{"version":1}')
    report = session_usage.usage_summary(spend_run.run_dir())
    assert report['totals']['known_cost_usd'] == 7
    assert report['monetary_view']['cost_complete']
    assert report['derived_estimate']['pricing_error']


def test_incremental_totals_follow_stream_updates_and_same_stat_revision(spend_run, monkeypatch):
    from autofa.session import usage_estimates

    charge(spend_run, 2)
    assert session_usage.admission_usage(spend_run.run_dir())['totals']['known_cost_usd'] == 2
    # Every filesystem stat can remain unchanged (coarse timestamp filesystems);
    # the SQLite commit revision still invalidates the cached aggregate.
    original = usage_estimates.Path.stat
    snapshots = {}

    def stable_stat(path, *args, **kwargs):
        if path.name in {'usage.sqlite3', 'usage.sqlite3-wal'}:
            if path not in snapshots:
                snapshots[path] = original(path, *args, **kwargs)
            return snapshots[path]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(usage_estimates.Path, 'stat', stable_stat)
    session_usage.admission_usage(spend_run.run_dir())
    charge(spend_run, 3)  # same event ID: enriched observation, not a new call
    assert session_usage.admission_usage(spend_run.run_dir())['totals']['known_cost_usd'] == 3
    assert session_usage.admission_usage(spend_run.run_dir())['totals']['calls'] == 1


def test_killed_owner_retains_reservation_until_explicit_reconciliation(spend_run):
    charge(spend_run, 2)
    spend_run.campaign_policy = {'spend_limit_usd': 10}
    spend_run.save()
    script = '''import json,sys,time
from autofa.session.admission import paid_dispatch
with paid_dispatch(json.loads(sys.argv[1])):
 print('dispatched', flush=True)
 time.sleep(60)
'''
    worker = subprocess.Popen([sys.executable, '-c', script, json.dumps(spend_context(spend_run))],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert worker.stdout.readline().strip() == 'dispatched'
        worker.kill()
        worker.wait(timeout=5)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
    exposure = reservation_summary(spend_run.run_dir())
    assert exposure['unresolved_reserved_usd'] == 2
    assert directory_status(spend_run.run_dir())['stop_reason'] == 'cost_unavailable'
    spend_run.campaign_policy['unknown_cost_policy'] = 'allow_estimate'
    spend_run.save()
    assert directory_status(spend_run.run_dir())['remaining_cost_usd'] == 6
    assert directory_status(spend_run.run_dir())['continue_discovery']
    item = exposure['pending'][0]
    receipt = reconcile_reservation(spend_run.run_dir(), item['id'], 3, 'provider receipt example')
    assert receipt['ok']
    assert reconcile_reservation(spend_run.run_dir(), item['id'], 3,
                                 'provider receipt example')['idempotent']
    assert directory_status(spend_run.run_dir())['remaining_cost_usd'] == 5
    assert session_usage.usage_summary(spend_run.run_dir())['monetary_view']['cost_complete']


def test_continue_transfers_remaining_allowance_once_and_preserves_overtime(spend_run):
    spend_run.campaign_policy = {'spend_limit_usd': 50, 'overtime': {'usd': 10}}
    spend_run.campaign_state = {'overtime': {'used': True, 'base_spend_limit_usd': 50,
                                            'spend_limit_usd': 60}}
    spend_run.save()
    library = seeded_library(spend_run)
    child = RunConfig.create([], 'fixture', [], Config(campaign_policy={'spend_limit_usd': 50}),
                             seed_library_path=str(library))
    assert child.campaign_state['prior_spend_usd'] == 40
    assert child.campaign_state['overtime']['used']
    assert directory_status(child.run_dir())['remaining_cost_usd'] == 20
    refused(spend_context(spend_run), 'budget_transferred')
    with pytest.raises(ValueError, match='already transferred'):
        RunConfig.create([], 'fixture', [], Config(), seed_library_path=str(library))
    charge(child, 5)
    child_library = child.run_dir() / 'library.json'
    FunctionLibrary().save(child_library)
    grandchild = RunConfig.create([], 'fixture', [], Config(), seed_library_path=str(child_library))
    assert directory_status(grandchild.run_dir())['remaining_cost_usd'] == 15


def test_failed_create_cleans_partial_run_and_preserves_source_grant(spend_run, monkeypatch):
    path = seeded_library(spend_run)
    before = set(spend_run.run_dir().parent.iterdir())
    monkeypatch.setattr(session_usage, 'initialize_run_accounting', lambda *a, **k:
                        (_ for _ in ()).throw(OSError('disk write failed')))
    with pytest.raises(OSError, match='disk write failed'):
        RunConfig.create([], 'fixture', [], Config(), seed_library_path=str(path))
    assert set(spend_run.run_dir().parent.iterdir()) == before
    assert directory_status(spend_run.run_dir())['remaining_cost_usd'] == 10
    assert 'budget_transferred' not in RunConfig.load(spend_run.run_id).campaign_state


def test_continue_refuses_an_active_source_without_a_second_grant(spend_run):
    from autofa.session.recovery import run_lock

    path = seeded_library(spend_run)
    before = set(spend_run.run_dir().parent.iterdir())
    with run_lock(spend_run.run_dir()), pytest.raises(RuntimeError, match='already active'):
        RunConfig.create([], 'fixture', [], Config(), seed_library_path=str(path))
    assert set(spend_run.run_dir().parent.iterdir()) == before
    assert directory_status(spend_run.run_dir())['remaining_cost_usd'] == 10


def test_uncommitted_or_missing_source_transfer_cannot_admit_child_spend(spend_run):
    path = seeded_library(spend_run)
    child = RunConfig.create([], 'fixture', [], Config(), seed_library_path=str(path))
    source = RunConfig.load(spend_run.run_id)
    source.campaign_state.pop('budget_transferred')
    source.save()  # prepared child, source grant never committed
    assert directory_status(child.run_dir())['stop_reason'] == 'budget_transfer_pending'
    refused(spend_context(child), 'budget_transfer_pending')
    (source.run_dir() / 'config.json').unlink()
    assert not directory_status(child.run_dir())['cost_admission_allowed']


# Live policy updates against saved round and polishing checkpoints


def test_live_policy_update_does_not_ignore_unreadable_stop(batch_runtime):
    rt = batch_runtime
    path = rt.run.run_dir() / "supervision/lifecycle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{incomplete stop")
    before = deepcopy(vars(rt.run))
    with pytest.raises(ValueError, match="lifecycle.json"):
        policy.apply_policy_update(rt, {"max_rounds": rt.run.max_rounds + 10}, now=NOW)
    assert vars(rt.run) == before


def test_live_partial_policy_update_preserves_overtime_and_stop(batch_runtime):
    rt = batch_runtime
    rt.run.campaign_policy = validate_campaign_policy({
        "round_limit": 20, "deadline": "2030-01-01T10:00:00Z",
        "overtime": {"seconds": 900},
    })
    rt.run.campaign_state = {"overtime": {"used": True, "deadline": "2030-01-01T10:15:00Z"}}
    policy.apply_policy_update(rt, {"campaign_policy": {"coverage_target": .8}}, now=NOW)
    assert rt.run.campaign_policy["round_limit"] == 20
    assert rt.run.campaign_state["overtime"]["used"]
    before = deepcopy(vars(rt.run))
    atomic_write_json(rt.run.run_dir() / "supervision/lifecycle.json", {"action": "pause"})
    result = policy.apply_policy_update(rt, {"max_rounds": 30}, now=NOW)
    assert result["status"] == "deferred" and vars(rt.run) == before


def test_overtime_consumed_once_and_reused_on_resume(batch_runtime, monkeypatch):
    rt = batch_runtime
    rt.run.campaign_policy = validate_campaign_policy({
        "deadline": "2026-11-04T12:00:00Z", "overtime": {"seconds": 3600}})
    monkeypatch.setattr("autofa.session.validation_completion.validated_good_target_for_runtime",
                        lambda rt: {"minimum_met": False})
    before = deepcopy(rt.library.to_dict())
    result = policy.check_campaign_boundary(rt, now=NOW)
    assert result["continue_discovery"] and result["overtime_used"]
    assert result["seconds_remaining"] == 3600
    saved = json.loads((rt.run.run_dir() / "config.json").read_text())
    assert saved["campaign_state"]["overtime"]["used"]
    later = policy.check_campaign_boundary(rt, now=NOW.replace(hour=14))
    assert later["stop_reason"] == "deadline_exhausted" and not later["overtime_pending"]
    assert rt.library.to_dict() == before


def test_live_coverage_target_excludes_ok_detectors(batch_runtime, monkeypatch):
    rt = batch_runtime
    inventory = corpus()
    rt._family_failed_task_inventory = inventory
    good = coverage_detector("supported", inventory, [f"f-{i}-0" for i in range(5)])
    broad_ok = coverage_detector("below_good", inventory, [f"f-{i}-0" for i in range(5, 20)],
                                 precision=.6)
    rt.library.functions = [good, broad_ok]
    for fn in rt.library.functions:
        fn.scores["evaluation_run_id"] = rt.run.run_id
        fn.scores.update(build_task_support(fn, inventory))
    rt.run.campaign_policy = validate_campaign_policy({
        "mode": "finish_on_target", "round_limit": 20,
        "coverage_target": .8, "coverage_unit": "failed_tasks",
    })
    # Isolate the independent coverage condition; qualification itself has its
    # own full-validation tests and must not conceal this denominator regression.
    monkeypatch.setattr("autofa.session.validation_completion.validated_good_target_for_runtime",
                        lambda rt: {"minimum_met": True})
    assert library_quality_coverage_for_runtime(rt)["coverage"] == 1
    status = policy.campaign_status(rt, now=NOW)
    assert status["coverage"] == .25
    assert not status["coverage_target_met"] and status["continue_discovery"]


def test_sampled_refresh_cannot_reuse_full_validation_label(batch_runtime, monkeypatch):
    rt = batch_runtime
    rt.run.min_failure_modes = 3
    rt.run.campaign_policy = validate_campaign_policy({
        "mode": "finish_on_target", "round_limit": 30})
    rt.run.save()
    path = rt.run.run_dir() / "discovery_target.json"
    atomic_write_json(path, {"validation_scope": "current_full_corpus_and_representative_review",
                             "n_validated_failure_modes": 2, "minimum_met": False,
                             "campaign_complete": True, "goal_met": True})
    monkeypatch.setattr(round_validation, "discovery_good_target_for_runtime",
                        lambda rt: {"n_validated_failure_modes": 3, "minimum_met": True})
    refreshed = refresh_validated_target(rt)
    assert "validation_scope" not in refreshed and "campaign_complete" not in refreshed
    assert not refreshed["goal_met"]
    assert policy.directory_status(rt.run.run_dir())["continue_discovery"]


def test_full_checkpoint_is_nonterminal_in_horizon(polishing_round):
    rt, calls = polishing_round
    rt.run.campaign_policy = validate_campaign_policy({"round_limit": 20,
                                                      "checkpoint_every_rounds": 2})
    first = tools_library.advance_round()
    assert first["refinement_pending"] and calls
    assert active_round_refinement(rt).get("mode") != "polishing"
    assert active_polishing(rt) is None
    second = tools_library.advance_round(validation_response="Checked controls; retain findings.")
    assert second["advanced"] and rt.run.round == 2
    assert active_polishing(rt) is None
    assert tools_library._pareto_status()["campaign"]["continue_discovery"]


@pytest.mark.parametrize("guard", ["campaign", "receipt_only", "policy_only"])
def test_resume_campaign_full_checkpoint_never_runs_validation_recovery(
    polishing_round, monkeypatch, guard,
):
    rt, _ = polishing_round
    rt.run.campaign_policy = validate_campaign_policy({
        "round_limit": 20, "checkpoint_every_rounds": 2})
    tools_library.advance_round()
    rt.run.round = max(rt.run.round, rt.run.max_rounds)
    # A later advance_round interrupted after full-corpus scoring began is the
    # recovery path; only the campaign guards keep it from finalizing the run.
    atomic_write_json(rt.run.run_dir() / "operations" / "interrupted.json", {
        "tool": "advance_round", "state": "interrupted", "started_at": NOW.isoformat()})
    with (rt.run.run_dir() / "status_log.jsonl").open("a") as stream:
        stream.write(json.dumps({"ts": (NOW + timedelta(minutes=1)).isoformat(),
                                 "source": "round_validation",
                                 "context": {"phase": "scoring", "fraction": 1.0}}) + "\n")
    path = rt.run.run_dir() / "round_validation.json"
    if guard == "receipt_only":
        rt.run.campaign_policy = None  # The intermediate receipt alone still refuses.
    elif guard == "policy_only":
        path.unlink()  # Interrupted before its first receipt; the policy alone refuses.
    before = path.read_bytes() if path.exists() else None
    monkeypatch.setattr(validation_recovery, "prepare_polishing",
                        lambda *a, **k: pytest.fail("An intermediate checkpoint is not final"))
    assert recover_interrupted_validation(rt) is None
    assert (path.read_bytes() if path.exists() else None) == before


def test_live_goal_update_reopens_polishing_and_keeps_pending_response(polishing_round):
    rt, _ = polishing_round
    tools_library.advance_round()
    path = rt.run.run_dir() / "round_validation.json"
    receipt = json.loads(path.read_text())
    receipt["refinement"].update(status="validating_response", response="Retained on controls")
    atomic_write_json(path, receipt)
    library = deepcopy(rt.library.to_dict())
    result = policy.apply_policy_update(rt, {"max_rounds": 15,
                                            "campaign_policy": {"round_limit": 15}})
    assert result["transition"]["pending_validation_preserved"]
    assert active_polishing(rt) is None
    assert rt.library.to_dict() == library
    assert json.loads(open(result["transition"]["checkpoint"]).read()) == receipt
    assert rt.run.round == 1 and rt.run.max_rounds == 15
    current = active_round_refinement(rt)
    assert current["mode"] == "round_checkpoint"
    assert current["status"] == "validating_response"
    assert current["response"] == "Retained on controls"
    assert policy.apply_policy_update(rt, {"max_rounds": 16})["transition"] is None


def test_live_update_inside_validation_preserves_reopened_phase(polishing_round, monkeypatch):
    rt, _ = polishing_round
    tools_library.advance_round()
    original = round_validation.validate_round

    def validation(runtime):
        submit_update(rt.run.run_dir(), {"max_rounds": 15}, request_id="mid-advance")
        return original(runtime)

    monkeypatch.setattr(round_validation, "validate_round", validation)
    monkeypatch.setattr("autofa.mcp.tools_eval.evaluate_round_functions",
                        lambda **kw: {"ok": True})
    tools_library.advance_round(
        validation_response="Retain supported controls and investigate gaps")
    saved = json.loads((rt.run.run_dir() / "round_validation.json").read_text())
    assert saved["refinement"]["mode"] == "round_checkpoint"
    assert saved["refinement"]["response"] == "Retain supported controls and investigate gaps"


def test_next_closing_checkpoint_preserves_patience_after_live_round_edit(
    stagnation_runtime,  # noqa: F811 -- imported fixture
):
    rt = stagnation_runtime
    rt.library.functions = [detector(rt, "unchanged")]
    close(rt)
    assert next_round(rt)["rounds_without_improve"] == 1
    assert next_round(rt)["rounds_without_improve"] == 2
    path = rt.run.run_dir() / "discovery_round_stagnation.json"
    before = deepcopy(json.loads(path.read_text()))
    rt.run.max_rounds = 6
    migrate_round_policy(rt, old_max_rounds=3)
    checkpoint = next_round(rt)
    assert checkpoint["status"] != "baseline" and checkpoint["rounds_without_improve"] == 2
    after = json.loads(path.read_text())
    assert set(before["rounds"]) <= set(after["rounds"])
    assert after["known_detectors"] == before["known_detectors"]
    assert after["goal_revisions"][-1]["consumed_patience"] == 2
    assert next_round(rt)["should_stop"]


# Session runner: boundaries, resumes and live extensions. The runner logic is shared;
# codex covers its own session loop where noted and the rest use the default claude harness.


def test_campaign_continues_beyond_round_hint_and_updates_live(interactive_runner, monkeypatch):
    harness, _, rt = interactive_runner
    rt.run.min_failure_modes = 0
    rt.run.campaign_policy = validate_campaign_policy({"round_limit": rt.run.round + 2})
    rt.config.discovery_wait_for_plateau = False
    prompts = []

    async def turn(prompt, **kwargs):
        prompts.append(prompt)
        assert len(prompts) <= 2
        rt.run.round += 1
        return TurnResult(session_id="same-thread", text="Saved current evidence")

    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, exit_after_discovery=True, publish_on_exit=False)
    assert len(prompts) == 2
    assert all("ACTIVE CAMPAIGN POLICY" in prompt for prompt in prompts)


def test_hard_limit_preserves_partial_finalization(interactive_runner, monkeypatch):
    _, _, rt = interactive_runner
    rt.run.campaign_policy = validate_campaign_policy({"round_limit": rt.run.round})

    async def finalize(console):
        raise CampaignBoundaryReached({"stop_reason": "spend_exhausted",
                                       "coverage_target_met": False}, phase="finalization")

    async def highlights(*args, **kwargs):
        pytest.fail("Hard-limit finalization must not start optional inference")

    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    monkeypatch.setattr("autofa.session.report_highlights.prepare_report_highlights_async",
                        highlights)
    runner.run_session(rt.run, cfg=rt.config, exit_after_discovery=True, publish_on_exit=False)
    outcome = json.loads((rt.run.run_dir() / "finalization_outcome.json").read_text())
    assert outcome["state"] == "completed_with_gaps"
    assert outcome["stop_reason"] == "spend_exhausted"
    assert not outcome["retained_results_complete"]


@pytest.mark.parametrize("interactive,closed_by", [
    ("codex", "round_limit"), ("claude", "estimated_spend_reserve"),
], indirect=["interactive"])
def test_closed_campaign_resume_finalizes_without_buying_a_turn(
    interactive_runner, monkeypatch, closed_by,
):
    harness, _, rt = interactive_runner
    if closed_by == "round_limit":
        rt.run.campaign_policy = validate_campaign_policy({"round_limit": rt.run.round})
    else:
        rt.run.campaign_policy = validate_campaign_policy({
            "spend_limit_usd": 10, "delivery_reserve_usd": 2})
        usage = {"available": True, "totals": {"calls": 2, "known_cost_usd": 7},
                 "estimated_dispatch_cost_usd": 2}
        monkeypatch.setattr("autofa.session.usage.admission_usage", lambda directory: usage)
    events = []

    async def turn(*args, **kwargs):
        pytest.fail("A closed campaign must not buy another discovery turn")

    async def validate(*args, **kwargs):
        pytest.fail("Pending feedback must not override a campaign boundary")

    async def finalize(console):
        events.append("finalize")

    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.turn_validation.turn_validation_prompt", validate)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    # A repeated resume of the closed campaign stays closed.
    repeats = 1 if closed_by == "round_limit" else 2
    for _ in range(repeats):
        runner.run_session(rt.run, cfg=rt.config, continue_mode=True,
                           exit_after_discovery=True, publish_on_exit=False)
    assert events == ["finalize"] * repeats
    target = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert target["campaign_complete"] and target["stop_reason"] == closed_by
    assert not target["minimum_met"]


@pytest.mark.parametrize("interactive", ["codex"], indirect=True)
def test_startup_finalization_keeps_live_extension_continuation(interactive_runner, monkeypatch):
    harness, _, rt = interactive_runner
    rt.run.campaign_policy = validate_campaign_policy({"round_limit": rt.run.round})
    session_path = rt.run.run_dir() / "session.json"
    atomic_write_json(session_path, {"session_id": "original-session",
                                     "harness": rt.config.harness})
    finalizations, turns, probes = [], [], []

    async def finalize(console):
        finalizations.append(rt.run.round)
        if len(finalizations) == 1:
            submit_update(rt.run.run_dir(), {"campaign_policy": {"round_limit": rt.run.round + 1}},
                          request_id="live-finalization-extension", expected_revision=0)

    async def turn(prompt, **kwargs):
        turns.append((prompt, kwargs["resume"]))
        assert len(turns) == 1
        rt.run.round += 1
        return TurnResult(session_id="original-session", text="Extended discovery completed")

    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    monkeypatch.setattr(runner, "check_llm_readiness",
                        lambda *args, **kwargs: probes.append("probe") or {})
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True,
                       exit_after_discovery=True, publish_on_exit=False)

    assert probes == ["probe"]
    assert len(turns) == 1 and turns[0][1] == "original-session"
    assert "ACTIVE CAMPAIGN POLICY" in turns[0][0]
    assert len(finalizations) == 2 and rt.run.control_revision == 1
    target = json.loads((rt.run.run_dir() / "discovery_target.json").read_text())
    assert target["campaign_complete"] and target["stop_reason"] == "round_limit"


def test_boundary_finalization_resumes_latest_session_after_live_extension(
    interactive_runner, monkeypatch,
):
    harness, _, rt = interactive_runner
    rt.run.campaign_policy = validate_campaign_policy({"round_limit": rt.run.round + 1})
    session_path = rt.run.run_dir() / "session.json"
    atomic_write_json(session_path, {"session_id": "old-session", "harness": rt.config.harness})
    resumes, finalizations = [], []

    async def finalize(console):
        finalizations.append(rt.run.round)
        if len(finalizations) == 1:
            submit_update(rt.run.run_dir(), {"campaign_policy": {"round_limit": rt.run.round + 1}},
                          request_id="outer-finalization-extension", expected_revision=0)

    async def turn(prompt, **kwargs):
        resumes.append(kwargs["resume"])
        assert len(resumes) <= 2
        rt.run.round += 1
        rt.run.save()
        if len(resumes) == 1:
            # Both real harnesses persist the current identity as soon as it is
            # learned, even when a later paid admission aborts the session call.
            atomic_write_json(session_path, {"session_id": "latest-session",
                                             "harness": rt.config.harness})
            raise CampaignBoundaryReached({"stop_reason": "round_limit"})
        return TurnResult(session_id="latest-session", text="Extended discovery completed")

    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, exit_after_discovery=True, publish_on_exit=False)

    assert resumes == ["old-session", "latest-session"]
    assert len(finalizations) == 2 and rt.run.control_revision == 1
    assert json.loads(session_path.read_text())["session_id"] == "latest-session"


def test_non_campaign_extension_during_finalization_reopens_same_session(
    interactive_runner, monkeypatch,
):
    harness, _, rt = interactive_runner
    rt.config.discovery_wait_for_plateau = False
    rt.library.functions = [keeper(i, evaluation_run_id=rt.run.run_id) for i in range(4)]
    session_path = rt.run.run_dir() / "session.json"
    atomic_write_json(session_path, {"session_id": "original", "harness": rt.config.harness})
    finalizations, turns = [], []

    async def finalize(console):
        finalizations.append(True)
        if len(finalizations) == 1:
            submit_update(rt.run.run_dir(), {"max_rounds": 6}, request_id="extend-finalization")

    async def turn(prompt, **kwargs):
        turns.append(kwargs["resume"])
        assert len(turns) == 1
        assert prompt.count("ACTIVE DISCOVERY GOAL UPDATE") == 1
        rt.run.round = 6
        return TurnResult(session_id="original", text="Additional discovery complete")

    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    monkeypatch.setattr(harness, "run_turn", turn)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True,
                       exit_after_discovery=True, publish_on_exit=False)
    assert turns == ["original"] and len(finalizations) == 2


def test_explicit_resume_goals_win_over_older_queued_settings(interactive_runner, monkeypatch):
    harness, _, rt = interactive_runner
    rt.run.save()
    submit_update(rt.run.run_dir(), {"max_rounds": 20, "min_failure_modes": 2}, request_id="old")
    observed = []

    async def session(**kwargs):
        observed.append((rt.run.max_rounds, rt.run.min_failure_modes))
        return "same-session"

    monkeypatch.setattr(harness, "run_session", session)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True, max_rounds=6,
                       min_failure_modes=8, exit_after_discovery=True, publish_on_exit=False)
    assert observed == [(6, 8)]
    saved = json.loads((rt.run.run_dir() / "config.json").read_text())
    assert saved["max_rounds"] == 6 and saved["min_failure_modes"] == 8


def test_resume_recovers_applying_journal_before_persisting_top_x(
    interactive_runner, monkeypatch,
):
    harness, _, rt = interactive_runner
    directory = rt.run.run_dir()
    rt.run.top_x = 5
    rt.run.save()
    original_checkpoint = {"round": rt.run.round, "message": "saved checkpoint"}
    atomic_write_json(directory / "round_validation.json", original_checkpoint)
    receipt = submit_update(directory, {"max_rounds": 9}, request_id="interrupted-update")
    transaction = control._transaction_path(directory, receipt["id"])
    transaction.mkdir(parents=True)
    for name in ("config.json", "round_validation.json"):
        (transaction / name).write_bytes((directory / name).read_bytes())
    atomic_write_json(transaction / "manifest.json", {
        "config.json": True, "round_validation.json": True,
        "discovery_round_stagnation.json": False,
    })
    # A crash changed phase state before config.json committed the new revision.
    atomic_write_json(directory / "round_validation.json", {"message": "partial write"})
    receipt_path = directory / "supervision/controls/interrupted-update.json"
    atomic_write_json(receipt_path, {**receipt, "state": "applying"})
    observed = []

    async def session(**kwargs):
        saved = read_json(directory / "config.json")
        observed.append((rt.run.top_x, saved["top_x"], saved["max_rounds"]))
        assert "top_x=1" in kwargs["system_prompt"]
        assert read_json(receipt_path)["state"] == "applied"
        assert read_json(directory / "round_validation.json") == original_checkpoint
        assert not transaction.exists()
        return "resumed-session"

    monkeypatch.setattr(harness, "run_session", session)
    runner.run_session(rt.run, cfg=rt.config, continue_mode=True, top_x=1,
                       exit_after_discovery=True, publish_on_exit=False)

    assert observed == [(1, 1, 9)]
    assert rt.run.control_revision == 1


def test_pending_limit_during_required_readiness_closes_honestly(interactive_runner, monkeypatch):
    harness, _, rt = interactive_runner
    rt.run.campaign_policy = validate_campaign_policy({"round_limit": rt.run.round + 2})
    rt.run.require_llm_detectors = True
    finalizations = []

    def readiness(*args, **kwargs):
        submit_update(rt.run.run_dir(), {"campaign_policy": {"round_limit": rt.run.round}},
                      request_id="limit-during-readiness")
        raise CampaignBoundaryReached({"stop_reason": "round_limit"})

    async def turn(*args, **kwargs):
        pytest.fail("Pending restriction should close discovery before a harness turn")

    async def finalize(console):
        finalizations.append(True)

    monkeypatch.setattr(runner, "check_llm_readiness", readiness)
    monkeypatch.setattr(harness, "run_turn", turn)
    monkeypatch.setattr("autofa.session.finalization.finalize_discovery", finalize)
    with pytest.raises(RuntimeError, match="Required detector readiness was not verified"):
        runner.run_session(rt.run, cfg=rt.config, exit_after_discovery=True, publish_on_exit=False)
    assert finalizations == []


def test_required_endpoint_failure_is_not_hidden_by_a_new_boundary(interactive_runner, monkeypatch):
    harness, _, rt = interactive_runner
    rt.run.campaign_policy = validate_campaign_policy({"round_limit": 20})
    rt.run.require_llm_detectors = True

    def readiness(*args, **kwargs):
        rt.run.campaign_policy["round_limit"] = rt.run.round
        return {"available": False, "reason": "Revoked API credential"}

    async def session(**kwargs):
        pytest.fail("A required access failure must stop before a discovery turn")

    monkeypatch.setattr(runner, "check_llm_readiness", readiness)
    monkeypatch.setattr(harness, "run_session", session)
    with pytest.raises(RuntimeError, match="Revoked API credential"):
        runner.run_session(rt.run, cfg=rt.config, publish_on_exit=False)
