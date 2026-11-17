# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live control requests apply at safe boundaries without corrupting the owned campaign."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextvars import Context
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FunctionLibrary
from autofa.mcp import server
from autofa.mcp.runtime import Runtime
from autofa.session import control, supervision
from autofa.session.admission import paid_dispatch
from autofa.session.campaign_policy import CampaignBoundaryReached, directory_status
from autofa.session.control import (
    apply_pending_updates,
    consume_control_requests,
    control_status,
    submit_contribution,
    submit_update,
)
from autofa.session.operation_admission import admit_operation
from autofa.session.recovery import read_json, run_lock
from autofa.session.supervision import discovery_steering, lifecycle_request, submit_steering
from autofa.session.usage import record_usage
from autofa.util.atomic import atomic_write_json

PAST = '2000-01-01T00:00:00+00:00'
INSPECTION = {'item_ids': ['inspected'], 'reason': 'source audit'}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from autofa.mcp import runtime as module

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    run = RunConfig(run_id="live", parquet_paths=[], model_id="test", max_rounds=3)
    run.save()
    rt = Runtime(dataset=SimpleNamespace(), library=FunctionLibrary(), run=run, config=Config())
    monkeypatch.setattr(module, "get_runtime", lambda: rt)
    return rt


@pytest.fixture
def fence(tmp_path):
    root = tmp_path / "campaign"
    atomic_write_json(root / "campaign.json", {"state": "active", "revision": 2})
    atomic_write_json(root / "supervision/owner.json", {"state": "running", "generation": "one"})
    return {"campaign_path": str(root), "revision": 2, "owner_generation": "one"}


def supersede_owner(fence):
    atomic_write_json(Path(fence['campaign_path']) / 'supervision/owner.json',
                      {'state': 'running', 'generation': 'replacement'})


def ctx(rt, phase='discovery'):
    return {'run_dir': str(rt.run.run_dir()), 'phase': phase, 'role': 'detector', 'round_index': 0}


def refused(dispatch, match=None):
    """Assert that ``dispatch`` cannot admit paid work."""
    with pytest.raises(CampaignBoundaryReached, match=match):
        with paid_dispatch(dispatch):
            pytest.fail('Paid work was admitted past a queued campaign limit')


def configure(rt, **policy):
    rt.run.campaign_policy = policy
    rt.run.save()


def future(hours=1):
    return (datetime.now(UTC) + timedelta(hours=hours)).isoformat()


# Queued goal updates


def test_live_update_does_not_take_owner_run_lock(runtime):
    directory = runtime.run.run_dir()
    with run_lock(directory):
        receipt = submit_update(directory, {"max_rounds": 6, "min_failure_modes": 8},
                                request_id="extend", expected_revision=0)
        assert receipt["state"] == "queued"
        assert runtime.run.max_rounds == 3
        applied = apply_pending_updates(runtime)
    assert applied[0]["state"] == "applied"
    assert runtime.run.max_rounds == 6
    assert runtime.run.min_failure_modes == 8
    assert runtime.run.control_revision == 1
    assert runtime.run.round == 0


def test_additional_rounds_idempotent_across_retry_and_resume(runtime):
    directory = runtime.run.run_dir()
    first = submit_update(directory, {"additional_rounds": 14}, request_id="add")
    assert first["settings"]["max_rounds"] == 17
    apply_pending_updates(runtime)
    retried = submit_update(directory, {"additional_rounds": 14}, request_id="add")
    assert retried["state"] == "applied"
    runtime.run = RunConfig.load("live")
    assert apply_pending_updates(runtime) == []
    assert runtime.run.max_rounds == 17
    with pytest.raises(ValueError, match="different"):
        submit_update(directory, {"additional_rounds": 15}, request_id="add")


def test_committed_update_reconciles_missing_receipt(runtime, monkeypatch):
    directory = runtime.run.run_dir()
    submit_update(directory, {"additional_rounds": 2}, request_id="crash")
    original = control._finish

    def fail_receipt(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(control, "_finish", fail_receipt)
    with pytest.raises(OSError):
        apply_pending_updates(runtime)
    assert RunConfig.load("live").max_rounds == 5
    runtime.run = RunConfig.load("live")
    monkeypatch.setattr(control, "_finish", original)
    assert apply_pending_updates(runtime)[0]["state"] == "applied"
    assert runtime.run.max_rounds == 5


def test_failed_policy_commit_restores_memory_and_phase(runtime, monkeypatch):
    directory = runtime.run.run_dir()
    frozen = {"refinement": {"mode": "polishing", "status": "active"}, "round": 0}
    atomic_write_json(directory / "round_validation.json", frozen)
    before = (directory / "config.json").read_bytes()
    original = RunConfig.save

    def fail_save(self):
        raise OSError("failed commit")

    monkeypatch.setattr(RunConfig, "save", fail_save)
    submit_update(directory, {"max_rounds": 9}, request_id="rollback")
    with pytest.raises(OSError, match="failed commit"):
        apply_pending_updates(runtime)
    assert runtime.run.max_rounds == 3
    assert runtime.run.control_revision == 0
    assert (directory / "config.json").read_bytes() == before
    assert read_json(directory / "round_validation.json") == frozen
    monkeypatch.setattr(RunConfig, "save", original)
    assert apply_pending_updates(runtime)[0]["state"] == "applied"
    assert runtime.run.max_rounds == 9


def test_stale_revision_and_user_stop_preserve_existing_state(runtime):
    directory = runtime.run.run_dir()
    submit_update(directory, {"max_rounds": 5}, request_id="first", expected_revision=0)
    submit_update(directory, {"max_rounds": 6}, request_id="stale", expected_revision=0)
    applied = apply_pending_updates(runtime)
    assert [row["state"] for row in applied] == ["applied", "rejected"]
    assert runtime.run.max_rounds == 5
    submit_update(directory, {"max_rounds": 9}, request_id="stopped")
    lifecycle_request(directory, "stop")
    assert apply_pending_updates(runtime)[0]["state"] == "rejected"
    assert runtime.run.max_rounds == 5


@pytest.mark.parametrize('settings', [
    {'campaign_policy': {'deadline': '2030-01-01T00:00:00Z', 'coverage_unit': ['failed_tasks']}},
    {'max_rounds': 1.5},
], ids=['policy', 'max_rounds'])
def test_malformed_update_rejected_before_queue_and_legacy_receipt_cannot_wedge(runtime, settings):
    directory = runtime.run.run_dir()
    with pytest.raises(ValueError):
        submit_update(directory, settings, request_id='bad')
    assert not (directory / 'supervision/controls/bad.json').exists()
    atomic_write_json(directory / 'supervision/controls/legacy-bad.json', {
        'id': 'legacy-bad', 'kind': 'update', 'state': 'queued', 'expected_revision': 0,
        'settings': settings,
    })
    assert apply_pending_updates(runtime)[0]['state'] == 'rejected'
    submit_update(directory, {'max_rounds': 6}, request_id='good')
    assert apply_pending_updates(runtime)[0]['state'] == 'applied'
    assert runtime.run.max_rounds == 6


def test_rollback_journal_cannot_overwrite_newer_round_at_same_control_revision(runtime):
    directory = runtime.run.run_dir()
    submit_update(directory, {'max_rounds': 9}, request_id='old')
    transaction = control._transaction_path(directory, 'old')
    transaction.mkdir(parents=True)
    (transaction / 'config.json').write_bytes((directory / 'config.json').read_bytes())
    atomic_write_json(transaction / 'round_validation.json', {'round': 0})
    atomic_write_json(transaction / 'manifest.json', {
        'config.json': True, 'round_validation.json': True,
    })
    runtime.run.round = 2
    runtime.run.save()
    atomic_write_json(directory / 'round_validation.json', {'round': 2})
    before = (directory / 'config.json').read_bytes()
    result = apply_pending_updates(runtime)[0]
    assert result['state'] == 'uncertain'
    assert (directory / 'config.json').read_bytes() == before
    assert read_json(directory / 'round_validation.json') == {'round': 2}


# Fenced requests from a supervising campaign owner


def test_old_owner_cannot_apply_update_or_deliver_steering(runtime, fence):
    directory = runtime.run.run_dir()
    submit_update(directory, {"max_rounds": 9}, request_id="old", fence=fence)
    submit_steering(directory, "inspect gaps", receipt_id="old-note", fence=fence)
    supersede_owner(fence)
    assert apply_pending_updates(runtime)[0]["state"] == "rejected"
    assert discovery_steering(directory, deliver=True) == []
    assert runtime.run.max_rounds == 3


def test_fenced_update_defers_while_driver_state_is_locked(runtime, fence, monkeypatch):
    hold = control.guarded_fence
    # Contended commits defer after the lock timeout; keep the wait short here.
    monkeypatch.setattr(control, 'guarded_fence', partial(hold, timeout=.05))
    directory = runtime.run.run_dir()
    submit_update(directory, {'max_rounds': 9}, request_id='busy-fence', fence=fence)
    submit_steering(directory, 'retained until campaign edit finishes',
                    receipt_id='busy-note', fence=fence)
    with hold(fence):
        assert apply_pending_updates(runtime)[0]['state'] == 'deferred'
        assert runtime.run.max_rounds == 3
        assert discovery_steering(directory, deliver=True) == []
        assert read_json(directory / 'supervision/steering/busy-note.json')['state'] == 'queued'
    assert apply_pending_updates(runtime)[0]['state'] == 'applied'
    assert discovery_steering(directory, deliver=True)[0]['id'] == 'busy-note'


def test_fenced_request_waits_for_short_campaign_edit(runtime, fence):
    started = threading.Event()

    def edit():
        with control.guarded_fence(fence):
            started.set()
            time.sleep(.1)

    thread = threading.Thread(target=edit)
    thread.start()
    assert started.wait(1)
    try:
        result = submit_update(runtime.run.run_dir(), {'max_rounds': 6}, fence=fence)
        assert result['state'] == 'queued'
    finally:
        thread.join(2)


def test_superseded_uncommitted_update_restores_phase_journal(runtime, fence):
    directory = runtime.run.run_dir()
    original_phase = {'refinement': {'mode': 'polishing', 'status': 'active'}}
    atomic_write_json(directory / 'round_validation.json', original_phase)
    submit_update(directory, {'max_rounds': 9}, request_id='crashed', fence=fence)
    transaction = control._transaction_path(directory, 'crashed')
    transaction.mkdir(parents=True)
    (transaction / 'config.json').write_bytes((directory / 'config.json').read_bytes())
    atomic_write_json(transaction / 'round_validation.json', original_phase)
    atomic_write_json(transaction / 'manifest.json', {
        'config.json': True, 'round_validation.json': True,
        'discovery_round_stagnation.json': False})
    # A killed writer changed this phase file, but never committed config+revision.
    atomic_write_json(directory / 'round_validation.json', {
        'refinement': {'mode': 'polishing', 'status': 'superseded'}})
    supersede_owner(fence)
    assert apply_pending_updates(runtime)[0]['state'] == 'rejected'
    assert read_json(directory / 'round_validation.json') == original_phase
    assert runtime.run.max_rounds == RunConfig.load('live').max_rounds == 3
    assert not transaction.exists()


def test_fenced_lifecycle_cannot_overwrite_saved_user_stop(runtime, fence):
    directory = runtime.run.run_dir()
    lifecycle_request(directory, 'stop')
    before = (directory / 'supervision/lifecycle.json').read_bytes()
    result = lifecycle_request(directory, 'release', fence=fence)
    assert result['ok'] is False and result['status'] == 'stopped_by_supervisor'
    assert (directory / 'supervision/lifecycle.json').read_bytes() == before
    assert supervision.stop_requested(directory)
    assert lifecycle_request(directory, 'release')['action'] == 'release'


@pytest.mark.parametrize('deadline_offset,overtime_eligible', [(30, True), (-400, False)])
def test_fenced_finalize_waits_only_for_eligible_overtime(
    runtime, fence, deadline_offset, overtime_eligible,
):
    directory = runtime.run.run_dir()
    runtime.run.campaign_policy = {
        'deadline': (datetime.now(UTC) + timedelta(seconds=deadline_offset)).isoformat(),
        'delivery_reserve_seconds': 60, 'overtime': {'seconds': 300},
    }
    runtime.run.save()
    result = lifecycle_request(directory, 'finalize', fence=fence)
    if not overtime_eligible:
        # The unused overtime horizon has already expired.
        assert result['action'] == 'finalize'
        return
    assert result['status'] == 'deferred' and result['retryable']
    assert result['reason'] == 'overtime_pending'
    assert supervision.lifecycle_action(directory) is None
    # A manual finalize is still the user's explicit stopping choice.
    assert lifecycle_request(directory, 'finalize')['action'] == 'finalize'


# Queued contributions


def test_explicit_update_dependency_admits_contribution_once(runtime, monkeypatch):
    directory = runtime.run.run_dir()
    calls = []

    async def invoke(name, arguments, *, caller):
        calls.append((name, arguments, caller, runtime.run.max_rounds,
                      runtime.run.control_revision))
        return {'ok': True}

    monkeypatch.setattr(server, 'invoke_tool', invoke)
    submit_update(directory, {'max_rounds': 6}, request_id='goals')
    submit_contribution(directory, 'record_external_inspection', INSPECTION,
                        request_id='candidate', expected_revision=0, after_update='goals')
    receipts = asyncio.run(consume_control_requests(runtime))
    assert [row['state'] for row in receipts] == ['applied', 'applied']
    # The native tool runs once, as the driver, after its goal update committed.
    assert calls == [('record_external_inspection', INSPECTION, 'driver', 6, 1)]
    assert asyncio.run(consume_control_requests(runtime)) == []


@pytest.mark.parametrize('interruption', [
    pytest.param(None, id='cancelled'),
    {'status': 'stopped_by_supervisor'},
])
def test_interrupted_contribution_is_uncertain_and_never_replayed(
    runtime, monkeypatch, interruption,
):
    directory = runtime.run.run_dir()
    calls = []

    async def invoke(name, arguments, *, caller):
        calls.append(name)
        if interruption is None:
            raise asyncio.CancelledError()
        atomic_write_json(directory / 'native-commit.json', {'done': True})
        return {'content': [{'type': 'text', 'text': json.dumps({
            'ok': False, **interruption,
        })}]}

    monkeypatch.setattr(server, 'invoke_tool', invoke)
    submit_contribution(directory, 'record_external_inspection', INSPECTION,
                        request_id='candidate')
    if interruption is None:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(consume_control_requests(runtime))
    else:
        # The supervisor interrupted after the native commit.
        assert asyncio.run(consume_control_requests(runtime))[0]['state'] == 'uncertain'
        assert read_json(directory / 'native-commit.json')['done']
    assert control_status(directory)['receipts'][0]['state'] == 'uncertain'
    assert asyncio.run(consume_control_requests(runtime)) == []
    assert len(calls) == 1


def test_status_does_not_expose_candidate_source(runtime):
    directory = runtime.run.run_dir()
    submit_contribution(directory, "record_external_inspection",
                        {"item_ids": ["private-example"], "reason": "private-evidence"})
    assert "private-evidence" not in json.dumps(control_status(directory))
    assert control_status(directory)["pending"] == 1


def test_successful_finalization_survives_advisory_snapshot_failure(runtime, monkeypatch):
    from autofa import config
    from autofa.mcp import runtime as runtime_module
    from autofa.mcp import tools_library
    from autofa.session import finalization

    directory = runtime.run.run_dir()
    completed = {'state': 'completed', 'retained_results_complete': True}
    monkeypatch.setattr(config, 'load_config', lambda: runtime.config)
    monkeypatch.setattr(runtime_module, 'load_runtime_from_run', lambda *args: runtime)
    monkeypatch.setattr(finalization, 'evaluate_final_library', lambda console: completed)
    monkeypatch.setattr(tools_library, 'library_save', lambda **kw: runtime.persist_library())

    def fail(*args, **kwargs):
        raise OSError('observation disk unavailable')

    monkeypatch.setattr(supervision, 'save_quality_snapshot', fail)
    result = supervision.finalize_run(runtime.run.run_id)
    assert result['state'] == 'returned' and result['outcome'] == completed
    assert 'observation disk unavailable' in result['snapshot_warning']
    assert read_json(directory / 'finalization.json')['state'] == 'returned'


# Supervisor CLI


@pytest.mark.parametrize("revision,code", [("0", 0), ("100", 1)])
def test_cli_update_queues_without_owner_lock_or_returns_json_rejection(
    runtime, tmp_path, capsys, revision, code,
):
    from autofa.supervise_cli import main

    settings = tmp_path / "update.json"
    settings.write_text(json.dumps({"max_rounds": 9}))
    with run_lock(runtime.run.run_dir()):
        assert main(["update", "--run", "live", "--settings-file", str(settings),
                     "--id", "cli", "--expected-revision", revision]) == code
    result = json.loads(capsys.readouterr().out)
    if code:
        # The supervisor reconciles a stale revision from the JSON result.
        assert result["ok"] is False and "Stale control revision" in result["error"]
    else:
        assert result["state"] == "queued"
    assert runtime.run.max_rounds == 3


def test_lifecycle_fence_rechecked_after_cli_validation(
    runtime, fence, tmp_path, monkeypatch, capsys,
):
    from autofa.supervise_cli import main

    directory = runtime.run.run_dir()
    lifecycle_request(directory, 'stop')
    path = tmp_path / 'fence.json'
    atomic_write_json(path, fence)
    original = control.verify_fence
    calls = 0

    def supersede_after_first_check(value, **kwargs):
        nonlocal calls
        original(value, **kwargs)
        calls += 1
        if calls == 1:
            supersede_owner(fence)

    monkeypatch.setattr(control, 'verify_fence', supersede_after_first_check)
    assert main(['request', '--run', 'live', 'release', '--fence-file', str(path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['ok'] is False
    assert 'Superseded' in result['error']
    assert read_json(directory / 'supervision/lifecycle.json')['action'] == 'stop'


def test_cli_exposes_idempotent_cost_reconciliation(runtime, monkeypatch, capsys):
    from autofa.session import admission
    from autofa.supervise_cli import main

    calls = []

    def reconcile(*args):
        calls.append(args)
        return {'ok': True, 'state': 'reconciled'}

    monkeypatch.setattr(admission, 'reconcile_reservation', reconcile, raising=False)
    assert main(['reconcile-cost', '--run', 'live', '--id', 'pending', '--cost-usd', '2.5',
                 '--source', 'provider invoice #1']) == 0
    assert calls == [(runtime.run.run_dir(), 'pending', 2.5, 'provider invoice #1')]
    assert json.loads(capsys.readouterr().out)['state'] == 'reconciled'


# Queued campaign limits restrict new paid work without changing an in-flight cohort


def test_tighter_deadline_stops_next_call_while_current_tool_drains(runtime):
    configure(runtime, deadline=future())
    directory = runtime.run.run_dir()
    before = (directory / 'config.json').read_bytes()
    frozen = {'scope': 'frozen-controls', 'budget': {'spent': 68}}
    atomic_write_json(directory / 'round_validation.json', frozen)
    # A separate context, like an already running scoring worker, owns admission.
    owner, scope = Context(), admit_operation('evaluate_functions', {})
    owner.run(scope.__enter__)
    try:
        with paid_dispatch(ctx(runtime)):
            submit_update(directory, {'campaign_policy': {'deadline': PAST}}, request_id='tighten')
            assert apply_pending_updates(runtime) == []
            with pytest.raises(CampaignBoundaryReached) as caught:
                with paid_dispatch(ctx(runtime)):
                    pytest.fail('Queued shorter deadline did not stop new calls')
            assert caught.value.reason == 'deadline_exhausted'
            overlay = caught.value.status['pending_admission_restriction']
            assert overlay['request_id'] == 'tighten' and not overlay['full_policy_applied']
            completed = 'already admitted result'
        assert completed == 'already admitted result'
        assert (directory / 'config.json').read_bytes() == before
        assert read_json(directory / 'round_validation.json') == frozen
    finally:
        owner.run(scope.__exit__, None, None, None)
    assert apply_pending_updates(runtime)[0]['state'] == 'applied'
    assert runtime.run.campaign_policy['deadline'] == PAST


def test_extension_and_overtime_never_grant_early_dispatch(runtime):
    configure(runtime, deadline=PAST)
    directory = runtime.run.run_dir()
    submit_update(directory, {'campaign_policy': {'deadline': future(),
                                                  'overtime': {'seconds': 600}}})
    refused(ctx(runtime), 'deadline_exhausted')
    assert not directory_status(directory).get('pending_admission_restriction')
    assert apply_pending_updates(runtime)[0]['state'] == 'applied'
    with paid_dispatch(ctx(runtime)):
        pass


def test_tighter_spend_cap_stops_paid_delivery_without_mutating_budget(runtime):
    configure(runtime, spend_limit_usd=100)
    directory = runtime.run.run_dir()
    record_usage(ctx(runtime), event_id='spent', provider='fixture', model='fixture',
                 cost_usd=1, cost_kind='reported', usage={'input_tokens': 1, 'output_tokens': 1})
    submit_update(directory, {'campaign_policy': {'spend_limit_usd': .5}})
    for phase in ('discovery', 'finalization'):
        refused(ctx(runtime, phase), 'spend_exhausted')
    assert runtime.run.campaign_policy['spend_limit_usd'] == 100
    assert read_json(directory / 'config.json')['campaign_policy']['spend_limit_usd'] == 100


def test_queued_round_limit_stops_exploration_and_allows_delivery(runtime):
    runtime.run.round = 2
    configure(runtime, round_limit=30)
    submit_update(runtime.run.run_dir(), {'campaign_policy': {'round_limit': 2}})
    refused(ctx(runtime), 'round_limit')
    with paid_dispatch(ctx(runtime, 'finalization')):
        pass
    assert runtime.run.campaign_policy['round_limit'] == 30


def test_first_valid_update_owns_revision_even_when_it_only_extends(runtime):
    configure(runtime, deadline=future())
    directory = runtime.run.run_dir()
    submit_update(directory, {'campaign_policy': {'deadline': future(3)}}, request_id='first')
    submit_update(directory, {'campaign_policy': {'deadline': PAST}}, request_id='later')
    assert not directory_status(directory).get('pending_admission_restriction')
    with paid_dispatch(ctx(runtime)):
        pass
    result = apply_pending_updates(runtime)
    assert [row['state'] for row in result] == ['applied', 'rejected']
    with paid_dispatch(ctx(runtime)):
        pass


def test_invalid_update_does_not_block_next_valid_tightening(runtime):
    configure(runtime, deadline=future())
    directory = runtime.run.run_dir()
    with pytest.raises(ValueError):
        submit_update(directory, {'min_failure_modes': -1,
                                  'campaign_policy': {'deadline': PAST}}, request_id='invalid')
    submit_update(directory, {'campaign_policy': {'deadline': PAST}}, request_id='valid')
    assert directory_status(directory)['pending_admission_restriction']['request_id'] == 'valid'


@pytest.mark.parametrize('change', ['rejected', 'stale_revision', 'superseded_owner'])
def test_stale_or_rejected_or_fenced_requests_do_not_restrict(runtime, fence, change):
    configure(runtime, deadline=future())
    directory = runtime.run.run_dir()
    receipt = submit_update(directory, {'campaign_policy': {'deadline': PAST}},
                            request_id='restrict', fence=fence)
    assert directory_status(directory).get('pending_admission_restriction')
    if change == 'rejected':
        atomic_write_json(directory / 'supervision/controls/restrict.json',
                          {**receipt, 'state': 'rejected'})
    elif change == 'stale_revision':
        runtime.run.control_revision += 1
        runtime.run.save()
    else:
        supersede_owner(fence)
    with paid_dispatch(ctx(runtime)):
        pass
    assert not directory_status(directory).get('pending_admission_restriction')


def test_larger_total_and_reserve_do_not_shorten_old_horizon(runtime):
    configure(runtime, deadline=future(), delivery_reserve_seconds=0)
    submit_update(runtime.run.run_dir(), {'campaign_policy': {
        'deadline': future(3), 'delivery_reserve_seconds': 7200}})
    # The requested horizon is later overall; applying its reserve alone to the
    # old deadline would invent an earlier stop the user did not request.
    with paid_dispatch(ctx(runtime)):
        pass
    assert not directory_status(runtime.run.run_dir()).get('pending_admission_restriction')


def test_new_policy_limits_legacy_run_without_rewriting_it(runtime):
    assert not runtime.run.campaign_policy
    submit_update(runtime.run.run_dir(), {'campaign_policy': {'deadline': PAST}})
    refused(ctx(runtime))
    assert runtime.run.campaign_policy == {}
