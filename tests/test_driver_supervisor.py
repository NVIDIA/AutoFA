# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver supervisor reviews, steers and stops native work within agreed limits."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "agent-skills/autofa-driver/scripts/supervisor.py"
SPEC = importlib.util.spec_from_file_location("driver_supervisor", HELPER)
module = importlib.util.module_from_spec(SPEC)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class Native:
    def __init__(self):
        self.calls = []
        self.status = {
            "run_id": "R1",
            "round": 0,
            "grade_counts": {"GOOD": 0},
            "turn": {"state": "running"},
            "control": {"revision": 0},
            "usage": {"totals": {"known_cost_usd": 1.0}},
            "library_sha256": "a" * 64,
        }
        self.timeout = False
        self.responses = {}

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if self.timeout:
            raise subprocess.TimeoutExpired(command, 1)
        if "status" in command:
            run_id = command[command.index("--run") + 1]
            status = deepcopy(self.responses.get(run_id, self.status))
            status["run_id"] = run_id
            return 0, json.dumps(status)
        return 0, json.dumps({"ok": True, "state": "queued", "id": "native-ack"})


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "campaign"
    goal = write(
        tmp_path / "goal.json",
        {
            "objective": "Improve supported failure coverage",
            "deadline": "2030-01-01T09:00:00+01:00",
        },
    )
    module.records.state_action(SimpleNamespace(command="init", campaign=root, goal=goal))
    host = tmp_path / "fake_host.py"
    host.write_text("""import json, sys
from pathlib import Path
if sys.argv[1] == "--probe":
 print(json.dumps({"review_protocol": 1})); raise SystemExit
context = json.loads(Path(sys.argv[1]).read_text())
decision = {"assessment": "Healthy predictions are advancing; keep native work running.",
            "evidence": ["observations/R1"], "actions": [],
            "message": "R1 is progressing. Known native cost is $1; no restart needed."}
Path(sys.argv[2]).write_text(json.dumps(decision))
""")
    config = {
        "native": {"argv": ["fake-autofa"], "cwd": str(tmp_path)},
        "host": {
            "kind": "command",
            "argv": [sys.executable, str(host), "{context_file}", "{decision_file}"],
            "probe_argv": [sys.executable, str(host), "--probe"],
        },
        "runs": [{"run_id": "R1", "label": "Synthetic"}],
        "allowed_actions": ["steer", "update", "contribute"],
        "review_seconds": 300,
        "poll_seconds": 60,
        "delivery": {"kind": "outbox", "outbox_only_acknowledged": True},
        "watchdog": {"enabled": True, "intent": "finalize", "reserve_seconds": 60},
    }
    write(root / "supervision/config.json", config)
    clock = [1000.0]
    native = Native()
    supervisor = module.Supervisor(root, clock=lambda: clock[0], runner=native)
    supervisor.acquire()
    yield SimpleNamespace(
        root=root, config=config, clock=clock, native=native, supervisor=supervisor, host=host
    )
    supervisor.terminate_worker()


def finish_review(supervisor):
    assert supervisor.worker is not None
    supervisor.worker.wait(timeout=5)
    supervisor.poll_review()
    return supervisor.state()


def set_goal(setup, **changes):
    """Save goal changes to the campaign record and return the record."""
    campaign = setup.supervisor.campaign()
    campaign["goal"].update(changes)
    write(setup.root / "campaign.json", campaign)
    return campaign


def state_action(setup, command, reason="Synthetic user request"):
    module.records.state_action(SimpleNamespace(command=command, campaign=setup.root,
                                                reason=reason))


def review(supervisor):
    return {"id": "review-1", "revision": supervisor.campaign()["revision"],
            "library_sha256_by_run": {"R1": "a" * 64}}


def action(kind="steer", action_id="steering-1", payload=None):
    return {
        "id": action_id,
        "kind": kind,
        "run_id": "R1",
        "reason": "Investigate evidence without replacing native discovery",
        "payload_json": json.dumps(payload or {"message": "Inspect uncovered task clusters."}),
    }


SPEC.loader.exec_module(module)


def test_real_worker_review_cadence_does_not_mutate_healthy_analysis(setup):
    supervisor = setup.supervisor
    supervisor.tick()
    assert not supervisor.state().get("last_completed_review_at")
    state = finish_review(supervisor)
    assert state["last_completed_review_at"] == 1000
    receipt = module.document(Path(state["last_review"]["directory"]) / "receipt.json")
    assert receipt["decision"]["actions"] == []
    assert all("status" in call for call in setup.native.calls)
    outbox = list((setup.root / "supervision/outbox").glob("*.json"))
    assert outbox and all(module.document(path)["state"] == "queued" for path in outbox)
    setup.clock[0] = 1299
    supervisor.tick()
    assert supervisor.worker is None
    setup.clock[0] = 1300
    supervisor.tick(collect=False)
    assert supervisor.worker is not None
    finish_review(supervisor)
    assert supervisor.state()["last_completed_review_at"] == 1300


def test_round_event_triggers_review_before_timer_and_keeps_round_recap(setup):
    supervisor = setup.supervisor
    supervisor.tick()
    finish_review(supervisor)
    setup.clock[0] += 10
    setup.native.status["round"] = 1
    supervisor.tick()
    state = finish_review(supervisor)
    events = state["last_review"]["event_ids"]
    assert any(event["kind"] == "round_completed" and event["after"] == 1 for event in events)
    assert len(list((setup.root / "supervision/outbox").glob("*.json"))) == 2


def test_steering_is_fenced_idempotent_and_only_acknowledged(setup):
    supervisor = setup.supervisor
    proposal = action()
    result = supervisor.dispatch(proposal, review(supervisor))
    assert result["state"] == "acknowledged"
    command = setup.native.calls[-1]
    assert command[1:3] == ["supervise", "steer"]
    fence = module.document(command[command.index("--fence-file") + 1])
    assert fence == {"campaign_path": str(setup.root), "revision": 0, "monitoring_revision": 0,
                     "owner_generation": supervisor.generation}
    calls = len(setup.native.calls)
    assert supervisor.dispatch(proposal, review(supervisor)) == result
    assert len(setup.native.calls) == calls
    with pytest.raises(ValueError, match="Conflicting"):
        supervisor.dispatch(action(payload={"message": "Different instruction"}),
                            review(supervisor))


def test_uncertain_native_effect_is_not_automatically_retried(setup):
    setup.native.timeout = True
    supervisor = setup.supervisor
    proposal = action()
    assert supervisor.dispatch(proposal, review(supervisor))["state"] == "uncertain"
    setup.native.timeout = False
    assert supervisor.dispatch(proposal, review(supervisor))["state"] == "uncertain"
    assert len(setup.native.calls) == 1


def test_verification_requires_evidence_and_is_separate_from_acknowledgment(setup):
    supervisor = setup.supervisor
    supervisor.dispatch(action(), review(supervisor))
    verification = action(
        "verify", "verify-1",
        {"action_id": "steering-1", "state": "effect_verified", "evidence": []},
    )
    with pytest.raises(ValueError, match="evidence"):
        supervisor.dispatch(verification, review(supervisor))
    verification["payload_json"] = json.dumps({
        "action_id": "steering-1", "state": "effect_verified",
        "evidence": ["native/comparison-7:completed"],
    })
    result = supervisor.dispatch(verification, review(supervisor))
    assert result["state"] == "effect_verified"
    assert result["verification_history"][0]["evidence"] == ["native/comparison-7:completed"]
    assert len(setup.native.calls) == 1


def test_goal_revision_or_owner_takeover_prevents_delayed_mutation(setup):
    supervisor = setup.supervisor
    old_review = review(supervisor)
    state_action(setup, "stop")
    state_action(setup, "reopen")
    with pytest.raises(ValueError, match="revision"):
        supervisor.dispatch(action(), old_review)
    new_review = review(supervisor)
    owner = module.document(setup.root / "supervision/owner.json")
    owner["generation"] = "replacement-owner"
    write(setup.root / "supervision/owner.json", owner)
    with pytest.raises(ValueError, match="ownership"):
        supervisor.dispatch(action(), new_review)
    assert not setup.native.calls


def test_update_and_candidate_use_native_queue_with_expected_revision(setup):
    supervisor = setup.supervisor
    update = action("update", "limits-1", {"settings": {"max_rounds": 6}, "expected_revision": 2})
    supervisor.dispatch(update, review(supervisor))
    assert "--expected-revision" in setup.native.calls[-1]
    assert setup.native.calls[-1][setup.native.calls[-1].index("--expected-revision") + 1] == "2"
    contribution = action("contribute", "candidate-1", {
        "tool": "import_candidate", "arguments": {"source": "unscored candidate"},
        "expected_revision": 3,
    })
    supervisor.dispatch(contribution, review(supervisor))
    assert setup.native.calls[-1][2] == "contribute"
    over_budget = action("update", "over-budget", {
        "settings": {"campaign_policy": {"spend_limit_usd": 999999}}, "expected_revision": 3,
    })
    with pytest.raises(ValueError, match="agreed manifest"):
        supervisor.dispatch(over_budget, review(supervisor))


def test_watchdog_operates_while_llm_worker_is_busy_without_repeated_requests(setup):
    supervisor = setup.supervisor
    supervisor.begin_review()
    set_goal(setup, deadline=module.timestamp(setup.clock[0] + 60))
    supervisor.watchdog()
    assert [call[2] for call in setup.native.calls] == ["status", "request"]
    assert "finalize" in setup.native.calls[-1]
    supervisor.watchdog()
    assert len(setup.native.calls) == 2
    assert not supervisor.state().get("last_completed_review_at")


def test_user_stop_requests_native_stop_and_never_launches_another_review(setup):
    supervisor = setup.supervisor
    state_action(setup, "stop")
    supervisor.tick()
    assert supervisor.stopping and supervisor.worker is None
    assert "stop" in setup.native.calls[-1]
    assert supervisor.campaign()["state"] == "stopped"


def test_completed_campaign_does_not_issue_deadline_work(setup):
    supervisor = setup.supervisor
    campaign = supervisor.campaign()
    campaign["state"] = "complete"
    campaign["goal"]["deadline"] = module.timestamp(0)
    write(setup.root / "campaign.json", campaign)
    supervisor.tick()
    assert not setup.native.calls and supervisor.stopping


@pytest.mark.parametrize("failure", ["failed", "timed_out", "superseded"])
def test_failed_review_keeps_round_recap_for_next_success_without_paid_retry_spin(setup, failure):
    supervisor = setup.supervisor
    supervisor.tick()
    finish_review(supervisor)
    setup.native.status["round"] = 1
    supervisor.tick()
    supervisor.worker.wait(timeout=5)
    receipt = supervisor.state()["review"]
    if failure == "failed":
        write(Path(receipt["directory"]) / "decision.json", {})
    elif failure == "timed_out":
        setup.clock[0] += 241
    else:
        campaign = supervisor.campaign()
        campaign["revision"] += 1
        write(setup.root / "campaign.json", campaign)
    supervisor.poll_review()
    state = supervisor.state()
    receipt_key = "last_superseded_review" if failure == "superseded" else "last_review_error"
    assert state[receipt_key]["state"] == failure
    assert state["last_completed_review_at"] == 1000
    pending = [event for event in state["pending_events"] if event["kind"] == "round_completed"]
    assert len(pending) == 1
    supervisor.tick(collect=False)
    if failure == "superseded":
        assert supervisor.worker is not None  # A new goal gets an immediate fresh review.
    else:
        assert supervisor.worker is None  # Failure retry pacing survives the restored event.
        setup.clock[0] += 300
        supervisor.tick(collect=False)
    state = finish_review(supervisor)
    assert any(event["kind"] == "round_completed" and event["after"] == 1
               for event in state["last_review"]["event_ids"])
    message_id = "review-" + state["last_review"]["id"]
    assert any(module.document(path)["id"] == message_id
               for path in (setup.root / "supervision/outbox").glob("*.json"))


def test_takeover_restores_unconsumed_round_event_once(setup):
    supervisor = setup.supervisor
    supervisor.tick()
    finish_review(supervisor)
    setup.native.status["round"] = 1
    supervisor.tick()
    prior = supervisor.state()["review"]
    supervisor.terminate_worker()
    # Simulate the same event also surviving a late observation write.
    state = supervisor.state()
    state["pending_events"] = prior["event_ids"]
    supervisor.save(state)
    supervisor.acquire()
    pending = supervisor.state()["pending_events"]
    assert len([event for event in pending if event["kind"] == "round_completed"]) == 1
    supervisor.tick(collect=False)
    state = finish_review(supervisor)
    assert any(event["kind"] == "round_completed" for event in state["last_review"]["event_ids"])


def test_delivery_requires_same_id_channel_and_remote_ack_not_exit_zero(setup):
    supervisor = setup.supervisor
    supervisor.config["delivery"] = {
        "kind": "command", "argv": ["deliver", "{message_file}"], "channel": "host-thread",
    }
    supervisor.queue_message("event-1", "One completed round")
    supervisor.runner = lambda *args, **kwargs: (0, "{}")
    supervisor.deliver()
    supervisor._delivery_worker.result(timeout=5)
    message = module.document(next((setup.root / "supervision/outbox").glob("*.json")))
    assert message["state"] == "uncertain"
    supervisor.queue_message("event-2", "Another completed round")
    receipt = {"id": "event-2", "delivered": True, "channel": "host-thread",
               "remote_id": "message-9"}
    supervisor.runner = lambda *args, **kwargs: (0, json.dumps(receipt))
    supervisor.deliver()
    supervisor._delivery_worker.result(timeout=5)
    messages = [module.document(path)
                for path in (setup.root / "supervision/outbox").glob("*.json")]
    assert {item["id"]: item["state"] for item in messages} == {
        "event-1": "uncertain", "event-2": "delivered",
    }


@pytest.mark.parametrize("kind", ["codex", "claude"])
def test_builtin_adapters_check_local_help_and_use_readonly_review_commands(setup, kind):
    supervisor = setup.supervisor
    supervisor.config["host"] = {"kind": kind, "executable": "verified-cli", "model": "chosen"}
    help_text = (
        "--print --output-format --json-schema --tools --allowedTools "
        "--strict-mcp-config --mcp-config --disable-slash-commands --safe-mode "
        "--output-schema --output-last-message --sandbox --color"
    )
    calls = []

    def help_only(command, **kwargs):
        calls.append(command)
        assert command[-1] == "--help"
        return 0, help_text

    result = module.capabilities(supervisor.config, runner=help_only)
    assert result["help_verified"] and not result["model_access_verified"]
    folder = setup.root / "review"
    folder.mkdir()
    command = supervisor._host_command(folder)
    assert "verified-cli" == command[0]
    assert "chosen" in command
    assert ("read-only" in command) if kind == "codex" else ("Read,Glob,Grep" in command)
    assert len(calls) == 1


def test_unsupported_host_is_not_assumed_to_support_automation(setup):
    setup.config["host"] = {"kind": "codex"}
    with pytest.raises(ValueError, match="lacks"):
        module.capabilities(setup.config, runner=lambda *args, **kwargs: (0, "old CLI"))


def test_custom_adapter_capability_probe_and_configuration_do_not_start_review(setup):
    result = module.capabilities(setup.config)
    assert result["review_protocol"] == 1
    assert result["conversation_delivery"] == "unavailable_outbox_only"
    assert not list((setup.root / "supervision/reviews").glob("*"))


def test_takeover_marks_old_review_interrupted_without_claiming_completion(setup):
    supervisor = setup.supervisor
    supervisor.begin_review()
    old_folder = Path(supervisor.state()["review"]["directory"])
    supervisor.terminate_worker()
    old_generation = supervisor.generation
    supervisor.acquire()
    assert supervisor.generation != old_generation
    assert module.document(old_folder / "receipt.json")["state"] == "interrupted"
    assert not supervisor.state().get("last_completed_review_at")


def test_no_shell_expansion_or_unconfigured_operation(setup):
    with pytest.raises(ValueError, match="argument arrays"):
        module.argv("autofa; echo unsafe")
    supervisor = setup.supervisor
    supervisor.config["allowed_actions"].append("operation")
    with pytest.raises(ValueError, match="not authorized"):
        supervisor.dispatch(
            action("operation", "arbitrary", {"name": "invented-command"}), review(supervisor)
        )


def test_native_approved_overtime_is_not_preempted_by_original_deadline(setup):
    supervisor = setup.supervisor
    original, extended = module.timestamp(1060), module.timestamp(4660)
    set_goal(setup, deadline=original, overtime={"seconds": 3600})
    setup.native.status["campaign"] = {
        "active": True, "deadline": original, "overtime_pending": True,
        "policy": {"deadline": original, "delivery_reserve_seconds": 60,
                   "overtime": {"seconds": 3600, "when": "minimum_unmet"}},
    }
    supervisor.observe()
    supervisor.watchdog()
    assert all("request" not in call for call in setup.native.calls)
    setup.clock[0] = 4600
    setup.native.status["campaign"].update(
        deadline=extended, overtime_pending=False, overtime_used=True)
    supervisor.observe()
    supervisor.watchdog()
    assert len([call for call in setup.native.calls if "request" in call]) == 1
    assert "finalize" in setup.native.calls[-1]


def test_bound_native_deadline_later_than_agreement_is_flagged_and_paused(setup):
    supervisor = setup.supervisor
    set_goal(setup, deadline=module.timestamp(1060))
    setup.native.status["campaign"] = {
        "active": True, "deadline": module.timestamp(4660),
        "policy": {"deadline": module.timestamp(4660), "delivery_reserve_seconds": 60},
    }
    setup.clock[0] = 1060
    supervisor.observe()
    assert "horizon" in supervisor.state()["resource_issues"][0]["reason"]
    supervisor.watchdog()
    assert "pause" in setup.native.calls[-1]


def test_watchdog_clamps_unchecked_native_deadline_and_preserves_user_stop(setup):
    supervisor = setup.supervisor
    set_goal(setup, deadline=module.timestamp(1060), overtime={"seconds": 600})
    state = supervisor.state()
    state["observations"] = {
        "R1": {"campaign": {"active": True, "deadline": module.timestamp(4660), "policy": {}}},
    }
    supervisor.save(state)
    setup.clock[0] = 1600
    supervisor.watchdog()
    assert "finalize" in setup.native.calls[-1]
    count = len(setup.native.calls)
    state["observations"]["R1"]["lifecycle"] = {"action": "stop"}
    state["resource_issues"] = [{"run_id": "R1", "reason": "Unapproved native deadline"}]
    supervisor.save(state)
    supervisor.watchdog()
    assert len(setup.native.calls) == count


@pytest.mark.parametrize("policy", [
    {"deadline": module.timestamp(1660), "spend_limit_usd": 40},
    {"deadline": module.timestamp(1060), "spend_limit_usd": 50},
    {"deadline": module.timestamp(460), "spend_limit_usd": 40, "overtime": {"seconds": 1200}},
    {"deadline": module.timestamp(1060), "spend_limit_usd": 30, "overtime": {"usd": 20}},
])
def test_conditional_overtime_cannot_be_converted_into_a_larger_base_allowance(policy):
    limits = {"deadline": module.timestamp(1060), "overtime_seconds": 600,
              "spend_limit_usd": 40, "overtime_usd": 10}
    with pytest.raises(ValueError, match="horizon|allocation"):
        module.check_policy_resources(policy, limits)


def test_stale_or_absent_status_uses_only_bounded_agreed_extension(setup):
    supervisor = setup.supervisor
    set_goal(setup, deadline=module.timestamp(1060), overtime={"seconds": 600})
    supervisor.watchdog()
    assert not setup.native.calls
    setup.clock[0] = 1600
    supervisor.watchdog()
    assert "finalize" in setup.native.calls[-1]


def test_shared_cap_copied_to_each_run_is_detected_and_paused(setup):
    supervisor = setup.supervisor
    supervisor.config["runs"].append({"run_id": "R2"})
    set_goal(setup, budgets={
        "spend_limit_usd": 100, "run_allocations_usd": {"R1": 40, "R2": 40},
        "driver_reserve_usd": 10, "delivery_reserve_usd": 10,
    })
    setup.native.status["campaign"] = {
        "active": True, "policy": {"spend_limit_usd": 100}, "spend_limit_usd": 100,
    }
    supervisor.observe()
    assert len(supervisor.state()["resource_issues"]) == 2
    supervisor.watchdog()
    pauses = [call for call in setup.native.calls if "pause" in call]
    assert len(pauses) == 2
    assert {call[call.index("--run") + 1] for call in pauses} == {"R1", "R2"}


@pytest.mark.parametrize("change", [
    {"deadline": "2035-01-01T00:00:00Z"}, {"spend_limit_usd": None}, {"spend_limit_usd": 1000},
    {"overtime": {"seconds": 9999999}}, {"overtime": {"usd": 1000}},
])
def test_broad_policy_allowlist_cannot_expand_authorized_resources(setup, change):
    supervisor = setup.supervisor
    supervisor.config["allowed_update_fields"] = ["campaign_policy"]
    campaign = set_goal(setup, budgets={"spend_limit_usd": 40})
    setup.native.status["campaign"] = {
        "active": True,
        "policy": {"spend_limit_usd": 40, "deadline": campaign["goal"]["deadline"]},
    }
    supervisor.observe()
    update = action("update", "beyond-authorization",
                    {"settings": {"campaign_policy": change}, "expected_revision": 0})
    with pytest.raises(ValueError, match="agreed|allocation"):
        supervisor.dispatch(update, review(supervisor))
    assert all("update" not in call for call in setup.native.calls)


@pytest.mark.parametrize("value", ["x" * 101, "../escape"])
def test_action_ids_match_native_receipt_contract(setup, value):
    with pytest.raises(ValueError, match="filename-safe"):
        setup.supervisor.dispatch(action(action_id=value), review(setup.supervisor))


def test_unattended_outbox_requires_explicit_acknowledgment(setup):
    setup.config["delivery"] = {"kind": "outbox"}
    with pytest.raises(ValueError, match="outbox-only"):
        module.ready_to_start(setup.config)
    setup.config["delivery"]["outbox_only_acknowledged"] = True
    module.ready_to_start(setup.config)


def capped_reviews(setup, *, allowance=2, estimate=1):
    set_goal(setup, budgets={
        "spend_limit_usd": 20, "driver_reserve_usd": allowance, "cost_scope": "campaign",
    })
    setup.supervisor.config["host"]["max_review_cost_usd"] = estimate


def test_campaign_cap_requires_outer_cost_reservation_or_explicit_exclusion(setup):
    campaign = setup.supervisor.campaign()
    campaign["goal"]["budgets"] = {"spend_limit_usd": 20}
    with pytest.raises(ValueError, match="driver_reserve_usd"):
        module.review_budget(campaign, setup.config)
    campaign["goal"]["budgets"]["cost_scope"] = "native_only"
    assert module.review_budget(campaign, setup.config)["allowance_usd"] is None


def test_unknown_outer_cost_consumes_durable_reservation_and_stops_new_reviews(setup):
    supervisor = setup.supervisor
    capped_reviews(setup, allowance=1, estimate=1)
    supervisor.begin_review()
    state = finish_review(supervisor)
    assert state["driver_usage"]["unknown_cost_reviews"] == 1
    assert state["driver_usage"]["reserved_unknown_usd"] == 1
    assert state["driver_usage"]["cost_complete"] is False
    supervisor.acquire()  # A restarted owner cannot reset spent/reserved review allowance.
    supervisor.begin_review()
    assert supervisor.worker is None
    assert supervisor.state()["review_budget_blocked_revision"] == 0
    setup.clock[0] = module.epoch(supervisor.campaign()["goal"]["deadline"])
    supervisor.watchdog()
    assert any("request" in call and "finalize" in call for call in setup.native.calls)
    assert any("Paid driver reviews are paused" in module.document(path)["message"]
               for path in (setup.root / "supervision/outbox").glob("*.json"))


def test_known_host_cost_refunds_only_reported_unused_reservation(setup):
    supervisor = setup.supervisor
    capped_reviews(setup, allowance=1.5, estimate=1)
    supervisor.begin_review()
    folder = Path(supervisor.state()["review"]["directory"])
    write(folder / "cost.json", {"total_cost_usd": 0.5, "usage": {"input_tokens": 12}})
    state = finish_review(supervisor)
    assert state["driver_usage"]["known_cost_usd"] == 0.5
    assert state["driver_usage"]["unknown_cost_reviews"] == 0
    supervisor.begin_review()
    assert supervisor.worker is not None  # The remaining approved $1 admits one more review.
    folder = Path(supervisor.state()["review"]["directory"])
    write(folder / "cost.json", {"total_cost_usd": 2})  # An estimate is not a provider cap.
    state = finish_review(supervisor)
    assert state["driver_usage"]["known_cost_usd"] == 2.5
    supervisor.begin_review()
    assert supervisor.worker is None


def test_bad_optional_cost_receipt_does_not_invalidate_completed_review(setup):
    supervisor = setup.supervisor
    supervisor.begin_review()
    folder = Path(supervisor.state()["review"]["directory"])
    (folder / "cost.json").write_text("not json")
    state = finish_review(supervisor)
    assert state["last_completed_review_at"] == setup.clock[0]
    assert state["last_review"]["accounting_gap"]
    assert state["driver_usage"]["unknown_cost_reviews"] == 1


@pytest.mark.parametrize("verbose", [False, True], ids=["envelope", "verbose_array"])
def test_claude_review_result_reports_outer_cost_in_safe_mode(setup, verbose):
    supervisor = setup.supervisor
    supervisor.begin_review()
    supervisor.worker.wait(timeout=5)
    folder = Path(supervisor.state()["review"]["directory"])
    decision = module.document(folder / "decision.json")
    result = {"type": "result", "structured_output": decision, "total_cost_usd": 0.27,
              "usage": {"input_tokens": 50, "output_tokens": 20}}
    write(folder / "stdout.log", [{"type": "system"}, result] if verbose else result)
    supervisor.config["host"] = {"kind": "claude"}
    assert "--safe-mode" in supervisor._host_command(folder)
    supervisor.poll_review()
    state = supervisor.state()
    assert state["last_review"]["state"] == "completed"
    assert state["driver_usage"]["known_cost_usd"] == 0.27
    assert state["last_review"]["accounting"]["tokens"]["input_tokens"] == 50


@pytest.mark.parametrize("host, match", [
    ({"kind": "codex", "extra_argv": ["-s", "danger-full-access"]}, "only model"),
    ({"kind": "codex", "extra_argv": ["-c", 'sandbox_mode="danger-full-access"']}, "only model"),
    ({"kind": "claude", "extra_argv": ["--permission-mode", "bypassPermissions"]}, "only model"),
    ({"kind": "codex", "model": "A", "extra_argv": ["--model", "B"]}, "model once"),
])
def test_builtin_extra_flags_cannot_override_permissions_or_model(setup, host, match):
    setup.config["host"] = host
    with pytest.raises(ValueError, match=match):
        module.validate_config(setup.config)


def test_codex_filesystem_sandbox_is_not_advertised_as_external_tool_isolation(setup):
    setup.config["host"] = {"kind": "codex"}
    with pytest.raises(ValueError, match="MCP/apps/hooks"):
        module.ready_to_start(setup.config)
    setup.config["host"]["external_tool_review"] = "reviewed"
    module.ready_to_start(setup.config)
    result = module.capabilities(setup.config, runner=lambda *args, **kwargs: (
        0, "--output-schema --output-last-message --sandbox --color"))
    assert result["external_tool_isolation"] == "not_proven_by_filesystem_sandbox"


def test_fence_files_match_native_control_contract(setup):
    from autofa.session.control import verify_fence

    folder = setup.root / "fence-test"
    folder.mkdir()
    fence = module.document(setup.supervisor.fence_file(folder, 0))
    verify_fence(fence)
    state_action(setup, "stop")
    with pytest.raises(ValueError, match="Superseded"):
        verify_fence(fence)
    fence = module.document(
        setup.supervisor.fence_file(folder, setup.supervisor.campaign()["revision"]))
    verify_fence(fence, protective=True)
    with pytest.raises(ValueError, match="Superseded"):
        verify_fence(fence)


@pytest.mark.skipif(os.name == "nt", reason="Detached POSIX service integration")
def test_detached_service_completes_review_after_launcher_exits_and_honors_stop(setup, tmp_path):
    native = tmp_path / "native_fake.py"
    native.write_text("""import json, sys
from pathlib import Path
if "status" in sys.argv:
 print(json.dumps({"run_id":"R1","round":0,"turn":{"state":"running"}}))
else:
 Path(__file__).with_suffix(".request").write_text(json.dumps(sys.argv[1:]))
 print(json.dumps({"ok":True,"state":"requested"}))
""")
    config = setup.config
    config["native"]["argv"] = [sys.executable, str(native)]
    write(setup.root / "supervision/config.json", config)
    command = [sys.executable, str(HELPER)]
    subprocess.run([*command, "doctor", "--campaign", str(setup.root)],
                   capture_output=True, text=True, check=True, timeout=10)
    started = subprocess.run([*command, "start", "--campaign", str(setup.root)],
                             capture_output=True, text=True, check=True, timeout=10)
    pid = json.loads(started.stdout)["pid"]

    def wait_for(predicate):
        until = time.monotonic() + 8
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(0.05)
        pytest.fail("Detached fake supervisor failed to reach the expected boundary")

    try:
        wait_for(lambda: module.document(
            setup.root / "supervision/state.json").get("last_completed_review_at"))
        owner = module.document(setup.root / "supervision/owner.json")
        assert owner["pid"] == pid and owner["state"] == "running"
        # The configuring launcher has exited, but a real separate reviewer ran.
        assert list((setup.root / "supervision/reviews").glob("*/decision.json"))
        state_action(setup, "stop")
        wait_for(lambda: module.document(
            setup.root / "supervision/owner.json")["state"] == "stopped")
        assert "stop" in json.loads(native.with_suffix(".request").read_text())
    finally:
        owner = module.document(setup.root / "supervision/owner.json")
        if owner.get("pid") == pid and owner.get("state") == "running":
            os.kill(pid, signal.SIGTERM)


def test_stop_retries_busy_native_receipt_without_claiming_confirmation(setup):
    supervisor = setup.supervisor
    state_action(setup, "stop")
    requests = []

    def busy_once(command, **kwargs):
        if "request" in command:
            requests.append(command)
            if len(requests) == 1:
                return 1, json.dumps({"ok": False, "retryable": True, "error": "Fence busy"})
            return 0, json.dumps({"ok": True, "action": "stop"})
        return setup.native(command, **kwargs)

    supervisor.runner = busy_once
    supervisor.tick()
    assert not supervisor.stopping and supervisor.worker is None
    first_message = module.ordered_messages(supervisor.directory / "outbox")[0][1]["message"]
    assert "not confirmed" in first_message
    setup.clock[0] += 1
    supervisor.tick()
    assert len(requests) == 1
    setup.clock[0] += 1
    supervisor.tick()
    assert supervisor.stopping and len(requests) == 2


def test_uncertain_stop_reconciles_saved_intent_without_duplicate_mutation(setup):
    state_action(setup, "stop")
    count = 0

    def timed_out_after_save(command, **kwargs):
        nonlocal count
        if "request" in command:
            count += 1
            setup.native.status["lifecycle"] = {"action": "stop"}
            raise subprocess.TimeoutExpired(command, 10)
        return setup.native(command, **kwargs)

    setup.supervisor.runner = timed_out_after_save
    setup.supervisor.tick()
    assert not setup.supervisor.stopping
    setup.clock[0] += 3
    setup.supervisor.tick()
    assert setup.supervisor.stopping and count == 1


def test_watchdog_rechecks_overtime_when_observation_predates_reserve(setup):
    supervisor = setup.supervisor
    set_goal(setup, deadline=module.timestamp(1065), overtime={"seconds": 3600})
    setup.native.status["campaign"] = {
        "active": True, "deadline": module.timestamp(1065), "overtime_pending": False,
        "policy": {"deadline": module.timestamp(1065), "delivery_reserve_seconds": 60,
                   "overtime": {"seconds": 3600}},
    }
    supervisor.observe()
    setup.clock[0] = 1006
    setup.native.status["campaign"]["overtime_pending"] = True
    supervisor.watchdog()
    assert not any("request" in command for command in setup.native.calls)


def test_empty_policy_cannot_disable_agreed_native_limit(setup):
    supervisor = setup.supervisor
    supervisor.config["allowed_update_fields"] = ["campaign_policy"]
    setup.native.status["campaign"] = {
        "active": True, "policy": {"deadline": supervisor.campaign()["goal"]["deadline"]},
    }
    supervisor.observe()
    update = action("update", "disable",
                    {"settings": {"campaign_policy": {}}, "expected_revision": 0})
    with pytest.raises(ValueError, match="remove the agreed deadline"):
        supervisor.dispatch(update, review(supervisor))
    assert not any("update" in row for row in setup.native.calls)


def finished(setup):
    setup.native.status.update(turn={"state": "completed"}, active_operations=[], snapshots={
        "finalization_outcome": {"record": {"state": "completed_with_gaps"}, "sha256": "final"},
    })


def test_completed_runs_cannot_buy_hourly_reviews_forever(setup):
    finished(setup)
    supervisor = setup.supervisor
    for _ in range(13):
        supervisor.tick()
        if supervisor.worker:
            finish_review(supervisor)
        setup.clock[0] += 300
    assert module.driver_usage(supervisor.state())["reviews"] == 1
    assert supervisor.campaign()["state"] == "active"  # No fabricated goal achievement.
    assert "native_runs_finished" == supervisor.closing_reason()


def test_absolute_horizon_caps_closing_reviews_even_with_new_events(setup):
    supervisor = setup.supervisor
    set_goal(setup, deadline=module.timestamp(999))
    for index in range(13):
        setup.native.status["round"] = index
        supervisor.tick()
        if supervisor.worker:
            finish_review(supervisor)
        setup.clock[0] += 300
    assert module.driver_usage(supervisor.state())["reviews"] == 2


def test_complete_action_requires_terminal_evidence_and_records_partial_outcome(setup):
    proposal = action("complete", "finish", {"outcome": "partial", "evidence": ["native report"]})
    with pytest.raises(ValueError, match="terminal"):
        setup.supervisor.dispatch(proposal, review(setup.supervisor))
    finished(setup)
    setup.supervisor.observe()
    setup.supervisor.dispatch(proposal, review(setup.supervisor))
    campaign = setup.supervisor.campaign()
    assert campaign["state"] == "complete" and campaign["completion"]["outcome"] == "partial"
    assert campaign["completion"]["quality_achievement_inferred"] is False


def test_unexpected_observation_failure_keeps_watchdog_alive(setup, monkeypatch):
    state_action(setup, "stop")
    monkeypatch.setattr(setup.supervisor, "poll_operations",
                        lambda: (_ for _ in ()).throw(RuntimeError("Synthetic poll failure")))
    setup.supervisor.safe_tick()
    assert any("stop" in row for row in setup.native.calls)
    error = module.document(setup.supervisor.directory / "service-error.json")["error"]
    assert "Synthetic poll failure" in error


def test_exhausted_review_allowance_still_sends_round_and_heartbeat_status(setup):
    supervisor = setup.supervisor
    state = supervisor.state()
    state["review_budget_blocked_revision"] = supervisor.campaign()["revision"]
    supervisor.save(state)
    supervisor.config["heartbeat_seconds"] = 30
    supervisor.tick()
    setup.native.status["round"] = 1
    setup.clock[0] += 10
    supervisor.tick()
    setup.clock[0] += 30
    supervisor.tick()
    assert supervisor.worker is None
    assert len(module.ordered_messages(supervisor.directory / "outbox")) == 3


def test_cost_rollup_bounds_state_and_retains_unknown_reservations(setup):
    supervisor = setup.supervisor
    state = supervisor.state()
    state["driver_cost_records"] = {
        str(index): {"reported_cost_usd": None if index % 2 else 0.2, "reservation_usd": 0.3}
        for index in range(200)
    }
    before = module.driver_usage(state)
    supervisor.save(state)
    after = module.driver_usage(supervisor.state())
    assert len(supervisor.state()["driver_cost_records"]) == 64
    assert after["reviews"] == before["reviews"] == 200
    assert after["known_cost_usd"] == pytest.approx(before["known_cost_usd"])
    assert after["reserved_unknown_usd"] == pytest.approx(before["reserved_unknown_usd"])
    supervisor.acquire()
    assert module.driver_usage(supervisor.state())["unknown_cost_reviews"] == 100


def test_operation_cannot_override_bound_run_and_legitimate_operation_executes(setup, tmp_path):
    supervisor = setup.supervisor
    supervisor.config["allowed_actions"].append("operation")
    output = tmp_path / "operation.txt"
    supervisor.config["operations"] = {"inspect": {
        "effect": "read", "parameters": ["run_id"],
        "argv": [sys.executable, "-c",
                 "import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
                 str(output), "{run_id}"],
    }}
    wrong = action("operation", "wrong", {"name": "inspect", "values": {"run_id": "R2"}})
    with pytest.raises(ValueError, match="unsupported parameters"):
        supervisor.dispatch(wrong, review(supervisor))
    supervisor.config["operations"]["inspect"]["parameters"] = []
    receipt = supervisor.dispatch(
        action("operation", "inspect", {"name": "inspect"}), review(supervisor))
    process = supervisor.operations["inspect"][0]
    process.wait(timeout=5)
    supervisor.poll_operations()
    assert output.read_text() == "R1" and receipt["state"] == "issued"


def test_contribution_uses_reviewed_library_hash_and_prior_update_dependency(setup):
    supervisor = setup.supervisor
    supervisor.observe()
    captured = review(supervisor)
    supervisor.dispatch(
        action("update", "change", {"settings": {"max_rounds": 7}, "expected_revision": 0}),
        captured)
    state = supervisor.state()
    state["observations"]["R1"]["library_sha256"] = "b" * 64
    supervisor.save(state)
    supervisor.dispatch(action("contribute", "candidate", {
        "tool": "submit_function", "arguments": {"name": "candidate"}, "expected_revision": 0,
    }), captured)
    call = setup.native.calls[-1]
    assert call[call.index("--library-sha256") + 1] == "a" * 64
    assert call[call.index("--after-update") + 1] == "change"


def test_outbox_delivery_is_ordered_and_does_not_block_watchdog(setup):
    supervisor = setup.supervisor
    supervisor.config["delivery"] = {
        "kind": "command", "argv": ["send", "{message_file}"], "channel": "test",
    }
    supervisor.queue_message("z", "First")
    setup.clock[0] += 1
    supervisor.queue_message("a", "Second")
    gate = threading.Event()
    entered = threading.Event()
    sent = []

    def slow_delivery(command, **kwargs):
        if command[0] != "send":
            return setup.native(command, **kwargs)
        item = module.document(command[1])
        sent.append(item["id"])
        entered.set()
        assert gate.wait(timeout=5)
        return 0, json.dumps(
            {"id": item["id"], "channel": "test", "delivered": True, "remote_id": item["id"]})

    supervisor.runner = slow_delivery
    supervisor.deliver()
    assert entered.wait(timeout=5) and sent == ["z"]
    state_action(setup, "stop")
    supervisor.tick(collect=False)
    assert not supervisor.stopping and any("stop" in call for call in setup.native.calls)
    gate.set()
    supervisor._delivery_worker.result(timeout=5)


@pytest.mark.parametrize("patch", [
    {"budgets": []}, {"budgets": {"run_allocations_usd": []}},
    {"campaign_policy": {"deadline": 3}}, {"budgets": {"usage_baselines_usd": {"R1": -1}}},
])
def test_goal_resource_shapes_fail_as_value_errors(patch):
    with pytest.raises(ValueError):
        module.records.validate_goal({"objective": "Synthetic", **patch})


def test_exact_fractional_allocations_and_baselines_are_supported(setup):
    goal = {"objective": "Synthetic", "budgets": {
        "spend_limit_usd": 0.3, "run_allocations_usd": {"R1": 0.1, "R2": 0.2},
        "usage_baselines_usd": {"R1": 5},
    }}
    module.records.validate_goal(goal)
    config = {**setup.config, "runs": [{"run_id": "R1"}, {"run_id": "R2"}]}
    limits = module.resource_limits({"goal": goal}, config, "R1")
    assert limits["spend_limit_usd"] == pytest.approx(5.1)
    module.check_policy_resources({"spend_limit_usd": 5.1}, limits)


def test_failed_status_retains_last_good_limits_without_false_pause(setup):
    supervisor = setup.supervisor
    supervisor.observe()
    setup.native.timeout = True
    supervisor.observe()
    state = supervisor.state()
    assert state["observations"]["R1"]["round"] == 0
    assert state["observations"]["R1"]["stale"]
    assert state["resource_issues"] == []


def test_committed_review_remains_completed_when_message_write_fails(setup, monkeypatch):
    supervisor = setup.supervisor
    supervisor.tick()
    monkeypatch.setattr(supervisor, "queue_message",
                        lambda *_: (_ for _ in ()).throw(OSError("disk issue")))
    state = finish_review(supervisor)
    receipt = module.document(Path(state["last_review"]["directory"]) / "receipt.json")
    assert receipt["state"] == "completed"
    assert "disk issue" in receipt["postprocessing_errors"][0]
    assert not state.get("last_review_error")


def test_codex_cli_review_runs_end_to_end_without_model_calls(setup, tmp_path):
    executable = tmp_path / "codex-fake"
    executable.write_text("#!" + sys.executable + "\n" + """import json, sys
from pathlib import Path
if sys.argv[-1] == "--help":
 print("--output-schema --output-last-message --sandbox --color")
 raise SystemExit()
assert sys.argv[1] == "exec"
assert sys.argv[sys.argv.index("--sandbox")+1] == "read-only"
assert "Native AutoFA owns the analysis" in sys.stdin.read()
path = Path(sys.argv[sys.argv.index("--output-last-message")+1])
path.write_text(json.dumps({"assessment":"Native progress is healthy", "evidence":["saved status"],
                           "actions":[], "message":"Work is progressing"}))
"""
    )
    executable.chmod(0o700)
    supervisor = setup.supervisor
    supervisor.config["host"] = {"kind": "codex", "executable": str(executable),
                                 "model": "test-model", "external_tool_review": "reviewed"}
    assert module.capabilities(supervisor.config)["help_verified"]
    supervisor.tick()
    state = finish_review(supervisor)
    assert state["last_review"]["assessment"] == "Native progress is healthy"
    assert state["last_review"]["state"] == "completed"


def test_duplicate_service_owner_is_rejected_before_any_worker(setup):
    old_generation = setup.supervisor.generation
    replacement = module.Supervisor(setup.root, runner=setup.native)
    with module.records.state_lock(setup.supervisor.directory, timeout=0):
        with pytest.raises(OSError):
            replacement.run()
    owner = module.document(setup.supervisor.directory / "owner.json")
    assert owner["generation"] == old_generation
    assert replacement.worker is None and not setup.native.calls


def test_status_returns_newest_messages_in_creation_order(setup, capsys):
    supervisor = setup.supervisor
    for number in range(25):
        supervisor.queue_message(f"hash-order-{25 - number}", str(number))
        setup.clock[0] += 1
    assert module.main(["status", "--campaign", str(setup.root)]) == 0
    messages = json.loads(capsys.readouterr().out)["messages"]
    assert [item["message"] for item in messages] == [str(number) for number in range(5, 25)]


def test_large_status_projection_preserves_limits_and_lifecycle(setup):
    supervisor = setup.supervisor
    campaign = set_goal(setup, budgets={"spend_limit_usd": 10})
    setup.native.status.update(
        campaign={"active": True, "deadline": campaign["goal"]["deadline"],
                  "policy": {"deadline": campaign["goal"]["deadline"], "spend_limit_usd": 10}},
        lifecycle={"action": "stop"},
        bulk={str(index): "x" * 2000 for index in range(48)},
        other_bulk={str(index): "y" * 2000 for index in range(48)},
    )
    supervisor.observe()
    saved = supervisor.state()
    assert saved["observations"]["R1"]["campaign"]["policy"]["spend_limit_usd"] == 10
    assert saved["observations"]["R1"]["lifecycle"]["action"] == "stop"
    assert saved["resource_issues"] == []
    supervisor.watchdog()
    assert not any("request" in command for command in setup.native.calls)
