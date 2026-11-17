# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervision: status, steering, stop intent, checkpoints, run capacity, finalizing, monitoring."""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import cli, config, supervise_cli
from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.harness import claude, codex
from autofa.harness.claude import TurnResult
from autofa.mcp import runtime, server, tools_library
from autofa.mcp.runtime import load_runtime_from_run
from autofa.mcp.tools_eval import _evaluation_context, _llm_eval_settings
from autofa.session import (
    detector_service,
    finalization,
    polishing_guard,
    report_highlights,
    supervision,
    validation_completion,
    validation_summary,
)
from autofa.session.admission import paid_dispatch
from autofa.session.campaign_policy import CampaignBoundaryReached
from autofa.session.control import verify_fence
from autofa.session.guidance import guidance_prompt
from autofa.session.managed_launch import enforce_supervision, supervision_available
from autofa.session.recovery import read_json, run_lock
from autofa.session.run_capacity import (
    SETTINGS_FILE,
    apply_run_capacity,
    effective_capacity,
    set_run_capacity,
)
from autofa.session.status_log import read_status
from autofa.util import run_copy
from autofa.util.atomic import atomic_write_json as save
from autofa.util.cancellation import (
    Cancellation,
    cancellation_guard,
    cancellation_scope,
    check_cancelled,
    current_cancellation,
)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    directory = tmp_path / "runs" / "example"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({
        "run_id": "example", "round": 4, "max_rounds": 14,
    }))
    return directory


@pytest.fixture
def protocol(run_dir, monkeypatch):
    """A synthetic runtime for MCP tool calls, with unrelated native side effects disabled."""
    rt = SimpleNamespace(run=SimpleNamespace(run_id="example", run_dir=lambda: run_dir))
    monkeypatch.setattr(runtime, "get_runtime", lambda: rt)
    monkeypatch.setattr(polishing_guard, "polishing_tool_guard", lambda *a: None)
    monkeypatch.setattr(server, "_try_record_tool_exposure", lambda *a: None)
    monkeypatch.setattr(server, "_with_discovery_recovery", lambda name, result: result)
    monkeypatch.setattr(supervision, "save_quality_snapshot", lambda *a, **kw: None)
    return rt


def call_tool(monkeypatch, name, function, **kwargs):
    """Register a synthetic native tool, invoke it over MCP and return the wire text."""
    monkeypatch.setitem(server.TOOL_CALLABLES, name, function)
    return asyncio.run(server.invoke_tool(name, {}, **kwargs))["content"][0]["text"]


def load_saved_run(monkeypatch, run, rt=None):
    monkeypatch.setattr(config.RunConfig, "load", lambda value: run)
    monkeypatch.setattr(config, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(runtime, "load_runtime_from_run",
                        lambda *a: rt or SimpleNamespace(run=run))


def fake_harness(name, monkeypatch):
    """A discovery harness without its CLI; returns it and the prompts of its turns."""
    cls = {"claude": claude.ClaudeHarness, "codex": codex.CodexHarness}[name]
    harness = cls.__new__(cls)
    harness.model, harness.cli_path, harness.effort = "model", "/fake", "high"
    monkeypatch.setattr("autofa.ui.menus.prepare_step", lambda: None)
    if name == "claude":
        monkeypatch.setattr(claude, "describe_auth_mode", lambda *_: "fake")
    turns = []

    async def run_turn(text, **kwargs):
        turns.append(text)
        return TurnResult(session_id="synthetic")

    monkeypatch.setattr(harness, "run_turn", run_turn)
    return harness, turns


# Status


def test_status_reads_saved_state_without_runtime_or_writes(run_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "load_runtime_from_run", lambda *a, **kw: pytest.fail("runtime"))
    (run_dir / "validation_summary.json").write_text(json.dumps({
        "grade_counts": {"GOOD": 3}, "final_validation": {"n_validated_failure_modes": 1},
    }))
    before = {str(p): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    status = supervision.run_status(supervision.resolve_run_dir("example"))
    assert status["round"] == 4
    assert status["planned_rounds"] == 14
    assert status["grade_counts"] == {"GOOD": 3}
    assert status["final_validation"]["n_validated_failure_modes"] == 1
    assert status["snapshots"]["coverage_summary"]["record"] == {}
    assert before == {str(p): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        supervision.resolve_run_dir("../example")
    # A symlinked runs root is not mistaken for a path escape.
    other = tmp_path / "other-home"
    other.mkdir()
    (other / "runs").symlink_to(run_dir.parent, target_is_directory=True)
    monkeypatch.setenv("AUTOFA_HOME", str(other))
    assert supervision.resolve_run_dir("example") == run_dir.resolve()


def test_status_tail_handles_large_unicode_and_incomplete_records(run_dir, monkeypatch):
    path = run_dir / "status_log.jsonl"
    expected = {"message": "chart 🖼 " * 30000, "source": "agent"}
    path.write_bytes(b'{"old":true}\n' + json.dumps(expected, ensure_ascii=False).encode()
                     + b'\n{"incomplete":')
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("whole-file read"))
    assert read_status(run_dir, limit=1) == [expected]


def test_status_sanitizes_legacy_provider_errors_without_rewriting_evidence(run_dir):
    message = "Provider failed: Authorization: Bearer private-token-value"
    research = {"state": "stopped", "error": message, "finalization_error": message}
    publication = {"state": "failed", "error": {"message": message, "status_code": 401}}
    research_path = run_dir / "research_status.json"
    publication_path = run_dir / "publication.json"
    research_path.write_text(json.dumps(research))
    publication_path.write_text(json.dumps(publication))
    before = (research_path.read_bytes(), publication_path.read_bytes())
    status = supervision.run_status(run_dir)
    assert "private-token-value" not in json.dumps(status)
    assert "[redacted]" in status["research"]["error"]
    assert status["snapshots"]["publication"]["record"]["error"]["status_code"] == 401
    assert (research_path.read_bytes(), publication_path.read_bytes()) == before


def test_status_bounds_large_snapshots_and_full_mode_is_explicit(run_dir):
    value = {"history": [{"payload": "x" * 1000}] * 3000, "state": "complete"}
    path = run_dir / "round_validation.json"
    path.write_text(json.dumps(value))
    status = supervision.run_status(run_dir)
    snapshot = status["snapshots"]["round_validation"]
    assert len(json.dumps(status)) < 20000
    assert snapshot["size_bytes"] == path.stat().st_size
    assert snapshot["record"]["history"]["count"] == 3000
    assert snapshot["sha256"]
    assert supervision.run_status(run_dir, full_snapshots=True)["snapshots"][
        "round_validation"]["record"] == value


def test_dead_operation_owner_is_interrupted_and_pruned_but_foreign_owner_kept(
    run_dir, monkeypatch,
):
    directory = run_dir / "supervision" / "active"
    directory.mkdir(parents=True)
    dead = {"id": "dead", "tool": "judge_function", "state": "running", "pid": 987654321,
            "hostname": socket.gethostname()}
    (directory / "dead.json").write_text(json.dumps(dead))
    (directory.parent / "operation.json").write_text(json.dumps(dead))

    def kill(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(supervision.os, "kill", kill)
    status = supervision.run_status(run_dir)
    assert status["active_operations"] == []
    assert status["stale_operations"][0]["state"] == "interrupted"
    assert status["active_operation"]["state"] == "interrupted"
    foreign = {**dead, "id": "foreign", "hostname": "another-machine.invalid"}
    (directory / "foreign.json").write_text(json.dumps(foreign))
    with run_lock(run_dir):
        assert not (directory / "dead.json").exists()
        assert (directory / "foreign.json").exists()


def test_operation_owner_identity_survives_timezone_and_locale_changes(run_dir, monkeypatch):
    supervision.observe_operation(
        run_dir, "live", {"tool": "evaluate_functions", "state": "running", "started_at": "start"}
    )
    record = json.loads((run_dir / "supervision" / "operation.json").read_text())
    assert isinstance(record["process_started_at"], int)
    monkeypatch.setenv("TZ", "Pacific/Honolulu")
    monkeypatch.setenv("LC_ALL", "fr_FR.UTF-8")
    assert supervision._observed_operation(record)["liveness"] == "owner_present"
    record["hostname"] = "another-machine.invalid"
    assert supervision._observed_operation(record)["liveness"] == "remote_owner"


# Steering


def test_discovery_steering_is_idempotent_and_never_changes_judge_context(protocol, run_dir):
    run = SimpleNamespace(analysis_focus="model", reference_data=[], run_dir=lambda: run_dir,
                          skill_snapshots=[
                              {"name": "harness", "path": "reference", "sha256": "a",
                               "text": "shared-harness-contract"},
                              {"name": "operation", "path": "note", "sha256": "b",
                               "text": "private-steering", "audience": "discovery"},
                          ])
    judge = guidance_prompt(run, audience="judge")
    # Legacy snapshots without an audience stay shared; discovery-only ones do not.
    assert "shared-harness-contract" in judge and "private-steering" not in judge
    assert "private-steering" in guidance_prompt(run, audience="discovery")
    receipt = supervision.submit_steering(run_dir, "unique-supervisor-note", receipt_id="request1")
    assert receipt["state"] == "queued"
    assert supervision.submit_steering(
        run_dir, "unique-supervisor-note", receipt_id="request1") == receipt
    with pytest.raises(ValueError, match="different"):
        supervision.submit_steering(run_dir, "changed instruction", receipt_id="request1")
    assert "unique-supervisor-note" not in guidance_prompt(run, audience="discovery")
    assert "unique-supervisor-note" in supervision.steering_prompt(run_dir)
    assert supervision.steering_prompt(run_dir) == ""
    assert supervision.run_status(run_dir)["steering"][0]["state"] == "delivered"
    assert supervision.respond_to_steering("missing", "answer")["ok"] is False
    answer = "Investigated uncovered tasks; retaining existing boundaries"
    first = supervision.respond_to_steering("request1", answer)
    assert first["ok"] and first["receipt"]["state"] == "addressed"
    assert supervision.respond_to_steering("request1", answer) == first
    assert supervision.respond_to_steering("request1", "Different")["ok"] is False
    assert "unique-supervisor-note" not in guidance_prompt(run, audience="discovery")
    assert guidance_prompt(run, audience="judge") == judge


def test_short_supervision_lock_waits_for_writer(run_dir):
    entered = threading.Event()

    def owner():
        with supervision.supervision_lock(run_dir):
            entered.set()
            time.sleep(.1)

    thread = threading.Thread(target=owner)
    thread.start()
    assert entered.wait(2)
    receipt = supervision.submit_steering(run_dir, "queue while another receipt is saving")
    thread.join()
    assert receipt["state"] == "queued"


def test_large_queued_note_does_not_block_a_later_note_that_fits(run_dir):
    supervision.submit_steering(run_dir, "a" * 1500, receipt_id="first")
    supervision.submit_steering(run_dir, "Short note", receipt_id="second")
    notes = supervision.discovery_steering(run_dir, deliver=True, max_chars=400)
    assert [note["id"] for note in notes] == ["second"]
    records = {note["id"]: note for note in supervision.steering_records(run_dir)}
    assert records["first"]["state"] == "queued"
    assert records["second"]["state"] == "delivered"


def test_turn_steering_excerpt_retains_path_to_complete_instruction(run_dir):
    supervision.submit_steering(run_dir, "x" * 17000, receipt_id="long")
    prompt = supervision.steering_prompt(run_dir)
    assert str(run_dir / "supervision" / "steering" / "long.json") in prompt
    assert "read this file before acting" in prompt
    assert len(prompt) < 2500
    assert supervision.steering_prompt(run_dir) == ""


def test_steering_contention_preserves_committed_tool_result(protocol, run_dir, monkeypatch):
    supervision.submit_steering(run_dir, "Inspect uncovered errors", receipt_id="pending")
    calls = []

    def commit():
        calls.append("committed")
        return {"ok": True, "saved": "unchanged-evidence"}

    with run_lock(run_dir / "supervision"):
        text = call_tool(monkeypatch, "synthetic_commit", commit)
    assert calls == ["committed"]
    assert "unchanged-evidence" in text
    assert supervision.run_status(run_dir)["steering"][0]["state"] == "queued"


def test_tool_result_serializes_datetime_without_losing_completed_work(
    protocol, run_dir, monkeypatch,
):
    # A queued note makes the steering budget serialize the result before the wire does.
    supervision.submit_steering(run_dir, "Inspect uncovered errors", receipt_id="queued")
    when = datetime(2026, 1, 2, tzinfo=UTC)
    data = {"ok": True, "message": {"timestamp": when}, "path": Path("sample")}
    payload = json.loads(call_tool(monkeypatch, "synthetic_read", lambda: data))
    assert payload["ok"] and payload["message"]["timestamp"] == str(when)
    assert payload["path"] == "sample"
    assert payload["supervisor_steering"]


def test_unicode_steering_is_delivered_within_wire_budget(protocol, run_dir, monkeypatch):
    message = "🧭" * 1500  # Astral characters have the largest escaped JSON size.
    supervision.submit_steering(run_dir, message, receipt_id="unicode")
    text = call_tool(monkeypatch, "synthetic_read", lambda: {"ok": True, "data": "x" * 30000})
    assert json.loads(text)["supervisor_steering"][0]["message"] == message
    assert len(text) < server.MCP_HARD_CEILING_CHARS


@pytest.mark.parametrize("caller", ["native", "driver"])
def test_large_steering_is_bounded_delivered_once_and_not_by_supervisor(
    protocol, run_dir, monkeypatch, caller,
):
    supervision.submit_steering(run_dir, "x" * 17000, receipt_id="long")
    text = call_tool(monkeypatch, "synthetic_read", lambda: {"ok": True, "data": "z" * 30000},
                     caller=caller)
    payload = json.loads(text)
    assert payload["data"] == "z" * 30000
    assert len(text) < server.MCP_HARD_CEILING_CHARS
    if caller == "native":
        assert payload["supervisor_steering"][0]["path"].endswith("long.json")
        assert supervision.steering_prompt(run_dir) == ""
    else:
        assert "supervisor_steering" not in payload
        assert supervision.steering_records(run_dir)[0]["state"] == "queued"


def test_service_guidance_is_budgeted_before_steering_delivery(protocol, run_dir, monkeypatch):
    health = {"infrastructure_blocked": True, "next_action": "recover " + "x" * 1200}
    monkeypatch.setattr(detector_service, "detector_service_status", lambda *a: health)
    supervision.submit_steering(run_dir, "Inspect " + "x" * 1000, receipt_id="pending")
    result = {"ok": True, "data": "z" * (server.MCP_HARD_CEILING_CHARS - 2000)}

    text = call_tool(monkeypatch, "synthetic_read", lambda: result)
    payload = json.loads(text)
    assert len(text) < server.MCP_HARD_CEILING_CHARS
    assert payload["data"] == result["data"]
    assert payload["detector_service"] == health
    assert "supervisor_steering" not in payload
    assert supervision.steering_records(run_dir)[0]["state"] == "queued"

    result["data"] = "small result"
    payload = json.loads(call_tool(monkeypatch, "synthetic_read", lambda: result))
    assert payload["detector_service"] == health
    assert payload["supervisor_steering"][0]["id"] == "pending"
    assert supervision.steering_records(run_dir)[0]["state"] == "delivered"


def test_judge_citation_presentation_is_budgeted_before_steering_delivery(
    protocol, run_dir, monkeypatch,
):
    monkeypatch.setattr(runtime, "_RUNTIME", protocol)
    message = "Inspect " + "x" * 1000
    supervision.submit_steering(run_dir, message, receipt_id="pending")
    offered = {"id": "pending", "message": message, "truncated": False,
               "path": str(run_dir / "supervision/steering/pending.json")}
    citation = {"quote": "λ" * 90}
    result = {"ok": True, "verdicts": [{"citations": [citation, dict(citation)]}],
              "padding": ""}

    def size(value):
        return len(json.dumps(value, ensure_ascii=False))

    # The note fits the unpresented result; lossless Unicode citation referencing
    # adds metadata that must be counted before the note is marked delivered.
    result["padding"] = "z" * (server.MCP_HARD_CEILING_CHARS - size(result) - size(offered) - 514)
    presentation = tools_library.reference_review_response(result, verdict_path=("verdicts",))
    assert size(presentation) > size(result) + 512
    text = call_tool(monkeypatch, "judge_function", lambda: result)
    payload = json.loads(text)
    assert len(text) < server.MCP_HARD_CEILING_CHARS
    assert payload["padding"] == result["padding"]
    assert payload["citation_references"]["count"] == 1
    assert payload["verdicts"][0]["citations"][1] == {"$ref": "#/verdicts/0/citations/0"}
    assert "supervisor_steering" not in payload
    assert supervision.steering_records(run_dir)[0]["state"] == "queued"

    result["padding"] = "small result"
    payload = json.loads(call_tool(monkeypatch, "judge_function", lambda: result))
    assert payload["supervisor_steering"][0]["id"] == "pending"
    assert supervision.steering_records(run_dir)[0]["state"] == "delivered"


@pytest.mark.parametrize("harness_name", ["claude", "codex"])
def test_compact_does_not_consume_or_rewrite_queued_steering(run_dir, monkeypatch, harness_name):
    harness, turns = fake_harness(harness_name, monkeypatch)
    supervision.submit_steering(run_dir, "Follow the uncovered cases", receipt_id="queued")
    for prompt in ("/compact", "Continue"):
        asyncio.run(harness.run_session(cwd=str(run_dir), system_prompt="native",
                                        initial_prompt=prompt, exit_after_discovery=True))
        if prompt == "/compact":
            assert turns == ["/compact"]
            assert supervision.steering_records(run_dir)[0]["state"] == "queued"
    assert "Follow the uncovered cases" in turns[1]
    assert supervision.steering_records(run_dir)[0]["state"] == "delivered"


@pytest.mark.parametrize("harness_name", ["codex", "claude"])
def test_unanswered_note_does_not_restart_finished_batch(run_dir, monkeypatch, harness_name):
    harness, turns = fake_harness(harness_name, monkeypatch)
    supervision.submit_steering(run_dir, "Inspect uncovered errors", receipt_id="pending")
    stops = []

    def continuation():
        stops.append("budget checked")

    async def validation():
        return None

    asyncio.run(harness.run_session(
        cwd=str(run_dir), system_prompt="native protocol",
        initial_prompt="Discover recurring failures",
        discovery_continuation=continuation, turn_validation=validation,
        exit_after_discovery=True))
    assert len(turns) == 1 and "Inspect uncovered errors" in turns[0]
    assert stops == ["budget checked"]
    assert supervision.steering_records(run_dir)[0]["state"] == "delivered"


# Lifecycle intent and cancellation


def test_stop_intent_survives_status_and_requires_explicit_release(run_dir):
    supervision.lifecycle_request(run_dir, "stop")
    assert supervision.run_status(run_dir)["lifecycle"]["action"] == "stop"
    for phase in ("discovery", "validation"):
        with pytest.raises(KeyboardInterrupt):
            supervision.check_lifecycle(run_dir, phase=phase)
    supervision.lifecycle_request(run_dir, "finalize")
    with pytest.raises(KeyboardInterrupt):
        supervision.check_lifecycle(run_dir)
    supervision.check_lifecycle(run_dir, phase="validation")
    supervision.lifecycle_request(run_dir, "release")
    supervision.check_lifecycle(run_dir)


def test_nested_guard_finishes_checkpoint_before_polling_stop():
    stop = False
    token = Cancellation(lambda: stop)
    writes = []
    with cancellation_scope(token):
        with cancellation_guard():
            writes.append("validation")
            stop = True
            token._last_poll = 0
            with cancellation_guard():
                check_cancelled()
                writes.append("library")
        with pytest.raises(asyncio.CancelledError):
            check_cancelled()
    assert writes == ["validation", "library"]
    assert token.supervisor_requested


@pytest.mark.parametrize("completed", [False, True])
def test_supervisor_stop_keeps_tool_protocol_alive(protocol, run_dir, monkeypatch, completed):
    def mutation():
        supervision.lifecycle_request(run_dir, "stop")
        if not completed:
            current_cancellation()._last_poll = 0
            check_cancelled()
        return {"ok": True, "committed": True}

    payload = json.loads(call_tool(monkeypatch, "remove_function", mutation))
    if completed:
        assert payload["committed"]
    else:
        assert payload["status"] == "stopped_by_supervisor"
    with pytest.raises(supervision.SupervisorInterrupted, match="supervise request.*release"):
        supervision.check_lifecycle(run_dir)


@pytest.mark.parametrize("operation", ["highlight", "finalize"])
def test_released_stop_remains_supervisor_cancellation(protocol, run_dir, monkeypatch, operation):
    def stopped(*a, **kw):
        current_cancellation().request_supervisor_stop("pause")
        supervision.lifecycle_request(run_dir, "release")
        raise asyncio.CancelledError

    if operation == "highlight":
        monkeypatch.setattr(report_highlights, "_prepare_report_highlights", stopped)
        assert report_highlights.prepare_report_highlights(protocol) == {
            "ok": False, "status": "pending", "reason": "supervisor_stop",
        }
    else:
        monkeypatch.setattr(finalization, "_evaluate_final_library", stopped)
        with pytest.raises(supervision.SupervisorInterrupted) as caught:
            finalization.evaluate_final_library(None)
        assert caught.value.action == "pause"


def test_highlight_error_fallback_honors_racing_supervisor_stop(run_dir, monkeypatch):
    rt = SimpleNamespace(run=SimpleNamespace(run_dir=lambda: run_dir))

    def fail(*args, **kwargs):
        raise ValueError("provider failed")

    def stop_during_error_receipt(rt):
        supervision.lifecycle_request(run_dir, "stop")
        token._last_poll = 0
        return {"status": "pending"}

    monkeypatch.setattr(report_highlights, "_prepare_report_highlights", fail)
    monkeypatch.setattr(report_highlights, "_receipt", stop_during_error_receipt)
    token = Cancellation(lambda: supervision.stop_requested(run_dir))
    with cancellation_scope(token):
        result = report_highlights.prepare_report_highlights(rt)
    assert result == {"ok": False, "status": "pending", "reason": "supervisor_stop"}


@pytest.mark.parametrize("already_requested", [False, True])
def test_async_finalization_surfaces_supervisor_intent_without_tearing_down_worker(
    protocol, run_dir, monkeypatch, already_requested,
):
    if already_requested:
        supervision.lifecycle_request(run_dir, "pause")
    else:
        def stop(console):
            raise supervision.SupervisorInterrupted(run_dir, "pause")

        monkeypatch.setattr(finalization, "evaluate_final_library", stop)
    with pytest.raises(supervision.SupervisorInterrupted) as caught:
        asyncio.run(finalization.finalize_discovery(None))
    assert caught.value.action == "pause"


def test_one_shot_supervisor_stop_has_nonzero_exit(run_dir, monkeypatch):
    def stopped(*args):
        raise supervision.SupervisorInterrupted(run_dir, "stop")

    monkeypatch.setattr(cli, "_dispatch", stopped)
    with pytest.raises(SystemExit) as caught:
        cli._run_command("resume", SimpleNamespace(once=True), interactive=False, status_lines=[])
    assert caught.value.code == 1


def test_supervisor_cli_reports_saved_stop_as_json(run_dir, capsys):
    supervision.lifecycle_request(run_dir, "stop")
    assert supervise_cli.main(["finalize", "--run", "example"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "stopped_by_supervisor"
    assert result["release_command"] == "autofa supervise request --run example release"


# Checkpoints


def test_checkpoint_includes_committed_wal_and_respects_run_lock(run_dir, tmp_path):
    database = run_dir / "predictions.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE predictions (item TEXT, value INTEGER)")
        conn.execute("INSERT INTO predictions VALUES ('original', 1)")
        conn.commit()
        destination = tmp_path / "checkpoint"
        receipt = supervision.checkpoint_run(run_dir, destination)
        with sqlite3.connect(destination / database.name) as copied:
            assert copied.execute("SELECT * FROM predictions").fetchall() == [("original", 1)]
        assert "predictions.sqlite" in receipt["files"]
        assert not list(destination.glob("*-wal"))
    with run_lock(run_dir), pytest.raises(RuntimeError, match="already active"):
        supervision.checkpoint_run(run_dir, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_checkpoint_does_not_block_steering_or_stop(run_dir, tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = run_copy.copy_run_artifacts

    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(run_copy, "copy_run_artifacts", delayed)
    thread = threading.Thread(
        target=supervision.checkpoint_run, args=(run_dir, tmp_path / "copy")
    )
    thread.start()
    try:
        assert entered.wait(2)
        assert supervision.submit_steering(run_dir, "New instruction")["state"] == "queued"
        assert supervision.lifecycle_request(run_dir, "pause")["action"] == "pause"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert (tmp_path / "copy" / "checkpoint.json").is_file()


def test_checkpoint_keeps_legacy_receipt_but_omits_staging_payloads_and_credentials(
    run_dir, tmp_path,
):
    state = "uploading"
    key, stale = "a" * 64, "b" * 64
    for name in (key, stale):
        directory = run_dir / "publication-artifacts" / name
        directory.mkdir(parents=True)
        (directory / "failure_cards.html").write_text("immutable report")
    (run_dir / "publication_intent.json").write_text(json.dumps({"state": state, "key": key}))
    (run_dir / ".env").write_text("private credential")
    target = tmp_path / "copy"
    supervision.checkpoint_run(run_dir, target)
    assert not (target / "publication-artifacts").exists()
    assert not (target / ".env").exists()
    assert json.loads((target / "publication_intent.json").read_text())["state"] == state


def test_failed_symlink_checkpoint_can_retry_same_destination(run_dir, tmp_path):
    target = tmp_path / "copy"
    link = run_dir / "linked-data"
    link.symlink_to(tmp_path / "external-data")
    with pytest.raises(ValueError, match="symlink"):
        supervision.checkpoint_run(run_dir, target)
    assert not target.exists()
    link.unlink()
    supervision.checkpoint_run(run_dir, target)
    assert (target / "checkpoint.json").is_file()


# Finalization and native invocation


def test_finalize_refreshes_coverage_after_commit_despite_recent_snapshot(run_dir, monkeypatch):
    """Finalization status reflects the last committed scores, including incomplete results."""
    run = SimpleNamespace(round=4, run_dir=lambda: run_dir, harness_settings={})
    rt = SimpleNamespace(run=run, matches={"GOOD": 2, "OK": 3})
    (run_dir / "library.json").write_text(json.dumps(rt.matches))
    completion = {"n_validated_failure_modes": 1, "minimum_met": False}
    outcome = {"state": "completed_with_gaps", "retained_results_complete": True,
               "review": {"unresolved": [{"name": "pending"}]}}
    load_saved_run(monkeypatch, run, rt)
    # Both writes occur inside the normal thirty-second throttle window.
    monkeypatch.setattr(supervision.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(validation_completion, "validated_good_target_for_runtime",
                        lambda _: completion)

    def save_summary(current, *, completion):
        (run_dir / "validation_summary.json").write_text(json.dumps({
            "final_validation": completion,
        }))

    def coverage(current, *, cached_only, minimum_grade):
        assert cached_only is True  # Observation never requests another corpus scan.
        assert current.matches == json.loads((run_dir / "library.json").read_text())
        return {"n_matched_failed_attempts": current.matches[minimum_grade]}

    monkeypatch.setattr(validation_summary, "save_validation_summary", save_summary)
    monkeypatch.setattr(
        "autofa.metrics.library_quality_coverage.library_quality_coverage_for_runtime", coverage,
    )
    supervision.save_quality_snapshot(rt)
    assert supervision.run_status(run_dir)["snapshots"]["coverage_summary"]["record"][
        "good_or_ok"]["n_matched_failed_attempts"] == 3

    def validate(console):
        # Native validation and the library save both run under the run lock.
        with pytest.raises(RuntimeError, match="already active"), run_lock(run_dir):
            pass
        rt.matches = {"GOOD": 4, "OK": 6}
        return outcome

    def save_library(**kwargs):
        with pytest.raises(RuntimeError, match="already active"), run_lock(run_dir):
            pass
        (run_dir / "library.json").write_text(json.dumps(rt.matches))

    monkeypatch.setattr(finalization, "evaluate_final_library", validate)
    monkeypatch.setattr(tools_library, "library_save", save_library)
    receipt = supervision.finalize_run("example")
    status = supervision.run_status(run_dir)
    coverage_record = status["snapshots"]["coverage_summary"]["record"]
    assert coverage_record["good"]["n_matched_failed_attempts"] == 4
    assert coverage_record["good_or_ok"]["n_matched_failed_attempts"] == 6
    assert receipt["validation"] == completion == status["final_validation"]
    assert receipt["outcome"] == outcome
    assert receipt["state"] == "returned"
    assert receipt["report_ready"] is False
    assert receipt["round"] == run.round == 4
    assert not list(run_dir.glob("*.html"))


def test_finalization_cancellation_receipt_is_interrupted(run_dir, monkeypatch):
    load_saved_run(monkeypatch, SimpleNamespace(round=4, run_dir=lambda: run_dir,
                                                harness_settings={}))

    def cancel(console):
        raise asyncio.CancelledError("native cooperative stop")

    monkeypatch.setattr(finalization, "evaluate_final_library", cancel)
    with pytest.raises(asyncio.CancelledError):
        supervision.finalize_run("example")
    assert json.loads((run_dir / "finalization.json").read_text())["state"] == "interrupted"


def test_configured_workers_are_not_silently_capped_at_32():
    cfg = Config(llm_eval_workers=56, llm_eval_max_workers=64, llm_eval_queue_mib=128)
    assert _llm_eval_settings(cfg)[2] == 56
    cfg.llm_eval_workers = 96
    assert _llm_eval_settings(cfg)[2] == 64
    with pytest.raises(ValueError, match="llm_eval_queue_mib"):
        Config(llm_eval_queue_mib=0)


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.delenv("AUTOFA_CONFIG", raising=False)
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"model_id": "tested", "benchmark": "bench", "problem_id": "one", "run_idx": 0,
         "response_text": "The required operation failed.", "is_correct": False},
    ]), source)
    cfg = Config(llm_eval_workers=4, llm_eval_max_workers=16, llm_eval_queue_mib=64,
                 judge_workers=2)
    run = RunConfig.create([source], "tested", ["bench"], cfg)
    library = FunctionLibrary(functions=[FeatureFunction(
        name="failure", description="Required operation failed",
        source="def detect(item): return True",
        scores={"evaluated": True, "recall": .5, "evaluation_context": "historical"},
    )])
    library.save(run.run_dir() / "library.json")
    (tmp_path / "config.toml").write_text(
        '# retained global config\nllm_eval_workers=4\nllm_eval_max_workers=16\n'
        'llm_eval_queue_mib=64\njudge_workers=2\n')
    (run.run_dir() / "predictions.sqlite").write_bytes(b"untouched cache fixture")
    return run, cfg


def test_capacity_change_persists_only_scoped_receipt_and_is_idempotent(saved_run):
    run, cfg = saved_run
    directory = run.run_dir()
    original = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    global_path = directory.parent.parent / "config.toml"
    global_bytes = global_path.read_bytes()
    receipt = set_run_capacity(run.run_id, {"llm_eval_workers": 32, "llm_eval_queue_mib": 128,
                                           "judge_workers": 4})
    assert receipt["changed_fields"] == ["judge_workers", "llm_eval_queue_mib", "llm_eval_workers"]
    assert receipt["effective_values"]["effective_llm_eval_workers"] == 16
    assert receipt["effective_values"]["limiting_reason"] == "configured_ceiling"
    assert not receipt["global_config_changed"]
    assert {path: path.read_bytes() for path in original} == original
    assert global_path.read_bytes() == global_bytes
    before = (directory / SETTINGS_FILE).read_bytes()
    again = set_run_capacity(run.run_id, {"llm_eval_workers": 32})
    assert again["already_applied"] and not again["changed_fields"]
    assert (directory / SETTINGS_FILE).read_bytes() == before
    set_run_capacity(run.run_id, {"llm_eval_max_workers": 64})
    scoped = apply_run_capacity(cfg, run)
    assert effective_capacity(scoped)["effective_llm_eval_workers"] == 32
    assert scoped.llm_eval_queue_mib == 128 and scoped.judge_workers == 4
    assert cfg.llm_eval_workers == 4 and cfg.judge_workers == 2


def test_real_runtime_resume_honors_capacity_without_changing_score_identity(saved_run):
    run, cfg = saved_run
    baseline = asdict(cfg)
    initial = load_runtime_from_run(run, cfg)
    context = _evaluation_context(initial, routing_identity={"test": "unchanged"})
    original_source = initial.library.get("failure").source
    original_scores = dict(initial.library.get("failure").scores)
    set_run_capacity(run.run_id, {"llm_eval_workers": 12, "llm_eval_max_workers": 64,
                                 "llm_eval_queue_mib": 128, "judge_workers": 6})
    resumed = load_runtime_from_run(RunConfig.load(run.run_id), cfg)
    assert resumed.config.llm_eval_workers == 12
    assert resumed.config.llm_eval_queue_mib == 128
    assert resumed.config.judge_workers == 6
    assert _evaluation_context(resumed, routing_identity={"test": "unchanged"}) == context
    assert resumed.library.get("failure").source == original_source
    assert resumed.library.get("failure").scores == original_scores
    assert asdict(cfg) == baseline
    for field in ("feature_model", "feature_request_options", "llm_eval_sample",
                  "behavior_judge_model"):
        assert getattr(resumed.config, field) == getattr(cfg, field)


def test_active_run_rejects_tuning_and_bad_inputs_do_not_write(saved_run):
    run, _ = saved_run
    with run_lock(run.run_dir()), pytest.raises(RuntimeError, match="already active"):
        set_run_capacity(run.run_id, {"llm_eval_workers": 12})
    assert not (run.run_dir() / SETTINGS_FILE).exists()


@pytest.mark.parametrize("settings", [
    None, {"feature_model": "other"}, {"llm_eval_workers": True}, {"llm_eval_queue_mib": 4097},
])
def test_invalid_capacity_is_rejected_before_any_mutation(saved_run, settings):
    run, _ = saved_run
    before = {p: p.read_bytes() for p in run.run_dir().iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        set_run_capacity(run.run_id, settings)
    assert {p: p.read_bytes() for p in run.run_dir().iterdir() if p.is_file()} == before


def test_saved_capacity_identity_and_types_are_revalidated_on_resume(saved_run):
    run, cfg = saved_run
    path = run.run_dir() / SETTINGS_FILE
    path.write_text(json.dumps({"version": 1, "run_id": "other", "settings": {}, "events": []}))
    with pytest.raises(ValueError, match="do not belong"):
        apply_run_capacity(cfg, run)
    path.write_text(json.dumps({"version": 1, "run_id": run.run_id,
                               "settings": {"judge_workers": False}, "events": []}))
    with pytest.raises(ValueError, match="judge_workers"):
        apply_run_capacity(cfg, run)


@pytest.mark.parametrize('name,value', [('llm_eval_workers', 257), ('judge_workers', 9)])
def test_invalid_capacity_fails_during_config_load(tmp_path, monkeypatch, name, value):
    from autofa.config import load_config

    path = tmp_path / 'config.toml'
    path.write_text(f'{name} = {value}\n')
    monkeypatch.setenv('AUTOFA_CONFIG', str(path))
    monkeypatch.setenv('AUTOFA_HOME', str(tmp_path / 'profile'))
    with pytest.raises(ValueError, match=name):
        load_config()
    assert not (tmp_path / 'profile' / 'runs').exists()


def test_new_registered_native_tool_is_discoverable_without_a_catalog_edit(
    monkeypatch, tmp_path, capsys,
):
    def unfamiliar_workflow(source: str, count: int = 2) -> dict:
        """Inspect a newly registered workflow."""
        raise AssertionError("Inspection must not execute it")

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "absent"))
    monkeypatch.setitem(server.TOOL_CALLABLES, "unfamiliar_workflow", unfamiliar_workflow)
    assert supervise_cli.main(["tools", "--name", "unfamiliar_workflow"]) == 0
    result = json.loads(capsys.readouterr().out)["unfamiliar_workflow"]
    assert result["input_schema"]["required"] == ["source"]
    assert result["input_schema"]["properties"]["count"]["default"] == 2
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("ok", [True, False])
def test_invoke_cli_preserves_native_result_and_exit_status(run_dir, monkeypatch, capsys, ok):
    load_saved_run(monkeypatch, SimpleNamespace(run_dir=lambda: run_dir, harness_settings={}))
    monkeypatch.setitem(server.TOOL_CALLABLES, "new_workflow", lambda source: None)
    calls = []

    async def invoke(name, arguments, *, caller):
        assert caller == "driver"
        calls.append((name, arguments))
        return {"content": [{"type": "text", "text": json.dumps({
            "ok": ok, "receipt": "native-operation", "saved": True,
        })}]}

    monkeypatch.setattr(server, "invoke_tool", invoke)
    arguments = run_dir / "arguments.json"
    arguments.write_text(json.dumps({"source": "source-a"}))
    code = supervise_cli.main(["invoke", "--run", "example", "--tool", "new_workflow",
                               "--arguments-file", str(arguments)])
    assert code == (0 if ok else 1)
    assert calls == [("new_workflow", {"source": "source-a"})]
    result = json.loads(capsys.readouterr().out)
    assert json.loads(result["content"][0]["text"]) == {
        "ok": ok, "receipt": "native-operation", "saved": True,
    }


@pytest.mark.parametrize("action", ["finalize", "invoke"])
def test_supervision_restores_saved_judge_identity(run_dir, monkeypatch, action):
    run = SimpleNamespace(round=2, run_dir=lambda: run_dir,
                          harness_settings={"behavior_judge_harness": "claude",
                                            "behavior_judge_model": "saved-judge"})
    load_saved_run(monkeypatch, run)
    monkeypatch.setattr(config, "load_config", lambda: SimpleNamespace(
        behavior_judge_harness="codex", behavior_judge_model="global-judge"))
    observed = []

    def load(run, cfg):
        observed.append((cfg.behavior_judge_harness, cfg.behavior_judge_model))
        return SimpleNamespace(run=run)

    monkeypatch.setattr(runtime, "load_runtime_from_run", load)
    if action == "finalize":
        monkeypatch.setattr(finalization, "evaluate_final_library", lambda _: {})
        monkeypatch.setattr(tools_library, "library_save", lambda **kw: {})
        monkeypatch.setattr(supervision, "save_quality_snapshot", lambda *a, **kw: None)
        supervision.finalize_run("example")
    else:
        monkeypatch.setitem(server.TOOL_CALLABLES, "synthetic", lambda: {})

        async def invoke(*args, **kwargs):
            return {"ok": True}

        monkeypatch.setattr(server, "invoke_tool", invoke)
        arguments = run_dir / "args.json"
        arguments.write_text("{}")
        assert supervise_cli.main(["invoke", "--run", "example", "--tool", "synthetic",
                                   "--arguments-file", str(arguments)]) == 0
    assert observed == [("claude", "saved-judge")]


# Optional host monitoring never substitutes for native resource policy


@pytest.fixture
def managed(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    run = RunConfig(run_id="independent", parquet_paths=[], model_id="fixture")
    run.campaign_policy = {
        "mode": "improve_until",
        "deadline": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    run.save()
    root = tmp_path / "campaign"
    save(root / "campaign.json", {"id": "campaign", "revision": 0, "state": "active"})
    save(root / "supervision/config.json", {
        "supervision_mode": "host", "host": {"session_id": "first"},
        "review_seconds": 600, "host_grace_seconds": 30,
    })
    save(root / "supervision/owner.json", {
        "state": "running", "generation": "owner", "heartbeat_epoch": 1000,
    })
    binding = {"campaign_path": str(root), "campaign_id": "campaign", "launch_id": "launch"}
    save(run.run_dir() / "supervision/driver_launch.json", binding)
    return run, root, binding


def test_missing_host_and_guard_do_not_block_native_work_by_default(managed):
    run, root, binding = managed
    assert not supervision_available(binding)
    before = (run.run_dir() / "config.json").read_bytes()
    with paid_dispatch({"run_dir": str(run.run_dir()), "phase": "discovery", "role": "detector"}):
        pass
    assert (run.run_dir() / "config.json").read_bytes() == before
    assert not (run.run_dir() / "supervision/lifecycle.json").exists()
    assert read_json(root / "campaign.json")["state"] == "active"


def test_explicit_pause_on_loss_and_live_monitoring_pause(managed):
    run, root, _ = managed
    settings = read_json(root / "supervision/config.json")
    settings["on_host_loss"] = "pause"
    save(root / "supervision/config.json", settings)
    save(root / "supervision/monitoring.json", {"enabled": False})
    enforce_supervision(run.run_dir())
    assert not (run.run_dir() / "supervision/lifecycle.json").exists()
    save(root / "supervision/monitoring.json", {"enabled": True, "on_host_loss": "continue"})
    enforce_supervision(run.run_dir())
    save(root / "supervision/monitoring.json", {"enabled": True, "on_host_loss": "pause"})
    with pytest.raises(supervision.SupervisorInterrupted):
        enforce_supervision(run.run_dir())
    assert read_json(run.run_dir() / "supervision/lifecycle.json")["action"] == "pause"
    # Changing monitoring is not permission to erase a saved native pause.
    save(root / "supervision/monitoring.json", {"enabled": False})
    enforce_supervision(run.run_dir())
    assert read_json(run.run_dir() / "supervision/lifecycle.json")["action"] == "pause"


def test_liveness_respects_live_interval_and_new_host_session(managed):
    _, root, binding = managed
    save(root / "supervision/host-ack.json", {
        "generation": "owner", "revision": 0, "monitoring_revision": 0,
        "session_id": "first", "acknowledged_at_epoch": 500,
        "last_review_at_epoch": 500, "next_review_at_epoch": 1100,
    })
    assert supervision_available(binding, now=1000)
    save(root / "supervision/monitoring.json", {"review_seconds": 60})
    assert not supervision_available(binding, now=1000)
    save(root / "supervision/monitoring.json", {
        "review_seconds": 1200, "revision": 1, "host_session_id": "second",
    })
    assert not supervision_available(binding, now=1000)
    ack = read_json(root / "supervision/host-ack.json")
    ack.update(monitoring_revision=1, session_id="second")
    save(root / "supervision/host-ack.json", ack)
    assert supervision_available(binding, now=1000)


# Delivery phases share one admission rule; one of them represents the set.
@pytest.mark.parametrize("phase", ["discovery", "finalization"])
def test_explicit_campaign_stop_still_reaches_native_if_guard_is_gone(managed, phase):
    run, root, _ = managed
    campaign = read_json(root / "campaign.json")
    campaign["state"] = "stopped"
    save(root / "campaign.json", campaign)
    with pytest.raises(CampaignBoundaryReached, match="user_stop"):
        with paid_dispatch({"run_dir": str(run.run_dir()), "phase": phase}):
            pytest.fail("Explicit analysis stop was ignored")
    assert read_json(run.run_dir() / "supervision/lifecycle.json")["action"] == "stop"


# Deadline and spend limits share the native policy check; a saved stop is separate.
@pytest.mark.parametrize("boundary", ["deadline", "stop"])
def test_monitoring_absence_does_not_weaken_native_boundaries(managed, boundary):
    run, _, _ = managed
    if boundary == "deadline":
        run.campaign_policy["deadline"] = "2000-01-01T00:00:00Z"
    else:
        supervision.lifecycle_request(run.run_dir(), "stop")
    run.save()
    with pytest.raises(CampaignBoundaryReached):
        with paid_dispatch({"run_dir": str(run.run_dir()), "phase": "discovery"}):
            pytest.fail("Native resource or stop boundary was bypassed")


def test_stale_monitoring_fences_rejected_but_guard_can_finalize(managed):
    run, root, _ = managed
    fence = {"campaign_path": str(root), "revision": 0, "owner_generation": "owner"}
    verify_fence(fence)  # Compatible with an untouched legacy campaign.
    save(root / "supervision/monitoring.json", {"enabled": False, "revision": 1})
    with pytest.raises(ValueError, match="Monitoring"):
        verify_fence(fence)
    guard = {**fence, "scope": "guard"}
    assert supervision.lifecycle_request(
        run.run_dir(), "finalize", fence=guard)["action"] == "finalize"
    supervision.lifecycle_request(run.run_dir(), "stop")
    assert supervision.lifecycle_request(
        run.run_dir(), "finalize", fence=guard)["action"] == "stop"
    save(root / "supervision/monitoring.json", {"enabled": True, "revision": 2})
    with pytest.raises(ValueError, match="Monitoring"):
        verify_fence({**fence, "monitoring_revision": 1})
    verify_fence({**fence, "monitoring_revision": 2})
    with pytest.raises(ValueError, match="Superseded"):
        verify_fence({**guard, "owner_generation": "old"})
