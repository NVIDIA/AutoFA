# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review budgets, reconnects and bounded continuations keep the same frozen obligations."""

import asyncio
import itertools
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from autofa.metrics import claude_judge, codex_judge
from autofa.metrics.claim_calibration import attach_calibration
from autofa.metrics.judge_budget import (
    MAX_TEXT_CHARS,
    ReviewBudget,
    ReviewBudgetError,
    plan_review_budget,
)

from support.judge import (
    DESCRIPTION,
    FINAL,
    LOCAL_DESCRIPTION,
    QUOTE,
    SOURCE,
    _reply,
    _review,
    local_reply,
    local_review,
)

HARNESSES = ["claude", "codex"]


def codex_result(text, session="saved-thread"):
    return SimpleNamespace(
        text=text, thread_id=session, turn_id="turn", status="completed", error=None,
        error_code=None, retryable=False, usage={},
    )


def install_claude_tools(monkeypatch):
    import claude_agent_sdk as sdk

    callbacks = {}

    def tool(name, description, schema):
        def register(callback):
            callbacks[name] = callback
            return callback

        return register

    monkeypatch.setattr(sdk, "tool", tool)
    monkeypatch.setattr(sdk, "create_sdk_mcp_server", lambda **kwargs: {})
    monkeypatch.setattr("autofa.harness.claude_cli.ensure_claude_auth", lambda **kwargs: "claude")
    return sdk, callbacks


def install_harness(monkeypatch, harness, turn, sessions="saved-thread"):
    """Fake ``harness`` around ``turn`` and return its real judge session runner.

    Each model turn first announces its session handle (the next of ``sessions``; None
    announces nothing), then awaits ``turn(call, prompt, resume, allowance)``, where
    ``call(name, args)`` dispatches a real evidence tool and decodes its payload.
    """
    handles = itertools.repeat(sessions) if isinstance(sessions, str) else iter(sessions)
    if harness == "claude":
        sdk, callbacks = install_claude_tools(monkeypatch)

        async def query(*, prompt, options):
            messages = [message async for message in prompt]
            handle = next(handles)
            if handle:
                yield sdk.SystemMessage(subtype="init", data={"session_id": handle})

            async def call(name, args):
                return json.loads((await callbacks[name](args))["content"][0]["text"])

            text = await turn(call, messages[0]["message"]["content"], options.resume,
                              options.max_turns)
            yield sdk.ResultMessage(
                subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                num_turns=1, session_id=handle, stop_reason="end_turn", result=text,
            )

        monkeypatch.setattr(sdk, "query", query)
        return claude_judge._run_claude_judge_session

    async def run_turn(**kwargs):
        handle = next(handles)
        if handle:
            await kwargs["on_thread"](handle)

        async def call(name, args):
            return json.loads((await kwargs["call_tool"](name, args))["contentItems"][0]["text"])

        text = await turn(call, kwargs["prompt"], kwargs["resume"], kwargs["max_tool_calls"])
        return codex_result(text, handle)

    monkeypatch.setattr(codex_judge, "_run_turn", run_turn)
    return codex_judge.run_codex_judge_session


@pytest.fixture
def no_retry_sleep(monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(claude_judge.asyncio, "sleep", no_sleep)


def run_review(tmp_path, review, harness):
    return claude_judge._run_review(
        prompt="Frozen original review", cwd=tmp_path, model="offline", cli_path=None,
        review=review, max_turns=64, harness=harness,
    )


def read_all(review, *fields):
    for field in fields or review.fields:
        offset = 0
        while offset < len(review.fields[field]):
            offset = review.read(field, offset)["end"]


async def cite_and_reply(call, review):
    citation = await call("cite_evidence", {"field": "response_text", "quote": QUOTE})
    return _reply(review, unknown=False, citation_id=citation["citations"][0]["citation_id"])


def test_full_text_review_has_same_individual_tool_allowance(tmp_path, monkeypatch):
    review = _review(tmp_path, chars=600_000)
    plan = plan_review_budget(review)
    # This trace needs more calls than the 64-call baseline and still reads everything.
    assert plan.tool_call_limit > 64
    seen = []

    async def turn(call, prompt, resume, allowance):
        seen.append(allowance)
        await call("review_status", {})
        for field in review.fields:
            if (field != "detector_source"
                    and review.snapshot["manifest"][field]["origin"] != "stored"):
                continue
            offset = 0
            while offset < len(review.fields[field]):
                offset = (await call("read_evidence", {"field": field, "offset": offset}))["end"]
        await call("save_review_notes", {"notes": "Completed all frozen original scope reads."})
        return await cite_and_reply(call, review)

    runner = install_harness(monkeypatch, "claude", turn)
    raw = asyncio.run(runner(prompt="Review the frozen claim.", cwd=tmp_path, model="offline",
                             cli_path=None, review=review))
    parsed = claude_judge.parse_session_result(
        raw, [{"item_id": "case"}], source=SOURCE, reviews={"case": review},
    )
    assert parsed["verdicts"][0]["verdict"] == "yes"
    assert seen == [plan.tool_call_limit]
    assert review.session["budget_total_used"] == plan.required_read_calls + 3
    assert review.session["budget_remaining"] == plan.tool_call_limit - plan.required_read_calls - 3


@pytest.mark.parametrize("harness", HARNESSES)
def test_unaffordable_scope_fails_before_dispatch(tmp_path, monkeypatch, harness):
    review = _review(tmp_path, chars=MAX_TEXT_CHARS + 20_000)
    if harness == "codex":
        monkeypatch.setattr(codex_judge, "_run_turn", lambda **kwargs: pytest.fail("paid dispatch"))
        runner = codex_judge.run_codex_judge_session
    else:
        sdk, _ = install_claude_tools(monkeypatch)
        monkeypatch.setattr(sdk, "query", lambda **kwargs: pytest.fail("paid dispatch"))
        runner = claude_judge._run_claude_judge_session
    with pytest.raises(ReviewBudgetError, match="before model dispatch.*required_read_calls"):
        asyncio.run(runner(prompt="Frozen review", cwd=tmp_path, model="offline", cli_path=None,
                           review=review))
    assert not review.reads


def test_limits_are_shared_and_never_replenished_by_transport(tmp_path):
    review = _review(tmp_path)
    budget = ReviewBudget(review)
    budget.plan = replace(budget.plan, text_char_limit=100, final_char_limit=50)
    budget.charge_tool("read_evidence")
    budget.charge_text({"content": [{"type": "text", "text": "x" * 75}]})
    budget.replacement()
    assert budget.used == 1 and budget.text_chars == 75
    with pytest.raises(ReviewBudgetError, match="replacement limit"):
        budget.replacement()
    with pytest.raises(ReviewBudgetError, match="final-output limit"):
        budget.final_text("x" * 51)
    with pytest.raises(ReviewBudgetError, match="delivered-text"):
        budget.charge_text({"content": [{"type": "text", "text": "x" * 30}]})
    budget.text_chars = 0
    budget.started -= budget.plan.timeout_seconds + 1
    with pytest.raises(ReviewBudgetError, match="elapsed-time"):
        budget.charge_tool("read_evidence")


@pytest.mark.usefixtures("no_retry_sleep")
def test_transport_retry_preserves_reads_notes_and_charges_partial_tools(tmp_path, monkeypatch):
    harness = "codex"
    review = _review(tmp_path, chars=100)
    calls = []

    async def turn(call, prompt, resume, allowance):
        calls.append(resume)
        if len(calls) == 1:
            await call("read_evidence", {"field": "response_text"})
            await call("save_review_notes", {"notes": "Keep original recovery hypothesis."})
            raise ConnectionError("connection interrupted")
        assert resume == "saved-thread"
        assert review.epoch == 1
        assert review.covers("response_text", 0, len(review.fields["response_text"]))
        assert not review.covers(
            "response_text", 0, len(review.fields["response_text"]), current=True
        )
        assert review.notes == "Keep original recovery hypothesis."
        await call("read_evidence", {"field": "detector_source"})
        await call("read_evidence", {"field": "response_text"})
        return await cite_and_reply(call, review)

    install_harness(monkeypatch, harness, turn)
    raw = run_review(tmp_path, review, harness)
    assert "supported" in raw and calls == [None, "saved-thread"]
    assert review.session["budget_total_used"] == 5
    assert review.session["budget_remaining"] == 59
    assert review.session["budget_transport_retries"] == 1
    assert len(review.session["attempts"]) == 2


@pytest.mark.usefixtures("no_retry_sleep")
def test_transport_retries_stop_with_shared_allowance(tmp_path, monkeypatch):
    harness = "claude"
    review = _review(tmp_path, chars=100)
    calls = []

    async def turn(call, prompt, resume, allowance):
        calls.append(resume)
        await call("read_evidence", {"field": "response_text"})
        raise ConnectionError("connection interrupted")

    install_harness(monkeypatch, harness, turn)
    with pytest.raises(ConnectionError):
        run_review(tmp_path, review, harness)
    assert calls == [None, "saved-thread", "saved-thread"]
    events = [
        json.loads(line) for line in (review.directory / "audit.jsonl").read_text().splitlines()
    ]
    assert len([row for row in events if row["event"] == "review_budget"]) == 1
    assert len([row for row in events if row["event"] == "connection_retry"]) == 2
    assert len([row for row in events if row["event"] == "review_budget_tool"]) == 3
    assert not any(row["event"] == "isolated_replacement_budget" for row in events)


@pytest.mark.usefixtures("no_retry_sleep")
def test_missing_resume_handle_has_one_fresh_context_with_no_inherited_scope_reads(
    tmp_path, monkeypatch,
):
    harness = "codex"
    review = _review(tmp_path, chars=100)
    calls = []
    frozen = review.snapshot["snapshot_id"]

    async def turn(call, prompt, resume, allowance):
        calls.append(resume)
        if len(calls) == 1:
            await call("read_evidence", {"field": "response_text"})
            await call("save_review_notes", {"notes": "Read original evidence before outage."})
            raise ConnectionError("connection interrupted")
        if len(calls) == 2:
            assert resume == "saved-thread"
            raise RuntimeError("session not found")
        assert resume is None
        assert review.snapshot["snapshot_id"] == frozen
        assert review.notes == "Read original evidence before outage."
        assert not review.covers("response_text", 0, len(review.fields["response_text"]))
        await call("read_evidence", {"field": "detector_source"})
        await call("read_evidence", {"field": "response_text"})
        return await cite_and_reply(call, review)

    install_harness(monkeypatch, harness, turn, ["saved-thread", None, "replacement-thread"])
    raw = run_review(tmp_path, review, harness)
    assert "supported" in raw and calls == [None, "saved-thread", None]
    assert review.session["budget_total_used"] == 6
    assert review.session["budget_replacements"] == 1
    assert len(review.session["attempts"]) == 3


def test_elapsed_timeout_is_actionable_budget_failure(tmp_path, monkeypatch):
    review = _review(tmp_path, chars=100)
    calls = []

    async def timed_out(**kwargs):
        budget = kwargs["budget"]
        calls.append(budget)
        budget.begin_attempt()
        budget.started -= budget.plan.timeout_seconds + 1
        raise TimeoutError("SDK stream timed out")

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", timed_out)
    with pytest.raises(ReviewBudgetError, match="elapsed-time budget exhausted"):
        run_review(tmp_path, review, "claude")
    assert len(calls) == 1
    assert review.session["budget_total_used"] == 1


def test_repair_lists_all_artifact_prerequisites_independent_of_latest_error(tmp_path):
    review = _review(tmp_path, chars=100)
    # A frozen catalog is enough to prescribe work; constructing the repair must
    # not retrieve files, view images, or silently create read receipts.
    review.snapshot["artifact_manifest"] = {
        "catalog": [
            {"artifact_idx": 0, "source_kind": "retained_input_image", "mime_type": "image/png"},
            {"artifact_idx": 1, "mime_type": "application/pdf"},
            {"artifact_idx": 2, "mime_type": "text/plain"},
        ]
    }
    review.artifact_inspections[2] = {
        "status": "available",
        "segments": [{"locator": "text:1", "text": "Original data"}],
    }
    budget = ReviewBudget(review)
    # An unrelated latest error still prescribes every whole-trace artifact.
    prompt = claude_judge._verdict_repair_prompt(
        review, ["invalid structured judge reply"], budget=budget)
    navigation = json.loads(prompt.splitlines()[-1])
    artifacts = navigation["whole_trace_artifact_work"]
    assert [row["artifact_idx"] for row in artifacts] == [0, 1, 2]
    assert artifacts[0]["inspection_status"] == "uninspected"
    assert artifacts[2]["required_text_segments"] == [
        {"locator": "text:1", "start": 0, "end": len("Original data")},
    ]
    assert not review.reads and not review.artifact_reads and not review.citations
    assert set(review.artifact_inspections) == {2}


def test_both_review_wrappers_receive_identical_upfront_scope_and_budget_plan(
    tmp_path, monkeypatch
):
    prompts = []
    reviews = []

    async def capture(**kwargs):
        review = kwargs["review"]
        prompts.append(kwargs["prompt"])
        reviews.append(review)
        assert not review.reads and not review.artifact_reads and not review.citations
        assert kwargs["budget"].used == 0
        return "offline captured response"

    monkeypatch.setattr(claude_judge, "_run_claude_judge_session", capture)
    monkeypatch.setattr(codex_judge, "run_codex_judge_session", capture)
    for harness in HARNESSES:
        review = _review(tmp_path, chars=300_000, name=harness)
        review.snapshot["artifact_manifest"] = {"catalog": [
            {"artifact_idx": 0, "source_kind": "retained_input_image", "mime_type": "image/png"},
        ]}
        review.snapshot["claim_contract"]["requirements"].append({
            "id": "bounded", "required_review_scope": "bounded_interval",
            "review_interval": {"start": "run_start", "end": "run_end"},
        })
        assert claude_judge._run_review(
            prompt="Original unchanged frozen claim", review=review, max_turns=64,
            harness=harness,
        ) == "offline captured response"
    assert prompts[0] == prompts[1]
    assert prompts[0].startswith("Original unchanged frozen claim\n\n")
    plan = json.loads(prompts[0].splitlines()[-1])
    assert plan["remaining_tool_calls"] == plan_review_budget(reviews[0]).tool_call_limit == 64
    fields = {row["field"]: row for row in plan["required_complete_fields"]}
    assert fields["response_text"]["total_chars"] == 300_000
    assert fields["response_text"]["full_field_pages"] == 19
    assert fields["response_text"]["unread_page_calls"] == 19
    assert fields["detector_source"]["full_field_pages"] == 1
    assert plan["required_artifacts"][0]["artifact_idx"] == 0
    assert plan["bounded_requirements"][0]["requirement_id"] == "bounded"
    assert (reviews[0].directory / "execution-prompt.txt").read_text() == prompts[0]
    assert not reviews[0].reads and not reviews[1].artifact_reads


def test_upfront_local_plan_does_not_require_complete_unrelated_trace(tmp_path):
    # Local scope does not inherit the exhaustive cost of an unaffordable trace.
    review = _review(tmp_path, chars=MAX_TEXT_CHARS + 20_000, scope="local")
    plan = plan_review_budget(review)
    assert plan.scope == "local" and plan.tool_call_limit == 64
    assert plan.required_read_calls == 9  # Eight evidence pages plus complete detector.
    prompt = claude_judge._review_work_plan_prompt(review, ReviewBudget(review))
    plan = json.loads(prompt.splitlines()[-1])
    assert [row["field"] for row in plan["required_complete_fields"]] == ["detector_source"]
    assert plan["required_artifacts"] == [] and plan["whole_trace_requirements"] == []


# Bounded completion: mechanical review gaps are finished without forcing confirmations.


# Each mode runs once, alternating the two harness transports.
@pytest.mark.parametrize("harness,mode,expected_calls,expected_verdict", [
    ("claude", "complete", 2, "yes"),
    ("codex", "advancing_completion", 3, "yes"),
    ("claude", "still_incomplete", 2, "uncertain"),
    ("codex", "unavailable", 1, "uncertain"),
    ("claude", "mislabelled_unavailable", 2, "uncertain"),
    ("codex", "malformed_then_complete", 3, "yes"),
    ("claude", "no_budget", 1, "uncertain"),
])
def test_bounded_completion_same_session_both_harnesses(
    tmp_path, monkeypatch, harness, mode, expected_calls, expected_verdict,
):
    review = _review(tmp_path)
    calls = []
    max_turns = 8
    budget = ReviewBudget(review, max_turns)
    if mode == "no_budget":
        # Exercise actual exhaustion, not an intentionally inadequate baseline.
        budget.used = budget.limit - 1
    initial_remaining = budget.remaining

    async def turn(call, prompt, resume, allowance):
        calls.append((resume, allowance))
        if len(calls) == 1:
            assert resume is None
            review.read("detector_source")
            review.read("response_text", 0, 50)
            review.save_notes("Keep the original input-loss hypothesis and check later recovery.")
            if mode == "malformed_then_complete":
                return '{"verdicts":'
        else:
            assert resume == "saved-thread"
            assert "review_status" in prompt
            assert review.notes.startswith("Keep the original")
        if mode == "advancing_completion" and len(calls) == 2:
            assert "That evidence work is funded" in prompt
            review.read("response_text", 50, 50)
        if (mode in {"complete", "malformed_then_complete", "advancing_completion"}
                and len(calls) == expected_calls):
            assert "no_backup" in prompt
            assert ('"start": 100' if mode == "advancing_completion" else '"start": 50') in prompt
            read_all(review)
            return _reply(review, unknown=False)
        if mode == "unavailable":
            # This is genuinely unresolved after the frozen scope was read.
            read_all(review, "response_text")
        reason = {"unavailable": "evidence_unavailable",
                  "mislabelled_unavailable": "evidence_unavailable"}.get(mode, "review_incomplete")
        return _reply(review, unknown_reason=reason)

    runner = install_harness(monkeypatch, harness, turn)
    raw = asyncio.run(runner(prompt="Review the original claim.", cwd=tmp_path, model="model",
                             cli_path=None, review=review, max_turns=max_turns, budget=budget))
    parsed = claude_judge.parse_session_result(raw, [{"item_id": "case"}], source=SOURCE,
                                              reviews={"case": review}, harness=harness)
    assert parsed["verdicts"][0]["verdict"] == expected_verdict
    assert len(calls) == expected_calls
    assert [allowance for _, allowance in calls] == list(
        range(initial_remaining, initial_remaining - len(calls), -1))
    assert parsed["n_errors"] == 0
    unfinished = mode in {"still_incomplete", "no_budget", "mislabelled_unavailable"}
    assert parsed["n_review_incomplete"] == int(unfinished)
    assert len(review.session["attempts"]) == expected_calls
    if mode == "mislabelled_unavailable":
        assessment = next(a for a in parsed["verdicts"][0]["claim_assessments"]
                          if a["requirement_id"] == "no_backup")
        assert assessment["unknown_reason"] == "review_incomplete"
        assert "original_unknown_reason" in assessment
        assert assessment["completion_evidence"]["unread_ranges"]


def test_unread_range_navigation_is_exact_and_does_not_create_read_receipts(tmp_path):
    review = _review(tmp_path)
    review.read("response_text", 10, 10)
    review.read("response_text", 25, 5)
    before = list(review.reads)
    spans = [{"field": "response_text", "start": 5, "end": 35}]
    assert review.unread_ranges(spans) == [
        {"field": "response_text", "start": 5, "end": 10},
        {"field": "response_text", "start": 20, "end": 25},
        {"field": "response_text", "start": 30, "end": 35},
    ]
    review.status()
    assert review.reads == before


def test_incomplete_review_does_not_pass_through_editorial_narrowing():
    result = attach_calibration({"verdicts": [{
        "verdict": "uncertain", "processing_status": "completed",
        "completion_status": "needs_evidence_review",
    }]}, {"claim_calibration": {
        "status": "overclaim", "claim_note": "No proof because I did not read the trace.",
        "rewrite_hint": "Narrow the detector to one task.",
    }})
    assert result["claim_calibration"]["status"] == "mixed"
    assert "one task" not in str(result)


def test_bad_unknown_reason_is_a_processing_error(tmp_path):
    review = _review(tmp_path)
    review.read("detector_source")
    review.read("response_text", 0, 50)
    parsed = claude_judge.parse_session_result(
        _reply(review, unknown_reason=["review_incomplete"]), [{"item_id": "case"}],
        source=SOURCE, reviews={"case": review},
    )
    assert parsed["n_errors"] == 1
    assert "unknown_reason" in parsed["verdicts"][0]["validation_error"]


def test_pending_scope_read_blocks_additional_cases_and_requests_completion(tmp_path):
    from autofa.metrics.adaptive_review import sample_request
    from autofa.metrics.review_diagnosis import build_review_diagnosis

    review = _review(tmp_path)
    review.read("detector_source")
    review.read("response_text", 0, 50)
    reply = json.loads(_reply(review, unknown_reason="evidence_unavailable"))
    reply["verdicts"][0]["additional_sample_request"] = {
        "reason": "across_task_consistency", "question": "Does this occur elsewhere?",
        "expected_evidence": "Another task with an overwrite.",
        "preserve_original_findings": True,
    }
    result = claude_judge.parse_session_result(
        json.dumps(reply), [{"item_id": "case"}], source=SOURCE, reviews={"case": review},
    )
    row = result["verdicts"][0]
    assert row["completion_status"] == "needs_evidence_review"
    assert sample_request(result) is None
    assessment = next(a for a in row["claim_assessments"] if a["requirement_id"] == "no_backup")
    diagnosis = build_review_diagnosis(row, assessment, requirement=DESCRIPTION, name="detect")
    assert diagnosis["evidence_availability"] == "available"
    assert diagnosis["next_action"]["kind"] == "inspect_evidence"


# Local inspection: unfinished local proof gets one bounded continuation, not a whole-trace read.


# Each mode runs once, alternating the two harness transports.
@pytest.mark.parametrize("harness,mode,calls_expected,verdict_expected", [
    ("claude", "missing_ack", 2, "yes"),
    ("codex", "pending", 2, "no"),
    ("claude", "claimed_complete_unread", 2, "yes"),
    ("codex", "still_pending", 2, "uncertain"),
    ("claude", "unavailable", 1, "uncertain"),
])
def test_local_inspection_continues_in_same_budget(
    tmp_path, monkeypatch, harness, mode, calls_expected, verdict_expected,
):
    review = local_review(tmp_path)
    end = len(review.fields["response_text"])
    span = {"field": "response_text", "start": end - len(FINAL), "end": end}
    calls = []

    async def turn(call, prompt, resume, allowance):
        calls.append((resume, allowance))
        if len(calls) == 1:
            assert resume is None
            review.read("detector_source")
            review.read("response_text", 0, 31)
        else:
            assert resume == "saved-thread"
            assert "evidence_inspection" in prompt and "local" in prompt
            if mode != "missing_ack":
                assert f'"start": {span["start"]}' in prompt
            if mode != "still_pending":
                review.read("response_text", span["start"], len(FINAL))
                # The continuation can confirm OR refute. No verdict is forced.
                return local_reply(
                    review, status="contradicted" if mode == "pending" else "supported")
        if mode == "missing_ack":
            return local_reply(review)
        if mode == "unavailable":
            return local_reply(review, inspection={
                "status": "unavailable", "ranges": [],
                "reason": "The required external reference answer was never retained.",
            })
        return local_reply(review, inspection={
            "status": "complete" if mode == "claimed_complete_unread" else "pending",
            "ranges": [span], "reason": "The final artifact determines whether recovery succeeded.",
        })

    runner = install_harness(monkeypatch, harness, turn)
    prompt = claude_judge.build_verification_prompt(prompt=claude_judge.build_judge_prompt(
        name="final_file", description=LOCAL_DESCRIPTION, mechanism="", source=SOURCE,
        evidence=[],
    ))
    assert "evidence_inspection" in prompt
    raw = asyncio.run(runner(prompt=prompt, cwd=tmp_path, model="test", cli_path=None,
                             review=review, max_turns=8))
    parsed = claude_judge.parse_session_result(raw, [{"item_id": "case"}], source=SOURCE,
                                              reviews={"case": review}, harness=harness)
    assert parsed["verdicts"][0]["verdict"] == verdict_expected
    assert parsed["n_errors"] == 0
    assert parsed["n_review_incomplete"] == int(mode == "still_pending")
    allowance = plan_review_budget(review, 8).tool_call_limit
    assert calls == [(None, allowance)] + (
        [("saved-thread", allowance - 1)] if calls_expected == 2 else [])
    assert not review.covers("response_text", 31, span["start"])


@pytest.mark.parametrize("span", [
    {"field": "response_text", "start": 0, "end": 1_000_000},
    {"field": "detector_source", "start": 0, "end": 1},  # Detector code is not trace evidence.
], ids=["beyond_field", "detector_source"])
def test_invalid_inspection_ranges_cannot_claim_evidence_completion(tmp_path, span):
    review = local_review(tmp_path)
    review.read("detector_source")
    review.read("response_text", 0, 31)
    raw = local_reply(review, inspection={"status": "complete", "ranges": [span],
                                          "reason": "Claimed inspected evidence."})
    parsed = claude_judge.parse_session_result(raw, [{"item_id": "case"}], source=SOURCE,
                                              reviews={"case": review})
    assert parsed["n_errors"] == 1
    assert "evidence_inspection" in parsed["verdicts"][0]["validation_error"]
