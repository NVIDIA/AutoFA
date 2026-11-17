# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codex judges keep blinded compilation, audited evidence, isolation and one shared budget."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from autofa.harness import codex_app_server
from autofa.metrics import claim_compiler, claude_judge, codex_judge
from autofa.metrics.claim_contract import build_claim_contract
from autofa.metrics.judge_budget import plan_review_budget
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot
from autofa.metrics.repair_validation import merge_processing_retry

CLAIM = {
    "description": "Assistant omitted a required check.",
    "mechanism": "Assistant skipped verification.",
    "source": "def detect(item): return True # PRIVATE_DETECTOR_SOURCE",
}
TEXT = "The required check was skipped."


def _compilation():
    return {"defining_clauses": [
        {"id": "check", "source_field": "description", "quote": CLAIM["description"],
         "required_review_scope": "local",
         "predicate": CLAIM["description"]},
        {"id": "verification", "source_field": "mechanism", "quote": CLAIM["mechanism"],
         "required_review_scope": "local",
         "predicate": CLAIM["mechanism"]},
    ], "exclusions": []}


def _contract():
    return build_claim_contract(**CLAIM, defining_clauses=_compilation()["defining_clauses"],
                                compilation={
                                    "policy_hash": claim_compiler.compiler_policy_hash(),
                                    "model": "same-model", "exclusions": [],
                                    "coverage_review": {"approved": True, "issues": []},
                                })


def _reply(review, *, status="supported", citation_id=None):
    if citation_id is None:
        review.read("detector_source")
        review.read("response_text")
        citation_id = review.cite("response_text", TEXT)["citations"][0]["citation_id"]
    contract = review.snapshot["claim_contract"]
    return json.dumps({"verdicts": [{
        "item_id": review.snapshot["item_id"], "snapshot_id": review.snapshot["snapshot_id"],
        "claim_contract_id": review.snapshot["claim_contract"]["contract_id"], "scope": "local",
        "claim_assessments": [
            {"requirement_id": requirement["id"], "status": status,
             **({"defining_requirement_ids": [contract["defining_clauses"][0]["id"]]}
                if status != "supported" and contract.get("defining_clauses")
                and requirement["kind"] in {"claim", "review_check"} else {}),
             **({"evidence_inspection": {
                 "status": "complete", "ranges": [{"field": "response_text", "start": 0,
                                                       "end": len(review.fields["response_text"])}],
                 "reason": "The inspected original evidence remains ambiguous.",
             }} if status == "unknown" else {}),
             "rationale": "Checked the original evidence independently.",
             "citations": [{"citation_id": citation_id}], "counterevidence": []}
            for requirement in review.snapshot["claim_contract"]["requirements"]
        ],
    }], "behavior_specificity": 0.8})


def _result(text, session="session", **kwargs):
    return SimpleNamespace(text=text, thread_id=session, turn_id="turn", status="completed",
                           usage={"input_tokens": 11, "output_tokens": 7},
                           error=None, error_code=None, retryable=False, **kwargs)


def _review(directory):
    return EvidenceReview(make_snapshot(
        {"response_text": TEXT}, item_id="case", source=CLAIM["source"], claim_contract=_contract(),
    ), directory)


def _session(review, **kwargs):
    """Coroutine for one restricted judge session over ``review``'s frozen evidence."""
    return codex_judge.run_codex_judge_session(**{
        "prompt": "Review original evidence", "cwd": review.directory, "model": "test",
        "cli_path": None, "review": review, **kwargs,
    })


def _isolation_failure(thread):
    result = _result("", thread)
    result.status, result.turn_id = "failed", ""
    result.error_code, result.error = "restrictedToolsUnsupported", "cannot isolate"
    return result


def _tool_text(response):
    """First text block of a Claude (``content``) or Codex (``contentItems``) tool response."""
    return (response.get("content") or response["contentItems"])[0]["text"]


def _tool_failed(response):
    return bool(response.get("isError")) or response.get("success") is False


def _exercise_tool(monkeypatch, review, harness, tool, exercise):
    """Run ``exercise(call)`` through the harness's real evidence-tool transport."""
    advertised = []
    if harness == "claude":
        import claude_agent_sdk as sdk

        handlers = {}
        create = sdk.create_sdk_mcp_server

        def capture(**kwargs):
            handlers.update({spec.name: spec.handler for spec in kwargs["tools"]})
            return create(**kwargs)

        monkeypatch.setattr(sdk, "create_sdk_mcp_server", capture)
        _, names = claude_judge._evidence_server(review)
        assert f"mcp__judge_evidence__{tool}" in names

        async def call(name, args):
            return await handlers[name](args)

        asyncio.run(exercise(call))
    else:
        async def run(**kwargs):
            advertised.extend(spec for spec in kwargs["tools"] if spec["name"] == tool)
            await exercise(kwargs["call_tool"])
            return _result(_reply(review, status="unknown"))

        monkeypatch.setattr(codex_judge, "_run_turn", run)
        asyncio.run(_session(review))
        assert len(advertised) == 1
    return advertised


def _judge(tmp_path, **kwargs):
    model = kwargs.pop("model", "same-model")
    return claude_judge.judge_behavior_with_claude(
        name="check", **CLAIM, items=[{"_item_id": "case", "response_text": TEXT}],
        assigned_labels=[True], cwd=tmp_path, model=model, max_chars=0, **kwargs,
    )


def test_codex_runs_compiler_coverage_initial_and_independent_confirmation(
    tmp_path, monkeypatch,
):
    profile = "contextual"
    calls = []
    reviews = []
    original_specs = claude_judge._evidence_tool_specs

    def specs(review):
        reviews.append(review)
        return original_specs(review)

    monkeypatch.setattr(claude_judge, "_evidence_tool_specs", specs)
    monkeypatch.setattr("autofa.harness.claude_cli.ensure_claude_auth",
                        lambda **kwargs: pytest.fail("Codex must not require Claude auth"))

    async def run(**kwargs):
        calls.append(kwargs)
        assert kwargs["restricted"] is True
        assert kwargs["model"] == "same-model" and kwargs["effort"] == "ultra"
        assert "thinking" not in kwargs and kwargs.get("resume") is None
        system = kwargs["system_prompt"]
        session = f"independent-{len(calls)}"
        compiler_system = claim_compiler.compiler_system_prompt(profile)
        coverage_system = claim_compiler.coverage_system_prompt(profile)
        if system in (compiler_system, coverage_system):
            assert kwargs["tools"] == [] and kwargs["max_tool_calls"] == 0
            assert "PRIVATE_DETECTOR_SOURCE" not in kwargs["prompt"]
            assert TEXT not in kwargs["prompt"]
            text = json.dumps(_compilation() if system == compiler_system
                              else {"approved": True, "issues": []})
        else:
            review = reviews[-1]
            assert not review.reads and not review.citations
            await kwargs["on_thread"](session)
            names = {tool["name"] for tool in kwargs["tools"]}
            assert {"read_evidence", "cite_evidence", "view_artifact",
                    "review_interval", "interval_events"} <= names
            assert "shell" not in names
            for field in ("detector_source", "response_text"):
                response = await kwargs["call_tool"]("read_evidence", {"field": field})
                assert response["success"]
            response = await kwargs["call_tool"]("cite_evidence", {
                "field": "response_text", "quote": TEXT,
            })
            citation = json.loads(response["contentItems"][0]["text"])["citations"][0]
            text = _reply(review, citation_id=citation["citation_id"])
        return _result(text, session)

    monkeypatch.setattr(codex_judge, "_run_turn", run)
    result = _judge(tmp_path, harness="codex", effort="ultra", behavior_strictness=profile)
    assert result["n_yes"] == 1 and result["n_errors"] == 0
    assert result["backend"] == "codex_app_server" and result["judge_harness"] == "codex"
    assert result["judge_effort"] == "ultra" and result["judge_thinking"] is None
    assert [call["system_prompt"] for call in calls] == [
        claim_compiler.compiler_system_prompt(profile),
        claim_compiler.coverage_system_prompt(profile),
        claude_judge.judge_system_prompt(profile),
        claude_judge.verification_system_prompt(profile),
    ]
    assert reviews[0] is not reviews[1]
    assert [session["session_id"] for session in result["sessions"]] == [
        "independent-3", "independent-4",
    ]
    assert all(session["judge_harness"] == "codex"
               for session in result["claim_compilation"]["sessions"])


def test_checkpoint_reuse_requires_same_backend_model_and_effort(tmp_path, monkeypatch):
    calls = []

    def run(*, review, **kwargs):
        calls.append(kwargs)
        return _reply(review, status="contradicted")

    monkeypatch.setattr(claude_judge, "_run_review", run)
    baseline = _judge(tmp_path, compiled_contract=_contract())
    assert _judge(tmp_path, compiled_contract=_contract())["reused_stages"] == 1
    codex = _judge(tmp_path, compiled_contract=_contract(), harness="codex")
    assert baseline["judge_rubric_hash"] != codex["judge_rubric_hash"]
    assert len(calls) == 2 and calls[-1]["harness"] == "codex"
    assert _judge(tmp_path, compiled_contract=_contract(), harness="codex")["reused_stages"] == 1
    changed = _judge(tmp_path, compiled_contract=_contract(), harness="codex", effort="ultra")
    assert changed["reused_stages"] == 0 and len(calls) == 3
    changed_model = _judge(tmp_path, compiled_contract=_contract(), harness="codex",
                           effort="ultra", model="another-model")
    assert changed_model["reused_stages"] == 0 and len(calls) == 4


def test_isolated_replacement_has_budget_to_reread_after_expensive_first_attempt(
    tmp_path, monkeypatch,
):
    review = _review(tmp_path / "review")
    calls = []
    snapshot = review.snapshot["snapshot_id"]

    async def use_tools(kwargs, count):
        for _ in range(count):
            await kwargs["call_tool"]("review_status", {})

    async def run(**kwargs):
        calls.append(kwargs)
        assert kwargs["restricted"] is True
        if len(calls) == 1:
            await kwargs["on_thread"]("initial")
            await kwargs["call_tool"]("read_evidence", {"field": "response_text"})
            await use_tools(kwargs, 51)
            return _result("invalid JSON", "initial")
        if len(calls) == 2:
            assert kwargs["resume"] == "initial" and kwargs["max_tool_calls"] == 12
            return _isolation_failure("initial")
        if len(calls) == 3:
            # A fresh isolated session restarts from the frozen claim and evidence; reads
            # from the abandoned session no longer count as coverage.
            assert kwargs["resume"] is None and kwargs["max_tool_calls"] == 64
            assert "Review original evidence" in kwargs["prompt"]
            assert "replacement isolated session" in kwargs["prompt"]
            assert review.epoch == 1 and review.snapshot["snapshot_id"] == snapshot
            assert not review.covers("response_text", 0, len(TEXT), current=True)
            await kwargs["on_thread"]("replacement")
            await use_tools(kwargs, 21)
            return _result("invalid JSON", "replacement")
        assert len(calls) == 4
        assert kwargs["resume"] == "replacement" and kwargs["max_tool_calls"] == 43
        return _result(_reply(review, status="contradicted"), "replacement")

    monkeypatch.setattr(codex_judge, "_run_turn", run)
    raw = asyncio.run(_session(review))
    assert "contradicted" in raw and len(calls) == 4
    assert review.session["budget_limit"] == 116  # 52 used + one replacement's 64
    assert review.session["budget_total_used"] == 74
    assert review.session["budget_remaining"] == 42
    assert sum(row["budget_used"] for row in review.session["attempts"]) == 74
    events = [json.loads(line)
              for line in (review.directory / "audit.jsonl").read_text().splitlines()]
    grants = [event for event in events if event.get("event") == "isolated_replacement_budget"]
    assert len(grants) == 1
    assert any(event.get("event") == "isolated_resume_failed" for event in events)


def test_replacement_cannot_replenish_budget_a_second_time(tmp_path, monkeypatch):
    review = _review(tmp_path / "review")
    calls = []

    async def run(**kwargs):
        calls.append(kwargs)
        if kwargs["resume"] is None:
            await kwargs["on_thread"]("replacement")
            await kwargs["call_tool"]("review_status", {})
            return _result("invalid JSON", "replacement")
        if len(calls) == 3:
            assert kwargs["max_tool_calls"] == plan_review_budget(review, 9).tool_call_limit - 1
        return _isolation_failure(kwargs["resume"])

    monkeypatch.setattr(codex_judge, "_run_turn", run)
    with pytest.raises(RuntimeError, match="cannot isolate"):
        asyncio.run(_session(review, max_turns=9, resume="interrupted"))
    assert len(calls) == 3


def test_codex_compaction_invalidates_pre_compaction_read_epoch(tmp_path, monkeypatch):
    review = _review(tmp_path / "review")

    async def run(**kwargs):
        await kwargs["call_tool"]("read_evidence", {"field": "response_text"})
        await kwargs["on_event"]("item/completed", {"item": {"type": "contextCompaction"}})
        assert review.epoch == 1
        return _result(_reply(review, status="unknown"))

    monkeypatch.setattr(codex_judge, "_run_turn", run)
    asyncio.run(_session(review))
    assert '"compact_boundary"' in (review.directory / "audit.jsonl").read_text()


def test_codex_evidence_images_are_delivered_as_actual_image_content():
    result = codex_judge._codex_tool_result({"content": [
        {"type": "text", "text": '{"sha256":"image-receipt"}'},
        {"type": "image", "mimeType": "image/png", "data": "actual-image-bytes"},
    ]})
    assert result == {"success": True, "contentItems": [
        {"type": "inputText", "text": '{"sha256":"image-receipt"}'},
        {"type": "inputImage", "imageUrl": "data:image/png;base64,actual-image-bytes"},
    ]}


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_interval_tool_preserves_boundaries_and_receipts_across_harnesses(
    tmp_path, monkeypatch, harness,
):
    """Both harness transports must expose the same bounded-review capability."""
    review = _review(tmp_path / harness)
    arguments = {"requirement_id": "before_write", "start_event": 1, "end_event": 4,
                 "end_citation_id": "cite-boundary"}
    receipt = {"interval_id": "interval-before-write", "complete": False,
               "required_ranges": [{"field": "response_text", "start": 11, "end": 47}]}
    received = []
    navigated = []
    navigation = {"available": True, "event_count": 40, "next_event": 23,
                  "events": [{"event_index": 21, "ranges": receipt["required_ranges"]}]}

    def bind(**kwargs):
        received.append(kwargs)
        if kwargs["end_event"] == 0:
            raise ValueError("end boundary precedes the start")
        return receipt

    def navigate(**kwargs):
        navigated.append(kwargs)
        return navigation

    monkeypatch.setattr(review, "review_interval", bind)
    monkeypatch.setattr(review, "interval_events", navigate)
    specs = {name: schema for name, _description, schema, _fn
             in claude_judge._evidence_tool_specs(review)}
    assert specs["review_interval"]["required"] == [
        "requirement_id", "start_event", "end_event",
    ]
    assert specs["review_interval"]["additionalProperties"] is False
    assert specs["review_interval"]["properties"]["end_citation_id"] == {"type": "string"}
    assert specs["interval_events"]["required"] == []
    assert specs["interval_events"]["properties"] == {
        "start_event": {"type": "integer"}, "limit": {"type": "integer"},
    }

    async def exercise(call):
        page = await call("interval_events", {"start_event": 21, "limit": 2})
        assert json.loads(_tool_text(page)) == navigation
        response = await call("review_interval", arguments)
        assert not _tool_failed(response)
        assert json.loads(_tool_text(response)) == receipt
        failed = await call("review_interval", {**arguments, "end_event": 0})
        assert _tool_failed(failed)
        assert "end boundary precedes the start" in json.dumps(failed)

    advertised = _exercise_tool(monkeypatch, review, harness, "review_interval", exercise)
    assert all(tool["inputSchema"] == specs["review_interval"] for tool in advertised)
    assert received == [arguments, {**arguments, "end_event": 0}]
    assert navigated == [{"start_event": 21, "limit": 2}]


def test_supporting_passage_request_is_explicit_and_scoped(tmp_path, monkeypatch):
    from autofa.core.localisation import passage, text_hash

    review = _review(tmp_path / "codex")
    requests = []

    def retrieve():
        requests.append(1)
        return {"item_id": "case", "source_sha256": text_hash(CLAIM["source"]),
                "passages": [passage("response_text", TEXT, 0, len(TEXT), method="llm")]}

    review._supporting_passage_resolver = retrieve

    async def exercise(call):
        assert not requests
        for _ in range(2):
            response = await call("retrieve_supporting_passages", {})
            assert json.loads(_tool_text(response))["passages"][0]["quote"] == TEXT
        assert requests == [1] and not review.reads and not review.citations
        assert _tool_failed(await call("retrieve_supporting_passages", {"item_id": "other"}))

    _exercise_tool(monkeypatch, review, "codex", "retrieve_supporting_passages", exercise)
    assert requests == [1]


@pytest.mark.parametrize("retryable,exception", [(False, RuntimeError), (True, ConnectionError)])
def test_codex_incomplete_compilation_is_archived_and_never_accepted(
    tmp_path, monkeypatch, retryable, exception,
):
    async def run(**kwargs):
        result = _result(json.dumps(_compilation()))
        result.status = "failed"
        result.error = "Unavailable"
        result.retryable = retryable
        return result

    monkeypatch.setattr(codex_judge, "_run_turn", run)
    with pytest.raises(exception, match="incomplete claim compilation"):
        asyncio.run(claim_compiler._compiler_session(
            prompt="Claim", model="test", cli_path=None, directory=tmp_path,
            attempt=0, harness="codex",
        ))
    metadata = json.loads((tmp_path / "session-0.json").read_text())
    assert metadata["judge_harness"] == "codex" and metadata["is_error"]


@pytest.mark.parametrize("field", ["judge_harness", "backend"])
def test_processing_retry_cannot_change_harness_even_for_same_model(field):
    provenance = {"claim_hash": "claim", "judge_policy_hash": "policy",
                  "judge_contract_version": "v2", "judge_model": "same-model"}
    before = {**provenance, "judge_harness": "claude", "judge_backend": "claude_agent_sdk"}
    fresh = {**provenance, "judge_harness": "claude", "backend": "claude_agent_sdk"}
    fresh[field] = "codex" if field == "judge_harness" else "codex_app_server"
    with pytest.raises(ValueError, match="judge.harness|judge backend"):
        merge_processing_retry(before, fresh, {"case"})


def test_codex_parser_errors_do_not_report_claude_provenance():
    result = claude_judge.parse_session_result("invalid response", [], reviews={}, harness="codex")
    assert result["backend"] == "codex_app_server"


def test_schema_repairs_resume_own_session_over_live_connection_and_one_work_budget(
    tmp_path, monkeypatch,
):
    review = _review(tmp_path / "review")
    calls = []

    async def run(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            await kwargs["on_thread"]("same-live-thread")
        if len(calls) < 3:
            return _result("invalid JSON", "same-live-thread")
        return _result(_reply(review, status="contradicted"), "same-live-thread")

    monkeypatch.setattr(codex_judge, "_run_turn", run)
    raw = asyncio.run(_session(review))
    assert "contradicted" in raw and len(calls) == 3
    assert (review.directory / "repair-1.txt").exists()
    assert calls[0]["connection"] is calls[1]["connection"] is calls[2]["connection"]
    assert [call["max_tool_calls"] for call in calls] == [64, 63, 62]
    assert [call["resume"] for call in calls] == [None, "same-live-thread", "same-live-thread"]
    assert review.session["budget_total_used"] == 3
    assert review.session["budget_limit"] == 64
    assert review.session["budget_replacements"] == 0


_SERVER = r'''
import json, os, sys

def send(value):
    print(json.dumps(value), flush=True)

def event(method, params):
    send({'method': method, 'params': params})

def reply(request, result):
    send({'id': request['id'], 'result': result})

mode = os.environ['AUTOFA_PERSISTENT_MODE']
thread = 'isolated-thread'
turn_count = 0
turn = ''
pending = []
tool_count = 0
final = json.loads(open(os.environ['AUTOFA_PERSISTENT_REPLY']).read())

def finish():
    text = 'invalid JSON' if turn_count <= 2 else json.dumps(final)
    event('item/completed', {'threadId': thread, 'turnId': turn, 'item': {
        'id': 'answer-' + turn, 'type': 'agentMessage', 'phase': 'final_answer', 'text': text}})
    event('turn/completed', {'threadId': thread, 'turn': {
        'id': turn, 'status': 'completed', 'items': []}})

def next_tool():
    global tool_count
    if not pending:
        finish()
        return
    tool, arguments = pending.pop(0)
    tool_count += 1
    send({'id': 'call-' + str(tool_count), 'method': 'item/tool/call', 'params': {
        'threadId': thread, 'turnId': turn, 'callId': str(tool_count),
        'tool': tool, 'arguments': arguments}})

for line in sys.stdin:
    request = json.loads(line)
    with open(os.environ['AUTOFA_PERSISTENT_LOG'], 'a') as log:
        log.write(json.dumps({'pid': os.getpid(), **request}) + '\n')
    method = request.get('method')
    params = request.get('params', {})
    if method == 'initialize':
        reply(request, {})
    elif method == 'config/read':
        reply(request, {'config': {'mcp_servers': {'ambient': {'command': '/unused'}}}})
    elif method == 'thread/start':
        reply(request, {'thread': {'id': thread, 'environments': []}})
    elif method == 'thread/resume':
        # Reproduce the installed CLI: a persisted resume cannot preserve the
        # empty environment and must be rejected by the isolation check.
        reply(request, {'thread': {'id': thread, 'environments': [{'id': 'ambient'}]}})
    elif method == 'turn/start':
        turn_count += 1
        turn = 'turn-' + str(turn_count)
        if turn_count == 2 and mode == 'stale':
            # Arrive before the new start acknowledgement, when turn_id is not
            # known yet; completed-turn receipts still must deny this callback.
            send({'id': 'late-old-tool', 'method': 'item/tool/call', 'params': {
                'threadId': thread, 'turnId': 'turn-1', 'callId': 'late',
                'tool': 'read_evidence', 'arguments': {'field': 'response_text'}}})
            event('item/completed', {'threadId': thread, 'turnId': 'turn-1',
                                    'item': {'id': 'old-compaction',
                                             'type': 'contextCompaction'}})
        reply(request, {'turn': {'id': turn, 'status': 'inProgress', 'items': []}})
        event('turn/started', {'threadId': thread, 'turn': {'id': turn}})
        if turn_count == 2 and mode == 'cancel':
            continue
        if turn_count == 2 and mode == 'disconnect':
            sys.exit(3)
        if turn_count == 2 and mode == 'failure':
            event('turn/completed', {'threadId': thread, 'turn': {
                'id': turn, 'status': 'failed', 'items': [], 'error': {
                    'codexErrorInfo': {'httpConnectionFailed': {'httpStatusCode': 503}}}}})
            continue
        pending = ([('read_evidence', {'field': 'detector_source'}),
                    ('read_evidence', {'field': 'response_text'}),
                    ('cite_evidence', {'field': 'response_text',
                                       'quote': 'The required check was skipped.'})]
                   if turn_count == 1 else [('read_evidence', {'field': 'detector_source'})])
        next_tool()
    elif method == 'turn/interrupt':
        reply(request, {})
    elif method is None and str(request.get('id', '')).startswith('call-'):
        if not request.get('result', {}).get('success'):
            raise RuntimeError('Evidence callback failed: ' + repr(request))
        next_tool()
'''


@pytest.fixture
def protocol_review(tmp_path, monkeypatch):
    """A real local app-server child that scripts judge repairs without models or network."""
    review = _review(tmp_path / "review")
    executable = tmp_path / "codex"
    executable.write_text(f"#!{sys.executable}\n{_SERVER}")
    executable.chmod(0o755)
    log, reply = tmp_path / "protocol.jsonl", tmp_path / "reply.json"
    reply.write_text(_reply(review, status="contradicted", citation_id="cite-1"))
    monkeypatch.setenv("AUTOFA_PERSISTENT_LOG", str(log))
    monkeypatch.setenv("AUTOFA_PERSISTENT_REPLY", str(reply))
    monkeypatch.setenv("AUTOFA_PERSISTENT_MODE", "success")
    closed = []
    close = codex_app_server.CodexAppServer.close

    async def tracked_close(client):
        try:
            await close(client)
        finally:
            closed.append(client)

    monkeypatch.setattr(codex_app_server.CodexAppServer, "close", tracked_close)
    return review, executable, log, closed


def _messages(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _run(fixture, **overrides):
    review, executable, _, _ = fixture
    return _session(review, cli_path=str(executable), **overrides)


def test_live_protocol_repairs_keep_isolation_and_budget_despite_late_previous_turn_events(
    protocol_review, monkeypatch,
):
    # Late previous-turn callbacks arrive before the second start acknowledgement.
    monkeypatch.setenv("AUTOFA_PERSISTENT_MODE", "stale")
    review, _, log, closed = protocol_review
    raw = asyncio.run(_run(protocol_review))
    assert "contradicted" in raw
    messages = _messages(log)
    denied, = [row["result"] for row in messages if row.get("id") == "late-old-tool"]
    assert denied["success"] is False and "earlier" in denied["contentItems"][0]["text"]
    assert sum(row["field"] == "response_text" for row in review.reads) == 1
    starts = [row for row in messages if row.get("method") == "thread/start"]
    turns = [row for row in messages if row.get("method") == "turn/start"]
    assert len(starts) == 1 and len(turns) == 3
    assert len({row["pid"] for row in messages}) == 1
    assert not any(row.get("method") == "thread/resume" for row in messages)
    assert starts[0]["params"]["environments"] == []
    assert starts[0]["params"]["config"]["mcp_servers"]["ambient"]["enabled"] is False
    assert all(row["params"]["environments"] == [] for row in turns)
    assert all(row["params"]["threadId"] == "isolated-thread" for row in turns)
    assert review.epoch == 0 and review.session["budget_replacements"] == 0
    assert review.session["budget_total_used"] == 5
    assert [row["tool_calls"] for row in review.session["attempts"]] == [3, 1, 1]
    assert review.citations["cite-1"]["quote"] == TEXT
    assert closed and all(client.process.returncode is not None for client in closed)


def test_initial_persisted_resume_still_checks_isolation_before_one_fresh_replacement(
    protocol_review,
):
    review, _, log, closed = protocol_review
    raw = asyncio.run(_run(protocol_review, resume="isolated-thread"))
    assert "contradicted" in raw
    messages = _messages(log)
    assert len({row["pid"] for row in messages}) == 2
    assert sum(row.get("method") == "thread/resume" for row in messages) == 1
    assert sum(row.get("method") == "thread/start" for row in messages) == 1
    assert sum(row.get("method") == "turn/start" for row in messages) == 3
    assert review.epoch == 1 and review.session["budget_replacements"] == 1
    assert review.session["budget_total_used"] == 5
    assert len({id(client) for client in closed}) == 2
    assert all(client.process.returncode is not None for client in closed)


def test_true_connection_loss_discards_verified_thread_before_reconnect(
    protocol_review, monkeypatch,
):
    monkeypatch.setenv("AUTOFA_PERSISTENT_MODE", "disconnect")
    review, executable, log, closed = protocol_review

    async def accept_tool(name, arguments):
        return {"success": True, "contentItems": [{"type": "inputText", "text": "fixture"}]}

    async def scenario():
        async with codex_app_server.CodexTurnConnection() as connection:
            args = {"prompt": "first", "system_prompt": "frozen", "model": "offline",
                    "cwd": str(review.directory), "cli_path": str(executable),
                    "restricted": True, "connection": connection, "call_tool": accept_tool,
                    "tools": [{"name": name, "description": name, "inputSchema": {}}
                              for name in ("read_evidence", "cite_evidence")]}
            first = await codex_app_server.run_codex_turn(**args)
            assert first.status == "completed" and connection.client is not None
            failed = await codex_app_server.run_codex_turn(**args, resume=first.thread_id)
            assert failed.error_code == "transportClosed" and failed.retryable
            assert connection.client is None and not connection.thread_id
            resumed = await codex_app_server.run_codex_turn(**args, resume=first.thread_id)
            assert resumed.error_code == "restrictedToolsUnsupported"
            assert connection.client is None

    asyncio.run(scenario())
    messages = _messages(log)
    assert len({row["pid"] for row in messages}) == 2
    assert sum(row.get("method") == "thread/resume" for row in messages) == 1
    assert sum(row.get("method") == "turn/start" for row in messages) == 2
    assert closed and all(client.process.returncode is not None for client in closed)


@pytest.mark.parametrize("mode", ["failure", "disconnect"])
def test_failed_repair_closes_the_persistent_child_without_faking_completion(
    protocol_review, monkeypatch, mode,
):
    monkeypatch.setenv("AUTOFA_PERSISTENT_MODE", mode)
    review, _, log, closed = protocol_review
    with pytest.raises(ConnectionError, match="incomplete judge session"):
        asyncio.run(_run(protocol_review))
    assert len([row for row in _messages(log) if row.get("method") == "turn/start"]) == 2
    assert closed and all(client.process.returncode is not None for client in closed)
    assert not (review.directory / "reply-attempt-1.txt").exists()


def test_cancelling_a_repair_interrupts_and_reaps_the_same_child(protocol_review, monkeypatch):
    monkeypatch.setenv("AUTOFA_PERSISTENT_MODE", "cancel")
    review, _, log, closed = protocol_review

    async def scenario():
        task = asyncio.create_task(_run(protocol_review))
        try:
            async with asyncio.timeout(5):
                while len([row for row in _messages(log)
                           if row.get("method") == "turn/start"]) < 2:
                    await asyncio.sleep(.01)
            await asyncio.sleep(.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    messages = _messages(log)
    interrupts = [row["params"] for row in messages if row.get("method") == "turn/interrupt"]
    assert interrupts == [{"threadId": "isolated-thread", "turnId": "turn-2"}]
    assert len({row["pid"] for row in messages}) == 1
    assert closed and all(client.process.returncode is not None for client in closed)
    assert not (review.directory / "reply-attempt-1.txt").exists()
