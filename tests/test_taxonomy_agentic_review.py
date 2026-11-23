# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic taxonomy review: harness providers, evidence tools and submissions."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from autofa.harness.claude_transport import CLAUDE_MAX_BUFFER_SIZE
from autofa.harness.codex_app_server import CodexAppServerError
from autofa.metrics.claude_judge import _evidence_tool_result
from autofa.taxonomy import agentic_review
from autofa.taxonomy.agentic_review import (
    AgenticClassificationProvider,
    TaxonomyEvidenceReview,
)
from autofa.taxonomy.schema import Taxonomy
from autofa.util.cancellation import Cancellation, cancellation_scope


def _taxonomy() -> Taxonomy:
    return Taxonomy.from_dict({
        "id": "test", "name": "Test", "version": "1",
        "root": {"id": "failure", "label": "Failure", "description": "Failure", "children": [
            {"id": "missing", "code": "D.1", "label": "Missing",
             "description": "Required evidence was missed."},
            {"id": "wrong", "code": "M.1", "label": "Wrong",
             "description": "The model chose an unsupported answer."},
        ]},
    })


def _diagnosis() -> dict[str, Any]:
    return {
        "observed_system_failure": "The answer was incorrect.",
        "causal_chain": [{
            "source_type": "trace_event", "event_ref": 1, "artifact_ref": None,
            "claim": "The model ignored the required value.", "evidence": "unique evidence",
        }],
        "critical_event": {
            "event_ref": 1, "failure": "Ignored required value.",
            "why_earliest_unrecovered": "No later correction occurred.",
        },
        "counterfactual_edit": {
            "target_event_ref": 1, "minimal_edit": "Use the required value.",
            "expected_effect": "The answer would be correct.", "support": "supported",
        },
        "evidence_support": "complete",
        "abstention_reason": None,
    }


INCIDENT = {"id": "rollout", "evidence": [{"observation": {"ordered_trace": [{
    "event_ref": {"event_idx": 1, "role": "assistant"}, "text": {"value": "unique evidence"},
}]}}]}


def _review(
    *,
    artifact_source: dict[str, Any] | None = None,
) -> TaxonomyEvidenceReview:
    return TaxonomyEvidenceReview(
        taxonomy=_taxonomy(), incident=INCIDENT,
        first_pass={"selected_code": "no_supported_failure"}, windows=[], episodes=[],
        artifact_source=artifact_source,
    )


def _complete(review: TaxonomyEvidenceReview, rationale: str) -> tuple[dict, dict]:
    """Read and cite the incident, record the checks and submit D.1."""
    review.read("incident", limit=20_000)
    citation = review.cite("incident", "unique evidence")
    review.update_status(
        task_context_checked=True, outcome_checked=True, recovery_checked=True,
        counterevidence_checked=True, candidate_leaves=["D.1"],
    )
    result = review.submit(selected_code="D.1", rationale=rationale, diagnosis=_diagnosis(),
                           citation_ids=[citation["citation_id"]])
    return result, citation


def test_agentic_tools_are_taxonomy_only_and_require_read_citations() -> None:
    review = _review()

    assert {spec[0] for spec in review.specs()} == {
        "evidence_overview", "read_evidence", "search_evidence", "cite_evidence",
        "trace_overview", "list_trace_events", "read_trace_event_field",
        "cite_trace_event_field", "read_trace_event_part", "cite_trace_event_part",
        "list_trace_resource", "read_trace_resource_field", "cite_trace_resource",
        "read_trace_step", "search_trace",
        "trace_context", "cite_trace", "list_artifacts", "inspect_artifact", "read_artifact",
        "view_artifact", "cite_artifact", "search_taxonomy", "update_review_status",
        "review_status", "submit_classification",
    }
    with pytest.raises(ValueError, match="requires reading"):
        review.cite("incident", "unique evidence")
    result, citation = _complete(review, "The trace directly supports D.1.")

    assert result["accepted"] is True
    assert review.submission["selected_code"] == "D.1"
    with pytest.raises(ValueError, match="already submitted"):
        review.submit(selected_code="D.1", alternatives=[], rationale="Duplicate.",
                      diagnosis=_diagnosis(), citation_ids=[citation["citation_id"]])


def test_search_evidence_stops_after_a_partial_page() -> None:
    result = _review().search("incident", "unique evidence", limit=20)

    assert len(result["matches"]) == 1
    assert result["next_offset"] is None


def test_agentic_trace_tools_reuse_normalized_evalrdl_navigation() -> None:
    def event(index, kind, role, text, lane=""):
        return {"event_idx": index, "lane": lane, "kind": kind, "role": role,
                "parts": [{"part_idx": 0, "text": text}]}

    source = {"trace": {
        "evidence_scope": {"identity": "evalrdl_tables_v1", "event_lane": ""},
        "execution": {
            "rollout": {"run_id": "run", "task_id": "task", "repeat": 0},
            "events": [event(0, "message", "user", "task"),
                       event(1, "message", "assistant", "wrong branch"),
                       event(2, "tool_result", "tool", "failed"),
                       event(1, "message", "assistant", "auxiliary fact", "worker")],
            "tool_calls": [{"event_idx": 1, "call_idx": 0, "tool_name": "test",
                            "args_full": "{}", "result_event_idx": 2}],
            "attempts": [{"attempt_idx": 0, "error": "retry detail"}],
            "artifacts": [], "tool_definitions": [], "provenance": [],
            "relations": [], "prompt_messages": [
                {"msg_hash": "prompt-hash", "msg_json": "exact prompt"}
            ], "prompt_summaries": [{
                "summary": "bounded prompt summary",
                "kept": [{"content": "large nested value" * 100} for _ in range(20)],
            }],
        },
        "steps": [
            {"step_id": 0, "source": "user", "message": "task"},
            {"step_id": 1, "source": "assistant", "reasoning_content": "wrong branch",
             "message": "trying", "tool_calls": [{"name": "test", "arguments": {}}]},
            {"step_id": 2, "source": "tool", "observation": "failed", "paired_event_idx": 1},
        ],
    }}
    review = _review(artifact_source=source)

    overview = review.trace_overview()
    assert overview["resources"]["events"] == 4
    assert overview["resources"]["steps"] == 3
    assert review.list_trace_events()["events"][1]["event_idx"] == 1
    hit = review.search_trace("wrong branch")["matches"][0]
    assert hit["step_index"] == 2
    assert hit["source_event"]["event_idx"] == 1
    assert len(review.trace_context(2)["events"]) == 3
    event_read = review.read_trace_event_part("worker", 1, 0)
    assert event_read["text"] == "auxiliary fact"
    read = review.read_trace_step(2, "reasoning")
    assert read["text"] == "wrong branch"
    citation = review.cite_trace(2, "reasoning", "wrong branch")
    assert citation["source_event"]["event_idx"] == 1
    event_citation = review.cite_trace_event_part("worker", 1, 0, "auxiliary fact")
    assert event_citation["lane"] == "worker"
    assert event_citation["field"] == "text"
    event_field = review.read_trace_event_field("worker", 1, "kind")
    assert event_field["text"] == "message"
    with pytest.raises(ValueError, match="available: event_idx, lane, kind, role"):
        review.read_trace_event_field("worker", 1, "tool_calls")
    field_citation = review.cite_trace_event_field(
        "worker", 1, "kind", "message"
    )
    assert field_citation["kind"] == "trace_event_field"
    assert review.list_trace_resource("prompt_messages")["total"] == 1
    nested = review.list_trace_resource("prompt_summaries")["rows"][0]["kept"]
    assert nested["truncated"] is True
    assert nested["characters"] > 500
    resource_read = review.read_trace_resource_field(
        "prompt_messages", 0, "msg_json"
    )
    assert resource_read["text"] == "exact prompt"
    resource_citation = review.cite_trace_resource(
        "prompt_messages", 0, "msg_json", "exact prompt"
    )
    assert resource_citation["resource"] == "prompt_messages"


def test_list_trace_events_summarizes_large_nested_event_fields() -> None:
    source = {"trace": {
        "execution": {
            "rollout": {"run_id": "run", "task_id": "task", "repeat": 0},
            "events": [{
                "event_idx": 0,
                "lane": "",
                "kind": "model_call",
                "event_text": {"request_messages": "x" * 5_000_000},
                "parts": [],
            }],
            "artifacts": [],
        },
        "steps": [],
    }}

    review = _review(artifact_source=source)
    event = review.list_trace_events()["events"][0]

    assert event["event_text"]["truncated"] is True
    assert event["event_text"]["characters"] > 5_000_000
    assert len(event["event_text"]["preview"]) == 500
    exact = review.read_trace_event_field("", 0, "event_text", limit=20_000)
    assert exact["text"].startswith('{"request_messages":"')
    assert exact["next_offset"] == 20_000


def test_agentic_tool_results_fail_closed_before_transport_overflow() -> None:
    review = _review()

    with pytest.raises(ValueError, match="narrow the request"):
        review._record(
            "oversized_test",
            {},
            {"value": "x" * (agentic_review.MAX_AGENTIC_TOOL_RESULT_BYTES + 1)},
        )

    assert review.tool_calls == []


def test_malformed_tool_arguments_explain_signature_without_spending_budget() -> None:
    review = _review()
    review.max_tool_calls = 1
    tools = {name: callback for name, _, _, callback in review.specs()}

    with pytest.raises(
        agentic_review.ToolArgumentError, match="accepted arguments: field, offset, limit"
    ):
        tools["read_evidence"](field="incident", kind="text")

    assert review.evidence_tool_calls == 0
    assert review.budget_exhausted is False
    assert review.tool_calls[-1]["error_type"] == "ToolArgumentError"
    assert review.tool_calls[-1]["argument_names"] == ["field", "kind"]
    with pytest.raises(ValueError, match="query must be text"):
        tools["search_taxonomy"](query=None)


def test_agentic_artifact_tools_read_view_and_cite_verified_evidence(monkeypatch) -> None:
    source = {"trace": {"execution": {
        "rollout": {"run_id": "run", "task_id": "task", "repeat": 0},
        "source": {"kind": "trace_api", "base_url": "https://data.example"},
        "artifacts": [{
            "artifact_idx": 2, "name": "failure.pdf", "role": "verifier",
            "text": "secret extracted text", "data_url": "data:image/png;base64,U0VDUkVU",
            "download_url": "https://data.example/secret", "path": "/tmp/secret",
        }],
        "events": [],
    }}}
    review = _review(artifact_source=source)
    artifact_sha = "a" * 64
    identity = {"run_id": "run", "task_id": "task", "repeat": 0, "artifact_idx": 2}
    monkeypatch.setattr("autofa.trace.artifact", lambda item, artifact_idx: {
        "status": "available", "verified": True, "sha256": artifact_sha,
        "source_identity": identity, "transform_version": "v1",
        "segments": [{"locator": "page:1", "text": "exact verifier failure"}], "complete": True,
    })
    image = b"image"
    image_sha = hashlib.sha256(image).hexdigest()

    def rendered(render_sha):
        return lambda item, artifact_idx, page=1: {
            "status": "available", "verified": True, "artifact_sha256": artifact_sha,
            "sha256": render_sha, "source_identity": identity, "transform_version": "v1",
            "locator": f"page:{page}",
            "data_url": "data:image/png;base64," + base64.b64encode(image).decode(),
        }

    monkeypatch.setattr("autofa.trace.artifact_image", rendered(image_sha))

    listed = review.list_artifacts()["artifacts"][0]
    assert listed["artifact_idx"] == 2
    assert {"text", "data_url", "download_url", "path"}.isdisjoint(listed)
    inspection = review.inspect_artifact(2)
    assert "text" not in inspection["segments"][0]
    assert inspection["segments"][0]["total_characters"] == 22
    read = review.read_artifact(2, "page:1")
    assert read["text"] == "exact verifier failure"
    text_citation = review.cite_artifact(2, "page:1", quote="verifier failure")
    image = b"i" * (agentic_review.MAX_AGENTIC_TOOL_RESULT_BYTES + 1)
    image_sha = hashlib.sha256(image).hexdigest()
    monkeypatch.setattr("autofa.trace.artifact_image", rendered(image_sha))
    viewed = review.view_artifact(2)
    assert viewed["receipt"]["sha256"] == image_sha
    image_citation = review.cite_artifact(2, "page:1")

    assert text_citation["artifact_idx"] == 2
    assert image_citation["kind"] == "image"
    assert review.artifact_pins[2]["sha256"] == artifact_sha
    monkeypatch.setattr("autofa.trace.artifact_image", rendered("0" * 64))
    with pytest.raises(ValueError, match="changed during this review"):
        review.view_artifact(2)


def test_agentic_submission_requires_completed_evidence_checks() -> None:
    review = _review()
    review.read("incident", limit=20_000)
    citation = review.cite("incident", "unique evidence")

    with pytest.raises(ValueError, match="complete task context"):
        review.submit(selected_code="D.1", rationale="The cited event supports D.1.",
                      diagnosis=_diagnosis(), citation_ids=[citation["citation_id"]])


def test_codex_preflight_fails_closed_on_authentication_error(monkeypatch) -> None:
    monkeypatch.setattr("autofa.harness.model_catalog._find_codex_cli",
                        lambda cli_path: "/usr/bin/codex")
    monkeypatch.setattr(agentic_review, "_authentication_status",
                        lambda cli_path, harness: "not signed in")

    with pytest.raises(agentic_review.HarnessNotFoundError, match="not signed in"):
        agentic_review._preflight_harness("codex", None)


def test_claude_accepts_semantic_submission_despite_generic_error_flag() -> None:
    assert not agentic_review._claude_result_incomplete(
        subtype="success", is_error=True, submitted=True
    )
    assert agentic_review._claude_result_incomplete(
        subtype="success", is_error=True, submitted=False
    )
    assert not agentic_review._claude_result_incomplete(
        subtype="error", is_error=False, submitted=True
    )


def _codex(monkeypatch, tmp_path, review, run_codex_turn, **kwargs) -> dict:
    monkeypatch.setattr("autofa.harness.codex_app_server.run_codex_turn", run_codex_turn)
    return asyncio.run(agentic_review._run_codex(
        prompt="review", system_prompt="system", review=review, model="judge-model",
        cli_path="codex", effort="medium", directory=tmp_path, **kwargs,
    ))


def _codex_turn(
    status: str,
    error: str | None = None,
    usage: dict | None = None,
    *,
    error_code: str | None = None,
    retryable: bool = False,
):
    return SimpleNamespace(status=status, error=error, thread_id="thread", turn_id="turn",
                           usage=usage or {}, error_code=error_code, retryable=retryable)


def _track_overview_threads(review: TaxonomyEvidenceReview) -> list:
    threads = []
    overview = review.overview

    def tracked_overview():
        threads.append(threading.current_thread())
        return overview()

    review.overview = tracked_overview
    return threads


def test_codex_review_uses_the_configured_agentic_budget(monkeypatch, tmp_path) -> None:
    captured = {}
    review = _review()
    tool_threads = _track_overview_threads(review)

    async def run_codex_turn(**kwargs):
        captured.update(kwargs)
        await kwargs["call_tool"]("evidence_overview", {})
        return _codex_turn("completed")

    result = _codex(monkeypatch, tmp_path, review, run_codex_turn)

    assert result["status"] == "completed"
    assert captured.get("max_tool_calls") is None
    assert captured["timeout_s"] == agentic_review.AGENTIC_SESSION_TIMEOUT_SECONDS
    assert tool_threads and tool_threads[0] is not threading.main_thread()


def test_codex_keeps_a_valid_submission_after_a_terminal_limit(monkeypatch, tmp_path) -> None:
    async def run_codex_turn(**kwargs):
        return _codex_turn("max_tool_calls", "tool call limit reached", {"input_tokens": 10})

    review = _review()
    with pytest.raises(RuntimeError, match="agentic taxonomy classification"):
        _codex(monkeypatch, tmp_path, review, run_codex_turn, mode="classification")
    review.submission = {"selected_code": "D.1"}
    result = _codex(monkeypatch, tmp_path, review, run_codex_turn)

    assert result["status"] == "max_tool_calls"
    assert result["usage"] == {"input_tokens": 10}


def test_codex_promotes_retryable_turn_failure(monkeypatch, tmp_path) -> None:
    async def run_codex_turn(**kwargs):
        return _codex_turn(
            "failed",
            "transport closed",
            error_code="transportClosed",
            retryable=True,
        )

    with pytest.raises(CodexAppServerError) as caught:
        _codex(
            monkeypatch,
            tmp_path,
            _review(),
            run_codex_turn,
            mode="classification",
        )

    assert caught.value.code == "transportClosed"
    assert caught.value.retryable


@pytest.mark.parametrize("max_turns", [12, None])
def test_restricted_codex_session_restarts_with_clean_receipts(
    monkeypatch, tmp_path, max_turns
) -> None:
    review = _review()
    attempts = 0

    async def flaky_codex(**kwargs):
        nonlocal attempts
        attempts += 1
        active_review = kwargs["review"]
        if attempts == 1:
            active_review.trace_reads.append({"kind": "partial"})
            active_review.tool_calls.append({"name": "partial"})
            active_review.evidence_tool_calls += 1
            error = CodexAppServerError({"code": "transportClosed"})
            error.usage = {"input_tokens": 5}
            raise error
        assert active_review.trace_reads == []
        assert active_review.tool_calls == [{
            "name": "partial",
            "discarded_transport_attempt": True,
        }]
        return {
            "session_id": "session",
            "status": "completed",
            "usage": {"input_tokens": 7},
        }

    monkeypatch.setattr(agentic_review, "_run_codex", flaky_codex)
    monkeypatch.setattr(agentic_review, "autofa_home", lambda: tmp_path)
    monkeypatch.setattr(agentic_review, "_codex_transport_retry_delay", lambda *_: 0)

    result = asyncio.run(agentic_review._run_restricted_session_async(
        harness="codex",
        prompt="review",
        system_prompt="system",
        review=review,
        model="judge-model",
        cli_path="codex",
        effort="medium",
        thinking=None,
        thinking_budget_tokens=None,
        max_turns=max_turns,
    ))

    assert attempts == 2
    assert review.max_tool_calls == max_turns
    assert review.evidence_tool_calls == 1
    assert result["usage"] == {"input_tokens": 12}
    assert result["transport_attempts"] == 2
    assert result["transport_restarts"] == 1


def test_codex_transport_retry_cannot_reset_exhausted_tool_budget(
    monkeypatch, tmp_path
) -> None:
    review = _review()
    attempts = 0

    async def failed_codex(**kwargs):
        nonlocal attempts
        attempts += 1
        kwargs["review"].evidence_tool_calls += 1
        raise CodexAppServerError({"code": "transportClosed"})

    monkeypatch.setattr(agentic_review, "_run_codex", failed_codex)
    monkeypatch.setattr(agentic_review, "autofa_home", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="exhausting the taxonomy tool budget"):
        asyncio.run(agentic_review._run_restricted_session_async(
            harness="codex",
            prompt="review",
            system_prompt="system",
            review=review,
            model="judge-model",
            cli_path="codex",
            effort="medium",
            thinking=None,
            thinking_budget_tokens=None,
            max_turns=1,
        ))

    assert attempts == 1


def test_spent_evidence_budget_still_allows_citation_and_submission() -> None:
    review = _review()
    review.max_tool_calls = 1
    tools = {name: fn for name, _description, _schema, fn in review.specs()}

    tools["read_evidence"](field="incident", limit=20_000)
    refused = _evidence_tool_result("search_taxonomy", tools["search_taxonomy"], {})
    citation = tools["cite_evidence"](field="incident", quote="unique evidence")
    tools["update_review_status"](
        task_context_checked=True, outcome_checked=True, recovery_checked=True,
        counterevidence_checked=True, candidate_leaves=["D.1"],
    )
    tools["submit_classification"](
        selected_code="D.1", rationale="Supported.", diagnosis=_diagnosis(),
        citation_ids=[citation["citation_id"]],
    )

    assert refused["isError"]
    assert "Evidence budget spent (1 calls)" in refused["content"][0]["text"]
    assert review.submission["selected_code"] == "D.1"
    assert review.evidence_tool_calls == 1
    assert review.total_tool_calls == 5
    assert review.budget_exhausted
    assert [call.get("error_type") for call in review.tool_calls] == [
        None, "EvidenceBudgetExhausted", None, None, None,
    ]


def _claude(monkeypatch, tmp_path, review, *, query, tool, options, result) -> dict:
    import claude_agent_sdk  # Imported lazily: the SDK is slow to import.

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", options)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", result)
    monkeypatch.setattr(claude_agent_sdk, "SystemMessage", type("System", (), {}))
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", lambda **kwargs: kwargs)
    monkeypatch.setattr(claude_agent_sdk, "query", query)
    monkeypatch.setattr(claude_agent_sdk, "tool", tool)
    return asyncio.run(agentic_review._run_claude(
        prompt="review", system_prompt="system", review=review, model="judge-model",
        cli_path="claude", effort="medium", thinking=None, thinking_budget_tokens=None,
        directory=tmp_path,
    ))


def test_claude_tools_run_outside_the_event_loop(monkeypatch, tmp_path) -> None:
    exposed = {}
    captured = {}
    review = _review()
    tool_threads = _track_overview_threads(review)

    class Result:
        subtype = "success"
        is_error = False
        session_id = "session"
        stop_reason = "end_turn"
        num_turns = 1
        usage = {}
        total_cost_usd = 0
        errors = []
        terminal_reason = None

    def tool(name, description, schema):
        def decorate(fn):
            exposed[name] = fn
            return fn

        return decorate

    async def query(**kwargs):
        await exposed["evidence_overview"]({})
        yield Result()

    def options(**kwargs):
        captured.update(kwargs)
        return kwargs

    result = _claude(monkeypatch, tmp_path, review, query=query, tool=tool, options=options,
                     result=Result)

    assert result["status"] == "success"
    assert "max_turns" not in captured
    assert captured["max_buffer_size"] == CLAUDE_MAX_BUFFER_SIZE
    assert tool_threads and tool_threads[0] is not threading.main_thread()


def test_claude_keeps_a_valid_submission_after_session_timeout(monkeypatch, tmp_path) -> None:
    review = _review()

    async def query(**_kwargs):
        review.submission = {"selected_code": "D.1"}
        await asyncio.Event().wait()
        yield  # pragma: no cover

    monkeypatch.setattr(agentic_review, "AGENTIC_SESSION_TIMEOUT_SECONDS", 0.001)

    result = _claude(monkeypatch, tmp_path, review, query=query,
                     tool=lambda *_: (lambda fn: fn), options=lambda **kwargs: kwargs,
                     result=type("Result", (), {}))

    assert result["status"] == "timeout"


def test_restricted_session_honors_batch_cancellation(monkeypatch) -> None:
    async def never_finishes(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(agentic_review, "_run_restricted_session_async", never_finishes)
    cancellation = Cancellation()
    cancellation.cancel()

    with cancellation_scope(cancellation), pytest.raises(asyncio.CancelledError):
        agentic_review._run_restricted_session()


def _provider(runner) -> AgenticClassificationProvider:
    return AgenticClassificationProvider(
        harness="codex", model="judge-model", cli_path="codex", effort="medium",
        thinking=None, thinking_budget_tokens=None, max_turns=12, runner=runner,
    )


def test_agentic_provider_returns_harness_and_tool_receipts() -> None:
    first_pass = {"selected_code": "D.1", "selected_episode_id": "e2",
                  "diagnosis": {"critical_event": {"event_ref": 473}}}
    episodes = [{"episode_id": "e2", "leaf_code": "D.1", "first_event_idx": 417,
                 "last_event_idx": 473, "status": "open_primary"}]

    def runner(**kwargs):
        assert kwargs["mode"] == "review"
        review = kwargs["review"]
        # The evidence overview starts with the complete phase-one findings.
        overview = review.overview()
        assert overview["phase_one"]["proposal"] == first_pass
        assert overview["phase_one"]["episodes"] == episodes
        _complete(review, "The cited event is the earliest unrecovered failure.")
        return {"session_id": "session", "usage": {"input_tokens": 10}}

    result = _provider(runner).review(
        taxonomy=_taxonomy(), incident=INCIDENT, first_pass=first_pass, windows=[],
        episodes=episodes,
    )

    assert result["selected_code"] == "D.1"
    assert result["usage"] == {"input_tokens": 10}
    assert result["receipts"]["harness"] == "codex"
    assert result["receipts"]["tool_budget_semantics"] == "investigation_calls_v1"
    assert result["receipts"]["total_tool_calls"] == 0
    assert result["receipts"]["budget_exhausted"] is False
    assert len(result["receipts"]["citations"]) == 1


def test_agentic_provider_classifies_without_a_prior_proposal() -> None:
    def runner(**kwargs):
        assert kwargs["mode"] == "classification"
        review = kwargs["review"]
        # A direct classification has no prior proposal to show.
        overview = review.overview()
        assert overview["classification_mode"] == "direct"
        assert "phase_one" not in overview
        assert "first_pass" not in overview["fields"]
        _complete(review, "The cited event supports the classification.")
        return {"session_id": "session", "usage": {"input_tokens": 11}}

    result = _provider(runner).classify(taxonomy=_taxonomy(), incident=INCIDENT, windows=[])

    assert result["selected_code"] == "D.1"
    assert result["receipts"]["mode"] == "classification"
    assert result["usage"] == {"input_tokens": 11}
