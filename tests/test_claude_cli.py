# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code CLI harness: discovery and auth, SDK options and framing, and turn recovery."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import claude_agent_sdk as sdk
import pytest
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from autofa.harness import claude_cli as claude_cli_module
from autofa.harness.claude import ClaudeHarness, get_harness
from autofa.harness.claude_cli import ensure_claude_auth, find_claude_cli
from autofa.harness.claude_transport import CLAUDE_MAX_BUFFER_SIZE
from autofa.metrics.claim_compiler import _compiler_session
from autofa.metrics.claude_judge import _run_claude_judge_session
from autofa.metrics.judge_evidence import EvidenceReview, make_snapshot
from autofa.session.recovery import context_recovery_prompt
from autofa.session.status_log import latest_turn_status
from autofa.ui.activity import AgentActivity

# CLI discovery, authentication and SDK options.


def _fake_cli(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    path.chmod(0o755)
    return path


SOURCES = ["argument", "env", "path", "execpath", "sdk_package", "none"]


@pytest.mark.parametrize("source", SOURCES)
def test_find_claude_cli_resolution_order(tmp_path: Path, monkeypatch, source):
    """The highest-priority available CLI wins; every lower-priority source is also present."""
    for name in ("AUTOFA_CLAUDE_CLI", "CLAUDE_CLI", "CLAUDE_CODE_EXECPATH"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setattr(claude_cli_module, "DEFAULT_CLI_CANDIDATES", ("claude",))
    monkeypatch.setattr(claude_cli_module, "_sdk_package_dir", lambda: tmp_path / "sdk")
    paths = {
        "argument": tmp_path / "explicit" / "claude", "env": tmp_path / "env" / "claude",
        "path": tmp_path / "bin" / "claude", "execpath": tmp_path / "exec" / "claude",
        "sdk_package": tmp_path / "sdk" / "_bundled" / "claude",
    }
    available = SOURCES[SOURCES.index(source):-1]
    for name in available:
        _fake_cli(paths[name])
    if "env" in available:
        monkeypatch.setenv("AUTOFA_CLAUDE_CLI", str(paths["env"]))
    if "execpath" in available:
        monkeypatch.setenv("CLAUDE_CODE_EXECPATH", str(paths["execpath"]))
    explicit = str(paths["argument"]) if source == "argument" else None
    expected = None if source == "none" else str(paths[source].resolve())
    assert find_claude_cli(explicit, required=False) == expected


def test_api_key_fallback_when_no_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_API_KEY", "CLAUDE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    missing = str(tmp_path / "no-such-claude")
    with pytest.raises(FileNotFoundError, match="CLI not found"):
        find_claude_cli(missing)
    with pytest.raises(FileNotFoundError, match="ANTHROPIC_API_KEY"):
        ensure_claude_auth(cli_path=missing)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert ensure_claude_auth(cli_path=missing) is None  # use bundled SDK CLI


# The thinking/effort mapping is shared; compiler and judge only need to pass controls through.
# ENABLED checks the budget and ADAPTIVE the effort, which ENABLED drops.
ENABLED = ("enabled", "high", {"type": "enabled", "budget_tokens": 2048}, None)
ADAPTIVE = ("adaptive", "medium", {"type": "adaptive"}, "medium")


@pytest.mark.parametrize("role,thinking,effort,expected_thinking,expected_effort", [
    ("discovery", "default", "default", None, None),
    ("discovery", "default", "low", None, "low"),
    ("discovery", *ADAPTIVE),
    ("discovery", *ENABLED),
    ("discovery", "disabled", "high", {"type": "disabled"}, None),
    *[(role, *case) for role in ("compiler", "judge") for case in (ADAPTIVE, ENABLED)],
])
def test_reasoning_controls_reach_sdk(
    tmp_path, monkeypatch, role, thinking, effort, expected_thinking, expected_effort,
):
    calls = []

    async def query(*, prompt, options):
        calls.append(options)
        yield sdk.ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
            num_turns=1, session_id="test-session", stop_reason="end_turn",
            result='{"verdicts": []}',
        )

    monkeypatch.setattr(sdk, "query", query)
    monkeypatch.setattr(claude_cli_module, "ensure_claude_auth", lambda **kwargs: None)
    monkeypatch.setattr("autofa.harness.claude.ensure_claude_auth", lambda **kwargs: None)
    monkeypatch.setattr(AgentActivity, "start", lambda self: None)
    monkeypatch.setattr(AgentActivity, "stop", lambda self, **kwargs: None)
    settings = dict(thinking=thinking, effort=effort, thinking_budget_tokens=2048)
    if role == "discovery":
        harness = get_harness("claude", "custom-model", **settings)
        asyncio.run(harness.run_turn("Inspect", cwd=str(tmp_path), system_prompt="test"))
    elif role == "compiler":
        asyncio.run(_compiler_session(
            prompt="Compile", model="custom-model", cli_path=None,
            directory=tmp_path, attempt=0, **settings,
        ))
    else:
        # One request shows its options; repairing the uncompiled reply is covered elsewhere.
        monkeypatch.setattr("autofa.metrics.claude_judge.MAX_VERDICT_REPAIRS", 0)
        review = EvidenceReview(make_snapshot(
            {"response_text": "Evidence"}, item_id="one", source="source",
        ), tmp_path / "review")
        asyncio.run(_run_claude_judge_session(
            prompt="Review", cwd=review.directory, model="custom-model", cli_path=None,
            review=review, **settings,
        ))
    assert len(calls) == 1
    assert calls[0].model == "custom-model"
    assert calls[0].max_buffer_size == CLAUDE_MAX_BUFFER_SIZE
    assert calls[0].thinking == expected_thinking
    assert calls[0].effort == expected_effort


def test_configured_buffer_preserves_complete_image_evidence():
    # Reproduce the incident's shape: a ~397 KB image appears both in the message and
    # in toolUseResult, taking one JSON event beyond the SDK's default 1 MiB buffer.
    image_bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 1550
    content = [{"type": "image", "source": {
        "type": "base64", "media_type": "image/png",
        "data": base64.b64encode(image_bytes).decode("ascii"),
    }}]
    event = {
        "type": "user",
        "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "view-artifact", "content": content,
        }]},
        "toolUseResult": content,
    }
    line = json.dumps(event, separators=(",", ":")) + "\n"
    assert 1024**2 < len(line) < CLAUDE_MAX_BUFFER_SIZE

    async def read():
        async def stdout():
            for offset in range(0, len(line), 8191):
                yield line[offset:offset + 8191]

        async def wait():
            return 0

        # Only exercise SDK framing/decoding: never connect or spawn a CLI.
        transport = SubprocessCLITransport(
            prompt="Inspect", options=ClaudeAgentOptions(max_buffer_size=CLAUDE_MAX_BUFFER_SIZE),
        )
        transport._process = SimpleNamespace(wait=wait)
        transport._stdout_stream = stdout()
        return [message async for message in transport.read_messages()]

    assert asyncio.run(read()) == [event]


# Transient Claude failures resume existing work; permanent errors remain visible.


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(AgentActivity, "start", lambda self: None)
    monkeypatch.setattr(AgentActivity, "stop", lambda self, **kwargs: None)
    h = ClaudeHarness.__new__(ClaudeHarness)
    h.model = "test-model"
    h.cli_path = None
    h.reasoning_options = {}
    return h


@pytest.fixture
def sleeps(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("autofa.harness.claude.asyncio.sleep", sleep)
    return delays


@pytest.fixture
def turn(tmp_path, monkeypatch, harness):
    """Run one harness turn against a fake SDK query generator."""

    def run(query, prompt="Inspect", **kwargs):
        monkeypatch.setattr(sdk, "query", query)
        kwargs.setdefault("system_prompt", "test")
        return asyncio.run(harness.run_turn(prompt, cwd=str(tmp_path), **kwargs))

    return run


def result(**kwargs):
    return sdk.ResultMessage(**{
        "subtype": "success", "duration_ms": 1, "duration_api_ms": 1,
        "is_error": False, "num_turns": 1, "session_id": "existing-session",
        **kwargs,
    })


def init():
    return sdk.SystemMessage(subtype="init", data={"session_id": "existing-session"})


def _saved_turn(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "turn.json").read_text())


def test_campaign_boundary_releases_activity_panel(monkeypatch, turn):
    from autofa.session.campaign_policy import CampaignBoundaryReached

    stopped = []
    monkeypatch.setattr(AgentActivity, "stop", lambda self, **kwargs: stopped.append(kwargs))

    async def query(**kwargs):
        raise CampaignBoundaryReached({"stop_reason": "spend_reserve"})
        yield  # pragma: no cover -- SDK query is an async iterator

    with pytest.raises(CampaignBoundaryReached):
        turn(query, "Continue")
    assert stopped == [{"final": "stopped"}]


@pytest.mark.parametrize("failure", ["exception", "result"])
def test_resumes_same_session_after_mid_response_disconnect(tmp_path, turn, sleeps, failure):
    calls = []
    library = tmp_path / "library.json"

    async def query(**kwargs):
        calls.append(kwargs)
        assert latest_turn_status(tmp_path)["status"] == "started"
        if len(calls) == 1:
            yield init()
            # Saved at init, so Resume works even if the process dies mid-response.
            assert json.loads((tmp_path / "session.json").read_text()) == {
                "session_id": "existing-session",
            }
            library.write_text('{"functions": ["already-saved"]}')
            message = "API Error: Connection closed mid-response. The response may be incomplete."
            if failure == "exception":
                raise RuntimeError(message)
            yield result(is_error=True, errors=[message])
        else:
            assert library.read_text() == '{"functions": ["already-saved"]}'
            yield result(result="Recovered")

    answer = turn(query, "Discover 10 rounds", system_prompt="system",
                  allowed_tools=["Read"], show_result=False)
    assert (answer.error, answer.text, answer.session_id) == (None, "Recovered", "existing-session")
    assert [call["options"].resume for call in calls] == [None, "existing-session"]
    assert calls[1]["options"].system_prompt == "system"
    assert calls[1]["options"].allowed_tools == ["Read"]
    assert "do not repeat completed work" in calls[1]["prompt"]
    assert "Original request for this turn:\nDiscover 10 rounds" in calls[1]["prompt"]
    assert sleeps == [5]
    assert latest_turn_status(tmp_path)["status"] == "finished"


def test_disconnect_before_init_retries_original_request(turn, sleeps):
    prompts = []

    async def query(**kwargs):
        prompts.append(kwargs["prompt"])
        if len(prompts) == 1:
            raise ConnectionResetError("connection reset")
        yield result(result="ok")

    answer = turn(query)
    assert prompts == ["Inspect", "Inspect"] and sleeps == [5]
    assert answer.error is None


def test_retry_limit_returns_failure_and_retains_session(tmp_path, turn, sleeps):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        yield init()
        raise RuntimeError("Connection closed mid-response")

    answer = turn(query)
    assert len(calls) == 3
    assert sleeps == [5, 10]
    assert "Connection closed mid-response" in answer.error
    assert answer.session_id == "existing-session"
    assert latest_turn_status(tmp_path)["status"] == "interrupted"


@pytest.mark.parametrize("message", ["HTTP 429 insufficient_quota", "provider error"])
def test_permanent_errors_do_not_retry(tmp_path, turn, sleeps, message):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        yield result(is_error=True, errors=[message])

    answer = turn(query)
    assert message in answer.error
    assert len(calls) == 1 and sleeps == []
    assert latest_turn_status(tmp_path)["status"] == "failed"


def _provider_failure_query(calls, failure, denied, *, model, error):
    """Report ``denied`` via a result, an exception, or only an assistant error message."""

    async def query(**kwargs):
        calls.append(kwargs)
        yield init()
        if failure.startswith("assistant"):
            text = "API Error" if failure == "assistant_error_code" else denied
            yield sdk.AssistantMessage(content=[sdk.TextBlock(text=text)], model=model, error=error)
            if failure in {"assistant_only", "assistant_error_code"}:
                return
        if failure == "exception":
            raise RuntimeError(f"Claude Code returned an error result: {denied} (exit code: 1)")
        if failure == "assistant_then_exception":
            raise RuntimeError("CLI exited with code 1")
        yield result(is_error=True, result=denied if failure == "result" else None)

    return query


# Each failure reaches a different detection branch: result, exception, or assistant-only.
@pytest.mark.parametrize("failure", ["exception", "assistant_then_result", "assistant_only"])
def test_model_denial_is_actionable_and_preserves_work(tmp_path, harness, turn, sleeps, failure):
    harness.model = "claude-opus-test"
    calls = []
    library = tmp_path / "library.json"
    library.write_text('{"functions":["saved"]}')
    denied = "There's an issue with the selected model. It may not exist or you may not " \
             "have access to it. Run --model to pick a different model."

    answer = turn(_provider_failure_query(
        calls, failure, denied, model=harness.model, error="invalid_request",
    ), "Discover")
    assert len(calls) == 1 and sleeps == []
    assert not answer.retryable
    assert "Claude Code cannot use model 'claude-opus-test'" in answer.error
    assert "Settings > LLM endpoints > AutoFA harness" in answer.error
    assert answer.session_id == "existing-session"
    assert library.read_text() == '{"functions":["saved"]}'
    assert _saved_turn(tmp_path)["state"] == latest_turn_status(tmp_path)["status"] == "failed"


@pytest.mark.parametrize("failure", ["result", "assistant_then_exception", "assistant_error_code"])
def test_login_failure_explains_recovery_and_retains_work(
    tmp_path, harness, turn, sleeps, failure,
):
    harness.cli_path = "/custom Claude/claude"
    calls = []
    library = tmp_path / "library.json"
    library.write_text('{"functions":["saved"]}')

    answer = turn(_provider_failure_query(
        calls, failure, "Not logged in · Please run /login",
        model=harness.model, error="authentication_failed",
    ), "Discover")
    assert len(calls) == 1 and sleeps == []
    assert not answer.retryable
    assert answer.text is None
    assert "Claude Code authentication failed" in answer.error
    assert "'/custom Claude/claude' auth login" in answer.error
    assert answer.session_id == "existing-session"
    assert library.read_text() == '{"functions":["saved"]}'
    saved = _saved_turn(tmp_path)
    assert (saved["state"], saved["prompt"], saved["session_id"]) == (
        "failed", "Discover", "existing-session",
    )
    assert latest_turn_status(tmp_path)["status"] == "failed"


@pytest.mark.parametrize("text", [
    "This trace failed because the model was not found.",
    "The trace says: Not logged in · Please run /login",
])
def test_successful_answer_quoting_provider_errors_remains_successful(harness, turn, text):
    async def query(**kwargs):
        yield sdk.AssistantMessage(content=[sdk.TextBlock(text=text)], model=harness.model)
        yield result(result=text)

    answer = turn(query)
    assert answer.error is None
    assert answer.text == text


def test_cancellation_during_backoff_stops_retry(tmp_path, monkeypatch, turn):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        yield result(is_error=True, errors=["HTTP 503 temporarily unavailable"])

    async def sleep(_):
        raise asyncio.CancelledError

    monkeypatch.setattr("autofa.harness.claude.asyncio.sleep", sleep)
    answer = turn(query)
    assert len(calls) == 1
    assert answer.stopped is True
    assert answer.session_id == "existing-session"
    assert latest_turn_status(tmp_path)["status"] == "interrupted"


def test_partial_work_without_session_id_is_not_replayed(turn):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        yield sdk.AssistantMessage(content=[sdk.TextBlock(text="Work started")], model="test")
        raise RuntimeError("Connection closed mid-response")

    answer = turn(query)
    assert answer.error
    assert answer.retryable is False
    assert len(calls) == 1


def test_completed_turn_is_not_repeated_on_cli_teardown_error(tmp_path, turn):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        yield result(result="Completed all requested work")
        raise RuntimeError("Connection closed mid-response")

    answer = turn(query)
    assert answer.error is None
    assert answer.text == "Completed all requested work"
    assert len(calls) == 1
    assert latest_turn_status(tmp_path)["status"] == "finished"


@pytest.mark.parametrize("fields", [None, {"stop_reason": "max_tokens"}])
def test_incomplete_turn_is_not_reported_as_success_or_retried(turn, fields):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        if fields is not None:
            yield result(**fields)

    answer = turn(query)
    assert answer.error == "Claude turn ended without confirmed normal completion"
    assert len(calls) == 1


def test_explicit_abort_is_not_retried_even_with_network_error(tmp_path, turn):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        yield result(is_error=True, errors=["Connection closed mid-response"],
                     terminal_reason="aborted_tools")
        raise RuntimeError("Connection closed mid-response")

    answer = turn(query)
    assert answer.stopped
    assert len(calls) == 1
    assert latest_turn_status(tmp_path)["status"] == "interrupted"


@pytest.mark.parametrize("failure", ["exception", "terminal_reason"])
def test_full_context_compacts_same_session_without_repeating_saved_work(tmp_path, turn, failure):
    calls = []
    original = "Continue to ten validated modes; preserve the selected scope"
    (tmp_path / "library.json").write_text('{"functions":["saved-detector"]}')
    (tmp_path / "config.json").write_text('{"round":4,"max_rounds":8}')
    saved = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    async def query(**kwargs):
        calls.append(kwargs)
        assert kwargs["options"].resume == "full-session"
        if len(calls) == 1:
            if failure == "exception":
                raise RuntimeError("Claude Code returned an error result: Prompt is too long")
            yield result(session_id="full-session",
                         terminal_reason="model_context_window_exceeded")
        elif len(calls) == 2:
            assert kwargs["prompt"] == "/compact"
            assert kwargs["options"].fork_session is False
            yield sdk.SystemMessage(subtype="compact_boundary", data={
                "compact_metadata": {"trigger": "manual", "pre_tokens": 190000},
            })
            yield result(session_id="full-session", result="Compacted")
        else:
            assert "not a new discovery round" in kwargs["prompt"]
            assert original in kwargs["prompt"]
            assert all((tmp_path / name).read_bytes() == content
                       for name, content in saved.items())
            yield result(session_id="full-session", result="Continued from saved detectors")

    answer = turn(query, original, system_prompt="same policy", resume="full-session")
    assert answer.error is None and answer.session_id == "full-session"
    assert len(calls) == 3
    assert all(c["options"].system_prompt == "same policy" for c in calls)
    assert _saved_turn(tmp_path)["prompt"] == original
    assert not (tmp_path / "context_recovery").exists()
    assert latest_turn_status(tmp_path)["status"] == "finished"


def test_context_recovery_is_bounded_if_compacted_prompt_is_also_too_large(tmp_path, turn):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        if kwargs["prompt"] == "/compact":
            yield sdk.SystemMessage(subtype="compact_boundary", data={})
            yield result(result="Compacted")
        else:
            yield result(is_error=True, errors=["Context length exceeded"])

    answer = turn(query)
    assert "Context length exceeded" in answer.error
    assert answer.session_id == "existing-session"
    assert len(calls) == 4
    assert all(call["options"].resume == "existing-session" for call in calls[1:3])
    assert calls[-1]["options"].resume is None
    assert latest_turn_status(tmp_path)["status"] == "interrupted"
    saved = _saved_turn(tmp_path)
    assert saved["resumable"] and saved["recovery_reason"] == "context_recovery_exhausted"


@pytest.mark.parametrize("failure", ["no_boundary", "error", "exception", "other_session"])
def test_failed_native_compaction_preserves_request_and_replaces_recoverable_session(
    tmp_path, turn, failure,
):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            yield result(is_error=True, errors=["Prompt is too long"])
            return
        if len(calls) == 3:
            assert kwargs["options"].resume is None
            assert "Continue the selected scope" in kwargs["prompt"]
            yield result(session_id="replacement", result="Recovered")
            return
        assert kwargs["options"].resume == "existing-session"
        assert kwargs["prompt"] == "/compact"
        if failure == "exception":
            raise RuntimeError("Compaction service unavailable")
        if failure != "no_boundary":
            yield sdk.SystemMessage(subtype="compact_boundary", data={})
        yield result(
            result="Not enough messages to compact." if failure == "no_boundary" else None,
            is_error=failure == "error", errors=["Compaction failed"] if failure == "error" else [],
            session_id="different-session" if failure == "other_session" else "existing-session",
        )

    answer = turn(query, "Continue the selected scope", resume="existing-session")
    saved = _saved_turn(tmp_path)
    assert saved["prompt"] == "Continue the selected scope"
    if failure == "other_session":
        assert "Claude native compaction failed" in answer.error
        assert answer.session_id == "existing-session" and len(calls) == 2
        assert saved["state"] == latest_turn_status(tmp_path)["status"] == "failed"
    else:
        assert not answer.error and answer.session_id == "replacement" and len(calls) == 3
        assert saved["state"] == latest_turn_status(tmp_path)["status"] == "finished"
    assert saved["session_id"] == answer.session_id


def test_manual_compaction_preserves_pending_turn_and_forwards_focus(tmp_path, turn):
    calls = []
    pending = '{"prompt":"Original request","session_id":"existing-session","state":"running"}'
    (tmp_path / "turn.json").write_text(pending)

    async def query(**kwargs):
        calls.append(kwargs)
        yield sdk.SystemMessage(subtype="compact_boundary", data={})
        yield result(result="Compacted")

    answer = turn(query, "/compact Retain all selected scope constraints",
                  resume="existing-session", allowed_tools=["Read"])
    assert answer.error is None
    assert answer.session_id == "existing-session"
    assert len(calls) == 1
    assert calls[0]["prompt"] == "/compact Retain all selected scope constraints"
    assert calls[0]["options"].allowed_tools == ["Read"]
    assert (tmp_path / "turn.json").read_text() == pending


def test_manual_compaction_requires_existing_session(tmp_path, turn):
    def query(**kwargs):
        pytest.fail("Compaction without a session must not start a model call")

    answer = turn(query, "/compact")
    assert "Start or resume" in answer.error
    assert answer.session_id is None
    assert not (tmp_path / "turn.json").exists()


def test_cancellation_during_native_compaction_keeps_session(tmp_path, turn):
    calls = []

    async def query(**kwargs):
        calls.append(kwargs)
        if kwargs["prompt"] == "/compact":
            raise asyncio.CancelledError
        yield result(is_error=True, errors=["Prompt is too long"])

    answer = turn(query)
    assert answer.stopped is True
    assert answer.session_id == "existing-session"
    assert len(calls) == 2
    assert _saved_turn(tmp_path)["state"] == latest_turn_status(tmp_path)["status"] == "interrupted"


def test_context_handoff_preserves_conversation_but_bounds_prompt(tmp_path, monkeypatch):
    messages = [
        SimpleNamespace(type="user", message={"content": "Keep user constraint " * 4000}),
        SimpleNamespace(type="assistant", message={"content": [
            {"type": "text", "text": "Completed evaluation; next inspect missing evidence"},
        ]}),
    ]
    monkeypatch.setattr(sdk, "get_session_messages", lambda *args, **kwargs: messages)
    prompt = context_recovery_prompt(tmp_path, prompt="Continue", session_id="old")
    assert len(prompt) < 2000
    handoff = json.loads(next((tmp_path / "context_recovery").glob("*/handoff.json")).read_text())
    assert handoff["prior_requests"] == [messages[0].message["content"]]
    assert handoff["last_assistant_text"].startswith("Completed evaluation")
    saved = tmp_path / "tool_results" / (handoff["conversation"]["result_id"] + ".json")
    conversation = json.loads(json.loads(saved.read_text())["text"])
    assert conversation[0]["message"] == messages[0].message
