# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness sessions: model access errors, stops and failures end discovery without losing it."""

import asyncio
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from autofa.harness import claude, codex
from autofa.harness.base import BatchSessionCancelled
from autofa.harness.errors import is_model_access_error, model_access_message
from autofa.ui.errors import show_action_error


@pytest.mark.parametrize("message,expected", [
    ("The 'gpt-6-astra' model is not supported when using Codex with a ChatGPT account.", True),
    ("The model `gpt-6-astra` does not exist or you do not have access to it.", True),
    ('API Error: 403 {"error":{"type":"permission_error","message":"model: opus"}}', True),
    ('{"code":"model_not_found"}', True),
    ("Unknown model", True),
    ("HTTP 503: model temporarily unavailable", False),
    ("HTTP 401: invalid API key", False),
    ("This model's context window is full", False),
    ("Model does not support reasoning effort ultra", False),
    ("The model response uses a tool that is not supported by the sandbox", False),
])
def test_model_access_errors_are_distinguished_from_other_failures(message, expected):
    assert is_model_access_error(message) is expected


@pytest.fixture(params=["claude", "codex"])
def session(request, tmp_path, monkeypatch):
    """A Claude or Codex harness whose turns and prompts each test replaces."""
    import prompt_toolkit

    module, harness_type = (
        (claude, claude.ClaudeHarness)
        if request.param == "claude" else (codex, codex.CodexHarness)
    )
    harness = harness_type.__new__(harness_type)
    harness.model, harness.cli_path, harness.effort = "selected-model", None, None
    output = StringIO()
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setattr(module, "console", Console(file=output, width=200))
    monkeypatch.setattr(module, "show_action_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda **kwargs: object())
    monkeypatch.setattr("autofa.ui.menus.prepare_step", lambda: None)
    if request.param == "claude":
        monkeypatch.setattr(module, "describe_auth_mode", lambda path: "test")

    async def end_session(session):
        raise EOFError

    monkeypatch.setattr(module, "_async_prompt", end_session)
    return SimpleNamespace(provider=request.param, module=module, harness=harness, output=output)


def test_model_denial_keeps_session_usable_and_stops_automatic_discovery(
    session, tmp_path, monkeypatch,
):
    harness = session.harness
    monkeypatch.setattr(session.module, "show_action_error", show_action_error)
    prompts = iter(["/help"])
    turns = []

    async def prompt(_):
        command = next(prompts, None)
        if command is None:
            raise EOFError
        return command

    async def turn(text, **kwargs):
        turns.append(text)
        return claude.TurnResult(
            session_id="saved-session", error=model_access_message(session.provider, harness.model),
        )

    monkeypatch.setattr(session.module, "_async_prompt", prompt)
    monkeypatch.setattr(harness, "run_turn", turn)
    result = asyncio.run(harness.run_session(
        cwd=str(tmp_path), system_prompt="test", initial_prompt="/discover",
        discovery_continuation=lambda: pytest.fail("Discovery must stop after model denial"),
    ))
    assert result == "saved-session"
    assert turns == ["/discover"]  # /help is handled locally after the failure.
    assert "Settings > LLM endpoints" in session.output.getvalue()


@pytest.mark.parametrize("batch", [False, True])
def test_validation_reopens_automatic_discovery_in_same_session(
    session, tmp_path, monkeypatch, batch,
):
    import prompt_toolkit

    harness = session.harness
    if batch:
        monkeypatch.setattr(prompt_toolkit, "PromptSession",
                            lambda **kwargs: pytest.fail("Batch discovery must not prompt"))
    turns, finalizations = [], []
    followups = iter(["Keep investigating", None, None])
    validation_results = iter(["Full validation found only 2/3 Good; continue discovery", None])

    async def turn(prompt, **kwargs):
        turns.append((prompt, kwargs["resume"]))
        return claude.TurnResult(session_id="retained-session")

    async def finalize():
        finalizations.append(len(turns))
        return next(validation_results)

    monkeypatch.setattr(harness, "run_turn", turn)
    result = asyncio.run(harness.run_session(
        cwd=str(tmp_path), system_prompt="test", initial_prompt="/discover",
        resume="existing-session", discovery_continuation=lambda: next(followups),
        discovery_finalization=finalize, exit_after_discovery=batch,
    ))

    assert result == "retained-session"
    assert turns == [
        ("/discover", "existing-session"),
        ("Keep investigating", "retained-session"),
        ("Full validation found only 2/3 Good; continue discovery", "retained-session"),
    ]
    assert finalizations == [2, 3]


def test_interactive_user_stop_ignores_validation_continuation(session, tmp_path, monkeypatch):
    harness = session.harness
    turns, finalizations = [], []

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        return claude.TurnResult(session_id="retained-session", stopped=True)

    async def finalize():
        finalizations.append(True)
        return "Validation requests more discovery, but the user stopped"

    monkeypatch.setattr(harness, "run_turn", turn)
    result = asyncio.run(harness.run_session(
        cwd=str(tmp_path), system_prompt="test", initial_prompt="/discover",
        discovery_continuation=lambda: pytest.fail("A user stop cannot request followup work"),
        discovery_finalization=finalize,
    ))

    assert result == "retained-session"
    assert turns == ["/discover"]
    assert finalizations == [True]


def test_batch_user_stop_propagates_without_finalizing_or_restarting(
    session, tmp_path, monkeypatch,
):
    harness = session.harness

    async def turn(prompt, **kwargs):
        return claude.TurnResult(session_id="retained-session", stopped=True)

    async def finalize():
        pytest.fail("Batch cancellation must propagate before finalization")

    monkeypatch.setattr(harness, "run_turn", turn)
    with pytest.raises(BatchSessionCancelled):
        asyncio.run(harness.run_session(
            cwd=str(tmp_path), system_prompt="test", initial_prompt="/discover",
            discovery_continuation=lambda: pytest.fail("Cancelled discovery cannot continue"),
            discovery_finalization=finalize, exit_after_discovery=True,
        ))


def test_finalization_failure_does_not_restart_discovery(session, tmp_path, monkeypatch):
    harness = session.harness
    turns = []

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        return claude.TurnResult(session_id="retained-session")

    async def finalize():
        raise RuntimeError("Final validation interrupted")

    monkeypatch.setattr(harness, "run_turn", turn)
    with pytest.raises(RuntimeError, match="Final validation interrupted"):
        asyncio.run(harness.run_session(
            cwd=str(tmp_path), system_prompt="test", initial_prompt="/discover",
            discovery_continuation=lambda: None, discovery_finalization=finalize,
            exit_after_discovery=True,
        ))
    assert turns == ["/discover"]


@pytest.mark.parametrize("error_and_stopped", [False, True])
def test_turn_error_never_finalizes_or_restarts(
    session, tmp_path, monkeypatch, error_and_stopped,
):
    harness = session.harness
    turns = []

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        return claude.TurnResult(
            session_id="retained-session", error="Provider unavailable", stopped=error_and_stopped,
        )

    async def finalize():
        pytest.fail("Failed discovery must not finalize or restart automatically")

    monkeypatch.setattr(harness, "run_turn", turn)
    result = asyncio.run(harness.run_session(
        cwd=str(tmp_path), system_prompt="test", initial_prompt="/discover",
        discovery_continuation=lambda: pytest.fail("Failed discovery cannot continue"),
        discovery_finalization=finalize,
    ))
    assert result == "retained-session"
    assert turns == ["/discover"]


def test_ordinary_conversation_never_enters_validation_loop(session, tmp_path, monkeypatch):
    harness = session.harness
    prompts, turns = iter(["Explain this detector"]), []

    async def prompt(_):
        value = next(prompts, None)
        if value is None:
            raise EOFError
        return value

    async def turn(text, **kwargs):
        turns.append(text)
        return claude.TurnResult(session_id="retained-session")

    async def finalize():
        pytest.fail("An ordinary conversation must not enter the discovery loop")

    monkeypatch.setattr(session.module, "_async_prompt", prompt)
    monkeypatch.setattr(harness, "run_turn", turn)
    asyncio.run(harness.run_session(
        cwd=str(tmp_path), system_prompt="test",
        discovery_continuation=lambda: pytest.fail("An ordinary conversation cannot continue"),
        discovery_finalization=finalize,
    ))
    assert turns == ["Explain this detector"]
