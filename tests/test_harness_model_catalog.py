# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness metadata discovery must never issue an inference request."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import claude_agent_sdk
import pytest

from autofa.harness import model_catalog
from autofa.harness.claude_transport import CLAUDE_MAX_BUFFER_SIZE
from autofa.harness.model_catalog import HarnessModel, ModelCatalog


def test_parse_catalog_preserves_harness_aliases_and_capabilities():
    catalog = model_catalog._parse_catalog({"models": [
        {
            "value": "opus",
            "resolvedModel": "gateway/team-opus",
            "displayName": "Team Opus",
            "description": "Organization model",
            "supportsEffort": True,
            "supportedEffortLevels": ["low", "high", "max"],
            "supportsAdaptiveThinking": True,
        },
        {"value": "restricted", "disabled": True},
        # Claude Code omits false capability flags.
        {"value": "haiku", "displayName": "Haiku", "disabled": False},
    ]})

    opus, haiku = catalog.models
    assert catalog.error is None
    assert opus == HarnessModel(
        value="opus", display_name="Team Opus", resolved_model="gateway/team-opus",
        description="Organization model", supports_effort=True,
        effort_levels=("low", "high", "max"), supports_adaptive_thinking=True,
    )
    assert catalog.find("gateway/team-opus") is opus
    assert catalog.find("opus") is opus
    assert catalog.find("custom-model") is None
    assert haiku.supports_effort is False
    assert haiku.supports_adaptive_thinking is False
    assert haiku.effort_levels == ()


def test_lookup_prefers_exact_selector_then_model_label_over_default_alias():
    default = HarnessModel("default", "Default", resolved_model="claude-sonnet-latest")
    sonnet = HarnessModel("sonnet", "Sonnet", resolved_model="claude-sonnet-latest")
    catalog = ModelCatalog((default, sonnet))
    assert catalog.find("claude-sonnet-latest") is sonnet
    assert catalog.find("default") is default
    assert ModelCatalog((default,)).find("claude-sonnet-latest") is default
    alias = HarnessModel("default", "Default", resolved_model="sonnet")
    assert ModelCatalog((alias, sonnet)).find("sonnet") is sonnet


def test_unknown_capabilities_remain_unknown_and_malformed_entries_are_ignored():
    catalog = model_catalog._parse_catalog({"models": [
        None, "opus", {}, {"value": ""}, {"value": 3},
        {"value": "custom", "displayName": 1, "resolvedModel": {},
         "supportsEffort": "yes", "supportsAdaptiveThinking": 1,
         "supportedEffortLevels": ["low", "low", None, "future-effort"]},
        {"value": "custom", "displayName": "Duplicate"},
    ]})
    assert len(catalog.models) == 1
    model = catalog.models[0]
    assert model.display_name == "custom"
    assert model.resolved_model is None
    assert model.supports_effort is None
    assert model.supports_adaptive_thinking is None
    assert model.effort_levels == ("low", "future-effort")


@pytest.mark.parametrize("info", [None, {"models": []}])
def test_missing_catalog_is_reported_without_inventing_models(info):
    catalog = model_catalog._parse_catalog(info)
    assert catalog.models == ()
    assert catalog.error


def _install_client(monkeypatch, *, wait=False, error=None):
    state = {"entered": False, "closed": False}
    monkeypatch.setattr(model_catalog, "find_claude_cli", lambda path: path or "/test/claude")

    class MetadataClient:
        def __init__(self, options):
            state["options"] = options

        async def __aenter__(self):
            state["entered"] = True
            return self

        async def __aexit__(self, *exc):
            state["closed"] = True

        async def get_server_info(self):
            if error:
                raise error
            if wait:
                await asyncio.Event().wait()
            return {"models": [{"value": "harness-new-model", "displayName": "New model"}]}

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", MetadataClient)
    return state


def test_discovery_reads_only_metadata_with_project_automation_disabled(monkeypatch):
    state = _install_client(monkeypatch)
    catalog = model_catalog.load_model_catalog("/explicit/claude")
    assert catalog.models[0].value == "harness-new-model"
    assert state["entered"] and state["closed"]
    options = state["options"]
    assert options.cli_path == "/explicit/claude"
    assert options.max_buffer_size == CLAUDE_MAX_BUFFER_SIZE
    assert options.tools == []
    assert options.setting_sources == ["user"]
    assert json.loads(options.settings) == {"disableAllHooks": True}
    assert options.env["CLAUDE_CODE_SAFE_MODE"] == "1"
    assert options.extra_args == {"no-session-persistence": None, "strict-mcp-config": None}
    assert not Path(options.cwd).exists()


def test_cli_errors_do_not_expose_credentials(monkeypatch):
    state = _install_client(monkeypatch, error=RuntimeError("Authorization: secret-token"))
    catalog = model_catalog.load_model_catalog()
    assert "secret-token" not in catalog.error
    assert catalog.models == ()
    assert state["closed"]


def test_codex_catalog_preserves_returned_efforts_defaults_and_model_id():
    catalog = model_catalog._parse_codex_catalog([
        {"id": "picker-id", "model": "actual-model-id", "displayName": "New model",
         "description": "Available from Codex", "isDefault": True,
         "defaultReasoningEffort": "ultra", "supportedReasoningEfforts": [
             {"reasoningEffort": "low"}, {"reasoningEffort": "ultra"},
             {"reasoningEffort": "ultra"}, {"reasoningEffort": ""}, None,
         ]},
        {"model": "hidden-model", "hidden": True},
        {"model": "actual-model-id", "displayName": "Duplicate"},
        {"model": "custom-model"}, None, {}, {"model": 1},
    ])
    assert len(catalog.models) == 2
    first, custom = catalog.models
    assert first.value == first.resolved_model == "actual-model-id"
    assert first.display_name == "New model"
    assert first.description == "Available from Codex"
    assert first.default_effort == "ultra"
    assert first.effort_levels == ("low", "ultra")
    assert first.supports_effort is True
    assert first.supports_adaptive_thinking is False
    assert custom.display_name == "custom-model"
    assert custom.supports_effort is False


def _install_codex_process(monkeypatch, *, pages=None, wait=False, error=None):
    state = {"messages": [], "closed": False, "terminated": False, "listed": False}
    monkeypatch.setattr(model_catalog, "_find_codex_cli", lambda path: path or "/test/codex")
    pages = pages or [{"data": [{"model": "codex-model"}]}]

    class Process:
        returncode = None

        def __init__(self):
            self.stdin = self
            self.stdout = self
            self.lines = asyncio.Queue()
            self.stopped = asyncio.Event()
            self.page = 0

        def write(self, data):
            message = json.loads(data)
            state["messages"].append(message)
            if message["method"] == "initialize":
                response = {"id": message["id"], "result": {"userAgent": "Codex"}}
            elif message["method"] == "model/list":
                state["listed"] = True
                if wait:
                    return
                response = {"id": message["id"], "result": pages[self.page]}
                self.page += 1
                if error:
                    response = {"id": message["id"], "error": error}
            else:
                assert message["method"] == "initialized"
                return
            self.lines.put_nowait(b'{"method":"notification","params":{}}\n')
            self.lines.put_nowait(json.dumps(response).encode() + b"\n")

        async def drain(self):
            pass

        async def readline(self):
            return await self.lines.get()

        def close(self):
            state["closed"] = True

        def terminate(self):
            state["terminated"] = True
            self.returncode = 0
            self.stopped.set()

        async def wait(self):
            await self.stopped.wait()

    async def create_process(*args, **kwargs):
        state["args"] = args
        state["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    return state


def test_codex_discovery_initializes_then_paginates_without_starting_thread(monkeypatch):
    state = _install_codex_process(monkeypatch, pages=[
        {"data": [{"model": "first"}], "nextCursor": "page-2"},
        {"data": [{"model": "second"}], "nextCursor": None},
    ])
    catalog = model_catalog.load_model_catalog("/explicit/codex", harness="codex")
    assert [model.value for model in catalog.models] == ["first", "second"]
    assert state["args"] == ("/explicit/codex", "app-server", "--listen", "stdio://")
    assert state["kwargs"]["stderr"] == asyncio.subprocess.DEVNULL
    assert not Path(state["kwargs"]["cwd"]).exists()
    assert state["closed"] and state["terminated"]
    messages = state["messages"]
    assert [message["method"] for message in messages] == [
        "initialize", "initialized", "model/list", "model/list",
    ]
    assert messages[0]["params"]["capabilities"] == {"explicitGatewayOauth": True}
    assert messages[2]["params"] == {"limit": 100, "includeHidden": False}
    assert messages[3]["params"]["cursor"] == "page-2"


@pytest.mark.parametrize("pages", [
    [{"data": {}}],
    [{"data": [], "nextCursor": "repeated"}, {"data": [], "nextCursor": "repeated"}],
], ids=["data_not_list", "repeated_cursor"])
def test_invalid_codex_catalog_fails_cleanly(monkeypatch, pages):
    state = _install_codex_process(monkeypatch, pages=pages)
    catalog = model_catalog.load_model_catalog(harness="codex")
    assert catalog.models == ()
    assert catalog.error
    assert state["closed"] and state["terminated"]


def test_codex_rpc_errors_do_not_expose_server_details(monkeypatch):
    state = _install_codex_process(monkeypatch, error={"message": "Bearer private-secret"})
    catalog = model_catalog.load_model_catalog(harness="codex")
    assert "private-secret" not in catalog.error
    assert state["closed"] and state["terminated"]


def _unanswered(monkeypatch, harness):
    """A metadata client that never answers, its started flag and its release flags."""
    if harness == "claude":
        return _install_client(monkeypatch, wait=True), "entered", ("closed",)
    return _install_codex_process(monkeypatch, wait=True), "listed", ("closed", "terminated")


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_discovery_timeout_releases_metadata_client(monkeypatch, harness):
    state, _, released = _unanswered(monkeypatch, harness)
    catalog = model_catalog.load_model_catalog(harness=harness, timeout_s=0.01)
    assert catalog.models == ()
    assert "timed out" in catalog.error
    assert all(state[key] for key in released)


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_cancelling_discovery_releases_metadata_client_and_propagates(monkeypatch, harness):
    state, started, released = _unanswered(monkeypatch, harness)

    async def run():
        task = asyncio.create_task(model_catalog.load_model_catalog_async(harness=harness))
        while not state[started]:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert all(state[key] for key in released)


@pytest.mark.parametrize("harness,finder,message", [
    ("claude", "find_claude_cli", "not found"), ("codex", "_find_codex_cli", "Codex was not found"),
])
def test_missing_configured_cli_is_reported_without_its_details(
    monkeypatch, harness, finder, message,
):
    def missing(path):
        raise FileNotFoundError("Secret helper argument")

    monkeypatch.setattr(model_catalog, finder, missing)
    catalog = model_catalog.load_model_catalog("/missing/cli", harness=harness)
    assert message in catalog.error
    assert "Secret" not in catalog.error


def test_unimplemented_harness_discovery_does_not_guess_a_catalog():
    catalog = model_catalog.load_model_catalog(harness="unknown")
    assert catalog.models == ()
    assert "unavailable" in catalog.error
