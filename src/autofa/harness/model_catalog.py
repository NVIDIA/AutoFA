# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Available models reported by the installed harness, without running a prompt."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from autofa.harness.claude_cli import find_claude_cli
from autofa.harness.claude_transport import CLAUDE_MAX_BUFFER_SIZE


@dataclass(frozen=True)
class HarnessModel:
    """A model selector and the capabilities advertised by its harness.

    Keep selectors such as ``opus`` intact: they can resolve to gateway-specific
    models. ``None`` denotes unknown capabilities for manually entered models.
    """

    value: str
    display_name: str
    resolved_model: str | None = None
    description: str = ""
    supports_effort: bool | None = None
    effort_levels: tuple[str, ...] = ()
    supports_adaptive_thinking: bool | None = None
    default_effort: str | None = None


@dataclass(frozen=True)
class ModelCatalog:
    models: tuple[HarnessModel, ...] = ()
    error: str | None = None

    def find(self, model_id: str) -> HarnessModel | None:
        """Find a saved selector or concrete model ID, preferring exact IDs."""
        for model in self.models:
            if model.value == model_id:
                return model
        resolved = [model for model in self.models if model.resolved_model == model_id]
        # A pinned model should not acquire the label of the moving default
        # alias when a model-specific row describes it accurately.
        return next(
            (model for model in resolved if model.value != "default"),
            resolved[0] if resolved else None,
        )


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _parse_catalog(info: Any) -> ModelCatalog:
    raw_models = info.get("models") if isinstance(info, dict) else None
    if not isinstance(raw_models, list):
        return ModelCatalog(
            error="Claude Code did not report available models. Update the harness.",
        )
    models: list[HarnessModel] = []
    seen: set[str] = set()
    for raw in raw_models:
        if not isinstance(raw, dict) or raw.get("disabled") is True:
            continue
        value = raw.get("value")
        if not isinstance(value, str) or not value.strip() or value in seen:
            continue
        seen.add(value)
        name = raw.get("displayName")
        resolved = raw.get("resolvedModel")
        description = raw.get("description")
        levels = raw.get("supportedEffortLevels")
        models.append(
            HarnessModel(
                value=value,
                display_name=name if isinstance(name, str) and name else value,
                resolved_model=resolved if isinstance(resolved, str) and resolved else None,
                description=description if isinstance(description, str) else "",
                # Claude Code omits false capability flags from ModelInfo.
                supports_effort=_optional_bool(raw.get("supportsEffort", False)),
                effort_levels=tuple(
                    dict.fromkeys(level for level in levels if isinstance(level, str) and level)
                ) if isinstance(levels, list) else (),
                supports_adaptive_thinking=_optional_bool(
                    raw.get("supportsAdaptiveThinking", False),
                ),
            )
        )
    if not models:
        return ModelCatalog(
            error="Claude Code reported no available models. Check its configuration.",
        )
    return ModelCatalog(models=tuple(models))


async def _load_claude_model_catalog_async(
    cli_path: str | None = None, *, timeout_s: float = 15.0,
) -> ModelCatalog:
    """Read the harness initialization metadata, without sending any user turn.

    An isolated working directory, disabled hooks/tools/MCP, and no persistence
    keep opening settings from running project automation. Authentication and
    model routing remain those of the installed Claude Code CLI. Cancellation
    propagates, allowing a closed settings form to terminate its subprocess.
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        binary = find_claude_cli(cli_path)
        with TemporaryDirectory(prefix="autofa-models-") as cwd:
            options = ClaudeAgentOptions(
                cli_path=binary,
                max_buffer_size=CLAUDE_MAX_BUFFER_SIZE,
                cwd=cwd,
                # Preserve the installed harness's model picker and auth;
                # project automation is excluded by the isolated directory.
                setting_sources=["user"],
                tools=[],
                settings=json.dumps({"disableAllHooks": True}),
                extra_args={"no-session-persistence": None, "strict-mcp-config": None},
                env={"CLAUDE_CODE_SAFE_MODE": "1"},
            )
            async with asyncio.timeout(timeout_s):
                async with ClaudeSDKClient(options=options) as client:
                    return _parse_catalog(await client.get_server_info())
    except TimeoutError:
        return ModelCatalog(
            error="Claude Code model discovery timed out. Retry or enter a model ID.",
        )
    except FileNotFoundError:
        return ModelCatalog(error="Claude Code was not found. Check the configured CLI path.")
    except ImportError:
        return ModelCatalog(error="Install claude-agent-sdk to load Claude Code models.")
    except Exception:
        # CLI errors can contain auth helpers, stderr, or gateway credentials.
        # Do not expose them in the settings UI.
        return ModelCatalog(
            error="Could not load Claude Code models. Check its authentication and setup.",
        )


def _parse_codex_catalog(rows: list[Any]) -> ModelCatalog:
    models: list[HarnessModel] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict) or raw.get("hidden") is True:
            continue
        value = raw.get("model")
        if not isinstance(value, str) or not value.strip() or value in seen:
            continue
        seen.add(value)
        name = raw.get("displayName")
        description = raw.get("description")
        raw_levels = raw.get("supportedReasoningEfforts")
        levels = tuple(dict.fromkeys(
            option["reasoningEffort"] for option in raw_levels
            if isinstance(option, dict)
            and isinstance(option.get("reasoningEffort"), str)
            and option["reasoningEffort"].strip()
        )) if isinstance(raw_levels, list) else ()
        default = raw.get("defaultReasoningEffort")
        models.append(HarnessModel(
            value=value,
            display_name=name if isinstance(name, str) and name else value,
            resolved_model=value,
            description=description if isinstance(description, str) else "",
            supports_effort=bool(levels),
            effort_levels=levels,
            supports_adaptive_thinking=False,
            default_effort=default if isinstance(default, str) and default else None,
        ))
    if not models:
        return ModelCatalog(error="Codex reported no available models. Check its configuration.")
    return ModelCatalog(models=tuple(models))


def _find_codex_cli(explicit: str | None) -> str:
    configured = explicit or os.environ.get("AUTOFA_CODEX_CLI") or os.environ.get("CODEX_CLI")
    candidates = [configured] if configured else ["codex", str(Path.home() / ".local/bin/codex")]
    for candidate in candidates:
        if candidate and (binary := shutil.which(str(Path(candidate).expanduser()))):
            return binary
    raise FileNotFoundError("Codex CLI not found")


async def _codex_send(process: asyncio.subprocess.Process, message: dict[str, Any]) -> None:
    assert process.stdin is not None
    process.stdin.write((json.dumps(message) + "\n").encode())
    await process.stdin.drain()


async def _codex_request(
    process: asyncio.subprocess.Process, request_id: int, method: str, params: dict[str, Any],
) -> dict[str, Any]:
    await _codex_send(process, {"id": request_id, "method": method, "params": params})
    assert process.stdout is not None
    while line := await process.stdout.readline():
        message = json.loads(line)
        if not isinstance(message, dict) or message.get("id") != request_id:
            continue
        if "error" in message or not isinstance(message.get("result"), dict):
            raise RuntimeError("Codex model discovery request failed")
        return message["result"]
    raise RuntimeError("Codex closed the model discovery connection")


async def _stop_codex_process(process: asyncio.subprocess.Process) -> None:
    if process.stdin is not None:
        process.stdin.close()
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        await process.wait()


async def _load_codex_model_catalog_async(
    cli_path: str | None = None, *, timeout_s: float = 15.0,
) -> ModelCatalog:
    """Read Codex's model/list catalog; never start a thread or inference turn."""
    try:
        binary = _find_codex_cli(cli_path)
        with TemporaryDirectory(prefix="autofa-codex-models-") as cwd:
            process = await asyncio.create_subprocess_exec(
                binary, "app-server", "--listen", "stdio://", cwd=cwd,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, limit=1024 * 1024,
            )
            try:
                async with asyncio.timeout(timeout_s):
                    await _codex_request(process, 1, "initialize", {
                        "clientInfo": {"name": "autofa-model-picker", "version": "0.1.0"},
                        # Metadata discovery must not launch a browser login.
                        "capabilities": {"explicitGatewayOauth": True},
                    })
                    await _codex_send(process, {"method": "initialized", "params": {}})
                    rows: list[Any] = []
                    cursor = None
                    seen_cursors: set[str] = set()
                    request_id = 2
                    while True:
                        params: dict[str, Any] = {"limit": 100, "includeHidden": False}
                        if cursor is not None:
                            params["cursor"] = cursor
                        page = await _codex_request(process, request_id, "model/list", params)
                        if not isinstance(page.get("data"), list):
                            raise RuntimeError("Codex returned an invalid model catalog")
                        rows.extend(page["data"])
                        cursor = page.get("nextCursor")
                        if cursor is None:
                            return _parse_codex_catalog(rows)
                        if not isinstance(cursor, str) or cursor in seen_cursors:
                            raise RuntimeError("Codex returned an invalid model cursor")
                        seen_cursors.add(cursor)
                        request_id += 1
            finally:
                await _stop_codex_process(process)
    except TimeoutError:
        return ModelCatalog(error="Codex model discovery timed out. Retry or enter a model ID.")
    except FileNotFoundError:
        return ModelCatalog(error="Codex was not found. Install Codex CLI to browse its models.")
    except Exception:
        return ModelCatalog(
            error="Could not load Codex models. Check its authentication and setup.",
        )


async def load_model_catalog_async(
    cli_path: str | None = None, *, harness: str = "claude", timeout_s: float = 15.0,
) -> ModelCatalog:
    """Ask the selected installed harness for its models without inference.

    A harness may return a bundled or cached catalog; inclusion does not verify
    account entitlement. Cancellation propagates and closes the metadata process.
    """
    if harness == "claude":
        return await _load_claude_model_catalog_async(cli_path, timeout_s=timeout_s)
    if harness == "codex":
        return await _load_codex_model_catalog_async(cli_path, timeout_s=timeout_s)
    return ModelCatalog(error=f"Model discovery is unavailable for harness: {harness}")


def load_model_catalog(
    cli_path: str | None = None, *, harness: str = "claude", timeout_s: float = 15.0,
) -> ModelCatalog:
    """Synchronous entry point for callers outside an event loop."""
    return asyncio.run(load_model_catalog_async(cli_path, harness=harness, timeout_s=timeout_s))
