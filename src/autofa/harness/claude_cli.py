# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code CLI resolution.

Auth modes (prefer first available):
1. Installed Claude Code CLI (`claude` on PATH) — subscription login
2. ``ANTHROPIC_API_KEY`` (or ``CLAUDE_CODE_API_KEY``) — API key; Agent SDK
   uses its bundled CLI. Put the key in the project ``.env``.
3. The Claude Code binary bundled inside an installed ``claude-agent-sdk`` —
   agent sandboxes run AutoFA with an API key but no ``claude`` on ``PATH``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_CLI_CANDIDATES = (
    "claude",
    str(Path.home() / ".local" / "bin" / "claude"),
    "/usr/local/bin/claude",
)

API_KEY_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_API_KEY",
    "CLAUDE_API_KEY",
)

_UNSET = object()


def _sdk_package_dir() -> Path | None:
    """Directory of the installed ``claude_agent_sdk`` package, if any."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("claude_agent_sdk")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def _bundled_sdk_cli() -> str | None:
    """Claude Code binary shipped inside claude-agent-sdk, when it has one.

    Agent SDK sandboxes spawn that binary and advertise it as
    ``CLAUDE_CODE_EXECPATH``; otherwise it sits in the package's ``_bundled``.
    """
    candidates: list[Path] = []
    exec_path = os.environ.get("CLAUDE_CODE_EXECPATH", "").strip()
    if exec_path:
        candidates.append(Path(exec_path).expanduser())
    sdk_dir = _sdk_package_dir()
    if sdk_dir is not None:
        candidates.append(sdk_dir / "_bundled" / "claude")
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return None


def get_anthropic_api_key() -> str | None:
    for name in API_KEY_ENV_NAMES:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return None


def has_anthropic_api_key() -> bool:
    return get_anthropic_api_key() is not None


def find_claude_cli(explicit: str | None = None, *, required: bool = True) -> str | None:
    """Resolve path to Claude Code binary.

    Returns ``None`` when not found and ``required=False`` (API-key mode).
    """
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        which = shutil.which(explicit)
        if which:
            return which
        if required:
            raise FileNotFoundError(f"Claude Code CLI not found: {explicit}")
        return None

    env = os.environ.get("CLAUDE_CLI") or os.environ.get("AUTOFA_CLAUDE_CLI")
    if env:
        return find_claude_cli(env, required=required)

    for cand in DEFAULT_CLI_CANDIDATES:
        which = shutil.which(cand)
        if which:
            return str(Path(which).resolve())

    bundled = _bundled_sdk_cli()
    if bundled:
        return bundled

    if not required:
        return None

    raise FileNotFoundError(
        "Claude Code CLI not found.\n"
        "  • Install Claude Code: https://docs.claude.com/en/docs/claude-code\n"
        "  • Or set ANTHROPIC_API_KEY in the project .env / environment\n"
        "    (Agent SDK will use its bundled CLI with the API key).\n"
        "  • Optional: AUTOFA_CLAUDE_CLI=/path/to/claude"
    )


def describe_auth_mode(cli_path: str | None | object = _UNSET) -> str:
    """Human-readable auth mode. Pass ``None`` to mean API-key / bundled CLI."""
    if cli_path is _UNSET:
        cli = find_claude_cli(required=False)
    else:
        cli = cli_path  # type: ignore[assignment]
    if cli:
        return f"Claude Code CLI ({cli})"
    if has_anthropic_api_key():
        key_env = next(n for n in API_KEY_ENV_NAMES if os.environ.get(n, "").strip())
        return f"ANTHROPIC API key ({key_env}) · bundled Agent SDK CLI"
    return "not configured"


def ensure_claude_auth(*, cli_path: str | None = None) -> str | None:
    """Return resolved cli_path (or None for bundled+API key). Raise if neither works."""
    cli = find_claude_cli(cli_path, required=False)
    if cli:
        return cli
    if has_anthropic_api_key():
        return None
    raise FileNotFoundError(
        "No Claude auth configured.\n"
        "Install Claude Code (https://docs.claude.com/en/docs/claude-code) OR set "
        "ANTHROPIC_API_KEY in your .env file, then restart autofa."
    )
