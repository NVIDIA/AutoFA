# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve discovery and judge settings for the selected execution harness."""

from __future__ import annotations

import subprocess
from typing import Any

from autofa.defaults import default_value, harness_default

HARNESS_NAMES = {"claude": "Claude Code", "codex": "Codex"}


class HarnessNotFoundError(FileNotFoundError):
    """No authenticated harness is available for a new AutoFA configuration."""


def _authentication_status(binary: str, harness: str) -> str | None:
    """Return a setup problem, or None when the CLI reports authenticated credentials."""
    command = [binary, "login", "status"] if harness == "codex" else [binary, "auth", "status"]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5, check=False,
        )
    except subprocess.TimeoutExpired:
        return "authentication check timed out"
    except OSError:
        return "could not run authentication check"
    # Both status commands document exit 0 as authenticated and exit 1 as logged out.
    # Never display their raw output: it may include account or credential details.
    if result.returncode == 0:
        return None
    if result.returncode == 1:
        return "not signed in"
    return "could not verify authentication; update the CLI and check its login status"


def detect_default_harness(
    *, codex_cli: str | None = None, claude_cli: str | None = None,
) -> str:
    """Prefer authenticated Codex, then Claude Code, using login checks without inference."""
    from autofa.harness.claude_cli import find_claude_cli
    from autofa.harness.model_catalog import _find_codex_cli

    try:
        codex_binary = _find_codex_cli(codex_cli)
    except FileNotFoundError:
        codex_problem = "not installed or executable not found"
    else:
        codex_problem = _authentication_status(codex_binary, "codex")
        if codex_problem is None:
            return "codex"
    # The SDK dependency's binary also works if the user's Claude credentials
    # are configured. Its presence alone is not evidence of authentication.
    claude_binary = find_claude_cli(claude_cli, required=False)
    claude_problem = "not installed or executable not found"
    if claude_binary:
        claude_problem = _authentication_status(claude_binary, "claude")
        if claude_problem is None:
            return "claude"
    raise HarnessNotFoundError(
        "No authenticated harness found. AutoFA requires Codex CLI or Claude Code.\n"
        f"  Codex: {codex_problem}.\n"
        f"  Claude Code: {claude_problem}.\n"
        "Install a harness if needed, then sign in:\n"
        "  Codex CLI: https://developers.openai.com/codex/cli/\n"
        "    Then run: codex login\n"
        "  Claude Code: https://code.claude.com/docs/en/quickstart\n"
        "    Then run: claude auth login\n"
        "Make sure codex or claude is executable on PATH, or set\n"
        "AUTOFA_CODEX_CLI=/path/to/codex or AUTOFA_CLAUDE_CLI=/path/to/claude.\n"
        "For the bundled Claude SDK, set ANTHROPIC_API_KEY in .env or the environment.\n"
        "Verify with codex login status or claude auth status, then restart autofa.\n"
        "No new configuration was saved."
    )


def judge_harness(cfg: Any) -> str:
    selected = getattr(cfg, "behavior_judge_harness", default_value("behavior_judge_harness"))
    return getattr(cfg, "harness", default_value("harness")) if selected == "auto" else selected


def harness_model(harness: str, model: str, *, judge: bool = False, cfg: Any = None) -> str:
    # Switching only the harness in an existing TOML file should also switch
    # its untouched shipped model. Explicit custom deployments remain intact.
    if harness not in HARNESS_NAMES:
        return model
    field = "behavior_judge_model" if judge else "model"
    for other in HARNESS_NAMES:
        if other != harness and model in {
            harness_default(other, field), harness_default(other, field, cfg),
        }:
            return harness_default(harness, field, cfg)
    return model


def harness_cli(cfg: Any, harness: str) -> str | None:
    field = f"{harness}_cli"
    return getattr(cfg, field, default_value(field)) or None


def behavior_judge_settings(cfg: Any) -> dict[str, Any]:
    harness = judge_harness(cfg)

    def configured(name):
        field = "behavior_judge_" + name
        return getattr(cfg, field, harness_default(harness, field, cfg))

    return {
        "harness": harness,
        "model": harness_model(harness, configured("model"), judge=True, cfg=cfg),
        "cli_path": harness_cli(cfg, harness),
        "effort": configured("effort"),
        "thinking": (harness_default(harness, "behavior_judge_thinking", cfg)
                     if harness == "codex" else configured("thinking")),
        "thinking_budget_tokens": configured("thinking_budget_tokens"),
    }


def validate_codex_effort(effort: str) -> None:
    # Codex advertises extensible strings, including levels beyond Claude's
    # SDK enum. The model picker supplies its actual supported choices.
    if not isinstance(effort, str) or not effort.strip() or len(effort) > 64:
        raise ValueError("effort must be default or a nonblank Codex reasoning effort")


def validate_harness_settings(cfg: Any) -> None:
    from autofa.harness.thinking import reasoning_options

    if not isinstance(cfg.harness, str) or cfg.harness not in HARNESS_NAMES:
        raise ValueError("harness must be claude or codex")
    if (not isinstance(cfg.behavior_judge_harness, str)
            or cfg.behavior_judge_harness not in {"auto", *HARNESS_NAMES}):
        raise ValueError("behavior_judge_harness must be auto, claude or codex")
    for prefix, harness in (("", cfg.harness), ("behavior_judge_", judge_harness(cfg))):
        try:
            if harness == "codex":
                validate_codex_effort(getattr(cfg, prefix + "effort"))
            else:
                reasoning_options(
                    thinking=getattr(cfg, prefix + "thinking"),
                    effort=getattr(cfg, prefix + "effort"),
                    thinking_budget_tokens=getattr(cfg, prefix + "thinking_budget_tokens"),
                )
        except ValueError as exc:
            raise ValueError(prefix + str(exc)) from exc
