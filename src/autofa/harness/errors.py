# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe, actionable explanations for model selection failures in either harness."""

from __future__ import annotations

import re
import shlex


def is_claude_auth_error(message: str) -> bool:
    """Identify a login/credential failure, without conflating model permissions."""
    text = " ".join(message.lower().split())
    return any(marker in text for marker in (
        "not logged in", "please run /login", "authentication_error", "authentication_failed",
        "invalid api key", "invalid x-api-key", "invalid authentication credentials",
        "oauth token has expired", "oauth token is expired",
    ))


def claude_auth_message(cli_path: str | None = None) -> str:
    """Direct users to the actual CLI login, which AutoFA's prompt cannot run."""
    command = shlex.join([cli_path or "claude", "auth", "login"])
    return (
        "Claude Code authentication failed. In another terminal, run "
        f"`{command}`, then retry /discover here or resume the saved AutoFA run. "
        "If you use an API key, correct the configured key and restart AutoFA. "
        "AutoFA's prompt does not implement Claude's /login; `autofa login` signs in "
        "to the library service only. Your saved library and conversation are retained."
    )


def is_model_access_error(message: str) -> bool:
    """Recognize provider rejections without treating temporary outages as access failures."""
    text = " ".join(message.lower().split())
    if re.search(
        r"\b(?:model_not_found|model_not_supported|unsupported_model|model_access_denied|"
        r"model_not_allowed|invalid_model|unknown_model)\b", text,
    ):
        return True
    if re.search(r"\b(?:unknown|invalid|unsupported) model\b", text):
        return True
    if not re.search(r"\bmodel\b", text):
        return False
    if re.search(r"\bmodel(?:\s+[`'\"]?[\w./:-]+[`'\"]?)?\s+(?:is\s+)?not supported\b", text):
        return True
    return any(marker in text for marker in (
        "not exist", "doesn't exist", "not found", "not_found_error",
        "not have access", "don't have access",
        "doesn't have access", "no access to", "not have permission",
        "not authorized", "not authorised", "not allowed", "not permitted",
        "not entitled", "access denied", "access_denied", "permission_denied",
        "permission_error", "not available for", "not available to",
        "not enabled for", "not enabled on", "not included in",
    ))


def model_access_message(provider: str, model: str | None = None) -> str:
    """Use our own wording; upstream error bodies can contain credentials."""
    selected = f"model '{model}'" if model else "the selected model"
    return (
        f"{provider} cannot use {selected}: it is unavailable or your account does not have "
        "access. Choose a model your account can use in Settings > LLM endpoints > "
        "AutoFA harness (or Judge endpoint for judging), or ask your administrator to "
        "enable access. In a session, press Ctrl+D to return to the menu, then change "
        "Settings and resume the saved run."
    )
