# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common protocol for AutoFA's discovery harnesses."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class BatchSessionCancelled(RuntimeError):
    pass


class Harness(Protocol):
    name: str

    async def run_turn(
        self, prompt: str, *, cwd: str, system_prompt: str,
        mcp_servers: dict[str, Any] | None = None, allowed_tools: list[str] | None = None,
        resume: str | None = None, show_result: bool = True, activity_hint: str | None = None,
    ) -> Any:
        """Run one resumable discovery turn."""
        ...

    async def run_session(
        self,
        *,
        cwd: str,
        system_prompt: str,
        initial_prompt: str | None = None,
        mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        resume: str | None = None,
        discovery_continuation: Callable[[], str | None] | None = None,
        discovery_finalization: Callable[[], Awaitable[str | None]] | None = None,
        turn_validation: Callable[[], Awaitable[str | None]] | None = None,
        exit_after_discovery: bool = False,
    ) -> str | None:
        """Run an interactive or batch agent session. Returns session_id if available.

        After automatic completion, finalization may return a discovery prompt
        when validation leaves work unfinished. A user stop never resumes work.
        """
        ...
