# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Claude Code compaction, preserving the existing conversation identity."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any


class ClaudeCompactionError(RuntimeError):
    """The harness did not confirm that its native compaction completed."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.code = code
        super().__init__(message)


async def compact_claude_session(
    *, options: Any, session_id: str, instructions: str = "",
    on_message: Callable[[Any], Awaitable[None]] | None = None,
) -> None:
    """Dispatch /compact and require its compact_boundary acknowledgment.

    A success result alone is insufficient: Claude Code also returns success
    when there is no history to compact, or a command was not dispatched.
    """
    from claude_agent_sdk import ResultMessage, SystemMessage, query

    from autofa.harness.usage import track_claude_messages

    if not session_id:
        raise ClaudeCompactionError("Start or resume a Claude session before using /compact.")
    compact_options = replace(
        options, resume=session_id, continue_conversation=False, fork_session=False,
        max_turns=1, verbatim_prompts=False,
    )
    prompt = "/compact" + (" " + instructions.strip() if instructions.strip() else "")
    compacted = False
    result = None
    async for message in track_claude_messages(
        query(prompt=prompt, options=compact_options), run_dir=getattr(options, "cwd", None),
        model=getattr(options, "model", None), effort=getattr(options, "effort", None),
        resume=session_id, role="compaction",
    ):
        if on_message is not None:
            await on_message(message)
        if isinstance(message, SystemMessage) and message.subtype == "compact_boundary":
            compacted = True
        if isinstance(message, ResultMessage):
            result = message
            if message.session_id and message.session_id != session_id:
                raise ClaudeCompactionError("Claude returned a different session while compacting.")
    if result is None or result.is_error or result.subtype != "success":
        detail = "; ".join(getattr(result, "errors", None) or [])
        raise ClaudeCompactionError(
            detail or "Claude did not complete native compaction.",
            code="compactionUnconfirmed" if not detail or detail.lower() == "compaction failed"
            else None,
        )
    if not compacted:
        reason = str(result.result or "No native compact_boundary acknowledgment.")[:240]
        raise ClaudeCompactionError(reason, code="compactionUnconfirmed")
