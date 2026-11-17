# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""codex's own logs, as codex writes them under its home directory.

    sessions/YYYY/MM/DD/rollout-<ts>-<session id>.jsonl   codex's transcript and turn settings
    codex.txt                                            its JSON-mode stdout

Which directory that is, and which trial it belongs to, is the host's: the `Mount` places it.
"""

from __future__ import annotations

from evalrdl_readers.core import AnyOf, Card, Landmark

LOGS = Landmark(
    name="codex_logs",
    marker=AnyOf("codex.txt", "agent_error.json", "last_llm_error.json"),
    where=("",),
    expect=Card.ONE(
        because=(
            "codex's stdout, or the error files a failed run leaves instead, sit in the "
            "directory its logs are kept in"
        ),
    ),
)
