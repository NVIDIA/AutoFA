# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where inspect_ai writes its logs: one JSON file per task, in one `logs/` directory."""

from __future__ import annotations

from evalrdl_readers.core import Card, Landmark

LOGS = Landmark(
    name="inspect_logs",
    marker="*.json",
    where=("logs",),
    expect=Card.ONE(
        because=(
            "inspect writes one log file per task, named after the task, into one `logs/` "
            "directory, and the run is all of them together"
        ),
    ),
)
