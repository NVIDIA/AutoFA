# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi's own logs, as pi writes them under its log directory.

    pi/sessions/<ts>_<session id>.jsonl   pi's transcript, one record per message or setting
    pi.txt                                its JSON-mode stdout, tool progress updates cut
    pi.full.txt                           the same stdout with every delta

Which directory that is, and which trial it belongs to, is the host's: the `Mount` places it.
"""

from __future__ import annotations

from evalrdl_readers.core import AnyOf, Card, Landmark

LOGS = Landmark(
    name="pi_logs",
    marker=AnyOf("pi.txt", "pi.full.txt"),
    where=("",),
    expect=Card.ONE(
        because=(
            "pi writes its stdout into the directory it logs to; a run ended from outside "
            "leaves `pi.full.txt` and no `pi.txt`, so either marks it"
        ),
    ),
)
