# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trace of one rollout: where it is read from, and how it is served.

source.py       `TraceSource`: what a store answers about one rollout; `ArtifactSource` and
                `ArtifactResolver`: where an artifact's bytes come from
assemble.py     what is computed from those answers, the same over every source
routes.py       the HTTP routes over any `TraceSource`, and the artifact routes over any
                `ArtifactSource` (the `serve` extra)
conformance.py  what every `TraceSource` must hold to
tables.py       `TablesSource`: a source over one run's tables on disk (the `tables` extra)
static/         the viewer: trace.js, its graph mode (graph.js), the helpers it shares with its
                page (kit.js), style.css, and viewer.html, a page of its own
"""

from pathlib import Path

from .source import ArtifactBytes, ArtifactResolver, ArtifactSource, RolloutKey, TraceSource

#: The viewer's files, served as `/static/<name>`.
STATIC = Path(__file__).resolve().parent / "static"

__all__ = [
    "STATIC",
    "ArtifactBytes",
    "ArtifactResolver",
    "ArtifactSource",
    "RolloutKey",
    "TraceSource",
]
