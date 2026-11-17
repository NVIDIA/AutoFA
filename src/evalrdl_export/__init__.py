# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exporters: a run, read through the declared schema, written in another tool's format.

api.py          the contract: `Exporter`, its scopes, `write_parquet`
autofa.py       AutoFA: one row per rollout, its conversation, as parquet
trace.py        trace: every rollout-keyed table, a ZIP of parquet files
conformance.py  what every exporter must pass, for a registry to run over its own exporters
"""

from .api import ROLLOUT, RUN, SCOPES, Exporter, ExportTooLarge, Only, write_parquet
from .autofa import AUTOFA
from .trace import TRACE

#: The exporters this package ships. A service adds its own beside them.
PUBLIC: dict[str, Exporter] = {e.name: e for e in (AUTOFA, TRACE)}

__all__ = [
    "AUTOFA",
    "PUBLIC",
    "ExportTooLarge",
    "Exporter",
    "Only",
    "ROLLOUT",
    "RUN",
    "SCOPES",
    "TRACE",
    "write_parquet",
]
