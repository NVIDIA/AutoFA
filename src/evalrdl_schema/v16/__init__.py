# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""store-v16: v15 plus which readers decoded a run and the revision that ran them.

`harness` and `binding_version` name the binding cell, which the rebinding worklist, reidentify
and the reparse guard key on; they say nothing about the layers whose files were read. Runs
written before hold NULL.
"""

from __future__ import annotations

from ..dsl import Add, Column, Dtype, Schema, evolve
from ..v15 import V15

V16: Schema = evolve(
    V15,
    "store-v16",
    note="the readers that decoded a run, and the revision that ran them",
    add=(
        Add(
            "runs",
            columns=(
                Column(
                    "reader_stack",
                    Dtype.STR,
                    "the producer layers whose files answered this run's facts, outermost "
                    "first, as the declared reader names them (`nemo-evaluator / nemo-gym`); "
                    "`adapter:<harness>` where the adapter matrix decoded it",
                ),
                Column(
                    "reader_rev",
                    Dtype.STR,
                    "the EvalRDL build that decoded the run: its version "
                    "`MAJ.MIN.BUILD-YYYYMMDD-<sha7>` (EVALRDL_VERSION), or a bare git "
                    "revision where the build has none (EVALRDL_GIT_REVISION)",
                ),
            ),
        ),
    ),
)
