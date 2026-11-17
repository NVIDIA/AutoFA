# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""store-v13: v12 plus the repeat beside every task_id that names a rollout.

A rollout is (task_id, repeat). `provenance` and `relations` named it by task_id alone, which
was unambiguous only while readers put the repeat inside the task id; once a task keeps one
id across its repeats, the repeat has to travel with it. Rows written before read as NULL.
"""

from __future__ import annotations

from ..dsl import Add, Column, Dtype, Schema, evolve
from ..v12 import V12

V13: Schema = evolve(
    V12,
    "store-v13",
    note="the repeat beside task_id in provenance and relations",
    add=(
        Add(
            "provenance",
            columns=(
                Column(
                    "repeat",
                    Dtype.INT,
                    "the repeat of the one rollout this record covers, else NULL",
                ),
            ),
        ),
        Add(
            "relations",
            columns=(
                Column(
                    "frm_repeat",
                    Dtype.INT,
                    "the repeat of the rollout the edge leaves, where it has one",
                ),
                Column(
                    "to_repeat",
                    Dtype.INT,
                    "the repeat of the rollout the edge points at, where it has one",
                ),
            ),
        ),
    ),
)
