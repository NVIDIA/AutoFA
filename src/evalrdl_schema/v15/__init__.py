# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""store-v15: v14 plus the tool definitions a rollout's model was offered.

`tool_defs` holds one row per distinct definition in first-seen order, keyed by rollout like
`artifacts`; the definition's JSON is in `tool_def_texts`, the texts tier, as `part_texts` is
to `parts`. Runs written before hold no rows.
"""

from __future__ import annotations

from ..dsl import Column, Dtype, Grain, Schema, Table, evolve
from ..v14 import V14

_KEY = ("run_id", "task_id", "repeat", "agent_ref", "def_idx")


def _key_columns() -> tuple[Column, ...]:
    return (
        Column("run_id", Dtype.STR, "the run"),
        Column("task_id", Dtype.STR, "the rollout's task"),
        Column("repeat", Dtype.INT, "the rollout's repeat"),
        Column("agent_ref", Dtype.STR, "the rollout's agent"),
        Column(
            "def_idx",
            Dtype.INT,
            "the definition's position among the rollout's distinct definitions, in the order "
            "first offered: the rollout record's `responses_create_params`, then each captured "
            "request's",
        ),
    )


V15: Schema = evolve(
    V14,
    "store-v15",
    note="the tool definitions a rollout's model was offered",
    tables=(
        Table(
            "tool_defs",
            Grain.TRACE,
            key=_KEY,
            columns=_key_columns()
            + (
                Column("name", Dtype.STR, "the tool's name, `name` or `function.name`"),
                Column(
                    "tool_type", Dtype.STR, "the definition's own `type`, as the producer wrote it"
                ),
                Column("def_chars", Dtype.INT, "the length of the definition's JSON"),
                Column(
                    "sha256",
                    Dtype.STR,
                    "of the definition's JSON: one tool set offered to many rollouts is one "
                    "value, comparable without the texts tier",
                ),
                Column(
                    "offered_by",
                    Dtype.STR,
                    "the model call id of the first captured request that carried this exact "
                    "definition, the `model_call` lane's `source_event_id`; NULL where the "
                    "rollout record's `responses_create_params` carried it",
                ),
            ),
            means="one distinct tool definition offered to the model in one rollout",
        ),
        Table(
            "tool_def_texts",
            Grain.TEXTS,
            key=_KEY,
            columns=_key_columns()
            + (Column("def_json", Dtype.STR, "the definition as the producer wrote it, as JSON"),),
            means="the full JSON of a `tool_defs` row, in the texts tier",
        ),
    ),
)
