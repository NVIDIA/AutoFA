# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""store-v14: v13 plus the file in the key of `provenance`, and the site in the key of `relations`.

The bridge writes one provenance row per (question, input, file), but the key named the
directory (`site`), so several files under one directory shared a key: Postgres, which dedups
by key, kept one of them and parquet kept all. `file` is the path the row's values came from,
or "" for a row minted rather than read; rows written before read as "".

`relations` named an end by (input, row name), and a row's name is counted within one input's
output at ONE site, so four gdpval deliverables directories each gave an edge from
`gdpval_metadata/0`: one key, four edges. `frm_site`/`to_site` complete the address.
"""

from __future__ import annotations

from ..dsl import Add, Column, Dtype, Schema, evolve
from ..v13 import V13

V14: Schema = evolve(
    V13,
    "store-v14",
    note="the file in the provenance key, the site in the relations key",
    add=(
        Add(
            "provenance",
            columns=(
                Column(
                    "file",
                    Dtype.STR,
                    "the file this record's values came from, relative to the run root; "
                    '"" for a row minted rather than read. In the key: one (question, input) '
                    "reads several files under one site",
                    backfill="",
                ),
            ),
            key=("run_id", "question", "input", "site", "file"),
        ),
        Add(
            "relations",
            columns=(
                Column(
                    "frm_site",
                    Dtype.STR,
                    "the site the row the edge leaves was read at; its name is unique only there",
                    backfill="",
                ),
                Column(
                    "to_site",
                    Dtype.STR,
                    'the site the row the edge points at was read at; "" where the edge '
                    "is unresolved",
                    backfill="",
                ),
            ),
            key=(
                "run_id",
                "space",
                "frm_question",
                "frm_key",
                "to_question",
                "to_key",
                "kind",
                "frm_site",
                "to_site",
            ),
        ),
    ),
)
