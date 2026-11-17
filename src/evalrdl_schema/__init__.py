# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stored schema, declared and versioned.

    dsl.py        the IR: Column, Table, Schema, and `evolve`, which only ever adds
    v11/          the deployed schema, frozen
    v12/          v11 + the reader layer's columns, as a delta
    registry.py   VERSIONS and CURRENT
    backends.py   the IR mapped onto polars / duckdb / postgres types
"""

from .backends import (
    DUCK,
    PARQUET_ROW_GROUP_BYTES,
    POSTGRES,
    grains,
    keys,
    parquet_row_group_rows,
    polars_schemas,
    sql_literal,
    widest_value,
)
from .dsl import (
    Add,
    Column,
    Diff,
    Dtype,
    Grain,
    Refusal,
    Schema,
    Table,
    check_additive,
    diff,
    evolve,
)
from .registry import CURRENT, CURRENT_DIALECTS, EARLIEST, VERSIONS
from .v11 import V11
from .v12 import LANE_DEFAULT, V12

__all__ = [
    "Add", "Column", "Diff", "Dtype", "Grain", "Refusal", "Schema", "Table",
    "check_additive", "diff", "evolve",
    "CURRENT", "CURRENT_DIALECTS", "EARLIEST", "VERSIONS",
    "V11", "V12", "LANE_DEFAULT",
    "DUCK", "POSTGRES", "grains", "keys", "polars_schemas", "sql_literal",
    "PARQUET_ROW_GROUP_BYTES", "parquet_row_group_rows", "widest_value",
]
