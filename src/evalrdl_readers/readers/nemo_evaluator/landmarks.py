# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The nemo-evaluator envelope: the wrapper's files beside the harness's output."""

from __future__ import annotations

from evalrdl_readers.core import AnyOf, Card, Landmark

RUN_CONFIG = "run_config.yml"
METADATA = "metadata.yaml"
RESULTS = "results.yml"
CONFIG = "config.yml"
FACTORY_METRICS = "eval_factory_metrics.json"
EXPORT_CONFIG = "export_config.yml"
EVAL_METADATA = "eval_metadata.json"

ENVELOPE = Landmark(
    name="ne_envelope",
    marker=AnyOf(
        RUN_CONFIG, METADATA, RESULTS, CONFIG, FACTORY_METRICS, EXPORT_CONFIG, EVAL_METADATA
    ),
    where=("",),
    expect=Card.ONE(
        because=(
            "the wrapper writes its files into one directory per evaluation; a copy may keep "
            "any one of them"
        ),
    ),
)
