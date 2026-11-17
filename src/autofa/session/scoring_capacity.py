# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed queue capacity, separate from classifier identity and quality."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from autofa.defaults import setting_value
from autofa.util.atomic import atomic_write_json
from autofa.util.capacity import effective_llm_workers


def capacity_observer(rt: Any, workers: int):
    started = time.monotonic()
    last = 0.0
    completed = 0
    requested = int(setting_value(rt.config, "llm_eval_workers"))
    effective = effective_llm_workers(rt.config)

    def observe(event: dict) -> None:
        nonlocal last, completed
        now = time.monotonic()
        completed += 1
        if now - last < 5:
            return
        last = now
        snapshot = {
            "version": 1, "observed_at": datetime.now(UTC).isoformat(), **event,
            "requested_workers": requested, "effective_workers": workers,
            "configured_ceiling": int(setting_value(rt.config, "llm_eval_max_workers")),
            "limiting_reason": ("queue_bytes" if event.get("byte_limited") else
                                "configured_ceiling" if effective < requested
                                and workers == effective else
                                "stream_limit" if workers < effective else None),
            "completed_before_current_result": completed - 1,
            "observed_results_per_second": (completed - 1) / max(now - started, 0.001),
            "scope": "current scoring stream, including reused cache results",
        }
        try:
            atomic_write_json(rt.run.run_dir() / "scoring_capacity.json", snapshot)
        except OSError:
            logging.getLogger(__name__).warning("Unable to save scoring capacity observation")

    return observe
