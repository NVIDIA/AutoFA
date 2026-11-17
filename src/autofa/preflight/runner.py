# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run configured detectors on the analysis selection, without selecting data again."""

from pathlib import Path

from autofa.config import Config, RunConfig
from autofa.preflight import PreflightResult
from autofa.preflight.registry import detectors, validate_detectors


def run_preflight(run: RunConfig, cfg: Config) -> list[PreflightResult]:
    """Caller holds the run lock. Resume reuses detector caches and retries failures."""
    enabled = validate_detectors(run.preflight_detectors)
    if not enabled:
        return []
    from autofa.data.parquet_store import Dataset

    dataset = Dataset(
        parquet_paths=[Path(path) for path in run.parquet_paths],
        model_id=run.model_id, benchmarks=run.resolved_benchmarks(),
    )
    results = []
    for detector in detectors():
        if detector.name not in enabled:
            continue
        print(f"Preflight: {detector.label}")
        result = detector.load().run_dataset(
            dataset, cfg, run.run_dir() / "preflight" / detector.name,
            sources=run.reference_data,
        )
        results.append(result)
        if result.status:
            raise RuntimeError(
                f"{detector.label} preflight failed (status {result.status}); "
                f"see {result.report}"
            )
    return results
