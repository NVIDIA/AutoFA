# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime estimates: bounded workload profiling, calibrated forecasts and the live ETA."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config
from autofa.estimates import (
    SCHEMA_VERSION,
    Estimator,
    LiveRuntimeEstimate,
    RoundProgress,
    Workload,
    active_discovery_seconds,
    live_timing_label,
    mark_model_fallback,
    model_signature,
    profile_workload,
    round_progress,
)


def sample(cfg, work, seconds=(600, 1200, 1800)):
    return {"schema_version": SCHEMA_VERSION, "signature": model_signature(cfg),
            "workload": asdict(work), "top_x": 5,
            "round_seconds": {str(i): v for i, v in enumerate(seconds)}}


def event(seconds, state=None, round_index=None):
    return {"ts": datetime.fromtimestamp(seconds, UTC).isoformat(),
            "source": "harness" if state else "advance_round",
            "context": {"turn_status": state} if state else {"round": round_index}}


def test_profile_counts_filtered_rows_and_samples_proportionally_across_row_groups(tmp_path):
    paths = []
    for index, n in enumerate([20, 100]):
        path = tmp_path / f"{index}.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"model_id": "m", "benchmark": "b", "response_text": "x" * (i + 1)}
            for i in range(n)
        ] + [{"model_id": "other", "benchmark": "ignored", "response_text": "z"}]),
                       path, row_group_size=10)
        paths.append(path)
    work = profile_workload(paths, "m", ["b"])
    assert work.rows == 120
    assert work.sampled_rows == 48
    assert 35 < work.mean_chars < 50  # Includes groups across both files, not only first rows.
    assert profile_workload(paths, "missing").rows == 0


def test_forecast_uses_observed_round_totals_and_range_without_scaling():
    cfg, work = Config(), Workload(100, 1000, 2, 48)
    history = [sample(cfg, work, (600, 1200)), sample(cfg, work, (1200, 1500)),
               sample(cfg, work, (1800, 1800))]
    estimate = Estimator(cfg, work, history).estimate(2)
    assert (estimate.low_minutes, estimate.median_minutes, estimate.high_minutes) == (30, 45, 60)
    assert not estimate.rough
    # Size/depth matches select observations; no invented runtime multiplier.
    assert Estimator(cfg, replace(work, rows=110), history).estimate(2) == estimate
    assert Estimator(cfg, work, history).estimate(3).rough


@pytest.mark.parametrize("cfg,work", [
    pytest.param(replace(Config(), model="different"), Workload(100, 1000, 2, 48),
                 id="other_model"),
    pytest.param(Config(), Workload(10000, 1000, 2, 48), id="other_workload"),
])
def test_mismatched_history_only_supplies_a_labeled_rough_range(cfg, work):
    history = [sample(Config(), Workload(100, 1000, 2, 48))] * 3
    estimate = Estimator(cfg, work, history).estimate(1)
    assert estimate.rough
    assert "uncalibrated" in estimate.basis


def test_round_progress_excludes_idle_and_invalidates_only_interrupted_rounds():
    rows = [event(100, "started"), event(160, "finished"), event(1000, "started"),
            event(1060, round_index=1), event(1120, "finished")]
    progress = round_progress(rows)
    assert progress.completed == {0: 120}  # 840 idle seconds excluded.
    assert progress.round_index == 1  # This round did not close, even though the turn finished.
    rows = [event(0, "started"), event(60, round_index=1), event(80, "interrupted"),
            event(1000, "started"), event(1060, round_index=2), event(1120, round_index=3)]
    assert round_progress(rows).completed == {0: 60, 2: 60}


def test_model_fallback_estimate_failure_does_not_block_discovery(tmp_path, monkeypatch):
    context = {"schema_version": SCHEMA_VERSION}
    path = tmp_path / "estimate_context.json"
    path.write_text(json.dumps(context))

    def failed_write(*args):
        raise OSError("read-only directory")

    monkeypatch.setattr("autofa.util.atomic.atomic_write_json", failed_write)
    mark_model_fallback(tmp_path)
    assert json.loads(path.read_text()) == context


def test_live_label_reports_remaining_time_and_range():
    text = live_timing_label(RoundProgress({0: 600, 1: 1200}, 2, 300), 3)
    assert "Round 3 of 3 · ETA:" in text
    assert "~10 min remaining" in text
    assert "Range: 5–15 min" in text


def test_live_reader_uses_this_run(tmp_path, monkeypatch):
    (tmp_path / "estimate_context.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION}))
    (tmp_path / "config.json").write_text('{"max_rounds": 3}')
    rows = [event(0, "started"), event(600, round_index=1), event(1800, round_index=2)]
    (tmp_path / "status_log.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr("autofa.estimates.time.time", lambda: 2100)
    assert "Range: 5–15 min" in LiveRuntimeEstimate(tmp_path)()


def test_elapsed_accumulates_all_turns_and_interrupted_work_without_idle():
    rows = [
        event(0, "started"), event(60, round_index=1), event(100, "finished"),
        event(1000, "started"), event(1060, "interrupted"),
        event(2000, "started"), event(2030, "failed"),
        event(3000, "started"),
    ]
    assert active_discovery_seconds(rows, now=3020) == 210
    assert active_discovery_seconds([], now=3020) is None
    # Without an end timestamp, an abandoned turn's duration is unknown, so a later
    # start never turns the overnight gap before Resume into active work.
    abandoned = [event(0, "started"), event(60, "finished"),
                 event(1000, "started"), event(90000, "started")]
    assert active_discovery_seconds(abandoned, now=90020) == 80
