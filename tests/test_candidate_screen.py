# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only cheap evidence prerequisites can hold a candidate back from corpus evaluation."""

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import server
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.session import candidate_screen as screen

NOT_REQUIRED = {
    "ok": True, "name": "marker", "status": "not_required", "evaluation_allowed": True,
    "provisional": True, "cache_reused": False, "controls": [],
    "feedback": "No causal control screen is required; evaluation may proceed.",
}


class UnreadDataset(Dataset):
    def iter_batches(self, **kwargs):
        pytest.fail("Candidate checks must not scan the corpus")


@pytest.fixture
def screen_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    fn = FeatureFunction(
        name="marker",
        source="def marker(item):\n    return 'EXECUTED' in item['response_text']\n",
        description="Executes a destructive command before preserving the required data.",
        mechanism="Premature destruction removes the evidence needed to complete the task.",
    )
    rt = Runtime(
        dataset=UnreadDataset(parquet_paths=[], model_id="m", benchmarks=["b"]),
        library=FunctionLibrary(functions=[fn]),
        run=RunConfig(run_id="screen-test", model_id="m", benchmarks=["b"], parquet_paths=[]),
        config=Config(harness="claude", judge_workers=1, behavior_strictness="causal"),
    )
    set_runtime(rt)
    return rt, fn


def test_candidate_without_prerequisites_is_not_screened(screen_runtime):
    rt, fn = screen_runtime
    assert screen.ensure_candidate_screen(rt, fn) == NOT_REQUIRED
    assert screen.ensure_candidate_screen(rt, fn, check_evidence_prerequisites=False) == (
        NOT_REQUIRED)


def test_blocked_preflight_is_returned_unchanged_and_skippable(screen_runtime, monkeypatch):
    rt, fn = screen_runtime
    blocked = {"ok": True, "name": fn.name, "status": "evidence_prerequisite_pending",
               "evaluation_allowed": False, "source_checks": [{"item_id": "case-0"}]}
    calls = []

    def preflight(runtime, function):
        calls.append(function.name)
        return blocked

    monkeypatch.setattr("autofa.session.evidence_prerequisites.evidence_preflight", preflight)
    assert screen.ensure_candidate_screen(rt, fn) is blocked
    assert screen.ensure_candidate_screen(rt, fn, check_evidence_prerequisites=False) == (
        NOT_REQUIRED)
    assert calls == [fn.name]


def test_retained_prerequisites_are_reported_with_the_allowed_result(screen_runtime, monkeypatch):
    rt, fn = screen_runtime
    checks = [{"item_id": "case-0", "field": "input", "availability": "retained"}]
    monkeypatch.setattr(
        "autofa.session.evidence_prerequisites.evidence_preflight",
        lambda runtime, function: {"ok": True, "status": "declared_sources_retained",
                                   "evaluation_allowed": True, "source_checks": checks},
    )
    assert screen.ensure_candidate_screen(rt, fn) == {**NOT_REQUIRED, "source_checks": checks}


def test_screen_function_tool_checks_only_named_candidates(screen_runtime):
    assert set(server._tool_input_schema(screen.screen_function)["properties"]) == {"name"}
    assert screen.screen_function("marker") == NOT_REQUIRED
    missing = screen.screen_function("missing")
    assert not missing["ok"] and "unknown function" in missing["error"]
