# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-mode ledger: persist modes across detector drops."""

from __future__ import annotations

from pathlib import Path

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_eval, tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.mcp.server import TOOL_CALLABLES
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.session.modes_ledger import get_mode, summarize_modes, upsert_mode

SRC = """def shipped_guess(item, llm_client=None):
    return "guess" in (item.get("response_text") or "").lower()
"""
DESC = "Cannot derive the answer so it ships an admitted-wrong placeholder"
MECH = (
    "After failing to solve the core subproblem the model admits it cannot find "
    "the answer then writes a best-guess value to the deliverable and stops. "
    "The grader checks the true value which the model never derived."
)
SUBMIT = dict(name="shipped_guess", source=SRC, description=DESC, mechanism=MECH,
              auto_evaluate=False)


def _runtime(tmp_path: Path, monkeypatch, *, functions: list[FeatureFunction] | None = None):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    run = RunConfig(run_id="ledger-test", parquet_paths=[], model_id="m", benchmarks=["b"])
    lib = FunctionLibrary(model_id="m", benchmark="b", functions=list(functions or []))
    ds = Dataset(parquet_paths=[], model_id="m", benchmarks=["b"])
    rt = Runtime(dataset=ds, library=lib, run=run, config=Config(behavior_strictness="causal"),
                 scores_history=[])
    set_runtime(rt)
    return rt


def _current_judgment(purity: float, **counts) -> dict:
    return {"behavior_purity": purity, "judge_backend": "claude_agent_sdk",
            "judge_contract_version": JUDGE_CONTRACT_VERSION,
            "judge_policy_hash": judge_policy_hash(),
            "claim_hash": claim_hash(source=SRC, description=DESC, mechanism=MECH), **counts}


def test_submit_remove_restore_roundtrip(tmp_path: Path, monkeypatch):
    for name in ("list_failure_modes", "get_failure_mode", "restore_failure_mode"):
        assert name in TOOL_CALLABLES
    rt = _runtime(tmp_path, monkeypatch)
    out = tools_library.submit_function(
        **SUBMIT, function_mechanism="Regex over response_text for guess language.",
    )
    assert out["ok"]
    assert out["failure_modes_summary"]["counts"]["active"] == 1

    refused = tools_library.remove_function(name="shipped_guess", reason="")
    assert not refused["ok"] and "reason" in refused["error"].lower()
    removed = tools_library.remove_function(
        name="shipped_guess", reason="behavior_purity=0.1; mode still real", status="open",
    )
    assert removed["ok"] and removed["status"] == "open"
    assert rt.library.get("shipped_guess") is None
    assert "source" in removed["removed_snapshot"]

    listed = tools_library.list_failure_modes()
    assert [(m["name"], m["status"]) for m in listed["modes"]] == [("shipped_guess", "open")]
    assert tools_library.list_library_functions()["functions"] == []
    assert tools_library.get_failure_mode("shipped_guess")["mode"]["detector"]["source"] == SRC

    assert tools_library.restore_failure_mode("shipped_guess")["ok"]
    assert rt.library.get("shipped_guess") is not None
    assert summarize_modes(rt.run.run_dir())["counts"]["active"] == 1


def test_clear_library_marks_rejected(tmp_path: Path, monkeypatch):
    fn = FeatureFunction(name="x", description=DESC, source=SRC, mechanism=MECH)
    rt = _runtime(tmp_path, monkeypatch, functions=[fn])
    assert tools_library.clear_library(confirm=True)["ok"]
    assert rt.library.functions == []
    assert get_mode(rt.run.run_dir(), "x")["status"] == "rejected"


@pytest.mark.parametrize("current_contract", [False, True])
def test_mode_list_labels_historical_judgments_without_rewriting_snapshot(
    tmp_path, monkeypatch, current_contract,
):
    rt = _runtime(tmp_path, monkeypatch)
    fn = FeatureFunction(name="shipped_guess", source=SRC, description=DESC, mechanism=MECH)
    counts = dict(behavior_specificity=0.9, behavior_n_judged=4, behavior_n_yes=4,
                  behavior_n_completed=4)
    if current_contract:
        fn.scores = {**_current_judgment(1.0, **counts), "judge_policy_hash": "previous-policy"}
    else:
        fn.scores = {"behavior_purity": 1.0, "judge_backend": "claude_agent_sdk", **counts}
    upsert_mode(
        rt.run.run_dir(), name=fn.name, description=DESC, mechanism=MECH,
        status="open", reason="saved review", detector=fn.to_dict(),
    )
    before = get_mode(rt.run.run_dir(), fn.name)
    scores = tools_library.list_failure_modes()["modes"][0]["scores"]
    assert scores["behavior_validation"]["status"] == ("stale" if current_contract else "legacy")
    assert scores["behavior_purity"] == (None if current_contract else 1.0)
    assert scores["behavior_specificity"] == (None if current_contract else 0.9)
    assert tools_library.get_failure_mode(fn.name)["mode"] == before
    assert get_mode(rt.run.run_dir(), fn.name) == before


# Saved scores belong to the exact detector and claim, never to a reused name.
@pytest.mark.parametrize("changed", ["source", "mechanism", None])
def test_resubmission_cannot_reuse_scores_for_a_changed_detector_or_claim(
    tmp_path, monkeypatch, changed,
):
    rt = _runtime(tmp_path, monkeypatch)
    args = dict(SUBMIT)
    assert tools_library.submit_function(**args)["ok"]
    scores = {"behavior_purity": 0.9, "recall": 0.4,
              "judge_backend": "claude_agent_sdk", "improvement": {"targets": ["model"]}}
    rt.library.get("shipped_guess").scores = dict(scores)
    if changed:
        args[changed] += "\n# revised" if changed == "source" else " Revised claim."
    assert tools_library.submit_function(**args)["ok"]
    assert rt.library.get("shipped_guess").scores == ({} if changed else scores)


@pytest.mark.parametrize("changed,judge_fails", [
    ("source", False), ("mechanism", True), (None, False),
])
def test_auto_evaluated_claim_edit_refreshes_prior_judgment(
    tmp_path, monkeypatch, changed, judge_fails,
):
    rt = _runtime(tmp_path, monkeypatch)
    args = dict(SUBMIT)
    tools_library.submit_function(**args)
    rt.library.get("shipped_guess").scores = _current_judgment(
        0.9, behavior_n_judged=10, behavior_n_yes=9, behavior_n_no=1, behavior_n_completed=10,
    )
    if changed:
        args[changed] += "\n# revised" if changed == "source" else " Revised claim."
    calls = []

    def evaluate(**kwargs):
        rt.library.get("shipped_guess").scores.update(recall=0.2, ppv_at_pi0=0.8, n_flagged=10)
        return {"ok": True}

    def judge(name, **kwargs):
        calls.append(name)
        if judge_fails:
            raise RuntimeError("judge unavailable")
        current = rt.library.get(name)
        current.scores.update(
            behavior_purity=0.6, judge_backend="claude_agent_sdk", behavior_n_judged=10,
            behavior_n_yes=6, behavior_n_no=4, behavior_n_completed=10,
            judge_contract_version=JUDGE_CONTRACT_VERSION, judge_policy_hash=judge_policy_hash(),
            claim_hash=claim_hash(source=current.source, description=current.description,
                                  mechanism=current.mechanism),
        )
        return {"ok": True}

    monkeypatch.setattr(tools_eval, "evaluate_functions", evaluate)
    monkeypatch.setattr(tools_eval, "judge_function", judge)
    result = tools_library.submit_function(**{**args, "auto_evaluate": True})
    assert result["ok"]
    assert calls == (["shipped_guess"] if changed else [])
    expected = (None if judge_fails else 0.6) if changed else 0.9
    assert result["feedback"]["behavior_purity"] == expected
    assert rt.library.get("shipped_guess").scores.get("behavior_purity") == expected
    if changed:
        assert result["judge_refresh"]["refreshed"] is (not judge_fails)
    if judge_fails:
        assert "judge unavailable" in result["judge_refresh"]["error"]
        assert "get_function_repair" in result["message"]
