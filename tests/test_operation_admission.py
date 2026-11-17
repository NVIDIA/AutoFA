# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent MCP requests cannot overlap in-flight operations; tools record only real exposure."""

import asyncio
import json
import threading

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import runtime as mcp_runtime
from autofa.mcp import server, tools_eval
from autofa.mcp.runtime import Runtime
from autofa.session import operation_admission
from autofa.session.recovery import execute_operation
from autofa.session.review_evidence import review_context
from autofa.util.cancellation import check_cancelled

from support.repair_review import exposure_runtime


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "autofa"))
    rt = Runtime(
        dataset=Dataset(parquet_paths=[]), library=FunctionLibrary(), config=Config(),
        run=RunConfig(run_id="admission", parquet_paths=[], model_id="synthetic"),
    )
    monkeypatch.setattr(mcp_runtime, "_RUNTIME", rt)
    for name in ("alpha", "beta"):
        rt.library.upsert(FeatureFunction(
            name=name, description="Synthetic failure",
            source='def detect(item):\n    return "failure" in item.get("response_text", "")\n',
        ))
    yield rt
    assert rt.run.run_dir().resolve() not in operation_admission._ACTIVE


async def _call(name, args):
    response = await server.invoke_tool(name, args)
    return json.loads(response["content"][0]["text"])


def _records(rt):
    return [json.loads(path.read_text())
            for path in (rt.run.run_dir() / "operations").glob("*.json")]


class BlockingTool:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def __call__(self, **args):
        self.calls.append(args)
        self.entered.set()
        assert self.release.wait(5), "test did not release active tool"
        check_cancelled()
        return {"ok": True}


# Detector tools share one selector; evaluate_functions resolves a list of names.
@pytest.mark.parametrize("name, args", [
    ("judge_function", {"name": "alpha", "evidence_n": 10}),
    ("evaluate_functions", {"names": ["alpha"]}),
    ("submit_function", {"name": "alpha", "source": "unused"}),
])
def test_same_detector_rejected_before_paid_work_or_recovery_record(rt, monkeypatch, name, args):
    active = BlockingTool()
    rejected_calls = []
    monkeypatch.setitem(server.TOOL_CALLABLES, name,
                        lambda **kwargs: rejected_calls.append(kwargs) or {"ok": True})
    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", active)

    async def scenario():
        running = asyncio.create_task(_call("judge_function", {"name": "alpha", "evidence_n": 10}))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            rejected = await asyncio.wait_for(_call(name, args), 1)
            assert rejected["busy"] and not rejected["ok"]
            assert rejected["conflicting_operations"][0]["tool"] == "judge_function"
            assert rejected["conflicting_operations"][0]["detectors"] == ["alpha"]
            assert len(_records(rt)) == 1
            assert _records(rt)[0]["state"] == "running"
            assert rejected_calls == [] and len(active.calls) == 1
        finally:
            active.release.set()
            assert (await running)["ok"]
        assert _records(rt)[0]["state"] == "complete"

    asyncio.run(scenario())


def test_independent_detectors_run_concurrently_and_library_remains_readable(rt, monkeypatch):
    first, second = BlockingTool(), BlockingTool()
    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", first)
    monkeypatch.setitem(server.TOOL_CALLABLES, "evaluate_functions", second)
    monkeypatch.setitem(server.TOOL_CALLABLES, "list_library_functions",
                        lambda: {"ok": True, "names": [f.name for f in rt.library.functions]})

    async def scenario():
        running = [asyncio.create_task(_call("judge_function", {"name": "alpha"})),
                   asyncio.create_task(_call("evaluate_functions", {"names": ["beta"]}))]
        try:
            assert await asyncio.to_thread(first.entered.wait, 2)
            assert await asyncio.to_thread(second.entered.wait, 2)
            assert (await _call("list_library_functions", {}))["names"] == ["alpha", "beta"]
            assert len(_records(rt)) == 2
        finally:
            first.release.set()
            second.release.set()
            assert all(row["ok"] for row in await asyncio.gather(*running))

    asyncio.run(scenario())


# Library-wide tools share one scope; unscoped evaluation and family records join it.
@pytest.mark.parametrize("name, args", [
    ("evaluate_functions", {}),
    ("library_save", {}),
    ("family_discovery", {"action": "record_family", "family_name": "new"}),
])
@pytest.mark.parametrize("global_first", [False, True])
def test_library_operations_conflict_in_both_directions(rt, monkeypatch, name, args, global_first):
    active = BlockingTool()
    first_name, first_args = (name, args) if global_first else ("judge_function", {"name": "alpha"})
    next_name, next_args = (("submit_function", {"name": "new-detector"})
                            if global_first else (name, args))
    monkeypatch.setitem(server.TOOL_CALLABLES, first_name, active)
    monkeypatch.setitem(server.TOOL_CALLABLES, next_name,
                        lambda **kwargs: pytest.fail("conflicting tool must not start"))

    async def scenario():
        running = asyncio.create_task(_call(first_name, first_args))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            rejected = await asyncio.wait_for(_call(next_name, next_args), 1)
            assert rejected["busy"]
        finally:
            active.release.set()
            assert (await running)["ok"]

    asyncio.run(scenario())


@pytest.mark.parametrize("tool, args", [
    ("submit_function", {"name": "child", "depends_on": [" alpha "]}),
    ("evaluate_functions", {"names": ["beta"], "within": " alpha "}),
    ("smoke_test_feature", {"name": "alpha", "item_ids": ["case"]}),
])
def test_dependencies_are_shared_reads_that_block_replacement(rt, monkeypatch, tool, args):
    active = BlockingTool()
    monkeypatch.setitem(server.TOOL_CALLABLES, tool, active)
    monkeypatch.setitem(server.TOOL_CALLABLES, "remove_function",
                        lambda **kwargs: pytest.fail("cannot replace a dependency in use"))
    sibling_calls = []
    # A second read of the same dependency is compatible, unlike replacing it.
    sibling_tool = "smoke_test_feature" if tool != "smoke_test_feature" else "submit_function"
    monkeypatch.setitem(server.TOOL_CALLABLES, sibling_tool,
                        lambda **kwargs: sibling_calls.append(kwargs) or {"ok": True})

    async def scenario():
        running = asyncio.create_task(_call(tool, args))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            assert (await _call("remove_function", {"name": "alpha"}))["busy"]
            assert (await _call(sibling_tool, {"name": "sibling", "depends_on": ["alpha"]}))["ok"]
            assert len(sibling_calls) == 1
        finally:
            active.release.set()
            assert (await running)["ok"]

    asyncio.run(scenario())


@pytest.mark.parametrize("tool, args", [
    ("evaluate_functions", {"names": "alpha"}),
    ("submit_function", {"name": "child", "depends_on": [17]}),
    ("remove_function", {"name": {"name": "alpha"}}),
])
def test_malformed_selectors_cannot_bypass_admission(rt, monkeypatch, tool, args):
    active = BlockingTool()
    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", active)
    monkeypatch.setitem(server.TOOL_CALLABLES, tool,
                        lambda **kwargs: pytest.fail("unresolved scope cannot run concurrently"))

    async def scenario():
        running = asyncio.create_task(_call("judge_function", {"name": "alpha"}))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            assert (await _call(tool, args))["busy"]
            assert len(_records(rt)) == 1
        finally:
            active.release.set()
            assert (await running)["ok"]

    asyncio.run(scenario())


def test_multi_target_rejection_leaves_no_partial_reservation(rt, monkeypatch):
    active = BlockingTool()
    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", active)
    monkeypatch.setitem(server.TOOL_CALLABLES, "evaluate_functions", lambda **kwargs: {"ok": True})

    async def scenario():
        running = asyncio.create_task(_call("judge_function", {"name": "beta"}))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            assert (await _call("evaluate_functions", {"names": ["alpha", "beta"]}))["busy"]
            assert (await _call("evaluate_functions", {"names": ["alpha"]}))["ok"]
        finally:
            active.release.set()
            assert (await running)["ok"]

    asyncio.run(scenario())


@pytest.mark.parametrize("args", [
    {"names": ["child"]},
    {"name": "new-child", "repair_plan_id": "split-plan"},
])
def test_split_children_and_owner_share_admission(rt, monkeypatch, args):
    directory = rt.run.run_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "detector_repairs.json").write_text(json.dumps({
        "schema_version": "detector-repair-v1",
        "repairs": {"alpha": {"name": "alpha", "history": [{
            "plan_id": "split-plan", "phase": "awaiting_validation", "action": "split_detector",
            "candidates": {"child": {}, "sibling": {}},
        }]}},
    }))
    tool = "evaluate_functions" if "names" in args else "submit_function"
    active = BlockingTool()
    monkeypatch.setitem(server.TOOL_CALLABLES, tool, active)
    monkeypatch.setitem(server.TOOL_CALLABLES, "defer_function_repair",
                        lambda **kwargs: pytest.fail("cannot change the active split"))
    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", lambda **kwargs: {"ok": True})

    async def scenario():
        running = asyncio.create_task(_call(tool, args))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            assert (await _call("defer_function_repair", {"name": "alpha"}))["busy"]
            assert (await _call("judge_function", {"name": "sibling"}))["busy"]
            assert (await _call("judge_function", {"name": "beta"}))["ok"]
        finally:
            active.release.set()
            assert (await running)["ok"]

    asyncio.run(scenario())


def test_real_submit_keeps_reservation_through_nested_evaluation_and_judging(rt, monkeypatch):
    active = BlockingTool()
    calls = []

    def evaluate(**args):
        calls.append("evaluate")
        rt.library.get("new").scores = {"evaluated": True, "n_flagged": 1, "n_items": 2}
        return {"ok": True}

    def judge(**args):
        calls.append("judge")
        return active(**args)

    # Keep the real submit workflow; only replace paid evaluation/review stages.
    # Re-entering admission explicitly also covers future internal dispatch reuse.
    monkeypatch.setattr(tools_eval, "evaluate_functions",
                        lambda **args: execute_operation(evaluate, "evaluate_functions", args))
    monkeypatch.setattr(tools_eval, "judge_function", lambda name, **args:
                        execute_operation(judge, "judge_function", {"name": name, **args}))
    monkeypatch.setitem(server.TOOL_CALLABLES, "remove_function",
                        lambda **kwargs: pytest.fail("cannot drop a submitted detector mid-review"))

    async def scenario():
        running = asyncio.create_task(_call("submit_function", {
            "name": "new", "source": rt.library.get("alpha").source,
            "description": "Synthetic failure", "mechanism": "Synthetic cause and effect",
        }))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            assert (await _call("remove_function", {"name": "new", "reason": "test"}))["busy"]
            assert calls == ["evaluate", "judge"]
        finally:
            active.release.set()
            result = await running
        assert result["ok"] and result["evaluated"], result
        assert all(record["state"] == "complete" for record in _records(rt))

    asyncio.run(scenario())


def test_cancellation_keeps_admission_until_worker_drains_then_releases(rt, monkeypatch):
    active = BlockingTool()
    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", active)
    monkeypatch.setitem(server.TOOL_CALLABLES, "evaluate_functions", lambda **args: {"ok": True})

    async def scenario():
        running = asyncio.create_task(_call("judge_function", {"name": "alpha"}))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            running.cancel()
            # Let invoke_tool signal cancellation to its still-blocked worker.
            await asyncio.sleep(0)
            assert (await _call("evaluate_functions", {"names": ["alpha"]}))["busy"]
        finally:
            active.release.set()
            with pytest.raises(asyncio.CancelledError):
                await running
        assert (await _call("evaluate_functions", {"names": ["alpha"]}))["ok"]
        assert sorted(row["state"] for row in _records(rt)) == ["cancelled", "complete"]

    asyncio.run(scenario())


def test_error_releases_admission_for_recovery(rt, monkeypatch):
    def fail(**args):
        raise ValueError("synthetic interruption")

    monkeypatch.setitem(server.TOOL_CALLABLES, "judge_function", fail)
    monkeypatch.setitem(server.TOOL_CALLABLES, "evaluate_functions", lambda **args: {"ok": True})

    async def scenario():
        failed = await _call("judge_function", {"name": "alpha"})
        assert not failed["ok"] and "synthetic interruption" in failed["error"]
        assert (await _call("evaluate_functions", {"names": ["alpha"]}))["ok"]
        assert sorted(row["state"] for row in _records(rt)) == ["complete", "interrupted"]

    asyncio.run(scenario())


# Classifier profile calls share detector admission. Tool exposure counts only evidence that a
# call returned or a saved decision cited, never a rejected request's unread controls.


@pytest.mark.parametrize("action", ["compare", "adopt"])
def test_profile_mutations_cannot_overlap_scoring(rt, monkeypatch, action):
    active = BlockingTool()
    calls = []
    monkeypatch.setitem(server.TOOL_CALLABLES, "evaluate_functions", active)
    monkeypatch.setitem(server.TOOL_CALLABLES, "classifier_profiles",
                        lambda **kwargs: calls.append(kwargs) or {"ok": True})

    async def scenario():
        running = asyncio.create_task(_call("evaluate_functions", {"names": ["alpha"]}))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            blocked = await _call("classifier_profiles", {"action": action, "name": "alpha"})
            assert blocked["busy"] and not blocked["ok"]
            assert not calls
            assert len(_records(rt)) == 1
        finally:
            active.release.set()
            await running

    asyncio.run(scenario())


def test_profile_status_remains_read_only_during_comparison(rt, monkeypatch):
    active = BlockingTool()

    def profile(**kwargs):
        return {"ok": True, "status": "configured"} if kwargs["action"] == "status" else active()

    monkeypatch.setitem(server.TOOL_CALLABLES, "classifier_profiles", profile)

    async def scenario():
        running = asyncio.create_task(_call("classifier_profiles", {"action": "compare"}))
        try:
            assert await asyncio.to_thread(active.entered.wait, 2)
            status = await _call("classifier_profiles", {"action": "status"})
            assert status["ok"]
            assert len(_records(rt)) == 1
            blocked = await _call("classifier_profiles", {"action": "compare"})
            assert blocked["busy"]
        finally:
            active.release.set()
            await running

    asyncio.run(scenario())


@pytest.mark.parametrize("ok", [True, False])
def test_paired_and_partial_classifier_results_preserve_all_exposure(tmp_path, monkeypatch, ok):
    rt = exposure_runtime(tmp_path, monkeypatch)
    server._record_tool_exposure("classifier_profiles", {
        "action": "compare", "item_ids": ["opaque-0", "opaque-2", "opaque-4", "opaque-6"],
    }, {
        "ok": ok,
        "baseline": {"results": [{"item_id": "opaque-0", "status": "value", "value": True},
                                 {"item_id": "opaque-2", "status": "error", "error": "failed"}]},
        "fallback": {"results": [{"item_id": "opaque-4", "status": "abstained"}]},
    })
    assert review_context(rt.run.run_dir(), "f")["reviewed_item_ids"] == [
        "opaque-0", "opaque-2", "opaque-4",
    ]


def test_rejected_classifier_request_does_not_expose_unread_controls(tmp_path, monkeypatch):
    rt = exposure_runtime(tmp_path, monkeypatch)
    server._record_tool_exposure("classifier_profiles", {
        "action": "compare", "item_ids": ["opaque-0", "opaque-2"],
    }, {"ok": False, "error": "Invalid configured profile"})
    assert review_context(rt.run.run_dir(), "f")["reviewed_item_ids"] == []


@pytest.mark.parametrize("saved", [False, True])
def test_only_saved_pending_family_decisions_record_checked_exposure(tmp_path, monkeypatch, saved):
    rt = exposure_runtime(tmp_path, monkeypatch)
    server._record_tool_exposure("family_discovery", {
        "action": "no_coherent_family",
        "evidence": [{"item_id": "opaque-0", "quotes": [{"field": "response_text",
                                                            "quote": "checked text"}]}],
    }, {"ok": False, "decision_recorded": saved, "error": "Unresolved or invalid"})
    assert review_context(rt.run.run_dir(), "f")["reviewed_item_ids"] == (
        ["opaque-0"] if saved else [])
