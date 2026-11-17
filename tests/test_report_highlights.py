# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional quote extraction belongs to final reports, with durable reuse."""

import asyncio
import json
import threading
import time
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from autofa.config import Config, RunConfig
from autofa.core import localisation
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import run_detector
from autofa.llm.usage import observed_call
from autofa.mcp import tools_eval
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.session import report_highlights
from autofa.session.supervision import lifecycle_request
from autofa.session.usage import usage_scope, usage_summary
from autofa.util.cancellation import _CURRENT, check_cancelled

SOURCE = '''
def guessed(item, llm_client=None):
    return llm_client.classify("Was the answer guessed?", item["response_text"]) == "yes"
'''


class LLM:
    def __init__(self):
        self.scored = 0
        self.extracted = 0

    def classify(self, *args, **kwargs):
        self.scored += 1
        return "yes"

    def localise(self, *args, **kwargs):
        self.extracted += 1
        return json.dumps({"quotes": ["I guessed the answer."]})


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    rows = [{"_item_id": str(i), "response_text": "I guessed the answer."} for i in range(3)]
    dataset = SimpleNamespace(iter_batches=lambda **kwargs: iter([rows]), parquet_paths=[])
    llm = LLM()
    fn = FeatureFunction(name="guessed", description="Guesses the answer", source=SOURCE,
                         kind="llm")
    fn.scores = {"evaluated": True, "precision": .8, "behavior_n_yes": 5}
    for row in rows:
        record = {}
        assert run_detector(SOURCE, row, llm, localisation=record) is True
        fn.failure_localisations[row["_item_id"]] = record
    rt = Runtime(dataset=dataset, library=FunctionLibrary(model_id="m", functions=[fn]),
                 run=RunConfig(run_id="highlights", parquet_paths=[], model_id="m"),
                 config=Config(llm_eval_workers=1), hub_llm=llm)
    set_runtime(rt)
    return rt, fn, rows, llm


def test_scoring_defers_generation_and_final_report_prepares_once(runtime):
    rt, fn, rows, llm = runtime
    assert llm.scored == 3 and llm.extracted == 0
    scores = deepcopy(fn.scores)
    result = report_highlights.prepare_report_highlights(rt, console=Console(file=StringIO()))
    assert result["completed"] == 3 and llm.extracted == 3
    assert fn.scores == scores and llm.scored == 3
    saved = json.loads(rt.working_library_path().read_text())
    records = saved["functions"][0]["failure_localisations"].values()
    assert all(record["passages"] for record in records)
    assert report_highlights.prepare_report_highlights(rt)["status"] == "not_required"
    assert llm.extracted == 3


def test_highlight_worker_usage_is_separate_from_discovery(runtime, monkeypatch):
    rt, _, _, llm = runtime
    original = llm.localise

    def extract(*args, **kwargs):
        observed_call(lambda: {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
                      client=SimpleNamespace(), usage_model="test", api="chat_completions",
                      role="highlighting")
        return original(*args, **kwargs)

    monkeypatch.setattr(llm, "localise", extract)
    with usage_scope(run_dir=rt.run.run_dir(), phase="discovery", round_index=5):
        result = report_highlights.prepare_report_highlights(rt, console=Console(file=StringIO()))
    assert result["completed"] == 3
    summary = usage_summary(rt.run.run_dir())
    assert summary["totals"]["calls"] == 3
    assert summary["by_phase"][0]["phase"] == "highlighting"
    assert summary["by_role"][0]["role"] == "highlighting"
    assert summary["by_round"][0]["round_index"] is None


def test_cancelled_highlighting_resumes_only_unfinished_items(runtime, monkeypatch):
    rt, fn, rows, llm = runtime
    original = llm.localise

    def interrupt(*args, **kwargs):
        if llm.extracted == 1:
            raise asyncio.CancelledError
        return original(*args, **kwargs)

    monkeypatch.setattr(llm, "localise", interrupt)
    with pytest.raises(asyncio.CancelledError):
        report_highlights.prepare_report_highlights(rt)
    receipt = json.loads((rt.run.run_dir() / "report_highlights.json").read_text())
    assert receipt["status"] == "pending" and len(receipt["remaining"]) == 2
    assert len(fn.failure_localisations["0"]["passages"]) == 1
    # Reload the saved library just as Resume does.
    rt.library = FunctionLibrary.load(rt.working_library_path())
    monkeypatch.setattr(llm, "localise", original)
    result = report_highlights.prepare_report_highlights(rt, pending_only=True)
    assert result["completed"] == 2 and llm.extracted == 3


@pytest.mark.parametrize("error,workers", [(ConnectionError, 1), (ValueError, 2)])
def test_backend_failure_preserves_scores_and_stops_remaining_optional_calls(
    runtime, monkeypatch, error, workers,
):
    rt, fn, rows, llm = runtime
    rt.config.llm_eval_workers = workers
    scores = deepcopy(fn.scores)
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(True)
        raise error("Backend unavailable")

    monkeypatch.setattr(llm, "localise", unavailable)
    result = report_highlights.prepare_report_highlights(rt)
    assert not result["ok"] and result["status"] == "pending"
    assert 1 <= len(calls) <= workers and fn.scores == scores
    receipt = json.loads((rt.run.run_dir() / "report_highlights.json").read_text())
    assert receipt["status"] == "pending" and len(receipt["remaining"]) == 3


def test_stale_and_missing_sources_are_explicit_without_generation(runtime):
    rt, fn, rows, llm = runtime
    rows[0]["response_text"] = "Changed original content."
    rows.pop()
    result = report_highlights.prepare_report_highlights(rt)
    assert result["completed"] == 3 and llm.extracted == 1
    assert fn.failure_localisations["0"]["deferred_localisation"]["status"] == "unavailable"
    assert fn.failure_localisations["2"]["deferred_localisation"]["status"] == "unavailable"


def test_current_source_and_positive_label_are_required(runtime):
    rt, fn, rows, llm = runtime
    fn.failure_localisations["0"]["assigned_label"] = "False"
    fn.failure_localisations["1"]["detector_hash"] = "obsolete source"
    assert report_highlights.prepare_report_highlights(rt)["completed"] == 1
    assert llm.extracted == 1


def test_reevaluation_keeps_completed_highlights_from_current_record(runtime):
    rt, fn, rows, llm = runtime
    localisations = {fn.name: {}}
    # Populate genuine detector checkpoints with their initially deferred records.
    tools_eval._score_items_concurrently([(fn, row) for row in rows], workers=1,
                                         llm=llm, localisations=localisations)
    fn.failure_localisations = localisations[fn.name]
    report_highlights.prepare_report_highlights(rt)
    completed = deepcopy(fn.failure_localisations)
    old_calls = llm.scored, llm.extracted
    localisations = {fn.name: {}}
    tools_eval._score_items_concurrently([(fn, row) for row in rows], workers=1,
                                         llm=llm, localisations=localisations)
    assert localisations[fn.name] == completed
    assert (llm.scored, llm.extracted) == old_calls
    changed = deepcopy(rows[0])
    changed["response_text"] += " Later correction."
    pending = {}
    run_detector(SOURCE, changed, llm, localisation=pending)
    reused = report_highlights.reuse_report_highlights(pending, completed["0"], changed, SOURCE)
    assert reused["deferred_localisation"]["status"] == "pending"


def test_final_report_reuses_reviewer_requested_navigation_cache(runtime):
    rt, fn, rows, llm = runtime
    localisation.resolve_deferred_localisation(
        fn.failure_localisations["0"], rows[0], SOURCE, llm, reviewed_source=True,
        cache_dir=rt.run.run_dir() / "localisation_cache",
    )
    assert llm.extracted == 1
    # Review workers only write their shared cache, not the live library.
    assert not fn.failure_localisations["0"]["passages"]
    assert report_highlights.prepare_report_highlights(rt)["completed"] == 3
    assert llm.extracted == 3


def test_repeated_cancellation_still_drains_optional_worker(monkeypatch):
    started, cancelled, release = threading.Event(), threading.Event(), threading.Event()

    def prepare(rt, **kwargs):
        token = _CURRENT.get()
        started.set()
        assert token.event.wait(3)
        cancelled.set()
        assert release.wait(3)
        check_cancelled()

    monkeypatch.setattr(report_highlights, "prepare_report_highlights", prepare)

    async def invoke():
        task = asyncio.create_task(report_highlights.prepare_report_highlights_async(object()))
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        assert await asyncio.to_thread(cancelled.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)

    asyncio.run(invoke())


def test_deadline_preserves_completed_later_items_and_does_not_publish_late_writes(
    runtime, monkeypatch,
):
    rt, fn, rows, llm = runtime
    rt.config.llm_eval_workers = 3
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = localisation.resolve_deferred_localisation

    def resolve(record, item, *args, **kwargs):
        if item["_item_id"] == "0":
            started.set()
            assert release.wait(3)
            try:
                return original(record, item, *args, **kwargs)
            finally:
                finished.set()
        assert started.wait(3)
        return original(record, item, *args, **kwargs)

    monkeypatch.setattr(localisation, "resolve_deferred_localisation", resolve)
    start = time.monotonic()
    try:
        result = report_highlights.prepare_report_highlights(
            rt, request_timeout=.15, overall_timeout=.5)
        assert time.monotonic() - start < 1
        assert result["status"] == "pending" and result["completed"] == 2
        receipt = json.loads((rt.run.run_dir() / "report_highlights.json").read_text())
        assert [row["item_id"] for row in receipt["remaining"]] == ["0"]
        assert fn.failure_localisations["1"]["passages"]
        assert fn.failure_localisations["2"]["passages"]
        cache_before = list((rt.run.run_dir() / "localisation_cache").glob("*.json"))
    finally:
        release.set()
    assert finished.wait(3)
    assert list((rt.run.run_dir() / "localisation_cache").glob("*.json")) == cache_before
    assert not fn.failure_localisations["0"]["passages"]


def test_highlighting_overall_budget_retains_pending_work(runtime, monkeypatch):
    rt, fn, rows, llm = runtime
    release, finished = threading.Event(), threading.Event()
    original = localisation.resolve_deferred_localisation

    def blocked(*args, **kwargs):
        assert release.wait(3)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(localisation, "resolve_deferred_localisation", blocked)
    try:
        result = report_highlights.prepare_report_highlights(
            rt, request_timeout=1, overall_timeout=.1)
        assert result["status"] == "pending" and result["completed"] == 0
    finally:
        release.set()
    assert finished.wait(3)
    assert not llm.extracted


def test_bad_request_preserves_completed_highlights_and_safe_error_details(runtime, monkeypatch):
    import httpx
    from openai import BadRequestError

    rt, fn, rows, llm = runtime
    rt.config.llm_eval_workers = 3
    barrier = threading.Barrier(3)
    original = localisation.resolve_deferred_localisation

    def resolve(record, item, *args, **kwargs):
        barrier.wait(3)
        if item["_item_id"] == "0":
            raise BadRequestError(
                "Bad request", response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
                body={"error": {"code": "invalid_parameter", "param": "max_output_tokens",
                                "message": "Invalid budget; api_key=SECRET bearer PRIVATE; "
                                "'api_key': 'PRIVATEKEY'; Authorization: Basic PRIVATEBASIC; "
                                "https://user:PRIVATEPASS@service.test/path?token=PRIVATEQUERY"}})
        return original(record, item, *args, **kwargs)

    monkeypatch.setattr(localisation, "resolve_deferred_localisation", resolve)
    result = report_highlights.prepare_report_highlights(rt)
    assert result["completed"] == 2 and result["remaining"] == 1
    failure = result["errors"][0]
    assert failure["status_code"] == 400 and failure["code"] == "invalid_parameter"
    assert failure["param"] == "max_output_tokens"
    assert "Invalid budget" in failure["message"]
    assert "SECRET" not in json.dumps(result) and "PRIVATE" not in json.dumps(result)
    monkeypatch.setattr(localisation, "resolve_deferred_localisation", original)
    assert report_highlights.prepare_report_highlights(rt, pending_only=True)["completed"] == 1


# Stopping during optional highlighting must not submit the next model request.
def test_supervisor_stop_during_highlights_prevents_next_request(runtime, monkeypatch):
    rt, _fn, _rows, llm = runtime
    original = llm.localise

    def stop_after_request(*args, **kwargs):
        result = original(*args, **kwargs)
        lifecycle_request(rt.run.run_dir(), "stop")
        return result

    monkeypatch.setattr(llm, "localise", stop_after_request)
    result = report_highlights.prepare_report_highlights(rt)
    assert result == {"ok": False, "status": "pending", "reason": "supervisor_stop"}
    assert llm.extracted == 1
    receipt = json.loads((rt.run.run_dir() / "report_highlights.json").read_text())
    assert receipt["status"] == "pending"
    assert receipt["remaining"]
