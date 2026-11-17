# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library mutations, submissions and imports commit or restore atomically under cancellation."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from autofa.config import Config, RunConfig
from autofa.core.helpers import HelperFunction
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import runtime, tools_library
from autofa.metrics.claim_contract import claim_hash
from autofa.session import family_discovery, repair_workflow, review_evidence
from autofa.session.modes_ledger import get_mode, upsert_mode
from autofa.session.review_evidence import _ledger, record_discovery_exposure, review_context
from autofa.util.cancellation import (
    Cancellation,
    cancellable,
    cancellation_guard,
    cancellation_scope,
    check_cancelled,
)

# A cancellation signal cannot erase a child operation's completed outcome.


@pytest.mark.parametrize("signal,failed", [("tool", False), ("judge_batch", True)])
def test_completed_mutation_keeps_result_or_error_after_late_cancel(tmp_path, signal, failed):
    cancellation = Cancellation()
    batch_cancelled = threading.Event()
    saved = tmp_path / "committed.json"
    result = {"ok": True, "removed": "old-detector"}
    failure = ValueError("Committed operation returned a failure")
    signal_cancel = cancellation.cancel if signal == "tool" else batch_cancelled.set

    async def mutation():
        with cancellation_guard():
            saved.write_text('{"functions": []}')
        signal_cancel()
        if failed:
            raise failure
        return result

    async def scenario():
        with cancellation_scope(cancellation):
            if failed:
                with pytest.raises(ValueError) as caught:
                    await cancellable(mutation(), cancel_event=batch_cancelled)
                assert caught.value is failure
            else:
                assert await cancellable(mutation(), cancel_event=batch_cancelled) is result

    asyncio.run(scenario())
    assert saved.read_text() == '{"functions": []}'


@pytest.mark.parametrize("signal", ["tool", "judge_batch"])
def test_running_child_is_cancelled_and_drained(signal):
    cancellation = Cancellation()
    batch_cancelled = threading.Event()
    signal_cancel = cancellation.cancel if signal == "tool" else batch_cancelled.set
    closed = []

    async def child():
        signal_cancel()
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    async def scenario():
        with cancellation_scope(cancellation), pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancellable(child(), cancel_event=batch_cancelled), 2)
        assert closed == [True]

    asyncio.run(scenario())


def test_pre_cancelled_tool_does_not_start_child():
    cancellation = Cancellation()
    cancellation.cancel()
    started = []

    async def child():
        started.append(True)

    async def scenario():
        with cancellation_scope(cancellation), pytest.raises(asyncio.CancelledError):
            await cancellable(child())
        assert not started

    asyncio.run(scenario())


# Library mutators commit or restore their checkpoint before another save.


@pytest.fixture
def empty_rt(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    result = runtime.Runtime(
        dataset=Dataset(parquet_paths=[]), library=FunctionLibrary(),
        run=RunConfig(run_id="mutations", model_id="m", parquet_paths=[]), config=Config(),
    )
    monkeypatch.setattr(runtime, "_RUNTIME", result)
    return result


@pytest.fixture
def rt(empty_rt):
    for name, value in (("target", False), ("peer", True)):
        empty_rt.library.upsert(FeatureFunction(
            name=name, description=name, source=f"def {name}(item): return {value}"))
    empty_rt.persist_library()
    saved = FeatureFunction(name="target", description="Saved target",
                            source="def target(item): return True")
    upsert_mode(empty_rt.run.run_dir(), name="target", detector=saved.to_dict(), status="active")
    return empty_rt


# remove, clear and restore share one mutation checkpoint helper.
MUTATIONS = {
    "remove": lambda: tools_library.remove_function("target", reason="Test removal"),
    "clear": lambda: tools_library.clear_library(confirm=True),
}


def test_cancelled_mutation_does_not_touch_library_or_mode(rt):
    before = deepcopy(rt.library)
    previous = rt.working_library_path().read_bytes()
    mode = get_mode(rt.run.run_dir(), "target")
    token = Cancellation()
    token.cancel()

    with cancellation_scope(token), pytest.raises(asyncio.CancelledError):
        MUTATIONS["remove"]()

    assert rt.library == before
    assert rt.working_library_path().read_bytes() == previous
    assert get_mode(rt.run.run_dir(), "target") == mode


# A BaseException proves rollback does not rely on ``except Exception``.
@pytest.mark.parametrize("operation,failure", [
    ("remove", OSError), ("clear", asyncio.CancelledError),
])
def test_failed_mutation_rolls_back_before_concurrent_save(rt, monkeypatch, operation, failure):
    before = deepcopy(rt.library)
    original_functions = list(rt.library.functions)
    mode = get_mode(rt.run.run_dir(), "target")
    saving, release, other_started, other_done = (threading.Event() for _ in range(4))
    save = FunctionLibrary.save
    attempts = []

    def fail_first_save(library, path):
        attempts.append(library.to_dict())
        if len(attempts) == 1:
            saving.set()
            assert release.wait(5)
            raise failure("save interrupted")
        return save(library, path)

    monkeypatch.setattr(FunctionLibrary, "save", fail_first_save)

    def concurrent_save():
        other_started.set()
        rt.persist_library()
        other_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        changing = pool.submit(MUTATIONS[operation])
        assert saving.wait(5)
        other = pool.submit(concurrent_save)
        try:
            assert other_started.wait(5)
            assert not other_done.wait(0.1)
        finally:
            release.set()
        with pytest.raises(failure, match="save interrupted"):
            changing.result(timeout=5)
        other.result(timeout=5)

    assert rt.library == before
    assert all(left is right for left, right in zip(
        rt.library.functions, original_functions, strict=True,
    ))
    assert FunctionLibrary.load(rt.working_library_path()) == before
    assert get_mode(rt.run.run_dir(), "target") == mode


def test_retirement_failure_restores_already_persisted_library(rt, monkeypatch):
    before = deepcopy(rt.library)

    def fail(*args):
        raise OSError("retirement ledger unavailable")

    monkeypatch.setattr(repair_workflow, "record_retirement", fail)
    with pytest.raises(OSError, match="retirement ledger unavailable"):
        MUTATIONS["remove"]()
    assert rt.library == before
    assert FunctionLibrary.load(rt.working_library_path()) == before


def test_library_import_merges_against_locked_current_library(rt, tmp_path, monkeypatch):
    imported = tmp_path / "import.json"
    FunctionLibrary(functions=[FeatureFunction(
        name="imported", description="Imported", source="def imported(item): return True",
    )]).save(imported)
    entered = threading.Event()
    original_guard = tools_library.cancellation_guard

    def guard():
        entered.set()
        return original_guard()

    monkeypatch.setattr(tools_library, "cancellation_guard", guard)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with rt._library_lock:
            importing = pool.submit(tools_library.library_load, str(imported))
            assert entered.wait(5)
            rt.library.upsert(FeatureFunction(
                name="concurrent", description="Concurrent",
                source="def concurrent(item): return True",
            ))
            rt.persist_library()
        assert importing.result(timeout=5)["ok"]
    assert {fn.name for fn in rt.library.functions} == {"target", "peer", "concurrent", "imported"}
    assert FunctionLibrary.load(rt.working_library_path()).get("concurrent") is not None


# Cancelled or failed submissions cannot leak into a later library save.

SOURCE = "def candidate(item): return 'wrong' in item.get('response_text', '')"


@pytest.fixture
def submission_mode():
    """Overridden by direct parametrization: a new, replaced or planned-repair detector."""
    return "repair"


@pytest.fixture
def submission_rt(tmp_path, monkeypatch, submission_mode):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    library = FunctionLibrary(functions=[FeatureFunction(
        name="peer", description="Another detector", source="def peer(item): return False",
    )])
    if submission_mode == "replacement":
        library.upsert(FeatureFunction(
            name="candidate", description="Original detector",
            source="def candidate(item): return False", scores={"retained": True},
        ))
    rt = runtime.Runtime(
        dataset=Dataset(parquet_paths=[]), library=library,
        run=RunConfig(run_id="atomic-submit", parquet_paths=[], model_id="m"), config=Config(),
    )
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    rt.persist_library()
    if submission_mode == "repair":
        _plan_submission(rt)
    return rt


def _plan_submission(rt):
    # Repair an evaluated, still-live baseline.
    before = FeatureFunction(
        name="candidate", description="Changed detector", mechanism="Changed detector",
        source="def candidate(item): return False", scores={"evaluated": True},
    )
    rt.library.upsert(before)
    rt.persist_library()
    feedback = {
        "claim_hash": claim_hash(source=before.source, description=before.description,
                                 mechanism=before.mechanism),
        "allowed_actions": ["tighten_detector"],
        "findings": [{"id": "counterexample", "status": "contradicted",
                      "item_ids": ["item-1"]}],
    }
    repair_workflow.plan_repair(
        rt.run.run_dir(), before.to_dict(), feedback, action="tighten_detector",
        rationale="Exclude the inspected counterexample while preserving the claim.",
        finding_ids=["counterexample"], evidence_ids=["item-1"],
    )


def _assert_committed(rt):
    assert rt.library.get("candidate").source == SOURCE
    assert FunctionLibrary.load(rt.working_library_path()).get("candidate").source == SOURCE
    workflow = repair_workflow.get_repair(rt.run.run_dir(), "candidate")
    assert workflow["state"] == "awaiting_validation"
    assert len(workflow["history"]) == 1
    assert workflow["latest"]["after"]["source"] == SOURCE


def _submit():
    return tools_library.submit_function(
        name="candidate", source=SOURCE, description="Changed detector",
        auto_evaluate=False,
    )


def _cancel_before_commit(patch, token):
    # This status boundary is immediately before the atomic submission guard,
    # before any family attempt, repair or library commit starts.
    patch.setattr(tools_library, "emit_runtime_status", lambda *args, **kwargs: token.cancel())


@pytest.mark.parametrize("submission_mode,failure", [
    ("new", "cancel_at_entry"), ("replacement", "cancel_before_commit"),
    ("replacement", "exposure_ledger"),
])
def test_interrupted_submission_preserves_live_entry_and_later_saves(
    submission_rt, monkeypatch, failure,
):
    before = deepcopy(submission_rt.library)
    previous = submission_rt.working_library_path().read_bytes()
    token = Cancellation()
    expected, match = asyncio.CancelledError, None
    with monkeypatch.context() as patch:
        if failure == "cancel_at_entry":
            token.cancel()
        elif failure == "cancel_before_commit":
            _cancel_before_commit(patch, token)
        else:
            def unavailable_ledger(path, data):
                raise OSError("Exposure ledger unavailable")

            patch.setattr(review_evidence, "atomic_write_json", unavailable_ledger)
            expected, match = OSError, "Exposure ledger unavailable"
        with cancellation_scope(token), pytest.raises(expected, match=match):
            _submit()
    assert submission_rt.library == before
    assert submission_rt.working_library_path().read_bytes() == previous
    submission_rt.persist_library()
    assert FunctionLibrary.load(submission_rt.working_library_path()) == before


def test_cancellation_during_commit_finishes_one_consistent_checkpoint(
    submission_rt, monkeypatch,
):
    token = Cancellation()
    committing, cancel_requested, cancelled, release = (threading.Event() for _ in range(4))
    persist = submission_rt.persist_library

    def paused_persist(**kwargs):
        committing.set()
        assert release.wait(5)
        return persist(**kwargs)

    monkeypatch.setattr(submission_rt, "persist_library", paused_persist)

    def submit():
        with cancellation_scope(token):
            return _submit()

    def cancel():
        cancel_requested.set()
        token.cancel()
        cancelled.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        submitting = pool.submit(submit)
        assert committing.wait(5)
        cancelling = pool.submit(cancel)
        try:
            assert cancel_requested.wait(5)
            assert not cancelled.wait(0.1)
        finally:
            release.set()
        assert submitting.result(timeout=5)["ok"]
        cancelling.result(timeout=5)
    assert cancelled.is_set()
    _assert_committed(submission_rt)


# A new detector is removed again; a replaced one gets its original object back.
@pytest.mark.parametrize("submission_mode", ["new", "replacement"])
def test_failed_commit_restores_entry_before_an_unrelated_save(submission_rt, monkeypatch):
    before = deepcopy(submission_rt.library)
    original = submission_rt.library.get("candidate")
    peer = submission_rt.library.get("peer")
    previous = submission_rt.working_library_path().read_bytes()
    saving, other_started, other_done, release = (threading.Event() for _ in range(4))
    save = FunctionLibrary.save
    attempted = []

    def fail_first_save(library, path):
        attempted.append(library.get("candidate"))
        if len(attempted) == 1:
            saving.set()
            assert release.wait(5)
            raise asyncio.CancelledError("Checkpoint interrupted")
        return save(library, path)

    monkeypatch.setattr(FunctionLibrary, "save", fail_first_save)

    def unrelated_save():
        other_started.set()
        submission_rt.persist_library()
        other_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        submitting = pool.submit(_submit)
        assert saving.wait(5)
        saving_other = pool.submit(unrelated_save)
        try:
            assert other_started.wait(5)
            assert not other_done.wait(0.1)
            assert submission_rt.working_library_path().read_bytes() == previous
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError, match="Checkpoint interrupted"):
            submitting.result(timeout=5)
        saving_other.result(timeout=5)
    assert attempted[0].source == SOURCE
    assert attempted[1] is original
    assert submission_rt.library.get("candidate") is original
    assert submission_rt.library.get("peer") is peer  # Unrelated objects stay live.
    assert submission_rt.library == before
    assert FunctionLibrary.load(submission_rt.working_library_path()) == before


@pytest.mark.parametrize("failure", ["cancel_before_commit", "library_save", "repair_commit"])
def test_failed_repair_submission_keeps_plan_retryable(submission_rt, monkeypatch, failure):
    before = submission_rt.library.get("candidate")
    path = submission_rt.run.run_dir() / "detector_repairs.json"
    ledger_before = path.read_bytes()
    token = Cancellation()
    with monkeypatch.context() as patch:
        expected = OSError
        if failure == "cancel_before_commit":
            _cancel_before_commit(patch, token)
            expected = asyncio.CancelledError
        elif failure == "library_save":
            def fail_save(*args, **kwargs):
                raise OSError("Library save failed")

            patch.setattr(FunctionLibrary, "save", fail_save)
        else:
            replace = repair_workflow.os.replace

            def fail_repair_commit(source, target):
                if str(target) == str(path):
                    raise OSError("Repair ledger commit failed")
                return replace(source, target)

            patch.setattr(repair_workflow.os, "replace", fail_repair_commit)
        with cancellation_scope(token), pytest.raises(expected):
            _submit()
    assert submission_rt.library.get("candidate").source == before.source
    saved = FunctionLibrary.load(submission_rt.working_library_path())
    assert saved.get("candidate").source == before.source
    assert path.read_bytes() == ledger_before
    workflow = repair_workflow.get_repair(submission_rt.run.run_dir(), "candidate")
    assert workflow["state"] == "planned"
    assert workflow["latest"]["candidates"] == {}

    # The same planned edit must still work: the failure consumed no retry, and the
    # submission uses its staged ledger instead of reopening the locked repair ledger.
    def nested_read(*args, **kwargs):
        pytest.fail("Submission reopened its already-locked repair ledger")

    with monkeypatch.context() as patch:
        patch.setattr(repair_workflow, "list_repairs", nested_read)
        result = _submit()
    assert result["ok"], result
    _assert_committed(submission_rt)


def test_cancel_after_family_commit_finishes_library_and_repair_checkpoint(
    submission_rt, monkeypatch,
):
    family_discovery.initialize_family_discovery(submission_rt)
    family_path = submission_rt.run.run_dir() / family_discovery.STATE_FILE
    token = Cancellation()
    guard = family_discovery.family_submission_guard
    attempts_at_cancellation = []

    def cancel_after_family_commit(*args, **kwargs):
        result = guard(*args, **kwargs)
        if kwargs.get("consume_attempt"):
            # This ledger is already durable; cancelling the nested library save
            # here would split one logical submission into partial state.
            attempts_at_cancellation.append(json.loads(family_path.read_text())[
                "mutation_attempts"]["candidate"])
            token.cancel()
        return result

    monkeypatch.setattr(family_discovery, "family_submission_guard", cancel_after_family_commit)
    with cancellation_scope(token):
        # The tool may stop after its commit, but no committed ledger is rolled back
        # in isolation, and the next operation cannot begin with the cancelled token.
        with pytest.raises(asyncio.CancelledError):
            _submit()
        with pytest.raises(asyncio.CancelledError):
            check_cancelled()
        committed_library = submission_rt.working_library_path().read_bytes()
        committed_family = family_path.read_bytes()
        with pytest.raises(asyncio.CancelledError):
            _submit()
        assert submission_rt.working_library_path().read_bytes() == committed_library
        assert family_path.read_bytes() == committed_family
    assert attempts_at_cancellation == [1]
    _assert_committed(submission_rt)


# Manual library imports retain the same exposure history as Continue.


def detector(name, item_id):
    return FeatureFunction(name=name, description="A supported cause",
                           source=f"def {name}(item): return False",
                           scores={"behavior_verdicts": [{"item_id": item_id, "verdict": "yes"}]})


def test_library_load_unions_portable_removed_detector_and_local_exposure_history(
    empty_rt, tmp_path,
):
    empty_rt.library.upsert(detector("shared", "destination-before"))
    empty_rt.persist_library()
    record_discovery_exposure(empty_rt.run.run_dir(), ["destination-discovery"])
    source = tmp_path / "portable"
    source.mkdir()
    seed = FunctionLibrary(functions=[detector("removed", "source-old")])
    seed.remove("removed")
    seed.upsert(detector("shared", "source-current"))
    seed.save(source / "library.json")
    before = (source / "library.json").read_bytes()

    assert tools_library.library_load(str(source / "library.json"))["ok"]

    assert review_context(empty_rt.run.run_dir(), "shared")["reviewed_item_ids"] == [
        "destination-before", "destination-discovery", "source-current", "source-old",
    ]
    assert (source / "library.json").read_bytes() == before
    assert list(source.iterdir()) == [source / "library.json"]
    reloaded = FunctionLibrary.load(empty_rt.working_library_path())
    assert "removed" in reloaded.review_history["detectors"]
    verdict, = empty_rt.library.get("shared").scores["behavior_verdicts"]
    assert verdict["item_id"] == "source-current"


def test_library_load_recovers_adjacent_active_seed_ledger_without_writing_source(
    empty_rt, tmp_path,
):
    source = tmp_path / "active-source"
    source.mkdir()
    path = source / "library.json"
    FunctionLibrary(functions=[detector("candidate", "latest")]).save(path)
    # The active source has progressed since its last library export.
    record_discovery_exposure(source, [
        {"_item_id": "survey-old", "benchmark": "b", "problem_id": "task-old"},
    ])
    files_before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.iterdir()}

    assert tools_library.library_load(str(path))["ok"]

    assert review_context(empty_rt.run.run_dir(), "candidate")["reviewed_item_ids"] == [
        "latest", "survey-old",
    ]
    groups = empty_rt.library.review_history["discovery_task_groups"]
    assert groups["survey-old"] == '["b","task-old"]'
    assert "review_evidence.json" in files_before
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in source.iterdir()} == files_before


def _import_seed(empty_rt, *functions):
    empty_rt.library.upsert(detector("shared", "keep-this-version"))
    empty_rt.persist_library()
    seed = FunctionLibrary(functions=[detector("shared", "replacement-version"), *functions])
    seed.helpers.upsert(HelperFunction(
        name="imported_helper", source="def helper(value): return value", entrypoint="helper",
    ))
    return seed


@pytest.mark.parametrize("conflict,late_update", [
    ("task_identity", True), ("frozen_allocation", False),
])
def test_conflicting_import_uses_latest_disk_history_without_mutating_live_or_saved_state(
    empty_rt, tmp_path, monkeypatch, conflict, late_update,
):
    seed = _import_seed(empty_rt, detector("new_detector", "new-review"))
    if conflict == "task_identity":
        seed.review_history["discovery_item_ids"] = ["same-id"]
        seed.review_history["discovery_task_groups"] = {"same-id": '["b","incoming-task"]'}

        def advance_ledger():
            record_discovery_exposure(empty_rt.run.run_dir(), [
                {"_item_id": "same-id", "benchmark": "b", "problem_id": "latest-task"},
            ])
    else:
        seed.review_history["detectors"]["shared"] = {
            "allocations": {"claim:confirmation": {"item_ids": ["incoming-reservation"]}},
        }

        def advance_ledger():
            with _ledger(empty_rt.run.run_dir(), write=True) as data:
                data["detectors"]["shared"]["allocations"] = {
                    "claim:confirmation": {"item_ids": ["existing-reservation"]},
                }

    path = tmp_path / "conflicting-import.json"
    seed.save(path)
    live = empty_rt.library
    before = deepcopy(live)
    library_before = empty_rt.working_library_path().read_bytes()
    ledger_path = empty_rt.run.run_dir() / "review_evidence.json"
    ledger_before = []
    if late_update:
        persist = empty_rt.persist_library

        def persist_after_fresh_exposure(**kwargs):
            assert empty_rt.library is live and empty_rt.library == before
            advance_ledger()
            ledger_before.append(ledger_path.read_bytes())
            return persist(**kwargs)

        monkeypatch.setattr(empty_rt, "persist_library", persist_after_fresh_exposure)
    else:
        advance_ledger()
        ledger_before.append(ledger_path.read_bytes())

    with pytest.raises(ValueError, match="Conflicting"):
        tools_library.library_load(str(path))

    assert empty_rt.library is live and empty_rt.library == before
    assert empty_rt.working_library_path().read_bytes() == library_before
    assert ledger_path.read_bytes() == ledger_before[0]
    assert not (empty_rt.run.run_dir() / "helpers.json").exists()


@pytest.mark.parametrize("helpers_only", [False, True])
def test_failed_import_save_leaves_live_library_and_review_ledger_unchanged(
    empty_rt, tmp_path, monkeypatch, helpers_only,
):
    seed = _import_seed(empty_rt)
    path = tmp_path / "incoming.json"
    path.write_text(json.dumps(seed.helpers.to_dict() if helpers_only else seed.to_dict()))
    live = empty_rt.library
    before = deepcopy(live)
    library_before = empty_rt.working_library_path().read_bytes()
    ledger_path = empty_rt.run.run_dir() / "review_evidence.json"
    ledger_before = ledger_path.read_bytes()

    def fail_save(candidate, path):
        assert candidate is not live
        assert empty_rt.library is live and live == before
        raise OSError("storage unavailable")

    monkeypatch.setattr(FunctionLibrary, "save", fail_save)
    with pytest.raises(OSError, match="storage unavailable"):
        tools_library.library_load(str(path))
    assert empty_rt.library is live and live == before
    assert empty_rt.working_library_path().read_bytes() == library_before
    assert ledger_path.read_bytes() == ledger_before
