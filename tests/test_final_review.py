# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final report review uses saved matches and never silently invents adherence."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.mcp import tools_eval
from autofa.metrics import claude_judge, final_review
from autofa.metrics.judge_checkpoints import JudgeCheckpoints
from autofa.metrics.verdicts import resolved_behavior_purity, verdict_for_scores

from support.evaluation import (
    _llm_flag,
    _runtime,
    forbid,
    judge_rows,
    record_reviews,
    reply,
    runtime,
)
from support.judge import _reply, _review, local_reply, local_review


@pytest.mark.parametrize("change", ["none", "operation_id", "source_content", "identity"])
def test_round_preserves_only_identical_native_full_review(
    tmp_path, monkeypatch, stub_claim_compiler, change,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(claude_judge, "_run_review", lambda **kw: reply(kw["review"], "supported"))
    final, _ = final_review._finalize_one(rt, fn)
    assert final["status"] == "complete"
    original = deepcopy(final)
    if change in {"operation_id", "identity"}:
        fn.scores["final_review"][change] = "obsolete"
    if change == "source_content":
        item = next(i for i in rt.dataset._fake_items if i["_item_id"] == final["item_ids"][0])
        item["response_text"] = "BAD changed evidence"
        monkeypatch.setattr(claude_judge, "_run_review", lambda **kw: reply(kw["review"]))
    else:
        monkeypatch.setattr(claude_judge, "_run_review", forbid)
    checkpoint, _ = final_review._finalize_one(rt, fn, scope="round_checkpoint")
    assert checkpoint["scope"] == "round_checkpoint"
    if change == "none":
        assert checkpoint["status"] == "complete"
        assert fn.scores["final_review"] == original
    else:
        assert "final_review" not in fn.scores


def _interrupt_second_review(monkeypatch, calls, failed=()):
    """Interrupt the second review session once; reject every other reviewed case."""

    def run(**kwargs):
        item_id = kwargs["review"].snapshot["item_id"]
        calls.append(item_id)
        if len(calls) == 2:
            raise KeyboardInterrupt
        if item_id in failed:
            raise ValueError("invalid returned item identifier")
        return reply(kwargs["review"])

    monkeypatch.setattr(claude_judge, "_run_review", run)
    return run


def _capture_operations(monkeypatch, operations):
    """Record each judge operation's items, recovery identity and resumed policy."""
    judge = claude_judge.judge_behavior_with_claude

    def capture(**kwargs):
        operations.append(([item["_item_id"] for item in kwargs["items"]],
                           kwargs["_recovery_id"], kwargs.get("_resume_policy_hash")))
        return judge(**kwargs)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", capture)


def _small_target(patch):
    patch.setattr(tools_eval, "judge_evidence_sample_size", lambda n: min(n, 3))


def _raise(error):
    def fail(*args, **kwargs):
        raise error

    return fail


def test_final_review_scores_unjudged_matches_ignores_discovery_gates_and_reuses(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    fn.scores["judge_prefilter"] = {"status": "skip", "reason": "Discovery efficiency filter"}
    calls = record_reviews(monkeypatch)
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    monkeypatch.setattr(tools_eval, "_judge_function", forbid)
    monkeypatch.setattr("autofa.session.candidate_screen.ensure_candidate_screen", forbid)
    out = final_review.final_review_functions()
    assert out["ok"] and out["reviewed"] == [fn.name]
    assert len(calls) == 3  # Ten percent of twelve matches, with the three-case floor.
    assert fn.scores["behavior_purity"] == 0  # Actual rejections, not a fallback zero.
    assert fn.scores["behavior_n_no"] == 3
    assert fn.scores["behavior_sample"]["phase"] == "development"
    assert fn.scores["final_review"]["status"] == "complete"
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert final_review.final_review_functions()["ok"]


@pytest.mark.parametrize("scope,receipt", [
    ("full_corpus", "final_review"), ("round_checkpoint", "round_review"),
])
def test_final_review_reuses_selected_discovery_reviews_without_model_calls_or_fetches(
    tmp_path, monkeypatch, stub_claim_compiler, scope, receipt,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    items, _ = judge_rows(rt, fn, monkeypatch, representative=True)
    previous = deepcopy(fn.scores["behavior_verdicts"])
    expected = {item["_item_id"] for item in items}
    original = tools_eval._fetch_items_by_id

    def fetch(dataset, ids, **kwargs):
        assert set(ids) == expected  # No unneeded expanded cohort is fetched.
        return original(dataset, ids, **kwargs)

    monkeypatch.setattr(tools_eval, "_fetch_items_by_id", fetch)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    # Low precision would defer new round reviews, but never discards selected ones.
    fn.scores["ppv_at_pi0"] = .2
    out = final_review.final_review_functions(scope=scope)
    assert out["ok"] and out["reviewed"] == [] and not out.get("deferred")
    assert fn.scores[receipt]["status"] == "complete"
    assert fn.scores["behavior_verdicts"] == previous
    assert fn.scores["behavior_n_judged"] == 3


def test_final_review_does_not_retain_rows_under_a_stale_compiler_contract(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    from autofa.metrics.claim_compiler import compiler_policy_hash
    from autofa.metrics.review_assessments import reusable_contract
    from autofa.metrics.verdicts import current_behavior_scores, stale_judge_policy

    rt, fn = runtime(tmp_path, monkeypatch)
    judge_rows(rt, fn, monkeypatch, representative=True)
    assert not stale_judge_policy(fn.scores)
    # A pre-GA approved contract keeps the current judge policy hash on its scores.
    fn.scores["claim_contract"]["compilation"]["policy_hash"] = "0" * 64
    assert reusable_contract(fn.scores) is None
    assert stale_judge_policy(fn.scores)
    assert current_behavior_scores(
        fn.scores, source=fn.review_source, description=fn.description,
        mechanism=fn.mechanism,
    )["behavior_judgment_stale"]
    calls = record_reviews(monkeypatch)
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    out = final_review.final_review_functions()
    assert out["ok"] and out["reviewed"] == [fn.name]
    # Every saved row is re-reviewed under the fresh contract; none is merged in.
    assert len(calls) == fn.scores["behavior_n_judged"] == 3
    assert fn.scores["behavior_n_no"] == 3
    assert fn.scores["claim_contract"]["compilation"]["policy_hash"] == compiler_policy_hash()
    assert not stale_judge_policy(fn.scores)


def test_final_review_expands_discovery_review_to_ten_percent_of_full_matches(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    judge_rows(rt, fn, monkeypatch, representative=True)
    previous = deepcopy(fn.scores["behavior_verdicts"])
    calls = record_reviews(monkeypatch)
    monkeypatch.setattr(tools_eval, "run_detector", forbid)
    out = final_review.final_review_functions()
    assert out["ok"] and out["reviewed"] == [fn.name]
    assert fn.scores["n_flagged"] == 60
    assert fn.scores["behavior_n_judged"] == 6
    assert fn.scores["behavior_n_no"] == 6
    assert len(calls) == 3
    retained = {row["item_id"]: row for row in fn.scores["behavior_verdicts"]}
    assert all(retained[row["item_id"]] == row for row in previous)
    assert not set(calls) & {row["item_id"] for row in previous}
    assert fn.scores["final_review"]["required_n"] == 6
    assert fn.scores["final_review"]["matched_n"] == 60
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert final_review.final_review_functions()["ok"]


@pytest.mark.parametrize("settled_error", [False, True], ids=["clean", "settled-error"])
def test_final_review_recovers_frozen_operation_before_expanding_to_larger_target(
    tmp_path, monkeypatch, stub_claim_compiler, settled_error,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    operations, calls, failed = [], [], set()
    _capture_operations(monkeypatch, operations)
    _interrupt_second_review(monkeypatch, calls, failed)
    with monkeypatch.context() as small:
        _small_target(small)
        with pytest.raises(KeyboardInterrupt):
            final_review.final_review_functions()
        frozen = deepcopy(fn.scores["final_review"])
        assert frozen["status"] == "running" and len(frozen["item_ids"]) == 3
        # A source outage keeps the interrupted cohort instead of reselecting it.
        with monkeypatch.context() as offline:
            offline.setattr(tools_eval, "_fetch_items_by_id",
                            _raise(OSError("source temporarily offline")))
            assert not final_review.final_review_functions()["ok"]
    saved = fn.scores["final_review"]
    assert saved["status"] == "running" and saved["evidence_unavailable"]
    for key in ("identity", "operation_id", "item_ids", "policy_hash"):
        assert saved[key] == frozen[key]
    if settled_error:
        failed.add(frozen["item_ids"][-1])
    assert final_review.final_review_functions()["ok"] is not settled_error
    # Same items and checkpoint/recovery identity, then a disjoint expansion.
    assert operations[0][:2] == operations[1][:2]
    assert len(operations[2][0]) == 3
    assert not set(operations[2][0]) & set(frozen["item_ids"])
    assert len(calls) == 7  # Six completed reviews and the interrupted attempt.
    assert calls.count(calls[0]) == calls.count(frozen["item_ids"][-1]) == 1
    assert fn.scores["behavior_n_judged"] == 6
    assert fn.scores["behavior_n_errors"] == int(settled_error)
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert final_review.final_review_functions()["ok"] is not settled_error


def test_recovery_preserves_judgments_added_outside_its_frozen_cohort(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    calls = []
    run = _interrupt_second_review(monkeypatch, calls)
    with pytest.raises(KeyboardInterrupt):
        final_review.final_review_functions()
    frozen = deepcopy(fn.scores["final_review"])
    full_dataset = rt.dataset._fake_items
    rt.dataset._fake_items = [item for item in full_dataset
                              if item["_item_id"] not in frozen["item_ids"]]
    # Simulate a separate completed review between interruption and finalization.
    judge_rows(rt, fn, monkeypatch, count=1, runner=run, label="True")
    added = deepcopy(fn.scores["behavior_verdicts"][0])
    rt.dataset._fake_items = full_dataset
    # Recovery/imported state may retain the earlier frozen operation as well.
    fn.scores["final_review"] = frozen
    assert final_review.final_review_functions()["ok"]
    saved = {row["item_id"]: row for row in fn.scores["behavior_verdicts"]}
    assert added["item_id"] not in saved
    assert fn.scores["diagnostic_review"]["verdicts"] == [added]
    assert set(frozen["item_ids"]) == set(saved)
    assert fn.scores["behavior_n_judged"] == 3
    assert calls.count(added["item_id"]) == 1
    assert calls.count(calls[0]) == 1


def test_obsolete_policy_final_receipt_requires_fresh_reviews_and_preserves_history(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    old_policy = "26ff5754dbbdba019fbb069e07abb5b9228680154fd52b30e69b5c79d3486dec"
    new_policy = claude_judge.judge_policy_hash()
    calls, operations = [], []
    _capture_operations(monkeypatch, operations)
    _interrupt_second_review(monkeypatch, calls)
    # Recreate a persisted interrupted operation under the superseded policy.
    # The mock assessments are valid in both formats; policy identity still
    # prevents the old completed stage from bypassing the new scope validation.
    with monkeypatch.context() as old:
        old.setattr(claude_judge, "judge_policy_hash", lambda *args, **kwargs: old_policy)
        _small_target(old)
        with pytest.raises(KeyboardInterrupt):
            result = final_review.final_review_functions()
            pytest.fail(f"Review did not reach the planned interruption: {result}")
    frozen = deepcopy(fn.scores["final_review"])
    assert frozen["policy_hash"] == old_policy
    archived = next((rt.run.run_dir() / "behavior_judge").glob("*/0/review-verdict.json"))
    original = archived.read_bytes()
    assert not claude_judge.compatible_judge_policy(old_policy)
    assert final_review.final_review_functions()["ok"]
    assert operations[1][2] is None
    assert operations[1][1] != frozen["operation_id"]
    assert len(operations[1][0]) == 6
    assert len(calls) == 8  # Six fresh reviews, one historical review and its interruption.
    assert archived.read_bytes() == original
    assert fn.scores["judge_policy_hash"] == new_policy
    assert fn.scores["behavior_n_judged"] == 6
    assert fn.scores["behavior_n_no"] == 6
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert final_review.final_review_functions()["ok"]


def test_final_review_retries_errors_once_and_keeps_counterexamples(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)

    def broken(**kw):
        if kw["review"].snapshot["item_id"] == "id-1":
            raise ValueError("invalid returned item identifier")
        return reply(kw["review"])

    _, before = judge_rows(rt, fn, monkeypatch, runner=broken, representative=True)
    calls = []

    def again(**kw):
        calls.append(kw["review"].snapshot["item_id"])
        return broken(**kw)

    monkeypatch.setattr(claude_judge, "_run_review", again)
    out = final_review.final_review_functions()
    assert not out["ok"] and out["unresolved"][0]["name"] == fn.name
    assert len(calls) == 1  # Retain two rejections and retry only the failed case.
    assert fn.scores["behavior_n_judged"] == 3
    assert fn.scores["behavior_n_errors"] == 1
    assert fn.scores["behavior_n_no"] == 2
    saved = {row["item_id"]: row for row in fn.scores["behavior_verdicts"]}
    assert saved[before["verdicts"][1]["item_id"]] == before["verdicts"][1]
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert not final_review.final_review_functions()["ok"]


@pytest.mark.parametrize("recovery", ["explicit_final", "explicit_interrupted", "agent_repair"])
def test_explicit_final_review_recovery_preserves_cohort_and_completed_stages(
    tmp_path, monkeypatch, stub_claim_compiler, recovery,
):
    from autofa.mcp import tools_repair

    rt, fn = runtime(tmp_path, monkeypatch)
    calls = []
    broken = True
    interrupted = False

    def run(**kwargs):
        review = kwargs["review"]
        item = review.snapshot["item_id"]
        stage = "verification" if review.directory.name == "confirmation" else "initial"
        calls.append((item, stage))
        nonlocal interrupted
        if item == "id-1" and stage == "verification":
            if broken:
                raise ValueError("invalid scope annotation")
            if recovery == "explicit_interrupted" and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
        return reply(review, status="supported" if item == "id-1" else "contradicted")

    monkeypatch.setattr(claude_judge, "_run_review", run)
    assert not final_review.final_review_functions()["ok"]
    frozen = deepcopy(fn.scores["final_review"])
    original = {row["item_id"]: deepcopy(row) for row in fn.scores["behavior_verdicts"]}
    assert fn.scores["behavior_n_errors"] == 1
    assert fn.scores["behavior_n_no"] == 2
    paid_before = len(calls)
    assert not final_review.final_review_functions()["ok"]
    assert len(calls) == paid_before

    broken = False
    if recovery == "agent_repair":
        feedback = tools_repair.get_function_repair(fn.name)["feedback"]
        assert feedback["recommended_action"] == "retry_review"
        finding = next(row for row in feedback["findings"]
                       if row["status"] == "processing_error")
        plan = tools_repair.plan_function_repair(
            name=fn.name, action="retry_review", rationale="Recover the failed scope annotation.",
            finding_ids=[finding["id"]], evidence_ids=["id-1"],
        )
        assert plan["ok"], plan
        reviewed = tools_eval.judge_function(fn.name)
        assert reviewed["ok"], reviewed
        assert final_review.final_review_functions()["ok"]
    else:
        if recovery == "explicit_interrupted":
            with pytest.raises(KeyboardInterrupt):
                final_review.final_review_functions(names=[fn.name], retry_incomplete=True)
            assert fn.scores["final_review"]["status"] == "running"
            assert final_review.final_review_functions()["ok"]
        else:
            assert final_review.final_review_functions(
                names=[fn.name], retry_incomplete=True)["ok"]
        assert fn.scores["final_review"]["retry_generation"] == 1

    assert calls[paid_before:] == [("id-1", "verification")] * (
        2 if recovery == "explicit_interrupted" else 1)
    assert fn.scores["final_review"]["item_ids"] == frozen["item_ids"]
    assert fn.scores["final_review"]["operation_id"] == frozen["operation_id"]
    assert fn.scores["final_review"]["status"] == "complete"
    assert fn.scores["behavior_n_yes"] == 1 and fn.scores["behavior_n_no"] == 2
    current = {row["item_id"]: row for row in fn.scores["behavior_verdicts"]}
    assert all(current[key] == row for key, row in original.items() if key != "id-1")
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert final_review.final_review_functions()["ok"]
    assert final_review.final_review_functions(names=[fn.name], retry_incomplete=True)["ok"]


def test_settled_uncertainty_is_never_zero_or_rerolled_by_explicit_recovery(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    judge_rows(rt, fn, monkeypatch, representative=True)
    for row in fn.scores["behavior_verdicts"]:
        row.update(verdict="uncertain", completion_status="evidence_unavailable")
        row.pop("review_path", None)
    fn.scores.update(behavior_purity=None, behavior_n_no=0,
                     behavior_n_uncertain=3, behavior_n_completed=0)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    out = final_review.final_review_functions()
    assert not out["ok"]
    assert "inconclusive" in out["unresolved"][0]["reason"]
    assert fn.scores["behavior_purity"] is None
    assert not final_review.final_review_functions(
        names=[fn.name], retry_incomplete=True)["ok"]
    assert fn.scores["behavior_n_uncertain"] == 3
    assert not fn.scores["final_review"].get("retry_generation")


def _saved_uncertain_review_pending(tmp_path, *, inspection=None, review_path=None,
                                    snapshot_id=None):
    review = local_review(tmp_path / "behavior_judge")
    review.read("detector_source")
    review.read("response_text", 0, 60)
    row = json.loads(local_reply(review, inspection=inspection))["verdicts"][0]
    row.update(verdict="uncertain", review_path=review_path or str(review.directory))
    if snapshot_id:
        row["snapshot_id"] = snapshot_id
    rt = SimpleNamespace(run=SimpleNamespace(run_dir=lambda: tmp_path))
    return final_review._saved_completion_pending(rt, row)


@pytest.mark.parametrize("inspection,expected", [
    (None, True),
    ({"status": "unavailable", "ranges": [], "reason": "The task contract is absent."}, False),
    ({"status": "pending", "ranges": [], "reason": "The final artifact is unread."}, True),
    ({"status": "pending", "ranges": ["malformed"], "reason": "Invalid saved shape."}, False),
])
def test_explicit_recovery_rechecks_legacy_local_completion_without_rerolling_evidence(
    tmp_path, inspection, expected,
):
    assert _saved_uncertain_review_pending(tmp_path, inspection=inspection) is expected


@pytest.mark.parametrize("saved_path,snapshot_matches,expected", [
    ("/previous/run/behavior_judge/review", True, True),
    ("/previous/run/behavior_judge/review", False, False),
    ("/previous/run/behavior_judge/../behavior_judge/review", True, False),
    ("/previous/behavior_judge/run/behavior_judge/review", True, False),
])
def test_explicit_recovery_rebases_only_verified_archives_within_current_run(
    tmp_path, saved_path, snapshot_matches, expected,
):
    assert _saved_uncertain_review_pending(
        tmp_path, review_path=saved_path,
        snapshot_id=None if snapshot_matches else "changed-content",
    ) is expected


def test_final_review_cannot_reuse_changed_source_evidence(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    judge_rows(rt, fn, monkeypatch)
    rt.dataset._fake_items[1]["response_text"] += " changed"
    calls = record_reviews(monkeypatch)
    out = final_review.final_review_functions()
    assert out["ok"]
    assert "id-1" in calls
    assert "id-5" not in calls and "id-9" not in calls


def test_final_review_distinguishes_empty_bucket_from_all_execution_errors(tmp_path, monkeypatch):
    rt, fn = runtime(tmp_path, monkeypatch)
    fn.source = "def detect(item): return False"
    tools_eval.final_evaluate_functions()
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    assert final_review.final_review_functions()["ok"]
    assert fn.scores["final_review"]["status"] == "not_required"
    fn.scores.update(n_scored=0, n_items=0, n_unresolved=48, errors=48)
    out = final_review.final_review_functions()
    assert not out["ok"] and "No resolved predictions" in out["unresolved"][0]["reason"]


@pytest.mark.parametrize("failure", ["changed_content_then_backend_error", "source_decode_error"])
def test_unverifiable_evidence_cannot_restore_old_confirmations(
    tmp_path, monkeypatch, stub_claim_compiler, failure,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    judge_rows(rt, fn, monkeypatch)
    if failure == "source_decode_error":
        monkeypatch.setattr(tools_eval, "_fetch_items_by_id",
                            _raise(ValueError("corrupt source record")))
    else:
        rt.dataset._fake_items[1]["response_text"] += " changed"
        monkeypatch.setattr(claude_judge, "judge_behavior_with_claude",
                            _raise(RuntimeError("review backend offline")))
    out = final_review.final_review_functions()
    assert not out["ok"] and out["unresolved"][0]["name"] == fn.name
    assert fn.scores["behavior_evidence_stale"] is True
    assert resolved_behavior_purity(fn.scores) is None


def _checkpoint_args(tmp_path, snapshot, purpose, **extra):
    return dict(
        checkpoints=JudgeCheckpoints(tmp_path, {"purpose": purpose}), stage="initial",
        snapshot=snapshot, label=True, name="claim", mechanism="",
        description="No backup is made anywhere in the run.",
        source="def detect(item): return True", model="test", cli_path=None, max_chars=0,
        max_turns=64, analysis_guidance="", **extra,
    )


def test_final_recovery_checkpoint_retries_incomplete_once_but_not_uncertainty(
    tmp_path, monkeypatch,
):
    original = _review(tmp_path)
    calls = []

    def run(**kwargs):
        review = kwargs["review"]
        calls.append(review.directory)
        review.read("detector_source")
        review.read("response_text", 0, 50)
        return _reply(review, unknown_reason="review_incomplete")

    monkeypatch.setattr(claude_judge, "_run_review", run)
    args = _checkpoint_args(tmp_path, original.snapshot, "final-recovery")
    archive = tmp_path / "behavior_judge"
    first = claude_judge._checkpointed_review(**args, directory=archive / "first")
    assert first[1]["completion_status"] == "needs_evidence_review"
    assert claude_judge._checkpointed_review(**args, directory=archive / "reuse")[3]
    recovered = claude_judge._checkpointed_review(
        **args, directory=archive / "final", recovery_id="frozen-operation"
    )
    assert not recovered[3] and len(calls) == 2
    repeated = claude_judge._checkpointed_review(
        **args, directory=archive / "resume", recovery_id="frozen-operation"
    )
    assert repeated[3] and len(calls) == 2


def test_final_recovery_checkpoint_keeps_processing_error_after_interruption(
    tmp_path, monkeypatch,
):
    review = _review(tmp_path)
    calls = []

    def broken(**kwargs):
        calls.append(kwargs["review"].directory)
        raise ValueError("invalid response")

    monkeypatch.setattr(claude_judge, "_run_review", broken)
    args = _checkpoint_args(tmp_path, review.snapshot, "final-error",
                            recovery_id="frozen-operation")
    archive = tmp_path / "behavior_judge"
    first = claude_judge._checkpointed_review(**args, directory=archive / "first")
    assert first[1]["verdict"] == "error"
    repeated = claude_judge._checkpointed_review(**args, directory=archive / "resume")
    assert repeated[1] == first[1] and repeated[3] and len(calls) == 1


def test_successful_review_aggregate_does_not_count_backend_error_as_rejection(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    statuses = {}

    def run(**kwargs):
        review = kwargs["review"]
        item_id = review.snapshot["item_id"]
        if item_id == "id-1":
            raise ValueError("invalid reply")
        # One confirmed and one inconclusive review besides the processing error.
        statuses.setdefault(item_id, "unknown" if statuses else "supported")
        return reply(review, status=statuses[item_id])

    monkeypatch.setattr(claude_judge, "_run_review", run)
    assert not final_review.final_review_functions()["ok"]
    # Uncertainty stays in the adherence denominator; processing errors do not.
    assert fn.scores["behavior_purity"] == .5
    assert fn.scores["behavior_n_yes"] == fn.scores["behavior_n_uncertain"] == 1
    assert fn.scores["behavior_n_errors"] == 1 and fn.scores["behavior_n_no"] == 0
    assert fn.scores["final_review"]["status"] == "unavailable"


@pytest.mark.parametrize("failure,kind,retries", [
    pytest.param("refusal", "backend_refusal", 0, id="refusal-never-repeated"),
    pytest.param("invalid", "invalid_reply", 1, id="invalid-reply-retried-once"),
])
def test_explicit_retry_of_failed_review_is_bounded_by_failure_kind(
    tmp_path, monkeypatch, stub_claim_compiler, failure, kind, retries,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    calls = []

    def run(**kwargs):
        review = kwargs["review"]
        calls.append(review.snapshot["item_id"])
        if review.snapshot["item_id"] != "id-1":
            return reply(review, status="contradicted")
        if failure == "invalid":
            return "invalid JSON"
        raise RuntimeError("success/refusal: safeguards flagged [reasoning_extraction]")

    monkeypatch.setattr(claude_judge, "_run_review", run)
    assert not final_review.final_review_functions()["ok"]
    before = len(calls)
    for _ in range(2):
        assert not final_review.final_review_functions(names=[fn.name], retry_incomplete=True)["ok"]
    assert len(calls) == before + retries
    processing = fn.scores["behavior_processing_failures"][0]
    assert processing["kind"] == kind and not processing["detector_rejected"]
    if not retries:
        assert fn.scores["final_review"]["recovery"]["status"] == "blocked"


def test_legacy_count_only_completion_cannot_keep_a_failure_only_judge_cohort(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    # Sixty matches, six old reviews: the previous count shortcut retained them
    # even though the remaining fifty-four matched attempts passed the benchmark.
    matched = [item for item in rt.dataset._fake_items if item["response_text"] == "BAD"]
    for item in matched[6:]:
        item["is_correct"] = True
    fn.scores = {}
    rt._family_failed_task_inventory = None
    assert tools_eval.final_evaluate_functions()["ok"]
    judge_rows(rt, fn, monkeypatch, count=6,
               runner=lambda **kw: reply(kw["review"], status="supported"))
    previous = deepcopy(fn.scores["behavior_verdicts"])
    fn.scores["final_review"] = {
        "status": "complete", "required_n": 6,
        "item_ids": [row["item_id"] for row in previous],
    }
    assert fn.scores["behavior_purity"] == 1
    calls = record_reviews(monkeypatch)
    assert final_review.final_review_functions()["ok"]
    cohort = set(fn.scores["final_review"]["item_ids"])
    assert len(cohort) == fn.scores["behavior_n_judged"] == 6
    assert cohort & {item["_item_id"] for item in matched[6:]}
    assert fn.scores["behavior_purity"] < 1
    # Previously inspected evidence was not discarded or allowed to pad scores.
    all_saved = fn.scores["behavior_verdicts"] + fn.scores["diagnostic_review"]["verdicts"]
    saved_by_id = {row["item_id"]: row for row in all_saved}
    assert all(saved_by_id[row["item_id"]] == row for row in previous)
    assert not set(calls) & {row["item_id"] for row in previous}
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert final_review.final_review_functions()["ok"]


def test_review_cohort_uses_distinct_tasks_and_stays_fixed_across_rounds(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    matched = [item for item in rt.dataset._fake_items if item["response_text"] == "BAD"]
    for index, item in enumerate(matched):
        item["problem_id"] = "repeated" if index < 40 else str(index)
        item["benchmark"] = "same"
    rt._family_failed_task_inventory = None
    fn.scores = {}
    assert tools_eval.final_evaluate_functions()["ok"]
    monkeypatch.setattr(claude_judge, "_run_review", lambda **kw: reply(kw["review"]))
    assert final_review.final_review_functions()["ok"]
    cohort = fn.scores["final_review"]["item_ids"]
    tasks = {item["problem_id"] for item in matched if item["_item_id"] in cohort}
    assert len(tasks) == len(cohort) == 6
    rt.run.round += 1
    rt.run.exploration_seed += 321
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert final_review.final_review_functions()["ok"]
    assert fn.scores["final_review"]["item_ids"] == cohort


def checkpoint_scores(rt, fn):
    from autofa.metrics.association import StreamingBucketAccumulator, pick_primary_bucket

    panel = rt.dataset._fake_items[:120]
    acc = StreamingBucketAccumulator()
    for item in panel:
        acc.update(item["response_text"] == "BAD", is_correct=item["is_correct"],
                   item_id=item["_item_id"], weight=len(rt.dataset._fake_items) / len(panel))
    buckets = acc.finalize(sampled=True)
    fn.scores.update(pick_primary_bucket(buckets))
    fn.scores.update(
        buckets=buckets, sampled=True, n_scored=len(panel),
        n_unresolved=0, evaluation_scope_complete=False,
        evaluation_item_ids=[item["_item_id"] for item in panel],
        round_validation={"version": 1, "complete": True, "fraction": .3,
                          "allocation": "proportional_benchmark_outcome",
                          "panel_fingerprint": "test-proportional-panel"},
    )
    return panel


@pytest.mark.parametrize("missing,reason", [
    ("complete", "checkpoint scoring"), ("allocation", "checkpoint scoring"),
    ("panel_fingerprint", "checkpoint scoring"),
    # An incomplete match inventory is not reported as "no matches".
    ("buckets", "membership is incomplete"),
])
def test_round_review_requires_complete_checkpoint_receipt(
    tmp_path, monkeypatch, missing, reason,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    checkpoint_scores(rt, fn)
    if missing == "buckets":
        fn.scores["buckets"] = []
    else:
        fn.scores["round_validation"].pop(missing)
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    result = final_review.final_review_functions(scope="round_checkpoint")
    assert not result["ok"]
    assert reason in result["unresolved"][0]["reason"]
    assert "final_review" not in fn.scores


def test_real_round_scoring_and_review_reuse_checkpoint_across_rounds(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    fn = _llm_flag(description="The agent guesses.", mechanism="It acts on a guess.")
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=2, functions=[fn], n=160)
    rt.config.judge_workers = 1
    calls = record_reviews(monkeypatch)
    assert tools_eval.evaluate_functions()["ok"]
    assert tools_eval.evaluate_round_functions()["ok"]
    panel = set(fn.scores["evaluation_item_ids"])
    assert len(panel) == 48
    matches = {item["_item_id"] for item in rt.dataset._fake_items
               if item["_item_id"] in panel and item["response_text"] == "BAD"}
    result = final_review.final_review_functions(scope="round_checkpoint")
    assert result["ok"], result
    receipt = deepcopy(fn.scores["round_review"])
    assert receipt["matched_n"] == len(matches)
    assert receipt["required_n"] == tools_eval.judge_evidence_sample_size(len(matches))
    assert set(receipt["item_ids"]) <= matches
    n_calls = len(calls)
    rt.run.round += 1
    rt.run.exploration_seed += 100
    assert tools_eval.evaluate_round_functions()["ok"]
    assert final_review.final_review_functions(scope="round_checkpoint")["ok"]
    assert fn.scores["round_review"] == receipt
    assert len(calls) == n_calls
    assert tools_eval.final_evaluate_functions()["ok"]
    assert final_review.final_review_functions()["ok"]
    final_ids = fn.scores["final_review"]["item_ids"]
    assert len(final_ids) == fn.scores["behavior_n_judged"] == 4
    assert len(calls) == n_calls + len(set(final_ids) - set(receipt["item_ids"]))


def test_final_review_keeps_extra_reviews_as_diagnostics_until_their_evidence_changes(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    from autofa.metrics.diagnostic_review import diagnostic_scores

    rt, fn = runtime(tmp_path, monkeypatch)
    judge_rows(rt, fn, monkeypatch, count=8)
    previous = deepcopy(fn.scores["behavior_verdicts"])
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    assert final_review.final_review_functions()["ok"]
    saved, diagnostics = fn.scores["behavior_verdicts"], fn.scores["diagnostic_review"]
    assert len(saved) == fn.scores["behavior_n_judged"] == 3
    assert {row["item_id"] for row in saved}.isdisjoint(
        row["item_id"] for row in diagnostics["verdicts"])
    assert {row["item_id"] for row in saved + diagnostics["verdicts"]} == {
        row["item_id"] for row in previous}
    assert diagnostics["excluded_from_adherence"] is True
    assert diagnostics["excluded_from_confirmed_support"] is True
    receipt = deepcopy(fn.scores["final_review"])
    diagnostic = fn.scores["diagnostic_review"]["verdicts"][0]
    item_id = diagnostic["item_id"]
    assert item_id not in receipt["item_ids"]
    assert diagnostic_scores(fn.scores) is not None
    for item in rt.dataset._fake_items:
        if item["_item_id"] == item_id:
            item["response_text"] += " changed off-cohort evidence"
    monkeypatch.setattr(claude_judge, "_run_review", forbid)
    assert final_review.final_review_functions()["ok"]
    assert fn.scores["final_review"] == receipt
    assert fn.scores["diagnostic_evidence_stale"] is True
    saved = next(row for row in fn.scores["diagnostic_review"]["verdicts"]
                 if row["item_id"] == item_id)
    assert saved["verification_status"] == "historical"
    assert saved["historical_review"] == diagnostic
    assert diagnostic_scores(fn.scores) is None


def test_selected_diagnostic_review_can_be_reused_with_its_original_contract(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    _, result = judge_rows(rt, fn, monkeypatch, count=1, representative=True)
    original = deepcopy(fn.scores)
    item_id = result["verdicts"][0]["item_id"]
    diagnostic = {
        **result, "review_mode": "diagnostic", "excluded_from_adherence": True,
        "excluded_from_confirmed_support": True, "excluded_from_repair_validation": True,
        "sample": {"limited": True}, "review_localisations": {item_id: None},
    }
    fn.scores = {
        key: value for key, value in original.items()
        if not key.startswith(("behavior_", "judge_"))
        and key not in {"claim_hash", "claim_contract", "claim_compilation"}
    }
    fn.scores["diagnostic_review"] = diagnostic
    judge = claude_judge.judge_behavior_with_claude
    calls = []

    def capture(**kwargs):
        assert kwargs["compiled_contract"] == diagnostic["claim_contract"]
        calls.extend(item["_item_id"] for item in kwargs["items"])
        return judge(**kwargs)

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", capture)
    assert final_review.final_review_functions()["ok"]
    assert len(calls) == 2 and item_id not in calls
    assert fn.scores["claim_contract"] == diagnostic["claim_contract"]
    assert fn.scores["behavior_n_judged"] == 3
    assert fn.scores["judge_policy_hash"] == diagnostic["judge_policy_hash"]


@pytest.mark.parametrize("precision", [.2, .55])
def test_round_review_defers_new_paid_work_using_existing_schedule(
    tmp_path, monkeypatch, stub_claim_compiler, precision,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    judge_rows(rt, fn, monkeypatch, count=1, representative=True,
               runner=lambda **kw: reply(kw["review"], status="supported"))
    checkpoint_scores(rt, fn)
    fn.scores["ppv_at_pi0"] = precision
    previous = deepcopy(fn.scores["behavior_verdicts"])
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    result = final_review.final_review_functions(scope="round_checkpoint")
    assert result["ok"] and result["reviewed"] == []
    assert result["deferred"][0]["name"] == fn.name
    assert result["deferred"][0]["next_steps"]
    assert fn.scores["round_review"]["status"] == "deferred"
    assert fn.scores["behavior_verdicts"] == previous
    assert fn.scores["behavior_purity"] == 1
    assert fn.scores["review_schedule"]["status"] == "deferred"


def test_missing_final_match_inventory_cannot_certify_good_from_old_biased_reviews(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    judge_rows(rt, fn, monkeypatch, count=6,
               runner=lambda **kw: reply(kw["review"], status="supported"))
    assert fn.scores["behavior_purity"] == 1
    buckets = fn.scores.pop("buckets")
    assert not final_review.final_review_functions()["ok"]
    assert fn.scores["behavior_evidence_stale"] is True
    assert resolved_behavior_purity(fn.scores) is None
    assert verdict_for_scores(fn.scores)["code"] != "GOOD"
    fn.scores["buckets"] = buckets
    assert final_review.final_review_functions()["ok"]
    assert fn.scores["final_review"]["cohort_policy"] == final_review.COHORT_POLICY
    assert resolved_behavior_purity(fn.scores) == 1
    assert verdict_for_scores(fn.scores)["code"] == "GOOD"


def test_round_review_recovers_frozen_allocation_despite_new_scheduling_deferral(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch)
    calls = record_reviews(monkeypatch, interrupt_at=2)
    with pytest.raises(KeyboardInterrupt):
        final_review.final_review_functions(scope="round_checkpoint")
    frozen = deepcopy(fn.scores["round_review"])
    fn.scores["ppv_at_pi0"] = .2
    result = final_review.final_review_functions(scope="round_checkpoint")
    assert result["ok"] and result["deferred"] == []
    assert fn.scores["round_review"]["status"] == "complete"
    assert fn.scores["round_review"]["item_ids"] == frozen["item_ids"]
    assert len(calls) == 4 and calls.count(calls[0]) == 1


def test_thrown_review_error_cannot_leave_old_biased_aggregate_good(
    tmp_path, monkeypatch, stub_claim_compiler,
):
    rt, fn = runtime(tmp_path, monkeypatch, n=240)
    judge_rows(rt, fn, monkeypatch, count=6,
               runner=lambda **kw: reply(kw["review"], status="supported"))
    old_rows = deepcopy(fn.scores["behavior_verdicts"])
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude",
                        _raise(RuntimeError("judge backend unavailable")))
    assert not final_review.final_review_functions()["ok"]
    assert fn.scores["final_review"]["identity"]
    assert fn.scores["final_review"]["status"] == "unavailable"
    assert fn.scores["behavior_verdicts"] == old_rows
    assert fn.scores["behavior_purity"] == 1  # Preserved as historical measured data.
    assert resolved_behavior_purity(fn.scores) is None
    assert verdict_for_scores(fn.scores)["code"] != "GOOD"
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    assert not final_review.final_review_functions()["ok"]
    assert resolved_behavior_purity(fn.scores) is None
    assert verdict_for_scores(fn.scores)["code"] != "GOOD"


@pytest.mark.parametrize("failure", [
    TimeoutError("request timed out"), RuntimeError("API error: 503"),
])
def test_raw_transient_final_review_failure_resumes_completed_stage_checkpoints(
    tmp_path, monkeypatch, stub_claim_compiler, failure,
):
    from autofa.session.recovery import RunInterruptedError

    rt, fn = runtime(tmp_path, monkeypatch)
    sessions = []

    def reviewed(**kwargs):
        sessions.append(kwargs["review"].snapshot["item_id"])
        return reply(kwargs["review"], status="supported")

    monkeypatch.setattr(claude_judge, "_run_review", reviewed)
    original = claude_judge.judge_behavior_with_claude
    attempts = []

    def interrupted_orchestration(**kwargs):
        attempts.append(kwargs["_recovery_id"])
        result = original(**kwargs)
        if len(attempts) == 1:
            # The service failed after review stages completed but before the
            # aggregate reached the caller. Those paid stages must survive.
            raise failure
        return result

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", interrupted_orchestration)
    with pytest.raises(RunInterruptedError, match="temporary review service failure"):
        final_review.final_review_functions()
    frozen = deepcopy(fn.scores["final_review"])
    assert frozen["status"] == "running" and frozen["retryable"] is True
    assert len(attempts) == 1  # No hidden retry loop or second allocation.
    assert len(sessions) == 6  # Initial review and confirmation for three items.
    assert resolved_behavior_purity(fn.scores) is None

    assert final_review.final_review_functions()["ok"]
    assert attempts == [frozen["operation_id"], frozen["operation_id"]]
    assert len(sessions) == 6  # Recovery uses saved stages, not paid sessions.
    assert fn.scores["final_review"]["item_ids"] == frozen["item_ids"]
    assert fn.scores["final_review"]["status"] == "complete"
    assert resolved_behavior_purity(fn.scores) == 1
    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", forbid)
    assert final_review.final_review_functions()["ok"]


@pytest.mark.parametrize("when", ["before_commit", "during_nested_save"])
def test_final_review_checkpoint_observes_stop_only_before_commit(tmp_path, monkeypatch, when):
    import asyncio

    from autofa.core.library import FunctionLibrary
    from autofa.util.cancellation import Cancellation, cancellation_scope, check_cancelled

    rt, fn = runtime(tmp_path, monkeypatch, n=8)
    rt.persist_library()
    before_scores = deepcopy(fn.scores)
    before_bytes = rt.working_library_path().read_bytes()
    stopped = when == "before_commit"
    token = Cancellation(lambda: stopped)
    persist = rt.persist_library

    def stop_at_nested_save(**kwargs):
        nonlocal stopped
        stopped = True
        token._last_poll = 0
        return persist(**kwargs)

    if when == "during_nested_save":
        monkeypatch.setattr(rt, "persist_library", stop_at_nested_save)
    receipt = {"scope": "final", "status": "complete", "regression": "saved-review"}
    with cancellation_scope(token):
        if when == "before_commit":
            with pytest.raises(asyncio.CancelledError):
                final_review._commit(rt, fn, before_scores, receipt)
            assert fn.scores == before_scores
            assert rt.working_library_path().read_bytes() == before_bytes
        else:
            final_review._commit(rt, fn, before_scores, receipt)
            assert fn.scores["final_review"] == receipt
            saved = FunctionLibrary.load(rt.working_library_path()).get(fn.name)
            assert saved.scores["final_review"] == receipt
            with pytest.raises(asyncio.CancelledError):
                check_cancelled()
