# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trace-blind defining-clause compiler, its shared budget and its place before review."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.metrics import claim_compiler, claude_judge, codex_judge
from autofa.metrics.claim_compiler import ClaimCompilationError, compiler_policy_hash
from autofa.metrics.claim_contract import build_claim_contract, claim_hash
from autofa.metrics.review_failures import processing_failure, review_failures
from autofa.util.cancellation import Cancellation, cancellation_scope

DESCRIPTION = "The assistant closes as fixed while service remains broken."
MECHANISM = (
    "It mistakes activity for success, e.g. repeated resets. The fix is final verification."
)
SOURCE = "DETECTOR_CODE_MUST_NOT_REACH_COMPILER"
CLAIM = {"description": DESCRIPTION, "mechanism": MECHANISM, "source": SOURCE}
CLAUSES = [
    {"id": "clause_cause", "source_field": "mechanism",
     "quote": "mistakes activity for success",
     "required_review_scope": "local",
     "predicate": "The assistant mistakes activity for success."},
    {"id": "clause_outcome", "source_field": "description",
     "quote": "service remains broken",
     "required_review_scope": "local",
     "predicate": "The service remains broken when the assistant closes."},
]
APPROVED = {"approved": True, "issues": []}


def _payload(**changes):
    return {
        "defining_clauses": deepcopy(CLAUSES),
        "exclusions": [
            {"source_field": "mechanism", "quote": "e.g. repeated resets",
             "reason": "An illustrative example, not a required action."},
            {"source_field": "mechanism", "quote": "The fix is final verification.",
             "reason": "A proposed remedy, not an observed defining condition."},
        ],
        **changes,
    }


def _coverage_issue():
    return {"source_field": "mechanism", "quote": "mistakes activity for success",
            "reason": "The asserted reasoning error needs its own defining clause."}


def _answer(kwargs, payload=None, *, approved=True):
    """Reply as the drafting compiler, or as the coverage auditor for an audit prompt."""
    if kwargs.get("system_prompt") == claim_compiler.COVERAGE_SYSTEM:
        return json.dumps(APPROVED if approved else
                          {"approved": False, "issues": [_coverage_issue()]})
    return json.dumps(_payload() if payload is None else payload)


def _stub_compiler(monkeypatch, payload=None, *, approved=True):
    calls = []
    monkeypatch.setattr(claim_compiler, "_run_compiler", lambda **kwargs: calls.append(
        kwargs) or _answer(kwargs, payload, approved=approved))
    return calls


def _parse(payload):
    return claim_compiler.parse_compilation(json.dumps(payload), **CLAIM)


def _validate(contract, **claim):
    return claim_compiler.validate_compiled_contract(contract, **{**CLAIM, **claim})


def _compile(directory):
    return claim_compiler.compile_claim_contract(
        **CLAIM, model="test-model", directory=directory, cli_path="test-cli",
    )


def _compile_cached(tmp_path, name, **changes):
    return claim_compiler.compile_claim_contract(**{
        "description": DESCRIPTION, "mechanism": MECHANISM, "source": "def f(x): return True",
        "model": "test", "directory": tmp_path / name, "cache_dir": tmp_path / "shared",
        **changes,
    })


def _approved_contract():
    return build_claim_contract(**CLAIM, defining_clauses=CLAUSES, compilation={
        "policy_hash": compiler_policy_hash(), "model": "test", "exclusions": [],
        "coverage_review": APPROVED,
    })


# Parsing and validation of a frozen compilation.


def test_exact_defining_clauses_are_frozen_with_the_original_claim_and_detector_identity():
    contract = _parse(_payload())
    assert contract["defining_clauses"] == CLAUSES
    assert contract["description"] == DESCRIPTION and contract["mechanism"] == MECHANISM
    assert contract["claim_hash"] == claim_hash(**CLAIM)
    assert contract["compilation"] == {
        "policy_hash": claim_compiler.compiler_policy_hash(),
        "model": "unspecified", "exclusions": _payload()["exclusions"],
    }
    with pytest.raises(ValueError, match="coverage review"):
        _validate(contract)
    approved = build_claim_contract(
        **CLAIM, defining_clauses=contract["defining_clauses"],
        compilation={**contract["compilation"], "coverage_review": APPROVED},
    )
    assert _validate(approved) == approved


@pytest.mark.parametrize("raw, error", [
    ("not JSON", None),
    ("{}", None),
    (json.dumps(_payload(defining_clauses=[
        {**CLAUSES[0], "id": f"clause_{index}"} for index in range(33)])), None),
    (json.dumps(_payload(defining_clauses=[
        {**CLAUSES[0], "quote": "the assistant knowingly lies"}])), None),
    (json.dumps(_payload(exclusions=[
        {"source_field": "trace", "quote": "service remains broken", "reason": "example"}])),
     None),
    (json.dumps(_payload(defining_clauses=[
        {key: value for key, value in clause.items() if key != "required_review_scope"}
        for clause in CLAUSES])), "required_review_scope"),
], ids=["not_json", "empty", "too_many_clauses", "unanchored_clause", "unanchored_exclusion",
        "legacy_without_review_scope"])
def test_malformed_or_unanchored_compilation_never_becomes_a_contract(raw, error):
    with pytest.raises(ValueError, match=error):
        claim_compiler.parse_compilation(raw, **CLAIM)


def test_frozen_compilation_cannot_be_reused_for_a_different_claim():
    with pytest.raises(ValueError, match="does not match"):
        _validate(_parse(_payload()), mechanism=MECHANISM + " ")


@pytest.mark.parametrize("policy", ["COMPILER_SYSTEM", "COVERAGE_SYSTEM"])
def test_previous_compiler_policy_cannot_be_reused_under_the_current_policy(monkeypatch, policy):
    contract = _parse(_payload())
    original_policy = contract["compilation"]["policy_hash"]
    monkeypatch.setattr(claim_compiler, policy, "Changed defining-clause policy.")
    assert claim_compiler.compiler_policy_hash() != original_policy
    with pytest.raises(ValueError, match="policy"):
        _validate(contract)


def _without_last_requirement():
    contract = _parse(_payload())
    contract["requirements"].pop()
    return contract


@pytest.mark.parametrize("make_contract, error", [
    (lambda: build_claim_contract(**CLAIM), "compiled"),
    (_without_last_requirement, "does not match"),
    (lambda: build_claim_contract(**CLAIM, defining_clauses=CLAUSES), "policy"),
    (lambda: build_claim_contract(**CLAIM, defining_clauses=CLAUSES, compilation={
        **_parse(_payload())["compilation"],
        "coverage_review": {"approved": False, "issues": [_coverage_issue()]},
    }), "approved coverage"),
], ids=["uncompiled", "missing_requirement", "no_policy_provenance", "rejected_coverage"])
def test_altered_or_unapproved_compilation_cannot_enter_production(make_contract, error):
    with pytest.raises(ValueError, match=error):
        _validate(make_contract())


# Drafting, schema repair and the failure paths without a weaker fallback.


def test_compilation_and_coverage_audit_use_the_same_explicit_effort(tmp_path, monkeypatch):
    calls = _stub_compiler(monkeypatch)
    contract = claim_compiler.compile_claim_contract(
        **CLAIM, model="claude-sonnet-5-5", effort="medium", thinking="enabled",
        thinking_budget_tokens=2048, directory=tmp_path / "compile",
    )
    assert len(calls) == 2
    assert all((call["model"], call["effort"], call["thinking"], call["thinking_budget_tokens"])
               == ("claude-sonnet-5-5", "medium", "enabled", 2048) for call in calls)
    assert contract["compilation"]["model"] == "claude-sonnet-5-5"
    assert contract["compilation"]["coverage_review"]["approved"]


def test_compiler_repairs_schema_with_original_claim_only_and_archives_both_attempts(
    tmp_path, monkeypatch,
):
    calls = []
    malformed = _payload()
    malformed["defining_clauses"][0]["quote"] = "a paraphrase, not an exact quote"
    replies = [json.dumps(malformed), json.dumps(_payload())]

    def run(**kwargs):
        if kwargs.get("system_prompt") == claim_compiler.COVERAGE_SYSTEM:
            return json.dumps(APPROVED)
        calls.append(kwargs)
        return replies[len(calls) - 1]

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    directory = tmp_path / "compilation"
    contract = _compile(directory)
    assert json.loads(calls[0]["prompt"]) == {"description": DESCRIPTION, "mechanism": MECHANISM}
    assert calls[1]["prompt"].startswith(calls[0]["prompt"])
    for index, call in enumerate(calls):
        assert call["attempt"] == index
        assert SOURCE not in call["prompt"]
        assert (directory / f"reply-{index}.txt").read_text() == replies[index]
    assert (directory / "error-0.txt").is_file()
    assert json.loads((directory / "contract.json").read_text()) == contract
    assert contract["compilation"]["coverage_review"] == APPROVED


def test_validation_repair_budget_exhaustion_raises_without_weaker_fallback(tmp_path, monkeypatch):
    attempts = []

    def run(**kwargs):
        attempts.append(kwargs["attempt"])
        return json.dumps({"defining_clauses": [], "exclusions": []})

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    directory = tmp_path / "compilation"
    with pytest.raises(ValueError, match="failed validation"):
        _compile(directory)
    assert attempts == [0, 1, 2]
    assert not (directory / "contract.json").exists()


# The stop reason decides acceptance; the prompt and effort pass through.
@pytest.mark.parametrize("subtype,is_error,stop_reason,coverage,effort", [
    ("success", False, "end_turn", False, None),
    ("success", False, "max_tokens", True, "medium"),
    ("error_during_execution", True, "end_turn", False, None),
])
def test_sdk_compiler_is_isolated_and_archives_usage_even_on_failure(
    tmp_path, monkeypatch, subtype, is_error, stop_reason, coverage, effort,
):
    import claude_agent_sdk as sdk

    monkeypatch.setattr("autofa.harness.claude_cli.ensure_claude_auth",
                        lambda **kwargs: "fake-claude")
    cleaned_up = []

    async def query(*, prompt, options):
        assert prompt == "claim only"
        assert options.system_prompt == (
            claim_compiler.COVERAGE_SYSTEM if coverage else claim_compiler.COMPILER_SYSTEM
        )
        assert options.tools == options.allowed_tools == options.setting_sources == []
        assert options.strict_mcp_config and not options.mcp_servers
        assert options.max_turns == 1 and not options.resume
        assert options.cwd == str(tmp_path) and options.cli_path == "fake-claude"
        assert options.model == "test-model" and options.effort == effort
        assert options.thinking == ({"type": "adaptive"} if effort else None)
        original_task = asyncio.current_task()
        try:
            yield sdk.ResultMessage(
                subtype=subtype, duration_ms=1, duration_api_ms=1, is_error=is_error,
                num_turns=1, session_id="compiler-session", stop_reason=stop_reason,
                result=json.dumps(_payload()), usage={"input_tokens": 100, "output_tokens": 50},
                total_cost_usd=0.01,
            )
        finally:
            assert asyncio.current_task() is original_task
            cleaned_up.append(True)

    monkeypatch.setattr(sdk, "query", query)
    kwargs = {"prompt": "claim only", "model": "test-model", "cli_path": None,
              "directory": tmp_path, "attempt": 0, "effort": effort}
    if coverage:
        kwargs["system_prompt"] = claim_compiler.COVERAGE_SYSTEM

    async def run():
        try:
            return await claim_compiler._compiler_session(**kwargs)
        finally:
            # Assert before loop shutdown, which can otherwise hide leaked SDK streams.
            assert cleaned_up == [True]

    if not is_error and stop_reason == "end_turn":
        assert asyncio.run(run()) == json.dumps(_payload())
    else:
        with pytest.raises(RuntimeError, match="incomplete claim compilation"):
            asyncio.run(run())
    metadata = json.loads((tmp_path / "session-0.json").read_text())
    assert metadata["usage"] == {"input_tokens": 100, "output_tokens": 50}
    assert metadata["cost_usd"] == 0.01 and metadata["cost_kind"] == "estimated"
    assert (metadata["is_error"], metadata["stop_reason"]) == (is_error, stop_reason)


@pytest.mark.parametrize("inside_loop", [False, True])
def test_compiler_runner_does_not_retry_sdk_errors_with_a_second_context(monkeypatch, inside_loop):
    calls = []

    async def session(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("real SDK error")

    monkeypatch.setattr(claim_compiler, "_compiler_session", session)

    async def in_loop():
        return claim_compiler._run_compiler(prompt="claim only")

    with pytest.raises(RuntimeError, match="real SDK error"):
        if inside_loop:
            asyncio.run(in_loop())
        else:
            claim_compiler._run_compiler(prompt="claim only")
    assert calls == [{"prompt": "claim only"}]


def test_codex_compiler_keeps_completed_result_while_accounting_finishes(tmp_path, monkeypatch):
    wait_for = asyncio.wait_for

    async def short_outer_deadline(awaitable, timeout):
        return await wait_for(awaitable, .01 if timeout == 180 else timeout)

    async def completed_turn(**kwargs):
        assert kwargs["timeout_s"] == 180
        # The model has completed; only its separately bounded accounting remains.
        await asyncio.sleep(.03)
        return SimpleNamespace(
            thread_id="compiler", turn_id="turn", status="completed", error=None,
            text="compiled claim", usage={}, error_code=None,
        )

    monkeypatch.setattr(asyncio, "wait_for", short_outer_deadline)
    monkeypatch.setattr(codex_judge, "_run_turn", completed_turn)
    result = claim_compiler._run_compiler(
        prompt="claim only", model="test-model", cli_path=None, directory=tmp_path,
        attempt=0, harness="codex",
    )
    assert result == "compiled claim"
    assert json.loads((tmp_path / "session-0.json").read_text())["subtype"] == "success"


# The independent, trace-blind coverage audit.


def test_coverage_audit_is_fresh_trace_blind_and_does_not_repair_the_draft(tmp_path, monkeypatch):
    draft = _parse(_payload())
    original = deepcopy(draft)
    draft["private_rollout"] = "PRIVATE-TRACE-MUST-NOT-REACH-AUDITOR"
    response = {"approved": False, "issues": [_coverage_issue()]}
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        assert kwargs["system_prompt"] == claim_compiler.COVERAGE_SYSTEM
        assert json.loads(kwargs["prompt"]) == {
            "description": DESCRIPTION, "mechanism": MECHANISM,
            "defining_clauses": CLAUSES, "exclusions": _payload()["exclusions"],
        }
        return json.dumps(response)

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    directory = tmp_path / "coverage"
    result = claim_compiler.audit_claim_compilation(
        description=DESCRIPTION, mechanism=MECHANISM, contract=draft,
        model="test-model", directory=directory,
    )
    assert result == response and len(calls) == 1
    assert draft == {**original, "private_rollout": "PRIVATE-TRACE-MUST-NOT-REACH-AUDITOR"}
    assert json.loads((directory / "audit.json").read_text()) == response


@pytest.mark.parametrize("response", [
    "not JSON",
    {"approved": True, "issues": [_coverage_issue()]},
    {"approved": False, "issues": [{**_coverage_issue(), "quote": "a missing source quote"}]},
    # A truthy string must not turn a rejecting audit into an approval.
    {"approved": "false", "issues": [_coverage_issue()]},
], ids=["not_json", "approved_with_issues", "unanchored_issue", "string_approval"])
def test_invalid_or_ambiguous_audit_cannot_approve_a_draft(tmp_path, monkeypatch, response):
    monkeypatch.setattr(claim_compiler, "_run_compiler", lambda **kwargs: json.dumps(response))
    directory = tmp_path / "coverage"
    with pytest.raises(ValueError):
        claim_compiler.audit_claim_compilation(
            description=DESCRIPTION, mechanism=MECHANISM, contract=_parse(_payload()),
            model="test-model", directory=directory,
        )
    assert (directory / "error.txt").is_file()
    assert not (directory / "audit.json").exists()


def test_rejected_coverage_exhausts_the_shared_draft_budget_without_a_fallback(
    tmp_path, monkeypatch,
):
    calls = _stub_compiler(monkeypatch, approved=False)
    directory = tmp_path / "compilation"
    with pytest.raises(ValueError, match="coverage audit rejected"):
        _compile(directory)
    assert len(calls) == 6
    assert len(list(directory.glob("coverage-*/audit.json"))) == 3
    assert not (directory / "contract.json").exists()


def test_coverage_session_failure_never_falls_back_to_the_valid_but_unaudited_draft(
    tmp_path, monkeypatch,
):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        if kwargs.get("system_prompt") == claim_compiler.COVERAGE_SYSTEM:
            raise RuntimeError("coverage SDK failed")
        return _answer(kwargs)

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    directory = tmp_path / "compilation"
    with pytest.raises(RuntimeError, match="coverage SDK failed"):
        _compile(directory)
    assert len(calls) == 2
    assert not (directory / "contract.json").exists()


def test_no_backup_before_probe_repairs_scope_and_start_anchor_before_item_review(
    tmp_path, monkeypatch,
):
    description = "The assistant mutates the recovery source without first copying it."
    mechanism = "The source disappears, and the submitted output guesses the missing data."
    draft = {"defining_clauses": [
        {"id": "prior_backup", "source_field": "description", "quote": "without first copying it",
         "predicate": "No copy precedes the claimed state-mutating probe of the original source.",
         "required_review_scope": "whole_trace"},
        {"id": "mutation", "source_field": "description",
         "quote": "The assistant mutates the recovery source",
         "predicate": "The assistant mutates the original recovery source.",
         "required_review_scope": "local"},
    ], "exclusions": []}
    interval = {
        "start": {"kind": "run_start"},
        "end": {"kind": "event",
                "predicate": "The claimed state-mutating probe of the original source",
                "position": "before"},
    }
    issues = [
        "The absence of a prior copy needs a complete prefix, not unrelated later turns.",
        "The prefix must begin at run_start; starting at inspection could hide an earlier copy.",
    ]
    calls, audits = [], []
    directory = tmp_path / "compile"

    def run(**kwargs):
        assert SOURCE not in kwargs["prompt"]
        if kwargs.get("system_prompt") == claim_compiler.COVERAGE_SYSTEM:
            audits.append(kwargs["directory"])
            clause = json.loads(kwargs["prompt"])["defining_clauses"][0]
            approved = clause.get("review_interval") == interval
            issue = {"source_field": "description", "quote": "without first copying it",
                     "reason": issues[min(len(audits) - 1, 1)]}
            return json.dumps({"approved": approved, "issues": [] if approved else [issue]})
        calls.append(kwargs)
        revised = deepcopy(draft)
        if len(calls) > 1:
            # Each rejection repairs the draft before any final contract is published.
            assert issues[len(calls) - 2] in kwargs["prompt"]
            assert not (directory / "contract.json").exists()
            revised["defining_clauses"][0].update(
                required_review_scope="bounded_interval", review_interval=deepcopy(interval),
            )
        if len(calls) == 2:
            revised["defining_clauses"][0]["review_interval"]["start"] = {
                "kind": "event", "predicate": "The initial database inspection",
                "position": "before",
            }
        return json.dumps(revised)

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    contract = claim_compiler.compile_claim_contract(
        description=description, mechanism=mechanism, source=SOURCE, model="test",
        directory=directory,
    )
    assert len(calls) == len(audits) == len(set(audits)) == 3
    assert contract["defining_clauses"][0]["review_interval"] == interval
    assert claim_compiler.validate_compiled_contract(
        contract, description=description, mechanism=mechanism, source=SOURCE,
    ) == contract


# One persistent compilation budget shared across review allocations.


def test_successful_compilation_rebinds_source_without_repeating_extraction(tmp_path, monkeypatch):
    calls = _stub_compiler(monkeypatch)
    first = _compile_cached(tmp_path, "first")
    second = _compile_cached(tmp_path, "second", source="def other(x): return False")
    assert len(calls) == 2
    assert first["defining_clauses"] == second["defining_clauses"]
    assert first["claim_hash"] != second["claim_hash"]
    assert all("def f" not in call["prompt"] and "def other" not in call["prompt"]
               for call in calls)
    with pytest.raises(ValueError, match="does not match"):
        claim_compiler.validate_compiled_contract(
            first, description=DESCRIPTION, mechanism=MECHANISM, source="changed",
        )


@pytest.mark.parametrize("change", [
    {"model": "another-model"}, {"harness": "codex"},
    {"description": DESCRIPTION + " A changed claim."}, {"behavior_strictness": "behavioral"},
], ids=["model", "harness", "claim", "policy"])
def test_actual_claim_backend_or_policy_change_has_its_own_attempt(tmp_path, monkeypatch, change):
    calls = _stub_compiler(monkeypatch, {})
    with pytest.raises(ClaimCompilationError):
        _compile_cached(tmp_path, "first")
    with pytest.raises(ClaimCompilationError):
        _compile_cached(tmp_path, "changed", **change)
    assert len(calls) == 6


def test_explicit_recovery_is_shared_and_bounded_even_when_ids_or_sources_change(
    tmp_path, monkeypatch,
):
    calls = _stub_compiler(monkeypatch, {})
    for index in range(4):
        with pytest.raises(ClaimCompilationError) as error:
            _compile_cached(tmp_path, str(index), source=f"source-{index}",
                            retry_failed=index > 0)
        assert error.value.receipt["attempt"] == min(index, 1)
    assert len(calls) == 6
    assert error.value.failure["explicit_retry_used"] is True


def test_concurrent_allocations_do_not_multiply_the_failed_compiler_budget(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(timeout=5)
        return '{}'

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_compile_cached, tmp_path, "first")
        assert entered.wait(timeout=5)
        second = pool.submit(_compile_cached, tmp_path, "second")
        release.set()
        for future in (first, second):
            with pytest.raises(ClaimCompilationError):
                future.result(timeout=5)
    assert len(calls) == 3


def test_cancelled_cache_waiter_does_not_dispatch_or_consume_recovery(tmp_path, monkeypatch):
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    token = Cancellation()
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(timeout=5)
        return '{}'

    original_lock = claim_compiler._compiler_cache_lock

    @contextmanager
    def observed_lock(path):
        if entered.is_set():
            waiting.set()
        with original_lock(path):
            yield

    def waiter():
        with cancellation_scope(token):
            return _compile_cached(tmp_path, "second", retry_failed=True)

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    monkeypatch.setattr(claim_compiler, "_compiler_cache_lock", observed_lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_compile_cached, tmp_path, "first")
        assert entered.wait(timeout=5)
        second = pool.submit(waiter)
        assert waiting.wait(timeout=5)
        token.cancel()
        release.set()
        with pytest.raises(ClaimCompilationError):
            first.result(timeout=5)
        with pytest.raises(asyncio.CancelledError):
            second.result(timeout=5)
    assert len(calls) == 3
    saved = json.loads(next((tmp_path / "shared").glob("*/checkpoint.json")).read_text())
    assert saved["attempt"] == 0


@pytest.mark.parametrize("failure", [
    asyncio.CancelledError(), TimeoutError(),
    RuntimeError("incomplete claim compilation: error_during_execution/None: "
                 "API Error: 401 authentication_error"),
    RuntimeError("incomplete claim compilation: success/max_tokens: reply truncated"),
], ids=["cancelled", "timeout", "authentication", "truncated"])
def test_infrastructure_failure_is_not_settled_and_resumes_saved_draft(
    tmp_path, monkeypatch, failure,
):
    calls = []

    def run(**kwargs):
        calls.append(kwargs["system_prompt"])
        if len(calls) == 2:
            raise failure
        return _answer(kwargs)

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    with pytest.raises(type(failure)):
        _compile_cached(tmp_path, "interrupted")
    saved = json.loads(next((tmp_path / "shared").glob("*/checkpoint.json")).read_text())
    assert saved["status"] == "running" and saved["attempt"] == 0
    # An ordinary request resumes the same attempt; no explicit recovery is needed or spent.
    contract = _compile_cached(tmp_path, "resumed")
    assert contract["compilation"]["coverage_review"] == APPROVED
    assert calls == [claim_compiler.COMPILER_SYSTEM] + [claim_compiler.COVERAGE_SYSTEM] * 2
    saved = json.loads(next((tmp_path / "shared").glob("*/checkpoint.json")).read_text())
    assert saved["status"] == "complete" and saved["attempt"] == 0


def test_backend_refusal_still_settles_the_shared_stage(tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("incomplete claim compilation: success/refusal: API Error: "
                           "the safeguards flagged this message")

    monkeypatch.setattr(claim_compiler, "_run_compiler", run)
    for index, retry in enumerate((False, True)):
        with pytest.raises(ClaimCompilationError) as error:
            _compile_cached(tmp_path, str(index), retry_failed=retry)
        assert error.value.failure["kind"] == "backend_refusal"
    assert len(calls) == 1
    assert error.value.failure["retry_policy"] == "requires_backend_or_configuration_change"


def test_shared_blocker_reports_its_stage_outcome_after_the_explicit_retry():
    first = {**processing_failure("Every draft was rejected.", kind="claim_decomposition_error"),
             "stage": "claim_compilation", "cache_key": "shared", "explicit_retry_used": False}
    spent = {**first, "explicit_retry_used": True,
             "retry_policy": "requires_backend_or_configuration_change"}
    # A partial retry replaces some rows; the others retain the first attempt.
    rows = [{"item_id": key, "verdict": "error", "processing_failure": failure}
            for key, failure in (("a", first), ("b", spent), ("c", first))]
    [blocker] = review_failures(rows)
    assert blocker["explicit_retry_used"] is True
    assert blocker["retry_policy"] == "requires_backend_or_configuration_change"
    assert blocker["item_ids"] == ["a", "b", "c"] and blocker["affected_items"] == 3


# Compilation precedes evidence review and fails without any weaker judging path.


def _judge(tmp_path, **kwargs):
    return claude_judge.judge_behavior_with_claude(
        name="claim", **CLAIM,
        items=[{"_item_id": key, "response_text": "Service still fails."} for key in ("a", "b")],
        assigned_labels=[True, True], cwd=tmp_path, model="test", **kwargs,
    )


def _unknown_reply(review):
    contract = review.snapshot["claim_contract"]
    return json.dumps({"verdicts": [{
        "item_id": review.snapshot["item_id"], "snapshot_id": review.snapshot["snapshot_id"],
        "claim_contract_id": contract["contract_id"], "scope": "local",
        "claim_assessments": [
            {"requirement_id": requirement["id"], "status": "unknown",
             **({"defining_requirement_ids": [contract["defining_clauses"][0]["id"]]}
                if requirement["kind"] in {"claim", "review_check"} else {}),
             "rationale": "The final service state is unavailable.",
             "citations": [], "counterevidence": []}
            for requirement in contract["requirements"]
        ],
    }]})


def test_one_frozen_contract_precedes_every_item_and_retains_stable_claim_hash(
    tmp_path, monkeypatch,
):
    events = []
    contract = _approved_contract()

    def compile_claim(**kwargs):
        assert kwargs["cache_dir"] == tmp_path / "behavior_judge" / "compilations"
        assert kwargs["retry_failed"] is False
        assert not list(tmp_path.rglob("evidence.json"))
        events.append("compile")
        directory = kwargs["directory"]
        (directory / "coverage-0").mkdir(parents=True)
        (directory / "session-0.json").write_text(json.dumps({"cost_usd": 0.1}))
        (directory / "coverage-0/session-0.json").write_text(json.dumps({"cost_usd": 0.02}))
        return contract

    def run(*, review, **kwargs):
        events.append(review.snapshot["item_id"])
        assert review.snapshot["claim_contract"] == contract
        return _unknown_reply(review)

    monkeypatch.setattr(claude_judge, "compile_claim_contract", compile_claim)
    monkeypatch.setattr(claude_judge, "_run_review", run)
    result = _judge(tmp_path, workers=4)
    # Compilation precedes all reviews; parallel item reviews may start in either order.
    assert events[0] == "compile"
    assert sorted(events[1:]) == ["a", "b"]
    assert result["n_uncertain"] == 2 and result["n_errors"] == 0
    assert result["claim_hash"] == claim_hash(**CLAIM) != contract["contract_id"]
    assert result["claim_contract"] == contract
    sessions = result["claim_compilation"]["sessions"]
    assert sum(session["cost_usd"] for session in sessions) == pytest.approx(0.12)


def test_compilation_failure_is_unscored_processing_error_for_every_item(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("compiler exhausted its budget")

    monkeypatch.setattr(claude_judge, "compile_claim_contract", fail)
    monkeypatch.setattr(claude_judge, "_run_review", lambda **kwargs: pytest.fail(
        "a failed compilation must never start a review"))
    result = _judge(tmp_path)
    assert result["n_errors"] == result["n_judged"] == 2
    assert result["n_yes"] == result["n_no"] == result["n_uncertain"] == 0
    assert result["behavior_purity"] is result["behavior_specificity"] is None
    assert "compiler exhausted" in result["claim_compilation"]["error"]


def test_frozen_contract_can_be_reused_but_never_for_a_different_claim(tmp_path, monkeypatch):
    reviewed = []

    def run(*, review, **kwargs):
        reviewed.append(review.snapshot["claim_contract"]["contract_id"])
        return _unknown_reply(review)

    monkeypatch.setattr(claude_judge, "compile_claim_contract", lambda **kwargs: pytest.fail(
        "a supplied frozen contract must not trigger another compilation"))
    monkeypatch.setattr(claude_judge, "_run_review", run)
    contract = _approved_contract()
    result = _judge(tmp_path, compiled_contract=contract)
    assert reviewed == [contract["contract_id"]] * 2
    assert result["claim_compilation"]["reused"] is True
    mismatched = {**contract, "description": "A different claim."}
    with pytest.raises(ValueError, match="does not match"):
        _judge(tmp_path, compiled_contract=mismatched)
    assert len(reviewed) == 2


def test_three_review_requests_share_one_failed_compilation_despite_source_and_cohort_changes(
    tmp_path, monkeypatch,
):
    calls = _stub_compiler(monkeypatch, {})
    monkeypatch.setattr(claude_judge, "_run_review", lambda **kwargs: pytest.fail(
        "An uncompiled claim must never receive an item review"))
    results = []
    for index, size in enumerate((1, 6, 2)):
        results.append(claude_judge.judge_behavior_with_claude(
            name=f"renamed_{index}", description=DESCRIPTION, mechanism=MECHANISM,
            source=f"def f(x): return {index}",
            items=[{"_item_id": str(item), "response_text": "A retained draft."}
                   for item in range(size)], assigned_labels=[True] * size,
            cwd=tmp_path, model="test", workers=1,
            _recovery_id=f"automatic-final-allocation-{index}",
        ))
    assert len(calls) == 3  # One bounded compiler attempt, not three attempts per request.
    assert len({r["claim_compilation"]["cache_key"] for r in results}) == 1
    assert [r["claim_compilation"]["reused"] for r in results] == [False, True, True]
    assert len({r["claim_hash"] for r in results}) == 3  # Reviews still bind source.
    for result, size in zip(results, (1, 6, 2), strict=True):
        assert result["n_errors"] == size
        assert result["n_completed"] == result["n_no"] == 0
        failures = review_failures(result["verdicts"])
        assert len(failures) == 1
        assert failures[0]["affected_items"] == size
        assert failures[0]["kind"] == "claim_decomposition_error"
        assert failures[0]["detector_rejected"] is False
