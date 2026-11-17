# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeated prose and retries do not become substantive recovery rounds."""

import json
from copy import deepcopy

import pytest

from autofa.mcp import tools_library
from autofa.session.polishing_progress import (
    _claim,
    polishing_response_assessment,
    recovery_work,
)
from autofa.session.round_stagnation import complete_discovery_round, round_stagnation_status
from autofa.session.round_validation import active_round_refinement

from support.discovery import close, detector
from support.discovery import rt as rt
from support.families import batch_runtime as batch_runtime
from support.families import polishing_round as polishing_round


def smoke(rt, identifier="probe", *, source="def detect(item): return False", state="complete",
          errors=0, fn=None, item_ids=("one",), value=False, row_metadata=None):
    from autofa.session.repair_workflow import _version

    directory = rt.run.run_dir() / "operations"
    directory.mkdir(exist_ok=True)
    (directory / f"{identifier}.json").write_text(json.dumps({
        "tool": "smoke_test_feature", "state": state,
        "args": {"name": fn.name} if fn else {"source": source},
        "result": {"ok": True, "errors": errors,
                   **({"name": fn.name, "version_hash": _version(fn.to_dict()),
                       "claim_hash": _claim(fn)} if fn else {}),
                   "results": [{"item_id": item, "status": "value", "value": value,
                                **(row_metadata or {})}
                               for item in item_ids]},
    }))


def extra_round(polishing_round):
    rt, calls = polishing_round
    tools_library.advance_round()
    result = tools_library.advance_round(validation_response="Inspect the validated findings.")
    assert result["advanced"]
    assert rt.run.round == rt.run.max_rounds
    path = rt.run.run_dir() / "round_validation.json"
    receipt = json.loads(path.read_text())
    controls = receipt["refinement"]["controls"]
    controls[rt.library.functions[0].name]["item_ids"] = ["one"]
    path.write_text(json.dumps(receipt))
    return rt, calls


def test_unchanged_closure_returns_action_without_scan_or_round(polishing_round):
    rt, calls = extra_round(polishing_round)
    before = len(calls)
    for text in ("Retain all.", "Same conclusion, no new work."):
        result = tools_library.advance_round(validation_response=text)
        assert not result["advanced"]
        assert not result["recovery_attempt"]["substantive"]
    assert len(calls) == before and rt.run.round == rt.run.max_rounds


def test_new_diagnostic_can_close_once_repeated_test_cannot(polishing_round):
    rt, calls = extra_round(polishing_round)
    smoke(rt)
    first = tools_library.advance_round(validation_response="The tested refinement failed.")
    assert first["advanced"]
    before = len(calls)
    smoke(rt, "retry", source="# same function\ndef detect(item):\n    return False\n")
    repeated = tools_library.advance_round(validation_response="The refinement still fails.")
    assert not repeated["advanced"]
    assert len(calls) == before


def test_smoke_annotations_cannot_create_work_or_exhaustion(polishing_round):
    # Annotations are stripped from rows before either identity kind is hashed.
    rt, _ = extra_round(polishing_round)
    fn = rt.library.functions[0]
    smoke(rt, source=fn.source)
    phase = active_round_refinement(rt)
    baseline = recovery_work(rt, phase["frozen_names"], phase.get("controls"))
    phase["recovery_work"] = baseline
    phase["recovery_initial_work"] = list(baseline["work"])
    _defer_current(rt, fn, ["one"])
    smoke(rt, "annotated_retry", source=fn.source, row_metadata={
        "expected_output": False, "matches_expected": True, "expected_abstention": True,
        "matches_saved_bucket": True, "display_note": "Same cached measurement",
    })
    assessed = polishing_response_assessment(rt, phase)
    assert not assessed["substantive"] and not assessed["can_close"]
    assert not assessed["exhausted"] and not assessed["limitations"]
    assert set(assessed["snapshot"]["tests"]) == set(baseline["tests"])


@pytest.mark.parametrize("change,named", [("prediction", False), ("revision", True)],
                         ids=["new-prediction", "named-revision"])
def test_new_smoke_execution_evidence_remains_substantive(polishing_round, change, named):
    rt, _ = extra_round(polishing_round)
    fn = rt.library.functions[0]
    smoke(rt, fn=fn if named else None, source=fn.source)
    phase = active_round_refinement(rt)
    baseline = recovery_work(rt, phase["frozen_names"], phase.get("controls"))
    phase["recovery_work"] = baseline
    phase["recovery_initial_work"] = list(baseline["work"])
    if change == "revision":
        fn.source += "\ndef new_distinction(item): return bool(item.get('missing'))\n"
    smoke(rt, "new_measurement", fn=fn if named else None, source=fn.source,
          value=change == "prediction")
    _defer_current(rt, fn, ["one"])
    assessed = polishing_response_assessment(rt, phase)
    assert assessed["substantive"] and assessed["exhausted"]
    assert len(set(assessed["snapshot"]["tests"]) - set(baseline["tests"])) == 1


@pytest.mark.parametrize("operation", [
    {"state": "running"}, {"errors": 1}, {"item_ids": ("not-a-retained-control",)},
], ids=["in-flight", "failed", "unrelated-item"])
def test_unfinished_failed_or_unrelated_smoke_is_not_completed_evidence(
    polishing_round, operation,
):
    rt, _ = extra_round(polishing_round)
    smoke(rt, **operation)
    assert not polishing_response_assessment(rt, active_round_refinement(rt))["can_close"]


def test_direct_grounded_deferral_needs_no_repair_budget_or_causal_rejection(
    polishing_round,
):
    from autofa.mcp.tools_repair import defer_function_repair
    from autofa.session.repair_workflow import get_repair

    rt, _ = extra_round(polishing_round)
    fn = rt.library.functions[0]
    fn.scores.update(evaluated=True, evaluation_claim_hash=_claim(fn),
                     evaluation_item_ids=["one"])
    response = tools_library.get_function_feedback(fn.name)
    finding = next(row for row in response["recovery_findings"] if row["status"] == "measured")
    result = defer_function_repair(
        fn.name, "The available traces omit the distinguishing fact; no supported refinement.",
        finding_ids=[finding["id"]], evidence_ids=["one"],
    )
    assert result["ok"] and result["limitation"]["reason"]
    assert not get_repair(rt.run.run_dir(), fn.name).get("history")
    # A reference plus free-text conclusion is not an exhausted investigation.
    phase = active_round_refinement(rt)
    snapshot = recovery_work(rt, phase["frozen_names"], phase.get("controls"))
    phase["recovery_work"] = snapshot
    phase["recovery_initial_work"] = sorted(snapshot["work"])
    assert not polishing_response_assessment(rt, phase)["exhausted"]
    smoke(rt, fn=fn)
    assert polishing_response_assessment(rt, phase)["exhausted"]
    # Saved measurements changing makes the decision stale instead of stopping.
    fn.scores["ppv_at_pi0"] = .8
    assert not polishing_response_assessment(rt, active_round_refinement(rt))["exhausted"]


def test_unknown_evidence_rejected_for_direct_deferral(polishing_round):
    from autofa.mcp.tools_repair import defer_function_repair

    rt, _ = extra_round(polishing_round)
    result = defer_function_repair(rt.library.functions[0].name, "Unavailable evidence",
                                   finding_ids=["invented"], evidence_ids=["invented"])
    assert not result["ok"]


def test_current_evidence_backed_deferrals_allow_explicit_exhaustion(polishing_round):
    rt, _ = extra_round(polishing_round)
    repairs = {}
    for fn in rt.library.functions:
        repairs[fn.name] = {"history": [{
            "phase": "deferred", "before_claim_hash": _claim(fn),
            "before": {"scores": deepcopy(fn.scores)},
            "finding_ids": ["missing_source_evidence"], "evidence_ids": ["one"],
            "deferral_reason": "The required attachment was not retained in the source.",
        }]}
    path = rt.run.run_dir() / "detector_repairs.json"
    path.write_text(json.dumps({"repairs": repairs}))
    for fn in rt.library.functions:
        smoke(rt, fn.name, fn=fn)
    decision = polishing_response_assessment(rt, active_round_refinement(rt))
    assert decision["can_close"] and decision["exhausted"]
    assert decision["limitations"]
    # Changing prose cannot create another investigation; changing source scores
    # invalidates an old limitation, which must be reconsidered with current facts.
    phase = active_round_refinement(rt)
    baseline = recovery_work(rt, phase["frozen_names"], phase.get("controls"))
    repairs[next(iter(repairs))]["history"][-1]["deferral_reason"] = "Same missing attachment."
    path.write_text(json.dumps({"repairs": repairs}))
    refreshed = recovery_work(rt, phase["frozen_names"], phase.get("controls"))
    assert refreshed["work"] == baseline["work"]
    rt.library.functions[0].scores["ppv_at_pi0"] = .8
    assert not polishing_response_assessment(rt, active_round_refinement(rt))["exhausted"]


def _defer_current(rt, fn, evidence_ids):
    path = rt.run.run_dir() / "detector_repairs.json"
    path.write_text(json.dumps({"repairs": {fn.name: {"history": [{
        "phase": "deferred", "before_claim_hash": _claim(fn),
        "before": {"scores": deepcopy(fn.scores)},
        "finding_ids": ["measured_limitation"], "evidence_ids": evidence_ids,
        "deferral_reason": "The tested refinement did not repair the current candidate.",
    }]}}}))


def test_other_detector_smoke_on_same_item_cannot_exhaust_core_repair(polishing_round):
    from autofa.core.library import FeatureFunction

    rt, _ = extra_round(polishing_round)
    core = rt.library.functions[0]
    core.name = "core_deliverable_left_undelivered"
    artifact = FeatureFunction(name="artifact_left_in_hidden_reasoning",
                               description="Artifact stays in private reasoning",
                               source="def detect(item): return False")
    rt.library.functions.append(artifact)
    phase = active_round_refinement(rt)
    phase["frozen_names"] = [core.name]
    shared = "glm-5.3-flash|lmarena_v3|1286|0"
    phase["controls"] = {core.name: {"item_ids": [shared]}}
    _defer_current(rt, core, [shared])
    smoke(rt, fn=artifact, item_ids=("1859", shared, "1729"))
    assessment = polishing_response_assessment(rt, phase)
    assert assessment["substantive"]  # Completed experiment, not proof for another detector.
    assert not assessment["exhausted"] and not assessment["limitations"]
    # The same outcomes from a different detector are a distinct execution and
    # may support only the tested detector's disposition.
    prior_work = set(assessment["snapshot"]["work"])
    smoke(rt, "core", fn=core, item_ids=("1859", shared, "1729"))
    tested = polishing_response_assessment(rt, phase)
    assert tested["exhausted"]
    new_work = set(tested["snapshot"]["work"]) - prior_work
    assert new_work
    assert set(tested["limitations"][core.name]["tested_work_ids"]) <= new_work


# A current named smoke supporting a deferral is covered by the exhaustion tests above.
@pytest.mark.parametrize("named,change", [
    (False, None), (False, "source"), (False, "deps"), (True, "source"),
], ids=["anonymous-current", "anonymous-changed", "anonymous-frozen-deps", "named-changed"])
def test_deferral_smoke_must_match_current_candidate(polishing_round, named, change):
    rt, _ = extra_round(polishing_round)
    fn = rt.library.functions[0]
    smoke(rt, source=fn.source, fn=fn if named else None)
    if change == "source":
        fn.source += "\n# Changed revision\n" if named else "\ndef unused(): return False\n"
    elif change == "deps":  # Anonymous smoke did not run the frozen dependency snapshot.
        fn.meta["deps"] = {"helper": "def helper(item): return False"}
    _defer_current(rt, fn, ["one"])
    assessment = polishing_response_assessment(rt, active_round_refinement(rt))
    assert assessment["exhausted"] is (change is None)


@pytest.mark.parametrize("current", [False, True])
def test_regression_checks_support_only_their_current_claim(polishing_round, current):
    rt, _ = extra_round(polishing_round)
    fn = rt.library.functions[0]
    directory = rt.run.run_dir() / "polishing_evaluations"
    directory.mkdir(exist_ok=True)
    (directory / "checks.json").write_text(json.dumps({
        "name": fn.name, "run_id": rt.run.run_id, "complete": True, "n": 1,
        "claim_hash": _claim(fn) if current else "older-revision", "errors": 0,
        "results": [{"item_id": "one", "status": "value", "value": False}],
    }))
    _defer_current(rt, fn, ["one"])
    assessed = polishing_response_assessment(rt, active_round_refinement(rt))
    assert assessed["substantive"]
    assert assessed["exhausted"] is current


def test_unreferenced_deferral_cannot_manufacture_exhaustion(polishing_round):
    rt, _ = extra_round(polishing_round)
    fn = rt.library.functions[0]
    (rt.run.run_dir() / "detector_repairs.json").write_text(json.dumps({
        "repairs": {fn.name: {"history": [{"phase": "deferred",
                                            "deferral_reason": "Nothing else can work."}]}},
    }))
    assert not polishing_response_assessment(rt, active_round_refinement(rt))["can_close"]


def test_explicit_source_unavailability_does_not_require_pointless_test(
    polishing_round, monkeypatch,
):
    rt, _ = extra_round(polishing_round)
    fn = rt.library.functions[0]
    (rt.run.run_dir() / "detector_repairs.json").write_text(json.dumps({
        "repairs": {fn.name: {"history": [{
            "phase": "deferred", "before_claim_hash": _claim(fn),
            "before": {"scores": deepcopy(fn.scores)}, "finding_ids": ["missing_source"],
            "evidence_ids": ["one"], "deferral_reason": "The required attachment is unavailable.",
        }]}},
    }))
    monkeypatch.setattr("autofa.metrics.repair_feedback.build_repair_feedback", lambda fn: {
        "investigations": [{"finding_id": "missing_source", "action": "defer_claim"}],
    })
    assessed = polishing_response_assessment(rt, active_round_refinement(rt))
    assert assessed["exhausted"]
    assert assessed["limitations"][fn.name]["unavailable_finding_ids"] == ["missing_source"]
    assert assessed["limitations"][fn.name]["tested_work_ids"] == []


def test_round_patience_deduplicates_substantive_work_ids(rt):
    close(rt)
    validation = {"ok": True, "fraction": 1., "refinement": {
        "mode": "polishing", "recovery_attempt": {"substantive": True, "work_ids": ["test1"]},
    }}
    rt.run.round += 1
    assert complete_discovery_round(rt, validation)["rounds_without_improve"] == 1
    rt.run.round += 1
    repeated = complete_discovery_round(rt, validation)
    assert not repeated["counted"] and repeated["rounds_without_improve"] == 1
    assert repeated["status"] == "no_new_recovery_work"


def test_exhausted_recovery_honors_stall_option_and_planned_rounds(rt):
    validation = {"ok": True, "fraction": 1., "refinement": {
        "mode": "polishing", "recovery_attempt": {
            "exhausted": True, "limitations": {"detector": {"reason": "Missing evidence"}},
        },
    }}
    assert not complete_discovery_round(rt, validation)["should_stop"]
    rt.run.round += 1
    result = complete_discovery_round(rt, validation)
    assert result["should_stop"] and result["recovery_exhausted"]
    rt.config.discovery_terminate_if_stalled = False
    assert not complete_discovery_round(rt, validation)["should_stop"]


def test_verified_all_good_library_with_too_few_frozen_candidates_can_close(rt):
    rt.run.min_failure_modes = 3
    rt.library.functions = [detector(rt, "only_keeper")]
    rt.run.round = rt.run.max_rounds - 1
    phase = {"mode": "polishing", "frozen_names": ["only_keeper"],
             "recovery_work": recovery_work(rt, ["only_keeper"])}
    assessment = polishing_response_assessment(rt, phase)
    assert assessment["capacity_exhausted"] and not assessment["substantive"]
    validation = {"ok": True, "fraction": 1., "refinement": {
        **phase, "recovery_attempt": assessment,
    }}
    assert not complete_discovery_round(rt, validation)["should_stop"]
    rt.run.round += 1
    result = complete_discovery_round(rt, validation)
    assert result["should_stop"] and not result["counted"]
    assert result["quality"]["good"] == 1 and rt.run.min_failure_modes == 3


def test_capacity_cannot_stop_until_full_validation_and_reviews_complete(rt):
    rt.run.min_failure_modes = 3
    fn = detector(rt, "pending_keeper")
    rt.library.functions = [fn]
    fn.scores["sampled"] = True
    rt.run.round = rt.run.max_rounds
    phase = {"mode": "polishing", "frozen_names": [fn.name],
             "recovery_work": recovery_work(rt, [fn.name])}
    assessment = polishing_response_assessment(rt, phase)
    result = complete_discovery_round(rt, {"ok": True, "fraction": 1., "refinement": {
        **phase, "recovery_attempt": assessment,
    }})
    assert not result["should_stop"] and not result["counted"]
    assert result["status"] == "validation_incomplete"


def test_exhaustion_is_invalidated_by_changed_current_evidence_or_target(rt):
    rt.run.min_failure_modes = 3
    rt.library.functions = [detector(rt, "keeper")]
    close(rt)
    rt.run.round += 1
    validation = {"ok": True, "fraction": 1., "refinement": {
        "mode": "polishing", "recovery_attempt": {
            "exhausted": True, "limitations": {"keeper": {"reason": "Tested limitation"}},
        },
    }}
    assert complete_discovery_round(rt, validation)["should_stop"]
    fn = rt.library.functions[0]
    original = deepcopy(fn.scores)
    fn.scores["ppv_at_pi0"] = .95
    assert not round_stagnation_status(rt)["should_stop"]
    fn.scores = original
    assert round_stagnation_status(rt)["should_stop"]
    fn.scores["evaluation_stale"] = True
    assert not round_stagnation_status(rt)["should_stop"]
    fn.scores = deepcopy(original)
    fn.source += "\n# Changed implementation\n"
    assert not round_stagnation_status(rt)["should_stop"]
