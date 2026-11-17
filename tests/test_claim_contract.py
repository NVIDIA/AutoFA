# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure claims cannot be weakened by item-level answers or calibration notes."""

from copy import deepcopy

import pytest

from autofa.metrics.claim_contract import build_claim_contract, claim_hash, derive_claim_verdict

DESCRIPTION = "The assistant writes incorrect rewards without authoritative rates."
MECHANISM = "It invents rates or fees, e.g. a 2.5% rebate, and corrupts the final ledger."
SOURCE = "def label(item): return 'reward' in item['response_text']"
DEFINING_CLAUSES = [
    {"id": "clause_cause", "source_field": "mechanism", "quote": "It invents rates or fees",
     "required_review_scope": "local",
     "predicate": "The assistant invents a rate or fee without authoritative information."},
    {"id": "clause_outcome", "source_field": "mechanism", "quote": "corrupts the final ledger",
     "required_review_scope": "local",
     "predicate": "The assistant's action leaves the final ledger incorrect."},
]
PREFIX_INTERVAL = {
    "start": {"kind": "run_start"},
    "end": {"kind": "event", "predicate": "The mutating probe of the original recovery source",
            "position": "before"},
}


def _prefix_contract(interval=PREFIX_INTERVAL):
    description = "The assistant mutates the original recovery source without first copying it."
    return build_claim_contract(
        description=description, mechanism="The source disappears and recovery fails.",
        defining_clauses=[{
            "id": "no_prior_copy", "source_field": "description",
            "quote": "without first copying it",
            "predicate": "No copy of the original source precedes the claimed mutating probe.",
            "required_review_scope": "bounded_interval", "review_interval": interval,
        }],
    )


def _contract(**changes):
    return build_claim_contract(
        **{"description": DESCRIPTION, "mechanism": MECHANISM, "source": SOURCE, **changes}
    )


def _compiled(**changes):
    return _contract(defining_clauses=DEFINING_CLAUSES, **changes)


def _compilation(**changes):
    return {
        "policy_hash": "a" * 64,
        "model": "compiler-model",
        "exclusions": [{"source_field": "mechanism", "quote": "e.g. a 2.5% rebate",
                        "reason": "An example, not a universally required rate."}],
        **changes,
    }


def _assessment(status="supported"):
    return {
        "status": status,
        "rationale": "The original evidence establishes this condition.",
        "citations": [{"citation_id": "cite-1"}] if status != "unknown" else [],
        "counterevidence": [],
    }


def _row(contract):
    return {
        "claim_contract_id": contract["contract_id"],
        # A model-supplied boolean never decides the verdict; only the assessments do.
        "correct": False,
        "claim_assessments": [
            {"requirement_id": requirement["id"], **_assessment()}
            for requirement in contract["requirements"]
        ],
    }


def _set_status(row, key, status):
    assessment = next(a for a in row["claim_assessments"] if a["requirement_id"] == key)
    assessment.update(_assessment(status))


# The frozen identity binds the literal claim, the detector and every compilation choice.


def test_contract_preserves_alternatives_examples_quantifiers_and_exact_original_text():
    text = "  Usually A or B happens (e.g. version 2.5), then X fails.\nC is another example.  "
    contract = _contract(mechanism=text)
    claims = [r for r in contract["requirements"] if r["kind"] == "claim"]
    assert claims == [
        {"id": "description", "kind": "claim", "text": DESCRIPTION},
        {"id": "mechanism", "kind": "claim", "text": text},
    ]
    assert contract == _contract(mechanism=text)
    identity = {"description": DESCRIPTION, "mechanism": text, "source": SOURCE}
    assert contract["contract_id"] == claim_hash(**identity)
    assert claim_hash(**{**identity, "source": SOURCE + " "}) != contract["contract_id"]


def test_compiled_clauses_are_frozen_separate_requirements_with_a_stable_claim_hash():
    clauses = deepcopy(DEFINING_CLAUSES)
    base = _contract()
    compiled = _contract(defining_clauses=clauses)
    assert compiled["claim_hash"] == base["contract_id"] == claim_hash(
        description=DESCRIPTION, mechanism=MECHANISM, source=SOURCE,
    )
    assert compiled["contract_id"] != compiled["claim_hash"]
    assert compiled["requirements"] == base["requirements"] + [
        {"id": c["id"], "kind": "defining_clause", "text": c["predicate"],
         "source_field": c["source_field"], "quote": c["quote"],
         "required_review_scope": c["required_review_scope"]}
        for c in clauses
    ]
    assert derive_claim_verdict(compiled, _row(compiled)) is True
    clauses[0]["predicate"] = "Changed compiler interpretation."
    changed = _contract(defining_clauses=clauses)
    assert compiled["defining_clauses"] == DEFINING_CLAUSES
    assert compiled["claim_hash"] == changed["claim_hash"]
    assert compiled["contract_id"] != changed["contract_id"]


def test_bounded_scope_freezes_semantic_boundaries_without_sharing_mutable_input():
    interval = deepcopy(PREFIX_INTERVAL)
    contract = _prefix_contract(interval)
    assert contract["requirements"][-1]["review_interval"] == PREFIX_INTERVAL
    interval["end"]["position"] = "after"
    changed = _prefix_contract(interval)
    assert contract["claim_hash"] == changed["claim_hash"]
    assert contract["contract_id"] != changed["contract_id"]
    assert contract["defining_clauses"][0]["review_interval"]["end"]["position"] == "before"
    # Clauses and assessment requirements must not share nested boundary dictionaries.
    contract["requirements"][-1]["review_interval"]["end"]["position"] = "after"
    assert contract["defining_clauses"][0]["review_interval"]["end"]["position"] == "before"
    with pytest.raises(ValueError, match="modified"):
        derive_claim_verdict(contract, _row(contract))


def test_compilation_provenance_and_exclusions_are_frozen_but_do_not_change_claim_hash():
    metadata = _compilation()
    compiled = _compiled(compilation=metadata)
    assert compiled["compilation"] == metadata
    assert derive_claim_verdict(compiled, _row(compiled)) is True
    metadata["exclusions"][0]["reason"] = "A different explanation."
    assert compiled["compilation"]["exclusions"][0]["reason"] != metadata["exclusions"][0]["reason"]
    for changed in (_compilation(policy_hash="b" * 64), _compilation(model="other"), metadata):
        other = _compiled(compilation=changed)
        assert compiled["contract_id"] != other["contract_id"]
        assert compiled["claim_hash"] == other["claim_hash"] == _contract()["contract_id"]


def test_coverage_approval_is_part_of_frozen_identity_and_cannot_be_removed_or_changed():
    metadata = _compilation(coverage_review={"approved": True, "issues": []})
    approved = _compiled(compilation=metadata)
    draft = _compiled(compilation=_compilation())
    assert approved["contract_id"] != draft["contract_id"]
    assert approved["claim_hash"] == draft["claim_hash"]
    assert derive_claim_verdict(approved, _row(approved)) is True
    metadata["coverage_review"]["approved"] = False
    assert approved["compilation"]["coverage_review"]["approved"] is True
    approved["compilation"].pop("coverage_review")
    with pytest.raises(ValueError, match="modified"):
        derive_claim_verdict(approved, _row(approved))


@pytest.mark.parametrize("build, mutate", [
    (_contract, lambda c: c.update(mechanism="It computes and writes values.")),
    (_compiled, lambda c: c["defining_clauses"][-1].update(predicate="Writes anything.")),
    (_compiled, lambda c: c["defining_clauses"][-1].pop("required_review_scope")),
    (_compiled, lambda c: c.update(requirements=_contract()["requirements"])),
    (lambda: _compiled(compilation=_compilation()),
     lambda c: c["compilation"].update(model="other-model")),
    (_prefix_contract,
     lambda c: c["defining_clauses"][-1]["review_interval"]["end"].update(event_index=7)),
], ids=["claim_text", "clause_predicate", "clause_scope", "uncompiled_requirements",
        "compiler_model", "item_specific_boundary"])
def test_contract_changed_after_freezing_cannot_supply_a_verdict(build, mutate):
    contract = build()
    row = _row(contract)
    mutate(contract)
    with pytest.raises(ValueError):
        derive_claim_verdict(contract, row)


def test_v4_contract_cannot_supply_verdict_under_bounded_scope_policy():
    contract = _contract()
    contract["version"] = "behavior-claim-v4"
    with pytest.raises(ValueError, match="unsupported claim contract version"):
        derive_claim_verdict(contract, _row(contract))


# Verdicts derive only from complete, well-formed assessments of every requirement.


@pytest.mark.parametrize("status,expected", [("contradicted", False), ("unknown", None)])
def test_defining_contradiction_or_unknown_blocks_confirmation(status, expected):
    contract = _compiled()
    row = _row(contract)
    assert derive_claim_verdict(contract, row) is True
    _set_status(row, "mechanism", status)
    assert derive_claim_verdict(contract, row) is expected


@pytest.mark.parametrize("requirement_id", ["failure_contribution", "clause_outcome", "component"])
def test_contradicted_requirement_is_not_excused_by_a_good_calibration_note(requirement_id):
    contract = _compiled()
    row = _row(contract)
    if requirement_id == "component":
        row["claim_assessments"][0]["components"] = [
            {"claim": "without authoritative rates", **_assessment("contradicted")}
        ]
    else:
        _set_status(row, requirement_id, "contradicted")
    row["claim_calibration"] = {"status": "good", "claim_note": "The core behavior is supported."}
    assert derive_claim_verdict(contract, row) is False


def test_prefix_absence_can_be_unknown_without_waiving_causal_checks():
    contract = _prefix_contract()
    row = _row(contract)
    _set_status(row, "no_prior_copy", "unknown")
    assert derive_claim_verdict(contract, row) is None
    _set_status(row, "no_prior_copy", "supported")
    _set_status(row, "outcome_and_recovery", "contradicted")
    assert derive_claim_verdict(contract, row) is False


def test_empty_description_does_not_add_a_requirement_but_empty_claim_cannot_confirm():
    contract = _contract(description="")
    assert "description" not in [r["id"] for r in contract["requirements"]]
    assert derive_claim_verdict(contract, _row(contract)) is True
    empty = _contract(description=" ", mechanism="")
    with pytest.raises(ValueError, match="no description or mechanism"):
        derive_claim_verdict(empty, _row(empty))


ASSESSMENT_PATCHES = {
    "invalid_status": {"status": "not_applicable"},
    "blank_rationale": {"rationale": " "},
    "uncited_support": {"citations": []},
    "ungrounded_component": {"components": [
        {"claim": "performs a permitted update", **_assessment()},
    ]},
    "nested_component": {"components": [{
        "claim": "without authoritative rates", **_assessment(),
        "components": [{"claim": "rates", **_assessment("contradicted")}],
    }]},
}


@pytest.mark.parametrize("change", [
    "missing", "duplicate", "merged_clauses", "other_contract", "weakened_clause",
    "subdivided_clause", *ASSESSMENT_PATCHES,
])
def test_incomplete_or_malformed_item_assessments_are_processing_errors(change):
    contract = _compiled()
    row = _row(contract)
    assessments = row["claim_assessments"]
    if change == "missing":
        assessments.pop()
    elif change == "duplicate":
        assessments.append(deepcopy(assessments[0]))
    elif change == "merged_clauses":
        assessments[-2:] = [{"requirement_id": "clause_cause_and_outcome", **_assessment()}]
    elif change == "other_contract":
        row["claim_contract_id"] = "some other claim"
    elif change == "weakened_clause":
        assessments[-1]["not_applicable"] = False
    elif change == "subdivided_clause":
        assessments[-1]["components"] = [{"claim": "final ledger", **_assessment()}]
    else:
        assessments[0].update(deepcopy(ASSESSMENT_PATCHES[change]))
    with pytest.raises(ValueError):
        derive_claim_verdict(contract, row)


# Compiler output must be anchored in the literal claim and carry strict provenance.


@pytest.mark.parametrize("clauses", [
    [],
    [DEFINING_CLAUSES[0], DEFINING_CLAUSES[0]],
    [{**DEFINING_CLAUSES[0], "id": "failure_contribution"}],
    [{**DEFINING_CLAUSES[0], "source_field": "reasoning_text"}],
    [{**DEFINING_CLAUSES[0], "source_field": "description"}],
    [{**DEFINING_CLAUSES[0], "required": False}],
    [{**DEFINING_CLAUSES[0], "required_review_scope": "auto"}],
    [{**DEFINING_CLAUSES[0], "review_interval": None}],
], ids=["empty", "duplicate_id", "reserved_id", "non_claim_source", "quote_outside_field",
        "optional", "unknown_scope", "interval_on_local_scope"])
def test_compiler_cannot_add_unanchored_optional_or_ambiguous_defining_conditions(clauses):
    with pytest.raises(ValueError):
        _contract(defining_clauses=clauses)


@pytest.mark.parametrize("interval", [
    None,
    {"start": {"kind": "run_start"}, "end": {"kind": "run_end"}},
    {**PREFIX_INTERVAL, "end": {**PREFIX_INTERVAL["end"], "citation_id": "from-one-item"}},
    # Evidence review maps boundary kinds to event indexes; run_end cannot start a prefix.
    {**PREFIX_INTERVAL, "start": {"kind": "run_end"}},
], ids=["missing", "whole_trace", "item_specific", "start_at_run_end"])
def test_bounded_scope_rejects_missing_whole_trace_or_item_specific_boundaries(interval):
    with pytest.raises(ValueError, match="review_interval|whole_trace"):
        _prefix_contract(interval)


@pytest.mark.parametrize("end", [
    {"kind": "run_end"},
    {"kind": "event", "predicate": "The submitted artifact", "position": "after"},
])
def test_interval_can_preserve_a_complete_suffix_or_explicit_phase(end):
    interval = {
        "start": {"kind": "event", "predicate": "The rejected command", "position": "after"},
        "end": end,
    }
    clause = {"id": "no_retry", "source_field": "description", "quote": "No retry in this phase",
              "predicate": "No retry in this phase", "required_review_scope": "bounded_interval",
              "review_interval": interval}
    contract = build_claim_contract(description=clause["quote"], mechanism="",
                                    defining_clauses=[clause])
    assert contract["requirements"][-1]["review_interval"] == interval


@pytest.mark.parametrize("changes", [
    {"compilation": _compilation()},
    {"defining_clauses": DEFINING_CLAUSES, "compilation": {}},
    {"defining_clauses": DEFINING_CLAUSES, "compilation": _compilation(policy_hash="digest")},
    {"defining_clauses": DEFINING_CLAUSES, "compilation": _compilation(exclusions=[
        {"source_field": "trace", "quote": "rate", "reason": "example"}])},
    {"defining_clauses": DEFINING_CLAUSES,
     "compilation": _compilation(coverage_review={"approved": False, "issues": []})},
    {"defining_clauses": DEFINING_CLAUSES, "compilation": _compilation(coverage_review={
        "approved": False, "issues": [{"source_field": "trace", "quote": "failure",
                                       "reason": "Must not use trace evidence."}]})},
], ids=["without_clauses", "missing_fields", "invalid_policy_hash", "unanchored_exclusion",
        "rejection_without_issue", "unanchored_coverage_issue"])
def test_compilation_metadata_must_be_strict_and_source_anchored(changes):
    with pytest.raises(ValueError):
        _contract(**changes)
