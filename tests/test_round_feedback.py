# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round feedback reports observed progress and grounded next steps without inventing work."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.session.discovery_recovery import (
    _recovery_tasks,
    recovery_round_context,
    update_recovery_assignment,
)
from autofa.session.exploration import resolve_exploration_seed
from autofa.session.family_discovery import STATE_FILE, initialize_family_discovery
from autofa.session.round_exploration import build_round_exploration
from autofa.session.round_feedback import round_bulletin, round_change, round_snapshot
from autofa.util.atomic import atomic_write_json

from support.discovery import snapshot
from support.families import FamilyDataset

# Round change and bulletin: boundaries survive Resume; status reads cannot manufacture progress.


def test_same_grade_progress_requires_comparable_evaluation_evidence():
    before, after = snapshot(), snapshot(score=.4, confirmed=3)
    result = round_change(before, after)
    assert result["encouraging"]
    assert result["search_score_change"] == pytest.approx(.1)
    assert result["confirmed_task_net_change"] == 1
    assert not round_change(before, before)["encouraging"]
    assert not round_change(after, before)["encouraging"]


@pytest.mark.parametrize("after,unavailable", [
    (snapshot(score=.9, cohort=None), "search_score_change"),
    (snapshot(confirmed=20, corpus="other"), "confirmed_task_net_change"),
], ids=["unknown-sample", "other-corpus"])
def test_incomparable_evidence_does_not_show_improvement(after, unavailable):
    result = round_change(snapshot(), after)
    assert result[unavailable] is None
    assert not result["encouraging"]


@pytest.fixture
def rt(tmp_path, monkeypatch):
    runtime = SimpleNamespace(
        dataset=FamilyDataset(), library=FunctionLibrary(), config=Config(), scores_history=[],
        run=SimpleNamespace(run_id="round-test", round=0, min_failure_modes=1,
                            max_rounds=5, top_x=5, run_dir=lambda: tmp_path, save=lambda: None,
                            exploration_ssot=False, exploration_sequence_index=0,
                            exploration_base_seed=resolve_exploration_seed(run_id="round-test")),
        persist_library=lambda: None, llm_readiness={},
    )
    monkeypatch.setattr("autofa.mcp.runtime.get_runtime", lambda: runtime)
    monkeypatch.setattr("autofa.mcp.tools_library.get_runtime", lambda: runtime)
    return runtime


def test_start_baseline_is_durable_and_status_is_read_only(rt):
    state = initialize_family_discovery(rt)
    saved = (rt.run.run_dir() / STATE_FILE).read_bytes()
    baseline = deepcopy(state["round_baseline"])
    rt.library.upsert(FeatureFunction(name="new_lead", description="A new unreviewed lead",
                                      source="def detect(item): return False"))
    current = round_snapshot(rt)
    for _ in range(2):
        bulletin = round_bulletin(rt, grades=current["grades"])
        assert bulletin["current_round"]["baseline_available"]
        assert not bulletin["encouraging"]
    assert (rt.run.run_dir() / STATE_FILE).read_bytes() == saved
    # Reconstructing runtime memory for Resume preserves the true start.
    resumed = SimpleNamespace(**vars(rt))
    assert initialize_family_discovery(resumed)["round_baseline"] == baseline
    assert (rt.run.run_dir() / STATE_FILE).read_bytes() == saved


def test_multiple_advances_in_one_turn_archive_each_round_once(rt, monkeypatch):
    from autofa.mcp import tools_library

    monkeypatch.setattr(tools_library, "_detector_exploration", lambda: {"message": "Explore"})
    baseline = initialize_family_discovery(rt)["round_baseline"]
    for expected in (1, 2):
        state = json.loads((rt.run.run_dir() / STATE_FILE).read_text())
        state["phase"] = "specialize"
        atomic_write_json(rt.run.run_dir() / STATE_FILE, state)
        result = tools_library.advance_round()
        assert result["ok"] and result["advanced"]
        bulletin = result["agent_grade_feedback"]["round_bulletin"]
        assert bulletin["round"] == expected
        assert bulletin["last_completed_round"]["round"] == expected - 1
        state = initialize_family_discovery(rt)
        assert len(state["history"]) == expected
        assert state["history"][-1]["round_baseline"] == baseline
        baseline = state["history"][-1]["round_outcome"]
        assert state["round_baseline"] == baseline


def test_source_outage_does_not_erase_baseline(rt, monkeypatch):
    baseline = initialize_family_discovery(rt)["round_baseline"]
    monkeypatch.setattr("autofa.session.task_coverage.task_coverage_snapshot",
                        lambda rt: {"available": False})
    bulletin = round_bulletin(rt, grades=baseline["grades"])
    assert bulletin["current_round"]["confirmed_task_net_change"] is None
    assert initialize_family_discovery(rt)["round_baseline"] == baseline


def test_pending_family_round_is_not_silently_archived(rt):
    state = initialize_family_discovery(rt)
    state["phase"] = "assess_family"
    atomic_write_json(rt.run.run_dir() / STATE_FILE, state)
    rt.run.round += 1
    current = initialize_family_discovery(rt)
    assert current["round"] == state["round"]
    assert current["round_baseline"] == state["round_baseline"]
    assert current["history"] == []


# Round exploration advice explores library gaps without inventing taxonomy or doing paid work.


def _exploration_runtime(tmp_path, *, grades=("GOOD",), mechanisms=None):
    mechanisms = mechanisms or ["A missing write leaves the agreed state unchanged."] * len(grades)
    functions = [
        FeatureFunction(name=f"detector_{i}", source="def detect(item): return True",
                        description="Recorded observable behavior.", mechanism=mechanism)
        for i, mechanism in enumerate(mechanisms)
    ]
    rt = SimpleNamespace(
        run=SimpleNamespace(run_id="round-test", run_dir=lambda: tmp_path),
        config=Config(), library=FunctionLibrary(functions=functions),
    )
    summary = {"functions": [
        {"name": fn.name, "code": code} for fn, code in zip(functions, grades, strict=True)
    ]}
    return rt, summary


def _comparison(*, name="known_family", note="The upload record is missing."):
    return {
        "round": 1,
        "survey": {"family_name": name,
                   "mechanism": "A missing input could explain an incomplete answer.",
                   "evidence": [{"item_id": "case-1", "task_id": "task-1"}]},
        "assessment": {"outcome": "deferred", "blocker": note,
                       "next_step": "Inspect retained input-delivery metadata."},
    }


def test_read_only_context_cannot_execute_detectors_or_inventory(tmp_path):
    rt, grades = _exploration_runtime(tmp_path)
    state = {"version": 3, "run_id": rt.run.run_id, "comparisons": [_comparison()]}
    path = tmp_path / "family_discovery.json"
    path.write_text(json.dumps(state))
    originals = (deepcopy(vars(rt)), grades.copy(), path.read_bytes())
    # Runtime intentionally has no dataset: this path must not invent one or scan a corpus.
    first = build_round_exploration(rt, grades=grades)
    assert build_round_exploration(rt, grades=grades) == first
    assert path.read_bytes() == originals[2]
    assert grades == originals[1]
    assert set(vars(rt)) == set(originals[0])
    assert rt.library.functions[0].scores == originals[0]["library"].functions[0].scores
    assert len(list(tmp_path.iterdir())) == 1


def test_saved_lessons_keep_writer_shapes_and_removed_modes(tmp_path):
    rt, grades = _exploration_runtime(tmp_path, grades=())
    # Exact writer shape for a synthesis inspection after the comparison budget.
    state = {"history": [{
        "round": 1, "outcome": "synthesis_inspection_only",
        "survey": {"outcome": "no_coherent_family", "family_name": None,
                   "evidence": [{"item_id": "case-1"}, {"item_id": "case-2"}]},
        "assessment": "The checked seeds require different remedies; a shared cause is unproven.",
    }]}
    original = deepcopy(state)
    (tmp_path / "modes_ledger.jsonl").write_text(json.dumps({
        "name": "rejected_mode", "status": "rejected", "mechanism": "Explicit recorded claim.",
        "reason": "The recorded counterevidence.",
    }))
    inspection, rejected = build_round_exploration(rt, grades=grades,
                                                   family_state=state)["lessons"]
    assert inspection["outcome"] == "no_coherent_family"
    assert inspection["lesson"] == state["history"][0]["assessment"]
    assert inspection["reference"] == "family_discovery.json#/history/0"
    assert inspection["evidence_item_ids"] == ["case-1", "case-2"]
    assert rejected["outcome"] == "rejected"
    assert rejected["reference"]["tool"] == "list_failure_modes"
    assert state == original


def test_bounded_output_preserves_references_to_full_details(tmp_path):
    rt, grades = _exploration_runtime(
        tmp_path, grades=("GOOD",) * 30, mechanisms=["Long claim. " * 1000] * 30,
    )
    state = {
        "comparisons": [_comparison(name=f"claim_{i}", note="Details. " * 1000) for i in range(30)]
    }
    before = deepcopy(state)
    result = build_round_exploration(rt, grades=grades, family_state=state)
    assert len(result["claim_cards"]) == 6
    assert result["n_claim_cards_omitted"] == 24
    assert len(result["lessons"]) == 5
    assert len(result["directions"]) <= 3
    assert all(len(row["mechanism"]) <= 260 for row in result["claim_cards"])
    assert all(row["reference"] for row in result["lessons"] + result["claim_cards"])
    assert len(result["message"].split()) < 400
    assert state == before


@pytest.mark.parametrize(
    "contents", ["broken json", json.dumps({"version": 3, "run_id": "other-run"})],
    ids=["broken", "foreign-run"],
)
def test_unavailable_or_foreign_history_is_not_silently_presented_as_current(tmp_path, contents):
    rt, grades = _exploration_runtime(tmp_path)
    (tmp_path / "family_discovery.json").write_text(contents)
    result = build_round_exploration(rt, grades=grades)
    assert result["unavailable"]
    assert result["lessons"] == []
    assert result["n_retained_claims"] == 1


# Recovery rounds rotate source-backed evidence without inventing completed work.


@pytest.fixture
def recovery_rt(tmp_path):
    tasks = {"a1": "alpha:1", "a1-repeat": "alpha:1", "a2": "alpha:2",
             "b1": "beta:1", "b2": "beta:2"}
    return SimpleNamespace(
        config=Config(), library=FunctionLibrary(),
        run=SimpleNamespace(run_id="recovery", round=0, run_dir=lambda: tmp_path),
        _family_failed_task_inventory={
            "corpus_fingerprint": "corpus", "task_by_failed_item_id": tasks,
            "task_by_item_id": tasks,
            "identity_by_item_id": {
                item: {"benchmark": task.split(":")[0]} for item, task in tasks.items()
            },
        },
    )


def _concluded_recovery(rt):
    """Assign the first recovery batch and record an incoherent-family conclusion for it."""
    state = {"records": {}, "processed": []}
    first = update_recovery_assignment(rt, state, active=True)
    (rt.run.run_dir() / "family_discovery.json").write_text(json.dumps({
        "run_id": rt.run.run_id, "phase": "survey", "comparisons": [{
            "survey": {"family_name": "investigated_mechanism", "evidence": [
                {"item_id": item, "quotes": [{"field": "response_text", "quote": "Delivered"}]}
                for item in first["item_ids"]
            ]},
            "assessment": {"outcome": "no_coherent_family",
                           "notes": "The final response delivers the requested content."},
        }],
    }))
    return state, first


def test_task_selection_rotates_verified_benchmarks_and_avoids_other_rollouts(recovery_rt):
    ledger = {"history_complete": True}
    first, selection = _recovery_tasks(recovery_rt, ledger, [])
    assert first == ["a1", "b1"]
    assert selection == "not_previously_exposed_tasks"
    history = [{"item_ids": first, "result": {"outcome": "no_coherent_family"}}]
    second, selection = _recovery_tasks(recovery_rt, ledger, history)
    assert second == ["a2", "b2"]
    assert selection == "not_previously_exposed_tasks"


def test_unknown_task_identity_is_never_reported_as_fresh_task(recovery_rt):
    recovery_rt._family_failed_task_inventory["task_by_failed_item_id"] = {"opaque": None}
    items, selection = _recovery_tasks(recovery_rt, {"history_complete": True}, [])
    assert items == ["opaque"]
    assert selection == "revisit_or_exposure_unknown"


def test_recovery_lessons_survive_successful_phase_and_resume(recovery_rt):
    rt = recovery_rt
    state, first = _concluded_recovery(rt)
    following = update_recovery_assignment(rt, state, active=True)
    assert following["item_ids"] == ["a2", "b2"]
    assert following["previous_investigation"]["item_ids"] == first["item_ids"]
    history = deepcopy(state["recovery_history"])
    assert history[0]["round"] == 0
    assert history[0]["result"]["outcome"] == "no_coherent_family"

    assert update_recovery_assignment(rt, state, active=False) is None
    assert state["recovery_history"] == history
    rt.run.round = 1
    resumed = update_recovery_assignment(rt, state, active=True)
    assert resumed["item_ids"] == ["a2", "b2"]
    assert state["recovery_history"] == history
    assert state["processed"] == []


def test_new_lead_retrieval_or_round_change_cannot_complete_recovery(recovery_rt):
    rt = recovery_rt
    state = {"records": {}, "processed": []}
    first = update_recovery_assignment(rt, state, active=True)
    rt.run.round = 3
    (rt.run.run_dir() / "review_evidence.json").write_text(json.dumps({
        "discovery_item_ids": first["item_ids"], "history_complete": True,
    }))
    (rt.run.run_dir() / "family_discovery.json").write_text(json.dumps({
        "run_id": rt.run.run_id, "phase": "assess_family", "comparisons": [],
        "survey": {"family_name": "new_lead", "evidence": [{"item_id": first["item_ids"][0]}]},
    }))
    assert update_recovery_assignment(rt, state, active=True) == first
    assert not state["recovery_history"]
    assert state["processed"] == []


def test_new_corpus_does_not_inherit_negative_task_exclusions(recovery_rt):
    rt = recovery_rt
    state, _ = _concluded_recovery(rt)
    update_recovery_assignment(rt, state, active=True)
    update_recovery_assignment(rt, state, active=False)
    rt._family_failed_task_inventory["corpus_fingerprint"] = "other-corpus"
    resumed = update_recovery_assignment(rt, state, active=True)
    assert resumed["item_ids"] == ["a1", "b1"]
    assert "previous_result" not in resumed
    assert len(state["recovery_history"]) == 1


def test_round_context_is_read_only_and_retains_evidence_references(recovery_rt):
    rt = recovery_rt
    state, first = _concluded_recovery(rt)
    update_recovery_assignment(rt, state, active=True)
    path = rt.run.run_dir() / "discovery_stall.json"
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    context = recovery_round_context(rt)
    assert context == recovery_round_context(rt)
    assert path.read_bytes() == original
    lesson = context["completed_investigations"][0]
    assert lesson["reference"] == "discovery_stall.json#/recovery_history/0"
    assert lesson["item_ids"] == first["item_ids"]
