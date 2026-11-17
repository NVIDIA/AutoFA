# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review allocation keeps held-out tasks, frozen cohorts and portable exposure history."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import server, tools_eval
from autofa.metrics import claude_judge
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract
from autofa.session.family_coverage import verified_identity_projection
from autofa.session.family_discovery import initialize_family_discovery
from autofa.session.review_evidence import (
    allocate_judge_expansion,
    allocate_review,
    complete_review_allocation,
    empty_history,
    find_review_allocation,
    import_seed_review_history,
    merge_histories,
    record_discovery_exposure,
    review_context,
    sync_review_history,
    task_group,
)
from autofa.session.synthesis_review import synthesis_review_strata

from support.coverage import detector
from support.coverage import rt as rt
from support.repair_review import DESCRIPTION, MECHANISM, REQUEST, exposure_runtime
from support.repair_review import repair_runtime as repair_runtime


def items(tasks=10, repeats=3):
    return [{"_item_id": f"opaque-{task * repeats + repeat}", "benchmark": "terminal-bench",
             "problem_id": f"task-{task}", "response_text": "large trace not retained"}
            for task in range(tasks) for repeat in range(repeats)]


def allocate(path, rows, *, key="first", phase="development", name="family", target=4, related=()):
    return allocate_review(path, name, rows, target_n=target, seed=37, review_key=key,
                           phase=phase, related_names=related)


def groups(rows, ids):
    """Task groups of the given item IDs."""
    lookup = {row["_item_id"]: row for row in rows}
    return {task_group(lookup[key]) for key in ids}


def test_task_reservations_precede_development_and_confirmation_uses_distinct_tasks(tmp_path):
    rows = items()
    dev = allocate(tmp_path, rows)
    reserved = set(dev["reserved_confirmation_item_ids"])
    assert reserved and len(dev["item_ids"]) == 4
    dev_groups = groups(rows, dev["item_ids"])
    assert len(dev_groups) == 4
    assert not dev_groups & groups(rows, reserved)
    confirm = allocate(tmp_path, rows, phase="confirmation")
    assert not dev_groups & groups(rows, confirm["item_ids"])
    assert confirm["sample_info"]["independent_task_n"] >= 2
    assert not confirm["sample_info"]["reused_n"]
    assert not confirm["sample_info"]["limited"]


def test_frozen_allocation_survives_resume_and_refuses_missing_evidence(tmp_path):
    rows = items()
    first = allocate(tmp_path, rows)
    record_discovery_exposure(tmp_path, first["item_ids"])
    assert allocate(tmp_path, list(reversed(rows))) == first
    remaining = [row for row in rows if row["_item_id"] != first["item_ids"][0]]
    with pytest.raises(ValueError, match="Frozen review evidence is missing"):
        allocate(tmp_path, remaining)


def test_opaque_exposure_id_later_excludes_all_attempts_of_its_task(tmp_path):
    rows = items()
    record_discovery_exposure(tmp_path, [rows[0]["_item_id"]])
    confirmation = allocate(tmp_path, rows, phase="confirmation", target=25)
    assert not set(confirmation["item_ids"]) & {row["_item_id"] for row in rows[:3]}
    assert confirmation["sample_info"]["independence_scope"] == (
        "recorded_discovery_and_judge_exposure"
    )


def test_single_task_remains_reviewable_but_repeats_are_not_independent(tmp_path):
    rows = items(tasks=1, repeats=12)
    dev = allocate(tmp_path, rows)
    assert len(dev["item_ids"]) == 4
    assert dev["reserved_confirmation_item_ids"] == []
    confirmation = allocate(tmp_path, rows, phase="confirmation")
    assert confirmation["sample_info"]["limited"]
    assert confirmation["sample_info"]["independent_task_n"] == 0
    assert confirmation["sample_info"]["reused_n"] == 4
    assert set(confirmation["item_ids"]) == set(dev["item_ids"])


def test_missing_task_metadata_never_infers_task_from_item_id(tmp_path):
    rows = [{"_item_id": f"model|benchmark|task-{i}|0"} for i in range(5)]
    confirmation = allocate(tmp_path, rows, phase="confirmation")
    assert len(confirmation["item_ids"]) == 4
    assert confirmation["sample_info"]["unknown_task_n"] == 4
    assert confirmation["sample_info"]["limited"]
    assert confirmation["sample_info"]["independent_task_n"] == 0
    assert task_group({"benchmark": "b", "problem_id": 0}) is not None


def test_child_inherits_parent_exposure_and_reservations(tmp_path):
    rows = items()
    parent = allocate(tmp_path, rows)
    child = allocate(tmp_path, rows, phase="confirmation", name="child", related=["family"])
    assert not groups(rows, parent["item_ids"]) & groups(rows, child["item_ids"])
    assert child["sample_info"]["independent_task_n"] > 0


def test_concurrent_diagnostic_allocations_do_not_reuse_items(tmp_path):
    rows = items(tasks=20)
    with ThreadPoolExecutor(max_workers=2) as executor:
        allocations = list(executor.map(lambda key: allocate(tmp_path, rows, key=key), ["a", "b"]))
    assert not set(allocations[0]["item_ids"]) & set(allocations[1]["item_ids"])
    assert len(review_context(tmp_path, "family")["reviewed_item_ids"]) == 8


def test_portable_library_preserves_old_versions_and_removed_detectors(tmp_path):
    fn = FeatureFunction(name="family", source="def family(item): return True", description="cause",
                         scores={"behavior_verdicts": [{"item_id": "old", "verdict": "yes"}]})
    library = FunctionLibrary(functions=[fn])
    library.upsert(FeatureFunction(name="family", source=fn.source, description="repaired cause",
                                   scores={"behavior_verdicts": [
                                       {"item_id": "new", "verdict": "no"},
                                   ]}))
    library.remove("family")
    library.save(tmp_path / "export.json")
    continued = FunctionLibrary.load(tmp_path / "export.json")
    assert continued.functions == []
    sync_review_history(continued, tmp_path / "continue")
    assert review_context(tmp_path / "continue", "family")["reviewed_item_ids"] == ["new", "old"]


def test_continue_preserves_reservations_and_global_discovery_exposure(tmp_path):
    library = FunctionLibrary()
    rows = items()
    first = allocate(tmp_path / "old", rows)
    record_discovery_exposure(tmp_path / "old", [rows[-1]])
    sync_review_history(library, tmp_path / "old")
    continued = FunctionLibrary.from_dict(library.to_dict())
    sync_review_history(continued, tmp_path / "new")
    continued_allocation = allocate(tmp_path / "new", rows)
    assert continued_allocation["origin_run_id"] == "new"
    assert not set(continued_allocation["item_ids"]) & set(first["item_ids"])
    context = review_context(tmp_path / "new", "family")
    assert set(first["item_ids"]) | {rows[-1]["_item_id"]} <= set(context["reviewed_item_ids"])
    assert context["history_complete"]


def test_unavailable_confirmation_does_not_freeze_when_new_tasks_arrive(tmp_path):
    rows = items(tasks=4, repeats=1)
    record_discovery_exposure(tmp_path, rows)
    unavailable = allocate(tmp_path, rows, phase="confirmation")
    assert unavailable["sample_info"]["independent_task_n"] == 0
    assert find_review_allocation(tmp_path, "family", review_key="first",
                                  phase="confirmation") is None
    fresh = allocate(tmp_path, items(tasks=20, repeats=1), phase="confirmation")
    assert fresh["sample_info"]["independent_task_n"] == 4
    assert not fresh["sample_info"]["limited"]


def test_legacy_library_does_not_gain_fresh_provenance_after_continue(tmp_path):
    legacy = FunctionLibrary.from_dict({"functions": [{
        "name": "old", "source": "def old(item): return True", "description": "cause",
        "scores": {"behavior_verdicts": [{"item_id": "opaque-0", "verdict": "yes"}]},
    }]})
    sync_review_history(legacy, tmp_path)
    assert review_context(tmp_path, "old")["reviewed_item_ids"] == ["opaque-0"]
    result = allocate(tmp_path, items(), phase="confirmation", name="old")
    assert not result["sample_info"]["history_complete"]
    assert result["sample_info"]["limited"]
    assert "history is incomplete" in result["sample_info"]["reason"]
    assert not FunctionLibrary.from_dict(legacy.to_dict()).review_history["history_complete"]


def test_conflicting_exposure_identity_fails_closed(tmp_path):
    row = items()[0]
    record_discovery_exposure(tmp_path, [row])
    with pytest.raises(ValueError, match="Task identity changed"):
        record_discovery_exposure(tmp_path, [{**row, "problem_id": "different"}])
    old = empty_history()
    old["discovery_task_groups"]["same-id"] = "first"
    new = deepcopy(old)
    new["discovery_task_groups"]["same-id"] = "second"
    with pytest.raises(ValueError, match="Conflicting task"):
        merge_histories(old, new)


def test_old_local_seed_recovers_archived_exclusions_without_touching_source(tmp_path):
    source = tmp_path / "runs" / "old"
    source.mkdir(parents=True)
    ledger = {"schema_version": "detector-repair-v1", "repairs": {"family": {
        "name": "family", "reviewed_item_ids": ["archived", "removed", "latest"],
        "history": [{"candidates": {"child": {}}}],
    }}}
    old_path = source / "detector_repairs.json"
    old_path.write_text(json.dumps(ledger))
    stat = old_path.stat()
    library = FunctionLibrary.from_dict({"functions": [{
        "name": "child", "source": "def child(item): return True", "description": "cause",
        "scores": {"evaluation_run_id": "old"},
    }]})
    import_seed_review_history(library, tmp_path / "catalog" / "library.json",
                               runs_root=tmp_path / "runs")
    sync_review_history(library, tmp_path / "new")
    assert review_context(tmp_path / "new", "child")["reviewed_item_ids"] == [
        "archived", "latest", "removed",
    ]
    assert old_path.stat().st_mtime_ns == stat.st_mtime_ns
    assert list(source.iterdir()) == [old_path]
    assert not library.review_history["history_complete"]


def test_frozen_allocation_rejects_changed_task_identity(tmp_path):
    rows = items()
    first = allocate(tmp_path, rows)
    changed = [{**row, "problem_id": "replacement"} if row["_item_id"] in first["item_ids"]
               else row for row in rows]
    with pytest.raises(ValueError, match="Task identity changed for frozen"):
        allocate(tmp_path, changed)


def test_renaming_or_unrelated_family_cannot_make_previous_review_fresh(tmp_path):
    rows = items()
    first = allocate(tmp_path, rows, name="first")
    second = allocate(tmp_path, rows, name="unrelated")
    assert not set(first["item_ids"]) & set(second["item_ids"])
    assert not (groups(rows, first["reserved_confirmation_item_ids"])
                & groups(rows, second["item_ids"]))
    confirmation = allocate(tmp_path, rows, phase="confirmation", name="renamed")
    assert not (groups(rows, first["item_ids"] + second["item_ids"])
                & groups(rows, confirmation["item_ids"]))


def test_disjoint_family_reserves_its_own_relevant_confirmation_tasks(tmp_path):
    # Another family's reservations on other tasks do not count toward this one's.
    first = items(tasks=5, repeats=1)
    second = [{**row, "_item_id": "second-" + row["_item_id"],
               "problem_id": "second-" + row["problem_id"]} for row in first]
    allocate(tmp_path, first, name="a", target=5)
    development = allocate(tmp_path, second, name="b", target=5)
    assert development["reserved_confirmation_item_ids"] and len(development["item_ids"]) < 5
    confirmation = allocate(tmp_path, second, name="b", phase="confirmation", target=5)
    assert confirmation["sample_info"]["independent_task_n"] >= 1
    assert not set(development["item_ids"]) & set(confirmation["item_ids"])


def test_reservation_preserves_confirmation_capacity_before_development_uses_it(tmp_path):
    # Single-attempt tasks: a reserved task count alone would leave confirmation short.
    rows = items(tasks=15, repeats=1)
    development = allocate(tmp_path, rows, target=10)
    assert len(development["item_ids"]) == 5
    assert len(development["reserved_confirmation_item_ids"]) == 10
    confirmation = allocate(tmp_path, rows, phase="confirmation", target=10)
    assert confirmation["sample_info"]["independent_task_n"] == 10
    assert not confirmation["sample_info"]["limited"]


def test_two_task_cohort_can_develop_on_one_and_confirm_on_the_other(tmp_path):
    rows = items(tasks=2, repeats=10)
    development = allocate(tmp_path, rows, target=10)
    confirmation = allocate(tmp_path, rows, phase="confirmation", target=10)
    assert len(development["item_ids"]) == len(confirmation["item_ids"]) == 10
    assert not groups(rows, development["item_ids"]) & groups(rows, confirmation["item_ids"])
    assert confirmation["sample_info"]["independent_task_n"] == 1
    assert not confirmation["sample_info"]["limited"]


def test_support_expansion_retains_findings_and_uses_five_failed_tasks(tmp_path):
    rows = items(tasks=7, repeats=2)
    failed = {row['_item_id'] for row in rows[:10]}
    lookup = {row['_item_id']: row for row in rows}
    first = allocate(tmp_path, rows[:10], target=3)
    retained = first['item_ids'] + [rows[-1]['_item_id']]
    record_discovery_exposure(tmp_path, [lookup[key] for key in retained])
    expanded = allocate_review(
        tmp_path, 'family', rows, target_n=5, seed=37, review_key='expanded',
        support_review={'retained_item_ids': retained, 'failed_item_ids': list(failed),
                        'min_failed_tasks': 5},
    )
    selected = set(expanded['item_ids'])
    assert set(retained) <= selected  # Includes the retained passing/counterexample control.
    assert len(groups(rows, selected & failed)) == 5
    assert expanded['sample_info']['reused_n'] == len(retained)
    assert expanded['sample_info']['limited']
    assert expanded['sample_info']['support_review']['selected_failed_task_n'] == 5
    # A retry preserves the entire frozen cohort even when corpus order changes.
    assert allocate_review(tmp_path, 'family', list(reversed(rows)), target_n=9,
                           seed=37, review_key='expanded') == expanded
    confirm = allocate(tmp_path, rows, phase='confirmation', target=10)
    assert not groups(rows, confirm['item_ids']) & groups(rows, selected)


def test_support_expansion_cannot_drop_a_saved_counterexample(tmp_path):
    rows = items(tasks=5, repeats=1)
    with pytest.raises(ValueError, match='Current review evidence is missing'):
        allocate_review(tmp_path, 'family', rows, target_n=5, seed=37, review_key='expanded',
                        support_review={'retained_item_ids': ['missing-counterexample'],
                                        'failed_item_ids': [], 'min_failed_tasks': 5})


# Adaptive expansion and outcome-selected diagnostic controls.


def test_development_fills_target_with_reused_nonreserved_evidence(tmp_path):
    rows = items(3, 1)
    record_discovery_exposure(tmp_path, [rows[0]])
    allocated = allocate_review(tmp_path, "f", rows, target_n=3, seed=2, review_key="initial")
    assert rows[0]["_item_id"] in allocated["item_ids"]
    assert len(allocated["item_ids"]) == 2  # third task remains reserved for confirmation
    assert allocated["sample_info"]["reused_n"] == 1
    assert allocated["sample_info"]["fresh_item_n"] == 1
    assert allocated["sample_info"]["limited"]
    confirm = allocate_review(tmp_path, "f", rows, target_n=3, seed=2,
                              review_key="confirm", phase="confirmation")
    assert len(confirm["item_ids"]) == 1
    assert not set(confirm["item_ids"]) & set(allocated["item_ids"])
    assert confirm["sample_info"]["reused_n"] == 0


def test_expansion_is_frozen_distinct_bounded_and_does_not_consume_reservations(tmp_path):
    rows = items(20, 3)
    lookup = {row["_item_id"]: row for row in rows}
    base = allocate_review(tmp_path, "f", rows, target_n=3, seed=9, review_key="initial")
    added = allocate_judge_expansion(tmp_path, "f", rows, review_key="initial", seed=9,
                                    request=REQUEST)
    info = added["sample_info"]["judge_expansion"]
    assert len(added["item_ids"]) == 6
    assert added["item_ids"][:3] == base["item_ids"]
    assert len(groups(rows, added["item_ids"])) == 6
    assert not set(added["item_ids"]) & set(base["reserved_confirmation_item_ids"])
    assert info["max_total_n"] == 6
    # A repeated judge request, new evidence, or traversal order cannot buy another batch.
    again = allocate_judge_expansion(tmp_path, "f", list(reversed(rows)) + items(30, 1),
                                    review_key="initial", seed=999, request=REQUEST)
    assert again == added
    assert find_review_allocation(tmp_path, "f", review_key="initial", phase="development")[
        "item_ids"] == added["item_ids"]
    resumed = allocate_review(tmp_path, "f", [lookup[key] for key in added["item_ids"]],
                               target_n=20, seed=999, review_key="initial")
    assert resumed == added


def test_no_additional_distinct_task_is_durable_and_cannot_reset_budget(tmp_path):
    rows = items(2, 2)
    base = allocate_review(tmp_path, "f", rows, target_n=3, seed=9, review_key="initial")
    expanded = allocate_judge_expansion(tmp_path, "f", rows, review_key="initial", seed=9,
                                       request=REQUEST)
    assert expanded["item_ids"] == base["item_ids"]
    assert expanded["sample_info"]["judge_expansion"]["status"] == "unavailable"
    assert allocate_judge_expansion(tmp_path, "f", items(100), review_key="initial", seed=9,
                                   request=REQUEST) == expanded


def test_expansion_cannot_expand_original_pool_or_recover_another_run(tmp_path):
    rows = items(10, 1)
    allocate_review(tmp_path, "f", rows, target_n=3, seed=9, review_key="initial")
    with pytest.raises(ValueError, match="Original matched pool"):
        allocate_judge_expansion(tmp_path, "f", items(11, 1), review_key="initial", seed=9,
                                request=REQUEST)
    with pytest.raises(ValueError, match="original same-run"):
        allocate_judge_expansion(tmp_path / "continued", "f", rows, review_key="initial", seed=9,
                                request=REQUEST)


def test_supplement_preserves_original_cohort_and_exposes_one_missing_outcome(tmp_path):
    rows = items(12, 2)
    baseline = allocate_review(
        tmp_path / "baseline", "f", rows, target_n=3, seed=4, review_key="first")
    original = set(baseline["item_ids"])
    reserved = set(baseline["reserved_confirmation_item_ids"])
    missing = next(row["_item_id"] for row in rows
                   if row["_item_id"] not in original | reserved
                   and task_group(row) not in {
                       task_group(r) for r in rows if r["_item_id"] in original})
    outcomes = {
        "failed": [missing], "passed": [r["_item_id"] for r in rows if r["_item_id"] != missing],
    }
    allocation = allocate_review(
        tmp_path / "actual", "f", rows, target_n=3, seed=4, review_key="first",
        outcome_ids=outcomes)
    assert allocation["item_ids"] == baseline["item_ids"]
    assert allocation["diagnostic_item_ids"] == [missing]
    assert allocation["reserved_confirmation_item_ids"] == (
        baseline["reserved_confirmation_item_ids"])
    info = allocation["sample_info"]["diagnostic_sample"]
    assert info["excluded_from_adherence"] and info["excluded_from_confirmed_support"]
    assert info["strata"]["failed"]["status"] == "supplemental"
    assert info["strata"]["passed"]["status"] == "represented"
    assert missing in review_context(tmp_path / "actual", "f")["reviewed_item_ids"]
    assert allocate_review(
        tmp_path / "actual", "f", list(reversed(rows)), target_n=30, seed=99,
        review_key="first", outcome_ids={"failed": [], "passed": []}) == allocation
    with pytest.raises(ValueError, match="Frozen review evidence is missing"):
        allocate_review(tmp_path / "actual", "f",
                        [row for row in rows if row["_item_id"] != missing],
                        target_n=3, seed=4, review_key="first")


def test_old_cohort_is_not_retroactively_enlarged_and_reservations_remain_protected(tmp_path):
    rows = items(10, 1)
    old = allocate_review(tmp_path, "f", rows, target_n=3, seed=4, review_key="first")
    reserved = old["reserved_confirmation_item_ids"]
    outcomes = {"failed": reserved, "passed": old["item_ids"]}
    assert allocate_review(tmp_path, "f", rows, target_n=3, seed=4, review_key="first",
                           outcome_ids=outcomes) == old
    new = allocate_review(tmp_path, "f", rows, target_n=3, seed=4, review_key="second",
                          outcome_ids=outcomes)
    assert not set(new["diagnostic_item_ids"]) & set(reserved)
    assert new["sample_info"]["diagnostic_sample"]["strata"]["failed"]["status"] == "unavailable"


def test_diagnostic_task_never_enters_adaptive_scored_expansion(tmp_path):
    rows = items(16, 2)
    record_discovery_exposure(tmp_path, rows[:2])
    # Exposed minority controls remain useful for diagnosis.
    base = allocate_review(
        tmp_path, "f", rows, target_n=3, seed=4, review_key="first",
        outcome_ids={"failed": [rows[0]["_item_id"]],
                     "passed": [r["_item_id"] for r in rows[2:]]})
    assert base["diagnostic_item_ids"] == [rows[0]["_item_id"]]
    expanded = allocate_judge_expansion(
        tmp_path, "f", rows, review_key="first", seed=4, request=REQUEST)
    assert expanded["diagnostic_item_ids"] == base["diagnostic_item_ids"]
    assert not groups(rows, base["diagnostic_item_ids"]) & groups(rows, expanded["item_ids"])


def test_unknown_scored_outcomes_cannot_request_more_than_one_control(tmp_path):
    rows = items(16, 1)
    base = allocate_review(
        tmp_path / "preview", "f", rows, target_n=3, seed=4, review_key="first")
    available = [r["_item_id"] for r in rows if r["_item_id"] not in (
        base["item_ids"] + base["reserved_confirmation_item_ids"])]
    actual = allocate_review(
        tmp_path / "actual", "f", rows, target_n=3, seed=4, review_key="first",
        outcome_ids={"failed": available[:1], "passed": available[1:]})
    assert actual["diagnostic_item_ids"] == available[:1]
    assert actual["sample_info"]["diagnostic_sample"]["strata"]["passed"]["status"] == (
        "budget_limited")


def test_smoke_diagnostics_preserve_reserved_confirmation_tasks(tmp_path):
    rows = items(8, 1)
    first = allocate_review(tmp_path, "other", rows, target_n=3, seed=4, review_key="first")
    reserved = first["reserved_confirmation_item_ids"]
    assert reserved
    control = next(row for row in rows if row["_item_id"] == reserved[0])
    separate = allocate_review(
        tmp_path, "new", [], target_n=0, seed=4, review_key="new",
        diagnostic_items=[control],
    )
    assert separate["item_ids"] == []
    assert separate["diagnostic_item_ids"] == []


def test_legacy_empty_development_allocation_does_not_freeze_future_evidence(tmp_path):
    empty = allocate_review(tmp_path, "f", [], target_n=0, seed=4, review_key="first")
    path = tmp_path / "review_evidence.json"
    state = json.loads(path.read_text())
    state["detectors"]["f"]["allocations"][f"development:{tmp_path.name}:first"] = empty
    path.write_text(json.dumps(state))
    assert find_review_allocation(tmp_path, "f", review_key="first", phase="development") is None
    resumed = allocate_review(tmp_path, "f", items(8, 1), target_n=3, seed=4, review_key="first")
    assert len(resumed["item_ids"]) == 3


# Compatible operations preserve cohorts; obsolete reviews remain historical.


def _context(tmp_path, profile="causal"):
    rt = SimpleNamespace(run=SimpleNamespace(run_id=tmp_path.name, run_dir=lambda: tmp_path))
    fn = {"name": "family", "source": "def detect(item): return True",
          "description": "An observed mechanism", "mechanism": "A concrete mistake", "scores": {}}
    settings = {"judge_objective_strategy": "recall", "judge_behavior_strictness": profile,
                "judge_model": "test"}
    rows = [{"_item_id": f"item-{i}", "benchmark": "test", "problem_id": f"task-{i}"}
            for i in range(40)]
    return rt, fn, settings, rows


@pytest.mark.parametrize("profile,phase", [
    ("behavioral", "development"), ("causal", "confirmation"),
])
def test_current_policy_allocation_resolves_original_ids_and_target(
    tmp_path, profile, phase,
):
    rt, fn, settings, rows = _context(tmp_path, profile)
    key = tools_eval._review_key(fn, settings, 4)
    original = allocate_review(tmp_path, "family", rows, target_n=4, seed=17,
                               review_key=key, phase=phase)
    found_key, allocation = tools_eval._find_compatible_allocation(
        rt, fn, settings, 4, None, phase)
    assert found_key == key
    assert allocation["item_ids"] == original["item_ids"]
    assert allocation["sample_info"]["target_n"] == original["sample_info"]["target_n"] == 4
    assert not allocation["operation_completed"]


def _obsolete_key(monkeypatch, fn, settings, policy):
    with monkeypatch.context() as old:
        old.setattr(claude_judge, "judge_policy_hash", lambda **kwargs: policy)
        return tools_eval._review_key(fn, settings, 4)


def test_unrelated_policy_and_changed_request_cannot_recover_a_prior_allocation(
    tmp_path, monkeypatch,
):
    rt, fn, settings, rows = _context(tmp_path)
    key = _obsolete_key(monkeypatch, fn, settings, "0" * 64)
    allocate_review(tmp_path, "family", rows, target_n=4, seed=17,
                    review_key=key, phase="development")
    assert tools_eval._find_compatible_allocation(
        rt, fn, settings, 4, None, "development")[1] is None
    current_key = tools_eval._review_key(fn, settings, 4)
    allocate_review(tmp_path, "family", rows, target_n=4, seed=17,
                    review_key=current_key, phase="development")
    assert tools_eval._find_compatible_allocation(
        rt, fn, settings, 5, None, "development")[1] is None


def test_completed_compatible_ordinary_confirmation_is_returned_without_paid_work(
    repair_runtime, monkeypatch,
):
    rt, statuses, calls = repair_runtime
    statuses.clear()
    statuses["default"] = "supported"
    first = tools_eval.judge_function("guess", evidence_n=4, review_phase="confirmation")
    assert first["ok"], first
    assert rt.library.get("guess").scores["judge_policy_hash"] == claude_judge.judge_policy_hash()
    before = deepcopy(rt.library.get("guess").scores)
    calls.clear()

    def unexpected(**kwargs):
        pytest.fail("Completed prior review must not start paid review work")

    monkeypatch.setattr(claude_judge, "judge_behavior_with_claude", unexpected)
    repeated = tools_eval.judge_function("guess", evidence_n=4, review_phase="confirmation")
    assert repeated["ok"] and repeated["status"] == "review_already_completed", repeated
    assert rt.library.get("guess").scores == before
    assert calls == []


@pytest.mark.parametrize("profile,completed", [("behavioral", False), ("causal", True)])
def test_obsolete_policy_allocation_stays_in_history_but_is_not_resumed(
    tmp_path, monkeypatch, profile, completed,
):
    rt, fn, settings, rows = _context(tmp_path, profile)
    obsolete = {
        "behavioral": "95d4d13c94fdb469cbfc27b553468806b349043408117bc2d72b8dca7d59ced0",
        "causal": "26ff5754dbbdba019fbb069e07abb5b9228680154fd52b30e69b5c79d3486dec",
    }[profile]
    key = _obsolete_key(monkeypatch, fn, settings, obsolete)
    allocate_review(tmp_path, "family", rows, target_n=4, seed=17,
                    review_key=key, phase="development")
    if completed:
        complete_review_allocation(tmp_path, "family", review_key=key, phase="development")
    ledger = tmp_path / "review_evidence.json"
    original = ledger.read_bytes()
    assert tools_eval._find_compatible_allocation(
        rt, fn, settings, 4, None, "development",
    )[1] is None
    assert ledger.read_bytes() == original


# Synthesized claims spend the existing review budget on distinct task evidence.


def record(rt, tmp_path, *seeds):
    rt.run.run_dir = lambda: tmp_path
    # The saved contracts below deliberately exercise historical causal reviews.
    rt.config = Config(behavior_strictness="causal")
    for fn in seeds:
        contract = build_claim_contract(description=fn.description, mechanism=fn.mechanism,
                                        source=fn.source)
        fn.scores.update(judge_contract_version=JUDGE_CONTRACT_VERSION,
                         judge_policy_hash=claude_judge.judge_policy_hash(),
                         claim_hash=contract["contract_id"], claim_contract=contract,
                         behavior_verdicts=[])
        for item_id in fn.failure_localisations:
            initial = {"item_id": item_id, "snapshot_id": "snapshot-" + item_id,
                       "claim_contract_id": contract["contract_id"], "verdict": "yes"}
            fn.scores["behavior_verdicts"].append({
                **initial, "verification_status": "confirmed", "initial_review": initial,
            })
    survey = {"family_name": "parent", "seed_functions": [
        {"name": fn.name, "claim_hash": fn.scores["evaluation_claim_hash"],
         "support_item_ids": list(fn.failure_localisations)[:1]} for fn in seeds
    ]}
    (tmp_path / "family_discovery.json").write_text(json.dumps({"survey": survey}))
    return survey


def priorities(rt, tmp_path):
    record(rt, tmp_path, detector(rt, "left", ["a1"]), detector(rt, "right", ["a2"]))
    return synthesis_review_strata(rt, "parent", {row["_item_id"] for row in rt.dataset.rows})


def test_seed_repeats_are_retention_not_new_tasks_and_unknowns_never_invent_tasks(rt, tmp_path):
    rt.dataset.rows.append({"_item_id": "opaque-repeat-999", "is_correct": False})
    record(rt, tmp_path, detector(rt, "left", ["a1"], n_items=5),
           detector(rt, "right", ["b1"], n_items=5))
    result = synthesis_review_strata(rt, "parent", {"a1", "a2", "b1", "opaque-repeat-999"})
    assert result["baseline_scope"] == "complete_seed_task_inventory"
    assert result["strata"]["retained_seed_matches"] == ["a1", "b1"]
    assert result["strata"]["new_task_matches"] == []
    assert rt.dataset.reads == 1


def test_synthesis_sampling_uses_explicit_detector_link(rt, tmp_path):
    survey = record(rt, tmp_path, detector(rt, "left", ["a1"]), detector(rt, "right", ["b1"]))
    survey.update(family_name="abstract_family", detector_name="actual_parent_detector")
    (tmp_path / "family_discovery.json").write_text(json.dumps({"survey": survey}))
    result = synthesis_review_strata(rt, "actual_parent_detector", {"a1", "a2", "b1"})
    assert result["baseline_scope"] == "complete_seed_task_inventory"
    assert result["strata"]["retained_seed_matches"] == ["a1", "b1"]
    assert synthesis_review_strata(rt, "abstract_family", {"a1", "a2", "b1"}) is None


@pytest.mark.parametrize("change", ["source", "missing", "rejected", "stale_policy"])
def test_changed_or_unsupported_seed_disables_novelty_priority(rt, tmp_path, change):
    priorities(rt, tmp_path)
    seed = rt.library.get("left")
    if change == "source":
        seed.source += "\n# revised"
    elif change == "missing":
        rt.library.remove(seed.name)
    elif change == "rejected":
        seed.scores["behavior_verdicts"][0]["verdict"] = "no"
    else:
        seed.scores["judge_policy_hash"] = "previous-policy"
    result = synthesis_review_strata(rt, "parent", {"a1", "b1", "c1"})
    assert result["baseline_scope"] == "unavailable_seed_revision"
    assert result["stale_seed_names"] == ["left"]
    assert result["strata"] == {}


@pytest.mark.parametrize("state", [
    "invalid JSON", '{"survey": {"family_name": "parent", "seed_functions": [1]}}',
])
def test_malformed_optional_priority_metadata_falls_back_to_normal_review(rt, tmp_path, state):
    rt.run.run_dir = lambda: tmp_path
    (tmp_path / "family_discovery.json").write_text(state)
    result = synthesis_review_strata(rt, "parent", {"a1"})
    assert result["baseline_scope"] == "unavailable_synthesis_metadata"
    allocation = allocate_review(tmp_path, "parent", rt.dataset.rows, target_n=2, seed=37,
                                 review_key="claim", coverage_review=result)
    assert len(allocation["item_ids"]) == 2
    assert allocation["sample_info"]["coverage_review"]["reason"]


def test_incomplete_seed_baseline_is_explicitly_observed_only(rt, tmp_path):
    context = priorities(rt, tmp_path)
    assert context["baseline_scope"] == "complete_seed_task_inventory"
    rt.library.get("left").scores["sampled"] = True
    result = synthesis_review_strata(rt, "parent", {"a1", "b1", "c1"})
    assert result["baseline_scope"] == "observed_seed_matches"
    assert result["strata"]["new_task_matches"] == ["b1"]


def test_latest_lineage_is_found_after_comparison_and_replaced_by_new_top_down_record(rt, tmp_path):
    old = record(rt, tmp_path, detector(rt, "left", ["a1"]), detector(rt, "right", ["a2"]))
    path = tmp_path / "family_discovery.json"
    path.write_text(json.dumps({"history": [{"comparisons": [{"survey": old}]}]}))
    assert synthesis_review_strata(rt, "parent", {"b1"})["strata"]["new_task_matches"] == ["b1"]
    path.write_text(json.dumps({"history": [{"survey": old}], "survey": {
        "family_name": "parent", "seed_functions": [],
    }}))
    assert synthesis_review_strata(rt, "parent", {"b1"}) is None


def test_development_reviews_new_task_retained_support_and_passing_match_within_cap(rt, tmp_path):
    context = priorities(rt, tmp_path)
    # Discovery already inspected seed support. The parent's own claim still
    # needs review there, without calling that evidence independent validation.
    record_discovery_exposure(tmp_path, rt.dataset.rows[:2])
    allocation = allocate_review(tmp_path, "parent", rt.dataset.rows, target_n=3,
                                 seed=37, review_key="claim", coverage_review=context)
    selected = set(allocation["item_ids"])
    assert len(selected) == 3
    # One untouched task is reserved, so available development strata retain
    # their priority, without spending that reserved evidence early.
    reserved = set(allocation["reserved_confirmation_item_ids"])
    assert not selected & reserved
    counts = allocation["sample_info"]["coverage_review"]["strata"]
    assert counts["new_task_matches"]["selected_n"] == int("b1" not in reserved)
    assert counts["passing_matches"]["selected_n"] == int("c1" not in reserved)
    assert counts["retained_seed_matches"]["selected_n"] >= 1
    assert allocation["sample_info"]["reused_n"] >= 1
    assert allocation["sample_info"]["limited"]
    # Frozen evidence survives source ordering and priority changes on resume.
    again = allocate_review(tmp_path, "parent", list(reversed(rt.dataset.rows)), target_n=3,
                            seed=0, review_key="claim", coverage_review={"strata": {}})
    assert again == allocation


def test_task_diversity_precedes_repeats_even_with_many_seed_attempts(rt, tmp_path):
    context = priorities(rt, tmp_path)
    extra = [{"_item_id": f"new-{i}", "benchmark": "bench", "problem_id": f"new-{i}"}
             for i in range(5)]
    rows = rt.dataset.rows + extra + [
        {"_item_id": f"repeat-{i}", "benchmark": "bench", "problem_id": "a"}
        for i in range(30)
    ]
    context["strata"]["new_task_matches"] += [row["_item_id"] for row in extra]
    record_discovery_exposure(tmp_path, rows)
    allocation = allocate_review(tmp_path, "parent", rows, target_n=8, seed=37,
                                 review_key="claim", coverage_review=context)
    assert len(groups(rows, allocation["item_ids"])) == 8


def test_prior_reservations_and_task_level_confirmation_exclusions_are_preserved(rt, tmp_path):
    context = priorities(rt, tmp_path)
    record_discovery_exposure(tmp_path, [rt.dataset.rows[0]])
    dev = allocate_review(tmp_path, "parent", rt.dataset.rows, target_n=2, seed=37,
                          review_key="claim", coverage_review=context)
    confirmation = allocate_review(tmp_path, "parent", rt.dataset.rows, target_n=2, seed=37,
                                   review_key="claim", phase="confirmation",
                                   coverage_review=context)
    assert not {"a1", "a2"} & set(confirmation["item_ids"])
    assert not set(dev["item_ids"]) & set(confirmation["item_ids"])
    assert confirmation["sample_info"]["reused_n"] == 0


def test_family_judge_reviews_new_tasks_and_controls_with_proportional_budget(repair_runtime):
    rt, statuses, calls = repair_runtime
    statuses.clear()
    statuses["default"] = "supported"
    rt.dataset.items[1]["response_text"] += " left trigger"
    rt.dataset.items[2]["response_text"] += " right trigger"
    for name, task in (("left", "1"), ("right", "2"), ("parent", None)):
        source = (f"def {name}(item): return True" if task is None else
                  f"def {name}(item): return '{name} trigger' in item['response_text']")
        rt.library.upsert(FeatureFunction(name=name, source=source, description=DESCRIPTION,
                                          mechanism=MECHANISM))
    assert tools_eval.evaluate_functions(names=["left", "right", "parent"], mode="full")["ok"]
    assert tools_eval.judge_function("left")["ok"]
    assert tools_eval.judge_function("right")["ok"]
    calls.clear()
    seeds = [{"name": name, "claim_hash": rt.library.get(name).scores["evaluation_claim_hash"],
              "support_item_ids": [item_id]}
             for name, item_id in (("left", "case-1"), ("right", "case-2"))]
    state = initialize_family_discovery(rt)
    state.update(phase="assess_family", survey={
        "family_name": "parent", "description": DESCRIPTION, "mechanism": MECHANISM,
        "approach": "bottom_up", "evidence": [], "seed_functions": seeds,
    })
    path = rt.run.run_dir() / "family_discovery.json"
    path.write_text(json.dumps(state))
    record_discovery_exposure(rt.run.run_dir(), rt.dataset.items[:3])
    result = tools_eval.judge_function("parent")
    assert result["ok"], result
    scores = rt.library.get("parent").scores
    assert scores["n_flagged"] == 64
    assert scores["behavior_n_judged"] == len(set(calls)) == 7
    sample = scores["behavior_sample"]
    assert "development_cap" not in sample
    assert sample["target_n"] == sample["actual_n"] == 7
    coverage = sample["coverage_review"]
    assert coverage["baseline_scope"] == "complete_seed_task_inventory"
    assert all(row["selected_n"] == row["selected_task_n"] >= 1
               for row in coverage["strata"].values())
    assert sum(row["selected_n"] for row in coverage["strata"].values()) == 7
    assert sample["reused_n"] >= 1
    assert scores["behavior_review_phase"] == "development"


# Both harnesses record visible evidence through the shared tool invocation boundary.


def call(monkeypatch, name, args, result):
    monkeypatch.setitem(server.TOOL_CALLABLES, name, lambda **kwargs: result)
    response = asyncio.run(server.invoke_tool(name, args))
    return json.loads(response["content"][0]["text"])


@pytest.mark.parametrize("oversized", [False, True])
def test_exposure_error_preserves_committed_result_and_operation_receipt(
    tmp_path, monkeypatch, oversized,
):
    from autofa.diagnostics import Diagnostics
    from autofa.session import review_evidence

    rt = exposure_runtime(tmp_path, monkeypatch)
    diagnostics = Diagnostics()
    monkeypatch.setattr(server, "get_diagnostics", lambda: diagnostics)
    commits = []
    original = {"ok": True, "name": "f", "item_id": "opaque-0",
                "text": "visible evidence" * (1000 if oversized else 1)}

    def submit():
        rt.library.upsert(FeatureFunction(
            name="f", description="Saved detector", source="def f(item): return True",
        ))
        rt.persist_library()
        commits.append("f")
        return original

    def fail_recording(*args):
        raise TypeError("exposure write failed")  # Any recording error is contained.

    monkeypatch.setitem(server.TOOL_CALLABLES, "submit_function", submit)
    monkeypatch.setattr(review_evidence, "record_discovery_exposure", fail_recording)
    if oversized:
        monkeypatch.setattr(server, "MCP_HARD_CEILING_CHARS", 4000)
    response = asyncio.run(server.invoke_tool("submit_function", {}))
    result = json.loads(response["content"][0]["text"])
    assert result["ok"] is True
    assert result["_exposure_warning"]["code"] == "mcp_exposure_recording_failed"
    assert "do not repeat" in result["_exposure_warning"]["remediation"]
    assert "error" not in result
    if oversized:
        assert result["result_saved"] is True
    else:
        assert {key: result[key] for key in original} == original
    assert "_exposure_warning" not in original
    assert commits == ["f"]
    assert FunctionLibrary.load(rt.working_library_path()).get("f") is not None
    operation = json.loads(next((rt.run.run_dir() / "operations").glob("*.json")).read_text())
    assert operation["state"] == "complete"
    assert operation["result"] == original
    assert diagnostics.list()[0].code == "mcp_exposure_recording_failed"
    assert review_context(rt.run.run_dir(), "f")["history_complete"] is False


def test_exposure_failure_does_not_replace_existing_tool_error(tmp_path, monkeypatch):
    exposure_runtime(tmp_path, monkeypatch)
    original = {"ok": False, "error": "Original failure", "retryable": False}

    def fail_recording(*args):
        raise OSError("exposure unavailable")

    monkeypatch.setattr(server, "_record_tool_exposure", fail_recording)
    result = call(monkeypatch, "library_save", {}, original)
    assert {key: result[key] for key in original} == original
    assert result["_exposure_warning"]["code"] == "mcp_exposure_recording_failed"


def test_conflicting_exposure_identity_preserves_read_and_limits_future_confirmation(
    tmp_path, monkeypatch,
):
    from autofa.session.review_evidence import record_discovery_exposure

    rt = exposure_runtime(tmp_path, monkeypatch)
    record_discovery_exposure(rt.run.run_dir(), [
        {**rt.dataset.rows[0], "problem_id": "different-old-task"},
    ])
    original = {"ok": True, "item_id": "opaque-0", "text": "visible evidence"}
    result = call(monkeypatch, "get_item", {"item_id": "opaque-0"}, original)
    assert {key: result[key] for key in original} == original
    assert "Task identity changed" in result["_exposure_warning"]["context"]["error"]
    confirmation = allocate_review(rt.run.run_dir(), "f", rt.dataset.rows, target_n=2,
                                   seed=0, review_key="v1", phase="confirmation")
    assert confirmation["sample_info"]["history_complete"] is False
    assert confirmation["sample_info"]["limited"] is True


def test_failed_exposure_ledger_write_stays_incomplete_until_durable_and_portable(
    tmp_path, monkeypatch,
):
    from autofa.session import review_evidence

    rt = exposure_runtime(tmp_path, monkeypatch)
    rt.persist_library()
    original = {"ok": True, "item_id": "opaque-0", "text": "visible evidence"}

    def fail_write(*args):
        raise OSError("ledger is temporarily read-only")

    with monkeypatch.context() as patch:
        patch.setattr(review_evidence, "atomic_write_json", fail_write)
        result = call(patch, "get_item", {"item_id": "opaque-0"}, original)
        assert result["ok"] is True
        assert result["_exposure_warning"]["code"] == "mcp_exposure_recording_failed"
        assert review_context(rt.run.run_dir(), "f")["history_complete"] is False

    # After storage recovers, a new allocation remains limited and commits the
    # incomplete history, which also survives export into another run.
    confirmation = allocate_review(rt.run.run_dir(), "f", rt.dataset.rows, target_n=2,
                                   seed=0, review_key="v1", phase="confirmation")
    assert confirmation["sample_info"]["history_complete"] is False
    assert confirmation["sample_info"]["limited"] is True
    assert rt.run.run_dir().resolve() not in review_evidence._INCOMPLETE_RUNS
    assert review_context(rt.run.run_dir(), "f")["history_complete"] is False
    rt.persist_library()
    imported = FunctionLibrary.load(rt.working_library_path())
    review_evidence.sync_review_history(imported, tmp_path / "continued")
    assert review_context(tmp_path / "continued", "f")["history_complete"] is False


def test_read_arguments_resolve_metadata_once_and_exclude_other_attempts(tmp_path, monkeypatch):
    rt = exposure_runtime(tmp_path, monkeypatch)
    result = {"ok": True, "item_id": "opaque-0", "text": "visible evidence"}
    assert call(monkeypatch, "read_field_chunk", {"item_id": "opaque-0"}, result)["ok"]
    assert call(monkeypatch, "read_field_chunk", {"item_id": "opaque-0"}, result)["ok"]
    assert rt.dataset.lookups == 1
    assert not {"response_text", "trace_json", "messages"} & set(rt.dataset.columns)
    confirmation = allocate_review(rt.run.run_dir(), "f", rt.dataset.rows, target_n=10,
                                   seed=0, review_key="v1", phase="confirmation")
    assert not {"opaque-0", "opaque-1"} & set(confirmation["item_ids"])
    assert confirmation["sample_info"]["unknown_exposure_item_n"] == 0


def test_sample_rows_and_reviewed_examples_count_but_bucket_id_lists_do_not(tmp_path, monkeypatch):
    rt = exposure_runtime(tmp_path, monkeypatch)
    visible = {**rt.dataset.rows[0], "response_text": "cause"}
    result = {"ok": True, "items": [visible], "verdicts": [
        {"item_id": "opaque-2", "verdict": "no", "rationale": "Recovered"},
    ], "buckets": [{"example_fail_ids": ["opaque-4", "opaque-5"]}]}
    assert call(monkeypatch, "sample_items", {}, result)["ok"]
    assert review_context(rt.run.run_dir(), "f")["reviewed_item_ids"] == ["opaque-0", "opaque-2"]
    assert rt.dataset.lookups == 1  # One metadata-only inventory supplies both identities.


def test_library_membership_alone_does_not_consume_holdout(tmp_path, monkeypatch):
    rt = exposure_runtime(tmp_path, monkeypatch)
    result = {"ok": True, "functions": [{"name": "family", "example_fail_ids": ["opaque-0"],
                                         "buckets": [{"item_ids": ["opaque-1"]}]}]}
    assert call(monkeypatch, "list_library_functions", {}, result)["ok"]
    assert review_context(rt.run.run_dir(), "f")["reviewed_item_ids"] == []
    assert rt.dataset.lookups == 0


def test_nested_localisation_content_inherits_enclosing_source_identity(tmp_path, monkeypatch):
    rt = exposure_runtime(tmp_path, monkeypatch)
    result = {"ok": True, "function": {"name": "f", "failure_localisations": {
        "opaque-0": {"item_id": "opaque-0", "problem_id": "display-alias",
                     "passages": [{"context": "visible evidence"}]},
        "opaque-2": {"item_id": "opaque-2", "passages": [{"quote": "visible trace evidence"}]},
        "opaque-4": {"item_id": "opaque-4", "passages": [{"quote": "  ", "context": ""}]},
    }, "example_fail_ids": ["opaque-6", "opaque-7"]}}
    assert call(monkeypatch, "get_library_function", {"name": "f"}, result)["ok"]
    context = review_context(rt.run.run_dir(), "f")
    assert context["reviewed_item_ids"] == ["opaque-0", "opaque-2"]
    assert context["reviewed_task_groups"] == ['["b","0"]', '["b","1"]']
    confirmation = allocate_review(rt.run.run_dir(), "f", rt.dataset.rows, target_n=10,
                                   seed=0, review_key="v1", phase="confirmation")
    assert not {f"opaque-{i}" for i in range(4)} & set(confirmation["item_ids"])


def test_default_library_preview_consumes_only_five_shown_localisations(tmp_path, monkeypatch):
    from autofa.mcp import tools_library

    rt = exposure_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(tools_library, "_family_search_status", lambda _: None)
    ids = [f"opaque-{i}" for i in (0, 2, 4, 6, 8, 10)]
    fn = FeatureFunction(name="f", description="A cause", source="def f(item): return True",
                         failure_localisations={
                             item_id: {"item_id": item_id,
                                       "passages": [{"quote": "shown trace evidence"}]}
                             for item_id in ids
                         }, example_fail_ids=ids)
    rt.library.upsert(fn)
    response = asyncio.run(server.invoke_tool("get_library_function", {"name": "f"}))
    result = json.loads(response["content"][0]["text"])
    assert result["ok"] and len(result["function"]["failure_localisations"]) == 5
    assert set(review_context(rt.run.run_dir(), "f")["reviewed_item_ids"]) == set(ids[:5])
    confirmation = allocate_review(rt.run.run_dir(), "f", rt.dataset.rows, target_n=1,
                                   seed=0, review_key="v1", phase="confirmation")
    assert set(confirmation["item_ids"]) <= {"opaque-10", "opaque-11"}


def test_unresolvable_exposure_cannot_establish_independent_confirmation(tmp_path, monkeypatch):
    rt = exposure_runtime(tmp_path, monkeypatch)
    assert call(monkeypatch, "get_item", {"item_id": "missing-id"},
                {"ok": True, "item_id": "missing-id", "text": "visible"})["ok"]
    confirmation = allocate_review(rt.run.run_dir(), "f", rt.dataset.rows, target_n=2,
                                   seed=0, review_key="v1", phase="confirmation")
    assert confirmation["sample_info"]["limited"]
    assert confirmation["sample_info"]["unknown_exposure_item_n"] == 1


def test_legacy_sessions_do_not_enable_reservations_implicitly(tmp_path, monkeypatch):
    rt = exposure_runtime(tmp_path, monkeypatch, family=False)
    assert call(monkeypatch, "get_item", {"item_id": "opaque-0"},
                {"ok": True, "item": {**rt.dataset.rows[0], "response_text": "visible"}})["ok"]
    assert not (rt.run.run_dir() / "review_evidence.json").exists()
    assert rt.dataset.lookups == 0


def test_random_smoke_records_all_results_including_errors_and_abstentions(tmp_path, monkeypatch):
    rt = exposure_runtime(tmp_path, monkeypatch)
    results = [
        {"item_id": row["_item_id"], "status": "value", "value": False}
        for row in rt.dataset.rows
    ]
    results[-2] = {"item_id": "opaque-10", "status": "error", "error": "execution failed"}
    results[-1] = {"item_id": "opaque-11", "status": "abstained", "reason": "missing context"}
    result = {"ok": True, "results": results,
              "sample_ids": [row["item_id"] for row in results[:10]]}
    assert call(monkeypatch, "smoke_test_feature", {"n": 12}, result)["ok"]
    assert set(review_context(rt.run.run_dir(), "f")["reviewed_item_ids"]) == {
        row["_item_id"] for row in rt.dataset.rows
    }
    assert rt.dataset.lookups == 1


def real_runtime(tmp_path, monkeypatch, *, genuine_tasks):
    rt = exposure_runtime(tmp_path, monkeypatch)
    path = tmp_path / "task-identities.parquet"
    rows = []
    for index, question in enumerate(("same:question", "same:question", "other", "boundary")):
        row = {"item_id": f"opaque_{index}", "benchmark": "bench:variant", "model_id": "m",
               "is_correct": index == 3, "response_text": "Required action rejected."}
        if genuine_tasks:
            row["question_id"] = question
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), path)
    rt.dataset = Dataset(parquet_paths=[path])
    return rt, rt.dataset.load_items()


@pytest.mark.parametrize("genuine_tasks", [True, False])
def test_real_mcp_family_evidence_and_wrapper_share_source_identity(
    tmp_path, monkeypatch, genuine_tasks,
):
    rt, loaded = real_runtime(tmp_path, monkeypatch, genuine_tasks=genuine_tasks)
    (rt.run.run_dir() / "family_discovery.json").unlink()
    evidence = [{
        "item_id": item["_item_id"], "role": "boundary" if item["is_correct"] else "support",
        "causal_chain": "Rejected prerequisite left a required output absent.",
        "quotes": [{"field": "response_text", "quote": "Required action rejected."}],
    } for item in (loaded[0], loaded[2], loaded[3])]
    response = asyncio.run(server.invoke_tool("family_discovery", {
        "action": "record_family", "family_name": "required_action_rejected",
        "description": "Required actions remain rejected", "mechanism": "Rejection blocks output",
        "boundary_notes": "Exclude successful recovery", "evidence": evidence,
        "scope_limit": "Only opaque attempts are known" if not genuine_tasks else "",
    }))
    result = json.loads(response["content"][0]["text"])
    assert result["ok"], result
    context = review_context(rt.run.run_dir(), "f")
    assert len(context["reviewed_item_ids"]) == 3
    if genuine_tasks:
        assert task_group({"benchmark": "bench:variant", "question_id": "same:question"}) in (
            context["reviewed_task_groups"]
        )
    else:
        assert context["reviewed_task_groups"] == []


@pytest.mark.parametrize("genuine_tasks", [True, False])
def test_real_mcp_exposure_rejects_normalized_opaque_aliases(tmp_path, monkeypatch, genuine_tasks):
    rt, loaded = real_runtime(tmp_path, monkeypatch, genuine_tasks=genuine_tasks)
    item_id = loaded[0]["_item_id"]
    assert call(monkeypatch, "get_item", {"item_id": item_id},
                {"ok": True, "items": [loaded[0]]})["ok"]
    context = review_context(rt.run.run_dir(), "f")
    assert context["reviewed_item_ids"] == [item_id]
    verified = [verified_identity_projection(rt, item) for item in loaded]
    allocation = allocate_review(rt.run.run_dir(), "f", verified, target_n=1,
                                 seed=0, review_key="v1", phase="confirmation")
    if genuine_tasks:
        assert context["reviewed_task_groups"] == ['["bench:variant","same:question"]']
        assert not {loaded[0]["_item_id"], loaded[1]["_item_id"]} & set(allocation["item_ids"])
    else:
        assert context["reviewed_task_groups"] == []
        assert allocation["sample_info"]["independent_task_n"] == 0
        assert allocation["sample_info"]["limited"]
