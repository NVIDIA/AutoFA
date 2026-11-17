# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family, library and search-ranking coverage detectors."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config
from autofa.core.dependencies import review_source
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.localisation import text_hash
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.search_panel import build_search_panel
from autofa.metrics.task_recall import build_task_recall
from autofa.metrics.task_support import build_task_support
from autofa.objectives import discovery_strategy
from autofa.session.family_coverage import failed_task_inventory

from support.task_support import attach_task_support


# Family coverage.
class MetadataDataset:
    parquet_paths = []

    def __init__(self, rows):
        self.rows = rows
        self.reads = 0

    def iter_batches(self, *, columns, max_items):
        assert max_items is None
        assert not {"response_text", "trace", "raw_row_json"} & set(columns)
        self.reads += 1
        yield self.rows


@pytest.fixture
def rt():
    rows = [
        {"_item_id": "a1", "benchmark": "bench", "problem_id": "a", "is_correct": False},
        {"_item_id": "a2", "benchmark": "bench", "problem_id": "a", "is_correct": False},
        {"_item_id": "b1", "benchmark": "bench", "problem_id": "b", "is_correct": False},
        {"_item_id": "c1", "benchmark": "bench", "problem_id": "c", "is_correct": True},
    ]
    return SimpleNamespace(dataset=MetadataDataset(rows), run=SimpleNamespace(run_id="r1"),
                           library=FunctionLibrary())


def detector(rt, name, failed_ids, *, pass_ids=(), **scores):
    fn = FeatureFunction(name=name, description="A mechanism", mechanism="A causal defect",
                         source=f"def {name}(item): return True")
    fn.scores = {
        "evaluated": True, "evaluation_run_id": rt.run.run_id,
        "evaluation_claim_hash": claim_hash(source=fn.source, description=fn.description,
                                            mechanism=fn.mechanism),
        "evaluation_scope_complete": True, "sampled": False, "bucket": "True",
        "n_items": 4.0, "n_flagged": float(len(failed_ids) + len(pass_ids)),
        "n_fail_caught": float(len(failed_ids)), "recall": len(failed_ids) / 3,
        **scores,
    }
    fn.failure_localisations = {
        item_id: {"item_id": item_id, "assigned_label": "True",
                  "source_sha256": text_hash(fn.source)} for item_id in [*failed_ids, *pass_ids]
    }
    rt.library.upsert(fn)
    return fn


# Library coverage.
def reviewed_detector(name, matches=(), confirmed=(), *, parent=None, strategy="balanced"):
    fn = FeatureFunction(name=name, description="Invalid calls prevent the required action",
                         mechanism="A rejected call leaves the required operation undone",
                         source=f"def {name}(item): return True", explains=parent)
    contract = build_claim_contract(description=fn.description, mechanism=fn.mechanism,
                                    source=fn.source)
    fn.scores = {
        "evaluated": True, "evaluation_run_id": "r1",
        "evaluation_claim_hash": contract["contract_id"],
        "evaluation_scope_complete": True,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(),
        "judge_objective_strategy": strategy,
        "claim_hash": contract["contract_id"], "claim_contract": contract,
        "behavior_verdicts": [],
    }
    for item_id in matches:
        fn.failure_localisations[item_id] = {
            "item_id": item_id, "source_sha256": text_hash(fn.source),
            "assigned_label": "True", "passages": [], "status": "unavailable",
        }
    for item_id in confirmed:
        initial = {"item_id": item_id, "snapshot_id": "snapshot-" + item_id,
                   "claim_contract_id": contract["contract_id"], "verdict": "yes"}
        fn.scores["behavior_verdicts"].append({
            **initial, "verification_status": "confirmed", "initial_review": dict(initial),
        })
    return fn


# Library quality coverage.
def corpus(*, repeats=1, unknown_task=False, unknown_outcome=False):
    rows = [
        {"_item_id": f"f-{task}-{repeat}", "problem_id": str(task), "benchmark": "bench",
         "is_correct": False}
        for task in range(20) for repeat in range(repeats)
    ]
    rows.extend({"_item_id": f"p-{task}", "problem_id": f"p{task}",
                 "benchmark": "bench", "is_correct": True} for task in range(5))
    if unknown_task:
        rows[0].pop("problem_id")
    if unknown_outcome:
        rows.append({"_item_id": "unknown", "benchmark": "bench", "problem_id": "unknown"})
    return failed_task_inventory(SimpleNamespace(load_items=lambda: rows))


def coverage_detector(name, inventory, matches, *, coverage=.3, precision=.9, adherence=1.0):
    fn = attach_task_support(FeatureFunction(
        name=name, source=f"def {name}(item): return True",
        description="Required work is abandoned",
        mechanism="Abandoned work leaves the result missing",
        scores={"evaluated": True, "recall": coverage, "ppv_at_pi0": precision,
                "behavior_purity": adherence},
    ))
    fn.failure_localisations = {
        item_id: {"item_id": item_id, "source_sha256": text_hash(fn.source),
                  "assigned_label": True}
        for item_id in matches
    }
    # Exactly one judge-confirmed example: match coverage must still include all
    # matches, because aggregate adherence and breadth measure different things.
    reviewed_id = matches[0]
    review = deepcopy(fn.scores["behavior_verdicts"][0])
    review["item_id"] = review["initial_review"]["item_id"] = reviewed_id
    fn.scores.update(
        behavior_verdicts=[review], behavior_n_judged=1, behavior_n_completed=1,
        behavior_n_yes=1, sampled=False, evaluation_scope_complete=True,
        n_items=float(inventory["n_items"]), n_scored=inventory["n_items"],
        n_flagged=float(len(matches)),
        n_fail_caught=float(len(set(matches) & inventory["task_by_failed_item_id"].keys())),
        task_recall_corpus_fingerprint=inventory["corpus_fingerprint"],
    )
    fn.scores.update(build_task_support(fn, inventory))
    return fn


# Search ranking.
def candidate(name="narrow", *, reviewed=("yes", "yes", "uncertain"), config=None,
              matched=("a1", "a2"), reused=0, panel_ids=None):
    config = config or Config(behavior_strictness="causal")
    fn = FeatureFunction(name=name, description="Required action is bypassed",
                         mechanism="A required operation is replaced by a fixture lookup.",
                         source=f"def {name}(item): return True")
    contract = build_claim_contract(source=review_source(fn), description=fn.description,
                                    mechanism=fn.mechanism)
    fingerprint = contract["contract_id"]
    rows = []
    for index, verdict in enumerate(reviewed):
        row = {"item_id": f"review_{index}", "snapshot_id": f"snapshot-{index}",
               "claim_contract_id": fingerprint, "verdict": verdict,
               "completion_status": "complete"}
        if verdict == "yes":
            row.update(verification_status="confirmed", initial_review=dict(row))
        rows.append(row)
    fn.scores = {
        "evaluated": True, "evaluation_claim_hash": fingerprint, "evaluation_run_id": "r1",
        "evaluation_scope_complete": True, "evaluation_complete": True, "sampled": False,
        "n_items": 6, "n_scored": 6, "n_flagged": len(matched),
        "n_fail_caught": len(matched), "ppv_at_pi0": 1., "recall": len(matched) / 5,
        "behavior_purity": reviewed.count("yes") / len(reviewed),
        "claim_hash": fingerprint, "claim_contract": contract,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(),
        "judge_objective_strategy": discovery_strategy(config),
        "behavior_n_judged": len(reviewed), "behavior_n_yes": reviewed.count("yes"),
        "behavior_n_no": reviewed.count("no"), "behavior_n_uncertain": reviewed.count("uncertain"),
        "behavior_n_errors": 0, "behavior_n_completed": len(reviewed) - reviewed.count("uncertain"),
        "behavior_verdicts": rows,
        "behavior_sample": {"phase": "development", "limited": True, "reused_n": reused,
                            "actual_n": len(reviewed), "target_n": 3},
    }
    fn.failure_localisations = {
        item: {"item_id": item, "assigned_label": "True", "source_sha256": text_hash(fn.source)}
        for item in matched
    }
    inventory = {
        "corpus_fingerprint": "corpus", "inventory_complete": True, "n_items": 6,
        "n_failed_attempts": 5, "n_failed_tasks": 4, "n_unknown_outcomes": 0,
        "item_ids": ["a1", "a2", "b1", "c1", "d1", "p1"],
        "task_by_failed_item_id": {"a1": "a", "a2": "a", "b1": "b", "c1": "c", "d1": "d"},
    }
    fn.scores.update(build_task_recall(fn, inventory, run_id="r1"))
    # A census evaluation freezes its scoring panel over every corpus item.
    fn.scores["search_panel"] = build_search_panel(
        fn, inventory, item_ids=inventory["item_ids"] if panel_ids is None else panel_ids,
        run_id="r1",
    )
    return fn
