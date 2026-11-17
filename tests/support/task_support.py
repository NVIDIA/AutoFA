# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-identified task support and the detectors that quality reports render."""

import json
import shutil
import subprocess
from io import StringIO

import pytest
from rich.console import Console

from autofa.core.dependencies import review_source
from autofa.core.library import FeatureFunction
from autofa.core.localisation import text_hash
from autofa.metrics.association import contingency
from autofa.metrics.claim_contract import JUDGE_CONTRACT_VERSION, build_claim_contract, claim_hash
from autofa.metrics.claude_judge import judge_policy_hash
from autofa.metrics.task_support import build_task_support
from autofa.metrics.verdicts import verdict_js_classify_fn
from autofa.reports.failure_cards import render_html


# Source-identified task support for quality presentation integration tests.
def attach_task_support(fn, *, task_n=5, repeat_n=1):
    """Attach explicit failed matches and per-case reviews, including repeated tasks."""
    scores = fn.scores
    source = review_source(fn)
    strictness = scores.get("judge_behavior_strictness", "causal")
    contract = build_claim_contract(source=source, description=fn.description,
                                    mechanism=fn.mechanism or fn.description,
                                    behavior_strictness=strictness)
    fingerprint = claim_hash(source=source, description=fn.description,
                             mechanism=fn.mechanism or fn.description)
    tasks = {f"item-{task}-{repeat}": f"task-{task}"
             for task in range(task_n) for repeat in range(repeat_n)}
    rows = [{"item_id": item_id, "verdict": "yes", "verification_status": "confirmed",
             "completion_status": "complete", "processing_status": "completed",
             "validation_error": None,
             "claim_contract_id": contract["contract_id"], "snapshot_id": f"snapshot-{item_id}",
             "initial_review": {"item_id": item_id, "verdict": "yes",
                                "completion_status": "complete", "processing_status": "completed",
                                "validation_error": None,
                                "claim_contract_id": contract["contract_id"],
                                "snapshot_id": f"snapshot-{item_id}"}}
            for item_id in tasks]
    fn.failure_localisations = {
        item_id: {"item_id": item_id, "source_sha256": text_hash(source), "assigned_label": True}
        for item_id in tasks
    }
    scores.update(
        evaluated=True, evaluation_run_id="quality-test", evaluation_claim_hash=fingerprint,
        claim_hash=fingerprint, claim_contract=contract,
        judge_contract_version=JUDGE_CONTRACT_VERSION,
        judge_policy_hash=judge_policy_hash(behavior_strictness=strictness),
        behavior_verdicts=rows, task_recall_corpus_fingerprint="quality-test-corpus",
        n_matched_failed_tasks=task_n, n_matched_failed_attempts=len(tasks),
    )
    scores.update(build_task_support(fn, {
        "inventory_complete": True, "corpus_fingerprint": "quality-test-corpus",
        "task_by_failed_item_id": tasks,
    }))
    return fn


def attach_sampled_task_support(fn, *, task_n=2, full_task_n=20, panel_task_n=5):
    """Source-backed stratified sample: one attempt per failed task plus one pass."""
    assert 0 < task_n <= panel_task_n < full_task_n
    attach_task_support(fn, task_n=task_n)
    tasks = {f"item-{task}-0": f"task-{task}" for task in range(full_task_n)}
    panel = [f"item-{task}-0" for task in range(panel_task_n)] + ["passing-item"]
    item_ids = [*tasks, "passing-item"]
    inventory = {
        "inventory_complete": True, "corpus_fingerprint": "quality-test-corpus",
        "item_ids": item_ids, "n_items": len(item_ids),
        "n_failed_attempts": full_task_n, "n_failed_tasks": full_task_n,
        "n_passed_attempts": 1, "n_unknown_outcomes": 0,
        "task_by_failed_item_id": tasks,
        "identity_by_item_id": {item_id: {"_item_id": item_id, "benchmark": "bench"}
                                for item_id in item_ids},
    }
    fn.scores.update(
        sampled=True, sample_reason="llm_detector", sample_n=len(panel),
        sample_requested_n=len(panel),
        sample_frame_n=len(item_ids), evaluation_item_ids=panel,
        evaluation_scope_complete=True, eval_truncated=False,
        n_scored=len(panel), n_unresolved=0,
    )
    fn.scores.update(build_task_support(fn, inventory))
    return fn


# Verdict CLI.
def _function(state):
    fn = FeatureFunction(name="example", source="def detect(item): return True",
                         description="Observable decision", mechanism="Observable decision")
    fn.scores = {
        "evaluated": True, "n_items": 100, "n_scored": 100,
        "recall": .4, "precision": .9, "ppv_at_pi0": .9, "pct_caught": .3,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_policy_hash": judge_policy_hash(),
        "claim_hash": claim_hash(
            source=fn.source, description=fn.description, mechanism=fn.mechanism,
        ),
        "behavior_n_judged": 5, "behavior_n_completed": 5,
        "behavior_n_yes": 5, "behavior_n_no": 0,
        "behavior_n_uncertain": 0, "behavior_n_errors": 0,
        "behavior_purity": 1.0,
    }
    attach_task_support(fn)
    if state == "good_coverage":
        fn.scores["recall"] = .0101
    elif state == "passing_matches":
        fn.scores.update(recall=0, precision=0, ppv_at_pi0=0)
    elif state == "broad_small_review":
        attach_task_support(fn, task_n=23)
        fn.scores["behavior_verdicts"] = fn.scores["behavior_verdicts"][:2]
        fn.scores.update(behavior_n_judged=2, behavior_n_completed=2, behavior_n_yes=2)
    elif state == "historical_review":
        fn.scores["judge_contract_version"] = "obsolete"
    elif state == "sampled_support":
        attach_sampled_task_support(fn)
        fn.scores.update(behavior_n_judged=2, behavior_n_completed=2, behavior_n_yes=2)
    elif state == "sampled_matches_without_confirmations":
        attach_sampled_task_support(fn, task_n=3)
        fn.scores.update(recall=.13, behavior_n_judged=3,
                         behavior_n_completed=3, behavior_n_yes=3)
        for row in fn.scores["behavior_verdicts"]:
            row["verification_status"] = "unverified"
    elif state == "full_matches_with_two_confirmations":
        attach_task_support(fn, task_n=15)
        fn.scores["behavior_verdicts"] = fn.scores["behavior_verdicts"][:2]
        fn.scores.update(recall=.115, sampled=False, evaluation_scope_complete=True,
                         behavior_n_judged=2, behavior_n_completed=2, behavior_n_yes=2)
    elif state == "unknown_tasks":
        del fn.scores["task_support"]
    elif state == "pending_precision":
        del fn.scores["ppv_at_pi0"]
    elif state == "pending_review":
        fn.scores = {key: value for key, value in fn.scores.items()
                     if not key.startswith(("behavior_", "judge_", "claim_"))}
    elif state in {"diagnostic_only", "diagnostic_empty", "diagnostic_stale"}:
        fn.scores.update(
            behavior_n_judged=0, behavior_n_completed=0, behavior_n_yes=0,
            behavior_n_no=0, behavior_n_uncertain=0, behavior_n_errors=0,
            behavior_purity=None, behavior_verdicts=[],
            judge_model="offline-model", judge_harness="claude",
            diagnostic_evidence_stale=state == "diagnostic_stale",
        )
        fn.scores["diagnostic_review"] = {
            **{key: fn.scores[key] for key in (
                "judge_contract_version", "claim_hash", "judge_policy_hash",
                "judge_model", "judge_harness",
            )},
            "n_judged": 3, "n_completed": 3, "n_yes": 3, "n_no": 0, "n_uncertain": 0,
            "behavior_purity": 1., "excluded_from_adherence": True,
            "excluded_from_confirmed_support": True,
            "verdicts": [{"item_id": f"control-{i}", "verdict": "yes"} for i in range(3)],
        }
        if state == "diagnostic_empty":
            fn.scores.update(n_flagged=0, recall=0, precision=0, ppv_at_pi0=0, pct_caught=0)
    elif state in {"sampled_review_skipped", "missing_strata"}:
        fn.scores = {
            "evaluated": True, "n_items": 3713.0, "n_scored": 200,
            "sampled": True, "sample_n": 200, "sample_frame_n": 3713,
            "evaluation_scope_complete": True, "evaluation_complete": True,
            "n_rows_scored": 1, "n_flagged": 18.565,
            "bucket": "True", "bucket_mode": "exact",
            "recall": 0.0, "precision": 0.0, "ppv_at_pi0": 0.0, "pct_caught": .005,
            "judge_prefilter": {
                "status": "skip", "reason": "Saved coverage gate skipped review.",
            },
        }
        if state == "missing_strata":
            fn.scores.update(
                metrics_scope="unrepresented_strata",
                sampling_limitation="Population rates unavailable: missing benchmark|fail.",
                sample_unrepresented_strata=["benchmark|fail"],
                evaluation_complete=False, sample_n=1, n_scored=1,
                a=0, b=3151, c=0, d=0, n_items=3151., n_flagged=3151.,
                # Legacy fallbacks must not reconstruct invalid population rates
                # from partial-stratum counts or a retained bucket purity.
                recall=None, precision=None, raw_ppv=None, ppv_at_pi0=None,
                pct_caught=None, purity=0.,
            )
            fn.scores.pop("judge_prefilter")
    elif state == "inconclusive_review":
        fn.scores.update(
            behavior_purity=None, behavior_n_judged=1, behavior_n_completed=0,
            behavior_n_yes=0, behavior_n_no=0, behavior_n_errors=0, behavior_n_uncertain=1,
            behavior_sample={"target_n": 3, "actual_n": 1, "limited": True},
            behavior_verdicts=[{
                "verdict": "uncertain", "completion_status": "complete",
                "claim_assessments": [{
                    "requirement_id": "failure_contribution", "status": "unknown",
                    "unknown_reason": "ambiguous_evidence",
                }],
            }],
        )
    elif state == "review":
        fn.description = "Changed decision claim"
    elif state == "weak":
        fn.scores.update(behavior_purity=.25, behavior_n_yes=1, behavior_n_no=3)
    elif state == "empty":
        fn.scores.update(recall=0, precision=0, pct_caught=0)
    elif state == "incomplete":
        fn.scores.update(n_scored=0, errors=100, recall=0, precision=0, pct_caught=0)
    elif state == "partial_errors":
        fn.scores.update(n_scored=99, errors=1, evaluation_complete=False,
                         metrics_scope="resolved_items")
    elif state == "stale_evaluation":
        fn.scores["evaluation_stale"] = True
    elif state == "unscored":
        fn.scores = {}
    return fn


def _console():
    return Console(file=StringIO(), width=400, height=60, force_terminal=True,
                   color_system="truecolor", record=True)


# Embedded report renderers, executed in Node.
def run_node(script):
    """Run ``script`` in Node and return its stdout; skip the test when Node is unavailable."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the embedded report renderer")
    return subprocess.run([node], input=script, check=True, capture_output=True, text=True).stdout


def report_renderers(html):
    """Report card renderers with the minimal browser globals they read."""
    return ("const STATE = {selection: null}; const document = {addEventListener() {}};\n"
            + verdict_js_classify_fn()
            + html[html.index("function escapeHtml("):html.index("function boot(")])


def _render_card(data):
    # Execute the real detail and bucket renderers as well as the summary card.
    return run_node(report_renderers(render_html(data))
                    + "console.log(renderCard(" + json.dumps(data["functions"][0]) + ", 1, 1));")


# Task-supported detectors with given recall, precision and purity, for ordering tests.
def _ranked_function(name, recall, ppv, purity):
    fn = FeatureFunction(name=name, source="def detect(item): return True",
                         description="Synthetic mechanism", mechanism="Recorded causal action")
    fn.scores = {
        **contingency(recall * 100, 10, (1 - recall) * 100, 90),
        "evaluated": True, "n_scored": 200, "ppv_at_pi0": ppv,
        "behavior_purity": purity, "behavior_specificity": 1,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "judge_objective_strategy": "recall", "judge_policy_hash": judge_policy_hash(),
        "claim_hash": claim_hash(source=fn.source, description=fn.description,
                                  mechanism=fn.mechanism),
        "behavior_n_judged": 10, "behavior_n_yes": round(purity * 10),
        "behavior_n_no": 10 - round(purity * 10),
    }
    return attach_task_support(fn)
