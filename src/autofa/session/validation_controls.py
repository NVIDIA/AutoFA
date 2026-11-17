# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed validation examples for diagnostic repairs, never population scoring."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.core.localisation import text_hash
from autofa.defaults import default_value, setting_value
from autofa.metrics.association import matches_selected_bucket, selected_bucket
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.verdicts import behavior_validation, current_behavior_scores
from autofa.session.repair_workflow import _version
from autofa.session.smoke_work import LLM_ROWS


def polishing_controls_identity(controls: dict) -> str:
    """Bind both retained cases and the unflagged-success reservoir."""
    from autofa.metrics.judge_checkpoints import fingerprint

    keys = ["source_signature", "panel_item_ids", "controls", "run_id", "unflagged_success_pool"]
    return fingerprint({key: controls.get(key) for key in keys})


def validation_controls(  # noqa: C901
    rt: Any, fn: Any, *, scores: dict | None = None, limit: int | None = None,
) -> dict[str, Any]:
    """Select bounded, task-diverse controls from the current validation cohort.

    Read saved predictions and cached source identities only. No evidence fetch,
    detector execution, invented expectations, or modifications to score panels.
    Retain up to half the configured discovery sample, rounded down (minimum one).
    Census mode has no finite sample budget, so use half the packaged default.
    """
    if limit is None:
        sample = int(setting_value(rt.config, "llm_eval_sample") or 0)
        limit = max(1, (sample if sample > 0 else default_value("llm_eval_sample")) // 2)
    if type(limit) is not int or limit < 1:
        raise ValueError("Validation controls limit must be a positive integer")
    source = review_source(fn)
    claim = claim_hash(source=source, description=fn.description or "",
                       mechanism=fn.mechanism or fn.description or fn.name)
    scores = current_behavior_scores(
        fn.scores if scores is None else scores, source=source,
        description=fn.description or "", mechanism=fn.mechanism or fn.description or fn.name,
        behavior_strictness=strictness_for_config(rt.config),
    )
    result = {
        "available": False, "scope": "targeted_regression_controls",
        "name": fn.name, "claim_hash": claim, "version_hash": _version(fn.to_dict()),
        "controls": [], "item_ids": [], "limit": limit,
        "feedback_action": {"name": "get_function_feedback", "arguments": {"name": fn.name}},
    }

    def unavailable(reason: str) -> dict[str, Any]:
        return {**result, "reason": reason}

    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or scores.get("evaluation_claim_hash") != claim
            or scores.get("evaluation_run_id") != rt.run.run_id):
        return unavailable("The current revision has no current validation predictions.")
    inventory = getattr(rt, "_family_failed_task_inventory", None)
    if (not isinstance(inventory, Mapping) or inventory.get("inventory_complete") is not True
            or not inventory.get("corpus_fingerprint")
            or inventory["corpus_fingerprint"] != scores.get("task_recall_corpus_fingerprint")):
        return unavailable("Source-verified validation task identities are unavailable.")
    source_ids = set(inventory.get("item_ids") or [])
    if scores.get("sampled"):
        receipt = scores.get("round_validation") or {}
        ids = scores.get("evaluation_item_ids")
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or receipt.get("version") != 1 or receipt.get("complete") is not True
                or receipt.get("allocation") != "proportional_benchmark_outcome"
                or receipt.get("corpus_fingerprint") != inventory["corpus_fingerprint"]
                or receipt.get("panel_fingerprint") != hashlib.sha256(
                    json.dumps(sorted(ids)).encode()).hexdigest()):
            return unavailable("The completed round-validation cohort is unavailable.")
        cohort = set(ids)
        cohort_hash = receipt["panel_fingerprint"]
        cohort_scope = "round_checkpoint"
    elif scores.get("sampled") is False and scores.get("evaluation_scope_complete") is True:
        cohort = source_ids
        cohort_hash = inventory["corpus_fingerprint"]
        cohort_scope = "full_dataset"
    else:
        return unavailable("These predictions have not completed round or full validation.")
    if not cohort <= source_ids:
        return unavailable("Validation item IDs do not match the current source inventory.")
    result["cohort"] = {"scope": cohort_scope, "fingerprint": cohort_hash,
                        "corpus_fingerprint": inventory["corpus_fingerprint"], "n": len(cohort)}
    tasks = inventory.get("task_by_item_id") or {}
    failed = inventory.get("task_by_failed_item_id") or {}
    passed = set(inventory.get("passed_item_ids") or [])
    groups: dict[str, list[str]] = {}
    categories: dict[str, list[str]] = {}

    def add(item_id: Any, category: str) -> None:
        if not isinstance(item_id, str) or item_id not in cohort:
            return
        groups.setdefault(category, []).append(item_id)
        if category not in categories.setdefault(item_id, []):
            categories[item_id].append(category)

    for row in scores.get("error_examples") or []:
        if isinstance(row, Mapping):
            add(row.get("item_id"), "execution_error")
    contract_id = (scores.get("claim_contract") or {}).get("contract_id")
    review_status = behavior_validation(scores, include_details=False)["status"]
    if contract_id and review_status not in {"stale", "legacy", "unjudged", "invalid"}:
        for row in scores.get("behavior_verdicts") or []:
            if (not isinstance(row, Mapping) or row.get("claim_contract_id") != contract_id
                    or row.get("historical_review")
                    or row.get("verification_status") == "historical"):
                continue
            if row.get("completion_status") == "needs_evidence_review":
                category = "incomplete_review"
            else:
                category = {"yes": "reviewed_support", "no": "reviewed_counterexample",
                            "uncertain": "unresolved_review"}.get(row.get("verdict"),
                                                                   "review_processing_error")
            add(row.get("item_id"), category)
    selection = selected_bucket(scores)
    source_hash = text_hash(source)
    for item_id, row in sorted((fn.failure_localisations or {}).items()):
        if (not isinstance(row, Mapping) or row.get("item_id") != item_id
                or row.get("source_sha256") != source_hash
                or "assigned_label" not in row
                or not matches_selected_bucket(row["assigned_label"], selection)):
            continue
        if item_id in passed:
            add(item_id, "flagged_success_contrast")
        elif item_id in failed:
            add(item_id, "matched_failed_attempt")
    for bucket in scores.get("buckets") or []:
        if (isinstance(bucket, Mapping)
                and not matches_selected_bucket(bucket.get("bucket"), selection)):
            for item_id in bucket.get("example_fail_ids") or []:
                if item_id in failed:
                    add(item_id, "unflagged_failed_attempt")

    # Round-robin across issues and controls, with distinct known tasks first.
    # Unknown identities remain explicitly unknown; attempt IDs do not establish breadth.
    selected, seen_tasks = [], set()
    queues = [iter(dict.fromkeys(group)) for group in groups.values()]
    while queues and len(selected) < limit:
        remaining = []
        for queue in queues:
            for item_id in queue:
                task = tasks.get(item_id)
                if item_id in selected or (task is not None and task in seen_tasks):
                    continue
                selected.append(item_id)
                if task is not None:
                    seen_tasks.add(task)
                remaining.append(queue)
                break
            if len(selected) == limit:
                break
        queues = remaining
    if not selected:
        return unavailable("No observed validation controls are available; inspect saved feedback.")
    # A revision can use an LLM even when the baseline is deterministic. Keep
    # every suggested batch runnable under the shared LLM diagnostic row limit.
    smoke_batches = [{"name": "smoke_test_feature", "arguments": {
        "name": fn.name, "expected_version_hash": result["version_hash"],
        "item_ids": selected[start:start + LLM_ROWS],
    }} for start in range(0, len(selected), LLM_ROWS)]
    result.update(
        available=True, item_ids=selected,
        controls=[{"item_id": item_id, "task_id": tasks.get(item_id),
                   "categories": categories[item_id]} for item_id in selected],
        baseline_smoke=smoke_batches[0],
        baseline_smoke_batches=smoke_batches,
        guidance=(
            "Inspect these items and the cited review findings. Passing or failing the benchmark "
            "does not establish whether this mechanism occurred; an unflagged failure is not "
            "automatically a false negative. baseline_smoke is the first batch; "
            "baseline_smoke_batches covers the full retained control set within per-call limits. "
            "Test a proposed revision with smoke_test_feature"
            "(source=<revised source>, item_ids=<one batch's IDs>), including dependencies "
            "when needed. "
            "Use the same batches for comparison and respect the existing diagnostic work budget. "
            "For a submitted revision, use its current name/version. Set expected outputs only "
            "after inspecting the evidence under that revision's claim. These selected controls "
            "diagnose regressions; they do not update population scores or provide fresh "
            "independent confirmation. Keep the fixed discovery panel unchanged; round validation "
            "will remeasure the revised detector on its original validation cohort."
        ),
    )
    return result


def _fixed_panel_ids(rt: Any, scores: dict, inventory: Mapping[str, Any]) -> list[str]:
    """Rebuild the evaluator's exact ID reservoir and verify its saved identity."""
    from contextlib import closing

    from autofa.data.parquet_store import _is_correct
    from autofa.mcp.tools_eval import _evaluation_seed, _llm_eval_settings, resolve_eval_limit
    from autofa.metrics.sampling import SCORING_SAMPLE_VERSION, StratifiedReservoir, stratum_key
    from autofa.session.family_coverage import _IDENTITY_COLUMNS
    from autofa.util.cancellation import check_cancelled

    saved = scores.get("search_panel") or {}
    if (saved.get("version") != 1 or not saved.get("cohort_fingerprint")
            or saved.get("claim_hash") != scores.get("evaluation_claim_hash")
            or saved.get("run_id") != rt.run.run_id
            or saved.get("corpus_fingerprint") != inventory.get("corpus_fingerprint")):
        raise ValueError("The original fixed scoring panel has no verifiable saved identity.")
    size, minimum, _ = _llm_eval_settings(rt.config)
    frame_limit = resolve_eval_limit(
        rt.run.eval_limit, cfg_default=setting_value(rt.config, "default_eval_limit"),
        cfg_max=setting_value(rt.config, "max_eval_limit"),
    )
    reservoir = (StratifiedReservoir(per_stratum_cap=max(minimum, size),
                                    seed=_evaluation_seed(rt.run)) if size else None)
    ids = []
    stream = rt.dataset.iter_batches(columns=_IDENTITY_COLUMNS, max_items=frame_limit)
    with closing(stream) as batches:
        for batch in batches:
            for item in batch:
                check_cancelled()
                item_id = str(item.get("_item_id") or "")
                if not item_id:
                    raise ValueError("Source identities cannot reconstruct the original panel.")
                if reservoir is not None:
                    reservoir.observe(item_id, key=stratum_key(item, is_fail=not _is_correct(item)))
                else:
                    ids.append(item_id)
    if reservoir is not None:
        ids = reservoir.allocate(target_n=size, min_per_stratum=minimum).item_ids
    version = SCORING_SAMPLE_VERSION if reservoir is not None else None
    identity = hashlib.sha256(json.dumps(
        [inventory["corpus_fingerprint"], sorted(ids), version], separators=(",", ":"),
    ).encode()).hexdigest()
    if identity != saved["cohort_fingerprint"]:
        raise ValueError("Source/settings do not reproduce the original fixed scoring panel.")
    return sorted(ids)


def collect_full_validation_controls(  # noqa: C901
    rt: Any, fn: Any, *, previous_scores: dict | None = None,
    previous_controls: dict | None = None, historical_controls: dict | None = None,
) -> dict[str, Any]:
    """Keep every observed full-validation issue plus the original comparison panel.

    Unlike the compact 30% repair suggestions, this set has no row cap and does
    not discard additional attempts from the same task. Cumulative controls are
    diagnostic evidence, never a population sample or automatic behavior labels.
    """
    from autofa.mcp.tools_eval import _evaluation_context
    from autofa.metrics.discovery_grades import grade_for_scores
    from autofa.metrics.task_support import task_support_assessment
    from autofa.session.family_coverage import cached_task_inventory

    source = review_source(fn)
    claim = claim_hash(source=source, description=fn.description or "",
                       mechanism=fn.mechanism or fn.description or fn.name)
    scores = current_behavior_scores(
        fn.scores, source=source, description=fn.description or "",
        mechanism=fn.mechanism or fn.description or fn.name,
        behavior_strictness=strictness_for_config(rt.config),
    )
    result = {"version": 1, "available": False, "complete": False,
              "scope": "targeted_polishing_controls", "name": fn.name,
              "claim_hash": claim, "version_hash": _version(fn.to_dict()),
              "run_id": rt.run.run_id, "panel_item_ids": [], "item_ids": [],
              "controls": [], "groups": {}, "unavailable_reasons": []}

    def unavailable(reason: str) -> dict:
        return {**result, "available": False, "complete": False,
                "reason": reason, "unavailable_reasons": [reason]}

    if (scores.get("evaluated") is not True or scores.get("sampled") is not False
            or scores.get("evaluation_scope_complete") is not True
            or scores.get("evaluation_stale") or scores.get("evaluation_claim_hash") != claim
            or scores.get("evaluation_run_id") != rt.run.run_id):
        return unavailable("Current complete full-dataset predictions are required for polishing.")
    try:
        context = _evaluation_context(rt)
        if not context or context != scores.get("evaluation_context"):
            return unavailable("The full-validation source/settings changed or cannot be verified.")
        inventory = cached_task_inventory(rt)
        if (inventory.get("inventory_complete") is not True
                or inventory.get("corpus_fingerprint")
                != scores.get("task_recall_corpus_fingerprint")):
            return unavailable("Source-verified full-validation identities are unavailable.")
        panel = _fixed_panel_ids(rt, scores, inventory)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return unavailable(str(exc))
    corpus_ids = set(inventory["item_ids"])
    result.update(source_signature=context, corpus_fingerprint=inventory["corpus_fingerprint"],
                  panel_item_ids=panel, panel_fingerprint=(scores.get("search_panel") or {}).get(
                      "cohort_fingerprint"), baseline_scores={key: scores.get(key) for key in (
                          "recall", "ppv_at_pi0", "behavior_purity", "n_flagged", "n_scored")})
    result["score_changes"] = {
        key: {"before": (previous_scores or {}).get(key), "after": scores.get(key)}
        for key in ("recall", "ppv_at_pi0", "behavior_purity")
    }
    support_fields = ("task_support_comparison", "task_support_comparison_basis",
                      "n_matched_failed_tasks", "estimated_n_matched_failed_tasks")
    result["task_support_change"] = {
        position: {key: assessment.get(key) for key in support_fields}
        for position, assessment in (
            ("before", task_support_assessment(previous_scores)),
            ("after", task_support_assessment(scores)),
        )
    }
    result["score_changes"]["task_support_comparison"] = {
        position: assessment["task_support_comparison"]
        for position, assessment in result["task_support_change"].items()
    }
    result["baseline_grade"] = grade_for_scores(scores)["code"]
    result["initial_score_changes"] = deepcopy(result["score_changes"])
    result["initial_task_support_change"] = deepcopy(result["task_support_change"])
    result["initial_validation_grade"] = result["baseline_grade"]
    result["previous_claim_hash"] = (previous_scores or {}).get("evaluation_claim_hash")
    rows: dict[str, dict] = {}
    limitations = []
    if previous_controls and historical_controls:
        return unavailable("Supply current or historical controls, not both.")
    previous_controls = previous_controls or historical_controls
    if previous_controls:
        if ((not historical_controls and previous_controls.get("source_signature") != context)
                or previous_controls.get("corpus_fingerprint") != inventory["corpus_fingerprint"]
                or previous_controls.get("panel_item_ids") != panel
                or previous_controls.get("run_id") != rt.run.run_id):
            return unavailable("Earlier polishing controls belong to a different corpus or panel.")
        result["initial_score_changes"] = deepcopy(
            previous_controls.get("initial_score_changes")
            or previous_controls.get("score_changes") or result["score_changes"])
        result["initial_task_support_change"] = deepcopy(
            previous_controls.get("initial_task_support_change")
            or previous_controls.get("task_support_change") or result["task_support_change"])
        result["initial_validation_grade"] = previous_controls.get(
            "initial_validation_grade", previous_controls.get(
                "baseline_grade", result["baseline_grade"]))
        retained = list(previous_controls.get("controls") or [])
        if not historical_controls:
            from autofa.session.polishing_generalization import retained_diagnostic_problems

            retained += retained_diagnostic_problems(rt, fn.name, previous_controls)
        for row in retained:
            if row.get("item_id") not in corpus_ids:
                return unavailable("A retained polishing control is missing from the source.")
            rows[row["item_id"]] = {**row, "categories": list(row.get("categories") or []),
                                    "review_findings": list(row.get("review_findings") or [])}
            if historical_controls:
                # Preserve diagnostic membership and evidence, not a prediction
                # made under unverified old settings. Current census labels are
                # attached below only when they have actually been recomputed.
                rows[row["item_id"]].pop("baseline_label", None)
                rows[row["item_id"]].pop("baseline_claim_hash", None)
                rows[row["item_id"]]["historical_categories"] = list(row.get("categories") or [])
                rows[row["item_id"]]["categories"] = ["historical_regression_case"]
                rows[row["item_id"]]["historical_review_findings"] = [
                    {**finding, "historical": True}
                    for finding in row.get("review_findings") or []
                ]
                rows[row["item_id"]]["review_findings"] = []
    previous_ids = None
    if previous_scores and previous_scores.get("evaluation_context") == context:
        if isinstance(previous_scores.get("evaluation_item_ids"), list):
            previous_ids = set(previous_scores["evaluation_item_ids"])
        elif previous_scores.get("sampled") is False:
            previous_ids = corpus_ids
    failed = inventory.get("task_by_failed_item_id") or {}
    passed = set(inventory.get("passed_item_ids") or [])
    identities = inventory.get("identity_by_item_id") or {}

    def add(item_id: Any, category: str, *, label: Any = None, has_label: bool = False) -> None:
        if not isinstance(item_id, str) or item_id not in corpus_ids:
            return
        row = rows.setdefault(item_id, {
            "item_id": item_id, "task_id": (inventory.get("task_by_item_id") or {}).get(item_id),
            "benchmark": (identities.get(item_id) or {}).get("benchmark"), "categories": [],
            "benchmark_outcome": "passed" if item_id in passed else (
                "failed" if item_id in failed else "unknown"),
            "review_findings": [], "new_since_checkpoint": (
                item_id not in previous_ids if previous_ids is not None else None),
        })
        if category not in row["categories"]:
            row["categories"].append(category)
        if has_label:
            row.update(baseline_label=label, baseline_claim_hash=claim)

    selected = selected_bucket(scores)
    resolved: dict[str, Any] = {}
    membership_complete = True
    for bucket in scores.get("buckets") or []:
        ids = [*(bucket.get("example_fail_ids") or []), *(bucket.get("example_pass_ids") or [])]
        if len(set(ids)) != bucket.get("n") or not set(ids) <= corpus_ids:
            membership_complete = False
        for item_id in ids:
            if item_id in resolved:
                membership_complete = False
            resolved[item_id] = bucket.get("bucket")
            matched = matches_selected_bucket(bucket.get("bucket"), selected)
            if matched and item_id in passed:
                add(item_id, "flagged_success_contrast", label=bucket["bucket"], has_label=True)
            elif item_id in failed:
                add(item_id, "matched_failed_attempt" if matched else "unflagged_failed_attempt",
                    label=bucket["bucket"], has_label=True)
    if len(resolved) != scores.get("n_scored"):
        membership_complete = False
    unknown = corpus_ids - set(resolved)
    if membership_complete and len(unknown) == scores.get("n_unresolved", 0):
        for item_id in sorted(unknown):
            add(item_id, "unresolved_prediction")
        if unknown:
            limitations.append("All unresolved prediction IDs are retained; saved outcomes do not "
                               "distinguish every execution error from every abstention.")
    else:
        membership_complete = False
        limitations.append("Full per-item prediction membership was not retained consistently; "
                           "unknown outcomes are explicit, not inferred detector misses.")
        for item_id in sorted(unknown):
            add(item_id, "prediction_outcome_unavailable")
    for row in scores.get("error_examples") or []:
        if isinstance(row, Mapping):
            add(row.get("item_id"), "execution_error")
            if row.get("item_id") in rows:
                rows[row["item_id"]]["execution_error"] = row.get("error")
    contract = (scores.get("claim_contract") or {}).get("contract_id")
    status = behavior_validation(scores, include_details=False)["status"]
    if contract and status not in {"stale", "legacy", "unjudged", "invalid"}:
        for row in scores.get("behavior_verdicts") or []:
            if (row.get("claim_contract_id") != contract or row.get("historical_review")
                    or row.get("verification_status") == "historical"):
                continue
            category = ("incomplete_review"
                        if row.get("completion_status") == "needs_evidence_review"
                        else {"yes": "reviewed_support", "no": "reviewed_counterexample",
                              "uncertain": "unresolved_review"}.get(row.get("verdict"),
                                                                    "review_processing_error"))
            add(row.get("item_id"), category)
            if row.get("item_id") in rows:
                finding = {"claim_hash": claim, "verdict": row.get("verdict"),
                           "claim_description": fn.description or "",
                           "claim_mechanism": fn.mechanism or fn.description or fn.name,
                           "source_sha256": text_hash(source),
                           "reason": row.get("rationale", row.get("reason")),
                           "claim_assessments": row.get("claim_assessments") or [],
                           "completion_status": row.get("completion_status")}
                findings = rows[row["item_id"]]["review_findings"]
                if finding not in findings:
                    findings.append(finding)
    for item_id in panel:
        add(item_id, "fixed_scoring_panel", label=resolved.get(item_id),
            has_label=item_id in resolved)
    # Keep previously selected items even if the latest result no longer gives
    # them a problematic category. A successful repair must not erase its test.
    for item_id in rows:
        rows[item_id]["benchmark_outcome"] = ("passed" if item_id in passed else
                                              "failed" if item_id in failed else "unknown")
        if item_id in resolved:
            rows[item_id].update(baseline_label=resolved[item_id], baseline_claim_hash=claim)
        else:
            rows[item_id].pop("baseline_label", None)
            rows[item_id].pop("baseline_claim_hash", None)
    if previous_controls and not historical_controls:
        old_selection = previous_controls.get("selection")
        old_labels = {
            row["item_id"]: row["baseline_label"]
            for row in [*(previous_controls.get("controls") or []),
                        *(previous_controls.get("unflagged_success_pool") or [])]
            if row.get("item_id") in passed and "baseline_label" in row
            and row.get("baseline_claim_hash") == previous_controls.get("claim_hash")
        }
        compared = passed & old_labels.keys() & resolved.keys()
        removed = sorted(item_id for item_id in compared
                         if matches_selected_bucket(old_labels[item_id], old_selection)
                         and not matches_selected_bucket(resolved[item_id], selected))
        added = sorted(item_id for item_id in compared
                       if not matches_selected_bucket(old_labels[item_id], old_selection)
                       and matches_selected_bucket(resolved[item_id], selected))
        for item_id in added:
            add(item_id, "introduced_success_match", label=resolved[item_id], has_label=True)
        result["full_validation_passing_match_changes"] = {
            "scope": "successive full validations", "n": len(compared),
            "complete": compared == passed,
            "previous_flagged": sum(matches_selected_bucket(old_labels[item_id], old_selection)
                                    for item_id in compared),
            "flagged": sum(matches_selected_bucket(resolved[item_id], selected)
                           for item_id in compared),
            "added_matches": added, "lost_matches": removed,
            "unresolved": sorted(passed - compared),
        }
    # Population census membership supplies known nonmatches, not presumed
    # behavior labels. Revision diagnostics draw a separate rotating sample from
    # this pool so fixing known passing matches cannot hide newly introduced ones.
    result["unflagged_success_pool"] = [
        {"item_id": item_id, "task_id": (inventory.get("task_by_item_id") or {}).get(item_id),
         "benchmark": (identities.get(item_id) or {}).get("benchmark"),
         "benchmark_outcome": "passed",
         "baseline_label": resolved[item_id], "baseline_claim_hash": claim}
        for item_id in sorted(passed - set(rows))
        if item_id in resolved and not matches_selected_bucket(resolved[item_id], selected)
    ]
    groups: dict[str, list[str]] = {}
    for item_id, row in sorted(rows.items()):
        for category in row["categories"]:
            groups.setdefault(category, []).append(item_id)
    result.update(available=True, complete=membership_complete, controls=list(rows.values()),
                  item_ids=sorted(rows), groups=groups, unavailable_reasons=limitations,
                  selection=selected)
    result["controls_fingerprint"] = polishing_controls_identity(result)
    try:
        if _evaluation_context(rt) != context:
            return unavailable("Source/settings changed while assembling polishing controls.")
    except (OSError, ValueError, TypeError) as exc:
        return unavailable(f"Source could not be rechecked: {exc}")
    result["guidance"] = (
        "This is the original fixed scoring panel plus every retained full-validation issue and "
        "support control, including other attempts from the same task. Review findings identify "
        "actual judged requirements; benchmark outcomes do not establish behavior labels. "
        "Unflagged failures are investigation candidates, not proven false negatives. Compare "
        "revisions on this cumulative diagnostic set plus a fresh task-distinct sample of "
        "previously unflagged passing attempts. Preserve new problematic cases across revisions. "
        "These additional controls are diagnostic, not independent causal confirmation. "
        "Never report their rates as population "
        "precision, coverage, adherence or a quality grade. Full validation occurs when polishing "
        "closes; the unchanged fixed discovery panel remains the population estimate."
    )
    return result
