# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate outcome-selected controls from adherence and confirmed-support scores."""

from __future__ import annotations

from contextlib import closing
from copy import copy, deepcopy


def known_failed_smoke_matches(
    rt, fn, *, client, frozen_ids=None, excluded_task_groups=(),
) -> list[tuple[dict, object]]:
    """Find actual current controls without executing detectors or estimating coverage."""
    from autofa.core.dependencies import execution_bundle
    from autofa.core.fields import get_field_policy
    from autofa.mcp.tools_eval import _fn_deps
    from autofa.metrics.association import matches_selected_bucket, selected_bucket
    from autofa.session.family_coverage import cached_task_inventory
    from autofa.session.review_evidence import review_context, task_group
    from autofa.session.smoke_work import SmokeWork
    from autofa.util.cancellation import check_cancelled

    work = SmokeWork(rt, source=fn.source, bundle=execution_bundle(fn),
                     policy=get_field_policy(rt.config), client=client, uses_llm=False,
                     deps=_fn_deps(fn))
    inventory = cached_task_inventory(rt)
    failed = inventory["task_by_failed_item_id"]
    ids = work.known_item_ids() & failed.keys()
    if frozen_ids is not None:
        ids &= set(frozen_ids)
    else:
        ids -= set((fn.scores or {}).get("evaluation_item_ids") or [])
        excluded = set(excluded_task_groups) | set(
            review_context(rt.run.run_dir(), fn.name)["reserved_confirmation_task_groups"])
        ids = {item_id for item_id in ids
               if task_group(inventory["identity_by_item_id"][item_id]) not in excluded}
    if not ids:
        return []
    from autofa.data.row_index import _row_is_correct
    from autofa.mcp.tools_eval import _fetch_items_by_id

    results = []
    represented = set()
    rows = _fetch_items_by_id(rt.dataset, ids, max_items=None)
    try:
        for item in rows:
            check_cancelled()
            item_id = str(item["_item_id"])
            # The inventory establishes the canonical task, the current row the
            # outcome. Missing identities and unknown outcomes are never selected.
            group = failed.get(item_id)
            if group is None or group in represented or _row_is_correct(item) is not False:
                continue
            saved = work.cached_result(item)
            if (not saved or saved.get("status") != "value"
                    or not matches_selected_bucket(saved.get("value"),
                                                   selected_bucket(fn.scores or {}))):
                continue
            results.append((item, saved["value"]))
            represented.add(group)
            if len(results) >= 3:
                break
    finally:
        rows.close()
    return results


def summarize_diagnostic_review(scores: dict, *, name: str) -> dict | None:
    """Keep provider sessions and full cited assessments behind paged retrieval."""
    result = scores.get("diagnostic_review")
    if not isinstance(result, dict):
        return None
    return {
        **{key: result.get(key) for key in (
            "n_judged", "n_yes", "n_no", "n_uncertain", "n_errors",
            "n_review_incomplete", "sample", "excluded_from_adherence",
            "excluded_from_confirmed_support", "excluded_from_repair_validation", "review_mode",
        )},
        "current": diagnostic_scores(scores) is not None,
        "verdicts": [
            {key: row.get(key) for key in ("item_id", "verdict", "processing_status")}
            for row in result.get("verdicts") or [] if isinstance(row, dict)
        ],
        "details_tool": {"name": "get_function_feedback", "arguments": {"name": name}},
    }


def retain_current_diagnostic_feedback(rt, fn, scores: dict, result: dict) -> dict:
    """Retain same-claim counterexamples without enlarging the latest paid cohort."""
    previous = scores.get("diagnostic_review")
    if (not isinstance(previous, dict) or diagnostic_scores(scores) is None
            or any(previous.get(key) != result.get(key) for key in (
                "claim_hash", "claim_contract", "judge_policy_hash", "judge_model", "judge_harness",
                "judge_effort", "judge_thinking", "judge_thinking_budget_tokens",
            ))):
        return result
    current_ids = {row["item_id"] for row in result.get("verdicts") or []}
    if not any(row.get("item_id") not in current_ids for row in previous.get("verdicts") or []):
        return result
    from autofa.mcp.tools_eval import _fetch_items_by_id
    from autofa.metrics.adaptive_review import merge_expanded_review
    from autofa.metrics.review_content import RetainedReviewContent
    from autofa.util.cancellation import check_cancelled

    # Only retained diagnostics need source checks; prior scored reviews keep
    # their own unchanged provenance and do not enter this diagnostic aggregate.
    diagnostic_fn = copy(fn)
    diagnostic_fn.scores = {**scores, "behavior_verdicts": []}
    retained = RetainedReviewContent([diagnostic_fn])
    with closing(_fetch_items_by_id(
        rt.dataset, retained.missing_item_ids(), max_items=None,
    )) as rows:
        for item in rows:
            check_cancelled()
            retained.observe(item)
    checked = retained.apply(fn.name, diagnostic_fn.scores)["diagnostic_review"]
    extra_rows = [
        row for row in checked.get("verdicts") or []
        if row.get("item_id") not in current_ids and row.get("snapshot_id")
        and not row.get("historical_review")
        and row.get("verification_status") != "historical"
    ]
    if not extra_rows:
        return result
    extra_ids = {row["item_id"] for row in extra_rows}
    extra = {**checked, "verdicts": extra_rows, "n_judged": len(extra_rows),
             "item_calibrations": [
                 row for row in checked.get("item_calibrations") or []
                 if row.get("item_id") in extra_ids],
             "sessions": []}
    combined = merge_expanded_review(result, extra)
    if result.get("review_mode") == "diagnostic" or previous.get("review_mode") == "diagnostic":
        combined["review_mode"] = "diagnostic"
        combined["excluded_from_repair_validation"] = True
    combined["review_localisations"] = {
        **{key: value for key, value in checked.get("review_localisations", {}).items()
           if key in extra_ids},
        **result.get("review_localisations", {}),
    }
    combined["sample"] = {
        **result["sample"], "latest_cohort_item_ids": sorted(current_ids),
        "latest_cohort_review_mode": result.get("review_mode", "outcome_diagnostic"),
        "latest_cohort_n": len(current_ids), "retained_prior_item_ids": sorted(extra_ids),
        "retained_prior_review_mode": previous.get("review_mode", "outcome_diagnostic"),
        "retained_prior_n": len(extra_ids), "total_diagnostic_n": len(current_ids | extra_ids),
    }
    combined["rationale"] = (
        f"Latest diagnostic cohort: {len(current_ids)} items. Retained {len(extra_ids)} "
        "current prior diagnostic findings separately from adherence and confirmed support.")
    return combined


def diagnostic_scores(scores: dict) -> dict | None:
    """Adapt a current supplemental review for existing feedback, never for grading."""
    from autofa.metrics.verdicts import stale_judge_policy

    result = scores.get("diagnostic_review")
    explicit = isinstance(result, dict) and result.get("review_mode") == "diagnostic"
    if (not isinstance(result, dict) or not result.get("verdicts")
            or result.get("excluded_from_adherence") is not True
            or result.get("excluded_from_confirmed_support") is not True
            or scores.get("diagnostic_evidence_stale") or result.get("evidence_stale")
            or (explicit and (
                result.get("claim_hash") != scores.get("evaluation_claim_hash")
                or scores.get("evaluated") is not True
                or scores.get("evaluation_stale") or stale_judge_policy(result)))
            or (not explicit and (any(scores.get(key) for key in (
                "diagnostic_evidence_stale", "behavior_judgment_stale",
                "behavior_policy_stale", "behavior_evidence_stale",
            ))
            or stale_judge_policy(scores)
            or any(result.get(key) != scores.get(key)
                   for key in (
                       "claim_hash", "judge_policy_hash", "judge_model", "judge_harness"))))):
        return None
    from autofa.metrics.review_scores import apply_review_result

    adapted = deepcopy(scores)
    adapted.pop("diagnostic_review", None)
    adapted.pop("diagnostic_evidence_stale", None)
    apply_review_result(adapted, result, judge_harness=result.get("judge_harness", "claude"))
    if explicit:
        adapted.pop("behavior_policy_stale", None)
    adapted["behavior_review_phase"] = "development"
    adapted["behavior_sample"] = {
        "phase": "development", "limited": True,
        "target_n": len(result["verdicts"]), "actual_n": len(result["verdicts"]),
        "reason": ("Bounded development diagnosis; excluded from adherence and support."
                   if explicit else
                   "Outcome-selected diagnostic controls; excluded from adherence and support."),
    }
    adapted["claim_calibration"] = result.get("claim_calibration")
    adapted["item_calibrations"] = result.get("item_calibrations") or []
    return adapted
