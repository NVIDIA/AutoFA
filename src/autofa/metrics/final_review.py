# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Representative checkpoint and final review, with resumable saved judgments."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import review_source
from autofa.defaults import setting_value
from autofa.metrics.association import matches_selected_bucket, selected_bucket
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.judge_checkpoints import fingerprint
from autofa.metrics.review_scores import apply_review_result
from autofa.metrics.verdicts import behavior_validation, current_behavior_scores, verdict_for_scores
from autofa.util.cancellation import cancellation_guard, check_cancelled

COHORT_POLICY = "representative_task_sample_v1"


def _matches(scores: dict) -> dict[str, Any]:
    """Use complete saved bucket membership, never execute a detector again."""
    labels = {}
    for bucket in scores.get("buckets") or []:
        value = bucket.get("bucket")
        if not matches_selected_bucket(value, selected_bucket(scores)):
            continue
        ids = [*(bucket.get("example_fail_ids") or []), *(bucket.get("example_pass_ids") or [])]
        expected = bucket.get("n_rows_scored") if scores.get("sampled") else bucket.get("n")
        if len(set(ids)) != expected:
            raise ValueError(
                "Evaluated match membership is incomplete; rescore before detector review."
            )
        labels.update({item_id: value for item_id in ids})
    if scores.get("sampled"):
        panel = scores.get("evaluation_item_ids")
        if not isinstance(panel, list) or not set(labels) <= set(panel):
            raise ValueError("Checkpoint match membership is outside the evaluated panel.")
        observed_n = scores.get("n_rows_scored", 0 if scores.get("n_flagged") == 0 else None)
        if observed_n != len(labels):
            raise ValueError("Checkpoint match membership is incomplete; rescore before review.")
    elif len(labels) != scores.get("n_flagged"):
        raise ValueError("Full-scan match membership is incomplete; rescore before final review.")
    return labels


def _pending(row: dict) -> bool:
    return (
        row.get("verdict") not in {"yes", "no", "uncertain"}
        or row.get("processing_status") == "error"
        or row.get("completion_status") == "needs_evidence_review"
    )


def _saved_completion_pending(rt, row: dict) -> bool:
    """Recheck uncertain saved reads only for an explicitly requested recovery."""
    if _pending(row):
        return True
    if row.get("verdict") != "uncertain" or not row.get("review_path"):
        return False
    from autofa.metrics.judge_evidence import EvidenceReview

    try:
        saved_path = Path(row["review_path"])
        if ".." in saved_path.parts:
            return False
        archive = (rt.run.run_dir() / "behavior_judge").resolve()
        directory = saved_path.resolve()
        if not directory.is_relative_to(archive):
            # Saved-run copies retain their original absolute review paths.
            # Resolve only an unambiguous archive suffix inside this run.
            if saved_path.parts.count("behavior_judge") != 1:
                return False
            suffix = saved_path.parts[saved_path.parts.index("behavior_judge") + 1:]
            directory = archive.joinpath(*suffix).resolve()
            if not directory.is_relative_to(archive):
                return False
        snapshot = json.loads((directory / "evidence.json").read_text())
        if not isinstance(snapshot, dict) or snapshot.get("snapshot_id") != row.get("snapshot_id"):
            return False
        review = EvidenceReview.restore(snapshot, directory)
        assessments = row.get("claim_assessments") or []
        if (not isinstance(assessments, list)
                or any(not isinstance(part, dict)
                       or not isinstance(part.get("components", []), list)
                       or any(not isinstance(child, dict) for child in part.get("components", []))
                       for part in assessments)
                or review._inspection_error({"claim_assessments": assessments})):
            return False
        return bool(review.completion_plan(assessments))
    except (OSError, ValueError, KeyError, TypeError):
        # An unavailable archive is not evidence that a settled judgment was wrong.
        return False


def _reason(scores: dict) -> str:
    errors = int(scores.get("behavior_n_errors") or 0)
    incomplete = int(scores.get("behavior_n_review_incomplete") or 0)
    uncertain = int(scores.get("behavior_n_uncertain") or 0)
    parts = []
    if errors:
        parts.append(
            f"{errors} review(s) have processing failures, not detector rejections"
        )
    if incomplete:
        parts.append(f"{incomplete} review(s) could not finish checking the required evidence")
    if uncertain:
        parts.append(f"{uncertain} review(s) remain inconclusive; uncertainty is retained")
    if scores.get("behavior_purity") is None:
        parts.append("No confirmed or rejected cases establish a detector-adherence score")
    for failure in scores.get("behavior_processing_failures") or []:
        action = failure.get("recovery_action")
        if action and action not in parts:
            parts.append(action)
    return ". ".join(parts) or "Current detector adherence and review results are available."


def _commit(rt, fn, scores: dict, receipt: dict) -> None:
    from autofa.metrics.final_validation_selection import _all_predictions_abstained
    from autofa.metrics.task_support import build_task_support
    from autofa.session.family_coverage import cached_task_inventory

    key = {"round_checkpoint": "round_review", "selection_checkpoint": "selection_review"}.get(
        receipt.get("scope"), "final_review")
    scores = dict(scores)
    if receipt.get("status") == "unavailable" and not receipt.get("identity"):
        # Without a verified population/cohort, an earlier hand-picked review
        # cannot certify the current validation result. Bounded errors in a
        # verified cohort have an identity and retain their measured score.
        scores["behavior_evidence_stale"] = True
    previous_receipt = (fn.scores or {}).get(key) or {}
    if (receipt.get("status") == "unavailable" and not receipt.get("identity")
            and previous_receipt.get("status") == "running"
            and previous_receipt.get("claim_hash") == receipt.get("claim_hash")):
        # A transient source outage is not a completed review attempt. Preserve
        # the original reservation while callers report the unavailable evidence.
        receipt = {**previous_receipt, "evidence_unavailable": True,
                   "reason": receipt.get("reason", "Review evidence is unavailable.")}
    scores = {**scores, key: receipt}
    selection = scores.get("final_validation_selection") or {}
    settled_uncertainty = (
        receipt.get("status") == "unavailable"
        and bool(receipt.get("identity"))
        and scores.get("behavior_n_uncertain", 0) > 0
        and not scores.get("behavior_n_errors")
        and not scores.get("behavior_n_review_incomplete")
        and not scores.get("behavior_evidence_stale")
    )
    if (key == "final_review" and selection.get("status") == "selected"
            and selection.get("claim_hash") == receipt.get("claim_hash")
            and (receipt.get("status") in {"complete", "not_required"}
                 or settled_uncertainty or _all_predictions_abstained(scores))):
        scores["final_validation_selection"] = {
            **selection, "status": "complete",
            "reason": "Selected full-dataset scoring and review have completed.",
        }
    previous = fn.scores
    with cancellation_guard(), rt._library_lock:
        try:
            fn.scores = scores
            try:
                inventory = cached_task_inventory(rt)
            except (OSError, ValueError, TypeError):
                inventory = None
            scores.update(build_task_support(fn, inventory))
            verdict = verdict_for_scores(scores)
            scores.update(badge=verdict["label"], verdict=verdict)
            scores["feedback"] = {
                **(scores.get("feedback") or {}),
                **{
                    key: scores.get(key)
                    for key in (
                        "behavior_purity",
                        "behavior_specificity",
                        "n_confirmed_failed_tasks",
                        "confirmed_task_support_status",
                        "judge_prefilter",
                    )
                },
                "behavior_validation": behavior_validation(scores, include_details=False),
            }
            rt.persist_library()
        except BaseException:
            fn.scores = previous
            raise


def _merge_result(scores: dict, fresh: dict, retained: list[dict], cohort: list[str]) -> dict:
    """Preserve counterexamples and substantive uncertainty in the exact denominator."""
    by_id = {row["item_id"]: row for row in retained}
    by_id.update({row["item_id"]: row for row in fresh.get("verdicts") or []})
    if set(by_id) != set(cohort):
        raise ValueError("Final judge results do not match the frozen review cohort")
    rows = [deepcopy(by_id[key]) for key in cohort]
    counts = Counter(row["verdict"] for row in rows)
    decisive = counts["yes"] + counts["no"]
    purity = counts["yes"] / (len(rows) - counts["error"]) if decisive else None
    metadata = {
        key: value
        for key, value in scores.items()
        if key.startswith("judge_") or key in {
            "claim_hash", "claim_contract", "claim_compilation",
            "behavior_specificity", "specificity_rationale",
        }
    }
    metadata.update(
        backend=scores.get("judge_backend"), evidence_version=scores.get("judge_evidence_version")
    )
    calibrations = {
        row["item_id"]: row
        for row in scores.get("item_calibrations") or []
        if row.get("item_id") in {old["item_id"] for old in retained}
    }
    calibrations.update({row["item_id"]: row for row in fresh.get("item_calibrations") or []})
    result = {
        **metadata,
        **fresh,
        "verdicts": rows,
        "n_judged": len(rows),
        "n_yes": counts["yes"],
        "n_no": counts["no"],
        "n_uncertain": counts["uncertain"],
        "n_errors": counts["error"],
        "n_completed": decisive,
        "n_review_incomplete": sum(
            row.get("completion_status") == "needs_evidence_review" for row in rows
        ),
        "behavior_purity": purity,
        "item_calibrations": list(calibrations.values()),
        "rationale": (
            f"{counts['yes']}/{len(rows)} labels confirmed; {counts['no']} rejected; "
            f"{counts['uncertain']} uncertain; {counts['error']} processing errors. "
            "Only the selected review cohort determines detector adherence."
        ),
    }
    # Advice must summarize the selected cohort, not just its newly judged part.
    from autofa.metrics.claim_calibration import attach_calibration

    result.pop("claim_calibration", None)
    result.pop("calibration_hint", None)
    return attach_calibration(result)


def _finalize_one(  # noqa: C901
    rt, fn, *, scope: str = "full_corpus", retry_incomplete: bool = False,
) -> tuple[dict, bool]:
    from autofa.mcp.tools_eval import (
        _fetch_items_by_id,
        _prefilter_decision,
        _review_settings,
        _review_settings_match,
        judge_evidence_sample_size,
    )
    from autofa.metrics.claude_judge import judge_behavior_with_claude, judge_policy_hash
    from autofa.metrics.diagnostic_review import (
        diagnostic_scores,
        retain_current_diagnostic_feedback,
    )
    from autofa.metrics.final_validation_selection import checkpoint_scoring_complete
    from autofa.metrics.judge_backend import backend_arguments
    from autofa.metrics.review_content import RetainedReviewContent, saved_review_localisations
    from autofa.session.family_coverage import cached_task_inventory, verified_identity_projection
    from autofa.session.guidance import guidance_prompt
    from autofa.session.recovery import RunInterruptedError
    from autofa.session.review_evidence import _diverse_ids, allocate_review

    source = review_source(fn)
    description = fn.description or ""
    mechanism = fn.mechanism or description or fn.name
    claim = claim_hash(source=source, description=description, mechanism=mechanism)
    receipt_key = {
        "round_checkpoint": "round_review", "selection_checkpoint": "selection_review",
    }.get(scope, "final_review")
    receipt = {"claim_hash": claim, "scope": scope, "status": "unavailable", "reason": ""}
    scores = deepcopy(fn.scores or {})
    full_scan = (scores.get("sampled") is False
                 and scores.get("evaluation_scope_complete") is True)
    checkpoint = scores.get("round_validation") or {}
    checkpoint_scan = (
        scope == "round_checkpoint"
        and checkpoint.get("version") == 1
        and checkpoint.get("allocation") == "proportional_benchmark_outcome"
        and bool(checkpoint.get("panel_fingerprint"))
        and isinstance(scores.get("evaluation_item_ids"), list)
        and checkpoint.get("complete") is True
        and (scores.get("n_scored", 0) + scores.get("n_unresolved", 0)
             == len(scores["evaluation_item_ids"]))
    )
    selection_scan = (
        scope == "selection_checkpoint"
        and checkpoint_scoring_complete(scores)
        and isinstance(scores.get("evaluation_item_ids"), list)
        and (scores.get("n_scored", 0) + scores.get("n_unresolved", 0)
             == len(scores["evaluation_item_ids"]))
    )
    if not (
        scores.get("evaluated")
        and (full_scan or checkpoint_scan or selection_scan)
        and scores.get("evaluation_claim_hash") == claim
        and scores.get("evaluation_run_id") == rt.run.run_id
        and not scores.get("evaluation_stale")
    ):
        receipt["reason"] = ("Complete the round checkpoint scoring before detector review."
                             if scope != "full_corpus" else
                             "Complete full-dataset scoring before final detector review.")
        _commit(rt, fn, scores, receipt)
        return receipt, False
    if not scores.get("n_scored") or any(
        scores.get(key) is None for key in ("recall", "ppv_at_pi0")
    ):
        receipt["reason"] = (
            "No resolved predictions establish coverage and precision; "
            "inspect detector execution errors."
        )
        _commit(rt, fn, scores, receipt)
        return receipt, False
    try:
        labels = _matches(scores)
    except ValueError as exc:
        receipt["reason"] = str(exc)
        _commit(rt, fn, scores, receipt)
        return receipt, False
    if not labels:
        receipt.update(status="not_required", reason="No matched examples require detector review.")
        _commit(rt, fn, scores, receipt)
        return receipt, False
    required_n = judge_evidence_sample_size(len(labels))
    receipt.update(required_n=required_n, matched_n=len(labels))
    config, reasoning, settings = _review_settings(rt.config)
    strictness = strictness_for_config(rt.config)
    current = current_behavior_scores(
        scores,
        source=source,
        description=description,
        mechanism=mechanism,
        behavior_strictness=strictness,
    )
    compatible = not current.get("behavior_judgment_stale") and _review_settings_match(
        scores, settings
    )
    previous_receipt = scores.get(receipt_key) or {}
    diagnostic = diagnostic_scores(scores)
    diagnostic_compatible = (
        diagnostic is not None
        and _review_settings_match(diagnostic, settings)
        and not current_behavior_scores(
            diagnostic, source=source, description=description, mechanism=mechanism,
            behavior_strictness=strictness,
        ).get("behavior_judgment_stale")
    )
    old_rows = scores.get("behavior_verdicts") or []
    diagnostic_rows = ((scores.get("diagnostic_review") or {}).get("verdicts") or [])
    from autofa.metrics.review_assessments import archived_assessments, reusable_contract

    archived = []
    for saved in archived_assessments(scores):
        projected = current_behavior_scores(
            saved, source=source, description=description, mechanism=mechanism,
            behavior_strictness=strictness,
        )
        if (not projected.get("behavior_judgment_stale")
                and _review_settings_match(saved, settings)
                and (saved.get("claim_contract") or {}).get("contract_id")
                == (scores.get("claim_contract") or {}).get("contract_id")):
            archived.append(saved)
    reusable_rows = ([*old_rows] if compatible else []) + (
        diagnostic_rows if diagnostic_compatible else [])
    reusable_rows += [row for saved in archived for row in saved.get("behavior_verdicts") or []]
    for row in reusable_rows:
        if (
            isinstance(row, dict)
            and row.get("item_id") in labels
            and matches_selected_bucket(row.get("assigned_label"), selected_bucket(scores))
        ):
            # Bucket keys are serialized strings, but checkpoints distinguish a
            # boolean prediction from the string 'True'. Preserve its saved type.
            labels[row["item_id"]] = row["assigned_label"]
    localisations = saved_review_localisations(fn)
    existing_ids = {
        row["item_id"]
        for row in [*old_rows, *diagnostic_rows, *reusable_rows]
        if isinstance(row, dict) and isinstance(row.get("item_id"), str)
    }
    # Check saved evidence before selecting additions. Discovery may have
    # reviewed fewer matches than the full corpus now requires.
    try:
        with closing(_fetch_items_by_id(rt.dataset, existing_ids, max_items=None)) as rows:
            items = {str(item["_item_id"]): item for item in rows}
    except (OSError, ValueError, TypeError, KeyError):
        scores["behavior_evidence_stale"] = True
        _commit(
            rt, fn, scores, {**receipt, "reason": "Saved review evidence could not be checked."}
        )
        raise
    retained_content = RetainedReviewContent([fn])
    for item in items.values():
        retained_content.observe(item)
    checked = retained_content.apply(fn.name, current)
    if (
        checked.get("behavior_verdicts") != scores.get("behavior_verdicts")
        or bool(checked.get("behavior_judgment_stale"))
        != bool(scores.get("behavior_judgment_stale"))
        or bool(checked.get("behavior_evidence_stale"))
        != bool(scores.get("behavior_evidence_stale"))
        or checked.get("diagnostic_review") != scores.get("diagnostic_review")
        or bool(checked.get("diagnostic_evidence_stale"))
        != bool(scores.get("diagnostic_evidence_stale"))
    ):
        _commit(rt, fn, checked, {**receipt, "reason": "Checking current review provenance."})
    scores = checked
    valid_by_id = {}
    # An explicit standard review may have replaced the displayed aggregate
    # since this representative operation completed. Recover its saved findings
    # against current source content before reconstructing the frozen cohort.
    # Current findings take precedence over historical versions of the same row.
    checked_archives = []
    for saved in archived:
        saved_fn = copy(fn)
        saved_fn.scores = saved
        verifier = RetainedReviewContent([saved_fn])
        for item in items.values():
            verifier.observe(item)
        verified = verifier.apply(fn.name, saved)
        checked_archives.append(verified)
        for item_id, localisation in saved_review_localisations(saved_fn).items():
            localisations.setdefault(item_id, localisation)
    for rows, usable in (
        *((saved.get("behavior_verdicts") or [], True) for saved in checked_archives),
        ((checked.get("diagnostic_review") or {}).get("verdicts") or [], diagnostic_compatible),
        (checked.get("behavior_verdicts") or [], compatible),
    ):
        for row in rows:
            if (usable and isinstance(row.get("item_id"), str)
                    and matches_selected_bucket(row.get("assigned_label"), selected_bucket(scores))
                    and row.get("verification_status") != "historical"
                    and not row.get("historical_review")):
                valid_by_id[row["item_id"]] = row
    valid_rows = list(valid_by_id.values())
    # Select before consulting saved judgments: a full count of hand-picked
    # discovery examples does not establish a representative review cohort.
    base_seed = getattr(rt.run, "exploration_base_seed", None)
    seed = int(base_seed if base_seed is not None else
               getattr(rt.run, "exploration_seed", 0) or 0)
    identities = cached_task_inventory(rt)["identity_by_item_id"]
    if not set(labels) <= identities.keys():
        raise ValueError("Current task inventory lacks matched examples; restore source access.")
    representative = _diverse_ids(
        identities, set(labels), target_n=required_n, seed=seed,
    )
    identity_settings = {
        key: value for key, value in settings.items() if key != "judge_objective_strategy"
    }
    identity_inputs = {
        "claim": claim, "settings": identity_settings,
        "matches": {key: str(value) for key, value in labels.items()},
        "cohort_policy": COHORT_POLICY, "seed": seed, "scope": scope,
    }
    policy = judge_policy_hash(behavior_strictness=strictness)
    identity = fingerprint({**identity_inputs, "policy": policy})
    old_policy = previous_receipt.get("policy_hash")
    compatible_policy = old_policy == policy
    new_receipt_matches = previous_receipt.get("identity") == fingerprint(
        {**identity_inputs, "policy": old_policy})
    recovering = (
        previous_receipt.get("status") == "running" and compatible_policy
        and new_receipt_matches
    )
    cohort = representative
    if recovering:
        # Finish an interrupted request with its original policy and item IDs.
        cohort = list(previous_receipt["item_ids"])
        identity, policy = previous_receipt["identity"], old_policy
    matching_receipt = compatible_policy and new_receipt_matches
    saved_retry_ids = set(previous_receipt.get("retry_item_ids") or []) if recovering else set()
    retry_ids = {
        row["item_id"] for row in valid_rows
        if (row["item_id"] in saved_retry_ids or retry_incomplete and not recovering
            and matching_receipt
            and previous_receipt.get("status") in {"complete", "unavailable"})
        and row["item_id"] in cohort and _saved_completion_pending(rt, row)
    }
    if retry_incomplete and matching_receipt and previous_receipt.get("status") == "unavailable":
        retry_ids.update(set(cohort) - valid_by_id.keys())
    retry_generation = (int(previous_receipt.get("retry_generation") or 0)
                        if matching_receipt or recovering else 0)
    if retry_ids and not recovering:
        from autofa.metrics.review_failures import review_failures

        blocked = {item_id for row in review_failures(valid_rows)
                   if row["retry_policy"] == "requires_backend_or_configuration_change"
                   for item_id in row.get("item_ids", [row["item_id"]])}
        retry_ids.difference_update(blocked)
        # One deliberate repair of a failed stage under the same frozen request.
        # Changing reviewer configuration changes identity and creates a new request.
        if retry_generation >= 1:
            retry_ids.clear()
        if not retry_ids:
            previous_receipt["recovery"] = {
                "status": "blocked", "retry_generation": retry_generation,
                "max_retry_generations": 1,
                "blocked_item_ids": sorted(blocked),
                "reason": ("Failed review stages need a backend/configuration correction; "
                           "the same frozen request will not be rerun indefinitely. "
                           "Completed judgments and the detector are retained."),
            }
        if retry_ids:
            retry_generation += 1
    settled_ids = (
        set(previous_receipt.get("item_ids") or [])
        if matching_receipt and previous_receipt.get("status") in {"complete", "unavailable"}
        else set()
    )
    missing = set(cohort) - items.keys()
    if missing:
        with closing(_fetch_items_by_id(rt.dataset, missing, max_items=None)) as rows:
            items.update({str(item["_item_id"]): item for item in rows})
    if not set(cohort) <= items.keys():
        receipt["reason"] = "Frozen review evidence is missing; restore the original dataset."
        _commit(rt, fn, scores, receipt)
        return receipt, False
    retained = [
        row for row in valid_rows
        if row["item_id"] in cohort
        and row["item_id"] not in retry_ids
        and (not _pending(row) or row["item_id"] in settled_ids)
    ]
    # A separately completed review may have added judgments after this final
    # operation was interrupted. Preserve them without changing its frozen
    # request or losing its checkpoints; reconcile any pending additions next.
    other_rows = [row for row in valid_rows if row["item_id"] not in cohort]
    merged_cohort = cohort
    operation_id = fingerprint(
        {
            "identity": identity,
            "items": [items[key] for key in cohort],
            "localisations": [
                localisations.get(key, fn.failure_localisations.get(key)) for key in cohort
            ],
        }
    )
    receipt.update(identity=identity, operation_id=operation_id, item_ids=cohort,
                   policy_hash=policy,
                   cohort_policy=(previous_receipt.get("cohort_policy") if recovering
                                  else COHORT_POLICY))
    if retry_generation:
        receipt["retry_generation"] = retry_generation
        receipt["retry_item_ids"] = sorted(retry_ids)
    recovery_id = (fingerprint({"operation_id": operation_id,
                               "retry_generation": retry_generation})
                   if retry_generation else operation_id)
    aggregate_is_current = (
        {row.get("item_id") for row in scores.get("behavior_verdicts") or []} == set(cohort)
        and set(cohort) <= valid_by_id.keys()
    )
    unresolved_aggregate = (any(
        _pending(valid_by_id[key]) for key in cohort if key in valid_by_id)
        or scores.get("behavior_purity") is None)
    recorded_assessment = (previous_receipt.get("assessment_recorded") is True
                           or "processing_failures" in previous_receipt)
    failed_before_assessment = (
        previous_receipt.get("status") == "unavailable" and not recorded_assessment
        and not aggregate_is_current
    )
    if (not retry_ids and (aggregate_is_current or failed_before_assessment)
            and previous_receipt.get("operation_id") == operation_id and (
        previous_receipt.get("status") == "unavailable"
        and (unresolved_aggregate or failed_before_assessment)
        or previous_receipt.get("status") == "complete"
    )):
        if failed_before_assessment:
            # The representative operation failed before producing results;
            # rechecking the old cohort's content does not validate that cohort
            # as representative or authorize an automatic backend retry.
            scores["behavior_evidence_stale"] = True
        # Source verification can change diagnostic history independently of
        # the selected cohort. Persist it without losing the completed receipt.
        _commit(rt, fn, scores, previous_receipt)
        return previous_receipt, False
    pending = [key for key in cohort if key not in {row["item_id"] for row in retained}]
    if scope == "round_checkpoint" and pending and not recovering:
        from autofa.metrics.review_scheduling import automatic_review_schedule

        schedule = automatic_review_schedule(scores)
        decision = _prefilter_decision(fn, scores)
        if schedule["status"] == "deferred" or decision["status"] == "skip":
            schedule.update(
                evaluation_claim_hash=scores.get("evaluation_claim_hash"),
                evaluation_run_id=scores.get("evaluation_run_id"),
            )
            scores["review_schedule"] = schedule
            scores["judge_prefilter"] = decision
            receipt.update(
                status="deferred", reason=(decision["reason"] if decision["status"] == "skip"
                                           else schedule["reason"]),
                next_steps=schedule["next_steps"],
            )
            _commit(rt, fn, scores, receipt)
            return receipt, False
    allocation = allocate_review(
        rt.run.run_dir(),
        fn.name,
        [verified_identity_projection(rt, items[key]) for key in cohort],
        target_n=len(cohort),
        seed=0,
        review_key={"full_corpus": "final:", "round_checkpoint": "round:",
                    "selection_checkpoint": "selection:"}[scope] + operation_id,
        phase="development",
        support_review={"retained_item_ids": cohort, "failed_item_ids": [], "min_failed_tasks": 0},
    )
    sample = {
        **allocation["sample_info"],
        "limited": True,
        "reason": "Representative development review; it is not independent confirmation.",
        "scan_scope": "full_corpus_matches" if full_scan else "round_checkpoint_matches",
        "selection": COHORT_POLICY,
        "population_n": len(labels),
        "seed": seed,
        "n_scanned": scores.get("n_scored"),
    }
    receipt.update(status="running", reason="Finishing the frozen representative review cohort.")
    if pending and not aggregate_is_current:
        # Retain old findings, but do not advertise their biased aggregate while
        # a new representative cohort is pending or its judge call fails.
        scores["behavior_evidence_stale"] = True
    _commit(rt, fn, scores, receipt)
    fresh = {}
    review_metadata = scores if compatible or not diagnostic_compatible else diagnostic
    try:
        if pending:
            kwargs = {}
            contract = reusable_contract(review_metadata)
            if (compatible or diagnostic_compatible) and contract:
                kwargs["compiled_contract"] = contract
            fresh = judge_behavior_with_claude(
                name=fn.name,
                source=source,
                description=description,
                mechanism=mechanism,
                items=[items[key] for key in pending],
                assigned_labels=[labels[key] for key in pending],
                failure_localisations=[
                    localisations.get(key, fn.failure_localisations.get(key)) for key in pending
                ],
                cwd=rt.run.run_dir(),
                model=config["model"],
                cli_path=config["cli_path"],
                objective_strategy=settings["judge_objective_strategy"],
                behavior_strictness=strictness,
                **backend_arguments(config["harness"]),
                **reasoning,
                **kwargs,
                workers=setting_value(rt.config, "judge_workers"),
                max_chars=min(5000, rt.config.text_truncate_chars * 2),
                analysis_guidance=guidance_prompt(rt.run),
                _recovery_id=recovery_id,
                _retry_compilation=bool(retry_generation),
            )
        check_cancelled()
        result = _merge_result(review_metadata, fresh, retained, merged_cohort)
    except (asyncio.CancelledError, KeyboardInterrupt, RunInterruptedError):
        # The running receipt freezes membership. Core checkpoints preserve every
        # completed stage; the next finalization resumes the same operation.
        raise
    except Exception as exc:
        from autofa.util.transient import is_transient_error

        if is_transient_error(exc):
            # Review backends normally translate transport failures themselves,
            # but compilation/orchestration can also fail outside that wrapper.
            # Do not permanently settle an interrupted request as unavailable.
            # There is no additional attempt here: explicit recovery resumes the
            # same cohort and its completed stage checkpoints.
            receipt.update(
                status="running", retryable=True,
                reason=("A temporary review service failure interrupted final review. "
                        "Completed stages are saved; resume the same validation to recover."),
            )
            _commit(rt, fn, scores, receipt)
            raise RunInterruptedError(receipt["reason"]) from exc
        receipt.update(
            status="unavailable", reason=f"Final detector review could not finish: {exc}",
        )
        _commit(rt, fn, scores, receipt)
        return receipt, bool(pending)
    # Keep off-cohort counterexamples accessible through existing diagnostic
    # feedback, never in the representative adherence numerator/denominator.
    if other_rows:
        diagnostics = _merge_result(
            review_metadata, {}, other_rows, [row["item_id"] for row in other_rows])
        diagnostics.update(
            review_mode="diagnostic", excluded_from_adherence=True,
            excluded_from_confirmed_support=True, excluded_from_repair_validation=True,
            sample={"selection": "outside_representative_cohort", "limited": True},
            review_localisations={row["item_id"]: localisations.get(row["item_id"])
                                  for row in other_rows},
        )
        scores["diagnostic_review"] = retain_current_diagnostic_feedback(
            rt, fn, scores, diagnostics)
        scores.pop("diagnostic_evidence_stale", None)
    # Preserve obsolete and changed-content judgments as historical evidence.
    history = list(scores.get("behavior_verdict_history") or [])
    for row in [*old_rows, *diagnostic_rows]:
        if row.get("item_id") not in {item["item_id"] for item in valid_rows}:
            if row not in history:
                history.append(deepcopy(row))
    if history:
        scores["behavior_verdict_history"] = history
    apply_review_result(scores, result, judge_harness=config["harness"])
    scores.update(
        behavior_sample={**sample, "actual_n": len(merged_cohort)},
        behavior_review_phase="development",
        behavior_n_sampled=len(merged_cohort),
        behavior_n_fires=len(labels),
        behavior_review_localisations={
            key: localisations.get(key, fn.failure_localisations.get(key)) for key in merged_cohort
        },
        claim_calibration=result.get("claim_calibration"),
        item_calibrations=result.get("item_calibrations") or [],
        calibration_hint=result.get("calibration_hint"),
        behavior_review_operation={"origin_run_id": rt.run.run_id,
                                   "scope": scope, "operation_id": operation_id},
    )
    scores["judge_prefilter"] = {
        "status": "pass",
        "claim_hash": claim,
        "reason": "Current representative review results are available.",
    }
    scores.pop("review_schedule", None)
    scores.pop("confirmation_availability", None)
    receipt.update(
        assessment_recorded=True,
        status=(
            "complete"
            if scores.get("behavior_purity") is not None
            and not scores.get("behavior_n_errors")
            and not scores.get("behavior_n_review_incomplete")
            else "unavailable"
        ),
        reason=_reason(scores),
        processing_failures=scores.get("behavior_processing_failures") or [],
    )
    _commit(rt, fn, scores, receipt)
    if recovering and cohort != representative:
        # Finish the original operation first: changing its cohort/recovery key
        # would abandon saved stages and repeat already completed work. The
        # expansion retains those results and reviews only additional evidence.
        expanded, attempted = _finalize_one(rt, fn, scope=scope)
        return expanded, bool(pending) or attempted
    return receipt, bool(pending)


def final_review_functions(
    *, progress: Callable[[dict], None] | None = None,
    scope: str = "full_corpus", names: list[str] | None = None,
    retry_incomplete: bool = False,
) -> dict[str, Any]:
    """Review a census or completed checkpoint without reusing a biased cohort.

    Explicit ``retry_incomplete`` on named detectors retries failed processing or
    unfinished evidence review in the frozen cohort. Decisive judgments and
    genuinely settled uncertainty are retained. Ordinary finalization never
    repeatedly reruns a settled failed attempt.
    """
    from autofa.mcp.runtime import get_runtime
    from autofa.metrics.live_progress import refresh_display_progress
    from autofa.session.operation_admission import admit_operation

    if scope not in {"full_corpus", "round_checkpoint", "selection_checkpoint"}:
        raise ValueError(f"Unsupported detector review scope: {scope}")
    if type(retry_incomplete) is not bool or retry_incomplete and not names:
        raise ValueError("Explicit incomplete-review recovery requires named detectors")
    rt = get_runtime()
    functions = [fn for fn in rt.library.functions if names is None or fn.name in names]
    reviewed, skipped, unresolved, deferred = [], [], [], []
    if progress and not functions:
        progress({"total": 0, "completed": 0, "detector": "", "phase": "saved"})
    for index, fn in enumerate(functions):
        check_cancelled()
        if progress:
            progress(
                {
                    "total": len(functions),
                    "completed": index,
                    "detector": fn.name,
                    "phase": "reviewing",
                }
            )
        with admit_operation("judge_function", {"name": fn.name}) as busy:
            if busy:
                receipt = {"status": "unavailable", "reason": busy["error"]}
                attempted = False
            else:
                try:
                    receipt, attempted = _finalize_one(
                        rt, fn, scope=scope, retry_incomplete=retry_incomplete)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    receipt = {
                        "claim_hash": claim_hash(
                            source=review_source(fn),
                            description=fn.description or "",
                            mechanism=fn.mechanism or fn.description or fn.name,
                        ),
                        "scope": scope,
                        "status": "unavailable",
                        "reason": f"Final review evidence unavailable: {exc}",
                    }
                    scores = deepcopy(fn.scores or {})
                    # A failed source check cannot continue advertising historical
                    # causal confirmations as verified against the current corpus.
                    if isinstance(exc, OSError):
                        scores["behavior_evidence_stale"] = True
                    _commit(rt, fn, scores, receipt)
                    attempted = False
        (reviewed if attempted else skipped).append(fn.name)
        if receipt["status"] == "unavailable":
            unresolved.append({"name": fn.name, "reason": receipt["reason"]})
        if receipt["status"] == "deferred":
            deferred.append({"name": fn.name, "reason": receipt["reason"],
                             "next_steps": receipt.get("next_steps", [])})
        if progress:
            progress(
                {
                    "total": len(functions),
                    "completed": index + 1,
                    "detector": fn.name,
                    "phase": "saved",
                }
            )
    refresh_display_progress(rt)
    return {
        "ok": not unresolved,
        "reviewed": reviewed,
        "skipped": skipped,
        "unresolved": unresolved,
        "deferred": deferred,
        "n": len(reviewed),
    }
