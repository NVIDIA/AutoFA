# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged, evidence-specific polishing feedback without targeted population estimates."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any

from autofa.session.polishing_generalization import diagnostic_controls

_METRICS = {
    "recall": ("Coverage", "unflagged_failed_attempt",
               "Inspect missed failed attempts; they are not proven false negatives."),
    "ppv_at_pi0": ("Normalized precision", "flagged_success_contrast",
                   "Inspect passing matches for a supported boundary; "
                   "a pass is not a causal rejection."),
    "behavior_purity": ("Adherence", "reviewed_counterexample",
                        "Inspect rejected claim requirements and later recovery "
                        "in the cited reviews."),
}
_GROUP_MEANING = {
    "flagged_success_contrast": (
        "Passing matches reduce failure association, not necessarily adherence."),
    "matched_failed_attempt": (
        "Matched failures to preserve; matches alone do not confirm the claim."),
    "unflagged_failed_attempt": (
        "Missed failed attempts to inspect, not established false negatives."),
    "reviewed_counterexample": (
        "A review rejected the claim; inspect its condition and cited evidence."),
    "reviewed_support": "Reviewed support to preserve under the same claim.",
    "unresolved_review": "Review could not establish the claim; inspect its specific evidence gap.",
    "incomplete_review": "Required evidence was unread; complete evidence inspection.",
    "review_processing_error": (
        "Review execution failed; recover it without assuming a false claim."),
    "execution_error": "Detector execution failed; repair implementation and recheck support.",
    "unresolved_prediction": (
        "Missing prediction may be an error or abstention; inspect the record."),
    "prediction_outcome_unavailable": (
        "Prediction outcome was not retained; do not infer a negative."),
    "fixed_scoring_panel": "Original comparison panel, retained unchanged across revisions.",
    "fresh_unflagged_success": (
        "Previously unflagged passing attempts checked on distinct tasks for this revision; "
        "task novelty is reported separately. These are diagnostic controls, not a population "
        "or independent review sample."),
    "introduced_success_match": (
        "A previous repair introduced a passing match; retain it as a regression control."),
    "retained_diagnostic_problem": (
        "Problem found by an earlier revision; checked again for recovery."),
    "passing_attempt": (
        "Passing attempts checked here, including both prior matches and nonmatches."),
}


def polishing_claim_identity(fn: Any) -> dict:
    """Identify literal claims and executed sources without copying trace evidence."""
    from autofa.core.dependencies import review_source
    from autofa.core.localisation import text_hash
    from autofa.metrics.claim_contract import claim_hash

    source = review_source(fn)
    description = fn.description or ""
    mechanism = fn.mechanism or description or fn.name
    return {"claim_hash": claim_hash(source=source, description=description, mechanism=mechanism),
            "source_sha256": text_hash(source), "description": description, "mechanism": mechanism}


def _counterexample_residuals(
    name: str, controls: dict, diagnostic: dict | None, current_claim: dict | None,
) -> dict:
    """Recheck historical contradictions without transferring a verdict across edits."""
    from autofa.metrics.association import matches_selected_bucket

    current_claim = current_claim or {}
    selection = controls.get("selection")
    current = bool(diagnostic and current_claim.get("claim_hash")
                   and diagnostic.get("claim_hash") == current_claim["claim_hash"]
                   and isinstance(selection, (list, tuple)) and len(selection) == 2)
    outcomes = {row["item_id"]: row for row in (diagnostic or {}).get("results") or []} if current \
        else {}
    residuals, unresolved, excluded = [], [], []
    known_n = 0
    for control in controls.get("controls") or []:
        findings = [row for row in [*(control.get("review_findings") or []),
                                   *(control.get("historical_review_findings") or [])]
                    if row.get("verdict") == "no"]
        if not findings and "reviewed_counterexample" not in control.get("categories", []):
            continue
        known_n += 1
        item_id = control["item_id"]
        outcome = outcomes.get(item_id) or {}
        if outcome.get("status") != "value":
            unresolved.append(item_id)
            continue
        if not matches_selected_bucket(outcome.get("prediction"), selection):
            excluded.append(item_id)
            continue
        provenance = []
        for finding in findings:
            same_revision = finding.get("claim_hash") == current_claim.get("claim_hash")
            has_text = all(key in finding for key in ("claim_description", "claim_mechanism"))
            same_text = has_text and all(finding["claim_" + key] == current_claim.get(key)
                                         for key in ("description", "mechanism"))
            provenance.append({
                "review_claim_hash": finding.get("claim_hash"),
                "claim_compatibility": ("same_revision" if same_revision else
                                        "same_literal_claim_changed_revision" if same_text else
                                        "changed_literal_claim" if has_text else "unverified"),
                "detector_source_compatibility": (
                    "same" if same_revision or (finding.get("source_sha256")
                                               and finding["source_sha256"]
                                               == current_claim.get("source_sha256")) else
                    "changed" if finding.get("source_sha256") else "unverified"),
                "review_evidence_status": ("historical_unverified" if finding.get("historical")
                                           else "retained_control_evidence"),
                "reason_preview": str(finding.get("reason") or "")[:300],
                "reason_truncated": len(str(finding.get("reason") or "")) > 300,
            })
        prediction = _case(control, outcome, compact=True)["diagnostic"]
        residuals.append({
            "item_id": item_id, "task_id": control.get("task_id"),
            **{key: value for key, value in prediction.items() if key.startswith("prediction")},
            "prediction_current": True,
            "prior_findings": provenance[:2], "prior_findings_n": len(provenance),
            "full_evidence_action": f"get_function_feedback(name={name!r}, item_id={item_id!r})",
        })
    residuals.sort(key=lambda row: row["item_id"])
    return {
        "known_case_n": known_n, "still_matching_n": len(residuals),
        "excluded_n": len(excluded), "unresolved_n": len(unresolved),
        "current_predictions_available": current,
        "status": ("requires_reconciliation" if residuals else "checks_pending" if unresolved
                   else "no_remaining_matches" if known_n else "no_saved_counterexamples"),
        "cases_preview": residuals[:5], "preview_complete": len(residuals) <= 5,
        "current_claim_hash": current_claim.get("claim_hash"),
        "current_negative_verdicts_inferred": False, "changes_adherence": False,
        "next_action": (
            "Reconcile these still-matching cases before claiming the repair succeeded. Inspect "
            "the saved rejected requirement, full evidence and later recovery against the current "
            "claim. Either fix the general predicate or explain with evidence why the old finding "
            "does not apply; report remaining counterexamples as explicit limitations. Use "
            "current-claim diagnostic review if the distinction remains unresolved. "
            "Removing a smoke expectation or changing a comment does not resolve "
            "the finding. A historical rejection is not automatically a current negative verdict."
            if residuals else
            "Finish current-version checks for saved counterexamples; missing predictions are "
            "not evidence that the repair excludes them." if unresolved else
            "Retain the regression evidence. Exclusion alone does not establish population "
            "quality, and historical verdicts are not transferred to a changed claim."
        ),
    }


def _rate(value: Any) -> float | None:
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and 0 <= value <= 1 else None)


def _support_count(value: Any) -> float | None:
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value) and value >= 0 else None)


def _representative_rows(rows: list[dict], categories: list[str], limit: int) -> list[dict]:
    """Choose reproducible, task-diverse illustrations without discarding any controls."""
    selected, seen_ids, seen_tasks, seen_benchmarks = [], set(), set(), set()

    def choose(candidates: list[dict]) -> None:
        candidates = [row for row in candidates if row["item_id"] not in seen_ids]
        if not candidates or len(selected) >= limit:
            return
        row = min(candidates, key=lambda row: (
            row.get("task_id") is not None and row["task_id"] in seen_tasks,
            row.get("benchmark") in seen_benchmarks,
            not bool(row.get("review_findings")),
            row.get("new_since_checkpoint") is not True,
            row["item_id"],
        ))
        selected.append(row)
        seen_ids.add(row["item_id"])
        if row.get("task_id") is not None:
            seen_tasks.add(row["task_id"])
        seen_benchmarks.add(row.get("benchmark"))

    # Include relevant problem groups and a support contrast before filling the
    # preview. These are illustrative cases, never an independent review sample.
    support_categories = {"reviewed_support", "matched_failed_attempt"}
    support = [row for row in rows if support_categories & set(row.get("categories", []))]
    for category in categories:
        if (len(selected) == limit - 1 and support
                and not any(support_categories & set(row.get("categories", []))
                            for row in selected)):
            choose(support)
        choose([row for row in rows if category in row.get("categories", [])])
    candidates = [row for row in rows if set(row.get("categories", [])) & set(categories)]
    for _ in range(limit - len(selected)):
        choose(candidates or rows)
    return selected


def _groups(controls: dict, diagnostic: dict | None) -> dict:
    ids: dict[str, list[str]] = defaultdict(list)
    by_id = {row["item_id"]: row for row in controls.get("controls") or []}
    for row in controls.get("controls") or []:
        categories = list(row.get("categories") or [])
        if row.get("benchmark_outcome") == "passed":
            categories.append("passing_attempt")
        for category in categories:
            for key in (category, f"benchmark:{row.get('benchmark') or 'unknown'}:{category}"):
                ids[key].append(row["item_id"])
    result = {}
    diagnostics = (diagnostic or {}).get("groups") or {}
    for key in sorted(set(ids) | set(diagnostics)):
        group = diagnostics.get(key) or {}
        new_ids = [item_id for item_id in ids[key]
                   if by_id[item_id].get("new_since_checkpoint") is True]
        summary = {
            "control_n": len(ids[key]), "item_ids_preview": ids[key][:3],
            "distinct_task_n": (
                len({by_id[item_id]["task_id"] for item_id in ids[key]})
                if all(by_id[item_id].get("task_id") is not None for item_id in ids[key])
                else None),
            "new_since_checkpoint": {"n": len(new_ids), "item_ids_preview": new_ids[:3]},
            "meaning": _GROUP_MEANING.get(key.rsplit(":", 1)[-1], "Inspect the saved findings."),
        }
        summary.update({field: group[field] for field in (
            "n", "resolved", "flagged", "previous_flagged",
        ) if field in group})
        for field in ("added_matches", "lost_matches", "unresolved"):
            values = group.get(field)
            if isinstance(values, list):
                summary[field] = {"n": len(values), "item_ids_preview": values[:3]}
        result[key] = summary
    return result


def _case(control: dict, outcome: dict | None, *, compact: bool = False) -> dict:
    row = {key: control[key] for key in (
        "item_id", "task_id", "benchmark", "categories", "baseline_label",
        "baseline_claim_hash", "new_since_checkpoint", "execution_error", "review_findings",
    ) if key in control}
    if not compact and control.get("historical_review_findings"):
        row["historical_review_findings"] = control["historical_review_findings"]
    if compact:
        findings = row.pop("review_findings", [])
        row["review_findings_n"] = len(findings)
        row["review_reason_preview"] = [str(finding.get("reason") or "")[:200]
                                         for finding in findings[:2]]
    row["diagnostic"] = ({key: outcome[key] for key in (
        "status", "prediction", "error",
    ) if key in outcome} if outcome else {"status": "not_run_for_current_version"})
    if compact:
        for record, fields in ((row, ("baseline_label", "execution_error")),
                               (row["diagnostic"], ("prediction", "error"))):
            for field in fields:
                if isinstance(record.get(field), str) and len(record[field]) > 300:
                    record[field] = record[field][:300]
                    record[field + "_truncated"] = True
    return row


def _passing_match_changes(controls: dict, diagnostic: dict | None) -> dict:
    group = ((diagnostic or {}).get("groups") or {}).get("passing_attempt") or {}
    from_full = not group and bool(controls.get("full_validation_passing_match_changes"))
    if from_full:
        group = controls["full_validation_passing_match_changes"]
    added, removed = group.get("added_matches") or [], group.get("lost_matches") or []
    fresh = ((diagnostic or {}).get("groups") or {}).get("fresh_unflagged_success") or {}
    sampling = (diagnostic or {}).get("generalization_sample") or {}
    unresolved = len(group.get("unresolved") or [])
    return {
        "available": bool(group), "previous_matches": group.get("previous_flagged"),
        "scope": "successive full validations" if from_full else "targeted diagnostic controls",
        "compared_n": group.get("n"),
        "complete": bool(group.get("complete", (diagnostic or {}).get("complete", False)))
        and not unresolved,
        "current_matches": group.get("flagged"),
        "removed": {"n": len(removed), "item_ids_preview": removed[:5]},
        "introduced": {"n": len(added), "item_ids_preview": added[:5]},
        "introduced_on_fresh_controls": len(fresh.get("added_matches") or []),
        "unresolved_n": unresolved,
        "generalization_sample": sampling,
        "summary": (
            f"On checked passing attempts: removed {len(removed)} previous matches and introduced "
            f"{len(added)} new matches. Inspect both groups before treating the repair as an "
            "improvement. Passing matches are association counterexamples, not automatic causal "
            "rejections. " + ("This compares the two saved corpus scans."
                              if from_full else
                              "These diagnostic counts are not population precision.")
            + (f" {unresolved} passing attempts have no comparable resolved outcome; "
               "the comparison is incomplete." if unresolved else "")
            if group else "No current passing-match comparison is available."),
    }


def _panel_revision_comparison(
    controls: dict, diagnostic: dict | None, current_claim: dict | None,
) -> dict:
    """Compare retained predictions on the exact panel, never the targeted union."""
    from autofa.metrics.association import contingency, matches_selected_bucket

    panel = controls.get("panel_item_ids") or []
    selection = controls.get("selection")
    diagnostic = diagnostic or {}
    result = {
        "scope": "observed_fixed_panel", "population_metrics": False,
        "weighting": "unweighted; descriptive rates on the same retained panel only",
        "cohort_fingerprint": controls.get("panel_fingerprint"), "panel_n": len(panel),
        "before_claim_hash": controls.get("claim_hash"),
        "after_claim_hash": (current_claim or {}).get("claim_hash"),
        "comparable": False, "metric_changes": {},
        "interpretation": (
            "These rates compare revisions on one fixed set. They are not weighted population "
            "estimates, adherence scores, or a grade. Compare new scope separately on fresh "
            "controls. The saved full-corpus baseline and a current sample cannot establish "
            "a revision improvement or regression by direct subtraction."
        ),
    }
    if (not panel or len(set(panel)) != len(panel) or not result["cohort_fingerprint"]
            or not result["before_claim_hash"] or not result["after_claim_hash"]
            or diagnostic.get("claim_hash") != result["after_claim_hash"]
            or not controls.get("source_signature")
            or diagnostic.get("source_signature") != controls["source_signature"]
            or not controls.get("run_id") or diagnostic.get("run_id") != controls["run_id"]
            or not controls.get("controls_fingerprint")
            or diagnostic.get("controls_fingerprint") != controls["controls_fingerprint"]
            or diagnostic.get("baseline_claim_hash") != result["before_claim_hash"]
            or not isinstance(selection, (tuple, list)) or len(selection) != 2):
        return {**result, "reason": "Current same-panel prediction provenance is unavailable."}
    rows = {row["item_id"]: row for row in controls.get("controls") or []}
    predictions = {row["item_id"]: row for row in diagnostic.get("results") or []}
    if (len(rows) != len(controls.get("controls") or [])
            or len(predictions) != len(diagnostic.get("results") or [])):
        return {**result, "reason": "Repeated prediction or control IDs prevent panel comparison."}
    sides = {}
    for side in ("before", "after"):
        counts = dict.fromkeys(("a", "b", "c", "d"), 0)
        unresolved = 0
        for item_id in panel:
            row, prediction = rows.get(item_id) or {}, predictions.get(item_id) or {}
            outcome = row.get("benchmark_outcome")
            valid = ("baseline_label" in row and row.get("baseline_claim_hash")
                     == result["before_claim_hash"] if side == "before" else
                     prediction.get("status") == "value")
            if not valid or outcome not in {"failed", "passed"}:
                unresolved += 1
                continue
            label = row["baseline_label"] if side == "before" else prediction.get("prediction")
            matched = matches_selected_bucket(label, selection)
            counts[("a" if matched else "c") if outcome == "failed" else
                   ("b" if matched else "d")] += 1
        rates = contingency(**counts)
        sides[side] = {**counts, "resolved_n": sum(counts.values()),
                       "unresolved_n": unresolved,
                       "recall": rates["recall"] if counts["a"] + counts["c"] else None,
                       "ppv_at_pi0": rates["ppv_at_pi0"]
                       if counts["a"] + counts["c"] and counts["b"] + counts["d"] else None}
    comparable = not any(side["unresolved_n"] for side in sides.values())
    changes = {}
    for key in ("recall", "ppv_at_pi0"):
        before, after = sides["before"][key], sides["after"][key]
        delta = after - before if comparable and before is not None and after is not None else None
        changes[key] = {"before": before, "after": after, "delta": delta,
                        "direction": ("unavailable" if delta is None else "gain" if delta > 0
                                      else "regression" if delta < 0 else "unchanged")}
    return {**result, **sides, "comparable": comparable, "metric_changes": changes,
            "reason": None if comparable else (
                "Some panel outcomes or predictions are unresolved. The displayed rates use "
                "each revision's resolved rows; no directional revision delta is established.")}


def _current_score_scope(scores: dict | None) -> dict:
    from autofa.metrics.live_progress import _evaluation_cohort

    scores = scores or {}
    cohort = _evaluation_cohort(scores)
    return {"scope": cohort[2] if cohort else "unidentified",
            "sampled": scores.get("sampled"), "sample_n": scores.get("sample_n"),
            "sample_frame_n": scores.get("sample_frame_n"), "n_scored": scores.get("n_scored"),
            "claim_hash": scores.get("evaluation_claim_hash"),
            "note": "Current qualification uses these scores, not targeted diagnostic rates."}


def compact_polishing_feedback(
    name: str, controls: dict, diagnostic: dict | None, *,
    baseline_grades: dict | None = None, current_scores: dict | None = None,
    current_claim: dict | None = None,
) -> dict:
    """Keep summaries bounded; all controls and full reasons remain paged by the tool."""
    controls = diagnostic_controls(controls, diagnostic)
    revision = _panel_revision_comparison(controls, diagnostic, current_claim)
    residuals = _counterexample_residuals(name, controls, diagnostic, current_claim)
    groups = _groups(controls, diagnostic)
    changes = {}
    for metric, (label, category, action) in _METRICS.items():
        saved = (controls.get("initial_score_changes") or controls.get("score_changes") or {}).get(
            metric) or {}
        before, after = _rate(saved.get("before")), _rate(saved.get("after"))
        changes[metric] = {
            "label": label, "before": before, "after": after,
            "delta": after - before if before is not None and after is not None else None,
            "evidence_group": category, "next_action": action,
            "item_ids_preview": groups.get(category, {}).get("item_ids_preview", []),
        }
    support = (controls.get("initial_task_support_change")
               or controls.get("task_support_change") or {})
    support_scores = (controls.get("initial_score_changes") or controls.get("score_changes") or {}
                      ).get("task_support_comparison") or {}
    before = _support_count(support_scores.get("before"))
    after = _support_count(support_scores.get("after"))
    changes["task_support_comparison"] = {
        "label": "Matched failed-task support", "before": before, "after": after,
        "delta": after - before if before is not None and after is not None else None,
        "before_basis": (support.get("before") or {}).get("task_support_comparison_basis"),
        "after_basis": (support.get("after") or {}).get("task_support_comparison_basis"),
        "before_observed": (support.get("before") or {}).get("n_matched_failed_tasks"),
        "after_observed": (support.get("after") or {}).get("n_matched_failed_tasks"),
        "evidence_group": "matched_failed_attempt",
        "item_ids_preview": groups.get("matched_failed_attempt", {}).get("item_ids_preview", []),
        "next_action": (
            "Inspect matched failures across distinct source tasks. Estimated support can fall "
            "when a full scan establishes observed breadth. Use the canonical matched-task "
            "assessment, not review counts or repeated attempts; unavailable support is not zero."
        ),
    }
    old_grade = next((row.get("code") for row in (baseline_grades or {}).get("functions", [])
                      if row.get("name") == name), None)
    from autofa.metrics.discovery_grades import grade_gap_for_scores

    gap = (grade_gap_for_scores(current_scores, target_grade="GOOD")
           if current_scores is not None else None)
    blockers = [
        {key: dimension[key] for key in ("key", "label", "current", "required", "operator")}
        for dimension in (gap or {}).get("dimensions", []) if dimension["met"] is not True
    ]
    # Current requirements govern the next action. The original downgrade is
    # retained as history and must not tell a successful repair to fix it again.
    focus = ([row["key"] for row in blockers] if gap is not None else
             [key for key, change in changes.items()
              if change["delta"] is not None and change["delta"] < 0])
    categories = [changes["task_support_comparison" if key == "matched_failed_tasks" else key][
        "evidence_group"] for key in focus]
    if (diagnostic or {}).get("errors") or (current_scores or {}).get("errors"):
        categories.insert(0, "execution_error")
    if "behavior_purity" in focus:
        categories += ["unresolved_review", "incomplete_review", "review_processing_error"]
    if ((diagnostic or {}).get("groups") or {}).get("fresh_unflagged_success", {}).get(
            "added_matches"):
        categories.insert(0, "fresh_unflagged_success")
    if (controls.get("full_validation_passing_match_changes") or {}).get("added_matches"):
        categories.insert(0, "introduced_success_match")
    categories += ["reviewed_support", "matched_failed_attempt"]
    if gap is None and not focus:
        categories += [key for key in groups if not key.startswith("benchmark:")]
    categories = list(dict.fromkeys(key for key in categories
                                    if groups.get(key, {}).get("control_n")))
    shown_groups = {key: groups[key] for key in categories[:3]}
    drops = []
    for key, change in changes.items():
        if change["delta"] is None or change["delta"] >= 0:
            continue
        values = (f"{change['before']:g} → {change['after']:g}"
                  if key == "task_support_comparison" else
                  f"{change['before']:.1%} → {change['after']:.1%}")
        drops.append(f"{change['label']} {values}")
    after_grade = controls.get("initial_validation_grade", controls.get("baseline_grade"))
    validation_change = (
        f"Original validation: {old_grade or 'unknown grade'} → {after_grade or 'unknown grade'}. "
        + ("; ".join(drops) + "." if drops else "No measured score decrease is recorded.")
    )
    outcomes = {row["item_id"]: row for row in (diagnostic or {}).get("results") or []}
    rows = controls.get("controls") or []
    if gap is not None and gap["target_met"] and not categories:
        rows = []
    preview = [_case(row, outcomes.get(row["item_id"]), compact=True)
               for row in _representative_rows(rows, categories, 5)]
    for row in preview:
        row["full_evidence_action"] = (
            f"get_function_feedback(name={name!r}, item_id={row['item_id']!r})")
    return {
        "name": name, "phase": "polishing", "target_grade": "GOOD",
        "summary": {
            "validation_change": validation_change,
            "counterexample_checks": (
                f"{residuals['still_matching_n']} saved counterexample cases still match the "
                f"current detector; {residuals['unresolved_n']} checks remain unresolved. "
                "These need evidence-based reconciliation; no historical verdict was "
                "transferred into current adherence."
                if residuals["known_case_n"] else "No saved reviewed counterexamples."
            ),
            "current_goal": (gap["summary"] if gap is not None else
                             "Current qualification is unavailable here; inspect current scores."),
            "current_blockers": blockers,
            "current_score_scope": _current_score_scope(current_scores),
            "revision_comparison": (
                "A same-panel revision comparison is available below; it is separate from "
                "the original discovery-to-validation score change."
                if revision["comparable"] else
                "Same-panel revision comparison unavailable: " + revision["reason"]
            ),
            "passing_match_changes": _passing_match_changes(controls, diagnostic),
            "investigation": (
                residuals["next_action"] if residuals["still_matching_n"] else
                "Current scores meet Good requirements. Preserve the supported version and "
                "finish the required verification; the original downgrade is historical."
                if gap is not None and gap["target_met"] else
                "Use representative problem cases and supported matches to identify a general "
                "distinction to test. Group counts locate evidence; they do not establish the "
                "mechanism or prescribe the implementation. Missing evidence needs retrieval "
                "or an explicit limitation, not an invented predicate change."
            ),
        },
        "scope": "targeted_polishing_controls", "population_metrics": False,
        "known_counterexamples": residuals,
        "revision_comparison": revision,
        "full_corpus_baseline": {
            "scope": "full_selected_corpus", "claim_hash": controls.get("claim_hash"),
            "metrics": controls.get("baseline_scores") or {},
            "note": "Historical full-validation scores; these are not fixed-panel scores.",
        },
        "available": bool(controls.get("available")), "complete": bool(controls.get("complete")),
        "total": len(controls.get("item_ids") or []),
        "panel_n": len(controls.get("panel_item_ids") or []),
        "unavailable_reasons": controls.get("unavailable_reasons") or [],
        "downgrade": {"before_grade": old_grade, "after_grade": after_grade,
                      "metric_changes": changes,
                      "historical": True,
                      "revision_comparison": False,
                      "interpretation": (
                          "Original discovery/prior-validation scores versus full validation. "
                          "Deltas describe scoring-stage changes, not a repair's effect. "
                          "Use revision_comparison for a same-panel revision comparison."),
                      "scope": "saved validation change, not targeted regression performance"},
        "groups": shown_groups, "cases_preview": preview,
        "group_summary": {"shown": len(shown_groups), "total": len(groups),
                          "complete": len(shown_groups) == len(groups),
                          "note": "Detailed feedback includes every category and benchmark group."},
        "diagnostic": {"status": "complete" if (diagnostic or {}).get("complete") else
                       "partial" if diagnostic else "not_run_for_current_version",
                       **{key: diagnostic[key] for key in (
                           "completed", "n", "errors", "abstentions", "error",
                       ) if diagnostic and key in diagnostic}},
        "evidence_action": f"get_function_feedback(name={name!r}, offset=0, limit=10)",
        "guidance": (
            "Start with this summary and representative cases; retrieve full findings when needed. "
            "Every control remains in runtime regression checks, including cases omitted here. "
            "These selected cases do not produce population rates or grades. A passing match is "
            "not itself a rejected mechanism, and an unflagged failure is not a proven "
            "false negative. Benchmark groups describe data composition, not causal mechanisms. "
            "Do not memorize task IDs, answers or incidental wording. State a general rule and "
            "use existing evidence tools to test it on other tasks and previously uninspected "
            "cases when available; report the limit if none are available. This preview is not "
            "an independent review sample. Preserve supported matches and keep population "
            "estimates separate; final full validation determines the report grade."
        ),
    }


def polishing_feedback(
    rt: Any, fn: Any, *, offset: int = 0, limit: int = 10, item_id: str | None = None,
    summary_only: bool = False,
) -> dict | None:
    """Return all saved controls/outcomes through a stable, bounded per-detector page."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("offset must be nonnegative and limit must be between 1 and 50")
    from autofa.behavior_policy import strictness_for_config
    from autofa.core.dependencies import review_source
    from autofa.metrics.verdicts import current_behavior_scores
    from autofa.session.polishing_evaluation import current_polishing_evaluation
    from autofa.session.polishing_guard import active_polishing

    phase = active_polishing(rt)
    if not phase or fn.name not in phase["frozen_names"]:
        return None
    controls = (phase.get("controls") or {}).get(fn.name) or {}
    diagnostic = current_polishing_evaluation(rt, fn, controls)
    result = compact_polishing_feedback(
        fn.name, controls, diagnostic, baseline_grades=phase.get("baseline_grades"),
        current_claim=polishing_claim_identity(fn),
        current_scores=current_behavior_scores(
            fn.scores, source=review_source(fn), description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
            behavior_strictness=strictness_for_config(rt.config),
        ),
    )
    if summary_only:
        return result
    controls = diagnostic_controls(controls, diagnostic)
    result["groups"] = _groups(controls, diagnostic)
    result["group_summary"].update(shown=len(result["groups"]), complete=True)
    results = {row["item_id"]: row for row in (diagnostic or {}).get("results") or []}
    rows = sorted((row for row in controls.get("controls") or []
                   if item_id is None or row["item_id"] == item_id), key=lambda row: row["item_id"])
    result.pop("cases_preview", None)
    result["cases"] = [_case(row, results.get(row["item_id"]))
                       for row in rows[offset:offset + limit]]
    result["pagination"] = {
        "offset": offset, "limit": limit, "total": len(rows),
        "next_offset": offset + limit if offset + limit < len(rows) else None,
    }
    return result
