# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic cumulative repair diagnostics, separate from population scoring."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any

from autofa.core.dependencies import review_source
from autofa.core.sandbox import DetectorAbstained
from autofa.metrics.association import matches_selected_bucket
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.judge_checkpoints import fingerprint
from autofa.session.polishing_generalization import (
    diagnostic_controls,
    prepare_diagnostic_controls,
)
from autofa.session.status_log import emit_runtime_status
from autofa.session.validation_controls import polishing_controls_identity
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import check_cancelled


def _controls_identity(controls: dict) -> str:
    return polishing_controls_identity(controls)


def _result_path(rt: Any, claim: str, context: str, controls: dict) -> Path:
    identity = fingerprint({"claim_hash": claim, "source_signature": context,
                            "controls": controls.get("controls_fingerprint")})
    return Path(rt.run.run_dir()) / "polishing_evaluations" / f"{identity}.json"


def current_polishing_evaluation(rt: Any, fn: Any, controls: dict) -> dict | None:
    """Read current diagnostic feedback; never use it as a population score."""
    from autofa.mcp.tools_eval import _evaluation_context

    try:
        context = _evaluation_context(rt)
        if (not context or controls.get("source_signature") != context
                or controls.get("run_id") != rt.run.run_id
                or controls.get("controls_fingerprint") != _controls_identity(controls)):
            return None
        claim = claim_hash(source=review_source(fn), description=fn.description or "",
                           mechanism=fn.mechanism or fn.description or fn.name)
        result = json.loads(_result_path(rt, claim, context, controls).read_text())
    except (OSError, TypeError, ValueError):
        return None
    if (not isinstance(result, dict)
            or result.get("claim_hash") != claim or result.get("run_id") != rt.run.run_id
            or result.get("source_signature") != context
            or result.get("controls_fingerprint") != controls.get("controls_fingerprint")
            or (claim != controls.get("claim_hash")
                and (result.get("generalization_sample") or {}).get("version") != 1)
            or ("additional_controls" in result and result.get(
                "additional_controls_fingerprint") != fingerprint({
                    "controls": result["additional_controls"],
                    "sampling": result.get("generalization_sample")}))
            or result.get("scope") != "targeted_polishing_controls"):
        return None
    return result


def _diagnostic_groups(rows: list[dict], controls: dict, selection: Any) -> dict:
    by_id = {row["item_id"]: row for row in controls["controls"]}
    groups: dict[str, dict] = {}
    for row in rows:
        old = by_id[row["item_id"]]
        matched = (matches_selected_bucket(row.get("prediction"), selection)
                   if row["status"] == "value" else None)
        before = (matches_selected_bucket(old["baseline_label"], controls["selection"])
                  if "baseline_label" in old else None)
        categories = list(old["categories"])
        if old.get("benchmark_outcome") == "passed":
            categories.append("passing_attempt")
        for category in categories:
            for key in (category, f"benchmark:{old.get('benchmark') or 'unknown'}:{category}"):
                group = groups.setdefault(key, {
                    "n": 0, "resolved": 0, "flagged": 0, "previous_flagged": 0,
                    "added_matches": [], "lost_matches": [], "unresolved": [],
                    "new_since_checkpoint": [],
                })
                group["n"] += 1
                if old.get("new_since_checkpoint") is True:
                    group["new_since_checkpoint"].append(row["item_id"])
                if before is True:
                    group["previous_flagged"] += 1
                if matched is None:
                    group["unresolved"].append(row["item_id"])
                    continue
                group["resolved"] += 1
                group["flagged"] += int(matched)
                if before is not None and matched != before:
                    group["added_matches" if matched else "lost_matches"].append(row["item_id"])
    return groups


def run_polishing_evaluation(
    rt: Any, fn: Any, controls: dict, *, progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run a revision on the fixed panel plus all retained repair controls.

    Execution uses the normal bounded worker pool and prediction cache. Only the
    diagnostic artifact is written: association scores, judge scores, source
    localisations, grade and population panel are left untouched.
    """
    from autofa.mcp.tools_eval import _evaluation_context
    from autofa.session.operation_admission import admit_operation

    check_cancelled()
    source = review_source(fn)
    claim = claim_hash(source=source, description=fn.description or "",
                       mechanism=fn.mechanism or fn.description or fn.name)
    base = {"ok": False, "scope": "targeted_polishing_controls", "name": fn.name,
            "claim_hash": claim, "baseline_claim_hash": controls.get("claim_hash"),
            "population_metrics": False, "grade": None, "results": [], "groups": {}}
    ids = controls.get("item_ids")
    try:
        context = _evaluation_context(rt)
    except (OSError, TypeError, ValueError) as exc:
        return {**base, "error": f"Polishing source unavailable: {exc}"}
    if (not controls.get("available") or controls.get("scope") != "targeted_polishing_controls"
            or controls.get("run_id") != rt.run.run_id
            or controls.get("controls_fingerprint") != _controls_identity(controls)
            or not context or controls.get("source_signature") != context
            or not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
            or {row.get("item_id") for row in controls.get("controls", [])} != set(ids)
            or not set(controls.get("panel_item_ids") or []) <= set(ids)):
        return {**base, "error": "Current verified polishing controls and their original panel "
                "are required; restore the full-validation checkpoint."}
    path = _result_path(rt, claim, context, controls)

    with admit_operation("evaluate_functions", {"names": [fn.name]}) as busy:
        if busy:
            return {**base, "error": busy["error"]}
        return _execute_polishing(rt, fn, controls, base, claim, context, path, progress=progress)


def _execute_polishing(
    rt: Any, fn: Any, controls: dict, base: dict, claim: str, context: str, path: Path, *,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    from autofa.mcp.tools_eval import (
        _evaluation_context,
        _fetch_items_by_id,
        _iter_scored_items,
        _llm_adapter,
        _llm_eval_settings,
    )

    previous = current_polishing_evaluation(rt, fn, controls)
    extra = prepare_diagnostic_controls(rt, fn.name, claim, controls, path)
    controls = diagnostic_controls(controls, extra)
    ids = controls["item_ids"]
    rows = deepcopy((previous or {}).get("results") or [])
    if (len({row.get("item_id") for row in rows}) != len(rows)
            or any(row.get("item_id") not in ids or row.get("status") not in {
                "value", "error", "abstained"} for row in rows)
            or (previous and previous.get("additional_controls_fingerprint")
                != extra["additional_controls_fingerprint"])):
        rows = []
    total = len(ids)
    completed_ids = {row["item_id"] for row in rows if row["status"] != "error"}
    retry_ids = {row["item_id"] for row in rows if row["status"] == "error"}
    controls_by_id = {row["item_id"]: row for row in controls["controls"]}
    _, _, workers = _llm_eval_settings(rt.config)
    result = {
        **base, **extra, "run_id": rt.run.run_id, "source_signature": context,
        "controls_fingerprint": controls.get("controls_fingerprint"),
        "result_path": str(path), "n": total, "panel_n": len(controls["panel_item_ids"]),
        "controls_complete": controls.get("complete", False),
        "control_limitations": controls.get("unavailable_reasons") or [],
        "validation_score_changes": controls.get("score_changes") or {},
        "guidance": "These counts describe selected repair cases, not population rates or a "
                    "quality grade. Losing a passing match is not proof of a corrected false "
                    "positive; inspect the behavior and causal findings. Preserve supported "
                    "positives and inspect newly added passing matches, including on fresh "
                    "diagnostic tasks. Compare removals against introductions; fixing old "
                    "counterexamples can create different errors. Final full validation determines "
                    "population scores and the report grade.",
    }

    def checkpoint(*, complete: bool = False) -> dict:
        errors = sum(row["status"] == "error" for row in rows)
        abstentions = sum(row["status"] == "abstained" for row in rows)
        missing = sorted(set(ids) - completed_ids)
        saved = {**result, "ok": complete and not errors and not missing,
                 "complete": complete and not missing, "completed": len(rows),
                 "errors": errors, "abstentions": abstentions, "missing_item_ids": missing,
                 "results": rows,
                 "groups": _diagnostic_groups(rows, controls, controls["selection"])}
        atomic_write_json(path, saved)
        return saved

    def update(*, finished: bool = False) -> None:
        event = {"detector": fn.name, "completed": len(rows), "total": total,
                 "phase": "polishing", "result_path": str(path), "finished": finished}
        emit_runtime_status(
            f"Polishing {fn.name}: {len(rows)}/{total} repair cases checked "
            "(fixed panel + cumulative issues + new passing controls).",
            source="polishing_evaluation", context=event,
        )
        if progress:
            progress(event)

    # Freeze the per-revision sample before a detector call can be interrupted.
    checkpoint()
    update()
    try:
        with closing(_fetch_items_by_id(rt.dataset, set(ids) - completed_ids,
                                        max_items=None)) as items:
            with closing(_iter_scored_items(
                ((fn, item) for item in items), workers=workers, llm=_llm_adapter(),
            )) as outcomes:
                for _, item, prediction, error in outcomes:
                    check_cancelled()
                    item_id = str(item.get("_item_id") or "")
                    if item_id in completed_ids or item_id not in ids:
                        raise ValueError("Polishing source returned inconsistent item IDs.")
                    completed_ids.add(item_id)
                    row = {"item_id": item_id, "status": "value", "prediction": prediction}
                    control = controls_by_id[item_id]
                    row.update({key: control[key] for key in (
                        "categories", "benchmark", "baseline_label", "review_findings",
                    ) if key in control})
                    if error is not None:
                        row.update(status="abstained" if isinstance(error, DetectorAbstained)
                                   else "error", prediction=None,
                                   error=f"{type(error).__name__}: {error}")
                    # Preserve an old processing error until its retry actually
                    # returns; a second early interruption must not erase it.
                    if item_id in retry_ids:
                        rows[:] = [saved for saved in rows if saved["item_id"] != item_id]
                        retry_ids.discard(item_id)
                    rows.append(row)
                    if len(rows) % 25 == 0 or len(rows) == total:
                        checkpoint()
                        update()
        if _evaluation_context(rt) != context:
            raise ValueError("Polishing source/settings changed during diagnostic evaluation.")
        saved = checkpoint(complete=True)
        update(finished=True)
        return saved
    except BaseException:
        # Prediction checkpoints preserve completed expensive calls. Keep
        # partial diagnostic results too, without publishing population scores.
        checkpoint()
        raise
