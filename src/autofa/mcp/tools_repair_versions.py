# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect and explicitly select previously scored detector implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from autofa.core.dependencies import execution_bundle
from autofa.core.fields import get_field_policy
from autofa.core.library import FeatureFunction
from autofa.core.sandbox import compile_detector
from autofa.mcp.runtime import err, get_runtime, ok
from autofa.metrics.version_comparison import _comparison, version_recommendation, version_summary
from autofa.session import repair_workflow
from autofa.session.discovery_target import agent_target_grade_for_runtime
from autofa.session.modes_ledger import upsert_mode
from autofa.session.repair_versions import collect_versions
from autofa.session.status_log import emit_runtime_status
from autofa.util.cancellation import cancellation_guard


def _summary(fingerprint: str, function: dict, current_version: str, **kwargs) -> dict[str, Any]:
    return version_summary(fingerprint, function, current_version, **kwargs)


def list_function_versions(name: str) -> dict[str, Any]:
    """Compare saved scores for exact prior implementations, without evaluating.

    Review samples can differ between versions. Their judged proportions are
    historical evidence, not a controlled comparison of detector quality.
    """
    rt = get_runtime()
    current = rt.library.get(name)
    snapshot = current.to_dict() if current else None
    current_version = repair_workflow._version(snapshot) if snapshot else ""
    try:
        with repair_workflow._ledger(rt.run.run_dir()) as data:
            versions = collect_versions(data, name, snapshot)
    except (OSError, ValueError, TypeError) as exc:
        return err(f"Cannot read detector versions: {exc}")
    if not versions and current is None:
        return err(f"No scored versions found for function: {name}")
    target = agent_target_grade_for_runtime(rt)
    rows = [_summary(key, function, current_version, config=rt.config,
                     run_id=rt.run.run_id, target_grade=target)
            for key, function in versions.items()]
    if snapshot and current_version not in versions:
        rows.append(_summary(current_version, snapshot, current_version, config=rt.config,
                             run_id=rt.run.run_id, target_grade=target))
    by_version = {row["version_hash"]: row for row in rows}
    comparisons = [
        _comparison(function, snapshot, by_version[key], by_version[current_version],
                    config=rt.config, run_id=rt.run.run_id)
        for key, function in versions.items() if snapshot and key != current_version
    ]
    return ok(
        name=name, current_version_hash=current_version,
        versions=rows, comparisons=comparisons,
        recommendation=version_recommendation(name, rows, comparisons, current_version),
        message=("Saved evaluations retain their original errors, uncertainty and review cohorts. "
                 "Selecting a version does not validate it or reset repair attempts. "
                 "Use expected_current_version_hash when restoring; an empty string means "
                 "the function is absent."),
    )


def restore_function_version(
    name: str, version_hash: str, reason: str, expected_current_version_hash: str,
) -> dict[str, Any]:
    """Select an exact scored version without re-running evaluation or judging.

    Finish or explicitly defer any active repair first. Historical reviews and
    frozen helper/dependency sources are preserved; no new validation is claimed.
    An empty expected_current_version_hash requires that the function be absent.
    """
    rt = get_runtime()
    from autofa.session.polishing_guard import polishing_tool_guard

    if blocked := polishing_tool_guard(rt, "restore_function_version", {"name": name}):
        return blocked
    current = rt.library.get(name)
    before = current.to_dict() if current else None
    actual_version = repair_workflow._version(before) if before else ""
    if actual_version != expected_current_version_hash:
        return err("Current detector version changed; list versions again before selecting")
    try:
        with repair_workflow._ledger(rt.run.run_dir()) as data:
            snapshot = collect_versions(data, name, before).get(version_hash)
        if snapshot is None:
            return err("Unknown scored detector version")
        candidate = FeatureFunction.from_dict(snapshot)
        compile_detector(
            candidate.source, bundle=execution_bundle(candidate),
            policy=get_field_policy(rt.config),
        )
    except Exception as exc:  # noqa: BLE001
        return err(f"Saved detector version cannot be restored: {exc}")

    changed = False
    published = False

    def apply(selected: dict, repairs: list[dict]) -> None:
        nonlocal changed, published
        # Compilation can overlap other tool work. Reject changed scores as well
        # as changed code, rather than replace a newly completed evaluation.
        observed = rt.library.get(name)
        if observed is not current or (observed.to_dict() if observed else None) != before:
            raise ValueError("Current detector changed while selecting a saved version")
        if selected != snapshot:
            raise ValueError("Saved detector scores changed while selecting their version")
        restored = FeatureFunction.from_dict(selected)
        rt.library.upsert(restored)
        changed = True
        rt.persist_library(repairs=repairs)
        published = True

    try:
        # Selection holds the repair ledger while its callback saves the
        # library. Take the library lock first, matching ordinary persistence
        # and submission, so unrelated detector commits cannot deadlock.
        with cancellation_guard(), rt._library_lock:
            history_before = deepcopy(rt.library.review_history)
            functions_before = list(rt.library.functions)
            try:
                result = repair_workflow.select_version(
                    rt.run.run_dir(), name, version_hash=version_hash, reason=reason,
                    current_function=before,
                    expected_current_version_hash=expected_current_version_hash,
                    apply=apply,
                )
            except BaseException:
                if changed:
                    rt.library.functions[:] = functions_before
                    rt.library.review_history = history_before
                    if published:
                        # The repair-ledger commit follows the library write. Its
                        # lock has unwound; restore the durable library before an
                        # unrelated save can observe a selection that never committed.
                        rt.persist_library()
                raise
    except (OSError, ValueError, TypeError) as exc:
        return err(f"Detector version selection failed: {exc}")
    selected = rt.library.get(name)
    diagnostics = []
    try:
        upsert_mode(
            rt.run.run_dir(), name=name, description=selected.description,
            mechanism=selected.mechanism, status="active",
            reason=f"Selected historical scored version: {reason.strip()}",
            detector=selected.to_dict(),
        )
        emit_runtime_status(
            f"Selected saved version of {name}; historical scores retained",
            source="restore_function_version", context={"name": name, "version_hash": version_hash},
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics.append(f"Selection committed; mode/status refresh unavailable: {exc}")
    # Selection is committed. Optional historical guidance must never turn a
    # successful restore into an error that invites another mutation.
    try:
        summary = _summary(
            version_hash, selected.to_dict(), version_hash, config=rt.config,
            run_id=rt.run.run_id, target_grade=agent_target_grade_for_runtime(rt),
        )
    except Exception as exc:  # noqa: BLE001
        summary = {"version_hash": version_hash, "current": True, "summary_unavailable": str(exc)}
        diagnostics.append("Selection committed; inspect saved scores before comparing quality.")
    try:
        recovery = _restored_version_guidance(rt, name, selected.to_dict())
    except Exception as exc:  # noqa: BLE001
        recovery = {
            "recovery_guidance_unavailable": f"{type(exc).__name__}: {exc}",
            "next_action": "The version was restored. Inspect get_function_repair and "
                           "detector_repairs.json for lessons before editing it again.",
        }
    return ok(
        name=name, restored_version_hash=version_hash, selection=result.get("selection"),
        version=summary, **recovery, diagnostics=diagnostics,
        message=("Selected the saved detector and its historical scores. Repair attempts, "
                 "review exclusions and prior uncertainty are unchanged. No evaluation or "
                 "fresh validation was performed. Lessons from later edits remain relevant; "
                 "inspect the suggested counterexample checks before the next edit."),
    )


def _restored_version_guidance(rt: Any, name: str, selected: dict) -> dict:
    from autofa.metrics.version_lessons import saved_version_findings, version_lessons

    lessons = version_lessons(rt, name, current_function=selected)
    with repair_workflow._ledger(rt.run.run_dir()) as data:
        findings = saved_version_findings(data, name)
    item_ids = list(dict.fromkeys(
        row["item_id"] for row in findings
        if row.get("category") == "counterexample" and row.get("item_id")
    ))
    fingerprint = repair_workflow._version(selected)
    # Both Python and LLM detectors support these bounded, explicit controls.
    # No predictions or reviews are run as a side effect of selecting a version.
    checks = [{
        "tool": "smoke_test_feature",
        "arguments": {
            "name": name, "expected_version_hash": fingerprint, "item_ids": item_ids[i:i + 30],
        },
    } for i in range(0, len(item_ids), 30)]
    return {
        "retained_lessons": lessons,
        "counterexample_checks": {
            "status": "pending" if item_ids else "no_saved_counterexamples",
            "item_n": len(item_ids),
            "actions": checks,
            "interpretation": (
                "Run the relevant bounded check before another edit; compatible diagnostic "
                "results are reused. A nonmatch is observed exclusion from this version's "
                "bucket. A match means inspect whether the saved finding actually contradicts "
                "the restored claim. Abstentions and errors remain unresolved. Historical "
                "judgments are not transferred, and these checks do not estimate population "
                "coverage or confer causal validation."
            ),
        },
    }
