# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval / smoke / judge MCP tools."""

from __future__ import annotations

import hashlib
import json
import math
import random
import zlib
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import closing, nullcontext
from copy import copy, deepcopy
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.core.dependencies import (
    build_bundle,
    execution_bundle,
    node_bundle,
    review_source,
    validate_edges,
)
from autofa.core.fields import get_field_policy
from autofa.core.sandbox import DetectorAbstained, RegexTimeoutError, compile_detector, run_detector
from autofa.data.parquet_store import _is_correct
from autofa.defaults import default_value, setting_value
from autofa.llm.detection import LLM_CALL_RE as _LLM_SOURCE_RE
from autofa.mcp.runtime import Runtime, err, get_runtime, ok
from autofa.metrics.association import matches_selected_bucket, selected_bucket
from autofa.metrics.detector_checkpoints import DETECTOR_EXECUTION_VERSION
from autofa.metrics.discovery_feedback import build_discovery_feedback
from autofa.metrics.search_ranking import search_score
from autofa.metrics.verdicts import behavior_validation, verdict_for_scores
from autofa.objectives import discovery_strategy
from autofa.session.discovery_stall import checkpoint_discovery_stall
from autofa.session.discovery_target import agent_target_grade_for_runtime
from autofa.util.cancellation import cancellation_guard, check_cancelled


def _current_scores(rt: Runtime, fn: Any) -> dict[str, Any]:
    """Resolve display quality under its saved strategy and the active Judge Policy."""
    from autofa.metrics.task_support import build_task_support, task_support_assessment
    from autofa.metrics.verdicts import current_behavior_scores

    scores = current_behavior_scores(
        {**(fn.scores or {}), **build_task_support(fn, review_history=rt.library.review_history)},
        source=review_source(fn), description=fn.description or "",
        mechanism=fn.mechanism or fn.description or fn.name,
        behavior_strictness=strictness_for_config(rt.config),
    )
    support = task_support_assessment(scores)
    # This score count is paired with the frozen task_recall cohort. The quality
    # classifier derives broader known-match support directly from its receipt.
    support.pop("n_matched_failed_tasks")
    return {**scores, **support}


def _current_verdict(rt: Runtime, fn: Any) -> dict[str, str]:
    return verdict_for_scores(_current_scores(rt, fn))


def _current_discovery_grade(rt: Runtime, fn: Any) -> dict[str, str]:
    from autofa.metrics.discovery_grades import grade_for_scores

    return grade_for_scores(_current_scores(rt, fn))


def judge_evidence_sample_size(n_bucket: int, *, evidence_n: int | None = None) -> int:
    """How many fire-bucket items to send to the behavior judge.

    Default: ``max(3, ceil(10% of bucket))``, capped by bucket size.
    If ``evidence_n`` is set, it raises that floor (still capped by bucket size).
    """
    if n_bucket <= 0:
        return 0
    auto = max(3, (n_bucket + 9) // 10)
    floor = max(auto, int(evidence_n)) if evidence_n is not None else auto
    return min(n_bucket, floor)


def judge_reservoir_cap(scan_limit: int, *, evidence_n: int | None = None) -> int:
    """Reservoir size that can never clip ``judge_evidence_sample_size``.

    The judge sample is at most ``max(3, ceil(10% of fires), evidence_n)`` and the
    fire count cannot exceed ``scan_limit``, so bounding on ``scan_limit`` keeps the
    streamed scan off the full eval slice without changing the sample.
    """
    return max(
        int(evidence_n or 0), judge_evidence_sample_size(max(1, int(scan_limit)))
    )


def sample_fire_bucket_evidence(
    pool: list[Any],
    *,
    n: int,
    seed: int,
) -> list[Any]:
    """Uniform random sample from the fire bucket (deterministic given ``seed``)."""
    if n <= 0 or not pool:
        return []
    if n >= len(pool):
        return list(pool)
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    return rng.sample(list(pool), n)


def _attach_repair_coverage(rt: Runtime, fn: Any) -> dict[str, Any] | None:
    """Expose the repair's costs alongside evaluation and judge feedback."""
    from autofa.metrics.repair_tradeoff import compare_repair_coverage
    from autofa.session.repair_workflow import get_repair

    plan = (get_repair(rt.run.run_dir(), fn.name) or {}).get("latest") or {}
    if not plan.get("before"):
        return None
    comparison = compare_repair_coverage(
        plan["before"], fn.to_dict(), config=rt.config,
        objective_settings=plan.get("objective_settings"),
    )
    fn.scores["repair_coverage_comparison"] = comparison
    fn.scores.setdefault("feedback", {})["repair_coverage_comparison"] = comparison
    return comparison


def smoke_test_feature(source: str | None = None, n: int = 30, seed: int = 0,
                       depends_on: list[str] | None = None,
                       uses_helpers: dict[str, str] | None = None,
                       item_ids: list[str] | None = None,
                       expected_outputs: dict[str, bool | int | float | str | None] | None = None,
                       expected_abstentions: list[str] | None = None,
                       name: str | None = None,
                       expected_version_hash: str | None = None) -> dict[str, Any]:
    """Run a detector on random rows or explicit full-trace item_ids.

    Explicit controls may specify expected_outputs={id: true/false/scalar} and
    expected_abstentions=[id]. Abstention means library.abstain(), not a label
    named "abstain" or a None result. Explicit IDs override n/seed; every ID is
    resolved before execution. Results and checks_passed diagnose the author's
    controls only, never independently validate a causal claim or update scores.
    A smoke panel is limited to 30 LLM rows or 500 Python rows; use submit_function
    and capped evaluate_functions for larger experiments. Completed item results
    are reused after interruption only when source, dependencies, model settings
    and the full source item are unchanged.

    Supply source OR name. A name checks the exact saved detector, including its
    frozen helpers and dependencies; expected_version_hash rejects intervening
    edits. Use this after restoration to check historical counterexamples against
    the selected implementation. Firing still requires inspecting whether the
    old counterexample contradicts this claim; no verdict transfers automatically.
    """
    return _smoke_test_feature(
        get_runtime(), source=source, n=n, seed=seed, depends_on=depends_on,
        uses_helpers=uses_helpers, item_ids=item_ids, expected_outputs=expected_outputs,
        expected_abstentions=expected_abstentions, name=name,
        expected_version_hash=expected_version_hash,
    )


def _smoke_test_feature(  # noqa: C901
    rt: Runtime, *, source: str | None = None, n: int = 30, seed: int = 0,
    depends_on: list[str] | None = None, uses_helpers: dict[str, str] | None = None,
    item_ids: list[str] | None = None,
    expected_outputs: dict[str, bool | int | float | str | None] | None = None,
    expected_abstentions: list[str] | None = None, name: str | None = None,
    expected_version_hash: str | None = None, client: Any = None,
    frozen_items: list[dict[str, Any]] | None = None,
    reserve_model_call: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute native diagnostics against an explicit runtime and optional client.

    Classifier comparisons use an isolated client without replacing the active
    run's adapter. Scoring, dependency contracts and diagnostic accounting match
    the ordinary public tool.
    """
    from autofa.core.detector_lint import evidence_budget_warnings, lint_detector_source

    expected = expected_outputs if expected_outputs is not None else {}
    abstain_ids = expected_abstentions if expected_abstentions is not None else []
    if not isinstance(expected, dict) or any(
        not isinstance(key, str) or not key
        or (value is not None and type(value) not in (bool, int, float, str))
        or (isinstance(value, float) and not math.isfinite(value))
        for key, value in expected.items()
    ):
        return err("expected_outputs must map item IDs to finite JSON scalar values")
    if not isinstance(abstain_ids, list) or any(
        not isinstance(item_id, str) or not item_id for item_id in abstain_ids
    ):
        return err("expected_abstentions must be a list of nonempty item IDs")
    if item_ids is None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            return err("n must be a positive integer")
        if expected_outputs is not None or expected_abstentions is not None:
            return err("expectations require explicit item_ids")
    elif (not isinstance(item_ids, list) or not item_ids
          or any(not isinstance(item_id, str) or not item_id for item_id in item_ids)
          or len(set(item_ids)) != len(item_ids)):
        return err("item_ids must be a nonempty list of unique item IDs")
    expected_ids = set(expected) | set(abstain_ids)
    if expected_ids - set(item_ids or []):
        return err("expectations must refer to requested item_ids")
    if set(expected) & set(abstain_ids):
        return err("an item cannot have both an expected output and expected abstention")

    saved_function = None
    saved_version = None
    deps = None
    try:
        if name is not None:
            from autofa.core.library import FeatureFunction
            from autofa.session.repair_workflow import _version

            if source is not None or depends_on is not None or uses_helpers is not None:
                return err("name uses the saved source and dependencies; omit source, "
                           "depends_on and uses_helpers")
            if not isinstance(name, str) or not name or rt.library.get(name) is None:
                return err(f"Unknown saved detector: {name}")
            snapshot = deepcopy(rt.library.get(name).to_dict())
            saved_version = _version(snapshot)
            if expected_version_hash is not None and expected_version_hash != saved_version:
                return err("Saved detector changed; list versions before checking it")
            saved_function = FeatureFunction.from_dict(snapshot)
            source = saved_function.source
            bundle, deps = execution_bundle(saved_function), _fn_deps(saved_function)
        else:
            if expected_version_hash is not None:
                return err("expected_version_hash requires name")
            if not isinstance(source, str) or not source.strip():
                return err("Supply detector source or a saved detector name")
            bundle = (build_bundle(rt.library, depends_on=depends_on, uses_helpers=uses_helpers)
                      if depends_on or uses_helpers else None)
            validate_edges(source, (bundle or {}).get("helpers", {}),
                           (bundle or {}).get("detectors", {}))
        compile_detector(source, bundle=bundle, policy=get_field_policy(rt.config))
    except (ValueError, SyntaxError, TypeError) as exc:
        return err(str(exc))
    sources = [source, *(deps or {}).values(),
               *(node["source"] for node in (bundle or {}).get("nodes", {}).values())]
    evidence_warnings = list(dict.fromkeys(
        warning for candidate_source in sources
        for warning in evidence_budget_warnings(candidate_source)
    ))
    from autofa.session.smoke_work import (
        LLM_ROWS,
        PYTHON_ROWS,
        SmokeBudgetExceeded,
        SmokeWork,
        diagnostic_budget,
    )

    uses_llm = any(_LLM_SOURCE_RE.search(s) for s in sources)
    requested = len(item_ids) if item_ids is not None else n
    limit = LLM_ROWS if uses_llm else PYTHON_ROWS
    if requested > limit:
        return err(
            f"Smoke diagnostics allow at most {limit} "
            f"{'LLM' if uses_llm else 'Python'} rows; requested {requested}. "
            "Use submit_function and evaluate_functions(mode='capped') for larger experiments.",
            diagnostic_limit=limit, requested_n=requested,
        )
    ok_lint, lint_reason = lint_detector_source("\n".join(sources))
    if not ok_lint:
        return err(
            lint_reason or "detector rejected as structural proxy",
            lint_ok=False,
        )

    if item_ids is not None:
        rows = (frozen_items if frozen_items is not None else
                _fetch_items_by_id(rt.dataset, set(item_ids), max_items=None))
        by_id = {str(row.get("_item_id")): row for row in rows}
        missing = sorted(set(item_ids) - by_id.keys())
        if missing:
            return err("missing dataset items", missing_item_ids=missing)
        items = [by_id[item_id] for item_id in item_ids]
    else:
        from autofa.data.parquet_store import sample_items

        sample = sample_items(
            rt.dataset,
            n_fail=max(1, n // 2),
            n_pass=max(1, n // 2),
            prefer_informative=True,
            seed=seed,
            # Sampling selects rows; display previews are not detector evidence.
            max_chars=None,
            max_total=n,
        )
        items = sample["items"]
    if client is None:
        if uses_llm and (unavailable := _llm_backend_error()):
            return unavailable
        client = _llm_adapter()
    elif uses_llm and getattr(client, "available", True) is False:
        return err("Diagnostic classifier is unavailable", code="llm_backend_unavailable")
    from autofa.data.parquet_store import MAX_BATCH_BYTES, item_memory_bytes
    from autofa.session.status_log import emit_runtime_status
    from autofa.util.model_budget import model_call_budget
    from autofa.util.work_queue import ordered_bounded_map

    policy = get_field_policy(rt.config)
    work = SmokeWork(rt, source=source, bundle=bundle, policy=policy,
                     client=client, uses_llm=uses_llm, deps=deps)
    outputs: list[Any] = []
    results = []
    errors = abstentions = mismatches = 0
    def execute(it):
        check_cancelled()
        item_id = str(it.get("_item_id"))
        with work.item(it) as (cached, complete):
            if cached is not None:
                return dict(cached)
            artifact_dependencies = []
            try:
                value = run_detector(
                    source, it, llm=client, bundle=bundle, policy=policy,
                    artifact_dependencies=artifact_dependencies,
                )
                result = {"item_id": item_id, "status": "value", "value": value}
            except DetectorAbstained as exc:
                result = {"item_id": item_id, "status": "abstained", "reason": str(exc)}
            except Exception as exc:  # noqa: BLE001
                from autofa.session.detector_service import raise_if_budget_exhausted
                from autofa.session.recovery import RunInterruptedError
                from autofa.util.transient import is_transient_error

                raise_if_budget_exhausted(exc)
                if not isinstance(exc, RegexTimeoutError) and is_transient_error(exc):
                    raise RunInterruptedError(
                        "Diagnostic model connection interrupted. Completed item results "
                        "are saved; retry the same smoke call after reconnecting."
                    ) from exc
                result = {"item_id": item_id, "status": "error",
                          "error": f"{type(exc).__name__}: {exc}"}
            complete(result, artifact_dependencies)
            return result

    emit_runtime_status(f"Smoke diagnostics: 0/{len(items)} completed.",
                        source="smoke_test_feature", context={"total": len(items)})
    from autofa.util.capacity import effective_llm_workers

    workers = min(4, effective_llm_workers(rt.config)) if uses_llm else 1
    def reserve_dispatch():
        if reserve_model_call is not None:
            reserve_model_call()
        work.reserve_model_call()

    try:
        with model_call_budget(reserve_dispatch):
            for result in ordered_bounded_map(
                execute, items, workers=workers, max_pending=workers,
                max_bytes=MAX_BATCH_BYTES, size_of=item_memory_bytes,
            ):
                results.append(result)
                emit_runtime_status(
                    f"Smoke diagnostics: {len(results)}/{len(items)} completed "
                    f"({work.reused} reused, {work.calls} model calls).",
                    source="smoke_test_feature",
                    context={"completed": len(results), "total": len(items),
                             "reused": work.reused, "model_calls": work.calls},
                )
    except SmokeBudgetExceeded as exc:
        return err(str(exc), status="diagnostic_budget_exhausted", n=len(results),
                   results=results, reused=work.reused, diagnostic_budget=diagnostic_budget(rt),
                   next_action="Check pareto_status; assess or defer the current family. "
                               "Use capped evaluation for a supported larger experiment.")
    for result in results:
        item_id = result["item_id"]
        if result["status"] == "value":
            outputs.append(result["value"])
        elif result["status"] == "abstained":
            abstentions += 1
        else:
            errors += 1
        if item_id in expected_ids:
            if item_id in abstain_ids:
                result["expected_abstention"] = True
                matches = result["status"] == "abstained"
            else:
                wanted = expected[item_id]
                result["expected_output"] = wanted
                actual = result.get("value")
                matches = (result["status"] == "value" and actual == wanted
                           and isinstance(actual, bool) == isinstance(wanted, bool))
            result["matches_expected"] = matches
            mismatches += int(not matches)
    hist = Counter(str(o) for o in outputs)
    histogram = {}
    histogram_references = []
    value_indices = {str(row["value"]): index for index, row in reversed(list(enumerate(results)))
                     if row["status"] == "value"}
    for value, count in hist.most_common(20):
        if len(value) > 2_048 and isinstance(results[value_indices[value]]["value"], str):
            # Full diagnostic strings already live in results. Repeating them
            # as object keys can double a whole-trace response without adding evidence.
            histogram_references.append({
                "value_ref": f"#/results/{value_indices[value]}/value", "count": count,
            })
        else:
            histogram[value] = count
    response = ok(
        n=len(items),
        errors=errors,
        abstentions=abstentions,
        histogram=histogram,
        sample_ids=[it.get("_item_id") for it in items[:10]],
        results=results,
        diagnostic_scope="author_controls" if item_ids is not None else "random_sample",
        checked=len(expected_ids),
        mismatches=mismatches,
        checks_passed=(mismatches == 0 and errors == 0) if expected_ids else None,
        note="Diagnostic execution only; author expectations are not independent causal "
             "validation. Fix mismatches and execution errors before full submission.",
        lint_ok=True,
        reused=work.reused,
        model_calls=work.calls,
        diagnostic_budget=diagnostic_budget(rt),
        evidence_warnings=evidence_warnings,
    )
    if histogram_references:
        response["histogram_references"] = histogram_references
        response["histogram_reference_note"] = (
            "Long histogram values are stored exactly in results; value_ref is a JSON Pointer "
            "from this response's root. The referenced value's string is the histogram key."
        )
    if saved_function is not None:
        from autofa.metrics.claim_contract import claim_hash

        response.update(
            name=name, version_hash=saved_version,
            claim_hash=claim_hash(
                source=review_source(saved_function),
                description=saved_function.description or "",
                mechanism=saved_function.mechanism or saved_function.description or name,
            ),
            claim_validation="not_performed",
            next_action="For historical counterexamples that still match, inspect the "
                        "restored claim and full evidence. An old verdict may address a "
                        "different claim; firing alone does not establish that it applies. "
                        "Abstentions and errors remain unresolved. Use existing diagnostic "
                        "review when needed; these controls do not update population scores.",
        )
        for result in results:
            result["matches_saved_bucket"] = (
                matches_selected_bucket(result["value"], selected_bucket(saved_function.scores))
                if result["status"] == "value" else None
            )
    if expected_ids:
        from autofa.session.diagnostic_receipts import save_diagnostic_receipt
        from autofa.session.smoke_work import digest

        receipt_id = save_diagnostic_receipt(
            rt, source=source, execution_identity=work.identity, items=items,
            expected_outputs=expected, expected_abstentions=abstain_ids, result=response,
            artifact_reads={str(item["_item_id"]): work.artifact_read_receipts(item)
                            for item in items},
            binding=({"name": name, "version_hash": saved_version} if saved_function is not None
                     else {"depends_on": depends_on, "uses_helpers": uses_helpers,
                           "bundle_identity": digest(bundle)}),
        )
        if receipt_id:
            response["diagnostic_receipt_id"] = receipt_id
            response["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return response


def resolve_eval_limit(
    requested: int | None,
    *,
    cfg_default: int | None = None,
    cfg_max: int | None = None,
) -> int:
    """Pick a safe evaluate row cap."""
    if cfg_default is None:
        cfg_default = default_value("default_eval_limit")
    if cfg_max is None:
        cfg_max = default_value("max_eval_limit")
    if requested is None or int(requested) <= 0:
        n = int(cfg_default)
    else:
        n = int(requested)
    return max(1, min(n, int(cfg_max)))


def evaluate_functions(
    names: list[str] | None = None,
    limit: int | None = None,
    mode: str = "capped",
    chunk_size: int | None = None,
    within: str | None = None,
) -> dict[str, Any]:
    """Score library functions on the bound dataset.

    ``mode``:
      - ``capped`` (default): local detectors scan the whole matching corpus;
        detectors that can call an LLM (including helpers and parents) stay on
        ``llm_eval_sample`` within the configured evaluation frame. An explicit
        ``limit`` caps both kinds for debugging. Search comparisons retain the
        shared scoring panel independently of the full local measurements.
      - ``full``: stream the **entire** matching corpus in chunks (safe for multi-GB
        parquets; does not materialize all rows in RAM). LLM detectors are scored
        as a **census** (every row, weight 1) regardless of ``llm_eval_sample`` —
        use for an explicit full-validation request or when runtime
        ``ready_for_final_validation`` and ``final_validation_guidance`` request
        it. If guidance hands final scoring to the driver, return the summary.
        The runtime owns round-end checkpoints: a proportional 30% panel during
        planned discovery, then a full scan at the final planned/extra rounds.
        Do not duplicate those scans; saves do not score.

    ``within``: optional parent function name. When set, each target is also scored
    on the parent's fire set (``within_*`` metrics) — used for explain-WHY detectors.
    """
    from autofa.session.polishing_guard import active_polishing, polishing_tool_guard

    rt = get_runtime()
    phase = active_polishing(rt)
    if phase is None or (mode or "capped").strip().lower() != "capped":
        return _evaluate_functions(names, limit, mode, chunk_size, within)
    if names is not None and (not isinstance(names, list)
                              or any(not isinstance(name, str) or not name for name in names)):
        return err("names must be a list of detector names")
    requested = set(names or [fn.name for fn in rt.library.functions])
    unknown = requested - {fn.name for fn in rt.library.functions}
    if unknown:
        return err(f"Unknown functions: {', '.join(sorted(unknown))}")
    if limit is not None and resolve_eval_limit(
        limit, cfg_default=setting_value(rt.config, "default_eval_limit"),
        cfg_max=setting_value(rt.config, "max_eval_limit"),
    ) != resolve_eval_limit(
        rt.run.eval_limit, cfg_default=setting_value(rt.config, "default_eval_limit"),
        cfg_max=setting_value(rt.config, "max_eval_limit"),
    ):
        return err("Polishing keeps the original fixed scoring panel. Omit limit rather than "
                   "selecting a different population cohort for the comparison.")
    if within and any(
        within != (getattr(fn, "explains", None) or (fn.meta or {}).get("explains"))
        for fn in rt.library.functions if fn.name in requested
    ):
        return err("Polishing keeps the detector's existing parent scope. Omit within or use "
                   "its recorded explains parent rather than changing the comparison scope.")
    if rejected := polishing_tool_guard(rt, "evaluate_functions", {"names": sorted(requested)}):
        return rejected

    from autofa.session.candidate_screen import ensure_candidate_screen
    from autofa.session.operation_admission import admit_operation
    from autofa.session.polishing_evaluation import (
        current_polishing_evaluation,
        run_polishing_evaluation,
    )
    from autofa.session.polishing_feedback import (
        compact_polishing_feedback,
        polishing_claim_identity,
    )

    # Hold revisions stable across diagnostics and population scoring. Costly
    # diagnostics seed the prediction cache; cheap diagnostics are collected
    # directly from their ordinary full scan without executing the cases twice.
    with admit_operation("evaluate_functions", {"names": sorted(requested),
                                                "within": within}) as busy:
        if busy:
            return busy
        diagnostics = {}
        census_candidates = {}
        for fn in rt.library.functions:
            if fn.name not in requested:
                continue
            try:
                costly = evaluation_uses_llm(fn, within=within, runtime=rt)
            except ValueError as exc:
                return err(str(exc))
            controls = (phase.get("controls") or {}).get(fn.name) or {}
            screen = ensure_candidate_screen(rt, fn)
            if not screen.get("evaluation_allowed"):
                diagnostic = {"ok": False, "complete": False,
                              "error": "Evidence prerequisite checks must clear this revision "
                                       "before polishing diagnostics can run."}
            elif not controls.get("available"):
                diagnostic = {"ok": False, "complete": False,
                              "error": controls.get("reason") or
                              "No verified full-validation repair set is available."}
            elif not costly:
                census_candidates[fn.name] = (fn, controls)
                diagnostic = None
            else:
                diagnostic = current_polishing_evaluation(rt, fn, controls)
                if not (diagnostic and diagnostic.get("complete")
                        and not diagnostic.get("errors")
                        and not diagnostic.get("missing_item_ids")):
                    try:
                        diagnostic = run_polishing_evaluation(rt, fn, controls)
                    except (OSError, ValueError, TypeError) as exc:
                        diagnostic = {"ok": False, "complete": False,
                                      "error": f"Polishing checks could not finish: {exc}"}
            diagnostics[fn.name] = {"controls": controls, "diagnostic": diagnostic}
        if census_candidates:
            from autofa.session.polishing_census import polishing_census_diagnostics

            with polishing_census_diagnostics(rt, census_candidates) as collector:
                result = _evaluate_functions(
                    names, limit, mode, chunk_size, within,
                    _result_observer=collector.observe,
                )
                for name, diagnostic in collector.finish(result).items():
                    diagnostics[name]["diagnostic"] = diagnostic
        else:
            result = _evaluate_functions(names, limit, mode, chunk_size, within)
        result["polishing_feedback"] = {
            fn.name: compact_polishing_feedback(
                fn.name, diagnostics[fn.name]["controls"], diagnostics[fn.name]["diagnostic"],
                baseline_grades=phase.get("baseline_grades"), current_scores=fn.scores,
                current_claim=polishing_claim_identity(fn),
            ) for fn in rt.library.functions if fn.name in diagnostics
        }
        return result


_CURRENT_ROUTING = object()


def _evaluation_context(rt: Any, *, routing_identity: Any = _CURRENT_ROUTING) -> str | None:
    """Identify reusable dataset/endpoint settings without reading credentials."""
    from pathlib import Path

    from autofa.data.source_export import require_current_source_cache
    from autofa.llm.capabilities import TRANSPORT_CONTRACT_VERSION
    from autofa.llm.endpoints import endpoint_for_config
    from autofa.session.smoke_work import dataset_benchmark_slot

    files = []
    for raw in getattr(rt.dataset, "parquet_paths", []):
        path = Path(raw)
        require_current_source_cache(path)
        stat = path.stat()
        files.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    source_identity = files
    if not source_identity:
        # Non-file sources must supply a current content identity. A model name
        # and filter alone cannot prove that retained rows have not changed.
        fingerprint = getattr(rt.dataset, "evaluation_fingerprint", None)
        source_identity = fingerprint() if callable(fingerprint) else None
        if not isinstance(source_identity, str) or not source_identity:
            return None
    endpoint = endpoint_for_config(rt.config)
    policy = get_field_policy(rt.config)
    # Dataclass repr embeds frozenset iteration order, which changes between
    # Python processes. Context identity must survive restarting a saved run.
    policy_identity = {
        "mode": policy.mode, "allowed": sorted(policy.allowed),
        "banned": sorted(policy.banned), "allowlist": policy.allowlist,
    }
    routing = (_routing_identity(_llm_adapter())
               if routing_identity is _CURRENT_ROUTING else routing_identity)
    identity = [source_identity, getattr(rt.dataset, "model_id", None),
         getattr(rt.dataset, "benchmarks", None), dataset_benchmark_slot(rt.dataset),
         vars(endpoint), TRANSPORT_CONTRACT_VERSION, routing,
         policy_identity, _evaluation_seed(rt.run),
         _llm_eval_settings(rt.config)[:2], rt.run.eval_limit]
    return hashlib.sha256(json.dumps(
        identity, sort_keys=True, default=str,
    ).encode()).hexdigest()


def _routing_identity(adapter: Any) -> dict | None:
    # Some offline adapters deliberately reject arbitrary attribute access.
    # Only a declared method participates; inspecting this never probes a server.
    method = getattr(type(adapter), "routing_identity", None)
    return method(adapter) if callable(method) else None


def _evaluation_parent_identity(rt: Any, fn: Any, within: str | None = None) -> str | None:
    """Bind a scoped scan to the effective pinned or live parent execution graph."""
    name = within or getattr(fn, "explains", None) or (fn.meta or {}).get("explains")
    if not name:
        return None
    parent = _within_parent(rt, [fn], str(name))
    return hashlib.sha256(json.dumps(
        [name, review_source(parent), parent.kind], sort_keys=True,
    ).encode()).hexdigest()


def _reusable_evaluation(
    rt: Any, fn: Any, *, fraction: float | None = None, preserve_errors: bool = False,
    within: str | None = None,
) -> bool:
    from autofa.metrics.claim_contract import claim_hash

    scores = fn.scores or {}
    parent_name = within or getattr(fn, "explains", None) or (fn.meta or {}).get("explains") or None
    claim = claim_hash(source=review_source(fn), description=fn.description or "",
                       mechanism=fn.mechanism or fn.description or fn.name)
    if (scores.get("evaluated") is not True or scores.get("evaluation_stale")
            or scores.get("evaluation_scope_complete") is not True
            or scores.get("evaluation_run_id") != rt.run.run_id
            or scores.get("evaluation_claim_hash") != claim
            or (not preserve_errors and (scores.get("errors") or scores.get("parent_errors")))
            or scores.get("within_parent") != parent_name):
        return False
    try:
        if scores.get("evaluation_parent_identity") != _evaluation_parent_identity(rt, fn, within):
            return False
        if (evaluation_uses_llm(fn, within=within, runtime=rt)
                and scores.get("evaluation_execution_version") != DETECTOR_EXECUTION_VERSION):
            return False
        context = _evaluation_context(rt)
        if context is None or scores.get("evaluation_context") != context:
            return False
    except (OSError, ValueError, TypeError):
        return False
    if scores.get("sampled") is False:
        return True
    review = scores.get("round_validation") or {}
    return (review.get("version") == 1
            and review.get("complete") is True
            and isinstance(review.get("fraction"), (int, float))
            and review["fraction"] >= (fraction or 0)
            and bool(review.get("panel_fingerprint")))


def _retained_evaluation_result(rt: Any, fn: Any) -> dict[str, Any]:
    """Return authoritative larger-scope scores without rerunning the small panel."""
    scores = _current_scores(rt, fn)
    result = {
        "name": fn.name, "reused": True, "scores": scores.get("feedback", {}),
        "verdict": _current_verdict(rt, fn),
        "discovery_grade": _current_discovery_grade(rt, fn),
        "search_score": search_score(fn, rt.config, run_id=rt.run.run_id),
        "discovery_feedback": build_discovery_feedback(
            scores, behavior_strictness=strictness_for_config(rt.config),
            target_grade=agent_target_grade_for_runtime(rt)),
        "sampled": scores.get("sampled", False),
        "example_fail_ids": list(fn.example_fail_ids or []),
    }
    if scores.get("errors") or scores.get("parent_errors"):
        result["note"] = (
            "The larger completed scan retains unresolved execution errors. Repair the "
            "implementation or use an explicit round/full validation retry; a capped "
            "request does not erase larger-scope errors or start a hidden full scan."
        )
    return result


def evaluate_round_functions(
    names: list[str] | None = None, *, fraction: float = 0.3,
) -> dict[str, Any]:
    """Validate retained versions on a larger representative panel at a round boundary.

    Keep the configured discovery panel and settings unchanged. Already current
    full scans or equally large checkpoints remain authoritative; completed item
    predictions are reused when a newly revised definition needs validation.
    """
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 < fraction <= 1):
        return err("Validation fraction must be greater than 0 and at most 1")
    return _evaluate_functions(names=names, _finalizing=True, _round_fraction=fraction)


def _evaluate_functions(
    names: list[str] | None = None,
    limit: int | None = None,
    mode: str = "capped",
    chunk_size: int | None = None,
    within: str | None = None,
    *,
    _finalizing: bool = False,
    _round_fraction: float | None = None,
    _result_observer: Callable | None = None,
) -> dict[str, Any]:
    rt = get_runtime()
    cfg = rt.config
    mode_l = (mode or "capped").strip().lower()
    if mode_l not in {"capped", "full", "stream"}:
        return err("mode must be 'capped' or 'full'")
    if mode_l == "stream":
        mode_l = "full"

    batch = int(chunk_size or setting_value(cfg, "eval_chunk_size") or 512)
    batch = max(1, min(batch, 4096))

    targets = rt.library.functions
    if names:
        name_set = set(names)
        targets = [f for f in targets if f.name in name_set]
    if not targets:
        return err("no functions to evaluate")

    if not within and len(targets) > 1:
        scopes: dict[str | None, list[str]] = {}
        for fn in targets:
            parent = getattr(fn, "explains", None) or (fn.meta or {}).get("explains") or None
            scopes.setdefault(parent, []).append(fn.name)
        if len(scopes) > 1 or next(iter(scopes)) is not None:
            # Each explainer keeps its actual parent even in a mixed batch.
            # Every group uses the same fixed search panel and caches.
            groups = []
            for parent, scoped_names in scopes.items():
                result = _evaluate_functions(
                    scoped_names, limit, mode, chunk_size, parent, _finalizing=_finalizing,
                    _round_fraction=_round_fraction, _result_observer=_result_observer)
                groups.append((parent, result))
            errors = [group.get("error") for _, group in groups if not group.get("ok")]
            results = [row for _, group in groups for row in group.get("results", [])]
            screens = {name: screen for _, group in groups
                       for name, screen in group.get("candidate_screens", {}).items()}
            blocked = [name for _, group in groups for name in group.get("blocked_functions", [])]
            return {
                "ok": not errors, "results": results, "eval_mode": "mixed",
                "n_functions": len(results),
                "n_items": max(group.get("n_items", 0) for _, group in groups),
                "n_batches": sum(group.get("n_batches", 0) for _, group in groups),
                "eval_truncated": any(group.get("eval_truncated", False) for _, group in groups),
                "family_discovery": groups[-1][1].get("family_discovery"),
                "warning": " ".join(group["warning"] for _, group in groups
                                    if group.get("warning")) or None,
                "evaluated_names": [name for _, group in groups
                                    for name in group.get("evaluated_names", [])],
                "reused_names": [name for _, group in groups
                                 for name in group.get("reused_names", [])],
                "evaluation_groups": [{"within": parent, **{
                    key: value for key, value in group.items()
                    if key not in {"results", "candidate_screens", "family_discovery"}}}
                                      for parent, group in groups],
                **({"candidate_screens": screens} if screens else {}),
                **({"blocked_functions": blocked, "candidate_screen_blocked": True}
                   if blocked else {}),
                **({"error": "; ".join(str(error) for error in errors), "partial": bool(results)}
                   if errors else {}),
            }

    from autofa.session.candidate_screen import ensure_candidate_screen

    # Evidence prerequisites control discovery spending. Once discovery ends, score
    # every saved definition; its measured outcomes do not establish a valid claim.
    # Explicit full scoring can measure an unresolved claim, so it skips the cheap
    # source-prerequisite reminder.
    candidate_screens = ({} if _finalizing else {
        fn.name: (ensure_candidate_screen(rt, fn, check_evidence_prerequisites=False)
                  if mode_l == "full" else ensure_candidate_screen(rt, fn))
        for fn in targets
    })
    blocked = {name: screen for name, screen in candidate_screens.items()
               if not screen.get("evaluation_allowed")}
    targets = [fn for fn in targets if fn.name not in blocked]
    if not targets:
        return err(
            "Evidence prerequisite checks have not cleared these candidates for corpus "
            "evaluation. Inspect candidate_screens and follow each next_action; provide "
            "retained support, revise the claim to what is evidenced, or defer the candidate.",
            candidate_screen_blocked=True, candidate_screens=candidate_screens,
            blocked_functions=list(blocked), evaluated_names=[],
        )

    within_name = (within or "").strip() or None
    # Auto within for explainers when caller didn't pass within=
    if within_name is None and len(targets) == 1:
        explains = getattr(targets[0], "explains", None) or (targets[0].meta or {}).get(
            "explains"
        )
        if explains:
            within_name = str(explains)
    if within_name:
        try:
            _within_parent(rt, targets, within_name)
        except ValueError as exc:
            return err(str(exc))

    retained = []
    if mode_l == "capped" and limit is None:
        retained = [fn for fn in targets if _reusable_evaluation(
            rt, fn, fraction=_round_fraction, preserve_errors=_round_fraction is None,
            within=within_name)]
        targets = [fn for fn in targets if fn not in retained]
        if not targets:
            result = ok(
                results=[_retained_evaluation_result(rt, fn) for fn in retained],
                evaluated_names=[], reused_names=[fn.name for fn in retained],
                n_functions=len(retained),
                note="Current larger-scope scores retained; discovery settings are unchanged.",
            )
            if candidate_screens:
                result["candidate_screens"] = candidate_screens
            if blocked:
                result.update(
                    ok=False, candidate_screen_blocked=True, blocked_functions=list(blocked),
                    error="Some candidates await evidence prerequisite checks; current scores "
                          "were retained for the eligible functions. Inspect candidate_screens.",
                )
            return result

    from autofa.session.status_log import emit_runtime_status
    from autofa.ui.activity import detector_activity_context

    for fn in targets:
        emit_runtime_status(
            f"Evaluating {fn.name}.", source="evaluate_functions",
            context=detector_activity_context(
                fn.name, "evaluate", scores=_current_scores(rt, fn),
                target_grade=agent_target_grade_for_runtime(rt)))
    try:
        if _round_fraction is not None:
            result = _evaluate_streaming(
                targets, chunk_size=batch, within=within_name, round_fraction=_round_fraction,
                result_observer=_result_observer,
            )
        elif mode_l == "full":
            result = _evaluate_streaming(
                targets, chunk_size=batch, within=within_name, llm_census=True,
                result_observer=_result_observer,
            )
        else:
            raw = limit if limit is not None else rt.run.eval_limit
            capped = resolve_eval_limit(
                raw,
                cfg_default=setting_value(cfg, "default_eval_limit"),
                cfg_max=setting_value(cfg, "max_eval_limit"),
            )
            result = _evaluate_streaming(
                targets, chunk_size=batch,
                max_items=(capped if limit is not None or all(
                    evaluation_uses_llm(fn, within=within_name, runtime=rt) for fn in targets)
                    else None),
                within=within_name, llm_capped=True, result_observer=_result_observer,
            )
    finally:
        for fn in targets:
            emit_runtime_status(
                f"Evaluation finished for {fn.name}.", source="evaluate_functions",
                context=detector_activity_context(
                    fn.name, "evaluate", scores=_current_scores(rt, fn), finished=True,
                    target_grade=agent_target_grade_for_runtime(rt)))
    result["evaluated_names"] = [fn.name for fn in targets] if result.get("ok") else []
    result["reused_names"] = [fn.name for fn in retained]
    if retained:
        result.setdefault("results", []).extend(
            _retained_evaluation_result(rt, fn) for fn in retained)
        result["n_functions"] = len(result["results"])
    if any(screen.get("status") != "not_required" or screen.get("source_checks")
           for screen in candidate_screens.values()):
        result["candidate_screens"] = candidate_screens
    if blocked:
        # A deferred candidate must not prevent scoring eligible keepers, nor may
        # a partial census be reported as evaluation of every requested function.
        result.update(
            ok=False, partial=bool(result["evaluated_names"]),
            error="Some candidates await evidence prerequisite checks; eligible "
                  "functions were evaluated. Inspect candidate_screens before retrying.",
            candidate_screen_blocked=True, blocked_functions=list(blocked),
        )
    return result


def final_evaluate_functions(names: list[str] | None = None) -> dict[str, Any]:
    """Finish selected corpus scans once, reusing current complete evaluations.

    A completed scan can still contain abstentions or processing errors. Preserve
    those as unresolved evidence; finalization never converts them into negatives
    or upgrades a causal review. Ordinary evaluation checkpoints remain reusable.
    Automatic finalization supplies its frozen OK-or-better selection. An explicit
    caller may request other detectors; an empty selection never scans the library.
    """
    from autofa.metrics.claim_contract import claim_hash

    rt = get_runtime()
    targets = rt.library.functions
    if names is not None:
        unknown = set(names) - {fn.name for fn in targets}
        if unknown:
            return err(f"Unknown functions: {', '.join(sorted(unknown))}")
        targets = [fn for fn in targets if fn.name in names]
    pending_names = []
    try:
        context = _evaluation_context(rt)
    except (OSError, ValueError, TypeError):
        context = None
    for fn in targets:
        scores = fn.scores or {}
        fingerprint = claim_hash(
            source=review_source(fn), description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
        )
        current_census = (
            scores.get("evaluated") is True
            and scores.get("sampled") is False
            and scores.get("evaluation_scope_complete") is True
            and scores.get("evaluation_run_id") == rt.run.run_id
            and scores.get("evaluation_claim_hash") == fingerprint
            and context is not None and scores.get("evaluation_context") == context
            and not scores.get("evaluation_stale")
            and (not is_llm_detector(fn)
                 or scores.get("evaluation_execution_version") == DETECTOR_EXECUTION_VERSION)
        )
        if not current_census:
            pending_names.append(fn.name)
    if not pending_names:
        return ok(censused=[], n=0,
                  note="No selected detectors require another full corpus scan.")
    out = _evaluate_functions(names=pending_names, mode="full", _finalizing=True)
    censused = out.get("evaluated_names", [])
    if not out.get("ok"):
        return {**out, "censused": censused, "n": len(censused)}
    return ok(
        censused=censused, n=len(censused), eval=out,
        note="Final corpus scoring attempted; unresolved predictions and claim reviews "
             "retain their separate status.",
    )


def _fn_deps(fn: Any) -> dict[str, str] | None:
    deps = (getattr(fn, "meta", None) or {}).get("deps")
    if isinstance(deps, dict) and deps:
        return {str(k): str(v) for k, v in deps.items() if isinstance(v, str)}
    return None


def is_llm_detector(fn: Any) -> bool:
    """Whether scoring this detector or its dependencies can call an LLM.

    Trusts the declared ``kind`` but also sniffs the source, because ``kind``
    defaults to ``"regex"`` and an agent can submit an LLM detector without
    setting it — in which case a full-corpus eval would quietly issue one HTTP
    request per row.
    """
    if str(getattr(fn, "kind", "") or "").strip().lower() == "llm":
        return True
    sources = [getattr(fn, "source", "") or "", *(_fn_deps(fn) or {}).values()]
    nodes = (execution_bundle(fn) or {}).get("nodes", {}).values()
    if any(str(node.get("kind", "")).lower() == "llm" for node in nodes):
        return True
    sources.extend(node["source"] for node in nodes)
    return any(_LLM_SOURCE_RE.search(source) for source in sources)


def _within_parent(rt: Any, targets: list[Any], within: str | None) -> Any:
    """Resolve the same pinned parent for cost classification and execution."""
    if not within:
        return None
    snapshots = [(execution_bundle(fn) or {}) for fn in targets]
    refs = {snap.get("detectors", {}).get(within) for snap in snapshots}
    refs.discard(None)
    if len(refs) > 1:
        raise ValueError("within parents have different pinned revisions; "
                         "evaluate targets separately")
    parent = rt.library.get(within)
    if refs:
        from autofa.core.library import FeatureFunction

        ref = next(iter(refs))
        snapshot = next(s for s in snapshots if ref in s.get("nodes", {}))
        node = snapshot["nodes"][ref]
        parent = FeatureFunction(
            name=within, source=node["source"], description="", kind=node.get("kind", "regex"),
            meta={"dependency_bundle": node_bundle(snapshot, ref)},
        )
    if parent is None:
        raise ValueError(f"within parent not found: {within}")
    return parent


def evaluation_uses_llm(
    fn: Any, *, within: str | None = None, runtime: Any = None,
) -> bool:
    """Include executed dependencies and the effective parent, not just the kind label."""
    if is_llm_detector(fn):
        return True
    parent_name = within or getattr(fn, "explains", None) or (fn.meta or {}).get("explains")
    if not parent_name:
        return False
    parent = _within_parent(runtime or get_runtime(), [fn], str(parent_name))
    return is_llm_detector(parent)


def _llm_backend_error() -> dict[str, Any] | None:
    if getattr(_llm_adapter(), "available", True) is not False:
        return None
    from autofa.llm.endpoints import endpoint_setup_help

    return err(
        "LLM DETECTORS NOT CONFIGURED: evaluation was not run. "
        f"{endpoint_setup_help(get_runtime().config)} "
        "missing LLM access is not a negative detector result.",
        code="llm_backend_unavailable",
    )


def _llm_eval_settings(cfg: Any) -> tuple[int, int, int]:
    """(sample_size, min_per_stratum, workers) clamped to sane bounds."""
    sample = int(setting_value(cfg, "llm_eval_sample") or 0)
    sample = max(0, min(sample, 20_000))
    min_stratum = max(1, int(setting_value(cfg, "llm_eval_min_per_stratum") or 1))
    from autofa.util.capacity import effective_llm_workers

    workers = effective_llm_workers(cfg)
    return sample, min_stratum, workers


def _evaluation_seed(run: Any) -> int:
    """Pin scoring to the main seed while discovery advances its round sequence."""
    from autofa.session.exploration import resolve_exploration_seed

    return resolve_exploration_seed(run_id=run.run_id, override=int(run.exploration_base_seed))


def _detach_detector_error(exc: Exception) -> Exception:
    """Keep an execution outcome without retaining its stack or nested failures.

    Sampled scoring holds these outcomes until all sample weights are known.
    A traceback can otherwise pin every row in a finished scoring batch, along
    with worker futures, detector locals, and nested exception tracebacks.
    Transient errors are raised before this cleanup so interruption diagnostics
    retain their original cause.
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
    return exc


def _score_items_concurrently(
    pairs: list[tuple[Any, dict[str, Any]]],
    *,
    workers: int,
    llm: Any,
    localisations: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[Any, dict[str, Any], Any, BaseException | None]]:
    """Materializing adapter for callers that already own a small set of rows."""
    return list(_iter_scored_items(
        pairs, workers=workers, llm=llm, localisations=localisations,
    ))


def _iter_scored_items(
    pairs: Iterable[tuple[Any, dict[str, Any]]],
    *,
    workers: int,
    llm: Any,
    localisations: dict[str, dict[str, Any]] | None = None,
    executor: Executor | None = None,
) -> Iterator[tuple[Any, dict[str, Any], Any, BaseException | None]]:
    """Stream ordered results through one pool bounded by jobs and resident rows.

    Read batches never become execution barriers. Results and localisation writes
    stay in input order, preserving deterministic sampling and accumulation.
    """
    from autofa.data.parquet_store import item_memory_bytes
    from autofa.llm.endpoints import endpoint_for_config
    from autofa.metrics.detector_checkpoints import DetectorCheckpoints
    from autofa.session.recovery import RunInterruptedError
    from autofa.session.scoring_capacity import capacity_observer
    from autofa.util.transient import is_transient_error
    from autofa.util.work_queue import ordered_bounded_map

    cache = None
    rt = get_runtime()
    queue_bytes = int(setting_value(rt.config, "llm_eval_queue_mib")) * 1024**2
    # A configured one-worker LLM stream is still a capacity observation.
    # Regex-only streams never activate it.
    observe = None

    def observe_llm(event):
        if observe is not None:
            observe(event)

    def prepared():
        nonlocal cache, observe
        source = iter(pairs)
        try:
            for fn, item in source:
                use_cache = is_llm_detector(fn)
                if use_cache and observe is None:
                    observe = capacity_observer(rt, workers)
                if use_cache and cache is None:
                    if getattr(llm, "available", True) is False:
                        raise RuntimeError(
                            "LLM backend unavailable; detector checkpoints were not read or written"
                        )
                    endpoint = endpoint_for_config(rt.config)
                    cache = DetectorCheckpoints(
                        rt.run.run_dir(), model=endpoint.model, endpoint=endpoint.base_url,
                        api_key_env=endpoint.api_key_env, request_options=endpoint.request_options,
                        api=endpoint.api,
                        routing_identity=(lambda: _routing_identity(llm))
                        if callable(getattr(type(llm), "routing_identity", None)) else None,
                    )
                yield fn, item, use_cache
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()

    def score(pair):
        fn, item, use_cache = pair
        try:
            deps = _fn_deps(fn)
            bundle = execution_bundle(fn)
            policy = get_field_policy(get_runtime().config)
            from autofa.session.report_highlights import reuse_report_highlights

            def highlighted(record):
                return reuse_report_highlights(
                    record, fn.failure_localisations.get(str(item.get("_item_id", ""))),
                    item, fn.source, deps=deps, bundle=bundle,
                )

            key = cache.key(fn.source, deps, item, bundle=bundle) if use_cache else None
            if key is not None:
                saved = cache.get(key)
                if saved is not None:
                    from autofa.core.fields import wrap_item_ban_sentinels
                    from autofa.data.artifact_access import validate_artifact_reads
                    from autofa.data.trace_fields import derive_trace_fields

                    receipts = (saved[1] or {}).get("artifact_reads", [])
                    # Match run_detector's input exactly: retained images can
                    # move from a raw envelope to derived input_messages. The
                    # same bytes must keep the same source identity on reuse.
                    evidence_item = (wrap_item_ban_sentinels(derive_trace_fields(item), policy)
                                     if receipts else item)
                    if validate_artifact_reads(evidence_item, receipts):
                        return fn, item, saved[0], highlighted(saved[1]), None
            record = {}
            pred = run_detector(fn.source, item, llm=llm, deps=deps, bundle=bundle,
                                policy=policy, localisation=record)
            record = highlighted(record)
            if key is not None:
                # The first request can resolve an automatic transport route.
                # Save under that resolved identity, not its preflight unknown state.
                cache.put(cache.key(fn.source, deps, item, bundle=bundle), pred, record)
            return fn, item, pred, record, None
        except Exception as exc:  # noqa: BLE001
            from autofa.session.detector_service import raise_if_budget_exhausted

            raise_if_budget_exhausted(exc)
            if not isinstance(exc, RegexTimeoutError) and is_transient_error(exc):
                from autofa.session.status_log import emit_runtime_status
                from autofa.util.model_errors import model_error_diagnostic

                emit_runtime_status(
                    "Scoring transport interrupted; completed predictions remain saved.",
                    source="scoring_capacity", context={"error": model_error_diagnostic(exc)},
                )
                raise RunInterruptedError(
                    "Evaluation connection interrupted. Completed item results are saved; "
                    "retry evaluation after reconnecting."
                ) from exc
            return fn, item, None, None, _detach_detector_error(exc)

    scored = ordered_bounded_map(
        score, prepared(), workers=workers, max_pending=max(workers * 16, 16),
        max_bytes=queue_bytes, size_of=lambda pair: item_memory_bytes(pair[1]),
        key_of=lambda pair: id(pair[1]), executor=executor,
        observe=observe_llm,
    )
    with closing(scored):
        for fn, item, pred, record, error in scored:
            if localisations is not None:
                records = localisations.setdefault(fn.name, {})
                item_id = str(item.get("_item_id") or "")
                records.pop(item_id, None)
                if record and item_id:
                    records[item_id] = record
            yield fn, item, pred, error


def _apply_llm_results(
    scored: Iterable[tuple[Any, dict[str, Any], Any, BaseException | None]],
    *,
    accs: dict[str, Any],
    within_accs: dict[str, Any],
    parent_fn: Any,
    parent_fires_by_id: dict[str, bool],
    result_observer: Callable | None = None,
) -> None:
    """Fold pool results into the accumulators on the calling thread."""
    from autofa.ui.evaluation_progress import current_evaluation_progress

    progress = current_evaluation_progress()
    for fn, item, pred, exc in scored:
        if result_observer is not None:
            result_observer(fn, item, pred, exc)
        if progress is not None:
            progress.advance(fn.name)
        acc = accs[fn.name]
        item_id = str(item.get("_item_id") or "")
        ok_label = _is_correct(item)
        if exc is not None:
            if isinstance(exc, DetectorAbstained):
                acc.note_abstention(is_correct=ok_label)
                if parent_fn is not None and parent_fires_by_id.get(item_id, False):
                    within_accs[fn.name].note_abstention(is_correct=ok_label)
                continue
            acc.note_error(
                is_correct=ok_label,
                error=f"{type(exc).__name__}: {exc}", item_id=item_id,
            )
            if parent_fn is not None and parent_fires_by_id.get(item_id, False):
                within_accs[fn.name].note_error(
                    is_correct=ok_label,
                    error=f"{type(exc).__name__}: {exc}", item_id=item_id,
                )
            continue
        acc.update(pred, is_correct=ok_label, item_id=item_id)
        if parent_fn is not None and parent_fires_by_id.get(item_id, False):
            within_accs[fn.name].update(pred, is_correct=ok_label, item_id=item_id)


def _fetch_items_by_id(dataset: Any, ids: set[str], *, max_items: int | None):
    """Yield the rows for ``ids``, preferring O(1) index reads over a rescan."""
    if not ids:
        return
    remaining = set(ids)
    index = None
    try:
        index = dataset.identity_index()
    except Exception:  # noqa: BLE001
        index = None
    if index is not None and getattr(index, "by_id", None):
        locations = [index.by_id[i] for i in remaining if i in index.by_id]
        if len(locations) == len(remaining):
            # Physical order avoids reopening shards and re-decoding earlier
            # row-group prefixes. Sample membership and weights are unchanged.
            locations.sort(key=lambda loc: (str(loc.path), loc.row_group, loc.local_offset))
            with closing(dataset.iter_locations(locations)) as rows:
                yield from rows
            return
    # Fallback: one extra sequential pass. Costs I/O, never an extra API call.
    batches = iter(dataset.iter_batches(batch_size=256, max_items=max_items))
    try:
        for batch in batches:
            for item in batch:
                item_id = str(item.get("_item_id") or "")
                if item_id in remaining:
                    remaining.discard(item_id)
                    yield item
            if not remaining:
                return
    finally:
        close = getattr(batches, "close", None)
        if close is not None:
            close()


def _evaluate_streaming(  # noqa: C901
    targets: list[Any],
    *,
    chunk_size: int,
    max_items: int | None = None,
    within: str | None = None,
    llm_census: bool = False,
    round_fraction: float | None = None,
    llm_capped: bool = False,
    result_observer: Callable | None = None,
) -> dict[str, Any]:
    """Run detectors over streamed batches; collect association stats and passages.

    Regex detectors are scored on every streamed row. LLM detectors cost one API
    call per row: under ``mode=capped`` they use a stratified subsample
    (``llm_eval_sample``); under ``mode=full`` / ``llm_census=True`` they score
    every row (weight 1) so saved metrics are exact census counts.
    """
    from autofa.metrics.association import (
        StreamingBucketAccumulator,
        pick_primary_bucket,
        truthy_bucket,
    )
    from autofa.metrics.review_content import RetainedReviewContent
    from autofa.metrics.sampling import (
        ROUND_SAMPLE_VERSION,
        SCORING_SAMPLE_VERSION,
        StratifiedReservoir,
        interval_fields,
        proportional_sample,
        stratum_key,
    )
    from autofa.ui.evaluation_progress import current_evaluation_progress

    check_cancelled()
    rt = get_runtime()
    accs = {fn.name: StreamingBucketAccumulator() for fn in targets}
    localisations: dict[str, dict[str, Any]] = {fn.name: {} for fn in targets}
    retained_reviews = RetainedReviewContent(targets)
    within_accs = {fn.name: StreamingBucketAccumulator() for fn in targets} if within else {}
    try:
        parent_fn = _within_parent(rt, targets, within)
    except ValueError as exc:
        return err(str(exc))

    parent_is_llm = parent_fn is not None and is_llm_detector(parent_fn)
    llm_targets = [fn for fn in targets if parent_is_llm or is_llm_detector(fn)]
    llm_names = {fn.name for fn in llm_targets}
    serial_targets = [fn for fn in targets if fn.name not in llm_names]
    if llm_targets or (parent_fn is not None and is_llm_detector(parent_fn)):
        if unavailable := _llm_backend_error():
            return unavailable
    progress = current_evaluation_progress()
    if progress is not None:
        inventory = getattr(rt, "_family_failed_task_inventory", None) or {}
        rows = inventory.get("n_items") if inventory.get("inventory_complete") else None
        estimated = rows is None
        if rows is None:
            rows = rt.dataset.estimate_matching_rows()
        if rows is not None and max_items is not None:
            rows = min(rows, max_items)
        progress.start([fn.name for fn in targets], rows,
                       estimated=estimated or round_fraction is not None)
    sample_size, min_per_stratum, workers = _llm_eval_settings(rt.config)
    panel_size = sample_size
    panel_frame_limit = max_items if max_items is not None else resolve_eval_limit(
        rt.run.eval_limit, cfg_default=setting_value(rt.config, "default_eval_limit"),
        cfg_max=setting_value(rt.config, "max_eval_limit"),
    )
    # Full-corpus mode always censuses LLM detectors — sampling is for the
    # mid-loop capped path only.
    if llm_census or (max_items is None and round_fraction is None and not llm_capped):
        sample_size = 0
    subsampling = bool(llm_targets) and (sample_size > 0 or round_fraction is not None)
    reservoir = (
        StratifiedReservoir(
            per_stratum_cap=max(min_per_stratum, panel_size),
            seed=_evaluation_seed(rt.run),
        )
        if panel_size > 0
        else None
    )
    # Python and census evaluations retain their full scores, but project search
    # breadth onto the same cheap ID-only panel. No additional detector calls.
    stream_item_ids: list[str] | None = [] if reservoir is None else None
    round_frame: dict[str, str] | None = {} if round_fraction is not None else None
    # Parent gating for LLM targets needs the same per-item verdict in both passes.
    parent_fires_by_id: dict[str, bool] = {}

    n_seen = 0
    n_batches = 0
    n_within = 0
    n_within_panel_frame = 0
    parent_errors = 0

    scan_error = None

    def source_rows():
        nonlocal n_batches, n_seen
        batches = iter(rt.dataset.iter_batches(batch_size=chunk_size, max_items=max_items))
        try:
            for batch in batches:
                check_cancelled()
                n_batches += 1
                for item in batch:
                    check_cancelled()
                    n_seen += 1
                    retained_reviews.observe(item)
                    yield item
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()

    def rows_with_parent():
        with closing(source_rows()) as rows:
            if parent_is_llm and not subsampling:
                with closing(_iter_scored_items(
                    ((parent_fn, item) for item in rows), workers=workers, llm=_llm_adapter(),
                    executor=shared_pool,
                )) as scored:
                    for _, item, pred, error in scored:
                        yield item, pred, error
            else:
                for item in rows:
                    yield item, None, None

    def scan_pairs():  # noqa: C901
        nonlocal n_within, n_within_panel_frame, parent_errors, scan_error
        with closing(rows_with_parent()) as rows:
            for row_index, (it, parent_pred, parent_error) in enumerate(rows, start=1):
                check_cancelled()
                item_id = str(it.get("_item_id") or "")
                ok_label = _is_correct(it)
                if round_frame is not None:
                    if not item_id or item_id in round_frame:
                        raise ValueError("Round validation requires unique source item IDs")
                    round_frame[item_id] = stratum_key(it, is_fail=not ok_label)
                parent_fires = True
                if parent_fn is not None and not (parent_is_llm and subsampling):
                    try:
                        if parent_is_llm:
                            if parent_error is not None:
                                # This is a stored outcome. Raising it would attach
                                # the whole streaming frame to its traceback again.
                                parent_errors += 1
                                parent_pred = False
                        else:
                            parent_pred = run_detector(
                                parent_fn.source,
                                it,
                                llm=_llm_adapter(),
                                bundle=execution_bundle(parent_fn),
                            )
                        parent_fires = truthy_bucket(parent_pred)
                    except Exception:  # noqa: BLE001
                        parent_errors += 1
                        parent_fires = False
                    if parent_fires:
                        n_within += 1
                        if row_index <= panel_frame_limit:
                            n_within_panel_frame += 1
                    if llm_targets and item_id:
                        parent_fires_by_id[item_id] = parent_fires

                if row_index <= panel_frame_limit:
                    if reservoir is not None:
                        reservoir.observe(item_id, key=stratum_key(it, is_fail=not ok_label))
                    elif stream_item_ids is not None:
                        stream_item_ids.append(item_id)

                for fn in serial_targets:
                    check_cancelled()
                    acc = accs[fn.name]
                    pred, outcome_error = None, None
                    try:
                        record: dict[str, Any] = {}
                        pred = run_detector(
                            fn.source,
                            it,
                            llm=_llm_adapter(),
                            deps=_fn_deps(fn),
                            bundle=execution_bundle(fn),
                            policy=get_field_policy(rt.config),
                            localisation=record,
                        )
                        if record and item_id:
                            localisations[fn.name][item_id] = record
                        acc.update(pred, is_correct=ok_label, item_id=item_id)
                        if parent_fn is not None and parent_fires:
                            within_accs[fn.name].update(
                                pred, is_correct=ok_label, item_id=item_id
                            )
                    except DetectorAbstained as exc:
                        outcome_error = _detach_detector_error(exc)
                        acc.note_abstention(is_correct=ok_label)
                        if parent_fn is not None and parent_fires:
                            within_accs[fn.name].note_abstention(is_correct=ok_label)
                    except TimeoutError as exc:
                        if result_observer is not None:
                            result_observer(fn, it, None, _detach_detector_error(exc))
                        scan_error = err(
                            f"detector {fn.name!r} regex timed out on item "
                            f"{item_id or 'unknown'!r}; simplify nested/unbounded "
                            "quantifiers before evaluating again"
                        )
                        if progress is not None:
                            progress.advance(fn.name)
                        return
                    except Exception as exc:  # noqa: BLE001
                        outcome_error = _detach_detector_error(exc)
                        acc.note_error(
                            is_correct=ok_label,
                            error=f"{type(exc).__name__}: {exc}", item_id=item_id,
                        )
                        if parent_fn is not None and parent_fires:
                            within_accs[fn.name].note_error(
                                is_correct=ok_label,
                                error=f"{type(exc).__name__}: {exc}", item_id=item_id,
                            )
                    if result_observer is not None:
                        result_observer(fn, it, pred, outcome_error)
                    if progress is not None:
                        progress.advance(fn.name)

                if (llm_targets and not subsampling
                        and (not llm_capped or row_index <= panel_frame_limit)):
                    for fn in llm_targets:
                        yield fn, it

    # Parent and child pipelines may overlap, but the configured detector
    # concurrency is shared across both. Each queue has its own bounded window.
    pool_context = (
        ThreadPoolExecutor(max_workers=workers)
        if parent_is_llm and llm_targets and not subsampling else nullcontext(None)
    )
    with pool_context as shared_pool:
        with closing(scan_pairs()) as pairs:
            if llm_targets and not subsampling:
                with closing(_iter_scored_items(
                    pairs, workers=workers, llm=_llm_adapter(), localisations=localisations,
                    executor=shared_pool,
                )) as scored:
                    _apply_llm_results(
                        scored, accs=accs, within_accs=within_accs, parent_fn=parent_fn,
                        parent_fires_by_id=parent_fires_by_id,
                        result_observer=result_observer,
                    )
            else:
                # Local detectors and reservoir allocation run during the scan.
                for _ in pairs:
                    pass
    if scan_error is not None:
        return scan_error

    if n_seen == 0:
        return err("no items in dataset")

    missing_reviews = retained_reviews.missing_item_ids()
    if missing_reviews:
        try:
            with closing(_fetch_items_by_id(rt.dataset, missing_reviews, max_items=None)) as rows:
                for item in rows:
                    check_cancelled()
                    retained_reviews.observe(item)
        except (OSError, ValueError):
            pass  # Unavailable source cannot certify retained reviews as current.

    sample = (reservoir.allocate(target_n=panel_size, min_per_stratum=min_per_stratum)
              if reservoir is not None else None)
    # Preserve allocation before sample.reweight accounts for missing point reads.
    # Those rows cannot silently disappear from only one detector's search cohort.
    panel_item_ids = tuple(sample.item_ids if sample is not None else stream_item_ids or [])
    search_sample = sample
    if round_frame is not None:
        sample = proportional_sample(
            round_frame, fraction=round_fraction, seed=_evaluation_seed(rt.run),
            minimum_n=len(panel_item_ids),
        )
    round_panel_ids = tuple(sample.item_ids) if round_frame is not None else ()
    search_accs = {fn.name: StreamingBucketAccumulator() for fn in llm_targets}
    search_localisations: dict[str, dict] = {}
    attempted_ids: list[str] = []
    sample_meta: dict[str, Any] | None = None
    if sample is not None and subsampling:
        wanted = set(sample.item_ids)
        requested = wanted | set(panel_item_ids) if round_frame is not None else wanted
        if progress is not None:
            progress.set_targets({
                **{fn.name: n_seen for fn in serial_targets},
                **{fn.name: len(requested) for fn in llm_targets},
            })
        # Score in bounded chunks (rows are large) but hold only the verdicts, so
        # weighting can wait until we know which rows were actually reachable.
        verdicts: list[tuple[Any, str, bool, Any, BaseException | None]] = []
        scored_ids: list[str] = []
        def selected_rows():
            nonlocal parent_errors
            with closing(_fetch_items_by_id(rt.dataset, requested, max_items=max_items)) as rows:
                if not parent_is_llm:
                    yield from rows
                    return
                # A costly parent is part of the sampled operation. Calling it
                # during the reservoir pass would spend on the entire corpus.
                with closing(_iter_scored_items(
                    ((parent_fn, item) for item in rows), workers=workers, llm=_llm_adapter(),
                    executor=sample_executor,
                )) as parent_results:
                    for _, item, parent_pred, parent_error in parent_results:
                        item_id = str(item.get("_item_id") or "")
                        if parent_error is not None:
                            parent_errors += 1
                        parent_fires_by_id[item_id] = (
                            parent_error is None and truthy_bucket(parent_pred))
                        yield item

        def selected_pairs():
            with closing(selected_rows()) as rows:
                for item in rows:
                    check_cancelled()
                    retained_reviews.observe(item)
                    item_id = str(item.get("_item_id") or "")
                    attempted_ids.append(item_id)
                    if item_id in wanted:
                        scored_ids.append(item_id)
                    for fn in llm_targets:
                        yield fn, item

        sample_pool = (ThreadPoolExecutor(max_workers=workers) if parent_is_llm
                       else nullcontext(None))
        with sample_pool as sample_executor:
            with closing(_iter_scored_items(
                selected_pairs(), workers=workers, llm=_llm_adapter(), localisations=localisations,
                executor=sample_executor,
            )) as scored:
                for fn, item, pred, exc in scored:
                    if result_observer is not None:
                        result_observer(fn, item, pred, exc)
                    if progress is not None:
                        progress.advance(fn.name)
                    verdicts.append(
                        (fn, str(item.get("_item_id") or ""), _is_correct(item), pred, exc)
                    )
        # Do not retain the final trace alongside scalar verdicts during metrics.
        item = None

        # Rows can go missing between allocation and read (dataset filters applied
        # on point-read, ids absent from the index). Rescale to what we really got.
        sample.reweight(scored_ids)
        if parent_is_llm:
            n_within = sum(sample.weight_for(item_id) for item_id in scored_ids
                           if parent_fires_by_id.get(item_id, False))
        sample_meta = {
            "sampled": not sample.is_census(),
            "sample_n": sample.sample_n,
            "requested_n": len(wanted),
            "frame_n": sample.frame_n,
            "n_strata": len(sample.strata),
            "allocation_version": (ROUND_SAMPLE_VERSION if round_frame is not None
                                   else SCORING_SAMPLE_VERSION),
            "strata": sample.strata,
            "unrepresented_strata": sample.unrepresented_strata,
            "workers": workers,
        }
        search_ids = set(panel_item_ids)
        for fn, item_id, ok_label, pred, exc in verdicts:
            if round_frame is not None and item_id in search_ids:
                search_acc = search_accs[fn.name]
                search_weight = search_sample.weight_for(item_id) if search_sample else 1.0
                if isinstance(exc, DetectorAbstained):
                    search_acc.note_abstention(is_correct=ok_label, weight=search_weight)
                elif exc is not None:
                    search_acc.note_error(is_correct=ok_label, weight=search_weight,
                                          error=f"{type(exc).__name__}: {exc}", item_id=item_id)
                else:
                    search_acc.update(pred, is_correct=ok_label, item_id=item_id,
                                      weight=search_weight)
            if item_id not in wanted:
                continue
            acc = accs[fn.name]
            w = sample.weight_for(item_id)
            if isinstance(exc, DetectorAbstained):
                acc.note_abstention(is_correct=ok_label, weight=w)
                if parent_fn is not None and parent_fires_by_id.get(item_id, False):
                    within_accs[fn.name].note_abstention(is_correct=ok_label, weight=w)
                continue
            if exc is not None:
                acc.note_error(
                    is_correct=ok_label, weight=w,
                    error=f"{type(exc).__name__}: {exc}", item_id=item_id,
                )
                if parent_fn is not None and parent_fires_by_id.get(item_id, False):
                    within_accs[fn.name].note_error(
                        is_correct=ok_label, weight=w,
                        error=f"{type(exc).__name__}: {exc}", item_id=item_id,
                    )
                continue
            acc.update(pred, is_correct=ok_label, item_id=item_id, weight=w)
            if parent_fn is not None and parent_fires_by_id.get(item_id, False):
                within_accs[fn.name].update(
                    pred, is_correct=ok_label, item_id=item_id, weight=w
                )
        if round_frame is not None:
            for fn in llm_targets:
                search_localisations[fn.name] = {
                    key: value for key, value in localisations[fn.name].items() if key in search_ids
                }
                localisations[fn.name] = {
                    key: value for key, value in localisations[fn.name].items() if key in wanted
                }

    if progress is not None:
        progress.finish_scoring()
    est = rt.dataset.estimate_matching_rows()
    scope_complete = max_items is None or (est is not None and n_seen >= est)
    results = []
    for fn in targets:
        acc = accs[fn.name]
        fn_sampled = bool(sample_meta and sample_meta["sampled"] and fn.name in llm_names)
        buckets = acc.finalize(sampled=fn_sampled or None)
        primary = pick_primary_bucket(buckets)
        if primary is not None:
            scores = {k: primary[k] for k in primary if k != "example_fail_ids"}
            scores["bucket"] = primary["bucket"]
        else:
            scores = {
                "a": 0,
                "b": 0,
                "c": float(acc.n_fail_total),
                "d": float(acc.n_pass_total),
                "recall": 0.0,
                "ppv_at_pi0": 0.0,
                "pct_caught": 0.0,
                "raw_ppv": 0.0,
                # Match the weighted contingency cells; n_scored below records
                # the actual sample size even when no positive bucket exists.
                "n_items": float(acc.n_fail_total + acc.n_pass_total),
                "n_fail_caught": 0.0,
                "n_flagged": 0.0,
                "bucket": "positive",
                "bucket_mode": "truthy",
            }
            if fn_sampled:
                # An all-negative panel still has sampling uncertainty. Negative
                # buckets retain the same resolved failure/pass effective counts.
                n_eff_fail = max((row.get("n_eff_fail", 0) for row in buckets), default=0)
                n_eff_pass = max((row.get("n_eff_pass", 0) for row in buckets), default=0)
                scores.update(n_eff_fail=n_eff_fail, n_eff_pass=n_eff_pass, n_eff_flagged=0)
                scores.update(interval_fields(
                    recall=0.0, raw_ppv=0.0, n_eff_fail=n_eff_fail, n_eff_flagged=0,
                    fpr=0.0, n_eff_pass=n_eff_pass,
                ))

        if parent_fn is not None:
            wacc = within_accs[fn.name]
            wbuckets = wacc.finalize(sampled=fn_sampled or None)
            wprimary = pick_primary_bucket(wbuckets)
            scores["within_errors"] = wacc.errors
            scores["within_n_insufficient"] = wacc.n_abstained
            scores["within_n_unresolved"] = wacc.n_abstained + wacc.errors
            if wacc.n_abstained:
                scores["within_abstention_bounds"] = wacc.abstention_bounds(wprimary or {})
            if wacc.n_abstained or wacc.errors:
                scores["within_unresolved_bounds"] = wacc.unresolved_bounds(wprimary or {})
                scores["within_assessment_coverage"] = (
                    wacc.n_items / (wacc.n_items + wacc.n_abstained + wacc.errors)
                )
            scores["within_parent"] = within
            scores["within_n"] = float(
                n_within_panel_frame if (sample_meta and fn.name in llm_names
                                        and not parent_is_llm and round_fraction is None)
                else n_within)
            scores["within_parent_sampled"] = bool(parent_is_llm and fn_sampled)
            for cell in ("a", "b", "c", "d", "n_flagged"):
                scores["within_" + cell] = (
                    float(wprimary[cell]) if wprimary is not None else
                    float(wacc.n_fail_total) if cell == "c" else
                    float(wacc.n_pass_total) if cell == "d" else 0.0
                )
            if wprimary is not None:
                scores["within_precision"] = float(
                    wprimary.get("precision") or wprimary.get("raw_ppv") or 0
                )
                scores["within_recall"] = float(wprimary.get("recall") or 0)
                scores["within_ppv_at_pi0"] = float(wprimary.get("ppv_at_pi0") or 0)
                scores["within_pct_caught"] = float(wprimary.get("pct_caught") or 0)
                # Explainers display within-parent recall, so the interval shown
                # must be the within-parent one — not the corpus-wide bucket's.
                for src, dst in (
                    ("recall_ci_low", "within_recall_ci_low"),
                    ("recall_ci_high", "within_recall_ci_high"),
                    ("fpr_ci_low", "within_fpr_ci_low"),
                    ("fpr_ci_high", "within_fpr_ci_high"),
                    ("fpr", "within_fpr"),
                    ("fisher_p", "within_fisher_p"),
                ):
                    if src in wprimary:
                        scores[dst] = wprimary[src]
                parent_fail_rate = (
                    float(wacc.n_fail_total) / float(wacc.n_items)
                    if wacc.n_items
                    else 0.0
                )
                child_fail_rate = float(scores["within_precision"])
                scores["within_lift"] = (
                    child_fail_rate / parent_fail_rate if parent_fail_rate > 0 else 0.0
                )
            else:
                scores["within_precision"] = 0.0
                scores["within_recall"] = 0.0
                scores["within_ppv_at_pi0"] = 0.0
                scores["within_pct_caught"] = 0.0
                scores["within_lift"] = 0.0

        prev = fn.scores or {}
        if "improvement" in prev:
            from autofa.metrics.improvements import resolve_improvement

            scores["improvement"] = resolve_improvement(
                prev["improvement"], source=review_source(fn), description=fn.description,
                mechanism=fn.mechanism,
            )
        # Preserve a real judge score only — never stamp placeholder 0.0 for unjudged
        # (agents were reading beh=0.00 as failed purity and silently dropping modes).
        from autofa.metrics.verdicts import (
            current_behavior_scores,
            resolved_behavior_purity,
            resolved_behavior_specificity,
        )

        # A previous scheduling decision must not erase archived judge evidence
        # or prevent a new evaluation from making this candidate eligible again.
        prev = retained_reviews.apply(fn.name, dict(prev))
        prev.pop("judge_prefilter", None)
        prev = current_behavior_scores(
            prev, source=review_source(fn), description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
            objective_strategy=discovery_strategy(rt.config),
            behavior_strictness=strictness_for_config(rt.config),
        )
        judged_bp = resolved_behavior_purity(prev)
        for key in (
            "judge_backend",
            "judge_harness",
            "judge_rationale",
            "behavior_n_judged",
            "behavior_n_yes",
            "behavior_n_no",
            "behavior_n_uncertain",
            "behavior_n_completed",
            "behavior_n_errors",
            "behavior_n_review_incomplete",
            "behavior_review_phase",
            "ordinary_confirmation_request",
            "confirmation_availability",
            "behavior_sample",
            "behavior_n_fires",
            "behavior_n_sampled",
            "behavior_judgment_stale",
            "behavior_evidence_stale",
            "behavior_verdicts",
            "behavior_verdict_history",
            "behavior_review_assessments",
            "behavior_review_operation",
            "behavior_review_localisations",
            "diagnostic_review",
            "diagnostic_evidence_stale",
            "task_support",
            "judge_evidence_version",
            "judge_contract_version",
            "judge_model",
            "judge_effort",
            "judge_thinking",
            "judge_thinking_budget_tokens",
            "judge_rubric_hash",
            "judge_policy_hash",
            "judge_objective_strategy",
            "judge_behavior_strictness",
            "claim_hash",
            "claim_contract",
            "claim_compilation",
            "claim_calibration",
            "item_calibrations",
            "calibration_hint",
            "repair_validation",
            "round_review",
            "final_review",
            "selection_review",
            "final_validation_selection",
        ):
            if key in prev:
                scores[key] = prev[key]
        if judged_bp is not None:
            scores["behavior_purity"] = judged_bp
        elif prev.get("behavior_evidence_stale") and "behavior_purity" in prev:
            scores["behavior_purity"] = prev["behavior_purity"]
        else:
            scores.pop("behavior_purity", None)

        judged_spec = resolved_behavior_specificity(prev)
        if judged_spec is not None:
            scores["behavior_specificity"] = judged_spec
        elif prev.get("behavior_evidence_stale") and "behavior_specificity" in prev:
            scores["behavior_specificity"] = prev["behavior_specificity"]
        else:
            scores.pop("behavior_specificity", None)

        scores["errors"] = acc.errors
        scores["error_examples"] = acc.error_examples
        scores["n_insufficient"] = acc.n_abstained
        scores["n_unresolved"] = acc.n_abstained + acc.errors
        if acc.n_abstained or acc.errors:
            scores["metrics_scope"] = "resolved_items"
            scores["assessment_coverage"] = (
                acc.n_items / (acc.n_items + acc.n_abstained + acc.errors)
            )
            scores["evaluation_complete"] = False
            scores["unresolved_bounds"] = acc.unresolved_bounds(scores)
        if acc.n_abstained:
            scores["abstention_bounds"] = acc.abstention_bounds(scores)
        scores["parent_errors"] = parent_errors
        scores["buckets"] = buckets
        scores["evaluated"] = True
        scores["evaluation_run_id"] = rt.run.run_id
        scores["evaluation_context"] = _evaluation_context(rt)
        scores["evaluation_parent_identity"] = _evaluation_parent_identity(rt, fn, within)
        if fn.name in llm_names:
            scores["evaluation_execution_version"] = DETECTOR_EXECUTION_VERSION
        if sample_meta and fn.name in llm_names:
            scores["evaluation_item_ids"] = sorted(scored_ids)
        else:
            scores.pop("evaluation_item_ids", None)
        scores["eval_mode"] = ("round_validation" if round_fraction is not None
                               else "capped" if llm_capped and fn.name in llm_names
                               else "full" if max_items is None else "capped")
        fn_scope_complete = scope_complete and not (
            llm_capped and fn.name in llm_names and panel_frame_limit < n_seen)
        scores["evaluation_scope_complete"] = bool(fn_scope_complete)
        scores["eval_truncated"] = not fn_scope_complete
        # Finishing the requested capped frame is sufficient to select a
        # candidate for a later census; it is not proof of corpus coverage.
        checkpoint_complete = scope_complete or (
            max_items is not None and n_seen >= max_items
        ) or (
            max_items is None and llm_capped and n_seen >= panel_frame_limit
        )
        if sample_meta and fn.name in llm_names:
            checkpoint_complete = checkpoint_complete and (
                sample_meta["sample_n"] == sample_meta["requested_n"]
            )
        scores["evaluation_checkpoint_complete"] = bool(checkpoint_complete)
        from autofa.metrics.claim_contract import claim_hash

        scores["evaluation_claim_hash"] = claim_hash(
            source=review_source(fn), description=fn.description or "",
            mechanism=fn.mechanism or fn.description or fn.name,
        )
        n_scored = int(getattr(acc, "n_items", 0) or 0)
        scores["n_scored"] = n_scored
        if fn_sampled and sample_meta:
            # These metrics are estimates from a stratified subsample; say so in the
            # scores dict itself so no render site can present them as exact.
            scores["sampled"] = True
            scores["sample_n"] = int(sample_meta["sample_n"])
            scores["sample_requested_n"] = int(sample_meta["requested_n"])
            scores["sample_frame_n"] = int(sample_meta["frame_n"])
            scores["sample_strata"] = int(sample_meta["n_strata"])
            scores["sample_reason"] = "llm_detector"
            scores["sample_allocation_version"] = sample_meta["allocation_version"]
            scores["sample_stratum_counts"] = sample_meta["strata"]
            scores["sample_unrepresented_strata"] = sample_meta["unrepresented_strata"]
            for key in (
                "recall_ci_low",
                "recall_ci_high",
                "fpr_ci_low",
                "fpr_ci_high",
                "raw_ppv_ci_low",
                "raw_ppv_ci_high",
                "fpr",
                "fisher_p",
                "n_eff_fail",
                "n_eff_flagged",
                "n_eff_pass",
            ):
                if primary is not None and key in primary and primary[key] is not None:
                    scores[key] = primary[key]
        else:
            scores["sampled"] = False
            # Census / regex path: CIs and Fisher still come from the primary bucket.
            if primary is not None:
                for key in (
                    "recall_ci_low",
                    "recall_ci_high",
                    "fpr_ci_low",
                    "fpr_ci_high",
                    "fpr",
                    "fisher_p",
                    "or_ci_low",
                    "or_ci_high",
                ):
                    if key in primary:
                        scores[key] = primary[key]
        if fn_sampled and sample_meta["unrepresented_strata"]:
            # Tiny budgets or missing point reads can omit a populated stratum.
            # The retained predictions remain useful, but cannot stand in for
            # outcomes/benchmarks that were never measured.
            scores["metrics_scope"] = "unrepresented_strata"
            scores["evaluation_complete"] = False
            scores["sampling_limitation"] = (
                "Population rates unavailable: the scoring panel has no readable rows for "
                + ", ".join(sample_meta["unrepresented_strata"])
                + ". Increase llm_eval_sample or restore missing rows before comparing rates."
            )
            for row in [scores, *buckets]:
                for key in ("recall", "sensitivity", "precision", "purity", "raw_ppv", "ppv_at_pi0",
                            "pct_caught", "fpr", "f1_at_pi0", "fisher_p", "within_recall",
                            "within_precision", "within_ppv_at_pi0", "within_pct_caught"):
                    if key in row:
                        row[key] = None
                if "is_high_purity" in row:
                    row["is_high_purity"] = False
                for metric in ("recall", "raw_ppv", "fpr"):
                    row[f"{metric}_ci_low"] = 0.0
                    row[f"{metric}_ci_high"] = 1.0
        if explains := fn.explains or (fn.meta or {}).get("explains"):
            scores["explains"] = explains
        scores["judge_prefilter"] = _prefilter_decision(fn, scores)
        judged_bp = resolved_behavior_purity(scores)
        judged_spec = resolved_behavior_specificity(scores)
        verdict = verdict_for_scores(scores)
        scores["badge"] = verdict["label"]
        scores["verdict"] = verdict
        scores["feedback"] = {
            "ppv_at_pi0": scores.get("ppv_at_pi0"),
            "recall": scores.get("recall"),
            "fpr": scores.get("fpr"),
            "behavior_purity": judged_bp,
            "behavior_specificity": judged_spec,
            "behavior_validation": behavior_validation(scores, include_details=False),
            "judge_prefilter": scores["judge_prefilter"],
            "pct_caught": scores.get("pct_caught"),
            "pi0": 0.30,
            "within_ppv_at_pi0": scores.get("within_ppv_at_pi0"),
            "within_recall": scores.get("within_recall"),
            "errors": acc.errors,
            "error_examples": acc.error_examples,
            "n_scored": n_scored,
            "sampled": scores.get("sampled", False),
            "recall_ci": [
                scores.get("recall_ci_low"),
                scores.get("recall_ci_high"),
            ],
            "fpr_ci": [
                scores.get("fpr_ci_low"),
                scores.get("fpr_ci_high"),
            ],
            "fisher_p": scores.get("fisher_p"),
        }
        scores["feedback"].update({
            key: scores[key] for key in (
                "n_insufficient", "n_unresolved", "metrics_scope", "assessment_coverage",
                "evaluation_complete", "unresolved_bounds", "abstention_bounds",
                "within_errors", "within_n_insufficient", "within_n_unresolved",
                "within_assessment_coverage", "within_unresolved_bounds",
            ) if key in scores
        })
        if scores.get("sampled"):
            scores["feedback"].update(
                {
                    "sample_n": scores.get("sample_n"),
                    "sample_frame_n": scores.get("sample_frame_n"),
                    "sample_allocation_version": scores.get("sample_allocation_version"),
                    "sample_stratum_counts": scores.get("sample_stratum_counts"),
                    "sample_unrepresented_strata": scores.get("sample_unrepresented_strata"),
                    "sampling_limitation": scores.get("sampling_limitation"),
                    "raw_ppv_ci": [
                        scores.get("raw_ppv_ci_low"),
                        scores.get("raw_ppv_ci_high"),
                    ],
                    "note": (
                        f"Estimates from a stratified sample of {scores.get('sample_n')} "
                        f"of {scores.get('sample_frame_n')} rows (LLM detector). "
                        + ("Round validation samples the configured fraction of each "
                           "benchmark/outcome stratum; " if round_fraction is not None else
                           "Failed and successful outcomes receive equal budgets when available; ")
                        +
                        "benchmark/outcome weights restore population rates. Iterative breadth "
                        "uses the same fixed panel for every detector engine. "
                        "Increase llm_eval_sample if intervals remain too loose."
                    ),
                }
            )
        example_fails = list(primary.get("example_fail_ids") or [])[:5] if primary else []
        evaluated_fn = copy(fn)
        evaluated_fn.scores = scores
        evaluated_fn.example_fail_ids = example_fails
        evaluated_fn.failure_localisations = localisations[fn.name]
        from autofa.metrics.search_panel import build_search_panel
        from autofa.metrics.task_recall import build_task_recall
        from autofa.metrics.task_support import build_task_support
        from autofa.session.family_coverage import cached_task_inventory

        # Freeze task breadth from this evaluation's matches. Later judge scans
        # can add evidence outside the scoring panel, but cannot expand its score.
        inventory = None
        try:
            inventory = cached_task_inventory(rt)
            task_scores = build_task_recall(
                evaluated_fn, inventory, run_id=rt.run.run_id,
            )
            panel_fn = evaluated_fn
            if round_frame is not None and fn.name in llm_names:
                # The larger audit can change public scores without changing the
                # cohort used to compare iterative breadth across engines.
                panel_fn = copy(evaluated_fn)
                panel_acc = search_accs[fn.name]
                panel_fn.scores = {
                    **scores, "evaluation_item_ids": list(panel_item_ids),
                    "n_scored": panel_acc.n_items, "errors": panel_acc.errors,
                    "n_insufficient": panel_acc.n_abstained,
                    "n_unresolved": panel_acc.errors + panel_acc.n_abstained,
                    "sampled": search_sample is not None and not search_sample.is_census(),
                }
                panel_fn.failure_localisations = search_localisations[fn.name]
            scores["search_panel"] = build_search_panel(
                panel_fn, inventory, item_ids=panel_item_ids, run_id=rt.run.run_id,
                attempted_item_ids=attempted_ids if sample_meta and fn.name in llm_names else None,
                allocation_version=SCORING_SAMPLE_VERSION if search_sample is not None else None,
            )
        except (OSError, ValueError, TypeError) as exc:
            task_scores = {
                "task_recall": None, "task_recall_scope": "unavailable",
                "task_recall_reason": f"Source task inventory unavailable: {exc}",
            }
            scores["search_panel"] = {
                "version": 1, "available": False, "scope": "fixed_scoring_panel",
                "task_recall": None, "reason": task_scores["task_recall_reason"],
            }
        scores.update(task_scores)
        if round_frame is not None and fn.name in llm_names:
            scores["round_validation"] = {
                "version": 1, "fraction": round_fraction,
                "effective_fraction": len(round_panel_ids) / n_seen,
                "sample_n": len(round_panel_ids), "frame_n": n_seen,
                "allocation": "proportional_benchmark_outcome",
                "panel_fingerprint": hashlib.sha256(
                    json.dumps(round_panel_ids).encode()).hexdigest(),
                "corpus_fingerprint": (inventory or {}).get("corpus_fingerprint"),
                "round": rt.run.round,
                "complete": set(round_panel_ids) == set(scored_ids),
            }
        support_scores = build_task_support(evaluated_fn, inventory)
        scores.update(support_scores)
        # Quality is finalized after the source-verified task receipt exists.
        verdict = verdict_for_scores(scores)
        scores["badge"] = verdict["label"]
        scores["verdict"] = verdict
        scores["feedback"].update(task_scores)
        scores["feedback"].update({key: value for key, value in support_scores.items()
                                   if key != "task_support"})
        scores["feedback"]["search_panel"] = scores["search_panel"]
        if "round_validation" in scores:
            scores["feedback"]["round_validation"] = scores["round_validation"]
        _attach_repair_coverage(rt, evaluated_fn)
        with cancellation_guard():
            fn.scores = scores
            fn.example_fail_ids = example_fails
            fn.failure_localisations = localisations[fn.name]
        discovery_feedback = build_discovery_feedback(
            scores, behavior_strictness=strictness_for_config(rt.config),
            target_grade=agent_target_grade_for_runtime(rt))
        results.append(
            {
                "name": fn.name,
                "scores": scores["feedback"],
                "verdict": _current_verdict(rt, fn),
                "discovery_grade": _current_discovery_grade(rt, fn),
                "search_score": search_score(fn, rt.config, run_id=rt.run.run_id),
                "discovery_feedback": discovery_feedback,
                "bucket": scores.get("bucket"),
                "buckets": [
                    {
                        "bucket": b["bucket"],
                        "n": b["n"],
                        "n_fail": b["n_fail"],
                        "purity": b["purity"],
                        "precision": b["precision"],
                        "is_high_purity": b["is_high_purity"],
                        "recall": b["recall"],
                        "ppv_at_pi0": b["ppv_at_pi0"],
                    }
                    for b in buckets[:10]
                ],
                "n_fail_caught": scores.get("n_fail_caught"),
                "n_flagged": scores.get("n_flagged"),
                "n_items": scores.get("n_items"),
                "sampled": scores.get("sampled", False),
                "example_fail_ids": example_fails,
            }
        )

    check_cancelled()
    rt.persist_library()
    from autofa.metrics.pareto import library_frontier_metrics

    # Frontier must reflect the *full* scored library (keepers + just-evaluated),
    # not only this call's targets — otherwise submit_function auto-eval spikes
    # frontier_value from a single detector and misleads discovery progress checks.
    lib_metrics = library_frontier_metrics(rt.library.functions, config=rt.config)
    points = [list(p) for p in lib_metrics["points"]]
    from autofa.metrics.completeness import incomplete_evidence

    evaluation_complete = (
        not any(incomplete_evidence(fn.scores) for fn in targets)
        and not (sample_meta and sample_meta["sampled"])
        and (max_items is None or (est is not None and n_seen >= est))
    )
    import time as _time

    from autofa.config import resolve_objective_axes
    from autofa.metrics.claim_contract import claim_hash
    from autofa.metrics.live_progress import runtime_display_progress_snapshot

    round_snap = {
        "round": rt.run.round,
        "results": results,
        "points": points,
        "evaluated_names": [r["name"] for r in results],
        "evaluated_claims": {
            fn.name: claim_hash(
                source=review_source(fn), description=fn.description or "",
                mechanism=fn.mechanism or fn.description or fn.name,
            ) for fn in targets
        },
        "objective_axes": resolve_objective_axes(rt.config),
        "eval_mode": ("round_validation" if round_fraction is not None
                      else "full" if max_items is None else "capped"),
        "n_items": n_seen,
        "n_batches": n_batches,
        "n_functions": len(rt.library.functions),
        "n_scored": int(lib_metrics["n_scored"]),
        "frontier_value": float(lib_metrics["frontier_value"]),
        "hypervolume": float(lib_metrics["hypervolume"]),
        "wall_clock": _time.time(),
        "evaluation_complete": bool(evaluation_complete),
        "dataset_evaluation_complete": bool(evaluation_complete),
        "display_progress": runtime_display_progress_snapshot(rt),
    }
    with cancellation_guard(), rt._scores_history_lock:
        rt.scores_history.append(round_snap)
        rt.persist_scores_history()
    checkpoint_discovery_stall(rt)
    warnings: list[str] = []
    if max_items is not None:
        warnings.append(
            f"Capped eval on {n_seen} rows (limit={max_items}). "
            "Call evaluate_functions(mode='full') to stream the entire corpus."
        )
    if sample_meta and sample_meta["sampled"]:
        warnings.append(
            f"{len(llm_targets)} LLM detector(s) scored on a stratified sample of "
            f"{sample_meta['sample_n']}/{sample_meta['frame_n']} rows across "
            f"{sample_meta['n_strata']} strata. "
            + ("Round validation uses proportional benchmark/outcome allocation; "
               if round_fraction is not None else "Failure/success allocation is balanced when "
               "available; ")
            + "stratum weights restore population rates. Estimates carry 95% CIs "
            "(recall_ci_low/high, raw_ppv_ci_low/high). Iterative breadth uses the same fixed "
            "scoring panel for every detector engine; full-corpus results remain separate."
        )
    from autofa.mcp.tools_library import _family_search_status

    with cancellation_guard():
        family_status = _family_search_status(rt)
    return ok(
        n_functions=len(results),
        n_items=n_seen,
        n_batches=n_batches,
        chunk_size=chunk_size,
        eval_mode=("round_validation" if round_fraction is not None
                   else "full" if max_items is None else "capped"),
        eval_limit=max_items,
        eval_truncated=bool(max_items is not None and est and est > max_items),
        estimated_corpus_rows=est,
        llm_sample=sample_meta,
        results=results,
        family_discovery=family_status,
        warning=(" ".join(warnings) if warnings else None),
    )


def _failed_repair_review(rt, plan_id: str, result: dict, error: Exception) -> dict[str, Any]:
    """Keep failed review evidence recoverable without publishing new scores."""
    import json
    from uuid import uuid4

    try:
        archive = rt.run.run_dir() / "repair_validation_errors"
        archive.mkdir(parents=True, exist_ok=True)
        path = archive / f"{plan_id}-{uuid4().hex}.json"
        path.write_text(json.dumps({"error": str(error), "result": result},
                                   ensure_ascii=False, indent=2))
    except OSError as archive_error:
        return err(f"Repair validation could not be recorded: {error}",
                   archive_error=str(archive_error))
    return err(f"Repair validation could not be recorded: {error}",
               attempted_review_path=str(path))


def _prefilter_decision(fn, scores: dict) -> dict[str, Any]:
    """Bind the scheduling decision to the evaluated detector and current policy."""
    from autofa.metrics.claim_contract import claim_hash
    from autofa.metrics.judge_prefilter import judge_prefilter

    fingerprint = claim_hash(
        source=review_source(fn), description=fn.description or "",
        mechanism=fn.mechanism or fn.description or fn.name,
    )
    association_scores = scores
    if not scores.get("within_parent") and (
        explains := fn.explains or (fn.meta or {}).get("explains")
    ):
        # A multi-target corpus evaluation may lack the parent's stratum. Never
        # substitute its global association for an explainer's missing metrics.
        association_scores = {**scores, "explains": explains}
    decision = judge_prefilter(association_scores)
    if scores.get("evaluation_claim_hash", fingerprint) != fingerprint:
        decision.update(
            status="not_evaluated", criteria=[],
            reason="The detector or claim changed after its association evaluation.",
        )
    decision["claim_hash"] = fingerprint
    return decision


def _review_settings(config) -> tuple[dict, dict, dict]:
    """Resolve the configured judge identity before any paid workflow work."""
    from autofa.harness.settings import behavior_judge_settings
    from autofa.metrics.judge_backend import judge_reasoning_arguments, judge_reasoning_metadata

    judge_config = behavior_judge_settings(config)
    reasoning = judge_reasoning_arguments(
        harness=judge_config["harness"], thinking=judge_config["thinking"],
        effort=judge_config["effort"],
        thinking_budget_tokens=judge_config["thinking_budget_tokens"],
    )
    settings = {"judge_model": judge_config["model"],
                "judge_objective_strategy": discovery_strategy(config),
                "judge_behavior_strictness": strictness_for_config(config),
                **judge_reasoning_metadata(harness=judge_config["harness"], **reasoning)}
    return judge_config, reasoning, settings


def _review_key(function: dict, settings: dict, evidence_n: int | None, plan_id=None) -> str:
    from autofa.metrics.claim_contract import claim_hash
    from autofa.metrics.claude_judge import judge_policy_hash

    fingerprint = claim_hash(
        source=review_source(function), description=function.get("description") or "",
        mechanism=function.get("mechanism") or function.get("description") or function["name"],
    )
    # The default causal profile is omitted so explicit and implicit causal
    # settings share one stable reservation key.
    key_settings = dict(settings)
    key_settings.pop("judge_objective_strategy", None)
    if key_settings.get("judge_behavior_strictness", "causal") == "causal":
        key_settings.pop("judge_behavior_strictness", None)
    return hashlib.sha256(json.dumps({
        "claim": fingerprint, "policy": judge_policy_hash(
            behavior_strictness=settings.get("judge_behavior_strictness", "causal")),
        "plan": plan_id, "settings": key_settings, "evidence_n": evidence_n,
    }, sort_keys=True).encode()).hexdigest()


def _review_settings_match(scores: dict, settings: dict) -> bool:
    """Search preferences are metadata, not a review setting or recovery gate."""
    return all(scores.get(key) == value
               for key, value in settings.items() if key != "judge_objective_strategy")


def _find_compatible_allocation(rt, function, settings, evidence_n, plan_id, phase,
                               *, support=False):
    from autofa.session.review_evidence import find_review_allocation

    key = _review_key(function, settings, evidence_n, plan_id)
    if support:
        key = _development_support_key(key)
    allocation = find_review_allocation(rt.run.run_dir(), function["name"],
                                         review_key=key, phase=phase)
    if allocation is None and phase == "development":
        diagnostic_key = key + ":smoke-only"
        diagnostic = find_review_allocation(
            rt.run.run_dir(), function["name"], review_key=diagnostic_key, phase=phase)
        if diagnostic is not None and (
            not diagnostic["operation_completed"]
            or ((function.get("scores") or {}).get("sampled") is True
                and float((function.get("scores") or {}).get("n_flagged") or 0) == 0)
        ):
            return diagnostic_key, diagnostic
    return key, allocation


def _development_support_key(review_key: str) -> str:
    from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS

    return hashlib.sha256(json.dumps({
        "review_key": review_key, "development_support": MIN_GOOD_FAILED_TASKS,
    }, sort_keys=True).encode()).hexdigest()


def judge_function(
    name: str, evidence_n: int | None = None, review_phase: str = "auto",
    review_mode: str = "standard",
) -> dict[str, Any]:
    """Judge sampled rollouts in fresh sessions using the selected judge harness.

    Judges a **random sample** from the fire / detection bucket:
    ``max(3, ceil(10% of fires))`` items (or all fires if fewer), unless
    ``evidence_n`` raises that floor. After family comparison, explicit development
    requests of at least five also retain current review cases and seek support
    across five distinct failed tasks. They remain diagnostic, not held-out evidence.
    Development reviews of LLM detectors reuse
    the current evaluation panel, including during repair; insufficient support
    produces a limited review instead of additional detector calls. Cheap
    deterministic repair scans and explicit confirmation may extend the initial
    scan up to ``max_eval_limit``. If no fresh matches remain, re-review previous
    matches and mark the scores as using reused evidence.
    In family discovery, auto uses development evidence and reserves other tasks
    for optional independent confirmation. Development findings can support Good
    when the shared quality criteria are met. If fresh generalization evidence is
    needed, freeze the candidate, then request review_phase="confirmation" to scan
    for untouched task evidence.
    Synthesized families sample retained seed support, matches on additional
    source-identified tasks, and passing matches within the existing budget.
    New ordinary development allocations also reserve at most one separate
    outcome-diagnostic control when their scored cohort misses failed or passing
    matches. This control does not change adherence or confirmed-support scores.
    When a sampled scoring panel has no failed matches, up to three compatible
    failed-task smoke positives outside the panel can be reviewed separately.
    These author-selected controls provide development feedback only, never
    representative adherence, confirmed support, or population coverage.
    Unavailable confirmation evidence is reported without repeating paid reviews.

    ``review_mode="diagnostic"`` reviews at most three development examples for
    actionable feedback only. It never expands, adds outcome controls, establishes
    adherence or task support, or validates a repair. ``standard`` is the default
    deliberate review; ``automatic`` schedules new reviews by normalized precision
    on usable scored matches. Low coverage or few distinct matched tasks do not
    block adherence review; they still affect detector quality grades.
    """
    return _judge_function(get_runtime(), name, evidence_n=evidence_n,
                           review_phase=review_phase, review_mode=review_mode)


def _judge_function(  # noqa: C901
    rt: Runtime, name: str, evidence_n: int | None = None, review_phase: str = "auto",
    review_mode: str = "standard",
) -> dict[str, Any]:
    check_cancelled()
    if (not isinstance(review_mode, str)
            or review_mode not in {"standard", "diagnostic", "automatic"}):
        return err("review_mode must be standard, diagnostic, or automatic")
    if (not isinstance(review_phase, str)
            or review_phase not in {"auto", "development", "confirmation"}):
        return err("review_phase must be auto, development, or confirmation")
    if evidence_n is not None and (type(evidence_n) is not int or evidence_n <= 0):
        return err("evidence_n must be a positive integer or omitted")
    diagnostic_mode = review_mode == "diagnostic"
    if diagnostic_mode and review_phase == "confirmation":
        return err("Diagnostic review uses development evidence; "
                   "confirmation requires standard mode")
    if diagnostic_mode and evidence_n is not None and evidence_n > 3:
        return err("Diagnostic review allows at most three examples; evidence_n must be 1, 2, or 3")
    tracked_review = (diagnostic_mode or review_phase != "auto"
                      or (rt.run.run_dir() / "family_discovery.json").exists())
    phase = "development" if review_phase == "auto" else review_phase
    fn = rt.library.get(name)
    if fn is None:
        return err(f"unknown function: {name}")

    from autofa.mcp.tools_repair import review_next_action, review_recovery_action
    from autofa.metrics.claude_judge import compatible_judge_policy, judge_policy_hash
    from autofa.session import repair_workflow

    # Check before any detector scan, claim compilation, review session, or repair
    # holdout reservation. Explicit judge calls follow the same policy as submit.
    decision = _prefilter_decision(fn, fn.scores or {})
    if decision["status"] == "skip":
        from autofa.metrics.claim_calibration import summarize_judge_feedback
        from autofa.session.status_log import emit_runtime_status

        scores = dict(fn.scores or {})
        scores["judge_prefilter"] = decision
        verdict = verdict_for_scores(scores)
        scores.update(badge=verdict["label"], verdict=verdict)
        feedback = dict(scores.get("feedback") or {})
        feedback.update(
            behavior_purity=None, behavior_specificity=None,
            behavior_validation=behavior_validation(scores, include_details=False),
            judge_prefilter=decision,
        )
        scores["feedback"] = feedback
        with cancellation_guard():
            fn.scores = scores
            rt.persist_library()
        from autofa.metrics.live_progress import refresh_display_progress

        refresh_display_progress(rt)
        checkpoint_discovery_stall(rt)
        judge_feedback = summarize_judge_feedback(
                scores, name=name, behavior_strictness=strictness_for_config(rt.config))
        message = f"Behavior judging skipped for {name}: {decision['reason']}"
        emit_runtime_status(message, source="judge_function", context={"name": name})
        return ok(
            name=name, skipped=True, status="skipped_pre_filter",
            verdict=_current_verdict(rt, fn),
            discovery_grade=_current_discovery_grade(rt, fn),
            reason=decision["reason"], judge_prefilter=decision,
            behavior_purity=None, behavior_specificity=None,
            n_judged=0, feedback=feedback, judge_feedback=judge_feedback,
            search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
            discovery_feedback=build_discovery_feedback(
                scores, judge_feedback=judge_feedback,
                behavior_strictness=strictness_for_config(rt.config),
                target_grade=agent_target_grade_for_runtime(rt)),
            hint=" ".join(judge_feedback["next_steps"]), message=message,
        )
    objective_strategy = discovery_strategy(rt.config)
    behavior_strictness = strictness_for_config(rt.config)
    judge_config, judge_reasoning, judge_settings = _review_settings(rt.config)
    from autofa.metrics.task_support import MIN_GOOD_FAILED_TASKS
    from autofa.session.family_discovery import family_comparison_pending
    from autofa.session.polishing_guard import active_polishing

    comparison_pending = (active_polishing(rt) is None
                          and family_comparison_pending(rt.run.run_dir()))
    support_expansion_requested = (not diagnostic_mode
                                   and review_phase == "development" and not comparison_pending
                                   and type(evidence_n) is int
                                   and evidence_n >= MIN_GOOD_FAILED_TASKS)
    confirmation_key = (_find_compatible_allocation(
        rt, fn.to_dict(), judge_settings, evidence_n, None, "confirmation",
    )[0] if phase == "confirmation" else None)
    development_key = (_find_compatible_allocation(
        rt, fn.to_dict(), judge_settings, evidence_n, None, "development", support=True,
    )[0] if support_expansion_requested else None)
    try:
        repair_context = repair_workflow.validation_context(
            rt.run.run_dir(), fn.to_dict(), resume_reserved=True,
            allow_deferred_confirmation=phase == "confirmation",
            allow_deferred_development=support_expansion_requested,
            development_review_key=development_key,
            objective_strategy=objective_strategy,
            behavior_strictness=behavior_strictness,
            confirmation_review_key=confirmation_key,
            judge_settings=judge_settings,
        )
    except (ValueError, OSError) as exc:
        return err(f"Repair validation is not ready: {exc}",
                   next_action=review_next_action(rt, fn.to_dict(), evidence_n=evidence_n))
    resume_policy_hash = (repair_context or {}).get("judge_policy_hash")
    if repair_context:
        if repair_context["judge_behavior_strictness"] != behavior_strictness:
            return err("Repair validation requires its original Judge Policy. Restore it or "
                       "defer this plan before requesting a new validation.",
                       next_action=review_next_action(rt, fn.to_dict()))
        # Recover the operation's policy and provenance even if search weights
        # changed; this does not change the runtime's optimization settings.
        objective_strategy = repair_context["judge_objective_strategy"]

    previous_scores = deepcopy(fn.scores or {})
    if diagnostic_mode and (
        not previous_scores.get("evaluated") or previous_scores.get("evaluation_stale")
        or previous_scores.get("evaluation_claim_hash") != decision["claim_hash"]
        or previous_scores.get("evaluation_run_id") != rt.run.run_id
    ):
        return err("Evaluate the current detector revision before diagnostic review.",
                   evaluation_required=True)
    from autofa.metrics.review_content import saved_review_localisations

    previous_review_localisations = saved_review_localisations(fn)
    scores = dict(previous_scores)
    scores["judge_prefilter"] = decision
    from autofa.metrics.judge_backend import backend_arguments, backend_name

    judge_harness = judge_config["harness"]
    judge_model = judge_config["model"]
    excluded_ids = set((repair_context or {}).get("exclusion_item_ids") or [])
    retry_ids = set((repair_context or {}).get("retry_item_ids") or [])
    reserved_ids = (repair_context or {}).get("reserved_item_ids")
    if diagnostic_mode and (retry_ids or reserved_ids is not None):
        return err(
            "A reserved repair review or processing retry must resume in standard mode; "
            "diagnostic review cannot replace its frozen validation cohort.",
            next_action={"tool": "judge_function",
                         "arguments": {"name": name, "review_mode": "standard",
                                       "review_phase": review_phase}},
        )
    selected_ids = set(reserved_ids) if reserved_ids is not None else retry_ids
    if tracked_review and (retry_ids or reserved_ids is not None):
        # A resumed operation keeps its original cohort and review purpose.
        saved_phase = ((repair_context or {}).get("sample") or
                       scores.get("behavior_sample") or {}).get("phase")
        if saved_phase and review_phase != "auto" and review_phase != saved_phase:
            return err("A reserved review must resume its original review_phase")
        phase = saved_phase or "development"
    expand_support = bool(
        not diagnostic_mode and tracked_review and phase == "development" and not comparison_pending
        and type(evidence_n) is int and evidence_n >= MIN_GOOD_FAILED_TASKS
        and reserved_ids is None and not retry_ids
    )
    retained_review_ids = []
    if (expand_support and scores.get("claim_hash") == decision["claim_hash"]
            and not scores.get("behavior_judgment_stale")
            and compatible_judge_policy(
                scores.get("judge_policy_hash"), behavior_strictness=behavior_strictness)
            and _review_settings_match(scores, judge_settings)):
        # Retain every current finding, including counterexamples; checkpoints
        # decide whether a previous per-case result is reusable under this policy.
        retained_review_ids = [row["item_id"] for row in scores.get("behavior_verdicts", [])
                               if isinstance(row, dict) and isinstance(row.get("item_id"), str)]
    review_key = None
    frozen_allocation = None
    preserve_review_authority = False
    if tracked_review and reserved_ids is None and not retry_ids:
        # The automatic target depends on scanned fires, which can change after
        # exposure. Identify the operation first and retain its original target.
        if diagnostic_mode:
            from autofa.session.review_evidence import find_review_allocation

            review_key = _review_key(
                fn.to_dict(), judge_settings, evidence_n,
                (repair_context or {}).get("plan_id")) + ":diagnostic"
            frozen_allocation = find_review_allocation(
                rt.run.run_dir(), name, review_key=review_key, phase="development")
            saved_policy = None
        else:
            review_key, frozen_allocation = _find_compatible_allocation(
                rt, fn.to_dict(), judge_settings, evidence_n,
                (repair_context or {}).get("plan_id"), phase)
            saved_policy = judge_policy_hash(behavior_strictness=behavior_strictness)
        if expand_support and (frozen_allocation is None
                               or frozen_allocation["operation_completed"]):
            # An explicit post-comparison expansion is a new operation, not a
            # mutation of the earlier bounded cohort. Interrupted operations keep
            # their original IDs and target before any expansion is considered.
            review_key, frozen_allocation = _find_compatible_allocation(
                rt, fn.to_dict(), judge_settings, evidence_n,
                (repair_context or {}).get("plan_id"), phase, support=True)
        if frozen_allocation is not None:
            selected_ids = set(frozen_allocation["item_ids"])
            from autofa.metrics.review_assessments import (
                archived_assessments,
                assessment_record,
                assessment_summary,
                cohort_ids,
                completed_assessment,
            )

            requested_assessment = next((saved for saved in reversed(archived_assessments(scores))
                                         if cohort_ids(saved) == selected_ids
                                         and saved.get("claim_hash") == decision["claim_hash"]
                                         and saved.get("behavior_review_phase") == phase
                                         and (not saved.get("behavior_review_operation") or
                                              saved["behavior_review_operation"] == {
                                                  "origin_run_id": rt.run.run_id,
                                                  "scope": "standard", "phase": phase,
                                                  "review_key": review_key,
                                              })
                                         and saved.get("judge_policy_hash")
                                         == (saved_policy or scores.get("judge_policy_hash"))
                                         and _review_settings_match(saved, judge_settings)), None)
            # Reopening an already completed operation, including an active repair's
            # development review, must not replace the newer assessment with its older
            # cohort, even while the newer one is unfinished: its own operation finishes it.
            # Legacy saves lack the assessment archive: only migrate a default-size
            # development replay with at least its original support, never an explicitly
            # sized request. Earlier archives may still carry the pending-cohort marker.
            superseded_review = (
                frozen_allocation["operation_completed"]
                and bool(cohort_ids(scores)) and cohort_ids(scores) != selected_ids
                and (requested_assessment is not None
                     and completed_assessment(assessment_record(requested_assessment))
                     or (requested_assessment is None and evidence_n is None
                         and phase == "development"
                         and len(cohort_ids(scores)) >= len(selected_ids)))
            )
            # Resuming an older unfinished operation saves its cohort separately, even
            # behind an unfinished newer one. A repair's tracked development review
            # records no repair outcome, so it follows the same rule.
            preserve_review_authority = (
                (not repair_context or phase == "development")
                and not diagnostic_mode
                and not frozen_allocation["operation_completed"]
                and requested_assessment is not None and bool(cohort_ids(scores))
                and cohort_ids(scores) != selected_ids
                and scores.get("claim_hash") == decision["claim_hash"]
                and scores.get("behavior_review_phase") == phase
                and compatible_judge_policy(
                    scores.get("judge_policy_hash"), behavior_strictness=behavior_strictness)
                and _review_settings_match(scores, judge_settings)
            )
            if preserve_review_authority:
                from autofa.metrics.review_content import RetainedReviewContent

                verifier = RetainedReviewContent([fn])
                try:
                    with closing(_fetch_items_by_id(
                        rt.dataset, verifier.missing_item_ids(), max_items=None,
                    )) as rows:
                        for item in rows:
                            check_cancelled()
                            verifier.observe(item)
                except (OSError, ValueError) as exc:
                    return err(f"Current assessment evidence could not be checked: {exc}.")
                preserve_review_authority = not verifier.missing_item_ids() and not verifier.apply(
                    fn.name, scores).get("behavior_evidence_stale")
            frozen_smoke_ids = ((frozen_allocation.get("sample_info") or {}).get(
                "diagnostic_sample") or {}).get("off_panel_item_ids")
            if frozen_smoke_ids:
                from autofa.metrics.diagnostic_review import known_failed_smoke_matches

                current_controls = known_failed_smoke_matches(
                    rt, fn, client=_llm_adapter(), frozen_ids=frozen_smoke_ids)
                if {str(item["_item_id"]) for item, _ in current_controls} != set(frozen_smoke_ids):
                    return err("Frozen smoke controls are missing or stale. Restore the original "
                               "source, model settings, field policy, and evidence before "
                               "resuming, or re-run smoke diagnostics on these same items.",
                               stale_diagnostic_controls=True, item_ids=frozen_smoke_ids)
            # A superseded replay never displaces the newer assessment, whatever its
            # phase and even while a later checkpoint left it marked pending: the
            # content check below decides whether its evidence actually changed.
            if (not diagnostic_mode and frozen_allocation["operation_completed"]
                    and (superseded_review or not scores.get("behavior_evidence_stale"))
                    and not scores.get("diagnostic_evidence_stale")
                    and scores.get("claim_hash") == decision["claim_hash"]
                    and compatible_judge_policy(
                        scores.get("judge_policy_hash"), behavior_strictness=behavior_strictness)
                    and _review_settings_match(scores, judge_settings)
                    and (scores.get("behavior_review_phase") == phase or superseded_review)
                    and (cohort_ids(scores) == selected_ids or superseded_review)
                    and (not frozen_allocation.get("diagnostic_item_ids") or (
                        {row.get("item_id") for row in
                         (scores.get("diagnostic_review") or {}).get("verdicts", [])}
                        >= set(frozen_allocation["diagnostic_item_ids"])
                        and all(
                            row.get("verdict") not in {"error", "processing_error"}
                            and row.get("completion_status") != "needs_evidence_review"
                            for row in (scores.get("diagnostic_review") or {}).get("verdicts", [])
                            if row.get("item_id") in frozen_allocation["diagnostic_item_ids"])))):
                from autofa.metrics.review_content import RetainedReviewContent

                # A completed reservation freezes membership, not evidence content.
                # Check the original cohort before bypassing the per-item snapshot
                # checks, including newly accessible retained image evidence.
                retained = RetainedReviewContent([fn])
                try:
                    with closing(_fetch_items_by_id(
                        rt.dataset, retained.missing_item_ids(), max_items=None,
                    )) as rows:
                        for item in rows:
                            check_cancelled()
                            retained.observe(item)
                except (OSError, ValueError) as exc:
                    return err(f"Saved review evidence could not be checked: {exc}. "
                               "Restore source access before reusing this completed review.")
                if retained.missing_item_ids():
                    return err("Saved review evidence is missing; restore the original dataset "
                               "before reusing this completed review. Its cohort is unchanged.")
                # A later checkpoint's pending marker is not a content change: the replay
                # still returns early, but the newer assessment stays reported unfinished.
                pending = superseded_review and bool(scores.get("behavior_evidence_stale"))
                scores = retained.apply(fn.name, scores)
                if (not scores.get("behavior_evidence_stale")
                        and not scores.get("diagnostic_evidence_stale")):
                    from autofa.metrics.claim_calibration import summarize_judge_feedback
                    from autofa.metrics.diagnostic_review import summarize_diagnostic_review

                    if pending:
                        scores = {**scores, "behavior_evidence_stale": True}
                    verdict = _current_verdict(rt, fn)
                    judge_feedback = summarize_judge_feedback(
                        scores, name=name, behavior_strictness=behavior_strictness)
                    checkpoint_discovery_stall(rt)
                    unfinished = superseded_review and not completed_assessment(scores)
                    # An active repair continues from its saved review, not a replay; failed
                    # current processing is recovered through its own operation.
                    next_action = (review_next_action(rt, fn.to_dict(), evidence_n=evidence_n)
                                   if repair_context else
                                   review_recovery_action(rt, fn.to_dict()) if unfinished
                                   else None)
                    return ok(name=name, skipped=True, status="review_already_completed",
                              sample=scores.get("behavior_sample"), verdict=verdict,
                              requested_review=(
                                  {"status": "completed", "authoritative": False,
                                   **(assessment_summary(requested_assessment)
                                      if requested_assessment is not None else
                                      {"item_ids": sorted(selected_ids)})}
                                  if superseded_review else None),
                              authoritative_review=assessment_summary(scores),
                              judge_feedback=judge_feedback,
                              discovery_feedback=build_discovery_feedback(
                                  scores, judge_feedback=judge_feedback,
                                  behavior_strictness=behavior_strictness,
                                  target_grade=agent_target_grade_for_runtime(rt)),
                              discovery_grade=_current_discovery_grade(rt, fn),
                              diagnostic_review=summarize_diagnostic_review(scores, name=name),
                              scores={**{key: value for key, value in scores.items()
                                         if key not in {"task_support", "diagnostic_review",
                                                        "behavior_review_assessments"}},
                                      "badge": verdict["label"], "verdict": verdict},
                              search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
                              message=(
                                  "This older review operation already completed. The current "
                                  "assessment is retained; the requested cohort is separate "
                                  "historical feedback. " if superseded_review else
                                  "This review operation already completed; its saved results "
                                  "are unchanged. "
                              ) + ("The current assessment is unfinished; repeating this older "
                                   "request cannot finish it. " if unfinished else "")
                              + "These are not new independent confirmations.",
                              **({"next_action": next_action} if next_action else {}))
    from autofa.metrics.repair_feedback import detector_execution_findings

    if ((repair_context or diagnostic_mode) and reserved_ids is None and not retry_ids
            and frozen_allocation is None and detector_execution_findings(fn.to_dict())):
        return err(
            "The detector still has execution errors. Inspect them before a new diagnostic "
            "or repair review; defer any active unreserved repair before another attempt.",
            repair_execution_blocked=True,
            next_action=review_next_action(rt, fn.to_dict(), evidence_n=evidence_n),
        )
    if (review_mode == "automatic" and reserved_ids is None and not retry_ids
            and frozen_allocation is None):
        from autofa.metrics.claim_calibration import summarize_judge_feedback
        from autofa.metrics.review_scheduling import automatic_review_schedule
        from autofa.session.status_log import emit_runtime_status

        schedule = automatic_review_schedule(_current_scores(rt, fn))
        schedule.update(
            evaluation_claim_hash=scores.get("evaluation_claim_hash"),
            evaluation_run_id=scores.get("evaluation_run_id"),
            cohort_fingerprint=(scores.get("search_panel") or {}).get("cohort_fingerprint"),
            evaluation_cohort_hash=hashlib.sha256(json.dumps(
                sorted(scores.get("evaluation_item_ids") or []),
                separators=(",", ":")).encode()).hexdigest(),
        )
        if schedule["status"] == "deferred":
            scores["review_schedule"] = schedule
            with cancellation_guard():
                fn.scores = scores
                rt.persist_library()
            checkpoint_discovery_stall(rt)
            judge_feedback = summarize_judge_feedback(
                scores, name=name, behavior_strictness=behavior_strictness)
            message = f"Behavior review deferred for {name}: {schedule['reason']}"
            emit_runtime_status(message, source="judge_function", context={"name": name})
            return ok(
                name=name, skipped=True, status="review_deferred", review_mode=review_mode,
                reason=schedule["reason"], review_schedule=schedule,
                verdict=_current_verdict(rt, fn),
                discovery_grade=_current_discovery_grade(rt, fn),
                judge_feedback=judge_feedback,
                discovery_feedback=build_discovery_feedback(
                    scores, judge_feedback=judge_feedback, behavior_strictness=behavior_strictness,
                    target_grade=agent_target_grade_for_runtime(rt)),
                search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
                n_judged=0, message=message,
            )
    if is_llm_detector(fn) and (unavailable := _llm_backend_error()):
        return unavailable
    # Existing reservations/retries keep their frozen cohort. New confirmation
    # waits until comparison is complete; it must not consume the search budget.
    if (comparison_pending and phase == "confirmation"
            and reserved_ids is None and not retry_ids and frozen_allocation is None):
        return err("Complete the family comparison phase before confirmation. "
                   "Use the bounded development review to compare candidates, then freeze "
                   "the selected claim and request confirmation.", family_workflow_pending=True,
                   next_action=review_next_action(rt, fn.to_dict()))
    # Workflow blockers above return first. The candidate check must clear
    # before any corpus scan or model work.
    from autofa.session.candidate_screen import ensure_candidate_screen

    screen = ensure_candidate_screen(rt, fn, check_evidence_prerequisites=False)
    if not screen.get("evaluation_allowed"):
        return err(
            "The candidate check must pass before a corpus scan or full review. "
            "Inspect the candidate_screen feedback, then revise or defer the candidate.",
            candidate_screen_blocked=True, candidate_screen=screen,
            next_action={"tool": "screen_function", "arguments": {"name": name},
                         "reason": "Recheck the candidate's evidence prerequisites and "
                                   "inspect the feedback before revising."},
        )
    record_repair_outcome = bool(not diagnostic_mode and repair_context and
                                (not tracked_review or phase == "confirmation" or retry_ids))
    if retry_ids:
        if (scores.get("judge_harness") != judge_harness
                or scores.get("judge_backend") != backend_name(judge_harness)):
            return err("Processing retry requires the original judge harness. Restore it or defer "
                       "this plan and request a fresh validation.")
        for key in ("judge_model", "judge_effort", "judge_thinking",
                    "judge_thinking_budget_tokens",
                    "judge_behavior_strictness"):
            if scores.get(key) != judge_settings.get(key):
                return err(f"Processing retry requires the original {key}. Restore it or defer "
                           "this plan and request a fresh validation.")
        from autofa.metrics.review_assessments import stale_compiled_contract

        if stale_compiled_contract(scores.get("claim_contract")):
            # A recompiled contract cannot merge into the frozen cohort; fail before paying.
            return err("Processing retry cannot reuse a claim contract frozen by an earlier "
                       "compiler policy. Defer this plan and request a fresh validation.")
    # Collect fire-bucket labels, then randomly sample for the judge.
    capped = resolve_eval_limit(
        rt.run.eval_limit,
        cfg_default=setting_value(rt.config, "default_eval_limit"),
        cfg_max=setting_value(rt.config, "max_eval_limit"),
    )
    selection = selected_bucket(scores)

    seed = int(getattr(rt.run, "exploration_seed", 0) or 0) ^ (
        zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    )
    extend_scan = (not diagnostic_mode and ((repair_context and not is_llm_detector(fn))
                   or (tracked_review and phase == "confirmation")))
    max_scan = (resolve_eval_limit(
        setting_value(rt.config, "max_eval_limit"),
        cfg_max=setting_value(rt.config, "max_eval_limit"),
    ) if extend_scan else capped)
    exposure_context = {}
    if tracked_review:
        from autofa.session.review_evidence import review_context

        exposure_context = review_context(rt.run.run_dir(), name)
    exposed_ids = set(exposure_context.get("reviewed_item_ids") or [])
    exposed_groups = set(exposure_context.get("reviewed_task_groups") or [])
    reservoir_cap = judge_reservoir_cap(capped, evidence_n=evidence_n)
    if selected_ids:
        reservoir_cap = max(reservoir_cap, len(selected_ids))
    fire_pool: list[tuple[dict[str, Any], Any]] = []
    fire_identities: list[dict[str, Any]] = []
    fire_labels: dict[str, Any] = {}
    reused_pool: list[tuple[dict[str, Any], Any]] = []
    n_reused_fires = 0
    reused_rng = random.Random(seed ^ 0x52455553)
    localisations = {fn.name: dict(fn.failure_localisations)}
    n_fires = 0
    n_eligible = 0
    seen_fire_ids: set[str] = set()
    rng = random.Random(seed)
    workers = (
        _llm_eval_settings(rt.config)[2]
        if is_llm_detector(fn) else 1
    )
    n_scanned = 0
    scan_ids = None
    scan_scope = "capped_corpus"
    if reserved_ids is not None or retry_ids:
        scan_ids = selected_ids
        scan_scope = "reserved_repair_evidence"
    elif frozen_allocation is not None:
        off_panel = set((frozen_allocation["sample_info"].get("diagnostic_sample") or {})
                        .get("off_panel_item_ids") or [])
        scan_ids = selected_ids | (set(frozen_allocation.get("diagnostic_item_ids") or [])
                                   - off_panel)
        scan_scope = "reserved_review_evidence"
    elif phase == "development" and is_llm_detector(fn):
        # Repair plans must not turn a bounded scoring panel into a fresh paid
        # corpus scan. Completed detector checkpoints also make this reuse free
        # of additional detector calls while still validating artifact freshness.
        current_evaluation = (
            scores.get("evaluated")
            and scores.get("evaluation_run_id") == rt.run.run_id
            and scores.get("evaluation_claim_hash") == decision["claim_hash"]
            and not scores.get("evaluation_stale")
        )
        panel_ids = scores.get("evaluation_item_ids")
        if (not current_evaluation
                or (scores.get("sampled") and not isinstance(panel_ids, list))):
            return err(
                "Evaluate this LLM detector's current revision before development review. "
                "Use evaluate_functions(mode='capped') to establish the bounded scoring panel; "
                "judge_function will reuse it, including during repair.",
                evaluation_required=True,
            )
        if isinstance(panel_ids, list):
            scan_ids = set(panel_ids) | set(retained_review_ids)
            scan_scope = "evaluated_llm_panel"
        else:
            # A census (or sampling disabled) has no panel IDs. Reuse only the
            # physical prefix already attempted, even if the run limit grew.
            evaluated_n = int(scores.get("n_scored") or 0) + int(scores.get("n_unresolved") or 0)
            capped = max_scan = (evaluated_n if expand_support
                                  else min(capped, evaluated_n))
    if scan_ids is not None:
        capped = max_scan = len(scan_ids)
        scan_rows = _fetch_items_by_id(rt.dataset, scan_ids, max_items=None)
        # Preserve the source iterator's byte bound, including oversized rows.
        batches = ([row] for row in scan_rows)
    else:
        scan_rows = None
        batches = iter(rt.dataset.iter_batches(max_items=max_scan))
    pending_batch: deque[dict[str, Any]] = deque()

    def scan_pairs(scan_limit):
        nonlocal n_scanned
        # Stop the producer at the exact phase boundary: a repair scan may
        # extend later, but must not prefetch extra paid detector calls now.
        while n_scanned < scan_limit:
            check_cancelled()
            if not pending_batch:
                pending_batch.extend(next(batches, []))
                if not pending_batch:
                    return
            n_scanned += 1
            yield fn, pending_batch.popleft()

    try:
        for scan_limit in dict.fromkeys((capped, max_scan)):
            if scan_limit > capped:
                from autofa.session.status_log import emit_runtime_status

                # A shortage means every eligible prefix item still fits in the
                # reservoir, so enlarging it here cannot bias the extended sample.
                reservoir_cap = max(
                    reservoir_cap, judge_reservoir_cap(scan_limit, evidence_n=evidence_n),
                )
                emit_runtime_status(
                    f"Extending repair evidence scan for {name} "
                    f"from {capped} to {scan_limit} items",
                    source="judge_function", context={"name": name, "scan_limit": scan_limit},
                )
            with closing(_iter_scored_items(
                scan_pairs(scan_limit), workers=workers, llm=_llm_adapter(),
                localisations=localisations,
            )) as scored:
                for _, it, pred, error in scored:
                    if error is not None:
                        continue
                    if not matches_selected_bucket(pred, selection):
                        continue
                    item_id = str(it.get("_item_id") or "")
                    if repair_context or tracked_review:
                        if not item_id.strip() or item_id in seen_fire_ids:
                            return err(
                                "Repair validation requires unique, nonempty item IDs in the "
                                "fire bucket. Fix the dataset identity before retrying."
                            )
                        seen_fire_ids.add(item_id)
                    n_fires += 1
                    if tracked_review and reserved_ids is None and not retry_ids:
                        from autofa.session.family_coverage import verified_identity_projection
                        from autofa.session.review_evidence import task_group

                        # Keep metadata, not every full trajectory, for task-diverse allocation.
                        identity = verified_identity_projection(rt, it)
                        fire_identities.append(identity)
                        fire_labels[item_id] = pred
                        # Total fires are not the available confirmation pool. Continue
                        # the bounded scan when the prefix contains only exposed tasks.
                        if (phase != "confirmation" or (
                            item_id not in exposed_ids and task_group(identity) is not None
                            and task_group(identity) not in exposed_groups
                        )):
                            n_eligible += 1
                        continue
                    if repair_context:
                        if reserved_ids is not None or retry_ids:
                            if item_id not in selected_ids:
                                continue
                        elif item_id in excluded_ids:
                            n_reused_fires += 1
                            if len(reused_pool) < reservoir_cap:
                                reused_pool.append((it, pred))
                            else:
                                j = reused_rng.randrange(n_reused_fires)
                                if j < reservoir_cap:
                                    reused_pool[j] = (it, pred)
                            continue
                    n_eligible += 1
                    if len(fire_pool) < reservoir_cap:
                        fire_pool.append((it, pred))
                    else:
                        j = rng.randrange(n_eligible)
                        if j < reservoir_cap:
                            fire_pool[j] = (it, pred)

            n_sample = len(selected_ids) or judge_evidence_sample_size(
                n_fires, evidence_n=evidence_n,
            )
            if diagnostic_mode and frozen_allocation is None:
                n_sample = min(n_fires, evidence_n or 3)
            needs_confirmation_pool = tracked_review and phase == "confirmation"
            if ((not repair_context and not needs_confirmation_pool)
                    or n_scanned < scan_limit or reserved_ids == []
                    or (n_sample and n_eligible >= n_sample)):
                break
    finally:
        # Release source buffers before review, including early returns/errors.
        close_batches = getattr(batches, "close", None)
        if close_batches is not None:
            close_batches()
        if scan_rows is not None:
            scan_rows.close()
        pending_batch.clear()
    it = None

    if reserved_ids is not None:
        if n_eligible != len(reserved_ids):
            return err("Reserved repair evidence is missing or no longer fires. Restore the "
                       "original dataset or defer this plan; recovery cannot replace its sample.")
        if reserved_ids:
            n_sample = len(reserved_ids)
    elif retry_ids:
        n_sample = len(retry_ids)
    # The target controls validation completeness, not whether useful evidence
    # can be reviewed. Exhaust the permitted scan before accepting a shortfall.
    target_n = (frozen_allocation["sample_info"]["target_n"]
                if frozen_allocation is not None else n_sample)
    n_sample = min(n_sample, n_eligible)
    reused_n = 0
    if not n_eligible and reserved_ids is None and not retry_ids and reused_pool:
        # Score the current version even when its fresh validation pool is
        # exhausted. Reused evidence never establishes a successful repair.
        n_sample = min(target_n, len(reused_pool))
        fire_pool = reused_pool
        reused_n = n_sample
    sample_info = {
        "target_n": target_n, "actual_n": n_sample, "eligible_n": n_eligible,
        "n_scanned": n_scanned, "scan_limit": scan_limit,
        "scan_scope": scan_scope,
        "reused_n": reused_n,
        "limited": bool(repair_context and (n_sample < target_n or not n_sample or reused_n)),
        "reason": None,
    }
    if sample_info["limited"]:
        sample_info["reason"] = (
            f"Only {n_sample} fresh validation examples available for a target of {target_n} "
            f"after scanning {n_scanned} items (scan limit {scan_limit})."
            if n_sample else "No fresh validation examples are available in the scanned data."
        )
    if reused_n:
        sample_info["reason"] = (
            f"No fresh validation examples remain; {reused_n} previously reviewed matches "
            "were re-judged for the current detector. These scores are not fresh validation."
        )
    if retry_ids:
        # A retry replaces errors within the original cohort; it cannot shrink
        # the original target or make a limited review fully validated.
        sample_info = deepcopy(scores.get("behavior_sample") or {
            "target_n": scores["behavior_n_judged"],
            "actual_n": scores["behavior_n_judged"], "limited": False, "reason": None,
        })
    elif reserved_ids is not None and repair_context.get("sample"):
        sample_info = deepcopy(repair_context["sample"])
    if tracked_review and (retry_ids or reserved_ids is not None):
        independent = (
            sample_info.get("phase") == "confirmation"
            and sample_info.get("history_complete") is True
            and sample_info.get("independence_scope") == "recorded_discovery_and_judge_exposure"
            and type(sample_info.get("independent_task_n")) is int
            and sample_info["independent_task_n"] > 0
            and sample_info.get("unknown_task_n") == 0
            and sample_info.get("unknown_exposure_item_n") == 0
            and sample_info.get("reused_n") == 0
            and sample_info.get("task_group_limited") is False
        )
        if not independent:
            sample_info["limited"] = True
            sample_info["phase"] = phase
            sample_info["reason"] = (
                "Reserved evidence lacks complete task-independence provenance. "
                "Its review remains diagnostic and cannot establish fresh confirmation. "
                + (sample_info.get("reason") or "")
            ).strip()
    diagnostic_items, diagnostic_labels = [], []
    diagnostic_result = None
    known_controls = []
    if tracked_review and reserved_ids is None and not retry_ids:
        from autofa.session.family_coverage import verified_identity_projection
        from autofa.session.review_evidence import allocate_review, task_group
        from autofa.session.synthesis_review import synthesis_review_strata

        try:
            if (phase == "confirmation" and repair_context is None
                    and (repair_workflow.get_repair(rt.run.run_dir(), name).get("state")
                         in {"deferred", "exhausted"})):
                request = {"origin_run_id": rt.run.run_id, "review_key": review_key,
                           "evidence_n": evidence_n}
                if previous_scores.get("ordinary_confirmation_request") != request:
                    # Freeze the ordinary operation before reserving any tasks.
                    # An interruption can leave the development scores intact;
                    # that must not authorize a different confirmation cohort.
                    previous_scores = {**previous_scores, "ordinary_confirmation_request": request}
                    scores["ordinary_confirmation_request"] = request
                    with cancellation_guard():
                        fn.scores = deepcopy(previous_scores)
                        rt.persist_library()
            support_review = None
            outcome_ids = None
            if not diagnostic_mode and phase == "development" and not expand_support:
                from autofa.session.family_coverage import cached_task_inventory

                inventory = cached_task_inventory(rt)
                outcomes = {
                    "failed": list(inventory["task_by_failed_item_id"]),
                    "passed": inventory["passed_item_ids"],
                }
                if not repair_context:
                    outcome_ids = outcomes
                frozen_diagnostics = ((frozen_allocation or {}).get("sample_info") or {}).get(
                    "diagnostic_sample") or {}
                frozen_ids = frozen_diagnostics.get("off_panel_item_ids")
                if ((scores.get("sampled") or frozen_ids is not None) and
                        not set(fire_labels) & set(outcomes["failed"])):
                    from autofa.metrics.diagnostic_review import known_failed_smoke_matches

                    known_controls = known_failed_smoke_matches(
                        rt, fn, client=_llm_adapter(), frozen_ids=frozen_ids,
                        excluded_task_groups={task_group(item) for item in fire_identities})
                    if frozen_ids is not None and {
                        str(item["_item_id"]) for item, _ in known_controls
                    } != set(frozen_ids):
                        return err("Frozen smoke controls are missing or stale. Restore the "
                                   "original evidence and detector settings before resuming.")
            if expand_support and frozen_allocation is None:
                from autofa.session.family_coverage import cached_task_inventory

                support_review = {
                    "retained_item_ids": retained_review_ids,
                    "failed_item_ids": list(cached_task_inventory(rt)["task_by_failed_item_id"]),
                    "min_failed_tasks": MIN_GOOD_FAILED_TASKS,
                }
            if not fire_identities and known_controls and not review_key.endswith(":smoke-only"):
                # A completed controls-only operation cannot freeze an empty
                # representative cohort after later scoring finds real matches.
                review_key += ":smoke-only"
            allocation = allocate_review(
                rt.run.run_dir(), name, fire_identities, target_n=target_n,
                seed=seed, review_key=review_key, phase=phase,
                related_names=[*fn.depends_on, *([fn.explains] if fn.explains else [])],
                coverage_review=(None if diagnostic_mode else
                                 synthesis_review_strata(rt, name, set(fire_labels))),
                support_review=support_review,
                outcome_ids=outcome_ids,
                diagnostic_items=[
                    verified_identity_projection(rt, item) for item, _ in known_controls
                ],
            )
            sample_info = {**allocation["sample_info"],
                           "n_scanned": n_scanned, "scan_limit": scan_limit,
                           "scan_scope": scan_scope}
            n_eligible = sample_info["eligible_n"]
            if phase == "confirmation" and not sample_info["independent_task_n"]:
                availability = {"status": "confirmation_unavailable", **sample_info}
                with cancellation_guard():
                    fn.scores = {**previous_scores, "confirmation_availability": availability}
                    rt.persist_library()
                return ok(
                    name=name, skipped=True, status="confirmation_unavailable", sample=sample_info,
                    verdict=_current_verdict(rt, fn),
                    discovery_grade=_current_discovery_grade(rt, fn),
                    search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
                    next_action=review_next_action(rt, fn.to_dict()),
                    message="No tasks without recorded discovery or review exposure remain. "
                            "Existing diagnostic scores are retained; collect new task evidence "
                            "or defer confirmation. Repeating this call cannot establish "
                            "freshness.",
                )
            selected = allocation["item_ids"]
            diagnostic_ids = allocation.get("diagnostic_item_ids") or []
            allocated_ids = set(selected) | set(diagnostic_ids)
            rows = {str(it["_item_id"]): it for it in _fetch_items_by_id(
                rt.dataset, allocated_ids,
                max_items=None if scan_ids is not None or known_controls else n_scanned,
            )}
            if set(rows) != allocated_ids:
                return err("Allocated review evidence is missing; restore the original dataset")
            sample = [(rows[item_id], fire_labels[item_id]) for item_id in selected]
            diagnostic_items = [rows[item_id] for item_id in diagnostic_ids]
            diagnostic_values = {**fire_labels,
                                 **{str(item["_item_id"]): label for item, label in known_controls}}
            diagnostic_labels = [diagnostic_values[item_id] for item_id in diagnostic_ids]
            n_sample = len(sample)
            if phase == "development":
                sample_info["limited"] = True
                sample_info["reason"] = (
                    "Development evidence can support discovery quality; it does not establish "
                    "independent confirmation on reserved tasks. "
                    + (sample_info.get("reason") or "")
                ).strip()
            if diagnostic_mode:
                sample_info.update(
                    review_mode="diagnostic", excluded_from_adherence=True,
                    excluded_from_confirmed_support=True,
                    reason="Bounded development diagnosis only; excluded from adherence, "
                           "confirmed support, and repair validation.",
                )
        except (OSError, ValueError) as exc:
            return err(f"Review evidence could not be allocated: {exc}")
    elif reserved_ids is not None:
        by_id = {str(item["_item_id"]): (item, label) for item, label in fire_pool}
        sample = [by_id[item_id] for item_id in reserved_ids]
    else:
        sample = sample_fire_bucket_evidence(fire_pool, n=n_sample, seed=seed)
    evidence_items = [p[0] for p in sample]
    assigned_labels = [p[1] for p in sample]
    if retry_ids:
        if not evidence_items:
            return err("No requested processing-error cases are available and still match; "
                       "restore their original evidence or defer the retry.")
        # Full-review membership serializes bucket labels as strings, while a
        # detector rerun can return bools. Recover the exact original label so
        # completed primary stages keep their content-addressed checkpoints.
        original_labels = {row["item_id"]: row["assigned_label"]
                           for row in scores.get("behavior_verdicts") or []
                           if row.get("item_id") in retry_ids and "assigned_label" in row}
        assigned_labels = [original_labels.get(str(item["_item_id"]), label)
                           for item, label in zip(evidence_items, assigned_labels, strict=True)]
    with cancellation_guard():
        fn.failure_localisations = localisations[fn.name]
    evidence_localisations = [fn.failure_localisations.get(str(it.get("_item_id") or ""))
                              for it in evidence_items]
    if record_repair_outcome and repair_context.get("plan_id"):
        review_item_ids = [str(it.get("_item_id") or "") for it in evidence_items]
        try:
            with cancellation_guard():
                repair_workflow.reserve_validation(
                    rt.run.run_dir(), name, review_item_ids,
                    expected_plan_id=repair_context["plan_id"],
                    resume_existing=reserved_ids is not None,
                    sample=sample_info,
                )
                if retry_ids:
                    # The reservation can contain a subset of the planned errors.
                    # Merge exactly that subset and retain every other original row.
                    retry_ids = set(review_item_ids)
        except (ValueError, OSError) as exc:
            return err(f"Repair evidence could not be reserved: {exc}")

    from autofa.metrics.claude_judge import judge_behavior_with_claude
    from autofa.session.guidance import guidance_prompt
    from autofa.session.recovery import RunInterruptedError
    from autofa.session.status_log import emit_runtime_status
    from autofa.ui.activity import detector_activity_context

    activity_operation = ("diagnostic" if diagnostic_mode or not evidence_items
                          else "review_adherence")
    if evidence_items or diagnostic_items:
        emit_runtime_status(
            (f"Reviewing diagnostic examples for {name}." if activity_operation == "diagnostic"
             else f"Reviewing adherence for {name}."), source="judge_function",
            context=detector_activity_context(
                name, activity_operation, scores=_current_scores(rt, fn),
                target_grade=agent_target_grade_for_runtime(rt)))

    def finish_review_activity():
        if evidence_items or diagnostic_items:
            emit_runtime_status(
                f"Review finished for {name}.", source="judge_function",
                context=detector_activity_context(
                    name, activity_operation, scores=_current_scores(rt, fn), finished=True,
                    target_grade=agent_target_grade_for_runtime(rt)))

    from autofa.metrics.review_assessments import reusable_contract

    saved_contract = reusable_contract(scores)
    judge_kwargs = dict(
            name=fn.name,
            mechanism=fn.mechanism or fn.description or fn.name,
            description=fn.description or "",
            source=review_source(fn),
            **({"compiled_contract": saved_contract} if retry_ids and saved_contract else {}),
            _retry_compilation=bool(retry_ids),
            cwd=rt.run.run_dir(),
            model=judge_model,
            objective_strategy=objective_strategy,
            behavior_strictness=behavior_strictness,
            **({"_resume_policy_hash": resume_policy_hash} if resume_policy_hash else {}),
            **backend_arguments(judge_harness),
            **judge_reasoning,
            workers=setting_value(rt.config, "judge_workers"),
            cli_path=judge_config["cli_path"],
            max_chars=min(5000, rt.config.text_truncate_chars * 2),
            analysis_guidance=guidance_prompt(rt.run),
            review_batch_context={
                "phase": phase,
                "batch": ("diagnostic" if diagnostic_mode else
                          "expanded" if sample_info.get("judge_expansion") else "initial"),
                "initial_target_n": (
                    sample_info.get("judge_expansion", {}).get("initial_target_n")
                    or sample_info["target_n"]),
                "cohort_n": len(evidence_items),
                "expansion_available": bool(
                    not diagnostic_mode and tracked_review and phase == "development"
                    and not repair_context
                    and not retry_ids and reserved_ids is None and review_key is not None
                    and evidence_n is None and not expand_support
                    and not sample_info.get("judge_expansion")),
            },
        )

    if not evidence_items:
        from autofa.metrics.claim_contract import claim_hash

        result = {
            "behavior_purity": None,
            "behavior_specificity": None,
            "n_judged": 0,
            "n_yes": 0,
            "n_no": 0,
            "n_uncertain": 0,
            "n_errors": 0,
            "n_completed": 0,
            "rationale": "no labeled datapoints to judge",
            "verdicts": [],
            "backend": "heuristic",
            "claim_hash": claim_hash(
                source=review_source(fn), description=fn.description or "",
                mechanism=fn.mechanism or fn.description or fn.name,
            ),
        }
    else:
        try:
            result = judge_behavior_with_claude(
                **judge_kwargs, items=evidence_items, assigned_labels=assigned_labels,
                failure_localisations=evidence_localisations,
            )
        except RunInterruptedError as e:
            finish_review_activity()
            return err(str(e), resumable=True,
                       resume_command=f"autofa resume --run {rt.run.run_id}")
        except Exception as e:  # noqa: BLE001
            from autofa.metrics.claim_contract import (
                JUDGE_CONTRACT_VERSION,
                build_claim_contract,
                claim_hash,
            )
            from autofa.metrics.judge_evidence import EVIDENCE_VERSION

            contract = build_claim_contract(
                source=review_source(fn), description=fn.description or "",
                mechanism=fn.mechanism or fn.description or fn.name,
                behavior_strictness=behavior_strictness,
            )
            error = f"{judge_harness} judge unavailable: {e}"
            result = {
                "behavior_purity": None,
                "behavior_specificity": None,
                "n_judged": len(evidence_items),
                "n_yes": 0,
                "n_no": 0,
                "n_uncertain": 0,
                "n_errors": len(evidence_items),
                "n_completed": 0,
                "rationale": error,
                "verdicts": [
                    {"item_id": item.get("_item_id"), "assigned_label": label,
                     "verdict": "error", "processing_status": "error",
                     "validation_error": error, "rationale": error,
                     "claim_contract_id": contract["contract_id"], "claim_assessments": []}
                    for item, label in zip(evidence_items, assigned_labels, strict=True)
                ],
                "backend": backend_name(judge_harness),
                "judge_contract_version": JUDGE_CONTRACT_VERSION,
                "claim_hash": claim_hash(
                    source=review_source(fn), description=fn.description or "",
                    mechanism=fn.mechanism or fn.description or fn.name,
                ),
                "claim_contract": contract,
                "evidence_version": EVIDENCE_VERSION,
                **judge_settings,
                "judge_objective_strategy": objective_strategy,
                "judge_policy_hash": judge_policy_hash(behavior_strictness=behavior_strictness)
                if resume_policy_hash is None else resume_policy_hash,
            }

    from autofa.metrics.adaptive_review import merge_expanded_review, sample_request

    request = sample_request(result)
    if request and not sample_info.get("judge_expansion"):
        can_expand = (not diagnostic_mode and tracked_review and phase == "development"
                      and not repair_context
                      and not retry_ids and reserved_ids is None and review_key is not None
                      and evidence_n is None and not expand_support)
        if not can_expand:
            sample_info["judge_expansion"] = {
                "status": "not_available", "request": request,
                "reason": "Automatic expansion applies only to ordinary development review. "
                          "Repair and confirmation keep their frozen evidence allocations.",
            }
        else:
            from autofa.session.family_coverage import verified_identity_projection
            from autofa.session.review_evidence import allocate_judge_expansion

            try:
                # Only IDs already matched in the original scoring pool are eligible.
                # Resume reconstructs their small identity projections, not a new scan.
                pool_ids = set(allocation.get("candidate_item_ids") or [])
                if not pool_ids:
                    raise ValueError("Historical allocation has no frozen matched pool")
                identities = [verified_identity_projection(rt, item) for item in
                              _fetch_items_by_id(rt.dataset, pool_ids, max_items=None)]
                expanded = allocate_judge_expansion(
                    rt.run.run_dir(), name, identities, review_key=review_key,
                    seed=seed, request=request,
                )
                sample_info.update(expanded["sample_info"])
                added_ids = set(expanded["sample_info"]["judge_expansion"]["added_item_ids"])
                added_items, added_labels = [], []
                with closing(_fetch_items_by_id(rt.dataset, added_ids, max_items=None)) as rows:
                    with closing(_iter_scored_items(
                        ((fn, item) for item in rows), workers=workers, llm=_llm_adapter(),
                        localisations=localisations,
                    )) as predictions:
                        for _, item, label, error in predictions:
                            if error is not None or not matches_selected_bucket(label, selection):
                                raise ValueError("Frozen additional evidence no longer fires; "
                                                 "restore the original dataset before retrying")
                            added_items.append(item)
                            added_labels.append(label)
                if {str(item["_item_id"]) for item in added_items} != added_ids:
                    raise ValueError("Frozen additional review evidence is missing")
                if added_items:
                    # Source traversal order may change on resume; preserve the allocation.
                    ordered = {str(item["_item_id"]): (item, label)
                               for item, label in zip(added_items, added_labels, strict=True)}
                    added_items, added_labels = map(list, zip(*(ordered[item_id] for item_id in
                        expanded["sample_info"]["judge_expansion"]["added_item_ids"]), strict=True))
                    added_localisations = [localisations[fn.name].get(str(item["_item_id"]))
                                           for item in added_items]
                    from autofa.session.status_log import emit_runtime_status

                    emit_runtime_status(
                        f"Judge requested {len(added_items)} additional distinct-task reviews "
                        f"for {name}; original findings retained.",
                        source="judge_function", context={"name": name, "request": request},
                    )
                    additional = judge_behavior_with_claude(
                        **{**judge_kwargs, "compiled_contract": reusable_contract(result),
                           "_retry_compilation": False,
                           "review_batch_context": {
                               "phase": phase, "batch": "additional",
                               "initial_target_n": (
                                   sample_info["judge_expansion"]["initial_target_n"]),
                               "cohort_n": len(evidence_items) + len(added_items),
                               "expansion_available": False,
                           }},
                        items=added_items, assigned_labels=added_labels,
                        failure_localisations=added_localisations,
                    )
                    result = merge_expanded_review(result, additional)
                    evidence_items += added_items
                    assigned_labels += added_labels
                    evidence_localisations += added_localisations
                    n_sample = len(evidence_items)
                    sample_info["judge_expansion"]["status"] = (
                        "incomplete" if additional.get("n_errors")
                        or additional.get("n_review_incomplete") else "complete")
            except RunInterruptedError as exc:
                finish_review_activity()
                return err(str(exc), resumable=True,
                           resume_command=f"autofa resume --run {rt.run.run_id}")
            except Exception as exc:  # noqa: BLE001 - preserve completed original judgments
                # The useful original verdicts survive allocation/source failures.
                sample_info["actual_n"] = len(evidence_items)
                sample_info["judge_expansion"] = {
                    **sample_info.get("judge_expansion", {}), "status": "blocked",
                    "request": request, "reason": str(exc),
                }

    expansion = sample_info.get("judge_expansion") or {}
    if expansion.get("added_item_ids") and expansion.get("status") != "blocked":
        expected = set(expansion["initial_item_ids"] + expansion["added_item_ids"])
        expansion["status"] = (
            "complete" if {row["item_id"] for row in result["verdicts"]} == expected
            and not result.get("n_errors") and not result.get("n_review_incomplete")
            else "incomplete")

    if diagnostic_items and (result.get("claim_contract") or not evidence_items):
        try:
            diagnostic_result = judge_behavior_with_claude(
                **{**judge_kwargs,
                   **({"compiled_contract": reusable_contract(result)}
                      if reusable_contract(result) else {}),
                   "_retry_compilation": False,
                   "review_batch_context": {
                       "phase": phase, "batch": "outcome_diagnostic",
                       "cohort_n": len(diagnostic_items), "expansion_available": False,
                       "excluded_from_adherence": True,
                   }},
                items=diagnostic_items, assigned_labels=diagnostic_labels,
                failure_localisations=[
                    localisations[fn.name].get(str(item["_item_id"]))
                    for item in diagnostic_items
                ],
            )
            diagnostic_result.update(
                sample=deepcopy(sample_info["diagnostic_sample"]),
                excluded_from_adherence=True, excluded_from_confirmed_support=True,
                review_localisations={
                    str(item["_item_id"]): deepcopy(
                        localisations[fn.name].get(str(item["_item_id"])))
                    for item in diagnostic_items
                },
            )
            sample_info["diagnostic_sample"]["status"] = (
                "complete" if not diagnostic_result.get("n_errors")
                and not diagnostic_result.get("n_review_incomplete") else "incomplete")
            if not evidence_items:
                # Carry freshness/contract metadata without treating these selected
                # cases as representative adherence observations.
                result.update({key: value for key, value in diagnostic_result.items()
                               if key.startswith("judge_") or key in {
                                   "claim_hash", "claim_contract", "claim_compilation",
                                   "evidence_version",
                               }})
        except RunInterruptedError as exc:
            finish_review_activity()
            return err(str(exc), resumable=True,
                       resume_command=f"autofa resume --run {rt.run.run_id}")
        except Exception as exc:  # noqa: BLE001 - keep the completed scored cohort
            sample_info["diagnostic_sample"].update(status="blocked", error=str(exc))

    finish_review_activity()
    check_cancelled()
    result.setdefault("judge_objective_strategy", objective_strategy)
    result.setdefault("judge_behavior_strictness", behavior_strictness)
    result["required_items"] = sample_info["target_n"]
    if diagnostic_mode:
        from autofa.metrics.claim_calibration import summarize_judge_feedback
        from autofa.metrics.diagnostic_review import (
            retain_current_diagnostic_feedback,
            summarize_diagnostic_review,
        )
        from autofa.metrics.live_progress import refresh_display_progress
        from autofa.session.review_evidence import complete_review_allocation

        result.update(
            review_mode="diagnostic", sample=deepcopy(sample_info),
            excluded_from_adherence=True, excluded_from_confirmed_support=True,
            excluded_from_repair_validation=True,
            review_localisations={
                str(item["_item_id"]): deepcopy(localisation)
                for item, localisation in zip(
                    evidence_items, evidence_localisations, strict=True)
            },
        )
        # Diagnostic feedback has its own provenance. Do not apply_review_result:
        # even three "yes" decisions must leave all standard quality scores intact.
        scores["diagnostic_review"] = retain_current_diagnostic_feedback(
            rt, fn, previous_scores, result)
        scores.pop("diagnostic_evidence_stale", None)
        with cancellation_guard():
            if (result.get("n_judged") and not result.get("n_errors")
                    and not result.get("n_review_incomplete")
                    and result.get("n_completed", 0) + result.get("n_uncertain", 0)
                    == result["n_judged"]):
                complete_review_allocation(
                    rt.run.run_dir(), name, review_key=review_key, phase="development")
            fn.scores = scores
            rt.persist_library()
        refresh_display_progress(rt)
        checkpoint_discovery_stall(rt)
        judge_feedback = summarize_judge_feedback(
            scores, name=name, behavior_strictness=behavior_strictness)
        status = ("diagnostic_review_unavailable" if not result.get("n_judged") else
                  "diagnostic_review_incomplete" if (
                      result.get("n_errors") or result.get("n_review_incomplete")) else
                  "diagnostic_review_completed")
        return ok(
            name=name, review_mode="diagnostic", status=status,
            excluded_from_adherence=True, excluded_from_confirmed_support=True,
            excluded_from_repair_validation=True,
            sample=sample_info, n_judged=result.get("n_judged", 0),
            n_yes=result.get("n_yes", 0), n_no=result.get("n_no", 0),
            n_uncertain=result.get("n_uncertain", 0), n_errors=result.get("n_errors", 0),
            n_completed=result.get("n_completed", 0),
            n_review_incomplete=result.get("n_review_incomplete", 0),
            n_sampled=n_sample, n_fires=n_fires, n_scanned=n_scanned,
            verdicts=result.get("verdicts"), diagnostic_review=summarize_diagnostic_review(
                scores, name=name), judge_feedback=judge_feedback,
            verdict=_current_verdict(rt, fn),
            discovery_grade=_current_discovery_grade(rt, fn),
            search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
            behavior_purity=previous_scores.get("behavior_purity"),
            behavior_specificity=previous_scores.get("behavior_specificity"),
            message=(
                "No matched development examples were available for diagnostic review. "
                if status == "diagnostic_review_unavailable" else
                "Diagnostic review is incomplete; repeat the same diagnostic request to "
                "resume its frozen cohort. "
                if status == "diagnostic_review_incomplete" else
                "Reviewed bounded development examples for diagnostic feedback only. "
            ) + "Adherence, confirmed support, and repair validation are unchanged.",
        )
    fresh_repair_result = result
    if retry_ids:
        from autofa.metrics.repair_validation import merge_processing_retry

        try:
            result = merge_processing_retry(scores, result, retry_ids)
        except ValueError as exc:
            return _failed_repair_review(rt, repair_context["plan_id"], result, exc)
        n_sample = result["n_judged"]

    has_judgment = result.get("backend") != "heuristic" and bool(
        result.get("n_completed", result.get("n_judged"))
    )
    purity = float(result["behavior_purity"]) if has_judgment else None
    specificity = result.get("behavior_specificity")
    specificity = (float(specificity) if has_judgment
                   and type(specificity) in (int, float) and 0 <= specificity <= 1
                   and math.isfinite(specificity) else None)
    from autofa.metrics.review_scores import apply_review_result

    apply_review_result(scores, result, judge_harness=judge_harness)
    scores.pop("review_schedule", None)
    if diagnostic_result is not None:
        from autofa.metrics.diagnostic_review import retain_current_diagnostic_feedback

        scores["diagnostic_review"] = retain_current_diagnostic_feedback(
            rt, fn, previous_scores, diagnostic_result)
        scores.pop("diagnostic_evidence_stale", None)
    elif ((scores.get("diagnostic_review") or {}).get("claim_hash") != result.get("claim_hash")):
        scores.pop("diagnostic_review", None)
        scores.pop("diagnostic_evidence_stale", None)
    if tracked_review:
        scores["behavior_review_phase"] = phase
        scores.pop("confirmation_availability", None)
    current_review_localisations = {
        str(item.get("_item_id") or ""): deepcopy(localisation)
        for item, localisation in zip(evidence_items, evidence_localisations, strict=True)
    }
    scores["behavior_review_localisations"] = {
        row["item_id"]: (current_review_localisations[row["item_id"]]
                         if row["item_id"] in current_review_localisations
                         else previous_review_localisations.get(row["item_id"]))
        for row in scores["behavior_verdicts"]
        if isinstance(row, dict) and isinstance(row.get("item_id"), str)
    }
    scores["behavior_n_fires"] = n_fires
    scores["behavior_n_sampled"] = n_sample
    scores["behavior_sample"] = sample_info
    scores["behavior_review_operation"] = {
        "origin_run_id": rt.run.run_id, "scope": "standard", "phase": phase,
        "review_key": review_key,
    }
    cal = result.get("claim_calibration")
    scores["claim_calibration"] = cal
    scores["item_calibrations"] = result.get("item_calibrations") or []
    cal_hint = result.get("calibration_hint")
    scores["calibration_hint"] = cal_hint
    if repair_context:
        scores["repair_validation"] = {
            "plan_id": repair_context["plan_id"],
            "sample_kind": ("processing_retry" if retry_ids else
                            "development" if tracked_review and phase == "development" else
                            "reused_review_evidence" if sample_info.get("reused_n") else
                            "new_repair_validation"),
            "excluded_item_ids": sorted(excluded_ids),
            "scope": ("Re-reviews recorded cases for scoring; not fresh repair validation."
                      if sample_info.get("reused_n") else
                      "Excludes recorded discovery and judge exposure by task; "
                      "unrecorded external reads are outside this scope."
                      if tracked_review else
                      "Excludes recorded review/repair cases; "
                      "other agent exposure is not tracked."),
        }
    else:
        scores.pop("repair_validation", None)
    completed_requested_review = None
    if preserve_review_authority:
        from autofa.metrics.review_assessments import restore_assessment

        completed_requested_review = assessment_summary(scores)
        restore_assessment(scores, previous_scores)
        purity = scores.get("behavior_purity")
        specificity = scores.get("behavior_specificity")
        cal = scores.get("claim_calibration")
        cal_hint = scores.get("calibration_hint")
    from autofa.metrics.task_support import build_task_support
    from autofa.session.family_coverage import cached_task_inventory

    reviewed_fn = copy(fn)
    reviewed_fn.scores = scores
    try:
        support_inventory = cached_task_inventory(rt)
    except (OSError, ValueError, TypeError):
        support_inventory = None
    support_scores = build_task_support(reviewed_fn, support_inventory)
    scores.update(support_scores)
    verdict = verdict_for_scores(scores)
    scores["badge"] = verdict["label"]
    scores["verdict"] = verdict
    scores["feedback"] = {
        "ppv_at_pi0": float(scores.get("ppv_at_pi0") or 0),
        "recall": float(scores.get("recall") or 0),
        "behavior_purity": purity,
        "behavior_specificity": specificity,
        "behavior_validation": behavior_validation(scores, include_details=False),
        "judge_prefilter": decision,
        "pct_caught": float(scores.get("pct_caught") or 0),
        "pi0": 0.30,
        "claim_calibration": cal,
    }
    scores["feedback"].update({
        key: value for key, value in scores.items()
        if key.startswith("task_recall") or key in {
            "n_matched_failed_tasks", "n_failed_tasks", "n_matched_failed_attempts",
            "n_matched_failed_attempts_without_task_identity", "search_panel",
            "n_confirmed_failed_tasks", "confirmed_task_support_status",
        }
    })
    scores["feedback"].update({key: value for key, value in support_scores.items()
                               if key != "task_support"})
    from autofa.metrics.improvements import classify_improvement, focus_feedback
    from autofa.session.analysis_skills import skills_prompt

    improvement = (previous_scores.get("improvement") or {}) if preserve_review_authority else (
        classify_improvement(
        fn=fn, items=evidence_items, verdicts=result.get("verdicts") or [],
        cfg=rt.config, skills_context=skills_prompt(rt.run),
    ))
    scores["improvement"] = improvement
    focus_hint = focus_feedback(improvement, getattr(rt.run, "analysis_focus", "model"))
    if repair_context and (rt.library.get(name) is not fn or fn.scores != previous_scores):
        return _failed_repair_review(
            rt, repair_context["plan_id"], result,
            ValueError("Detector or scores changed while its repair review was running"),
        )
    with cancellation_guard():
        fn.scores = scores
        coverage_comparison = _attach_repair_coverage(rt, fn)
        repair = None
        if record_repair_outcome and repair_context.get("plan_id"):
            try:
                repair = repair_workflow.record_validation(
                    rt.run.run_dir(), fn.to_dict(), fresh_repair_result,
                    eligible_items=max(n_eligible, sample_info.get("reused_n", 0)),
                    expected_plan_id=repair_context["plan_id"],
                )
                if repair is None:
                    raise ValueError("The repair plan is no longer active")
            except (ValueError, OSError) as exc:
                # Preserve the prior scores: an ordinary re-evaluation must not later
                # resurrect a judgment that failed repair provenance validation.
                fn.scores = previous_scores
                return _failed_repair_review(rt, repair_context["plan_id"], result, exc)
        try:
            if (review_key is not None and (result.get("n_judged", 0) > 0 or diagnostic_items)
                    and result.get("n_completed", 0) + result.get("n_uncertain", 0)
                    == result["n_judged"]
                    and not (expansion.get("added_item_ids")
                             and expansion.get("status") != "complete")
                    and (not diagnostic_items or (diagnostic_result is not None
                        and diagnostic_result.get("n_judged") == len(diagnostic_items)
                        and not diagnostic_result.get("n_errors")
                        and not diagnostic_result.get("n_review_incomplete")))
                    and not result.get("n_errors") and not result.get("n_review_incomplete")):
                from autofa.session.review_evidence import complete_review_allocation

                complete_review_allocation(
                    rt.run.run_dir(), name, review_key=review_key, phase=phase,
                )
            rt.persist_library()
        except OSError as exc:
            if not repair_context:
                raise
            fn.scores = previous_scores
            return _failed_repair_review(rt, repair_context["plan_id"], result, exc)
    from autofa.metrics.claim_calibration import summarize_judge_feedback
    from autofa.metrics.diagnostic_review import summarize_diagnostic_review
    from autofa.metrics.live_progress import refresh_display_progress

    refresh_display_progress(rt)
    checkpoint_discovery_stall(rt)

    judge_feedback = summarize_judge_feedback(
                scores, name=name, behavior_strictness=strictness_for_config(rt.config))
    if completed_requested_review is not None:
        completed = completed_requested_review["complete"]
        unfinished = not completed_assessment(scores)
        next_action = (review_next_action(rt, fn.to_dict(), evidence_n=evidence_n)
                       if repair_context else review_recovery_action(rt, fn.to_dict()))
        return ok(
            name=name, status=("review_completed_authority_retained" if completed
                               else "review_incomplete_authority_retained"),
            requested_review={"status": "completed" if completed else "incomplete",
                              "authoritative": False,
                              **completed_requested_review},
            authoritative_review=assessment_summary(scores),
            sample=scores.get("behavior_sample"), verdict=_current_verdict(rt, fn),
            discovery_grade=_current_discovery_grade(rt, fn), judge_feedback=judge_feedback,
            discovery_feedback=build_discovery_feedback(
                scores, judge_feedback=judge_feedback, behavior_strictness=behavior_strictness,
                target_grade=agent_target_grade_for_runtime(rt)),
            message=("The older cohort was reviewed and saved separately. " if completed else
                     "The older cohort still has unfinished processing; its attempt was saved "
                     "separately. ") + "The newer authoritative assessment and its measured "
                     "scores are retained." + (" It is unfinished; repeating this older request "
                                               "cannot finish it." if unfinished else ""),
            **({"next_action": next_action} if next_action else {}),
        )
    purity_hint = " ".join(judge_feedback["next_steps"]) or None
    # Keep factual outcome guidance distinct from the judge's editorial calibration.
    # Specificity describes detail; it must not override the selected family breadth.
    hint = purity_hint or cal_hint
    if cal_hint and purity_hint:
        hint = f"{purity_hint} {cal_hint}"
    if focus_hint:
        hint = f"{hint} {focus_hint}" if hint else focus_hint
    specificity_text = f"{specificity:.2f}" if specificity is not None else "unavailable"
    msg = (
        f"behavior_purity={purity:.2f} behavior_specificity={specificity_text} "
        f"(sampled {n_sample}/{n_fires} fires)"
        if has_judgment else
        (
            "Evidence review unresolved: " if result.get("n_uncertain") else "Unjudged: "
        ) + (result.get("rationale") or "no evidence-backed judgment")
    )
    if cal and cal.get("status") and cal.get("status") != "good":
        msg += f" · claim_calibration={cal.get('status')}"
    if sample_info["limited"]:
        msg += f" · Limited evidence: {sample_info['reason']}"
    if hint:
        msg += f". {hint}"
    elif (has_judgment and not sample_info["limited"] and result.get("judge_contract_version")
          and result.get("n_yes") == result.get("n_judged")):
        msg += " — full claim confirmed on the reviewed sample."
    from autofa.mcp.tools_library import _family_search_status
    from autofa.mcp.tools_repair import compact_plan

    with cancellation_guard():
        family_status = _family_search_status(rt)
    return ok(
        name=name,
        verdict=_current_verdict(rt, fn),
        discovery_grade=_current_discovery_grade(rt, fn),
        sample=sample_info,
        repair=compact_plan(repair),
        coverage_comparison=coverage_comparison,
        behavior_purity=purity,
        behavior_specificity=specificity,
        n_judged=result.get("n_judged"),
        n_yes=result.get("n_yes"),
        n_no=result.get("n_no"),
        n_uncertain=result.get("n_uncertain"),
        n_completed=result.get("n_completed"),
        n_errors=result.get("n_errors", 0),
        n_review_incomplete=result.get("n_review_incomplete", 0),
        evidence_version=result.get("evidence_version"),
        judge_contract_version=result.get("judge_contract_version"),
        judge_model=result.get("judge_model"),
        judge_harness=result.get("judge_harness", judge_harness),
        judge_effort=result.get("judge_effort"),
        judge_thinking=result.get("judge_thinking"),
        judge_thinking_budget_tokens=result.get("judge_thinking_budget_tokens"),
        judge_rubric_hash=result.get("judge_rubric_hash"),
        judge_policy_hash=result.get("judge_policy_hash"),
        judge_objective_strategy=result.get("judge_objective_strategy"),
        judge_behavior_strictness=result.get("judge_behavior_strictness"),
        claim_hash=result.get("claim_hash"),
        claim_contract=result.get("claim_contract"),
        claim_compilation=result.get("claim_compilation"),
        n_fires=n_fires,
        n_scanned=n_scanned,
        scan_limit=scan_limit,
        n_sampled=n_sample,
        rationale=result.get("rationale"),
        verdicts=result.get("verdicts"),
        diagnostic_review=summarize_diagnostic_review(scores, name=name),
        claim_calibration=cal,
        item_calibrations=scores["item_calibrations"],
        calibration_hint=cal_hint,
        specificity_rationale=result.get("specificity_rationale"),
        backend=result.get("backend"),
        improvement=improvement,
        focus_hint=focus_hint,
        feedback=scores["feedback"],
        judge_feedback=judge_feedback,
        search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
        family_discovery=family_status,
        discovery_feedback=build_discovery_feedback(
                scores, judge_feedback=judge_feedback,
                behavior_strictness=strictness_for_config(rt.config),
                target_grade=agent_target_grade_for_runtime(rt)),
        hint=hint,
        example_fail_ids=list(fn.example_fail_ids or [])[:8],
        message=msg,
    )


def llm_classify(
    text: str,
    question: str = "",
    prompt: str | None = None,
    options: list[str] | None = None,
    focus: str = "auto",
    artifact_images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify text via the configured endpoint (same path as LLM detectors at eval time).

    ``focus`` matches ``HubLLM.classify``: ``auto`` | ``head`` | ``tail`` | ``ends`` | ``full``.
    Pass full surfaces; do not pre-slice with ``text[-4000:]``. artifact_images accepts
    [{item_id, artifact_idx, page}] for visual classification through the same cached files.
    """
    rt = get_runtime()
    try:
        hub = rt.ensure_hub_llm()
    except Exception as e:  # noqa: BLE001
        return err(str(e))

    try:
        kwargs: dict[str, Any] = {"focus": focus}
        if artifact_images:
            from autofa import trace
            from autofa.mcp.tools_data import _load_item_raw

            images = []
            for selector in artifact_images:
                item = _load_item_raw(selector["item_id"])
                if item is None:
                    return err(f"item not found: {selector['item_id']}")
                images.append(trace.artifact_image(item, selector["artifact_idx"],
                                                   page=selector.get("page", 1),
                                                   **({"member_path": selector["member_path"]}
                                                      if selector.get("member_path") is not None
                                                      else {})))
            kwargs["images"] = images
        if prompt and options is None:
            out = hub.classify(prompt, text, **kwargs)
            return ok(label=str(out), focus=focus)
        q = prompt or question or "Classify the text."
        out = hub.classify_options(text, q, options=options, **kwargs)
        return ok(label=str(out), options=options or ["yes", "no"], focus=focus)
    except Exception as e:  # noqa: BLE001
        return err(str(e))


def _llm_adapter():
    """Configured OpenAI-compatible client passed into sandboxed detectors."""
    rt = get_runtime()
    try:
        return rt.ensure_hub_llm()
    except Exception:  # noqa: BLE001
        from autofa.core.sandbox import NullLLM

        return NullLLM()
