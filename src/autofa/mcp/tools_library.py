# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library MCP tools."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from autofa.behavior_policy import strictness_for_config
from autofa.config import library_root
from autofa.core.dependencies import build_bundle, review_source, validate_edges
from autofa.core.fields import get_field_policy
from autofa.core.helpers import HelperLibrary
from autofa.core.library import (
    FUNCTION_KINDS,
    FeatureFunction,
    FunctionKind,
    FunctionLibrary,
)
from autofa.core.sandbox import compile_detector
from autofa.defaults import setting_value
from autofa.mcp.runtime import err, get_runtime, ok
from autofa.metrics.association import pct_caught_from_scores
from autofa.metrics.discovery_feedback import build_discovery_feedback
from autofa.metrics.discovery_grades import grade_for_scores, grade_summary, grading_table
from autofa.metrics.pareto import pareto_front, rank_functions
from autofa.metrics.search_ranking import search_ranking, search_score
from autofa.metrics.verdicts import verdict_for_scores
from autofa.objectives import discovery_strategy
from autofa.session.discovery_target import agent_target_grade_for_runtime
from autofa.session.family_discovery import family_transaction
from autofa.session.modes_ledger import (
    VALID_STATUSES,
    get_mode,
    ledger_peers_for_novelty,
    read_modes,
    summarize_modes,
    upsert_mode,
)
from autofa.session.status_log import emit_runtime_status, read_status
from autofa.ui.activity import detector_activity_context
from autofa.util.cancellation import cancellation_guard, check_cancelled


def _run_dir() -> Path:
    return get_runtime().run.run_dir()


@contextmanager
def _library_mutation(rt: Any):
    """Keep mutations invisible to concurrent saves until their checkpoint succeeds."""
    with cancellation_guard(), rt._library_lock:
        functions_before = list(rt.library.functions)
        history_before = deepcopy(rt.library.review_history)
        published = False

        def persist() -> None:
            nonlocal published
            rt.persist_library()
            published = True

        try:
            yield persist
        except BaseException:
            # Preserve the original detector objects, including unrelated work.
            rt.library.functions[:] = functions_before
            rt.library.review_history = history_before
            if published:
                rt.persist_library()
            raise


def _committed_mode_update(rt: Any, fn: FeatureFunction, **kwargs) -> list[str]:
    """Mode display follows a committed library update; failures are advisory."""
    try:
        upsert_mode(
            rt.run.run_dir(), name=fn.name, description=fn.description,
            mechanism=fn.mechanism, detector=fn.to_dict(), **kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        return [f"Library change committed; failure-mode display unavailable: {exc}"]
    return []


def _current_function_scores(rt: Any, fn: FeatureFunction) -> dict[str, Any]:
    """Read quality under the saved review strategy, honoring the active Judge Policy."""
    from autofa.metrics.task_support import build_task_support, task_support_assessment
    from autofa.metrics.verdicts import current_behavior_scores

    scores = current_behavior_scores(
        {**(fn.scores or {}), **build_task_support(fn, review_history=rt.library.review_history)},
        source=review_source(fn), description=fn.description or "",
        mechanism=fn.mechanism or fn.description or fn.name,
        behavior_strictness=strictness_for_config(rt.config),
    )
    support = task_support_assessment(scores)
    # Keep the evaluation-cohort count paired with task_recall; verdicts resolve
    # all source-verified matched-task support independently from the receipt.
    support.pop("n_matched_failed_tasks")
    return {**scores, **support}


def _discovery_grades(rt: Any) -> dict[str, Any]:
    """Agent-only quality ladder; public verdicts and display ordering stay separate."""
    return grade_summary(
        [fn.to_dict() for fn in rt.library.functions],
        behavior_strictness=strictness_for_config(rt.config),
        review_history=rt.library.review_history,
    )


def _failure_modes_summary() -> dict[str, Any]:
    return summarize_modes(_run_dir())


def _detector_exploration() -> dict[str, Any]:
    from autofa.session.diversity import detector_exploration
    from autofa.session.polishing_guard import active_polishing

    rt = get_runtime()
    result = detector_exploration(
        [fn.to_dict() for fn in rt.library.functions],
        llm_available=(rt.llm_readiness or {}).get("available"),
    )
    if active_polishing(rt):
        result.update(suggested_family=None, phase="polishing", message=(
            "Polish existing detectors toward Good using their validation findings. "
            "Keep the supported implementation approach or repair it as the evidence requires; "
            "do not introduce new detectors or satisfy an implementation mix quota."
        ))
    return result


def _family_search_status(rt: Any) -> dict[str, Any] | None:
    """Surface due synthesis after evaluation/review without initializing standalone work."""
    from autofa.session.polishing_guard import active_polishing

    if phase := active_polishing(rt):
        return {"phase": "polishing", "round": rt.run.round, "synthesis_pending": False,
                "frozen_names": phase["frozen_names"],
                "next_action": "Polish existing detectors toward Good using validation feedback."}
    from autofa.session.family_discovery import (
        STATE_FILE,
        family_progress_summary,
        initialize_family_discovery,
    )

    if not (rt.run.run_dir() / STATE_FILE).exists():
        return None
    initialize_family_discovery(rt)
    return family_progress_summary(rt.run.run_dir())


def _record_active_mode(fn: FeatureFunction, *, reason: str = "submitted") -> None:
    upsert_mode(
        _run_dir(),
        name=fn.name,
        description=fn.description,
        mechanism=fn.mechanism,
        status="active",
        reason=reason,
        detector=fn.to_dict(),
    )


def report_status(message: str) -> dict[str, Any]:
    """Post a short natural-language status line to the live panel and status log.

    Call at milestones (after evidence check, before/after a drop, when pivoting).
    One sentence, present progressive or past tense — no essays.
    """
    msg = (message or "").strip()
    if not msg:
        return err("message required — e.g. message='Dropped X: purity 0.1 after two rewrites'")
    row = emit_runtime_status(msg, source="agent")
    return ok(
        status=row,
        message=f"Status posted: {msg}",
    )


def list_status_updates(limit: int = 30) -> dict[str, Any]:
    """List recent natural-language status updates for this run (agent + auto tool mirrors)."""
    rt = get_runtime()
    lim = max(1, min(int(limit or 30), 200))
    rows = read_status(rt.run.run_dir(), limit=lim)
    return ok(
        updates=rows,
        n=len(rows),
        path=str(rt.run.run_dir() / "status_log.jsonl"),
        message="Also see run status.md. Prefer report_status for new lines.",
    )

# Detector-centric openers that make failure cards useless to readers.
_DETECTOR_PROSE_RE = re.compile(
    r"^\s*(fires|detects|flags|matches|returns\s+true|labels|tags)\s+when\b",
    re.I,
)


def mechanism_prose_warning(
    mechanism: str, description: str = "", *, behavior_strictness: str = "causal",
) -> str | None:
    """Return a rewrite hint when mechanism is detector-centric rather than a why-narrative."""
    mech = (mechanism or "").strip()
    desc = (description or "").strip()
    if behavior_strictness != "causal":
        if not mech or _DETECTOR_PROSE_RE.match(mech) or len(mech) < 120:
            relevance = (" Explain its plausible connection to the failed attempt and account "
                         "for later recovery." if behavior_strictness == "contextual" else "")
            return (
                "Describe the observable behavior and the evidence distinguishing it from "
                "counterexamples, instead of just stating a detector trigger."
                + relevance + " Qualify hypotheses and missing evidence."
            )
        return None
    if not mech:
        return (
            "Missing mechanism — describe the observed decision, supported cause, and "
            "observed consequence in plain language, identifying any evidence limits. "
            "Keep one compact paragraph of at most six short sentences."
        )
    if _DETECTOR_PROSE_RE.match(mech):
        return (
            "Mechanism reads like a detector trigger ('Fires when…'). Rewrite as an actionable "
            "failure story: the observed decision, evidence for its cause, and the observed "
            "consequence. Keep description as a punchy title, not a trigger rule."
        )
    if desc and mech == desc and _DETECTOR_PROSE_RE.match(desc):
        return (
            "description and mechanism are the same detector rule. Split them: punchy failure "
            "title in description; a clear account of the mistake and its effect in mechanism."
        )
    # Prefer at least two clauses / a short causal stretch.
    sentences = [s for s in re.split(r"[.!?]+", mech) if s.strip()]
    if len(sentences) < 2 and len(mech) < 120:
        return (
            "Mechanism is too thin — explain the observed behavior, supported cause, and "
            "observed consequence; qualify hypotheses and missing evidence."
        )
    return None


def submit_function(  # noqa: C901
    name: str,
    source: str,
    description: str = "",
    mechanism: str = "",
    function_mechanism: str = "",
    kind: FunctionKind = "regex",
    llm_prompt: str | None = None,
    auto_evaluate: bool = True,
    depends_on: list[str] | None = None,
    explains: str | None = None,
    repair_plan_id: str | None = None,
    uses_helpers: dict[str, str] | None = None,
    evidence_prerequisites: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile + save a detector. By default immediately evaluates it for feedback metrics.

    ``description`` is a punchy failure-mode title (card headline), e.g.
    "Writes its own test, sees it pass, ships on it".
    ``mechanism`` is the report's "Why this happens" explanation. Use a plain-language
    paragraph: concrete action, what made it mistaken or worth investigating, and the
    supported effect. Usually use 3–6 short sentences, never more than six; use fewer when
    enough. Explain necessary technical terms so readers understand on the first reading.
    Include failure relevance or causal contribution as required by the selected validation
    policy, with explicit evidence limits.
    ``repair_plan_id`` links a split child to a previously recorded repair plan.
    ``function_mechanism`` is a fluent explanation of *what the detector code does*
    (which fields it reads, regex/LLM checks, when it returns true).
    ``kind`` is ``regex``, ``llm``, or ``python``. Hybrid code uses ``python``.
    LLM use is detected from code and dependencies regardless of this label. Strategy is
    inferred separately.
    ``depends_on`` lists peer library function names this detector may call via
    ``library.fire(name, item)`` (sources are snapshotted for portability).
    ``explains`` names a parent whose fire set this detector explains (WHY) —
    triggers within-parent stratified eval.
    ``uses_helpers`` maps local aliases to helper names or immutable revisions.
    Call them using ``helpers.call(alias, ...)``. Helpers are never scored.
    Claims must be true of the items this detector labels — after eval, check
    ``example_fail_ids`` and revise claim or detector until they align.
    Default submission evaluates and immediately judges new, changed, or stale
    detector claims with fires that survive the association pre-filter.
    Unchanged current reviews, including unresolved
    reviews, are retained; use an explicit repair plan for further review.
    Set ``auto_evaluate=False`` to save an edit without evaluation or judge calls.
    ``evidence_prerequisites`` optionally declares essential source material before
    expensive evaluation: [{"fact": "the exact premise needing source evidence",
    "item_id": "an inspected support ID", "field": "a retained content field",
    "pointer": "/optional/JSON/path"}]. Use actual source locations found with MCP,
    at most three support items and eight checks. Missing source pauses auto-evaluation;
    restore it or revise the declaration/claim and resubmit. Omit to retain declarations
    for an unchanged claim; [] clears them after an evidence-grounded claim reassessment.
    Field presence is not proof of the fact. This check does not declare other items
    or the entire corpus to lack evidence, and does not replace semantic review.
    If mechanism is omitted, description is used as a fallback (discouraged).
    """
    from autofa.core.detector_lint import evidence_budget_warnings, lint_detector_source
    from autofa.mcp.server import invocation_actor
    from autofa.session.contributions import ACTORS, record_revision
    from autofa.session.evidence_prerequisites import validate_prerequisites
    from autofa.session.family_discovery import family_submission_guard
    from autofa.session.polishing_guard import polishing_tool_guard

    check_cancelled()
    actor = invocation_actor()
    if actor not in ACTORS:
        return err(f"actor must be one of {sorted(ACTORS)}")
    if evidence_prerequisites is not None:
        try:
            evidence_prerequisites = validate_prerequisites(evidence_prerequisites)
        except ValueError as exc:
            return err(str(exc))
    kind = "regex" if kind == "" else kind
    if kind not in FUNCTION_KINDS:
        return err(
            f"Invalid function kind {kind!r}; expected 'regex', 'llm' or 'python'. "
            "Use 'python' for hybrid or structured-code detectors; strategy is inferred "
            "from source."
        )
    rt = get_runtime()
    if blocked := polishing_tool_guard(rt, "submit_function", {"name": name}):
        return blocked
    if family_block := family_submission_guard(
        rt, name, explains=explains, description=description, mechanism=mechanism,
    ):
        return err(family_block, family_workflow_pending=True)
    dep_names = [str(n).strip() for n in (depends_on or []) if str(n).strip()]
    explains_name = (explains or "").strip() or None
    if explains_name and explains_name not in dep_names:
        # Explaining a parent implies depending on it (or at least having it snapshotted).
        dep_names = list(dict.fromkeys([*dep_names, explains_name]))

    dep_sources: dict[str, str] = {}
    for dep in dep_names:
        if dep == name:
            return err("depends_on cannot include the function itself")
        peer = rt.library.get(dep)
        if peer is None:
            return err(f"depends_on unknown function: {dep}")
        dep_sources[dep] = peer.source
        # Nested deps: include their snapshotted sources too
        nested = (peer.meta or {}).get("deps") or {}
        if isinstance(nested, dict):
            for k, v in nested.items():
                if isinstance(k, str) and isinstance(v, str) and k not in dep_sources:
                    dep_sources[k] = v

    if explains_name and explains_name not in dep_sources:
        return err(f"explains unknown function: {explains_name}")

    try:
        bundle = build_bundle(rt.library, depends_on=dep_names, uses_helpers=uses_helpers)
        validate_edges(source, (bundle or {}).get("helpers", {}),
                       (bundle or {}).get("detectors", {}))
        compile_detector(source, bundle=bundle, policy=get_field_policy(rt.config))
    except Exception as e:  # noqa: BLE001
        return err(f"compile failed: {e}")

    lint_source = source
    if bundle:
        lint_source += "\n" + "\n".join(n["source"] for n in bundle["nodes"].values())
    ok_lint, lint_reason = lint_detector_source(lint_source)
    if not ok_lint:
        return err(lint_reason or "detector rejected as structural proxy")
    budget_warnings = evidence_budget_warnings(lint_source)

    for dep_src in dep_sources.values():
        ok_d, reason_d = lint_detector_source(dep_src)
        # Dependencies may themselves be coarse proxies (stratum); only the
        # composed function must add behavior. Do not re-reject parents here.
        _ = (ok_d, reason_d)

    desc = (description or "").strip()
    mech = (mechanism or "").strip() or desc
    how = (function_mechanism or "").strip()
    prose_warn = mechanism_prose_warning(
        mech, desc, behavior_strictness=strictness_for_config(rt.config),
    )
    from autofa.session.exploration import novelty_warning

    peers = [
        {"name": f.name, "description": f.description, "mechanism": f.mechanism}
        for f in rt.library.functions
        if f.name != name
    ]
    # Also compare against ledger modes (open/rejected/redundant) so dropped claims
    # are not reinvented under a new name.
    seen = {p["name"] for p in peers}
    for peer in ledger_peers_for_novelty(rt.run.run_dir(), exclude_name=name):
        if peer["name"] not in seen:
            peers.append(peer)
            seen.add(peer["name"])
    novelty_warn = novelty_warning(description=desc, mechanism=mech, existing=peers)
    from autofa.session.diversity import infer_strategy, strategy_monoculture_warning

    inferred = infer_strategy(source, kind=kind)
    # Prefer inferred strategy when kind is the legacy default.
    if kind in {"", "regex"} and inferred != "regex":
        kind = inferred if inferred in {"regex", "llm"} else kind
    mono_warn = strategy_monoculture_warning(
        [f.to_dict() for f in rt.library.functions if f.name != name]
        + [{"name": name, "source": source, "kind": kind}]
    )
    meta: dict[str, Any] = {"strategy": inferred}
    if dep_sources:
        meta["deps"] = dict(dep_sources)
    if bundle:
        meta["dependency_bundle"] = bundle
    if explains_name:
        meta["explains"] = explains_name
    fn = FeatureFunction(
        name=name,
        description=desc,
        source=source,
        mechanism=mech,
        function_mechanism=how,
        kind=kind,
        llm_prompt=llm_prompt,
        depends_on=dep_names,
        uses_helpers=(bundle or {}).get("helpers", {}),
        explains=explains_name,
        meta=meta,
    )
    existing = rt.library.get(name)
    if evidence_prerequisites is not None:
        fn.meta["evidence_prerequisites"] = evidence_prerequisites
    elif (existing and existing.description == desc and existing.mechanism == mech
          and "evidence_prerequisites" in (existing.meta or {})):
        fn.meta["evidence_prerequisites"] = deepcopy(existing.meta["evidence_prerequisites"])
    unchanged = existing and (
        all(getattr(existing, key) == getattr(fn, key)
            for key in ("source", "description", "mechanism", "llm_prompt",
                        "depends_on", "explains", "uses_helpers"))
        and existing.kind == fn.kind
        and (existing.meta or {}).get("deps") == meta.get("deps")
        and (existing.meta or {}).get("dependency_bundle") == bundle
    )
    from autofa.mcp.implementation_recovery import check_implementation_submission
    from autofa.session.repair_workflow import record_submission

    if blocked := check_implementation_submission(rt, fn.to_dict(), repair_plan_id):
        return blocked
    if unchanged and existing.scores:
        # Scores and judge verdicts belong to this exact detector/claim, not its name.
        fn.scores = dict(existing.scores)
        fn.failure_localisations = dict(existing.failure_localisations)
    # Keep prior function_mechanism if this submit omitted it.
    if not how and existing and (existing.function_mechanism or "").strip():
        fn.function_mechanism = existing.function_mechanism
    record_revision(fn, existing, actor=actor)
    emit_runtime_status(
        f"{'Applying repair to' if existing and not unchanged else 'Saving'} detector {name}",
        source="submit_function",
        context=detector_activity_context(
            name, "repair" if existing and not unchanged else "submit",
            scores=_current_function_scores(rt, existing) if existing else None,
            target_grade=agent_target_grade_for_runtime(rt),
        ),
    )
    # Mutation and persistence form one cancellation checkpoint. Until it
    # succeeds, another detector's save must not serialize a tentative entry.
    # Keep the live library and unrelated detector objects intact: replacing a
    # deep-copied library would discard concurrent work on other detectors.
    # The consuming family guard below can write the first related ledger;
    # cancellation after entering this checkpoint must not split its writes.
    with cancellation_guard(), rt._library_lock:
        history_before = deepcopy(rt.library.review_history)
        published = False

        def apply_submission(repairs):
            nonlocal published
            if family_block := family_submission_guard(
                rt, name, explains=explains, description=description, mechanism=mechanism,
                consume_attempt=True,
            ):
                raise ValueError(family_block)
            rt.library.upsert(fn)
            rt.persist_library(repairs=repairs)
            published = True

        def restore_submission():
            rt.library.functions[:] = [
                existing if saved is fn else saved for saved in rt.library.functions
                if saved is not fn or existing is not None
            ]
            rt.library.review_history = history_before
            if published:
                # A failed repair-ledger commit follows the library write.
                # Its lock is now released and still contains the old plan;
                # restore that library checkpoint before any unrelated save.
                rt.persist_library()

        try:
            repair = record_submission(
                rt.run.run_dir(), fn.to_dict(), repair_plan_id,
                current_function=existing.to_dict() if existing else None,
                apply=apply_submission,
            )
        except ValueError as exc:
            # Validation failed before publication, or persistence rejected the
            # staged state. Restore the live entry before exposing the failure.
            restore_submission()
            from autofa.mcp.tools_repair import repair_rejection_next_action

            return err(
                f"Repair plan rejected this submission: {exc}",
                next_action=repair_rejection_next_action(
                    rt, existing.to_dict() if existing else fn.to_dict(), str(exc),
                ),
            )
        except BaseException:
            restore_submission()
            raise

    from autofa.mcp.tools_eval import _current_verdict

    feedback: dict[str, Any] = {}
    if auto_evaluate:
        from autofa.mcp import tools_eval
        from autofa.metrics.claim_calibration import summarize_judge_feedback
        from autofa.metrics.verdicts import format_metrics_line

        emit_runtime_status(
            f"Scoring detector {name}", source="submit_function",
            context=detector_activity_context(name, "evaluate"),
        )
        ev = tools_eval.evaluate_functions(
            names=[name],
            within=explains_name,
        )
        if ev.get("ok"):
            judge_refresh = None
            evaluated_fn = rt.library.get(name)
            evaluated_scores = _current_function_scores(rt, evaluated_fn or fn)
            prefilter = evaluated_scores.get("judge_prefilter") or {}
            if prefilter.get("status") == "skip":
                judge_refresh = {
                    "refreshed": False, "skipped": True, "semantic_reviews": 0,
                    "status": "skipped_pre_filter", "reason": prefilter.get("reason"),
                }
                emit_runtime_status(
                    f"Skipping behavior judging for {name}: {prefilter.get('reason')}",
                    source="submit_function", context={"name": name},
                )
            # Judge fires from the selected bucket, including benchmark passes.
            # A current UNKNOWN or processing error is a recorded review, not an
            # invitation to repeat the same evidence until the answer changes.
            refresh_judge = (
                (float(evaluated_scores.get("n_flagged") or 0) > 0
                 or (evaluated_scores.get("sampled") is True
                     and evaluated_scores.get("evaluated") is True
                     and float(evaluated_scores.get("a") or 0) == 0))
                and summarize_judge_feedback(
                    evaluated_scores, name=name,
                    behavior_strictness=strictness_for_config(rt.config))["status"]
                in {"unjudged", "stale", "legacy"}
            )
            if refresh_judge:
                emit_runtime_status(
                    f"Checking adherence review requirements for {name}",
                    source="submit_function", context={"name": name},
                )
                try:
                    judged = tools_eval.judge_function(name, review_mode="automatic")
                except Exception as exc:  # noqa: BLE001
                    judged = {"ok": False, "error": str(exc)}
                current = rt.library.get(name)
                current_scores = (current.scores if current else None) or {}
                judge_feedback = summarize_judge_feedback(
                current_scores, name=name, behavior_strictness=strictness_for_config(rt.config))
                # Older judge backends may lack explicit outcome counts. Processing
                # failures never count as completed semantic reviews in either format.
                semantic_reviews = (
                    sum(int(current_scores.get(key) or 0) for key in (
                        "behavior_n_yes", "behavior_n_no", "behavior_n_uncertain",
                    )) if current_scores.get("judge_contract_version") else
                    int(current_scores.get("behavior_n_judged") or 0)
                    - int(current_scores.get("behavior_n_errors") or 0)
                )
                refreshed = bool(
                    judged.get("ok")
                    and current_scores.get("judge_backend") in {
                        "claude_agent_sdk", "claude_print", "codex_app_server",
                    }
                    and semantic_reviews > 0
                    and judge_feedback["status"] not in {"stale", "error", "unjudged"}
                )
                judge_refresh = {"refreshed": refreshed, "semantic_reviews": semantic_reviews,
                                 "status": judge_feedback["status"]}
                if judged.get("status") == "review_deferred":
                    judge_refresh.update(
                        skipped=True, deferred=True, status="review_deferred",
                        reason=judged["reason"], review_schedule=judged["review_schedule"],
                    )
                diagnostic_feedback = judge_feedback.get("diagnostic_feedback") or {}
                diagnostic_only = bool(
                    not semantic_reviews and diagnostic_feedback.get("sampled", 0)
                )
                if diagnostic_only:
                    judge_refresh["diagnostic_only"] = True
                    judge_refresh["status"] = "diagnostic_review_completed"
                if not refreshed and not judge_refresh.get("deferred"):
                    judge_refresh["error"] = (
                        judged.get("error") or judged.get("rationale")
                        or "No current evidence-backed judgment"
                    )
            fn2 = rt.library.get(name)
            scored_fn = fn2 or fn
            scores = _current_function_scores(rt, scored_fn)
            judge_feedback = summarize_judge_feedback(
                scores, name=name, behavior_strictness=strictness_for_config(rt.config))
            example_ids = list((fn2.example_fail_ids if fn2 else None) or [])[:5]
            verdict = verdict_for_scores(scores)
            feedback = dict(scores.get("feedback") or {
                "pct_caught": scores.get("pct_caught"),
                "recall": scores.get("recall"),
                "ppv_at_pi0": scores.get("ppv_at_pi0"),
                "precision": scores.get("raw_ppv") or scores.get("precision"),
                "behavior_purity": scores.get("behavior_purity"),
                "behavior_specificity": scores.get("behavior_specificity"),
                "pi0": 0.30,
            })
            feedback.update(_task_recall_feedback(scores))
            validation = feedback.get("behavior_validation")
            if (isinstance(validation, dict) and feedback.get("judge_prefilter")
                    and validation.get("judge_prefilter") == feedback["judge_prefilter"]):
                # The submission is a compact summary; retain the identical policy
                # once here. Full saved assessments remain available via feedback.
                feedback["behavior_validation"] = {
                    key: value for key, value in validation.items() if key != "judge_prefilter"
                }
            if judge_feedback.get("review_schedule"):
                feedback["review_schedule"] = judge_feedback["review_schedule"]
            evidence_hint = _evidence_check_hint(example_ids)
            # Prefer prose, then novelty, then metric hints.
            primary_warn = prose_warn or novelty_warn
            hint = _feedback_hint(
                scores,
                primary_warn,
                evidence_hint=evidence_hint,
                config=rt.config,
                target_grade=agent_target_grade_for_runtime(rt),
            )
            # Ledger after eval so detector snapshot includes scores / examples.
            _record_active_mode(fn2 or fn, reason="submitted+evaluated")
            label = verdict.get("label") or verdict.get("code") or "?"
            emit_runtime_status(
                f"Submitted {name} → {label}",
                source="submit_function",
                context={**detector_activity_context(name, "submit", finished=True),
                         "verdict": label},
            )
            msg = (
                f"Submitted + evaluated → {verdict.get('symbol')} {verdict.get('label')}. "
                "See verdict.reason and judge_feedback."
            )
            if judge_refresh is not None:
                if judge_refresh.get("deferred"):
                    msg += (
                        f" Automatic adherence review deferred: {judge_refresh['reason']} "
                        "Follow discovery_feedback.grade_gap and judge_feedback.next_steps "
                        "to address the actual blocking dimensions or evidence gaps. "
                        "The candidate is retained; adherence remains unmeasured."
                    )
                elif judge_refresh.get("skipped"):
                    msg += (
                        f" Behavior judging skipped: {judge_refresh['reason']} "
                        "Improve association and re-evaluate, or remove the detector. "
                        "This is not a behavioral purity verdict."
                    )
                else:
                    msg += (
                        " Selected development matches were reviewed separately; "
                        "inspect judge_feedback.diagnostic_feedback. "
                        "The main adherence score remains unmeasured."
                        if judge_refresh.get("diagnostic_only") else
                        " Judgment refreshed for the submitted detector/claim; "
                        f"{judge_feedback['label']}."
                        if judge_refresh["refreshed"] else
                        " Automatic judgment did not produce a current semantic review. "
                        "Inspect judge_feedback and get_function_repair for the next action."
                    )
                msg += " Full item assessments and calibration: get_function_feedback(name)."
            if example_ids:
                msg += " Inspect example_fail_ids with get_item or get_trajectory."
            if prose_warn:
                msg += f" PROSE: {prose_warn}"
            if novelty_warn:
                msg += f" NOVELTY: {novelty_warn}"
            if mono_warn:
                msg += f" STRATEGY: {mono_warn}"
            if budget_warnings:
                msg += " EVIDENCE: " + " ".join(budget_warnings)
            coverage_comparison = None
            if repair:
                # Reporting follows the completed mutation; a diagnostic failure must
                # not make a successful submission look like a failed operation.
                try:
                    from autofa.mcp.tools_repair import get_function_repair

                    coverage_comparison = get_function_repair(name).get("coverage_comparison")
                except Exception as exc:  # noqa: BLE001
                    coverage_comparison = {
                        "comparison_complete": False, "automatic_promotion": False,
                        "reason": f"Coverage comparison unavailable: {exc}",
                    }
            return ok(
                name=name,
                n_functions=len(rt.library.functions),
                evaluated=True,
                candidate_screen=(ev.get("candidate_screens") or {}).get(name),
                polishing_feedback=(ev.get("polishing_feedback") or {}).get(name),
                feedback=feedback,
                judge_refresh=judge_refresh,
                judge_feedback=judge_feedback,
                discovery_feedback=build_discovery_feedback(
                    scores, judge_feedback=judge_feedback,
                    behavior_strictness=strictness_for_config(rt.config),
                    target_grade=agent_target_grade_for_runtime(rt),
                ),
                search_score=search_score(fn2 or fn, rt.config, run_id=rt.run.run_id),
                family_discovery=_family_search_status(rt),
                repair_plan_id=repair.get("plan_id") if repair else None,
                coverage_comparison=coverage_comparison,
                verdict=verdict,
                discovery_grade=grade_for_scores(scores),
                hint=hint,
                example_fail_ids=example_ids,
                mechanism_warning=prose_warn,
                novelty_warning=novelty_warn,
                strategy_warning=mono_warn,
                evidence_warnings=budget_warnings,
                strategy=inferred,
                depends_on=dep_names,
                explains=explains_name,
                metrics_line=format_metrics_line(scores),
                failure_modes_summary=_failure_modes_summary(),
                message=msg,
            )
        screen_blocked = bool(ev.get("candidate_screen_blocked"))
        screen = (ev.get("candidate_screens") or {}).get(name)
        _record_active_mode(fn, reason=("awaiting evidence prerequisite check" if screen_blocked
                                        else "submitted (eval failed)"))
        emit_runtime_status(
            (f"Saved {name}; evidence prerequisite check: {screen.get('status')}"
             if screen_blocked and screen else f"Submitted {name} but evaluate failed"),
            source="submit_function",
            context={**detector_activity_context(name, "submit", finished=True),
                     "evaluated": False},
        )
        return ok(
            name=name,
            n_functions=len(rt.library.functions),
            evaluated=False,
            eval_error=ev.get("error"),
            polishing_feedback=(ev.get("polishing_feedback") or {}).get(name),
            candidate_screen=screen,
            candidate_screen_blocked=screen_blocked,
            next_action=screen.get("next_action") if screen else None,
            search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
            family_discovery=_family_search_status(rt),
            verdict=_current_verdict(rt, rt.library.get(name) or fn),
            discovery_grade=grade_for_scores(
                _current_function_scores(rt, rt.library.get(name) or fn)),
            mechanism_warning=prose_warn,
            novelty_warning=novelty_warn,
            strategy_warning=mono_warn,
            evidence_warnings=budget_warnings,
            strategy=inferred,
            failure_modes_summary=_failure_modes_summary(),
            message=(
                screen["feedback"] if screen and screen.get("status") ==
                "evidence_prerequisite_pending" else
                "Saved without corpus metrics. Inspect candidate_screen feedback; revise the "
                "claim or its evidence and recheck with screen_function, or defer_family. "
                "Repeating evaluation cannot turn this check into a pass."
                if screen_blocked else "Submitted but evaluate failed — call evaluate_functions."
            )
            + (f" PROSE: {prose_warn}" if prose_warn else "")
            + (f" NOVELTY: {novelty_warn}" if novelty_warn else "")
            + (f" STRATEGY: {mono_warn}" if mono_warn else ""),
        )

    verdict = _current_verdict(rt, rt.library.get(name) or fn)
    _record_active_mode(fn, reason="submitted without evaluation")
    emit_runtime_status(
        f"Submitted {name} → {verdict['label']} (evaluation not run)",
        source="submit_function",
        context={**detector_activity_context(name, "submit", finished=True),
                 "evaluated": False},
    )
    return ok(
        name=name,
        n_functions=len(rt.library.functions),
        evaluated=False,
        repair_plan_id=repair.get("plan_id") if repair else None,
        search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
        family_discovery=_family_search_status(rt),
        verdict=verdict,
        discovery_grade=grade_for_scores(_current_function_scores(rt, rt.library.get(name) or fn)),
        mechanism_warning=prose_warn,
        novelty_warning=novelty_warn,
        strategy_warning=mono_warn,
        evidence_warnings=budget_warnings,
        strategy=inferred,
        depends_on=dep_names,
        explains=explains_name,
        failure_modes_summary=_failure_modes_summary(),
        message="Saved without running evaluation; any current saved scores were retained. "
        "Call evaluate_functions for fresh metrics feedback."
        + (f" PROSE: {prose_warn}" if prose_warn else "")
        + (f" NOVELTY: {novelty_warn}" if novelty_warn else "")
        + (f" STRATEGY: {mono_warn}" if mono_warn else "")
        + (" EVIDENCE: " + " ".join(budget_warnings) if budget_warnings else ""),
    )


def remove_function(
    name: str,
    reason: str = "",
    status: str = "open",
) -> dict[str, Any]:
    """Remove one function from the working library; keep the failure mode on the ledger.

    ``reason`` is required so drops are auditable.
    ``status`` (ledger): ``open`` = mode still believed, improve later (default);
    ``rejected`` = mode claim was wrong; ``redundant`` = folded into another keeper.
    """
    rt = get_runtime()
    from autofa.session.polishing_guard import polishing_tool_guard

    if blocked := polishing_tool_guard(rt, "remove_function", {"name": name}):
        return blocked
    why = (reason or "").strip()
    if not why:
        return err(
            "refusing to remove without reason= "
            "(e.g. reason='WEAK: ppv@π₀=0.32 after rewrite attempts — too broad'). "
            "Do not drop solely because beh=— / behavior_purity missing (call judge_function). "
            "Pass status='open' if the failure mode is still real (default), "
            "status='rejected' if the claim was wrong, "
            "status='redundant' if folded into another mode."
        )
    st = (status or "open").strip().lower()
    if st not in VALID_STATUSES or st == "active":
        # Removing from library cannot leave status=active.
        st = "open" if st == "active" else (st if st in VALID_STATUSES else "open")
    from autofa.session.repair_versions import archive_scored_version
    from autofa.session.repair_workflow import record_retirement

    try:
        with _library_mutation(rt) as persist:
            fn = rt.library.get(name)
            if fn is None:
                return err(f"unknown function: {name}")
            archive_scored_version(rt.run.run_dir(), fn.to_dict())
            rt.library.remove(name)
            persist()
            # Failed retirement recording rolls the published library back while
            # its lock is still held. Archiving the old version is conservative.
            record_retirement(rt.run.run_dir(), name, why, st)
            diagnostics = _committed_mode_update(rt, fn, status=st, reason=why)
    except ValueError as exc:
        return err(f"Repair plan rejected retirement: {exc}")
    snap = {
        "name": fn.name,
        "description": fn.description,
        "mechanism": fn.mechanism,
        "function_mechanism": fn.function_mechanism,
        "source": fn.source,
        "scores": {
            k: (fn.scores or {}).get(k)
            for k in (
                "ppv_at_pi0",
                "recall",
                "raw_ppv",
                "pct_caught",
                "behavior_purity",
                "behavior_specificity",
                "judge_backend",
            )
        },
        "example_fail_ids": list(fn.example_fail_ids or [])[:10],
        "status": st,
    }
    try:
        emit_runtime_status(
            f"Dropped {name} ({st}): {why}",
            source="remove_function",
            context={"name": name, "status": st, "reason": why},
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics.append(f"Library change committed; status display unavailable: {exc}")
    return ok(
        removed=name,
        reason=why,
        status=st,
        removed_snapshot=snap,
        diagnostics=diagnostics,
        n_functions=len(rt.library.functions),
        failure_modes_summary=_failure_modes_summary(),
        message=(
            f"Removed {name} from library; ledger status={st}. {why} "
            "Call list_failure_modes / get_failure_mode to revisit; "
            "restore_failure_mode to bring the detector back for a rewrite."
        ),
    )


def clear_library(confirm: bool = False) -> dict[str, Any]:
    """Clear all functions from the working library. Requires confirm=true."""
    if not confirm:
        return err("refusing to clear — pass confirm=true")
    rt = get_runtime()
    from autofa.session.polishing_guard import polishing_tool_guard

    if blocked := polishing_tool_guard(rt, "clear_library", {}):
        return blocked
    with _library_mutation(rt) as persist:
        removed = list(rt.library.functions)
        rt.library.functions[:] = []
        persist()
        diagnostics = []
        for fn in removed:
            diagnostics.extend(_committed_mode_update(
                rt, fn, status="rejected", reason="user clear_library",
            ))
    return ok(
        cleared=len(removed),
        removed=[fn.name for fn in removed],
        n_functions=0,
        diagnostics=diagnostics,
        failure_modes_summary=_failure_modes_summary(),
    )


def list_library_functions() -> dict[str, Any]:
    from autofa.metrics.verdicts import (
        behavior_validation,
        is_sampled,
        resolved_behavior_purity,
        resolved_behavior_specificity,
    )

    rt = get_runtime()
    rows = []
    n_sampled = 0
    for fn in rt.library.functions:
        s = _current_function_scores(rt, fn)
        sampled = is_sampled(s)
        n_sampled += int(sampled)
        scores = {
            "pct_caught": pct_caught_from_scores(s),
            "recall": s.get("recall"),
            **_task_recall_feedback(s),
            "ppv_at_pi0": s.get("ppv_at_pi0"),
            "fpr": s.get("fpr"),
            "behavior_purity": resolved_behavior_purity(s),
            "behavior_specificity": resolved_behavior_specificity(s),
            "behavior_validation": behavior_validation(s, include_details=False),
            "pi0": 0.30,
            "sampled": sampled,
            "recall_ci": [s.get("recall_ci_low"), s.get("recall_ci_high")],
            "fpr_ci": [s.get("fpr_ci_low"), s.get("fpr_ci_high")],
            "fisher_p": s.get("fisher_p"),
        }
        if sampled:
            scores.update(
                {
                    "sample_n": s.get("sample_n"),
                    "sample_frame_n": s.get("sample_frame_n"),
                    "raw_ppv_ci": [s.get("raw_ppv_ci_low"), s.get("raw_ppv_ci_high")],
                }
            )
        rows.append(
            {
                "name": fn.name,
                "kind": fn.kind,
                "mechanism": fn.mechanism,
                "function_mechanism": fn.function_mechanism,
                "description": fn.description,
                "depends_on": fn.depends_on,
                "uses_helpers": fn.uses_helpers,
                "scores": scores,
                "verdict": verdict_for_scores(s),
                "discovery_grade": grade_for_scores(s),
            }
        )
    return ok(
        functions=rows,
        discovery_grades=_discovery_grades(rt),
        discovery_grading_table=grading_table(),
        search_ranking=search_ranking(rt.library.functions, rt.config, run_id=rt.run.run_id),
        family_discovery=_family_search_status(rt),
        detector_exploration=_detector_exploration(),
        n_helpers=len(rt.helpers.aliases),
        helpers_hint="Use list_helpers/get_helper for reusable unscored functions.",
        model_id=rt.library.model_id,
        benchmark=rt.library.benchmark,
        failure_modes_summary=_failure_modes_summary(),
        sampled_note=_sampled_scoring_note(n_sampled),
    )


def _reference_review_citations(
    verdicts: list[dict[str, Any]], *, pointer: str,
) -> tuple[list[dict[str, Any]], int]:
    """Keep one exact copy of repeated large citations in a presentation-only review."""
    result = deepcopy(verdicts)
    canonical: dict[str, str] = {}
    references = 0

    def citations(rows: list, path: str) -> None:
        nonlocal references
        for index, citation in enumerate(rows):
            if not isinstance(citation, dict):
                continue
            # Compare the complete object, never just citation_id or quoted text:
            # offsets, source identity and completeness are evidence too.
            value = json.dumps(citation, sort_keys=True, ensure_ascii=True)
            if len(value) < 512:
                continue
            if value in canonical:
                rows[index] = {"$ref": canonical[value]}
                references += 1
            else:
                canonical[value] = f"{path}/{index}"

    def review(row: dict, path: str) -> None:
        citations(row.get("citations") or [], f"{path}/citations")
        for index, assessment in enumerate(row.get("claim_assessments") or []):
            if isinstance(assessment, dict):
                citations(assessment.get("citations") or [],
                          f"{path}/claim_assessments/{index}/citations")
        for key in ("initial_review", "verification_review"):
            if isinstance(row.get(key), dict):
                review(row[key], f"{path}/{key}")

    for index, row in enumerate(result):
        if isinstance(row, dict):
            review(row, f"{pointer}/{index}")
    return result, references


def _citation_reference_note(count: int) -> dict[str, Any]:
    return {
        "count": count, "format": "JSON Pointer", "root": "response",
        "note": "Repeated large citations use $ref to an exact citation in this response. "
                "Resolve pointers from the full response root, including when reading a "
                "selected section. All review text is retained; saved library and judge "
                "evidence are unchanged. full_result retrieves the original response.",
    }


def reference_review_response(
    response: dict[str, Any], *, verdict_path: tuple[str, ...],
    extra_verdict_paths: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    """Return lossless citation references for an MCP presentation, never saved scores.

    Only the supplied review lists are transformed. The immutable original response
    remains available even if the detector is later changed or removed.
    """
    from autofa.mcp.tool_results import save_tool_result

    updates = []
    count = 0
    for path in dict.fromkeys((verdict_path, *extra_verdict_paths)):
        if not path:
            continue
        rows: Any = response
        for key in path:
            if not isinstance(rows, dict) or key not in rows:
                rows = None
                break
            rows = rows[key]
        if not isinstance(rows, list):
            continue
        pointer = "#/" + "/".join(key.replace("~", "~0").replace("/", "~1") for key in path)
        referenced, references = _reference_review_citations(rows, pointer=pointer)
        if references:
            updates.append((path, referenced))
            count += references
    if not count:
        return response
    try:
        saved = save_tool_result(response)
    except (OSError, RuntimeError, TypeError, ValueError):
        # Keep the original full response if the immutable copy cannot be saved.
        return response
    result = dict(response)
    for path, rows in updates:
        parent = result
        for key in path[:-1]:
            parent[key] = dict(parent[key])
            parent = parent[key]
        parent[path[-1]] = rows
    result["citation_references"] = _citation_reference_note(count)
    result["full_result"] = {
        **saved, "read_with": f"read_tool_result(result_id={saved['result_id']!r}, offset=0)",
    }
    return result


def get_library_function(name: str, item_id: str | None = None) -> dict[str, Any]:
    """Read a detector with up to five localisations, or one selected by item_id.

    All passage records remain in library.json; bounded tool replies keep source
    inspection usable for detectors that fire on thousands of examples.
    """
    from dataclasses import replace
    from itertools import islice

    rt = get_runtime()
    fn = rt.library.get(name)
    if fn is None:
        return err(f"unknown function: {name}")
    records = fn.failure_localisations
    selected = [item_id] if item_id is not None else list(islice(records, 5))
    preview = replace(fn, failure_localisations={key: records[key]
                                                 for key in selected if key in records})
    from autofa.metrics.verdicts import (
        behavior_validation,
        judge_prefilter_skipped,
        resolved_behavior_purity,
        resolved_behavior_specificity,
    )

    current = _current_function_scores(rt, fn)
    if judge_prefilter_skipped(current):
        # Keep the saved evidence intact; make current vs historical metrics
        # explicit in this inspection response as in feedback/list responses.
        current["historical_behavior_scores"] = {
            key: current.get(key) for key in ("behavior_purity", "behavior_specificity")
        }
        current.update(
            behavior_purity=resolved_behavior_purity(current),
            behavior_specificity=resolved_behavior_specificity(current),
            behavior_validation=behavior_validation(current, include_details=False),
        )
    verdict = verdict_for_scores(current)
    # Labels are a view of current measurements; do not expose a saved badge after
    # its claim or review policy became stale, or rewrite the saved evidence.
    current.update(badge=verdict["label"], verdict=verdict)
    # The identity receipt can contain hundreds of matches. Return its validated
    # count; individual evidence remains available through the existing selectors.
    preview_scores = {key: value for key, value in current.items() if key != "task_support"}
    preview = replace(preview, scores=preview_scores)
    response = ok(function=preview.to_dict(), verdict=verdict, n_localisations=len(records),
                  discovery_grade=grade_for_scores(current),
                  discovery_feedback=build_discovery_feedback(
            current, behavior_strictness=strictness_for_config(rt.config),
            target_grade=agent_target_grade_for_runtime(rt)),
                  search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
                  family_discovery=_family_search_status(rt),
                  localisation_hint="Pass item_id to read that example's localisation.")
    return reference_review_response(
        response, verdict_path=("function", "scores", "behavior_verdicts"),
    )


def get_function_feedback(
    name: str, offset: int = 0, limit: int = 10, item_id: str | None = None,
) -> dict[str, Any]:
    """Page saved judgments and polishing controls; optionally select one item ID.

    Supplemental outcome diagnostics appear separately in diagnostic_review and
    do not change the adherence sample or confirmed-support counts.
    """
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
        return err("offset must be nonnegative and limit must be between 1 and 50")
    rt = get_runtime()
    fn = rt.library.get(name)
    if fn is None:
        return err(f"unknown function: {name}")
    from autofa.metrics.verdicts import (
        behavior_validation,
        is_sampled,
        resolved_behavior_purity,
    )

    scores = _current_function_scores(rt, fn)
    cal = scores.get("claim_calibration")
    from autofa.metrics.claim_calibration import summarize_judge_feedback

    rows = list(scores.get("behavior_verdicts") or [])
    if item_id is not None:
        rows = [row for row in rows if str(row.get("item_id")) == item_id]
    page = rows[offset:offset + limit]
    page_ids = {str(row.get("item_id")) for row in page}
    validation = behavior_validation(scores, include_details=False)
    validation.update(claim_contract=scores.get("claim_contract"), verdicts=page)
    feedback = {
        "pct_caught": pct_caught_from_scores(scores),
        "recall": scores.get("recall"),
        **_task_recall_feedback(scores),
        "ppv_at_pi0": scores.get("ppv_at_pi0"),
        "fpr": scores.get("fpr"),
        "behavior_purity": resolved_behavior_purity(scores),
        "behavior_validation": validation,
        "pi0": 0.30,
        "claim_calibration": cal,
        "sampled": is_sampled(scores),
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
    if feedback["sampled"]:
        feedback.update(
            {
                "sample_n": scores.get("sample_n"),
                "sample_frame_n": scores.get("sample_frame_n"),
                "raw_ppv_ci": [
                    scores.get("raw_ppv_ci_low"),
                    scores.get("raw_ppv_ci_high"),
                ],
                "note": _sampled_scoring_note(1),
            }
        )
    evidence_hint = _evidence_check_hint(fn.example_fail_ids or [])
    from autofa.session.polishing_guard import active_polishing
    from autofa.session.polishing_progress import public_limitation_findings

    response = ok(
        name=name,
        recovery_findings=public_limitation_findings(
            fn, offset=offset, limit=limit, item_id=item_id) if active_polishing(rt) else [],
        feedback=feedback,
        verdict=verdict_for_scores(scores),
        discovery_grade=grade_for_scores(scores),
        mechanism=fn.mechanism,
        description=fn.description,
        example_fail_ids=fn.example_fail_ids,
        claim_calibration=cal,
        item_calibrations=[row for row in scores.get("item_calibrations") or []
                           if str(row.get("item_id")) in page_ids],
        pagination={"offset": offset, "limit": limit, "total": len(rows),
                    "next_offset": offset + limit if offset + limit < len(rows) else None},
        calibration_hint=scores.get("calibration_hint"),
        judge_feedback=summarize_judge_feedback(
                scores, name=name, behavior_strictness=strictness_for_config(rt.config)),
        discovery_feedback=build_discovery_feedback(
            scores, behavior_strictness=strictness_for_config(rt.config),
            target_grade=agent_target_grade_for_runtime(rt)),
        search_score=search_score(fn, rt.config, run_id=rt.run.run_id),
        family_discovery=_family_search_status(rt),
        hint=_feedback_hint(
            scores,
            evidence_hint=evidence_hint,
            config=rt.config,
            target_grade=agent_target_grade_for_runtime(rt),
        ),
    )
    diagnostic = scores.get("diagnostic_review")
    if isinstance(diagnostic, dict):
        from autofa.metrics.diagnostic_review import diagnostic_scores

        diagnostic_rows = [
            row for row in diagnostic.get("verdicts") or [] if isinstance(row, dict)
            and (item_id is None or str(row.get("item_id")) == item_id)
        ]
        # Keep complete per-item assessments accessible with the same pagination.
        # Provider sessions and internal locators remain in the saved library;
        # they are not repeated in every feedback response.
        response["diagnostic_review"] = {
            key: value for key, value in diagnostic.items()
            if key not in {"verdicts", "sessions", "claim_compilation",
                           "review_localisations", "item_calibrations"}
        } | {
            "verdicts": diagnostic_rows[offset:offset + limit],
            "current": diagnostic_scores(scores) is not None,
            "excluded_from_adherence": True,
            "excluded_from_confirmed_support": True,
            "pagination": {
                "offset": offset, "limit": limit, "total": len(diagnostic_rows),
                "next_offset": (offset + limit if offset + limit < len(diagnostic_rows) else None),
            },
        }
    from autofa.session.polishing_feedback import polishing_feedback

    polishing = polishing_feedback(rt, fn, offset=offset, limit=limit, item_id=item_id)
    if polishing is not None:
        response["polishing_feedback"] = polishing
    return reference_review_response(
        response, verdict_path=("feedback", "behavior_validation", "verdicts"),
        extra_verdict_paths=(("diagnostic_review", "verdicts"),),
    )


def library_save(to_catalog: bool = True, path: str | None = None) -> dict[str, Any]:
    """Persist available results without running detectors or judge reviews."""
    rt = get_runtime()
    with cancellation_guard(), rt._library_lock:
        working = rt.persist_library()
        saved = [str(working)]
        if path:
            p = Path(path).expanduser()
            rt.library.save(p)
            saved.append(str(p))
        if to_catalog:
            from autofa.data.library_provenance import save_catalog_library

            cat = rt.library.catalog_path(rt.config.resolved_library_root() or library_root())
            save_catalog_library(rt.library, cat, rt.run)
            saved.append(str(cat))
    return ok(saved=saved)


def library_load(path: str) -> dict[str, Any]:
    """Merge a portable detector library or a standalone unscored helpers.json catalog."""
    rt = get_runtime()
    from autofa.session.polishing_guard import polishing_tool_guard

    if blocked := polishing_tool_guard(rt, "library_load", {}):
        return blocked
    p = Path(path).expanduser()
    if not p.exists():
        return err(f"not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "aliases" in data and "revisions" in data and "functions" not in data:
        loaded_helpers = HelperLibrary.from_dict(data)
        with cancellation_guard(), rt._library_lock:
            staged = deepcopy(rt.library)
            staged.helpers.merge(loaded_helpers)
            rt.persist_library(library=staged)
        return ok(n_helpers=len(rt.helpers.aliases), loaded_from=str(p))
    loaded = FunctionLibrary.from_dict(data)
    from autofa.config import runs_root
    from autofa.session.contributions import merge_records
    from autofa.session.review_evidence import import_seed_review_history, merge_histories

    # Importing a library must preserve exposure from older versions and removed
    # detectors, just like Continue. Read any available seed ledgers without
    # mutating their source run, which may still be active.
    import_seed_review_history(loaded, p, runs_root=runs_root())
    with cancellation_guard(), rt._library_lock:
        staged = deepcopy(rt.library)
        staged.review_history = merge_histories(staged.review_history, loaded.review_history)
        staged.contributions = merge_records(staged.contributions, loaded.contributions)
        # Snapshot the live library only after acquiring its lock: another
        # submission may have committed while the imported file was read.
        for fn in loaded.functions:
            staged.upsert(fn)
        staged.helpers.merge(loaded.helpers)
        rt.persist_library(library=staged)
    return ok(n_functions=len(rt.library.functions), loaded_from=str(p))


def _pareto_status() -> dict[str, Any]:
    """Internal driver status using the configured Good stopping minimum."""
    from autofa.metrics.completeness import comparable_evaluations
    from autofa.metrics.library_quality_coverage import library_quality_coverage_for_runtime
    from autofa.metrics.progress import latest_frontier_stats
    from autofa.session.discovery_stall import observe_discovery_stall
    from autofa.session.discovery_target import (
        discovery_good_target_for_runtime,
        failure_mode_target_for_runtime,
    )
    from autofa.session.task_progress import final_validation_handoff, observe_task_progress

    rt = get_runtime()
    final_target = failure_mode_target_for_runtime(rt)
    target = discovery_good_target_for_runtime(rt)
    history = comparable_evaluations(rt.scores_history)
    progress = observe_task_progress(rt)
    family = _family_search_status(rt)
    synthesis_pending = bool((family or {}).get("synthesis_pending"))
    progress_blocked = progress["progress_blocked"] or synthesis_pending
    from autofa.behavior_policy import strictness_for_config
    from autofa.reports.ordering import rank_display_functions

    functions = [f.to_dict() for f in rt.library.functions]
    ranked = rank_functions(functions, top_x=len(functions), config=rt.config)
    validated_top = [row for row in ranked if row["ranking"]["eligible"]][:rt.run.top_x]
    top = rank_display_functions(
        functions, behavior_strictness=strictness_for_config(rt.config),
        review_history=rt.library.review_history,
    )[:rt.run.top_x]
    candidates = [{"name": row["name"], **row["ranking"]}
                  for row in ranked if not row["ranking"]["eligible"]]
    patience = 6
    try:
        patience = max(1, int(setting_value(rt.config, "discovery_plateau_patience")))
    except Exception:  # noqa: BLE001
        patience = 6

    stats = latest_frontier_stats(history)
    current_points = (history[-1].get("points") or []) if history else []
    previous_points = history[-2].get("points") if len(history) >= 2 else []
    best_fv = float(stats["best_frontier_value"])
    since = progress["experiments_since_improve"]
    # Numerical frontier improvements remain diagnostics; plateau and progress
    # follow current quality-confirmed coverage across distinct failed tasks.
    plateau = since >= patience and progress["completed_attempts"] >= 2
    wait_for_plateau = bool(setting_value(rt.config, "discovery_wait_for_plateau"))
    regular_round_boundary = (not wait_for_plateau
                              and rt.run.round >= rt.run.max_rounds)
    progress_boundary = plateau or (
        rt.run.round >= rt.run.max_rounds and since >= max(2, patience // 2)
    )
    if not wait_for_plateau:
        # Off ends exploration at the planned round boundary once the real Good
        # minimum is met. Unrelated candidates do not prolong discovery; the
        # driver's existing finalization still scores the saved library fully.
        goal_met = target["minimum_met"] and regular_round_boundary
        validation_ready = regular_round_boundary
    else:
        # Optional plateau policy waits for supported coverage to stabilize.
        goal_met = (target["minimum_met"] and not progress_blocked
                    and progress_boundary)
        validation_ready = not synthesis_pending and progress_boundary
    from autofa.session.polishing_guard import active_polishing

    phase = active_polishing(rt)
    if phase and not phase.get("transition_pending"):
        # Sampled improvements do not close the required polishing exchange.
        # The completed previous-round receipt may still hand off normally.
        goal_met = validation_ready = False
    validation = final_validation_handoff(
        rt, progress, final_target, ready=validation_ready,
        allow_pending=regular_round_boundary,
        discovery_target=target if regular_round_boundary else None,
    )
    from autofa.session.campaign_policy import campaign_status

    campaign = campaign_status(rt)
    if campaign["active"]:
        # Quality is a milestone in improve_until. The authorized horizon owns
        # termination independently of checkpoint cadence or legacy round hints.
        goal_met = bool(campaign["finalize_due"] and campaign["quality_target_met"])
        validation["ready_for_final_validation"] = bool(campaign["finalize_due"])
        validation["final_validation_guidance"] = (
            "The campaign boundary is reached. Return saved findings for native finalization."
            if campaign["finalize_due"] else
            "Continue improving confirmed failed-task coverage within the campaign horizon."
        )
    delta = stats.get("frontier_delta")
    improved = progress["improved"]

    from autofa.session.diversity import library_coverage

    coverage = library_coverage([f.to_dict() for f in rt.library.functions])
    from autofa.metrics.verdicts import is_sampled

    sampled_names = [f.name for f in rt.library.functions if is_sampled(f.scores or {})]
    return ok(
        goal_met=goal_met,
        campaign=campaign,
        quality_stall=observe_discovery_stall(rt),
        **target,
        final_validation_target=final_target,
        progress_blocked=progress_blocked,
        discovery_wait_for_plateau=wait_for_plateau,
        improved=improved,
        progress_objective="quality_confirmed_distinct_task_coverage",
        task_coverage=progress["task_coverage"],
        library_quality_coverage=library_quality_coverage_for_runtime(rt, cached_only=True),
        confirmed_task_gain=progress["confirmed_task_gain"],
        best_confirmed_failed_tasks=progress["best_confirmed_failed_tasks"],
        coverage_baseline_initialized=progress["baseline_initialized"],
        **validation,
        round=rt.run.round,
        max_rounds=rt.run.max_rounds,
        current_front=pareto_front([tuple(p) for p in current_points]),
        previous_front=pareto_front([tuple(p) for p in previous_points]) if previous_points else [],
        top_x=[t["name"] for t in top],
        top_x_order="public_quality_grade_then_coverage",
        validated_top_x=[t["name"] for t in validated_top],
        validated_top_x_note=("Scalar objective ranking of candidates meeting the stricter "
                              "validation eligibility; display order does not grant eligibility."),
        research_candidates=candidates,
        search_ranking=search_ranking(rt.library.functions, rt.config, run_id=rt.run.run_id),
        family_discovery=family,
        frontier_value=float(stats["frontier_value"]),
        best_frontier_value=best_fv,
        frontier_delta=delta,
        hypervolume=float(stats["hypervolume"]),
        n_functions=int(stats["n_functions"]) or len(rt.library.functions),
        experiments_since_improve=since,
        patience=patience,
        strategy_coverage=coverage,
        detector_exploration=_detector_exploration(),
        diversity_warnings=coverage.get("warnings") or [],
        sampled_functions=sampled_names,
        sampled_note=_sampled_scoring_note(len(sampled_names)),
        failure_modes_summary=_failure_modes_summary(),
    )


def pareto_status() -> dict[str, Any]:
    """Show this round's exploration target and the authoritative stop decision."""
    from autofa.session.discovery_budget import discovery_budget_feedback
    from autofa.session.discovery_target import (
        agent_failure_mode_target,
        discovery_grade_assessment,
        discovery_grade_feedback,
    )
    from autofa.session.polishing_guard import active_polishing
    from autofa.session.round_feedback import round_bulletin

    internal = _pareto_status()
    rt = get_runtime()
    target = agent_failure_mode_target(
        internal, round_idx=rt.run.round, max_rounds=rt.run.max_rounds,
        target_grade="GOOD" if active_polishing(rt) else None,
    )
    grades = _discovery_grades(rt)
    budget = discovery_budget_feedback(
        rt.run.round, rt.run.max_rounds,
        campaign_policy=getattr(rt.run, "campaign_policy", None),
    )
    return {
        **{key: value for key, value in internal.items() if key != "final_validation_target"},
        **target,
        "goal_met": internal["goal_met"],
        "discovery_grades": grades,
        "discovery_grading_table": grading_table(),
        "budget_feedback": budget,
        "agent_grade_feedback": discovery_grade_feedback(
            rt, target=internal, task_coverage=internal.get("task_coverage"),
        ) if active_polishing(rt) else discovery_grade_assessment(
            grades, getattr(rt, "_previous_discovery_grades", None),
            minimum=rt.run.min_failure_modes, budget_feedback=budget,
            target=target, pending=target["pending_failure_modes"],
            task_coverage=internal.get("task_coverage"),
            round_feedback=round_bulletin(
                rt, grades=grades, task_coverage=internal.get("task_coverage"),
            ),
        ),
    }


def list_failure_modes(status: str | None = None) -> dict[str, Any]:
    """List discovered failure modes from the ledger (including dropped detectors).

    Always call this before proposing new modes. Optional ``status`` filter:
    active | open | rejected | redundant.
    """
    from autofa.metrics.verdicts import (
        behavior_validation,
        current_behavior_scores,
        resolved_behavior_purity,
        resolved_behavior_specificity,
    )

    rt = get_runtime()
    rows = read_modes(rt.run.run_dir())
    want = (status or "").strip().lower() or None
    if want and want not in VALID_STATUSES:
        return err(f"unknown status={status!r}; use active|open|rejected|redundant")
    out = []
    for row in rows:
        st = str(row.get("status") or "")
        if want and st != want:
            continue
        det = row.get("detector") if isinstance(row.get("detector"), dict) else {}
        description = det.get("description") or row.get("description") or ""
        scores = current_behavior_scores(
            det.get("scores"), source=review_source(det), description=description,
            mechanism=det.get("mechanism") or row.get("mechanism")
            or description or row.get("name") or "",
            objective_strategy=discovery_strategy(rt.config),
            behavior_strictness=strictness_for_config(rt.config),
        )
        out.append(
            {
                "name": row.get("name"),
                "status": st,
                "description": row.get("description") or "",
                "mechanism": (row.get("mechanism") or "")[:400],
                "reason": row.get("reason") or "",
                "scores": {
                    "ppv_at_pi0": scores.get("ppv_at_pi0"),
                    "recall": scores.get("recall"),
                    "behavior_purity": resolved_behavior_purity(scores),
                    "behavior_specificity": resolved_behavior_specificity(scores),
                    "behavior_validation": behavior_validation(scores, include_details=False),
                    "pct_caught": scores.get("pct_caught"),
                },
                "ts": row.get("ts"),
            }
        )
    summary = summarize_modes(rt.run.run_dir())
    return ok(
        modes=out,
        failure_modes_summary=summary,
        message=(
            "Open modes are unfinished — improve/restore before inventing nearby duplicates. "
            "Use get_failure_mode(name) to inspect the last detector; "
            "restore_failure_mode(name) to bring it back for a rewrite."
        ),
    )


def get_failure_mode(name: str) -> dict[str, Any]:
    """Full ledger entry for one failure mode, including the last detector source."""
    rt = get_runtime()
    row = get_mode(rt.run.run_dir(), name)
    if row is None:
        return err(f"unknown failure mode: {name}")
    return ok(mode=row, failure_modes_summary=_failure_modes_summary())


def restore_failure_mode(name: str) -> dict[str, Any]:
    """Restore the last detector for a ledger mode into the working library (no auto-eval).

    After restore, rewrite the detector, then evaluate_functions / judge_function.
    """
    rt = get_runtime()
    from autofa.session.polishing_guard import polishing_tool_guard

    if blocked := polishing_tool_guard(rt, "restore_failure_mode", {"name": name}):
        return blocked
    with _library_mutation(rt) as persist:
        row = get_mode(rt.run.run_dir(), name)
        if row is None:
            return err(f"unknown failure mode: {name}")
        det = row.get("detector")
        if not isinstance(det, dict) or not det.get("source"):
            return err(f"no detector snapshot stored for mode: {name}")
        fn = FeatureFunction.from_dict(det)
        rt.library.upsert(fn)
        persist()
        diagnostics = _committed_mode_update(
            rt, fn, status="active", reason="restored from ledger for rewrite",
        )
    try:
        emit_runtime_status(
            f"Restored {fn.name} from ledger for rewrite",
            source="restore_failure_mode",
            context={"name": fn.name},
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics.append(f"Library change committed; status display unavailable: {exc}")
    return ok(
        restored=fn.name,
        diagnostics=diagnostics,
        n_functions=len(rt.library.functions),
        function=get_library_function(fn.name)["function"],
        failure_modes_summary=_failure_modes_summary(),
        message=(
            f"Restored {fn.name} into the working library (unevaluated). "
            "Rewrite the detector for the claimed mode, then evaluate_functions / judge_function."
        ),
    )


@family_transaction
def advance_round(validation_response: str | None = None) -> dict[str, Any]:  # noqa: C901
    """Finish a completed family-comparison batch and advance its discovery seed.

    Assess or explicitly defer the current hypothesis and finish the existing
    comparison budget first. Verified exhaustion of failed-task evidence also
    completes a batch. Detector success is not required. To pivot or sample
    different evidence within a round, use family_discovery and sample_items.
    A retained library first receives validation feedback without advancing.
    Respond with validation_response after one repair or retain/defer assessment;
    revised detectors are revalidated before the round advances.
    """
    from autofa.session.exploration import (
        exploration_lens,
        exploration_lens_hint,
        kickoff_ssot_lines,
        round_exploration_seed,
    )
    from autofa.session.round_validation import active_round_refinement
    from autofa.util.atomic import atomic_write_json

    rt = get_runtime()
    from autofa.session.campaign_policy import check_campaign_boundary
    from autofa.session.control import apply_pending_updates

    apply_pending_updates(rt)
    campaign = check_campaign_boundary(rt)
    if campaign["active"] and not campaign["continue_discovery"]:
        return ok(advanced=False, campaign=campaign, stop_reason=campaign["stop_reason"],
                  goal_met=campaign["quality_target_met"], refinement_pending=False,
                  message="The campaign boundary is reached. Return saved findings for "
                          "native finalization; do not begin another discovery operation.")
    if validation_response is not None and not isinstance(validation_response, str):
        return err("validation_response must describe the response to validation feedback.",
                   advanced=False)
    response = (validation_response or "").strip()
    validation_path = rt.run.run_dir() / "round_validation.json"
    from autofa.session.round_stagnation import round_stagnation_status

    stopped = round_stagnation_status(rt)
    if stopped.get("review_capacity_block"):
        return ok(
            advanced=False, round=rt.run.round, max_rounds=rt.run.max_rounds,
            phase="polishing", goal_met=False, refinement_pending=False,
            quality_stall=stopped, stop_reason="review_capacity_exhausted",
            message=stopped["reason"] + " Return the partial library summary for saving.",
        )
    refinement = active_round_refinement(rt)
    if refinement is None:
        from autofa.session.polishing_guard import active_polishing

        pending = active_polishing(rt)
        if pending and pending.get("transition_pending"):
            from autofa.session.polishing import continue_polishing

            status = _pareto_status()
            stall = status.get("quality_stall") or {}
            if status["goal_met"] or stall.get("should_stop"):
                return ok(
                    advanced=False, phase="polishing", goal_met=status["goal_met"],
                    quality_stall=stall,
                    stop_reason=(stall.get("stop_reason") or "stalled"
                                 if stall.get("should_stop") else None),
                    message=(str(stall["reason"]) if stall.get("review_capacity_block") else
                             "Polishing complete. Return the summary for finalization and saving."),
                )
            saved = json.loads(validation_path.read_text())
            refinement = continue_polishing(rt, saved)["refinement"]
    if refinement is None:
        from autofa.session.discovery_stall import observe_discovery_stall

        stall = observe_discovery_stall(rt)
        if stall["should_stop"]:
            return ok(
                advanced=False, round=rt.run.round, max_rounds=rt.run.max_rounds,
                quality_stall=stall, stop_reason=stall.get("stop_reason") or "stalled",
                message=(str(stall["reason"]) + ". Return the library summary for "
                         "runtime-managed final scoring, review, and saving; do not begin "
                         "another discovery round."),
            )
    completion_boundary = False
    # One harness turn can contain many round transitions. Enforce the driver's
    # decision here as well, before an agent can reset family state or advance
    # the sampling sequence while the harness awaits the end of that turn.
    if (rt.run.round >= rt.run.max_rounds
            or getattr(getattr(rt, "config", None), "discovery_wait_for_plateau", False)):
        status = _pareto_status()
        if status["goal_met"] or status["ready_for_final_validation"]:
            # A provisional target is also a validation boundary. Give its
            # findings a closing response before advancing or handing off; an
            # already validated idle Resume still exits without new discovery.
            from autofa.session.family_discovery import family_progress_summary

            try:
                existing = family_progress_summary(rt.run.run_dir()) or {}
            except ValueError:
                # A completed no-work handoff need not load obsolete family
                # state. If more work is needed, initialization below still
                # validates the file and preserves its original error.
                existing = {}
            completed_batch = (existing.get("phase") == "specialize"
                               and existing.get("round") == rt.run.round)
            completion_boundary = bool(rt.library.functions and (
                refinement or completed_batch or status["ready_for_final_validation"]))
            message = (
                "The provisional stopping boundary has been reached. Return the summary "
                "for driver-managed full-dataset scoring and review. The runtime rechecks "
                "completion afterward and supplies further work if validation falls short."
                if status["goal_met"] else status["final_validation_guidance"]
            )
            if not completion_boundary and not refinement:
                return ok(
                    advanced=False, goal_met=status["goal_met"],
                    round=rt.run.round, max_rounds=rt.run.max_rounds,
                    ready_for_final_validation=status["ready_for_final_validation"],
                    final_validation_candidates=status.get("final_validation_candidates", []),
                    final_validation_guidance=status.get("final_validation_guidance"),
                    message=message,
                )
    from autofa.session.family_discovery import (
        family_progress_summary,
        initialize_family_discovery,
    )

    state = initialize_family_discovery(rt)
    progress = family_progress_summary(rt.run.run_dir())
    if not (refinement or completion_boundary) and (state.get("synthesis_required")
            or (state["phase"] == "assess_family"
                and (state.get("survey") or {}).get("approach") == "bottom_up")):
        return err(
            "Complete the bottom-up inspection or parent assessment before advancing "
            "the round. " + progress["next_action"],
            advanced=False, family_discovery=progress,
        )
    if not (refinement or completion_boundary) and state["phase"] != "specialize":
        return err(
            f"Round not complete: {len(state['comparisons'])}/{state['comparison_budget']} "
            "family comparisons finished. " + progress["next_action"]
            + " Stay in this round while investigating another hypothesis. Use sample_items "
            "with coverage filters or an explicit seed for different examples. Failed tool "
            "calls do not complete a comparison; inspect their results before advancing.",
            advanced=False, family_discovery=progress,
        )
    validation = None
    from autofa.session.polishing import current_claims, public_validation_receipt
    from autofa.session.round_validation import refinement_guidance, validate_round

    def awaiting_response(receipt):
        from autofa.session.round_validation import round_refinement_prompt

        return ok(
            advanced=False, round=rt.run.round, max_rounds=rt.run.max_rounds,
            refinement_pending=True, round_validation=public_validation_receipt(receipt),
            message=round_refinement_prompt(rt) or (
                f"Round {rt.run.round + 1} remains open for one validation-feedback pass. "
                + refinement_guidance() + " Then call "
                "advance_round(validation_response='candidates, blocking dimensions, "
                "tested changes and results, or evidence for retain/defer'). "
                "Unchanged requests reuse this feedback without another validation."
            ),
        )

    if refinement:
        saved = json.loads(validation_path.read_text(encoding="utf-8"))
        if refinement["status"] == "awaiting_response":
            if not response:
                return awaiting_response(saved)
            if refinement.get("mode") == "polishing":
                from autofa.mcp.tools_eval import evaluate_functions
                from autofa.session.polishing import unchecked_revisions
                from autofa.session.round_validation import validation_feedback

                unchecked = unchecked_revisions(rt, refinement)
                if unchecked:
                    diagnostics = evaluate_functions(names=unchecked)
                    # Diagnostics can upgrade a resumed checkpoint's control
                    # inventory. Keep that persisted state when adding feedback.
                    saved = json.loads(validation_path.read_text(encoding="utf-8"))
                    saved = {**saved, **validation_feedback(rt),
                             "polishing_checks": diagnostics.get("polishing_feedback"),
                             "polishing_check_error": diagnostics.get("error")}
                    atomic_write_json(validation_path, saved)
                    result = awaiting_response(saved)
                    result["message"] = (
                        "These revisions needed regression checks before full verification. "
                        "Inspect the new per-case/group feedback, repair further if needed, "
                        "then send validation_response when ready. No full scan was run. "
                        + result["message"]
                    )
                    return result
                from autofa.session.polishing_progress import polishing_response_assessment

                assessment = polishing_response_assessment(rt, refinement)
                if not assessment["can_close"]:
                    result = awaiting_response(saved)
                    result["recovery_attempt"] = {
                        key: value for key, value in assessment.items() if key != "snapshot"
                    }
                    result["message"] = assessment["next_action"] + " " + result["message"]
                    return result
                refinement = {
                    **refinement,
                    "recovery_attempt": {
                        key: value for key, value in assessment.items() if key != "snapshot"
                    },
                    "recovery_work": assessment["snapshot"],
                }
            refinement = {**refinement, "status": "validating_response", "response": response}
            with cancellation_guard():
                atomic_write_json(validation_path, {**saved, "refinement": refinement})

    validation_revision = getattr(rt.run, "control_revision", 0)
    validation = validate_round(rt)
    if getattr(rt.run, "control_revision", 0) != validation_revision:
        # A goal update admitted at validation entry can supersede polishing.
        # Never restore the caller's pre-update frozen state over that transition.
        refinement = active_round_refinement(rt)
    if not validation.get("ok"):
        return {**validation, "advanced": False}
    if refinement is None and rt.library.functions:
        refinement = validation.get("refinement") or {
            "version": 1, "run_id": rt.run.run_id, "round": rt.run.round,
            "status": "awaiting_response", "evaluated_claims": current_claims(rt),
        }
        validation = {**validation, "refinement": refinement}
        with cancellation_guard():
            atomic_write_json(validation_path, validation)
        emit_runtime_status(
            f"Round {rt.run.round + 1}: inspect validation feedback before closing the round.",
            source="advance_round", context={"refinement": refinement},
        )
        return awaiting_response(validation)
    if refinement:
        refinement = validation.get("refinement") or refinement
        before, after = refinement["evaluated_claims"], current_claims(rt)
        refinement = {
            **refinement, "status": "complete",
            "changed_detectors": sorted(name for name in after
                                        if before.get(name) != after[name]),
            "removed_detectors": sorted(set(before) - set(after)),
            "retained_detectors": sorted(name for name in after
                                         if before.get(name) == after[name]),
        }
        validation = {**validation, "refinement": refinement}
    # Resume reads config.json; Continue reads the portable library cursor. Once
    # a round transition starts, cancellation must allow all checkpoints to finish.
    round_stall = None
    with cancellation_guard():
        if validation is not None:
            from autofa.session.discovery_stall import complete_discovery_round

            round_stall = complete_discovery_round(rt, validation)
        rt.run.round += 1
        rt.run.exploration_sequence_index += 1
        ssot = bool(getattr(rt.run, "exploration_ssot", True))
        templates = bool(getattr(rt.run, "exploration_templates", False))
        rt.run.exploration_seed = round_exploration_seed(
            rt.run.exploration_base_seed, rt.run.exploration_sequence_index,
        )
        rt.run.save()
        rt.persist_library()
        # Reset/archive the completed batch under the same family lock and
        # cancellation guard as the run and portable sampling cursor.
        family_state = initialize_family_discovery(rt)
        if validation is not None and refinement is not None:
            atomic_write_json(validation_path, validation)
    payload: dict[str, Any] = {
        "advanced": True,
        "round": rt.run.round,
        "exploration_ssot": ssot,
        "exploration_templates": templates,
        "exploration_seed": rt.run.exploration_seed,
        "exploration_base_seed": rt.run.exploration_base_seed,
        "exploration_sequence_index": rt.run.exploration_sequence_index,
        "detector_exploration": _detector_exploration(),
    }
    if refinement and refinement.get("mode") == "polishing":
        from autofa.session.polishing import continue_polishing, public_validation_receipt
        from autofa.session.round_validation import round_refinement_prompt

        status = _pareto_status()
        stop = bool(round_stall and round_stall["should_stop"])
        if not stop and not status["goal_met"]:
            validation = continue_polishing(rt, validation)
        message = (round_refinement_prompt(rt) if not stop and not status["goal_met"] else
                   "Polishing verification is complete. Return the library summary for "
                   "runtime-managed finalization and saving; do not start new discovery.")
        if stop and round_stall.get("review_capacity_block"):
            message = (round_stall["reason"] + " Return the partial library summary for "
                       "runtime-managed finalization and saving; do not start new discovery.")
        emit_runtime_status(
            ("Polishing blocked by exhausted review capacity." if stop
             and round_stall.get("review_capacity_block") else
             "Polishing verification complete." + (" Existing detectors need further repair."
                if not stop and not status["goal_met"] else "")),
            source="advance_round", context={"round": rt.run.round, "phase": "polishing"},
        )
        return ok(
            advanced=True, round=rt.run.round, max_rounds=rt.run.max_rounds,
            phase="polishing", target_grade="GOOD", goal_met=status["goal_met"],
            quality_stall=round_stall,
            stop_reason=(round_stall.get("stop_reason") or "stalled") if stop else None,
            refinement_pending=not stop and not status["goal_met"],
            round_validation=public_validation_receipt(validation), message=message,
        )
    if validation is not None:
        payload["round_validation"] = validation
        payload["quality_stall"] = round_stall
    if family_state is not None:
        payload["family_discovery_phase"] = family_state["phase"]
        payload["family_discovery_hint"] = (
            "Call family_discovery(action='status') before creating new detectors."
        )
    from autofa.session.discovery_budget import discovery_budget_feedback
    from autofa.session.discovery_target import (
        agent_failure_mode_target,
        discovery_good_target_for_runtime,
        discovery_grade_assessment,
    )

    target = agent_failure_mode_target(
        discovery_good_target_for_runtime(rt),
        round_idx=rt.run.round, max_rounds=rt.run.max_rounds,
    )
    payload.update(target)
    budget = discovery_budget_feedback(
        rt.run.round, rt.run.max_rounds,
        campaign_policy=getattr(rt.run, "campaign_policy", None),
    )
    payload["budget_feedback"] = budget
    payload["max_rounds"] = rt.run.max_rounds
    from autofa.session.round_feedback import round_bulletin
    from autofa.session.task_coverage import task_coverage_snapshot

    grades = _discovery_grades(rt)
    coverage = task_coverage_snapshot(rt)
    grade_feedback = discovery_grade_assessment(
        grades, getattr(rt, "_previous_discovery_grades", None),
        minimum=rt.run.min_failure_modes, budget_feedback=budget,
        target=target, pending=target["pending_failure_modes"],
        # Reuse cached source identities and saved results; no detector or judge calls.
        task_coverage=coverage,
        round_feedback=round_bulletin(rt, grades=grades, task_coverage=coverage),
    )
    payload["agent_grade_feedback"] = grade_feedback
    if ssot:
        domain = str(getattr(rt.run, "corpus_domain", "unknown"))
        if templates:
            lens = exploration_lens(rt.run.exploration_seed, domain=domain)
            payload["exploration_lens"] = lens
            payload["exploration_lens_hint"] = exploration_lens_hint(lens)
        msg = (
            f"Advanced to round {rt.run.round}. "
            + kickoff_ssot_lines(
                exploration_seed=rt.run.exploration_seed,
                exploration_templates=templates,
                domain=domain,
                objective_config=rt.config,
            )
        )
    else:
        msg = f"Advanced to round {rt.run.round}. exploration_ssot is off."
    stop = bool(round_stall and round_stall["should_stop"])
    if stop:
        payload["stop_reason"] = round_stall.get("stop_reason") or "stalled"
        msg = (str(round_stall["reason"]) + ". Return the library summary for "
               "runtime-managed final scoring, review, and saving. Do not start another round.")
    emit_runtime_status(
        msg if stop else (
            f"Completed investigation batch; starting round {rt.run.round + 1} "
            f"(seed={rt.run.exploration_seed})"
        ),
        source="advance_round",
        context={
            "round": rt.run.round,
            "exploration_seed": rt.run.exploration_seed,
        },
    )
    return ok(**payload, message=(validation["message"] + "\n" if validation else "")
              + msg + " " + payload["detector_exploration"]["message"]
              + "\n" + grade_feedback["message"])


def _sampled_scoring_note(n_sampled: int) -> str | None:
    if not n_sampled:
        return None
    return (
        f"{n_sampled} detector(s) have stratified-sample estimates; retain their uncertainty. "
        "Use search_ranking for iterative comparisons on the same scoring panel and compare "
        "only matching cohort_fingerprint values. Corpus results and frontier eligibility "
        "are separate; a sampled detector is not necessarily on that frontier. Under "
        "task-recall objectives, sampled candidates await a complete corpus scan for the "
        "corpus frontier. Follow the runtime's final_validation_guidance for that scan."
    )


def _evidence_check_hint(example_fail_ids: list[str] | None) -> str | None:
    ids = [i for i in (example_fail_ids or []) if i]
    if not ids:
        return None
    return (
        f"get_item on example_fail_ids={ids[:5]} and verify description/mechanism "
        "name the behavior those traces actually show (or retune the detector)."
    )


def _task_recall_feedback(scores: dict[str, Any]) -> dict[str, Any]:
    from autofa.metrics.task_support import task_support_assessment

    support = task_support_assessment(scores)
    support.pop("n_matched_failed_tasks")
    return {key: value for key, value in scores.items()
            if key.startswith("task_recall") or key in {
                "n_matched_failed_tasks", "n_failed_tasks", "n_matched_failed_attempts",
                "n_matched_failed_attempts_without_task_identity",
            }} | support


def _feedback_hint(
    scores: dict[str, Any],
    prose_warn: str | None = None,
    *,
    evidence_hint: str | None = None,
    config=None,
    target_grade: str | None = None,
) -> str:
    """Use the shared evidence-based advice instead of a second set of thresholds."""
    from autofa.config import load_config
    from autofa.metrics.claim_calibration import summarize_judge_feedback

    cfg = config or load_config()
    strictness = strictness_for_config(cfg)
    judge_feedback = summarize_judge_feedback(
        scores, name="<function name>", behavior_strictness=strictness,
    )
    # Current per-case findings and stale-review recovery take precedence over
    # advisory aggregate metrics. Editorial warnings never override those findings.
    if judge_feedback["status"] not in {"legacy", "unjudged"}:
        actions = judge_feedback["next_steps"]
        if actions:
            return " ".join(actions)
    if prose_warn:
        return prose_warn
    advice = build_discovery_feedback(
        scores, judge_feedback=judge_feedback, behavior_strictness=strictness,
        target_grade=target_grade,
    )
    steps = list(advice["next_steps"])
    if evidence_hint:
        steps.append(evidence_hint)
    return " ".join(steps)
