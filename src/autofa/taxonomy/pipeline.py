# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Digest and classify one prepared rollout."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from autofa import trace
from autofa.data.evidence_refs import expand_trace
from autofa.data.input_view import response_value
from autofa.data.source_execution import execution_record
from autofa.data.structured import decode_container

from .classifier import TaxonomyClassifier
from .outcome_evidence import OutcomeSource, load_outcome_evidence_record
from .record import classification_sha256
from .trace_digest import (
    DEFAULT_JUDGEMENT_CHARACTER_BUDGET,
    build_trace_digest,
    build_trace_judgement_context,
)
from .trace_windows import TraceWindowingError, build_trace_judgement_windows

EVIDENCE_CACHE_VERSION = "taxonomy-evidence-v2"


def evidence_quality_report(incident: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic pre-judge evidence checks."""
    evidence = incident.get("evidence") or []
    first_evidence = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
    if first_evidence.get("kind") != "bounded_ordered_trace_digest":
        return {"status": "pass", "issues": []}
    observation = first_evidence.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    trace = observation.get("ordered_trace") or []
    outcome = observation.get("outcome_evidence") or {}
    coverage = observation.get("coverage") or {}
    issues: list[dict[str, str]] = []

    def issue(code: str, detail: str, *, severity: str = "error") -> None:
        issues.append({"code": code, "severity": severity, "detail": detail})

    if not trace:
        issue("trace_absent", "No ordered trace events are available.")
    roles = [((event.get("event_ref") or {}).get("role")) for event in trace]
    if "user" not in roles and "system" not in roles and "developer" not in roles:
        issue("task_context_absent", "No user, system, or developer task context is available.")
    first_user = next(
        (
            index
            for index, event in enumerate(trace)
            if (event.get("event_ref") or {}).get("role") == "user"
        ),
        None,
    )
    if first_user is not None:
        for event in trace[:first_user]:
            ref = event.get("event_ref") or {}
            if ref.get("role") in {"assistant", "agent"} and ref.get("phase") != "pre_task_context":
                issue(
                    "pre_task_model_attribution",
                    f"Event {ref.get('event_idx')} precedes the first user task without a "
                    "pre_task_context marker.",
                )
                break
    observed = outcome.get("observed_outcome") or {}
    items = outcome.get("items") or []
    reward = observed.get("reward", (incident.get("source_provenance") or {}).get("reward"))
    if (
        reward is not None
        and not items
        and outcome.get("semantics") == "unknown"
        and not coverage.get("verifier_artifact_count")
    ):
        issue(
            "outcome_cause_unavailable",
            "Only a scalar outcome is available; no evaluator or verifier cause is present.",
            severity="warning",
        )
    errors = sum(item["severity"] == "error" for item in issues)
    return {
        "status": "fail" if errors else "warn" if issues else "pass",
        "issues": issues,
    }


def prepare_rollout(
    item: dict[str, Any],
    source: OutcomeSource,
    *,
    run_id: str,
    max_serialized_characters: int = DEFAULT_JUDGEMENT_CHARACTER_BUDGET,
) -> dict[str, Any]:
    """Build cache-ready evidence for one prepared dataset row without calling a judge."""
    task_id, repeat = str(item["problem_id"]), int(item["run_idx"])
    row: dict[str, Any] = {
        "rollout_id": f"{run_id}:{task_id}:{repeat}",
        "run_id": run_id,
        "task_id": task_id,
        "repeat": repeat,
        "lane": str(item.get("lane") or ""),
        "benchmark": item.get("benchmark"),
        "model": item.get("model_id"),
        "cache_version": EVIDENCE_CACHE_VERSION,
        "max_serialized_characters": max_serialized_characters,
    }
    try:
        execution = trace.record(item)
        if execution["availability"]["events"]["status"] == "unavailable":
            raise ValueError("prepared row has no execution trace; prepare it with full traces")
        execution = {
            **execution,
            **execution_record({}, [], [], execution=execution),
        }
        lane = item.get("lane") or ""
        row["agent_ref"] = execution["rollout"].get("agent_ref")
        rollout = {
            **execution["rollout"],
            "run_id": run_id,
            "task_id": task_id,
            "repeat": repeat,
            "reward": item.get("reward"),
        }
        events = [event for event in execution["events"] if (event.get("lane") or "") == lane]
        auxiliary_events = [
            event for event in execution["events"] if (event.get("lane") or "") != lane
        ]
        tool_calls = [call for call in execution["tool_calls"] if (call.get("lane") or "") == lane]
        auxiliary_tool_calls = [
            call for call in execution["tool_calls"] if (call.get("lane") or "") != lane
        ]
        artifacts = source.artifacts(task_id, repeat)
        verifier_artifacts = [item for item in artifacts if item.get("role") == "verifier"]
        outcome_evidence, outcome_receipt, raw_record = load_outcome_evidence_record(
            source, rollout, artifacts
        )
        events, excluded_evaluator_prompts = _exclude_auxiliary_evaluator_prompts(
            events, raw_record
        )
        retained_event_indexes = {
            event.get("event_idx") for event in events if type(event.get("event_idx")) is int
        }
        tool_calls = [
            call for call in tool_calls if call.get("event_idx") in retained_event_indexes
        ]
        row["artifact_source"] = _artifact_source(
            item,
            execution,
            rollout,
            events=[*events, *auxiliary_events],
            tool_calls=[*tool_calls, *auxiliary_tool_calls],
        )
        digest_args = {
            "rollout": rollout,
            "tool_calls": tool_calls,
            "events": events,
            "verifier_artifacts": verifier_artifacts,
            "outcome_evidence": outcome_evidence,
            "evidence_receipt": _evidence_receipt(
                events,
                tool_calls,
                artifacts,
                verifier_artifacts,
                outcome_receipt,
                excluded_evaluator_prompts=excluded_evaluator_prompts,
            ),
            "auxiliary_events": auxiliary_events,
            "auxiliary_tool_calls": auxiliary_tool_calls,
            "trace_resources": _trace_resources(execution),
            "max_serialized_characters": max_serialized_characters,
        }
        fallback_compaction = False
        digest = build_trace_digest(**digest_args, compact_to_fit=False)
        try:
            windows = (
                build_trace_judgement_windows(
                    digest,
                    max_serialized_characters=max_serialized_characters,
                )
                if digest["budget"]["requires_windowing"]
                else []
            )
        except TraceWindowingError:
            digest = build_trace_digest(**digest_args, compact_to_fit=True)
            fallback_compaction = digest["budget"]["compaction_level"] > 0
            windows = (
                build_trace_judgement_windows(
                    digest,
                    max_serialized_characters=max_serialized_characters,
                )
                if digest["budget"]["requires_windowing"]
                else []
            )
        digest["budget"]["fallback_compaction"] = fallback_compaction
        row["digest_budget"] = digest["budget"]
        row["evidence_receipt"] = digest["evidence_receipt"]
        incident = _rollout_incident(rollout, digest)
        row["windows"] = windows
        row["incident"] = incident
        row["evidence_quality"] = evidence_quality_report(incident)
        if row["evidence_quality"]["status"] == "fail":
            row["status"] = "needs_evidence"
            return row
    except TraceWindowingError as exc:
        budget = dict(digest["budget"])
        budget["fallback_compaction"] = fallback_compaction
        return {
            **row,
            "status": "needs_windowing",
            "digest_budget": budget,
            "evidence_receipt": digest["evidence_receipt"],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - one rollout must not abort the batch
        return {**row, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    row["status"] = "prepared"
    return row


def classify_prepared(
    classifier: TaxonomyClassifier,
    prepared: dict[str, Any],
    *,
    parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one prepared evidence record without accessing the Data API."""
    row = {
        key: prepared.get(key)
        for key in (
            "rollout_id",
            "run_id",
            "task_id",
            "repeat",
            "lane",
            "benchmark",
            "model",
            "agent_ref",
            "digest_budget",
            "evidence_receipt",
            "evidence_quality",
        )
    }
    if parent is not None:
        row["parent_classification"] = {
            "record_sha256": classification_sha256(parent),
            "results_path": parent.get("_results_path"),
            "incident_id": parent.get("incident_id"),
            "selected_code": parent.get("selected_code"),
            "second_pass": parent.get("second_pass") or {},
        }
    row["judge_attempted"] = False
    quality = prepared.get("evidence_quality") or (
        evidence_quality_report(prepared["incident"]) if prepared.get("incident") else None
    )
    row["evidence_quality"] = quality
    if quality and quality.get("status") == "fail":
        return {
            **row,
            "status": "needs_evidence",
            "error_type": "EvidenceQualityError",
            "error": "; ".join(issue["detail"] for issue in quality.get("issues") or []),
        }
    if prepared.get("status") != "prepared":
        return {
            **row,
            "status": str(prepared.get("status") or "error"),
            "error_type": prepared.get("error_type"),
            "error": prepared.get("error") or "prepared evidence is unavailable",
        }
    try:
        incident = prepared["incident"]
        windows = prepared.get("windows") or []
        started = time.perf_counter()
        row["judge_attempted"] = True
        if parent is not None:
            result = classifier.classify_second_pass(
                incident,
                parent,
                windows=windows,
                artifact_source=prepared.get("artifact_source"),
            )
        else:
            result = (
                classifier.classify_windows(
                    incident,
                    windows,
                    artifact_source=prepared.get("artifact_source"),
                )
                if windows
                else classifier.classify(
                    incident,
                    artifact_source=prepared.get("artifact_source"),
                )
            )
        row["analysis"] = result.to_dict()
        row["classification_latency_ms"] = round((time.perf_counter() - started) * 1000)
    except Exception as exc:  # noqa: BLE001 - one rollout must not abort the batch
        return {**row, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    second_pass = row["analysis"].get("second_pass") or {}
    if second_pass.get("status") == "failed":
        row["status"] = "review_failed"
        row["error_type"] = second_pass.get("error_type") or "SecondPassError"
        row["error"] = second_pass.get("error") or "second pass failed"
        return row
    row["status"] = "ok"
    return row


def _artifact_source(
    item: dict[str, Any],
    execution: dict[str, Any],
    rollout: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Retain the trusted normalized execution for trace and artifact navigation."""
    retained = deepcopy(execution)
    retained["rollout"] = deepcopy(rollout)
    retained["events"] = deepcopy(events)
    retained["tool_calls"] = deepcopy(tool_calls)
    envelope = decode_container(response_value(item))
    if isinstance(envelope, list) and len(envelope) == 1:
        envelope = envelope[0]
    source = expand_trace(envelope) if isinstance(envelope, dict) else {}
    source["steps"] = _retained_steps(source.get("steps"), events)
    source["execution"] = retained
    return {"trace": source}


def _retained_steps(
    value: Any,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only normalized steps tied to retained execution events."""
    if not isinstance(value, list):
        return []
    retained = {
        (event.get("event_idx"), str(event.get("lane") or ""))
        for event in events
        if type(event.get("event_idx")) is int
    }
    steps = []
    for step in value:
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata")
        if not isinstance(metadata, dict):
            continue
        event_idx = step.get("step_id", metadata.get("event_idx"))
        step_lane = str(metadata.get("lane") or "")
        if (
            type(event_idx) is int
            and (event_idx, step_lane) in retained
        ):
            steps.append(deepcopy(step))
    return steps


def _trace_resources(execution: dict[str, Any]) -> dict[str, list[Any]]:
    """Project only standardized resources already filtered with the execution."""
    names = (
        "attempts",
        "relations",
        "provenance",
        "tool_definitions",
        "prompt_messages",
        "prompt_summaries",
    )
    return {
        name: list(values)
        for name in names
        for values in [execution.get(name)]
        if isinstance(values, list) and values
    }


def _rollout_incident(rollout: dict[str, Any], digest: dict[str, Any]) -> dict[str, Any]:
    run_id = str(rollout.get("run_id") or "")
    task_id = str(rollout.get("task_id") or "")
    repeat = int(rollout.get("repeat") or 0)
    return {
        "id": f"{run_id}:{task_id}:{repeat}",
        "observed_event": (
            f"The benchmark trajectory has observed reward={rollout.get('reward')!r}. "
            "The ordered trace digest is evidence, not a causal conclusion."
        ),
        "evidence": [
            {
                "evidence_id": "ordered-trace",
                "kind": "bounded_ordered_trace_digest",
                "observation": build_trace_judgement_context(digest),
            }
        ],
        "source_provenance": {
            "run_id": run_id,
            "task_id": task_id,
            "repeat": repeat,
            "reward": rollout.get("reward"),
        },
    }


def _evidence_receipt(
    events: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    verifier_artifacts: list[dict[str, Any]],
    raw_outcome: dict[str, Any],
    *,
    excluded_evaluator_prompts: int = 0,
) -> dict[str, Any]:
    return {
        "trace": {
            "status": "included",
            "event_count": len(events),
            "tool_call_count": len(tool_calls),
            **(
                {"excluded_auxiliary_evaluator_prompts": excluded_evaluator_prompts}
                if excluded_evaluator_prompts
                else {}
            ),
        },
        "verifier": {
            "status": "included" if verifier_artifacts else "absent",
            "count": len(verifier_artifacts),
            "read_failures": sum(
                bool(artifact.get("text_error")) for artifact in verifier_artifacts
            ),
        },
        "artifacts": {
            "status": "inventory_only" if artifacts else "absent",
            "count": len(artifacts),
            "roles": sorted(
                {
                    str(artifact["role"])
                    for artifact in artifacts
                    if artifact.get("role") is not None
                }
            ),
        },
        "raw_outcome": raw_outcome,
    }


def _exclude_auxiliary_evaluator_prompts(
    events: list[dict[str, Any]],
    raw_record: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Remove nested pairwise-judge prompts incorrectly merged into the candidate lane."""
    if not isinstance(raw_record, dict):
        return events, 0
    primary = _request_system_messages(raw_record.get("responses_create_params"))
    auxiliary: set[str] = set()
    for game in raw_record.get("games") or []:
        if isinstance(game, dict):
            auxiliary.update(_request_system_messages(game.get("responses_create_params")))
    auxiliary.difference_update(primary)
    if not auxiliary:
        return events, 0

    kept: list[dict[str, Any]] = []
    excluded = 0
    for event in events:
        source_id = event.get("source_event_id")
        role = event.get("role")
        text = _event_text(event)
        if (
            isinstance(source_id, str)
            and source_id.startswith("reminder:")
            and role in {"system", "developer"}
            and text in auxiliary
        ):
            excluded += 1
            continue
        kept.append(event)
    return kept, excluded


def _request_system_messages(request: Any) -> set[str]:
    if not isinstance(request, dict):
        return set()
    items = request.get("input")
    if not isinstance(items, list):
        items = request.get("messages")
    if not isinstance(items, list):
        return set()
    return {
        text
        for item in items
        if isinstance(item, dict) and item.get("role") in {"system", "developer"}
        for text in [_message_content_text(item.get("content"))]
        if text
    }


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or block.get("input_text") or "")
        for block in content
        if isinstance(block, dict)
    )


def _event_text(event: dict[str, Any]) -> str:
    return "".join(
        str(part.get("text") or "") for part in event.get("parts") or [] if isinstance(part, dict)
    )
