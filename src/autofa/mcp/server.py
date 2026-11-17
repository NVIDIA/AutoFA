# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared in-process AutoFA tools and their Claude MCP / Codex adapters."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, get_type_hints

from autofa.diagnostics import get_diagnostics
from autofa.mcp import (
    tools_artifacts,
    tools_classifier,
    tools_contributions,
    tools_data,
    tools_eval,
    tools_helpers,
    tools_library,
    tools_repair,
    tools_repair_versions,
)
from autofa.mcp.tool_results import describe_result_value, read_tool_result, save_tool_result
from autofa.session.candidate_screen import screen_function
from autofa.session.family_discovery import family_discovery
from autofa.session.recovery import RunInterruptedError
from autofa.session.supervision import respond_to_steering
from autofa.util.cancellation import Cancellation, cancellation_scope

log = logging.getLogger(__name__)
_INVOCATION_ACTOR: ContextVar[str] = ContextVar("autofa_invocation_actor", default="native")


def invocation_actor() -> str:
    """Caller identity supplied by the adapter, never by model tool arguments."""
    return _INVOCATION_ACTOR.get()


# Bound individual responses even when the SDK cannot compact a full conversation.
# Claude Code moves any tool result above 50,000 characters into a file the agent must
# read back; results above this ceiling use AutoFA's saved-result envelope instead.
MCP_SOFT_WARN_CHARS = 30_000
MCP_HARD_CEILING_CHARS = 45_000
# The envelope repeats short top-level fields and the head of the response's guidance,
# so outcomes such as ok, advanced or round need no extra read.
ENVELOPE_FIELD_CHARS = 400
ENVELOPE_FIELDS_CHARS = 6_000
ENVELOPE_HEAD_CHARS = 2_000
_ENVELOPE_HEAD_KEYS = (
    "error", "message", "next_action", "_exposure_warning", "discovery_recovery",
)
_ENVELOPE_KEYS = frozenset({
    "ok", "tool", "result_saved", "result_id", "total_chars", "truncated_fields",
    "_exposure_warning", "discovery_recovery", "full_result_complete", "structure", "note",
    "read_with", "inspect_with",
})

TOOL_CALLABLES: dict[str, Callable[..., dict]] = {
    "library_import_candidates": tools_contributions.library_import_candidates,
    "record_external_inspection": tools_contributions.record_external_inspection,
    "record_source_audit": tools_contributions.record_source_audit,
    "audit_sensitivity": tools_contributions.audit_sensitivity,
    "respond_to_steering": respond_to_steering,
    "download_artifact": tools_artifacts.download_artifact,
    "read_artifact": tools_artifacts.read_artifact,
    "view_artifact": tools_artifacts.view_artifact,
    "prepare_artifact_support": tools_artifacts.prepare_artifact_support,
    "transcribe_artifact": tools_artifacts.transcribe_artifact,
    "get_trace_api": tools_data.get_trace_api,
    "get_trace_record": tools_data.get_trace_record,
    "read_tool_result": read_tool_result,
    "dataset_info": tools_data.dataset_info,
    "describe_item_fields": tools_data.describe_item_fields,
    "dataset_chunks": tools_data.dataset_chunks,
    "peek_chunk": tools_data.peek_chunk,
    "sample_items": tools_data.sample_items,
    "library_coverage": tools_data.library_coverage,
    "family_discovery": family_discovery,
    "search_items": tools_data.search_items,
    "get_item": tools_data.get_item,
    "get_input": tools_data.get_input,
    "get_model_calls": tools_data.get_model_calls,
    "read_parsed_field": tools_data.read_parsed_field,
    "get_field_meta": tools_data.get_field_meta,
    "read_field_chunk": tools_data.read_field_chunk,
    "corpus_trace_stats": tools_data.corpus_trace_stats,
    "get_trajectory": tools_data.get_trajectory,
    "list_trajectory_tools": tools_data.list_trajectory_tools,
    "get_trajectory_step": tools_data.get_trajectory_step,
    "search_trajectory": tools_data.search_trajectory,
    "get_trajectory_context": tools_data.get_trajectory_context,
    "smoke_test_feature": tools_eval.smoke_test_feature,
    "classifier_profiles": tools_classifier.classifier_profiles,
    "screen_function": screen_function,
    "evaluate_functions": tools_eval.evaluate_functions,
    "judge_function": tools_eval.judge_function,
    "llm_classify": tools_eval.llm_classify,
    "submit_function": tools_library.submit_function,
    "submit_helper": tools_helpers.submit_helper,
    "list_helpers": tools_helpers.list_helpers,
    "get_helper": tools_helpers.get_helper,
    "test_helper": tools_helpers.test_helper,
    "retain_as_helper": tools_helpers.retain_as_helper,
    "deprecate_helper": tools_helpers.deprecate_helper,
    "remove_function": tools_library.remove_function,
    "clear_library": tools_library.clear_library,
    "list_library_functions": tools_library.list_library_functions,
    "get_library_function": tools_library.get_library_function,
    "get_function_feedback": tools_library.get_function_feedback,
    "get_function_repair": tools_repair.get_function_repair,
    "plan_function_repair": tools_repair.plan_function_repair,
    "defer_function_repair": tools_repair.defer_function_repair,
    "list_function_versions": tools_repair_versions.list_function_versions,
    "restore_function_version": tools_repair_versions.restore_function_version,
    "list_failure_modes": tools_library.list_failure_modes,
    "get_failure_mode": tools_library.get_failure_mode,
    "restore_failure_mode": tools_library.restore_failure_mode,
    "report_status": tools_library.report_status,
    "list_status_updates": tools_library.list_status_updates,
    "library_save": tools_library.library_save,
    "library_load": tools_library.library_load,
    "pareto_status": tools_library.pareto_status,
    "advance_round": tools_library.advance_round,
}


def _json_head(text: str, limit: int) -> str:
    """Longest prefix whose JSON encoding fits ``limit``; escapes can expand text."""
    head = text[:limit]
    while head and len(encoded := json.dumps(head)) > limit:
        head = head[:len(head) * limit // len(encoded)]
    return head


def _envelope_fields(result: Any, result_id: str) -> dict[str, Any]:
    """Exact short top-level values, plus bounded heads of guidance and diagnostics."""
    if not isinstance(result, dict):
        return {}
    fields: dict[str, Any] = {}
    budget = ENVELOPE_FIELDS_CHARS
    for key, value in result.items():
        if not isinstance(key, str) or key in _ENVELOPE_KEYS or key in _ENVELOPE_HEAD_KEYS:
            continue
        size = len(json.dumps({key: value}, default=str))
        if size <= min(ENVELOPE_FIELD_CHARS, budget):
            fields[key] = value
            budget -= size
    truncated = {}
    for key in _ENVELOPE_HEAD_KEYS:
        if key not in result:
            continue
        text = json.dumps(result[key], default=str)
        if len(text) <= ENVELOPE_HEAD_CHARS:
            fields[key] = result[key]
            continue
        value = result[key] if isinstance(result[key], str) else text
        fields[key] = _json_head(value, ENVELOPE_HEAD_CHARS)
        truncated[key] = {
            "total_chars": len(text),
            "read_with": f"read_tool_result(result_id={result_id!r}, pointer='/{key}')",
        }
    if truncated:
        fields["truncated_fields"] = truncated
    return fields


def _serialize_overflow_envelope(envelope: dict[str, Any]) -> str:
    """Apply the transport ceiling to previews as well as their saved source result."""
    payload = json.dumps(envelope, default=str)
    if len(payload) <= MCP_HARD_CEILING_CHARS:
        return payload
    # A future preview expansion must not bypass the ceiling. Preserve the
    # operation outcome and lossless paging handle; discard optional previews
    # largest first, keeping short outcome flags whenever possible.
    required = {"ok", "tool", "result_saved", "result_id", "total_chars",
                "full_result_complete", "read_with"}
    optional = sorted((key for key in envelope if key not in required),
                      key=lambda key: len(json.dumps(envelope[key], default=str)),
                      reverse=True)
    for key in optional:
        envelope.pop(key)
        payload = json.dumps(envelope, default=str)
        if len(payload) <= MCP_HARD_CEILING_CHARS:
            return payload
    # Required fields are runtime-owned scalar values and a bounded saved-result
    # handle, not producer evidence or free-form recovery messages.
    return payload


def _prepare_tool_presentation(result: Any, tool_name: str) -> Any:
    if tool_name == "judge_function" and isinstance(result, dict):
        # Presentation only: internal judge consumers and saved reviews retain
        # their complete original evidence objects.
        return tools_library.reference_review_response(result, verdict_path=("verdicts",))
    return result


def clamp_mcp_payload(
    result: Any, tool_name: str, *, presentation_prepared: bool = False,
) -> str:
    """Serialize a tool result without mid-JSON truncation.

    Soft-warn above ``MCP_SOFT_WARN_CHARS`` (still return full payload).
    Hard-ceiling above ``MCP_HARD_CEILING_CHARS`` saves the result and returns a valid
    envelope with its short top-level fields and the heads of its guidance.
    """
    if not presentation_prepared:
        result = _prepare_tool_presentation(result, tool_name)
    payload = json.dumps(result, default=str, ensure_ascii=False)
    n = len(payload)

    if n > MCP_HARD_CEILING_CHARS:
        # The operation already ran. Preserve its result and its success/failure;
        # an overflow is not a reason to submit or judge the same detector again.
        try:
            saved = save_tool_result(result)
        except (OSError, RuntimeError):
            log.warning("Could not save oversized MCP result", exc_info=True)
        else:
            get_diagnostics().warning(
                "mcp_response_saved",
                f"MCP {tool_name} response saved for paged reading ({n} chars)",
                context={"tool": tool_name, **saved},
            )
            try:
                structure = describe_result_value(result, limit=20, max_chars=4000)
            except ValueError:
                # Unusually long keys must not prevent access to the full result.
                structure = None
            return _serialize_overflow_envelope({
                "ok": result.get("ok", True) if isinstance(result, dict) else True,
                "tool": tool_name, "result_saved": True, **saved,
                **_envelope_fields(result, saved["result_id"]),
                "full_result_complete": False,
                "structure": structure,
                "note": "The tool already ran; do not repeat it. Short top-level fields are "
                        "exact; truncated_fields marks heads to read by pointer. The structure "
                        "is a preview, not the evidence. Read exact fields with pointer or page "
                        "the full result. "
                        "Stop on next_offset=null, not complete. Stop on read errors or a "
                        "nonadvancing cursor; never restart at zero automatically. "
                        "Required evidence review scope still applies.",
                "read_with": f"read_tool_result(result_id={saved['result_id']!r}, offset=0)",
                "inspect_with": (
                    f"read_tool_result(result_id={saved['result_id']!r}, describe=True)"
                ),
            })
        item_id = None
        if isinstance(result, dict):
            item_id = result.get("item_id") or result.get("_item_id")
        if tool_name in {"judge_function", "get_function_feedback"} and isinstance(result, dict):
            read_with = (
                f"get_function_feedback(name={result.get('name')!r}, offset=0, limit=1); "
                "follow pagination.next_offset, or use get_function_repair for a compact diagnosis"
            )
        elif item_id:
            read_with = (
                f"read_field_chunk(item_id={item_id!r}, field='...', offset=0) "
                "or get a smaller page"
            )
        else:
            read_with = (
                "read_field_chunk(item_id=..., field='...', offset=0) or get a smaller page"
            )
        get_diagnostics().critical(
            "mcp_response_too_large",
            f"MCP {tool_name} response {n} chars exceeds hard ceiling",
            context={"tool": tool_name, "total_chars": n},
            remediation=read_with,
        )
        return json.dumps(
            {
                "ok": False,
                "error": "response_too_large",
                "total_chars": n,
                "tool": tool_name,
                "read_with": read_with,
            }
        )

    if n > MCP_SOFT_WARN_CHARS:
        ctx: dict[str, Any] = {"tool": tool_name, "chars": n}
        if isinstance(result, dict):
            for key in ("item_id", "_item_id", "problem_id"):
                if result.get(key) is not None:
                    ctx[key] = result[key]
            get_diagnostics().warning(
                "mcp_response_large",
                f"MCP {tool_name} response is large ({n} chars)",
                context=ctx,
            )
            annotated = {**result, "_mcp_size_chars": n}
            return json.dumps(annotated, default=str, ensure_ascii=False)
        get_diagnostics().warning(
            "mcp_response_large",
            f"MCP {tool_name} response is large ({n} chars)",
            context=ctx,
        )

    return payload



_ITEM_READ_TOOLS = frozenset({
    "get_trace_record", "get_item", "get_input", "get_model_calls", "read_parsed_field",
    "read_field_chunk", "get_trajectory", "list_trajectory_tools", "get_trajectory_step",
    "search_trajectory", "get_trajectory_context", "download_artifact", "read_artifact",
    "view_artifact", "transcribe_artifact",
})
_EXAMPLE_CONTENT_KEYS = frozenset({
    "prompt_text", "response_text", "reasoning_text", "generation", "trace_json", "messages",
    "conversation", "trace", "quote", "quotes", "verdict", "claim_assessments", "citations",
    "causal_chain", "evidence", "excerpt", "text", "response", "prompt", "match", "context",
})


def _record_tool_exposure(name: str, args: dict[str, Any], result: Any) -> None:  # noqa: C901
    """Track visible evidence, not bare detector bucket memberships or score rows."""
    from autofa.mcp.runtime import get_runtime
    from autofa.session.family_coverage import verified_identity_projection
    from autofa.session.family_discovery import STATE_FILE
    from autofa.session.review_evidence import record_discovery_exposure

    try:
        rt = get_runtime()
    except RuntimeError:
        return
    if not (rt.run.run_dir() / STATE_FILE).exists():
        return
    if not isinstance(result, dict):
        return
    failed = result.get("ok") is False
    saved_decision = name == "family_discovery" and result.get("decision_recorded") is True
    if (failed and name != "classifier_profiles" and not saved_decision
            and not result.get("candidate_screen_blocked")):
        return
    candidates: set[str] = set()

    def visible(value: Any) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return any(visible(child) for child in value.values())
        if isinstance(value, list):
            return any(visible(child) for child in value)
        return value is not None

    def collect(value: Any, enclosing_item_id: str | None = None) -> None:
        if isinstance(value, dict):
            item_id = value.get("_item_id", value.get("item_id", enclosing_item_id))
            if (not failed and isinstance(item_id, str) and item_id
                    and any(visible(value[key]) for key in
                            _EXAMPLE_CONTENT_KEYS.intersection(value))):
                candidates.add(item_id)
            for key, child in value.items():
                if (key == "source_checks" and isinstance(child, list)
                        and name in {"submit_function", "evaluate_functions", "screen_function"}):
                    from autofa.core.fields import DEFAULT_CONTENT_FIELDS

                    # A checked source's existence or absence informs this claim.
                    # Unknown checks and bare membership lists disclose no such fact.
                    candidates.update(
                        row["item_id"] for row in child if isinstance(row, dict)
                        and isinstance(row.get("item_id"), str) and row["item_id"]
                        and isinstance(row.get("field"), str)
                        and row.get("field") in DEFAULT_CONTENT_FIELDS
                        and isinstance(row.get("availability"), str)
                        and row.get("availability") in {"retained", "not_retained"}
                    )
                if isinstance(child, (dict, list)):
                    collect(child, item_id)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    collect(child, enclosing_item_id)

    if not failed and name in _ITEM_READ_TOOLS and isinstance(args.get("item_id"), str):
        candidates.add(args["item_id"])
    if not failed and name in {"smoke_test_feature", "test_helper"}:
        for item_id in args.get("item_ids") or []:
            if isinstance(item_id, str):
                candidates.add(item_id)
        # Random smoke tests have no explicit IDs. Every shown diagnostic can
        # inform a repair, including errors and abstentions; sample_ids is only
        # a short preview and must not define the exposure set.
        for row in result.get("results") or []:
            if isinstance(row, dict) and isinstance(row.get("item_id"), str) and row["item_id"]:
                candidates.add(row["item_id"])
    if name == "classifier_profiles":
        # Internal paired smoke calls do not pass through invoke_tool. Retain
        # exposure from actual returned outcomes, including a completed baseline
        # when fallback execution failed. Requested IDs alone disclose nothing.
        def collect_diagnostics(value: Any) -> None:
            if isinstance(value, dict):
                for row in value.get("results") or []:
                    if (isinstance(row, dict) and isinstance(row.get("item_id"), str)
                            and row["item_id"]
                            and row.get("status") in {"value", "error", "abstained"}):
                        candidates.add(row["item_id"])
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        collect_diagnostics(child)
            elif isinstance(value, list):
                for child in value:
                    collect_diagnostics(child)

        collect_diagnostics(result)
    # Family proposals cite evidence in their arguments. Merely naming a function
    # or listing its positive IDs does not expose the underlying trace.
    if name == "family_discovery" and (not failed or saved_decision):
        if saved_decision:
            candidates.update(row["item_id"] for row in args.get("evidence") or []
                              if isinstance(row, dict) and isinstance(row.get("item_id"), str)
                              and row["item_id"])
        else:
            collect(args.get("evidence"))
    collect(result)
    if not candidates:
        return
    try:
        exposed = [verified_identity_projection(rt, item_id) for item_id in sorted(candidates)]
    except (AttributeError, OSError, ValueError):
        # Retain exposure even if source metadata is unavailable. Display aliases
        # must never manufacture independent tasks or overwrite verified groups.
        log.warning("Could not resolve task identity for exposed evidence", exc_info=True)
        exposed = [{"_item_id": item_id} for item_id in sorted(candidates)]
    record_discovery_exposure(rt.run.run_dir(), exposed)


def _try_record_tool_exposure(name: str, args: dict[str, Any], result: Any) -> dict | None:
    """Exposure bookkeeping cannot change the outcome of an already completed tool."""
    try:
        _record_tool_exposure(name, args, result)
    except Exception as exc:
        log.exception("Exposure recording failed after tool %s completed", name)
        from autofa.mcp.runtime import get_runtime
        from autofa.session.review_evidence import mark_exposure_history_incomplete

        try:
            mark_exposure_history_incomplete(get_runtime().run.run_dir())
        except Exception:
            log.warning("Could not persist incomplete exposure history", exc_info=True)
        diagnostic = {
            "code": "mcp_exposure_recording_failed",
            "message": f"Tool {name} completed, but its evidence exposure could not be recorded.",
            "context": {"tool": name, "error": str(exc)},
            "remediation": "Use the returned result; do not repeat the operation to repair "
                           "logging. Exposure history is incomplete, so fresh independent "
                           "confirmation cannot be established.",
        }
        get_diagnostics().warning(**diagnostic)
        # The collector aggregates earlier failures; this response describes only
        # the current call and must retain its own tool name and error.
        return diagnostic
    return None


def _with_discovery_recovery(name: str, result: Any) -> Any:
    """Deliver current recovery advice without changing a completed tool's outcome."""
    if (name not in {"submit_function", "evaluate_functions", "judge_function", "pareto_status"}
            or not isinstance(result, dict) or result.get("ok") is not True):
        return result
    if name == "pareto_status" and (
        result.get("goal_met") or result.get("ready_for_final_validation")
    ):
        return result
    try:
        from autofa.mcp.runtime import get_runtime
        from autofa.session.discovery_stall import (
            discovery_stall_recovery,
            observe_discovery_stall,
        )
        from autofa.session.polishing_guard import active_polishing

        rt = get_runtime()
        if active_polishing(rt):
            return result
        recovery = discovery_stall_recovery(observe_discovery_stall(rt))
    except Exception:
        # The evaluation/review and its operation receipt are already committed.
        # An advisory accounting failure must not induce another paid operation.
        log.warning("Could not provide discovery recovery guidance", exc_info=True)
        return result
    return {**result, "discovery_recovery": recovery} if recovery else result


def _with_service_recovery(result: Any) -> Any:
    """Surface an observed outage without turning an API error into a negative label."""
    if not isinstance(result, dict):
        return result
    try:
        from autofa.mcp.runtime import get_runtime
        from autofa.session.detector_service import detector_service_status

        health = detector_service_status(get_runtime().run.run_dir())
        if health.get("infrastructure_blocked"):
            return {**result, "detector_service": health}
    except (OSError, RuntimeError, ValueError, AttributeError):
        # Observation is advisory. Never discard a completed native tool result.
        log.warning("Could not attach detector service guidance", exc_info=False)
    return result


async def invoke_tool(
    name: str, args: dict[str, Any], *, caller: str = "native",
) -> dict:
    from autofa.session.usage import usage_connection_scope

    if caller not in {"native", "driver"}:
        raise ValueError("Unknown invocation actor")
    token = _INVOCATION_ACTOR.set(caller)
    try:
        with usage_connection_scope():
            return await _invoke_tool(name, args)
    finally:
        _INVOCATION_ACTOR.reset(token)


async def _invoke_tool(name: str, args: dict[str, Any]) -> dict:  # noqa: C901
    """Execute a registered tool against this process's runtime and checkpoints."""
    fn = TOOL_CALLABLES.get(name)
    if fn is None:
        return tool_response({"ok": False, "error": f"Unknown AutoFA tool: {name}"}, name)
    if not isinstance(args, dict):
        return tool_response({"ok": False, "error": "Tool arguments must be an object"}, name)
    try:
        inspect.signature(fn).bind(**args)
    except TypeError as exc:
        # Validate before reserving an operation or invoking model-backed work.
        # Both adapters must teach the same executable API, including recovery
        # from arguments guessed in an older saved conversation.
        schema = _tool_input_schema(fn)
        result = {
            "ok": False, "error": f"bad args for {name}: {exc}",
            "allowed_arguments": list(schema["properties"]),
            "required_arguments": schema["required"],
            "argument_schema": schema,
            "next_action": f"Retry {name} with the documented arguments above.",
        }
        if name == "family_discovery":
            from autofa.session.family_discovery import family_argument_guidance

            result.update(family_argument_guidance(args.get("action", "status")))
        return tool_response(result, name)
    tool_completed = False
    try:
        from autofa.mcp.runtime import get_runtime
        from autofa.session.campaign_policy import CampaignBoundaryReached
        from autofa.session.polishing_guard import polishing_tool_guard
        from autofa.session.recovery import execute_operation, track_discovery_worker
        from autofa.session.supervision import (
            SupervisorInterrupted,
            check_lifecycle,
            stop_action,
            supervisor_cancelled,
            supervisor_interruption,
        )

        runtime = get_runtime()
        run_dir = runtime.run.run_dir()
        check_lifecycle(run_dir)
        if invocation_actor() == "native":
            from autofa.session.control import consume_control_requests

            await consume_control_requests(runtime)
            check_lifecycle(run_dir)
        if blocked := polishing_tool_guard(runtime, name, args):
            return tool_response(blocked, name)
        cancellation = Cancellation(lambda: stop_action(run_dir))
        # to_thread copies this context, and the bounded worker pools pass it on.
        # Both harnesses must stop the real work, not just its awaiting callback.
        with cancellation_scope(cancellation):
            worker = asyncio.create_task(asyncio.to_thread(execute_operation, fn, name, args))
        track_discovery_worker(worker, cancellation)
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            if supervisor_cancelled(cancellation):
                return tool_response(supervisor_interruption(run_dir, cancellation).as_result(),
                                     name)
            # A checkpoint can hold the token lock across disk I/O. Keep that
            # wait off the event loop so other tools and cancellation stay live.
            cancelling = asyncio.create_task(asyncio.to_thread(cancellation.cancel))
            # A synchronous HTTP call already sent cannot be revoked safely.
            # Drain that bounded work and child-session cleanup before permitting
            # another turn to use this runtime. Further calls/writes are blocked.
            while not cancelling.done() or not worker.done():
                try:
                    await asyncio.shield(cancelling)
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    log.warning("Interrupted tool %s finished with an error", name, exc_info=True)
                    break
            if worker.done() and not worker.cancelled():
                worker.exception()  # retrieve any exception from an interrupted worker
            raise
    except CampaignBoundaryReached as exc:
        result = exc.as_result()
    except SupervisorInterrupted as exc:
        result = exc.as_result()
    except RunInterruptedError as exc:
        from autofa.mcp.runtime import get_runtime

        result = {"ok": False, "resumable": True, "error": str(exc),
                  "resume_command": f"autofa resume --run {get_runtime().run.run_id}"}
    except Exception as exc:  # pragma: no cover
        log.exception("tool %s crashed", name)
        get_diagnostics().critical(
            "mcp_tool_crash", f"MCP tool {name} crashed: {exc}",
            context={"tool": name, "error": str(exc)},
        )
        result = {"ok": False, "error": f"tool crashed: {exc}"}
    else:
        tool_completed = True
        diagnostic = await asyncio.to_thread(_try_record_tool_exposure, name, args, result)
        if diagnostic is not None and isinstance(result, dict):
            result = {**result, "_exposure_warning": diagnostic}
        result = await asyncio.to_thread(_with_discovery_recovery, name, result)
        from autofa.session.supervision import (
            discovery_steering,
            save_quality_snapshot,
            steering_records,
        )

        if name in {"evaluate_functions", "judge_function", "submit_function", "advance_round"}:
            try:
                await asyncio.to_thread(
                    save_quality_snapshot, runtime, force=name == "advance_round")
            except Exception:
                log.warning("Unable to refresh saved supervision quality", exc_info=True)
    if invocation_actor() == "native":
        result = await asyncio.to_thread(_with_service_recovery, result)
    result = _prepare_tool_presentation(result, name)
    if tool_completed:
        # Budget the final presentation and native guidance before consuming a
        # note; later citation reference metadata could otherwise hide it in an envelope.
        try:
            notes = []
            if invocation_actor() == "native" and isinstance(result, dict):
                queued = await asyncio.to_thread(steering_records, run_dir)
                if any(note.get("state") == "queued" for note in queued):
                    budget = max(0, MCP_HARD_CEILING_CHARS - len(json.dumps(
                        result, default=str, ensure_ascii=False)) - 512)
                    notes = await asyncio.to_thread(discovery_steering, run_dir, deliver=True,
                                                    max_chars=min(8000, budget))
        except (OSError, RuntimeError):
            # A controller can be writing the next note. Preserve the committed tool
            # result; queued notes remain available at the next discovery boundary.
            log.warning("Steering delivery deferred; native tool result retained", exc_info=True)
            notes = []
        if notes and isinstance(result, dict):
            result = {**result, "supervisor_steering": notes,
                      "steering_response_tool": "respond_to_steering"}
    return tool_response(result, name, presentation_prepared=True)


def _tool_input_schema(fn: Callable) -> dict[str, Any]:
    """Describe actual Python parameters for a native tool adapter."""
    from pydantic import TypeAdapter

    hints = get_type_hints(fn)
    properties, required = {}, []
    for param in inspect.signature(fn).parameters.values():
        annotation = hints.get(param.name, Any)
        schema = TypeAdapter(annotation).json_schema()
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
        else:
            schema["default"] = param.default
        properties[param.name] = schema
    if fn is family_discovery:
        from autofa.session.family_discovery import FAMILY_DECISION_SCHEMA, FAMILY_EVIDENCE_SCHEMA

        properties["evidence"] = {"anyOf": [FAMILY_EVIDENCE_SCHEMA, {"type": "null"}],
                                  "default": None}
        properties["decision"] = {
            "anyOf": [FAMILY_DECISION_SCHEMA, {"type": "null"}], "default": None,
            "description": "Structured synthesis disposition. Required for new "
                           "no_coherent_family calls; diagnostics remain author controls.",
        }
        properties["detector_name"]["description"] = (
            "Optional detector reference for record_family or assess_family. Defaults to "
            "family_name or its previously saved link. Use an existing differently named "
            "detector with the same description/mechanism; never duplicate it just for the name."
        )
    if fn is tools_library.submit_function:
        from autofa.session.evidence_prerequisites import EVIDENCE_PREREQUISITE_SCHEMA

        properties["evidence_prerequisites"] = {
            "anyOf": [EVIDENCE_PREREQUISITE_SCHEMA, {"type": "null"}], "default": None,
            "description": "Optional exact source locations for essential claim premises. "
                           "Checked on at most three selected support items before evaluation; "
                           "retention does not establish semantic truth.",
        }
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def codex_tool_specs(allowed_tools: list[str] | None = None) -> list[dict[str, Any]]:
    """Describe actual Python parameters for Codex's dynamic function tools."""
    allowed = None if allowed_tools is None else set(allowed_tools)
    specs = []
    for name, fn in TOOL_CALLABLES.items():
        full_name = f"mcp__autofa__{name}"
        if allowed is not None and full_name not in allowed:
            continue
        specs.append({
            # Codex reserves the mcp__ prefix for actual MCP servers. Its dynamic
            # functions use the names documented throughout AutoFA's prompts.
            "type": "function", "name": name,
            "description": (fn.__doc__ or "").strip() or name,
            "inputSchema": _tool_input_schema(fn),
        })
    return specs


def codex_tool_response(response: dict[str, Any]) -> dict[str, Any]:
    """Convert bounded MCP text and verified image bytes to Codex content items."""
    content = []
    success = not response.get("isError", False)
    for block in response.get("content", []):
        if block.get("type") == "text":
            content.append({"type": "inputText", "text": block["text"]})
            try:
                payload = json.loads(block["text"])
                if isinstance(payload, dict) and payload.get("ok") is False:
                    success = False
            except (TypeError, ValueError):
                pass
        elif block.get("type") == "image":
            content.append({
                "type": "inputImage",
                "imageUrl": f"data:{block['mimeType']};base64,{block['data']}",
            })
    return {"contentItems": content, "success": success}


def build_mcp_server(name: str = "autofa") -> Any:
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "claude-agent-sdk is required for the agent session. "
            "Install with: pip install claude-agent-sdk"
        ) from e

    wrapped: list[Any] = []
    for tname, fn in TOOL_CALLABLES.items():
        description = (fn.__doc__ or "").strip() or tname
        # Expose actual parameters to both harnesses.
        input_schema = _tool_input_schema(fn)

        @tool(tname, description, input_schema)
        async def _thunk(args: dict, _name: str = tname) -> dict:
            return await invoke_tool(_name, args or {})

        wrapped.append(_thunk)

    return create_sdk_mcp_server(name=name, version="0.1.0", tools=wrapped)


def tool_names() -> list[str]:
    return list(TOOL_CALLABLES.keys())


def tool_response(result: Any, tool_name: str, *, presentation_prepared: bool = False) -> dict:
    """Keep actual image content out of the JSON text budget and overflow archive."""
    receipt = result.get("_artifact_image") if isinstance(result, dict) else None
    payload = ({key: value for key, value in result.items() if key != "_artifact_image"}
               if receipt else result)
    content = [{"type": "text", "text": clamp_mcp_payload(
        payload, tool_name, presentation_prepared=presentation_prepared)}]
    if receipt:
        from autofa.data.artifact_access import image_content

        content.append(image_content(receipt))
    return {"content": content}
