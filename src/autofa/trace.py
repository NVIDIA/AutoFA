# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sandbox-safe trajectory helpers for detectors.

Importable inside the detector sandbox as ``autofa.trace``. Parses the
trajectory embedded in ``response_text`` / ``generation`` on demand.

Also provides ``evidence_text`` / ``text_window`` so LLM detectors pick a
deliberate evidence window instead of inventing ``text[-4000:]``. Step-based
closing/opening selection is used when a trajectory is present; otherwise the
helpers fall back to character windows on whatever surface is available — they
work on non-agentic corpora too.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

from autofa.data.input_view import parsed_input, response_value
from autofa.data.structured import decode_container, tool_call
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trajectory import (
    _obs_to_str,
    _step_reasoning,
    extract_tool_call,
    format_actions,
    parse_trajectory,
)
from autofa.llm.windowing import (
    DEFAULT_MAX_CHARS,
    WindowEvidence,
    text_window,
    text_window_evidence,
)
from autofa.trace_commands import command_prefix
from autofa.trace_types import (
    ArtifactEvidence,
    ArtifactImage,
    ArtifactSegment,
    ArtifactTranscript,
    CommandAnchor,
    CommandFragment,
    CommandPrefix,
    ContentBlock,
    EventRequestText,
    ExecutionMetadata,
    LocatedToolCall,
    Message,
    MessageContent,
    ModelCall,
    ModelCallMetadata,
    ParseFailure,
    ParseSuccess,
    ParseSummary,
    Passage,
    PassageExtraction,
    PromptSummary,
    RawValue,
    RecordedPromptMessage,
    RecordedToolDefinition,
    RenderedPassages,
    ResourceCoverage,
    SourceEvent,
    SourceToolCall,
    StepEvidence,
    ToolCall,
    ToolCount,
    ToolDefinition,
    ToolFunction,
    TraceArtifact,
    TraceAttempt,
    TraceCoverage,
    TraceEvent,
    TracePart,
    TraceProvenance,
    TraceRecord,
    TraceRelation,
)

__all__ = [
    "Step",
    "steps",
    "command_prefix",
    "CommandPrefix",
    "CommandFragment",
    "CommandAnchor",
    "reasoning",
    "actions",
    "observations",
    "tool_calls",
    "final_message",
    "parse",
    "input_messages",
    "input_tools",
    "input_text",
    "model_calls",
    "text_window",
    "text_window_evidence",
    "WindowEvidence",
    "evidence_text",
    "step_evidence",
    "StepEvidence",
    "regex_passages",
    "render_passages",
    "record",
    "events",
    "metadata",
    "attempts",
    "artifacts",
    "artifact",
    "artifact_image",
    "artifact_transcript",
    "ArtifactEvidence",
    "ArtifactImage",
    "ArtifactSegment",
    "ArtifactTranscript",
    "tool_definitions",
    "prompt_messages",
    "prompt_summaries",
    "provenance",
    "relations",
    "coverage",
    "RawValue",
    "ContentBlock",
    "MessageContent",
    "Message",
    "ToolFunction",
    "ToolDefinition",
    "ToolCall",
    "LocatedToolCall",
    "SourceEvent",
    "ModelCall",
    "ToolCount",
    "ParseSuccess",
    "ParseFailure",
    "ParseSummary",
    "Passage",
    "PassageExtraction",
    "RenderedPassages",
    "ExecutionMetadata",
    "TracePart",
    "TraceEvent",
    "SourceToolCall",
    "TraceAttempt",
    "TraceArtifact",
    "RecordedToolDefinition",
    "TraceProvenance",
    "TraceRelation",
    "ResourceCoverage",
    "TraceCoverage",
    "TraceRecord",
    "ModelCallMetadata",
    "EventRequestText",
    "PromptSummary",
    "RecordedPromptMessage",
]

_FOCUS_ALIASES = {
    "closing": "closing",
    "close": "closing",
    "end": "closing",
    "ending": "closing",
    "opening": "opening",
    "open": "opening",
    "start": "opening",
    "beginning": "opening",
}


@dataclass(frozen=True)
class Step:
    """One parsed event, accessed with attributes (step.reasoning, not step['reasoning']).

    Frozen dataclass: attribute assignment is disabled, but nested dictionaries
    are ordinary mutable values. All roles are included, not only assistant turns.
    Missing readable content is an empty string; unavailable identities are None.
    """

    index: Annotated[int, "1-based position in steps(item), not an external viewer event ID."]
    source: Annotated[str, "Normalized event role, usually assistant/agent/user/system/tool."]
    reasoning: Annotated[str, "Full available readable reasoning for this event, or ''."]
    message: Annotated[str, "Full available readable message for this event, or ''."]
    observation: Annotated[str, "Readable tool result, or ''."]
    tool_names: Annotated[tuple[str, ...], "Tool names, in the same order as tool_calls."]
    tool_calls: Annotated[tuple[ToolCall, ...], "Normalized calls belonging to this event."]
    observation_data: Annotated[
        RawValue, "Decoded structured result or unchanged raw value.",
    ] = None
    step_id: Annotated[
        RawValue, "Original source step identity; no assumed type or numbering.",
    ] = None
    source_event: Annotated[SourceEvent | None, "Verified viewer identity when available."] = None
    paired_step_index: Annotated[
        int | None, "1-based calling step associated with this result.",
    ] = None
    result_step_indices: Annotated[
        tuple[int, ...], "1-based result steps associated with this call.",
    ] = ()
    metadata: Annotated[
        ExecutionMetadata | RawValue, "Original event/header fields, including legacy metadata.",
    ] = field(default_factory=dict)
    parts: Annotated[
        tuple[TracePart | RawValue, ...], "Original event parts with recorded text/media metadata.",
    ] = ()

    @property
    def has_reasoning(self) -> bool:
        """Whether the readable reasoning string is nonempty."""
        return bool(self.reasoning)

    @property
    def has_observation(self) -> bool:
        """Whether the readable observation string is nonempty."""
        return bool(self.observation)


def _blob(item: dict[str, Any]) -> Any:
    return response_value(item) if isinstance(item, dict) else ""


def _parsed_steps(item: dict[str, Any]) -> list[dict[str, Any]]:
    blob = _blob(item)
    if not blob:
        return []
    parsed = parse_trajectory(blob)
    if not parsed.ok:
        return []
    return list(parsed._steps or [])


_EXECUTION_RESOURCES = (
    "rollout", "events", "tool_calls", "attempts", "artifacts", "tool_definitions",
    "provenance", "relations",
)
_NO_UNPARSED = object()


def _execution(item: dict[str, Any]) -> dict[str, Any]:
    envelope = decode_container(_blob(item))
    if isinstance(envelope, list) and len(envelope) == 1:
        envelope = envelope[0]
    value = envelope.get("execution") if isinstance(envelope, dict) else None
    return value if isinstance(value, dict) else {}


def _stored_resource(
    item: dict[str, Any], execution: dict[str, Any], name: str,
) -> tuple[Any, ResourceCoverage, Any]:
    """Read one resource without copying unrelated large content."""
    statuses = execution.get("availability")
    status = statuses.get(name) if isinstance(statuses, dict) else None
    supplied = (isinstance(status, dict)
                and status.get("status") in {"available", "unavailable", "partial"})
    if name in execution:
        value = execution[name]
        valid = isinstance(value, dict if name == "rollout" else list)
        if valid and name != "rollout":
            valid = all(isinstance(entry, dict) for entry in value)
        if valid:
            return value, (status if supplied else {"status": "available"}), _NO_UNPARSED
        kept = ([entry for entry in value if isinstance(entry, dict)]
                if isinstance(value, list) and name != "rollout" else
                {} if name == "rollout" else [])
        return kept, {
            "status": "partial" if kept else "unavailable",
            "reason": "Unsupported resource shape; original retained in unparsed_resources.",
        }, value
    if name in {"events", "tool_calls"} and not supplied:
        parsed = _parsed_steps(item)
        if parsed:
            if name == "events":
                value = [{**step, "step_index": index} for index, step in enumerate(parsed, 1)]
            else:
                value = [{**call, "step": index} for index, step in enumerate(parsed, 1)
                         for call in step.get("tool_calls") or [] if isinstance(call, dict)]
            return value, {
                "status": "partial",
                "reason": "Legacy parsed-step projection; original records were not imported.",
            }, _NO_UNPARSED
    return ({} if name == "rollout" else []), (
        status if supplied and status["status"] == "unavailable" else {
            "status": "unavailable", "reason": "Resource not recorded in this execution export.",
        }
    ), _NO_UNPARSED


def _resource(
    item: dict[str, Any], execution: dict[str, Any], name: str,
) -> tuple[Any, ResourceCoverage, Any]:
    value, status, original = _stored_resource(item, execution, name)
    if name != "artifacts":
        return value, status, original
    from autofa.data.inline_images import image_inventory

    first_index = 1 + max((row["artifact_idx"] for row in value
                          if type(row.get("artifact_idx")) is int), default=-1)
    images, limitations = image_inventory(item, first_index=first_index)
    if not images and not limitations:
        return value, status, original
    combined = [*value, *images]
    # This extends the catalog with retained input content; it does not claim
    # that an unavailable external artifact inventory was exhaustively imported.
    coverage = {**status, "retained_input_images": len(images),
                "input_image_inventory_complete": not limitations,
                "input_image_limitations": limitations}
    if limitations or status.get("status") != "available":
        coverage["status"] = "partial"
    return combined, cast(ResourceCoverage, coverage), original


def _coverage(item: dict[str, Any], execution: dict[str, Any]) -> TraceCoverage:
    statuses = execution.get("availability")
    result = {}
    if isinstance(statuses, dict):
        for name, status in statuses.items():
            if isinstance(status, dict) and status.get("status") in {
                "available", "unavailable", "partial",
            }:
                result[name] = deepcopy(status)
            else:
                result[name] = {"status": "unavailable", "reason": "Invalid availability record."}
    for name in _EXECUTION_RESOURCES:
        result[name] = deepcopy(_resource(item, execution, name)[1])
    return cast(TraceCoverage, result)


def record(item: dict[str, Any]) -> TraceRecord:
    """Full imported execution, including resources/coverage, with no text budget or network fetch.

    Returns an independent deep copy, preserving source extensions. Legacy items
    get partial parsed events/calls and explicit unavailable resources; no original
    event IDs, timings or empty-source claims are invented. Use coverage to tell
    recorded empty collections from missing data. Unrecognized collection shapes
    are retained under unparsed_resources rather than silently discarded.
    When raw recovery supplies steps, recovered_steps exposes that sequence
    separately from the original API events, which may be empty.
    """
    execution = _execution(item)
    result = {key: deepcopy(value) for key, value in execution.items()
              if key not in _EXECUTION_RESOURCES and key != "availability"}
    unparsed = {}
    for name in _EXECUTION_RESOURCES:
        value, _, original = _resource(item, execution, name)
        result[name] = deepcopy(value)
        if original is not _NO_UNPARSED:
            unparsed[name] = deepcopy(original)
    if unparsed:
        result["unparsed_resources"] = {**result.get("unparsed_resources", {}), **unparsed}
    if execution.get("recovered_steps_source") == "steps":
        result["recovered_steps"] = deepcopy(_parsed_steps(item))
    result["availability"] = _coverage(item, execution)
    return cast(TraceRecord, result)


def events(item: dict[str, Any], *, lane: str | None = None) -> list[TraceEvent]:
    """Original imported events, full parts and metadata, including auxiliary execution lanes.

    lane=None selects all recorded lanes; lane='' selects explicitly recorded
    conversation-lane events. Filtering preserves source order and text. Legacy
    fallback is a partial parsed-step projection with step_index, not invented
    source event IDs; events without a lane cannot match an explicit lane filter.
    Returns independent copies. Check coverage()['events'] before interpreting [].
    """
    value, _, _ = _resource(item, _execution(item), "events")
    return cast(list[TraceEvent], deepcopy(
        [event for event in value if lane is None or event.get("lane") == lane],
    ))


def metadata(item: dict[str, Any]) -> ExecutionMetadata:
    """Copy of imported rollout metadata; {} when unavailable, with status in coverage()."""
    return cast(ExecutionMetadata, deepcopy(_resource(item, _execution(item), "rollout")[0]))


def attempts(item: dict[str, Any]) -> list[TraceAttempt]:
    """Copied execution attempts/errors. [] can be empty or unavailable; see coverage()."""
    return cast(list[TraceAttempt], deepcopy(_resource(item, _execution(item), "attempts")[0]))


def artifacts(item: dict[str, Any]) -> list[TraceArtifact]:
    """Artifact metadata, including retained input images; no URI fetching or text budget.

    Inspect coverage()['artifacts'] and artifact_texts when present. Missing text
    is not an empty file; binary content is not claimed to have been inspected.
    """
    return cast(list[TraceArtifact], deepcopy(_resource(item, _execution(item), "artifacts")[0]))


def artifact(
    item: dict[str, Any], artifact_idx: int, *, member_path: str | None = None,
) -> ArtifactEvidence:
    """Inspect verified original bytes and located text from a retained artifact.

    Select an exact ZIP member with member_path. Check status, complete and limitations;
    unavailable evidence requires abstention. Layout needs artifact_image(). Accepts only
    retained identities, never arbitrary URLs/paths. All consumers share the pinned cache.
    """
    from autofa.data.artifact_access import artifact_evidence

    options = {"member_path": member_path} if member_path is not None else {}
    return cast(ArtifactEvidence, artifact_evidence(item, artifact_idx, **options))


def artifact_image(
    item: dict[str, Any], artifact_idx: int, *, page: int = 1, member_path: str | None = None,
) -> ArtifactImage:
    """Render a 1-based page/image/video frame or 30-second audio signal plot.

    Select ZIP members with member_path. Pass available receipts to
    llm_client.classify(prompt, text, images=[image]) with a vision-capable model.
    Check status/limitations: signal plots are not listening; Office needs LibreOffice.
    Audio/video need a user-built PyAV; a missing reader or codec is an evidence gap.
    Use artifact() for text. Failed image reads never silently fall back to text.
    """
    from autofa.data.artifact_access import artifact_image as read_image

    options = {"member_path": member_path} if member_path is not None else {}
    return cast(ArtifactImage, read_image(item, artifact_idx, page=page, **options))


def artifact_transcript(
    item: dict[str, Any], artifact_idx: int, *, start_seconds: float = 0,
    end_seconds: float | None = None, model: str = "base", member_path: str | None = None,
) -> ArtifactTranscript:
    """Read a local, timestamped speech estimate; never implies sound-quality inspection.

    Use prepare_artifact_support('speech', install=True) in discovery to download the
    selected model first, once the user has installed PyAV and faster-whisper.
    Detector calls never install or download models.
    """
    from autofa.data.artifact_speech_access import artifact_transcript as transcribe

    return cast(ArtifactTranscript, transcribe(
        item, artifact_idx, start_seconds=start_seconds, end_seconds=end_seconds,
        model=model, member_path=member_path,
    ))


def tool_definitions(item: dict[str, Any]) -> list[RecordedToolDefinition]:
    """Copied source definition records and recorded full def_json; [] if unavailable/empty.

    Check coverage(). input_tools instead describes only the initial model request.
    """
    return cast(list[RecordedToolDefinition], deepcopy(
        _resource(item, _execution(item), "tool_definitions")[0],
    ))


def prompt_messages(item: dict[str, Any]) -> list[RecordedPromptMessage]:
    """Copied prompt-message records retained after evaluation filtering."""
    return cast(list[RecordedPromptMessage], deepcopy(
        _resource(item, _execution(item), "prompt_messages")[0],
    ))


def prompt_summaries(item: dict[str, Any]) -> list[PromptSummary]:
    """Copied prompt-summary records; summaries are not exact prompt messages."""
    return cast(list[PromptSummary], deepcopy(
        _resource(item, _execution(item), "prompt_summaries")[0],
    ))


def provenance(item: dict[str, Any]) -> list[TraceProvenance]:
    """Copied imported provenance/source locations; [] if unavailable/empty, per coverage()."""
    return cast(list[TraceProvenance], deepcopy(_resource(item, _execution(item), "provenance")[0]))


def relations(item: dict[str, Any]) -> list[TraceRelation]:
    """Copied imported source relations, without inferring missing links. See coverage()."""
    return cast(list[TraceRelation], deepcopy(_resource(item, _execution(item), "relations")[0]))


def coverage(item: dict[str, Any]) -> TraceCoverage:
    """Copied resource availability: available, partial or unavailable with recorded reasons.

    Distinguishes a recorded empty collection from an unavailable resource and
    legacy parsed projections. Additional text/table coverage keys are preserved.
    Availability describes imported scope, not whether an agent reviewed it.
    """
    return _coverage(item, _execution(item))


def _step_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    # Raw/legacy steps often keep status, timestamps and usage alongside their
    # content, whereas source API imports carry a dedicated metadata dictionary.
    content = {"metadata", "message", "observation", "reasoning_content", "reasoning",
               "thought", "content", "tool_calls", "parts", "source_parts"}
    content.update(key + "_raw" for key in tuple(content))
    result = {key: value for key, value in raw.items() if key not in content}
    supplied = raw.get("metadata")
    if isinstance(supplied, dict):
        result.update(supplied)
    elif "metadata" in raw:
        result["metadata"] = supplied
    return deepcopy(result)


def steps(item: dict[str, Any]) -> list[Step]:
    """Ordered content with stable identities and explicit tool/result relationships.

    Prefer this view for sequence/recovery claims. Combined text helpers remain
    available for simple searches and existing detectors. Returns [] for absent
    or unparseable trajectories. Includes every parsed role without a text budget;
    this does not imply unavailable/encrypted source content was recovered.
    """
    from autofa.data.trace_navigation import trace_events

    events = trace_events(item, include_content=False)
    out: list[Step] = []
    for i, raw in enumerate(_parsed_steps(item), 1):
        tcs_raw = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
        names: list[str] = []
        tcs: list[ToolCall] = []
        for tc in tcs_raw:
            if not isinstance(tc, dict):
                continue
            name, args = extract_tool_call(tc)
            names.append(name)
            tcs.append(cast(ToolCall, tool_call(tc)))
        msg = raw.get("message")
        message = str(msg) if isinstance(msg, str) else ""
        out.append(
            Step(
                index=i,
                source=str(raw.get("source") or ""),
                reasoning=_step_reasoning(raw),
                message=message,
                observation=_obs_to_str(raw.get("observation")),
                tool_names=tuple(names),
                tool_calls=tuple(tcs),
                observation_data=deepcopy(decode_container(raw.get("observation"))),
                step_id=deepcopy(raw.get("step_id")),
                source_event=events[i - 1].get("source_event"),
                paired_step_index=events[i - 1].get("paired_step_index"),
                result_step_indices=tuple(events[i - 1].get("result_step_indices", [])),
                metadata=_step_metadata(raw),
                parts=tuple(deepcopy(raw.get("parts", [])))
                if isinstance(raw.get("parts"), (list, tuple)) else (),
            )
        )
    return out


def reasoning(item: dict[str, Any]) -> str:
    """Combined reasoning text, or ''; no added text budget. Prefer steps for chronology."""
    enriched = derive_trace_fields(item if isinstance(item, dict) else {})
    return str(enriched.get("reasoning_text") or "")


def actions(item: dict[str, Any]) -> str:
    """Combined messages, calls and results, or ''; no added text budget."""
    enriched = derive_trace_fields(item if isinstance(item, dict) else {})
    return str(enriched.get("actions_text") or "")


def observations(item: dict[str, Any]) -> str:
    """Combined readable tool results, or ''; no added text budget."""
    enriched = derive_trace_fields(item if isinstance(item, dict) else {})
    return str(enriched.get("observations_text") or "")


def final_message(item: dict[str, Any]) -> str:
    """Final available assistant message, or '' when unavailable; no added text budget."""
    enriched = derive_trace_fields(item if isinstance(item, dict) else {})
    return str(enriched.get("final_message") or "")


def tool_calls(item: dict[str, Any]) -> list[LocatedToolCall]:
    """Flat normalized calls with 1-based originating step; [] if none are parseable."""
    out: list[LocatedToolCall] = []
    for s in steps(item):
        for tc in s.tool_calls:
            out.append({**tc, "step": s.index})
    return out


def input_messages(item: dict[str, Any]) -> list[Message]:
    """Structured initial messages, or [] when unavailable/unparseable.

    Preserves recognized roles and multimodal blocks; unsupported raw content
    can remain opaque. This is the initial input, not the full execution history.
    """
    return parsed_input(item)["messages"]


def input_tools(item: dict[str, Any]) -> list[ToolDefinition]:
    """Recorded initial tool definitions, or [] when unavailable/unparseable.

    Definition keys vary by provider; schemas and extra fields remain source data.
    """
    return parsed_input(item)["tools"]


def input_text(item: dict[str, Any]) -> str:
    """Readable initial input, or ''; inspect item['input_parse'] for limitations.

    Non-text content may be present in input_messages even when absent here.
    """
    return parsed_input(item)["text"]


def model_calls(
    item: dict[str, Any], *, offset: int = 0, limit: int | None = None,
) -> list[ModelCall]:
    """Expand recorded requests/responses separately from chronological steps.

    Returns [] when model-call evidence is unavailable. offset is 0-based and
    limit=None returns the remainder; negative values clamp to zero. Snapshots
    can repeat earlier context. Invalid pool references raise ValueError rather
    than returning partial evidence. Preserved provider metadata is not validated.
    """
    from autofa.data.structured import input_view
    from autofa.data.structured import messages as parse_messages

    blob = decode_container(_blob(item))
    if isinstance(blob, list) and len(blob) == 1:
        blob = blob[0]
    evidence = blob.get("model_call_evidence") if isinstance(blob, dict) else None
    if not isinstance(evidence, dict):
        return []
    result: list[ModelCall] = []
    pool, definitions = evidence.get("messages", []), evidence.get("tools", [])
    calls = evidence.get("calls", [])
    stop = None if limit is None else max(0, offset) + max(0, limit)
    for call in calls[max(0, offset):stop]:
        out = deepcopy(call)
        try:
            for side in ("request", "response"):
                indices = call.get(side, [])
                if any(type(i) is not int or i < 0 for i in indices):
                    raise IndexError
                out[side], issues = parse_messages(
                    [pool[i] for i in indices], role="user" if side == "request" else "assistant",
                )
                if issues:
                    out.setdefault("limitations", []).extend(issues)
            index = call.get("tools")
            if index is not None and (type(index) is not int or index < 0):
                raise IndexError
            out["tools"] = (
                input_view(None, tools=definitions[index])["tools"] if index is not None else []
            )
        except (IndexError, TypeError, KeyError) as exc:
            raise ValueError("invalid model-call pool references") from exc
        result.append(out)
    return result


def parse(item: dict[str, Any]) -> ParseSummary:
    """Summary discriminated by result['ok']; this does not return event contents.

    Success includes counts and limitations. Missing input returns {'ok': False};
    a failed parse also includes a warning. Use steps(item) for the ordered events.
    """
    blob = _blob(item)
    if not blob:
        return {"ok": False}
    parsed = parse_trajectory(blob)
    if not parsed.ok:
        return {"ok": False, "warning": parsed.warning}
    return {
        "ok": True,
        "dialect": parsed.dialect,
        "n_steps": parsed.n_steps,
        "n_agent_steps": parsed.n_agent_steps,
        "n_tool_calls": parsed.n_tool_calls,
        "tools": [{"name": t.name, "count": t.count} for t in parsed.tools],
        "agent_name": parsed.agent_name,
        "limitations": parsed.limitations,
    }


def _surface_text(item: dict[str, Any], surface: str) -> str:
    """Resolve a named surface; ``auto`` picks the richest available text."""
    it = item if isinstance(item, dict) else {}
    surf = (surface or "auto").strip().lower()
    enriched = derive_trace_fields(it)

    def _get(*keys: str) -> str:
        for k in keys:
            val = enriched.get(k) if k in enriched else it.get(k)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    if surf in {"reasoning", "cot", "reasoning_text"}:
        return _get("reasoning_text", "reasoning_content", "chain_of_thought")
    if surf in {"actions", "actions_text"}:
        return _get("actions_text")
    if surf in {"observations", "observations_text", "obs"}:
        return _get("observations_text")
    if surf in {"final", "final_message"}:
        return _get("final_message")
    if surf in {"prompt", "input", "prompt_text", "input_text"}:
        return _get("input_text", "prompt_text", "question", "input", "prompt", "instruction")
    if surf in {"response", "response_text", "generation"}:
        return _get("response_text", "generation")
    if surf != "auto":
        val = it.get(surf)
        if isinstance(val, str) and val.strip():
            return val
        return ""

    for candidate in (
        "reasoning_text",
        "actions_text",
        "observations_text",
        "final_message",
        "response_text",
        "generation",
        "prompt_text",
    ):
        got = _get(candidate)
        if got:
            return got
    return ""


def _surface_steps(item: dict[str, Any], surface: str) -> list[str]:
    """Select content from the requested field, retaining original step numbers."""
    surf = (surface or "auto").strip().lower()
    aliases = {
        "cot": "reasoning", "reasoning_text": "reasoning",
        "actions_text": "actions", "obs": "observations",
        "observations_text": "observations",
    }
    surf = aliases.get(surf, surf)
    if surf == "auto":
        surf = next((s for s in ("reasoning", "actions", "observations", "final", "response")
                     if _surface_text(item, s)), "response")
    if surf not in {"reasoning", "actions", "observations"}:
        return []
    # A native field can differ from the trajectory; keep detector-visible values.
    field = {"reasoning": "reasoning_text", "actions": "actions_text",
             "observations": "observations_text"}[surf]
    derived = derive_trace_fields(item, overwrite=True).get(field, "")
    if _surface_text(item, surf) != derived:
        return []
    snippets = []
    for i, raw in enumerate(_parsed_steps(item), 1):
        if surf == "reasoning":
            content = _step_reasoning(raw)
        elif surf == "observations":
            content = _obs_to_str(raw.get("observation"))
        else:
            content = format_actions([raw], start_index=i)
        if content.strip():
            snippets.append(content if surf == "actions" else f"[step {i}]\n{content}")
    return snippets


def evidence_text(
    item: dict[str, Any],
    *,
    surface: str = "auto",
    focus: str = "auto",
    max_steps: int = 5,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Build classify-ready evidence for any corpus shape.

    Prefer this over ``text[-4000:]``.

    ``focus``:
      - ``closing`` / ``opening`` — when steps exist, take the last/first
        ``max_steps`` content-bearing steps from the chosen surface; when they do not (non-agentic
        dumps, unparseable trajectories), fall back to a char ``tail`` / ``head``
        window on the chosen surface.
      - ``head`` / ``tail`` / ``ends`` / ``full`` / ``auto`` — character windows
        on the surface (same policy as ``llm_client.classify(..., focus=)``).

    ``surface``: ``reasoning`` | ``actions`` | ``observations`` | ``final`` |
    ``response`` | ``auto`` (richest available field).

    Default budget is 16,000 characters; even focus='full' is budget-limited.
    Missing usable content returns ''. Selection does not establish full-trace review.
    """
    raw_focus = (focus or "auto").strip().lower()
    focus_n = _FOCUS_ALIASES.get(raw_focus, raw_focus)
    max_steps = max(1, int(max_steps))
    max_chars = max(64, int(max_chars))

    if focus_n in {"closing", "opening"}:
        contentful = _surface_steps(item, surface)
        char_focus = "tail" if focus_n == "closing" else "head"
        if contentful:
            selected = contentful[-max_steps:] if focus_n == "closing" else contentful[:max_steps]
            marker = ""
            if len(selected) < len(contentful):
                marker = f"[{len(contentful) - len(selected)} {surface} steps omitted]\n"
            joined = "\n\n".join(selected)
            joined = marker + joined if focus_n == "closing" else joined + "\n" + marker
            return text_window(joined, focus=char_focus, max_chars=max_chars)
        # No usable steps — char fallback on the surface.
        char_focus = "tail" if focus_n == "closing" else "head"
        return text_window(
            _surface_text(item, surface),
            focus=char_focus,
            max_chars=max_chars,
        )

    char_focus = (
        focus_n if focus_n in {"head", "tail", "ends", "full", "auto"} else "auto"
    )
    return text_window(
        _surface_text(item, surface),
        focus=char_focus,
        max_chars=max_chars,
    )


def step_evidence(
    item: dict[str, Any], pattern: str, *, flags: int = 0,
    context_steps: int = 1, closing_steps: int = 3, max_chars: int = 16_000,
) -> StepEvidence:
    """Search all steps; render bounded ordered evidence without LLM calls.

    Search reasoning, message, normalized calls and result. Select matching,
    neighboring, paired call/result and closing steps. Include the artifact name
    in the pattern to select later repairs too. Regex timeouts propagate.

    Oversized steps retain marked exact excerpts near a match or closing ending.
    Excerpt offsets refer to the labeled rendered step, not original fields.
    steps(item)[step_index - 1] reads full structured content; input_text reads
    the task. full_trace covers every parsed step's textual projection;
    selection_complete covers only selected steps. Neither covers attachments,
    images or unrepresented parts, nor guarantees source availability.

    Read omitted/truncated history when the claim needs it. Selected evidence,
    a final message or a regex miss cannot prove semantic absence or no later
    repair; unresolved required facts must remain unknown.
    """
    from autofa.core.sandbox import SAFE_REGEX

    budget = max(256, min(100_000, int(max_chars)))
    seq = steps(item)
    result: StepEvidence = {
        "status": "ok" if seq else "unavailable", "text": "",
        "full_trace": False, "selection_complete": False, "total_steps": len(seq),
        "matched_step_indices": [], "selected_step_indices": [],
        "omitted_step_indices": [], "truncated_step_indices": [],
        "budget_chars": budget, "reason": "" if seq else "Missing or unparseable trajectory",
    }
    if not seq:
        return result
    # Validate once even if every recorded step happens to have empty content.
    expression = SAFE_REGEX.compile(pattern, flags)
    rendered: dict[int, str] = {}
    anchors: dict[int, int] = {}
    for step in seq:
        fields = [("reasoning", step.reasoning), ("message", step.message),
                  ("calls", json.dumps(step.tool_calls, ensure_ascii=False)
                   if step.tool_calls else ""), ("result", step.observation)]
        text = f"[step {step.index}; role={step.source}]\n" + "\n".join(
            f"{name}:\n{value}" for name, value in fields if value
        )
        rendered[step.index] = text
        match = expression.search(text)
        if match is not None:
            anchors[step.index] = match.start()
    matched = set(anchors)
    neighbors = max(0, int(context_steps))
    closing = {step.index for step in seq[max(0, len(seq) - max(0, int(closing_steps))):]}
    selected = closing | matched
    for index in matched:
        selected.update(range(max(1, index - neighbors), min(len(seq), index + neighbors) + 1))
    # Keep the result linked to a selected call, even when it is not adjacent.
    pending = list(selected)
    while pending:
        step = seq[pending.pop() - 1]
        linked = set(step.result_step_indices)
        if step.paired_step_index is not None:
            linked.add(step.paired_step_index)
        linked &= rendered.keys()
        pending.extend(linked - selected)
        selected.update(linked)
    selected &= rendered.keys()
    indices = sorted(selected)
    result["matched_step_indices"] = sorted(matched)
    result["selected_step_indices"] = indices
    whole_size = sum(len(rendered[index]) for index in indices) + max(0, len(indices) - 1) * 2
    # Thousands of selected steps cannot each receive a readable excerpt. Keep
    # some matching and closing evidence, and explicitly list the omitted steps.
    max_rendered = max(1, budget // 256)
    if whole_size > budget and len(indices) > max_rendered:
        priority = ([min(matched)] if matched else []) + ([max(closing)] if closing else [])
        priority += ([max(matched)] if matched else []) + sorted(closing) + indices
        indices = sorted(list(dict.fromkeys(priority))[:max_rendered])
        whole_size = sum(len(rendered[index]) for index in indices) + (len(indices) - 1) * 2
    # A shared cap retains evidence across the selected chronology. Small steps
    # keep their complete text and donate unused space to larger steps.
    cap = budget
    if whole_size > budget and indices:
        lo, hi = 0, budget
        available = max(0, budget - max(0, len(indices) - 1) * 2)
        while lo < hi:
            middle = (lo + hi + 1) // 2
            if sum(min(len(rendered[index]), middle) for index in indices) <= available:
                lo = middle
            else:
                hi = middle - 1
        cap = lo
    chunks: list[str] = []
    emitted: set[int] = set()
    for index in indices:
        text = rendered[index]
        if len(text) > cap:
            # Reserve enough room for the scope marker, whose numeric offsets
            # may be long. A small positive excerpt is preferable to an empty
            # all-or-nothing rejection of the entire trace.
            allowance = cap - 80
            if allowance <= 0:
                continue
            anchor = anchors.get(index, len(text) if index in closing else 0)
            left = max(0, min(len(text) - allowance, anchor - allowance // 3))
            right = min(len(text), left + allowance)
            text = f"[step {index}; excerpt {left}:{right}/{len(text)} chars]\n{text[left:right]}"
            result["truncated_step_indices"].append(index)
        chunks.append(text)
        emitted.add(index)
    result["text"] = "\n\n".join(chunks)
    result["omitted_step_indices"] = sorted(set(rendered) - emitted)
    result["selection_complete"] = not (selected - emitted or result["truncated_step_indices"])
    result["full_trace"] = result["selection_complete"] and not result["omitted_step_indices"]
    if not result["full_trace"]:
        result["reason"] = (
            "Selected evidence only; read omitted or truncated steps when the claim requires them"
        )
    return result


def regex_passages(
    item: dict[str, Any], field: str, pattern: str, *, flags: int = 0,
    context_chars: int = 240, max_passages: int = 12,
) -> PassageExtraction:
    """Extract exact field spans before classification, without assigning a label.

    Empty complete results mean no match. Missing fields and capped match sets are
    explicit; regex timeouts propagate as execution errors. Offsets are Unicode
    characters in the named original field, never offsets in reserialized JSON.
    """
    from autofa.core.localisation import passage, text_hash
    from autofa.core.sandbox import SAFE_REGEX, active_field_policy

    if not active_field_policy().is_allowed(field):
        raise ValueError(f"passage extraction disallows field {field!r}")
    value = item.get(field)
    if not isinstance(value, str):
        return {"status": "unavailable", "complete": False, "value": [],
                "reason": f"missing text field {field}"}
    max_passages = max(1, min(64, int(max_passages)))
    context_chars = max(0, min(4000, int(context_chars)))
    results = []
    fingerprint = text_hash(value)
    complete = True
    for match in SAFE_REGEX.finditer(pattern, value, flags):
        if match.start() == match.end():
            continue
        if len(results) >= max_passages:
            complete = False
            break
        record = passage(field, value, match.start(), match.end(), method="regex_extractor",
                         field_sha256=fingerprint)
        left = max(0, match.start() - context_chars)
        right = min(len(value), match.end() + context_chars)
        # Expand to complete lines when possible, under a bounded extra allowance.
        before = value.rfind("\n", max(0, left - 240), left)
        after = value.find("\n", right, min(len(value), right + 240))
        if before >= 0:
            left = before + 1
        if after >= 0:
            right = after
        record.update(context_start=left, context_end=right, context=value[left:right])
        results.append(record)
    return {"status": "ok", "complete": complete, "value": results,
            "reason": "" if complete else "match limit reached"}


def render_passages(result: PassageExtraction, *, max_chars: int = 16_000) -> RenderedPassages:
    """Render whole extraction contexts within a default 16,000-character budget.

    Never slice a context to fit. complete=False if extraction was incomplete or
    unavailable, or any context was omitted for budget. No matches yield text=''.
    """
    budget = max(64, min(100_000, int(max_chars)))
    rendered = []
    complete = result.get("status") == "ok" and result.get("complete") is True
    size = 0
    for p in result.get("value", []):
        text = f"[{p['field']}:{p['start']}:{p['end']}]\n{p['context']}"
        if size + len(text) + (2 if rendered else 0) > budget:
            complete = False
            break
        rendered.append(text)
        size += len(text) + (2 if len(rendered) > 1 else 0)
    return {"text": "\n\n".join(rendered), "complete": complete, "budget_chars": budget}
