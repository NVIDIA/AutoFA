# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network-free conversion of normalized run tables to detector-visible traces."""

from __future__ import annotations

from collections import Counter
from typing import Any

from autofa.data.execution_policy import is_evaluation_record, sanitize_metadata
from autofa.data.source_completeness import (
    GENERATION_CUTOFF_NOTICE,
    TRUNCATED_EVENT_LIMITATION,
    generation_cutoff,
)
from autofa.data.source_errors import SourceDataError as DataApiError


def _array(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise DataApiError(f"Expected an array of objects from {name}")
    return value


def _count(value: dict, key: str) -> int:
    count = value.get(key)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise DataApiError(f"Source API omitted a valid {key}; completeness is unknown")
    return count


def _validate_identities(metadata: dict, events: list[dict], calls: list[dict]) -> None:
    expected = tuple(metadata.get(k) for k in ("run_id", "task_id", "repeat"))
    for row in [*events, *calls]:
        if tuple(row.get(k) for k in ("run_id", "task_id", "repeat")) != expected:
            raise DataApiError("Source event/call identity mismatch")
        for part in row.get("parts", []):
            if tuple(part.get(k) for k in ("run_id", "task_id", "repeat", "event_idx")) != (
                *expected, row["event_idx"],
            ):
                raise DataApiError("Source part identity mismatch")


def normalize_events(  # noqa: C901
    metadata: dict[str, Any], events: list[dict], calls: list[dict],
    *, conversation_lane: str = "", execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve execution evidence while isolating task outcomes and grader records."""
    if any((row.get("lane") or "") != conversation_lane for row in [*events, *calls]):
        raise DataApiError("Source API returned auxiliary lanes instead of the conversation lane")
    if len(events) != _count(metadata, "n_events"):
        raise DataApiError("Source event count does not match rollout metadata")
    if len(calls) != _count(metadata, "n_tool_calls"):
        raise DataApiError("Source tool-call count does not match rollout metadata")
    indices = [e.get("event_idx") for e in events]
    if any(type(i) is not int for i in indices) or indices != sorted(set(indices)):
        raise DataApiError("Source events have duplicate, invalid, or unordered indices")
    by_event: dict[int, list[dict]] = {}
    call_ids = set()
    for call in calls:
        key = (call.get("event_idx"), call.get("call_idx"))
        if key[0] not in indices or type(key[1]) is not int or key in call_ids:
            raise DataApiError("Source tool call has an invalid or duplicate event/call key")
        call_ids.add(key)
        if not isinstance(call.get("args_full"), str):
            raise DataApiError("Full tool arguments unavailable; args_head is not evidence")
        if len(call["args_full"]) != _count(call, "args_chars"):
            raise DataApiError("Full tool arguments length does not match source metadata")
        by_event.setdefault(key[0], []).append(call)

    limitations: Counter[str] = Counter()
    notices: Counter[str] = Counter()
    from autofa.data.source_snapshots import recover_request_snapshots

    snapshots = recover_request_snapshots(events, lane=conversation_lane)
    snapshot_events = snapshots.pop("event_indices")
    steps = []
    for event in events:
        idx = event["event_idx"]
        parts = _array(event.get("parts"), f"event {idx} parts")
        if len(parts) != _count(event, "n_parts"):
            raise DataApiError(f"Missing source parts at event {idx}")
        # Nullable event call counts do not weaken rollout-level call checks.
        if event.get("n_tool_calls") is not None and (
            len(by_event.get(idx, [])) != _count(event, "n_tool_calls")
        ):
            raise DataApiError(f"Missing source calls at event {idx}")
        part_indices = [p.get("part_idx") for p in parts]
        if idx not in snapshot_events and (
            any(type(i) is not int for i in part_indices) or len(set(part_indices)) != len(parts)
        ):
            raise DataApiError(f"Invalid or duplicate part indices at event {idx}")
        for part in parts:
            if (part.get("lane") or "") != conversation_lane:
                raise DataApiError(f"Source part belongs to an auxiliary lane at event {idx}")
            if part.get("part_type") in {"text", "reasoning"}:
                text = part.get("text")
                if not isinstance(text, str) or len(text) != _count(part, "chars"):
                    raise DataApiError(f"Missing or shortened full text at event {idx}")
        if idx in snapshot_events or event.get("is_copied_context"):
            continue  # Request context is retained separately; never replayed as actions.
        kind, role = event.get("kind"), event.get("role")
        if (kind not in {"system", "message", "reasoning", "tool_call", "tool_result"}
                or is_evaluation_record(event)):
            limitations[f"excluded event kind/actor: {kind}/{event.get('actor')}"] += 1
            continue
        if role == "output" and kind in {"message", "reasoning", "tool_call"}:
            role = "assistant"
        elif kind == "tool_result":
            role = "tool"
        source = role or (
            "tool" if kind == "tool_result" else "system" if kind == "system" else "assistant"
        )
        if source not in {"assistant", "agent", "user", "system", "developer", "tool"}:
            limitations[f"unsupported event role: {source}"] += 1
            continue
        step: dict[str, Any] = {
            "step_id": idx, "source": source,
            "metadata": sanitize_metadata({k: v for k, v in event.items() if k != "parts"}),
            "parts": sanitize_metadata(parts),
        }
        content, reasoning = [], []
        source_parts = []
        for part in sorted(parts, key=lambda p: p["part_idx"]):
            if is_evaluation_record(part):
                continue
            if part.get("part_type") not in {"text", "reasoning"}:
                limitations[f"non-text part not inspected: {part.get('part_type')}"] += 1
                step.setdefault("uninspected_parts", []).append({
                    key: part[key] for key in (
                        "part_idx", "part_type", "media_type", "artifact_idx", "chars",
                    ) if key in part
                })
                continue
            text = part.get("text")
            is_reasoning = (
                part.get("channel") in {"reasoning", "analysis"}
                or part.get("part_type") == "reasoning" or kind == "reasoning"
            )
            target = reasoning if is_reasoning else content
            start = sum(len(t) + 2 for t in target)
            source_parts.append({
                "part_idx": part["part_idx"], "start": start, "end": start + len(text),
                "field": ("reasoning_content" if is_reasoning else
                          "observation" if source == "tool" else "message"),
            })
            target.append(text)
        if reasoning:
            step["reasoning_content"] = "\n\n".join(reasoning)
        if content:
            step["observation" if source == "tool" else "message"] = "\n\n".join(content)
        if source_parts:
            step["source_parts"] = source_parts
        if generation_cutoff(event):
            notices[GENERATION_CUTOFF_NOTICE] += 1
        elif event.get("length_truncated"):
            limitations[TRUNCATED_EVENT_LIMITATION] += 1
        if event.get("has_encrypted_reasoning"):
            limitations["encrypted reasoning unavailable"] += 1
        if by_event.get(idx):
            step["tool_calls"] = [
                {**sanitize_metadata(c), "function_name": c.get("tool_name") or "unknown",
                 "arguments": c["args_full"], "source_call_index": c["call_idx"],
                 "result_event_idx": c.get("result_event_idx")}
                for c in sorted(by_event[idx], key=lambda c: c["call_idx"])
                if not is_evaluation_record(c)
            ]
        if source == "tool":
            step["paired_event_idx"] = event.get("paired_event_idx")
            step["paired_call_idx"] = event.get("paired_call_idx")
        steps.append(step)
    if metadata.get("n_artifacts"):
        limitations["artifact file contents not inspected"] = int(metadata["n_artifacts"])
    if not events:
        limitations["source rollout contains no events"] = 1
    trace = {
        "steps": steps,
        "agent": {"name": metadata.get("agent_ref") or "unknown"},
        "evidence_scope": {
            "source": "original API event parts and tool arguments",
            "event_lane": conversation_lane,
            "excluded_lanes": (
                "Steps follow the selected conversation lane. Other exported execution lanes "
                "are available in execution.events. Grading records are excluded, with the "
                "request messages and tools only they sent: events, parts and tool calls "
                "labelled as grader, judge or evaluator work, and model calls (and their "
                "exported events) that the raw NeMo Gym record attributes to a grader's "
                "model reference (e.g. judge_model)."
            ),
            "limitations": dict(limitations),
            "note": "Complete retrieval is not a claim of complete agent review. "
                    "Target metadata is excluded. Execution metadata and exported artifact text "
                    "remain available; availability records describe missing resources.",
        },
    }
    from autofa.data.source_execution import execution_record

    trace["execution"] = execution_record(metadata, events, calls, execution=execution)
    if notices:
        trace["evidence_scope"]["notices"] = dict(notices)
    if snapshots["calls"]:
        trace["model_call_evidence"] = snapshots
        trace["evidence_scope"]["model_calls"] = (
            "Recorded request snapshots; responses remain in the event trajectory. "
            "Tool definitions were not included in the snapshots."
        )
        first = snapshots["calls"][0]
        if first["initial_request_recorded"]:
            trace["initial_request"] = {
                "messages": [snapshots["messages"][i] for i in first["request"]],
                "limitations": ["tool definitions not recorded in request snapshots"],
            }
    else:
        from autofa.data.source_model_calls import recover_event_requests

        requests = recover_event_requests(
            trace["execution"]["events"],
            prompt_messages=trace["execution"].get("prompt_messages", []),
            tool_definitions=trace["execution"].get("tool_definitions", []),
        )
        capture_limits = requests.pop("limitations", {})
        trace["evidence_scope"]["limitations"].update(capture_limits)
        if capture_limits:
            # Keep the contribution separate so later raw recovery cannot clear
            # unrelated execution limitations with the same wording.
            trace["evidence_scope"]["request_capture_limitations"] = capture_limits
        notices = requests.pop("notices", {})
        if notices:
            trace["evidence_scope"].setdefault("notices", {}).update(notices)
        if requests["calls"]:
            trace["model_call_evidence"] = requests
            trace["evidence_scope"]["model_calls"] = (
                "Full exported request messages, tools, and parameters; source responses "
                "remain in the event trajectory. Request snapshots are not replayed as steps."
            )
            first = next((call for call in requests["calls"]
                          if call.get("source_lane") == conversation_lane
                          and call.get("initial_request_recorded")), None)
            if first is not None:
                trace["initial_request"] = {
                    "messages": [requests["messages"][i] for i in first["request"]],
                    "tools": requests["tools"][first["tools"]],
                }
    return trace
