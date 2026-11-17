# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered rollout digest within a judge character budget."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .outcome_evidence import empty_outcome_evidence

DEFAULT_JUDGEMENT_CHARACTER_BUDGET = 512_000
ERROR_STATUSES = frozenset({"error", "failed", "failure"})
_MODEL_ROLES = frozenset({"assistant", "agent"})
_KEYSTROKE_SCOPE_CHARS = 240
_PROTECTED_PREVIEW_CHARS = 256
_AUDIT_ONLY_KEYS = frozenset({"budget", "sha256", "evidence_receipt"})
_VERIFIER_FAILURE_PATTERN = re.compile(
    r"\b(fail(?:ed|ure)?|error|exception|expected|actual|traceback|assertion)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Profile:
    """Preview sizes for one compaction level."""

    head: int
    tail: int
    protected_limit: int

    def preview(self, text: str, *, widen: bool = False) -> dict[str, Any]:
        minimum = _PROTECTED_PREVIEW_CHARS if widen else 0
        return _preview(text, head=max(self.head, minimum), tail=max(self.tail, minimum))


_COMPACTION_PROFILES = (
    _Profile(24_000, 8_000, 64_000),
    _Profile(12_000, 4_000, 32_000),
    _Profile(6_000, 2_000, 16_000),
    _Profile(480, 160, 12_000),
    _Profile(320, 120, 8_000),
    _Profile(160, 64, 4_000),
    _Profile(80, 32, 2_000),
    _Profile(40, 16, 512),
)
TRACE_COMPACTION_LEVEL_COUNT = len(_COMPACTION_PROFILES)


@dataclass(frozen=True)
class _Call:
    raw: dict[str, Any]
    lane: str
    tool_name: str
    status: str
    args: str
    result: str
    event_idx: Any
    call_idx: Any
    result_event_idx: Any

    @property
    def is_error(self) -> bool:
        return self.status in ERROR_STATUSES

    @property
    def signature(self) -> tuple[str, str, str, str]:
        return (self.lane, self.tool_name, self.args, self.result)


def serialized_characters(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def build_trace_digest(
    *,
    rollout: dict[str, Any],
    tool_calls: Iterable[dict[str, Any]],
    events: Iterable[dict[str, Any]],
    verifier_artifacts: Iterable[dict[str, Any]] = (),
    outcome_evidence: dict[str, Any] | None = None,
    evidence_receipt: dict[str, Any] | None = None,
    auxiliary_events: Iterable[dict[str, Any]] = (),
    auxiliary_tool_calls: Iterable[dict[str, Any]] = (),
    trace_resources: dict[str, Iterable[Any]] | None = None,
    max_serialized_characters: int = DEFAULT_JUDGEMENT_CHARACTER_BUDGET,
    compact_to_fit: bool = True,
) -> dict[str, Any]:
    """Return a digest, optionally compacting before declaring it windowable."""
    ordered_events = sorted(
        (dict(event) for event in events if isinstance(event, dict)),
        key=_event_sort_key,
    )
    events_by_key = {
        key: event for event in ordered_events if (key := _event_key(event)) is not None
    }
    calls = [
        _parse_call(call, events_by_key)
        for call in sorted(
            (call for call in tool_calls if isinstance(call, dict)),
            key=lambda call: (
                *_event_sort_key(call),
                _int_or_default(call.get("call_idx")),
            ),
        )
    ]
    artifacts = [artifact for artifact in verifier_artifacts if isinstance(artifact, dict)]
    auxiliary_events_by_lane: dict[str, list[dict[str, Any]]] = {}
    for event in auxiliary_events:
        if not isinstance(event, dict):
            continue
        lane = str(event.get("lane") or "")
        auxiliary_events_by_lane.setdefault(lane, []).append(dict(event))
    auxiliary_calls_by_lane: dict[str, list[dict[str, Any]]] = {}
    for call in auxiliary_tool_calls:
        if not isinstance(call, dict):
            continue
        lane = str(call.get("lane") or "")
        auxiliary_calls_by_lane.setdefault(lane, []).append(dict(call))
    resources = {
        str(name): list(values)
        for name, values in (trace_resources or {}).items()
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes, dict))
    }
    levels = range(TRACE_COMPACTION_LEVEL_COUNT + 1) if compact_to_fit else (0,)
    for level in levels:
        profile = None if level == 0 else _COMPACTION_PROFILES[level - 1]
        digest = _build_digest(
            rollout=rollout,
            calls=calls,
            events=ordered_events,
            artifacts=artifacts,
            outcome_evidence=outcome_evidence or empty_outcome_evidence(),
            evidence_receipt=evidence_receipt or {},
            profile=profile,
        )
        digest["auxiliary_lanes"] = [
            _auxiliary_lane_digest(
                lane=lane,
                events=auxiliary_events_by_lane.get(lane, []),
                tool_calls=auxiliary_calls_by_lane.get(lane, []),
                profile=profile,
            )
            for lane in sorted(auxiliary_events_by_lane.keys() | auxiliary_calls_by_lane.keys())
        ]
        digest["trace_resources"] = _resource_digest(resources, profile)
        digest["coverage"]["auxiliary_lane_count"] = len(digest["auxiliary_lanes"])
        digest["coverage"]["auxiliary_event_count"] = sum(
            lane["coverage"]["observed_events"] for lane in digest["auxiliary_lanes"]
        )
        digest["coverage"]["trace_resource_count"] = sum(
            len(values) for values in digest["trace_resources"].values()
        )
        judgement_characters = serialized_characters(build_trace_judgement_context(digest))
        if judgement_characters <= max_serialized_characters:
            break
    digest["budget"] = {
        "serialized_characters": serialized_characters(digest),
        "judgement_serialized_characters": judgement_characters,
        "target_serialized_characters": max_serialized_characters,
        "compaction_level": level,
        "requires_windowing": judgement_characters > max_serialized_characters,
    }
    return digest


def _auxiliary_lane_digest(
    *,
    lane: str,
    events: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    profile: _Profile | None,
) -> dict[str, Any]:
    ordered_events = sorted(events, key=_event_sort_key)
    events_by_key = {
        key: event for event in ordered_events if (key := _event_key(event)) is not None
    }
    calls = [
        _parse_call(call, events_by_key)
        for call in sorted(
            tool_calls,
            key=lambda call: (*_event_sort_key(call), _int_or_default(call.get("call_idx"))),
        )
    ]
    lane_digest = _build_digest(
        rollout={"n_events": len(ordered_events), "n_tool_calls": len(calls)},
        calls=calls,
        events=ordered_events,
        artifacts=[],
        outcome_evidence=empty_outcome_evidence(),
        evidence_receipt={},
        profile=profile,
    )
    return {
        "lane": lane,
        "ordering": "independent",
        "coverage": {
            key: lane_digest["coverage"][key]
            for key in (
                "observed_events",
                "observed_tool_calls",
                "error_call_count",
            )
        },
        "error_extract": lane_digest["error_extract"],
        "ordered_trace": lane_digest["ordered_trace"],
        "unattached_tool_calls": lane_digest["unattached_tool_calls"],
    }


def _resource_digest(
    resources: dict[str, list[Any]],
    profile: _Profile | None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            {
                "resource_idx": index,
                "value": _compact_resource_value(value, profile),
            }
            for index, value in enumerate(values)
        ]
        for name, values in sorted(resources.items())
        if values
    }


def _compact_resource_value(value: Any, profile: _Profile | None) -> Any:
    if profile is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= profile.protected_limit else profile.preview(value)
    if isinstance(value, list):
        return [_compact_resource_value(item, profile) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _compact_resource_value(item, profile)
            for key, item in value.items()
        }
    return value


def build_trace_judgement_context(digest: dict[str, Any]) -> dict[str, Any]:
    """Return the digest without audit-only fields."""
    if not isinstance(digest, dict):
        raise TypeError("digest must be a dictionary")

    def project(value: Any, *, source_evidence: bool = False) -> Any:
        if isinstance(value, list):
            return [project(item, source_evidence=source_evidence) for item in value]
        if isinstance(value, dict):
            return {
                key: project(
                    item,
                    source_evidence=source_evidence or key == "outcome_evidence",
                )
                for key, item in value.items()
                if source_evidence or key not in _AUDIT_ONLY_KEYS
            }
        return value

    return project(digest)


def _first_user_events(events: list[dict[str, Any]]) -> dict[str, int]:
    first: dict[str, int] = {}
    for event in events:
        lane, index = str(event.get("lane") or ""), event.get("event_idx")
        if event.get("role") == "user" and type(index) is int and lane not in first:
            first[lane] = index
    return first


def _is_pre_task_model_event(event: dict[str, Any], first_user_by_lane: dict[str, int]) -> bool:
    lane, index = str(event.get("lane") or ""), event.get("event_idx")
    first_user = first_user_by_lane.get(lane)
    return (
        event.get("role") in _MODEL_ROLES
        and type(index) is int
        and first_user is not None
        and index < first_user
    )


def _build_digest(
    *,
    rollout: dict[str, Any],
    calls: list[_Call],
    events: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    outcome_evidence: dict[str, Any],
    evidence_receipt: dict[str, Any],
    profile: _Profile | None,
) -> dict[str, Any]:
    failures = _failure_groups(calls)
    first_user_by_lane = _first_user_events(events)
    event_keys = {
        key
        for event in events
        if (key := _event_key(event)) is not None
    }

    def event_phase(event: dict[str, Any]) -> str | None:
        if _is_pre_task_model_event(event, first_user_by_lane):
            return "pre_task_context"
        return None

    calls_by_event: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for call in calls:
        event_key = _event_key(call.raw)
        if event_key in event_keys:
            record = _call_record(call, failures, profile)
            calls_by_event.setdefault(event_key, []).append(record)
    unattached_calls = [
        _call_record(call, failures, profile)
        for call in calls
        if _event_key(call.raw) not in event_keys
    ]
    event_text = _EventText.for_trace(events, failures, profile)
    verifier_evidence = [_verifier_entry(artifact, profile) for artifact in artifacts]
    return {
        "coverage": {
            "declared_events": rollout.get("n_events"),
            "observed_events": len(events),
            "declared_tool_calls": rollout.get("n_tool_calls"),
            "observed_tool_calls": len(calls),
            "error_call_count": sum(len(group) for group in failures.values()),
            "verifier_artifact_count": len(verifier_evidence),
            "verifier_artifact_read_failures": sum(
                bool(artifact["text_error"]) for artifact in verifier_evidence
            ),
            "outcome_evidence_items": len(outcome_evidence.get("items") or []),
        },
        "error_extract": [_error_extract(group) for group in failures.values()],
        "ordered_trace": [
            {
                "event_ref": {
                    "lane": str(event.get("lane") or ""),
                    "event_idx": event.get("event_idx"),
                    "kind": event.get("kind"),
                    "role": event.get("role"),
                    **({"phase": phase} if (phase := event_phase(event)) else {}),
                },
                **event_text.render(event),
                "tool_calls": calls_by_event.get(_event_key(event), []),
            }
            for event in events
        ],
        "unattached_tool_calls": unattached_calls,
        "outcome_evidence": outcome_evidence,
        "evidence_receipt": evidence_receipt,
        "verifier_artifacts": verifier_evidence,
    }


@dataclass(frozen=True)
class _EventText:
    """Set event text limits for one compaction level."""

    profile: _Profile | None
    failure_boundaries: frozenset[tuple[str, int]]
    failure_results: dict[tuple[str, int], str]
    terminal_ref: tuple[str, int] | None

    @classmethod
    def for_trace(
        cls,
        events: list[dict[str, Any]],
        failures: dict[tuple[str, str, str, str], list[_Call]],
        profile: _Profile | None,
    ) -> _EventText:
        first_user_by_lane = _first_user_events(events)
        model_events = [
            event
            for event in events
            if event.get("role") in _MODEL_ROLES
            and not _is_pre_task_model_event(event, first_user_by_lane)
        ]
        return cls(
            profile=profile,
            failure_boundaries=_failure_boundaries(failures, events),
            failure_results={
                (call.lane, call.result_event_idx): call.result
                for group in failures.values()
                for call in group
                if isinstance(call.result_event_idx, int)
            },
            terminal_ref=_event_key(model_events[-1]) if model_events else None,
        )

    def render(self, event: dict[str, Any]) -> dict[str, Any]:
        text = _event_text(event)
        if self._repeats_failure_result(event, text):
            omitted = {"omitted": "duplicate of the tool-call result", "characters": len(text)}
            return {"text": omitted, "full_reason": "duplicate_tool_result"}
        if self.profile is None:
            return {"text": _full_text(text), "full_reason": "uncompacted"}
        protected = self._is_protected(event)
        if protected and len(text) <= self.profile.protected_limit:
            return {"text": _full_text(text), "full_reason": "task_or_terminal"}
        if _event_key(event) in self.failure_boundaries:
            return {
                "text": _protected_preview(text, self.profile),
                "full_reason": "failure_boundary",
            }
        return {"text": self.profile.preview(text, widen=protected), "full_reason": None}

    def _repeats_failure_result(self, event: dict[str, Any], text: str) -> bool:
        return (
            event.get("role") == "tool"
            and bool(text)
            and self.failure_results.get(_event_key(event)) == text
        )

    def _is_protected(self, event: dict[str, Any]) -> bool:
        """Task instructions, user text, and the final task-time model message."""
        role = event.get("role")
        if role == "user" or _event_key(event) == self.terminal_ref:
            return True
        return role in {"system", "developer"}


def _parse_call(
    call: dict[str, Any],
    events_by_key: dict[tuple[str, int], dict[str, Any]],
) -> _Call:
    result_idx = call.get("result_event_idx")
    result_key = (str(call.get("lane") or ""), result_idx) if isinstance(result_idx, int) else None
    result_event = events_by_key.get(result_key) if result_key is not None else None
    return _Call(
        raw=call,
        lane=str(call.get("lane") or ""),
        tool_name=str(call.get("tool_name") or ""),
        status=str(call.get("status") or "unknown").lower(),
        args=_canonical(call.get("args_full")),
        result=_event_text(result_event) if result_event else "",
        event_idx=call.get("event_idx"),
        call_idx=call.get("call_idx"),
        result_event_idx=call.get("result_event_idx"),
    )


def _event_key(value: dict[str, Any]) -> tuple[str, int] | None:
    event_idx = value.get("event_idx")
    if not isinstance(event_idx, int):
        return None
    return str(value.get("lane") or ""), event_idx


def _event_sort_key(value: dict[str, Any]) -> tuple[int, str]:
    return _int_or_default(value.get("event_idx")), str(value.get("lane") or "")


def _int_or_default(value: Any, default: int = -1) -> int:
    return value if isinstance(value, int) else default


def _failure_groups(calls: list[_Call]) -> dict[tuple[str, str, str, str], list[_Call]]:
    """Group identical failed calls."""
    groups: dict[tuple[str, str, str, str], list[_Call]] = {}
    for call in calls:
        if call.is_error:
            groups.setdefault(call.signature, []).append(call)
    return groups


def _failure_boundaries(
    failures: dict[tuple[str, str, str, str], list[_Call]], events: list[dict[str, Any]]
) -> frozenset[tuple[str, int]]:
    """Keep evidence around repeated failures."""
    indexes: set[tuple[str, int]] = set()
    for group in failures.values():
        first, last = group[0], group[-1]
        for call in (first, last):
            indexes.update(
                (call.lane, idx)
                for idx in (call.event_idx, call.result_event_idx)
                if isinstance(idx, int)
            )
        reply = _next_model_event(events, first.lane, first.result_event_idx)
        if reply is not None and reply not in {call.event_idx for call in group[1:-1]}:
            indexes.add((first.lane, reply))
    return frozenset(indexes)


def _next_model_event(
    events: list[dict[str, Any]], lane: str, after_idx: Any
) -> int | None:
    if not isinstance(after_idx, int):
        return None
    return next(
        (
            event["event_idx"]
            for event in events
            if isinstance(event.get("event_idx"), int)
            and str(event.get("lane") or "") == lane
            and event["event_idx"] > after_idx
            and event.get("role") in _MODEL_ROLES
        ),
        None,
    )


def _error_extract(group: list[_Call]) -> dict[str, Any]:
    first = group[0]
    return {
        "tool_name": first.tool_name,
        "event_idx": first.event_idx,
        "result_event_idx": first.result_event_idx,
        "repeat_count": len(group),
        "event_idxs": [call.event_idx for call in group],
    }


def _call_record(
    call: _Call,
    failures: dict[tuple[str, str, str, str], list[_Call]],
    profile: _Profile | None,
) -> dict[str, Any]:
    scope = _tool_scope(call.raw)
    if profile is None:
        body = {"args": _full_text(call.args), "result": _full_text(call.result)}
    elif call.is_error:
        body = _failure_body(call, failures[call.signature], profile)
    else:
        body = _success_body(call, scope, profile)
    return {
        "event_idx": call.event_idx,
        "call_idx": call.call_idx,
        "tool_name": call.tool_name,
        "status": call.status,
        "result_event_idx": call.result_event_idx,
        "scope": scope,
        **body,
    }


def _success_body(call: _Call, scope: dict[str, Any] | None, profile: _Profile) -> dict[str, Any]:
    if scope and "keystrokes" in scope:
        args = {"represented_by": "scope", "characters": len(call.args)}
    else:
        args = profile.preview(call.args)
    if isinstance(call.result_event_idx, int):
        result = {"event_ref": call.result_event_idx, "characters": len(call.result)}
    else:
        result = profile.preview(call.result)
    return {"args": args, "result": result}


def _failure_body(call: _Call, group: list[_Call], profile: _Profile) -> dict[str, Any]:
    """Keep bounded first and last copies; point middle copies at the first."""
    first = group[0]
    if call is first:
        marker = {"repeat_count": len(group)}
    else:
        marker = {"same_failure_as_event_idx": first.event_idx}
    if call is first or call is group[-1]:
        return {
            **marker,
            "args": _protected_preview(call.args, profile),
            "result": _protected_preview(call.result, profile),
        }
    return {
        **marker,
        "args": {"same_as_event_idx": first.event_idx, "characters": len(call.args)},
        "result": {"same_as_event_idx": first.result_event_idx, "characters": len(call.result)},
    }


def _verifier_entry(artifact: dict[str, Any], profile: _Profile | None) -> dict[str, Any]:
    text = _canonical(artifact.get("text"))
    failure_lines = "\n".join(
        line for line in text.splitlines() if _VERIFIER_FAILURE_PATTERN.search(line)
    )
    if profile is None or (failure_lines and len(text) <= profile.protected_limit):
        text_value = _full_text(text)
    else:
        text_value = profile.preview(text)
    if not failure_lines:
        failure_extract = None
    elif profile is None or len(failure_lines) <= profile.protected_limit:
        failure_extract = _full_text(failure_lines)
    else:
        failure_extract = _protected_preview(failure_lines, profile)
    return {
        "artifact_idx": artifact.get("artifact_idx"),
        "name": artifact.get("name"),
        "kind": artifact.get("kind"),
        "role": artifact.get("role"),
        "text": text_value,
        "failure_extract": failure_extract,
        "text_error": artifact.get("text_error"),
    }


def _protected_preview(text: str, profile: _Profile) -> dict[str, Any]:
    if len(text) <= profile.protected_limit:
        return _full_text(text)
    head = int(profile.protected_limit * 0.75)
    return _preview(text, head=head, tail=profile.protected_limit - head)


def _tool_scope(call: dict[str, Any]) -> dict[str, Any] | None:
    """Keep the action target during compaction."""
    arguments = call.get("args_full")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    scope: dict[str, Any] = {}
    if isinstance(arguments.get("path"), str):
        scope["path"] = arguments["path"]
    if isinstance(arguments.get("command"), str):
        command = arguments["command"]
        scope["operation"] = command[:_KEYSTROKE_SCOPE_CHARS]
        if len(command) > _KEYSTROKE_SCOPE_CHARS:
            scope["operation_truncated"] = True
    keystrokes = arguments.get("keystrokes")
    if isinstance(keystrokes, str) and keystrokes.strip():
        scope["keystrokes"] = keystrokes[:_KEYSTROKE_SCOPE_CHARS]
        if len(keystrokes) > _KEYSTROKE_SCOPE_CHARS:
            scope["keystrokes_truncated"] = True
    return scope or None


def _event_text(event: dict[str, Any]) -> str:
    parts = event.get("parts")
    if not isinstance(parts, list):
        return ""
    return "\n".join(
        str(part["text"])
        for part in parts
        if isinstance(part, dict) and part.get("text") is not None
    )


def _canonical(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _full_text(value: str) -> dict[str, Any]:
    return {"full": value, "characters": len(value), "sha256": _sha256(value)}


def _preview(value: str, *, head: int, tail: int) -> dict[str, Any]:
    if len(value) <= head + tail:
        return {
            "value": value,
            "truncated": False,
            "characters": len(value),
            "sha256": _sha256(value),
        }
    return {
        "head": value[:head],
        "tail": value[-tail:] if tail else "",
        "truncated": True,
        "characters": len(value),
        "sha256": _sha256(value),
    }
