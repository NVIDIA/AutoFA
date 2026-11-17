# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator outcome evidence for the judge."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from autofa.data.artifact_evidence import (
    is_candidate_patch_artifact,
    is_outcome_text_artifact,
)
from autofa.data.source_errors import SourceDataError

MAX_OUTCOME_EVIDENCE_CHARACTERS = 8_000
MAX_ITEM_TEXT_CHARACTERS = 1_200
_MAX_ENTRIES = 8
_MAX_INVENTORY_ARTIFACTS = 12

_SUMMARY_FIELDS = (
    "reward",
    "score",
    "reason_score",
    "format_error",
    "win",
    "loss",
    "tie",
    "total_wins",
    "total_losses",
    "total_ties",
    "step_results",
    "num_steps_passed",
    "num_steps_total",
    "problem_accuracy",
    "subtask_accuracy",
)
_METRIC_SUMMARY_FIELDS = frozenset(
    {
        "score",
        "reason_score",
        "step_results",
        "num_steps_passed",
        "num_steps_total",
        "problem_accuracy",
        "subtask_accuracy",
    }
)
_REFERENCE_ANSWER_FIELDS = (
    "expected_answer",
    "reference_answer",
    "answer_option",
    "target",
    "bbox",
)
# A bare answer is treated as a reference only when the record also identifies candidate context.
_ANSWER_CONTEXT_FIELDS = ("response", "choices", "candidates", "answer_option")
_REFERENCE_CONTEXT_FIELDS = ("choices", "candidates", "baseline_answer")
_RUBRIC_FIELDS = ("rubric_pretty", "rubric_json")
_RATIONALE_KEYS = ("rationale", "reasoning", "feedback", "explanation", "summary")
_SEMANTICS_PRIORITY = ("comparative", "judge", "reference", "metric")
_EVALUATOR_METADATA_KEYS = (
    "status",
    "error_code",
    "error_message",
    "error",
    "timeout",
    "timed_out",
    "output",
    "actual",
    "expected",
)

_BOXED_VERDICT_RE = re.compile(r"BOXED\[(A|B|TIE)\]", re.IGNORECASE)
_PAIRWISE_VERDICT_RE = re.compile(r"\[\[(A>>B|A>B|A=B|B>A|B>>A|BB)\]\]")
_FAILURE_LINE_RE = re.compile(
    r"\b(fail(?:ed|ure)?|error|exception|assert(?:ion)?|expected|actual|traceback)\b",
    re.IGNORECASE,
)
_BOXED_WINNER = {"A": "A", "B": "B", "TIE": "tie"}
_PAIRWISE_WINNER = {
    "A>>B": "A",
    "A>B": "A",
    "B>A": "B",
    "B>>A": "B",
    "A=B": "tie",
    "BB": "both_bad",
}


@dataclass(frozen=True)
class _Source:
    """Items read from one top-level record field."""

    field: str
    items: list[dict[str, Any]]
    semantics: str | None = None


class OutcomeSource(Protocol):
    """Grader evidence withheld from prepared detector data."""

    def artifacts(self, task_id: str, repeat: int) -> list[dict[str, Any]]:
        """All rollout artifacts; verifier artifacts carry ``text`` or ``text_error``."""

    def read_raw_record(self, run_id: str, source_ref: str) -> dict[str, Any]:
        """The source record at ``path:offset``; raises ``SourceDataError`` when unreadable."""


def empty_outcome_evidence() -> dict[str, Any]:
    return {"semantics": "unknown", "observed_outcome": {}, "items": []}


def load_outcome_evidence_record(
    client: OutcomeSource,
    rollout: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Return projected evidence and the source record used to build it."""
    evidence, receipt, record = _raw_outcome(client, rollout)
    fallback_fields = [
        f"artifact:{artifact.get('name')}"
        for artifact in artifacts
        if isinstance(artifact.get("text"), str)
        and is_outcome_text_artifact(artifact)
    ]
    if record is None and fallback_fields:
        receipt = {
            **receipt,
            "status": "artifact_fallback",
            "fields_found": fallback_fields,
            "artifact_fallback": True,
        }
    patch = next(
        (
            artifact
            for artifact in artifacts
            if is_candidate_patch_artifact(artifact)
            and isinstance(artifact.get("text"), str)
        ),
        None,
    )
    if patch is not None and not any(
        item.get("id") == "candidate-patch" for item in evidence["items"]
    ):
        text = patch["text"]
        evidence["items"].append(
            _item(
                item_id="candidate-patch",
                kind="candidate_patch",
                subject="candidate",
                source_path="rollout.artifacts/model.patch",
                text=text,
                value={
                    "full_characters": len(text),
                    "artifact_fallback": record is None,
                },
            )
        )
        receipt["artifact_fields_found"] = fallback_fields
    if artifacts:
        evidence["items"].append(_artifact_inventory(artifacts))
    bounded, omitted, truncated = _bound_items(evidence["items"])
    evidence["items"] = bounded
    receipt["item_ids_included"] = [item["id"] for item in bounded]
    receipt["items_omitted"] = int(receipt.get("items_omitted") or 0) + omitted
    receipt["truncated"] = bool(receipt.get("truncated")) or truncated
    return evidence, receipt, record


def _raw_outcome(
    client: OutcomeSource, rollout: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    source_ref = rollout.get("raw_record_ref")
    if not isinstance(source_ref, str) or ":" not in source_ref:
        return empty_outcome_evidence(), _unavailable_receipt("absent"), None
    try:
        record = client.read_raw_record(str(rollout.get("run_id") or ""), source_ref)
    except SourceDataError as exc:
        receipt = _unavailable_receipt("error", source_ref=source_ref, error=str(exc))
        return empty_outcome_evidence(), receipt, None
    evidence, receipt = project_outcome_evidence(record, source_ref=source_ref)
    return evidence, receipt, record


def project_outcome_evidence(
    record: dict[str, Any],
    *,
    source_ref: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return known outcome fields of a raw record and an audit receipt."""
    observed = {
        key: _small_value(record[key]) for key in _SUMMARY_FIELDS if record.get(key) is not None
    }
    candidates = (
        *_execution_outcome_sources(record),
        _judge_evaluation_source(record),
        _comparison_trial_source(record),
        _pairwise_game_source(record),
        _tau2_reward_source(record),
        _evaluator_metadata_source(record),
        *_metric_sources(record),
        _generation_status_source(record),
        _pairwise_game_rubric_source(record),
        _rubric_source(record),
        *_reference_sources(record),
        _candidate_patch_source(record),
    )
    sources = [source for source in candidates if source is not None]

    present = {source.semantics for source in sources}
    if _METRIC_SUMMARY_FIELDS & observed.keys():
        present.add("metric")
    semantics = next((name for name in _SEMANTICS_PRIORITY if name in present), "unknown")

    items, omitted, truncated = _bound_items([item for source in sources for item in source.items])
    evidence = {"semantics": semantics, "observed_outcome": observed, "items": items}
    receipt = {
        "status": "included",
        "source_ref": source_ref,
        "semantics": semantics,
        "fields_found": sorted({*observed, *(source.field for source in sources)}),
        "item_ids_included": [item["id"] for item in items],
        "items_omitted": omitted,
        "truncated": truncated,
    }
    return evidence, receipt


def _unavailable_receipt(
    status: str,
    *,
    source_ref: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "source_ref": source_ref,
        "semantics": "unknown",
        "fields_found": [],
        "item_ids_included": [],
        "items_omitted": 0,
        "truncated": False,
        "error": error,
    }


def _artifact_inventory(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "artifact-inventory",
        "kind": "artifact_inventory",
        "subject": "candidate",
        "source_path": "rollout.artifacts",
        "value": [
            {key: artifact.get(key) for key in ("name", "kind", "role")}
            for artifact in artifacts[:_MAX_INVENTORY_ARTIFACTS]
        ],
        "truncated": len(artifacts) > _MAX_INVENTORY_ARTIFACTS,
    }


def _reference_sources(record: dict[str, Any]) -> list[_Source]:
    """Return explicit references and context-qualified bare answers."""
    answer_fields = _REFERENCE_ANSWER_FIELDS
    if any(record.get(key) is not None for key in _ANSWER_CONTEXT_FIELDS):
        answer_fields = (*answer_fields, "answer")
    return [
        _field_source(
            key,
            item_id=f"reference-{key}",
            kind="reference",
            subject="reference",
            value=record[key],
            semantics="reference" if key in answer_fields else None,
        )
        for key in dict.fromkeys((*answer_fields, *_REFERENCE_CONTEXT_FIELDS))
        if record.get(key) is not None
    ]


def _rubric_source(record: dict[str, Any]) -> _Source | None:
    for key in _RUBRIC_FIELDS:
        rubric = record.get(key)
        if isinstance(rubric, str) and rubric.strip():
            return _field_source(key, item_id="rubric", kind="rubric", subject="task", text=rubric)
    return None


def _metric_sources(record: dict[str, Any]) -> list[_Source]:
    return [
        _field_source(
            key,
            item_id=f"metric-{key}",
            kind="metric",
            subject="candidate",
            value={key: value},
            semantics="metric",
        )
        for key, value in record.items()
        if "/" in key and isinstance(value, (int, float))
    ]


def _generation_status_source(record: dict[str, Any]) -> _Source | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    status = {
        key: response[key] for key in ("status", "error", "incomplete_details") if response.get(key)
    }
    if not status:
        return None
    return _field_source(
        "response",
        item_id="generation-status",
        kind="generation_status",
        subject="candidate",
        value=status,
    )


def _execution_outcome_sources(record: dict[str, Any]) -> list[_Source]:
    """Project bounded post-execution evidence without exposing reference solutions."""
    summary = {
        key: record[key]
        for key in (
            "resolved",
            "evaluation_completed",
            "patch_applied",
            "terminus2_completed",
            "failure_kind",
            "failure_reason",
            "error",
        )
        if record.get(key) is not None
    }
    sources = []
    if summary:
        sources.append(
            _Source(
                "execution",
                [
                    _item(
                        item_id="execution-summary",
                        kind="evaluator_result",
                        subject="candidate",
                        source_path="execution",
                        value=summary,
                    )
                ],
                semantics="judge",
            )
        )
    test_output = record.get("test_output")
    if isinstance(test_output, str) and test_output.strip():
        excerpt = _failure_excerpt(test_output)
        sources.append(
            _Source(
                "test_output",
                [
                    _item(
                        item_id="test-output",
                        kind="evaluator_result",
                        subject="candidate",
                        source_path="test_output",
                        text=excerpt or test_output,
                        value={
                            "full_characters": len(test_output),
                            "failure_excerpt": bool(excerpt),
                        },
                    )
                ],
                semantics="judge",
            )
        )
    test_results = record.get("test_results")
    if test_results is not None:
        sources.append(
            _Source(
                "test_results",
                [
                    _item(
                        item_id="test-results",
                        kind="evaluator_result",
                        subject="candidate",
                        source_path="test_results",
                        value=test_results,
                    )
                ],
                semantics="judge",
            )
        )
    return sources


def _evaluator_metadata_source(record: dict[str, Any]) -> _Source | None:
    """Keep generic evaluator verdict metadata."""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    summary = {
        key: metadata[key]
        for key in _EVALUATOR_METADATA_KEYS
        if metadata.get(key) is not None
    }
    result = record.get("result")
    if result is not None and not isinstance(result, dict):
        summary["result"] = result
    if not summary:
        return None
    return _Source(
        "metadata",
        [
            _item(
                item_id="evaluator-metadata",
                kind="evaluator_result",
                subject="candidate",
                source_path="metadata",
                value=summary,
            )
        ],
        semantics="judge",
    )


def _candidate_patch_source(record: dict[str, Any]) -> _Source | None:
    """Keep a bounded candidate patch while excluding reference/golden patches."""
    patch = record.get("model_patch")
    if not isinstance(patch, str) or not patch.strip():
        return None
    return _Source(
        "model_patch",
        [
            _item(
                item_id="candidate-patch",
                kind="candidate_patch",
                subject="candidate",
                source_path="model_patch",
                text=patch,
                value={"full_characters": len(patch)},
            )
        ],
    )


def _failure_excerpt(text: str, *, limit: int = MAX_ITEM_TEXT_CHARACTERS) -> str:
    """Keep failure-focused lines from a long evaluator log."""
    lines = text.splitlines()
    matches = [index for index, line in enumerate(lines) if _FAILURE_LINE_RE.search(line)]
    if not matches:
        return ""
    selected: set[int] = set()
    characters = 0
    for index in reversed(matches):
        for position in (index, index - 1, index + 1):
            if position < 0 or position >= len(lines) or position in selected:
                continue
            size = len(lines[position]) + 1
            if characters + size > limit:
                if position == index and not selected:
                    selected.add(position)
                    characters = limit
                continue
            selected.add(position)
            characters += size
        if characters >= limit:
            break
    return "\n".join(lines[index] for index in sorted(selected))[:limit]


def _tau2_reward_source(record: dict[str, Any]) -> _Source | None:
    """Project Tau2's concrete DB/action scoring evidence from its result envelope."""
    result = record.get("result")
    reward_info = result.get("reward_info") if isinstance(result, dict) else None
    if not isinstance(reward_info, dict):
        return None
    summary = {
        key: reward_info[key]
        for key in (
            "reward",
            "reward_basis",
            "reward_breakdown",
            "db_check",
            "env_assertions",
            "nl_assertions",
            "communicate_checks",
            "info",
        )
        if reward_info.get(key) is not None
    }
    items = [
        _item(
            item_id="tau2-reward-summary",
            kind="evaluator_result",
            subject="candidate",
            source_path="result.reward_info",
            value=summary,
        )
    ]
    checks = reward_info.get("action_checks")
    if isinstance(checks, list):
        for index, check in enumerate(checks[:_MAX_ENTRIES]):
            if not isinstance(check, dict):
                continue
            action = check.get("action")
            action_summary = (
                {
                    key: action[key]
                    for key in ("action_id", "requestor", "name", "arguments")
                    if action.get(key) is not None
                }
                if isinstance(action, dict)
                else None
            )
            items.append(
                _item(
                    item_id=f"tau2-action-check-{index}",
                    kind="evaluator_action_check",
                    subject="candidate",
                    source_path=f"result.reward_info.action_checks[{index}]",
                    value={
                        "action": action_summary,
                        **{
                            key: check[key]
                            for key in ("action_match", "action_reward", "tool_type", "info")
                            if check.get(key) is not None
                        },
                    },
                )
            )
    return _Source("result.reward_info", items, semantics="judge")


def _field_source(
    key: str,
    *,
    item_id: str,
    kind: str,
    subject: str,
    value: Any = None,
    text: str | None = None,
    semantics: str | None = None,
) -> _Source:
    item = _item(
        item_id=item_id, kind=kind, subject=subject, source_path=key, text=text, value=value
    )
    return _Source(key, [item], semantics)


def _judge_evaluation_source(record: dict[str, Any]) -> _Source | None:
    evaluations = record.get("judge_evaluations")
    if not isinstance(evaluations, list):
        return None
    items = []
    for index, evaluation in enumerate(evaluations[:_MAX_ENTRIES]):
        if not isinstance(evaluation, dict):
            continue
        summary = {
            key: evaluation[key]
            for key in ("verdict_label", "verdict", "score", "reward")
            if evaluation.get(key) is not None
        }
        text = _find_text(evaluation)
        if summary or text:
            items.append(
                _item(
                    item_id=f"judge-evaluation-{index}",
                    kind="judge_rationale",
                    subject="candidate",
                    source_path=f"judge_evaluations[{index}]",
                    text=text,
                    value=summary or None,
                )
            )
    return _Source("judge_evaluations", items, semantics="judge") if items else None


def _comparison_trial_source(record: dict[str, Any]) -> _Source | None:
    """Project alternating-side reference comparisons."""
    response = record.get("judge_response")
    comparisons = response.get("per_ref_repeat") if isinstance(response, dict) else None
    if not isinstance(comparisons, list) or not comparisons:
        return None
    if not all(_is_oriented_comparison(comparison) for comparison in comparisons):
        return None
    items = []
    for ref_index, comparison in enumerate(comparisons):
        judges = comparison.get("trial_judges")
        reference_id = comparison.get("ref_id", comparison.get("ref_repeat"))
        for trial, text in enumerate(comparison["raw_responses"][:_MAX_ENTRIES]):
            if not isinstance(text, str) or not text.strip():
                continue
            candidate = "A" if trial % 2 else "B"
            verdicts = _BOXED_VERDICT_RE.findall(text)
            winner = _BOXED_WINNER[verdicts[-1].upper()] if verdicts else None
            items.append(
                _item(
                    item_id=f"comparison-{ref_index}-trial-{trial}",
                    kind="judge_rationale",
                    subject="comparison",
                    source_path=f"judge_response.per_ref_repeat[{ref_index}].raw_responses[{trial}]",
                    text=text,
                    value={
                        "candidate_label": candidate,
                        "reference_label": _other_side(candidate),
                        "candidate_outcome": _candidate_outcome(winner, candidate),
                        "judge": (
                            judges[trial]
                            if isinstance(judges, list) and trial < len(judges)
                            else None
                        ),
                        "reference_id": reference_id,
                    },
                )
            )
    return _Source("judge_response", items, semantics="comparative") if items else None


def _is_oriented_comparison(comparison: Any) -> bool:
    if not isinstance(comparison, dict) or not (
        comparison.get("ref_id") is not None or comparison.get("ref_repeat") is not None
    ):
        return False
    responses = comparison.get("raw_responses")
    judges = comparison.get("trial_judges")
    return (
        isinstance(responses, list)
        and bool(responses)
        and (judges is None or (isinstance(judges, list) and len(judges) >= len(responses)))
    )


def _pairwise_game_source(record: dict[str, Any]) -> _Source | None:
    """Project paired candidate-versus-baseline games."""
    games = record.get("games")
    if record.get("policy_answer") is None or record.get("baseline_answer") is None:
        return None
    if not isinstance(games, list) or not games or len(games) % 2:
        return None
    verdicts = [_pairwise_verdict(game) for game in games]
    if None in verdicts:
        return None
    items = []
    for index, verdict in enumerate(verdicts[:_MAX_ENTRIES]):
        game = games[index]
        candidate = "B" if index % 2 else "A"
        items.append(
            _item(
                item_id=f"pairwise-game-{index}",
                kind="judge_rationale",
                subject="comparison",
                source_path=f"games[{index}]",
                text=_output_text(game.get("response")),
                value={
                    "candidate_label": candidate,
                    "reference_label": _other_side(candidate),
                    "verdict": verdict,
                    "candidate_outcome": _candidate_outcome(_PAIRWISE_WINNER[verdict], candidate),
                },
            )
        )
    return _Source("games", items, semantics="comparative")


def _pairwise_game_rubric_source(record: dict[str, Any]) -> _Source | None:
    """Keep pairwise-judge instructions as evaluator context."""
    games = record.get("games")
    if not isinstance(games, list):
        return None
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for game_index, game in enumerate(games):
        if not isinstance(game, dict):
            continue
        request = game.get("responses_create_params")
        if not isinstance(request, dict):
            continue
        messages = request.get("input")
        if not isinstance(messages, list):
            messages = request.get("messages")
        for message_index, message in enumerate(messages if isinstance(messages, list) else []):
            if not isinstance(message, dict) or message.get("role") not in {
                "system",
                "developer",
            }:
                continue
            text = _prompt_content_text(message.get("content"))
            if not text or text in seen:
                continue
            seen.add(text)
            items.append(
                _item(
                    item_id=f"pairwise-evaluator-rubric-{len(items)}",
                    kind="evaluator_rubric",
                    subject="evaluator",
                    source_path=(
                        f"games[{game_index}].responses_create_params."
                        f"{'input' if isinstance(request.get('input'), list) else 'messages'}"
                        f"[{message_index}]"
                    ),
                    text=text,
                )
            )
    return _Source("games", items, semantics="judge") if items else None


def _prompt_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or block.get("input_text") or "")
        for block in content
        if isinstance(block, dict)
    )


def _pairwise_verdict(game: Any) -> str | None:
    if not isinstance(game, dict):
        return None
    match = _PAIRWISE_VERDICT_RE.search(str(game.get("verdict") or ""))
    return match.group(1) if match else None


def _candidate_outcome(winner: str | None, candidate: str) -> str:
    if winner is None:
        return "unknown"
    if winner in {"tie", "both_bad"}:
        return winner
    return "win" if winner == candidate else "loss"


def _other_side(label: str) -> str:
    return "B" if label == "A" else "A"


def _output_text(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    parts = [
        content["text"]
        for output in response.get("output") or []
        if isinstance(output, dict)
        for content in output.get("content") or []
        if isinstance(content, dict) and isinstance(content.get("text"), str)
    ]
    return "\n".join(parts) or None


def _item(
    *,
    item_id: str,
    kind: str,
    subject: str,
    source_path: str,
    text: str | None = None,
    value: Any = None,
) -> dict[str, Any]:
    item = {
        "id": item_id,
        "kind": kind,
        "subject": subject,
        "source_path": source_path,
    }
    if text:
        item["text"] = text
    if value is not None:
        item["value"] = _small_value(value)
    return item


def _bound_items(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, bool]:
    bounded = []
    characters = 0
    truncated = False
    for raw_item in items:
        item = dict(raw_item)
        text = item.get("text")
        if isinstance(text, str) and len(text) > MAX_ITEM_TEXT_CHARACTERS:
            head = int(MAX_ITEM_TEXT_CHARACTERS * 0.75)
            item["text"] = {
                "head": text[:head],
                "tail": text[-(MAX_ITEM_TEXT_CHARACTERS - head) :],
                "characters": len(text),
                "truncated": True,
            }
            truncated = True
        rendered = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if characters + len(rendered) > MAX_OUTCOME_EVIDENCE_CHARACTERS:
            item = _shrink_item(item, MAX_OUTCOME_EVIDENCE_CHARACTERS - characters)
            if item is None:
                continue
            rendered = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            truncated = True
        bounded.append(item)
        characters += len(rendered)
    return bounded, len(items) - len(bounded), truncated or len(bounded) < len(items)


def _shrink_item(item: dict[str, Any], budget: int) -> dict[str, Any] | None:
    """Shorten long strings so a useful part of an item still fits."""
    for limit in (2_000, 1_000, 500, 200, 80, 32):
        candidate = _limit_strings(item, limit)
        candidate["truncated"] = True
        if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) <= budget:
            return candidate
    return None


def _limit_strings(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        head = max(1, int(limit * 0.75))
        tail = max(1, limit - head)
        return value[:head] + "…" + value[-tail:]
    if isinstance(value, list):
        return [_limit_strings(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: _limit_strings(item, limit) for key, item in value.items()}
    return value


def _find_text(value: Any, *, depth: int = 0) -> str | None:
    if depth > 5:
        return None
    if isinstance(value, dict):
        for key in _RATIONALE_KEYS:
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text
        children = value.values()
    elif isinstance(value, list):
        children = value[:_MAX_ENTRIES]
    else:
        return None
    for child in children:
        found = _find_text(child, depth=depth + 1)
        if found:
            return found
    return None


def _small_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:2_000]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_small_value(item) for item in value[:30]]
    if isinstance(value, dict):
        return {str(key): _small_value(item) for key, item in list(value.items())[:30]}
    return str(value)[:500]
