# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-only JSONL classification records."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RECORD_KIND = "taxonomy_classification"
log = logging.getLogger(__name__)
_SCALAR_FIELDS = (
    "incident_id",
    "incident_sha256",
    "taxonomy_id",
    "taxonomy_name",
    "taxonomy_version",
    "taxonomy_sha256",
    "decision",
    "selected_leaf_id",
    "selected_code",
    "selected_label",
    "owner",
    "abstention_reason",
    "provider_calls",
    "aggregation",
    "selected_episode_id",
)
_LIST_FIELDS = (
    "decision_reasons",
    "selected_path",
    "selected_path_ids",
    "selected_path_codes",
    "alternatives",
    "remediation",
    "related",
    "trace",
    "window_sha256s",
    "window_receipts",
    "episodes",
)


def classification_record(
    *,
    status: str,
    judge_provider: str,
    analysis: dict[str, Any] | None = None,
    rollout_id: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    repeat: int | None = None,
    lane: str | None = None,
    benchmark: str | None = None,
    model: str | None = None,
    agent_ref: str | None = None,
    digest_budget: dict[str, Any] | None = None,
    evidence_receipt: dict[str, Any] | None = None,
    classification_latency_ms: int | None = None,
    classification_method: str = "llm_as_a_judge",
    judge_request: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
    parent_classification: dict[str, Any] | None = None,
    error_type: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """One record with the same shape for every judge and status."""
    analysis = analysis or {}
    return {
        "kind": RECORD_KIND,
        "classification_method": classification_method,
        "protocol_version": analysis.get("protocol_version"),
        "status": status,
        "source": {
            "rollout_id": rollout_id,
            "run_id": run_id,
            "task_id": task_id,
            "repeat": repeat,
            "lane": lane,
            "benchmark": benchmark,
            "model": model,
            "agent_ref": agent_ref,
        },
        "judge": {
            "provider": judge_provider,
            "model": analysis.get("model"),
            **(judge_request or {}),
        },
        "selection": selection or {},
        "parent_classification": parent_classification or {},
        **{key: analysis.get(key) for key in _SCALAR_FIELDS},
        **{key: list(analysis.get(key) or []) for key in _LIST_FIELDS},
        "first_pass": analysis.get("first_pass") or {},
        "second_pass": dict(analysis.get("second_pass") or {}),
        "diagnosis": analysis.get("diagnosis") or {},
        "windowed": bool(analysis.get("windowed")),
        "window_count": analysis.get("window_count") or 1,
        "digest_budget": digest_budget,
        "evidence_receipt": evidence_receipt or {},
        "classification_latency_ms": classification_latency_ms,
        "usage": analysis.get("usage") or {},
        "error_type": error_type,
        "error": error,
    }


def classification_sha256(record: Mapping[str, Any]) -> str:
    """Hash a parsed classification record without local loader metadata."""
    public = {key: value for key, value in record.items() if not str(key).startswith("_")}
    encoded = json.dumps(
        public,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def has_usable_first_pass(record: Mapping[str, Any]) -> bool:
    """Return whether a record contains a completed classification proposal."""
    if record.get("status") == "ok":
        return True
    first_pass = record.get("first_pass")
    return (
        record.get("status") == "review_failed"
        and isinstance(first_pass, Mapping)
        and first_pass.get("status") == "completed"
        and bool(record.get("selected_code") or record.get("selected_leaf_id"))
    )


def append_classification(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    needs_delimiter = destination.is_file() and destination.stat().st_size > 0
    if needs_delimiter:
        with destination.open("rb") as handle:
            handle.seek(-1, 2)
            needs_delimiter = handle.read(1) not in (b"\n", b"\r")
    with destination.open("a", encoding="utf-8") as handle:
        if needs_delimiter:
            handle.write("\n")
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_classifications(
    path: str | Path,
    *,
    tolerate_truncated_final: bool = False,
    repair_truncated_final: bool = False,
) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    if repair_truncated_final and not tolerate_truncated_final:
        raise ValueError("repair_truncated_final requires tolerate_truncated_final")
    data = source.read_bytes()
    lines = data.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            offset += len(raw_line)
            continue
        try:
            record = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_truncated_final = (
                tolerate_truncated_final
                and line_number == len(lines)
                and not raw_line.endswith((b"\n", b"\r"))
            )
            if not is_truncated_final:
                detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ValueError(f"{source}:{line_number}: malformed JSON: {detail}") from exc
            log.warning("ignoring truncated final JSONL line %s:%d", source, line_number)
            if repair_truncated_final:
                with source.open("r+b") as handle:
                    handle.truncate(offset)
            break
        if not isinstance(record, dict) or record.get("kind") != RECORD_KIND:
            raise ValueError(f"{source}:{line_number}: not a taxonomy classification record")
        rows.append(record)
        offset += len(raw_line)
    return rows
