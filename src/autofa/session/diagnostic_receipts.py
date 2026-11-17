# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable author-control execution receipts, separate from scientific scores."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from autofa.session.smoke_work import dataset_benchmark_slot, digest
from autofa.util.atomic import atomic_write_json


def _dataset_identity(rt: Any) -> str:
    files = []
    for raw in getattr(rt.dataset, "parquet_paths", []):
        path = Path(raw)
        try:
            stat = path.stat()
            files.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
        except FileNotFoundError:
            files.append([str(path.resolve()), None])
    return digest([files, getattr(rt.dataset, "model_id", None),
                   dataset_benchmark_slot(rt.dataset),
                   getattr(rt.dataset, "benchmarks", None)])


def save_diagnostic_receipt(
    rt: Any, *, source: str, execution_identity: str, items: list[dict],
    expected_outputs: dict, expected_abstentions: list[str], result: dict,
    artifact_reads: dict[str, list[dict]] | None = None,
    binding: dict | None = None,
) -> str | None:
    """Save declared controls and actual outcomes; these never establish a causal grade.

    No receipt is issued for random, unlabelled smoke panels or non-JSON outputs.
    Receipt IDs bind the exact outcomes and expectations, independently of repeat
    call counters. Changing expectations creates a distinct diagnostic record.
    """
    if not expected_outputs and not expected_abstentions:
        return None
    run_id = getattr(rt.run, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        return None
    item_ids = [str(item["_item_id"]) for item in items]
    rows = result.get("results") or []
    complete = (result.get("ok") is True and len(rows) == len(item_ids)
                and {row.get("item_id") for row in rows} == set(item_ids)
                and all(row.get("status") in {"value", "abstained", "error"} for row in rows))
    payload = {
        "schema_version": 1, "run_id": run_id, "kind": "smoke_test_feature",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "execution_identity": execution_identity, "dataset_identity": _dataset_identity(rt),
        "source_item_hashes": {str(item["_item_id"]): digest(item) for item in items},
        "artifact_reads": artifact_reads or {}, "item_ids": item_ids,
        "binding": binding or {},
        "expected_outputs": expected_outputs, "expected_abstentions": expected_abstentions,
        "results": rows, "complete": complete, "checks_passed": result.get("checks_passed"),
        "errors": result.get("errors", 0), "mismatches": result.get("mismatches", 0),
        "scope": "author_controls_not_independent_validation",
    }
    try:
        normalized = json.loads(json.dumps(payload, allow_nan=False))
    except (ValueError, TypeError):
        return None
    if normalized != payload:
        return None
    receipt_id = digest(payload)
    atomic_write_json(rt.run.run_dir() / "diagnostic_receipts" / f"{receipt_id}.json", payload)
    return receipt_id


def read_diagnostic_receipt(rt: Any, receipt_id: str) -> dict:
    """Resolve a receipt against this run and its current original source evidence."""
    if not isinstance(receipt_id, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt_id):
        raise ValueError("Supply a diagnostic_receipt_id returned by smoke_test_feature")
    try:
        receipt = json.loads((rt.run.run_dir() / "diagnostic_receipts"
                              / f"{receipt_id}.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("Diagnostic receipt is unavailable in this run") from exc
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != 1
            or receipt.get("kind") != "smoke_test_feature" or digest(receipt) != receipt_id
            or receipt.get("run_id") != rt.run.run_id):
        raise ValueError("Diagnostic receipt identity or run does not match")
    if receipt.get("dataset_identity") != _dataset_identity(rt):
        raise ValueError("Diagnostic receipt source dataset changed; inspect fresh evidence")
    binding = receipt.get("binding") or {}
    if binding.get("name"):
        from autofa.session.repair_workflow import _version

        fn = rt.library.get(binding["name"])
        if fn is None or _version(fn.to_dict()) != binding["version_hash"]:
            raise ValueError("Diagnosed detector version changed; cite its current execution")
    elif binding.get("depends_on") or binding.get("uses_helpers"):
        from autofa.core.dependencies import build_bundle

        bundle = build_bundle(rt.library, depends_on=binding.get("depends_on"),
                              uses_helpers=binding.get("uses_helpers"))
        if digest(bundle) != binding["bundle_identity"]:
            raise ValueError("Diagnostic dependencies changed; cite their current execution")
    from autofa.data.artifact_access import validate_artifact_reads
    from autofa.data.trace_fields import derive_trace_fields
    from autofa.mcp.tools_eval import _fetch_items_by_id

    ids = receipt.get("item_ids") or []
    rows = _fetch_items_by_id(rt.dataset, set(ids), max_items=None)
    current = {str(item["_item_id"]): item for item in rows}
    if ({item_id: digest(item) for item_id, item in current.items()}
            != receipt.get("source_item_hashes")):
        raise ValueError("Diagnostic control evidence changed or is unavailable")
    for item_id, reads in receipt.get("artifact_reads", {}).items():
        if reads and not validate_artifact_reads(derive_trace_fields(current[item_id]), reads):
            raise ValueError("Diagnostic artifact evidence changed or is unavailable")
    return receipt
