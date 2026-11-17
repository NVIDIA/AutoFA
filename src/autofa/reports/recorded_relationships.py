# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read existing discovery links for reports without changing or running discovery."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from autofa.core.dependencies import review_source
from autofa.core.library import FunctionLibrary
from autofa.metrics.claim_contract import claim_hash

_STATE_BYTES = 32 * 1024 * 1024
_MAX_RECORDS = 4096
_ORIGINS = {"explains": "explains", "synthesis": "assessed_synthesis"}


def read_state(paths: Iterable[Path | None]) -> dict | None:
    """Read an optional sibling state file; malformed/missing records leave a flat report."""
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw)
        directory = path if path.is_dir() else path.parent
        candidate = directory / "family_discovery.json"
        try:
            with candidate.open("rb") as stream:
                content = stream.read(_STATE_BYTES + 1)
            if len(content) > _STATE_BYTES:
                continue
            value = json.loads(content)
        except (OSError, ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _records(state: dict | None) -> Iterator[dict]:
    stack, seen = [state], set()
    while stack and len(seen) < _MAX_RECORDS:
        row = stack.pop()
        if not isinstance(row, dict) or id(row) in seen:
            continue
        seen.add(id(row))
        yield row
        for key in ("comparisons", "history"):
            children = row.get(key)
            if isinstance(children, list):
                stack.extend(reversed(children[:_MAX_RECORDS]))


def recorded_relationships(
    library: FunctionLibrary, state: dict | None = None, previous: list | None = None,
) -> list[dict[str, Any]]:
    """Export only recorded current links, retaining their actual relationship type.

    Synthesis lineage and execution refinements are typed connections. Similar names,
    dependencies and overlapping matches never create an edge. No model, dataset,
    runtime state writer or discovery tool is used.
    """
    functions = {fn.name: fn for fn in library.functions}
    claims = {}
    for name, fn in functions.items():
        try:
            claims[name] = claim_hash(
                source=review_source(fn), description=fn.description or "",
                mechanism=fn.mechanism or fn.description or fn.name,
            )
        except (ValueError, TypeError, KeyError):
            continue
    edges: dict[tuple[str, str, str], dict] = {}

    def add(source, target, relation, rationale, origin):
        if (not isinstance(source, str) or not isinstance(target, str)
                or source == target or source not in claims or target not in claims):
            return
        edges[source, target, relation] = {
            "source": source, "target": target, "relation": relation,
            "rationale": rationale if isinstance(rationale, str) else "",
            "recorded_by": origin, "source_claim_hash": claims[source],
            "target_claim_hash": claims[target],
        }

    # Exported links remain portable when the run-side discovery ledger is absent.
    # Their exact claim identities prevent stale connections after edits/removals.
    for edge in previous if isinstance(previous, list) else []:
        if not isinstance(edge, dict):
            continue
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("relation")
        if (not isinstance(source, str) or not isinstance(target, str)
                or source not in claims or target not in claims
                or edge.get("source_claim_hash") != claims[source]
                or edge.get("target_claim_hash") != claims[target]
                or not isinstance(relation, str) or relation not in _ORIGINS
                or edge.get("recorded_by") != _ORIGINS[relation]):
            continue
        if relation == "explains" and functions[source].explains != target:
            continue
        add(source, target, relation, edge.get("rationale"), edge["recorded_by"])

    for fn in library.functions:
        if fn.explains:
            add(fn.name, fn.explains, "explains",
                "Recorded explanatory refinement evaluated within the other detector's matches.",
                "explains")

    for record in _records(state):
        survey, assessment = record.get("survey"), record.get("assessment")
        if not isinstance(survey, dict) or not isinstance(assessment, dict):
            continue
        name = assessment.get("detector_name") or survey.get("detector_name")
        name = name or survey.get("family_name")
        if not isinstance(name, str) or name not in claims:
            continue
        fn = functions[name]
        if (assessment.get("outcome") != "assessed"
                or assessment.get("claim_hash") != claims[name]
                or any(not isinstance(survey.get(key), str)
                       or survey[key].strip() != (getattr(fn, key) or "").strip()
                       for key in ("description", "mechanism"))):
            continue
        if survey.get("approach") != "bottom_up":
            continue
        seeds = survey.get("seed_functions")
        if not isinstance(seeds, list):
            continue
        for seed in seeds:
            if (isinstance(seed, dict) and isinstance(seed.get("name"), str)
                    and seed["name"] in claims
                    and seed.get("claim_hash") == claims[seed["name"]]):
                add(name, seed["name"], "synthesis",
                    "Recorded as a synthesis of this detector's inspected failure mechanism.",
                    "assessed_synthesis")

    return [edges[key] for key in sorted(edges)]
