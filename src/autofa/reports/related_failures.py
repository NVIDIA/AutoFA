# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project recorded detector connections for display, without discovery side effects.

The caller supplies explicit relationships already found by research. Names,
overlapping matches and execution dependencies alone never establish a connection.
Grades, target counts, coverage and the input detector order remain untouched.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from autofa.reports.ordering import visible_report_functions

MAX_RELATIONSHIPS = 512
_RELATIONS = frozenset({"synthesis", "explains"})


def _edge_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["source"], row["target"], row["relation"], row["rationale"]


def _deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result = {}
    for row in sorted(rows, key=_edge_key):
        result.setdefault((row["source"], row["target"], row["relation"]), row)
    return list(result.values())


def project_related_failures(
    functions: list[dict[str, Any]], relationships: list[dict[str, Any]],
) -> dict[str, Any]:
    """Group visible, explicitly connected detectors; never rescore or infer families.

    Synthesis and explains remain separately typed, directed connections.
    A report group is navigation, not a new failure or an independent-family count.
    """
    eligible = {row.get("name") for row in visible_report_functions(functions)}
    names = list(dict.fromkeys(
        row["name"] for row in functions
        if isinstance(row.get("name"), str) and row["name"] and row["name"] in eligible
    ))
    allowed = set(names)
    edges = []
    for raw in relationships if isinstance(relationships, list) else []:
        if not isinstance(raw, dict):
            continue
        source, target, relation = (raw.get(key) for key in ("source", "target", "relation"))
        if (not all(isinstance(value, str) for value in (source, target, relation))
                or source == target or source not in allowed or target not in allowed
                or relation not in _RELATIONS):
            continue
        rationale = raw.get("rationale")
        edges.append({"source": source, "target": target, "relation": relation,
                      "rationale": rationale if isinstance(rationale, str) else ""})
    edges = _deduplicate(edges)[:MAX_RELATIONSHIPS]

    neighbors: dict[str, set[str]] = {}
    for row in edges:
        neighbors.setdefault(row["source"], set()).add(row["target"])
        neighbors.setdefault(row["target"], set()).add(row["source"])
    groups, visited = [], set()
    for name in names:
        if name in visited or name not in neighbors:
            continue
        pending, members = [name], set()
        while pending:
            current = pending.pop()
            if current not in members:
                members.add(current)
                pending.extend(neighbors.get(current, ()))
        visited.update(members)
        stable = json.dumps(sorted(members), ensure_ascii=False, separators=(",", ":"))
        groups.append({
            "id": "related-" + hashlib.sha256(stable.encode()).hexdigest()[:16],
            "names": [value for value in names if value in members],
            "relationships": [row for row in edges if row["source"] in members],
        })
    return {"version": 1, "relationships": edges, "groups": groups}
