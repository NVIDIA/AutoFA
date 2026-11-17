# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable authorship, inspection and source-audit records, separate from scores."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from autofa.core.dependencies import digest

ACTORS = frozenset({"native", "driver", "unknown"})


def definition_identity(function: Any) -> str:
    from autofa.core.dependencies import review_source

    row = function.to_dict() if hasattr(function, "to_dict") else function
    return digest({"source": review_source(row), "description": row.get("description"),
                   "mechanism": row.get("mechanism"), "kind": row.get("kind"),
                   "llm_prompt": row.get("llm_prompt")})


def record_revision(function: Any, previous: Any = None, *, actor: str = "native",
                    imported_from: dict | None = None) -> None:
    """Keep original authorship when native optimization changes an imported detector."""
    if actor not in ACTORS:
        raise ValueError(f"actor must be one of {sorted(ACTORS)}")
    source = previous if previous is not None else function
    provenance = deepcopy((source.meta or {}).get("contribution_provenance") or {})
    identity = definition_identity(function)
    if not provenance:
        provenance = {"version": 1, "origin": {
            "actor": "unknown" if previous is not None or imported_from else actor,
            "definition_sha256": definition_identity(source),
        }, "revisions": []}
    if (not isinstance(provenance, dict) or provenance.get("version") != 1
            or not isinstance(provenance.get("origin"), dict)
            or not isinstance(provenance.get("revisions"), list)):
        raise ValueError("Unsupported contribution provenance")
    changed = previous is None or definition_identity(previous) != identity
    if changed or imported_from:
        event = {"actor": actor, "action": "import" if imported_from else
                 "create" if previous is None else "edit", "definition_sha256": identity,
                 "at": datetime.now(UTC).isoformat()}
        if imported_from:
            event["source"] = deepcopy(imported_from)
        provenance["revisions"].append(event)
    function.meta["contribution_provenance"] = provenance


def merge_records(*groups: list[dict]) -> list[dict]:
    """Append-only merge; a reused receipt ID must retain its original contents."""
    found = {}
    for records in groups:
        if not isinstance(records, list):
            raise ValueError("Contribution records must be an array")
        for row in records:
            if (not isinstance(row, dict) or row.get("version") != 1
                    or not isinstance(row.get("id"), str) or not row["id"]
                    or row.get("kind") not in {"inspection", "source_audit", "import"}
                    or row.get("actor") not in ACTORS):
                raise ValueError("Invalid contribution record")
            if row["kind"] == "source_audit":
                required = ("name", "item_id", "finding", "artifact_reference", "artifact_sha256",
                            "definition_sha256")
                if (any(not isinstance(row.get(key), str) or not row[key] for key in required)
                        or not isinstance(row.get("review_identity"), dict)
                        or row.get("proposed_verdict") not in {None, "yes", "no"}
                        or (row.get("resolves") is not None
                            and not isinstance(row["resolves"], str))):
                    raise ValueError("Invalid source audit receipt")
            elif row["kind"] == "inspection":
                if (not isinstance(row.get("items"), list)
                        or any(not isinstance(item, dict)
                               or not isinstance(item.get("_item_id"), str)
                               for item in row["items"])):
                    raise ValueError("Invalid inspection receipt")
            elif not isinstance(row.get("name"), str):
                raise ValueError("Invalid import receipt")
            if row["id"] in found and found[row["id"]] != row:
                raise ValueError("Conflicting contribution receipt")
            found[row["id"]] = deepcopy(row)
    return list(found.values())


def new_record(kind: str, actor: str, **details) -> dict:
    if actor not in ACTORS:
        raise ValueError(f"actor must be one of {sorted(ACTORS)}")
    row = {"version": 1, "kind": kind, "actor": actor, **details}
    # Identical requests are idempotent; the first receipt's timestamp is retained.
    row["id"] = digest(row)
    row["at"] = datetime.now(UTC).isoformat()
    return row


def append_record(library: Any, row: dict) -> dict:
    existing = next((value for value in library.contributions if value["id"] == row["id"]), None)
    if existing is not None:
        return existing
    library.contributions = merge_records(library.contributions, [row])
    return row


def unresolved_audits(library: Any) -> list[dict]:
    audits = [row for row in library.contributions if row["kind"] == "source_audit"]
    resolved = {row.get("resolves") for row in audits if row.get("resolves")}
    return [row for row in audits if not row.get("resolves") and row["id"] not in resolved]


def review_identity(function: Any, item_id: str) -> dict:
    scores = function.scores or {}
    rows = [row for row in scores.get("behavior_verdicts", []) if row.get("item_id") == item_id]
    return {"claim_hash": scores.get("claim_hash"),
            "judge_policy_hash": scores.get("judge_policy_hash"),
            "item_verdicts_sha256": digest(rows),
            "cohort_sha256": digest(scores.get("behavior_verdicts", []))}


def audit_matches(library: Any, row: dict, *, require_review: bool = False) -> bool:
    """Match a dispute to its claim, and optionally the exact reviewed cohort."""
    fn = library.get(row["name"])
    if fn is None or definition_identity(fn) != row["definition_sha256"]:
        return False
    if not require_review:
        return True
    try:
        return review_identity(fn, row["item_id"]) == row["review_identity"]
    except (ValueError, TypeError):
        # Legacy verdict JSON can contain NaN or other non-canonical values. Such
        # a cohort cannot establish an exact sensitivity claim, but must not make
        # the unresolved audit (or the rest of the report) disappear.
        return False


def current_disputed_names(library: Any, audits: list[dict] | None = None) -> list[str]:
    rows = unresolved_audits(library) if audits is None else audits
    return sorted({row["name"] for row in rows if audit_matches(library, row)})


def qualification_sensitivity(library: Any) -> list[dict]:
    """Hypothetical adherence under explicit proposed verdicts and the exact saved cohort."""
    from autofa.metrics.verdicts import QUALITY_THRESHOLDS

    audits = unresolved_audits(library)
    output = []
    for name in sorted({row["name"] for row in audits}):
        fn = library.get(name)
        selected = [row for row in audits if row["name"] == name]
        result = {"name": name, "available": False, "hypothetical": True,
                  "native_verdicts_changed": False, "audit_ids": [row["id"] for row in selected]}
        output.append(result)
        if any(not audit_matches(library, row, require_review=True) for row in selected):
            result["reason"] = "The claim or review cohort changed; the dispute remains unresolved."
            continue
        scores = fn.scores or {}
        n, yes = scores.get("behavior_n_completed"), scores.get("behavior_n_yes")
        purity = scores.get("behavior_purity")
        if (type(n) is not int or type(yes) is not int or not 0 <= yes <= n or n == 0
                or not isinstance(purity, (float, int)) or abs(purity - yes / n) > 1e-9):
            result["reason"] = "Saved review counts do not establish an exact adherence fraction."
            continue
        proposals = {}
        invalid = False
        for audit in selected:
            proposed = audit.get("proposed_verdict")
            rows = [row for row in scores.get("behavior_verdicts", [])
                    if row.get("item_id") == audit["item_id"]]
            if (proposed not in {"yes", "no"} or len(rows) != 1
                    or rows[0].get("verdict") not in {"yes", "no"}
                    or (audit["item_id"] in proposals
                        and proposals[audit["item_id"]][1] != proposed)):
                invalid = True
                break
            proposals[audit["item_id"]] = (rows[0]["verdict"], proposed)
        if invalid:
            result["reason"] = (
                "Explicit, consistent yes/no proposals for exact reviewed items are required."
            )
            continue
        delta = sum(int(new == "yes") - int(old == "yes") for old, new in proposals.values())
        revised = yes + delta
        if not 0 <= revised <= n:
            result["reason"] = "Proposed changes conflict with the saved review counts."
            continue
        floor = QUALITY_THRESHOLDS["behavior_purity"]["good"]
        result.update(available=True, native_yes=yes, hypothetical_yes=revised, reviewed=n,
                      native_adherence=purity, hypothetical_adherence=revised / n,
                      good_adherence_floor=floor, meets_good_adherence_floor=revised / n >= floor,
                      assumptions="Apply only these proposed changes to this frozen cohort; "
                      "all other evidence is unchanged. This is not a new native grade.")
    return output


def coverage_sensitivity(library: Any, inventory: dict | None, *, exclude_names: list[str],
                         minimum_grade: str = "GOOD", **kwargs) -> dict:
    """Calculate native union twice on the same inventory; never edit saved grades."""
    from autofa.metrics.library_quality_coverage import library_quality_coverage

    names = set(exclude_names)
    if names - {fn.name for fn in library.functions}:
        raise ValueError("Sensitivity names must refer to current library detectors")
    filtered = deepcopy(library)
    filtered.functions = [fn for fn in filtered.functions if fn.name not in names]
    return {"excluded_names": sorted(names), "hypothetical": True,
            "native": library_quality_coverage(library, inventory,
                                                minimum_grade=minimum_grade, **kwargs),
            "excluding_disputed": library_quality_coverage(filtered, inventory,
                                                            minimum_grade=minimum_grade, **kwargs)}
