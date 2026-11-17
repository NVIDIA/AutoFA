# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded external contributions using native admission and evidence machinery."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from autofa.core.library import FunctionLibrary
from autofa.mcp.runtime import err, get_runtime, ok
from autofa.session.contributions import (
    ACTORS,
    append_record,
    coverage_sensitivity,
    current_disputed_names,
    definition_identity,
    merge_records,
    new_record,
    qualification_sensitivity,
    record_revision,
    review_identity,
    unresolved_audits,
)
from autofa.session.operation_admission import admit_operation
from autofa.util.cancellation import cancellation_guard


def _verified_items(rt: Any, item_ids: list[str]) -> list[dict]:
    from autofa.session.family_coverage import cached_task_inventory, verified_identity_projection

    if (not isinstance(item_ids, list) or not item_ids
            or any(not isinstance(value, str) or not value for value in item_ids)):
        raise ValueError("item_ids must contain exact nonempty destination item IDs")
    inventory = cached_task_inventory(rt)
    identities = inventory["identity_by_item_id"]
    if any(item_id not in identities for item_id in item_ids):
        raise ValueError("Unknown or ambiguous source identity; map exact destination IDs first")
    return [verified_identity_projection(rt, item_id) for item_id in dict.fromkeys(item_ids)]


def _save_exposure(rt: Any, items: list[dict], receipt: dict) -> dict:
    from autofa.session.review_evidence import record_discovery_exposure

    with cancellation_guard(), rt._library_lock:
        # Reserve exposure before saving a receipt. A failed save may conservatively
        # retain exposure, but must never allow these items as fresh confirmation.
        record_discovery_exposure(rt.run.run_dir(), items)
        staged = deepcopy(rt.library)
        saved = append_record(staged, receipt)
        rt.persist_library(library=staged)
    return saved


def record_external_inspection(item_ids: list[str], reason: str,
                               references: list[dict] | None = None) -> dict[str, Any]:
    """Register externally viewed examples as exposed, with portable actor/source receipts."""
    from autofa.mcp.server import invocation_actor

    actor = invocation_actor()
    with admit_operation("record_external_inspection", {}) as busy:
        if busy:
            return busy
        rt = get_runtime()
        try:
            if not reason.strip() or actor not in ACTORS:
                raise ValueError("A reason and supported actor are required")
            if references is not None and (not isinstance(references, list)
                                           or any(not isinstance(ref, dict) for ref in references)):
                raise ValueError("references must be source/artifact reference objects")
            items = _verified_items(rt, item_ids)
            receipt = new_record("inspection", actor, reason=reason, items=items,
                                 references=references or [], source_run_id=rt.run.run_id)
            return ok(receipt=_save_exposure(rt, items, receipt),
                      confirmation_scope="These examples are development-exposed")
        except (ValueError, KeyError, TypeError) as exc:
            return err(str(exc))


def record_source_audit(name: str, item_id: str, finding: str, artifact_reference: str,
                        artifact_sha256: str,
                        resolves: str | None = None,
                        proposed_verdict: str | None = None) -> dict[str, Any]:
    """Record a source-grounded discrepancy or reconciliation without editing verdicts.

    Benchmark PASS/FAIL labels alone do not establish a review error. Identify the
    original task requirement and source evidence that supports or contradicts the
    reviewed claim; preserve outcome labels and distinguish benchmark/source
    disagreements from judging errors. Proposed verdicts need that source evidence.
    """
    from autofa.mcp.server import invocation_actor

    actor = invocation_actor()
    with admit_operation("record_source_audit", {}) as busy:
        if busy:
            return busy
        rt = get_runtime()
        try:
            fn = rt.library.get(name)
            if fn is None:
                raise ValueError("Unknown detector")
            if not finding.strip() or not artifact_reference.strip():
                raise ValueError("A finding and original artifact reference are required")
            if not re.fullmatch(r"[a-fA-F0-9]{64}", artifact_sha256):
                raise ValueError("artifact_sha256 must identify the inspected original artifact")
            if proposed_verdict not in {None, "yes", "no"}:
                raise ValueError("proposed_verdict must be yes, no, or omitted")
            items = _verified_items(rt, [item_id])
            if resolves:
                prior = next((row for row in unresolved_audits(rt.library)
                              if row["id"] == resolves), None)
                if prior is None or prior.get("name") != name or prior.get("item_id") != item_id:
                    raise ValueError("resolves must identify this detector/item's open audit")
            receipt = new_record(
                "source_audit", actor, name=name, item_id=item_id, finding=finding,
                artifact_reference=artifact_reference, artifact_sha256=artifact_sha256.lower(),
                source_run_id=rt.run.run_id, definition_sha256=definition_identity(fn),
                review_identity=review_identity(fn, item_id), resolves=resolves,
                proposed_verdict=proposed_verdict,
            )
            return ok(receipt=_save_exposure(rt, items, receipt), native_verdicts_changed=False,
                      note="Audit/reconciliation records are attributed findings, "
                      "not new judgments. Benchmark labels alone neither confirm nor refute "
                      "a source-grounded review.")
        except (ValueError, KeyError, TypeError) as exc:
            return err(str(exc))


def audit_sensitivity(exclude_names: list[str] | None = None,
                      minimum_grade: str = "GOOD") -> dict[str, Any]:
    """Read native coverage with/without current disputed detectors on the same denominator."""
    from autofa.session.family_coverage import failed_task_inventory

    rt = get_runtime()
    try:
        audits = unresolved_audits(rt.library)
        if exclude_names is None:
            exclude_names = current_disputed_names(rt.library, audits)
        return ok(audits=audits, qualification_sensitivity=qualification_sensitivity(rt.library),
                  sensitivity=coverage_sensitivity(
            rt.library, failed_task_inventory(rt.dataset), exclude_names=exclude_names,
            minimum_grade=minimum_grade, run_id=rt.run.run_id,
        ))
    except (ValueError, KeyError, TypeError) as exc:
        return err(str(exc))


def library_import_candidates(path: str, names: list[str] | None = None,  # noqa: C901
                               apply: bool = False) -> dict[str, Any]:
    """Preview/import unscored portable candidates; conflicts require ordinary native repair.

    Dependencies/helpers and prior exposure are retained. Applying never scores/judges;
    native admission limits still apply. A rejected batch reports any candidates already
    committed, and retry recognizes them rather than replacing later optimization.
    """
    from autofa.config import runs_root
    from autofa.core.dependencies import execution_bundle, validate_edges
    from autofa.core.detector_lint import lint_detector_source
    from autofa.core.fields import get_field_policy
    from autofa.core.sandbox import compile_detector
    from autofa.mcp.implementation_recovery import check_implementation_submission
    from autofa.mcp.server import invocation_actor
    from autofa.session.family_discovery import family_submission_guard
    from autofa.session.modes_ledger import ledger_peers_for_novelty
    from autofa.session.polishing_guard import polishing_tool_guard
    from autofa.session.repair_workflow import record_submission
    from autofa.session.review_evidence import import_seed_review_history, merge_histories

    actor = invocation_actor()
    with admit_operation("library_import_candidates", {}) as busy:
        if busy:
            return busy
        rt = get_runtime()
        committed = []
        try:
            if actor not in ACTORS:
                raise ValueError(f"actor must be one of {sorted(ACTORS)}")
            source_path = Path(path).expanduser()
            source_bytes = source_path.read_bytes()
            seed = FunctionLibrary.from_dict(json.loads(source_bytes))
            if len({fn.name for fn in seed.functions}) != len(seed.functions):
                raise ValueError("Source candidate names are ambiguous")
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            origin = {"library_sha256": source_hash, "model_id": seed.model_id,
                      "benchmark": seed.benchmark}
            # Remote cache sidecars bind immutable version identity to exact bytes.
            sidecar = source_path.with_name("metadata.json")
            if sidecar.is_file():
                metadata = json.loads(sidecar.read_text())
                version = metadata.get("version", {})
                if isinstance(version, dict) and version.get("library_sha256") == source_hash:
                    origin.update({key: version[key] for key in
                                   ("id", "library_id", "branch_id", "source_run_id")
                                   if version.get(key)})
            from autofa.data.library_provenance import read_catalog_provenance

            provenance = read_catalog_provenance(source_path)
            if provenance is not None:
                origin["catalog_lineage"] = {key: value for key, value in
                                              provenance.to_dict().items()
                                              if key in {"source_run_id", "remote_library_id",
                                                         "remote_branch_id", "remote_version_id"}}
            if names is not None and (not isinstance(names, list)
                                      or any(not isinstance(name, str) for name in names)):
                raise ValueError("names must be an array of detector names")
            selected = list(dict.fromkeys(names if names is not None else
                                          (fn.name for fn in seed.functions)))
            if not selected:
                raise ValueError("Select at least one candidate")
            ordered, visiting = [], set()

            def visit(name):
                if name in visiting:
                    raise ValueError("Candidate dependency cycle")
                if name in ordered:
                    return
                fn = seed.get(name)
                if fn is None:
                    raise ValueError(f"Missing source dependency: {name}")
                visiting.add(name)
                for dep in list(fn.depends_on) + ([fn.explains] if fn.explains else []):
                    visit(dep)
                visiting.remove(name)
                ordered.append(name)

            for name in selected:
                visit(name)
            conflicts, skipped = [], []
            peers = [fn.to_dict() for fn in rt.library.functions]
            peers += ledger_peers_for_novelty(rt.run.run_dir())
            for name in ordered:
                fn, existing = seed.get(name), rt.library.get(name)
                if existing is not None:
                    prior_imports = (existing.meta.get("contribution_provenance") or {}).get(
                        "revisions", [])
                    already = any(row.get("action") == "import"
                                  and (row.get("source") or {}).get("library_sha256") == source_hash
                                  for row in prior_imports)
                    if already or definition_identity(existing) == definition_identity(fn):
                        skipped.append(name)
                    else:
                        conflicts.append({"name": name, "reason": "existing detector differs"})
                for peer in peers:
                    if (peer["name"] != name and fn.description == peer.get("description")
                            and fn.mechanism == peer.get("mechanism")):
                        conflicts.append({"name": name, "reason": "claim already has an owner",
                                          "owner": peer["name"]})
                bundle = execution_bundle(fn)
                validate_edges(fn.source, (bundle or {}).get("helpers", {}),
                               (bundle or {}).get("detectors", {}))
                compile_detector(fn.source, bundle=bundle, policy=get_field_policy(rt.config))
                valid, reason = lint_detector_source(fn.source)
                if not valid:
                    raise ValueError(reason or "Candidate failed native source lint")
            if conflicts or not apply:
                return ok(applied=False, candidates=ordered, skipped=skipped, conflicts=conflicts,
                          source_sha256=source_hash,
                          next_action="Resolve collisions through native repair; "
                          "then apply explicitly")
            if blocked := polishing_tool_guard(rt, "library_load", {}):
                return blocked
            import_seed_review_history(seed, source_path, runs_root=runs_root())
            with cancellation_guard(), rt._library_lock:
                staged = deepcopy(rt.library)
                staged.helpers.merge(seed.helpers)
                staged.review_history = merge_histories(staged.review_history, seed.review_history)
                staged.contributions = merge_records(staged.contributions, seed.contributions)
                # Preserve exposure first, even if an individual candidate is rejected.
                rt.persist_library(library=staged)
                for name in ordered:
                    if name in skipped:
                        continue
                    original = seed.get(name)
                    candidate = deepcopy(original)
                    candidate.scores = {}
                    candidate.example_fail_ids = []
                    candidate.failure_localisations = {}
                    record_revision(candidate, original, actor=actor, imported_from={
                        **origin,
                        "definition_sha256": definition_identity(original),
                    })
                    if blocked := check_implementation_submission(rt, candidate.to_dict(), None):
                        return {**blocked, "committed": committed}

                    def commit(repairs, candidate=candidate):
                        family_block = family_submission_guard(
                            rt, candidate.name, explains=candidate.explains,
                            description=candidate.description, mechanism=candidate.mechanism,
                            consume_attempt=True,
                        )
                        if family_block:
                            raise ValueError(family_block)
                        staged = deepcopy(rt.library)
                        staged.upsert(candidate)
                        append_record(staged, new_record(
                            "import", actor, name=candidate.name, library_sha256=source_hash,
                            source=origin,
                            definition_sha256=definition_identity(candidate),
                            source_run_id=rt.run.run_id,
                        ))
                        rt.persist_library(library=staged, repairs=repairs)

                    before = rt.library
                    try:
                        record_submission(rt.run.run_dir(), candidate.to_dict(),
                                          current_function=None, apply=commit)
                    except BaseException:
                        if rt.library is not before:
                            rt.persist_library(library=before)
                        raise
                    committed.append(name)
            return ok(applied=True, committed=committed, skipped=skipped,
                      evaluation_required=True, scores_transferred=False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return err(str(exc), committed=committed)
