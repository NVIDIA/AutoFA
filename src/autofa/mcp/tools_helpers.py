# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create, find, and exercise unscored reusable functions. No detector evaluation."""

from __future__ import annotations

import ast
import inspect
from copy import deepcopy
from typing import Any

from autofa.core.dependencies import build_bundle, digest, execution_bundle, validate_edges
from autofa.core.detector_lint import evidence_budget_warnings
from autofa.core.fields import get_field_policy, wrap_item_ban_sentinels
from autofa.core.helpers import HelperFunction, bounded_result, check_schema
from autofa.core.sandbox import compile_detector, validate_source
from autofa.mcp.runtime import err, get_runtime, ok


def submit_helper(
    name: str,
    source: str,
    description: str = "",
    entrypoint: str = "",
    purpose: str = "transform",
    uses_helpers: dict[str, str] | None = None,
    input_schema: dict | None = None,
    output_schema: dict | None = None,
    required_surfaces: list[str] | None = None,
) -> dict[str, Any]:
    """Store an unscored function. Supply name, source, description, optional entrypoint,
    input_schema/output_schema and uses_helpers={alias: name_or_revision}.
    Helpers can accept arbitrary declared arguments and return bounded JSON values.
    No detector claim, dataset, score, judge, or promotion is needed.
    """
    rt = get_runtime()
    if not name.strip() or len(name) > 160:
        return err("helper name must contain 1–160 characters")
    if purpose not in {"transform", "extractor", "predicate"}:
        return err("purpose must be transform, extractor, or predicate")
    try:
        tree = ast.parse(source)
        functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        entrypoint = entrypoint or (functions[0] if functions else "")
        if entrypoint not in functions:
            raise ValueError("helper entrypoint must name a top-level function")
        bundle = build_bundle(rt.library, uses_helpers=uses_helpers)
        validate_edges(source, (bundle or {}).get("helpers", {}), {})
        policy = get_field_policy(rt.config)
        validate_source(source, policy, helper=True)
        compile_detector(source, entrypoint=entrypoint, bundle=bundle, policy=policy)
        helper = HelperFunction(
            name=name.strip(),
            source=source,
            entrypoint=entrypoint,
            description=description,
            purpose=purpose,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            required_surfaces=required_surfaces or [],
            origin={"run_id": rt.run.run_id},
            bundle=bundle or {},
        )
        # Round-trip executable definitions before persistence; no sample labels needed.
        helper = HelperFunction.from_dict(helper.to_dict())
    except (ValueError, TypeError, SyntaxError) as exc:
        return err(f"helper rejected: {exc}")
    rt.helpers.upsert(helper)
    rt.persist_library()
    return ok(
        name=helper.name,
        revision=helper.revision,
        entrypoint=entrypoint,
        n_helpers=len(rt.helpers.aliases),
        message="Helper saved. Use uses_helpers={alias: revision} in a detector or helper.",
        evidence_warnings=evidence_budget_warnings(source),
    )


def list_helpers(
    query: str = "",
    purpose: str | None = None,
    surfaces: list[str] | None = None,
    offset: int = 0,
    limit: int = 20,
    include_deprecated: bool = False,
) -> dict[str, Any]:
    """Find unscored reusable functions by description, purpose, or input surfaces."""
    rt = get_runtime()
    words = query.casefold().split()
    rows = []
    for alias, ref in sorted(rt.helpers.aliases.items()):
        helper = rt.helpers.revisions[ref]
        if not include_deprecated and helper.status == "deprecated":
            continue
        if purpose and helper.purpose != purpose:
            continue
        if surfaces and not set(surfaces) & set(helper.required_surfaces):
            continue
        text = f"{alias} {helper.description} {helper.purpose}".casefold()
        if not all(word in text for word in words):
            continue
        consumers = [
            fn.name
            for fn in rt.library.functions
            if ref in (execution_bundle(fn) or {}).get("nodes", {})
        ]
        rows.append(
            {
                "name": alias,
                "revision": ref,
                "description": helper.description,
                "purpose": helper.purpose,
                "entrypoint": helper.entrypoint,
                "input_schema": helper.input_schema,
                "output_schema": helper.output_schema,
                "required_surfaces": helper.required_surfaces,
                "consumers": consumers,
                "status": helper.status,
            }
        )
    offset, limit = max(0, offset), max(1, min(50, limit))
    return ok(
        helpers=rows[offset : offset + limit],
        total=len(rows),
        next_offset=offset + limit if offset + limit < len(rows) else None,
    )


def get_helper(name: str, revision: str | None = None) -> dict[str, Any]:
    """Read exact helper code, contracts and pinned dependencies, without scoring it."""
    helper = get_runtime().helpers.get(revision or name)
    return (ok(helper=helper.to_dict(), evidence_warnings=evidence_budget_warnings(helper.source))
            if helper else err(f"unknown helper: {revision or name}"))


def test_helper(
    name: str, inputs: list[dict] | None = None, item_ids: list[str] | None = None
) -> dict[str, Any]:
    """Exercise a helper on inputs=[{args: [...], kwargs: {...}}] or full dataset item_ids.
    Returns values/errors in requested order, with item_id or input_index on every result.
    Never computes detector scores or judges.
    """
    rt = get_runtime()
    helper = rt.helpers.get(name)
    if helper is None:
        return err(f"unknown helper: {name}")
    cases = list(inputs or [])
    if len(cases) + len(item_ids or []) > 20:
        return err("test at most 20 inputs per call")
    if item_ids:
        from autofa.data.trace_fields import derive_trace_fields
        from autofa.mcp.tools_eval import _fetch_items_by_id

        rows = list(_fetch_items_by_id(rt.dataset, set(item_ids), max_items=None))
        by_id = {str(row.get("_item_id")): row for row in rows}
        if set(item_ids) - by_id.keys():
            return err(f"missing dataset items: {sorted(set(item_ids) - by_id.keys())}")
        policy = get_field_policy(rt.config)
        cases.extend(
            {
                "item_id": item_id,
                "args": [wrap_item_ban_sentinels(derive_trace_fields(by_id[item_id]), policy)],
            }
            for item_id in item_ids
        )
    validate_source(helper.source, get_field_policy(rt.config), helper=True)
    fn = compile_detector(
        helper.source,
        entrypoint=helper.entrypoint,
        bundle=helper.bundle or None,
        policy=get_field_policy(rt.config),
    )
    results = []
    for index, case in enumerate(cases):
        identity = {"item_id": case["item_id"]} if "item_id" in case else {"input_index": index}
        try:
            args, kwargs = deepcopy(case.get("args", [])), deepcopy(case.get("kwargs", {}))
            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()
            check_schema(dict(bound.arguments), helper.input_schema, "arguments")
            value = bounded_result(fn(*args, **kwargs), helper.output_schema)
            results.append({"ok": True, **identity, "value": value})
        except Exception as exc:  # noqa: BLE001
            results.append({"ok": False, **identity, "error": f"{type(exc).__name__}: {exc}"})
    return ok(name=helper.name, revision=helper.revision, results=results,
              evidence_warnings=evidence_budget_warnings(helper.source))


def retain_as_helper(
    detector_name: str,
    name: str,
    reason: str,
    source: str | None = None,
    entrypoint: str = "",
    description: str = "",
    purpose: str = "predicate",
    output_schema: dict | None = None,
) -> dict[str, Any]:
    """Retain useful code from an active or archived detector as an unscored helper.
    Optional source extracts just the reusable part. Original claims/reviews stay archived.
    For composed or LLM detectors, extract deterministic code with source= explicitly.
    """
    from autofa.session.modes_ledger import get_mode
    from autofa.session.polishing_guard import polishing_tool_guard

    rt = get_runtime()
    if blocked := polishing_tool_guard(rt, "retain_as_helper", {"name": name}):
        return blocked
    fn = rt.library.get(detector_name)
    record = (
        fn.to_dict() if fn else (get_mode(rt.run.run_dir(), detector_name) or {}).get("detector")
    )
    if not record or not reason.strip():
        return err("an existing active/archived detector and a retention reason are required")
    if source is None and (record.get("depends_on") or execution_bundle(record)):
        return err(
            "extract deterministic reusable code with source=; preserve dependencies explicitly"
        )
    out = submit_helper(
        name=name,
        source=source or record["source"],
        entrypoint=entrypoint,
        description=description or reason,
        purpose=purpose,
        output_schema=output_schema,
    )
    if out["ok"]:
        helper = rt.helpers.get(out["revision"])
        helper.origin.update(
            {"detector_name": detector_name, "detector_revision": digest(record), "reason": reason}
        )
        rt.persist_library()
    return out


def deprecate_helper(name: str, reason: str) -> dict[str, Any]:
    """Hide a catalog helper from default retrieval; existing pinned consumers still run."""
    rt = get_runtime()
    helper = rt.helpers.get(name)
    if helper is None or not reason.strip():
        return err("known helper name and reason required")
    helper.status = "deprecated"
    helper.origin["deprecation_reason"] = reason
    rt.persist_library()
    return ok(name=name, revision=helper.revision, status=helper.status)
