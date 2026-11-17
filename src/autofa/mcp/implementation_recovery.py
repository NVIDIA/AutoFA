# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded implementation recovery uses existing smoke execution and repair history."""

from __future__ import annotations

from typing import Any

from autofa.mcp.runtime import err
from autofa.session import repair_workflow


def diagnose_implementation(rt: Any, function: dict, evidence_ids: list[str],
                            expected_outputs: dict | None) -> dict:
    from autofa.mcp.tools_eval import smoke_test_feature
    from autofa.metrics.repair_feedback import detector_execution_findings

    if expected_outputs is not None and not isinstance(expected_outputs, dict):
        raise ValueError("implementation_checks must map item IDs to expected output values")
    expected = dict(expected_outputs or {})
    errors = [item for finding in detector_execution_findings(function)
              for item in finding["item_ids"]]
    ids = list(dict.fromkeys([*evidence_ids, *expected, *errors]))
    if not ids:
        raise ValueError("Provide implementation_checks with item IDs and expected outputs, "
                         "or cite a current execution error")
    if len(ids) > 30:
        raise ValueError("Implementation recovery accepts at most 30 regression cases")
    # Retain a few supported cases as controls. Actual baseline predictions are
    # preserved below; a categorical detector is not assumed to return True.
    scores = function.get("scores") or {}
    controls = [str(row["item_id"]) for row in scores.get("behavior_verdicts", [])
                if row.get("verdict") == "yes" and row.get("item_id")]
    controls = [item for item in dict.fromkeys(controls) if item not in ids][:3]
    ids = [*ids, *controls][:30]
    response = smoke_test_feature(
        name=function["name"], expected_version_hash=repair_workflow._version(function),
        item_ids=ids, expected_outputs=expected,
    )
    if not response.get("ok"):
        raise ValueError("Implementation reproduction failed: " + str(response.get("error")))
    receipt = {key: response[key] for key in ("ok", "version_hash", "results")}
    repair_workflow.implementation_diagnostic_finding(function, receipt)
    receipt["item_ids"] = ids
    receipt["expected_outputs"] = {
        row["item_id"]: row["value"] for row in response["results"]
        if row.get("status") == "value"
    } | expected
    receipt["interpretation"] = (
        "Author-specified regression expectations; no causal validation or population "
        "estimate. Preserve the claim and test the implementation before replacement.")
    return receipt


def check_implementation_submission(rt: Any, function: dict,
                                     repair_plan_id: str | None) -> dict | None:
    """A failed proposal never replaces the baseline or clears its current scores."""
    from autofa.mcp.tools_eval import smoke_test_feature

    workflow = repair_workflow.get_repair(rt.run.run_dir(), function["name"])
    plan = workflow.get("latest") or {}
    diagnostic = plan.get("implementation_diagnostics")
    if (not diagnostic or plan.get("phase") != "planned"
            or (repair_plan_id and repair_plan_id != plan["plan_id"])):
        return None
    before = plan["before"]
    if any(function.get(key) != before.get(key)
           for key in ("description", "mechanism", "depends_on", "explains")):
        return err("Implementation recovery must preserve the literal claim and relationships")
    current = rt.library.get(function["name"])
    if (current is None
            or repair_workflow._version(current.to_dict()) != plan["before_version_hash"]):
        return err("Implementation recovery baseline changed; inspect the saved repair")
    response = smoke_test_feature(
        source=function["source"], depends_on=function.get("depends_on"),
        uses_helpers=function.get("uses_helpers"), item_ids=diagnostic["item_ids"],
        expected_outputs=diagnostic["expected_outputs"],
    )
    # Error-only reproductions may have no expected values. All selected cases
    # must still execute to a value; an abstention does not repair a parser.
    response["checks_passed"] = bool(response.get("ok") and response.get("results")
                                      and not response.get("errors")
                                      and not response.get("abstentions")
                                      and not response.get("mismatches"))
    receipt = repair_workflow.record_implementation_check(
        rt.run.run_dir(), function, plan["plan_id"], response)
    if not receipt["passed"]:
        return err(
            "Implementation regression checks failed; the prior detector and scores remain "
            "unchanged. The failed proposal and checks are retained in the repair ledger.",
            implementation_check=receipt, baseline_retained=True,
            next_action={"tool": "get_function_repair", "arguments": {"name": function["name"]}},
        )
    return None
