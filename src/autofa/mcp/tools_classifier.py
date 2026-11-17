# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-facing bounded classifier comparison and run-scoped adoption."""

from __future__ import annotations

import json
from typing import Any, Literal

from autofa.mcp.runtime import err, get_runtime, ok
from autofa.session.classifier_profiles import (
    adopt_profile,
    compare_profiles,
    operation_lock,
    profile_status,
)


def classifier_profiles(
    action: Literal["status", "compare", "adopt", "recover"] = "status", *,
    profile: str | None = None, name: str | None = None, source: str | None = None,
    depends_on: list[str] | None = None, uses_helpers: dict[str, str] | None = None,
    item_ids: list[str] | None = None, expected_outputs: dict[str, bool] | None = None,
    expected_abstentions: list[str] | None = None, diagnosis: str = "",
    comparison_id: str | None = None,
) -> dict[str, Any]:
    """Diagnose classifier capability on fixed author controls, never assign grades.

    status: list ordered configured profiles and persistent run budgets.
    compare: supply the next untested profile, source OR saved detector name, 2–30 explicit
    controls including positive and negative expectations, and a diagnosis of the
    source-supported mismatch after checking evidence delivery/implementation.
    The same source and inputs run under both profiles; expectations are frozen.
    Passing baseline controls end the experiment without fallback model calls.
    A completed failed alternative permits the next approved tier on the same frozen
    source and controls, without adopting the failed classifier. Interrupted trials
    must be resumed; changing controls does not authorize skipping a tier.
    adopt: select a completed eligible comparison_id. This changes only this run,
    preserves saved scores, and directs ordinary re-evaluation/review to use the new profile.
    recover: when the detector service is unavailable, supply the same source-supported
    positive/negative controls and diagnosis. Bounded approved alternatives may replace
    the unavailable service without a successful baseline comparison. This is availability
    recovery, never a measured quality improvement. Normal evaluation/review remains required.
    Discovery only: compare/adopt are closed during final polishing or after completion.
    """
    rt = get_runtime()
    try:
        if action == "status":
            return ok(**profile_status(rt))
        if action not in {"compare", "adopt", "recover"}:
            return err("action must be status, compare, adopt or recover")
        from autofa.session.polishing_guard import active_polishing

        if active_polishing(rt) and action != "recover":
            return err("Classifier comparisons are closed during final polishing")
        target_path = rt.run.run_dir() / "discovery_target.json"
        if target_path.exists() and json.loads(target_path.read_text()).get("campaign_complete"):
            from autofa.session.campaign_policy import campaign_status

            if not campaign_status(rt).get("continue_discovery"):
                return err("Classifier comparisons are closed after campaign completion")
        lock = operation_lock(rt)
        if not lock.acquire(blocking=False):
            return err("Another classifier comparison or adoption is active")
        try:
            if action == "adopt":
                return ok(**adopt_profile(rt, comparison_id))
            request = {"source": source, "name": name, "depends_on": depends_on,
                       "uses_helpers": uses_helpers, "item_ids": item_ids,
                       "expected_outputs": expected_outputs,
                       "expected_abstentions": expected_abstentions}
            if action == "recover":
                from autofa.session.classifier_recovery import recover_classifier

                return ok(**recover_classifier(rt, request=request, diagnosis=diagnosis))
            return ok(**compare_profiles(rt, profile=profile, request=request, diagnosis=diagnosis))
        finally:
            lock.release()
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return err(str(exc))
