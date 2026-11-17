# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-evaluation candidate checks: cheap source-evidence prerequisites only.

No causal control screen runs before corpus evaluation. A candidate whose declared
or known evidence prerequisites are unmet is held back; every other candidate may
proceed to evaluation.
"""

from __future__ import annotations

from typing import Any


def ensure_candidate_screen(
    rt: Any, fn: Any, *, check_evidence_prerequisites: bool = True,
) -> dict:
    """Return the pre-evaluation check for ``fn``; only evidence prerequisites can block."""
    from autofa.session.evidence_prerequisites import evidence_preflight

    preflight = evidence_preflight(rt, fn) if check_evidence_prerequisites else None
    if preflight is not None and not preflight["evaluation_allowed"]:
        return preflight
    return {
        "ok": True,
        "name": fn.name,
        "status": "not_required",
        "evaluation_allowed": True,
        "provisional": True,
        "cache_reused": False,
        "controls": [],
        "feedback": "No causal control screen is required; evaluation may proceed.",
        **({"source_checks": preflight["source_checks"]} if preflight else {}),
    }


def screen_function(name: str) -> dict[str, Any]:
    """Check a candidate's source-evidence prerequisites before corpus evaluation.

    Verifies that declared evidence_prerequisites are retained on their support items
    and that no unchanged saved review already shows a missing premise. Retention does
    not establish semantic truth. Results are provisional feedback, never corpus metrics.
    submit_function and evaluate_functions apply the same check automatically; explicit
    full scoring skips it.
    """
    from autofa.mcp.runtime import err, get_runtime

    rt = get_runtime()
    fn = rt.library.get(name)
    if fn is None:
        return err(f"unknown function: {name}")
    try:
        return ensure_candidate_screen(rt, fn)
    except (OSError, ValueError) as exc:
        return err(str(exc), evaluation_allowed=False)
