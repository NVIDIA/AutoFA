# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-axis diversity: failure mode, detector strategy, input surface."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from autofa.core.dependencies import execution_bundle, review_source, validate_bundle
from autofa.core.fields import referenced_fields
from autofa.data.trace_fields import SURFACE_FIELDS
from autofa.llm.detection import LLM_CALL_RE

_TRACE_IMPORT_RE = re.compile(r"autofa\.trace|json\.loads")
_COMPOSE_RE = re.compile(r"library\.fire|helpers\.call")


def infer_strategy(source: str, *, kind: str | None = None) -> str:
    """Infer detector strategy from source (and optional declared kind)."""
    src = source or ""
    if kind == "llm" or LLM_CALL_RE.search(src):
        return "llm"
    if _COMPOSE_RE.search(src):
        return "composed"
    if _TRACE_IMPORT_RE.search(src) or "actions_text" in src or "observations_text" in src:
        return "trace"
    fields = referenced_fields(src)
    surfaces = surfaces_from_fields(fields | _trace_helper_fields(src))
    if len(surfaces) >= 2:
        return "cross_field"
    return "regex"


def _trace_helper_fields(source: str) -> set[str]:
    out: set[str] = set()
    if re.search(r"\breasoning\s*\(|reasoning_text", source):
        out.add("reasoning_text")
    if re.search(r"\bactions\s*\(|actions_text", source):
        out.add("actions_text")
    if re.search(r"\bobservations\s*\(|observations_text", source):
        out.add("observations_text")
    if re.search(r"\bfinal_message\s*\(|final_message", source):
        out.add("final_message")
    if re.search(r"\btool_calls\s*\(", source):
        out.add("actions_text")
    if re.search(r"\bsteps\s*\(", source):
        out.update({"actions_text", "observations_text", "reasoning_text"})
    return out


def surfaces_from_fields(fields: set[str]) -> set[str]:
    found: set[str] = set()
    for surface, names in SURFACE_FIELDS.items():
        if fields & names:
            found.add(surface)
    return found


def infer_surfaces(source: str) -> set[str]:
    fields = referenced_fields(source or "") | _trace_helper_fields(source or "")
    return surfaces_from_fields(fields)


def library_coverage(functions: list[dict[str, Any]]) -> dict[str, Any]:
    """Strategy × surface coverage matrix for the working library."""
    strategy_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    rows = []
    for fn in functions:
        source = str(fn.get("source") or "")
        kind = str(fn.get("kind") or "") or None
        strategy = infer_strategy(source, kind=kind)
        surfaces = sorted(infer_surfaces(source)) or ["unknown"]
        strategy_counts[strategy] += 1
        for s in surfaces:
            surface_counts[s] += 1
            pairs[f"{strategy}×{s}"] += 1
        rows.append(
            {
                "name": fn.get("name"),
                "strategy": strategy,
                "composed": bool(_COMPOSE_RE.search(source)),
                "uses_helpers": fn.get("uses_helpers") or {},
                "surfaces": surfaces,
            }
        )

    n = len(functions) or 1
    monoculture: list[str] = []
    for strat, c in strategy_counts.items():
        if c / n >= 0.7 and n >= 3:
            monoculture.append(
                f"{c}/{len(functions)} functions use strategy={strat}; "
                "inspect whether another strategy can establish a distinction in the evidence. "
                "Strategy counts alone are not a reason to replace a working detector."
            )
    for surf, c in surface_counts.items():
        if c / n >= 0.7 and n >= 3:
            monoculture.append(
                f"{c}/{len(functions)} functions read surface={surf}; "
                "consider other surfaces when they contain evidence needed by the claim."
            )
    # Missing surfaces that exist in taxonomy.
    missing = [s for s in SURFACE_FIELDS if surface_counts.get(s, 0) == 0]
    if missing and n >= 2:
        monoculture.append(
            f"No function reads: {', '.join(missing)}. "
            "If corpus_trace_stats shows those surfaces carry signal, look there."
        )

    return {
        "n": len(functions),
        "strategies": dict(strategy_counts),
        "surfaces": dict(surface_counts),
        "matrix": dict(pairs),
        "functions": rows,
        "composed_detectors": sum(row["composed"] for row in rows),
        "warnings": monoculture,
    }


def strategy_monoculture_warning(functions: list[dict[str, Any]]) -> str | None:
    cov = library_coverage(functions)
    warns = cov.get("warnings") or []
    return warns[0] if warns else None


def _uses_llm(function: dict[str, Any]) -> bool:
    """Include the pinned execution graph, not just the root detector's label."""
    source = function.get("source", "")
    meta = function.get("meta", {})
    if meta is None:
        meta = {}
    if not isinstance(source, str) or not isinstance(meta, dict):
        raise ValueError("invalid detector source or metadata")
    deps = meta.get("deps", {})
    if deps is None:
        deps = {}
    if not isinstance(deps, dict) or any(not isinstance(s, str) for s in deps.values()):
        raise ValueError("invalid legacy detector dependencies")
    bundle = execution_bundle(function)
    if bundle is not None:
        validate_bundle(bundle, source=source)
    nodes = (bundle or {}).get("nodes", {}).values()
    declared_llm = str(function.get("kind") or "").strip().lower() == "llm"
    dependency_llm = any(str(node.get("kind") or "").strip().lower() == "llm"
                         for node in nodes)
    sources = [review_source(function), *deps.values()]
    return declared_llm or dependency_llm or any(LLM_CALL_RE.search(s) for s in sources)


def _unknown_exploration() -> dict[str, Any]:
    return {
        "counts": None,
        "suggested_family": None,
        "message": (
            "The current library could not be read reliably; its detector mix is unknown. "
            "Check the current library before using mix-based exploration guidance. "
            "Choose detectors by evidence and validation quality in the meantime."
        ),
    }


def detector_exploration(
    functions: list[dict[str, Any]], *, llm_available: bool | None = None,
) -> dict[str, Any]:
    """Suggest what to explore from the current library, without changing retention.

    A hybrid or composed detector that uses an LLM counts once as LLM. All other
    strategies count as deterministic. ``suggested_family`` identifies a less explored
    family, conditional on evidence supporting it; it is not a quality score or quota.
    """
    counts = {"deterministic": 0, "llm": 0}
    try:
        for function in functions:
            counts["llm" if _uses_llm(function) else "deterministic"] += 1
    except (AttributeError, KeyError, TypeError, ValueError, SyntaxError, RecursionError):
        # Advice must not stop library inspection or round transitions. A partial
        # count would also incorrectly imply that unclassified detectors are absent.
        return _unknown_exploration()
    deterministic, llm = counts["deterministic"], counts["llm"]
    suggested = None
    if not functions:
        message = (
            "The current library is empty. Choose the behavioral hypothesis before the detector "
            "strategy, using inspected positive and contrast evidence. Python, hybrid, "
            "or LLM implementations can be the first candidate when the evidence warrants."
        )
    else:
        message = f"Current library: {deterministic} deterministic and {llm} LLM detectors. "
        if deterministic < llm:
            suggested = "deterministic"
            message += (
                "If inspected evidence makes the distinction explicit, explore deterministic "
                "Python comparisons or state tracking."
            )
        elif llm < deterministic:
            if llm_available is not False:
                suggested = "llm"
                message += (
                    "If inspected evidence leaves a semantic distinction unresolved, explore "
                    "an LLM rubric or hybrid extraction plus semantic classification."
                )
            else:
                message += "LLM detectors are less represented, but the endpoint is unavailable."
        else:
            message += (
                "Neither family is less represented; let the evidence determine the approach."
            )
    message += (
        " Python can compare structured values, link calls to results, or track ordered state. "
        "An LLM rubric can interpret meaning or applicability. A hybrid can extract and compare "
        "explicit evidence in Python and classify only the unresolved semantic distinction. "
        "Choose the simplest implementation that preserves the claim and its coverage."
    )
    if llm_available is False:
        message += (
            " Explore deterministic detectors now and revisit LLM candidates "
            "when the endpoint is ready."
        )
    elif llm_available is None:
        message += (
            " LLM availability is unverified; check the endpoint before trying LLM candidates."
        )
    message += (
        " A supported behavioral hypothesis takes priority over family balance. "
        "Keep detectors based on validation quality; this guidance does not require "
        "a balanced count or rewriting working detectors."
    )
    return {"counts": counts, "suggested_family": suggested, "message": message}


def load_detector_exploration(
    run_dir: Path | None, *, llm_available: bool | None = None,
) -> dict[str, Any]:
    """Read only the current run's library; unavailable data never implies zero counts."""
    if run_dir is None:
        return detector_exploration([], llm_available=llm_available)
    try:
        data = json.loads((Path(run_dir) / "library.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return detector_exploration([], llm_available=llm_available)
    except (OSError, UnicodeError, ValueError, RecursionError):
        data = None
    try:
        if not isinstance(data, dict) or not isinstance(data.get("functions"), list):
            raise ValueError("invalid library envelope")
        functions = data["functions"]
        if any(not isinstance(fn, dict) or not isinstance(fn.get("source"), str)
               for fn in functions):
            raise ValueError("invalid library detector")
        return detector_exploration(functions, llm_available=llm_available)
    except (AttributeError, KeyError, TypeError, ValueError, SyntaxError, RecursionError):
        return _unknown_exploration()
