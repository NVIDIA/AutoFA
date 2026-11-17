# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus health / parse-coverage diagnostics (Step 0 of the trace-blindness fix).

Reports dialect parse rates, surface byte shares, and whether the corpus looks
like ATIF agent traces vs flattened gym output. Feeds preflight, A4 stats, and
the input-surface diversity axis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from autofa.data.trace_fields import surface_byte_shares, surfaces_present
from autofa.data.trajectory import looks_like_agent_json, parse_trajectory
from autofa.diagnostics import CRITICAL, WARNING, get_diagnostics


def _response_text(item: dict[str, Any]) -> Any:
    from autofa.data.input_view import response_value

    return response_value(item)


def measure_parse_coverage(
    items: Iterable[dict[str, Any]],
    *,
    record_diagnostics: bool = True,
) -> dict[str, Any]:
    """Measure dialect coverage and surface byte shares over a sample of items."""
    n = 0
    parsed = 0
    unparsed_structured = 0
    plain = 0
    empty = 0
    dialects: Counter[str] = Counter()
    surface_totals: Counter[str] = Counter()
    surface_present_counts: Counter[str] = Counter()
    warnings: list[str] = []
    total_response_chars = 0

    for it in items:
        n += 1
        text = _response_text(it)
        if not text:
            empty += 1
            continue
        import json

        total_response_chars += len(text if isinstance(text, str) else json.dumps(text))
        shares = surface_byte_shares(it)
        for s, nbytes in shares.items():
            surface_totals[s] += nbytes
        for s in surfaces_present(it):
            surface_present_counts[s] += 1

        result = parse_trajectory(text)
        if result.ok:
            parsed += 1
            if result.dialect:
                dialects[result.dialect] += 1
        elif looks_like_agent_json(text):
            unparsed_structured += 1
            if result.warning and result.warning not in warnings:
                warnings.append(result.warning)
        else:
            plain += 1

    parse_rate = (parsed / n) if n else 0.0
    structured_rate = ((parsed + unparsed_structured) / n) if n else 0.0
    out: dict[str, Any] = {
        "n_sampled": n,
        "parsed": parsed,
        "parse_rate": round(parse_rate, 4),
        "looks_like_agent_json_but_unparsed": unparsed_structured,
        "plain_text": plain,
        "empty": empty,
        "dialects": dict(dialects),
        "surface_byte_totals": dict(surface_totals),
        "surface_present_counts": dict(surface_present_counts),
        "mean_response_chars": int(total_response_chars / n) if n else 0,
        "trajectory_capable": parse_rate >= 0.5,
        "warnings": warnings,
    }

    if n and unparsed_structured / n >= 0.15:
        out["warning"] = (
            f"{unparsed_structured}/{n} sampled responses look like structured agent "
            "traces but no dialect matched. Supported: OpenHands/ATIF and generic "
            "steps envelopes. Detectors will see raw text; get_trajectory will return nothing."
        )
    elif warnings:
        out["warning"] = warnings[0]

    if record_diagnostics and n:
        diag = get_diagnostics()
        if parse_rate == 0 and structured_rate >= 0.15:
            diag.critical(
                "dialect_unparsed",
                (
                    f"0/{n} responses parsed as agent traces "
                    f"({unparsed_structured} look structured but unmatched)."
                ),
                context={"n_sampled": n, "unparsed_structured": unparsed_structured},
                remediation=(
                    "get_trajectory / autofa.trace will return nothing on this corpus. "
                    "Add a dialect parser or use read_field_chunk on raw response_text."
                ),
            )
        elif parse_rate < 0.15 and structured_rate < 0.15:
            diag.info(
                "plain_text_corpus",
                f"Corpus looks like plain text (parse_rate={parse_rate:.0%}); "
                "trajectory tools are not needed.",
                context={"n_sampled": n, "plain_text": plain},
            )
        elif parse_rate < 0.5:
            diag.warning(
                "dialect_partial",
                f"Only {parsed}/{n} responses parsed as agent traces "
                f"(dialects={dict(dialects)}).",
                context=out,
                remediation="Inspect unparsed rows with get_item / read_field_chunk.",
            )
        else:
            diag.info(
                "dialect_ok",
                f"{parsed}/{n} responses parsed "
                f"(dialects={dict(dialects) or 'n/a'}).",
                context={"n_sampled": n, "dialects": dict(dialects)},
            )

        # Surface coverage signal for the input-surface diversity axis.
        if parse_rate >= 0.3:
            obs_share = surface_totals.get("observations", 0)
            total_share = sum(surface_totals.values()) or 1
            if obs_share / total_share >= 0.3:
                diag.info(
                    "observations_heavy",
                    (
                        f"Observations are ~{100 * obs_share / total_share:.0f}% of "
                        "trace bytes in the sample — prefer detectors that read them."
                    ),
                    context={"surface_byte_totals": dict(surface_totals)},
                )

    return out


def detect_corpus_domain(
    items: Iterable[dict[str, Any]],
    coverage: dict[str, Any] | None = None,
) -> str:
    """Classify the corpus domain so mechanism templates stay in scope."""
    from autofa.session.exploration import domain_for_corpus

    trajectory_ok = bool((coverage or {}).get("trajectory_capable"))
    tools: set[str] = set()
    for it in items:
        text = _response_text(it)
        if not text or not looks_like_agent_json(text):
            continue
        result = parse_trajectory(text)
        if result.ok:
            tools.update(t.name for t in result.tools)
        if len(tools) >= 40:
            break
    return domain_for_corpus(trajectory_parsing=trajectory_ok, tool_names=tools)


def capabilities_from_coverage(coverage: dict[str, Any]) -> dict[str, Any]:
    """Translate coverage into a capability summary for the agent / preflight."""
    trajectory_ok = bool(coverage.get("trajectory_capable"))
    surfaces = sorted(
        s for s, n in (coverage.get("surface_present_counts") or {}).items() if n
    )
    return {
        "trajectory_parsing": trajectory_ok,
        "dialects": coverage.get("dialects") or {},
        "surfaces_populated": surfaces,
        "llm_backend": None,  # filled by caller
        "parse_rate": coverage.get("parse_rate"),
        "warning": coverage.get("warning"),
        "hint": (
            "Use get_trajectory / get_trajectory_step / autofa.trace for step walks."
            if trajectory_ok
            else (
                "Trajectory parsing unavailable on this corpus — read raw fields via "
                "read_field_chunk / get_item. Do not invent schema-token heuristics."
            )
        ),
    }


# Re-export severity constants for callers that want to mirror them.
__all__ = [
    "measure_parse_coverage",
    "capabilities_from_coverage",
    "detect_corpus_domain",
    "CRITICAL",
    "WARNING",
]
