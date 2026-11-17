# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contrastive corpus-level statistics over trajectory surfaces."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from autofa.data.parquet_store import _is_correct
from autofa.data.trace_fields import surface_byte_shares
from autofa.data.trajectory import parse_trajectory

_ERROR_RE = re.compile(
    r"(?:Error|Exception|Traceback|FAILED|AssertionError|exit code[=:]?\s*\d+|"
    r"No such file|Permission denied|ModuleNotFoundError)[^\n]{0,120}",
    re.I,
)
_NORMALIZE_RE = (
    (re.compile(r"/[^\s:]+"), "<path>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<hex>"),
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "<hash>"),
    (re.compile(r"\bline \d+\b", re.I), "line N"),
    (re.compile(r"\b\d+\b"), "N"),
)


def _normalize_error(s: str) -> str:
    out = " ".join(s.split())
    for pat, repl in _NORMALIZE_RE:
        out = pat.sub(repl, out)
    return out[:160]


def _response_text(item: dict[str, Any]) -> str:
    for key in ("response_text", "generation"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def contrastive_corpus_stats(
    items: Iterable[dict[str, Any]],
    *,
    max_error_sigs: int = 20,
) -> dict[str, Any]:
    """Fail-vs-pass lift over tools, step counts, surfaces, and error signatures.

    Lift for a feature f is ``P(fail|f) / P(fail)``. Values >1 mean the feature
    is enriched in failures. These stats suggest WHERE to look — not the detector
    itself (high-lift tokens piped into a regex are shallow proxies).
    """
    items_list = list(items)
    n = len(items_list)
    if n == 0:
        return {"n_sampled": 0, "base_fail_rate": 0.0, "tools": [], "hint": "empty sample"}

    n_fail = sum(1 for it in items_list if not _is_correct(it))
    base = n_fail / n if n else 0.0

    tool_fail: Counter[str] = Counter()
    tool_pass: Counter[str] = Counter()
    step_fail: list[int] = []
    step_pass: list[int] = []
    surface_fail: dict[str, list[int]] = defaultdict(list)
    surface_pass: dict[str, list[int]] = defaultdict(list)
    err_fail: Counter[str] = Counter()
    err_pass: Counter[str] = Counter()
    parsed = 0

    for it in items_list:
        fail = not _is_correct(it)
        text = _response_text(it)
        shares = surface_byte_shares(it)
        bucket = surface_fail if fail else surface_pass
        for s, nbytes in shares.items():
            bucket[s].append(nbytes)

        result = parse_trajectory(text) if text else None
        if result and result.ok:
            parsed += 1
            n_steps = result.n_steps
            (step_fail if fail else step_pass).append(n_steps)
            for t in result.tools:
                (tool_fail if fail else tool_pass)[t.name] += t.count
            # Error signatures from observations
            for step in result._steps or []:
                obs = step.get("observation")
                if not isinstance(obs, str):
                    continue
                for m in _ERROR_RE.findall(obs):
                    sig = _normalize_error(m)
                    if len(sig) >= 8:
                        (err_fail if fail else err_pass)[sig] += 1
        else:
            (step_fail if fail else step_pass).append(0)

    def _lift(n_f: int, n_p: int) -> float | None:
        total = n_f + n_p
        if total == 0 or base <= 0:
            return None
        p_fail_given = n_f / total
        return round(p_fail_given / base, 3)

    def _mean(xs: list[int]) -> float:
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    all_tools = set(tool_fail) | set(tool_pass)
    tools_out = []
    for name in sorted(all_tools, key=lambda t: -(tool_fail[t] + tool_pass[t])):
        nf, np_ = tool_fail[name], tool_pass[name]
        tools_out.append(
            {
                "name": name,
                "n_fail_uses": nf,
                "n_pass_uses": np_,
                "lift": _lift(nf, np_),
            }
        )

    surfaces_out = {}
    for s in sorted(set(surface_fail) | set(surface_pass)):
        surfaces_out[s] = {
            "mean_bytes_fail": _mean(surface_fail.get(s, [])),
            "mean_bytes_pass": _mean(surface_pass.get(s, [])),
            "present_fail": len(surface_fail.get(s, [])),
            "present_pass": len(surface_pass.get(s, [])),
        }

    err_out = []
    for sig, nf in err_fail.most_common(max_error_sigs):
        np_ = err_pass.get(sig, 0)
        err_out.append({"signature": sig, "n_fail": nf, "n_pass": np_, "lift": _lift(nf, np_)})

    return {
        "n_sampled": n,
        "n_fail": n_fail,
        "n_pass": n - n_fail,
        "n_parsed": parsed,
        "base_fail_rate": round(base, 4),
        "steps": {
            "mean_fail": _mean(step_fail),
            "mean_pass": _mean(step_pass),
        },
        "tools": tools_out[:40],
        "surfaces": surfaces_out,
        "error_signatures": err_out,
        "hint": (
            "Lift > 1 means the feature is enriched in failures — use it to decide "
            "WHERE to look (which surface / tool / error), not as the detector itself. "
            "Piping a high-lift token into a bare regex is a shallow proxy and will be rejected."
        ),
    }
