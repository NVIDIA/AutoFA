# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recompute saved aggregates through normal evaluation when their context changes."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from autofa.defaults import setting_value
from autofa.ui.evaluation_progress import evaluation_progress
from autofa.util.cancellation import check_cancelled


def _saved_capped_limit(scores: dict) -> int:
    limit = scores.get("eval_limit")
    if limit is None and scores.get("evaluation_checkpoint_complete") is True:
        # Older checkpoints omit the explicit cap. A completed, truncated
        # checkpoint exhausted that cap, even when every row in its frame ran.
        if scores.get("sampled") is True:
            limit = scores.get("sample_frame_n")
        elif scores.get("sampled") is False:
            resolved, unresolved = scores.get("n_scored"), scores.get("n_unresolved")
            if type(resolved) is int and type(unresolved) is int:
                limit = resolved + unresolved
    if type(limit) is not int or limit < 1:
        raise RuntimeError("Cannot refresh capped evaluation without its original row limit.")
    return limit


def refresh_evaluation_contexts(
    rt: Any, *, progress: Callable[[dict], None] | None = None, names: set[str] | None = None,
) -> dict[str, Any]:
    """Refresh each previously scored scope, reusing ordinary prediction checkpoints.

    Opaque legacy context hashes cannot prove current aggregate validity. Revisit
    the source using the evaluator instead of relabeling old scores. Its normal
    commit and cancellation behavior preserves recoverability on interruption.
    Missing contexts and never-evaluated candidates retain ordinary eligibility
    handling; this helper does not manufacture a prior evaluation scope for them.
    """
    from autofa.mcp.tools_eval import _evaluate_functions, _evaluation_context, resolve_eval_limit

    candidates = [fn for fn in rt.library.functions
                  if (names is None or fn.name in names)
                  and (fn.scores or {}).get("evaluated") is True
                  and (fn.scores or {}).get("evaluation_context")]
    refreshed = []
    for fn in candidates:
        check_cancelled()
        # A preceding detector call can resolve a previously unknown API capability.
        context = _evaluation_context(rt)
        if not context:
            raise RuntimeError(
                "Cannot refresh saved evaluation: current source identity is unavailable.")
        scores = fn.scores or {}
        if scores.get("evaluation_context") == context:
            continue
        kwargs = {"names": [fn.name], "_finalizing": True}
        if scores.get("eval_mode") == "capped" and (
            scores.get("eval_limit") is not None or scores.get("eval_truncated") is True
        ):
            limit = _saved_capped_limit(scores)
            maximum = setting_value(rt.config, "max_eval_limit")
            if resolve_eval_limit(limit, cfg_max=maximum) != limit:
                raise RuntimeError(
                    f"Cannot refresh {fn.name} with its saved row limit of {limit}. "
                    f"Restore max_eval_limit to at least {limit} before refreshing.")
            kwargs.update(mode="capped", limit=limit)
        elif scores.get("sampled") is False:
            kwargs["mode"] = "full"
        elif (scores.get("round_validation") or {}).get("fraction") is not None:
            fraction = scores["round_validation"]["fraction"]
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not (
                0 < fraction <= 1
            ):
                raise RuntimeError(
                    f"Cannot refresh {fn.name}: saved validation fraction is invalid.")
            kwargs["_round_fraction"] = fraction
        else:
            kwargs.update(mode="capped", limit=scores.get("eval_limit", rt.run.eval_limit))
        with evaluation_progress(progress) if progress is not None else nullcontext():
            result = _evaluate_functions(**kwargs)
        if not result.get("ok"):
            raise RuntimeError(f"Saved evaluation refresh is incomplete for {fn.name}: "
                               + str(result.get("error") or "inspect the saved detector scores"))
        refreshed.append(fn.name)
    return {"refreshed_names": refreshed}
