# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive polishing diagnostics from the same cheap full-corpus prediction scan."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from autofa.core.dependencies import review_source
from autofa.core.sandbox import DetectorAbstained
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.judge_checkpoints import fingerprint
from autofa.session.polishing_evaluation import (
    _diagnostic_groups,
    _result_path,
    current_polishing_evaluation,
)
from autofa.session.polishing_generalization import diagnostic_controls
from autofa.session.validation_controls import polishing_controls_identity
from autofa.util.atomic import atomic_write_json


def _claim(fn: Any) -> str:
    return claim_hash(source=review_source(fn), description=fn.description or "",
                      mechanism=fn.mechanism or fn.description or fn.name)


class _CensusDiagnostic:
    def __init__(self, rt: Any, fn: Any, controls: dict, context: str, inventory: dict):
        if (not controls.get("available")
                or controls.get("scope") != "targeted_polishing_controls"
                or controls.get("run_id") != rt.run.run_id
                or controls.get("source_signature") != context
                or controls.get("controls_fingerprint") != polishing_controls_identity(controls)
                or inventory.get("inventory_complete") is not True
                or not inventory.get("corpus_fingerprint")
                or inventory["corpus_fingerprint"] != controls.get("corpus_fingerprint")):
            raise ValueError(
                "Verified current source identities and polishing controls are required.")
        self.rt, self.fn, self.context, self.claim = rt, fn, context, _claim(fn)
        self.base_controls = controls
        self.expected = set(inventory.get("item_ids") or [])
        base_ids = set(controls.get("item_ids") or [])
        if (not self.expected or len(base_ids) != len(controls.get("item_ids") or [])
                or not base_ids <= self.expected
                or {row.get("item_id") for row in controls.get("controls") or []} != base_ids
                or not set(controls.get("panel_item_ids") or []) <= base_ids):
            raise ValueError("The retained diagnostic controls do not match the verified corpus.")
        self.path = _result_path(rt, self.claim, context, controls)
        pool = {row["item_id"]: row for row in controls.get("unflagged_success_pool") or []}
        identities = inventory.get("identity_by_item_id") or {}
        tasks = inventory.get("task_by_item_id") or {}
        failed, passed = set(inventory.get("task_by_failed_item_id") or []), set(
            inventory.get("passed_item_ids") or [])
        extra = []
        for item_id in sorted(self.expected - base_ids):
            row = {
                **deepcopy(pool.get(item_id) or {}), "item_id": item_id,
                "task_id": tasks.get(item_id),
                "benchmark": (identities.get(item_id) or {}).get("benchmark"),
                "benchmark_outcome": ("passed" if item_id in passed else
                                      "failed" if item_id in failed else "unknown"),
                "categories": (["full_scan_unflagged_success"] if item_id in pool else
                               ["prediction_outcome_unavailable"]),
                "review_findings": [],
            }
            extra.append(row)
        self.extra = {
            "additional_controls": extra,
            "generalization_sample": {
                "version": 1, "status": "full_scan", "sampled": False,
                "requested_n": len(extra), "selected_n": len(extra),
                "distinct_task_n": len({row["task_id"] for row in extra
                                        if row.get("task_id") is not None}),
                "known_previous_nonmatch_n": len(set(pool) & self.expected),
                "scope": "Full-corpus prediction outcomes reused as diagnostics; no additional "
                         "detector calls and no independent causal confirmation.",
            },
        }
        self.extra["additional_controls_fingerprint"] = fingerprint({
            "controls": extra, "sampling": self.extra["generalization_sample"]})
        self.controls = diagnostic_controls(controls, self.extra)
        self.by_id = {row["item_id"]: row for row in self.controls["controls"]}
        previous = current_polishing_evaluation(rt, fn, controls)
        self.rows = {
            row["item_id"]: deepcopy(row) for row in (previous or {}).get("results") or []
            if row.get("item_id") in self.expected
            and row.get("status") in {"value", "error", "abstained", "unresolved"}
        }
        self.observed: set[str] = set()
        self.limitations = list(controls.get("unavailable_reasons") or [])
        self.error: str | None = None
        self.saved: dict | None = None
        self.checkpoint_every = max(500, len(self.expected) // 20)

    def observe(self, item: dict, prediction: Any, error: BaseException | None) -> None:
        item_id = str(item.get("_item_id") or "")
        if item_id not in self.expected or item_id in self.observed:
            raise ValueError("Full scoring returned inconsistent diagnostic item identities.")
        self.observed.add(item_id)
        if error is None:
            self.rows[item_id] = {"item_id": item_id, "status": "value", "prediction": prediction}
        else:
            self.rows[item_id] = {
                "item_id": item_id, "status": ("abstained" if isinstance(error, DetectorAbstained)
                                                else "error"),
                "prediction": None, "error": f"{type(error).__name__}: {error}",
            }
        if len(self.observed) % self.checkpoint_every == 0:
            self.checkpoint()

    def _current_census(self) -> bool:
        scores = self.fn.scores or {}
        return (self.claim == _claim(self.fn) and scores.get("evaluated") is True
                and not scores.get("evaluation_stale") and scores.get("sampled") is False
                and scores.get("evaluation_scope_complete") is True
                and scores.get("evaluation_claim_hash") == self.claim
                and scores.get("evaluation_context") == self.context
                and scores.get("evaluation_run_id") == self.rt.run.run_id
                and scores.get("task_recall_corpus_fingerprint")
                == self.base_controls.get("corpus_fingerprint"))

    def from_saved_census(self) -> bool:
        """Recover complete bucket membership without executing unchanged code."""
        if not self._current_census():
            self.error = "Saved predictions are not a current complete census of this revision."
            return False
        scores = self.fn.scores
        labels = {}
        for bucket in scores.get("buckets") or []:
            ids = [*(bucket.get("example_fail_ids") or []),
                   *(bucket.get("example_pass_ids") or [])]
            if len(set(ids)) != bucket.get("n"):
                self.error = "Saved census bucket membership is incomplete."
                return False
            for item_id in ids:
                if item_id in labels or item_id not in self.expected:
                    self.error = "Saved census membership does not match the source corpus."
                    return False
                labels[item_id] = bucket.get("bucket")
        missing = self.expected - labels.keys()
        if (len(labels) != scores.get("n_scored")
                or len(missing) != scores.get("n_unresolved", 0)):
            self.error = "Saved census outcomes do not account for every source item."
            return False
        errors = {row.get("item_id"): row.get("error")
                  for row in scores.get("error_examples") or []}
        for item_id, label in labels.items():
            self.rows[item_id] = {"item_id": item_id, "status": "value", "prediction": label}
        for item_id in missing:
            old = self.rows.get(item_id) or {}
            if old.get("status") in {"error", "abstained"}:
                continue
            self.rows[item_id] = {
                "item_id": item_id, "status": "error" if item_id in errors else "unresolved",
                "prediction": None,
                "error": (errors[item_id] if item_id in errors else
                          "Saved census records an unresolved prediction; its exact outcome "
                          "was not retained. Do not infer a negative prediction."),
            }
        if any(row["status"] == "unresolved" for row in self.rows.values()):
            self.limitations.append("Saved census has unresolved predictions whose individual "
                                    "error/abstention types are unavailable.")
        return True

    def checkpoint(self, *, complete: bool = False, census_complete: bool = False) -> dict:
        rows = []
        for item_id in sorted(self.rows):
            control = self.by_id[item_id]
            rows.append({**self.rows[item_id], **{key: control[key] for key in (
                "categories", "benchmark", "baseline_label", "review_findings",
            ) if key in control}})
        missing = sorted(self.expected - self.rows.keys())
        errors = sum(row["status"] == "error" for row in rows)
        abstentions = sum(row["status"] == "abstained" for row in rows)
        if census_complete and self._current_census():
            # A reused aggregate can retain exact totals without retaining the
            # distinction for every unresolved item. Never classify unknowns as misses.
            errors = int(self.fn.scores.get("errors") or 0)
            abstentions = int(self.fn.scores.get("n_insufficient") or 0)
        self.saved = {
            "ok": complete and not missing and not errors and not self.error,
            "complete": complete and not missing and not self.error,
            "scope": "targeted_polishing_controls", "execution_scope": "full_dataset",
            "population_metrics": False, "grade": None, "name": self.fn.name,
            "claim_hash": self.claim, "baseline_claim_hash": self.base_controls.get("claim_hash"),
            "run_id": self.rt.run.run_id, "source_signature": self.context,
            "controls_fingerprint": self.base_controls["controls_fingerprint"],
            **self.extra, "result_path": str(self.path), "n": len(self.expected),
            "panel_n": len(self.base_controls.get("panel_item_ids") or []),
            "controls_complete": self.base_controls.get("complete", False),
            "control_limitations": list(dict.fromkeys(self.limitations)),
            "validation_score_changes": self.base_controls.get("score_changes") or {},
            "completed": len(rows), "observed_in_scan": len(self.observed),
            "errors": errors, "abstentions": abstentions,
            "unknown_outcomes": sum(row["status"] == "unresolved" for row in rows),
            "missing_item_ids": missing, "results": rows,
            "groups": _diagnostic_groups(rows, self.controls, self.base_controls["selection"]),
            "guidance": "These diagnostics reuse the ordinary full-corpus prediction scan. "
                        "Inspect removed and introduced passing matches and preserve supported "
                        "cases. A passing match is not itself a causal rejection; adherence "
                        "still requires the existing review process. Population scores are "
                        "published separately by the evaluator.",
        }
        if self.error:
            self.saved["error"] = self.error
        atomic_write_json(self.path, self.saved)
        return self.saved


class PolishingCensusCollector:
    """Observer for eligible cheap detectors; never executes or judges a detector."""

    def __init__(self, rt: Any, candidates: dict[str, tuple[Any, dict]]):
        from autofa.mcp.tools_eval import _evaluation_context
        from autofa.session.family_coverage import cached_task_inventory

        self.rt, self.states, self.results = rt, {}, {}
        self.finished = False
        if not candidates:
            return
        try:
            context = _evaluation_context(rt)
            if not context:
                raise ValueError("Current source identity is unavailable.")
            inventory = cached_task_inventory(rt)
            for name, (fn, controls) in candidates.items():
                try:
                    self.states[name] = _CensusDiagnostic(rt, fn, controls, context, inventory)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self.results[name] = {"ok": False, "complete": False, "error": str(exc)}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.results.update({name: {"ok": False, "complete": False, "error": str(exc)}
                                 for name in candidates})

    def observe(self, fn: Any, item: dict, prediction: Any, error: BaseException | None) -> None:
        if state := self.states.get(fn.name):
            state.observe(item, prediction, error)

    def finish(self, result: dict) -> dict[str, dict]:
        from autofa.mcp.tools_eval import _evaluation_context

        try:
            context = _evaluation_context(self.rt)
        except (OSError, ValueError, TypeError):
            context = None
        reused = set(result.get("reused_names") or [])
        evaluated = set(result.get("evaluated_names") or [])
        for name, state in self.states.items():
            complete = False
            if context != state.context or _claim(state.fn) != state.claim:
                state.error = "Source/settings or detector revision changed during full scoring."
            elif name in reused:
                complete = state.from_saved_census()
            elif (name in evaluated and state._current_census()
                  and state.expected == state.observed):
                complete = True
            else:
                state.error = "A complete current prediction scan is still required."
            self.results[name] = state.checkpoint(complete=complete, census_complete=complete)
        self.finished = True
        return self.results

    def interrupted(self) -> None:
        if not self.finished:
            for name, state in self.states.items():
                self.results[name] = state.checkpoint()


@contextmanager
def polishing_census_diagnostics(rt: Any, candidates: dict[str, tuple[Any, dict]]):
    """Hold alongside evaluator admission; persist partial outcomes on interruption."""
    collector = PolishingCensusCollector(rt, candidates)
    try:
        yield collector
    finally:
        collector.interrupted()
