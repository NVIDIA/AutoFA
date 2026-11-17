# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring runtimes, final review and polishing evaluation."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_eval
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics import claude_judge
from autofa.metrics.review_scores import apply_review_result
from autofa.util.atomic import atomic_write_json


# LLM eval sampling.
class _FakeEvalDataset(Dataset):
    def __init__(self, items):
        super().__init__(parquet_paths=[], model_id="m", benchmarks=None)
        self._fake_items = items

    def evaluation_fingerprint(self):
        return hashlib.sha256(json.dumps(self._fake_items, sort_keys=True).encode()).hexdigest()

    def available_columns(self):
        return set((self._fake_items or [{}])[0].keys())

    def estimate_matching_rows(self):
        return len(self._fake_items)

    def identity_index(self, *, refresh: bool = False):
        raise RuntimeError("no parquet index in tests")

    def iter_batches(self, *, batch_size=512, max_items=None, columns=None):
        items = list(self._fake_items)
        if max_items is not None:
            items = items[: max(0, int(max_items))]
        bs = max(1, int(batch_size))
        for i in range(0, len(items), bs):
            yield items[i : i + bs]


_REGEX_SOURCE = (
    "def plain(item, llm_client=None):\n"
    "    return 'BAD' in (item.get('response_text') or '')\n"
)

# Deterministic body so sampled vs exact scores are comparable, with the
# llm_client.classify marker that makes it classify as an LLM detector.
_LLM_SOURCE = (
    "def flag(item, llm_client=None):\n"
    "    return 'BAD' in (item.get('response_text') or '')\n"
    "    # llm_client.classify marker\n"
)


def _llm_flag(**changes):
    """The LLM-classified ``flag`` detector that most evaluation tests score."""
    fields = {"name": "flag", "description": "semantic behavior", "source": _LLM_SOURCE,
              "kind": "llm"}
    return FeatureFunction(**{**fields, **changes})


def _corpus(n=600):
    items = []
    for i in range(n):
        items.append(
            {
                "model_id": "m",
                "benchmark": "alpha" if i < n - 20 else "rare",
                "problem_id": str(i),
                "run_idx": 0,
                "is_correct": i % 2 == 0,
                "response_text": "BAD" if i % 4 == 1 else "fine",
                "_item_id": f"id-{i}",
            }
        )
    return items


def _scoring_runtime(tmp_path, monkeypatch):
    """Install a minimal runtime for direct ``tools_eval`` scoring calls."""
    runtime = SimpleNamespace(
        run=SimpleNamespace(run_dir=lambda: tmp_path),
        config=Config(feature_model="model", hub_base_url="https://endpoint.example/v1",
                      hub_api_key_env="ENDPOINT_API_KEY"),  # pragma: allowlist secret
    )
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: runtime)
    return runtime


def _record_calls(monkeypatch, key=lambda source, item: item["_item_id"]):
    """Record ``key(source, item)`` for each detector execution; results stay real."""
    calls, execute = [], tools_eval.run_detector

    def recorded(source, item, **kwargs):
        calls.append(key(source, item))
        return execute(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", recorded)
    return calls


def _runtime(tmp_path, monkeypatch, *, sample, workers, functions, n=600):
    """Build a runtime over ``_corpus(n)``; keep ``n`` small unless the size is the point."""
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path / "runs")
    # These detectors exercise sampling using a local predicate. Do not depend
    # on credentials from another test or the developer's global environment.
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    cfg = Config()
    cfg.llm_eval_sample = sample
    cfg.llm_eval_workers = workers
    cfg.llm_eval_min_per_stratum = 5
    rt = Runtime(
        dataset=_FakeEvalDataset(_corpus(n)),
        library=FunctionLibrary(model_id="m", benchmark="all", functions=functions),
        run=RunConfig(run_id="llm-sample", parquet_paths=[], model_id="m", benchmarks=[]),
        config=cfg,
        scores_history=[],
    )
    set_runtime(rt)
    return rt


# Judge reply giving every requirement one status.
def reply(review, status="contradicted"):
    review.read("detector_source")
    passage = review.read("response_text")["text"]
    citation = review.cite("response_text", passage)["citations"][0]
    contract = review.snapshot["claim_contract"]
    defining_ids = [row["id"] for row in contract["requirements"]
                    if row["kind"] == "defining_clause"]
    return json.dumps({"verdicts": [{
        "item_id": review.snapshot["item_id"], "snapshot_id": review.snapshot["snapshot_id"],
        "claim_contract_id": contract["contract_id"], "scope": "local",
        "claim_assessments": [
            {"requirement_id": row["id"], "status": status, "rationale": "Reviewed evidence",
             **({"defining_requirement_ids": defining_ids}
                if defining_ids and status != "supported"
                and row["kind"] in {"claim", "review_check"} else {}),
             "citations": [{"citation_id": citation["citation_id"]}], "counterevidence": []}
            for row in contract["requirements"]
        ],
    }], "behavior_specificity": 0.8})


# Final review.
def runtime(tmp_path, monkeypatch, *, n=48):
    fn = FeatureFunction(
        name="flag",
        description="The agent guesses.",
        mechanism="It acts on a guess.",
        source=_REGEX_SOURCE,
        kind="python",
    )
    rt = _runtime(tmp_path, monkeypatch, sample=20, workers=1, functions=[fn], n=n)
    rt.config.judge_workers = 1
    assert tools_eval.final_evaluate_functions()["ok"]
    return rt, fn


def judge_rows(rt, fn, monkeypatch, *, count=3, runner=None, label=True, representative=False):
    monkeypatch.setattr(claude_judge, "_run_review", runner or (lambda **kw: reply(kw["review"])))
    config, reasoning, settings = tools_eval._review_settings(rt.config)
    from autofa.metrics.judge_backend import backend_arguments
    from autofa.session.guidance import guidance_prompt

    items = [item for item in rt.dataset._fake_items if item["response_text"] == "BAD"]
    if representative:
        from autofa.session.family_coverage import cached_task_inventory
        from autofa.session.review_evidence import _diverse_ids

        by_id = {item["_item_id"]: item for item in items}
        chosen = _diverse_ids(cached_task_inventory(rt)["identity_by_item_id"], set(by_id),
                             target_n=count, seed=rt.run.exploration_seed or 0)
        items = [by_id[key] for key in chosen]
    else:
        items = items[:count]
    result = claude_judge.judge_behavior_with_claude(
        name=fn.name,
        description=fn.description,
        mechanism=fn.mechanism,
        source=fn.review_source,
        items=items,
        assigned_labels=[label] * count,
        cwd=rt.run.run_dir(),
        model=config["model"],
        cli_path=config["cli_path"],
        objective_strategy=settings["judge_objective_strategy"],
        behavior_strictness=settings["judge_behavior_strictness"],
        workers=1,
        max_chars=min(5000, rt.config.text_truncate_chars * 2),
        analysis_guidance=guidance_prompt(rt.run),
        **reasoning,
        **backend_arguments(config["harness"]),
    )
    apply_review_result(fn.scores, result, judge_harness=config["harness"])
    fn.scores["behavior_sample"] = {"target_n": count, "actual_n": count, "limited": True}
    fn.scores["behavior_review_localisations"] = {item["_item_id"]: None for item in items}
    return items, result


def record_reviews(monkeypatch, *, interrupt_at=None, status="contradicted"):
    """Answer every review with ``status``; interrupt the ``interrupt_at``-th call."""
    calls = []

    def run(**kwargs):
        calls.append(kwargs["review"].snapshot["item_id"])
        if len(calls) == interrupt_at:
            raise KeyboardInterrupt
        return reply(kwargs["review"], status=status)

    monkeypatch.setattr(claude_judge, "_run_review", run)
    return calls


def forbid(*args, **kwargs):
    pytest.fail("This operation must not run during final review")


# Polishing evaluation.
# Items case-0..case-{MATCHED_N - 1} match; even items pass and odd items fail,
# so every task has one passing and one failing attempt. The smallest stratified
# panel (two outcome strata of five) leaves a larger fresh unflagged-success pool,
# and a revision's diagnostic union stays smaller than the corpus.
CORPUS_N, MATCHED_N, PANEL_N = 60, 24, 10


@pytest.fixture
def polishing_runtime(tmp_path, monkeypatch):
    fn = FeatureFunction(name="flag", description="Failure behavior", source=_LLM_SOURCE,
                         kind="llm", mechanism="The required action is left undone")
    rt = _runtime(tmp_path, monkeypatch, sample=PANEL_N, workers=3, functions=[fn])
    rt.dataset = _FakeEvalDataset([
        {"model_id": "m", "benchmark": "bench", "problem_id": str(i // 2), "run_idx": i % 2,
         "is_correct": i % 2 == 0, "response_text": "BAD" if i < MATCHED_N else "fine",
         "_item_id": f"case-{i}"} for i in range(CORPUS_N)
    ])
    assert tools_eval.evaluate_functions()["ok"]
    before = deepcopy(fn.scores)
    panel = set(fn.scores["evaluation_item_ids"])
    assert tools_eval.final_evaluate_functions()["ok"]
    return rt, fn, before, panel


def write_phase(rt, **changes):
    phase = {
        "version": 1, "run_id": rt.run.run_id, "round": rt.run.round,
        "mode": "polishing", "polishing_id": f"{rt.run.run_id}:{rt.run.round}",
        "status": "awaiting_response", "frozen_names": ["existing"],
        "target_grade": "GOOD", "evaluated_claims": {"existing": "old-claim"},
        **changes,
    }
    atomic_write_json(rt.run.run_dir() / "round_validation.json", {"refinement": phase})
    return phase
