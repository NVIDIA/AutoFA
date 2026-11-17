# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final scoring runs selected detectors once, reusing saved results only while identity matches."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.dependencies import build_bundle, review_source
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import DetectorAbstained
from autofa.data.source_errors import SourceDataError
from autofa.mcp import tools_eval
from autofa.mcp.tools_eval import _evaluation_context
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.detector_checkpoints import DETECTOR_EXECUTION_VERSION, DetectorCheckpoints
from autofa.session.evaluation_refresh import refresh_evaluation_contexts
from autofa.session.recovery import RunInterruptedError

from support.evaluation import _LLM_SOURCE, _REGEX_SOURCE, _FakeEvalDataset, _runtime


def _forbid_judging(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Final detector scoring must not launch a causal review")

    monkeypatch.setattr(tools_eval, "_judge_function", unexpected)
    monkeypatch.setattr("autofa.metrics.claude_judge.judge_behavior_with_claude", unexpected)


def test_final_scoring_only_executes_explicit_selection_and_preserves_skipped_scores(
    tmp_path, monkeypatch,
):
    selected = FeatureFunction(name="selected", description="Selected", source=_LLM_SOURCE,
                               kind="llm")
    skipped = FeatureFunction(name="skipped", description="Skipped", source=_LLM_SOURCE,
                              kind="llm")
    _runtime(tmp_path, monkeypatch, sample=12, workers=2, functions=[selected, skipped], n=60)
    _forbid_judging(monkeypatch)
    assert tools_eval.evaluate_functions(names=[selected.name, skipped.name])["ok"]
    before = deepcopy(skipped.scores)

    result = tools_eval.final_evaluate_functions(names=[selected.name])

    assert result["ok"] and result["censused"] == [selected.name]
    assert selected.scores["sampled"] is False
    assert selected.scores["n_scored"] == 60
    assert skipped.scores == before and skipped.scores["sampled"] is True
    # Explicit full scoring remains available for an otherwise skipped candidate.
    assert tools_eval.final_evaluate_functions(names=[skipped.name])["censused"] == [skipped.name]


@pytest.mark.parametrize("names,success", [([], True), (["unknown"], False)])
def test_empty_or_invalid_final_selection_never_scans_the_library(
    tmp_path, monkeypatch, names, success,
):
    fn = FeatureFunction(name="untouched", description="Untouched",
                         source=_REGEX_SOURCE, kind="python")
    rt = _runtime(tmp_path, monkeypatch, sample=40, workers=1, functions=[fn])
    before = deepcopy(rt.library.to_dict())
    monkeypatch.setattr(tools_eval, "_evaluate_functions",
                        lambda **kwargs: pytest.fail("An empty/invalid selection must not scan"))
    assert tools_eval.final_evaluate_functions(names=names)["ok"] is success
    assert rt.library.to_dict() == before


def test_final_scoring_finishes_sampled_capped_and_unscored_detectors_once(tmp_path, monkeypatch):
    sampled = FeatureFunction(name="sampled", description="Sampled", source=_LLM_SOURCE, kind="llm")
    capped = FeatureFunction(name="capped", description="Capped",
                             source=_REGEX_SOURCE, kind="python")
    unscored = FeatureFunction(name="unscored", description="Unscored",
                              source="def f(item): return False", kind="regex")
    blocked = FeatureFunction(name="blocked", description="Awaiting screen",
                              source=_LLM_SOURCE, kind="llm")
    rt = _runtime(tmp_path, monkeypatch, sample=12, workers=2,
                  functions=[sampled, capped, unscored, blocked], n=60)
    _forbid_judging(monkeypatch)
    monkeypatch.setattr(
        "autofa.session.candidate_screen.ensure_candidate_screen",
        lambda rt, fn, **kwargs: {
            "status": "inconclusive" if fn.name == "blocked" else "not_required",
            "evaluation_allowed": fn.name != "blocked",
        },
    )
    assert tools_eval.evaluate_functions(names=[sampled.name, capped.name], limit=24)["ok"]
    assert sampled.scores["sampled"] and sampled.scores["n_scored"] == 12
    assert not capped.scores["sampled"] and capped.scores["n_scored"] == 24
    assert not capped.scores["evaluation_scope_complete"]
    assert not unscored.scores
    # The discovery screen stays enforced, but does not keep a candidate out of final scoring.
    discovery = tools_eval.evaluate_functions(names=[blocked.name], mode="full")
    assert not discovery["ok"] and discovery["candidate_screen_blocked"]
    assert not blocked.scores

    result = tools_eval.final_evaluate_functions()

    assert result["ok"] and set(result["censused"]) == {fn.name for fn in rt.library.functions}
    assert result["n"] == 4
    for fn in rt.library.functions:
        assert fn.scores["sampled"] is False
        assert fn.scores["evaluation_scope_complete"] is True
        assert fn.scores["n_scored"] == 60
        assert "evaluation_item_ids" not in fn.scores
        assert "behavior_n_judged" not in fn.scores
    original = deepcopy(rt.library.to_dict())

    def unexpected(*args, **kwargs):
        pytest.fail("Repeated finalization must reuse completed corpus scans")

    monkeypatch.setattr(tools_eval, "evaluate_functions", unexpected)
    assert tools_eval.final_evaluate_functions()["censused"] == []
    assert rt.library.to_dict() == original


@pytest.mark.parametrize("change", ["claim", "other_run", "stale"])
def test_final_scoring_refreshes_a_census_that_is_not_current(tmp_path, monkeypatch, change):
    fn = FeatureFunction(name="flag", source=_REGEX_SOURCE, description="Original claim",
                         kind="python")
    rt = _runtime(tmp_path, monkeypatch, sample=12, workers=1, functions=[fn], n=60)
    _forbid_judging(monkeypatch)
    assert tools_eval.final_evaluate_functions()["ok"]
    if change == "claim":
        fn.description = "Revised claim"
    elif change == "other_run":
        fn.scores["evaluation_run_id"] = "an-earlier-run"
    else:
        fn.scores["evaluation_stale"] = True

    result = tools_eval.final_evaluate_functions()

    assert result["ok"] and result["censused"] == [fn.name]
    assert fn.scores["evaluation_run_id"] == rt.run.run_id
    assert not fn.scores.get("evaluation_stale")
    assert fn.scores["evaluation_claim_hash"] == claim_hash(
        source=review_source(fn), description=fn.description,
        mechanism=fn.mechanism or fn.description or fn.name,
    )
    assert fn.scores["n_scored"] == 60


def test_final_census_preserves_abstentions_and_does_not_retry_them_automatically(
    tmp_path, monkeypatch,
):
    fn = FeatureFunction(name="partial", description="Partial evidence", kind="python", source="""
def detect(item, llm_client=None):
    if item['response_text'] == 'BAD':
        library.abstain('The relevant trace evidence is unavailable')
    return False
""")
    rt = _runtime(tmp_path, monkeypatch, sample=12, workers=1, functions=[fn], n=60)
    _forbid_judging(monkeypatch)

    result = tools_eval.final_evaluate_functions()

    assert result["ok"] and result["censused"] == [fn.name]
    scores = fn.scores
    assert scores["evaluation_scope_complete"] is True
    assert scores["evaluation_complete"] is False
    assert scores["n_scored"] == 45
    assert scores["n_insufficient"] == scores["n_unresolved"] == 15
    assert scores["errors"] == 0
    assert scores["c"] == 15  # The other 15 failed attempts remain unknown, not negative.
    assert scores["d"] == 30
    assert scores["metrics_scope"] == "resolved_items"
    before = deepcopy(rt.library.to_dict())
    monkeypatch.setattr(tools_eval, "evaluate_functions",
                        lambda **kwargs: pytest.fail("Do not retry a completed scan on save"))
    repeated = tools_eval.final_evaluate_functions()
    assert repeated["ok"] and repeated["censused"] == []
    assert rt.library.to_dict() == before


@pytest.mark.parametrize("stored_version,dependency", [
    pytest.param(4, False, id="outdated-llm"),
    pytest.param(4, True, id="outdated-llm-dependency"),
    pytest.param(DETECTOR_EXECUTION_VERSION, False, id="current"),
])
def test_final_scoring_refreshes_only_outdated_llm_execution_censuses(
    tmp_path, monkeypatch, stored_version, dependency,
):
    peer = FunctionLibrary(functions=[
        FeatureFunction(name="peer", description="Peer", source=_LLM_SOURCE, kind="python"),
    ])
    llm = FeatureFunction(
        name="semantic", description="Semantic predicate", kind="python",
        source=("def detect(item): return library.fire('peer', item)"
                if dependency else _LLM_SOURCE),
        depends_on=["peer"] if dependency else [],
        meta={"deps": {"peer": _LLM_SOURCE},
              "dependency_bundle": build_bundle(peer, depends_on=["peer"])}
        if dependency else {},
    )
    python = FeatureFunction(name="exact", description="Exact predicate",
                             source=_REGEX_SOURCE, kind="python")
    rt = _runtime(tmp_path, monkeypatch, sample=12, workers=2, functions=[llm, python], n=60)
    _forbid_judging(monkeypatch)
    assert tools_eval.final_evaluate_functions()["ok"]
    assert llm.scores["evaluation_execution_version"] == DETECTOR_EXECUTION_VERSION
    llm.scores["evaluation_execution_version"] = stored_version
    # Old deterministic censuses remain current; only LLM-backed execution changed.
    python.scores["evaluation_execution_version"] = 4
    before_python = deepcopy(python.scores)
    before_reviews = deepcopy(rt.library.review_history)
    calls = []
    evaluate = tools_eval._evaluate_functions

    def record(names=None, **kwargs):
        calls.append(names)
        return evaluate(names=names, **kwargs)

    monkeypatch.setattr(tools_eval, "_evaluate_functions", record)
    result = tools_eval.final_evaluate_functions()
    refresh = stored_version != DETECTOR_EXECUTION_VERSION
    assert result["ok"] and result["censused"] == ([llm.name] if refresh else [])
    assert calls == ([[llm.name]] if refresh else [])
    assert llm.scores["evaluation_execution_version"] == DETECTOR_EXECUTION_VERSION
    assert python.scores == before_python
    assert rt.library.review_history == before_reviews
    assert tools_eval.final_evaluate_functions()["censused"] == []
    assert calls == ([[llm.name]] if refresh else [])


# --- refreshing legacy evaluation contexts -------------------------------------


# Reads a forwarded trace field through a helper; the marker comment makes it LLM-backed.
_FORWARDED_SOURCE = (
    "def text_of(row):\n    return row.get('response_text') or ''\n"
    "def detect(item, llm_client=None):\n"
    "    return 'BAD' in text_of(item)\n    # llm_client.classify marker\n"
)


def _scored_runtime(tmp_path, monkeypatch, scope="full", *, source=_LLM_SOURCE):
    fn = FeatureFunction(name="flag", description="Failure behavior", source=source,
                         kind="llm", mechanism="The required action is left undone")
    rt = _runtime(tmp_path, monkeypatch, sample=12, workers=1, functions=[fn])
    rt.dataset = _FakeEvalDataset([
        {"model_id": "m", "benchmark": "bench", "problem_id": str(i), "run_idx": 0,
         "is_correct": i % 2 == 0, "response_text": "BAD" if i < 24 else "fine",
         "_item_id": f"case-{i}"} for i in range(60)
    ])
    result = (tools_eval.evaluate_round_functions(fraction=.3) if scope == "round_validation"
              else tools_eval.evaluate_functions(names=[fn.name], mode=scope))
    assert result["ok"], result
    return rt, fn


def _forbid(*args, **kwargs):
    pytest.fail("A complete unchanged prediction cache must avoid detector execution")


def _drop_one_prediction(rt):
    with sqlite3.connect(rt.run.run_dir() / "detector_results.sqlite3") as db:
        key, payload = db.execute("SELECT key, result FROM results LIMIT 1").fetchone()
        item_id = json.loads(payload)["localisation"]["item_id"]
        db.execute("DELETE FROM results WHERE key = ?", (key,))
    return item_id


@pytest.mark.parametrize("scope,source", [
    ("full", _LLM_SOURCE), ("capped", _LLM_SOURCE), ("round_validation", _LLM_SOURCE),
    pytest.param("full", _FORWARDED_SOURCE, id="forwarded-trace-inputs"),
])
def test_context_refresh_reaggregates_saved_predictions_on_original_scope(
    tmp_path, monkeypatch, scope, source,
):
    rt, fn = _scored_runtime(tmp_path, monkeypatch, scope, source=source)
    before = deepcopy(fn.scores)
    cohort = before.get("evaluation_item_ids")
    panel = before["search_panel"]["cohort_fingerprint"]
    fn.scores["evaluation_context"] = "legacy-opaque-context"
    # Discovery may already have advanced; that must not resample scoring rows.
    rt.run.round += 2
    rt.run.exploration_seed += 321
    history_length = len(rt.scores_history)
    monkeypatch.setattr(tools_eval, "run_detector", _forbid)
    result = refresh_evaluation_contexts(rt)

    assert result["refreshed_names"] == [fn.name]
    assert fn.scores["evaluation_context"] == tools_eval._evaluation_context(rt)
    assert fn.scores["evaluation_context"] != "legacy-opaque-context"
    assert len(rt.scores_history) == history_length + 1
    assert fn.scores["sampled"] == before["sampled"]
    assert fn.scores["n_scored"] == before["n_scored"]
    assert fn.scores.get("evaluation_item_ids") == cohort
    assert fn.scores["search_panel"]["cohort_fingerprint"] == panel
    assert {key: fn.scores[key] for key in ("recall", "ppv_at_pi0", "a", "b", "c", "d")} == {
        key: before[key] for key in ("recall", "ppv_at_pi0", "a", "b", "c", "d")
    }
    if scope == "round_validation":
        assert fn.scores["round_validation"]["fraction"] == .3
        assert fn.scores["round_validation"]["panel_fingerprint"] == (
            before["round_validation"]["panel_fingerprint"])
    persisted = json.loads(rt.working_library_path().read_text())
    assert persisted["functions"][0]["scores"]["evaluation_context"] == (
        fn.scores["evaluation_context"])
    assert refresh_evaluation_contexts(rt)["refreshed_names"] == []
    assert len(rt.scores_history) == history_length + 1


def test_interrupted_context_refresh_preserves_saved_measurements_until_resumed(
    tmp_path, monkeypatch,
):
    rt, fn = _scored_runtime(tmp_path, monkeypatch)
    missing = _drop_one_prediction(rt)
    fn.scores["evaluation_context"] = "legacy-context"
    rt.persist_library()
    before = deepcopy(fn.to_dict())
    saved = rt.working_library_path().read_bytes()
    history_length = len(rt.scores_history)
    execute, calls = tools_eval.run_detector, []

    def interrupted(source, item, **kwargs):
        calls.append(item["_item_id"])
        raise ConnectionResetError("Connection lost while refreshing the missing prediction")

    monkeypatch.setattr(tools_eval, "run_detector", interrupted)
    with pytest.raises(RunInterruptedError):
        refresh_evaluation_contexts(rt)
    assert calls == [missing]
    assert fn.to_dict() == before
    assert rt.working_library_path().read_bytes() == saved
    assert len(rt.scores_history) == history_length

    def resumed(source, item, **kwargs):
        calls.append(item["_item_id"])
        return execute(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", resumed)
    assert refresh_evaluation_contexts(rt)["refreshed_names"] == [fn.name]
    assert calls == [missing, missing]
    assert fn.scores["evaluation_context"] == tools_eval._evaluation_context(rt)
    assert fn.scores["n_scored"] == before["scores"]["n_scored"]


def test_context_refresh_retries_unresolved_case_without_converting_it_to_a_negative(
    tmp_path, monkeypatch,
):
    execute, calls = tools_eval.run_detector, []

    def abstain(source, item, **kwargs):
        calls.append(item["_item_id"])
        if item["_item_id"] == "case-0":
            raise DetectorAbstained("The required source evidence is unavailable")
        return execute(source, item, **kwargs)

    monkeypatch.setattr(tools_eval, "run_detector", abstain)
    rt, fn = _scored_runtime(tmp_path, monkeypatch)
    assert fn.scores["n_insufficient"] == fn.scores["n_unresolved"] == 1
    assert fn.scores["n_scored"] == 59
    fn.scores["evaluation_context"] = "legacy-context"
    calls.clear()

    assert refresh_evaluation_contexts(rt)["refreshed_names"] == [fn.name]
    assert calls == ["case-0"]
    assert fn.scores["n_insufficient"] == fn.scores["n_unresolved"] == 1
    assert fn.scores["n_scored"] == 59 and fn.scores["errors"] == 0
    assert fn.scores["evaluation_complete"] is False
    assert fn.scores["evaluation_context"] == tools_eval._evaluation_context(rt)


def test_context_refresh_does_not_evaluate_candidates_without_saved_context(tmp_path, monkeypatch):
    rt, fn = _scored_runtime(tmp_path, monkeypatch)
    never_evaluated = FeatureFunction(
        name="new", description="Not yet evaluated", source=_LLM_SOURCE,
        scores={"evaluated": False, "evaluation_context": "legacy-context"},
    )
    missing_context = FeatureFunction.from_dict({**deepcopy(fn.to_dict()), "name": "no_context"})
    missing_context.scores.pop("evaluation_context")
    rt.library.functions.extend([never_evaluated, missing_context])
    untouched = deepcopy([candidate.to_dict() for candidate in rt.library.functions[1:]])
    fn.scores["evaluation_context"] = "legacy-context"
    monkeypatch.setattr(tools_eval, "run_detector", _forbid)

    assert refresh_evaluation_contexts(rt)["refreshed_names"] == [fn.name]
    assert [candidate.to_dict() for candidate in rt.library.functions[1:]] == untouched


# --- saved evaluation identity across restarts and settings -------------------

# These identity tests freeze transport routing explicitly; they neither create
# a global MCP runtime nor probe a live detector endpoint (including subprocesses).


def context_runtime():
    return SimpleNamespace(
        config=Config(),
        dataset=SimpleNamespace(
            parquet_paths=[], model_id="test-model", benchmarks=["benchmark"],
            evaluation_fingerprint=lambda: "source-content-fingerprint",
        ),
        run=SimpleNamespace(run_id="saved-evaluation", eval_limit=200, exploration_base_seed=42),
    )


def test_evaluation_context_is_identical_across_python_hash_seeds():
    script = """
import json
from types import SimpleNamespace
from autofa.config import Config
from autofa.mcp.tools_eval import _evaluation_context

rt = SimpleNamespace(
    config=Config(),
    dataset=SimpleNamespace(parquet_paths=[], model_id="model", benchmarks=["bench"],
                            evaluation_fingerprint=lambda: "source-content-fingerprint"),
    run=SimpleNamespace(run_id="saved-run", eval_limit=200, exploration_base_seed=42),
)
contexts = {}
for mode in ("content", "metadata", "custom"):
    rt.config.detector_fields = mode
    rt.config.allowed_fields = ["prompt_text", "response_text", "input_text"]
    rt.config.extra_allowed_fields = ["additional_evidence"]
    rt.config.banned_fields = ["private_field"]
    contexts[mode] = _evaluation_context(rt, routing_identity=None)
print(json.dumps(contexts, sort_keys=True))
"""
    contexts = []
    for seed in ("0", "1", "17"):
        env = dict(os.environ, PYTHONHASHSEED=seed,
                   PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        completed = subprocess.run(
            [sys.executable, "-c", script], env=env, check=True,
            capture_output=True, text=True, timeout=30,
        )
        contexts.append(json.loads(completed.stdout))
    assert contexts[0] == contexts[1] == contexts[2]
    assert len(set(contexts[0].values())) == 3


def test_field_policy_membership_order_does_not_change_context():
    rt = context_runtime()
    rt.config.detector_fields = "custom"
    rt.config.allowed_fields = ["prompt_text", "response_text", "input_text"]
    rt.config.banned_fields = ["private_field", "sensitive_field"]
    original = _evaluation_context(rt, routing_identity=None)
    rt.config.allowed_fields.reverse()
    rt.config.banned_fields.reverse()
    assert _evaluation_context(rt, routing_identity=None) == original


def test_saved_identities_record_only_a_sole_benchmark_in_its_slot():
    from autofa.session.smoke_work import dataset_benchmark_slot

    assert dataset_benchmark_slot(SimpleNamespace(benchmarks=["bench"])) == "bench"
    assert dataset_benchmark_slot(SimpleNamespace(benchmarks=["a", "b"])) is None
    assert dataset_benchmark_slot(SimpleNamespace(benchmarks=[])) is None


def test_loaded_single_benchmark_run_keys_saved_identities_by_its_slot(tmp_path, monkeypatch):
    """Saved evaluation, diagnostic and smoke caches of a one-benchmark run stay reusable."""
    from autofa.mcp import runtime
    from autofa.session import diagnostic_receipts, smoke_work

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(runtime, "_RUNTIME", None)
    parquet = tmp_path / "data.parquet"
    parquet.write_bytes(b"rows")
    run = RunConfig.create([parquet], "model-a", ["bench"], Config())
    rt = runtime.load_runtime_from_run(RunConfig.load(run.run_id), Config())
    stat = parquet.stat()
    dataset = [[[str(parquet.resolve()), stat.st_size, stat.st_mtime_ns]],
               "model-a", "bench", ["bench"]]
    assert diagnostic_receipts._dataset_identity(rt) == smoke_work.digest(dataset)

    digested, digest = [], smoke_work.digest

    def recording_digest(value):
        digested.append(value)
        return digest(value)

    monkeypatch.setattr(smoke_work, "digest", recording_digest)
    smoke_work.SmokeWork(rt, source="", bundle=None, policy=None, client=None, uses_llm=False)
    assert [value["dataset"] for value in digested
            if isinstance(value, dict) and "dataset" in value] == [dataset]

    # The evaluation context hashes the same slot; without it the context changes.
    context = _evaluation_context(rt, routing_identity=None)
    monkeypatch.setattr(smoke_work, "dataset_benchmark_slot", lambda dataset: None)
    assert context is not None
    assert _evaluation_context(rt, routing_identity=None) != context


@pytest.mark.parametrize("target,setting,value", [
    ("config", "detector_fields", "metadata"),
    ("config", "banned_fields", ["response_text"]),
    ("config", "feature_model", "different-detector-model"),
    ("config", "llm_eval_sample", 123),
    ("dataset", "evaluation_fingerprint", lambda: "different-source-content"),
])
def test_changed_source_policy_or_evaluation_settings_invalidates_context(target, setting, value):
    rt = context_runtime()
    original = _evaluation_context(rt, routing_identity=None)
    setattr(getattr(rt, target), setting, value)
    assert _evaluation_context(rt, routing_identity=None) != original


def test_saved_scores_on_a_cache_prepared_by_an_earlier_adapter_are_never_reused(tmp_path):
    """Such a cache can hold graders' calls: no saved census on it skips reading rows."""
    path = tmp_path / "source-v14" / "dataset.parquet"
    path.parent.mkdir()
    path.write_bytes(b"")
    (path.parent / "manifest.json").write_text(json.dumps({
        "adapter_version": 14, "output_sha256": "unchanged-prepared-source",
    }))
    rt = context_runtime()
    rt.dataset.parquet_paths = [path]
    with pytest.raises(SourceDataError, match="re-prepare the source"):
        _evaluation_context(rt, routing_identity=None)


# --- detector checkpoint keys: request policy and transport -------------------


@pytest.mark.parametrize("options,legacy_changes,reused", [
    ({"reasoning_effort": "none"}, {}, False),
    ({"extra_body": {"reasoning": {"enabled": False}}},
     {"request_options_scope": "all_detector_chat_calls_v1"}, False),
    ({}, {}, True),
    # Execution contract 4 could mistake window markers for original defects.
    ({}, {"version": 4}, False),
])
def test_legacy_checkpoint_reused_only_when_request_policy_and_execution_contract_match(
    tmp_path, options, legacy_changes, reused,
):
    source = ("def detect(item, llm_client=None): "
              "return llm_client.chat.completions.with_raw_response.create()")
    item = {"response_text": "synthetic evidence"}
    cache = DetectorCheckpoints(tmp_path, model="model", endpoint="https://provider.example/v1",
                                request_options=options)
    legacy_identity = {key: value for key, value in cache.identity.items()
                       if key != "request_options_scope"}
    assert set(legacy_changes) <= set(cache.identity)  # Each change targets a keyed field.
    legacy_identity.update(legacy_changes)
    legacy_key = hashlib.sha256(json.dumps(
        [legacy_identity, source, None, item], sort_keys=True, default=str,
    ).encode()).hexdigest()
    cache.put(legacy_key, True, {"item_id": "old"})
    current_key = cache.key(source, None, item)
    if not reused:
        assert current_key != legacy_key and cache.get(current_key) is None
        cache.put(current_key, False, {"item_id": "new"})
        resumed = DetectorCheckpoints(tmp_path, model="model", endpoint="https://provider.example/v1",
                                      request_options=options)
        assert resumed.get(resumed.key(source, None, item)) == (False, {"item_id": "new"})
        assert resumed.get(legacy_key) == (True, {"item_id": "old"})
    else:
        assert current_key == legacy_key
        assert cache.get(current_key) == (True, {"item_id": "old"})


def test_checkpoint_keys_follow_resolved_transport_without_losing_saved_results(tmp_path):
    route = {"transport_contract_version": 2, "api": "auto",
             "decisions_available": None, "responses_available": None}
    source, item = "def detect(item, llm_client=None): return True", {"response_text": "evidence"}
    cache = DetectorCheckpoints(tmp_path, model="model", endpoint="https://provider.invalid/v1",
                                api="auto", routing_identity=lambda: dict(route))
    unknown = cache.key(source, None, item)
    route.update(decisions_available=True, responses_available=True)
    resolved = cache.key(source, None, item)
    assert resolved != unknown
    cache.put(resolved, True, {"route": "decisions"})
    assert cache.get(unknown) is None
    resumed = DetectorCheckpoints(tmp_path, model="model", endpoint="https://provider.invalid/v1",
                                  api="auto", routing_identity=lambda: dict(route))
    assert resumed.get(resumed.key(source, None, item)) == (True, {"route": "decisions"})
    route["decisions_available"] = False
    assert resumed.get(resumed.key(source, None, item)) is None
    assert resumed.get(resolved) == (True, {"route": "decisions"})
