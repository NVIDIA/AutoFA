# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exploration advice follows the current library, never a type quota or history."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from autofa.config import Config, RunConfig
from autofa.core.dependencies import build_bundle, digest
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.data.parquet_store import Dataset
from autofa.mcp import tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.session.diversity import detector_exploration, load_detector_exploration

DETERMINISTIC = {
    "name": "exact_failure",
    "kind": "regex",
    "source": "def detect(item, llm_client=None):\n    return 'failed' in item['response_text']\n",
}
LLM = {
    "name": "semantic_failure",
    "kind": "regex",  # Authored code takes precedence over the legacy default kind.
    "source": (
        "def detect(item, llm_client=None):\n"
        "    return llm_client.classify_options('rubric', item['response_text'], ['yes', 'no'])\n"
    ),
}


@pytest.mark.parametrize(
    ("functions", "expected_counts", "family"),
    [
        # Every non-LLM kind belongs to the one deterministic family.
        ([{**DETERMINISTIC, "kind": kind} for kind in ("regex", "python", None)]
         + [LLM], {"deterministic": 3, "llm": 1}, "llm"),
        ([DETERMINISTIC] + [LLM] * 3, {"deterministic": 1, "llm": 3}, "deterministic"),
        ([DETERMINISTIC, LLM], {"deterministic": 1, "llm": 1}, None),
        ([], {"deterministic": 0, "llm": 0}, None),
    ],
    ids=["fewer-llm", "fewer-deterministic", "balanced", "empty"],
)
def test_less_represented_family_is_conditional_on_evidence(functions, expected_counts, family):
    before = deepcopy(functions)
    result = detector_exploration(functions, llm_available=True)
    assert result["counts"] == expected_counts
    assert result["suggested_family"] == family
    assert functions == before


def _pinned_parent(role, uses_llm):
    node = {"role": role, "source": (LLM if uses_llm else DETERMINISTIC)["source"],
            "helpers": {}, "detectors": {}}
    if role == "helper":
        node.update(entrypoint="detect", input_schema={}, output_schema={})
    revision = digest(node)
    edge, invocation = (("helpers", "helpers.call") if role == "helper"
                        else ("detectors", "library.fire"))
    bundle = {"version": 1, "nodes": {revision: node}, "helpers": {}, "detectors": {}}
    bundle[edge]["parent"] = revision
    return {"source": f"def detect(item):\n    return {invocation}('parent', item)\n",
            "meta": {"dependency_bundle": bundle}}


# A hybrid or composed detector that uses an LLM anywhere in its pinned graph counts once as LLM.
@pytest.mark.parametrize("function,uses_llm", [
    ({"kind": "python", "source": (  # Hybrid: deterministic gate plus one LLM decision.
        "from autofa.trace import observations\n"
        "def detect(item, llm_client=None):\n"
        "    text = observations(item)\n"
        "    return 'failed' in text and llm_client.decide('rubric', text)\n"
    )}, True),
    ({"source": "def detect(item):\n    return library.fire('parent', item)\n",
      "depends_on": ["parent"],
      "meta": {"deps": {"parent": LLM["source"]}, "dependency_bundle": build_bundle(
          FunctionLibrary(functions=[FeatureFunction(name="parent", description="",
                                                     source=LLM["source"])]),
          depends_on=["parent"])}}, True),  # Composed dependency, as submission saves it.
    (_pinned_parent("helper", uses_llm=True), True),
    (_pinned_parent("detector", uses_llm=False), False),
], ids=["hybrid", "composed-dependency", "pinned-llm-helper", "pinned-deterministic-detector"])
def test_llm_use_is_classified_from_the_pinned_execution_graph(function, uses_llm):
    counts = detector_exploration([function])["counts"]
    assert counts == {"deterministic": int(not uses_llm), "llm": int(uses_llm)}


@pytest.mark.parametrize("availability,family", [(False, None), (None, "llm")])
def test_unavailable_or_unknown_endpoint_makes_llm_suggestion_conditional(availability, family):
    result = detector_exploration([DETERMINISTIC], llm_available=availability)
    assert result["counts"] == {"deterministic": 1, "llm": 0}
    assert result["suggested_family"] == family


def test_loader_uses_current_library_each_time_and_does_not_modify_it(tmp_path):
    path = tmp_path / "library.json"
    history = tmp_path / "experiment_history.json"
    history.write_text(json.dumps({"functions": [LLM] * 20}))
    for functions, family in (([DETERMINISTIC], "llm"), ([LLM], "deterministic")):
        contents = json.dumps({"version": 1, "functions": functions})
        path.write_text(contents)
        result = load_detector_exploration(tmp_path, llm_available=True)
        assert result["suggested_family"] == family
        assert path.read_text() == contents
    assert sorted(p.name for p in tmp_path.iterdir()) == [history.name, path.name]


def test_missing_library_is_empty_without_creating_files(tmp_path):
    result = load_detector_exploration(tmp_path / "missing")
    assert result["counts"] == {"deterministic": 0, "llm": 0}
    assert result["suggested_family"] is None
    assert list(tmp_path.iterdir()) == []
    assert load_detector_exploration(None) == result


# A non-list envelope or a record without string source must not count as empty/deterministic.
@pytest.mark.parametrize("contents", ["broken json", "[]", '{"functions": [{}]}'])
def test_malformed_library_never_fabricates_empty_counts(tmp_path, contents):
    path = tmp_path / "library.json"
    path.write_text(contents)
    result = load_detector_exploration(tmp_path)
    assert result["counts"] is result["suggested_family"] is None
    assert path.read_text() == contents


def test_unreadable_library_never_fabricates_empty_counts(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "read_text", denied)
    result = load_detector_exploration(tmp_path)
    assert result["counts"] is result["suggested_family"] is None


@pytest.mark.parametrize("malformed", [
    {**DETERMINISTIC, "source": 123},
    {**DETERMINISTIC, "meta": {"deps": {"broken": 3}}},
    {**DETERMINISTIC, "meta": {"dependency_bundle": {}}},
], ids=["source", "legacy-dependencies", "dependency-bundle"])
def test_direct_guidance_with_malformed_metadata_is_safe_and_never_partial(malformed):
    functions = [DETERMINISTIC, LLM, malformed]
    before = deepcopy(functions)
    result = detector_exploration(functions, llm_available=True)
    assert result["counts"] is result["suggested_family"] is None
    assert functions == before


def detector(name, *, llm=False):
    return FeatureFunction(name=name, description="Guesses an answer",
                           source=(LLM if llm else DETERMINISTIC)["source"],
                           kind="llm" if llm else "regex")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    rt = Runtime(
        dataset=Dataset(parquet_paths=[], model_id="m"),
        library=FunctionLibrary(), config=Config(),
        run=RunConfig(run_id="mix", parquet_paths=[], model_id="m", min_failure_modes=0),
        llm_readiness={"available": True},
    )
    set_runtime(rt)
    return rt


def test_tools_update_guidance_from_current_library_without_requiring_scores(runtime):
    # Old evaluation snapshots and retired functions do not determine the next nudge.
    runtime.scores_history = [{"points": [], "frontier_value": 0.0, "hypervolume": 0.0,
                               "n_functions": 0, "evaluation_complete": True}]
    runtime.library.functions = [detector("regex"), detector("trace")]
    before = deepcopy([fn.to_dict() for fn in runtime.library.functions])
    for tool in (tools_library.list_library_functions, tools_library.pareto_status):
        guidance = tool()["detector_exploration"]
        assert guidance["counts"] == {"deterministic": 2, "llm": 0}
        assert guidance["suggested_family"] == "llm"
    assert [fn.to_dict() for fn in runtime.library.functions] == before

    runtime.library.functions = [detector("semantic", llm=True)]
    for tool in (tools_library.list_library_functions, tools_library.pareto_status):
        guidance = tool()["detector_exploration"]
        assert guidance["counts"] == {"deterministic": 0, "llm": 1}
        assert guidance["suggested_family"] == "deterministic"


def test_next_round_returns_actionable_guidance_and_respects_endpoint_availability(
    runtime, monkeypatch,
):
    monkeypatch.setattr("autofa.session.round_validation.validate_round",
                        lambda rt: {"ok": True, "message": "Guidance-only fixture checked."})
    runtime.library.functions = [detector("rule")]
    assert tools_library.advance_round()["refinement_pending"]
    next_round = tools_library.advance_round(validation_response="Retain the checked rule.")
    assert next_round["round"] == 1
    assert next_round["detector_exploration"]["suggested_family"] == "llm"
    assert next_round["detector_exploration"]["message"] in next_round["message"]

    runtime.llm_readiness = {"available": False}
    assert tools_library.advance_round()["refinement_pending"]
    next_round = tools_library.advance_round(validation_response="Retain the checked rule.")
    assert next_round["detector_exploration"]["suggested_family"] is None
