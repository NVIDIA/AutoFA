# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector labels match the publication contract; retired labels fail clearly."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.dependencies import build_bundle, execution_bundle, review_source, validate_bundle
from autofa.core.library import FUNCTION_KINDS, FeatureFunction, FunctionLibrary
from autofa.core.sandbox import run_detector
from autofa.data.parquet_store import Dataset
from autofa.data.remote_library_api import (
    RemoteLibraryError,
    _json_payload,
    _validate_library_file,
)
from autofa.mcp import server, tools_eval, tools_library
from autofa.mcp.runtime import Runtime, set_runtime
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.detector_checkpoints import DetectorCheckpoints
from autofa.session.repair_workflow import _version
from autofa_library_server.library_identity import same_library_content
from autofa_library_server.schemas import validate_library_payload

# Free-form strategy labels saved by pre-GA runs; current code rejects them.
RETIRED_KINDS = ("hybrid", "deterministic", "trace", "composed", "cross_field")
SOURCE = "def detect(item):\n    return 'signal' in item.get('response_text', '')\n"
LLM_SOURCE = (
    "def detect(item, llm_client=None):\n"
    "    text = item.get('response_text') or ''\n"
    "    return llm_client.classify('classify behavior', text) == 'yes'\n"
)


def _payload(kind):
    return {"version": 1, "model_id": "test-model", "benchmark": "test-suite",
            "functions": [{"name": "signal", "description": "Signal", "source": SOURCE,
                           "kind": kind}]}


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path)
    live = Runtime(
        dataset=Dataset(parquet_paths=[]), library=FunctionLibrary(),
        run=RunConfig(run_id="function-kind-contract", parquet_paths=[], model_id="m"),
        config=Config(),
    )
    set_runtime(live)
    return live


def _composed_library():
    parent = FeatureFunction(name="parent", description="Parent", source=LLM_SOURCE,
                             kind="python")
    library = FunctionLibrary(functions=[parent])
    bundle = build_bundle(library, depends_on=[parent.name])
    child = FeatureFunction(
        name="child", description="Child", mechanism="Shared behavior",
        source="def detect(item):\n    return library.fire('parent', item)\n",
        kind="python", depends_on=[parent.name], meta={"dependency_bundle": bundle},
        scores={"recall": 0.1, "behavior_purity": 1.0, "claim_hash": "saved"},
    )
    library.upsert(child)
    return library


# None omits the field, which keeps the historical regex default.
@pytest.mark.parametrize("kind", ["llm", None])
def test_supported_or_omitted_labels_survive_publication_and_local_catalog(kind):
    payload = _payload(kind)
    if kind is None:
        del payload["functions"][0]["kind"]
    original = deepcopy(payload)
    uploaded = json.loads(_json_payload(payload))
    assert validate_library_payload(uploaded) == ("test-model", ["test-suite"], 1)
    assert uploaded == original
    assert FunctionLibrary.from_dict(uploaded).get("signal").kind == (kind or "regex")
    assert payload == original


@pytest.mark.parametrize("kind", RETIRED_KINDS)
def test_retired_labels_fail_clearly_on_load_publication_download_and_catalog(tmp_path, kind):
    payload = _payload(kind)
    path = tmp_path / "library.json"
    path.write_text(json.dumps(payload))
    remedy = re.escape(
        f"Invalid function kind {kind!r} for detector 'signal'; expected 'regex', 'llm' or "
        "'python'. Retired strategy labels ran as Python: set kind to 'python' or recreate "
        "the detector."
    )
    with pytest.raises(ValueError, match=remedy):
        FunctionLibrary.load(path)
    with pytest.raises(RemoteLibraryError, match=remedy):
        _json_payload(path)
    with pytest.raises(RemoteLibraryError, match=remedy):
        _validate_library_file(path)
    with pytest.raises(ValueError, match=remedy):
        validate_library_payload(payload)
    assert json.loads(path.read_text()) == payload


@pytest.mark.parametrize("kind", ["LLM", None, {}], ids=["wrong-case", "null", "unhashable"])
def test_unknown_or_malformed_kinds_are_rejected_before_publication_or_import(kind):
    payload = _payload(kind)
    with pytest.raises(ValueError, match="kind"):
        validate_library_payload(payload)
    with pytest.raises(RemoteLibraryError, match="kind"):
        _json_payload(payload)
    with pytest.raises(ValueError, match="kind"):
        FeatureFunction(name="signal", description="Signal", source=SOURCE, kind=kind)
    with pytest.raises(ValueError, match="kind"):
        FunctionLibrary.from_dict(payload)


def test_python_submission_retains_llm_execution_policy(rt):
    result = tools_library.submit_function(
        "semantic", LLM_SOURCE, description="Behavior detector", kind="python",
        auto_evaluate=False,
    )
    assert result["ok"], result
    fn = rt.library.get("semantic")
    assert fn.kind == "python"
    assert fn.meta["strategy"] == "llm"
    assert tools_eval.is_llm_detector(fn)
    assert FunctionLibrary.load(rt.run.run_dir() / "library.json").get(fn.name).kind == "python"


@pytest.mark.parametrize("kind", ["custom-engine", *RETIRED_KINDS])
def test_invalid_submission_kind_leaves_library_unchanged(rt, kind):
    result = tools_library.submit_function(
        "semantic", LLM_SOURCE, kind=kind, auto_evaluate=False,
    )
    assert not result["ok"]
    assert repr(kind) in result["error"] and "Use 'python'" in result["error"]
    assert rt.library.functions == []


def test_unchanged_resubmission_preserves_review_and_repair_identity(rt):
    existing = FeatureFunction(
        name="semantic", description="Behavior detector", mechanism="Shared behavior",
        source=LLM_SOURCE, kind="python", scores={"recall": .1, "behavior_purity": 1.0},
    )
    rt.library.upsert(existing)
    before_version = _version(existing.to_dict())
    before_review = review_source(existing)
    result = tools_library.submit_function(
        existing.name, LLM_SOURCE, description=existing.description,
        mechanism=existing.mechanism, kind="python", auto_evaluate=False,
    )
    assert result["ok"], result
    saved = rt.library.get(existing.name)
    assert saved.kind == "python"
    assert saved.scores == existing.scores
    assert _version(saved.to_dict()) == before_version
    assert review_source(saved) == before_review


def test_composed_local_roundtrip_preserves_review_and_repair_identities(tmp_path):
    library = _composed_library()
    fn = library.get("child")
    reviewed = review_source(fn)
    repair = _version(fn.to_dict())
    before = deepcopy(library.to_dict())
    path = tmp_path / "library.json"
    library.save(path)
    restored = FunctionLibrary.load(path)
    assert restored.to_dict() == before
    assert review_source(restored.get("child")) == reviewed
    assert _version(restored.get("child").to_dict()) == repair


@pytest.mark.parametrize("form", ["mapping", "text", "bytes", "path"])
def test_upload_preserves_pinned_dependencies_and_cached_evidence(tmp_path, form):
    library = _composed_library()
    original = deepcopy(library.to_dict())
    text = json.dumps(original)
    path = tmp_path / "library.json"
    path.write_text(text)
    supplied = {"mapping": original, "text": text, "bytes": text.encode(), "path": path}[form]
    uploaded = json.loads(_json_payload(supplied))
    expected = deepcopy(original)
    # The child has saved coverage; publication ranks it ahead of its unscored parent.
    expected["functions"] = [expected["functions"][1], expected["functions"][0]]
    assert uploaded == expected
    assert library.to_dict() == original
    assert json.loads(path.read_text()) == original

    old = library.get("child")
    new = FunctionLibrary.from_dict(uploaded).get("child")
    assert execution_bundle(new) == execution_bundle(old)
    validate_bundle(execution_bundle(new), source=new.source)
    assert review_source(new) == review_source(old)
    old_hash, new_hash = (
        claim_hash(description=fn.description, mechanism=fn.mechanism, source=review_source(fn))
        for fn in (old, new)
    )
    assert old_hash == new_hash
    cache = DetectorCheckpoints(tmp_path, model="model", endpoint="https://example.test")
    item = {"item_id": "task-1", "response_text": "evidence"}
    old_key = cache.key(old.source, None, item, bundle=execution_bundle(old))
    cache.put(old_key, True, {"item_id": "task-1"})
    new_key = cache.key(new.source, None, item, bundle=execution_bundle(new))
    assert cache.get(new_key) == (True, {"item_id": "task-1"})
    assert tools_eval.is_llm_detector(new)


@pytest.mark.parametrize("kind", RETIRED_KINDS)
def test_retired_label_in_a_pinned_node_still_loads_publishes_and_downloads(tmp_path, kind):
    # Pre-GA publication relabelled only top-level kinds; pinned nodes kept theirs.
    saved = SimpleNamespace(source=SOURCE, kind=kind, meta={})
    bundle = build_bundle(SimpleNamespace(get=lambda name: saved), depends_on=["parent"])
    # Listed in publication order, so publication leaves the saved library unchanged.
    library = FunctionLibrary(model_id="test-model", benchmark="test-suite", functions=[
        FeatureFunction(
            name="child", description="Child",
            source="def detect(item):\n    return library.fire('parent', item)\n",
            kind="python", depends_on=["parent"], meta={"dependency_bundle": bundle},
        ),
        FeatureFunction(name="parent", description="Parent", source=SOURCE, kind="python"),
    ])
    payload = library.to_dict()
    restored = FunctionLibrary.from_dict(deepcopy(payload))
    assert restored.to_dict() == payload
    pinned = execution_bundle(restored.get("child"))
    validate_bundle(pinned, source=restored.get("child").source)
    assert {node["kind"] for node in pinned["nodes"].values()} == {kind}
    assert validate_library_payload(payload) == ("test-model", ["test-suite"], 2)
    uploaded = json.loads(_json_payload(payload))
    assert uploaded == payload
    assert same_library_content(payload, uploaded)
    path = tmp_path / "library.json"
    path.write_text(json.dumps(uploaded))
    _validate_library_file(path)


def test_retired_label_in_a_pinned_parent_runs_but_fails_clearly_as_a_parent(rt):
    # Pre-GA publication relabelled only top-level kinds; pinned nodes kept theirs.
    saved = SimpleNamespace(source=SOURCE, kind="hybrid", meta={})
    bundle = build_bundle(SimpleNamespace(get=lambda name: saved), depends_on=["parent"])
    child = FeatureFunction(
        name="child", description="Child",
        source="def detect(item):\n    return library.fire('parent', item)\n",
        kind="python", depends_on=["parent"], meta={"dependency_bundle": bundle},
    )
    assert run_detector(child.source, {"response_text": "signal"}, bundle=bundle) is True
    with pytest.raises(ValueError, match="'hybrid'.*set kind to 'python' or recreate the"):
        tools_eval.evaluation_uses_llm(child, within="parent", runtime=rt)


def test_both_harnesses_advertise_only_server_supported_kinds(monkeypatch):
    import claude_agent_sdk
    import jsonschema

    wrapped = []
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server",
                        lambda **kwargs: wrapped.extend(kwargs["tools"]))
    server.build_mcp_server()
    claude = next(tool for tool in wrapped if tool.name == "submit_function")
    codex = next(spec for spec in server.codex_tool_specs() if spec["name"] == claude.name)
    native = codex["inputSchema"]
    assert native["properties"]["kind"]["enum"] == list(FUNCTION_KINDS)
    assert native["properties"]["kind"]["default"] == "regex"
    assert claude.input_schema == native
    for kind in FUNCTION_KINDS:
        args = {"name": "signal", "source": SOURCE, "kind": kind}
        jsonschema.validate(args, native)
    for kind in (*RETIRED_KINDS, "custom-engine"):
        args = {"name": "signal", "source": SOURCE, "kind": kind}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(args, native)
