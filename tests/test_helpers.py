# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unscored functions, immutable composition, and passage-to-classifier execution."""

from __future__ import annotations

from copy import deepcopy

import pytest

from autofa.config import Config, RunConfig
from autofa.core.dependencies import build_bundle, digest, execution_bundle, review_source
from autofa.core.helpers import HelperLibrary
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import DetectorAbstained, compile_detector, run_detector
from autofa.data.parquet_store import Dataset
from autofa.data.remote_library_api import _validate_library_data
from autofa.mcp import tools_eval, tools_helpers, tools_library
from autofa.mcp.runtime import Runtime, load_runtime_from_run, set_runtime
from autofa.metrics.claim_contract import claim_hash
from autofa.metrics.detector_checkpoints import DetectorCheckpoints
from autofa.trace import regex_passages, render_passages


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setattr("autofa.config.runs_root", lambda: tmp_path)
    runtime = Runtime(
        dataset=Dataset(parquet_paths=[]),
        library=FunctionLibrary(),
        run=RunConfig(run_id="helpers", parquet_paths=[], model_id="m"),
        config=Config(),
    )
    set_runtime(runtime)
    return runtime


def submit(name, source, **kwargs):
    out = tools_helpers.submit_helper(name, source, **kwargs)
    assert out["ok"], out
    return out["revision"]


def detector(name, source, **kwargs):
    out = tools_library.submit_function(name, source, auto_evaluate=False, **kwargs)
    assert out["ok"], out
    return out


def run(fn, item=None, client=None):
    return run_detector(fn.source, item or {}, llm=client, bundle=execution_bundle(fn))


def test_helper_has_no_detector_lifecycle_or_dataset_requirement(rt, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a helper must never be scored or judged")

    monkeypatch.setattr(tools_eval, "evaluate_functions", forbidden)
    monkeypatch.setattr(tools_eval, "judge_function", forbidden)
    ref = submit(
        "normalize", "def normalize(text):\n    return text.strip().lower()\n",
        input_schema={"type": "object", "required": ["text"]}, output_schema={"type": "string"},
    )
    result = tools_helpers.test_helper("normalize", inputs=[{"args": [" HELLO "]}])
    assert result["results"] == [{"ok": True, "input_index": 0, "value": "hello"}]
    assert rt.library.functions == []
    data = rt.helpers.get(ref).to_dict()
    assert not {"scores", "mechanism", "example_fail_ids"} & data.keys()
    assert not (rt.run.run_dir() / "modes_ledger.jsonl").exists()


def test_two_detectors_reuse_a_helper_and_pin_versions(rt):
    first = submit("normalize", "def normalize(text):\n    return text.lower()\n")
    source = (
        "def detect(item):\n"
        '    return helpers.call("norm", item.get("response_text", "")) == "failed"\n'
    )
    for name in ("one", "two"):
        detector(name, source, uses_helpers={"norm": "normalize"})
    second = submit("normalize", "def normalize(text):\n    return text.upper()\n")
    assert first != second
    assert all(run(fn, {"response_text": "FAILED"}) for fn in rt.library.functions)
    detector("three", source, uses_helpers={"norm": "normalize"})
    assert not run(rt.library.get("three"), {"response_text": "FAILED"})
    assert tools_helpers.list_helpers()["helpers"][0]["consumers"] == ["three"]


def test_nested_diamond_preserves_different_revisions(rt):
    h1 = submit("value", "def value(text):\n    return text.lower()\n")
    a = submit("left", 'def left(text):\n    return helpers.call("value", text)\n',
               uses_helpers={"value": h1})
    h2 = submit("value", "def value(text):\n    return text.upper()\n")
    b = submit("right", 'def right(text):\n    return helpers.call("value", text)\n',
               uses_helpers={"value": h2})
    source = (
        'def detect(item):\n    return helpers.call("a", "Failed") != helpers.call("b", "Failed")\n'
    )
    detector("diamond", source, uses_helpers={"a": a, "b": b})
    assert run(rt.library.get("diamond")) is True


def test_parent_diamond_keeps_each_parent_pinned_revision(rt):
    parent_source = 'def detect(item):\n    return library.fire("shared", item)\n'
    for name, value in (("a", "True"), ("b", "False")):
        shared = FeatureFunction(name="shared", description="",
                                 source=f"def detect(item):\n    return {value}\n")
        rt.library.upsert(FeatureFunction(
            name=name, description="", source=parent_source, depends_on=["shared"],
            meta={"dependency_bundle": build_bundle(FunctionLibrary(functions=[shared]),
                                                    depends_on=["shared"])},
        ))
    source = (
        "def detect(item):\n"
        '    return (library.fire("a", item) and library.fire("b", item)\n'
        '            and "failed" in item.get("response_text", ""))\n'
    )
    detector("child", source, depends_on=["a", "b"])
    assert run(rt.library.get("child"), {"response_text": "failed"}) is False


def test_export_is_self_contained_and_preserves_unused_helpers(rt, tmp_path):
    ref = submit("identity", "def identity(text):\n    return text\n")
    submit("unused", "def unused(text):\n    return text.strip()\n")
    detector("uses", 'def detect(item):\n    return helpers.call("id", "failed") == "failed"\n',
             uses_helpers={"id": ref})
    data = rt.library.to_dict()
    _validate_library_data(data)
    assert digest(FunctionLibrary.from_dict(data).to_dict()) == digest(data)
    path = tmp_path / "export" / "library.json"
    rt.library.save(path)
    loaded = FunctionLibrary.load(path)
    assert loaded.version == 2
    assert loaded.helpers.get("unused") is not None
    loaded.helpers = HelperLibrary()
    assert run(loaded.get("uses")) is True
    assert path.with_name("helpers.json").exists()
    resumed = load_runtime_from_run(rt.run, rt.config)
    assert resumed.helpers.get("unused") is not None


def test_import_conflicting_alias_preserves_both(rt):
    first = submit("normalize", "def normalize(text):\n    return text.lower()\n")
    other = HelperLibrary.from_dict(rt.helpers.to_dict())
    second = submit("normalize", "def normalize(text):\n    return text.upper()\n")
    rt.helpers.merge(other)
    assert rt.helpers.get("normalize").revision == second
    assert rt.helpers.get(f"normalize@{first[:12]}").revision == first


def test_retain_archived_code_does_not_restore_or_score_claim(rt):
    from autofa.session.modes_ledger import get_mode

    detector("broad", 'def detect(item):\n    return "failed" in item.get("response_text", "")\n')
    assert tools_library.remove_function("broad", reason="too broad")["ok"]
    out = tools_helpers.retain_as_helper("broad", "broad_filter", "useful initial filter")
    assert out["ok"], out
    assert rt.library.get("broad") is None
    assert rt.helpers.get("broad_filter") is not None
    assert get_mode(rt.run.run_dir(), "broad")["status"] == "open"


@pytest.mark.parametrize("source", [
    pytest.param('def helper(item):\n    return item.get("is_correct")\n', id="banned-field"),
    pytest.param("def helper(text):\n    import os\n    return text\n", id="import"),
    pytest.param('def helper(item, llm_client=None):\n    return llm_client.classify("p", "t")\n',
                 id="llm-call"),
    pytest.param("def helper(text):\n    return helpers.call(text, text)\n", id="dynamic-alias"),
])
def test_invalid_helpers_are_rejected_without_scoring(rt, source):
    assert not tools_helpers.submit_helper("invalid", source)["ok"]
    assert not rt.helpers.aliases


def test_transitive_policy_check_happens_before_branch_execution(rt):
    rt.library.upsert(FeatureFunction(
        name="bad", description="", source='def detect(item):\n    return item.get("is_correct")\n',
    ))
    out = tools_library.submit_function(
        "root", "def detect(item):\n    return False\n", depends_on=["bad"], auto_evaluate=False
    )
    assert not out["ok"] and "is_correct" in out["error"]


def test_corrupted_snapshot_and_missing_alias_fail_eagerly(rt):
    ref = submit("id", "def identity(text):\n    return text\n")
    bundle = build_bundle(rt.library, uses_helpers={"id": ref})
    source = 'def detect(item):\n    return helpers.call("id", "text")\n'
    broken = deepcopy(bundle)
    broken["nodes"][ref]["source"] += "# changed\n"
    with pytest.raises(ValueError, match="corrupt"):
        compile_detector(source, bundle=broken)
    with pytest.raises(ValueError, match="undeclared"):
        compile_detector(source.replace('"id"', '"missing"'), bundle=bundle)


def test_helper_arguments_and_results_are_detached(rt):
    ref = submit("append", 'def helper(values):\n    values.append("failed")\n    return values\n')
    fn = compile_detector(
        'def detect(item):\n    return helpers.call("append", item["messages"])\n',
        bundle=build_bundle(rt.library, uses_helpers={"append": ref}),
    )
    item = {"messages": []}
    assert fn(item) == ["failed"]
    assert item == {"messages": []}


def test_contracts_reject_wrong_result_and_unsupported_schema(rt):
    ref = submit("wrong", "def helper(text):\n    return 42\n", output_schema={"type": "string"})
    out = tools_helpers.test_helper(ref, inputs=[{"args": ["x"]}])
    assert not out["results"][0]["ok"]
    assert not tools_helpers.submit_helper(
        "bad_schema", "def helper(x):\n    return x\n", output_schema={"made_up": True}
    )["ok"]


def test_dependency_upgrade_invalidates_claim_and_checkpoint(rt, tmp_path):
    first = submit("h", "def helper(text):\n    return text.lower()\n")
    source = 'def detect(item):\n    return helpers.call("h", "failed") == "failed"\n'
    detector("d", source, uses_helpers={"h": first})
    a = deepcopy(rt.library.get("d"))
    rt.library.get("d").scores = {"raw_ppv": 1.0}
    second = submit("h", "def helper(text):\n    return text.upper()\n")
    detector("d", source, uses_helpers={"h": second})
    b = rt.library.get("d")
    assert not b.scores
    assert claim_hash(source=review_source(a), description="", mechanism="") != claim_hash(
        source=review_source(b), description="", mechanism=""
    )
    cache = DetectorCheckpoints(tmp_path, model="m", endpoint="e")
    assert cache.key(source, None, {}, bundle=execution_bundle(a)) != cache.key(
        source, None, {}, bundle=execution_bundle(b)
    )


def test_body_typeerror_executes_once():
    """A TypeError raised inside the body must not trigger a second (paid) call."""
    class Client:
        calls = 0

        def classify(self, prompt, text):
            self.calls += 1
            raise TypeError("inside body")

    client = Client()
    source = 'def detect(item, llm_client=None):\n    return llm_client.classify("p", "t")\n'
    with pytest.raises(TypeError, match="inside body"):
        run_detector(source, {}, llm=client)
    assert client.calls == 1


def test_passage_coordinates_and_completeness():
    text = "é failed\nsecond failed\nthen recovered"
    item = {"reasoning_text": text}
    result = regex_passages(item, "reasoning_text", "failed", max_passages=2)
    assert result["complete"] and len(result["value"]) == 2
    for p in result["value"]:
        assert text[p["start"] : p["end"]] == p["quote"]
        assert text[p["context_start"] : p["context_end"]] == p["context"]
    assert not regex_passages(item, "reasoning_text", "failed", max_passages=1)["complete"]
    assert regex_passages({}, "reasoning_text", "failed")["status"] == "unavailable"
    assert regex_passages({"reasoning_text": ""}, "reasoning_text", "failed")["complete"]
    assert not render_passages(result, max_chars=64)["complete"]


def test_extractor_short_circuits_llm_and_abstains_on_missing(rt):
    ref = submit(
        "extract",
        "def extract(item):\n    from autofa.trace import regex_passages\n"
        '    return regex_passages(item, "reasoning_text", "failed")\n',
    )
    source = """def detect(item, llm_client=None):
    result = helpers.call("extract", item)
    if not result["complete"]:
        return library.abstain("missing evidence")
    if not result["value"]:
        return False
    text = helpers.render_passages(result)["text"]
    return llm_client.classify("failure?", text) == "yes"
"""
    detector("cascade", source, uses_helpers={"extract": ref})

    class Client:
        calls = []

        def classify(self, prompt, text):
            self.calls.append(text)
            return "yes"

    fn, client = rt.library.get("cascade"), Client()
    assert run(fn, {"reasoning_text": "tests passed"}, client) is False
    assert not client.calls
    assert run(fn, {"reasoning_text": "tests failed"}, client) is True
    assert len(client.calls) == 1
    with pytest.raises(DetectorAbstained):
        run_detector(fn.source, {}, llm=client, bundle=execution_bundle(fn), derive_trace=False)


def test_helpers_do_not_count_as_modes(rt):
    from autofa.session.discovery_target import failure_mode_target

    submit("normalize", "def helper(text):\n    return text.lower()\n")
    target = failure_mode_target(rt.library.functions, minimum=1, config=rt.config)
    assert target["n_validated_failure_modes"] == 0


def test_deprecated_helper_still_runs_in_saved_consumer(rt):
    submit("h", "def helper(text):\n    return text.lower()\n")
    detector("d", 'def detect(item):\n    return helpers.call("h", "FAILED") == "failed"\n',
             uses_helpers={"h": "h"})
    assert tools_helpers.deprecate_helper("h", "replacement available")["ok"]
    assert tools_helpers.list_helpers()["helpers"] == []
    assert run(rt.library.get("d")) is True


@pytest.mark.parametrize("kind,mode", [("regex", "full"), ("llm", "capped")])
def test_full_evaluator_keeps_abstentions_out_of_negative_bucket(
    rt, tmp_path, monkeypatch, kind, mode
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {"model_id": "m", "benchmark": "b", "problem_id": str(i), "run_idx": 0,
         "is_correct": i % 2 == 1, "response_text": text}
        for i, text in enumerate(["failure", "okay", "missing", "missing"])
    ]
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    rt.dataset = Dataset(parquet_paths=[path])
    monkeypatch.setattr(tools_eval, "_llm_backend_error", lambda: None)
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: None)
    detector(
        "abstainer",
        """def detect(item):
    text = item.get("response_text", "")
    if text == "missing":
        library.abstain("required passage unavailable")
    return "failure" in text
""",
        kind=kind,
    )
    result = tools_eval.evaluate_functions(mode=mode)
    assert result["ok"], result
    scores = rt.library.get("abstainer").scores
    assert (scores["errors"], scores["n_insufficient"], scores["n_scored"]) == (0, 2, 2)
    assert (scores["a"], scores["c"], scores["d"]) == (1, 0, 1)
    assert scores["assessment_coverage"] == 0.5
    assert scores["evaluation_complete"] is False
    assert scores["abstention_bounds"]["recall"] == [0.5, 1.0]


def test_passage_helper_obeys_invocation_field_policy():
    from autofa.core.fields import policy_from_settings

    source = """def extract(item):
    from autofa.trace import regex_passages
    return regex_passages(item, "custom_text", "failure")
"""
    policy = policy_from_settings(detector_fields="custom", allowed_fields=["custom_text"])
    fn = compile_detector(source, policy=policy)
    assert fn({"custom_text": "failure"})["value"][0]["quote"] == "failure"


@pytest.mark.parametrize("change", ["entrypoint", "schema"])
def test_imported_dependency_is_validated_even_on_unreached_branch(rt, change):
    ref = submit("test", "def extract(text):\n    return text\n")
    bundle = build_bundle(rt.library, uses_helpers={"h": ref})
    node = bundle["nodes"].pop(ref)
    if change == "entrypoint":
        node["entrypoint"] = "missing"
    else:
        node["output_schema"] = {"enum": "invalid"}
    new_ref = digest(node)
    bundle["nodes"][new_ref] = node
    bundle["helpers"]["h"] = new_ref
    with pytest.raises(ValueError):
        compile_detector("def detect(item):\n    return False\n", bundle=bundle)


def test_composed_detector_cannot_score_structured_helper_output_by_truthiness(rt):
    ref = submit("extract", "def extract(item):\n    return {'complete': False, 'value': []}\n")
    source = 'def detect(item):\n    return helpers.call("h", item)\n'
    bundle = build_bundle(rt.library, uses_helpers={"h": ref})
    with pytest.raises(ValueError, match="scalar label"):
        run_detector(source, {}, bundle=bundle)


def test_helper_examples_preserve_requested_order_and_item_ids(rt, monkeypatch):
    submit("text", 'def text(item):\n    return item.get("response_text", "")\n')
    rows = [{"_item_id": name, "response_text": name} for name in ["b", "a"]]
    monkeypatch.setattr(tools_eval, "_fetch_items_by_id", lambda *a, **k: iter(rows))
    result = tools_helpers.test_helper("text", item_ids=["a", "b", "a"])
    assert result["results"] == [
        {"ok": True, "item_id": name, "value": name} for name in ["a", "b", "a"]
    ]


@pytest.mark.parametrize("workers", [1, 2])
def test_regex_timeout_in_llm_detector_is_local_execution_error(rt, monkeypatch, workers):
    from autofa.core.sandbox import RegexTimeoutError

    fn = FeatureFunction(name="timed", description="", source="def detect(item): return False",
                         kind="llm")

    def timeout(*args, **kwargs):
        raise RegexTimeoutError("regex timed out")

    monkeypatch.setattr(tools_eval, "run_detector", timeout)
    results = tools_eval._score_items_concurrently(
        [(fn, {"_item_id": "a"}), (fn, {"_item_id": "b"})], workers=workers, llm=None
    )
    assert all(isinstance(row[3], RegexTimeoutError) for row in results)


def test_standalone_helper_catalog_can_be_imported_without_detectors(rt, tmp_path):
    submit("normalize", "def normalize(text):\n    return text.lower()\n")
    path = tmp_path / "exported_helpers.json"
    rt.helpers.save(path)
    rt.library.helpers = HelperLibrary()
    result = tools_library.library_load(str(path))
    assert result["ok"] and result["n_helpers"] == 1
    assert not rt.library.functions
    detector("imported", 'def detect(item):\n    return helpers.call("n", "Failed") == "failed"',
             uses_helpers={"n": "normalize"})
    assert run(rt.library.get("imported")) is True


def test_declared_llm_parent_kind_survives_snapshot_even_if_client_is_aliased(rt):
    rt.library.upsert(FeatureFunction(
        name="parent", description="", kind="llm",
        source="def detect(item, llm_client=None):\n    client = llm_client\n"
        '    return client.classify("failure?", item.get("response_text", "")) == "yes"',
    ))
    bundle = build_bundle(rt.library, depends_on=["parent"])
    child = FeatureFunction(
        name="child", description="", meta={"dependency_bundle": bundle},
        source='def detect(item): return library.fire("parent", item)',
    )
    rt.library.remove("parent")
    assert tools_eval.is_llm_detector(child)
