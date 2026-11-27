# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke controls run on full evidence within a shared, cancellable budget and leave receipts."""

import asyncio
import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from autofa.core.helpers import HelperFunction
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import NullLLM
from autofa.data.parquet_store import Dataset
from autofa.mcp import runtime, server, tools_eval
from autofa.session import smoke_work
from autofa.session.diagnostic_receipts import read_diagnostic_receipt, save_diagnostic_receipt
from autofa.util.cancellation import _CURRENT, Cancellation

SOURCE = '''
def detect(item, llm_client=None):
    text = item.get("response_text", "")
    if text.endswith("INCOMPLETE"):
        library.abstain("The outcome is missing.")
    if text.endswith("BROKEN"):
        return 1 / 0
    return text.endswith("DEFECT")
'''


class ControlDataset(Dataset):
    def __init__(self):
        super().__init__(parquet_paths=[])
        self.items = [
            {"_item_id": name, "response_text": text, "is_correct": name == "recovered"}
            for name, text in [
                ("positive", "x" * 230_000 + "DEFECT"),
                ("recovered", "DEFECT then verified repair"),
                ("incomplete", "INCOMPLETE"),
                ("broken", "BROKEN"),
            ]
        ]

    def iter_batches(self, *, batch_size=512, max_items=None, columns=None):
        yield self.items[:max_items]

    def available_columns(self):
        return set(self.items[0])


@pytest.fixture
def smoke_runtime(monkeypatch, tmp_path):
    rt = SimpleNamespace(dataset=ControlDataset(), library=FunctionLibrary(),
                         config=SimpleNamespace(text_truncate_chars=100),
                         run=SimpleNamespace(run_id="smoke", round=0, max_rounds=3,
                                             run_dir=lambda: tmp_path))
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: rt)
    monkeypatch.setattr(tools_eval, "_llm_adapter", NullLLM)
    return rt


def test_explicit_controls_preserve_order_full_trace_and_abstention(smoke_runtime):
    output = tools_eval.smoke_test_feature(
        SOURCE, n=1, item_ids=["recovered", "incomplete", "positive"],
        expected_outputs={"positive": True, "recovered": False},
        expected_abstentions=["incomplete"],
    )
    assert output["ok"] and output["checks_passed"]
    assert output["n"] == output["checked"] == 3
    assert output["errors"] == output["mismatches"] == 0
    assert output["abstentions"] == 1
    assert output["histogram"] == {"False": 1, "True": 1}
    assert output["diagnostic_scope"] == "author_controls"
    assert [row["item_id"] for row in output["results"]] == [
        "recovered", "incomplete", "positive",
    ]
    assert all(row["matches_expected"] for row in output["results"])
    assert output["results"][1]["reason"] == "The outcome is missing."


def test_mismatches_and_errors_do_not_become_negative_predictions(smoke_runtime):
    output = tools_eval.smoke_test_feature(
        SOURCE, item_ids=["positive", "recovered", "incomplete", "broken"],
        expected_outputs={"positive": False, "incomplete": False, "broken": False},
    )
    assert output["ok"] and not output["checks_passed"]
    assert output["checked"] == output["mismatches"] == 3
    assert output["errors"] == output["abstentions"] == 1
    assert output["histogram"] == {"True": 1, "False": 1}
    assert [row["status"] for row in output["results"]] == [
        "value", "value", "abstained", "error",
    ]
    assert "matches_expected" not in output["results"][1]
    assert "ZeroDivisionError" in output["results"][-1]["error"]


def test_error_on_unlabelled_control_still_fails_preflight(smoke_runtime):
    output = tools_eval.smoke_test_feature(
        SOURCE, item_ids=["positive", "broken"], expected_outputs={"positive": True},
    )
    assert output["checked"] == 1 and output["mismatches"] == 0
    assert output["errors"] == 1 and output["checks_passed"] is False


@pytest.mark.parametrize("kwargs", [
    {"item_ids": []},
    {"item_ids": ["positive", "missing"]},
    {"expected_outputs": {"positive": True}},
    {"item_ids": ["positive"], "expected_outputs": {"missing": True}},
    {"item_ids": ["positive"], "expected_outputs": {"positive": float("nan")}},
    {"item_ids": ["positive"], "expected_outputs": {"positive": True},
     "expected_abstentions": ["positive"]},
], ids=["empty_ids", "missing_item", "expectation_without_ids", "expectation_outside_ids",
        "non_finite_expectation", "output_and_abstention"])
def test_invalid_controls_fail_before_any_detector_or_llm_call(smoke_runtime, monkeypatch, kwargs):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid controls must not execute detectors or initialize an LLM")

    monkeypatch.setattr(tools_eval, "run_detector", unexpected)
    monkeypatch.setattr(tools_eval, "_llm_adapter", unexpected)
    # LLM syntax ensures even backend readiness is deferred until IDs resolve.
    source = "def detect(item, llm_client=None): return llm_client.classify('rule', 'text')"
    output = tools_eval.smoke_test_feature(source, **kwargs)
    assert output["ok"] is False
    if "missing" in kwargs.get("item_ids", []):
        assert output["missing_item_ids"] == ["missing"]


@pytest.mark.parametrize(("literal", "expected", "matches"), [
    ("True", 1, False), ("1", True, False), ("1", 1.0, True), ("'abstain'", "abstain", True),
])
def test_scalar_expectations_are_not_coerced(smoke_runtime, literal, expected, matches):
    output = tools_eval.smoke_test_feature(
        f"def detect(item): return {literal}", item_ids=["positive"],
        expected_outputs={"positive": expected},
    )
    assert output["checks_passed"] is matches
    assert output["abstentions"] == 0


def test_controls_use_pinned_helpers_and_peer_dependencies(smoke_runtime):
    helper = HelperFunction(name="tail", source="def tail(text): return text[-30:]",
                            entrypoint="tail")
    smoke_runtime.library.helpers.upsert(helper)
    smoke_runtime.library.upsert(FeatureFunction(name="peer", description="Peer", source=SOURCE))
    output = tools_eval.smoke_test_feature(
        'def detect(item):\n'
        '    return library.fire("peer", item) and '
        'helpers.call("tail", item["response_text"]).endswith("DEFECT")',
        item_ids=["positive", "recovered"],
        expected_outputs={"positive": True, "recovered": False},
        depends_on=["peer"], uses_helpers={"tail": helper.revision},
    )
    assert output["ok"] and output["checks_passed"]
    assert not smoke_runtime.library.get("peer").scores


def test_named_controls_execute_frozen_bundle_without_live_dependencies(smoke_runtime, monkeypatch):
    from copy import deepcopy

    from autofa.core.dependencies import build_bundle
    from autofa.session.repair_workflow import _version

    helper = HelperFunction(name="tail", source="def tail(text): return text[-30:]",
                            entrypoint="tail")
    smoke_runtime.library.helpers.upsert(helper)
    smoke_runtime.library.upsert(FeatureFunction(name="peer", description="Peer", source=SOURCE))
    bundle = build_bundle(smoke_runtime.library, depends_on=["peer"],
                          uses_helpers={"tail": helper.revision})
    fn = FeatureFunction(
        name="restored", description="Original narrower claim", mechanism="Frozen scope",
        source='def detect(item):\n'
               '    return library.fire("peer", item) and '
               'helpers.call("tail", item["response_text"]).endswith("DEFECT")',
        meta={"dependency_bundle": bundle},
        scores={"bucket_mode": "truthy", "behavior_purity": .9},
    )
    smoke_runtime.library = FunctionLibrary(functions=[fn])
    before = deepcopy(fn.to_dict())
    output = tools_eval.smoke_test_feature(
        name=fn.name, expected_version_hash=_version(before),
        item_ids=["positive", "recovered", "incomplete", "broken"],
    )
    assert output["ok"] and output["claim_validation"] == "not_performed"
    assert output["version_hash"] == _version(before)
    assert output["histogram"] == {"True": 1, "False": 1}
    assert [row["matches_saved_bucket"] for row in output["results"]] == [
        True, False, None, None,
    ]
    assert output["abstentions"] == output["errors"] == 1
    assert fn.to_dict() == before
    assert "old verdict" in output["next_action"]
    # A restart/rollback can reuse completed controls without either live alias.
    monkeypatch.setattr(tools_eval, "run_detector",
                        lambda *a, **kw: pytest.fail("unchanged result should be cached"))
    cached = tools_eval.smoke_test_feature(name=fn.name, item_ids=["positive", "recovered"])
    assert cached["ok"] and cached["reused"] == 2


def test_named_controls_check_pinned_dependencies_and_invalidate_changes(smoke_runtime):
    from autofa.core.dependencies import build_bundle
    from autofa.session.repair_workflow import _version

    def pinned(source):
        peer = FeatureFunction(name="peer", description="Peer", source=source)
        bundle = build_bundle(FunctionLibrary(functions=[peer]), depends_on=["peer"])
        return {"deps": {"peer": source}, "dependency_bundle": bundle}

    fn = FeatureFunction(
        name="restored", description="Frozen dependency",
        source='def detect(item):\n'
               '    text = item["response_text"]\n'
               '    return library.fire("peer", item) and text.endswith("DEFECT")',
        depends_on=["peer"], meta=pinned(SOURCE),
    )
    smoke_runtime.library.upsert(fn)
    version = _version(fn.to_dict())
    output = tools_eval.smoke_test_feature(name=fn.name, item_ids=["positive"])
    assert output["ok"] and output["results"][0]["value"] is True
    fn.meta = pinned('def detect(item): return item["response_text"].endswith("OTHER")')
    stale = tools_eval.smoke_test_feature(
        name=fn.name, expected_version_hash=version, item_ids=["positive"],
    )
    assert not stale["ok"]
    changed = tools_eval.smoke_test_feature(name=fn.name, item_ids=["positive"])
    assert changed["ok"] and changed["reused"] == 0, changed
    assert changed["results"][0]["value"] is False


@pytest.mark.parametrize("kwargs", [
    {},
    {"name": "absent"},
    {"name": "saved", "source": SOURCE},
    {"name": "saved", "expected_version_hash": "changed"},
    {"source": SOURCE, "expected_version_hash": "ambiguous"},
], ids=["nothing", "unknown_name", "name_and_source", "stale_version", "version_without_name"])
def test_named_controls_reject_ambiguous_or_stale_requests(smoke_runtime, monkeypatch, kwargs):
    smoke_runtime.library.upsert(FeatureFunction(name="saved", description="test", source=SOURCE))
    monkeypatch.setattr(tools_eval, "_llm_adapter",
                        lambda: pytest.fail("invalid request cannot start work"))
    result = tools_eval.smoke_test_feature(item_ids=["positive"], **kwargs)
    assert not result["ok"]


def test_random_smoke_remains_available_without_expectations(smoke_runtime, monkeypatch):
    def sampled(dataset, **kwargs):
        assert kwargs["max_chars"] is None
        return {"items": dataset.items[:2]}

    monkeypatch.setattr("autofa.data.parquet_store.sample_items", sampled)
    output = tools_eval.smoke_test_feature(SOURCE)
    assert output["ok"] and output["histogram"] == {"True": 1, "False": 1}
    assert output["diagnostic_scope"] == "random_sample"
    assert output["checked"] == 0 and output["checks_passed"] is None


def test_smoke_control_schemas_are_advertised_in_both_harnesses(monkeypatch):
    import claude_agent_sdk
    import jsonschema

    tools = []
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server",
                        lambda **kwargs: tools.extend(kwargs["tools"]))
    server.build_mcp_server()
    claude = next(tool for tool in tools if tool.name == "smoke_test_feature")
    codex = next(spec for spec in server.codex_tool_specs() if spec["name"] == claude.name)
    assert claude.input_schema == codex["inputSchema"]
    assert codex["inputSchema"]["required"] == []
    assert {"item_ids", "expected_outputs", "expected_abstentions",
            "name", "expected_version_hash"} <= set(
        claude.input_schema["properties"]
    )
    args = {"source": SOURCE, "item_ids": ["positive"], "expected_outputs": {"positive": True}}
    jsonschema.validate(args, claude.input_schema)


def _hub(monkeypatch, calls, *, fail_at=None, failure=None, model="test",
         record=lambda kwargs: kwargs):
    from autofa.llm.hub import HubLLM

    def create(**kwargs):
        calls.append(record(kwargs))
        if len(calls) == fail_at:
            raise failure or ConnectionError("connection reset")
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content="yes"), finish_reason="stop"),
        ])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                             base_url="https://mock.invalid/v1", api_key="fake-key")
    hub = HubLLM(client, model=model, api="chat_completions")
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: hub)
    monkeypatch.setattr(tools_eval, "_llm_backend_error", lambda: None)
    return hub


LLM_SOURCE = '''def detect(item, llm_client=None):
    return llm_client.classify_options(
        item["response_text"], "Is the answer defective?", options=["yes", "no"]) == "yes"
'''


@pytest.mark.parametrize(("source", "n", "limit"), [(LLM_SOURCE, 31, 30), (SOURCE, 501, 500)])
def test_large_diagnostics_fail_before_fetch_or_model(smoke_runtime, monkeypatch, source, n, limit):
    monkeypatch.setattr(tools_eval, "_llm_adapter", lambda: pytest.fail("Must reject before model"))
    result = tools_eval.smoke_test_feature(source, n=n)
    assert not result["ok"] and result["diagnostic_limit"] == limit
    assert "capped" in result["error"]
    result = tools_eval.smoke_test_feature(source, item_ids=[str(i) for i in range(n)])
    assert not result["ok"] and result["diagnostic_limit"] == limit


@pytest.mark.parametrize("provider_cap", [False, True])
def test_interrupted_smoke_reuses_only_completed_items_after_resume(
    smoke_runtime, monkeypatch, provider_cap,
):
    from autofa.session.recovery import RunInterruptedError

    calls = []
    failure = None
    if provider_cap:
        failure = RuntimeError("Synthetic provider allowance exhausted")
        failure.status_code = 422
        failure.body = {"error": {"type": "budget_exceeded"}}
    _hub(monkeypatch, calls, fail_at=2, failure=failure)
    ids = ["positive", "recovered"]
    interruption = (pytest.raises(RuntimeError, match="Classifier provider budget exhausted")
                    if provider_cap else pytest.raises(RunInterruptedError))
    with interruption:
        tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=ids)
    assert len(calls) == 2
    smoke_runtime._smoke_results = {}
    _hub(monkeypatch, calls)
    result = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=ids)
    assert result["ok"] and result["reused"] == 1
    assert len(calls) == 3
    assert result["diagnostic_budget"]["model_calls"] == 3
    again = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=ids)
    assert again["reused"] == 2 and again["model_calls"] == 0 and len(calls) == 3


def test_negative_smoke_result_survives_runtime_restart_and_rechecks_controls(
    smoke_runtime, monkeypatch,
):
    first = tools_eval.smoke_test_feature(
        SOURCE, item_ids=["recovered"], expected_outputs={"recovered": False},
    )
    assert first["checks_passed"] and first["results"][0]["value"] is False
    # A new runtime shares only the persisted run directory, not in-memory caches.
    resumed = SimpleNamespace(dataset=ControlDataset(), library=FunctionLibrary(),
                              config=smoke_runtime.config, run=smoke_runtime.run)
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: resumed)
    monkeypatch.setattr(
        tools_eval, "run_detector", lambda *args, **kwargs: pytest.fail("Must reuse negative"),
    )
    second = tools_eval.smoke_test_feature(
        SOURCE, item_ids=["recovered"], expected_outputs={"recovered": True},
    )
    assert second["ok"] and second["reused"] == 1
    assert second["results"][0]["value"] is False
    assert second["checks_passed"] is False and second["mismatches"] == 1
    assert second["diagnostic_budget"]["rows"] == first["diagnostic_budget"]["rows"] == 1


def test_diagnostic_budget_is_shared_across_probes_and_runtime_restart(
    smoke_runtime, monkeypatch,
):
    calls = []
    _hub(monkeypatch, calls)
    monkeypatch.setattr(smoke_work, "CALLS_PER_PANEL", 2)
    smoke_runtime.config.discovery_plateau_patience = 1
    smoke_runtime.run.round = smoke_runtime.run.max_rounds
    first = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=["positive"])
    second = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=["recovered"])
    assert first["ok"] and second["ok"] and len(calls) == 2
    assert second["diagnostic_budget"]["model_calls"] == 2
    assert second["diagnostic_budget"]["exhausted"]
    resumed = SimpleNamespace(dataset=ControlDataset(), library=FunctionLibrary(),
                              config=smoke_runtime.config, run=smoke_runtime.run)
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: resumed)
    # Editing unrelated checkpoint metadata or restarting does not renew this round.
    stall_path = smoke_runtime.run.run_dir() / "discovery_stall.json"
    stall_path.write_text(json.dumps({"round": 7, "quality_generation": 0}))
    cached = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=["positive"])
    assert cached["ok"] and cached["reused"] == 1 and len(calls) == 2
    blocked = tools_eval.smoke_test_feature(
        LLM_SOURCE + "\n# different probe\n", item_ids=["positive"],
    )
    assert blocked["status"] == "diagnostic_budget_exhausted" and len(calls) == 2
    assert blocked["diagnostic_budget"]["model_calls"] == 2
    # A verified quality improvement can also renew diagnostics within a round.
    stall_path.write_text(json.dumps({"round": 7, "quality_generation": 1}))
    renewed = tools_eval.smoke_test_feature(
        LLM_SOURCE + "\n# different probe\n", item_ids=["positive"],
    )
    assert renewed["ok"] and renewed["diagnostic_budget"]["model_calls"] == 1
    assert len(calls) == 3
    smoke_runtime.run.round += 1
    next_round = tools_eval.smoke_test_feature(
        LLM_SOURCE + "\n# next recovery investigation\n", item_ids=["recovered"],
    )
    assert next_round["ok"] and next_round["diagnostic_budget"]["model_calls"] == 1
    assert len(calls) == 4


def test_cumulative_diagnostic_allowance_cannot_block_planned_rounds(smoke_runtime, monkeypatch):
    calls = []
    _hub(monkeypatch, calls)
    smoke_runtime.run.round = 2  # The last planned round.
    budget = smoke_work.diagnostic_budget(smoke_runtime)
    state = {"version": 1, "epoch": smoke_work._epoch(smoke_runtime.run.run_dir()),
             "rows": 0, "model_calls": budget["model_call_limit"]}
    (smoke_runtime.run.run_dir() / "smoke_work.json").write_text(json.dumps(state))

    # Both admission and model dispatch remain possible, including the last
    # planned round. Work is still counted instead of resetting the allowance.
    result = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=["positive"])
    assert result["ok"] and result["errors"] == 0 and len(calls) == 1
    used = result["diagnostic_budget"]
    assert used["rows"] == state["rows"] + 1
    assert used["model_calls"] == state["model_calls"] + 1
    assert used["allowance_exhausted"] and not used["enforced"] and not used["exhausted"]

    # Planned work must not pre-exhaust the first extra recovery round.
    smoke_runtime.run.round = smoke_runtime.run.max_rounds
    resumed = SimpleNamespace(dataset=ControlDataset(), library=FunctionLibrary(),
                              config=smoke_runtime.config, run=smoke_runtime.run)
    monkeypatch.setattr(tools_eval, "get_runtime", lambda: resumed)
    recovered = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=["recovered"])
    assert recovered["ok"] and len(calls) == 2
    assert recovered["diagnostic_budget"]["enforced"]
    assert not recovered["diagnostic_budget"]["exhausted"]
    assert recovered["diagnostic_budget"]["rows"] == 1
    cached = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=["positive"])
    assert cached["ok"] and cached["reused"] == 1 and len(calls) == 2


def test_smoke_cache_invalidates_changed_item_model_source_and_policy(smoke_runtime, monkeypatch):
    calls = []
    _hub(monkeypatch, calls)
    def run(source=LLM_SOURCE):
        return tools_eval.smoke_test_feature(source, item_ids=["positive"])
    assert run()["reused"] == 0
    assert run()["reused"] == 1
    smoke_runtime.dataset.items[0]["response_text"] = "Changed original evidence."
    assert run()["reused"] == 0
    _hub(monkeypatch, calls, model="new-model")
    assert run()["reused"] == 0
    assert run(LLM_SOURCE + "\n# revised implementation\n")["reused"] == 0
    smoke_runtime.config.detector_fields = "metadata"
    assert run()["reused"] == 0
    assert len(calls) == 5


def test_nested_detector_dispatches_share_budget_and_cannot_swallow_limit(
    smoke_runtime, monkeypatch,
):
    calls = []
    _hub(monkeypatch, calls)
    monkeypatch.setattr(smoke_work, "CALLS_PER_PANEL", 2)
    smoke_runtime.config.discovery_plateau_patience = 1
    smoke_runtime.run.round = smoke_runtime.run.max_rounds
    peer_source = '''def detect(item, llm_client=None):
    for _ in range(8):
        llm_client.classify_options(
            item["response_text"], "Is it defective?", options=["yes", "no"])
    return True
'''
    smoke_runtime.library.upsert(
        FeatureFunction(name="peer", description="peer", source=peer_source))
    source = '''def detect(item):
    for _ in range(8):
        try:
            library.fire("peer", item)
        except Exception:
            pass
    return True
'''
    result = tools_eval.smoke_test_feature(source, item_ids=["positive"], depends_on=["peer"])
    assert not result["ok"] and result["status"] == "diagnostic_budget_exhausted", result
    assert len(calls) == 2
    assert result["diagnostic_budget"]["exhausted"]
    assert not result["results"]
    again = tools_eval.smoke_test_feature(source, item_ids=["positive"], depends_on=["peer"])
    assert not again["ok"] and len(calls) == 2


def test_sdk_retry_is_charged_but_classifier_cache_hit_is_not(smoke_runtime, monkeypatch):
    calls = []
    hub = _hub(monkeypatch, calls)
    create = hub._client.chat.completions.create

    class Unsupported(Exception):
        status_code = 400
        body = {"error": {"code": "unsupported_parameter", "param": "temperature",
                          "message": "temperature not supported"}}

    def retry(**kwargs):
        if "temperature" in kwargs:
            calls.append(kwargs)
            raise Unsupported()
        return create(**kwargs)

    hub._client.chat.completions.create = retry
    source = '''def detect(item, llm_client=None):
    first = llm_client.classify("Is it defective?", item["response_text"])
    second = llm_client.classify("Is it defective?", item["response_text"])
    return first == second
'''
    result = tools_eval.smoke_test_feature(source, item_ids=["positive"])
    assert result["ok"] and result["errors"] == 0
    assert result["model_calls"] == len(calls) == 2


def test_raw_chat_dispatch_uses_the_same_budget(smoke_runtime, monkeypatch):
    calls = []
    _hub(monkeypatch, calls)
    monkeypatch.setattr(smoke_work, "CALLS_PER_PANEL", 1)
    source = '''def detect(item, llm_client=None):
    for _ in range(3):
        llm_client.chat.completions.create(
            model="custom", messages=[{"role": "user", "content": item["response_text"]}])
    return True
'''
    result = tools_eval.smoke_test_feature(source, item_ids=["positive"])
    assert not result["ok"] and len(calls) == 1
    # The per-panel limit still applies during a protected planned round.
    assert result["diagnostic_budget"]["enforced"] is False


def test_changed_dependency_invalidates_saved_control(smoke_runtime, monkeypatch):
    peer = FeatureFunction(name="peer", description="peer",
                           source="def detect(item): return 'DEFECT' in item['response_text']")
    smoke_runtime.library.upsert(peer)
    source = ('def detect(item): return library.fire("peer", item) '
              'and "already resolved" not in item["response_text"]')
    def run():
        return tools_eval.smoke_test_feature(source, item_ids=["positive"], depends_on=["peer"])
    assert run()["results"][0]["value"] is True
    assert run()["reused"] == 1
    peer.source = "def detect(item): return 'absent marker' in item['response_text']"
    result = run()
    assert result["reused"] == 0 and result["results"][0]["value"] is False


def test_final_reserved_row_can_complete_its_model_call(smoke_runtime, monkeypatch):
    calls = []
    _hub(monkeypatch, calls)
    smoke_runtime.run.round = smoke_runtime.run.max_rounds
    budget = smoke_work.diagnostic_budget(smoke_runtime)
    (smoke_runtime.run.run_dir() / "smoke_work.json").write_text(json.dumps({
        "version": 1, "epoch": smoke_work._epoch(smoke_runtime.run.run_dir(),
                                                 round_idx=smoke_runtime.run.round),
        "rows": budget["row_limit"] - 1, "model_calls": 0,
    }))
    result = tools_eval.smoke_test_feature(LLM_SOURCE, item_ids=["positive"])
    assert result["ok"] and result["errors"] == 0 and len(calls) == 1
    assert result["diagnostic_budget"]["exhausted"]


@pytest.mark.parametrize("initial_status", ["available", "unavailable"])
def test_external_artifact_change_invalidates_value_and_abstention(
    smoke_runtime, monkeypatch, initial_status,
):
    from autofa.data import artifact_access

    artifact = {"status": initial_status, "text": "assertion failed"}

    def read(item, artifact_idx, **kwargs):
        return artifact_access._record(dict(artifact), artifact_idx)

    monkeypatch.setattr(artifact_access, "artifact_evidence", read)
    source = '''from autofa.trace import artifact
def detect(item):
    evidence = artifact(item, 0)
    if evidence["status"] != "available":
        library.abstain("Artifact is unavailable.")
    return evidence["text"] == "assertion failed"
'''
    first = tools_eval.smoke_test_feature(source, item_ids=["positive"])
    assert first["errors"] == 0
    assert first["results"][0]["status"] == (
        "value" if initial_status == "available" else "abstained")
    same = tools_eval.smoke_test_feature(source, item_ids=["positive"])
    assert same["reused"] == 1
    # The item and Parquet identity stay unchanged; only external evidence changes.
    artifact.update(status="available", text="assertion passed")
    changed = tools_eval.smoke_test_feature(source, item_ids=["positive"])
    assert changed["reused"] == 0 and changed["results"][0]["value"] is False


# --- diagnostic receipts cite actual control execution, never a grade ---


def execute():
    return tools_eval.smoke_test_feature(
        SOURCE, item_ids=["positive", "recovered"],
        expected_outputs={"positive": True, "recovered": False},
    )


def test_native_control_receipt_is_immutable_reusable_and_not_a_grade(smoke_runtime):
    before = smoke_runtime.library.to_dict()
    first, again = execute(), execute()
    assert first["diagnostic_receipt_id"] == again["diagnostic_receipt_id"]
    assert again["reused"] == 2
    receipt = read_diagnostic_receipt(smoke_runtime, first["diagnostic_receipt_id"])
    assert first["source_sha256"] == again["source_sha256"] == receipt["source_sha256"]
    assert receipt["complete"] and receipt["checks_passed"]
    assert receipt["expected_outputs"] == {"positive": True, "recovered": False}
    assert receipt["scope"] == "author_controls_not_independent_validation"
    assert receipt["results"][0]["value"] is True
    assert smoke_runtime.library.to_dict() == before
    # Changed control expectations get their own receipt without rewriting the old one.
    changed = tools_eval.smoke_test_feature(
        SOURCE, item_ids=["positive", "recovered"],
        expected_outputs={"positive": False, "recovered": False},
    )
    assert changed["diagnostic_receipt_id"] != first["diagnostic_receipt_id"]
    assert not changed["checks_passed"]
    assert read_diagnostic_receipt(smoke_runtime, first["diagnostic_receipt_id"])["checks_passed"]


@pytest.mark.parametrize("change", ["content", "missing", "foreign_run", "tampered"])
def test_receipt_rejects_stale_or_unrelated_evidence(smoke_runtime, change):
    receipt_id = execute()["diagnostic_receipt_id"]
    if change == "content":
        smoke_runtime.dataset.items[0]["response_text"] = "changed original evidence"
    elif change == "missing":
        smoke_runtime.dataset.items.pop(0)
    elif change == "foreign_run":
        smoke_runtime.run.run_id = "another-run"
    else:
        path = smoke_runtime.run.run_dir() / "diagnostic_receipts" / f"{receipt_id}.json"
        record = json.loads(path.read_text())
        record["checks_passed"] = False
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        read_diagnostic_receipt(smoke_runtime, receipt_id)


def test_artifact_reads_must_still_match_when_citing_execution(smoke_runtime, monkeypatch):
    output = execute()
    receipt_id = save_diagnostic_receipt(
        smoke_runtime, source=SOURCE, execution_identity="pinned-test-execution",
        items=smoke_runtime.dataset.items[:2],
        expected_outputs={"positive": True, "recovered": False}, expected_abstentions=[],
        result=output, artifact_reads={"positive": [{"path": "original-image"}]},
    )
    monkeypatch.setattr("autofa.data.artifact_access.validate_artifact_reads", lambda *a: False)
    with pytest.raises(ValueError, match="artifact evidence changed"):
        read_diagnostic_receipt(smoke_runtime, receipt_id)


def test_named_diagnostic_cannot_be_cited_after_detector_changes(smoke_runtime):
    detector = FeatureFunction(name="diagnosed", description="Synthetic diagnostic", source=SOURCE)
    smoke_runtime.library.upsert(detector)
    output = tools_eval.smoke_test_feature(
        name="diagnosed", item_ids=["positive", "recovered"],
        expected_outputs={"positive": True, "recovered": False},
    )
    receipt_id = output["diagnostic_receipt_id"]
    assert read_diagnostic_receipt(smoke_runtime, receipt_id)["complete"]
    detector.source = "def detect(item): return False"
    with pytest.raises(ValueError, match="detector version changed"):
        read_diagnostic_receipt(smoke_runtime, receipt_id)


@pytest.mark.parametrize("receipt_id,match", [
    ("../outside", "diagnostic_receipt_id"), (None, "diagnostic_receipt_id"),
    ("0" * 64, "unavailable in this run"),  # well-formed, but never saved in this run
])
def test_only_native_receipt_ids_are_resolved(smoke_runtime, receipt_id, match):
    with pytest.raises(ValueError, match=match):
        read_diagnostic_receipt(smoke_runtime, receipt_id)


# --- concurrency through MCP's shared cancellation scope ---

class SmokeDataset(ControlDataset):
    def __init__(self):
        super().__init__()
        self.items = [
            {"_item_id": name, "response_text": name, "is_correct": False}
            for name in ("first", "second", "third")
        ]


async def _invoke(item_ids):
    response = await server.invoke_tool(
        "smoke_test_feature", {"source": LLM_SOURCE, "item_ids": item_ids},
    )
    return json.loads(response["content"][0]["text"])


def _cancellation_scope(kwargs):
    assert _CURRENT.get() is not None, "MCP cancellation must reach every SDK call"
    return _CURRENT.get()


def _setup(directory, monkeypatch):
    rt = SimpleNamespace(
        dataset=SmokeDataset(), library=FunctionLibrary(),
        config=SimpleNamespace(llm_eval_workers=2, discovery_plateau_patience=1),
        run=SimpleNamespace(run_id="parallel-smoke", round=3, max_rounds=3,
                            run_dir=lambda: directory),
    )
    calls = []
    hub = _hub(monkeypatch, calls, record=_cancellation_scope)
    monkeypatch.setattr(runtime, "_RUNTIME", rt)
    return rt, hub, calls


def test_smoke_respects_configured_worker_ceiling(tmp_path, monkeypatch):
    rt, hub, calls = _setup(tmp_path, monkeypatch)
    rt.config.llm_eval_workers = 4
    rt.config.llm_eval_max_workers = 2
    original_create = hub._client.chat.completions.create
    first_pair = threading.Barrier(2)
    third_entered = threading.Event()
    lock = threading.Lock()
    active = peak = started = 0

    def concurrent_create(**kwargs):
        nonlocal active, peak, started
        with lock:
            index = started
            started += 1
            active += 1
            peak = max(peak, active)
        try:
            if index < 2:
                first_pair.wait(timeout=2)
                # A third configured worker would enter while both slots wait.
                third_entered.wait(.05)
            else:
                third_entered.set()
            return original_create(**kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(hub._client.chat.completions, "create", concurrent_create)
    result = asyncio.run(_invoke(["first", "second", "third"]))
    assert result["ok"] and result["errors"] == 0
    assert len(calls) == 3 and all(call is calls[0] for call in calls)
    assert peak == 2


def _checkpoint_interleaving(directory, monkeypatch):
    """Force completion against a second worker's row reservation.

    Previously the writer owned cancellation.lock and then requested _LOCK,
    while the second worker held _LOCK and requested cancellation.lock. With
    the fix, the second worker waits before acquiring either reservation lock.
    """
    rt, _, calls = _setup(directory, monkeypatch)
    monkeypatch.setattr(smoke_work, "CALLS_PER_PANEL", 2)
    saving, contender = threading.Event(), threading.Event()
    worker = threading.local()
    original_lock = smoke_work._LOCK

    class ObservedLock:
        def __enter__(self):
            acquired = original_lock.acquire(blocking=False)
            if getattr(worker, "second", False):
                worker.attempts += 1
                if not acquired or worker.attempts == 2:
                    contender.set()
            if not acquired:
                original_lock.acquire()
            return self

        def __exit__(self, *exc):
            original_lock.release()

    original_item = smoke_work.SmokeWork.item

    @contextmanager
    def scheduled_item(self, item):
        if item["_item_id"] == "second":
            assert saving.wait(3), "First worker never reached its checkpoint"
            worker.second, worker.attempts = True, 0
        with original_item(self, item) as result:
            yield result

    original_write = smoke_work.atomic_write_json

    def checkpoint(path, value):
        original_write(path, value)
        if value.get("result", {}).get("item_id") == "first":
            saving.set()
            assert contender.wait(3), "Second worker never attempted its reservation"

    monkeypatch.setattr(smoke_work, "_LOCK", ObservedLock())
    monkeypatch.setattr(smoke_work.SmokeWork, "item", scheduled_item)
    monkeypatch.setattr(smoke_work, "atomic_write_json", checkpoint)
    first = asyncio.run(_invoke(["first", "second"]))
    assert first["ok"] and first["errors"] == 0 and first["model_calls"] == 2
    assert len(calls) == 2 and calls[0] is calls[1]
    assert first["diagnostic_budget"]["rows"] == 2
    assert first["diagnostic_budget"]["model_calls"] == 2
    index = next((directory / "smoke_results").glob("index-*.json"))
    assert set(json.loads(index.read_text())) == {"first", "second"}
    assert len(rt.library.functions) == 0

    # The blocked historical run saved a result before it could update its
    # index. Recover that state without paying to run completed controls again.
    index.unlink()
    rt._smoke_results = {}
    # A new MCP token can reuse both persisted controls even at the work limit.
    resumed = asyncio.run(_invoke(["first", "second"]))
    assert resumed["ok"] and resumed["reused"] == 2 and resumed["model_calls"] == 0
    assert len(calls) == 2
    assert resumed["diagnostic_budget"]["rows"] == 2
    assert set(json.loads(index.read_text())) == {"first", "second"}
    exhausted = asyncio.run(_invoke(["third"]))
    assert exhausted["status"] == "diagnostic_budget_exhausted"
    assert exhausted["diagnostic_budget"]["model_calls"] == 2 and len(calls) == 2


def _cancel_checkpoint(directory, monkeypatch, *, stage):
    _, _, calls = _setup(directory, monkeypatch)
    writing, cancel_requested, release = (
        threading.Event(), threading.Event(), threading.Event(),
    )

    class ObservedCancellation(Cancellation):
        def cancel(self):
            cancel_requested.set()
            super().cancel()

    original_write = smoke_work.atomic_write_json

    def checkpoint(path, value):
        original_write(path, value)
        matches = (path.name.startswith("index-") if stage == "index"
                   else value.get("result", {}).get("item_id") == "first")
        if matches and not writing.is_set():
            writing.set()
            assert release.wait(3), "Cancellation blocked the event loop"

    monkeypatch.setattr(server, "Cancellation", ObservedCancellation)
    monkeypatch.setattr(smoke_work, "atomic_write_json", checkpoint)

    async def scenario():
        task = asyncio.create_task(_invoke(["first"]))
        try:
            assert await asyncio.to_thread(writing.wait, 3)
            task.cancel()
            assert await asyncio.to_thread(cancel_requested.wait, 3)
            assert not task.done(), "An active checkpoint must drain before cancellation"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    asyncio.run(scenario())
    index = next((directory / "smoke_results").glob("index-*.json"))
    assert set(json.loads(index.read_text())) == {"first"}
    resumed = asyncio.run(_invoke(["first"]))
    assert resumed["ok"] and resumed["reused"] == 1 and resumed["model_calls"] == 0
    assert resumed["diagnostic_budget"]["rows"] == 1
    assert resumed["diagnostic_budget"]["model_calls"] == len(calls) == 1


def _cancel_before_checkpoint(directory, monkeypatch):
    _, _, calls = _setup(directory, monkeypatch)
    original_detector = tools_eval.run_detector

    def cancelled_detector(*args, **kwargs):
        result = original_detector(*args, **kwargs)
        _CURRENT.get().cancel()
        return result

    monkeypatch.setattr(tools_eval, "run_detector", cancelled_detector)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_invoke(["first"]))
    assert not list((directory / "smoke_results").glob("*.json"))
    assert len(calls) == 1
    monkeypatch.setattr(tools_eval, "run_detector", original_detector)
    resumed = asyncio.run(_invoke(["first"]))
    assert resumed["ok"] and resumed["reused"] == 0 and resumed["model_calls"] == 1
    assert resumed["diagnostic_budget"]["rows"] == 2
    assert resumed["diagnostic_budget"]["model_calls"] == len(calls) == 2


@pytest.mark.parametrize("scenario", [
    "checkpoint_interleaving", "cancel_result", "cancel_index", "cancel_before_checkpoint",
])
def test_smoke_mcp_concurrency_and_checkpoint_recovery(tmp_path, scenario):
    # A regressed executor deadlock cannot leave pytest or its shutdown blocked.
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), scenario, str(tmp_path)],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Smoke scenario {scenario} deadlocked: {exc}")
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    scenario, directory = sys.argv[1], Path(sys.argv[2])
    with pytest.MonkeyPatch.context() as patches:
        if scenario in {"cancel_result", "cancel_index"}:
            _cancel_checkpoint(directory, patches, stage=scenario.removeprefix("cancel_"))
        else:
            globals()["_" + scenario](directory, patches)
