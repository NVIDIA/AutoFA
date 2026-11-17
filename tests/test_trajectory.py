# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent trajectories and the trace API: parsing, generated reference, records and reads."""

import copy
import gc
import hashlib
import inspect
import json
import weakref
from dataclasses import fields
from types import SimpleNamespace
from typing import Annotated, Any, NotRequired, Required, TypedDict, get_type_hints

import pytest

from autofa import trace, trace_reference
from autofa.config import Config, RunConfig
from autofa.core.detector_lint import lint_detector_source
from autofa.core.library import FunctionLibrary
from autofa.core.sandbox import NullLLM, compile_detector, run_detector
from autofa.data.corpus_health import measure_parse_coverage
from autofa.data.corpus_stats import contrastive_corpus_stats
from autofa.data.trace_fields import derive_trace_fields
from autofa.data.trajectory import (
    looks_like_agent_json,
    parse_trajectory,
)
from autofa.diagnostics import Diagnostics, get_diagnostics
from autofa.llm import hub as hub_mod
from autofa.mcp import runtime as rtmod
from autofa.mcp import tools_data
from autofa.mcp.runtime import Runtime
from autofa.mcp.server import MCP_SOFT_WARN_CHARS, TOOL_CALLABLES, tool_response
from autofa.mcp.tools_data import get_trace_api
from autofa.reports.failure_cards import _compact_item
from autofa.session.diversity import infer_strategy, infer_surfaces
from autofa.session.prompts import build_system_prompt, write_session_workspace
from autofa.trace_reference import render_trace_api_reference, trace_api_reference

from support.traces import execution_item

# --- trace API reference ---


def test_trace_contracts_are_injected_and_saved_for_the_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    reference = render_trace_api_reference()
    assert build_system_prompt(exploration_ssot=False).count(reference) == 1
    workspace = tmp_path / "run"
    write_session_workspace(workspace, exploration_ssot=False,
                            objective_config=Config(harness="claude"))
    assert (workspace / "trace_api.md").read_text() == reference
    assert reference in (workspace / "autofa_protocol.md").read_text()
    assert "trace_api.md" in (workspace / "CLAUDE.md").read_text()
    assert "get_trace_api" in (
        workspace / ".claude" / "skills" / "discover" / "SKILL.md"
    ).read_text()


def _read_reference_pages(name=None, **kwargs):
    expected = render_trace_api_reference(name)
    fingerprint = hashlib.sha256(expected.encode()).hexdigest()
    pages, offset = [], 0
    while True:
        page = get_trace_api(name=name, offset=offset, **kwargs)
        assert page["ok"] is True
        assert page["offset"] == offset
        assert page["offset_unit"] == "characters"
        assert page["total_chars"] == len(expected)
        assert page["reference_sha256"] == fingerprint
        assert page["selected_symbol"] == name
        assert "steps" in page["symbol_names"]
        assert len(json.dumps(page)) < MCP_SOFT_WARN_CHARS
        encoded = tool_response(page, "get_trace_api")["content"][0]["text"]
        assert json.loads(encoded) == page  # No overflow envelope or hidden remainder.
        pages.append(page)
        end = offset + len(page["reference"])
        if page["next_offset"] is None:
            assert end == len(expected)
            break
        assert page["next_offset"] == end > offset
        offset = page["next_offset"]
    assert "".join(page["reference"] for page in pages) == expected
    return pages


def test_reference_tool_needs_no_dataset_and_matches_agent_contracts(monkeypatch):
    def no_runtime():
        raise AssertionError("API documentation must not access an evaluation dataset")

    monkeypatch.setattr("autofa.mcp.tools_data.get_runtime", no_runtime)
    assert TOOL_CALLABLES["get_trace_api"] is get_trace_api
    full_pages = _read_reference_pages()
    assert len(full_pages) > 1
    assert not any(page["complete"] for page in full_pages)
    focused = "".join(page["reference"] for page in _read_reference_pages("steps"))
    assert "source_event" in focused
    assert "paired_step_index" in focused


def test_reference_tool_single_page_end_and_out_of_range_offset():
    pages = _read_reference_pages("input_text")
    assert len(pages) == 1
    assert pages[0]["complete"] is True
    end = get_trace_api("input_text", offset=pages[0]["total_chars"])
    assert end["reference"] == ""
    assert end["next_offset"] is None
    assert end["complete"] is False
    result = get_trace_api("input_text", offset=10**9)
    assert result["ok"] is False
    assert "offset exceeds" in result["error"]
    assert result["total_chars"] == pages[0]["total_chars"]


@pytest.mark.parametrize("offset,limit", [(-1, 100), (0, 0)])
def test_reference_tool_rejects_invalid_page_arguments(offset, limit):
    result = get_trace_api(offset=offset, limit=limit)
    assert result["ok"] is False
    assert "offset" in result["error"] and "limit" in result["error"]


def test_reference_tool_bounds_serialized_unicode_without_losing_contracts(monkeypatch):
    monkeypatch.setattr(trace.input_text, "__doc__", "音声📖" * 12_000)
    pages = _read_reference_pages("input_text", limit=10**9)
    assert len(pages) > 1
    assert all(len(page["reference"]) <= 12_000 for page in pages)


def test_unknown_or_private_trace_symbol_is_explicit_error():
    result = get_trace_api(name="no_such_helper")
    assert result["ok"] is False
    assert "no_such_helper" in result["error"]
    assert "steps" in result["error"]
    with pytest.raises(ValueError, match="Unknown autofa.trace symbol"):
        trace_api_reference("_blob")


# --- generated reference stays synchronized with the executable API ---


def test_full_reference_covers_every_public_symbol_and_step_attribute():
    result = trace_api_reference()
    text = result["reference"]
    assert result["module"] == "autofa.trace"
    assert result["selected_symbol"] is None
    assert result["symbol_names"] == trace.__all__
    for symbol in trace.__all__:
        assert f"`{symbol}`" in text
    for field in fields(trace.Step):
        assert f"`{field.name}:" in text
    gc.collect()
    assert render_trace_api_reference() == text  # Repeated renders reach the same schemas.


def test_symbol_lookup_includes_nested_schemas_without_unrelated_functions():
    result = trace_api_reference("steps")
    assert result["selected_symbol"] == "steps"
    assert "## `steps`" in result["reference"]
    assert "### `Step`" in result["reference"]
    assert "## `evidence_text`" not in result["reference"]
    assert "-> list[Step]" in result["reference"]
    assert "## `steps`" not in render_trace_api_reference("Step")
    assert "### `Step`" in render_trace_api_reference("Step")


def test_new_export_signature_and_reachable_schema_are_generated(monkeypatch):
    class Partial(TypedDict, total=False):
        omitted: str
        present: Required[int | None]

    class FutureResult(Partial):
        nullable: Annotated[str | None, "Nullable but always present."]
        missing: NotRequired[Annotated[int, "May be omitted entirely."]]
        explicit: Annotated[Required[bool], "Always available."]

    def future_helper(item, *, limit=37):
        """A new helper's behavior is visible without editing the renderer."""
        return {"nullable": None, "explicit": True}

    future_helper.__annotations__ = {
        "item": dict[str, str], "limit": int, "return": FutureResult,
    }
    monkeypatch.setattr(trace, "future_helper", future_helper, raising=False)
    monkeypatch.setattr(trace, "__all__", [*trace.__all__, "future_helper"])
    reference = render_trace_api_reference("future_helper")
    assert "future_helper(item: dict[str, str], *, limit: int = 37) -> FutureResult" in reference
    assert "A new helper's behavior is visible" in reference
    assert "`nullable: str | None` (required key); Nullable but always present." in reference
    assert "`missing: int` (optional key); May be omitted entirely." in reference
    assert "`explicit: bool` (required key); Always available." in reference
    # Inherited keys keep the totality of the schema that declared them.
    assert "`omitted: str` (optional key)" in reference
    assert "`present: int | None` (required key)" in reference

    # Runtime annotation changes must reach the reference; cached schema copies would drift.
    future_helper.__annotations__["limit"] = float
    future_helper.__kwdefaults__["limit"] = 3.5
    assert "limit: float = 3.5" in render_trace_api_reference("future_helper")


def test_every_public_function_exposes_a_return_annotation():
    for name in trace.__all__:
        value = getattr(trace, name)
        if inspect.isfunction(value):
            hints = get_type_hints(value, include_extras=True)
            assert "return" in hints, name
            assert hints["return"] not in (
                Any, dict, list, dict[str, Any], list[dict[str, Any]],
            ), name


def test_dependency_walk_retains_temporary_annotations(monkeypatch):
    class Earlier(TypedDict):
        first: int

    class Later(TypedDict):
        second: str

    def first():
        pass

    def second():
        pass

    temporary = None
    original = trace_reference.get_type_hints

    def hints(value, **kwargs):
        nonlocal temporary
        if value is first:
            annotation = list[Earlier]
            temporary = weakref.ref(annotation)
            return {"return": annotation}
        if value is second:
            gc.collect()
            # Keeping only IDs lets this annotation die and its ID be reused by
            # list[Later], incorrectly skipping Later's schema in the same walk.
            assert temporary() is not None
            return {"return": list[Later]}
        return original(value, **kwargs)

    monkeypatch.setattr(trace_reference, "get_type_hints", hints)
    schemas = trace_reference._Reference().dependencies([first, second])
    assert schemas == {"Earlier": Earlier, "Later": Later}
    gc.collect()
    assert temporary() is None  # Retention is scoped to this walk, not a global cache.


# --- execution-record reads ---


@pytest.fixture
def item(monkeypatch):
    value = execution_item()
    monkeypatch.setattr(tools_data, "_load_item_raw", lambda key: value if key == "a" else None)
    return value


def test_registered_reader_starts_with_honest_resource_coverage(item, monkeypatch):
    def unexpected_copy(*args):
        raise AssertionError("Coverage should not copy a potentially multi-megabyte record")

    monkeypatch.setattr("autofa.trace.record", unexpected_copy)
    assert TOOL_CALLABLES["get_trace_record"] is tools_data.get_trace_record
    result = tools_data.get_trace_record("a")
    assert result["ok"]
    assert result["value"]["artifacts"]["status"] == "partial"
    assert result["value"]["events"]["status"] == "available"
    assert "unavailable resource" in result["hint"]
    assert not tools_data.get_trace_record("missing")["ok"]


def test_explicit_fetch_uses_complete_imported_metadata_without_a_source_adapter(item, monkeypatch):
    def no_source():
        raise AssertionError("Imported metadata must not need a configured source API")

    monkeypatch.setattr(tools_data, "get_runtime", no_source)
    result = tools_data.get_trace_record("a", section="provenance", fetch_missing=True)
    assert result["ok"] and result["value"][0]["reader"] == "example"
    assert result["availability"]["status"] == "available"


def test_exact_source_event_selection_includes_lane_and_full_text_paging(item):
    original = copy.deepcopy(item)
    kwargs = dict(section="events", lane="", event_idx=41, pointer="/parts/0/text", max_chars=1000)
    offset, chunks = 0, []
    while True:
        result = tools_data.get_trace_record("a", offset=offset, **kwargs)
        assert result["ok"]
        chunks.append(result["value"])
        assert len(result["value"]) <= 1000
        if result["is_final"]:
            assert result["next_offset"] is None
            break
        assert result["next_offset"] > offset
        offset = result["next_offset"]
    assert "".join(chunks) == item["response_text"]["execution"]["events"][0]["parts"][0]["text"]
    assert item == original


def test_shared_event_indices_require_lane_and_never_use_parsed_step_numbers(item):
    item["response_text"]["execution"]["events"][1]["event_idx"] = 41
    ambiguous = tools_data.get_trace_record("a", section="events", event_idx=41)
    assert not ambiguous["ok"] and ambiguous["matches"] == 2
    exact = tools_data.get_trace_record("a", section="events", lane="worker/2", event_idx=41)
    assert exact["ok"] and exact["value"]["status"] == "retrying"
    assert not tools_data.get_trace_record("a", section="events", event_idx=1)["ok"]


def test_structured_overflow_is_navigable_and_does_not_cut_json(item):
    result = tools_data.get_trace_record("a", section="events", limit=1, max_chars=256)
    assert not result["ok"]
    assert result["child_pointers"] == ["/0"]
    metadata = tools_data.get_trace_record(
        "a", section="events", pointer="/0/provider_extension", max_chars=256,
    )
    assert metadata["ok"] and metadata["value"] == {"cache": ["cold"]}
    nested = tools_data.read_parsed_field(
        "a", field="execution", pointer="/tool_calls/0/args_full",
    )
    assert nested["ok"] and json.loads(nested["value"]) == {"command": "pytest"}


def test_future_metadata_keys_are_accessible_with_rfc6901_escaping(item):
    item["response_text"]["execution"]["custom/resource"] = {"~key": [3, 4]}
    result = tools_data.get_trace_record("a", section="record", pointer="/custom~1resource/~0key/1")
    assert result["ok"] and result["value"] == 4


@pytest.mark.parametrize("kwargs", [
    {"section": "unknown"}, {"pointer": "invalid"},
    {"section": "events", "pointer": "/-1"},  # not Python's last-element indexing
    {"section": "events", "pointer": "/0/missing"},
    {"section": "rollout", "event_idx": 41},
])
def test_invalid_navigation_returns_actionable_errors(item, kwargs):
    result = tools_data.get_trace_record("a", **kwargs)
    assert not result["ok"] and result["error"]


# --- trajectory parsing and MCP trajectory tools ---


def _atif_blob(steps):
    return json.dumps([{
        "schema_version": "ATIF-v1.5",
        "agent": {"name": "OpenHands-SDK", "tool_definitions": [
            {"function": {"name": "terminal"}}, {"function": {"name": "file_editor"}},
        ]},
        "steps": steps,
        "final_metrics": {"total_prompt_tokens": 100, "total_completion_tokens": 50,
                          "total_cost_usd": 0.01, "workspace_diff_preview": "--- a/x\n+++ b/x\n"},
    }])


_AGENT_STEPS = [
    {"source": "system", "message": "You are an agent."},
    {"source": "user", "message": "Fix the bug."},
    {"source": "agent", "reasoning_content": "I will inspect the repo first.",
     "tool_calls": [{"function_name": "terminal",
                     "arguments": {"command": "ls", "is_input": False}}],
     "observation": "file.py\nREADME.md\n"},
    {"source": "agent", "reasoning_content": "Now patch the file.", "message": "Applying fix.",
     "tool_calls": [{"function": {
         "name": "file_editor", "arguments": '{"path": "file.py", "command": "str_replace"}',
     }}]},
]

# A failed test run followed by an unsupported claim that the issue is fixed.
_FIX_STEPS = [
    {"source": "user", "message": "Fix it."},
    {"source": "agent", "reasoning_content": "I will run the tests.",
     "tool_calls": [{"function_name": "terminal", "arguments": {"command": "pytest"}}],
     "observation": "AssertionError: expected 1 got 2\nFAILED"},
    {"source": "agent", "reasoning_content": "All tests passed, the issue is now fixed.",
     "message": "Done.", "observation": ""},
]


def test_parse_steps_envelope():
    result = parse_trajectory(json.dumps({"agent": {"name": "generic"}, "steps": _AGENT_STEPS}))
    assert result.ok and result.dialect == "steps_envelope"
    assert (result.n_steps, result.n_agent_steps, result.n_tool_calls) == (4, 2, 2)
    assert {t.name for t in result.tools} == {"terminal", "file_editor"}


def test_parse_atif_openhands():
    result = parse_trajectory(_atif_blob(_AGENT_STEPS))
    assert result.ok and result.dialect == "atif_openhands"
    assert (result.agent_name, result.schema_version) == ("OpenHands-SDK", "ATIF-v1.5")
    assert "terminal" in result.tool_definitions
    assert "--- a/x" in result.workspace_diff_preview


def test_only_unparsed_structured_json_warns():
    blob = json.dumps({"tool_definitions": [{"name": "bash"}], "tool_calls": [{"name": "bash"}],
                       "events": [{"type": "run"}]})
    assert looks_like_agent_json(blob)
    result = parse_trajectory(blob)
    assert not result.ok and result.warning
    plain = parse_trajectory("The answer is 42 because of algebra.")
    assert not plain.ok and plain.warning is None
    assert not looks_like_agent_json("The answer is 42.")


def test_failure_cards_still_agent_transcript():
    compact = _compact_item({"_item_id": "m|b|x|0", "prompt_text": "Fix it",
                             "response_text": _atif_blob(_AGENT_STEPS), "is_correct": False,
                             "benchmark": "swe"})
    assert compact["response_kind"] == "agent_transcript"
    assert compact["trajectory_dialect"] == "atif_openhands"
    assert compact["n_agent_steps"] == 2
    assert "terminal" in compact["response_head"]


class _Dataset:
    """Minimal in-memory dataset behind the MCP trace-reading tools."""

    parquet_paths = []

    def __init__(self, items):
        self._items = items

    def load_items(self, *, limit=None):
        return self._items[:limit] if limit else self._items

    def iter_batches(self, *, batch_size, max_items):
        yield self._items[:max_items]

    def info(self, scan_limit=50_000):
        return {"n_items": len(self._items), "n_fail": 1, "n_pass": 0, "fail_rate": 1.0}

    def estimate_matching_rows(self):
        return len(self._items)


def _install_runtime(monkeypatch, items):
    rt = Runtime(dataset=_Dataset(items), library=FunctionLibrary(), config=Config(),
                 run=RunConfig(model_id="m", run_id="t", parquet_paths=[], benchmarks=["b"]))
    monkeypatch.setattr(rtmod, "_RUNTIME", rt)
    return rt


@pytest.fixture
def trajectory_runtime(monkeypatch):
    row = {"model_id": "m", "benchmark": "b", "run_idx": 0, "is_correct": False}
    return _install_runtime(monkeypatch, [
        {**row, "_item_id": "traj-1", "problem_id": "1", "prompt_text": "fix it",
         "response_text": _atif_blob(_AGENT_STEPS)},
        {**row, "_item_id": "plain-1", "problem_id": "2", "prompt_text": "q",
         "response_text": "just text, no tools"},
    ])


def test_mcp_get_trajectory_smoke(trajectory_runtime):
    assert {"get_trajectory", "list_trajectory_tools", "get_trajectory_step"} <= set(
        TOOL_CALLABLES)
    out = tools_data.get_trajectory("traj-1", limit=10)
    assert out["ok"] and out["ok_parse"]
    assert out["parsed"]["dialect"] == "atif_openhands"
    assert out["steps_total"] == len(out["steps"]) == 4

    tools = tools_data.list_trajectory_tools("traj-1")
    assert tools["ok"] and tools["ok_parse"]
    assert {t["name"] for t in tools["tools"]} == {"terminal", "file_editor"}

    step = tools_data.get_trajectory_step("traj-1", 3)
    assert step["ok"]
    assert step["step"]["source"] == "agent"
    assert "inspect the repo" in step["step"]["reasoning"]
    assert step["step"]["tool_calls"][0]["name"] == "terminal"

    info = tools_data.dataset_info()
    assert info["ok"]
    coverage = info["trajectory_coverage"]
    assert (coverage["n_sampled"], coverage["parsed"], coverage["plain_text"],
            coverage["dialects"]) == (2, 1, 1, {"atif_openhands": 1})
    assert "get_trajectory" in info["hint"]


def test_dataset_info_exposes_failed_startup_probe(trajectory_runtime, monkeypatch):
    rt = trajectory_runtime
    rt.llm_readiness = {
        "status": "unavailable", "configured": True, "available": False,
        "message": "LLM detectors unavailable: authentication failed.",
    }
    monkeypatch.setattr(rt, "ensure_hub_llm", lambda: pytest.fail("Do not hide the failed probe"))
    info = tools_data.dataset_info()
    assert info["capabilities"]["llm_backend"] is False
    assert info["llm_detectors"]["status"] == "unavailable"
    assert "authentication failed" in info["llm_detector_warning"]


def test_dataset_info_missing_backend_names_the_configured_key(trajectory_runtime, monkeypatch):
    def missing():
        raise RuntimeError("missing key")

    rt = trajectory_runtime
    rt.config.feature_base_url = "https://inference.example/v1"
    rt.config.feature_api_key_env = "INFERENCE_HUB_API_KEY"
    monkeypatch.setattr(rt, "ensure_hub_llm", missing)
    info = tools_data.dataset_info()
    assert info["capabilities"]["llm_backend"] is False
    assert info["llm_detectors"]["status"] == "not_configured"
    assert "INFERENCE_HUB_API_KEY" in info["llm_detector_warning"]
    assert "OPENAI_API_KEY" not in info["llm_detector_warning"]


# --- trace fields, detector diagnostics and chunked reads ---


def test_derive_trace_fields_atif():
    out = derive_trace_fields({"response_text": _atif_blob(_FIX_STEPS), "prompt_text": "q"})
    assert "AssertionError" in (out.get("observations_text") or "")
    actions = out.get("actions_text") or ""
    assert "pytest" in actions or "terminal" in actions
    assert "fixed" in (out.get("reasoning_text") or "").lower()
    assert out.get("final_message") == "Done."


def test_derive_trace_fields_plain_noop():
    item = {"response_text": "just a normal answer", "reasoning_text": "think"}
    out = derive_trace_fields(item)
    assert out["response_text"] == "just a normal answer"
    assert out.get("actions_text") in (None, "")
    assert out["reasoning_text"] == "think"


def test_autofa_trace_detector_compiles_and_lints():
    source = '''
def claim_without_evidence(item, llm_client=None):
    from autofa.trace import steps
    saw_error = False
    claimed_fixed = False
    for s in steps(item):
        if "AssertionError" in s.observation or "FAILED" in s.observation:
            saw_error = True
        if "fixed" in s.reasoning.lower() or "all tests passed" in s.reasoning.lower():
            claimed_fixed = True
    return bool(saw_error and claimed_fixed)
'''
    ok, reason = lint_detector_source(source)
    assert ok, reason
    assert callable(compile_detector(source))
    assert run_detector(source, {"response_text": _atif_blob(_FIX_STEPS)}, llm=NullLLM()) is True


def test_parse_coverage_records_critical_on_unknown_dialect(monkeypatch):
    monkeypatch.setattr("autofa.diagnostics._DIAGNOSTICS", Diagnostics())
    # Structured-looking (has "tool_calls") but neither an ATIF nor a steps envelope.
    blob = json.dumps({"events": [{"type": "tool", "name": "bash"}], "tool_calls": []})
    cov = measure_parse_coverage([{"response_text": blob, "is_correct": False}] * 20)
    assert cov["n_sampled"] == 20 and cov["parsed"] == 0
    assert [(d.code, d.severity) for d in get_diagnostics().list()] == [
        ("dialect_unparsed", "critical")]


def test_contrastive_stats_planted_tool():
    def blob(tool, observation, reasoning):
        return json.dumps({"steps": [{"source": "agent", "observation": observation,
                                      "tool_calls": [{"function_name": tool, "arguments": {}}],
                                      "reasoning_content": reasoning}]})

    items = ([{"response_text": blob("evil_tool", "AssertionError: boom", "hmm"),
               "is_correct": False}] * 10
             + [{"response_text": blob("good_tool", "ok", "done"), "is_correct": True}] * 10)
    names = {t["name"]: t for t in contrastive_corpus_stats(items)["tools"]}
    assert names["evil_tool"]["lift"] is not None
    assert names["evil_tool"]["lift"] >= 1.0


def test_diversity_infers_trace_strategy_and_surfaces():
    source = '''
def detect(item, llm_client=None):
    from autofa.trace import steps
    for s in steps(item):
        if "AssertionError" in s.observation:
            return True
    return False
'''
    assert infer_strategy(source) == "trace"
    surfaces = infer_surfaces(source)
    assert "observations" in surfaces or "actions" in surfaces or "reasoning" in surfaces


def test_read_field_chunk_round_trips_large_field_and_its_end(monkeypatch):
    # Cross the old 5 MB intermediate cap, not just the outward page limit.
    original = "".join(f"line {i:07d} lorem ipsum dolor sit amet\n" for i in range(150_000))
    _install_runtime(monkeypatch, [{
        "_item_id": "big-1", "model_id": "m", "benchmark": "b", "problem_id": "1",
        "run_idx": 0, "is_correct": False, "response_text": original,
    }])
    total = len(original)
    assert tools_data.get_field_meta("big-1", "response_text")["total_chars"] == total
    parts, offset = [], 0
    for _ in range(200):
        out = tools_data.read_field_chunk("big-1", "response_text", offset=offset, limit=80_000)
        assert out["ok"] and out["total_chars"] == total
        parts.append(out["text"])
        if out["is_final"]:
            assert out["next_offset"] is None
            break
        offset = out["next_offset"]
        assert offset == sum(len(p) for p in parts)
    else:
        pytest.fail("read_field_chunk never reported is_final")
    assert "".join(parts) == original
    at_end = tools_data.read_field_chunk("big-1", "response_text", offset=total, limit=1000)
    assert at_end["returned_chars"] == 0 and at_end["is_final"] and at_end["next_offset"] is None
    last = tools_data.read_field_chunk("big-1", "response_text", offset=total - 10, limit=1000)
    assert last["text"] == original[-10:] and last["is_final"]


def test_classify_cache_issues_one_http_call(monkeypatch):
    hub_mod.HubLLM._CLASSIFY_CACHE.clear()
    llm = hub_mod.HubLLM(
        SimpleNamespace(chat=SimpleNamespace()), model="test-model",
        temperature=0.0, max_tokens=64, request_options={}, api="chat_completions",
    )
    calls = []

    def fake_completion(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))])

    monkeypatch.setattr(hub_mod, "chat_completions_safe", fake_completion)
    prompt, text = "Did the model skip verification?", "the model said done without tests"
    assert llm.classify(prompt, text) == llm.classify(prompt, text) == "yes"
    assert len(calls) == 1
    llm.classify(prompt, text + " (different)")
    llm.classify(prompt + "?", text)  # Distinct prompt on identical text is a distinct entry.
    assert len(calls) == 3
