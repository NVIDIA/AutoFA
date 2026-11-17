# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A grader's model calls are outcomes, never detector- or judge-visible evidence."""

import hashlib
import json
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import trace
from autofa.core.sandbox import run_detector
from autofa.data.gym_adapter import iter_gym_rows, materialize_eval_source
from autofa.data.local_import import import_local_run, materialize_local_traces
from autofa.data.parquet_store import Dataset
from autofa.data.source_errors import SourceDataError
from autofa.data.source_execution import execution_record
from autofa.data.source_export import materialize_source_traces
from autofa.data.source_model_calls import count_excluded_calls, drop_grader_calls
from autofa.data.sources import discover_eval_sources
from autofa.data.trace_fields import derive_trace_fields
from autofa.metrics.judge_evidence import make_snapshot

from support.traces import (
    BASELINE,
    QUESTION,
    REPLY,
    _model_call,
    _policy,
    _rollout,
    arena_record,
    prepare,
    write_trace_archive,
)

EXCLUDED = "evaluation model call excluded from execution evidence"
GOLD = "[correct_answer]: Hello there"
HLE_JUDGE = (
    "Judge whether the following [response] to [question] is correct or not based on the "
    f"precise and unambiguous [correct_answer] below.\n\n[question]: {QUESTION}\n\n"
    f"[response]: {REPLY}\n\n{GOLD}"
)
# Distinctive grader text: the arena prompt, the competitor's answer, both verdicts and
# the HLE judge's correct answer.
GRADER_TEXT = ("impartial judge", BASELINE, "Assistant B is better: [[B>>A]]",
               "Assistant A is better: [[A>>B]]", "[correct_answer]")
DETECTOR = """import re
from autofa.trace import model_calls

def detect(item, llm_client=None):
    text = item["response_text"] + str(model_calls(item))
    return bool(re.search(r"(?i)\\[\\[[AB]>>[AB]\\]\\]|final verdict|correct_answer", text))
"""


def hle_record():
    """NeMo Gym HLE rollout: the policy's answer, then a judge given the correct answer."""
    judge = _model_call("hle-judge", "judge_model", [{"role": "user", "content": HLE_JUDGE}],
                        "extracted_final_answer: Hello\ncorrect: no", item="msg-judge")
    return _rollout([_policy(), judge])


def assert_no_grader_text(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    assert not [marker for marker in GRADER_TEXT if marker in text]


def assert_no_grader_evidence(item):
    """Nothing a detector or behavior judge can read holds a grader's call."""
    assert_no_grader_text(item["response_text"])
    assert_no_grader_text(trace.model_calls(item))
    assert_no_grader_text(trace.record(item))
    assert [event["source_event_id"] for event in trace.events(item)
            if event.get("kind_raw") == "model_call"] in ([], ["policy"])
    assert run_detector(DETECTOR, item) is False
    snapshot = make_snapshot(item, item_id="0", source=DETECTOR)
    assert_no_grader_text({k: v for k, v in snapshot["fields"].items()
                           if k != "detector_source"})


def event_copies(record):
    """The event export's account of the raw calls: model_call events naming each call's
    id, with the request messages they sent pooled by hash."""
    events, texts, pool = [], [], {}
    for index, call in enumerate(record["ng_trajectory"]["model_calls"]):
        hashes = []
        for message in call["request"]["input"]:
            payload = json.dumps(message)
            hashes.append(hashlib.sha256(payload.encode()).hexdigest()[:16])
            pool.setdefault(hashes[-1], payload)
        events.append({"event_idx": index, "lane": "model_call", "lane_kind": "model_call",
                       "kind": "other", "kind_raw": "model_call", "role": None, "n_parts": 0,
                       "n_tool_calls": None, "source_event_id": call["model_call_id"],
                       "model_name": call["response"]["model"], "parts": []})
        texts.append({"lane": "model_call", "event_idx": index, "request_dialect": "responses",
                      "request_messages": json.dumps(hashes), "request_tools": None,
                      "request_params": "{}"})
    messages = [{"msg_idx": index, "msg_hash": digest, "msg_json": payload}
                for index, (digest, payload) in enumerate(pool.items())]
    return events, {"event_texts": texts, "prompt_messages": messages}


def assert_only_policy_call(item):
    calls = trace.model_calls(item)
    assert len(calls) == 1
    assert calls[0]["metadata"]["response_metadata"]["model_ref"]["name"] == "policy_model"
    assert calls[0]["request"] == [{"role": "user", "content": QUESTION}]
    assert calls[0]["response"][0]["content"][0]["text"] == REPLY
    assert_no_grader_evidence(item)


class Api:
    base_url = "https://api.test"

    def __init__(self, record):
        self.record = record

    def _get_json(self, path, query=None):
        assert path == "/v1/runs/run"
        return {"run_id": "run"}

    def download_trace_export(self, run_id, dest, *, progress=None, rollouts=None):
        def event(idx, role, text):
            return {"event_idx": idx, "kind": "message", "role": role, "n_parts": 1,
                    "n_tool_calls": 0, "parts": [{"part_idx": 0, "part_type": "text",
                                                  "channel": "content", "text": text,
                                                  "chars": len(text)}]}

        # Like the arena and HLE exports: the conversation lane holds only the policy's
        # exchange, and a model_call lane holds every captured call, the judges' too.
        copies, tables = event_copies(self.record)
        return write_trace_archive(dest, [{
            "metadata": {"run_id": "run", "task_id": "0", "repeat": 0, "n_events": 2,
                         "n_tool_calls": 0, "record_schema": "responses-api",
                         "raw_record_ref": "raw.jsonl:0", "agent_ref": "arena_agent"},
            "events": [event(0, "user", QUESTION), event(1, "assistant", REPLY), *copies],
            "calls": [], "tables": tables,
        }])


def prepared_export(tmp_path, monkeypatch, record):
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr("autofa.data.source_inputs._source_record", lambda *a, **k: source)
    export = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "problem_id": "0", "run_idx": 0, "response_text": "Summary", "reasoning_text": "",
        "is_correct": False, "reward": 0.0,
    }]), export)
    return export


@pytest.mark.parametrize("record", [arena_record, hle_record])
def test_grader_calls_and_their_exported_copies_never_reach_detectors_or_judges(
    tmp_path, monkeypatch, record,
):
    raw = record()
    export = prepared_export(tmp_path, monkeypatch, raw)
    output = materialize_source_traces(export, api=Api(raw), run_id="run")
    manifest = json.loads((output.parent / "manifest.json").read_text())
    judges = len(raw["ng_trajectory"]["model_calls"]) - 1
    assert manifest["n_model_calls_recovered"] == 1
    assert manifest["limitations"] == {EXCLUDED: judges}
    [row] = Dataset(parquet_paths=[output]).load_items()
    assert_only_policy_call(row)
    assert [event["source_event_id"] for event in trace.events(row, lane="model_call")] == [
        "policy",
    ]
    # The policy's request stays resolvable; only messages that graders alone sent go.
    messages = trace.record(row)["prompt_messages"]
    assert [json.loads(message["msg_json"]) for message in messages] == [
        {"role": "user", "content": QUESTION},
    ]
    assert "Evaluation records were withheld" in trace.coverage(row)["prompt_messages"]["reason"]
    fields = derive_trace_fields(row)
    assert fields["final_message"] == REPLY
    assert fields["input_messages"] == [{"role": "user", "content": QUESTION}]
    assert row["is_correct"] is False  # The label stays in its own column.


def test_caches_prepared_by_an_earlier_adapter_must_be_re_prepared(tmp_path, monkeypatch):
    """Saved runs stay bound to their cache; an earlier one could expose graders' calls."""
    from autofa.data import source_export

    # The refusal follows the grader-exclusion threshold, not later adapter bumps.
    assert source_export.GRADER_CALLS_EXCLUDED_VERSION == 15
    monkeypatch.setattr(source_export, "ADAPTER_VERSION", source_export.ADAPTER_VERSION + 1)
    path = tmp_path / "source-v14-legacy" / "dataset.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist([{
        "problem_id": "0", "run_idx": 0, "response_text": "{}", "is_correct": False,
    }]), path)
    dataset = Dataset(parquet_paths=[path])
    [location] = dataset.identity_index().by_id.values()
    for manifest, refused in (
        ({"adapter_version": 14, "output_sha256": "checked"}, True),
        ({"adapter_version": 15, "output_sha256": "checked"}, False),
        # Local import stores number their own adapter versions.
        ({"adapter_version": 3, "dataset_sha256": "checked", "store_id": "s"}, False),
    ):
        (path.parent / "manifest.json").write_text(json.dumps(manifest))
        for read in (dataset.load_items, lambda: dataset.read_locations([location])):
            if refused:
                with pytest.raises(SourceDataError, match="re-prepare the source"):
                    read()
            else:
                assert [row["problem_id"] for row in read()] == ["0"]


@pytest.mark.parametrize("name, grader", [
    ("aalcr_v1_1_judge_model", True), ("reward_model", True), ("prejudged_model", False),
])
def test_only_whole_role_words_name_a_grader(name, grader):
    from autofa.data.execution_policy import is_evaluation_model_ref

    assert is_evaluation_model_ref(name) is grader


@pytest.mark.parametrize("name", ["genrm_model", "Reward-Model", "org/judge"])
def test_the_raw_text_check_never_skips_a_graders_reference(name):
    """Rows the cheap text check rules out are never parsed, so it must miss no grader."""
    from autofa.data.execution_policy import is_evaluation_record, mentions_evaluation_model

    assert is_evaluation_record({"model_ref": name})
    for reference in (name, {"type": "responses_api_models", "name": name}):
        record = json.dumps({"response_metadata": {"model_ref": reference}})
        encoded = json.dumps({"response_metadata": json.dumps({"model_ref": reference})})
        for text in (record, encoded, json.dumps(encoded)):
            assert mentions_evaluation_model(text) and mentions_evaluation_model(text.encode())
    assert not mentions_evaluation_model(json.dumps({"model_ref": {"name": "policy_model"}}))


def test_converted_calls_naming_a_grader_in_encoded_metadata_are_dropped():
    def call(metadata):
        return {"index": 0, "request": [0], "response": [1], "metadata": metadata}

    trace_text = {"evidence_scope": {"limitations": {}}, "steps": [], "model_call_evidence": {
        "messages": [{"role": "user", "content": QUESTION},
                     {"role": "assistant", "content": "My final verdict is [[B>>A]]"}],
        "tools": [], "calls": [call({
            "response_metadata": json.dumps({"model_ref": {"name": "judge_model"}}),
        })],
    }}
    count_excluded_calls(trace_text, drop_grader_calls(trace_text))
    assert "model_call_evidence" not in trace_text and "[[B>>A]]" not in json.dumps(trace_text)
    assert trace_text["evidence_scope"]["limitations"] == {EXCLUDED: 1}


def test_requests_only_excluded_events_sent_are_withheld():
    """An excluded evaluation event (here an interactive judge lane) takes its requests too."""
    def message(text):
        payload = json.dumps({"role": "user", "content": text})
        return {"msg_hash": hashlib.sha256(payload.encode()).hexdigest()[:16],
                "msg_json": payload}

    shared, judge = message(QUESTION), message(HLE_JUDGE)

    def capture(lane, *messages):
        return {"lane": lane, "lane_kind": "model_call", "event_idx": 0, "kind": "other",
                "kind_raw": "model_call", "event_text": {"request_messages": json.dumps(
                    [m["msg_hash"] for m in messages])}}

    result = execution_record({}, [], [], execution={
        "events": [capture("model_call/policy_model", shared),
                   capture("model_call/judge_model", shared, judge)],
        "prompt_messages": [shared, judge],
        "availability": {"prompt_messages": {"status": "available", "reason": "Read."}},
    })
    assert [event["lane"] for event in result["events"]] == ["model_call/policy_model"]
    assert result["prompt_messages"] == [shared]
    assert result["availability"]["prompt_messages"]["reason"].endswith(
        "Evaluation records were withheld.",
    )


def test_local_nemo_gym_imports_withhold_grader_calls(tmp_path, monkeypatch):
    """The reader's tables do not say whose call each was; the raw record does."""
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    raw = arena_record()
    run = tmp_path / "run"
    run.mkdir()
    (run / "rollouts.jsonl").write_text(json.dumps(raw) + "\n")
    [source] = discover_eval_sources(run)
    assert source.kind == "evalrdl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parquet = materialize_eval_source(source)
    [row] = Dataset(parquet_paths=[parquet]).load_items()
    assert_no_grader_evidence(row)
    [call] = trace.model_calls(row)
    assert call["request"] == [{"role": "user", "content": QUESTION}]
    assert [event["source_event_id"] for event in trace.events(row)
            if event.get("kind_raw") == "model_call"] == ["policy"]
    loaded = json.loads(row["response_text"])
    judges = len(raw["ng_trajectory"]["model_calls"]) - 1
    assert loaded["evidence_scope"]["limitations"][EXCLUDED] == judges
    assert derive_trace_fields(row)["final_message"] == REPLY


def test_judge_calls_before_any_policy_call_are_not_the_agent_input(tmp_path, monkeypatch):
    record = prepare(tmp_path, monkeypatch, arena_record(policy_call=False))
    assert "initial_request" not in record["trace"]
    assert "model_call_evidence" not in record["trace"]
    assert record["trace"]["evidence_scope"]["limitations"][EXCLUDED] == 2
    assert record["trace"]["agent_request"]["messages"] == [
        {"role": "user", "content": QUESTION},
    ]
    assert record["agent_prompt"] == f"[Recorded agent request: user]\n{QUESTION}"
    assert_no_grader_text(record)


def test_generic_rollouts_without_a_trajectory_omit_graders_reasoning(tmp_path):
    record = arena_record()
    del record["ng_trajectory"]
    inline = record["ng_model_call_capture"]["calls"]
    for call, reasoning in zip(inline, ("Policy plan", "Judge: B wins", "Judge: A wins"),
                               strict=True):
        call["reasoning_content"] = reasoning
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    [row] = iter_gym_rows(path)
    assert row["reasoning_text"] == "Policy plan"


# Compact local imports keep grader attribution when discovery hydrates selected traces.
@pytest.fixture
def compact_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    run = tmp_path / "run"
    run.mkdir()
    records = [arena_record(), hle_record()]
    for index, record in enumerate(records):
        record["_ng_task_index"] = index
    (run / "rollouts.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        imported = import_local_run(run, efficiency_only=True)
    return run, imported


def test_lazy_selected_trace_uses_saved_attribution(compact_run, tmp_path):
    run, imported = compact_run
    # Re-reading changed logs would attribute calls wrongly without failing.
    (run / "rollouts.jsonl").write_text("not the imported run\n")
    compact = pq.read_table(imported.parquet_path).slice(1, 1)
    selected = tmp_path / "selected.parquet"
    pq.write_table(compact, selected)
    output = tmp_path / "rich.parquet"

    materialize_local_traces(imported.parquet_path.parent, selected, output)

    [row] = pq.read_table(output).to_pylist()
    assert row["problem_id"] == compact["problem_id"][0].as_py()
    assert row["is_correct"] == compact["is_correct"][0].as_py()
    assert_no_grader_evidence(row)
    assert json.loads(row["response_text"])["evidence_scope"]["limitations"][EXCLUDED] == 1
    stored = output.read_bytes()
    materialize_local_traces(imported.parquet_path.parent, selected, output)
    assert output.read_bytes() == stored


def _edit_attribution(imported, attribution):
    """Replace the saved grader attribution; None removes it."""
    path = imported.parquet_path.parent / "manifest.json"
    manifest = json.loads(path.read_text())
    if attribution is None:
        manifest.pop("grader_call_ids_by_record")
    else:
        manifest["grader_call_ids_by_record"] = attribution
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("attribution, error", [
    (None, "re-import the run"),
    ({}, "invalid grader attribution"),  # never read as "no graders"
])
def test_unverifiable_compact_attribution_fails_closed(compact_run, tmp_path, attribution, error):
    _, imported = compact_run
    _edit_attribution(imported, attribution)
    output = tmp_path / "rich.parquet"

    with pytest.raises(SourceDataError, match=error):
        materialize_local_traces(imported.parquet_path.parent, imported.parquet_path, output)

    assert not output.exists()


def test_old_hydrated_cache_is_rebuilt_before_reuse(compact_run, tmp_path):
    _, imported = compact_run
    output = tmp_path / "rich.parquet"
    rows = pq.read_table(imported.parquet_path).to_pylist()
    for row in rows:
        row["response_text"] = "stale evidence containing grader output"
    pq.write_table(pa.Table.from_pylist(rows), output)

    materialize_local_traces(imported.parquet_path.parent, imported.parquet_path, output)

    for row in pq.read_table(output).to_pylist():
        assert "stale evidence" not in row["response_text"]
        assert_no_grader_evidence(row)
