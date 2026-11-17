# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only recorded model-request or declared compact content, streamed boundedly, fills the input."""

import errno
import hashlib
import io
import json
import ssl
import threading
import urllib.error
import urllib.parse
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa import trace
from autofa.core.sandbox import run_detector
from autofa.data import source_inputs
from autofa.data.input_view import compact_export_input_json, derive_input_fields, parsed_input
from autofa.data.json_stream import parse_json
from autofa.data.parquet_store import Dataset
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.row_index import RowLocation
from autofa.data.source_inputs import (
    _read_input_record,
    _source_file,
    _source_record,
    cache_raw_source_file,
    initial_request,
    recover_initial_inputs,
)
from autofa.data.source_model_calls import recover_model_calls
from autofa.data.source_trajectory import read_raw_trajectory
from autofa.metrics.behavior import item_trace_for_judge
from autofa.metrics.judge_evidence import make_snapshot


def raw_record():
    return {
        "instance_config": {"problem_info": {"instance_id": "task", "patch": "SECRET_PATCH"}},
        "_ng_rollout_index": 0, "reward": 1.0, "resolved": True,
        "ng_trajectory": {"model_calls": [{"request": {
            "messages": [{"role": "system", "content": "Actual harness instruction"},
                         {"role": "user", "content": "Fix the submitted task"}],
            "tools": [{"type": "function", "function": {"name": "terminal"}}],
            "authorization": "SECRET_TOKEN",
        }}]},
    }


def recover_raw(tmp_path, monkeypatch, raw, *, task="task", repeat=0, steps=None):
    """Recover the inputs of one rollout whose selected raw source record is ``raw``."""
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw if isinstance(raw, bytes) else json.dumps(raw).encode() + b"\n")
    monkeypatch.setattr("autofa.data.source_inputs._source_record", lambda *a, **kw: source)
    trace = {"agent": {}, "evidence_scope": {"limitations": {}}}
    if steps is not None:
        trace["steps"] = steps
    record = {"metadata": {"record_schema": "responses-api", "task_id": task, "repeat": repeat,
                           "raw_record_ref": "source.jsonl:0"}, "trace": trace}
    recover_initial_inputs([record], api=None, run_id="run", root=tmp_path)
    return record


def test_only_exact_model_input_and_tools_are_extracted():
    result = initial_request(raw_record(), task="task", repeat=0)
    assert "Actual harness instruction" in result["prompt"]
    assert "Fix the submitted task" in result["prompt"]
    assert result["tools"][0]["function"]["name"] == "terminal"
    assert "SECRET" not in json.dumps(result)
    assert "reward" not in json.dumps(result) and "resolved" not in json.dumps(result)
    # The exact model request wins over the agent request and the benchmark statement.
    raw = raw_record()
    raw["responses_create_params"] = {"input": [{"role": "user", "content": "Agent input"}]}
    raw["instance_config"]["problem_info"]["problem_statement"] = "Benchmark context"
    assert initial_request(raw, task="task", repeat=0) == result


def test_recorded_agent_request_retains_input_tools_with_distinct_provenance(
    tmp_path, monkeypatch,
):
    raw = raw_record()
    del raw["ng_trajectory"]
    messages = [
        {"role": "system", "content": "Agent rules"},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Inspect the image"},
            {"type": "input_image", "image_url": "data:image/png;base64,IMAGE"},
        ]},
    ]
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    request = {
        "input": messages,
        "tools": [{"type": "function", "name": "search", "parameters": json.dumps(schema)}],
        "authorization": "SECRET_TOKEN", "metadata": {"reward": "SECRET_GRADER"},
    }
    raw["responses_create_params"] = json.dumps(request)
    record = recover_raw(tmp_path, monkeypatch, raw, steps=[])
    projected = _read_input_record(tmp_path / "source.jsonl", 0)
    assert "SECRET" not in json.dumps(projected)
    recovered = initial_request(projected, task="task", repeat=0)
    assert recovered["origin"] == "recorded_agent_request"
    assert recovered["messages"] == messages
    assert recovered["tools"][0]["parameters"] == schema
    assert record["initial_prompt"] == ""
    assert "initial_request" not in record["trace"]
    item = {"response_text": record["trace"]}
    view = parsed_input(item)
    assert view["origin"] == "recorded_agent_request"
    assert trace.input_messages(item) == messages
    assert trace.input_tools(item)[0]["parameters"] == schema
    assert trace.model_calls(item) == []
    assert any("exact initial model request not recorded" in s for s in view["limitations"])
    assert "SECRET" not in json.dumps(record)


def test_empty_agent_request_claims_input_only_when_tools_were_recorded():
    raw = raw_record()
    del raw["ng_trajectory"]
    raw["responses_create_params"] = {"input": [], "tools": []}
    recovered = initial_request(raw, task="task", repeat=0)
    assert recovered["prompt"] == ""
    assert "messages" not in recovered
    assert recovered["limitations"] == ["initial model request not recorded"]
    tools = [{"type": "function", "name": "search"}]
    raw["responses_create_params"]["tools"] = tools
    recovered = initial_request(raw, task="task", repeat=0)
    assert recovered["messages"] == []
    assert recovered["tools"] == tools
    assert recovered["origin"] == "recorded_agent_request"


@pytest.mark.parametrize("captured_request,index,repeat", [(True, 0, 0), (False, 704, 3)])
def test_gym_index_identity_recovers_input_from_raw_source(
    tmp_path, monkeypatch, captured_request, index, repeat,
):
    raw = raw_record()
    raw["_ng_task_index"] = index
    raw["_ng_rollout_index"] = repeat
    raw["instance_config"]["problem_info"].update(
        instance_id="instance_NodeBB__NodeBB-04998908-vnan", problem_statement="Repair NodeBB",
    )
    if not captured_request:
        del raw["ng_trajectory"]
    record = recover_raw(tmp_path, monkeypatch, raw, task=str(index), repeat=repeat)
    if captured_request:
        assert "Fix the submitted task" in record["initial_prompt"]
        assert record["trace"]["agent"]["tool_definitions"]
    else:
        assert record["initial_prompt"] == ""
        # Benchmark context is retained, but never marked as a recovered model request.
        assert "Repair NodeBB" in record["task_prompt"]
        assert "initial_request" not in record["trace"]["evidence_scope"]
        assert "not an exact model-request" in record["trace"]["evidence_scope"]["task_input"]
    assert "SECRET" not in json.dumps(record)


@pytest.mark.parametrize("identity,index", [("task", 704), ("terminal_bench::task", None)])
def test_matching_named_or_namespaced_identity_recovers_unchanged_request(identity, index):
    raw = raw_record()
    raw["instance_config"]["problem_info"]["instance_id"] = identity
    if index is not None:  # A gym index never replaces a matching named identity.
        raw["_ng_task_index"] = index
    before = json.dumps(raw)
    assert initial_request(raw, task="task", repeat=0) == initial_request(
        raw_record(), task="task", repeat=0,
    )
    assert json.dumps(raw) == before


@pytest.mark.parametrize("index,task,repeat", [(1, "0", 0), (0, "00", 0), (0, "0", 1)])
def test_gym_index_alias_requires_exact_task_and_repeat(index, task, repeat):
    raw = raw_record()
    raw["_ng_task_index"] = index
    with pytest.raises(DataApiError, match="identity does not match rollout"):
        initial_request(raw, task=task, repeat=repeat)


def test_gym_index_identity_requires_exact_integer():
    raw = raw_record()
    raw["instance_config"]["problem_info"].pop("instance_id")
    raw["_ng_task_index"] = False
    with pytest.raises(DataApiError, match="task identity does not match rollout"):
        initial_request(raw, task="0", repeat=0)


@pytest.mark.parametrize("raw,task,repeat,error", [
    (None, "task", 1, "identity"),
    # A corrupt raw record is an error, never a missing prompt.
    (b"not json\n", "task", 0, "complete JSON"),
])
def test_raw_source_errors_include_run_and_source_context(
    tmp_path, monkeypatch, raw, task, repeat, error,
):
    with pytest.raises(DataApiError, match=f"{error}.*run=run.*source=source.jsonl:0"):
        recover_raw(tmp_path, monkeypatch, raw or raw_record(), task=task, repeat=repeat)


@pytest.mark.parametrize("identity,task_name", [
    ("other_benchmark::task", "task"), ("terminal_bench::task", "wrong"),
])
def test_namespaced_identity_does_not_relax_task_checks(identity, task_name):
    raw = raw_record()
    raw["instance_config"]["problem_info"].update(instance_id=identity, task_name=task_name)
    with pytest.raises(DataApiError, match="identity does not match rollout"):
        initial_request(raw, task="task", repeat=0)


def test_uncaptured_request_retains_only_explicit_benchmark_task():
    raw = raw_record()
    del raw["ng_trajectory"]
    raw["ng_model_call_capture"] = {"calls": [{"tokens_in": 100}]}
    raw["instance_config"]["problem_info"]["problem_statement"] = "The reported task"
    raw["instance_config"]["problem_info"]["instance_dict"] = {"patch": "SECRET_PATCH"}
    result = initial_request(raw, task="task", repeat=0)
    assert result["prompt"] == ""
    assert result["task_statement"] == "The reported task"
    assert result["limitations"] == ["initial model request not recorded"]
    assert "SECRET" not in json.dumps(result)
    del raw["instance_config"]["problem_info"]["instance_id"]
    assert initial_request(raw, task="task", repeat=0)["task_statement"] == ""


def test_missing_request_and_nontext_are_explicit():
    raw = raw_record()
    raw["ng_trajectory"]["model_calls"] = []
    assert initial_request(raw, task="task", repeat=0)["limitations"]
    raw = raw_record()
    raw["ng_trajectory"]["model_calls"][0]["request"]["messages"][1]["content"] = [
        {"type": "text", "text": "Read this"}, {"type": "image_url", "image_url": "not fetched"},
    ]
    result = initial_request(raw, task="task", repeat=0)
    assert "Read this" in result["prompt"]
    assert "non-text initial request content not inspected" in result["limitations"]


def test_selected_recovery_fetches_only_referenced_raw_record(tmp_path):
    prefix = json.dumps({"irrelevant": "before"}).encode() + b"\n"
    wanted = json.dumps(raw_record()).encode() + b"\n"
    suffix = json.dumps({"irrelevant": "after"}).encode() + b"\n"
    raw = prefix + wanted + suffix
    requests = []

    class SliceApi:
        def _get_json(self, path, query, *, on_retry=None):
            requests.append((path, dict(query)))
            start = query["offset"]
            chunk = raw[start:start + query["length"]]
            return {
                "text": chunk.decode(),
                "length": len(chunk),
                "size": len(raw),
                "eof": start + len(chunk) == len(raw),
            }

    reference = f"source.jsonl:{len(prefix)}"
    record = {
        "metadata": {"record_schema": "responses-api", "task_id": "task", "repeat": 0,
                     "raw_record_ref": reference},
        "trace": {"agent": {}, "steps": [], "evidence_scope": {"limitations": {}}},
    }
    files = {}
    recover_initial_inputs(
        [record], api=SliceApi(), run_id="run", root=tmp_path / "cache",
        source_files=files,
    )

    assert "Fix the submitted task" in record["initial_prompt"]
    assert list(files) == [reference]
    assert files[reference].read_bytes() == wanted
    assert requests[0][1]["offset"] == len(prefix)
    assert all("/raw/slice" in path for path, _ in requests)


def test_projection_bounds_dense_suffix_after_long_string(tmp_path, monkeypatch):
    peak = 0

    class BoundedEvents(list):
        def send(self, event):
            nonlocal peak
            self.append(event)
            peak = max(peak, len(self))

    monkeypatch.setattr(ijson, "sendable_list", BoundedEvents)
    monkeypatch.setattr("autofa.data.source_inputs._JSON_CHUNK_BYTES", 64)
    raw = raw_record()  # Later requests repeat the history; only the first is projected.
    raw["ng_trajectory"]["model_calls"].append({"request": {"messages": [
        {"role": "user", "content": "LATER_HISTORY"},
    ]}})
    source = tmp_path / "dense.jsonl"
    source.write_bytes(
        b'{"ignored_blob":"' + b"x" * 100_000 + b'","ignored_dense":['
        + b'{"x":0},' * 50_000 + b"null]," + json.dumps(raw).encode()[1:] + b"\n"
    )
    projected = _read_input_record(source, 0)
    assert initial_request(projected, task="task", repeat=0) == initial_request(
        raw_record(), task="task", repeat=0,
    )
    serialized = json.dumps(projected)
    assert "ignored_blob" not in serialized and "ignored_dense" not in serialized
    assert "LATER_HISTORY" not in serialized and "SECRET" not in serialized
    assert peak <= 66


def test_streaming_preserves_utf8_and_jsonl_boundaries(tmp_path, monkeypatch):
    raw = raw_record()
    raw["ng_trajectory"]["model_calls"][0]["request"]["messages"][0]["content"] = (
        'Escapes: " \\ \n ' + "\u03bb\U0001f680" * 20
    )
    prefix = b'{"other":"\xce\xbb"}\r\n'
    source = tmp_path / "source.jsonl"
    source.write_bytes(
        prefix + json.dumps(raw, ensure_ascii=False).encode() + b"\nnot valid JSON\n"
    )
    monkeypatch.setattr("autofa.data.source_inputs._JSON_CHUNK_BYTES", 7)
    projected = _read_input_record(source, len(prefix))
    assert initial_request(projected, task="task", repeat=0) == initial_request(
        raw, task="task", repeat=0,
    )


@pytest.mark.parametrize("suffix", [
    b",\"ignored\": [1\n]}\n",  # the next line must not repair a truncated record
    b",\"ignored\": \"\xff\"}\n",  # invalid UTF-8 even in an excluded field
    b"} {}\n",  # multiple JSON values on the selected line
])
def test_streaming_validates_complete_record(tmp_path, suffix):
    source = tmp_path / "source.jsonl"
    source.write_bytes(json.dumps(raw_record()).encode()[:-1] + suffix)
    with pytest.raises(DataApiError, match="complete JSON|trailing JSON"):
        _read_input_record(source, 0)


def test_offset_inside_record_is_rejected(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"nested":' + json.dumps(raw_record()).encode() + b"}\n")
    with pytest.raises(DataApiError, match="record boundary"):
        _read_input_record(source, len(b'{"nested":'))


def test_later_request_cannot_replace_missing_first_request(tmp_path):
    raw = raw_record()
    raw["ng_trajectory"]["model_calls"].insert(0, {})
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(raw))
    result = initial_request(_read_input_record(source, 0), task="task", repeat=0)
    assert result["prompt"] == ""
    assert result["limitations"] == ["initial model request not recorded"]


def test_tool_schema_numbers_remain_json_serializable(tmp_path):
    raw = raw_record()
    raw["ignored_large_integer"] = 2**128
    raw["ng_trajectory"]["model_calls"][0]["request"]["tools"][0]["parameters"] = {
        "properties": {"amount": {"type": "number", "minimum": 0.125, "maximum": 2**80}},
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(raw))
    result = initial_request(_read_input_record(source, 0), task="task", repeat=0)
    assert json.dumps(result) == json.dumps(initial_request(raw, task="task", repeat=0))


# --- bounded JSON event stream behind the projection ---

# Both pass-through flags are checked against their defaults with one enabled pair.
@pytest.mark.parametrize("with_prefix,use_float,buf_size", [(False, False, 1), (True, True, 7)])
def test_matches_ijson_events(with_prefix, use_float, buf_size):
    data = json.dumps({
        "text": 'quotes " / slash \\ / newline\n / accented è / emoji 🦉',
        "array": [None, True, False, -123, 1.25, {"nested": []}],
    }, ensure_ascii=False).encode()
    original = ijson.parse if with_prefix else ijson.basic_parse
    expected = list(original(io.BytesIO(data), use_float=use_float))
    assert list(parse_json(
        io.BytesIO(data), with_prefix=with_prefix, use_float=use_float, buf_size=buf_size,
    )) == expected


def test_escape_parity_across_every_feed_boundary():
    values = ["\\" * count + '"tail' for count in range(12)] + ["\\" * 100_000, "end"]
    data = json.dumps(values).encode()
    events = list(parse_json(io.BytesIO(data), buf_size=1))
    assert events == list(ijson.basic_parse(io.BytesIO(data)))


def test_number_completed_at_eof_is_emitted():
    assert list(parse_json(io.BytesIO(b"-1.25e2"), buf_size=1)) == [
        ("number", Decimal("-1.25e2")),
    ]


# --- raw source downloads ---

class RawApi(DataApiClient):
    def __init__(self, size, *, max_attempts=3):
        super().__init__("https://example.test", timeout=1, max_attempts=max_attempts)
        self.size = size

    def _get_json(self, *args):
        return {"size": self.size}


def cache_paths(root):
    dest = root / (hashlib.sha256(b"source.jsonl").hexdigest() + ".jsonl")
    return dest, dest.with_suffix(".partial"), dest.with_suffix(".sha256")


def test_complete_non_responses_trace_skips_whole_raw_file_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "autofa.data.source_inputs._source_file",
        lambda *args, **kwargs: pytest.fail("raw file should not be downloaded"),
    )
    source_files = {}
    record = {"metadata": {"record_schema": "atif-v1.6", "raw_record_ref": "source.jsonl:0"},
              "trace": {"steps": [{"source": "assistant", "message": "done"}]}}
    cache_raw_source_file(
        record, api=object(), run_id="run", root=tmp_path, source_files=source_files,
    )
    assert source_files == {}


def test_download_over_2_gib_short_body_is_incomplete_and_cleaned(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b"{}\n"))
    with pytest.raises(DataApiError, match="IncompleteRead"):
        _source_file(RawApi(2 * 1024**3 + 1, max_attempts=1), "run", "source.jsonl", tmp_path)
    assert not list(tmp_path.iterdir())


def test_raw_download_cache_is_reused_only_when_its_checksum_matches(tmp_path, monkeypatch):
    payload = b"x" * (2 * 1024**2 + 7)
    downloads = []

    def download(*args, **kwargs):
        downloads.append(True)
        return io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", download)
    api = RawApi(len(payload))
    source = _source_file(api, "run", "source.jsonl", tmp_path)
    assert source.read_bytes() == payload
    assert _source_file(api, "run", "source.jsonl", tmp_path) == source
    assert len(downloads) == 1
    # Same-size corrupted cached bytes fail the checksum and are fetched again.
    source.write_bytes(payload.replace(b"x", b"y"))
    assert _source_file(api, "run", "source.jsonl", tmp_path).read_bytes() == payload
    assert len(downloads) == 2


@pytest.fixture
def source_preparation_receipts(tmp_path, monkeypatch):
    """Capture the bounded on-disk events consumed by the managed launch observer."""
    from autofa.session.managed_launch import preparation_event

    launch = tmp_path / "launch" / "launch.json"
    launch.parent.mkdir()
    launch.write_text(json.dumps({"launch_id": "selected-record-test"}))
    monkeypatch.setenv("AUTOFA_DRIVER_LAUNCH", str(launch))
    monkeypatch.setenv("AUTOFA_DRIVER_LAUNCH_ID", "selected-record-test")
    receipts = []

    def record_event(stage, *, diagnostic=None):
        preparation_event(stage, diagnostic=diagnostic)
        receipt = launch.parent / "preparation.json"
        if receipt.is_file():
            receipts.append(json.loads(receipt.read_text()))

    monkeypatch.setattr("autofa.data.source_inputs.preparation_event", record_event)
    return receipts


def test_selected_record_retry_and_cache_keep_exact_slice_content(tmp_path, monkeypatch):
    prefix, wanted, suffix = b"discarded\n", b'{"selected": true}\n', b"discarded too\n"
    raw = prefix + wanted + suffix
    queries, waits = [], []
    monkeypatch.setattr("autofa.data.source_inputs._JSON_CHUNK_BYTES", 8)
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", waits.append)

    def fetch(request, *, timeout):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
        start, length = int(query["offset"][0]), int(query["length"][0])
        queries.append((start, length))
        if len(queries) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 503, "unavailable", {"Retry-After": "17"}, io.BytesIO(),
            )
        chunk = raw[start:start + length]
        return io.BytesIO(json.dumps({
            "text": chunk.decode(), "length": len(chunk), "size": len(raw),
            "eof": start + len(chunk) == len(raw),
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    api = DataApiClient("https://api.example.test", max_attempts=3)
    cache = tmp_path / "cache"
    selected = _source_record(api, "run", "source.jsonl", len(prefix), cache)
    assert selected.read_bytes() == wanted
    assert selected.with_suffix(".sha256").read_text() == hashlib.sha256(wanted).hexdigest()
    assert queries == [(10, 8), (10, 8), (18, 8), (26, 8)]
    assert waits == [17]
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("cache miss"))
    assert _source_record(api, "run", "source.jsonl", len(prefix), cache) == selected


@pytest.mark.parametrize("failure", ["reset", "interrupt"])
def test_selected_record_failure_preserves_exception_and_sanitizes_receipt(
    tmp_path, monkeypatch, source_preparation_receipts, failure,
):
    attempts, waits = [], []
    secret = "https://signed.example.test/?token=SECRET"
    underlying = {
        "reset": urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, secret)),
        "interrupt": KeyboardInterrupt(secret),
    }[failure]

    def fetch(*args, **kwargs):
        attempts.append(True)
        raise underlying

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", waits.append)
    cache = tmp_path / "cache"
    api = DataApiClient("https://api.example.test", max_attempts=2)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else DataApiError) as caught:
        _source_record(api, "run", "source.jsonl", 0, cache)
    assert len(attempts) == (2 if failure == "reset" else 1)
    assert waits == ([5] if failure == "reset" else [])
    assert list(cache.iterdir()) == []
    receipt = source_preparation_receipts[-1]
    assert receipt["stage"] == "source_preparation_failed"
    assert "SECRET" not in json.dumps(source_preparation_receipts)
    assert "signed.example.test" not in json.dumps(source_preparation_receipts)
    assert receipt["diagnostic"]["retryable"] is (failure == "reset")
    if failure == "reset":
        assert caught.value.__cause__ is underlying
        assert caught.value.diagnostic == receipt["diagnostic"]
    else:
        assert caught.value is underlying


def test_raw_download_retries_body_from_zero(tmp_path, monkeypatch):
    payload = b'{"trace":"complete"}\n'
    calls, delays = [], []
    dest, partial, checksum = cache_paths(tmp_path)

    class Interrupted(io.BytesIO):
        def read(self, n=-1):
            chunk = super().read(n)
            if chunk:
                return chunk
            raise ConnectionResetError("reset")

    def request(url, timeout):
        calls.append(url)
        assert not partial.exists()
        return Interrupted(b"bad") if len(calls) == 1 else io.BytesIO(payload)

    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", delays.append)
    api = RawApi(len(payload))
    assert _source_file(api, "run", "source.jsonl", tmp_path) == dest
    assert len(calls) == 2 and calls[0] == calls[1]
    assert "/v1/runs/run/raw/file?path=source.jsonl" in calls[0]
    assert dest.read_bytes() == payload
    assert checksum.read_text() == hashlib.sha256(payload).hexdigest()
    assert not partial.exists()
    assert delays == [5]


def test_raw_retry_exhaustion_preserves_existing_cache(tmp_path, monkeypatch):
    dest, partial, checksum = cache_paths(tmp_path)
    dest.write_bytes(b"old cache")
    checksum.write_text("old checksum")
    calls, delays = [], []

    def request(*args, **kwargs):
        calls.append(True)
        assert not partial.exists()
        return io.BytesIO(b"short")

    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", delays.append)
    with pytest.raises(DataApiError, match="IncompleteRead"):
        _source_file(RawApi(100), "run", "source.jsonl", tmp_path)
    assert len(calls) == 3 and delays == [5, 10]
    assert dest.read_bytes() == b"old cache"
    assert checksum.read_text() == "old checksum"
    assert not partial.exists()


SIGNED_URL = "https://storage.test/object?signature=DO_NOT_LOG"


@pytest.mark.parametrize("failure,attempts,delays", [
    pytest.param(urllib.error.HTTPError(SIGNED_URL, 404, SIGNED_URL, {"Retry-After": "17"},
                                        io.BytesIO(SIGNED_URL.encode())),
                 1, [], id="http_404"),
    pytest.param(urllib.error.HTTPError(SIGNED_URL, 503, SIGNED_URL, {"Retry-After": "17"},
                                        io.BytesIO(SIGNED_URL.encode())),
                 3, [17, 17], id="http_503"),
    pytest.param(urllib.error.URLError(SIGNED_URL), 3, [5, 10], id="network"),
    pytest.param(urllib.error.URLError(ssl.SSLCertVerificationError(SIGNED_URL)), 1, [],
                 id="invalid_certificate"),
])
def test_raw_transport_errors_respect_retry_policy_and_redact_signed_urls(
    tmp_path, monkeypatch, caplog, failure, attempts, delays,
):
    calls, waits = [], []

    def request(*args, **kwargs):
        calls.append(True)
        raise failure

    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", waits.append)
    with pytest.raises(DataApiError) as error:
        _source_file(RawApi(100), "run", "source.jsonl", tmp_path)
    assert "DO_NOT_LOG" not in str(error.value) + caplog.text
    assert error.value.__cause__ is None
    assert (len(calls), waits) == (attempts, delays)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["oversize", "disk", "cancel"])
def test_raw_nontransport_failures_do_not_retry(tmp_path, monkeypatch, failure):
    calls, delays = [], []

    class Broken(io.BytesIO):
        def read(self, n=-1):
            if self.tell():
                if failure == "disk":
                    raise OSError(28, "No space left on device")
                raise KeyboardInterrupt()
            return super().read(n)

    def request(*args, **kwargs):
        calls.append(True)
        return io.BytesIO(b"too large") if failure == "oversize" else Broken(b"x")

    monkeypatch.setattr("urllib.request.urlopen", request)
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", delays.append)
    error = KeyboardInterrupt if failure == "cancel" else DataApiError
    with pytest.raises(error):
        _source_file(RawApi(3), "run", "source.jsonl", tmp_path)
    assert len(calls) == 1 and delays == []
    assert not list(tmp_path.iterdir())


def test_actual_truncated_chunked_response_is_retried(tmp_path, monkeypatch):
    """Reproduce the stdlib IncompleteRead from an HTTP stream, including partial bytes."""
    payload = b"x" * (1024**2 + 27419)
    attempts = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if "/slice?" in self.path:
                body = json.dumps({"size": len(payload)}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            attempts.append(self.path)
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            # First response closes without the terminating zero-length chunk.
            if len(attempts) > 1:
                self.wfile.write(b"0\r\n\r\n")
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr("autofa.data.remote_api.time.sleep", lambda _: None)
    try:
        api = DataApiClient(f"http://127.0.0.1:{server.server_port}", max_attempts=2)
        dest = _source_file(api, "run", "source.jsonl", tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(attempts) == 2
    assert dest.read_bytes() == payload
    assert dest.with_suffix(".sha256").read_text() == hashlib.sha256(payload).hexdigest()
    assert not dest.with_suffix(".partial").exists()


# --- large excluded scalars ---

def read_evidence(source, offset, reader):
    if reader == "input":
        return initial_request(_read_input_record(source, offset), task="0", repeat=0)
    if reader == "calls":
        return recover_model_calls(source, offset)
    return read_raw_trajectory(source, offset, task="0", repeat=0)


@pytest.fixture
def large_record():
    content = 'Model-visible λ 🚀 " \\\n' * 8192
    raw = {
        "ng_trajectory": {"model_calls": [{
            "request": {"messages": [{"role": "user", "content": content}]},
            "response": {"choices": [{"message": {
                "role": "assistant", "content": "Done",
            }}]},
        }]},
        "trajectory": {"steps": [
            {"step_id": 0, "source": "user", "message": content},
            {"step_id": 1, "source": "agent", "message": "Done"},
        ]},
        "test_output": "SECRET_GRADER" + 'Grader log \\\n' * 262144,
        # These fields occur after the large excluded scalar.
        "_ng_task_index": 0, "_ng_rollout_index": 0, "problem_idx": "0", "repeat": 0,
    }
    return raw, content


@pytest.mark.parametrize("reader,encoded", [
    ("input", False), ("calls", True), ("trajectory", True),
])
def test_large_scalars_preserve_evidence_and_record_boundaries(
    tmp_path, monkeypatch, large_record, reader, encoded,
):
    raw, content = large_record
    if encoded:
        for key in ("ng_trajectory", "trajectory"):
            raw[key] = json.dumps(raw[key], ensure_ascii=False)
    prefix = b'{"other":true}\n'
    source = tmp_path / "source.jsonl"
    source.write_bytes(prefix + json.dumps(raw, ensure_ascii=False).encode() + b"\ninvalid\n")
    sizes = []

    class RecordingReader(source_inputs._JsonlRecordReader):
        def read(self, size):
            sizes.append(size)
            return super().read(size)

    monkeypatch.setattr(source_inputs, "_JsonlRecordReader", RecordingReader)
    result = read_evidence(source, len(prefix), reader)
    if reader == "input":
        assert result["messages"][0]["content"] == content
    else:
        assert result["steps"][0]["message"] == content
        assert result["steps"][-1]["message"] == "Done"
    assert "SECRET_GRADER" not in json.dumps(result)
    assert max(sizes) > 64 * 1024


def test_malformed_tail_after_large_excluded_scalar_is_rejected(tmp_path, large_record):
    raw, _ = large_record
    source = tmp_path / "source.jsonl"
    source.write_bytes(json.dumps(raw).encode()[:-1] + b',"corrupt":[1,}\n')
    with pytest.raises(DataApiError, match="complete"):
        read_evidence(source, 0, "calls")


# --- declared compact export input ---

def _write_export(tmp_path, prompt, **changes):
    path = tmp_path / "dataset.parquet"
    item = {
        "problem_id": "task", "run_idx": 0,
        "system_text": "ACTUAL_SYSTEM", "prompt_text": prompt,
        "response_text": "Answer", "reasoning_text": "", **changes,
    }
    pq.write_table(pa.Table.from_pylist([item]), path)
    (tmp_path / "source.json").write_text(json.dumps({
        "run_id": "run", "base_url": "https://api.example.test",
        "export_path": "/v1/runs/run/export/autofa",
    }))
    return path


@pytest.mark.parametrize("prompt", [
    '[{"role":"system","content":"TEXT_INSIDE_USER_TASK"}]',
    '{"messages":[{"role":"developer","content":"Quoted rules"}]}',
])
def test_compact_user_json_retains_literal_text_and_roles_for_every_reader(tmp_path, prompt):
    # Model output shaped like an initial request cannot replace the declared input.
    path = _write_export(tmp_path, prompt, response_text=json.dumps({"initial_request": {
        "messages": [{"role": "system", "content": "JSON_IN_MODEL_OUTPUT"}],
    }}))
    dataset = Dataset([path])
    rows = [dataset.load_items()[0], dataset.read_locations([RowLocation(path, 0, 0)])[0]]
    expected = [
        {"role": "system", "content": "ACTUAL_SYSTEM"},
        {"role": "user", "content": prompt},
    ]
    for item in rows:
        view = parsed_input(item)
        assert view["messages"] == expected
        assert view["origin"] == "frontier_compact_export"
        assert run_detector(
            "def detect(item): return (item['input_messages'][0]['content'] == 'ACTUAL_SYSTEM' "
            "and item['input_messages'][1]['content'] == item['prompt_text'])", item,
        )
        snapshot = make_snapshot(item, item_id="task", source="")
        assert json.loads(snapshot["fields"]["input_messages"]) == expected
        assert f"[Exported input: user]\n{prompt}" in item_trace_for_judge(item)


def test_source_trace_capture_overrides_compact_fallback_marker():
    item = {"system_text": "Summary system", "prompt_text": "Summary task"}
    item["input_json"] = compact_export_input_json(item)
    item["response_text"] = json.dumps({"initial_request": {"messages": [
        {"role": "developer", "content": "Exact captured request"},
    ]}, "steps": []})
    view = parsed_input(item)
    assert view["origin"] == "captured_initial_request"
    assert view["messages"] == [{"role": "developer", "content": "Exact captured request"}]


@pytest.mark.parametrize("call_identity", [
    {"source_lane": "model_call"}, {"source_lane": "", "metadata": {"actor": "user_simulator"}},
])
def test_auxiliary_request_does_not_replace_explicit_agent_input(call_identity):
    item = {"system_text": "Agent rules", "prompt_text": "Actual customer question"}
    item["input_json"] = compact_export_input_json(item)
    item["response_text"] = json.dumps({
        "evidence_scope": {"event_lane": ""},
        "agent": {"name": "tested_agent"},
        "steps": [
            {"source": "system", "message": "Agent rules"},
            {"source": "assistant", "message": "How may I help?"},
            {"source": "user", "message": "Actual customer question"},
        ],
        "model_call_evidence": {
            "calls": [{"index": 0, "request_recorded": True,
                       "initial_request_recorded": True, "request": [0], **call_identity}],
            "messages": [{"role": "system", "content": "Simulator private instructions"}],
        },
    })
    view = parsed_input(item)
    assert view["origin"] == "frontier_compact_export"
    assert view["messages"] == [
        {"role": "system", "content": "Agent rules"},
        {"role": "user", "content": "Actual customer question"},
    ]


@pytest.mark.parametrize("call_identity, selected_lane", [
    ({}, ""),  # Legacy raw captures have no lane declaration.
    ({"source_lane": "conversation", "metadata": {"actor": "main"}}, "conversation"),
])
def test_agent_request_evidence_still_overrides_compact_fallback(call_identity, selected_lane):
    messages = [{"role": "developer", "content": "Full initial request"}]
    tools = [{"name": "lookup", "parameters": {"type": "object"}}]
    item = {"system_text": "Summary rules", "prompt_text": "Summary question"}
    item["input_json"] = compact_export_input_json(item)
    item["response_text"] = json.dumps({
        "evidence_scope": {"event_lane": selected_lane},
        "steps": [],
        "model_call_evidence": {
            "calls": [{"index": 0, "request_recorded": True, "request": [0],
                       "tools": 0, **call_identity}],
            "messages": messages, "tools": [tools],
        },
    })
    view = parsed_input(item)
    assert view["origin"] == "captured_initial_request"
    assert view["messages"] == messages
    assert view["tools"] == tools


@pytest.mark.parametrize("response_is_literal,system", [
    (True, "Additional instructions"), (False, "Recovered instructions"),
    (False, "Additional instructions"),
])
def test_empty_compact_task_only_allows_recovered_trace_for_source_fallback(
    response_is_literal, system,
):
    item = {"system_text": system, "prompt_text": ""}
    item["input_json"] = compact_export_input_json(
        item, response_is_literal=response_is_literal,
    )
    item["response_text"] = json.dumps({"steps": [
        {"source": "system", "message": "Recovered instructions"},
        {"source": "user", "message": "Recovered task"},
        {"source": "assistant", "message": "Answer"},
    ]})
    view = parsed_input(item)
    if response_is_literal:
        assert view["messages"] == [{"role": "system", "content": system}]
        assert view["origin"] == "frontier_compact_export"
    else:
        expected = [
            {"role": "system", "content": "Recovered instructions"},
            {"role": "user", "content": "Recovered task"},
        ]
        if system == "Additional instructions":
            expected.insert(0, {"role": "system", "content": system})
        assert view["messages"] == expected
        assert view["origin"] == "leading_trace_messages"


def test_loader_does_not_reinterpret_prepared_source_prompt_as_compact_text(tmp_path):
    prompt = json.dumps([{"role": "developer", "content": "Recorded input"}])
    path = _write_export(tmp_path, prompt)
    (tmp_path / "manifest.json").write_text(json.dumps({"adapter_version": 8}))
    item = Dataset([path]).load_items()[0]
    assert "input_json" not in item
    assert parsed_input(item)["messages"][1]["role"] == "developer"


def test_generic_parquet_with_export_like_columns_keeps_structured_input_support(tmp_path):
    prompt = json.dumps([{"role": "developer", "content": "Rules"}])
    path = _write_export(tmp_path, prompt)
    (tmp_path / "source.json").unlink()
    item = Dataset([path]).load_items()[0]
    assert "input_json" not in item
    assert parsed_input(item)["messages"][1] == {"role": "developer", "content": "Rules"}


def test_existing_input_capture_wins_literal_output_for_all_readers(tmp_path):
    request = {"messages": [{"role": "user", "content": "Existing capture"}]}
    encoded_request = json.dumps(request, indent=2)
    response = json.dumps({"initial_request": {"messages": [
        {"role": "system", "content": "JSON_IN_MODEL_OUTPUT"},
    ]}})
    path = _write_export(
        tmp_path, "Summary task", input_json=encoded_request, response_text=response,
    )
    dataset = Dataset([path])
    rows = [dataset.load_items()[0], dataset.read_locations([RowLocation(path, 0, 0)])[0]]
    for item in rows:
        assert item["input_json"] == encoded_request
        for view_item in (item, derive_input_fields(item)):
            assert parsed_input(view_item)["messages"] == request["messages"]
        snapshot = make_snapshot(item, item_id="task", source="")
        assert json.loads(snapshot["fields"]["input_messages"]) == request["messages"]
        assert snapshot["fields"]["input_json"] == encoded_request
        assert run_detector(
            "def detect(item): return item['input_messages'] == "
            "[{'role': 'user', 'content': 'Existing capture'}]", item,
        )
        assert "Existing capture" in item_trace_for_judge(item)


def test_only_explicit_literal_response_provenance_changes_precedence():
    request = {"messages": [{"role": "user", "content": "Existing capture"}]}
    item = {
        "input_json": json.dumps(request),
        "response_text": json.dumps({"initial_request": {"messages": [
            {"role": "user", "content": "Trace request"},
        ]}}),
        "input_parse": {  # Truthy, but not literally True.
            "response_origin": "frontier_compact_export", "response_is_literal": 1,
        },
    }
    assert parsed_input(item)["messages"][0]["content"] == "Trace request"
    item["input_parse"]["response_is_literal"] = True
    assert parsed_input(item)["messages"] == request["messages"]
