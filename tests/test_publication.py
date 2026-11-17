# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication orders by report quality and uploads saved evidence without usage accounting."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.library_transport import (
    PACKED_KEY,
    REVIEW_PACKED_KEY,
    pack_bucket_examples,
    unpack_bucket_examples,
)
from autofa.data.remote_library_api import (
    RemoteLibraryClient,
    RemoteLibraryError,
    _json_payload,
    prepare_library_payload,
)
from autofa.data.remote_library_publish import publish_run, published_library_is_current
from autofa.mcp.runtime import Runtime, load_runtime_from_run
from autofa.reports.failure_cards import _extract_autofa_data, build_data, render_html
from autofa_library_server.config import Settings
from autofa_library_server.lazy_report import report_for_saved_run
from autofa_library_server.main import create_app

from support.task_support import attach_task_support


@pytest.fixture
def publication_client(tmp_path, monkeypatch):
    """A remote-library client wired to an in-process local hub."""
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    settings = Settings(data_dir=tmp_path / "server", library_dir=tmp_path / "library",
                        runs_dir=tmp_path / "runs", local_token="test-token")
    client = RemoteLibraryClient(settings.public_url, token=settings.local_token)
    calls = []
    with TestClient(create_app(settings), base_url=settings.public_url) as server:
        def request(method, url, *, body=None, headers=None, **kwargs):
            calls.append((method, url, headers or {}))
            response = server.request(method, url, content=body,
                                      headers={**(headers or {}),
                                               "Authorization": "Bearer test-token"})
            if response.status_code >= 400:
                raise RemoteLibraryError(response.text, status=response.status_code)
            return response.content

        monkeypatch.setattr(client, "_request", request)
        yield client, calls


def test_fresh_process_publication_preserves_scores_and_ignores_usage(
    publication_client, tmp_path, monkeypatch,
):
    client, calls = publication_client
    cfg = Config(remote_library_base_url=client.base_url)
    monkeypatch.setattr("autofa.config.load_config", lambda: cfg)
    monkeypatch.setattr("autofa.session.usage.snapshot_usage",
                        lambda *a: pytest.fail("Report/publication must not read usage"))
    monkeypatch.setattr(Runtime, "ensure_hub_llm",
                        lambda *a: pytest.fail("Loading/publishing must not construct an LLM"))
    parquet = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "model_id": "model", "benchmark": "bench", "problem_id": "problem",
        "is_correct": False, "response_text": "saved response",
    }]), parquet)
    run = RunConfig(
        run_id="saved-profile", model_id="model", benchmarks=["bench"],
        parquet_paths=[str(parquet)], analysis_outputs=["agent"],
        classifier_profile_settings={"name": "strong", "comparison_id": "comparison",
            "settings": {"feature_model": "strong-model", "feature_api": "responses",
                         "feature_base_url": "https://model.example.test/v1",
                         "feature_api_key_env": "SYNTHETIC_KEY",
                         "feature_request_options": {"reasoning_effort": "medium"}}},
    )
    run.save()
    library = FunctionLibrary(model_id="model", benchmark="bench")
    library.upsert(FeatureFunction(name="saved", description="Saved condition",
        source="def detect(item):\n    return True\n", scores={
            "evaluated": True, "evaluation_context": "original-process-context",
            "recall": .42, "precision": .9, "behavior_verdicts": [{"verdict": "yes"}],
        }))
    library.save(run.run_dir() / "library.json")
    expected_scores = deepcopy(library.get("saved").scores)

    runtime = load_runtime_from_run(run, cfg)
    assert runtime.config.feature_model == "strong-model"
    assert runtime.config.feature_request_options == {"reasoning_effort": "medium"}
    remote = publish_run(run, cfg, runtime=runtime, client=client, show_progress=False,
                         trigger="manual_cli")

    assert remote.version_number == 1
    assert len(calls) == 1 and calls[0][0] == "POST"
    assert "Idempotency-Key" not in calls[0][2]
    assert not (run.run_dir() / "publication_intent.json").exists()
    assert FunctionLibrary.load(run.run_dir() / "library.json").get("saved").scores == (
        expected_scores)
    data = _extract_autofa_data((run.run_dir() / "failure_cards_local.html").read_text())
    assert "autofa_usage" not in data["meta"] and "contributions" not in data["meta"]
    receipt = json.loads((run.run_dir() / "publication.json").read_text())
    assert "usage_sha256" not in receipt
    assert published_library_is_current(run, cfg)
    (run.run_dir() / "usage.json").write_text('{"total_tokens":999}')
    assert published_library_is_current(run, cfg)


def test_packed_review_evidence_upload_round_trips_without_repreparing(
    publication_client, monkeypatch,
):
    client, calls = publication_client
    library = FunctionLibrary(version=2, model_id="model", benchmark="bench")
    library.upsert(FeatureFunction(name="large", description="Large evidence",
        source="def detect(item):\n    return True\n", scores={
            "recall": .42, "behavior_review_assessments": [{"evidence": "saved evidence " * 5000}],
        }))
    original = library.to_dict()
    packed = prepare_library_payload(original, max_bytes=2048)
    assert REVIEW_PACKED_KEY in json.loads(packed)
    monkeypatch.setattr("autofa.data.remote_library_api.prepare_library_payload",
                        lambda *a, **k: pytest.fail("Prepared evidence was normalized twice"))
    remote = client.upload_library(
        metadata={"branch_key": "large", "source_run_id": "large", "model_id": "model",
                  "benchmarks": ["bench"], "data_identity": None},
        library_json=packed, _prepared_library=True,
        failure_cards_html=b"<html>Saved report</html>",
    )
    restored = client._request("GET", remote.library_url)
    assert FunctionLibrary.from_dict(json.loads(restored)).to_dict() == original
    assert [method for method, _, _ in calls] == ["POST", "GET"]


# Quality ordering


def _function(name, coverage, precision=.9):
    row = {
        "name": name, "kind": "python", "source": "def detect(item): return True",
        "description": "An observable mistake", "verdict": {"code": "GOOD"},
        "scores": {
            "evaluated": True, "n_items": 100, "n_scored": 100, "n_flagged": 20,
            "recall": coverage, "precision": precision, "ppv_at_pi0": precision,
            "behavior_purity": 1, "behavior_specificity": 1,
        },
        "failure_localisations": {"item": {"passages": [{"quote": "saved evidence"}]}},
    }
    fn = attach_task_support(FeatureFunction.from_dict(row))
    row["scores"] = fn.scores
    row["scores"].update(behavior_n_judged=5, behavior_n_completed=5, behavior_n_yes=5)
    row["failure_localisations"].update(fn.failure_localisations)
    return row


def _payload():
    empty = _function("empty", 0)
    empty["scores"].update(n_flagged=0, precision=0, ppv_at_pi0=0, behavior_purity=None)
    pending = _function("pending", .99)
    pending["scores"]["evaluated"] = False
    return {"version": 1, "model_id": "model", "benchmark": "bench", "functions": [
        _function("weak_broad", .95, .3), pending,
        _function("good_narrow", .05), _function("ok_narrow", .4, .6), empty,
        _function("good_broad", .5), _function("weak_narrow", .2, .3),
        _function("ok_broad", .8, .6),
    ]}


EXPECTED = ["good_broad", "good_narrow", "ok_broad", "ok_narrow",
            "weak_broad", "weak_narrow", "empty", "pending"]


@pytest.mark.parametrize("input_type", ["dict", "path"])
def test_publication_orders_all_tiers_and_coverage_without_changing_local_evidence(
    tmp_path, input_type,
):
    payload = _payload()
    before = deepcopy(payload)
    serialized = json.dumps(payload).encode()
    path = tmp_path / "library.json"
    path.write_bytes(serialized)

    outgoing = json.loads(_json_payload(payload if input_type == "dict" else path))

    by_name = {row["name"]: row for row in before["functions"]}
    assert outgoing["functions"] == [by_name[name] for name in EXPECTED]
    assert payload == before
    assert path.read_bytes() == serialized
    assert _json_payload(outgoing) == _json_payload(json.loads(_json_payload(outgoing)))
    # The local report shows every visible tier in the same order that is published.
    report_html = render_html(build_data(FunctionLibrary.from_dict(_payload())))
    report_names = [row["name"] for row in _extract_autofa_data(report_html)["functions"]]
    assert report_names == [name for name in EXPECTED if name in report_names]


def test_prepacked_publication_preserves_membership_positions_and_all_saved_fields():
    payload = _payload()
    payload["version"] = 2
    payload["extra_metadata"] = {"preserve": ["unknown", "saved", "fields"]}
    for function in payload["functions"]:
        name = function["name"]
        function["extra_metadata"] = {"saved_name": name}
        function["scores"]["buckets"] = [{
            "bucket": "positive", "example_fail_ids": [f"{name}-fail-{i}" for i in range(3)],
            "example_pass_ids": [f"{name}-pass"], "saved_bucket_metadata": name,
        }, {"bucket": "negative", "example_fail_ids": [f"{name}-missed"]}]
    packed = pack_bucket_examples(payload)
    before = deepcopy(packed)

    outgoing_bytes = _json_payload(packed)
    outgoing = json.loads(outgoing_bytes)
    assert PACKED_KEY in outgoing
    by_name = {row["name"]: row for row in payload["functions"]}
    assert unpack_bucket_examples(outgoing) == {
        **payload, "functions": [by_name[name] for name in EXPECTED]}
    assert packed == before
    assert _json_payload(outgoing_bytes) == outgoing_bytes


@pytest.mark.parametrize("change", [None, "source", "retired_kind"])
def test_local_hydration_accepts_only_display_order_changes(
    tmp_path, monkeypatch, change,
):
    saved = _payload()
    saved["functions"][0]["kind"] = "python"
    uploaded = deepcopy(saved)
    uploaded["functions"].reverse()
    alias = uploaded["functions"][-1]
    if change == "source":
        alias["source"] = "def detect(item): return False"
    elif change == "retired_kind":
        # A pre-GA local save is not hydration authority for its published copy.
        saved["functions"][0]["kind"] = "hybrid"
    run = tmp_path / "saved"
    run.mkdir()
    serialized = json.dumps(saved)
    (run / "library.json").write_text(serialized)
    (run / "config.json").write_text(json.dumps({"reference_data": []}))
    settings = SimpleNamespace(runs_dir=tmp_path, max_local_library_bytes=10**6,
                               max_metadata_bytes=10**6)
    monkeypatch.setattr("autofa_library_server.lazy_report.build_lazy_report",
                        lambda library, references, **kwargs: "hydrated")
    before = deepcopy(uploaded)

    result = report_for_saved_run(settings, "saved", uploaded)

    assert result == ("hydrated" if change is None else None)
    assert uploaded == before
    assert (run / "library.json").read_text() == serialized
