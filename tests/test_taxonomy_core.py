# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy definitions, classification records and manifests, outcome sources, assets, settings."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autofa.config import Config
from autofa.data import cohort as cohort_module
from autofa.data.cohort import Cohort
from autofa.data.source_errors import SourceDataError
from autofa.taxonomy.assets import (
    ASSET_INVESTIGATION_PROTOCOL,
    MAX_ASSET_REQUESTS,
    MAX_ASSET_ROUNDS,
    MAX_TOTAL_IMAGE_BYTES,
    AssetEvidenceError,
    artifact_descriptors,
    investigate_requests,
    task_input_indexes,
)
from autofa.taxonomy.classifier import TaxonomyClassifier
from autofa.taxonomy.manifest import EXPORTER, manifest_path, populations, write_manifest
from autofa.taxonomy.providers import OpenAIProvider, provider_configuration_sha256
from autofa.taxonomy.record import (
    append_classification,
    classification_record,
    has_usable_first_pass,
    read_classifications,
)
from autofa.taxonomy.schema import (
    DEFAULT_TAXONOMY_PATH,
    TAXONOMY_V0_PATH,
    TAXONOMY_V1_PATH,
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
)
from autofa.taxonomy.sources import LocalOutcomeSource, RemoteOutcomeSource
from autofa.ui import settings, taxonomy_settings
from autofa.ui.menus import MENU_BACK

# Taxonomy definition


def _leaf(node_id: str, code: str) -> dict:
    return {"id": node_id, "code": code, "label": node_id, "description": f"{node_id} mode"}


def _a1(data: dict) -> dict:
    return data["root"]["children"][0]["children"][0]


def _b1(data: dict) -> dict:
    return data["root"]["children"][1]["children"][0]


def _taxonomy(**overrides) -> dict:
    def edge(letter):
        return {"id": f"edge_{letter}", "label": f"Edge {letter.upper()}",
                "description": f"Edge {letter}.",
                "children": [_leaf(f"mode_{letter}1", f"{letter.upper()}1"),
                             _leaf(f"mode_{letter}2", f"{letter.upper()}2")]}

    return {
        "id": "test.taxonomy", "name": "Test taxonomy", "version": "1.0.0",
        "root": {"id": "root", "label": "Root", "description": "All failures.",
                 "children": [edge("a"), edge("b")]},
        **overrides,
    }


def test_shipped_taxonomy_versions_have_expected_contract():
    v0 = load_taxonomy(TAXONOMY_V0_PATH)
    v1 = load_taxonomy(TAXONOMY_V1_PATH)

    assert DEFAULT_TAXONOMY_PATH == TAXONOMY_V1_PATH
    assert load_taxonomy().content_sha256 == v1.content_sha256
    assert v0.taxonomy_id == v1.taxonomy_id == "autofa.interaction"
    assert (v0.version, v1.version) == ("0", "1")
    assert v0.content_sha256 != v1.content_sha256
    assert v0.source["url"] == v1.source["url"]
    assert {component["id"] for component in v1.components} >= {"model", "tool", "grader"}

    v0_leaves = v0.leaves()
    v1_leaves = v1.leaves()
    v0_codes = {leaf.code for leaf in v0_leaves}
    v1_codes = {leaf.code for leaf in v1_leaves}
    refined_codes = {
        "OBS-VAL",
        "OBS-TXT",
        "OBS-OBJ",
        "OBS-STR",
        "OBS-LNK",
        "RSN-CMS",
        "RSN-SOB",
        "RSN-UAI",
        "RSN-INF",
        "RSN-STM",
        "RSN-OTH",
        "GRD-REF",
        "GRD-EQV",
    }

    assert len(v0_leaves) == 41
    assert len(v1_leaves) == 53
    assert "OWN-RSN" in v0_codes
    assert "OWN-RSN" not in v1_codes
    assert refined_codes <= v1_codes
    assert refined_codes.isdisjoint(v0_codes)
    assert all(leaf.code and leaf.fault_side for leaf in (*v0_leaves, *v1_leaves))
    assert all(len(v0.path_ids(leaf.id)) == 4 for leaf in v0_leaves)
    assert {len(v1.path_ids(leaf.id)) for leaf in v1_leaves} == {4, 5}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["root"]["children"][0]["children"].pop(), "has one child"),
        (lambda d: _b1(d).update(code="A1"), "duplicate node code"),
        (lambda d: _b1(d).update(id="mode_a1"), "duplicate node id"),
        (lambda d: d["root"]["children"][0].update(id="Bad-Id"), "must match"),
        (lambda d: _a1(d).update(related=["mode_b1"]), "symmetric"),
    ],
    ids=["single-child", "duplicate-code", "duplicate-id", "bad-id", "asymmetric-related"],
)
def test_invalid_taxonomies_are_rejected(mutate, message):
    data = _taxonomy()
    mutate(data)

    with pytest.raises(TaxonomyError, match=message):
        Taxonomy.from_dict(data)


def test_load_reports_unreadable_files(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(TaxonomyError, match="cannot load taxonomy"):
        load_taxonomy(path)


def test_load_custom_taxonomy_with_key_order_independent_hash(tmp_path):
    data = _taxonomy(name="Custom", version="9")
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(dict(reversed(list(data.items())))), encoding="utf-8")

    taxonomy = load_taxonomy(path)

    assert taxonomy.name == "Custom"
    assert taxonomy.version == "9"
    assert {leaf.id for leaf in taxonomy.leaves()} == {"mode_a1", "mode_a2", "mode_b1", "mode_b2"}
    assert taxonomy.content_sha256 == Taxonomy.from_dict(data).content_sha256


# Classification records


def ok_record(rollout_id: str) -> dict:
    return classification_record(status="ok", judge_provider="chat", rollout_id=rollout_id)


def test_record_roundtrip_preserves_analysis_fields(tmp_path: Path) -> None:
    destination = tmp_path / "taxonomy_classifications.jsonl"
    record = classification_record(
        status="ok", judge_provider="chat", rollout_id="run:task:0", run_id="run",
        task_id="task", repeat=0, lane="conversation",
        evidence_receipt={"trace": {"status": "included"}},
        analysis={
            "incident_id": "run:task:0", "model": "test-model", "decision": "accept",
            "selected_leaf_id": "leaf_b", "selected_path_ids": ["root", "leaf_b"],
            "selected_path_codes": [None, "B"], "owner": "model", "window_count": 2,
            "window_sha256s": ["a", "b"],
            "trace": [{"question": "window_ledger", "selected_leaf_id": "leaf_b",
                       "rationale": None}],
        },
    )
    append_classification(destination, record)

    [line] = read_classifications(destination)
    assert line["trace"][0]["rationale"] is None
    assert line["evidence_receipt"]["trace"]["status"] == "included"
    assert line["source"]["lane"] == "conversation"
    assert line["judge"]["provider"] == "chat"
    assert line["selected_path_codes"] == [None, "B"]
    assert line["owner"] == "model"
    assert line["window_count"] == 2
    # Second-pass replay compares these hashes against the current windows.
    assert line["window_sha256s"] == ["a", "b"]


def test_codeless_leaf_is_usable_for_review_resume() -> None:
    assert has_usable_first_pass({
        "status": "review_failed", "selected_leaf_id": "missing", "selected_code": None,
        "first_pass": {"status": "completed"},
    })


def test_append_adds_a_delimiter_after_complete_unterminated_record(tmp_path: Path) -> None:
    destination = tmp_path / "results.jsonl"
    first, second = ok_record("run:first:0"), ok_record("run:second:0")
    destination.write_text(json.dumps(first), encoding="utf-8")

    append_classification(destination, second)

    assert read_classifications(destination) == [first, second]


def test_resume_repairs_only_a_truncated_final_jsonl_line(tmp_path: Path) -> None:
    destination = tmp_path / "results.jsonl"
    append_classification(destination, ok_record("run:task:0"))
    complete = destination.read_bytes()
    with destination.open("ab") as handle:
        handle.write(b'{"kind":"taxonomy_classification"')

    records = read_classifications(
        destination, tolerate_truncated_final=True, repair_truncated_final=True,
    )

    assert len(records) == 1
    assert destination.read_bytes() == complete


@pytest.mark.parametrize("following", [True, False], ids=["nonfinal", "terminated-final"])
def test_malformed_complete_jsonl_line_reports_path_and_line(tmp_path: Path, following) -> None:
    destination = tmp_path / "results.jsonl"
    text = '{"kind":"taxonomy_classification"\n'
    if following:
        text += json.dumps(ok_record("run:task:0")) + "\n"
    destination.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(f"{destination}:1: malformed JSON")):
        read_classifications(destination, tolerate_truncated_final=True)


# Result manifests: judges, cohort provenance and source populations


def _classifier() -> TaxonomyClassifier:
    taxonomy = Taxonomy.from_dict({
        "id": "example.taxonomy", "name": "Example", "version": "0",
        "root": {"id": "failure", "label": "Failure", "description": "Failure", "children": [
            {"id": "leaf", "label": "Leaf", "description": "Leaf", "code": "LEAF"},
            {"id": "other", "label": "Other", "description": "Other", "code": "OTHER"},
        ]},
    })
    provider = SimpleNamespace(
        api_key="secret-value", model="judge-model", endpoint="https://judge.example.test/v1",
        api_key_env="JUDGE_API_KEY", api="chat_completions", timeout_seconds=300.0,
        max_retries=3, backoff_seconds=0.5,
        request_options={"reasoning_effort": "high", "max_tokens": 4096},
    )
    return TaxonomyClassifier(taxonomy, provider)  # type: ignore[arg-type]


def _append(results: Path, run: str = "a", **fields) -> None:
    append_classification(results, classification_record(
        status="ok", judge_provider=fields.pop("judge_provider", "chat"),
        rollout_id=f"run-{run}:task-{run}:0", run_id=f"run-{run}", task_id=f"task-{run}",
        repeat=0, **fields,
    ))


def _population(rows: int, scored: int, zero: int, positive: int) -> dict:
    return {"rows": rows, "scored": scored, "zero_reward": zero, "positive_reward": positive,
            "unscored": rows - scored}


def _cohort(run: str, population: dict, dataset: dict, *, row=None, **source) -> Cohort:
    return Cohort(
        runs_where="", where="reward <= 0",
        sources=[{"kind": "remote", "base_url": "https://data.example.test",
                  "run_id": f"run-{run}", **source, "population": population}],
        rows=[{"run_id": f"run-{run}", "task_id": f"task-{run}", "repeat": 0, **(row or {})}],
        datasets=[{"run_id": f"run-{run}", **dataset}],
    )


def _manifest(results: Path, classifier=None, *, cohort=None, selection=None, workers=1) -> dict:
    output = write_manifest(results, cohort=cohort, classifier=classifier or _classifier(),
                            selection=selection or {}, workers=workers)
    return json.loads(output.read_text(encoding="utf-8"))


def test_classification_manifest_uses_evalrdl_names_and_population(tmp_path: Path) -> None:
    results = tmp_path / "judge.results.jsonl"
    _append(results, lane="conversation", benchmark="benchmark-a", model="model-a",
            agent_ref="agent-a")
    cohort = _cohort(
        "a", _population(10, 9, 3, 6),
        {"export_url": "https://data.example.test/v1/runs/run-a/export/autofa",
         "source_harness": "gym"},
        row={"agent_ref": "agent-a"}, dataset_contract="contract-a",
        metadata_watermark="watermark-a",
    )

    output = write_manifest(results, cohort=cohort, classifier=_classifier(),
                            selection={"sample": "random", "per_run": 3}, workers=4)
    document = json.loads(output.read_text(encoding="utf-8"))

    assert output == manifest_path(results) == tmp_path / "judge.results.manifest.json"
    assert document["exporter"] == EXPORTER
    assert document["schema"] == "store-v17"
    assert document["selection"]["cohort_id"] == cohort.id
    assert document["runs"][0]["population"]["zero_reward"] == 3
    assert document["runs"][0]["kind"] == "remote"
    assert document["runs"][0]["base_url"] == "https://data.example.test"
    assert document["runs"][0]["dataset_contract"] == "contract-a"
    assert document["runs"][0]["metadata_watermark"] == "watermark-a"
    assert document["rollouts"] == [{
        "run_id": "run-a", "task_id": "task-a", "repeat": 0, "agent_ref": "agent-a",
        "lane": "conversation",
    }]
    assert document["taxonomy"]["first_pass"] == "llm"
    assert document["taxonomy"]["second_pass"] == "llm"
    assert document["judge"]["api_key_env"] == "JUDGE_API_KEY"
    assert len(document["judge"]["configuration_sha256"]) == 64
    assert "secret-value" not in output.read_text(encoding="utf-8")
    assert document["results"]["status"] == {"ok": 1}
    assert populations(document)["run-a"]["scored"] == 9

    overridden = _manifest(results, cohort=cohort, workers=4,
                           selection={"cohort_id": "original-selection"})
    assert overridden["selection"]["cohort_id"] == "original-selection"


def test_judge_hash_and_manifest_keep_the_retired_max_tokens_key(tmp_path: Path) -> None:
    # Output limits live in request_options, but saved results and manifests were recorded
    # with a top-level max_tokens key; dropping it would change every configuration hash.
    provider = OpenAIProvider(api_key="k", model="m", endpoint="https://e/v1",
                              api_key_env="JUDGE_API_KEY", request_options={"max_tokens": 100})
    payload = {
        "model": "m",
        "endpoint": "https://e/v1",
        "api": "chat_completions",
        "request_options": {"max_tokens": 100},
        "reasoning_effort": "endpoint_default",
        "max_tokens": None,
        "asset_investigation": {
            "protocol": ASSET_INVESTIGATION_PROTOCOL,
            "max_requests": MAX_ASSET_REQUESTS,
            "max_rounds": MAX_ASSET_ROUNDS,
            "max_total_image_bytes": MAX_TOTAL_IMAGE_BYTES,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()

    assert provider_configuration_sha256(provider) == expected
    results = tmp_path / "results.jsonl"
    _append(results)
    classifier = TaxonomyClassifier(_classifier().taxonomy, provider)
    assert _manifest(results, classifier)["judge"] == {
        "provider": "chat", "model": "m", "base_url": "https://e/v1",
        "api_key_env": "JUDGE_API_KEY", "api": "chat_completions", "timeout_seconds": 300.0,
        "max_retries": 3, "backoff_seconds": 0.5, "reasoning_effort": "endpoint_default",
        "max_tokens": None, "configuration_sha256": expected,
    }


@pytest.mark.parametrize("second_pass", ["off", "llm"])
def test_agentic_first_pass_manifest_uses_harness_profile(tmp_path: Path, second_pass) -> None:
    results = tmp_path / "agentic.jsonl"
    _append(results, judge_provider="agentic",
            classification_method="agentic_taxonomy_classification")
    agentic = SimpleNamespace(model="gpt-6.1-sol", harness="codex", effort="medium",
                              max_turns=32, configuration_sha256=lambda: "a" * 64)
    base = _classifier()
    classifier = TaxonomyClassifier(
        base.taxonomy, base.provider if second_pass == "llm" else None,
        first_pass="agentic", second_pass=second_pass, agentic_provider=agentic,
    )

    document = _manifest(results, classifier)

    assert document["taxonomy"]["first_pass"] == "agentic"
    assert document["taxonomy"]["second_pass"] == second_pass
    assert document["judge"]["provider"] == "agentic"
    assert document["judge"]["harness"] == "codex"
    assert document["judge"]["model"] == "gpt-6.1-sol"
    assert document["judge"]["configuration_sha256"] == "a" * 64
    assert "base_url" not in document["judge"]
    if second_pass == "llm":
        second_judge = document["taxonomy"]["second_pass_judge"]
        assert second_judge["provider"] == "chat"
        assert second_judge["model"] == base.provider.model


def test_manifest_without_replacement_cohort_preserves_existing_provenance(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.jsonl"
    _append(results)
    cohort = _cohort("a", _population(2, 2, 1, 1), {"parquet_path": "/source/run-a.parquet"})
    _manifest(results, cohort=cohort, selection={"sample": "first"})

    document = _manifest(results, selection={"limit": 1})

    assert document["runs"][0]["population"]["zero_reward"] == 1
    assert document["reference_data"] == [
        {"run_id": "run-a", "parquet_path": "/source/run-a.parquet"}
    ]
    assert document["selection"]["cohort_id"] == cohort.id
    assert document["selection"]["sample"] == "first"
    assert document["selection"]["limit"] == 1

    _append(results, "b")
    second = _cohort("b", _population(3, 3, 2, 1), {"parquet_path": "/source/run-b.parquet"})
    document = _manifest(results, cohort=second, selection={"sample": "random"})

    assert {run["run_id"] for run in document["runs"]} == {"run-a", "run-b"}
    assert {
        run["run_id"]: run["population"]["zero_reward"] for run in document["runs"]
    } == {"run-a": 1, "run-b": 2}
    assert {
        reference["run_id"] for reference in document["reference_data"]
    } == {"run-a", "run-b"}
    assert {rollout["run_id"] for rollout in document["rollouts"]} == {"run-a", "run-b"}
    assert document["selection"]["cohort_id"] == second.id


def test_remote_preparation_counts_the_full_export_without_an_analytics_query(
    tmp_path: Path, monkeypatch
) -> None:
    full = tmp_path / "full.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"problem_id": task, "run_idx": 0, "reward": reward}
        for task, reward in (("a", 0.0), ("b", -1.0), ("c", 1.0), ("d", None))
    ]), full)

    class Api:
        base_url = "https://data.example.test"

        def get_run(self, run_id: str):
            assert run_id == "run-a"
            return {"harness": "gym"}

    monkeypatch.setattr(cohort_module, "materialize_remote_run", lambda *args, **kwargs: full)
    monkeypatch.setattr(cohort_module, "stage_export_subset", lambda path, identities: path)
    monkeypatch.setattr(cohort_module, "load_config",
                        lambda: SimpleNamespace(remote_full_traces=False))
    source = {"kind": "remote", "base_url": Api.base_url, "run_id": "run-a"}

    cohort_module._prepare_remote(Api(), source, [("a", 0)], None)

    # Unscored rows count toward the population but not toward failures or passes.
    assert source["population"] == _population(4, 3, 2, 1)


# Outcome sources


def _store(tmp_path: Path, source: Path) -> Path:
    store = tmp_path / "store"
    (store / "tables").mkdir(parents=True)
    (store / "manifest.json").write_text(json.dumps({"source_path": str(source)}))
    rows = [("0", 0, "out.txt", "output"), ("0", 1, "tests.log", "verifier"),
            ("0", 2, "grade.log", "verifier"), ("1", 0, "other", "verifier")]
    pq.write_table(pa.Table.from_pylist([
        {"task_id": "t1", "repeat": repeat, "artifact_idx": idx, "name": name, "role": role}
        for repeat, idx, name, role in rows
    ]), store / "tables" / "artifacts.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"task_id": "t1", "repeat": "0", "artifact_idx": 1, "text": "1 failed"},
    ]), store / "tables" / "artifact_texts.parquet")
    return store


def test_local_source_reads_rollout_artifacts_and_verifier_text(tmp_path: Path) -> None:
    artifacts = LocalOutcomeSource(_store(tmp_path, tmp_path / "run")).artifacts("t1", 0)

    assert [artifact["name"] for artifact in artifacts] == ["out.txt", "tests.log", "grade.log"]
    assert "text" not in artifacts[0]
    assert artifacts[1]["text"] == "1 failed"
    assert artifacts[2]["text_error"] == "artifact text was not imported"


def test_local_source_reads_raw_records_from_the_original_folder(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    first = json.dumps({"reward": 1}) + "\n"
    (run / "rollouts.jsonl").write_text(first + json.dumps({"reward": 0}) + "\n")
    source = LocalOutcomeSource(_store(tmp_path, run / "rollouts.jsonl"))

    assert source.read_raw_record("run", f"rollouts.jsonl:{len(first)}") == {"reward": 0}
    with pytest.raises(SourceDataError, match="escapes"):
        source.read_raw_record("run", "../outside.jsonl:0")
    with pytest.raises(SourceDataError, match="unavailable"):
        source.read_raw_record("run", "missing.jsonl:0")


def test_remote_source_hydrates_only_verifier_artifacts() -> None:
    class Api:
        def get_rollout_artifacts(self, run_id, task_id, repeat):
            assert (run_id, task_id, repeat) == ("run", "t1", 0)
            return [{"name": "out", "role": "output"}, {"name": "log", "role": "verifier"}]

        def get_rollout_verifier_artifacts(self, artifacts, **identity):
            assert identity == {"run_id": "run", "task_id": "t1", "repeat": 0}
            return [{**artifact, "text": "ok"} for artifact in artifacts
                    if artifact["role"] == "verifier"]

    artifacts = RemoteOutcomeSource(Api(), "run").artifacts("t1", 0)

    assert artifacts == [{"name": "out", "role": "output"},
                         {"name": "log", "role": "verifier", "text": "ok"}]


def test_remote_source_generic_hydration_is_strictly_allowlisted() -> None:
    class Api:
        def get_rollout_artifacts(self, run_id, task_id, repeat):
            return [
                {"name": "verifier.log", "role": "verifier", "kind": "log"},
                {"name": "model.patch", "role": "output", "kind": "patch"},
                {"name": "other.patch", "role": "output", "kind": "patch"},
                {"name": "golden.patch", "role": "reference", "kind": "patch"},
            ]

        def get_rollout_artifact_texts(self, artifacts, **identity):
            assert identity == {"run_id": "run", "task_id": "t1", "repeat": 0}
            return [{**artifact, "text": f"text:{artifact['name']}"} for artifact in artifacts]

    artifacts = RemoteOutcomeSource(Api(), "run").artifacts("t1", 0)

    assert artifacts[0]["text"] == "text:verifier.log"
    assert artifacts[1]["text"] == "text:model.patch"
    assert "text" not in artifacts[2]
    assert "text" not in artifacts[3]


# Visual asset evidence


def test_task_inputs_are_visuals_attached_before_the_first_reply(monkeypatch) -> None:
    monkeypatch.setattr("autofa.trace.events", lambda source: [
        {"event_idx": 0, "role": "system"},
        {"event_idx": 1, "role": "user"},
        {"event_idx": 2, "role": "assistant"},
        {"event_idx": 3, "role": "tool"},
        {"event_idx": 4, "role": "user"},
        {"event_idx": 1, "lane": "verdict", "role": None},
    ])
    catalog = [
        {"artifact_idx": 0, "lane": "", "event_idx": 1, "renderable": True},
        {"artifact_idx": 1, "event_idx": 3, "renderable": True},
        {"artifact_idx": 2, "event_idx": 4, "renderable": True},
        {"artifact_idx": 3, "event_idx": 1, "renderable": False},
        {"artifact_idx": 4, "renderable": True},
    ]

    assert task_input_indexes({"trace": "source"}, catalog) == [0]


def test_artifact_descriptors_exclude_bytes_urls_paths_and_references(monkeypatch) -> None:
    descriptor = {
        "artifact_idx": 3, "name": "prompt.png", "kind": "image", "origin": "inline",
        "mime": "image/png", "size_bytes": 42, "sha256": "a" * 64, "event_idx": 0,
        "part_idx": 1,
    }
    monkeypatch.setattr("autofa.trace.artifacts", lambda source: [
        {
            **descriptor,
            "text": "data:image/png;base64,SECRET",
            "data_url": "data:image/png;base64,SECRET",
            "download_url": "https://example.test/secret",
            "path": "/tmp/secret",
        },
        {"artifact_idx": 4, "name": "answer.json", "role": "reference"},
        {"artifact_idx": 5, "name": "golden.patch", "role": "output"},
    ])

    assert artifact_descriptors({"trace": "source"}) == [{**descriptor, "renderable": True}]


def test_investigate_requests_returns_verified_image_and_safe_receipt(monkeypatch) -> None:
    image = b"verified image"
    image_sha = hashlib.sha256(image).hexdigest()
    artifact_sha = "b" * 64
    monkeypatch.setattr(
        "autofa.trace.artifact_image",
        lambda source, artifact_idx, page=1: {
            "status": "available", "verified": True, "artifact_idx": artifact_idx,
            "page": page, "artifact_sha256": artifact_sha, "sha256": image_sha,
            "source_identity": {"run_id": "run", "task_id": "task", "repeat": 0},
            "transform_version": "v1", "locator": "page:1",
            "data_url": "data:image/png;base64," + base64.b64encode(image).decode(),
        },
    )
    catalog = [{"artifact_idx": 3, "name": "prompt.png", "kind": "image", "renderable": True}]
    requested: set[tuple[int, int]] = set()

    images, receipts = investigate_requests(
        {"trace": "source"}, catalog,
        [{"artifact_idx": 3, "page": 1, "reason": "The task depends on the image."}],
        requested=requested,
    )

    assert images[0]["data_url"].startswith("data:image/png;base64,")
    assert receipts[0]["artifact_sha256"] == artifact_sha
    assert receipts[0]["render_sha256"] == image_sha
    assert "data_url" not in receipts[0]
    assert requested == {(3, 1)}


@pytest.mark.parametrize(
    "asset_request, message",
    [
        ({"artifact_idx": 9, "page": 1, "reason": "needed"}, "unknown artifact_idx"),
        ({"artifact_idx": 3, "page": 0, "reason": "needed"}, "positive integer"),
    ],
)
def test_investigate_requests_rejects_invalid_requests(asset_request, message) -> None:
    with pytest.raises(AssetEvidenceError, match=message):
        investigate_requests(
            {"trace": "source"},
            [{"artifact_idx": 3, "kind": "image", "renderable": True}],
            [asset_request],
            requested=set(),
        )


# Settings menu


def menu(monkeypatch, *choices):
    """Answer the taxonomy settings menu with option values, ending with MENU_BACK."""
    answers = iter(choices)

    def choose(title, options, **kwargs):
        value = next(answers)
        if value is MENU_BACK:
            return MENU_BACK
        return next(option for option in options if option.value == value)

    monkeypatch.setattr(taxonomy_settings, "select_option", choose)


def test_taxonomy_settings_share_workflow_and_agentic_limit(monkeypatch) -> None:
    menu(monkeypatch, "workflow", ("agentic", "llm"), "turns", 64, MENU_BACK)
    saved = []

    result = taxonomy_settings.configure_taxonomy(Config(), save=saved.append)

    assert (result.taxonomy_first_pass, result.taxonomy_second_pass) == ("agentic", "llm")
    assert result.taxonomy_agentic_max_turns == 64
    assert saved[-1] == result


def test_agentic_limit_menu_preserves_saved_value(monkeypatch) -> None:
    captured = {}

    def choose(_title, options, **kwargs):
        captured["values"] = [option.value for option in options]
        captured["default_index"] = kwargs["default_index"]
        return MENU_BACK

    monkeypatch.setattr(taxonomy_settings, "select_option", choose)

    assert taxonomy_settings.select_agentic_max_turns(32) is None
    assert captured["values"][captured["default_index"]] == 32


def test_taxonomy_settings_are_validated_before_saving(monkeypatch) -> None:
    saved = []
    monkeypatch.setattr(settings, "_save_fields", lambda cfg, fields: saved.append((cfg, fields)))

    with pytest.raises(ValueError, match="positive integer"):
        settings.save_taxonomy_settings(Config(taxonomy_agentic_max_turns=0))

    assert saved == []


def test_taxonomy_settings_keep_menu_open_after_save_failure(monkeypatch) -> None:
    menu(monkeypatch, "workflow", ("agentic", "llm"), MENU_BACK)
    errors = []
    monkeypatch.setattr(taxonomy_settings, "show_action_error",
                        lambda error, **kwargs: errors.append(error))

    result = taxonomy_settings.configure_taxonomy(
        Config(), save=lambda candidate: (_ for _ in ()).throw(OSError("read-only config")),
    )

    assert (result.taxonomy_first_pass, result.taxonomy_second_pass) == ("llm", "llm")
    assert len(errors) == 1
