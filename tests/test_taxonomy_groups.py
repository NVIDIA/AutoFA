# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy classification groups become cohorts with success controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from autofa.data.cohort import Cohort, load_cohort, save_cohort, select_cohort
from autofa.taxonomy.groups import (
    add_success_controls,
    classification_facets,
    classification_rollout_identities,
    cohort_from_classifications,
)
from autofa.wizard.dataset import run_dataset_wizard


def _record(task: str, code: str, judge: str, *, evidence: str = "complete") -> dict:
    return {
        "kind": "taxonomy_classification", "status": "ok", "taxonomy_sha256": "taxonomy",
        "protocol_version": "protocol", "selected_code": code,
        "decision": "review" if code == "no_supported_failure" else "accept",
        "diagnosis": {"evidence_support": evidence},
        "judge": {"model": judge, "configuration_sha256": judge},
        "source": {"run_id": "run", "task_id": task, "repeat": 0},
    }


def _results(path: Path, cohort: Cohort, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.stem}.manifest.json").write_text(json.dumps({
        "selection": {"cohort_id": cohort.id}, "results": {"sha256": digest},
    }), encoding="utf-8")
    return path


def _row(task: str, reward: float, run_id: str = "run") -> dict:
    return {"run_id": run_id, "task_id": task, "repeat": 0, "reward": reward, "status": "ok",
            "ingest_status": "ok", "error": None, "is_scored": True}


def _remote(*run_ids: str) -> list[dict]:
    return [{"kind": "remote", "base_url": "https://api.test", "run_id": run_id}
            for run_id in run_ids]


API = SimpleNamespace(base_url="https://api.test")


@pytest.fixture
def home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))


@pytest.fixture
def parent(home) -> Cohort:
    cohort = Cohort(runs_where="", where="reward <= 0", sources=_remote("run"),
                    rows=[_row(task, 0.0) for task in ("same", "different", "unsupported")])
    save_cohort(cohort)
    return cohort


def _gym_run(path: Path, rewards: tuple[float, ...]) -> Path:
    path.mkdir()
    (path / "rollouts.jsonl").write_text(
        "".join(
            json.dumps({
                "_ng_task_index": index,
                "_ng_rollout_index": 0,
                "reward": reward,
                "responses_create_params": {
                    "model": "model",
                    "input": [{"role": "user", "content": "question"}],
                },
                "response": {
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "answer"}],
                    }],
                },
            }) + "\n"
            for index, reward in enumerate(rewards)
        ),
        encoding="utf-8",
    )
    return path


def test_taxonomy_groups_build_exact_child_selections(tmp_path, parent) -> None:
    first = _results(tmp_path / "first.jsonl", parent, [
        _record("same", "X", "judge-a"),
        _record("different", "X", "judge-a"),
        _record("unsupported", "no_supported_failure", "judge-a"),
    ])
    second = _results(tmp_path / "second.jsonl", parent, [
        _record("same", "X", "judge-b"),
        _record("different", "Y", "judge-b"),
        _record("unsupported", "no_supported_failure", "judge-b"),
    ])

    disagreement = cohort_from_classifications([tmp_path], group="judge-disagreement")
    unsupported = cohort_from_classifications([first, second], group="no-supported")
    chosen_judge = cohort_from_classifications([first, second], judge="Judge B", labels=["Y"])

    assert disagreement.identities() == [("different", 0)]
    assert unsupported.identities() == [("unsupported", 0)]
    assert chosen_judge.identities() == [("different", 0)]
    assert disagreement.parent_selection == parent.id
    assert disagreement.taxonomy_filter["group"] == "judge-disagreement"
    assert len(disagreement.taxonomy_filter["results"]) == 2

    facets = classification_facets([tmp_path])
    assert facets["paths"] == [first, second]
    assert facets["judges"] == ["Judge A", "Judge B"]
    assert facets["labels"] == {
        "X": "X", "Y": "Y", "no_supported_failure": "no_supported_failure",
    }
    assert facets["evidence"] == ["complete"]


def test_taxonomy_group_uses_report_consensus_and_evidence(tmp_path, parent) -> None:
    results = _results(tmp_path / "results.jsonl", parent, [
        _record("same", "X", "judge-a", evidence="complete"),
        _record("same", "X", "judge-b", evidence="complete"),
        _record("same", "Y", "judge-c", evidence="insufficient"),
    ])

    selected = cohort_from_classifications(
        [results], group="accepted", labels=["X"], evidence=["insufficient"],
    )

    assert selected.identities() == [("same", 0)]


def test_taxonomy_group_excludes_failed_reviews_kept_for_resume(tmp_path, parent) -> None:
    previous = _record("different", "Y", "judge-a")
    previous.update(
        selected_code=None, selected_leaf_id="leaf-y", selected_label="Leaf Y",
        first_pass={"method": "llm", "status": "completed", "selected_code": None,
                    "selected_leaf_id": "leaf-y"},
        second_pass={"method": "llm", "status": "completed"},
    )
    failed = {**previous, "status": "review_failed",
              "second_pass": {"method": "llm", "status": "failed"}}
    results = _results(tmp_path / "results.jsonl", parent,
                       [_record("same", "X", "judge-a"), previous, failed])

    facets = classification_facets([results])
    selected = cohort_from_classifications([results])
    retryable = classification_rollout_identities([results], include_review_failed=True)

    assert facets["labels"] == {"X": "X"}
    assert selected.identities() == [("same", 0)]
    assert retryable == {("run", "same", 0), ("run", "different", 0)}


def test_same_model_judge_configurations_are_separate_views(tmp_path, parent) -> None:
    first = _record("same", "X", "judge")
    first["judge"]["api"] = "chat_completions"
    second = _record("same", "Y", "judge")
    second["judge"]["api"] = "responses"
    results = _results(tmp_path / "results.jsonl", parent, [first, second])

    facets = classification_facets([results])

    assert len(facets["judges"]) == 2
    assert all(view.startswith("judge;api=") for view in facets["judges"])
    with pytest.raises(ValueError, match="Available judges"):
        cohort_from_classifications([results], judge="judge")
    selected = cohort_from_classifications(
        [results], labels=["Y"],
        judge=next(view for view in facets["judges"] if "api=responses" in view),
    )
    assert selected.identities() == [("same", 0)]


def test_taxonomy_group_rejects_results_changed_after_manifest(tmp_path, parent) -> None:
    results = _results(tmp_path / "results.jsonl", parent, [_record("same", "X", "judge-a")])
    results.write_text(results.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        cohort_from_classifications([results])


def test_taxonomy_labels_accept_names_and_reject_typos(tmp_path, parent) -> None:
    record = _record("same", "OWN-IGM", "judge-a")
    record["selected_label"] = "Incorrect goal management"
    results = _results(tmp_path / "results.jsonl", parent, [record])

    selected = cohort_from_classifications([results], labels=["Incorrect goal management"])
    assert selected.identities() == [("same", 0)]
    assert selected.taxonomy_filter["labels"] == ["OWN-IGM"]

    with pytest.raises(ValueError, match="Did you mean"):
        cohort_from_classifications([results], labels=["Incorrect goal managment"])


def test_taxonomy_group_adds_at_most_one_success_per_failure(
    tmp_path, parent, monkeypatch,
) -> None:
    results = _results(tmp_path / "results.jsonl", parent,
                       [_record("same", "X", "judge-a"), _record("different", "X", "judge-a")])
    selected = cohort_from_classifications([results], labels=["X"])
    selected.limit = 1
    calls = []

    def controls(**kwargs):
        calls.append(kwargs)
        return Cohort(runs_where="", where="reward > 0", sources=[],
                      rows=[_row(task, 1.0) for task in ("pass-a", "pass-b")])

    monkeypatch.setattr("autofa.data.cohort.select_cohort", controls)
    balanced = add_success_controls(selected, api=API)

    assert [row["task_id"] for row in balanced.rows] == [
        "same", "different", "pass-a", "pass-b",
    ]
    assert calls[0]["rewards"] == [">=0.5"]
    assert calls[0]["per_run"] == 2
    assert balanced.taxonomy_filter["controls"]["count"] == 2
    assert balanced.limit == 1


def test_taxonomy_bucket_hands_off_to_discovery_preparation(tmp_path, home) -> None:
    # A local run selects its success controls from the same imported store.
    parent = select_cohort(folders=[_gym_run(tmp_path / "gym", (0.0, 1.0))], rewards=["<=0"])
    save_cohort(parent)
    failure = parent.rows[0]
    record = _record(str(failure["task_id"]), "X", "judge-a")
    record["source"] = {
        "run_id": failure["run_id"], "task_id": failure["task_id"], "repeat": failure["repeat"],
    }
    results = _results(tmp_path / "results.jsonl", parent, [record])

    prepared = run_dataset_wizard(inputs=[str(results)], labels=["X"])

    assert prepared is not None
    cohort = load_cohort(prepared.selection_id)
    assert cohort.taxonomy_filter["labels"] == ["X"]
    assert cohort.taxonomy_filter["controls"]["count"] == 1
    assert pq.read_table(prepared.parquet_paths[0]).num_rows == 2


def test_every_selected_run_requires_its_own_success_control(monkeypatch) -> None:
    selected = Cohort(
        runs_where="", where="reward <= 0", sources=_remote("run-a", "run-b"),
        rows=[_row("failure", 0.0, run_id) for run_id in ("run-a", "run-b")],
        taxonomy_filter={"labels": ["X"]},
    )

    def controls(**kwargs):
        run_id = kwargs["api_runs"][0]
        rows = [] if run_id == "run-b" else [_row("success", 1.0, run_id)]
        return Cohort(runs_where="", where="reward > 0", sources=[], rows=rows)

    monkeypatch.setattr("autofa.data.cohort.select_cohort", controls)

    with pytest.raises(ValueError, match="run-b"):
        add_success_controls(selected, api=API)


def test_partial_taxonomy_failure_is_not_counted_as_its_own_control(monkeypatch) -> None:
    selected = Cohort(runs_where="", where="reward < 1", sources=_remote("run"),
                      rows=[_row("partial-failure", 0.3)], taxonomy_filter={"labels": ["X"]})
    calls = []

    def controls(**kwargs):
        calls.append(kwargs)
        return Cohort(runs_where="", where="reward >= 0.5", sources=[],
                      rows=[_row("control", 0.5)])

    monkeypatch.setattr("autofa.data.cohort.select_cohort", controls)

    balanced = add_success_controls(selected, api=API)
    repeated = add_success_controls(balanced, api=API)

    assert calls[0]["rewards"] == [">=0.5"]
    assert len(calls) == 1
    assert [row["task_id"] for row in balanced.rows] == ["partial-failure", "control"]
    assert repeated.rows == balanced.rows
    assert balanced.taxonomy_filter["controls"]["count"] == 1
