# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy CLI: selection, evidence cache, judge profiles, resume and reports."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import OpenAI

from autofa.data.cohort import Cohort, cohorts_root, load_cohort, save_cohort
from autofa.data.source_errors import SourceDataError
from autofa.taxonomy import agentic_review, cli
from autofa.taxonomy.classifier import TaxonomyClassifier
from autofa.taxonomy.manifest import manifest_path, populations
from autofa.taxonomy.models import ProviderResponse
from autofa.taxonomy.record import read_classifications
from autofa.taxonomy.schema import load_taxonomy
from autofa.taxonomy.sources import LocalOutcomeSource

from support.taxonomy import _diagnosis_answer

pytestmark = pytest.mark.filterwarnings("ignore:Local import retained:UserWarning")


# AutoFA ships no default endpoint; provider tests name one explicitly.
JUDGE_URL = "https://judge.example/v1"


def cli_config(**values: Any):
    from autofa.config import load_config

    cfg = load_config()
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


class _FirstChoice:
    def evaluate(self, state: Any, questions: dict[str, dict[str, Any]]) -> ProviderResponse:
        answers = {
            question_id: {"type": "choice", "choice": next(iter(question["criteria"])),
                          "alternatives": [], "rationale": "scripted"}
            for question_id, question in questions.items()
            if question["type"] == "choice"
        }
        if "diagnosis" in questions:
            answers["diagnosis"] = _diagnosis_answer()
        return ProviderResponse("scripted", answers, {})


class _TrackingProvider(_FirstChoice):
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    def evaluate(self, state, questions):
        self.calls.append(state)
        return super().evaluate(state, questions)


_real_classifier = cli._classifier


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        cli, "_classifier", lambda args: TaxonomyClassifier(load_taxonomy(None), _FirstChoice())
    )
    return tmp_path / "home"


def _gym_run(tmp_path: Path, rewards: tuple[float, ...]) -> Path:
    run = tmp_path / "gym"
    run.mkdir()
    rollout = {
        "_ng_rollout_index": 0, "expected_answer": "42",
        "responses_create_params": {
            "model": "model", "input": [{"role": "user", "content": "What is six times seven?"}],
        },
        "response": {"status": "completed", "output": [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "41"}],
        }]},
    }
    (run / "rollouts.jsonl").write_text("".join(
        json.dumps({**rollout, "_ng_task_index": index, "reward": reward}) + "\n"
        for index, reward in enumerate(rewards)
    ))
    return run


def _prefetch(tmp_path: Path, rewards: tuple[float, ...]) -> tuple[Path, Path]:
    """Cache the evidence of a local run and return the cache and its saved selection."""
    cache = tmp_path / "evidence"
    run = _gym_run(tmp_path, rewards)
    assert cli.main(["--input", str(run), "--all", "--cache", str(cache), "--prefetch-only"]) == 0
    [selection] = list(cohorts_root().iterdir())
    return cache, selection


def _judge_args(*argv: str):
    return cli.build_classify_parser().parse_args(list(argv))


def _scored(task: str, reward: float) -> dict[str, Any]:
    return {"run_id": "run", "task_id": task, "repeat": 0, "agent_ref": "agent", "status": "ok",
            "ingest_status": "ok", "error": None, "is_scored": True, "reward": reward}


def _result_line(profile: dict[str, Any], rollout_id: str = "run:task:0", **fields) -> dict:
    """A stored classification row written under ``profile``."""
    return {
        "kind": "taxonomy_classification", "status": "ok",
        "protocol_version": profile["protocol_version"],
        "taxonomy_sha256": profile["taxonomy_sha256"],
        "judge": {
            "requested_model": profile["requested_model"],
            "configuration_sha256": profile["configuration_sha256"],
            "second_pass_configuration_sha256": profile["second_pass_configuration_sha256"],
        },
        "source": {"rollout_id": rollout_id}, **fields,
    }


def _write_lines(path: Path, *records: dict[str, Any]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _append_reviewed(destination: Path, provider: str, classifier, args=None, *,
                     status: str = "ok", **analysis) -> dict[str, Any]:
    """Write one classified row through the CLI's record writer and read it back."""
    cli._append_classification(destination, provider, {
        "status": status, "rollout_id": "run:task:0", "run_id": "run", "task_id": "task",
        "repeat": 0, "judge_attempted": True, "analysis": analysis,
    }, args or _judge_args(), classifier=classifier)
    [record] = read_classifications(destination)
    return record


def _agentic_profile(model: str, sha: str) -> SimpleNamespace:
    return SimpleNamespace(model=model, effort="high", harness="codex", max_turns=12,
                           configuration_sha256=lambda: sha)


# --- end-to-end classification ---


def test_local_folder_classifies_failures_and_writes_report(tmp_path) -> None:
    run = _gym_run(tmp_path, (0.0, 1.0, 0.0))

    cache = tmp_path / "evidence"
    assert cli.main(["--input", str(run), "--all", "--cache", str(cache)]) == 0

    [selection] = list(cohorts_root().iterdir())
    cached = [json.loads(path.read_text()) for path in cache.glob("*/*.json")]
    assert len(cached) == 2
    assert all(len(prepared["source_identity_sha256"]) == 64 for prepared in cached)
    assert all(prepared["incident"]["source_provenance"]["reward"] == 0.0 for prepared in cached)
    output = selection / "taxonomy_classifications.jsonl"
    records = read_classifications(output)
    assert [record["status"] for record in records] == ["ok", "ok"]
    assert {record["source"]["task_id"] for record in records} == {"0", "2"}
    assert records[0]["source"]["benchmark"]
    # Classification writes its own report and never a discovery usage scope.
    assert not output.with_suffix(".usage").exists()
    assert output.with_suffix(".html").read_text().startswith("<!DOCTYPE html>")

    report = tmp_path / "again.html"
    assert cli.main(["report", str(selection), "--output", str(report)]) == 0
    assert report.read_text().startswith("<!DOCTYPE html>")


def test_classification_restores_existing_logging_handlers(tmp_path) -> None:
    root = logging.getLogger("autofa")
    handler = logging.StreamHandler(io.StringIO())
    handler._autofa_console = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    try:
        assert cli.main(["--input", str(_gym_run(tmp_path, (0.0,))), "--all"]) == 0
        assert handler in root.handlers
        assert not handler.stream.closed
    finally:
        root.removeHandler(handler)
        handler.close()


def test_taxonomy_preview_and_prefetch_create_no_usage_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_classifier", lambda *a: pytest.fail("classifier setup"))
    source = _gym_run(tmp_path, (0.0,))
    preview, prefetch = tmp_path / "preview.jsonl", tmp_path / "prefetch.jsonl"
    assert cli.main(["--input", str(source), "--all", "--dry-run",
                     "--output", str(preview)]) == 0
    assert cli.main(["--input", str(source), "--all", "--prefetch-only",
                     "--cache", str(tmp_path / "cache"), "--output", str(prefetch)]) == 0
    assert not preview.with_suffix(".usage").exists()
    assert not prefetch.with_suffix(".usage").exists()


def test_existing_results_can_receive_only_a_later_second_pass(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    provider = _TrackingProvider(calls)
    monkeypatch.setattr(cli, "_classifier", lambda args: TaxonomyClassifier(
        load_taxonomy(None), provider, second_pass=args.second_pass or "llm",
    ))
    first = tmp_path / "first.jsonl"
    assert cli.main(["--input", str(_gym_run(tmp_path, (0.0,))), "--all",
                     "--second-pass", "off", "--output", str(first)]) == 0
    assert len(calls) == 1
    [retained] = read_classifications(first)
    retained["status"] = "review_failed"
    retained["second_pass"] = {"method": "llm", "status": "failed",
                               "error_type": "ProviderError", "error": "review failed"}
    _write_lines(first, retained)
    sidecar = manifest_path(first)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["results"]["sha256"] = hashlib.sha256(first.read_bytes()).hexdigest()
    manifest["results"]["status"] = {"review_failed": 1}
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")

    assert cli.main(["--input", str(first), "--second-pass", "llm"]) == 0

    [record] = read_classifications(tmp_path / "first.second-pass-llm.jsonl")
    assert len(calls) == 2
    assert "first_pass" in calls[-1]
    assert record["second_pass"]["method"] == "llm"
    assert record["second_pass"]["status"] == "completed"
    assert len(record["parent_classification"]["record_sha256"]) == 64
    assert record["parent_classification"]["results_path"] == str(first.resolve())


def test_existing_results_can_receive_second_pass_from_exact_evidence_cache(
    tmp_path, monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []
    preflights: list[Any] = []

    def classifier(args):
        if not getattr(args, "_defer_judge_preflight", False):
            preflights.append("immediate")
        return TaxonomyClassifier(load_taxonomy(None), _TrackingProvider(calls),
                                  second_pass=args.second_pass or "llm")

    monkeypatch.setattr(cli, "_classifier", classifier)
    monkeypatch.setattr(cli, "_preflight_classifier",
                        lambda _classifier, *, methods: preflights.append(methods))
    cache, first = tmp_path / "evidence", tmp_path / "first.jsonl"
    assert cli.main(["--input", str(_gym_run(tmp_path, (0.0,))), "--all", "--cache", str(cache),
                     "--second-pass", "off", "--output", str(first)]) == 0
    assert len(calls) == 1
    assert preflights == ["immediate"]
    first_manifest = json.loads(manifest_path(first).read_text(encoding="utf-8"))
    # The review must not need the saved selection: the evidence cache is exact.
    shutil.rmtree(cohorts_root())

    second = tmp_path / "second.jsonl"
    review = ["--cache", str(cache), "--input", str(first), "--second-pass", "llm",
              "--output", str(second)]
    with pytest.raises(SystemExit):
        cli.main([*review, "--per-run", "1"])
    assert cli.main(review) == 0

    [record] = read_classifications(second)
    assert len(calls) == 2
    assert "first_pass" in calls[-1]
    assert record["second_pass"]["status"] == "completed"
    assert record["parent_classification"]["results_path"] == str(first.resolve())
    second_manifest = json.loads(manifest_path(second).read_text(encoding="utf-8"))
    assert second_manifest["runs"] == first_manifest["runs"]
    assert second_manifest["reference_data"] == first_manifest["reference_data"]
    assert second_manifest["selection"]["parent_results"] == [
        {"file": first.name, "sha256": first_manifest["results"]["sha256"]}
    ]
    # Only the pending review backend is preflighted, and a finished resume calls nothing.
    assert preflights == ["immediate", {"llm"}]
    assert cli.main([*review, "--resume"]) == 0
    assert len(calls) == 2
    assert preflights == ["immediate", {"llm"}]


def test_fatal_cached_judge_error_still_writes_manifest(tmp_path, monkeypatch) -> None:
    cache, selection = _prefetch(tmp_path, (0.0,))
    output = tmp_path / "results.jsonl"
    written = []

    def fail(*args, **kwargs):
        output.write_text("partial", encoding="utf-8")
        raise RuntimeError("fatal judge error")

    monkeypatch.setattr(cli, "_classify_cached", fail)
    monkeypatch.setattr(cli, "_write_manifest",
                        lambda args, destination, cohort, classifier: written.append(destination))

    assert cli.main(["--input", selection.name, "--cache", str(cache),
                     "--output", str(output)]) == 2
    assert written == [output]


# --- hand-off to discovery ---


def _discovery_cli(monkeypatch, run_session):
    from autofa import cli as discovery_cli
    from autofa.config import Config

    monkeypatch.setattr(discovery_cli, "load_config",
                        lambda: Config(analysis_outputs=["agent"], data_api_base_url=""))
    monkeypatch.setattr("autofa_library_server.local.managed_dashboard",
                        lambda **kwargs: nullcontext(False))
    monkeypatch.setattr("autofa.session.runner.run_session", run_session)
    return discovery_cli


def test_new_cli_consumes_real_taxonomy_results_with_controls(tmp_path, monkeypatch) -> None:
    output = tmp_path / "taxonomy.jsonl"
    run = _gym_run(tmp_path, (0.0, 1.0))
    assert cli.main(["--input", str(run), "--all", "--output", str(output)]) == 0
    label = read_classifications(output)[0]["selected_code"]
    started = []
    monkeypatch.setattr("autofa.session.guidance.snapshot_guidance", lambda _, **kwargs: {})
    discovery_cli = _discovery_cli(
        monkeypatch, lambda selected, **kwargs: started.append((selected, kwargs)),
    )

    discovery_cli.main([
        "new", "--input", str(output), "--label", label, "--analysis-output", "agent", "--once",
    ])

    [(analysis_run, options)] = started
    prepared = load_cohort(analysis_run.selection_id)
    assert sorted(row["reward"] for row in prepared.rows) == [0.0, 1.0]
    assert prepared.taxonomy_filter["labels"] == [label]
    assert prepared.taxonomy_filter["controls"]["count"] == 1
    assert options["exit_after_discovery"] is True


def test_taxonomy_list_returns_before_prepare_or_discovery(tmp_path, monkeypatch, capsys) -> None:
    from autofa.data import cohort as cohort_module

    output = tmp_path / "taxonomy.jsonl"
    assert cli.main(["--input", str(_gym_run(tmp_path, (0.0,))), "--all",
                     "--output", str(output)]) == 0

    assert cohort_module.main(["--input", str(output), "--list"]) == 0
    discovery_cli = _discovery_cli(
        monkeypatch, lambda *args, **kwargs: pytest.fail("taxonomy listing must not start"),
    )
    discovery_cli.main([
        "new", "--input", str(output), "--list", "--analysis-output", "agent", "--once",
    ])

    assert capsys.readouterr().out.count("Labels:") == 2


# --- selection ---


def test_default_is_a_small_sample(tmp_path, capsys) -> None:
    assert cli.main(["--input", str(_gym_run(tmp_path, (0.0,) * 12)), "--dry-run"]) == 0

    assert ": 10 rollout(s) selected" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flags", "sample"),
    [
        (["--seed", "7"], "random"),
        (["--per-run", "2"], "random"),
        (["--per-run", "2", "--sample", "first"], "first"),
    ],
    ids=["seed", "per-run", "explicit-first"],
)
def test_effective_sampling_defaults_are_applied_before_validation(
    tmp_path, monkeypatch, flags, sample
) -> None:
    selected = {}

    def capture(**options):
        selected.update(options)
        return Cohort(runs_where="", where="", sources=[], rows=[])

    monkeypatch.setattr(cli, "select_cohort", capture)

    assert cli.main(["--input", str(tmp_path), "--dry-run", *flags]) == 0
    assert selected["sample"] == sample


def test_selection_flags_conflict_with_saved_selection(capsys) -> None:
    cohort = Cohort(runs_where="", where="", sources=[], rows=[])
    save_cohort(cohort)

    with pytest.raises(SystemExit):
        cli.main(["--input", cohort.id, "--api-run", "run"])
    assert "saved selections cannot be combined" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["--api-run", "run", "--all", "--per-run", "3"])


@pytest.mark.parametrize(
    ("rows", "expected"),
    [([_scored("passed", 1.0)], "No rollouts matched"),
     ([_scored(str(index), 0.0) for index in range(12)], "12 rollout(s) selected")],
    ids=["failures-only", "not-resampled"],
)
def test_saved_selection_keeps_its_rows_but_only_failures(rows, expected, capsys) -> None:
    cohort = Cohort(runs_where="", where="", rows=rows, sources=[
        {"kind": "remote", "base_url": "https://runs.example", "run_id": "run"},
    ])
    save_cohort(cohort)

    assert cli.main(["--input", cohort.id, "--dry-run"]) == 0
    assert expected in capsys.readouterr().out


def test_saved_whole_run_selection_can_be_refined(monkeypatch, capsys) -> None:
    cohort = Cohort(runs_where="", where="", rows=[], sources=[{
        "kind": "remote", "base_url": "https://runs.example", "run_id": "run",
        "selection": "all", "n_rollouts": 10,
    }])
    save_cohort(cohort)

    class Api:
        base_url = "https://runs.example"

        def __init__(self, *args, **kwargs):
            self.pages = [[_scored("failed", 0.0)], []]

        def query_analytics(self, sql):
            return {"rows": self.pages.pop(0)}

    monkeypatch.setattr(cli, "DataApiClient", Api)

    assert cli.main(["--input", cohort.id, "--dry-run"]) == 0
    assert "1 rollout(s) selected" in capsys.readouterr().out


@pytest.mark.parametrize("reward", [None, 1.0], ids=["unscored", "success"])
def test_classification_input_must_be_a_scored_failure(reward) -> None:
    with pytest.raises(SourceDataError, match="outside the reward filter"):
        cli._require_classification_reward(
            {"problem_id": "task", "run_idx": 0, "reward": reward}, "run", [("<=", 0.0)],
        )


def test_without_a_source_non_interactive_use_is_an_error(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        cli.main([])


def test_shards_require_a_shard_specific_output(capsys) -> None:
    assert cli.main(["--shard-count", "2"]) == 2
    assert "shard-specific filename" in capsys.readouterr().err


def test_interactive_mode_asks_how_many(monkeypatch, tmp_path, capsys) -> None:
    run = _gym_run(tmp_path, (0.0,) * 3)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("autofa.data.cohort._pick_interactive_sources",
                        lambda api, **kwargs: ([], [str(run)]))
    monkeypatch.setattr(cli, "_pick_count", lambda: 2)

    assert cli.main(["--dry-run"]) == 0

    assert ": 2 rollout(s) selected" in capsys.readouterr().out


def test_interactive_workflow_sets_the_same_passes(monkeypatch) -> None:
    monkeypatch.setattr("autofa.ui.taxonomy_settings.select_taxonomy_workflow",
                        lambda first, second: ("agentic", "llm"))
    args = _judge_args()

    assert cli._select_interactive_workflow(args, cli_config())
    assert (args.first_pass, args.second_pass) == ("agentic", "llm")


def test_workflow_uses_the_prefetched_selection(monkeypatch, tmp_path) -> None:
    selected = Cohort(runs_where="", where="", sources=[], rows=[], per_run=1)
    judge = SimpleNamespace(name="judge", model="judge-model", harness=None, params=(),
                            workers=1, output=tmp_path / "results.jsonl")
    workflow = SimpleNamespace(
        judge_base_url="https://judge.example/v1", judge_api_key_env="none",
        source=SimpleNamespace(api_runs=("run-a",), evalrdl_url=None, local_folder=None,
                               api_base_url=None),
        selection=SimpleNamespace(per_run=1, sample="first", seed=0, cache=tmp_path / "cache"),
        runs=(SimpleNamespace(
            first_pass="llm", second_pass="off", judges=(judge,),
            report=SimpleNamespace(output=tmp_path / "report.html", publish=None),
        ),),
    )
    calls = []

    def fake_classify(argv, **kwargs):
        calls.append(argv)
        if callback := kwargs.get("on_selection"):
            callback(selected)
        return 0

    config = cli_config()
    monkeypatch.setattr("autofa.config.load_config", lambda: config)
    monkeypatch.setattr(cli, "classify_main", fake_classify)
    monkeypatch.setattr(cli.report, "main", lambda argv: 0)

    assert cli.run_taxonomy_workflow(workflow) == 0
    # The prefetch picks the selection; the judge run classifies exactly that selection.
    assert len(calls) == 2
    assert calls[1][calls[1].index("--input") + 1] == selected.id

    calls.clear()
    workflow.source = SimpleNamespace(api_runs=(), evalrdl_url=None,
                                      local_folder=tmp_path / "local-run", api_base_url=None)
    assert cli.run_taxonomy_workflow(workflow) == 0
    assert calls[0][calls[0].index("--input") + 1] == str(tmp_path / "local-run")


# --- resume and limits ---


def test_resume_skips_classified_rollouts(tmp_path) -> None:
    run = _gym_run(tmp_path, (0.0, 0.0))
    command = ["--input", str(run), "--all", "--output", str(tmp_path / "out.jsonl")]
    assert cli.main(command) == 0

    assert cli.main([*command, "--resume"]) == 0

    assert len(read_classifications(tmp_path / "out.jsonl")) == 2


def test_resume_retries_only_a_failed_review(tmp_path, monkeypatch) -> None:
    calls = []

    class FailOneReview(_FirstChoice):
        def evaluate(self, state, questions):
            calls.append(state)
            if "first_pass" in state and len(calls) == 2:
                raise cli.ProviderError("review failed")
            return super().evaluate(state, questions)

    provider = FailOneReview()
    monkeypatch.setattr(cli, "_classifier",
                        lambda args: TaxonomyClassifier(load_taxonomy(None), provider))
    output = tmp_path / "out.jsonl"
    command = ["--input", str(_gym_run(tmp_path, (0.0,))), "--all", "--output", str(output)]

    assert cli.main(command) == 1
    [failed] = read_classifications(output)
    assert failed["status"] == "review_failed"
    assert cli.main([*command, "--resume"]) == 0

    failed, completed = read_classifications(output)
    # The resume reuses the stored first pass and only repeats the review.
    assert ["first_pass" in state for state in calls] == [False, True, True]
    assert completed["status"] == "ok"
    assert completed["first_pass"]["selected_code"] == failed["first_pass"]["selected_code"]
    assert completed["judge"]["resumed_pass"] == "llm"


def test_limit_counts_pending_rollouts_after_resume(tmp_path, monkeypatch) -> None:
    prepared_rows = []
    original_prepare = cli.prepare_data

    def track_prepare(cohort, **kwargs):
        prepared_rows.append(len(cohort.rows))
        return original_prepare(cohort, **kwargs)

    monkeypatch.setattr(cli, "prepare_data", track_prepare)
    output = tmp_path / "out.jsonl"
    command = ["--input", str(_gym_run(tmp_path, (0.0, 0.0, 0.0))), "--all", "--limit", "1",
               "--output", str(output)]

    assert cli.main(command) == 0
    assert cli.main([*command, "--resume"]) == 0

    # Each run reads only the pending rollouts it will classify, not the whole selection.
    assert prepared_rows == [1, 1]
    records = read_classifications(output)
    assert len({record["source"]["rollout_id"] for record in records}) == 2
    document = json.loads(manifest_path(output).read_text(encoding="utf-8"))
    # The manifest keeps the full selection and its population, not just the limited rows.
    assert len(document["rollouts"]) == 3
    assert document["selection"]["limit"] == 1
    assert [population["scored"] for population in populations(document).values()] == [3]


def test_append_rejects_a_different_judge_profile(tmp_path, monkeypatch, capsys) -> None:
    command = ["--input", str(_gym_run(tmp_path, (0.0,))), "--all",
               "--output", str(tmp_path / "results.jsonl")]
    assert cli.main(command) == 0

    provider = _FirstChoice()
    provider.model = "different-judge"
    monkeypatch.setattr(cli, "_classifier",
                        lambda args: TaxonomyClassifier(load_taxonomy(None), provider))

    assert cli.main(command) == 2
    assert "use a separate --output file" in capsys.readouterr().err
    assert len(read_classifications(tmp_path / "results.jsonl")) == 1


def test_resume_requires_the_same_taxonomy_and_judge_profile() -> None:
    profile = {"taxonomy_sha256": "taxonomy-a", "protocol_version": "protocol-a",
               "requested_model": "judge-a", "configuration_sha256": "configuration-a",
               "second_pass": "llm"}
    record = {
        "taxonomy_sha256": "taxonomy-a", "protocol_version": "protocol-a",
        "second_pass": {"method": "llm", "status": "completed", "outcome": "confirmed"},
        "judge": {"requested_model": "judge-a", "configuration_sha256": "configuration-a"},
    }
    assert cli._matches_classification_profile(record, profile)

    for field in ("taxonomy_sha256", "protocol_version"):
        assert not cli._matches_classification_profile({**record, field: "different"}, profile)
    for field in ("requested_model", "configuration_sha256"):
        changed = {**record, "judge": {**record["judge"], field: "different"}}
        assert not cli._matches_classification_profile(changed, profile)
    changed = {**record, "second_pass": {**record["second_pass"], "method": "off"}}
    assert not cli._matches_classification_profile(changed, profile)


def test_explicit_resume_file_is_not_filtered_by_its_name(tmp_path) -> None:
    classifier = cli._classifier(_judge_args())
    directory = tmp_path / "results"
    directory.mkdir()
    profile = cli._classification_profile(classifier)
    path = _write_lines(directory / "partial-results.jsonl", _result_line(
        profile, second_pass={"method": profile["second_pass"], "status": "completed"},
    ))

    for resume_from in (path, directory):
        args = _judge_args("--resume-from", str(resume_from))
        assert cli._resume_rollout_ids(args, tmp_path / "unused.jsonl", classifier) == {
            "run:task:0"
        }

    with pytest.raises(ValueError, match="does not exist"):
        cli._resume_rollout_ids(_judge_args("--resume-from", str(tmp_path / "missing.jsonl")),
                                tmp_path / "unused.jsonl", classifier)


def test_resume_retries_failed_structured_second_pass(tmp_path) -> None:
    profile = cli._classification_profile(cli._classifier(_judge_args()))
    path = _write_lines(tmp_path / "results.jsonl", *(
        _result_line(profile, f"run:{status}:0",
                     second_pass={"method": profile["second_pass"], "status": status})
        for status in ("failed", "completed", "skipped")
    ))

    assert cli._ok_rollout_ids(path, profile) == {"run:completed:0", "run:skipped:0"}


def test_saved_result_resume_requires_the_same_parent(tmp_path) -> None:
    classifier = TaxonomyClassifier(load_taxonomy(None), _FirstChoice(), first_pass="llm",
                                    second_pass="llm", saved_first_pass="llm")
    parent = {
        "kind": "taxonomy_classification", "status": "ok",
        "classification_method": "llm_as_a_judge",
        "judge": {"provider": "openai", "requested_model": "parent-model"},
        "first_pass": {"method": "llm"}, "second_pass": {"method": "off", "status": "skipped"},
        "selected_code": "A", "source": {"rollout_id": "run:task:0"},
    }
    parent_configuration = cli.report._judge_configuration(parent)
    profile = cli._classification_profile(classifier, parent_configuration=parent_configuration)
    path = _write_lines(tmp_path / "review.jsonl", _result_line(
        profile, classification_method="llm_as_a_judge", first_pass={"method": "llm"},
        second_pass={"method": "llm", "status": "completed"},
        parent_classification={"record_sha256": cli.classification_sha256(parent),
                                "judge_configuration": parent_configuration},
    ))
    args = _judge_args("--resume-from", str(path), "--second-pass", "llm")

    args._second_pass_parents = {"run:task:0": parent}
    assert cli._resume_rollout_ids(args, tmp_path / "unused.jsonl", classifier) == {
        "run:task:0"
    }
    # A revised parent verdict must be reviewed again.
    args._second_pass_parents = {"run:task:0": {**parent, "selected_code": "B"}}
    assert cli._resume_rollout_ids(args, tmp_path / "unused.jsonl", classifier) == set()

    different_judge = {**parent, "judge": {"provider": "openai",
                                           "requested_model": "different-parent"}}
    with pytest.raises(ValueError, match="different taxonomy, judge, or second-pass"):
        cli._validate_output_profile(
            path, classifier, resume=True,
            parent_configuration=cli.report._judge_configuration(different_judge),
        )


def test_interrupt_checkpoints_running_review_after_cooperative_cancel(monkeypatch) -> None:
    class CompletingFuture(Future):
        def cancel(self):
            if not self.done():
                self.set_result("preserved first pass")
            return False

    first, running, pending = Future(), CompletingFuture(), Future()
    first.set_result("first")

    def interrupted(_futures):
        yield first
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "as_completed", interrupted)
    cancellations = []
    consumed = []

    with pytest.raises(KeyboardInterrupt):
        cli._consume_futures(
            [first, running, pending], lambda future: consumed.append(future.result()),
            cancellation=SimpleNamespace(cancel=lambda: cancellations.append(True)),
        )

    assert cancellations == [True]
    assert consumed == ["first", "preserved first pass"]
    assert pending.cancelled()


# --- evidence cache ---


def test_saved_selection_filters_evidence_cache_and_keeps_population(
    tmp_path, monkeypatch
) -> None:
    cache, selection = _prefetch(tmp_path, (0.0, 1.0, 0.0))
    cached = json.loads(next(cache.glob("*/*.json")).read_text(encoding="utf-8"))
    foreign = {**cached, "rollout_id": "foreign:task:0", "run_id": "foreign"}
    cli._write_cache(cli._cache_path(cache, foreign["rollout_id"]), foreign)
    monkeypatch.setattr(cli, "prepare_data", lambda *args, **kwargs: pytest.fail("read sources"))

    output = tmp_path / "selected.jsonl"
    assert cli.main(["--cache", str(cache), "--input", selection.name,
                     "--output", str(output)]) == 0

    records = read_classifications(output)
    assert len(records) == 2
    assert all(record["source"]["run_id"] != "foreign" for record in records)
    document = json.loads(manifest_path(output).read_text(encoding="utf-8"))
    assert document["selection"]["cohort_id"] == selection.name
    assert document["runs"][0]["population"] == {
        "rows": 3, "scored": 3, "zero_reward": 2, "positive_reward": 1, "unscored": 0,
    }


def test_evidence_cache_saves_a_refined_selection(tmp_path) -> None:
    cache, selection = _prefetch(tmp_path, (0.0, 0.0))
    original = load_cohort(selection)
    reordered = Cohort(runs_where=original.runs_where, where=original.where,
                       sources=original.sources, rows=list(reversed(original.rows)))
    save_cohort(reordered)
    output = tmp_path / "results.jsonl"

    assert cli.main(["--cache", str(cache), "--input", reordered.id, "--output", str(output)]) == 0

    refined_id = json.loads(manifest_path(output).read_text())["selection"]["cohort_id"]
    assert refined_id != reordered.id
    assert (cohorts_root() / refined_id / "cohort.json").is_file()


@pytest.mark.parametrize("damage", ["missing", "different-source", "outside-reward"])
def test_evidence_cache_reprepares_unusable_entries_from_sources(
    tmp_path, damage, monkeypatch
) -> None:
    cache, selection = _prefetch(tmp_path, (0.0, 0.0))
    path = next(cache.glob("*/*.json"))
    if damage == "missing":
        path.unlink()
    else:
        prepared = json.loads(path.read_text(encoding="utf-8"))
        if damage == "different-source":
            prepared["source_identity_sha256"] = "different"
        else:
            prepared["incident"]["source_provenance"]["reward"] = 1.0
        cli._write_cache(path, prepared)
    original_prepare = cli.prepare_rollout
    prepared_rollouts = []

    def track_prepare(*args, **kwargs):
        prepared_rollouts.append(args[0])
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(cli, "prepare_rollout", track_prepare)
    output = tmp_path / "results.jsonl"

    assert cli.main(["--input", selection.name, "--cache", str(cache),
                     "--output", str(output)]) == 0

    assert [record["status"] for record in read_classifications(output)] == ["ok", "ok"]
    assert prepared_rollouts
    if damage == "outside-reward":
        rebuilt = json.loads(path.read_text(encoding="utf-8"))
        assert rebuilt["incident"]["source_provenance"]["reward"] == 0.0


def _cache_entry(args, status: str, **fields) -> dict[str, Any]:
    return {"cache_version": cli.EVIDENCE_CACHE_VERSION,
            "max_serialized_characters": args.max_characters, "rollout_id": "run:task:0",
            "run_id": "run", "task_id": "task", "repeat": 0, "status": status, **fields}


def test_prefetch_caches_non_prepared_records(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    args = _judge_args("--cache", str(cache), "--prefetch-only")
    calls = []

    def prepare(*call_args, **kwargs):
        calls.append(call_args)
        return _cache_entry(args, "needs_evidence", error="trace unavailable")

    monkeypatch.setattr(cli, "prepare_rollout", prepare)

    for _ in range(2):
        prepared = cli._prepare_dataset_item(
            args, {"problem_id": "task", "run_idx": 0}, "run", object(), "source-identity",
        )

    cached = cli._read_cache(cli._cache_path(cache, "run:task:0"))
    assert prepared["status"] == cached["status"] == "needs_evidence"
    # A cached failure is retried from the source instead of being trusted.
    assert len(calls) == 2


@pytest.mark.parametrize(
    "receipt",
    [{"verifier": {"read_failures": 1}}, {"raw_outcome": {"status": "error"}}],
    ids=["verifier-read-failure", "raw-outcome-error"],
)
def test_source_backed_preparation_refreshes_cached_evidence_read_failures(
    tmp_path, monkeypatch, receipt
) -> None:
    args = _judge_args("--cache", str(tmp_path / "cache"), "--prefetch-only")
    item = {"problem_id": "task", "run_idx": 0, "reward": 0.0}
    cached = _cache_entry(
        args, "prepared",
        source_sha256=cli._json_sha256({"source": "source-identity", "row": item}),
        source_identity_sha256="source-identity", evidence_receipt=receipt,
        incident={"source_provenance": {"reward": 0.0}},
    )
    cli._write_cache(cli._cache_path(tmp_path / "cache", "run:task:0"), cached)
    cached_rows, missing = cli._cached_evidence(args, ["run:task:0"], {"run": "source-identity"})
    assert cached_rows == []
    assert "run:task:0" in missing
    calls = []
    refreshed = {"verifier": {"read_failures": 0}, "raw_outcome": {"status": "included"}}

    def prepare(*call_args, **kwargs):
        calls.append(call_args)
        return {**cached, "evidence_receipt": refreshed}

    monkeypatch.setattr(cli, "prepare_rollout", prepare)

    prepared = cli._prepare_dataset_item(args, item, "run", object(), "source-identity")

    assert len(calls) == 1
    assert prepared["evidence_receipt"]["verifier"]["read_failures"] == 0


def test_local_outcome_source_reports_missing_manifest_source_path(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SourceDataError, match="missing source_path"):
        LocalOutcomeSource(tmp_path)


# --- judge configuration and preflight ---


def _stub_providers(monkeypatch, configured, *, http, agentic=None) -> list[tuple]:
    """Replace judge construction; the returned list records each backend a run builds."""
    built: list[tuple] = []

    def provider(args):
        built.append(("llm", args.judge_model))
        return http

    def from_config(cfg, *, max_turns):
        assert cfg is configured  # The agentic pass reuses the configured judge endpoint.
        built.append(("agentic", max_turns))
        return agentic

    monkeypatch.setattr("autofa.config.load_config", lambda: configured)
    monkeypatch.setattr(cli, "_check_judge", lambda provider: None)
    monkeypatch.setattr(cli, "_provider", provider)
    monkeypatch.setattr(agentic_review.AgenticClassificationProvider, "from_config", from_config)
    return built


def test_judge_model_targets_only_llm_passes(monkeypatch) -> None:
    built = _stub_providers(monkeypatch, cli_config(taxonomy_agentic_max_turns=19),
                            http=_FirstChoice(), agentic=object())

    _real_classifier(_judge_args("--judge-model", "m", "--second-pass", "off"))
    _real_classifier(_judge_args("--judge-model", "m", "--second-pass", "agentic"))
    _real_classifier(_judge_args("--judge-model", "m", "--first-pass", "agentic",
                                 "--second-pass", "llm"))

    assert built == [("llm", "m"), ("llm", "m"), ("agentic", 19), ("llm", "m"),
                     ("agentic", 19)]
    with pytest.raises(ValueError, match="llm passes only"):
        _real_classifier(_judge_args("--judge-model", "m",
                                     "--first-pass", "agentic", "--second-pass", "off"))
    with pytest.raises(ValueError, match="llm passes only"):
        _real_classifier(_judge_args("--judge-param", "reasoning_effort=high",
                                     "--first-pass", "agentic", "--second-pass", "off"))
    with pytest.raises(SystemExit):
        _judge_args("--judge-param", "reasoning_effort")


def test_agentic_pass_uses_judge_endpoint_and_turn_precedence(monkeypatch) -> None:
    configured = cli_config(taxonomy_second_pass="agentic", taxonomy_agentic_max_turns=19)
    built = _stub_providers(monkeypatch, configured, http=_FirstChoice(), agentic=object())

    _real_classifier(_judge_args())
    _real_classifier(_judge_args("--agentic-max-turns", "7"))
    _real_classifier(_judge_args("--agentic-max-turns", "2048"))
    _real_classifier(_judge_args("--agentic-max-turns", "none"))

    agentic_builds = [entry for entry in built if entry[0] == "agentic"]
    assert agentic_builds == [
        ("agentic", 19),
        ("agentic", 7),
        ("agentic", 2048),
        ("agentic", None),
    ]

    with pytest.raises(SystemExit):
        _judge_args("--agentic-max-turns", "0")


@pytest.mark.parametrize("second_pass", ["off", "llm"])
def test_agentic_first_pass_builds_http_provider_only_for_llm_review(
    monkeypatch, second_pass,
) -> None:
    configured = cli_config(taxonomy_first_pass="agentic", taxonomy_second_pass="llm")
    http, agentic = _FirstChoice(), object()
    built = _stub_providers(monkeypatch, configured, http=http, agentic=agentic)

    classifier = _real_classifier(_judge_args("--second-pass", second_pass))

    assert (classifier.first_pass, classifier.second_pass) == ("agentic", second_pass)
    assert classifier.agentic_provider is agentic
    llm = second_pass == "llm"
    assert classifier.provider is (http if llm else None)
    assert [entry for entry in built if entry[0] == "llm"] == ([("llm", None)] if llm else [])


def test_saved_agentic_first_pass_initializes_only_llm_review(monkeypatch) -> None:
    http = _FirstChoice()
    built = _stub_providers(monkeypatch, cli_config(), http=http)
    args = _judge_args("--second-pass", "llm")
    args._saved_first_pass_method = "agentic"

    classifier = _real_classifier(args)

    # The saved first pass is replayed, so its agentic provider is never initialized.
    assert built == [("llm", None)]

    assert (classifier.first_pass, classifier.saved_first_pass) == ("agentic", "agentic")
    assert classifier.provider is http
    assert classifier.agentic_provider is None


def test_no_pending_work_needs_no_llm_credentials(monkeypatch) -> None:
    key_env = "AUTOFA_TEST_MISSING_JUDGE_KEY"
    monkeypatch.delenv(key_env, raising=False)
    args = _judge_args()
    args._defer_judge_preflight = True
    provider = cli._provider(args, config=cli_config(
        judge_api_key_env=key_env, judge_base_url="https://judge.test/v1",
        judge_model="judge-model",
    ))
    classifier = SimpleNamespace(provider=provider, agentic_provider=None, first_pass="llm",
                                 second_pass="off")

    cli._preflight_classifier(classifier, methods=set())

    assert provider.api_key is None
    with pytest.raises(cli.ProviderError, match=key_env):
        provider.ensure_api_key()


def test_deferred_agentic_preflight_keeps_resolved_cli_path(monkeypatch) -> None:
    agentic = agentic_review.AgenticClassificationProvider(
        harness="claude", model="model", cli_path=None, effort=None, thinking=None,
        thinking_budget_tokens=None,
    )
    classifier = SimpleNamespace(provider=None, agentic_provider=agentic)
    monkeypatch.setattr(agentic_review, "_preflight_harness",
                        lambda harness, cli_path: "/usr/local/bin/claude")

    cli._preflight_classifier(classifier, methods={"agentic"})

    assert classifier.agentic_provider.cli_path == "/usr/local/bin/claude"


def test_judge_defaults_follow_the_configured_judge_endpoint(monkeypatch, home) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("STORED_JUDGE_KEY=stored\n")
    monkeypatch.delenv("STORED_JUDGE_KEY", raising=False)
    cfg = cli_config(judge_model="cfg-judge", judge_base_url="https://cfg.test/v1",
                     judge_api_key_env="STORED_JUDGE_KEY",
                     judge_request_options={"max_tokens": 5})
    monkeypatch.setattr("autofa.config.load_config", lambda: cfg)

    provider = cli._provider(_judge_args())

    assert (provider.model, provider.endpoint, provider.api_key) == (
        "cfg-judge", "https://cfg.test/v1", "stored",
    )
    assert provider.request_options == {"max_tokens": 5}


def test_missing_judge_key_points_to_the_stored_keys_file(monkeypatch) -> None:
    monkeypatch.delenv("UNSET_JUDGE_KEY", raising=False)
    cfg = cli_config(judge_base_url=JUDGE_URL, judge_api_key_env="UNSET_JUDGE_KEY")
    monkeypatch.setattr("autofa.config.load_config", lambda: cfg)

    with pytest.raises(cli.ProviderError, match=r"Save UNSET_JUDGE_KEY=<your-key> in .*\.env"):
        cli._provider(_judge_args())


@pytest.mark.parametrize(
    ("api", "effort_source"),
    [(None, "endpoint"), (None, "cli"), ("responses", "configured")],
)
def test_taxonomy_wire_defaults_ignore_discovery_policy_and_honor_judge_overrides(
    monkeypatch, api, effort_source,
):
    # A leaked discovery "responses" policy would visibly change the default API.
    settings = {
        "feature_api": "responses", "judge_api_key_env": "none",
        "judge_base_url": "https://judge.example/v1", "judge_model": "taxonomy-model",
        "feature_request_options": {"reasoning_effort": "none", "max_tokens": 1},
    }
    if effort_source != "endpoint":
        settings["judge_request_options"] = {"reasoning_effort": "medium", "max_tokens": 32}
    cfg = cli_config(**settings)
    monkeypatch.setattr("autofa.config.load_config", lambda: cfg)
    arguments = ["--judge-param", f"api={api}"] if api else []
    if effort_source == "cli":
        arguments += ["--judge-param", "reasoning_effort=high", "--judge-param", "max_tokens=64"]
    provider = cli._provider(_judge_args(*arguments))
    assert provider.api == (api or "chat_completions")
    body = json.dumps({"answers": {"kind": {
        "type": "choice", "choice": "a", "rationale": "Event 1 supports A.",
    }}})
    requests = []

    def respond(request):
        requests.append(request)
        message = {"role": "assistant", "content": body}
        if api == "responses":
            response = {"id": "resp_test", "model": "taxonomy-model", "created_at": 1,
                        "status": "completed", "output": [{
                            **message, "type": "message", "id": "message_test",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": body,
                                         "annotations": []}]}]}
        else:
            response = {"id": "chat_test", "model": "taxonomy-model", "created": 1,
                        "object": "chat.completion",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": message}]}
        return httpx.Response(200, json=response)

    with OpenAI(api_key="test", base_url="https://judge.example/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        monkeypatch.setattr("autofa.llm.hub.make_hub_client", lambda **kwargs: client)
        result = provider.evaluate("Event 1", {"kind": {
            "type": "choice", "criteria": {"a": "A"},
        }})
    assert result.answers["kind"]["choice"] == "a"
    [request] = requests
    assert request.url.path == ("/v1/responses" if api else "/v1/chat/completions")
    sent = json.loads(request.content)
    if effort_source == "endpoint":
        assert "reasoning" not in sent and "reasoning_effort" not in sent
        assert not {"max_tokens", "max_completion_tokens", "max_output_tokens"} & sent.keys()
    else:
        effort = "high" if effort_source == "cli" else "medium"
        budget = 64 if effort_source == "cli" else 32
        assert sent.get("reasoning") == ({"effort": effort} if api else None)
        assert sent.get("reasoning_effort") == (None if api else effort)
        assert sent["max_output_tokens" if api else "max_tokens"] == budget
    if api:
        assert "text" not in sent and "response_format" not in sent
    else:
        assert sent["response_format"] == {"type": "json_object"}


def test_rejected_judge_fails_before_any_download(monkeypatch, tmp_path, capsys) -> None:
    def rejected(self, state, questions):
        raise cli.ProviderError("auth failed (HTTP 403)")

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    cfg = cli_config(judge_base_url=JUDGE_URL, judge_api_key_env="OPENAI_API_KEY")
    monkeypatch.setattr("autofa.config.load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_classifier", _real_classifier)
    monkeypatch.setattr(cli.OpenAIProvider, "evaluate", rejected)
    monkeypatch.setattr(cli, "prepare_data", lambda *a, **k: pytest.fail("downloaded data"))

    assert cli.main(["--input", str(_gym_run(tmp_path, (0.0,))), "--all"]) == 2
    assert "auth failed" in capsys.readouterr().err


def test_judge_preflight_tolerates_slow_judges(monkeypatch) -> None:
    def slow(self, state, questions, *, images=None):
        raise cli.ProviderError("judge request failed") from TimeoutError()

    monkeypatch.setattr(cli.OpenAIProvider, "evaluate", slow)

    cli._check_judge(cli.OpenAIProvider(endpoint=JUDGE_URL, api_key="secret", model="judge"))


def test_judge_preflight_marks_rejected_image_input_as_text_only(monkeypatch) -> None:
    retries = []

    def evaluate(self, state, questions, *, images=None):
        retries.append(self.max_retries)
        if images:
            raise cli.ProviderError("image input is unsupported")
        return {}

    monkeypatch.setattr(cli.OpenAIProvider, "evaluate", evaluate)
    provider = cli.OpenAIProvider(endpoint=JUDGE_URL, api_key="secret", model="judge",
                                  max_retries=10)

    cli._check_judge(provider)

    assert provider.supports_images is False
    assert retries == [0, 0]  # The preflight probes never retry.


def test_judge_preflight_does_not_hide_transient_image_errors(monkeypatch) -> None:
    class RateLimited(RuntimeError):
        status_code = 429

    def evaluate(self, state, questions, *, images=None):
        if images:
            raise cli.ProviderError("image request was rate limited") from RateLimited()
        return {}

    monkeypatch.setattr(cli.OpenAIProvider, "evaluate", evaluate)
    provider = cli.OpenAIProvider(endpoint=JUDGE_URL, api_key="secret", model="judge")

    with pytest.raises(cli.ProviderError, match="image probe failed"):
        cli._check_judge(provider)

    assert provider.supports_images is True


# --- judge profiles recorded with results ---


def test_judge_profile_does_not_depend_on_image_probe_result() -> None:
    provider = cli.OpenAIProvider(endpoint=JUDGE_URL, api_key="secret", model="judge")
    before = cli.provider_configuration_sha256(provider)

    provider.supports_images = False

    assert cli.provider_configuration_sha256(provider) == before


def test_agentic_first_pass_profile_tracks_llm_reviewer_configuration() -> None:
    agentic = SimpleNamespace(model="agentic-first", configuration_sha256=lambda: "a" * 64)

    def profile(model):
        return cli._classification_profile(TaxonomyClassifier(
            load_taxonomy(None),
            cli.OpenAIProvider(endpoint=JUDGE_URL, api_key="secret", model=model),
            first_pass="agentic", second_pass="llm", agentic_provider=agentic,
        ))

    first, second = profile("reviewer-a"), profile("reviewer-b")
    assert first["configuration_sha256"] == second["configuration_sha256"] == "a" * 64
    assert first["second_pass_configuration_sha256"] != second["second_pass_configuration_sha256"]


@pytest.mark.parametrize(
    ("harness", "prior_tool_limit"),
    [("codex", 128), ("claude", None), ("claude", 128)],
)
def test_resume_accepts_legacy_completed_results_from_a_different_tool_limit(
    harness, prior_tool_limit
) -> None:
    agentic = agentic_review.AgenticClassificationProvider(
        harness=harness,
        model="agentic-first",
        cli_path=None,
        effort="high",
        thinking=None,
        thinking_budget_tokens=None,
        max_turns=None,
    )
    classifier = TaxonomyClassifier(
        load_taxonomy(None),
        None,
        first_pass="agentic",
        second_pass="off",
        agentic_provider=agentic,
    )
    profile = cli._classification_profile(classifier)
    prior = {
        **agentic.configuration(),
        "configured_max_turns": 128,
        "tool_call_limit": prior_tool_limit,
    }
    prior.pop("tool_budget_semantics")
    prior_sha256 = hashlib.sha256(
        json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record = {
        "taxonomy_sha256": profile["taxonomy_sha256"],
        "protocol_version": profile["protocol_version"],
        "classification_method": "agentic_taxonomy_classification",
        "first_pass": {"method": "agentic"},
        "second_pass": {"method": "off", "status": "skipped"},
        "judge": {
            "requested_model": agentic.model,
            "configuration_sha256": prior_sha256,
            "configured_max_turns": 128,
        },
    }

    assert cli._matches_classification_profile(record, profile)
    current_semantics = {
        **prior,
        "tool_budget_semantics": agentic_review.AGENTIC_TOOL_BUDGET_SEMANTICS,
    }
    record["judge"]["configuration_sha256"] = hashlib.sha256(
        json.dumps(current_semantics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert not cli._matches_classification_profile(record, profile)
    record["judge"]["configuration_sha256"] = prior_sha256
    record["judge"]["configured_max_turns"] = 64
    assert not cli._matches_classification_profile(record, profile)


def test_taxonomy_resume_rejects_profile_from_before_multimodal_judging(tmp_path, monkeypatch):
    cfg = cli_config(
        feature_api="decisions", feature_request_options={"reasoning_effort": "none"},
        judge_base_url="https://judge.example/v1", judge_api_key_env="none",
        judge_model="legacy-taxonomy-model",
    )
    monkeypatch.setattr("autofa.config.load_config", lambda: cfg)
    monkeypatch.setattr("autofa.llm.capabilities.TRANSPORT_CONTRACT_VERSION", 999)
    classifier = TaxonomyClassifier(load_taxonomy(None), cli._provider(_judge_args()))
    # Recorded by main's provider for this endpoint/model, omitted API/effort,
    # and empty request options. Do not derive this fixture with the new hash.
    historical_hash = "fadc326be5a718d71a81b9f0d0f6c0a0eb0959de703e1c40ac23935ecc73af42"
    profile = cli._classification_profile(classifier)
    assert profile["configuration_sha256"] != historical_hash
    destination = _write_lines(tmp_path / "results.jsonl", {
        "kind": "taxonomy_classification", "status": "ok",
        "protocol_version": profile["protocol_version"],
        "taxonomy_sha256": profile["taxonomy_sha256"],
        "judge": {"requested_model": "legacy-taxonomy-model",
                  "configuration_sha256": historical_hash},
        "second_pass": {"method": profile["second_pass"], "status": "completed"},
        "source": {"rollout_id": "run:task:0"},
    })

    with pytest.raises(ValueError):
        cli._validate_output_profile(destination, classifier, resume=True)
    assert cli._resume_rollout_ids(_judge_args("--resume"), destination, classifier) == set()


def test_saved_agentic_replay_records_the_active_llm_judge(tmp_path) -> None:
    provider = cli.OpenAIProvider(api_key="secret", model="review-model",
                                  endpoint="https://judge.test/v1", api="responses")
    classifier = TaxonomyClassifier(load_taxonomy(None), provider, first_pass="agentic",
                                    second_pass="llm", saved_first_pass="agentic")

    record = _append_reviewed(
        tmp_path / "review.jsonl", "responses", classifier,
        _judge_args("--judge-param", "reasoning_effort=high"),
        first_pass={"method": "agentic"}, second_pass={"method": "llm", "status": "completed"},
    )

    profile = cli._classification_profile(classifier)
    assert record["first_pass"]["method"] == "agentic"
    judge = record["judge"]
    assert (judge["provider"], judge["requested_model"]) == ("responses", "review-model")
    assert judge["configuration_sha256"] == profile["configuration_sha256"]
    assert judge["second_pass_requested_model"] == "review-model"
    assert (judge["second_pass_configuration_sha256"]
            == profile["second_pass_configuration_sha256"])


def test_saved_llm_replay_records_the_active_agentic_judge(tmp_path) -> None:
    classifier = TaxonomyClassifier(
        load_taxonomy(None), None, first_pass="llm", second_pass="agentic",
        agentic_provider=_agentic_profile("agentic-review-model", "a" * 64),
        saved_first_pass="llm",
    )

    record = _append_reviewed(
        tmp_path / "review.jsonl", "responses", classifier,
        first_pass={"method": "llm"}, second_pass={"method": "agentic", "status": "completed"},
    )

    assert record["first_pass"]["method"] == "llm"
    assert record["judge"]["provider"] == "agentic"
    assert record["judge"]["requested_model"] == "agentic-review-model"
    assert record["judge"]["configuration_sha256"] == "a" * 64


def test_fresh_llm_agentic_workflow_persists_review_configuration(tmp_path) -> None:
    classifier = TaxonomyClassifier(
        load_taxonomy(None),
        cli.OpenAIProvider(api_key="secret", model="first-model",
                           endpoint="https://judge.test/v1"),
        second_pass="agentic", agentic_provider=_agentic_profile("review-model", "b" * 64),
    )
    completed = {"method": "agentic", "status": "completed",
                 "receipts": {"harness": "codex", "model": "review-model", "max_turns": 12}}
    configurations = []
    for status, model, second_pass in (
        ("review_failed", "first-model", {"method": "agentic", "status": "failed"}),
        ("ok", "review-model", completed),
    ):
        record = _append_reviewed(
            tmp_path / f"{status}.jsonl", "chat", classifier, status=status, model=model,
            selected_code="OWN-IF", first_pass={"method": "llm", "status": "completed"},
            second_pass=second_pass,
        )
        assert record["judge"]["provider"] == "chat"
        assert record["judge"]["harness"] == "codex"
        assert record["judge"]["configured_max_turns"] == 12
        configurations.append(cli.report._judge_configuration(record))

    # A failed and a completed review group under the same judge configuration.
    assert configurations[0] == configurations[1]
