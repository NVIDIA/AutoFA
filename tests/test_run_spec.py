# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run configs parse, preflight and launch workflows under a frozen, verified setup contract."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autofa.config import Config, RunConfig, load_config
from autofa.data.reference_data import parse_evalrdl_run_url
from autofa.run_config import AutoFAConfig, AutoFAExecution, SourceSpec
from autofa.run_spec import (
    _materialize_source,
    _require_dataset_labels,
    apply_run_spec,
    load_autofa_config,
    load_run_spec,
    preflight_autofa_config,
)
from autofa.setup_contract import (
    inspect_config,
    monetary_readiness,
    prepare_config,
    required_capabilities,
    validate_launch_contract,
)
from autofa.taxonomy.workflow_config import (
    TaxonomyWorkflowSpec,
    parse_taxonomy_workflow,
)

RUN_ID = "0123456789abcdef"
INSPECT_URL = f"https://eval.example.org/#/run/{RUN_ID}"
EXPORT_URL = f"https://eval.example.org/v1/runs/{RUN_ID}/export/autofa"
REMOTE_SPEC = f"version: 2\nsource:\n  evalrdl_url: {INSPECT_URL}\n"
LOCAL_SPEC = "version: 2\nsource:\n  local_folder: local\n"


def make_autofa_config(settings: Config, *, execution: AutoFAExecution | None = None):
    return AutoFAConfig(settings=settings, source=SourceSpec(evalrdl_url=INSPECT_URL),
                        execution=execution or AutoFAExecution(mode="session"))


def write_spec(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def write_parquet(path: Path) -> Path:
    pq.write_table(pa.Table.from_pylist([{
        "model_id": "model-a", "benchmark": "bench-a", "problem_id": "task-1", "run_idx": 0,
        "is_correct": False, "prompt_text": "prompt", "response_text": "response",
    }]), path)
    return path


def taxonomy_root(*, source=None, endpoint=True, **run):
    """A one-run classification config; keyword arguments replace fields of that run."""
    taxonomy = {"runs": [{
        "first_pass": "llm", "second_pass": "off",
        "judges": [{"name": "llm", "model": "model-a", "workers": 1, "output": "results.jsonl"}],
        "report": {"output": "report.html"}, **run,
    }]}
    if endpoint:
        taxonomy.update(judge_base_url="https://judge.example/v1",
                        judge_api_key_env="JUDGE_API_KEY")
    return {"version": 2, "workflow": "classification",
            "source": source or {"api_runs": ["run-a"]},
            "selection": {"per_run": 1, "sample": "first", "cache": "evidence"},
            "taxonomy": taxonomy}


def write_taxonomy(tmp_path, root):
    return write_spec(tmp_path / "taxonomy.yaml", yaml.safe_dump(root))


def test_shared_run_config_supports_taxonomy_workflows(tmp_path):
    from autofa.taxonomy import cli as taxonomy_cli

    root = taxonomy_root(
        source={"api_runs": ["run-a", "run-b"]}, second_pass="agentic",
        judges=[{"name": "llm", "model": "model-a", "workers": 1, "output": "llm.jsonl"},
                {"name": "agentic", "model": "model-b", "harness": "codex", "workers": 1,
                 "output": "agentic.jsonl"}],
        report={"output": "report.html", "publish": "taxonomy-e2e"},
    )
    root["selection"].update(per_run=2, sample="random", seed=42)
    spec = load_run_spec(write_taxonomy(tmp_path, root))
    assert isinstance(spec, TaxonomyWorkflowSpec)
    assert spec.source.api_runs == ("run-a", "run-b")
    assert spec.selection.cache == (tmp_path / "evidence").resolve()
    assert (spec.runs[0].first_pass, spec.runs[0].second_pass) == ("llm", "agentic")
    assert spec.runs[0].judges[1].harness == "codex"
    assert spec.runs[0].report.publish == "taxonomy-e2e"

    snapshot = taxonomy_cli._workflow_config_snapshot(spec, spec.runs[0])
    assert "run" not in snapshot["taxonomy"]
    assert parse_taxonomy_workflow(snapshot, tmp_path) == spec


@pytest.mark.parametrize("section,changes,message", [
    ("selection", {"stale_selection_id": "old"}, "selection has unknown keys: stale_selection_id"),
    ("report", {"output": "results.jsonl"}, "output paths must be unique"),
])
def test_taxonomy_workflow_rejects_unknown_keys_and_colliding_outputs(
    tmp_path, section, changes, message,
):
    root = taxonomy_root()
    target = root["selection"] if section == "selection" else root["taxonomy"]["runs"][0]["report"]
    target.update(changes)
    with pytest.raises(ValueError, match=message):
        parse_taxonomy_workflow(root, tmp_path)


@pytest.mark.parametrize("first_pass,judge,message", [
    ("agentic", {"model": "http-only-model"}, "model applies to llm passes only"),
    ("llm", {"harness": "codex"}, "harness applies to agentic passes only"),
])
def test_taxonomy_judge_overrides_require_a_matching_pass(tmp_path, first_pass, judge, message):
    path = write_taxonomy(tmp_path, taxonomy_root(endpoint=False, first_pass=first_pass, judges=[
        {"name": "judge", "workers": 1, "output": "results.jsonl", **judge},
    ]))
    with pytest.raises(ValueError, match=message):
        load_run_spec(path)


@pytest.mark.parametrize("source", [{"local_folder": "local-run"}, {"evalrdl_url": INSPECT_URL}])
def test_taxonomy_workflow_accepts_discovery_sources(tmp_path, source):
    spec = load_run_spec(write_taxonomy(tmp_path, taxonomy_root(source=source)))
    if "local_folder" in source:
        assert spec.source.local_folder == (tmp_path / "local-run").resolve()
        assert spec.source.api_runs == ()
    else:
        assert spec.source.evalrdl_url == INSPECT_URL
        assert spec.source.api_runs == (RUN_ID,)
        assert spec.source.api_base_url == "https://eval.example.org"


def test_run_command_dispatches_taxonomy_workflow_with_configured_judge_defaults(
    tmp_path, monkeypatch,
):
    from autofa import cli

    path = write_taxonomy(tmp_path, taxonomy_root(
        endpoint=False, first_pass="agentic", second_pass="llm",
        judges=[{"name": "configured-defaults", "workers": 1, "output": "results.jsonl"}],
    ))
    spec = load_run_spec(path)
    # Omitted judge settings stay unset, so the configured defaults apply at run time.
    assert spec.judge_base_url is None and spec.judge_api_key_env is None
    assert spec.runs[0].judges[0].model is None
    calls = []
    monkeypatch.setattr("autofa.run_spec.load_autofa_config", lambda *args: spec)
    monkeypatch.setattr(
        "autofa.taxonomy.cli.run_taxonomy_workflow",
        lambda workflow: (calls.append(workflow), 0)[1],
    )
    cli._run_config(SimpleNamespace(run_config_path=path))
    assert calls == [spec]


def test_classification_config_supplies_later_handoff_results(tmp_path):
    from autofa import cli

    names = ("first", "second")
    path = write_taxonomy(tmp_path, taxonomy_root(judges=[
        {"name": name, "model": "model-a", "workers": 1, "output": f"{name}.jsonl"}
        for name in names
    ]))
    assert cli._taxonomy_handoff_inputs(path) == [
        str((tmp_path / f"{name}.jsonl").resolve()) for name in names
    ]


@pytest.mark.parametrize(("body", "message"), [
    (REMOTE_SPEC.replace("version: 2", "version: 1"), "Unsupported run config version"),
    (REMOTE_SPEC + "  surprise: true\n", "source has unknown keys"),
    ("version: 2\nsource:\n  evalrdl_url: https://example.test/not-a-run\n", "EvalRDL URL"),
    (REMOTE_SPEC + "config:\n  max_rounds: 0\n", "config.max_rounds must be greater than zero"),
    (REMOTE_SPEC + "  local_folder: local\n", "source must define exactly one"),
    (REMOTE_SPEC + "config:\n  optimization: unknown\n", "config.optimization must be one of"),
])
def test_load_run_spec_rejects_invalid_input(tmp_path, body, message):
    path = write_spec(tmp_path / "run.yaml", body)
    with pytest.raises(ValueError, match=message):
        load_run_spec(path)


@pytest.mark.parametrize("url", [INSPECT_URL, EXPORT_URL])
def test_evalrdl_url_forms_normalize(url):
    base_url, run_id, export_url = parse_evalrdl_run_url(url)
    assert base_url == "https://eval.example.org"
    assert run_id == RUN_ID
    assert export_url == EXPORT_URL


def test_apply_run_spec_overrides_harness(tmp_path, monkeypatch):
    path = write_spec(tmp_path / "run.yaml", REMOTE_SPEC + """config:
  harness: codex
  model: gpt-6.1-sol
  effort: high
  analysis_focus: both
""")
    monkeypatch.setattr("autofa.session.guidance.snapshot_guidance", lambda config: None)
    config = apply_run_spec(Config(harness="claude", model="claude-opus-5-5", effort="medium"),
                            load_run_spec(path))
    assert config.harness == "codex"
    assert config.model == "gpt-6.1-sol"
    assert config.effort == "high"
    assert config.analysis_focus == "both"


@pytest.mark.parametrize("optimization", [None, "recall"])
def test_batch_preset_overrides_inherited_balanced_only_when_selected(
    tmp_path, monkeypatch, optimization,
):
    from autofa.objectives import apply_preset, preset_name, snapshot_objectives

    inherited = apply_preset(Config(feature_model="detector", max_rounds=7), "balanced")
    before = snapshot_objectives(inherited)
    choice = f"config:\n  optimization: {optimization}\n" if optimization else ""
    path = write_spec(tmp_path / "run.yaml", "version: 2\nsource:\n  local_folder: .\n" + choice)
    monkeypatch.setattr("autofa.session.guidance.snapshot_guidance", lambda config: {})

    effective = load_autofa_config(path, inherited).settings

    assert preset_name(effective) == (optimization or "balanced").capitalize()
    assert effective.feature_model == "detector"
    assert effective.max_rounds == 7
    assert snapshot_objectives(inherited) == before


def test_load_autofa_config_builds_resolved_batch_config(tmp_path, monkeypatch):
    path = write_spec(tmp_path / "run.yaml", REMOTE_SPEC + """config:
  min_failure_modes: 2
  max_rounds: 4
  top_x: 7
  default_eval_limit: 100
execution:
  publish: true
  require_llm_detectors: true
""")
    monkeypatch.setattr("autofa.session.guidance.snapshot_guidance", lambda config: None)
    config = load_autofa_config(path, Config(analysis_outputs=["efficiency"]))
    assert config.settings.analysis_outputs == ["agent", "efficiency"]
    assert config.source.evalrdl_url == INSPECT_URL
    assert config.settings.max_rounds == 4
    assert config.settings.top_x == 7
    assert config.settings.min_failure_modes == 2
    assert config.settings.default_eval_limit == 100
    assert config.execution.mode == "batch"
    assert config.execution.publish is True
    assert config.execution.require_llm_detectors is True


def test_resume_restores_persisted_harness_settings(tmp_path, monkeypatch):
    from autofa import cli
    from autofa.config import snapshot_harness_settings

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    configured = Config(
        harness="codex", model="custom-deployment", effort="high",
        behavior_judge_harness="auto", behavior_judge_model="gpt-6.1-sol",
        behavior_judge_effort="medium",
    )
    run = RunConfig(run_id="batch-run", parquet_paths=[], model_id="model-a",
                    harness=configured.harness, analysis_outputs=["agent"],
                    harness_settings=snapshot_harness_settings(configured))
    run.save()
    resumed = []
    monkeypatch.setattr(cli, "load_config",
                        lambda: Config(harness="claude", model="claude-opus-5-5", effort="low"))
    monkeypatch.setattr("autofa.session.research_status.read_research_status", lambda path: {})
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda selected, **kwargs: resumed.append((selected, kwargs)))
    cli._resume(SimpleNamespace(run_id=run.run_id, min_failure_modes=None))
    resume_config = resumed[0][1]["cfg"]
    assert resume_config.harness == "codex"
    assert resume_config.model == "custom-deployment"
    assert resume_config.effort == "high"
    assert resume_config.behavior_judge_harness == "codex"
    assert resume_config.behavior_judge_model == "gpt-6.1-sol"
    assert resume_config.behavior_judge_effort == "medium"


def test_preflight_checks_effective_discovery_and_judge_models(monkeypatch):
    loaded, checked = [], []

    def load(harness, cli_path):
        loaded.append((harness, cli_path))
        return harness

    monkeypatch.setattr("autofa.run_spec._harness_catalog", load)
    monkeypatch.setattr("autofa.run_spec._require_model", lambda *args: checked.append(args))
    preflight_autofa_config(make_autofa_config(Config(
        harness="codex", model="gpt-6.1-sol", effort="high", codex_cli="/test/codex",
        behavior_judge_harness="auto", behavior_judge_model="gpt-6.1-sol",
    )))
    # Both roles share one catalog query for the same harness binary.
    assert loaded == [("codex", "/test/codex")]
    assert checked == [
        ("codex", "codex", "gpt-6.1-sol", "high", "discovery"),
        ("codex", "codex", "gpt-6.1-sol", "medium", "behavior judge"),
    ]
    checked.clear()
    # A model from another harness falls back to the effective harness default.
    preflight_autofa_config(make_autofa_config(Config(
        harness="claude", model="gpt-6-astra", behavior_judge_harness="auto",
        behavior_judge_model="gpt-6.1-sol",
    )))
    assert checked == [
        ("claude", "claude", "claude-opus-5-5", "default", "discovery"),
        ("claude", "claude", "claude-sonnet-5-5", "medium", "behavior judge"),
    ]


def test_harness_catalog_errors_are_actionable(monkeypatch):
    from autofa.run_spec import _harness_catalog

    monkeypatch.setattr("autofa.harness.model_catalog._find_codex_cli",
                        lambda path: (_ for _ in ()).throw(FileNotFoundError("missing")))
    with pytest.raises(RuntimeError, match="Install Codex.*AUTOFA_CODEX_CLI"):
        _harness_catalog("codex", None)
    monkeypatch.setattr("autofa.harness.model_catalog._find_codex_cli", lambda path: "/test/codex")
    monkeypatch.setattr("autofa.harness.settings._authentication_status",
                        lambda binary, harness: "not signed in")
    with pytest.raises(RuntimeError, match="not signed in.*codex login"):
        _harness_catalog("codex", None)


def test_preflight_allows_custom_models_but_rejects_unavailable_effort():
    from autofa.harness.model_catalog import HarnessModel, ModelCatalog
    from autofa.run_spec import _require_model

    catalog = ModelCatalog(models=(HarnessModel(
        value="gpt-6.1-sol", display_name="GPT-6.1 Sol", effort_levels=("low", "medium", "high"),
    ),))
    _require_model(catalog, "codex", "custom-deployment", "high", "discovery")
    _require_model(ModelCatalog(error="catalog unavailable"), "codex", "custom-deployment", "high",
                   "discovery")
    with pytest.raises(RuntimeError, match="ultra.*Available efforts: low, medium, high"):
        _require_model(catalog, "codex", "gpt-6.1-sol", "ultra", "discovery")


def test_publication_preflight_has_actionable_error(monkeypatch):
    from autofa.run_spec import _require_publication

    monkeypatch.setattr("autofa.data.remote_library_api.client_from_config",
                        lambda config: (_ for _ in ()).throw(RuntimeError("missing token")))
    with pytest.raises(RuntimeError, match="execution.publish to false.*missing token"):
        _require_publication(Config())


def test_local_run_keeps_local_reference(tmp_path, monkeypatch):
    folder = tmp_path / "local"
    folder.mkdir()
    (folder / "marker").write_text("present", encoding="utf-8")
    parquet = write_parquet(tmp_path / "dataset.parquet")
    path = write_spec(tmp_path / "run.yaml", LOCAL_SPEC)
    calls = []
    source = SimpleNamespace(kind="evalrdl", path=folder)
    monkeypatch.setattr("autofa.data.sources.discover_eval_sources", lambda selected: [source])
    monkeypatch.setattr("autofa.data.cohort.select_cohort",
                        lambda **kwargs: calls.append(("select", kwargs)) or object())
    reference = {"kind": "local_evalrdl", "store_id": "a" * 64, "run_id": "local-run",
                 "source_harness": "nemo-gym", "parquet_path": str(parquet.resolve())}
    monkeypatch.setattr("autofa.data.cohort.prepare_data", lambda cohort, **kwargs: SimpleNamespace(
        parquet_paths=[parquet], source_paths=[str(folder)], source_kind="evalrdl",
        reference_data=[reference],
    ))
    selection = _materialize_source(load_run_spec(path).source, Config())
    assert selection.parquet_paths == [parquet]
    assert selection.source_paths == [str(folder)]
    assert selection.source_kind == "evalrdl"
    assert selection.reference_data == [reference]
    assert calls[0][1]["folders"] == [folder]


def test_local_run_keeps_plain_parquet_compatibility(tmp_path):
    folder = tmp_path / "local"
    folder.mkdir()
    parquet = write_parquet(folder / "dataset.parquet")
    selection = _materialize_source(
        load_run_spec(write_spec(tmp_path / "run.yaml", LOCAL_SPEC)).source, Config(),
    )
    assert selection.parquet_paths == [parquet]
    assert selection.source_paths == [str(folder)]
    assert selection.source_kind == "parquet"


# Only an efficiency-only analysis may skip the full traces that failure analysis needs.
@pytest.mark.parametrize("outputs,efficiency_only", [
    (["efficiency"], True), (["efficiency", "agent"], False),
], ids=["efficiency_only", "with_failure_analysis"])
def test_remote_source_downloads_evalrdl_run(tmp_path, monkeypatch, outputs, efficiency_only):
    parquet = write_parquet(tmp_path / "dataset.parquet")
    path = write_spec(tmp_path / "run.yaml", REMOTE_SPEC)
    calls = []
    api = SimpleNamespace(base_url="https://eval.example.org")
    monkeypatch.setattr("autofa.data.remote_api.DataApiClient",
                        lambda base_url: calls.append(("api", base_url)) or api)
    cohort = object()
    monkeypatch.setattr("autofa.data.cohort.select_cohort",
                        lambda **kwargs: calls.append(("select", kwargs)) or cohort)
    prepared = SimpleNamespace(source_kind="remote_run", parquet_paths=[parquet])
    monkeypatch.setattr(
        "autofa.data.cohort.prepare_data",
        lambda selected, **kwargs: calls.append(("prepare", selected, kwargs)) or prepared,
    )
    selection = _materialize_source(load_run_spec(path).source, Config(analysis_outputs=outputs))
    assert calls == [
        ("api", "https://eval.example.org"),
        ("select", {"api_runs": [RUN_ID], "api": api}),
        ("prepare", cohort, {"api": api, "refresh": False, "efficiency_only": efficiency_only}),
    ]
    assert selection.source_kind == "remote_run"


def test_missing_local_source_fails_without_remote_fallback(tmp_path):
    path = write_spec(tmp_path / "run.yaml", LOCAL_SPEC)
    with pytest.raises(ValueError, match="source.local_folder is not a directory"):
        _materialize_source(load_run_spec(path).source, Config())


@pytest.mark.parametrize(("row", "message"), [
    ({"benchmark": "bench-a"}, "no model_id"), ({"model_id": "model-a"}, "no benchmark"),
])
def test_batch_source_requires_dataset_labels(tmp_path, row, message):
    parquet = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pylist([
        {**row, "problem_id": "task-1", "run_idx": 0, "is_correct": False},
    ]), parquet)
    with pytest.raises(ValueError, match=message):
        _require_dataset_labels(SimpleNamespace(parquet_paths=[parquet]))


def test_write_run_report_renders_saved_results_without_evaluating(tmp_path, monkeypatch):
    from autofa.core.library import FeatureFunction, FunctionLibrary
    from autofa.mcp import tools_eval
    from autofa.reports import failure_cards
    from autofa.session import runner

    library = FunctionLibrary(functions=[FeatureFunction(
        name="saved", description="Saved claim", source="def detect(item): return True",
        scores={"evaluated": True, "sampled": True},
    )])
    run = SimpleNamespace(run_dir=lambda: tmp_path, source_harness="codex", reference_data=[])
    events = []
    dataset = object()
    runtime = SimpleNamespace(
        library=library, dataset=dataset,
        persist_library=lambda: events.append("persist"),
    )
    monkeypatch.setattr(runner, "load_runtime_from_run", lambda *args: runtime)
    monkeypatch.setattr("autofa.data.library_provenance.save_catalog_library",
                        lambda *args: events.append("catalog"))
    for name in ("evaluate_functions", "final_evaluate_functions", "judge_function"):
        monkeypatch.setattr(tools_eval, name, lambda *args, **kwargs: pytest.fail(
            "Rendering saved results must not launch detector evaluation or judge work",
        ))

    def render(selected_library, *, out, **kwargs):
        assert selected_library is library and kwargs["dataset"] is dataset
        assert selected_library.functions[0].scores == {"evaluated": True, "sampled": True}
        events.append("render")
        return out

    monkeypatch.setattr(failure_cards, "write_failure_cards_with_status", render)
    assert runner.write_run_report(
        run, cfg=Config(library_root=str(tmp_path / "catalog")),
    ) == tmp_path / "failure_cards.html"
    assert events == ["persist", "catalog", "render"]


def batch_cli(monkeypatch, *, publish=False):
    """Stub the stages of a batch ``autofa run --config`` and record their order."""
    from autofa import cli

    settings = Config(analysis_outputs=["agent"])
    config = make_autofa_config(settings, execution=AutoFAExecution(mode="batch", publish=publish))
    run = SimpleNamespace(run_id="run-1", save=lambda: None)
    events = []
    monkeypatch.setattr("autofa.run_spec.load_autofa_config", lambda path, loaded: config)
    monkeypatch.setattr("autofa.run_spec.preflight_autofa_config",
                        lambda selected: events.append("preflight"))
    monkeypatch.setattr("autofa.run_spec.materialize_autofa_config",
                        lambda selected: events.append("source") or SimpleNamespace())
    monkeypatch.setattr(cli, "load_config", Config)
    monkeypatch.setattr(cli, "_create_analysis_run",
                        lambda *args, **kwargs: events.append(("create", kwargs)) or run)
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda *args, **kwargs: events.append(("session", kwargs)))
    monkeypatch.setattr("autofa.session.runner.require_batch_target",
                        lambda *args, **kwargs: events.append(("target", args, kwargs)))
    monkeypatch.setattr("autofa.session.runner.write_run_report",
                        lambda *args, **kwargs: events.append("report") or Path("report.html"))
    args = SimpleNamespace(run_config_path=Path("run.yaml"))
    return SimpleNamespace(settings=settings, run=run, events=events,
                           invoke=lambda: cli._run_config(args))


def test_config_command_runs_batch_and_writes_report(monkeypatch):
    batch = batch_cli(monkeypatch)
    batch.invoke()
    assert batch.events == [
        "preflight", "source", ("create", {"show_guidance": True}),
        ("session", {"cfg": batch.settings, "auto_discover": True, "continue_mode": False,
                     "exit_after_discovery": True, "publish_on_exit": False}),
        ("target", (batch.run,), {"cfg": batch.settings}), "report",
    ]


def test_config_command_fails_when_required_publish_fails(monkeypatch):
    batch = batch_cli(monkeypatch, publish=True)
    monkeypatch.setattr(
        "autofa.data.remote_library_publish.maybe_publish_run", lambda *args, **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="Required batch publication failed"):
        batch.invoke()


@pytest.mark.parametrize("stage", ["run_session", "require_batch_target"])
def test_cancelled_or_unmet_batch_never_writes_report(monkeypatch, stage):
    from autofa.harness.base import BatchSessionCancelled

    batch = batch_cli(monkeypatch)
    cancelled = stage == "run_session"
    error = BatchSessionCancelled("cancelled") if cancelled else RuntimeError("target unmet")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(f"autofa.session.runner.{stage}", fail)
    # Cancellation becomes an interrupt; an unmet target keeps its own error.
    with pytest.raises(KeyboardInterrupt if cancelled else RuntimeError,
                       match=None if cancelled else "target unmet"):
        batch.invoke()
    assert "report" not in batch.events


@pytest.mark.parametrize(
    ("delivery", "expected_status"),
    [({"status": None}, "unknown"),
     ({"status": "publication_uncertain"}, "publication_uncertain")],
)
def test_managed_batch_preserves_delivery_failure_message(
    tmp_path, monkeypatch, delivery, expected_status,
):
    from autofa import cli

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AUTOFA_SETUP_CONTRACT", str(tmp_path / "contract.json"))
    monkeypatch.delenv("AUTOFA_DRIVER_LAUNCH", raising=False)
    run = RunConfig(
        run_id="delivery-failure", parquet_paths=[], model_id="model", analysis_outputs=["agent"],
    )
    config = make_autofa_config(
        Config(analysis_outputs=["agent"]),
        execution=AutoFAExecution(mode="batch", publish=True),
    )
    monkeypatch.setattr("autofa.setup_contract.validate_launch_contract", lambda *a, **kw: None)
    monkeypatch.setattr("autofa.run_spec.preflight_autofa_config", lambda *a, **kw: None)
    monkeypatch.setattr(
        "autofa.run_spec.materialize_autofa_config", lambda *a, **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(cli, "_create_analysis_run", lambda *a, **kw: run)
    monkeypatch.setattr("autofa.session.runner.run_session", lambda *a, **kw: None)
    monkeypatch.setattr("autofa.session.runner.require_batch_target", lambda *a, **kw: None)
    monkeypatch.setattr(
        "autofa.session.runner.write_run_report", lambda *a, **kw: tmp_path / "report.html",
    )
    monkeypatch.setattr(
        "autofa.data.discovery_delivery.saved_only_delivery", lambda *a, **kw: delivery,
    )

    with pytest.raises(RuntimeError, match=f"Required cloud delivery remains {expected_status}"):
        cli.run_autofa(config)


def test_unified_launch_preserves_cohort_and_recovery_settings(tmp_path, monkeypatch):
    from autofa import cli
    from autofa.data.preparation import PreparedData

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    settings = Config(
        analysis_outputs=["agent"], max_rounds=8, min_failure_modes=3, default_eval_limit=120,
        discovery_terminate_if_stalled=False, discovery_stall_rounds=7,
        behavior_strictness="causal",
    )
    selection = PreparedData(
        parquet_paths=[], model_id="model", benchmarks=["bench"],
        source_kind="selection", selection_id="selected-taxonomy-cohort",
    )
    configured = AutoFAConfig(settings=settings, source=selection,
                              execution=AutoFAExecution(mode="session"))
    checked, launched = [], []
    monkeypatch.setattr("autofa.run_spec.preflight_autofa_config",
                        lambda selected, **kw: checked.append(selected))
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning", lambda *a, **kw: None)
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda run, **kw: launched.append((run, kw["cfg"])))
    monkeypatch.setattr("autofa.session.runner.require_batch_target", lambda *a, **kw: None)
    monkeypatch.setattr("autofa.session.runner.write_run_report",
                        lambda run, **kw: run.run_dir() / "failure_cards.html")

    result = cli.run_autofa(configured)

    assert len(checked) == len(launched) == 1
    assert checked[0].source is selection
    assert checked[0].execution is configured.execution
    saved = RunConfig.load(result.run_id)
    assert (saved.selection_id, saved.max_rounds, saved.min_failure_modes, saved.eval_limit) == (
        selection.selection_id, 8, 3, 120)
    assert (saved.discovery_terminate_if_stalled, saved.discovery_stall_rounds) == (False, 7)
    assert saved.behavior_strictness == "causal"
    assert launched[0][0] is result
    assert launched[0][1] is checked[0].settings


def test_batch_run_uses_current_objectives_without_rewriting_historical_settings(
    tmp_path, monkeypatch,
):
    from autofa import cli
    from autofa.objectives import (
        OBJECTIVE_LABELS,
        config_for_run,
        preset_name,
        snapshot_objectives,
    )

    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "home"))
    historical = Config(objective_axes=["ppv_at_pi0", "recall", "behavior_purity",
                                        "behavior_specificity"],
                        objective_weights=[25] * 4)
    original = snapshot_objectives(historical)
    path = write_spec(
        tmp_path / "run.yaml",
        "version: 2\nsource:\n  local_folder: .\nconfig:\n  min_failure_modes: 0\n"
        "  max_rounds: 1\n",
    )
    monkeypatch.setattr(cli, "load_config", lambda: historical)
    monkeypatch.setattr("autofa.run_spec.preflight_autofa_config", lambda *a, **kw: None)
    monkeypatch.setattr("autofa.run_spec.materialize_autofa_config", lambda *a: SimpleNamespace(
        parquet_paths=[], model_id="model", benchmarks=["bench"], source_paths=[],
        source_kind="parquet", source_harness="unknown", reference_data=[],
    ))
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning", lambda *a, **kw: None)
    launched = []
    monkeypatch.setattr("autofa.session.runner.run_session",
                        lambda run, **kw: launched.append((run, kw["cfg"])))
    monkeypatch.setattr("autofa.session.runner.require_batch_target", lambda *a, **kw: None)
    monkeypatch.setattr("autofa.session.runner.write_run_report",
                        lambda run, **kw: run.run_dir() / "failure_cards.html")

    cli._run_config(SimpleNamespace(run_config_path=path))

    run, effective = launched[0]
    saved = RunConfig.load(run.run_id)
    # A legacy weighted configuration launches with the current default preset.
    assert preset_name(effective) == "Recall"
    assert saved.objective_settings["objective_rank"] == "geometric_mean"
    assert saved.objective_settings["objective_axes"] == list(OBJECTIVE_LABELS)
    assert snapshot_objectives(config_for_run(historical, saved)) == snapshot_objectives(effective)
    assert snapshot_objectives(historical) == original


def test_require_batch_target_fails_and_saves_status(tmp_path, monkeypatch):
    from autofa.session import runner

    run = SimpleNamespace(run_id="run-1", min_failure_modes=1, run_dir=lambda: tmp_path)
    runtime = SimpleNamespace(library=SimpleNamespace(functions=[]), config=Config(), run=run)
    monkeypatch.setattr(runner, "load_runtime_from_run", lambda run, config: runtime)
    with pytest.raises(RuntimeError, match="0/1 validated"):
        runner.require_batch_target(run, cfg=Config())
    target = json.loads((tmp_path / "discovery_target.json").read_text())
    assert target["minimum_met"] is False
    assert target["remaining_failure_modes"] == 1


def test_preflight_checks_requested_services_but_defers_paid_detector_probe(monkeypatch):
    requested = []
    catalog = SimpleNamespace(error=None, find=lambda model: SimpleNamespace(effort_levels=()))
    monkeypatch.setattr("autofa.run_spec._harness_catalog", lambda *args: catalog)
    monkeypatch.setattr("autofa.llm.hub.make_hub_client",
                        lambda **kwargs: requested.append("detector_credentials"))
    monkeypatch.setattr("autofa.run_spec._require_publication",
                        lambda config: requested.append("publication"))
    # Even legacy round-limited discovery waits for run accounting before a paid probe.
    config = make_autofa_config(Config(campaign_policy={"round_limit": 10}),
                                execution=AutoFAExecution(
        mode="batch", require_llm_detectors=True, publish=True))
    preflight_autofa_config(config)
    assert requested == ["detector_credentials", "publication"]
    assert config.settings.require_llm_detectors


@pytest.mark.parametrize("allowed", [True, False])
def test_preflight_accepts_approved_fallback_credentials_without_baseline(monkeypatch, allowed):
    settings = Config(classifier_comparison_limit=1 if allowed else 0, classifier_profiles=[{
        "name": "fallback", "feature_model": "fallback-model",
        "feature_api_key_env": "FALLBACK_API_KEY",
    }])
    monkeypatch.setattr("autofa.run_spec._harness_catalog", lambda *args: SimpleNamespace())
    monkeypatch.setattr("autofa.run_spec._require_model", lambda *args: None)
    closed = []

    def client(**kwargs):
        if kwargs["api_key_env"] != "FALLBACK_API_KEY":
            raise RuntimeError("Baseline credential missing")
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr("autofa.llm.hub.make_hub_client", client)
    config = make_autofa_config(
        settings, execution=AutoFAExecution(mode="batch", require_llm_detectors=True))
    if not allowed:
        with pytest.raises(RuntimeError, match="No approved detector route"):
            preflight_autofa_config(config)
        assert not closed
    else:
        preflight_autofa_config(config)
        assert closed == [True]
        assert settings.require_llm_detectors


# --- Setup contract: read-only inspection, frozen agreement and launch verification ---

@pytest.fixture
def setup_files(tmp_path, monkeypatch):
    from support.config import save_config

    home = tmp_path / "home"
    monkeypatch.setenv("AUTOFA_HOME", str(home))
    monkeypatch.delenv("AUTOFA_CONFIG", raising=False)
    monkeypatch.delenv("AUTOFA_SETUP_CONTRACT", raising=False)
    cfg = Config(max_rounds=14, min_failure_modes=4,
                 campaign_policy={"coverage_target": .8, "round_limit": 14})
    save_config(cfg)
    batch = tmp_path / "run.yaml"
    batch.write_text("version: 2\nsource:\n  local_folder: data\n"
                     "execution:\n  publish: true\n  require_llm_detectors: true\n")
    return home, batch


def freeze(tmp_path, snapshot):
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(snapshot))
    return expected


def test_read_only_inspection_creates_nothing_and_does_not_authenticate(tmp_path, monkeypatch):
    home = tmp_path / "absent"
    monkeypatch.setenv("AUTOFA_HOME", str(home))
    monkeypatch.delenv("AUTOFA_CONFIG", raising=False)
    monkeypatch.setattr("autofa.harness.settings.detect_default_harness",
                        lambda **kwargs: pytest.fail("Inspection must not authenticate"))
    result = inspect_config()
    assert result["provenance"]["first_launch_detection_pending"]
    assert result["pricing"]["inference_calls"] == 0
    assert not home.exists()


def test_read_only_legacy_resolution_matches_real_migration(tmp_path, monkeypatch):
    path = tmp_path / "legacy.toml"
    path.write_text('harness = "claude"\nmax_rounds = 3\n'
                    'judge_request_options = {}\n')
    monkeypatch.setenv("AUTOFA_CONFIG", str(path))
    original = path.read_bytes()
    inspected = load_config(read_only=True)
    assert path.read_bytes() == original
    executed = load_config()
    assert asdict(inspected) == asdict(executed)


def test_freeze_agreement_and_verify_real_run_snapshot(setup_files, tmp_path, monkeypatch):
    _, batch = setup_files
    monkeypatch.setenv("OPENAI_API_KEY", "this-value-is-secret")
    monkeypatch.setenv("INFERENCE_HUB_API_KEY", "another-secret-value")
    snapshot = inspect_config(batch, capabilities=["vision"])
    # Inspection never exposes environment credentials.
    assert "this-value-is-secret" not in json.dumps(snapshot)
    assert "another-secret-value" not in json.dumps(snapshot)
    expected = freeze(tmp_path, snapshot)
    result = prepare_config(batch, expected, tmp_path / "prepared")
    for name, value in result["env"].items():
        monkeypatch.setenv(name, value)
    config = load_autofa_config(result["run_config"], load_config(read_only=True))
    assert config.execution.publish
    assert config.settings.max_rounds == 14
    validate_launch_contract(config)
    config.settings.require_llm_detectors = True
    run = RunConfig.create(parquet_paths=[], model_id="test", benchmarks=None, cfg=config.settings)
    validate_launch_contract(config, run=run)
    assert required_capabilities(run.run_dir()) == ["vision"]
    assert (run.run_dir() / "usage_initialization.json").is_file()


@pytest.mark.parametrize("command", ["inspect", "prepare"])
def test_classification_config_has_actionable_setup_error_without_side_effects(
    setup_files, tmp_path, capsys, command,
):
    from autofa.setup_contract import main

    _, batch = setup_files
    expected = freeze(tmp_path, inspect_config(batch))
    taxonomy = write_spec(tmp_path / "classification.yaml",
                          yaml.safe_dump(taxonomy_root(endpoint=False)))
    before = set(tmp_path.rglob("*"))
    argv = [command, "--run-config", str(taxonomy)]
    if command == "prepare":
        argv.extend(["--expected", str(expected), "--output", str(tmp_path / "prepared")])

    assert main(argv) == 1
    captured = capsys.readouterr()
    assert not captured.out
    result = json.loads(captured.err)
    assert result["ok"] is False
    assert "managed discovery only" in result["error"]
    assert "autofa run --config" in result["error"]
    assert set(tmp_path.rglob("*")) == before


def test_changed_settings_publish_or_runtime_are_rejected_before_any_preflight(
    setup_files, tmp_path, monkeypatch,
):
    from dataclasses import replace

    from autofa import cli, setup_contract

    _, batch = setup_files
    snapshot = inspect_config(batch)
    monkeypatch.setenv("AUTOFA_SETUP_CONTRACT", str(freeze(tmp_path, snapshot)))
    config = load_autofa_config(batch, load_config(read_only=True))
    monkeypatch.setattr("autofa.run_spec.preflight_autofa_config",
                        lambda *args, **kwargs: pytest.fail("Must fail before probes"))
    with pytest.raises(ValueError, match=r"settings.max_rounds"):
        cli.run_autofa(replace(config, settings=replace(config.settings, max_rounds=3)))
    unpublished = replace(config, execution=replace(config.execution, publish=False))
    with pytest.raises(ValueError, match=r"execution.publish"):
        validate_launch_contract(unpublished)
    changed = {**snapshot["runtime"], "source_sha256": "another-source-tree"}
    monkeypatch.setattr(setup_contract, "runtime_identity", lambda: changed)
    with pytest.raises(ValueError, match=r"runtime.source_sha256"):
        validate_launch_contract(config)


def test_capability_tampering_invalidates_agreement(setup_files, tmp_path, monkeypatch):
    _, batch = setup_files
    snapshot = inspect_config(batch, capabilities=["vision"])
    snapshot["required_capabilities"] = []
    monkeypatch.setenv("AUTOFA_SETUP_CONTRACT", str(freeze(tmp_path, snapshot)))
    with pytest.raises(ValueError, match="fingerprint changed"):
        validate_launch_contract(load_autofa_config(batch, load_config(read_only=True)))


def test_prepare_detects_changed_config_without_leaving_partial_output(
    setup_files, tmp_path,
):
    home, batch = setup_files
    expected = freeze(tmp_path, inspect_config(batch))
    with (home / "config.toml").open("a") as stream:
        stream.write('\n# unrelated comment\n')
    prepared = tmp_path / "prepared"
    # Formatting does not change effective settings.
    prepare_config(batch, expected, prepared)
    with pytest.raises(FileExistsError):
        prepare_config(batch, expected, prepared)
    assert (prepared / "autofa.toml").is_file()
    config_path = home / "config.toml"
    config_path.write_text(config_path.read_text().replace('max_rounds = 14', 'max_rounds = 3'))
    with pytest.raises(ValueError, match="Setup changed"):
        prepare_config(batch, expected, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_preparation_pins_default_home_even_without_environment_override(
    setup_files, tmp_path, monkeypatch,
):
    home, batch = setup_files
    monkeypatch.setenv("AUTOFA_CONFIG", str(home / "config.toml"))
    monkeypatch.delenv("AUTOFA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    snapshot = inspect_config(batch)
    result = prepare_config(batch, freeze(tmp_path, snapshot), tmp_path / "prepared")
    assert result["env"]["AUTOFA_HOME"] == str(tmp_path / "user" / ".autofa")
    assert snapshot["runtime"]["autofa_home"] == result["env"]["AUTOFA_HOME"]


def test_pricing_readiness_distinguishes_fallback_and_unpriced_harness():
    cfg = Config(classifier_profiles=[{
        "name": "fallback", "feature_model": "fallback-model",
    }])
    from autofa.llm.endpoints import endpoint_for_config

    endpoint = endpoint_for_config(cfg)
    cfg.usage_pricing = {"version": 1, "currency": "USD", "rates": [{
        "provider": "openai_compatible", "model": endpoint.model,
        "endpoint": endpoint.base_url, "input_per_million": 1,
        "output_per_million": 2, "source": "configured test tariff",
        "effective_at": "2026-11-05",
    }]}
    result = monetary_readiness(cfg)
    rows = {row.get("profile", row["role"]): row for row in result["roles"]}
    assert rows["baseline"]["cost_status"] == "estimated_when_usage_available"
    assert rows["fallback"]["cost_status"] == "unverified"
    assert rows["discovery"]["configured_rate"] is None
    assert result["needs_pricing_setup"]
    assert not result["estimates_ready"]
    assert {row.get("profile", row["role"]) for row in result["missing_rate_roles"]} >= {
        "discovery", "behavior_judge", "fallback",
    }


def test_harness_rate_readiness_is_reference_only_not_verified_billing():
    from autofa.session.usage import validate_pricing

    cfg = Config(harness="codex")
    inspected = monetary_readiness(cfg)
    rates = {}
    for role in inspected["roles"]:
        rates[(role["provider"], role["model"], role["endpoint"])] = {
            "provider": role["provider"], "model": role["model"],
            **({"endpoint": role["endpoint"]} if role["endpoint"] else {}),
            "input_per_million": 1, "output_per_million": 2,
            "source": "synthetic readiness test tariff", "effective_at": "2026-11-05",
        }
    cfg.usage_pricing = validate_pricing({
        "version": 1, "currency": "USD", "rates": list(rates.values()),
    })
    result = monetary_readiness(cfg)
    assert not result["needs_pricing_setup"]
    assert result["estimates_ready"]
    assert result["missing_rate_roles"] == []
    for role in result["roles"]:
        assert role["billing_status"] == "unverified_until_runtime"
        if role["provider"] in {"codex", "claude"}:
            assert role["cost_status"] == "token_reference_estimate_available"
        else:
            assert role["cost_status"] == "estimated_when_usage_available"
    assert result["inference_calls"] == 0
