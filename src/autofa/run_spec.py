# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from autofa.config import Config, validate_min_failure_modes
from autofa.data.preparation import PreparedData
from autofa.data.reference_data import parse_evalrdl_run_url
from autofa.objectives import PRESETS, apply_preset
from autofa.run_config import (
    AutoFAConfig,
    AutoFAExecution,
    SourceSpec,
    _boolean,
    _integer,
    _keys,
    _mapping,
    _optional_choice,
    _optional_mapping,
    _optional_positive_integer,
    _optional_text,
    _text,
)
from autofa.taxonomy.workflow_config import (
    TaxonomyWorkflowSpec,
    parse_taxonomy_workflow,
)


@dataclass(frozen=True)
class ConfigSpec:
    harness: str | None = None
    model: str | None = None
    effort: str | None = None
    analysis_focus: str | None = None
    min_failure_modes: int | None = None
    max_rounds: int | None = None
    top_x: int | None = None
    default_eval_limit: int | None = None
    optimization: str | None = None
    campaign_policy: dict[str, Any] | None = None
    usage_pricing: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionSpec:
    publish: bool = False
    require_llm_detectors: bool = False


@dataclass(frozen=True)
class RunSpec:
    path: Path
    source: SourceSpec
    config: ConfigSpec
    execution: ExecutionSpec


def load_run_spec(path: str | Path) -> RunSpec | TaxonomyWorkflowSpec:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Run config does not exist: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read run config {resolved}: {exc}") from exc
    root = _mapping(raw, "run config")
    workflow = root.get("workflow", "discovery")
    if workflow == "classification":
        _keys(
            root,
            {"version", "workflow", "source", "selection", "taxonomy"},
            "run config",
        )
        version = _integer(root.get("version"), "version")
        if version != 2:
            raise ValueError(f"Unsupported run config version: {version}")
        return parse_taxonomy_workflow(root, resolved.parent)
    if workflow != "discovery":
        raise ValueError("workflow must be discovery or classification")
    _keys(
        root,
        {"version", "workflow", "source", "config", "execution"},
        "run config",
    )
    version = _integer(root.get("version"), "version")
    if version != 2:
        raise ValueError(f"Unsupported run config version: {version}")

    source_raw = _mapping(root.get("source"), "source")
    _keys(source_raw, {"evalrdl_url", "local_folder", "refresh"}, "source")
    evalrdl_value = source_raw.get("evalrdl_url")
    evalrdl_url = None
    if evalrdl_value is not None:
        evalrdl_url = _text(evalrdl_value, "source.evalrdl_url")
        parse_evalrdl_run_url(evalrdl_url)
    local_value = source_raw.get("local_folder")
    local_folder = None
    if local_value is not None:
        local_text = _text(local_value, "source.local_folder")
        local_folder = (resolved.parent / local_text).expanduser().resolve()
    if (evalrdl_url is None) == (local_folder is None):
        raise ValueError("source must define exactly one of evalrdl_url or local_folder")
    refresh = _boolean(source_raw.get("refresh", False), "source.refresh")

    config_raw = _optional_mapping(root.get("config"), "config")
    _keys(
        config_raw,
        {
            "harness",
            "model",
            "effort",
            "analysis_focus",
            "min_failure_modes",
            "max_rounds",
            "top_x",
            "default_eval_limit",
            "optimization",
            "campaign_policy",
            "usage_pricing",
        },
        "config",
    )
    harness = _optional_choice(
        config_raw.get("harness"),
        {"codex", "claude"},
        "config.harness",
    )
    model = _optional_text(config_raw.get("model"), "config.model")
    effort = _optional_text(config_raw.get("effort"), "config.effort")
    analysis_focus = _optional_choice(
        config_raw.get("analysis_focus"),
        {"model", "harness", "both"},
        "config.analysis_focus",
    )
    minimum = config_raw.get("min_failure_modes")
    min_failure_modes = None
    if minimum is not None:
        min_failure_modes = validate_min_failure_modes(
            _integer(minimum, "config.min_failure_modes")
        )
    max_rounds = _optional_positive_integer(
        config_raw.get("max_rounds"),
        "config.max_rounds",
    )
    top_x = _optional_positive_integer(config_raw.get("top_x"), "config.top_x")
    default_eval_limit = _optional_positive_integer(
        config_raw.get("default_eval_limit"),
        "config.default_eval_limit",
    )
    optimization = _optional_choice(
        config_raw.get("optimization"), set(PRESETS), "config.optimization",
    )
    from autofa.session.campaign_policy import validate_campaign_policy
    from autofa.session.usage import validate_pricing

    campaign = (validate_campaign_policy(config_raw["campaign_policy"])
                if "campaign_policy" in config_raw else None)
    pricing = None
    if "usage_pricing" in config_raw:
        pricing = ({} if config_raw["usage_pricing"] == {} else
                   validate_pricing(config_raw["usage_pricing"]))

    execution_raw = _optional_mapping(root.get("execution"), "execution")
    _keys(execution_raw, {"publish", "require_llm_detectors"}, "execution")
    publish = _boolean(execution_raw.get("publish", False), "execution.publish")
    require_llm_detectors = _boolean(
        execution_raw.get("require_llm_detectors", False),
        "execution.require_llm_detectors",
    )

    return RunSpec(
        path=resolved,
        source=SourceSpec(
            evalrdl_url=evalrdl_url,
            local_folder=local_folder,
            refresh=refresh,
        ),
        config=ConfigSpec(
            harness=harness,
            model=model,
            effort=effort,
            analysis_focus=analysis_focus,
            min_failure_modes=min_failure_modes,
            max_rounds=max_rounds,
            top_x=top_x,
            default_eval_limit=default_eval_limit,
            optimization=optimization,
            campaign_policy=campaign,
            usage_pricing=pricing,
        ),
        execution=ExecutionSpec(
            publish=publish,
            require_llm_detectors=require_llm_detectors,
        ),
    )


def load_autofa_config(
    path: str | Path, settings: Config,
) -> AutoFAConfig | TaxonomyWorkflowSpec:
    spec = load_run_spec(path)
    if isinstance(spec, TaxonomyWorkflowSpec):
        return spec
    effective = apply_run_spec(settings, spec)
    effective.analysis_outputs = sorted(set(effective.analysis_outputs) | {"agent"})
    return AutoFAConfig(
        settings=effective,
        source=spec.source,
        execution=AutoFAExecution(
            mode="batch",
            publish=spec.execution.publish,
            require_llm_detectors=spec.execution.require_llm_detectors,
        ),
    )


def apply_run_spec(config: Config, spec: RunSpec) -> Config:
    if spec.config.optimization is not None:
        config = apply_preset(config, spec.config.optimization)
    if spec.config.harness:
        from autofa.defaults import harness_default

        changed = spec.config.harness != config.harness
        config.harness = spec.config.harness
        if spec.config.model is None and changed:
            config.model = harness_default(config.harness, "model", config)
        if spec.config.effort is None and changed:
            config.effort = harness_default(config.harness, "effort", config)
            config.thinking = harness_default(config.harness, "thinking", config)
    if spec.config.model:
        config.model = spec.config.model
    if spec.config.effort:
        config.effort = spec.config.effort
    if spec.config.analysis_focus:
        config.analysis_focus = spec.config.analysis_focus
    if spec.config.min_failure_modes is not None:
        config.min_failure_modes = spec.config.min_failure_modes
    if spec.config.campaign_policy is not None:
        config.campaign_policy = spec.config.campaign_policy
    if spec.config.usage_pricing is not None:
        config.usage_pricing = spec.config.usage_pricing
    config.max_rounds = spec.config.max_rounds or config.max_rounds
    config.top_x = spec.config.top_x or config.top_x
    if spec.config.default_eval_limit is not None:
        config.default_eval_limit = spec.config.default_eval_limit
    from autofa.session.guidance import snapshot_guidance

    snapshot_guidance(config)
    return config


def preflight_autofa_config(config: AutoFAConfig) -> None:
    from autofa.harness.settings import (
        behavior_judge_settings,
        harness_cli,
        harness_model,
    )

    settings = config.settings
    selected = []
    if "agent" in settings.analysis_outputs:
        selected.append(
            (
                settings.harness,
                harness_model(settings.harness, settings.model, cfg=settings),
                settings.effort,
                harness_cli(settings, settings.harness),
                "discovery",
            )
        )
        judge = behavior_judge_settings(settings)
        selected.append(
            (
                judge["harness"],
                judge["model"],
                judge["effort"],
                judge["cli_path"],
                "behavior judge",
            )
        )
    catalogs = {}
    for harness, model, effort, cli_path, role in selected:
        key = (harness, cli_path)
        if key not in catalogs:
            catalogs[key] = _harness_catalog(harness, cli_path)
        _require_model(catalogs[key], harness, model, effort, role)
    if config.execution.require_llm_detectors:
        # Validate credentials here without inference. Paid readiness belongs to
        # the initialized run, including legacy round-limited discovery, so cost
        # and availability recovery are recorded from the first physical call.
        from copy import deepcopy

        from autofa.llm.endpoints import endpoint_for_config
        from autofa.llm.hub import make_hub_client

        alternatives = settings.classifier_profiles if settings.classifier_comparison_limit else []
        credential_error = None
        for profile in [{"name": "baseline"}, *alternatives]:
            scoped = deepcopy(settings)
            for key, value in profile.items():
                if key != "name":
                    setattr(scoped, key, deepcopy(value))
            endpoint = endpoint_for_config(scoped)
            try:
                client = make_hub_client(
                    base_url=endpoint.base_url, api_key_env=endpoint.api_key_env)
            except RuntimeError as exc:
                credential_error = exc
                continue
            if callable(getattr(client, "close", None)):
                client.close()
            break
        else:
            raise RuntimeError("No approved detector route has usable credentials. "
                               "Configure a baseline or approved fallback credential.") \
                from credential_error
        settings.require_llm_detectors = True
    if config.execution.publish:
        _require_publication(settings)


def _harness_catalog(harness: str, cli_path: str | None):
    from autofa.harness.model_catalog import (
        _find_codex_cli,
        load_model_catalog,
    )
    from autofa.harness.settings import HARNESS_NAMES, _authentication_status

    if harness not in HARNESS_NAMES:
        raise RuntimeError(f"Batch readiness checks do not support harness: {harness}")
    if harness == "codex":
        try:
            binary = _find_codex_cli(cli_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Codex readiness check failed: CLI executable not found. Install Codex or set "
                "AUTOFA_CODEX_CLI, then run `codex login`."
            ) from exc
    else:
        from autofa.harness.claude_cli import ensure_claude_auth

        try:
            binary = ensure_claude_auth(cli_path=cli_path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Claude Code readiness check failed: {exc}") from exc
    if binary:
        problem = _authentication_status(binary, harness)
        if problem:
            command = "codex login" if harness == "codex" else "claude auth login"
            raise RuntimeError(
                f"{HARNESS_NAMES[harness]} readiness check failed: {problem}. "
                f"Run `{command}` and retry."
            )
    catalog = load_model_catalog(cli_path=binary, harness=harness)
    return catalog


def _require_model(catalog, harness: str, model: str, effort: str, role: str) -> None:
    from autofa.harness.settings import HARNESS_NAMES

    if catalog.error:
        return
    selected = catalog.find(model)
    if selected is None:
        return
    if effort != "default" and selected.effort_levels and effort not in selected.effort_levels:
        available = ", ".join(selected.effort_levels)
        raise RuntimeError(
            f"{HARNESS_NAMES[harness]} {role} effort readiness check failed: `{effort}` is not "
            f"supported by `{model}`. Available efforts: {available}."
        )


def _require_publication(config: Config) -> None:
    from autofa.data.remote_library_api import client_from_config

    try:
        client = client_from_config(config)
        client.me()
    except Exception as exc:
        raise RuntimeError(
            "Publication readiness check failed: configure and validate the library credential "
            "or set execution.publish to false. "
            f"{exc}"
        ) from exc


def materialize_autofa_config(config: AutoFAConfig) -> PreparedData:
    if isinstance(config.source, PreparedData):
        return config.source
    return _materialize_source(config.source, config.settings)


def _materialize_source(source: SourceSpec, config: Config) -> PreparedData:
    folder = source.local_folder
    if folder is not None:
        if not folder.is_dir():
            raise ValueError(f"source.local_folder is not a directory: {folder}")
        return _materialize_local(source, folder, config)
    base_url, run_id, _ = parse_evalrdl_run_url(source.evalrdl_url or "")
    return _materialize_remote(source, base_url, run_id, config)


def _materialize_local(
    source_spec: SourceSpec,
    folder: Path,
    config: Config,
) -> PreparedData:
    from autofa.data.cohort import prepare_data, select_cohort
    from autofa.data.efficiency_support import check_local_sources
    from autofa.data.local_import import reference_for_import
    from autofa.data.sources import discover_eval_sources

    sources = discover_eval_sources(folder)
    if len(sources) != 1:
        raise ValueError(
            f"source.local_folder must contain exactly one recognized eval run; "
            f"found {len(sources)} under {folder}"
        )
    check_local_sources(sources)
    source = sources[0]
    efficiency_only = (
        set(config.analysis_outputs) == {"efficiency"} and not config.preflight_detectors
    )
    if source.kind != "evalrdl" and reference_for_import(source.path) is None:
        from autofa.data.gym_adapter import materialize_eval_source
        from autofa.data.parquet_store import infer_models_benchmarks

        parquet = materialize_eval_source(
            source,
            force=source_spec.refresh,
            efficiency_only=efficiency_only,
        )
        metadata = infer_models_benchmarks([parquet])
        models = [str(model) for model, _count in metadata["models"]]
        benchmarks = [str(benchmark) for benchmark, _count in metadata["benchmarks"]]
        prepared = PreparedData(
            parquet_paths=[parquet],
            model_id=models[0] if len(models) == 1 else "*",
            benchmarks=benchmarks,
            source_paths=[str(folder)],
            source_kind=source.kind,
        )
        _require_dataset_labels(prepared)
        return prepared
    prepared = prepare_data(
        select_cohort(
            folders=[source.path],
            refresh=source_spec.refresh,
            efficiency_only=efficiency_only,
        ),
        efficiency_only=efficiency_only,
    )
    _require_dataset_labels(prepared)
    return prepared


def _materialize_remote(
    source: SourceSpec,
    base_url: str,
    run_id: str,
    config: Config,
) -> PreparedData:
    from autofa.data.cohort import prepare_data, select_cohort
    from autofa.data.remote_api import DataApiClient

    api = DataApiClient(base_url or config.data_api_base_url)
    efficiency_only = (
        set(config.analysis_outputs) == {"efficiency"} and not config.preflight_detectors
    )
    prepared = prepare_data(
        select_cohort(api_runs=[run_id], api=api),
        api=api,
        refresh=source.refresh,
        efficiency_only=efficiency_only,
    )
    _require_dataset_labels(prepared)
    return prepared


def _require_dataset_labels(prepared: PreparedData) -> None:
    from autofa.data.parquet_store import infer_models_benchmarks

    metadata = infer_models_benchmarks(prepared.parquet_paths)
    if not metadata["models"]:
        raise ValueError("Selected EvalRDL data has no model_id")
    if not metadata["benchmarks"]:
        raise ValueError("Selected EvalRDL data has no benchmark")
