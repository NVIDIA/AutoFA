# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only discovery setup inspection and launch agreement verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from autofa.config import (
    Config,
    RunConfig,
    config_source_path,
    global_env_path,
    load_config,
    site_config_path,
)
from autofa.core.dependencies import digest
from autofa.harness.settings import behavior_judge_settings, harness_model
from autofa.llm.endpoints import endpoint_for_config
from autofa.objectives import config_for_new_discovery, weights_by_axis
from autofa.run_config import AutoFAConfig, SourceSpec
from autofa.session.recovery import atomic_write_json

_FIELDS = (
    "analysis_outputs", "analysis_focus", "harness", "model", "effort", "thinking",
    "thinking_budget_tokens", "codex_cli", "claude_cli", "behavior_judge_harness",
    "behavior_judge_model", "behavior_judge_effort", "behavior_judge_thinking",
    "behavior_judge_thinking_budget_tokens", "behavior_strictness", "max_rounds",
    "min_failure_modes", "top_x", "campaign_policy", "usage_pricing",
    "require_llm_detectors", "discovery_wait_for_plateau", "discovery_terminate_if_stalled",
    "discovery_stall_rounds", "discovery_plateau_patience", "default_eval_limit",
    "objective_axes", "objective_weights", "objective_rank",
    "hub_base_url", "hub_api_key_env", "feature_model", "feature_api", "feature_base_url",
    "feature_api_key_env", "feature_request_options", "judge_model", "judge_api",
    "judge_base_url", "judge_api_key_env", "judge_request_options", "improvement_request_options",
    "classifier_profiles", "classifier_comparison_limit", "detector_fields", "allow_metadata",
    "allowed_fields", "extra_allowed_fields", "banned_fields",
    "llm_eval_sample", "llm_eval_workers", "llm_eval_max_workers", "llm_eval_queue_mib",
    "judge_workers", "remote_library_base_url", "remote_library_mode",
    "remote_library_token_env", "data_api_base_url", "remote_full_traces",
)


def runtime_identity() -> dict[str, str]:
    """Pin the interpreter/environment and installed native source, without Git commands."""
    package = Path(__file__).resolve().parent
    checksum = hashlib.sha256()
    paths = sorted({*package.rglob("*.py"), *package.rglob("*.toml")})
    for path in paths:
        checksum.update(path.relative_to(package).as_posix().encode() + b"\0")
        checksum.update(hashlib.sha256(path.read_bytes()).digest())
    return {"python_executable": str(Path(sys.executable).absolute()),
            "environment": sys.prefix, "package_root": str(package),
            "autofa_home": str(global_env_path().parent.resolve()),
            "source_sha256": checksum.hexdigest()}


def _safe(value: Any) -> Any:
    # Endpoint validation forbids credential-bearing URLs. Request options may
    # contain provider extensions; reuse the existing recursive secret redactor.
    from autofa.preflight.classifiers import _redact_options

    return _redact_options(value)


def monetary_readiness(cfg: Config) -> dict[str, Any]:
    """Describe configured rate coverage; provider-reported money is not promised."""
    from autofa.session.usage import select_usage_rate

    roles: list[dict[str, Any]] = []
    judge = behavior_judge_settings(cfg)
    for role, provider, model in (
        ("discovery", cfg.harness, harness_model(cfg.harness, cfg.model, cfg=cfg)),
        ("behavior_judge", judge["harness"], judge["model"]),
    ):
        roles.append({"role": role, "provider": provider, "model": model, "endpoint": None})
    for profile in [{"name": "baseline"}, *cfg.classifier_profiles]:
        scoped = deepcopy(cfg)
        for key, value in profile.items():
            if key != "name":
                setattr(scoped, key, deepcopy(value))
        endpoint = endpoint_for_config(scoped)
        roles.append({"role": "detector", "profile": profile["name"],
                      "provider": "openai_compatible", "model": endpoint.model,
                      "endpoint": endpoint.base_url})
    auxiliary = endpoint_for_config(cfg, "judge")
    roles.append({"role": "auxiliary_classifier", "provider": "openai_compatible",
                  "model": auxiliary.model, "endpoint": auxiliary.base_url})
    for role in roles:
        rate = select_usage_rate(Path(), role["provider"], role["model"], role["endpoint"],
                                 pricing=cfg.usage_pricing or {})
        role["configured_rate"] = _safe(rate) if rate else None
        role["billing_status"] = "unverified_until_runtime"
        role["cost_status"] = (
            "token_reference_estimate_available"
            if rate and role["provider"] in {"codex", "claude"}
            else "estimated_when_usage_available" if rate else "unverified"
        )
    missing = [{key: role[key] for key in ("role", "provider", "model", "endpoint", "profile")
                if key in role}
               for role in roles if role["configured_rate"] is None]
    return {"currency": "USD", "roles": roles, "inference_calls": 0,
            "missing_rate_roles": missing, "needs_pricing_setup": bool(missing),
            "estimates_ready": not missing,
            "next_action": (
                "Resolve the missing roles' exact deployment prices or a documented public-rate "
                "reference before promising dollar estimates; save rates in the scoped setup."
                if missing else "Rates are configured; verify emitted token and monetary data "
                "after the first native work."
            ),
            "unknown_cost_policy": cfg.campaign_policy.get("unknown_cost_policy"),
            "notice": "No calls made. Missing rates or failed-call counters are unknown cost, "
                      "not zero. Native dollars may be reported at runtime; credits are not USD. "
                      "Harness token-reference estimates are not billing data and cannot resolve "
                      "missing money for spending admission. "
                      "Outer host supervision cost is separate unless explicitly measured."}


def resolved_contract(cfg: Config, *, execution: dict | None = None,
                      source: dict | None = None,
                      capabilities: list[str] | None = None) -> dict[str, Any]:
    """Return effective discovery choices without credentials or model access probes."""
    cfg = config_for_new_discovery(deepcopy(cfg))
    # Canonical resolved settings avoid treating inherited endpoint/harness aliases
    # as an actual model change when the native run pins them explicitly.
    from autofa.config import snapshot_harness_settings

    settings = {name: deepcopy(getattr(cfg, name)) for name in _FIELDS}
    settings.update(snapshot_harness_settings(cfg))
    for role in ("feature", "judge"):
        endpoint = endpoint_for_config(cfg, role)
        settings.update({f"{role}_model": endpoint.model, f"{role}_api": endpoint.api,
                         f"{role}_base_url": endpoint.base_url,
                         f"{role}_api_key_env": endpoint.api_key_env,
                         f"{role}_request_options": _safe(endpoint.request_options)})
    settings = _safe(settings)
    capabilities = capabilities or []
    if any(item not in ("text", "vision") for item in capabilities):
        raise ValueError("Required capabilities must be text and/or vision")
    core = {"settings": settings, "execution": execution or {}, "source": source or {},
            "required_capabilities": sorted(set(capabilities)), "runtime": runtime_identity()}
    selected = config_source_path()
    return {"schema_version": 1, **core, "fingerprint": digest(core),
            "provenance": {"config_source": str(selected),
                           "source_exists": selected.is_file(),
                           "site_config": str(site_config_path() or ""),
                           "first_launch_detection_pending": not selected.is_file(),
                           "source_checkout": str(Path(__file__).resolve().parents[2])},
            "resolved": {"objective_weights": weights_by_axis(cfg),
                         "coverage_grade_scope": "GOOD-only",
                         "coverage_unit": cfg.campaign_policy.get("coverage_unit"),
                         "discovery_model": settings["model"],
                         "judge_model": settings["behavior_judge_model"],
                         "detector_model": settings["feature_model"]},
            "pricing": monetary_readiness(cfg)}


def _source(config: AutoFAConfig) -> dict:
    if not isinstance(config.source, SourceSpec):
        raise ValueError("Setup contracts require the original source specification")
    return {"evalrdl_url": config.source.evalrdl_url,
            "local_folder": str(config.source.local_folder) if config.source.local_folder else None,
            "refresh": config.source.refresh}


def _load_discovery_config(run_config: str | Path, settings: Config) -> AutoFAConfig:
    from autofa.run_spec import load_autofa_config

    config = load_autofa_config(run_config, settings)
    if not isinstance(config, AutoFAConfig):
        raise ValueError(
            "Setup inspection and preparation support managed discovery only. "
            "For workflow: classification, use autofa run --config <classification.yaml>; "
            "use autofa new --config <classification.yaml> or "
            "autofa continue --config <classification.yaml> for a later "
            "taxonomy-to-discovery handoff."
        )
    return config


def inspect_config(run_config: str | Path | None = None, *,
                   capabilities: list[str] | None = None) -> dict[str, Any]:
    cfg = load_config(read_only=True)
    if run_config is None:
        return resolved_contract(cfg, capabilities=capabilities)
    config = _load_discovery_config(run_config, cfg)
    if config.execution.require_llm_detectors:
        config.settings.require_llm_detectors = True
    return resolved_contract(config.settings, execution=asdict(config.execution),
                             source=_source(config), capabilities=capabilities)


def _read_contract(path: Path) -> dict:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Setup contract exceeds 2 MiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or not isinstance(value.get("settings"), dict)
            or set(value["settings"]) != set(_FIELDS)
            or not isinstance(value.get("execution"), dict)
            or not isinstance(value.get("source"), dict)
            or not isinstance(value.get("runtime"), dict)):
        raise ValueError("Invalid setup contract; inspect the matching checkout again")
    capabilities = value.get("required_capabilities", [])
    if (not isinstance(capabilities, list)
            or any(item not in ("text", "vision") for item in capabilities)):
        raise ValueError("required_capabilities must contain text and/or vision")
    core = {key: value.get(key, []) for key in
            ("settings", "execution", "source", "required_capabilities", "runtime")}
    if value.get("fingerprint") != digest(core):
        raise ValueError("Setup contract fingerprint changed; inspect the agreed config again")
    return value


def required_capabilities(run_dir: Path | None = None) -> list[str]:
    """Capabilities explicitly selected by the driver; never inferred from a dataset name."""
    path = Path(run_dir) / "setup_contract.json" if run_dir is not None else None
    if path is None or not path.exists():
        supplied = os.environ.get("AUTOFA_SETUP_CONTRACT")
        path = Path(supplied) if supplied else None
    return list(_read_contract(path).get("required_capabilities", [])) if path else []


def validate_launch_contract(config: AutoFAConfig, *, run: RunConfig | None = None) -> None:
    """Fail before paid discovery if execution differs from the approved snapshot.

    This is a new-launch guard, not a Resume override: saved live controls and
    authorized classifier recovery remain authoritative after initial binding.
    """
    supplied = os.environ.get("AUTOFA_SETUP_CONTRACT")
    if not supplied:
        return
    expected = _read_contract(Path(supplied))
    cfg = deepcopy(config.settings)
    if config.execution.require_llm_detectors:
        cfg.require_llm_detectors = True
    if run is not None:
        from autofa.config import restore_harness_settings
        from autofa.objectives import config_for_run
        from autofa.session.classifier_profiles import apply_run_classifier

        cfg = apply_run_classifier(restore_harness_settings(config_for_run(cfg, run), run), run)
        for name in ("max_rounds", "min_failure_modes", "top_x", "require_llm_detectors",
                     "analysis_outputs", "usage_pricing"):
            setattr(cfg, name, getattr(run, name))
    actual = resolved_contract(cfg, execution=asdict(config.execution), source=_source(config),
                               capabilities=expected.get("required_capabilities"))
    mismatches = []
    for section in ("settings", "execution", "source", "runtime"):
        for key in sorted(set(expected[section]) | set(actual[section])):
            if expected[section].get(key) != actual[section].get(key):
                mismatches.append(f"{section}.{key}")
    if mismatches:
        raise ValueError("Launch differs from agreed setup: " + ", ".join(mismatches)
                         + ". Reconcile the agreement before paid discovery.")
    if run is not None:
        atomic_write_json(run.run_dir() / "setup_contract.json", expected)


def prepare_config(run_config: Path, expected_path: Path, output: Path) -> dict:
    """Freeze a verified setup into a private profile and explicit batch specification."""
    import yaml

    from autofa.config import _toml_value
    expected = _read_contract(expected_path)
    config = _load_discovery_config(run_config, load_config(read_only=True))
    if config.execution.require_llm_detectors:
        config.settings.require_llm_detectors = True
    config = replace(config, settings=config_for_new_discovery(config.settings))
    actual = resolved_contract(config.settings, execution=asdict(config.execution),
                               source=_source(config),
                               capabilities=expected.get("required_capabilities"))
    if actual["fingerprint"] != expected["fingerprint"]:
        raise ValueError("Setup changed since agreement; inspect it again before preparing launch")
    output = output.expanduser().resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        profile = output / "autofa.toml"
        values = asdict(config.settings)
        if values["exploration_seed"] is None:
            values["exploration_seed"] = "auto"
        lines = [f"{key} = {_toml_value(value)}" for key, value in values.items()]
        profile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        profile.chmod(0o600)
        batch = output / "run.yaml"
        source = {key: value for key, value in _source(config).items() if value is not None}
        batch.write_text(yaml.safe_dump({"version": 2, "source": source,
                                        "execution": {"publish": config.execution.publish,
                                                      "require_llm_detectors":
                                                      config.execution.require_llm_detectors}}),
                         encoding="utf-8")
        atomic_write_json(output / "setup_contract.json", expected)
        contract = output / "setup_contract.json"
        env = {"AUTOFA_CONFIG": str(profile), "AUTOFA_SETUP_CONTRACT": str(contract),
               "AUTOFA_HOME": str(global_env_path().parent.resolve())}
        files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (profile, batch, contract)}
        argv = [sys.executable, "-m", "autofa", "run", "--config", str(batch), "--once"]
        spec = {"launch_id": secrets.token_hex(16), "argv": argv, "cwd": str(Path.cwd()),
                "env": env, "config_files": files}
        return {"ok": True, "profile": str(profile), "run_config": str(batch),
                "contract": str(contract), "fingerprint": expected["fingerprint"],
                "env": env, "config_files": files, "argv": argv, "launch_spec": spec}
    except BaseException:
        import shutil

        shutil.rmtree(output)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autofa config")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read effective setup without calls or writes")
    inspect.add_argument("--run-config", type=Path)
    inspect.add_argument("--require-capability", action="append", choices=("text", "vision"),
                         help="Explicit required capability included in the agreed fingerprint")
    inspect.add_argument("--json", action="store_true", help="Structured output (default)")
    prepare = commands.add_parser("prepare", help="Freeze an agreed setup for managed launch")
    prepare.add_argument("--run-config", required=True, type=Path)
    prepare.add_argument("--expected", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = (inspect_config(args.run_config, capabilities=args.require_capability)
                  if args.command == "inspect" else
                  prepare_config(args.run_config, args.expected, args.output))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
