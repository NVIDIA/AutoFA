# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect and explicitly configure permitted detector profiles without inference."""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

import tomlkit

from autofa.config import (
    Config,
    _config_template,
    _write_config_atomically,
    config_path,
    config_source_path,
    ensure_default_config,
)
from autofa.llm.endpoints import LLM_APIS, NO_AUTH, endpoint_for_config
from autofa.llm.hub import HubLLM, resolve_api_key_env
from autofa.session.classifier_profiles import PROFILE_FIELDS, validate_profile_config


def _redact_options(value: Any, secrets: list[str] | None = None) -> Any:
    """Hide nested credentials and transport headers while preserving tuning controls."""
    def collect(raw: Any) -> None:
        if isinstance(raw, str) and raw and secrets is not None:
            secrets.append(raw)
        elif isinstance(raw, dict):
            for child in raw.values():
                collect(child)
        elif isinstance(raw, (list, tuple)):
            for child in raw:
                collect(child)

    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            normalized = "".join(c for c in str(key).lower() if c.isalnum())
            sensitive = normalized in {
                "authorization", "proxyauthorization", "authentication", "auth",
                "token", "tokens", "secret", "password", "passwd", "credential",
                "credentials", "cookie", "cookies", "apikey", "key", "privatekey",
                "headers", "extraheaders",
            } or normalized.endswith(("token", "apikey", "password", "secret", "headers"))
            if sensitive:
                collect(child)
                result[key] = "[redacted]"
            else:
                result[key] = _redact_options(child, secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_options(child, secrets) for child in value]
    return value


def read_profile_config(path: Path) -> Config:
    """Read only endpoint/profile fields; never initialize or migrate user settings."""
    cfg = Config()
    template, _ = _config_template()
    data = {**template.unwrap(), **(tomllib.loads(path.read_text()) if path.exists() else {})}
    for key in PROFILE_FIELDS | {"hub_base_url", "hub_api_key_env", "classifier_profiles",
                                 "usage_pricing", "classifier_comparison_limit"}:
        if key in data:
            setattr(cfg, key, deepcopy(data[key]))
    validate_profile_config(cfg)
    if cfg.usage_pricing:
        from autofa.session.usage import validate_pricing

        cfg.usage_pricing = validate_pricing(cfg.usage_pricing)
    return cfg


def profile_readiness(cfg: Config) -> dict[str, Any]:
    """Report configured permission and routing, not unverified model capabilities."""
    validate_profile_config(cfg)
    rows = []
    for profile in [{"name": "baseline"}, *cfg.classifier_profiles]:
        scoped = deepcopy(cfg)
        for key, value in profile.items():
            if key != "name":
                setattr(scoped, key, deepcopy(value))
        endpoint = endpoint_for_config(scoped)
        rows.append({
            "name": profile["name"], "model": endpoint.model, "api": endpoint.api,
            "base_url": endpoint.base_url, "key_env": endpoint.api_key_env,
            "credential_present": (endpoint.api_key_env == NO_AUTH
                                   or resolve_api_key_env(endpoint.api_key_env) is not None),
            "request_options": _redact_options(endpoint.request_options),
            "capabilities": {"text": "unverified", "vision": "unverified",
                             "reasoning": "unverified"},
        })
    return {"profiles": rows, "comparison_limit": cfg.classifier_comparison_limit,
            "alternatives_permitted": len(cfg.classifier_profiles),
            "inference_calls": 0,
            "note": "Only explicitly configured profiles are permitted. Credential presence "
                    "does not establish model access or vision/reasoning capability. "
                    "Use --probe NAME explicitly for a bounded text/API check; "
                    "validate vision on native artifact controls before adoption."}


def propose_profile(cfg: Config, *, name: str, model: str, api: str | None = None,
                    base_url: str | None = None, key_env: str | None = None,
                    request_options: dict | None = None) -> dict:
    """Build a complete, provider-scoped opt-in profile using explicit model selection."""
    scoped = deepcopy(cfg)
    values = {"feature_model": model, "feature_api": api, "feature_base_url": base_url,
              "feature_api_key_env": key_env, "feature_request_options": request_options}
    if base_url is not None and base_url.rstrip("/") != endpoint_for_config(cfg).base_url:
        if key_env is None:
            raise ValueError("A different provider requires an explicit --key-env (or none)")
    for key, value in values.items():
        if value is not None:
            setattr(scoped, key, deepcopy(value))
    endpoint = endpoint_for_config(scoped)
    profile = {"name": name, "feature_model": endpoint.model, "feature_api": endpoint.api,
               "feature_base_url": endpoint.base_url, "feature_api_key_env": endpoint.api_key_env,
               "feature_request_options": endpoint.request_options}
    scoped.classifier_profiles = [*cfg.classifier_profiles, profile]
    validate_profile_config(scoped)
    return profile


def write_profile(path: Path, cfg: Config, profile: dict, *, expected_bytes: bytes | None) -> None:
    """Append one authorized alternative without rewriting unrelated configuration."""
    current = path.read_bytes() if path.exists() else None
    if current != expected_bytes:
        raise ValueError("Configuration changed during setup; inspect it before retrying")
    if current is None:
        path = ensure_default_config(path=path)
        current = path.read_bytes()
    document = tomlkit.parse(current.decode())
    document["classifier_profiles"] = [*cfg.classifier_profiles, profile]
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_config_atomically(path, tomlkit.dumps(document))


def probe_profile(cfg: Config, name: str) -> dict:
    """Explicit bounded inference for one permitted model; never auto-adopt another."""
    scoped = deepcopy(cfg)
    if name != "baseline":
        selected = next((p for p in cfg.classifier_profiles if p["name"] == name), None)
        if selected is None:
            raise ValueError("Probe name must be baseline or an explicitly configured profile")
        for key, value in selected.items():
            if key != "name":
                setattr(scoped, key, deepcopy(value))
    endpoint = endpoint_for_config(scoped)
    from autofa.llm.hub import make_hub_client

    sdk = make_hub_client(base_url=endpoint.base_url, api_key_env=endpoint.api_key_env)
    try:
        # Disable SDK retries. Native route discovery and bounded completion recovery
        # still apply; a successful text probe never certifies visual reasoning.
        client = sdk.with_options(max_retries=0, timeout=20)
        hub = HubLLM(client, model=endpoint.model, api=endpoint.api,
                     request_options=endpoint.request_options, usage_role="classifier_probe")
        result = hub.probe(timeout_s=20)
        from autofa.session.llm_readiness import _safe_error

        secrets = [str(getattr(sdk, "api_key", ""))]
        _redact_options(endpoint.request_options, secrets)
        return {"name": name, "ok": result.ok, "label_api": result.label_api,
                "text_api": result.text_api, "vision": "unverified",
                "reasoning": "unverified",
                "error": _safe_error(result.error, secrets) if result.error else None,
                "adopted": False}
    finally:
        sdk.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="autofa preflight classifiers",
                                    description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--add-profile", metavar="NAME", help="Preview a permitted alternative")
    parser.add_argument("--model", help="Provider deployment/model ID, selected by the user")
    parser.add_argument("--api", choices=LLM_APIS)
    parser.add_argument("--base-url")
    parser.add_argument("--key-env")
    parser.add_argument("--request-options", type=json.loads, metavar="JSON")
    parser.add_argument("--suggest-ladder", action="store_true",
                        help="Preview the Luna/Sol escalation preference without authorizing it")
    parser.add_argument("--luna-model", help="Resolved Luna deployment for the ladder preview")
    parser.add_argument("--sol-model", help="Resolved Sol deployment for the ladder preview")
    parser.add_argument("--write-config", action="store_true",
                        help="Save the previewed alternative")
    parser.add_argument("--probe", metavar="NAME",
                        help="Explicit bounded inference; may incur cost")
    parser.add_argument("--usage-dir", type=Path,
                        help="Probe accounting directory; defaults to CONFIG.probe.usage")
    args = parser.parse_args(argv)
    if (args.luna_model or args.sol_model) and not args.suggest_ladder:
        parser.error("--luna-model and --sol-model require --suggest-ladder")
    if args.suggest_ladder and (args.write_config or args.probe or args.add_profile):
        parser.error("Ladder preview is read-only; agree and configure profiles separately")
    if bool(args.add_profile) != bool(args.model):
        parser.error("--add-profile and --model must be supplied together")
    if args.write_config and not args.add_profile:
        parser.error("--write-config requires --add-profile")
    if args.probe and args.add_profile:
        parser.error("Save/inspect profiles first, then probe one configured name separately")
    if args.usage_dir and not args.probe:
        parser.error("--usage-dir requires --probe")
    path = args.config or config_path(create_home=False)
    try:
        if args.write_config and not path.exists():
            # Reuse first-launch detection and legacy migration before reading
            # endpoint defaults; a sparse profile file would shadow both.
            path = ensure_default_config(path=args.config)
        before = path.read_bytes() if path.exists() else None
        source_path = config_source_path(args.config)
        cfg = read_profile_config(source_path)
        report = profile_readiness(cfg)
        report["config_path"] = str(path)
        report["config_source_path"] = str(source_path)
        if args.suggest_ladder:
            from autofa.session.classifier_profiles import proposed_classifier_ladder

            report["proposed_ladder"] = _redact_options(proposed_classifier_ladder(
                cfg, luna_model=args.luna_model, sol_model=args.sol_model))
        if args.add_profile:
            profile = propose_profile(cfg, name=args.add_profile, model=args.model, api=args.api,
                                      base_url=args.base_url, key_env=args.key_env,
                                      request_options=args.request_options)
            report["proposed_profile"] = _redact_options(profile)
            report["configuration_written"] = args.write_config
            if args.write_config:
                write_profile(path, cfg, profile, expected_bytes=before)
        if args.probe:
            from autofa.session.usage import initialize_run_accounting, snapshot_usage, usage_scope

            usage_dir = (args.usage_dir or path.with_suffix(".probe.usage")).expanduser().resolve()
            usage_dir.mkdir(parents=True, exist_ok=True)
            initialize_run_accounting(usage_dir, cfg.usage_pricing or None)
            with usage_scope(run_dir=usage_dir, role="classifier_probe", phase="preflight",
                             exact_dir=True):
                report["probe"] = probe_profile(cfg, args.probe)
            report["usage_dir"] = str(usage_dir)
            report["autofa_usage"] = snapshot_usage(usage_dir)
            report.pop("inference_calls", None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # Never include credential values, even if a provider included one in an error.
    serialized = json.dumps(report, indent=2)
    if args.probe:
        for row in report["profiles"]:
            key = os.environ.get(row["key_env"], "")
            if key:
                serialized = serialized.replace(key, "[redacted]")
    print(serialized)
    return 0 if not args.probe or report["probe"]["ok"] else 1
