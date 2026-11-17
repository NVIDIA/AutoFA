# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent connection settings for detector and classifier calls."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlsplit

from autofa.defaults import default_value, setting_value

NO_AUTH = "none"
LLM_APIS = ("auto", "decisions", "responses", "chat_completions")
# AutoFA ships no detector endpoint; users or a deployment preset choose one.
NO_ENDPOINT_HELP = (
    "No LLM endpoint is configured for LLM detectors and classifiers. Open Settings > "
    "LLM endpoints > Detector LLM endpoint, or set feature_base_url, feature_api_key_env "
    "and feature_model in autofa.toml."
)


def validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("LLM base URL must be an HTTP(S) API root")
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("LLM base URL must be an HTTP(S) API root without credentials or query")
    if parsed.path.endswith(("/chat/completions", "/decisions", "/responses")):
        raise ValueError("Use the LLM API root, not a /chat/completions, /decisions "
                         "or /responses URL")
    return value


def _api_root(value: Any) -> str:
    """Validate a configured API root; a blank value means no endpoint is configured."""
    if isinstance(value, str) and not value.strip():
        return ""
    return validate_base_url(value)


@dataclass(frozen=True)
class LLMEndpoint:
    base_url: str
    api_key_env: str
    model: str
    request_options: dict[str, Any]
    api: str = field(default_factory=lambda: default_value("feature_api"))


def endpoint_for_config(cfg: Any, role: str = "feature", *, api: str | None = None) -> LLMEndpoint:
    """Resolve connection settings, with an optional workflow-specific API choice.

    A blank ``base_url`` means no endpoint is configured. Resolution still succeeds so
    runs without LLM detectors can start; creating a client reports the missing endpoint.
    """
    if role not in {"feature", "judge"}:
        raise ValueError("LLM endpoint role must be feature or judge")
    feature_url = setting_value(cfg, "feature_base_url") or setting_value(cfg, "hub_base_url")
    base_url = _api_root(
        (setting_value(cfg, "judge_base_url") or feature_url) if role == "judge" else feature_url
    )
    key_env = setting_value(cfg, f"{role}_api_key_env")
    if not key_env:
        if role == "judge" and base_url != _api_root(feature_url):
            raise ValueError("Set judge_api_key_env explicitly when changing judge_base_url")
        key_env = setting_value(cfg, "feature_api_key_env")
        if not key_env:
            if base_url != _api_root(setting_value(cfg, "hub_base_url")):
                raise ValueError(
                    "Set feature_api_key_env explicitly when changing feature_base_url"
                )
            key_env = setting_value(cfg, "hub_api_key_env")
    model = setting_value(cfg, f"{role}_model")
    options = setting_value(cfg, f"{role}_request_options")
    api_setting = "feature_api" if api is None else "api"
    if api is None:
        api = ((setting_value(cfg, "judge_api") or setting_value(cfg, "feature_api"))
               if role == "judge" else setting_value(cfg, "feature_api"))
    if not isinstance(api, str) or api not in LLM_APIS:
        raise ValueError(f"{api_setting} must be auto, decisions, responses or chat_completions")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{role}_model must be a nonempty model ID")
    # An unconfigured endpoint has no key variable to check.
    if (base_url or key_env) and (
            not isinstance(key_env, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env.strip())):
        raise ValueError(f"{role}_api_key_env must name an API key environment variable")
    if not isinstance(options, dict):
        raise ValueError(f"{role}_request_options must be a TOML table")
    completion_options({}, options)  # Validate before constructing a client.
    return LLMEndpoint(base_url, (key_env or "").strip(), model.strip(), dict(options), api)


def improvement_endpoint_for_config(cfg: Any) -> LLMEndpoint:
    """Apply discovery-only defaults without changing shared classifier settings."""
    endpoint = endpoint_for_config(cfg, "judge")
    scoped = setting_value(cfg, "improvement_request_options")
    if not isinstance(scoped, dict):
        raise ValueError("improvement_request_options must be a TOML table")
    completion_options({}, scoped)
    options = completion_options(deepcopy(scoped), endpoint.request_options)
    if "omit_parameters" in endpoint.request_options:
        # Keep the directive for the final request too; resolving defaults here
        # must not reintroduce sampling parameters supplied later by the caller.
        options["omit_parameters"] = deepcopy(endpoint.request_options["omit_parameters"])
    elif "omit_parameters" in scoped:
        options["omit_parameters"] = [
            key for key in scoped["omit_parameters"] if key not in endpoint.request_options
        ]
    return replace(endpoint, request_options=options)


def endpoint_setup_help(cfg: Any, role: str = "feature") -> str:
    """Describe the effective detector credentials without exposing their values."""
    from autofa.config import global_env_path

    settings = ("Open Settings > LLM endpoints > Detector LLM endpoint "
                "to set the API root, model and authentication.")
    try:
        endpoint = endpoint_for_config(cfg, role)
    except ValueError:
        return settings
    if not endpoint.base_url:
        return NO_ENDPOINT_HELP
    if endpoint.api_key_env == NO_AUTH:
        return f"{settings} No API key is required; check that {endpoint.base_url} is running."
    names = endpoint.api_key_env
    if names == "INFERENCE_HUB_API_KEY":
        names += " or INFERENCE_API_KEY"
    return f"{settings} Save {names} in {global_env_path()}, then restart or resume AutoFA."


def completion_options(defaults: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Apply configured options, preserving evidence and unrelated request fields."""
    reserved = {"model", "messages", "input", "instructions", "questions", "stream",
                "api_key", "base_url", "extra_headers",
                "extra_query", "timeout", "max_retries", "n"}
    if reserved.intersection(options):
        raise ValueError("Request options cannot replace model, messages, credentials or transport")
    result = dict(defaults)
    omit = options.get("omit_parameters", [])
    if not isinstance(omit, list) or any(
        not isinstance(k, str) or k not in {"temperature", "top_p"} for k in omit
    ):
        raise ValueError("omit_parameters may contain temperature and top_p only")
    budget_keys = {"max_tokens", "max_completion_tokens", "max_output_tokens"}
    if len(budget_keys.intersection(options)) > 1:
        raise ValueError("Configure only one of max_tokens, max_completion_tokens "
                         "or max_output_tokens")
    for key in budget_keys:
        if key in options:
            if type(options[key]) is not int or options[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
            for alias in budget_keys - {key}:
                result.pop(alias, None)
    extra = options.get("extra_body")
    if extra is not None and (not isinstance(extra, dict) or
            (reserved | budget_keys | {"temperature", "top_p"}).intersection(extra)):
        raise ValueError("extra_body cannot replace core Chat Completions parameters")
    configured_extra = extra or {}
    reasoning = configured_extra.get("reasoning")
    controls_reasoning = (
        "reasoning_effort" in options or "reasoning" in options
        or "reasoning_effort" in configured_extra
        or (isinstance(reasoning, dict) and bool(
            {"enabled", "effort", "max_tokens"}.intersection(reasoning)
        ))
    )
    supplied_extra = result.get("extra_body")
    merged_extra = deepcopy(supplied_extra) if isinstance(supplied_extra, dict) else {}
    # The SDK flattens extra_body after normal parameters. Caller aliases there
    # must not restore omitted sampling or override a configured output budget.
    controlled = set(omit) | ({"temperature", "top_p"} & options.keys())
    if budget_keys.intersection(options):
        controlled.update(budget_keys)
    for key in controlled:
        merged_extra.pop(key, None)
    if controls_reasoning:
        # These Chat/OpenRouter spellings control the same generation setting.
        # A generated detector must not override configured "none" through an
        # extra_body alias (the SDK flattens that body after its normal fields).
        result.pop("reasoning_effort", None)
        result.pop("reasoning", None)
        merged_extra.pop("reasoning_effort", None)
        merged_extra.pop("reasoning", None)
    result.update({k: v for k, v in options.items()
                   if k not in {"omit_parameters", "extra_body"}})
    if isinstance(extra, dict):
        result["extra_body"] = _merge_extra_body(merged_extra, extra)
    elif "extra_body" in options:
        result["extra_body"] = extra
    elif isinstance(supplied_extra, dict):
        result["extra_body"] = merged_extra
    for key in omit:
        result.pop(key, None)
    return result


def _merge_extra_body(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Keep caller routing/format details when endpoint options set nested controls."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_extra_body(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
