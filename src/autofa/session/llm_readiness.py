# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visible startup readiness for the endpoint used by semantic LLM detectors."""

from __future__ import annotations

import os
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from autofa.config import global_env_path
from autofa.llm.endpoints import endpoint_for_config, endpoint_setup_help
from autofa.llm.hub import humanize_hub_error, resolve_api_key_env
from autofa.ui.errors import set_action_stage
from autofa.ui.theme import GREEN, WARN


def _redact(message: str, secrets: list[str]) -> str:
    for secret in sorted(set(secrets), key=len, reverse=True):
        if not secret:
            continue
        message = message.replace(secret, "[redacted]")
        # HubProbeResult.error may already have been shortened by humanize_hub_error.
        # Remove any visible beginning of a key cut off by that shortening too.
        for ending in ("…", "..."):
            if message.endswith(ending):
                prefix = message[:-len(ending)]
                for length in range(min(len(secret) - 1, len(prefix)), 3, -1):
                    if prefix.endswith(secret[:length]):
                        message = prefix[:-length] + "[redacted]" + ending
                        break
    return message


def _safe_error(error: BaseException | str, secrets: list[str]) -> str:
    # Redact before humanization truncates the message, preserving status metadata.
    message = _redact(str(error), secrets)
    safe = TimeoutError(message) if isinstance(error, TimeoutError) else RuntimeError(message)
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, (str, int)):
            setattr(safe, attribute, _redact(value, secrets) if isinstance(value, str) else value)
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        safe.body = {key: body[key] for key in ("status", "code") if key in body}
    return _redact(humanize_hub_error(safe), secrets)


def _secrets(rt: Any, hub: Any = None) -> list[str]:
    names = [rt.config.hub_api_key_env, "INFERENCE_HUB_API_KEY", "INFERENCE_API_KEY"]
    from autofa.defaults import setting_value

    names.extend(setting_value(rt.config, field) for field in
                 ("feature_api_key_env", "judge_api_key_env"))
    values = [os.environ.get(name.strip(), "").strip() for name in names
              if isinstance(name, str)]
    for client in (hub, getattr(hub, "_client", None)):
        value = getattr(client, "api_key", None)
        if isinstance(value, str):
            values.append(value)
    return values


def check_llm_readiness(rt: Any, *, console: Console) -> dict[str, Any]:
    """Probe once and report unavailable versus uncertain without disabling a client."""
    set_action_stage("probe detector LLM endpoint")
    model = rt.config.feature_model
    fix = endpoint_setup_help(rt.config)
    result = {
        "model": model, "env_path": str(global_env_path()),
        "configured": False, "available": False, "status": "not_configured",
        "credential_help": fix, "key_env": None,
    }
    result["classifier_profiles"] = [p["name"] for p in
                                     getattr(rt.config, "classifier_profiles", [])]
    if not result["classifier_profiles"]:
        result["classifier_profile_setup"] = (
            "No classifier alternatives permitted. Use autofa preflight classifiers "
            "--add-profile NAME --model MODEL to preview an allowed alternative; "
            "--write-config opts future runs in. No model is enabled automatically.")
    hub = None
    try:
        endpoint = endpoint_for_config(rt.config)
        result["base_url"] = endpoint.base_url
        result["api"] = endpoint.api
        result["model"] = model = endpoint.model
        hub = rt.ensure_hub_llm()
        result["key_env"] = resolve_api_key_env(endpoint.api_key_env)
        if hub is None or getattr(hub, "available", True) is False:
            raise RuntimeError("Configured LLM endpoint client is unavailable")
    except Exception as exc:  # noqa: BLE001 - report the limitation before discovery
        result["error"] = _safe_error(exc, _secrets(rt, hub))
    else:
        result["configured"] = True
        try:
            with Status(Text(f"Probing LLM endpoint ({model})…", style=GREEN), console=console):
                probe = hub.probe()
            if probe.ok:
                result.update(status="slow" if probe.slow else "ready", available=True,
                              latency_s=probe.latency_s)
                result.update({field: getattr(probe, field) for field in ("label_api", "text_api")
                               if getattr(probe, field, None)})
            else:
                result["error"] = _safe_error(probe.error or "Probe failed", _secrets(rt, hub))
        except Exception as exc:  # noqa: BLE001 - a failed probe is not a missing key
            result["error"] = _safe_error(exc, _secrets(rt, hub))
        if result.get("error"):
            error = result["error"].lower()
            rejected = any(token in error for token in (
                "auth failed", "invalid_api_key", "unauthorized", "http 401", "http 403",
                "model unavailable", "model not found", "http 404",
            ))
            result.update(status="unavailable" if rejected else "unverified",
                          available=False if rejected else None)
    status = result["status"]
    if status in {"ready", "slow"}:
        latency = result.get("latency_s")
        timing = f" ({latency:.1f}s)" if latency is not None else ""
        if status == "ready":
            message = (f"LLM endpoint {endpoint.base_url} ready for LLM detectors: "
                       f"{model}{timing}.")
        else:
            message = (f"LLM endpoint {endpoint.base_url} reachable but slow: {model}{timing}. "
                       "LLM detectors remain available; evaluation may take longer.")
        result["message"] = message
        console.print(Text(message, style=GREEN if status == "ready" else WARN))
        route = _routing_message(result)
        if route:
            result["routing_message"] = route
            console.print(Text(route, style=GREEN))
        return result

    if status == "not_configured":
        title = "WARNING: LLM DETECTORS NOT CONFIGURED"
        impact = ("Detector evaluation is limited to regex/Python rules. Semantic LLM "
                  "detectors cannot be evaluated until the LLM client is configured.")
    elif status == "unavailable":
        title = "WARNING: LLM DETECTORS UNAVAILABLE"
        impact = ("The LLM endpoint is configured, but the startup probe was rejected. "
                  "Semantic LLM detectors cannot currently be evaluated with this configuration. "
                  "Regex/Python detectors remain available. Check the key and feature_model.")
    else:
        title = "WARNING: LLM DETECTOR READINESS UNVERIFIED"
        impact = ("The LLM endpoint is configured, but its startup probe failed. "
                  "LLM detector calls remain enabled; they may fail until connectivity or "
                  "service availability recovers. Regex/Python detectors remain available.")
    result["message"] = impact
    body = Text(impact, style="bold")
    body.append(f"\n\nReason: {result['error']}\n\n{fix}\n\n")
    body.append("Endpoint credentials are separate from harness credentials used for discovery "
                "and behavior judging.")
    console.print(Panel(body, title=Text(title, style=f"bold {WARN}"),
                        border_style=WARN, padding=(1, 2), expand=True))
    return result


def _routing_message(readiness: dict[str, Any]) -> str:
    label_api = readiness.get("label_api")
    labels = {"responses": "Responses", "chat_completions": "Chat Completions"}
    if label_api == "decisions":
        message = "Decisions API available; using it for LLM detector classifications."
        message += " Optional report highlights are deferred until finalization."
        return message
    if label_api in labels:
        if readiness.get("api") == "auto":
            return ("Decisions API unavailable for this endpoint/model; "
                    f"using {labels[label_api]} for LLM detectors.")
        return f"{labels[label_api]} API selected; using it for LLM detectors."
    return ""


def recheck_saved_routing(rt: Any, *, console: Console) -> None:
    """Repeat discovery's endpoint probe before a fresh finalizer compares score contexts."""
    from autofa.mcp.tools_eval import is_llm_detector
    from autofa.session.recovery import read_json

    if getattr(rt, "llm_readiness", None) is not None:
        return
    saved = read_json(rt.run.run_dir() / "llm_readiness.json")
    if saved.get("available") is not True or saved.get("api") not in {
        "auto", "responses", "decisions",
    }:
        return
    endpoint = endpoint_for_config(rt.config)
    if (saved.get("base_url"), saved.get("model"), saved["api"]) != (
        endpoint.base_url, endpoint.model, endpoint.api
    ):
        return
    if any(is_llm_detector(fn) for fn in rt.library.functions):
        # Capability observations belong to this client. Reproduce them with the
        # configured endpoint instead of copying a prior process's route flags.
        rt.llm_readiness = check_llm_readiness(rt, console=console)


def check_and_recover_readiness(rt: Any, *, console: Console,
                               probe: Any = None) -> dict[str, Any]:
    """Verify required capabilities, then use only the saved availability-recovery ladder."""
    from autofa.session.classifier_recovery import recover_classifier, startup_controls
    from autofa.session.detector_service import record_readiness
    from autofa.session.smoke_work import SmokeBudgetExceeded
    from autofa.session.usage import usage_scope
    from autofa.setup_contract import required_capabilities
    from autofa.util.atomic import atomic_write_json
    from autofa.util.model_budget import model_call_budget

    with usage_scope(run_dir=rt.run.run_dir(), role="classifier_probe", phase="discovery",
                     round_index=rt.run.round):
        readiness = (probe or check_llm_readiness)(rt, console=console)
        if not readiness:
            return readiness
        capabilities = required_capabilities(rt.run.run_dir())
        if readiness.get("available") is True and "vision" in capabilities:
            calls = 0

            def reserve() -> None:
                nonlocal calls
                if calls >= 12:
                    raise SmokeBudgetExceeded("Startup capability probe allowance exhausted")
                calls += 1

            try:
                with model_call_budget(reserve):
                    controls = startup_controls(rt.ensure_hub_llm(), capabilities=capabilities)
                readiness["capability_controls"] = controls
                if not controls["checks_passed"]:
                    readiness.update(available=False, status="unverified",
                                     error="Required startup capability controls did not pass")
            except (Exception, SmokeBudgetExceeded) as exc:
                readiness.update(available=False, status="unverified",
                                 error=_safe_error(exc, _secrets(rt)))
        required = bool(getattr(rt.run, "require_llm_detectors", False)
                        or getattr(rt.config, "classifier_profiles", []) or capabilities)
        record_readiness(rt, readiness,
                         blocking=required or readiness.get("available") is True)
        if readiness.get("available") is not True and required:
            from autofa.session.classifier_profiles import operation_lock

            with operation_lock(rt):
                recovery = recover_classifier(rt, startup=True)
            readiness["availability_recovery"] = recovery
            if recovery.get("recovered"):
                endpoint = endpoint_for_config(rt.config)
                readiness.update(available=True, configured=True, status="ready",
                                 model=endpoint.model, base_url=endpoint.base_url, api=endpoint.api,
                                 message="Approved detector service passed availability controls; "
                                         "detector quality still requires native validation.")
                for key in ("error", "label_api", "text_api"):
                    readiness.pop(key, None)
                console.print(Text(readiness["message"] + f" Model: {endpoint.model}.",
                                   style=GREEN))
        atomic_write_json(rt.run.run_dir() / "llm_readiness.json", readiness)
        return readiness


def llm_readiness_guidance(readiness: dict[str, Any]) -> str:
    """Give the discovery agent the same availability limits shown to the user."""
    if not readiness:
        return ""
    api = readiness.get("label_api") or readiness.get("api", "auto")
    text = (f"\n\nLLM detector backend: {readiness['status']} "
            f"(feature_model={readiness['model']}, label_api={api}). {readiness['message']}")
    if readiness.get("available") is True:
        text += (" Semantic LLM detectors are available. Use classify_options for fixed labels. "
                 "Do not generate quotes or rationales during scoring merely to justify a label; "
                 "optional supporting passages are retrieved on reviewer request "
                 "or at finalization. "
                 "Use classify only when generated content actually determines the verdict. "
                 "Existing extraction-based detectors require revision and revalidation before "
                 "removing those checks.")
    profiles = readiness.get("classifier_profiles")
    if profiles:
        text += (f" Configured classifier alternatives: {profiles}. Use native "
                 "classifier_profiles status/compare/adopt on source-supported controls; "
                 "comparison controls do not establish a grade. A completed unsuccessful "
                 "trial permits testing the next approved tier on the same frozen controls "
                 "without adopting the failed tier. Resume incomplete trials first.")
        text += (" When the selected service is unavailable, use classifier_profiles(recover) "
                 "with unchanged source-supported positive/negative controls. Availability "
                 "recovery is separate from paired quality comparisons and remains bounded.")
    if api == "decisions" and readiness.get("available") is True:
        text_api = readiness.get("text_api", "the configured generation API")
        text += (" classify_options uses the Decisions API; decide supports predicate, "
                 "choice and score questions. Handle refusals as unknown. "
                 f"Necessary free text uses {text_api}.")
    else:
        text += (" Native Decisions access has not been selected or verified; use classify_options "
                 "for semantic labels through the configured API instead of decide.")
    if readiness.get("available") is not True:
        text += (f" Reason: {readiness.get('error', 'not verified')}. "
                 f"{readiness['credential_help']} "
                 "Report this limitation when discussing detector evaluation; do not treat "
                 "an unavailable LLM backend as evidence that a semantic detector has no matches. "
                 "Discovery and behavior judging use separate harness credentials.")
    return text
