# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, resumable classifier comparisons; diagnostics never become grades."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

from autofa.defaults import setting_value
from autofa.llm.endpoints import endpoint_for_config
from autofa.session.smoke_work import CALLS_PER_PANEL, LLM_ROWS, SmokeBudgetExceeded, digest
from autofa.util.atomic import atomic_write_json
from autofa.util.cancellation import cancellation_guard

STATE_FILE = "classifier_comparisons.json"
PROFILE_FIELDS = frozenset({
    "feature_model", "feature_api", "feature_base_url", "feature_api_key_env",
    "feature_request_options",
})
_ENDPOINT_FIELDS = frozenset({"feature_base_url", "feature_api_key_env"})
_STATE_LOCK = threading.RLock()
_OPERATIONS: dict[str, threading.Lock] = {}


def validate_profile_config(cfg: Any) -> None:
    """Validate opt-in profiles without constructing clients or reading credentials."""
    profiles = setting_value(cfg, "classifier_profiles")
    limit = setting_value(cfg, "classifier_comparison_limit")
    if type(limit) is not int or not 0 <= limit <= 20:
        raise ValueError("classifier_comparison_limit must be an integer between 0 and 20")
    if not isinstance(profiles, list) or len(profiles) > 8:
        raise ValueError("classifier_profiles must be an ordered list of at most eight profiles")
    names = set()
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) - PROFILE_FIELDS - {"name"}:
            raise ValueError("Classifier profiles accept name and supported feature_* fields only")
        name = profile.get("name")
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name)
                or name == "baseline" or name in names):
            raise ValueError("Classifier profile names must be unique lowercase identifiers")
        names.add(name)
        if (not isinstance(profile.get("feature_model"), str)
                or not profile["feature_model"].strip()):
            raise ValueError("Each classifier profile needs feature_model")
        candidate = copy(cfg)
        if not _ENDPOINT_FIELDS & profile.keys():
            # An inherited endpoint is the baseline's. Its errors surface when a client is
            # created, so an incomplete endpoint never prevents loading the settings.
            for key in (*_ENDPOINT_FIELDS, "hub_base_url", "hub_api_key_env"):
                setattr(candidate, key, "")
        for key, value in profile.items():
            if key != "name":
                setattr(candidate, key, deepcopy(value))
        endpoint_for_config(candidate)


def _settings(cfg: Any) -> dict:
    endpoint = endpoint_for_config(cfg)
    return {"feature_model": endpoint.model, "feature_api": endpoint.api,
            "feature_base_url": endpoint.base_url, "feature_api_key_env": endpoint.api_key_env,
            "feature_request_options": deepcopy(endpoint.request_options)}


def proposed_classifier_ladder(cfg: Any, *, luna_model: str | None = None,
                               sol_model: str | None = None) -> dict:
    """Describe the default preference without granting model access or making calls."""
    tiers = [("baseline", "luna", "auto", "none"),
             ("luna_medium", "luna", "responses", "medium"),
             ("sol_medium", "sol", "responses", "medium"),
             ("sol_high", "sol", "responses", "high")]
    models = {"luna": luna_model, "sol": sol_model}
    result = {"authorized": False, "comparison_limit": 3, "tiers": [
        {"name": name, "model_alias": alias, "model": models[alias], "api": api,
         "reasoning_effort": effort, "capabilities_verified": False}
        for name, alias, api, effort in tiers],
        "note": "Resolve deployments and access, then agree this policy before configuring "
                "alternatives. Baseline auto selects the API, not reasoning effort."}
    if luna_model and sol_model:
        candidate = deepcopy(cfg)
        candidate.classifier_profiles = []
        for name, alias, api, effort in tiers[1:]:
            candidate.classifier_profiles.append({
                "name": name, "feature_model": models[alias], "feature_api": api,
                "feature_request_options": {
                    **deepcopy(setting_value(cfg, "feature_request_options")),
                    "reasoning_effort": effort}})
        validate_profile_config(candidate)
        result["proposed_profiles"] = candidate.classifier_profiles
    return result


def initialize_classifier_policy(cfg: Any, run: Any, *, new_run: bool = False) -> dict:
    """Pin an explicitly configured discovery policy before dispatch, under its owner lock."""
    if getattr(run, "classifier_policy", None):
        return deepcopy(run.classifier_policy)
    if not new_run:
        # Runs without agent analysis pin no detector/auxiliary-judge routes. Loading
        # them cannot turn today's global choices into a historical authorization.
        return {}
    policy, baseline = _policy(cfg), _settings(cfg)
    judge = endpoint_for_config(cfg, "judge")
    saved = {"version": 1, **policy, "baseline_settings": baseline,
             "judge_settings": {"judge_model": judge.model, "judge_api": judge.api,
                                "judge_base_url": judge.base_url,
                                "judge_api_key_env": judge.api_key_env,
                                "judge_request_options": deepcopy(judge.request_options)}}
    previous = getattr(run, "classifier_policy", {})
    run.classifier_policy = saved
    try:
        run.save()
    except BaseException:
        run.classifier_policy = previous
        raise
    return deepcopy(saved)


def apply_run_classifier(cfg: Any, run: Any) -> Any:
    """Restore the durable run-specific selection before constructing any adapters."""
    policy = getattr(run, "classifier_policy", {}) or {}
    if policy:
        if (policy.get("version") != 1 or set(policy.get("baseline_settings", {})) != PROFILE_FIELDS
                or set(policy.get("judge_settings", {})) != {
                    "judge_model", "judge_api", "judge_base_url", "judge_api_key_env",
                    "judge_request_options"}):
            raise ValueError("Invalid saved classifier policy")
        cfg = deepcopy(cfg)
        cfg.classifier_profiles = deepcopy(policy["profiles"])
        cfg.classifier_comparison_limit = policy["comparison_limit"]
        settings = {**policy["baseline_settings"], **policy["judge_settings"]}
        for key, value in settings.items():
            setattr(cfg, key, deepcopy(value))
        validate_profile_config(cfg)
        endpoint_for_config(cfg, "judge")
    saved = getattr(run, "classifier_profile_settings", {}) or {}
    if not saved:
        return cfg
    values = saved.get("settings")
    if (not isinstance(values, dict) or set(values) != PROFILE_FIELDS
            or not isinstance(saved.get("comparison_id"), str)):
        raise ValueError("Invalid saved classifier profile selection")
    inherited = saved.get("inherited_judge_settings", {})
    if (not isinstance(inherited, dict)
            or set(inherited) - {"judge_base_url", "judge_api_key_env", "judge_api"}
            or any(not isinstance(value, str) or not value for value in inherited.values())):
        raise ValueError("Invalid preserved judge endpoint settings")
    result = deepcopy(cfg)
    for key, value in inherited.items():
        if not getattr(result, key, ""):
            setattr(result, key, value)
    for key, value in values.items():
        setattr(result, key, deepcopy(value))
    endpoint_for_config(result)
    endpoint_for_config(result, "judge")
    return result


def _policy(cfg: Any) -> dict:
    validate_profile_config(cfg)
    return {"profiles": deepcopy(setting_value(cfg, "classifier_profiles")),
            "comparison_limit": setting_value(cfg, "classifier_comparison_limit")}


def _read_state(rt: Any) -> dict:
    path = Path(rt.run.run_dir()) / STATE_FILE
    policy = _policy(rt.config)
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("version") != 1 or state.get("run_id") != rt.run.run_id:
            raise ValueError("Classifier comparison checkpoint does not belong to this run")
        if state.get("policy_sha256") != digest(policy):
            raise ValueError("Classifier profile policy changed; preserve this run's saved policy")
        state["policy"] = policy
        return state
    baseline = _settings(rt.config)
    profiles = []
    for configured in policy["profiles"]:
        cfg = copy(rt.config)
        for key, value in configured.items():
            if key != "name":
                setattr(cfg, key, deepcopy(value))
        profiles.append({"name": configured["name"], "settings": _settings(cfg)})
    return {"version": 1, "run_id": rt.run.run_id, "policy": policy,
            "baseline_settings": baseline, "profiles": profiles, "model_calls": 0,
            "comparisons": {}, "adoptions": []}


def _trial_key(identity: dict, expected: dict, abstentions: list[str]) -> str:
    return digest({key: identity[key] for key in (
        "source_identity", "input_identity", "baseline_settings", "baseline_client")} |
        {"expected_outputs": expected, "expected_abstentions": abstentions})


def _next_trial_index(rt: Any, state: dict, trial_key: str) -> int:
    """Advance failed trials on these same controls without adopting their classifier."""
    index = _active_index(rt, state) + 1
    while index < len(state["profiles"]):
        entries = [key for key, record in state["comparisons"].items()
                   if record.get("trial_key") == trial_key
                   and record["profile"] == state["profiles"][index]["name"]]
        if not entries:
            break
        record = read_classifier_comparison(rt, entries[0])
        if record["status"] != "complete" or record["outcome"] != "no_demonstrated_improvement":
            break
        index += 1
    return index


def _save(rt: Any, state: dict) -> None:
    saved = deepcopy(state)
    # Config is the authoritative policy. The ledger retains its binding, not a
    # second independently editable copy.
    saved["policy_sha256"] = digest(saved.pop("policy"))
    atomic_write_json(Path(rt.run.run_dir()) / STATE_FILE, saved)


def _active_index(rt: Any, state: dict) -> int:
    saved = getattr(rt.run, "classifier_profile_settings", {}) or {}
    if not saved:
        return -1
    return next((i for i, p in enumerate(state["profiles"])
                 if p["name"] == saved.get("name") and p["settings"] == saved.get("settings")), -2)


def profile_status(rt: Any) -> dict:
    from autofa.session.classifier_recovery import recovery_status
    from autofa.session.detector_service import detector_service_status

    state = _read_state(rt)
    index = _active_index(rt, state)
    limit = state["policy"]["comparison_limit"]
    return {"active_profile": (state["profiles"][index]["name"] if index >= 0 else "baseline"),
            "available_profiles": [{"name": p["name"], "model": p["settings"]["feature_model"],
                                    "api": p["settings"]["feature_api"]}
                                   for p in state["profiles"][index + 1:]],
            "comparisons_used": len(state["comparisons"]), "comparison_limit": limit,
            "model_calls": state["model_calls"], "model_call_limit": limit * 2 * CALLS_PER_PANEL,
            "enabled": bool(state["profiles"] and limit),
            "service": detector_service_status(rt.run.run_dir()),
            "availability_recovery": recovery_status(rt),
            "comparisons": [{"comparison_id": key, "profile": value["profile"],
                             "status": value["status"], "outcome": value.get("outcome"),
                             "adoption_eligible": value.get("adoption_eligible", False)}
                            for key, value in state["comparisons"].items()],
            "tested_tier_cursors": {
                trial: [key for key, value in state["comparisons"].items()
                        if value.get("trial_key") == trial
                        and value.get("status") == "complete"
                        and value.get("outcome") == "no_demonstrated_improvement"]
                for trial in {value["trial_key"] for value in state["comparisons"].values()
                              if value.get("trial_key")}},
            "scope": "author_controls_not_independent_validation"}


def _source(rt: Any, request: dict) -> dict:
    from autofa.core.dependencies import build_bundle, execution_bundle
    from autofa.mcp.tools_eval import _fn_deps
    from autofa.session.repair_workflow import _version

    name, source = request.get("name"), request.get("source")
    if name is not None:
        if source is not None or request.get("depends_on") or request.get("uses_helpers"):
            raise ValueError("Use name alone for a saved detector, or source with dependencies")
        fn = rt.library.get(name)
        if fn is None:
            raise ValueError("Unknown saved detector")
        return {"source": fn.source, "bundle": execution_bundle(fn), "deps": _fn_deps(fn),
                "version_hash": _version(fn.to_dict())}
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Supply a saved detector name or candidate source")
    bundle = (build_bundle(rt.library, depends_on=request.get("depends_on"),
                           uses_helpers=request.get("uses_helpers"))
              if request.get("depends_on") or request.get("uses_helpers") else None)
    return {"source": source, "bundle": bundle, "deps": None, "version_hash": None}


def _items(rt: Any, ids: list[str]) -> list[dict]:
    from autofa.mcp.tools_eval import _fetch_items_by_id

    values = {str(row["_item_id"]): row
              for row in _fetch_items_by_id(rt.dataset, set(ids), max_items=None)}
    if set(values) != set(ids):
        raise ValueError("Classifier comparison controls are unavailable")
    return [values[item_id] for item_id in ids]


def read_classifier_comparison(rt: Any, comparison_id: str) -> dict:
    """Read a comparison only while its source, controls, artifacts and policy still match."""
    from autofa.session.diagnostic_receipts import read_diagnostic_receipt

    if not isinstance(comparison_id, str) or not re.fullmatch(r"[0-9a-f]{64}", comparison_id):
        raise ValueError("Supply a comparison_id returned by classifier_profiles(compare)")
    state = _read_state(rt)
    record = state["comparisons"].get(comparison_id)
    if record is None or record.get("run_id") != rt.run.run_id:
        raise ValueError("Classifier comparison is unavailable in this run")
    if digest(_source(rt, record["request"])) != record["source_identity"]:
        raise ValueError("Compared detector or its dependencies changed")
    if digest(_items(rt, record["item_ids"])) != record["input_identity"]:
        raise ValueError("Compared control evidence changed")
    for key in ["baseline", "fallback"]:
        result = record.get(key)
        if result and result.get("diagnostic_receipt_id"):
            receipt = read_diagnostic_receipt(rt, result["diagnostic_receipt_id"])
            if (receipt["source_sha256"] != record["source_sha256"]
                    or receipt["item_ids"] != record["item_ids"]
                    or receipt["expected_outputs"] != record["expected_outputs"]
                    or receipt["expected_abstentions"] != record["expected_abstentions"]
                    or receipt["results"] != result.get("results")
                    or receipt["checks_passed"] != result.get("checks_passed")):
                raise ValueError("Compared diagnostic source, controls or results do not match")
        elif result and result.get("ok"):
            raise ValueError("Completed comparison panel is missing its native diagnostic receipt")
    if record["status"] == "complete":
        outcome, eligible = _compare_summary(record["baseline"], record.get("fallback", {}),
                                             record["expected_outputs"])
        if (outcome != record["outcome"] or eligible != record["adoption_eligible"]
                or outcome == "incomplete"):
            raise ValueError("Saved comparison conclusion does not match its diagnostic results")
    return deepcopy(record)


def _compare_summary(baseline: dict, fallback: dict, expected: dict) -> tuple[str, bool]:
    if not baseline.get("ok"):
        return "incomplete", False
    if baseline.get("checks_passed"):
        return "baseline_controls_pass", False
    if not fallback.get("ok"):
        return "incomplete", False
    base = {row["item_id"]: row for row in baseline.get("results", [])}
    other = {row["item_id"]: row for row in fallback.get("results", [])}
    if set(base) != set(other) or not base:
        return "incomplete", False
    preserved = all(not row.get("matches_expected") or other[k].get("matches_expected")
                    for k, row in base.items())
    improved_control = any(k in expected and not row.get("matches_expected")
                           and other[k].get("matches_expected") for k, row in base.items())
    eligible = bool(preserved and improved_control and fallback.get("checks_passed")
                    and not fallback.get("errors"))
    return ("capability_improved" if eligible else "no_demonstrated_improvement"), eligible


def _client(cfg: Any) -> Any:
    from autofa.llm.hub import HubLLM, make_hub_client

    endpoint = endpoint_for_config(cfg)
    return HubLLM(make_hub_client(base_url=endpoint.base_url, api_key_env=endpoint.api_key_env),
                  model=endpoint.model, request_options=endpoint.request_options, api=endpoint.api,
                  usage_role="classifier_probe")


def compare_profiles(rt: Any, *, profile: str, request: dict, diagnosis: str) -> dict:
    from autofa.llm.detection import LLM_CALL_RE
    from autofa.mcp.tools_eval import _smoke_test_feature
    from autofa.session.smoke_work import _model_identity

    ids, expected, abstentions = (request.get("item_ids"), request.get("expected_outputs"),
                                 request.get("expected_abstentions") or [])
    if (not isinstance(ids, list) or not 2 <= len(ids) <= LLM_ROWS
            or any(not isinstance(v, str) or not v for v in ids) or len(ids) != len(set(ids))
            or not isinstance(expected, dict) or any(type(v) is not bool for v in expected.values())
            or set(expected.values()) != {False, True}
            or not isinstance(abstentions, list)
            or any(not isinstance(v, str) or not v for v in abstentions)
            or set(expected) | set(abstentions) != set(ids)
            or set(expected) & set(abstentions)):
        raise ValueError("Use 2–30 unique controls with declared positive and negative Boolean "
                         "expectations; label every control or declare expected abstention")
    # Control order is presentation, not a new experiment or permission to relabel.
    ids, abstentions = sorted(ids), sorted(abstentions)
    request = {**request, "item_ids": ids, "expected_abstentions": abstentions}
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        raise ValueError("Explain the source-supported mismatch and evidence/implementation checks")
    source = _source(rt, request)
    if not LLM_CALL_RE.search(source["source"] + json.dumps(source["bundle"]) +
                              json.dumps(source["deps"])):
        raise ValueError("Classifier comparison requires a detector that uses an LLM")
    items = deepcopy(_items(rt, ids))
    state = _read_state(rt)
    index = _active_index(rt, state)
    target_index = next((i for i, p in enumerate(state["profiles"]) if p["name"] == profile), -1)
    if index < -1 or target_index < 0 or target_index <= index:
        raise ValueError("Compare a permitted profile above the active profile")
    target = state["profiles"][target_index]
    fallback_cfg = deepcopy(rt.config)
    for key, value in target["settings"].items():
        setattr(fallback_cfg, key, deepcopy(value))
    baseline_client, fallback_client = rt.ensure_hub_llm(), _client(fallback_cfg)
    try:
        request = deepcopy(request)
        identity = {"source_identity": digest(source), "input_identity": digest(items),
                    "baseline_settings": _settings(rt.config), "profile": profile,
                    "baseline_client": _model_identity(baseline_client),
                    "fallback_client": _model_identity(fallback_client)}
        comparison_id = digest(identity)
        trial_key = _trial_key(identity, expected, abstentions)
        with _STATE_LOCK:
            state = _read_state(rt)
            record = state["comparisons"].get(comparison_id)
            if record:
                if (record["expected_outputs"] != expected
                        or record["expected_abstentions"] != abstentions):
                    raise ValueError("This comparison's expectations are frozen; keep the original "
                                     "controls and record contrary evidence instead of relabeling")
                record = read_classifier_comparison(rt, comparison_id)
                if record["status"] == "complete":
                    return {**record, "comparison_id": comparison_id, "reused": True}
            else:
                if target_index != _next_trial_index(rt, state, trial_key):
                    raise ValueError("Compare the next configured profile in order; a failed "
                                     "earlier tier needs a completed trial on these same controls")
                if len(state["comparisons"]) >= state["policy"]["comparison_limit"]:
                    raise ValueError("Run classifier comparison budget exhausted")
                record = {**identity, "run_id": rt.run.run_id,
                          "request": request, "item_ids": ids,
                          "expected_outputs": expected, "expected_abstentions": abstentions,
                          "source_sha256": hashlib.sha256(source["source"].encode()).hexdigest(),
                          "diagnosis": diagnosis, "status": "incomplete", "outcome": "incomplete",
                          "trial_key": trial_key,
                          "adoption_eligible": False, "model_calls": 0,
                          "scope": "author_controls_not_independent_validation"}
                state["comparisons"][comparison_id] = record
                _save(rt, state)

        def reserve() -> None:
            with _STATE_LOCK, cancellation_guard():
                saved = _read_state(rt)
                entry = saved["comparisons"][comparison_id]
                if (entry["model_calls"] >= 2 * CALLS_PER_PANEL
                        or saved["model_calls"] >= saved["policy"]["comparison_limit"]
                        * 2 * CALLS_PER_PANEL):
                    raise SmokeBudgetExceeded("Classifier comparison dispatch budget exhausted")
                entry["model_calls"] += 1
                saved["model_calls"] += 1
                _save(rt, saved)

        for side, cfg, client in [("baseline", rt.config, baseline_client),
                                  ("fallback", fallback_cfg, fallback_client)]:
            if record.get(side, {}).get("ok"):
                if side == "baseline" and record[side].get("checks_passed"):
                    break
                continue
            scoped = copy(rt)
            scoped.config, scoped.hub_llm = cfg, client
            result = _smoke_test_feature(scoped, **request, client=client,
                                         frozen_items=items, reserve_model_call=reserve)
            with _STATE_LOCK:
                state = _read_state(rt)
                record = state["comparisons"][comparison_id]
                record[side] = result
                _save(rt, state)
            if not result.get("ok"):
                return {**record, "comparison_id": comparison_id}
            if side == "baseline" and result.get("checks_passed"):
                break
        # Guard source, raw controls, and artifact reads after both executions too.
        read_classifier_comparison(rt, comparison_id)
        with _STATE_LOCK:
            state = _read_state(rt)
            record = state["comparisons"][comparison_id]
            outcome, eligible = _compare_summary(record["baseline"], record.get("fallback", {}),
                                                 expected)
            if record["baseline"].get("checks_passed"):
                outcome = "baseline_controls_pass"
            record.update(outcome=outcome, adoption_eligible=eligible,
                          status="incomplete" if outcome == "incomplete" else "complete")
            _save(rt, state)
        return {**record, "comparison_id": comparison_id}
    finally:
        close = getattr(getattr(fallback_client, "_client", None), "close", None)
        if callable(close):
            close()


def _check_accounts(rt: Any, record: dict, target: dict) -> None:
    """A completed trial cannot authorize adoption under changed endpoint credentials."""
    from autofa.session.smoke_work import _model_identity

    for side, settings in [("baseline", record["baseline_settings"]),
                           ("fallback", target["settings"])]:
        cfg = deepcopy(rt.config)
        for key, value in settings.items():
            setattr(cfg, key, deepcopy(value))
        client = _client(cfg)
        try:
            if _model_identity(client) != record[f"{side}_client"]:
                raise ValueError("Compared classifier credentials or settings changed")
        finally:
            close = getattr(getattr(client, "_client", None), "close", None)
            if callable(close):
                close()


def adopt_profile(rt: Any, comparison_id: str) -> dict:
    with _STATE_LOCK, cancellation_guard():
        state = _read_state(rt)
        record = read_classifier_comparison(rt, comparison_id)
        selected = getattr(rt.run, "classifier_profile_settings", {}) or {}
        already = selected.get("comparison_id") == comparison_id
        if record["status"] != "complete" or not record["adoption_eligible"]:
            raise ValueError("This comparison does not support classifier adoption")
        index = _active_index(rt, state)
        if already:
            target = state["profiles"][index] if index >= 0 else None
        else:
            if record["baseline_settings"] != _settings(rt.config):
                raise ValueError("The run's current classifier differs from the compared baseline")
            trial_key = record.get("trial_key") or _trial_key(
                record, record["expected_outputs"], record["expected_abstentions"])
            trial_index = _next_trial_index(rt, state, trial_key)
            target = (state["profiles"][trial_index]
                      if trial_index < len(state["profiles"]) else None)
        if target is None or target["name"] != record["profile"]:
            raise ValueError("Only the next eligible tested classifier profile may be adopted")
        _check_accounts(rt, record, target)
        # Persist the selected profile for Resume; evaluation owns score updates.
        if not already:
            apply_profile_selection(rt, target, comparison_id, kind="quality_comparison")
        event = {"comparison_id": comparison_id, "profile": target["name"]}
        if event not in state["adoptions"]:
            state["adoptions"].append(event)
        _save(rt, state)
        return {"adopted": True, "already_adopted": already, "profile": target["name"],
                "comparison_id": comparison_id,
                "next_action": "Evaluate detectors with the adopted profile through native scoring "
                               "and review; diagnostic controls do not establish a GOOD grade."}


def apply_profile_selection(rt: Any, target: dict, identifier: str, *, kind: str) -> None:
    """Commit an authorized selection without changing grades, judge routing or budgets."""
    with _STATE_LOCK, cancellation_guard():
        selected = getattr(rt.run, "classifier_profile_settings", {}) or {}
        judge = endpoint_for_config(rt.config, "judge")
        inherited = {key: value for key, value in {
            "judge_base_url": judge.base_url, "judge_api_key_env": judge.api_key_env,
            "judge_api": judge.api,
        }.items() if value and not getattr(rt.config, key, "")}
        inherited = {**selected.get("inherited_judge_settings", {}), **inherited}
        rt.run.classifier_profile_settings = {
            **deepcopy(target), "comparison_id": identifier, "selection_kind": kind,
            "inherited_judge_settings": inherited,
        }
        try:
            rt.run.save()
        except BaseException:
            rt.run.classifier_profile_settings = selected
            raise
        rt.config = apply_run_classifier(rt.config, rt.run)
        rt.hub_llm = rt.llm_readiness = None


def operation_lock(rt: Any) -> threading.Lock:
    with _STATE_LOCK:
        return _OPERATIONS.setdefault(str(Path(rt.run.run_dir()).resolve()), threading.Lock())
