# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable per-run execution capacity, separate from model and scoring policy."""

from __future__ import annotations

import json
import tomllib
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autofa.config import Config, config_path
from autofa.defaults import setting_value
from autofa.session.recovery import run_lock
from autofa.util.atomic import atomic_write_json
from autofa.util.capacity import CAPACITY_BOUNDS, effective_llm_workers, validate_capacity

SETTINGS_FILE = "scoring_settings.json"


def _validate_settings(settings: Any) -> dict[str, int]:
    if not isinstance(settings, dict):
        raise ValueError("Capacity settings must be an object")
    return validate_capacity(settings)


def _read_settings(run_dir: Path, run_id: str) -> dict[str, Any]:
    path = run_dir / SETTINGS_FILE
    if not path.exists():
        return {"version": 1, "run_id": run_id, "settings": {}, "events": []}
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or type(record.get("version")) is not int
            or record["version"] != 1 or record.get("run_id") != run_id
            or not isinstance(record.get("events"), list)):
        raise ValueError("Capacity settings do not belong to this run or have an invalid version")
    _validate_settings(record.get("settings"))
    return record


def apply_run_capacity(cfg: Any, run: Any) -> Any:
    """Overlay only saved operational overrides, without mutating config or run state."""
    record = _read_settings(Path(run.run_dir()), run.run_id)
    if not record["settings"]:
        return cfg
    scoped = deepcopy(cfg)
    for name, value in record["settings"].items():
        setattr(scoped, name, value)
    return scoped


def effective_capacity(cfg: Any) -> dict[str, Any]:
    """Requested and enforced limits; measured queue throughput remains separate."""
    settings = _validate_settings({name: setting_value(cfg, name) for name in CAPACITY_BOUNDS})
    workers = effective_llm_workers(cfg)
    return {**settings, "effective_llm_eval_workers": workers,
            "queue_bytes": settings["llm_eval_queue_mib"] * 1024 * 1024,
            "limiting_reason": ("configured_ceiling" if workers < settings["llm_eval_workers"]
                                else None)}


def _current_capacity_config() -> tuple[Config, Path]:
    """Read capacity defaults only; configuring a run must not migrate global settings."""
    cfg, path = Config(), config_path(create_home=False)
    if path.is_file():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        saved = _validate_settings({key: data[key] for key in CAPACITY_BOUNDS if key in data})
        for name, value in saved.items():
            setattr(cfg, name, value)
    return cfg, path


def set_run_capacity(run_id: str, settings: dict[str, int]) -> dict[str, Any]:
    """Set capacity for an inactive saved run; Resume applies it before any execution.

    Unspecified keys keep their prior per-run override or inherit runtime config.
    Effective values in this receipt use the current configuration as a baseline;
    the resumed runtime's capacity observation records what actually runs.
    """
    from autofa.session.supervision import resolve_run_dir

    requested = _validate_settings(settings)
    directory = resolve_run_dir(run_id)
    with run_lock(directory):
        saved_run = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if saved_run.get("run_id") != run_id:
            raise ValueError("Saved run identity does not match the requested run")
        record = _read_settings(directory, run_id)
        cfg, baseline_path = _current_capacity_config()
        inherited = {name: setting_value(cfg, name) for name in CAPACITY_BOUNDS}
        before = {**inherited, **record["settings"]}
        updated = {**record["settings"], **requested}
        changed = sorted(name for name, value in requested.items()
                         if record["settings"].get(name) != value)
        for name, value in updated.items():
            setattr(cfg, name, value)
        effective = effective_capacity(cfg)
        receipt = {"version": 1, "run_id": run_id, "settings": updated,
                   "changed_fields": changed, "effective_values": effective,
                   "configuration_reference": str(baseline_path),
                   "activation": "next_runtime_load", "global_config_changed": False,
                   "scope": "operational_capacity_only"}
        if not changed:
            return {**receipt, "already_applied": True}
        event = {"sequence": len(record["events"]) + 1,
                 "recorded_at": datetime.now(UTC).isoformat(), "requested": requested,
                 "changed_fields": changed, "previous_effective_settings": before,
                 "effective_values": effective, "configuration_reference": str(baseline_path)}
        record.update(settings=updated, events=[*record["events"], event])
        atomic_write_json(directory / SETTINGS_FILE, record)
        return {**receipt, "already_applied": False, "sequence": event["sequence"]}
