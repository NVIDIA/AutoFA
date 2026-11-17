# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read configurable defaults from the packaged TOML, without importing Config."""

from __future__ import annotations

import tomllib
from copy import deepcopy
from dataclasses import Field, field
from functools import lru_cache
from importlib.resources import files
from typing import Any

_DEFAULTS_TEXT = files("autofa").joinpath("defaults.toml").read_text(encoding="utf-8")


def defaults_text() -> str:
    """Canonical defaults used for fresh configs; reload AutoFA after editing TOML."""
    return _DEFAULTS_TEXT


@lru_cache(maxsize=4)
def _parse_defaults(text: str) -> dict[str, Any]:
    return tomllib.loads(text)


def default_value(name: str) -> Any:
    """Return an isolated setting value, with TOML's automatic seed normalized."""
    value = deepcopy(_parse_defaults(defaults_text())[name])
    if name == "exploration_seed" and value == "auto":
        return None
    return value


def default_field(name: str) -> Field[Any]:
    """Dataclass factory so mutable defaults and edited TOML are never cached in fields."""
    return field(default_factory=lambda: default_value(name))


def setting_value(cfg: Any, name: str) -> Any:
    """Read a configured value, preserving explicit empty/false/zero overrides."""
    if isinstance(cfg, dict):
        return cfg[name] if name in cfg else default_value(name)
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    return default_value(name)


def harness_default(harness: str, name: str, cfg: Any = None) -> Any:
    """Read a configured harness profile, falling back to its bundled TOML profile."""
    profiles = setting_value(cfg, "harness_defaults")
    profile = profiles.get(harness, {})
    if name in profile:
        return deepcopy(profile[name])
    bundled = default_value("harness_defaults").get(harness, {})
    return deepcopy(bundled[name]) if name in bundled else default_value(name)
