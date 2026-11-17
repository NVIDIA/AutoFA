# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional remote-service configuration shared by CLI, menus and clients."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from autofa.defaults import setting_value


def validate_service_address(value: str, *, name: str, origin_only: bool = False) -> str:
    """Normalize an optional HTTP(S) address without credentials or browser state."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    raw = value.strip()
    if not raw:
        return ""
    if any(character.isspace() or ord(character) < 32 for character in raw):
        raise ValueError(f"{name} cannot contain whitespace or control characters")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Enter a valid {name} address and port") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or port == 0:
        raise ValueError(f"{name} must be an absolute HTTP(S) address")
    if (parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError(f"{name} cannot contain credentials, query, or fragment")
    if origin_only and parsed.path not in {"", "/"}:
        raise ValueError(f"{name} must be a server origin, e.g. http://analysis-server:8765")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def eval_server_name(config: Any) -> str:
    value = setting_value(config, "data_api_name")
    if not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("data_api_name must be a single-line string")
    # Earlier defaults used title case; show the current default instead.
    value = value.strip()
    return "EvalRDL remote server" if value in {"", "EvalRDL Remote Server"} else value


def remote_library_enabled(config: Any) -> bool:
    return bool(str(setting_value(config, "remote_library_base_url") or "").strip())


def validate_remote_services(config: Any) -> None:
    config.data_api_base_url = validate_service_address(
        config.data_api_base_url, name="EvalRDL remote server")
    config.data_api_name = eval_server_name(config)
    config.remote_library_base_url = validate_service_address(
        config.remote_library_base_url, name="AutoFA library server",
        origin_only=config.remote_library_mode == "server")


LIBRARY_NOT_CONFIGURED = (
    "No AutoFA Cloud/server address is configured. Set remote_library_base_url in the "
    "config file or open Settings > AutoFA Cloud / library server."
)
EVAL_NOT_CONFIGURED = (
    "No EvalRDL remote server is configured. Set data_api_base_url in the config file, "
    "open Settings > EvalRDL remote server, or pass --api-base."
)
