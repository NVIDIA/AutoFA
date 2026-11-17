# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, path-free input identity used to group published dataset cards."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


def canonical_origin(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip()
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError("data origin must be an HTTP(S) origin or API prefix")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("data origin cannot contain credentials, query, or fragment")
    port = parsed.port
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is not None and (parsed.scheme, port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    return urlunsplit((parsed.scheme.lower(), host, parsed.path.rstrip("/"), "", ""))


def normalize_data_identity(value: Any) -> dict | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "sources"}
        or type(value.get("version")) is not int
        or value["version"] != 1
        or not isinstance(value.get("sources"), list)
        or not 1 <= len(value["sources"]) <= 100
    ):
        raise ValueError("data_identity requires version 1 and 1..100 sources")
    canonical: dict[str, dict] = {}
    for source in value["sources"]:
        if not isinstance(source, dict):
            raise ValueError("data_identity sources must be objects")
        if source.get("kind") == "evalrdl" and set(source) == {"kind", "origin", "run_id"}:
            run_id = source["run_id"]
            if isinstance(run_id, str):
                run_id = run_id.strip()
            if not isinstance(run_id, str) or not _RUN.fullmatch(run_id):
                raise ValueError("invalid EvalRDL data run_id")
            item = {
                "kind": "evalrdl",
                "origin": canonical_origin(source["origin"]),
                "run_id": run_id,
            }
        elif source.get("kind") == "local" and set(source) == {"kind", "sha256"}:
            if not isinstance(source["sha256"], str) or not _HEX.fullmatch(source["sha256"]):
                raise ValueError("local data sha256 must contain 64 lowercase hex characters")
            item = {"kind": "local", "sha256": source["sha256"]}
        else:
            raise ValueError("unsupported data_identity source or fields")
        canonical[json.dumps(item, sort_keys=True, separators=(",", ":"))] = item
    return {"version": 1, "sources": [canonical[key] for key in sorted(canonical)]}


def data_identity_key(value: dict | None, branch_key: str) -> str:
    if value is None:
        return "unknown:" + hashlib.sha256(branch_key.encode()).hexdigest()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "data:" + hashlib.sha256(encoded.encode()).hexdigest()
