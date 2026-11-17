# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist successful expensive detector calls across evaluation restarts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Version 5 classifies exact fragments with analyzer omission provenance.
# Share this execution contract with aggregate census reuse, not causal reviews.
DETECTOR_EXECUTION_VERSION = 5


class DetectorCheckpoints:
    def __init__(self, run_dir: Path, *, model: str, endpoint: str,
                 api_key_env: str = "", request_options: dict[str, Any] | None = None,
                 api: str = "chat_completions",
                 routing_identity: Callable[[], dict[str, Any]] | None = None):
        self.path = Path(run_dir) / "detector_results.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Earlier predictions can mistake window markers for original defects.
        self.identity = {"version": DETECTOR_EXECUTION_VERSION,
                         "model": model, "endpoint": endpoint, "api": api,
                         "api_key_env": api_key_env, "request_options": request_options or {}}
        self._routing_identity = routing_identity
        if request_options:
            # Older raw Chat calls ignored these options, even though their cache
            # keys included them. Do not reuse those predictions under enforcement.
            self.identity["request_options_scope"] = "all_detector_chat_calls_v2"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, result TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def key(self, source: str, deps: dict | None, item: dict[str, Any],
            *, bundle: dict | None = None) -> str:
        identity = self.identity
        if self._routing_identity is not None:
            # Auto routing can resolve during the first dispatch. The caller
            # recomputes this key before saving, so a result is stored under
            # its resolved transport rather than the pre-dispatch unknown state.
            identity = {**identity, "routing": self._routing_identity()}
        parts = [identity, source, deps, item]
        if bundle:
            parts.append(bundle)
        payload = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> tuple[Any, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT result FROM results WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
            return value["prediction"], value["localisation"]
        except (ValueError, KeyError, TypeError):
            return None

    def put(self, key: str, prediction: Any, localisation: Any) -> None:
        # Commit every completed item; an interrupted batch cannot lose earlier calls.
        payload = json.dumps({"prediction": prediction, "localisation": localisation})
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO results VALUES (?, ?)", (key, payload))
