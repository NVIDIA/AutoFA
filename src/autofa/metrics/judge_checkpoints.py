# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-local, content-addressed checkpoints for completed judge stages."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autofa.core.dependencies import review_source
from autofa.session.recovery import RunInterruptedError
from autofa.util.atomic import atomic_write_json

log = logging.getLogger(__name__)
CHECKPOINT_VERSION = 1


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class JudgeInterruptedError(RunInterruptedError):
    """Transport failed; saved reviews remain reusable and no batch score is final."""



class JudgeCheckpoints:
    def __init__(self, run_dir: Path, identity: dict[str, Any]):
        self.archive = (Path(run_dir) / "behavior_judge").resolve()
        self.key = fingerprint({"version": CHECKPOINT_VERSION, **identity})
        self.directory = self.archive / "checkpoints" / self.key

    def contract(self) -> dict[str, Any] | None:
        try:
            value = json.loads((self.directory / "contract.json").read_text())
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def save_contract(self, contract: dict[str, Any]) -> None:
        atomic_write_json(self.directory / "contract.json", contract)

    def _path(self, snapshot_id: str, label: Any, stage: str) -> Path:
        return self.directory / f"{fingerprint([snapshot_id, label, stage])}.json"

    def save(self, *, snapshot_id: str, label: Any, stage: str, directory: Path) -> None:
        # These archives are immutable. A missing/truncated/changed file invalidates reuse.
        directory = directory.resolve()
        files = ("evidence.json", "audit.jsonl", "reply.txt", "coverage.json",
                 "review-verdict.json")
        payload = {
            "directory": str(directory.relative_to(self.archive)),
            "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       for name in files},
        }
        atomic_write_json(self._path(snapshot_id, label, stage), payload)

    def load(self, *, snapshot_id: str, label: Any, stage: str) -> Path | None:
        try:
            value = json.loads(self._path(snapshot_id, label, stage).read_text())
            directory = (self.archive / value["directory"]).resolve()
            if not directory.is_relative_to(self.archive):
                return None
            hashes = value["sha256"]
            if set(hashes) != {"evidence.json", "audit.jsonl", "reply.txt", "coverage.json",
                               "review-verdict.json"}:
                return None
            for name, expected in hashes.items():
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                    return None
            return directory
        except (OSError, ValueError, KeyError, TypeError):
            return None


def write_progress(batch_dir: Path, progress: dict[str, Any], **updates: Any) -> None:
    from autofa.session.status_log import append_status

    progress.update(updates, updated_at=datetime.now(UTC).isoformat())
    atomic_write_json(batch_dir / "progress.json", progress)
    message = (
        f"Judging {progress['name']}: {progress['completed']}/{progress['total']} reviewed "
        f"({progress['reused_stages']} saved stages reused); {progress['state']}"
    )
    if progress.get("active_items"):
        active = progress["active_items"]
        message += f" · {len(active)} active: " + ", ".join(
            f"item {item['index'] + 1} {item['stage']}" for item in active
        )
    elif progress.get("current_stage"):
        message += f" · item {progress['current_index'] + 1}: {progress['current_stage']}"
    try:
        append_status(batch_dir.parent.parent, message, source="judge", context=progress.copy())
    except OSError:
        log.warning("Could not append judge status", exc_info=True)


def pending_judgments(run_dir: Path, functions: list[Any]) -> list[dict[str, Any]]:
    """Latest unfinished batch for each unchanged detector, including killed processes."""
    from autofa.metrics.claim_contract import claim_hash

    current = {
        fn.name: claim_hash(description=fn.description or "",
                            mechanism=fn.mechanism or fn.description or fn.name,
                            source=review_source(fn))
        for fn in functions
    }
    latest = {}
    for path in (Path(run_dir) / "behavior_judge").glob("*/progress.json"):
        try:
            row = json.loads(path.read_text())
            name = row["name"]
            if row.get("claim_hash") != current.get(name):
                continue
            if name not in latest or row["updated_at"] > latest[name]["updated_at"]:
                latest[name] = row
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return [row for row in latest.values() if row["state"] != "complete"]
