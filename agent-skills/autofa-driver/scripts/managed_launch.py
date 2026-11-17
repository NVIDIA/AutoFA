#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own one native subprocess and persist its outcome, including pre-run failures."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import host_controller as host


def run(path: Path) -> int:
    receipt = host.document(path)
    root = Path(receipt["campaign_path"])
    child = None

    def stop(_signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    try:
        with host.records.state_lock(root):
            receipt = host.document(path)
            campaign, config, _ = host.owner_state(root, receipt["owner_generation"])
            if campaign["revision"] != receipt["revision"] or not host.host_available(
                root, receipt["owner_generation"]
            ):
                raise ValueError("Launch authorization/host supervision changed before dispatch")
            spec = receipt["spec"]
            host._validate_spec(spec, config)
            host.validate_delivery(root, spec)
            env = {
                **os.environ,
                **receipt["native"].get("env", {}),
                **spec.get("env", {}),
                "AUTOFA_DRIVER_LAUNCH": str(path),
                "AUTOFA_DRIVER_LAUNCH_ID": receipt["launch_id"],
            }
            with Path(receipt["log_path"]).open("ab") as output:
                child = subprocess.Popen(
                    spec["argv"],
                    cwd=spec["cwd"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=output,
                )
            receipt.update(
                state="running",
                process=host.process_identity(child.pid),
                updated_at_epoch=time.time(),
            )
            host.records.atomic(path, receipt)
        while True:
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                campaign = host.document(root / "campaign.json")
                if (campaign["state"] == "stopped"
                        and not any((path.parent / "bindings").glob("*.json"))):
                    # Preparation can precede a native run/lifecycle file. The
                    # worker still owns this exact Popen handle, so a user stop
                    # cannot leave an unbound download or later analysis running.
                    child.terminate()
                    try:
                        code = child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        code = child.wait(timeout=5)
                    break
        with host.records.state_lock(root):
            receipt = host.document(path)
            bindings = list((path.parent / "bindings").glob("*.json"))
            receipt.update(
                state="exited",
                exit_code=code,
                updated_at_epoch=time.time(),
                run_ids=[host.document(item)["run_id"] for item in bindings],
            )
            if code != 0 or not bindings:
                receipt["failure"] = {
                    "error_type": "NativeExit" if code else "MissingRunBinding",
                    "preparation_failed": not bindings,
                }
            host.records.atomic(path, receipt)
        return code
    except BaseException as exc:
        with host.records.state_lock(root):
            receipt = host.document(path)
            receipt.update(
                state="interrupted" if isinstance(exc, KeyboardInterrupt) else "launch_failed",
                updated_at_epoch=time.time(),
                failure={"error_type": type(exc).__name__},
            )
            host.records.atomic(path, receipt)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]).resolve(strict=True)))
