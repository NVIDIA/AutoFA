#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable AutoFA campaign records and explicitly requested native delivery coordination."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

SKILL_ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - host helper supports Python 3.9


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical(value)  # Reject nonfinite JSON numbers before any mutation.
    return value


def object_file(path):
    value = read(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def required_text(value, key):
    if not isinstance(value.get(key), str) or not value[key].strip():
        raise ValueError(f"Required nonempty string: {key}")


def atomic(path, value):
    path = Path(path)
    payload = canonical(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def campaign_path(path):
    root = Path(path).expanduser().resolve()
    if root == SKILL_ROOT or SKILL_ROOT in root.parents:
        raise ValueError("Campaign state must live outside the skill package")
    return root


@contextmanager
def state_lock(root, timeout=2):
    """Serialize helper records only; never claims AutoFA's native run lock."""
    with (root / ".driver.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN} or time.monotonic() >= deadline:
                    raise
                time.sleep(.02)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def merge(before, patch):
    result = dict(before)
    for key, value in patch.items():
        result[key] = (merge(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else value)
    return result


def deadline_epoch(value):
    """Parse a portable date and report invalid inputs as ordinary validation errors."""
    try:
        if not isinstance(value, str):
            raise ValueError("deadline must be a timezone-aware ISO timestamp")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("deadline must include its timezone")
        return parsed.astimezone(timezone.utc).timestamp()  # noqa: UP017 - Python 3.9
    except (TypeError, OverflowError, OSError) as exc:
        raise ValueError("deadline is outside the supported timestamp range") from exc


def money_exceeds(amount, maximum):
    """Ignore floating representation noise, never a material additional allowance."""
    return amount > maximum and not math.isclose(amount, maximum, rel_tol=1e-12, abs_tol=1e-9)


def resource_object(value, name):
    if not isinstance(value, dict):
        raise ValueError(name + " must be an object")
    return value


def nonnegative(value, name):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(name + " must be finite and nonnegative")
    return value


def validate_policy(policy):
    """Validate portable policy fields used by this standalone supervisor."""
    resource_object(policy, "campaign_policy")
    if policy.get("deadline") is not None:
        deadline_epoch(policy["deadline"])
    for key in ("spend_limit_usd", "delivery_reserve_seconds", "delivery_reserve_usd",
                "coverage_target"):
        if policy.get(key) is not None:
            nonnegative(policy[key], "campaign_policy." + key)
    if (policy.get("coverage_target") or 0) > 1:
        raise ValueError("coverage_target must be between zero and one")
    for key, choices in (("mode", {"improve_until", "finish_on_target"}),
                         ("coverage_unit", {"failed_attempts", "failed_tasks"}),
                         ("unknown_cost_policy", {"stop", "allow_estimate"})):
        if key in policy and (not isinstance(policy[key], str) or policy[key] not in choices):
            raise ValueError("Invalid campaign_policy." + key)
    for key in ("round_limit", "checkpoint_every_rounds"):
        if key in policy and (type(policy[key]) is not int or policy[key] < 1):
            raise ValueError("campaign_policy." + key + " must be a positive integer")
    if "overtime" in policy:
        validate_overtime(policy["overtime"])


def validate_overtime(value):
    resource_object(value, "overtime")
    if value.get("when", "minimum_unmet") != "minimum_unmet":
        raise ValueError("overtime.when must be minimum_unmet")
    for key in ("seconds", "usd"):
        nonnegative(value.get(key, 0), "overtime." + key)


def delivery_helper():
    spec = importlib.util.spec_from_file_location(
        "autofa_driver_delivery", Path(__file__).with_name("delivery.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_goal(goal):
    """Validate the resource agreement before any service or paid worker starts."""
    resource_object(goal, "goal")
    required_text(goal, "objective")
    if "delivery" in goal:
        delivery_helper().delivery_contract(goal)
    if goal.get("deadline") is not None:
        deadline_epoch(goal["deadline"])
    policy = resource_object(goal.get("campaign_policy", {}), "campaign_policy")
    validate_policy(policy)
    if "overtime" in goal:
        validate_overtime(goal["overtime"])
    budget = resource_object(goal.get("budgets", {}), "budgets")
    scope = budget.get("cost_scope", "campaign")
    if not isinstance(scope, str) or scope not in {"campaign", "native_only", "unbounded"}:
        raise ValueError("budgets.cost_scope must be campaign, native_only or unbounded")
    cap = budget.get("spend_limit_usd", policy.get("spend_limit_usd"))
    maps = {}
    for key in ("run_allocations_usd", "usage_baselines_usd", "run_overtime_allocations_usd"):
        maps[key] = resource_object(budget.get(key, {}), "budgets." + key)
        for run_id, amount in maps[key].items():
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("Budget allocation keys must be native run IDs")
            nonnegative(amount, key)
    for key in ("spend_limit_usd", "driver_reserve_usd", "delivery_reserve_usd"):
        if key in budget:
            nonnegative(budget[key], "budgets." + key)
    allocated = (math.fsum(maps["run_allocations_usd"].values())
                 + budget.get("driver_reserve_usd", 0) + budget.get("delivery_reserve_usd", 0))
    if cap is not None and money_exceeds(allocated, cap):
        raise ValueError("Run allocations plus driver/delivery reserves exceed the campaign cap")
    overtime = policy.get("overtime", goal.get("overtime", {}))
    if money_exceeds(math.fsum(maps["run_overtime_allocations_usd"].values()),
                     overtime.get("usd", 0)):
        raise ValueError("Run overtime allocations exceed the agreed extension")


def state_action(args):
    root = campaign_path(args.campaign)
    if args.command == "init":
        goal = object_file(args.goal)
        validate_goal(goal)
        root.mkdir(parents=True, exist_ok=True)
    if args.command == "show":
        return object_file(root / "campaign.json")
    completion = None
    if args.command == "complete":
        before = object_file(root / "campaign.json")
        if "delivery" in before["goal"]:
            completion = delivery_helper().completion_evidence(root)
            if not completion["complete"]:
                raise ValueError("Delivery is incomplete: " + "; ".join(completion["reasons"]))
    with state_lock(root):
        path = root / "campaign.json"
        if args.command == "init":
            if path.exists():
                raise ValueError("Campaign already exists; use revise")
            state = {"version": 1, "id": str(uuid4()), "created_at": now(),
                     "goal": goal, "revision": 0, "state": "active", "history": []}
        else:
            state = object_file(path)
            if completion is not None:
                if state["revision"] != before["revision"]:
                    raise ValueError(
                        "Campaign changed during delivery verification; retry completion")
                required_text(vars(args), "reason")
                if state["state"] == "complete":
                    # Rechecking the accepted contract must not create another
                    # revision or require a new user-delivery acknowledgment.
                    return state
                state["completion"] = completion
            required_text(vars(args), "reason")
            event = {"at": now(), "action": args.command, "reason": args.reason}
            if args.command == "revise":
                updated = merge(state["goal"], object_file(args.patch))
                validate_goal(updated)
                event.update(before=state["goal"], after=updated)
                state["goal"] = updated
                state["revision"] += 1
            else:
                target = {"stop": "stopped", "reopen": "active", "complete": "complete"}
                event.update(before=state["state"], after=target[args.command])
                if state["state"] != target[args.command]:
                    # Stop/reopen must fence out actions approved before the stop,
                    # even if no quality-goal field changed in between.
                    state["revision"] += 1
                state["state"] = target[args.command]
                if args.command == "complete" and completion is not None:
                    state["completion"]["completed_revision"] = state["revision"]
            state["history"].append(event)
        state["updated_at"] = now()
        atomic(path, state)
        return state


def absolute(path):
    # Preserve the named symlink so a changed release target invalidates the cache.
    return str(Path(path).expanduser().absolute())


def identify(executable, source_root, config=()):
    found = shutil.which(executable) if not Path(executable).is_file() else executable
    if not found:
        raise ValueError("Executable not found; identify the actual environment first")
    named_executable, named_source = absolute(found), absolute(source_root)
    source = Path(named_source).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Source root must be a directory")
    package = source / "src" / "autofa"
    if not package.is_dir():
        package = source / "autofa" if (source / "autofa").is_dir() else source
    sources = {}
    for path in sorted(package.rglob("*")):
        if any(part in {".git", ".venv", "__pycache__", "node_modules"}
               for part in path.relative_to(package).parts):
            continue
        if path.is_file() and path.suffix in {".py", ".toml", ".md"}:
            sources[str(path.relative_to(package))] = digest(path)
    if not sources:
        raise ValueError("Source root has no inspectable Python/TOML/Markdown files")
    configuration = {absolute(path): digest(path) for path in sorted(config)}
    binding = {
        "executable": named_executable,
        "resolved_executable": str(Path(named_executable).resolve(strict=True)),
        "executable_sha256": digest(named_executable), "source_root": named_source,
        "resolved_package": str(package),
        "source_sha256": hashlib.sha256(canonical(sources)).hexdigest(),
        "config_files": configuration,
    }
    return {**binding, "fingerprint": hashlib.sha256(canonical(binding)).hexdigest(),
            "source_file_count": len(sources), "binding_verified_by_helper": False}


def refresh_identity(identity):
    return identify(identity["executable"], identity["source_root"],
                    identity.get("config_files", {}).keys())


def cache_action(args):
    root = campaign_path(args.campaign)
    with state_lock(root):
        object_file(root / "campaign.json")
        path = root / "capabilities.json"
        if args.command == "cache-get":
            cached = object_file(path)
            current = refresh_identity(cached["identity"])
            if current["fingerprint"] != cached["identity"]["fingerprint"]:
                raise ValueError("Capability cache stale: runtime/source/config changed")
            if any(digest(p) != sha for p, sha in cached["evidence"].items()):
                raise ValueError("Capability cache stale: inspection evidence changed")
            return cached
        identity = object_file(args.identity)
        if refresh_identity(identity)["fingerprint"] != identity["fingerprint"]:
            raise ValueError("Identity is stale; identify the current runtime again")
        mapping = object_file(args.map)
        entries = mapping.get("capabilities")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Map requires a nonempty capabilities list")
        evidence = {}
        for entry in entries:
            for key in ("name", "surface", "purpose"):
                required_text(entry, key)
            if not entry.get("entrypoint") or not isinstance(entry.get("evidence"), list):
                raise ValueError("Each capability needs an entrypoint and evidence paths")
            if not entry["evidence"]:
                raise ValueError("Capability evidence cannot be empty")
            for item in entry["evidence"]:
                p = Path(item).expanduser()
                if not p.is_absolute():
                    p = Path(args.map).resolve().parent / p
                evidence[absolute(p)] = digest(p)
        cached = {"version": 1, "recorded_at": now(), "identity": identity,
                  "map": mapping, "evidence": evidence}
        atomic(path, cached)
        return cached


def record(args):
    root = campaign_path(args.campaign)
    receipt = object_file(args.receipt)
    for key in ("id", "state", "reason"):
        required_text(receipt, key)
    with state_lock(root):
        object_file(root / "campaign.json")
        path = root / f"{args.kind}s.jsonl"
        if path.exists():
            data = path.read_bytes()
            lines = data.split(b"\n")
            offset = 0
            for index, line in enumerate(lines):
                if not line and index == len(lines) - 1:
                    break
                try:
                    old = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if index != len(lines) - 1:
                        raise
                    # Preserve an interrupted append for audit, then recover
                    # only the incomplete tail under the campaign lock.
                    path.with_name(f"{path.name}.torn-{uuid4().hex}").write_bytes(line)
                    with path.open("r+b") as stream:
                        stream.truncate(offset)
                    break
                offset += len(line) + 1
                if (old["receipt"]["id"], old["receipt"]["state"]) == (
                        receipt["id"], receipt["state"]):
                    if canonical(old["receipt"]) != canonical(receipt):
                        raise ValueError("Conflicting receipt for this id/state; "
                                         "reconcile explicitly")
                    return {"recorded": False, "receipt": old}
            else:
                if data and not data.endswith(b"\n"):
                    with path.open("ab") as stream:
                        stream.write(b"\n")
        item = {"recorded_at": now(), "kind": args.kind, "receipt": receipt}
        with path.open("ab") as stream:
            stream.write(canonical(item) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return {"recorded": True, "receipt": item}


def snapshot(args):
    root = campaign_path(args.campaign)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.label):
        raise ValueError("Snapshot label must be a short filename-safe label")
    with state_lock(root):
        object_file(root / "campaign.json")
        parent = root / "snapshots"
        parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
        entries = []
        try:
            for index, value in enumerate(args.file):
                source = Path(value).expanduser().resolve(strict=True)
                with source.open("rb") as stream:
                    magic = stream.read(16)
                if (magic.startswith(b"SQLite format 3")
                        or source.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
                        or source.name.endswith(("-wal", "-shm"))):
                    raise ValueError("Use native locked SQLite backup; "
                                     "snapshots are observations only")
                before = digest(source)
                destination = temporary / f"{index:03d}-{source.name}"
                shutil.copyfile(source, destination)
                if digest(destination) != before or digest(source) != before:
                    raise ValueError("Source changed during snapshot; retry a fresh observation")
                entries.append({"source": str(source), "path": destination.name, "sha256": before})
            manifest = {"version": 1, "created_at": now(), "label": args.label,
                        "kind": "observation", "consistent_native_checkpoint": False,
                        "artifacts": entries}
            atomic(temporary / "manifest.json", manifest)
            destination = parent / f"{args.label}-{uuid4().hex}"
            temporary.rename(destination)
            return {"path": str(destination), **manifest}
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def verify(manifest_path):
    manifest = object_file(manifest_path)
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest requires nonempty artifacts")
    for entry in entries:
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError("Artifact requires a lowercase SHA-256")
        path = Path(entry["path"]).expanduser()
        if not path.is_absolute():
            path = Path(manifest_path).resolve().parent / path
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {path}")
    return {"verified": True, "artifacts": len(entries), "scope": "local_artifact_bytes_only"}


def preferences(args):
    """Explicit user preferences, separate from native config and saved-run identity."""
    path = Path(args.path).expanduser().resolve()
    if not args.settings_file:
        return object_file(path) if path.exists() else {"version": 1, "preferences": {}}
    values = object_file(args.settings_file)
    required_text(vars(args), "reason")

    def reject_secrets(value):
        if isinstance(value, dict):
            if set(value) & {"api_key", "password", "access_token", "refresh_token", "secret"}:
                raise ValueError("Preferences store credential references, never secret values")
            for item in value.values():
                reject_secrets(item)
        elif isinstance(value, list):
            for item in value:
                reject_secrets(item)

    reject_secrets(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with state_lock(path.parent):
        previous = object_file(path) if path.exists() else {"version": 1, "preferences": {}}
        output = {"version": 1, "updated_at": now(),
                  "preferences": merge(previous["preferences"], values),
                  "history": [*previous.get("history", []),
                              {"at": now(), "reason": args.reason,
                               "previous": previous["preferences"]}]}
        atomic(path, output)
        return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("init", "revise", "stop", "reopen", "complete", "show", "cache-put",
                 "cache-get", "record", "snapshot"):
        sub = commands.add_parser(name)
        sub.add_argument("--campaign", required=True)
        if name == "init":
            sub.add_argument("--goal", required=True)
        if name == "revise":
            sub.add_argument("--patch", required=True)
        if name in {"revise", "stop", "reopen", "complete"}:
            sub.add_argument("--reason", required=True)
        if name == "cache-put":
            sub.add_argument("--identity", required=True)
            sub.add_argument("--map", required=True)
        if name == "record":
            sub.add_argument("--kind", choices=("intervention", "checkpoint", "publication",
                                               "review", "delivery", "allocation"),
                             required=True)
            sub.add_argument("--receipt", required=True)
        if name == "snapshot":
            sub.add_argument("--label", required=True)
            sub.add_argument("--file", action="append", required=True)
    sub = commands.add_parser("delivery")
    sub.add_argument("action", choices=(
        "configure", "reconcile", "summary", "check", "prepare-run"))
    sub.add_argument("--campaign", required=True)
    sub.add_argument("--native", help="JSON native executable/config binding")
    sub.add_argument("--run", action="append", default=[])
    sub.add_argument("--summary", help="JSON findings, cost, next actions and final message")
    sub.add_argument("--allow-publish", action="store_true")
    sub.add_argument("--run-spec", help="Native run specification as a JSON object")
    sub.add_argument("--output", help="Path for the generated native JSON run specification")
    sub = commands.add_parser("identify")
    sub.add_argument("--executable", required=True)
    sub.add_argument("--source-root", required=True)
    sub.add_argument("--config", action="append", default=[])
    sub = commands.add_parser("verify")
    sub.add_argument("--manifest", required=True)
    sub = commands.add_parser("preferences")
    sub.add_argument("--path", required=True)
    sub.add_argument("--settings-file")
    sub.add_argument("--reason")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "delivery":
            helper, root = delivery_helper(), campaign_path(args.campaign)
            if args.action == "prepare-run":
                if not args.run_spec or not args.output:
                    raise ValueError("prepare-run requires --run-spec and --output")
                output = helper.prepare_run_config(object_file(root / "campaign.json")["goal"],
                                                   object_file(args.run_spec), Path(args.output))
            elif args.action == "configure":
                if not args.native:
                    raise ValueError("configure requires --native")
                output = helper.configure(root, object_file(args.native), args.run)
            elif args.action == "summary":
                if not args.summary:
                    raise ValueError("summary requires --summary")
                output = helper.prepare_summary(root, object_file(args.summary))
            elif args.action == "check":
                output = helper.completion_evidence(root, args.run or None)
            else:
                if len(args.run) != 1:
                    raise ValueError("reconcile requires exactly one --run")
                output = helper.reconcile_delivery(root, args.run[0],
                                                   allow_publish=args.allow_publish)
        elif args.command == "identify":
            output = identify(args.executable, args.source_root, args.config)
        elif args.command.startswith("cache-"):
            output = cache_action(args)
        elif args.command == "record":
            output = record(args)
        elif args.command == "snapshot":
            output = snapshot(args)
        elif args.command == "verify":
            output = verify(args.manifest)
        elif args.command == "preferences":
            output = preferences(args)
        else:
            output = state_action(args)
        print(canonical(output).decode())
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
