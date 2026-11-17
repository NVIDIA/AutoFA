# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject overlapping MCP work before it spends budget or changes detector state."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_DETECTOR_TOOLS = frozenset({
    "submit_function", "judge_function", "screen_function", "remove_function",
    "restore_failure_mode", "restore_function_version", "plan_function_repair",
    "defer_function_repair",
})
_LIBRARY_TOOLS = frozenset({
    "driver_control",
    "library_load", "library_save", "clear_library", "advance_round",
    "submit_helper", "retain_as_helper", "deprecate_helper",
    "library_import_candidates", "record_external_inspection", "record_source_audit",
})
_READER_TOOLS = frozenset({"smoke_test_feature", "test_helper", "llm_classify"})
_GUARD = threading.Lock()
_CURRENT: ContextVar[tuple[Path, object] | None] = ContextVar("autofa_operation", default=None)


@dataclass(frozen=True)
class _Scope:
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()

    def conflicts(self, other: _Scope) -> bool:
        return bool(self.writes & (other.reads | other.writes)
                    or self.reads & other.writes)


@dataclass(frozen=True)
class _Operation:
    owner: object
    operation_id: str
    tool: str
    scope: _Scope

    def describe(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id, "tool": self.tool,
            "detectors": sorted(key.removeprefix("detector:")
                                for key in self.scope.writes if key.startswith("detector:")),
            "whole_library": "library" in self.scope.writes,
        }


_ACTIVE: dict[Path, list[_Operation]] = {}


def _names(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {name for name in values if isinstance(name, str) and name}


def _repair_group(directory: Path, names: set[str], plan_id: Any) -> set[str]:
    """Read the atomic ledger snapshot without taking its nested persistence locks.

    An owner and its split candidates share validation reservations and outcomes.
    New children identify their owner through repair_plan_id before submission.
    Admission is held while reading this snapshot, so a related operation cannot
    start between resolving its names and reserving them.
    """
    path = directory / "detector_repairs.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return names
    if data.get("schema_version") != "detector-repair-v1":
        raise ValueError("Unknown repair ledger schema")
    groups = []
    for owner, state in data["repairs"].items():
        history = state["history"]
        if not history:
            continue
        plan = history[-1]
        group = {owner, *plan["candidates"]}
        groups.append(group)
        if plan_id and plan["plan_id"] == plan_id:
            names |= group
    # A candidate may itself own a later repair. Preserve transitive ownership.
    while True:
        expanded = names | set().union(*(group for group in groups if names & group))
        if expanded == names:
            return names
        names = expanded


def _scope(rt: Any, directory: Path, name: str, args: dict[str, Any]) -> _Scope:
    if name == "classifier_profiles":
        if args.get("action", "status") != "compare":
            return _Scope(writes=frozenset({"library", "classifier_profiles"}))
        dependencies = _names(args.get("depends_on")) | _names([args.get("name")])
        return _Scope(
            reads=frozenset({"library", *(f"detector:{dep}" for dep in dependencies)}),
            writes=frozenset({"classifier_profiles"}),
        )
    if (name in _LIBRARY_TOOLS
            or (name == "family_discovery" and args.get("action", "status") != "status")
            or (name == "evaluate_functions" and not args.get("names"))):
        return _Scope(writes=frozenset({"library"}))

    # MCP schemas normally reject these shapes, but saved calls and Python
    # adapters can reach the callable directly (e.g. set(names) accepts a str).
    # Do not run an unresolvable target under an empty detector reservation.
    if ((name in _DETECTOR_TOOLS
         and (not isinstance(args.get("name"), str) or not args["name"]))
            or (name == "evaluate_functions"
                and (not isinstance(args.get("names"), list)
                     or any(not isinstance(target, str) or not target
                            for target in args["names"])))
            or (args.get("depends_on") is not None
                and (not isinstance(args["depends_on"], list)
                     or any(not isinstance(dep, str) for dep in args["depends_on"])))
            or any(args.get(key) is not None and not isinstance(args[key], str)
                   for key in ("within", "explains"))):
        return _Scope(writes=frozenset({"library"}))

    targets = (_names(args.get("names")) if name == "evaluate_functions"
               else _names([args.get("name")]) if name in _DETECTOR_TOOLS else set())
    dependencies = {dep.strip() for dep in _names(args.get("depends_on")) if dep.strip()}
    if name == "smoke_test_feature":
        # Named diagnostics execute the saved version and must not overlap its
        # replacement/restoration. Frozen dependency sources need no live locks.
        dependencies |= _names([args.get("name")])
    for key in ("within", "explains"):
        dependencies |= {dep.strip() for dep in _names([args.get(key)]) if dep.strip()}
    library = getattr(rt, "library", None)
    for target in targets:
        fn = library.get(target) if library is not None else None
        if fn is not None:
            dependencies |= _names(getattr(fn, "depends_on", None))
            dependencies |= _names([getattr(fn, "explains", None)])
    try:
        targets = _repair_group(directory, targets, args.get("repair_plan_id"))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        # Let the tool explain unavailable/corrupt state, without admitting work
        # whose shared repair owner cannot be resolved safely.
        return _Scope(writes=frozenset({"library"}))
    return _Scope(
        reads=frozenset({"library", *(f"detector:{dep}" for dep in dependencies - targets)}),
        writes=frozenset(f"detector:{target}" for target in targets),
    )


@contextmanager
def admit_operation(name: str, args: dict[str, Any]) -> Iterator[dict[str, Any] | None]:
    """Reserve a complete tool lifetime, or yield a busy response without waiting.

    One short registry lock atomically reserves every affected name. It is never
    held while running a tool or taking family/repair/evidence ledger locks.
    Nested calls inherit ownership; unrelated detector operations remain parallel.
    """
    governed = (name in _DETECTOR_TOOLS | _LIBRARY_TOOLS | _READER_TOOLS
                or name == "evaluate_functions"
                or (name == "classifier_profiles" and args.get("action", "status") != "status")
                or (name == "family_discovery" and args.get("action", "status") != "status"))
    if not governed:
        yield None
        return
    from autofa.mcp.runtime import get_runtime
    from autofa.util.cancellation import check_cancelled

    check_cancelled()
    rt = get_runtime()
    directory = Path(rt.run.run_dir()).resolve()
    current = _CURRENT.get()
    owner = current[1] if current is not None and current[0] == directory else object()
    with _GUARD:
        scope = _scope(rt, directory, name, args)
        conflicts = [op for op in _ACTIVE.get(directory, [])
                     if op.owner is not owner and scope.conflicts(op.scope)]
        if not conflicts:
            operation = _Operation(owner, uuid4().hex, name, scope)
            _ACTIVE.setdefault(directory, []).append(operation)
    if conflicts:
        yield {
            "ok": False, "busy": True, "status": "operation_in_progress", "tool": name,
            "error": "Conflicting work is already running; this request was not started.",
            "conflicting_operations": [op.describe() for op in conflicts],
            "hint": "Wait for the existing tool result before deciding whether more work is "
                    "needed. Do not resubmit or launch another review while it is running.",
        }
        return
    token = _CURRENT.set((directory, owner))
    try:
        check_cancelled()
        yield None
    finally:
        _CURRENT.reset(token)
        with _GUARD:
            _ACTIVE[directory].remove(operation)
            if not _ACTIVE[directory]:
                del _ACTIVE[directory]
