# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable, immutable execution graphs for detectors and ordinary helper functions."""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

BUNDLE_VERSION = 1
MAX_NODES = 128
MAX_DEPTH = 24


def digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def call_aliases(source: str, namespace: str, method: str) -> set[str]:
    """New bundles require literal dependency aliases, checked before any execution."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == namespace
            and node.func.attr == method
        ):
            if (
                not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                raise ValueError(f"{namespace}.{method} requires a literal declared alias")
            found.add(node.args[0].value)
    return found


def validate_edges(source: str, helpers: dict, detectors: dict) -> None:
    for namespace, method, declared in (
        ("helpers", "call", helpers),
        ("library", "fire", detectors),
    ):
        missing = call_aliases(source, namespace, method) - set(declared)
        if missing:
            raise ValueError(f"undeclared {namespace} dependencies: {sorted(missing)}")


def validate_bundle(bundle: dict, *, source: str | None = None) -> None:
    if not isinstance(bundle, dict) or bundle.get("version") != BUNDLE_VERSION:
        raise ValueError("unsupported dependency bundle")
    nodes = bundle.get("nodes")
    if not isinstance(nodes, dict) or len(nodes) > MAX_NODES:
        raise ValueError(f"dependency bundle requires at most {MAX_NODES} nodes")
    done: set[str] = set()

    def visit(ref: str, role: str, stack: tuple[str, ...]) -> None:
        if ref in stack or len(stack) >= MAX_DEPTH:
            raise ValueError("dependency cycle or maximum depth exceeded")
        node = nodes.get(ref)
        if not isinstance(node, dict) or digest(node) != ref:
            raise ValueError(f"missing or corrupt dependency revision: {ref}")
        if node.get("role") != role:
            raise ValueError(f"dependency {ref} is not a {role}")
        if ref in done:
            return
        walk(node, (*stack, ref))
        done.add(ref)

    def walk(node: dict, stack: tuple[str, ...]) -> None:
        helpers, detectors = node.get("helpers", {}), node.get("detectors", {})
        if not isinstance(helpers, dict) or not isinstance(detectors, dict):
            raise ValueError("dependency edges must be alias-to-revision objects")
        if node.get("role") == "helper" and detectors:
            raise ValueError("deterministic helpers may depend on helpers, not detectors")
        if "source" in node:
            if not isinstance(node["source"], str):
                raise ValueError("dependency source must be a string")
            functions = {
                n.name for n in ast.parse(node["source"]).body if isinstance(n, ast.FunctionDef)
            }
            if not functions:
                raise ValueError("dependency source must define a function")
            if node.get("role") == "helper":
                from autofa.core.helpers import validate_schema

                if node.get("entrypoint") not in functions:
                    raise ValueError("helper entrypoint must name a top-level function")
                validate_schema(node.get("input_schema", {}))
                validate_schema(node.get("output_schema", {}))
            validate_edges(node["source"], helpers, detectors)
        for edges, role in ((helpers, "helper"), (detectors, "detector")):
            for alias, ref in edges.items():
                if not isinstance(alias, str) or not isinstance(ref, str):
                    raise ValueError("dependency aliases and revisions must be strings")
                visit(ref, role, stack)

    walk(bundle, ())
    if source is not None:
        validate_edges(source, bundle.get("helpers", {}), bundle.get("detectors", {}))
    if done != set(nodes):
        raise ValueError("dependency bundle contains unreachable nodes")


def execution_bundle(function: Any) -> dict | None:
    meta = function.get("meta", {}) if isinstance(function, dict) else getattr(function, "meta", {})
    return (meta or {}).get("dependency_bundle")


def review_source(function: Any) -> str:
    """Code visible to the judge and its freshness hashes, including pinned sources.

    Runtime source remains authored code. Legacy claims retain their existing identity.
    Comments keep this representation parseable by existing specificity analyzers.
    """
    source = function.get("source", "") if isinstance(function, dict) else function.source
    bundle = execution_bundle(function)
    if not bundle:
        return source
    parts = [source, f"\n# AutoFA execution bundle: {digest(bundle)}"]
    for ref, node in sorted(bundle["nodes"].items()):
        parts.append(f"# {node['role']} revision {ref}; entrypoint={node.get('entrypoint')}")
        parts.extend(f"# {line}" for line in node["source"].splitlines())
    return "\n".join(parts) + "\n"


def build_bundle(
    library: Any, *, depends_on: list[str] | None = None, uses_helpers: dict[str, str] | None = None
) -> dict | None:
    """Resolve current catalog names once; consumers subsequently use only snapshots."""
    nodes: dict[str, dict] = {}

    def insert(node: dict) -> str:
        ref = digest(node)
        nodes[ref] = node
        return ref

    detectors = {}
    for name in dict.fromkeys(depends_on or []):
        peer = library.get(name)
        if peer is None:
            raise ValueError(f"depends_on unknown function: {name}")
        saved = execution_bundle(peer)
        if saved:
            validate_bundle(saved, source=peer.source)
            nodes.update(saved["nodes"])
            detectors[name] = insert(
                {
                    "role": "detector",
                    "kind": peer.kind,
                    "source": peer.source,
                    "helpers": saved["helpers"],
                    "detectors": saved["detectors"],
                }
            )
        else:
            # Current detectors pin every library.fire peer, so an unpinned peer is a leaf.
            if call_aliases(peer.source, "library", "fire"):
                raise ValueError(
                    f"depends_on {name!r} calls library.fire without a pinned dependency "
                    "bundle; recreate it with depends_on= in the current AutoFA"
                )
            detectors[name] = insert({
                "role": "detector", "source": peer.source, "kind": peer.kind,
                "helpers": {}, "detectors": {},
            })
    helpers = {}
    for alias, selector in (uses_helpers or {}).items():
        helper = library.helpers.get(selector)
        if helper is None:
            raise ValueError(f"unknown helper {selector!r} for alias {alias!r}")
        nodes.update(helper.bundle.get("nodes", {}))
        nodes[helper.revision] = helper.node()
        helpers[alias] = helper.revision
    if not helpers and not detectors:
        return None
    bundle = {"version": BUNDLE_VERSION, "helpers": helpers, "detectors": detectors, "nodes": nodes}
    validate_bundle(bundle)
    return bundle


def node_bundle(bundle: dict, ref: str) -> dict:
    """Reachable graph for a pinned parent (conditional evaluation / export)."""
    root = bundle["nodes"][ref]
    nodes = {}

    def visit(node: dict) -> None:
        for key in ("helpers", "detectors"):
            for child in node.get(key, {}).values():
                if child not in nodes:
                    nodes[child] = bundle["nodes"][child]
                    visit(nodes[child])

    visit(root)
    return {
        "version": BUNDLE_VERSION,
        "helpers": root.get("helpers", {}),
        "detectors": root.get("detectors", {}),
        "nodes": nodes,
    }
