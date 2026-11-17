# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate hierarchical failure taxonomies."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
TAXONOMY_VERSIONS_DIR = Path(__file__).with_name("versions")
TAXONOMY_V0_PATH = TAXONOMY_VERSIONS_DIR / "taxonomy_v0.json"
TAXONOMY_V1_PATH = TAXONOMY_VERSIONS_DIR / "taxonomy_v1.json"
DEFAULT_TAXONOMY_PATH = TAXONOMY_V1_PATH
# Source license fields do not affect classification, and the shipped taxonomies gained them
# after results had recorded their hashes, so the hash leaves them out.
_UNHASHED_SOURCE_KEYS = frozenset({"license", "license_url"})


class TaxonomyError(ValueError):
    """Raised when a taxonomy does not follow the AutoFA taxonomy format."""


def document_sha256(document: Any) -> str:
    """Hash a taxonomy document independently of key order and source license fields."""
    source = document.get("source") if isinstance(document, dict) else None
    if isinstance(source, dict) and not _UNHASHED_SOURCE_KEYS.isdisjoint(source):
        source = {key: value for key, value in source.items() if key not in _UNHASHED_SOURCE_KEYS}
        document = {**document, "source": source}
    canonical = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class TaxonomyNode:
    id: str
    label: str
    description: str
    code: str | None = None
    fault_side: str | None = None
    related: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()
    children: tuple[TaxonomyNode, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def criterion(self) -> str:
        parts = [self.description.strip()]
        if self.include:
            parts.append("Includes: " + "; ".join(self.include))
        if self.exclude:
            parts.append("Excludes: " + "; ".join(self.exclude))
        if self.examples:
            parts.append("Examples: " + "; ".join(self.examples))
        return " | ".join(part for part in parts if part)

    @classmethod
    def from_dict(cls, data: Any, location: str = "root") -> TaxonomyNode:
        if not isinstance(data, dict):
            raise TaxonomyError(f"{location} must be an object")
        node_id = data.get("id")
        if not isinstance(node_id, str) or not ID_RE.fullmatch(node_id):
            raise TaxonomyError(f"{location}.id must match {ID_RE.pattern!r}; got {node_id!r}")

        def text(name: str, *, required: bool) -> str | None:
            value = data.get(name)
            if value is None and not required:
                return None
            if not isinstance(value, str) or not value.strip():
                qualifier = "a non-empty string" if required else "a non-empty string or null"
                raise TaxonomyError(f"{location}.{name} must be {qualifier}")
            return value.strip()

        def strings(name: str) -> tuple[str, ...]:
            value = data.get(name, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise TaxonomyError(f"{location}.{name} must be a list of strings")
            return tuple(value)

        raw_children = data.get("children", [])
        if not isinstance(raw_children, list):
            raise TaxonomyError(f"{location}.children must be a list")
        return cls(
            id=node_id,
            label=str(text("label", required=True)),
            description=str(text("description", required=True)),
            code=text("code", required=False),
            fault_side=text("fault_side", required=False),
            related=strings("related"),
            include=strings("include"),
            exclude=strings("exclude"),
            examples=strings("examples"),
            remediation=strings("remediation"),
            children=tuple(
                cls.from_dict(child, f"{location}.children[{index}]")
                for index, child in enumerate(raw_children)
            ),
        )


@dataclass(frozen=True)
class Taxonomy:
    name: str
    version: str
    root: TaxonomyNode
    taxonomy_id: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    attribution_rule: str = ""
    components: tuple[dict[str, Any], ...] = ()
    content_sha256: str = ""
    _index: dict[str, TaxonomyNode] = field(init=False, repr=False, compare=False)
    _parents: dict[str, str | None] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise TaxonomyError("taxonomy name must not be empty")
        if not self.version.strip():
            raise TaxonomyError("taxonomy version must not be empty")
        index: dict[str, TaxonomyNode] = {}
        parents: dict[str, str | None] = {}
        codes: dict[str, str] = {}

        def visit(node: TaxonomyNode, parent_id: str | None = None) -> None:
            if node.id in index:
                raise TaxonomyError(f"duplicate node id: {node.id}")
            index[node.id] = node
            parents[node.id] = parent_id
            if node.code:
                if node.code in codes:
                    raise TaxonomyError(
                        f"duplicate node code {node.code!r}: {codes[node.code]} and {node.id}"
                    )
                codes[node.code] = node.id
            if len(node.children) == 1:
                raise TaxonomyError(
                    f"node {node.id!r} has one child; collapse it or add alternatives"
                )
            for child in node.children:
                visit(child, node.id)

        visit(self.root)
        if self.root.is_leaf:
            raise TaxonomyError("taxonomy root must have at least two children")
        for node in index.values():
            for related_id in node.related:
                if related_id == node.id:
                    raise TaxonomyError(f"node {node.id!r} cannot relate to itself")
                if related_id not in index:
                    raise TaxonomyError(f"node {node.id!r} links to unknown node {related_id!r}")
                if node.id not in index[related_id].related:
                    raise TaxonomyError(
                        f"related link must be symmetric: {node.id!r} -> {related_id!r}"
                    )
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_parents", parents)

    @classmethod
    def from_dict(cls, data: Any) -> Taxonomy:
        if not isinstance(data, dict):
            raise TaxonomyError("taxonomy must be an object")
        name, version, taxonomy_id = data.get("name"), data.get("version"), data.get("id", "")
        for key, value in (("name", name), ("version", version), ("id", taxonomy_id)):
            if not isinstance(value, str):
                raise TaxonomyError(f"taxonomy {key} must be a string")
        source = data.get("source", {})
        if not isinstance(source, dict):
            raise TaxonomyError("taxonomy source must be an object")
        attribution_rule = data.get("attribution_rule", "")
        if not isinstance(attribution_rule, str):
            raise TaxonomyError("taxonomy attribution_rule must be a string")
        components = data.get("components", [])
        if not isinstance(components, list) or not all(
            isinstance(item, dict) and isinstance(item.get("id"), str) for item in components
        ):
            raise TaxonomyError("taxonomy components must be a list of objects with an id")
        return cls(
            name=name,
            version=version,
            root=TaxonomyNode.from_dict(data.get("root")),
            taxonomy_id=taxonomy_id,
            source=dict(source),
            attribution_rule=attribution_rule.strip(),
            components=tuple(dict(item) for item in components),
            content_sha256=document_sha256(data),
        )

    @classmethod
    def load(cls, path: str | Path) -> Taxonomy:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaxonomyError(f"cannot load taxonomy {source}: {exc}") from exc
        return cls.from_dict(data)

    def node(self, node_id: str) -> TaxonomyNode:
        try:
            return self._index[node_id]
        except KeyError as exc:
            raise TaxonomyError(f"unknown taxonomy node: {node_id}") from exc

    def walk(self) -> Iterator[TaxonomyNode]:
        yield from self._index.values()

    def leaves(self) -> tuple[TaxonomyNode, ...]:
        return tuple(node for node in self.walk() if node.is_leaf)

    def path_ids(self, node_id: str) -> tuple[str, ...]:
        self.node(node_id)
        path: list[str] = []
        cursor: str | None = node_id
        while cursor is not None:
            path.append(cursor)
            cursor = self._parents[cursor]
        return tuple(reversed(path))

    def labels_for_path(self, node_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self.node(node_id).label for node_id in node_ids)


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    """Load a taxonomy JSON file, or the latest shipped interaction-centric taxonomy."""
    return Taxonomy.load(DEFAULT_TAXONOMY_PATH if path is None else path)
