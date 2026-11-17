# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FeatureFunction + FunctionLibrary persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from autofa.core.dependencies import execution_bundle, review_source, validate_bundle
from autofa.core.helpers import HelperLibrary
from autofa.session.review_evidence import (
    empty_history,
    function_review_history,
    merge_histories,
)
from autofa.util.atomic import atomic_write_json

FunctionKind = Literal["regex", "llm", "python"]
FUNCTION_KINDS = get_args(FunctionKind)


@dataclass
class FeatureFunction:
    name: str
    description: str
    source: str
    mechanism: str = ""
    # Natural-language explanation of *what the detector code does* (fields, regex/LLM, …).
    # Distinct from ``mechanism`` (why the model fails).
    function_mechanism: str = ""
    kind: str = "regex"  # regex | llm | python; strategy is separate metadata
    llm_prompt: str | None = None
    # Peer library functions this detector may call via ``library.fire(name, item)``.
    depends_on: list[str] = field(default_factory=list)
    # Optional parent name when this detector explains WHY the parent fires
    # (stratified within-parent eval).
    explains: str | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    example_fail_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    # Detector-specific evidence for each positively labeled item, including passes.
    failure_localisations: dict[str, dict[str, Any]] = field(default_factory=dict)
    uses_helpers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Reject unsupported labels before an import can change a live library.
        if self.kind not in FUNCTION_KINDS:
            raise ValueError(
                f"Invalid function kind {self.kind!r} for detector {self.name!r}; expected "
                "'regex', 'llm' or 'python'. Retired strategy labels ran as Python: set kind "
                "to 'python' or recreate the detector."
            )

    @property
    def review_source(self) -> str:
        return review_source(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureFunction:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        fn = cls(**{k: v for k, v in data.items() if k in known})
        if bundle := execution_bundle(fn):
            validate_bundle(bundle, source=fn.source)
        return fn


@dataclass
class FunctionLibrary:
    version: int = 1
    model_id: str = ""
    benchmark: str = ""
    functions: list[FeatureFunction] = field(default_factory=list)
    helpers: HelperLibrary = field(default_factory=HelperLibrary)
    # Portable union across versions/removals, including Continue/import boundaries.
    review_history: dict[str, Any] = field(default_factory=empty_history)
    # Cursor for deterministic discovery across Continue/export/publication.
    exploration_state: dict[str, int] | None = None
    # Portable append-only inspections, source disputes and transfer receipts.
    contributions: list[dict[str, Any]] = field(default_factory=list)

    def get(self, name: str) -> FeatureFunction | None:
        for fn in self.functions:
            if fn.name == name:
                return fn
        return None

    def _remember_reviews(self) -> None:
        self.review_history = merge_histories(
            self.review_history, function_review_history(self.functions),
        )

    def upsert(self, fn: FeatureFunction) -> None:
        self._remember_reviews()
        for i, existing in enumerate(self.functions):
            if existing.name == fn.name:
                self.functions[i] = fn
                return
        self.functions.append(fn)

    def remove(self, name: str) -> bool:
        self._remember_reviews()
        before = len(self.functions)
        self.functions = [f for f in self.functions if f.name != name]
        return len(self.functions) < before

    def to_dict(self) -> dict[str, Any]:
        self._remember_reviews()
        extended = bool(self.helpers.revisions or any(execution_bundle(f) for f in self.functions))
        return {
            "version": 2 if extended else self.version,
            "model_id": self.model_id,
            "benchmark": self.benchmark,
            "functions": [f.to_dict() for f in self.functions],
            "review_history": self.review_history,
            **({"contributions": self.contributions} if self.contributions else {}),
            **({"exploration_state": dict(self.exploration_state)}
               if self.exploration_state is not None else {}),
            **({"helper_library": self.helpers.to_dict()} if extended else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FunctionLibrary:
        if data.get("version", 1) not in {1, 2}:
            raise ValueError("unsupported function library version")
        from autofa.core.library_transport import unpack_bucket_examples
        from autofa.session.contributions import merge_records

        data = unpack_bucket_examples(data)
        fns = [FeatureFunction.from_dict(x) for x in data.get("functions", [])]
        return cls(
            version=int(data.get("version", 1)),
            model_id=str(data.get("model_id", "")),
            benchmark=str(data.get("benchmark", "")),
            functions=fns,
            helpers=HelperLibrary.from_dict(data.get("helper_library", {})),
            review_history=merge_histories(
                data.get("review_history") or empty_history(complete=False),
                function_review_history(fns),
            ),
            exploration_state=_exploration_state(data.get("exploration_state")),
            contributions=merge_records(data.get("contributions", [])),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        # The envelope is authoritative and portable. The separate catalog is inspectable
        # and recoverable from it if a process stops between these atomic replacements.
        if self.helpers.revisions:
            self.helpers.save(path.with_name("helpers.json"))
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> FunctionLibrary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def catalog_path(self, root: Path) -> Path:
        model = _safe_segment(self.model_id or "unknown")
        bench = _safe_segment(self.benchmark or "all")
        return Path(root) / model / bench / "library.json"


def _safe_segment(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:180]


def _exploration_state(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if (not isinstance(value, dict) or type(value.get("version")) is not int
            or value["version"] != 1 or type(value.get("base_seed")) is not int
            or type(value.get("sequence_index")) is not int or value["sequence_index"] < 0):
        raise ValueError(
            "Invalid exploration state: expected version 1, base_seed and sequence_index",
        )
    return {key: value[key] for key in ("version", "base_seed", "sequence_index")}
