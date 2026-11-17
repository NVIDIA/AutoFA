# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector field policy — prevent metadata/label leakage (cheat signals).

Default mode ``content``: detectors may only read input / CoT / trace / answer
style text. Grader JSON, judge_text, patch/test status, identity columns, and
outcome labels are blocked so functions stay portable to SFT / training data.

Imported trace records also retain execution metadata: tool status, timing, usage,
provider metadata and artifact references. These are available through autofa.trace;
they are not external task targets. Import filtering excludes external outcomes
and credentials. Discovery/evidence review must reject circular labels and coarse
length/count/timing proxies even when their underlying measurements are accessible.

Derived trajectory surfaces (``actions_text``, ``observations_text``,
``final_message``) are allowed under content mode because they are extracted
from the model's own trajectory content — not grader metadata. Note that
``observations_text`` / ``actions_text`` are eval-harness scaffolding; an SFT
trace may lack them, so detectors that *require* those surfaces are less
portable than ones that also fall back to ``response_text`` / CoT.

Users can switch to ``metadata`` or ``custom`` in the active AutoFA config.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from autofa.defaults import default_value, setting_value

# Outcome labels — always banned regardless of mode.
LABEL_BANNED: frozenset[str] = frozenset(
    {
        "is_correct",
        "accuracy",
        "symbolic_correct",
        "pop_pass_rate",
        "pop_n_models",
        "peer_outcomes",
        "flipped_models",
        "n_passing_peers",
        "pass_rate",
        "pass_bucket",
        "n_runs",
        "n_passes",
        "label",
        "y_true",
        "ground_truth_correct",
    }
)

# Portable content the detector may use under default ``content`` mode.
DEFAULT_CONTENT_FIELDS: frozenset[str] = frozenset(
    {
        # input / prompt
        "system_text",
        "prompt_text",
        "question",
        "problem",
        "input",
        "prompt",
        "user_message",
        "instruction",
        "input_text",
        "input_messages",
        "input_tools",
        "input_parse",
        "input_json",
        "tools",
        # answer / final response
        "response_text",
        "generation",
        "answer",
        "final_answer",
        "completion",
        "output",
        "model_answer",
        # CoT / reasoning / trace
        "chain_of_thought",
        "reasoning_text",
        "reasoning_content",
        "cot",
        "trace",
        "thinking",
        "thoughts",
        "scratchpad",
        # Derived trajectory surfaces (from response_text JSON via derive_trace_fields)
        "actions_text",
        "observations_text",
        "final_message",
        # optional dialogue without grader
        "messages",
        "conversation",
        "dialog",
    }
)

# Common metadata / grader fields — blocked in ``content`` mode (cheat-prone).
DEFAULT_METADATA_FIELDS: frozenset[str] = frozenset(
    {
        "judge_text",
        "hb_scoring_details_json",
        "scoring_details",
        "scoring_details_json",
        "grader",
        "grader_text",
        "report",
        "tests_status",
        "patch_successfully_applied",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "PASS_TO_FAIL",
        "model_id",
        "benchmark",
        "problem_id",
        "run_idx",
        "hb_problem_idx",
        "split",
        "dataset",
        "source_file",
        "uuid",
        "item_id",
        "_item_id",
    }
)


@dataclass(frozen=True)
class FieldPolicy:
    """Resolved detector field access policy."""

    mode: str  # content | metadata | custom
    allowed: frozenset[str]
    banned: frozenset[str]
    allowlist: bool  # True = only ``allowed`` (minus banned); False = ban-list only

    def is_allowed(self, field: str) -> bool:
        if field in self.banned:
            return False
        if not self.allowlist:
            return True
        return field in self.allowed


def policy_from_settings(
    *,
    detector_fields: str | None = None,
    allow_metadata: bool | None = None,
    allowed_fields: Iterable[str] | None = None,
    extra_allowed_fields: Iterable[str] | None = None,
    banned_fields: Iterable[str] | None = None,
) -> FieldPolicy:
    """Build a FieldPolicy from config knobs."""
    if allow_metadata is None:
        allow_metadata = default_value("allow_metadata")
    if allowed_fields is None:
        allowed_fields = default_value("allowed_fields")
    if extra_allowed_fields is None:
        extra_allowed_fields = default_value("extra_allowed_fields")
    if banned_fields is None:
        banned_fields = default_value("banned_fields")
    mode = (detector_fields or default_value("detector_fields")).strip().lower()
    if allow_metadata and mode == "content":
        mode = "metadata"

    extra_banned = frozenset(banned_fields or ())
    banned = LABEL_BANNED | extra_banned

    custom = frozenset(f for f in (allowed_fields or ()) if f)
    extra = frozenset(f for f in (extra_allowed_fields or ()) if f)

    if mode == "metadata":
        # Ban labels only; metadata/grader fields allowed (user opted in).
        return FieldPolicy(
            mode="metadata",
            allowed=frozenset(),
            banned=banned,
            allowlist=False,
        )

    if mode == "custom":
        allowed = custom or DEFAULT_CONTENT_FIELDS
        return FieldPolicy(
            mode="custom",
            allowed=allowed | extra,
            banned=banned,
            allowlist=True,
        )

    # Default: content-only allowlist.
    return FieldPolicy(
        mode="content",
        allowed=DEFAULT_CONTENT_FIELDS | custom | extra,
        banned=banned | DEFAULT_METADATA_FIELDS,
        allowlist=True,
    )


def get_field_policy(cfg: Any | None = None) -> FieldPolicy:
    """Resolve policy from a Config object or the active AutoFA config."""
    if cfg is None:
        from autofa.config import load_config

        cfg = load_config()
    return policy_from_settings(
        detector_fields=setting_value(cfg, "detector_fields"),
        allow_metadata=bool(setting_value(cfg, "allow_metadata")),
        allowed_fields=setting_value(cfg, "allowed_fields") or [],
        extra_allowed_fields=setting_value(cfg, "extra_allowed_fields") or [],
        banned_fields=setting_value(cfg, "banned_fields") or [],
    )


class _FieldAccessVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found: set[str] = set()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "item":
            key = _const_str(node.slice)
            if key:
                self.found.add(key)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "item":
                if node.args:
                    key = _const_str(node.args[0])
                    if key:
                        self.found.add(key)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "item":
                if len(node.args) >= 2:
                    key = _const_str(node.args[1])
                    if key:
                        self.found.add(key)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "item":
            # item.get(...) is handled in visit_Call — do not treat "get" as a field.
            if node.attr not in {"get", "keys", "values", "items", "copy", "setdefault"}:
                self.found.add(node.attr)
        self.generic_visit(node)


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def referenced_fields(source: str) -> set[str]:
    """All literal ``item[...]`` / ``item.get(...)`` field names in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    visitor = _FieldAccessVisitor()
    visitor.visit(tree)
    return visitor.found


def scan_banned_fields(source: str, policy: FieldPolicy | None = None) -> list[str]:
    """Return disallowed field names referenced in detector source."""
    pol = policy or get_field_policy()
    found = referenced_fields(source)
    bad = {f for f in found if not pol.is_allowed(f)}
    return sorted(bad)


def wrap_item_ban_sentinels(
    item: dict[str, Any],
    policy: FieldPolicy | None = None,
) -> dict[str, Any]:
    """Copy item; replace banned / non-allowed fields with raising sentinels."""

    class _Ban:
        def __init__(self, name: str) -> None:
            self.name = name

        def __bool__(self) -> bool:
            raise RuntimeError(f"banned field access: {self.name}")

        def __str__(self) -> str:
            raise RuntimeError(f"banned field access: {self.name}")

        def __repr__(self) -> str:
            return f"<banned:{self.name}>"

        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(f"banned field access: {self.name}")

        def __getitem__(self, key: Any) -> Any:
            raise RuntimeError(f"banned field access: {self.name}")

    pol = policy or get_field_policy()
    out = dict(item)
    for name in list(out.keys()):
        if not pol.is_allowed(name):
            out[name] = _Ban(name)
    # Ensure known banned keys exist as sentinels even if absent (catch .get).
    for name in pol.banned:
        if name not in out:
            out[name] = _Ban(name)
    return out


def policy_summary(policy: FieldPolicy | None = None) -> str:
    """One-line summary for prompts / help."""
    pol = policy or get_field_policy()
    if pol.mode == "metadata":
        return (
            "detector_fields=metadata — grader/judge fields allowed; "
            "outcome labels still banned"
        )
    sample = ", ".join(sorted(pol.allowed)[:8])
    more = f" (+{len(pol.allowed) - 8} more)" if len(pol.allowed) > 8 else ""
    return (
        f"detector_fields={pol.mode} — only content fields "
        f"({sample}{more}); no judge/grader/metadata"
    )
