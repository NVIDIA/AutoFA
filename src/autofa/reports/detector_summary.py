# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort natural-language summary of what a detector function does."""

from __future__ import annotations

import ast
import re

_FIELD_NAMES = (
    "response_text",
    "chain_of_thought",
    "reasoning_text",
    "reasoning_content",
    "system_text",
    "prompt_text",
    "question",
    "generation",
    "judge_text",
)


def summarize_function_mechanism(source: str, *, kind: str = "regex") -> str:
    """Fluent 1–3 sentence summary of how the detector labels an item.

    Used as a card fallback when ``function_mechanism`` was not authored.
    Prefer an agent-written ``function_mechanism`` when available.
    """
    src = (source or "").strip()
    if not src:
        return "No detector source available."

    doc = _extract_docstring(src)
    # Prefer a substantive docstring that already explains the how (not a one-liner slogan).
    if doc and len(doc) >= 60 and _looks_like_how(doc):
        return doc

    fields = _fields_read(src)
    uses_re = bool(re.search(r"\bre\.(search|findall|finditer|match|fullmatch|compile)\b", src))
    uses_llm = bool(
        re.search(r"llm_client\.(classify|chat)|chat_completions_safe", src)
    ) or (kind == "llm")
    pattern = _first_regex_literal(src) if uses_re else None

    if fields:
        readable = ", ".join(f"`{f}`" for f in fields[:4])
        if len(fields) > 4:
            readable += ", …"
        head = f"Reads {readable} from the item"
    else:
        head = "Inspects the item's text fields"

    if uses_llm:
        mid = (
            "then asks an LLM classifier whether the claimed failure pattern is present"
        )
    elif uses_re and pattern:
        mid = f"and matches the regex {pattern}"
    elif uses_re:
        mid = "and applies regex / string-match heuristics on that text"
    else:
        mid = "and applies a boolean heuristic on that text"

    prose = f"{head}, {mid}, returning true when the detector fires."
    if doc and len(doc) < 60:
        prose = f"{doc.rstrip('.')}. {prose}"
    return prose


def _looks_like_how(text: str) -> bool:
    low = text.lower()
    return any(
        k in low
        for k in (
            "read",
            "regex",
            "search",
            "match",
            "count",
            "llm",
            "classif",
            "response_text",
            "chain_of_thought",
            "looks for",
            "checks",
            "scans",
        )
    )


def _extract_docstring(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            return " ".join(doc.split())
    return ""


def _fields_read(source: str) -> list[str]:
    found: list[str] = []
    for name in _FIELD_NAMES:
        if re.search(rf"['\"]{re.escape(name)}['\"]", source):
            found.append(name)
    return found


def _first_regex_literal(source: str) -> str | None:
    m = re.search(
        r"re\.(?:search|findall|finditer|match|fullmatch|compile)\(\s*(r?(?:'''[\s\S]*?'''|"
        r'"""[\s\S]*?"""|'
        r"'(?:\\.|[^'\\])*'|"
        r'"(?:\\.|[^"\\])*"))',
        source,
    )
    if not m:
        return None
    lit = m.group(1).strip()
    if len(lit) > 72:
        return lit[:69] + "…"
    return lit
