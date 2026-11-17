# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight Python syntax highlighting for prompt_toolkit UIs."""

from __future__ import annotations


def highlight_python(source: str) -> list[tuple[str, str]]:
    """Return prompt_toolkit (style, text) fragments for Python source."""
    if not source:
        return []
    try:
        from pygments import lex
        from pygments.lexers.python import PythonLexer
        from pygments.token import Token
    except Exception:  # noqa: BLE001
        return [("class:py.text", source)]

    style_map = {
        Token.Keyword: "class:py.keyword",
        Token.Keyword.Constant: "class:py.keyword",
        Token.Name.Builtin: "class:py.builtin",
        Token.Name.Function: "class:py.function",
        Token.Name.Class: "class:py.class",
        Token.Name.Decorator: "class:py.decorator",
        Token.String: "class:py.string",
        Token.String.Doc: "class:py.string",
        Token.Number: "class:py.number",
        Token.Comment: "class:py.comment",
        Token.Operator: "class:py.operator",
        Token.Punctuation: "class:py.punct",
    }

    out: list[tuple[str, str]] = []
    for ttype, value in lex(source, PythonLexer()):
        cur = ttype
        style = "class:py.text"
        while cur is not None:
            if cur in style_map:
                style = style_map[cur]
                break
            cur = cur.parent
        if value:
            out.append((style, value))
    return out


def highlight_python_lines(
    source: str,
    *,
    indent: str = "",
    line_numbers: bool = True,
) -> list[tuple[str, str]]:
    """Highlight source and split into display lines with optional gutters."""
    fragments = highlight_python(source.rstrip("\n") + "\n")
    lines: list[list[tuple[str, str]]] = [[]]
    for style, text in fragments:
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append([])
            if part:
                lines[-1].append((style, part))

    # Drop trailing empty line from final newline.
    if lines and not lines[-1]:
        lines.pop()

    width = max(2, len(str(len(lines))))
    out: list[tuple[str, str]] = []
    for i, frags in enumerate(lines, 1):
        out.append(("", indent))
        if line_numbers:
            out.append(("class:py.lineno", f"{i:>{width}} │ "))
        out.extend(frags if frags else [("class:py.text", "")])
        out.append(("", "\n"))
    return out
