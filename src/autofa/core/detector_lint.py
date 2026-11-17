# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject shallow structural-proxy detectors (schema-token / count / length only).

A good detector isolates a behavioral mechanism in the model's own content
(reasoning, claims, error text, tool arguments). Counting transcript JSON
schema tokens (``source_call_id``, ``"command":``, ``"name":"finish"``) or
bare length is a metadata/quantity proxy — not a failure mode.

Length/schema counts are allowed only as an *extra stratum* when the source
also reads behavioral content (marathon-style conjunction).
"""

from __future__ import annotations

import ast
import re

from autofa.llm.detection import LLM_CALL_RE

REWRITE_HINT = (
    "Rewrite to isolate a behavioral mechanism: read the model's reasoning/"
    "claims/errors via autofa.trace (or get_trajectory_step / json.loads + walk "
    "steps), not schema-token counts / length alone. Schema counts are OK only "
    "as an extra stratum on top of a content signal. Or compose with depends_on= "
    "to explain WHY a coarse proxy fires (add a behavioral gate inside the "
    "parent fire set)."
)

FOCUS_HINT = (
    "Pass the full surface into llm_client.classify(..., focus='tail'|'head'|'ends') "
    "or use autofa.trace.evidence_text(item, focus='closing'|'opening'|'ends'). "
    "Do not hand-slice text[-4000:] / text[:4000] — that recreates envelope "
    "blindness. focus='closing' uses last agent steps when a trajectory exists, "
    "and falls back to a char tail on non-agentic corpora."
)

# Hand-rolled classify budgets — the old envelope habit on derived fields.
_MAGIC_SLICE_RE = re.compile(
    r"""\[\s*(?:-\s*)?\d{3,}\s*:\s*\]|\[\s*:\s*\d{3,}\s*\]"""
)

# Transcript JSON / ATIF schema tokens — counting these is not a behavior.
# Patterns that collide with the autofa.trace API (tool_calls, step_id) only
# match quoted string literals, so `s.tool_calls` attribute access is fine.
_SCHEMA_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"source_call_id", re.I),
    re.compile(r"tool_call_id", re.I),
    re.compile(r"""['"]tool_calls['"]"""),
    re.compile(r"""['"]step_id['"]"""),
    re.compile(r'"command"\s*:'),
    re.compile(r"'command'\s*:"),
    re.compile(r'"name"\s*:\s*"finish"', re.I),
    re.compile(r'"function"\s*:'),
    re.compile(r'"role"\s*:'),
    re.compile(r'"arguments"\s*:'),
    re.compile(r"schema_version", re.I),
    re.compile(r"tool_definitions", re.I),
)

# Scaffold / metadata count fields (even if accessed via string heuristics).
_SCAFFOLD_COUNT_FIELDS = frozenset(
    {
        "num_generated_tokens",
        "n_tool_calls",
        "n_agent_steps",
        "tb_trajectory_length",
        "n_steps",
        "token_count",
        "num_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "total_tokens",
        "latency_s",
        "latency_ttft_ms",
    }
)

# Importing the API, or opening a container of metadata, is not itself a
# behavioral signal.
_TRACE_CONTENT_RE = re.compile(
    r"actions_text|observations_text|final_message|"
    r"\b(?:reasoning|actions|observations|evidence_text|text_window)\s*\(|"
    r"\.(?:reasoning|message|observation|observation_data|arguments|arguments_raw)\b",
    re.I,
)

# Signals that the detector reads model-authored behavioral content.
_BEHAVIORAL_CONTENT_RE = re.compile(
    r"("
    r"pytest|passed\s+in|all\s+tests?\s+pass|"
    r"AssertionError|Traceback|FAILED|Error:|"
    r"already\s+(been\s+)?(fixed|resolved)|cannot\s+reproduce|"
    r"issue\s+is\s+(now\s+)?(fixed|resolved)|no\s+regressions?|"
    r"reasoning_content|chain_of_thought|thought|"
    r"actions_text|observations_text|final_message|"
    r"llm_client\.classify|classify\(|"
    r"confident|fix\s+is\s+complete|successfully\s+implemented"
    r")",
    re.I,
)

_COUNT_CALL_RE = re.compile(
    r"""(?:\.count\s*\(|re\.(?:findall|finditer|search|match)\s*\(|len\s*\(\s*re\.)""",
    re.I,
)

_LEN_THRESH_RE = re.compile(
    r"""len\s*\(\s*(?:rt|text|response|resp|cot|hay|blob|s)\s*\)\s*[><=!]+\s*\d{3,}""",
    re.I,
)

_CHAR_THRESH_RE = re.compile(
    r"""(?:len\s*\([^)]+\)|[a-z_]+\.count\s*\([^)]+\))\s*[><=!]+\s*\d+""",
    re.I,
)


def _string_literals(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.append(v.value)
    return out


def _has_schema_token_literal(literals: list[str], source: str) -> bool:
    blob = "\n".join(literals) + "\n" + source
    return any(p.search(blob) for p in _SCHEMA_TOKEN_PATTERNS)


def _reads_scaffold_count_field(source: str, literals: list[str]) -> bool:
    for name in _SCAFFOLD_COUNT_FIELDS:
        if re.search(rf"""['"]{re.escape(name)}['"]""", source):
            return True
        if name in literals:
            return True
    return False


def _has_behavioral_content(source: str, literals: list[str]) -> bool:
    if LLM_CALL_RE.search(source):
        return True
    # Structured trajectory helper or derived fields.
    if _TRACE_CONTENT_RE.search(source):
        return True
    # json.loads + walking steps / tool arguments / observations / reasoning
    if re.search(r"json\.loads", source) and re.search(
        r"arguments|observation|reasoning", source, re.I
    ):
        return True
    if _BEHAVIORAL_CONTENT_RE.search(source):
        return True
    # Literals with behavioral phrases (regex patterns for model language)
    for lit in literals:
        if any(p.search(lit) for p in _SCHEMA_TOKEN_PATTERNS):
            continue
        if re.search(
            r"pass|fail|error|assert|fix|reproduc|already|"
            r"resolved|pytest|traceback|confidence|complete|"
            r"cannot\s+reproduce|nothing\s+to\s+fix",
            lit,
            re.I,
        ) and re.search(r"[a-zA-Z]{4,}", lit):
            return True
    return False


def _is_library_passthrough(source: str) -> bool:
    """True if the detector only returns library.fire(...) with no extra logic."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn = node
            break
    if fn is None:
        return False
    # Strip docstring
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    # Allow trivial early returns for empty text, then a single library.fire return
    nontrivial = [
        s
        for s in body
        if not (
            isinstance(s, ast.If)
            and isinstance(s.test, (ast.UnaryOp, ast.Compare, ast.Name, ast.Call))
            and len(s.body) == 1
            and isinstance(s.body[0], ast.Return)
        )
    ]
    if len(nontrivial) != 1:
        return False
    stmt = nontrivial[0]
    if not isinstance(stmt, ast.Return) or stmt.value is None:
        return False
    return "library.fire" in ast.unparse(stmt.value)


def lint_detector_source(source: str) -> tuple[bool, str | None]:
    """Return ``(ok, reason)``. ``ok=False`` means hard-reject at submit."""
    src = source or ""
    if not src.strip():
        return False, "empty detector source"

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    literals = _string_literals(tree)
    has_behavior = _has_behavioral_content(src, literals)
    uses_schema = _has_schema_token_literal(literals, src)
    uses_scaffold = _reads_scaffold_count_field(src, literals)
    uses_count = bool(_COUNT_CALL_RE.search(src) or _CHAR_THRESH_RE.search(src))
    uses_len = bool(_LEN_THRESH_RE.search(src))

    if _is_library_passthrough(src) and not has_behavior:
        return (
            False,
            "Composed detector is a bare library.fire(...) passthrough — adds no "
            "behavioral signal. " + REWRITE_HINT,
        )

    if uses_scaffold and not has_behavior:
        return (
            False,
            "Detector reads scaffold count metadata "
            f"({', '.join(sorted(_SCAFFOLD_COUNT_FIELDS)[:4])}…) without a "
            "behavioral content signal. " + REWRITE_HINT,
        )

    # Schema-token / count / length as the sole decision → reject
    if (uses_schema or uses_count or uses_len) and not has_behavior:
        kind = []
        if uses_schema:
            kind.append("schema-token")
        if uses_count:
            kind.append("count")
        if uses_len:
            kind.append("length")
        return (
            False,
            f"Detector is a {'/'.join(kind)} proxy with no behavioral content "
            "read (e.g. counting source_call_id / \"command\": / finish tokens, "
            "or bare len thresholds). " + REWRITE_HINT,
        )

    # Presence/absence of finish/edit schema tokens alone (no count needed)
    if uses_schema and not has_behavior:
        return (
            False,
            "Detector keys only on transcript JSON schema tokens. " + REWRITE_HINT,
        )

    # Magic ±N000 char slices around classify recreate envelope blindness.
    uses_classify = bool(LLM_CALL_RE.search(src))
    if uses_classify and _MAGIC_SLICE_RE.search(src):
        return (
            False,
            "LLM detector hand-slices a fixed char budget (e.g. text[-4000:]). "
            + FOCUS_HINT,
        )

    return True, None


def _budget_negative_warnings(tree: ast.AST) -> list[str]:
    """Follow length-guarded text through simple aliases into classification.

    A length limit can be a legitimate semantic predicate. Diagnose suspicious
    budget skips without rejecting the detector or interpreting its claim.
    """
    warnings = []
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        nodes = sorted(ast.walk(function), key=lambda n: getattr(n, "lineno", 0))
        for node in nodes:
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                continue
            comparison = node.test
            if len(comparison.ops) != 1:
                continue
            left, right = comparison.left, comparison.comparators[0]
            op = comparison.ops[0]
            if isinstance(right, ast.Call) and isinstance(right.func, ast.Name):
                left, right = right, left
                reverse = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}
                op = reverse.get(type(op), type(op))()
            if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                    and left.func.id == "len" and len(left.args) == 1
                    and isinstance(left.args[0], ast.Name)):
                continue
            branch = (node.body if isinstance(op, (ast.Gt, ast.GtE))
                      else node.orelse if isinstance(op, (ast.Lt, ast.LtE)) else [])
            if not (len(branch) == 1 and isinstance(branch[0], ast.Return)
                    and isinstance(branch[0].value, ast.Constant)
                    and branch[0].value.value is False):
                continue
            aliases = {left.args[0].id}
            for later in nodes:
                if getattr(later, "lineno", 0) <= node.lineno:
                    continue
                if isinstance(later, ast.Assign):
                    targets = {n.id for target in later.targets for n in ast.walk(target)
                               if isinstance(n, ast.Name)}
                    reads = {n.id for n in ast.walk(later.value) if isinstance(n, ast.Name)}
                    if aliases & reads:
                        aliases.update(targets)
                    else:
                        aliases.difference_update(targets)
                if not (isinstance(later, ast.Call) and isinstance(later.func, ast.Attribute)
                        and later.func.attr in {"classify", "classify_options", "decide"}):
                    continue
                keywords = {kw.arg: kw.value for kw in later.keywords}
                if "max_chars" not in keywords:
                    continue
                index = 0 if later.func.attr == "classify_options" else 1
                text = keywords.get("text")
                if text is None and len(later.args) > index:
                    text = later.args[index]
                if text is not None and aliases & {
                    n.id for n in ast.walk(text) if isinstance(n, ast.Name)
                }:
                    warnings.append(
                        f"Line {node.lineno}: a length guard returns False before classifying "
                        "the same evidence. If this is an evidence-budget skip, retrieve "
                        "sufficient evidence or call library.abstain(reason); missing evidence "
                        "is not a negative prediction. A genuine semantic length condition may "
                        "remain negative. Inspect this branch and test an over-budget control "
                        "with expected_abstentions when the claim remains unresolved."
                    )
                    break
    return warnings


def evidence_budget_warnings(source: str) -> list[str]:
    """Diagnose evidence-budget skips without rejecting semantic length rules.

    This is a diagnostic, not a proof that a claim can be decided from excerpts.
    Some claims legitimately require more evidence than the available budget.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    warnings = _budget_negative_warnings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if len(test.ops) != 1:
            continue
        left, right = test.left, test.comparators[0]

        def is_length(value: ast.AST) -> bool:
            return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "len")

        # Detect both `len(text) > budget` and `budget < len(text)`, including
        # their else-branch forms. A conjunction with a content condition is not
        # treated as an unconditional length rejection.
        over_budget = ((is_length(left) and isinstance(test.ops[0], (ast.Gt, ast.GtE)))
                       or (is_length(right) and isinstance(test.ops[0], (ast.Lt, ast.LtE))))
        under_budget = ((is_length(left) and isinstance(test.ops[0], (ast.Lt, ast.LtE)))
                        or (is_length(right) and isinstance(test.ops[0], (ast.Gt, ast.GtE))))
        branch = node.body if over_budget else node.orelse if under_budget else []
        for statement in branch:
            if not isinstance(statement, ast.Return) or not isinstance(statement.value, ast.Dict):
                continue
            values = {
                key.value: value for key, value in zip(
                    statement.value.keys, statement.value.values, strict=True,
                ) if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            text = values.get("text")
            if not isinstance(text, ast.Constant) or text.value != "":
                continue
            warnings.append(
                f"Line {node.lineno}: a length limit returns empty evidence. Long traces can "
                "then cause blanket abstentions. Search the full available content, use "
                "autofa.trace.step_evidence or regex_passages for bounded relevant evidence, "
                "and inspect later edits/results with steps(item). Preserve scope: excerpts "
                "do not establish full-history absence or whether later recovery occurred; abstain "
                "when the facts required by the claim remain unavailable."
            )
            break
    return warnings
