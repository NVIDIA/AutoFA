# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sandboxed detector execution."""

from __future__ import annotations

import ast
import inspect
import json
import math
from collections.abc import Callable
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from types import ModuleType
from typing import Any

import regex

from autofa.core.dependencies import validate_bundle
from autofa.core.fields import (
    FieldPolicy,
    get_field_policy,
    scan_banned_fields,
    wrap_item_ban_sentinels,
)
from autofa.core.helpers import bounded_result, check_schema
from autofa.core.localisation import (
    ACTIVE_CAPTURE,
    CapturingLLM,
    PassageCapture,
    record_count,
    record_lookup,
    record_membership,
    record_regex_result,
)
from autofa.llm.windowing import window_provenance

_REGEX_TIMEOUT_SECONDS = 0.1
_ACTIVE_LLM: ContextVar[Any] = ContextVar("autofa_detector_llm", default=None)
_ACTIVE_POLICY: ContextVar[Any] = ContextVar("autofa_detector_policy", default=None)
_TIMED_REGEX_METHODS = (
    "findall",
    "finditer",
    "fullmatch",
    "match",
    "search",
    "split",
    "sub",
    "subn",
)
# Zero-based positional indices for (module function, compiled-pattern method).
# regex exposes these arguments positionally as well as by keyword.
_REGEX_CONCURRENT_POSITIONS = {
    "findall": (6, 4),
    "finditer": (7, 4),
    "fullmatch": (6, 3),
    "match": (6, 3),
    "search": (6, 3),
    "split": (4, 2),
    "sub": (7, 5),
    "subn": (7, 5),
}


class RegexTimeoutError(TimeoutError):
    """A local execution limit, not a retryable model/network timeout."""


def _regex_result(call: Callable, name: str, args: tuple, kwargs: dict) -> Any:
    try:
        result = call(*args, **kwargs)
    except TimeoutError as exc:
        raise RegexTimeoutError(str(exc)) from exc
    if name == "finditer":
        return _regex_iterator(result)
    return result


def _regex_iterator(iterator: Any):
    try:
        yield from iterator
    except TimeoutError as exc:
        raise RegexTimeoutError(str(exc)) from exc


class _TimeoutPattern:
    """Proxy a compiled regex and enforce a deadline on matching methods."""

    def __init__(self, pattern: Any) -> None:
        self._pattern = pattern

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._pattern, name)
        if name not in _TIMED_REGEX_METHODS:
            return attr

        def timed(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("timeout", _REGEX_TIMEOUT_SECONDS)
            if len(args) <= _REGEX_CONCURRENT_POSITIONS[name][1]:
                kwargs.setdefault("concurrent", False)
            result = _regex_result(attr, name, args, kwargs)
            if name == "findall" and ACTIVE_CAPTURE.get() is not None:
                _capture_findall_matches(self._pattern.finditer(*args, **kwargs))
            return record_regex_result(name, result)

        return timed


class _TimeoutRegex(ModuleType):
    """``re``-compatible module that interrupts pathological matching."""

    def __init__(self) -> None:
        super().__init__("re")

    def __getattr__(self, name: str) -> Any:
        return getattr(regex, name)

    def compile(self, pattern: Any, flags: int = 0, **kwargs: Any) -> _TimeoutPattern:
        return _TimeoutPattern(regex.compile(pattern, flags, **kwargs))


def _timed_regex_method(name: str) -> Callable[..., Any]:
    def timed(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", _REGEX_TIMEOUT_SECONDS)
        # regex's process-CPU timeout otherwise counts other Python threads
        # while its default matching mode releases the GIL on str/bytes.
        if len(args) <= _REGEX_CONCURRENT_POSITIONS[name][0]:
            kwargs.setdefault("concurrent", False)
        result = _regex_result(getattr(regex, name), name, args, kwargs)
        if name == "findall" and ACTIVE_CAPTURE.get() is not None:
            # Module finditer inserts `partial` before findall's positional
            # `concurrent`; retain the caller's concurrency in the evidence pass.
            capture_args = args[:6] + (False,) + args[6:] if len(args) > 6 else args
            _capture_findall_matches(regex.finditer(*capture_args, **kwargs))
        return record_regex_result(name, result)

    return timed


def _capture_findall_matches(matches: Any) -> None:
    # Keep native findall's exact result and timeout behavior; the bounded
    # evidence pass must not turn a successfully computed label into an error.
    from itertools import islice

    from autofa.core.localisation import MAX_MATCHES

    capture = ACTIVE_CAPTURE.get()
    try:
        for match in islice(matches, MAX_MATCHES + 1):
            capture.record_match(match)
    except TimeoutError:
        capture.notes.append("Regex evidence scan timed out; passages may be incomplete.")
        capture.truncated = True


SAFE_REGEX = _TimeoutRegex()
for _method_name in _TIMED_REGEX_METHODS:
    setattr(SAFE_REGEX, _method_name, _timed_regex_method(_method_name))

_SAFE_NAMES = (
    "True",
    "False",
    "None",
    "abs",
    "all",
    "any",
    "bool",
    "chr",
    "dict",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "getattr",
    "hasattr",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "KeyError",
)


def _build_safe_builtins() -> dict[str, Any]:
    base = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    out: dict[str, Any] = {}
    for name in _SAFE_NAMES:
        if name in base:
            out[name] = base[name]
    return out


SAFE_BUILTINS = _build_safe_builtins()


def _allowed_modules() -> dict[str, Any]:
    import autofa.llm.compat as compat
    import autofa.trace as trace

    llm_pkg = _pkg_with("autofa.llm", compat=compat)
    return {
        "re": SAFE_REGEX,
        "json": json,
        "math": math,
        "autofa.llm.compat": compat,
        "autofa.llm": llm_pkg,
        "autofa.trace": trace,
    }


def _pkg_with(name: str, **attrs: Any) -> ModuleType:
    m = ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class NullLLM:
    """Stand-in when the LLM endpoint is unavailable; ``available`` is False."""

    available = False

    def classify(
        self,
        prompt: str,
        text: str,
        *,
        focus: str = "auto",
        max_chars: int | None = None,
        images: list[dict[str, Any]] | None = None,
    ) -> str:
        raise RuntimeError(
            "LLM classify unavailable — check the configured endpoint, model and API key variable"
        )


def active_field_policy() -> FieldPolicy:
    return _ACTIVE_POLICY.get() or get_field_policy()


class DetectorAbstained(Exception):
    """The detector could not decide; never a positive or negative prediction."""


def invoke_detector(fn: Callable, item: Any, client: Any) -> Any:
    """Choose a compatible signature before executing, without retrying body errors."""
    signature = inspect.signature(fn)
    for args, kwargs in (((item, client), {}), ((item,), {"llm_client": client}),
                         ((item,), {"llm": client}), ((item,), {})):
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        # An item-only parent can still call LLM-backed peers through library.fire.
        # Bind independently of its public signature, and preserve worker context.
        token = _ACTIVE_LLM.set(client)
        try:
            return fn(*args, **kwargs)
        finally:
            _ACTIVE_LLM.reset(token)
    raise TypeError("detector must accept item and optionally an LLM client")


def validate_source(source: str, policy: FieldPolicy, *, helper: bool = False) -> None:
    banned = scan_banned_fields(source, policy)
    if banned:
        raise ValueError(f"detector references disallowed fields {banned} (mode={policy.mode})")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            allowed = {"re", "json", "math", "autofa", "autofa.trace"}
            if not helper:
                allowed.update({"autofa.llm", "autofa.llm.compat"})
            if any(name not in allowed for name in names) or getattr(node, "level", 0):
                raise ValueError(f"import not allowed in sandbox: {names}")
        if helper and isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in {"llm_client", "llm"}:
                raise ValueError("helpers must be deterministic; put LLM decisions in the detector")
        if helper and isinstance(node, ast.Attribute) and node.attr in {
            "classify", "classify_options", "decide", "chat", "llm", "abstain",
        }:
            raise ValueError("helpers cannot call LLMs or decide detector abstention")


def _restricted_import(name: str, globals: Any = None, locals: Any = None, fromlist=(), level=0):
    allowed = _allowed_modules()
    if name in allowed:
        return allowed[name]
    if name == "autofa":
        pkg = ModuleType("autofa")
        pkg.llm = allowed["autofa.llm"]  # type: ignore[attr-defined]
        pkg.trace = allowed["autofa.trace"]  # type: ignore[attr-defined]
        return pkg
    raise ImportError(f"import not allowed in detector sandbox: {name}")


class _CaptureEvidence(ast.NodeTransformer):
    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            return ast.copy_location(ast.Call(
                func=ast.Name(id="_autofa_contains", ctx=ast.Load()),
                args=[node.left, node.comparators[0],
                      ast.Constant(isinstance(node.ops[0], ast.NotIn))],
                keywords=[],
            ), node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.ctx, ast.Load)
                and not any(isinstance(part, ast.Slice) for part in ast.walk(node.slice))):
            return ast.copy_location(ast.Call(
                func=ast.Name(id="_autofa_lookup", ctx=ast.Load()),
                args=[node.value, node.slice], keywords=[],
            ), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "count":
            return ast.copy_location(ast.Call(
                func=ast.Name(id="_autofa_count", ctx=ast.Load()),
                args=[node.func, *node.args], keywords=node.keywords,
            ), node)
        return node


def compile_detector(
    source: str,
    *,
    policy: FieldPolicy | None = None,
    bundle: dict | None = None,
    entrypoint: str | None = None,
    _bundle_node: str | None = None,
) -> Callable[..., Any]:
    """Compile detector source into a callable ``detect(item, llm=None)``.

    ``bundle`` pins the peer detectors and helpers the source may call through the
    read-only ``library.fire(name, item)`` and ``helpers.call(alias, ...)`` objects.
    """
    pol = policy if policy is not None else get_field_policy()
    if bundle is not None and _bundle_node is None:
        validate_bundle(bundle, source=source)
        for node in bundle["nodes"].values():
            validate_source(node["source"], pol, helper=node["role"] == "helper")
    validate_source(source, pol, helper=bool(
        bundle and _bundle_node and bundle["nodes"][_bundle_node]["role"] == "helper"
    ))
    banned = scan_banned_fields(source, pol)
    if banned:
        raise ValueError(
            f"detector references disallowed fields {banned} "
            f"(mode={pol.mode}; edit detector_fields in the active AutoFA config to change)"
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"syntax error: {e}") from e

    fn_name: str | None = entrypoint
    for node in tree.body:
        if entrypoint:
            break
        if isinstance(node, ast.FunctionDef):
            if node.name in {"detect", "feature", "label", "classify"}:
                fn_name = node.name
                break
            if fn_name is None:
                fn_name = node.name
    if fn_name is None:
        raise ValueError("detector source must define a function")

    edges = (bundle["nodes"][_bundle_node] if _bundle_node else bundle) or {}
    compiled_nodes: dict[str, Callable[..., Any]] = {}

    def compiled_node(ref: str) -> Callable:
        if ref not in compiled_nodes:
            node = bundle["nodes"][ref]
            compiled_nodes[ref] = compile_detector(
                node["source"], policy=pol, bundle=bundle, _bundle_node=ref,
                entrypoint=node.get("entrypoint"),
            )
        return compiled_nodes[ref]

    class _Helpers:
        def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
            ref = edges.get("helpers", {}).get(name)
            if ref is None:
                raise KeyError(f"unknown helper {name!r}; declare uses_helpers= and resubmit")
            node = bundle["nodes"][ref]
            fn = compiled_node(ref)
            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()
            check_schema(dict(bound.arguments), node.get("input_schema", {}), "arguments")
            # Ordinary helper functions get detached arguments, including wrapped items.
            result = fn(*deepcopy(args), **deepcopy(kwargs))
            return bounded_result(result, node.get("output_schema", {}))

        def render_passages(self, result: dict, max_chars: int = 16_000) -> dict:
            from autofa.trace import render_passages

            return render_passages(result, max_chars=max_chars)

    class _Library:
        """Read-only helper: ``library.fire(name, item)`` runs a snapshotted peer."""

        def fire(self, name: str, item: Any, llm_client: Any = None) -> Any:
            if llm_client is None:
                llm_client = _ACTIVE_LLM.get()
            key = str(name)
            if bundle is None:
                # Only detectors saved before dependency bundles reach this point.
                raise KeyError(
                    f"detector has no pinned dependency bundle for {key!r}; "
                    "recreate it with depends_on= in the current AutoFA"
                )
            ref = edges.get("detectors", {}).get(key)
            if ref is None:
                raise KeyError(f"unknown detector dependency {key!r}; declare depends_on=")
            return invoke_detector(compiled_node(ref), item, llm_client)

        def abstain(self, reason: str) -> None:
            raise DetectorAbstained(str(reason))

    glb: dict[str, Any] = {
        "__builtins__": {**SAFE_BUILTINS, "__import__": _restricted_import},
        "re": SAFE_REGEX,
        "json": json,
        "math": math,
        "library": _Library(),
        "helpers": _Helpers(),
        "_autofa_contains": record_membership,
        "_autofa_lookup": record_lookup,
        "_autofa_count": record_count,
    }
    tree = ast.fix_missing_locations(_CaptureEvidence().visit(tree))
    exec(compile(tree, "<detector>", "exec"), glb, glb)
    fn = glb.get(fn_name)
    if not callable(fn):
        raise ValueError(f"compiled object {fn_name!r} is not callable")

    @wraps(fn)
    def invoke(*args: Any, **kwargs: Any) -> Any:
        if bundle and _bundle_node and bundle["nodes"][_bundle_node]["role"] == "helper":
            policy_token = _ACTIVE_POLICY.set(pol)
            try:
                return fn(*args, **kwargs)
            finally:
                _ACTIVE_POLICY.reset(policy_token)
        # Dependencies inherit the current call's client, never another worker's.
        client = args[1] if len(args) > 1 else kwargs.get(
            "llm_client", kwargs.get("llm", _ACTIVE_LLM.get())
        )
        token = _ACTIVE_LLM.set(client)
        policy_token = _ACTIVE_POLICY.set(pol)
        try:
            with window_provenance():
                return fn(*args, **kwargs)
        finally:
            _ACTIVE_LLM.reset(token)
            _ACTIVE_POLICY.reset(policy_token)

    return invoke


def run_detector(
    source: str,
    item: dict[str, Any],
    llm: Any | None = None,
    *,
    ban_sentinels: bool = True,
    policy: FieldPolicy | None = None,
    deps: dict[str, str] | None = None,
    bundle: dict | None = None,
    derive_trace: bool = True,
    localisation: dict[str, Any] | None = None,
    artifact_dependencies: list[dict[str, Any]] | None = None,
) -> Any:
    """Return the original label; capture evidence without additional model calls."""
    pol = policy if policy is not None else get_field_policy()
    fn = compile_detector(source, policy=pol, bundle=bundle)
    enriched = item
    if derive_trace:
        from autofa.data.trace_fields import derive_trace_fields

        enriched = derive_trace_fields(item)
    payload = wrap_item_ban_sentinels(enriched, pol) if ban_sentinels else dict(enriched)
    client = llm if llm is not None else NullLLM()
    capture = (PassageCapture(enriched, source, deps, bundle=bundle)
               if localisation is not None else None)
    if localisation is not None:
        localisation.clear()
    detector_client = CapturingLLM(client, capture) if capture is not None else client
    token = ACTIVE_CAPTURE.set(capture)
    from autofa.data.artifact_access import capture_artifact_reads

    try:
        with capture_artifact_reads() as artifact_reads:
            try:
                result = invoke_detector(fn, payload, detector_client)
            finally:
                if artifact_dependencies is not None:
                    artifact_dependencies[:] = artifact_reads
        if bundle and (isinstance(result, (dict, list, tuple, set))
                       or isinstance(result, float) and not math.isfinite(result)):
            raise ValueError("composed detectors must return a scalar label; "
                             "inspect helper results explicitly or use library.abstain")
    finally:
        ACTIVE_CAPTURE.reset(token)
    if capture is not None:
        localisation.update(capture.finish(result))
        if artifact_reads:
            localisation["artifact_reads"] = artifact_reads
    return result
