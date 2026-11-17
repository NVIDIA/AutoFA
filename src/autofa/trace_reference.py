# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the agent-facing trace API reference from its Python contracts.

Keep schemas, defaults, and documentation on the public API itself. This module
only renders them, so the reference cannot silently retain an old return shape.
"""

from __future__ import annotations

import inspect
import types
from dataclasses import MISSING, fields, is_dataclass
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Literal,
    NotRequired,
    Required,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from autofa import trace


def _unwrap(annotation: Any) -> tuple[Any, list[str], bool | None]:
    """Separate human documentation and key presence from the value's type."""
    documentation: list[str] = []
    required: bool | None = None
    while get_origin(annotation) in (Annotated, Required, NotRequired):
        origin, args = get_origin(annotation), get_args(annotation)
        if origin is Annotated:
            documentation.extend(value for value in args[1:] if isinstance(value, str))
        else:
            required = origin is Required
        annotation = args[0]
    return annotation, documentation, required


def _is_record(value: Any) -> bool:
    return is_typeddict(value) or (inspect.isclass(value) and is_dataclass(value))


class _Reference:
    def __init__(self) -> None:
        self.exports = {name: getattr(trace, name) for name in trace.__all__}
        self.aliases = {
            name: value for name, value in self.exports.items()
            if not inspect.isfunction(value) and not _is_record(value)
        }

    def alias_name(self, annotation: Any, *, exclude: str | None = None) -> str | None:
        return next((name for name, value in self.aliases.items()
                     if value is annotation and name != exclude), None)

    def type_name(self, annotation: Any, *, exclude: str | None = None) -> str:
        alias = self.alias_name(annotation, exclude=exclude)
        if alias:
            return alias
        annotation, _, _ = _unwrap(annotation)
        alias = self.alias_name(annotation, exclude=exclude)
        if alias:
            return alias
        if annotation in (None, type(None)):
            return "None"
        if annotation is Any:
            return "Any"
        if annotation is Ellipsis:
            return "..."
        if isinstance(annotation, ForwardRef):
            return annotation.__forward_arg__
        if isinstance(annotation, str):
            return annotation
        if isinstance(annotation, TypeAliasType):
            return self.type_name(annotation.__value__, exclude=exclude)
        origin, args = get_origin(annotation), get_args(annotation)
        if origin in (Union, types.UnionType):
            return " | ".join(self.type_name(arg) for arg in args)
        if origin is Literal:
            return "Literal[" + ", ".join(repr(arg) for arg in args) + "]"
        if origin:
            return self.type_name(origin) + "[" + ", ".join(
                self.type_name(arg) for arg in args
            ) + "]"
        return getattr(annotation, "__name__", str(annotation).removeprefix("typing."))

    def signature(self, name: str, function: Any) -> str:
        signature = inspect.signature(function)
        hints = get_type_hints(function, include_extras=True)
        parameters: list[str] = []
        inserted_star = False
        values = list(signature.parameters.values())
        for index, parameter in enumerate(values):
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not inserted_star:
                parameters.append("*")
                inserted_star = True
            prefix = ""
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                prefix, inserted_star = "*", True
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                prefix = "**"
            rendered = prefix + parameter.name
            if parameter.name in hints:
                rendered += ": " + self.type_name(hints[parameter.name])
            if parameter.default is not inspect.Parameter.empty:
                rendered += " = " + repr(parameter.default)
            parameters.append(rendered)
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY and (
                index + 1 == len(values)
                or values[index + 1].kind is not inspect.Parameter.POSITIONAL_ONLY
            ):
                parameters.append("/")
        result = name + "(" + ", ".join(parameters) + ")"
        if "return" in hints:
            result += " -> " + self.type_name(hints["return"])
        return result

    def dependencies(self, values: list[Any]) -> dict[str, Any]:
        """Walk nested return/parameter schemas, including unexported records."""
        found: dict[str, Any] = {}
        # get_type_hints can create temporary generic aliases. Keep their objects
        # alive while traversing: an ID alone can be reused for a later schema.
        seen: dict[int, Any] = {}

        def visit(value: Any) -> None:
            if id(value) in seen:
                return
            seen[id(value)] = value
            alias = self.alias_name(value)
            if alias:
                found[alias] = value
            base, _, _ = _unwrap(value)
            if base is not value:
                visit(base)
            if _is_record(base):
                found[base.__name__] = base
                for annotation in get_type_hints(base, include_extras=True).values():
                    visit(annotation)
                if is_dataclass(base):
                    for prop in vars(base).values():
                        if isinstance(prop, property) and prop.fget:
                            for annotation in get_type_hints(
                                prop.fget, include_extras=True,
                            ).values():
                                visit(annotation)
            elif inspect.isfunction(base):
                for annotation in get_type_hints(base, include_extras=True).values():
                    visit(annotation)
            elif isinstance(base, TypeAliasType):
                visit(base.__value__)
            else:
                for argument in get_args(base):
                    if argument is not Ellipsis:
                        visit(argument)

        for value in values:
            visit(value)
        return found

    def record(self, name: str, value: Any) -> str:
        lines = [f"### `{name}`"]
        if _is_record(value):
            documentation = inspect.getdoc(value)
            if documentation:
                lines.extend(["", documentation])
            hints = get_type_hints(value, include_extras=True)
            if is_typeddict(value):
                lines.extend(["", "Dictionary keys:", ""])
                for key, annotation in hints.items():
                    _, docs, required = _unwrap(annotation)
                    if required is None:
                        required = key in value.__required_keys__
                    presence = "required key" if required else "optional key"
                    suffix = "; " + " ".join(docs) if docs else ""
                    lines.append(
                        f"- `{key}: {self.type_name(annotation)}` ({presence}){suffix}"
                    )
            else:
                lines.extend(["", "Dataclass attributes (use `step.field`):", ""])
                for field in fields(value):
                    annotation = hints[field.name]
                    _, docs, _ = _unwrap(annotation)
                    if field.metadata.get("description"):
                        docs.append(str(field.metadata["description"]))
                    if field.default is not MISSING:
                        default = f"; default `{field.default!r}`"
                    elif field.default_factory is not MISSING:
                        factory = getattr(field.default_factory, "__name__", "factory")
                        default = f"; default factory `{factory}()`"
                    else:
                        default = "; required constructor argument"
                    suffix = "; " + " ".join(docs) if docs else ""
                    lines.append(
                        f"- `{field.name}: {self.type_name(annotation)}`{default}{suffix}"
                    )
                for key, prop in vars(value).items():
                    if isinstance(prop, property) and prop.fget:
                        returned = get_type_hints(prop.fget, include_extras=True).get("return")
                        docs = inspect.getdoc(prop) or ""
                        suffix = "; " + docs if docs else ""
                        lines.append(
                            f"- `{key}: {self.type_name(returned)}`; read-only property{suffix}"
                        )
        else:
            base = value.__value__ if isinstance(value, TypeAliasType) else value
            _, docs, _ = _unwrap(base)
            lines.extend(["", f"`{name} = {self.type_name(base, exclude=name)}`"])
            if docs:
                lines.extend(["", " ".join(docs)])
        return "\n".join(lines)

    def render(self, name: str | None) -> str:
        if name is not None and name not in self.exports:
            raise ValueError(
                f"Unknown autofa.trace symbol {name!r}. Available: "
                + ", ".join(self.exports)
            )
        selected = self.exports if name is None else {name: self.exports[name]}
        parts = [
            "# autofa.trace API reference",
            "Generated from the installed API signatures, annotations, and docstrings. "
            "These are static contracts, not runtime validators. Dictionary keys use "
            "`value[\"key\"]`; `Step` attributes use `step.field`. An **optional key** may "
            "be absent; `| None` means a present value may be null. They are independent.",
        ]
        for symbol, value in selected.items():
            if inspect.isfunction(value):
                parts.extend([
                    f"## `{symbol}`",
                    "```python\n" + self.signature(symbol, value) + "\n```",
                ])
                documentation = inspect.getdoc(value)
                if documentation:
                    parts.append(documentation)
        schemas = self.dependencies(list(selected.values()))
        if schemas:
            parts.append("## Value types")
            # Preserve API export order, then include any reachable internal types.
            ordered = [symbol for symbol in self.exports if symbol in schemas]
            ordered.extend(symbol for symbol in schemas if symbol not in ordered)
            parts.extend(self.record(symbol, schemas[symbol]) for symbol in ordered)
        return "\n\n".join(parts) + "\n"


def render_trace_api_reference(name: str | None = None) -> str:
    """Render all exports, or one export and its reachable value types.

    ``name`` must be an exact name from ``autofa.trace.__all__``; unknown names
    raise ``ValueError`` so callers cannot mistake an empty result for a schema.
    """
    return _Reference().render(name)


def trace_api_reference(name: str | None = None) -> dict[str, Any]:
    """Return JSON-serializable reference text and the available symbol names."""
    reference = _Reference()
    return {
        "module": "autofa.trace",
        "selected_symbol": name,
        "symbol_names": list(reference.exports),
        "reference": reference.render(name),
    }
