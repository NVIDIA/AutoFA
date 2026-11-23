# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared taxonomy pass-setup controls for settings and interactive classification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from rich.console import Console

from autofa.config import Config
from autofa.ui.errors import show_action_error
from autofa.ui.menus import MenuItem, is_menu_back, select_option

console = Console()

_WORKFLOWS = (
    ("llm", "llm", "LLM + LLM review", "Recommended budget-friendly default."),
    ("llm", "off", "LLM only", "One HTTP judge pass."),
    (
        "llm",
        "agentic",
        "LLM + agentic review",
        "HTTP classification followed by restricted harness review.",
    ),
    ("agentic", "off", "Agentic only", "One restricted harness classification."),
    (
        "agentic",
        "llm",
        "Agentic + LLM review",
        "Restricted harness classification followed by HTTP review.",
    ),
    (
        "agentic",
        "agentic",
        "Agentic + agentic review",
        "Two restricted harness passes.",
    ),
)


def workflow_label(first_pass: str, second_pass: str) -> str:
    for first, second, label, _detail in _WORKFLOWS:
        if (first, second) == (first_pass, second_pass):
            return label
    return f"{first_pass} + {second_pass}"


def select_taxonomy_workflow(
    first_pass: str,
    second_pass: str,
) -> tuple[str, str] | None:
    current = (first_pass, second_pass)
    values = [(first, second) for first, second, _label, _detail in _WORKFLOWS]
    choice = select_option(
        "Taxonomy pass setup",
        [
            MenuItem(label, (first, second), detail)
            for first, second, label, detail in _WORKFLOWS
        ],
        default_index=values.index(current) if current in values else 0,
        allow_back=True,
    )
    if choice is None or is_menu_back(choice):
        return None
    return choice.value


def select_agentic_max_turns(default: int) -> int | None:
    values = sorted({default, 64, 128, 256, 512, 1024, 2048})
    choice = select_option(
        "Agentic interaction limit per pass",
        [
            MenuItem(
                str(turns),
                turns,
                "Investigation-tool calls per pass.",
            )
            for turns in values
        ],
        default_index=values.index(default) if default in values else 0,
        allow_back=True,
    )
    if choice is None or is_menu_back(choice):
        return None
    return int(choice.value)


def configure_taxonomy(
    cfg: Config,
    *,
    save: Callable[[Config], None],
) -> Config:
    draft = replace(cfg)
    while True:
        choice = select_option(
            "Taxonomy classification",
            [
                MenuItem(
                    "Pass setup",
                    "workflow",
                    workflow_label(
                        draft.taxonomy_first_pass,
                        draft.taxonomy_second_pass,
                    ),
                ),
                MenuItem(
                    "Agentic interaction limit per pass",
                    "turns",
                    (
                        f"{draft.taxonomy_agentic_max_turns} · "
                        "investigation-tool calls per pass"
                    ),
                ),
                MenuItem("Back", "back"),
            ],
            allow_back=True,
        )
        if choice is None or is_menu_back(choice) or choice.value == "back":
            return draft
        candidate = draft
        if choice.value == "workflow":
            workflow = select_taxonomy_workflow(
                draft.taxonomy_first_pass,
                draft.taxonomy_second_pass,
            )
            if workflow is not None:
                candidate = replace(
                    draft,
                    taxonomy_first_pass=workflow[0],
                    taxonomy_second_pass=workflow[1],
                )
        else:
            turns = select_agentic_max_turns(draft.taxonomy_agentic_max_turns)
            if turns is not None:
                candidate = replace(draft, taxonomy_agentic_max_turns=turns)
        if candidate != draft:
            try:
                save(candidate)
            except (OSError, ValueError) as exc:
                show_action_error(exc, console=console, context="Save taxonomy settings")
                continue
            draft = candidate
