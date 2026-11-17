# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoFA brand colors — NVIDIA green primary."""

from __future__ import annotations

from prompt_toolkit.styles import Style

from autofa.metrics.verdicts import VERDICT_PRESENTATIONS

# Brand palette
GREEN = "#76B900"  # NVIDIA green (primary)
GREEN_BRIGHT = "#8FE000"
GREEN_DIM = "#5A8F00"
MUTED = "#737373"
TEXT = "#FFFFFF"
WARN = "#E0A800"  # warnings only (not brand red)
LITERAL = "#B45F06"  # burnt orange, balanced contrast on light/dark terminals
VERDICT_COLORS = {"good": GREEN, "warn": WARN, "bad": "#CC4444", "neutral": MUTED}

# prompt_toolkit styles for arrow menus / filesystem browser
MENU_STYLE = Style.from_dict(
    {
        "title": f"bold {GREEN}",
        "hint": MUTED,
        "key": "bold",
        "literal": f"bold {LITERAL}",
        "frame.border": MUTED,
        "filter-focused frame.border": GREEN_DIM,
        "filter-label": f"bold {MUTED}",
        "filter-label-focused": f"bold {GREEN}",
        "selected": f"bold reverse {GREEN}",
        "item": "",
        "dim": MUTED,
        "header": f"bold {GREEN}",
        "accent": f"bold {GREEN}",
        "good": f"bold {GREEN}",
        "countdown": f"bg:{GREEN_BRIGHT} #101800 bold",
        "folder-recognized": f"bg:{GREEN} #101800 bold",
        "folder-confirm-selected": f"bg:{GREEN_BRIGHT} #101800 bold",
        "service-good": GREEN,
        "service-bad": "#CC4444",
        "ok": "bold #E0A800",
        "bad": "bold #CC4444",
        # Python source (explorer)
        "py.keyword": f"bold {GREEN}",
        "py.builtin": f"bold {GREEN_BRIGHT}",
        "py.function": "bold #79C0FF",
        "py.class": "bold #79C0FF",
        "py.decorator": "bold #D2A8FF",
        "py.string": "#A5D6FF",
        "py.number": "#FFA657",
        "py.comment": f"italic {MUTED}",
        "py.operator": TEXT,
        "py.punct": MUTED,
        "py.text": TEXT,
        "py.lineno": MUTED,
    }
)


def verdict_color(code: str) -> str:
    """Rich color for the same semantic tone used by the report."""
    tone = VERDICT_PRESENTATIONS.get(code, {}).get("tone", "neutral")
    return VERDICT_COLORS[tone]
