# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Home banner / ASCII logos."""

from __future__ import annotations

import webbrowser

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.mouse_events import MouseButton, MouseEventType

# NVIDIA eye — true aspect (~1.51:1), half-blocks. Same height as AutoFA (8).
NVIDIA_EYE = r"""
         ███████████████
    ▄▄▄██▄▄█▀▀██████████
 ▄▄█▀▀▄▄▄▀▀▀██▄ ▀███████
███ ▄██▀ █▄  ██▀ ▄██████
 ██▄ ██▄ ████▀ ▄███▀▀███
  ▀██▄▀▀████▄▄██▀   ▄▄██
    ▀▀██▄▀▀▀▀▀▄▄▄███████
         ███████████████
""".strip(
    "\n"
)

# Taller AutoFA (8 rows) so it matches the eye without squashing the mark.
AUTOFA_LOGO = r"""
 █████╗ ██╗   ██╗████████╗ ██████╗ ███████╗ █████╗
██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝██╔══██╗
██║  ██║██║   ██║   ██║   ██║   ██║██║     ██║  ██║
███████║██║   ██║   ██║   ██║   ██║█████╗  ███████║
██╔══██║██║   ██║   ██║   ██║   ██║██╔══╝  ██╔══██║
██║  ██║██║   ██║   ██║   ██║   ██║██║     ██║  ██║
██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║     ██║  ██║
╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝  ╚═╝
""".strip(
    "\n"
)


def _side_by_side(left: str, right: str, *, gap: str = "   ") -> list[str]:
    """Horizontally join two ASCII logos, vertically centered."""
    lrows = left.splitlines()
    rrows = right.splitlines()
    lw = max((len(r) for r in lrows), default=0)
    h = max(len(lrows), len(rrows))
    lpad = (h - len(lrows)) // 2
    rpad = (h - len(rrows)) // 2
    out: list[str] = []
    for i in range(h):
        li = i - lpad
        ri = i - rpad
        left_s = lrows[li].ljust(lw) if 0 <= li < len(lrows) else " " * lw
        right_s = rrows[ri] if 0 <= ri < len(rrows) else ""
        out.append(f"{left_s}{gap}{right_s}".rstrip())
    return out


def combined_logo_rows() -> list[str]:
    """NVIDIA eye on the left, AutoFA wordmark on the right."""
    return _side_by_side(NVIDIA_EYE, AUTOFA_LOGO)


def library_server_link(url: str, *, local: bool = True) -> StyleAndTextTuples:
    """A mouse-clickable home link, with its URL visible for terminal link detection."""
    if local:
        from autofa.local_service import validate_service_url

        url = validate_service_url(url)
    else:
        from autofa.services import validate_service_address

        url = validate_service_address(url, name="AutoFA library server")

    def open_server(event):
        if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                # A browser launch failure must not end the server-owning CLI.
                # The visible URL and keyboard action remain available.
                pass
        return None

    return [
        ("class:service-good", f"  Library: {url}",
         open_server),
        ("class:dim", "\n"),
    ]


REMOTE_LIBRARY_PREFIX = "AutoFA Library Server: "


def home_status_fragments(status_lines: list[str]) -> StyleAndTextTuples:
    """Keep home status concise; full startup details remain available in Settings."""
    lines: StyleAndTextTuples = []
    for text in status_lines:
        if text.startswith(("Loaded env:", "Config:", "detector_fields=", "mem:", "Analysis:",
                            "Iterative LLM scoring:")):
            continue
        if text.startswith("AutoFA Local Library Server: "):
            url = text.removeprefix("AutoFA Local Library Server: ").partition(" ")[0]
            lines.extend(library_server_link(url))
            continue
        if text.startswith(REMOTE_LIBRARY_PREFIX):
            lines.extend(library_server_link(text.removeprefix(REMOTE_LIBRARY_PREFIX),
                                             local=False))
            continue
        style = "class:dim"
        if text.startswith("AutoFA Cloud Library:"):
            logged_in = "logged in as @" in text
            style = "class:service-good" if logged_in else "class:service-bad"
            text = ("Cloud: @" + text.split("logged in as @", 1)[1] if logged_in
                    else "Cloud: " + text.split(":", 1)[1].split("·", 1)[0].strip()
                    + " · autofa login")
        elif text.startswith("AutoFA Local Library:"):
            style = "class:service-good" if ": ready" in text else "class:service-bad"
            text = text.replace("AutoFA Local Library:", "Library:", 1).split("·", 1)[0]
        elif text.startswith("Local dashboard unavailable:"):
            style, text = "class:service-bad", "Library unavailable · details in Settings"
        lines.append((style, f"  {text}\n"))
    return lines


def home_menu_header(status_lines: list[str] | None = None) -> StyleAndTextTuples:
    """NVIDIA eye + AutoFA logos and tagline for the home menu header."""
    lines: StyleAndTextTuples = [("", "\n")]
    for row in combined_logo_rows():
        lines.append(("class:header", f"  {row}\n"))
    lines.append(("", "\n"))
    lines.append(("class:dim", "  Automated failure discovery from eval logs\n"))
    if status_lines:
        lines.extend(home_status_fragments(status_lines))
    lines.append(("", "\n"))
    return lines
