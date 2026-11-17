# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow-key menus and filesystem browser (Claude Code–like navigation)."""

from __future__ import annotations

import asyncio
import math
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from textwrap import wrap
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from autofa.ui.errors import notice_fragments
from autofa.ui.pt_app import run_app
from autofa.ui.theme import MENU_STYLE


@dataclass
class MenuItem:
    label: str
    value: Any
    detail: str = ""
    action: str = "select"  # select | delete | publish


class MenuBack:
    """Sentinel returned when the user presses Esc/q to go back one wizard step."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MENU_BACK"

    def __bool__(self) -> bool:
        return False


MENU_BACK = MenuBack()


def is_menu_back(value: object) -> bool:
    return value is MENU_BACK


def terminal_rows() -> int:
    """Current terminal height (rows), with a sane fallback."""
    try:
        from prompt_toolkit.application.current import get_app_or_none

        app = get_app_or_none()
        if app is not None:
            return int(app.output.get_size().rows)
    except Exception:  # noqa: BLE001
        pass
    return shutil.get_terminal_size((80, 24)).lines


def menu_viewport_capacity(*, header_rows: int = 6, n_items: int | None = None) -> int:
    """How many option rows fit on screen under the title/hints."""
    available = max(1, terminal_rows() - header_rows)
    if n_items is not None and n_items > available:
        # Up/down indicators and the blank line plus position counter.
        available = max(1, available - 4)
    return available


def ensure_menu_scroll(scroll: int, index: int, n_items: int, capacity: int) -> int:
    """Keep ``index`` inside the visible window ``[scroll, scroll+capacity)``.

    Wrapping to the last item (↑ from 0) or first item (↓ from end) jumps the
    window so the selection stays on-screen.
    """
    if n_items <= 0:
        return 0
    if n_items <= capacity:
        return 0
    index = max(0, min(index, n_items - 1))
    if index < scroll:
        scroll = index
    elif index >= scroll + capacity:
        scroll = index - capacity + 1
    return max(0, min(scroll, n_items - capacity))


def prepare_step() -> None:
    """Clear the terminal so only the current menu / wizard step is visible.

    Clear prior output before the next picker takes over the screen.
    """
    try:
        from rich.console import Console

        Console().clear()
    except Exception:  # noqa: BLE001
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()


def _matches_glob(path: Path, file_glob: str | None) -> bool:
    """Match a file against a glob, or several globs separated by ``|`` / ``,``.

    Examples: ``*.parquet``, ``*.parquet|*.jsonl``, ``*.{parquet,jsonl}``.
    """
    if not file_glob:
        return True
    # Brace expansion: *.{parquet,jsonl} → *.parquet | *.jsonl
    if "{" in file_glob and "}" in file_glob:
        start = file_glob.index("{")
        end = file_glob.index("}")
        prefix = file_glob[:start]
        suffix = file_glob[end + 1 :]
        alts = file_glob[start + 1 : end].split(",")
        return any(_matches_glob(path, f"{prefix}{a.strip()}{suffix}") for a in alts)
    if "|" in file_glob or ("," in file_glob and "*." in file_glob):
        parts = [p.strip() for p in file_glob.replace("|", ",").split(",") if p.strip()]
        return any(_matches_one_glob(path, p) for p in parts)
    return _matches_one_glob(path, file_glob)


def _matches_one_glob(path: Path, file_glob: str) -> bool:
    if file_glob.startswith("*."):
        return path.suffix == file_glob[1:]
    if any(ch in file_glob for ch in "*?[]"):
        return fnmatch(path.name, file_glob)
    return path.name == file_glob


def select_option(  # noqa: C901
    title: str,
    options: list[str] | list[MenuItem],
    *,
    default_index: int = 0,
    allow_cancel: bool = True,
    allow_back: bool = False,
    allow_delete: bool = False,
    allow_publish: bool = False,
    clear_screen: bool = True,
    header: StyleAndTextTuples | Callable[[], StyleAndTextTuples] | None = None,
    footer: StyleAndTextTuples | None = None,
    auto_select_seconds: float | None = None,
) -> MenuItem | MenuBack | None:
    """Arrow-key single select.

    Returns the chosen ``MenuItem``, ``MENU_BACK`` when ``allow_back`` and the
    user presses Esc/q, or ``None`` when ``allow_cancel`` and Esc/q (and back is
    not enabled). Prefer ``allow_back`` for intermediate wizard steps so Esc
    returns to the previous menu instead of aborting the whole flow.
    """
    if not allow_cancel and not allow_back:
        raise ValueError("select_option must allow either cancel or back navigation")
    if clear_screen:
        prepare_step()
    if notice_fragments() and not header:
        header = notice_fragments()
    items: list[MenuItem] = []
    for opt in options:
        if isinstance(opt, MenuItem):
            items.append(opt)
        else:
            items.append(MenuItem(label=str(opt), value=opt))

    if not items:
        return None

    if auto_select_seconds is not None and (
        not math.isfinite(auto_select_seconds) or auto_select_seconds <= 0
    ):
        raise ValueError("Auto-selection needs a positive timeout")
    deadline: float | None = None

    state = {
        "index": max(0, min(default_index, len(items) - 1)),
        "scroll": 0,
        "done": False,
        "cancel": False,
        "back": False,
        "delete": False,
        "publish": False,
    }
    def current_header() -> StyleAndTextTuples:
        return header() if callable(header) else header or []

    rendered_footer_rows = (
        "".join(fragment[1] for fragment in footer).count("\n") if footer else 0
    )

    def _capacity() -> int:
        header_rows = "".join(fragment[1] for fragment in current_header()).count("\n")
        reserve = header_rows + 4 + (rendered_footer_rows + 1 if footer else 0)
        reserve += 2 if auto_select_seconds is not None else 0
        return menu_viewport_capacity(header_rows=reserve, n_items=len(items))

    state["scroll"] = ensure_menu_scroll(
        0, state["index"], len(items), _capacity()
    )

    def _sync_scroll() -> None:
        cap = _capacity()
        state["scroll"] = ensure_menu_scroll(
            int(state["scroll"]), int(state["index"]), len(items), cap
        )

    def _render() -> StyleAndTextTuples:
        _sync_scroll()
        cap = _capacity()
        start = int(state["scroll"])
        end = min(len(items), start + cap)
        lines: StyleAndTextTuples = []
        lines.extend(current_header())
        lines.append(("class:title", f"  {title}\n"))
        lines.extend(
            [
                ("class:hint", "  "),
                ("class:key", "↑/↓"),
                ("class:hint", " move · "),
                ("class:key", "PgUp/PgDn"),
                ("class:hint", " page · "),
                ("class:key", "Enter"),
                ("class:hint", " select"),
            ]
        )
        if allow_delete:
            lines.extend([("class:hint", " · "), ("class:key", "d"), ("class:hint", " delete")])
        if allow_publish:
            lines.extend([("class:hint", " · "), ("class:key", "p"), ("class:hint", " publish")])
        if allow_back:
            lines.extend([("class:hint", " · "), ("class:key", "Esc/q"), ("class:hint", " back")])
        elif allow_cancel:
            lines.extend([("class:hint", " · "), ("class:key", "Esc/q"), ("class:hint", " cancel")])
        lines.append(("", "\n\n"))
        if start > 0:
            lines.append(("class:dim", f"  … {start} more above\n"))
        for i in range(start, end):
            item = items[i]
            prefix = "❯ " if i == state["index"] else "  "
            style = "class:selected" if i == state["index"] else "class:item"
            label = item.label
            if item.detail:
                label = f"{label}  {item.detail}"
            if i == state["index"]:
                lines.append(("[SetCursorPosition]", ""))
            lines.append((style, f"{prefix}{label}\n"))
        if end < len(items):
            lines.append(("class:dim", f"  … {len(items) - end} more below\n"))
        if len(items) > cap:
            lines.append(
                ("class:dim", f"\n  ({state['index'] + 1}/{len(items)})\n")
            )
        if footer:
            lines.append(("", "\n"))
            lines.extend(footer)
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event) -> None:  # noqa: ANN001
        state["index"] = (state["index"] - 1) % len(items)
        _sync_scroll()
        event.app.invalidate()

    @kb.add("down")
    @kb.add("c-n")
    def _down(event) -> None:  # noqa: ANN001
        state["index"] = (state["index"] + 1) % len(items)
        _sync_scroll()
        event.app.invalidate()

    @kb.add("pageup")
    def _pageup(event) -> None:  # noqa: ANN001
        cap = _capacity()
        state["index"] = max(0, state["index"] - cap)
        _sync_scroll()
        event.app.invalidate()

    @kb.add("pagedown")
    def _pagedown(event) -> None:  # noqa: ANN001
        cap = _capacity()
        state["index"] = min(len(items) - 1, state["index"] + cap)
        _sync_scroll()
        event.app.invalidate()

    @kb.add("home")
    @kb.add("c-a")
    def _home(event) -> None:  # noqa: ANN001
        state["index"] = 0
        _sync_scroll()
        event.app.invalidate()

    @kb.add("end")
    @kb.add("c-e")
    def _end(event) -> None:  # noqa: ANN001
        state["index"] = len(items) - 1
        _sync_scroll()
        event.app.invalidate()

    @kb.add("enter")
    def _enter(event) -> None:  # noqa: ANN001
        state["done"] = True
        event.app.exit()

    if allow_delete:

        @kb.add("d")
        @kb.add("delete")
        def _delete(event) -> None:  # noqa: ANN001
            state["delete"] = True
            state["done"] = True
            event.app.exit()

    if allow_publish:

        @kb.add("p")
        def _publish(event) -> None:  # noqa: ANN001
            state["publish"] = True
            state["done"] = True
            event.app.exit()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event) -> None:  # noqa: ANN001
        if allow_back:
            state["back"] = True
            event.app.exit()
            return
        if not allow_cancel:
            return
        state["cancel"] = True
        event.app.exit()

    def render_countdown() -> StyleAndTextTuples:
        remaining = (max(0, math.ceil(deadline - time.monotonic())) if deadline is not None
                     else math.ceil(auto_select_seconds or 0))
        return [("class:countdown", f"  Auto-select in {remaining}s  \n"),
                ("class:accent", f"  → {items[state['index']].label}")]

    windows = [Window(FormattedTextControl(_render), always_hide_cursor=True)]
    if auto_select_seconds is not None:
        # Separate, fixed-height window: scrolling/wrapping cannot hide the timer.
        windows.insert(0, Window(FormattedTextControl(render_countdown), height=2,
                                 always_hide_cursor=True))
    app = Application(
        layout=Layout(HSplit(windows)),
        key_bindings=kb,
        style=MENU_STYLE,
        full_screen=clear_screen,
        mouse_support=any(len(fragment) == 3 for fragment in [*current_header(), *(footer or [])]),
    )

    async def countdown() -> None:
        while deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state["done"] = True
                app.exit()
                return
            app.invalidate()
            await asyncio.sleep(min(1, remaining))

    def start_countdown() -> None:
        nonlocal deadline
        deadline = time.monotonic() + auto_select_seconds
        app.create_background_task(countdown())

    if auto_select_seconds is not None:
        app.pre_run_callables.append(start_countdown)
    run_app(app)
    if state["back"]:
        return MENU_BACK
    if state["cancel"]:
        return None
    if not state["done"]:
        raise RuntimeError(f"Menu '{title}' closed without a selection or cancellation")
    chosen = items[state["index"]]
    action = "select"
    if state["delete"]:
        action = "delete"
    elif state["publish"]:
        action = "publish"
    return MenuItem(
        label=chosen.label,
        value=chosen.value,
        detail=chosen.detail,
        action=action,
    )

def select_yes_no(
    title: str,
    *,
    default_yes: bool = True,
    allow_back: bool = False,
) -> bool | MenuBack | None:
    """Select Yes or No, while always allowing Escape/Ctrl+C to leave the prompt.

    Nested wizard steps can set ``allow_back=True`` to receive ``MENU_BACK``.
    Other callers receive ``None`` so confirmation prompts safely cancel.
    """
    yes_label = "Yes (default)" if default_yes else "Yes"
    no_label = "No (default)" if not default_yes else "No"
    choice = select_option(
        title,
        [MenuItem(yes_label, True), MenuItem(no_label, False)],
        default_index=0 if default_yes else 1,
        allow_cancel=not allow_back,
        allow_back=allow_back,
    )
    if is_menu_back(choice):
        return MENU_BACK
    if choice is None:
        return None
    return bool(choice.value)


def browse_filesystem(  # noqa: C901
    start: str | Path | None = None,
    *,
    title: str = "Select path",
    select_files: bool = True,
    select_dirs: bool = True,
    file_glob: str | None = "*.parquet",
    allow_select_dir_with_matches: bool = True,
    clear_screen: bool = True,
    folder_description: Callable[[Path], str | None] | None = None,
) -> Path | None:
    """Interactive filesystem browser with arrow keys.

    Listings are shallow (current directory only) and cached per folder so
    ↑/↓ stay responsive. No recursive tree walks while navigating.
    """
    if clear_screen:
        prepare_step()
    cwd = Path(start or Path.home()).expanduser()
    if cwd.is_file():
        cwd = cwd.parent
    if not cwd.exists():
        cwd = Path.home()
    cwd = cwd.resolve()

    state: dict[str, Any] = {
        "cwd": cwd,
        "index": 0,
        "result": None,
        "cancel": False,
        "done": False,
        "cache": {},  # cwd str -> list[MenuItem]
        "errors": {},
        "recognized": {},
    }

    def _build_entries(cur: Path) -> list[MenuItem]:
        import os

        entries: list[MenuItem] = []
        if cur.parent != cur:
            entries.append(MenuItem("..", cur.parent, detail="(up)"))

        if select_dirs and allow_select_dir_with_matches:
            # No recursive walks while browsing — discovery runs after select.
            detail = f"all {file_glob or 'files'} below" if file_glob else "select folder"
            if folder_description:
                try:
                    recognized = folder_description(cur)
                    if recognized:
                        state["recognized"][str(cur)] = recognized
                        detail = "Enter to use"
                except OSError:
                    pass  # The listing below displays permission/read errors.
            entries.append(MenuItem("[ USE THIS FOLDER ]", cur, detail=detail))

        dirs: list[Path] = []
        files: list[Path] = []
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    name = entry.name
                    if name.startswith("."):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(Path(entry.path))
                        elif select_files and entry.is_file(follow_symlinks=False):
                            p = Path(entry.path)
                            if _matches_glob(p, file_glob):
                                files.append(p)
                    except OSError as exc:
                        state["errors"][str(cur)] = f"Could not inspect {entry.name}: {exc}"
                        continue
        except OSError as exc:
            state["errors"][str(cur)] = f"Cannot read this folder: {exc}. Check permissions."

        for d in sorted(dirs, key=lambda p: p.name.lower()):
            entries.append(MenuItem(f"{d.name}/", d, detail="dir"))
        for f in sorted(files, key=lambda p: p.name.lower()):
            try:
                size = f.stat().st_size
                human = f"{size / 1e9:.2f} GB" if size > 1e9 else f"{size / 1e6:.1f} MB"
            except OSError as exc:
                human = f"unavailable: {exc}"
            entries.append(MenuItem(f.name, f, detail=human))
        return entries

    def _entries() -> list[MenuItem]:
        key = str(state["cwd"])
        cache: dict = state["cache"]
        if key not in cache:
            cache[key] = _build_entries(state["cwd"])
        return cache[key]

    def _focus_folder_option() -> None:
        for i, e in enumerate(_entries()):
            if "USE THIS FOLDER" in e.label:
                state["index"] = i
                return
        state["index"] = 0

    _focus_folder_option()

    state["scroll"] = 0

    def _render() -> list[tuple[str, str]]:
        entries = _entries()
        if state["index"] >= len(entries):
            state["index"] = max(0, len(entries) - 1)
        error = state["errors"].get(str(state["cwd"]))
        recognized = state["recognized"].get(str(state["cwd"]))
        width = max(1, min(72, app.output.get_size().columns - 4))
        recognition_lines = wrap(f"✓ {recognized}", width=width) if recognized else []
        if recognized and len(recognition_lines) > 1 and " — " in recognized:
            message, question = recognized.split(" — ", 1)
            recognition_lines = [*wrap(f"✓ {message}", width=width),
                                 *wrap(f"— {question}", width=width)]
        cap = menu_viewport_capacity(
            header_rows=5 + len(recognition_lines) + (2 if error else 0), n_items=len(entries),
        )
        state["scroll"] = ensure_menu_scroll(
            int(state.get("scroll", 0)), int(state["index"]), len(entries), cap
        )
        start_i = int(state["scroll"])
        end_i = min(len(entries), start_i + cap)
        lines: list[tuple[str, str]] = [
            ("class:title", f"  {title}\n"),
            ("class:dim", f"  {state['cwd']}\n"),
            ("class:hint", "  "),
            ("class:key", "↑/↓"),
            ("class:hint", " move · "),
            ("class:key", "Enter"),
            ("class:hint", " open/select · "),
            ("class:key", "←/Backspace"),
            ("class:hint", " up · "),
            ("class:key", "Esc"),
            ("class:hint", " back\n\n"),
        ]
        if error:
            lines.append(("class:bad", f"  {error}\n\n"))
        if not entries:
            lines.append(("class:dim", "  (empty)\n"))
            return lines
        if start_i > 0:
            lines.append(("class:dim", f"  … {start_i} more above\n"))
        for i in range(start_i, end_i):
            item = entries[i]
            prefix = "❯ " if i == state["index"] else "  "
            style = "class:selected" if i == state["index"] else "class:item"
            if recognized and "USE THIS FOLDER" in item.label:
                style = ("class:folder-confirm-selected" if i == state["index"]
                         else "class:folder-recognized")
                if i == state["index"]:
                    lines.append(("[SetCursorPosition]", ""))
                # One selectable block: recognition sits directly above its action.
                for line in recognition_lines:
                    lines.append((style, f"  {line.ljust(width)}\n"))
                lines.append((style, f"{prefix}{item.label.ljust(width)}\n"))
                continue
            label = item.label
            if item.detail:
                label = f"{label}  {item.detail}"
            if i == state["index"]:
                lines.append(("[SetCursorPosition]", ""))
            lines.append((style, f"{prefix}{label}\n"))
        if end_i < len(entries):
            lines.append(("class:dim", f"  … {len(entries) - end_i} more below\n"))
        if len(entries) > cap:
            lines.append(("class:dim", f"\n  ({state['index'] + 1}/{len(entries)})\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event) -> None:  # noqa: ANN001
        entries = _entries()
        if entries:
            state["index"] = (state["index"] - 1) % len(entries)
            event.app.invalidate()

    @kb.add("down")
    @kb.add("c-n")
    def _down(event) -> None:  # noqa: ANN001
        entries = _entries()
        if entries:
            state["index"] = (state["index"] + 1) % len(entries)
            event.app.invalidate()

    @kb.add("left")
    @kb.add("backspace")
    def _up_dir(event) -> None:  # noqa: ANN001
        cur: Path = state["cwd"]
        if cur.parent != cur:
            state["cwd"] = cur.parent
            _focus_folder_option()
            event.app.invalidate()

    @kb.add("enter")
    def _enter(event) -> None:  # noqa: ANN001
        entries = _entries()
        if not entries:
            return
        item = entries[state["index"]]
        path: Path = item.value
        if item.label == "..":
            state["cwd"] = path
            _focus_folder_option()
            event.app.invalidate()
            return
        if "USE THIS FOLDER" in item.label:
            if select_dirs:
                state["result"] = path
                state["done"] = True
                event.app.exit()
            return
        if path.is_dir():
            state["cwd"] = path
            _focus_folder_option()
            event.app.invalidate()
            return
        if path.is_file() and select_files:
            state["result"] = path
            state["done"] = True
            event.app.exit()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event) -> None:  # noqa: ANN001
        state["cancel"] = True
        event.app.exit()

    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(_render), always_hide_cursor=True)])),
        key_bindings=kb,
        style=MENU_STYLE,
        full_screen=clear_screen,
    )
    run_app(app)
    if state["cancel"]:
        return None
    if not state["done"]:
        raise RuntimeError(f"File browser '{title}' closed unexpectedly")
    return state["result"]


def select_items_multi(
    options: list[MenuItem],
    *,
    title: str = "Select items",
    empty_means_all: bool = False,
    clear_screen: bool = True,
    confirm: bool = False,
    confirm_noun: str = "items",
) -> list[Any] | None:
    """Multi-select with Enter toggle and a Done row at the top.

    - ↑/↓ move
    - Enter on a row toggles it
    - Enter on **Done** (first row) confirms
    - ``a`` toggles select-all
    - Esc cancels

    Returns selected ``value``s, or None if cancelled.
    When ``empty_means_all`` and every item is selected (via ``a`` / all toggled),
    or Done with nothing checked, returns ``[]`` meaning “all”.
    When ``confirm`` is True, shows a Yes/No summary before accepting.
    """
    if not options:
        return []
    if len(options) == 1 and not confirm:
        return [options[0].value]

    while True:
        picked = _select_items_multi_once(
            options,
            title=title,
            empty_means_all=empty_means_all,
            clear_screen=clear_screen,
        )
        if picked is None:
            return None
        if not confirm:
            return picked
        summary = _format_multi_summary(
            picked, options, empty_means_all=empty_means_all, noun=confirm_noun
        )
        ok = select_yes_no(summary, default_yes=True)
        if ok:
            return picked
        clear_screen = True


def _format_multi_summary(
    picked: list[Any],
    options: list[MenuItem],
    *,
    empty_means_all: bool,
    noun: str,
) -> str:
    if not picked:
        if empty_means_all:
            names = [str(o.value) for o in options]
            preview = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
            return f"You selected ALL {len(options)} {noun} ({preview}). Is this OK?"
        return "Nothing selected. Is this OK?"
    names = [str(v) for v in picked]
    if len(names) == 1:
        return f"You selected {names[0]}. Is this OK?"
    if len(names) <= 8:
        return f"You selected {', '.join(names)}. Is this OK?"
    head = ", ".join(names[:6])
    return f"You selected {len(names)} {noun}: {head}, …. Is this OK?"


def _select_items_multi_once(  # noqa: C901
    options: list[MenuItem],
    *,
    title: str,
    empty_means_all: bool,
    clear_screen: bool,
) -> list[Any] | None:
    if clear_screen:
        prepare_step()
    selected: set[int] = set()
    # index 0 = Done row; 1..n = options[0..n-1]
    n_opts = len(options)
    state = {"index": 1 if n_opts else 0, "finished": False, "cancel": False, "error": ""}

    def _is_done_row(i: int) -> bool:
        return i == 0

    def _opt_i(i: int) -> int:
        return i - 1

    def _render() -> list[tuple[str, str]]:
        n = len(selected)
        all_on = n == n_opts and n_opts > 0
        if all_on and empty_means_all:
            count = f"all {n_opts}"
        else:
            count = str(n)
        lines: list[tuple[str, str]] = [
            ("class:title", f"  {title}\n"),
            ("class:hint", "  "),
            ("class:key", "↑/↓"),
            ("class:hint", " move · "),
            ("class:key", "Enter"),
            ("class:hint", " toggle · "),
            ("class:key", "a"),
            ("class:hint", " all · Done (top) + "),
            ("class:key", "Enter"),
            ("class:hint", " to finish · "),
            ("class:key", "Esc"),
            ("class:hint", " cancel\n\n"),
        ]
        done_focus = state["index"] == 0
        if state["error"]:
            lines.append(("class:bad", f"  {state['error']}\n"))
        done_prefix = "❯ " if done_focus else "  "
        done_style = "class:selected" if done_focus else "class:title"
        done_label = "[ DONE ]  confirm selection"
        if n == 0 and empty_means_all:
            done_label += "  (none checked → all)"
        elif all_on and empty_means_all:
            done_label += "  (all)"
        elif n:
            done_label += f"  ({n})"
        if done_focus:
            lines.append(("[SetCursorPosition]", ""))
        lines.append((done_style, f"{done_prefix}{done_label}\n"))
        lines.append(("class:dim", "  ─\n"))

        for i, item in enumerate(options):
            row = i + 1
            on = i in selected
            mark = "[x]" if on else "[ ]"
            prefix = "❯ " if row == state["index"] else "  "
            style = "class:selected" if row == state["index"] else "class:item"
            label = item.label
            if item.detail:
                label = f"{label}  {item.detail}"
            if row == state["index"]:
                lines.append(("[SetCursorPosition]", ""))
            lines.append((style, f"{prefix}{mark} {label}\n"))
        lines.append(("class:dim", f"\n  {count} selected\n"))
        return lines

    def _toggle_at(opt_idx: int) -> None:
        if opt_idx in selected:
            selected.discard(opt_idx)
        else:
            selected.add(opt_idx)

    kb = KeyBindings()
    total_rows = n_opts + 1  # Done + options

    @kb.add("up")
    @kb.add("c-p")
    def _up(event) -> None:  # noqa: ANN001
        state["index"] = (state["index"] - 1) % total_rows
        event.app.invalidate()

    @kb.add("down")
    @kb.add("c-n")
    def _down(event) -> None:  # noqa: ANN001
        state["index"] = (state["index"] + 1) % total_rows
        event.app.invalidate()

    @kb.add("space")
    @kb.add("enter")
    def _enter_or_space(event) -> None:  # noqa: ANN001
        i = state["index"]
        if _is_done_row(i):
            if not selected and not empty_means_all:
                state["error"] = "Select at least one item, or press Esc to cancel."
                event.app.invalidate()
                return
            state["finished"] = True
            event.app.exit()
            return
        _toggle_at(_opt_i(i))
        event.app.invalidate()

    @kb.add("a")
    def _all(event) -> None:  # noqa: ANN001
        if len(selected) == n_opts:
            selected.clear()
        else:
            selected.update(range(n_opts))
        event.app.invalidate()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event) -> None:  # noqa: ANN001
        state["cancel"] = True
        event.app.exit()

    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(_render), always_hide_cursor=True)])),
        key_bindings=kb,
        style=MENU_STYLE,
        full_screen=clear_screen,
    )
    run_app(app)
    if state["cancel"]:
        return None
    if not state["finished"]:
        raise RuntimeError(f"Selection '{title}' closed unexpectedly")

    if len(selected) == n_opts and empty_means_all:
        return []
    if not selected:
        if empty_means_all:
            return []
        return None
    return [options[i].value for i in sorted(selected)]


def select_paths_multi(
    paths: list[Path],
    *,
    title: str = "Select parquet files",
    clear_screen: bool = True,
) -> list[Path] | None:
    """Multi-select with Space toggle, Enter confirm."""
    if not paths:
        return []
    if len(paths) == 1:
        return list(paths)

    if clear_screen:
        prepare_step()
    labels = []
    for p in paths:
        size = p.stat().st_size
        human = f"{size / 1e9:.2f} GB" if size > 1e9 else f"{size / 1e6:.1f} MB"
        labels.append(f"{p.name}  ({human})  — {p.parent.name}/")

    selected: set[int] = {0}
    state = {"index": 0, "done": False, "cancel": False}

    def _render() -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [
            ("class:title", f"  {title}\n"),
            ("class:hint", "  "),
            ("class:key", "↑/↓"),
            ("class:hint", " move · "),
            ("class:key", "Space"),
            ("class:hint", " toggle · "),
            ("class:key", "a"),
            ("class:hint", " all · "),
            ("class:key", "Enter"),
            ("class:hint", " confirm · "),
            ("class:key", "Esc"),
            ("class:hint", " cancel\n\n"),
        ]
        for i, label in enumerate(labels):
            mark = "[x]" if i in selected else "[ ]"
            prefix = "❯ " if i == state["index"] else "  "
            style = "class:selected" if i == state["index"] else "class:item"
            if i == state["index"]:
                lines.append(("[SetCursorPosition]", ""))
            lines.append((style, f"{prefix}{mark} {label}\n"))
        lines.append(("class:dim", f"\n  {len(selected)} selected\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:  # noqa: ANN001
        state["index"] = (state["index"] - 1) % len(paths)
        event.app.invalidate()

    @kb.add("down")
    def _down(event) -> None:  # noqa: ANN001
        state["index"] = (state["index"] + 1) % len(paths)
        event.app.invalidate()

    @kb.add("space")
    def _toggle(event) -> None:  # noqa: ANN001
        i = state["index"]
        if i in selected:
            if len(selected) > 1:
                selected.discard(i)
        else:
            selected.add(i)
        event.app.invalidate()

    @kb.add("a")
    def _all(event) -> None:  # noqa: ANN001
        if len(selected) == len(paths):
            selected.clear()
            selected.add(state["index"])
        else:
            selected.update(range(len(paths)))
        event.app.invalidate()

    @kb.add("enter")
    def _enter(event) -> None:  # noqa: ANN001
        state["done"] = True
        event.app.exit()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event) -> None:  # noqa: ANN001
        state["cancel"] = True
        event.app.exit()

    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(_render), always_hide_cursor=True)])),
        key_bindings=kb,
        style=MENU_STYLE,
        full_screen=clear_screen,
    )
    run_app(app)
    if state["cancel"]:
        return None
    if not state["done"]:
        raise RuntimeError(f"File selection '{title}' closed unexpectedly")
    return [paths[i] for i in sorted(selected)]
