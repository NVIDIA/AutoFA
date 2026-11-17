# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Searchable, multi-select table for remote rollout-set catalogs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.widgets import Frame

from autofa.ui.menus import ensure_menu_scroll, prepare_step, terminal_rows
from autofa.ui.pt_app import run_app
from autofa.ui.theme import MENU_STYLE

_FIELD_ALIASES = {
    "id": "id",
    "date": "date",
    "model": "model",
    "benchmark": "benchmark",
    "bench": "benchmark",
    "harness": "harness",
    "rollouts": "rollouts",
    "count": "rollouts",
    "user": "user",
    "name": "name",
    "label": "name",
}

_COLUMNS = (
    ("model", "MODEL", 38),
    ("benchmark", "BENCHMARK", 24),
    ("harness", "HARNESS", 16),
    ("rollouts", "ROLLOUTS", 10),
    ("user", "USER", 16),
    ("date", "DATE", 19),
    ("name", "NAME", 30),
    ("id", "ID", 18),
)

_SORT_FIELDS = tuple(key for key, _, _ in _COLUMNS)
_SORT_LABELS = {key: title.title() for key, title, _ in _COLUMNS}
_DESCENDING_DEFAULT_FIELDS = {"date", "rollouts"}


@dataclass(frozen=True)
class IndexedRollout:
    """One catalog record plus normalized display/search values."""

    run: dict[str, Any]
    display: dict[str, str]
    search: dict[str, str]
    search_all: str
    sort_key: tuple[str, str]

    @property
    def run_id(self) -> str:
        return str(self.run.get("run_id") or "")


@dataclass(frozen=True)
class FilteredRollouts:
    """Result of applying a live query to an indexed catalog."""

    rows: list[IndexedRollout]
    error: str = ""


def _text(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def _date_value(run: dict[str, Any]) -> tuple[str, str]:
    raw = str(run.get("run_ts") or run.get("ingested_at") or "")
    return raw[:19].replace("T", " ") if raw else "—", raw


def index_rollout_runs(runs: list[dict[str, Any]]) -> list[IndexedRollout]:
    """Normalize catalog rows once and return them newest-first."""
    indexed: list[IndexedRollout] = []
    for run in runs:
        run_id = str(run.get("run_id") or "").strip()
        if not run_id:
            continue
        date_display, date_raw = _date_value(run)
        n_rollouts = run.get("n_rollouts")
        if isinstance(n_rollouts, int):
            rollouts_display = f"{n_rollouts:,}"
            rollouts_search = f"{n_rollouts} {rollouts_display}"
        else:
            rollouts_display = _text(n_rollouts)
            rollouts_search = rollouts_display
        display = {
            "id": run_id,
            "date": date_display,
            "model": _text(run.get("model")),
            "benchmark": _text(run.get("benchmark")),
            "harness": _text(run.get("harness")),
            "rollouts": rollouts_display,
            "user": _text(run.get("source_user")),
            "name": _text(run.get("label")),
        }
        search = {key: value.casefold() for key, value in display.items()}
        search["date"] = f"{date_display} {date_raw}".casefold()
        search["rollouts"] = rollouts_search.casefold()
        indexed.append(
            IndexedRollout(
                run=run,
                display=display,
                search=search,
                search_all=" ".join(search.values()),
                sort_key=(date_raw, run_id),
            )
        )
    indexed.sort(key=lambda item: item.sort_key, reverse=True)
    return indexed


def filter_rollout_runs(
    rows: list[IndexedRollout], query: str
) -> FilteredRollouts:
    """Apply case-insensitive global and ``field:value`` substring filters."""
    global_terms: list[str] = []
    field_terms: list[tuple[str, str]] = []
    invalid_fields: list[str] = []
    for raw_term in query.split():
        if ":" not in raw_term:
            global_terms.append(raw_term.casefold())
            continue
        raw_field, value = raw_term.split(":", 1)
        canonical = _FIELD_ALIASES.get(raw_field.casefold())
        if canonical is None:
            invalid_fields.append(raw_field)
            continue
        if value:
            field_terms.append((canonical, value.casefold()))

    matched = [
        row
        for row in rows
        if all(term in row.search_all for term in global_terms)
        and all(term in row.search[field] for field, term in field_terms)
    ]
    error = ""
    if invalid_fields:
        unique = ", ".join(dict.fromkeys(invalid_fields))
        error = f"Unknown field: {unique}"
    return FilteredRollouts(matched, error)


def _terminal_columns() -> int:
    try:
        app = get_app_or_none()
        if app is not None:
            return int(app.output.get_size().columns)
    except Exception:  # noqa: BLE001
        pass
    return shutil.get_terminal_size((100, 24)).columns


def _fit_cell(value: str, width: int) -> str:
    if len(value) > width:
        value = value[: max(1, width - 1)] + "…"
    return value.ljust(width)


def format_rollout_row(
    row: IndexedRollout | None = None,
    *,
    sort_field: str | None = None,
    sort_desc: bool = False,
) -> str:
    """Format a complete table header or row before horizontal clipping."""
    if row is not None:
        values = row.display
    else:
        values = {
            key: f"{title} {'↓' if sort_desc else '↑'}" if key == sort_field else title
            for key, title, _ in _COLUMNS
        }
    return " │ ".join(_fit_cell(values[key], width) for key, _, width in _COLUMNS)


class _RolloutTableState:
    def __init__(self, rows: list[IndexedRollout]):
        self.all_rows = rows
        self.sort_field = "date"
        self.sort_desc = True
        self.sorting = False
        self.sort_index = _SORT_FIELDS.index(self.sort_field)
        self.rows = self._sorted_rows(rows)
        self.query = ""
        self.filter_error = ""
        self.index = 0
        self.scroll = 0
        self.horizontal = 0
        self.selected: set[str] = set()

    def capacity(self) -> int:
        return max(3, terminal_rows() - 11)

    def sort_label(self) -> str:
        arrow = "↓" if self.sort_desc else "↑"
        return f"{_SORT_LABELS[self.sort_field]} {arrow}"

    def _sort_value(self, row: IndexedRollout) -> Any:
        if self.sort_field == "date":
            return row.sort_key[0]
        if self.sort_field == "rollouts":
            value = row.run.get("n_rollouts")
            return value if isinstance(value, int) else None
        value = row.search[self.sort_field]
        return None if row.display[self.sort_field] == "—" else value

    def _sorted_rows(self, rows: list[IndexedRollout]) -> list[IndexedRollout]:
        present = [row for row in rows if self._sort_value(row) not in {None, ""}]
        missing = [row for row in rows if self._sort_value(row) in {None, ""}]
        present.sort(key=lambda row: row.run_id)
        present.sort(key=self._sort_value, reverse=self.sort_desc)
        missing.sort(key=lambda row: row.run_id)
        return present + missing

    def apply_sort(self, field: str, *, descending: bool | None = None) -> None:
        if field not in _SORT_FIELDS:
            raise ValueError(f"Unknown sort field: {field}")
        previous_id = self.selected_row_id()
        changed_field = field != self.sort_field
        self.sort_field = field
        if descending is not None:
            self.sort_desc = descending
        elif changed_field:
            self.sort_desc = field in _DESCENDING_DEFAULT_FIELDS
        self.sort_index = _SORT_FIELDS.index(field)
        self.rows = self._sorted_rows(self.rows)
        self._restore_row(previous_id)

    def reverse_sort(self) -> None:
        self.apply_sort(self.sort_field, descending=not self.sort_desc)

    def _restore_row(self, run_id: str) -> None:
        if run_id:
            new_index = next(
                (i for i, row in enumerate(self.rows) if row.run_id == run_id),
                None,
            )
        else:
            new_index = None
        self.index = (
            new_index
            if new_index is not None
            else min(self.index, max(0, len(self.rows) - 1))
        )
        self.sync_scroll()

    def selected_row_id(self) -> str:
        if not self.rows:
            return ""
        return self.rows[min(self.index, len(self.rows) - 1)].run_id

    def apply_query(self, query: str) -> None:
        previous_id = self.selected_row_id()
        result = filter_rollout_runs(self.all_rows, query)
        self.query = query
        self.rows = self._sorted_rows(result.rows)
        self.filter_error = result.error
        self._restore_row(previous_id)

    def sync_scroll(self) -> None:
        self.scroll = ensure_menu_scroll(
            self.scroll, self.index, len(self.rows), self.capacity()
        )

    def move(self, amount: int) -> None:
        if not self.rows:
            return
        self.index = max(0, min(len(self.rows) - 1, self.index + amount))
        self.sync_scroll()

    def toggle(self) -> None:
        run_id = self.selected_row_id()
        if not run_id:
            return
        if run_id in self.selected:
            self.selected.remove(run_id)
        else:
            self.selected.add(run_id)

    def chosen(self) -> list[dict[str, Any]]:
        return [
            row.run
            for row in self._sorted_rows(self.all_rows)
            if row.run_id in self.selected
        ]

    def toggle_visible(self) -> None:
        visible = {row.run_id for row in self.rows}
        if visible.issubset(self.selected):
            self.selected.difference_update(visible)
        else:
            self.selected.update(visible)

    def selection_summary(self) -> str:
        if not self.selected:
            return "Check rollout sets with Space or a before downloading."
        benchmarks = sorted({
            row.display["benchmark"] for row in self.all_rows if row.run_id in self.selected
        })
        hidden = len(self.selected - {row.run_id for row in self.rows})
        summary = "Benchmarks: " + ", ".join(benchmarks)
        if hidden:
            summary += f" · {hidden} selected rollout set(s) hidden by filter"
        return summary


def select_rollout_runs(  # noqa: C901
    runs: list[dict[str, Any]],
    *,
    title: str = "Select remote rollouts",
    clear_screen: bool = True,
) -> list[dict[str, Any]] | None:
    """Return explicitly checked catalog rows, including those hidden by a filter."""
    indexed = index_rollout_runs(runs)
    if not indexed:
        return []
    if clear_screen:
        prepare_step()

    state = _RolloutTableState(indexed)
    result: dict[str, Any] = {"done": False, "cancel": False, "rows": []}

    def _invalidate() -> None:
        app = get_app_or_none()
        if app is not None:
            app.invalidate()

    def _query_changed(buffer: Buffer) -> None:
        state.apply_query(buffer.text)
        _invalidate()

    filter_buffer = Buffer(multiline=False, on_text_changed=_query_changed)

    def _filter_prompt() -> list[tuple[str, str]]:
        app = get_app_or_none()
        style = (
            "class:filter-label-focused"
            if app is not None and app.layout.has_focus(filter_buffer)
            else "class:filter-label"
        )
        return [(style, " Filter › ")]

    filter_control = BufferControl(
        buffer=filter_buffer,
        input_processors=[BeforeInput(_filter_prompt)],
        focusable=True,
    )

    def _header() -> list[tuple[str, str]]:
        return [
            ("class:title", f"  {title}\n"),
            ("class:hint", "  Type to filter every column · "),
            ("class:literal", "field:value"),
            ("class:hint", " filters one column · Example: "),
            ("class:literal", "model:nemotron-3.5"),
            ("class:hint", "\n"),
            (
                "class:hint",
                "  Fields: model benchmark harness user name date rollouts id\n",
            ),
        ]

    def _table() -> list[tuple[str, str]]:
        state.sync_scroll()
        app = get_app_or_none()
        table_focused = bool(app is not None and app.layout.has_focus(table_control))
        terminal_columns = _terminal_columns()
        columns = max(20, terminal_columns - 7)
        header = format_rollout_row(
            sort_field=state.sort_field,
            sort_desc=state.sort_desc,
        )
        full_width = len(header)
        state.horizontal = max(0, min(state.horizontal, max(0, full_width - columns)))

        def clipped(text: str) -> str:
            return text[state.horizontal : state.horizontal + columns]

        left_hint = "‹" if state.horizontal else " "
        right_hint = "›" if state.horizontal + columns < full_width else " "
        if state.sorting:
            lines: list[tuple[str, str]] = [
                ("class:title", "  Sort by\n"),
                ("class:dim", f"  {'─' * max(1, terminal_columns - 2)}\n"),
            ]
            for i, field in enumerate(_SORT_FIELDS):
                active = i == state.sort_index
                pointer = "❯" if active else " "
                arrow = "↓" if state.sort_desc else "↑"
                current = f"  {arrow} current" if field == state.sort_field else ""
                style = "class:selected" if active else "class:item"
                lines.append(
                    (style, f"  {pointer} {_SORT_LABELS[field]}{current}\n")
                )
            return lines
        lines: list[tuple[str, str]] = [
            (
                "class:header",
                f"       {left_hint}{clipped(header)}{right_hint}\n",
            ),
            ("class:dim", f"  {'─' * max(1, terminal_columns - 2)}\n"),
        ]
        if not state.rows:
            lines.append(("class:dim", "  No rollout sets match this filter.\n"))
            return lines
        end = min(len(state.rows), state.scroll + state.capacity())
        for i in range(state.scroll, end):
            row = state.rows[i]
            active = table_focused and i == state.index
            pointer = "❯" if active else " "
            mark = "[x]" if row.run_id in state.selected else "[ ]"
            style = "class:selected" if active else "class:item"
            lines.append((style, f"  {pointer} {mark} {clipped(format_rollout_row(row))}\n"))
        return lines

    def _footer() -> list[tuple[str, str]]:
        if state.sorting:
            return [
                ("class:dim", f"  Current sort: {state.sort_label()}\n"),
                ("class:hint", "  "),
                ("class:key", "↑/↓"),
                ("class:hint", " choose · "),
                ("class:key", "Enter"),
                ("class:hint", " apply · "),
                ("class:key", "Esc"),
                ("class:hint", " close\n"),
            ]
        lines: list[tuple[str, str]] = [
            (
                "class:dim",
                f"  {len(state.rows):,} / {len(state.all_rows):,} matches · ",
            ),
            ("class:key", f"{len(state.selected):,}"),
            ("class:dim", " selected"),
        ]
        if state.filter_error:
            lines.append(("class:dim", f" · {state.filter_error}"))
        lines.extend(
            [
                ("class:dim", f"\n  {state.selection_summary()}\n"),
                ("class:hint", "  "),
                ("class:key", "Tab"),
                ("class:hint", " toggle filter/table · "),
                ("class:key", "↑/↓"),
                ("class:hint", " move · "),
                ("class:key", "PgUp/PgDn"),
                ("class:hint", " page · "),
                ("class:key", "←/→"),
                ("class:hint", " columns · "),
                ("class:key", "Space"),
                ("class:hint", " select · "),
                ("class:key", "a"),
                ("class:hint", " select all matches · "),
                ("class:key", "Enter"),
                ("class:hint", " download selected · "),
                ("class:key", "s"),
                ("class:hint", f" sort by ({state.sort_label()}) · "),
                ("class:key", "r"),
                ("class:hint", " sort ↑/↓ · "),
                ("class:key", "Esc"),
                ("class:hint", " cancel\n"),
            ]
        )
        return lines

    table_control = FormattedTextControl(_table, focusable=True)
    filter_frame = Frame(Window(filter_control, height=1))

    def _filter_frame_style() -> str:
        app = get_app_or_none()
        if app is not None and app.layout.has_focus(filter_control):
            return "class:frame class:filter-focused"
        return "class:frame"

    filter_frame.container.style = _filter_frame_style
    layout = Layout(
        HSplit(
            [
                Window(FormattedTextControl(_header), height=3, always_hide_cursor=True),
                VSplit(
                    [
                        Window(width=2, char=" "),
                        filter_frame,
                    ]
                ),
                Window(table_control, always_hide_cursor=True),
                Window(FormattedTextControl(_footer), height=3, always_hide_cursor=True),
            ]
        ),
        focused_element=filter_control,
    )
    kb = KeyBindings()
    in_filter = has_focus(filter_control)
    in_table = has_focus(table_control)

    @kb.add("tab", filter=in_filter)
    @kb.add("enter", filter=in_filter)
    @kb.add("down", filter=in_filter)
    def _focus_table(event) -> None:  # noqa: ANN001
        event.app.layout.focus(table_control)
        event.app.invalidate()

    @kb.add("tab", filter=in_table)
    @kb.add("/", filter=in_table)
    def _focus_filter(event) -> None:  # noqa: ANN001
        state.sorting = False
        event.app.layout.focus(filter_control)
        filter_buffer.cursor_position = len(filter_buffer.text)
        event.app.invalidate()

    @kb.add("up", filter=in_table)
    @kb.add("c-p", filter=in_table)
    def _up(event) -> None:  # noqa: ANN001
        if state.sorting:
            state.sort_index = (state.sort_index - 1) % len(_SORT_FIELDS)
            event.app.invalidate()
            return
        if state.index == 0:
            event.app.layout.focus(filter_control)
            filter_buffer.cursor_position = len(filter_buffer.text)
            event.app.invalidate()
            return
        state.move(-1)
        event.app.invalidate()

    @kb.add("down", filter=in_table)
    @kb.add("c-n", filter=in_table)
    def _down(event) -> None:  # noqa: ANN001
        if state.sorting:
            state.sort_index = (state.sort_index + 1) % len(_SORT_FIELDS)
            event.app.invalidate()
            return
        state.move(1)
        event.app.invalidate()

    @kb.add("pageup", filter=in_table)
    def _page_up(event) -> None:  # noqa: ANN001
        if state.sorting:
            state.sort_index = max(0, state.sort_index - state.capacity())
            event.app.invalidate()
            return
        state.move(-state.capacity())
        event.app.invalidate()

    @kb.add("pagedown", filter=in_table)
    def _page_down(event) -> None:  # noqa: ANN001
        if state.sorting:
            state.sort_index = min(
                len(_SORT_FIELDS) - 1, state.sort_index + state.capacity()
            )
            event.app.invalidate()
            return
        state.move(state.capacity())
        event.app.invalidate()

    @kb.add("home", filter=in_table)
    def _home(event) -> None:  # noqa: ANN001
        if state.sorting:
            state.sort_index = 0
            event.app.invalidate()
            return
        state.index = 0
        state.sync_scroll()
        event.app.invalidate()

    @kb.add("end", filter=in_table)
    def _end(event) -> None:  # noqa: ANN001
        if state.sorting:
            state.sort_index = len(_SORT_FIELDS) - 1
            event.app.invalidate()
            return
        state.index = max(0, len(state.rows) - 1)
        state.sync_scroll()
        event.app.invalidate()

    @kb.add("left", filter=in_table)
    def _left(event) -> None:  # noqa: ANN001
        if state.sorting:
            return
        state.horizontal = max(0, state.horizontal - 8)
        event.app.invalidate()

    @kb.add("right", filter=in_table)
    def _right(event) -> None:  # noqa: ANN001
        if state.sorting:
            return
        state.horizontal += 8
        event.app.invalidate()

    @kb.add("space", filter=in_table)
    def _toggle(event) -> None:  # noqa: ANN001
        if state.sorting:
            return
        state.toggle()
        event.app.invalidate()

    @kb.add("a", filter=in_table)
    def _toggle_visible(event) -> None:  # noqa: ANN001
        if state.sorting:
            return
        state.toggle_visible()
        event.app.invalidate()

    @kb.add("enter", filter=in_table)
    def _submit(event) -> None:  # noqa: ANN001
        if state.sorting:
            state.apply_sort(_SORT_FIELDS[state.sort_index])
            state.sorting = False
            event.app.invalidate()
            return
        chosen = state.chosen()
        if not chosen:
            return
        result["done"] = True
        result["rows"] = chosen
        event.app.exit()

    @kb.add("s", filter=in_table)
    def _sort(event) -> None:  # noqa: ANN001
        state.sort_index = _SORT_FIELDS.index(state.sort_field)
        state.sorting = True
        event.app.invalidate()

    @kb.add("r", filter=in_table)
    def _reverse_sort(event) -> None:  # noqa: ANN001
        if state.sorting:
            return
        state.reverse_sort()
        event.app.invalidate()

    @kb.add("q", filter=in_table)
    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:  # noqa: ANN001
        if state.sorting:
            state.sorting = False
            event.app.invalidate()
            return
        result["cancel"] = True
        event.app.exit()

    app: Application[Any] = Application(
        layout=layout,
        key_bindings=kb,
        style=MENU_STYLE,
        full_screen=False,
    )
    run_app(app)
    if result["cancel"]:
        return None
    if not result["done"]:
        raise RuntimeError("Remote rollout picker closed without a selection or cancellation")
    return result["rows"]
